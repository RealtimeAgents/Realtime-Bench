from __future__ import annotations

# Openai realtime
"""OpenAI Realtime WebSocket wire driver."""

import asyncio as asyncio
import base64 as base64
import logging as logging
import os as os
import threading as threading
import uuid as uuid
from copy import copy as copy
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from dataclasses import replace as replace
from typing import ClassVar as ClassVar

from openai import APIConnectionError as APIConnectionError
from openai import APITimeoutError as APITimeoutError
from openai import OpenAI as OpenAI
from websockets.exceptions import ConnectionClosed as ConnectionClosed

from realtime_framework.providers.registry import load_project_secrets
from realtime_framework.entity.model import ConnectRequest
from realtime_framework.entity.model import Deployment
from realtime_framework.entity.events import GenerationBoundary
from realtime_framework.entity.model import GenerationHandle
from realtime_framework.entity.model import GenerationRequest
from realtime_framework.entity.model import HistorySyncRequest
from realtime_framework.entity.model import HistorySyncResult
from realtime_framework.entity.model import InterruptResult
from realtime_framework.entity.events import ProviderError
from realtime_framework.entity.events import ProviderHistoryState
from realtime_framework.entity.model import ProviderObservation
from realtime_framework.entity.model import RealtimeCapabilities
from realtime_framework.entity.model import SyncDisposition
from realtime_framework.entity.events import TextDelta
from realtime_framework.entity.events import UsageSnapshot
from realtime_framework.entity.context_policy import FixedTexts
from realtime_framework.entity.request import FrameImage
from realtime_framework.entity.request import RequestItem
from realtime_framework.providers.litellm import iterate_blocking
from realtime_framework.providers.litellm import object_dict
from realtime_framework.entity.model import FATAL
from realtime_framework.entity.model import QUOTA_EXHAUSTED
from realtime_framework.entity.model import RATE_LIMITED
from realtime_framework.entity.model import TRANSIENT
from realtime_framework.entity.model import TransportFailure
from realtime_framework.entity.model import classify_http_failure
from realtime_framework.entity.model import exception_chain
from realtime_framework.entity.model import retry_after_seconds
from realtime_framework.providers.capture import sync_ws_connection

logger = logging.getLogger(__name__)
_CLIENT_ID_MAX_LENGTH = 32
_ITEM_ACKS = {"conversation.item.added", "conversation.item.created", "conversation.item.done"}


def classify_realtime_message(direction: str, message: dict):
    """Name a Realtime message and its acknowledgement keys, for the agent trace.

    A client message lists the keys a later server message may answer; a server message lists
    the keys it answers and its origin when it answers none.
    """
    kind = str(message.get("type", ""))
    if direction == "client":
        keys = []
        if kind in {"session.update", "response.create", "response.cancel"}:
            keys.append((kind,))
        elif kind == "conversation.item.create":
            keys.append(("item", (message.get("item") or {}).get("id")))
        elif kind == "conversation.item.delete":
            keys.append(("deleted", message.get("item_id")))
        if message.get("event_id"):
            keys.append(("event", message["event_id"]))  # answered by an error naming it
        return kind, keys, None
    if kind == "session.updated":
        return kind, [("session.update",)], "unsolicited"
    if kind == "response.created":
        return kind, [("response.create",)], "response"
    if kind == "response.done" and (message.get("response") or {}).get("status") == "cancelled":
        return kind, [("response.cancel",)], "response"
    if kind in _ITEM_ACKS:
        # Output items the server creates during a response answer no client message.
        return kind, [("item", (message.get("item") or {}).get("id"))], "response"
    if kind == "conversation.item.deleted":
        return kind, [("deleted", message.get("item_id"))], "unsolicited"
    if kind == "error":
        event_id = (message.get("error") or {}).get("event_id")
        return kind, [("event", event_id)] if event_id else [], "unsolicited"
    # Response events, and what the server sends on its own (session.created, rate_limits.updated).
    return kind, [], "response" if kind.startswith("response.") else "unsolicited"


@dataclass(frozen=True)
class _Create:
    """One planned conversation.item.create of a message item. In-place edits anchor it and name
    its client event, so the server's acknowledgement or error can be matched to it."""

    item_id: str
    role: str
    text: str
    image: FrameImage | None = None
    previous_item_id: str | None = None
    event_id: str | None = None


@dataclass(frozen=True)
class _Delete:
    """One planned conversation.item.delete."""

    item_id: str
    event_id: str | None = None


@dataclass(frozen=True)
class _EditBatch:
    """One planned in-place edit and the session it leaves: the ledger item and the frames."""

    operations: tuple[_Create | _Delete, ...]
    ledger_item_id: str | None
    ledger_text: str
    frames: dict[str, _SentFrame]
    visuals_folded: int
    prompts_edited: int

    @property
    def created(self) -> set[str]:
        return {op.item_id for op in self.operations if isinstance(op, _Create)}

    @property
    def deleted(self) -> set[str]:
        return {op.item_id for op in self.operations if isinstance(op, _Delete)}

    @property
    def client_events(self) -> set[str]:
        return {op.event_id for op in self.operations}


# Neutral request roles as Realtime message roles.
_ROLES = {"instruction": "system", "input": "user", "model": "assistant"}


@dataclass(frozen=True)
class _ViewFrame:
    """One frame of a history view, with the frame prefix before it and the output after it."""

    item: RequestItem
    prefix: str = ""
    output: RequestItem | None = None

    @property
    def id(self) -> str:
        return self.item.block.id

    @property
    def text(self) -> str:
        """The frame's user item text: the frame prefix, then the frame."""
        text = self.item.block.text
        return f"{self.prefix}\n{text}" if self.prefix else text


def _history_frames(items) -> tuple[RequestItem | None, list[_ViewFrame]]:
    """Read request items as this session holds them: the ledger, then the frames in order."""
    ledger, frames, prefix = None, [], ""
    for item in items:
        block = item.block
        if block.kind == "action_ledger":
            ledger = item
        elif item.intent.position == "frame_prefix":
            prefix = block.text
        elif block.kind == "frame":
            frames.append(_ViewFrame(item, prefix))
            prefix = ""
        elif block.kind == "model_output":
            frames[-1] = replace(frames[-1], output=item)
    return ledger, frames


@dataclass
class _SentFrame:
    """A frame as this driver last wrote it into the conversation."""

    user_item_id: str
    text: str
    folded: bool
    assistant_item_ids: list[str] = field(default_factory=list)


class OpenAIRealtimeDriver:
    capabilities = RealtimeCapabilities(
        protocol_family="openai_realtime",
        persistent_session=True,
        can_compact_history_in_place=True,
        terminal_output_usage=True,
        classifies_transport_errors=True,
        maps_output_cap=True,
    )
    deployment = Deployment(
        family="streaming",
        provider="openai",
        key_env="OPENAI_API_KEY",
        vendor="openai",
        litellm_provider="openai",
    )
    model_defaults: ClassVar[dict[str, dict]] = {}

    def __init__(self, config):
        self.config = config
        load_project_secrets()
        key = config.api_key or os.environ.get(config.api_key_env or "OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAI Realtime needs OPENAI_API_KEY or ModelConfig.api_key")
        kwargs = {"api_key": key}
        if config.api_base_url:
            kwargs["base_url"] = config.api_base_url
        self.client = OpenAI(**kwargs)
        self._ws_connection = sync_ws_connection(
            self, "openai_realtime.ws", classify_realtime_message
        )
        self._cm = None
        self._conn = None
        self._send_lock = threading.Lock()
        # The conversation as this driver last wrote it, which each history view is diffed
        # against: every frame block id with its items, and the archived-action ledger item.
        self._frames: dict[str, _SentFrame] = {}
        self._ledger_item_id: str | None = None
        self._ledger_text = ""

    def bind_trace(self, trace) -> None:
        self._trace = trace

    @staticmethod
    def classify_transport_error(exc: BaseException) -> TransportFailure:
        """Classify a failed turn; the harness retries it on a rebuilt session.

        Realtime WebSocket writes are not automatically replayed by the SDK. The harness retries
        only when it has received no output or usage for the failed turn, so rebuilding from
        canonical history cannot duplicate a completed action.
        """
        transient = (
            ConnectionClosed,
            APIConnectionError,
            APITimeoutError,
            ConnectionError,
            TimeoutError,
        )
        for current in exception_chain(exc):
            if isinstance(current, transient):
                return TransportFailure(TRANSIENT)
            status = getattr(current, "status_code", None)
            if isinstance(status, int):
                headers = getattr(getattr(current, "response", None), "headers", None)
                return classify_http_failure(status, str(current), retry_after_seconds(headers))
        # Server `error` events and failed responses arrive as their error-code text.
        detail = str(exc).lower()
        if "insufficient_quota" in detail:
            return TransportFailure(QUOTA_EXHAUSTED)
        if "rate_limit" in detail:
            return TransportFailure(RATE_LIMITED)
        if "server_error" in detail:
            return TransportFailure(TRANSIENT)
        return TransportFailure(FATAL)

    @staticmethod
    def transport_output_token_cap(provider_cap: int | None) -> int | str:
        """Use Realtime's unbounded wire value when no physical provider cap is requested."""
        if provider_cap is None:
            return "inf"
        provider_cap = max(1, int(provider_cap))
        return "inf" if provider_cap > 4096 else provider_cap

    def _wire(self, operation, *args, **kwargs):
        with self._send_lock:
            return operation(*args, **kwargs)

    @staticmethod
    def _new_client_id(kind: str) -> str:
        """Return an OpenAI Realtime client ID within the API's 32-character limit."""
        prefix = f"rtb_{kind}_"
        suffix_length = _CLIENT_ID_MAX_LENGTH - len(prefix)
        if suffix_length < 1:
            raise ValueError(f"Realtime client ID kind is too long: {kind!r}")
        return prefix + uuid.uuid4().hex[:suffix_length]

    def _send(self, operation: _Create | _Delete) -> None:
        """Send one planned conversation item create or delete."""
        if self._conn is None:
            raise RuntimeError("OpenAI Realtime session is not connected")
        if isinstance(operation, _Delete):
            kwargs = {"item_id": operation.item_id}
            if operation.event_id is not None:
                kwargs["event_id"] = operation.event_id
            self._wire(self._conn.conversation.item.delete, **kwargs)
            return
        content = [
            {
                "type": "output_text" if operation.role == "assistant" else "input_text",
                "text": operation.text,
            }
        ]
        if operation.image is not None:
            encoded = base64.b64encode(operation.image.data).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": (f"data:{operation.image.mime_type};base64,{encoded}"),
                }
            )
        kwargs = {
            "item": {
                "id": operation.item_id,
                "type": "message",
                "role": operation.role,
                "content": content,
            }
        }
        if operation.previous_item_id is not None:
            kwargs["previous_item_id"] = operation.previous_item_id
        if operation.event_id is not None:
            kwargs["event_id"] = operation.event_id
        self._wire(self._conn.conversation.item.create, **kwargs)

    def _frame_create(
        self,
        frame: _ViewFrame,
        *,
        previous_item_id=None,
        event_id: str | None = None,
    ) -> _Create:
        """The user item of one frame: its text, and its screenshot unless folded."""
        return _Create(
            self._new_client_id("user"),
            _ROLES[frame.item.intent.role],
            frame.text,
            frame.item.block.image,
            previous_item_id,
            event_id,
        )

    def _output_create(self, frame: _ViewFrame) -> _Create | None:
        """The assistant item of the model output that answered a frame, if it has one."""
        if frame.output is None:
            return None
        return _Create(
            self._new_client_id("assistant"),
            _ROLES[frame.output.intent.role],
            frame.output.block.text,
        )

    def _ledger_create(
        self,
        ledger: RequestItem,
        texts: FixedTexts,
        *,
        previous_item_id=None,
        event_id: str | None = None,
    ) -> _Create:
        """The archived-action ledger item, in the role the policy gives the ledger."""
        return _Create(
            self._new_client_id("ledger"),
            _ROLES[ledger.intent.role],
            texts.ledger_header + "\n" + ledger.block.text,
            None,
            previous_item_id,
            event_id,
        )

    def _wait_for_item_edits(
        self,
        *,
        created_item_ids: set[str],
        deleted_item_ids: set[str],
        client_event_ids: set[str],
    ) -> None:
        """Drain server acknowledgements for one ordered in-place edit batch."""
        if self._conn is None:
            raise RuntimeError("OpenAI Realtime session is not connected")
        pending_created = set(created_item_ids)
        pending_deleted = set(deleted_item_ids)
        while pending_created or pending_deleted:
            event = self._wire(self._conn.recv)
            event_type = getattr(event, "type", "")
            if event_type == "error":
                error = getattr(event, "error", None)
                source_event_id = getattr(error, "event_id", None)
                if source_event_id is None or source_event_id in client_event_ids:
                    raise RuntimeError(f"Realtime history edit failed: {error!r}")
                continue
            if event_type in {"conversation.item.created", "conversation.item.done"}:
                pending_created.discard(str(getattr(getattr(event, "item", None), "id", "")))
            elif event_type == "conversation.item.deleted":
                pending_deleted.discard(str(getattr(event, "item_id", "")))

    async def _close_current(self) -> None:
        cm, self._cm = self._cm, None
        self._conn = None
        self._frames = {}
        self._ledger_item_id = None
        self._ledger_text = ""
        if cm is not None:
            try:
                await asyncio.to_thread(cm.__exit__, None, None, None)
            except Exception:  # noqa: BLE001, S110 - best-effort SDK cleanup
                pass

    async def connect(self, request: ConnectRequest) -> None:
        await self._close_current()

        def open_session():
            cm = self.client.realtime.connect(
                model=request.seed.model_name,
                websocket_connection_options={"create_connection": self._ws_connection},
            )
            conn = cm.__enter__()
            self._cm, self._conn = cm, conn
            conn.session.update(
                session={
                    "type": "realtime",
                    "output_modalities": ["text"],
                    "instructions": request.seed.system_instruction,
                    "truncation": "disabled",
                }
            )

        last_error = None
        for attempt in range(5):
            try:
                await asyncio.to_thread(open_session)
                break
            except Exception as exc:
                last_error = exc
                await self._close_current()
                if attempt == 4:
                    raise RuntimeError(
                        f"OpenAI Realtime session open failed after retries: {exc!r}"
                    ) from exc
                logger.warning(
                    "OpenAI Realtime connect attempt %d failed: %s",
                    attempt + 1,
                    repr(exc)[:200],
                )
                await asyncio.sleep(2 * (attempt + 1))
        if self._conn is None:
            raise RuntimeError(f"OpenAI Realtime session open failed: {last_error!r}")
        operations, ledger, ledger_text, sent = self._replay(request)
        for operation in operations:
            await asyncio.to_thread(self._send, operation)
        if ledger is not None:
            self._ledger_item_id = ledger.item_id
            self._ledger_text = ledger_text
        self._frames = sent

    def _replay(self, request: ConnectRequest):
        """Plan a new session's items from the history view: the ledger, then each frame's user
        item and its output's assistant item. Also returns the session they make."""
        history = request.history  # None when the runtime opens the episode's first session
        ledger, frames = (None, []) if history is None else _history_frames(history.items)
        operations, sent = [], {}
        ledger_create = None
        if ledger is not None:
            ledger_create = self._ledger_create(ledger, history.policy.texts)
            operations.append(ledger_create)
        for frame in frames:
            user = self._frame_create(frame)
            assistant = self._output_create(frame)
            operations += [user] if assistant is None else [user, assistant]
            sent[frame.id] = _SentFrame(
                user.item_id,
                frame.text,
                frame.item.block.folded,
                [] if assistant is None else [assistant.item_id],
            )
        return operations, ledger_create, "" if ledger is None else ledger.block.text, sent

    async def send_observation(self, observation: ProviderObservation, history) -> dict:
        # The history ends with this observation's frame.
        frame = _history_frames(history.items)[1][-1]
        user = self._frame_create(frame)
        await asyncio.to_thread(self._send, user)
        self._frames[frame.id] = _SentFrame(user.item_id, frame.text, frame.item.block.folded)
        return {"user_item_id": user.item_id, "provider_observation_prompt": frame.text}

    async def start_generation(self, request: GenerationRequest) -> GenerationHandle:
        if self._conn is None:
            raise RuntimeError("OpenAI Realtime session is not connected")
        # The frame this response answers is the request's newest; its output items join it.
        _, frames = _history_frames(request.canonical.items)
        answered = self._frames[frames[-1].id]
        response = {"max_output_tokens": request.transport_output_token_cap}
        if self.config.reasoning_effort is not None:
            response["reasoning"] = {"effort": self.config.reasoning_effort}
        await asyncio.to_thread(
            self._wire,
            self._conn.response.create,
            response=response,
        )
        return GenerationHandle(
            request.generation_id,
            opaque=iter(self._conn),
            metadata={
                "output_item_ids": [],
                "response_id": None,
                "answered_frame": answered,
            },
        )

    async def events(self, generation: GenerationHandle):
        sequence = 0
        started = False
        async for event in iterate_blocking(generation.opaque):
            event_type = getattr(event, "type", "")
            if event_type == "response.created":
                started = True
                response = getattr(event, "response", None)
                generation.metadata["response_id"] = getattr(response, "id", None)
                sequence += 1
                yield GenerationBoundary(
                    generation.generation_id,
                    sequence,
                    event_type,
                    "started",
                    ProviderHistoryState.UNKNOWN,
                )
                continue
            if event_type == "error":
                error = getattr(event, "error", None)
                cancel_event_id = generation.metadata.get("cancel_event_id")
                cancel_race = bool(
                    cancel_event_id and getattr(error, "event_id", None) == cancel_event_id
                )
                sequence += 1
                yield ProviderError(
                    generation.generation_id,
                    sequence,
                    event_type,
                    repr(error or "realtime error")[:500],
                    not cancel_race,
                )
                continue
            if not started:
                continue
            if event_type in {"response.output_item.added", "response.output_item.done"}:
                item_id = getattr(getattr(event, "item", None), "id", None)
                if item_id and item_id not in generation.metadata["output_item_ids"]:
                    generation.metadata["output_item_ids"].append(item_id)
                continue
            if event_type.endswith(("output_text.delta", "text.delta")):
                sequence += 1
                yield TextDelta(
                    generation.generation_id,
                    sequence,
                    event_type,
                    getattr(event, "delta", "") or "",
                    "visible",
                )
                continue
            if event_type == "response.done":
                response = getattr(event, "response", None)
                for item in getattr(response, "output", None) or []:
                    item_id = getattr(item, "id", None)
                    if item_id and item_id not in generation.metadata["output_item_ids"]:
                        generation.metadata["output_item_ids"].append(item_id)
                usage = object_dict(getattr(response, "usage", None))
                if usage is not None:
                    sequence += 1
                    yield UsageSnapshot(
                        generation.generation_id,
                        sequence,
                        event_type,
                        usage,
                    )
                status = getattr(response, "status", None)
                details = getattr(response, "status_details", None)
                reason = getattr(details, "reason", None)
                generation.metadata.update({"status": status, "status_reason": reason})
                if status == "failed":
                    # A failed response is a provider fault, exactly like an `error` event; it
                    # must invalidate the turn rather than parse as an empty WAIT decision.
                    error = getattr(details, "error", None)
                    sequence += 1
                    yield ProviderError(
                        generation.generation_id,
                        sequence,
                        event_type,
                        repr(
                            {
                                "status": status,
                                "reason": reason,
                                "error": object_dict(error) or error,
                            }
                        )[:500],
                    )
                    return
                # A superseded response's session is closed and replayed before its next use.
                generation.metadata["answered_frame"].assistant_item_ids = list(
                    generation.metadata["output_item_ids"]
                )
                boundary_metadata = {
                    "provider_response_status": status,
                    "provider_response_status_reason": reason,
                }
                state = (
                    ProviderHistoryState.EXACT
                    if status == "completed"
                    else ProviderHistoryState.CONTAMINATED
                )
                sequence += 1
                yield GenerationBoundary(
                    generation.generation_id,
                    sequence,
                    event_type,
                    "turn_complete" if status == "completed" else "interrupted",
                    state,
                    boundary_metadata,
                )
                return

    async def interrupt(self, generation: GenerationHandle) -> InterruptResult:
        if self._conn is None:
            return InterruptResult(ProviderHistoryState.UNKNOWN)
        kwargs = {}
        response_id = generation.metadata.get("response_id")
        if response_id:
            kwargs["response_id"] = response_id
        cancel_event_id = self._new_client_id("cancel")
        kwargs["event_id"] = cancel_event_id
        generation.metadata["cancel_event_id"] = cancel_event_id
        try:
            await asyncio.to_thread(self._wire, self._conn.response.cancel, **kwargs)
            return InterruptResult(ProviderHistoryState.CONTAMINATED)
        except Exception as exc:  # noqa: BLE001 - provider control boundary
            return InterruptResult(
                ProviderHistoryState.UNKNOWN,
                {"cancel_error": repr(exc)[:300]},
            )

    async def _rebuild(self, request: HistorySyncRequest) -> HistorySyncResult:
        try:
            await self.connect(
                ConnectRequest(seed=request.seed, history=request.history)
            )
            return HistorySyncResult(
                SyncDisposition.REBUILT,
                ProviderHistoryState.EXACT,
                detail=request.reason,
                provider_metadata={"history_session_reconnected": True},
            )
        except Exception as exc:  # noqa: BLE001 - provider session boundary
            return HistorySyncResult(
                SyncDisposition.BROKEN,
                ProviderHistoryState.UNKNOWN,
                detail=repr(exc)[:500],
            )

    async def _fallback_from_in_place(
        self,
        request: HistorySyncRequest,
        cause: str,
    ) -> HistorySyncResult:
        rebuilt = await self._rebuild(request)
        return HistorySyncResult(
            rebuilt.disposition,
            rebuilt.history_state,
            rebuilt.detail,
            {
                **rebuilt.provider_metadata,
                "history_in_place_edit_fallback": True,
                "history_in_place_edit_error": cause[:500],
            },
        )

    async def _compact_in_place(self, request: HistorySyncRequest) -> HistorySyncResult:
        """Apply SoM removal and block folding without replacing the live WebSocket session."""
        if self._conn is None:
            return await self._rebuild(request)

        # Validate every reference before the first mutating wire event. Any missing/stale provider
        # identity falls back to a canonical replay instead of risking a partial local plan.
        ledger, frames = _history_frames(request.history.items)
        for frame in frames:
            sent = self._frames.get(frame.id)
            if sent is None:
                return await self._fallback_from_in_place(
                    request,
                    "retained turn is missing its provider user item ID",
                )
            if sent.folded and not frame.item.block.folded:
                return await self._fallback_from_in_place(
                    request,
                    "provider visual state cannot be unfolded in place",
                )

        try:
            batch = self._edit_batch(ledger, frames, request.history.policy.texts)
            for operation in batch.operations:
                self._send(operation)
            await asyncio.to_thread(
                self._wait_for_item_edits,
                created_item_ids=batch.created,
                deleted_item_ids=batch.deleted,
                client_event_ids=batch.client_events,
            )
        except Exception as exc:  # noqa: BLE001 - a partial edit requires canonical replay
            return await self._fallback_from_in_place(request, repr(exc))

        self._ledger_item_id = batch.ledger_item_id
        self._ledger_text = batch.ledger_text
        self._frames = batch.frames
        return HistorySyncResult(
            SyncDisposition.EXACT,
            ProviderHistoryState.EXACT,
            detail="observation history updated in the existing Realtime session",
            provider_metadata={
                "history_session_reconnected": False,
                "history_items_created": len(batch.created),
                "history_items_deleted": len(batch.deleted),
                "history_visuals_folded_in_place": batch.visuals_folded,
                "history_prompts_edited_in_place": batch.prompts_edited,
            },
        )

    def _edit_batch(
        self, ledger: RequestItem | None, frames: list[_ViewFrame], texts: FixedTexts
    ) -> _EditBatch:
        """Plan one in-place edit from the history view, in send order: delete the items of sent
        frames the view no longer holds, replace the ledger if its text changed, then replace each
        user item whose text changed or whose screenshot folded, after the item before it. Every
        operation names its client event."""
        operations: list[_Create | _Delete] = []
        retained = {frame.id for frame in frames}
        for frame_id, sent in self._frames.items():
            if frame_id not in retained:
                for item_id in (sent.user_item_id, *sent.assistant_item_ids):
                    operations.append(_Delete(item_id, self._new_client_id("delete")))

        ledger_text = "" if ledger is None else ledger.block.text
        ledger_item_id = self._ledger_item_id
        if ledger_text != self._ledger_text:
            if self._ledger_item_id:
                operations.append(_Delete(self._ledger_item_id, self._new_client_id("delete")))
            ledger_item_id = None
            if ledger is not None:
                create = self._ledger_create(
                    ledger, texts, previous_item_id="root", event_id=self._new_client_id("create")
                )
                operations.append(create)
                ledger_item_id = create.item_id

        previous_item_id = ledger_item_id or "root"
        sent_frames: dict[str, _SentFrame] = {}
        folded_count = 0
        prompt_edit_count = 0
        for frame in frames:
            sent = self._frames[frame.id]
            folded = frame.item.block.folded
            user_item_id = sent.user_item_id
            prompt_changed = frame.text != sent.text
            if prompt_changed or (folded and not sent.folded):
                operations.append(_Delete(user_item_id, self._new_client_id("delete")))
                replacement = self._frame_create(
                    frame,
                    previous_item_id=previous_item_id,
                    event_id=self._new_client_id("create"),
                )
                operations.append(replacement)
                folded_count += int(folded and not sent.folded)
                prompt_edit_count += int(prompt_changed)
                user_item_id = replacement.item_id
            assistant_item_ids = list(sent.assistant_item_ids)
            sent_frames[frame.id] = _SentFrame(
                user_item_id, frame.text, folded, assistant_item_ids
            )
            previous_item_id = assistant_item_ids[-1] if assistant_item_ids else user_item_id
        return _EditBatch(
            tuple(operations),
            ledger_item_id,
            ledger_text,
            sent_frames,
            folded_count,
            prompt_edit_count,
        )

    async def sync_history(self, request: HistorySyncRequest) -> HistorySyncResult:
        generation = request.current_generation
        if (
            not request.force_rebuild
            and request.reason == "history_block_compaction"
            and self._conn is not None
        ):
            return await self._compact_in_place(request)
        if request.force_rebuild or generation is None or self._conn is None:
            return await self._rebuild(request)
        output_ids = list(generation.metadata.get("output_item_ids") or [])
        try:
            if not output_ids:
                raise RuntimeError("response did not expose output item IDs")
            # The response answered the newest frame; its output replaces what the server kept.
            _, frames = _history_frames(request.history.items)
            replacement = self._output_create(frames[-1])
            for item_id in output_ids:
                await asyncio.to_thread(self._send, _Delete(item_id))
            if replacement is not None:
                await asyncio.to_thread(self._send, replacement)
            self._frames[frames[-1].id].assistant_item_ids = (
                [] if replacement is None else [replacement.item_id]
            )
            return HistorySyncResult(
                SyncDisposition.EXACT,
                ProviderHistoryState.EXACT,
                detail="cancelled response replaced with the complete collected output",
                provider_metadata={"deleted_output_items": len(output_ids)},
            )
        except Exception:  # noqa: BLE001 - fall back to canonical replay
            return await self._rebuild(request)

    async def close(self) -> None:
        await self._close_current()
