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
from realtime_framework.providers.openai_realtime_session import Delete
from realtime_framework.providers.openai_realtime_session import Operation
from realtime_framework.providers.openai_realtime_session import RealtimeSessionCore
from realtime_framework.providers.openai_realtime_session import history_frames

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
        # The open connection's conversation: what it holds and what became of each operation.
        self._core: RealtimeSessionCore | None = None

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

    def _send(self, operation: Operation) -> None:
        """Send one planned conversation item create or delete."""
        if self._conn is None:
            raise RuntimeError("OpenAI Realtime session is not connected")
        if isinstance(operation, Delete):
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

    async def _send_all(self, operations) -> None:
        """Send planned operations in order; the session core records what became of each."""
        core = self._core
        for operation in operations:
            core.sending(operation)
            try:
                await asyncio.to_thread(self._send, operation)
            except Exception:
                core.send_failed(operation)
                raise

    async def _wait_for_item_edits(self) -> None:
        """Receive until the server has answered the pending in-place edit; the session core
        decides which events answer it."""
        if self._conn is None:
            raise RuntimeError("OpenAI Realtime session is not connected")
        while not self._core.edit_answered:
            self._core.receive(await asyncio.to_thread(self._wire, self._conn.recv))

    async def _close_current(self) -> None:
        cm, self._cm = self._cm, None
        self._conn = None
        self._core = None
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
        self._core = RealtimeSessionCore(self._new_client_id)
        # The history is None when the runtime opens the episode's first session.
        await self._send_all(self._core.replay(request.history))

    async def send_observation(self, observation: ProviderObservation, history) -> dict:
        # The history ends with this observation's frame.
        await self._send_all([self._core.observe(history)])
        return {}

    async def start_generation(self, request: GenerationRequest) -> GenerationHandle:
        if self._conn is None:
            raise RuntimeError("OpenAI Realtime session is not connected")
        # The frame this response answers is the request's newest; its output items join it.
        _, frames = history_frames(request.canonical.items)
        answered = self._core.frames[frames[-1].id]
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
        problem = self._core.in_place_problem(request.history)
        if problem is not None:
            return await self._fallback_from_in_place(request, problem)

        try:
            batch = self._core.begin_edit(request.history)
            await self._send_all(batch.operations)
            await self._wait_for_item_edits()
            self._core.finish_edit()
        except Exception as exc:  # noqa: BLE001 - a partial edit requires canonical replay
            return await self._fallback_from_in_place(request, repr(exc))

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
            await self._send_all(self._core.repair(request.history, output_ids))
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
