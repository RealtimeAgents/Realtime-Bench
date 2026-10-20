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

from openai import APIConnectionError as APIConnectionError
from openai import APITimeoutError as APITimeoutError
from openai import OpenAI as OpenAI
from websockets.exceptions import ConnectionClosed as ConnectionClosed

from realtime_framework.providers.registry import load_project_secrets
from realtime_framework.entity.model import ConnectRequest
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
from realtime_framework.providers.legacy import session_frame_text
from realtime_framework.providers.litellm import iterate_blocking
from realtime_framework.providers.litellm import object_dict
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER
from realtime_framework.entity.model import FATAL
from realtime_framework.entity.model import QUOTA_EXHAUSTED
from realtime_framework.entity.model import RATE_LIMITED
from realtime_framework.entity.model import TRANSIENT
from realtime_framework.entity.model import TransportFailure
from realtime_framework.entity.model import classify_http_failure
from realtime_framework.entity.model import exception_chain
from realtime_framework.entity.model import retry_after_seconds
from realtime_framework.entity.model import FOLDED_VISUAL_PLACEHOLDER
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


class OpenAIRealtimeDriver:
    capabilities = RealtimeCapabilities(
        protocol_family="openai_realtime",
        persistent_session=True,
        can_compact_history_in_place=True,
        terminal_output_usage=True,
    )

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

    def _create_item(
        self,
        item: dict,
        *,
        previous_item_id=None,
        event_id: str | None = None,
    ) -> str | None:
        if self._conn is None:
            raise RuntimeError("OpenAI Realtime session is not connected")
        kwargs = {"item": item}
        if previous_item_id is not None:
            kwargs["previous_item_id"] = previous_item_id
        if event_id is not None:
            kwargs["event_id"] = event_id
        self._wire(self._conn.conversation.item.create, **kwargs)
        return item.get("id")

    def _user_item(
        self,
        observation: ProviderObservation,
        *,
        folded: bool = False,
        previous_item_id=None,
        event_id: str | None = None,
    ) -> str:
        item_id = self._new_client_id("user")
        content = [
            {
                "type": "input_text",
                "text": FOLDED_VISUAL_PLACEHOLDER if folded else session_frame_text(observation),
            }
        ]
        if not folded:
            encoded = base64.b64encode(observation.image_bytes).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": (f"data:{observation.mime_type};base64,{encoded}"),
                }
            )
        self._create_item(
            {
                "id": item_id,
                "type": "message",
                "role": "user",
                "content": content,
            },
            previous_item_id=previous_item_id,
            event_id=event_id,
        )
        return item_id

    def _assistant_item(self, text: str) -> str | None:
        if not text:
            return None
        item_id = self._new_client_id("assistant")
        self._create_item(
            {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )
        return item_id

    def _system_item(
        self,
        text: str,
        *,
        previous_item_id=None,
        event_id: str | None = None,
    ) -> str | None:
        if not text:
            return None
        item_id = self._new_client_id("ledger")
        self._create_item(
            {
                "id": item_id,
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": text}],
            },
            previous_item_id=previous_item_id,
            event_id=event_id,
        )
        return item_id

    def _delete_item(self, item_id: str, *, event_id: str | None = None) -> None:
        if self._conn is None:
            raise RuntimeError("OpenAI Realtime session is not connected")
        kwargs = {"item_id": item_id}
        if event_id is not None:
            kwargs["event_id"] = event_id
        self._wire(self._conn.conversation.item.delete, **kwargs)

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
        if request.executed_action_ledger:
            self._ledger_item_id = await asyncio.to_thread(
                self._system_item,
                ARCHIVED_ACTION_LEDGER_HEADER + "\n" + request.executed_action_ledger,
            )
            self._ledger_text = request.executed_action_ledger
        for turn in request.turns:
            user_id = await asyncio.to_thread(
                self._user_item,
                turn.observation,
                folded=turn.visual_folded,
            )
            assistant_id = await asyncio.to_thread(self._assistant_item, turn.output)
            turn.provider_refs["user_item_id"] = user_id
            turn.provider_refs["assistant_item_ids"] = [assistant_id] if assistant_id else []
            turn.provider_refs["provider_visual_folded"] = turn.visual_folded
            turn.provider_refs["provider_observation_prompt"] = session_frame_text(turn.observation)

    async def send_observation(self, observation: ProviderObservation, history) -> dict:
        item_id = await asyncio.to_thread(self._user_item, observation)
        return {
            "user_item_id": item_id,
            "provider_observation_prompt": session_frame_text(observation),
        }

    async def start_generation(self, request: GenerationRequest) -> GenerationHandle:
        if self._conn is None:
            raise RuntimeError("OpenAI Realtime session is not connected")
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
                generation.metadata["completed_turn_refs"] = {
                    "assistant_item_ids": list(generation.metadata["output_item_ids"]),
                    "provider_visual_folded": False,
                }
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
                ConnectRequest(
                    seed=request.seed,
                    turns=request.turns,
                    executed_action_ledger=request.executed_action_ledger,
                )
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

    @staticmethod
    def _turn_item_ids(turn) -> tuple[str, ...]:
        values = []
        user_id = turn.provider_refs.get("user_item_id")
        if user_id:
            values.append(str(user_id))
        values.extend(
            str(value) for value in turn.provider_refs.get("assistant_item_ids", ()) if value
        )
        return tuple(values)

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
        for turn in request.removed_turns:
            if not turn.provider_refs.get("user_item_id"):
                return await self._fallback_from_in_place(
                    request,
                    "archived turn is missing its provider user item ID",
                )
        for turn in request.turns:
            if not turn.provider_refs.get("user_item_id"):
                return await self._fallback_from_in_place(
                    request,
                    "retained turn is missing its provider user item ID",
                )
            provider_folded = bool(turn.provider_refs.get("provider_visual_folded", False))
            if provider_folded and not turn.visual_folded:
                return await self._fallback_from_in_place(
                    request,
                    "provider visual state cannot be unfolded in place",
                )

        created: set[str] = set()
        deleted: set[str] = set()
        client_events: set[str] = set()
        observation_updates: list[tuple[object, str]] = []
        folded_count = 0
        prompt_edit_count = 0
        new_ledger_id = self._ledger_item_id
        ledger_changed = request.executed_action_ledger != self._ledger_text

        try:
            for turn in request.removed_turns:
                for item_id in self._turn_item_ids(turn):
                    event_id = self._new_client_id("delete")
                    self._delete_item(item_id, event_id=event_id)
                    deleted.add(item_id)
                    client_events.add(event_id)

            if ledger_changed:
                if self._ledger_item_id:
                    event_id = self._new_client_id("delete")
                    self._delete_item(self._ledger_item_id, event_id=event_id)
                    deleted.add(self._ledger_item_id)
                    client_events.add(event_id)
                new_ledger_id = None
                if request.executed_action_ledger:
                    event_id = self._new_client_id("create")
                    new_ledger_id = self._system_item(
                        ARCHIVED_ACTION_LEDGER_HEADER + "\n" + request.executed_action_ledger,
                        previous_item_id="root",
                        event_id=event_id,
                    )
                    if new_ledger_id:
                        created.add(new_ledger_id)
                    client_events.add(event_id)

            previous_item_id = new_ledger_id or "root"
            for turn in request.turns:
                user_id = str(turn.provider_refs["user_item_id"])
                provider_folded = bool(turn.provider_refs.get("provider_visual_folded", False))
                prompt_changed = (
                    turn.provider_refs.get("provider_observation_prompt")
                    != session_frame_text(turn.observation)
                )
                if prompt_changed or (turn.visual_folded and not provider_folded):
                    delete_event_id = self._new_client_id("delete")
                    self._delete_item(user_id, event_id=delete_event_id)
                    deleted.add(user_id)
                    client_events.add(delete_event_id)

                    create_event_id = self._new_client_id("create")
                    replacement_id = self._user_item(
                        turn.observation,
                        folded=turn.visual_folded,
                        previous_item_id=previous_item_id,
                        event_id=create_event_id,
                    )
                    created.add(replacement_id)
                    client_events.add(create_event_id)
                    observation_updates.append((turn, replacement_id))
                    folded_count += int(turn.visual_folded and not provider_folded)
                    prompt_edit_count += int(prompt_changed)
                    user_id = replacement_id

                assistant_ids = [
                    str(value)
                    for value in turn.provider_refs.get("assistant_item_ids", ())
                    if value
                ]
                previous_item_id = assistant_ids[-1] if assistant_ids else user_id

            await asyncio.to_thread(
                self._wait_for_item_edits,
                created_item_ids=created,
                deleted_item_ids=deleted,
                client_event_ids=client_events,
            )
        except Exception as exc:  # noqa: BLE001 - a partial edit requires canonical replay
            return await self._fallback_from_in_place(request, repr(exc))

        self._ledger_item_id = new_ledger_id
        self._ledger_text = request.executed_action_ledger
        for turn, replacement_id in observation_updates:
            turn.provider_refs["user_item_id"] = replacement_id
            turn.provider_refs["provider_visual_folded"] = turn.visual_folded
            turn.provider_refs["provider_observation_prompt"] = session_frame_text(turn.observation)
        return HistorySyncResult(
            SyncDisposition.EXACT,
            ProviderHistoryState.EXACT,
            detail="observation history updated in the existing Realtime session",
            provider_metadata={
                "history_session_reconnected": False,
                "history_items_created": len(created),
                "history_items_deleted": len(deleted),
                "history_visuals_folded_in_place": folded_count,
                "history_prompts_edited_in_place": prompt_edit_count,
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
            for item_id in output_ids:
                await asyncio.to_thread(
                    self._wire,
                    self._conn.conversation.item.delete,
                    item_id=item_id,
                )
            replacement = await asyncio.to_thread(
                self._assistant_item,
                request.turns[-1].output if request.turns else "",
            )
            if request.turns:
                request.turns[-1].provider_refs["assistant_item_ids"] = (
                    [replacement] if replacement else []
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
