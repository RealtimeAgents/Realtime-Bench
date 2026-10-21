from __future__ import annotations

# Openai realtime
"""OpenAI Realtime WebSocket wire driver."""

import asyncio as asyncio
import base64 as base64
import errno
import logging as logging
import os as os
import socket
import ssl
import threading as threading
import uuid as uuid
from copy import copy as copy
from functools import partial
from typing import ClassVar as ClassVar

from openai import APIConnectionError as APIConnectionError
from openai import APITimeoutError as APITimeoutError
from openai import OpenAI as OpenAI
from websockets.exceptions import ConnectionClosed as ConnectionClosed
from websockets.exceptions import ConnectionClosedOK as ConnectionClosedOK

from realtime_framework.clock import SystemClock
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
from realtime_framework.entity.events import TextDone
from realtime_framework.entity.events import UsageSnapshot
from realtime_framework.providers.litellm import object_dict
from realtime_framework.entity.model import FATAL
from realtime_framework.entity.model import QUOTA_EXHAUSTED
from realtime_framework.entity.model import RATE_LIMITED
from realtime_framework.entity.model import TRANSIENT
from realtime_framework.entity.model import RequestRetry
from realtime_framework.entity.model import TransportFailure
from realtime_framework.entity.model import TransportFailureError
from realtime_framework.entity.model import classify_http_failure
from realtime_framework.entity.model import exception_chain
from realtime_framework.entity.model import retry_after_seconds
from realtime_framework.entity.model import retry_cancel_event
from realtime_framework.entity.records import TraceEvent
from realtime_framework.entity.records import trace_scope
from realtime_framework.providers.capture import sync_ws_connection
from realtime_framework.providers.openai_realtime_session import Delete
from realtime_framework.providers.openai_realtime_session import Operation
from realtime_framework.providers.openai_realtime_session import RealtimeSessionCore
from realtime_framework.providers.openai_realtime_session import history_frames
from realtime_framework.providers.session_lifecycle import SessionLifecycle

logger = logging.getLogger(__name__)
_CLIENT_ID_MAX_LENGTH = 32
# What the trace links to an item create as its answers; the first is its acknowledgement. The
# session core confirms a create only on .created or .done (openai_realtime_session._CREATE_ACKS).
_ITEM_ACKS = {"conversation.item.added", "conversation.item.created", "conversation.item.done"}
# The OSErrors of a network or host that is down or unreachable, which no ConnectionError covers.
_NETWORK_ERRNOS = {
    errno.ENETUNREACH,
    errno.EHOSTUNREACH,
    errno.ENETDOWN,
    errno.EHOSTDOWN,
    errno.ENETRESET,
}


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


class _Session:
    """One open connection: the number the trace gives its WebSocket connection, its session core
    and its one reader.

    On a turn-based session the reader reads on demand: only while something waits for an
    acknowledgement or a response event, and at most one read at a time, which every waiter
    awaits instead of receiving on its own. On a native-streaming session it reads every event as
    it arrives, from the connect until the connection ends, and waiters only wait for its reads.
    Each server event it reads goes to the core, which hands it to the state it belongs to.
    """

    def __init__(self, number: int, conn, core: RealtimeSessionCore):
        self.number = number
        self.core = core
        self._conn = conn
        self._read: asyncio.Task | None = None
        # Native streaming: the reader that reads every event, the clock it reads each event's
        # arrival on, and what tells waiters that it read one.
        self._reader: asyncio.Task | None = None
        self._clock = None
        self._read_one: asyncio.Event | None = None
        # How the connection ended: closed normally (by either side), or a failed read.
        self.closed: ConnectionClosedOK | None = None
        self.failure: Exception | None = None

    def read_continuously(self, clock) -> None:
        """Read every event as it arrives until the connection ends, and hand each to the core
        with the time ``clock`` reads then (native streaming)."""
        self._clock, self._read_one = clock, asyncio.Event()
        # The reader runs in the context of the call that starts it: the trace records every
        # message it reads in that call's scope.
        self._reader = asyncio.create_task(self._read_all())

    async def _read_all(self) -> None:
        try:
            while self.closed is None and self.failure is None:
                await self._receive()
                self._read_one.set()
        except asyncio.CancelledError:
            # Only a close that may have left the connection open cancels the reader: the
            # session is over all the same for whoever waits on it.
            self.failure = ConnectionError("OpenAI Realtime session closed")
            self._read_one.set()
            raise
        except Exception as exc:  # noqa: BLE001 - the core failed on an event: every waiter raises it
            self.failure = exc
            self._read_one.set()

    async def stop_reading(self, *, closed: bool) -> None:
        """Once the connection is closed, wait for the reader to end; a connection that may
        still be open cancels it instead."""
        if self._reader is None:
            return
        if not closed:
            self._reader.cancel()
        await asyncio.wait({self._reader})

    async def receive_until(self, ready) -> None:
        """Receive until ``ready()`` holds or the connection closes normally. A failed read
        raises in every waiter, once it has taken what was received before."""
        while not ready():
            if self.failure is not None:
                raise self.failure
            if self.closed:
                return
            if self._reader is not None:
                self._read_one.clear()
                await self._read_one.wait()
                continue
            if self._read is None:
                # The read runs in the context of the waiter that starts it: the trace records the
                # message in that waiter's scope, whichever state the message turns out to be for.
                self._read = asyncio.create_task(self._receive())
            await asyncio.wait({self._read})

    async def _receive(self) -> None:
        try:
            event = await asyncio.to_thread(self._conn.recv)
        except ConnectionClosedOK as closed:
            self.closed = closed
        except Exception as exc:  # noqa: BLE001 - every waiter raises it
            self.failure = exc
        else:
            if self._clock is None:
                taken = self.core.receive(event)
            else:
                taken = self.core.arrived(event, self._clock.monotonic())
            if not taken:
                logger.debug(
                    "OpenAI Realtime connection %d: no state takes %s",
                    self.number,
                    getattr(event, "type", ""),
                )
        finally:
            self._read = None


class OpenAIRealtimeDriver(SessionLifecycle):
    capabilities = RealtimeCapabilities(
        protocol_family="openai_realtime",
        persistent_session=True,
        roles=("instruction", "input", "model"),
        feedback_placements=("separate_message",),
        server_context="unknown",
        can_compact_history_in_place=True,
        terminal_output_usage=True,
        classifies_transport_errors=True,
        maps_output_cap=True,
        interaction_modes=frozenset({"turn_based", "streaming"}),
        # The 1 fps native streaming is designed for, until a smoke test finds the rate the API
        # keeps up with (docs/streaming_smoke_test.md).
        max_streaming_fps=1.0,
    )
    deployment = Deployment(
        family="streaming",
        provider="openai",
        key_env="OPENAI_API_KEY",
        vendor="openai",
        litellm_provider="openai",
    )
    model_defaults: ClassVar[dict[str, dict]] = {}
    # The API's native-streaming settings: a frame older than the newest retained_frames folds in
    # place, as a turn-based one does. A placeholder, the turn-based visual budget (20 s at
    # 1 fps), until a smoke test measures what a frame costs against the model's context
    # (docs/streaming_smoke_test.md). A model YAML's streaming settings replace those they name
    # (ModelConfig.streaming); a driver holds the values it runs with.
    streaming_settings: ClassVar[dict] = {"retained_frames": 20}

    def __init__(self, config, *, clock=None, streaming: bool = False):
        """``streaming``: the session native streaming drives, which reads every event as it
        arrives and is rebuilt from the streaming history view."""
        self.config = config
        self.clock = clock or SystemClock()
        self.streaming = streaming
        self.streaming_settings = {**self.streaming_settings, **config.streaming}
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
        # The open connection's conversation and its receiver.
        self._session: _Session | None = None

    def bind_trace(self, trace) -> None:
        self._trace = trace

    @staticmethod
    def classify_transport_error(exc: BaseException) -> TransportFailure:
        """Classify a failed turn, which the harness retries on a rebuilt session, or a failed
        connect attempt, which connect() retries.

        Realtime WebSocket writes are not automatically replayed by the SDK. The harness retries
        only when it has received no output or usage for the failed turn, so rebuilding from
        canonical history cannot duplicate a completed action.

        A connect attempt fails below HTTP with a ConnectionError (refused or reset), an OSError
        naming a network or host that is down or unreachable, a failed name lookup
        (socket.gaierror), ssl.SSLError or a timeout or, when the connection ends before the
        upgrade response, with websockets' InvalidMessage caused by EOFError; these are
        transient. Another OSError is local (a permission, a missing file, an address the host
        lacks) and fatal.
        """
        transient = (
            ConnectionClosed,
            APIConnectionError,
            APITimeoutError,
            ConnectionError,
            socket.gaierror,
            ssl.SSLError,
            TimeoutError,
            EOFError,
        )
        for current in exception_chain(exc):
            # A reconnect inside the turn already spent the shared retry budget.
            if isinstance(current, TransportFailureError):
                return current.failure
            if isinstance(current, transient) or (
                isinstance(current, OSError) and current.errno in _NETWORK_ERRNOS
            ):
                return TransportFailure(TRANSIENT)
            # A refused WebSocket upgrade (websockets InvalidStatus) carries its HTTP response.
            status = getattr(current, "status_code", None) or getattr(
                getattr(current, "response", None), "status_code", None
            )
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

    def _send(self, conn, operation: Operation) -> None:
        """Send one planned conversation item create or delete on the SDK connection ``conn``."""
        if isinstance(operation, Delete):
            kwargs = {"item_id": operation.item_id}
            if operation.event_id is not None:
                kwargs["event_id"] = operation.event_id
            self._wire(conn.conversation.item.delete, **kwargs)
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
        self._wire(conn.conversation.item.create, **kwargs)

    async def _send_all(self, session: _Session, operations) -> None:
        """Send operations planned on ``session`` in order, on its own connection; its core
        records what became of each."""
        core = session.core
        for operation in operations:
            core.sending(operation)
            try:
                await asyncio.to_thread(self._send, session._conn, operation)
            except BaseException:
                core.send_failed(operation)
                raise
            core.sent(operation)

    @staticmethod
    async def _wait_for_item_edits(session: _Session) -> None:
        """Wait until the server has answered the pending in-place edit; a connection that closes
        first fails it with that close."""
        await session.receive_until(lambda: session.core.edit_answered)
        if not session.core.edit_answered:
            raise session.closed

    async def _close_current(self) -> None:
        cm, self._cm = self._cm, None
        session, self._session = self._session, None
        self._conn = None
        closed = True
        if cm is not None:
            try:
                await asyncio.to_thread(cm.__exit__, None, None, None)
            except Exception:  # noqa: BLE001 - best-effort SDK cleanup
                closed = False
        if session is not None:
            await session.stop_reading(closed=closed)

    async def _connect(self, request: ConnectRequest) -> None:
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

        async def send(attempt):
            try:
                with trace_scope(connect_attempt=attempt):
                    await asyncio.to_thread(open_session)
            except Exception:
                await self._close_current()
                raise
            self._connect_finished(attempt, "ok")

        def record(attempt, exc, failure, delay):
            self._connect_finished(
                attempt, "failed" if delay is None else "retry", exc, failure, delay
            )
            if delay is None:
                raise TransportFailureError(failure, exc, attempt - 1) from exc
            # RequestRetry waits the delay on self.clock once this returns, unless the
            # generation was abandoned.
            if not retry_cancel_event().is_set():
                with trace_scope(connect_attempt=attempt):
                    self._record_wait("connect_backoff", delay)

        # The session the connect resumes: the newest retained frame, or none at all.
        items = () if request.history is None else request.history.items
        frames = [item.block.id for item in items if item.block.kind == "frame"]
        key = f"{frames[-1] if frames else 'empty'}:connect"
        retry = RequestRetry(
            key,
            self.classify_transport_error,
            record,
            cancelled=lambda attempt, exc: self._connect_finished(attempt, "cancelled", exc),
            clock=self.clock,
        )
        await retry.run_async(send)
        if self._conn is None:  # a concurrent close() ended the session that just opened
            raise RuntimeError("OpenAI Realtime session closed while it opened")
        # The SDK keeps the WebSocket connection, which the trace numbered, as ``_connection``
        # (openai/resources/realtime/realtime.py:765).
        connection = self._conn._connection
        session = self._session = _Session(
            connection.trace_connection_number,
            self._conn,
            RealtimeSessionCore(
                self._new_client_id,
                partial(self._record_delivery, connection=connection.trace_connection_id),
            ),
        )
        # The history is None when the runtime opens the episode's first session.
        if self.streaming:
            session.read_continuously(self.clock)
            replay = session.core.replay_streaming(request.history)
            await self._send_all(session, replay)
            # A fold anchors a frame on the item its confirmation says the server put it after:
            # the session takes pushes once the replay's are settled.
            pushed = [op for op in replay if op.event_id is not None]
            await session.receive_until(lambda: all(map(session.core.settled, pushed)))
            if not all(map(session.core.settled, pushed)):
                raise session.closed
        else:
            await self._send_all(session, session.core.replay(request.history))

    def _connect_finished(self, attempt, outcome, error=None, failure=None, delay=None) -> None:
        """One ModelAttemptFinished per open_session attempt of the shared connect retry, and one
        for the attempt a cancellation stopped."""
        with trace_scope(connect_attempt=attempt):
            self._trace.capture(
                TraceEvent.MODEL_ATTEMPT_FINISHED,
                {
                    "kind": "session_connect",
                    "outcome": outcome,
                    "failure_kind": failure.kind if failure is not None else None,
                    "error": repr(error)[:500] if error is not None else None,
                    "retry_delay_s": delay,
                },
            )

    def _connected_session(self) -> _Session:
        if self._session is None:
            raise RuntimeError("OpenAI Realtime session is not connected")
        return self._session

    async def send_observation(self, observation: ProviderObservation, history) -> dict:
        # The history ends with this observation's frame, after the feedback on the actions
        # before it.
        session = self._connected_session()
        core = session.core
        await self._send_all(session, [*core.new_feedback(history), core.observe(history)])
        return {}

    async def start_generation(self, request: GenerationRequest) -> GenerationHandle:
        session = self._connected_session()
        # The frame this response answers is the request's newest; its output items join it.
        _, frames = history_frames(request.canonical.items)
        core_generation = session.core.start_generation(frames[-1].id)
        return await self._create_response(session, core_generation, request)

    async def _create_response(
        self, session: _Session, core_generation, request: GenerationRequest
    ) -> GenerationHandle:
        logger.debug(
            "OpenAI Realtime connection %d: generation %d is %s",
            session.number,
            core_generation.number,
            request.generation_id,
        )
        response = {"max_output_tokens": request.transport_output_token_cap}
        if self.config.reasoning_effort is not None:
            response["reasoning"] = {"effort": self.config.reasoning_effort}
        await asyncio.to_thread(
            self._wire,
            session._conn.response.create,
            response=response,
        )
        return GenerationHandle(request.generation_id, opaque=(session, core_generation))

    async def push_frame(self, observation: ProviderObservation, history) -> None:
        """Native streaming: push the newest frame of the streaming history view, after the
        feedback the view holds that the session lacks, and fold the frames the view folds
        (decision 23). A frame the session already holds, as a rebuild replayed it or
        request_reply pushed it, is not pushed again. Returns once the server confirmed the
        frame; raises when the server refused it, which loses that frame alone, or the
        connection closed first. A fold the server fails leaves the session dirty, for the next
        ensure_ready to rebuild it from the view.

        It plans and writes holding the session as a connection operation does
        (SessionLifecycle), so its writes go to the session it planned them on once a connect
        in progress took its replay; it waits for the server without holding it, so a rebuild
        meanwhile closes that session, which fails the push."""
        async with self._session_lock():
            session = self._connected_session()
            core = session.core
            await self._send_all(session, core.push(history))
            batch = core.queue_edit(history)
            if batch is not None:
                await self._send_all(session, batch.operations)
        await self._fold(session, batch)
        frame = core.frame_push(history)
        await session.receive_until(lambda: core.settled(frame))
        if not core.settled(frame):
            raise session.closed
        if frame.error is not None:
            raise RuntimeError(f"OpenAI Realtime refused {frame.block_id}: {frame.error!r}")

    async def _fold(self, session: _Session, batch) -> None:
        """Bring the session to the view's folds, one edit at a time (queue_edit): wait for the
        server to answer the edit sent, then send the view kept behind it, as push_frame writes."""
        core = session.core
        while batch is not None:
            await self._wait_for_item_edits(session)
            try:
                core.finish_edit()
            except RuntimeError:
                logger.warning(
                    "OpenAI Realtime connection %d: a fold failed", session.number, exc_info=True
                )
                self.mark_dirty()
                return
            async with self._session_lock():
                batch = core.next_edit()
                if batch is not None:
                    await self._send_all(session, batch.operations)

    async def request_reply(self, request: GenerationRequest) -> GenerationHandle:
        """Native streaming: ask for a reply on the session as it is, after pushing what the
        streaming history view ``request.canonical`` holds that the session lacks: the feedback
        accepted since the newest frame, and that frame when push_frame has not pushed it yet,
        which then only waits for its confirmation. Its events come as a turn's do. It holds
        the session as push_frame's writes do, so the reply goes to the session it pushed to."""
        async with self._session_lock():
            session = self._connected_session()
            await self._send_all(session, session.core.push(request.canonical))
            return await self._create_response(session, session.core.start_reply(), request)

    async def events(self, generation: GenerationHandle):
        session, core_generation = generation.opaque
        pending = core_generation.events
        numbers = {"connection": session.number, "generation": core_generation.number}
        sequence = 0
        while True:
            await session.receive_until(lambda: pending)
            if not pending:
                return  # the connection closed normally
            event = pending.popleft()
            event_type = getattr(event, "type", "")
            if event_type == "response.created":
                sequence += 1
                yield GenerationBoundary(
                    generation.generation_id,
                    sequence,
                    event_type,
                    "started",
                    ProviderHistoryState.UNKNOWN,
                    **numbers,
                )
                continue
            if event_type == "error":
                error = getattr(event, "error", None)
                cancel_event_id = core_generation.cancel_event_id
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
                    **numbers,
                )
                continue
            if event_type.endswith(("output_text.delta", "text.delta")):
                sequence += 1
                yield TextDelta(
                    generation.generation_id,
                    sequence,
                    event_type,
                    getattr(event, "delta", "") or "",
                    "visible",
                    **numbers,
                )
                continue
            if event_type == "response.output_text.done":
                # Native streaming's reader recorded when it came off the wire
                # (RealtimeSessionCore.arrived); a turn's reader records no time.
                received_at = core_generation.received_at.get(getattr(event, "event_id", None))
                if received_at is not None:
                    sequence += 1
                    yield TextDone(
                        generation.generation_id,
                        sequence,
                        event_type,
                        getattr(event, "text", "") or "",
                        received_at=received_at,
                        **numbers,
                    )
                continue
            if event_type == "response.done":
                response = getattr(event, "response", None)
                usage = object_dict(getattr(response, "usage", None))
                if usage is not None:
                    sequence += 1
                    yield UsageSnapshot(
                        generation.generation_id,
                        sequence,
                        event_type,
                        usage,
                        **numbers,
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
                        **numbers,
                    )
                    return
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
                    **numbers,
                )
                return

    async def interrupt(self, generation: GenerationHandle) -> InterruptResult:
        session, core_generation = generation.opaque
        # A response on a session closed or replaced since was closed with its connection.
        if self._session is not session:
            return InterruptResult(ProviderHistoryState.UNKNOWN)
        kwargs = {}
        response_id = core_generation.response_id
        if response_id:
            kwargs["response_id"] = response_id
        cancel_event_id = self._new_client_id("cancel")
        kwargs["event_id"] = cancel_event_id
        core_generation.cancel_event_id = cancel_event_id
        try:
            await asyncio.to_thread(self._wire, session._conn.response.cancel, **kwargs)
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
        session = self._session
        problem = session.core.in_place_problem(request.history)
        if problem is not None:
            return await self._fallback_from_in_place(request, problem)

        try:
            batch = session.core.begin_edit(request.history)
            await self._send_all(session, batch.operations)
            await self._wait_for_item_edits(session)
            session.core.finish_edit()
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

    async def _sync_history(self, request: HistorySyncRequest) -> HistorySyncResult:
        if self.streaming:
            # Native streaming edits the session as frames are pushed (push_frame); any sync
            # rebuilds it from the streaming view (decisions 4.7, 4.14).
            return await self._rebuild(request)
        generation = request.current_generation
        if (
            not request.force_rebuild
            and request.reason == "history_block_compaction"
            and self._conn is not None
        ):
            return await self._compact_in_place(request)
        if request.force_rebuild or generation is None or self._conn is None:
            return await self._rebuild(request)
        _, core_generation = generation.opaque
        output_ids = list(core_generation.output_item_ids)
        try:
            if not output_ids:
                raise RuntimeError("response did not expose output item IDs")
            # The response answered the newest frame; its output replaces what the server kept.
            session = self._session
            await self._send_all(session, session.core.repair(request.history, output_ids))
            return HistorySyncResult(
                SyncDisposition.EXACT,
                ProviderHistoryState.EXACT,
                detail="cancelled response replaced with the complete collected output",
                provider_metadata={"deleted_output_items": len(output_ids)},
            )
        except Exception:  # noqa: BLE001 - fall back to canonical replay
            return await self._rebuild(request)
