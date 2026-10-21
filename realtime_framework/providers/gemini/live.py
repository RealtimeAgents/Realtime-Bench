from __future__ import annotations

# Gemini live
"""Gemini Live protocol driver for the shared realtime runtime."""

import asyncio as asyncio
import contextvars
import logging as logging
import os as os
from dataclasses import dataclass, field
from typing import Any
from typing import ClassVar as ClassVar

from realtime_framework.clock import Clock
from realtime_framework.clock import SystemClock
from realtime_framework.entity.model import ConnectRequest
from realtime_framework.entity.model import Deployment
from realtime_framework.entity.events import FunctionCall
from realtime_framework.entity.events import FunctionCallsCancelled
from realtime_framework.entity.events import GenerationBoundary
from realtime_framework.entity.model import GenerationHandle
from realtime_framework.entity.model import GenerationRequest
from realtime_framework.entity.model import HistorySyncRequest
from realtime_framework.entity.model import HistorySyncResult
from realtime_framework.entity.model import InterruptResult
from realtime_framework.entity.events import ProviderHistoryState
from realtime_framework.entity.model import ProviderObservation
from realtime_framework.entity.model import RealtimeCapabilities
from realtime_framework.entity.model import SessionSeed
from realtime_framework.entity.model import SyncDisposition
from realtime_framework.entity.events import TextDelta
from realtime_framework.entity.events import UsageSnapshot
from realtime_framework.entity.events import EmptyInterruptedToolTurnError
from realtime_framework.entity.model import FATAL
from realtime_framework.entity.model import QUOTA_EXHAUSTED
from realtime_framework.entity.model import RATE_LIMITED
from realtime_framework.entity.model import TRANSIENT
from realtime_framework.entity.model import TransportFailure
from realtime_framework.entity.records import current_trace_scope
from realtime_framework.entity.records import enter_trace_scope
from realtime_framework.entity.request import CanonicalRequest
from realtime_framework.providers.capture import async_ws_connection
from realtime_framework.providers.gemini.session import Connect
from realtime_framework.providers.gemini.session import Content
from realtime_framework.providers.gemini.session import Orphan
from realtime_framework.providers.gemini.session import Part
from realtime_framework.providers.gemini.session import RealtimeInput
from realtime_framework.providers.gemini.session import Received
from realtime_framework.providers.gemini.session import SessionCore
from realtime_framework.providers.session_lifecycle import RenewalRequired
from realtime_framework.providers.session_lifecycle import SessionLifecycle

logger = logging.getLogger(__name__)

_SESSION_RELEASE_GRACE_S = 2.0
_CONNECTION_CLOSE_TIMEOUT_S = 1.0


# Wire options carried in ModelConfig.provider_options: (default, allowed values). Gemini 3.8
# Extended Thinking requires asynchronous tool declarations and remains active across
# intermediate utterance-level turnComplete messages until interaction_status becomes IDLE.
# Ordered screenshot/text turns (client_content) avoid synthesizing an audio activity for GUI input.
LIVE_PROTOCOL_OPTIONS = {
    "tool_behavior": ("blocking", ("blocking", "non_blocking")),
    "completion_signal": ("turn_complete", ("turn_complete", "interaction_idle")),
    "input_transport": ("realtime_input", ("realtime_input", "client_content")),
    # when_idle and omit both send no scheduling field; omit names that choice for the models
    # without function scheduling (3.1 Flash Live, 3.8 Extended Thinking).
    "tool_response_scheduling": ("when_idle", ("when_idle", "silent", "omit")),
}
# Context policy item 10: how past tool calls are replayed into a rebuilt session. It stays in
# provider_options, where ContextPolicy.for_path reads it and this driver records it.
LIVE_HISTORY_OPTIONS = {
    "history_format": ("native_tools", ("native_tools", "tool_transcript")),
}
LIVE_OPTIONS = {**LIVE_PROTOCOL_OPTIONS, **LIVE_HISTORY_OPTIONS}

# Per-model presets, merged into the catalog entry's defaults (providers/registry.py).
_CLIENT_CONTENT_TOOLS = {
    "tool_behavior": "non_blocking",
    "completion_signal": "turn_complete",
    "input_transport": "client_content",
    "tool_response_scheduling": "silent",
}
_TOOL_TRANSCRIPT = {"history_format": "tool_transcript"}  # context policy, see above


_RESPONSE_MESSAGES = {"serverContent", "toolCall", "toolCallCancellation", "usageMetadata"}


def classify_live_message(direction: str, message: dict):
    """Name a Gemini Live message for the agent trace; only setup is acknowledged.

    client_content, realtime_input and tool_response have no acknowledgement in the protocol, so
    they stay sent. goAway, sessionResumptionUpdate and voice activity come unsolicited.
    """
    kind = next(iter(message), "")
    if direction == "client":
        return kind, [("setup",)] if kind == "setup" else [], None
    if "setupComplete" in message:
        return kind, [("setup",)], "unsolicited"
    return kind, [], "response" if _RESPONSE_MESSAGES & set(message) else "unsolicited"


def live_options(config) -> dict:
    """Resolve Gemini Live options with their documented defaults.

    The values are validated before the run (providers/preflight.py).
    """
    options = config.provider_options
    return {name: options.get(name, default) for name, (default, _) in LIVE_OPTIONS.items()}


# reasoning_effort values the session setup maps (GeminiLiveDriver._thinking_config).
LIVE_REASONING_EFFORTS = (None, "fixed", "none", "minimal", "low", "medium", "high")


class GoAwayDeadlineExceeded(RuntimeError):
    """An incomplete native turn must remain invalid, even without visible output."""


@dataclass
class _StreamedReply:
    """Streaming: a reply the reader reads for, until its events() is done, on the connection
    it started on (``session`` and its reader). ``messages`` holds what the reader read and
    events() has not taken yet, or the reader's failure; ``ended``, whether it read the reply's
    end."""

    session: Any
    receiver: _Receiver
    messages: asyncio.Queue = field(default_factory=asyncio.Queue)
    ended: bool = False


class _Receiver:
    """The one reader of a connection once it is set up: each server message, applied to the
    session core, which stamps it with the connection and the generation whose reply was being
    read. The setup's acknowledgement (setupComplete), the only one Gemini Live sends, is read by
    google-genai inside its connect, before the receiver exists.

    google-genai's ``session.receive()`` iterator ends at every bare ``turn_complete`` and at
    IDLE, so the receiver opens the next one and spans every generation on the connection.
    Extended Thinking uses ``turn_complete`` for intermediate spoken fillers; the core decides
    where a generation ends. Turn-based, it reads only when a generation asks for a message.
    Streaming, the driver's reader asks for every message as it arrives, and gets the orphans
    too (SessionCore.arrived).

    A read that fails or is cancelled leaves the connection unusable: every later read raises the
    failure again. A cancelled read is kept as a RuntimeError caused by the cancellation, so that
    a later caller is not reported as cancelled itself.
    """

    def __init__(self, session, number: int, core: SessionCore, clock: Clock):
        self.number = number
        self.failure: BaseException | None = None
        self._session = session
        self._core = core
        self._clock = clock
        self._messages = None  # the SDK's current receive() iterator
        self._opened_empty = False  # it has given no message yet

    async def next(self) -> Received | Orphan:
        if self.failure is None:
            try:
                return await self._read()
            except asyncio.CancelledError as exc:
                self.failure = RuntimeError(
                    f"Gemini Live connection {self.number}: a read was cancelled"
                )
                self.failure.__cause__ = exc
                self._core.reader_failed(self.number)
                raise
            except Exception as exc:
                self.failure = exc
        self._core.reader_failed(self.number)
        raise self.failure

    async def _read(self) -> Received | Orphan:
        while True:
            if self._messages is None:
                self._messages = self._session.receive().__aiter__()
                self._opened_empty = True
            try:
                message = await anext(self._messages)
            except StopAsyncIteration:
                if self._opened_empty:
                    raise RuntimeError(
                        f"Gemini Live connection {self.number}: receive() ended without a message"
                    ) from None
                self._messages = None
                continue
            self._opened_empty = False
            if self._core.streaming:
                return self._core.arrived(self.number, message, self._clock.monotonic())
            received = self._core.received(self.number, message, self._clock.monotonic())
            if received is not None:
                return received
            logger.debug(
                "Gemini Live connection %d: no generation takes %s",
                self.number,
                type(message).__name__,
            )


class GeminiLiveDriver(SessionLifecycle):
    capabilities = RealtimeCapabilities(
        protocol_family="gemini_live",
        persistent_session=True,
        # The system instruction goes into the session setup, not into a content role.
        roles=("input", "model"),
        feedback_placements=("separate_message",),
        server_context="unknown",
        # tool_transcript: text that keeps the wire fields of each call and its receipt.
        tool_call_replay=("native_tools", "tool_transcript"),
        max_video_fps=1.0,
        terminal_output_usage=True,
        action_channel="function_call",
        classifies_transport_errors=True,
        connection_renewal=True,
        interaction_modes=frozenset({"turn_based", "streaming"}),
    )
    deployment = Deployment(
        family="streaming",
        provider="gemini",
        key_env="GEMINI_API_KEY",
        vendor="google",
        litellm_provider="gemini",
    )
    model_defaults: ClassVar[dict[str, dict]] = {
        # 3.1 Flash Live has no asynchronous function calling: a call blocks until its
        # response, and response scheduling does not apply.
        "gemini-3.1-flash-live-preview": {
            "provider_options": {
                **_CLIENT_CONTENT_TOOLS,
                "tool_behavior": "blocking",
                "tool_response_scheduling": "omit",
                **_TOOL_TRANSCRIPT,
            },
        },
        "gemini-3.8-live": {
            "provider_options": {**_CLIENT_CONTENT_TOOLS, **_TOOL_TRANSCRIPT},
        },
        "gemini-3.8-live-extended-thinking": {
            "provider_options": {
                **_CLIENT_CONTENT_TOOLS,
                "completion_signal": "interaction_idle",
                "tool_response_scheduling": "omit",
                **_TOOL_TRANSCRIPT,
            },
        },
    }
    # The API's native-streaming settings: the setup's context_window_compression. Once the
    # session reaches compression_trigger_tokens, the server shortens it to
    # compression_target_tokens, from the start of a user turn, and says nothing of what it
    # removed; the system instruction stays (google-genai SlidingWindow: target_tokens "Should be
    # < trigger_tokens", which the catalog requires). Placeholders until a smoke test
    # (docs/streaming_smoke_test.md): a trigger at half the 131,072-token input of these models in
    # LiteLLM's model map, and a target at half the trigger, the server's default when none is
    # set. A model YAML's streaming settings replace those they name (ModelConfig.streaming); a
    # driver holds the values it runs with.
    streaming_settings: ClassVar[dict] = {
        "compression_trigger_tokens": 65_536,
        "compression_target_tokens": 32_768,
    }

    def __init__(self, config, *, clock=None, streaming: bool = False):
        """``streaming``: the session native streaming drives, which pushes frames, triggers
        replies and reads every message as it arrives."""
        from google import genai as genai
        from google.genai import errors as errors
        from google.genai import types as types
        from websockets.exceptions import ConnectionClosed as ConnectionClosed

        self.options = live_options(config)
        if config.api_key:
            keys = [config.api_key]
        elif config.api_key_env and config.api_key_env != "GEMINI_API_KEY":
            keys = [os.environ.get(config.api_key_env)]
        else:
            keys = [os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")]
            keys.extend(os.environ.get(f"GEMINI_API_KEY_{index}") for index in range(2, 10))
        self._api_keys = tuple(dict.fromkeys(key for key in keys if key))
        if not self._api_keys:
            raise RuntimeError(
                "Gemini Live needs GEMINI_API_KEY/GOOGLE_API_KEY or ModelConfig.api_key"
            )
        self.config = config
        self.clock = clock or SystemClock()
        self.streaming = streaming
        self.streaming_settings = {**self.streaming_settings, **config.streaming}
        self.types = types
        self._api_error_type = errors.APIError
        self._connection_closed_type = ConnectionClosed
        self._client_type = genai.Client
        self._clients = {}
        requested_slot_raw = os.environ.get("GEMINI_API_KEY_START_SLOT", "1")
        try:
            requested_slot = int(requested_slot_raw)
        except ValueError as exc:
            raise ValueError("GEMINI_API_KEY_START_SLOT must be a positive integer") from exc
        if requested_slot < 1:
            raise ValueError("GEMINI_API_KEY_START_SLOT must be a positive integer")
        self._api_key_requested_slot = requested_slot
        self._api_key_index = (requested_slot - 1) % len(self._api_keys)
        self._ws_connection = async_ws_connection(self, "gemini_live.ws", classify_live_message)
        self.client = self._client_for_slot(self._api_key_index)
        self._cm = None
        self._session = None
        self._receiver: _Receiver | None = None  # the open connection's reader
        self._reading: asyncio.Task | None = None  # streaming: the task reading it
        self._reading_context: contextvars.Context | None = None  # and the context it runs in
        # Streaming: the replies the reader reads for, by generation id; and the cancellations
        # read between replies, which the next reply's events() hands over.
        self._replies: dict[str, _StreamedReply] = {}
        self._cancelled_between: list[Orphan] = []
        self._trace_connection: str | None = None  # the trace's id of the last connection opened
        self._trace_connections: dict[int, str] = {}  # and of each one, by the core's number
        self._core = SessionCore(self.options, streaming=streaming, on_delivery=self._delivered)
        self._activity_open = False
        self._last_video_sent_at: float | None = None
        # The goAway deadline the trace has a wait for, on the open connection.
        self._go_away_wait_recorded: float | None = None
        self._connect_metadata: dict = {}

    def _client_for_slot(self, index: int):
        client = self._clients.get(index)
        if client is None:
            # google-genai passes create_connection only to the WebSocket connect.
            http_options = {"async_client_args": {"create_connection": self._ws_connection}}
            if self.config.api_base_url:
                http_options["base_url"] = self.config.api_base_url
            client = self._client_type(api_key=self._api_keys[index], http_options=http_options)
            self._clients[index] = client
        return client

    def bind_trace(self, trace) -> None:
        self._trace = trace

    def _delivered(self, state: str, contents: list[tuple], connection: int | None) -> None:
        """What became of writes the core recorded, on the trace's connection the core numbered
        ``connection``: also an acknowledgement an abandoned reply wrote late on a connection
        closed since. One being set up has no number yet: the connection opened last."""
        self._record_delivery(
            state, contents, self._trace_connections.get(connection, self._trace_connection)
        )

    @staticmethod
    def _exception_chain(exc: BaseException):
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            yield current
            seen.add(id(current))
            current = current.__cause__ or current.__context__

    def _rate_limit_kind(self, exc: BaseException) -> str | None:
        for current in self._exception_chain(exc):
            code = int(getattr(current, "code", 0) or 0)
            status = str(getattr(current, "status", "") or "").strip().lower()
            detail = (
                str(getattr(current, "message", None) or getattr(current, "details", "") or "")
                .strip()
                .lower()
            )
            if isinstance(current, self._connection_closed_type):
                close = getattr(current, "rcvd", None) or getattr(current, "sent", None)
                code = int(getattr(close, "code", code) or 0)
                detail = str(getattr(close, "reason", detail) or "").strip().lower()
            if "exceeded your current quota" in detail or "quota exceeded" in detail:
                return QUOTA_EXHAUSTED
            if code == 429 or status == "resource_exhausted":
                return RATE_LIMITED
        return None

    def classify_transport_error(self, exc: BaseException) -> TransportFailure:
        """Classify a failed turn for the harness's shared retry policy."""
        if any(
            isinstance(current, GoAwayDeadlineExceeded) for current in self._exception_chain(exc)
        ):
            return TransportFailure(FATAL)
        limited = self._rate_limit_kind(exc)
        if limited is not None:
            return TransportFailure(limited)
        return TransportFailure(TRANSIENT if self._is_transient(exc) else FATAL)

    def _is_transient(self, exc: BaseException) -> bool:
        """Classify only connection failures that are safe for canonical replay.

        WebSocket close code 1007 normally covers invalid payloads and must not be retried
        indiscriminately. Gemini Live also uses it for the transient session-level
        ``Precondition check failed.`` close observed after an accepted setup/history write, so
        restrict that code to the exact provider reason.
        The unified runtime additionally requires that no text, native call, or audio was received.
        Usage must either be absent or a terminal interrupted turn proving exactly zero generated
        tokens.
        Generic ABORTED and service-unavailable closes are also transient candidates; other 1008
        policy violations and explicit GoAway deadline failures remain terminal.
        """
        for current in self._exception_chain(exc):
            if isinstance(current, EmptyInterruptedToolTurnError):
                return True
            if isinstance(current, self._connection_closed_type):
                close = getattr(current, "rcvd", None) or getattr(current, "sent", None)
                code = int(getattr(close, "code", 0) or 0)
                reason = str(getattr(close, "reason", "") or "").strip().lower()
                if code in {0, 1001, 1006, 1012, 1013}:
                    return True
                if code == 1007 and reason == "precondition check failed.":
                    return True
                if code == 1008 and reason == "the operation was aborted.":
                    return True
                if code == 1011 and (
                    "internal error" in reason or reason == "the service is currently unavailable."
                ):
                    return True
            if isinstance(current, (ConnectionError, TimeoutError)):
                return True
            if isinstance(current, self._api_error_type):
                code = int(getattr(current, "code", 0) or 0)
                detail = (
                    str(getattr(current, "message", None) or getattr(current, "details", "") or "")
                    .strip()
                    .lower()
                )
                if code == 1006:
                    return True
                if code == 1007 and detail == "precondition check failed.":
                    return True
                if code == 1008 and detail == "the operation was aborted.":
                    return True
                if code == 1011 and (
                    "internal error" in detail or detail == "the service is currently unavailable."
                ):
                    return True
        return False

    def _realtime_input_config(self):
        handling = self.types.ActivityHandling
        return self.types.RealtimeInputConfig(
            automatic_activity_detection=self.types.AutomaticActivityDetection(disabled=True),
            # A streaming trigger is an activity, and must not cut off the reply still running.
            activity_handling=(
                handling.NO_INTERRUPTION
                if self.streaming
                else handling.START_OF_ACTIVITY_INTERRUPTS
            ),
            turn_coverage=self.types.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
        )

    def _context_window_compression(self):
        settings = self.streaming_settings
        return self.types.ContextWindowCompressionConfig(
            trigger_tokens=settings["compression_trigger_tokens"],
            sliding_window=self.types.SlidingWindow(
                target_tokens=settings["compression_target_tokens"]
            ),
        )

    def _thinking_config(self):
        effort = self.config.reasoning_effort
        if effort is None:
            return None
        # Gemini 3.8 Live has a fixed interleaved-reasoning profile and rejects thinking_config.
        # The catalog's explicit `fixed` value keeps that provider behavior identity-bearing while
        # ensuring the field is omitted from session setup.
        if effort == "fixed":
            return None
        if effort == "none":
            return self.types.ThinkingConfig(thinking_budget=0)
        levels = {
            "minimal": self.types.ThinkingLevel.MINIMAL,
            "low": self.types.ThinkingLevel.LOW,
            "medium": self.types.ThinkingLevel.MEDIUM,
            "high": self.types.ThinkingLevel.HIGH,
        }
        return self.types.ThinkingConfig(thinking_level=levels[effort])

    def _function_declarations(self, seed):
        behavior = self.options["tool_behavior"]
        declarations = []
        for item in seed.answer.function_declarations:
            values = dict(item)
            if behavior == "non_blocking":
                values["behavior"] = self.types.Behavior.NON_BLOCKING
            else:
                # 3.8 defaults to NON_BLOCKING on the server; select blocking explicitly.
                values["behavior"] = self.types.Behavior.BLOCKING
            declarations.append(self.types.FunctionDeclaration(**values))
        return declarations

    def _live_config(self, seed: SessionSeed, connect: Connect):
        """The setup of one planned connection. Streaming takes realtime input whatever the
        model's transport, and has the server compress the context: without it an audio-video
        session lasts two minutes."""
        streaming = self.streaming
        return self.types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=self.types.AudioTranscriptionConfig(),
            tools=[self.types.Tool(function_declarations=self._function_declarations(seed))],
            thinking_config=self._thinking_config(),
            system_instruction=self.types.Content(
                parts=[self.types.Part(text=seed.system_instruction)]
            ),
            session_resumption=self.types.SessionResumptionConfig(handle=connect.resume_handle),
            history_config=(
                self.types.HistoryConfig(initial_history_in_client_content=True)
                if connect.seed
                else None
            ),
            realtime_input_config=(
                self._realtime_input_config()
                if streaming or self.options["input_transport"] == "realtime_input"
                else None
            ),
            context_window_compression=self._context_window_compression() if streaming else None,
        )

    def _content(self, content: Content):
        """A planned Content as google-genai's Content."""
        return self.types.Content(
            role=content.role, parts=[self._part(part) for part in content.parts]
        )

    def _part(self, part: Part):
        if part.image is not None:
            return self.types.Part(
                inline_data=self.types.Blob(mime_type=part.image.mime_type, data=part.image.data)
            )
        if part.function_call is not None:
            values = {"function_call": self.types.FunctionCall(**part.function_call)}
            if part.thought_signature is not None:
                values["thought_signature"] = part.thought_signature
            return self.types.Part(**values)
        if part.function_response is not None:
            return self.types.Part(
                function_response=self.types.FunctionResponse(**part.function_response)
            )
        return self.types.Part(text=part.text)

    async def _close_current(self) -> bool:
        """Close the open connection, if any; returns whether there was one."""
        cm, self._cm = self._cm, None
        reading, self._reading = self._reading, None
        session, self._session = self._session, None
        self._receiver = None
        self._activity_open = False
        self._last_video_sent_at = None
        self._go_away_wait_recorded = None
        self._core.closed()
        if self.streaming:
            # Closing cut off whatever the session was taking: it is rebuilt, not resumed. A
            # renewal's connect plans its own resume after this.
            self._core.unresumable()
        if reading is not None:
            # Its replies end with the cancelled read, not with whatever the close would raise.
            reading.cancel()
            await asyncio.wait({reading})
        if cm is not None:
            self._record_wait("connection_close", _CONNECTION_CLOSE_TIMEOUT_S)
            try:
                await self.clock.wait_for(
                    cm.__aexit__(None, None, None),
                    _CONNECTION_CLOSE_TIMEOUT_S,
                )
            except Exception:  # noqa: BLE001 - best-effort SDK cleanup
                # A close stuck on a write that does not drain leaves that write stuck too, and
                # a streaming push holds the session while it writes: drop the connection under
                # both, which fails the write. google-genai keeps the WebSocket as ``_ws``.
                session._ws.transport.abort()
        return cm is not None

    async def _connect(self, request: ConnectRequest) -> None:
        """One attempt: a Gemini Live connect has no retry of its own."""
        replacing = await self._close_current()
        connect = self._core.connect(request.history, request.resume_handle, replacing=replacing)
        if connect.release_grace:
            # Closing the local WebSocket and releasing the server-side Live session are not one
            # atomic operation. Apply the grace period to every reconnect/history replay, including
            # a retry whose failed predecessor no longer has a local context manager. This is
            # wall-clock-only and avoids briefly exceeding a one-session provider quota.
            await self._wait("session_release", _SESSION_RELEASE_GRACE_S)
        config = self._live_config(request.seed, connect)
        cm = self.client.aio.live.connect(model=request.seed.model_name, config=config)
        try:
            session = await cm.__aenter__()
            if getattr(session, "setup_complete", None) is None:
                raise RuntimeError("Gemini Live did not acknowledge session setup")
            # google-genai keeps the WebSocket connection, which the trace numbered, as ``_ws``.
            self._trace_connection = session._ws.trace_connection_id
            # Seed history explicitly before either input transport. Merely leaving
            # turn_complete=False can cause historical model content to be echoed as new
            # output. HistoryConfig makes this a non-generating initialization boundary.
            # Bound the messages so screenshots do not form one oversized WebSocket frame.
            index = 0
            try:
                for index, batch in enumerate(connect.seed):
                    await session.send_client_content(
                        turns=[self._content(content) for content in batch],
                        turn_complete=index == len(connect.seed) - 1,
                    )
            except BaseException:
                self._core.seed_failed(connect, index)
                raise
        except BaseException as exc:
            try:
                await cm.__aexit__(type(exc), exc, exc.__traceback__)
            except Exception:  # noqa: BLE001, S110 - preserve the handshake failure
                pass
            raise
        self._cm = cm
        self._session = session
        self._activity_open = False
        self._last_video_sent_at = None
        self._core.connected(connect)
        self._trace_connections[self._core.connection] = self._trace_connection
        self._receiver = _Receiver(session, self._core.connection, self._core, self.clock)
        if self.streaming:
            # The reader records what it reads in the scope of the reply it reads for
            # (start_generation hands it over), not in the scope of the call that connected.
            self._reading_context = contextvars.Context()
            self._reading = asyncio.get_running_loop().create_task(
                self._read_on(self._receiver), context=self._reading_context
            )
        self._connect_metadata = {
            "session_connect_mode": connect.mode,
            "session_resume_handle_used": connect.resume_handle is not None,
            "history_seed_contents": connect.contents,
            "history_seed_messages": len(connect.seed),
            "session_replacement_grace_s": (
                _SESSION_RELEASE_GRACE_S if connect.release_grace else 0.0
            ),
            "api_key_slot": self._api_key_index + 1,
            "api_key_requested_slot": self._api_key_requested_slot,
            "api_key_pool_size": len(self._api_keys),
        }
        if self.streaming:
            self._connect_metadata.update(
                {
                    "gemini_live_context_window_compression": dict(self.streaming_settings),
                    # The server drops the oldest turns past the trigger and does not say which.
                    "gemini_live_server_context": "unknown",
                }
            )

    async def _read_on(self, receiver: _Receiver) -> None:
        """Streaming: read every message of the connection as it arrives. A reply's messages go
        to its events(); an orphan's goAway and resumption update are the core's to keep, and
        nothing else of it is taken but its cancellations. Those are held, also past this
        connection's close, for the next reply's events(), on this connection or a later one: it
        hands them over first, in its own generation, telling the connection they were read on;
        their calls, of any earlier reply, are told by id only. A read that fails or is
        cancelled ends every reply waiting with the receiver's failure."""
        try:
            while True:
                arrived = await receiver.next()
                reply = (
                    self._replies.get(arrived.generation_id)
                    if isinstance(arrived, Received)
                    else None
                )
                if reply is not None:
                    reply.messages.put_nowait(arrived)
                    reply.ended = reply.ended or arrived.ends
                    if arrived.ends:
                        enter_trace_scope(None)  # until the next reply hands its own over
                    continue
                if isinstance(arrived, Orphan) and arrived.cancelled:
                    self._cancelled_between.append(arrived)
                    continue
                # An orphan, or the rest of a reply whose events() already failed.
                logger.debug(
                    "Gemini Live connection %d: no generation takes %s",
                    receiver.number,
                    type(arrived.message).__name__,
                )
        except asyncio.CancelledError:
            self._end_replies(receiver.failure)
            raise
        except Exception:  # noqa: BLE001 - the receiver keeps it as its failure
            self._end_replies(receiver.failure)

    def _end_replies(self, failure: BaseException) -> None:
        for reply in self._replies.values():
            reply.messages.put_nowait(failure)

    async def _streamed(self, reply: _StreamedReply, receiver: _Receiver) -> Received:
        """Streaming: the next message the reader read for this reply. One it read already is
        taken at once: the goAway deadline bounds only the wait for one not read yet, as it
        bounds a turn-based read."""
        if not reply.messages.empty():
            received = reply.messages.get_nowait()
        elif receiver.failure is not None:
            raise receiver.failure
        else:
            received = await self._before_go_away_deadline(reply.messages.get())
        if isinstance(received, BaseException):
            raise received
        return received

    def connection_renewal_reason(self) -> str | None:
        # Let the provider-defined connection lifetime be observable. Reconnect only when Gemini
        # sends GoAway and supplies a resumption handle; there is no client-side age threshold.
        if self._core.renewal_required:
            return "go_away"
        return None

    async def _before_go_away_deadline(self, operation):
        """Bound receive and tool acknowledgements by the server's connection-close deadline."""
        deadline = self._core.go_away_deadline
        if deadline is None:
            return await operation
        remaining = max(0.0, deadline - self.clock.monotonic())
        # One wait until the deadline, however many reads it bounds; none once it has passed.
        if remaining > 0 and deadline != self._go_away_wait_recorded:
            self._go_away_wait_recorded = deadline
            self._record_wait("go_away_deadline", remaining, until=deadline)
        try:
            return await self.clock.wait_for(operation, remaining)
        except asyncio.TimeoutError:
            await self._close_current()
            # A partial native turn cannot safely be replayed or charged as zero. This distinct
            # failure is deliberately non-retryable; the task must be rerun with complete usage.
            raise GoAwayDeadlineExceeded(
                "gemini_live_go_away_deadline_before_turn_complete"
            ) from None

    def _session_for_input(self):
        """The open session, which takes input until a goAway asks for a new connection, and in
        streaming while its reader reads."""
        if self._core.renewal_required:
            raise RenewalRequired(
                "Gemini Live requires canonical history synchronization before input"
            )
        session = self._session
        if session is None:
            raise RuntimeError("Gemini Live session is not connected")
        if self.streaming and self._receiver.failure is not None:
            raise self._receiver.failure
        return session

    async def _send_realtime(self, session, item: RealtimeInput) -> None:
        if item.kind == "video":
            blob = self.types.Blob(data=item.image.data, mime_type=item.image.mime_type)
            await session.send_realtime_input(video=blob)
        elif item.kind == "text":
            await session.send_realtime_input(text=item.text)
        elif item.kind == "audio_carrier":
            await session.send_realtime_input(audio=self._audio_carrier())
        elif item.kind == "activity_start":
            await session.send_realtime_input(activity_start=self.types.ActivityStart())
        else:
            await session.send_realtime_input(activity_end=self.types.ActivityEnd())

    def _audio_carrier(self):
        # Native audio activity must contain audio. This fixed 100-ms silent carrier has no task
        # information; its input usage is audited, never output-charged.
        return self.types.Blob(data=bytes(3200), mime_type="audio/pcm;rate=16000")

    async def push_frame(self, observation: ProviderObservation, history: CanonicalRequest) -> dict:
        """Streaming: push the history view's newest frame, one the session does not hold yet,
        on the realtime_input video stream and then its text, outside any activity, so that it
        starts no reply.

        A push and a trigger hold the session as a connection operation does
        (SessionLifecycle): their messages do not interleave, none goes to a connection being
        replaced, and one given up while it waits has planned nothing.

        After a goAway, a push and a trigger raise RenewalRequired until the session is renewed;
        a frame so refused is not in the session until a push of a view it is the newest of.
        The reply in flight then still ends on this connection, by the goAway deadline; renew()
        once it has ended resumes the session with the latest resumable handle
        (SessionCore.resume_handle_for), or rebuilds it from the history where there is none, or
        where this driver closed the connection (an abort) or a trigger did not go out whole.
        renew() before then closes the connection under the reply, whose events() raise."""
        async with self._session_lock():
            session = self._session_for_input()
            if self._activity_open:
                raise RuntimeError("Gemini Live input activity is already open")
            push = self._core.push(history)
            try:
                for item in push.inputs:
                    await self._send_realtime(session, item)
            except BaseException:
                self._core.send_failed(push.keys)
                raise
        self._core.sent(push.keys)
        return {"observation_id": push.observation_id}

    async def request_reply(self, request: GenerationRequest) -> GenerationHandle:
        """Streaming: start the next reply with one activity, holding the execution feedback of
        ``request.canonical`` that the session lacks (SessionCore.trigger), once the reply before
        it ended: a connection carries one reply at a time. Its generation starts before the
        activity goes out, so that the reader takes the reply's first message for it. The
        session's NO_INTERRUPTION keeps the activity from cutting off what the server may still
        be generating. It holds the session as push_frame does."""
        async with self._session_lock():
            session = self._session_for_input()
            if self._activity_open:
                raise RuntimeError("Gemini Live input activity is already open")
            if self._core.replies.get(self._core.connection) is not None:
                raise RuntimeError("Gemini Live has a reply in flight on the connection")
            generation = await self.start_generation(request)
            trigger = None
            try:
                trigger = self._core.trigger(request.canonical)
                self._activity_open = True
                for item in trigger.inputs:
                    await self._send_realtime(session, item)
            except BaseException:
                # No reply comes, and an incomplete activity cannot be resumed as canonical
                # context: the session is rebuilt before the next reply, and its seed replays
                # the feedback.
                if trigger is not None:
                    self._core.send_failed(trigger.keys)
                self._core.reply_abandoned()
                self._core.unresumable()
                self._replies.pop(request.generation_id, None)
                raise
            self._activity_open = False
        self._core.sent(trigger.keys)
        return generation

    async def send_observation(
        self, observation: ProviderObservation, history: CanonicalRequest
    ) -> dict:
        """Send the history view's newest frame, after the feedback the session does not hold
        yet, as one input activity."""
        session = self._session_for_input()
        if self._activity_open:
            raise RuntimeError("Gemini Live input activity is already open")
        frame = self._core.observe(history)
        if self.options["input_transport"] == "client_content":
            # Send the screenshot with its grounding text in one Content, after the feedback
            # Contents, in one message that triggers generation after any seeded history has been
            # ingested. A realtime text trigger mixed with synthetic silent audio can create a
            # second user activity (including hallucinated speech) and leave synchronous tools
            # unfinished.
            try:
                await session.send_client_content(
                    turns=[self._content(content) for content in frame.contents],
                    turn_complete=True,
                )
            except BaseException:
                self._core.send_failed(frame.keys)
                raise
            self._core.sent(frame.keys)
            return {"observation_id": frame.observation_id}
        try:
            await session.send_realtime_input(activity_start=self.types.ActivityStart())
        except BaseException:
            self._core.send_failed(frame.keys)
            raise
        self._activity_open = True
        try:
            if self._last_video_sent_at is not None:
                # Video frames arrive no faster than the model-input frame rate.
                interval = 1.0 / self.config.fps
                await self._wait(
                    "video_interval",
                    max(0.0, interval - (self.clock.monotonic() - self._last_video_sent_at)),
                )
            await session.send_realtime_input(
                video=self.types.Blob(
                    data=frame.image.data,
                    mime_type=frame.image.mime_type,
                )
            )
            self._last_video_sent_at = self.clock.monotonic()
            # Let media ingestion catch up before the corresponding SoM text arrives.
            await self._wait("video_ingestion", 1.0)
            # Realtime input has no Contents: each feedback goes as its own text, before the
            # frame's.
            for content in frame.feedback:
                await session.send_realtime_input(text=content.parts[0].text)
            await session.send_realtime_input(text=frame.text)
            await session.send_realtime_input(audio=self._audio_carrier())
            await session.send_realtime_input(activity_end=self.types.ActivityEnd())
            self._activity_open = False
        except BaseException:
            # An incomplete activity cannot be resumed as canonical context.
            self._activity_open = True
            self._core.send_failed(frame.keys)
            raise
        self._core.sent(frame.keys)
        return {"observation_id": frame.observation_id}

    async def start_generation(self, request: GenerationRequest) -> GenerationHandle:
        if self._session is None:
            raise RuntimeError("Gemini Live session is not connected")
        if self.streaming:
            self._replies[request.generation_id] = _StreamedReply(self._session, self._receiver)
            # What the reader reads from here is this reply's, as for a turn-based reply's read.
            self._reading_context.run(enter_trace_scope, current_trace_scope())
        self._core.start_generation(request.generation_id)
        transport = "realtime_input" if self.streaming else self.options["input_transport"]
        return GenerationHandle(
            request.generation_id,
            # The policy's queued acknowledgement, sent for every call of this generation.
            opaque=request.canonical.policy.texts.gemini_queued_tool_response,
            metadata={
                **self._connect_metadata,
                "input_audio_carrier_ms": 100 if transport == "realtime_input" else 0,
                # A streaming push sends a frame's text right after its video.
                "video_ingestion_wait_ms": (
                    1000 if transport == "realtime_input" and not self.streaming else 0
                ),
                "gemini_live_input_transport": transport,
                "gemini_live_tool_response_scheduling": self.options["tool_response_scheduling"],
                "gemini_live_history_format": self.options["history_format"],
                "gemini_live_tool_behavior": self.options["tool_behavior"],
                "gemini_live_completion_signal": self.options["completion_signal"],
                "completed_turn_refs": {
                    "native_function_calls": [],
                    "native_tool_exchanges": [],
                },
            },
        )

    @staticmethod
    def _usage_dict(usage) -> dict | None:
        """Normalize Live usage and exclude forced audio output from the token clock.

        Gemini 3.1 Live requires AUDIO response modality even when the semantic output is a
        function call. ``response_token_count`` includes any generated audio, while the benchmark
        charges only model deliberation and the non-audio function-call response. Keep the native
        totals alongside the normalized values so provider usage and billing remain auditable.
        """
        if usage is None:
            return None
        raw_usage = (
            usage if isinstance(usage, dict) else usage.model_dump(mode="json", exclude_none=True)
        )

        def member(value, snake_name: str, camel_name: str | None = None):
            if isinstance(value, dict):
                if snake_name in value:
                    return value.get(snake_name)
                return value.get(camel_name) if camel_name else None
            return getattr(value, snake_name, None)

        def value(name: str, camel_name: str | None = None) -> int:
            return int(member(usage, name, camel_name) or 0)

        details_value = member(usage, "response_tokens_details", "responseTokensDetails")
        details = details_value or []
        response_details = [
            detail if isinstance(detail, dict) else detail.model_dump(mode="json")
            for detail in details
        ]
        prompt_details_value = member(
            usage,
            "prompt_tokens_details",
            "promptTokensDetails",
        )
        prompt_details = prompt_details_value or []
        normalized_prompt_details = [
            detail if isinstance(detail, dict) else detail.model_dump(mode="json")
            for detail in prompt_details
        ]

        def modality(detail) -> str:
            raw = member(detail, "modality")
            raw = getattr(raw, "value", raw)
            return str(raw or "").rsplit(".", 1)[-1].upper()

        def detail_tokens(detail) -> int:
            return int(member(detail, "token_count", "tokenCount") or 0)

        def first_reported(*names: str):
            for name in names:
                if name in raw_usage and raw_usage.get(name) is not None:
                    return raw_usage.get(name), name
            return None, None

        response_value, response_source = first_reported(
            "response_token_count",
            "responseTokenCount",
            "candidates_token_count",
            "candidatesTokenCount",
        )
        if response_value is not None:
            native_response = int(response_value)
            native_response_exact = True
        elif details_value is not None:
            native_response = sum(detail_tokens(detail) for detail in details)
            native_response_exact = True
            response_source = "response_tokens_details"
        else:
            native_response = 0
            native_response_exact = False
        audio_response = sum(
            detail_tokens(detail) for detail in details if modality(detail) == "AUDIO"
        )
        native_total_value = member(usage, "total_token_count", "totalTokenCount")
        native_total = int(native_total_value or 0)
        prompt_value = member(usage, "prompt_token_count", "promptTokenCount")
        tool_prompt_value = member(
            usage,
            "tool_use_prompt_token_count",
            "toolUsePromptTokenCount",
        )
        reasoning_value = member(usage, "thoughts_token_count", "thoughtsTokenCount")
        reasoning = int(reasoning_value or 0)
        tool_prompt = int(tool_prompt_value or 0)
        if (
            not native_response_exact
            and native_total_value is not None
            and prompt_value is not None
        ):
            residual = native_total - int(prompt_value)
            # With non-negative usage components, a zero residual proves that response,
            # tool-use prompt, and thought counts are all zero even when protobuf omits those
            # optional zero-valued fields. For a positive residual, derive only when both other
            # components were explicitly reported so the response count remains exact.
            if residual == 0 or (
                residual >= 0 and tool_prompt_value is not None and reasoning_value is not None
            ):
                native_response = residual - tool_prompt - reasoning
                native_response_exact = native_response >= 0
                if native_response_exact:
                    response_source = "total_minus_prompt_tool_use_and_thoughts"
        if native_response < 0:
            raise RuntimeError("Gemini response token count is negative")
        if audio_response > native_response:
            raise RuntimeError("Gemini response AUDIO token detail exceeds response token count")
        chargeable_response = native_response - audio_response
        input_tokens = (
            int(prompt_value)
            if prompt_value is not None
            else max(0, native_total - native_response - reasoning - tool_prompt)
        )
        clock_output = chargeable_response + reasoning
        input_details = {
            "cached_tokens": value(
                "cached_content_token_count",
                "cachedContentTokenCount",
            ),
        }
        if prompt_details_value is not None:
            for name in ("TEXT", "IMAGE", "VIDEO", "AUDIO"):
                input_details[f"{name.lower()}_tokens"] = sum(
                    detail_tokens(detail) for detail in prompt_details if modality(detail) == name
                )
        return {
            "input_tokens": input_tokens,
            "output_tokens": clock_output,
            "visible_output_tokens": chargeable_response,
            "total_tokens": input_tokens + clock_output,
            "input_token_details": input_details,
            "output_token_details": {
                "reasoning_tokens": reasoning,
                "audio_tokens": audio_response,
            },
            "native_response_tokens": native_response,
            "native_response_tokens_source": response_source,
            "native_audio_response_tokens": audio_response,
            "native_audio_response_tokens_reported": any(
                modality(detail) == "AUDIO" for detail in details
            ),
            "native_non_audio_response_tokens": chargeable_response,
            "native_total_tokens": (native_total if native_total_value is not None else None),
            "native_total_tokens_reported": native_total_value is not None,
            "native_tool_use_prompt_tokens": tool_prompt,
            "output_accounting": "gemini_native_tool_turn_usage_excludes_audio_v2",
            "native_output_tokens_source": "gemini_turn_usage_metadata",
            "native_audio_tokens_source": "gemini_response_tokens_details",
            "raw_native_usage": raw_usage,
            "native_response_usage_reported": (
                raw_usage.get("response_token_count") is not None
                or raw_usage.get("responseTokenCount") is not None
                or raw_usage.get("candidates_token_count") is not None
                or raw_usage.get("candidatesTokenCount") is not None
                or bool(raw_usage.get("response_tokens_details"))
                or bool(raw_usage.get("responseTokensDetails"))
            ),
            "native_response_usage_exact": native_response_exact,
            "response_tokens_details": response_details,
            "prompt_tokens_details": normalized_prompt_details,
        }

    @staticmethod
    def _text_parts(message) -> list[str]:
        server = getattr(message, "server_content", None)
        transcript = getattr(server, "output_transcription", None)
        text = getattr(transcript, "text", None)
        return [text] if text else []

    async def events(self, generation: GenerationHandle):
        """The reply's events. A streaming reply is read on the connection it started on,
        whichever is open now: one closed since ends it with its reader's failure."""
        try:
            if self.streaming:
                reply = self._replies[generation.generation_id]
                session, receiver = reply.session, reply.receiver

                def take():
                    return self._streamed(reply, receiver)

                def bound(operation):
                    # A reply read to its end is the server's whole turn, whatever the deadline.
                    return operation if reply.ended else self._before_go_away_deadline(operation)

            else:
                session, receiver = self._session, self._receiver
                if session is None:
                    raise RuntimeError("Gemini Live session is not connected")

                def take():
                    return self._before_go_away_deadline(receiver.next())

                bound = self._before_go_away_deadline
            try:
                async for event in self._events(generation, session, take, bound):
                    yield event
            except Exception:
                # The core applied each message as it was read; a reply that then failed here is
                # not a clean boundary to resume from.
                self._core.generation_failed(receiver.number)
                raise
        finally:
            self._replies.pop(generation.generation_id, None)

    async def _events(self, generation: GenerationHandle, session, take, bound):
        """The reply's events: ``take()`` gives its next message, ``bound(write)`` bounds a tool
        acknowledgement by the goAway deadline where it applies."""
        sequence = 0
        latest_usage = None
        tool_calls_seen = 0
        held, self._cancelled_between = self._cancelled_between, []
        for orphan in held:
            sequence += 1
            yield FunctionCallsCancelled(
                generation.generation_id,
                sequence,
                type(orphan.message).__name__,
                orphan.cancelled,
                received_at=orphan.at,
                connection=orphan.connection,
            )
        while True:
            received = await take()
            message = received.message
            numbers = {"connection": received.connection, "generation": received.generation}
            raw_type = type(message).__name__
            server = getattr(message, "server_content", None)
            turn = getattr(server, "model_turn", None)
            parts = getattr(turn, "parts", None) or []
            if parts and not generation.metadata.get("native_output_received"):
                # Native output may precede its transcription and terminal usage. The
                # zero-output transport retry must not discard audio or untranscribed text.
                generation.metadata["native_output_received"] = True
                sequence += 1
                yield GenerationBoundary(
                    received.generation_id,
                    sequence,
                    raw_type,
                    "started",
                    ProviderHistoryState.UNKNOWN,
                    {"native_output_received": True},
                    **numbers,
                )
            for part in parts:
                audio = getattr(part, "inline_data", None)
                if audio is not None and getattr(audio, "data", None):
                    generation.metadata["audio_bytes_received"] = generation.metadata.get(
                        "audio_bytes_received", 0
                    ) + len(audio.data)
            usage = self._usage_dict(getattr(message, "usage_metadata", None))
            if usage is not None:
                latest_usage = usage
                sequence += 1
                yield UsageSnapshot(
                    received.generation_id,
                    sequence,
                    raw_type,
                    usage,
                    **numbers,
                )
            if received.status is not None:
                generation.metadata["gemini_live_interaction_status"] = received.status
            if received.control:
                generation.metadata.update(received.control)
                sequence += 1
                yield GenerationBoundary(
                    received.generation_id,
                    sequence,
                    raw_type,
                    "go_away",
                    (
                        ProviderHistoryState.EXACT
                        if received.complete
                        else ProviderHistoryState.UNKNOWN
                    ),
                    received.control,
                    **numbers,
                )
            if received.cancelled:
                if not self.streaming:
                    raise RuntimeError("Gemini cancelled a queued GUI function call")
                sequence += 1
                yield FunctionCallsCancelled(
                    received.generation_id,
                    sequence,
                    raw_type,
                    received.cancelled,
                    received_at=received.at,
                    **numbers,
                )
            # A transcript delivered with a new call precedes that call's acknowledgement.
            # Keep it with the previous exchange (or initial model text), in provider order.
            for text in self._text_parts(message):
                refs = generation.metadata["completed_turn_refs"]
                exchanges = refs["native_tool_exchanges"]
                if exchanges:
                    exchange = exchanges[-1]
                    exchange["completion_text"] = exchange.get("completion_text", "") + text
                else:
                    refs["native_initial_text"] = refs.get("native_initial_text", "") + text
                sequence += 1
                yield TextDelta(
                    received.generation_id,
                    sequence,
                    raw_type,
                    text,
                    "visible",
                    **numbers,
                )
            calls = getattr(getattr(message, "tool_call", None), "function_calls", None) or []
            if calls:
                tool_calls_seen += len(calls)
                self._core.calls_received([(call.id, call.name) for call in calls])
                call_records = []
                for call in calls:
                    arguments = dict(call.args or {})
                    call_record = {
                        "id": call.id or "",
                        "name": call.name or "",
                        "arguments": arguments,
                    }
                    call_records.append(call_record)
                    sequence += 1
                    yield FunctionCall(
                        received.generation_id,
                        sequence,
                        raw_type,
                        call.id or "",
                        call.name or "",
                        arguments,
                        received_at=received.at if self.streaming else None,
                        **numbers,
                    )
                # Acknowledge every pending call as queued; the next observation carries its
                # execution result.
                acknowledged = self._core.acknowledging()
                function_responses = [
                    self.types.FunctionResponse(
                        id=call_id,
                        name=name,
                        response=dict(generation.opaque),
                        **self._core.tool_response_options,
                    )
                    for call_id, name in acknowledged
                ]
                # Finalize the queue request, not GUI execution. SILENT on a non-blocking tool
                # prevents its acknowledgement from triggering another model generation while
                # the current decision drains through authoritative terminal usage.
                try:
                    await bound(session.send_tool_response(function_responses=function_responses))
                except BaseException:
                    self._core.acknowledgement_failed(acknowledged, received.connection)
                    raise
                self._core.acknowledgement_sent(acknowledged, received.connection)
                completed_refs = generation.metadata["completed_turn_refs"]
                completed_refs["native_function_calls"].extend(call_records)
                completed_refs["native_tool_exchanges"].append(
                    {
                        "function_calls": call_records,
                        "function_responses": [
                            response.model_dump(mode="json", exclude_none=True)
                            for response in function_responses
                        ],
                    }
                )
                if tool_calls_seen == 1:
                    completed_refs["native_function_call"] = call_records[0]
                else:
                    completed_refs.pop("native_function_call", None)
                generation.metadata["function_calls_seen"] = tool_calls_seen
                generation.metadata["tool_response_policy"] = (
                    "queue_acknowledgement_before_execution"
                )
            if received.interrupted:
                sequence += 1
                yield GenerationBoundary(
                    received.generation_id,
                    sequence,
                    raw_type,
                    "interrupted",
                    ProviderHistoryState.CONTAMINATED,
                    **numbers,
                )
            if (
                server is not None
                and getattr(server, "generation_complete", False)
                and self.options["completion_signal"] == "turn_complete"
            ):
                sequence += 1
                yield GenerationBoundary(
                    received.generation_id,
                    sequence,
                    raw_type,
                    "generation_complete",
                    (
                        ProviderHistoryState.CONTAMINATED
                        if received.contaminated
                        else ProviderHistoryState.UNKNOWN
                    ),
                    **numbers,
                )
            if received.complete:
                generation.metadata["clean_turn_complete"] = not received.contaminated
                generation.metadata["function_calls_seen"] = tool_calls_seen
                if (
                    generation.metadata.get("audio_bytes_received", 0)
                    and latest_usage is not None
                    and not latest_usage["native_audio_response_tokens_reported"]
                ):
                    raise RuntimeError(
                        "Gemini returned audio without an AUDIO response token breakdown"
                    )
                sequence += 1
                yield GenerationBoundary(
                    received.generation_id,
                    sequence,
                    raw_type,
                    "turn_complete",
                    (
                        ProviderHistoryState.CONTAMINATED
                        if received.contaminated
                        else ProviderHistoryState.EXACT
                    ),
                    {
                        **generation.metadata,
                        "output_modality": "AUDIO",
                        "terminal_usage_received": usage is not None,
                        "action_text_source": "native_function_call",
                        "submission_boundary": (
                            "native_interaction_idle"
                            if self.options["completion_signal"] == "interaction_idle"
                            else "native_turn_complete"
                        ),
                    },
                    **numbers,
                )
            if received.ends:
                return

    async def interrupt(self, generation: GenerationHandle) -> InterruptResult:
        if self._session is None or self._activity_open:
            return InterruptResult(ProviderHistoryState.UNKNOWN)
        if self.streaming or self.options["input_transport"] == "client_content":
            # client_content does not opt into manual audio activity detection, and streaming's
            # NO_INTERRUPTION lets an ActivityStart cut nothing off. Closing the connection
            # cancels the incomplete turn without injecting an ActivityStart.
            await self._close_current()
            return InterruptResult(
                ProviderHistoryState.CONTAMINATED,
                {"interrupt_method": "connection_close"},
            )
        try:
            await self._session.send_realtime_input(activity_start=self.types.ActivityStart())
            self._activity_open = True
            self._core.interrupt_sent()
            return InterruptResult(ProviderHistoryState.CONTAMINATED)
        except Exception as exc:  # noqa: BLE001 - provider control boundary
            return InterruptResult(
                ProviderHistoryState.UNKNOWN,
                {"interrupt_error": repr(exc)[:300]},
            )

    async def _sync_history(self, request: HistorySyncRequest) -> HistorySyncResult:
        resume_error = None
        try:
            current = request.current_generation
            handle = self._core.resume_handle_for(
                request.reason,
                force_rebuild=request.force_rebuild,
                generation_clean=current is None
                or bool(current.metadata.get("clean_turn_complete")),
            )
            if handle:
                try:
                    await self._connect(
                        ConnectRequest(
                            seed=request.seed,
                            resume_handle=handle,
                        )
                    )
                    # Streaming resumes at the latest resumable handle, not at a clean boundary
                    # (SessionCore.resume_handle_for); the session takes the next reply all the
                    # same, and what the server keeps of its context is not known anyway.
                    resumed = (
                        "resumed Gemini Live session at its latest resumable handle"
                        if self.streaming
                        else "resumed clean Gemini Live session"
                    )
                    return HistorySyncResult(
                        SyncDisposition.EXACT,
                        ProviderHistoryState.EXACT,
                        detail=f"{resumed}: {request.reason}",
                        provider_metadata={
                            **self._connect_metadata,
                            "session_resumed": True,
                            "history_session_reconnected": True,
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - fall through to canonical replay
                    resume_error = repr(exc)[:300]
            # Renew and replay with the same task credential, including resume failure fallback.
            await self._connect(ConnectRequest(seed=request.seed, history=request.history))
            return HistorySyncResult(
                SyncDisposition.REBUILT,
                ProviderHistoryState.EXACT,
                detail=request.reason,
                provider_metadata={
                    **self._connect_metadata,
                    "history_session_reconnected": True,
                    **({"resume_error": resume_error} if resume_error else {}),
                },
            )
        except Exception as exc:
            # Preserve typed transport failures so the outer zero-output retry gate can safely
            # rebuild the session. Converting them to a string-only BROKEN result would discard the
            # WebSocket close code and make a later retry impossible to classify.
            if self.classify_transport_error(exc).retryable:
                raise
            return HistorySyncResult(
                SyncDisposition.BROKEN,
                ProviderHistoryState.UNKNOWN,
                detail=repr(exc)[:500],
            )
