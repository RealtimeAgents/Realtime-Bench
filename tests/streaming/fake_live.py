"""A fake live encoder for the streaming agent's tests: the real SessionLifecycle around
connection operations that write down what they did, with the view each one was given, and
replies that follow a script on the clock."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import ClassVar

from PIL import Image

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.actions import Observation
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.entity.events import (
    FunctionCall,
    FunctionCallsCancelled,
    GenerationBoundary,
    ProviderError,
    ProviderHistoryState,
    TextDelta,
    TextDone,
    UsageSnapshot,
)
from realtime_framework.entity.model import (
    FATAL,
    TRANSIENT,
    GenerationHandle,
    HistorySyncResult,
    PushReceipt,
    RealtimeCapabilities,
    SyncDisposition,
    TransportFailure,
)
from realtime_framework.entity.records import UNTRACED
from realtime_framework.harnesses.default.prompts import PromptOptions
from realtime_framework.harnesses.streaming import agent as agent_module
from realtime_framework.harnesses.streaming.agent import NativeStreamingAgent
from realtime_framework.providers.session_lifecycle import RenewalRequired, SessionLifecycle

LIVE = "gemini-3.8-live"  # configs/models/gemini-3.8-live.yaml; its policy row gemini_live
REALTIME = "gpt-realtime-2"  # configs/models/gpt-realtime-2.yaml; openai_realtime, actions in text
BUDGET = DecisionBudget(None, None, None, "wall")
# The prompt options of a streaming run, which runs on the wall clock only.
STREAMING = PromptOptions("streaming", "wall", False)
EXACT, CONTAMINATED = ProviderHistoryState.EXACT, ProviderHistoryState.CONTAMINATED
EXACT_SYNC = SyncDisposition.EXACT


@dataclass(frozen=True)
class Call:
    """A function call message ``at`` seconds into the reply, or, ``after_push``, at the instant
    that observation's push went out; handed over ``late`` seconds after it came off the wire,
    as a reader's queue may."""

    at: float
    call_id: str
    name: str
    arguments: dict
    after_push: str | None = None
    late: float = 0.0


@dataclass(frozen=True)
class Text:
    at: float
    text: str


@dataclass(frozen=True)
class TextEnd:
    """The output text is complete ``at`` seconds into the reply, as OpenAI Realtime's
    response.output_text.done says."""

    at: float


@dataclass(frozen=True)
class Cancel:
    """The provider cancels ``call_ids``, of this reply or an earlier one."""

    at: float
    call_ids: tuple[str, ...]


@dataclass(frozen=True)
class Fail:
    """The reply's stream raises ``error`` ``at`` seconds into it."""

    at: float
    error: Exception


@dataclass(frozen=True)
class Error:
    """A ProviderError ``at`` seconds into the reply; a non-fatal one, as a cancel race is,
    leaves it going."""

    at: float
    detail: str
    fatal: bool = True


@dataclass(frozen=True)
class Hang:
    """From ``at`` seconds into the reply nothing comes, and closing the connection does not end
    it either: only its cancellation, as stopping the execution thread cancels it, does."""

    at: float


@dataclass(frozen=True)
class GoAway:
    """The server asks for a new connection ``at`` seconds into the reply: from then on the
    session takes no input until it is renewed."""

    at: float


@dataclass(frozen=True)
class Reply:
    """A reply's messages, in order, and when it ends, seconds after it started, with its usage
    and the boundaries it ends with: a turn_complete, exact unless an interrupted came before it;
    none when the stream just stops, as a connection that closed normally ends it."""

    messages: tuple = ()
    ends_at: float = 0.0
    usage: dict | None = None
    boundaries: tuple[str, ...] = ("turn_complete",)


def _ids(history) -> tuple[str, ...]:
    return tuple(item.block.id for item in history.items)


class FakeLiveDriver(SessionLifecycle):
    """``calls`` holds each connection operation, push and reply as (name, ...), with the ids
    of the view it was given, and ``views`` each push's view. A push of an observation id in
    ``failing`` raises. Each reply takes the next of ``replies``; closing its connection fails
    it, as Gemini Live's reader does. Once ``renewal_required``, as after a goAway, the session
    takes no input until a sync renewed it. send_observation and start_generation are stubs: the
    streaming agent pushes frames and asks for replies instead."""

    closed_fails_reply = True

    capabilities = RealtimeCapabilities(
        protocol_family="fake_live",
        action_channel="function_call",
        feedback_placements=("separate_message",),
    )
    streaming_settings: ClassVar[dict] = {"retained_frames": 2}

    def __init__(self, config, *, clock=None, streaming=False):
        self.config, self.clock, self.streaming = config, clock, streaming
        self.calls: list[tuple] = []
        self.views: list = []
        self.failing: set[str] = set()
        self.replies: list[Reply] = []
        self.renewal_required = False
        self.open = False
        self.connect_s = 0.0  # how long opening a connection takes
        self.connect_failures: list[Exception] = []  # what the next connects raise, in turn
        self.connect_attempts = 0
        # The next push or reply ("push", "reply") of each listed reads a goAway before it
        # writes, as one waiting for the session's lock may (Gemini Live's reader).
        self.go_away_on: list[str] = []
        self.resumes = False  # a goAway renewal resumes the session instead of rebuilding it
        self._pushes: dict[str, asyncio.Event] = {}  # set once that observation's push went out

    def bind_trace(self, trace) -> None:
        self._trace = trace

    async def _connect(self, request) -> None:
        # Opening a connection closes the one before, as both live drivers' _connect does.
        await self._close_current()
        await asyncio.sleep(self.connect_s)
        self.connect_attempts += 1
        if self.connect_failures:
            raise self.connect_failures.pop(0)
        self._open()
        self.calls.append(("connect", _ids(request.history)))

    async def _sync_history(self, request) -> HistorySyncResult:
        await self._close_current()
        self._open()
        self.calls.append(("sync", request.reason, _ids(request.history)))
        if request.reason == "go_away" and self.resumes:
            # As Gemini Live resumes a goAway renewal with its latest handle.
            metadata = {"session_resumed": True, "history_session_reconnected": True}
            return HistorySyncResult(EXACT_SYNC, EXACT, provider_metadata=metadata)
        return HistorySyncResult(SyncDisposition.REBUILT, ProviderHistoryState.EXACT)

    def _open(self) -> None:
        self.open, self._closed, self.renewal_required = True, asyncio.Event(), False

    async def _close_current(self) -> None:
        if self.open:
            self.open = False
            self._closed.set()
            self.calls.append(("close",))

    async def _until(self, at: float) -> bool:
        """Wait until ``at`` on the clock; False once the connection is closed, which fails the
        reply where ``closed_fails_reply``."""
        try:
            await asyncio.wait_for(self._closed.wait(), max(0.0, at - self.clock.monotonic()))
        except TimeoutError:
            return True
        if self.closed_fails_reply:
            raise ConnectionError("connection closed")
        return False

    async def interrupt(self, generation):
        self.calls.append(("interrupt", generation.generation_id))

    def classify_transport_error(self, exc: BaseException) -> TransportFailure:
        """A dropped connection is transient, as both live drivers take one; anything else
        fatal."""
        return TransportFailure(TRANSIENT if isinstance(exc, ConnectionError) else FATAL)

    def connection_renewal_reason(self) -> str | None:
        # Read on the execution thread, where the reader applies a goAway: read anywhere else,
        # whether a goAway came first depends on how the threads run.
        asyncio.get_running_loop()
        return "go_away" if self.renewal_required else None

    def _take_input(self, operation: str) -> None:
        if operation in self.go_away_on:
            self.go_away_on.remove(operation)
            self.renewal_required = True
        if self.renewal_required:
            raise RenewalRequired("the session requires a renewal before input")

    async def push_frame(self, observation, history) -> PushReceipt:
        self.calls.append(("push", observation.observation_id, _ids(history)))
        self.views.append(history)
        self._take_input("push")
        if observation.observation_id in self.failing:
            raise RuntimeError("push failed")
        self._pushes.setdefault(observation.observation_id, asyncio.Event()).set()
        return await self._receipt()

    async def _receipt(self) -> PushReceipt:
        """Written now, and confirmed by no server, as Gemini Live's push."""
        return PushReceipt(written_at=self.clock.monotonic())

    async def send_observation(self, observation, history):
        raise NotImplementedError

    async def start_generation(self, request):
        raise NotImplementedError

    async def request_reply(self, request) -> GenerationHandle:
        self.calls.append(("reply", request.generation, _ids(request.canonical)))
        self._take_input("reply")
        return GenerationHandle(request.generation_id, opaque=self.replies.pop(0))

    async def events(self, generation: GenerationHandle):
        reply, start = generation.opaque, self.clock.monotonic()
        gid = generation.generation_id
        text = ""
        for sequence, message in enumerate(reply.messages, 1):
            if not await self._until(start + message.at):
                return
            if getattr(message, "after_push", None) is not None:
                await self._pushes.setdefault(message.after_push, asyncio.Event()).wait()
            received = self.clock.monotonic()
            if isinstance(message, Fail):
                raise message.error
            if isinstance(message, Hang):
                await asyncio.Event().wait()
            if isinstance(message, GoAway):
                self.calls.append(("go_away", generation.generation_id))
                self.renewal_required = True
            elif isinstance(message, Error):
                yield ProviderError(gid, sequence, detail=message.detail, fatal=message.fatal)
            elif isinstance(message, Cancel):
                yield FunctionCallsCancelled(
                    gid, sequence, call_ids=message.call_ids, received_at=received
                )
            elif isinstance(message, Call):
                if message.late and not await self._until(received + message.late):
                    return
                yield FunctionCall(
                    gid,
                    sequence,
                    call_id=message.call_id,
                    name=message.name,
                    arguments=message.arguments,
                    received_at=received,
                )
            elif isinstance(message, TextEnd):
                yield TextDone(gid, sequence, text=text, received_at=received)
            else:
                text += message.text
                yield TextDelta(gid, sequence, text=message.text)
        if not await self._until(start + reply.ends_at):
            return
        if reply.usage is not None:
            yield UsageSnapshot(gid, 0, usage=reply.usage)
        for kind in reply.boundaries:
            if kind == "interrupted":
                yield GenerationBoundary(gid, 0, kind=kind, history_state=CONTAMINATED)
            else:
                state = CONTAMINATED if "interrupted" in reply.boundaries else EXACT
                metadata = {"terminal_usage_received": reply.usage is not None}
                yield GenerationBoundary(gid, 0, kind=kind, history_state=state, metadata=metadata)


class TextLiveDriver(FakeLiveDriver):
    """A fake live encoder whose model answers in text, as OpenAI Realtime's does: a closed
    connection ends a reply's stream with no boundary, and the server confirms each pushed
    frame, ``confirm_s`` after it was written, before the push returns."""

    closed_fails_reply = False
    confirm_s = 0.0

    async def _receipt(self) -> PushReceipt:
        written = self.clock.monotonic()
        await asyncio.sleep(self.confirm_s)
        return PushReceipt(written_at=written, confirmed_at=self.clock.monotonic())

    capabilities = RealtimeCapabilities(
        protocol_family="fake_live_text",
        action_channel="text_json",
        feedback_placements=("separate_message",),
    )


class RecordingToolRuntime:
    """A tool runtime whose inspections find no element, and which writes each finished one
    to ``rows`` as (observation id, outcome, raw action output, latency), once, as the SoM
    runtime does (sandbox/observation/som.py ToolRuntime.finish), and its reference resolution
    to ``resolutions`` by observation id."""

    def __init__(self):
        self.rows: list[tuple] = []
        self.resolutions: dict[str, object] = {}
        self._pending: set[str] = set()

    def inspect(self, observation, *, requested_observation_id, **sizes) -> dict:
        call_id = f"inspection-{len(self._pending) + len(self.rows)}"
        self._pending.add(call_id)
        proof = {
            "valid": True,
            "call_id": call_id,
            "observation_id": requested_observation_id,
            "requested_observation_id": requested_observation_id,
        }
        return {"text": "", "proof": proof, "marks": {}}

    def finish(self, execution, *, action_latency_s=0.0, raw_action_output="", outcome=""):
        call_id = execution["proof"]["call_id"]
        if call_id in self._pending:
            self._pending.remove(call_id)
            observation_id = execution["proof"]["observation_id"]
            self.rows.append((observation_id, outcome, raw_action_output, action_latency_s))
            self.resolutions[observation_id] = execution.get("reference_resolution")

    def close(self) -> None:
        pass


def screenshot() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(stream, format="PNG")
    return stream.getvalue()


def observation(clock, n: int) -> Observation:
    return Observation(screenshot(), clock.time(), n, {"observation_id": f"obs-{n}"})


class RecordingTrace:
    """An episode trace that keeps each capture as (event, payload)."""

    attempt_id = "recording"

    def __init__(self):
        self.captured: list[tuple] = []

    def capture(self, event, payload, *, body=None, classify=None):
        self.captured.append((event, payload))

    def of(self, event) -> list[dict]:
        return [payload for captured, payload in self.captured if captured == event]


def make_agent(
    monkeypatch, clock, driver=FakeLiveDriver, model=LIVE, log_dir=None, trace=UNTRACED
) -> NativeStreamingAgent:
    """An initialized agent on ``clock`` whose driver is ``driver``, on ``model``'s policy,
    writing its trajectory to ``log_dir`` if given and its records to ``trace``. It is not
    configured (harnesses/registry.py configure_agent): its prompts have no scenario prompt and
    no clock note."""
    monkeypatch.setattr(agent_module, "adapter_class", lambda adapter_type: driver)
    config = catalog.to_model_config(model, fps=1.0, reasoning_effort="fixed")
    agent = NativeStreamingAgent(config, log_dir, trace=trace, clock=clock)
    agent.bind_tool_runtime(RecordingToolRuntime())
    agent.task_category = "realtime"
    agent.prompt_options = STREAMING
    agent.initialize("Click the button.", observation(clock, 0))
    return agent
