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
    UsageSnapshot,
)
from realtime_framework.entity.model import (
    GenerationHandle,
    HistorySyncResult,
    RealtimeCapabilities,
    SyncDisposition,
)
from realtime_framework.entity.records import UNTRACED
from realtime_framework.harnesses.streaming import agent as agent_module
from realtime_framework.harnesses.streaming.agent import NativeStreamingAgent
from realtime_framework.providers.session_lifecycle import SessionLifecycle

LIVE = "gemini-3.8-live"  # configs/models/gemini-3.8-live.yaml; its policy row gemini_live
REALTIME = "gpt-realtime-2"  # configs/models/gpt-realtime-2.yaml; openai_realtime, actions in text
BUDGET = DecisionBudget(None, None, None, "wall")
EXACT, CONTAMINATED = ProviderHistoryState.EXACT, ProviderHistoryState.CONTAMINATED


@dataclass(frozen=True)
class Call:
    """A function call message ``at`` seconds into the reply, or, ``after_push``, at the instant
    that observation's push went out."""

    at: float
    call_id: str
    name: str
    arguments: dict
    after_push: str | None = None


@dataclass(frozen=True)
class Text:
    at: float
    text: str


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
    takes no input until a sync renewed it."""

    closed_fails_reply = True

    capabilities = RealtimeCapabilities(
        protocol_family="fake_live",
        persistent_session=True,
        action_channel="function_call",
        connection_renewal=True,
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
        self._pushes: dict[str, asyncio.Event] = {}  # set once that observation's push went out

    def bind_trace(self, trace) -> None:
        self._trace = trace

    async def _connect(self, request) -> None:
        # Opening a connection closes the one before, as both live drivers' _connect does.
        await self._close_current()
        await asyncio.sleep(self.connect_s)
        self._open()
        self.calls.append(("connect", _ids(request.history)))

    async def _sync_history(self, request) -> HistorySyncResult:
        await self._close_current()
        self._open()
        self.calls.append(("sync", request.reason, _ids(request.history)))
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

    def connection_renewal_reason(self) -> str | None:
        return "go_away" if self.renewal_required else None

    def _take_input(self) -> None:
        if self.renewal_required:
            raise RuntimeError("the session requires a renewal before input")

    async def push_frame(self, observation, history) -> dict:
        self.calls.append(("push", observation.observation_id, _ids(history)))
        self.views.append(history)
        self._take_input()
        if observation.observation_id in self.failing:
            raise RuntimeError("push failed")
        self._pushes.setdefault(observation.observation_id, asyncio.Event()).set()
        return {"observation_id": observation.observation_id}

    async def request_reply(self, request) -> GenerationHandle:
        self.calls.append(("reply", request.generation, _ids(request.canonical)))
        self._take_input()
        return GenerationHandle(request.generation_id, opaque=self.replies.pop(0))

    async def events(self, generation: GenerationHandle):
        reply, start = generation.opaque, self.clock.monotonic()
        gid = generation.generation_id
        for sequence, message in enumerate(reply.messages, 1):
            if not await self._until(start + message.at):
                return
            if getattr(message, "after_push", None) is not None:
                await self._pushes.setdefault(message.after_push, asyncio.Event()).wait()
            received = self.clock.monotonic()
            if isinstance(message, Fail):
                raise message.error
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
                yield FunctionCall(
                    gid,
                    sequence,
                    call_id=message.call_id,
                    name=message.name,
                    arguments=message.arguments,
                    received_at=received,
                )
            else:
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
    connection ends a reply's stream with no boundary."""

    closed_fails_reply = False
    capabilities = RealtimeCapabilities(
        protocol_family="fake_live_text",
        persistent_session=True,
        action_channel="text_json",
        feedback_placements=("separate_message",),
    )


def screenshot() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(stream, format="PNG")
    return stream.getvalue()


def observation(clock, n: int) -> Observation:
    return Observation(screenshot(), clock.time(), n, {"observation_id": f"obs-{n}"})


def make_agent(
    monkeypatch, clock, driver=FakeLiveDriver, model=LIVE, trace=UNTRACED
) -> NativeStreamingAgent:
    """An initialized agent on ``clock`` whose driver is ``driver``, on ``model``'s policy,
    writing its records to ``trace``."""
    monkeypatch.setattr(agent_module, "adapter_class", lambda adapter_type: driver)
    config = catalog.to_model_config(model, fps=1.0, reasoning_effort="fixed")
    agent = NativeStreamingAgent(config, trace=trace, clock=clock)
    agent.task_category = "realtime"
    agent.initialize("Click the button.", observation(clock, 0))
    return agent
