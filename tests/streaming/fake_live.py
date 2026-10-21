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
    GenerationBoundary,
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


@dataclass(frozen=True)
class Call:
    """A function call message ``at`` seconds into the reply."""

    at: float
    call_id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Text:
    at: float
    text: str


@dataclass(frozen=True)
class Fail:
    """The reply's stream raises ``error`` ``at`` seconds into it."""

    at: float
    error: Exception


@dataclass(frozen=True)
class Reply:
    """A reply's messages, in order, and when it ends, seconds after it started. Closing the
    connection fails it."""

    messages: tuple = ()
    ends_at: float = 0.0
    usage: dict | None = None


def _ids(history) -> tuple[str, ...]:
    return tuple(item.block.id for item in history.items)


class FakeLiveDriver(SessionLifecycle):
    """``calls`` holds each connection operation, push and reply as (name, ...), with the ids
    of the view it was given, and ``views`` each push's view. A push of an observation id in
    ``failing`` raises. Each reply takes the next of ``replies``."""

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
        self.open = False

    def bind_trace(self, trace) -> None:
        self._trace = trace

    async def _connect(self, request) -> None:
        self._open()
        self.calls.append(("connect", _ids(request.history)))

    async def _sync_history(self, request) -> HistorySyncResult:
        self._open()
        self.calls.append(("sync", request.reason, _ids(request.history)))
        return HistorySyncResult(SyncDisposition.REBUILT, ProviderHistoryState.EXACT)

    def _open(self) -> None:
        self.open, self._closed = True, asyncio.Event()

    async def _close_current(self) -> None:
        if self.open:
            self.open = False
            self._closed.set()
            self.calls.append(("close",))

    async def _until(self, at: float) -> None:
        """Wait until ``at`` on the clock; raise once the connection is closed."""
        try:
            await asyncio.wait_for(self._closed.wait(), max(0.0, at - self.clock.monotonic()))
        except TimeoutError:
            return
        raise ConnectionError("connection closed")

    async def interrupt(self, generation):
        self.calls.append(("interrupt", generation.generation_id))

    def connection_renewal_reason(self) -> str | None:
        return None

    async def push_frame(self, observation, history) -> dict:
        self.calls.append(("push", observation.observation_id, _ids(history)))
        self.views.append(history)
        if observation.observation_id in self.failing:
            raise RuntimeError("push failed")
        return {"observation_id": observation.observation_id}

    async def request_reply(self, request) -> GenerationHandle:
        self.calls.append(("reply", request.generation, _ids(request.canonical)))
        return GenerationHandle(request.generation_id, opaque=self.replies.pop(0))

    async def events(self, generation: GenerationHandle):
        reply, start = generation.opaque, self.clock.monotonic()
        for sequence, message in enumerate(reply.messages, 1):
            await self._until(start + message.at)
            received = self.clock.monotonic()
            gid = generation.generation_id
            if isinstance(message, Fail):
                raise message.error
            if isinstance(message, Call):
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
        await self._until(start + reply.ends_at)
        if reply.usage is not None:
            yield UsageSnapshot(generation.generation_id, 0, usage=reply.usage)
        yield GenerationBoundary(generation.generation_id, 0, kind="turn_complete")


class TextLiveDriver(FakeLiveDriver):
    """A fake live encoder whose model answers in text, as OpenAI Realtime's does."""

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


def make_agent(monkeypatch, clock, driver=FakeLiveDriver, model=LIVE) -> NativeStreamingAgent:
    """An initialized agent on ``clock`` whose driver is ``driver``, on ``model``'s policy."""
    monkeypatch.setattr(agent_module, "adapter_class", lambda adapter_type: driver)
    config = catalog.to_model_config(model, fps=1.0, reasoning_effort="fixed")
    agent = NativeStreamingAgent(config, trace=UNTRACED, clock=clock)
    agent.task_category = "realtime"
    agent.initialize("Click the button.", observation(clock, 0))
    return agent
