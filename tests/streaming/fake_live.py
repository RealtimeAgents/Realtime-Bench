"""A fake live encoder for the streaming agent's tests: the real SessionLifecycle around
connection operations that write down what they did, with the view each one was given."""

from __future__ import annotations

import io
from typing import ClassVar

from PIL import Image

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.actions import Observation
from realtime_framework.entity.events import ProviderHistoryState
from realtime_framework.entity.model import (
    HistorySyncResult,
    RealtimeCapabilities,
    SyncDisposition,
)
from realtime_framework.entity.records import UNTRACED
from realtime_framework.harnesses.streaming import agent as agent_module
from realtime_framework.harnesses.streaming.agent import NativeStreamingAgent
from realtime_framework.providers.session_lifecycle import SessionLifecycle

MODEL_KEY = "gemini-3.8-live"  # configs/models/gemini-3.8-live.yaml; its policy row gemini_live


def _ids(history) -> tuple[str, ...]:
    return tuple(item.block.id for item in history.items)


class FakeLiveDriver(SessionLifecycle):
    """``calls`` holds each connection operation and push as (name, ...), with the ids of the
    view it was given, and ``views`` each push's view. A push of an observation id in
    ``failing`` raises."""

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
        self.open = False

    def bind_trace(self, trace) -> None:
        self._trace = trace

    async def _connect(self, request) -> None:
        self.open = True
        self.calls.append(("connect", _ids(request.history)))

    async def _sync_history(self, request) -> HistorySyncResult:
        self.open = True
        self.calls.append(("sync", request.reason, _ids(request.history)))
        return HistorySyncResult(SyncDisposition.REBUILT, ProviderHistoryState.EXACT)

    async def _close_current(self) -> None:
        if self.open:
            self.open = False
            self.calls.append(("close",))

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


def screenshot() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(stream, format="PNG")
    return stream.getvalue()


def observation(clock, n: int) -> Observation:
    return Observation(screenshot(), clock.time(), n, {"observation_id": f"obs-{n}"})


def make_agent(monkeypatch, clock, driver=FakeLiveDriver) -> NativeStreamingAgent:
    """An initialized agent on ``clock`` whose driver is ``driver``, on the Gemini Live policy."""
    monkeypatch.setattr(agent_module, "adapter_class", lambda adapter_type: driver)
    config = catalog.to_model_config(MODEL_KEY, fps=1.0, reasoning_effort="fixed")
    agent = NativeStreamingAgent(config, trace=UNTRACED, clock=clock)
    agent.task_category = "realtime"
    agent.initialize("Click the button.", observation(clock, 0))
    return agent
