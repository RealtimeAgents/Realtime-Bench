"""The Gemini Live driver's receive loop, on google-genai's AsyncSession over a scripted socket.

FakeSocket stands in for the WebSocket under the SDK's session (its constructor is the SDK's own,
not a public API), so the driver reads server messages through the SDK's real receive().
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace

from google.genai import live as genai_live
from google.genai import types

from realtime_framework.entity.actions import gui_function_declarations
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.model import ConnectRequest, GenerationRequest, SessionSeed
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.request import build_request, system_block
from realtime_framework.providers.gemini import live


class FakeSocket:
    """What google-genai's AsyncSession reads and writes: scripted server frames, in order."""

    def __init__(self, frames: list[dict]):
        self.frames = [json.dumps(frame) for frame in frames]
        self.sent: list[dict] = []

    async def recv(self, decode=None):
        assert self.frames, "the driver read past the script"
        return self.frames.pop(0)

    async def send(self, message):
        self.sent.append(json.loads(message))


class FakeLive:
    """client.aio.live: each connect opens a session on the next socket."""

    def __init__(self, scripts: list[list[dict]]):
        self.sockets = [FakeSocket(frames) for frames in scripts]
        self._next = iter(self.sockets)

    def connect(self, *, model, config):
        session = genai_live.AsyncSession(
            api_client=SimpleNamespace(vertexai=False),
            websocket=next(self._next),
            setup_complete=types.LiveServerSetupComplete(),
        )
        return contextlib.asynccontextmanager(lambda: _yield(session))()


async def _yield(value):
    yield value


USAGE = {"usageMetadata": {"promptTokenCount": 20, "responseTokenCount": 3, "totalTokenCount": 23}}
IDLE = {"serverContent": {"interactionStatus": "IDLE"}, **USAGE}


def test_receive_loop_stamps_connection_and_generation(monkeypatch):
    """Two generations on one connection, one on the next: every event carries the connection
    it arrived on and the generation whose reply it belongs to. The bare turnComplete of a spoken
    filler ends google-genai's receive iterator; the loop reads on to IDLE."""
    monkeypatch.setattr(live, "_SESSION_RELEASE_GRACE_S", 0.0)
    model = "gemini-3.8-live-extended-thinking"
    config = ModelConfig(
        adapter_type="gemini_live",
        model_name=model,
        api_key="replay-dummy-key",
        **live.GeminiLiveDriver.model_defaults[model],
    )
    driver = live.GeminiLiveDriver(config)
    connections = FakeLive(
        [
            [
                {"toolCall": {"functionCalls": [{"id": "call-0", "name": "wait", "args": {}}]}},
                {"serverContent": {"turnComplete": True}},
                IDLE,
                IDLE,
            ],
            [IDLE],
        ]
    )
    driver.client = SimpleNamespace(aio=SimpleNamespace(live=connections))
    answer = AnswerRequirements("function_call", "standard_vlm", tuple(gui_function_declarations()))
    seed = SessionSeed("system instruction", model, None, 1.0, answer)
    policy = current_context_policy().for_path(config.adapter_type, config.provider_options)
    call = CallParameters(model, config.reasoning_effort, None)
    system = system_block(seed.system_instruction, None)
    canonical = build_request(policy, system, (), None, answer, call)

    async def run() -> list[list[tuple]]:
        stamps = []
        await driver.connect(ConnectRequest(seed=seed))
        for k in range(3):
            if k == 2:
                await driver.connect(ConnectRequest(seed=seed))
            request = GenerationRequest(f"turn-{k}", None, "inf", "wall", canonical=canonical)
            generation = await driver.start_generation(request)
            stamps.append(
                [
                    (type(event).__name__, event.connection, event.generation, event.generation_id)
                    async for event in driver.events(generation)
                ]
            )
        await driver.close()
        return stamps

    turn_0, turn_1, turn_2 = asyncio.run(run())
    assert turn_0 == [
        ("FunctionCall", 1, 1, "turn-0"),
        ("UsageSnapshot", 1, 1, "turn-0"),
        ("GenerationBoundary", 1, 1, "turn-0"),
    ]
    assert turn_1 == [
        ("UsageSnapshot", 1, 2, "turn-1"),
        ("GenerationBoundary", 1, 2, "turn-1"),
    ]
    assert turn_2 == [
        ("UsageSnapshot", 2, 1, "turn-2"),
        ("GenerationBoundary", 2, 1, "turn-2"),
    ]
    first, second = connections.sockets
    assert first.frames == [] and second.frames == []
    assert [next(iter(message)) for message in first.sent] == ["tool_response"]
    assert second.sent == []
