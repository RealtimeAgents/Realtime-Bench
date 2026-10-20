"""The Gemini Live session core (providers/gemini/session.py) and the driver's receive loop,
without a WebSocket.

The server messages are google-genai's LiveServerMessage, built from the wire JSON the replay stub
sends (tests/replay/ws_gemini_live.py); the history is the Contents GeminiLiveDriver renders.
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
from realtime_framework.entity.context_policy import phase1_context_policy
from realtime_framework.entity.model import ConnectRequest, GenerationRequest, SessionSeed
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.request import build_request, system_block
from realtime_framework.providers.gemini import live
from realtime_framework.providers.gemini.session import (
    ACKNOWLEDGED,
    SENT,
    UNKNOWN,
    SessionCore,
)

GO_AWAY_RECONNECT = {
    "reconnect_required": True,
    "reconnect_reason": "go_away",
    "go_away_time_left": "30s",
}


def message(wire: dict) -> types.LiveServerMessage:
    return types.LiveServerMessage.model_validate(wire)


def update(handle: str | None) -> types.LiveServerMessage:
    if handle is None:
        return message({"sessionResumptionUpdate": {"resumable": False}})
    return message({"sessionResumptionUpdate": {"newHandle": handle, "resumable": True}})


def go_away(time_left: str = "30s") -> types.LiveServerMessage:
    return message({"goAway": {"timeLeft": time_left}})


def server_content(**fields) -> types.LiveServerMessage:
    return message({"serverContent": fields})


def exchange(k: int) -> list[types.Content]:
    """One retained turn of a native_tools history: the frame, the call and its receipt."""
    call = types.FunctionCall(id=f"call-{k}", name="click", args={"target": "P1"})
    receipt = types.FunctionResponse(id=f"call-{k}", name="click", response={"status": "queued"})
    return [
        types.Content(role="user", parts=[types.Part(text=f"frame {k}")]),
        types.Content(role="model", parts=[types.Part(function_call=call)]),
        types.Content(role="user", parts=[types.Part(function_response=receipt)]),
    ]


def connected(core: SessionCore, history=(), seeded=(), resume_handle=None):
    plan = core.connect(list(history), tuple(seeded), resume_handle)
    core.connected(plan)
    return plan


def receive(core: SessionCore, message, now: float = 0.0):
    """A message on the open connection."""
    return core.received(core.connection, message, now)


def start_turn(core: SessionCore, generation_id: str) -> None:
    key = ("block", f"frame-{generation_id}")
    core.input_started([key])
    core.sent([key])
    core.generation_started(generation_id)


def clean_turn(core: SessionCore, generation_id: str, handle: str, *messages) -> None:
    """One observation and its reply: a resumption update, ``messages``, then turnComplete."""
    start_turn(core, generation_id)
    for received in (update(handle), *messages, server_content(turnComplete=True)):
        receive(core, received)


def test_new_session_then_rebuild():
    core = SessionCore("turn_complete")
    first = connected(core)
    assert (first.mode, first.release_grace, first.seed) == ("new_session", False, ())
    assert core.connection == 1
    assert core.delivery == {("setup",): ACKNOWLEDGED}
    assert not core.last_turn_clean  # nothing on the session yet: no handle to resume

    history = [content for k in range(10) for content in exchange(k)]
    seeded = tuple(f"block-{k}" for k in range(10))
    rebuild = core.connect(history, seeded, None)
    assert core.connection is None  # the open connection is closed first
    assert (rebuild.mode, rebuild.release_grace) == ("canonical_history_replay", True)
    # Twelve contents per message at most, and never one exchange across two messages.
    assert [len(batch) for batch in rebuild.seed] == [12, 12, 6]
    assert [content for batch in rebuild.seed for content in batch] == history
    assert all(batch[0].parts[0].text for batch in rebuild.seed)

    core.connected(rebuild)
    assert core.connection == 2
    assert core.delivery == {
        ("setup",): ACKNOWLEDGED,
        **{("block", block_id): SENT for block_id in seeded},
    }
    assert core.last_turn_clean and not core.renewal_required


def test_observation_and_acknowledgement_delivery():
    core = SessionCore("turn_complete")
    connected(core)
    keys = [("block", "frame-0"), ("block", "feedback-0")]
    core.input_started(keys)
    assert all(core.delivery[key] == UNKNOWN for key in keys)
    core.sent(keys)
    assert all(core.delivery[key] == SENT for key in keys)

    core.generation_started("turn-0")
    core.calls_received([("call-0", "click")])
    core.calls_received([("call-1", "key")])
    # One queued acknowledgement answers every call still pending.
    assert core.acknowledging() == (("call-0", "click"), ("call-1", "key"))
    assert core.delivery[("tool_response", "call-1")] == UNKNOWN
    core.acknowledgement_sent()
    assert core.delivery[("tool_response", "call-0")] == SENT
    assert core.delivery[("tool_response", "call-1")] == SENT
    assert core.acknowledging() == ()


def test_go_away_resumes_after_a_clean_turn():
    core = SessionCore("turn_complete")
    connected(core)
    start_turn(core, "turn-0")

    received = receive(core, update("handle-0"), now=100.0)
    assert (received.connection, received.generation, received.generation_id) == (1, 1, "turn-0")
    assert not received.complete and received.control == {}
    # A handle captured mid-generation is promoted only at a clean boundary.
    assert core.resume_handle is None

    received = receive(core, go_away("30s"), now=100.0)
    assert received.control == GO_AWAY_RECONNECT
    assert core.renewal_required
    # The server closes the connection after timeLeft; the client keeps a 2-second margin.
    assert core.go_away_deadline == 128.0
    receive(core, go_away("5s"), now=110.0)
    assert core.go_away_deadline == 113.0  # the earliest deadline holds
    receive(core, go_away("soon"), now=111.0)
    assert core.go_away_deadline == 111.0  # a malformed timeLeft closes at once

    received = receive(core, server_content(turnComplete=True), now=112.0)
    assert received.complete and received.ends and not received.contaminated
    assert core.resume_handle == "handle-0" and core.last_turn_clean
    assert core.replies == {}

    assert core.resume_handle_for("go_away", force_rebuild=False, generation_clean=True) == (
        "handle-0"
    )
    assert core.resume_handle_for("go_away", force_rebuild=True, generation_clean=True) is None
    assert core.resume_handle_for("go_away", force_rebuild=False, generation_clean=False) is None
    assert (
        core.resume_handle_for(
            "history_block_compaction", force_rebuild=False, generation_clean=True
        )
        is None
    )

    resume = core.connect([], (), "handle-0")
    assert (resume.mode, resume.resume_handle, resume.seed, resume.release_grace) == (
        "session_resumption",
        "handle-0",
        (),
        True,
    )
    assert core.go_away_deadline is None  # it belonged to the closed connection
    core.connected(resume)
    assert core.connection == 2
    assert core.last_turn_clean and not core.renewal_required


def test_resume_failure_rebuilds():
    core = SessionCore("turn_complete")
    connected(core)
    clean_turn(core, "turn-0", "handle-0", go_away())

    core.connect([], (), "handle-0")
    # The resume was refused: connected() never comes. The renewal is still owed.
    assert core.connection is None and core.renewal_required
    assert core.resume_handle == "handle-0"

    rebuild = core.connect(exchange(0), ("frame-0", "tools-0"), None)
    # No open connection is left to replace, but a replayed history still waits for the release.
    assert rebuild.release_grace and rebuild.mode == "canonical_history_replay"
    assert core.resume_handle is None
    core.connected(rebuild)
    assert core.connection == 2  # the refused resume never counted as a connection
    assert core.last_turn_clean and not core.renewal_required


def test_interrupted_or_unresumable_turn_does_not_resume():
    core = SessionCore("turn_complete")
    connected(core)
    clean_turn(core, "turn-0", "handle-0")
    assert core.resume_handle == "handle-0"

    start_turn(core, "turn-1")
    assert core.resume_handle is None and not core.last_turn_clean
    assert receive(core, update("handle-1")).generation == 2  # the second on this connection
    received = receive(core, server_content(interrupted=True))
    assert received.interrupted and received.contaminated and not received.ends
    received = receive(core, server_content(turnComplete=True))
    assert received.complete and received.contaminated and not received.interrupted
    assert core.resume_handle is None and not core.last_turn_clean
    assert core.resume_handle_for("go_away", force_rebuild=False, generation_clean=True) is None

    start_turn(core, "turn-2")
    receive(core, update("handle-2"))
    receive(core, update(None))  # not resumable any more
    receive(core, server_content(turnComplete=True))
    assert core.resume_handle is None and core.last_turn_clean


def test_extended_thinking_spans_intermediate_turn_complete():
    core = SessionCore("interaction_idle")
    connected(core)
    start_turn(core, "turn-0")
    receive(core, update("handle-0"))

    # Spoken fillers end with turnComplete, with or without an IN_PROGRESS status.
    for wire in (
        {"turnComplete": True},
        {"turnComplete": True, "interactionStatus": "IN_PROGRESS"},
    ):
        received = receive(core, server_content(**wire))
        assert not received.complete and not received.ends
        assert (received.generation, received.generation_id) == (1, "turn-0")
    assert received.status == "IN_PROGRESS"
    assert core.resume_handle is None and not core.last_turn_clean

    received = receive(core, server_content(interactionStatus="IDLE"))
    assert received.status == "IDLE" and received.complete and received.ends
    assert core.resume_handle == "handle-0" and core.last_turn_clean
    assert core.replies == {}


def test_idle_ends_a_turn_complete_interaction_uncleanly():
    core = SessionCore("turn_complete")
    connected(core)
    start_turn(core, "turn-0")
    receive(core, update("handle-0"))
    received = receive(core, server_content(interactionStatus="IDLE"))
    assert received.ends and not received.complete
    assert core.resume_handle is None and not core.last_turn_clean


def test_reply_read_after_close_keeps_its_generation():
    """An abort closes the connection while its reply is still being read. What its reader
    drains keeps that connection and generation, even once the next connection has started a
    generation of its own."""
    core = SessionCore("turn_complete")
    connected(core)
    start_turn(core, "turn-0")
    core.calls_received([("call-0", "click")])
    receive(core, go_away())
    core.closed()
    assert core.connection is None and core.delivery == {}
    assert core.acknowledging() == () and core.go_away_deadline is None
    assert core.renewal_required  # the session state stays until the next connection

    connected(core, exchange(0), ("frame-0",))
    assert core.connection == 2 and not core.renewal_required
    start_turn(core, "turn-1")

    stale = core.received(1, server_content(turnComplete=True), 0.0)
    assert (stale.connection, stale.generation, stale.generation_id) == (1, 1, "turn-0")
    assert stale.complete and stale.ends
    current = receive(core, server_content(turnComplete=True))
    assert (current.connection, current.generation, current.generation_id) == (2, 1, "turn-1")
    assert current.complete and core.replies == {}


# ---- the driver's receive loop ---------------------------------------------------------------


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
    policy = phase1_context_policy().for_path(config.adapter_type, config.provider_options)
    call = CallParameters(model, config.reasoning_effort, None)
    canonical = build_request(policy, system_block(seed.system_instruction), (), None, answer, call)

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
