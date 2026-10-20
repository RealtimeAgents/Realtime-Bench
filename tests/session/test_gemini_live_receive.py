"""The Gemini Live driver's receive loop, on google-genai's AsyncSession over a scripted socket.

FakeSocket stands in for the WebSocket under the SDK's session (its constructor is the SDK's own,
not a public API), so the driver reads server messages through the SDK's real receive().
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from types import SimpleNamespace

from google.genai import errors as genai_errors
from google.genai import live as genai_live
from google.genai import types
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from realtime_framework.clock import SystemClock
from realtime_framework.entity.actions import gui_function_declarations
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationRequest,
    HistorySyncRequest,
    SessionSeed,
    SyncDisposition,
)
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.request import build_request, system_block
from realtime_framework.providers.gemini import live

HANG = object()  # script step: the server sends nothing more


class FakeSocket:
    """What google-genai's AsyncSession reads and writes: scripted server frames, in order. A
    step can also be an exception the read raises, or HANG."""

    def __init__(self, frames: list):
        self.frames = [
            frame if frame is HANG or isinstance(frame, BaseException) else json.dumps(frame)
            for frame in frames
        ]
        self.sent: list[dict] = []

    async def recv(self, decode=None):
        assert self.frames, "the driver read past the script"
        frame = self.frames[0]
        if frame is HANG:
            await asyncio.Event().wait()
        self.frames.pop(0)
        if isinstance(frame, BaseException):
            raise frame
        return frame

    async def send(self, message):
        self.sent.append(json.loads(message))


class FakeLive:
    """client.aio.live: each connect opens a session on the next socket, and is logged with the
    handle its setup resumes."""

    def __init__(self, scripts: list[list[dict]], log: list):
        self.sockets = [FakeSocket(frames) for frames in scripts]
        self._next = iter(self.sockets)
        self._log = log
        self.session_type = genai_live.AsyncSession

    def connect(self, *, model, config):
        self._log.append(("connect", config.session_resumption.handle))
        session = self.session_type(
            api_client=SimpleNamespace(vertexai=False),
            websocket=next(self._next),
            setup_complete=types.LiveServerSetupComplete(),
        )
        return contextlib.asynccontextmanager(lambda: _yield(session))()


async def _yield(value):
    yield value


class RecordingClock(SystemClock):
    """Real time, except that the driver's sleeps are logged instead of taken."""

    def __init__(self, log: list):
        self._log = log

    async def sleep_async(self, seconds: float) -> None:
        self._log.append(("sleep", seconds))
        await asyncio.sleep(0)


class Rig:
    """A GeminiLiveDriver for ``model`` on scripted connections; ``log`` holds its connects and
    sleeps in order."""

    def __init__(self, monkeypatch, model: str, scripts: list[list[dict]]):
        self.log: list = []
        config = ModelConfig(
            adapter_type="gemini_live",
            model_name=model,
            api_key="replay-dummy-key",
            **live.GeminiLiveDriver.model_defaults[model],
        )
        self.driver = live.GeminiLiveDriver(config, clock=RecordingClock(self.log))
        self.live = FakeLive(scripts, self.log)
        self.driver.client = SimpleNamespace(aio=SimpleNamespace(live=self.live))
        answer = AnswerRequirements(
            "function_call", "standard_vlm", tuple(gui_function_declarations())
        )
        self.seed = SessionSeed("system instruction", model, None, 1.0, answer)
        policy = current_context_policy().for_path(config.adapter_type, config.provider_options)
        call = CallParameters(model, config.reasoning_effort, None)
        self.history = build_request(
            policy, system_block(self.seed.system_instruction, None), (), None, answer, call
        )

    async def connect(self) -> None:
        await self.driver.connect(ConnectRequest(seed=self.seed))

    async def start(self, generation_id: str):
        request = GenerationRequest(generation_id, None, "inf", "wall", canonical=self.history)
        return await self.driver.start_generation(request)


def stamps(events) -> list[tuple]:
    return [
        (type(event).__name__, event.connection, event.generation, event.generation_id)
        for event in events
    ]


async def drain(events) -> list:
    return [event async for event in events]


USAGE = {"usageMetadata": {"promptTokenCount": 20, "responseTokenCount": 3, "totalTokenCount": 23}}
IDLE = {"serverContent": {"interactionStatus": "IDLE"}, **USAGE}
TURN_COMPLETE = {"serverContent": {"turnComplete": True}, **USAGE}


def test_receive_loop_stamps_connection_and_generation(monkeypatch):
    """Two generations on one connection, one on the next: every event carries the connection
    it arrived on and the generation whose reply it belongs to. The bare turnComplete of a spoken
    filler ends google-genai's receive iterator; the loop reads on to IDLE."""
    rig = Rig(
        monkeypatch,
        "gemini-3.8-live-extended-thinking",
        [
            [
                {"toolCall": {"functionCalls": [{"id": "call-0", "name": "wait", "args": {}}]}},
                {"serverContent": {"turnComplete": True}},
                IDLE,
                IDLE,
            ],
            [IDLE],
        ],
    )

    async def run() -> list[list[tuple]]:
        replies = []
        await rig.connect()
        for k in range(3):
            if k == 2:
                await rig.connect()
            generation = await rig.start(f"turn-{k}")
            replies.append(stamps(await drain(rig.driver.events(generation))))
        await rig.driver.close()
        return replies

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
    first, second = rig.live.sockets
    assert first.frames == [] and second.frames == []
    assert [next(iter(message)) for message in first.sent] == ["tool_response"]
    assert second.sent == []


def test_resume_waits_for_the_session_release(monkeypatch):
    """A goAway renewal resumes on a new connection. Closing the old one and the server
    releasing its session are not atomic, so the resume's setup waits the release grace."""
    rig = Rig(
        monkeypatch,
        "gemini-3.8-live",
        [
            [
                {"sessionResumptionUpdate": {"newHandle": "handle-0", "resumable": True}},
                {"goAway": {"timeLeft": "30s"}},
                TURN_COMPLETE,
            ],
            [],
        ],
    )

    async def run():
        await rig.connect()
        generation = await rig.start("turn-0")
        await drain(rig.driver.events(generation))
        request = HistorySyncRequest(rig.seed, "go_away", rig.history, generation)
        return await rig.driver.sync_history(request)

    result = asyncio.run(run())
    assert (result.disposition, result.detail) == (
        SyncDisposition.EXACT,
        "resumed clean Gemini Live session: go_away",
    )
    assert result.provider_metadata["session_replacement_grace_s"] == live._SESSION_RELEASE_GRACE_S
    assert rig.log == [
        ("connect", None),
        ("sleep", live._SESSION_RELEASE_GRACE_S),
        ("connect", "handle-0"),
    ]


def server_closed(code: int, reason: str) -> ConnectionClosedError:
    return ConnectionClosedError(Close(code, reason), None)


def test_a_failed_read_leaves_the_connection_unusable(monkeypatch):
    """A read that fails ends the connection's receiver: the generation reading it raises the
    failure, and so does every later generation on that connection, with the same exception."""
    rig = Rig(monkeypatch, "gemini-3.8-live", [[server_closed(1011, "Internal error.")]])

    async def run():
        failures = []
        await rig.connect()
        for k in range(2):
            generation = await rig.start(f"turn-{k}")
            try:
                await drain(rig.driver.events(generation))
            except Exception as exc:  # noqa: BLE001 - the failure under test
                failures.append(exc)
        return failures

    first, second = asyncio.run(run())
    assert isinstance(first, genai_errors.APIError) and first.code == 1011
    assert second is first
    assert rig.driver._core.replies == {}  # neither reply is waited for any more
    # The failure keeps the close code the transport classification reads.
    assert rig.driver.classify_transport_error(second).retryable


def test_a_cancelled_read_leaves_the_connection_unusable(monkeypatch):
    rig = Rig(monkeypatch, "gemini-3.8-live", [[HANG]])

    async def run():
        await rig.connect()
        generation = await rig.start("turn-0")
        reading = asyncio.ensure_future(drain(rig.driver.events(generation)))
        await asyncio.sleep(0)
        reading.cancel()
        try:
            await reading
        except asyncio.CancelledError:
            pass
        generation = await rig.start("turn-1")
        try:
            await drain(rig.driver.events(generation))
        except RuntimeError as exc:
            return exc

    failure = asyncio.run(run())
    assert str(failure) == "Gemini Live connection 1: a read was cancelled"
    assert isinstance(failure.__cause__, asyncio.CancelledError)
    assert rig.driver._core.replies == {}


def test_an_orphan_message_is_logged_and_skipped(monkeypatch, caplog):
    """With no generation's reply in flight on the connection, a message is an orphan: the
    receiver logs it and reads on, here into the server's close."""
    rig = Rig(
        monkeypatch, "gemini-3.8-live", [[TURN_COMPLETE, server_closed(1011, "Internal error.")]]
    )

    async def run():
        await rig.connect()
        events = rig.driver.events(await rig.start("turn-0"))
        rig.driver._core.replies.clear()  # as if that reply had ended before this read
        try:
            await drain(events)
        except genai_errors.APIError as exc:
            return exc

    with caplog.at_level(logging.DEBUG, logger=live.__name__):
        failure = asyncio.run(run())
    assert failure.code == 1011
    assert "Gemini Live connection 1: no generation takes LiveServerMessage" in caplog.messages


def test_a_receive_that_ends_without_a_message_fails(monkeypatch):
    """google-genai's receive() ends only after a message. An iterator that ends without one
    fails the read instead of being reopened forever, which would hold the event loop."""

    class EmptyReceive(genai_live.AsyncSession):
        async def receive(self):
            return
            yield

    rig = Rig(monkeypatch, "gemini-3.8-live", [[]])
    rig.live.session_type = EmptyReceive
    outcome = []

    async def run():
        await rig.connect()
        try:
            await drain(rig.driver.events(await rig.start("turn-0")))
        except RuntimeError as exc:
            outcome.append(exc)

    thread = threading.Thread(target=asyncio.run, args=(run(),), daemon=True)
    thread.start()
    thread.join(10)
    assert not thread.is_alive(), "the receiver kept reopening an empty receive()"
    (failure,) = outcome
    assert str(failure) == "Gemini Live connection 1: receive() ended without a message"
