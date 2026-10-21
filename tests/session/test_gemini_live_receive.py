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

import pytest
from google.genai import errors as genai_errors
from google.genai import live as genai_live
from google.genai import types
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from realtime_framework.clock import SystemClock
from realtime_framework.entity.actions import Observation, gui_function_declarations
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.events import ProviderHistoryState
from realtime_framework.entity.model import (
    CanonicalTurn,
    ConnectRequest,
    GenerationRequest,
    HistorySyncRequest,
    ProviderObservation,
    SessionSeed,
    SyncDisposition,
)
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.request import build_request, system_block
from realtime_framework.providers.gemini import live
from realtime_framework.providers.gemini.session import Delivery

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
        self.fail = None  # a predicate on a client message whose send raises
        self.stall = None  # a predicate on a client message whose send never returns

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
        message = json.loads(message)
        if self.fail is not None and self.fail(message):
            raise server_closed(1011, "Internal error.")
        if self.stall is not None and self.stall(message):
            await asyncio.Event().wait()
        self.sent.append(message)


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

    def __init__(self, model: str, scripts: list[list[dict]]):
        self.log: list = []
        config = ModelConfig(
            adapter_type="gemini_live",
            model_name=model,
            api_key="replay-dummy-key",
            **live.GeminiLiveDriver.model_defaults.get(model, {}),
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
        self.view = lambda turns: build_request(
            policy,
            system_block(self.seed.system_instruction, None),
            tuple(turns),
            None,
            answer,
            call,
        )
        self.history = self.view(())

    async def connect(self) -> None:
        await self.driver.connect(ConnectRequest(seed=self.seed))

    async def start(self, generation_id: str):
        request = GenerationRequest(generation_id, None, "inf", "wall", canonical=self.history)
        return await self.driver.start_generation(request)


def frame_turn(k: int) -> CanonicalTurn:
    observation = ProviderObservation(
        observation=Observation(b"", float(k), k, {"observation_id": f"obs-{k}"}),
        prompt=f"frame {k}",
        image_bytes=b"png %d" % k,
        mime_type="image/png",
        image_detail="auto",
        geometry={},
    )
    return CanonicalTurn(observation, decision=k)


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


def test_receive_loop_stamps_connection_and_generation():
    """Two generations on one connection, one on the next: every event carries the connection
    it arrived on and the generation whose reply it belongs to. The bare turnComplete of a spoken
    filler ends google-genai's receive iterator; the loop reads on to IDLE."""
    rig = Rig(
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


def test_resume_waits_for_the_session_release():
    """A goAway renewal resumes on a new connection. Closing the old one and the server
    releasing its session are not atomic, so the resume's setup waits the release grace."""
    rig = Rig(
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


def test_a_failed_read_leaves_the_connection_unusable():
    """A read that fails ends the connection's receiver: the generation reading it raises the
    failure, and so does every later generation on that connection, with the same exception."""
    rig = Rig("gemini-3.8-live", [[server_closed(1011, "Internal error.")]])

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


def test_a_cancelled_read_leaves_the_connection_unusable():
    rig = Rig("gemini-3.8-live", [[HANG]])

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


def test_an_orphan_message_is_logged_and_skipped(caplog):
    """With no generation's reply in flight on the connection, a message is an orphan: the
    receiver logs it and reads on, here into the server's close."""
    rig = Rig("gemini-3.8-live", [[TURN_COMPLETE, server_closed(1011, "Internal error.")]])

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


def test_a_receive_that_ends_without_a_message_fails():
    """google-genai's receive() ends only after a message. An iterator that ends without one
    fails the read instead of being reopened forever, which would hold the event loop."""

    class EmptyReceive(genai_live.AsyncSession):
        async def receive(self):
            return
            yield

    rig = Rig("gemini-3.8-live", [[]])
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


def test_a_failed_reply_does_not_resume():
    """The core promotes the resumption handle when it reads the clean completion; the driver
    then rejects that message's usage. The failed reply is not a boundary to resume from, so the
    goAway renewal rebuilds instead."""
    bad_usage = {
        "promptTokenCount": 20,
        "responseTokenCount": 3,
        "totalTokenCount": 23,
        "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 5}],
    }
    rig = Rig(
        "gemini-3.8-live",
        [
            [
                {"sessionResumptionUpdate": {"newHandle": "handle-0", "resumable": True}},
                {"goAway": {"timeLeft": "30s"}},
                {"serverContent": {"turnComplete": True}, "usageMetadata": bad_usage},
            ],
            [],
        ],
    )

    async def run():
        await rig.connect()
        generation = await rig.start("turn-0")
        try:
            await drain(rig.driver.events(generation))
        except RuntimeError as exc:
            assert "AUDIO token detail exceeds" in str(exc)
        assert rig.driver.connection_renewal_reason() == "go_away"
        request = HistorySyncRequest(rig.seed, "go_away", rig.history)
        return await rig.driver.sync_history(request)

    result = asyncio.run(run())
    assert result.disposition == SyncDisposition.REBUILT
    assert [entry for entry in rig.log if entry[0] == "connect"] == [
        ("connect", None),
        ("connect", None),
    ]


def realtime_input_with(key: str):
    return lambda message: key in message.get("realtime_input", {})


@pytest.mark.parametrize(
    ("model", "failing"),
    [
        ("gemini-3.8-live", None),
        ("gemini-3.8-live", lambda message: "client_content" in message),
        ("replay-gemini-live", None),  # the driver defaults: realtime_input
        ("replay-gemini-live", realtime_input_with("activityStart")),
        ("replay-gemini-live", realtime_input_with("video")),
    ],
    ids=["client-content", "client-content-fails", "realtime", "start-fails", "video-fails"],
)
def test_observation_delivery(model, failing):
    """An observation's blocks count as sent once handed to the connection, and as unknown when
    a send of the observation fails, on either input transport."""
    rig = Rig(model, [[]])
    rig.live.sockets[0].fail = failing
    history = rig.view([frame_turn(0)])

    async def run():
        await rig.connect()
        try:
            await rig.driver.send_observation(None, history)
        except genai_errors.APIError:
            assert failing is not None
        except ConnectionClosedError:
            assert failing is not None

    asyncio.run(run())
    expected = Delivery.SENT if failing is None else Delivery.UNKNOWN
    assert rig.driver._core.delivery[("block", "d0.frame")] == expected


TRANSCRIPT = {"serverContent": {"outputTranscription": {"text": "partial"}}}


def test_a_reply_read_after_reconnecting_keeps_its_connection_and_generation():
    """The second reply on connection 1 is read halfway when the driver reconnects and starts a
    generation on connection 2. Reading on from the old events() still reads connection 1, and
    numbers what it reads as that connection's second generation."""
    rig = Rig(
        "gemini-3.8-live",
        [[TURN_COMPLETE, TRANSCRIPT, TURN_COMPLETE], [TURN_COMPLETE]],
    )

    async def run():
        await rig.connect()
        replies = [stamps(await drain(rig.driver.events(await rig.start("turn-0"))))]
        old = rig.driver.events(await rig.start("turn-1"))
        replies.append(stamps([await anext(old)]))
        await rig.connect()
        new = rig.driver.events(await rig.start("turn-2"))
        replies.append(stamps(await drain(old)))
        replies.append(stamps(await drain(new)))
        return replies

    turn_0, turn_1_before, turn_1_after, turn_2 = asyncio.run(run())
    assert turn_0 == [("UsageSnapshot", 1, 1, "turn-0"), ("GenerationBoundary", 1, 1, "turn-0")]
    assert turn_1_before == [("TextDelta", 1, 2, "turn-1")]
    assert turn_1_after == [
        ("UsageSnapshot", 1, 2, "turn-1"),
        ("GenerationBoundary", 1, 2, "turn-1"),
    ]
    assert turn_2 == [("UsageSnapshot", 2, 1, "turn-2"), ("GenerationBoundary", 2, 1, "turn-2")]


def test_a_message_is_numbered_by_the_generation_in_flight_when_read():
    """The receiver numbers a message when it reads it, by the generation then in flight on the
    connection, whichever events() asked: here the first reply's consumer reads on after the next
    generation has started."""
    rig = Rig("gemini-3.8-live", [[TRANSCRIPT, TURN_COMPLETE]])

    async def run():
        await rig.connect()
        first = rig.driver.events(await rig.start("turn-0"))
        before = stamps([await anext(first)])
        await rig.start("turn-1")
        return before, stamps(await drain(first))

    before, after = asyncio.run(run())
    assert before == [("TextDelta", 1, 1, "turn-0")]
    assert after == [("UsageSnapshot", 1, 2, "turn-1"), ("GenerationBoundary", 1, 2, "turn-1")]


def test_idle_ends_a_turn_complete_reply():
    """With turn_complete as the completion signal, an IDLE status still ends the reply, without
    a clean turn_complete boundary; events() returns without reading further."""
    rig = Rig("gemini-3.8-live", [[IDLE]])

    async def run():
        await rig.connect()
        return await drain(rig.driver.events(await rig.start("turn-0")))

    assert stamps(asyncio.run(run())) == [("UsageSnapshot", 1, 1, "turn-0")]


GO_AWAY_SOON = {"goAway": {"timeLeft": "2.1s"}}  # the client closes 2 s early: 0.1 s from now
CALL = {"toolCall": {"functionCalls": [{"id": "call-0", "name": "wait", "args": {}}]}}


@pytest.mark.parametrize(
    ("script", "stall"),
    [
        ([GO_AWAY_SOON, HANG], None),
        ([GO_AWAY_SOON, CALL], lambda message: "tool_response" in message),
    ],
    ids=["read", "acknowledgement"],
)
def test_the_go_away_deadline_bounds_the_reply(script, stall):
    """After a goAway the server closes the connection; a read or a tool acknowledgement still
    pending at the deadline fails the reply, and the driver closes the connection."""
    rig = Rig("gemini-3.8-live", [script])
    rig.live.sockets[0].stall = stall

    async def run():
        await rig.connect()
        try:
            await drain(rig.driver.events(await rig.start("turn-0")))
        except live.GoAwayDeadlineExceeded as exc:
            return exc

    assert isinstance(asyncio.run(asyncio.wait_for(run(), 5)), live.GoAwayDeadlineExceeded)
    assert rig.driver._session is None and rig.driver._core.connection is None


def test_an_interrupted_reply_is_contaminated():
    rig = Rig(
        "gemini-3.8-live",
        [
            [
                {"serverContent": {"interrupted": True}},
                {"serverContent": {"generationComplete": True}},
                TURN_COMPLETE,
            ]
        ],
    )

    async def run():
        await rig.connect()
        return await drain(rig.driver.events(await rig.start("turn-0")))

    events = asyncio.run(run())
    boundaries = [(event.kind, event.history_state) for event in events if hasattr(event, "kind")]
    contaminated = ProviderHistoryState.CONTAMINATED
    assert boundaries == [
        ("interrupted", contaminated),
        ("generation_complete", contaminated),
        ("turn_complete", contaminated),
    ]
    assert events[-1].metadata["clean_turn_complete"] is False


def test_a_cancelled_tool_call_fails_the_reply():
    rig = Rig("gemini-3.8-live", [[{"toolCallCancellation": {"ids": ["call-0"]}}]])

    async def run():
        await rig.connect()
        try:
            await drain(rig.driver.events(await rig.start("turn-0")))
        except RuntimeError as exc:
            return exc

    assert str(asyncio.run(run())) == "Gemini cancelled a queued GUI function call"


GO_AWAY_AFTER_A_CLEAN_TURN = [
    {"sessionResumptionUpdate": {"newHandle": "handle-0", "resumable": True}},
    {"goAway": {"timeLeft": "30s"}},
    TURN_COMPLETE,
]


def test_input_waits_for_the_renewal_a_go_away_asks_for():
    rig = Rig("gemini-3.8-live", [GO_AWAY_AFTER_A_CLEAN_TURN])

    async def run():
        await rig.connect()
        await drain(rig.driver.events(await rig.start("turn-0")))
        try:
            await rig.driver.send_observation(None, rig.view([frame_turn(1)]))
        except RuntimeError as exc:
            return exc

    failure = asyncio.run(run())
    assert str(failure) == "Gemini Live requires canonical history synchronization before input"
    assert rig.live.sockets[0].sent == []  # nothing of the observation was written


def test_an_interrupted_session_is_not_resumed():
    """On the realtime_input transport an interrupt opens a new activity on the same connection;
    the session it leaves has no clean boundary, so the goAway renewal rebuilds."""
    rig = Rig("replay-gemini-live", [GO_AWAY_AFTER_A_CLEAN_TURN, []])

    async def run():
        await rig.connect()
        generation = await rig.start("turn-0")
        await drain(rig.driver.events(generation))
        await rig.driver.interrupt(generation)
        request = HistorySyncRequest(rig.seed, "go_away", rig.history, generation)
        return await rig.driver.sync_history(request)

    assert asyncio.run(run()).disposition == SyncDisposition.REBUILT
    assert [entry for entry in rig.log if entry[0] == "connect"] == [
        ("connect", None),
        ("connect", None),
    ]
