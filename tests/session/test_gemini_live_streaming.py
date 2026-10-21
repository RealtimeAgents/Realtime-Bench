"""The Gemini Live driver in native streaming (providers/gemini/live.py, ``streaming=True``), on
google-genai's AsyncSession over a socket the test feeds.

A streaming connection is read as messages arrive, so the scripted socket of
tests/session/test_gemini_live_receive.py, which hands out its whole script at once, would turn a
reply into orphans before its generation started. FedSocket hands out only what the test fed it,
when it fed it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace

from google.genai import errors as genai_errors
from google.genai import live as genai_live
from google.genai import types

from realtime_framework.entity.actions import gui_function_declarations
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.events import (
    FunctionCall,
    FunctionCallsCancelled,
    GenerationBoundary,
)
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationRequest,
    HistorySyncRequest,
    SessionSeed,
    SyncDisposition,
)
from realtime_framework.entity.records import UNTRACED
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.request import system_block
from realtime_framework.harnesses.streaming.history import (
    feedback_block,
    history_view,
    output_block,
)
from realtime_framework.providers.gemini import live
from realtime_framework.providers.gemini.session import Delivery
from tests.session.test_gemini_live_receive import RecordingClock, server_closed
from tests.session.test_gemini_live_session import pushed

MODEL = (
    "gemini-3.8-live"  # the client_content preset: streaming goes on realtime_input all the same
)


class FedSocket:
    """What google-genai's AsyncSession reads and writes: the server messages the test feeds, in
    order, each once fed; an exception fed is raised by the read that takes it."""

    def __init__(self, number: int):
        self.trace_connection_id = f"gemini_live.ws-{number}"
        self.sent: list[dict] = []
        self._frames: asyncio.Queue = asyncio.Queue()
        self._answers: list[tuple] = []  # (kind, frames), in the order they are due

    def feed(self, *frames) -> None:
        for frame in frames:
            self._frames.put_nowait(
                frame if isinstance(frame, BaseException) else json.dumps(frame)
            )

    def answer(self, kind, *frames) -> None:
        """Feed ``frames`` as soon as the client sends a message that is ``kind`` (wire()), as a
        server answers it, before the send returns."""
        self._answers.append((kind, frames))

    async def recv(self, decode=None):
        frame = await self._frames.get()
        if isinstance(frame, BaseException):
            raise frame
        return frame

    async def send(self, message):
        message = json.loads(message)
        self.sent.append(message)
        if self._answers and wire(message) == self._answers[0][0]:
            self.feed(*self._answers.pop(0)[1])
        # A write gives the loop its turns, as a real socket's does: a reader then takes what
        # came meanwhile.
        await settle()


class FedLive:
    """client.aio.live: each connect opens a session on a new FedSocket; ``configs`` holds each
    connect's setup."""

    def __init__(self):
        self.sockets: list[FedSocket] = []
        self.configs: list = []

    def connect(self, *, model, config):
        self.configs.append(config)
        socket = FedSocket(len(self.sockets) + 1)
        self.sockets.append(socket)
        session = genai_live.AsyncSession(
            api_client=SimpleNamespace(vertexai=False),
            websocket=socket,
            setup_complete=types.LiveServerSetupComplete(),
        )
        return contextlib.asynccontextmanager(lambda: _yield(session))()


async def _yield(value):
    yield value


class Rig:
    """A GeminiLiveDriver for ``model`` on fed connections, in native streaming unless
    ``streaming`` is False; ``log`` holds its waits."""

    def __init__(self, model: str = MODEL, *, streaming: bool = True):
        self.log: list = []
        config = ModelConfig(
            adapter_type="gemini_live",
            model_name=model,
            api_key="replay-dummy-key",
            **live.GeminiLiveDriver.model_defaults.get(model, {}),
        )
        self.driver = live.GeminiLiveDriver(
            config, clock=RecordingClock(self.log), streaming=streaming
        )
        self.driver.bind_trace(UNTRACED)
        self.live = FedLive()
        self.driver.client = SimpleNamespace(aio=SimpleNamespace(live=self.live))
        self.answer = AnswerRequirements(
            "function_call", "standard_vlm", tuple(gui_function_declarations())
        )
        self.seed = SessionSeed("system instruction", model, None, 1.0, self.answer)
        self.policy = current_context_policy().for_path(
            config.adapter_type, config.provider_options, config.context_preset
        )
        self.call = CallParameters(model, config.reasoning_effort, None)

    def view(self, blocks, *, frames: int = 2):
        """The streaming history view of the accepted ``blocks``."""
        return history_view(
            blocks,
            policy=self.policy,
            system=system_block(self.seed.system_instruction, None),
            retained_frames=frames,
            answer=self.answer,
            call=self.call,
        )

    @property
    def socket(self) -> FedSocket:
        return self.live.sockets[-1]

    async def connect(self, history=None) -> None:
        await self.driver.connect(ConnectRequest(seed=self.seed, history=history))

    async def start(self, generation_id: str, history=None):
        """A generation registered on the session, without its trigger."""
        canonical = history if history is not None else self.view([])
        request = GenerationRequest(generation_id, None, "inf", "wall", canonical=canonical)
        return await self.driver.start_generation(request)


async def settle() -> None:
    """Let the reader take everything fed so far."""
    for _ in range(20):
        await asyncio.sleep(0)


async def drain(events) -> list:
    """Every event of a reply; one that never ends fails within a few seconds."""

    async def read():
        return [event async for event in events]

    return await asyncio.wait_for(read(), 5)


def update(handle: str | None) -> dict:
    if handle is None:
        return {"sessionResumptionUpdate": {"resumable": False}}
    return {"sessionResumptionUpdate": {"newHandle": handle, "resumable": True}}


GO_AWAY = {"goAway": {"timeLeft": "30s"}}
USAGE = {"usageMetadata": {"promptTokenCount": 20, "responseTokenCount": 3, "totalTokenCount": 23}}
TURN_COMPLETE = {"serverContent": {"turnComplete": True}, **USAGE}


def call(call_id: str, name: str = "wait") -> dict:
    return {"toolCall": {"functionCalls": [{"id": call_id, "name": name, "args": {}}]}}


def test_a_streaming_setup_compresses_the_context_and_lets_a_trigger_not_interrupt():
    """A streaming session's setup turns on the server's sliding-window compression at the
    driver's thresholds, takes realtime input whatever transport the model's preset names, and
    lets a new activity leave a reply in progress alone; the connection records the thresholds,
    and that what the server keeps of the context is unknown. A turn-based setup has neither."""
    rig, turn_based = Rig(), Rig(streaming=False)

    async def run():
        await rig.connect()
        await turn_based.connect()
        await rig.driver.close()
        await turn_based.driver.close()

    asyncio.run(run())
    (config,) = rig.live.configs
    settings = live.GeminiLiveDriver.streaming_settings
    compression = config.context_window_compression
    assert compression.trigger_tokens == settings["compression_trigger_tokens"]
    assert compression.sliding_window.target_tokens == settings["compression_target_tokens"]
    realtime = config.realtime_input_config
    assert realtime.automatic_activity_detection.disabled
    assert realtime.activity_handling == types.ActivityHandling.NO_INTERRUPTION
    assert realtime.turn_coverage == types.TurnCoverage.TURN_INCLUDES_ALL_INPUT
    assert config.session_resumption.transparent
    metadata = rig.driver._connect_metadata
    assert metadata["gemini_live_context_window_compression"] == settings
    assert metadata["gemini_live_server_context"] == "unknown"

    (turn_based_config,) = turn_based.live.configs
    assert turn_based_config.context_window_compression is None
    assert turn_based_config.realtime_input_config is None  # the preset's client_content
    assert not turn_based_config.session_resumption.transparent
    assert "gemini_live_context_window_compression" not in turn_based.driver._connect_metadata


def test_a_go_away_and_a_handle_with_no_reply_in_flight_renew_by_resuming_the_latest():
    """Between replies the reader still takes the session's control: a goAway asks for a new
    connection, and the renewal resumes with the latest handle the server called resumable,
    though no reply came to a clean end."""
    rig = Rig()

    async def run():
        await rig.connect()
        rig.socket.feed(update("handle-1"), update(None), GO_AWAY)
        await settle()
        reason = rig.driver.connection_renewal_reason()
        request = HistorySyncRequest(rig.seed, "go_away", rig.view([]))
        result = await rig.driver.sync_history(request)
        await rig.driver.close()
        return reason, result

    reason, result = asyncio.run(run())
    assert reason == "go_away"
    assert result.disposition == SyncDisposition.EXACT
    assert [config.session_resumption.handle for config in rig.live.configs] == [None, "handle-1"]


def test_a_reply_s_messages_reach_its_events_as_they_arrive():
    """The reader takes a reply's messages as they come, whether its events() is read yet or
    not, and hands them to it in order: its text, its call, acknowledged as queued, and its end,
    each with the connection and the generation."""
    rig = Rig()

    async def run():
        await rig.connect()
        generation = await rig.start("turn-0")
        rig.socket.feed(
            {"serverContent": {"outputTranscription": {"text": "on it"}}},
            call("call-0"),
            TURN_COMPLETE,
        )
        await settle()
        read_ahead = rig.socket._frames.empty()
        received = await drain(rig.driver.events(generation))
        await rig.driver.close()
        return read_ahead, received

    read_ahead, received = asyncio.run(run())
    assert read_ahead
    assert [(type(e).__name__, e.connection, e.generation, e.generation_id) for e in received] == [
        ("TextDelta", 1, 1, "turn-0"),
        ("FunctionCall", 1, 1, "turn-0"),
        ("UsageSnapshot", 1, 1, "turn-0"),
        ("GenerationBoundary", 1, 1, "turn-0"),
    ]
    assert [next(iter(message)) for message in rig.socket.sent] == ["tool_response"]


def test_a_call_carries_the_time_its_message_came_off_the_wire():
    """A call's arrival is when the reader read its message, on the driver's monotonic clock,
    not when its events() got to it."""
    rig = Rig()

    async def run():
        await rig.connect()
        generation = await rig.start("turn-0")
        before = rig.driver.clock.monotonic()
        rig.socket.feed(call("call-0"))
        await settle()
        read = rig.driver.clock.monotonic()
        await asyncio.sleep(0.05)
        rig.socket.feed(TURN_COMPLETE)
        events = await drain(rig.driver.events(generation))
        await rig.driver.close()
        return before, read, events

    before, read, events = asyncio.run(run())
    (function_call,) = [event for event in events if isinstance(event, FunctionCall)]
    assert before <= function_call.received_at <= read


def cancelled(*call_ids: str) -> dict:
    return {"toolCallCancellation": {"ids": list(call_ids)}}


def test_a_cancellation_is_an_event_and_the_reply_goes_on():
    """A cancellation is handed over as an event naming the calls by id, in the server's
    order: here one of an earlier reply, likely executed by now, and one of this reply, not yet;
    which is which is the kernel's to tell. The reply goes on to its end."""
    rig = Rig()

    async def run():
        await rig.connect()
        first = await rig.start("turn-0")
        rig.socket.feed(call("call-0"), TURN_COMPLETE)
        await drain(rig.driver.events(first))
        second = await rig.start("turn-1")
        rig.socket.feed(call("call-1"), cancelled("call-1", "call-0"), TURN_COMPLETE)
        events = await drain(rig.driver.events(second))
        await rig.driver.close()
        return events

    events = asyncio.run(run())
    (cancellation,) = [e for e in events if isinstance(e, FunctionCallsCancelled)]
    assert cancellation.call_ids == ("call-1", "call-0")
    assert (cancellation.generation_id, cancellation.connection, cancellation.generation) == (
        "turn-1",
        1,
        2,
    )
    assert cancellation.received_at is not None
    assert isinstance(events[-1], GenerationBoundary) and events[-1].kind == "turn_complete"


def test_a_cancellation_between_replies_comes_with_the_next_reply():
    """A cancellation that no reply takes, read between two, is handed over first thing in the
    next reply's events, which it belongs to by id only."""
    rig = Rig()

    async def run():
        await rig.connect()
        first = await rig.start("turn-0")
        rig.socket.feed(call("call-0"), TURN_COMPLETE)
        await drain(rig.driver.events(first))
        rig.socket.feed(cancelled("call-0"))
        await settle()
        second = await rig.start("turn-1")
        rig.socket.feed(TURN_COMPLETE)
        events = await drain(rig.driver.events(second))
        await rig.driver.close()
        return events

    first_event, *_ = asyncio.run(run())
    assert isinstance(first_event, FunctionCallsCancelled)
    assert (first_event.call_ids, first_event.generation_id) == (("call-0",), "turn-1")
    assert (first_event.connection, first_event.generation) == (1, None)


def test_a_failed_read_ends_every_reply_on_the_connection():
    """A read that fails ends the reader: the reply waiting raises the failure, and so does
    every later reply on that connection."""
    rig = Rig()

    async def run():
        failures = []
        await rig.connect()
        rig.socket.feed(server_closed(1011, "Internal error."))
        for k in range(2):
            generation = await rig.start(f"turn-{k}")
            try:
                await drain(rig.driver.events(generation))
            except genai_errors.APIError as exc:
                failures.append(exc)
        await rig.driver.close()
        return failures

    first, second = asyncio.run(run())
    assert first.code == 1011 and second is first


def test_a_pushed_frame_goes_on_the_video_stream_then_its_text_and_starts_no_reply():
    """A frame goes out as a video frame, then its text, with no activity around it and no wait
    between: its block is sent once both are written. After a goAway nothing is pushed until the
    session is renewed."""
    rig = Rig()

    async def run():
        await rig.connect()
        pushed_frame = await rig.driver.push_frame(None, rig.view([pushed(1)]))
        delivery = rig.driver._core.delivery[("block", "f1.frame")]
        rig.socket.feed(GO_AWAY)
        await settle()
        try:
            await rig.driver.push_frame(None, rig.view([pushed(1), pushed(2)]))
        except RuntimeError as exc:
            refused = str(exc)
        await rig.driver.close()
        return pushed_frame, delivery, refused

    pushed_frame, delivery, refused = asyncio.run(run())
    assert pushed_frame == {"observation_id": "obs-1"}
    video, text = rig.socket.sent
    assert list(video["realtime_input"]) == ["video"]
    assert text == {"realtime_input": {"text": "frame 1"}}
    assert rig.log == []  # no wait
    assert delivery == Delivery.SENT
    assert refused == "Gemini Live requires canonical history synchronization before input"


def wire(message: dict):
    """A client message as what it is: the realtime input's one field (a text with its text),
    or the message's kind."""
    [(kind, body)] = message.items()
    if kind != "realtime_input":
        return kind
    [(field, value)] = body.items()
    return ("text", value) if field == "text" else field


ACTIVITY = ["activityStart", "audio", "activityEnd"]


def test_a_reply_is_one_activity_with_the_feedback_the_session_lacks():
    """request_reply starts the reply's generation, then sends one activity: the feedback the
    session does not hold yet, and the silent audio an activity must hold. The reply's first
    message, right after its activity, is that reply's; the next trigger carries only the
    feedback that came since."""
    rig = Rig()
    blocks = [pushed(1)]
    triggers = []

    async def reply(generation_id: str, *messages):
        """One reply, which the server starts as soon as its activity ends; what request_reply
        sent goes to ``triggers``."""
        request = GenerationRequest(generation_id, None, "inf", "wall", canonical=rig.view(blocks))
        rig.socket.answer("activityEnd", *messages, TURN_COMPLETE)
        before = len(rig.socket.sent)
        generation = await rig.driver.request_reply(request)
        triggers.append(rig.socket.sent[before:])
        return generation, await drain(rig.driver.events(generation))

    async def run():
        await rig.connect()
        await rig.driver.push_frame(None, rig.view(blocks))
        first = await reply("turn-0", call("call-0"))
        for decision, feedback in ((1, "EXECUTION RESULT: ok"), (2, "EXECUTION RESULT: missed")):
            blocks.extend(
                [
                    output_block(decision, "click P1", "obs-1", None),
                    feedback_block(decision, None, feedback, None),
                ]
            )
            await reply(f"turn-{decision}")
        delivery = rig.driver._core.delivery[("block", "d2.feedback")]
        await rig.driver.close()
        return first, delivery

    (generation, events), delivery = asyncio.run(run())
    call_event = next(event for event in events if isinstance(event, FunctionCall))
    assert (call_event.generation_id, call_event.call_id) == ("turn-0", "call-0")
    assert generation.metadata["video_ingestion_wait_ms"] == 0
    assert generation.metadata["gemini_live_input_transport"] == "realtime_input"
    assert [[wire(message) for message in trigger] for trigger in triggers] == [
        ACTIVITY,
        ["activityStart", ("text", "EXECUTION RESULT: ok"), "audio", "activityEnd"],
        ["activityStart", ("text", "EXECUTION RESULT: missed"), "audio", "activityEnd"],
    ]
    assert delivery == Delivery.SENT
    assert rig.log == []


def test_closing_the_connection_ends_the_reply_waiting_on_it():
    rig = Rig()

    async def run():
        await rig.connect()
        generation = await rig.start("turn-0")
        reading = asyncio.ensure_future(drain(rig.driver.events(generation)))
        await settle()
        await rig.driver.close()
        try:
            await reading
        except RuntimeError as exc:
            return exc

    assert str(asyncio.run(run())) == "Gemini Live connection 1: a read was cancelled"
