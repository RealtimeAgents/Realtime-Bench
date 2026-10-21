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

import pytest
from google.genai import _common, _live_converters, types
from google.genai import errors as genai_errors
from google.genai import live as genai_live

from realtime_framework.entity.actions import gui_function_declarations
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.events import (
    FunctionCall,
    FunctionCallsCancelled,
    GenerationBoundary,
    ProviderHistoryState,
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
from tests.session.test_gemini_live_receive import DeliveryLog, RecordingClock, server_closed
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
        self.fail = None  # a predicate on a client message whose send raises

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
        if self.fail is not None and self.fail(message):
            raise RuntimeError("the activity end does not go out")
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


def setup_message(config) -> dict:
    """The setup google-genai sends for ``config`` to the Gemini Developer API, which the
    driver's client connects to, converted as its AsyncLive.connect converts it
    (google/genai/live.py); FedLive takes the config before that conversion."""
    parameters = types.LiveConnectParameters(model="models/gemini-live", config=config)
    request = _common.convert_to_dict(
        _live_converters._LiveConnectParameters_to_mldev(
            api_client=SimpleNamespace(vertexai=False),
            from_object=parameters.model_dump(exclude_none=True),
        )
    )
    del request["config"]
    return json.loads(json.dumps(_common.encode_unserializable_types(request)))["setup"]


class Rig:
    """A GeminiLiveDriver for ``model`` on fed connections, in native streaming unless
    ``streaming`` is False, with the model YAML's ``streaming_settings``; ``log`` holds its
    waits."""

    def __init__(self, model: str = MODEL, *, streaming: bool = True, streaming_settings=None):
        self.log: list = []
        config = ModelConfig(
            adapter_type="gemini_live",
            model_name=model,
            api_key="replay-dummy-key",
            streaming=streaming_settings or {},
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
    setup = setup_message(rig.live.configs[0])
    settings = live.GeminiLiveDriver.streaming_settings
    # google-genai sends these two as the config's field names, which the server's JSON takes.
    assert setup["contextWindowCompression"] == {
        "trigger_tokens": settings["compression_trigger_tokens"],
        "sliding_window": {"target_tokens": settings["compression_target_tokens"]},
    }
    assert setup["realtimeInputConfig"] == {
        "automatic_activity_detection": {"disabled": True},
        "activity_handling": "NO_INTERRUPTION",
        "turn_coverage": "TURN_INCLUDES_ALL_INPUT",
    }
    metadata = rig.driver._connect_metadata
    assert metadata["gemini_live_context_window_compression"] == settings
    assert metadata["gemini_live_server_context"] == "unknown"

    turn_based_setup = setup_message(turn_based.live.configs[0])
    assert "contextWindowCompression" not in turn_based_setup
    assert "realtimeInputConfig" not in turn_based_setup  # the preset's client_content
    assert setup["sessionResumption"] == turn_based_setup["sessionResumption"]
    assert "gemini_live_context_window_compression" not in turn_based.driver._connect_metadata


def test_a_model_yaml_sets_the_compression_thresholds_the_setup_sends():
    """The model YAML's streaming settings replace the driver's defaults they name
    (ModelConfig.streaming); the connection records the values it set up with."""
    rig = Rig(streaming_settings={"compression_trigger_tokens": 100_000})

    async def run():
        await rig.connect()
        await rig.driver.close()

    asyncio.run(run())
    settings = {**live.GeminiLiveDriver.streaming_settings, "compression_trigger_tokens": 100_000}
    assert rig.driver.streaming_settings == settings
    assert setup_message(rig.live.configs[0])["contextWindowCompression"] == {
        "trigger_tokens": 100_000,
        "sliding_window": {"target_tokens": settings["compression_target_tokens"]},
    }
    assert rig.driver._connect_metadata["gemini_live_context_window_compression"] == settings


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
    assert (result.disposition, result.detail) == (
        SyncDisposition.EXACT,
        "resumed Gemini Live session at its latest resumable handle: go_away",
    )
    assert result.provider_metadata["session_resumed"]
    assert [config.session_resumption.handle for config in rig.live.configs] == [None, "handle-1"]


def test_an_interrupt_closes_the_connection():
    """Under NO_INTERRUPTION an activity start cuts no reply off: interrupting a streaming reply
    closes its connection, which cancels the turn, and sends nothing, also for a model whose
    turn-based transport is realtime input."""
    rig = Rig("replay-gemini-live")

    async def run():
        await rig.connect()
        generation = await rig.start("turn-0")
        result = await rig.driver.interrupt(generation)
        return result, rig.driver._session

    result, session = asyncio.run(run())
    assert result.history_state == ProviderHistoryState.CONTAMINATED
    assert result.provider_metadata == {"interrupt_method": "connection_close"}
    assert session is None and rig.socket.sent == []


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
    assert first_event.received_at is not None


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


def test_a_push_or_a_trigger_is_sent_once_its_last_message_is_written():
    """The event log hears a pushed frame as sent after its text went out, and a trigger's
    feedback after the activity ended, on the connection they went out on; and once each reply
    is done, no queue is left for it."""
    rig = Rig()
    blocks = [
        pushed(1),
        output_block(1, "click P1", "obs-1", None),
        feedback_block(1, None, "EXECUTION RESULT: ok", None),
    ]
    log = DeliveryLog(lambda: rig.live.sockets[-1].sent if rig.live.sockets else [])
    rig.driver.bind_trace(log)

    async def run():
        await rig.connect()
        await rig.driver.push_frame(None, rig.view(blocks))
        request = GenerationRequest("turn-0", None, "inf", "wall", canonical=rig.view(blocks))
        rig.socket.answer("activityEnd", TURN_COMPLETE)
        await drain(rig.driver.events(await rig.driver.request_reply(request)))
        left = dict(rig.driver._replies)
        await rig.driver.close()
        return left

    assert asyncio.run(run()) == {}
    connection = "gemini_live.ws-1"
    assert log.events[1:] == [
        ("sent", ["f1.frame"], connection, 2),
        ("sent", ["d1.feedback"], connection, 6),
    ]


def test_a_push_and_a_trigger_go_out_one_after_the_other():
    """A frame pushed while a reply is requested goes out whole before or after the activity,
    never inside it."""
    rig = Rig()

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", None, "inf", "wall", canonical=rig.view([pushed(1)]))
        await asyncio.gather(
            rig.driver.push_frame(None, rig.view([pushed(1)])),
            rig.driver.request_reply(request),
        )
        await rig.driver.close()

    asyncio.run(run())
    sent = [wire(message) for message in rig.socket.sent]
    push = [
        wire(message)
        for message in rig.socket.sent
        if wire(message) in ("video", ("text", "frame 1"))
    ]
    assert sent in (push + ACTIVITY, ACTIVITY + push)


def test_a_write_cancelled_while_it_waits_for_the_session_records_nothing():
    """A push or a reply that is given up while a connection operation holds the session plans
    nothing: no block counts as written, no generation is waiting."""
    rig = Rig()
    blocks = [
        pushed(1),
        output_block(1, "click P1", "obs-1", None),
        feedback_block(1, None, "EXECUTION RESULT: ok", None),
        pushed(2),
    ]

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", None, "inf", "wall", canonical=rig.view(blocks))
        async with rig.driver._session_lock():
            writes = [
                asyncio.ensure_future(rig.driver.push_frame(None, rig.view(blocks))),
                asyncio.ensure_future(rig.driver.request_reply(request)),
            ]
            await settle()
            for write in writes:
                write.cancel()
            await asyncio.wait(writes)
        state = (dict(rig.driver._core.delivery), dict(rig.driver._core.carried))
        replies = (dict(rig.driver._core.replies), dict(rig.driver._replies))
        await rig.driver.close()
        return state, replies

    (delivery, carried), replies = asyncio.run(run())
    assert list(delivery) == [("setup",)] and carried == {}
    assert replies == ({}, {})
    assert rig.socket.sent == []


def test_a_reply_is_refused_while_an_activity_is_left_open():
    """An activity whose write failed cannot be closed as canonical context: no reply is
    requested on that connection until the session is rebuilt."""
    rig = Rig()

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", None, "inf", "wall", canonical=rig.view([pushed(1)]))
        rig.socket.fail = lambda message: wire(message) == "activityEnd"
        failures = []
        for _ in range(2):
            try:
                await rig.driver.request_reply(request)
            except RuntimeError as exc:
                failures.append(str(exc))
        await rig.driver.close()
        return failures

    assert asyncio.run(run()) == [
        "the activity end does not go out",
        "Gemini Live input activity is already open",
    ]


def go_away(time_left: str) -> dict:
    return {"goAway": {"timeLeft": time_left}}


@pytest.mark.parametrize("time_left", ["0s", "2s"])
def test_a_reply_read_to_its_end_before_a_go_away_is_handed_over_whole(time_left):
    """The reader reads a reply to its end, then a goAway whose deadline has passed by the time
    events() takes the reply: the reply is handed over whole and its call acknowledged, as a
    turn-based read of the same messages does; the connection is still to be renewed."""
    rig = Rig()

    async def run():
        await rig.connect()
        generation = await rig.start("turn-0")
        rig.socket.feed(call("call-0"), TURN_COMPLETE, go_away(time_left))
        await settle()
        events = await drain(rig.driver.events(generation))
        reason = rig.driver.connection_renewal_reason()
        await rig.driver.close()
        return events, reason

    events, reason = asyncio.run(run())
    assert [type(event).__name__ for event in events] == [
        "FunctionCall",
        "UsageSnapshot",
        "GenerationBoundary",
    ]
    assert reason == "go_away"
    assert [wire(message) for message in rig.socket.sent] == ["tool_response"]


def test_a_go_away_in_a_reply_leaves_the_reply_to_end_before_the_deadline():
    """A goAway in the middle of a reply: the reply that ends before the deadline is handed over
    whole; one that has not ended by then is no result (GoAwayDeadlineExceeded), as turn-based."""
    rig = Rig()

    async def run():
        await rig.connect()
        generation = await rig.start("turn-0")
        rig.socket.feed(call("call-0"), GO_AWAY, TURN_COMPLETE)
        ended = await drain(rig.driver.events(generation))
        generation = await rig.start("turn-1")
        rig.socket.feed(call("call-1"), go_away("2.1s"))
        try:
            await drain(rig.driver.events(generation))
        except live.GoAwayDeadlineExceeded as exc:
            return ended, exc

    ended, failure = asyncio.run(run())
    assert [getattr(event, "kind", type(event).__name__) for event in ended] == [
        "FunctionCall",
        "go_away",
        "UsageSnapshot",
        "turn_complete",
    ]
    assert isinstance(failure, live.GoAwayDeadlineExceeded)


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
