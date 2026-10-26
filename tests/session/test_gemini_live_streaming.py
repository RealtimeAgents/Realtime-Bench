"""The Gemini Live driver in native streaming (providers/gemini/live.py, ``streaming=True``), on
google-genai's AsyncSession over a socket the test feeds.

A streaming connection is read as messages arrive, so the scripted socket of
tests/session/test_gemini_live_receive.py, which hands out its whole script at once, would turn a
reply into orphans before its generation started. FedSocket hands out only what the test fed it,
when it fed it.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from google.genai import _common, _live_converters, types
from google.genai import errors as genai_errors
from google.genai import live as genai_live
from websockets.exceptions import ConnectionClosedError

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
    RetryCancelled,
    RuntimeState,
    SessionSeed,
    SyncDisposition,
    retry_cancellation,
)
from realtime_framework.entity.records import UNTRACED, current_trace_scope, trace_scope
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.request import system_block
from realtime_framework.harnesses.streaming.history import (
    feedback_block,
    history_view,
    output_block,
)
from realtime_framework.providers.gemini import live
from realtime_framework.providers.gemini.session import Delivery
from realtime_framework.providers.session_lifecycle import RenewalRequired
from tests.session.test_gemini_live_receive import (
    DeliveryLog,
    RecordingClock,
    server_closed,
    yaml_options,
)
from tests.session.test_gemini_live_session import pushed

MODEL = (
    "gemini-3.8-live"  # the client_content preset: streaming goes on realtime_input all the same
)


class FedSocket:
    """What google-genai's AsyncSession reads and writes: the server messages the test feeds, in
    order, each once fed; an exception fed is raised by the read that takes it."""

    def __init__(self, number: int):
        self.trace_connection_number = number
        self.trace_connection_id = f"gemini_live.ws-{number}"
        self.sent: list[dict] = []
        self.read_scopes: list[tuple] = []  # each server message's kind and its trace scope
        self._frames: asyncio.Queue = asyncio.Queue()
        self._answers: list[tuple] = []  # (kind, frames), in the order they are due
        self.fail = None  # a predicate on a client message whose send raises
        self.stall = None  # a predicate on a client message whose send does not drain
        # Closing waits for a write that does not drain, as websockets' close does.
        self.close_stalls = False
        self._dropped = asyncio.Event()
        self._drained = asyncio.Event()  # release() lets a stalled write go out
        # websockets' transport: an abort drops the connection, which fails a write stuck in it.
        self.transport = SimpleNamespace(abort=self._dropped.set)

    def release(self) -> None:
        self._drained.set()

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
        # The trace records a server message in the scope it is read in, as here
        # (providers/capture.py, the recording connection's recv).
        self.read_scopes.append((next(iter(json.loads(frame))), current_trace_scope()))
        return frame

    async def send(self, message):
        message = json.loads(message)
        if self.fail is not None and self.fail(message):
            raise RuntimeError("the activity end does not go out")
        if self.stall is not None and self.stall(message):
            waits = [
                asyncio.ensure_future(event.wait()) for event in (self._dropped, self._drained)
            ]
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            for wait in waits:
                wait.cancel()
            if self._dropped.is_set():
                raise ConnectionClosedError(None, None)
        self.sent.append(message)
        if self._answers and wire(message) == self._answers[0][0]:
            self.feed(*self._answers.pop(0)[1])
        # A write gives the loop its turns, as a real socket's does: a reader then takes what
        # came meanwhile.
        await settle()


class FedLive:
    """client.aio.live: each connect opens a session on a new FedSocket; ``configs`` holds each
    connect's setup. While ``setup_stalls``, a connect's setupComplete does not come; ``stall``
    is each new socket's FedSocket.stall."""

    def __init__(self):
        self.sockets: list[FedSocket] = []
        self.configs: list = []
        self.setup_stalls = False
        self.stall = None

    def connect(self, *, model, config):
        self.configs.append(config)
        socket = FedSocket(len(self.sockets) + 1)
        socket.stall = self.stall
        self.sockets.append(socket)
        session = genai_live.AsyncSession(
            api_client=SimpleNamespace(vertexai=False),
            websocket=socket,
            setup_complete=types.LiveServerSetupComplete(),
        )
        return _Opened(session, socket, setup_stalls=self.setup_stalls)


class _Opened:
    """What client.aio.live.connect returns: entered, the session once the setup is
    acknowledged; exited, the connection closed, which waits for a write stuck in its socket
    where FedSocket.close_stalls."""

    def __init__(self, session, socket: FedSocket, *, setup_stalls: bool = False):
        self._session = session
        self._socket = socket
        self._setup_stalls = setup_stalls

    async def __aenter__(self):
        if self._setup_stalls:
            await asyncio.Event().wait()
        return self._session

    async def __aexit__(self, *exc):
        if self._socket.close_stalls:
            await self._socket._dropped.wait()


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
            provider_options=yaml_options(model),
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
        request = GenerationRequest(generation_id, "inf", canonical=canonical)
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
        await rig.driver.interrupt(generation)
        return rig.driver._session

    session = asyncio.run(run())
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


def test_a_cancellation_between_replies_is_held_across_a_reconnect():
    """A cancellation read between replies is not dropped with its connection: the next reply,
    on the new one, hands it over first, telling the connection it was read on."""
    rig = Rig()

    async def run():
        await rig.connect()
        first = await rig.start("turn-0")
        rig.socket.feed(call("call-0"), TURN_COMPLETE)
        await drain(rig.driver.events(first))
        rig.socket.feed(cancelled("call-0"))
        await settle()
        await rig.connect()
        second = await rig.start("turn-1")
        rig.socket.feed(TURN_COMPLETE)
        events = await drain(rig.driver.events(second))
        await rig.driver.close()
        return events

    first_event, *rest = asyncio.run(run())
    assert isinstance(first_event, FunctionCallsCancelled)
    assert (first_event.call_ids, first_event.generation_id) == (("call-0",), "turn-1")
    assert (first_event.connection, rest[-1].connection) == (1, 2)


def test_a_cancellation_read_after_a_reply_ended_is_no_event_of_that_reply():
    """A cancellation read once a reply ended, before its events() began, comes with the next
    reply, not first thing in that one, ahead of the call it names."""
    rig = Rig()

    async def run():
        await rig.connect()
        first = await rig.start("turn-0")
        rig.socket.feed(call("call-0"), TURN_COMPLETE, cancelled("call-0"))
        await settle()
        events = await drain(rig.driver.events(first))
        second = await rig.start("turn-1")
        rig.socket.feed(TURN_COMPLETE)
        later = await drain(rig.driver.events(second))
        await rig.driver.close()
        return events, later

    events, later = asyncio.run(run())
    assert [type(e).__name__ for e in events] == [
        "FunctionCall",
        "UsageSnapshot",
        "GenerationBoundary",
    ]
    assert isinstance(later[0], FunctionCallsCancelled) and later[0].call_ids == ("call-0",)


def test_a_cancellation_is_stamped_when_read_and_handed_over_once():
    """A cancellation's arrival is when the reader read it, held between replies or in one;
    one held is the first of the next reply's events, numbered as they are, and of no later
    reply's."""
    rig = Rig()

    async def run():
        await rig.connect()
        first = await rig.start("turn-0")
        rig.socket.feed(call("call-0"), call("call-1"), TURN_COMPLETE)
        await drain(rig.driver.events(first))
        held = [rig.driver.clock.monotonic()]
        rig.socket.feed(cancelled("call-0"))
        await settle()
        held.append(rig.driver.clock.monotonic())
        await asyncio.sleep(0.05)
        second = await rig.start("turn-1")
        within = [rig.driver.clock.monotonic()]
        rig.socket.feed(cancelled("call-1"))
        await settle()
        within.append(rig.driver.clock.monotonic())
        await asyncio.sleep(0.05)
        rig.socket.feed(TURN_COMPLETE)
        events = await drain(rig.driver.events(second))
        third = await rig.start("turn-2")
        rig.socket.feed(TURN_COMPLETE)
        later = await drain(rig.driver.events(third))
        await rig.driver.close()
        return held, within, events, later

    held, within, events, later = asyncio.run(run())
    assert [(type(e).__name__, e.sequence) for e in events] == [
        ("FunctionCallsCancelled", 1),
        ("FunctionCallsCancelled", 2),
        ("UsageSnapshot", 3),
        ("GenerationBoundary", 4),
    ]
    assert [e.call_ids for e in events[:2]] == [("call-0",), ("call-1",)]
    assert held[0] <= events[0].received_at <= held[1]
    assert within[0] <= events[1].received_at <= within[1]
    assert not [e for e in later if isinstance(e, FunctionCallsCancelled)]


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
        before = rig.driver.clock.monotonic()
        pushed_frame = await rig.driver.push_frame(None, rig.view([pushed(1)]))
        written = (before, rig.driver.clock.monotonic())
        delivery = rig.driver._core.delivery[("block", "f1.frame")]
        rig.socket.feed(GO_AWAY)
        await settle()
        try:
            await rig.driver.push_frame(None, rig.view([pushed(1), pushed(2)]))
        except RuntimeError as exc:
            refused = str(exc)
        await rig.driver.close()
        return pushed_frame, written, delivery, refused

    pushed_frame, written, delivery, refused = asyncio.run(run())
    # The receipt (decision N43): written once both went out; Gemini Live confirms no frame.
    assert pushed_frame.confirmed_at is None
    assert written[0] <= pushed_frame.written_at <= written[1]
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
        request = GenerationRequest(generation_id, "inf", canonical=rig.view(blocks))
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
        request = GenerationRequest("turn-0", "inf", canonical=rig.view(blocks))
        rig.socket.answer("activityEnd", TURN_COMPLETE)
        await drain(rig.driver.events(await rig.driver.request_reply(request)))
        left = dict(rig.driver._core.replies)
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
        request = GenerationRequest("turn-0", "inf", canonical=rig.view([pushed(1)]))
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
        request = GenerationRequest("turn-0", "inf", canonical=rig.view(blocks))
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
        replies = dict(rig.driver._core.replies)
        await rig.driver.close()
        return state, replies

    (delivery, carried), replies = asyncio.run(run())
    assert list(delivery) == [("setup",)] and carried == {}
    assert replies == {}
    assert rig.socket.sent == []


def test_a_reply_is_refused_while_an_activity_is_left_open():
    """An activity whose write failed cannot be closed as canonical context: no reply is
    requested on that connection until the session is rebuilt."""
    rig = Rig()

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", "inf", canonical=rig.view([pushed(1)]))
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


def test_an_abort_fails_a_push_stuck_in_a_write_and_frees_the_session(monkeypatch):
    """A push stuck in a write that does not drain holds the session. An abort's close of the
    connection waits on the same write until its bound, then drops the connection under it: the
    push fails, and the next reply's ensure_ready takes the session and rebuilds it."""
    monkeypatch.setattr(live, "_CONNECTION_CLOSE_TIMEOUT_S", 0.05)
    rig = Rig()

    async def run():
        await rig.connect()
        rig.socket.stall = lambda message: wire(message) == "video"
        rig.socket.close_stalls = True
        push = asyncio.ensure_future(rig.driver.push_frame(None, rig.view([pushed(1)])))
        await settle()
        await rig.driver.abort(None)
        await settle()
        failure = push.exception() if push.done() else None
        await asyncio.wait_for(
            rig.driver.ensure_ready(rig.seed, rig.view([]), lambda *_, **__: None), 2
        )
        await rig.driver.close()
        return failure

    assert isinstance(asyncio.run(run()), ConnectionClosedError)
    assert len(rig.live.sockets) == 2


@pytest.mark.parametrize(
    ("stuck", "streaming"), [("setup", False), ("setup", True), ("seed", True)]
)
def test_an_abort_ends_a_connect_its_abandoned_generation_is_stuck_in(stuck, streaming):
    """A connect waiting for the server's setupComplete, or for a seed write that does not
    drain, holds the session. The abort of the generation it readies the session for ends it,
    so the next generation readies a session of its own."""
    rig = Rig(streaming=streaming)
    history = rig.view([pushed(1)]) if stuck == "seed" else None
    if stuck == "setup":
        rig.live.setup_stalls = True
    else:
        rig.live.stall = lambda message: wire(message) == "client_content"

    async def run():
        cancel = threading.Event()
        with retry_cancellation(cancel):
            readying = asyncio.ensure_future(
                rig.driver.ensure_ready(rig.seed, history, lambda *_, **__: None)
            )
        await settle()
        cancel.set()
        await asyncio.wait_for(rig.driver.abort(None), 2)
        done, _ = await asyncio.wait({readying}, timeout=0.5)
        assert done, "the connect still holds the session after the abort"
        rig.live.setup_stalls, rig.live.stall = False, None
        await asyncio.wait_for(rig.driver.ensure_ready(rig.seed, history, lambda *_, **__: None), 2)
        state = rig.driver.state
        await rig.driver.close()
        return readying.exception(), state

    ended, state = asyncio.run(run())
    assert isinstance(ended, RetryCancelled)
    assert state == RuntimeState.READY and len(rig.live.sockets) == 2


def test_an_abort_drops_a_connect_whose_close_waits_on_its_stuck_seed_write(monkeypatch):
    """A connect that gives up a seed write that does not drain closes its connection, and the
    close waits on the same write, as websockets' close waits for its buffer to drain first. The
    abort of the generation it readies the session for waits for the connect only until the
    connection-close bound, then drops the connection under both: the abort returns and the
    connect ends, where it would otherwise hold the session until the peer read again."""
    monkeypatch.setattr(live, "_CONNECTION_CLOSE_TIMEOUT_S", 0.05)
    rig = Rig()
    rig.live.stall = lambda message: wire(message) == "client_content"

    async def run():
        cancel = threading.Event()
        with retry_cancellation(cancel):
            readying = asyncio.ensure_future(
                rig.driver.ensure_ready(rig.seed, rig.view([pushed(1)]), lambda *_, **__: None)
            )
        await settle()
        socket = rig.socket
        socket.close_stalls = True
        cancel.set()
        aborting = asyncio.ensure_future(rig.driver.abort(None))
        done, _ = await asyncio.wait({aborting, readying}, timeout=0.5)
        dropped = socket._dropped.is_set()
        # A connect the abort left in its write would take asyncio.run's one cancel there and
        # then wait in its close for good.
        socket.transport.abort()
        return aborting in done, readying.exception() if readying in done else None, dropped

    aborted, ended, dropped = asyncio.run(run())
    assert aborted
    assert isinstance(ended, RetryCancelled)
    assert dropped


def test_a_close_ends_a_connect_still_opening():
    """A close while a connect waits for the server's setupComplete ends that connect, which
    holds the session meanwhile: the connect raises that the session closed while it opened."""
    rig = Rig()
    rig.live.setup_stalls = True

    async def run():
        connecting = asyncio.ensure_future(rig.connect())
        await settle()
        await asyncio.wait_for(rig.driver.close(), 2)
        done, _ = await asyncio.wait({connecting}, timeout=0.5)
        assert done, "the connect is still opening after the close"
        return str(connecting.exception())

    assert asyncio.run(run()) == "Gemini Live session closed while it opened"


class _SetupEnded(_Opened):
    """A connection whose setup the server ends, once its WebSocket opened."""

    async def __aenter__(self):
        raise server_closed(1011, "setup failed")


def test_events_carry_the_number_the_trace_gives_their_connection():
    """A connection whose setup failed counts in the trace, not in the session core, which counts
    the connections it set up: the events of the next one carry the trace's number, as OpenAI
    Realtime's do and as the delivery records name the connection (review-encoders 设计 8)."""
    rig = Rig()
    connect = rig.live.connect

    def setup_ends_first(*, model, config):
        opened = connect(model=model, config=config)
        return opened if len(rig.live.sockets) > 1 else _SetupEnded(opened._session, opened._socket)

    rig.live.connect = setup_ends_first

    async def run():
        with pytest.raises(ConnectionClosedError):
            await rig.connect()
        await rig.connect()
        request = GenerationRequest("turn-1", "inf", canonical=rig.view([]))
        rig.socket.answer("activityEnd", call("call-1"), TURN_COMPLETE)
        events = await drain(rig.driver.events(await rig.driver.request_reply(request)))
        await rig.driver.close()
        return events

    events = asyncio.run(run())
    assert {(event.connection, event.generation) for event in events} == {(2, 1)}
    assert rig.driver._core.connection is None and len(rig.live.sockets) == 2


def test_each_message_is_traced_under_the_reply_it_was_read_for():
    """The reader records each message in the scope of the reply in flight when it reads it, as
    a turn-based reply's own read does: not in the scope of the call that opened the connection,
    and in none between replies."""
    rig = Rig()

    async def run():
        with trace_scope(phase="initialize"):
            await rig.connect()
        for k in (1, 2):
            request = GenerationRequest(f"turn-{k}", "inf", canonical=rig.view([]))
            with trace_scope(phase="decision", generation=k):
                rig.socket.answer("activityEnd", call(f"call-{k}"), TURN_COMPLETE)
                await drain(rig.driver.events(await rig.driver.request_reply(request)))
            rig.socket.feed(update(f"handle-{k}"))
            await settle()
        await rig.driver.close()

    asyncio.run(run())
    scopes = [{"phase": "decision", "generation": k} for k in (1, 2)]
    assert rig.socket.read_scopes == [
        ("toolCall", scopes[0]),
        ("serverContent", scopes[0]),
        ("sessionResumptionUpdate", None),
        ("toolCall", scopes[1]),
        ("serverContent", scopes[1]),
        ("sessionResumptionUpdate", None),
    ]


def test_a_message_read_before_the_first_reply_is_traced_under_no_scope():
    """The reader runs in a context of its own, not in a copy of the connecting call's: a
    message it reads before the connection's first reply, as a resumption update right after
    the setup, is traced in no scope, not in that of the call that connected, which for a push
    that rebuilt the session is the push's."""
    rig = Rig()

    async def run():
        with trace_scope(phase="push", observation_id="obs-1"):
            await rig.connect()
        rig.socket.feed(update("handle-0"))
        await settle()
        await rig.driver.close()

    asyncio.run(run())
    assert rig.socket.read_scopes == [("sessionResumptionUpdate", None)]


def test_nothing_read_after_a_reply_is_traced_under_it_however_it_ended():
    """What the reader reads after a reply is in no reply's scope, also when its events() was
    closed before its end, or when its trigger failed and it never came."""
    rig = Rig()

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", "inf", canonical=rig.view([]))
        with trace_scope(phase="decision", generation=1):
            events = rig.driver.events(await rig.driver.request_reply(request))
            rig.socket.feed(call("call-0"))
            await anext(events)
            await events.aclose()
        rig.socket.feed(TURN_COMPLETE, update("handle-1"))
        await settle()
        rig.socket.fail = lambda message: wire(message) == "activityEnd"
        with trace_scope(phase="decision", generation=2), pytest.raises(RuntimeError):
            await rig.driver.request_reply(replace(request, generation_id="turn-1"))
        rig.socket.feed(update("handle-2"))
        await settle()
        await rig.driver.close()

    asyncio.run(run())
    assert rig.socket.read_scopes == [
        ("toolCall", {"phase": "decision", "generation": 1}),
        ("serverContent", {"phase": "decision", "generation": 1}),
        ("sessionResumptionUpdate", None),
        ("sessionResumptionUpdate", None),
    ]


def test_nothing_goes_out_once_the_reader_failed():
    """A reader that failed with no reply waiting leaves a connection that takes nothing: a push
    and a reply raise its failure before anything is written."""
    rig = Rig()

    async def run():
        await rig.connect()
        rig.socket.feed(server_closed(1011, "Internal error."))
        await settle()
        request = GenerationRequest("turn-0", "inf", canonical=rig.view([pushed(1)]))
        failures = []
        for write in (
            rig.driver.push_frame(None, rig.view([pushed(1)])),
            rig.driver.request_reply(request),
        ):
            try:
                await write
            except genai_errors.APIError as exc:
                failures.append(exc)
        await rig.driver.close()
        return failures

    first, second = asyncio.run(run())
    assert first.code == 1011 and second is first
    assert rig.socket.sent == []


def test_a_reader_that_failed_leaves_the_session_to_be_rebuilt():
    """A connection whose reader failed takes nothing, so the session is dirty: readying it
    rebuilds it on a new connection, which takes the next push."""
    rig = Rig()

    async def run():
        await rig.connect()
        rig.socket.feed(server_closed(1011, "Internal error."))
        await settle()
        state = rig.driver.state
        await rig.driver.ensure_ready(rig.seed, rig.view([pushed(1)]), lambda *_, **__: None)
        await rig.driver.push_frame(None, rig.view([pushed(1), pushed(2)]))
        await rig.driver.close()
        return state

    assert asyncio.run(run()) == RuntimeState.DIRTY
    assert len(rig.live.sockets) == 2
    assert [wire(message) for message in rig.socket.sent][-2:] == ["video", ("text", "frame 2")]


def test_a_reply_is_refused_while_one_is_in_flight():
    """A connection carries one reply at a time: the next is refused until the one in flight
    ended, and nothing of it is written."""
    rig = Rig()

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", "inf", canonical=rig.view([pushed(1)]))
        await rig.driver.request_reply(request)
        written = len(rig.socket.sent)
        try:
            await rig.driver.request_reply(replace(request, generation_id="turn-1"))
        except RuntimeError as exc:
            refused = str(exc)
        unchanged = len(rig.socket.sent) == written
        await rig.driver.close()
        return refused, unchanged

    refused, unchanged = asyncio.run(run())
    assert refused == "Gemini Live has a reply in flight on the connection"
    assert unchanged


def test_a_reply_read_to_its_end_is_in_flight_until_its_events_are_done():
    """The reader having read a reply's end, its events() may still acknowledge its calls: the
    next reply is refused until they are done."""
    rig = Rig()

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", "inf", canonical=rig.view([pushed(1)]))
        await rig.driver.request_reply(request)
        rig.socket.feed(call("call-0"), TURN_COMPLETE)
        await settle()
        refused = None
        try:
            await rig.driver.request_reply(replace(request, generation_id="turn-1"))
        except RuntimeError as exc:
            refused = str(exc)
        await rig.driver.close()
        return refused

    assert asyncio.run(run()) == "Gemini Live has a reply in flight on the connection"


def test_a_failed_trigger_leaves_no_reply_registered_and_the_activity_open():
    """A trigger whose write failed is no reply: its feedback is unknown, no generation takes
    the connection's messages, so a cancellation read later waits for the next reply, as one
    held for this reply does; and the activity left open refuses a push as well as a reply."""
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
        rig.socket.feed(cancelled("call-0"))
        await settle()
        rig.socket.fail = lambda message: wire(message) == "activityEnd"
        request = GenerationRequest("turn-0", "inf", canonical=rig.view(blocks))
        try:
            await rig.driver.request_reply(request)
        except RuntimeError:
            pass
        state = dict(rig.driver._core.replies)
        rig.socket.feed(cancelled("call-1"))
        await settle()
        held = [orphan.cancelled for orphan in rig.driver._core.cancelled_between]
        try:
            await rig.driver.push_frame(None, rig.view([*blocks, pushed(2)]))
        except RuntimeError as exc:
            refused = str(exc)
        await rig.driver.close()
        return state, held, refused

    state, held, refused = asyncio.run(run())
    assert state == {}
    assert held == [("call-0",), ("call-1",)]
    assert refused == "Gemini Live input activity is already open"
    # Its start, feedback and audio went out before its end failed.
    assert ("unknown", ["d1.feedback"], "gemini_live.ws-1", 3) in log.events


def test_a_push_whose_write_failed_is_unknown():
    rig = Rig()
    log = DeliveryLog(lambda: rig.live.sockets[-1].sent if rig.live.sockets else [])
    rig.driver.bind_trace(log)

    async def run():
        await rig.connect()
        rig.socket.fail = lambda message: wire(message) == ("text", "frame 1")
        try:
            await rig.driver.push_frame(None, rig.view([pushed(1)]))
        except RuntimeError:
            pass
        await rig.driver.close()

    asyncio.run(run())
    assert log.events[1:] == [("unknown", ["f1.frame"], "gemini_live.ws-1", 1)]


def test_after_a_go_away_input_waits_for_the_renewal_that_resumes_once_the_reply_ended():
    """A goAway in a reply: pushes raise RenewalRequired, the reply goes on to its end, and the
    renewal after it resumes the session with the latest resumable handle; it takes pushes
    again."""
    rig = Rig()
    record = []

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", "inf", canonical=rig.view([pushed(1)]))
        generation = await rig.driver.request_reply(request)
        rig.socket.feed(update("handle-1"), GO_AWAY)
        await settle()
        try:
            await rig.driver.push_frame(None, rig.view([pushed(1), pushed(2)]))
        except RenewalRequired as exc:
            refused = exc
        rig.socket.feed(TURN_COMPLETE)
        await drain(rig.driver.events(generation))
        renewed = await rig.driver.renew(
            rig.seed, rig.view([pushed(1), pushed(2)]), lambda *args, **_: record.append(args)
        )
        await rig.driver.push_frame(None, rig.view([pushed(1), pushed(2)]))
        await rig.driver.close()
        return refused, renewed

    refused, renewed = asyncio.run(run())
    assert isinstance(refused, RuntimeError)
    assert renewed.provider_metadata["session_resumed"]
    assert [config.session_resumption.handle for config in rig.live.configs] == [None, "handle-1"]
    assert [wire(message) for message in rig.socket.sent] == ["video", ("text", "frame 2")]


def test_a_reply_a_go_away_refuses_while_it_waits_for_the_session_is_asked_again_after_renewal():
    """A goAway read while a reply waits for the session a push holds: the reply raises
    RenewalRequired having written and started nothing, and leaves the session as it was. The
    renewal resumes it, and the reply asked again goes out on the new connection."""
    rig = Rig()
    view = rig.view([pushed(1)])
    request = GenerationRequest("turn-0", "inf", canonical=view)

    async def run():
        await rig.connect()
        first = rig.socket
        first.stall = lambda message: wire(message) == "video"
        push = asyncio.ensure_future(rig.driver.push_frame(None, view))
        await settle()
        reply = asyncio.ensure_future(rig.driver.request_reply(request))
        first.feed(update("handle-1"), GO_AWAY)
        await settle()
        first.release()
        await push
        refused = None
        try:
            await reply
        except RenewalRequired as exc:
            refused = exc
        left = (
            [wire(message) for message in first.sent],
            dict(rig.driver._core.replies),
            rig.driver.state,
            rig.driver.connection_renewal_reason(),
        )
        renewed = await rig.driver.renew(rig.seed, view, lambda *_, **__: None)
        generation = await rig.driver.request_reply(request)
        rig.socket.feed(TURN_COMPLETE)
        await drain(rig.driver.events(generation))
        await rig.driver.close()
        return refused, left, renewed

    refused, left, renewed = asyncio.run(run())
    assert isinstance(refused, RenewalRequired)
    assert left == (["video", ("text", "frame 1")], {}, RuntimeState.READY, "go_away")
    assert renewed.provider_metadata["session_resumed"]
    assert [config.session_resumption.handle for config in rig.live.configs] == [None, "handle-1"]
    assert [wire(message) for message in rig.socket.sent] == ACTIVITY


def test_a_renewal_before_the_reply_ended_closes_the_connection_under_it():
    rig = Rig()

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", "inf", canonical=rig.view([pushed(1)]))
        generation = await rig.driver.request_reply(request)
        rig.socket.feed(update("handle-1"), GO_AWAY)
        await settle()
        await rig.driver.renew(rig.seed, rig.view([pushed(1)]), lambda *args, **_: None)
        try:
            await drain(rig.driver.events(generation))
        except RuntimeError as exc:
            failure = str(exc)
        await rig.driver.close()
        return failure

    assert asyncio.run(run()) == "Gemini Live connection 1: a read was cancelled"
    assert [config.session_resumption.handle for config in rig.live.configs] == [None, "handle-1"]


@pytest.mark.parametrize("lost", ["abort", "failed trigger"])
def test_after_an_abort_or_a_failed_trigger_the_renewal_rebuilds(lost):
    """Once an abort closed the connection, or a trigger did not go out whole, what the session
    holds is not known: the abort leaves it closed, the agent leaves the failed reply's dirty
    (harnesses/streaming/agent.py _reply), and the agent readies the session before it renews it
    (_ready). That reopens or rebuilds it from the history, though a resumable handle and a goAway
    had come, and no renewal is due after it; the seed replays the feedback."""
    rig = Rig()
    blocks = [
        pushed(1),
        output_block(1, "click P1", "obs-1", None),
        feedback_block(1, None, "EXECUTION RESULT: ok", None),
    ]

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", "inf", canonical=rig.view(blocks))
        rig.socket.feed(update("handle-1"))
        await settle()
        if lost == "abort":
            generation = await rig.driver.request_reply(request)
            rig.socket.feed(GO_AWAY)
            await settle()
            await rig.driver.abort(generation)
        else:
            rig.socket.fail = lambda message: wire(message) == "activityEnd"
            with pytest.raises(RuntimeError):
                await rig.driver.request_reply(request)
            rig.driver.mark_dirty()
            rig.socket.feed(GO_AWAY)
            await settle()
        record = lambda *_, **__: None
        ready = await rig.driver.ensure_ready(rig.seed, rig.view(blocks), record)
        renewed = await rig.driver.renew(rig.seed, rig.view(blocks), record)
        await rig.driver.close()
        return ready, renewed

    ready, renewed = asyncio.run(run())
    if lost == "abort":
        assert ready is None  # a session reopened, not a sync
    else:
        assert (ready.disposition, ready.detail) == (SyncDisposition.REBUILT, "runtime_recovery")
    assert renewed is None
    assert [config.session_resumption.handle for config in rig.live.configs] == [None, None]
    assert "d1.feedback" in rig.driver._core.carried


def test_a_handle_that_comes_after_a_failed_trigger_resumes_nothing_either():
    """A resumable handle the server sends on the connection once a trigger did not go out whole
    resumes its session no more than one before: the failed reply left the session dirty, which
    the readying before the renewal rebuilds."""
    rig = Rig()

    async def run():
        await rig.connect()
        rig.socket.fail = lambda message: wire(message) == "activityEnd"
        request = GenerationRequest("turn-0", "inf", canonical=rig.view([pushed(1)]))
        with pytest.raises(RuntimeError):
            await rig.driver.request_reply(request)
        rig.driver.mark_dirty()
        rig.socket.feed(update("handle-2"), GO_AWAY)
        await settle()
        record = lambda *_, **__: None
        ready = await rig.driver.ensure_ready(rig.seed, rig.view([pushed(1)]), record)
        renewed = await rig.driver.renew(rig.seed, rig.view([pushed(1)]), record)
        await rig.driver.close()
        return ready, renewed

    ready, renewed = asyncio.run(run())
    assert (ready.disposition, ready.detail, renewed) == (
        SyncDisposition.REBUILT,
        "runtime_recovery",
        None,
    )
    assert [config.session_resumption.handle for config in rig.live.configs] == [None, None]


def test_an_acknowledgement_written_after_a_reconnect_is_recorded_on_its_connection():
    """An acknowledgement whose write returns once the next connection is open is sent on the
    connection it went out on, and is no write of the new one."""
    rig = Rig()
    log = DeliveryLog(lambda: rig.live.sockets[0].sent)
    rig.driver.bind_trace(log)

    async def run():
        await rig.connect()
        first = rig.socket
        first.stall = lambda message: wire(message) == "tool_response"
        generation = await rig.start("turn-0")
        reading = asyncio.ensure_future(drain(rig.driver.events(generation)))
        first.feed(call("call-0"))
        await settle()
        await rig.connect()
        first.release()
        await asyncio.wait({reading})
        new_writes = dict(rig.driver._core.delivery)
        await rig.driver.close()
        return new_writes

    new_writes = asyncio.run(run())
    assert ("sent", [None], "gemini_live.ws-1", 1) in log.events
    assert ("tool_response", "call-0") not in new_writes


def test_a_reply_is_read_on_the_connection_it_started_on():
    """A reply started before a reconnect is read on its own connection, which the reconnect
    closed: its events() hands over what was read on it, raises that it closed, and nothing of
    it, not even the acknowledgement of a call read on the old one, goes to the new one."""
    rig = Rig()

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", "inf", canonical=rig.view([pushed(1)]))
        generation = await rig.driver.request_reply(request)
        rig.socket.feed(call("call-0"))
        await settle()
        await rig.connect()
        rig.socket.feed(call("call-1"))
        await settle()
        try:
            await drain(rig.driver.events(generation))
        except RuntimeError as exc:
            failure = str(exc)
        left = dict(rig.driver._core.replies)
        await rig.driver.close()
        return failure, left

    failure, left = asyncio.run(run())
    assert failure == "Gemini Live connection 1: a read was cancelled"
    assert rig.live.sockets[1].sent == [] and left == {}


def test_a_reply_on_a_closed_connection_is_bound_by_no_go_away_of_the_new_one():
    """A goAway on the new connection bounds nothing of a reply read on the one before: past its
    deadline, that reply still hands over what was read and leaves the new connection open."""
    rig = Rig()

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", "inf", canonical=rig.view([pushed(1)]))
        old = await rig.driver.request_reply(request)
        rig.socket.feed(call("call-0"))
        await settle()
        await rig.connect()
        rig.socket.feed(go_away("0s"))
        await settle()
        try:
            await drain(rig.driver.events(old))
        except RuntimeError as exc:
            failure = exc
        opened = rig.driver._session is not None
        await rig.driver.close()
        return failure, opened

    failure, opened = asyncio.run(run())
    assert str(failure) == "Gemini Live connection 1: a read was cancelled"
    assert opened


def test_a_call_handed_over_after_a_reconnect_is_no_call_of_the_new_connection():
    """A call read before a reconnect, which its reply hands over after it, is pending on the
    connection it was read on only: the new one's records hold none of it, and the next reply
    there answers only its own call."""
    rig = Rig()

    async def run():
        await rig.connect()
        request = GenerationRequest("turn-0", "inf", canonical=rig.view([pushed(1)]))
        old = await rig.driver.request_reply(request)
        rig.socket.feed(call("call-0"))
        await settle()
        await rig.connect()
        with pytest.raises(RuntimeError):
            await drain(rig.driver.events(old))
        records = (list(rig.driver._core.pending_calls), dict(rig.driver._core.delivery))
        new = await rig.driver.request_reply(replace(request, generation_id="turn-1"))
        rig.socket.feed(call("call-1"), TURN_COMPLETE)
        await drain(rig.driver.events(new))
        await rig.driver.close()
        return records

    pending, delivery = asyncio.run(run())
    assert pending == [] and ("tool_response", "call-0") not in delivery
    answered = [
        [response["id"] for response in message["tool_response"]["functionResponses"]]
        for message in rig.socket.sent
        if wire(message) == "tool_response"
    ]
    assert answered == [["call-1"]]


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
