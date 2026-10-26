"""When a native-streaming push went out and when a reply's completed text came off the wire
(decisions N43 and 4.17), on both live drivers and in the streaming agent, each on a clock the
test sets or on the fake clock."""

from __future__ import annotations

import asyncio

from realtime_framework.entity.events import TextDone
from realtime_framework.entity.model import GenerationHandle, PushReceipt
from realtime_framework.entity.records import UNTRACED
from realtime_framework.providers.openai_realtime import OpenAIRealtimeDriver, _Session
from tests.fake_clock import FakeClock
from tests.session.test_gemini_live_receive import RecordingClock
from tests.session.test_gemini_live_session import pushed
from tests.session.test_gemini_live_streaming import Rig, settle, wire
from tests.session.test_openai_realtime_receive import (
    _clean_close,
    _FakeConnection,
    _offline_driver,
    _within,
)
from tests.session.test_openai_realtime_session import (
    Server,
    new_core,
    response_events,
    stream,
    streaming_view,
)

from .fake_live import BUDGET, REALTIME, Reply, Text, TextEnd, TextLiveDriver, make_agent
from .openai_session import _observation
from .test_agent import ORIGIN, _two_frames
from .test_openai_realtime_streaming import text_done, until


class Dial(RecordingClock):
    """RecordingClock, whose monotonic time is whatever the test last set ``now`` to."""

    def __init__(self, now: float, log: list | None = None):
        super().__init__([] if log is None else log)
        self.now = now

    def monotonic(self) -> float:
        return self.now


def _openai_driver(clock) -> OpenAIRealtimeDriver:
    """The offline native-streaming driver, on ``clock``."""
    driver = OpenAIRealtimeDriver(_offline_driver().config, clock=clock, streaming=True)
    driver.bind_trace(UNTRACED)
    return driver


def test_a_gemini_push_is_written_once_its_video_and_its_text_went_out():
    """A push's receipt says when its frame went out (decision N43): once both its writes
    returned, the video and then the text, here a text write that drains only at 2 s."""
    rig = Rig()
    clock = rig.driver.clock = Dial(1.0, rig.log)

    async def run():
        await rig.connect()
        rig.socket.stall = lambda message: wire(message) == ("text", "frame 1")
        push = asyncio.ensure_future(rig.driver.push_frame(None, rig.view([pushed(1)])))
        await settle()
        stalled = [wire(message) for message in rig.socket.sent]
        clock.now = 2.0
        rig.socket.release()
        receipt = await push
        await rig.driver.close()
        return stalled, receipt

    stalled, receipt = asyncio.run(run())
    assert stalled == ["video"]
    assert receipt.written_at == 2.0


def test_an_openai_push_is_written_once_its_frame_write_returned():
    """A push's receipt says when its frame went out (decision N43): once the write returned,
    here one that takes from 1 s to 1.25 s; the server confirms the frame as the write ends."""

    async def scenario():
        clock, connection, server = Dial(1.0), _FakeConnection(), Server()
        driver = _openai_driver(clock)
        session = _Session(1, connection, new_core())
        session.read_continuously(clock)
        driver._session, driver._conn = session, connection

        async def send(conn, operation):
            clock.now += 0.25
            conn.queue.put_nowait(server.answer(operation))

        driver._send = send
        receipt = await driver.push_frame(_observation(1), streaming_view(stream("F"), frames=5))
        connection.queue.put_nowait(_clean_close())
        await session.receive_until(lambda: False)
        return receipt

    assert _within(scenario()) == PushReceipt(written_at=1.25, confirmed_at=1.25)


def test_an_openai_reply_s_completed_text_keeps_the_time_it_came_off_the_wire():
    """The reader reads response.output_text.done at 3 s; taken from the reply's events only at
    7 s, its TextDone still says 3 s (decisions 19 and 4.17)."""

    async def scenario():
        clock, connection, core = Dial(3.0), _FakeConnection(), new_core()
        session = _Session(1, connection, core)
        session.read_continuously(clock)
        reply = core.start_reply()
        created, added, _, finished = response_events()
        for event in (created, added, text_done("event_7", "{}"), finished):
            connection.queue.put_nowait(event)
        await until(lambda: len(reply.events) == 4)
        clock.now = 7.0
        handle = GenerationHandle("reply-1", opaque=(session, reply))
        events = [event async for event in _openai_driver(clock).events(handle)]
        connection.queue.put_nowait(_clean_close())
        await session.receive_until(lambda: False)
        return events

    (completed,) = (event for event in _within(scenario()) if isinstance(event, TextDone))
    assert (completed.text, completed.received_at) == ("{}", 3.0)


class LateTextLiveDriver(TextLiveDriver):
    """TextLiveDriver, whose completed text is handed over 0.25 s after it came off the wire, as
    a reader's queue may."""

    async def events(self, generation):
        async for event in super().events(generation):
            if isinstance(event, TextDone):
                await asyncio.sleep(0.25)
            yield event


def test_an_action_in_text_arrives_when_its_text_came_off_the_wire_not_when_handed_over(
    monkeypatch,
):
    """Its text completed 0.75 s into the reply and was handed over 0.25 s later: the action
    arrived when the text came off the wire (decisions 19 and 4.17), and is posted when it is
    handed over."""
    clock = FakeClock(wall_origin=1000.0)
    with clock.participant():
        agent = make_agent(monkeypatch, clock, LateTextLiveDriver, REALTIME)
        posted = []
        agent.bind_inbox(lambda event: posted.append((event, clock.monotonic())), ORIGIN)
        frame = _two_frames(agent, clock)
        action = '{"action": "click", "x": 10, "y": 20, "button": "left", "frame": 2}'
        agent.driver.replies.append(Reply((Text(0.5, action), TextEnd(0.75)), 2.0))
        agent.generate(0, 0, frame, BUDGET)
        agent.close()
    [(arrival, posted_at)] = posted
    assert (arrival.call_id, arrival.arrived_at, arrival.frame) == ("text", 2.75, 2)
    assert (arrival.frame_age_s, posted_at) == (0.75, 2.0)
