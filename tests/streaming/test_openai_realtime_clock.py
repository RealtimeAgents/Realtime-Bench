"""Whether the OpenAI Realtime driver's receive runs on the fake clock (stream-plan 2.6).

The driver reads with ``event = await self._conn.recv()`` (providers/openai_realtime.py
_Session._receive), the async SDK's recv on websockets' asyncio connection
(openai/resources/realtime/realtime.py:326-343). On a loop the fake clock made, that waits in the
loop's select, which the clock sees as waiting, so a server message due later comes when the clock
reads its time, as Gemini Live's does.

cassettes/openai_realtime-timed-reply.jsonl is one turn, converted from a session against the
stub of test_cassette_from_trace.py, its one text delta split in two, and then timed by hand: the
acknowledgements are reactive, the response starts at 0.2 s, its two text deltas arrive at 0.5 s
and 0.8 s and it is done at 1.0 s, when the driver closes. The server messages come in the order
the driver read them: it reads only while it waits for something (openai_realtime.py _Session),
first after response.create.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from realtime_framework.entity.records import UNTRACED
from tests.fake_clock import FakeClock

from . import ws_cassette
from .clocked_ws import CassetteServer
from .openai_session import driver_for, run_on_fake_clock, run_turns

CASSETTE = Path(__file__).parent / "cassettes" / "openai_realtime-timed-reply.jsonl"


def test_the_driver_receives_each_message_when_the_fake_clock_reads_its_time():
    clock = FakeClock()
    with CassetteServer(ws_cassette.read(CASSETTE), clock) as server:
        driver = driver_for(server.port, clock, UNTRACED)
        deltas = run_on_fake_clock(clock, run_turns(driver, 1))
    # Every send, the close at 1.0 s included, happened at its recorded time.
    assert server.finish() == []
    assert deltas == [('{"action": ', 0.5), ('"wait"}', 0.8)]


def test_a_run_leaves_no_task_pending():
    """A task left pending when the loop closes is destroyed pending, which asyncio logs."""
    left = []

    async def leaves_one():
        left.append(asyncio.ensure_future(asyncio.Event().wait()))

    run_on_fake_clock(FakeClock(), leaves_one())
    assert left[0].cancelled()


def test_a_replay_stuck_in_real_time_fails_instead_of_hanging():
    """A coroutine busy in real time over and over keeps the fake clock still and, as things keep
    changing on it, from finding the stall; the real-time limit cancels the run instead."""
    clock = FakeClock()

    async def spinning():
        while True:
            await asyncio.to_thread(time.sleep, 0.01)  # a thread the clock started, busy

    failures = []

    def run():
        try:
            run_on_fake_clock(clock, spinning(), real_limit_s=1.0)
        except AssertionError as exc:
            failures.append(str(exc))

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(10.0)
    assert failures == ["the replay made no progress for 1 real second(s)"]
