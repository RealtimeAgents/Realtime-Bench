"""Whether the OpenAI Realtime driver's blocking receive runs on the fake clock (stream-plan 2.6).

The driver reads with ``event = await asyncio.to_thread(self._conn.recv)``
(providers/openai_realtime.py:134). On a loop the fake clock made, that runs on a thread the clock
starts, and the SDK's recv blocks there in websockets' message queue
(openai/resources/realtime/realtime.py:822; websockets/sync/messages.py:87): the clock sees a busy
thread and never moves, so a server message due later never comes. With clocked_connection
passed through the driver's own seam, the connection class it hands the SDK as
``create_connection`` (openai_realtime.py:186-188, 327-329), the receive waits on the clock, and
the driver, its SDK and its send path run unchanged.

cassettes/openai_realtime-timed-reply.jsonl is one turn, converted from a session against the
stub of test_cassette_from_trace.py and then timed by hand: the acknowledgements are reactive, the
response starts at 0.2 s, its two text deltas arrive at 0.5 s and 0.8 s and it is done at 1.0 s,
when the driver closes. The server messages come in the order the driver read them: it reads only
while it waits for something (openai_realtime.py:103-107), first after response.create.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from pathlib import Path

import pytest

from realtime_framework.entity.records import UNTRACED
from tests.fake_clock import FakeClock

from . import ws_cassette
from .clocked_ws import CassetteServer, clocked_connection
from .openai_session import driver_for, run_on_fake_clock, run_turns

CASSETTE = Path(__file__).parent / "cassettes" / "openai_realtime-timed-reply.jsonl"


def test_the_driver_receives_each_message_when_the_fake_clock_reads_its_time():
    clock = FakeClock()
    with CassetteServer(ws_cassette.read(CASSETTE), clock, tolerance_s=0.0) as server:
        driver = driver_for(server.port, clock, UNTRACED)
        driver._ws_connection = clocked_connection(driver._ws_connection, clock)
        deltas = run_on_fake_clock(clock, run_turns(driver, 1))
    # Every send, the close at 1.0 s included, happened at its recorded time.
    assert server.finish() == []
    assert deltas == [('{"action": ', 0.5), ('"wait"}', 0.8)]


def test_without_the_clocked_connection_the_fake_clock_never_moves():
    clock = FakeClock()
    loop = clock.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        with CassetteServer(ws_cassette.read(CASSETTE), clock) as server:
            driver = driver_for(server.port, clock, UNTRACED)
            future = asyncio.run_coroutine_threadsafe(run_turns(driver, 1), loop)
            # Real seconds: on the fake clock nothing would ever time out.
            with pytest.raises(concurrent.futures.TimeoutError):
                future.result(timeout=1.0)
            assert clock.monotonic() == 0.0
            assert server.cursor.next_due(1) == 0.2  # response.created, never sent
        # Closing the server ends the connection, and with it the blocked receive.
        assert future.exception(timeout=10.0) is not None
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()
