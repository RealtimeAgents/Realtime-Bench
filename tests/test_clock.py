"""The episode clocks (realtime_framework/clock.py) and the fake clock tests run them on
(tests/fake_clock.py). Virtual waits of seconds take milliseconds of real time here."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time

import anyio.to_thread
import pytest

from realtime_framework.clock import FrameClock, SystemClock, episode_clocks
from realtime_framework.entity.model import TRANSIENT, RequestRetry, TransportFailure

from .fake_clock import FakeClock, FakeClockStalled, _unacknowledged, wait_delivered


def test_episode_clocks_share_one_clock_on_the_wall_clock():
    clocks = episode_clocks("wall", 1.0)
    assert isinstance(clocks.semantic, SystemClock)
    assert clocks.semantic is clocks.operational


def test_episode_clocks_read_frames_under_the_token_clock():
    clocks = episode_clocks("token", 2.0)
    assert isinstance(clocks.semantic, FrameClock)
    assert isinstance(clocks.operational, SystemClock)
    clocks.semantic.frame = 3
    assert clocks.semantic.monotonic() == 1.5


def test_system_clock_waits_for_an_event_or_a_future():
    clock = SystemClock()
    event = clock.event()
    assert clock.wait(event, 0.01) is False
    clock.timer(0.01, event.set)
    assert clock.wait(event, 5.0) is True
    loop = _running_loop(clock)
    future = asyncio.run_coroutine_threadsafe(clock.sleep_async(0.01), loop)
    assert clock.wait(future, 5.0) is True
    loop.call_soon_threadsafe(loop.stop)


def _running_loop(clock):
    """Like RealtimeRuntime's loop: the loop takes part, the thread that runs it does not."""
    loop = clock.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def test_sleep_moves_time_only_when_everyone_waits():
    clock = FakeClock(wall_origin=1000.0)
    woke = []

    def sleeper():
        clock.sleep(2.0)
        woke.append(clock.monotonic())

    with clock.participant():
        clock.start_thread(sleeper)
        time.sleep(0.05)  # the test thread is busy: virtual time stands still
        assert clock.monotonic() == 0.0 and not woke
        clock.sleep(5.0)
        assert clock.monotonic() == 5.0
        assert clock.time() == 1005.0
    assert woke == [2.0]


@pytest.mark.parametrize("made_by", ["clock", "threading"])
def test_wait_returns_when_another_participant_sets_the_event(made_by):
    clock = FakeClock()
    with clock.participant():
        event = clock.event() if made_by == "clock" else threading.Event()

        def setter():
            clock.sleep(1.25)
            event.set()

        clock.start_thread(setter)
        assert clock.wait(event, 10.0) is True
        assert clock.monotonic() == 1.25
        assert clock.wait(clock.event(), 3.0) is False
        assert clock.monotonic() == 4.25


def test_timer_fires_on_its_own_thread_unless_cancelled():
    clock = FakeClock()
    fired = []
    with clock.participant():
        clock.timer(0.5, lambda: fired.append(("a", clock.monotonic())))
        clock.timer(0.25, lambda: fired.append(("b", clock.monotonic()))).cancel()
        clock.sleep(1.0)
    assert fired == [("a", 0.5)]


def test_event_loop_runs_on_virtual_time():
    clock = FakeClock()
    with clock.participant():
        loop = _running_loop(clock)

        async def work():
            await clock.sleep_async(2.5)
            started = loop.time()
            await asyncio.to_thread(clock.sleep, 1.0)  # an executor thread takes part too
            with pytest.raises(TimeoutError):
                await clock.wait_for(asyncio.sleep(60), 0.5)
            return started, loop.time()

        future = asyncio.run_coroutine_threadsafe(work(), loop)
        assert clock.wait(future, 100.0) is True
        assert future.result() == (2.5, 4.0)
        assert clock.monotonic() == 4.0
        # anyio's worker threads would not take part; on a fake loop the executor runs the call.
        future = asyncio.run_coroutine_threadsafe(anyio.to_thread.run_sync(clock.sleep, 1.5), loop)
        assert clock.wait(future, 100.0) is True
        assert clock.monotonic() == 5.5
        loop.call_soon_threadsafe(loop.stop)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["run", "run_async"])
def test_retry_backoff_waits_on_the_clock(asynchronous):
    """RequestRetry (entity/model.py) sends each retry once its delay has passed on its clock."""
    clock = FakeClock()
    sent, delays = [], []

    def send(attempt):
        sent.append(clock.monotonic())
        if attempt < 3:
            raise ConnectionError("dropped")
        return attempt

    async def send_async(attempt):
        return send(attempt)

    def record(attempt, exc, failure, delay):
        delays.append(delay)

    retry = RequestRetry("unit", lambda exc: TransportFailure(TRANSIENT), record, clock=clock)
    with clock.participant():
        if asynchronous:
            loop = _running_loop(clock)
            future = asyncio.run_coroutine_threadsafe(retry.run_async(send_async), loop)
            assert clock.wait(future, 100.0) is True
            assert future.result() == 3
            loop.call_soon_threadsafe(loop.stop)
        else:
            assert retry.run(send) == 3
    first, second = delays
    assert first > 0 and second > 0
    assert sent == [0.0, first, first + second]


def test_bytes_the_peer_has_not_taken_count_as_in_flight():
    """The send-queue query behind the in-flight check (SO_NWRITE on macOS, TIOCOUTQ on Linux)
    sees what fills the send buffer while the peer reads nothing."""
    with socket.create_server(("127.0.0.1", 0)) as server:
        client = socket.create_connection(server.getsockname())
        peer, _ = server.accept()
        with client, peer:
            assert _unacknowledged(client.fileno()) == 0
            client.setblocking(False)
            with contextlib.suppress(BlockingIOError):
                while True:
                    client.send(b"x" * 65536)
            assert _unacknowledged(client.fileno()) > 0


def test_loopback_reply_is_never_overtaken_by_a_later_deadline():
    """An asyncio client asks a server thread that answers after 0.75 s of virtual time while the
    test waits with a 2 s timeout. Bytes in flight between participants keep time still."""
    clock = FakeClock()
    with clock.participant(), socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]

        def serve():
            assert clock.wait_readable(server)
            connection, _ = server.accept()
            with connection:
                assert connection.recv(5) == b"hello"
                clock.sleep(0.75)
                connection.sendall(b"world")
                wait_delivered(connection)

        loop = _running_loop(clock)

        async def ask():
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"hello")
            await writer.drain()
            reply = await reader.read(5)
            writer.close()
            return reply, loop.time()

        for _ in range(3):
            clock.start_thread(serve)
            future = asyncio.run_coroutine_threadsafe(ask(), loop)
            assert clock.wait(future, 2.0) is True
            reply, at = future.result()
            assert reply == b"world" and at == clock.monotonic()
        loop.call_soon_threadsafe(loop.stop)
    assert clock.monotonic() == 2.25


def test_a_wait_nothing_can_end_raises():
    clock = FakeClock()
    with clock.participant(), pytest.raises(FakeClockStalled):
        clock.wait(clock.event(), None)


def test_a_thread_that_does_not_take_part_cannot_wait():
    clock = FakeClock()
    with pytest.raises(RuntimeError, match="without taking part"):
        clock.sleep(1.0)
    failures = []

    def stray():
        try:
            clock.sleep(1.0)
        except RuntimeError as exc:
            failures.append(exc)

    thread = threading.Thread(target=stray)
    thread.start()
    thread.join()
    assert failures
