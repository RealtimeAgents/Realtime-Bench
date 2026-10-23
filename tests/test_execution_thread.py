"""The execution thread (harnesses/default/execution_thread.py), on the fake clock: every driver
operation runs on its loop, which the clock made, and a caller waits for it on the clock."""

from __future__ import annotations

import asyncio
import threading

import pytest

from realtime_framework.entity.model import (
    FATAL,
    RequestRetry,
    RetryCancelled,
    TransportFailure,
    retry_cancellation,
)
from realtime_framework.entity.records import current_trace_scope, trace_scope
from realtime_framework.harnesses.default.execution_thread import ExecutionThread

from .fake_clock import FakeClock


def _started(clock: FakeClock) -> ExecutionThread:
    thread = ExecutionThread(clock, "execution-test")
    thread.start()
    return thread


def test_a_coroutine_runs_on_the_thread_on_the_clocks_time():
    clock = FakeClock()

    async def work():
        await clock.sleep_async(1.5)
        return threading.current_thread().name, asyncio.get_running_loop().time()

    with clock.participant():
        thread = _started(clock)
        try:
            assert thread.run(work()) == ("execution-test", 1.5)
            assert clock.monotonic() == 1.5
        finally:
            thread.stop()


def test_a_coroutine_runs_in_its_callers_context():
    """The trace scope and the retry cancellation the caller is in follow the coroutine onto the
    thread: the agent trace attributes its records by them, and request units stop by them."""
    clock = FakeClock()
    abandoned = clock.event()
    abandoned.set()

    async def work():
        async def send(attempt):
            return attempt

        try:
            retry = RequestRetry(
                "unit", lambda exc: None, lambda *args: None, cancelled=lambda *args: None
            )
            sent = await retry.run_async(send)
        except RetryCancelled:
            sent = None
        return current_trace_scope(), sent

    with clock.participant():
        thread = _started(clock)
        try:
            assert thread.run(work()) == (None, 1)
            with trace_scope(phase="generation", decision=7), retry_cancellation(abandoned):
                assert thread.run(work()) == ({"phase": "generation", "decision": 7}, None)
        finally:
            thread.stop()


def test_start_keeps_the_running_loop():
    clock = FakeClock()
    with clock.participant():
        thread = _started(clock)
        try:
            loop = thread._loop
            thread.start()
            assert thread._loop is loop
        finally:
            thread.stop()


def test_a_failure_reaches_the_caller():
    clock = FakeClock()

    async def fail():
        raise ConnectionError("dropped")

    with clock.participant():
        thread = _started(clock)
        try:
            with pytest.raises(ConnectionError, match="^dropped$"):
                thread.run(fail())
        finally:
            thread.stop()


def test_a_run_past_its_timeout_is_cancelled():
    clock = FakeClock()
    cancelled = []

    async def slow():
        try:
            await clock.sleep_async(10.0)
        except asyncio.CancelledError:
            cancelled.append(clock.monotonic())
            raise

    with clock.participant():
        thread = _started(clock)
        try:
            with pytest.raises(TimeoutError, match="exceeded 2s$"):
                thread.run(slow(), timeout=2.0)
            assert clock.monotonic() == 2.0
            # The cancellation runs on the loop; an idle run lets it through first.
            thread.run(asyncio.sleep(0))
            assert cancelled == [2.0]
        finally:
            thread.stop()


def test_submitted_coroutines_start_in_order():
    """What an abort relies on: submitted first, its coroutine starts first, so it is under way
    before the next generation's coroutine looks at the session."""
    clock = FakeClock()
    started = []

    async def step(name: str):
        started.append(name)
        await clock.sleep_async(1.0)
        return name

    with clock.participant():
        thread = _started(clock)
        try:
            first, second = thread.submit(step("abort")), thread.submit(step("ensure_ready"))
            assert clock.wait(second, 10.0) and clock.wait(first, 10.0)
            assert started == ["abort", "ensure_ready"]
            assert clock.monotonic() == 1.0
        finally:
            thread.stop()


def test_stop_ends_the_coroutines_still_pending_on_the_loop():
    """A coroutine still waiting when the thread stops is cancelled, and its cleanup runs to its
    end on the thread's loop before stop returns, a to_thread in it too. Left pending on the
    closed loop, it would be finalized wherever the garbage collector frees it, and its cleanup
    would run on that thread instead."""
    clock = FakeClock()
    cleaned = []

    async def waiting():
        try:
            await clock.sleep_async(1e9)
        finally:
            name = threading.current_thread().name
            cleaned.append((name, await asyncio.to_thread(lambda: "closed")))

    with clock.participant():
        thread = _started(clock)
        coroutine = waiting()
        thread.submit(coroutine)
        thread.run(asyncio.sleep(0))  # the coroutine is waiting
        thread.stop()
    assert cleaned == [("execution-test", "closed")]
    assert coroutine.cr_frame is None
    assert thread._loop.is_closed() and not thread._thread.is_alive()


def test_stop_leaves_a_cleanup_that_does_not_end_pending():
    """Stop waits for the cancelled coroutines at most its timeout, then stops the loop."""
    clock = FakeClock()

    async def stuck():
        try:
            await clock.sleep_async(1e9)
        finally:
            await asyncio.Event().wait()

    with clock.participant():
        thread = _started(clock)
        coroutine = stuck()
        thread.submit(coroutine)
        thread.run(asyncio.sleep(0))
        thread.stop(timeout=0.2)
    assert coroutine.cr_frame is not None
    assert thread._loop.is_closed() and not thread._thread.is_alive()
    coroutine.close()


def test_an_operation_stop_cancels_for_an_abandoned_generation_stops_its_request_unit():
    """The harness abandons the generations under way before it stops the thread, which cancels
    the operations still pending: a request unit whose operation was so cancelled stops as an
    abandoned one does, and its attempt is recorded cancelled, not failed (review-encoders 低
    14)."""
    clock = FakeClock()
    abandoned, seen, ended = clock.event(), [], clock.event()

    with clock.participant():
        thread = _started(clock)

        def send(attempt):
            return thread.run(asyncio.Event().wait())

        retry = RequestRetry(
            "unit",
            lambda exc: TransportFailure(FATAL),
            lambda attempt, exc, failure, delay: seen.append(("failed", type(exc))),
            cancelled=lambda attempt, exc: seen.append(("cancelled", type(exc))),
        )

        def generation():
            try:
                with retry_cancellation(abandoned):
                    retry.run(send)
            except BaseException as exc:  # noqa: BLE001 - the test reads it
                seen.append(("raised", type(exc)))
            finally:
                ended.set()

        clock.start_thread(generation)
        thread.run(asyncio.sleep(0))  # the operation is waiting
        abandoned.set()
        thread.stop()
        assert clock.wait(ended, 1.0)
    assert seen == [("cancelled", RetryCancelled), ("raised", RetryCancelled)]


def test_a_thread_whose_loop_stays_blocked_through_stop_takes_no_more_work():
    """A loop blocked in a synchronous call past both of stop's timeouts stops only once that
    call returns. The thread counts as stopped meanwhile, so that a later run refuses at once
    instead of waiting, for good without a timeout, on a loop that runs nothing (review-encoders
    低 14)."""
    clock = FakeClock()
    blocking, release = threading.Event(), threading.Event()

    async def block():
        blocking.set()
        release.wait()

    with clock.participant():
        thread = _started(clock)
        thread.submit(block())
        assert blocking.wait(5)
        try:
            thread.stop(timeout=0.1)
            assert not thread.running
            coroutine = asyncio.sleep(0)
            with pytest.raises(RuntimeError, match="^the execution thread is not running$"):
                thread.run(coroutine)
        finally:
            release.set()


def test_a_stopped_thread_refuses_work():
    clock = FakeClock()

    async def work():
        return 1

    with clock.participant():
        thread = _started(clock)
        assert thread.running
        thread.stop()
        assert thread._loop.is_closed() and not thread._thread.is_alive()
        assert not thread.running
        for call in (thread.run, thread.submit):
            coroutine = work()
            with pytest.raises(RuntimeError, match="^the execution thread is not running$"):
                call(coroutine)
            assert coroutine.cr_frame is None  # closed, not left unawaited
