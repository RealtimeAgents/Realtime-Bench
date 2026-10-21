"""The execution thread (harnesses/default/execution_thread.py), on the fake clock: every driver
operation runs on its loop, which the clock made, and a caller waits for it on the clock."""

from __future__ import annotations

import asyncio
import threading

import pytest

from realtime_framework.entity.model import RequestRetry, RetryCancelled, retry_cancellation
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


def test_a_stopped_thread_refuses_work():
    clock = FakeClock()

    async def work():
        return 1

    with clock.participant():
        thread = _started(clock)
        thread.stop()
        assert thread._loop.is_closed() and not thread._thread.is_alive()
        for call in (thread.run, thread.submit):
            coroutine = work()
            with pytest.raises(RuntimeError, match="^the execution thread is not running$"):
                call(coroutine)
            assert coroutine.cr_frame is None  # closed, not left unawaited
