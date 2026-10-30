from __future__ import annotations

# Clock
"""The time of one episode. Framework code reads the time and waits only through a Clock, except
in the files pyproject.toml exempts from the clock rule (TID251) and on lines marked
``# noqa: TID251``; the timeouts it gives sockets, locks and HTTP requests are real seconds
(sandbox/budget.py).

An episode has two clocks (phase2.md, 时钟). The semantic clock is world time: deadlines, the
frame grid and action timing read it. The operational clock times watchdogs, retry backoff,
request pacing and the timestamps in records. On the wall clock both are one real clock. Under the
token clock and in pause mode the semantic clock is the environment's frame counter
(FrameClock), which only the environment advances, and the operational clock is real time.

The interface follows .NET's TimeProvider: reads (wall time and monotonic time), a blocking sleep,
a wait with a timeout for another thread's Event or Future, timers, and waits on an event loop the
clock made. A Clock also starts the threads that wait on it and makes their event loops, so that a
test clock can tell when every one of them is waiting and only then move time on.
"""

import asyncio as asyncio
import concurrent.futures as concurrent_futures
import threading as threading
import time as time
from collections.abc import Awaitable as Awaitable
from collections.abc import Callable as Callable
from dataclasses import dataclass as dataclass
from typing import Protocol as Protocol
from typing import TypeVar as TypeVar

T = TypeVar("T")


class Clock(Protocol):
    def time(self) -> float:
        """Wall time, Unix seconds."""

    def time_ns(self) -> int: ...

    def monotonic(self) -> float:
        """Seconds on a clock that never goes back; only differences mean anything."""

    def monotonic_ns(self) -> int: ...

    def sleep(self, seconds: float) -> None: ...

    def event(self) -> threading.Event:
        """An Event that wait() can wait for."""

    def wait(
        self, waitable: threading.Event | concurrent_futures.Future, timeout: float | None
    ) -> bool:
        """Block until ``waitable`` (an Event from event(), or a Future another thread completes)
        is done or ``timeout`` seconds pass; True when it is done."""

    def timer(self, delay: float, callback: Callable[[], object]):
        """Run ``callback`` on its own thread after ``delay`` seconds; the started timer has
        cancel()."""

    def start_thread(self, target: Callable[[], object], *, name: str | None = None):
        """Start a daemon thread that runs ``target`` and may wait on this clock."""

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        """An event loop whose loop.time() is this clock's monotonic time."""

    async def sleep_async(self, seconds: float) -> None:
        """asyncio.sleep on the running loop, which this clock made."""

    async def wait_for(self, awaitable: Awaitable[T], timeout: float | None) -> T:
        """asyncio.wait_for on the running loop, which this clock made."""


class SystemClock:
    """Real time: each method is the standard-library call it stands for."""

    def time(self) -> float:
        return time.time()

    def time_ns(self) -> int:
        return time.time_ns()

    def monotonic(self) -> float:
        return time.monotonic()

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def event(self) -> threading.Event:
        return threading.Event()

    def wait(
        self, waitable: threading.Event | concurrent_futures.Future, timeout: float | None
    ) -> bool:
        if isinstance(waitable, concurrent_futures.Future):
            done, _ = concurrent_futures.wait((waitable,), timeout)
            return bool(done)
        return waitable.wait(timeout)

    def timer(self, delay: float, callback: Callable[[], object]) -> threading.Timer:
        timer = threading.Timer(delay, callback)
        timer.daemon = True
        timer.start()
        return timer

    def start_thread(
        self, target: Callable[[], object], *, name: str | None = None
    ) -> threading.Thread:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        return thread

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.new_event_loop()

    async def sleep_async(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def wait_for(self, awaitable: Awaitable[T], timeout: float | None) -> T:
        return await asyncio.wait_for(awaitable, timeout)


class FrameClock:
    """World time under the token clock and in pause mode: the frames the environment has
    advanced. Only the environment moves it (GuestEnvironment._advance_clock, for the kernel's
    shell's step_clock and a key tap's frame), so the world stands still while a model generates.
    It can only be read; nothing waits on it."""

    def __init__(self, fps: float):
        self.frame_seconds = 1.0 / fps
        self.frame = 0  # in environment ticks, fractional between whole ticks

    def monotonic(self) -> float:
        return self.frame * self.frame_seconds


@dataclass(frozen=True)
class Clocks:
    semantic: Clock | FrameClock
    operational: Clock


def episode_clocks(clock_mode: str, fps: float) -> Clocks:
    """Real clocks for an episode: one clock on the wall clock, otherwise the environment's frames
    and real time."""
    if clock_mode == "token":
        return Clocks(FrameClock(fps), SystemClock())
    clock = SystemClock()
    return Clocks(clock, clock)
