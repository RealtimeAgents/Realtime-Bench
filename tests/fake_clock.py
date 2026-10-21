"""A virtual clock for tests: time moves only while every participant waits on it.

FakeClock implements realtime_framework.clock.Clock. Its participants are the threads it starts
(start_thread, and the thread each timer callback runs on), the event loops it makes
(new_event_loop) and threads that join with ``participant()``. A participant is either busy or
waiting for a condition: an Event from event() to be set, a Future to be done, a socket to turn
readable (wait_readable), or a deadline. When every participant waits and no condition holds, the
clock jumps to the earliest deadline and wakes what is due.

An event loop waits while it sits in select with nothing ready, and while it is not running (made
and not run yet, or stopped and not closed). A thread-safe callback, a ready socket, a pending
connect or write, or bytes it sent that the peer's kernel has not acknowledged yet keep it busy:
loopback delivery is asynchronous, so a reply can be in flight while neither end is ready. Test
servers that wait with wait_readable() call wait_delivered() before they close a connection. Work
a loop hands to another thread must run on a participating thread too: the loop's default
executor does, and on a fake loop anyio.to_thread.run_sync (LiteLLM runs its end-of-stream chunk
builder there) is routed to that executor instead of anyio's own worker threads.

When every participant waits and nothing is due, nothing can happen any more. A thread that joined
with ``participant()`` and is waiting then raises FakeClockStalled instead of hanging the test.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import fcntl
import functools
import heapq
import itertools
import select
import selectors
import socket
import struct
import sys
import termios
import threading
import time
from contextlib import contextmanager

import anyio.to_thread

# A waiting thread blocks on the condition variable for at most this much real time before it
# checks again (a socket it waits for, bytes in flight). It never decides when virtual time moves.
_SOCKET_POLL_S = 0.005
_RECHECK_S = 0.05
_SO_NWRITE = 0x1024  # macOS: bytes in the send buffer the peer has not acknowledged
_anyio_run_sync = anyio.to_thread.run_sync


class FakeClockStalled(RuntimeError):
    pass


def _unacknowledged(fd: int) -> int:
    """Bytes written to socket ``fd`` that its peer has not acknowledged; 0 for other files."""
    try:
        if sys.platform == "darwin":
            sock = socket.socket(fileno=fd)
            try:
                return sock.getsockopt(socket.SOL_SOCKET, _SO_NWRITE)
            finally:
                sock.detach()
        return struct.unpack("i", fcntl.ioctl(fd, termios.TIOCOUTQ, b"\0\0\0\0"))[0]
    except OSError:
        return 0


def _fds_ready(fds: dict[int, int]) -> bool:
    """True when one of ``fds`` (fd -> selectors event mask) is ready or closed."""
    if not fds:
        return False
    poller = select.poll()
    for fd, events in fds.items():
        mask = select.POLLIN if events & selectors.EVENT_READ else 0
        mask |= select.POLLOUT if events & selectors.EVENT_WRITE else 0
        poller.register(fd, mask)
    return bool(poller.poll(0))


def wait_delivered(sock) -> None:
    """Busy until the peer's kernel holds every byte written to ``sock``, or the peer has closed
    the connection and never will (a client that gave up on a late reply)."""
    while _unacknowledged(sock.fileno()) > 0:
        if _fds_ready({sock.fileno(): selectors.EVENT_READ}):
            try:
                if not sock.recv(1, socket.MSG_PEEK):
                    return
            except OSError:
                return
        time.sleep(0.0001)


class _Waiter:
    """A participating thread: busy, or waiting until ``ready()`` or ``deadline``. While
    ``in_flight()`` holds it keeps time still without being woken."""

    def __init__(self, name: str, *, primary: bool = False):
        self.name = name
        self.primary = primary
        self.waiting = False
        self.ready = None
        self.in_flight = None
        self.deadline = None


class _LoopState:
    """A participating event loop: busy, or waiting in select until a socket is ready, a
    thread-safe callback arrives (``woken``) or ``deadline``. A loop outside run_forever can do
    nothing by itself and counts as waiting, unless a callback arrived before it first ran: a
    thread is about to run it."""

    def __init__(self, name: str):
        self.name = name
        self.running = False
        self.ran = False
        self.waiting = False
        self.woken = False
        self.deadline = None
        self.selector: selectors.BaseSelector | None = None
        self.loop: _FakeLoop | None = None

    def io_pending(self) -> bool:
        """A socket is ready, a connect or write is pending, or sent bytes are in flight."""
        fds = {key.fd: key.events for key in self.selector.get_map().values()}
        return (
            any(events & selectors.EVENT_WRITE for events in fds.values())
            or _fds_ready(fds)
            or any(_unacknowledged(fd) > 0 for fd in fds)
        )


class _Event(threading.Event):
    def __init__(self, clock: FakeClock):
        super().__init__()
        self._clock = clock

    def set(self) -> None:
        with self._clock._cond:
            super().set()
            self._clock._cond.notify_all()


class _Timer:
    def __init__(self, clock: FakeClock, callback):
        self._clock = clock
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        with self._clock._cond:
            self.cancelled = True


class _Selector(selectors.BaseSelector):
    """The loop's selector. A select that would block marks the loop waiting; the clock writes to
    the loop's self-pipe when the loop's deadline comes."""

    def __init__(self, clock: FakeClock, state: _LoopState):
        self._clock = clock
        self._state = state
        self._selector = selectors.DefaultSelector()
        state.selector = self._selector

    def register(self, fileobj, events, data=None):
        return self._selector.register(fileobj, events, data)

    def unregister(self, fileobj):
        return self._selector.unregister(fileobj)

    def modify(self, fileobj, events, data=None):
        return self._selector.modify(fileobj, events, data)

    def get_key(self, fileobj):
        return self._selector.get_key(fileobj)

    def get_map(self):
        return self._selector.get_map()

    def close(self):
        self._selector.close()

    def select(self, timeout=None):
        ready = self._selector.select(0)
        if ready or (timeout is not None and timeout <= 0):
            self._clock._loop_runs(self._state)
            return ready
        self._clock._loop_waits(self._state, timeout)
        try:
            return self._selector.select(None)
        finally:
            self._clock._loop_runs(self._state)


class _Executor(concurrent.futures.ThreadPoolExecutor):
    """The loop's default executor (asyncio.to_thread, getaddrinfo): one participating thread per
    call, which completes the future before it leaves."""

    def __init__(self, clock: FakeClock):
        super().__init__(max_workers=1)
        self._clock = clock

    def submit(self, fn, /, *args, **kwargs):
        future = concurrent.futures.Future()

        def run():
            if not future.set_running_or_notify_cancel():
                return
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - delivered through the future
                future.set_exception(exc)
            else:
                future.set_result(result)

        self._clock.start_thread(run, name="fake-clock-executor")
        return future


class _FakeLoop(asyncio.SelectorEventLoop):
    def __init__(self, clock: FakeClock, state: _LoopState):
        self._fake_clock = clock
        self._fake_state = state
        super().__init__(_Selector(clock, state))
        self.set_default_executor(_Executor(clock))
        state.loop = self

    def time(self) -> float:
        return self._fake_clock.monotonic()

    def run_forever(self) -> None:
        self._fake_clock._loop_starts(self._fake_state)
        try:
            super().run_forever()
        finally:
            self._fake_clock._loop_stops(self._fake_state)

    def call_soon_threadsafe(self, callback, *args, context=None):
        self._fake_clock._wake(self._fake_state)
        return super().call_soon_threadsafe(callback, *args, context=context)

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._fake_clock._remove_loop(self._fake_state)


async def _run_sync(func, *args, abandon_on_cancel=False, cancellable=None, limiter=None):
    """anyio.to_thread.run_sync, on a fake loop through the loop's default executor."""
    loop = asyncio.get_running_loop()
    if not isinstance(loop, _FakeLoop):
        return await _anyio_run_sync(
            func,
            *args,
            abandon_on_cancel=abandon_on_cancel,
            cancellable=cancellable,
            limiter=limiter,
        )
    context = contextvars.copy_context()
    return await loop.run_in_executor(None, functools.partial(context.run, func, *args))


class FakeClock:
    def __init__(self, wall_origin: float = 0.0):
        self._cond = threading.Condition(threading.RLock())
        self._now = 0.0
        self._wall_origin = wall_origin
        self._local = threading.local()
        self._waiters: list[_Waiter] = []
        self._loops: list[_LoopState] = []
        self._timers: list[tuple[float, int, _Timer]] = []
        self._sequence = itertools.count()
        self._stalled: FakeClockStalled | None = None

    # ---- reads ----

    def time(self) -> float:
        return self._wall_origin + self._now

    def time_ns(self) -> int:
        return round(self.time() * 1_000_000_000)

    def monotonic(self) -> float:
        return self._now

    def monotonic_ns(self) -> int:
        return round(self._now * 1_000_000_000)

    # ---- participants ----

    @contextmanager
    def participant(self, name: str = "test"):
        """The calling thread takes part until the block ends."""
        waiter = self._register(name, primary=True)
        self._local.waiter = waiter
        try:
            yield self
        finally:
            self._local.waiter = None
            self._unregister(waiter)

    def start_thread(self, target, *, name: str | None = None) -> threading.Thread:
        waiter = self._register(name or getattr(target, "__name__", "thread"))

        def run():
            self._local.waiter = waiter
            try:
                target()
            finally:
                self._unregister(waiter)

        thread = threading.Thread(target=run, name=name, daemon=True)
        thread.start()
        return thread

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        anyio.to_thread.run_sync = _run_sync
        state = _LoopState(f"loop-{next(self._sequence)}")
        with self._cond:
            self._loops.append(state)
        return _FakeLoop(self, state)

    def _register(self, name: str, *, primary: bool = False) -> _Waiter:
        waiter = _Waiter(name, primary=primary)
        with self._cond:
            self._waiters.append(waiter)
        return waiter

    def _unregister(self, waiter: _Waiter) -> None:
        with self._cond:
            self._waiters.remove(waiter)
            self._settle()

    def _current(self) -> _Waiter:
        waiter = getattr(self._local, "waiter", None)
        if waiter is None:
            raise RuntimeError(
                f"{threading.current_thread().name} waits on the fake clock without taking part"
            )
        return waiter

    # ---- waits ----

    def sleep(self, seconds: float) -> None:
        with self._cond:
            deadline = self._now + max(0.0, seconds)
        self._wait_until(None, deadline)

    def event(self) -> threading.Event:
        return _Event(self)

    def wait(self, waitable, timeout: float | None) -> bool:
        if isinstance(waitable, threading.Event):
            # Setting an Event not made by event() does not wake the waiter before its next
            # recheck, but the clock never advances while the Event is set.
            ready = waitable.is_set
        elif isinstance(waitable, concurrent.futures.Future):
            ready = waitable.done
            waitable.add_done_callback(lambda _: self._notify())
        else:
            raise TypeError("wait for a threading.Event or a concurrent Future")
        with self._cond:
            deadline = None if timeout is None else self._now + max(0.0, timeout)
        return self._wait_until(ready, deadline)

    def wait_readable(self, sock, timeout: float | None = None) -> bool:
        """For test servers: block until ``sock`` is readable (or closed), or ``timeout``."""
        fd = sock.fileno()
        with self._cond:
            deadline = None if timeout is None else self._now + max(0.0, timeout)
        return self._wait_until(
            lambda: _fds_ready({fd: selectors.EVENT_READ}),
            deadline,
            in_flight=lambda: _unacknowledged(fd) > 0,
            poll_s=_SOCKET_POLL_S,
        )

    def _wait_until(
        self, ready, deadline: float | None, *, in_flight=None, poll_s: float = _RECHECK_S
    ) -> bool:
        with self._cond:
            waiter = self._current()
            try:
                while True:
                    if ready is not None and ready():
                        return True
                    if deadline is not None and self._now >= deadline:
                        return False
                    if self._stalled is not None and waiter.primary:
                        raise self._stalled
                    waiter.waiting, waiter.ready, waiter.deadline = True, ready, deadline
                    waiter.in_flight = in_flight
                    self._settle()
                    if self._runnable(waiter) or (self._stalled is not None and waiter.primary):
                        continue
                    self._cond.wait(poll_s)
            finally:
                waiter.waiting, waiter.ready, waiter.deadline = False, None, None
                waiter.in_flight = None

    def _notify(self) -> None:
        with self._cond:
            self._cond.notify_all()

    # ---- timers ----

    def timer(self, delay: float, callback) -> _Timer:
        timer = _Timer(self, callback)
        with self._cond:
            heapq.heappush(self._timers, (self._now + max(0.0, delay), next(self._sequence), timer))
            self._fire_due_timers()
        return timer

    def _fire_due_timers(self) -> None:
        while self._timers and (self._timers[0][2].cancelled or self._timers[0][0] <= self._now):
            _, _, timer = heapq.heappop(self._timers)
            if not timer.cancelled:
                self.start_thread(timer.callback, name="fake-clock-timer")

    # ---- event loops ----

    async def sleep_async(self, seconds: float) -> None:
        self._check_loop()
        await asyncio.sleep(seconds)

    async def wait_for(self, awaitable, timeout: float | None):
        self._check_loop()
        return await asyncio.wait_for(awaitable, timeout)

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if not (isinstance(loop, _FakeLoop) and loop._fake_clock is self):
            raise RuntimeError("an async wait on the fake clock runs on a loop it did not make")

    def _wake(self, state: _LoopState) -> None:
        with self._cond:
            state.woken = True

    def _loop_starts(self, state: _LoopState) -> None:
        with self._cond:
            state.running = state.ran = True

    def _loop_stops(self, state: _LoopState) -> None:
        with self._cond:
            state.running = False
            self._settle()

    def _loop_waits(self, state: _LoopState, timeout: float | None) -> None:
        with self._cond:
            state.waiting = True
            state.deadline = None if timeout is None else self._now + timeout
            self._settle()

    def _loop_runs(self, state: _LoopState) -> None:
        with self._cond:
            state.waiting = False
            state.woken = False
            state.deadline = None

    def _remove_loop(self, state: _LoopState) -> None:
        with self._cond:
            if state in self._loops:
                self._loops.remove(state)
                self._settle()

    # ---- moving time ----

    def _runnable(self, waiter: _Waiter) -> bool:
        return (waiter.ready is not None and waiter.ready()) or (
            waiter.deadline is not None and waiter.deadline <= self._now
        )

    def _quiescent(self) -> bool:
        for waiter in self._waiters:
            if not waiter.waiting or self._runnable(waiter):
                return False
            if waiter.in_flight is not None and waiter.in_flight():
                return False
        for state in self._loops:
            if not state.running:
                if state.woken and not state.ran:
                    return False
                continue
            if not state.waiting or state.woken:
                return False
            if state.deadline is not None and state.deadline <= self._now:
                return False
            if state.io_pending():
                return False
        return True

    def _settle(self) -> None:
        """Move time to the earliest deadline if every participant waits (caller holds the lock)."""
        self._fire_due_timers()
        if self._stalled is not None or not self._quiescent():
            return
        deadlines = [w.deadline for w in self._waiters if w.deadline is not None]
        deadlines += [s.deadline for s in self._loops if s.deadline is not None]
        deadlines += [self._timers[0][0]] if self._timers else []
        if not deadlines:
            if any(waiter.primary for waiter in self._waiters):
                self._stalled = FakeClockStalled(
                    f"every participant waits and nothing is due at t={self._now:g}: "
                    + ", ".join(w.name for w in self._waiters)
                    + (", " if self._waiters and self._loops else "")
                    + ", ".join(s.name for s in self._loops)
                )
                self._cond.notify_all()
            return
        self._now = max(self._now, min(deadlines))
        self._fire_due_timers()
        for state in self._loops:
            if state.deadline is not None and state.deadline <= self._now:
                state.woken = True
                state.loop._write_to_self()
        self._cond.notify_all()
