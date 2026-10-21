from __future__ import annotations

# Budget
"""Deadlines for environment operations and the sandbox calls they make.

An environment operation (executing an action, sending one key event under the token clock,
capturing a frame, reading task state, advancing the token clock) runs synchronously on the loop
thread, and the scored snapshot waits for it. It takes one absolute deadline when it starts. A CDP
call made inside it spends what is left on the lock wait, the connect and each receive, and a
message that is not the awaited reply does not restart the timer; a guest request gets at most
what is left for its connect and for each read.
Waits that are not cut this way (a WebSocket send, the task runtimes' own HTTP requests, the wall
clock's lock on its key holds) are caught when the operation ends: an operation that ends after
its deadline fails. Task runtimes pass no budget; the clients read the current operation's
deadline here.

A budget holds only on a real clock: the deadline is read on the operation's clock, but the
clients wait for what is left of it as real seconds (socket, lock and request timeouts).
Production times every operation on the system clock.
"""

from contextlib import contextmanager as contextmanager
from contextvars import ContextVar as ContextVar

from realtime_framework.clock import Clock as Clock
from realtime_framework.clock import SystemClock as SystemClock


class BudgetExceeded(TimeoutError):
    """An environment operation or a sandbox call ran past its deadline."""


# The operation running on this thread; a new thread starts outside any operation.
_OPERATION: ContextVar[Deadline | None] = ContextVar("environment_operation", default=None)


class Deadline:
    """One absolute deadline shared by every wait of a call, never later than its operation's.

    ``seconds=None`` leaves only the operation's deadline, if any. ``clock`` defaults to the
    operation's clock, or the system clock outside an operation; inside one it must be the
    operation's.
    """

    def __init__(self, seconds: float | None, what: str, clock: Clock | None = None):
        self.operation = _OPERATION.get()
        # Both deadlines must be read on one clock to be compared.
        if clock is not None and self.operation is not None and clock is not self.operation.clock:
            raise ValueError(f"{what} is timed on another clock than its operation")
        self.clock = clock or (
            self.operation.clock if self.operation is not None else SystemClock()
        )
        self.at = None if seconds is None else self.clock.monotonic() + seconds
        self.message = None if seconds is None else f"{what} took longer than {seconds:g} s"
        self.expired = False
        if self.operation is not None and (self.at is None or self.operation.at <= self.at):
            self.at, self.message = self.operation.at, self.operation.message
        else:
            self.operation = None

    def remaining(self) -> float | None:
        """Seconds left before the deadline; raise once it has passed."""
        if self.at is None:
            return None
        left = self.at - self.clock.monotonic()
        if left <= 0:
            raise self.exceeded()
        return left

    def exceeded(self) -> BudgetExceeded:
        self.expired = True
        if self.operation is not None:
            self.operation.expired = True
        return BudgetExceeded(self.message)

    def overrun(self) -> bool:
        """Whether a wait ran out of this deadline or it has passed."""
        return self.expired or self.clock.monotonic() >= self.at


@contextmanager
def operation_budget(name: str, seconds: float, clock: Clock | None = None):
    """Run one environment operation within ``seconds`` of ``clock``; running past them fails it.

    Task code may catch a call's timeout and return nothing (``cdp_eval``) or raise its own error
    (a game's ``episode_over``), so an operation that ends after its deadline fails with the
    overrun as its cause, whether it returns or raises.
    """
    deadline = Deadline(seconds, name, clock)
    token = _OPERATION.set(deadline)
    try:
        yield
    except Exception as exc:
        if deadline.overrun() and str(exc) != deadline.message:
            # One line keeps both, so a truncated failure detail still names the budget.
            raise BudgetExceeded(f"{deadline.message} ({type(exc).__name__}: {exc})") from exc
        raise
    finally:
        _OPERATION.reset(token)
    if deadline.overrun():
        raise deadline.exceeded()


def bounded(seconds: float) -> float:
    """``seconds``, cut to what is left of the current operation's budget."""
    operation = _OPERATION.get()
    return seconds if operation is None else min(seconds, operation.remaining())
