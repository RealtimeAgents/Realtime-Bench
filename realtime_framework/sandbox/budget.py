from __future__ import annotations

# Budget
"""Deadlines for environment operations and the sandbox calls they make.

An environment operation (executing an action, capturing a frame, reading task state, advancing
the token clock) runs synchronously on the loop thread, and the scored snapshot waits for it. It
takes one absolute deadline when it starts. Every call it makes through the CDP and guest clients
spends what is left: waiting for a lock, connecting and each receive. A message that is not the
awaited reply does not restart the timer. Task runtimes pass no budget; the clients read the
current operation's deadline here.
"""

import time as time
from contextlib import contextmanager as contextmanager
from contextvars import ContextVar as ContextVar


class BudgetExceeded(TimeoutError):
    """An environment operation or a sandbox call ran past its deadline."""


# The operation running on this thread; a new thread starts outside any operation.
_OPERATION: ContextVar[Deadline | None] = ContextVar("environment_operation", default=None)


class Deadline:
    """One absolute deadline shared by every wait of a call, never later than its operation's.

    ``seconds=None`` leaves only the operation's deadline, if any.
    """

    def __init__(self, seconds: float | None, what: str):
        self.at = None if seconds is None else time.monotonic() + seconds
        self.message = None if seconds is None else f"{what} took longer than {seconds:g} s"
        self.expired = False
        self.operation = _OPERATION.get()
        if self.operation is not None and (self.at is None or self.operation.at <= self.at):
            self.at, self.message = self.operation.at, self.operation.message
        else:
            self.operation = None

    def remaining(self) -> float | None:
        """Seconds left before the deadline; raise once it has passed."""
        if self.at is None:
            return None
        left = self.at - time.monotonic()
        if left <= 0:
            raise self.exceeded()
        return left

    def exceeded(self) -> BudgetExceeded:
        self.expired = True
        if self.operation is not None:
            self.operation.expired = True
        return BudgetExceeded(self.message)


@contextmanager
def operation_budget(name: str, seconds: float):
    """Run one environment operation within ``seconds``; running past them fails it.

    Task code may catch a call's timeout (``cdp_eval`` returns None), so an operation that returns
    after its deadline fails too.
    """
    deadline = Deadline(seconds, name)
    token = _OPERATION.set(deadline)
    try:
        yield
    finally:
        _OPERATION.reset(token)
    if deadline.expired or time.monotonic() >= deadline.at:
        raise deadline.exceeded()


def bounded(seconds: float) -> float:
    """``seconds``, cut to what is left of the current operation's budget."""
    operation = _OPERATION.get()
    return seconds if operation is None else min(seconds, operation.remaining())
