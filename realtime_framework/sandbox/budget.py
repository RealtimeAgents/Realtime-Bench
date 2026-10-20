from __future__ import annotations

# Budget
"""Deadlines for calls into the sandbox.

A call that waits several times (for a lock, a connection, each reply) takes one absolute deadline
when it starts. Every wait spends what is left of it, so a message that is not the awaited reply
does not restart the timer.
"""

import time as time


class BudgetExceeded(TimeoutError):
    """A sandbox call ran past its deadline."""


class Deadline:
    """One absolute deadline shared by every wait of a call."""

    def __init__(self, seconds: float, what: str):
        self.at = time.monotonic() + seconds
        self.message = f"{what} took longer than {seconds:g} s"

    def remaining(self) -> float:
        """Seconds left before the deadline; raise once it has passed."""
        left = self.at - time.monotonic()
        if left <= 0:
            raise self.exceeded()
        return left

    def exceeded(self) -> BudgetExceeded:
        return BudgetExceeded(self.message)
