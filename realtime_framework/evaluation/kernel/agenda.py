"""The event kernel's two schedules and its inbox.

A timer belongs to one clock and comes due only by that clock's reading. The semantic schedule
holds what falls due in world time (the deadline, the next frame); the operational schedule holds
the watchdogs. An encoder's own waits (backoff, pacing) stay in the encoder, on the operational
clock, and are not filed here. Within one schedule entries run by time, then priority, then the
order they were filed in. Entries of different schedules are never compared by time, only by
priority and filing order. On the wall clock the two clocks are one, and so are the schedules.

Threads other than the kernel's only post to the inbox. The kernel takes the posted events when it
next steps and files them in the semantic schedule at the current reading (phase2.md, 事件内核).
"""

from __future__ import annotations

import heapq
import itertools
import threading
from collections import deque
from collections.abc import Mapping
from typing import Any

from realtime_framework.evaluation.kernel.events import Clock, Priority


class Inbox:
    """Thread-safe FIFO of events waiting for the kernel."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: deque[Any] = deque()

    def post(self, event: Any) -> None:
        with self._lock:
            self._events.append(event)

    def take(self) -> list[Any]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events


class Agenda:
    """Filed entries, popped in due order. ``add`` returns a handle for ``cancel``."""

    def __init__(self, merged: bool) -> None:
        self.merged = merged
        self._schedules: dict[Clock, list] = {Clock.SEMANTIC: []}
        if not merged:
            self._schedules[Clock.OPS] = []
        self._order = itertools.count()
        self._cancelled: set[int] = set()
        self._timers: set[int] = set()

    def add(self, clock: Clock, at: float, priority: Priority, item: Any, *, timer: bool = False):
        handle = next(self._order)
        schedule = self._schedules[Clock.SEMANTIC if self.merged else clock]
        heapq.heappush(schedule, (at, priority, handle, item))
        if timer:
            self._timers.add(handle)
        return handle

    def cancel(self, handle: int) -> None:
        if handle in self._timers:
            self._cancelled.add(handle)

    def cancel_timers(self) -> None:
        self._cancelled |= self._timers

    def pop(self, now: Mapping[Clock, float]) -> tuple[int, Any] | None:
        """Remove and return the next due ``(handle, item)``, or None when nothing is due."""
        due = []
        for clock, schedule in self._schedules.items():
            while schedule and schedule[0][2] in self._cancelled:
                self._discard(heapq.heappop(schedule)[2])
            if schedule and schedule[0][0] <= now[clock]:
                due.append(schedule)
        if not due:
            return None
        schedule = min(due, key=lambda entries: entries[0][1:3])
        _, _, handle, item = heapq.heappop(schedule)
        self._discard(handle)
        return handle, item

    def _discard(self, handle: int) -> None:
        self._timers.discard(handle)
        self._cancelled.discard(handle)
