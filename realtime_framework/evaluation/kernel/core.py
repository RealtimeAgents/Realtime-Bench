"""The event kernel: a state machine that does no I/O.

The shell posts events and calls ``step`` with the current clock readings; each step handles the
next due event or timer and returns the commands to run, in order. An interaction policy (turn
based, or later streaming) decides the cadence; the kernel numbers the episode log, keeps the one
generation in flight, judges stale results, and owns the end: the first end signal fixes the cause,
freezes the snapshot and interrupts the generation, and once the policy has settled the decision in
hand the kernel writes the termination record (phase2.md, 事件内核; 时间和结束判定).

A policy provides ``merged_clocks``, ``settled``, ``start(kernel)``, ``on_event(event)``,
``on_result(result)``, ``on_timer(handle, timer)`` and ``on_end()``.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.kernel.agenda import Agenda, Inbox
from realtime_framework.evaluation.kernel.events import (
    AbortGeneration,
    AcceptResult,
    AgentRecord,
    Clock,
    DiscardResult,
    Entry,
    Finish,
    FreezeSnapshot,
    GenerationFinished,
    Invalidated,
    Priority,
    Record,
    ReleaseInputs,
    SetTimer,
    SnapshotFrozen,
    StartGeneration,
    Termination,
    Timer,
)

# Stop reasons of the end causes that are not the environment's own (runner.py:791-805).
STOP_REASONS = {
    "invalid": "invalid_run",
    "think_budget": "think_budget",
    "deadline": "time_limit",
    "horizon": "tick_limit",
}


@dataclass(frozen=True)
class Ending:
    """The first end signal: its cause, the task's reason and the semantic time by rule."""

    cause: str
    reason: str | None
    at: float
    ops: float


class Kernel:
    """Single-writer state machine. Only ``post`` may be called from another thread."""

    def __init__(self, policy) -> None:
        self.policy = policy
        self.inbox = Inbox()
        self.agenda = Agenda(merged=policy.merged_clocks)
        self.semantic = 0.0  # clock readings of the step being handled
        self.ops = 0.0
        self.generation: int | None = None  # the generation in flight
        self.ending: Ending | None = None
        self.invalid: Invalidated | None = None
        self.termination: Termination | None = None
        self._snapshot: tuple[float, float] | None = None
        self._snapshot_done = False
        self._seq = 0
        self._out: list = []
        self._generations = itertools.count(1)
        self._requests = itertools.count(1)

    # ---- the shell's side -------------------------------------------------------------------

    def post(self, event) -> None:
        self.inbox.post(event)

    def start(self, semantic: float, ops: float | None = None) -> list:
        self._read(semantic, ops)
        self.policy.start(self)
        return self._flush()

    def step(self, semantic: float, ops: float | None = None) -> list | None:
        """Handle the next due event or timer; None when nothing is due yet."""
        self._read(semantic, ops)
        for event in self.inbox.take():
            self.agenda.add(Clock.SEMANTIC, self.semantic, event.priority, event)
        due = self.agenda.pop({Clock.SEMANTIC: self.semantic, Clock.OPS: self.ops})
        if due is None:
            return None
        handle, item = due
        if isinstance(item, Timer):
            self.record(
                "TimerFired", timer=handle, kind=item.kind, clock=item.clock.value, at=item.at
            )
            if self.termination is None:
                self.policy.on_timer(handle, item)
        else:
            self._log(item.kind if isinstance(item, AgentRecord) else type(item).__name__, item)
            if self.termination is None:
                self._dispatch(item)
        if (
            self.termination is None
            and self.ending is not None
            and self._snapshot_done
            and self.policy.settled
        ):
            self._finish()
        return self._flush()

    # ---- the policy's side ------------------------------------------------------------------

    def emit(self, command) -> None:
        self._out.append(command)

    def record(self, kind: str, /, **data) -> None:
        self._log(kind, data)

    def request(self) -> int:
        """A fresh id to match an answer to its command."""
        return next(self._requests)

    def set_timer(self, kind: str, clock: Clock, at: float, priority: Priority, /, **data) -> int:
        """Record the intended wake-up first, then ask the shell for it."""
        timer = Timer(kind, clock, at, data)
        handle = self.agenda.add(clock, at, priority, timer, timer=True)
        self.record("TimerSet", timer=handle, kind=kind, clock=clock.value, at=at)
        self.emit(SetTimer(clock, at))
        return handle

    def cancel_timer(self, handle: int | None) -> None:
        if handle is not None:
            self.agenda.cancel(handle)

    def start_generation(self, decision: int, observation_id, observation_tick, remaining) -> int:
        generation = next(self._generations)
        self.generation = generation
        self.record(
            TraceEvent.INFERENCE_STARTED.value,
            decision=decision,
            generation=generation,
            observation_id=observation_id,
            observation_tick=observation_tick,
            remaining_clock_tokens=remaining,
        )
        self.emit(StartGeneration(generation, decision, observation_id, remaining))
        return generation

    def abandon_generation(self, reason: str) -> None:
        """Write that the generation is given up on before interrupting it."""
        generation = self.generation
        if generation is None:
            return
        self.generation = None
        self.record("GenerationAbandoned", generation=generation, reason=reason)
        self.emit(AbortGeneration(generation))

    def invalidate(self, category: str, reason: str, detail=None) -> None:
        """Latch the first non-model failure; it ends the episode and the run is never scored."""
        if self.invalid is None:
            # The end of a traceback names the exception (runner.py:205-207).
            self.invalid = Invalidated(
                category, reason, None if detail is None else str(detail)[-2000:]
            )
            self.record(TraceEvent.RUN_INVALIDATED.value, **asdict(self.invalid))
        self.end("invalid", self.semantic)

    def end(self, cause: str, at: float, reason: str | None = None) -> None:
        """The first end signal fixes the cause, freezes the snapshot, interrupts the generation
        and stops every timer. Later signals change nothing."""
        if self.ending is not None:
            return
        self.ending = Ending(cause, reason, at, self.ops)
        self.record("EpisodeEnded", cause=cause, reason=reason, at=at)
        self.agenda.cancel_timers()
        self.emit(FreezeSnapshot())
        self.abandon_generation(cause)
        self.policy.on_end()

    # ---- internals --------------------------------------------------------------------------

    def _read(self, semantic: float, ops: float | None) -> None:
        self.semantic = float(semantic)
        self.ops = self.semantic if ops is None else float(ops)

    def _log(self, kind: str, data) -> None:
        self._seq += 1
        self._out.append(Record(Entry(self._seq, self.semantic, self.ops, kind, data)))

    def _flush(self) -> list:
        out, self._out = self._out, []
        return out

    def _dispatch(self, event) -> None:
        if isinstance(event, Invalidated):
            self.invalidate(event.category, event.reason, event.detail)
        elif isinstance(event, GenerationFinished):
            if event.generation != self.generation:
                self.record("ResultDiscarded", generation=event.generation)
                self.emit(DiscardResult(event.generation))
                return
            self.generation = None
            self.emit(AcceptResult(event.generation))
            self.policy.on_result(event.result)
        elif isinstance(event, SnapshotFrozen):
            self._snapshot_done = True
            if event.error:
                self.invalidate("environment", "evaluation_snapshot_failed", event.error)
            else:
                self._snapshot = (event.at, event.ops_at)
        elif not isinstance(event, AgentRecord):
            self.policy.on_event(event)

    def _finish(self) -> None:
        ending = self.ending
        if self.invalid is not None:
            stop_reason = STOP_REASONS["invalid"]
        elif ending.cause == "environment":
            stop_reason = ending.reason or "game_decided"
        else:
            stop_reason = STOP_REASONS[ending.cause]
        snapshot_at, ops_snapshot_at = self._snapshot or (None, None)
        self.termination = Termination(
            stop_reason=stop_reason,
            cause=ending.cause,
            ended_at=ending.at,
            snapshot_at=snapshot_at,
            ops_ended_at=ending.ops,
            ops_snapshot_at=ops_snapshot_at,
            invalid=self.invalid,
        )
        self.record("Termination", **asdict(self.termination))
        self.emit(ReleaseInputs())
        self.emit(Finish(self.termination))
