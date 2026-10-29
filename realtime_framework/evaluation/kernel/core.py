"""The event kernel: a state machine that does no I/O.

The shell posts events and calls ``step`` with the current clock readings; each step handles the
next due event or timer and returns the commands to run, in order. An interaction policy (turn
based, or later streaming) decides the cadence; the kernel numbers the episode log, keeps the one
generation in flight, judges stale results, and owns the end: the first end signal fixes the cause,
freezes the snapshot and interrupts the generation, and once the policy has settled the decision in
hand the kernel writes the termination record (phase2.md, 事件内核; 时间和结束判定).

A policy provides ``merged_clocks``, ``settled``, ``totals`` (what Finish carries beside the
termination, events.py), ``start(kernel)``, ``on_event(event)``, ``on_result(result)``,
``on_timer(handle, timer)`` and ``on_end()``.

The shell answers every command that names its answer (events.py) with exactly one event, posted
before it steps again: CheckEnd, Capture, Execute, AdvanceClock, ReleaseKeys, RenderFrame, Commit
and FreezeSnapshot. A command that fails, the shell's own exception included, is answered with the
error, never left unanswered: the decision in hand settles only on its answer, and the episode
cannot finish before. StartGeneration is answered by GenerationFinished whenever the worker is
done; the kernel does not wait for one it aborted. Record, SetTimer, AbortGeneration,
AcceptResult, DiscardResult and ReleaseInputs want no answer; SetTimer asks to be stepped once its
clock reads ``at``. After Finish the kernel handles nothing more: an event it is stepped through is
only numbered into the log, which is how the shell writes the agent's last records (shell.py
_write_handed). A log entry's kind is the class name of the event handled, an AgentRecord's own
kind, or the name of a record the kernel writes, a TraceEvent value for the trace's; shell.py says
how the episode trace names each.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.kernel.agenda import Agenda, Inbox
from realtime_framework.evaluation.kernel.events import (
    END_CAUSES,
    AbortGeneration,
    AcceptResult,
    ActionExecuted,
    AgentRecord,
    Clock,
    DiscardResult,
    EndChecked,
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
        self._asked: dict[int, float] = {}  # request -> semantic reading it was made at
        self._by_arrival: set[int] = set()  # requests made ``by_arrival``, until answered

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
            # An end check's answer is filed at the reading it was asked at, the time its end is
            # dated by: a check asked before the deadline is handled before it (phase2.md,
            # 时间和结束判定). An answer that finds no end to a check asked ``by_arrival`` is
            # filed as it is taken in instead (``request``). An action that reports when it
            # finished, as one on a lane of its own does, is filed there, however late the shell
            # took its answer in.
            at = self.semantic
            if isinstance(event, EndChecked):
                if event.ended or event.request not in self._by_arrival:
                    at = self._asked.get(event.request, at)
            elif isinstance(event, ActionExecuted) and event.finished_at is not None:
                at = min(at, event.finished_at)
            self.agenda.add(Clock.SEMANTIC, at, event.priority, event)
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
            if isinstance(item, EndChecked):
                self._asked.pop(item.request, None)
                self._by_arrival.discard(item.request)
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

    def request(self, by_arrival: bool = False) -> int:
        """A fresh id to match an answer to its command. An end check asked ``by_arrival`` has an
        answer that finds no end filed at the reading the kernel takes it in, as another posted
        event is, so that it comes after what was posted before it and after a timer due by then;
        one that finds the end is filed where it was asked all the same (``step``). The read
        right before a streaming action is asked so (streaming.py ``_ask``)."""
        request = next(self._requests)
        self._asked[request] = self.semantic
        if by_arrival:
            self._by_arrival.add(request)
        return request

    def asked_at(self, request: int) -> float:
        """The semantic reading a request was made at, until its answer has been handled."""
        return self._asked[request]

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

    def start_generation(
        self,
        decision: int,
        observation_id,
        observation_tick,
        remaining,
        bound_at=None,
        deadline=None,
    ) -> int:
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
        self.emit(
            StartGeneration(generation, decision, observation_id, remaining, bound_at, deadline)
        )
        return generation

    def abandon_generation(self, reason: str) -> None:
        """Write that the generation is given up on before interrupting it."""
        generation = self.generation
        if generation is None:
            return
        self.generation = None
        self.record("GenerationAbandoned", generation=generation, reason=reason)
        why = self.invalid.reason if reason == "invalid" else reason
        self.emit(AbortGeneration(generation, why))

    def invalidate(self, category: str, reason: str, detail=None, at: float | None = None) -> None:
        """Latch the first non-model failure; it ends the episode and the run is never scored.
        The end is dated ``at``, the reading being handled unless the failure is dated otherwise,
        as a state read is by its asking."""
        if self.invalid is None:
            # The end of a traceback names the exception (legacy_loop.py _invalidate).
            self.invalid = Invalidated(
                category, reason, None if detail is None else str(detail)[-2000:]
            )
            self.record(TraceEvent.RUN_INVALIDATED.value, **asdict(self.invalid))
        self.end("invalid", self.semantic if at is None else at)

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
            stop_reason = END_CAUSES["invalid"].stop_reason
        elif ending.cause == "environment":
            stop_reason = ending.reason or "game_decided"
        else:
            stop_reason = END_CAUSES[ending.cause].stop_reason
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
        self.emit(Finish(self.termination, **self.policy.totals))
