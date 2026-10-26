"""How an episode ends, as either shell hands it to main() (evaluation/runner.py), and what the two
shells share of writing it up, whatever threads they run it on."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.clock import TokenAccounting
from realtime_framework.evaluation.kernel.core import Kernel
from realtime_framework.evaluation.kernel.events import Entry, Invalidated, Termination


def invalidated(record: dict) -> Invalidated:
    """An invalid-run record the environment latched, as the kernel's event."""
    return Invalidated(record["category"], record["reason"], record.get("detail"))


def invalid_record(invalid: Invalidated) -> dict:
    """An invalid run as main() is handed it and writes it as RunInvalidated: the detail only when
    there is one (legacy_loop.py ``_invalidate``)."""
    record = {"category": invalid.category, "reason": invalid.reason}
    if invalid.detail:
        record["detail"] = invalid.detail
    return record


def invalid_line(invalid: Invalidated) -> str:
    """The log line of an invalid run (legacy_loop.py ``_invalidate``)."""
    detail = invalid.detail
    return f"[invalid-run] {invalid.category}:{invalid.reason}" + (f" — {detail}" if detail else "")


def only_logged(record: dict) -> str:
    """The log line of an invalid run the environment latched once the snapshot was frozen: the
    run's verdict stands, as for a failure of the cleanup's own (decisions 30, 33)."""
    detail = record.get("detail")
    return (
        f"[cleanup] {record['category']}:{record['reason']} once the snapshot was frozen, only "
        "logged" + (f" — {detail}" if detail else "")
    )


def with_unheard(termination: Termination, unheard: Invalidated | None, log) -> Termination:
    """A failure a shell found in the batch of the kernel's Finish, such as the termination
    record's own write, invalidates the run as the kernel would have, at the same end (decision
    32)."""
    if unheard is None or termination.invalid is not None:
        return termination
    # The end of a traceback names the exception (legacy_loop.py _invalidate).
    detail = None if unheard.detail is None else unheard.detail[-2000:]
    invalid = Invalidated(unheard.category, unheard.reason, detail)
    log(invalid_line(invalid))
    return replace(termination, stop_reason="invalid_run", invalid=invalid)


def escaped_termination(
    kernel: Kernel,
    semantic: float,
    ops: float,
    snapshot_at: float | None,
    failure: str,
    latched: dict | None,
) -> Termination:
    """How an episode ended whose kernel, or its shell's reading of the clocks, raised (N11), as
    a shell tells it without the kernel. The run is invalid for the first failure: the kernel's
    or the environment's when one was latched, run_exception otherwise, ``failure`` being the
    first traceback the shell latched, the summary's error. It ended at the first end signal:
    the kernel's when it had one, else the reading the failure came at."""
    ending = kernel.ending
    invalid = kernel.invalid or (
        invalidated(latched)
        if latched is not None
        # The end of a traceback names the exception (legacy_loop.py _invalidate).
        else Invalidated("runner", "run_exception", failure[-2000:])
    )
    if ending is None:
        return Termination("invalid_run", "invalid", semantic, snapshot_at, ops, None, invalid)
    return Termination(
        "invalid_run", ending.cause, ending.at, snapshot_at, ending.ops, None, invalid
    )


class TraceWriter:
    """A shell's writes to the episode trace. The first that fails invalidates the run: ``lost``
    is handed the failure, and nothing more is written."""

    def __init__(self, trace, lost: Callable[[Invalidated], None]) -> None:
        self.trace = trace
        self.failed = False
        self._lost = lost

    def emit(self, event: TraceEvent | str, payload: dict, entry: Entry | None = None) -> None:
        """Write one record under the kernel's record of it, ``entry``: its number and readings."""
        if self.trace is None or self.failed:
            return
        fields = (
            {}
            if entry is None
            else {"kernel_seq": entry.seq, "semantic": entry.semantic, "ops": entry.ops}
        )
        try:
            self.trace.emit(event, payload, **fields)
        except Exception:  # noqa: BLE001 - any write failure invalidates the run
            self.failed = True
            self._lost(Invalidated("harness", "episode_trace_write_failed", traceback.format_exc()))

    def checkpoint(self) -> None:
        """Make the trace durable, as a shell does at each outcome it reports."""
        if self.trace is None or self.failed:
            return
        try:
            self.trace.checkpoint()
        except Exception:  # noqa: BLE001 - any write failure invalidates the run
            self.failed = True
            self._lost(
                Invalidated("harness", "episode_trace_checkpoint_failed", traceback.format_exc())
            )


def record_run_exception(
    writer: TraceWriter, invalid: Invalidated, semantic: float, ops, log
) -> None:
    """The run_exception a wrap-up (N11) invalidates the run for, the first failure, recorded as
    the runner records it: logged, and written as RunInvalidated outside the kernel's numbering
    while the trace takes writes."""
    record = invalid_record(invalid)
    log(invalid_line(invalid))
    if writer.trace is not None and not writer.failed:
        writer.trace.emit(
            TraceEvent.RUN_INVALIDATED, record, kernel_seq=None, semantic=semantic, ops=ops
        )


@dataclass(frozen=True)
class EpisodeResult:
    """One episode as its shell hands it to main(), which reads nothing of it off the environment.
    A shell that raised hands nothing: main() then has only what the environment latched itself.
    """

    # Turn-based: (k, t_k, d_ticks, tau_k, action) for each decision reported. A streaming episode
    # has no turns.
    timeline: list[tuple]
    # How the episode ended, with a failure the shell found in the kernel's last batch; the
    # wrap-up's when the kernel raised (N11). None when the wrap-up got no further.
    termination: Termination | None
    # What the shell wrote of it at the end: the stop reason, the termination record in world
    # seconds to nine places (legacy_loop.py _termination) and the verdict, the environment's own
    # record of a failure it latched itself. Until the shell wrote them none, and the verdict is
    # what the environment latched.
    stop_reason: str | None
    termination_record: dict | None
    invalid_run: dict | None
    # The scorer's snapshot: whether the last read took it, what a read took and the last read's
    # error; and when that read began, Unix seconds, the presentation boundary. None when no read
    # began.
    snapshot_captured: bool
    snapshot: Any
    snapshot_error: str | None
    active_end_unix_s: float | None
    loop_error: str | None  # the first traceback the shell latched, the summary's error
    # What Finish carried for the run summary (kernel/events.py); a streaming episode counts neither.
    think_spent: float
    dropped_intermediate_frames: int
    stream_freeze_count: int  # the frames that froze the stream (kernel/turns.py _captured)
    token_accounting: TokenAccounting | None  # turn-based, once a result was settled
    # Turn-based: wait, bounded, for the workers of generations the kernel gave up on, once the
    # snapshot is frozen and before the agent is closed (shell.py _drain_abandoned).
    drain_abandoned: Callable[[], None] | None

    @property
    def stream_frozen(self) -> bool:
        return self.stream_freeze_count > 0


@dataclass
class EpisodeEnd:
    """What a shell writes of the episode's end as it goes, as EpisodeResult names it; ``judged``
    once the shell gave the verdict."""

    termination: Termination | None = None
    stop_reason: str | None = None
    termination_record: dict | None = None
    judged: bool = False
    invalid_run: dict | None = None
    snapshot_captured: bool = False
    snapshot: Any = None
    snapshot_error: str | None = None
    active_end_unix_s: float | None = None
    loop_error: str | None = None
    think_spent: float = 0.0
    dropped_intermediate_frames: int = 0
    stream_freeze_count: int = 0

    def latched(self, env) -> dict | None:
        """The run's invalid record so far: the shell's verdict once it gave it, until then what
        the environment latched itself."""
        return self.invalid_run if self.judged else getattr(env, "_invalid_run", None)

    def result(
        self, env, timeline: list[tuple], token_accounting=None, drain_abandoned=None
    ) -> EpisodeResult:
        return EpisodeResult(
            timeline=timeline,
            termination=self.termination,
            stop_reason=self.stop_reason,
            termination_record=self.termination_record,
            invalid_run=self.latched(env),
            snapshot_captured=self.snapshot_captured,
            snapshot=self.snapshot,
            snapshot_error=self.snapshot_error,
            active_end_unix_s=self.active_end_unix_s,
            loop_error=self.loop_error,
            think_spent=self.think_spent,
            dropped_intermediate_frames=self.dropped_intermediate_frames,
            stream_freeze_count=self.stream_freeze_count,
            token_accounting=token_accounting,
            drain_abandoned=drain_abandoned,
        )
