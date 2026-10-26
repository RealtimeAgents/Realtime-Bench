"""How an episode ends, as either shell hands it to main() (evaluation/runner.py)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from realtime_framework.evaluation.clock import TokenAccounting
from realtime_framework.evaluation.kernel.events import Termination


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
