"""What the event kernel is told, and what it asks the shell to do.

Events are facts handed to the kernel: the results of commands the shell ran, and records that
other threads post through the inbox. Commands are what the kernel wants done next. The kernel
itself never reads a clock, starts a thread or calls the environment or the model; the shell does,
and hands the results back as events (phase2.md, 事件内核).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, ClassVar

from realtime_framework.entity.actions import Action


class Clock(str, Enum):
    """The clock a timer is due by (phase2.md, 时钟)."""

    # Environment ticks on the token clock, which only the kernel moves; seconds since the episode
    # origin on the wall clock.
    SEMANTIC = "semantic"
    # Watchdogs, backoff and pacing, in seconds. On the wall clock it is the semantic clock.
    OPS = "ops"


class Priority(IntEnum):
    """Order of entries due at the same time, so that an end beats a landing at that instant."""

    CONTROL = 0  # invalidation, the environment's end, the deadline, an interruption
    COMPLETION = 1  # a generation finished, an action executed, the clock advanced
    START = 2  # the next frame, a periodic end check


@dataclass(frozen=True)
class Entry:
    """One episode-log record. ``seq`` is assigned by the kernel only, in handling order."""

    seq: int
    semantic: float
    ops: float
    kind: str
    data: Any


@dataclass(frozen=True)
class Timer:
    """A wait the kernel set: due at ``at`` on ``clock``."""

    kind: str
    clock: Clock
    at: float
    data: dict = field(default_factory=dict)


# ---- events -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Invalidated:
    """A non-model failure. The run is never scored (runner.py:171-180)."""

    category: str
    reason: str
    detail: str | None = None

    priority: ClassVar[Priority] = Priority.CONTROL


@dataclass(frozen=True)
class EndChecked:
    """The environment's answer to CheckEnd. Asking also samples a game's score."""

    request: int
    ended: bool = False
    reason: str | None = None  # the task's terminal reason, when it names one
    error: str | None = None

    @property
    def priority(self) -> Priority:
        return Priority.CONTROL if self.ended or self.error else Priority.COMPLETION


@dataclass(frozen=True)
class FrameCaptured:
    """One captured frame: its tick, its timestamp on the semantic clock and its identity."""

    observation_id: str | None = None
    tick: float = 0.0
    timestamp: float | None = None
    # Producer identity (observation id, tick, screenshot digest) for the frozen-stream check.
    fingerprint: tuple = ()
    error: str | None = None

    @property
    def priority(self) -> Priority:
        return Priority.CONTROL if self.error else Priority.COMPLETION


@dataclass(frozen=True)
class GenerationResult:
    """One generation's result as the agent holds it until the kernel accepts it."""

    action: Action | None = None
    # Complete output tokens to charge to the token clock, chosen by the shell in the runner's
    # source order (runner.py:209-231); None when the provider reported none.
    output_tokens: int | None = None
    latency: float = 0.0  # the agent's decision latency in seconds, charged on the wall clock
    protocol_failure: str | None = None
    # An unusable result: the generation's own failure, or an invalid inspection proof, in the
    # runner's order (runner.py:455-477).
    invalid: Invalidated | None = None


@dataclass(frozen=True)
class GenerationFinished:
    """The worker's result for one generation, stale unless that generation is still in flight."""

    generation: int
    result: GenerationResult

    priority: ClassVar[Priority] = Priority.COMPLETION


@dataclass(frozen=True)
class ActionExecuted:
    """The environment ran an action. On the token clock a key tap moves the tick itself."""

    decision: int
    result: Any = None  # the environment's execution record
    held_keys: tuple[str, ...] = ()  # key identities a hold_key now holds under a lease
    rejection: str | None = None  # the guest refused the action; the run stays valid
    error: str | None = None  # the environment failed; the run is invalid

    @property
    def priority(self) -> Priority:
        return Priority.CONTROL if self.error else Priority.COMPLETION


@dataclass(frozen=True)
class ClockAdvanced:
    """The environment advanced one AdvanceClock step."""

    error: str | None = None

    @property
    def priority(self) -> Priority:
        return Priority.CONTROL if self.error else Priority.COMPLETION


@dataclass(frozen=True)
class KeysReleased:
    """Keys whose lease ended were released, by ReleaseKeys or by the shell's release timer."""

    keys: tuple[str, ...]
    error: str | None = None

    @property
    def priority(self) -> Priority:
        return Priority.CONTROL if self.error else Priority.COMPLETION


@dataclass(frozen=True)
class FrameRendered:
    """The zero-time render barrier ran; ``rendered`` is whether the task committed a frame."""

    rendered: bool = True
    error: str | None = None

    @property
    def priority(self) -> Priority:
        return Priority.CONTROL if self.error else Priority.COMPLETION


@dataclass(frozen=True)
class OutcomeCommitted:
    """The agent received the decision's outcome."""

    decision: int
    error: str | None = None

    @property
    def priority(self) -> Priority:
        return Priority.CONTROL if self.error else Priority.COMPLETION


@dataclass(frozen=True)
class SnapshotFrozen:
    """The scorer's snapshot was taken; the shell's clocks read ``at`` and ``ops_at`` when the
    read returned. The shell may run other commands before the kernel handles this event."""

    at: float
    ops_at: float
    error: str | None = None

    @property
    def priority(self) -> Priority:
        return Priority.CONTROL if self.error else Priority.COMPLETION


@dataclass(frozen=True)
class AgentRecord:
    """An agent or encoder record (a request sent, a reply, a history change, a session message)
    that the kernel numbers into the episode log."""

    kind: str
    data: dict = field(default_factory=dict)
    generation: int | None = None

    priority: ClassVar[Priority] = Priority.COMPLETION


# ---- commands ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """Write one episode-log entry. Entries arrive in ``seq`` order."""

    entry: Entry


@dataclass(frozen=True)
class SetTimer:
    """Wake the kernel once ``clock`` reads ``at``."""

    clock: Clock
    at: float


@dataclass(frozen=True)
class CheckEnd:
    """Ask the environment whether the episode is over; answer with EndChecked(request)."""

    request: int


@dataclass(frozen=True)
class Capture:
    """Capture one frame; answer with FrameCaptured."""


@dataclass(frozen=True)
class StartGeneration:
    """Hand the last captured frame to the agent and start one generation on a worker."""

    generation: int
    decision: int
    observation_id: str | None
    remaining_clock_tokens: int | None  # the finite-B allowance left before the horizon


@dataclass(frozen=True)
class AbortGeneration:
    """Interrupt a generation the kernel gave up on. Its late result will be discarded."""

    generation: int


@dataclass(frozen=True)
class AcceptResult:
    """Let the agent write the generation's held result into its history."""

    generation: int


@dataclass(frozen=True)
class DiscardResult:
    """Drop a stale generation's held result without writing history."""

    generation: int


@dataclass(frozen=True)
class Execute:
    """Run one action in the environment; answer with ActionExecuted."""

    decision: int
    action: Action


@dataclass(frozen=True)
class AdvanceClock:
    """Token clock: advance the environment by ``step`` ticks; answer with ClockAdvanced."""

    step: float


@dataclass(frozen=True)
class ReleaseKeys:
    """Token clock: release keys whose lease ended at this tick; answer with KeysReleased."""

    keys: tuple[str, ...]


@dataclass(frozen=True)
class ScheduleKeyRelease:
    """Wall clock: release each key at its time on the shell's own timer, not through the kernel,
    so a synchronous operation cannot make a key stay down longer."""

    releases: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class RenderFrame:
    """Token clock: commit the action's pixels without advancing time; answer with FrameRendered."""


@dataclass(frozen=True)
class Outcome:
    """What one decision came to, for the agent's feedback and the lifecycle record."""

    decision: int
    status: str
    detail: str
    terminal_after_action: bool
    action_clock_position: float
    next_observation_tick: float
    dropped_intermediate_frames: int
    post_action_rendered: bool
    frame_age_s: float | None
    token_delay_frames: float | None
    action_output_tokens: int | None
    post_horizon_output_tokens: int


@dataclass(frozen=True)
class Commit:
    """Report the decision's outcome to the agent; answer with OutcomeCommitted."""

    outcome: Outcome


@dataclass(frozen=True)
class FreezeSnapshot:
    """Take the scorer's snapshot now; answer with SnapshotFrozen."""


@dataclass(frozen=True)
class ReleaseInputs:
    """Release every held key and mouse button, after the snapshot."""


@dataclass(frozen=True)
class Termination:
    """How the episode ended. Reports and metrics read this, not the clock after teardown."""

    stop_reason: str
    cause: str  # the first end signal
    ended_at: float  # semantic time by rule: the deadline, the horizon or the answering check
    snapshot_at: float | None  # semantic time the snapshot was taken; None when it failed
    ops_ended_at: float
    ops_snapshot_at: float | None
    invalid: Invalidated | None


@dataclass(frozen=True)
class Finish:
    """The episode is over; nothing more will be asked."""

    termination: Termination
