"""The native-streaming interaction policy: a live session fed frames while its actions arrive at
any time (phase2.md, native streaming).

Wall clock only, which the config insists on, so the two clocks are one. Frames are captured on
the input-frame grid and pushed into the session one at a time. A frame captured while one is being
pushed waits in a single slot and a newer one takes its place; the older one counts as dropped, as
does a grid point that comes while the capture before is still under way or passes before it can
be handled. Replies follow one another: the next starts as soon as the last is accepted, without
waiting for its actions' outcomes, and frames keep going in meanwhile. A reply hands over each of
its actions as soon as it appears, with the call it came in and the time it came off the wire.

One action executes at a time; one that arrives meanwhile waits in a single slot, and a newer one
takes its place and supersedes it. Three refusals keep an action from executing: the episode is
over or past its deadline, its reply was given up, or the frame it names cannot be found. A call
the provider cancels is withdrawn while its action waits; one that executed is recorded as it ran.
A wait ends where it arrives and goes neither into the execution lane nor into its slot. A reply
is given up only at the end, or when its watchdog finds it stuck, or the session takes no frames:
a push its watchdog finds stuck, or pushes failing in a row. Then the next reply starts in the
same step. The shell runs the abort first, which returns once the agent has cleaned the session up
and closed it, which fails a push stuck in it, and the next reply rebuilds the session from the
accepted history (phase2.md, 什么时候触发生成). A session that still takes no frame after that, or
one that takes none before the first reply, when there is none to give up, invalidates the run.
Only a watchdog finds a session stuck, so one that stops taking frames less than a watchdog before
the end, or whose rebuilt session has not had a watchdog's time when the end comes, leaves the run
valid.

The state is read at most once an environment frame, at the frame's grid point, during replies too,
and once more right after every action executed: that read settles the action, as the one that
ended the task or not, before anything else runs, so nothing executes once the task has ended but
after a failed read, which tells nothing, as on the turn-based clock (turns.py ``_end_checked``).
A capture out when the action finishes holds that read up, but it stands for one right as the
action finished: an end it finds is dated there, and the deadline waits for it (N31), as the
turn-based clock reads right then. A failed read counts as the frame's. Three failed reads in a
row invalidate the run. Every outcome, the refusals included, is committed to the agent as its
feedback.

Captures, reads and actions run on lanes of their own. A capture and an action may run at once,
so that frames keep coming during a long action (phase2.md, native streaming); a read runs only
while neither does (decision 21, 4.9), as the shell's lock by resource lets them (decision 50).
The policy keeps to that itself: what comes due while it cannot run, a grid point's read or
capture or an action arriving, waits, and goes in the order it came due once what it waits for is
answered. A read that waits for an action executing holds nothing back, as the read after that
action stands for it; a read that waits for a capture holds back what came due after it. So each
operation takes the environment when it is sent: a read samples the task, and an action presses
its keys, at the time the policy sent it, which is the time the policy judges it by; the shell
reports when an action really held the environment, and the records keep that.

The kernel freezes the snapshot at the first end signal. An action executing then finishes first,
and a capture out with it: the shell takes the snapshot next after them, bounded by the larger of
the action's budget and the capture's (N32), as the turn-based shell takes it right after the
action on its one thread (evaluation/streaming_shell.py ``_freeze``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.entity.config import RunnerTimingConfig
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.kernel.events import (
    ActionExecuted,
    Capture,
    CheckEnd,
    Clock,
    Commit,
    EndChecked,
    Execute,
    FrameCaptured,
    GenerationResult,
    Outcome,
    OutcomeCommitted,
    Priority,
)
from realtime_framework.evaluation.kernel.turns import NOT_APPLIED, READ_FAILURE_LIMIT

# Outcomes of an action that never executed, besides the end's (turns.NOT_APPLIED, invalid_run) and
# a frame that cannot be found (harnesses/default/frames.py INVALID_GROUNDING_FRAME).
SUPERSEDED = "superseded"  # a newer action took its place in the waiting slot
CANCELLED_GENERATION = "cancelled_generation"  # its reply was given up
CANCELLED_CALL = "cancelled_call"  # the provider cancelled its call while it waited


def _floor(value: float) -> int:
    return math.floor(value + 1e-9)


@dataclass(frozen=True)
class StreamingConfig:
    """One streaming episode on the wall clock."""

    duration_s: float  # the task horizon in seconds
    env_fps: float = 1.0  # the end checks' cadence
    input_fps: float = 1.0  # frames fed to the model
    # A reply that has not ended, or a frame push not answered, after this long is stuck, as is a
    # session that failed this many pushes in a row. Placeholders until the smoke tests
    # (phase2.md, 落地前要验证).
    reply_watchdog_s: float = 30.0
    push_watchdog_s: float = 10.0
    push_failure_limit: int = 3
    timing: RunnerTimingConfig = field(default_factory=RunnerTimingConfig)  # the frozen stream
    # The run's. A capture and an action run at once (decision 50), which the wall clock's
    # environment lets them; on the token clock an action moves the clock a capture reads.
    clock_mode: Literal["token", "wall"] = "wall"

    def __post_init__(self) -> None:
        # A grid of no rate divides by zero; a watchdog of no time gives replies up at one
        # instant forever. None of them can be switched off.
        for name in ("duration_s", "env_fps", "input_fps", "reply_watchdog_s", "push_watchdog_s"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, not {getattr(self, name)!r}")
        if self.push_failure_limit < 1:
            raise ValueError("push_failure_limit must be at least 1")
        if self.clock_mode != "wall":
            raise ValueError(f"native streaming runs on the wall clock, not {self.clock_mode!r}")


class _FrozenStream:
    """The same producer identity ``threshold`` times while time moves on: the stream is replaying
    one frame instead of advancing (legacy_loop.py ObservationStreamWatchdog)."""

    def __init__(self, threshold: int) -> None:
        self.threshold = max(2, int(threshold))
        self._fingerprint = None
        self._first_seen_at: float | None = None
        self._count = 0

    def observe(self, fingerprint: tuple, now: float) -> bool:
        if fingerprint != self._fingerprint:
            self._fingerprint = fingerprint
            self._first_seen_at = now
            self._count = 1
            return False
        self._count += 1
        return self._count >= self.threshold and now > self._first_seen_at


# ---- streaming's own events and commands -------------------------------------------------------


@dataclass(frozen=True)
class PushFrame:
    """Push a captured frame into the live session; answer with FramePushed."""

    observation_id: str | None


@dataclass(frozen=True)
class FramePushed:
    """The agent pushed the frame: it went out, and the provider confirmed it where the provider
    confirms items. A push that failed loses that frame only."""

    observation_id: str | None
    sent_at: float | None = None  # semantic seconds the frame went out
    error: str | None = None

    priority: ClassVar[Priority] = Priority.COMPLETION


@dataclass(frozen=True)
class ActionArrived:
    """One action of a reply, handed over as soon as it appears: a function call message, or the
    completed text its JSON is parsed from. A reply may hand over several, in order. The agent
    posts them in the order the messages came, so all before the reply's result and each before a
    cancellation of its call."""

    generation: int
    action: Action | None  # None when its frame cannot be found
    arrived_at: float  # semantic seconds the message carrying it came off the wire
    # The provider's function call id; for an action parsed from text, one the agent gives it,
    # unique within the reply, so that its outcome can be told apart.
    call_id: str | None = None
    # The frame-table entry it names, and that frame's age when it arrived (FrameTable.age).
    frame: int | None = None
    frame_age_s: float | None = None
    frame_age_frames: int | None = None
    rejection: str | None = None  # frames.INVALID_GROUNDING_FRAME when its frame cannot be found
    detail: str = ""  # why: frames.MISSING_FRAME, INVALID_FRAME_SYNTAX or FRAME_NOT_IN_TABLE

    priority: ClassVar[Priority] = Priority.COMPLETION


@dataclass(frozen=True)
class CallsCancelled:
    """The provider cancelled calls of a reply; the reply goes on. A completion like an arrival,
    so that it is handled after the arrivals of the calls it names, posted before it."""

    generation: int
    call_ids: tuple[str, ...]

    priority: ClassVar[Priority] = Priority.COMPLETION


@dataclass
class _Action:
    index: int  # in arrival order over the episode
    decision: int  # the reply it came in
    arrival: ActionArrived
    action: Action | None  # a DONE or FAIL became a wait
    protocol_failure: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    status: str | None = None
    execution_result: Any = None
    superseded_by: int | None = None


class StreamingPolicy:
    """Rules of one streaming episode; ``Kernel(StreamingPolicy(config))`` runs it."""

    merged_clocks = True

    def __init__(self, config: StreamingConfig) -> None:
        self.config = config
        self.kernel = None
        self.decisions = 0  # replies started
        self.generation: int | None = None  # the reply in flight
        self.abandoned: set[int] = set()  # replies given up
        self.actions: list[_Action] = []
        self.current: _Action | None = None  # the action executing, or the read after it out
        self.waiting: _Action | None = None  # the newest action that arrived meanwhile
        self._late: list[_Action] = []  # arrived while the end waited for a read
        self._executing = False  # its Execute is out
        self.frames_pushed = 0
        self.dropped_frames = 0
        self._decision_of: dict[int, int] = {}  # generation -> decision
        self._handed: dict[int, int] = {}  # generation -> actions it handed over
        self._calls: dict[tuple[int, str], _Action] = {}  # (generation, call id) -> its action
        self._latest: FrameCaptured | None = None  # the newest frame pushed
        self._capturing = False
        self._pushing: FrameCaptured | None = None
        self._queued: FrameCaptured | None = None  # the newest frame captured meanwhile
        self._failed_pushes = 0  # in a row
        self._rebuilt = False  # the session was given up for its frames, and none went in since
        self._asked: dict[int, float] = {}  # the end check out: request -> semantic time asked
        # The reads, captures and the action waiting that came due while they could not run,
        # in the order they came due.
        self._line: list[str] = []
        self._read_frame: int | None = None  # the environment frame read last
        self._deadline_due = False  # the deadline passed while a read asked before it was out
        self._failed_reads = 0  # state reads failed in a row
        self._reply_watchdog: int | None = None
        self._push_watchdog: int | None = None
        self._feedback: list[Outcome] = []  # resolved in this step, committed at its end
        self._commits = 0  # commits not answered yet
        self._frozen = _FrozenStream(config.timing.observation_freeze_n)

    @property
    def settled(self) -> bool:
        """Nothing is out whose answer can still change the run: an action executing, a capture
        or an end check whose failure invalidates it, or feedback. A push changes nothing then."""
        return (
            self.current is None and not self._capturing and not self._asked and not self._commits
        )

    # ---- kernel callbacks -------------------------------------------------------------------

    def start(self, kernel) -> None:
        self.kernel = kernel
        kernel.record("StreamingPolicyStarted", **asdict(self.config))
        kernel.set_timer("deadline", Clock.SEMANTIC, self.config.duration_s, Priority.CONTROL)
        self._schedule("end_check", self.config.env_fps)
        self._schedule("next_frame", self.config.input_fps)
        self._due("read")
        self._due("capture")
        self._proceed()

    def on_event(self, event) -> None:
        match event:
            case EndChecked():
                self._end_checked(event)
            case FrameCaptured():
                self._captured(event)
            case FramePushed():
                self._pushed(event)
            case ActionArrived():
                self._arrived(event)
            case CallsCancelled():
                self._calls_cancelled(event)
            case ActionExecuted():
                self._executed(event)
            case OutcomeCommitted():
                self._committed(event)
        self._report()

    def on_result(self, result: GenerationResult) -> None:
        """The reply ended. Its actions arrived on their own; ``result`` carries none, only why
        the reply, or what else it said, came to no action, or its failure. A reply that came to
        no action, or to a protocol failure, has an outcome of its own."""
        kernel = self.kernel
        generation, self.generation = self.generation, None
        kernel.cancel_timer(self._reply_watchdog)
        self._reply_watchdog = None
        if result.invalid is not None:
            invalid = result.invalid
            kernel.invalidate(invalid.category, invalid.reason, invalid.detail)
        elif not self._stopped():
            if result.protocol_failure is not None or not self._handed[generation]:
                outcome = self._outcome(
                    self._decision_of[generation],
                    "none",
                    kernel.semantic,
                    output_tokens=result.output_tokens,
                    protocol_failure=result.protocol_failure,
                )
                self._feedback.append(outcome)
            self._start_reply()
        self._report()

    def on_timer(self, handle: int, timer) -> None:
        kernel, config = self.kernel, self.config
        if timer.kind == "deadline":
            self._apply_deadline()
        elif timer.kind == "end_check":
            if not self._stopped():
                if self._frame() != self._read_frame:
                    self._due("read")
                    self._proceed()
                self._schedule("end_check", config.env_fps)
        elif timer.kind == "next_frame":
            if not self._stopped():
                # The grid points that passed before this one could be handled are skipped.
                fps = config.input_fps
                for _ in range(_floor(kernel.semantic * fps) - _floor(timer.at * fps)):
                    self._drop(None, "late")
                if self._capturing or "capture" in self._line:
                    # The capture before is still under way or waiting: this one is skipped.
                    self._drop(None, "capture_busy")
                else:
                    self._due("capture")
                    self._proceed()
                self._schedule("next_frame", fps)
        elif timer.kind == "reply_watchdog":
            self._reply_watchdog = None
            self._stuck("reply_watchdog")
        elif timer.kind == "push_watchdog":
            self._push_watchdog = None
            if not self._stopped():
                self._session_stuck(
                    "push_watchdog",
                    "frame_push_stuck",
                    f"a frame push did not answer in {config.push_watchdog_s:g}s",
                )
                if kernel.ending is None:
                    # The push is still out: the rebuilt session has as long again.
                    self._arm_push_watchdog()
        self._report()

    def on_end(self) -> None:
        """The reply in flight was given up and the action waiting will not execute; an action
        whose read was due is the episode's last, and one executing settles when its answer
        arrives."""
        if self.generation is not None:
            self.abandoned.add(self.generation)
            self.generation = None
        self._reply_watchdog = self._push_watchdog = None
        if self._queued is not None:
            self._drop(self._queued.observation_id, "ended")
            self._queued = None
        self._line.clear()
        self._deadline_due = False
        if self.current is not None and not self._executing:
            after, self.current = self.current, None
            self._resolve(after, "applied", terminal_after_action=True)
        if self.waiting is not None:
            waiting, self.waiting = self.waiting, None
            self._resolve(waiting, self._ended())
        late, self._late = self._late, []
        for a in late:
            self._resolve(a, self._ended())
        self._report()

    # ---- the environment ----------------------------------------------------------------------

    def _due(self, operation: str) -> None:
        if operation not in self._line:
            self._line.append(operation)

    def _proceed(self) -> None:
        """Start what came due, in that order, as far as what is out lets it: a read waits for
        the capture and the action out, and a capture and an action for the read out. A read
        that waits for a capture holds back what came due after it, and a capture waiting goes
        before the next read once a read answers, so that reads and captures take turns however
        slow either is; a read that waits for an action holds nothing back, so that frames keep
        coming during it. The action waiting may have gone meanwhile, withdrawn or given up. Past
        the deadline only the read after an action goes, which the end waits for."""
        if self._stopped():
            waits = self._deadline_due and self.current is not None
            if waits and not (self._asked or self._capturing):
                self._ask()
            return
        for operation in list(self._line):
            if self._asked:
                return
            if operation == "read":
                if self._executing:
                    continue
                if not self._capturing:
                    self._ask()
                return
            if operation == "capture":
                self._line.remove(operation)
                self._capturing = True
                self.kernel.emit(Capture())
            elif self.current is None:
                self._line.remove(operation)
                if self.waiting is not None:
                    waiting, self.waiting = self.waiting, None
                    self._execute(waiting)

    # ---- frames -----------------------------------------------------------------------------

    def _captured(self, event: FrameCaptured) -> None:
        self._capturing = False
        kernel = self.kernel
        # Also once the episode is over: the capture was under way at the end, and every frame
        # captured is judged, failed or frozen, as on the turn-based clock; a good one is dropped.
        if event.error is not None:
            return kernel.invalidate("environment", "observation_capture_failed", event.error)
        if self._frozen.observe(event.fingerprint, kernel.ops):
            return kernel.invalidate(
                "environment", "observation_stream_frozen", repr(event.fingerprint)
            )
        if self._stopped():
            self._drop(event.observation_id, "ended")
            return self._proceed()
        if self._pushing is None:
            self._push(event)
        else:
            if self._queued is not None:
                self._drop(self._queued.observation_id, "superseded")
            self._queued = event
        self._proceed()

    def _drop(self, observation_id: str | None, reason: str) -> None:
        self.dropped_frames += 1
        self.kernel.record(
            "FrameDropped",
            observation_id=observation_id,
            reason=reason,
            dropped=self.dropped_frames,
        )

    def _push(self, frame: FrameCaptured) -> None:
        self._pushing = frame
        self._arm_push_watchdog()
        self.kernel.emit(PushFrame(frame.observation_id))

    def _arm_push_watchdog(self) -> None:
        kernel = self.kernel
        self._push_watchdog = kernel.set_timer(
            "push_watchdog", Clock.OPS, kernel.ops + self.config.push_watchdog_s, Priority.CONTROL
        )

    def _pushed(self, event: FramePushed) -> None:
        kernel = self.kernel
        frame, self._pushing = self._pushing, None
        kernel.cancel_timer(self._push_watchdog)
        self._push_watchdog = None
        # Counted also once the episode is over: every frame captured goes in or is dropped, but
        # one that invalidates the run and one in a push still out when the run finishes.
        if event.error is None:
            self.frames_pushed += 1
        else:
            self._drop(frame.observation_id, "push_failed")
        if self._stopped():
            return
        if event.error is None:
            self._latest = frame
            self._failed_pushes, self._rebuilt = 0, False
            # The first reply waits for the first frame; later ones follow each other.
            if self.generation is None:
                self._start_reply()
        else:
            self._failed_pushes += 1
            limit = self.config.push_failure_limit
            if self._failed_pushes >= limit:
                self._failed_pushes = 0
                self._session_stuck(
                    "push_failed", "frame_push_failed", f"{limit} pushes failed: {event.error}"
                )
        if self._queued is not None and not self._stopped():
            queued, self._queued = self._queued, None
            self._push(queued)

    def _session_stuck(self, reason: str, invalid_reason: str, detail: str) -> None:
        """The session takes no frames: give its reply up, so that the next one rebuilds it. The
        run is invalid once a rebuilt session takes none either, or with no reply to give up."""
        if self._rebuilt or self.generation is None:
            return self.kernel.invalidate("provider", invalid_reason, detail)
        self._rebuilt = True
        self._stuck(reason)

    # ---- replies ----------------------------------------------------------------------------

    def _start_reply(self) -> None:
        if self._stopped():
            return
        kernel, frame = self.kernel, self._latest
        decision = self.decisions
        self.decisions += 1
        generation = kernel.start_generation(decision, frame.observation_id, frame.tick, None)
        self.generation = generation
        self._decision_of[generation] = decision
        self._handed[generation] = 0
        self._reply_watchdog = kernel.set_timer(
            "reply_watchdog", Clock.OPS, kernel.ops + self.config.reply_watchdog_s, Priority.CONTROL
        )

    def _stuck(self, reason: str) -> None:
        """Give the reply in flight up and start the next."""
        if self._stopped():
            return
        generation = self._abandon(reason)
        if generation is None:
            return
        waiting = self.waiting
        if waiting is not None and waiting.arrival.generation == generation:
            self.waiting = None
            self._resolve(waiting, CANCELLED_GENERATION)
        self._start_reply()

    def _abandon(self, reason: str) -> int | None:
        """Give the reply in flight up, if one is; its generation."""
        generation = self.generation
        if generation is None:
            return None
        kernel = self.kernel
        self.generation = None
        self.abandoned.add(generation)
        kernel.cancel_timer(self._reply_watchdog)
        self._reply_watchdog = None
        kernel.abandon_generation(reason)
        return generation

    # ---- actions ----------------------------------------------------------------------------

    def _arrived(self, event: ActionArrived) -> None:
        generation, action, protocol_failure = event.generation, event.action, None
        # There are no terminal actions here either (legacy_loop.py _decide).
        if action is not None and action.action_type in (ActionType.DONE, ActionType.FAIL):
            action, protocol_failure = Action.wait(), "unsupported_terminal_action"
        a = _Action(
            len(self.actions), self._decision_of[generation], event, action, protocol_failure
        )
        self.actions.append(a)
        self._handed[generation] += 1
        if event.call_id is not None:
            self._calls[(generation, event.call_id)] = a
        # As on the turn-based clock, a DONE or FAIL turned wait is no proposal.
        if action is not None and protocol_failure is None:
            self.kernel.record(TraceEvent.ACTION_PROPOSED.value, **self._fields(a))
        if self._stopped():
            if self.kernel.ending is None:
                # The end waits for a read; it settles this action.
                return self._late.append(a)
            return self._resolve(a, self._ended())
        if generation in self.abandoned:
            return self._resolve(a, CANCELLED_GENERATION)
        if event.rejection is not None:
            return self._resolve(a, event.rejection, event.detail)
        if action.action_type is ActionType.WAIT:
            # A wait ends where it arrives, whatever executes or waits.
            a.started_at = a.finished_at = self.kernel.semantic
            a.execution_result = {
                "duration_simulated_seconds": 0.0,
                "requested_duration_simulated_seconds": float(
                    action.parameters.get("duration", 0.0)
                ),
            }
            return self._resolve(a, "wait")
        if self.waiting is not None:
            self.waiting.superseded_by = a.index
            self._resolve(self.waiting, SUPERSEDED)
        else:
            self._due("action")
        self.waiting = a
        self._proceed()

    def _execute(self, a: _Action) -> None:
        """Run the action now: no read is out, so it takes the environment at once, and its keys
        are held from now."""
        kernel, action = self.kernel, a.action
        a.started_at = kernel.semantic
        release_at = None
        if action.action_type is ActionType.HOLD_KEY:
            # A wall hold's lease ends this long after it is pressed (sandbox/actions.py
            # GuestEnvironment._execute_wall_hold).
            duration = float(action.parameters.get("duration", 1.0))
            if duration > 0:
                release_at = a.started_at + duration
                kernel.record(
                    "KeyReleaseScheduled", decision=a.decision, index=a.index, at=release_at
                )
        self.current, self._executing = a, True
        kernel.emit(Execute(a.decision, action, release_at))

    def _executed(self, event: ActionExecuted) -> None:
        """Settle the action before an end it brings settles the one waiting, so that the agent
        hears of the two in that order."""
        kernel = self.kernel
        a, self._executing = self.current, False
        # When the action took the environment and returned, as the lane read them; a run that
        # raised before it was read has neither.
        if event.finished_at is None:
            a.finished_at = kernel.semantic
        else:
            a.started_at, a.finished_at = event.started_at, event.finished_at
        # A refusal or a failure leaves its own execution record (sandbox/actions.py
        # GuestEnvironment._execution_failed).
        a.execution_result = event.result
        if event.error is not None:
            self.current = None
            self._resolve(a, "invalid_run")
            return kernel.invalidate("environment", "action_execution_failed", event.error)
        if event.rejection is not None:
            # A refused action changed nothing to read, as on the turn-based clock.
            self.current = None
            self._resolve(a, "rejected", event.rejection)
            return self._proceed()
        if kernel.ending is not None:
            # The end came while it ran, and the snapshot waited for it: it is the last action.
            self.current = None
            return self._resolve(a, "applied", terminal_after_action=True)
        # An action can end the task at once: the read after it settles it before anything else
        # starts, once a capture out is answered (decision 31). Its answer was filed before the
        # deadline, so the deadline waits for that read (N31).
        if "read" in self._line:
            self._line.remove("read")
        self._line.insert(0, "read")
        self._proceed()

    def _calls_cancelled(self, event: CallsCancelled) -> None:
        # Past the deadline the end settles the action waiting.
        stopped = self._stopped()
        for call_id in event.call_ids:
            a = self._calls.get((event.generation, call_id))
            if a is None:
                continue  # no action arrived in this call
            if a is self.waiting and not stopped:
                self.waiting = None
                self._resolve(a, CANCELLED_CALL)
            else:
                # Executed, executing, or resolved otherwise: it stays as it is.
                self.kernel.record(
                    "CallCancelled",
                    index=a.index,
                    generation=event.generation,
                    call_id=call_id,
                    status=a.status,
                )

    def _ended(self) -> str:
        """The status of an action the end keeps from executing."""
        kernel = self.kernel
        return "invalid_run" if kernel.invalid is not None else NOT_APPLIED[kernel.ending.cause]

    def _fields(self, a: _Action) -> dict:
        arrival = a.arrival
        return {
            "index": a.index,
            "decision": a.decision,
            "generation": arrival.generation,
            "call_id": arrival.call_id,
            "action": a.action,
            "protocol_failure": a.protocol_failure,
            "arrived_at": arrival.arrived_at,
            "frame": arrival.frame,
            "frame_age_s": arrival.frame_age_s,
            "frame_age_frames": arrival.frame_age_frames,
        }

    def _resolve(
        self, a: _Action, status: str, detail: str = "", terminal_after_action: bool = False
    ) -> None:
        """The action's one outcome: recorded now, committed at the end of the step."""
        kernel = self.kernel
        a.status = status
        record = self._fields(a) | {
            "status": status,
            "detail": detail,
            "terminal_after_action": terminal_after_action,
            "started_at": a.started_at,
            "finished_at": a.finished_at,
            "execution_result": a.execution_result,
            "superseded_by": a.superseded_by,
        }
        kernel.record(TraceEvent.ACTION_RESOLVED.value, **record)
        if status in {"applied", "wait"}:
            kernel.record(TraceEvent.ACTION_APPLIED.value, **record)
        arrival = a.arrival
        self._feedback.append(
            self._outcome(
                a.decision,
                status,
                kernel.semantic if a.started_at is None else a.started_at,
                detail=detail,
                terminal_after_action=terminal_after_action,
                frame_age_s=arrival.frame_age_s,
                action=a.action,
                protocol_failure=a.protocol_failure,
                call_id=arrival.call_id,
                arrived_at=arrival.arrived_at,
            )
        )

    def _outcome(
        self,
        decision: int,
        status: str,
        landed_at: float,
        *,
        detail: str = "",
        terminal_after_action: bool = False,
        frame_age_s: float | None = None,
        output_tokens: int | None = None,
        **streaming,
    ) -> Outcome:
        """An outcome to commit; the action lands where it executed, or where it was resolved.
        The output tokens are a reply's, known only for a reply that came to no action: an action
        may be resolved before its reply ends. The fields of the turn-based pacing and of the token
        clock stay empty."""
        return Outcome(
            decision=decision,
            status=status,
            detail=detail,
            terminal_after_action=terminal_after_action,
            action_clock_position=round(landed_at * self.config.env_fps, 9),
            next_observation_tick=0.0,
            dropped_intermediate_frames=0,
            post_action_rendered=False,
            frame_age_s=frame_age_s,
            token_delay_frames=None,
            action_output_tokens=output_tokens,
            post_horizon_output_tokens=0,
            **streaming,
        )

    def _report(self) -> None:
        """Commit this step's outcomes after its commands for the world: the shell gives feedback
        on its own thread, and must not hold an execution up behind it."""
        for outcome in self._feedback:
            self._commits += 1
            self.kernel.emit(Commit(outcome))
        self._feedback.clear()

    def _committed(self, event: OutcomeCommitted) -> None:
        self._commits -= 1
        if event.error is not None:
            self.kernel.invalidate("provider", "action_commit_feedback_failed", event.error)

    # ---- end checks and the deadline --------------------------------------------------------

    def _apply_deadline(self) -> None:
        """End the episode once the deadline has passed. An answer filed before the deadline can
        be handled after it, behind a command the shell ran across it; the deadline then ends the
        episode before the answer leads to anything, but for an action's. Two reads decide first,
        and the end waits for them: one asked before the deadline that is still out, which dates
        a task's end it finds by when it was asked, as the turn-based kernel files such an answer
        (core.py ``step``); and the read after an action that finished before the deadline, due
        behind a capture out or behind the action's answer, which dates it where the action
        finished (N31). The read's budget bounds the wait, and the capture's. The reply in flight
        is given up at the deadline all the same: nothing it says past it may land."""
        kernel = self.kernel
        if kernel.ending is not None or self._deadline_due:
            return
        if kernel.semantic < self.config.duration_s:
            return
        if self._asked or (self.current is not None and not self._executing):
            self._deadline_due = True
            self._abandon("deadline")
        else:
            kernel.end("deadline", self.config.duration_s)

    def _stopped(self) -> bool:
        """Whether nothing more may be done in the world, once the deadline is applied."""
        self._apply_deadline()
        return self.kernel.ending is not None or self._deadline_due

    def _ask(self) -> None:
        """Read the state now: no capture or action is out, so the read samples the task now, the
        time its end is dated by, but for the read after an action, which stands for one as the
        action finished. It is this frame's read, and whichever came due meanwhile."""
        if "read" in self._line:
            self._line.remove("read")
        kernel = self.kernel
        self._read_frame = self._frame()
        request = kernel.request()
        self._asked[request] = kernel.semantic
        kernel.emit(CheckEnd(request))

    def _end_checked(self, event: EndChecked) -> None:
        kernel = self.kernel
        asked_at = self._asked.pop(event.request)
        # The read after an action stands for one as the action finished (N31).
        at = asked_at if self.current is None else self.current.finished_at
        # A read that fails once the episode is over still counts.
        if self._count_read(event.error, at):
            return
        if event.ended:
            return kernel.end("environment", at, reason=event.reason)
        if self._deadline_due:
            return kernel.end("deadline", self.config.duration_s)
        if kernel.ending is not None:
            return
        # No action executes while a read is out: ``current`` is the one this read follows.
        if self.current is not None:
            after, self.current = self.current, None
            self._resolve(after, "applied")
        if "capture" in self._line:
            self._line.remove("capture")
            self._line.insert(0, "capture")
        self._proceed()

    def _frame(self) -> int:
        """The environment frame the clock is in."""
        return _floor(self.kernel.semantic * self.config.env_fps)

    def _count_read(self, error: str | None, at: float) -> bool:
        """Count one state read; True when it is the failure that invalidates the run, which ends
        where that read stands, as an end a read sees does (N44)."""
        if error is None:
            self._failed_reads = 0
            return False
        self._failed_reads += 1
        if self._failed_reads < READ_FAILURE_LIMIT:
            return False
        self.kernel.invalidate("environment", "terminal_state_check_failed", error, at=at)
        return True

    def _schedule(self, kind: str, fps: float) -> None:
        """A timer at the next point after now of a grid of ``fps`` points per second."""
        kernel = self.kernel
        at = (_floor(kernel.semantic * fps) + 1) / fps
        kernel.set_timer(kind, Clock.SEMANTIC, at, Priority.START)
