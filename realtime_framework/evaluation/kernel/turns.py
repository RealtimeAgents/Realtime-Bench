"""The turn-based interaction policy: today's decision loop as rules on the event kernel.

One decision at a time: check for the end, capture one frame, start one generation on it, land the
returned action under the skip order, pace to the next observation, and report the outcome before
the next decision starts. The four turn-based limits follow from that order alone: at most one
generation in flight, a frame only when a generation starts, nothing sent to the model during a
generation, and the next generation only once the previous outcome is reported. Two exceptions
stay inside the generation, as today: a Decision call asks all its questions there and cannot be
interrupted between them, and Gemini Live acknowledges tool calls there; the kernel only numbers
their records and discards a result that comes back after an abort.

The state is read once more after every action the environment executed, a refused one
excepted, and otherwise not twice over the same thing (TurnPolicy._read_already). The token
clock reads the tick each step of a move comes to, and the loop top and the check before landing
read only a tick not read yet; a tap's own reads, in the environment, are the reads of the ticks
it steps to. On the wall clock the periodic check, during generations too, and the loop top read
only an environment frame not read yet, the periodic check the frame it is handled in; the check
before landing reads unless the state was read at that same reading, as real time moves on
within a frame. A periodic check an execution delays is left to the check right after the
action, or, for a refused action, which gets none, made when the refusal is handled. A check left
out for a read made already goes on once that read has answered.

The three clocks differ in five places: the bound on one call, the action time, whether the clock
moves to the action time before landing, the finite-B allowance, and the frame age. Frame times use
both rates (phase2.md, 帧率): B counts tokens per input frame, observations sit on the input-frame
grid at the first environment tick at or after each grid point, and the horizon is in environment
ticks. With both rates equal to 1 every expression reduces to the runner's.

Written from the runner's loop (legacy_loop.py _run_stream_legacy, _decide) and the environment
clock (sandbox/actions.py GuestEnvironment.wait_until_tick; legacy_environment.py
LegacyStateReads.execute), not shared with them, so that the two can be compared.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.entity.config import RunnerTimingConfig
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.kernel.events import (
    END_CAUSES,
    ActionExecuted,
    AdvanceClock,
    Capture,
    CheckEnd,
    Clock,
    ClockStepped,
    Commit,
    EndChecked,
    Execute,
    FrameCaptured,
    FrameRendered,
    GenerationResult,
    KeysReleased,
    Outcome,
    OutcomeCommitted,
    Priority,
    ReleaseKeys,
    RenderFrame,
)

# The run is invalid once this many state reads in a row have failed; a read that answers starts
# the count over (phase2.md, 时间和结束判定). A failed read short of it answers "not over".
READ_FAILURE_LIMIT = 3


@dataclass(frozen=True)
class TurnConfig:
    """One episode's clock. ``tokens_per_frame`` is None on the wall clock and when paused."""

    clock_mode: str  # "token" or "wall"
    duration_s: float  # the task horizon in world seconds
    env_fps: float = 1.0
    input_fps: float = 1.0
    tokens_per_frame: int | None = None  # B, per input frame
    think_budget_s: float | None = None  # legacy diagnostics (runner.py main)
    per_move_cap_s: float | None = None
    timing: RunnerTimingConfig = field(default_factory=RunnerTimingConfig)

    def __post_init__(self) -> None:
        if self.clock_mode not in {"token", "wall"}:
            raise ValueError(f"unsupported clock mode: {self.clock_mode!r}")
        if self.tokens_per_frame is not None and (
            type(self.tokens_per_frame) is not int or self.tokens_per_frame <= 0
        ):
            raise ValueError("tokens_per_frame must be a positive integer or None")

    @property
    def max_ticks(self) -> float:
        """The horizon T in environment ticks. Within 1e-9 of a whole tick it is that tick, where
        the environment's clock snaps to (sandbox/actions.py GuestEnvironment.wait_until_tick):
        0.17 min at 5 fps is tick 51, not 51.00000000000001."""
        ticks = self.duration_s * self.env_fps
        whole = round(ticks)
        return float(whole) if abs(ticks - whole) < 1e-9 else ticks


class _Phase(str, Enum):
    GUARD = "guard"  # the end check at the top of the loop
    CAPTURE = "capture"
    GENERATE = "generate"
    APPROACH = "approach"  # token clock: moving to the action time
    PRE_CHECK = "pre_check"  # the end check right before execution
    EXECUTE = "execute"
    WAIT = "wait"  # a wait action letting its time pass
    POST_CHECK = "post_check"  # the end check right after execution
    PACE = "pace"  # moving to the next observation
    RENDER = "render"
    COMMIT = "commit"
    HORIZON = "horizon"  # finite B with no allowance left: moving to the horizon
    SETTLED = "settled"


@dataclass
class _Decision:
    index: int
    observation_id: str | None
    t_k: float
    timestamp: float | None
    remaining: int | None = None
    timeout: int | None = None  # the timer bounding the generation
    action: Action | None = None
    protocol_failure: str | None = None
    timed_out: bool = False
    over_budget: bool = False
    output_tokens: int | None = None
    post_horizon: int = 0
    d_frames: float | None = None  # token clock: the action's delay in input frames
    d_ticks: float = 0.0
    tau: float = 0.0
    target: float = 0.0
    approach_from: float = 0.0
    landing_at: float | None = None
    execute_at: float = 0.0
    wait: tuple[float, float] | None = None  # (start position, requested seconds)
    failed: str | None = None  # the error of a step of the wait's own clock move
    status: str | None = None
    detail: str = ""
    terminal_after_action: bool = False
    execution_result: Any = None
    next_tick: float = 0.0
    post_before: float = 0.0
    rendered: bool = False
    dropped: int = 0


def _ceil(value: float) -> int:
    """The next integer, or this one when ``value`` is within 1e-9 of it (legacy_loop.py
    _whole_up)."""
    rounded = round(value)
    return int(rounded) if abs(value - rounded) < 1e-9 else math.ceil(value)


def _floor(value: float) -> int:
    return math.floor(value + 1e-9)


class _FrozenStream:
    """The same producer identity ``threshold`` times while operational time moves on: the stream
    is replaying one frame instead of advancing (legacy_loop.py ObservationStreamWatchdog)."""

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


class TurnPolicy:
    """Rules of one turn-based episode; ``Kernel(TurnPolicy(config))`` runs it."""

    def __init__(self, config: TurnConfig) -> None:
        self.config = config
        self.token = config.clock_mode == "token"
        self.merged_clocks = not self.token
        self.kernel = None
        self.phase = _Phase.SETTLED
        self.decision: _Decision | None = None
        self.decisions = 0
        self.tau_prev = -1.0
        self.spent = 0.0  # cumulative deliberation, for the legacy budget
        self.dropped_frames = 0
        # Token-clock key leases in press order: key -> release tick.
        self.leases: dict[str, float] = {}
        self._frozen = _FrozenStream(config.timing.observation_freeze_n)
        self._failed_reads = 0  # state reads failed in a row
        self._read_at: float | None = None  # the semantic reading of the last state read
        self._unanswered: int | None = None  # the last state read, until its answer is handled
        self._periodic_owed = False  # an execution delayed a periodic check, left to the next read
        self._awaited: tuple[int, Callable] | None = None  # the end check the loop waits on
        self._then: Callable | None = None  # what follows the clock move in progress
        self._target = 0.0
        self._released: Callable | None = None  # what follows the key release in progress
        self._stepping = False  # an AdvanceClock or ReleaseKeys not answered yet

    @property
    def settled(self) -> bool:
        return self.phase is _Phase.SETTLED

    @property
    def totals(self) -> dict:
        return {"think_spent": self.spent, "dropped_intermediate_frames": self.dropped_frames}

    # ---- kernel callbacks -------------------------------------------------------------------

    def start(self, kernel) -> None:
        self.kernel = kernel
        kernel.record("TurnPolicyStarted", **asdict(self.config))
        if not self.token:
            # The wall horizon is the deadline; the tick limit coincides with it (runner.py
            # run_stream).
            kernel.set_timer("deadline", Clock.SEMANTIC, self.config.duration_s, Priority.CONTROL)
            self._schedule_end_check()
        self._loop_top()

    def on_event(self, event) -> None:
        match event:
            case EndChecked():
                self._end_checked(event)
            case FrameCaptured():
                self._captured(event)
            case ActionExecuted():
                self._executed(event)
            case ClockStepped():
                self._clock_stepped(event)
            case KeysReleased():
                self._keys_released(event)
            case FrameRendered():
                self._rendered(event)
            case OutcomeCommitted():
                self._committed(event)

    def on_result(self, result: GenerationResult) -> None:
        self.kernel.cancel_timer(self.decision.timeout)
        self.decision.timeout = None
        self._accept(result, timed_out=False)

    def on_timer(self, handle: int, timer) -> None:
        kernel = self.kernel
        if timer.kind == "deadline":
            kernel.end("deadline", timer.at)
        elif timer.kind == "end_check":
            # A check a synchronous command delayed past the deadline is not asked: the deadline
            # comes first, as in the legacy loop's environment. Nor is one an execution
            # delayed: the check right after the action reads at that reading, or, for an action
            # refused, which gets none, the check made in its place (_executed).
            if kernel.semantic < self.config.duration_s:
                if self.phase is _Phase.EXECUTE:
                    self._periodic_owed = True
                else:
                    self._check_end(None)
                self._schedule_end_check()
        elif timer.kind == "generation_watchdog":
            seconds = self.config.timing.provider_watchdog_s
            kernel.invalidate(
                "provider", "generation_watchdog_timeout", f"generation exceeded {seconds:g}s"
            )
        elif timer.kind == "per_move_cap":
            # A slower move places no action this turn and costs the cap (legacy_loop.py _charge).
            self.decision.timeout = None
            kernel.abandon_generation("per_move_cap")
            self._accept(GenerationResult(latency=timer.data["bound"]), timed_out=True)
        elif timer.kind in ("wake", "landing"):
            self._moved()

    def on_end(self) -> None:
        """Settle the decision in hand once the episode is over; nothing lands after the end. A
        clock step or key release out settles it on its answer, as an execution does: another end
        can come before that answer, and a wait whose own step failed is still a failed action."""
        self._awaited = None
        self._then = None
        self._released = None
        if not self._stepping:
            self._settle()

    def _settle(self) -> None:
        phase = self.phase
        if phase in (_Phase.GUARD, _Phase.CAPTURE, _Phase.GENERATE, _Phase.HORIZON):
            self.phase = _Phase.SETTLED
        elif phase is _Phase.APPROACH:
            self._approached()
        elif phase is _Phase.PRE_CHECK:
            self._resolve(self._skipped())
        elif phase is _Phase.WAIT:
            self._waited()
        elif phase is _Phase.POST_CHECK:
            self._resolve(self.decision.status, terminal_after_action=True)
        elif phase is _Phase.PACE:
            self._paced()
        # EXECUTE, RENDER and COMMIT settle when their answer arrives.

    # ---- the loop ---------------------------------------------------------------------------

    def _loop_top(self) -> None:
        self.phase = _Phase.GUARD
        self._check_end(self._guarded)

    def _past_deadline(self) -> bool:
        """Wall clock: an answer filed before the deadline can be handled after it, behind a
        synchronous command that ran across it. Then the answer leads to nothing in the world,
        and nothing lands after the end: the deadline's timer, due by now, ends the episode once
        what is filed before it has been handled, an end a check asked before the deadline saw
        among it (phase2.md, 时间和结束判定)."""
        return not self.token and self.kernel.semantic >= self.config.duration_s

    def _guarded(self) -> None:
        if self._past_deadline():
            return
        # The horizon is tested after the end check (legacy_loop.py _run_stream_legacy).
        if self.token and self._horizon_reached():
            self.phase = _Phase.SETTLED
            return self.kernel.end("horizon", self.config.max_ticks)
        self.phase = _Phase.CAPTURE
        self.kernel.emit(Capture())

    def _captured(self, event: FrameCaptured) -> None:
        kernel, config = self.kernel, self.config
        # Also once the episode is over: the capture was under way at the end, and today's loop
        # judges every frame it captures, failed or frozen (legacy_loop.py _decide).
        if event.error is not None:
            self.phase = _Phase.SETTLED
            return kernel.invalidate("environment", "observation_capture_failed", event.error)
        if self._frozen.observe(event.fingerprint, kernel.ops):
            self.phase = _Phase.SETTLED
            return kernel.invalidate(
                "environment", "observation_stream_frozen", repr(event.fingerprint)
            )
        if self.phase is not _Phase.CAPTURE:
            return
        if self._past_deadline():
            return
        d = self.decision = _Decision(
            self.decisions, event.observation_id, event.tick, event.timestamp
        )
        if self.token and config.tokens_per_frame is not None:
            # The allowance left before the horizon is the input frames left times B, rounded
            # down within float noise, which grows with it (legacy_loop.py _decide).
            allowance = self._frames(config.max_ticks - d.t_k) * config.tokens_per_frame
            d.remaining = max(0, math.floor(allowance + max(1e-9, allowance * 1e-12)))
            if d.remaining <= 0:
                self.phase = _Phase.HORIZON
                return self._move(config.max_ticks, self._at_horizon)
        # The bound on one call: an operational watchdog on the token clock, the legacy per-move cap
        # on the wall clock, whose deadline is a timer of its own (legacy_loop.py _move_timeout).
        # The shell is told when they fall due, as it settles no result back past them.
        watchdog, cap = config.timing.provider_watchdog_s, config.per_move_cap_s
        if self.token:
            bound_at = kernel.ops + watchdog if watchdog > 0 else None
        else:
            cap = None if cap is None else max(0.1, cap)
            bound_at = None if cap is None else kernel.semantic + cap
        deadline = None if self.token else config.duration_s
        kernel.start_generation(d.index, d.observation_id, d.t_k, d.remaining, bound_at, deadline)
        self.phase = _Phase.GENERATE
        if bound_at is None:
            return
        if self.token:
            d.timeout = kernel.set_timer(
                "generation_watchdog", Clock.OPS, bound_at, Priority.CONTROL
            )
        else:
            d.timeout = kernel.set_timer(
                "per_move_cap", Clock.SEMANTIC, bound_at, Priority.CONTROL, bound=cap
            )

    def _at_horizon(self) -> None:
        self.phase = _Phase.SETTLED
        self.kernel.end("horizon", self.config.max_ticks)

    def _accept(self, result: GenerationResult, timed_out: bool) -> None:
        kernel, config, d = self.kernel, self.config, self.decision
        tokens_per_frame = config.tokens_per_frame
        action, protocol_failure = result.action, result.protocol_failure
        # The loop has no terminal actions (legacy_loop.py _decide).
        if action is not None and action.action_type in (ActionType.DONE, ActionType.FAIL):
            action, protocol_failure = Action.wait(), "unsupported_terminal_action"
        tokens = result.output_tokens
        # Output past the horizon is billed but not charged to the clock (episode_records.py
        # account_tokens).
        if d.remaining is not None and tokens is not None and tokens > d.remaining:
            d.post_horizon = tokens - d.remaining
            tokens = d.remaining
        d.output_tokens = tokens
        proposed = action is not None and protocol_failure is None
        if proposed:
            kernel.record(
                TraceEvent.ACTION_PROPOSED.value,
                decision=d.index,
                observation_id=d.observation_id,
                action=action,
                action_output_tokens=tokens,
            )
        if result.invalid is not None:
            invalid = result.invalid
            return self._unusable(
                proposed, action, invalid.category, invalid.reason, invalid.detail
            )
        if self.token and tokens_per_frame is not None and tokens is None:
            return self._unusable(
                proposed,
                action,
                "provider",
                "missing_output_token_usage",
                "finite token-clock decisions require provider or deterministic local usage",
            )
        # The action time (legacy_loop.py _apply): n tokens are n / B input frames on the token
        # clock; paused, the action lands on the observed tick; on the wall clock it lands at the
        # clock position when the result is handled.
        if self.token and tokens_per_frame is not None:
            d.d_frames = tokens / tokens_per_frame
            d.d_ticks = tokens * config.env_fps / (tokens_per_frame * config.input_fps)
            if d.post_horizon:
                # The whole response ends past the horizon, so the action lands there, though the
                # allowance, whole tokens, can fall short of it: an action whose completed response
                # lands beyond the horizon is not applied (legacy_loop.py module docstring).
                d.d_ticks = config.max_ticks - d.t_k
                d.d_frames = self._frames(d.d_ticks)
            deliberation = d.d_ticks / config.env_fps
        elif self.token:
            d.d_frames = 0.0
            deliberation = 0.0
        else:
            deliberation = max(0.0, result.latency or 0.0)
            d.d_ticks = deliberation * config.env_fps
        cap = config.per_move_cap_s
        if cap is not None and deliberation > cap:
            self.spent += cap
            action, protocol_failure, timed_out = None, None, True
        else:
            self.spent += deliberation
        d.over_budget = config.think_budget_s is not None and self.spent >= config.think_budget_s
        d.tau = d.t_k + d.d_ticks if self.token else max(self._position(), self.tau_prev)
        self.tau_prev = d.tau
        d.target = min(d.tau, config.max_ticks)
        d.action, d.protocol_failure, d.timed_out = action, protocol_failure, timed_out
        d.approach_from = self._tick()
        self.phase = _Phase.APPROACH
        if self.token:
            # Token time stood still during the generation: move it to the action time first.
            return self._move(d.target, self._approached)
        # The shell accepts the result first, in this step's commands, and that takes time: the
        # action lands at the reading after it, where the deadline and the end are judged
        # (legacy_loop.py _land_action reads the clock right before it executes).
        self._then = self._approached
        self.kernel.set_timer("landing", Clock.SEMANTIC, self.kernel.semantic, Priority.COMPLETION)

    def _unusable(self, proposed: bool, action, category, reason, detail) -> None:
        """An unusable result invalidates the run before anything is paced or reported to the agent
        (legacy_loop.py _apply; episode_records.py result_invalid). A proposed action still gets its
        outcome, where today's loop writes none: every proposed action has one here."""
        d = self.decision
        self.phase = _Phase.SETTLED
        self.kernel.invalidate(category, reason, detail)
        if proposed:
            self.kernel.record(
                TraceEvent.ACTION_RESOLVED.value,
                decision=d.index,
                observation_id=d.observation_id,
                action=action,
                status="invalid_run",
                terminal_after_action=False,
            )

    def _approached(self) -> None:
        d = self.decision
        tick = self._tick()
        self.kernel.record(
            TraceEvent.CLOCK_ADVANCED.value,
            decision=d.index,
            phase="inference_to_action",
            clock_mode=self.config.clock_mode,
            from_tick=d.approach_from,
            requested_tick=d.target,
            to_tick=tick,
            delta_ticks=tick - d.approach_from,
            output_tokens=d.output_tokens,
            tokens_per_frame=self.config.tokens_per_frame,
        )
        d.landing_at = self.kernel.semantic
        status = self._skipped()
        if status is not None:
            return self._resolve(status)
        self.phase = _Phase.PRE_CHECK
        self._check_end(self._pre_checked, before_landing=True)

    def _skipped(self) -> str | None:
        """The skip order before execution (legacy_loop.py _land_action), or None to go on."""
        d, kernel = self.decision, self.kernel
        if kernel.invalid is not None:
            return "invalid_run"
        if d.action is None:
            return "skipped_timeout" if d.timed_out else "none"
        if d.over_budget:
            return "skipped_budget"
        if kernel.ending is not None:
            return END_CAUSES[kernel.ending.cause].not_applied
        return None

    def _pre_checked(self) -> None:
        if self._past_deadline():
            return
        # The horizon ends the episode at the next loop top, as today (legacy_loop.py _land_action,
        # run_stream).
        if self.token and self._horizon_reached():
            return self._resolve(END_CAUSES["horizon"].not_applied)
        d, kernel = self.decision, self.kernel
        d.execute_at = kernel.semantic
        if d.action.action_type is ActionType.WAIT:
            return self._start_wait()
        release_at = None
        if not self.token and d.action.action_type is ActionType.HOLD_KEY:
            # A wall hold's lease ends this long after it is pressed (sandbox/actions.py
            # GuestEnvironment._execute_wall_hold).
            duration = float(d.action.parameters.get("duration", 1.0))
            if duration > 0:
                release_at = d.execute_at + duration
                kernel.record("KeyReleaseScheduled", decision=d.index, at=release_at)
        self.phase = _Phase.EXECUTE
        kernel.emit(Execute(d.index, d.action, release_at))

    def _executed(self, event: ActionExecuted) -> None:
        if self.phase is not _Phase.EXECUTE:
            return
        d, kernel = self.decision, self.kernel
        # A tap's end checks were made before the action finished, the last on the tick it
        # stopped at, which the check right after the action reads again. One that saw the end
        # ends the episode there, by the task's reason, and no check follows. A repeated tap
        # presses on after a read saw the end (sandbox/actions.py
        # GuestEnvironment._execute_clock_key), so the end is dated where the whole action
        # stopped, and the reads after it count for nothing.
        for ended, reason, error in event.end_checks:
            if self._count_read(error):
                break
            if ended:
                kernel.end("environment", kernel.semantic, reason=reason)
                break
        # A refusal or a failure leaves its own execution record (sandbox/actions.py
        # GuestEnvironment._execution_failed).
        d.execution_result = event.result
        if event.error is not None:
            kernel.invalidate("environment", "action_execution_failed", event.error)
            return self._resolve("invalid_run")
        if event.rejection is not None:
            # A refused action skips the check after execution (legacy_loop.py _land_action), but
            # not a periodic check its execution delayed, unless the episode ended meanwhile.
            d.detail = event.rejection
            if self._periodic_owed and kernel.ending is None:
                self._check_end(None)
            return self._resolve("rejected")
        if self.token and event.held_keys:
            self._lease(event.held_keys)
        self._after_execution("applied")

    def _after_execution(self, status: str) -> None:
        # An executed action can end the task at once: check before any time passes
        # (legacy_loop.py _land_action).
        self.decision.status = status
        if self.kernel.ending is not None:
            return self._resolve(status, terminal_after_action=True)
        self.phase = _Phase.POST_CHECK
        self._check_end(lambda: self._resolve(status), after_action=True)

    def _lease(self, keys: tuple[str, ...]) -> None:
        """Token clock: a hold keeps each key down until its lease ends, and a held key is only
        extended (sandbox/actions.py GuestEnvironment._execute_clock_key)."""
        d = self.decision
        duration = float(d.action.parameters.get("duration", 1.0))
        start = d.execute_at
        end = start + duration * self.config.env_fps
        for key in keys:
            self.leases[key] = max(self.leases.get(key, start), end)

    def _start_wait(self) -> None:
        """A wait action is time passing: token ticks under end checks (legacy_environment.py
        LegacyStateReads.execute), or a wall timer that any end interrupts, where today's sleep
        stops only at the deadline."""
        d = self.decision
        duration = float(d.action.parameters.get("duration", 0.0))
        d.wait = (self._position(), duration)
        self.phase = _Phase.WAIT
        if duration <= 0:
            return self._waited()
        if self.token:
            return self._move(self.kernel.semantic + duration * self.config.env_fps, self._waited)
        self._move(self.kernel.semantic + duration, self._waited, Priority.COMPLETION)

    def _waited(self) -> None:
        d, config = self.decision, self.config
        if d.failed is not None:
            # The environment reports a wait whose step failed as a failed action
            # (legacy_environment.py LegacyStateReads.execute; legacy_loop.py _land_action).
            d.execution_result = {"error": d.failed}
            return self._resolve("invalid_run")
        started, duration = d.wait
        # The wait lasts until its own time or the end, whichever comes first, though either can
        # be handled later, behind a synchronous command: today's sleep(min(duration, remaining))
        # stops at the deadline.
        ending = self.kernel.ending
        if self.token:
            due = started + duration * config.env_fps
        else:
            due = started / config.env_fps + duration
        until = min(self.kernel.semantic, due)
        if ending is not None:
            until = min(until, ending.at)
        position = until if self.token else until * config.env_fps
        elapsed = round(max(0.0, (position - started) / config.env_fps), 9)
        horizon_truncates = self.token and started + duration * config.env_fps > (
            config.max_ticks + 1e-9
        )
        by_horizon = horizon_truncates and self._horizon_reached()
        # The wall deadline cuts a wait as today's sleep does, which flags neither truncation
        # (legacy_environment.py LegacyWallClock.execute); the task's end or an invalidation cuts
        # it on either clock.
        by_deadline = ending is not None and ending.cause == "deadline"
        d.execution_result = {
            "duration_simulated_seconds": elapsed,
            "requested_duration_simulated_seconds": duration,
            "truncated_by_episode_horizon": by_horizon,
            "truncated_by_terminal_condition": (
                elapsed + 1e-9 < duration and not by_horizon and not by_deadline
            ),
        }
        self._after_execution("wait")

    def _resolve(self, status: str, terminal_after_action: bool = False) -> None:
        d, kernel = self.decision, self.kernel
        d.status, d.terminal_after_action = status, terminal_after_action
        record = {
            "decision": d.index,
            "observation_id": d.observation_id,
            "action": d.action,
            "status": status,
            "terminal_after_action": terminal_after_action,
            "action_clock_position": round(d.tau, 9),
            "environment_tick": self._tick(),
            "execution_result": d.execution_result,
        }
        kernel.record(TraceEvent.ACTION_RESOLVED.value, **record)
        if d.action is not None and status in {"applied", "wait"}:
            kernel.record(TraceEvent.ACTION_APPLIED.value, **record)
        d.next_tick = self._tick()
        # Pace whether or not an action landed, so an empty decision cannot repeat within one
        # frame; never after the end, nor from the horizon on, on either clock: on the wall clock
        # an answer filed before the deadline can be handled after it (legacy_loop.py _apply).
        if (
            not terminal_after_action
            and status not in {"not_applied_terminal", "invalid_run"}
            and kernel.ending is None
            and not self._horizon_reached()
        ):
            action_end = max(float(d.tau), float(self._tick()))
            d.next_tick = self._next_observation_tick(d.t_k, action_end)
            d.post_before = self._tick()
            self.phase = _Phase.PACE
            return self._move(self._semantic_of(d.next_tick), self._paced)
        self._commit()

    def _paced(self) -> None:
        d = self.decision
        # At an exact boundary the move was empty, so commit the action's pixels at the same tick
        # before the next capture (legacy_loop.py _apply).
        if (
            self.token
            and d.action is not None
            and d.status == "applied"
            and abs(float(self._tick()) - float(d.post_before)) < 1e-9
            and self.kernel.ending is None
        ):
            self.phase = _Phase.RENDER
            return self.kernel.emit(RenderFrame())
        self._boundary_reached()

    def _rendered(self, event: FrameRendered) -> None:
        if self.phase is not _Phase.RENDER:
            return
        if event.error is not None:
            self.kernel.invalidate("environment", "frame_render_failed", event.error)
        else:
            self.decision.rendered = event.rendered
        self._boundary_reached()

    def _boundary_reached(self) -> None:
        d = self.decision
        post_after = self._tick()
        # Every input frame the decision crossed but one was dropped (legacy_loop.py _apply).
        crossed = max(0, int(_floor(self._frames(post_after)) - _floor(self._frames(d.t_k))))
        d.dropped = max(0, crossed - 1)
        self.kernel.record(
            TraceEvent.CLOCK_ADVANCED.value,
            decision=d.index,
            phase="action_to_observation_boundary",
            clock_mode=self.config.clock_mode,
            from_tick=d.post_before,
            requested_tick=d.next_tick,
            to_tick=post_after,
            delta_ticks=post_after - d.post_before,
            post_action_rendered=d.rendered,
            dropped_intermediate_frames=d.dropped,
        )
        if self.token:
            self.dropped_frames += d.dropped
        self._commit()

    def _commit(self) -> None:
        d, config = self.decision, self.config
        # The frame age: simulated time on the token clock, wall time on the wall clock
        # (legacy_loop.py _apply).
        if d.timestamp is None:
            frame_age = None
        elif self.token:
            frame_age = max(0.0, (d.tau - d.t_k) / config.env_fps)
        else:
            frame_age = max(0.0, d.landing_at - d.timestamp)
        outcome = Outcome(
            decision=d.index,
            status=d.status,
            detail=d.detail,
            terminal_after_action=d.terminal_after_action,
            action_clock_position=round(d.tau, 9),
            next_observation_tick=d.next_tick,
            dropped_intermediate_frames=d.dropped,
            post_action_rendered=d.rendered,
            frame_age_s=frame_age,
            token_delay_frames=d.d_frames,
            action_output_tokens=d.output_tokens,
            post_horizon_output_tokens=d.post_horizon,
            action=d.action,
            protocol_failure=d.protocol_failure,
        )
        self.phase = _Phase.COMMIT
        self.kernel.emit(Commit(outcome))

    def _committed(self, event: OutcomeCommitted) -> None:
        if self.phase is not _Phase.COMMIT:
            return
        kernel, d = self.kernel, self.decision
        self.phase = _Phase.SETTLED
        self.decision = None
        self.decisions += 1
        if event.error is not None:
            kernel.invalidate("provider", "action_commit_feedback_failed", event.error)
        if kernel.ending is not None:
            return
        # The legacy budget ends the episode once the decision is reported (legacy_loop.py _decide).
        if d.over_budget:
            return kernel.end("think_budget", kernel.semantic)
        self._loop_top()

    # ---- end checks -------------------------------------------------------------------------

    def _check_end(
        self, then: Callable | None, *, after_action: bool = False, before_landing: bool = False
    ) -> None:
        """Ask whether the episode is over; ``then`` runs when it is not. A check right after an
        action always reads; any other is not asked when it would read what was read already
        (_read_already), and ``then`` runs once that read has answered: at once, or when its
        answer is handled, should another event due earlier have come first. Should a periodic
        check read again before that answer is handled, ``then`` waits for the later read's
        answer (_end_checked)."""
        if then is not None and self._past_deadline():
            return
        if not after_action and self._read_already(before_landing=before_landing):
            if then is not None and self._unanswered is not None:
                self._awaited = (self._unanswered, then)
                return None
            return None if then is None else then()
        self._read_at = self.kernel.semantic
        self._periodic_owed = False
        request = self._unanswered = self.kernel.request()
        if then is not None:
            self._awaited = (request, then)
        self.kernel.emit(CheckEnd(request))

    def _schedule_end_check(self) -> None:
        # The wall clock's world comes to a new environment frame at each of these; the state is
        # read then, during generations too, unless that frame was read already.
        fps = self.config.env_fps
        at = (_floor(self.kernel.semantic * fps) + 1) / fps
        self.kernel.set_timer("end_check", Clock.SEMANTIC, at, Priority.START)

    def _end_checked(self, event: EndChecked) -> None:
        kernel = self.kernel
        asked_at = kernel.asked_at(event.request)
        if self._unanswered == event.request:
            self._unanswered = None
        if kernel.ending is not None:
            return
        if self._count_read(event.error, asked_at):
            return
        if event.ended:
            return kernel.end("environment", asked_at, reason=event.reason)
        if self._awaited is not None and self._awaited[0] == event.request:
            then = self._awaited[1]
            self._awaited = None
            # A periodic check has read since: what waits goes on from that later read's answer,
            # which may be the end, and nothing lands after the end.
            if self._unanswered is not None:
                self._awaited = (self._unanswered, then)
                return
            then()

    def _count_read(self, error: str | None, asked_at: float | None = None) -> bool:
        """Count one state read; True when it is the failure that invalidates the run, which ends
        when that read was asked, as an end a read sees does (legacy_loop.py _end_reported)."""
        if error is None:
            self._failed_reads = 0
            return False
        self._failed_reads += 1
        if self._failed_reads < READ_FAILURE_LIMIT:
            return False
        self.kernel.invalidate("environment", "terminal_state_check_failed", error, at=asked_at)
        return True

    # ---- moving the semantic clock ----------------------------------------------------------

    def _move(self, at: float, then: Callable, priority: Priority = Priority.START) -> None:
        """Bring the semantic clock to ``at``: token ticks advanced in steps, or a wall timer."""
        self._then = then
        if self.token:
            self._target = min(float(at), self.config.max_ticks)
            return self._advance()
        self.kernel.set_timer("wake", Clock.SEMANTIC, at, priority)

    def _moved(self) -> None:
        then, self._then = self._then, None
        then()

    def _advance(self) -> None:
        """One step of at most one tick, cut where a lease ends; each step is followed by the
        release of leases that ended and an end check (sandbox/actions.py
        GuestEnvironment.wait_until_tick)."""
        tick = self._tick()
        if not tick + 1e-9 < self._target:
            return self._moved()
        step = min(1.0, self._target - tick)
        if self.leases:
            step = min(step, max(0.0, min(self.leases.values()) - tick))
            if step <= 1e-9:
                # A lease ending at this tick is released without an end check.
                return self._release(self._advance)
        self._stepping = True
        self.kernel.emit(AdvanceClock(step, escapes=self.phase is not _Phase.WAIT))

    def _clock_stepped(self, event: ClockStepped) -> None:
        if not self._stepping:
            return
        self._stepping = False
        if event.error is not None:
            return self._step_failed("clock_advance_failed", event.error)
        if self.kernel.ending is not None:
            return self._settle()
        self._release(lambda: self._check_end(self._advance))

    def _step_failed(self, reason: str, error: str) -> None:
        """A step of a token-clock move failed to advance or to release keys: the run is invalid,
        for the environment's reasons (sandbox/actions.py GuestEnvironment.wait_until_tick,
        GuestEnvironment._release_clock_keys). A wait whose own step failed is a failed action."""
        if self.phase is _Phase.WAIT:
            self.decision.failed = error
        ended = self.kernel.ending is not None
        self.kernel.invalidate("environment", reason, error)
        if ended:
            self._settle()

    def _release(self, then: Callable) -> None:
        tick = self._tick()
        keys = tuple(key for key, end in self.leases.items() if end <= tick + 1e-9)
        if not keys:
            return then()
        for key in keys:
            del self.leases[key]
        self._released = then
        self._stepping = True
        self.kernel.emit(ReleaseKeys(keys, escapes=self.phase is not _Phase.WAIT))

    def _keys_released(self, event: KeysReleased) -> None:
        if not self._stepping:
            return
        self._stepping = False
        if event.error is not None:
            return self._step_failed("clock_key_release_failed", event.error)
        if self.kernel.ending is not None:
            return self._settle()
        then, self._released = self._released, None
        then()

    # ---- clock arithmetic -------------------------------------------------------------------

    def _horizon_reached(self) -> bool:
        """The tick is at the horizon, within 1e-9 as a move is at its target (_advance): a move
        cut at a horizon off the tick grid stops a float short of it."""
        return self._tick() >= self.config.max_ticks - 1e-9

    def _read_already(self, *, before_landing: bool) -> bool:
        """Whether a state read now would read what the last one read. On the token clock that is
        the same position, which every step of a move makes new. On the wall clock it is the same
        environment frame, except right before landing, where it must be the same reading: real
        time moves on within a frame, and an end since keeps the action from landing, as the
        legacy loop reads right before it executes (legacy_loop.py _land_action)."""
        last = self._read_at
        if last is None:
            return False
        if self.token or before_landing:
            return last == self.kernel.semantic
        return _floor(last * self.config.env_fps) == self._tick()

    def _tick(self) -> float:
        """The environment tick: the token clock itself, or whole wall frames since the origin."""
        if self.token:
            return self.kernel.semantic
        return _floor(self.kernel.semantic * self.config.env_fps)

    def _position(self) -> float:
        """The unquantized clock position in environment ticks."""
        if self.token:
            return self.kernel.semantic
        return self.kernel.semantic * self.config.env_fps

    def _semantic_of(self, tick: float) -> float:
        """The semantic time of a tick. On the wall clock the horizon's is the deadline, a few ulp
        past horizon / env_fps once the horizon snapped to a whole tick."""
        if self.token:
            return tick
        if tick >= self.config.max_ticks - 1e-9:
            return self.config.duration_s
        return tick / self.config.env_fps

    def _frames(self, ticks: float) -> float:
        """Environment ticks as input frames."""
        return ticks * self.config.input_fps / self.config.env_fps

    def _ticks(self, frames: float) -> float:
        """Input frames as environment ticks."""
        return frames * self.config.env_fps / self.config.input_fps

    def _next_observation_tick(self, t_k: float, action_end: float) -> float:
        """The first input frame after the observed one and not before the action ends, at the first
        environment tick at or after it, and never past the horizon (legacy_loop.py _apply)."""
        frame = max(_floor(self._frames(t_k)) + 1, _ceil(self._frames(action_end)))
        return min(_ceil(self._ticks(frame)), self.config.max_ticks)
