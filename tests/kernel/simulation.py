"""A discrete-event simulation shell for the event kernel (phase2.md, 事件内核).

No threads. A scripted world says how long each environment operation and each model call takes,
what each call answers, and from when the task reports its end. Time is simulated: ``ops`` is the
operational clock in seconds, which on the wall clock is also the world's clock; on the token clock
the world is the environment tick, which only AdvanceClock and key taps move. A synchronous command
occupies the shell for its duration, so whatever comes due meanwhile is handled after it returns,
as in the production shells. What other threads produce (a generation's result, an agent record,
a failure of the wall key-release timer) is posted when the simulated time reaches it.

The world plays GuestEnvironment's part only as far as the kernel sees it: a tap holds the key
1/60 s per repeat, advancing the token tick in steps of at most one tick with an end check after
each and never past the horizon, and a tick within 1e-9 of an integer snaps to it
(sandbox/actions.py GuestEnvironment._execute_clock_key, GuestEnvironment.wait_until_tick). Unlike
the environment, a tap's steps are not cut where a lease ends, and its end checks see ``ends_at``
only: ``ends_from_check`` stands for a web task's own timers, and only games tap on the task clock.
The render barrier commits a frame unless ``renderer`` is False, as for a task without a renderer.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.evaluation.kernel.core import Kernel
from realtime_framework.evaluation.kernel.events import (
    AbortGeneration,
    AcceptResult,
    ActionExecuted,
    AdvanceClock,
    AgentRecord,
    Capture,
    CheckEnd,
    Clock,
    ClockStepped,
    Commit,
    DiscardResult,
    EndChecked,
    Entry,
    Execute,
    Finish,
    FrameCaptured,
    FrameRendered,
    FreezeSnapshot,
    GenerationFinished,
    GenerationResult,
    Invalidated,
    KeysReleased,
    OutcomeCommitted,
    Record,
    ReleaseInputs,
    ReleaseKeys,
    RenderFrame,
    SetTimer,
    SnapshotFrozen,
    StartGeneration,
)
from realtime_framework.evaluation.kernel.turns import TurnConfig, TurnPolicy
from realtime_framework.sandbox.actions import WALL_RELEASE_MAX_ATTEMPTS, WALL_RELEASE_RETRY_S

TAP_HOLD_S = 1.0 / 60.0  # sandbox/actions.py TAP_HOLD_S: how long a token key tap holds its key


@dataclass(frozen=True)
class Reply:
    """One model call."""

    latency: float = 0.0  # operational seconds until the worker hands over its result
    action: Action | None = None
    output_tokens: int | None = None
    agent_latency: float | None = None  # GenerationResult.latency; the call's latency by default
    protocol_failure: str | None = None
    invalid: Invalidated | None = None
    # False for a Decision call: its questions run on after an abort, and the result comes back
    # at the scripted time anyway.
    interruptible: bool = True
    records: tuple[tuple[float, str], ...] = ()  # (offset, kind) the agent records meanwhile


@dataclass
class World:
    """The scripted environment and model of one episode."""

    replies: list[Reply] = field(default_factory=list)
    ends_at: float | None = None  # the task reports its end from this semantic time on
    # ... or from this end check (1-based) on, even at a tick an earlier check saw: a web task's
    # own timers run on real time on the token clock too (phase2.md, 时钟).
    ends_from_check: int | None = None
    end_reason: str | None = None
    start_s: float = 0.0  # wall clock: the loop starts this long after the episode origin
    # (ops time, event) failures other threads post, such as the wall key-release timer's
    invalidations: tuple[tuple[float, Invalidated], ...] = ()
    capture_s: float = 0.0  # operational seconds each command occupies the shell
    check_s: float = 0.0
    check_seconds: dict[int, float] = field(default_factory=dict)  # end check (1-based) -> its own
    execute_s: float = 0.0
    accept_s: float = 0.0
    advance_s: float = 0.0
    release_s: float = 0.0  # a ReleaseKeys
    commit_s: float = 0.0
    snapshot_s: float = 0.0
    renderer: bool = True  # whether the task commits a frame at the token render barrier
    frozen_from: int | None = None  # from this capture on every frame repeats the same identity
    capture_errors: dict[int, str] = field(default_factory=dict)  # capture (1-based) -> fault
    rejections: dict[int, str] = field(default_factory=dict)  # decision -> the guest's refusal
    execute_errors: dict[int, str] = field(default_factory=dict)  # decision -> environment fault
    check_errors: dict[int, str] = field(default_factory=dict)  # end check (1-based) -> fault
    # End check inside a key tap (1-based over all taps, apart from the others) -> fault.
    tap_check_errors: dict[int, str] = field(default_factory=dict)
    advance_errors: dict[int, str] = field(default_factory=dict)  # AdvanceClock (1-based) -> fault
    # Key release (1-based, by ReleaseKeys or the wall release timer) -> fault.
    release_errors: dict[int, str] = field(default_factory=dict)
    commit_errors: dict[int, str] = field(default_factory=dict)  # decision -> fault
    snapshot_error: str | None = None


def key_identities(action: Action) -> tuple[str, ...]:
    """The keys of a chord, lower-cased and de-duplicated in order."""
    items = action.parameters.get("keys") or [action.parameters.get("key", "")]
    keys: list[str] = []
    for item in items:
        for part in str(item).split("+"):
            key = part.strip().lower()
            if key and key not in keys:
                keys.append(key)
    return tuple(keys)


def _snap(tick: float) -> float:
    return round(tick) if abs(tick - round(tick)) < 1e-9 else tick


class Simulation:
    """Runs one kernel episode against a World and keeps everything it saw."""

    def __init__(self, config: TurnConfig, world: World) -> None:
        self.config = config
        self.world = world
        self.token = config.clock_mode == "token"
        self.kernel = Kernel(TurnPolicy(config))
        self.ops = 0.0 if self.token else world.start_s
        self.tick: float = 0
        self.records: list[Entry] = []
        self.commands: list = []
        self.command_times: list[float] = []  # the semantic time each command was run at
        self.checks: list[tuple[float, float]] = []  # (semantic, ops) at every end check
        self.after_action: list[bool] = []  # for every end check, whether it follows an action
        self.tap_checks: list[float] = []  # the tick at every end check inside a key tap
        self.captures: list[tuple[float, float]] = []  # (tick, timestamp) of every frame
        self.executed: list[tuple[int, float, Action]] = []  # (decision, semantic, action)
        self.advances: list[float] = []  # every AdvanceClock step
        self.releases = 0  # key releases tried
        self.key_events: list[tuple[float, str, tuple[str, ...]]] = []  # (semantic, up, keys)
        self.renders: list[float] = []
        self.outcomes: list = []
        self.aborted: list[tuple[int, float]] = []  # (generation, ops)
        self.accepted: list[int] = []
        self.discarded: list[int] = []
        self.snapshots: list[tuple[float, float]] = []
        self.released_inputs = False
        self.termination = None
        self._queue: deque = deque()
        self._pending: list = []  # (ops time, order, callable) of other threads' work
        self._order = itertools.count()
        self._wakes: list[tuple[Clock, float]] = []
        self._replies = iter(world.replies)
        self._results: dict[int, list] = {}  # generation -> its pending result entry
        self._releases: dict[str, float] = {}  # wall clock: key -> latest release time
        self._wall_release_reported = False  # the environment latched its invalid run
        self._frozen_tick: float | None = None

    # ---- driving ----------------------------------------------------------------------------

    def run(self, max_steps: int = 100_000) -> Simulation:
        for at, event in self.world.invalidations:
            self._later(at, lambda event=event: self.kernel.post(event))
        self._queue.extend(self.kernel.start(*self._now()))
        for _ in range(max_steps):
            if self._queue:
                self._run(self._queue.popleft())
                continue
            if self.termination is not None:
                return self
            out = self.kernel.step(*self._now())
            if out is not None:
                self._queue.extend(out)
            else:
                self._wait()
        raise RuntimeError("the simulated episode did not finish")

    def semantic(self) -> float:
        return self.tick if self.token else self.ops

    def kinds(self) -> list[str]:
        return [entry.kind for entry in self.records]

    def entries(self, kind: str) -> list[Entry]:
        return [entry for entry in self.records if entry.kind == kind]

    def _now(self) -> tuple[float, float]:
        return self.semantic(), self.ops

    def _wait(self) -> None:
        """Nothing is due: move the time to the next thing that will happen."""
        # Token time only moves by AdvanceClock, never by waiting; spent wakes are dropped.
        self._wakes = [
            (clock, at)
            for clock, at in self._wakes
            if (clock is Clock.OPS or not self.token) and at > self.ops
        ]
        times = [at for _, at in self._wakes]
        if self._pending:
            times.append(self._pending[0][0])
        if not times:
            raise RuntimeError("the simulated episode stalled")
        self._spend_until(min(times))

    def _spend(self, seconds: float) -> None:
        self._spend_until(self.ops + seconds)

    def _spend_until(self, until: float) -> None:
        while self._pending and self._pending[0][0] <= until:
            entry = heapq.heappop(self._pending)
            self.ops = max(self.ops, entry[0])
            entry[2]()
        self.ops = max(self.ops, until)

    def _later(self, at: float, work: Callable) -> list:
        entry = [at, next(self._order), work]
        heapq.heappush(self._pending, entry)
        return entry

    # ---- commands ---------------------------------------------------------------------------

    def _run(self, command) -> None:
        self.commands.append(command)
        self.command_times.append(self.semantic())
        world, kernel = self.world, self.kernel
        match command:
            case Record(entry=entry):
                self.records.append(entry)
            case SetTimer(clock=clock, at=at):
                self._wakes.append((clock, at))
            case CheckEnd(request=request):
                semantic = self.semantic()
                self.checks.append((semantic, self.ops))
                self.after_action.append(kernel.policy.phase.value == "post_check")
                error = world.check_errors.get(len(self.checks))
                ended = error is None and (
                    (world.ends_at is not None and semantic >= world.ends_at)
                    or (
                        world.ends_from_check is not None
                        and len(self.checks) >= world.ends_from_check
                    )
                )
                self._spend(world.check_seconds.get(len(self.checks), world.check_s))
                kernel.post(EndChecked(request, ended, world.end_reason if ended else None, error))
            case Capture():
                self._capture()
            case StartGeneration(generation=generation):
                self._start(generation)
            case AbortGeneration(generation=generation):
                self.aborted.append((generation, self.ops))
                entry = self._results.get(generation)
                if entry is not None and entry[3].interruptible and entry[0] > self.ops:
                    # The interrupted call hands back whatever it had, now.
                    entry[2] = lambda: None
                    self._later(
                        self.ops,
                        lambda: kernel.post(GenerationFinished(generation, GenerationResult())),
                    )
            case AcceptResult(generation=generation):
                self.accepted.append(generation)
                self._spend(world.accept_s)
            case DiscardResult(generation=generation):
                self.discarded.append(generation)
            case Execute(decision=decision, action=action, release_at=release_at):
                self._execute(decision, action, release_at)
            case AdvanceClock(step=step):
                self.advances.append(step)
                error = world.advance_errors.get(len(self.advances))
                self._spend(world.advance_s)
                if error is None:
                    self.tick = _snap(self.tick + step)
                kernel.post(ClockStepped(error))
            case ReleaseKeys(keys=keys):
                error = self._release(keys)
                self._spend(world.release_s)
                kernel.post(KeysReleased(keys, error))
            case RenderFrame():
                self.renders.append(self.semantic())
                kernel.post(FrameRendered(world.renderer))
            case Commit(outcome=outcome):
                self._spend(world.commit_s)
                self.outcomes.append(outcome)
                kernel.post(
                    OutcomeCommitted(outcome.decision, world.commit_errors.get(outcome.decision))
                )
            case FreezeSnapshot():
                self._spend(world.snapshot_s)
                self.snapshots.append(self._now())
                kernel.post(SnapshotFrozen(*self._now(), world.snapshot_error))
            case ReleaseInputs():
                self.released_inputs = True
            case Finish(termination=termination):
                self.termination = termination

    def _capture(self) -> None:
        world, index = self.world, len(self.captures)
        tick = self.tick if self.token else math.floor(self.ops * self.config.env_fps + 1e-9)
        self._spend(world.capture_s)
        timestamp = tick / self.config.env_fps if self.token else self.ops
        self.captures.append((tick, timestamp))
        frame = index
        if world.frozen_from is not None and index >= world.frozen_from:
            frame = world.frozen_from
            self._frozen_tick = tick if self._frozen_tick is None else self._frozen_tick
            tick = self._frozen_tick
        fingerprint = (f"frame-{frame}", tick, f"sha-{frame}")
        error = world.capture_errors.get(index + 1)
        self.kernel.post(FrameCaptured(f"obs-{index}", tick, timestamp, fingerprint, error))

    def _start(self, generation: int) -> None:
        reply = next(self._replies)
        result = GenerationResult(
            action=reply.action,
            output_tokens=reply.output_tokens,
            latency=reply.latency if reply.agent_latency is None else reply.agent_latency,
            protocol_failure=reply.protocol_failure,
            invalid=reply.invalid,
        )
        kernel = self.kernel
        entry = self._later(
            self.ops + reply.latency,
            lambda: kernel.post(GenerationFinished(generation, result)),
        )
        entry.append(reply)
        self._results[generation] = entry
        for offset, kind in reply.records:
            self._later(
                self.ops + offset,
                lambda kind=kind: kernel.post(AgentRecord(kind, {}, generation)),
            )

    def _execute(self, decision: int, action: Action, release_at: float | None) -> None:
        world = self.world
        self.executed.append((decision, self.semantic(), action))
        error, rejection = world.execute_errors.get(decision), world.rejections.get(decision)
        held: tuple[str, ...] = ()
        end_checks: list[tuple[bool, str | None, str | None]] = []
        if error is None and rejection is None:
            if self.token and action.action_type is ActionType.KEY:
                for _ in range(int(action.parameters.get("repeat", 1))):
                    end_checks += self._tap()
            duration = float(action.parameters.get("duration", 1.0))
            if self.token and action.action_type is ActionType.HOLD_KEY and duration > 1e-9:
                held = key_identities(action)
            if release_at is not None:
                # The release timer is armed right after the press; a key already held until
                # later keeps its later release.
                for key in key_identities(action):
                    at = max(self._releases.get(key, release_at), release_at)
                    self._releases[key] = at
                    self._later(at, lambda key=key, at=at: self._wall_release(key, at))
        self._spend(world.execute_s)
        self.kernel.post(
            ActionExecuted(
                decision,
                result={"action": action.action_type.value},
                held_keys=held,
                rejection=rejection,
                error=error,
                end_checks=tuple(end_checks),
            )
        )

    def _tap(self) -> list[tuple[bool, str | None, str | None]]:
        """One repeat of a token key tap: the tick moves TAP_HOLD_S / d, at most to the horizon,
        one tick at most per step, and the end is checked after each step, within the action's
        time; a check that sees the end stops the tap (sandbox/actions.py
        GuestEnvironment._execute_clock_key, GuestEnvironment.wait_until_tick). A failed check
        does not, as in the environment once the kernel counts its reads (shell.py
        kernel_counts_reads; under the legacy loop done() stops the tap). Each read goes to the
        kernel in ActionExecuted.end_checks as (ended, reason, error)."""
        world, reads = self.world, []
        target = min(self.tick + TAP_HOLD_S / (1.0 / self.config.env_fps), self.config.max_ticks)
        while self.tick + 1e-9 < target:
            self.tick = _snap(self.tick + min(1.0, target - self.tick))
            self.tap_checks.append(self.tick)
            error = world.tap_check_errors.get(len(self.tap_checks))
            ended = error is None and world.ends_at is not None and self.tick >= world.ends_at
            reads.append((ended, world.end_reason if ended else None, error))
            if ended:
                break
        return reads

    def _release(self, keys: tuple[str, ...]) -> str | None:
        """Release keys; the error of a scripted failure, which releases nothing."""
        self.releases += 1
        error = self.world.release_errors.get(self.releases)
        if error is None:
            self.key_events.append((self.semantic(), "up", keys))
        return error

    def _wall_release(self, key: str, at: float) -> None:
        """The environment's own release timer on the wall clock. It tells the kernel nothing of
        a release that worked. A scripted failure fails every attempt: the timer tries again
        every WALL_RELEASE_RETRY_S, gives up after its last attempt and reports the invalid run
        through on_invalid, once an episode (sandbox/actions.py
        GuestEnvironment._wall_release_due; shell.py)."""
        if self._releases.get(key) != at:
            return
        del self._releases[key]
        error = self._release((key,))
        if error is None:
            return
        given_up = at + (WALL_RELEASE_MAX_ATTEMPTS - 1) * WALL_RELEASE_RETRY_S
        self._later(given_up, lambda: self._wall_release_given_up(error))

    def _wall_release_given_up(self, error: str) -> None:
        if not self._wall_release_reported:
            self._wall_release_reported = True
            self.kernel.post(Invalidated("environment", "wall_key_release_failed", error))
