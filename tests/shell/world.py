"""A scripted environment, agent and episode trace for the kernel's shell (evaluation/shell.py), on
the fake clock (tests/fake_clock.py).

The environment and the agent implement the W0 protocols (ShellEnvironment, KernelAgent) and
nothing more than the shell uses. Each environment operation takes its scripted fake time on the
shell's thread and is logged in ``calls``; a generation takes its scripted time on the worker the
shell starts, and an abort cuts it short. The wall clock's semantic time is seconds since ``t0``;
on the token clock it is the FrameClock's frame, which only ``step_clock`` and a key tap move.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from realtime_framework.clock import Clocks, FrameClock
from realtime_framework.entity.actions import Action, ActionType, Observation
from realtime_framework.entity.config import RunnerTimingConfig
from realtime_framework.entity.model import PendingDecision
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.agent_trace import AgentTraceSink
from realtime_framework.evaluation.clock import TokenAccounting
from realtime_framework.evaluation.episode_end import EpisodeResult
from realtime_framework.evaluation.kernel.turns import TurnConfig
from realtime_framework.evaluation.shell import run_episode
from realtime_framework.sandbox.actions import TAP_HOLD_S

from ..fake_clock import FakeClock

ORIGIN = 1_048_576.0  # the fake clock's Unix time at monotonic 0; a power of two keeps sums exact
CLICK = Action.click(10, 20)


def _snap(tick: float) -> float:
    return round(tick) if abs(tick - round(tick)) < 1e-9 else tick


@dataclass
class World:
    """What the environment does. Decisions and calls count from 0, end checks from 1."""

    ends_at: float | None = None  # the task reports its end from this semantic time on
    ends_after: int | None = None  # ... or once this many actions have been applied
    end_reason: str | None = None
    capture_s: float = 0.0
    check_s: float = 0.0
    # (start, end, seconds): an end check asked in [start, end) of semantic time takes ``seconds``
    slow_checks: list[tuple[float, float, float]] = field(default_factory=list)
    execute_s: float = 0.0
    snapshot_s: float = 0.0
    frozen_from: int | None = None  # from this capture on, every frame repeats one identity
    capture_errors: dict[int, str] = field(default_factory=dict)  # decision -> fault
    check_errors: dict[int, str] = field(default_factory=dict)  # end check -> fault
    rejections: dict[int, str] = field(default_factory=dict)  # decision -> the guest's refusal
    failures: dict[int, str] = field(default_factory=dict)  # decision -> environment fault
    # decision -> the invalid run the environment latches itself while executing, then fails
    latches: dict[int, dict] = field(default_factory=dict)
    tap_checks: dict[int, tuple] = field(default_factory=dict)  # decision -> last_end_checks
    advance_errors: dict[int, str] = field(default_factory=dict)  # step_clock call -> fault
    release_errors: dict[int, str] = field(default_factory=dict)  # release_keys call -> fault
    render_errors: dict[int, str] = field(default_factory=dict)  # render call -> fault
    release_failures: set[int] = field(default_factory=set)  # decisions whose wall release fails
    lifecycle_errors: set[int] = field(default_factory=set)  # decisions
    renders: bool = True  # render_current_frame commits a frame
    snapshot_error: str | None = None
    snapshotter: bool = True
    on_release: Callable[[], None] | None = None  # what release_inputs does besides


class Environment:
    """GuestEnvironment as the shell sees it (evaluation/shell.py ShellEnvironment)."""

    def __init__(self, clocks: Clocks, clock_mode: str, fps: float, world: World):
        self.clocks, self.clock_mode, self.world = clocks, clock_mode, world
        self.token = clock_mode == "token"
        self.clock = clocks.operational
        self.d = 1.0 / fps
        self.t0 = self.clock.time()
        # A task without a snapshot reader has no evaluation_snapshot at all.
        self.task = (
            SimpleNamespace(evaluation_snapshot=self._snapshot)
            if world.snapshotter
            else SimpleNamespace()
        )
        self.observations: list = []  # weak references to every frame captured
        self.last_held_keys: tuple[str, ...] = ()
        self.last_end_checks: tuple[tuple[bool, str | None, str | None], ...] = ()
        self.on_invalid = None
        self.frame_ages: list[float] = []
        self.lifecycle: list[dict] = []
        self.calls: list[tuple] = []  # (call, semantic time, details)
        self.decision = -1
        self.checks = 0
        self.applied = 0
        self.advances = 0
        self.releases = 0
        self.renders = 0
        self.horizon = None
        self._frozen = None
        self._last_exec_error = None
        self._last_rejected_by_guest = False
        self._last_execution_result = None
        self._last_coordinate_trace = None

    def semantic(self) -> float:
        return self.clocks.semantic.frame if self.token else self.clock.time() - self.t0

    def log(self, call: str, **details) -> None:
        self.calls.append((call, self.semantic(), details))

    def names(self) -> list[str]:
        return [call for call, _, _ in self.calls]

    def tick_index(self) -> float:
        return self.semantic() if self.token else math.floor(self.semantic() / self.d + 1e-9)

    def set_episode_horizon(self, max_tick: float) -> None:
        self.horizon = max_tick

    def capture(self) -> Observation:
        self.decision += 1
        self.clock.sleep(self.world.capture_s)
        self.log("capture")
        if self.decision in self.world.capture_errors:
            raise RuntimeError(self.world.capture_errors[self.decision])
        tick = self.tick_index()
        timestamp = self.clocks.semantic.monotonic() if self.token else self.clock.time()
        observation = Observation(
            f"frame-{self.decision}".encode(),
            timestamp,
            tick,
            {"observation_id": f"obs-{self.decision}"},
        )
        frozen = self.world.frozen_from
        if frozen is not None and self.decision >= frozen:
            self._frozen = self._frozen or observation
            observation = Observation(
                self._frozen.screenshot, timestamp, self._frozen.tick_number, self._frozen.metadata
            )
        self.observations.append(weakref.ref(observation))
        return observation

    def check_end(self) -> tuple[bool, str | None]:
        self.checks += 1
        now = self.semantic()
        slow = [seconds for start, end, seconds in self.world.slow_checks if start <= now < end]
        self.clock.sleep(slow[0] if slow else self.world.check_s)
        self.log("check_end", check=self.checks)
        if self.checks in self.world.check_errors:
            raise RuntimeError(self.world.check_errors[self.checks])
        ends_at, ends_after = self.world.ends_at, self.world.ends_after
        ended = (ends_at is not None and self.semantic() >= ends_at) or (
            ends_after is not None and self.applied >= ends_after
        )
        return ended, self.world.end_reason if ended else None

    def execute(self, action: Action, *, release_at: float | None = None) -> str:
        world, decision = self.world, self.decision
        self.last_held_keys, self.last_end_checks = (), world.tap_checks.get(decision, ())
        self.log("execute", action=action.action_type.value, release_at=release_at)
        self.clock.sleep(world.execute_s)
        if decision in world.latches:
            self._invalid_run = world.latches[decision]
            return self._failed(world.latches[decision]["detail"], by_guest=False)
        if decision in world.rejections:
            return self._failed(world.rejections[decision], by_guest=True)
        if decision in world.failures:
            return self._failed(world.failures[decision], by_guest=False)
        if self.token and action.action_type is ActionType.KEY:
            self.clocks.semantic.frame = _snap(self.clocks.semantic.frame + TAP_HOLD_S / self.d)
        if self.token and action.action_type is ActionType.HOLD_KEY:
            self.last_held_keys = (str(action.parameters["key"]).lower(),)
        if release_at is not None:
            self.clock.timer(
                self.t0 + release_at - self.clock.time(), lambda: self._wall_release(decision)
            )
        self._last_execution_result = {"executed": decision}
        self._last_coordinate_trace = {"decision": decision}
        self.applied += 1
        return "applied"

    def _failed(self, message: str, *, by_guest: bool) -> str:
        self._last_exec_error = message
        self._last_rejected_by_guest = by_guest
        self._last_execution_result = {"error": message}
        self._last_coordinate_trace = None
        return "failed"

    def _wall_release(self, decision: int) -> None:
        """The wall clock's key-release timer, on its own thread."""
        self.log("wall_release", decision=decision)
        if decision in self.world.release_failures:
            self._invalid_run = {
                "category": "environment",
                "reason": "wall_key_release_failed",
                "detail": "keyup failed",
            }
            self.on_invalid(self._invalid_run)

    def step_clock(self, step: float) -> None:
        self.advances += 1
        self.log("step_clock", step=step)
        if self.advances in self.world.advance_errors:
            self._invalid_run = {
                "category": "environment",
                "reason": "clock_advance_failed",
                "detail": self.world.advance_errors[self.advances],
            }
            raise RuntimeError(self.world.advance_errors[self.advances])
        self.clocks.semantic.frame = _snap(self.clocks.semantic.frame + step)

    def release_keys(self, keys: tuple[str, ...]) -> None:
        self.releases += 1
        self.log("release_keys", keys=keys)
        if self.releases in self.world.release_errors:
            raise RuntimeError(self.world.release_errors[self.releases])

    def render_current_frame(self):
        self.renders += 1
        self.log("render")
        if self.renders in self.world.render_errors:
            raise RuntimeError(self.world.render_errors[self.renders])
        return {"rendered": True} if self.world.renders else None

    def release_inputs(self) -> None:
        self.log("release_inputs")
        if self.world.on_release is not None:
            self.world.on_release()

    def record_lifecycle(self, record: dict) -> None:
        if record["decision"] in self.world.lifecycle_errors:
            raise OSError("disk full")
        self.lifecycle.append(record)

    def _snapshot(self) -> dict:
        self.clock.sleep(self.world.snapshot_s)
        self.log("snapshot")
        if self.world.snapshot_error is not None:
            raise RuntimeError(self.world.snapshot_error)
        return {"tick": self.tick_index()}


@dataclass(frozen=True)
class Reply:
    """One generation: it takes ``seconds`` of fake time, or until it is aborted when it hangs."""

    seconds: float = 0.0
    action: Action | None = field(default_factory=lambda: CLICK)
    output_tokens: int | None = None
    hang: bool = False
    raises: str | None = None
    fields: dict = field(default_factory=dict)  # more of the result


class Agent:
    """A KernelAgent (evaluation/shell.py) that answers from a script."""

    def __init__(self, clock, replies: list[Reply]):
        self.clock = clock
        self.replies = list(replies)
        self.generated: list[tuple] = []  # (generation, decision, observation id, budget)
        self.calls: list[tuple] = []  # (call, argument)
        self.outcomes: list = []
        self.trajectory: list[dict] = []  # the settled results, as the trajectory audit gets them
        self._cancels: dict[int, object] = {}

    def finalize_decision_audit(self, result: dict) -> None:
        self.trajectory.append(dict(result))

    def generate(self, generation, decision, observation, budget) -> PendingDecision:
        reply = self.replies[len(self.generated)]
        self.generated.append(
            (generation, decision, observation.metadata["observation_id"], budget)
        )
        cancel = self._cancels[generation] = self.clock.event()
        started = self.clock.monotonic()
        self.clock.wait(cancel, None if reply.hang else reply.seconds)
        if reply.raises is not None:
            raise RuntimeError(reply.raises)
        aborted = cancel.is_set()
        result = {
            "action": None if aborted else reply.action,
            "latency": self.clock.monotonic() - started,
            "raw": "" if aborted else "a reply",
            "provider_output_tokens": None if aborted else reply.output_tokens,
            "stale": aborted,
            **reply.fields,
        }
        return PendingDecision(generation, decision, result)

    def accept(self, generation: int) -> None:
        self.calls.append(("accept", generation))

    def discard(self, generation: int) -> None:
        self.calls.append(("discard", generation))

    def commit(self, outcome) -> None:
        self.calls.append(("commit", outcome.decision))
        self.outcomes.append(outcome)

    def abort(self, generation: int | None = None) -> None:
        self.calls.append(("abort", generation))
        self._cancels[generation].set()


class Trace:
    """The episode trace as the shell writes it: ``emit`` and ``checkpoint``."""

    def __init__(self, fail_on: str | None = None, fail_checkpoint: bool = False):
        self.records: list[dict] = []
        self.checkpoints = 0
        self.fail_on = fail_on
        self.fail_checkpoint = fail_checkpoint

    def emit(self, event, payload=None, **fields) -> None:
        name = event.value if isinstance(event, TraceEvent) else event
        if name == self.fail_on:
            raise OSError("trace disk full")
        self.records.append({"event": name, "payload": payload, **fields})

    def checkpoint(self) -> None:
        if self.fail_checkpoint:
            raise OSError("trace fsync failed")
        self.checkpoints += 1

    def events(self) -> list[str]:
        return [record["event"] for record in self.records]

    def payloads(self, event: str) -> list[dict]:
        return [record["payload"] for record in self.records if record["event"] == event]


@dataclass
class Episode:
    config: TurnConfig
    env: Environment
    agent: Agent
    trace: Trace | None
    clock: FakeClock
    result: EpisodeResult
    logs: list[str]

    @property
    def timeline(self) -> list:
        return self.result.timeline

    def statuses(self) -> list[str]:
        return [outcome.status for outcome in self.agent.outcomes]

    def token_accounting(self) -> dict:
        """The token totals main() takes from the episode's settlements (runner.py main)."""
        return TokenAccounting(self.result.settlements).summary()

    @property
    def invalid(self) -> dict | None:
        return self.result.invalid_run


def run(
    clock_mode: str,
    duration_s: float,
    replies: list[Reply],
    world: World | None = None,
    *,
    fps: float = 1.0,
    input_fps: float | None = None,
    tokens_per_frame: int | None = None,
    timing: RunnerTimingConfig | None = None,
    trace=Trace,
    agent_type: type[Agent] = Agent,
    inputs_log: Path | None = None,
    env_type: type[Environment] = Environment,
    log=None,
    agent_trace_pack: Path | None = None,
    **config,
) -> Episode:
    """One episode through run_episode on a fresh fake clock; the abandoned workers are drained
    afterwards, as main() does. ``trace`` is a new Trace unless one is given, or None. With
    ``agent_trace_pack`` the agent gets an AgentTraceSink on the trace, as ``agent.sink``, which
    run_episode is given too."""
    clock = FakeClock(wall_origin=ORIGIN)
    clocks = Clocks(FrameClock(fps), clock) if clock_mode == "token" else Clocks(clock, clock)
    turn = TurnConfig(
        clock_mode,
        duration_s,
        env_fps=fps,
        input_fps=fps if input_fps is None else input_fps,
        tokens_per_frame=tokens_per_frame,
        timing=timing or RunnerTimingConfig(),
        **config,
    )
    env = env_type(clocks, clock_mode, fps, world or World())
    agent = agent_type(clock, replies)
    trace = trace() if trace is Trace else trace
    agent.sink = (
        None if agent_trace_pack is None else AgentTraceSink(trace, agent_trace_pack, clock=clock)
    )
    logs: list[str] = []
    with clock.participant():
        result = run_episode(
            agent,
            env,
            turn,
            clocks=clocks,
            log=log or logs.append,
            trace=trace,
            agent_trace=agent.sink,
            inputs_log=inputs_log,
        )
        result.drain_abandoned()
    return Episode(turn, env, agent, trace, clock, result, logs)
