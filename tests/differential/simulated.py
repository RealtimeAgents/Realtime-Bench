"""The simulation side: a script played on the kernel's discrete-event simulation
(tests/kernel/simulation.py), projected for comparison.

The simulation stands in for the shell and the world, without threads: each reply takes its
script's latency in operational seconds, a hang never comes back, and on the wall clock every
environment operation takes its script's time. Turn k is the reply to decision k.
"""

from __future__ import annotations

import itertools

from realtime_framework.entity.actions import ActionType
from realtime_framework.entity.config import RunnerTimingConfig
from realtime_framework.evaluation.kernel.events import (
    CheckEnd,
    Commit,
    FreezeSnapshot,
    Invalidated,
)
from realtime_framework.evaluation.kernel.turns import TurnConfig
from realtime_framework.sandbox.actions import TAP_HOLD_S, GuestEnvironment
from tests.kernel.simulation import Reply, Simulation, World

from .projection import ONCE, Once, Projection, decision, key_releases, wait_record
from .script import Script, action_key, sim_action

NEVER_S = 1e9  # a call that is never answered
STATE_READ_FAILED = "state read failed"


def turn_config(script: Script) -> TurnConfig:
    return TurnConfig(
        script.mode,
        script.duration_s,
        env_fps=script.env_fps,
        input_fps=script.input_fps,
        tokens_per_frame=script.tokens_per_frame,
        think_budget_s=script.think_budget_s,
        per_move_cap_s=script.per_move_cap_s,
        timing=RunnerTimingConfig(),
    )


def world(script: Script) -> World:
    replies = []
    for turn in script.turns:
        action, protocol_failure = sim_action(turn.action)
        replies.append(
            Reply(
                latency=NEVER_S if turn.hangs else turn.latency_s,
                action=action,
                output_tokens=turn.output_tokens,
                protocol_failure=protocol_failure,
            )
        )
    ticks = 1.0 if script.clock == "wall" else script.env_fps
    return World(
        replies=replies,
        ends_at=None if script.ends_at_s is None else script.ends_at_s * ticks,
        end_reason=script.end_reason,
        capture_s=script.capture_s,
        check_s=script.check_s,
        execute_s=script.execute_s,
        commit_s=script.commit_s,
        snapshot_s=script.snapshot_s,
        renderer=False,  # the replayed task has none (sandbox/actions.py render_current_frame)
        frozen_from=script.frozen_from,
        rejections={k: t.rejected for k, t in enumerate(script.turns) if t.rejected},
        execute_errors={k: t.fault for k, t in enumerate(script.turns) if t.fault},
        check_errors={n: STATE_READ_FAILED for n in script.check_errors},
        tap_check_errors={n: STATE_READ_FAILED for n in script.tap_check_errors},
        advance_errors={n: "page clock did not advance" for n in script.advance_errors},
        snapshot_error=script.snapshot_error,
    )


class ScriptedSimulation(Simulation):
    """The simulation with the replayed world's traits it lacks: the task ends once the script's
    ``ends_after`` actions have taken effect; on the wall clock every end check asked within one
    of the script's ``failing_reads_s`` fails; and a failed action, which the environment latches
    itself while execute() returns, reaches the kernel as its answer with the error and then as
    the Invalidated the environment latched, as the shell posts them (decision N6;
    evaluation/shell.py _Shell._execute). A clock step that raises is answered with its error
    only. A wall hold pressed late is held a while more (decision 38)."""

    def __init__(self, config, world, script: Script) -> None:
        super().__init__(config, world)
        self.script = script
        self.applied = 0
        self.reads: list[dict] = []  # every end check, with its answer

    def _execute(self, decision: int, action, release_at) -> None:
        world = self.world
        error = world.execute_errors.get(decision)
        if not self.token and release_at is not None and action.action_type is ActionType.HOLD_KEY:
            # The environment presses after the action's time and holds at least as long as a
            # tap, or the hold if shorter (decision 38, N9; sandbox/actions.py
            # GuestEnvironment._execute_wall_hold).
            duration = float(action.parameters.get("duration", 1.0))
            pressed = self.semantic() + world.execute_s
            release_at = max(release_at, pressed + min(duration, TAP_HOLD_S))
        super()._execute(decision, action, release_at)
        if error is not None:
            self.kernel.post(Invalidated("environment", "action_execution_failed", error))
        if decision in world.rejections or error is not None:
            return
        self.applied += 1
        if self.applied == self.script.ends_after:
            now = self.semantic()
            world.ends_at = now if world.ends_at is None else min(world.ends_at, now)

    def _run(self, command) -> None:
        if not isinstance(command, CheckEnd):
            return super()._run(command)
        world, asked = self.world, self.semantic()
        if self.script.read_fails_at(asked):
            world.check_errors[len(self.checks) + 1] = STATE_READ_FAILED
        super()._run(command)
        # Its answer, as Simulation._run gives it.
        failed = len(self.checks) in world.check_errors
        ended = not failed and (
            (world.ends_at is not None and asked >= world.ends_at)
            or (world.ends_from_check is not None and len(self.checks) >= world.ends_from_check)
        )
        self.reads.append(
            {"at": asked, "answer": "failed" if failed else "task" if ended else None}
        )


def simulate(script: Script) -> Projection:
    return project(script, _run(script))


def _run(script: Script) -> ScriptedSimulation:
    sim = ScriptedSimulation(turn_config(script), world(script), script)
    sim.run()
    return sim


SAME_S = 1e-9  # closer than this, two times are one instant of the replay's fake clock


def ties(script: Script) -> list[str]:
    """Wall clock: where the shell's own timer and a reply, or the key-release timer and the
    shell, come due at one instant of the fake clock, which goes first is the scheduler's call,
    and where two timers fall a few ulps apart the clock's reading decides whether one step takes
    both (wave2-plan.md, R13). The simulation orders each the same way every time; a random script
    keeps clear of them. What the tie is, for each found."""
    if script.clock != "wall":
        return []
    sim = _run(script)
    ended = sim.termination.ended_at
    replies = [entry[0] for entry in sim._results.values()]
    # Every timer set, fired or not: one the simulation cancels can fire first in the shell.
    timers = [entry.data for entry in sim.entries("TimerSet")]
    found = []
    checks = [timer["at"] for timer in timers if timer["kind"] == "end_check"]
    by_check = sim.termination.cause != "deadline" and _near(ended, checks)
    if by_check and _near(ended, replies):
        found.append("a reply comes back as a periodic check ends the episode")
    # The check's answer is filed at the reading it was asked at (kernel/core.py Kernel.step):
    # where the fake clock cannot hold that instant exactly, the reading lies a few ulps past a
    # timer due then, which the simulation's exact reading ties with.
    others = [timer["at"] for timer in timers if timer["kind"] != "end_check"]
    if by_check and _near(ended, others) and ended * 1024 != round(ended * 1024):
        found.append("a timer comes due as a periodic check ends the episode")
    for timer in timers:
        if timer["kind"] in ("deadline", "per_move_cap") and any(
            0 < abs(at - timer["at"]) <= SAME_S for at in replies
        ):
            found.append(f"a reply comes back a few ulps from the {timer['kind']}")
    times = sorted({timer["at"] for timer in timers})
    if any(0 < later - at <= SAME_S for at, later in itertools.pairwise(times)):
        found.append("two timers come due a few ulps apart")
    executed = [semantic for _, semantic, _ in sim.executed]
    for at, _, _ in sim.key_events:
        if abs(at - ended) <= SAME_S:
            found.append("a key goes up as the episode ends")
        if _near(at, executed):
            found.append("a key goes up as an action runs")
    return found


def _near(at: float, times) -> bool:
    return any(abs(at - other) <= SAME_S for other in times)


def project(script: Script, sim: ScriptedSimulation) -> Projection:
    token = script.mode == "token"
    seconds = 1.0 / script.env_fps if token else 1.0  # semantic time in world seconds
    decisions: dict[int, dict] = {}

    def of(k: int) -> dict:
        return decisions.setdefault(k, decision())

    once = Once()
    for entry in sim.records:
        data = entry.data
        if entry.kind in ONCE:
            once(entry.kind, data["decision"], data.get("phase"))
        if entry.kind == "InferenceStarted":
            d = of(data["decision"])
            d["observed"] = data["observation_tick"]
            d["remaining"] = data["remaining_clock_tokens"]
        elif entry.kind == "ActionProposed":
            action = data["action"]
            of(data["decision"])["proposed"] = action_key(
                action.action_type.value, action.parameters
            )
        elif entry.kind == "ClockAdvanced":
            d = of(data["decision"])
            moved = {
                "from": data["from_tick"],
                "requested": data["requested_tick"],
                "to": data["to_tick"],
            }
            if data["phase"] == "inference_to_action":
                d["approach"] = moved
            else:
                d["boundary"] = moved | {
                    "rendered": data["post_action_rendered"],
                    "dropped": data["dropped_intermediate_frames"],
                }
        elif entry.kind == "ActionResolved":
            d = of(data["decision"])
            d["resolved"] = {
                "status": data["status"],
                "terminal_after_action": data["terminal_after_action"],
                "action_clock_position": data.get("action_clock_position"),
                "environment_tick": data.get("environment_tick"),
            }
            action = data["action"]
            if (
                action is not None
                and action.action_type.value == "wait"
                and data.get("execution_result")
            ):
                d["wait_record"] = wait_record(data["execution_result"])
        elif entry.kind == "ActionApplied":
            of(data["decision"])["applied"] = True
    for k, semantic, action in sim.executed:
        of(k)["executed"].append(
            {
                "at": semantic * seconds,
                "action": action_key(action.action_type.value, action.parameters),
            }
        )
    commits = 0
    snapshot_after = None
    for command in sim.commands:
        if isinstance(command, Commit):
            outcome = command.outcome
            once("commit", outcome.decision)
            once("lifecycle", outcome.decision)
            d = of(outcome.decision)
            d["commit"] = {"status": outcome.status, "detail": outcome.detail}
            d["outcome"] = {
                "status": outcome.status,
                "terminal_after_action": outcome.terminal_after_action,
                "action_clock_position": outcome.action_clock_position,
                "next_observation_tick": outcome.next_observation_tick,
                "dropped_intermediate_frames": outcome.dropped_intermediate_frames,
                "post_action_rendered": outcome.post_action_rendered,
                # The lifecycle record keeps six decimals (evaluation/episode_records.py
                # lifecycle_record).
                "frame_age_s": None
                if outcome.frame_age_s is None
                else round(outcome.frame_age_s, 6),
                "token_delay_frames": outcome.token_delay_frames,
                # The shell writes none on the wall clock (evaluation/shell.py _Shell._commit).
                "action_output_tokens": outcome.action_output_tokens if token else None,
                "post_horizon_output_tokens": outcome.post_horizon_output_tokens,
            }
            commits += 1
        elif isinstance(command, FreezeSnapshot) and snapshot_after is None:
            snapshot_after = commits
    invalid = next((entry.data for entry in sim.records if entry.kind == "RunInvalidated"), None)
    termination = sim.termination
    episode = {
        "captures": [{"tick": tick, "timestamp": timestamp} for tick, timestamp in sim.captures],
        "checks": [semantic for semantic, _ in sim.checks] if token else None,
        "key_ups": key_releases(
            [
                # Under the environment's names for the keys (sandbox/actions.py).
                {"at": at * seconds, "keys": [GuestEnvironment._key_identity(key) for key in keys]}
                for at, _, keys in sim.key_events
            ],
            termination.ended_at * seconds,
        ),
        "aborts": len(sim.aborted),
        "invalid": None
        if invalid is None
        else {"category": invalid["category"], "reason": invalid["reason"]},
        "termination": {
            "stop_reason": termination.stop_reason,
            "end_s": round(termination.ended_at * seconds, 9),
            "snapshot_s": None
            if termination.snapshot_at is None
            else round(termination.snapshot_at * seconds, 9),
        },
        "snapshot_after": snapshot_after,
        "invalidation_records": sim.kinds().count("RunInvalidated"),
        "duplicates": once.again,
    }
    kernel: dict = {"unaborted": []}
    if not token:
        kernel["reads"] = [read for read in sim.reads if read["at"] <= termination.ended_at]
    return Projection("simulation", {"decisions": decisions, "episode": episode}, kernel=kernel)
