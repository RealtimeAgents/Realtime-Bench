"""The simulation side: a script played on the kernel's discrete-event simulation
(tests/kernel/simulation.py), projected for comparison.

The simulation stands in for the shell and the world, without threads: each reply takes its
script's latency in operational seconds, a hang never comes back, and on the wall clock every
environment operation takes its script's time. Turn k is the reply to decision k.
"""

from __future__ import annotations

from realtime_framework.entity.config import RunnerTimingConfig
from realtime_framework.evaluation.kernel.events import (
    CheckEnd,
    Commit,
    FrameRendered,
    FreezeSnapshot,
    RenderFrame,
)
from realtime_framework.evaluation.kernel.turns import TurnConfig
from tests.kernel.simulation import Reply, Simulation, World

from .projection import Projection, decision, key_releases
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
        frozen_from=script.frozen_from,
        rejections={k: t.rejected for k, t in enumerate(script.turns) if t.rejected},
        execute_errors={k: t.fault for k, t in enumerate(script.turns) if t.fault},
        check_errors={n: STATE_READ_FAILED for n in script.check_errors},
        tap_check_errors={n: STATE_READ_FAILED for n in script.tap_check_errors},
        advance_errors={n: "page clock did not advance" for n in script.advance_errors},
        snapshot_error=script.snapshot_error,
    )


class ScriptedSimulation(Simulation):
    """The simulation with the replayed world's traits it lacks: a token tap stops at the
    episode horizon, as GuestEnvironment's does (sandbox/actions.py:240-244, 548); the task ends
    once the script's ``ends_after`` actions have taken effect; on the wall clock every end check
    asked within one of the script's ``failing_reads_s`` fails; and it has no renderer, so the
    render barrier commits no frame (sandbox/actions.py:286-292)."""

    def __init__(self, config, world, script: Script) -> None:
        super().__init__(config, world)
        self.script = script
        self.applied = 0

    def _execute(self, decision: int, action, release_at) -> None:
        super()._execute(decision, action, release_at)
        if self.token:
            self.tick = min(self.tick, self.config.max_ticks)
        world = self.world
        if decision in world.rejections or decision in world.execute_errors:
            return
        self.applied += 1
        if self.applied == self.script.ends_after:
            now = self.semantic()
            world.ends_at = now if world.ends_at is None else min(world.ends_at, now)

    def _run(self, command) -> None:
        if isinstance(command, CheckEnd) and self.script.read_fails_at(self.semantic()):
            self.world.check_errors[len(self.checks) + 1] = STATE_READ_FAILED
        if isinstance(command, RenderFrame):
            self.commands.append(command)
            self.renders.append(self.semantic())
            self.kernel.post(FrameRendered(False))
            return
        super()._run(command)


def simulate(script: Script) -> Projection:
    config = turn_config(script)
    sim = ScriptedSimulation(config, world(script), script)
    sim.run()
    return project(script, sim)


def project(script: Script, sim: ScriptedSimulation) -> Projection:
    token = script.mode == "token"
    seconds = 1.0 / script.env_fps if token else 1.0  # semantic time in world seconds
    decisions: dict[int, dict] = {}

    def of(k: int) -> dict:
        return decisions.setdefault(k, decision())

    for entry in sim.records:
        data = entry.data
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
                d["wait_record"] = dict(data["execution_result"])
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
            d = of(outcome.decision)
            d["commit"] = {"status": outcome.status, "detail": outcome.detail}
            d["outcome"] = {
                "status": outcome.status,
                "terminal_after_action": outcome.terminal_after_action,
                "action_clock_position": outcome.action_clock_position,
                "next_observation_tick": outcome.next_observation_tick,
                "dropped_intermediate_frames": outcome.dropped_intermediate_frames,
                "post_action_rendered": outcome.post_action_rendered,
                # The lifecycle record keeps six decimals (evaluation/episode_records.py:543).
                "frame_age_s": None
                if outcome.frame_age_s is None
                else round(outcome.frame_age_s, 6),
                "token_delay_frames": outcome.token_delay_frames,
                "action_output_tokens": outcome.action_output_tokens,
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
            [{"at": at * seconds, "keys": keys} for at, _, keys in sim.key_events],
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
    }
    return Projection("simulation", {"decisions": decisions, "episode": episode})
