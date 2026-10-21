"""run_stream's model-input frame grid and finite-B horizon, against exact arithmetic.

Frame k is due at k * env fps / model-input fps ticks and observed at the first whole tick at or
after it; on the finite token clock the remaining budget is the remaining model-input frames times
B, and an action whose complete response lands past the horizon is not applied (the module
docstring of evaluation/runner.py). The event kernel computes these in floats, with tolerances in
_floor, _ceil and the remaining allowance (evaluation/kernel/turns.py); the cases here sit where
an exact answer and a float computed without those tolerances can part. A scripted agent answers
each decision with an action and its output tokens; the environment is GuestEnvironment on a task
clock (token_env.py).
"""

from __future__ import annotations

import math
from fractions import Fraction

from realtime_framework.clock import Clocks
from realtime_framework.entity.actions import Action
from realtime_framework.entity.model import PendingDecision
from realtime_framework.evaluation import runner

from .clock_task import ClockTask
from .frames import FrameSource
from .token_env import TokenEnvironment


class ScriptedAgent:
    """What the shell calls on an agent (evaluation/shell.py KernelAgent). Each generation takes
    the next ``(action, output tokens)`` of the script; a decision past its end fails the
    generation, which ends the run."""

    def __init__(self, script: list[tuple[Action | None, int]]):
        self.script = list(script)
        self.statuses: list[str] = []

    def generate(self, generation, decision, observation, budget) -> PendingDecision:
        action, output_tokens = self.script.pop(0)
        result = {"action": action, "latency": 0.0, "provider_output_tokens": output_tokens}
        return PendingDecision(generation, decision, result)

    def accept(self, generation) -> None:
        pass

    def discard(self, generation) -> None:
        pass

    def commit(self, outcome) -> None:
        self.statuses.append(outcome.status)

    def abort(self, generation=None) -> None:
        pass


def run(
    env_fps: float,
    model_input_fps: float,
    tokens_per_frame: int | None,
    max_minutes: float,
    script: list[tuple[Action | None, int]],
) -> tuple[TokenEnvironment, ScriptedAgent]:
    """run_stream with the arguments main() passes for these rates."""
    task = ClockTask(env_fps, tokens_per_frame)
    frames = FrameSource(task, text_observation=False, fps=model_input_fps)
    env = TokenEnvironment(frames, task, max_minutes=max_minutes)
    agent = ScriptedAgent(script)
    runner.run_stream(
        agent,
        env,
        clock_mode="token",
        duration_s=max_minutes * 60,
        env_fps=env_fps,
        input_fps=model_input_fps,
        tokens_per_frame=tokens_per_frame,
        log=lambda line: None,
        clocks=Clocks(env.clock, env.clock),
    )
    return env, agent


def captured_ticks(env: TokenEnvironment) -> list:
    return [call["tick"] for call in env.calls if call["call"] == "env.capture"]


def lifecycle(env: TokenEnvironment) -> list[dict]:
    return [call["record"] for call in env.calls if call["call"] == "env.record_lifecycle"]


def test_pause_mode_observes_every_model_input_frame_at_the_first_tick_at_or_after_it():
    """9/7 in pause mode, horizon 3.28125 s = 29.53125 ticks: frames 0 to 22, then frame 23
    (29.57) is past the horizon. With nothing to do the world moves one frame per decision. Frame 2
    (2.57) is the first observed at a tick (3) closer to the next frame than to its own. Frame 7 is
    due on tick 9, which 9 / (9/7) puts a float below frame 7; frame 21 is due on tick 27, which
    21 * (9/7) puts a float above. The kernel's ticks * 7 / 9 and frames * 9 / 7 give both
    exactly (kernel/turns.py _frames, _ticks)."""
    frames = 23
    env, agent = run(9.0, 7.0, None, 0.0546875, [(None, 0)] * frames)
    assert captured_ticks(env) == [math.ceil(Fraction(9 * k, 7)) for k in range(frames)]
    assert agent.statuses == ["none"] * frames
    assert [r["dropped_intermediate_frames"] for r in lifecycle(env)] == [0] * frames
    assert env._stop_reason == "tick_limit"


def test_a_response_past_the_horizon_is_not_applied_when_its_allowance_is_not_whole():
    """30/7, B=256, 5 minutes = 9000 ticks. Decision 0's 10 tokens land at 0.17; frame 1 (4.29)
    is observed at tick 5. From there 8995 ticks are 2098.83 frames, 537301.33 tokens: a response
    of 10^7 tokens is charged 537301 of them, which would land at 8999.994, inside the horizon.
    The whole response lands past it, so the wait is not applied and the clock stops at 9000."""
    env, agent = run(30.0, 7.0, 256, 5.0, [(None, 10), (Action.wait(1.0), 10**7)])
    records = lifecycle(env)
    assert captured_ticks(env) == [0, 5]
    assert [r["remaining_clock_tokens"] for r in records] == [537600, 537301]
    assert records[1]["post_horizon_output_tokens"] == 10**7 - 537301
    assert agent.statuses == ["none", "not_applied_tick_limit"]
    assert records[1]["action_clock_position"] == 9000.0
    assert env.tick_index() == 9000.0
    assert env._stop_reason == "tick_limit"


def test_a_large_allowance_is_rounded_down_within_float_noise():
    """25/4, B=100000, 5 minutes = 7500 ticks; a frame is 6.25 ticks. Decision 0's 52998400
    tokens are 529.984 frames, 3312.4 ticks; frame 530 (3312.5) is observed at tick 3313. The
    4187 ticks left are 669.92 frames, exactly 66992000 tokens, which floats put a hundred-
    millionth of a token below. A response of all of them lands on the horizon."""
    env, agent = run(25.0, 4.0, 100000, 5.0, [(None, 52998400), (Action.wait(1.0), 66992000)])
    records = lifecycle(env)
    assert captured_ticks(env) == [0, 3313]
    assert [r["remaining_clock_tokens"] for r in records] == [120000000, 66992000]
    assert records[1]["post_horizon_output_tokens"] == 0
    assert agent.statuses == ["none", "not_applied_tick_limit"]
    assert env._stop_reason == "tick_limit"
