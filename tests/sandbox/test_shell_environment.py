"""GuestEnvironment's primitives for the shell that runs the event kernel (evaluation/shell.py).

The kernel splits what the legacy loop's calls do in one go: it asks the task whether the episode
is over without the environment comparing a deadline or invalidating the run itself. On the wall
clock the fake clock (tests/fake_clock.py) times the environment; on the token clock a task clock
without a browser (tests/replay/clock_task.py) records what the environment pushes into it.
"""

from __future__ import annotations

import pytest

from realtime_framework.clock import Clocks, FrameClock
from realtime_framework.sandbox.actions import GuestEnvironment
from realtime_framework.sandbox.budget import BudgetExceeded

from ..fake_clock import FakeClock
from ..replay.clock_task import ClockTask

ORIGIN = 1_000.0


class _WallTask:
    """A web task's clock at 1 fps; GuestEnvironment takes its tick from it."""

    def clock_snapshot(self):
        return {"fps": 1.0}


def _wall(clock, horizon_s=60.0, **kwargs):
    return GuestEnvironment(
        None,
        ORIGIN + horizon_s,
        _WallTask(),
        clock_mode="wall",
        wall_start_time=ORIGIN,
        clocks=Clocks(clock, clock),
        **kwargs,
    )


def _token(fps=1.0, **kwargs):
    task = ClockTask(fps)
    return GuestEnvironment(None, None, task, clock_mode="token", **kwargs), task


# ---- check_end ------------------------------------------------------------------------------


def test_an_end_check_asks_the_task_past_the_deadline_and_in_an_invalid_run():
    """done() answers both without asking the task (sandbox/actions.py done)."""
    clock = FakeClock(wall_origin=ORIGIN)
    asked = []
    env = _wall(clock, 1.0, stop_check=lambda: asked.append(clock.monotonic()))
    env._invalid_run = {"category": "environment", "reason": "action_execution_failed"}
    with clock.participant():
        clock.sleep(2.0)
        assert env.past_deadline() and env.done()
        assert env.check_end() == (False, None)
    assert asked == [2.0]
    assert env._invalid_run == {"category": "environment", "reason": "action_execution_failed"}


@pytest.mark.parametrize(("answer", "reason"), [("task_success", "task_success"), (True, None)])
def test_an_end_check_latches_the_end_and_the_reason_the_task_names(answer, reason):
    env, _ = _token(stop_check=lambda: answer)
    assert env.check_end() == (True, reason)
    assert env._episode_over is True
    assert env._terminal_reason == reason


def test_a_failed_end_check_raises_and_leaves_the_run_valid():
    def stop_check():
        raise RuntimeError("game state backend returned no state")

    env, _ = _token(stop_check=stop_check)
    with pytest.raises(RuntimeError, match="returned no state"):
        env.check_end()
    assert getattr(env, "_invalid_run", None) is None
    assert env._episode_over is False


def test_an_end_check_without_a_task_check_never_ends():
    env, _ = _token()
    assert env.check_end() == (False, None)


@pytest.mark.parametrize(("clock_mode", "budget_s"), [("wall", 10.0), ("token", 120.0)])
def test_one_end_check_is_bounded_by_the_budget_of_its_clock(clock_mode, budget_s):
    """On the wall clock the kernel asks at the environment rate and one read gets 10 s; on the
    token clock it gets state_read_s, 120 s, as before."""
    clock = FakeClock(wall_origin=ORIGIN)
    takes = iter([budget_s - 1.0, budget_s + 1.0])

    def stop_check():
        clock.sleep(next(takes))
        return False

    if clock_mode == "wall":
        env = _wall(clock, 1_000.0, stop_check=stop_check)
    else:
        env, _ = _token(stop_check=stop_check, clocks=Clocks(FrameClock(1.0), clock))
    with clock.participant():
        assert env.check_end() == (False, None)
        with pytest.raises(BudgetExceeded, match=f"state read took longer than {budget_s:g} s"):
            env.check_end()
    assert getattr(env, "_invalid_run", None) is None


def test_the_legacy_loop_s_end_check_keeps_its_budget_on_the_wall_clock():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock, 1_000.0, stop_check=lambda: clock.sleep(11.0))
    with clock.participant():
        assert env.done() is False
    assert getattr(env, "_invalid_run", None) is None
