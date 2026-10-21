"""GuestEnvironment's primitives for the shell that runs the event kernel (evaluation/shell.py).

The kernel splits what the legacy loop's calls do in one go: it asks the task whether the episode
is over without the environment comparing a deadline or invalidating the run itself. On the wall
clock the fake clock (tests/fake_clock.py) times the environment; on the token clock a task clock
without a browser (tests/replay/clock_task.py) records what the environment pushes into it.
"""

from __future__ import annotations

import pytest

from realtime_framework.clock import Clocks, FrameClock
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.sandbox.actions import GuestEnvironment
from realtime_framework.sandbox.budget import BudgetExceeded

from ..fake_clock import FakeClock
from ..replay.clock_task import ClockTask

ORIGIN = 1_000.0


class _Guest:
    """Runs no command; records each with the virtual time it ran at."""

    def __init__(self, clock):
        self.clock = clock
        self.commands = []

    def execute_command(self, command, timeout=30):
        self.commands.append((self.clock.monotonic(), command))
        return {"returncode": 0}


class _WallTask:
    """A web task's clock at 1 fps; GuestEnvironment takes its tick from it."""

    def clock_snapshot(self):
        return {"fps": 1.0}


def _wall(clock, horizon_s=60.0, **kwargs):
    return GuestEnvironment(
        _Guest(clock),
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


# ---- step_clock -----------------------------------------------------------------------------


def _advances(task):
    return [(event["frames"], len(event["key_events"])) for event in _of(task, "advance")]


def _of(task, *kinds):
    return [event for event in task.events if event["event"] in kinds]


def test_a_clock_step_neither_ends_at_a_lease_nor_releases_nor_checks_the_end():
    checks = []
    env, task = _token(stop_check=lambda: checks.append(env.tick_index()))
    assert env.execute(Action.hold_key("a", 0.5)) == "applied"
    env.step_clock(1.0)
    assert _advances(task) == [(1.0, 0)]
    assert env.tick_index() == 1 and "a" in env._held_keys
    assert _of(task, "keyUp") == [] and checks == []


def test_a_clock_step_stops_at_the_episode_horizon():
    env, task = _token()
    env.set_episode_horizon(1.5)
    for _ in range(3):
        env.step_clock(1.0)
    assert _advances(task) == [(1.0, 0), (0.5, 0)]
    assert env.tick_index() == 1.5


def test_clock_steps_deliver_the_auto_repeat_of_a_held_key_as_the_legacy_wait_does():
    legacy, legacy_task = _token()
    shell, shell_task = _token()
    for env in (legacy, shell):
        assert env.execute(Action.hold_key("left", 5.0)) == "applied"
    legacy.wait_until_tick(3)
    for _ in range(3):
        shell.step_clock(1.0)
    assert shell_task.events == legacy_task.events
    assert _advances(shell_task) == [(1.0, 10), (1.0, 20), (1.0, 20)]


def test_a_failed_clock_step_raises_and_latches_the_advance_failure():
    env, task = _token()

    def advance_clock(frames=1, key_events=()):
        raise RuntimeError("browser frame advanced without a compositor commit")

    task.advance_clock = advance_clock
    with pytest.raises(RuntimeError, match="compositor commit"):
        env.step_clock(1.0)
    assert env._invalid_run["reason"] == "clock_advance_failed"
    assert env.tick_index() == 0


# ---- last_held_keys -------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["left", "arrowleft"])
def test_a_token_hold_reports_the_identities_of_its_keys(key):
    """The kernel ends a lease by these identities, so they carry the environment's aliases."""
    env, _ = _token()
    assert env.execute(Action.hold_key(f"shift+{key}", 1.0)) == "applied"
    assert env.last_held_keys == ("shift", "arrowleft")


def test_a_token_hold_that_extends_a_held_key_reports_it_too():
    env, task = _token()
    env.execute(Action.hold_key("a", 2.0))
    assert env.execute(Action.hold_key("a+b", 1.0)) == "applied"
    assert env.last_held_keys == ("a", "b")
    assert [event["key"] for event in _of(task, "keyDown")] == ["a", "b"]


def test_only_a_token_hold_that_lasts_reports_held_keys():
    env, _ = _token()
    env.execute(Action.hold_key("a", 1.0))
    for action in (
        Action.hold_key("b", 0.0),
        Action(ActionType.KEY, {"key": "c"}),
        Action.wait(1.0),
    ):
        assert env.execute(action) in {"applied", "wait"}
        assert env.last_held_keys == ()


def test_a_wall_hold_reports_no_held_keys():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    with clock.participant():
        assert env.execute(Action.hold_key("arrowleft", 1.0)) == "applied"
    assert env.last_held_keys == ()
