"""GuestEnvironment as the legacy loop had it (legacy_environment.py): on the wall clock it
compares the deadline it was built with in its waits, its WAIT and done(); on either clock done()
reads the task, and a token step's end check goes through it unless the kernel counts the reads.
Production's takes no deadline and has no done()."""

from __future__ import annotations

import inspect

import pytest

import realtime_tasks.registry as tasks
from realtime_framework.clock import Clocks, FrameClock
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.sandbox.actions import ENVIRONMENT_BUDGET, GuestEnvironment
from realtime_framework.sandbox.budget import Deadline

from ..fake_clock import FakeClock
from ..replay.clock_task import ClockTask
from ..test_environment_clock import ORIGIN, _Guest, _WallTask
from .legacy_environment import LegacyGuestEnvironment

TAP = Action(ActionType.KEY, {"key": "space"})


def _legacy(clock, horizon_s, **kwargs):
    return LegacyGuestEnvironment(
        _Guest(clock),
        ORIGIN + horizon_s,
        _WallTask(),
        clock_mode="wall",
        wall_start_time=ORIGIN,
        clocks=Clocks(clock, clock),
        **kwargs,
    )


def _legacy_token(fps=1.0, **kwargs):
    task = ClockTask(fps)
    return LegacyGuestEnvironment(None, None, task, clock_mode="token", **kwargs), task


def _answers(*answers):
    """A task check that gives ``answers`` in order, raising those that are exceptions."""
    answers = iter(answers)

    def stop_check():
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return stop_check


def test_production_s_environment_has_no_deadline():
    """Decision 11: the kernel judges the deadline, and production carries none of the legacy
    loop's."""
    assert "deadline" not in inspect.signature(GuestEnvironment).parameters
    assert not hasattr(GuestEnvironment, "time_remaining")
    assert not hasattr(GuestEnvironment, "past_deadline")


def test_wall_wait_is_cut_at_the_deadline():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _legacy(clock, 3.0)
    with clock.participant():
        clock.sleep(1.25)
        assert env.execute(Action.wait(5.0)) == "wait"
        assert clock.monotonic() == 3.0
        assert env._last_execution_result["duration_simulated_seconds"] == 1.75
        assert env.time_remaining() == 0.0 and env.past_deadline() and env.done()


class _SleepRecordingClock(FakeClock):
    def __init__(self):
        super().__init__(wall_origin=ORIGIN)
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        super().sleep(seconds)


def test_wall_wait_for_the_next_tick_polls_every_30_ms():
    """From 0.51 s the wait sleeps 30 ms at a time and wakes on the first poll past tick 2."""
    clock = _SleepRecordingClock()
    env = _legacy(clock, 10.0)
    with clock.participant():
        clock.sleep(0.51)
        clock.sleeps.clear()
        env.wait_until_tick(2)
        assert env.tick_index() == 2
        assert clock.sleeps == [0.03] * 50
        assert clock.monotonic() == pytest.approx(2.01)


def test_done_answers_past_the_deadline_without_asking_the_task():
    """The kernel's end check asks it all the same (sandbox/actions.py check_end)."""
    clock = FakeClock(wall_origin=ORIGIN)
    asked = []
    env = _legacy(clock, 1.0, stop_check=lambda: asked.append(clock.monotonic()))
    with clock.participant():
        clock.sleep(2.0)
        assert env.past_deadline() and env.done()
        assert env.check_end() == (False, None)
    assert asked == [2.0]
    assert getattr(env, "_invalid_run", None) is None


def test_production_s_environment_has_no_done():
    """Decision 11: the kernel's shell reads the task with check_end() and counts the reads that
    fail; production carries neither the legacy loop's done() nor its switch."""
    env = GuestEnvironment(None, ClockTask(1.0), clock_mode="token")
    assert not hasattr(GuestEnvironment, "done")
    assert not hasattr(env, "kernel_counts_reads")


# ---- done() and the token step's end check, moved from tests/sandbox and tests/ -----------------


def test_done_answers_an_invalid_run_without_asking_the_task():
    clock = FakeClock(wall_origin=ORIGIN)
    asked = []
    env = _legacy(clock, 10.0, stop_check=lambda: asked.append(clock.monotonic()))
    env._invalid_run = {"category": "environment", "reason": "action_execution_failed"}
    with clock.participant():
        clock.sleep(2.0)
        assert env.done()
    assert asked == []


def test_a_token_done_answers_an_invalid_run_alone():
    """The token clock has no deadline to pass; done() answers an invalid run alone."""
    asked = []
    env, _ = _legacy_token(stop_check=lambda: asked.append(True))
    env._invalid_run = {"category": "environment", "reason": "clock_advance_failed"}
    assert env.done()
    assert asked == []


def test_the_legacy_loop_s_end_check_keeps_its_budget_on_the_wall_clock():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _legacy(clock, 60.0, stop_check=lambda: clock.sleep(11.0))
    with clock.participant():
        assert env.done() is False
    assert getattr(env, "_invalid_run", None) is None


def test_an_environment_operation_is_timed_on_the_operational_clock():
    """A state read past its budget invalidates the run; a call inside it, such as a CDP call,
    spends what is left of the same deadline (sandbox/budget.py)."""
    clock = FakeClock(wall_origin=ORIGIN)
    budget = ENVIRONMENT_BUDGET.state_read_s
    left = []

    def stop_check():
        left.append(Deadline(None, "CDP call").remaining())
        clock.sleep(budget + 1.0)

    env = _legacy(clock, 300.0, stop_check=stop_check)
    with clock.participant():
        assert env.done()
    assert left == [budget] and clock.monotonic() == budget + 1.0
    invalid = env._invalid_run
    assert invalid["reason"] == "terminal_state_check_failed"
    assert invalid["detail"].rstrip().endswith("BudgetExceeded: state read took longer than 120 s")


def test_a_task_failure_past_the_budget_keeps_its_traceback():
    clock = FakeClock()

    def read_the_game_state():
        clock.sleep(121.0)
        raise RuntimeError("no state")

    env, _ = _legacy_token(stop_check=read_the_game_state, clocks=Clocks(FrameClock(1.0), clock))
    with clock.participant():
        assert env.done()
    detail = env._invalid_run["detail"]
    assert "in read_the_game_state" in detail
    assert detail.rstrip().endswith(
        "BudgetExceeded: state read took longer than 120 s (RuntimeError: no state)"
    )


def test_the_legacy_loop_s_tap_invalidates_the_run_at_its_first_failed_read():
    env, _ = _legacy_token(120.0, stop_check=_answers(RuntimeError("no state")))
    assert env.kernel_counts_reads is False
    assert env.execute(TAP) == "applied"
    assert env.last_end_checks == ()
    assert env.tick_index() == 1
    assert env._invalid_run["reason"] == "terminal_state_check_failed"


def test_a_tap_s_failed_reads_go_to_the_kernel_once_it_counts_them():
    env, _ = _legacy_token(120.0, stop_check=_answers(RuntimeError("no state"), False))
    env.kernel_counts_reads = True
    assert env.execute(TAP) == "applied"
    (failed, _, error), answered = env.last_end_checks
    assert not failed and error.rstrip().endswith("RuntimeError: no state")
    assert answered == (False, None, None)
    assert getattr(env, "_invalid_run", None) is None


def test_the_legacy_loop_s_tap_reads_the_game_and_samples_its_score(monkeypatch):
    """A game's episode_over is also its score sampler (realtime_tasks/game/runtime.py)."""
    task = tasks.get("gh_javascript_pong")
    state = {"game_state": {"player_returns": 3}, "terminal": {"isTerminal": False}}
    reads = []
    monkeypatch.setattr(task, "_state", lambda: reads.append(state) or state, raising=False)
    monkeypatch.setattr(task, "_score_start", None, raising=False)
    monkeypatch.setattr(task, "_score_best", None, raising=False)
    monkeypatch.setattr(task, "_clock_config", None, raising=False)
    # No browser: only the state read reaches the game.
    monkeypatch.setattr(task, "advance_clock", lambda frames=1, key_events=(): None)
    monkeypatch.setattr(task, "clock_key_event", lambda *args, **kwargs: None)
    task.configure_clock("token", 1.0)
    env = LegacyGuestEnvironment(None, None, task, clock_mode="token", stop_check=task.episode_over)
    assert env.execute(TAP) == "applied"
    assert len(reads) == 1
    assert task._score_start == task._score_best == 3.0
    assert env.last_end_checks == ()
