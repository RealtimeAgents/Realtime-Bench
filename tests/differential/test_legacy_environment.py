"""GuestEnvironment as the legacy loop had it on the wall clock (legacy_environment.py): it compares
the deadline it was built with in its waits, its WAIT and done(). Production's takes none."""

from __future__ import annotations

import inspect

import pytest

from realtime_framework.clock import Clocks
from realtime_framework.entity.actions import Action
from realtime_framework.sandbox.actions import GuestEnvironment

from ..fake_clock import FakeClock
from ..test_environment_clock import ORIGIN, _Guest, _WallTask
from .legacy_environment import LegacyGuestEnvironment


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
