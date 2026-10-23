"""GuestEnvironment on the episode clocks (sandbox/actions.py). On the wall clock the fake clock
times its key releases; under the token clock its tick is the semantic FrameClock, which only
advancing the environment moves."""

from __future__ import annotations

import pytest

from realtime_framework.clock import Clocks, FrameClock
from realtime_framework.entity.actions import Action
from realtime_framework.sandbox.actions import GuestEnvironment
from realtime_framework.sandbox.budget import Deadline, operation_budget

from .fake_clock import FakeClock
from .replay.clock_task import ClockTask

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


def _wall(clock, **kwargs):
    return GuestEnvironment(
        _Guest(clock),
        _WallTask(),
        clock_mode="wall",
        wall_start_time=ORIGIN,
        clocks=Clocks(clock, clock),
        **kwargs,
    )


def test_wall_hold_is_released_by_a_timer_on_the_clock():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    with clock.participant():
        clock.sleep(0.25)
        assert env.execute(Action.hold_key("arrowleft", 1.5)) == "applied"
        clock.sleep(5.0)
    assert [entry for entry in env.g.commands if "xdotool" in entry[1]] == [
        (0.25, "xdotool keydown -- Left"),
        (1.75, "xdotool keyup -- Left"),
    ]


def test_the_clocks_must_fit_the_clock_mode():
    """On the wall clock world time is real time, so both clocks are one; under the token clock
    the frame counter steps by the task's tick."""
    clock = FakeClock(wall_origin=ORIGIN)
    with pytest.raises(ValueError, match="^the wall clock reads one clock for world and real time$"):
        GuestEnvironment(
            _Guest(clock),
            _WallTask(),
            clock_mode="wall",
            wall_start_time=ORIGIN,
            clocks=Clocks(clock, FakeClock()),
        )
    with pytest.raises(ValueError, match=r"^the frame clock counts 1 s frames, the task 0.0333"):
        GuestEnvironment(
            None, ClockTask(30.0), clock_mode="token", clocks=Clocks(FrameClock(1.0), clock)
        )


def test_a_deadline_inside_an_operation_is_read_on_its_clock():
    with operation_budget("click action", 10.0, clock=FakeClock()), pytest.raises(
        ValueError, match="^CDP call is timed on another clock than its operation$"
    ):
        Deadline(1.0, "CDP call", clock=FakeClock())


def test_token_world_time_moves_only_when_the_environment_advances():
    operational = FakeClock()
    clocks = Clocks(FrameClock(2.0), operational)
    env = GuestEnvironment(None, ClockTask(2.0), clock_mode="token", clocks=clocks)
    with operational.participant():
        operational.sleep(300.0)  # e.g. the provider watchdog, while the world is frozen
        assert env.tick_index() == 0 and clocks.semantic.monotonic() == 0.0
        env.wait_until_tick(3)
        assert env.tick_index() == clocks.semantic.frame == 3
        assert clocks.semantic.monotonic() == 1.5
        assert operational.monotonic() == 300.0
