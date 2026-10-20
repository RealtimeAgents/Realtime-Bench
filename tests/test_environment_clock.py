"""GuestEnvironment on the episode clocks (sandbox/actions.py). On the wall clock the fake clock
times its waits, its deadline and its key releases; under the token clock its tick is the
semantic FrameClock, which only advancing the environment moves."""

from __future__ import annotations

from realtime_framework.clock import Clocks, FrameClock
from realtime_framework.entity.actions import Action
from realtime_framework.sandbox.actions import ENVIRONMENT_BUDGET, GuestEnvironment
from realtime_framework.sandbox.budget import Deadline

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


def _wall(clock, horizon_s, **kwargs):
    return GuestEnvironment(
        _Guest(clock),
        ORIGIN + horizon_s,
        _WallTask(),
        clock_mode="wall",
        wall_start_time=ORIGIN,
        clocks=Clocks(clock, clock),
        **kwargs,
    )


def test_wall_wait_is_cut_at_the_deadline():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock, 3.0)
    with clock.participant():
        clock.sleep(1.25)
        assert env.execute(Action.wait(5.0)) == "wait"
        assert clock.monotonic() == 3.0
        assert env._last_execution_result["duration_simulated_seconds"] == 1.75
        assert env.time_remaining() == 0.0 and env.past_deadline() and env.done()


def test_wall_wait_for_the_next_tick_polls_every_30_ms():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock, 10.0)
    with clock.participant():
        clock.sleep(0.5)
        env.wait_until_tick(2)
        assert env.tick_index() == 2
        assert 2.0 <= clock.monotonic() <= 2.03 + 1e-9


def test_wall_hold_is_released_by_a_timer_on_the_clock():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock, 60.0)
    with clock.participant():
        clock.sleep(0.25)
        assert env.execute(Action.hold_key("arrowleft", 1.5)) == "applied"
        clock.sleep(5.0)
    assert [entry for entry in env.g.commands if "xdotool" in entry[1]] == [
        (0.25, "xdotool keydown -- Left"),
        (1.75, "xdotool keyup -- Left"),
    ]


def test_an_environment_operation_is_timed_on_the_operational_clock():
    """A state read past its budget invalidates the run; a call inside it, such as a CDP call,
    spends what is left of the same deadline (sandbox/budget.py)."""
    clock = FakeClock(wall_origin=ORIGIN)
    budget = ENVIRONMENT_BUDGET.state_read_s
    left = []

    def stop_check():
        left.append(Deadline(None, "CDP call").remaining())
        clock.sleep(budget + 1.0)

    env = _wall(clock, 10 * budget, stop_check=stop_check)
    with clock.participant():
        assert env.done()
    assert left == [budget] and clock.monotonic() == budget + 1.0
    invalid = env._invalid_run
    assert invalid["reason"] == "terminal_state_check_failed"
    assert invalid["detail"].rstrip().endswith("BudgetExceeded: state read took longer than 120 s")


def test_token_world_time_moves_only_when_the_environment_advances():
    operational = FakeClock()
    clocks = Clocks(FrameClock(2.0), operational)
    env = GuestEnvironment(None, None, ClockTask(2.0), clock_mode="token", clocks=clocks)
    with operational.participant():
        operational.sleep(300.0)  # e.g. the provider watchdog, while the world is frozen
        assert env.tick_index() == 0 and clocks.semantic.monotonic() == 0.0
        env.wait_until_tick(3)
        assert env.tick_index() == clocks.semantic.frame == 3
        assert clocks.semantic.monotonic() == 1.5
        assert operational.monotonic() == 300.0
