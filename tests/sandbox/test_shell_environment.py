"""GuestEnvironment's primitives for the shell that runs the event kernel (evaluation/shell.py).

The kernel splits what the legacy loop's calls do in one go: it asks the task whether the episode
is over without the environment comparing a deadline or invalidating the run itself. On the wall
clock the fake clock (tests/fake_clock.py) times the environment; on the token clock a task clock
without a browser (tests/replay/clock_task.py) records what the environment pushes into it.
"""

from __future__ import annotations

import threading
import time

import pytest

import realtime_tasks.registry as tasks
from realtime_framework.clock import Clocks, FrameClock
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.sandbox.actions import (
    ENVIRONMENT_BUDGET,
    TAP_HOLD_S,
    WALL_RELEASE_MAX_ATTEMPTS,
    WALL_RELEASE_RETRY_S,
    GuestEnvironment,
)
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


def _wall(clock, **kwargs):
    return GuestEnvironment(
        _Guest(clock),
        _WallTask(),
        clock_mode="wall",
        wall_start_time=ORIGIN,
        clocks=Clocks(clock, clock),
        **kwargs,
    )


def _token(fps=1.0, **kwargs):
    task = ClockTask(fps)
    return GuestEnvironment(None, task, clock_mode="token", **kwargs), task


# ---- check_end ------------------------------------------------------------------------------

CLOCK_MODES = pytest.mark.parametrize("clock_mode", ["wall", "token"])


def _on(clock_mode, clock, **kwargs):
    """A GuestEnvironment on ``clock_mode``, whose operations ``clock`` times."""
    if clock_mode == "wall":
        return _wall(clock, **kwargs)
    env, _ = _token(clocks=Clocks(FrameClock(1.0), clock), **kwargs)
    return env


def test_an_end_check_asks_the_task_in_an_invalid_run():
    """The legacy loop's done() answered it without asking the task, and compared the deadline
    too (tests/differential/legacy_environment.py); the kernel's check does neither."""
    clock = FakeClock(wall_origin=ORIGIN)
    asked = []
    env = _wall(clock, stop_check=lambda: asked.append(clock.monotonic()))
    env._invalid_run = {"category": "environment", "reason": "action_execution_failed"}
    with clock.participant():
        clock.sleep(2.0)
        assert env.check_end() == (False, None)
    assert asked == [2.0]
    assert env._invalid_run == {"category": "environment", "reason": "action_execution_failed"}


def test_a_token_end_check_asks_the_task_in_an_invalid_run():
    asked = []
    env, _ = _token(stop_check=lambda: asked.append(True))
    env._invalid_run = {"category": "environment", "reason": "clock_advance_failed"}
    assert env.check_end() == (False, None)
    assert asked == [True]
    assert env._invalid_run == {"category": "environment", "reason": "clock_advance_failed"}


@CLOCK_MODES
@pytest.mark.parametrize(("answer", "reason"), [("task_success", "task_success"), (True, None)])
def test_an_end_check_latches_the_end_and_the_reason_the_task_names(clock_mode, answer, reason):
    env = _on(clock_mode, FakeClock(wall_origin=ORIGIN), stop_check=lambda: answer)
    assert env.check_end() == (True, reason)
    assert env._episode_over is True
    assert env._terminal_reason == reason


@CLOCK_MODES
def test_a_failed_end_check_raises_and_leaves_the_run_valid(clock_mode):
    def stop_check():
        raise RuntimeError("game state backend returned no state")

    env = _on(clock_mode, FakeClock(wall_origin=ORIGIN), stop_check=stop_check)
    with pytest.raises(RuntimeError, match="returned no state"):
        env.check_end()
    assert getattr(env, "_invalid_run", None) is None
    assert env._episode_over is False


@CLOCK_MODES
def test_an_end_check_without_a_task_check_never_ends(clock_mode):
    env = _on(clock_mode, FakeClock(wall_origin=ORIGIN))
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
        env = _wall(clock, stop_check=stop_check)
    else:
        env, _ = _token(stop_check=stop_check, clocks=Clocks(FrameClock(1.0), clock))
    with clock.participant():
        assert env.check_end() == (False, None)
        with pytest.raises(BudgetExceeded, match=f"state read took longer than {budget_s:g} s"):
            env.check_end()
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
    for action in (Action.hold_key("b", 0.0), Action(ActionType.KEY, {"key": "c"})):
        assert env.execute(action) == "applied"
        assert env.last_held_keys == ()


def test_a_wall_hold_reports_no_held_keys():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    with clock.participant():
        assert env.execute(Action.hold_key("arrowleft", 1.0)) == "applied"
    assert env.last_held_keys == ()


# ---- release_keys ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["left", "arrowleft"])
def test_a_release_finds_the_keys_by_the_identities_their_hold_reported(key):
    env, task = _token()
    env.execute(Action.hold_key(f"shift+{key}", 1.0))
    env.release_keys(env.last_held_keys)
    (up,) = _of(task, "keyUp")
    assert up["key"] == f"shift+{key}"
    assert env._held_keys == {} and env._repeat_key is None


def test_a_release_skips_keys_that_are_not_held():
    env, task = _token()
    env.execute(Action.hold_key("a+b", 1.0))
    env.release_keys(("b", "c"))
    env.release_keys(("c",))
    (up,) = _of(task, "keyUp")
    assert (up["key"], up["held_modifiers"]) == ("b", [])
    assert list(env._held_keys) == ["a"]


def test_keys_released_together_go_up_in_the_order_they_were_pressed():
    """The kernel names them in the order of its leases (evaluation/kernel/turns.py), which can
    differ from the order of the presses; the task gets the chord the legacy wait sends
    (_release_expired_clock_keys) in the order of the presses. The page's keyUp events for a
    chord are cdp_key_event's (sandbox/browser/clock.py), tested with it."""
    env, task = _token()
    env.execute(Action.hold_key("shift+a", 1.0))
    env.release_keys(("a", "shift"))
    (up,) = _of(task, "keyUp")
    assert up["key"] == "shift+a"


def test_clock_steps_and_releases_in_the_kernel_s_order_give_the_task_the_legacy_wait_s_events():
    """The kernel ends a step where a lease ends and releases the key after it
    (evaluation/kernel/turns.py _advance); wait_until_tick does both in one call."""
    legacy, legacy_task = _token()
    shell, shell_task = _token()
    for env in (legacy, shell):
        assert env.execute(Action.hold_key("left", 1.5)) == "applied"
    legacy.wait_until_tick(3)
    shell.step_clock(1.0)
    shell.step_clock(0.5)
    shell.release_keys(shell.last_held_keys)
    shell.step_clock(1.0)
    shell.step_clock(0.5)
    assert shell_task.events == legacy_task.events
    assert [event["event"] for event in shell_task.events] == [
        "keyDown",
        "advance",
        "advance",
        "keyUp",
        "advance",
        "advance",
    ]


def test_a_key_release_fails_past_its_budget():
    clock = FakeClock()
    env, task = _token(clocks=Clocks(FrameClock(1.0), clock))
    env.execute(Action.hold_key("a", 1.0))
    budget = ENVIRONMENT_BUDGET.key_event_s
    task.clock_key_event = lambda *args, **kwargs: clock.sleep(budget + 1.0)
    with (
        clock.participant(),
        pytest.raises(BudgetExceeded, match=f"key release took longer than {budget:g} s"),
    ):
        env.release_keys(("a",))


# ---- execute(release_at=...) ----------------------------------------------------------------


def _xdotool(env):
    return [entry for entry in env.g.commands if entry[1].startswith("xdotool")]


def test_a_wall_hold_is_released_at_the_kernel_s_time_not_a_duration_after_the_press():
    """The kernel sends the hold's release with the Execute: the time it sent it plus the
    duration, in semantic seconds since the origin (evaluation/kernel/turns.py _pre_checked)."""
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    with clock.participant():
        clock.sleep(0.25)
        assert env.execute(Action.hold_key("arrowleft", 1.5), release_at=1.5) == "applied"
        clock.sleep(5.0)
    assert _xdotool(env) == [(0.25, "xdotool keydown -- Left"), (1.5, "xdotool keyup -- Left")]


class _SlowFocus:
    """Gives the page its keyboard focus back (sandbox/observation/capture.py ensure_page_focus)
    in ``seconds``."""

    def __init__(self, clock, seconds):
        self.clock = clock
        self.seconds = seconds

    def ensure_page_focus(self):
        self.clock.sleep(self.seconds)
        return True


@pytest.mark.parametrize(
    ("duration", "release_at", "held_s"),
    [
        pytest.param(0.2, 1.5, 1.0, id="released-after-the-press"),
        pytest.param(0.2, 0.2, TAP_HOLD_S, id="release-time-passed-before-the-press"),
        pytest.param(0.2, 0.51, TAP_HOLD_S, id="release-time-less-than-a-tap-after-the-press"),
        pytest.param(0.01, 0.01, 0.01, id="a-hold-shorter-than-a-tap-keeps-its-length"),
    ],
)
def test_a_wall_hold_lasts_to_the_kernel_s_release_time_or_a_short_while_more(
    duration, release_at, held_s
):
    """The press waits for the page focus. A release time that has passed by then, or comes
    sooner, still leaves the key down min(duration, TAP_HOLD_S) more, 1/60 s or the whole hold if
    it is shorter; the execution result says how long, next to the duration asked for."""
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock, scene_capture=_SlowFocus(clock, 0.5))
    with clock.participant():
        assert env.execute(Action.hold_key("arrowleft", duration), release_at=release_at) == (
            "applied"
        )
        clock.sleep(5.0)
    result = env._last_execution_result
    assert result["duration_seconds"] == duration
    assert result["lease_seconds"] == round(held_s, 3)
    times, commands = zip(*_xdotool(env))
    assert commands == ("xdotool keydown -- Left", "xdotool keyup -- Left")
    assert times == pytest.approx((0.5, 0.5 + held_s))


class _SlowKeydownGuest(_Guest):
    """Its xdotool keydown reaches the desktop ``seconds`` after it is sent."""

    def __init__(self, clock, seconds):
        super().__init__(clock)
        self.seconds = seconds

    def execute_command(self, command, timeout=30):
        if "keydown" in command:
            self.clock.sleep(self.seconds)
        return super().execute_command(command, timeout)


def test_a_delayed_wall_hold_counts_its_short_while_from_the_press():
    """The keydown's own request takes time as well: the short while counts from when it
    returned. The lease the execution result gives counts from when it was sent, as the keyup's
    trip to the desktop will take about as long as the keydown's."""
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock, scene_capture=_SlowFocus(clock, 0.5))
    env.g = _SlowKeydownGuest(clock, 0.05)
    with clock.participant():
        assert env.execute(Action.hold_key("arrowleft", 0.2), release_at=0.2) == "applied"
        clock.sleep(5.0)
    times, commands = zip(*_xdotool(env))
    assert commands == ("xdotool keydown -- Left", "xdotool keyup -- Left")
    assert times == pytest.approx((0.55, 0.55 + TAP_HOLD_S))
    assert env._last_execution_result["lease_seconds"] == round(0.05 + TAP_HOLD_S, 3)


def test_a_wall_hold_on_time_records_its_lease_from_its_keydown_request():
    """The keydown's trip back is not taken off the lease (the keyup's trip there adds it)."""
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    env.g = _SlowKeydownGuest(clock, 0.03)
    with clock.participant():
        assert env.execute(Action.hold_key("arrowleft", 1.0), release_at=1.0) == "applied"
        clock.sleep(5.0)
    times, _ = zip(*_xdotool(env))
    assert times == pytest.approx((0.03, 1.0))
    assert env._last_execution_result["lease_seconds"] == 1.0


def test_a_legacy_wall_hold_records_no_lease():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    with clock.participant():
        assert env.execute(Action.hold_key("arrowleft", 1.0)) == "applied"
        clock.sleep(5.0)
    assert "lease_seconds" not in env._last_execution_result


def test_a_wall_hold_keeps_the_later_release_of_a_key_held_already():
    """The execution result gives the second hold's own lease, which the held key outlasts."""
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    with clock.participant():
        env.execute(Action.hold_key("arrowleft", 3.0), release_at=3.0)
        clock.sleep(1.0)
        env.execute(Action.hold_key("arrowleft+a", 1.0), release_at=2.0)
        clock.sleep(5.0)
    assert _xdotool(env) == [
        (0.0, "xdotool keydown -- Left"),
        (1.0, "xdotool keydown -- a"),
        (2.0, "xdotool keyup -- a"),
        (3.0, "xdotool keyup -- Left"),
    ]
    assert env._last_execution_result["lease_seconds"] == 1.0


# ---- the wall's key commands ----------------------------------------------------------------


class _TappingGuest(_Guest):
    """Also takes the guest's own key actions, which press and release their keys."""

    def execute_action(self, action):
        self.commands.append((self.clock.monotonic(), f"tap {action.parameters['key']}"))
        return {"status": "ok"}


def test_a_wall_hold_of_a_held_key_only_extends_its_lease():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    with clock.participant():
        env.execute(Action.hold_key("arrowleft", 1.0), release_at=1.0)
        env.execute(Action.hold_key("arrowleft", 3.0), release_at=3.0)
        clock.sleep(5.0)
    assert _xdotool(env) == [(0.0, "xdotool keydown -- Left"), (3.0, "xdotool keyup -- Left")]


def test_a_wall_hold_of_a_digit_goes_through_xdotool():
    """xdotool names letters and digits as they are (sandbox/actions.py _xdotool_keysym); the
    guest's own hold would block the action for the hold's duration."""
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    with clock.participant():
        assert env.execute(Action.hold_key("1", 1.0), release_at=1.0) == "applied"
        clock.sleep(5.0)
    assert _xdotool(env) == [(0.0, "xdotool keydown -- 1"), (1.0, "xdotool keyup -- 1")]


def test_a_tap_of_a_wall_held_key_presses_it_again_by_its_x_keysym():
    """The guest's tap releases the held key; the lease goes on with the key down again."""
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    env.g = _TappingGuest(clock)
    with clock.participant():
        env.execute(Action.hold_key("left", 3.0), release_at=3.0)
        clock.sleep(1.0)
        assert env.execute(Action(ActionType.KEY, {"key": "left"})) == "applied"
        clock.sleep(5.0)
    assert env._last_execution_result["restored_wall_holds"] == ["left"]
    assert [entry for entry in env.g.commands if not entry[1].startswith("xset")] == [
        (0.0, "xdotool keydown -- Left"),
        (1.0, "tap left"),
        (1.0, "xdotool keydown -- Left"),
        (3.0, "xdotool keyup -- Left"),
    ]


# ---- on_invalid -----------------------------------------------------------------------------


class _StuckGuest(_Guest):
    """A guest whose keyup fails, so the key stays physically down."""

    def execute_command(self, command, timeout=30):
        super().execute_command(command, timeout)
        if "keyup" in command:
            return {"returncode": 1, "stderr": "cannot release"}
        return {"returncode": 0}


def test_a_wall_release_that_keeps_failing_tells_the_shell_from_the_timer_s_thread():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    env.g = _StuckGuest(clock)
    test_thread = threading.current_thread()
    told = []
    env.on_invalid = lambda record: told.append(
        (clock.monotonic(), record, threading.current_thread() is not test_thread)
    )
    with clock.participant():
        env.execute(Action.hold_key("arrowleft", 1.0), release_at=1.0)
        clock.sleep(5.0)
    keyups = [at for at, command in env.g.commands if "keyup" in command]
    assert keyups == [1.0, 1.5, 2.0, 2.5]
    assert told == [(2.5, env._invalid_run, True)]
    assert env._invalid_run["reason"] == "wall_key_release_failed"


def test_the_cleanup_releases_a_key_the_release_timer_gave_up_on():
    """The key may still be down, as with a failed token-clock release (coordinator's decisions
    N20, N23): the timer stops trying it, also with a later lease's keyup, and the cleanup tries
    once more."""
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    env.g = _StuckGuest(clock)
    with clock.participant():
        env.execute(Action.hold_key("arrowleft", 1.0), release_at=1.0)
        clock.sleep(3.0)
        env.execute(Action.hold_key("a", 1.0), release_at=4.0)
        clock.sleep(2.0)
        with pytest.raises(RuntimeError, match="cannot release"):
            env.release_inputs()
    assert _xdotool(env) == [
        (0.0, "xdotool keydown -- Left"),
        *((at, "xdotool keyup -- Left") for at in (1.0, 1.5, 2.0, 2.5)),
        (3.0, "xdotool keydown -- a"),
        (4.0, "xdotool keyup -- a"),
        (5.0, "xdotool keyup -- Left a"),
    ]


def test_the_shell_is_told_outside_the_wall_locks():
    """The callback is the shell's (sandbox/actions.py _wall_release_due): told outside the
    environment's locks, it cannot hold up an action waiting for one."""
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    env.g = _StuckGuest(clock)
    owned = []
    env.on_invalid = lambda record: owned.append(
        (env._wall_lock._is_owned(), env._wall_commands._is_owned())
    )
    with clock.participant():
        env.execute(Action.hold_key("arrowleft", 1.0), release_at=1.0)
        clock.sleep(5.0)
    assert owned == [(False, False)]


class _FlakyGuest(_Guest):
    """A guest whose keyup succeeds at every third try."""

    def execute_command(self, command, timeout=30):
        super().execute_command(command, timeout)
        tries = sum("keyup" in entry for _, entry in self.commands)
        if "keyup" in command and tries % 3:
            return {"returncode": 1, "stderr": "cannot release"}
        return {"returncode": 0}


def test_a_wall_release_that_succeeds_starts_the_count_of_failed_tries_over():
    """Each lease's keyup fails twice before it goes through: four failures in all, but never
    WALL_RELEASE_MAX_ATTEMPTS in a row."""
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    env.g = _FlakyGuest(clock)
    told = []
    env.on_invalid = told.append
    with clock.participant():
        env.execute(Action.hold_key("a", 1.0), release_at=1.0)
        clock.sleep(2.5)
        env.execute(Action.hold_key("b", 0.5), release_at=3.0)
        clock.sleep(5.0)
    keyups = [(at, command) for at, command in env.g.commands if "keyup" in command]
    assert keyups == [
        *((at, "xdotool keyup -- a") for at in (1.0, 1.5, 2.0)),
        *((at, "xdotool keyup -- b") for at in (3.0, 3.5, 4.0)),
    ]
    assert told == [] and getattr(env, "_invalid_run", None) is None
    assert env._wall_holds == {}


def test_a_wall_release_failure_in_a_run_already_invalid_tells_the_shell_nothing():
    clock = FakeClock(wall_origin=ORIGIN)
    env = _wall(clock)
    env.g = _StuckGuest(clock)
    told = []
    env.on_invalid = told.append
    with clock.participant():
        env.execute(Action.hold_key("arrowleft", 1.0), release_at=1.0)
        env._invalid_run = {"category": "environment", "reason": "action_execution_failed"}
        clock.sleep(5.0)
    assert len([command for _, command in env.g.commands if "keyup" in command]) == 4
    assert told == []
    assert env._invalid_run["reason"] == "action_execution_failed"


class _RacingGuest:
    """In real time: the keydown of b takes KEYDOWN_S, and every keyup of b fails."""

    KEYDOWN_S = 0.15

    def __init__(self):
        self.started = time.monotonic()
        self.commands = []

    def execute_command(self, command, timeout=30):
        self.commands.append((time.monotonic() - self.started, command))
        if command == "xdotool keydown -- b":
            time.sleep(self.KEYDOWN_S)
        if command == "xdotool keyup -- b":
            return {"returncode": 1, "stderr": "cannot release"}
        return {"returncode": 0}


def test_a_release_timer_replaced_while_it_waited_for_a_keydown_leaves_one_retry_chain():
    """a's lease ends while b's keydown is in flight: a's timer has fired and waits for the key
    commands when the hold arms the next one. The replaced timer stands down, so b's keyup is
    tried WALL_RELEASE_MAX_ATTEMPTS times WALL_RELEASE_RETRY_S apart, not twice as often. In real
    time: the fake clock cannot tell a thread waiting for a lock from a busy one."""
    env = GuestEnvironment(
        _RacingGuest(), _WallTask(), clock_mode="wall", wall_start_time=time.time()
    )
    told = []
    env.on_invalid = told.append
    env.execute(Action.hold_key("a", 0.1))
    time.sleep(0.05)
    env.execute(Action.hold_key("b", 0.2))
    time.sleep(0.25 + WALL_RELEASE_MAX_ATTEMPTS * WALL_RELEASE_RETRY_S)
    keyups = [at for at, command in env.g.commands if command == "xdotool keyup -- b"]
    assert len(keyups) == WALL_RELEASE_MAX_ATTEMPTS
    assert min(later - earlier for earlier, later in zip(keyups, keyups[1:])) > (
        WALL_RELEASE_RETRY_S - 0.1
    )
    assert [record["reason"] for record in told] == ["wall_key_release_failed"]


# ---- the end checks inside a token key tap --------------------------------------------------

TAP = Action(ActionType.KEY, {"key": "space"})


def _answers(*answers):
    """A task check that gives ``answers`` in order, raising those that are exceptions."""
    answers = iter(answers)

    def stop_check():
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return stop_check


def test_a_tap_s_failed_reads_go_to_the_kernel_and_the_tap_goes_on():
    """At 120 fps a tap holds its key two ticks, with an end check after each (sandbox/actions.py
    TAP_HOLD_S); while the kernel counts, no failed read stops the tap or invalidates the run."""
    env, task = _token(120.0, stop_check=_answers(RuntimeError("no state"), False))
    assert env.execute(TAP) == "applied"
    (failed, _, error), answered = env.last_end_checks
    assert not failed and error.rstrip().endswith("RuntimeError: no state")
    assert answered == (False, None, None)
    assert env.tick_index() == 2
    assert [event["event"] for event in task.events] == ["keyDown", "advance", "advance", "keyUp"]
    assert getattr(env, "_invalid_run", None) is None


def test_a_tap_s_reads_that_run_past_their_budget_go_to_the_kernel_too():
    """A wedged environment, X12's main case: each read times out (BudgetExceeded)."""
    clock = FakeClock()

    def stop_check():
        clock.sleep(ENVIRONMENT_BUDGET.state_read_s + 1.0)
        return False

    env, _ = _token(120.0, stop_check=stop_check, clocks=Clocks(FrameClock(120.0), clock))
    with clock.participant():
        assert env.execute(TAP) == "applied"
    assert len(env.last_end_checks) == 2
    for ended, reason, error in env.last_end_checks:
        assert (ended, reason) == (False, None)
        assert error.rstrip().endswith("BudgetExceeded: state read took longer than 120 s")
    assert getattr(env, "_invalid_run", None) is None


def test_the_end_checks_of_a_repeated_tap_are_kept_in_order():
    no_state = RuntimeError("no state")
    env, _ = _token(stop_check=_answers(no_state, False, no_state))
    env.execute(Action(ActionType.KEY, {"key": "space", "repeat": 3}))
    first, second, third = env.last_end_checks
    assert first[:2] == (False, None) and first[2].rstrip().endswith("RuntimeError: no state")
    assert second == (False, None, None)
    assert third[:2] == (False, None) and third[2].rstrip().endswith("RuntimeError: no state")
    assert getattr(env, "_invalid_run", None) is None


def test_a_tap_whose_read_sees_the_end_stops_there():
    env, task = _token(120.0, stop_check=_answers("task_success"))
    assert env.execute(TAP) == "applied"
    assert env.last_end_checks == ((True, "task_success", None),)
    assert env.tick_index() == 1
    assert [event["event"] for event in task.events] == ["keyDown", "advance", "keyUp"]
    assert env._episode_over and env._terminal_reason == "task_success"


def test_a_repeated_tap_whose_first_read_sees_the_end_keeps_it():
    """The reads after it, in the tap's other repeats, find the episode still going."""
    env, _ = _token(stop_check=_answers("task_success", False, False))
    env.execute(Action(ActionType.KEY, {"key": "space", "repeat": 3}))
    assert env.last_end_checks == (
        (True, "task_success", None),
        (False, None, None),
        (False, None, None),
    )
    assert env._episode_over and env._terminal_reason == "task_success"


def test_each_execution_starts_its_end_checks_over():
    env, _ = _token(stop_check=_answers(RuntimeError("no state"), False))
    env.execute(TAP)
    ((_, _, error),) = env.last_end_checks
    assert error is not None
    env.execute(TAP)
    assert env.last_end_checks == ((False, None, None),)
    env.execute(Action.hold_key("a", 1.0))
    assert env.last_end_checks == ()


def test_a_tap_s_end_check_reads_the_game_and_samples_its_score(monkeypatch):
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
    env = GuestEnvironment(None, task, clock_mode="token", stop_check=task.episode_over)
    assert env.execute(TAP) == "applied"
    assert len(reads) == 1
    assert task._score_start == task._score_best == 3.0
    assert env.last_end_checks == ((False, None, None),)
    assert getattr(env, "_invalid_run", None) is None
