"""Each environment operation fails once it runs past its overall budget, and the run is invalid;
only the key release at the cleanup after the outcome is latched leaves the run as it was.

The real run_stream drives the real GuestEnvironment (sandbox/actions.py). Its GuestClient talks to
FakeGuest; a real catalog game (realtime_tasks/game/runtime.py) reads its state, advances the token
clock and takes key input over FakeCDP. Each case slows one operation down: it must fail within its
budget and latch the invalid-run reason the runner already uses for that operation.

The cases at the end run on the fake clock (tests/fake_clock.py), where an operation's parts spend
virtual time: which budget times which part is exact there, and nothing waits in real time.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest
import requests

import realtime_tasks.registry as tasks
from realtime_framework.clock import Clocks, FrameClock
from realtime_framework.entity.actions import MAX_ACTION_DURATION, Action, ActionType, Observation
from realtime_framework.evaluation import runner
from realtime_framework.sandbox import actions
from realtime_framework.sandbox.browser.cdp import client_for, close_port
from realtime_framework.sandbox.budget import BudgetExceeded, Deadline, operation_budget
from realtime_framework.sandbox.guest import GuestClient, GuestRefused, action_timeout
from realtime_tasks.game.runtime import _GET_STATE
from realtime_tasks.types import Task

from ..fake_clock import FakeClock
from ..replay.clock_task import ClockTask
from .fakes import SCREENSHOT_PNG, FakeCDP, FakeGuest

BUDGET = 0.5
# A failing operation must end by its budget plus this much scheduling slack, still well before
# the timeouts that bounded it before (3 s and more).
SLACK = 1.5
GAME = "gh_javascript_pong"
PLAYING = {"game_state": {"player_returns": 0}, "terminal": {"isTerminal": False}}
CLICK = Action(ActionType.CLICK, {"x": 10, "y": 10})
TAP = Action(ActionType.KEY, {"key": "space"})
DRAG_PATH = {"x1": 0, "y1": 0, "x2": 5, "y2": 5}


class ScriptedAgent:
    """What run_stream calls on an adapter (runner.py:973-984, 473-478); always the same action."""

    def __init__(self, action: Action):
        self.action = action
        self.statuses: list[str] = []

    def configure_decision(self, budget) -> None:
        pass

    def stream_prefill(self, observation) -> None:
        pass

    def stream_generate(self, tick: int) -> dict:
        return {"action": self.action, "latency": 0.0}

    def commit_action(self, status: str, detail: str = "") -> None:
        self.statuses.append(status)


def _token_clock(task: Task) -> Task:
    """The task clock the runner configures before it builds the environment (runner.py:1624)."""
    task.configure_clock("token", 1.0)
    return task


@pytest.fixture
def budgets(monkeypatch):
    """Shrink the configured operation budgets for one test."""

    def shrink(**fields):
        value = replace(actions.ENVIRONMENT_BUDGET, **fields)
        monkeypatch.setattr(actions, "ENVIRONMENT_BUDGET", value)
        monkeypatch.setattr(runner, "ENVIRONMENT_BUDGET", value)

    return shrink


@pytest.fixture
def game(monkeypatch):
    """The catalog game on a FakeCDP port, its token clock configured and active."""
    task = tasks.get(GAME)
    ports = []

    def on(fake: FakeCDP):
        monkeypatch.setattr(task, "dbg_port", fake.port, raising=False)
        monkeypatch.setattr(task, "_score_start", None, raising=False)
        monkeypatch.setattr(task, "_score_best", None, raising=False)
        monkeypatch.setattr(task, "_clock_config", None, raising=False)
        task.configure_clock("token", 1.0)
        task._clock_config["active"] = True
        ports.append(fake.port)
        return task

    yield on
    for port in ports:
        close_port(port)


def _environment(guest: FakeGuest, task: Task | None = None, **options):
    """A GuestEnvironment on the token clock at 1 fps; by default its task has only the clock."""
    return actions.GuestEnvironment(
        GuestClient(host="127.0.0.1", port=guest.port),
        None,
        task or _token_clock(Task()),
        honor_fail=False,
        clock_mode="token",
        **options,
    )


class _WallTask(Task):
    """A web task's clock at 10 fps on the wall clock; GuestEnvironment takes its tick from it."""

    def clock_snapshot(self) -> dict:
        return {"fps": 10.0}


def _wall_environment(guest: FakeGuest) -> actions.GuestEnvironment:
    """A GuestEnvironment on the wall clock, whose key holds run xdotool on the guest."""
    now = time.time()
    return actions.GuestEnvironment(
        GuestClient(host="127.0.0.1", port=guest.port),
        now + 60.0,
        _WallTask(),
        honor_fail=False,
        clock_mode="wall",
        wall_start_time=now,
    )


def _run(env, action: Action = CLICK) -> tuple[ScriptedAgent, float]:
    """Three decisions of run_stream, each taking ``action``; returns the agent and the wall time
    taken."""
    agent = ScriptedAgent(action)
    started = time.monotonic()
    runner.run_stream(agent, env, max_ticks=3, log=lambda line: None, clock_mode="token")
    return agent, time.monotonic() - started


def _spent_the_budget(elapsed: float) -> None:
    assert BUDGET - 0.05 <= elapsed < BUDGET + SLACK, elapsed


def _invalid(env, reason: str, operation: str, budget: float = BUDGET) -> None:
    """The run is invalid for ``reason``, and the detail names ``operation``'s exceeded budget."""
    assert env._invalid_run["category"] == "environment"
    assert env._invalid_run["reason"] == reason
    assert f"{operation} took longer than {budget:g} s" in env._invalid_run["detail"]


def test_an_action_budget_extends_only_for_text_input():
    """A valid action lasts at most MAX_ACTION_DURATION, 10 s (entity/actions.py), so the
    max(20, duration + 5) of a timed action stays 20 s; text input extends to 120 s as the
    guest's request does (sandbox/guest.py action_timeout), whose own 10 s a timed action does
    extend."""
    budget = actions.ENVIRONMENT_BUDGET.action_s
    assert budget == 20.0
    assert action_timeout(CLICK, budget) == 20.0
    assert action_timeout(Action(ActionType.TYPE, {"text": "Ada"}), budget) == 120.0
    hold = Action(ActionType.HOLD_KEY, {"key": "a", "duration": MAX_ACTION_DURATION})
    drag = Action(ActionType.DRAG, {**DRAG_PATH, "duration": MAX_ACTION_DURATION})
    assert action_timeout(hold, budget) == action_timeout(drag, budget) == 20.0
    assert action_timeout(drag, 10.0) == 15.0


def test_the_freeze_s_snapshot_read_has_16_s_and_a_token_key_event_10_s():
    assert actions.ENVIRONMENT_BUDGET.snapshot_read_s == 16.0
    assert actions.ENVIRONMENT_BUDGET.key_event_s == 10.0


def test_a_run_within_its_budgets_reads_acts_captures_and_freezes(game, monkeypatch):
    with FakeGuest() as guest, FakeCDP(values={_GET_STATE: PLAYING}) as fake:
        task = game(fake)
        # FakeCDP runs no page clock; reading, acting, capturing and freezing are what count here.
        monkeypatch.setattr(task, "advance_clock", lambda frames=1, key_events=(): None)
        env = _environment(guest, task, stop_check=task.episode_over)
        agent, _ = _run(env)
    assert getattr(env, "_invalid_run", None) is None
    assert env._stop_reason == "tick_limit"
    assert env._termination_snapshot["state"] == PLAYING
    assert agent.statuses == ["applied"] * 3
    assert len(guest.actions) == 3


def test_a_slow_action_fails_within_its_budget(budgets):
    budgets(action_s=BUDGET)
    with FakeGuest(action_delay=5) as guest:
        env = _environment(guest)
        agent, elapsed = _run(env)
    _invalid(env, "action_execution_failed", "click action")
    assert agent.statuses == ["invalid_run"]
    _spent_the_budget(elapsed)


def _until_the_release_is_in_flight(guest: FakeGuest) -> None:
    """Wait until the release timer's keyup request has reached the guest."""
    deadline = time.monotonic() + 5.0
    while not any("keyup" in command["command"] for command in guest.commands):
        assert time.monotonic() < deadline, "the release timer sent no keyup"
        time.sleep(0.01)


def test_a_wall_key_action_waiting_for_a_key_release_in_flight_fails(budgets):
    # The release timer holds the lock on the wall holds during its guest request
    # (sandbox/actions.py _wall_release_due), and the action reads the holds under that lock.
    # The wait is not cut, but it spends the action's budget: no key reaches the guest after it.
    budgets(action_s=BUDGET)
    with FakeGuest() as guest:
        env = _wall_environment(guest)
        assert env.execute(Action.hold_key("a", 1.0)) == "applied"
        guest.command_delay = 3 * BUDGET
        _until_the_release_is_in_flight(guest)
        started = time.monotonic()
        status = env.execute(Action(ActionType.KEY, {"key": "c"}))
        elapsed = time.monotonic() - started
        env.release_inputs()
    assert status == "failed"
    _invalid(env, "action_execution_failed", "key action")
    assert elapsed >= BUDGET
    assert guest.actions == []


def test_a_wall_shortcut_refused_after_its_budget_went_on_the_lock_is_still_refused(budgets):
    """Coordinator's decision N19, as user decision 35: the refusal says the environment works.
    The model is told why, without the budget's words."""
    budgets(action_s=BUDGET)
    with FakeGuest() as guest:
        env = _wall_environment(guest)
        assert env.execute(Action.hold_key("a", 1.0)) == "applied"
        guest.command_delay = 3 * BUDGET
        _until_the_release_is_in_flight(guest)
        status = env.execute(Action(ActionType.KEY, {"key": "ctrl+l"}))
        env.release_inputs()
    assert status == "failed"
    assert env._last_rejected_by_guest is True
    assert getattr(env, "_invalid_run", None) is None
    assert env._last_exec_error.startswith("browser shortcut 'ctrl+l' is disabled")
    assert env._last_execution_result["error"].startswith(
        f"key action took longer than {BUDGET:g} s (ActionRejected: browser shortcut"
    )


def test_a_wall_action_waiting_for_the_lock_after_the_guest_answered_fails(budgets):
    # The release timer sends its keyup while the guest runs the click; the hold snapshot the
    # click's result carries then waits for the lock.
    budgets(action_s=2.0)
    with FakeGuest() as guest:
        env = _wall_environment(guest)
        assert env.execute(Action.hold_key("a", 0.5)) == "applied"
        guest.command_delay = 2.0
        guest.action_delay = 1.2
        started = time.monotonic()
        status = env.execute(CLICK)
        elapsed = time.monotonic() - started
        env.release_inputs()
    assert status == "failed"
    _invalid(env, "action_execution_failed", "click action", 2.0)
    assert elapsed >= 2.0
    assert len(guest.actions) == 1


def test_a_wall_xdotool_command_gets_what_is_left_of_the_action_s_budget(budgets):
    # A tap of a held key presses it again with xdotool (sandbox/actions.py _restore_wall_holds),
    # a guest request with 15 s of its own.
    budgets(action_s=BUDGET)
    with FakeGuest() as guest:
        env = _wall_environment(guest)
        assert env.execute(Action.hold_key("a", MAX_ACTION_DURATION)) == "applied"
        guest.command_delay = 5.0
        started = time.monotonic()
        status = env.execute(Action(ActionType.KEY, {"key": "a"}))
        elapsed = time.monotonic() - started
        guest.command_delay = 0.0
        env.release_inputs()
    assert status == "failed"
    _invalid(env, "action_execution_failed", "key action")
    _spent_the_budget(elapsed)


def test_a_slow_capture_fails_within_its_budget(budgets):
    budgets(capture_s=BUDGET)
    with FakeGuest(screenshot_delay=5) as guest:
        env = _environment(guest)
        agent, elapsed = _run(env)
    _invalid(env, "observation_capture_failed", "capture")
    assert agent.statuses == []
    _spent_the_budget(elapsed)


def test_a_state_read_waiting_for_the_cdp_lock_fails_within_its_budget(budgets, game):
    budgets(state_read_s=BUDGET)
    with FakeGuest() as guest, FakeCDP(values={_GET_STATE: PLAYING}) as fake:
        task = game(fake)
        held, release = threading.Event(), threading.Event()

        def hold():
            with client_for(fake.port).transaction():
                held.set()
                release.wait(10)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        held.wait(5)
        try:
            env = _environment(guest, stop_check=task.episode_over)
            agent, elapsed = _run(env)
        finally:
            release.set()
            holder.join(5)
    _invalid(env, "terminal_state_check_failed", "state read")
    assert fake.commands == []
    _spent_the_budget(elapsed)


def test_a_clock_advance_with_a_slow_connect_fails_within_its_budget(budgets, game, monkeypatch):
    budgets(clock_advance_s=BUDGET)
    with FakeGuest() as guest, FakeCDP(handshake_delay=5) as fake:
        task = game(fake)
        # Only the advance is timed here; the snapshot read at the freeze would connect again.
        monkeypatch.setattr(task, "evaluation_snapshot", None)
        env = _environment(guest, task)
        agent, elapsed = _run(env)
    _invalid(env, "clock_advance_failed", "clock advance")
    assert env._stop_reason == "invalid_run"
    assert len(guest.actions) == 1
    assert fake.commands == []
    _spent_the_budget(elapsed)


def test_a_token_key_waiting_for_the_cdp_lock_fails_within_its_budget(budgets, game, monkeypatch):
    # The game sends the key in one CDP transaction (sandbox/browser/clock.py cdp_key_event);
    # before, waiting for its lock had no limit.
    budgets(key_event_s=BUDGET)
    with FakeGuest() as guest, FakeCDP() as fake:
        task = game(fake)
        monkeypatch.setattr(task, "advance_clock", lambda frames=1, key_events=(): None)
        monkeypatch.setattr(task, "evaluation_snapshot", None)
        held, release = threading.Event(), threading.Event()

        def hold():
            with client_for(fake.port).transaction():
                held.set()
                release.wait(10)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        held.wait(5)
        try:
            env = _environment(guest, task)
            agent, elapsed = _run(env, TAP)
        finally:
            release.set()
            holder.join(5)
    _invalid(env, "action_execution_failed", "key press")
    assert agent.statuses == ["invalid_run"]
    assert fake.commands == []
    _spent_the_budget(elapsed)


# ---- on the fake clock -------------------------------------------------------------------------


def _sandbox_call(clock: FakeClock, seconds: float, timeout: float = 120.0) -> None:
    """Take ``seconds`` as a call made inside an operation does: a CDP call (sandbox/browser/
    cdp.py) or a guest request (sandbox/guest.py) ends where its own timeout or the operation's
    budget ends, whichever is first, and fails then."""
    deadline = Deadline(timeout, "sandbox call")
    clock.sleep(min(seconds, deadline.remaining()))
    deadline.remaining()


class _ClockGuest:
    """The guest agent's screenshot and action endpoints; each action takes ``seconds`` of
    ``clock``."""

    def __init__(self, clock: FakeClock | None = None, seconds: float = 0.0):
        self.clock = clock
        self.seconds = seconds
        self.actions: list[ActionType] = []

    def screenshot(self) -> bytes:
        return SCREENSHOT_PNG

    def execute_action(self, action: Action) -> dict:
        self.actions.append(action.action_type)
        if self.seconds:
            _sandbox_call(self.clock, self.seconds, action_timeout(action, 10.0))
        return {"status": "ok"}


class _BarrierTask(ClockTask):
    """A web task's action barrier (realtime_tasks/web/runtime.py begin_gui_action and
    complete_gui_action), whose wait for the page's business request takes ``seconds``; its CDP
    call has 8 s (sandbox/browser/clock.py cdp_wait_realtime_action_barrier)."""

    def __init__(self, clock: FakeClock, seconds: float):
        super().__init__(1.0)
        self.clock = clock
        self.seconds = seconds

    def begin_gui_action(self) -> dict:
        return {}

    def complete_gui_action(self, snapshot: dict) -> dict:
        _sandbox_call(self.clock, self.seconds, 8.0)
        return {"ok": True}


def _timed(
    clock: FakeClock, guest: _ClockGuest | None = None, task: ClockTask | None = None, **options
) -> tuple[actions.GuestEnvironment, ClockTask]:
    """A GuestEnvironment on the token clock at 1 fps, timed on ``clock``; by default its task
    takes key input as a game's does (tests/replay/clock_task.py)."""
    task = task or ClockTask(1.0)
    env = actions.GuestEnvironment(
        guest, None, task, clock_mode="token", clocks=Clocks(FrameClock(1.0), clock), **options
    )
    return env, task


def _run_on(clock: FakeClock, env) -> ScriptedAgent:
    """Three decisions of run_stream on ``clock``, each clicking."""
    agent = ScriptedAgent(CLICK)
    runner.run_stream(
        agent, env, max_ticks=3, log=lambda line: None, clock_mode="token", clock=clock
    )
    return agent


def _swallowing_the_timeout(clock: FakeClock) -> None:
    """A read through cdp_eval (sandbox/browser/clock.py), which turns a timeout into no value; a
    CDP call that never answers spends the operation's budget."""
    try:
        _sandbox_call(clock, 3600.0)
    except TimeoutError:
        pass


def test_a_state_read_whose_task_swallows_the_timeout_fails_within_its_budget():
    """The game then raises its own error (realtime_tasks/game/runtime.py episode_over), which
    the read reports past its budget. That unrelated CDP events do not restart the call's timeout
    is test_cdp_deadline.py's."""
    clock = FakeClock()

    def episode_over():
        _swallowing_the_timeout(clock)
        raise RuntimeError("game state backend returned no state")

    env, _ = _timed(clock, _ClockGuest(), stop_check=episode_over)
    with clock.participant():
        agent = _run_on(clock, env)
        assert clock.monotonic() == 120.0
    _invalid(env, "terminal_state_check_failed", "state read", 120.0)
    assert env._invalid_run["detail"].rstrip().endswith("returned no state)")
    assert agent.statuses == []


def test_the_snapshot_read_at_the_freeze_fails_within_its_short_budget():
    """The game's snapshot is then an empty state (realtime_tasks/game/runtime.py
    evaluation_snapshot); the read still fails because it returned after its deadline."""
    clock = FakeClock()
    task = ClockTask(1.0)

    def evaluation_snapshot():
        _swallowing_the_timeout(clock)
        return {"kind": "browser_game", "state": {}}

    task.evaluation_snapshot = evaluation_snapshot
    env, _ = _timed(clock, _ClockGuest(), task, stop_check=lambda: True)
    with clock.participant():
        agent = _run_on(clock, env)
        assert clock.monotonic() == 16.0
    _invalid(env, "evaluation_snapshot_failed", "snapshot read", 16.0)
    assert env._termination_snapshot_captured is False
    assert agent.statuses == []


def test_an_action_and_its_page_barrier_share_one_budget():
    # The guest answers the longest drag, 10 s, within the action's 20 s and its own 15 s; the
    # page barrier never does. Without the shared budget the barrier's own 8 s CDP timeout would
    # apply.
    clock = FakeClock()
    guest = _ClockGuest(clock, 14.0)
    env, _ = _timed(clock, guest, _BarrierTask(clock, 3600.0))
    drag = Action(ActionType.DRAG, {**DRAG_PATH, "duration": MAX_ACTION_DURATION})
    with clock.participant():
        assert env.execute(drag) == "failed"
        assert clock.monotonic() == 20.0
    _invalid(env, "action_execution_failed", "drag action", 20.0)
    assert guest.actions == [ActionType.DRAG]


def test_a_call_that_ran_out_of_the_operation_s_budget_is_its_failure():
    """The CDP call inside the advance spends the advance's deadline and fails with its message,
    which the operation does not repeat."""
    clock = FakeClock()
    env, task = _timed(clock)
    task.advance_clock = lambda frames=1, key_events=(): _sandbox_call(clock, 3600.0)
    with clock.participant(), pytest.raises(TimeoutError):
        env.wait_until_tick(1)
    assert env._invalid_run["reason"] == "clock_advance_failed"
    assert env._invalid_run["detail"] == "clock advance took longer than 120 s"


def _raise_the_budget_s_words(clock: FakeClock) -> None:
    clock.sleep(2.0)
    raise RuntimeError("state read took longer than 1 s")


def _run_out_of_its_own_shorter_deadline(clock: FakeClock) -> None:
    """A call with its own 0.5 s whose wait is not cut finds its deadline passed."""
    deadline = Deadline(0.5, "sandbox call")
    clock.sleep(2.0)
    deadline.remaining()


@pytest.mark.parametrize(
    ("fail", "cause"),
    [
        pytest.param(
            _raise_the_budget_s_words,
            "RuntimeError: state read took longer than 1 s",
            id="another-error-in-its-words",
        ),
        pytest.param(
            _run_out_of_its_own_shorter_deadline,
            "BudgetExceeded: sandbox call took longer than 0.5 s",
            id="a-call-s-own-deadline",
        ),
    ],
)
def test_only_the_operation_s_own_budget_exceeded_goes_through_as_it_is(fail, cause):
    clock = FakeClock()
    with (
        clock.participant(),
        pytest.raises(BudgetExceeded) as raised,
        operation_budget("state read", 1.0, clock=clock),
    ):
        fail(clock)
    assert str(raised.value) == f"state read took longer than 1 s ({cause})"
    assert raised.value.__cause__ is not None


def test_a_task_failure_past_the_budget_keeps_its_traceback():
    clock = FakeClock()

    def read_the_game_state():
        clock.sleep(121.0)
        raise RuntimeError("no state")

    env, _ = _timed(clock, stop_check=read_the_game_state)
    with clock.participant():
        assert env.done()
    detail = env._invalid_run["detail"]
    assert "in read_the_game_state" in detail
    assert detail.rstrip().endswith(
        "BudgetExceeded: state read took longer than 120 s (RuntimeError: no state)"
    )


def test_text_input_has_its_extended_budget():
    clock = FakeClock()
    env, _ = _timed(clock, _ClockGuest(clock, 100.0))
    with clock.participant():
        assert env.execute(Action(ActionType.TYPE, {"text": "Ada"})) == "applied"
        assert clock.monotonic() == 100.0
    assert getattr(env, "_invalid_run", None) is None


def test_a_call_s_own_shorter_timeout_applies_inside_an_operation():
    """The capture's probe stands for a window probe, which gives up after its own 8 s
    (sandbox/browser/cdp.py _probe_one) while the capture goes on without it
    (sandbox/observation/capture.py _probe)."""
    clock = FakeClock()

    class ProbingCapture:
        def capture(self, tick):
            try:
                _sandbox_call(clock, 3600.0, 8.0)
            except TimeoutError:
                pass
            return Observation(SCREENSHOT_PNG, clock.time(), tick)

    env, _ = _timed(clock, scene_capture=ProbingCapture())
    with clock.participant():
        env.capture()
        assert clock.monotonic() == 8.0
    assert getattr(env, "_invalid_run", None) is None


def test_a_guest_refusal_that_arrives_after_the_action_s_budget_is_still_a_refusal():
    """The guest answered, so the environment works and was only slow (user decision 35). The
    refusal is raised as GuestClient raises a 4xx from /action (sandbox/guest.py
    execute_action)."""
    clock = FakeClock()
    response = requests.Response()
    response.status_code = 400

    class LateRefusal(_ClockGuest):
        def execute_action(self, action: Action) -> dict:
            self.actions.append(action.action_type)
            clock.sleep(25.0)
            raise GuestRefused("400 guest rejected the action", response=response)

    env, _ = _timed(clock, LateRefusal())
    with clock.participant():
        assert env.execute(CLICK) == "failed"
        assert clock.monotonic() == 25.0
    assert env._last_rejected_by_guest is True
    assert getattr(env, "_invalid_run", None) is None
    # The model is told the guest's words, not that the action took long; the record keeps both.
    assert env._last_exec_error == "400 guest rejected the action"
    assert env._last_execution_result["error"] == (
        "click action took longer than 20 s (GuestRefused: 400 guest rejected the action)"
    )


class _MissingCommandEndpoint:
    """The guest's command endpoint, which runs xset at once and answers an xdotool keydown with
    a 404 after ``seconds`` (GuestClient.execute_command raises it from raise_for_status)."""

    def __init__(self, clock: FakeClock, seconds: float):
        self.clock = clock
        self.seconds = seconds

    def execute_command(self, command: str, timeout: int = 30) -> dict:
        if "keydown" not in command:
            return {"returncode": 0}
        self.clock.sleep(self.seconds)
        response = requests.Response()
        response.status_code = 404
        raise requests.HTTPError("404 Client Error: NOT FOUND for url: /execute", response=response)


@pytest.mark.parametrize("seconds", [0.0, 25.0], ids=["in-time", "after-the-budget"])
def test_a_4xx_from_the_guest_s_command_endpoint_is_an_environment_failure(seconds):
    """Only the guest's refusal of an action shows the environment works (user decision 35)."""
    clock = FakeClock(wall_origin=1_000.0)
    env = actions.GuestEnvironment(
        _MissingCommandEndpoint(clock, seconds),
        1_060.0,
        _WallTask(),
        honor_fail=False,
        clock_mode="wall",
        wall_start_time=1_000.0,
        clocks=Clocks(clock, clock),
    )
    with clock.participant():
        assert env.execute(Action.hold_key("a", 1.0)) == "failed"
    assert env._last_rejected_by_guest is False
    assert env._invalid_run["reason"] == "action_execution_failed"


def test_an_action_s_guest_request_and_page_barrier_may_together_take_more_than_10_s():
    """Each stays within its own timeout (10 s, 8 s); an action has 20 s for both, where it had
    10 s at first."""
    clock = FakeClock()
    env, _ = _timed(clock, _ClockGuest(clock, 5.3), _BarrierTask(clock, 5.0))
    drag = Action(ActionType.DRAG, {**DRAG_PATH, "duration": 5})
    with clock.participant():
        assert env.execute(drag) == "applied"
        assert clock.monotonic() == pytest.approx(10.3)
    assert getattr(env, "_invalid_run", None) is None


def _slow_key_event(
    clock: FakeClock, task: ClockTask, index: int, seconds: float, *, delivered: bool = False
) -> None:
    """The task's key event number ``index`` from now on takes ``seconds``; the others take none.
    A slow event that runs out of its budget reached the page only when ``delivered``."""
    sent = []
    record = task.clock_key_event

    def clock_key_event(event_type, key, auto_repeat=False, held_modifiers=()):
        sent.append(event_type)
        slow = len(sent) == index + 1
        if slow and delivered:
            record(event_type, key, auto_repeat=auto_repeat, held_modifiers=held_modifiers)
        if slow:
            _sandbox_call(clock, seconds)
        if not (slow and delivered):
            record(event_type, key, auto_repeat=auto_repeat, held_modifiers=held_modifiers)

    task.clock_key_event = clock_key_event


@pytest.mark.parametrize(
    ("held", "action", "index", "operation"),
    [
        pytest.param(None, TAP, 0, "key press", id="a-tap-s-press"),
        pytest.param(None, TAP, 1, "key release", id="a-tap-s-release"),
        pytest.param("space", TAP, 2, "key press", id="pressing-a-tapped-held-key-again"),
        pytest.param(None, Action.hold_key("a", 1.0), 0, "key press", id="a-hold-s-press"),
        pytest.param(None, Action.hold_key("a", 0.0), 1, "key release", id="a-0-s-hold-s-release"),
    ],
)
def test_a_token_key_event_past_its_budget_fails_the_action(held, action, index, operation):
    clock = FakeClock()
    env, task = _timed(clock)
    if held is not None:
        env.execute(Action.hold_key(held, 5.0))
    _slow_key_event(clock, task, index, 60.0)
    budget = actions.ENVIRONMENT_BUDGET.key_event_s
    with clock.participant():
        assert env.execute(action) == "failed"
        assert clock.monotonic() == budget
    _invalid(env, "action_execution_failed", operation, budget)


def test_a_tap_s_frame_between_its_press_and_release_keeps_its_own_budgets():
    """The tap renders one frame with its key down (sandbox/actions.py TAP_HOLD_S): that advance
    and the end check after it are timed by the clock-advance and state-read budgets."""
    clock = FakeClock()
    env, task = _timed(clock, stop_check=lambda: _sandbox_call(clock, 15.0))
    advance = task.advance_clock

    def advance_clock(frames=1, key_events=()):
        _sandbox_call(clock, 15.0)
        return advance(frames, key_events)

    task.advance_clock = advance_clock
    assert actions.ENVIRONMENT_BUDGET.key_event_s < 15.0
    with clock.participant():
        assert env.execute(TAP) == "applied"
        assert clock.monotonic() == 30.0
    assert [event["event"] for event in task.events] == ["keyDown", "advance", "keyUp"]
    assert getattr(env, "_invalid_run", None) is None


def test_each_key_event_of_a_repeated_tap_has_its_own_budget():
    clock = FakeClock()
    env, task = _timed(clock)
    record = task.clock_key_event

    def clock_key_event(*args, **kwargs):
        _sandbox_call(clock, 6.0)
        record(*args, **kwargs)

    task.clock_key_event = clock_key_event
    assert actions.ENVIRONMENT_BUDGET.key_event_s < 12.0
    with clock.participant():
        assert env.execute(Action(ActionType.KEY, {"key": "space", "repeat": 3})) == "applied"
        assert clock.monotonic() == 36.0


def test_a_lease_s_release_past_its_budget_invalidates_the_run():
    """The legacy loop's advance releases a key whose lease ended (sandbox/actions.py
    wait_until_tick)."""
    clock = FakeClock()
    env, task = _timed(clock)
    env.execute(Action.hold_key("a", 0.5))
    _slow_key_event(clock, task, 0, 60.0)
    budget = actions.ENVIRONMENT_BUDGET.key_event_s
    with clock.participant():
        assert env.execute(Action.wait(1.0)) == "failed"
        assert clock.monotonic() == budget
    _invalid(env, "clock_key_release_failed", "key release", budget)
    assert env.tick_index() == 0.5


@pytest.mark.parametrize(
    ("delivered", "keyups"), [(False, ["a"]), (True, ["a", "a"])], ids=["lost", "delivered-late"]
)
def test_a_key_whose_release_failed_stays_held_and_the_cleanup_releases_it_again(delivered, keyups):
    """The environment cannot tell whether a release that ran out of its budget reached the
    page, so the key may still be down: the cleanup releases it again, a second keyUp when the
    first one did arrive."""
    clock = FakeClock()
    env, task = _timed(clock)
    env.execute(Action.hold_key("a", 0.5))
    _slow_key_event(clock, task, 0, 60.0, delivered=delivered)
    with clock.participant():
        assert env.execute(Action.wait(1.0)) == "failed"
        assert list(env._held_keys) == ["a"]
        env.release_inputs()
    assert [event["key"] for event in task.events if event["event"] == "keyUp"] == keyups
    assert env._held_keys == {}


@pytest.mark.parametrize("slow", [True, False], ids=["slow", "failing"])
def test_a_key_release_at_cleanup_that_fails_leaves_the_run_as_it_was(slow):
    """release_inputs runs once run_stream has latched the outcome, and the runner only logs its
    failure (evaluation/runner.py _finalize); main() would still take an invalid run latched here
    for the run's."""
    clock = FakeClock()
    guest = _ClockGuest()
    env, task = _timed(clock, guest)
    env.execute(Action.hold_key("a", 5.0))
    env.execute(Action(ActionType.MOUSE_DOWN, {"button": "left"}))
    budget = actions.ENVIRONMENT_BUDGET.key_event_s

    def clock_key_event(*args, **kwargs):
        if not slow:
            raise RuntimeError("the page is gone")
        _sandbox_call(clock, 60.0)

    task.clock_key_event = clock_key_event
    error = f"key release took longer than {budget:g} s" if slow else "the page is gone"
    with clock.participant():
        with pytest.raises(Exception, match=error):
            env.release_inputs()
        assert clock.monotonic() == (budget if slow else 0.0)
    assert getattr(env, "_invalid_run", None) is None
    assert guest.actions == [ActionType.MOUSE_DOWN, ActionType.MOUSE_UP]
