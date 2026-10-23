"""Each environment operation fails once it runs past its overall budget, and as a rule the run is
invalid. Not so for a refusal of the model's action that comes after the budget, which is still a
refusal; for a state read the kernel counts, which is invalid at the third failure in a row; and
for the key release at the cleanup after the outcome is latched, which leaves the run as it was.

The real run_stream drives the real GuestEnvironment (sandbox/actions.py). Its GuestClient talks to
FakeGuest; a real catalog game (realtime_tasks/game/runtime.py) reads its state, advances the token
clock and takes key input over FakeCDP. Each case slows one operation down: it must fail within its
budget and latch the invalid-run reason the runner already uses for that operation.

The cases at the end run on the fake clock (tests/fake_clock.py), where an operation's parts spend
virtual time: which budget times which part is exact there, and nothing waits in real time.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import replace

import pytest
import requests

import realtime_tasks.registry as tasks
from realtime_framework.clock import Clocks, FrameClock
from realtime_framework.entity.actions import MAX_ACTION_DURATION, Action, ActionType, Observation
from realtime_framework.entity.harness import StreamingAgent
from realtime_framework.evaluation import runner, shell
from realtime_framework.evaluation.kernel.turns import READ_FAILURE_LIMIT
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


class ScriptedAgent(StreamingAgent):
    """An adapter run through StreamingAgent's legacy turn (entity/harness.py); always the same
    action."""

    def __init__(self, action: Action):
        self.action = action
        self.statuses: list[str] = []

    def initialize(self, task_instruction, initial_observation) -> None:
        pass

    def discard_pending_observations(self) -> None:
        pass

    def stream_prefill(self, observation) -> None:
        pass

    def stream_generate(self, tick: int) -> dict:
        return {"action": self.action, "latency": 0.0}

    def commit_action(self, status: str, detail: str = "") -> None:
        self.statuses.append(status)


def _token_clock(task: Task) -> Task:
    """The task clock the runner configures (runner.py main) and activates (runner.py
    set_up_agent) before it builds the environment."""
    task.configure_clock("token", 1.0)
    task.activate_clock()
    return task


@pytest.fixture
def budgets(monkeypatch):
    """Shrink the configured operation budgets for one test."""

    def shrink(**fields):
        value = replace(actions.ENVIRONMENT_BUDGET, **fields)
        monkeypatch.setattr(actions, "ENVIRONMENT_BUDGET", value)
        monkeypatch.setattr(shell, "ENVIRONMENT_BUDGET", value)

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
    runner.run_stream(agent, env, duration_s=3.0, log=lambda line: None, clock_mode="token")
    return agent, time.monotonic() - started


def _spent_the_budget(elapsed: float, operations: int = 1) -> None:
    assert operations * BUDGET - 0.05 <= elapsed < operations * BUDGET + SLACK, elapsed


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
    """A token key event has as long as the guest's request for a key on the wall clock (user
    decision 34)."""
    assert actions.ENVIRONMENT_BUDGET.snapshot_read_s == 16.0
    assert actions.ENVIRONMENT_BUDGET.key_event_s == 10.0
    assert GuestClient().timeout == actions.ENVIRONMENT_BUDGET.key_event_s


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


@pytest.mark.parametrize(
    "action", [Action(ActionType.KEY, {"key": "c"}), CLICK], ids=["a-tap", "a-click"]
)
def test_a_wall_action_does_not_wait_for_a_key_release_in_flight(budgets, action):
    """The release timer sends its keyup holding only _wall_commands (sandbox/actions.py
    _wall_release_due): the action's checks and its hold snapshot read the leases without waiting
    for it, and the key being released counts as held until the keyup returns."""
    budgets(action_s=BUDGET)
    with FakeGuest() as guest:
        env = _wall_environment(guest)
        assert env.execute(Action.hold_key("a", 0.2)) == "applied"
        guest.command_delay = 2 * SLACK
        _until_the_release_is_in_flight(guest)
        started = time.monotonic()
        status = env.execute(action)
        elapsed = time.monotonic() - started
        holds = env._last_execution_result["active_holds"]
        guest.command_delay = 0.0
        env.release_inputs()
    assert status == "applied"
    assert getattr(env, "_invalid_run", None) is None
    assert elapsed < BUDGET
    assert len(guest.actions) == 1
    assert [hold["key"] for hold in holds] == ["a"]


@pytest.mark.parametrize(
    ("held", "action", "keydowns"),
    [
        pytest.param(None, Action.hold_key("b", 1.0), ["a"], id="a-hold"),
        pytest.param("b", Action(ActionType.KEY, {"key": "b"}), ["b", "a"], id="a-held-key-s-tap"),
    ],
)
def test_a_wall_key_command_waits_for_a_key_release_in_flight_only_while_its_budget_lasts(
    budgets, monkeypatch, held, action, keydowns
):
    """A hold's keydown, or the re-press after a tap of a held key, waits for the release timer's
    keyup in flight (sandbox/actions.py _wall_commands): no longer than the action's budget, so
    the freeze waits no longer either (user decision 18), and nothing is pressed after it."""
    budgets(action_s=BUDGET)
    # A hold's budget is its duration and 5 s (sandbox/guest.py action_timeout): shrunk too.
    monkeypatch.setattr(actions, "action_timeout", lambda action, timeout: timeout)
    with FakeGuest() as guest:
        env = _wall_environment(guest)
        if held is not None:
            assert env.execute(Action.hold_key(held, MAX_ACTION_DURATION)) == "applied"
        assert env.execute(Action.hold_key("a", 0.2)) == "applied"
        guest.command_delay = 2 * SLACK
        _until_the_release_is_in_flight(guest)
        started = time.monotonic()
        status = env.execute(action)
        elapsed = time.monotonic() - started
        guest.command_delay = 0.0
        env.release_inputs()
    assert status == "failed"
    _invalid(env, "action_execution_failed", f"{action.action_type.value} action")
    _spent_the_budget(elapsed)
    pressed = [command["command"] for command in guest.commands if "keydown" in command["command"]]
    assert pressed == [f"xdotool keydown -- {key}" for key in keydowns]


def test_a_wall_shortcut_is_refused_at_once_while_a_key_release_is_in_flight(budgets):
    """The check waits for no guest request, so the runtime's refusal never comes after the
    action's budget (coordinator's decision N19 no longer arises): the execution result has the
    refusal's words only."""
    budgets(action_s=BUDGET)
    with FakeGuest() as guest:
        env = _wall_environment(guest)
        assert env.execute(Action.hold_key("a", 0.2)) == "applied"
        guest.command_delay = 2 * SLACK
        _until_the_release_is_in_flight(guest)
        started = time.monotonic()
        status = env.execute(Action(ActionType.KEY, {"key": "ctrl+l"}))
        elapsed = time.monotonic() - started
        guest.command_delay = 0.0
        env.release_inputs()
    assert status == "failed"
    assert elapsed < BUDGET
    assert env._last_rejected_by_guest is True
    assert getattr(env, "_invalid_run", None) is None
    assert env._last_exec_error.startswith("browser shortcut 'ctrl+l' is disabled")
    assert env._last_execution_result["error"] == env._last_exec_error


def test_a_failed_wall_tap_counts_its_keys_as_held_only_while_its_budget_lasts(budgets):
    # The guest's request for the tap times out with the action's budget while the release
    # timer holds the wall's key commands for its keyup; the keys the guest may have left down go
    # unrecorded rather than keep the action, and the freeze, waiting (the run is invalid).
    budgets(action_s=1.0)
    with FakeGuest() as guest:
        env = _wall_environment(guest)
        assert env.execute(Action.hold_key("a", 0.4)) == "applied"
        guest.command_delay = 3.0
        guest.action_delay = 5.0
        started = time.monotonic()
        status = env.execute(Action(ActionType.KEY, {"key": "b"}))
        elapsed = time.monotonic() - started
        recorded = "b" in env._wall_holds
        env.release_inputs()
    assert status == "failed"
    _invalid(env, "action_execution_failed", "key action", 1.0)
    assert 1.0 - 0.05 <= elapsed < 1.0 + SLACK, elapsed
    assert not recorded


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
    _spent_the_budget(elapsed, READ_FAILURE_LIMIT)


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
        guest, task, clock_mode="token", clocks=Clocks(FrameClock(1.0), clock), **options
    )
    return env, task


def _run_on(clock: FakeClock, env) -> ScriptedAgent:
    """Three decisions of run_stream on ``clock``, each clicking."""
    agent = ScriptedAgent(CLICK)
    runner.run_stream(
        agent, env, duration_s=3.0, log=lambda line: None, clock_mode="token", clocks=env.clocks
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
    the read reports past its budget. The state is read once a frame and once after every action
    (X18): before the decision at tick 0, after its click, still at tick 0, and at tick 1, where
    the third failed read in a row invalidates the run (kernel/turns.py READ_FAILURE_LIMIT). That
    unrelated CDP events do not restart the call's timeout is test_cdp_deadline.py's."""
    clock = FakeClock()
    read_at = []

    def episode_over():
        read_at.append(env.tick_index())
        _swallowing_the_timeout(clock)
        raise RuntimeError("game state backend returned no state")

    env, _ = _timed(clock, _ClockGuest(), stop_check=episode_over)
    with clock.participant():
        agent = _run_on(clock, env)
        assert clock.monotonic() == READ_FAILURE_LIMIT * 120.0
    _invalid(env, "terminal_state_check_failed", "state read", 120.0)
    assert env._invalid_run["detail"].rstrip().endswith("returned no state)")
    assert agent.statuses == ["applied"]
    assert read_at == [0, 0, 1]


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


@pytest.mark.parametrize("seconds", [15.0, 121.0], ids=["within", "past"])
def test_a_frame_render_has_a_clock_advance_s_budget(seconds):
    """The kernel's shell asks for the action's frame without advancing the token clock
    (sandbox/actions.py render_current_frame); its CDP calls share the 120 s of a clock advance,
    and running past them invalidates the run."""
    clock = FakeClock()
    task = ClockTask(1.0)
    task.render_current_frame = lambda: _sandbox_call(clock, seconds) or {"rendered": True}
    env, _ = _timed(clock, task=task)
    budget = actions.ENVIRONMENT_BUDGET.clock_advance_s
    with clock.participant():
        if seconds < budget:
            assert env.render_current_frame() == {"rendered": True}
        else:
            with pytest.raises(TimeoutError):
                env.render_current_frame()
        assert clock.monotonic() == min(seconds, budget)
    if seconds < budget:
        assert getattr(env, "_invalid_run", None) is None
    else:
        _invalid(env, "frame_render_failed", "frame render", budget)


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


def test_the_guest_s_refusal_reaches_the_environment_as_a_rejection():
    """From the guest's 400 on /action to the environment: GuestClient raises it as a
    GuestRefused with the guest's message (sandbox/guest.py execute_action), and the run stays
    valid."""
    refusal = "unsupported key for pyautogui: 'hyper' (normalized: 'hyper')"
    hyper = Action(ActionType.KEY, {"key": "hyper"})
    with FakeGuest(refusal=refusal) as guest:
        client = GuestClient(host="127.0.0.1", port=guest.port)
        with pytest.raises(GuestRefused, match=re.escape(refusal)):
            client.execute_action(hyper)
        env = _environment(guest)
        assert env.execute(hyper) == "failed"
    assert env._last_rejected_by_guest is True
    assert getattr(env, "_invalid_run", None) is None
    assert env._last_exec_error == f"400 guest rejected the action: {refusal}"


def test_a_guest_refusal_that_arrives_after_the_action_s_budget_is_still_a_refusal():
    """The guest answered, so the environment works and was only slow (user decision 35). The
    double raises the refusal the way GuestClient raises it, which the test above checks."""
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
    """The kernel steps the clock to the end of the lease and releases its key (sandbox/actions.py
    step_clock, release_keys)."""
    clock = FakeClock()
    env, task = _timed(clock)
    env.execute(Action.hold_key("a", 0.5))
    env.step_clock(0.5)
    _slow_key_event(clock, task, 0, 60.0)
    budget = actions.ENVIRONMENT_BUDGET.key_event_s
    with clock.participant(), pytest.raises(BudgetExceeded):
        env.release_keys(env.last_held_keys)
    assert clock.monotonic() == budget
    _invalid(env, "clock_key_release_failed", "key release", budget)


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
    env.step_clock(0.5)
    _slow_key_event(clock, task, 0, 60.0, delivered=delivered)
    with clock.participant():
        with pytest.raises(BudgetExceeded):
            env.release_keys(env.last_held_keys)
        assert list(env._held_keys) == ["a"]
        env.release_inputs()
    assert [event["key"] for event in task.events if event["event"] == "keyUp"] == keyups
    assert env._held_keys == {}


@pytest.mark.parametrize("delivered", [True, False], ids=["delivered-late", "lost"])
def test_the_cleanup_releases_the_keys_of_a_hold_whose_press_failed(delivered):
    """Coordinator's decision N20: the press may have reached the page, so its keys count as
    maybe down. When it did not, the page still gets the cleanup's keyUp as an event, but only
    after the scored snapshot is frozen."""
    clock = FakeClock()
    env, task = _timed(clock)
    _slow_key_event(clock, task, 0, 60.0, delivered=delivered)
    with clock.participant():
        assert env.execute(Action.hold_key("shift+a", 1.0)) == "failed"
        env.release_inputs()
    pressed = [("keyDown", "shift+a")] if delivered else []
    assert [(event["event"], event["key"]) for event in task.events] == [
        *pressed,
        ("keyUp", "shift+a"),
    ]
    assert env._held_keys == {}


class _DesktopCommands:
    """The guest's command endpoint, which records each command; an xdotool keydown reaches the
    desktop, but its answer spends the action's budget (sandbox/guest.py execute_command)."""

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.commands: list[str] = []

    def execute_command(self, command: str, timeout: int = 30) -> dict:
        self.commands.append(command)
        if "keydown" in command:
            _sandbox_call(self.clock, 3600.0)
        return {"returncode": 0}


def _wall_on(clock: FakeClock, guest) -> actions.GuestEnvironment:
    """A GuestEnvironment on the wall clock, timed on ``clock``, whose guest is ``guest``."""
    return actions.GuestEnvironment(
        guest,
        _WallTask(),
        honor_fail=False,
        clock_mode="wall",
        wall_start_time=1_000.0,
        clocks=Clocks(clock, clock),
    )


def test_the_cleanup_releases_the_keys_of_a_wall_hold_whose_keydown_request_failed():
    clock = FakeClock(wall_origin=1_000.0)
    guest = _DesktopCommands(clock)
    env = _wall_on(clock, guest)
    with clock.participant():
        assert env.execute(Action.hold_key("a", 1.0)) == "failed"
        env.release_inputs()
    assert [command for command in guest.commands if command.startswith("xdotool")] == [
        "xdotool keydown -- a",
        "xdotool keyup -- a",
    ]


def _a_frame_that_fails(frames=1, key_events=()):
    raise RuntimeError("browser frame advanced without a compositor commit")


@pytest.mark.parametrize(
    ("action", "failing"),
    [
        pytest.param(TAP, "press", id="a-tap-s-press"),
        pytest.param(TAP, "frame", id="a-tap-s-frame"),
        pytest.param(TAP, "release", id="a-tap-s-release"),
        pytest.param(Action.hold_key("space", 0.0), "release", id="a-0-s-hold-s-release"),
    ],
)
def test_the_cleanup_releases_the_keys_of_a_tap_or_a_0_s_hold_that_failed(action, failing):
    """Coordinator's decision N23, as N20: the key may be down on the page."""
    clock = FakeClock()
    env, task = _timed(clock)
    if failing == "frame":
        task.advance_clock = _a_frame_that_fails
    else:
        _slow_key_event(clock, task, 0 if failing == "press" else 1, 60.0)
    with clock.participant():
        assert env.execute(action) == "failed"
        env.release_inputs()
    assert [event["key"] for event in task.events if event["event"] == "keyUp"] == ["space"]
    assert env._held_keys == {}


def test_the_cleanup_releases_the_key_of_a_repeated_tap_whose_second_press_failed():
    """Each repeat of the tap counts its keys as held when it fails, not the first alone."""
    clock = FakeClock()
    env, task = _timed(clock)
    _slow_key_event(clock, task, 2, 60.0, delivered=True)
    with clock.participant():
        assert env.execute(Action(ActionType.KEY, {"key": "space", "repeat": 3})) == "failed"
        env.release_inputs()
    keys = [(event["event"], event["key"]) for event in task.events if event["event"] != "advance"]
    assert keys == [("keyDown", "space"), ("keyUp", "space")] * 2
    assert env._held_keys == {}


def test_a_failed_tap_of_a_held_key_leaves_its_lease_as_it_was():
    clock = FakeClock()
    env, task = _timed(clock)
    env.execute(Action.hold_key("space", 5.0))
    _slow_key_event(clock, task, 1, 60.0)
    with clock.participant():
        assert env.execute(TAP) == "failed"
    assert env._held_keys["space"]["expires_at_tick"] == 5.0


class _FailingGuest(_DesktopCommands):
    """The guest, whose action endpoint fails with ``error``; it runs every command at once."""

    def __init__(self, clock: FakeClock, error: Exception):
        super().__init__(clock)
        self.error = error

    def execute_command(self, command: str, timeout: int = 30) -> dict:
        self.commands.append(command)
        return {"returncode": 0}

    def execute_action(self, action: Action) -> dict:
        raise self.error


@pytest.mark.parametrize(
    ("action", "error", "keyups"),
    [
        pytest.param(
            Action(ActionType.KEY, {"key": "a"}),
            requests.ReadTimeout("read timed out"),
            ["xdotool keyup -- a"],
            id="a-tap",
        ),
        pytest.param(
            Action.hold_key("a", 0.0),
            requests.ReadTimeout("read timed out"),
            ["xdotool keyup -- a"],
            id="a-0-s-hold",
        ),
        pytest.param(
            Action(ActionType.KEY, {"key": "a"}),
            GuestRefused("400 guest rejected the action"),
            [],
            id="a-refused-tap",
        ),
    ],
)
def test_the_cleanup_releases_the_keys_of_a_wall_tap_whose_guest_request_failed(
    action, error, keyups
):
    """On the wall clock the guest presses and releases a tap or a 0 s hold itself; a refusal
    pressed nothing."""
    clock = FakeClock(wall_origin=1_000.0)
    guest = _FailingGuest(clock, error)
    env = _wall_on(clock, guest)
    with clock.participant():
        assert env.execute(action) == "failed"
        env.release_inputs()
    assert [command for command in guest.commands if "keyup" in command] == keyups


def test_a_failed_wall_hold_keeps_a_held_key_s_lease_and_skips_keys_xdotool_cannot_name():
    """'é' has no X keysym, so the hold goes to the guest, which fails; the cleanup can release
    only the keys xdotool names, and 'a' keeps the lease of the hold that pressed it."""
    clock = FakeClock(wall_origin=1_000.0)
    guest = _FailingGuest(clock, requests.ReadTimeout("read timed out"))
    env = _wall_on(clock, guest)
    with clock.participant():
        assert env.execute(Action.hold_key("a", 5.0)) == "applied"
        assert env.execute(Action.hold_key("a+é", 1.0)) == "failed"
        held = {identity: item["expires_at"] for identity, item in env._wall_holds.items()}
        env.release_inputs()
    assert held == {"a": 1_005.0}
    assert [command for command in guest.commands if "keyup" in command] == ["xdotool keyup -- a"]


@pytest.mark.parametrize("slow", [True, False], ids=["slow", "failing"])
def test_a_key_release_at_cleanup_that_fails_leaves_the_run_as_it_was(slow):
    """release_inputs runs once run_stream has latched the outcome, and the runner only logs its
    failure (legacy_loop.py _finalize); main() would still take an invalid run latched here
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


def test_the_cleanup_releases_every_held_mouse_button_when_one_fails():
    """The first release fails, whichever button it is; the others are still released, and the
    failure is raised for the caller to log."""

    class FirstReleaseFails(_ClockGuest):
        def execute_action(self, action: Action) -> dict:
            first_release = (
                action.action_type == ActionType.MOUSE_UP and ActionType.MOUSE_UP not in self.actions
            )
            super().execute_action(action)
            if first_release:
                raise RuntimeError("the guest is gone")
            return {"status": "ok"}

    guest = FirstReleaseFails()
    env, _ = _timed(FakeClock(), guest)
    for button in ("left", "right", "middle"):
        assert env.execute(Action(ActionType.MOUSE_DOWN, {"button": button})) == "applied"
    with pytest.raises(ExceptionGroup) as raised:
        env.release_inputs()
    assert [str(error) for error in raised.value.exceptions] == ["the guest is gone"]
    assert guest.actions == [ActionType.MOUSE_DOWN] * 3 + [ActionType.MOUSE_UP] * 3
    assert len(env._held_mouse_buttons) == 1
