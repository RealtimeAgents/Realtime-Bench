"""Each environment operation fails once it runs past its overall budget, and the run is invalid.

The real run_stream drives the real GuestEnvironment (sandbox/actions.py). Its GuestClient talks to
FakeGuest; a real catalog game (realtime_tasks/game/runtime.py) reads its state and advances the
token clock over FakeCDP. Each case slows one operation down: it must fail within its budget and
latch the invalid-run reason the runner already uses for that operation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

import realtime_tasks.registry as tasks
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.evaluation import runner
from realtime_framework.sandbox import actions
from realtime_framework.sandbox.browser.cdp import client_for, close_port
from realtime_framework.sandbox.browser.clock import cdp_wait_realtime_action_barrier
from realtime_framework.sandbox.guest import GuestClient, action_timeout
from realtime_tasks.game.runtime import _GET_STATE
from realtime_tasks.types import Task

from .fakes import FakeCDP, FakeGuest

BUDGET = 0.5
# A failing operation must end by its budget plus this much scheduling slack, still well before
# the timeouts that bounded it before (3 s and more).
SLACK = 1.5
GAME = "gh_javascript_pong"
PLAYING = {"game_state": {"player_returns": 0}, "terminal": {"isTerminal": False}}
CLICK = Action(ActionType.CLICK, {"x": 10, "y": 10})


class ScriptedAgent:
    """The adapter calls run_stream makes (runner.py:908-919, 435), one fixed action each decision."""

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


class BarrierTask(Task):
    """A task whose action barrier waits on the page, as realtime_tasks/web/runtime.py:259-272."""

    def __init__(self, port: int):
        self.port = port
        _token_clock(self)

    def begin_gui_action(self) -> dict:
        return {"input_event_sequence": 0, "business_action_started": 0, "business_action_completed": 0}

    def complete_gui_action(self, snapshot: dict) -> dict:
        return cdp_wait_realtime_action_barrier(self.port, "/game", snapshot)


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


def _run(env) -> tuple[ScriptedAgent, float]:
    """Three decisions of run_stream, each clicking; returns the agent and the wall time taken."""
    agent = ScriptedAgent(CLICK)
    started = time.monotonic()
    runner.run_stream(agent, env, max_ticks=3, log=lambda line: None, clock_mode="token")
    return agent, time.monotonic() - started


def _spent_the_budget(elapsed: float) -> None:
    assert BUDGET - 0.05 <= elapsed < BUDGET + SLACK, elapsed


def _invalid(env, reason: str) -> None:
    assert env._invalid_run["category"] == "environment"
    assert env._invalid_run["reason"] == reason


def test_an_action_budget_extends_for_text_input_and_holds_as_the_guest_request_does():
    budget = actions.ENVIRONMENT_BUDGET.action_s
    assert budget == 10.0
    assert action_timeout(CLICK, budget) == 10.0
    assert action_timeout(Action(ActionType.TYPE, {"text": "Ada"}), budget) == 120.0
    assert action_timeout(Action(ActionType.HOLD_KEY, {"key": "a"}), budget) == 10.0
    assert action_timeout(Action(ActionType.HOLD_KEY, {"key": "a", "duration": 8}), budget) == 13.0
    drag = {"x1": 0, "y1": 0, "x2": 5, "y2": 5, "duration": 10}
    assert action_timeout(Action(ActionType.DRAG, drag), budget) == 15.0


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
    _invalid(env, "action_execution_failed")
    assert agent.statuses == ["invalid_run"]
    _spent_the_budget(elapsed)


def test_an_action_and_its_page_barrier_share_one_budget(budgets):
    # The guest answers within the budget and the page barrier never does. Without the shared
    # budget the barrier's own 8 s CDP timeout would apply.
    budgets(action_s=BUDGET)
    with FakeGuest(action_delay=0.3) as guest, FakeCDP(mode="flood") as fake:
        env = _environment(guest, BarrierTask(fake.port))
        try:
            agent, elapsed = _run(env)
        finally:
            close_port(fake.port)
    _invalid(env, "action_execution_failed")
    assert len(guest.actions) == 1
    _spent_the_budget(elapsed)


def test_a_slow_capture_fails_within_its_budget(budgets):
    budgets(capture_s=BUDGET)
    with FakeGuest(screenshot_delay=5) as guest:
        env = _environment(guest)
        agent, elapsed = _run(env)
    _invalid(env, "observation_capture_failed")
    assert agent.statuses == []
    _spent_the_budget(elapsed)


def test_a_state_read_flooded_with_events_fails_within_its_budget(budgets, game):
    budgets(state_read_s=BUDGET)
    with FakeGuest() as guest, FakeCDP(mode="flood") as fake:
        env = _environment(guest, stop_check=game(fake).episode_over)
        agent, elapsed = _run(env)
    _invalid(env, "terminal_state_check_failed")
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
    _invalid(env, "terminal_state_check_failed")
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
    _invalid(env, "clock_advance_failed")
    assert env._stop_reason == "invalid_run"
    assert len(guest.actions) == 1
    assert fake.commands == []
    _spent_the_budget(elapsed)


def test_the_snapshot_read_at_the_freeze_fails_within_its_short_budget(budgets, game):
    # The game reads its snapshot through cdp_eval, which turns the timeout into an empty state;
    # the read still fails because it returned after its deadline.
    budgets(snapshot_read_s=BUDGET)
    with FakeGuest() as guest, FakeCDP(mode="flood") as fake:
        env = _environment(guest, game(fake), stop_check=lambda: True)
        agent, elapsed = _run(env)
    _invalid(env, "evaluation_snapshot_failed")
    assert env._termination_snapshot_captured is False
    assert "snapshot read took longer than 0.5 s" in env._termination_snapshot_error
    assert agent.statuses == []
    _spent_the_budget(elapsed)
