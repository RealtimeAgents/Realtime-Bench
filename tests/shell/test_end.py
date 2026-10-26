"""The episode the task itself ends (evaluation/shell.py CheckEnd and EndChecked), on both
clocks: at the loop top, right before an action lands and right after it is executed. The stop
reason is the task's, the end time is when the check that saw it was asked (kernel/turns.py
_end_checked), an action the end comes before is not applied and one it comes right after is
(legacy_loop.py _land_action)."""

from __future__ import annotations

import pytest

from realtime_framework.entity.actions import Action, ActionType

from .world import CLICK, Reply, World, run

TAP = Action(ActionType.KEY, {"key": "a"})


def _ended(episode, end_s: float) -> None:
    result = episode.result
    assert result.stop_reason == "won"
    assert result.termination_record["stop_reason"] == "won"
    assert result.termination_record["end_s"] == end_s
    assert result.invalid_run is None


@pytest.mark.parametrize("clock_mode", ["wall", "token"])
def test_an_end_at_the_loop_top_ends_the_episode_before_any_decision(clock_mode):
    episode = run(clock_mode, 3.0, [Reply(0.5)], World(ends_at=0.0, end_reason="won"))
    _ended(episode, 0.0)
    assert episode.statuses() == []
    assert [call for call, _, _ in episode.env.calls] == ["check_end", "snapshot", "release_inputs"]


def test_a_wall_end_before_landing_leaves_the_action_unapplied():
    """The task ends at 0.25 s, during the generation; the check right before landing, asked at
    0.5 s, sees it."""
    episode = run("wall", 3.0, [Reply(0.5)], World(ends_at=0.25, end_reason="won"))
    _ended(episode, 0.5)
    assert episode.statuses() == ["not_applied_terminal"]
    assert [call for call, _, _ in episode.env.calls if call == "execute"] == []


def test_a_token_end_before_landing_leaves_the_action_unapplied():
    """The task ends at tick 1; the check after the step that moves the clock to the action time
    sees it."""
    world = World(ends_at=1.0, end_reason="won")
    episode = run("token", 3.0, [Reply(0.0, CLICK, 100)], world, tokens_per_frame=100)
    _ended(episode, 1.0)
    assert episode.statuses() == ["not_applied_terminal"]
    assert [call for call, _, _ in episode.env.calls if call == "execute"] == []


def test_a_wall_end_right_after_execution_is_the_actions_doing():
    """Executing takes 0.125 s from 0.5 s and the task ends at 0.5625 s: the check right after
    execution, asked at 0.625 s, sees it, and the action that did it is applied."""
    world = World(ends_at=0.5625, end_reason="won", execute_s=0.125)
    episode = run("wall", 3.0, [Reply(0.5)], world)
    _ended(episode, 0.625)
    assert episode.statuses() == ["applied"]
    assert episode.agent.outcomes[0].terminal_after_action is True


@pytest.mark.parametrize("action", [CLICK, TAP], ids=["click", "tap"])
def test_a_token_end_right_after_execution_is_the_actions_doing(action):
    """Paused at tick 0, the task ends once one action is applied: the check right after
    execution sees it at the tick the action left the clock at."""
    episode = run("token", 3.0, [Reply(0.0, action)], World(ends_after=1, end_reason="won"))
    _ended(episode, 0.0 if action is CLICK else round(1 / 60, 9))
    assert episode.statuses() == ["applied"]
    assert episode.agent.outcomes[0].terminal_after_action is True
