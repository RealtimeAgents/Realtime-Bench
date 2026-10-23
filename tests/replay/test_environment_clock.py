"""GuestEnvironment's token clock at environment rates other than 1 fps."""

from __future__ import annotations

import pytest

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.sandbox.actions import GuestEnvironment

from .clock_task import ClockTask


def token_environment(fps: float) -> tuple[GuestEnvironment, ClockTask]:
    task = ClockTask(fps)
    return GuestEnvironment(None, task, clock_mode="token"), task


@pytest.mark.parametrize("fps", [1.0, 2.0, 30.0, 60.0])
def test_tap_holds_the_key_for_one_sixtieth_of_a_second(fps):
    """A tapped key stays down for 1/60 s of world time, one 60 Hz rAF
    (sandbox/actions.py TAP_HOLD_S), whatever the environment rate. At 2 fps 1/60 tick would be
    1/120 s, which can hold no rendered frame (sandbox/browser/clock.py:428)."""
    env, task = token_environment(fps)
    assert env.execute(Action(ActionType.KEY, {"key": "space"})) == "applied"
    down, advance, up = task.events
    assert (down["event"], down["key"], down["at_s"]) == ("keyDown", "space", 0.0)
    assert advance["event"] == "advance"
    assert advance["frames"] == pytest.approx(fps / 60)
    assert (up["event"], up["key"]) == ("keyUp", "space")
    assert up["at_s"] == round(1 / 60, 9)
