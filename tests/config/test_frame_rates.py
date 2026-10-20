"""The runner starts only when the environment and model-input frame rates are equal."""

from __future__ import annotations

import pytest

import realtime_framework.providers.registry as catalog
import realtime_tasks.registry as tasks
from realtime_framework.providers.preflight import check_episode, validate_frame_rates

MESSAGE = "environment fps 2 and model-input fps 1 must be equal"


def test_equal_rates_pass():
    validate_frame_rates(1.0, 1.0)


def test_different_rates_fail():
    with pytest.raises(ValueError, match="must be equal"):
        validate_frame_rates(2.0, 1.0)


def test_episode_rule_follows_the_wall_clock_check():
    config = catalog.to_model_config("my-model", fps=1.0, reasoning_effort="provider_default")
    task = tasks.get("rt_waitlist_fit")
    with pytest.raises(ValueError) as caught:
        check_episode(config, clock_mode="wall", task=task, env_fps=2.0)
    assert str(caught.value) == MESSAGE
    with pytest.raises(ValueError) as caught:
        check_episode(config, clock_mode="token", task=task, env_fps=2.0)
    assert str(caught.value) == "The chat_completions API requires a wall clock"
