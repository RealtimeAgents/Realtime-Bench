"""The environment and model-input frame rates may differ; a model's video cap bounds only the
model-input rate, ModelConfig.fps (providers/preflight.py:132-138)."""

from __future__ import annotations

import pytest

import realtime_framework.providers.registry as catalog
import realtime_tasks.registry as tasks
from realtime_framework.providers.preflight import check_episode

GEMINI = "gemini-3.8-live"
TASK = "rt_waitlist_fit"


def gemini_config(model_input_fps: float):
    effort = catalog.get(GEMINI).defaults["reasoning_effort"]
    return catalog.to_model_config(GEMINI, fps=model_input_fps, reasoning_effort=effort)


def test_video_cap_reads_the_model_input_rate():
    check_episode(gemini_config(1.0), clock_mode="wall", task=tasks.get(TASK))
    with pytest.raises(ValueError) as caught:
        check_episode(gemini_config(2.0), clock_mode="wall", task=tasks.get(TASK))
    assert str(caught.value) == "gemini_live supports at most 1 fps; requested 2"
