"""The environment and model-input frame rates may differ; a model's video cap bounds only the
model-input rate, ModelConfig.fps (providers/preflight.py:132-138).

runner.main() takes the two as --fps and --model-input-fps. The effective-config export's runner
probe (evaluation/effective_config.py) runs main() in a fresh interpreter up to its decision loop
and records where each rate went.
"""

from __future__ import annotations

import os

import pytest

import realtime_framework.providers.registry as catalog
import realtime_tasks.registry as tasks
from realtime_framework.evaluation.effective_config import _runner_entry
from realtime_framework.providers.preflight import check_episode

GEMINI = "gemini-3.8-live"
WEB, GAME, CAPPED = "rt_waitlist_fit", "gh_timed_chess", "rt_accessible_trip_dispatch"
CAP_MESSAGE = "gemini_live supports at most 1 fps; requested 2"


def gemini_config(model_input_fps: float):
    effort = catalog.get(GEMINI).defaults["reasoning_effort"]
    return catalog.to_model_config(GEMINI, fps=model_input_fps, reasoning_effort=effort)


def test_video_cap_reads_the_model_input_rate():
    check_episode(gemini_config(1.0), clock_mode="wall", task=tasks.get(WEB))
    with pytest.raises(ValueError) as caught:
        check_episode(gemini_config(2.0), clock_mode="wall", task=tasks.get(WEB))
    assert str(caught.value) == CAP_MESSAGE


@pytest.fixture(scope="module")
def runner_cells(tmp_path_factory):
    """Gemini Live through runner.main(): on the wall clock at 2/1, on the token clock at 60/1
    with B=100, and at 2/2, which its cap refuses. Each cell is one task of the probe's run; the
    arguments are those _trial passes (evaluation/_trial.py:288-314)."""
    wall = ["--clock", "wall", "--protocol", "wall-clock"]
    token = ["--clock", "token", "--tokens-per-frame", "100", "--protocol", "time-aware-token-v1"]
    runs = [
        (WEB, [*wall, "--fps", "2", "--model-input-fps", "1"]),
        (GAME, [*token, "--fps", "60", "--model-input-fps", "1"]),
        (CAPPED, [*wall, "--fps", "2"]),
    ]
    effort = catalog.get(GEMINI).defaults["reasoning_effort"]
    environ = {"HOME": os.environ.get("HOME", "/"), "PATH": "/usr/bin:/bin"}
    exported = _runner_entry(
        environ,
        {},
        [
            (
                GEMINI,
                task,
                ["--model", GEMINI, "--task", task, *argv, "--reasoning-effort", effort]
                + ["--report-root", "<results>"],
            )
            for task, argv in runs
        ],
        env_file=None,
        scratch=tmp_path_factory.mktemp("runner"),
    )
    return exported["models"][GEMINI]


def test_environment_rate_above_the_cap_starts(runner_cells):
    cell = runner_cells[WEB]
    assert cell["error"] is None
    assert (cell["run"]["fps"], cell["run"]["model_input_fps"]) == (2.0, 1.0)
    # The environment: the task clock, GuestEnvironment's tick and the loop's ticks.
    assert cell["task"]["task_clock"]["fps"] == 2.0
    assert cell["guest_environment"]["tick_seconds"] == 0.5
    loop = cell["run_stream"]
    assert (loop["delta_s"], loop["frame_ticks"], loop["max_ticks"]) == (0.5, 2.0, 600.0)
    # The model: its config, and the rate its prompt names.
    assert cell["frame_rates"] == {"env_fps": 2.0, "model_input_fps": 1.0}
    assert cell["runtime_config"]["fps"] == 2.0
    assert cell["clock_prompt"].startswith("WALL CLOCK: observations are scheduled at 1 fps. ")
    # The identity the report expects is computed from the same two rates.
    assert cell["expected_runtime_config_hash"] == cell["runtime_config_hash"]


def test_token_clock_charges_b_per_model_input_frame(runner_cells):
    cell = runner_cells[GAME]
    assert cell["error"] is None
    loop = cell["run_stream"]
    assert (loop["frame_ticks"], loop["tokens_per_frame"]) == (60.0, 100)
    assert cell["frame_rates"] == {"env_fps": 60.0, "model_input_fps": 1.0}
    prompt = cell["clock_prompt"]
    assert "observations are scheduled at f=1 frames per simulated second" in prompt
    assert "kappa=B*f=100 charged tokens per simulated second" in prompt
    assert cell["expected_runtime_config_hash"] == cell["runtime_config_hash"]


def test_model_input_rate_above_the_cap_is_refused(runner_cells):
    """--model-input-fps defaults to --fps."""
    cell = runner_cells[CAPPED]
    assert cell["error"] == f"ValueError: {CAP_MESSAGE}"
