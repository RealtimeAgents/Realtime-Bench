"""The environment and model-input frame rates may differ; a model's video cap bounds only the
model-input rate, ModelConfig.fps (providers/preflight.py:132-138).

runner.main() takes the two as --fps and --model-input-fps. The effective-config export's runner
probe (evaluation/effective_config.py) runs main() in a fresh interpreter up to its decision loop
and records where each rate went.
"""

from __future__ import annotations

import json
import os

import pytest

import realtime_framework.providers.registry as catalog
import realtime_tasks.registry as tasks
from realtime_framework.evaluation.effective_config import _runner_entry, export
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


PLAN = """\
suite: public
models:
  - gemini-3.8-live
runtime:
  seed: 42
  mode: default
  clock_mode: wall
  fps: 2
  model_input_fps: 1
  duration_minutes: 5
  protocol: wall-clock
"""


def test_a_run_plan_carries_both_rates_to_the_runner(tmp_path):
    """``rtbench run -c PLAN``: the plan's runtime becomes the Harbor agent's options
    (evaluation/harbor_cli.py:209-235), the trial's identity and command
    (evaluation/harbor.py:445-520), the trial's plan and model config (evaluation/_trial.py), and
    the runner's arguments, which reach its decision loop."""
    plan = tmp_path / "split.yaml"
    plan.write_text(PLAN)
    entry = f"harbor:{plan}"
    environ = {"HOME": os.environ.get("HOME", "/"), "PATH": "/usr/bin:/bin"}
    exported = export(
        environ=environ, env_file="/nonexistent/.env", entries=[entry], models=[GEMINI],
        tasks=[WEB],
    )
    cell = exported["entries"][entry]["models"][GEMINI]
    assert cell["error"] is None
    harbor = cell["harbor"]
    assert (harbor["harbor_request"]["fps"], harbor["harbor_request"]["model_input_fps"]) == (2, 1)
    assert (harbor["task_json"]["fps"], harbor["task_json"]["model_input_fps"]) == (2, 1)
    argv = harbor["trial_argv"]
    assert argv[argv.index("--fps") + 1] == "2.0"
    assert argv[argv.index("--model-input-fps") + 1] == "1.0"
    runtime = cell["trial"]["settings"]["plan"]["runtime"]
    assert (runtime["fps"], runtime["model_input_fps"]) == (2, 1)
    assert cell["trial"]["settings"]["model_config"]["fps"] == 1.0
    assert cell["frame_rates"] == {"env_fps": 2.0, "model_input_fps": 1.0}
    assert cell["run_stream"]["frame_ticks"] == 2.0
    assert cell["expected_runtime_config_hash"] == cell["runtime_config_hash"]


def _trial(model_input_fps) -> tuple[dict, dict, dict]:
    """A trial's identity (evaluation/harbor.py:506-520), its request, and the one-task local
    report whose group carries the identity reporting.report found in the run summary."""
    task = {
        "suite_id": "public", "suite_version": "1.0", "suite_hash": "0" * 64,
        "harness_version": "1.0", "harness_hash": "1" * 64, "task_id": WEB,
        "category": "web", "fps": 2.0, "duration_minutes": 5.0, "seed": 42,
    }
    if model_input_fps is not None:
        task["model_input_fps"] = model_input_fps
    request = {
        "model": GEMINI, "mode": "default", "model_revision": None, "clock": "wall",
        "task": task, "suite_tasks": {WEB: "web"},
    }
    group = {
        **{field: task[field] for field in ("suite_id", "suite_version", "suite_hash")},
        "harness_hash": task["harness_hash"], "fps": task["fps"],
        "model_input_fps": model_input_fps, "seed": 42, "public_category": "web",
        "mode": "default", "protocol": "wall-clock", "clock_mode": "wall",
        "tokens_per_frame": None, "cap": 5.0, "expected": 1, "scored": 1, "invalid": 0,
        "missing": 0, "partial": False, "runtime_config_hash": "2" * 64,
        "tasks": [{"task_id": WEB, "status": "scored", "solved": True, "outcome": "pass"}],
    }
    report = {
        "integration": {
            "model": GEMINI, "model_revision": None, "mode": "default",
            "configuration_sha256": "3" * 64,
        },
        "groups": [group],
    }
    return task, request, report


def test_harbor_verify_compares_the_model_input_rate():
    from realtime_framework.evaluation.harbor import verify

    task, request, report = _trial(1.0)
    assert verify(report, task, request) == {"reward": 1.0}
    report["groups"][0]["model_input_fps"] = 2.0
    with pytest.raises(ValueError, match="Report model_input_fps mismatch"):
        verify(report, task, request)


def test_jobs_from_before_the_split_report_the_environment_rate(tmp_path):
    """A job recorded with one --fps has no model_input_fps; its model ran at that rate."""
    from realtime_framework.evaluation.harbor import aggregate

    _, request, _ = _trial(None)
    trial = tmp_path / f"{WEB}__trial" / "agent"
    trial.mkdir(parents=True)
    (trial / "request.json").write_text(json.dumps(request))
    report = aggregate(tmp_path)
    assert (report["fps"], report["model_input_fps"]) == (2.0, 2.0)
