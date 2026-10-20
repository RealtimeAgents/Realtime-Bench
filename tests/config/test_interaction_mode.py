"""The interaction mode, turn_based or streaming, is a run setting like the frame rates.

runner.main() takes it as --interaction-mode, records it in the run identity (the episode start,
run_config.json and its hash, the expected summary and the run summary) and passes it to the
pre-run checks (providers/preflight.py check_interaction_mode). Run plans, the Harbor agent's
options, each trial's task.json and the trial worker carry it to the runner. No driver declares
streaming yet, so every streaming run is refused before the agent is built.
"""

from __future__ import annotations

import json
import os
import types
from dataclasses import replace

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.evaluation.effective_config import _runner_entry, export

GEMINI = "gemini-3.8-live"
WEB, GAME = "rt_waitlist_fit", "gh_timed_chess"
UNSUPPORTED = "The gemini_live API does not support streaming"


@pytest.fixture(scope="module")
def runner_cells(tmp_path_factory):
    """Gemini Live through runner.main() on the wall clock: by default, and asking for streaming.
    Each cell is one task of the probe's run, as in tests/config/test_frame_rates.py."""
    wall = ["--clock", "wall", "--protocol", "wall-clock"]
    runs = [(WEB, wall), (GAME, [*wall, "--interaction-mode", "streaming"])]
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


def test_the_runner_records_turn_based_by_default(runner_cells):
    cell = runner_cells[WEB]
    assert cell["error"] is None
    assert cell["run"]["interaction_mode"] == "turn_based"
    assert cell["runtime_config"]["interaction_mode"] == "turn_based"
    # The hash the report expects (reporting.prepare_episode_report) covers the same mode.
    assert cell["expected_runtime_config_hash"] == cell["runtime_config_hash"]


def test_the_runner_refuses_streaming(runner_cells):
    assert runner_cells[GAME]["error"] == f"ValueError: {UNSUPPORTED}"


def run_main(tmp_path, monkeypatch, *argv) -> dict:
    """runner.main() in this interpreter with the export probe's stand-ins for the container, the
    guest, the scene capture, the agent and the task server (evaluation/effective_config.py), and a
    decision loop that fails at once, so the run ends invalid. Returns the run summary."""
    import sys
    import types

    import realtime_tasks.registry as tasks
    from realtime_framework.evaluation import effective_config as probe
    from realtime_framework.evaluation import reporting, runner
    from realtime_framework.sandbox import desktop

    for name in ("build_image_if_needed", "start_container", "_start_screensaver_keepalive"):
        monkeypatch.setattr(desktop, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(desktop, "_start_screen_recording", lambda run_dir=None: False)
    monkeypatch.setattr(reporting, "finalize_video", lambda run_dir: None)
    clock = types.SimpleNamespace(time=runner.time.time, sleep=lambda seconds: None)
    monkeypatch.setattr(runner, "time", clock)
    monkeypatch.setattr(runner, "GuestClient", probe._Guest)
    monkeypatch.setattr(runner, "BrowserSceneCapture", probe._Scene)
    monkeypatch.setattr(runner, "create_agent", lambda config, **kwargs: probe._Agent(config))
    registered = tasks.get
    monkeypatch.setattr(tasks, "get", lambda task_id: probe._Task(registered(task_id)))

    def run_stream(adapter, env, **loop):
        raise RuntimeError("decision loop failed")

    monkeypatch.setattr(runner, "run_stream", run_stream)
    for name in ("SANDBOX_GRAPHICS_MODE", "SCREEN_RECORDING_ENCODER"):
        monkeypatch.setenv(name, "cpu")  # main() sets both
    argv = ["--model", GEMINI, "--task", WEB, "--clock", "wall", *argv]
    argv += ["--results-dir", str(tmp_path), "--run-id", "run"]
    monkeypatch.setattr(sys, "argv", ["runner", *argv])
    with pytest.raises(SystemExit):  # an invalid run is not a scored result
        runner.main()
    return json.loads((tmp_path / "run" / "run_summary.json").read_text())


def test_the_run_summary_records_the_mode(tmp_path, monkeypatch):
    summary = run_main(tmp_path, monkeypatch)
    assert summary["error"].strip().endswith("RuntimeError: decision loop failed")
    assert summary["interaction_mode"] == "turn_based"
    assert summary["runtime_config"]["interaction_mode"] == "turn_based"


def test_a_refused_streaming_run_records_the_mode(tmp_path, monkeypatch):
    summary = run_main(tmp_path, monkeypatch, "--interaction-mode", "streaming")
    assert summary["error"].strip().endswith(f"ValueError: {UNSUPPORTED}")
    assert summary["interaction_mode"] == "streaming"


def test_the_report_compares_the_mode(tmp_path):
    """reporting.report checks each run summary against the identity expected before the run."""
    from realtime_framework.evaluation.reporting import summary_ok

    path = tmp_path / "run_summary.json"
    path.write_text(json.dumps({"stop_reason": "terminal", "interaction_mode": "turn_based"}))
    assert summary_ok(path, expected_interaction_mode="turn_based")
    assert not summary_ok(path, expected_interaction_mode="streaming")


PLAN = """\
suite: public
models:
  - gemini-3.8-live
runtime:
  seed: 42
  mode: default
  clock_mode: wall
  fps: 1
  interaction_mode: {mode}
  duration_minutes: 5
  protocol: wall-clock
"""


def test_a_run_plan_carries_the_mode_to_the_runner(tmp_path):
    """``rtbench run -c PLAN``, as tests/config/test_frame_rates.py follows the frame rates: the
    plan's runtime becomes the Harbor agent's options, the trial's task.json and command, the
    trial's plan, and the runner's arguments, which reach its decision loop."""
    plan = tmp_path / "turn_based.yaml"
    plan.write_text(PLAN.format(mode="turn_based"))
    entry = f"harbor:{plan}"
    environ = {"HOME": os.environ.get("HOME", "/"), "PATH": "/usr/bin:/bin"}
    exported = export(
        environ=environ, env_file="/nonexistent/.env", entries=[entry], models=[GEMINI],
        tasks=[WEB],
    )
    cell = exported["entries"][entry]["models"][GEMINI]
    assert cell["error"] is None
    harbor = cell["harbor"]
    assert harbor["harbor_request"]["interaction_mode"] == "turn_based"
    assert harbor["task_json"]["interaction_mode"] == "turn_based"
    argv = harbor["trial_argv"]
    assert argv[argv.index("--interaction-mode") + 1] == "turn_based"
    assert cell["trial"]["settings"]["plan"]["runtime"]["interaction_mode"] == "turn_based"
    argv = cell["runner_argv"]
    assert argv[argv.index("--interaction-mode") + 1] == "turn_based"
    assert cell["run"]["interaction_mode"] == "turn_based"
    assert cell["runtime_config"]["interaction_mode"] == "turn_based"
    assert cell["expected_runtime_config_hash"] == cell["runtime_config_hash"]


@pytest.fixture
def streaming_gemini(monkeypatch):
    """Gemini Live declaring native streaming, as it will once its streaming agent exists."""
    from realtime_framework.providers.gemini.live import GeminiLiveDriver

    capabilities = replace(
        GeminiLiveDriver.capabilities, interaction_modes=frozenset({"turn_based", "streaming"})
    )
    monkeypatch.setattr(GeminiLiveDriver, "capabilities", capabilities)


@pytest.fixture
def environment():
    """_trial.main writes os.environ (evaluation/_trial.py:107-108)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_each_step_passes_streaming_on(
    streaming_gemini, environment, monkeypatch, tmp_path, capsys
):
    """The same steps in this interpreter, where a driver can declare streaming: the value each
    step receives is the one it hands on, not its default."""
    from realtime_framework.evaluation import _trial, harbor_cli
    from realtime_framework.evaluation.effective_config import _trial_command
    from realtime_framework.evaluation.harbor import ApptainerEnvironment, RealtimeBenchOptions

    plan = tmp_path / "streaming.yaml"
    plan.write_text(PLAN.format(mode="streaming"))
    # suite_dataset links task folders under temp/; the job only needs the selected names.
    monkeypatch.setattr(
        harbor_cli, "suite_dataset",
        lambda suite, selected: {"path": suite, "task_names": selected},
    )
    (agent,) = harbor_cli.plan_job(plan)["agents"]
    options = RealtimeBenchOptions(**agent["kwargs"])
    assert options.interaction_mode == "streaming"
    # RealtimeBenchAgent.run: a preset named only by -m becomes the catalog model.
    request = {"model": GEMINI, "mode": "default", **options.model_dump(), "catalog_model": GEMINI}
    task = {
        "suite_id": "public", "task_id": WEB, "fps": request["fps"],
        "model_input_fps": request["model_input_fps"],
        "interaction_mode": request["interaction_mode"],
        "duration_minutes": request["duration_minutes"], "seed": request["seed"],
    }
    environment_ = types.SimpleNamespace(python="python", lane=1, task=task)
    command = ApptainerEnvironment._runner_command(environment_, {**request, "task": task})
    trial_argv = command[3:]
    assert trial_argv[trial_argv.index("--interaction-mode") + 1] == "streaming"
    assert _trial.main([*trial_argv, "--dry-run"]) == 0
    settings = json.loads(capsys.readouterr().out)
    assert settings["plan"]["runtime"]["interaction_mode"] == "streaming"
    runner_argv, _ = _trial_command(settings["plan"])
    assert runner_argv[runner_argv.index("--interaction-mode") + 1] == "streaming"


def _trial_identity(interaction_mode) -> tuple[dict, dict, dict]:
    """A trial's task.json (evaluation/harbor.py resolve), its request, and the one-task local
    report whose group carries the identity reporting.report found in the run summary."""
    task = {
        "suite_id": "public", "suite_version": "1.0", "suite_hash": "0" * 64,
        "harness_version": "1.0", "harness_hash": "1" * 64, "task_id": WEB,
        "category": "web", "fps": 1.0, "model_input_fps": 1.0, "duration_minutes": 5.0,
        "seed": 42,
    }
    if interaction_mode is not None:
        task["interaction_mode"] = interaction_mode
    request = {
        "model": GEMINI, "mode": "default", "model_revision": None, "clock": "wall",
        "task": task, "suite_tasks": {WEB: "web"},
    }
    group = {
        **{field: task[field] for field in ("suite_id", "suite_version", "suite_hash")},
        "harness_hash": task["harness_hash"], "fps": 1.0, "model_input_fps": 1.0,
        "interaction_mode": interaction_mode, "seed": 42, "public_category": "web",
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


def test_harbor_verify_compares_the_mode():
    from realtime_framework.evaluation.harbor import verify

    task, request, report = _trial_identity("turn_based")
    assert verify(report, task, request) == {"reward": 1.0}
    report["groups"][0]["interaction_mode"] = "streaming"
    with pytest.raises(ValueError, match="Report interaction_mode mismatch"):
        verify(report, task, request)


def _job(path, request):
    trial = path / f"{WEB}__trial" / "agent"
    trial.mkdir(parents=True)
    (trial / "request.json").write_text(json.dumps(request))
    return path


@pytest.mark.parametrize(
    "recorded, reported", [("streaming", "streaming"), (None, "turn_based")],
    ids=["recorded", "before-the-mode"],
)
def test_a_job_reports_its_mode(tmp_path, recorded, reported):
    """A job recorded before the interaction mode existed has none in its task.json."""
    from realtime_framework.evaluation.harbor import aggregate

    report = aggregate(_job(tmp_path, _trial_identity(recorded)[1]))
    assert report["interaction_mode"] == reported
