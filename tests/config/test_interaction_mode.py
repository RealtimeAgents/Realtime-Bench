"""The interaction mode, turn_based or streaming, is a run setting like the frame rates.

runner.main() takes it as --interaction-mode, records it in the run identity (the episode start,
run_config.json and its hash, the expected summary and the run summary) and passes it to the
pre-run checks (providers/preflight.py check_interaction_mode). No driver declares streaming yet,
so every streaming run is refused before the agent is built.
"""

from __future__ import annotations

import json
import os

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.evaluation.effective_config import _runner_entry

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
