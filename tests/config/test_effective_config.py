"""The effective-config export reaches the runner's decision loop from each entry point."""

from __future__ import annotations

import json
import os

from realtime_framework.evaluation.effective_config import export

MODEL, TASK = "claude-opus-5", "gh_timed_chess"
KEY = "sk-export-test-not-a-key"


def test_runner_and_trial_reach_the_decision_loop():
    environ = {"HOME": os.environ.get("HOME", "/"), "PATH": "/usr/bin:/bin"}
    environ["OPENROUTER_API_KEY"] = KEY
    exported = export(
        environ=environ,
        env_file="/nonexistent/.env",
        entries=["runner", "trial"],
        models=[MODEL],
        tasks=[TASK],
    )
    runner = exported["entries"]["runner"]["models"][MODEL]
    trial = exported["entries"]["trial"]["models"][MODEL]
    assert runner["error"] is None and trial["error"] is None
    # Each entry point keeps its own default clock (runner.py:1408; _trial.py:59).
    assert runner["run"]["clock_mode"] == "token" and trial["run"]["clock_mode"] == "wall"
    # The hash _trial's runner records before the episode (reporting.py:1503-1531) and after setup
    # (runner.py:1666-1687) agree.
    assert trial["expected_runtime_config_hash"] == trial["runtime_config_hash"]
    assert KEY not in json.dumps(exported), "the export must not contain secrets"
