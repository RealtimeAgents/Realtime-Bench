"""The launch file the Harbor trial worker writes for the runner, made by the trial worker's dry
run (evaluation/_trial.py main), for the tests that run the runner on one."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

from realtime_framework.evaluation import _trial


def trial_launch(monkeypatch, directory, *argv) -> Path:
    """Write the launch file the trial worker writes for ``argv`` into ``directory``, as its dry
    run prints it, and return its path. What the trial worker writes into os.environ, and the
    runner's environment it would exec the runner with (_trial.runner_environment), are set
    through ``monkeypatch``, which undoes them."""
    before = dict(os.environ)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            assert _trial.main([*argv, "--dry-run"]) == 0
    finally:
        written = {name: value for name, value in os.environ.items() if before.get(name) != value}
        os.environ.clear()
        os.environ.update(before)
    settings = json.loads(out.getvalue())
    for name, value in {**written, **_trial.runner_environment(settings["plan"])}.items():
        monkeypatch.setenv(name, value)
    path = Path(directory) / _trial.LAUNCH_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n")
    return path
