"""Harbor's verifier (evaluation/harbor_verifier.py) grades a trial twice. In the trial,
run_benchmark copies the module whole to /tests/verify.py, which the task's tests/test.sh runs with
the sandbox image's python3 (Ubuntu 22.04.5: Python 3.10.12), where only the standard library is
installed. On the host, harbor.aggregate grades the trial's report again and holds the reward the
trial recorded to its own. On the same input the delivered copy must reach the host's reward and
error.
"""

from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys

import pytest
from harbor.environments.base import ExecResult

from realtime_framework.evaluation import harbor

from .test_frame_rates import _trial

# The runner's evaluation details (logs/agent/benchmark/evaluation.json), which the verifier
# keeps beside the reward.
EVALUATION = {"resolved": True, "score": 3}


def _deliver(staging, task, request, report):
    """run_benchmark of a trial whose runner writes the given report (none for None) and its
    evaluation details: the trial directory, with the verifier run_benchmark delivered."""
    trial = object.__new__(harbor.ApptainerEnvironment)
    trial.root = trial.staging = staging
    for folder in ("workspace", "tests", "logs/agent/benchmark", "logs/verifier", "logs/artifacts"):
        (staging / folder).mkdir(parents=True)
    trial.task, trial._launched = task, False
    trial._runner_command = lambda request: ["runner"]
    trial._runner_environment = lambda extra_env: {}

    async def run(command, **options):
        if report is not None:
            (staging / "workspace/report.json").write_text(json.dumps(report))
        (staging / "logs/agent/benchmark/evaluation.json").write_text(
            json.dumps({"evaluation": EVALUATION})
        )
        return ExecResult(return_code=0, stdout="", stderr="")

    trial._run = run
    asyncio.run(trial.run_benchmark(request))
    return staging


def _verify_in_trial(root):
    """The delivered verifier run as test.sh runs it, in an interpreter that imports nothing but
    the standard library (-I -S), with the trial directory as its root: reward.json and
    evaluation.json."""
    subprocess.run(
        [sys.executable, "-I", "-S", str(root / "tests/verify.py"), str(root)],
        cwd=root / "workspace",
        capture_output=True,
        check=True,
    )
    return [
        json.loads((root / "logs/verifier" / name).read_text())
        for name in ("reward.json", "evaluation.json")
    ]


def _row(report):
    (group,) = report["groups"]
    (row,) = group["tasks"]
    return row


def _fail(task, request, report):
    _row(report).update(solved=False, outcome="fail")


def _timeout(task, request, report):
    _row(report).update(solved=False, outcome="timeout")


def _token_clock(task, request, report):
    request.update(clock="token", tokens_per_frame=64)
    report["groups"][0].update(
        protocol="time-aware-token-v1", clock_mode="token", tokens_per_frame=64
    )


def _invalid(task, request, report):
    _row(report).update(
        status="invalid", solved=False, outcome="error", error_category="evaluator",
        error_reason="result_validation_failed",
    )
    report["groups"][0].update(scored=0, invalid=1)


def _seed(task, request, report):
    report["groups"][0]["seed"] = 7


def _configuration_changed(task, request, report):
    request["configuration_sha256"] = "4" * 64


@pytest.mark.parametrize(
    "edit, reward, error",
    [
        (None, 1.0, None),
        (_fail, 0.0, None),
        (_timeout, 0.0, None),
        (_token_clock, 1.0, None),
        (_invalid, 0.0, "ValueError: Missing, duplicate, invalid or unscored task"),
        (_seed, 0.0, "ValueError: Report seed mismatch"),
        (_configuration_changed, 0.0, "ValueError: Model configuration changed after resolution"),
    ],
    ids=["pass", "fail", "timeout", "token-clock", "invalid-row", "seed", "configuration"],
)
def test_the_delivered_verifier_grades_as_the_host_does(tmp_path, edit, reward, error):
    task, request, report = _trial(1.0)
    if edit:
        edit(task, request, report)
    root = _deliver(tmp_path.resolve(), task, request, report)
    recorded, details = _verify_in_trial(root)
    # The host grades the files the trial recorded, as harbor.aggregate does.
    report, task, request = (
        json.loads((root / path).read_text())
        for path in ("workspace/report.json", "tests/task.json", "workspace/request.json")
    )
    try:
        graded, raised = harbor.verify(report, task, request), None
    except Exception as exc:
        graded, raised = {"reward": 0.0}, f"{type(exc).__name__}: {exc}"
    assert (graded, raised) == ({"reward": reward}, error)
    assert recorded == graded
    assert (details["reward"], details["error"], details["evaluation"]) == (
        graded["reward"], raised, EVALUATION,
    )


def test_a_trial_without_a_report_scores_0_as_a_missing_report(tmp_path):
    task, request, _ = _trial(1.0)
    root = _deliver(tmp_path.resolve(), task, request, None)
    recorded, details = _verify_in_trial(root)
    assert recorded == {"reward": 0.0}
    assert (
        details["outcome"], details["error_category"], details["error_reason"],
        details["evaluation"],
    ) == ("error", "infrastructure", "missing_report", EVALUATION)


def test_the_delivered_verifier_is_python_3_10(tmp_path):
    """The host runs Python 3.12 or later; this checks the syntax only, not the standard library
    API the verifier uses."""
    task, request, report = _trial(1.0)
    root = _deliver(tmp_path.resolve(), task, request, report)
    ast.parse((root / "tests/verify.py").read_text(), feature_version=(3, 10))
