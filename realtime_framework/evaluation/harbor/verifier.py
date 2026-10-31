"""Grade a harness-produced local report.

Only a verified, solved episode earns reward 1. Every other outcome, including any error, earns 0;
the recorded outcome and error reason keep errors distinguishable from model failures.

Harbor's environment copies this file whole into each trial as /tests/verify.py
(evaluation/harbor/integration.py run_benchmark), which the task's tests/test.sh runs with the sandbox image's
python3 (Ubuntu 22.04.5: Python 3.10.12): it imports only the standard library and keeps to
Python 3.10. The host grades a job's reports again with the same verify (evaluation/harbor/integration.py
aggregate).
"""

import json
import sys
from pathlib import Path


def verify(report, expected, request):
    if request.get("task") != expected:
        raise ValueError("Trial identity mismatch")
    integration = report.get("integration", {})
    if any(
        integration.get(key) != request[key] for key in ("model", "model_revision", "mode")
    ) or not integration.get("configuration_sha256"):
        raise ValueError("Report integration identity mismatch")
    if (request.get("catalog_model") or request.get("configuration_sha256")) and (
        not request.get("configuration_sha256")
        or integration["configuration_sha256"] != request["configuration_sha256"]
    ):
        raise ValueError("Model configuration changed after resolution")
    groups = report.get("groups", [])
    rows = [row for group in groups for row in group.get("tasks", [])]
    if (
        len(groups) != 1
        or len(rows) != 1
        or rows[0].get("status") != "scored"
        or rows[0].get("task_id", "").split("__", 1)[-1] != expected["task_id"]
        or type(rows[0].get("solved")) is not bool
        or rows[0].get("outcome") not in ("pass", "fail", "timeout")
        or (rows[0]["outcome"] == "pass") is not rows[0]["solved"]
    ):
        raise ValueError("Missing, duplicate, invalid or unscored task")
    group = groups[0]
    for field in (
        "suite_id", "suite_version", "suite_hash", "harness_hash", "fps", "model_input_fps",
        "interaction_mode", "seed",
    ):
        if group.get(field) != expected.get(field):
            raise ValueError(f"Report {field} mismatch")
    if (
        group.get("public_category") != expected["category"]
        or group.get("mode") != request["mode"]
        or group.get("protocol")
        != ("time-aware-token-v1" if request.get("clock", "wall") == "token"
            else "wall-clock")
        or group.get("clock_mode", "wall") != request.get("clock", "wall")
        or group.get("tokens_per_frame") != (
            request.get("tokens_per_frame")
            if request.get("clock") == "token" and request.get("tokens_per_frame") != "unlimited"
            else None
        )
        or group.get("cap") != (request.get("duration_minutes") or expected["duration_minutes"])
        or group.get("expected") != 1
        or group.get("scored") != 1
        or group.get("invalid") != 0
        or group.get("missing") != 0
        or group.get("partial") is not False
        or not group.get("runtime_config_hash")
    ):
        raise ValueError("Report runtime or completeness mismatch")
    return {"reward": float(rows[0]["solved"])}


def main(root="/"):
    """Grade the trial whose files are under root: / in the trial, a trial directory in tests."""
    root = Path(root)
    output = root / "logs/verifier/reward.json"
    details = output.with_name("evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    details.unlink(missing_ok=True)
    # Scoring details live with the verifier output; the reward alone stays authoritative.
    # outcome is pass, fail, timeout or error; only a verified pass earns reward 1.
    record = {
        "reward": None,
        "outcome": "error",
        "error_category": "infrastructure",
        "error_reason": "missing_report",
        "error": None,
        "evaluation": None,
    }
    try:
        record["evaluation"] = json.loads(
            (root / "logs/agent/benchmark/evaluation.json").read_text()
        ).get("evaluation")
    except (OSError, ValueError, AttributeError):
        pass
    try:
        report = json.loads((root / "workspace/report.json").read_text())
        rows = [row for group in report.get("groups", []) for row in group.get("tasks", [])]
        if len(rows) == 1:
            for key in ("outcome", "error_category", "error_reason"):
                record[key] = rows[0].get(key)
        expected = json.loads((root / "tests/task.json").read_text())
        request = json.loads((root / "workspace/request.json").read_text())
        reward = verify(report, expected, request)
    except Exception as exc:
        # An error is scored as a failure.
        record["error"] = f"{type(exc).__name__}: {exc}"
        if record["outcome"] != "error":
            record.update(
                outcome="error",
                error_category="evaluator",
                error_reason="report_verification_failed",
            )
        reward = {"reward": 0.0}
    record["reward"] = reward["reward"]
    details.write_text(json.dumps(record, indent=2) + "\n")
    output.write_text(json.dumps(reward) + "\n")
    reason = f" ({record['error_reason']})" if record["outcome"] == "error" else ""
    print(f"{record['outcome']}{reason}: reward={reward['reward']} (details: evaluation.json)")


if __name__ == "__main__":
    # tests/test.sh passes no root.
    main(*sys.argv[1:])
