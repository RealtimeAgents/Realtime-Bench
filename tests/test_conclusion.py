"""How a finished run is concluded: whether it counts and, when not, the failure it is invalid for.
A re-evaluation of its saved state (reporting.reevaluate_main) concludes it again from its run
summary as written, which may come from an earlier version of the runner."""

from __future__ import annotations

import json
import sys

from realtime_framework.evaluation import reporting

HEALTHY = {"resolved": True, "evaluation_version": 1}
TERMINATION = {"stop_reason": "time_limit", "end_s": 300.0, "snapshot_s": 300.5,
               "snapshot_delay_s": 0.5}  # fmt: skip


def _summary(**fields) -> dict:
    """A run summary as runner.py main writes one, with the fields a re-evaluation reads."""
    return {
        "benchmark_eligible": True,
        "run_status": "invalid",
        "invalid_run": None,
        "solved": None,
        "stop_reason": "invalid_run",
        "loop_stop_reason": "time_limit",
        "termination": TERMINATION,
        "max_minutes": 5,
        "world_time_s": 300.0,
        "verification": None,
        **fields,
    }


def _reevaluated(tmp_path, monkeypatch, capsys, summary: dict, evaluation: dict):
    """reevaluate --replace-summary of a run whose summary is ``summary``, the saved state's
    evaluation stubbed: the summary it writes and the line it prints."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(reporting, "reeval", lambda run_dir, **options: (evaluation, {}))
    monkeypatch.setattr(sys, "argv", ["reevaluate", str(run_dir), "--replace-summary"])
    capsys.readouterr()
    assert reporting.reevaluate_main() is None
    printed = json.loads(capsys.readouterr().out.splitlines()[-1])
    return json.loads((run_dir / "run_summary.json").read_text()), printed


def test_an_empty_invalid_run_does_not_keep_a_re_evaluated_run_invalid(
    tmp_path, monkeypatch, capsys
):
    """A summary whose invalid run is an empty record: an empty record names no failure, so a
    healthy re-evaluation scores the run, and the empty record is written back as it was."""
    replaced, printed = _reevaluated(
        tmp_path, monkeypatch, capsys, _summary(invalid_run={}), HEALTHY
    )
    assert (replaced["run_status"], replaced["invalid_run"], replaced["solved"]) == (
        "scored", {}, True,
    )  # fmt: skip
    assert (replaced["failure_attribution"], replaced["stop_reason"]) == (None, "time_limit")
    assert (replaced["outcome"], replaced["error_category"], replaced["error_reason"]) == (
        "error", "unknown", "unknown",
    )  # fmt: skip
    assert (printed["run_status"], printed["solved"], printed["invalid_reason"]) == (
        "scored", True, None,
    )  # fmt: skip


def test_a_failure_without_a_reason_keeps_a_re_evaluated_run_invalid(tmp_path, monkeypatch, capsys):
    """A summary whose environment failure has no reason: the run stays invalid for it, and its
    attribution is the failure's category; the line printed names no reason."""
    replaced, printed = _reevaluated(
        tmp_path, monkeypatch, capsys, _summary(invalid_run={"category": "environment"}), HEALTHY
    )
    assert (replaced["run_status"], replaced["invalid_run"], replaced["solved"]) == (
        "invalid", {"category": "environment"}, None,
    )  # fmt: skip
    assert (replaced["failure_attribution"], replaced["stop_reason"]) == (
        "environment",
        "invalid_run",
    )
    assert (replaced["error_category"], replaced["error_reason"]) == ("environment", "unknown")
    assert (printed["run_status"], printed["solved"], printed["invalid_reason"]) == (
        "scored", True, None,
    )  # fmt: skip


def test_a_failure_without_a_category_keeps_a_re_evaluated_run_invalid(
    tmp_path, monkeypatch, capsys
):
    """A summary whose failure has a reason but no category: not the evaluator's, so the run
    stays invalid for it, attributed to no category."""
    replaced, printed = _reevaluated(
        tmp_path, monkeypatch, capsys, _summary(invalid_run={"reason": "lost"}), HEALTHY
    )
    assert (replaced["run_status"], replaced["invalid_run"]) == ("invalid", {"reason": "lost"})
    assert (replaced["failure_attribution"], replaced["error_category"]) == (None, "unknown")
    assert (printed["run_status"], printed["invalid_reason"]) == ("invalid", "lost")


def test_a_summary_without_an_invalid_run_is_scored_by_a_healthy_re_evaluation(
    tmp_path, monkeypatch, capsys
):
    summary = _summary()
    del summary["invalid_run"]
    replaced, printed = _reevaluated(tmp_path, monkeypatch, capsys, summary, HEALTHY)
    assert (replaced["run_status"], replaced["invalid_run"], replaced["outcome"]) == (
        "scored", None, "pass",
    )  # fmt: skip
    assert printed["run_status"] == "scored"


def test_a_run_that_stays_invalid_needs_no_benchmark_eligibility(tmp_path, monkeypatch, capsys):
    """A summary without benchmark_eligible, invalid for its environment: the re-evaluated run
    stays invalid, which needs no eligibility to say."""
    summary = _summary(invalid_run={"category": "environment", "reason": "cdp_close_failed"})
    del summary["benchmark_eligible"]
    replaced, _ = _reevaluated(tmp_path, monkeypatch, capsys, summary, HEALTHY)
    assert (replaced["run_status"], replaced["invalid_run"]["reason"]) == (
        "invalid", "cdp_close_failed",
    )  # fmt: skip
