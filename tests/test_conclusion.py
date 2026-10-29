"""How a finished run is concluded: whether it counts and, when not, the failure it is invalid for.
A re-evaluation of its saved state (reporting.reevaluate_main) concludes it again from its run
summary as written, which may come from an earlier version of the runner."""

from __future__ import annotations

import json
import sys

import pytest

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
    evaluation stubbed: the summary it writes and the line it prints, which says what the
    summary says."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(reporting, "reeval", lambda run_dir, **options: (evaluation, {}))
    monkeypatch.setattr(sys, "argv", ["reevaluate", str(run_dir), "--replace-summary"])
    capsys.readouterr()
    assert reporting.reevaluate_main() is None
    printed = json.loads(capsys.readouterr().out.splitlines()[-1])
    written = json.loads((run_dir / "run_summary.json").read_text())
    assert (printed["run_status"], printed["solved"], printed["invalid_reason"]) == (
        written["run_status"], written["solved"], (written["invalid_run"] or {}).get("reason"),
    )  # fmt: skip
    return written, printed


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
    attribution is the failure's category; the line printed says so and names no reason."""
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
        "invalid", None, None,
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


ENVIRONMENT = {"category": "environment", "reason": "action_execution_failed", "detail": "down"}
EVALUATOR = {"category": "evaluator", "reason": "evaluation_backend_unhealthy", "detail": "{}"}


def test_a_run_without_a_failure_counts_as_its_evaluation_resolved():
    """A run that found no failure is scored, solved as its evaluation resolved and stopped as
    its episode stopped; one of a model outside maintained results is experimental."""
    evaluation = {"resolved": False, "evaluation_version": 1, "progress": 0.5}
    conclusion = reporting.conclude(
        [], evaluation, benchmark_eligible=True, stop_reason="time_limit"
    )
    assert (conclusion.run_status, conclusion.solved, conclusion.stop_reason) == (
        "scored", False, "time_limit",
    )  # fmt: skip
    assert (conclusion.invalid_run, conclusion.failure_attribution) == (None, None)
    assert (conclusion.progress, conclusion.listed()) == (0.5, {})
    experimental = reporting.conclude(
        [], evaluation, benchmark_eligible=False, stop_reason="time_limit"
    )
    assert experimental.run_status == "experimental"


def test_a_run_is_invalid_for_the_first_failure_it_found():
    """The run is invalid for the first failure, whatever its evaluation resolved, and lists each
    different failure once in the order found: a failure found twice, as a failed snapshot read
    is by the kernel and by main(), is listed once. Its evaluation's progress still stands."""
    again = {**ENVIRONMENT, "detail": "found again"}
    evaluation = {"resolved": True, "evaluation_version": 1, "progress": 1.0}
    conclusion = reporting.conclude(
        [ENVIRONMENT, EVALUATOR, again], evaluation, benchmark_eligible=True, stop_reason="x"
    )
    assert conclusion.invalid_run is ENVIRONMENT
    assert conclusion.listed() == {"invalid_runs": [ENVIRONMENT, EVALUATOR]}
    assert (conclusion.run_status, conclusion.solved, conclusion.stop_reason) == (
        "invalid", None, "invalid_run",
    )  # fmt: skip
    assert (conclusion.failure_attribution, conclusion.progress) == ("environment", 1.0)
    alone = reporting.conclude([EVALUATOR], evaluation, benchmark_eligible=True, stop_reason="x")
    assert (alone.invalid_run, alone.listed()) == (EVALUATOR, {})


def test_a_run_main_could_not_summarize_is_concluded_from_its_failures_alone():
    """The emergency summary has the failures and nothing else to conclude from."""
    conclusion = reporting.conclude([ENVIRONMENT])
    assert (conclusion.run_status, conclusion.invalid_run, conclusion.solved) == (
        "invalid", ENVIRONMENT, None,
    )  # fmt: skip
    assert (conclusion.stop_reason, conclusion.progress) == ("invalid_run", None)


@pytest.mark.parametrize(
    ("found", "environment_invalid", "valid"),
    [
        ([ENVIRONMENT], False, False),
        ([EVALUATOR], True, False),  # a check found the environment invalid as the run set up
        ([EVALUATOR], False, None),
        ([], False, None),  # nothing checks an environment valid
    ],
)
def test_an_environment_is_invalid_when_a_failure_or_a_check_says_so(
    found, environment_invalid, valid
):
    conclusion = reporting.conclude(
        found, HEALTHY, benchmark_eligible=True, stop_reason="time_limit"
    )
    assert conclusion.environment_valid(environment_invalid) is valid


def test_an_evaluation_not_safe_to_score_is_the_evaluator_s_failure():
    """Its reason is invalid_evaluation's, its detail the evaluation as JSON, to 2000
    characters."""
    assert reporting.evaluator_failure(HEALTHY) is None
    unhealthy = {**HEALTHY, "ok": False}
    assert reporting.evaluator_failure(unhealthy) == {
        "category": "evaluator",
        "reason": "evaluation_backend_unhealthy",
        "detail": json.dumps(unhealthy),
    }
    long = {**HEALTHY, "ok": False, "notes": "n" * 3000}
    assert reporting.evaluator_failure(long)["detail"] == json.dumps(long)[:2000]
    assert reporting.evaluator_failure("crashed") == {
        "category": "evaluator",
        "reason": "evaluation_not_an_object",
        "detail": '"crashed"',
    }


def test_a_re_evaluation_keeps_a_run_invalid_for_a_failure_not_the_evaluator_s():
    """A healthy re-evaluation decides the evaluator's failures alone: the run stays invalid for
    the first earlier failure that was not the evaluator's, from the failures its summary lists,
    or from its invalid run when it found only one."""
    listed = _summary(invalid_run=EVALUATOR, invalid_runs=[EVALUATOR, ENVIRONMENT])
    conclusion = reporting.conclude_reevaluation(listed, HEALTHY)
    assert (conclusion.invalid_run, conclusion.run_status, conclusion.solved) == (
        ENVIRONMENT, "invalid", None,
    )  # fmt: skip
    assert (conclusion.stop_reason, conclusion.failure_attribution) == (
        "invalid_run",
        "environment",
    )
    assert conclusion.listed() == {}  # the summary keeps the run's own list
    alone = reporting.conclude_reevaluation(_summary(invalid_run=ENVIRONMENT), HEALTHY)
    assert alone.invalid_run is ENVIRONMENT


def test_a_re_evaluation_scores_a_run_invalid_for_its_evaluator_alone():
    """Once a healthy evaluation decides the evaluator's failure, the run counts as its summary
    says it may: scored, or experimental outside maintained results; stopped as its loop
    stopped, or as its summary says when the loop's stop was not recorded."""
    evaluation = {**HEALTHY, "progress": 0.25}
    conclusion = reporting.conclude_reevaluation(_summary(invalid_run=EVALUATOR), evaluation)
    assert (conclusion.run_status, conclusion.invalid_run, conclusion.solved) == (
        "scored", None, True,
    )  # fmt: skip
    assert (conclusion.stop_reason, conclusion.progress) == ("time_limit", 0.25)
    experimental = reporting.conclude_reevaluation(
        _summary(invalid_run=EVALUATOR, benchmark_eligible=False, loop_stop_reason=None),
        HEALTHY,
    )
    assert (experimental.run_status, experimental.stop_reason) == ("experimental", "invalid_run")


UNHEALTHY = {**HEALTHY, "ok": False}


@pytest.mark.parametrize(
    "found",
    [[EVALUATOR, ENVIRONMENT], [ENVIRONMENT, EVALUATOR]],
    ids=["evaluator first", "environment first"],
)
def test_a_re_evaluation_not_safe_to_score_keeps_a_failure_not_the_evaluator_s(found):
    """A re-evaluation not safe to score leaves the run invalid for the first failure that was
    not the evaluator's, whichever the run found first: the new evaluation decides the
    evaluator's failure alone."""
    summary = _summary(invalid_run=found[0], invalid_runs=found)
    conclusion = reporting.conclude_reevaluation(summary, UNHEALTHY)
    assert (conclusion.invalid_run, conclusion.failure_attribution) == (ENVIRONMENT, "environment")
    assert (conclusion.run_status, conclusion.solved) == ("invalid", None)


def test_a_re_evaluation_not_safe_to_score_invalidates_a_run_with_no_other_failure():
    """Without a failure that was not the evaluator's, or with only an empty record, a
    re-evaluation not safe to score invalidates the run for the evaluator."""
    for summary in (_summary(invalid_run=EVALUATOR), _summary(), _summary(invalid_run={})):
        conclusion = reporting.conclude_reevaluation(summary, UNHEALTHY)
        assert conclusion.invalid_run == reporting.evaluator_failure(UNHEALTHY)
        assert (conclusion.run_status, conclusion.solved) == ("invalid", None)


def _reevaluated_again(run_dir, monkeypatch, capsys, evaluation: dict) -> dict:
    """reevaluate --replace-summary once more, on the summary the last one wrote."""
    monkeypatch.setattr(reporting, "reeval", lambda run_dir, **options: (evaluation, {}))
    monkeypatch.setattr(sys, "argv", ["reevaluate", str(run_dir), "--replace-summary"])
    reporting.reevaluate_main()
    capsys.readouterr()
    return json.loads((run_dir / "run_summary.json").read_text())


def test_re_evaluations_in_turn_keep_a_run_invalid_for_its_environment(
    tmp_path, monkeypatch, capsys
):
    """A run invalid for its environment alone, re-evaluated in turn not safe to score, healthy,
    and the same again: each keeps it invalid for its environment. None writes an evaluator's
    failure over the environment's, which a later healthy one would then clear."""
    replaced, _ = _reevaluated(
        tmp_path, monkeypatch, capsys, _summary(invalid_run=ENVIRONMENT), UNHEALTHY
    )
    seen = [replaced["invalid_run"]]
    for evaluation in (HEALTHY, UNHEALTHY, UNHEALTHY, HEALTHY):
        replaced = _reevaluated_again(tmp_path / "run", monkeypatch, capsys, evaluation)
        seen.append(replaced["invalid_run"])
    assert seen == [ENVIRONMENT] * 5
    assert (replaced["run_status"], replaced["failure_attribution"]) == ("invalid", "environment")


def test_re_evaluations_in_turn_decide_a_run_invalid_for_its_evaluator_alone(
    tmp_path, monkeypatch, capsys
):
    """A run invalid for its evaluator alone follows each new evaluation."""
    replaced, _ = _reevaluated(
        tmp_path, monkeypatch, capsys, _summary(invalid_run=EVALUATOR), UNHEALTHY
    )
    statuses = [replaced["run_status"]]
    for evaluation in (HEALTHY, UNHEALTHY, HEALTHY):
        statuses.append(
            _reevaluated_again(tmp_path / "run", monkeypatch, capsys, evaluation)["run_status"]
        )
    assert statuses == ["invalid", "scored", "invalid", "scored"]


def test_a_re_evaluation_reads_an_empty_or_missing_record_as_no_failure():
    """An empty invalid run names no failure: the run counts, and the empty record stands as
    its invalid run. A summary that does not say whether the run may count needs to only when
    it does."""
    empty = reporting.conclude_reevaluation(_summary(invalid_run={}), HEALTHY)
    assert (empty.run_status, empty.invalid_run, empty.failure_attribution) == ("scored", {}, None)
    summary = _summary(invalid_run=ENVIRONMENT)
    del summary["benchmark_eligible"]
    assert reporting.conclude_reevaluation(summary, HEALTHY).run_status == "invalid"


@pytest.mark.parametrize(
    ("summary", "valid"),
    [
        (_summary(invalid_run=EVALUATOR, invalid_runs=[EVALUATOR, ENVIRONMENT]), False),
        (_summary(invalid_run=EVALUATOR, environment_valid=False), False),
        (_summary(invalid_run=EVALUATOR, environment_valid=None), None),
    ],
    ids=["invalid for its environment now", "found invalid before", "never found invalid"],
)
def test_a_re_evaluation_says_the_environment_is_not_valid_once_anything_found_it_so(
    tmp_path, monkeypatch, capsys, summary, valid
):
    """A healthy re-evaluation that leaves the run invalid for its environment says the
    environment is not valid; one that scores a run whose environment was found not valid, as
    the run set up, keeps saying so. Nothing makes it valid."""
    replaced, _ = _reevaluated(tmp_path, monkeypatch, capsys, summary, HEALTHY)
    assert replaced["environment_valid"] is valid


def test_the_success_metrics_follow_the_score_and_the_end_of_the_episode():
    """The task-success rate is the score as 1/0, and the time to success the end of a solved
    episode over its horizon; a run summary written before the termination record has no end."""
    end = {**TERMINATION, "end_s": 30.0}
    assert reporting.success_metrics(True, end, 60.0) == {
        "task_success_rate": 1.0,
        "normalized_time_to_success": 0.5,
    }
    assert reporting.success_metrics(False, end, 60.0) == {
        "task_success_rate": 0.0,
        "normalized_time_to_success": None,
    }
    assert reporting.success_metrics(None, end, 60.0) == {
        "task_success_rate": None,
        "normalized_time_to_success": None,
    }
    assert reporting.success_metrics(True, None, 60.0)["normalized_time_to_success"] is None


def test_a_re_evaluation_of_a_summary_without_a_termination_record(tmp_path, monkeypatch, capsys):
    """A run summary written before the termination record, re-evaluated solved: its
    task-success rate counts the score, and it has no time to success."""
    summary = _summary(invalid_run=EVALUATOR, termination=None)
    replaced, _ = _reevaluated(tmp_path, monkeypatch, capsys, summary, HEALTHY)
    assert (replaced["solved"], replaced["task_success_rate"]) == (True, 1.0)
    assert replaced["normalized_time_to_success"] is None


@pytest.mark.parametrize("evaluation", [HEALTHY, UNHEALTHY], ids=["healthy", "not safe to score"])
def test_a_re_evaluation_keeps_the_failures_the_run_found(
    tmp_path, monkeypatch, capsys, evaluation
):
    """invalid_runs is the run's own record of the failures it found, which a re-evaluation
    keeps as written, though the failure it leaves the run invalid for is no longer the first;
    a summary that listed none lists none."""
    found = [EVALUATOR, ENVIRONMENT]
    listed = _summary(invalid_run=EVALUATOR, invalid_runs=found)
    replaced, _ = _reevaluated(tmp_path / "listed", monkeypatch, capsys, listed, evaluation)
    assert (replaced["invalid_run"], replaced["invalid_runs"]) == (ENVIRONMENT, found)
    alone = _summary(invalid_run=ENVIRONMENT)
    replaced, _ = _reevaluated(tmp_path / "alone", monkeypatch, capsys, alone, evaluation)
    assert "invalid_runs" not in replaced
