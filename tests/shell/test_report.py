"""The report over what the kernel's shell leaves (evaluation/reporting.py report): its action
lifecycle and its token accounting agree, also when the episode gave up on a generation whose
decision goes unreported (K-a; shell.py _given_up). The token totals still count every
generation, a synthesized record without usage leaving them incomplete (decision 48). A run whose
lifecycle and token accounting disagree is not scored, nor is a run summary that contradicts
itself (reporting.summary_ok)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from realtime_framework.clock import SystemClock
from realtime_framework.entity.actions import Action
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.reporting import (
    EpisodeTrace,
    episode_outcome,
    report,
    summarize_run_metrics,
)
from tests.config.test_interaction_mode import WEB, _report_root

from .world import CLICK, Reply, World, run

BASELINES = Path(__file__).parents[1] / "replay" / "baselines"
USAGE = {
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 40,
        "total_tokens": 1040,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
    "provider_protocol_family": "chat_completions",
}


def _task(
    root: Path,
    lifecycle: list[dict],
    token_accounting: dict,
    termination: dict,
    *,
    horizon_s: float,
    solved: bool = False,
) -> dict:
    """report()'s row for one task whose run left this lifecycle, token accounting and
    termination record, in a local report root as runner.main() leaves one (_report_root)."""
    _report_root(root, "turn_based")
    run_dir = root / "cell" / WEB
    path = run_dir / "run_summary.json"
    summary = json.loads(path.read_text())
    summary.update(
        token_accounting=token_accounting,
        decisions=len(lifecycle),
        termination=termination,
        stop_reason=termination["stop_reason"],
        max_minutes=horizon_s / 60,
        solved=solved,
        verification={"resolved": solved, "evaluation_version": 1},
    )
    summary["outcome"] = episode_outcome(summary)["outcome"]
    path.write_text(json.dumps(summary))
    (run_dir / "action_lifecycle.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in lifecycle)
    )
    (group,) = report(root)["groups"]
    (task,) = group["tasks"]
    return task


EPISODES = [
    # The deadline cuts the second generation, the usual wall-clock timeout.
    pytest.param(
        [Reply(0.5, CLICK, fields=USAGE), Reply(hang=True)],
        World(),
        False,
        "timeout",
        2,
        id="deadline-cuts-a-generation",
    ),
    # The episode ends during a wait: no generation is in flight.
    pytest.param(
        [Reply(0.5, CLICK, fields=USAGE)] * 2 + [Reply(0.5, Action.wait(5.0), fields=USAGE)],
        World(),
        False,
        "timeout",
        3,
        id="no-generation-in-flight",
    ),
    # Back at 2.9 s, during the end check asked at 2 s, which takes 1.5 s: settled, and then
    # discarded for the deadline the kernel handles first (decision 47, N3).
    pytest.param(
        [Reply(2.9, CLICK, fields=USAGE)],
        World(slow_checks=[(2.0, 2.5, 1.5)]),
        False,
        "timeout",
        1,
        id="settled-then-discarded",
    ),
    # Back at 1.9 s, during the end check asked at 1 s that sees the task won: settled, then
    # discarded.
    pytest.param(
        [Reply(1.9, CLICK, fields=USAGE)],
        World(ends_at=0.5, end_reason="won", slow_checks=[(1.0, 1.5, 1.0)]),
        True,
        "pass",
        1,
        id="won-settled-then-discarded",
    ),
    # The task is lost from 0.5 s, which the end check at 1 s sees, while the only generation
    # would take 2 s: the task's end gives it up. Its synthesized record stays out of the totals
    # (N34), which count no response, and no decision is reported.
    pytest.param(
        [Reply(2.0, CLICK, fields=USAGE)],
        World(ends_at=0.5, end_reason="lost"),
        False,
        "fail",
        0,
        id="task-end-during-the-first-generation",
    ),
]


@pytest.mark.parametrize("replies, world, solved, outcome, responses", EPISODES)
def test_a_wall_clock_episode_is_scored_whatever_became_of_its_last_generation(
    tmp_path, replies, world, solved, outcome, responses
):
    episode = run("wall", 3.0, replies, world)
    env, result = episode.env, episode.result
    accounting = result.token_accounting.summary()
    assert accounting["responses"] == responses
    task = _task(
        tmp_path, env.lifecycle, accounting, result.termination_record, horizon_s=3.0, solved=solved
    )
    assert (task["status"], task["outcome"], task.get("error")) == ("scored", outcome, None)
    assert task["charged_tokens"] == [40] * len(env.lifecycle)


def test_a_move_over_the_per_move_cap_leaves_a_valid_run_scored(tmp_path):
    """The second generation, asked at 1 s, hangs past the 1 s per-move cap: its move is reported
    as skipped, with no usage and so no clock count (episode_records.timed_out_result), and the
    third decision is applied. No decision goes unreported, and the totals the lifecycle is
    checked against have a known total but no total."""
    replies = [Reply(0.5, CLICK, fields=USAGE), Reply(hang=True), Reply(0.5, CLICK, fields=USAGE)]
    episode = run("wall", 3.0, replies, World(), per_move_cap_s=1.0)
    env, result = episode.env, episode.result
    accounting = result.token_accounting.summary()
    assert episode.statuses() == ["applied", "skipped_timeout", "applied"]
    assert [record["clock_output_tokens"] for record in env.lifecycle] == [40, None, 40]
    clock = accounting["counts"]["clock_output_tokens"]
    assert "reported_decisions" not in accounting
    assert (clock["total"], clock["known_total"]) == (None, 80)
    task = _task(tmp_path, env.lifecycle, accounting, result.termination_record, horizon_s=3.0)
    assert (task["status"], task["outcome"], task.get("error")) == ("scored", "timeout", None)
    assert (task["charged_tokens"], task["charged_tokens_complete"]) == ([], False)


def test_a_move_skipped_with_its_settled_result_leaves_a_valid_run_scored(tmp_path):
    """At 2 fps with a 1 s per-move cap, the first generation is back at 0.9 s, during the end
    check asked at 0.5 s, which takes 0.6 s: the kernel handles the cap first and skips the move
    with that result (decision 47). Three decisions are applied after it, and the deadline cuts
    the fifth generation. The skipped move is reported, with its count, so it stays in the
    subtotal of reported decisions; only the generation the deadline cut goes unreported."""
    replies = [Reply(0.9, CLICK, fields=USAGE)] + [Reply(0.3, CLICK, fields=USAGE)] * 3
    replies.append(Reply(hang=True))
    world = World(slow_checks=[(0.5, 0.6, 0.6)])
    episode = run("wall", 4.0, replies, world, fps=2.0, per_move_cap_s=1.0)
    env, result = episode.env, episode.result
    accounting = result.token_accounting.summary()
    assert episode.statuses() == ["skipped_timeout", "applied", "applied", "applied"]
    assert accounting["responses"] == 5
    task = _task(tmp_path, env.lifecycle, accounting, result.termination_record, horizon_s=4.0)
    assert (task["status"], task["outcome"], task.get("error")) == ("scored", "timeout", None)
    assert task["charged_tokens"] == [40] * 4


@pytest.mark.parametrize(
    "counts",
    [
        pytest.param(None, id="no-subtotal"),
        pytest.param({"known_total": 41}, id="known-total"),
        pytest.param({"complete": False}, id="complete"),
        pytest.param({"responses_with_count": 2}, id="responses-with-count"),
    ],
)
def test_a_token_accounting_unlike_the_lifecycle_is_not_scored(tmp_path, counts):
    """The deadline cuts the second generation: the lifecycle reports the first decision, with
    its 40 tokens, and so does the token accounting's subtotal of reported decisions. Without
    that subtotal, or with one of its counts changed, the report refuses the run."""
    episode = run("wall", 3.0, [Reply(0.5, CLICK, fields=USAGE), Reply(hang=True)], World())
    env, result = episode.env, episode.result
    accounting = result.token_accounting.summary()
    subtotal = accounting["reported_decisions"]["counts"]["clock_output_tokens"]
    assert subtotal == {
        "total": 40, "known_total": 40, "responses_with_count": 1, "complete": True,
    }  # fmt: skip
    if counts is None:
        del accounting["reported_decisions"]
    else:
        subtotal.update(counts)
    task = _task(tmp_path, env.lifecycle, accounting, result.termination_record, horizon_s=3.0)
    assert (task["status"], task["outcome"], task.get("error")) == (
        "invalid",
        "error",
        "Charged-token accounting mismatch",
    )


def test_a_decision_recorded_twice_is_not_scored(tmp_path):
    """Three decisions are reported, each with its 40 tokens, and the token accounting agrees
    with their records; but the lifecycle has the first decision's record twice and none of the
    second's."""
    replies = [Reply(0.5, CLICK, fields=USAGE)] * 2 + [Reply(0.5, Action.wait(5.0), fields=USAGE)]
    episode = run("wall", 3.0, replies, World())
    env, result = episode.env, episode.result
    first, _, third = env.lifecycle
    accounting = result.token_accounting.summary()
    task = _task(
        tmp_path, [first, first, third], accounting, result.termination_record, horizon_s=3.0
    )
    assert (task["status"], task["outcome"], task.get("error")) == (
        "invalid",
        "error",
        "Decision count mismatch",
    )


@pytest.mark.parametrize("left_out", [1, -1], ids=["one-more", "one-fewer"])
def test_a_subtotal_of_another_number_of_decisions_is_not_scored(tmp_path, left_out):
    """The second reply reports no usage, and the deadline cuts the third generation: the
    lifecycle reports two decisions, one without a count, and so does the subtotal of reported
    decisions. A subtotal that left one decision more or fewer out, its counts as they are, is
    refused: with a decision without a count, the counts alone cannot tell."""
    replies = [Reply(0.5, CLICK, fields=USAGE), Reply(0.5, CLICK), Reply(hang=True)]
    episode = run("wall", 3.0, replies, World())
    env, result = episode.env, episode.result
    accounting = result.token_accounting.summary()
    reported = accounting["reported_decisions"]
    assert [record["clock_output_tokens"] for record in env.lifecycle] == [40, None]
    assert reported["responses"] == 2
    reported["responses"] -= left_out
    task = _task(tmp_path, env.lifecycle, accounting, result.termination_record, horizon_s=3.0)
    assert (task["status"], task["outcome"], task.get("error")) == (
        "invalid",
        "error",
        "Charged-token accounting mismatch",
    )


def _baselines() -> list[str]:
    """The replay baselines of a valid run that record its token accounting (tests/replay/loop.py
    records): the real adapter and the shell left them."""
    names = []
    for path in sorted(BASELINES.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        outcome = next((row for row in rows if row.get("record") == "outcome"), None)
        if outcome is not None and "token_accounting" in outcome and not outcome["invalid_run"]:
            names.append(path.name)
    return names


@pytest.mark.parametrize("name", _baselines())
def test_a_replayed_run_passes_the_reports_charged_token_check(tmp_path, name):
    rows = [json.loads(line) for line in (BASELINES / name).read_text().splitlines()]
    header = next(row for row in rows if row.get("record") == "header")
    outcome = next(row for row in rows if row.get("record") == "outcome")
    lifecycle = [
        call["record"]
        for row in rows
        for call in row.get("calls", []) + row.get("finalize_calls", [])
        if call.get("call") == "env.record_lifecycle"
    ]
    task = _task(
        tmp_path,
        lifecycle,
        outcome["token_accounting"],
        outcome["termination"],
        horizon_s=60 * header["max_minutes"],
    )
    assert (task["status"], task.get("error")) == ("scored", None)


@pytest.mark.parametrize(
    "change",
    [
        # The recorded outcome is not the one its fields give.
        pytest.param({"outcome": "fail"}, id="outcome"),
        # An error is recorded, though the run is not marked invalid.
        pytest.param({"error": "Traceback (most recent call last): ..."}, id="error"),
        # The evaluation did not resolve the task the summary says is solved.
        pytest.param(
            {"verification": {"resolved": False, "evaluation_version": 1}}, id="verification"
        ),
    ],
)
def test_a_run_summary_that_contradicts_itself_is_not_scored(tmp_path, change):
    """The scored run summary of a solved task (_report_root), with one field changed."""
    root = _report_root(tmp_path, "turn_based")
    path = root / "cell" / WEB / "run_summary.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    (group,) = report(root)["groups"]
    (task,) = group["tasks"]
    assert (task["status"], task["error_reason"], task.get("error")) == (
        "invalid",
        "result_validation_failed",
        "Unscored result or identity mismatch",
    )


def test_a_run_whose_trace_has_no_score_is_not_scored(tmp_path):
    """The scored run summary of a solved task (_report_root), whose episode trace is intact and
    ends with EpisodeFinished but has no ScoreProduced (reporting._episode_trace_ok)."""
    root = _report_root(tmp_path, "turn_based")
    run_dir = root / "cell" / WEB
    (run_dir / "episode_trace.jsonl").unlink()
    trace = EpisodeTrace(run_dir / "episode_trace.jsonl", run_id=WEB, clock=SystemClock())
    trace.emit(TraceEvent.EPISODE_STARTED, {"interaction_mode": "turn_based"})
    trace.emit(TraceEvent.SNAPSHOT_FROZEN, {})
    trace.emit(TraceEvent.EPISODE_FINISHED, {})
    path = run_dir / "run_summary.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "episode_trace": trace.describe()}))
    trace.close()
    (group,) = report(root)["groups"]
    (task,) = group["tasks"]
    assert (task["status"], task["error_reason"], task.get("error")) == (
        "invalid",
        "result_validation_failed",
        "Unscored result or identity mismatch",
    )


def test_a_decision_of_exactly_one_frame_of_tokens_does_not_cross_a_frame(tmp_path):
    """On the token clock a decision crosses a frame when it is charged more than the B tokens of
    one (reporting.summarize_run_metrics): of a decision charged B and one charged B + 1, only the
    second does."""
    path = tmp_path / "action_lifecycle.jsonl"
    records = [
        {"decision": 0, "clock_output_tokens": 64},
        {"decision": 1, "clock_output_tokens": 65},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    metrics = summarize_run_metrics(
        path,
        total_decisions=2,
        solved=False,
        clock_mode="token",
        tokens_per_frame=64,
        fps=1.0,
        termination=None,
        horizon_s=300.0,
    )
    assert (metrics["cross_frame_decisions"], metrics["cross_frame_rate"]) == (1, 0.5)
