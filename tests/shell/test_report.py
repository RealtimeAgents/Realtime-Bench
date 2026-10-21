"""The report over what the kernel's shell leaves (evaluation/reporting.py report): its action
lifecycle and its token accounting agree, also when the episode gave up on a generation whose
decision goes unreported (K-a; shell.py _given_up). The token totals still count every
generation, a synthesized record without usage leaving them incomplete (decision 48)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from realtime_framework.entity.actions import Action
from realtime_framework.evaluation.reporting import episode_outcome, report
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
]


@pytest.mark.parametrize("replies, world, solved, outcome, responses", EPISODES)
def test_a_wall_clock_episode_is_scored_whatever_became_of_its_last_generation(
    tmp_path, replies, world, solved, outcome, responses
):
    episode = run("wall", 3.0, replies, world)
    env = episode.env
    accounting = env._token_accounting.summary()
    assert accounting["responses"] == responses
    task = _task(
        tmp_path, env.lifecycle, accounting, env._termination, horizon_s=3.0, solved=solved
    )
    assert (task["status"], task["outcome"], task.get("error")) == ("scored", outcome, None)
    assert task["charged_tokens"] == [40] * len(env.lifecycle)


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
