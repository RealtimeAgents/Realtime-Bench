"""Replay baselines for run_stream's own decision loop on the wall clock.

The real run_stream (evaluation/runner.py:102-1180) drives the chat_completions agent from
build_episode against FakeEnvironment (fake_env.py). Each baseline records, decision by decision,
the runner's calls into the environment, its trace events, the commit statuses the agent receives,
the action lifecycle records and the runner log, then the stop reason and the agent's history.
Record with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it each run is
compared with tests/replay/baselines/.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import pytest

from realtime_framework.evaluation import runner
from realtime_framework.evaluation.reporting import episode_outcome, summarize_run_metrics
from realtime_framework.harnesses.default.harness import DefaultHarness

from . import baseline
from .build import build_episode
from .fake_env import FakeClock, FakeEnvironment, Hang
from .frames import ReplayTask
from .loop import REAL_TIME_FIELDS, check_wall_clock_properties, commit_statuses, records, run_loop
from .scenario import CLICK_AGREE, CLICK_NAME, CLICK_SUBMIT, LONG_SCENARIO, TASK_INSTRUCTION
from .scenario import replies
from .steps import Step
from .stub import ChatCompletionsStub
from .test_chat_completions import MODEL_KEY, chat_completions_config

BASELINES = Path(__file__).parent / "baselines"

# Most replies take half a frame. Decisions 12 and 40 span frame boundaries and decision 21 lands
# exactly on one; LONG_SCENARIO's waits (decisions 5, 33, 56) add their durations. Decision 59
# starts at 63.0 s and replies at 63.875 s, after the 63.75 s deadline (LATE_REPLY_S).
LONG_LATENCIES = [
    {3: 0.25, 8: 0.75, 12: 2.5, 21: 1.0, 26: 0.25, 40: 1.75, 47: 0.75, 59: 0.875}.get(k, 0.5)
    for k in range(len(LONG_SCENARIO))
]
LONG_MAX_MINUTES = 1.0625
# Short cases: a 3.75 s episode at 1 fps, with decisions at 0, 1, 2 and 3 s.
SHORT_MAX_MINUTES = 0.0625
# A reply to a decision at 3.0 s that lands after the deadline but before tick 4, so only the
# deadline checks stop it, not the tick limit (runner.py:526-531).
LATE_REPLY_S = 0.875

TYPE_ADA = Step('{"action": "type", "text": "Ada"}')


@dataclass
class Run:
    config: object
    env: FakeEnvironment
    stub: ChatCompletionsStub
    agent: object
    timeline: list


def _run(tmp_path, steps: list[Step], latencies: list, *, stream: bool, max_minutes, **env_options):
    env_box = []
    with ChatCompletionsStub(
        replies(steps), on_request=lambda index: env_box[0].model_request(index)
    ) as stub:
        config = chat_completions_config(stub.base_url, stream=stream)
        episode = build_episode(
            config,
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=MODEL_KEY,
            clock_mode="wall",
        )
        env = FakeEnvironment(
            episode.frames,
            FakeClock(),
            fps=config.fps,
            max_minutes=max_minutes,
            reply_latencies=latencies,
            **env_options,
        )
        env_box.append(env)
        timeline = run_loop(episode, env)
    assert stub.unscripted == 0, "the agent sent requests beyond the script"
    assert stub.unused_replies == 0, "scripted replies were left unused"
    return Run(config, env, stub, episode.agent, timeline)


def _compare(name: str, run: Run, header: dict) -> None:
    header = {
        "record": "header",
        "api": "chat_completions",
        "clock": "wall",
        "scenario": name,
        "max_minutes": run.env.max_minutes,
        "reply_latencies": [
            {"hang_seconds": value.seconds} if isinstance(value, Hang) else value
            for value in run.env.reply_latencies
        ],
        "config": run.config.model_dump(),
        **header,
    }
    actual = baseline.Normalizer(REAL_TIME_FIELDS)(
        [
            header,
            *records(run.env, run.agent, run.timeline),
            *baseline.trajectory(run.agent.log_dir),
        ]
    )
    path = BASELINES / f"chat_completions-runner-{name}.jsonl"
    if os.environ.get("REPLAY_RECORD") == "1":
        baseline.write(path, actual)
        return
    problems = baseline.compare(baseline.read(path), actual)
    assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
        problems[:60]
    )


# Each episode runs once per module; its property test and its baseline test fail independently.


@pytest.fixture(scope="module", params=[True, False], ids=["stream", "nonstream"])
def long_run(request, tmp_path_factory):
    rejections = {
        k: step.detail for k, step in enumerate(LONG_SCENARIO) if step.status == "rejected"
    }
    return _run(
        tmp_path_factory.mktemp("long"),
        LONG_SCENARIO,
        LONG_LATENCIES,
        stream=request.param,
        max_minutes=LONG_MAX_MINUTES,
        rejections=rejections,
    )


def test_long_scenario_properties(long_run):
    check_wall_clock_properties(long_run.env, "time_limit")
    assert len(long_run.stub.requests) == len(LONG_SCENARIO)
    # run_stream reports the statuses and details steps.py scripts for the same replies.
    assert commit_statuses(long_run.env) == [
        (k, step.status, step.detail) for k, step in enumerate(LONG_SCENARIO)
    ]


def test_long_scenario_baseline(long_run):
    stream = long_run.config.chat_stream
    _compare(f"{'stream' if stream else 'nonstream'}-long", long_run, {"stream": stream})


def _decided_after(applied: int, reason: str):
    """The task decides itself once ``applied`` actions have landed."""

    def stop_check(env):
        landed = sum(execution["status"] == "applied" for execution in env.executions)
        return reason if landed >= applied else None

    return stop_check


@dataclass(frozen=True)
class Case:
    name: str
    steps: list[Step]
    latencies: list
    statuses: list[str]  # the status run_stream commits for each decision
    executed: list[str]  # the action types the environment executes, in order
    stop_reason: str
    env: dict = field(default_factory=dict)
    requests: int | None = None  # when the agent sends more requests than there are steps
    held: tuple[bool, ...] = ()


# The runner waits time_remaining + DEADLINE_GRACE_S for a decision (runner.py:307-309, 337).
# Decision 3 starts at 3.0 s with 0.75 s left.
BOUNDED_WAIT_S = 0.75 + runner.DEADLINE_GRACE_S

CASES = [
    # The reply to decision 3 lands at 3.875 s, after the deadline: not executed
    # (runner.py:526-527).
    Case(
        "late_reply",
        [Step(CLICK_NAME), TYPE_ADA, Step(CLICK_SUBMIT), Step(CLICK_AGREE)],
        [0.5, 0.5, 0.5, LATE_REPLY_S],
        ["applied", "applied", "applied", "not_applied_time_limit"],
        ["click", "type", "click"],
        "time_limit",
    ),
    # DONE becomes WAIT (runner.py:1143-1153) and the episode goes on to the deadline.
    Case(
        "done_reply",
        [Step(CLICK_NAME), Step('{"action": "done"}'), Step(CLICK_SUBMIT), Step(CLICK_AGREE)],
        [0.5, 0.5, 0.5, LATE_REPLY_S],
        ["applied", "wait", "applied", "not_applied_time_limit"],
        ["click", "wait", "click"],
        "time_limit",
    ),
    # The third applied action decides the task; run_stream stops at once (runner.py:545,
    # 1175-1176) and latches the task's reason (runner.py:795-796).
    Case(
        "environment_done",
        [Step(CLICK_NAME), TYPE_ADA, Step(CLICK_SUBMIT)],
        [0.5, 0.5, 0.5],
        ["applied", "applied", "applied"],
        ["click", "type", "click"],
        "task_completed",
        env={"stop_check": _decided_after(3, "task_completed")},
    ),
    # The task is decided at 2.25 s, while decision 2 waits for its reply: the action is not
    # executed (runner.py:528-529) and the reason defaults to game_decided (runner.py:796).
    Case(
        "environment_done_during_reply",
        [Step(CLICK_NAME), TYPE_ADA, Step(CLICK_SUBMIT)],
        [0.5, 0.5, 0.5],
        ["applied", "applied", "not_applied_terminal"],
        ["click", "type"],
        "game_decided",
        env={"stop_check": lambda env: env.clock.now >= env.t0 + 2.25},
    ),
    # Decision 3's reply never comes. The bounded wait expires, the agent is aborted and the
    # decision is skipped (runner.py:337-345, 395-411, 523). The abort closes the held request
    # and ends the decision's epoch, so the abandoned worker does not send it again
    # (harnesses/default/harness.py:1159, 1350; entity/model.py:479-491).
    Case(
        "bounded_call_timeout",
        [Step(CLICK_NAME), TYPE_ADA, Step(CLICK_SUBMIT)],
        [0.5, 0.5, 0.5, Hang(BOUNDED_WAIT_S)],
        ["applied", "applied", "applied", "skipped_timeout"],
        ["click", "type", "click"],
        "time_limit",
        requests=4,
        held=(True,),
    ),
    # A fault that is not the guest refusing the action invalidates the run (runner.py:538-541).
    Case(
        "environment_failure",
        [Step(CLICK_NAME), TYPE_ADA],
        [0.5, 0.5],
        ["applied", "invalid_run"],
        ["click", "type"],
        "invalid_run",
        env={"failures": {1: "guest connection reset"}},
    ),
]


@pytest.fixture(scope="module", params=CASES, ids=[case.name for case in CASES])
def short_run(request, tmp_path_factory):
    case = request.param
    run = _run(
        tmp_path_factory.mktemp(case.name),
        case.steps,
        case.latencies,
        stream=True,
        max_minutes=SHORT_MAX_MINUTES,
        **case.env,
    )
    return case, run


def test_short_case_properties(short_run):
    case, run = short_run
    check_wall_clock_properties(run.env, case.stop_reason)
    assert len(run.stub.requests) == (case.requests or len(case.steps))
    assert tuple(run.stub.held) == case.held
    assert [status for _, status, _ in commit_statuses(run.env)] == case.statuses
    assert [e["action"]["action_type"] for e in run.env.executions] == case.executed


def test_short_case_baseline(short_run):
    case, run = short_run
    _compare(case.name, run, {"stream": True})


# An exception that escapes the decision loop still ends the episode through _finalize
# (runner.py:1226-1236): the snapshot is frozen, the stop reason latched and the run invalid.
# These are property tests; an escaped exception's traceback names files and lines.


def _check_finalized(run: Run, *, at: float, raised: str) -> None:
    check_wall_clock_properties(run.env, "invalid_run")
    assert [call["call"] for call in run.env.calls if call["finalizing"]] == [
        "task.evaluation_snapshot",
        "env.release_inputs",
    ]
    assert run.env._termination_snapshot_captured
    assert run.env._termination == {
        "stop_reason": "invalid_run",
        "end_s": at,
        "snapshot_s": at,
        "snapshot_delay_s": 0.0,
    }
    logged = [call["line"] for call in run.env.calls if call["call"] == "runner.log"]
    assert any(
        line.startswith("[default] decision loop raised:") and line.rstrip().endswith(raised)
        for line in logged
    ), logged


def test_environment_exception_is_finalized(tmp_path):
    """Decision 1's wait for the next frame fails at 1.5 s, latched and raised by the
    environment: the run keeps the environment's reason."""
    run = _run(
        tmp_path,
        [Step(CLICK_NAME), TYPE_ADA],
        [0.5, 0.5],
        stream=True,
        max_minutes=SHORT_MAX_MINUTES,
        advance_failures={1: "page clock did not advance"},
    )
    assert run.env._invalid_run == {
        "category": "environment",
        "reason": "clock_advance_failed",
        "detail": "page clock did not advance",
    }
    # The wait comes before the commit (runner.py:628, 694), so decision 1 is never committed.
    assert [status for _, status, _ in commit_statuses(run.env)] == ["applied"]
    _check_finalized(run, at=1.5, raised="RuntimeError: page clock did not advance")


def test_agent_exception_is_finalized(tmp_path):
    """The agent raises as decision 1 binds its budget at 1.0 s (runner.py:953), before anything
    is latched: the runner invalidates the run itself."""
    configure_decision = DefaultHarness.configure_decision
    budgets = []

    def failing_configure_decision(agent, budget):
        budgets.append(budget)
        if len(budgets) == 2:
            raise RuntimeError("budget rejected")
        return configure_decision(agent, budget)

    with mock.patch.object(DefaultHarness, "configure_decision", failing_configure_decision):
        run = _run(
            tmp_path,
            [Step(CLICK_NAME)],
            [0.5],
            stream=True,
            max_minutes=SHORT_MAX_MINUTES,
        )
    invalid = run.env._invalid_run
    assert (invalid["category"], invalid["reason"]) == ("runner", "run_exception")
    assert invalid["detail"].rstrip().endswith("RuntimeError: budget rejected")
    assert [status for _, status, _ in commit_statuses(run.env)] == ["applied"]
    _check_finalized(run, at=1.0, raised="RuntimeError: budget rejected")


# After run_stream returns, main() still saves the artifact, drains abandoned workers, closes the
# agent and evaluates before it writes the summary (runner.py:1798-1928).
TEARDOWN_S = 2.0


@pytest.mark.parametrize(
    ("reason", "solved", "outcome"),
    [("deadline_expired", False, "fail"), ("task_success", True, "pass")],
)
def test_early_end_with_teardown_past_the_deadline(tmp_path, reason, solved, outcome):
    """The task ends the episode when the third action lands at 2.5 s, and teardown runs on past
    the 3.75 s deadline. The report reads the termination record: unsolved is a fail, not a
    timeout, and solved took 2.5 s of the horizon, without the teardown."""
    run = _run(
        tmp_path,
        [Step(CLICK_NAME), TYPE_ADA, Step(CLICK_SUBMIT)],
        [0.5, 0.5, 0.5],
        stream=True,
        max_minutes=SHORT_MAX_MINUTES,
        stop_check=_decided_after(3, reason),
    )
    run.env.clock.advance(TEARDOWN_S)
    assert run.env.past_deadline()
    termination = run.env._termination
    assert termination == {
        "stop_reason": reason,
        "end_s": 2.5,
        "snapshot_s": 2.5,
        "snapshot_delay_s": 0.0,
    }
    # The fields of run_summary.json that main() takes from the loop (runner.py:2054-2061).
    summary = {
        "invalid_run": None,
        "solved": solved,
        "stop_reason": run.env._stop_reason,
        "termination": termination,
        "world_time_s": termination["end_s"],
        "max_minutes": SHORT_MAX_MINUTES,
    }
    assert episode_outcome(summary)["outcome"] == outcome
    # The clock read after teardown is past the horizon; the outcome never reads a clock.
    late = {**summary, "world_time_s": run.env.tick_index() / run.env.fps}
    assert late["world_time_s"] >= SHORT_MAX_MINUTES * 60
    assert episode_outcome(late)["outcome"] == outcome
    lifecycle = tmp_path / "action_lifecycle.jsonl"
    lifecycle.write_text(
        "".join(
            json.dumps(call["record"]) + "\n"
            for call in run.env.calls
            if call["call"] == "env.record_lifecycle"
        )
    )
    metrics = summarize_run_metrics(
        lifecycle,
        total_decisions=len(run.timeline),
        solved=solved,
        clock_mode="wall",
        tokens_per_frame=None,
        fps=run.env.fps,
        termination=termination,
        horizon_s=SHORT_MAX_MINUTES * 60,
    )
    assert metrics["normalized_time_to_success"] == (round(2.5 / 3.75, 9) if solved else None)


def test_snapshot_time_is_when_its_read_returns(tmp_path):
    """The task ends the episode at 2.5 s and reading its state for the snapshot takes 1.5 s,
    past the 3.75 s deadline: the record keeps the end at 2.5 s and the snapshot at 4.0 s, 1.5 s
    late, and the task's reason still stops the episode."""
    run = _run(
        tmp_path,
        [Step(CLICK_NAME), TYPE_ADA, Step(CLICK_SUBMIT)],
        [0.5, 0.5, 0.5],
        stream=True,
        max_minutes=SHORT_MAX_MINUTES,
        stop_check=_decided_after(3, "task_success"),
        snapshot_seconds=1.5,
    )
    assert run.env._termination == {
        "stop_reason": "task_success",
        "end_s": 2.5,
        "snapshot_s": 4.0,
        "snapshot_delay_s": 1.5,
    }
    assert run.env._termination_snapshot["environment_tick"] == 4
