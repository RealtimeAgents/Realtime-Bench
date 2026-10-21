"""Replay baselines for run_stream's own decision loop on the wall clock.

The real run_stream, the event kernel's shell (evaluation/shell.py), drives the chat_completions
agent from build_episode against FakeEnvironment (fake_env.py), all on one fake clock
(tests/fake_clock.py): it moves only when the shell, its workers, the agent's event loop and the
stub all wait, so each episode runs the same way every time. Each baseline records, decision by
decision, the shell's calls into the environment, its trace events (the kernel's records among
them), the commit statuses the agent receives, the action lifecycle records and the runner log,
then the stop reason and the agent's history. Record with
``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it each run is compared with
tests/replay/baselines/.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import pytest

from realtime_framework.evaluation.reporting import episode_outcome, summarize_run_metrics
from realtime_framework.harnesses.default.harness import DefaultHarness

from ..fake_clock import FakeClock
from . import baseline
from .build import build_episode
from .fake_env import FakeEnvironment, Hang
from .frames import CLOCK_ORIGIN, ReplayTask
from .loop import (
    REAL_TIME_FIELDS,
    check_wall_clock_properties,
    commit_statuses,
    records,
    resolved,
    run_loop,
)
from .scenario import CLICK_AGREE, CLICK_NAME, CLICK_SUBMIT, LONG_SCENARIO, TASK_INSTRUCTION
from .scenario import replies
from .steps import Step
from .stub import ChatCompletionsStub
from .test_chat_completions import MODEL_KEY, chat_completions_config

BASELINES = Path(__file__).parent / "baselines"

# Most replies take half a frame. Decisions 12 and 40 span frame boundaries and decision 21 lands
# exactly on one; LONG_SCENARIO's waits (decisions 5, 33, 56) add their durations. Decision 59
# starts at 63.0 s and would reply at 63.875 s, after the 63.75 s deadline (LATE_REPLY_S).
LONG_LATENCIES = [
    {3: 0.25, 8: 0.75, 12: 2.5, 21: 1.0, 26: 0.25, 40: 1.75, 47: 0.75, 59: 0.875}.get(k, 0.5)
    for k in range(len(LONG_SCENARIO))
]
LONG_MAX_MINUTES = 1.0625
# Short cases: a 3.75 s episode at 1 fps, with decisions at 0, 1, 2 and 3 s.
SHORT_MAX_MINUTES = 0.0625
# A reply to a decision at 3.0 s that would land after the 3.75 s deadline but before tick 4: the
# kernel gives the generation up at the deadline (kernel/turns.py on_timer).
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
    clock = FakeClock(wall_origin=CLOCK_ORIGIN)
    env_box = []
    with (
        ChatCompletionsStub(
            replies(steps), on_request=lambda index: env_box[0].model_request(index), clock=clock
        ) as stub,
        clock.participant(),
    ):
        config = chat_completions_config(stub.base_url, stream=stream)
        episode = build_episode(
            config,
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=MODEL_KEY,
            clock_mode="wall",
            clock=clock,
        )
        env = FakeEnvironment(
            episode.frames,
            clock,
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
    # run_stream reports the statuses and details steps.py scripts for the same replies, but for
    # the last decision's: the kernel gives its generation up at the deadline, unreported.
    assert commit_statuses(long_run.env) == [
        (k, step.status, step.detail) for k, step in enumerate(LONG_SCENARIO[:-1])
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
    statuses: list[str]  # the status run_stream commits for each decision it reports
    executed: list[str]  # the action types the environment executes, in order
    stop_reason: str
    env: dict = field(default_factory=dict)
    requests: int | None = None  # when the agent sends more requests than there are steps
    held: tuple[bool, ...] = ()


# The kernel gives a generation up at the deadline, without grace (kernel/turns.py on_timer).
# Decision 3 starts at 3.0 s with 0.75 s left.
DEADLINE_WAIT_S = 0.75

CASES = [
    # The reply to decision 3 would land at 3.875 s. At the 3.75 s deadline the kernel gives the
    # generation up: nothing is executed and the decision is not reported.
    Case(
        "late_reply",
        [Step(CLICK_NAME), TYPE_ADA, Step(CLICK_SUBMIT), Step(CLICK_AGREE)],
        [0.5, 0.5, 0.5, LATE_REPLY_S],
        ["applied", "applied", "applied"],
        ["click", "type", "click"],
        "time_limit",
    ),
    # DONE becomes WAIT (kernel/turns.py _accept), whose time the kernel lets pass, and the
    # episode goes on to the deadline.
    Case(
        "done_reply",
        [Step(CLICK_NAME), Step('{"action": "done"}'), Step(CLICK_SUBMIT), Step(CLICK_AGREE)],
        [0.5, 0.5, 0.5, LATE_REPLY_S],
        ["applied", "wait", "applied"],
        ["click", "click"],
        "time_limit",
    ),
    # The third applied action decides the task, which the end check right after it finds: the
    # episode stops at once with the task's reason (kernel/turns.py _executed).
    Case(
        "environment_done",
        [Step(CLICK_NAME), TYPE_ADA, Step(CLICK_SUBMIT)],
        [0.5, 0.5, 0.5],
        ["applied", "applied", "applied"],
        ["click", "type", "click"],
        "task_completed",
        env={"stop_check": _decided_after(3, "task_completed")},
    ),
    # The task is decided at 2.25 s, while decision 2 waits for its reply. Frame 2 was read at
    # 2.0 s, so no read comes before the action (kernel/turns.py _check_end): it is executed at
    # 2.5 s, the read right after it sees the end, and the reason defaults to game_decided
    # (kernel/core.py _finish).
    Case(
        "environment_done_during_reply",
        [Step(CLICK_NAME), TYPE_ADA, Step(CLICK_SUBMIT)],
        [0.5, 0.5, 0.5],
        ["applied", "applied", "applied"],
        ["click", "type", "click"],
        "game_decided",
        env={"stop_check": lambda env: env.clock.time() >= env.t0 + 2.25},
    ),
    # Decision 3's reply never comes. At the deadline the kernel gives the generation up and the
    # agent is aborted (shell.py _abort); the decision is not reported. The abort closes the held
    # request and cancels the decision's generation, so the abandoned worker does not send it
    # again (DefaultHarness.abort; entity/model.py RequestRetry).
    Case(
        "bounded_call_timeout",
        [Step(CLICK_NAME), TYPE_ADA, Step(CLICK_SUBMIT)],
        [0.5, 0.5, 0.5, Hang(DEADLINE_WAIT_S)],
        ["applied", "applied", "applied"],
        ["click", "type", "click"],
        "time_limit",
        requests=4,
        held=(True,),
    ),
    # A fault that is not the guest refusing the action invalidates the run (shell.py _execute).
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


# An exception a command of the shell raises still ends the episode (shell.py _run): the run is
# invalid, the snapshot is frozen and the stop reason latched. These are property tests; an
# escaped exception's traceback names files and lines.


def _check_finalized(run: Run, *, at: float, raised: str) -> None:
    check_wall_clock_properties(run.env, "invalid_run")
    # Between them the kernel writes its records of the snapshot and the termination.
    assert [
        call["call"] for call in run.env.calls if call["finalizing"] and call["call"] != "trace"
    ] == [
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


def test_two_failed_state_reads_in_a_row_leave_the_run_valid(tmp_path):
    """The state is read at 0 s, after decision 0's action at 0.5 s, at frame 1 (1.0 s) and
    after decision 1's action at 1.5 s. The reads at 0.5 s and 1.0 s fail; the one at 1.5 s
    answers and starts the count over. A failed read short of three in a row answers "not over"
    (kernel/turns.py READ_FAILURE_LIMIT), and the episode runs to the deadline."""
    run = _run(
        tmp_path,
        [Step(CLICK_NAME), TYPE_ADA, Step(CLICK_SUBMIT), Step(CLICK_AGREE)],
        [0.5, 0.5, 0.5, LATE_REPLY_S],
        stream=True,
        max_minutes=SHORT_MAX_MINUTES,
        read_failures=frozenset({2, 3}),
    )
    check_wall_clock_properties(run.env, "time_limit")
    assert getattr(run.env, "_invalid_run", None) is None
    assert [status for _, status, _ in commit_statuses(run.env)] == ["applied"] * 3
    failed = [call["failed"] for call in run.env.calls if "failed" in call]
    assert failed == [2, 3]


def test_a_third_failed_state_read_in_a_row_invalidates_the_run(tmp_path):
    """The read after decision 1's action at 1.5 s fails too, the third in a row: the run is
    invalid with the environment's terminal_state_check_failed, and decision 1, whose read after
    the action the end cut short, is reported as applied."""
    run = _run(
        tmp_path,
        [Step(CLICK_NAME), TYPE_ADA],
        [0.5, 0.5],
        stream=True,
        max_minutes=SHORT_MAX_MINUTES,
        read_failures=frozenset({2, 3, 4}),
    )
    check_wall_clock_properties(run.env, "invalid_run")
    invalid = run.env._invalid_run
    assert (invalid["category"], invalid["reason"]) == (
        "environment",
        "terminal_state_check_failed",
    )
    assert invalid["detail"].rstrip().endswith("RuntimeError: state read 4 failed")
    assert [status for _, status, _ in commit_statuses(run.env)] == ["applied"] * 2
    assert run.env._termination["end_s"] == 1.5


def test_agent_exception_is_finalized(tmp_path):
    """The agent raises as the shell binds decision 1's budget at 1.0 s, on the shell's thread
    (shell.py _start_generation), before anything is latched: the shell invalidates the run
    itself. The generation it was starting is given up, so decision 1 is never committed."""
    transport_output_token_cap = DefaultHarness.transport_output_token_cap
    caps = []

    def failing_transport_output_token_cap(agent, provider_cap):
        caps.append(provider_cap)
        if len(caps) == 2:
            raise RuntimeError("budget rejected")
        return transport_output_token_cap(agent, provider_cap)

    with mock.patch.object(
        DefaultHarness, "transport_output_token_cap", failing_transport_output_token_cap
    ):
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
# agent and evaluates before it writes the summary (runner.py:1831-1962).
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
    with run.env.clock.participant():
        run.env.clock.sleep(TEARDOWN_S)
    assert run.env.past_deadline()
    termination = run.env._termination
    assert termination == {
        "stop_reason": reason,
        "end_s": 2.5,
        "snapshot_s": 2.5,
        "snapshot_delay_s": 0.0,
    }
    # The fields of run_summary.json that main() takes from the loop (runner.py:2088-2095).
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


def _failing_on_the_second_call(method):
    calls = []

    def failing(agent, *args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError(f"{method.__name__} failed")
        return method(agent, *args)

    return failing


def test_a_generation_that_raises_invalidates_the_run(tmp_path):
    """Decision 1's generation raises on the shell's worker, in the middle of the agent's turn
    (shell.py _generate): its result is a provider failure that leaves the decision without an
    action, and the run is invalid with the provider's generation_error
    (episode_records.result_invalid)."""
    failing = _failing_on_the_second_call(DefaultHarness._generate)
    with mock.patch.object(DefaultHarness, "_generate", failing):
        run = _run(tmp_path, [Step(CLICK_NAME)], [0.5], stream=True, max_minutes=SHORT_MAX_MINUTES)
    invalid = run.env._invalid_run
    assert (invalid["category"], invalid["reason"]) == ("provider", "generation_error")
    assert invalid["detail"].rstrip().endswith("RuntimeError: _generate failed")
    completed = [
        call["payload"]
        for call in run.env.calls
        if call["call"] == "trace" and call["event"] == "InferenceCompleted"
    ]
    assert [(p["decision"], p["error"], p["invalid_reason"]) for p in completed] == [
        (0, False, None),
        (1, True, "generation_error"),
    ]
    assert [status for _, status, _ in commit_statuses(run.env)] == ["applied"]
    check_wall_clock_properties(run.env, "invalid_run")


def test_a_failed_trajectory_write_invalidates_the_run(tmp_path):
    """Writing decision 1's trajectory row fails (episode_records.finalize_decision_audit): the
    result is unusable, the run is invalid with harness:trajectory_write_failed, and decision 1's
    proposed action is resolved as invalid_run without being executed."""
    failing = _failing_on_the_second_call(DefaultHarness.finalize_decision_audit)
    with mock.patch.object(DefaultHarness, "finalize_decision_audit", failing):
        run = _run(
            tmp_path,
            [Step(CLICK_NAME), TYPE_ADA],
            [0.5, 0.5],
            stream=True,
            max_minutes=SHORT_MAX_MINUTES,
        )
    invalid = run.env._invalid_run
    assert (invalid["category"], invalid["reason"]) == ("harness", "trajectory_write_failed")
    assert invalid["detail"].rstrip().endswith("RuntimeError: finalize_decision_audit failed")
    logged = [call["line"] for call in run.env.calls if call["call"] == "runner.log"]
    assert any(line.startswith("[default] trajectory audit write failed:") for line in logged)
    assert [p["status"] for p in resolved(run.env).values()] == ["applied", "invalid_run"]
    assert [e["action"]["action_type"] for e in run.env.executions] == ["click"]
    check_wall_clock_properties(run.env, "invalid_run")
