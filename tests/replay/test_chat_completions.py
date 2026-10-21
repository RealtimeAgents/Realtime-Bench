"""Replay baseline for the chat_completions API, streaming and non-streaming.

Record a baseline with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it the
run is compared with tests/replay/baselines/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import LEDGER_OMISSION
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER
from realtime_framework.entity.model import FOLDED_VISUAL_PLACEHOLDER
from realtime_framework.entity.model import TRANSIENT
from realtime_framework.entity.model import TransportFailure
from realtime_framework.entity.records import validate_episode_trace
from realtime_framework.evaluation.agent_trace import load_body
from realtime_framework.evaluation.clock import token_record

from . import baseline
from .build import build_episode
from .frames import ReplayTask
from .scenario import LONG_SCENARIO, OUTSIDE_PAGE, TASK_INSTRUCTION, replies
from .steps import run_steps
from .stub import ChatCompletionsStub

BASELINES = Path(__file__).parent / "baselines"
MODEL_KEY = "replay-model"

# With the default policies (entity/config.py:35-38) a request holding 21 screenshots folds the
# oldest 10 (harnesses/default/context.py:24-32), and one holding 51 turns archives the oldest 10
# (context.py:295-305). Decision k carries k earlier turns plus its own frame.
FIRST_FOLD, FOLDED = 20, 10
FIRST_ARCHIVE, ARCHIVED = 50, 10


def chat_completions_config(base_url: str, *, stream: bool) -> ModelConfig:
    """What catalog.to_model_config(key, fps=1.0) returns (providers/registry.py:593-647) for

        endpoint: {api: chat_completions, model: replay-vlm, base_url: <stub>,
                   api_key_env: REPLAY_STUB_API_KEY, stream: <stream>, include_usage: <stream>,
                   native_output_limit: 4096}
        reasoning_effort: low

    The key variable is never set; this API then sends a placeholder key
    (providers/litellm.py:494-503).
    """
    return ModelConfig(
        adapter_type="chat_completions",
        model_name="replay-vlm",
        api_base_url=base_url,
        api_key_env="REPLAY_STUB_API_KEY",
        chat_stream=stream,
        chat_include_usage=stream,
        native_output_limit=4096,
        native_output_limit_source="endpoint",
        reasoning_effort="low",
        fps=1.0,
    )


def _messages(record: dict) -> list[dict]:
    (request,) = record["requests"]
    return request["body"]["messages"]


def _check_history_changes(records: list[dict]) -> None:
    folds = [sum(m["content"] == FOLDED_VISUAL_PLACEHOLDER for m in _messages(r)) for r in records]
    ledgers = [
        [
            m["content"]
            for m in _messages(r)
            if isinstance(m["content"], str)
            and m["content"].startswith(ARCHIVED_ACTION_LEDGER_HEADER)
        ]
        for r in records
    ]
    assert next(k for k, folded in enumerate(folds) if folded) == FIRST_FOLD
    assert folds[FIRST_FOLD] == FOLDED
    runtime = records[FIRST_FOLD]["result"]["realtime_runtime"]
    assert runtime["history_folded_visuals"] == FOLDED
    assert next(k for k, ledger in enumerate(ledgers) if ledger) == FIRST_ARCHIVE
    (ledger,) = ledgers[FIRST_ARCHIVE]
    assert len(ledger.splitlines()) == 1 + ARCHIVED
    assert _messages(records[FIRST_ARCHIVE])[1]["content"] == ledger
    assert (
        records[FIRST_ARCHIVE]["result"]["realtime_runtime"]["history_archived_turns"] == ARCHIVED
    )


def _check_rejection_feedback(records: list[dict]) -> None:
    """A rejected action's detail reaches the model as its own message in the next request."""
    rejected = [k for k, r in enumerate(records) if r["commit"]["detail"]]
    assert rejected
    for k in rejected:
        detail = records[k]["commit"]["detail"]
        feedback = [
            m["content"]
            for m in _messages(records[k + 1])
            if m["role"] == "user" and isinstance(m["content"], str) and detail in m["content"]
        ]
        assert len(feedback) == 1
        assert feedback[0].startswith("EXECUTION RESULT: ") and "status=rejected." in feedback[0]


def _check_agent_trace(run_dir: Path, stub: ChatCompletionsStub, records: list[dict]) -> None:
    """The agent's records: one request per stub request, exactly as received, and the history
    changes at the first fold (decision 20, the 21st step) and the first archive (decision 50)."""
    validate_episode_trace(run_dir / "episode_trace.jsonl")
    text = (run_dir / "episode_trace.jsonl").read_text()
    # Bodies live only in the pack: the trace is exported, and requests carry task content.
    for private in (TASK_INSTRUCTION, "I agree to the terms", "EXECUTION RESULT", OUTSIDE_PAGE):
        assert private not in text
    events = [json.loads(line) for line in text.splitlines()]
    assert events[-1]["event"] == "AgentTraceClosed"
    closed = events[-1]["payload"]
    assert closed["written"] == closed["captures"] and closed["error"] is None
    sent = [e["payload"] for e in events if e["event"] == "ModelRequestSent"]
    assert len(sent) == len(stub.requests)
    for k, (request, received) in enumerate(zip(sent, stub.requests)):
        assert request["channel"] == "litellm.openai_sdk" and request["exact_wire"]
        assert request["scope"]["decision"] == k and request["scope"]["attempt"] == 1
        assert load_body(run_dir / "episode_blobs.pack", request["body"]) == received.body
    attempts = [e["payload"] for e in events if e["event"] == "ModelAttemptFinished"]
    assert [a["outcome"] for a in attempts] == ["ok"] * len(records)

    def turns(k: int, kind: str, policy_item: str) -> list[str]:
        (change,) = [
            e["payload"]
            for e in events
            if e["event"] == "HistoryChanged"
            and e["payload"]["scope"]["decision"] == k
            and e["payload"]["kind"] == kind
        ]
        assert change["policy_item"] == policy_item and change["trigger"] == "policy"
        return [turn["observation_id"] for turn in change["turns"]]

    observation_ids = [r["observation"]["observation_id"] for r in records]
    assert turns(FIRST_FOLD, "visual_fold", "visuals") == observation_ids[:FOLDED]
    assert turns(FIRST_ARCHIVE, "archive", "turns") == observation_ids[:ARCHIVED]
    first_changes = {
        e["payload"]["scope"]["decision"] for e in events if e["event"] == "HistoryChanged"
    }
    assert min(first_changes) == FIRST_FOLD


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
def test_long_scenario(stream, tmp_path):
    with ChatCompletionsStub(replies(LONG_SCENARIO)) as stub:
        config = chat_completions_config(stub.base_url, stream=stream)
        episode = build_episode(
            config, tmp_path, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key=MODEL_KEY
        )
        try:
            records = run_steps(episode, LONG_SCENARIO, stub)
        finally:
            invalid_run = episode.finish()
    assert stub.unscripted == 0, "the agent sent requests beyond the script"
    assert stub.unused_replies == 0, "scripted replies were left unused"
    assert all(len(r["requests"]) == 1 for r in records)
    assert all(r["result"]["inspection"]["valid"] for r in records)
    _check_history_changes(records)
    _check_rejection_feedback(records)
    assert invalid_run is None
    _check_agent_trace(tmp_path, stub, records)

    header = {
        "record": "header",
        "api": "chat_completions",
        "stream": stream,
        "scenario": "long",
        "config": config.model_dump(),
    }
    actual = baseline.Normalizer()([header, *records, *baseline.trajectory(tmp_path)])
    path = BASELINES / f"chat_completions-{'stream' if stream else 'nonstream'}-long.jsonl"
    if os.environ.get("REPLAY_RECORD") == "1":
        baseline.write(path, actual)
        return
    problems = baseline.compare(baseline.read(path), actual)
    assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
        problems[:60]
    )


# (context_window_tokens, context_reserve_tokens) that hold the system text and the current frame
# with its image allowance (providers/context.py estimated_request_tokens), plus few archived
# actions: from decision 1 the runtime archives every earlier turn to fit, and from decision 16
# the ledger alone outgrows the window.
LEDGER_WINDOW = (16096, 4096)
LEDGER_OMISSIONS = (16, 26, 36, 46, 56)  # the decisions that leave out the next ten actions


def test_history_overflow_ledger(tmp_path):
    """As for JEV (test_decisions.py test_history_overflow_ledger): with every turn archived and
    the request still over the window, the ledger message leaves out its earliest actions, ten at
    a time, behind a note, and every decision sends its request. Without the note, decision 16
    was invalid without a request."""
    with ChatCompletionsStub(replies(LONG_SCENARIO)) as stub:
        config = chat_completions_config(stub.base_url, stream=False).model_copy(
            update=dict(zip(("context_window_tokens", "context_reserve_tokens"), LEDGER_WINDOW))
        )
        episode = build_episode(
            config, tmp_path, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key=MODEL_KEY
        )
        try:
            records = run_steps(episode, LONG_SCENARIO, stub)
        finally:
            invalid_run = episode.finish()
    assert invalid_run is None
    assert all(len(r["requests"]) == 1 for r in records)
    assert not any(r["result"].get("invalid_reason") for r in records)
    runtimes = [r["result"]["realtime_runtime"] for r in records]
    assert [runtime["history_detailed_turns"] for runtime in runtimes] == [0] * len(records)
    assert not any(runtime["request_context"]["pressure"] for runtime in runtimes)
    trace = (tmp_path / "episode_trace.jsonl").read_text()
    events = [json.loads(line) for line in trace.splitlines()]
    omissions = [
        e["payload"]
        for e in events
        if e["event"] == "HistoryChanged" and e["payload"]["kind"] == "ledger_omission"
    ]
    assert [o["scope"]["decision"] for o in omissions] == list(LEDGER_OMISSIONS)
    assert [o["ledger"]["omitted_actions_after"] for o in omissions] == [10, 20, 30, 40, 50]
    first = LEDGER_OMISSIONS[0]
    totals = [runtime.get("history_omitted_actions_total") for runtime in runtimes]
    assert totals == [None] * first + [
        10 * sum(k >= decision for decision in LEDGER_OMISSIONS)
        for k in range(first, len(records))
    ]
    omitted_now = [runtime.get("request_context_omitted_actions") for runtime in runtimes]
    assert {k: n for k, n in enumerate(omitted_now) if n} == dict.fromkeys(LEDGER_OMISSIONS, 10)
    header_line, *lines = _messages(records[first - 1])[1]["content"].split("\n")
    assert header_line == ARCHIVED_ACTION_LEDGER_HEADER and len(lines) == first - 1
    for k in range(first, len(records)):
        header_line, note, *kept = _messages(records[k])[1]["content"].split("\n")
        omitted = 10 * sum(k >= decision for decision in LEDGER_OMISSIONS)
        assert header_line == ARCHIVED_ACTION_LEDGER_HEADER
        assert note == LEDGER_OMISSION.format(count=omitted)
        assert len(kept) == k - omitted and all(line.startswith("{") for line in kept)

    header = {
        "record": "header",
        "api": "chat_completions",
        "stream": False,
        "scenario": "history_overflow_ledger",
        "config": config.model_dump(),
    }
    actual = baseline.Normalizer()([header, *records, *baseline.trajectory(tmp_path)])
    path = BASELINES / "chat_completions-nonstream-history_overflow_ledger.jsonl"
    if os.environ.get("REPLAY_RECORD") == "1":
        baseline.write(path, actual)
        return
    problems = baseline.compare(baseline.read(path), actual)
    assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
        problems[:60]
    )


def test_context_exceeded_before_request(tmp_path, monkeypatch):
    """A chat request that does not fit the window even with nothing retained is not sent, and
    the decision is invalid as context_exceeded_before_request. The harness raised it, so the
    driver's classifier is not asked: even one that calls every error transient gets no retry."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        # Below the first request's estimate of 9459 (test_history_overflow_ledger).
        config = chat_completions_config(stub.base_url, stream=False).model_copy(
            update={"context_window_tokens": 6096, "context_reserve_tokens": 1000}
        )
        episode = build_episode(
            config, tmp_path, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key=MODEL_KEY
        )
        monkeypatch.setattr(
            episode.agent.driver,
            "classify_transport_error",
            lambda exc: TransportFailure(TRANSIENT),
        )
        try:
            observation = episode.frames.capture(0)
            episode.agent.configure_decision(episode.budget(observation.tick_number))
            episode.agent.stream_prefill(observation)
            result = episode.agent.stream_generate(tick=0)
        finally:
            episode.finish()
    assert stub.requests == [] and stub.unused_replies == 1
    assert (result["invalid_category"], result["invalid_reason"]) == (
        "provider",
        "context_exceeded_before_request",
    )
    # Its token record says why it has no usage (evaluation/clock.py token_record).
    assert token_record(result)["provider_usage_missing_reason"] == "request_not_sent"
    trace = (tmp_path / "episode_trace.jsonl").read_text()
    attempts = [
        e["payload"]
        for e in map(json.loads, trace.splitlines())
        if e["event"] == "ModelAttemptFinished"
    ]
    assert [(a["outcome"], a["invalid_reason"]) for a in attempts] == [
        ("failed", "context_exceeded_before_request")
    ]
