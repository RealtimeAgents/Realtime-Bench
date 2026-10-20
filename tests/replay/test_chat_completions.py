"""Replay baseline for the chat_completions API, streaming and non-streaming.

Record a baseline with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it the
run is compared with tests/replay/baselines/.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER
from realtime_framework.entity.model import FOLDED_VISUAL_PLACEHOLDER

from . import baseline
from .build import build_episode
from .frames import ReplayTask
from .scenario import LONG_SCENARIO, TASK_INSTRUCTION, replies
from .steps import run_steps
from .stub import ChatCompletionsStub

BASELINES = Path(__file__).parent / "baselines"
MODEL_KEY = "replay-model"

# With the default policies (entity/config.py:35-38) a request holding 21 screenshots folds the
# oldest 10 (harnesses/default/context.py:27-36), and one holding 51 turns archives the oldest 10
# (context.py:262-272). Decision k carries k earlier turns plus its own frame.
FIRST_FOLD, FOLDED = 20, 10
FIRST_ARCHIVE, ARCHIVED = 50, 10


def chat_completions_config(base_url: str, *, stream: bool) -> ModelConfig:
    """What catalog.to_model_config(key, fps=1.0) returns (providers/registry.py:609-663) for

        endpoint: {api: chat_completions, model: replay-vlm, base_url: <stub>,
                   api_key_env: REPLAY_STUB_API_KEY, stream: <stream>, include_usage: <stream>,
                   native_output_limit: 4096}
        reasoning_effort: low

    The key variable is never set; this API then sends a placeholder key
    (providers/litellm.py:496-505).
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
            episode.agent.close()
    assert stub.unscripted == 0, "the agent sent requests beyond the script"
    assert stub.unused_replies == 0, "scripted replies were left unused"
    assert all(len(r["requests"]) == 1 for r in records)
    assert all(r["result"]["inspection"]["valid"] for r in records)
    _check_history_changes(records)
    _check_rejection_feedback(records)

    header = {
        "record": "header",
        "api": "chat_completions",
        "stream": stream,
        "scenario": "long",
        "config": config.model_dump(),
    }
    actual = baseline.Normalizer()([header, *records])
    path = BASELINES / f"chat_completions-{'stream' if stream else 'nonstream'}-long.jsonl"
    if os.environ.get("REPLAY_RECORD") == "1":
        baseline.write(path, actual)
        return
    problems = baseline.compare(baseline.read(path), actual)
    assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
        problems[:60]
    )
