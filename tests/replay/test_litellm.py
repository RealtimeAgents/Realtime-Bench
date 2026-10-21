"""Replay baseline for the generic litellm API: wall clock only, non-streaming.

The model is set up the way a ``--model provider/model-id`` run does it: the Harbor trial worker
puts the endpoint into BENCHMARK_LITELLM_* (evaluation/harbor.py:443-479;
evaluation/_trial.py:111-128), the catalog's bridge key ``litellm-vlm`` reads them when the
catalog loads (providers/registry.py:500-505), and the runner builds the config from that key
(evaluation/runner.py:1430-1439, 1548-1553).

What this API's driver class (providers/litellm.py DirectLiteLLMDriver) and context policy
(entity/context_policy.py, the standard values) change against chat_completions:
- one complete JSON response per request (providers/litellm.py _start_complete_response), and
  only on the wall clock (providers/preflight.py check_wall_clock);
- the image detail stays in the request (no strip_image_detail; providers/litellm.py
  chat_messages);
- no physical context check runs (pressure_archive is off; harnesses/default/context.py
  _run_turn), so the vLLM counter's /tokenize call never happens on this API.

Since context policy version 2 each reply comes back in the next request rebuilt from the model
output, as on the other stateless APIs (harnesses/default/request.py _turn_items): without
LiteLLM's provider_specific_fields or the reply's reasoning, and not at all when the reply was
empty. Version 1 replayed LiteLLM's own dump of the response message.

Record the baseline with
``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay/test_litellm.py``; without it the run is
compared with tests/replay/baselines/litellm-nonstream-long.jsonl.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER
from realtime_framework.entity.model import FOLDED_VISUAL_PLACEHOLDER
from realtime_framework.providers import registry as catalog

from . import baseline
from .build import build_episode
from .frames import ReplayTask
from .scenario import LONG_SCENARIO, TASK_INSTRUCTION, replies
from .steps import run_steps
from .stub_litellm import LiteLLMStub

BASELINE = Path(__file__).parent / "baselines" / "litellm-nonstream-long.jsonl"
MODEL_KEY = "litellm-vlm"  # the runner records the catalog key as the model (runner.py:1244-1252)
LITELLM_MODEL = "openai/replay-vlm"  # no bundled native limit: no max_tokens (registry.py:142-156)
# A model LiteLLM's bundled map lists: its max_output_tokens becomes the native limit
# (registry.py:159-168, 507-508), which every request sends as max_tokens when no token clock
# bounds it (build.py:63-79). The value is the locked LiteLLM's, so an upgrade that changes the
# map row fails test_mapped_model_output_limit. The deepseek-v4.1-flash preset declares its own.
MAPPED_MODEL = "openrouter/deepseek/deepseek-v4.1-flash"
MAPPED_OUTPUT_LIMIT = 943718
API_KEY_ENV = "REPLAY_LITELLM_API_KEY"
API_KEY = "replay-dummy-key"
REASONING_EFFORT = "low"  # as --reasoning-effort low; sent top-level (providers/litellm.py:578-582)

# With the default policies (entity/config.py:35-38) a request holding 21 screenshots folds the
# oldest 10 (harnesses/default/context.py:24-32), and one holding 51 turns archives the oldest 10
# into the ledger that becomes messages[1] (context.py:295-305; harnesses/default/request.py:53-55;
# providers/litellm.py chat_messages). Decision k carries k earlier turns plus its own frame.
FIRST_FOLD, FOLDED = 20, 10
FIRST_ARCHIVE, ARCHIVED = 50, 10

# Inherited settings that change this API's requests, all read when a request is built or sent.
CLEARED_PREFIXES = (
    # LiteLLM's OpenAI handler reads OPENAI_ORGANIZATION (litellm/main.py:2624-2629); the client
    # it builds turns OPENAI_ORG_ID, OPENAI_PROJECT_ID and OPENAI_CUSTOM_HEADERS into request
    # headers (openai/_client.py:842-869).
    "OPENAI_",
)
CLEARED = (
    "BENCHMARK_MODEL_REVISION",  # the bridge key's model_revision (providers/registry.py:612-613)
    "EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER",  # another OpenAI handler (litellm/main.py:2660)
    # httpx in place of LiteLLM's aiohttp transport (litellm/llms/custom_httpx/http_handler.py:
    # 1230-1259)
    "DISABLE_AIOHTTP_TRANSPORT",
    "LITELLM_HTTP2",
    "STREAM_MIN_REQUEST_INTERVAL",  # a sleep before each request (providers/litellm.py:69-78, 914)
)


@pytest.fixture
def litellm_environment(monkeypatch):
    """Clear or pin, for this test only, the process settings this API's requests depend on."""
    for name in [name for name in os.environ if name.startswith(CLEARED_PREFIXES)] + list(CLEARED):
        monkeypatch.delenv(name, raising=False)
    # The aiohttp transport looks up a proxy for each request, from the environment or the system
    # settings, unless the host is in NO_PROXY (litellm/llms/custom_httpx/aiohttp_transport.py:
    # 414-415, 466-502).
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    import litellm

    # Read when litellm is imported (litellm/__init__.py:268-269), possibly by an earlier test;
    # when set it sends every openai/ request to the Responses API (litellm/main.py:1102-1104).
    monkeypatch.setattr(litellm, "route_all_chat_openai_to_responses", False)
    yield monkeypatch
    # Drop the catalog entries built from this test's BENCHMARK_LITELLM_* (registry.py:514-517).
    catalog._catalog_entries.cache_clear()


def litellm_vlm_config(monkeypatch, base_url: str, model: str = LITELLM_MODEL):
    """What the runner builds for ``rtbench --model <model> --api-base <stub>
    --api-key-env REPLAY_LITELLM_API_KEY --reasoning-effort low``."""
    monkeypatch.setenv("BENCHMARK_LITELLM_MODEL", model)
    monkeypatch.setenv("BENCHMARK_LITELLM_API_BASE", base_url)
    monkeypatch.setenv("BENCHMARK_LITELLM_KEY_ENV", API_KEY_ENV)
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    # The bridge key reads BENCHMARK_LITELLM_* when the catalog first loads, and the catalog is
    # cached (providers/registry.py:500-505, 514-517).
    catalog._catalog_entries.cache_clear()
    catalog.validate_reasoning_effort(MODEL_KEY, REASONING_EFFORT)  # runner.py:1433-1439
    return catalog.to_model_config(MODEL_KEY, fps=1.0, reasoning_effort=REASONING_EFFORT)


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
    assert _messages(records[FIRST_ARCHIVE])[1] == {"role": "user", "content": ledger}
    assert (
        records[FIRST_ARCHIVE]["result"]["realtime_runtime"]["history_archived_turns"] == ARCHIVED
    )


def _check_rejection_feedback(records: list[dict]) -> None:
    """A rejected action's detail reaches the model as its own message in the next request
    (harnesses/default/request.py:98-99; providers/litellm.py:364-365; harness.py:1612-1622)."""
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


def _check_assistant_replay(records: list[dict]) -> None:
    """In request k + 1, decision k's frame is followed by its reply rebuilt from the model output,
    or, after an empty reply, directly by its execution feedback."""
    for k, step in enumerate(LONG_SCENARIO[:-1]):
        messages = _messages(records[k + 1])
        frame = max(i for i, m in enumerate(messages[:-1]) if isinstance(m["content"], list))
        following = messages[frame + 1]
        if step.reply:
            assert following == {"role": "assistant", "content": step.reply}, f"decision {k}"
        else:
            assert following["role"] == "user", f"decision {k}"
            assert following["content"].startswith("EXECUTION RESULT: "), f"decision {k}"
    assert any(step.reasoning for step in LONG_SCENARIO[:-1])


def test_long_scenario(litellm_environment, tmp_path):
    with LiteLLMStub(replies(LONG_SCENARIO)) as stub:
        config = litellm_vlm_config(litellm_environment, stub.base_url)
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
    for record, headers in zip(records, stub.request_headers, strict=True):
        record["requests"][0]["headers"] = headers
    assert all(r["requests"][0]["headers"]["authorization"] == f"Bearer {API_KEY}" for r in records)
    assert all(r["result"]["inspection"]["valid"] for r in records)
    _check_history_changes(records)
    _check_rejection_feedback(records)
    _check_assistant_replay(records)

    header = {
        "record": "header",
        "api": "litellm",
        "stream": False,
        "scenario": "long",
        "config": config.model_dump(),
    }
    actual = baseline.Normalizer()([header, *records, *baseline.trajectory(tmp_path)])
    if os.environ.get("REPLAY_RECORD") == "1":
        baseline.write(BASELINE, actual)
        return
    problems = baseline.compare(baseline.read(BASELINE), actual)
    assert not problems, f"{len(problems)} difference(s) from {BASELINE.name}:\n" + "\n".join(
        problems[:60]
    )


def test_mapped_model_output_limit(litellm_environment, tmp_path):
    steps = LONG_SCENARIO[:2]
    with LiteLLMStub(replies(steps)) as stub:
        config = litellm_vlm_config(litellm_environment, stub.base_url, MAPPED_MODEL)
        episode = build_episode(
            config, tmp_path, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key=MODEL_KEY
        )
        try:
            records = run_steps(episode, steps, stub)
        finally:
            episode.agent.close()
    assert stub.unscripted == 0, "the agent sent requests beyond the script"
    assert stub.unused_replies == 0, "scripted replies were left unused"
    assert (config.native_output_limit, config.native_output_limit_source) == (
        MAPPED_OUTPUT_LIMIT,
        "litellm",
    )
    caps = [r["requests"][0]["body"]["max_tokens"] for r in records]
    assert caps == [MAPPED_OUTPUT_LIMIT] * len(steps)
