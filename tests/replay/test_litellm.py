"""Replay baseline for the generic litellm API: wall clock only, non-streaming.

The model is set up the way a ``--model provider/model-id`` run does it: the Harbor trial worker
puts the endpoint into BENCHMARK_LITELLM_* (evaluation/harbor.py:443-479;
evaluation/_trial.py:111-128), the catalog's bridge key ``litellm-vlm`` reads them when the
catalog loads (providers/registry.py:500-505), and the runner builds the config from that key
(evaluation/runner.py:1430-1439, 1548-1553).

What this API's driver class (providers/litellm.py:1202-1221) and context policy
(entity/context_policy.py:166-167) change against chat_completions:
- one complete JSON response per request (providers/litellm.py:1210, 665-666, 913-949), and only
  on the wall clock (providers/litellm.py:1203-1209; providers/preflight.py:49-50);
- the image detail stays in the request (no strip_image_detail; providers/litellm.py:346-350);
- each reply comes back in the next request as LiteLLM's own dump of the response message,
  provider_specific_fields included (providers/litellm.py:940, 947-948;
  harnesses/default/context.py:589; harnesses/default/request.py:87-89;
  providers/litellm.py:357-358). So an empty reply without reasoning still comes back, as an
  empty assistant message, where the other stateless APIs send none
  (harnesses/default/request.py:91). Reasoning replay is on too (entity/context_policy.py:166;
  harnesses/default/request.py:90), but the dump takes precedence (providers/litellm.py:356-363):
  the reasoning reaches the model as the dump's reasoning_content;
- no physical context check runs (entity/context_policy.py:150, 166; context.py:488-489), so the
  vLLM counter's /tokenize call never happens on this API.

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
LITELLM_MODEL = "openai/replay-vlm"  # no bundled native limit: no max_tokens (registry.py:140-154)
API_KEY_ENV = "REPLAY_LITELLM_API_KEY"
API_KEY = "replay-dummy-key"
REASONING_EFFORT = "low"  # as --reasoning-effort low; sent top-level (providers/litellm.py:578-582)

# With the default policies (entity/config.py:35-38) a request holding 21 screenshots folds the
# oldest 10 (harnesses/default/context.py:24-32), and one holding 51 turns archives the oldest 10
# into the ledger that becomes messages[1] (context.py:295-305; harnesses/default/request.py:53-55;
# providers/litellm.py:329-332). Decision k carries k earlier turns plus its own frame.
FIRST_FOLD, FOLDED = 20, 10
FIRST_ARCHIVE, ARCHIVED = 50, 10

# Inherited settings that change this API's requests, all read when a request is built or sent.
CLEARED_PREFIXES = (
    # LiteLLM's OpenAI handler reads OPENAI_ORGANIZATION (litellm/main.py:2594-2599); the client
    # it builds turns OPENAI_ORG_ID, OPENAI_PROJECT_ID and OPENAI_CUSTOM_HEADERS into request
    # headers (openai/_client.py:842-869).
    "OPENAI_",
)
CLEARED = (
    "BENCHMARK_MODEL_REVISION",  # the bridge key's model_revision (providers/registry.py:612-613)
    "EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER",  # another OpenAI handler (litellm/main.py:2625)
    # httpx in place of LiteLLM's aiohttp transport (litellm/llms/custom_httpx/http_handler.py:
    # 1213-1242)
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
    # 411-412, 463-500).
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    import litellm

    # Read when litellm is imported (litellm/__init__.py:263-264), possibly by an earlier test;
    # when set it sends every openai/ request to the Responses API (litellm/main.py:1072-1075).
    monkeypatch.setattr(litellm, "route_all_chat_openai_to_responses", False)
    yield monkeypatch
    # Drop the catalog entries built from this test's BENCHMARK_LITELLM_* (registry.py:514-517).
    catalog._catalog_entries.cache_clear()


def litellm_vlm_config(monkeypatch, base_url: str):
    """What the runner builds for ``rtbench --model openai/replay-vlm --api-base <stub>
    --api-key-env REPLAY_LITELLM_API_KEY --reasoning-effort low``."""
    monkeypatch.setenv("BENCHMARK_LITELLM_MODEL", LITELLM_MODEL)
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
    """Decision k's reply comes back in request k + 1 as LiteLLM's dump of the response message
    (providers/litellm.py:940, 947-948; context.py:589; harnesses/default/request.py:87-89;
    providers/litellm.py:357-358): with provider_specific_fields, with reasoning_content when the
    reply had reasoning, and with empty content when the reply was empty."""
    for k, step in enumerate(LONG_SCENARIO[:-1]):
        replayed = [m for m in _messages(records[k + 1]) if m["role"] == "assistant"][-1]
        expected = {
            "content": step.reply,
            "role": "assistant",
            "provider_specific_fields": {"refusal": None},
        }
        if step.reasoning:
            expected["reasoning_content"] = step.reasoning
        assert replayed == expected, f"decision {k}"


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
