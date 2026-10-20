"""Replay baselines for the moonshot API: the catalog cell kimi-k3 on the token clock.

Record a baseline with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it the
run is compared with tests/replay/baselines/.

The ModelConfig is the catalog's (runner.py:1433-1439, 1550-1555) and keeps the official base URL
(configs/models/kimi-k3.yaml:5), so the driver normalizes Kimi usage (providers/litellm.py:122-152,
906-908); stub_moonshot.redirect_to_stub sends the requests to the local stub. The long scenario
runs on a finite frame budget B, the short snapshot in pause mode (B unlimited).
"""

from __future__ import annotations

import os
from pathlib import Path

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER
from realtime_framework.entity.model import FOLDED_VISUAL_PLACEHOLDER
from realtime_framework.providers.litellm import DEFAULT_MOONSHOT_BASE_URL

from . import baseline
from .build import build_episode
from .frames import ReplayTask
from .scenario import LONG_SCENARIO, TASK_INSTRUCTION
from .steps import run_steps
from .stub_moonshot import KimiStub, compared_headers, kimi_replies, redirect_to_stub

BASELINES = Path(__file__).parent / "baselines"
MODEL_KEY = "kimi-k3"
FPS = 1.0
# The moonshot API requires a key (providers/litellm.py:494-500). Setting it also keeps the
# repository .env from supplying one: load_project_secrets only fills unset names
# (providers/registry.py:38).
DUMMY_KEY = "sk-replay-moonshot-dummy"
ENDPOINT = DEFAULT_MOONSHOT_BASE_URL + "/chat/completions"

# run_stream receives max_ticks = max_minutes * 60 * fps (runner.py:1775): a one-minute episode.
MAX_TICKS = 60
# Decision k observes tick k (steps.py). On the finite clock its cap is
# min(native, floor((MAX_TICKS - k) * B)) (runner.py:863-886). kimi-k3's native limit is 1048576
# (LiteLLM's bundled moonshot/kimi-k3, providers/registry.py:140-154), so with this B the native
# limit applies through decision 28 and the remaining clock from decision 29 (1015808) down to
# decision 59 (32768).
TOKENS_PER_FRAME = 32768
# As for chat_completions: the default policies (entity/config.py:35-38) fold the oldest 10
# screenshots of a request holding 21 (harnesses/default/context.py:24-32) and archive the oldest
# 10 turns of one holding 51 (context.py:295-305). Decision k carries k earlier turns.
FIRST_FOLD, FOLDED = 20, 10
FIRST_ARCHIVE, ARCHIVED = 50, 10
# Two reasoning replies (decisions 0 and 3), each replayed in a later request.
PAUSE_SCENARIO = LONG_SCENARIO[:5]


def _run(tmp_path, monkeypatch, steps, *, tokens_per_frame):
    monkeypatch.setenv("MOONSHOT_API_KEY", DUMMY_KEY)
    entry = catalog.get(MODEL_KEY)
    config = catalog.to_model_config(  # runner.py:1433-1439, 1550-1555
        MODEL_KEY, fps=FPS, reasoning_effort=entry.defaults["reasoning_effort"]
    )
    assert config.api_base_url == DEFAULT_MOONSHOT_BASE_URL
    if tokens_per_frame is not None:
        # runner.py:1455-1462
        assert catalog.adapter_class(config.adapter_type).capabilities.finite_frame_budget
    with KimiStub(kimi_replies(steps)) as stub:
        episode = build_episode(
            config,
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=MODEL_KEY,
            clock_mode="token",
            tokens_per_frame=tokens_per_frame,
            max_ticks=MAX_TICKS,
        )
        try:
            transport = redirect_to_stub(episode.agent.driver, stub.port)
            records = run_steps(episode, steps, stub)
        finally:
            episode.agent.close()
    assert stub.unscripted == 0, "the agent sent requests beyond the script"
    assert stub.unused_replies == 0, "scripted replies were left unused"
    # Every request was addressed to the official endpoint and answered by the stub.
    assert transport.urls == [ENDPOINT] * len(stub.requests)
    sent = iter(zip(transport.urls, stub.requests))
    for record in records:
        for request in record["requests"]:
            url, received = next(sent)
            request["url"] = url
            request["headers"] = compared_headers(received)
    assert all(len(r["requests"]) == 1 for r in records)
    assert all(r["result"]["inspection"]["valid"] for r in records)
    return config, records


def _body(record: dict) -> dict:
    (request,) = record["requests"]
    return request["body"]


def _messages(record: dict) -> list[dict]:
    return _body(record)["messages"]


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


def _check_reasoning_replay(records: list[dict], steps) -> None:
    """The moonshot policy keeps reasoning (entity/context_policy.py:170): the runtime stores it
    with the turn (harnesses/default/context.py:587) and the next request carries it on the turn's
    assistant message (harnesses/default/request.py:90-93, providers/litellm.py:356-363)."""
    reasoned = [k for k, step in enumerate(steps) if step.reasoning]
    assert reasoned
    for k in reasoned:
        # Turn k is the newest turn of request k + 1, so its assistant message comes last.
        *_, replayed = (m for m in _messages(records[k + 1]) if m["role"] == "assistant")
        assert replayed == {
            "role": "assistant",
            "content": steps[k].reply,
            "reasoning_content": steps[k].reasoning,
        }, f"request {k + 1}"


def _check_kimi_usage(records: list[dict]) -> None:
    """kimi_usage ran on every reply (providers/litellm.py:122-152): it adds the undivided text
    input, the cached and 5-minute cache-write splits and a zero audio output, which the token
    normalization then reports (entity/usage.py:77-78)."""
    for record in records:
        result = record["result"]
        usage = result["usage"]
        details = usage["prompt_tokens_details"]
        assert details["text_tokens"] == usage["prompt_tokens"]
        assert details["cached_tokens_details"] == {"text_tokens": details["cached_tokens"]}
        assert details["cache_write_tokens_details"] == {
            "ephemeral_5m_tokens": details["cache_write_tokens"],
            "ephemeral_1h_tokens": 0,
        }
        assert usage["completion_tokens_details"]["audio_tokens"] == 0
        assert result["provider_audio_output_tokens"] == 0


def _compare_or_record(path: Path, header: dict, records: list[dict], run_dir: Path) -> None:
    actual = baseline.Normalizer()([header, *records, *baseline.trajectory(run_dir)])
    if os.environ.get("REPLAY_RECORD") == "1":
        baseline.write(path, actual)
        return
    problems = baseline.compare(baseline.read(path), actual)
    assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
        problems[:60]
    )


def _header(config, scenario: str, tokens_per_frame) -> dict:
    return {
        "record": "header",
        "api": "moonshot",
        "model_key": MODEL_KEY,
        "scenario": scenario,
        "clock_mode": "token",
        "tokens_per_frame": tokens_per_frame,
        "max_ticks": MAX_TICKS,
        "config": config.model_dump(),
    }


def test_long_scenario_finite_token_clock(tmp_path, monkeypatch):
    config, records = _run(tmp_path, monkeypatch, LONG_SCENARIO, tokens_per_frame=TOKENS_PER_FRAME)
    native = config.native_output_limit
    # The driver names max_completion_tokens (providers/litellm.py:1152, 563-565); LiteLLM's
    # moonshot config sends it as max_tokens (litellm/llms/moonshot/chat/transformation.py:134-135).
    caps = [_body(r)["max_tokens"] for r in records]
    assert all("max_completion_tokens" not in _body(r) for r in records)
    assert caps == [min(native, (MAX_TICKS - k) * TOKENS_PER_FRAME) for k in range(len(records))]
    assert caps[0] == native > caps[-1] > 0
    _check_history_changes(records)
    _check_reasoning_replay(records, LONG_SCENARIO)
    _check_kimi_usage(records)
    _compare_or_record(
        BASELINES / "moonshot-token-long.jsonl",
        _header(config, "long", TOKENS_PER_FRAME),
        records,
        tmp_path,
    )


def test_pause_mode_snapshot(tmp_path, monkeypatch):
    config, records = _run(tmp_path, monkeypatch, PAUSE_SCENARIO, tokens_per_frame=None)
    # Without a finite B no clock remainder caps a decision, the harness asks for no provider cap
    # (runner.py:885-889; harnesses/default/harness.py:983) and the driver sends the declared
    # native limit (providers/litellm.py:528-537).
    assert [_body(r)["max_tokens"] for r in records] == [config.native_output_limit] * len(records)
    _check_reasoning_replay(records, PAUSE_SCENARIO)
    _check_kimi_usage(records)
    _compare_or_record(
        BASELINES / "moonshot-pause-short.jsonl",
        _header(config, "pause-short", None),
        records,
        tmp_path,
    )
