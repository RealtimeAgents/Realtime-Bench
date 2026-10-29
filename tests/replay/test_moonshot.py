"""Replay baselines for the moonshot API: the catalog cell kimi-k3 on the token clock.

Record a baseline with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it the
run is compared with tests/replay/baselines/.

The ModelConfig is the catalog's (runner.py main) and keeps the official base URL
(configs/models/kimi-k3.yaml), and the driver normalizes Kimi usage (providers/litellm.py
kimi_usage); stub_moonshot.redirect_to_stub sends the requests to the local stub. The long scenario
runs on a finite frame budget B, the short snapshot in pause mode (B unlimited).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.context_policy import resolve_context
from realtime_framework.evaluation.reporting import (
    REASONING_REPLAY_DEFINITION,
    summarize_reasoning_replay,
)
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER
from realtime_framework.entity.model import FOLDED_VISUAL_PLACEHOLDER
from realtime_framework.providers.capture import reasoning_messages
from realtime_framework.providers.litellm import DEFAULT_MOONSHOT_BASE_URL
from realtime_framework.providers.preflight import check_episode

from . import baseline
from .build import build_episode
from .frames import ReplayTask
from .scenario import LONG_SCENARIO, TASK_INSTRUCTION
from .steps import run_steps
from .stub_moonshot import KimiStub, compared_headers, kimi_replies, redirect_to_stub

BASELINES = Path(__file__).parent / "baselines"
MODEL_KEY = "kimi-k3"
FPS = 1.0
# The moonshot API requires a key (providers/litellm.py LiteLLMDriver._resolve_key). Setting it also
# keeps the repository .env from supplying one: load_project_secrets only fills unset names
# (providers/registry.py:38).
DUMMY_KEY = "sk-replay-moonshot-dummy"
ENDPOINT = DEFAULT_MOONSHOT_BASE_URL + "/chat/completions"

# run_stream receives duration_s = max_minutes * 60 and env_fps = fps (runner.py main), so the
# horizon is max_minutes * 60 * fps ticks (kernel/turns.py TurnConfig.max_ticks): a one-minute
# episode.
MAX_TICKS = 60
# Decision k observes tick k (steps.py). On the finite clock its cap is
# min(native, floor((MAX_TICKS - k) * B)) (kernel/turns.py _captured, episode_records.py
# decision_budget). kimi-k3's native limit is 131072 (configs/models/kimi-k3.yaml), so with this
# B the native limit applies through
# decision 56 and the remaining clock from decision 57 (98304) down to decision 59 (32768).
TOKENS_PER_FRAME = 32768
# As for chat_completions: the default policies (entity/context_policy.py effective_visual_fold_policy,
# effective_detailed_history_policy) fold the oldest 10 screenshots of a request holding 21
# (harnesses/default/history.py apply_active_visual_budget) and archive the oldest 10 turns of one
# holding 51 (history.py History._retain). Decision k carries k earlier turns.
FIRST_FOLD, FOLDED = 20, 10
FIRST_ARCHIVE, ARCHIVED = 50, 10
# Two reasoning replies (decisions 0 and 3), each replayed in a later request.
PAUSE_SCENARIO = LONG_SCENARIO[:5]


def _run(tmp_path, monkeypatch, steps, *, tokens_per_frame, **changes):
    monkeypatch.setenv("MOONSHOT_API_KEY", DUMMY_KEY)
    entry = catalog.get(MODEL_KEY)
    config = catalog.to_model_config(  # runner.py main
        MODEL_KEY, fps=FPS, reasoning_effort=entry.defaults["reasoning_effort"]
    ).model_copy(update=changes)
    assert config.api_base_url == DEFAULT_MOONSHOT_BASE_URL
    if tokens_per_frame is not None:
        # runner.py main
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
        transport = redirect_to_stub(monkeypatch, stub.port)
        try:
            records = run_steps(episode, steps, stub)
        finally:
            assert episode.finish() is None
    assert stub.unscripted == 0, "the agent sent requests beyond the script"
    assert stub.unused_replies == 0, "scripted replies were left unused"
    # Every request was addressed to the official endpoint and answered by the stub, and the
    # driver's capture hooks recorded each as sent.
    assert transport.urls == [ENDPOINT] * len(stub.requests)
    events = [json.loads(line) for line in (tmp_path / "episode_trace.jsonl").open()]
    sent = [e["payload"] for e in events if e["event"] == "ModelRequestSent"]
    assert [(r["channel"], r["url"]) for r in sent] == [
        ("litellm.openai_sdk", ENDPOINT)
    ] * len(stub.requests)
    # Each request records the past reasonings it carried on the wire.
    assert [
        e["payload"]["replayed_reasoning"] for e in events if e["event"] == "ModelRequestSent"
    ] == [reasoning_messages(json.loads(received.body)["messages"]) for received in stub.requests]
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
    """Kimi K3's YAML keeps its reasoning (configs/models/kimi-k3.yaml): the runtime stores it
    with the turn (harnesses/default/context.py RealtimeRuntime._run_turn) and the next request
    carries it on the turn's assistant message (harnesses/default/request.py _turn_items,
    providers/litellm.py chat_messages)."""
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
    """kimi_usage ran on every reply (providers/litellm.py kimi_usage): it adds the undivided text
    input, the cached and 5-minute cache-write splits and a zero audio output, which the token
    normalization then reports (entity/usage.py normalize_provider_token_usage)."""
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
    # The driver names max_tokens, the field Kimi takes (providers/litellm.py ChatCompat,
    # LiteLLMDriver._request_kwargs), and the run configuration records that name.
    caps = [_body(r)["max_tokens"] for r in records]
    assert all("max_completion_tokens" not in _body(r) for r in records)
    assert caps == [min(native, (MAX_TICKS - k) * TOKENS_PER_FRAME) for k in range(len(records))]
    assert caps[0] == native > caps[-1] > 0
    _check_history_changes(records)
    _check_reasoning_replay(records, LONG_SCENARIO)
    # The run's summary of them, from the requests as sent.
    counts = [reasoning_messages(_messages(r)) for r in records]
    assert sum(count > 0 for count in counts) > 0
    assert summarize_reasoning_replay(tmp_path / "episode_trace.jsonl", attempt_id="replay") == {
        "requests": len(records),
        "requests_with_reasoning": sum(count > 0 for count in counts),
        "replayed_reasoning": sum(counts),
        "exact_wire": True,
        "definition": REASONING_REPLAY_DEFINITION,
    }
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
    # (episode_records.py decision_budget; harnesses/default/harness.py
    # DefaultHarness.supports_unbounded_output) and the driver sends the declared native limit
    # (providers/litellm.py LiteLLMDriver.transport_output_token_cap).
    assert [_body(r)["max_tokens"] for r in records] == [config.native_output_limit] * len(records)
    _check_reasoning_replay(records, PAUSE_SCENARIO)
    _check_kimi_usage(records)
    _compare_or_record(
        BASELINES / "moonshot-pause-short.jsonl",
        _header(config, "pause-short", None),
        records,
        tmp_path,
    )


def test_kimi_usage_at_another_address(monkeypatch):
    """A moonshot driver addressed elsewhere than the official API (the YAML's base_url or
    MOONSHOT_BASE_URL) reads the usage as Kimi's all the same: the API decides it
    (providers/litellm.py ChatProtocol.usage, kimi_usage), not the address."""
    monkeypatch.setenv("MOONSHOT_API_KEY", DUMMY_KEY)
    config = catalog.to_model_config(MODEL_KEY, fps=FPS).model_copy(
        update={"api_base_url": "https://kimi-gateway.example/v1"}
    )
    driver = catalog.adapter_class(config.adapter_type)(config)
    (reply,) = kimi_replies(LONG_SCENARIO[:1])
    usage = driver._normalize_usage(reply.usage)
    details = usage["prompt_tokens_details"]
    assert details["text_tokens"] == usage["prompt_tokens"]
    assert details["cached_tokens_details"] == {"text_tokens": details["cached_tokens"]}
    assert details["cache_write_tokens_details"] == {
        "ephemeral_5m_tokens": details["cache_write_tokens"],
        "ephemeral_1h_tokens": 0,
    }
    assert usage["completion_tokens_details"]["audio_tokens"] == 0


def test_model_without_reasoning_replay(tmp_path, monkeypatch):
    """Another moonshot model, whose YAML sets no context, takes the standard values
    (realtime_framework/context_defaults.yaml): each request replays the earlier replies without
    their reasoning, and the pre-run checks accept it. Its YAML sets no reasoning_effort either:
    LiteLLM refuses one for a moonshot model its map does not list."""
    config, records = _run(
        tmp_path,
        monkeypatch,
        PAUSE_SCENARIO,
        tokens_per_frame=None,
        model_name="kimi-other",
        context=resolve_context("moonshot", "kimi-other"),
        reasoning_effort=None,
    )
    check_episode(config, clock_mode="token", task=ReplayTask(), env_fps=config.fps)
    assert all(_body(r)["model"] == "kimi-other" for r in records)
    summary = summarize_reasoning_replay(tmp_path / "episode_trace.jsonl", attempt_id="replay")
    assert (summary["requests"], summary["requests_with_reasoning"]) == (len(records), 0)
    assert not any("reasoning_content" in m for r in records for m in _messages(r))
    for k, step in enumerate(PAUSE_SCENARIO[:-1]):
        # Turn k is the newest turn of request k + 1, so its assistant message comes last.
        *_, replayed = (m for m in _messages(records[k + 1]) if m["role"] == "assistant")
        assert replayed == {"role": "assistant", "content": step.reply}, f"request {k + 1}"
