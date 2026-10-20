"""Replay baselines for the anthropic_messages API: LiteLLM's Anthropic Messages transport.

The cell is claude-opus-5-anthropic (configs/models/claude-opus-5-anthropic.yaml). The harness
builds chat messages (harnesses/default/harness.py:1029-1075); LiteLLM turns them into Messages API
turns before they leave the process, and the baselines hold those wire messages as the stub
received them. On this wire:

- consecutive user messages merge into one user turn
  (litellm/litellm_core_utils/prompt_templates/factory.py:2400-2502). The execution feedback
  shares a turn with the next frame, the archived-action ledger shares one with the oldest
  folded placeholder, and a step without output adds no assistant message (harness.py:1064-1068),
  so its frame, its feedback and the next frame share one turn;
- string content becomes text blocks (factory.py:2478-2491, 2710-2728), a data URL becomes a
  base64 image source (factory.py:2408-2438), and the system message moves to the top-level
  ``system`` field (litellm/llms/anthropic/chat/transformation.py:1977-1986);
- thinking is never sent back: this policy keeps no reasoning history (providers/litellm.py:339,
  373-384; harnesses/default/context.py:543-544).

Record with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay/test_anthropic_messages.py``;
without it the run is compared with tests/replay/baselines/.
"""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER
from realtime_framework.entity.model import FOLDED_VISUAL_PLACEHOLDER

from . import baseline
from .build import build_episode
from .frames import ReplayTask
from .scenario import LONG_SCENARIO, TASK_INSTRUCTION
from .steps import run_steps
from .stub_anthropic import AnthropicMessagesStub, Reply, compared_headers, replies

BASELINES = Path(__file__).parent / "baselines"
MODEL_KEY = "claude-opus-5-anthropic"
DUMMY_KEY = "replay-dummy-anthropic-key"

# With the default policies (entity/config.py:35-38) a request holding 21 screenshots folds the
# oldest 10 (harnesses/default/context.py:27-36), and one holding 51 turns archives the oldest 10
# (context.py:262-272). Decision k carries k earlier turns plus its own frame.
FIRST_FOLD, FOLDED = 20, 10
FIRST_ARCHIVE, ARCHIVED = 50, 10

# --tokens-per-frame 256 (the default B, entity/config.py:33) and --max-minutes 9 at 1 fps:
# run_stream gets max_ticks 9 * 60 * 1 = 540 (runner.py:1773). Decision k observes tick k, so it
# has (540 - k) * 256 clock tokens left (runner.py:864-871). That stays at or above the native
# limit of 128000 up to k = 40; from k = 41 the remaining clock is the cap (runner.py:873-884),
# down to 123136 at k = 59, never 0.
TOKENS_PER_FRAME = 256
MAX_TICKS = 540
FIRST_CAPPED = 41

# Pause mode (--tokens-per-frame unlimited, runner.py:1401-1402): nothing bounds a decision, the
# harness accepts unbounded output (harness.py:920), and the driver then sends the native limit
# (providers/litellm.py:527-536). The first four decisions of the long scenario, two of them with a
# thinking block.
PAUSE_STEPS = LONG_SCENARIO[:4]


def anthropic_config(base_url: str) -> ModelConfig:
    """What the runner builds for --model claude-opus-5-anthropic (runner.py:1424-1428, 1539-1544),
    with the base URL moved to the stub before create_agent (runner.py:1560).

    This is the only redirect: the policy reads no base-URL variable (providers/litellm.py:373-384,
    473-477), the driver always passes api_base, which LiteLLM prefers to ANTHROPIC_API_BASE
    (providers/litellm.py:613-614; litellm/main.py:2879-2886), and to_model_config takes no
    api_base_url override (providers/registry.py:658-663).
    """
    effort = catalog.get(MODEL_KEY).defaults["reasoning_effort"]
    config = catalog.to_model_config(MODEL_KEY, fps=1.0, reasoning_effort=effort)
    config.api_base_url = base_url
    return config


def _run(tmp_path, monkeypatch, steps, *, tokens_per_frame, max_ticks):
    # The cell's key variable; the driver refuses to start without a key
    # (providers/litellm.py:496-502).
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    scripted = replies(steps)
    with AnthropicMessagesStub(scripted) as stub:
        config = anthropic_config(stub.base_url)
        episode = build_episode(
            config,
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=MODEL_KEY,
            clock_mode="token",
            tokens_per_frame=tokens_per_frame,
            max_ticks=max_ticks,
        )
        try:
            records = run_steps(episode, steps, stub)
        finally:
            episode.agent.close()
    assert stub.unscripted == 0, "the agent sent requests beyond the script"
    assert stub.unused_replies == 0, "scripted replies were left unused"
    assert all(len(r["requests"]) == 1 for r in records)
    assert all(r["result"]["inspection"]["valid"] for r in records)
    _add_wire_details(records, stub)
    return config, scripted, records


def _add_wire_details(records: list[dict], stub: AnthropicMessagesStub) -> None:
    """Give each request its compared headers, and store each image by the SHA-256 of its bytes.

    run_steps lists every step's requests in arrival order (steps.py:33, 49-52). Images travel as
    base64 sources rather than data URLs, which baseline.Normalizer hashes.
    """
    received = iter(stub.requests)
    for record in records:
        for request in record["requests"]:
            request["headers"] = compared_headers(next(received))
            for turn in request["body"]["messages"]:
                for block in turn["content"]:
                    if block["type"] == "image":
                        data = base64.b64decode(block["source"]["data"])
                        block["source"]["data"] = "sha256:" + hashlib.sha256(data).hexdigest()


def _turns(record: dict) -> list[dict]:
    (request,) = record["requests"]
    return request["body"]["messages"]


def _user_texts(record: dict) -> list[str]:
    return [
        block["text"]
        for turn in _turns(record)
        if turn["role"] == "user"
        for block in turn["content"]
        if block["type"] == "text"
    ]


def _check_requests(records: list[dict], clock_text: str) -> None:
    for record in records:
        (request,) = record["requests"]
        body = request["body"]
        assert request["path"] == "/v1/messages"
        assert body["stream"] is True
        assert clock_text in body["system"][0]["text"]  # harnesses/default/prompts.py:169-201
        # The catalog effort "low" keeps adaptive thinking (providers/litellm.py:263-281, 573-574).
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body["output_config"] == {"effort": "low"}
        # No thinking block is sent back, only text and images.
        kinds = {block["type"] for turn in body["messages"] for block in turn["content"]}
        assert kinds <= {"text", "image"}
        # One breakpoint, on the last block of the newest turn (providers/litellm.py:293-302).
        marked = [
            (i, j)
            for i, turn in enumerate(body["messages"])
            for j, block in enumerate(turn["content"])
            if "cache_control" in block
        ]
        newest = body["messages"][-1]
        assert marked == [(len(body["messages"]) - 1, len(newest["content"]) - 1)]
        assert newest["content"][-1]["cache_control"] == {"type": "ephemeral"}
        # The newest image is the SoM-marked frame its inspection bound (harness.py:856-886).
        marked_image = record["result"]["inspection"]["marked_image_sha256"]
        assert newest["content"][-1]["source"]["data"] == "sha256:" + marked_image


def _check_usage(records: list[dict], scripted: list[Reply]) -> None:
    """Usage merges message_start and message_delta (providers/litellm.py:172-260, 836-854)."""
    for record, reply in zip(records, scripted):
        result = record["result"]
        native = result["usage"]["raw_native_usage"]
        assert native == {"message_start": reply.start_usage, "message_delta": reply.delta_usage}
        start = reply.start_usage
        cached = start["cache_read_input_tokens"] + start["cache_creation_input_tokens"]
        assert result["usage"]["input_tokens"] == start["input_tokens"] + cached
        assert result["provider_output_tokens"] == reply.delta_usage["output_tokens"]
        # Complete, as a finite token clock requires (runner.py:477-485).
        assert result["provider_usage_missing_reason"] is None


def _check_history_changes(records: list[dict]) -> None:
    folds = [_user_texts(r).count(FOLDED_VISUAL_PLACEHOLDER) for r in records]
    ledgers = [
        [text for text in _user_texts(r) if text.startswith(ARCHIVED_ACTION_LEDGER_HEADER)]
        for r in records
    ]
    assert next(k for k, folded in enumerate(folds) if folded) == FIRST_FOLD
    assert folds[FIRST_FOLD] == FOLDED
    kinds = [block["type"] for turn in _turns(records[FIRST_FOLD]) for block in turn["content"]]
    assert kinds.count("image") == FIRST_FOLD + 1 - FOLDED
    runtime = records[FIRST_FOLD]["result"]["realtime_runtime"]
    assert runtime["history_folded_visuals"] == FOLDED
    assert next(k for k, ledger in enumerate(ledgers) if ledger) == FIRST_ARCHIVE
    (ledger,) = ledgers[FIRST_ARCHIVE]
    assert len(ledger.splitlines()) == 1 + ARCHIVED
    # The ledger and the oldest kept turn's folded frame share the first user turn.
    first = _turns(records[FIRST_ARCHIVE])[0]
    assert first == {
        "role": "user",
        "content": [
            {"type": "text", "text": ledger},
            {"type": "text", "text": FOLDED_VISUAL_PLACEHOLDER},
        ],
    }
    runtime = records[FIRST_ARCHIVE]["result"]["realtime_runtime"]
    assert runtime["history_archived_turns"] == ARCHIVED


def _check_merged_turns(records: list[dict]) -> None:
    """A rejected action's detail reaches the model ahead of the next frame, in the same turn; a
    step without output leaves two frames in one turn."""
    rejected = [k for k, r in enumerate(records) if r["commit"]["detail"]]
    assert rejected
    for k in rejected:
        turn = _turns(records[k + 1])[-1]
        feedback, prompt, image = turn["content"]
        assert turn["role"] == "user"
        assert feedback["text"].startswith("EXECUTION RESULT: ")
        assert "status=rejected." in feedback["text"]
        assert records[k]["commit"]["detail"] in feedback["text"]
        assert (prompt["type"], image["type"]) == ("text", "image")
    silent = [k for k, r in enumerate(records) if not r["result"]["raw"]]
    assert silent
    for k in silent:
        kinds = [block["type"] for block in _turns(records[k + 1])[-1]["content"]]
        assert kinds.count("image") == 2


def _compare_or_record(name: str, header: dict, records: list[dict], run_dir: Path) -> None:
    actual = baseline.Normalizer()([header, *records, *baseline.trajectory(run_dir)])
    path = BASELINES / name
    if os.environ.get("REPLAY_RECORD") == "1":
        baseline.write(path, actual)
        return
    problems = baseline.compare(baseline.read(path), actual)
    assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
        problems[:60]
    )


def _header(config: ModelConfig, scenario: str, tokens_per_frame, max_ticks) -> dict:
    return {
        "record": "header",
        "api": "anthropic_messages",
        "model_key": MODEL_KEY,
        "scenario": scenario,
        "clock_mode": "token",
        "tokens_per_frame": tokens_per_frame,
        "max_ticks": max_ticks,
        "config": config.model_dump(),
    }


def test_long_scenario_token_clock(tmp_path, monkeypatch):
    # runner.py:1449-1456
    assert catalog.adapter_class("anthropic_messages").capabilities.finite_frame_budget
    config, scripted, records = _run(
        tmp_path,
        monkeypatch,
        LONG_SCENARIO,
        tokens_per_frame=TOKENS_PER_FRAME,
        max_ticks=MAX_TICKS,
    )
    _check_requests(records, f"B={TOKENS_PER_FRAME} charged tokens/frame")
    _check_usage(records, scripted)
    _check_history_changes(records)
    _check_merged_turns(records)
    # max_tokens is the decision's transport cap (providers/litellm.py:527-536, 561-563).
    caps = [r["requests"][0]["body"]["max_tokens"] for r in records]
    native = config.native_output_limit
    assert caps[:FIRST_CAPPED] == [native] * FIRST_CAPPED
    assert caps[FIRST_CAPPED:] == [
        (MAX_TICKS - k) * TOKENS_PER_FRAME for k in range(FIRST_CAPPED, len(records))
    ]
    _compare_or_record(
        "anthropic_messages-token-long.jsonl",
        _header(config, "long", TOKENS_PER_FRAME, MAX_TICKS),
        records,
        tmp_path,
    )


def test_short_scenario_pause_mode(tmp_path, monkeypatch):
    config, scripted, records = _run(
        tmp_path, monkeypatch, PAUSE_STEPS, tokens_per_frame=None, max_ticks=None
    )
    _check_requests(records, "B is unlimited")
    _check_usage(records, scripted)
    caps = [r["requests"][0]["body"]["max_tokens"] for r in records]
    assert caps == [config.native_output_limit] * len(records)
    _compare_or_record(
        "anthropic_messages-pause-short.jsonl",
        _header(config, "short", None, None),
        records,
        tmp_path,
    )
