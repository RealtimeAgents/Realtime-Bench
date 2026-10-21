"""Canonical-request generation and its chat and digest renderings, on hand-built turn records."""

from __future__ import annotations

import base64
import json

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.model import (
    ARCHIVED_ACTION_LEDGER_HEADER,
    FOLDED_VISUAL_PLACEHOLDER,
    CanonicalTurn,
    ProviderObservation,
)
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.context import session_frame_text
from realtime_framework.harnesses.default.request import build_request, ledger_block, system_block
from realtime_framework.providers.decisions.history import render_digest
from realtime_framework.providers.litellm import chat_messages
from realtime_framework.sandbox.observation.som import FRAME_INTERACTABLES_HEADER

POLICY = current_context_policy()
ANSWER = AnswerRequirements("text_json", "standard_vlm")
CALL = CallParameters("replay-vlm", "low", None)
SYSTEM = system_block("SYSTEM", None)
LEDGER = ledger_block('{"decision":1}', 1, None)


def turn(k, prompt="FRAME", *, output="", feedback="", prefix="", **refs) -> CanonicalTurn:
    observation = ProviderObservation(
        observation=Observation(b"raw", 100.0 + k, float(k), {"observation_id": f"o{k}"}),
        prompt=prompt,
        image_bytes=f"png{k}".encode(),
        mime_type="image/png",
        image_detail="auto",
        geometry={},
        feedback_prefix=prefix,
    )
    return CanonicalTurn(
        observation=observation,
        output=output,
        execution_feedback=feedback,
        provider_refs=dict(refs),
        decision=k,
    )


def request(api, turns, ledger=None):
    return build_request(POLICY.paths[api], SYSTEM, tuple(turns), ledger, ANSWER, CALL)


def image_part(k):
    url = "data:image/png;base64," + base64.b64encode(f"png{k}".encode()).decode()
    return {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}


def test_stateless_turns_and_ledger():
    folded = turn(1, output="A1", feedback="F1")
    folded.fold_visual()
    turns = [folded, turn(2, output="A2", feedback="F2"), turn(3)]
    messages = chat_messages(request("chat_completions", turns, LEDGER))
    assert messages == (
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": ARCHIVED_ACTION_LEDGER_HEADER + '\n{"decision":1}'},
        {"role": "user", "content": FOLDED_VISUAL_PLACEHOLDER},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "F1"},
        {"role": "user", "content": [{"type": "text", "text": "FRAME"}, image_part(2)]},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "F2"},
        {"role": "user", "content": [{"type": "text", "text": "FRAME"}, image_part(3)]},
    )


def test_session_prefix_is_rejoined_and_feedback_stays_on_the_turn():
    first = turn(1, output="A1", feedback="F1")
    second = turn(2, prefix="F1")
    assert session_frame_text(second.observation) == "F1\nFRAME"
    canonical = request("openai_realtime", [first, second])
    kinds = [(item.block.id, item.intent.position) for item in canonical.items]
    assert kinds == [
        ("system", "head"),
        ("d1.frame", "in_order"),
        ("d1.output", "in_order"),
        ("d2.prefix", "frame_prefix"),
        ("d2.frame", "in_order"),
    ]
    messages = chat_messages(canonical)
    assert messages[-1]["content"][0]["text"] == "F1\nFRAME"
    assert all(message["content"] != "F1" for message in messages)
    # Folding drops the prefix, as the frame prompt was dropped before.
    second.fold_visual()
    assert session_frame_text(second.observation) == ""
    assert chat_messages(request("openai_realtime", [first, second]))[-1] == {
        "role": "user",
        "content": FOLDED_VISUAL_PLACEHOLDER,
    }


def test_ledger_is_an_input():
    for api in ("openai_realtime", "openai_decisions"):
        canonical = request(api, [turn(1)], LEDGER)
        assert canonical.items[1].intent.role == "input"
        assert chat_messages(canonical)[1]["role"] == "user"


def test_reasoning_replay():
    reasoned = turn(1, output="A1")
    reasoned.output_reasoning = "R1"
    silent = turn(2)
    silent.output_reasoning = "R2"
    turns = [reasoned, silent, turn(3)]
    # Every API row rebuilds the reply from the model output alone, moonshot's included; a reply
    # that was only reasoning leaves no assistant message.
    for api in ("chat_completions", "litellm", "openrouter", "moonshot"):
        assert [m for m in chat_messages(request(api, turns)) if m["role"] == "assistant"] == [
            {"role": "assistant", "content": "A1"}
        ], api
    # Kimi K3's YAML names the kimi_k3 preset, which keeps the reasoning
    # (entity/context_policy.py PRESETS).
    k3 = POLICY.for_path("moonshot", {}, "kimi_k3")
    moonshot = chat_messages(build_request(k3, SYSTEM, tuple(turns), None, ANSWER, CALL))
    assert moonshot[2] == {"role": "assistant", "content": "A1", "reasoning_content": "R1"}
    assert moonshot[4] == {"role": "assistant", "content": "", "reasoning_content": "R2"}


def test_gemini_tool_exchanges_are_recorded_not_converted():
    exchange = {
        "function_calls": [{"id": "c1", "name": "wait", "arguments": {}}],
        "function_responses": [{"id": "c1", "name": "wait"}],
    }
    turns = [turn(1, output='{"action": "wait"}', native_tool_exchanges=[exchange]), turn(2)]
    canonical = request("gemini_live", turns)
    (tools,) = [item.block for item in canonical.items if item.block.kind == "tool_exchange"]
    assert tools.exchanges[0].calls == tuple(exchange["function_calls"])
    assert tools.based_on == "o1"
    assert [m["role"] for m in chat_messages(canonical)] == ["system", "user", "assistant", "user"]


def test_action_digest():
    frame = "Choose." + FRAME_INTERACTABLES_HEADER + json.dumps({"observation_id": "o3"})
    turns = [
        turn(1, output="A1", feedback="F1"),
        turn(2),
        turn(3, frame),
    ]
    canonical = request("jev_decisions", turns, LEDGER)
    assert canonical.history_form == "action_digest"
    assert all(item.block.image is None for item in canonical.items)
    history = [{"observation_id": "o1", "action": "A1", "execution_feedback": "F1"}]
    assert tuple(render_digest(canonical, compact_ledger=False)) == (
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": ARCHIVED_ACTION_LEDGER_HEADER + '\n{"decision":1}'},
        {"role": "user", "content": "Recent action and feedback history: " + json.dumps(history)},
        {"role": "user", "content": 'Current observation: {"observation_id": "o3"}'},
    )
    # A stale first decision leaves no turn: the audit summary still renders the system text.
    assert tuple(render_digest(request("jev_decisions", []), compact_ledger=False)) == (
        {"role": "system", "content": "SYSTEM"},
    )


def test_record_hashes_images():
    record = request("chat_completions", [turn(1)]).to_record()
    image = record["items"][1]["block"]["image"]["data"]
    assert set(image) == {"byte_length", "sha256"}
    json.dumps(record)
