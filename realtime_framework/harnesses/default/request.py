"""Canonical-request generation: retained turns, the ledger and the system text, by policy.

Blocks come from the history the runtime keeps: one turn record per decision, plus the
archived-action ledger. The context policy decides which blocks a request carries and with which
role and placement; encoders only translate the result.
"""

from __future__ import annotations

import json
from dataclasses import replace

from realtime_framework.entity.context_policy import PathPolicy
from realtime_framework.entity.model import CanonicalTurn
from realtime_framework.entity.request import (
    AnswerRequirements,
    BlockTime,
    CallParameters,
    CanonicalRequest,
    ContentBlock,
    FrameImage,
    Intent,
    RequestItem,
    SelectionMaterial,
    ToolExchange,
)


def system_block(text: str, wall_ns: int | None) -> ContentBlock:
    return ContentBlock(
        "system", "system_instruction", "agent", BlockTime(wall_ns=wall_ns), text=text
    )


def ledger_block(text: str, revision: int, wall_ns: int | None) -> ContentBlock | None:
    """The archived-action ledger (JSON Lines, after a note once the earliest are left out),
    or None before the first archive."""
    if not text:
        return None
    return ContentBlock(
        "ledger", "action_ledger", "agent", BlockTime(wall_ns=wall_ns), text=text, revision=revision
    )


def build_request(
    policy: PathPolicy,
    system: ContentBlock,
    turns: tuple[CanonicalTurn, ...],
    ledger: ContentBlock | None,
    answer: AnswerRequirements,
    call: CallParameters,
    selection: SelectionMaterial | None = None,
) -> CanonicalRequest:
    items = [RequestItem(system, Intent("instruction", "head"))]
    if ledger is not None:
        items.append(RequestItem(ledger, Intent(policy.ledger_role, "after_head")))
    history = policy.decision_history
    history_form = "turns" if history is None else history.form
    if history_form == "action_digest":
        items += _digest_items(policy, turns)
    else:
        for turn in turns:
            items += _turn_items(policy, turn)
    return CanonicalRequest(
        policy=policy,
        history_form=history_form,
        items=tuple(items),
        answer=answer,
        call=call,
        selection=selection,
    )


def _turn_items(policy: PathPolicy, turn: CanonicalTurn) -> list[RequestItem]:
    frame = _frame(policy, turn)
    items = []
    prefix = _prefix(turn, frame)
    if prefix is not None:
        items.append(RequestItem(prefix, Intent("input", "frame_prefix")))
    items.append(RequestItem(frame, Intent("input", "in_order")))
    reasoning = turn.output_reasoning if policy.reasoning_replay else ""
    if turn.output or reasoning:
        items.append(RequestItem(_output(turn, reasoning=reasoning), Intent("model", "in_order")))
    # Stateless transports rebuild from this canonical sequence on every request. Attach each
    # action's actual execution result at its causal position, rather than appending a growing
    # duplicate ledger after the newest observation. Older turns form a stable prefix after their
    # SoM text is retired, until a block fold/archive changes it.
    if policy.feedback == "separate_message" and turn.execution_feedback:
        items.append(RequestItem(_feedback(turn), Intent("input", "in_order")))
    if policy.tool_call_replay is not None:
        exchanges = turn.provider_refs.get("native_tool_exchanges", [])
        initial_text = turn.provider_refs.get("native_initial_text", "")
        if exchanges or initial_text:
            # Item 10: past calls replayed as text, never as calls the model would see as new.
            transcript = (
                policy.texts.gemini_tool_history_header
                + "\n"
                + json.dumps(exchanges, ensure_ascii=False)
                if exchanges and policy.tool_call_replay == "tool_transcript"
                else ""
            )
            tools = ContentBlock(
                f"d{turn.decision}.tools",
                "tool_exchange",
                "model",
                BlockTime(wall_ns=turn.output_wall_ns),
                text=transcript,
                based_on=turn.observation.observation_id,
                initial_text=initial_text,
                exchanges=tuple(
                    ToolExchange(
                        calls=tuple(exchange["function_calls"]),
                        receipts=tuple(exchange["function_responses"]),
                        text_after=exchange.get("completion_text", ""),
                    )
                    for exchange in exchanges
                ),
            )
            items.append(RequestItem(tools, Intent("model", "in_order")))
    return items


def _digest_items(policy: PathPolicy, turns: tuple[CanonicalTurn, ...]) -> list[RequestItem]:
    """Older turns contribute their action and feedback only, after the feedback their frame
    opened with; the newest one is the frame."""
    if not turns:
        return []
    items = []
    for turn in turns[:-1]:
        prefix = _prefix(turn, _frame(policy, turn))
        if prefix is not None:
            opened = replace(prefix, observation_id=turn.observation.observation_id)
            items.append(RequestItem(opened, Intent("input", "digest")))
        if turn.output or turn.execution_feedback:
            items.append(RequestItem(_output(turn), Intent("model", "digest")))
            items.append(RequestItem(_feedback(turn), Intent("input", "digest")))
    frame = _frame(policy, turns[-1])
    prefix = _prefix(turns[-1], frame)
    if prefix is not None:
        items.append(RequestItem(prefix, Intent("input", "frame_prefix")))
    items.append(RequestItem(frame, Intent("input", "in_order")))
    return items


def _frame(policy: PathPolicy, turn: CanonicalTurn) -> ContentBlock:
    observation = turn.observation
    raw = observation.observation
    header = (
        f"d{turn.decision}.frame",
        "frame",
        "environment",
        BlockTime(episode_s=raw.timestamp, tick=raw.tick_number),
    )
    if turn.visual_folded:
        return ContentBlock(
            *header,
            text=policy.texts.folded_visual,
            folded=True,
            observation_id=observation.observation_id,
        )
    image = (
        None
        if policy.observation == "observation_json_text"
        else FrameImage(observation.image_bytes, observation.mime_type, observation.image_detail)
    )
    return ContentBlock(
        *header, text=observation.prompt, image=image, observation_id=observation.observation_id
    )


def _prefix(turn: CanonicalTurn, frame: ContentBlock) -> ContentBlock | None:
    observation = turn.observation
    text = "\n".join(filter(None, (observation.feedback_prefix, observation.unattached_feedback)))
    if not text:
        return None
    return ContentBlock(
        f"d{turn.decision}.prefix",
        "execution_feedback",
        "environment",
        frame.time,
        text=text,
        attached_to=frame.id,
    )


def _output(turn: CanonicalTurn, *, reasoning="") -> ContentBlock:
    return ContentBlock(
        f"d{turn.decision}.output",
        "model_output",
        "model",
        BlockTime(wall_ns=turn.output_wall_ns),
        text=turn.output,
        based_on=turn.observation.observation_id,
        reasoning=reasoning,
    )


def _feedback(turn: CanonicalTurn) -> ContentBlock:
    return ContentBlock(
        f"d{turn.decision}.feedback",
        "execution_feedback",
        "environment",
        BlockTime(wall_ns=turn.feedback_wall_ns),
        text=turn.execution_feedback,
        attached_to=f"d{turn.decision}.output",
    )
