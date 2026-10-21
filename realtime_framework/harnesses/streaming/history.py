"""The native-streaming history view: what a live session should hold, computed from the content
blocks the streaming agent has accepted (phase2.md, native streaming 上下文怎么管).

The agent keeps the blocks in the order it accepted them: each pushed frame under its frame-table
number, each reply's model output, based on the frame its action names, and each execution
feedback, attached to the output whose action it reports. Feedback stays where it arrived, after
the frames pushed while the action ran. The view keeps that order and every block's id, so an
encoder can diff two views by id. Only the newest frames keep their screenshots; an older frame
takes the form the retention names, and never gets its screenshot back. Under the archive
retention the oldest replies leave for the action ledger, and never come back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from realtime_framework.entity.context_policy import PathPolicy, Retention
from realtime_framework.entity.model import ProviderObservation
from realtime_framework.entity.request import (
    AnswerRequirements,
    BlockTime,
    CallParameters,
    CanonicalRequest,
    ContentBlock,
    FrameImage,
    Intent,
    RequestItem,
)
from realtime_framework.harnesses.default.context import executed_action_ledger
from realtime_framework.harnesses.default.frames import FrameRecord
from realtime_framework.harnesses.default.request import ledger_block
from realtime_framework.sandbox.observation.som import without_observation_interactables

_ROLES = {"frame": "input", "model_output": "model", "execution_feedback": "input"}


@dataclass(frozen=True)
class StreamingRetention:
    """What the view keeps of the history older than the newest frames. Each item offers every
    form still open in the native-streaming plan (stream-plan.md 4.4, 4.6)."""

    frames: int  # the newest frames, which keep their screenshots
    # What an older frame becomes, folded (without its screenshot): "placeholder", the turn-based
    # fold placeholder in place of its text; "text", its text with its element list; "prompt",
    # its text without the element list.
    older_frames: Literal["placeholder", "text", "prompt"]
    # "all": every output and feedback stays. "archive": the policy's turn window counts replies
    # as the turn-based one counts turns, and the replies it archives leave for the action ledger.
    text: Literal["all", "archive"]


def frame_block(frame: FrameRecord, observation: ProviderObservation) -> ContentBlock:
    """A pushed frame under its frame-table number: its prompt with its element list, and its
    screenshot, with the number drawn on it (harnesses/default/harness.py bind_observation)."""
    raw = observation.observation
    return ContentBlock(
        f"f{frame.frame}.frame",
        "frame",
        "environment",
        BlockTime(episode_s=raw.timestamp, tick=raw.tick_number),
        text=observation.prompt,
        image=FrameImage(observation.image_bytes, observation.mime_type, observation.image_detail),
        observation_id=observation.observation_id,
    )


def output_block(
    generation: int, text: str, based_on: str | None, wall_ns: int | None
) -> ContentBlock:
    """The model output of the reply the kernel's ``generation`` produced. ``based_on`` is the
    observation_id of the frame its action names; None when no sent frame backs it."""
    return ContentBlock(
        f"g{generation}.output",
        "model_output",
        "model",
        BlockTime(wall_ns=wall_ns),
        text=text,
        based_on=based_on,
    )


def feedback_block(generation: int, text: str, wall_ns: int | None) -> ContentBlock:
    """The execution feedback on the action of the reply ``generation`` produced."""
    return ContentBlock(
        f"g{generation}.feedback",
        "execution_feedback",
        "environment",
        BlockTime(wall_ns=wall_ns),
        text=text,
        attached_to=f"g{generation}.output",
    )


def history_view(
    blocks: Sequence[ContentBlock],
    *,
    policy: PathPolicy,
    system: ContentBlock,
    retention: StreamingRetention,
    actions: Mapping[str, Sequence[dict]],
    answer: AnswerRequirements,
    call: CallParameters,
) -> CanonicalRequest:
    """The session the accepted ``blocks`` (frames, outputs and feedback, in acceptance order)
    make under ``retention``: the system instruction, the action ledger, then every block not
    archived, in order, the frames older than the newest ``retention.frames`` in their older form.

    An archived reply takes the feedback on its action, wherever that arrived, and the older
    frames accepted before it; the newest frames stay. ``actions`` has each reply's action
    records by output block id; the ledger lists those of the archived replies as the turn-based
    ledger does (harnesses/default/context.py executed_action_ledger).
    """
    frames = [index for index, block in enumerate(blocks) if block.kind == "frame"]
    older = set(frames[: max(0, len(frames) - retention.frames)])
    outputs = [index for index, block in enumerate(blocks) if block.kind == "model_output"]
    archived = _archived_replies(len(outputs), policy.turns) if retention.text == "archive" else 0
    cut = outputs[archived - 1] if archived else -1
    archived_ids = {blocks[index].id for index in outputs[:archived]}
    items = [RequestItem(system, Intent("instruction", "head"))]
    if archived:
        records = [record for index in outputs[:archived] for record in actions[blocks[index].id]]
        # Dated by the reply whose acceptance last grew the archive.
        grown = blocks[outputs[policy.turns.keep + archived - policy.turns.block]]
        ledger = ledger_block(executed_action_ledger(records), len(records), grown.time.wall_ns)
        if ledger is not None:
            items.append(RequestItem(ledger, Intent(policy.ledger_role, "after_head")))
    for index, block in enumerate(blocks):
        if block.kind != "frame":
            if block.id in archived_ids or block.attached_to in archived_ids:
                continue
        elif index in older:
            if index < cut:
                continue
            block = _older_frame(block, retention.older_frames, policy)
        items.append(RequestItem(block, Intent(_ROLES[block.kind], "in_order")))
    return CanonicalRequest(
        policy=policy, history_form="turns", items=tuple(items), answer=answer, call=call
    )


def _archived_replies(count: int, turns: Retention) -> int:
    """How many of ``count`` replies the turn window has archived: a block of the oldest each
    time one more would exceed it, so the window cycles as the turn-based one does
    (harnesses/default/context.py RealtimeRuntime._retained)."""
    archived = 0
    while count - archived > turns.keep:
        archived += turns.block
    return archived


def _older_frame(block: ContentBlock, form: str, policy: PathPolicy) -> ContentBlock:
    # Folded in every form: the session encoders take a folded frame for one whose screenshot
    # left (providers/openai_realtime_session.py _edit_batch).
    if form == "placeholder":
        text = policy.texts.folded_visual
    elif form == "text":
        text = block.text
    else:
        text = without_observation_interactables(block.text)
    return replace(block, text=text, image=None, folded=True)
