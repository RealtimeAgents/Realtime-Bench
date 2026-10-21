"""The native-streaming history view: what a live session should hold, computed from the content
blocks the streaming agent has accepted (phase2.md, native streaming 上下文怎么管).

The agent keeps the blocks in the order it accepted them: each pushed frame under its frame-table
number, each reply's model output, based on the frame its action names, and each execution
feedback, attached to the output whose action it reports. Feedback stays where it arrived, after
the frames pushed while the action ran. The view keeps that order and every block's id, so an
encoder can diff two views by id. Only the newest frames keep their screenshots; an older frame
takes the form the retention names, and never gets its screenshot back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from realtime_framework.entity.context_policy import PathPolicy
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
from realtime_framework.harnesses.default.frames import FrameRecord
from realtime_framework.sandbox.observation.som import without_observation_interactables

_ROLES = {"frame": "input", "model_output": "model", "execution_feedback": "input"}


@dataclass(frozen=True)
class StreamingRetention:
    """What the view keeps of the history older than the newest frames. Each item offers every
    form still open in the native-streaming plan (stream-plan.md 4.4)."""

    frames: int  # the newest frames, which keep their screenshots
    # What an older frame becomes, without its screenshot: "placeholder", the turn-based fold
    # placeholder in place of its text; "text", its text with its element list; "prompt", its
    # text without the element list.
    older_frames: Literal["placeholder", "text", "prompt"]


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
    answer: AnswerRequirements,
    call: CallParameters,
) -> CanonicalRequest:
    """The session the accepted ``blocks`` (frames, outputs and feedback, in acceptance order)
    make under ``retention``: the system instruction, then every block in order, the frames
    older than the newest ``retention.frames`` in their older form."""
    frames = [index for index, block in enumerate(blocks) if block.kind == "frame"]
    older = set(frames[: max(0, len(frames) - retention.frames)])
    items = [RequestItem(system, Intent("instruction", "head"))]
    for index, block in enumerate(blocks):
        if index in older:
            block = _older_frame(block, retention.older_frames, policy)
        items.append(RequestItem(block, Intent(_ROLES[block.kind], "in_order")))
    return CanonicalRequest(
        policy=policy, history_form="turns", items=tuple(items), answer=answer, call=call
    )


def _older_frame(block: ContentBlock, form: str, policy: PathPolicy) -> ContentBlock:
    if form == "placeholder":
        return replace(block, text=policy.texts.folded_visual, image=None, folded=True)
    text = block.text if form == "text" else without_observation_interactables(block.text)
    return replace(block, text=text, image=None)
