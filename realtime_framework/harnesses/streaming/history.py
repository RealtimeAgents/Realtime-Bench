"""The native-streaming history view: what a live session should hold, computed from the content
blocks the streaming agent has accepted (phase2.md, native streaming 上下文怎么管).

The agent keeps the blocks in the order they came to the session: each pushed frame under its
frame-table number, each accepted reply's model output where the output began, based on the frame
its action names, and each execution feedback, attached to the output whose action it reports.
The view keeps the frames and outputs in that order under their ids, so an encoder can diff two
views by id. Only the newest frames keep their screenshots; an older frame folds as a turn-based
frame does, and never gets its screenshot back. Every output stays, however old.

Feedback goes where the policy places it (PathPolicy.feedback), as in a turn-based request
(harnesses/default/request.py _turn_items). As its own block, it keeps its id and stays where it
arrived, after the frames pushed while the action ran. In front of the next frame, the feedback
accepted since the frame before is one block instead, named after its first feedback, after the
outputs accepted meanwhile; the frame loses it when it folds. The feedback since the newest frame
waits for the next one at the end of the view, in the same block, which gains a line with each
feedback: a session that takes feedback between frames, as Gemini Live's reply trigger does, finds
it there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

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

_KIND_ROLES = {"frame": "input", "model_output": "model", "execution_feedback": "input"}
# The feedback placements the view lays out (PathPolicy.feedback), and whether feedback opens the
# next frame; the action digest is a Decision form no live session takes.
_FRAME_PREFIX = {"separate_message": False, "frame_prefix": True}


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
    decision: int, text: str, based_on: str | None, wall_ns: int | None
) -> ContentBlock:
    """The model output of reply ``decision``, the kernel's number of it (Outcome.decision).
    ``based_on`` is the observation_id of the frame its action names; None when no sent frame
    backs it."""
    return ContentBlock(
        f"d{decision}.output",
        "model_output",
        "model",
        BlockTime(wall_ns=wall_ns),
        text=text,
        based_on=based_on,
    )


def feedback_block(
    decision: int, call_id: str | None, text: str, wall_ns: int | None
) -> ContentBlock:
    """The execution feedback on one outcome of reply ``decision``: of the action that came in
    call ``call_id`` (Outcome.call_id), or of the reply itself when it came to no action (None).
    A reply may hand over several actions, each with its own feedback."""
    return ContentBlock(
        f"d{decision}.feedback" if call_id is None else f"d{decision}.{call_id}.feedback",
        "execution_feedback",
        "environment",
        BlockTime(wall_ns=wall_ns),
        text=text,
        attached_to=f"d{decision}.output",
    )


def history_view(
    blocks: Sequence[ContentBlock],
    *,
    policy: PathPolicy,
    system: ContentBlock,
    retained_frames: int,
    answer: AnswerRequirements,
    call: CallParameters,
) -> CanonicalRequest:
    """The session the accepted ``blocks`` (frames, outputs and feedback, in the order the
    agent keeps them) make: the system instruction, then every block in order, the frames older
    than the newest ``retained_frames`` folded, and the feedback where the policy places it."""
    frame_prefix = _FRAME_PREFIX[policy.feedback]
    frames = [index for index, block in enumerate(blocks) if block.kind == "frame"]
    older = set(frames[: max(0, len(frames) - retained_frames)])
    items = [RequestItem(system, Intent("instruction", "head"))]
    pending: list[ContentBlock] = []  # frame_prefix: the feedback since the frame before
    for index, block in enumerate(blocks):
        if frame_prefix and block.kind == "execution_feedback":
            pending.append(block)
            continue
        if index in older:
            # The fold placeholder in place of its text and screenshot, without the feedback in
            # front of it, as a turn-based frame folds (harnesses/default/request.py _frame).
            block = replace(block, text=policy.texts.folded_visual, image=None, folded=True)
        elif block.kind == "frame" and pending:
            items.append(RequestItem(_prefix(pending, block), Intent("input", "frame_prefix")))
        if block.kind == "frame":
            pending = []
        items.append(RequestItem(block, Intent(_KIND_ROLES[block.kind], "in_order")))
    if pending:
        items.append(RequestItem(_prefix(pending, None), Intent("input", "frame_prefix")))
    return CanonicalRequest(
        policy=policy, history_form="turns", items=tuple(items), answer=answer, call=call
    )


def _prefix(feedback: list[ContentBlock], frame: ContentBlock | None) -> ContentBlock:
    """The feedback accepted since the frame before, in front of ``frame``'s prompt, as a
    turn-based frame carries the feedback pending when it was prepared; with no frame yet
    (None), waiting for the next. Named after its first feedback, it keeps its id while it waits
    and gains lines, and once its frame comes."""
    return ContentBlock(
        feedback[0].id.removesuffix("feedback") + "prefix",
        "execution_feedback",
        "environment",
        BlockTime() if frame is None else frame.time,
        text="\n".join(block.text for block in feedback),
        attached_to=None if frame is None else frame.id,
    )
