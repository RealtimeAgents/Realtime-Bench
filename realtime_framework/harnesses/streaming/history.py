"""The native-streaming history: what a live session should hold, kept from what the streaming
agent tells it, and the view it makes (phase2.md, native streaming 上下文怎么管).

The history keeps the blocks in the order they came to the session: each pushed frame under its
frame-table number, once its push went in; the model output of each reply accepted, or given up
once an action of it ran, as it was when first given up, where the output began, based on the
frame its action names, with the reply's tool rounds right after it where the policy replays them,
as a turn's (harnesses/default/request.py tool_exchange_block), outputs that began at one place in
the order their replies were asked for; and each execution feedback, attached to the output whose
action it reports. The view keeps the frames and outputs in that order under their ids, so an
encoder can diff two views by id. Only the newest frames keep their screenshots; an older frame
folds as a turn-based frame does, and never gets its screenshot back. Every output stays, however
old.

Each execution feedback is a block of its own, as a turn-based request places it under
separate_message (harnesses/default/request.py _turn_items): it keeps its id and stays where it
arrived, after the frames pushed while the action ran.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

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

_KIND_ROLES = {
    "frame": "input",
    "model_output": "model",
    "tool_exchange": "model",
    "execution_feedback": "input",
}


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
    history keeps them) make: the system instruction, then every block in order, the frames older
    than the newest ``retained_frames`` folded."""
    frames = [index for index, block in enumerate(blocks) if block.kind == "frame"]
    older = set(frames[: max(0, len(frames) - retained_frames)])
    items = [RequestItem(system, Intent("instruction", "head"))]
    for index, block in enumerate(blocks):
        if index in older:
            # The fold placeholder in place of its text and screenshot, as a turn-based frame
            # folds (harnesses/default/request.py _frame).
            block = replace(block, text=policy.texts.folded_visual, image=None, folded=True)
        items.append(RequestItem(block, Intent(_KIND_ROLES[block.kind], "in_order")))
    return CanonicalRequest(
        policy=policy, history_form="turns", items=tuple(items), answer=answer, call=call
    )


@dataclass
class _Unplaced:
    """A reply whose output may still join the blocks: from its start until it is accepted, or,
    given up, until an action of it ran, which an action still executing as its reply was given
    up reports later."""

    # The kernel's number of it: of outputs that began at one place, the first asked for goes first.
    generation: int
    # Where in the blocks its output began, which its output goes to: before the feedback on its
    # actions and the frames pushed since. The output of an earlier reply going in before it moves
    # it on.
    position: int | None = None
    ran: bool = False  # an action of it ran (feedback.RAN)
    kept: list[ContentBlock] | None = None  # its output, as it was given up


class StreamingHistory:
    """The blocks a native-streaming session should hold, in order, and the session they make
    (history_view). The agent tells it each frame that went into the session, and each reply's
    start, the start of its output, its acceptance or giving up, and each outcome committed; the
    history places each reply's output and lets go of each screenshot the view folds.

    It holds no lock: the agent calls it holding its own, which guards what the agent keeps beside
    it in the same step (harnesses/streaming/agent.py)."""

    def __init__(
        self,
        *,
        policy: PathPolicy,
        system: ContentBlock,
        retained_frames: int,
        answer: AnswerRequirements,
        call: CallParameters,
    ):
        self.retained_frames = retained_frames
        self._policy, self._system, self._answer, self._call = policy, system, answer, call
        self._blocks: list[ContentBlock] = []
        self._unplaced: dict[int, _Unplaced] = {}  # by decision

    @property
    def blocks(self) -> tuple[ContentBlock, ...]:
        return tuple(self._blocks)

    def view(self, frame: ContentBlock | None = None) -> CanonicalRequest:
        """The session the blocks make; with ``frame``, the one they make once that frame went in,
        which a push hands its driver before it did."""
        return history_view(
            self._blocks if frame is None else [*self._blocks, frame],
            policy=self._policy,
            system=self._system,
            retained_frames=self.retained_frames,
            answer=self._answer,
            call=self._call,
        )

    def frame_sent(self, frame: ContentBlock) -> ContentBlock | None:
        """``frame`` went into the session: it joins the blocks, and the frame it folds lets go of
        its screenshot, as a turn-based history folds one (CanonicalTurn.fold_visual): the view
        folds it already. Returns the frame folded, None while none is."""
        self._blocks.append(frame)
        frames = [index for index, block in enumerate(self._blocks) if block.kind == "frame"]
        if len(frames) <= self.retained_frames:
            return None
        index = frames[-self.retained_frames - 1]
        self._blocks[index] = replace(self._blocks[index], image=None)
        return self._blocks[index]

    def reply_started(self, decision: int, generation: int) -> None:
        """Reply ``decision`` started, the kernel's ``generation``: its output may join."""
        self._unplaced[decision] = _Unplaced(generation)

    def output_began(self, decision: int) -> None:
        """The reply's output began: here is where it joins. Told once per reply; of a reply given
        up that said nothing, which joins nothing, it may come after."""
        reply = self._unplaced.get(decision)
        if reply is not None:
            reply.position = len(self._blocks)

    def accepted(self, decision: int, blocks: list[ContentBlock]) -> None:
        """The reply was accepted: ``blocks``, its output with its tool rounds after it, none for
        a reply that failed, join where its output began."""
        self._place(decision, blocks)

    def given_up(self, decision: int, output: Callable[[], list[ContentBlock]]) -> None:
        """The reply was given up, by its abort or its discard: what ``output()`` returns, read the
        first time, as the reply is then, joins where the output began once an action of it ran,
        at once if one did already, so that a session rebuilt from the view shows the model the
        actions whose results it holds; one that said nothing joins nothing (decision 25)."""
        reply = self._unplaced.get(decision)
        if reply is None:
            return
        if reply.kept is None:
            reply.kept = output()
        if reply.ran or not reply.kept:
            self._place(decision, reply.kept)

    def committed(self, decision: int, ran: bool, feedback: Callable[[], ContentBlock]) -> None:
        """An outcome of reply ``decision`` was committed, ``ran`` if its action ran: the output
        of the reply given up joins first, then the block ``feedback()`` makes joins the end."""
        reply = self._unplaced.get(decision)
        if reply is not None and ran:
            reply.ran = True
            if reply.kept is not None:
                self._place(decision, reply.kept)
        self._blocks.append(feedback())

    def _place(self, decision: int, blocks: list[ContentBlock]) -> None:
        """The reply's ``blocks`` join where its output began, once, moving on where the output of
        each reply still to join began, after that place or there and asked for later."""
        reply = self._unplaced.pop(decision, None)
        if reply is None or not blocks:
            return
        at = reply.position
        for offset, block in enumerate(blocks):
            self._blocks.insert(at + offset, block)
        for other in self._unplaced.values():
            if other.position is not None and (
                other.position > at
                or (other.position == at and other.generation > reply.generation)
            ):
                other.position += len(blocks)
