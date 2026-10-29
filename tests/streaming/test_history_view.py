"""The native-streaming history (harnesses/streaming/history.py). Its view: the newest frames keep
their screenshots, older frames fold as turn-based frames do, and every frame, output and execution
feedback stays in the order it was accepted, under its own id. Its rules, on the history alone:
where each reply's output joins, what a reply given up keeps, and which screenshots it lets go."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.model import CanonicalTurn, ProviderObservation
from realtime_framework.entity.request import (
    AnswerRequirements,
    BlockTime,
    CallParameters,
    ContentBlock,
    Intent,
)
from realtime_framework.harnesses.default.frames import FrameTable
from realtime_framework.harnesses.default.request import build_request, system_block
from realtime_framework.harnesses.streaming.history import (
    StreamingHistory,
    feedback_block,
    frame_block,
    history_view,
    output_block,
)
from realtime_framework.sandbox.observation.som import (
    FRAME_INTERACTABLES_HEADER,
    observation_prompt_with_interactables,
)

from ..context import api_policy

POLICY = api_policy("openai_realtime")
SYSTEM = system_block("Fill in the form.", 1_000)
ANSWER = AnswerRequirements("text_json", "standard_vlm")
CALL = CallParameters("gpt-realtime-2", None, None)


def _observation(n: int) -> ProviderObservation:
    prompt = observation_prompt_with_interactables(
        f"Frame {n} of the form.", f"P1 button {n}", FRAME_INTERACTABLES_HEADER
    )
    return ProviderObservation(
        observation=Observation(
            screenshot=b"", timestamp=100.0 + n, tick_number=n, metadata={"observation_id": f"o{n}"}
        ),
        prompt=prompt,
        image_bytes=f"png {n}".encode(),
        mime_type="image/png",
        image_detail="high",
        geometry={},
    )


class History:
    """Blocks in acceptance order: frames pushed, replies on the newest frame, and the feedback
    on their actions, which run in the order the replies came."""

    def __init__(self) -> None:
        self.table = FrameTable()
        self.blocks: list = []
        self.frames = 0
        self.replies = 0
        self.reported = 0

    def frame(self) -> None:
        self.frames += 1
        observation = _observation(self.frames)
        record = self.table.register(observation.observation_id, 0.0, {}, None)
        self.blocks.append(frame_block(record, observation))

    def reply(self, based_on: str | None = "newest") -> None:
        if based_on == "newest":
            based_on = next(b.observation_id for b in reversed(self.blocks) if b.kind == "frame")
        text = f'{{"action": "click", "target": "P1", "frame": {self.frames}}}'
        self.blocks.append(output_block(self.replies, text, based_on, 2_000))
        self.replies += 1

    def feedback(self) -> None:
        decision, self.reported = self.reported, self.reported + 1
        self.blocks.append(feedback_block(decision, "a1", f"applied {decision}", 3_000))

    def play(self, script: str) -> History:
        """F: push a frame; R: a reply on the newest frame; B: feedback on the oldest reply
        that has none."""
        for step in script:
            {"F": self.frame, "R": self.reply, "B": self.feedback}[step]()
        return self


def _view(blocks, frames=20):
    return history_view(
        blocks, policy=POLICY, system=SYSTEM, retained_frames=frames, answer=ANSWER, call=CALL
    )


def _frames(view):
    return [item.block for item in view.items if item.block.kind == "frame"]


def test_the_view_opens_with_the_system_instruction_and_carries_the_request_parts():
    view = _view(History().play("FR").blocks)
    assert view.items[0].block is SYSTEM
    assert view.items[0].intent == Intent("instruction", "head")
    assert (view.policy, view.history_form, view.answer, view.call) == (
        POLICY,
        "turns",
        ANSWER,
        CALL,
    )
    assert view.selection is None


def test_blocks_are_numbered_by_frame_table_decision_and_call():
    history = History().play("FFRBFR")
    assert [block.id for block in history.blocks] == [
        "f1.frame",
        "f2.frame",
        "d0.output",
        "d0.a1.feedback",
        "f3.frame",
        "d1.output",
    ]
    first_reply = history.blocks[2]
    assert (first_reply.based_on, history.blocks[3].attached_to) == ("o2", "d0.output")
    assert history.blocks[0].observation_id == "o1"
    assert history.blocks[0].image.data == b"png 1"


def test_each_action_of_a_reply_has_its_own_feedback():
    history = History().play("FRF")
    history.blocks += [
        feedback_block(0, "a2", "superseded", 3_000),
        feedback_block(0, "a1", "applied", 3_001),
        output_block(1, "", None, 4_000),
        feedback_block(1, None, "no action", 4_001),
    ]
    view = _view(history.blocks)
    feedback = [item.block for item in view.items if item.block.kind == "execution_feedback"]
    assert [(block.id, block.attached_to) for block in feedback] == [
        ("d0.a2.feedback", "d0.output"),
        ("d0.a1.feedback", "d0.output"),
        ("d1.feedback", "d1.output"),
    ]


def test_a_frame_output_and_feedback_are_the_turn_based_blocks_but_for_their_ids():
    frame, output, feedback = History().play("FRB").blocks
    turn = CanonicalTurn(
        observation=_observation(1),
        decision=0,
        output=output.text,
        output_wall_ns=2_000,
        execution_feedback="applied 0",
        feedback_wall_ns=3_000,
    )
    request = build_request(POLICY, SYSTEM, (turn,), None, ANSWER, CALL)
    turn_frame, turn_output, turn_feedback = (item.block for item in request.items[1:])
    assert replace(frame, id=turn_frame.id) == turn_frame
    assert output == turn_output
    assert replace(feedback, id=turn_feedback.id) == turn_feedback


def test_only_the_newest_frames_keep_their_screenshots():
    history = History().play("F" * 30)
    frames = _frames(_view(history.blocks, frames=20))
    assert [frame.id for frame in frames] == [f"f{n}.frame" for n in range(1, 31)]
    assert [frame.image is not None for frame in frames] == [False] * 10 + [True] * 20
    assert [frame.folded for frame in frames] == [True] * 10 + [False] * 20
    assert frames[10:] == history.blocks[10:]


def test_a_history_with_no_more_frames_than_retained_is_the_view_unchanged():
    history = History().play("FRFRB")
    view = _view(history.blocks, frames=2)
    assert [item.block for item in view.items[1:]] == history.blocks


def test_an_older_frame_is_the_turn_based_fold():
    history = History().play("FF")
    older = _frames(_view(history.blocks, frames=1))[0]
    turn = CanonicalTurn(observation=_observation(1), decision=0)
    turn.fold_visual()
    folded = build_request(POLICY, SYSTEM, (turn,), None, ANSWER, CALL).items[1].block
    assert replace(older, id=folded.id) == folded
    # The element list goes with the screenshot.
    assert (older.text, older.image, older.folded) == (POLICY.texts.folded_visual, None, True)


def test_separate_feedback_stays_where_it_was_accepted():
    # The first action ran while three newer frames were pushed; its feedback stays after them.
    history = History().play("FRFFFBRFB")
    view = _view(history.blocks, frames=2)
    assert [item.block.id for item in view.items[1:]] == [block.id for block in history.blocks]
    texts = [item for item in view.items[1:] if item.block.kind != "frame"]
    assert [item.block for item in texts] == [b for b in history.blocks if b.kind != "frame"]
    assert [item.intent for item in texts] == [
        Intent("model", "in_order"),
        Intent("input", "in_order"),
        Intent("model", "in_order"),
        Intent("input", "in_order"),
    ]
    frames = [item for item in view.items[1:] if item.block.kind == "frame"]
    assert {item.intent for item in frames} == {Intent("input", "in_order")}


def test_every_output_and_separate_feedback_stays_past_the_turn_window():
    # More replies than the turn-based window keeps verbatim; none of them is archived.
    replies = POLICY.turns_kept + POLICY.turn_archive_block + 1
    history = History().play("FRB" * replies)
    view = _view(history.blocks)
    texts = [item.block for item in view.items if item.block.kind != "frame"]
    assert texts[0] is SYSTEM
    assert texts[1:] == [block for block in history.blocks if block.kind != "frame"]
    assert all(item.block.kind != "action_ledger" for item in view.items)


def test_a_reply_whose_action_names_no_sent_frame_stays_in_the_view():
    history = History().play("F")
    history.reply(based_on=None)
    reply = _view(history.blocks).items[-1].block
    assert (reply.id, reply.based_on) == ("d0.output", None)


def _script(rng: random.Random, length: int) -> str:
    """A history that opens with a frame and reports only actions already replied."""
    steps, replies, reported = ["F"], 0, 0
    for _ in range(length):
        step = rng.choice("FFFRB" if replies > reported else "FFFR")
        replies += step == "R"
        reported += step == "B"
        steps.append(step)
    return "".join(steps)


@pytest.mark.parametrize("seed", range(6))
def test_each_longer_history_changes_the_view_only_by_retiring_frames(seed):
    rng = random.Random(seed)
    retained = rng.randint(1, 25)
    history = History().play(_script(rng, 160))
    previous: dict = {}
    for count in range(len(history.blocks) + 1):
        accepted = history.blocks[:count]
        view = _view(accepted, frames=retained)
        blocks = [item.block for item in view.items[1:]]
        frames = [block for block in blocks if block.kind == "frame"]
        with_screenshot = [frame.id for frame in frames if frame.image is not None]
        assert with_screenshot == [frame.id for frame in frames[-retained:]]
        assert [block.id for block in blocks] == [block.id for block in accepted]
        assert [b for b in blocks if b.kind != "frame"] == [
            b for b in accepted if b.kind != "frame"
        ]
        # A block changes once, when its frame leaves the newest ones.
        retired = {frame.id for frame in frames if frame.folded}
        now = {block.id: block for block in blocks}
        for block_id, before in previous.items():
            after = now[block_id]
            assert after == before or (not before.folded and block_id in retired)
        previous = now


# ---- the history's rules ----------------------------------------------------------------------


def _history(retained: int = 20) -> StreamingHistory:
    return StreamingHistory(
        policy=POLICY, system=SYSTEM, retained_frames=retained, answer=ANSWER, call=CALL
    )


def _sent_frames(count: int) -> list:
    """Frames 1 to ``count``, as the frame table numbers them."""
    table = FrameTable()
    frames = []
    for n in range(1, count + 1):
        observation = _observation(n)
        record = table.register(observation.observation_id, 0.0, {}, None)
        frames.append(frame_block(record, observation))
    return frames


def _with_frames(count: int, retained: int = 20) -> tuple[StreamingHistory, list]:
    """A history holding frames 1 and 2 of ``count``; the rest are left to send."""
    history, frames = _history(retained), _sent_frames(count)
    for frame in frames[:2]:
        history.frame_sent(frame)
    return history, frames[2:]


def _started(history: StreamingHistory, decision: int, generation: int) -> None:
    """Reply ``decision`` started, and its output began at once."""
    history.reply_started(decision, generation)
    history.output_began(decision)


def _output(decision: int, text: str = "click", wall_ns: int = 2_000) -> list:
    """A reply's output with its tool rounds after it, as the agent makes them."""
    tools = ContentBlock(
        f"d{decision}.tools", "tool_exchange", "model", BlockTime(wall_ns=wall_ns), text=text
    )
    return [output_block(decision, text, "o2", wall_ns), tools]


def _feedback(decision: int, call_id: str = "c1"):
    return lambda: feedback_block(decision, call_id, f"applied {call_id}", 3_000)


def _ids(history: StreamingHistory) -> list[str]:
    return [item.block.id for item in history.view().items[1:]]


@pytest.mark.parametrize("accepted_first", [3, 4])
@pytest.mark.parametrize(
    "generations", [(7, 8), (8, 7)], ids=["asked_in_decision_order", "asked_in_reverse"]
)
def test_outputs_that_began_at_one_place_go_there_in_the_order_their_replies_were_asked_for(
    accepted_first, generations
):
    """By the generation the kernel asked each reply for, not by its decision, whichever is
    accepted first."""
    history, _ = _with_frames(2)
    for decision, generation in zip((3, 4), generations):
        _started(history, decision, generation)
    for decision in (accepted_first, 7 - accepted_first):
        history.accepted(decision, [output_block(decision, "click", "o2", 2_000)])
    asked_first = 3 if generations[0] < generations[1] else 4
    asked = (asked_first, 7 - asked_first)
    assert _ids(history) == ["f1.frame", "f2.frame", *(f"d{d}.output" for d in asked)]


@pytest.mark.parametrize(
    ("began", "generations", "expected"),
    [
        # The second began after a frame pushed since the first began.
        ("after", (7, 8), ["d3.output", "d3.tools", "f3.frame", "d4.output", "d4.tools"]),
        # Both began at one place: the one asked for later goes after both blocks.
        ("there", (7, 8), ["d3.output", "d3.tools", "d4.output", "d4.tools", "f3.frame"]),
        # Both began at one place: the one asked for earlier stays before them.
        ("there", (8, 7), ["d4.output", "d4.tools", "d3.output", "d3.tools", "f3.frame"]),
    ],
)
def test_an_output_with_its_tool_rounds_going_in_moves_on_the_place_of_each_reply_after_it(
    began, generations, expected
):
    """Reply 3 is given up, and its output and tool rounds go in once its click ran, while
    reply 4 is on: the place reply 4's output began moves on by both blocks, if it is later or,
    at the same place, if reply 4 was asked for later."""
    history, (third,) = _with_frames(3)
    _started(history, 3, generations[0])
    history.given_up(3, lambda: _output(3))
    if began == "after":
        history.frame_sent(third)
    _started(history, 4, generations[1])
    if began == "there":
        history.frame_sent(third)
    history.committed(3, True, _feedback(3))
    history.accepted(4, _output(4))
    assert _ids(history) == ["f1.frame", "f2.frame", *expected, "d3.c1.feedback"]


@pytest.mark.parametrize("ran_first", [False, True], ids=["given_up_first", "ran_first"])
def test_a_reply_given_up_joins_once_however_many_of_its_actions_ran(ran_first):
    history, (third,) = _with_frames(3)
    _started(history, 3, 7)
    if ran_first:
        history.committed(3, True, _feedback(3, "c1"))
    history.given_up(3, lambda: _output(3))
    history.frame_sent(third)
    if not ran_first:
        history.committed(3, True, _feedback(3, "c1"))
    history.committed(3, True, _feedback(3, "c2"))
    history.given_up(3, lambda: _output(3))  # its discard, after its abort
    feedback = ["d3.c1.feedback", "f3.frame"] if ran_first else ["f3.frame", "d3.c1.feedback"]
    assert _ids(history) == [
        "f1.frame",
        "f2.frame",
        "d3.output",
        "d3.tools",
        *feedback,
        "d3.c2.feedback",
    ]


@pytest.mark.parametrize("ran", [False, True])
def test_a_reply_given_up_keeps_its_output_once_an_action_of_it_ran_before_its_result(ran):
    """Decision 25: the click still executes as its reply is given up, and a frame goes in
    meanwhile. Once the click ran, the output stands where it began, before that frame and
    before the click's result; a click that did not run brings nothing in."""
    history, (third,) = _with_frames(3)
    _started(history, 3, 7)
    history.given_up(3, lambda: _output(3))
    history.frame_sent(third)
    assert _ids(history) == ["f1.frame", "f2.frame", "f3.frame"]
    history.committed(3, ran, _feedback(3))
    kept = ["d3.output", "d3.tools"] if ran else []
    assert _ids(history) == ["f1.frame", "f2.frame", *kept, "f3.frame", "d3.c1.feedback"]


def test_a_reply_given_up_after_an_action_of_it_ran_keeps_its_output_at_once():
    history, _ = _with_frames(2)
    _started(history, 3, 7)
    history.committed(3, True, _feedback(3))
    assert _ids(history) == ["f1.frame", "f2.frame", "d3.c1.feedback"]
    history.given_up(3, lambda: _output(3))
    assert _ids(history) == ["f1.frame", "f2.frame", "d3.output", "d3.tools", "d3.c1.feedback"]


def test_a_reply_given_up_keeps_its_output_as_it_was_the_first_time():
    """Its abort freezes its output; what comes of the reply later, more text, another tool
    round, a later time, changes nothing of what joins once its click ran, and its discard does
    not read its output again."""
    history, _ = _with_frames(2)
    _started(history, 3, 7)
    reply = {"text": "Clicking.", "wall_ns": 4_000}
    reads = []

    def output():
        reads.append(dict(reply))
        return _output(3, reply["text"], reply["wall_ns"])

    history.given_up(3, output)  # its abort
    first = _output(3, "Clicking.", 4_000)
    reply.update(text="Clicking. And another tool round.", wall_ns=9_000)
    history.given_up(3, output)  # its discard
    history.committed(3, True, _feedback(3))
    assert reads == [{"text": "Clicking.", "wall_ns": 4_000}]
    assert [block for block in history.blocks if block.kind != "frame"][:2] == first
    assert _ids(history) == ["f1.frame", "f2.frame", "d3.output", "d3.tools", "d3.c1.feedback"]


def test_a_reply_that_said_nothing_or_failed_adds_no_output_whatever_ran():
    history, _ = _with_frames(2)
    history.reply_started(3, 7)
    history.given_up(3, list)  # said nothing
    _started(history, 4, 8)
    history.accepted(4, [])  # failed, and accepted
    for decision in (3, 4):
        history.committed(decision, True, _feedback(decision))
    assert _ids(history) == ["f1.frame", "f2.frame", "d3.c1.feedback", "d4.c1.feedback"]


@pytest.mark.parametrize("committed_first", [True, False])
def test_an_accepted_output_goes_where_it_began_before_the_feedback_and_frames_since(
    committed_first,
):
    """A wait is settled, or a click runs, while its reply goes on, and a frame is pushed
    meanwhile: the reply's output stands where it began, before both."""
    history, (third,) = _with_frames(3)
    _started(history, 0, 0)
    if committed_first:
        history.committed(0, True, _feedback(0))
    history.frame_sent(third)
    history.accepted(0, _output(0))
    if not committed_first:
        history.committed(0, True, _feedback(0))
    tail = ["d0.c1.feedback", "f3.frame"] if committed_first else ["f3.frame", "d0.c1.feedback"]
    assert _ids(history) == ["f1.frame", "f2.frame", "d0.output", "d0.tools", *tail]


def test_the_history_lets_go_of_the_screenshot_of_each_frame_that_folds():
    history, frames = _history(retained=2), _sent_frames(4)
    assert [history.frame_sent(frame) for frame in frames[:2]] == [None, None]
    folded = [history.frame_sent(frame) for frame in frames[2:]]
    assert [(block.id, block.image) for block in folded] == [("f1.frame", None), ("f2.frame", None)]
    assert [block.image is not None for block in history.blocks] == [False, False, True, True]


def test_a_frame_previewed_but_not_sent_lets_go_of_no_screenshot():
    """The view a push hands its driver folds the oldest frame kept; the push failed, so the
    history keeps that frame's screenshot, and the view after does not fold it."""
    history, (third,) = _with_frames(3, retained=2)
    preview = history.view(third)
    assert [(item.block.id, item.block.folded) for item in preview.items[1:]] == [
        ("f1.frame", True),
        ("f2.frame", False),
        ("f3.frame", False),
    ]
    assert [block.image is not None for block in history.blocks] == [True, True]
    assert [item.block.folded for item in history.view().items[1:]] == [False, False]


def test_a_frame_that_folded_never_gets_its_screenshot_back():
    """Outputs going in before folded frames, and a preview, leave them folded."""
    history, frames = _history(retained=1), _sent_frames(5)
    history.frame_sent(frames[0])
    _started(history, 0, 0)
    for frame in frames[1:3]:
        history.frame_sent(frame)
    _started(history, 1, 1)
    history.accepted(0, _output(0))
    history.frame_sent(frames[3])
    history.accepted(1, _output(1))
    views = [history.view(), history.view(frames[4])]
    history.frame_sent(frames[4])
    views.append(history.view())
    stored = {block.id: block.image for block in history.blocks if block.kind == "frame"}
    assert [frame for frame, image in stored.items() if image is not None] == ["f5.frame"]
    for view in views:
        older = [item.block for item in view.items if item.block.id in {"f1.frame", "f2.frame"}]
        assert [(block.folded, block.image) for block in older] == [(True, None), (True, None)]
    assert _ids(history) == [
        "f1.frame",
        "d0.output",
        "d0.tools",
        "f2.frame",
        "f3.frame",
        "d1.output",
        "d1.tools",
        "f4.frame",
        "f5.frame",
    ]
