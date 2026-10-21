"""The native-streaming history view (harnesses/streaming/history.py): the newest frames keep their
screenshots, older frames fold as turn-based frames do, and every output and feedback stays in the
order it was accepted, under its own id."""

from __future__ import annotations

import random
from dataclasses import replace
from itertools import pairwise

import pytest

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.model import CanonicalTurn, ProviderObservation
from realtime_framework.entity.request import (
    AnswerRequirements,
    BlockTime,
    CallParameters,
    Intent,
)
from realtime_framework.harnesses.default.frames import FrameTable
from realtime_framework.harnesses.default.request import build_request, system_block
from realtime_framework.harnesses.streaming.history import (
    feedback_block,
    frame_block,
    history_view,
    output_block,
)
from realtime_framework.providers.gemini.session import SessionCore
from realtime_framework.providers.openai_realtime_session import streaming_blocks
from realtime_framework.sandbox.observation.som import (
    FRAME_INTERACTABLES_HEADER,
    observation_prompt_with_interactables,
)

POLICY = current_context_policy().paths["openai_realtime"]
# The view lays feedback out as the policy places it; the tests take both placements on this row.
SEPARATE = replace(POLICY, feedback="separate_message")
PREFIX = replace(POLICY, feedback="frame_prefix")
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


def _view(blocks, frames=20, policy=SEPARATE):
    return history_view(
        blocks, policy=policy, system=SYSTEM, retained_frames=frames, answer=ANSWER, call=CALL
    )


def _frames(view):
    return [item.block for item in view.items if item.block.kind == "frame"]


def test_the_view_opens_with_the_system_instruction_and_carries_the_request_parts():
    view = _view(History().play("FR").blocks)
    assert view.items[0].block is SYSTEM
    assert view.items[0].intent == Intent("instruction", "head")
    assert (view.policy, view.history_form, view.answer, view.call) == (
        SEPARATE,
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
    request = build_request(SEPARATE, SYSTEM, (turn,), None, ANSWER, CALL)
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
    replies = POLICY.turns.keep + POLICY.turns.block + 1
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


def _ids(view) -> list[str]:
    return [item.block.id for item in view.items]


def _layout(view) -> list[tuple]:
    return [(item.block.id, item.intent.position, item.block.attached_to) for item in view.items]


def test_under_frame_prefix_the_feedback_since_the_frame_before_opens_the_next_frame():
    history = History().play("FRRBBF")
    view = _view(history.blocks, policy=PREFIX)
    assert _layout(view)[1:] == [
        ("f1.frame", "in_order", None),
        ("d0.output", "in_order", None),
        ("d1.output", "in_order", None),
        ("d0.a1.prefix", "frame_prefix", "f2.frame"),
        ("f2.frame", "in_order", None),
    ]
    prefix = view.items[4]
    assert prefix.block.text == "applied 0\napplied 1"
    assert prefix.intent == Intent("input", "frame_prefix")


def test_under_frame_prefix_the_feedback_since_the_newest_frame_waits_at_the_end():
    """The waiting prefix keeps its id while it gains lines and once the next frame takes it."""
    history = History().play("FRB")
    waiting = _view(history.blocks, policy=PREFIX)
    assert _layout(waiting)[1:] == [
        ("f1.frame", "in_order", None),
        ("d0.output", "in_order", None),
        ("d0.a1.prefix", "frame_prefix", None),
    ]
    # No frame yet, so no time: it takes its frame's once that frame comes.
    waiting_prefix = waiting.items[-1].block
    assert (waiting_prefix.text, waiting_prefix.time) == ("applied 0", BlockTime())
    history.play("RB")
    grown = _view(history.blocks, policy=PREFIX).items[-1].block
    assert (grown.id, grown.attached_to) == ("d0.a1.prefix", None)
    assert grown.text == "applied 0\napplied 1"
    history.play("F")
    taken = _view(history.blocks, policy=PREFIX)
    assert _layout(taken)[-2:] == [
        ("d0.a1.prefix", "frame_prefix", "f2.frame"),
        ("f2.frame", "in_order", None),
    ]
    assert taken.items[-2].block.text == "applied 0\napplied 1"
    assert taken.items[-2].block.time == taken.items[-1].block.time


def test_the_prefix_block_is_the_turn_based_one():
    history = History().play("FRBF")
    prefix = _view(history.blocks, policy=PREFIX).items[3]
    observation = replace(_observation(2), feedback_prefix="applied 0")
    turn = CanonicalTurn(observation=observation, decision=1)
    expected = build_request(PREFIX, SYSTEM, (turn,), None, ANSWER, CALL).items[1]
    assert expected.block.id == "d1.prefix"
    assert replace(prefix.block, id="d1.prefix", attached_to="d1.frame") == expected.block
    assert prefix.intent == expected.intent


def test_under_frame_prefix_a_folded_frame_loses_its_prefix_as_a_turn_based_fold_does():
    history = History().play("FRBFF")
    view = _view(history.blocks, frames=1, policy=PREFIX)
    assert _ids(view) == [
        "system",
        "f1.frame",
        "d0.output",
        "f2.frame",
        "f3.frame",
    ]
    turn = CanonicalTurn(observation=replace(_observation(2), feedback_prefix="x"), decision=1)
    turn.fold_visual()
    folded = build_request(PREFIX, SYSTEM, (turn,), None, ANSWER, CALL)
    assert [item.block.id for item in folded.items] == ["system", "d1.frame"]


def test_both_session_cores_read_the_prefix_as_native_streaming_does():
    """OpenAI Realtime's streaming reader puts a prefix in front of its frame's user item and
    leaves the one waiting for the next frame alone; Gemini Live's streaming seed replays each
    prefix as a text of its own, where the view has it, the waiting one too."""
    history = History().play("FRBFRB")
    view = _view(history.blocks, policy=PREFIX)
    assert [(item.id, item.prefix) for item in streaming_blocks(view.items)] == [
        ("f1.frame", ""),
        ("d0.output", ""),
        ("f2.frame", "applied 0"),
        ("d1.output", ""),
    ]
    core = SessionCore(
        {"completion_signal": "turn_complete", "tool_response_scheduling": "when_idle"},
        streaming=True,
    )
    seed = core.connect(view, None, replacing=False)
    (contents,) = seed.seed
    frame_1, output_0, _, frame_2, output_1, _ = (block.text for block in history.blocks)
    assert [part.text for content in contents for part in content.parts if part.text] == [
        frame_1,
        output_0,
        "applied 0",
        frame_2,
        output_1,
        "applied 1",
    ]
    assert seed.seeded == (
        "f1.frame",
        "d0.output",
        "d0.a1.prefix",
        "f2.frame",
        "d1.output",
        "d1.a1.prefix",
    )


def test_a_feedback_placement_the_view_does_not_lay_out_is_refused():
    with pytest.raises(KeyError, match="digest"):
        _view(History().play("FRBF").blocks, policy=replace(POLICY, feedback="digest"))


def _script(rng: random.Random, length: int) -> str:
    """A history that opens with a frame and reports only actions already replied."""
    steps, replies, reported = ["F"], 0, 0
    for _ in range(length):
        step = rng.choice("FFFRB" if replies > reported else "FFFR")
        replies += step == "R"
        reported += step == "B"
        steps.append(step)
    return "".join(steps)


def _prefixes(accepted) -> dict[str | None, str]:
    """Each frame's prefix text by its definition: the feedback accepted since the frame before;
    under None, the feedback since the newest frame."""
    prefixes, since = {}, []
    for block in accepted:
        if block.kind == "execution_feedback":
            since.append(block.text)
        elif block.kind == "frame":
            if since:
                prefixes[block.id] = "\n".join(since)
            since = []
    if since:
        prefixes[None] = "\n".join(since)
    return prefixes


@pytest.mark.parametrize("policy", [SEPARATE, PREFIX], ids=["separate", "prefix"])
@pytest.mark.parametrize("seed", range(6))
def test_each_longer_history_changes_the_view_only_by_retiring_frames(seed, policy):
    rng = random.Random(seed)
    retained = rng.randint(1, 25)
    history = History().play(_script(rng, 160))
    previous: dict = {}
    for count in range(len(history.blocks) + 1):
        accepted = history.blocks[:count]
        view = _view(accepted, frames=retained, policy=policy)
        blocks = [item.block for item in view.items[1:]]
        frames = [block for block in blocks if block.kind == "frame"]
        with_screenshot = [frame.id for frame in frames if frame.image is not None]
        assert with_screenshot == [frame.id for frame in frames[-retained:]]
        others = [block for block in accepted if block.kind != "execution_feedback"]
        if policy is SEPARATE:
            assert [block.id for block in blocks] == [block.id for block in accepted]
            assert [b for b in blocks if b.kind != "frame"] == [
                b for b in accepted if b.kind != "frame"
            ]
        else:
            # Frames and outputs keep their order; each prefix comes just before its frame,
            # which has not folded, and the one waiting for the next frame comes last.
            prefixes = {b.attached_to: b.text for b in blocks if b.kind == "execution_feedback"}
            assert [b.id for b in blocks if b.kind != "execution_feedback"] == [
                b.id for b in others
            ]
            for before, after in pairwise(blocks):
                if before.kind == "execution_feedback":
                    assert (after.id, after.folded) == (before.attached_to, False)
            if blocks and blocks[-1].kind == "execution_feedback":
                assert blocks[-1].attached_to is None
            kept = {frame.id for frame in frames if not frame.folded} | {None}
            expected = _prefixes(accepted)
            assert prefixes == {key: expected[key] for key in expected if key in kept}
        # A block changes once, when its frame leaves the newest ones, and with it goes its
        # prefix; a prefix waiting for the next frame only gains lines until that frame takes it.
        retired = {frame.id for frame in frames if frame.folded}
        now = {block.id: block for block in blocks}
        for block_id, before in previous.items():
            after = now.get(block_id)
            if after is None:
                assert before.kind == "execution_feedback" and before.attached_to in retired
            elif before.kind == "execution_feedback" and before.attached_to is None:
                assert after.text.startswith(before.text)
                assert after.attached_to is None or after.attached_to not in previous
            else:
                assert after == before or (not before.folded and block_id in retired)
        previous = now
