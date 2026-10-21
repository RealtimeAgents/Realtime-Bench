"""The native-streaming history view (harnesses/streaming/history.py): the newest frames keep their
screenshots, older frames take the retention's form, and every output and feedback stays in the
order it was accepted, under its own id; or, under the archive retention, the oldest replies
leave for the action ledger as turn-based turns do."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.context_policy import Retention, current_context_policy
from realtime_framework.entity.model import CanonicalTurn, ProviderObservation
from realtime_framework.entity.request import AnswerRequirements, CallParameters, Intent
from realtime_framework.harnesses.default.context import executed_action_ledger
from realtime_framework.harnesses.default.frames import FrameTable
from realtime_framework.harnesses.default.request import build_request, system_block
from realtime_framework.harnesses.streaming.history import (
    StreamingRetention,
    feedback_block,
    frame_block,
    history_view,
    output_block,
)
from realtime_framework.sandbox.observation.som import (
    FRAME_INTERACTABLES_HEADER,
    observation_prompt_with_interactables,
)

POLICY = current_context_policy().paths["openai_realtime"]
SYSTEM = system_block("Fill in the form.", 1_000)
ANSWER = AnswerRequirements("text_json", "standard_vlm")
CALL = CallParameters("gpt-realtime-2", None, None)
OLDER_FORMS = ("placeholder", "text", "prompt")


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
        self.actions: dict[str, tuple[dict, ...]] = {}  # each reply's action records
        self.frames = 0
        self.generations = 0
        self.reported = 0

    def frame(self) -> None:
        self.frames += 1
        observation = _observation(self.frames)
        record = self.table.register(observation.observation_id, 0.0, {}, None)
        self.blocks.append(frame_block(record, observation))

    def reply(self, based_on: str | None = "newest") -> None:
        self.generations += 1
        if based_on == "newest":
            based_on = next(b.observation_id for b in reversed(self.blocks) if b.kind == "frame")
        text = f'{{"action": "click", "target": "P1", "frame": {self.frames}}}'
        block = output_block(self.generations, text, based_on, 2_000 + self.generations)
        self.blocks.append(block)
        self.actions[block.id] = (
            {
                "tick": self.generations,
                "observation_tick": self.frames,
                "requested": "click",
                "executed": "click",
                "result": "applied",
            },
        )

    def feedback(self) -> None:
        self.reported += 1
        self.blocks.append(feedback_block(self.reported, f"applied {self.reported}", 3_000))

    def play(self, script: str) -> History:
        """F: push a frame; R: a reply on the newest frame; B: feedback on the oldest reply
        that has none."""
        for step in script:
            {"F": self.frame, "R": self.reply, "B": self.feedback}[step]()
        return self


def _view(
    blocks, frames=20, older_frames="placeholder", replies="all", actions=None, policy=POLICY
):
    return history_view(
        blocks,
        policy=policy,
        system=SYSTEM,
        retention=StreamingRetention(frames=frames, older_frames=older_frames, replies=replies),
        actions=actions or {},
        answer=ANSWER,
        call=CALL,
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


def test_blocks_are_numbered_by_frame_table_and_generation():
    history = History().play("FFRBFR")
    assert [block.id for block in history.blocks] == [
        "f1.frame",
        "f2.frame",
        "g1.output",
        "g1.feedback",
        "f3.frame",
        "g2.output",
    ]
    first_reply = history.blocks[2]
    assert (first_reply.based_on, history.blocks[3].attached_to) == ("o2", "g1.output")
    assert history.blocks[0].observation_id == "o1"
    assert history.blocks[0].image.data == b"png 1"


@pytest.mark.parametrize("older_frames", OLDER_FORMS)
def test_only_the_newest_frames_keep_their_screenshots(older_frames):
    history = History().play("F" * 30)
    frames = _frames(_view(history.blocks, frames=20, older_frames=older_frames))
    assert [frame.id for frame in frames] == [f"f{n}.frame" for n in range(1, 31)]
    assert [frame.image is not None for frame in frames] == [False] * 10 + [True] * 20
    assert frames[10:] == history.blocks[10:]


def test_a_history_with_no_more_frames_than_retained_is_the_view_unchanged():
    history = History().play("FRFRB")
    view = _view(history.blocks, frames=2)
    assert [item.block for item in view.items[1:]] == history.blocks


def test_an_older_frame_in_placeholder_form_is_the_turn_based_fold():
    history = History().play("FF")
    older = _frames(_view(history.blocks, frames=1, older_frames="placeholder"))[0]
    turn = CanonicalTurn(observation=_observation(1), decision=0)
    turn.fold_visual()
    folded = build_request(POLICY, SYSTEM, (turn,), None, ANSWER, CALL).items[1].block
    assert replace(older, id=folded.id) == folded
    assert (older.text, older.image, older.folded) == (POLICY.texts.folded_visual, None, True)


def test_an_older_frame_in_text_form_keeps_its_element_list():
    history = History().play("FF")
    older = _frames(_view(history.blocks, frames=1, older_frames="text"))[0]
    assert older == replace(history.blocks[0], image=None, folded=True)
    assert FRAME_INTERACTABLES_HEADER in older.text


def test_an_older_frame_in_prompt_form_drops_its_element_list():
    history = History().play("FF")
    older = _frames(_view(history.blocks, frames=1, older_frames="prompt"))[0]
    assert older == replace(
        history.blocks[0], text="Frame 1 of the form.", image=None, folded=True
    )


@pytest.mark.parametrize("older_frames", OLDER_FORMS)
def test_every_older_frame_is_folded_and_the_newest_are_not(older_frames):
    history = History().play("FFF")
    frames = _frames(_view(history.blocks, frames=2, older_frames=older_frames))
    assert [frame.folded for frame in frames] == [True, False, False]


def test_outputs_and_feedback_stay_where_they_were_accepted():
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


def test_a_reply_whose_action_names_no_sent_frame_stays_in_the_view():
    history = History().play("F")
    history.reply(based_on=None)
    reply = _view(history.blocks).items[-1].block
    assert (reply.id, reply.based_on) == ("g1.output", None)


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
    older_frames = OLDER_FORMS[seed % len(OLDER_FORMS)]
    retained = rng.randint(1, 25)
    history = History().play(_script(rng, 160))
    previous = None
    for count in range(len(history.blocks) + 1):
        view = _view(history.blocks[:count], frames=retained, older_frames=older_frames)
        blocks = [item.block for item in view.items[1:]]
        assert [block.id for block in blocks] == [block.id for block in history.blocks[:count]]
        frames = [block for block in blocks if block.kind == "frame"]
        with_screenshot = [frame.id for frame in frames if frame.image is not None]
        assert with_screenshot == [frame.id for frame in frames[-retained:]]
        assert [b for b in blocks if b.kind != "frame"] == [
            b for b in history.blocks[:count] if b.kind != "frame"
        ]
        if previous is not None:
            retired = {frame.id for frame in frames if frame.image is None}
            for before, after in zip(previous, blocks):
                # A block changes once, when its frame leaves the newest ones.
                assert after == before or (before.image is not None and after.id in retired)
        previous = blocks


SMALL_WINDOW = replace(POLICY, turns=Retention(keep=3, block=2))


def _kept(view, kind=None):
    return [
        item.block.id
        for item in view.items
        if item.block.kind not in ("system_instruction", "action_ledger")
        and (kind is None or item.block.kind == kind)
    ]


def _ledger(view):
    return next((item for item in view.items if item.block.kind == "action_ledger"), None)


def test_all_text_keeps_every_reply_past_the_turn_window():
    history = History().play("FR" * 61)
    view = _view(history.blocks, actions=history.actions)
    assert len(_kept(view, "model_output")) == 61
    assert _ledger(view) is None


@pytest.mark.parametrize(
    ("replies", "kept"), [(50, 50), (51, 41), (60, 50), (61, 41), (75, 45)]
)
def test_the_archive_cycles_replies_as_the_turn_window_cycles_turns(replies, kept):
    # The maintained window keeps 41 to 50 turns (entity/config.py
    # effective_detailed_history_policy).
    assert POLICY.turns == Retention(keep=50, block=10)
    history = History().play("FR" * replies)
    view = _view(history.blocks, replies="archive", actions=history.actions)
    assert _kept(view, "model_output") == [
        f"g{n}.output" for n in range(replies - kept + 1, replies + 1)
    ]


def test_archived_replies_leave_with_their_feedback_and_the_older_frames_before_them():
    history = History().play("FRFRBFRBFR")
    view = _view(
        history.blocks, frames=2, replies="archive", actions=history.actions, policy=SMALL_WINDOW
    )
    assert _kept(view) == ["f3.frame", "g3.output", "f4.frame", "g4.output"]
    ledger = _ledger(view)
    assert view.items[1] is ledger
    assert ledger.intent == Intent(POLICY.ledger_role, "after_head")
    archived = history.actions["g1.output"] + history.actions["g2.output"]
    assert (ledger.block.text, ledger.block.revision) == (executed_action_ledger(archived), 2)
    # The fourth reply took the window past three replies.
    assert ledger.block.time.wall_ns == 2_004


def test_the_ledger_keeps_its_time_while_the_replies_archived_after_have_no_actions():
    history = History().play("FRFR")
    for _ in range(4):
        history.play("FR")
        history.actions[history.blocks[-1].id] = ()
    view = _view(
        history.blocks, frames=2, replies="archive", actions=history.actions, policy=SMALL_WINDOW
    )
    # Six replies archive four; the fourth reply archived the two that had actions.
    ledger = _ledger(view).block
    assert _kept(view, "model_output") == ["g5.output", "g6.output"]
    archived = history.actions["g1.output"] + history.actions["g2.output"]
    assert (ledger.text, ledger.revision) == (executed_action_ledger(archived), 2)
    assert ledger.time.wall_ns == 2_004


def test_there_is_no_ledger_while_no_archived_reply_has_actions():
    history = History().play("FR" * 4)
    for reply in ("g1.output", "g2.output"):
        history.actions[reply] = ()
    view = _view(
        history.blocks, frames=2, replies="archive", actions=history.actions, policy=SMALL_WINDOW
    )
    assert _kept(view, "model_output") == ["g3.output", "g4.output"]
    assert _ledger(view) is None


def test_an_older_frame_accepted_after_the_last_archived_reply_stays():
    history = History().play("FRFRBFRBFR")
    view = _view(
        history.blocks, frames=1, replies="archive", actions=history.actions, policy=SMALL_WINDOW
    )
    assert _kept(view) == ["f3.frame", "g3.output", "f4.frame", "g4.output"]
    assert [block.folded for block in _frames(view)] == [True, False]


def test_an_archive_takes_an_older_frame_a_kept_reply_is_based_on():
    # Replies outrun frames: the third and fourth answer frame 2, pushed before the second reply
    # that the archive takes, so the view opens with a reply whose frame has left.
    history = History().play("FFRRRRF")
    view = _view(
        history.blocks, frames=1, replies="archive", actions=history.actions, policy=SMALL_WINDOW
    )
    assert _kept(view) == ["g3.output", "g4.output", "f3.frame"]
    assert view.items[2].block.based_on == "o2"


def test_feedback_that_arrives_after_its_reply_was_archived_leaves_with_it():
    history = History().play("FRFRFRFRBBB")
    view = _view(
        history.blocks, frames=4, replies="archive", actions=history.actions, policy=SMALL_WINDOW
    )
    assert _kept(view, "model_output") == ["g3.output", "g4.output"]
    assert _kept(view, "execution_feedback") == ["g3.feedback"]


def test_the_newest_frames_are_never_archived():
    history = History().play("FFFFFRRRR")
    view = _view(
        history.blocks,
        frames=5,
        replies="archive",
        actions=history.actions,
        policy=replace(POLICY, turns=Retention(keep=1, block=1)),
    )
    assert _kept(view) == [f"f{n}.frame" for n in range(1, 6)] + ["g4.output"]
    assert all(frame.image is not None for frame in _frames(view))


@pytest.mark.parametrize("seed", range(6))
def test_each_longer_history_only_adds_to_the_archive(seed):
    rng = random.Random(seed)
    keep = rng.randint(2, 8)
    policy = replace(POLICY, turns=Retention(keep=keep, block=rng.randint(1, keep)))
    retained = rng.randint(1, 12)
    history = History().play(_script(rng, 160))
    archived: set[str] = set()
    ledger_text = ""
    for count in range(len(history.blocks) + 1):
        accepted = history.blocks[:count]
        view = _view(
            accepted,
            frames=retained,
            older_frames=OLDER_FORMS[seed % len(OLDER_FORMS)],
            replies="archive",
            actions=history.actions,
            policy=policy,
        )
        kept = _kept(view)
        assert not archived & set(kept)
        archived |= {block.id for block in accepted} - set(kept)
        newest = [block.id for block in accepted if block.kind == "frame"][-retained:]
        assert [frame.id for frame in _frames(view) if frame.image is not None] == newest
        outputs = [block.id for block in accepted if block.kind == "model_output"]
        replies, kept_replies = len(outputs), len(_kept(view, "model_output"))
        if replies > policy.turns.keep:
            assert policy.turns.keep - policy.turns.block < kept_replies <= policy.turns.keep
        else:
            assert kept_replies == replies
        # The oldest replies leave with their feedback and the older frames accepted before
        # the last of them.
        gone = set(outputs[: replies - kept_replies])
        last = max((n for n, block in enumerate(accepted) if block.id in gone), default=-1)
        assert kept == [
            block.id
            for n, block in enumerate(accepted)
            if block.id not in gone
            and block.attached_to not in gone
            and not (block.kind == "frame" and block.id not in newest and n < last)
        ]
        ledger = _ledger(view)
        text = "" if ledger is None else ledger.block.text
        assert text.startswith(ledger_text)
        assert len(text.splitlines()) == replies - kept_replies
        ledger_text = text
