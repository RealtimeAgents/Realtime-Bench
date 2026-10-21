"""The OpenAI Realtime session core, driven without a WebSocket.

History views come from the request builder the runtime uses (harnesses/default/request.py), and
native-streaming views from the streaming history view (harnesses/streaming/history.py); server
events are parsed the way the SDK's RealtimeConnection.recv parses them
(openai/resources/realtime/realtime.py).
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any, cast

import pytest
from openai._models import construct_type_unchecked
from openai.types.realtime import RealtimeServerEvent

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.model import CanonicalTurn, ProviderObservation
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.frames import FrameTable
from realtime_framework.harnesses.default.request import build_request, ledger_block, system_block
from realtime_framework.harnesses.streaming.history import (
    feedback_block,
    frame_block,
    history_view,
    output_block,
)
from realtime_framework.providers.openai_realtime_session import (
    ROLES,
    Create,
    Delete,
    Delivery,
    Push,
    RealtimeSessionCore,
)

POLICY = current_context_policy().for_path("openai_realtime", {}, None)
TEXTS = POLICY.texts
SYSTEM = system_block("system instruction", None)
ANSWER = AnswerRequirements("text_json", "standard_vlm")
CALL = CallParameters("gpt-realtime-2", "low", None)


def server(**event):
    return construct_type_unchecked(value=event, type_=cast(Any, RealtimeServerEvent))


def done(item_id: str, previous_item_id: str | None = None):
    event = {"type": "conversation.item.done", "item": {"id": item_id}}
    if previous_item_id is not None:
        event["previous_item_id"] = previous_item_id
    return server(**event)


def deleted(item_id: str):
    return server(type="conversation.item.deleted", item_id=item_id)


def error(event_id: str | None):
    body = {"type": "invalid_request_error", "message": "refused", "event_id": event_id}
    return server(type="error", error=body)


def turn(k: int, output: str = "", feedback: str = "") -> CanonicalTurn:
    observation = ProviderObservation(
        observation=Observation(b"", float(k), k, {"observation_id": f"obs-{k}"}),
        prompt=f"frame {k}",
        image_bytes=f"png-{k}".encode(),
        mime_type="image/png",
        image_detail="auto",
        geometry={},
        feedback_prefix=feedback,
    )
    return CanonicalTurn(observation=observation, output=output, decision=k)


def view(turns, ledger: str = ""):
    """The history view the runtime hands the driver: the retained turns and the ledger."""
    return build_request(POLICY, SYSTEM, tuple(turns), ledger_block(ledger, 1, None), ANSWER, CALL)


def new_core() -> RealtimeSessionCore:
    numbers = itertools.count(1)
    return RealtimeSessionCore(lambda kind: f"rtb_{kind}_{next(numbers)}")


def sent(core: RealtimeSessionCore, operations) -> list:
    for operation in operations:
        core.sending(operation)
        core.sent(operation)
    return list(operations)


def opened(turns, ledger: str = "") -> RealtimeSessionCore:
    """A core that replayed these turns, every item confirmed."""
    core = new_core()
    for operation in sent(core, core.replay(view(turns, ledger))):
        core.receive(done(operation.item_id))
    return core


def shape(operations) -> list[tuple]:
    """Each operation as (kind, item id or role, anchor, has an event id)."""
    return [
        ("delete", op.item_id, None, op.event_id is not None)
        if isinstance(op, Delete)
        else ("create", op.role, op.previous_item_id, op.event_id is not None)
        for op in operations
    ]


def test_replay_rebuilds_the_retained_history():
    """A new connection gets the ledger, then each frame's user item, folded ones as the
    placeholder without a screenshot, and each non-empty output as an assistant item; nothing is
    anchored or named, so nothing is waited for."""
    turns = [turn(0, "a0"), turn(1, "", feedback="EXECUTION RESULT: x"), turn(2, "a2")]
    turns[0].fold_visual()
    core = new_core()
    operations = core.replay(view(turns, ledger="L"))
    assert all(
        isinstance(op, Create) and op.previous_item_id is op.event_id is None for op in operations
    )
    assert [(op.role, op.text, op.image is not None) for op in operations] == [
        (ROLES[POLICY.ledger_role], TEXTS.ledger_header + "\nL", False),
        ("user", TEXTS.folded_visual, False),
        ("assistant", "a0", False),
        ("user", "EXECUTION RESULT: x\nframe 1", True),
        ("user", "frame 2", True),
        ("assistant", "a2", False),
    ]
    ledger, user0, assistant0, user1, user2, assistant2 = operations
    assert (core.ledger_item_id, core.ledger_text) == (ledger.item_id, "L")
    assert {
        k: (f.user_item_id, f.folded, f.assistant_item_ids) for k, f in core.frames.items()
    } == {
        "d0.frame": (user0.item_id, True, [assistant0.item_id]),
        "d1.frame": (user1.item_id, False, []),
        "d2.frame": (user2.item_id, False, [assistant2.item_id]),
    }
    assert new_core().replay(None) == []


def test_observe_adds_the_newest_frame():
    core = opened([turn(0, "a0")])
    user = core.observe(view([turn(0, "a0"), turn(1, feedback="EXECUTION RESULT: y")]))
    assert (user.role, user.text, user.image.data) == (
        "user",
        "EXECUTION RESULT: y\nframe 1",
        b"png-1",
    )
    assert user.previous_item_id is user.event_id is None
    assert core.frames["d1.frame"].user_item_id == user.item_id
    assert core.frames["d1.frame"].assistant_item_ids == []


# The policy with each action's execution feedback as its own block (context policy item 3).
SEPARATE = replace(POLICY, feedback="separate_message")


def fed(k: int, output: str = "", feedback: str = "") -> CanonicalTurn:
    """Decision k with the execution feedback on its action."""
    decided = turn(k, output)
    decided.execution_feedback = feedback
    return decided


def separate_view(turns, ledger: str = ""):
    return build_request(
        SEPARATE, SYSTEM, tuple(turns), ledger_block(ledger, 1, None), ANSWER, CALL
    )


def test_replay_puts_each_feedback_item_after_its_frame_s_output():
    turns = [fed(0, "a0", "F0"), fed(1, "", "F1"), turn(2)]
    core = new_core()
    operations = core.replay(separate_view(turns))
    assert [(op.role, op.text) for op in operations] == [
        ("user", "frame 0"),
        ("assistant", "a0"),
        ("user", "F0"),
        ("user", "frame 1"),
        ("user", "F1"),
        ("user", "frame 2"),
    ]
    _, assistant0, feedback0, _, feedback1, _ = operations
    assert feedback0.item_id.startswith("rtb_feedback_")
    assert {k: (f.assistant_item_ids, f.feedback_item_id) for k, f in core.frames.items()} == {
        "d0.frame": ([assistant0.item_id], feedback0.item_id),
        "d1.frame": ([], feedback1.item_id),
        "d2.frame": ([], None),
    }


def test_the_feedback_on_the_last_action_goes_ahead_of_the_next_frame():
    """The feedback reaches the history once the action ran, after its frame was sent: the next
    observation sends it first, at the end of the conversation, and only once."""
    core = new_core()
    sent(core, core.replay(separate_view([turn(0, "a0")])))
    history = separate_view([fed(0, "a0", "F0"), turn(1)])
    (feedback,) = core.new_feedback(history)
    user = core.observe(history)
    assert shape([feedback, user]) == [("create", "user", None, False)] * 2
    assert (feedback.text, user.text) == ("F0", "frame 1")
    assert core.frames["d0.frame"].feedback_item_id == feedback.item_id
    assert core.new_feedback(separate_view([fed(0, "a0", "F0"), turn(1, "a1"), turn(2)])) == []


def test_an_edit_deletes_an_archived_frame_s_feedback_and_anchors_after_a_kept_one():
    turns = [fed(0, "a0", "F0"), fed(1, "a1", "F1"), turn(2, "a2")]
    core = new_core()
    for operation in sent(core, core.replay(separate_view(turns))):
        core.receive(done(operation.item_id))
    held = {k: (f.user_item_id, f.feedback_item_id) for k, f in core.frames.items()}
    turns[2].fold_visual()
    batch = core.begin_edit(separate_view(turns[1:], ledger="L"))
    assert ("delete", held["d0.frame"][1], None, True) in shape(batch.operations)
    # The folded frame 2 goes back after frame 1's feedback item.
    *_, delete2, create2 = shape(batch.operations)
    assert delete2 == ("delete", held["d2.frame"][0], None, True)
    assert create2 == ("create", "user", held["d1.frame"][1], True)


def test_each_change_in_delivery_is_heard_with_its_block():
    """on_delivery hears what the event log records: sent once a write returned, confirmed by
    the first acknowledgement (one that comes before the write returns makes "sent" moot), and
    unknown once a write failed."""
    heard = []
    numbers = itertools.count(1)
    core = RealtimeSessionCore(
        lambda kind: f"rtb_{kind}_{next(numbers)}",
        lambda state, contents: heard.append((state, contents)),
    )
    user, assistant = core.replay(view([turn(0, "a0")]))
    sent(core, [user])
    core.sending(assistant)
    core.receive(done(assistant.item_id))
    core.sent(assistant)
    core.receive(done(user.item_id))
    core.receive(server(type="conversation.item.created", item={"id": user.item_id}))
    frame = core.observe(view([turn(0, "a0"), turn(1)]))
    core.sending(frame)
    core.send_failed(frame)
    # A write that raised after the server already confirmed it stays confirmed.
    later = core.observe(view([turn(0, "a0"), turn(1), turn(2)]))
    core.sending(later)
    core.receive(done(later.item_id))
    core.send_failed(later)
    assert heard == [
        ("sent", [(user.key, "d0.frame")]),
        ("confirmed", [(assistant.key, "d0.output")]),
        ("confirmed", [(user.key, "d0.frame")]),
        ("unknown", [(frame.key, "d1.frame")]),
        ("confirmed", [(later.key, "d2.frame")]),
    ]


def test_fold_in_place_waits_for_every_ack_in_any_order():
    turns = [turn(0, "a0"), turn(1, "a1"), turn(2, "a2")]
    core = opened(turns)
    before = {k: (f.user_item_id, f.assistant_item_ids) for k, f in core.frames.items()}
    turns[0].fold_visual()
    turns[1].fold_visual()
    history = view(turns)
    assert core.in_place_problem(history) is None

    batch = core.begin_edit(history)
    delete0, create0, delete1, create1 = sent(core, batch.operations)
    # Each folded user item is deleted, then the placeholder is created where it was: after the
    # previous turn's output, or at the root for the first turn.
    assert shape(batch.operations) == [
        ("delete", before["d0.frame"][0], None, True),
        ("create", "user", "root", True),
        ("delete", before["d1.frame"][0], None, True),
        ("create", "user", before["d0.frame"][1][-1], True),
    ]
    assert (create0.text, create0.image) == (TEXTS.folded_visual, None)
    assert (batch.visuals_folded, batch.prompts_edited) == (2, 2)

    # Acknowledgements arrive in any order; conversation.item.added does not confirm a create.
    for event in (deleted(delete1.item_id), done(create1.item_id), deleted(delete0.item_id)):
        core.receive(event)
        assert not core.edit_answered
    core.receive(server(type="conversation.item.added", item={"id": create0.item_id}))
    assert not core.edit_answered
    assert core.delivery[create0.key] == Delivery.SENT
    core.receive(server(type="conversation.item.created", item={"id": create0.item_id}))
    assert core.edit_answered

    assert core.finish_edit() is batch
    assert all(core.delivery[op.key] == Delivery.ACKNOWLEDGED for op in batch.operations)
    assert {k: (f.user_item_id, f.folded) for k, f in core.frames.items()} == {
        "d0.frame": (create0.item_id, True),
        "d1.frame": (create1.item_id, True),
        "d2.frame": (before["d2.frame"][0], False),
    }
    # The assistant items stay where they were.
    assert {k: f.assistant_item_ids for k, f in core.frames.items()} == {
        k: ids for k, (_, ids) in before.items()
    }


def test_archive_replaces_the_ledger_at_the_root():
    turns = [turn(k, f"a{k}") for k in range(4)]
    core = opened(turns)
    items = {k: [f.user_item_id, *f.assistant_item_ids] for k, f in core.frames.items()}

    # Archiving turns 0 and 1 deletes both of their items and creates the first ledger.
    batch = core.begin_edit(view(turns[2:], ledger="L1"))
    operations = sent(core, batch.operations)
    archived = items["d0.frame"] + items["d1.frame"]
    assert shape(operations) == [
        *(("delete", item_id, None, True) for item_id in archived),
        ("create", ROLES[POLICY.ledger_role], "root", True),
    ]
    first_ledger = operations[-1]
    assert first_ledger.text == TEXTS.ledger_header + "\nL1"
    for op in reversed(operations):
        core.receive(deleted(op.item_id) if isinstance(op, Delete) else done(op.item_id))
    core.finish_edit()
    assert (core.ledger_item_id, core.ledger_text) == (first_ledger.item_id, "L1")
    assert list(core.frames) == ["d2.frame", "d3.frame"]
    assert batch.created == {first_ledger.item_id} and batch.deleted == set(archived)

    # The next archive replaces the ledger; a fold in the same edit anchors on the new ledger.
    turns[3].fold_visual()
    batch = core.begin_edit(view(turns[3:], ledger="L1\nL2"))
    operations = sent(core, batch.operations)
    new_ledger = operations[3]
    assert shape(operations) == [
        ("delete", items["d2.frame"][0], None, True),
        ("delete", items["d2.frame"][1], None, True),
        ("delete", first_ledger.item_id, None, True),
        ("create", ROLES[POLICY.ledger_role], "root", True),
        ("delete", items["d3.frame"][0], None, True),
        ("create", "user", new_ledger.item_id, True),
    ]
    for op in operations:
        core.receive(deleted(op.item_id) if isinstance(op, Delete) else done(op.item_id))
    core.finish_edit()
    assert (core.ledger_item_id, core.ledger_text) == (new_ledger.item_id, "L1\nL2")


@pytest.mark.parametrize("named", ["first", "last", "none"])
def test_an_error_about_the_edit_fails_it(named):
    """An error that names one of the edit's client events, or names none, fails the edit; the
    mirror keeps the session as it was before the edit."""
    turns = [turn(0, "a0"), turn(1, "a1")]
    core = opened(turns)
    before = {k: f.user_item_id for k, f in core.frames.items()}
    turns[0].fold_visual()
    batch = core.begin_edit(view(turns))
    first, last = sent(core, batch.operations)
    core.receive(deleted(first.item_id))
    event_id = {"first": first.event_id, "last": last.event_id, "none": None}[named]
    core.receive(error(event_id))
    assert core.edit_answered
    with pytest.raises(RuntimeError, match=r"^Realtime history edit failed: RealtimeError\("):
        core.finish_edit()
    assert {k: f.user_item_id for k, f in core.frames.items()} == before
    assert not core.frames["d0.frame"].folded


def test_an_error_about_another_event_leaves_the_edit_waiting():
    turns = [turn(0, "a0"), turn(1, "a1")]
    core = opened(turns)
    turns[0].fold_visual()
    batch = core.begin_edit(view(turns))
    delete0, create0 = sent(core, batch.operations)
    core.receive(error("rtb_cancel_elsewhere"))
    core.receive(deleted(delete0.item_id))
    assert not core.edit_answered
    core.receive(done(create0.item_id))
    # A late error about the confirmed edit changes nothing.
    core.receive(error(None))
    assert core.finish_edit() is batch
    assert core.frames["d0.frame"].user_item_id == create0.item_id


def test_an_edit_the_mirror_cannot_reach_is_refused_before_any_send():
    turns = [turn(0, "a0"), turn(1, "a1")]
    core = opened(turns[:1])
    assert (
        core.in_place_problem(view(turns)) == "retained turn is missing its provider user item ID"
    )
    turns[0].fold_visual()
    folded = opened(turns[:1])
    assert (
        folded.in_place_problem(view([turn(0, "a0")]))
        == "provider visual state cannot be unfolded in place"
    )


@pytest.mark.parametrize("output", ["the whole output", ""])
def test_repair_replaces_the_output_items_of_a_cut_response(output):
    """After a cancelled or incomplete response the server's output items are deleted and the
    collected output, if any, follows as one assistant item. None of it is waited for."""
    core = opened([turn(0, "a0")])
    assert not core.receive(done("srv_extra"))  # an item the core never planned is not recorded
    core.frames["d0.frame"].assistant_item_ids = ["srv_1", "srv_2"]
    operations = core.repair(view([turn(0, output)]), ["srv_1", "srv_2"])
    expected = [("delete", "srv_1", None, False), ("delete", "srv_2", None, False)]
    if output:
        expected.append(("create", "assistant", None, False))
    assert shape(operations) == expected
    assert core.frames["d0.frame"].assistant_item_ids == [
        op.item_id for op in operations if isinstance(op, Create)
    ]
    assert ("create", "srv_extra") not in core.delivery


def response_events(status: str = "completed") -> list:
    """One response as the server streams it, with one output item."""
    item = {"id": "srv_1", "type": "message", "role": "assistant"}
    return [
        server(type="response.created", response={"id": "resp_1"}),
        server(type="response.output_item.added", response_id="resp_1", item=item),
        server(type="response.output_text.delta", response_id="resp_1", delta="{}"),
        server(type="response.done", response={"id": "resp_1", "status": status, "output": [item]}),
    ]


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_response_events_join_their_generation(status):
    core = opened([turn(0, "a0"), turn(1)])
    generation = core.start_generation("d1.frame")
    created, added, text, finished = response_events(status)
    for event in (created, done("srv_1"), added, text, finished):
        core.receive(event)
    # Acknowledgements are the core's; the response's events are the generation's.
    assert list(generation.events) == [created, added, text, finished]
    assert (generation.response_id, generation.output_item_ids) == ("resp_1", ["srv_1"])
    # A failed response leaves the frame without output items; a turn retry rebuilds it.
    expected = [] if status == "failed" else ["srv_1"]
    assert core.frames["d1.frame"].assistant_item_ids == expected
    # After response.done no response is in progress, so no state takes a late event.
    assert not core.receive(server(type="response.output_text.delta", delta="late"))
    assert len(generation.events) == 4
    assert (generation.number, core.start_generation("d1.frame").number) == (1, 2)


def test_a_late_event_of_an_earlier_response_stays_with_it():
    """A response.done of the first response that arrives after a second generation registered
    ends the first one; the second waits for its own response.created."""
    core = opened([turn(0, "a0"), turn(1, "a1"), turn(2)])
    first = core.start_generation("d1.frame")
    core.receive(server(type="response.created", response={"id": "resp_1"}))
    second = core.start_generation("d2.frame")
    output = [{"id": "srv_old"}]
    late = server(
        type="response.done", response={"id": "resp_1", "status": "cancelled", "output": output}
    )
    assert core.receive(late)
    assert [event.type for event in first.events] == ["response.created", "response.done"]
    assert list(second.events) == []
    assert core.frames["d1.frame"].assistant_item_ids == ["srv_old"]
    assert core.frames["d2.frame"].assistant_item_ids == []
    own = server(type="response.created", response={"id": "resp_2"})
    assert core.receive(own)
    assert (list(second.events), second.response_id) == ([own], "resp_2")
    # An ended response's events go nowhere.
    assert not core.receive(
        server(type="response.output_text.delta", response_id="resp_1", delta="x")
    )


def test_two_responses_in_flight_each_get_their_own_answer():
    """Review finding 中-1, at the core: two generations on one connection, both requested before
    the server answers either. Each response.created goes to the earliest generation still
    waiting for one, and every later event goes by its response id, so the answer to the first
    request never reaches the second generation.

    The runtime reached this when a decision abandoned by abort() rebuilt the session, marked the
    runtime ready and asked for its response on the connection the next decision then reused.
    That can no longer happen. abort() sets the abandoned generation's cancel event, which stops
    the rebuild before it connects (RequestRetry, entity/model.py), and _run_turn checks it before
    it sends the observation (harnesses/default/context.py). A decision abandoned after
    that check can still start its response, but only on the connection abort() closes: the next
    decision waits for that cleanup and opens a new connection (ensure_ready), whose core never
    observed the abandoned frame, so start_generation there fails on it. Hence the core test.
    """
    core = opened([turn(0, "a0"), turn(1), turn(2)])
    first, second = core.start_generation("d1.frame"), core.start_generation("d2.frame")
    item_1, item_2 = {"id": "srv_1"}, {"id": "srv_2"}
    events = [
        server(type="response.created", response={"id": "resp_1"}),
        server(type="response.created", response={"id": "resp_2"}),
        server(type="response.output_text.delta", response_id="resp_2", delta="two"),
        server(type="response.output_text.delta", response_id="resp_1", delta="one"),
        server(
            type="response.done",
            response={"id": "resp_1", "status": "completed", "output": [item_1]},
        ),
        server(
            type="response.done",
            response={"id": "resp_2", "status": "completed", "output": [item_2]},
        ),
    ]
    assert all(core.receive(event) for event in events)
    assert (first.response_id, second.response_id) == ("resp_1", "resp_2")
    assert list(first.events) == [events[0], events[3], events[4]]
    assert list(second.events) == [events[1], events[2], events[5]]
    assert core.frames["d1.frame"].assistant_item_ids == ["srv_1"]
    assert core.frames["d2.frame"].assistant_item_ids == ["srv_2"]


def test_an_error_goes_to_the_response_it_cancels():
    """With two responses in progress, an error echoing one's cancel goes to that one; an error
    that names no client event goes to both."""
    core = opened([turn(0, "a0"), turn(1)])
    first, second = core.start_generation("d0.frame"), core.start_generation("d1.frame")
    first.cancel_event_id = "rtb_cancel_1"
    cancel_race, unnamed = error("rtb_cancel_1"), error(None)
    core.receive(cancel_race)
    core.receive(unnamed)
    assert list(first.events) == [cancel_race, unnamed]
    assert list(second.events) == [unnamed]


@pytest.mark.parametrize(
    ("named", "fails_edit", "reaches_generation"),
    [("edit", True, False), ("none", True, True), ("cancel", False, True)],
)
def test_an_error_goes_to_the_state_it_names(named, fails_edit, reaches_generation):
    """With an edit waiting and a response in progress: an error about the edit fails only the
    edit, one about another client event (a cancel) goes only to the response, and one that names
    none goes to both."""
    turns = [turn(0, "a0"), turn(1)]
    core = opened(turns)
    generation = core.start_generation("d1.frame")
    turns[0].fold_visual()
    batch = core.begin_edit(view(turns))
    sent(core, batch.operations)
    event_id = {"edit": batch.operations[1].event_id, "none": None, "cancel": "rtb_cancel_1"}
    event = error(event_id[named])
    core.receive(event)
    assert core.edit_answered == fails_edit
    assert (list(generation.events) == [event]) == reaches_generation


@pytest.mark.parametrize("response_first", [True, False])
def test_an_edit_and_a_response_both_reach_the_mirror(response_first):
    """A response that ends while an edit waits, or after it is applied, still gives the frame it
    answered its output items, and the edit still replaces the folded user item."""
    turns = [turn(0, "a0"), turn(1)]
    core = opened(turns)
    generation = core.start_generation("d1.frame")
    turns[0].fold_visual()
    batch = core.begin_edit(view(turns))
    delete0, create0 = sent(core, batch.operations)
    events = response_events()
    if response_first:
        for event in events:
            core.receive(event)
    core.receive(deleted(delete0.item_id))
    core.receive(done(create0.item_id))
    core.finish_edit()
    if not response_first:
        for event in events:
            core.receive(event)
    assert len(generation.events) == 4
    assert (core.frames["d0.frame"].user_item_id, core.frames["d0.frame"].folded) == (
        create0.item_id,
        True,
    )
    assert core.frames["d1.frame"].assistant_item_ids == ["srv_1"]


def test_delivery_is_unknown_while_written_then_sent_then_acknowledged():
    core = new_core()
    acknowledged, early, failed, written = core.replay(view([turn(0, "a0"), turn(1, "a1")]))
    for operation in (acknowledged, early, failed, written):
        core.sending(operation)
    core.sent(acknowledged)
    core.receive(done(acknowledged.item_id))
    # Another waiter read the acknowledgement before this write returned.
    core.receive(done(early.item_id))
    core.sent(early)
    core.sent(written)
    # The write of ``failed`` never returned.
    assert [core.delivery[op.key] for op in (acknowledged, early, failed, written)] == [
        Delivery.ACKNOWLEDGED,
        Delivery.ACKNOWLEDGED,
        Delivery.UNKNOWN,
        Delivery.SENT,
    ]


# ---- native streaming ------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_a_streaming_reply_answers_no_frame(status):
    """A native-streaming reply answers the session as it is, not a frame: its events join it as
    a turn's do, and its output items stay the server's, kept under no frame."""
    core = opened([turn(0, "a0")])
    assistant = core.frames["d0.frame"].assistant_item_ids
    reply = core.start_reply()
    events = response_events(status)
    assert all(core.receive(event) for event in events)
    assert list(reply.events) == events
    assert (reply.answered, reply.output_item_ids) == (None, ["srv_1"])
    assert core.frames["d0.frame"].assistant_item_ids == assistant


def stream(script: str) -> list:
    """Blocks of a native-streaming session in the order the agent accepts them. F: a frame
    pushed, frame n showing turn n's prompt and screenshot; R: reply d; B: the feedback on the
    oldest reply without one."""
    table, blocks, replies, reported = FrameTable(), [], 0, 0
    for step in script:
        if step == "F":
            observation = turn(sum(block.kind == "frame" for block in blocks) + 1).observation
            record = table.register(observation.observation_id, 0.0, {}, None)
            blocks.append(frame_block(record, observation))
        elif step == "R":
            replies += 1
            blocks.append(output_block(replies, f"r{replies}", None, None))
        else:
            reported += 1
            blocks.append(feedback_block(reported, None, f"fb{reported}", None))
    return blocks


def streaming_view(blocks, frames: int = 2, feedback: str = "separate_message"):
    """The streaming history view of the blocks: the frames older than the newest ``frames``
    folded to the turn-based placeholder (decision 23), every reply and feedback kept (decision
    4.6), and the feedback where the policy's ``feedback`` places it."""
    return history_view(
        blocks,
        policy=replace(POLICY, feedback=feedback),
        system=SYSTEM,
        retained_frames=frames,
        answer=ANSWER,
        call=CALL,
    )


def test_a_new_connection_replays_the_streaming_view():
    """4.14: a rebuilt session holds the streaming view, one item per block in order: the frames
    older than the newest K as the placeholder without a screenshot, each reply as an assistant
    item (an empty one as none), each feedback as a user item. Nothing is anchored; frames and
    feedback name their client events as pushes do."""
    blocks = stream("FFRFBFR")
    blocks[-1] = replace(blocks[-1], text="")
    core = new_core()
    operations = core.replay_streaming(streaming_view(blocks))
    assert all(op.previous_item_id is None for op in operations)
    assert [op.event_id is not None for op in operations] == [True, True, False, True, True, True]
    assert [(op.role, op.text, op.image is not None) for op in operations] == [
        ("user", TEXTS.folded_visual, False),
        ("user", TEXTS.folded_visual, False),
        ("assistant", "r1", False),
        ("user", "frame 3", True),
        ("user", "fb1", False),
        ("user", "frame 4", True),
    ]
    f1, f2, _, f3, b1, f4 = (op.item_id for op in operations)
    assert {k: (f.user_item_id, f.text, f.folded) for k, f in core.frames.items()} == {
        "f1.frame": (f1, TEXTS.folded_visual, True),
        "f2.frame": (f2, TEXTS.folded_visual, True),
        "f3.frame": (f3, "frame 3", False),
        "d1.feedback": (b1, "fb1", False),
        "f4.frame": (f4, "frame 4", False),
    }
    assert new_core().replay_streaming(None) == []


def test_separate_feedback_is_the_same_item_in_native_streaming_as_turn_based():
    """Under separate_message an action's feedback is a user item of its own right after the
    reply it reports, with the feedback's text, whether a turn-based replay or a native-streaming
    one writes the session: both hold the same items."""
    turn_based = new_core().replay(separate_view([fed(1, "r1", "fb1"), turn(2)]))
    streaming = new_core().replay_streaming(streaming_view(stream("FRBF"), frames=5))
    assert [(op.role, op.text) for op in turn_based] == [
        ("user", "frame 1"),
        ("assistant", "r1"),
        ("user", "fb1"),
        ("user", "frame 2"),
    ]
    items = [[(op.role, op.text, op.image) for op in ops] for ops in (turn_based, streaming)]
    assert items[0] == items[1]


def items_of(core: RealtimeSessionCore) -> dict[str, str]:
    """Each frame's and feedback's item in the session."""
    return {block_id: sent.user_item_id for block_id, sent in core.frames.items()}


class Server:
    """The conversation as the Realtime server keeps it: a create without an anchor joins the
    end, one anchored at "root" goes first and one anchored on an item right after it; a delete
    takes its item out; a reply's item joins the end once the server makes it. A create's answer
    names the item before it; a create of an item in ``refuse`` is answered with an error naming
    its client event, and leaves nothing."""

    def __init__(self) -> None:
        self.items: list[str] = []
        self.refuse: set[str] = set()

    def answer(self, operation):
        if isinstance(operation, Delete):
            self.items.remove(operation.item_id)
            return deleted(operation.item_id)
        if operation.item_id in self.refuse:
            return error(operation.event_id)
        anchor = operation.previous_item_id
        if anchor is None:
            self.items.append(operation.item_id)
        else:
            at = 0 if anchor == "root" else self.items.index(anchor) + 1
            self.items.insert(at, operation.item_id)
        index = self.items.index(operation.item_id)
        return done(operation.item_id, self.items[index - 1] if index else None)


class Live:
    """A native-streaming session as its reader sees it (arrived): the core, and the server,
    which answers each operation sent, in order."""

    def __init__(self, blocks=(), frames: int = 2, feedback: str = "separate_message"):
        self.core, self.server, self.replies = new_core(), Server(), 0
        view = streaming_view(list(blocks), frames, feedback) if blocks else None
        self.deliver(self.core.replay_streaming(view))

    def send(self, operations) -> list:
        """Send the operations; return the server's answers, which have not arrived yet."""
        return [self.server.answer(op) for op in sent(self.core, operations)]

    def arrive(self, events) -> None:
        for event in events:
            self.core.arrived(event, 0.0)

    def deliver(self, operations) -> None:
        self.arrive(self.send(operations))

    def reply(self, *, end: bool = True):
        """A reply, whose one item the server makes at the end of the conversation."""
        self.replies += 1
        n = self.replies
        generation = self.core.start_reply()
        self.arrive([server(type="response.created", response={"id": f"resp_{n}"})])
        item = {"id": f"srv_{n}", "type": "message", "role": "assistant"}
        self.server.items.append(item["id"])
        self.arrive([server(type="response.output_item.added", response_id=f"resp_{n}", item=item)])
        if end:
            self.end(generation)
        return generation

    def end(self, generation) -> None:
        output = [{"id": item_id} for item_id in generation.output_item_ids]
        response = {"id": generation.response_id, "status": "completed", "output": output}
        self.arrive([server(type="response.done", response=response)])


def replacements(batch) -> dict[str, str]:
    """Each item a batch deletes and replaces, and the item that takes its place."""
    ops = batch.operations
    return {
        delete.item_id: create.item_id
        for delete, create in itertools.pairwise(ops)
        if isinstance(delete, Delete) and isinstance(create, Create)
    }


def in_place(live: Live, before: list[str], batch) -> bool:
    """Whether the server holds what it held before the batch, each replaced item's replacement
    where the item was."""
    placed = replacements(batch)
    return live.server.items == [placed.get(item, item) for item in before]


def test_a_fold_puts_the_placeholder_where_the_frame_was_on_the_server():
    """Decision 23: a folded frame's placeholder takes its place on the server, after the item
    the server had before it. Feedback pushed while a later reply runs follows that reply's item
    on the server, but comes before the reply in the view; the frame pushed after the feedback
    folds after the feedback, not after the reply."""
    live = Live(stream("F"), frames=5)
    live.reply()
    second = live.reply(end=False)
    live.deliver(live.core.push(streaming_view(stream("FRB"), frames=5)))
    live.end(second)
    for script in ("FRBRF", "FRBRFF"):
        live.deliver(live.core.push(streaming_view(stream(script), frames=5)))
    before = list(live.server.items)
    assert before[1:4] == ["srv_1", "srv_2", live.core.frames["d1.feedback"].user_item_id]

    batch = live.core.queue_edit(streaming_view(stream("FRBRFF"), frames=1))
    live.deliver(batch.operations)
    live.core.finish_edit()
    assert len(replacements(batch)) == 2
    assert in_place(live, before, batch)


def test_frame_prefix_feedback_goes_into_the_frame_after_it():
    """Under the frame_prefix placement the view puts the feedback since the frame before in
    front of the next frame, as a turn-based frame carries it: a rebuild and a push write it into
    that frame's item, which edits leave alone until the frame folds; then the feedback gives way
    to the placeholder with it, after the frame before. The feedback since the newest frame waits
    at the end of the view for the next frame, and nothing of it is written yet."""
    replayed = new_core().replay_streaming(streaming_view(stream("FRFBF"), 3, "frame_prefix"))
    assert [(op.role, op.text) for op in replayed] == [
        ("user", "frame 1"),
        ("assistant", "r1"),
        ("user", "frame 2"),
        ("user", "fb1\nframe 3"),
    ]
    waiting = new_core().replay_streaming(streaming_view(stream("FRFB"), 3, "frame_prefix"))
    assert [op.text for op in waiting] == ["frame 1", "r1", "frame 2"]

    live = Live(stream("FRF"), 3, "frame_prefix")
    core = live.core
    (pushed,) = core.push(streaming_view(stream("FRFBF"), 3, "frame_prefix"))
    assert (pushed.text, core.frames["f3.frame"].text) == ("fb1\nframe 3", "fb1\nframe 3")
    live.deliver([pushed])
    live.deliver(core.push(streaming_view(stream("FRFBFF"), 3, "frame_prefix")))
    item, reply_item = items_of(core), live.server.items[1]
    older = core.queue_edit(streaming_view(stream("FRFBFF"), 2, "frame_prefix"))
    assert shape(older.operations) == [
        ("delete", item["f1.frame"], None, True),
        ("create", "user", "root", True),
        ("delete", item["f2.frame"], None, True),
        ("create", "user", reply_item, True),
    ]
    assert (older.visuals_folded, older.prompts_edited) == (2, 2)
    live.deliver(older.operations)
    core.finish_edit()
    batch = core.queue_edit(streaming_view(stream("FRFBFF"), 1, "frame_prefix"))
    assert shape(batch.operations) == [
        ("delete", item["f3.frame"], None, True),
        ("create", "user", older.operations[3].item_id, True),
    ]
    assert batch.operations[1].text == TEXTS.folded_visual


def test_a_push_names_its_client_event():
    """A pushed block, a frame with its screenshot or an execution feedback, is appended like an
    observed frame, but its create names a client event."""
    core = Live(stream("FR")).core
    (frame,) = core.push(streaming_view(stream("FRF")))
    (feedback,) = core.push(streaming_view(stream("FRFB")))
    assert [(op.role, op.text, op.image is not None) for op in (frame, feedback)] == [
        ("user", "frame 2", True),
        ("user", "fb1", False),
    ]
    assert all(
        op.previous_item_id is None and op.event_id.startswith("rtb_push_")
        for op in (frame, feedback)
    )
    assert (core.pushes[frame.event_id], core.pushes[feedback.event_id]) == (
        Push("f2.frame", frame.item_id),
        Push("d1.feedback", feedback.item_id),
    )
    assert {
        k: (core.frames[k].user_item_id, core.frames[k].text, core.frames[k].folded)
        for k in ("f2.frame", "d1.feedback")
    } == {
        "f2.frame": (frame.item_id, "frame 2", False),
        "d1.feedback": (feedback.item_id, "fb1", False),
    }
    sent(core, [frame, feedback])
    assert core.receive(done(frame.item_id))
    assert (core.delivery[frame.key], core.delivery[feedback.key]) == (
        Delivery.ACKNOWLEDGED,
        Delivery.SENT,
    )


def test_a_push_plans_every_block_the_session_lacks():
    """A push takes the view as a whole: it plans, in the view's order, the item of each frame
    and feedback the session lacks, so the feedback accepted since the frame before goes out
    ahead of the frame. A reply's items are the server's, a block the server refused stays out,
    and a view with nothing new plans nothing."""
    live = Live(stream("FR"))
    core = live.core
    pushed = core.push(streaming_view(stream("FRBF")))
    assert [(op.block, op.text) for op in pushed] == [
        ("d1.feedback", "fb1"),
        ("f2.frame", "frame 2"),
    ]
    live.deliver(pushed)
    assert core.push(streaming_view(stream("FRBF"))) == []
    (refused,) = core.push(streaming_view(stream("FRBFF")))
    live.server.refuse.add(refused.item_id)
    live.deliver([refused])
    assert core.push(streaming_view(stream("FRBFF"))) == []


@pytest.mark.parametrize("frames", [1, 2, 3])
def test_a_streaming_edit_folds_the_frames_older_than_the_newest_k_where_they_were(frames):
    """Decision 23: each frame the view folds, those older than the newest K, gives way to the
    turn-based placeholder where it was: its item is deleted, and the placeholder is created
    after the item the server has before it (here the root, a reply's or a feedback's). Replies
    and feedback stay as they are."""
    blocks = stream("FRFBFRF")
    live = Live(blocks, frames=4)
    core, before, item = live.core, list(live.server.items), items_of(live.core)
    folded = ["f1.frame", "f2.frame", "f3.frame"][: 4 - frames]

    batch = core.queue_edit(streaming_view(blocks, frames))
    deletes = [op.item_id for op in batch.operations if isinstance(op, Delete)]
    assert deletes == [item[block_id] for block_id in folded]
    assert all(
        (op.text, op.image) == (TEXTS.folded_visual, None)
        for op in batch.operations
        if isinstance(op, Create)
    )
    assert (batch.removed, batch.visuals_folded, batch.prompts_edited) == (
        (),
        len(folded),
        len(folded),
    )
    live.deliver(batch.operations)
    assert core.finish_edit() is batch
    assert in_place(live, before, batch)
    assert [k for k, f in core.frames.items() if f.folded] == folded


@pytest.mark.parametrize("items", [2, 0])
def test_a_frame_after_a_reply_folds_after_its_last_item(items):
    """A reply the server made items for: the frame pushed after it folds after its last item,
    where the frame was. After a reply with no item it folds after the block before the reply."""
    live = Live(stream("F"), frames=5)
    core = live.core
    core.start_reply()
    live.arrive([server(type="response.created", response={"id": "resp_1"})])
    output = [{"id": f"srv_{n}", "type": "message", "role": "assistant"} for n in range(items)]
    for item in output:
        live.server.items.append(item["id"])
        live.arrive([server(type="response.output_item.added", response_id="resp_1", item=item)])
    response = {"id": "resp_1", "status": "completed", "output": output}
    live.arrive([server(type="response.done", response=response)])
    blocks = stream("FRFF")
    for n in (3, 4):
        live.deliver(core.push(streaming_view(blocks[:n], frames=5)))
    before = list(live.server.items)

    batch = core.queue_edit(streaming_view(blocks, frames=1))
    anchors = [op.previous_item_id for op in batch.operations if isinstance(op, Create)]
    assert anchors == ["root", "srv_1" if items else batch.operations[1].item_id]
    live.deliver(batch.operations)
    core.finish_edit()
    assert in_place(live, before, batch)


def test_a_frame_pushed_during_a_reply_folds_where_it_was_on_the_server():
    """A frame pushed after the server made a reply's item follows that item on the server,
    though the view, which takes the reply in once it ends, has the frame before the reply. Its
    placeholder goes where the frame was, after the reply's item."""
    live = Live(stream("F"), frames=5)
    core = live.core
    reply = live.reply(end=False)
    live.deliver(core.push(streaming_view(stream("FF"), frames=5)))
    live.end(reply)
    blocks = stream("FFRF")
    live.deliver(core.push(streaming_view(blocks, frames=5)))
    before = list(live.server.items)
    assert before[1] == "srv_1"

    batch = core.queue_edit(streaming_view(blocks, frames=1))
    anchors = [op.previous_item_id for op in batch.operations if isinstance(op, Create)]
    assert anchors == ["root", "srv_1"]
    live.deliver(batch.operations)
    core.finish_edit()
    assert in_place(live, before, batch)


def test_a_turn_based_edit_deletes_every_sent_frame_its_view_lacks():
    """A turn-based edit takes its view as the whole session: a sent frame the view lacks goes,
    however new. Only a native-streaming edit leaves the frames pushed after its view."""
    turns = [turn(0, "a0"), turn(1, "a1"), turn(2, "a2")]
    core = opened(turns)
    newest = [core.frames["d2.frame"].user_item_id, *core.frames["d2.frame"].assistant_item_ids]
    batch = core.begin_edit(view(turns[:2]))
    assert shape(batch.operations) == [("delete", item_id, None, True) for item_id in newest]
    assert batch.removed == ("d2.frame",)


def test_a_streaming_edit_leaves_the_blocks_not_pushed_yet():
    """A view made as a frame is pushed may already hold it: the edit covers the frames the
    session holds, and leaves the new one to its push."""
    blocks = stream("FFF")
    live = Live(blocks[:2], frames=5)
    core = live.core
    first = core.frames["f1.frame"].user_item_id
    view = streaming_view(blocks, frames=2)
    batch = core.queue_edit(view)
    assert shape(batch.operations) == [
        ("delete", first, None, True),
        ("create", "user", "root", True),
    ]
    assert list(batch.frames) == ["f1.frame"]
    live.deliver(batch.operations)
    core.finish_edit()
    assert [op.text for op in core.push(view)] == ["frame 3"]
    assert list(core.frames) == ["f1.frame", "f2.frame", "f3.frame"]


def test_a_streaming_edit_leaves_the_blocks_pushed_after_the_view():
    """A view made before the newest push lacks the frame. A view keeps every block it once had
    (decision 4.6), so the frame was pushed after it, and the edit leaves it where it is."""
    blocks = stream("FFF")
    live = Live(blocks[:2], frames=5)
    core = live.core
    first = core.frames["f1.frame"].user_item_id
    (pushed,) = core.push(streaming_view(blocks, frames=5))
    live.deliver([pushed])
    batch = core.queue_edit(streaming_view(blocks[:2], frames=1))
    assert shape(batch.operations) == [
        ("delete", first, None, True),
        ("create", "user", "root", True),
    ]
    live.deliver(batch.operations)
    core.finish_edit()
    assert (core.frames["f3.frame"].user_item_id, core.frames["f3.frame"].folded) == (
        pushed.item_id,
        False,
    )


def test_an_error_about_a_push_refuses_that_block_alone():
    """With a reply in progress and an edit waiting, an error naming a push's client event
    refuses that block: it leaves the mirror, and the reply and the edit go on. Later edits take
    the refused frame as the server does, as absent: the view still holds it, the edit leaves it
    out, and the frame after it folds where it is."""
    blocks = stream("FFFFF")
    live = Live(blocks[:2], frames=5)
    core = live.core
    reply = live.reply(end=False)
    (refused,) = core.push(streaming_view(blocks[:3], frames=5))
    live.server.refuse.add(refused.item_id)
    (refusal,) = live.send([refused])
    fold = core.queue_edit(streaming_view(blocks[:3], frames=2))
    answers = live.send(fold.operations)

    live.arrive([refusal])
    assert core.pushes[refused.event_id].error is refusal.error
    assert list(core.frames) == ["f1.frame", "f2.frame"]
    assert not core.edit_answered
    live.end(reply)
    assert [event.type for event in reply.events] == [
        "response.created",
        "response.output_item.added",
        "response.done",
    ]
    live.arrive(answers)
    assert core.finish_edit() is fold

    for n in (4, 5):
        live.deliver(core.push(streaming_view(blocks[:n], frames=5)))
    item, before = items_of(core), list(live.server.items)
    batch = core.queue_edit(streaming_view(blocks, frames=1))
    deletes = [op.item_id for op in batch.operations if isinstance(op, Delete)]
    assert deletes == [item["f2.frame"], item["f4.frame"]]
    live.deliver(batch.operations)
    core.finish_edit()
    assert in_place(live, before, batch)


def test_a_push_is_settled_once_acknowledged_or_refused():
    """A pushed block's delivery ends acknowledged, or refused once the server's error names its
    client event, as the server then does not have it; either way the push is settled."""
    core = Live(stream("F")).core
    (acknowledged,) = core.push(streaming_view(stream("FF")))
    (refused,) = core.push(streaming_view(stream("FFF")))
    sent(core, [acknowledged, refused])
    assert not core.settled(acknowledged) and not core.settled(refused)
    core.receive(done(acknowledged.item_id))
    core.receive(error(refused.event_id))
    assert (core.delivery[acknowledged.key], core.delivery[refused.key]) == (
        Delivery.ACKNOWLEDGED,
        Delivery.REFUSED,
    )
    assert core.settled(acknowledged) and core.settled(refused)


def test_a_pushed_blocks_fate_is_heard_with_its_block():
    """on_delivery hears a push's fate as the event log records it: confirmed by its
    acknowledgement, or refused once the server's error names its client event. A feedback
    pushed as its own item names its own block."""
    heard = []
    numbers = itertools.count(1)
    core = RealtimeSessionCore(
        lambda kind: f"rtb_{kind}_{next(numbers)}",
        lambda state, contents: heard.append((state, contents)),
    )
    blocks = stream("FRFB")
    _, frame = core.push(streaming_view(blocks[:3]))
    (feedback,) = core.push(streaming_view(blocks))
    sent(core, [frame, feedback])
    core.receive(done(frame.item_id))
    core.receive(error(feedback.event_id))
    assert blocks[3].kind == "execution_feedback"
    assert heard == [
        ("sent", [(frame.key, "f2.frame")]),
        ("sent", [(feedback.key, blocks[3].id)]),
        ("confirmed", [(frame.key, "f2.frame")]),
        ("refused", [(feedback.key, blocks[3].id)]),
    ]


def test_a_block_refused_again_after_a_rebuild_is_refused_alone():
    """A rebuilt session replays every block the view holds, a frame the old connection refused
    too. Its frames and feedback are planned as pushes, under client events, so the server
    refusing one again refuses it alone: the reply asked for meanwhile gets nothing of it, and
    later edits leave it out."""
    blocks = stream("FFRBFF")
    live = Live()
    core = live.core
    replayed = core.replay_streaming(streaming_view(blocks, frames=5))
    f1, f2, r1, fb1, f3, f4 = replayed
    assert [op.event_id is not None for op in replayed] == [True, True, False, True, True, True]
    assert {op.event_id[:9] for op in (f1, f2, fb1, f3, f4)} == {"rtb_push_"}
    live.server.refuse.add(f2.item_id)
    reply = core.start_reply()
    live.deliver(replayed)
    assert list(reply.events) == []
    assert "f2.frame" not in core.frames
    before = list(live.server.items)
    assert before == [f1.item_id, r1.item_id, fb1.item_id, f3.item_id, f4.item_id]

    batch = core.queue_edit(streaming_view(blocks, frames=1))
    assert [op.item_id for op in batch.operations if isinstance(op, Delete)] == [
        f1.item_id,
        f3.item_id,
    ]
    live.deliver(batch.operations)
    core.finish_edit()
    assert in_place(live, before, batch)


def test_a_refused_feedback_is_left_out_of_later_edits():
    """Feedback pushed as its own item and refused leaves the session as a refused frame does:
    the frame after it folds where it is, after the reply before it."""
    live = Live(stream("FFR"), frames=5)
    core = live.core
    (refused,) = core.push(streaming_view(stream("FFRB"), frames=5))
    live.server.refuse.add(refused.item_id)
    live.deliver([refused])
    assert "d1.feedback" not in core.frames
    for script in ("FFRBF", "FFRBFF"):
        live.deliver(core.push(streaming_view(stream(script), frames=5)))
    before = list(live.server.items)
    batch = core.queue_edit(streaming_view(stream("FFRBFF"), frames=1))
    anchors = [op.previous_item_id for op in batch.operations if isinstance(op, Create)]
    assert anchors == ["root", batch.operations[1].item_id, before[2]]
    live.deliver(batch.operations)
    core.finish_edit()
    assert in_place(live, before, batch)


def test_a_queued_edit_waits_for_the_one_before_it():
    """Each push brings a view that folds one more frame. Views that arrive while an edit waits
    are kept, the newest replacing the older; once the edit is finished, the newest is planned
    against the session it left, once."""
    blocks = stream("FFFFFF")
    live = Live(blocks[:3], frames=3)
    core = live.core
    users = items_of(core)

    def pushed(n: int):
        view = streaming_view(blocks[:n], frames=3)
        live.deliver(core.push(view))
        return view

    first = core.queue_edit(pushed(4))
    assert shape(first.operations) == [
        ("delete", users["f1.frame"], None, True),
        ("create", "user", "root", True),
    ]
    answers = live.send(first.operations)
    assert core.queue_edit(pushed(5)) is None
    assert core.queue_edit(pushed(6)) is None
    assert core.next_edit() is None

    live.arrive(answers)
    assert core.edit_answered and core.next_edit() is None
    assert core.finish_edit() is first
    second = core.next_edit()
    assert shape(second.operations) == [
        ("delete", users["f2.frame"], None, True),
        ("create", "user", first.operations[1].item_id, True),
        ("delete", users["f3.frame"], None, True),
        ("create", "user", second.operations[1].item_id, True),
    ]
    assert (second.visuals_folded, second.prompts_edited) == (2, 2)
    assert core.next_edit() is None
    live.deliver(second.operations)
    assert core.finish_edit() is second
    assert core.next_edit() is None
    assert [f.folded for f in core.frames.values()] == [True, True, True, False, False, False]


def test_a_failed_streaming_edit_drops_the_view_kept_behind_it():
    """A failed edit leaves the session whatever the server made of it, to be rebuilt; the view
    kept while it waited is not planned against that."""
    blocks = stream("FFFF")
    live = Live(blocks[:3], frames=5)
    core = live.core
    first = core.queue_edit(streaming_view(blocks[:3], frames=2))
    live.send(first.operations)
    live.deliver(core.push(streaming_view(blocks, frames=5)))
    assert core.queue_edit(streaming_view(blocks, frames=2)) is None
    core.receive(error(first.operations[0].event_id))
    with pytest.raises(RuntimeError, match=r"^Realtime history edit failed: "):
        core.finish_edit()
    assert core.next_edit() is None


def test_a_fold_while_a_reply_streams_loses_no_event():
    """Native streaming in small: while a reply streams, a frame is pushed and the view it brings
    folds the oldest frame. The reply's events and the acknowledgements of the edit and of the
    push arrive interleaved, and each reaches its state."""
    blocks = stream("FFF")
    live = Live(blocks[:2], frames=5)
    core = live.core
    reply = core.start_reply()
    view = streaming_view(blocks, frames=2)
    (pushed,) = core.push(view)
    (push_answer,) = live.send([pushed])
    fold = core.queue_edit(view)
    delete_answer, create_answer = live.send(fold.operations)

    created, added, text, finished = response_events()
    for event in (created, delete_answer, push_answer, added, create_answer, text, finished):
        assert core.arrived(event, 0.0)
    assert list(reply.events) == [created, added, text, finished]
    assert core.edit_answered and core.finish_edit() is fold
    assert core.delivery[pushed.key] == Delivery.ACKNOWLEDGED
    assert all(core.delivery[op.key] == Delivery.ACKNOWLEDGED for op in fold.operations)
    assert (core.frames["f1.frame"].user_item_id, core.frames["f1.frame"].folded) == (
        fold.operations[1].item_id,
        True,
    )
    assert reply.output_item_ids == ["srv_1"]


def test_the_streaming_reader_records_when_events_came_off_the_wire():
    """4.17: native streaming's reader hands each event over with the time it read it. A pushed
    block was acknowledged at its first acknowledgement; the actions parsed from a reply's
    completed output text arrived when that text did, kept by its server event id. receive(),
    which the turn-based reader calls, records neither, nor does an event no state takes, such as
    the acknowledgement of an item the server made itself."""
    blocks = stream("FF")
    core = new_core()
    for operation in sent(core, core.replay_streaming(streaming_view(blocks[:1], frames=5))):
        core.receive(done(operation.item_id))
    (pushed,) = core.push(streaming_view(blocks, frames=5))
    sent(core, [pushed])
    reply = core.start_reply()
    created, added, _, finished = response_events()
    text = server(
        type="response.output_text.done",
        event_id="event_7",
        response_id="resp_1",
        item_id="srv_1",
        output_index=0,
        content_index=0,
        text='{"action": "wait"}',
    )
    for event, now in (
        (created, 1.0),
        (server(type="conversation.item.created", item={"id": pushed.item_id}), 1.5),
        (added, 2.0),
        (done(pushed.item_id), 2.5),
        (text, 3.0),
        (finished, 3.5),
    ):
        assert core.arrived(event, now)
    assert not core.arrived(server(type="rate_limits.updated", rate_limits=[]), 4.0)
    assert not core.arrived(done("srv_1"), 4.5)
    assert list(reply.events) == [created, added, text, finished]
    assert core.acknowledged_at == {pushed.key: 1.5}
    assert reply.received_at == {"event_7": 3.0}
