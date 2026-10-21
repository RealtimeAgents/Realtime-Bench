"""The OpenAI Realtime session core, driven without a WebSocket.

History views come from the request builder the runtime uses (harnesses/default/request.py);
server events are parsed the way the SDK's RealtimeConnection.recv parses them
(openai/resources/realtime/realtime.py).
"""

from __future__ import annotations

import itertools
from typing import Any, cast

import pytest
from openai._models import construct_type_unchecked
from openai.types.realtime import RealtimeServerEvent

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.model import CanonicalTurn, ProviderObservation
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.request import build_request, ledger_block, system_block
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


def done(item_id: str):
    return server(type="conversation.item.done", item={"id": item_id})


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
    That can no longer happen. abort() sets the abandoned epoch's retry cancellation, which stops
    the rebuild before it connects (RequestRetry, entity/model.py), and _run_turn checks the epoch
    before it sends the observation (harnesses/default/context.py). A decision abandoned after
    that check can still start its response, but only on the connection abort() closes: the next
    decision waits for that cleanup and opens a new connection (_ensure_ready), whose core never
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


def test_a_push_names_its_client_event():
    """A pushed frame is appended like an observed one, but its create names a client event."""
    core = opened([turn(0, "a0")])
    user = core.push(view([turn(0, "a0"), turn(1, feedback="EXECUTION RESULT: y")]))
    assert (user.role, user.text, user.image.data) == (
        "user",
        "EXECUTION RESULT: y\nframe 1",
        b"png-1",
    )
    assert user.previous_item_id is None and user.event_id.startswith("rtb_push_")
    assert core.frames["d1.frame"].user_item_id == user.item_id
    assert core.pushes == {user.event_id: Push("d1.frame", user.item_id)}
    sent(core, [user])
    assert core.receive(done(user.item_id))
    assert core.delivery[user.key] == Delivery.ACKNOWLEDGED
    assert core.pushes[user.event_id].error is None


def test_an_error_about_a_push_refuses_that_frame_alone():
    """With a response in progress and an edit waiting, an error naming a push's client event
    refuses that frame: it leaves the mirror, and the response and the edit go on. The edit was
    planned after the push from a view that holds the frame, and leaves it as it was."""
    turns = [turn(0, "a0"), turn(1, "a1")]
    core = opened(turns)
    generation = core.start_generation("d1.frame")
    turns.append(turn(2))
    push = core.push(view(turns))
    sent(core, [push])
    turns[0].fold_visual()
    batch = core.begin_edit(view(turns))
    delete0, create0 = sent(core, batch.operations)

    refusal = error(push.event_id)
    assert core.receive(refusal)
    assert core.pushes[push.event_id].error is refusal.error
    assert core.delivery[push.key] == Delivery.SENT
    assert list(core.frames) == ["d0.frame", "d1.frame"]
    assert list(generation.events) == []
    assert not core.edit_answered

    core.receive(deleted(delete0.item_id))
    core.receive(done(create0.item_id))
    assert core.finish_edit() is batch
    assert list(core.frames) == ["d0.frame", "d1.frame"]
    assert (core.frames["d0.frame"].user_item_id, core.frames["d0.frame"].folded) == (
        create0.item_id,
        True,
    )
    # An error that names no client event still reaches the response.
    unnamed = error(None)
    assert core.receive(unnamed)
    assert list(generation.events) == [unnamed]


@pytest.mark.parametrize("pushing", [True, False])
def test_a_pushing_edit_leaves_out_frames_pushed_after_the_view(pushing):
    """A view made before the newest push archives decision 0 and folds decision 1. Pushing, the
    edit leaves the newer frame where it is; the turn-based edit deletes it, as it was never
    meant to meet one."""
    turns = [turn(k, f"a{k}") for k in range(3)]
    core = opened(turns)
    pushed = core.push(view([*turns, turn(3)]))
    sent(core, [pushed])
    archived = [core.frames["d0.frame"].user_item_id, *core.frames["d0.frame"].assistant_item_ids]
    folded = core.frames["d1.frame"].user_item_id
    turns[1].fold_visual()
    history = view(turns[1:], ledger="L")
    assert core.in_place_problem(history, pushing=pushing) is None

    batch = core.begin_edit(history, pushing=pushing)
    newer = [] if pushing else [pushed.item_id]
    assert batch.deleted == {*archived, *newer, folded}
    assert batch.removed == (("d0.frame",) if pushing else ("d0.frame", "d3.frame"))
    assert list(batch.frames) == ["d1.frame", "d2.frame"]
    for op in sent(core, batch.operations):
        core.receive(deleted(op.item_id) if isinstance(op, Delete) else done(op.item_id))
    core.finish_edit()
    expected = ["d1.frame", "d2.frame", "d3.frame"] if pushing else ["d1.frame", "d2.frame"]
    assert list(core.frames) == expected
    if pushing:
        assert core.frames["d3.frame"].user_item_id == pushed.item_id


def test_a_pushing_edit_leaves_out_frames_not_pushed_yet():
    """A view that already holds a frame the core has not pushed: the turn-based check refuses
    it, the pushing one leaves the frame to its push. A frame missing before the newest one the
    mirror holds is still refused."""
    turns = [turn(0, "a0"), turn(1, "a1"), turn(2)]
    core = opened(turns[:2])
    first = core.frames["d0.frame"].user_item_id
    turns[0].fold_visual()
    history = view(turns)
    assert core.in_place_problem(history) == "retained turn is missing its provider user item ID"
    assert core.in_place_problem(history, pushing=True) is None
    batch = core.begin_edit(history, pushing=True)
    assert shape(batch.operations) == [
        ("delete", first, None, True),
        ("create", "user", "root", True),
    ]
    assert list(batch.frames) == ["d0.frame", "d1.frame"]

    gap = opened([turn(0, "a0"), turn(2, "a2")])
    assert (
        gap.in_place_problem(view([turn(0, "a0"), turn(1, "a1"), turn(2, "a2")]), pushing=True)
        == "retained turn is missing its provider user item ID"
    )


def test_a_pushing_view_with_no_frame_in_common_keeps_every_sent_frame():
    """Such a view cannot tell the frames it dropped from those pushed after it, so the edit
    keeps them all; the next view that shares a frame with the session drops the old ones."""
    core = opened([turn(0, "a0")])
    items = [core.frames["d0.frame"].user_item_id, *core.frames["d0.frame"].assistant_item_ids]
    history = view([turn(1)], ledger="L")
    assert core.in_place_problem(history, pushing=True) is None
    batch = core.begin_edit(history, pushing=True)
    assert shape(batch.operations) == [("create", ROLES[POLICY.ledger_role], "root", True)]
    assert (batch.removed, batch.frames) == ((), {})
    for op in sent(core, batch.operations):
        core.receive(done(op.item_id))
    core.finish_edit()

    sent(core, [core.push(history)])
    batch = core.begin_edit(history, pushing=True)
    assert shape(batch.operations) == [("delete", item_id, None, True) for item_id in items]
    assert batch.removed == ("d0.frame",)


def acknowledge(core: RealtimeSessionCore, operations) -> None:
    for op in operations:
        core.receive(deleted(op.item_id) if isinstance(op, Delete) else done(op.item_id))


def test_a_queued_edit_waits_for_the_one_before_it():
    """Views that arrive while an edit waits are kept, the newest replacing the older; once the
    edit is finished, the newest is planned against the session it left."""
    turns = [turn(k, f"a{k}") for k in range(4)]
    core = opened(turns)
    assistant = {k: f.assistant_item_ids[-1] for k, f in core.frames.items()}
    users = {k: f.user_item_id for k, f in core.frames.items()}
    turns[0].fold_visual()
    first = core.queue_edit(view(turns))
    assert shape(first.operations) == [
        ("delete", users["d0.frame"], None, True),
        ("create", "user", "root", True),
    ]
    sent(core, first.operations)
    turns[1].fold_visual()
    assert core.queue_edit(view(turns)) is None
    turns[2].fold_visual()
    assert core.queue_edit(view(turns)) is None
    assert core.next_edit() is None

    acknowledge(core, first.operations)
    assert core.edit_answered and core.next_edit() is None
    assert core.finish_edit() is first
    second = core.next_edit()
    assert shape(second.operations) == [
        ("delete", users["d1.frame"], None, True),
        ("create", "user", assistant["d0.frame"], True),
        ("delete", users["d2.frame"], None, True),
        ("create", "user", assistant["d1.frame"], True),
    ]
    assert core.next_edit() is None
    sent(core, second.operations)
    acknowledge(core, second.operations)
    assert core.finish_edit() is second
    assert [f.folded for f in core.frames.values()] == [True, True, True, False]


def test_a_queued_view_the_session_cannot_reach_is_refused_before_any_plan():
    turns = [turn(0, "a0"), turn(1, "a1")]
    core = opened(turns)
    turns[0].fold_visual()
    batch = core.queue_edit(view(turns))
    sent(core, batch.operations)
    assert core.queue_edit(view([turn(0, "a0"), turn(1, "a1")])) is None
    acknowledge(core, batch.operations)
    core.finish_edit()
    with pytest.raises(
        RuntimeError,
        match=r"^Realtime history edit refused: provider visual state cannot be unfolded in place$",
    ):
        core.next_edit()
    assert core.next_edit() is None
    # Nothing was planned, so the next view is planned at once.
    assert core.queue_edit(view(turns)) is not None


def test_frames_dropped_while_a_response_streams_lose_no_event():
    """Native streaming in small: while a response streams, a frame is pushed and queued edits
    archive the two oldest frames, assistant items included, then fold the next one. The view
    was made before the push. The response's events and the acknowledgements of the edit and of
    the push arrive interleaved, and each reaches its state; the pushed frame stays, and the
    response's output joins the frame it answered."""
    turns = [turn(0, "a0"), turn(1, "a1"), turn(2, "a2"), turn(3)]
    core = opened(turns[:3])
    pushed = core.push(view(turns))
    sent(core, [pushed])
    generation = core.start_generation("d3.frame")
    archive = core.queue_edit(view(turns[2:3], ledger="L"))
    sent(core, archive.operations)
    turns[2].fold_visual()
    assert core.queue_edit(view(turns[2:3], ledger="L")) is None

    created, added, text, finished = response_events()
    *deletes, ledger = archive.operations
    assert len(deletes) == 4 and isinstance(ledger, Create)
    acks = [deleted(op.item_id) for op in deletes]
    for event in (
        created,
        acks[0],
        done(pushed.item_id),
        acks[1],
        added,
        acks[2],
        text,
        done(ledger.item_id),
        acks[3],
        finished,
    ):
        assert core.receive(event)
    assert list(generation.events) == [created, added, text, finished]
    assert core.edit_answered
    assert core.finish_edit() is archive
    assert core.delivery[pushed.key] == Delivery.ACKNOWLEDGED
    assert all(core.delivery[op.key] == Delivery.ACKNOWLEDGED for op in archive.operations)
    assert list(core.frames) == ["d2.frame", "d3.frame"]
    assert core.frames["d3.frame"].assistant_item_ids == ["srv_1"]

    fold = core.next_edit()
    assert shape(fold.operations) == [
        ("delete", core.frames["d2.frame"].user_item_id, None, True),
        ("create", "user", ledger.item_id, True),
    ]


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_a_streaming_reply_keeps_its_items_under_its_output_block(status):
    """A native-streaming reply answers no frame: its events join it as a turn's do, and at its
    response.done its output items are kept under the output block it becomes."""
    core = opened([turn(0, "a0")])
    assistant = core.frames["d0.frame"].assistant_item_ids
    reply = core.start_reply("g1.output")
    events = response_events(status)
    assert all(core.receive(event) for event in events)
    assert list(reply.events) == events
    assert core.outputs == ({} if status == "failed" else {"g1.output": ["srv_1"]})
    assert core.frames["d0.frame"].assistant_item_ids == assistant
