"""The native-streaming agent (harnesses/streaming/agent.py) on a fake live encoder
(tests/streaming/fake_live.py) and the fake clock: what it hands the encoder for each push, and
the view each connection operation is given."""

from __future__ import annotations

import json
import threading
from dataclasses import replace

import pytest

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.entity.model import TRANSIENT, TransportFailureError
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.kernel.events import Outcome
from realtime_framework.evaluation.kernel.streaming import CallsCancelled, PushHeld, PushResumed
from realtime_framework.harnesses.default.frames import FRAME_NOT_IN_TABLE, INVALID_GROUNDING_FRAME
from realtime_framework.harnesses.streaming.feedback import action_text, execution_feedback
from realtime_framework.providers.gemini.live import GoAwayDeadlineExceeded
from tests.fake_clock import FakeClock

from .fake_live import (
    BUDGET,
    REALTIME,
    Call,
    Cancel,
    Error,
    Fail,
    GoAway,
    Hang,
    RecordingToolRuntime,
    RecordingTrace,
    Reply,
    Text,
    TextEnd,
    TextLiveDriver,
    make_agent,
    observation,
)

ORIGIN = 999.0  # the episode's origin on the wall clock, which starts at 1000 s


@pytest.fixture
def clock():
    clock = FakeClock(wall_origin=1000.0)
    with clock.participant():
        yield clock


def test_the_session_opens_empty_on_an_encoder_built_for_streaming(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    assert agent.driver.streaming
    assert agent.driver.calls == [("connect", ("system",))]
    agent.close()


def test_a_push_numbers_the_frame_hands_the_view_ending_in_it_and_records_it_sent(
    monkeypatch, clock
):
    agent = make_agent(monkeypatch, clock)
    agent.push(observation(clock, 1))
    clock.sleep(0.5)
    agent.push(observation(clock, 2))
    assert agent.driver.calls[1:] == [
        ("push", "obs-1", ("system", "f1.frame")),
        ("push", "obs-2", ("system", "f1.frame", "f2.frame")),
    ]
    first, second = (agent.frames.locate(n)[0] for n in (1, 2))
    assert (first.send_status, first.sent_unix_s) == ("sent", 1000.0)
    assert (second.send_status, second.sent_unix_s) == ("sent", 1000.5)
    agent.close()


def test_a_frame_whose_push_failed_leaves_the_view_and_keeps_its_number(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    agent.push(observation(clock, 1))
    agent.driver.failing.add("obs-2")
    with pytest.raises(RuntimeError, match="push failed"):
        agent.push(observation(clock, 2))
    agent.push(observation(clock, 3))
    assert agent.driver.calls[-1] == ("push", "obs-3", ("system", "f1.frame", "f3.frame"))
    assert agent.frames.locate(2) == (None, FRAME_NOT_IN_TABLE)
    agent.close()


def test_a_frame_that_went_in_is_recorded_under_its_number(monkeypatch, clock):
    """FrameSent ties the kernel's observation id to the frame-table number the session's
    delivery records name the frame's block by (evaluation/reporting.py summarize_streaming)."""
    trace = RecordingTrace()
    agent = make_agent(monkeypatch, clock, trace=trace)
    agent.push(observation(clock, 1))
    agent.driver.failing.add("obs-2")
    with pytest.raises(RuntimeError, match="push failed"):
        agent.push(observation(clock, 2))
    agent.push(observation(clock, 3))
    assert [(r["frame"], r["observation_id"]) for r in trace.of(TraceEvent.FRAME_SENT)] == [
        (1, "obs-1"),
        (3, "obs-3"),
    ]
    agent.close()


def test_a_push_records_its_frame_sent_and_confirmed_when_its_receipt_says(monkeypatch, clock):
    """Decision N43: the frame went out when the receipt says it was written, not when the push
    came back, which waits for the server's confirmation; both times go into the frame table on
    the wall clock, and FrameSent records the frame-table row's capture, send and confirmation."""
    trace = RecordingTrace()
    agent = make_agent(monkeypatch, clock, TextLiveDriver, REALTIME, trace=trace)
    agent.driver.confirm_s = 0.4
    agent.push(observation(clock, 1))
    record, _ = agent.frames.locate(1)
    assert (record.sent_unix_s, record.ack_status) == (1000.0, "acked")
    assert record.acked_unix_s == pytest.approx(1000.4)
    [sent] = trace.of(TraceEvent.FRAME_SENT)
    assert (sent["captured_unix_s"], sent["sent_unix_s"]) == (record.captured_at, 1000.0)
    assert sent["acked_unix_s"] == record.acked_unix_s
    agent.close()


def test_a_push_on_a_protocol_that_confirms_no_frame_leaves_its_confirmation_unknown(
    monkeypatch, clock
):
    """Gemini Live confirms no frame: its receipt has no confirmation time."""
    trace = RecordingTrace()
    agent = make_agent(monkeypatch, clock, trace=trace)
    clock.sleep(0.25)
    agent.push(observation(clock, 1))
    record, _ = agent.frames.locate(1)
    assert (record.sent_unix_s, record.ack_status, record.acked_unix_s) == (
        1000.25,
        "unknown",
        None,
    )
    [sent] = trace.of(TraceEvent.FRAME_SENT)
    assert (sent["captured_unix_s"], sent["sent_unix_s"], sent["acked_unix_s"]) == (
        record.captured_at,
        1000.25,
        None,
    )
    agent.close()


def test_frames_beyond_the_encoders_retained_count_fold(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    for n in (1, 2, 3):
        agent.push(observation(clock, n))
    view = agent.driver.views[-1]
    folded = {item.block.id: item.block.folded for item in view.items if item.block.kind == "frame"}
    assert folded == {"f1.frame": True, "f2.frame": False, "f3.frame": False}
    agent.close()


def _click(call_id: str, at: float, frame) -> Call:
    return Call(at, call_id, "click", {"x": 10, "y": 20, "button": "left", "frame": frame})


def _two_frames(agent, clock):
    """Frame 1 pushed at 1000 s and frame 2 at 1001 s; the second, as the shell keeps it."""
    agent.push(observation(clock, 1))
    clock.sleep(1.0)
    second = observation(clock, 2)
    agent.push(second)
    return second


def test_a_reply_starts_on_the_view_and_posts_each_call_as_it_comes_off_the_wire(
    monkeypatch, clock
):
    """Each call is posted as it is read, while the reply goes on (decision 19), with the time
    it came off the wire (4.17): the second is read 0.25 s after it came."""
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(lambda event: posted.append((event, clock.monotonic())), ORIGIN)
    frame = _two_frames(agent, clock)
    late = replace(_click("c2", 1.0, 2), late=0.25)
    agent.driver.replies.append(Reply((_click("c1", 0.5, 1), late), ends_at=1.5))
    pending = agent.generate(7, 3, frame, BUDGET)
    assert agent.driver.calls[-1] == ("reply", 7, ("system", "f1.frame", "f2.frame"))
    assert [
        (a.generation, a.call_id, a.arrived_at, a.frame, a.frame_age_s, a.frame_age_frames, at)
        for a, at in posted
    ] == [(7, "c1", 2.5, 1, 1.5, 1, 1.5), (7, "c2", 3.0, 2, 1.0, 0, 2.25)]
    assert all(a.action.action_type == ActionType.CLICK and a.rejection is None for a, _ in posted)
    r = pending.result
    assert (pending.generation, pending.decision, r["observation_id"]) == (7, 3, "obs-2")
    assert r["action_protocol"]["protocol_failure"] is None
    assert r["raw"].splitlines() == [
        '{"action": "click", "x": 10, "y": 20, "button": "left", "frame": 1}',
        '{"action": "click", "x": 10, "y": 20, "button": "left", "frame": 2}',
    ]
    agent.close()


def test_an_action_naming_no_sent_frame_arrives_refused(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((_click("c1", 0.5, 9),), ends_at=1.0))
    agent.generate(0, 0, frame, BUDGET)
    [arrival] = posted
    assert (arrival.action, arrival.call_id, arrival.arrived_at) == (None, "c1", 2.5)
    assert (arrival.rejection, arrival.detail) == (INVALID_GROUNDING_FRAME, FRAME_NOT_IN_TABLE)
    agent.close()


def test_a_call_that_is_no_action_is_the_replys_protocol_failure_and_hands_nothing_over(
    monkeypatch, clock
):
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(
        Reply((Call(0.5, "c1", "fly", {"to": "moon"}), _click("c2", 0.75, 2)), ends_at=1.0)
    )
    r = agent.generate(0, 0, frame, BUDGET).result
    assert [a.call_id for a in posted] == ["c2"]
    assert r["action_protocol"]["protocol_failure"] == "invalid_action_output"
    agent.close()


def test_an_action_in_text_arrives_when_its_text_is_complete_named_by_the_agent(monkeypatch, clock):
    """Its text completed 0.5 s into a reply that ends 2 s in: the action arrived then
    (decisions 19 and 4.17), and is posted as it does, while the reply goes on."""
    agent = make_agent(monkeypatch, clock, TextLiveDriver, REALTIME)
    posted = []
    agent.bind_inbox(lambda event: posted.append((event, clock.monotonic())), ORIGIN)
    frame = _two_frames(agent, clock)
    action = '{"action": "click", "x": 10, "y": 20, "button": "left", "frame": 2}'
    agent.driver.replies.append(
        Reply((Text(0.25, action[:20]), Text(0.5, action[20:]), TextEnd(0.75)), 2.0)
    )
    r = agent.generate(0, 0, frame, BUDGET).result
    [(arrival, posted_at)] = posted
    assert (arrival.call_id, arrival.arrived_at, arrival.frame) == ("text", 2.75, 2)
    assert (arrival.frame_age_s, posted_at) == (0.75, 1.75)
    assert r["raw"] == action
    agent.close()


def test_an_action_in_text_that_never_completed_is_parsed_when_the_reply_ended(monkeypatch, clock):
    """With no completed text, as when the response was cut short, the reply's end stands in."""
    agent = make_agent(monkeypatch, clock, TextLiveDriver, REALTIME)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    action = '{"action": "click", "x": 10, "y": 20, "button": "left", "frame": 2}'
    agent.driver.replies.append(Reply((Text(0.5, action),), 2.0, boundaries=("interrupted",)))
    agent.generate(0, 0, frame, BUDGET)
    [arrival] = posted
    assert (arrival.call_id, arrival.arrived_at) == ("text", 4.0)
    agent.close()


def test_an_accepted_reply_adds_its_output_to_the_view_and_its_usage_to_the_episode(
    monkeypatch, clock
):
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    usage = {"output_tokens": 5, "input_tokens": 40, "total_tokens": 45}
    agent.driver.replies.append(Reply((_click("c1", 0.5, 1),), ends_at=1.0, usage=usage))
    agent.generate(7, 3, frame, BUDGET)
    agent.accept(7)
    agent.push(observation(clock, 3))
    view = agent.driver.views[-1]
    ids = [item.block.id for item in view.items]
    assert ids == ["system", "f1.frame", "f2.frame", "d3.output", "f3.frame"]
    output = view.items[3].block
    assert (output.kind, output.based_on) == ("model_output", "obs-1")
    assert output.text == '{"action": "click", "x": 10, "y": 20, "button": "left", "frame": 1}'
    assert (agent.token_usage["output_tokens"], agent.token_usage["responses"]) == (5, 1)
    agent.close()


def test_a_discarded_reply_adds_nothing(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    usage = {"output_tokens": 5}
    agent.driver.replies.append(Reply((_click("c1", 0.5, 1),), ends_at=1.0, usage=usage))
    agent.generate(7, 3, frame, BUDGET)
    agent.discard(7)
    agent.push(observation(clock, 3))
    assert agent.driver.calls[-1] == (
        "push",
        "obs-3",
        ("system", "f1.frame", "f2.frame", "f3.frame"),
    )
    assert agent.token_usage["responses"] == 0
    agent.close()


def test_a_reply_that_said_nothing_adds_no_output(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply(ends_at=1.0))
    agent.generate(0, 0, frame, BUDGET)
    agent.accept(0)
    agent.push(observation(clock, 3))
    assert agent.driver.calls[-1] == (
        "push",
        "obs-3",
        ("system", "f1.frame", "f2.frame", "f3.frame"),
    )
    agent.close()


def _outcome(decision, status, call_id=None, *, action=None, detail="", protocol_failure=None):
    """The kernel's Commit; the clock fields are not part of the feedback."""
    return Outcome(
        decision, status, detail, False, 0.0, 0.0, 0, False, None, None, None, 0,
        action=action, protocol_failure=protocol_failure, call_id=call_id,
    )  # fmt: skip


def test_each_outcome_is_feedback_on_the_action_its_call_handed_over(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((_click("c1", 0.5, 1), _click("c2", 0.75, 9)), 1.0))
    agent.generate(7, 3, frame, BUDGET)
    agent.accept(7)
    applied = posted[0]
    outcomes = [
        _outcome(3, "applied", "c1", action=applied.action),
        _outcome(3, INVALID_GROUNDING_FRAME, "c2", detail=FRAME_NOT_IN_TABLE),
    ]
    for outcome in outcomes:
        agent.commit(outcome)
    agent.push(observation(clock, 3))
    items = agent.driver.views[-1].items
    assert [item.block.id for item in items][3:] == [
        "d3.output",
        "d3.c1.feedback",
        "d3.c2.feedback",
        "f3.frame",
    ]
    requested = action_text(applied.action)
    assert [item.block.text for item in items[4:6]] == [
        execution_feedback(outcomes[0], requested),
        execution_feedback(outcomes[1], "none"),
    ]
    assert [(a["call_id"], a["requested"], a["result"]) for a in agent.action_history] == [
        ("c1", requested, "applied"),
        ("c2", "none", INVALID_GROUNDING_FRAME),
    ]
    agent.close()


def test_a_reply_that_came_to_no_action_has_feedback_of_its_own(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((Call(0.5, "c1", "fly", {"to": "moon"}),), ends_at=1.0))
    agent.generate(0, 0, frame, BUDGET)
    agent.accept(0)
    outcome = _outcome(0, "none", protocol_failure="invalid_action_output")
    agent.commit(outcome)
    agent.push(observation(clock, 3))
    items = agent.driver.views[-1].items
    assert [item.block.id for item in items][3:] == ["d0.output", "d0.feedback", "f3.frame"]
    assert items[4].block.text == execution_feedback(outcome, "none")
    agent.close()


def _on_worker(clock, run):
    """``run()`` on a thread of the clock's, as the shell runs a reply; its value or exception
    comes back in the dict once the event is set."""
    done, out = clock.event(), {}

    def worker():
        try:
            out["value"] = run()
        except BaseException as exc:  # noqa: BLE001 - the test reads it
            out["error"] = exc
        finally:
            done.set()

    clock.start_thread(worker, name="reply")
    return done, out


def test_an_abort_interrupts_the_reply_and_closes_the_session_before_it_returns(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((_click("c1", 0.5, 1),), ends_at=100.0))
    done, out = _on_worker(clock, lambda: agent.generate(7, 3, frame, BUDGET))
    clock.sleep(1.0)
    agent.abort(7)
    assert agent.driver.calls[-2:] == [("interrupt", "streaming-7"), ("close",)]
    assert clock.wait(done, 1.0)
    assert out["value"].cancelled
    assert [a.call_id for a in posted] == ["c1"]
    agent.discard(7)
    agent.push(observation(clock, 3))
    assert agent.driver.calls[-2:] == [
        ("connect", ("system", "f1.frame", "f2.frame")),
        ("push", "obs-3", ("system", "f1.frame", "f2.frame", "f3.frame")),
    ]
    agent.close()


def test_a_reply_aborted_before_its_worker_started_never_starts(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.abort(7)
    assert agent.driver.calls[-1] == ("close",)
    pending = agent.generate(7, 3, frame, BUDGET)
    assert pending.cancelled
    assert not [call for call in agent.driver.calls if call[0] == "reply"]
    agent.close()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("stream broke"),
        GoAwayDeadlineExceeded("gemini_live_go_away_deadline_before_turn_complete"),
    ],
)
def test_a_reply_that_fails_partway_raises_and_the_session_is_rebuilt_before_it_is_used(
    monkeypatch, clock, error
):
    """A reply that failed after it handed an action over, or that outlived its connection's
    goAway deadline, which the encoder takes as incomplete: the run cannot be scored. The
    session may still hold the reply, so the next push recovers it from the view first."""
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((_click("c1", 0.5, 1), Fail(0.75, error)), ends_at=1.0))
    with pytest.raises(type(error)):
        agent.generate(7, 3, frame, BUDGET)
    assert [a.call_id for a in posted] == ["c1"]
    agent.push(observation(clock, 3))
    assert agent.driver.calls[-2:] == [
        ("sync", "runtime_recovery", ("system", "f1.frame", "f2.frame")),
        ("push", "obs-3", ("system", "f1.frame", "f2.frame", "f3.frame")),
    ]
    agent.close()


def test_a_renewal_the_provider_asked_for_runs_before_the_next_push(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    _two_frames(agent, clock)
    agent.driver.renewal_required = True
    agent.push(observation(clock, 3))
    assert agent.driver.calls[-2:] == [
        ("sync", "go_away", ("system", "f1.frame", "f2.frame")),
        ("push", "obs-3", ("system", "f1.frame", "f2.frame", "f3.frame")),
    ]
    agent.close()


def test_after_a_go_away_during_a_reply_a_push_waits_for_it_and_then_renews(monkeypatch, clock):
    """Renewing the connection would end the reply, and the session takes no input before it is
    renewed: the push waits until the reply is accepted, renews from the view that holds it, and
    goes in then."""
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((_click("c1", 0.25, 2), GoAway(0.5)), ends_at=2.0))
    replied, out = _on_worker(clock, lambda: agent.generate(7, 3, frame, BUDGET))
    clock.sleep(1.0)
    pushed, pushing = _on_worker(clock, lambda: agent.push(observation(clock, 3)))
    assert clock.wait(replied, 2.0)
    assert not out["value"].cancelled
    clock.sleep(0.5)
    assert not pushed.is_set()
    agent.accept(7)
    assert clock.wait(pushed, 1.0)
    assert "error" not in pushing
    assert agent.driver.calls[-2:] == [
        ("sync", "go_away", ("system", "f1.frame", "f2.frame", "d3.output")),
        ("push", "obs-3", ("system", "f1.frame", "f2.frame", "d3.output", "f3.frame")),
    ]
    agent.close()


def test_a_push_while_a_reply_is_on_does_not_rebuild_the_session_under_it(monkeypatch, clock):
    """A session left dirty during a reply, as a failed fold leaves OpenAI Realtime's, is rebuilt
    once the reply is accepted: rebuilding it under the reply would close the reply's
    connection."""
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((_click("c1", 0.25, 2),), ends_at=2.0))
    replied, out = _on_worker(clock, lambda: agent.generate(7, 3, frame, BUDGET))
    clock.sleep(0.5)
    agent.driver.mark_dirty()
    agent.push(observation(clock, 3))
    assert agent.driver.calls[-1][:2] == ("push", "obs-3")
    assert clock.wait(replied, 2.0)
    assert "value" in out
    agent.accept(7)
    agent.push(observation(clock, 4))
    assert agent.driver.calls[-2][:2] == ("sync", "runtime_recovery")
    agent.close()


def test_a_reply_starts_on_a_renewed_session(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.renewal_required = True
    agent.driver.replies.append(Reply(ends_at=1.0))
    agent.generate(7, 3, frame, BUDGET)
    assert agent.driver.calls[-2:] == [
        ("sync", "go_away", ("system", "f1.frame", "f2.frame")),
        ("reply", 7, ("system", "f1.frame", "f2.frame")),
    ]
    agent.close()


def _kinds(posted) -> list[tuple]:
    """Each event posted: an arrival as (generation, call id), a cancellation as
    ("cancelled", generation, call ids)."""
    return [
        ("cancelled", e.generation, e.call_ids)
        if isinstance(e, CallsCancelled)
        else (e.generation, e.call_id)
        for e in posted
    ]


def test_a_cancellation_names_the_calls_it_withdraws_by_the_generation_they_came_in(
    monkeypatch, clock
):
    """Each call is posted before the cancellation of it. A call the agent never handed over,
    one that was no action or one it never had, is no one's to withdraw."""
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(
        Reply(
            (
                _click("c1", 0.25, 2),
                Call(0.375, "c2", "fly", {"to": "moon"}),
                _click("c3", 0.5, 2),
                Cancel(0.75, ("c3", "c2", "c1", "zz")),
            ),
            ends_at=1.0,
        )
    )
    agent.generate(7, 3, frame, BUDGET)
    assert _kinds(posted) == [(7, "c1"), (7, "c3"), ("cancelled", 7, ("c3", "c1"))]
    agent.close()


def test_a_reply_may_carry_the_cancellation_of_an_earlier_replys_calls(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies += [
        Reply((_click("c1", 0.25, 2),), ends_at=0.5),
        Reply((_click("c2", 0.25, 2), Cancel(0.5, ("c1", "c2"))), ends_at=1.0),
    ]
    agent.generate(7, 3, frame, BUDGET)
    agent.accept(7)
    agent.generate(8, 4, frame, BUDGET)
    assert _kinds(posted) == [
        (7, "c1"),
        (8, "c2"),
        ("cancelled", 7, ("c1",)),
        ("cancelled", 8, ("c2",)),
    ]
    agent.close()


# ---- how a reply ends, as a turn's does (harnesses/default/context.py _run_turn) -----------------


def test_a_fatal_provider_error_fails_the_reply_and_its_session_is_rebuilt(monkeypatch, clock):
    """A provider fault is no reply of the model's: the run cannot be scored, as a turn's."""
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(
        Reply((_click("c1", 0.25, 1), Error(0.5, "response failed")), ends_at=1.0)
    )
    with pytest.raises(RuntimeError, match="response failed"):
        agent.generate(7, 3, frame, BUDGET)
    assert [a.call_id for a in posted] == ["c1"]
    agent.push(observation(clock, 3))
    assert agent.driver.calls[-2][:2] == ("sync", "runtime_recovery")
    agent.close()


def test_a_provider_error_that_is_not_fatal_leaves_the_reply_going(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(
        Reply((Error(0.25, "cancel race", fatal=False), _click("c1", 0.5, 1)), ends_at=1.0)
    )
    r = agent.generate(7, 3, frame, BUDGET).result
    assert [a.call_id for a in posted] == ["c1"]
    assert r["action_protocol"]["protocol_failure"] is None
    agent.close()


@pytest.mark.parametrize("boundaries", [("interrupted",), ()], ids=["interrupted", "no_boundary"])
def test_a_text_reply_cut_short_is_parsed_as_a_turns_text_and_its_session_rebuilt(
    monkeypatch, clock, boundaries
):
    """A response that ended other than completed, or whose connection closed first, leaves the
    session unknown: its text is the reply's all the same, and the session is rebuilt from the
    view, which holds it once accepted, before it is used again."""
    agent = make_agent(monkeypatch, clock, TextLiveDriver, REALTIME)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    reply = Reply((Text(0.25, '{"action": "cli'),), ends_at=1.0, boundaries=boundaries)
    agent.driver.replies.append(reply)
    r = agent.generate(0, 0, frame, BUDGET).result
    assert posted == []
    assert r["action_protocol"]["protocol_failure"] == "invalid_action_output"
    agent.accept(0)
    agent.push(observation(clock, 3))
    assert agent.driver.calls[-2] == (
        "sync",
        "runtime_recovery",
        ("system", "f1.frame", "f2.frame", "d0.output"),
    )
    agent.close()


@pytest.mark.parametrize(
    ("messages", "usage", "boundaries"),
    [
        ((), {"output_tokens": 3}, ()),
        (("c1",), None, ("interrupted", "turn_complete")),
        ((), {"output_tokens": 3}, ("interrupted", "turn_complete")),
    ],
    ids=["no_turn_complete", "interrupted_without_usage", "interrupted_before_a_call"],
)
def test_a_function_call_reply_that_did_not_complete_fails(
    monkeypatch, clock, messages, usage, boundaries
):
    """As a native tool turn (FunctionCallTurnCollector.finish): a reply that never came to its
    turn_complete, or was interrupted before it handed a call over or without its usage, is no
    result."""
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    calls = tuple(_click(call_id, 0.25, 1) for call_id in messages)
    agent.driver.replies.append(Reply(calls, ends_at=1.0, usage=usage, boundaries=boundaries))
    with pytest.raises(RuntimeError, match="complete"):
        agent.generate(7, 3, frame, BUDGET)
    agent.close()


def test_a_function_call_reply_interrupted_after_its_call_stands_and_its_session_is_rebuilt(
    monkeypatch, clock
):
    """The call already fixed the action, and the terminal usage came: as a native tool turn,
    the interruption does not undo it, and leaves the session to be rebuilt."""
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(
        Reply(
            (_click("c1", 0.25, 1),),
            ends_at=1.0,
            usage={"output_tokens": 3},
            boundaries=("interrupted", "turn_complete"),
        )
    )
    agent.generate(7, 3, frame, BUDGET)
    assert [a.call_id for a in posted] == ["c1"]
    agent.accept(7)
    agent.push(observation(clock, 3))
    assert agent.driver.calls[-2][:2] == ("sync", "runtime_recovery")
    agent.close()


def test_a_failed_reply_the_kernel_accepts_adds_nothing(monkeypatch, clock):
    """The kernel accepts the result of the reply it was waiting for, failed or not
    (kernel/core.py): a failed one, as a turn's, joins nothing."""
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(
        Reply((_click("c1", 0.25, 1), Fail(0.5, RuntimeError("stream broke"))), ends_at=1.0)
    )
    with pytest.raises(RuntimeError):
        agent.generate(7, 3, frame, BUDGET)
    agent.accept(7)
    agent.push(observation(clock, 3))
    assert "d3.output" not in agent.driver.calls[-1][2]
    assert agent.token_usage["responses"] == 0
    agent.close()


def test_a_reply_aborted_while_a_push_reopens_the_session_is_never_asked_for(monkeypatch, clock):
    """A push reopens the session an abort closed; the next reply waits for it, and is aborted
    meanwhile. Once the session is open the reply is not asked for, as a turn is not once
    abandoned (context.py _run_turn): it would run on the session beside the next one."""
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.abort(6)
    agent.driver.connect_s = 2.0
    pushed, _ = _on_worker(clock, lambda: agent.push(observation(clock, 3)))
    clock.sleep(0.5)
    agent.driver.replies.append(Reply(ends_at=1.0))
    replied, out = _on_worker(clock, lambda: agent.generate(7, 3, frame, BUDGET))
    clock.sleep(0.5)
    agent.abort(7)
    assert clock.wait(replied, 5.0) and clock.wait(pushed, 5.0)
    assert out["value"].cancelled
    assert not [call for call in agent.driver.calls if call[0] == "reply"]
    agent.close()


def test_a_replys_output_comes_before_the_feedback_committed_before_it_was_accepted(
    monkeypatch, clock
):
    """A wait is settled where it arrives, before its reply is accepted: a session rebuilt from
    the view shows the model its action before the result of it."""
    agent = make_agent(monkeypatch, clock, TextLiveDriver, REALTIME)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((Text(0.25, '{"action": "wait", "frame": 2}'),), 1.0))
    agent.generate(0, 0, frame, BUDGET)
    [arrival] = posted
    agent.commit(_outcome(0, "wait", "text", action=arrival.action))
    agent.accept(0)
    agent.push(observation(clock, 3))
    assert agent.driver.calls[-1][2][3:] == ("d0.output", "d0.text.feedback", "f3.frame")
    agent.close()


def test_a_replys_output_goes_where_it_began_before_its_actions_feedback(monkeypatch, clock):
    """A click executes while its reply goes on, and a frame is pushed meanwhile: the reply's
    output stands where it began, before both."""
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((_click("c1", 0.25, 2),), ends_at=1.0))
    replied, _ = _on_worker(clock, lambda: agent.generate(7, 3, frame, BUDGET))
    clock.sleep(0.5)
    agent.commit(_outcome(3, "applied", "c1", action=posted[0].action))
    agent.push(observation(clock, 3))
    assert clock.wait(replied, 1.0)
    agent.accept(7)
    agent.push(observation(clock, 4))
    assert agent.driver.calls[-1][2][3:] == ("d3.output", "d3.c1.feedback", "f3.frame", "f4.frame")
    agent.close()


def test_an_action_that_comes_as_a_push_goes_out_finds_that_frame_sent(monkeypatch, clock):
    """The frame is in the session once its push went out: an action that comes in that same
    instant may name it, and counts it in the age of an older frame it names."""
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    first = observation(clock, 1)
    agent.push(first)
    calls = tuple(
        Call(0.0, f"c{n}", "click", {"x": 1, "y": 1, "button": "left", "frame": n}, "obs-2")
        for n in (1, 2)
    )
    agent.driver.replies.append(Reply(calls, ends_at=1.0))
    replied, _ = _on_worker(clock, lambda: agent.generate(7, 3, first, BUDGET))
    clock.sleep(0.5)
    agent.push(observation(clock, 2))
    assert clock.wait(replied, 1.0)
    assert [(a.call_id, a.rejection, a.frame, a.frame_age_frames) for a in posted] == [
        ("c1", None, 1, 1),
        ("c2", None, 2, 0),
    ]
    agent.close()


# ---- what the agent keeps of the frames -------------------------------------------------------


def test_the_agent_lets_go_of_a_frames_screenshot_once_it_folds(monkeypatch, clock):
    """As a turn-based history folds a screenshot (CanonicalTurn.fold_visual): a frame beyond
    the retained count keeps no image, and only the frames a reply may still start on, the two
    newest, stay with their prepared screenshots."""
    agent = make_agent(monkeypatch, clock)
    for n in range(1, 7):
        agent.push(observation(clock, n))
    frames = [block for block in agent._blocks if block.kind == "frame"]
    assert [block.image is not None for block in frames] == [False] * 4 + [True] * 2
    assert list(agent._pushed) == ["obs-5", "obs-6"]
    agent.close()


def test_a_frames_inspection_is_finished_once_with_every_reply_on_it(monkeypatch, clock):
    """Two replies start on frame 1; once frames 2 and 3 are pushed no reply can start on it
    again, and its inspection is finished with both replies' outputs. The frames left are
    finished at close, without a reply; a frame whose push failed, at once."""
    agent = make_agent(monkeypatch, clock)
    runtime = RecordingToolRuntime()
    agent.bind_tool_runtime(runtime)
    agent.bind_inbox(lambda event: None, ORIGIN)
    first = observation(clock, 1)
    agent.push(first)
    agent.driver.replies += [
        Reply((_click("a", 0.5, 1),), 1.0),
        Reply((_click("b", 0.25, 1),), 1.0),
    ]
    for generation in (0, 1):
        agent.generate(generation, generation, first, BUDGET)
        agent.accept(generation)
    agent.driver.failing.add("obs-2")
    with pytest.raises(RuntimeError):
        agent.push(observation(clock, 2))
    assert runtime.rows == [("obs-2", "push_failed", "", 0.0)]
    agent.push(observation(clock, 3))
    assert len(runtime.rows) == 1
    agent.push(observation(clock, 4))
    outputs = "\n".join(['{"action": "click", "x": 10, "y": 20, "button": "left", "frame": 1}'] * 2)
    assert runtime.rows[1] == ("obs-1", "completed", outputs, 2.0)
    agent.close()
    assert runtime.rows[2:] == [("obs-3", "no_reply", "", 0.0), ("obs-4", "no_reply", "", 0.0)]


def test_feedback_says_what_the_model_asked_for_not_what_ran(monkeypatch, clock):
    """The policy runs a done as a wait (evaluation/kernel/streaming.py _arrived): the outcome
    holds the wait, and the feedback says the done was asked for."""
    agent = make_agent(monkeypatch, clock, TextLiveDriver, REALTIME)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((Text(0.25, '{"action": "done", "frame": 2}'),), 1.0))
    agent.generate(0, 0, frame, BUDGET)
    agent.accept(0)
    [arrival] = posted
    asked = action_text(arrival.action)
    outcome = _outcome(
        0, "wait", "text", action=Action.wait(), protocol_failure="unsupported_terminal_action"
    )
    agent.commit(outcome)
    assert asked.startswith("done")
    assert agent.action_history[-1]["requested"] == asked
    agent.push(observation(clock, 3))
    feedback = agent.driver.views[-1].items[-2].block
    assert (feedback.id, feedback.text) == ("d0.text.feedback", execution_feedback(outcome, asked))
    agent.close()


# ---- the end -----------------------------------------------------------------------------------


def test_closing_ends_a_reply_the_episode_left_with_a_discarded_row(monkeypatch, clock, tmp_path):
    """After Termination the kernel sends no AcceptResult or DiscardResult (kernel/core.py): a
    reply still on is given up at close and written as discarded, as DefaultHarness.close
    discards the generations left."""
    agent = make_agent(monkeypatch, clock, log_dir=tmp_path)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((_click("c1", 0.25, 1),), ends_at=1.0))
    agent.generate(7, 3, frame, BUDGET)
    agent.close()
    rows = [json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    assert [(row["type"], row["generation"]) for row in rows] == [("reply_discarded", 7)]


def test_closing_while_a_push_reopens_the_session_closes_what_it_opened(monkeypatch, clock):
    """The push's reopen is under way when the episode closes the agent: close waits for it and
    closes the session it opened, whether the frame went in first or not."""
    agent = make_agent(monkeypatch, clock)
    _two_frames(agent, clock)
    agent.abort(6)
    agent.driver.connect_s = 2.0
    pushed, _ = _on_worker(clock, lambda: agent.push(observation(clock, 3)))
    clock.sleep(0.5)
    agent.close()
    assert not agent.driver.open
    operations = [call[0] for call in agent.driver.calls]
    assert operations[-1] == "close"
    assert operations.index("connect", 1) < len(operations) - 1
    assert clock.wait(pushed, 1.0)


# ---- what the trace says of the session --------------------------------------------------------


def test_the_trace_tells_a_resumed_renewal_from_a_rebuild_as_a_turns_does(monkeypatch, clock):
    """As RealtimeRuntime._record_sync and _record_reconnect: each session operation is a
    HistoryChanged of the kind the driver's result reports, with the result's detail and
    metadata, and a SessionStateChanged says whether the provider's context is known."""
    trace = RecordingTrace()
    agent = make_agent(monkeypatch, clock, trace=trace)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(
        Reply(
            (_click("c1", 0.25, 1),),
            ends_at=1.0,
            usage={"output_tokens": 3},
            boundaries=("interrupted", "turn_complete"),
        )
    )
    agent.generate(7, 3, frame, BUDGET)
    agent.accept(7)
    agent.push(observation(clock, 3))
    agent.driver.resumes = True
    agent.driver.renewal_required = True
    agent.push(observation(clock, 4))
    agent.abort(8)
    agent.push(observation(clock, 5))
    changes = trace.of(TraceEvent.HISTORY_CHANGED)
    assert [(c["kind"], c["trigger"], c["sync"]["reason"]) for c in changes] == [
        ("rebuild", "recovery", "runtime_recovery"),
        ("session_resumed", "provider", "go_away"),
        ("rebuild", "recovery", "reconnect"),
    ]
    assert changes[1]["sync"]["provider_metadata"]["session_resumed"] is True
    states = trace.of(TraceEvent.SESSION_STATE_CHANGED)
    assert [(s["state"], s["cause"], s["generation_id"]) for s in states] == [
        ("unknown", "generation_terminal_state", "streaming-7"),
        ("exact", "sync:runtime_recovery", None),
        ("unknown", "session_resumed", None),
        ("exact", "reconnect", None),
    ]
    agent.close()


def test_an_abort_naming_no_generation_gives_up_the_reply_on(monkeypatch, clock):
    """As KernelAgent.abort: None names whichever reply is on."""
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply(ends_at=100.0))
    replied, out = _on_worker(clock, lambda: agent.generate(7, 3, frame, BUDGET))
    clock.sleep(1.0)
    agent.abort(None)
    assert agent.driver.calls[-2:] == [("interrupt", "streaming-7"), ("close",)]
    assert clock.wait(replied, 1.0)
    assert out["value"].cancelled
    agent.close()


def test_a_push_waiting_for_a_renewal_goes_in_once_the_reply_is_accepted_whatever_comes_next(
    monkeypatch, clock
):
    """The reply after the one the push waited for starts at once, renewing the session, and goes
    on it: the push goes in then, while that reply is on, not once it ends."""
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies += [Reply((GoAway(0.5),), ends_at=2.0), Reply(ends_at=5.0)]
    replied, _ = _on_worker(clock, lambda: agent.generate(7, 3, frame, BUDGET))
    clock.sleep(1.0)
    pushed, pushing = _on_worker(clock, lambda: agent.push(observation(clock, 3)))
    assert clock.wait(replied, 2.0)
    agent.accept(7)
    next_reply, _ = _on_worker(clock, lambda: agent.generate(8, 4, frame, BUDGET))
    assert clock.wait(pushed, 1.0)
    assert "error" not in pushing
    assert not next_reply.is_set()
    operations = [call[:2] for call in agent.driver.calls]
    assert operations.index(("sync", "go_away")) < operations.index(("push", "obs-3"))
    assert clock.wait(next_reply, 10.0)
    agent.close()


def test_closing_while_a_reply_thread_retires_its_frame_finishes_every_frame(
    monkeypatch, clock, tmp_path
):
    """The episode ended with a reply's worker still in generate(): stopping the execution thread
    wakes it, and it retires its frame while close() finishes the frames left. Fixed here so: the
    worker retires once close is finishing the first frame. close() neither raises nor loses a
    frame's inspection, and still closes the tool runtime (review-final-streaming 中-6)."""
    agent = make_agent(monkeypatch, clock, log_dir=tmp_path)
    closing, retired = clock.event(), clock.event()
    worker_ids = set()

    class GatedToolRuntime(RecordingToolRuntime):
        closed = False

        def finish(self, execution, **fields):
            if threading.get_ident() not in worker_ids and not closing.is_set():
                closing.set()
                assert clock.wait(retired, 5.0)
            super().finish(execution, **fields)

        def close(self):
            self.closed = True

    runtime = GatedToolRuntime()
    agent.bind_tool_runtime(runtime)
    agent.bind_inbox(lambda event: None, ORIGIN)
    retire = agent._retire

    def gated_retire():
        if threading.get_ident() in worker_ids:
            assert clock.wait(closing, 5.0)
            retire()
            retired.set()
        else:
            retire()

    monkeypatch.setattr(agent, "_retire", gated_retire)
    first = observation(clock, 1)
    agent.push(first)
    agent.driver.replies.append(Reply((Hang(0.25),), ends_at=100.0))

    def reply():
        worker_ids.add(threading.get_ident())
        return agent.generate(7, 3, first, BUDGET)

    replied, _ = _on_worker(clock, reply)
    clock.sleep(0.5)
    agent.push(observation(clock, 2))
    agent.push(observation(clock, 3))
    agent.close()
    assert clock.wait(replied, 5.0)
    assert sorted(row[0] for row in runtime.rows) == ["obs-1", "obs-2", "obs-3"]
    assert runtime.closed


# ---- reopening the session under the shared retry policy (decision 39) -------------------------


def _ready_attempts(trace) -> list[tuple]:
    return [
        (r["outcome"], r["failure_kind"])
        for r in trace.of(TraceEvent.MODEL_ATTEMPT_FINISHED)
        if r["kind"] == "session_ready"
    ]


def test_a_reply_whose_session_fails_to_reopen_once_is_tried_again(monkeypatch, clock):
    """An abort closed the session; reopening it for the next reply meets a transient failure
    once. Nothing was generated yet, so it is tried again after the shared backoff, as a turn's
    connect is, and the reply goes on (review-final-streaming 中-4)."""
    trace = RecordingTrace()
    agent = make_agent(monkeypatch, clock, trace=trace)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.abort(6)
    agent.driver.connect_failures = [ConnectionError("1011 internal error")]
    agent.driver.replies.append(Reply(ends_at=1.0))
    pending = agent.generate(7, 3, frame, BUDGET)
    assert not pending.cancelled
    assert agent.driver.connect_attempts == 3  # the episode's first, the failed one, the retry
    assert [call[:2] for call in agent.driver.calls][-2:] == [
        ("connect", ("system", "f1.frame", "f2.frame")),
        ("reply", 7),
    ]
    assert _ready_attempts(trace) == [("retry", TRANSIENT), ("ok", None)]
    agent.close()


def test_a_push_whose_session_fails_to_reopen_once_is_tried_again(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    _two_frames(agent, clock)
    agent.abort(6)
    agent.driver.connect_failures = [ConnectionError("1011 internal error")]
    agent.push(observation(clock, 3))
    assert agent.driver.calls[-1][:2] == ("push", "obs-3")
    agent.close()


def test_a_session_that_keeps_failing_to_reopen_fails_the_reply_once_the_retries_are_spent(
    monkeypatch, clock
):
    """Four attempts, as the shared policy allows (decision 39), then the reply fails as an
    exhausted transport failure."""
    trace = RecordingTrace()
    agent = make_agent(monkeypatch, clock, trace=trace)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.abort(6)
    agent.driver.connect_failures = [ConnectionError("down")] * 4
    with pytest.raises(TransportFailureError):
        agent.generate(7, 3, frame, BUDGET)
    assert _ready_attempts(trace) == [("retry", TRANSIENT)] * 3 + [("failed", TRANSIENT)]
    assert not [call for call in agent.driver.calls if call[0] == "reply"]
    agent.close()


def test_a_session_that_fails_to_reopen_for_good_is_not_tried_again(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.abort(6)
    agent.driver.connect_failures = [ValueError("bad key")]
    with pytest.raises(TransportFailureError):
        agent.generate(7, 3, frame, BUDGET)
    assert agent.driver.connect_attempts == 2
    agent.close()


def test_closing_ends_a_push_waiting_to_try_reopening_the_session_again(monkeypatch, clock):
    """A push waits out the backoff between attempts to reopen the session when the episode
    closes the agent: close does not wait for the attempts left, and the push fails."""
    agent = make_agent(monkeypatch, clock)
    _two_frames(agent, clock)
    agent.abort(6)
    agent.driver.connect_failures = [ConnectionError("down")] * 3
    pushed, pushing = _on_worker(clock, lambda: agent.push(observation(clock, 3)))
    clock.sleep(0.5)
    started = clock.monotonic()
    agent.close()
    assert clock.monotonic() - started < 0.5
    assert clock.wait(pushed, 1.0)
    assert "error" in pushing


def test_a_push_held_for_a_renewal_tells_the_kernel_when_it_waits_and_when_it_goes_on(
    monkeypatch, clock
):
    """The held push is waiting, not stuck (review-final-streaming 中-2): PushHeld goes to the
    kernel as it starts waiting, PushResumed once the reply it waits for is off the session,
    before the frame goes in."""
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(lambda event: posted.append((event, clock.monotonic())), ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((GoAway(0.5),), ends_at=2.0))
    replied, _ = _on_worker(clock, lambda: agent.generate(7, 3, frame, BUDGET))
    clock.sleep(1.0)
    pushed, _ = _on_worker(clock, lambda: agent.push(observation(clock, 3)))
    assert clock.wait(replied, 2.0)
    clock.sleep(0.5)
    agent.accept(7)
    assert clock.wait(pushed, 1.0)
    holds = [
        (type(event).__name__, event.observation_id, at)
        for event, at in posted
        if isinstance(event, (PushHeld, PushResumed))
    ]
    assert holds == [("PushHeld", "obs-3", 2.0), ("PushResumed", "obs-3", 3.5)]
    agent.close()


# ---- a goAway read while an input waits for the session (review-final-drivers 中 5) ------------


def test_a_reply_refused_for_a_go_away_read_after_its_renewal_check_renews_and_is_asked_again(
    monkeypatch, clock
):
    """The trigger waited for the session while a push wrote, and the reader read a goAway
    meanwhile: refused before anything went out (RenewalRequired), the reply renews the session
    and is asked for again, instead of failing the run."""
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.go_away_on = ["reply"]
    agent.driver.replies.append(Reply(ends_at=1.0))
    pending = agent.generate(7, 3, frame, BUDGET)
    assert not pending.cancelled
    assert [call[:2] for call in agent.driver.calls][-4:] == [
        ("reply", 7),
        ("close",),
        ("sync", "go_away"),
        ("reply", 7),
    ]
    agent.close()


def test_a_push_refused_for_a_go_away_read_after_its_check_renews_and_goes_in(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    _two_frames(agent, clock)
    agent.driver.go_away_on = ["push"]
    agent.push(observation(clock, 3))
    assert [call[:2] for call in agent.driver.calls][-4:] == [
        ("push", "obs-3"),
        ("close",),
        ("sync", "go_away"),
        ("push", "obs-3"),
    ]
    record, _ = agent.frames.locate(3)
    assert record.send_status == "sent"
    agent.close()


def test_a_push_refused_for_a_go_away_while_a_reply_is_on_waits_for_it(monkeypatch, clock):
    """Refused the same way with a reply on the session, which renewing would end: the push
    waits at its gate, as one that saw the goAway first does."""
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply(ends_at=2.0))
    replied, _ = _on_worker(clock, lambda: agent.generate(7, 3, frame, BUDGET))
    clock.sleep(0.5)
    agent.driver.go_away_on = ["push"]
    pushed, pushing = _on_worker(clock, lambda: agent.push(observation(clock, 3)))
    assert clock.wait(replied, 2.0)
    clock.sleep(0.5)
    assert not pushed.is_set()
    agent.accept(7)
    assert clock.wait(pushed, 1.0)
    assert "error" not in pushing
    assert [type(e).__name__ for e in posted if isinstance(e, (PushHeld, PushResumed))] == [
        "PushHeld",
        "PushResumed",
    ]
    assert agent.driver.calls[-2][:2] == ("sync", "go_away")
    agent.close()


def test_a_close_that_outlasts_its_bound_is_logged_not_raised(monkeypatch, clock, caplog):
    """A push reopening the session takes longer than close() waits (review-final-drivers 中 3):
    as a turn's close (RealtimeRuntime.close), the timeout is logged and the episode's outcome
    stands; stopping the execution thread ends the push."""
    agent = make_agent(monkeypatch, clock)
    _two_frames(agent, clock)
    agent.abort(6)
    agent.driver.connect_s = 20.0
    pushed, pushing = _on_worker(clock, lambda: agent.push(observation(clock, 3)))
    clock.sleep(0.5)
    with caplog.at_level("WARNING"):
        agent.close()
    assert "close" in caplog.text
    assert clock.wait(pushed, 1.0)
    assert "error" in pushing
