"""The native-streaming agent (harnesses/streaming/agent.py) on a fake live encoder
(tests/streaming/fake_live.py) and the fake clock: what it hands the encoder for each push, and
the view each connection operation is given."""

from __future__ import annotations

import pytest

from realtime_framework.entity.actions import ActionType
from realtime_framework.evaluation.kernel.events import Outcome
from realtime_framework.evaluation.kernel.streaming import CallsCancelled
from realtime_framework.harnesses.default.frames import FRAME_NOT_IN_TABLE, INVALID_GROUNDING_FRAME
from realtime_framework.harnesses.streaming.feedback import action_text, execution_feedback
from realtime_framework.providers.gemini.live import GoAwayDeadlineExceeded
from tests.fake_clock import FakeClock

from .fake_live import (
    BUDGET,
    REALTIME,
    Call,
    Cancel,
    Fail,
    Reply,
    Text,
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
    agent = make_agent(monkeypatch, clock)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((_click("c1", 0.5, 1), _click("c2", 1.0, 2)), ends_at=1.5))
    pending = agent.generate(7, 3, frame, BUDGET)
    assert agent.driver.calls[-1] == ("reply", 7, ("system", "f1.frame", "f2.frame"))
    assert [
        (a.generation, a.call_id, a.arrived_at, a.frame, a.frame_age_s, a.frame_age_frames)
        for a in posted
    ] == [(7, "c1", 2.5, 1, 1.5, 1), (7, "c2", 3.0, 2, 1.0, 0)]
    assert all(a.action.action_type == ActionType.CLICK and a.rejection is None for a in posted)
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


def test_an_action_in_text_is_parsed_once_the_reply_ended_and_named_by_the_agent(
    monkeypatch, clock
):
    agent = make_agent(monkeypatch, clock, TextLiveDriver, REALTIME)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    frame = _two_frames(agent, clock)
    action = '{"action": "click", "x": 10, "y": 20, "button": "left", "frame": 2}'
    agent.driver.replies.append(Reply((Text(0.25, action[:20]), Text(0.5, action[20:])), 2.0))
    r = agent.generate(0, 0, frame, BUDGET).result
    [arrival] = posted
    assert (arrival.call_id, arrival.arrived_at, arrival.frame) == ("text", 4.0, 2)
    assert r["raw"] == action
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


def test_after_a_go_away_during_a_reply_pushes_fail_until_it_is_accepted_and_renewed(
    monkeypatch, clock
):
    """Renewing the connection would end the reply, and the session takes no input meanwhile:
    those frames are lost. Once the reply's output is in the view, the next push renews."""
    agent = make_agent(monkeypatch, clock)
    agent.bind_inbox(lambda event: None, ORIGIN)
    frame = _two_frames(agent, clock)
    agent.driver.replies.append(Reply((_click("c1", 0.25, 2),), ends_at=2.0))
    done, out = _on_worker(clock, lambda: agent.generate(7, 3, frame, BUDGET))
    clock.sleep(0.5)
    agent.driver.renewal_required = True
    with pytest.raises(RuntimeError, match="requires a renewal"):
        agent.push(observation(clock, 3))
    assert clock.wait(done, 2.0)
    assert not out["value"].cancelled
    agent.accept(7)
    agent.push(observation(clock, 4))
    assert agent.driver.calls[-2:] == [
        ("sync", "go_away", ("system", "f1.frame", "f2.frame", "d3.output")),
        ("push", "obs-4", ("system", "f1.frame", "f2.frame", "d3.output", "f4.frame")),
    ]
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
