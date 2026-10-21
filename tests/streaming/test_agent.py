"""The native-streaming agent (harnesses/streaming/agent.py) on a fake live encoder
(tests/streaming/fake_live.py) and the fake clock: what it hands the encoder for each push, and
the view each connection operation is given."""

from __future__ import annotations

import pytest

from realtime_framework.entity.actions import ActionType
from realtime_framework.harnesses.default.frames import FRAME_NOT_IN_TABLE, INVALID_GROUNDING_FRAME
from tests.fake_clock import FakeClock

from .fake_live import (
    BUDGET,
    REALTIME,
    Call,
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
