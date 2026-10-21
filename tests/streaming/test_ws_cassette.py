"""The cassette format and its strict cursor (tests/streaming/ws_cassette.py), without sockets."""

from __future__ import annotations

import pytest

from tests.replay import baseline

from . import ws_cassette
from .ws_cassette import Cursor


def _record(direction, msg=None, at=None, conn=1):
    return {"conn": conn, "dir": direction, "at": at, "msg": msg}


def _create(item_id: str) -> dict:
    return {"type": "conversation.item.create", "item": {"id": item_id, "role": "user"}}


def _added(item_id: str) -> dict:
    return {"type": "conversation.item.added", "item": {"id": item_id}}


RESPONSE_CREATE = {"type": "response.create"}
RESPONSE_DONE = {"type": "response.done", "response": {"id": "resp_1"}}


def _cursor(records) -> Cursor:
    return Cursor(records, baseline.Normalizer())


def test_a_cassette_reads_back_as_written(tmp_path):
    records = [
        _record("open", at=0.0),
        _record("c2s", _create("<rtb_user-1>"), 0.5),
        _record("s2c", _added("<rtb_user-1>")),
        _record("close", {"by": "server", "code": 1011, "reason": "scripted"}, 2.0),
    ]
    path = tmp_path / "session.jsonl"
    ws_cassette.write(path, records)
    assert ws_cassette.read(path) == records


def test_a_record_must_have_exactly_the_four_fields(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text('{"conn": 1, "dir": "open", "at": 0}\n')
    with pytest.raises(AssertionError):
        ws_cassette.read(path)


def test_reactive_messages_follow_their_trigger_and_timed_ones_their_time():
    cursor = _cursor(
        [
            _record("open", at=0.0),
            _record("s2c", {"type": "session.created"}),
            _record("c2s", RESPONSE_CREATE, 0.5),
            _record("s2c", {"type": "response.created"}),
            _record("s2c", RESPONSE_DONE, 2.0),
        ]
    )
    assert cursor.open(0.0) == 1
    assert [r["msg"]["type"] for r in cursor.release(1, 0.0)] == ["session.created"]
    assert cursor.release(1, 0.0) == []  # the next server record waits for the client
    cursor.receive(1, RESPONSE_CREATE, 0.5)
    assert [r["msg"]["type"] for r in cursor.release(1, 0.5)] == ["response.created"]
    assert cursor.next_due(1) == 2.0
    assert cursor.release(1, 1.999) == []
    assert [r["msg"]["type"] for r in cursor.release(1, 2.0)] == ["response.done"]
    assert cursor.next_due(1) is None
    assert cursor.finish() == []


def test_a_released_message_carries_this_sessions_ids():
    """Placeholders of ids the client sent become the ids it sent this time; an id the server
    made up keeps its placeholder."""
    cursor = _cursor(
        [
            _record("open", at=0.0),
            _record("c2s", _create("<rtb_user-1>"), 0.0),
            _record("s2c", {**_added("<rtb_user-1>"), "next": "<server_item-1>"}),
        ]
    )
    cursor.open(0.0)
    cursor.receive(1, _create("rtb_user_0a1b2c"), 0.0)
    (released,) = cursor.release(1, 0.0)
    assert released["msg"] == {**_added("rtb_user_0a1b2c"), "next": "<server_item-1>"}
    assert cursor.finish() == []


def test_a_message_sent_before_a_server_message_recorded_before_it_is_early():
    cursor = _cursor(
        [
            _record("open", at=0.0),
            _record("c2s", RESPONSE_CREATE, 1.0),
            _record("s2c", RESPONSE_DONE, 5.0),
            _record("c2s", _create("<rtb_user-1>"), 5.0),
        ]
    )
    cursor.open(0.0)
    cursor.receive(1, RESPONSE_CREATE, 1.0)
    assert cursor.release(1, 3.0) == []
    cursor.receive(1, _create("rtb_user_77"), 3.0)
    assert cursor.problems == [
        (
            "conn 1 record 2: c2s conversation.item.create sent at t=3, before the recorded"
            " s2c response.done at t=5"
        )
    ]
    assert cursor.release(1, 5.0) == []  # nothing more is released after a problem


def test_a_message_that_differs_names_the_field():
    cursor = _cursor([_record("open", at=0.0), _record("c2s", _create("<rtb_user-1>"), 0.0)])
    cursor.open(0.0)
    cursor.receive(1, {**_create("rtb_user_1"), "item": {"id": "rtb_user_1", "role": "system"}}, 0)
    assert cursor.problems == [
        'conn 1 record 1: c2s conversation.item.create differs:\n  msg.item.role: "user" -> "system"'
    ]


def test_records_left_unused_are_a_problem():
    cursor = _cursor(
        [
            _record("open", at=0.0),
            _record("c2s", RESPONSE_CREATE, 0.0),
            _record("s2c", RESPONSE_DONE, 1.0),
            _record("open", at=2.0, conn=2),
        ]
    )
    cursor.open(0.0)
    cursor.receive(1, RESPONSE_CREATE, 0.0)
    assert cursor.finish() == [
        "conn 1: 1 record(s) left unused, from record 2 (s2c response.done)",
        "conn 2: 1 record(s) left unused, from record 3 (open)",
    ]


def test_more_than_the_recording_is_a_problem():
    cursor = _cursor([_record("open", at=0.0), _record("c2s", RESPONSE_CREATE, 0.0)])
    cursor.open(0.0)
    cursor.receive(1, RESPONSE_CREATE, 0.0)
    cursor.receive(1, RESPONSE_CREATE, 0.0)
    assert cursor.problems == [
        "conn 1: c2s response.create sent, but the recording has nothing more on it"
    ]
    unrecorded = _cursor([_record("open", at=0.0)])
    unrecorded.open(0.0)
    assert unrecorded.open(1.0) is None
    assert unrecorded.problems == ["conn 2: opened, but the recording has nothing more on it"]


def test_a_client_record_with_a_time_must_happen_at_that_time():
    """A hand-written script's times are compared on the fake clock, to within its rounding."""
    records = [_record("open", at=0.0), _record("c2s", RESPONSE_CREATE, 1.0)]
    for sent_at, problems in [
        (1.0 + 1e-12, []),
        (1.5, ["conn 1 record 1: c2s response.create at t=1.5, recorded at t=1"]),
        (0.5, ["conn 1 record 1: c2s response.create at t=0.5, recorded at t=1"]),
    ]:
        cursor = _cursor(records)
        cursor.open(0.0)
        cursor.receive(1, RESPONSE_CREATE, sent_at)
        assert cursor.finish() == problems


def test_a_converted_recordings_client_records_are_compared_in_order_only():
    records = [_record("open"), _record("c2s", RESPONSE_CREATE), _record("s2c", RESPONSE_DONE, 0.4)]
    cursor = _cursor(records)
    cursor.open(7.0)
    cursor.receive(1, RESPONSE_CREATE, 0.2)
    assert cursor.release(1, 0.4) == [_record("s2c", RESPONSE_DONE, 0.4)]
    assert cursor.finish() == []


def test_a_client_close():
    # Without a recorded close the client may close once the records are used.
    cursor = _cursor([_record("open", at=0.0)])
    cursor.open(0.0)
    cursor.client_closed(1, 1000, 3.0)
    assert cursor.finish() == []
    # A recorded close must come next, with its code.
    for code, problems in [
        (1000, []),
        (1001, ["conn 1 record 1: the client closed it with 1001, recorded 1000"]),
    ]:
        cursor = _cursor(
            [_record("open", at=0.0), _record("close", {"by": "client", "code": 1000}, 1.0)]
        )
        cursor.open(0.0)
        cursor.client_closed(1, code, 1.0)
        assert cursor.finish() == problems
    # A close while records are left.
    cursor = _cursor([_record("open", at=0.0), _record("s2c", RESPONSE_DONE, 4.0)])
    cursor.open(0.0)
    cursor.client_closed(1, 1000, 1.0)
    assert cursor.problems == [
        (
            "conn 1: the client closed it (1000), but the recording has s2c response.done next"
            " (record 1)"
        )
    ]


def test_a_server_close_is_released_like_a_timed_message():
    close = {"by": "server", "code": 1011, "reason": "scripted"}
    cursor = _cursor([_record("open", at=0.0), _record("close", close, 2.0)])
    cursor.open(0.0)
    assert cursor.next_due(1) == 2.0
    assert cursor.release(1, 2.0) == [_record("close", close, 2.0)]
    assert cursor.finish() == []


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        # A pushed frame's item (providers/openai_realtime_session.py:290).
        ({"type": "conversation.item.create", "item": {"id": "rtb_user_1"}}, "push"),
        # A cancel (providers/openai_realtime.py:548).
        ({"type": "response.cancel"}, "cancel"),
    ],
)
def test_a_pushed_frames_and_a_cancels_event_ids_are_our_random_ids(message, kind):
    normalize = ws_cassette.NORMALIZERS["openai_realtime.ws"]
    recorded = normalize()({**message, "event_id": f"rtb_{kind}_0a1b2c"})
    assert recorded["event_id"] == f"<rtb_{kind}-1>"
    cursor = Cursor([_record("open", at=0.0), _record("c2s", recorded, 0.0)], normalize())
    cursor.open(0.0)
    cursor.receive(1, {**message, "event_id": f"rtb_{kind}_9f8e7d"}, 0.0)
    assert cursor.finish() == []


def test_in_a_converted_recording_the_client_may_go_ahead_of_the_servers_times():
    """M4: a converted recording compares the order only. A frame the client pushes on its own beat
    may come before a server message the recording read just ahead of it: that message is released
    then, ahead of its time, and the frame is compared as usual."""
    delta = {"type": "response.output_text.delta", "delta": "{"}
    records = [
        _record("open"),
        _record("c2s", RESPONSE_CREATE),
        _record("s2c", delta, 1.0002),
        _record("c2s", _create("<rtb_user-1>")),
        _record("s2c", _added("<rtb_user-1>")),
    ]
    cursor = _cursor(records)
    cursor.open(0.0)
    cursor.receive(1, RESPONSE_CREATE, 0.0)
    assert cursor.release(1, 1.0) == []
    cursor.receive(1, _create("rtb_user_77"), 1.0)
    assert [r["msg"] for r in cursor.release(1, 1.0)] == [delta, _added("rtb_user_77")]
    assert cursor.finish() == []


def test_nothing_gone_ahead_of_is_released_once_the_message_differs():
    records = [
        _record("open"),
        _record("s2c", {"type": "response.output_text.delta"}, 1.0),
        _record("c2s", RESPONSE_CREATE),
    ]
    cursor = _cursor(records)
    cursor.open(0.0)
    cursor.receive(1, {"type": "response.cancel"}, 0.5)
    assert cursor.failed
    assert cursor.release(1, 2.0) == []
