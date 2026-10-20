"""The local frame table numbers every prepared frame, keeps its geometry and marks for the whole
episode, and records when it was sent."""

from __future__ import annotations

from realtime_framework.harnesses.default.frames import (
    FRAME_NOT_IN_TABLE,
    INVALID_FRAME_SYNTAX,
    MISSING_FRAME,
    FrameTable,
)

GEOMETRY = {"image_size": [768, 432], "screen_size": [1920, 1080]}


def _marks(x: int) -> dict:
    return {"P1": {"kind": "point", "point": [x, 100], "bbox": [x - 10, 90, x + 10, 110]}}


def test_frames_are_numbered_from_one_in_preparation_order():
    table = FrameTable()
    assert table.latest() is None
    first = table.register("obs-1", 100.0, GEOMETRY, _marks(10))
    second = table.register("obs-2", 101.0, GEOMETRY, None)
    assert (first.frame, second.frame) == (1, 2)
    assert table.latest() is second
    assert (second.observation_id, second.captured_at, second.marks) == ("obs-2", 101.0, None)


def test_a_frame_keeps_its_own_marks_and_geometry_after_newer_frames():
    table = FrameTable()
    old_geometry = {"image_size": [1920, 1080], "screen_size": [1920, 1080]}
    first = table.register("obs-1", 100.0, old_geometry, _marks(10))
    for index in range(2, 40):
        table.register(f"obs-{index}", 100.0 + index, GEOMETRY, _marks(10 * index))
    assert first.marks == _marks(10)
    assert first.geometry == old_geometry
    assert table.latest().marks == _marks(390)


def test_a_frame_is_prepared_until_it_is_sent():
    table = FrameTable()
    record = table.register("obs-1", 100.0, GEOMETRY, None)
    assert (record.send_status, record.sent_unix_s) == ("prepared", None)
    table.record_sent(record.frame, 100.25)
    assert (record.send_status, record.sent_unix_s) == ("sent", 100.25)


def test_locate_finds_any_sent_frame_by_number():
    table = FrameTable()
    records = [table.register(f"obs-{n}", 100.0 + n, GEOMETRY, _marks(n)) for n in (1, 2, 3)]
    for record in records:
        table.record_sent(record.frame, record.captured_at + 0.1)
    assert [table.locate(n) for n in (1, 2, 3)] == [(record, None) for record in records]


def test_locate_says_why_a_frame_reference_finds_nothing():
    table = FrameTable()
    table.record_sent(table.register("obs-1", 100.0, GEOMETRY, None).frame, 100.1)
    table.register("obs-2", 101.0, GEOMETRY, None)  # prepared, never sent
    assert table.locate(None) == (None, MISSING_FRAME)
    for reference in ("1", 1.0, True, [1], {"frame": 1}):
        assert table.locate(reference) == (None, INVALID_FRAME_SYNTAX)
    for reference in (0, -1, 2, 3):
        assert table.locate(reference) == (None, FRAME_NOT_IN_TABLE)


def test_age_counts_the_frames_sent_since_and_the_seconds_since_capture():
    table = FrameTable()
    records = [table.register(f"obs-{n}", 100.0 + n, GEOMETRY, None) for n in (1, 2, 3, 4)]
    for record in records[:3]:
        table.record_sent(record.frame, record.captured_at + 0.1)
    # Frame 4 is prepared but not sent yet: it does not count.
    assert table.age(records[0], 104.5) == {"age_frames": 2, "age_s": 3.5}
    assert table.age(records[2], 104.5) == {"age_frames": 0, "age_s": 1.5}
