"""The local frame table numbers every prepared frame, keeps its geometry and marks for the whole
episode, and records when it was sent."""

from __future__ import annotations

from realtime_framework.harnesses.default.frames import FrameTable

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
    assert (second.observation_id, second.captured_unix_s, second.marks) == ("obs-2", 101.0, None)


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
