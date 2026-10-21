"""What streaming adds to the frame table (harnesses/default/frames.py): use from several threads,
the provider's acknowledgement of each pushed frame, and frame ages that count only sent frames."""

from __future__ import annotations

import sys
import threading

from realtime_framework.harnesses.default.frames import FRAME_NOT_IN_TABLE, FrameTable

GEOMETRY = {"image_size": [768, 432], "screen_size": [1920, 1080]}


def test_frames_entered_from_several_threads_get_every_number_once():
    table = FrameTable()
    threads, per_thread = 8, 500
    entered: list[list] = [[] for _ in range(threads)]
    start = threading.Barrier(threads)

    def enter(lane: int) -> None:
        start.wait()
        for index in range(per_thread):
            entered[lane].append(table.register(f"obs-{lane}-{index}", 100.0, GEOMETRY, None))

    workers = [threading.Thread(target=enter, args=(lane,)) for lane in range(threads)]
    # Switch threads as often as the interpreter allows, so that an unlocked read of the table's
    # size and the insert after it would interleave.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
    finally:
        sys.setswitchinterval(interval)
    count = threads * per_thread
    numbers = sorted(record.frame for lane in entered for record in lane)
    assert numbers == list(range(1, count + 1))
    assert table.latest().frame == count


def test_a_frame_has_no_acknowledgement_until_the_driver_reports_one():
    table = FrameTable()
    record = table.register("obs-1", 100.0, GEOMETRY, None)
    table.record_sent(record.frame, 100.2)
    assert (record.ack_status, record.acked_unix_s) == (None, None)


def test_an_acknowledged_frame_keeps_when_the_confirmation_arrived():
    table = FrameTable()
    record = table.register("obs-1", 100.0, GEOMETRY, None)
    table.record_sent(record.frame, 100.2)
    table.record_acked(record.frame, 100.45)
    assert (record.send_status, record.sent_unix_s) == ("sent", 100.2)
    assert (record.ack_status, record.acked_unix_s) == ("acked", 100.45)


def test_a_confirmation_read_before_the_push_returned_survives_the_send_record():
    # The driver can read the server's confirmation on its own thread before the pushing thread
    # records the send.
    table = FrameTable()
    record = table.register("obs-1", 100.0, GEOMETRY, None)
    table.record_acked(record.frame, 100.3)
    table.record_sent(record.frame, 100.2)
    assert (record.send_status, record.ack_status, record.acked_unix_s) == ("sent", "acked", 100.3)
    assert table.locate(record.frame) == (record, None)


def test_a_frame_without_a_protocol_confirmation_is_unknown():
    table = FrameTable()
    record = table.register("obs-1", 100.0, GEOMETRY, None)
    table.record_sent(record.frame, 100.2)
    table.record_ack_unknown(record.frame)
    assert (record.ack_status, record.acked_unix_s) == ("unknown", None)


def test_a_frame_whose_push_failed_keeps_its_number_and_leaves_a_gap():
    """Frame 2's push fails: the next frame is still 3, so the model sees 1 and 3, and an action
    naming 2 finds nothing."""
    table = FrameTable()
    records = [table.register(f"obs-{n}", 100.0 + n, GEOMETRY, None) for n in (1, 2, 3)]
    for record in (records[0], records[2]):
        table.record_sent(record.frame, record.captured_at + 0.1)
    assert [record.frame for record in records if record.send_status == "sent"] == [1, 3]
    assert table.locate(2) == (None, FRAME_NOT_IN_TABLE)


def test_a_frame_age_counts_only_the_frames_sent_after_it():
    table = FrameTable()
    records = [table.register(f"obs-{n}", 100.0 + n, GEOMETRY, None) for n in range(1, 7)]
    # The pushes of frames 2 and 4 failed; frame 6 is prepared but not sent yet.
    for frame in (1, 3, 5):
        table.record_sent(frame, 100.0 + frame + 0.1)
    assert table.age(records[0], 106.5) == {"age_frames": 2, "age_s": 5.5}
    assert table.age(records[2], 106.5) == {"age_frames": 1, "age_s": 3.5}
    assert table.age(records[4], 106.5) == {"age_frames": 0, "age_s": 1.5}


def test_a_frame_age_counts_a_frame_sent_at_the_moment_the_action_arrived():
    table = FrameTable()
    records = [table.register(f"obs-{n}", 100.0 + n, GEOMETRY, None) for n in (1, 2)]
    table.record_sent(1, 101.1)
    table.record_sent(2, 102.5)
    assert table.age(records[0], 102.5) == {"age_frames": 1, "age_s": 1.5}


def test_a_frame_age_leaves_out_the_frames_sent_after_the_action_arrived():
    # Another thread can push a frame between the action's arrival and the age reading.
    table = FrameTable()
    records = [table.register(f"obs-{n}", 100.0 + n, GEOMETRY, None) for n in (1, 2, 3)]
    for frame in (1, 2, 3):
        table.record_sent(frame, 100.0 + frame + 0.1)
    assert table.age(records[0], 102.5) == {"age_frames": 1, "age_s": 1.5}
