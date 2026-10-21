"""What streaming adds to the frame table (harnesses/default/frames.py): use from several threads."""

from __future__ import annotations

import sys
import threading

from realtime_framework.harnesses.default.frames import FrameTable

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
