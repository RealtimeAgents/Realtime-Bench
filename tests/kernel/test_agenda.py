"""The kernel's schedules order due entries by time, then priority, then filing order; the two
clocks' schedules never compare times; the inbox takes events from any thread in order."""

from __future__ import annotations

import threading

import pytest

from realtime_framework.evaluation.kernel.agenda import Agenda, Inbox
from realtime_framework.evaluation.kernel.events import Clock, Priority

SEMANTIC, OPS = Clock.SEMANTIC, Clock.OPS


def _drain(agenda: Agenda, semantic: float, ops: float | None = None) -> list:
    now = {SEMANTIC: semantic, OPS: semantic if ops is None else ops}
    items = []
    while (due := agenda.pop(now)) is not None:
        items.append(due[1])
    return items


def test_entries_run_by_time_first():
    agenda = Agenda(merged=False)
    agenda.add(SEMANTIC, 3.0, Priority.CONTROL, "late control")
    agenda.add(SEMANTIC, 1.0, Priority.START, "early start")
    agenda.add(SEMANTIC, 2.0, Priority.COMPLETION, "middle completion")
    assert _drain(agenda, 3.0) == ["early start", "middle completion", "late control"]


def test_at_one_time_control_beats_completion_beats_start():
    agenda = Agenda(merged=False)
    agenda.add(SEMANTIC, 5.0, Priority.START, "next frame")
    agenda.add(SEMANTIC, 5.0, Priority.COMPLETION, "action executed")
    agenda.add(SEMANTIC, 5.0, Priority.CONTROL, "deadline")
    assert _drain(agenda, 5.0) == ["deadline", "action executed", "next frame"]


def test_equal_time_and_priority_keep_filing_order():
    agenda = Agenda(merged=False)
    for name in ("first", "second", "third"):
        agenda.add(SEMANTIC, 1.0, Priority.COMPLETION, name)
    assert _drain(agenda, 1.0) == ["first", "second", "third"]


def test_nothing_is_due_before_its_time():
    agenda = Agenda(merged=False)
    agenda.add(SEMANTIC, 2.0, Priority.CONTROL, "deadline")
    assert agenda.pop({SEMANTIC: 1.999, OPS: 100.0}) is None
    assert _drain(agenda, 2.0) == ["deadline"]


def test_each_schedule_is_due_by_its_own_clock():
    agenda = Agenda(merged=False)
    agenda.add(OPS, 300.0, Priority.CONTROL, "watchdog")
    agenda.add(SEMANTIC, 5.0, Priority.START, "next frame")
    # Five ticks of world time do not make a 300-second watchdog due, nor the reverse.
    assert _drain(agenda, 5.0, ops=10.0) == ["next frame"]
    assert _drain(agenda, 5.0, ops=300.0) == ["watchdog"]


def test_due_entries_of_two_schedules_compare_priority_and_order_not_time():
    agenda = Agenda(merged=False)
    agenda.add(SEMANTIC, 1.0, Priority.COMPLETION, "tick-one completion")
    agenda.add(OPS, 500.0, Priority.CONTROL, "watchdog")
    agenda.add(SEMANTIC, 0.5, Priority.COMPLETION, "earlier completion")
    # 500 seconds is not "later" than tick 1: the control entry goes first.
    assert _drain(agenda, 1.0, ops=500.0) == [
        "watchdog",
        "earlier completion",
        "tick-one completion",
    ]


def test_same_priority_across_schedules_keeps_filing_order():
    agenda = Agenda(merged=False)
    agenda.add(OPS, 9.0, Priority.COMPLETION, "filed first")
    agenda.add(SEMANTIC, 1.0, Priority.COMPLETION, "filed second")
    assert _drain(agenda, 1.0, ops=9.0) == ["filed first", "filed second"]


def test_wall_clock_merges_the_schedules_and_compares_times():
    agenda = Agenda(merged=True)
    agenda.add(OPS, 3.0, Priority.CONTROL, "per-move cap")
    agenda.add(SEMANTIC, 1.0, Priority.START, "next frame")
    agenda.add(OPS, 1.0, Priority.COMPLETION, "operational entry")
    assert _drain(agenda, 3.0) == ["operational entry", "next frame", "per-move cap"]


def test_cancelled_timers_never_come_due():
    agenda = Agenda(merged=False)
    keep = agenda.add(SEMANTIC, 1.0, Priority.START, "kept", timer=True)
    drop = agenda.add(SEMANTIC, 1.0, Priority.CONTROL, "dropped", timer=True)
    agenda.cancel(drop)
    assert keep != drop
    assert _drain(agenda, 1.0) == ["kept"]


def test_cancel_timers_leaves_filed_events():
    agenda = Agenda(merged=True)
    agenda.add(SEMANTIC, 1.0, Priority.CONTROL, "deadline", timer=True)
    agenda.add(SEMANTIC, 1.0, Priority.COMPLETION, "posted event")
    agenda.cancel_timers()
    assert _drain(agenda, 1.0) == ["posted event"]


def test_cancel_ignores_events():
    agenda = Agenda(merged=True)
    handle = agenda.add(SEMANTIC, 1.0, Priority.COMPLETION, "posted event")
    agenda.cancel(handle)
    assert _drain(agenda, 1.0) == ["posted event"]


@pytest.mark.parametrize("threads", [1, 8])
def test_inbox_keeps_each_threads_order_and_loses_nothing(threads):
    inbox = Inbox()
    per_thread = 500

    def post(name):
        for index in range(per_thread):
            inbox.post((name, index))

    workers = [threading.Thread(target=post, args=(name,)) for name in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    events = inbox.take()
    assert len(events) == threads * per_thread
    for name in range(threads):
        assert [index for owner, index in events if owner == name] == list(range(per_thread))
    assert inbox.take() == []
