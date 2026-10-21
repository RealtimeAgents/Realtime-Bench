"""The shell's gate (evaluation/env_lanes.py EnvGate), turn by turn.

Each test takes places in the gate's line and runs, on one thread and in the order it says, the
operations whose turn has come, so that it sees whom the gate goes to next without timing anything.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from realtime_framework.evaluation.env_lanes import ACTION, ALONE, CAPTURE, EnvGate


class _Clock:
    """Waits that end at once: a turn that has come is held; for one that has not, ``meanwhile``
    runs, what the holder does during the wait, and then the wait runs out."""

    def __init__(self) -> None:
        self.meanwhile = None

    def event(self) -> threading.Event:
        return threading.Event()

    def wait(self, turn: threading.Event, timeout) -> bool:
        if turn.is_set():
            return True
        meanwhile, self.meanwhile = self.meanwhile, None
        if meanwhile is not None:
            meanwhile()
        return False


def _gate() -> tuple[_Clock, EnvGate]:
    clock = _Clock()
    return clock, EnvGate(SimpleNamespace(), lambda event: None, clock)


def test_a_turn_that_comes_as_its_wait_runs_out_is_held_and_handed_on():
    """The capture holds the gate; the end check's wait runs out just as the capture lets the gate
    go to it. The end check runs, and leaves the gate free."""
    clock, gate = _gate()
    capture = gate.reserve(CAPTURE)
    clock.meanwhile = lambda: gate.call(lambda: None, turn=capture)
    assert gate.call(lambda: "read", wait=1.0) == "read"
    assert gate.reserve(ALONE).is_set()


def test_an_action_asked_after_an_end_check_waiting_goes_after_it():
    """The capture holds the gate and the end check waits for it. The action asked next could share
    the gate with the capture, but goes after the end check, which asked first."""
    _, gate = _gate()
    capture = gate.reserve(CAPTURE)
    check = gate.reserve(ALONE)
    action = gate.reserve(ACTION)
    assert (capture.is_set(), check.is_set(), action.is_set()) == (True, False, False)
    gate.call(lambda: None, turn=capture)
    assert (check.is_set(), action.is_set()) == (True, False)
    gate.call(lambda: None, turn=check)
    assert action.is_set()


def test_the_snapshot_goes_right_after_an_action_still_waiting_and_ahead_of_the_rest():
    """An end check holds the gate; an action and another end check wait. The snapshot, asked to
    go after that action and ahead of the rest, takes the place right behind the action."""
    _, gate = _gate()
    check = gate.reserve(ALONE)
    action = gate.reserve(ACTION)
    later = gate.reserve(ALONE)
    snapshot = gate.reserve(ALONE, after=action, ahead=True)
    gate.call(lambda: None, turn=check)
    assert (action.is_set(), snapshot.is_set(), later.is_set()) == (True, False, False)
    gate.call(lambda: None, turn=action)
    assert (snapshot.is_set(), later.is_set()) == (True, False)
    gate.call(lambda: None, turn=snapshot)
    assert later.is_set()
