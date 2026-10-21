"""The environment lanes and the shell's gate (evaluation/env_lanes.py) on the fake clock.

A fake environment takes a set time on the clock for each operation and records when each one ran,
so the tests see whether two ever overlapped, in which order they held the gate, and when each
answer reached the kernel's inbox.
"""

from __future__ import annotations

import pytest

from realtime_framework.entity.actions import Action, ActionType, Observation
from realtime_framework.evaluation.env_lanes import EnvLanes, Executed
from realtime_framework.evaluation.kernel.events import (
    ActionExecuted,
    CheckEnd,
    EndChecked,
    Execute,
    FrameCaptured,
    Invalidated,
)
from tests.fake_clock import FakeClock

CLICK = Action(ActionType.CLICK, {"x": 1, "y": 2})
INVALID = {"category": "environment", "reason": "action_execution_failed", "detail": "guest"}


class FakeEnv:
    """The ShellEnvironment fields the lanes read, and operations that each take a set time."""

    def __init__(self, clock, *, capture_s=0.25, execute_s=2.0, check_s=0.5):
        self.clock = clock
        self.seconds = {"capture": capture_s, "execute": execute_s, "check_end": check_s}
        self.intervals: list[tuple[str, float, float]] = []
        self.running = 0
        self.overlapped = False
        self.captures = 0
        self.fail: set[str] = set()
        self._invalid_run = None
        self._last_exec_error = None
        self._last_rejected_by_guest = False
        self._last_execution_result = None
        self._last_coordinate_trace = None
        self.last_held_keys = ()
        self.last_end_checks = ()
        self.during: dict = {}  # name -> what an operation does halfway through

    def _operate(self, name: str) -> None:
        self.running += 1
        self.overlapped |= self.running > 1
        start = self.clock.monotonic()
        try:
            self.clock.sleep(self.seconds[name] / 2)
            self.during.pop(name, lambda: None)()
            self.clock.sleep(self.seconds[name] / 2)
            if name in self.fail:
                raise RuntimeError(f"{name} failed")
        finally:
            self.running -= 1
            self.intervals.append((name, start, self.clock.monotonic()))

    def capture(self) -> Observation:
        self._operate("capture")
        # What a later operation leaves behind: an action read after it would see this.
        self._last_exec_error, self._last_rejected_by_guest = None, False
        self._last_execution_result = {"left": "by a capture"}
        self.captures += 1
        return Observation(b"", self.clock.time(), 0, {"observation_id": f"obs-{self.captures}"})

    def execute(self, action, *, release_at=None) -> str:
        self._operate("execute")
        self._last_coordinate_trace = {"x": action.parameters["x"]}
        self._last_execution_result = {"executed": action.action_type.value}
        return "applied"

    def check_end(self) -> tuple[bool, str | None]:
        self._operate("check_end")
        return False, None


class Inbox:
    """The kernel's inbox as the lanes see it: each event posted, with the time it came."""

    def __init__(self, clock):
        self.clock = clock
        self.events: list[tuple[float, object]] = []
        self._arrived = clock.event()

    def post(self, event) -> None:
        self.events.append((self.clock.monotonic(), event))
        self._arrived.set()

    def wait_for(self, count: int) -> list[tuple[float, object]]:
        while len(self.events) < count:
            self._arrived.clear()
            if len(self.events) < count:
                assert self.clock.wait(self._arrived, 60.0), f"only {len(self.events)} events"
        return self.events


def _frame(observation: Observation) -> FrameCaptured:
    return FrameCaptured(observation.metadata["observation_id"], 0.0, observation.timestamp)


def _lanes():
    clock = FakeClock()
    env, inbox = FakeEnv(clock), Inbox(clock)
    return clock, env, inbox, EnvLanes(env, inbox.post, clock)


def test_lanes_run_one_operation_at_a_time_in_the_order_they_asked_for_the_gate():
    clock, env, inbox, lanes = _lanes()
    executed: list[Executed] = []
    with clock.participant():
        lanes.execute(Execute(0, CLICK), executed.append)
        clock.sleep(0.5)
        lanes.capture(_frame)
        clock.sleep(0.125)
        lanes.check_end(CheckEnd(7))
        events = inbox.wait_for(3)
    lanes.close()
    assert not env.overlapped
    assert env.intervals == [
        ("execute", 0.0, 2.0),
        ("capture", 2.0, 2.25),
        ("check_end", 2.25, 2.75),
    ]
    assert events == [
        (2.0, ActionExecuted(0, result={"executed": "click"})),
        (2.25, FrameCaptured("obs-1", 0.0, 2.25)),
        (2.75, EndChecked(7, False, None)),
    ]
    assert executed == [Executed(0, {"x": 1}, 0.0, 2.0)]


def test_an_action_is_read_before_another_operation_can_change_what_it_left():
    clock, env, inbox, lanes = _lanes()

    def refused(action, *, release_at=None):
        env._operate("execute")
        env._last_exec_error, env._last_rejected_by_guest = "HTTP 400", True
        env._last_execution_result = {"error": "HTTP 400"}
        return "failed"

    env.execute = refused
    with clock.participant():
        lanes.execute(Execute(3, CLICK), lambda record: None)
        clock.sleep(0.5)
        lanes.capture(_frame)  # waits for the gate, then leaves its own results behind
        events = inbox.wait_for(2)
    lanes.close()
    assert events[0] == (2.0, ActionExecuted(3, result={"error": "HTTP 400"}, rejection="HTTP 400"))
    assert env._last_execution_result == {"left": "by a capture"}


def test_an_invalid_run_the_environment_latched_is_posted_once_before_the_answer():
    """The action latches the run invalid; meanwhile the key-release timer reports the same
    record through on_invalid, and a capture after it finds it already latched."""
    clock, env, inbox, lanes = _lanes()

    def latch():
        env._invalid_run = INVALID
        clock.start_thread(lambda: lanes.gate.latched(INVALID), name="release-timer")

    env.during["execute"] = latch
    with clock.participant():
        lanes.execute(Execute(0, CLICK), lambda record: None)
        clock.sleep(0.5)
        lanes.capture(_frame)
        events = inbox.wait_for(3)
        clock.sleep(1.0)  # nothing more comes
    lanes.close()
    assert [event for _, event in inbox.events] == [
        Invalidated("environment", "action_execution_failed", "guest"),
        ActionExecuted(0, result={"executed": "click"}),
        FrameCaptured("obs-1", 0.0, 2.25),
    ]
    assert events[0][0] == 1.0


def test_an_invalid_run_latched_during_a_capture_is_not_posted_again_by_the_action():
    clock, env, inbox, lanes = _lanes()
    env.during["capture"] = lambda: setattr(env, "_invalid_run", INVALID)
    with clock.participant():
        lanes.capture(_frame)
        clock.sleep(0.125)
        lanes.execute(Execute(0, CLICK), lambda record: None)
        inbox.wait_for(3)
        clock.sleep(1.0)
    lanes.close()
    assert [type(event) for _, event in inbox.events] == [
        Invalidated,
        FrameCaptured,
        ActionExecuted,
    ]


def test_a_lane_takes_one_command_at_a_time():
    clock, _, inbox, lanes = _lanes()
    with clock.participant():
        lanes.execute(Execute(0, CLICK), lambda record: None)
        with pytest.raises(AssertionError, match="action lane is still running"):
            lanes.execute(Execute(1, CLICK), lambda record: None)
        inbox.wait_for(1)
        # Free again once it answered: the next command finds it free.
        lanes.execute(Execute(1, CLICK), lambda record: None)
        inbox.wait_for(2)
    lanes.close()
    assert [event.decision for _, event in inbox.events] == [0, 1]


def test_an_end_check_on_its_own_lane_is_taken_while_an_action_runs():
    clock, _, inbox, lanes = _lanes()
    with clock.participant():
        lanes.execute(Execute(0, CLICK), lambda record: None)
        lanes.check_end(CheckEnd(1))  # no assertion: the end-check lane is free
        events = inbox.wait_for(2)
    lanes.close()
    assert events[1] == (2.5, EndChecked(1, False, None))


def test_an_operation_that_raises_is_answered_with_its_error():
    clock, env, inbox, lanes = _lanes()
    env.fail = {"capture", "execute", "check_end"}
    executed = []
    with clock.participant():
        lanes.capture(_frame)
        clock.sleep(0.125)
        lanes.execute(Execute(4, CLICK), executed.append)
        clock.sleep(0.125)
        lanes.check_end(CheckEnd(9))
        events = [event for _, event in inbox.wait_for(3)]
    lanes.close()
    assert [type(event) for event in events] == [FrameCaptured, ActionExecuted, EndChecked]
    assert all("failed" in event.error for event in events)
    assert (events[1].decision, events[2].request, executed) == (4, 9, [])


def test_the_action_lane_says_when_the_action_it_runs_is_done():
    clock, _, inbox, lanes = _lanes()
    done = []
    with clock.participant():
        assert lanes.action_done(0.0) and lanes.executing is None
        command = Execute(0, CLICK)
        lanes.execute(command, lambda record: None)
        assert lanes.executing is command
        done.append((lanes.action_done(0.5), clock.monotonic()))
        done.append((lanes.action_done(5.0), clock.monotonic()))
        inbox.wait_for(1)
    lanes.close()
    assert done == [(False, 0.5), (True, 2.0)]
    assert lanes.executing is None


@pytest.mark.parametrize("check_first", [True, False])
def test_commands_sent_at_one_instant_take_the_gate_in_the_order_they_were_sent(check_first):
    clock, env, inbox, lanes = _lanes()
    sends = [lambda: lanes.check_end(CheckEnd(1)), lambda: lanes.capture(_frame)]
    with clock.participant():
        for send in sends if check_first else sends[::-1]:
            send()
        inbox.wait_for(2)
    lanes.close()
    order = [name for name, _, _ in env.intervals]
    assert order == (["check_end", "capture"] if check_first else ["capture", "check_end"])
