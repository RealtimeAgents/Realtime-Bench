"""The environment lanes and the shell's gate (evaluation/env_lanes.py) on the fake clock.

A fake environment takes a set time on the clock for each operation and records when each one ran,
so the tests see whether two ever overlapped, in which order they held the gate, and when each
answer reached the kernel's inbox.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from realtime_framework.entity.actions import Action, ActionType, Observation
from realtime_framework.entity.model import RetryCancelled
from realtime_framework.evaluation.env_lanes import EnvLanes, Executed, Lane
from realtime_framework.evaluation.kernel.events import (
    ActionExecuted,
    CheckEnd,
    EndChecked,
    Execute,
    FrameCaptured,
    Invalidated,
)
from tests.fake_clock import FakeClock

from .fakes import FakeEnv, Inbox

CLICK = Action(ActionType.CLICK, {"x": 1, "y": 2})
INVALID = {"category": "environment", "reason": "action_execution_failed", "detail": "guest"}


def _frame(observation: Observation) -> FrameCaptured:
    return FrameCaptured(observation.metadata["observation_id"], 0.0, observation.timestamp)


def _lanes():
    clock = FakeClock()
    env, inbox = FakeEnv(clock), Inbox(clock)
    return clock, env, inbox, EnvLanes(env, inbox.post, clock)


def test_each_command_is_answered_once_the_gate_lets_it_run():
    """The capture runs alongside the action; the end check asked after it holds the gate alone,
    once the action is done."""
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
    assert env.intervals == [
        ("capture", 0.5, 0.75),
        ("execute", 0.0, 2.0),
        ("check_end", 2.0, 2.5),
    ]
    assert events == [
        (0.75, FrameCaptured("obs-1", 0.0, 0.75)),
        (2.0, ActionExecuted(0, result={"executed": "click"}, started_at=0.0, finished_at=2.0)),
        (2.5, EndChecked(7, False, None)),
    ]
    assert executed == [Executed(0, {"x": 1}, 0.0, 2.0)]


@pytest.mark.parametrize(
    ("by_guest", "answer"),
    [
        (True, ActionExecuted(3, {"error": "HTTP 400"}, rejection="HTTP 400")),
        (False, ActionExecuted(3, {"error": "HTTP 400"}, error="HTTP 400")),
    ],
)
def test_a_guest_refusal_is_a_rejection_and_any_other_failure_an_error(by_guest, answer):
    """A refusal is the model's action failing and the run stays valid; any other failure is the
    environment's, and the kernel invalidates the run for it (kernel/streaming.py _executed)."""
    clock, env, inbox, lanes = _lanes()

    def failed(action, *, release_at=None):
        env._operate("execute")
        env._last_exec_error, env._last_rejected_by_guest = "HTTP 400", by_guest
        env._last_execution_result = {"error": "HTTP 400"}
        return "failed"

    env.execute = failed
    with clock.participant():
        lanes.execute(Execute(3, CLICK), lambda record: None)
        (event,) = [event for _, event in inbox.wait_for(1)]
    lanes.close()
    # The answer also says when the action held the gate and when it returned.
    assert event == replace(answer, started_at=0.0, finished_at=2.0)


def test_an_invalid_run_the_environment_latched_is_posted_once_before_the_answer():
    """The action latches the run invalid; meanwhile the key-release timer reports the same
    record through on_invalid. A capture alongside the action ended before it."""
    clock, env, inbox, lanes = _lanes()

    def latch():
        env._invalid_run = INVALID
        clock.start_thread(lambda: lanes.gate.latched(INVALID), name="release-timer")

    env.during["execute"] = latch
    with clock.participant():
        lanes.execute(Execute(0, CLICK), lambda record: None)
        clock.sleep(0.5)
        lanes.capture(_frame)
        inbox.wait_for(3)
        clock.sleep(1.0)  # nothing more comes
    lanes.close()
    assert inbox.events == [
        (0.75, FrameCaptured("obs-1", 0.0, 0.75)),
        (1.0, Invalidated("environment", "action_execution_failed", "guest")),
        (2.0, ActionExecuted(0, result={"executed": "click"}, started_at=0.0, finished_at=2.0)),
    ]


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
    """The end-check lane takes the command at once; the check waits for the action."""
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


def test_the_snapshot_goes_next_after_the_action_executing_ahead_of_an_end_check_waiting():
    """4.8: the end check asked at 0.5 s waits for the action; the snapshot asked after it goes
    first, right after the action."""
    clock, env, inbox, lanes = _lanes()
    with clock.participant():
        command = Execute(0, CLICK)
        lanes.execute(command, lambda record: None)
        clock.sleep(0.5)
        lanes.check_end(CheckEnd(1))
        turn, executing = lanes.freeze_turn()
        assert executing is command
        snapshot = lanes.gate.call(env.task.evaluation_snapshot, turn=turn)
        inbox.wait_for(2)
    lanes.close()
    assert snapshot == {"score": 1}
    assert env.intervals == [
        ("execute", 0.0, 2.0),
        ("snapshot", 2.0, 2.5),
        ("check_end", 2.5, 3.0),
    ]
    assert lanes.executing is None


def test_a_turn_that_does_not_come_in_time_leaves_the_line():
    """4.8's bound: a snapshot that gives up waiting leaves no turn behind, which would hold the
    end check after it for good."""
    clock, env, inbox, lanes = _lanes()
    with clock.participant():
        lanes.execute(Execute(0, CLICK), lambda record: None)
        turn, _ = lanes.freeze_turn()
        with pytest.raises(TimeoutError, match="still busy 0.5s later"):
            lanes.gate.call(env.task.evaluation_snapshot, turn=turn, wait=0.5)
        lanes.check_end(CheckEnd(1))
        inbox.wait_for(2)
    lanes.close()
    assert env.intervals == [("execute", 0.0, 2.0), ("check_end", 2.0, 2.5)]


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


def test_a_capture_runs_while_an_action_runs():
    """M5: a capture and an action share the gate, so new frames come in during a long action."""
    clock, env, inbox, lanes = _lanes()
    with clock.participant():
        lanes.execute(Execute(0, CLICK), lambda record: None)
        clock.sleep(0.5)
        lanes.capture(_frame)
        inbox.wait_for(2)
    lanes.close()
    assert env.intervals == [("capture", 0.5, 0.75), ("execute", 0.0, 2.0)]
    assert env.concurrent and not env.overlapped


def test_an_end_check_holds_the_gate_alone_and_waits_for_the_action():
    """4.9: an end check runs alone. Asked during the action it waits for it; a capture asked
    meanwhile goes past it and runs alongside the action, and one asked once the action is done
    waits behind it. So the check waits at most for the action and one capture."""
    clock, env, inbox, lanes = _lanes()
    with clock.participant():
        lanes.execute(Execute(0, CLICK), lambda record: None)
        clock.sleep(0.5)
        lanes.check_end(CheckEnd(1))
        clock.sleep(1.375)
        lanes.capture(_frame)  # at 1.875: runs to 2.125, past the action's end
        inbox.wait_for(2)
        lanes.capture(_frame)  # at 2.125, while the check runs
        inbox.wait_for(4)
    lanes.close()
    assert env.intervals == [
        ("execute", 0.0, 2.0),
        ("capture", 1.875, 2.125),
        ("check_end", 2.125, 2.625),
        ("capture", 2.625, 2.875),
    ]


def test_a_lane_closed_right_after_its_command_ends_once_it_answered():
    """L6: the close and the command's wake-up are one; the thread must still end."""
    clock, posted = FakeClock(), []
    lane = Lane("test", posted.append, clock)

    def command():
        clock.sleep(0.5)
        return "answer"

    with clock.participant():
        lane.submit(command)
        lane.close()
        clock.sleep(1.0)
    lane._thread.join(2.0)
    assert posted == ["answer"]
    assert not lane._thread.is_alive()


def test_an_operation_that_raises_a_base_exception_is_answered_too():
    """L6: RetryCancelled is a BaseException (entity/model.py:483); a lane that let it through
    would die with its command held, and the kernel would wait for that answer for good."""
    clock, env, inbox, lanes = _lanes()

    def cancelled(action, *, release_at=None):
        raise RetryCancelled("abandoned")

    env.execute = cancelled
    with clock.participant():
        lanes.execute(Execute(0, CLICK), lambda record: None)
        (event,) = [event for _, event in inbox.wait_for(1)]
        lanes.execute(Execute(1, CLICK), lambda record: None)  # the lane is free again
        inbox.wait_for(2)
    lanes.close()
    assert event.decision == 0 and "RetryCancelled: abandoned" in event.error


def test_a_lane_is_free_by_the_time_its_answer_is_posted():
    """The kernel may send the lane's next command as soon as it gets the answer."""
    clock, posted = FakeClock(), []
    lane = Lane(
        "test", lambda answer: (posted.append(answer), answer == 1 and lane.submit(two)), clock
    )

    def two():
        return 2

    with clock.participant():
        lane.submit(lambda: 1)
        clock.sleep(1.0)
    lane.close()
    assert posted == [1, 2]


def test_an_action_never_reports_what_the_one_before_left():
    """The turn-based shell clears the execution record before each action (shell.py _execute);
    the token clock's key path sets a result but no coordinate trace (sandbox/actions.py)."""
    clock, env, inbox, lanes = _lanes()
    executed: list[Executed] = []
    plain = env.execute

    def without_a_trace(action, *, release_at=None):
        env._operate("execute")
        return "applied"

    with clock.participant():
        lanes.execute(Execute(0, CLICK), executed.append)
        inbox.wait_for(1)
        env.execute = without_a_trace
        lanes.execute(Execute(1, CLICK), executed.append)
        events = [event for _, event in inbox.wait_for(2)]
    lanes.close()
    env.execute = plain
    assert [record.coordinate_trace for record in executed] == [{"x": 1}, None]
    assert events[1] == ActionExecuted(1, result=None, started_at=2.0, finished_at=4.0)


def test_an_action_starts_when_it_holds_the_gate():
    clock, _, inbox, lanes = _lanes()
    executed: list[Executed] = []
    with clock.participant():
        lanes.check_end(CheckEnd(1))
        clock.sleep(0.125)
        lanes.execute(Execute(0, CLICK), executed.append)
        events = [event for _, event in inbox.wait_for(2)]
    lanes.close()
    assert executed == [Executed(0, {"x": 1}, 0.5, 2.5)]
    assert (events[1].started_at, events[1].finished_at) == (0.5, 2.5)


def test_an_invalid_run_latched_before_the_lanes_began_is_not_theirs_to_post():
    """As the turn-based shell, which posts only what an operation latched (shell.py _env)."""
    clock, env, inbox, lanes = _lanes()
    env._invalid_run = INVALID
    with clock.participant():
        lanes.capture(_frame)
        inbox.wait_for(1)
    lanes.close()
    assert [type(event) for _, event in inbox.events] == [FrameCaptured]
