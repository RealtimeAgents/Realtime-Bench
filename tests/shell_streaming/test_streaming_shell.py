"""The streaming shell (evaluation/streaming_shell.py) on the fake clock.

A fake environment and a fake live session (tests/shell_streaming/fakes.py) run whole episodes of
the streaming policy. Every environment operation takes a set time, so the tests read off when each
one held the gate, and that the kernel kept handling frames and replies while an action ran.

The default world: a capture takes 0.25 s, an end check 0.5 s, an action 2 s and the snapshot
0.5 s; a push takes 0.125 s. The policy checks the end and captures once a second.
"""

from __future__ import annotations

import json
import time

import pytest

from realtime_framework.clock import Clocks
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.entity.model import RetryCancelled
from realtime_framework.entity.records import validate_episode_trace
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.env_lanes import EnvLanes
from realtime_framework.evaluation.kernel.core import Kernel
from realtime_framework.evaluation.kernel.events import Invalidated, SnapshotFrozen, Termination
from realtime_framework.evaluation.kernel.streaming import StreamingConfig
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.evaluation.streaming_shell import run_streaming_episode
from tests.fake_clock import FakeClock

from .fakes import Call, FakeAgent, FakeEnv, Reply

CLICK = Action(ActionType.CLICK, {"x": 1, "y": 2})
# The first reply starts on the first frame, pushed at 0.875 s; its click arrives 0.9 s later and
# executes from 1.775 s to 3.775 s, while the reply ends at 2.475 s.
ONE_CLICK = [Reply(1.6, (Call(0.9, CLICK, "c0"),))]


def _episode(
    duration_s, replies, *, tmp_path=None, env_setup=None, log=None, logs=None, wall_origin=0.0
):
    clock = FakeClock(wall_origin)
    env, agent = FakeEnv(clock), FakeAgent(clock, replies)
    if env_setup is not None:
        env_setup(env, clock)
    trace = (
        None
        if tmp_path is None
        else EpisodeTrace(tmp_path / "trace.jsonl", run_id="s", clock=clock)
    )
    logs = [] if logs is None else logs
    with clock.participant():
        termination = run_streaming_episode(
            agent,
            env,
            StreamingConfig(duration_s=duration_s),
            clocks=Clocks(clock, clock),
            log=log or logs.append,
            trace=trace,
        )
    if trace is not None:
        trace.close()
    assert not env.overlapped, "an operation that holds the gate alone ran with another"
    return termination, env, agent


def _records(tmp_path) -> list[dict]:
    path = tmp_path / "trace.jsonl"
    validate_episode_trace(path)
    return [json.loads(line) for line in path.read_text().splitlines()]


def _kernel(records, kind: str) -> list[dict]:
    return [
        r["payload"]["data"]
        for r in records
        if r["event"] == "KernelRecord" and r["payload"]["kind"] == kind
    ]


def test_the_agent_is_told_the_origin_its_arrival_times_count_from():
    """The episode's start on the wall clock, which is not the clock's zero."""

    def start_at_the_clock(env, clock):
        env.t0 = clock.time()

    _, env, agent = _episode(1.0, [], wall_origin=1000.0, env_setup=start_at_the_clock)
    assert agent.origin == env.t0 == 1000.0


def test_the_kernel_goes_on_with_frames_and_replies_while_an_action_executes(tmp_path):
    termination, env, agent = _episode(6.0, ONE_CLICK, tmp_path=tmp_path)
    assert env.intervals == [
        ("check_end", 0.0, 0.5),
        ("capture", 0.5, 0.75),
        ("check_end", 1.0, 1.5),
        ("capture", 1.5, 1.75),
        # The beats at 2 s and 3 s are captured alongside the action. The reads due then hold
        # the gate alone, so they wait for the action, and the read right after the action
        # stands for them and settles it. Then the capture due at 4 s and the read due with it
        # take turns, as do those at 5 s.
        ("capture", 2.0, 2.25),
        ("capture", 3.0, 3.25),
        ("execute", 1.775, 3.775),
        ("check_end", 3.775, 4.275),
        ("capture", 4.275, 4.525),
        ("check_end", 4.525, 5.025),
        ("capture", 5.025, 5.275),
        ("check_end", 5.275, 5.775),
        ("snapshot", 6.0, 6.5),
    ]
    # During the action, frames went out and the next reply started on one captured meanwhile.
    assert agent.pushed[:4] == [
        ("obs-1", 0.875),
        ("obs-2", 1.875),
        ("obs-3", 2.375),
        ("obs-4", 3.375),
    ]
    assert agent.started == [(1, "obs-1", 0.875), (2, "obs-3", 2.475)]
    assert (agent.accepted, agent.aborted, agent.discarded) == ([1], [(2, 6.0)], [2])
    ((committed_at, outcome),) = agent.committed
    assert (committed_at, outcome.status, outcome.call_id) == (4.275, "applied", "c0")
    assert outcome.terminal_after_action is False
    assert (outcome.arrived_at, outcome.frame_age_s) == (1.775, 0.9)
    assert termination == Termination("time_limit", "deadline", 6.0, 6.5, 6.0, 6.5, None)
    assert env._termination == {
        "stop_reason": "time_limit",
        "end_s": 6.0,
        "snapshot_s": 6.5,
        "snapshot_delay_s": 0.5,
    }
    assert (env._stop_reason, env._termination_snapshot, env._active_end_unix_s) == (
        "time_limit",
        {"score": 1},
        6.0,
    )
    assert env.released_at == 6.5
    records = _records(tmp_path)
    assert _kernel(records, "FrameDropped") == []  # no beat skipped
    (execution,) = [r["payload"] for r in records if r["event"] == "ActionExecution"]
    assert execution == {
        "decision": 0,
        "started_unix_s": 1.775,
        "finished_unix_s": 3.775,
        "coordinate_trace": {"x": 1},
    }
    assert records[-1]["payload"]["kind"] == "Termination"


def test_the_snapshot_at_the_deadline_waits_for_the_action_executing():
    """4.8: the deadline falls at 3 s, during the action. The snapshot goes right after it; the
    read due at 2 s waited for the action and was never sent. The presentation ends when the
    snapshot begins, after the action, as on the turn-based shell."""
    termination, env, agent = _episode(3.0, ONE_CLICK)
    assert env.intervals[4:] == [
        ("capture", 2.0, 2.25),
        ("execute", 1.775, 3.775),
        ("snapshot", 3.775, 4.275),
    ]
    assert termination == Termination("time_limit", "deadline", 3.0, 4.275, 3.0, 4.275, None)
    assert env._active_end_unix_s == 3.775
    ((_, outcome),) = agent.committed
    assert (outcome.status, outcome.terminal_after_action) == ("applied", True)


def test_a_snapshot_waits_for_a_slow_capture_alongside_the_action():
    """N32: the capture at 2 s runs alongside the click and takes 30 s, past the click's budget of
    20 s (sandbox/guest.py action_timeout) but within its own of 120 s. The snapshot, which holds
    the gate alone, waits for both: its bound is the larger budget, and the run is valid."""

    def slow_capture(env, clock):
        env.durations[("capture", 3)] = 30.0

    termination, env, _ = _episode(3.0, ONE_CLICK, env_setup=slow_capture)
    assert env.intervals[4:] == [
        ("execute", 1.775, 3.775),
        ("capture", 2.0, 32.0),
        ("snapshot", 32.0, 32.5),
    ]
    assert termination == Termination("time_limit", "deadline", 3.0, 32.5, 3.0, 32.5, None)


def test_a_snapshot_waits_for_a_long_hold_within_the_holds_own_budget():
    """4.8: a hold's budget is its duration and 5 s (sandbox/guest.py action_timeout), past a
    capture's; the snapshot waits for the 150 s hold executing at the deadline."""
    hold = Action(ActionType.HOLD_KEY, {"key": "space", "duration": 150.0})

    def long_hold(env, clock):
        env.durations[("execute", 1)] = 150.0

    termination, env, _ = _episode(3.0, [Reply(1.6, (Call(0.9, hold, "h0"),))], env_setup=long_hold)
    assert env.intervals[-2:] == [("execute", 1.775, 151.775), ("snapshot", 151.775, 152.275)]
    assert termination == Termination("time_limit", "deadline", 3.0, 152.275, 3.0, 152.275, None)


def test_the_snapshot_reads_in_its_own_budget_after_waiting_for_the_action():
    """The read's budget is 16 s (ENVIRONMENT_BUDGET.snapshot_read_s); the wait for the 19 s click
    before it is not part of it."""

    def slow_click(env, clock):
        env.durations[("execute", 1)] = 19.0

    termination, env, _ = _episode(3.0, ONE_CLICK, env_setup=slow_click)
    assert env.intervals[-1] == ("snapshot", 20.775, 21.275)
    assert termination == Termination("time_limit", "deadline", 3.0, 21.275, 3.0, 21.275, None)


def test_a_command_the_shell_fails_to_hand_out_is_a_run_exception(monkeypatch):
    """The capture lane raises when the shell hands it the capture; the kernel gets the
    run_exception before the failed capture's answer, so the run is invalid for the shell."""

    def capture(self, captured):
        raise RuntimeError("the capture lane broke")

    monkeypatch.setattr(EnvLanes, "capture", capture)
    termination, env, _ = _episode(3.0, [])
    assert (termination.invalid.category, termination.invalid.reason) == ("runner", "run_exception")
    assert "the capture lane broke" in env._loop_error


def test_a_snapshot_the_action_keeps_waiting_past_its_budget_fails():
    """4.8's bound, N32: the larger of the click's budget, 20 s, and a capture's, 120 s; this click
    runs 130 s. The snapshot gives up 120 s after the deadline, and the run is invalid."""

    def slow_action(env, clock):
        env.durations[("execute", 1)] = 130.0

    termination, env, _ = _episode(3.0, ONE_CLICK, env_setup=slow_action)
    assert ("execute", 1.775, 131.775) in env.intervals
    assert "snapshot" not in [name for name, _, _ in env.intervals]
    assert termination.stop_reason == "invalid_run"
    assert termination.snapshot_at is None
    assert termination.invalid.reason == "evaluation_snapshot_failed"
    assert "the environment was still busy 120s later" in termination.invalid.detail
    assert env._termination_snapshot_captured is False
    assert env._invalid_run["reason"] == "evaluation_snapshot_failed"


def test_an_end_check_due_during_an_action_reads_once_the_action_is_done():
    """4.9: the end check due at 2 s holds the gate alone, so it waits for the action, and the read
    right after the action stands for it. The task has ended at 2.5 s; the episode ends at the time
    that read saw it, and the snapshot goes next."""

    def ends(env, clock):
        env.ends_at = 2.5

    termination, env, _ = _episode(10.0, ONE_CLICK, env_setup=ends)
    assert (termination.stop_reason, termination.cause, termination.ended_at) == (
        "goal",
        "environment",
        3.775,
    )
    assert env.intervals[-2:] == [("check_end", 3.775, 4.275), ("snapshot", 4.275, 4.775)]


def test_a_capture_that_cannot_keep_up_skips_the_beats_it_misses(tmp_path):
    """4.10: a capture takes 1.5 s, so the beat after each capture comes while it is under way
    and is skipped and counted; the capture lane is never given a second capture."""

    def slow_capture(env, clock):
        env.seconds["capture"] = 1.5

    termination, env, _ = _episode(6.0, [], tmp_path=tmp_path, env_setup=slow_capture)
    assert termination.invalid is None and termination.stop_reason == "time_limit"
    records = _records(tmp_path)
    dropped = _kernel(records, "FrameDropped")
    captures = [interval for interval in env.intervals if interval[0] == "capture"]
    skipped = [data for data in dropped if data["reason"] == "capture_busy"]
    assert len(captures) + len(skipped) == 6  # a beat at each whole second before the deadline
    # The frame the last capture brings back at the deadline goes nowhere.
    assert [(data["observation_id"], data["reason"]) for data in dropped[len(skipped) :]] == [
        ("obs-3", "ended")
    ]
    assert [data["dropped"] for data in dropped] == list(range(1, len(dropped) + 1))


def test_an_invalid_run_the_release_timer_reports_is_posted_once(tmp_path):
    """The wall clock's key-release timer latches the run invalid on its own thread while the
    action executes and reports it through on_invalid; the gate's check after the action finds
    the same record."""
    invalid = {"category": "environment", "reason": "wall_key_release_failed", "detail": "keyup"}

    def release_fails(env, clock):
        def latch():
            env._invalid_run = invalid
            clock.start_thread(lambda: env.on_invalid(invalid), name="release-timer")

        env.during["execute"] = latch

    termination, _, _ = _episode(6.0, ONE_CLICK, tmp_path=tmp_path, env_setup=release_fails)
    assert termination.invalid == Invalidated("environment", "wall_key_release_failed", "keyup")
    records = _records(tmp_path)
    assert [r["payload"] for r in records if r["event"] == "RunInvalidated"] == [invalid]


def test_a_stuck_reply_is_given_up_on_the_shells_thread_and_the_next_one_starts():
    """A reply that never ends is stuck after the 30 s watchdog (kernel/streaming.py); the shell
    aborts it, and the next reply starts in the same step."""
    termination, _, agent = _episode(32.0, [])
    assert agent.aborted[0] == (1, 30.875)
    assert agent.started[:2] == [(1, "obs-1", 0.875), (2, agent.started[1][1], 30.875)]
    assert 1 in agent.discarded
    assert termination.stop_reason == "time_limit"


def test_the_shells_own_failure_on_a_lane_invalidates_the_run_as_a_run_exception(monkeypatch):
    """L4: the shell's frame handling raises on the capture lane. As on the turn-based shell, the
    run is invalid for the shell's exception and its traceback is the loop error."""

    def fingerprint(observation):
        raise RuntimeError("fingerprint failed")

    monkeypatch.setattr(episode_records, "observation_fingerprint", fingerprint)
    termination, env, _ = _episode(6.0, ONE_CLICK)
    assert (termination.invalid.category, termination.invalid.reason) == ("runner", "run_exception")
    assert "fingerprint failed" in env._loop_error


def _step_breaks(monkeypatch, when) -> None:
    step = Kernel.step

    def broken(self, semantic, ops=None):
        if when(self, semantic):
            raise RuntimeError("step broke")
        return step(self, semantic, ops)

    monkeypatch.setattr(Kernel, "step", broken)


def _no_snapshot_thread(env, clock) -> None:
    start_thread = clock.start_thread

    def refusing(target, *, name=None):
        if name == "snapshot":
            raise RuntimeError("no thread")
        return start_thread(target, name=name)

    clock.start_thread = refusing


@pytest.mark.parametrize("kernel_breaks", [False, True], ids=["kernel-holds", "kernel-raises"])
def test_a_snapshot_whose_thread_did_not_start_is_answered_as_failed(monkeypatch, kernel_breaks):
    """The deadline at 3 s asks for the snapshot, and its thread cannot start: the kernel is
    answered that the snapshot failed, and the run is invalid. Should the kernel raise then,
    the wrap-up does not wait for that snapshot."""
    if kernel_breaks:
        _step_breaks(monkeypatch, lambda kernel, semantic: kernel.ending is not None)
    termination, env, _ = _episode(3.0, ONE_CLICK, env_setup=_no_snapshot_thread)
    assert (termination.stop_reason, termination.ended_at, termination.snapshot_at) == (
        "invalid_run",
        3.0,
        None,
    )
    assert termination.invalid.reason == "run_exception"
    assert "RuntimeError: no thread" in env._loop_error


def test_a_snapshot_whose_thread_raised_is_answered_as_failed(monkeypatch):
    """The snapshot's thread raises as it takes its place at the gate: the kernel is answered
    that the snapshot failed, as for a command that raised, and the run is invalid."""

    def freeze_turn(lanes):
        raise RuntimeError("no turn")

    monkeypatch.setattr(EnvLanes, "freeze_turn", freeze_turn)
    termination, env, _ = _episode(3.0, ONE_CLICK)
    assert (termination.stop_reason, termination.snapshot_at) == ("invalid_run", None)
    assert termination.invalid.reason == "run_exception"
    assert "RuntimeError: no turn" in env._loop_error


@pytest.mark.parametrize("first", ["snapshot", "kernel"])
def test_the_first_of_a_failed_snapshot_and_a_kernel_that_raises_is_the_error(monkeypatch, first):
    """The snapshot's thread raises at the gate, and the kernel raises too. Each runs on a thread
    of its own, at the same instant unless one is made to come later: here the kernel raises at
    the step after the snapshot's failed answer was posted, or the snapshot's thread raises 0.1 s
    after the kernel did. The first is the summary's error and the run's, as the runner kept the
    exception that escaped its loop (legacy_loop.py _run_stream_legacy); the wrap-up waits for the
    snapshot's thread either way."""
    clocks = []

    def freeze_turn(lanes):
        if first == "kernel":
            clocks[0].sleep(0.1)
        raise RuntimeError("no turn")

    monkeypatch.setattr(EnvLanes, "freeze_turn", freeze_turn)
    answered = []
    post = Kernel.post

    def recording(kernel, event):
        if isinstance(event, SnapshotFrozen):
            answered.append(event)
        post(kernel, event)

    monkeypatch.setattr(Kernel, "post", recording)
    if first == "snapshot":
        _step_breaks(monkeypatch, lambda kernel, semantic: bool(answered))
    else:
        _step_breaks(monkeypatch, lambda kernel, semantic: kernel.ending is not None)
    termination, env, _ = _episode(
        3.0, ONE_CLICK, env_setup=lambda env, clock: clocks.append(clock)
    )
    assert (termination.stop_reason, termination.snapshot_at) == ("invalid_run", None)
    error = "RuntimeError: no turn" if first == "snapshot" else "RuntimeError: step broke"
    assert error in env._loop_error
    assert error in termination.invalid.detail and error in env._invalid_run["detail"]


def _refuses_the_record(monkeypatch) -> None:
    setattr_ = FakeEnv.__setattr__

    def refusing(env, name, value):
        if name == "_termination":
            raise RuntimeError("read-only")
        setattr_(env, name, value)

    monkeypatch.setattr(FakeEnv, "__setattr__", refusing)


@pytest.mark.parametrize("kernel_breaks", [False, True], ids=["kernel-holds", "kernel-raises"])
def test_an_episode_whose_termination_record_raises_still_ends(monkeypatch, kernel_breaks):
    """The environment refuses the termination record. The episode ends all the same and
    returns how it ended, its verdict latched first: main() reads it whatever else was written.
    A wrap-up that fails so is logged."""
    _refuses_the_record(monkeypatch)
    if kernel_breaks:
        _step_breaks(monkeypatch, lambda kernel, semantic: semantic >= 2.0)
    logs: list[str] = []
    termination, env, _ = _episode(3.0, ONE_CLICK, logs=logs)
    assert termination.stop_reason == ("invalid_run" if kernel_breaks else "time_limit")
    assert env._stop_reason == termination.stop_reason
    if kernel_breaks:
        assert env._invalid_run["reason"] == "run_exception"
        assert any(line.startswith("[streaming] wrapping up the episode failed") for line in logs)


def _log_raising_on(prefix: str):
    """A log whose stdout is gone for the lines that start with ``prefix``."""

    def log(message: str) -> None:
        if message.startswith(prefix):
            raise BrokenPipeError("stdout gone")

    return log


def test_a_log_that_raises_on_a_lane_loses_the_line_only(monkeypatch):
    """L3(a): the shell's frame handling raises on the capture lane, and so does the log of it.
    The run is still invalid for the shell's exception, and the capture is still answered."""

    def fingerprint(observation):
        raise RuntimeError("fingerprint failed")

    monkeypatch.setattr(episode_records, "observation_fingerprint", fingerprint)
    termination, env, _ = _episode(6.0, ONE_CLICK, log=_log_raising_on("[streaming] shell raised"))
    assert (termination.invalid.category, termination.invalid.reason) == ("runner", "run_exception")
    assert "fingerprint failed" in env._loop_error


def test_a_log_that_raises_on_a_reply_worker_loses_the_line_only():
    """A reply that fails, logged on its worker; the log raises, and the reply still finishes."""

    def generate(generation, decision, observation, budget):
        raise RuntimeError("the session broke")

    clock = FakeClock()
    env, agent = FakeEnv(clock), FakeAgent(clock, [])
    agent.generate = generate
    with clock.participant():
        termination = run_streaming_episode(
            agent,
            env,
            StreamingConfig(duration_s=6.0),
            clocks=Clocks(clock, clock),
            log=_log_raising_on("[streaming] reply"),
        )
    assert termination.stop_reason == "invalid_run"
    assert "the session broke" in termination.invalid.detail


def test_a_snapshot_read_that_raises_any_exception_fails_the_snapshot():
    """L3(b): a cancellation is a BaseException (entity/model.py RetryCancelled); the snapshot
    thread must answer all the same, or the kernel would wait for it."""

    def cancelled(env, clock):
        def snapshot():
            raise RetryCancelled("abandoned inside the snapshot read")

        env.task.evaluation_snapshot = snapshot

    termination, env, _ = _episode(3.0, ONE_CLICK, env_setup=cancelled)
    assert (termination.stop_reason, termination.snapshot_at) == ("invalid_run", None)
    assert termination.invalid.reason == "evaluation_snapshot_failed"
    assert "RetryCancelled" in env._termination_snapshot_error


def test_a_kernel_that_raises_leaves_the_episode_wrapped_up_by_the_shell(monkeypatch, tmp_path):
    """N11, as on the turn-based shell: the kernel raises at its step at 2 s, while the click
    executes and the reply runs. Nothing can be asked of the kernel any more: the shell gives
    the reply up, takes the snapshot right after the action (4.8), releases the inputs and
    writes the termination itself. The run is invalid for run_exception."""
    step = Kernel.step

    def broken(self, semantic, ops=None):
        if semantic >= 2.0:
            raise RuntimeError("step broke")
        return step(self, semantic, ops)

    monkeypatch.setattr(Kernel, "step", broken)
    logs: list[str] = []
    termination, env, agent = _episode(6.0, ONE_CLICK, tmp_path=tmp_path, logs=logs)
    assert (termination.stop_reason, termination.ended_at, termination.snapshot_at) == (
        "invalid_run",
        2.0,
        4.275,
    )
    # Recorded as the turn-based shell records it, readings and all.
    (written,) = [r for r in _records(tmp_path) if r["event"] == "RunInvalidated"]
    assert written["payload"]["reason"] == "run_exception"
    assert (written["kernel_seq"], written["semantic"], written["ops"]) == (None, 2.0, 2.0)
    assert any(line.startswith("[invalid-run] runner:run_exception") for line in logs)
    assert (termination.invalid.category, termination.invalid.reason) == ("runner", "run_exception")
    assert env.intervals[-1] == ("snapshot", 3.775, 4.275)
    assert env.released_at == 4.275
    assert env._termination == {
        "stop_reason": "invalid_run",
        "end_s": 2.0,
        "snapshot_s": 4.275,
        "snapshot_delay_s": 2.275,
    }
    assert env._invalid_run["reason"] == "run_exception"
    assert "RuntimeError: step broke" in env._loop_error
    assert agent.aborted == [(1, 2.0)]


def test_a_kernel_that_raises_after_the_end_keeps_its_end_and_its_snapshot(monkeypatch):
    """The deadline at 3 s ends the episode and the snapshot waits for the click executing; the
    kernel raises at the next step. The shell waits for that snapshot instead of taking another,
    and the end stays the deadline, the first end signal (decision 32). The snapshot's thread
    takes a moment of real time after the read, before it writes its reading: the wait, not
    the gate the inputs are released at, is what the wrap-up has it by."""
    setattr_ = FakeEnv.__setattr__

    def slow_after_the_read(env, name, value):
        if name == "_termination_snapshot_error":
            time.sleep(0.3)
        setattr_(env, name, value)

    monkeypatch.setattr(FakeEnv, "__setattr__", slow_after_the_read)
    step = Kernel.step

    def broken(self, semantic, ops=None):
        if self.ending is not None:
            raise RuntimeError("step broke")
        return step(self, semantic, ops)

    monkeypatch.setattr(Kernel, "step", broken)
    termination, env, _ = _episode(3.0, ONE_CLICK)
    assert (termination.stop_reason, termination.ended_at, termination.snapshot_at) == (
        "invalid_run",
        3.0,
        4.275,
    )
    assert [interval for interval in env.intervals if interval[0] == "snapshot"] == [
        ("snapshot", 3.775, 4.275)
    ]
    assert termination.invalid.reason == "run_exception"


def test_a_push_that_raises_any_exception_loses_that_frame_only():
    """A cancellation is a BaseException (entity/model.py RetryCancelled). A push lane that let
    it through would die with the push held: the next frame would find it busy."""
    clock = FakeClock()
    env, agent = FakeEnv(clock), FakeAgent(clock, ONE_CLICK)
    push, pushes = agent.push, []

    def first_push_cancelled(observation):
        pushes.append(observation.metadata["observation_id"])
        if len(pushes) == 1:
            raise RetryCancelled("abandoned")
        push(observation)

    agent.push = first_push_cancelled
    with clock.participant():
        termination = run_streaming_episode(
            agent, env, StreamingConfig(duration_s=3.0), clocks=Clocks(clock, clock), log=print
        )
    assert termination.invalid is None
    assert agent.started[0][1] == "obs-2"  # the first reply waits for the first frame that went out


def test_a_frozen_stream_is_left_on_the_environment_as_the_turn_based_shell_leaves_it():
    """Each capture brings the same frame back: the third makes the stream frozen and the run
    invalid, which main() reads off the environment for the run summary
    (observation_stream_frozen, observation_stream_freeze_count), as on the turn-based shell."""
    from realtime_framework.entity.actions import Observation

    def same_frame(env, clock):
        def capture():
            env._operate("capture")
            env.captures += 1
            return Observation(b"", clock.time(), 0, {"observation_id": "obs-1"})

        env.capture = capture

    termination, env, _ = _episode(6.0, [], env_setup=same_frame)
    assert termination.invalid.reason == "observation_stream_frozen"
    assert (env._stream_frozen, env._stream_freeze_count) == (True, 1)
