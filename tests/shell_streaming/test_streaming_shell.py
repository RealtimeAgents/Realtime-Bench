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
from realtime_framework.evaluation.env_lanes import EnvLanes, Lane
from realtime_framework.evaluation.kernel.core import Kernel
from realtime_framework.evaluation.kernel.events import Invalidated, SnapshotFrozen, Termination
from realtime_framework.evaluation.kernel.streaming import StreamingConfig
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.evaluation.streaming_shell import _StreamingShell, run_streaming_episode
from tests.fake_clock import FakeClock

from .fakes import Call, FakeAgent, FakeEnv, Reply

CLICK = Action(ActionType.CLICK, {"x": 1, "y": 2})
# The first reply starts on the first frame, pushed at 0.875 s; its click arrives 0.9 s later and
# executes from 1.775 s to 3.775 s, while the reply ends at 2.475 s.
ONE_CLICK = [Reply(1.6, (Call(0.9, CLICK, "c0"),))]


def _episode(
    duration_s,
    replies,
    *,
    tmp_path=None,
    env_setup=None,
    agent_setup=None,
    log=None,
    logs=None,
    wall_origin=0.0,
):
    clock = FakeClock(wall_origin)
    env, agent = FakeEnv(clock), FakeAgent(clock, replies)
    if env_setup is not None:
        env_setup(env, clock)
    if agent_setup is not None:
        agent_setup(agent, clock)
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


def test_a_snapshot_waits_for_an_action_whose_budget_is_past_a_capture_s():
    """4.8, N32: the snapshot waits for the larger of the executing action's budget and a
    capture's. No valid action's is larger (a timed action has at most 20 s, text input as long
    as a capture, 120 s); a 150 s hold, whose budget is its duration and 5 s (sandbox/guest.py
    action_timeout), stands in for one."""
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


def test_the_environment_keeps_its_own_record_of_the_invalid_run_it_latched():
    """The kernel keeps the end of a long detail (kernel/core.py invalidate); the record main()
    reads stays the environment's, whole, as on the turn-based shell."""
    detail = "x" * 3000
    invalid = {"category": "environment", "reason": "wall_key_release_failed", "detail": detail}

    def release_fails(env, clock):
        def latch():
            env._invalid_run = invalid
            clock.start_thread(lambda: env.on_invalid(invalid), name="release-timer")

        env.during["execute"] = latch

    termination, env, _ = _episode(6.0, ONE_CLICK, env_setup=release_fails)
    assert termination.invalid.detail == detail[-2000:]
    assert env._invalid_run == invalid


class _SlowActionRows:
    """An episode trace whose ActionExecution rows take ``seconds`` to write, as they do while a
    checkpoint's sync holds the trace's lock; it keeps each row as (event, payload)."""

    def __init__(self, clock, seconds: float):
        self.clock, self.seconds = clock, seconds
        self.rows: list[tuple] = []

    def emit(self, event, payload, **fields):
        if event == "ActionExecution":
            self.clock.sleep(self.seconds)
        self.rows.append((event, payload))

    def checkpoint(self):
        pass


def test_an_action_finished_just_before_the_deadline_is_taken_in_before_it_however_slow_its_row():
    """The click runs from 1.775 s to 4.995 s and ends the task, and its ActionExecution row
    takes 10 ms to write. The lane posts the click's answer without waiting for the row, so the
    read after the click settles it, and the task's end is dated at 4.995 s rather than the
    deadline's at 5 s; the row still comes before the kernel's record of the answer."""
    clock = FakeClock()
    env = FakeEnv(clock, execute_s=3.22)
    env.ends_at = 4.99
    agent = FakeAgent(clock, [Reply(10.0, (Call(0.9, CLICK, "c0"),))])
    trace = _SlowActionRows(clock, 0.01)
    with clock.participant():
        termination = run_streaming_episode(
            agent,
            env,
            StreamingConfig(duration_s=5.0),
            clocks=Clocks(clock, clock),
            log=lambda message: None,
            trace=trace,
        )
    assert (termination.stop_reason, termination.ended_at) == ("goal", 4.995)
    order = [
        event if event != "KernelRecord" else payload["kind"]
        for event, payload in trace.rows
        if event == "ActionExecution" or payload.get("kind") == "ActionExecuted"
    ]
    assert order == ["ActionExecution", "ActionExecuted"]


def test_an_action_answer_the_shell_takes_in_late_is_filed_when_the_action_finished():
    """The click runs from 1.775 s to 4.995 s and ends the task; a wait arrives at 4.9 s, and its
    feedback holds the shell's thread until 5.1 s, across the deadline at 5 s. The click's answer,
    taken in then, is filed when the click finished, before the deadline (kernel/core.py step):
    the read after it settles it, and the task's end is dated at 4.995 s."""
    clock = FakeClock()
    env = FakeEnv(clock, execute_s=3.22)
    env.ends_at = 4.99
    agent = FakeAgent(
        clock, [Reply(10.0, (Call(0.9, CLICK, "c0"), Call(4.025, Action.wait(0.5), "w")))]
    )
    commit = agent.commit

    def slow_wait_feedback(outcome):
        if outcome.status == "wait":
            clock.sleep(0.2)
        commit(outcome)

    agent.commit = slow_wait_feedback
    with clock.participant():
        termination = run_streaming_episode(
            agent,
            env,
            StreamingConfig(duration_s=5.0),
            clocks=Clocks(clock, clock),
            log=lambda message: None,
        )
    assert (termination.stop_reason, termination.ended_at) == ("goal", 4.995)


def test_the_episode_finishes_once_the_action_outlasting_the_snapshots_wait_returned():
    """The click executes for 200 s, past the deadline at 6 s and past the snapshot's wait for
    it, 120 s, the larger of its budget and a capture's (N32): the snapshot fails and the run is
    invalid, but the episode finishes only once the click returned and its outcome was committed."""

    def endless_click(env, clock):
        env.seconds["execute"] = 200.0

    termination, _, agent = _episode(6.0, ONE_CLICK, env_setup=endless_click)
    assert termination.invalid.reason == "evaluation_snapshot_failed"
    ((committed_at, outcome),) = agent.committed
    assert committed_at == 201.775
    assert (outcome.status, outcome.terminal_after_action) == ("applied", True)


def _broken(*args, **kwargs):
    raise RuntimeError("the shell broke")


def _refuse_aborts(monkeypatch):
    submit = Lane.submit

    def refusing(self, command):
        if self.name == "abort":
            _broken()
        return submit(self, command)

    monkeypatch.setattr(Lane, "submit", refusing)


@pytest.mark.parametrize(
    "breaks",
    [
        lambda patch: patch.setattr(EnvLanes, "check_end", _broken),
        lambda patch: patch.setattr(EnvLanes, "execute", _broken),
        lambda patch: patch.setattr(_StreamingShell, "_commit", _broken),
        _refuse_aborts,
    ],
    ids=["CheckEnd", "Execute", "Commit", "AbortGeneration"],
)
def test_a_command_the_shell_fails_to_run_is_answered_and_the_episode_finishes(monkeypatch, breaks):
    """The kernel waits for the answer to each of these (kernel/core.py; GenerationAborted,
    kernel/streaming.py): one whose run raises is answered with the failure, and the run is
    invalid for the shell's exception."""
    breaks(monkeypatch)
    termination, env, _ = _episode(6.0, ONE_CLICK)
    assert (termination.invalid.category, termination.invalid.reason) == ("runner", "run_exception")
    assert "the shell broke" in env._loop_error


def test_a_stuck_reply_is_given_up_and_the_next_one_starts_once_its_abort_returned():
    """A reply that never ends is stuck after the 30 s watchdog (kernel/streaming.py); the shell
    aborts it on the abort lane, and the next reply starts once the abort returned."""
    termination, _, agent = _episode(32.0, [])
    assert agent.aborted[0] == (1, 30.875)
    assert agent.started[:2] == [(1, "obs-1", 0.875), (2, agent.started[1][1], 30.875)]
    assert 1 in agent.discarded
    assert termination.stop_reason == "time_limit"


def _slow_abort(seconds: float):
    def agent_setup(agent, clock):
        abort = agent.abort

        def slow(generation=None):
            clock.sleep(seconds)
            abort(generation)

        agent.abort = slow

    return agent_setup


def test_the_next_reply_waits_for_a_slow_abort_and_the_episode_goes_on_meanwhile():
    """The stuck reply's abort takes 3 s: the next reply starts once it returned, and meanwhile
    the state is read and frames are captured, which wait for the rebuilt session."""
    termination, env, agent = _episode(36.0, [], agent_setup=_slow_abort(3.0))
    assert agent.aborted[0] == (1, 33.875)
    assert agent.started[1][2] == 33.875
    during = [(name, start) for name, start, _ in env.intervals if 30.875 < start < 33.875]
    assert during == [
        ("check_end", 31.0),
        ("capture", 31.5),
        ("check_end", 32.0),
        ("capture", 32.5),
        ("check_end", 33.0),
        ("capture", 33.5),
    ]
    assert all(not 30.875 < at < 33.875 for _, at in agent.pushed)
    assert termination.invalid is None


def test_a_reply_whose_worker_just_ended_is_aborted_all_the_same_to_close_the_session():
    """The third push hangs until an abort closes the session. Its watchdog is due at 12.75 s,
    while the wait's feedback holds the shell from 11.725 s to 12.825 s, and the reply ends at
    12.7 s meanwhile: the shell aborts that reply all the same, which fails the push, and the
    rebuilt session takes the next frames."""

    def stuck_third_push(agent, clock):
        closed = clock.event()
        push, abort, commit = agent.push, agent.abort, agent.commit
        pushes = []

        def stuck(observation):
            pushes.append(observation)
            if len(pushes) == 3:
                clock.wait(closed, None)
                raise RuntimeError("the session was closed")
            push(observation)

        def closing(generation=None):
            abort(generation)
            closed.set()

        def slow(outcome):
            clock.sleep(1.1)
            commit(outcome)

        agent.push, agent.abort, agent.commit = stuck, closing, slow

    replies = [Reply(11.825, (Call(10.85, Action.wait(0.5), "w"),))]
    termination, _, agent = _episode(20.0, replies, agent_setup=stuck_third_push)
    assert agent.ended[0] == (1, 12.7)
    assert agent.aborted == [(1, 12.825), (2, 20.0)]
    assert agent.started[1][2] == 12.825
    assert termination.invalid is None


def test_a_slow_abort_holds_up_neither_the_read_after_an_action_nor_the_snapshot():
    """The click executes from 30.05 to 31.05 s; the reply is stuck at 30.425 s and its abort
    takes 10 s. The read after the click, once the capture at 31 s alongside it is done, and the
    reads on the grid go on meanwhile and find no end, so the task's end at 38 s comes past the
    deadline at 37.5 s: the deadline ends the episode, and the snapshot is taken right after it,
    not once the abort returned."""

    def env_setup(env, clock):
        env.seconds["check_end"] = 0.05
        env.ends_at = 38.0
        env.durations[("execute", 1)] = 1.0

    replies = [Reply(calls=(Call(29.575, CLICK, "c0"),))]
    termination, env, agent = _episode(
        37.5, replies, env_setup=env_setup, agent_setup=_slow_abort(10.0)
    )
    reads = [start for name, start, _ in env.intervals if name == "check_end" and start > 30]
    assert reads[:3] == [31.25, 32.0, 33.0]
    assert termination == Termination("time_limit", "deadline", 37.5, 38.0, 37.5, 38.0, None)
    assert env._termination["snapshot_delay_s"] == 0.5
    assert agent.aborted == [(1, 40.425)]


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


def test_a_failed_release_of_the_inputs_at_the_end_is_only_logged():
    """Decision 33, as on the turn-based shell: the inputs are released once the outcome is
    latched and the snapshot read. A release that raises is logged; the run stays valid and the
    summary has no error."""

    def release_fails(env, clock):
        release = env.release_inputs

        def release_inputs():
            release()
            raise RuntimeError("the page is gone")

        env.release_inputs = release_inputs

    logs: list[str] = []
    termination, env, _ = _episode(6.0, ONE_CLICK, env_setup=release_fails, logs=logs)
    assert env.released_at == 6.5
    assert termination == Termination("time_limit", "deadline", 6.0, 6.5, 6.0, 6.5, None)
    assert (env._stop_reason, env._invalid_run) == ("time_limit", None)
    assert getattr(env, "_loop_error", None) is None
    (logged,) = [line for line in logs if line.startswith("[cleanup]")]
    assert logged.startswith("[cleanup] failed to release stateful inputs:")
    assert "RuntimeError: the page is gone" in logged


def _gives_up_the_key(env, clock) -> None:
    """The wall clock's key-release timer gives a key up for good: it latches the run invalid
    and, on its own thread, reports it through on_invalid (sandbox/actions.py
    _wall_release_due)."""
    invalid = {"category": "environment", "reason": "wall_key_release_failed", "detail": "keyup"}
    env._invalid_run = invalid
    clock.start_thread(lambda: env.on_invalid(invalid), name="release-timer")


def _gives_up_during_the_release(env, clock):
    release = env.release_inputs

    def release_inputs():
        release()
        _gives_up_the_key(env, clock)

    env.release_inputs = release_inputs


def _gives_up_while_the_abort_runs(env, clock):
    clock.timer(6.75, lambda: _gives_up_the_key(env, clock))


def _abort_takes_a_second(agent, clock):
    abort = agent.abort

    def slow_abort(generation=None):
        abort(generation)
        clock.sleep(1.0)

    agent.abort = slow_abort


@pytest.mark.parametrize(
    ("env_setup", "agent_setup", "released_at"),
    [
        (_gives_up_while_the_abort_runs, _abort_takes_a_second, 7.0),
        (_gives_up_during_the_release, None, 6.5),
    ],
    ids=["abort", "release"],
)
def test_a_failure_the_environment_reports_once_the_snapshot_is_frozen_is_only_logged(
    env_setup, agent_setup, released_at
):
    """Decision 30, as on the turn-based shell: the snapshot is read from 6 s to 6.5 s. The
    wall clock's key-release timer gives a key up for good after it, at 6.75 s while the reply's
    abort still runs, which the episode waits for, or during the release of the inputs: the
    failure is only logged, as one of the cleanup's own is, and the termination and the run's
    record on env both say the run is valid."""
    logs: list[str] = []
    termination, env, _ = _episode(
        6.0, ONE_CLICK, env_setup=env_setup, agent_setup=agent_setup, logs=logs
    )
    assert env.released_at == released_at
    assert termination == Termination("time_limit", "deadline", 6.0, 6.5, 6.0, 6.5, None)
    assert (env._stop_reason, env._invalid_run) == ("time_limit", None)
    (logged,) = [line for line in logs if line.startswith("[cleanup]")]
    assert logged == (
        "[cleanup] environment:wall_key_release_failed once the snapshot was frozen, only "
        "logged — keyup"
    )


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
