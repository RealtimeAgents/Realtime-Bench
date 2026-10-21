"""The streaming shell (evaluation/streaming_shell.py) on the fake clock.

A fake environment and a fake live session (tests/shell_streaming/fakes.py) run whole episodes of
the streaming policy. Every environment operation takes a set time, so the tests read off when each
one held the gate, and that the kernel kept handling frames and replies while an action ran.

The default world: a capture takes 0.25 s, an end check 0.5 s, an action 2 s and the snapshot
0.5 s; a push takes 0.125 s. The policy checks the end and captures once a second.
"""

from __future__ import annotations

import json

from realtime_framework.clock import Clocks
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.entity.records import validate_episode_trace
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.kernel.events import Invalidated, Termination
from realtime_framework.evaluation.kernel.streaming import StreamingConfig
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.evaluation.streaming_shell import run_streaming_episode
from tests.fake_clock import FakeClock

from .fakes import Call, FakeAgent, FakeEnv, Reply

CLICK = Action(ActionType.CLICK, {"x": 1, "y": 2})
# The first reply starts on the first frame, pushed at 0.875 s; its click arrives 0.9 s later and
# executes from 1.775 s to 3.775 s, while the reply ends at 2.475 s.
ONE_CLICK = [Reply(1.6, (Call(0.9, CLICK, "c0"),))]


def _episode(duration_s, replies, *, tmp_path=None, env_setup=None):
    clock = FakeClock()
    env, agent = FakeEnv(clock), FakeAgent(clock, replies)
    if env_setup is not None:
        env_setup(env, clock)
    trace = None if tmp_path is None else EpisodeTrace(tmp_path / "trace.jsonl", run_id="s")
    logs: list[str] = []
    with clock.participant():
        termination = run_streaming_episode(
            agent,
            env,
            StreamingConfig(duration_s=duration_s),
            clocks=Clocks(clock, clock),
            log=logs.append,
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


def test_the_kernel_goes_on_with_frames_and_replies_while_an_action_executes(tmp_path):
    termination, env, agent = _episode(6.0, ONE_CLICK, tmp_path=tmp_path)
    assert env.intervals == [
        ("check_end", 0.0, 0.5),
        ("capture", 0.5, 0.75),
        ("check_end", 1.0, 1.5),
        ("capture", 1.5, 1.75),
        # The beats at 2 s and 3 s are captured alongside the action. The reads due then hold
        # the gate alone, so they wait for the action, and the read right after the action
        # stands for them and settles it. Then the read due at 4 s, and the capture due with it;
        # the beat at 5 s finds that capture out and is skipped.
        ("capture", 2.0, 2.25),
        ("capture", 3.0, 3.25),
        ("execute", 1.775, 3.775),
        ("check_end", 3.775, 4.275),
        ("check_end", 4.275, 4.775),
        ("capture", 4.775, 5.025),
        ("check_end", 5.025, 5.525),
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
    # No beat skipped during the action; the one at 5 s found a capture out.
    assert [data["reason"] for data in _kernel(records, "FrameDropped")] == ["capture_busy"]
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
    read due at 2 s waited for the action and was never sent."""
    termination, env, agent = _episode(3.0, ONE_CLICK)
    assert env.intervals[4:] == [
        ("capture", 2.0, 2.25),
        ("execute", 1.775, 3.775),
        ("snapshot", 3.775, 4.275),
    ]
    assert termination == Termination("time_limit", "deadline", 3.0, 4.275, 3.0, 4.275, None)
    ((_, outcome),) = agent.committed
    assert (outcome.status, outcome.terminal_after_action) == ("applied", True)


def test_a_snapshot_the_action_keeps_waiting_past_its_budget_fails():
    """4.8's bound: the click's budget is 20 s (sandbox/guest.py action_timeout), and this one
    runs 30 s. The snapshot gives up 20 s after the deadline, and the run is invalid."""

    def slow_action(env, clock):
        env.durations[("execute", 1)] = 30.0

    termination, env, _ = _episode(3.0, ONE_CLICK, env_setup=slow_action)
    assert ("execute", 1.775, 31.775) in env.intervals
    assert "snapshot" not in [name for name, _, _ in env.intervals]
    assert termination.stop_reason == "invalid_run"
    assert termination.snapshot_at is None
    assert termination.invalid.reason == "evaluation_snapshot_failed"
    assert "the environment was still busy 20s later" in termination.invalid.detail
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
    assert len(captures) + len(dropped) == 6  # a beat at each whole second before the deadline
    assert {data["reason"] for data in dropped} == {"capture_busy"}
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
