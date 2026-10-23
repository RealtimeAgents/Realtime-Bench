"""When run_episode reads the clocks and wakes (evaluation/shell.py), on the fake clock.

The kernel files an event at the reading of the step that takes it, so the shell must be woken by
every post instead of polling; on the token clock the kernel's watchdog runs on the operational
reading the shell passes, while the semantic clock stands still; and the shell takes the
runner's timing, STREAM_* variables included, from the config it is given.
"""

from __future__ import annotations

import threading

import pytest

from realtime_framework.entity.config import RunnerTimingConfig
from realtime_framework.evaluation import shell
from realtime_framework.evaluation.kernel.core import Kernel

from .world import CLICK, Agent, Environment, Reply, World, run


@pytest.fixture
def steps(monkeypatch):
    """The semantic reading of every step the shell makes, and whether anything was due."""
    readings = []

    class Recording(Kernel):
        def step(self, semantic, ops=None):
            out = super().step(semantic, ops)
            readings.append((semantic, ops, out is not None))
            return out

    monkeypatch.setattr(shell, "Kernel", Recording)
    return readings


def test_the_shell_steps_only_when_something_is_due(steps):
    """A 2.5 s generation in a 3 s episode at 1 fps: the result at 2.5, the periodic end checks
    at 1 and 2, the deadline at 3. The shell never steps in between: it sleeps until a post or
    the next timer."""
    episode = run("wall", 3.0, [Reply(2.5)])
    assert episode.statuses() == ["applied"]
    assert sorted({semantic for semantic, _, _ in steps}) == [0.0, 1.0, 2.0, 2.5, 3.0]
    # Each sleep ended on something due, a returned result too: the step right after it handled
    # it.
    idle = [index for index, (_, _, due) in enumerate(steps) if not due]
    assert all(steps[index + 1][2] for index in idle)


@pytest.mark.parametrize(
    ("seconds", "statuses"),
    [(2.999, ["applied"]), (3.001, [])],
    ids=["a-millisecond-before", "a-millisecond-after"],
)
def test_a_result_just_before_the_deadline_is_handled_before_it(seconds, statuses):
    """The worker's post wakes the shell at once: a result 1 ms before the 3 s deadline lands
    before the deadline fires; one 1 ms after it is given up and never reported."""
    episode = run("wall", 3.0, [Reply(seconds)])
    assert episode.statuses() == statuses
    executed = [time for call, time, _ in episode.env.calls if call == "execute"]
    if statuses:
        assert executed == [pytest.approx(2.999)]
        assert ("abort", 1) not in episode.agent.calls
    else:
        assert executed == []
        assert episode.agent.calls[0] == ("abort", 1)
    assert episode.env._termination["stop_reason"] == "time_limit"
    assert episode.env._termination["end_s"] == 3.0


def test_a_result_is_settled_though_every_end_check_takes_a_frame():
    """At 1 fps every end check takes 1 s, so one falls due at every reading the shell takes.
    Only a bound on the generation (the deadline, the cap, the watchdog) holds a result back:
    the one back at 2.5 s is taken when the check that took it in is over."""
    episode = run("wall", 5.0, [Reply(1.5), Reply(1.5)], World(check_s=1.0))
    assert episode.statuses()[0] == "applied"
    assert episode.agent.calls[0] == ("accept", 1)


class _SlowTrajectory(Agent):
    """Writing a trajectory row takes 10 ms."""

    def finalize_decision_audit(self, result: dict) -> None:
        super().finalize_decision_audit(result)
        self.clock.sleep(0.01)


class _SlowAccept(Agent):
    """Accepting a result takes 10 ms."""

    def accept(self, generation: int) -> None:
        super().accept(generation)
        self.clock.sleep(0.01)


def test_an_action_accepted_across_the_deadline_does_not_land():
    """The result comes back 5 ms before the 3 s deadline and accepting it ends 5 ms after it.
    The action lands at the reading after the accept, past the deadline, so it is not executed:
    nothing lands after the deadline (phase2.md, 时间和结束判定)."""
    episode = run("wall", 3.0, [Reply(2.995)], agent_type=_SlowAccept)
    assert episode.agent.calls[0] == ("accept", 1)
    assert [call for call, _, _ in episode.env.calls if call == "execute"] == []
    assert episode.statuses() == ["not_applied_time_limit"]
    assert episode.env._termination["stop_reason"] == "time_limit"


def test_the_check_before_landing_is_asked_once_the_result_is_accepted():
    """At 60 fps the task ends at 3 s; the result comes back at 2.995 s, in a frame read at
    179/60 s, and accepting it takes 10 ms. The check before landing is asked after the accept
    and dates the end at its asking, 3.005 s: the action is not executed."""
    world = World(ends_at=3.0, end_reason="won")
    episode = run("wall", 10.0, [Reply(2.995)], world, fps=60.0, agent_type=_SlowAccept)
    assert [call for call, _, _ in episode.env.calls if call == "execute"] == []
    assert episode.statuses() == ["not_applied_terminal"]
    assert episode.env._termination["stop_reason"] == "won"
    assert episode.env._termination["end_s"] == pytest.approx(3.005)


def test_an_end_a_check_saw_before_the_deadline_is_the_tasks_though_it_answers_after():
    """60 fps, every end check 1.05 s, the task ends at 4.5 s, T = 5. The second result is back
    at 4.2 s and accepting it takes 10 ms, so its landing is due at 4.2 s; the check asked at
    4.21 s sees the end and answers at 5.26 s. The landing comes first, past the deadline, and
    lands nothing; the end, asked before the deadline, is the task's (phase2.md, 时间和结束判定;
    decision 4)."""
    world = World(check_s=1.05, ends_at=4.5, end_reason="won")
    replies = [Reply(0.05, CLICK), Reply(30.0)]
    episode = run("wall", 5.0, replies, world, fps=60.0, input_fps=1.0, agent_type=_SlowAccept)
    assert [call for call, _, _ in episode.env.calls if call == "execute"] == []
    assert episode.statuses() == ["not_applied_terminal"]
    assert episode.env._termination["stop_reason"] == "won"
    assert episode.env._termination["end_s"] == pytest.approx(4.21)


def test_an_action_does_not_land_on_an_older_read_once_a_later_one_saw_the_end():
    """The result is back at 1.995 s and accepting it takes 10 ms; the check before landing,
    asked at 2.005 s, takes 1.2 s. The periodic check due at 2 s reads from 3.205 s and sees the
    end the task reached at 3.23 s. The older "not over" is handled first; the click waits for
    the later read and is not executed (phase2.md, 时间和结束判定)."""
    world = World(check_s=0.05, slow_checks=[(2.0, 2.1, 1.2)], ends_at=3.23, end_reason="won")
    episode = run("wall", 10.0, [Reply(1.945), Reply(30.0)], world, agent_type=_SlowAccept)
    assert [call for call, _, _ in episode.env.calls if call == "execute"] == []
    assert episode.statuses() == ["not_applied_terminal"]
    assert episode.env._termination["stop_reason"] == "won"
    assert episode.env._termination["end_s"] == pytest.approx(3.205)


def test_a_result_settled_across_the_deadline_does_not_land():
    """The result comes back 5 ms before the 3 s deadline and settling it ends 5 ms after it;
    the kernel gets it at the reading taken after the settling. The deadline came first, and
    nothing lands after it (phase2.md, 时间和结束判定): the generation is given up and the decision
    goes unreported. The call did come back in time, so its records are written, once."""
    episode = run("wall", 3.0, [Reply(2.995)], agent_type=_SlowTrajectory)
    assert [call for call, _, _ in episode.env.calls if call == "execute"] == []
    assert episode.statuses() == []
    assert "ActionProposed" not in episode.trace.events()
    assert [p["decision"] for p in episode.trace.payloads("InferenceCompleted")] == [0]
    assert len(episode.agent.trajectory) == 1
    assert episode.agent.calls == [("discard", 1)]
    assert episode.env._termination["stop_reason"] == "time_limit"
    line = "[default] k=0 generation back in 2.995s, settled past its wall_deadline — discarded"
    assert line in episode.logs
    numbers = [record["kernel_seq"] for record in episode.trace.records]
    assert numbers == sorted(numbers)


def test_a_result_settled_across_the_cap_is_the_moves_only_record():
    """The result comes back 5 ms before the 1.5 s cap and settling it ends 5 ms after it; the
    kernel gets it after the cap and reports the move skipped. Its records are the call's own,
    once, written when the cap gives it up, before the move's: none is made up for a call given
    up on."""
    replies = [Reply(1.495), Reply(0.5)]
    episode = run("wall", 3.0, replies, agent_type=_SlowTrajectory, per_move_cap_s=1.5)
    assert episode.statuses() == ["skipped_timeout", "applied"]
    completed = episode.trace.payloads("InferenceCompleted")
    assert [(p["decision"], p["timed_out"]) for p in completed] == [(0, False), (1, False)]
    events = episode.trace.events()
    assert events.index("InferenceCompleted") < events.index("ActionResolved")
    rows = episode.agent.trajectory
    assert [(row["decision"], row.get("timed_out")) for row in rows] == [(0, None), (1, None)]
    assert episode.env.lifecycle[0]["model_latency_s"] == pytest.approx(1.495)
    line = "[default] k=0 generation back in 1.495s, settled past its per_move_cap — discarded"
    assert line in episode.logs
    numbers = [record["kernel_seq"] for record in episode.trace.records]
    assert numbers == sorted(numbers)


@pytest.mark.parametrize("tokens_per_frame", [100, None], ids=["token", "paused"])
def test_a_token_result_back_before_the_watchdog_is_taken_though_settling_it_crosses_it(
    tokens_per_frame,
):
    """The result comes back 5 ms before the 7 s operational watchdog and settling it takes 10 ms.
    The world stands still while it settles, so the kernel gets it at the reading the shell took as
    it woke for it, before the settling, as the runner's bounded wait returned with it
    (legacy_loop.py _bounded_call). (A shell that woke later than the watchdog would lose it: the
    window is the wake-up lag.)"""
    replies = [Reply(6.995, CLICK, output_tokens=100), Reply(0.0, CLICK, output_tokens=100)]
    timing = RunnerTimingConfig(provider_watchdog_s=7.0)
    episode = run(
        "token",
        2.0,
        replies,
        tokens_per_frame=tokens_per_frame,
        timing=timing,
        agent_type=_SlowTrajectory,
    )
    assert episode.invalid is None
    assert episode.statuses()[0] == "applied"


def test_a_token_result_is_held_back_only_by_an_operational_bound():
    """The first decision's 1000 tokens move the world to tick 10; the next call comes back
    0.5 s in, far before its 7 s watchdog. The watchdog is operational, so the hand-back is
    timed in operational seconds, not in ticks, which are past 7 by then."""
    replies = [
        Reply(0.0, CLICK, output_tokens=1000),
        Reply(0.5, CLICK, output_tokens=100),
        Reply(0.0, CLICK, output_tokens=100),
    ]
    timing = RunnerTimingConfig(provider_watchdog_s=7.0)
    episode = run("token", 12.0, replies, tokens_per_frame=100, timing=timing)
    assert episode.invalid is None
    assert episode.statuses()[:2] == ["applied", "applied"]


class _TicksOnTheShellsThreadOnly(Environment):
    """The environment runs everything on the shell's thread (ShellEnvironment): its tick reading
    raises on a worker."""

    def tick_index(self) -> float:
        if threading.current_thread().name.startswith("generation-"):
            raise RuntimeError("tick_index on a worker")
        return super().tick_index()


@pytest.mark.parametrize("tokens_per_frame", [100, None], ids=["token", "paused"])
def test_a_worker_hands_its_result_back_without_reading_the_environment(tokens_per_frame):
    replies = [Reply(0.25, CLICK, output_tokens=100)] * 3
    episode = run(
        "token",
        2.0,
        replies,
        tokens_per_frame=tokens_per_frame,
        env_type=_TicksOnTheShellsThreadOnly,
    )
    assert episode.invalid is None
    assert episode.statuses()[0] == "applied"


def test_the_token_watchdog_is_due_on_the_operational_clock_once_the_world_has_moved():
    """The first decision's 1000 tokens move the world to tick 10; the next call never answers.
    The watchdog's 7 s are operational, so it fires 7 s into that call though the tick is past
    7 (the shell waits on the operational reading for it)."""
    replies = [Reply(0.0, CLICK, output_tokens=1000), Reply(hang=True)]
    timing = RunnerTimingConfig(provider_watchdog_s=7.0)
    episode = run("token", 30.0, replies, tokens_per_frame=100, timing=timing)
    assert episode.invalid["reason"] == "generation_watchdog_timeout"
    assert episode.env._termination["end_s"] == 10.0
    assert episode.clock.monotonic() == 7.0


def test_a_generation_the_watchdog_gave_up_on_is_recorded_as_the_runner_records_it():
    """Decision 48: the call the 7 s watchdog gave up on leaves the runner's record of it
    (legacy_loop.py _run_step): an InferenceCompleted and a trajectory row naming the watchdog, and
    a token record without usage."""
    timing = RunnerTimingConfig(provider_watchdog_s=7.0)
    episode = run("token", 5.0, [Reply(hang=True)], timing=timing)
    assert episode.invalid["reason"] == "generation_watchdog_timeout"
    completed = episode.trace.payloads("InferenceCompleted")
    assert [(p["decision"], p["invalid_reason"], p["latency_s"]) for p in completed] == [
        (0, "generation_watchdog_timeout", 7.0)
    ]
    rows = episode.agent.trajectory
    assert [(row["decision"], row["invalid_reason"]) for row in rows] == [
        (0, "generation_watchdog_timeout")
    ]
    assert episode.env._token_accounting.summary()["responses"] == 1
    assert "[default] k=0 generation TIMED OUT (>7.0s, provider_watchdog) — discarded" in (
        episode.logs
    )


def test_the_token_watchdog_runs_on_operational_seconds_from_the_environment(monkeypatch, steps):
    """STREAM_PROVIDER_WATCHDOG_S bounds a token-clock generation that never answers. The world
    stands at tick 0 throughout; the watchdog fires 7 operational seconds in, which it can only
    do because the shell passes the operational reading on the token clock."""
    monkeypatch.setenv("STREAM_PROVIDER_WATCHDOG_S", "7")
    episode = run("token", 5.0, [Reply(hang=True)], timing=RunnerTimingConfig.from_environ())
    assert episode.invalid == {
        "category": "provider",
        "reason": "generation_watchdog_timeout",
        "detail": "generation exceeded 7s",
    }
    assert episode.env._stop_reason == "invalid_run"
    assert episode.env._termination["end_s"] == 0.0
    assert episode.agent.calls == [("abort", 1)]
    fired = [
        record
        for record in episode.trace.records
        if record["event"] == "KernelRecord" and record["payload"]["kind"] == "TimerFired"
    ]
    assert [(r["semantic"], r["ops"], r["payload"]["data"]["kind"]) for r in fired] == [
        (0, 7.0, "generation_watchdog")
    ]
    assert {semantic for semantic, _, _ in steps} == {0}
    assert max(ops for _, ops, _ in steps) == 7.0


@pytest.mark.parametrize(("variable", "committed"), [(None, 3), ("2", 2)])
def test_the_frozen_stream_threshold_follows_its_environment_variable(
    monkeypatch, variable, committed
):
    """From the second frame on every capture repeats one identity: the stream is frozen once
    STREAM_OBSERVATION_FREEZE_N identical frames have come in (3 by default)."""
    if variable is not None:
        monkeypatch.setenv("STREAM_OBSERVATION_FREEZE_N", variable)
    timing = RunnerTimingConfig.from_environ()
    episode = run("wall", 10.0, [Reply(0.5)] * 5, World(frozen_from=1), timing=timing)
    assert len(episode.statuses()) == committed
    assert episode.invalid["reason"] == "observation_stream_frozen"
    assert episode.env._stream_frozen is True
    assert episode.env._stream_freeze_count == 1
    assert f"(same id/tick/hash x{timing.observation_freeze_n}); invalidating run" in "\n".join(
        episode.logs
    )


def test_the_shell_stops_once_the_kernel_finishes(steps):
    """A generation that never answers is given up at the deadline. The abort lets it return at
    once, but the kernel has finished by the time its result could be handled: the shell runs
    nothing after the finish, and the agent is never asked to discard it."""
    episode = run("wall", 2.0, [Reply(hang=True)])
    assert episode.agent.calls == [("abort", 1)]
    assert [call for call, _, _ in episode.env.calls][-2:] == ["snapshot", "release_inputs"]
    assert episode.trace.records[-1]["payload"]["kind"] == "Termination"
    assert steps[-1][2] is True  # the step that finished was the last
