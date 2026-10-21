"""When run_episode reads the clocks and wakes (evaluation/shell.py), on the fake clock.

The kernel files an event at the reading of the step that takes it, so the shell must be woken by
every post instead of polling; on the token clock the kernel's watchdog runs on the operational
reading the shell passes, while the semantic clock stands still; and the shell takes the
runner's timing, STREAM_* variables included, from the config it is given.
"""

from __future__ import annotations

import pytest

from realtime_framework.entity.config import RunnerTimingConfig
from realtime_framework.evaluation import shell
from realtime_framework.evaluation.kernel.core import Kernel

from .world import Agent, Reply, World, run


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
    # Each sleep ended on something due: the step right after it handled it, or, woken by a
    # returned result, it found nothing else due and the step that took the result came next.
    idle = [index for index, (_, _, due) in enumerate(steps) if not due]
    for index in idle[:-1]:
        _, _, due = steps[index + 1]
        if not due:
            assert steps[index + 2][2] and steps[index + 2][0] == steps[index + 1][0]


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


class _SlowTrajectory(Agent):
    """Writing a trajectory row takes 10 ms."""

    def finalize_decision_audit(self, result: dict) -> None:
        super().finalize_decision_audit(result)
        self.clock.sleep(0.01)


def test_a_result_is_filed_when_it_came_back_not_when_it_was_settled():
    """The result comes back 5 ms before the 3 s deadline and settling it ends 5 ms after. It is
    handled as of its return, so its action is proposed; the deadline then comes before the
    check right before execution and the action is not applied, as the runner's check at
    landing decides (runner.py _land_action)."""
    episode = run("wall", 3.0, [Reply(2.995)], agent_type=_SlowTrajectory)
    assert episode.statuses() == ["not_applied_time_limit"]
    assert [p["decision"] for p in episode.trace.payloads("ActionProposed")] == [0]
    assert [call for call, _, _ in episode.env.calls if call == "execute"] == []


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
