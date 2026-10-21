"""The turn-based policy on the wall clock, run in the discrete-event simulation.

The first end signal (the deadline, an end the environment reports, an invalidation) stops the
episode at once: the snapshot is frozen, the generation in flight is given up and its late result
never lands. The environment is asked whether it ended on the environment frame grid, during
generations and waits too, besides the loop top and right before and after execution. Waits, key
releases, the next frame and the legacy per-move cap are timers.

Every expected value is worked out by hand: from today's runner where the kernel keeps its rules
(evaluation/runner.py, sandbox/actions.py) and from phase2.md (事件内核; 时间和结束判定) where it
changes them. Times are seconds since the episode origin; env_fps = input_fps = 1 unless a test
says otherwise.
"""

from __future__ import annotations

import pytest

from realtime_framework.entity.actions import Action
from realtime_framework.evaluation.kernel.events import (
    AcceptResult,
    Capture,
    CheckEnd,
    Execute,
    Finish,
    Invalidated,
    Outcome,
    Record,
    ReleaseInputs,
    SetTimer,
    StartGeneration,
)
from realtime_framework.evaluation.kernel.turns import TurnConfig
from tests.kernel.simulation import Reply, Simulation, World

CLICK = Action.click(10, 20)


def _wall(duration_s: float, **fields) -> TurnConfig:
    return TurnConfig(clock_mode="wall", duration_s=duration_s, **fields)


def _run(sim: Simulation) -> Simulation:
    """Run the episode. Every record is numbered from 1 without a gap, and the termination record
    is the last one, followed only by the input release and the finish."""
    sim.run()
    assert [entry.seq for entry in sim.records] == list(range(1, len(sim.records) + 1))
    assert sim.records[-1].kind == "Termination"
    assert sim.commands[-3:] == [Record(sim.records[-1]), ReleaseInputs(), Finish(sim.termination)]
    return sim


def _episode(config: TurnConfig, world: World) -> Simulation:
    return _run(Simulation(config, world))


def _names(commands) -> list[str]:
    return [c.entry.kind if isinstance(c, Record) else type(c).__name__ for c in commands]


def _from_end(sim: Simulation) -> list:
    """The commands from the first end signal on."""
    for index, command in enumerate(sim.commands):
        if isinstance(command, Record) and command.entry.kind == "EpisodeEnded":
            return sim.commands[index:]
    raise AssertionError("the episode never ended")


def _assert_nothing_after_the_end(sim: Simulation) -> None:
    """Once the episode is over nothing is captured, started, accepted, executed, paced or asked."""
    late = (Capture, StartGeneration, AcceptResult, Execute, SetTimer, CheckEnd)
    assert [c for c in _from_end(sim) if isinstance(c, late)] == []


def _asked(sim: Simulation) -> list[float]:
    """The times the environment was asked whether the episode is over."""
    return [semantic for semantic, _ in sim.checks]


def _data(sim: Simulation, kind: str) -> list:
    return [entry.data for entry in sim.entries(kind)]


def _timers(sim: Simulation, kind: str) -> list[float]:
    """When each timer of one kind was set to come due."""
    return [data["at"] for data in _data(sim, "TimerSet") if data["kind"] == kind]


def _fired(sim: Simulation, kind: str) -> list[tuple[float, float]]:
    """(handled at, due at) of each timer of one kind that fired."""
    return [(e.semantic, e.data["at"]) for e in sim.entries("TimerFired") if e.data["kind"] == kind]


def _landings(sim: Simulation) -> list[tuple]:
    """(from, requested, to, delta) of every move to the action time."""
    return [
        (d["from_tick"], d["requested_tick"], d["to_tick"], d["delta_ticks"])
        for d in _data(sim, "ClockAdvanced")
        if d["phase"] == "inference_to_action"
    ]


def _boundaries(sim: Simulation) -> list[tuple]:
    """(from, requested, to, dropped) of every move to the next observation."""
    return [
        (d["from_tick"], d["requested_tick"], d["to_tick"], d["dropped_intermediate_frames"])
        for d in _data(sim, "ClockAdvanced")
        if d["phase"] == "action_to_observation_boundary"
    ]


def _ended(sim: Simulation) -> tuple:
    t = sim.termination
    return t.stop_reason, t.cause, t.ended_at, t.snapshot_at


def _outcome(decision, status, tau, next_tick, age, *, terminal=False, dropped=0) -> Outcome:
    return Outcome(
        decision=decision,
        status=status,
        detail="",
        terminal_after_action=terminal,
        action_clock_position=tau,
        next_observation_tick=next_tick,
        dropped_intermediate_frames=dropped,
        post_action_rendered=False,
        frame_age_s=age,
        token_delay_frames=None,
        action_output_tokens=None,
        post_horizon_output_tokens=0,
    )


# ---- the deadline -------------------------------------------------------------------------------


def test_a_deadline_during_a_generation_writes_this_log():
    world = World(replies=[Reply(3.5, CLICK, interruptible=False)], snapshot_s=1.0)
    sim = _episode(_wall(3.0), world)
    assert sim.kinds() == [
        "TurnPolicyStarted",
        "TimerSet",
        "TimerSet",
        "EndChecked",
        "FrameCaptured",
        "InferenceStarted",
        "TimerFired",
        "TimerSet",
        "EndChecked",
        "TimerFired",
        "TimerSet",
        "EndChecked",
        "TimerFired",
        "EpisodeEnded",
        "GenerationAbandoned",
        "GenerationFinished",
        "ResultDiscarded",
        "SnapshotFrozen",
        "Termination",
    ]
    timers = [(d["kind"], d["clock"], d["at"]) for d in _data(sim, "TimerSet")]
    assert timers == [
        ("deadline", "semantic", 3.0),
        ("end_check", "semantic", 1.0),
        ("end_check", "semantic", 2.0),
        ("end_check", "semantic", 3.0),
    ]
    assert [(e.semantic, e.data["kind"]) for e in sim.entries("TimerFired")] == [
        (1.0, "end_check"),
        (2.0, "end_check"),
        (3.0, "deadline"),
    ]


@pytest.mark.parametrize("latency", [7.3, 9.0], ids=["within-the-old-grace", "after-the-old-grace"])
def test_a_reply_after_the_deadline_is_discarded_with_or_without_the_old_grace(latency):
    # A Decision call runs on after the abort; the slow snapshot (6.0 to 9.5) lets its reply, 1.3 s
    # or 3 s after the deadline, come back before the episode is over.
    world = World(replies=[Reply(latency, CLICK, interruptible=False)], snapshot_s=3.5)
    sim = _episode(_wall(6.0), world)
    assert _names(_from_end(sim))[:4] == [
        "EpisodeEnded",
        "FreezeSnapshot",
        "GenerationAbandoned",
        "AbortGeneration",
    ]
    assert _data(sim, "EpisodeEnded") == [{"cause": "deadline", "reason": None, "at": 6.0}]
    assert _data(sim, "GenerationAbandoned") == [{"generation": 1, "reason": "deadline"}]
    assert _data(sim, "ResultDiscarded") == [{"generation": 1}]
    assert (sim.accepted, sim.discarded) == ([], [1])
    assert not {"ActionProposed", "ActionResolved"} & set(sim.kinds())
    assert (sim.executed, sim.outcomes) == ([], [])
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert sim.snapshots == [(9.5, 9.5)]
    assert _ended(sim) == ("time_limit", "deadline", 6.0, 9.5)
    assert sim.termination.invalid is None
    _assert_nothing_after_the_end(sim)


def test_a_result_exactly_at_the_deadline_is_discarded_and_no_check_is_asked_then():
    sim = _episode(_wall(6.0), World(replies=[Reply(6.0, CLICK)]))
    # An end check was due at 6.0 as well; the deadline goes first and stops it.
    assert _timers(sim, "end_check")[-1] == 6.0
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert sim.kinds()[-7:] == [
        "TimerFired",
        "EpisodeEnded",
        "GenerationAbandoned",
        "GenerationFinished",
        "ResultDiscarded",
        "SnapshotFrozen",
        "Termination",
    ]
    assert _fired(sim, "deadline") == [(6.0, 6.0)]
    assert (sim.accepted, sim.discarded) == ([], [1])
    assert sim.outcomes == []
    assert _ended(sim) == ("time_limit", "deadline", 6.0, 6.0)
    _assert_nothing_after_the_end(sim)


def test_a_deadline_while_the_check_before_landing_runs_leaves_the_action_unapplied():
    world = World(replies=[Reply(5.25, CLICK)], check_s=0.5)
    sim = _episode(_wall(6.0), world)
    # The guard answers at 0.5, so the frame is tick 0 at 0.5; the reply lands at 5.75 and the
    # check before execution runs until 6.25.
    assert sim.captures == [(0, 0.5)]
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.75]
    assert _landings(sim) == [(5, 5.75, 5, 0)]
    assert sim.executed == []
    # Frame age runs to the moment landing begins, before its checks (runner.py:571).
    assert sim.outcomes == [_outcome(0, "not_applied_time_limit", 5.75, 6, 5.25)]
    (resolved,) = _data(sim, "ActionResolved")
    assert resolved["status"] == "not_applied_time_limit"
    assert (resolved["terminal_after_action"], resolved["environment_tick"]) == (False, 6)
    assert "ActionApplied" not in sim.kinds()
    assert _boundaries(sim) == []
    assert _ended(sim) == ("time_limit", "deadline", 6.0, 6.25)
    _assert_nothing_after_the_end(sim)


def test_the_snapshot_time_is_when_the_shell_took_it_even_when_a_commit_follows():
    # The deadline settles the decision in hand in the step that freezes the snapshot, so the
    # commit (0.25 s) runs right after the snapshot is taken at 6.25.
    world = World(replies=[Reply(5.25, CLICK)], check_s=0.5, commit_s=0.25)
    sim = _episode(_wall(6.0), world)
    assert sim.snapshots == [(6.25, 6.25)]
    assert sim.termination.snapshot_at == 6.25


def test_a_deadline_during_a_long_execution_keeps_it_applied_and_snapshots_after_it():
    world = World(replies=[Reply(5.25, CLICK)], execute_s=1.5)
    sim = _episode(_wall(6.0), world)
    assert sim.executed == [(0, 5.25, CLICK)]
    # The deadline and the end check due at 6.0 wait for the execution to return at 6.75; the
    # deadline stops the check, and no check follows the execution.
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.25]
    assert sim.snapshots == [(6.75, 6.75)]
    assert sim.outcomes == [_outcome(0, "applied", 5.25, 6, 5.25, terminal=True)]
    assert [d["status"] for d in _data(sim, "ActionApplied")] == ["applied"]
    assert _boundaries(sim) == []
    assert _ended(sim) == ("time_limit", "deadline", 6.0, 6.75)
    _assert_nothing_after_the_end(sim)


def test_a_deadline_inside_the_next_frame_wait_stops_it_at_the_deadline():
    sim = _episode(_wall(5.5), World(replies=[Reply(5.25, CLICK)]))
    # next = min(max(0 + 1, ceil(5.25)), T = 5.5) = 5.5 (runner.py:600-606); the deadline beats the
    # wake due with it, the move stops at tick 5 and every frame crossed but one was dropped.
    assert _timers(sim, "wake") == [5.5]
    assert _boundaries(sim) == [(5, 5.5, 5, 4)]
    assert sim.outcomes == [_outcome(0, "applied", 5.25, 5.5, 5.25, dropped=4)]
    assert _ended(sim) == ("time_limit", "deadline", 5.5, 5.5)
    _assert_nothing_after_the_end(sim)


# ---- the environment's end --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "stop_reason"),
    [("target_reached", "target_reached"), (None, "game_decided")],
    ids=["task-reason", "game-decided"],
)
def test_an_end_found_by_a_periodic_check_during_a_generation_dates_from_the_asking(
    reason, stop_reason
):
    world = World(replies=[Reply(5.0, CLICK)], ends_at=3.5, end_reason=reason, check_s=0.25)
    sim = _episode(_wall(10.0), world)
    # The check asked at 4.0 is the first after the end and answers at 4.25.
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0]
    (ended,) = sim.entries("EpisodeEnded")
    assert (ended.semantic, ended.data) == (
        4.25,
        {"cause": "environment", "reason": reason, "at": 4.0},
    )
    assert _data(sim, "GenerationAbandoned") == [{"generation": 1, "reason": "environment"}]
    assert sim.aborted == [(1, 4.25)]
    assert sim.snapshots == [(4.25, 4.25)]
    assert sim.kinds()[-5:] == [
        "EndChecked",
        "EpisodeEnded",
        "GenerationAbandoned",
        "SnapshotFrozen",
        "Termination",
    ]
    assert sim.outcomes == []
    assert _ended(sim) == (stop_reason, "environment", 4.0, 4.25)
    _assert_nothing_after_the_end(sim)


def test_an_end_found_by_the_check_before_landing_leaves_the_action_unapplied():
    world = World(replies=[Reply(4.5, CLICK)], ends_at=4.25, end_reason="target_reached")
    sim = _episode(_wall(10.0), world)
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0, 4.5]
    assert sim.executed == []
    assert sim.outcomes == [_outcome(0, "not_applied_terminal", 4.5, 4, 4.5)]
    assert _boundaries(sim) == []
    assert _ended(sim) == ("target_reached", "environment", 4.5, 4.5)
    _assert_nothing_after_the_end(sim)


def test_an_end_found_by_the_check_after_execution_keeps_the_action_applied():
    world = World(replies=[Reply(4.5, CLICK)], ends_at=4.625, execute_s=0.25)
    sim = _episode(_wall(10.0), world)
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 4.75]
    assert sim.executed == [(0, 4.5, CLICK)]
    assert _names(_from_end(sim))[:5] == [
        "EpisodeEnded",
        "FreezeSnapshot",
        "ActionResolved",
        "ActionApplied",
        "Commit",
    ]
    assert sim.outcomes == [_outcome(0, "applied", 4.5, 4, 4.5, terminal=True)]
    assert _boundaries(sim) == []
    assert _ended(sim) == ("game_decided", "environment", 4.75, 4.75)
    _assert_nothing_after_the_end(sim)


def test_an_end_during_the_next_frame_wait_still_writes_the_boundary_then_commits():
    # Environment at 2 fps, model input at 1 fps: the environment is asked every 0.5 s while the
    # next observation is a whole second away.
    config = _wall(5.0, env_fps=2.0, input_fps=1.0)
    world = World(replies=[Reply(1.25, CLICK)], ends_at=1.375)
    sim = _episode(config, world)
    assert _timers(sim, "end_check") == [0.5, 1.0, 1.5, 2.0]
    # tau = 1.25 s x 2 = 2.5 ticks = 1.25 input frames, so the next observation is input frame 2,
    # tick 4, at 2.0 s.
    assert _landings(sim) == [(2, 2.5, 2, 0)]
    assert _timers(sim, "wake") == [2.0]
    assert _asked(sim) == [0.0, 0.5, 1.0, 1.25, 1.25, 1.5]
    # The check at 1.5 s finds the end; the move stops at tick 3, inside input frame 1, which is
    # the only one crossed.
    assert _names(_from_end(sim))[:4] == [
        "EpisodeEnded",
        "FreezeSnapshot",
        "ClockAdvanced",
        "Commit",
    ]
    assert _boundaries(sim) == [(2, 4, 3, 0)]
    assert sim.outcomes == [_outcome(0, "applied", 2.5, 4, 1.25)]
    assert _ended(sim) == ("game_decided", "environment", 1.5, 1.5)
    _assert_nothing_after_the_end(sim)


# ---- the clock and the cadence ----------------------------------------------------------------


def test_the_action_lands_at_the_clock_position_and_the_next_frame_is_an_exact_timer():
    replies = [Reply(1.25, CLICK), Reply(0.5, CLICK), Reply(2.25, CLICK), Reply(10.0, CLICK)]
    sim = _episode(_wall(8.0), World(replies=replies))
    assert sim.captures == [(0, 0.0), (2, 2.0), (3, 3.0), (6, 6.0)]
    # The clock does not move before landing: tau is where it stands when the result is handled.
    assert _landings(sim) == [(1, 1.25, 1, 0), (2, 2.5, 2, 0), (5, 5.25, 5, 0)]
    taus = [outcome.action_clock_position for outcome in sim.outcomes]
    assert taus == sorted(taus)
    # next = max(floor(t_k) + 1, ceil(tau)): 2, max(3, 3) = 3, max(4, 6) = 6.
    assert _timers(sim, "wake") == [2.0, 3.0, 6.0]
    assert _fired(sim, "wake") == [(2.0, 2.0), (3.0, 3.0), (6.0, 6.0)]
    assert _boundaries(sim) == [(1, 2, 2, 1), (2, 3, 3, 0), (5, 6, 6, 2)]
    assert sim.outcomes == [
        _outcome(0, "applied", 1.25, 2, 1.25, dropped=1),
        _outcome(1, "applied", 2.5, 3, 0.5),
        _outcome(2, "applied", 5.25, 6, 2.25, dropped=2),
    ]
    assert _ended(sim) == ("time_limit", "deadline", 8.0, 8.0)


def test_the_runner_worked_example_of_one_wall_decision():
    # A1 section 1.2: the loop starts 0.6 s after the origin, the capture takes 0.15 s and the
    # reply returns at 2.6 s.
    approx = pytest.approx
    world = World(replies=[Reply(1.85, CLICK), Reply(10.0, CLICK)], start_s=0.6, capture_s=0.15)
    sim = _episode(_wall(6.0), world)
    assert _timers(sim, "end_check")[0] == 1.0
    assert sim.captures[0] == (0, approx(0.75))
    assert _landings(sim) == [(2, approx(2.6), 2, 0)]
    # next = min(max(0 + 1, ceil(2.6)), T) = 3, waited for until 3 s; dropped = 3 - 0 - 1 = 2.
    assert _timers(sim, "wake") == [3.0]
    assert _boundaries(sim) == [(2, 3, 3, 2)]
    (outcome,) = sim.outcomes
    assert (
        outcome.status,
        outcome.action_clock_position,
        outcome.next_observation_tick,
        outcome.dropped_intermediate_frames,
        outcome.frame_age_s,
    ) == ("applied", approx(2.6), 3, 2, approx(1.85))
    assert sim.captures[1] == (3, approx(3.15))


def test_end_checks_stay_on_the_frame_grid_when_an_execution_delays_one():
    world = World(replies=[Reply(2.25, CLICK), Reply(10.0, CLICK)], execute_s=2.5)
    sim = _episode(_wall(8.0), world)
    assert sim.executed == [(0, 2.25, CLICK)]
    # The check due at 3.0 waits for the execution (2.25 to 4.75), runs once then, and the next
    # one is due at 5.0, back on the grid.
    assert _timers(sim, "end_check") == [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0]
    assert _fired(sim, "end_check") == [
        (1.0, 1.0),
        (2.0, 2.0),
        (4.75, 3.0),
        (5.0, 5.0),
        (6.0, 6.0),
        (7.0, 7.0),
    ]
    # The delayed check, the check after execution and the next loop top all ask at 4.75.
    assert _asked(sim) == [0.0, 1.0, 2.0, 2.25, 4.75, 4.75, 4.75, 5.0, 6.0, 7.0]
    # next = max(0 + 1, max(tau 2.25, tick 4)) = 4, already passed; dropped = 4 - 0 - 1.
    assert sim.outcomes == [_outcome(0, "applied", 2.25, 4, 2.25, dropped=3)]
    assert _ended(sim) == ("time_limit", "deadline", 8.0, 8.0)


# ---- wait actions -----------------------------------------------------------------------------


def test_a_wait_is_a_timer_and_the_environment_is_asked_during_it():
    world = World(replies=[Reply(1.1, Action.wait(2.2)), Reply(10.0, CLICK)])
    sim = _episode(_wall(6.0), world)
    assert sim.executed == []
    # The wait runs from 1.1 to 3.3, then the next observation is tick 3, already passed.
    assert _timers(sim, "wake") == pytest.approx([3.3, 3.0])
    assert _asked(sim) == pytest.approx([0.0, 1.0, 1.1, 2.0, 3.0, 3.3, 3.3, 4.0, 5.0])
    (resolved,) = _data(sim, "ActionResolved")
    assert (resolved["status"], resolved["terminal_after_action"]) == ("wait", False)
    assert resolved["execution_result"] == {
        "duration_simulated_seconds": 2.2,
        "requested_duration_simulated_seconds": 2.2,
        "truncated_by_episode_horizon": False,
        "truncated_by_terminal_condition": False,
    }
    assert [d["status"] for d in _data(sim, "ActionApplied")] == ["wait"]
    assert sim.outcomes == [_outcome(0, "wait", 1.1, 3, 1.1, dropped=2)]


@pytest.mark.parametrize(
    ("duration_s", "ends_at", "latency", "wake_at", "asked", "end_at", "stop_reason", "cause"),
    [
        (
            6.0,
            None,
            4.25,
            7.25,
            [0.0, 1.0, 2.0, 3.0, 4.0, 4.25, 5.0],
            6.0,
            "time_limit",
            "deadline",
        ),
        (10.0, 2.5, 1.25, 4.25, [0.0, 1.0, 1.25, 2.0, 3.0], 3.0, "game_decided", "environment"),
    ],
    ids=["deadline", "environment"],
)
def test_the_end_cuts_a_wait_short_and_it_stays_a_wait(
    duration_s, ends_at, latency, wake_at, asked, end_at, stop_reason, cause
):
    world = World(replies=[Reply(latency, Action.wait(3.0))], ends_at=ends_at)
    sim = _episode(_wall(duration_s), world)
    assert _timers(sim, "wake") == [wake_at]
    assert _asked(sim) == asked
    (resolved,) = _data(sim, "ActionResolved")
    assert (resolved["status"], resolved["terminal_after_action"]) == ("wait", True)
    assert resolved["execution_result"] == {
        "duration_simulated_seconds": 1.75,
        "requested_duration_simulated_seconds": 3.0,
        "truncated_by_episode_horizon": False,
        "truncated_by_terminal_condition": False,
    }
    assert sim.outcomes == [_outcome(0, "wait", latency, end_at, latency, terminal=True)]
    assert _boundaries(sim) == []
    assert _ended(sim) == (stop_reason, cause, end_at, end_at)
    _assert_nothing_after_the_end(sim)


# ---- held keys --------------------------------------------------------------------------------


def test_a_hold_is_released_by_the_shell_at_its_lease_end_and_a_held_key_is_only_extended():
    replies = [
        Reply(1.25, Action.hold_key("a", 3.0)),
        Reply(1.25, Action.hold_key("a+b", 2.0)),
        Reply(0.5, Action.hold_key("a", 0.25)),
        Reply(10.0, CLICK),
    ]
    sim = _episode(_wall(8.0), World(replies=replies))
    assert [(decision, at) for decision, at, _ in sim.executed] == [(0, 1.25), (1, 3.25), (2, 4.5)]
    # Each hold's lease ends at its execution start + duration: 1.25 + 3, 3.25 + 2, 4.5 + 0.25.
    expected = [(0, 4.25), (1, 5.25), (2, 4.75)]
    assert [(d["decision"], d["at"]) for d in _data(sim, "KeyReleaseScheduled")] == expected
    executes = [c for c in sim.commands if isinstance(c, Execute)]
    assert [(c.decision, c.release_at) for c in executes] == expected
    # A held key keeps the later end (sandbox/actions.py:723-727): "a" goes up at
    # max(4.25, 5.25, 4.75) = 5.25, together with "b".
    assert sim.key_events == [(5.25, "up", ("a",)), (5.25, "up", ("b",))]
    assert len(sim.entries("KeysReleased")) == 2


def test_the_shell_releases_a_key_on_time_while_a_long_execution_runs():
    replies = [Reply(1.25, Action.hold_key("a", 3.5)), Reply(0.25, CLICK), Reply(10.0, CLICK)]
    sim = _episode(_wall(9.0), World(replies=replies, execute_s=2.0))
    assert [(decision, at) for decision, at, _ in sim.executed] == [(0, 1.25), (1, 3.5)]
    # The hold's execution returns at 3.25, but its lease runs from the start: 1.25 + 3.5 = 4.75,
    # inside the click's execution from 3.5 to 5.5.
    assert _data(sim, "KeyReleaseScheduled") == [{"decision": 0, "at": 4.75}]
    assert sim.key_events == [(4.75, "up", ("a",))]


def test_a_hold_longer_to_execute_than_to_last_is_released_on_time():
    # The hold's own execution takes 2 s; the shell's timer, armed right after the press, releases
    # the key at 1.25 + 0.5 while the execution is still running.
    replies = [Reply(1.25, Action.hold_key("a", 0.5)), Reply(10.0, CLICK)]
    sim = _episode(_wall(6.0), World(replies=replies, execute_s=2.0))
    assert _data(sim, "KeyReleaseScheduled") == [{"decision": 0, "at": 1.75}]
    assert sim.key_events == [(1.75, "up", ("a",))]


def test_no_end_check_is_asked_once_the_deadline_has_passed():
    # The execution runs from 4.5 to 6.5 over the deadline at 6.0; the check due at 5.0 comes due
    # only after it returns and is not asked, so nothing samples the task after the deadline.
    world = World(replies=[Reply(4.5, CLICK)], execute_s=2.0)
    sim = _episode(_wall(6.0), world)
    assert _fired(sim, "end_check")[-1] == (6.5, 5.0)
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0, 4.5]
    # Not paced after the end: the next observation tick stays the tick at resolution, 6.
    assert sim.outcomes == [_outcome(0, "applied", 4.5, 6, 4.5, terminal=True)]
    assert _ended(sim) == ("time_limit", "deadline", 6.0, 6.5)


# ---- the legacy caps --------------------------------------------------------------------------


@pytest.mark.parametrize("latency", [3.25, 2.0], ids=["slow-reply", "reply-at-the-cap"])
def test_the_per_move_cap_skips_the_decision_and_the_episode_goes_on(latency):
    replies = [Reply(latency, CLICK), Reply(1.25, CLICK), Reply(10.0, CLICK)]
    sim = _episode(_wall(5.0, per_move_cap_s=2.0), World(replies=replies))
    assert _timers(sim, "per_move_cap") == [2.0, 4.0, 6.0]
    assert _data(sim, "GenerationAbandoned") == [
        {"generation": 1, "reason": "per_move_cap"},
        {"generation": 3, "reason": "deadline"},
    ]
    assert (sim.accepted, sim.discarded) == ([2], [1])
    assert [d["decision"] for d in _data(sim, "ActionProposed")] == [1]
    assert sim.executed == [(1, 3.25, CLICK)]
    # The skipped decision lands at the cap, 2.0, and is still paced: next = max(1, 2) = 2.
    assert sim.outcomes == [
        _outcome(0, "skipped_timeout", 2.0, 2, 2.0, dropped=1),
        _outcome(1, "applied", 3.25, 4, 1.25, dropped=1),
    ]
    # A move over the cap places no action.
    assert [outcome.action for outcome in sim.outcomes] == [None, CLICK]
    assert _asked(sim) == [0.0, 1.0, 2.0, 2.0, 3.0, 3.25, 3.25, 4.0, 4.0]
    assert _ended(sim) == ("time_limit", "deadline", 5.0, 5.0)


def test_a_reply_over_the_cap_by_its_own_latency_is_skipped_though_it_came_back_in_time():
    replies = [Reply(1.25, CLICK, agent_latency=2.5), Reply(10.0, CLICK)]
    sim = _episode(_wall(4.0, per_move_cap_s=2.0), World(replies=replies))
    assert sim.accepted == [1]
    assert [d["decision"] for d in _data(sim, "ActionProposed")] == [0]
    assert sim.executed == []
    assert sim.outcomes == [_outcome(0, "skipped_timeout", 1.25, 2, 1.25, dropped=1)]
    # The deadline and the second cap are both due at 4.0; the deadline was set first.
    assert _data(sim, "GenerationAbandoned") == [{"generation": 2, "reason": "deadline"}]
    assert _ended(sim) == ("time_limit", "deadline", 4.0, 4.0)


def test_the_per_move_cap_is_at_least_a_tenth_of_a_second():
    sim = _episode(_wall(3.0, per_move_cap_s=0.05), World(replies=[Reply(1.0, CLICK)] * 3))
    assert _timers(sim, "per_move_cap") == pytest.approx([0.1, 1.1, 2.1])
    assert sim.outcomes[0] == _outcome(0, "skipped_timeout", 0.1, 1, 0.1)
    assert [outcome.status for outcome in sim.outcomes] == ["skipped_timeout"] * 3
    assert _ended(sim) == ("time_limit", "deadline", 3.0, 3.0)


def test_the_think_budget_skips_the_decision_then_ends_after_its_commit():
    replies = [Reply(1.25, CLICK), Reply(1.5, CLICK)]
    sim = _episode(_wall(10.0, think_budget_s=2.5), World(replies=replies))
    # 1.25 s spent, then 2.75 s >= 2.5 s: the second decision is over the budget.
    assert sim.executed == [(0, 1.25, CLICK)]
    assert [d["decision"] for d in _data(sim, "ActionProposed")] == [0, 1]
    assert sim.outcomes == [
        _outcome(0, "applied", 1.25, 2, 1.25, dropped=1),
        _outcome(1, "skipped_budget", 3.5, 4, 1.5, dropped=1),
    ]
    # A skipped action is not checked before or after execution.
    assert _asked(sim) == [0.0, 1.0, 1.25, 1.25, 2.0, 2.0, 3.0, 4.0]
    kinds = sim.kinds()
    assert kinds.count("OutcomeCommitted") == 2
    assert kinds[kinds.index("EpisodeEnded") - 1] == "OutcomeCommitted"
    assert _data(sim, "EpisodeEnded") == [{"cause": "think_budget", "reason": None, "at": 4.0}]
    assert _ended(sim) == ("think_budget", "think_budget", 4.0, 4.0)
    _assert_nothing_after_the_end(sim)


# ---- invalidations ----------------------------------------------------------------------------


def _failing(*checks: int) -> dict[int, str]:
    """End checks (1-based) whose state read fails, each with its own error."""
    return {check: f"state read {check} failed" for check in checks}


def _read_errors(sim: Simulation) -> list:
    return [event.error for event in _data(sim, "EndChecked")]


def test_three_failed_periodic_checks_in_a_row_invalidate_the_run_during_a_generation():
    # The checks at 1, 2 and 3 s fail; the third invalidates the run with its own error.
    world = World(replies=[Reply(5.0, CLICK)], check_errors=_failing(2, 3, 4))
    sim = _episode(_wall(10.0), world)
    invalid = Invalidated("environment", "terminal_state_check_failed", "state read 4 failed")
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0]
    assert _data(sim, "RunInvalidated") == [
        {
            "category": "environment",
            "reason": "terminal_state_check_failed",
            "detail": "state read 4 failed",
        }
    ]
    assert _data(sim, "EpisodeEnded") == [{"cause": "invalid", "reason": None, "at": 3.0}]
    assert _data(sim, "GenerationAbandoned") == [{"generation": 1, "reason": "invalid"}]
    assert sim.outcomes == []
    assert _ended(sim) == ("invalid_run", "invalid", 3.0, 3.0)
    assert sim.termination.invalid == invalid
    _assert_nothing_after_the_end(sim)


@pytest.mark.parametrize(
    ("failing", "asked", "status", "terminal", "executed"),
    [
        ((2, 3, 4), [0.0, 1.0, 2.0, 2.25], "invalid_run", False, []),
        ((3, 4, 5), [0.0, 1.0, 2.0, 2.25, 2.25], "applied", True, [(0, 2.25, CLICK)]),
    ],
    ids=["before-execution", "after-execution"],
)
def test_a_third_failed_check_in_a_row_around_execution_settles_the_decision(
    failing, asked, status, terminal, executed
):
    world = World(replies=[Reply(2.25, CLICK)], check_errors=_failing(*failing))
    sim = _episode(_wall(10.0), world)
    assert _asked(sim) == asked
    assert sim.executed == executed
    assert sim.outcomes == [_outcome(0, status, 2.25, 2, 2.25, terminal=terminal)]
    assert _ended(sim) == ("invalid_run", "invalid", 2.25, 2.25)
    assert sim.termination.invalid == Invalidated(
        "environment", "terminal_state_check_failed", f"state read {failing[-1]} failed"
    )
    _assert_nothing_after_the_end(sim)


@pytest.mark.parametrize(
    "failing", [(1,), (1, 2), (1, 2, 4, 5)], ids=["one", "two", "two-answer-two"]
)
def test_fewer_than_three_failed_checks_in_a_row_leave_the_run_valid(failing):
    # A failed read answers "not over" and the decision goes on as without it; the answer at 2 s
    # starts the count over, so the failures right before and after execution are one and two.
    # The task ends at 3 s, seen by the periodic check filed before the pace move's wake.
    world = World(replies=[Reply(2.25, CLICK)], ends_at=3.0, check_errors=_failing(*failing))
    sim = _episode(_wall(10.0), world)
    assert _asked(sim) == [0.0, 1.0, 2.0, 2.25, 2.25, 3.0]
    assert _read_errors(sim) == [_failing(*failing).get(check) for check in range(1, 7)]
    assert sim.executed == [(0, 2.25, CLICK)]
    assert sim.outcomes == [_outcome(0, "applied", 2.25, 3, 2.25, dropped=2)]
    assert sim.entries("RunInvalidated") == []
    assert _ended(sim) == ("game_decided", "environment", 3.0, 3.0)
    assert sim.termination.invalid is None


def test_an_invalidation_posted_during_a_generation_ends_the_episode_at_once():
    # Another thread's failure, such as the wall key-release timer's, arrives through the inbox.
    invalid = Invalidated("environment", "wall_key_release_failed", "keyup failed")
    world = World(replies=[Reply(4.0, CLICK)], invalidations=((2.5, invalid),))
    sim = _episode(_wall(6.0), world)
    assert sim.kinds()[-6:] == [
        "Invalidated",
        "RunInvalidated",
        "EpisodeEnded",
        "GenerationAbandoned",
        "SnapshotFrozen",
        "Termination",
    ]
    assert _data(sim, "EpisodeEnded") == [{"cause": "invalid", "reason": None, "at": 2.5}]
    assert _data(sim, "GenerationAbandoned") == [{"generation": 1, "reason": "invalid"}]
    assert sim.aborted == [(1, 2.5)]
    assert _asked(sim) == [0.0, 1.0, 2.0]
    assert sim.outcomes == []
    assert _ended(sim) == ("invalid_run", "invalid", 2.5, 2.5)
    assert sim.termination.invalid == invalid
    _assert_nothing_after_the_end(sim)


def test_a_failed_execution_invalidates_and_its_decision_is_still_reported():
    world = World(
        replies=[Reply(1.25, CLICK)], execute_errors={0: "guest unreachable"}, execute_s=0.25
    )
    sim = _episode(_wall(6.0), world)
    assert sim.executed == [(0, 1.25, CLICK)]
    assert _asked(sim) == [0.0, 1.0, 1.25]
    assert _data(sim, "RunInvalidated") == [
        {
            "category": "environment",
            "reason": "action_execution_failed",
            "detail": "guest unreachable",
        }
    ]
    assert sim.outcomes == [_outcome(0, "invalid_run", 1.25, 1, 1.25)]
    assert "ActionApplied" not in sim.kinds()
    assert _ended(sim) == ("invalid_run", "invalid", 1.5, 1.5)
    _assert_nothing_after_the_end(sim)


def test_a_failed_commit_invalidates_and_no_decision_follows():
    replies = [Reply(1.25, CLICK), Reply(1.0, CLICK)]
    sim = _episode(_wall(6.0), World(replies=replies, commit_errors={0: "feedback lost"}))
    assert sim.outcomes == [_outcome(0, "applied", 1.25, 2, 1.25, dropped=1)]
    assert _data(sim, "RunInvalidated") == [
        {
            "category": "provider",
            "reason": "action_commit_feedback_failed",
            "detail": "feedback lost",
        }
    ]
    assert len(sim.captures) == 1
    assert len(sim.entries("InferenceStarted")) == 1
    assert _ended(sim) == ("invalid_run", "invalid", 2.0, 2.0)
    _assert_nothing_after_the_end(sim)


def test_a_failed_snapshot_invalidates_and_has_no_time():
    world = World(replies=[Reply(10.0, CLICK)], snapshot_error="state read failed")
    sim = _episode(_wall(3.0), world)
    assert sim.kinds()[-6:] == [
        "TimerFired",
        "EpisodeEnded",
        "GenerationAbandoned",
        "SnapshotFrozen",
        "RunInvalidated",
        "Termination",
    ]
    assert _ended(sim) == ("invalid_run", "deadline", 3.0, None)
    assert sim.termination.ops_snapshot_at is None
    assert sim.termination.invalid == Invalidated(
        "environment", "evaluation_snapshot_failed", "state read failed"
    )


def test_a_frozen_stream_invalidates_on_the_third_identical_frame():
    world = World(replies=[Reply(1.25, CLICK)] * 3, frozen_from=0, capture_s=0.25)
    sim = _episode(_wall(10.0), world)
    # Frames at 0, 2 and 4 s, each taking 0.25 s, all with the first frame's identity.
    assert [timestamp for _, timestamp in sim.captures] == [0.25, 2.25, 4.25]
    assert len(sim.entries("InferenceStarted")) == 2
    assert [outcome.status for outcome in sim.outcomes] == ["applied", "applied"]
    assert _data(sim, "RunInvalidated") == [
        {
            "category": "environment",
            "reason": "observation_stream_frozen",
            "detail": repr(("frame-0", 0, "sha-0")),
        }
    ]
    assert _ended(sim) == ("invalid_run", "invalid", 4.25, 4.25)
    _assert_nothing_after_the_end(sim)
