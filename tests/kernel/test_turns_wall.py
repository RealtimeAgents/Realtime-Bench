"""The turn-based policy on the wall clock, run in the discrete-event simulation.

The first end signal (the deadline, an end the environment reports, an invalidation) stops the
episode at once: the snapshot is frozen, the generation in flight is given up and its late result
never lands. The environment is asked whether it ended once each environment frame: on the frame
grid, during generations and waits too, and at the loop top or right before execution only in a
frame not read yet; it is asked again after every action. Waits, key releases, the next frame and
the legacy per-move cap are timers.

Every expected value is worked out by hand: from today's runner where the kernel keeps its rules
(evaluation/runner.py, sandbox/actions.py) and from phase2.md (事件内核; 时间和结束判定) where it
changes them. Times are seconds since the episode origin; env_fps = input_fps = 1 unless a test
says otherwise.
"""

from __future__ import annotations

import dataclasses

import pytest

from realtime_framework.entity.actions import Action
from realtime_framework.entity.config import RunnerTimingConfig
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
from tests.kernel.test_scenarios import _past_the_deadline

CLICK = Action.click(10, 20)


def _wall(duration_s: float, **fields) -> TurnConfig:
    return TurnConfig(clock_mode="wall", duration_s=duration_s, **fields)


def _run(sim: Simulation) -> Simulation:
    """Run the episode. Every record is numbered from 1 without a gap, the termination record is
    the last one, followed only by the input release and the finish, and nothing is captured,
    started, executed or asked from the deadline on."""
    sim.run()
    assert [entry.seq for entry in sim.records] == list(range(1, len(sim.records) + 1))
    assert sim.records[-1].kind == "Termination"
    assert sim.commands[-3:] == [Record(sim.records[-1]), ReleaseInputs(), Finish(sim.termination)]
    assert _past_the_deadline(sim) == []
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
    # The reply comes back at 5 s, on the frame boundary ahead of the periodic check due then, so
    # frame 5 has not been read and is read before execution; that read takes until 6.25 s.
    world = World(replies=[Reply(5.0, CLICK)], check_seconds={6: 1.25})
    sim = _episode(_wall(6.0), world)
    assert sim.captures == [(0, 0.0)]
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert _landings(sim) == [(5, 5.0, 5, 0)]
    assert sim.executed == []
    # Frame age runs to the moment landing begins, before its checks (runner.py _apply).
    assert sim.outcomes == [_outcome(0, "not_applied_time_limit", 5.0, 6, 5.0)]
    (resolved,) = _data(sim, "ActionResolved")
    assert resolved["status"] == "not_applied_time_limit"
    assert (resolved["terminal_after_action"], resolved["environment_tick"]) == (False, 6)
    assert "ActionApplied" not in sim.kinds()
    assert _boundaries(sim) == []
    assert _ended(sim) == ("time_limit", "deadline", 6.0, 6.25)
    _assert_nothing_after_the_end(sim)


def test_the_snapshot_time_is_when_the_shell_took_it_even_when_a_commit_follows():
    # The environment at 2 fps: the click at 5.25 s waits for the next observation, input frame 6
    # at 6 s, and the check due at 5.5 s, the 13th, runs until 6.25 s. The deadline then settles
    # the decision in the step that freezes the snapshot, so the commit (0.25 s) runs right after
    # the snapshot is taken at 6.25.
    world = World(replies=[Reply(5.25, CLICK), Reply(9.0)], commit_s=0.25, check_seconds={13: 0.75})
    sim = _episode(_wall(6.0, env_fps=2.0), world)
    assert sim.checks[-1] == (5.5, 5.5)
    names = _names(_from_end(sim))
    assert names.index("FreezeSnapshot") < names.index("Commit")
    assert sim.snapshots == [(6.25, 6.25)]
    assert sim.termination.snapshot_at == 6.25


def test_a_deadline_during_a_long_execution_keeps_it_applied_and_snapshots_after_it():
    world = World(replies=[Reply(5.25, CLICK)], execute_s=1.5)
    sim = _episode(_wall(6.0), world)
    assert sim.executed == [(0, 5.25, CLICK)]
    # Frame 5 was read at 5 s, so nothing is read before the execution. The deadline and the end
    # check due at 6.0 wait for it to return at 6.75; the deadline stops the check, and no check
    # follows the execution.
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert sim.snapshots == [(6.75, 6.75)]
    assert sim.outcomes == [_outcome(0, "applied", 5.25, 6, 5.25, terminal=True)]
    assert [d["status"] for d in _data(sim, "ActionApplied")] == ["applied"]
    assert _boundaries(sim) == []
    assert _ended(sim) == ("time_limit", "deadline", 6.0, 6.75)
    _assert_nothing_after_the_end(sim)


@pytest.mark.parametrize(
    ("world", "asked", "captures", "started", "executed", "outcomes", "ended"),
    [
        # The loop starts at 2.5 s and its check answers at 3.25 s: nothing is captured.
        (
            World(replies=[Reply(9.0, CLICK)], start_s=2.5, check_s=0.75),
            [2.5],
            [],
            [],
            [],
            [],
            ("time_limit", "deadline", 3.0, 3.25),
        ),
        # The frame at 2 s is captured until 2.5 s; the periodic check due at 2 s runs first,
        # until 3.25 s. No generation is started on the frame.
        (
            World(replies=[Reply(9.0, CLICK)], start_s=1.25, check_s=0.75, capture_s=0.5),
            [1.25, 2.5],
            [(2, 2.5)],
            [],
            [],
            [],
            ("time_limit", "deadline", 3.0, 3.25),
        ),
        # The reply at 2 s, on the frame boundary ahead of frame 2's periodic check, is checked
        # until 3.5 s: the click is not executed.
        (
            World(replies=[Reply(2.0, CLICK), Reply(9.0)], check_seconds={3: 1.5}),
            [0.0, 1.0, 2.0],
            [(0, 0.0)],
            [0.0],
            [],
            [_outcome(0, "not_applied_time_limit", 2.0, 3, 2.0)],
            ("time_limit", "deadline", 3.0, 3.5),
        ),
        # The click at 1.5 s, in frame 1 read already, is executed until 2.75 s; the periodic
        # check due at 2 s runs first, until 3 s. The click stays applied and no check follows it.
        (
            World(replies=[Reply(1.25, CLICK), Reply(9.0)], check_s=0.25, execute_s=1.25),
            [0.0, 1.0, 2.75],
            [(0, 0.25)],
            [0.25],
            [(0, 1.5, CLICK)],
            [_outcome(0, "applied", 1.5, 3, 1.25, terminal=True)],
            ("time_limit", "deadline", 3.0, 3.0),
        ),
    ],
    ids=["loop-top", "capture", "before-landing", "after-execution"],
)
def test_an_answer_handled_after_the_deadline_is_not_acted_on(
    world, asked, captures, started, executed, outcomes, ended
):
    # An answer filed before the deadline at 3 s is handled after it: the answer to a check asked
    # before the deadline, or one that waited while a synchronous periodic check ran across it.
    # The deadline then ends the episode instead of what the answer would lead to (phase2.md,
    # 时间和结束判定).
    sim = _episode(_wall(3.0), world)
    assert _asked(sim) == asked
    assert sim.captures == captures
    assert [entry.semantic for entry in sim.entries("InferenceStarted")] == started
    assert sim.executed == executed
    assert sim.outcomes == outcomes
    assert _ended(sim) == ended
    _assert_nothing_after_the_end(sim)


def test_an_action_in_a_frame_read_already_is_executed_at_once_and_read_after():
    # The reply at 4.75 s lands in frame 4, read at 4 s: it is executed at once, with no check
    # before it (today's loop checks first and executes when that check returns, at 5.5 s). The
    # check after it runs until 5.5 s, during which the periodic check due at 5 s comes due; the
    # answer is filed at its asking, 4.75 s, and handled first, then frame 5 is read, until
    # 6.25 s, past the deadline.
    world = World(replies=[Reply(4.0, CLICK), Reply(9.0)], check_s=0.75)
    sim = _episode(_wall(6.0), world)
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0, 4.75, 5.5]
    assert sim.executed == [(0, 4.75, CLICK)]
    assert [(o.status, o.terminal_after_action) for o in sim.outcomes] == [("applied", False)]
    assert _ended(sim) == ("time_limit", "deadline", 6.0, 6.25)


def test_a_deadline_inside_the_next_frame_wait_stops_it_at_the_deadline():
    sim = _episode(_wall(5.5), World(replies=[Reply(5.25, CLICK)]))
    # next = min(max(0 + 1, ceil(5.25)), T = 5.5) = 5.5 (runner.py _apply); the deadline beats the
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


@pytest.mark.parametrize(
    ("latency", "check_s", "ends_at", "asked", "executed", "outcomes", "ended"),
    [
        # The click at 2.75 s lands in frame 2, read at 2 s before the task ended at 2.5 s; the
        # check after it is asked at 2.75 s, after the end, and answers at 3 s, with the deadline.
        (
            2.5,
            0.25,
            2.5,
            [0.0, 1.0, 2.0, 2.75],
            [(0, 2.75, CLICK)],
            [_outcome(0, "applied", 2.75, 3, 2.5, terminal=True)],
            ("game_decided", "environment", 2.75, 3.0),
        ),
        # Checks take 1.25 s: the periodic check due at 2 s is asked at 2.5 s, after the task ended
        # at 2.25 s, and answers at 3.75 s, during the generation.
        (9.0, 1.25, 2.25, [0.0, 1.25, 2.5], [], [], ("game_decided", "environment", 2.5, 3.75)),
    ],
    ids=["after-execution", "periodic"],
)
def test_an_end_found_by_a_check_asked_before_the_deadline_is_the_tasks_end(
    latency, check_s, ends_at, asked, executed, outcomes, ended
):
    # The answer comes after the deadline, but the end dates from the asking, before it: the task
    # ended before the time limit (phase2.md, 时间和结束判定; decision 4).
    world = World(replies=[Reply(latency, CLICK)], ends_at=ends_at, check_s=check_s)
    sim = _episode(_wall(3.0), world)
    assert _asked(sim) == asked
    assert sim.executed == executed
    assert sim.outcomes == outcomes
    assert _fired(sim, "deadline") == []
    assert _ended(sim) == ended
    _assert_nothing_after_the_end(sim)


def test_an_end_found_by_a_check_makes_a_result_back_during_it_stale():
    # The check at 2 s finds the end (from 1.5 s) and answers at 2.25 s; the reply came back at
    # 2.15 s meanwhile. The end is dated 2 s, before the result, which is discarded.
    world = World(replies=[Reply(1.9, CLICK)], ends_at=1.5, check_s=0.25)
    sim = _episode(_wall(10.0), world)
    assert _asked(sim) == [0.0, 1.0, 2.0]
    assert (sim.accepted, sim.discarded, sim.outcomes) == ([], [1], [])
    assert _ended(sim) == ("game_decided", "environment", 2.0, 2.25)


def test_an_end_found_by_the_check_before_landing_leaves_the_action_unapplied():
    # The reply comes back at 4 s, on the frame boundary ahead of the periodic check due then:
    # frame 4 has not been read, so it is read before execution, and that read finds the end.
    world = World(replies=[Reply(4.0, CLICK)], ends_at=3.5, end_reason="target_reached")
    sim = _episode(_wall(10.0), world)
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert sim.executed == []
    assert sim.outcomes == [_outcome(0, "not_applied_terminal", 4.0, 4, 4.0)]
    assert _boundaries(sim) == []
    assert _ended(sim) == ("target_reached", "environment", 4.0, 4.0)
    _assert_nothing_after_the_end(sim)


def test_an_end_in_a_frame_read_already_is_found_by_the_check_after_the_action():
    # The task ends at 4.25 s, after frame 4 was read at 4 s. The reply at 4.5 s lands in that
    # frame, which is not read again before execution: the click is executed, and the check after
    # it finds the end (today's loop reads before landing and leaves the click unapplied).
    world = World(replies=[Reply(4.5, CLICK)], ends_at=4.25, end_reason="target_reached")
    sim = _episode(_wall(10.0), world)
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0, 4.5]
    assert sim.executed == [(0, 4.5, CLICK)]
    assert sim.outcomes == [_outcome(0, "applied", 4.5, 4, 4.5, terminal=True)]
    assert _boundaries(sim) == []
    assert _ended(sim) == ("target_reached", "environment", 4.5, 4.5)
    _assert_nothing_after_the_end(sim)


def test_an_end_found_by_the_check_after_execution_keeps_the_action_applied():
    world = World(replies=[Reply(4.5, CLICK)], ends_at=4.625, execute_s=0.25)
    sim = _episode(_wall(10.0), world)
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0, 4.75]
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


def test_an_action_whose_check_after_it_answers_past_the_deadline_is_not_paced():
    # T = 3. The click back at 2.5 s executes until 2.8 s; the check after it, asked then, answers
    # "not over" at 3.3 s and is handled first, filed at its asking. Tick 3 is the horizon, so
    # nothing is paced (runner.py _apply paces only while tick_index() < max_ticks): the decision
    # commits at once, nothing dropped, and the deadline ends the episode.
    world = World(replies=[Reply(2.5, CLICK), Reply(9.0)], execute_s=0.3, check_seconds={4: 0.5})
    sim = _episode(_wall(3.0), world)
    assert _asked(sim) == [0.0, 1.0, 2.0, 2.8]
    assert _boundaries(sim) == []
    assert sim.outcomes == [_outcome(0, "applied", 2.5, 3, 2.5)]
    assert _ended(sim) == ("time_limit", "deadline", 3.0, 3.3)


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
    # The click at 1.25 s lands in frame 2, read at 1 s, and is read once after it.
    assert _asked(sim) == [0.0, 0.5, 1.0, 1.25, 1.5]
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


def test_the_state_is_read_once_a_frame_and_after_every_action():
    # The frames are the environment's, one a second. Decision 0's reply is back at 1 s, on the
    # frame boundary ahead of the periodic check due then, so frame 1 is read before landing and
    # again after the click; the periodic check due at 1 s then finds frame 1 read and asks
    # nothing, nor does decision 1's loop top, at 1 s, the click's next observation. Its reply
    # at 1.5 s lands in frame 1 too: read only after the click. Frames 2 and 3 are read by their
    # periodic checks, and the loop top in frame 2 reads nothing.
    replies = [Reply(1.0, CLICK), Reply(0.5, CLICK), Reply(9.0)]
    sim = _episode(_wall(4.0), World(replies=replies))
    assert _asked(sim) == [
        0.0,  # loop top, the first read of frame 0
        1.0,  # before landing, frame 1 not read yet
        1.0,  # after the click
        1.5,  # after the click
        2.0,  # frame 2's periodic check
        3.0,  # frame 3's periodic check
    ]
    assert _fired(sim, "end_check") == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    assert sim.executed == [(0, 1.0, CLICK), (1, 1.5, CLICK)]
    assert _ended(sim) == ("time_limit", "deadline", 4.0, 4.0)


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
    # Frame 2, read at 2.0, is not read again before the execution. The delayed check reads
    # frame 4 at 4.75 and the check after execution reads it again; the next loop top, in frame
    # 4, does not.
    assert _asked(sim) == [0.0, 1.0, 2.0, 4.75, 4.75, 5.0, 6.0, 7.0]
    # next = max(0 + 1, max(tau 2.25, tick 4)) = 4, already passed; dropped = 4 - 0 - 1.
    assert sim.outcomes == [_outcome(0, "applied", 2.25, 4, 2.25, dropped=3)]
    assert _ended(sim) == ("time_limit", "deadline", 8.0, 8.0)


# ---- wait actions -----------------------------------------------------------------------------


def test_a_wait_is_a_timer_and_the_environment_is_asked_during_it():
    world = World(replies=[Reply(1.1, Action.wait(2.2)), Reply(10.0, CLICK)])
    sim = _episode(_wall(6.0), world)
    assert sim.executed == []
    # The wait runs from 1.1 to 3.3, then the next observation is tick 3, already passed. Frame
    # 1 was read at 1.0, so nothing is read before the wait; its end is read at 3.3, and the next
    # loop top, in frame 3, reads nothing.
    assert _timers(sim, "wake") == pytest.approx([3.3, 3.0])
    assert _asked(sim) == pytest.approx([0.0, 1.0, 2.0, 3.0, 3.3, 4.0, 5.0])
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
    (
        "duration_s",
        "ends_at",
        "latency",
        "wake_at",
        "asked",
        "end_at",
        "stop_reason",
        "cause",
        "by_end",
    ),
    [
        # The deadline cuts it as today's sleep(min(duration, remaining)) does, and today's record
        # flags neither truncation (sandbox/actions.py GuestEnvironment.execute).
        (
            6.0,
            None,
            4.25,
            7.25,
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            6.0,
            "time_limit",
            "deadline",
            False,
        ),
        # The task's end, which today's sleep does not notice, cuts it as on the token clock.
        (
            10.0,
            2.5,
            1.25,
            4.25,
            [0.0, 1.0, 2.0, 3.0],
            3.0,
            "game_decided",
            "environment",
            True,
        ),
    ],
    ids=["deadline", "environment"],
)
def test_the_end_cuts_a_wait_short_and_it_stays_a_wait(
    duration_s, ends_at, latency, wake_at, asked, end_at, stop_reason, cause, by_end
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
        "truncated_by_terminal_condition": by_end,
    }
    assert sim.outcomes == [_outcome(0, "wait", latency, end_at, latency, terminal=True)]
    assert _boundaries(sim) == []
    assert _ended(sim) == (stop_reason, cause, end_at, end_at)
    _assert_nothing_after_the_end(sim)


# ---- held keys --------------------------------------------------------------------------------


def test_each_hold_carries_the_end_of_its_own_lease_and_the_shell_keeps_the_later_one():
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
    # The shell, as Execute asks of it, keeps a held key down until its later end
    # (sandbox/actions.py GuestEnvironment._execute_wall_hold); the simulated shell's timer lifts
    # "a" at max(4.25, 5.25, 4.75) = 5.25, together with "b".
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


def test_a_release_the_shells_timer_fails_invalidates_the_run():
    # The shell's timer cannot lift "a" at 1.25 s, during the second generation, and answers with
    # the error, as the environment gives up on a wall release after its retries
    # (sandbox/actions.py GuestEnvironment._wall_release_due).
    replies = [Reply(0.25, Action.hold_key("a", 1.0)), Reply(5.0, CLICK)]
    world = World(replies=replies, release_errors={1: "xdotool keyup failed"})
    sim = _episode(_wall(3.0), world)
    assert sim.key_events == []
    assert _data(sim, "RunInvalidated") == [
        {
            "category": "environment",
            "reason": "wall_key_release_failed",
            "detail": "xdotool keyup failed",
        }
    ]
    assert _data(sim, "GenerationAbandoned") == [{"generation": 2, "reason": "invalid"}]
    assert _ended(sim) == ("invalid_run", "invalid", 1.25, 1.25)
    _assert_nothing_after_the_end(sim)


@pytest.mark.parametrize("execute_s", [2.0, 1.5], ids=["past-the-deadline", "on-it"])
def test_no_end_check_is_asked_once_the_deadline_has_passed(execute_s):
    # The execution runs from 4.5 s over or up to the deadline at 6 s; frame 4 was read at 4 s, so
    # nothing is read before it. The check due at 5 s comes due only after it returns and is not
    # asked, so nothing samples the task from the deadline on.
    returned = 4.5 + execute_s
    world = World(replies=[Reply(4.5, CLICK)], execute_s=execute_s)
    sim = _episode(_wall(6.0), world)
    assert _fired(sim, "end_check")[-1] == (returned, 5.0)
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 4.0]
    # Not paced after the end: the next observation tick stays the tick at resolution, 6.
    assert sim.outcomes == [_outcome(0, "applied", 4.5, 6, 4.5, terminal=True)]
    assert _ended(sim) == ("time_limit", "deadline", 6.0, returned)


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
    # Each frame is read once, by its periodic check; the click is read once more after it.
    assert _asked(sim) == [0.0, 1.0, 2.0, 3.0, 3.25, 4.0]
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


def test_a_call_over_the_per_move_cap_costs_the_cap_against_the_think_budget():
    # Budget 1.5 s, cap 1 s: each call outlasts the cap and costs 1 s (runner.py _charge). The
    # second brings the spend to 2 s >= 1.5 s, and the episode ends once it is reported, at 2 s.
    world = World(replies=[Reply(5.0, CLICK)] * 3)
    sim = _episode(_wall(10.0, per_move_cap_s=1.0, think_budget_s=1.5), world)
    assert [outcome.status for outcome in sim.outcomes] == ["skipped_timeout", "skipped_timeout"]
    assert _ended(sim) == ("think_budget", "think_budget", 2.0, 2.0)


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
    # A skipped action is not checked before or after execution; each frame is read once, by its
    # periodic check, and the applied click once more after it.
    assert _asked(sim) == [0.0, 1.0, 1.25, 2.0, 3.0, 4.0]
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
    ("latency", "failing", "asked", "status", "terminal", "executed"),
    [
        # The reply at 2 s, on the frame boundary ahead of its periodic check, is read before
        # execution: the loop top, the periodic check at 1 s and that read fail.
        (2.0, (1, 2, 3), [0.0, 1.0, 2.0], "invalid_run", False, []),
        # The reply at 2.25 s lands in frame 2, read at 2 s: the periodic checks at 1 s and 2 s and
        # the read after execution fail.
        (2.25, (2, 3, 4), [0.0, 1.0, 2.0, 2.25], "applied", True, [(0, 2.25, CLICK)]),
    ],
    ids=["before-execution", "after-execution"],
)
def test_a_third_failed_check_in_a_row_around_execution_settles_the_decision(
    latency, failing, asked, status, terminal, executed
):
    world = World(replies=[Reply(latency, CLICK)], check_errors=_failing(*failing))
    sim = _episode(_wall(10.0), world)
    assert _asked(sim) == asked
    assert sim.executed == executed
    assert sim.outcomes == [_outcome(0, status, latency, 2, latency, terminal=terminal)]
    assert _ended(sim) == ("invalid_run", "invalid", latency, latency)
    assert sim.termination.invalid == Invalidated(
        "environment", "terminal_state_check_failed", f"state read {failing[-1]} failed"
    )
    _assert_nothing_after_the_end(sim)


@pytest.mark.parametrize("failing", [(1,), (1, 2), (1, 2, 4)], ids=["one", "two", "two-answer-one"])
def test_fewer_than_three_failed_checks_in_a_row_leave_the_run_valid(failing):
    # A failed read answers "not over" and the decision goes on as without it; the answer at 2 s
    # starts the count over, so the failure after execution is one. The task ends at 3 s, seen by
    # the periodic check filed before the pace move's wake.
    world = World(replies=[Reply(2.25, CLICK)], ends_at=3.0, check_errors=_failing(*failing))
    sim = _episode(_wall(10.0), world)
    assert _asked(sim) == [0.0, 1.0, 2.0, 2.25, 3.0]
    assert _read_errors(sim) == [_failing(*failing).get(check) for check in range(1, 6)]
    assert sim.executed == [(0, 2.25, CLICK)]
    assert sim.outcomes == [_outcome(0, "applied", 2.25, 3, 2.25, dropped=2)]
    assert sim.entries("RunInvalidated") == []
    assert _ended(sim) == ("game_decided", "environment", 3.0, 3.0)
    assert sim.termination.invalid is None


def test_failed_reads_at_one_instant_count_in_the_order_they_were_made():
    # The click is executed from 2.25 s to 3.25 s: the periodic check due at 3 s is asked at
    # 3.25 s, then the check after execution, both at 3.25 s. The periodic checks at 1 s and 2 s
    # fail, the first at 3.25 s answers and the second fails: failures in a row 1, 2, 0, 1, so the
    # run stays valid. A failed read is no end and is handled in its turn, not ahead of an earlier
    # answer (phase2.md, 时间和结束判定: in a row).
    world = World(
        replies=[Reply(2.25, CLICK), Reply(100.0)], execute_s=1.0, check_errors=_failing(2, 3, 5)
    )
    sim = _episode(_wall(4.0), world)
    assert _asked(sim)[:5] == [0.0, 1.0, 2.0, 3.25, 3.25]
    requests = [event.request for event in _data(sim, "EndChecked")]
    assert requests == sorted(requests)
    assert sim.entries("RunInvalidated") == []
    assert _ended(sim)[:2] == ("time_limit", "deadline")


def test_a_third_failed_read_asked_before_the_deadline_invalidates_though_it_answers_after():
    # Reads take 1.25 s and all fail: the loop top at 0, the periodic checks asked at 1.25 s and
    # 2.5 s. The third is asked before the deadline and answers at 3.75 s; dated by its asking, it
    # is handled before the deadline, and three reads in a row have failed.
    world = World(replies=[Reply(10.0, CLICK)], check_s=1.25, check_errors=_failing(1, 2, 3))
    sim = _episode(_wall(3.0), world)
    assert _asked(sim) == [0.0, 1.25, 2.5]
    assert _fired(sim, "deadline") == []
    assert sim.outcomes == []
    assert _ended(sim) == ("invalid_run", "invalid", 3.75, 3.75)
    assert sim.termination.invalid == Invalidated(
        "environment", "terminal_state_check_failed", "state read 3 failed"
    )


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
    assert _asked(sim) == [0.0, 1.0]
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


def test_a_capture_that_fails_after_the_deadline_still_invalidates():
    # The loop starts at 2.5 s and the capture runs until 3.5 s, then fails. The deadline is handled
    # first and ends the episode; the failure still makes the run invalid, as today's loop judges
    # every failed capture (runner.py _decide).
    world = World(replies=[], start_s=2.5, capture_s=1.0, capture_errors={1: "screenshot failed"})
    sim = _episode(_wall(3.0), world)
    assert sim.captures == [(2, 3.5)]
    assert sim.entries("InferenceStarted") == []
    assert sim.kinds()[-5:] == [
        "EpisodeEnded",
        "FrameCaptured",
        "RunInvalidated",
        "SnapshotFrozen",
        "Termination",
    ]
    assert _ended(sim) == ("invalid_run", "deadline", 3.0, 3.5)
    assert sim.termination.invalid == Invalidated(
        "environment", "observation_capture_failed", "screenshot failed"
    )


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


FREEZE_ON_THE_SECOND = dataclasses.replace(RunnerTimingConfig(), observation_freeze_n=2)


@pytest.mark.parametrize(
    ("config", "world", "cause", "ended_at"),
    [
        # T = 4.5: the second frame is captured from 4 s to 4.75 s, and its answer, filed then,
        # comes after the deadline's.
        (
            _wall(4.5, timing=FREEZE_ON_THE_SECOND),
            World(replies=[Reply(2.4)] * 6, frozen_from=0, start_s=0.6, capture_s=0.75),
            "deadline",
            4.5,
        ),
        # T = 5 at 2 fps: the second frame is captured from 4 s to 4.6 s, across the boundary at
        # 4.5 s, whose check runs first, until 5.2 s; the answer filed at 4.6 s comes first.
        (
            _wall(5.0, env_fps=2.0, timing=FREEZE_ON_THE_SECOND),
            World(replies=[Reply(2.9)] * 6, frozen_from=0, capture_s=0.6, check_seconds={10: 0.6}),
            "invalid",
            pytest.approx(5.2),
        ),
    ],
    ids=["answer-filed-after-the-deadline", "answer-filed-before-it"],
)
def test_a_frame_that_freezes_the_stream_invalidates_the_run_also_past_the_deadline(
    config, world, cause, ended_at
):
    """A frame captured under way at the deadline is judged like any other: today's loop judges
    every frame it captured (runner.py _decide, _accept_observation), as the kernel judges every
    failed capture. Where its answer is filed only changes which end comes first."""
    sim = _episode(config, world)
    assert len(sim.captures) == 2
    termination = sim.termination
    assert (termination.stop_reason, termination.cause) == ("invalid_run", cause)
    assert termination.ended_at == ended_at
    assert termination.invalid.reason == "observation_stream_frozen"
    _assert_nothing_after_the_end(sim)
