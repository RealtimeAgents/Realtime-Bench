"""The scenarios phase2.md lists under 怎么验证, run on the discrete-event simulation of the turn
policy, on both clocks where they apply.

Every expectation is worked out from the rules, not read off a run: today's loop where phase 2
keeps it (legacy_loop.py _run_stream_legacy) and phase2.md where it changes it (事件内核; 时间和结束判定).
The first end signal ends the episode at once: it freezes the snapshot, writes off and interrupts
the generation in flight, and stops every timer; the decision in hand is settled with the skip
order and reported; a late result is discarded. At one instant an end beats a completion, which
beats a start. The state is read once more after every executed action and otherwise not twice
at one reading: on the token clock after each step, the loop top and the check before landing
reading only a tick not read yet; on the wall clock once a frame by the check due every
1/env_fps s, during generations too, and by the loop top, and before landing at a new reading.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import asdict

import pytest

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.evaluation.kernel.events import (
    AbortGeneration,
    AcceptResult,
    AdvanceClock,
    AgentRecord,
    Capture,
    CheckEnd,
    Commit,
    EndChecked,
    Execute,
    Finish,
    FreezeSnapshot,
    GenerationFinished,
    GenerationResult,
    Invalidated,
    Record,
    ReleaseInputs,
    ReleaseKeys,
    RenderFrame,
    SetTimer,
    StartGeneration,
    Termination,
)
from realtime_framework.evaluation.kernel.turns import TurnConfig
from tests.kernel.simulation import Reply, Simulation, World

CLICK = Action.click(10, 20)
# What follows the end of the episode when a generation was in flight at the deadline.
ABANDONED_AT_THE_DEADLINE = [
    "EpisodeEnded",
    "GenerationAbandoned",
    "GenerationFinished",
    "ResultDiscarded",
    "SnapshotFrozen",
    "Termination",
]


def _token(duration_s: float, tokens_per_frame: int | None = 100, **fields) -> TurnConfig:
    return TurnConfig("token", duration_s, tokens_per_frame=tokens_per_frame, **fields)


def _wall(duration_s: float, **fields) -> TurnConfig:
    return TurnConfig("wall", duration_s, **fields)


def _run(config: TurnConfig, world: World) -> Simulation:
    sim = Simulation(config, world).run()
    _check_invariants(sim)
    return sim


def _is_record(command, kind: str) -> bool:
    return isinstance(command, Record) and command.entry.kind == kind


def _semantics(sim: Simulation) -> list[float]:
    """The semantic time of every end check."""
    return [semantic for semantic, _ in sim.checks]


def _from(sim: Simulation, kind: str) -> list[str]:
    """The record kinds from the first ``kind`` on."""
    kinds = sim.kinds()
    return kinds[kinds.index(kind) :]


def _past_the_deadline(sim: Simulation) -> list:
    """Wall clock: the captures, generations, executions and end checks run at or after the
    deadline."""
    if sim.token:
        return []
    world = (Capture, StartGeneration, Execute, CheckEnd)
    return [
        command
        for command, at in zip(sim.commands, sim.command_times, strict=True)
        if isinstance(command, world) and at >= sim.config.duration_s
    ]


def _reads_by_what_they_read(sim: Simulation) -> Counter:
    """How many times the state was read over each thing a read reads, besides the reads right
    after an action: on the token clock a tick, which a tap's reads also read; on the wall clock
    the environment frame, but for the check before landing its reading."""
    counts = Counter()
    for (semantic, _), after, before in zip(
        sim.checks, sim.after_action, sim.before_landing, strict=True
    ):
        if after:
            continue
        if sim.token:
            counts["tick", semantic] += 1
        elif before:
            counts["reading before landing", semantic] += 1
        else:
            counts["frame", math.floor(semantic * sim.config.env_fps + 1e-9)] += 1
    counts.update(("tick", tick) for tick in sim.tap_checks)
    return counts


def _check_invariants(sim: Simulation) -> None:
    """What every episode keeps, whatever ends it."""
    records, commands = sim.records, sim.commands
    assert [entry.seq for entry in records] == list(range(1, len(records) + 1))
    # The termination record is written last; then inputs are released and the episode finishes.
    assert records[-1].kind == "Termination"
    assert records[-1].data == asdict(sim.termination)
    assert commands[-2:] == [ReleaseInputs(), Finish(sim.termination)]
    assert sim.kinds().count("EpisodeEnded") == 1
    ended = next(i for i, command in enumerate(commands) if _is_record(command, "EpisodeEnded"))
    assert isinstance(commands[ended + 1], FreezeSnapshot)
    # Once the episode is over nothing lands, no clock moves and no end check is asked.
    banned = (Capture, StartGeneration, AcceptResult, Execute, AdvanceClock, CheckEnd)
    assert [command for command in commands[ended:] if isinstance(command, banned)] == []
    # Nor from the wall deadline on, though an answer filed before it may be handled after it.
    assert _past_the_deadline(sim) == []
    # The state is read at most once over what a read reads, and once more after every action
    # that did not end the episode.
    assert set(_reads_by_what_they_read(sim).values()) <= {1}
    landed = [
        entry
        for entry in sim.entries("ActionResolved")
        if entry.data["status"] in {"applied", "wait"} and not entry.data["terminal_after_action"]
    ]
    assert sum(sim.after_action) >= len(landed)
    # One generation at a time, and no outcome reported while one is in flight.
    in_flight = None
    for command in commands:
        if isinstance(command, StartGeneration):
            assert in_flight is None
            in_flight = command.generation
        elif isinstance(command, AcceptResult | AbortGeneration):
            if command.generation == in_flight:
                in_flight = None
        elif isinstance(command, Commit):
            assert in_flight is None
    # An interruption is written off before it is sent.
    for index, command in enumerate(commands):
        if isinstance(command, AbortGeneration):
            assert _is_record(commands[index - 1], "GenerationAbandoned")
            assert commands[index - 1].entry.data["generation"] == command.generation
    # A generation whose result came back is settled by exactly one accept or discard.
    started = [command.generation for command in commands if isinstance(command, StartGeneration)]
    arrived = {entry.data.generation for entry in sim.entries("GenerationFinished")}
    for generation in started:
        settled = sim.accepted.count(generation) + sim.discarded.count(generation)
        assert settled == int(generation in arrived)
    # A stale result never lands: its decision proposes and executes nothing.
    decision_of = {
        entry.data["generation"]: entry.data["decision"]
        for entry in sim.entries("InferenceStarted")
    }
    proposed = [entry.data["decision"] for entry in sim.entries("ActionProposed")]
    resolved = [entry.data["decision"] for entry in sim.entries("ActionResolved")]
    executed = [decision for decision, _, _ in sim.executed]
    for generation in sim.discarded:
        assert generation not in sim.accepted
        assert decision_of[generation] not in proposed
        assert decision_of[generation] not in executed
    # Every proposed action is resolved exactly once; a decision is reported at most once.
    assert all(resolved.count(decision) == 1 for decision in proposed)
    assert len(set(resolved)) == len(resolved)
    reported = [outcome.decision for outcome in sim.outcomes]
    assert len(set(reported)) == len(reported)
    # A generation the end interrupted leaves its decision unresolved and unreported.
    for entry in sim.entries("GenerationAbandoned"):
        if entry.data["reason"] != "per_move_cap":
            decision = decision_of[entry.data["generation"]]
            assert decision not in resolved
            assert decision not in reported


# ---- 宽限期 --------------------------------------------------------------------------------------


def test_a_reply_back_while_the_snapshot_is_taken_is_discarded():
    """宽限期内返回: the reply comes back 1 s after the deadline, inside today's 2 s grace. Today it
    would still reach the landing check and be not_applied_time_limit; now there is no grace: the
    deadline ends the episode at once, so the reply, back while the 2 s snapshot is being taken,
    is discarded and nothing is reported."""
    sim = _run(_wall(3.0), World(replies=[Reply(latency=4.0, action=CLICK)], snapshot_s=2.0))
    # The loop top and the checks every second; the deadline beats the check due at 3.
    assert sim.checks == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    assert _from(sim, "EpisodeEnded") == ABANDONED_AT_THE_DEADLINE
    assert sim.entries("EpisodeEnded")[0].data == {"cause": "deadline", "reason": None, "at": 3.0}
    assert sim.entries("GenerationAbandoned")[0].data == {"generation": 1, "reason": "deadline"}
    # The shell interrupts the call once the snapshot is in: 3 s + 2 s.
    assert sim.snapshots == [(5.0, 5.0)]
    assert sim.aborted == [(1, 5.0)]
    assert (sim.accepted, sim.discarded, sim.executed, sim.outcomes) == ([], [1], [], [])
    assert "ActionProposed" not in sim.kinds()
    assert sim.termination == Termination("time_limit", "deadline", 3.0, 5.0, 3.0, 5.0, None)


def test_a_reply_after_the_deadline_is_not_waited_for():
    """宽限期过了才返回: the reply would come back 5 s after the deadline. Today a call that slow
    is abandoned at the deadline plus the 2 s grace and reported skipped_timeout (runner.py
    _move_timeout); now there is no grace, so the episode ends at the deadline, interrupts the
    call and finishes without waiting for it. (A reply inside the old grace is the test before.)"""
    sim = _run(_wall(3.0), World(replies=[Reply(latency=8.0, action=CLICK)]))
    assert sim.checks == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    assert sim.aborted == [(1, 3.0)]
    assert sim.snapshots == [(3.0, 3.0)]
    assert (sim.accepted, sim.discarded, sim.executed, sim.outcomes) == ([], [], [], [])
    assert "GenerationFinished" not in sim.kinds()
    assert sim.ops == 3.0
    assert sim.termination == Termination("time_limit", "deadline", 3.0, 3.0, 3.0, 3.0, None)


# ---- 正好在截止时落地 ------------------------------------------------------------------------------


def test_a_wall_result_exactly_at_the_deadline_is_discarded():
    """正好在截止时落地 (wall): the result and the deadline fall due at 3 s. The deadline is a
    control entry and is handled first, so the generation is already written off when its result
    is handled, and the result is discarded."""
    sim = _run(_wall(3.0), World(replies=[Reply(latency=3.0, action=CLICK)]))
    assert _from(sim, "EpisodeEnded") == ABANDONED_AT_THE_DEADLINE
    fired = sim.records[sim.kinds().index("EpisodeEnded") - 1]
    assert (fired.kind, fired.data["kind"], fired.semantic) == ("TimerFired", "deadline", 3.0)
    (finished,) = sim.entries("GenerationFinished")
    assert finished.semantic == 3.0
    assert (sim.accepted, sim.discarded, sim.executed, sim.outcomes) == ([], [1], [], [])
    assert sim.termination == Termination("time_limit", "deadline", 3.0, 3.0, 3.0, 3.0, None)


def test_a_token_action_at_the_horizon_is_not_applied():
    """正好在截止时落地 (token): 200 tokens at B = 100 from tick 0 put the action at tau = 2 = T.
    The clock reaches the horizon before execution, so the action is not applied, and the horizon
    ends the episode at the next loop top (legacy_loop.py _land_action, _run_stream_legacy)."""
    sim = _run(_token(2.0), World(replies=[Reply(latency=1.0, action=CLICK, output_tokens=200)]))
    starts = [command for command in sim.commands if isinstance(command, StartGeneration)]
    assert [start.remaining_clock_tokens for start in starts] == [200]  # floor(2 * 100)
    assert sim.advances == [1.0, 1.0]
    # The loop top and each step; tick 2 is not read again before execution or at the loop top
    # that finds the horizon.
    assert sim.checks == [(0, 0.0), (1, 1.0), (2, 1.0)]
    assert sim.executed == []
    (outcome,) = sim.outcomes
    assert (outcome.status, outcome.action_clock_position) == ("not_applied_tick_limit", 2.0)
    assert (outcome.next_observation_tick, outcome.dropped_intermediate_frames) == (2, 0)
    assert (outcome.action_output_tokens, outcome.post_horizon_output_tokens) == (200, 0)
    assert sim.termination == Termination("tick_limit", "horizon", 2.0, 2.0, 1.0, 1.0, None)


# ---- wait 跨过截止 --------------------------------------------------------------------------------


def test_a_wall_wait_across_the_deadline_ends_at_the_deadline():
    """wait 跨过截止 (wall): a 5 s WAIT starts at 0.5 s with the deadline at 3 s. The WAIT is a
    kernel timer the end interrupts: it ran 2.5 s, as today's sleep(min(duration, remaining)), and
    is reported as wait with the episode over (legacy_loop.py _land_action)."""
    sim = _run(_wall(3.0), World(replies=[Reply(latency=0.5, action=Action.wait(5.0))]))
    (resolved,) = sim.entries("ActionResolved")
    assert resolved.semantic == 3.0
    assert resolved.data["execution_result"] == {
        "duration_simulated_seconds": 2.5,
        "requested_duration_simulated_seconds": 5.0,
        "truncated_by_episode_horizon": False,
        "truncated_by_terminal_condition": False,
    }
    (outcome,) = sim.outcomes
    assert (outcome.status, outcome.terminal_after_action) == ("wait", True)
    assert (outcome.action_clock_position, outcome.frame_age_s) == (0.5, 0.5)
    # A WAIT is no environment call; the state is read before it, at 0.5 s, as before any action
    # (legacy_loop.py _land_action), and no end check is asked after the deadline.
    assert sim.executed == []
    assert sim.checks == [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0)]
    assert sim.termination == Termination("time_limit", "deadline", 3.0, 3.0, 3.0, 3.0, None)


def test_a_wall_wait_ends_at_the_deadline_however_late_the_deadline_is_handled():
    """Checks take 0.75 s. A 3 s WAIT starts at 2.5 s, after the check before it, with the deadline
    at 3.5 s; the check due at 3 s is asked at 3.25 s and runs until 4 s, and only then is the
    deadline handled. The WAIT ran until the deadline, 1 s, as today's sleep(min(duration,
    remaining)), not until the deadline was handled."""
    world = World(replies=[Reply(latency=0.75, action=Action.wait(3.0)), Reply(9.0)], check_s=0.75)
    sim = _run(_wall(3.5), world)
    (resolved,) = sim.entries("ActionResolved")
    assert sim.checks == [(0.0, 0.0), (1.0, 1.0), (1.75, 1.75), (2.5, 2.5), (3.25, 3.25)]
    assert resolved.semantic == 4.0
    assert resolved.data["action_clock_position"] == 1.75
    assert resolved.data["execution_result"] == {
        "duration_simulated_seconds": 1.0,
        "requested_duration_simulated_seconds": 3.0,
        "truncated_by_episode_horizon": False,
        "truncated_by_terminal_condition": False,
    }
    assert sim.termination == Termination("time_limit", "deadline", 3.5, 4.0, 4.0, 4.0, None)


@pytest.mark.parametrize(
    ("duration_s", "ended"),
    [(3.5, ("time_limit", 3.5, True)), (10.0, ("time_limit", 10.0, False))],
    ids=["deadline-after-its-time", "no-deadline-near"],
)
def test_a_wall_wait_lasts_its_own_time_however_late_its_end_is_handled(duration_s, ended):
    """A 2.65 s WAIT from 0.75 s is due at 3.4 s, but the check due at 3 s, the fifth, runs until
    3.75 s, and only then is the wait's end handled. It lasted its own time, 2.65 s, not until it
    was handled, though that is past a deadline at 3.5 s."""
    world = World(
        replies=[Reply(latency=0.75, action=Action.wait(2.65)), Reply(30.0)],
        check_seconds={5: 0.75},
    )
    sim = _run(_wall(duration_s), world)
    assert sim.checks[:5] == [(0.0, 0.0), (0.75, 0.75), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    (resolved,) = sim.entries("ActionResolved")
    assert resolved.data["execution_result"] == {
        "duration_simulated_seconds": 2.65,
        "requested_duration_simulated_seconds": 2.65,
        "truncated_by_episode_horizon": False,
        "truncated_by_terminal_condition": False,
    }
    stop_reason, ended_at, terminal = ended
    assert (resolved.data["status"], resolved.data["terminal_after_action"]) == ("wait", terminal)
    assert (sim.termination.stop_reason, sim.termination.ended_at) == (stop_reason, ended_at)


def test_a_wall_wait_the_tasks_end_cuts_short_lasts_until_the_end_is_dated():
    """A 5 s WAIT from 1.25 s (the reply came back at 1 s, on the frame boundary, and frame 1 was
    read first). The task ends from 2.5 s; the check asked at 3 s answers so at 3.25 s, and the
    end is dated by its asking. The WAIT lasted until then, 1.75 s, cut by the task's end."""
    world = World(
        replies=[Reply(latency=0.75, action=Action.wait(5.0)), Reply(9.0)],
        check_s=0.25,
        ends_at=2.5,
        end_reason="target_reached",
    )
    sim = _run(_wall(10.0), world)
    assert sim.checks == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    (resolved,) = sim.entries("ActionResolved")
    assert resolved.semantic == 3.25
    assert resolved.data["execution_result"] == {
        "duration_simulated_seconds": 1.75,
        "requested_duration_simulated_seconds": 5.0,
        "truncated_by_episode_horizon": False,
        "truncated_by_terminal_condition": True,
    }
    termination = sim.termination
    assert (termination.stop_reason, termination.ended_at) == ("target_reached", 3.0)


def test_a_token_wait_across_the_horizon_ends_at_the_horizon():
    """wait 跨过截止 (token): a 5 s WAIT lands at tick 1 with T = 3. It moves the clock only to the
    horizon, one tick per step with an end check after each, and records that the horizon cut it
    (sandbox/actions.py GuestEnvironment.execute)."""
    wait = Action.wait(5.0)
    sim = _run(_token(3.0), World(replies=[Reply(latency=0.5, action=wait, output_tokens=100)]))
    assert sim.advances == [1.0, 1.0, 1.0]
    # Loop top; approach to 1; wait steps to 2 and 3; after execution. Neither tick 1 before the
    # wait nor tick 3 at the loop top is read again.
    assert _semantics(sim) == [0, 1, 2, 3, 3]
    (resolved,) = sim.entries("ActionResolved")
    assert (resolved.data["status"], resolved.data["terminal_after_action"]) == ("wait", False)
    assert resolved.data["execution_result"] == {
        "duration_simulated_seconds": 2.0,
        "requested_duration_simulated_seconds": 5.0,
        "truncated_by_episode_horizon": True,
        "truncated_by_terminal_condition": False,
    }
    assert sim.executed == []
    assert sim.termination == Termination("tick_limit", "horizon", 3.0, 3.0, 0.5, 0.5, None)


# ---- 按键在两帧之间到期 ----------------------------------------------------------------------------


def test_a_token_hold_expiring_between_frames_splits_the_step():
    """按键在两帧之间到期 (token): a 0.5 s hold lands at tick 1 (100 tokens at B = 100). The next
    decision moves the clock from 1 to 2: the step is cut at the lease end, 1.5, where the key is
    released and the end checked, before the rest of the step."""
    hold = Action.hold_key("a", 0.5)
    replies = [
        Reply(latency=0.5, action=hold, output_tokens=100),
        Reply(latency=0.5, action=CLICK, output_tokens=100),
    ]
    sim = _run(_token(2.0), World(replies=replies))
    assert sim.executed == [(0, 1, hold)]
    assert sim.advances == [1.0, 0.5, 0.5]
    assert sim.key_events == [(1.5, "up", ("a",))]
    second = [i for i, c in enumerate(sim.commands) if isinstance(c, StartGeneration)][1]
    moves = [
        type(command).__name__
        for command in sim.commands[second:]
        if isinstance(command, AdvanceClock | ReleaseKeys | CheckEnd)
    ]
    # Step to the lease end, release, check; the rest of the step, check. Tick 2 is not read again
    # before execution or at the loop top.
    assert moves == [
        "AdvanceClock",
        "ReleaseKeys",
        "CheckEnd",
        "AdvanceClock",
        "CheckEnd",
    ]
    assert _semantics(sim) == [0, 1, 1, 1.5, 2]
    # The hold lands on the boundary and renders it; the second action time is the horizon.
    assert sim.renders == [1]
    assert [outcome.status for outcome in sim.outcomes] == ["applied", "not_applied_tick_limit"]
    assert sim.termination == Termination("tick_limit", "horizon", 2.0, 2.0, 1.0, 1.0, None)


def test_a_wall_hold_is_released_by_the_environments_timer_between_frames():
    """按键在两帧之间到期 (wall): a 1 s hold pressed at 0.25 s is due up at 1.25 s, between the frames
    at 1 s and 2 s. The press carries its release time to the environment, whose own timer lifts
    the key on time though the shell is busy reporting the outcome from 1 s to 1.5 s; the kernel
    hears nothing of a release that worked."""
    hold = Action.hold_key("a", 1.0)
    replies = [Reply(latency=0.25, action=hold), Reply(latency=5.0, action=CLICK)]
    sim = _run(_wall(2.0), World(replies=replies, commit_s=0.5))
    assert sim.executed == [(0, 0.25, hold)]
    (scheduled,) = sim.entries("KeyReleaseScheduled")
    assert scheduled.data == {"decision": 0, "at": 1.25}  # pressed at 0.25 s, held 1 s
    assert Execute(0, hold, 1.25) in sim.commands
    assert sim.key_events == [(1.25, "up", ("a",))]
    assert "KeysReleased" not in sim.kinds()
    (outcome,) = sim.outcomes
    assert (outcome.status, outcome.next_observation_tick, outcome.frame_age_s) == (
        "applied",
        1,
        0.25,
    )
    # The next frame is captured once the report is done.
    assert sim.captures == [(0, 0.0), (1, 1.5)]
    assert sim.termination == Termination("time_limit", "deadline", 2.0, 2.0, 2.0, 2.0, None)


# ---- 重试跨过截止 ----------------------------------------------------------------------------------


def test_the_records_of_a_retry_across_the_deadline_are_numbered_and_its_result_discarded():
    """重试跨过截止 (wall): the call's first attempt fails and its retry runs past the deadline.
    The retry runs in the agent, which only posts the records of its attempts; the kernel numbers
    them in handling order before and after the end, writes the call off at the deadline, discards
    its result (back during the 2 s snapshot), and hands nothing posted after Finish to the
    policy."""
    records = (
        (0.5, "ModelRequestSent"),
        (1.5, "ModelAttemptFinished"),
        (2.5, "ModelRequestSent"),
        (4.0, "ModelAttemptFinished"),
    )
    reply = Reply(latency=4.5, action=CLICK, records=records)
    sim = _run(_wall(3.0), World(replies=[reply], snapshot_s=2.0))
    shown = {"ModelRequestSent", "ModelAttemptFinished", *ABANDONED_AT_THE_DEADLINE}
    assert [(entry.kind, entry.semantic) for entry in sim.records if entry.kind in shown] == [
        ("ModelRequestSent", 0.5),
        ("ModelAttemptFinished", 1.5),
        ("ModelRequestSent", 2.5),
        ("EpisodeEnded", 3.0),
        ("GenerationAbandoned", 3.0),
        # Posted at 4 s and 4.5 s while the snapshot ran; handled once it was in at 5 s.
        ("ModelAttemptFinished", 5.0),
        ("GenerationFinished", 5.0),
        ("ResultDiscarded", 5.0),
        ("SnapshotFrozen", 5.0),
        ("Termination", 5.0),
    ]
    agent = [entry for entry in sim.records if entry.kind.startswith("Model")]
    assert [entry.data.generation for entry in agent] == [1, 1, 1, 1]
    assert (sim.accepted, sim.discarded, sim.executed) == ([], [1], [])
    assert sim.termination == Termination("time_limit", "deadline", 3.0, 5.0, 3.0, 5.0, None)

    kernel, termination = sim.kernel, sim.termination
    calls = []
    for name in ("on_event", "on_result", "on_timer", "on_end"):
        setattr(kernel.policy, name, lambda *args, name=name: calls.append(name))
    kernel.post(AgentRecord("ModelAttemptFinished", {}, 1))
    kernel.post(GenerationFinished(1, GenerationResult(action=CLICK, output_tokens=5)))
    kernel.post(EndChecked(99, ended=True))
    kernel.post(Invalidated("environment", "late_failure"))
    later = []
    while (commands := kernel.step(6.0)) is not None:
        later += commands
    assert [command for command in later if not isinstance(command, Record)] == []
    assert calls == []
    assert kernel.termination is termination
    assert kernel.invalid is None


# ---- 画面冻结 ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "reply", "ended_at", "ops_at"),
    [
        # Decisions land at ticks 1, 2 and 2 (the third observes the frozen tick 1); the fourth
        # frame is captured at tick 2 after three 0.5 s calls.
        (_token(10.0), Reply(latency=0.5, action=CLICK, output_tokens=100), 2.0, 1.5),
        # Frames at 0, 1, 2 and 3 s.
        (_wall(10.0), Reply(latency=0.5, action=CLICK), 3.0, 3.0),
    ],
    ids=["token", "wall"],
)
def test_a_frozen_stream_invalidates_the_run(config, reply, ended_at, ops_at):
    """画面冻结: from the second capture on the producer repeats one frame. The third identical
    frame, with operational time moved on, invalidates the run before a fourth generation starts
    (legacy_loop.py _accept_observation; observation_freeze_n = 3)."""
    sim = _run(config, World(replies=[reply] * 3, frozen_from=1))
    assert len(sim.captures) == 4
    assert len(sim.entries("InferenceStarted")) == 3
    assert [outcome.status for outcome in sim.outcomes] == ["applied"] * 3
    (invalidated,) = sim.entries("RunInvalidated")
    assert (invalidated.data["category"], invalidated.data["reason"]) == (
        "environment",
        "observation_stream_frozen",
    )
    termination = sim.termination
    assert (termination.stop_reason, termination.cause) == ("invalid_run", "invalid")
    assert termination.invalid.reason == "observation_stream_frozen"
    assert (termination.ended_at, termination.snapshot_at) == (ended_at, ended_at)
    assert (termination.ops_ended_at, termination.ops_snapshot_at) == (ops_at, ops_at)


# ---- 一次生成跨过好几帧, 恰好落在帧边界 ---------------------------------------------------------------


def test_a_generation_spanning_several_frames_drops_them():
    """一次生成跨过好几帧 (token): 350 tokens at B = 100 from tick 0 put the action at 3.5. The clock
    moves 1, 1, 1, 0.5 with an end check after each step, the action lands at 3.5, the next
    observation is at ceil(3.5) = 4, and of the 4 frames crossed 3 were dropped
    (legacy_loop.py _apply)."""
    replies = [
        Reply(latency=1.0, action=CLICK, output_tokens=350),
        Reply(latency=1.0, action=CLICK, output_tokens=100),
    ]
    sim = _run(_token(5.0), World(replies=replies))
    assert sim.advances == [1.0, 1.0, 1.0, 0.5, 0.5, 1.0]
    # Decision 0: loop top, 4 approach steps, after execution, the step to 4. Decision 1: the step
    # to 5. No loop top or check before execution reads a tick read already.
    assert _semantics(sim) == [0, 1, 2, 3, 3.5, 3.5, 4, 5]
    assert sim.executed == [(0, 3.5, CLICK)]
    first = sim.outcomes[0]
    assert (first.action_clock_position, first.next_observation_tick) == (3.5, 4)
    assert (first.dropped_intermediate_frames, first.post_action_rendered) == (3, False)
    assert (first.frame_age_s, first.token_delay_frames) == (3.5, 3.5)
    boundary = [
        entry.data
        for entry in sim.entries("ClockAdvanced")
        if entry.data["phase"] == "action_to_observation_boundary"
    ]
    assert [
        (
            data["from_tick"],
            data["requested_tick"],
            data["to_tick"],
            data["dropped_intermediate_frames"],
        )
        for data in boundary
    ] == [(3.5, 4, 4, 3)]
    assert [tick for tick, _ in sim.captures] == [0, 4]
    assert sim.kernel.policy.dropped_frames == 3
    assert sim.outcomes[1].status == "not_applied_tick_limit"
    assert sim.termination == Termination("tick_limit", "horizon", 5.0, 5.0, 2.0, 2.0, None)


def test_a_wall_generation_spanning_several_frames_drops_them():
    """一次生成跨过好几帧 (wall): a 2.5 s call from the frame at 0 s lands at 2.5 s, tick 2.5, with
    the end checked at 1 s and 2 s meanwhile; the next observation is at ceil(2.5) = 3 s, and of
    the 3 frames crossed 2 were dropped. The wall clock keeps no dropped-frame total
    (legacy_loop.py _apply)."""
    sim = _run(_wall(5.0), World(replies=[Reply(latency=2.5, action=CLICK), Reply(latency=10.0)]))
    assert sim.executed == [(0, 2.5, CLICK)]
    # Loop top; every second; before and after execution; at 3 s the periodic check (set before
    # the move), and the loop top, at that reading, reads no more; 4 s; the deadline beats the
    # check due at 5 s.
    assert _semantics(sim) == [0, 1, 2, 2.5, 2.5, 3, 4]
    (outcome,) = sim.outcomes
    assert (outcome.action_clock_position, outcome.next_observation_tick) == (2.5, 3)
    assert (outcome.dropped_intermediate_frames, outcome.frame_age_s) == (2, 2.5)
    assert [
        (data["from_tick"], data["requested_tick"], data["to_tick"])
        for data in (entry.data for entry in sim.entries("ClockAdvanced"))
        if data["phase"] == "action_to_observation_boundary"
    ] == [(2, 3, 3)]
    assert sim.captures == [(0, 0.0), (3, 3.0)]
    assert sim.kernel.policy.dropped_frames == 0
    assert sim.termination == Termination("time_limit", "deadline", 5.0, 5.0, 5.0, 5.0, None)


def test_an_action_exactly_on_a_frame_boundary_renders_the_frame():
    """恰好落在帧边界 (token): 100 tokens at B = 100 put the click on tick 1, which is also the next
    observation; the move there is empty, so the frame is rendered at that tick before the next
    capture. A key tap on the next boundary moves the clock 1/60 tick itself, so the clock goes on
    to ceil(2 + 1/60) = 3 and nothing is rendered (legacy_loop.py _apply)."""
    tap = Action(ActionType.KEY, {"key": "a"})
    replies = [
        Reply(latency=1.0, action=CLICK, output_tokens=100),
        Reply(latency=1.0, action=tap, output_tokens=100),
    ]
    sim = _run(_token(3.0), World(replies=replies))
    assert sim.renders == [1]
    render = next(i for i, c in enumerate(sim.commands) if isinstance(c, RenderFrame))
    second_capture = [i for i, c in enumerate(sim.commands) if isinstance(c, Capture)][1]
    assert render < second_capture
    first, second = sim.outcomes
    assert (first.status, first.post_action_rendered, first.next_observation_tick) == (
        "applied",
        True,
        1,
    )
    assert first.dropped_intermediate_frames == 0
    assert (second.status, second.post_action_rendered, second.next_observation_tick) == (
        "applied",
        False,
        3,
    )
    assert second.dropped_intermediate_frames == 1  # frames 2 and 3 crossed from tick 1
    # Decision 0 only moves to its action; decision 1 moves to its action, then past the tap.
    assert sim.advances == pytest.approx([1.0, 1.0, 1 - 1 / 60])
    tap_resolved = sim.entries("ActionResolved")[1]
    assert tap_resolved.data["environment_tick"] == pytest.approx(2 + 1 / 60)
    assert sim.termination == Termination("tick_limit", "horizon", 3.0, 3.0, 2.0, 2.0, None)


def test_a_wall_action_exactly_on_a_frame_boundary_lands_before_the_check():
    """恰好落在帧边界 (wall): a 2 s call lands exactly on the frame at 2 s. The result, a completion,
    is handled before the end check due then, a start; the next observation is that frame, so the
    move to it is empty, and the wall clock renders nothing extra (the render barrier in
    legacy_loop.py _apply is token only). Frame 1 was crossed during the call and dropped."""
    sim = _run(_wall(5.0), World(replies=[Reply(latency=2.0, action=CLICK), Reply(latency=10.0)]))
    assert sim.executed == [(0, 2.0, CLICK)]
    (outcome,) = sim.outcomes
    assert (outcome.status, outcome.action_clock_position, outcome.next_observation_tick) == (
        "applied",
        2.0,
        2,
    )
    assert (outcome.dropped_intermediate_frames, outcome.post_action_rendered) == (1, False)
    assert sim.renders == []
    (finished,) = sim.entries("GenerationFinished")
    (resolved,) = sim.entries("ActionResolved")
    (periodic,) = [
        entry
        for entry in sim.entries("TimerFired")
        if entry.data["kind"] == "end_check" and entry.data["at"] == 2.0
    ]
    assert finished.seq < resolved.seq < periodic.seq
    # Before and after execution at 2 s, frame 2 not having been read; the periodic check due then
    # and the loop top read it no more; then 3 s and 4 s.
    assert _semantics(sim) == [0, 1, 2, 2, 3, 4]
    assert sim.captures == [(0, 0.0), (2, 2.0)]
    assert sim.termination == Termination("time_limit", "deadline", 5.0, 5.0, 5.0, 5.0, None)


# ---- 总帧数截断 ----------------------------------------------------------------------------------


def test_output_past_the_allowance_is_clamped_and_not_applied():
    """总帧数截断 (token): with T = 2 and B = 100 the second decision observes tick 1 and has floor((2 -
    1) * 100) = 100 tokens left. Its 180 tokens are clamped to 100, the other 80 are billed past the
    horizon, the action time is T, so the action is not applied and the episode ends at the horizon
    (legacy_loop.py _decide, _land_action; episode_records.py account_tokens)."""
    replies = [
        Reply(latency=1.0, action=CLICK, output_tokens=100),
        Reply(latency=1.0, action=CLICK, output_tokens=180),
    ]
    sim = _run(_token(2.0), World(replies=replies))
    starts = [command for command in sim.commands if isinstance(command, StartGeneration)]
    assert [start.remaining_clock_tokens for start in starts] == [200, 100]
    assert [entry.data["remaining_clock_tokens"] for entry in sim.entries("InferenceStarted")] == [
        200,
        100,
    ]
    assert sim.entries("ActionProposed")[1].data["action_output_tokens"] == 100
    second = sim.outcomes[1]
    assert (second.status, second.action_clock_position) == ("not_applied_tick_limit", 2.0)
    assert (second.action_output_tokens, second.post_horizon_output_tokens) == (100, 80)
    assert second.token_delay_frames == 1.0
    assert sim.executed == [(0, 1, CLICK)]
    assert sim.termination == Termination("tick_limit", "horizon", 2.0, 2.0, 2.0, 2.0, None)


def test_with_no_allowance_left_the_clock_moves_to_the_horizon():
    """总帧数截断 (token): with T = 2.5 and B = 1 the frame at tick 2 has floor(0.5 * 1) = 0 tokens
    left. No generation starts; the clock moves to T with an end check and the episode ends there
    (legacy_loop.py _decide)."""
    sim = _run(
        _token(2.5, tokens_per_frame=1),
        World(replies=[Reply(latency=1.0, action=CLICK, output_tokens=2)]),
    )
    assert [tick for tick, _ in sim.captures] == [0, 2]
    assert len([command for command in sim.commands if isinstance(command, StartGeneration)]) == 1
    assert sim.advances == [1.0, 1.0, 0.5]
    assert _semantics(sim) == [0, 1, 2, 2, 2.5]
    assert sim.termination == Termination("tick_limit", "horizon", 2.5, 2.5, 1.0, 1.0, None)


# ---- 暂停模式 ------------------------------------------------------------------------------------


def _paused(latencies: tuple[float, float, float]) -> Simulation:
    replies = [
        Reply(latency=latency, action=CLICK, output_tokens=tokens)
        for latency, tokens in zip(latencies, (None, 10, 100_000), strict=True)
    ]
    return _run(_token(3.0, tokens_per_frame=None), World(replies=replies))


def test_paused_mode_lands_on_the_observed_tick_and_advances_one_frame():
    """暂停模式: B unlimited. Each action lands on its observed tick whatever its output length, missing
    usage is no failure, and the clock then moves exactly one frame (legacy_loop.py _apply)."""
    sim = _paused((0.1, 50.0, 250.0))
    assert [tick for tick, _ in sim.captures] == [0, 1, 2]
    assert [semantic for _, semantic, _ in sim.executed] == [0, 1, 2]
    assert sim.advances == [1.0, 1.0, 1.0]
    assert [
        (
            outcome.action_clock_position,
            outcome.frame_age_s,
            outcome.token_delay_frames,
            outcome.next_observation_tick,
            outcome.dropped_intermediate_frames,
            outcome.post_action_rendered,
        )
        for outcome in sim.outcomes
    ] == [(0, 0.0, 0.0, 1, 0, False), (1, 0.0, 0.0, 2, 0, False), (2, 0.0, 0.0, 3, 0, False)]
    assert [outcome.action_output_tokens for outcome in sim.outcomes] == [None, 10, 100_000]
    assert [outcome.post_horizon_output_tokens for outcome in sim.outcomes] == [0, 0, 0]
    starts = [command for command in sim.commands if isinstance(command, StartGeneration)]
    assert [start.remaining_clock_tokens for start in starts] == [None, None, None]
    termination = sim.termination
    assert (termination.stop_reason, termination.ended_at, termination.snapshot_at) == (
        "tick_limit",
        3.0,
        3.0,
    )
    assert termination.ops_ended_at == pytest.approx(300.1)  # 0.1 + 50 + 250


def test_in_paused_mode_generation_latency_changes_nothing_in_world_time():
    """暂停模式: the same replies, instant or slow, give the same world timeline and the same
    commands; only the operational clock differs."""
    fast, slow = _paused((0.0, 0.0, 0.0)), _paused((0.1, 50.0, 250.0))
    assert [(e.kind, e.semantic) for e in fast.records] == [
        (e.kind, e.semantic) for e in slow.records
    ]

    def world(sim):
        # Finish carries the termination, whose operational times differ by design.
        return [c for c in sim.commands if not isinstance(c, Record | SetTimer | Finish)]

    assert world(fast) == world(slow)
    assert _semantics(fast) == _semantics(slow)
    assert (fast.termination.ended_at, slow.termination.ended_at) == (3.0, 3.0)
    assert (fast.termination.snapshot_at, slow.termination.snapshot_at) == (3.0, 3.0)
    assert fast.termination.ops_ended_at == 0.0
    assert slow.termination.ops_ended_at == pytest.approx(300.1)


# ---- end signals -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "world", "checks", "termination"),
    [
        # The reply returns at 2 s, when the task reports its end; the result (a completion) is
        # handled before the periodic check due then (a start), and the check before landing,
        # asked at 2 s, finds the end.
        (
            _wall(10.0),
            World(
                replies=[Reply(latency=2.0, action=CLICK)], ends_at=2.0, end_reason="task_success"
            ),
            [0, 1, 2],
            Termination("task_success", "environment", 2.0, 2.0, 2.0, 2.0, None),
        ),
        # 150 tokens put the action at 1.5, where the task reports its end; the check after the
        # last approach step finds it.
        (
            _token(10.0),
            World(replies=[Reply(latency=1.0, action=CLICK, output_tokens=150)], ends_at=1.5),
            [0, 1, 1.5],
            Termination("game_decided", "environment", 1.5, 1.5, 1.0, 1.0, None),
        ),
    ],
    ids=["wall", "token"],
)
def test_an_end_beats_a_landing_at_the_same_instant(config, world, checks, termination):
    """The task's end at the instant the action would land wins: the action is not applied
    (legacy_loop.py _land_action) and the end is dated at that instant."""
    sim = _run(config, world)
    (outcome,) = sim.outcomes
    assert (outcome.status, outcome.terminal_after_action) == ("not_applied_terminal", False)
    assert sim.executed == []
    assert _semantics(sim) == checks
    assert sim.termination == termination


@pytest.mark.parametrize(
    ("config", "world", "reason", "aborted_at", "checks", "termination"),
    [
        # The token clock's operational watchdog (300 s) fires; world time stood still at 0.
        (
            _token(10.0),
            World(replies=[Reply(latency=400.0, action=CLICK, output_tokens=100)]),
            "invalid",
            300.0,
            [0],
            Termination(
                "invalid_run",
                "invalid",
                0.0,
                0.0,
                300.0,
                300.0,
                Invalidated("provider", "generation_watchdog_timeout", "generation exceeded 300s"),
            ),
        ),
        (
            _wall(3.0),
            World(replies=[Reply(latency=10.0, action=CLICK)]),
            "deadline",
            3.0,
            [0, 1, 2],
            Termination("time_limit", "deadline", 3.0, 3.0, 3.0, 3.0, None),
        ),
        # The task is over from 2.5 s; each check takes 0.25 s. The check asked at 3 s finds it,
        # so the end is dated 3 s; it is handled, and the call interrupted, at 3.25 s.
        (
            _wall(10.0),
            World(
                replies=[Reply(latency=6.0, action=CLICK)],
                ends_at=2.5,
                end_reason="target_reached",
                check_s=0.25,
            ),
            "environment",
            3.25,
            [0, 1, 2, 3],
            Termination("target_reached", "environment", 3.0, 3.25, 3.25, 3.25, None),
        ),
    ],
    ids=["token-watchdog", "wall-deadline", "wall-periodic-check"],
)
def test_an_episode_ending_mid_generation_abandons_it(
    config, world, reason, aborted_at, checks, termination
):
    """The episode ends while the model is still generating. The generation is written off with
    the end's cause and interrupted, and nothing of it is proposed, executed or reported."""
    sim = _run(config, world)
    abandoned = sim.entries("GenerationAbandoned")
    assert [(entry.data["generation"], entry.data["reason"]) for entry in abandoned] == [
        (1, reason)
    ]
    assert sim.aborted == [(1, aborted_at)]
    assert _semantics(sim) == checks
    assert (sim.accepted, sim.executed, sim.outcomes) == ([], [], [])
    assert "ActionProposed" not in sim.kinds()
    assert sim.termination == termination


def test_the_deadline_during_a_synchronous_execution_waits_for_it():
    """The deadline falls due while the shell executes an action that takes 1 s (2.5 s to 3.5 s).
    The kernel handles it once the shell returns: the end is dated at the deadline, the snapshot is
    taken after the operation, and the executed action is reported applied with the episode over
    (legacy_loop.py _land_action; phase2.md 时间和结束判定, 冻结的延迟有上限)."""
    sim = _run(_wall(3.0), World(replies=[Reply(latency=2.5, action=CLICK)], execute_s=1.0))
    assert sim.executed == [(0, 2.5, CLICK)]
    (outcome,) = sim.outcomes
    assert (outcome.status, outcome.terminal_after_action) == ("applied", True)
    assert (outcome.action_clock_position, outcome.frame_age_s) == (2.5, 2.5)
    # The check before execution, and none after it: the episode was already over when the shell
    # returned.
    assert sim.checks == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (2.5, 2.5)]
    assert sim.snapshots == [(3.5, 3.5)]
    assert sim.termination == Termination("time_limit", "deadline", 3.0, 3.5, 3.5, 3.5, None)


# ---- stale results and resolutions -------------------------------------------------------------


@pytest.mark.parametrize(
    ("interruptible", "discarded_at"),
    [(False, 2.5), (True, 1.0)],
    ids=["decision-call", "interruptible-call"],
)
def test_a_stale_result_never_lands(interruptible, discarded_at):
    """A legacy 1 s per-move cap gives up on the first call at 1 s: that decision is skipped as a
    timeout, charged the cap, and the episode goes on (legacy_loop.py _charge). A Decision-style
    call goes on asking and returns at its own time, 2.5 s; an interruptible one hands back at once.
    Either way its result is stale: discarded while later decisions run, never executed, and the
    call's records are numbered with no other effect."""
    first = Reply(
        latency=2.5,
        action=Action.click(5, 5),
        interruptible=interruptible,
        records=((1.25, "ModelRequestSent"), (2.25, "ModelAttemptFinished")),
    )
    replies = [
        first,
        Reply(latency=0.5, action=Action.click(7, 7)),
        Reply(latency=0.75, action=Action.click(9, 9)),
        Reply(latency=10.0, action=CLICK),
    ]
    sim = _run(_wall(3.5, per_move_cap_s=1.0), World(replies=replies))
    abandoned = sim.entries("GenerationAbandoned")
    assert [(entry.data["generation"], entry.data["reason"]) for entry in abandoned] == [
        (1, "per_move_cap"),
        (4, "deadline"),
    ]
    assert sim.aborted == [(1, 1.0), (4, 3.5)]
    (discarded,) = sim.entries("ResultDiscarded")
    assert (discarded.data, discarded.semantic) == ({"generation": 1}, discarded_at)
    assert (sim.accepted, sim.discarded) == ([2, 3], [1])
    # Frames at 1 s and 2 s are ticks 1 and 2: decision 1 waits for tick 2, decision 2 for 3.
    assert sim.executed == [(1, 1.5, Action.click(7, 7)), (2, 2.75, Action.click(9, 9))]
    assert [(outcome.decision, outcome.status) for outcome in sim.outcomes] == [
        (0, "skipped_timeout"),
        (1, "applied"),
        (2, "applied"),
    ]
    skipped = sim.outcomes[0]
    assert (skipped.action_clock_position, skipped.frame_age_s) == (1.0, 1.0)
    assert skipped.next_observation_tick == 1
    agent = [entry for entry in sim.records if entry.kind.startswith("Model")]
    assert [(entry.kind, entry.semantic, entry.data.generation) for entry in agent] == [
        ("ModelRequestSent", 1.25, 1),
        ("ModelAttemptFinished", 2.25, 1),
    ]
    assert sim.termination == Termination("time_limit", "deadline", 3.5, 3.5, 3.5, 3.5, None)


def test_every_proposed_action_is_resolved_exactly_once():
    """A refused action and an unusable result are each resolved once. The refusal is reported with
    the guest's reason and renders nothing at the boundary (legacy_loop.py _land_action); the result
    without token usage on a finite-B clock invalidates the run first and gets no feedback
    (legacy_loop.py _apply; episode_records.py result_invalid), then an invalid_run resolution,
    which today's loop does not write."""
    replies = [
        Reply(latency=1.0, action=CLICK, output_tokens=100),
        Reply(latency=1.0, action=CLICK),
    ]
    sim = _run(_token(10.0), World(replies=replies, rejections={0: "unsupported key"}))
    assert [entry.data["decision"] for entry in sim.entries("ActionProposed")] == [0, 1]
    assert [
        (entry.data["decision"], entry.data["status"]) for entry in sim.entries("ActionResolved")
    ] == [(0, "rejected"), (1, "invalid_run")]
    (outcome,) = sim.outcomes
    assert (outcome.decision, outcome.status, outcome.detail) == (0, "rejected", "unsupported key")
    assert sim.renders == []
    assert sim.executed == [(0, 1, CLICK)]
    assert _from(sim, "RunInvalidated")[:3] == ["RunInvalidated", "EpisodeEnded", "ActionResolved"]
    detail = "finite token-clock decisions require provider or deterministic local usage"
    assert sim.termination == Termination(
        "invalid_run",
        "invalid",
        1.0,
        1.0,
        2.0,
        2.0,
        Invalidated("provider", "missing_output_token_usage", detail),
    )


DONE = Action(ActionType.DONE)


@pytest.mark.parametrize(
    ("config", "world", "status", "action", "protocol_failure"),
    [
        (
            _wall(10.0),
            World([Reply(1.25, CLICK), Reply(100.0)], rejections={0: "no"}),
            "rejected",
            CLICK,
            None,
        ),
        (
            _wall(10.0, think_budget_s=1.0),
            World([Reply(1.25, DONE)]),
            "skipped_budget",
            Action.wait(),
            "unsupported_terminal_action",
        ),
        # Back on the frame boundary at 4 s and at 2 s, ahead of the periodic check: the frame is
        # read before execution.
        (
            _wall(10.0),
            World([Reply(4.0, CLICK)], ends_at=3.5),
            "not_applied_terminal",
            CLICK,
            None,
        ),
        (
            _wall(3.0),
            World([Reply(2.0, CLICK)], check_seconds={3: 1.5}),
            "not_applied_time_limit",
            CLICK,
            None,
        ),
        (
            _wall(2.0, per_move_cap_s=1.0),
            World([Reply(2.0, CLICK)] * 2),
            "skipped_timeout",
            None,
            None,
        ),
        (
            _token(10.0),
            World([Reply(1.0, CLICK, output_tokens=100)], execute_errors={0: "no"}),
            "invalid_run",
            CLICK,
            None,
        ),
        (
            _token(10.0),
            World([Reply(1.0, DONE, output_tokens=100)], ends_at=1),
            "not_applied_terminal",
            Action.wait(),
            "unsupported_terminal_action",
        ),
        (
            _token(2.0),
            World([Reply(1.0, CLICK, output_tokens=200)]),
            "not_applied_tick_limit",
            CLICK,
            None,
        ),
    ],
    ids=[
        "wall-rejected",
        "wall-skipped_budget",
        "wall-not_applied_terminal",
        "wall-not_applied_time_limit",
        "wall-skipped_timeout",
        "token-invalid_run",
        "token-not_applied_terminal",
        "token-not_applied_tick_limit",
    ],
)
def test_the_outcome_carries_the_action_the_decision_came_to_whatever_came_of_it(
    config, world, status, action, protocol_failure
):
    """Outcome.action is the action the decision came to, a DONE already the wait it became, and
    Outcome.protocol_failure why, whatever its status: the status says whether it was executed, as
    in today's lifecycle record (legacy_loop.py _apply). A move over the per-move cap comes to
    none."""
    sim = _run(config, world)
    (outcome,) = sim.outcomes[:1]
    assert (outcome.status, outcome.action, outcome.protocol_failure) == (
        status,
        action,
        protocol_failure,
    )


@pytest.mark.parametrize(
    ("config", "world", "termination"),
    [
        # Checks take 0.25 s and the snapshot 0.5 s: the check asked at 3 s finds the end (from
        # 2.5 s), answered at 3.25 s; the snapshot is in at 3.75 s.
        (
            _wall(10.0),
            World(
                replies=[Reply(latency=6.0, action=CLICK)],
                ends_at=2.5,
                end_reason="task_success",
                check_s=0.25,
                snapshot_s=0.5,
            ),
            Termination("task_success", "environment", 3.0, 3.75, 3.25, 3.75, None),
        ),
        # Steps take 0.25 s from the reply at 2 s: the check at tick 2 (2.5 s) finds the end; the
        # 0.5 s snapshot is in at 3 s, while world time stays at tick 2.
        (
            _token(10.0),
            World(
                replies=[Reply(latency=2.0, action=CLICK, output_tokens=250)],
                ends_at=2.0,
                advance_s=0.25,
                snapshot_s=0.5,
            ),
            Termination("game_decided", "environment", 2.0, 2.0, 2.5, 3.0, None),
        ),
        # A WAIT across the deadline is reported after the end and the report fails: the run is
        # invalid, the cause stays the deadline.
        (
            _wall(3.0),
            World(
                replies=[Reply(latency=0.5, action=Action.wait(5.0))],
                commit_errors={0: "feedback failed"},
            ),
            Termination(
                "invalid_run",
                "deadline",
                3.0,
                3.0,
                3.0,
                3.0,
                Invalidated("provider", "action_commit_feedback_failed", "feedback failed"),
            ),
        ),
    ],
    ids=["wall-check", "token-check", "invalid-after-the-end"],
)
def test_the_termination_record(config, world, termination):
    """终止记录: the stop reason, the cause, the end time by rule (the time the answering check was
    asked), the time the snapshot was taken, on both clocks; an invalidation after the end still
    makes the run invalid. It is written after the snapshot, last, then inputs are released."""
    sim = _run(config, world)
    assert sim.termination == termination
    (entry,) = sim.entries("Termination")
    assert entry.data == asdict(termination)
    assert sim.kinds().index("SnapshotFrozen") < sim.kinds().index("Termination")
    assert sim.released_inputs


# ---- the two exceptions ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("latency", "discarded"),
    [(4.5, [1]), (9.0, [])],
    ids=["back-during-the-snapshot", "back-after-the-episode"],
)
def test_a_decision_call_aborted_at_the_deadline_is_not_waited_for(latency, discarded):
    """Decision exception: the call asks its questions inside the generation and cannot be cut
    short. At the deadline it is written off and, once the 2 s snapshot is in, interrupted; the
    question it asks meanwhile is numbered; a result back before the episode is over is discarded;
    and the episode finishes when the snapshot is in, not when the call returns."""
    questions = ((0.5, "ModelRequestSent"), (1.5, "ModelRequestSent"), (4.0, "ModelRequestSent"))
    reply = Reply(latency=latency, action=CLICK, interruptible=False, records=questions)
    sim = _run(_wall(3.0), World(replies=[reply], snapshot_s=2.0))
    asked = sim.entries("ModelRequestSent")
    assert [(entry.semantic, entry.data.generation) for entry in asked] == [
        (0.5, 1),
        (1.5, 1),
        (5.0, 1),
    ]
    assert sim.aborted == [(1, 5.0)]
    assert (sim.accepted, sim.discarded, sim.executed, sim.outcomes) == ([], discarded, [], [])
    assert sim.ops == 5.0
    assert sim.termination == Termination("time_limit", "deadline", 3.0, 5.0, 3.0, 5.0, None)


def test_session_messages_mid_generation_are_numbered_with_no_other_effect():
    """Gemini Live exception: the session acknowledges tool calls during the generation. Those
    records are numbered in order inside the generation, nothing more is sent to the agent, and the
    action lands just as it does without them."""

    def run(records):
        replies = [Reply(latency=1.75, action=CLICK, records=records), Reply(latency=5.0)]
        return _run(_wall(3.0), World(replies=replies))

    acks = ((0.4, "SessionMessage"), (0.9, "SessionMessage"), (1.4, "SessionMessage"))
    with_acks, without = run(acks), run(())
    messages = with_acks.entries("SessionMessage")
    assert [(entry.semantic, entry.data.generation) for entry in messages] == [
        (0.4, 1),
        (0.9, 1),
        (1.4, 1),
    ]
    started = with_acks.entries("InferenceStarted")[0].seq
    finished = with_acks.entries("GenerationFinished")[0].seq
    assert started < messages[0].seq < messages[1].seq < messages[2].seq < finished

    def others(sim):
        return [command for command in sim.commands if not isinstance(command, Record)]

    assert others(with_acks) == others(without)
    assert [(e.kind, e.semantic) for e in with_acks.records if e.kind != "SessionMessage"] == [
        (e.kind, e.semantic) for e in without.records
    ]
    # The click lands at 1.75 s and the next frame is at tick 2.
    assert with_acks.executed == without.executed == [(0, 1.75, CLICK)]
    assert with_acks.outcomes == without.outcomes
    assert with_acks.outcomes[0].next_observation_tick == 2
    assert with_acks.termination == without.termination


# ---- random wall episodes ---------------------------------------------------------------------


def _random_wall_episode(seed: int) -> tuple[TurnConfig, World]:
    """A wall episode whose environment operations and calls take random times, so that answers
    wait behind synchronous commands and come due around the deadline."""
    rng = random.Random(seed)
    duration_s = rng.choice([3.0, 5.0, 6.0, 8.0])
    actions = [
        CLICK,
        Action.wait(rng.choice([0.0, 0.3, 1.0, 2.5, 7.0])),
        Action(ActionType.KEY, {"key": "a"}),
        Action.hold_key("b", rng.choice([0.2, 1.0, 3.0])),
        None,
    ]
    replies = [
        Reply(
            latency=rng.choice([0.05, 0.3, 0.9, 1.7, 2.6, 4.2, 9.0]),
            action=rng.choice(actions),
            interruptible=rng.random() < 0.7,
            records=tuple((rng.random() * 3, "SessionMessage") for _ in range(rng.randint(0, 2))),
        )
        for _ in range(40)
    ]
    world = World(
        replies=replies,
        ends_at=rng.choice([None, None, rng.random() * duration_s * 1.2]),
        end_reason=rng.choice([None, "target_reached"]),
        check_s=rng.choice([0.05, 0.15, 0.3, 0.7]),
        capture_s=rng.choice([0.0, 0.1, 0.4]),
        execute_s=rng.choice([0.0, 0.2, 1.3]),
        commit_s=rng.choice([0.0, 0.1, 0.5]),
        snapshot_s=rng.choice([0.0, 0.5, 2.0]),
        start_s=rng.choice([0.0, 0.0, 0.6]),
    )
    config = _wall(duration_s, per_move_cap_s=rng.choice([None, None, None, 1.5]))
    world.accept_s = rng.choice([0.0, 0.0, 0.01, 0.3])
    return config, world


def _random_token_episode(seed: int) -> tuple[TurnConfig, World]:
    """A token-clock episode at one of the acceptance frame rates, finite B or paused, with taps,
    holds and waits, an end from some tick and some failed reads."""
    rng = random.Random(seed)
    env_fps, input_fps = rng.choice([(1, 1), (2, 2), (60, 1), (30, 7)])
    tokens_per_frame = rng.choice([None, 50, 100, 256])
    duration_s = rng.choice([2.0, 3.0])
    actions = [
        CLICK,
        Action.wait(rng.choice([0.0, 0.3, 1.0, 2.5])),
        Action(ActionType.KEY, {"key": "a", "repeat": rng.choice([1, 2])}),
        Action.hold_key("b", rng.choice([0.2, 1.0, 3.0])),
        None,
    ]
    tokens = [0, 30, 100, 300] if tokens_per_frame else [None, 10]
    replies = [
        Reply(rng.choice([0.05, 0.3, 1.0]), rng.choice(actions), rng.choice(tokens))
        for _ in range(60)
    ]
    max_ticks = duration_s * env_fps
    world = World(
        replies=replies,
        ends_at=rng.choice([None, None, rng.random() * max_ticks * 1.2]),
        end_reason=rng.choice([None, "target_reached"]),
        check_errors={rng.randint(1, 40): "read failed" for _ in range(rng.randint(0, 3))},
        tap_check_errors={rng.randint(1, 5): "read failed" for _ in range(rng.randint(0, 2))},
    )
    config = TurnConfig(
        "token",
        duration_s,
        env_fps=env_fps,
        input_fps=input_fps,
        tokens_per_frame=tokens_per_frame,
    )
    return config, world


def test_random_token_episodes_keep_the_invariants():
    """At every acceptance rate, with taps reading inside the action, no frame is read twice
    besides once after every action, and every other invariant holds."""
    for seed in range(500):
        config, world = _random_token_episode(seed)
        sim = Simulation(config, world).run()
        try:
            _check_invariants(sim)
        except AssertionError as error:
            raise AssertionError(f"random token episode {seed}") from error


def test_random_wall_episodes_keep_the_invariants():
    """Whatever the operations cost, no episode captures, starts, executes or asks anything from
    the deadline on, a check asked before the deadline after the task ended ends the episode by
    the task's end however late it answers, and every other invariant holds."""
    for seed in range(1000):
        config, world = _random_wall_episode(seed)
        sim = Simulation(config, world).run()
        try:
            _check_invariants(sim)
            if world.ends_at is not None and any(
                world.ends_at <= asked < config.duration_s for asked, _ in sim.checks
            ):
                assert sim.termination.cause == "environment"
        except AssertionError as error:
            raise AssertionError(f"random wall episode {seed}") from error
