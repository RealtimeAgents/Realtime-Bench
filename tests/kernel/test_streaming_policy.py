"""The streaming policy's rules, one at a time, on the discrete-event simulation.

Every expected value is worked out by hand from phase2.md (native streaming) and the decisions of
10-20: frames on the input-frame grid, end checks on the environment grid, one push, one reply, one
execution and one end check at a time, actions handed over as they appear, and every outcome
committed to the agent. Times are seconds since the episode origin; both frame rates are 1 and
every operation takes no time unless a test says otherwise.
"""

from __future__ import annotations

import pytest

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.evaluation.kernel.events import (
    CheckEnd,
    Clock,
    Commit,
    Execute,
    Invalidated,
    Outcome,
    Record,
    SetTimer,
    StartGeneration,
    Termination,
)
from realtime_framework.evaluation.kernel.streaming import StreamingConfig
from realtime_framework.harnesses.default.frames import (
    FRAME_NOT_IN_TABLE,
    INVALID_GROUNDING_FRAME,
    MISSING_FRAME,
    FrameTable,
)
from tests.kernel.streaming_simulation import Call, Reply, Simulation, World

CLICK = Action.click(10, 20)
TYPE = Action(ActionType.TYPE, {"text": "hi"})


def _run(world: World, duration_s: float = 5.0, **fields) -> Simulation:
    return Simulation(StreamingConfig(duration_s, **fields), world).run()


def _ended(sim: Simulation) -> tuple:
    t = sim.termination
    return t.stop_reason, t.cause, t.ended_at, t.snapshot_at


def _outcome(decision: int, status: str, at: float, **fields) -> Outcome:
    """An outcome as streaming commits it: the turn-based fields stay empty."""
    return Outcome(
        decision=decision,
        status=status,
        detail=fields.pop("detail", ""),
        terminal_after_action=fields.pop("terminal_after_action", False),
        action_clock_position=at,
        next_observation_tick=0.0,
        dropped_intermediate_frames=0,
        post_action_rendered=False,
        frame_age_s=fields.pop("frame_age_s", None),
        token_delay_frames=None,
        action_output_tokens=fields.pop("action_output_tokens", None),
        post_horizon_output_tokens=0,
        **fields,
    )


# ---- the cadence --------------------------------------------------------------------------------


def test_the_episode_asks_and_captures_at_once_then_on_both_grids():
    """The end checks follow the environment grid, the captures the input-frame grid; the deadline
    beats the check due with it. A read runs alone: the first capture follows the first read's
    answer."""
    sim = _run(World(), 3.0, env_fps=2.0)
    commands = [c for c in sim.commands[:8] if not isinstance(c, Record)]
    assert commands == [
        SetTimer(Clock.SEMANTIC, 3.0),
        SetTimer(Clock.SEMANTIC, 0.5),
        SetTimer(Clock.SEMANTIC, 1.0),
        CheckEnd(1),
    ]
    names = [c.entry.kind if isinstance(c, Record) else type(c).__name__ for c in sim.commands]
    assert names.index("EndChecked") < names.index("Capture")
    assert sim.checks == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert sim.captures == [0.0, 1.0, 2.0]
    assert sim.pushes == [("obs-0", 1), ("obs-1", 2), ("obs-2", 3)]
    assert _ended(sim) == ("time_limit", "deadline", 3.0, 3.0)


def test_the_first_reply_waits_for_the_first_frame_pushed():
    sim = _run(World(capture_s=0.25, push_s=0.5), 2.0)
    assert sim.started == [(1, 0.75)]
    (started,) = sim.data("InferenceStarted")
    assert started == {
        "decision": 0,
        "generation": 1,
        "observation_id": "obs-0",
        "observation_tick": 0,
        "remaining_clock_tokens": None,
    }


def test_a_failed_push_loses_that_frame_only():
    sim = _run(World(push_errors={1: "rejected item"}), 3.0)
    assert sim.started == [(1, 1.0)]
    assert sim.data("InferenceStarted")[0]["observation_id"] == "obs-1"
    assert sim.termination.invalid is None


def test_the_next_reply_starts_when_the_last_is_accepted_not_when_its_action_ends():
    """The click executes from 0.5 to 3.5; the replies after it start at 1 and 2 meanwhile. The
    second came to no action and is reported as such."""
    world = World(
        replies=[Reply(1.0, calls=(Call(0.5, CLICK, execute_s=3.0),)), Reply(1.0)],
    )
    sim = _run(world)
    assert sim.started == [(1, 0.0), (2, 1.0), (3, 2.0)]
    assert sim.accepted == [1, 2]
    assert sim.executed == [(0.5, CLICK)]
    assert sim.outcomes == [
        _outcome(1, "none", 2.0),
        _outcome(0, "applied", 0.5, frame_age_s=0.5, arrived_at=0.5),
    ]


# ---- frames -------------------------------------------------------------------------------------


def test_a_grid_point_that_comes_during_a_capture_is_skipped_and_counted():
    sim = _run(World(capture_times={1: 1.5}), 4.0)
    assert sim.captures == [0.0, 2.0, 3.0]
    assert sim.data("FrameDropped") == [
        {"observation_id": None, "reason": "capture_busy", "dropped": 1}
    ]
    assert sim.started == [(1, 1.5)]


def test_grid_points_that_pass_while_the_shell_is_held_are_skipped_and_counted():
    """The click's feedback, given once the read after it answers, holds the shell from 0.25 to
    2.75. The capture due at 1 is made then, and the one due at 2 is skipped."""
    world = World(replies=[Reply(5.0, calls=(Call(0.25, CLICK),))], commit_s=2.5)
    sim = _run(world)
    assert sim.captures == [0.0, 2.75, 3.0, 4.0]
    assert sim.data("FrameDropped") == [{"observation_id": None, "reason": "late", "dropped": 1}]
    assert sim.checks == [0.0, 0.25, 2.75, 3.0, 4.0]


def test_a_stream_that_repeats_one_frame_invalidates_the_run():
    """From the capture at 1 on the stream replays one frame; the third time it is seen, at 3,
    the stream is frozen."""
    sim = _run(World(frozen_from=1))
    assert _ended(sim) == ("invalid_run", "invalid", 3.0, 3.0)
    assert sim.termination.invalid == Invalidated(
        "environment", "observation_stream_frozen", repr(("frame-1", 1, "sha-1"))
    )


def test_a_capture_that_fails_invalidates_the_run():
    sim = _run(World(capture_errors={2: "screenshot failed"}), 4.0)
    assert _ended(sim) == ("invalid_run", "invalid", 1.0, 1.0)
    assert sim.termination.invalid == Invalidated(
        "environment", "observation_capture_failed", "screenshot failed"
    )


def test_a_capture_under_way_at_the_end_that_fails_still_invalidates_the_run():
    """The third capture runs from 2 to 3.5, across the deadline; the snapshot waits for it, and
    so does the termination."""
    sim = _run(World(capture_times={3: 1.5}, capture_errors={3: "screenshot failed"}), 3.0)
    assert _ended(sim) == ("invalid_run", "deadline", 3.0, 3.5)
    assert sim.termination.invalid.reason == "observation_capture_failed"
    assert sim.records[-1].semantic == 3.5


# ---- actions ------------------------------------------------------------------------------------


def test_an_action_executes_at_once_and_reports_its_call_frame_and_age():
    """The click comes off the wire at 1.5: the newest frame sent is frame 2, captured at 1. The
    typing names frame 1, captured at 0 with one frame sent after it, and waits for the click."""
    calls = (Call(1.5, CLICK, "call-1"), Call(1.75, TYPE, "call-2", frame=1))
    sim = _run(World(replies=[Reply(4.0, calls=calls)], execute_s=0.5))
    assert sim.executed == [(1.5, CLICK), (2.0, TYPE)]
    assert sim.outcomes == [
        _outcome(0, "applied", 1.5, frame_age_s=0.5, call_id="call-1", arrived_at=1.5),
        _outcome(0, "applied", 2.0, frame_age_s=1.75, call_id="call-2", arrived_at=1.75),
    ]
    first, second = sim.resolved()[0], sim.resolved()[1]
    assert (first["frame"], first["frame_age_frames"]) == (2, 0)
    assert (second["frame"], second["frame_age_frames"]) == (1, 1)
    assert (second["started_at"], second["finished_at"]) == (2.0, 2.5)
    assert [d["index"] for d in sim.data("ActionApplied")] == [0, 1]


def test_a_frame_age_leaves_out_a_frame_whose_push_failed_as_the_frame_table_does():
    """Frame 2's push fails at 1 and frame 3 goes out at 2: the typing at 2.5 names frame 1,
    which only frame 3 was sent after."""
    calls = (Call(2.5, TYPE, "call-1", frame=1),)
    sim = _run(World(replies=[Reply(4.0, calls=calls)], push_errors={2: "rejected item"}))
    assert sim.pushes[:3] == [("obs-0", 1), ("obs-1", 2), ("obs-2", 3)]
    (resolved,) = sim.resolved().values()
    assert (resolved["frame"], resolved["frame_age_frames"]) == (1, 1)
    table = FrameTable()
    records = [table.register(f"obs-{n}", float(n), {}, None) for n in range(3)]
    for record in (records[0], records[2]):
        table.record_sent(record.frame, record.captured_at)
    assert table.age(records[0], 2.5)["age_frames"] == resolved["frame_age_frames"]


def test_a_wait_ends_at_once_and_holds_the_execution_lane_no_longer():
    calls = (Call(0.5, Action.wait(2.0)), Call(0.75, CLICK))
    sim = _run(World(replies=[Reply(3.0, calls=calls)], execute_s=1.0))
    assert sim.statuses() == ["wait", "applied"]
    wait = sim.resolved()[0]
    assert (wait["started_at"], wait["finished_at"]) == (0.5, 0.5)
    assert wait["execution_result"] == {
        "duration_simulated_seconds": 0.0,
        "requested_duration_simulated_seconds": 2.0,
    }
    assert sim.executed == [(0.75, CLICK)]


def test_a_wait_that_arrives_during_an_execution_ends_there_and_supersedes_nothing():
    """The typing waits behind the click; the wait at 0.75 ends at once and leaves it there."""
    calls = (Call(0.25, CLICK, execute_s=2.0), Call(0.5, TYPE), Call(0.75, Action.wait(1.0)))
    sim = _run(World(replies=[Reply(3.0, calls=calls)]))
    assert sim.statuses() == ["applied", "applied", "wait"]
    assert sim.resolved()[2]["started_at"] == 0.75
    assert sim.resolved()[1]["superseded_by"] is None
    assert sim.executed == [(0.25, CLICK), (2.25, TYPE)]


def test_done_and_fail_become_a_wait_with_a_protocol_failure():
    """Neither is a proposal, as on the turn-based clock."""
    calls = (Call(0.5, Action(ActionType.DONE)), Call(1.0, Action(ActionType.FAIL)))
    sim = _run(World(replies=[Reply(3.0, calls=calls)]))
    assert sim.statuses() == ["wait", "wait"]
    assert sim.data("ActionProposed") == []
    assert sim.executed == []
    for outcome in sim.outcomes:
        assert (outcome.action, outcome.protocol_failure) == (
            Action.wait(),
            "unsupported_terminal_action",
        )


def test_an_action_whose_frame_cannot_be_found_is_refused_and_supersedes_nothing():
    """The typing waits for the click; the action with no frame is refused at once and leaves the
    typing in its slot."""
    calls = (
        Call(0.25, CLICK, execute_s=2.0),
        Call(0.5, TYPE),
        Call(0.75, CLICK, "c", grounding=FRAME_NOT_IN_TABLE),
    )
    sim = _run(World(replies=[Reply(4.0, calls=calls)]))
    assert sim.statuses() == ["applied", "applied", INVALID_GROUNDING_FRAME]
    assert sim.executed == [(0.25, CLICK), (2.25, TYPE)]
    refused = sim.outcomes[0]
    assert refused == _outcome(
        0, INVALID_GROUNDING_FRAME, 0.75, detail=FRAME_NOT_IN_TABLE, call_id="c", arrived_at=0.75
    )
    assert refused.action is None
    assert [data["index"] for data in sim.data("ActionProposed")] == [0, 1]


def test_a_refusal_by_the_guest_is_rejected_and_the_run_goes_on():
    sim = _run(World(replies=[Reply(3.0, calls=(Call(0.5, CLICK, rejection="occluded"),))]))
    assert sim.statuses() == ["rejected"]
    assert sim.outcomes[0].detail == "occluded"
    assert _ended(sim) == ("time_limit", "deadline", 5.0, 5.0)


def test_an_action_the_environment_fails_on_invalidates_the_run():
    """The typing waiting behind the failed click will not execute."""
    calls = (Call(0.5, CLICK, error="guest down", execute_s=1.0), Call(1.0, TYPE))
    sim = _run(World(replies=[Reply(3.0, calls=calls)]))
    assert sim.statuses() == ["invalid_run", "invalid_run"]
    assert sim.executed == [(0.5, CLICK)]
    assert _ended(sim) == ("invalid_run", "invalid", 1.5, 1.5)
    assert sim.termination.invalid == Invalidated(
        "environment", "action_execution_failed", "guest down"
    )


def test_a_wall_hold_is_released_by_the_shell_timer():
    calls = (Call(0.5, Action.hold_key("a", 1.5), execute_s=0.25),)
    sim = _run(World(replies=[Reply(3.0, calls=calls)]))
    (execute,) = [c for c in sim.commands if isinstance(c, Execute)]
    assert execute.release_at == 2.0
    assert sim.data("KeyReleaseScheduled") == [{"decision": 0, "index": 0, "at": 2.0}]
    assert sim.key_events == [(2.0, ("a",))]


def test_a_release_that_fails_invalidates_the_run():
    calls = (Call(0.5, Action.hold_key("a", 1.5)),)
    sim = _run(World(replies=[Reply(3.0, calls=calls)], release_errors={"a": "xdotool failed"}))
    assert _ended(sim) == ("invalid_run", "invalid", 2.0, 2.0)
    assert sim.termination.invalid == Invalidated(
        "environment", "wall_key_release_failed", "xdotool failed"
    )


def test_feedback_is_given_after_the_commands_for_the_world():
    """The click ends at 0.75, when the typing waits. The typing starts then, before the click's
    feedback, which occupies the shell for 0.25 s."""
    calls = (Call(0.25, CLICK, execute_s=0.5), Call(0.5, TYPE))
    sim = _run(World(replies=[Reply(3.0, calls=calls)], commit_s=0.25))
    assert sim.executed == [(0.25, CLICK), (0.75, TYPE)]
    execute = next(i for i, c in enumerate(sim.commands) if c == Execute(0, TYPE, None))
    commit = next(i for i, c in enumerate(sim.commands) if isinstance(c, Commit))
    assert execute < commit


def test_a_feedback_failure_invalidates_the_run():
    sim = _run(World(replies=[Reply(3.0, calls=(Call(0.5, CLICK),))], commit_errors={1: "closed"}))
    assert _ended(sim) == ("invalid_run", "invalid", 0.5, 0.5)
    assert sim.termination.invalid == Invalidated(
        "provider", "action_commit_feedback_failed", "closed"
    )


# ---- replies ------------------------------------------------------------------------------------


def test_a_reply_that_came_to_no_action_is_reported_with_why_and_its_tokens():
    world = World(
        replies=[Reply(1.0, protocol_failure="no_action_found", output_tokens=42), Reply(1.0)]
    )
    sim = _run(world, 3.0)
    assert sim.outcomes == [
        _outcome(0, "none", 1.0, action_output_tokens=42),
        _outcome(1, "none", 2.0),
    ]
    assert [o.protocol_failure for o in sim.outcomes] == ["no_action_found", None]


def test_a_failed_reply_invalidates_the_run():
    failure = Invalidated("provider", "provider_request_failed", "HTTP 500")
    sim = _run(World(replies=[Reply(1.0, invalid=failure)]))
    assert _ended(sim) == ("invalid_run", "invalid", 1.0, 1.0)
    assert sim.termination.invalid == failure
    assert sim.started == [(1, 0.0)]
    assert sim.outcomes == []


def test_one_end_check_is_out_at_a_time_and_one_due_meanwhile_is_asked_when_it_answers():
    """The first read takes 2.5 s; the checks due at 1 and 2 are asked once, when it answers."""
    sim = _run(World(check_times={1: 2.5}))
    assert sim.checks == [0.0, 2.5, 3.0, 4.0]


def test_the_environment_end_is_dated_by_the_check_that_saw_it():
    sim = _run(World(ends_at=2.0, end_reason="success", check_s=0.5))
    assert sim.checks == [0.0, 1.0, 2.0]
    assert _ended(sim) == ("success", "environment", 2.0, 2.5)


def test_a_third_failed_read_asked_before_the_deadline_invalidates_though_it_answers_after():
    """The reads at 0 and 1 fail; the one asked at 2 is answered at 3.5, after the deadline, and is
    the third failure in a row. The end waited for it, as the turn-based kernel files the answer
    before the deadline, and the run is invalid, dated where that read was asked (N44)."""
    errors = {1: "cdp timeout", 2: "cdp timeout", 3: "cdp timeout"}
    sim = _run(World(check_errors=errors, check_times={3: 1.5}), 3.0)
    assert _ended(sim) == ("invalid_run", "invalid", 2.0, 3.5)
    assert sim.termination.invalid == Invalidated(
        "environment", "terminal_state_check_failed", "cdp timeout"
    )


def test_a_third_failed_read_after_an_action_is_dated_where_the_action_finished():
    """The reads at 0 and 1 fail. The click executes from 1.25 to 1.5 alongside the capture at 1,
    which takes until 2; the read after the click waits for it and fails too, the third in a row.
    It stands for a read as the click finished (N31), so the run is invalid from 1.5 (N44)."""
    world = World(
        replies=[Reply(10.0, calls=(Call(1.25, CLICK, execute_s=0.25),))],
        check_errors={1: "cdp timeout", 2: "cdp timeout", 3: "cdp timeout"},
        capture_times={2: 1.0},
    )
    sim = _run(world, 5.0)
    assert sim.checks == [0.0, 1.0, 2.0]
    assert _ended(sim) == ("invalid_run", "invalid", 1.5, 2.0)


# ---- the end ------------------------------------------------------------------------------------


def test_the_deadline_beats_an_arrival_filed_before_it():
    """A refused action and the typing arrive at 4.75; the refusal's feedback holds the shell
    until 5.75, and then the typing, filed at 4.75, is handled before the deadline timer. The
    deadline ends the episode first, and the typing is not applied."""
    calls = (Call(4.75, CLICK, "c", grounding=MISSING_FRAME), Call(4.75, TYPE))
    sim = _run(World(replies=[Reply(10.0, calls=calls)], commit_s=1.0))
    assert sim.statuses() == [INVALID_GROUNDING_FRAME, "not_applied_time_limit"]
    assert sim.executed == []
    ended = sim.entries("EpisodeEnded")[0]
    assert (ended.semantic, ended.data) == (5.75, {"cause": "deadline", "reason": None, "at": 5.0})
    assert _ended(sim) == ("time_limit", "deadline", 5.0, 5.75)


@pytest.mark.parametrize(
    ("ends_at", "ended"),
    [
        (None, ("time_limit", "deadline", 5.0, 5.75)),
        (4.75, ("game_decided", "environment", 4.75, 5.75)),
    ],
)
def test_the_deadline_waits_for_the_read_after_an_execution_answered_before_it(ends_at, ended):
    """The click ends at 4.75 as the refused action arrives; the refusal's feedback holds the shell
    until 5.75, and the click's answer, filed at 4.75, is handled then. The deadline waits for the
    read after the click (N31), and the click is the last action; an end that read finds is dated
    where the click finished."""
    calls = (Call(0.5, CLICK, execute_s=4.25), Call(4.75, TYPE, grounding=MISSING_FRAME))
    sim = _run(World(replies=[Reply(10.0, calls=calls)], commit_s=1.0, ends_at=ends_at))
    assert sim.checks[-1] == 5.75
    assert sim.statuses() == ["applied", INVALID_GROUNDING_FRAME]
    assert sim.resolved()[0]["terminal_after_action"] is True
    assert _ended(sim) == ended


@pytest.mark.parametrize("input_fps", [1.0, 30.0])
def test_the_deadline_waits_for_the_read_after_an_action_answered_while_the_shell_is_busy(
    input_fps,
):
    """The click finishes at 4.75 and ends the task; its answer comes while the refusal's feedback,
    from 4.7, holds the shell until 5.7. It is filed where the click finished, before the deadline,
    so the deadline waits for the read after it (N31), as for an answer filed before the shell
    was held. At 30 frames a second a grid point between 4.7 and 4.75 is handled at 5.7 first; it
    leaves the end to the deadline's own timer, which comes after the click's answer."""
    calls = (Call(0.5, CLICK, "a", execute_s=4.25), Call(4.7, TYPE, "b", grounding=MISSING_FRAME))
    world = World(replies=[Reply(10.0, calls=calls)], commit_s=1.0, ends_at=4.75)
    sim = _run(world, input_fps=input_fps)
    assert sim.checks == [0.0, 5.7]
    assert _ended(sim) == ("game_decided", "environment", 4.75, 5.7)
    assert sim.resolved()[0]["terminal_after_action"] is True


def test_the_deadline_beats_a_watchdog_due_before_it():
    """The click's feedback holds the shell from 4.25 to 5.25, across the watchdog due at 4.5 and
    the deadline. The watchdog is handled first, but the deadline gives the reply up."""
    world = World(replies=[Reply(10.0, calls=(Call(4.25, CLICK),))], commit_s=1.0)
    sim = _run(world, reply_watchdog_s=4.5)
    assert sim.data("GenerationAbandoned") == [{"generation": 1, "reason": "deadline"}]
    assert sim.started == [(1, 0.0)]
    assert _ended(sim) == ("time_limit", "deadline", 5.0, 5.25)


def test_the_deadline_beats_a_cancellation_filed_before_it():
    """The typing waits behind the click, which executes until 6.5. The refusal at 4.5 is given as
    feedback until 5.5; the cancellation of the typing's call, filed at 4.5 too, is handled then,
    past the deadline, and withdraws nothing: the typing is left to the deadline, which settles
    it once its timer comes, after the cancellation."""
    calls = (
        Call(0.5, CLICK, "a", execute_s=6.0),
        Call(1.0, TYPE, "b"),
        Call(4.5, CLICK, "c", grounding=MISSING_FRAME),
    )
    reply = Reply(10.0, calls=calls, cancellations=((4.5, ("b",)),))
    sim = _run(World(replies=[reply], commit_s=1.0))
    assert sim.statuses() == ["applied", "not_applied_time_limit", INVALID_GROUNDING_FRAME]
    assert sim.data("CallCancelled") == [
        {"index": 1, "generation": 1, "call_id": "b", "status": None}
    ]
    assert _ended(sim) == ("time_limit", "deadline", 5.0, 6.5)


def test_an_action_handed_over_after_the_end_is_not_applied():
    """The task ends at 1; the snapshot holds the shell until 2, and the reply, which keeps
    talking after the abort, hands an action over at 1.5."""
    world = World(
        replies=[Reply(3.0, calls=(Call(1.5, CLICK),), interruptible=False)],
        ends_at=1.0,
        snapshot_s=1.0,
    )
    sim = _run(world)
    assert sim.statuses() == ["not_applied_terminal"]
    assert sim.executed == []
    assert _ended(sim) == ("game_decided", "environment", 1.0, 2.0)


def test_an_invalidation_from_another_thread_settles_the_waiting_action():
    calls = (Call(0.5, CLICK, execute_s=2.0), Call(1.0, TYPE))
    failure = Invalidated("environment", "wall_key_release_failed", "x")
    sim = _run(World(replies=[Reply(3.0, calls=calls)], posts=((1.5, failure),)))
    assert sim.statuses() == ["applied", "invalid_run"]
    applied = sim.resolved()[0]
    assert (applied["finished_at"], applied["terminal_after_action"]) == (2.5, True)
    assert _ended(sim) == ("invalid_run", "invalid", 1.5, 2.5)


def test_no_generation_starts_after_the_last_one_before_the_deadline():
    sim = _run(World(replies=[Reply(1.0)] * 10), 3.0)
    starts = [c for c in sim.commands if isinstance(c, StartGeneration)]
    assert [s.decision for s in starts] == [0, 1, 2]
    assert sim.started == [(1, 0.0), (2, 1.0), (3, 2.0)]


# ---- the shell's gate -----------------------------------------------------------------------------


def test_an_end_an_action_brings_is_dated_by_the_read_that_saw_it():
    """The click executes from 1 to 4 and the task reports its end from 3.9. The reads due at 1,
    2 and 3 wait for the click; the one at 4 sees the end."""
    calls = (Call(1.0, CLICK, execute_s=3.0),)
    sim = _run(World(replies=[Reply(10.0, calls=calls)], ends_at=3.9), 10.0)
    assert sim.checks == [0.0, 4.0]
    assert _ended(sim) == ("game_decided", "environment", 4.0, 4.0)


def test_an_action_that_arrives_while_a_read_holds_the_environment_waits_and_not_past_the_deadline():
    """The read asked at 4 takes 3 s; the click at 4.5 waits for it and the deadline comes first."""
    world = World(replies=[Reply(10.0, calls=(Call(4.5, CLICK),))], check_times={5: 3.0})
    sim = _run(world)
    assert sim.executed == []
    assert sim.statuses() == ["not_applied_time_limit"]
    assert sim.termination.cause == "deadline"


def test_a_hold_that_waits_for_the_environment_is_held_its_whole_duration():
    """The read asked at 1 takes 2 s; the 1 s hold arriving at 1.25 is pressed at 3 and released
    at 4."""
    hold = Action.hold_key("a", 1.0)
    world = World(replies=[Reply(5.0, calls=(Call(1.25, hold),))], check_times={2: 2.0})
    sim = _run(world)
    assert sim.executed == [(3.0, hold)]
    assert sim.data("KeyReleaseScheduled") == [{"decision": 0, "index": 0, "at": 4.0}]
    assert sim.key_events == [(4.0, ("a",))]


# ---- reading the state (decision 31) ------------------------------------------------------------


def test_the_state_is_read_right_after_an_action_and_an_end_it_brings_is_found_then():
    """The click executes from 4.25 to 4.75, after the read at 4, and ends the task. The read
    right after it finds the end before the deadline."""
    calls = (Call(4.25, CLICK, execute_s=0.5),)
    sim = _run(World(replies=[Reply(10.0, calls=calls)], ends_at=4.75))
    assert sim.checks == [0.0, 1.0, 2.0, 3.0, 4.0, 4.75]
    assert _ended(sim) == ("game_decided", "environment", 4.75, 4.75)
    assert sim.resolved()[0]["terminal_after_action"] is True
    assert sim.outcomes[0].terminal_after_action is True


def test_an_action_waiting_behind_one_that_ends_the_task_never_executes():
    """The click executes from 2.25 to 2.75 and ends the task; the typing waits behind it. The read
    after the click comes first and finds the end."""
    calls = (Call(2.25, CLICK, execute_s=0.5), Call(2.5, TYPE))
    sim = _run(World(replies=[Reply(10.0, calls=calls)], ends_at=2.75))
    assert sim.executed == [(2.25, CLICK)]
    assert sim.statuses() == ["applied", "not_applied_terminal"]
    assert _ended(sim) == ("game_decided", "environment", 2.75, 2.75)


def test_a_failed_read_after_an_action_lets_the_action_waiting_execute():
    """The click executes from 2.25 to 2.75 and ends the task; the read after it fails, which tells
    nothing, as on the turn-based clock. The typing waiting behind it executes then, and the read
    after the typing finds the end."""
    calls = (Call(2.25, CLICK, execute_s=0.5), Call(2.5, TYPE))
    errors = {4: "cdp timeout"}
    sim = _run(World(replies=[Reply(10.0, calls=calls)], ends_at=2.75, check_errors=errors), 10.0)
    assert sim.checks == [0.0, 1.0, 2.0, 2.75, 2.75]
    assert sim.executed == [(2.25, CLICK), (2.75, TYPE)]
    assert [data["terminal_after_action"] for data in sim.resolved().values()] == [False, True]
    assert _ended(sim) == ("game_decided", "environment", 2.75, 2.75)


def test_an_action_is_settled_by_the_read_after_it():
    """The click executes from 1.25 to 1.5; the read after it takes until 1.75. The click is applied
    once that read answers, and only then does the typing waiting behind it execute, followed by
    its own read."""
    calls = (Call(1.25, CLICK, execute_s=0.25), Call(1.375, TYPE))
    sim = _run(World(replies=[Reply(10.0, calls=calls)], check_times={3: 0.25}), 3.0)
    assert sim.checks == [0.0, 1.0, 1.5, 1.75, 2.0]
    assert sim.executed == [(1.25, CLICK), (1.75, TYPE)]
    click = sim.entries("ActionResolved")[0]
    assert (click.semantic, click.data["finished_at"]) == (1.75, 1.5)


def test_a_frame_the_read_after_an_action_read_is_not_read_again_at_its_grid_point():
    """The click ends at 1.875 as a refused action arrives; the refusal's feedback holds the shell
    until 2.125. The read after the click is then made in frame 2, before the grid point at 2
    is handled, which finds frame 2 read."""
    calls = (Call(1.25, CLICK, execute_s=0.625), Call(1.875, TYPE, grounding=MISSING_FRAME))
    sim = _run(World(replies=[Reply(10.0, calls=calls)], commit_s=0.25), 3.0)
    assert sim.checks == [0.0, 1.0, 2.125]


# ---- a read across the deadline ---------------------------------------------------------------


def test_a_read_asked_before_the_deadline_that_finds_the_end_beats_it():
    """The task reports its end from 2; the read asked at 2 takes 1.5 s and answers after the
    deadline at 3. The end waits for it and is the task's, dated by the read."""
    world = World(ends_at=2.0, end_reason="success", check_times={3: 1.5})
    sim = _run(world, 3.0)
    assert sim.termination == Termination("success", "environment", 2.0, 3.5, 3.5, 3.5, None)


def test_a_read_across_the_deadline_that_finds_no_end_ends_the_episode_at_the_deadline():
    sim = _run(World(check_times={3: 1.5}), 3.0)
    assert sim.termination == Termination("time_limit", "deadline", 3.0, 3.5, 3.5, 3.5, None)


def test_the_deadline_waiting_for_a_read_gives_the_reply_in_flight_up_at_once():
    """The read asked at 4 takes 1.5 s; the reply started at 1 would end at 5.2, past the deadline,
    while the end waits for that read. It is given up at the deadline, as it would be without the
    wait, and nothing it says lands."""
    world = World(replies=[Reply(1.0), Reply(4.2, output_tokens=77)], check_times={5: 1.5})
    sim = _run(world)
    assert sim.data("GenerationAbandoned") == [{"generation": 2, "reason": "deadline"}]
    assert sim.aborted == [(2, 5.0)]
    assert sim.accepted == [1]
    assert [o.decision for o in sim.outcomes] == [0]
    assert _ended(sim) == ("time_limit", "deadline", 5.0, 5.5)


def test_a_reply_that_would_fail_while_the_deadline_waits_for_a_read_leaves_the_run_valid():
    failure = Invalidated("provider", "provider_request_failed", "connection reset")
    world = World(replies=[Reply(3.25, invalid=failure)], check_times={3: 1.5})
    sim = _run(world, 3.0)
    assert sim.termination == Termination("time_limit", "deadline", 3.0, 3.5, 3.5, 3.5, None)


def _click_during_a_capture(capture_s: float, ends_at: float | None) -> Simulation:
    """The click executes from 4.1 to 4.6 alongside the capture due at 4, which takes
    ``capture_s``; the read after the click waits for that capture."""
    world = World(
        replies=[Reply(10.0, calls=(Call(4.1, CLICK, "c", execute_s=0.5),))],
        ends_at=ends_at,
        end_reason="goal",
        capture_times={5: capture_s},
    )
    return _run(world)


def test_the_deadline_waits_for_the_read_after_an_action_behind_a_capture_across_it():
    """N31: the capture runs until 5.2, past the deadline. The deadline waits for the read after
    the click, made once the capture is done; the end it finds is dated where the click finished,
    as the turn-based clock reads right then."""
    sim = _click_during_a_capture(1.2, ends_at=4.6)
    assert sim.checks == [0.0, 1.0, 2.0, 3.0, 4.0, 5.2]
    assert sim.termination == Termination("goal", "environment", 4.6, 5.2, 5.2, 5.2, None)
    assert sim.resolved()[0]["terminal_after_action"] is True


def test_an_end_the_read_after_an_action_finds_is_dated_where_the_action_finished():
    """The capture runs until 4.8, before the deadline; the read after the click waits for it,
    and the end it finds is dated at 4.6 all the same."""
    sim = _click_during_a_capture(0.8, ends_at=4.6)
    assert sim.checks == [0.0, 1.0, 2.0, 3.0, 4.0, 4.8]
    assert _ended(sim) == ("goal", "environment", 4.6, 4.8)


def test_a_read_after_an_action_past_the_deadline_that_finds_no_end_ends_it_at_the_deadline():
    sim = _click_during_a_capture(1.2, ends_at=None)
    assert sim.checks == [0.0, 1.0, 2.0, 3.0, 4.0, 5.2]
    assert sim.termination == Termination("time_limit", "deadline", 5.0, 5.2, 5.2, 5.2, None)
    assert sim.statuses() == ["applied"]


def test_an_action_that_arrives_while_the_end_waits_for_a_read_is_settled_by_that_end():
    """The click arrives at 3.25, while the end waits for the read asked at 2, from the reply given
    up at the deadline, whose messages keep coming as from the old connection."""
    reply = Reply(10.0, calls=(Call(3.25, CLICK),), interruptible=False)
    world = World(replies=[reply], ends_at=2.0, check_times={3: 1.5})
    sim = _run(world, 3.0)
    assert sim.executed == []
    assert sim.statuses() == ["not_applied_terminal"]
    assert sim.termination.cause == "environment"


# ---- frames that do not go into the session ------------------------------------------------------


def test_a_push_stuck_until_the_reply_is_given_up_recovers_with_the_rebuilt_session():
    """The second push hangs until the session is closed. Its watchdog gives the reply up at 3;
    closing the session fails that push, the frame waiting goes in, and the run goes on."""
    sim = _run(World(stuck_pushes={2}), 6.0, push_watchdog_s=2.0)
    assert sim.data("GenerationAbandoned")[0] == {"generation": 1, "reason": "push_watchdog"}
    assert sim.started[:2] == [(1, 0.0), (2, 3.0)]
    assert [frame for _, frame in sim.pushes] == [1, 2, 3, 4, 5, 6]
    assert sorted(sim.sent) == [1, 3, 4, 5, 6]
    assert sim.termination.stop_reason == "time_limit"


def test_a_push_still_stuck_after_the_session_was_rebuilt_invalidates_the_run():
    """The second push never answers. The reply is given up at 3, and the next starts on the newest
    frame that went in; at 5 no frame has gone in since, and the run is invalid. Meanwhile each
    frame captured drops the one waiting before it, and the last one still waits at the end."""
    sim = _run(World(push_times={2: 100.0}), 20.0, push_watchdog_s=2.0)
    assert sim.started == [(1, 0.0), (2, 3.0)]
    assert sim.data("InferenceStarted")[1]["observation_id"] == "obs-0"
    assert [(d["observation_id"], d["reason"]) for d in sim.data("FrameDropped")] == [
        ("obs-2", "superseded"),
        ("obs-3", "superseded"),
        ("obs-4", "ended"),
    ]
    assert _ended(sim)[:3] == ("invalid_run", "invalid", 5.0)
    assert sim.termination.invalid.reason == "frame_push_stuck"


def test_a_push_stuck_after_the_rebuild_with_the_end_before_its_watchdog_leaves_the_run_valid():
    """The third push hangs; its watchdog gives the reply up at 4, and the rebuilt session takes
    the next frame no better. The end at 5 comes before that push's watchdog at 6, so no watchdog
    has found the rebuilt session stuck, and the run stands."""
    sim = _run(World(stuck_pushes={3, 4}), 5.0, push_watchdog_s=2.0)
    assert sim.data("GenerationAbandoned") == [
        {"generation": 1, "reason": "push_watchdog"},
        {"generation": 2, "reason": "deadline"},
    ]
    assert sim.termination == Termination("time_limit", "deadline", 5.0, 5.0, 5.0, 5.0, None)


@pytest.mark.parametrize(
    "errors",
    [{2: "item rejected", 4: "item rejected"}, {4: "item rejected", 5: "item rejected"}],
)
def test_a_session_rebuilt_for_a_stuck_push_counts_its_own_failed_pushes_only(errors):
    """The third push hangs; its watchdog gives the reply up at 4, and the abort fails that push.
    It was the old session's, as was a failure before it: the rebuilt session has failed one or
    two pushes, not three in a row, and the run goes on."""
    sim = _run(World(push_errors=errors, stuck_pushes={3}), 12.0, push_watchdog_s=2.0)
    assert [data["reason"] for data in sim.data("GenerationAbandoned")] == [
        "push_watchdog",
        "deadline",
    ]
    assert sim.termination.invalid is None


def test_the_watchdogs_after_a_slow_abort_count_from_when_it_returned():
    """The second push hangs and its watchdog gives the reply up at 11; the abort holds the shell
    until 21.5 and fails that push as it closes the session. The rebuilt session and its reply
    have their whole watchdogs from then, so the next frame goes in and the run goes on."""
    sim = _run(World(stuck_pushes={2}, abort_s=10.5), 30.0)
    assert sim.data("GenerationAbandoned") == [
        {"generation": 1, "reason": "push_watchdog"},
        {"generation": 2, "reason": "deadline"},
    ]
    assert sim.started == [(1, 0.0), (2, 21.5)]
    assert sim.termination.invalid is None


def test_the_reply_after_a_slow_abort_has_its_whole_watchdog():
    """No reply ends. The first is given up at 4 and the abort holds the shell until 7, when the
    next one starts: its watchdog gives it up at 11, not 8. That abort runs past the deadline at
    12, so no reply starts after it."""
    sim = _run(World(abort_s=3.0), 12.0, reply_watchdog_s=4.0)
    assert sim.aborted == [(1, 4.0), (2, 11.0)]
    assert sim.started == [(1, 0.0), (2, 7.0)]


def test_a_push_stuck_before_the_first_reply_invalidates_the_run():
    """There is no reply to give up, so no session to rebuild."""
    sim = _run(World(push_times={1: 12.0}), 5.0, push_watchdog_s=2.0)
    assert sim.started == []
    assert _ended(sim)[:3] == ("invalid_run", "invalid", 2.0)
    assert sim.termination.invalid.reason == "frame_push_stuck"


def test_pushes_failing_in_a_row_rebuild_the_session_and_a_frame_that_goes_in_ends_it():
    """Pushes 2 to 4 fail: the third failure in a row, at 3, gives the reply up; push 5 goes in."""
    errors = {2: "item rejected", 3: "item rejected", 4: "item rejected"}
    sim = _run(World(push_errors=errors), 6.0)
    assert sim.data("GenerationAbandoned")[0] == {"generation": 1, "reason": "push_failed"}
    assert sim.started[:2] == [(1, 0.0), (2, 3.0)]
    assert sim.termination.invalid is None


def test_pushes_that_keep_failing_after_the_rebuild_invalidate_the_run():
    errors = {n: "item rejected" for n in range(2, 8)}
    sim = _run(World(push_errors=errors), 10.0)
    assert sim.started == [(1, 0.0), (2, 3.0)]
    assert _ended(sim)[:3] == ("invalid_run", "invalid", 6.0)
    assert sim.termination.invalid.reason == "frame_push_failed"


def test_every_push_failing_before_the_first_reply_invalidates_the_run():
    sim = _run(World(push_errors={1: "x", 2: "x", 3: "x"}), 5.0)
    assert sim.started == []
    assert _ended(sim)[:3] == ("invalid_run", "invalid", 2.0)
    assert sim.termination.invalid.reason == "frame_push_failed"


def test_a_reply_that_handed_actions_over_still_reports_its_protocol_failure():
    """The reply's click executes; what else it said came to no action, and the agent is told."""
    reply = Reply(1.0, calls=(Call(0.5, CLICK, "a"),), protocol_failure="malformed_second_call")
    sim = _run(World(replies=[reply]), 3.0)
    told = [(o.decision, o.status, o.call_id, o.protocol_failure) for o in sim.outcomes]
    assert told == [(0, "applied", "a", None), (0, "none", None, "malformed_second_call")]


def test_every_frame_captured_is_pushed_or_counted_as_dropped():
    """Each push takes 1.5 s; the second fails. Frame obs-1 is lost to the failure, obs-3 is
    dropped by obs-4 while waiting, and obs-5 still waits at the end; the push answered at the
    deadline went in."""
    sim = _run(World(push_s=1.5, push_errors={2: "item rejected"}), 6.0)
    assert [frame for _, frame in sim.pushes] == [1, 2, 3, 4]
    assert [(d["observation_id"], d["reason"]) for d in sim.data("FrameDropped")] == [
        ("obs-1", "push_failed"),
        ("obs-3", "superseded"),
        ("obs-5", "ended"),
    ]
    assert (sim.policy.frames_pushed, sim.policy.dropped_frames) == (3, 3)
    assert len(sim.captures) == 6


def test_a_frame_captured_across_the_end_is_counted_as_dropped():
    """The capture at 2 takes until 3.5, past the deadline at 3; its frame goes nowhere."""
    sim = _run(World(capture_times={3: 1.5}), 3.0)
    assert [(d["observation_id"], d["reason"]) for d in sim.data("FrameDropped")] == [
        ("obs-2", "ended")
    ]
    assert (sim.policy.frames_pushed, sim.policy.dropped_frames) == (2, 1)


def test_an_action_answered_past_the_deadline_is_settled_before_the_one_waiting():
    """The click ends at 4.75 as a refused action arrives; that feedback holds the shell until
    5.75, when the click's answer is handled past the deadline and the read after it is made. The
    click is settled first, with the time it really finished, then the end settles the typing
    waiting behind it."""
    calls = (
        Call(0.5, CLICK, "a", execute_s=4.25),
        Call(1.0, TYPE, "b"),
        Call(4.75, CLICK, "c", grounding=MISSING_FRAME),
    )
    sim = _run(World(replies=[Reply(10.0, calls=calls)], commit_s=1.0))
    assert [(o.call_id, o.status) for o in sim.outcomes] == [
        ("c", INVALID_GROUNDING_FRAME),
        ("a", "applied"),
        ("b", "not_applied_time_limit"),
    ]
    assert sim.resolved()[0]["finished_at"] == 4.75


@pytest.mark.parametrize(
    "fields",
    [
        {"duration_s": 0.0},
        {"env_fps": 0.0},
        {"input_fps": -1.0},
        {"reply_watchdog_s": 0.0},
        {"push_watchdog_s": 0.0},
        {"push_failure_limit": 0},
    ],
)
def test_a_config_that_would_divide_by_zero_or_never_let_time_pass_is_refused(fields):
    """A watchdog of no time gives replies up at one instant forever; a rate of 0 divides by 0."""
    with pytest.raises(ValueError):
        StreamingConfig(**({"duration_s": 5.0} | fields))


def test_a_config_off_the_wall_clock_is_refused():
    """A capture and an action run at once only on the wall clock: on the token clock an action
    moves the environment's clock while a capture reads it."""
    with pytest.raises(ValueError, match="wall clock"):
        StreamingConfig(5.0, clock_mode="token")


# ---- what a single change to the rules would break --------------------------------------------


def test_a_session_that_took_frames_after_its_rebuild_is_rebuilt_again_when_it_sticks_later():
    """The second push hangs and its watchdog rebuilds the session at 3; frames go in again, so
    the ninth push hanging long after is a stall of its own, which rebuilds the session once more
    instead of invalidating the run."""
    sim = _run(World(stuck_pushes={2, 9}), 15.0, push_watchdog_s=2.0)
    assert [data["reason"] for data in sim.data("GenerationAbandoned")] == [
        "push_watchdog",
        "push_watchdog",
        "deadline",
    ]
    assert sim.started == [(1, 0.0), (2, 3.0), (3, 10.0)]
    assert sim.termination.invalid is None


def test_pushes_failing_in_a_row_count_from_the_last_frame_that_went_in():
    """Pushes 2, 3 and 5 fail and push 4 goes in between: never three in a row."""
    sim = _run(World(push_errors={2: "item rejected", 3: "item rejected", 5: "item rejected"}), 8.0)
    assert sim.data("GenerationAbandoned") == [{"generation": 1, "reason": "deadline"}]
    assert sim.started == [(1, 0.0)]


def test_a_grid_point_that_comes_while_a_capture_waits_is_skipped_and_counted():
    """The read asked at 1 takes until 3.5; the capture due at 1 waits for it, and the grid points
    at 2 and 3 come while it waits."""
    sim = _run(World(check_times={2: 2.5}))
    assert sim.captures == [0.0, 3.5, 4.0]
    assert [(d["observation_id"], d["reason"]) for d in sim.data("FrameDropped")] == [
        (None, "capture_busy"),
        (None, "capture_busy"),
    ]


def test_giving_a_reply_up_leaves_the_action_another_reply_left_waiting():
    """The first reply ends at 1 with its typing waiting behind its click; the second is given up
    at 3. The typing is the first reply's and executes after the click."""
    first = Reply(1.0, calls=(Call(0.25, CLICK, "a", execute_s=3.0), Call(0.5, TYPE, "b")))
    sim = _run(World(replies=[first]), reply_watchdog_s=2.0)
    assert sim.data("GenerationAbandoned")[0] == {"generation": 2, "reason": "reply_watchdog"}
    assert sim.statuses() == ["applied", "applied"]
    assert sim.executed == [(0.25, CLICK), (3.25, TYPE)]


def test_the_watchdog_of_a_reply_given_up_does_not_fire_on_the_next():
    """The push watchdog gives the first reply up at 3; its own watchdog, due at 3.5, goes with it
    and the second reply runs until the run is found invalid at 5."""
    sim = _run(World(push_times={2: 100.0}), 10.0, push_watchdog_s=2.0, reply_watchdog_s=3.5)
    assert sim.data("GenerationAbandoned") == [
        {"generation": 1, "reason": "push_watchdog"},
        {"generation": 2, "reason": "invalid"},
    ]


def test_a_reply_result_handled_past_the_deadline_is_not_reported():
    """The second reply, started at 1 as the first one's feedback goes out, hands a refused action
    over as it ends at 4.75; that feedback holds the shell until 5.75, when the reply's result,
    protocol failure and all, is handled past the deadline. The deadline ends the episode first,
    and nothing reports the result."""
    second = Reply(3.75, calls=(Call(3.75, CLICK, grounding=MISSING_FRAME),), protocol_failure="p")
    sim = _run(World(replies=[Reply(1.0), second], commit_s=1.0))
    assert [o.status for o in sim.outcomes] == ["none", INVALID_GROUNDING_FRAME]
    assert sim.started == [(1, 0.0), (2, 1.0)]


def test_an_action_waiting_behind_one_the_guest_refused_then_executes():
    calls = (Call(0.25, CLICK, rejection="occluded", execute_s=1.0), Call(0.5, TYPE))
    sim = _run(World(replies=[Reply(3.0, calls=calls)]))
    assert sim.statuses() == ["rejected", "applied"]
    assert sim.executed == [(0.25, CLICK), (1.25, TYPE)]
    # The typing, waiting since before the read due at 1 came due, goes first once the refused
    # click is answered; the read after it stands for the grid's.
    assert sim.checks == [0.0, 1.25, 2.0, 3.0, 4.0]


def test_a_frame_repeated_once_the_episode_is_over_still_invalidates_the_run():
    """From the capture at 1 on the stream repeats a frame; the third time comes from the capture
    at 3, which answers at 3.5, after the deadline at 3.25. The frame is judged as a failed
    capture would be, as on the turn-based clock."""
    sim = _run(World(frozen_from=1, capture_times={4: 0.5}), 3.25)
    assert len(sim.captures) == 4
    assert _ended(sim) == ("invalid_run", "deadline", 3.25, 3.5)
    assert sim.termination.invalid.reason == "observation_stream_frozen"


# ---- frame rates ------------------------------------------------------------------------------


def test_the_landing_position_is_in_environment_ticks():
    """At 2 environment frames a second the click landing at 1.25 lands at tick 2.5."""
    sim = _run(World(replies=[Reply(3.0, calls=(Call(1.25, CLICK),))]), env_fps=2.0)
    assert sim.outcomes[0].action_clock_position == 2.5
    assert sim.checks[:4] == [0.0, 0.5, 1.0, 1.25]


@pytest.mark.parametrize("input_fps", [30.0, 7.0])
def test_a_fast_grid_lands_on_every_point_once(input_fps):
    """At 30 environment frames a second every grid point is read once and every frame point
    captured once, with nothing dropped; 1/30 s steps do not add up to whole points exactly."""
    sim = _run(World(), 1.0, env_fps=30.0, input_fps=input_fps)
    assert len(sim.checks) == 30
    assert len(sim.captures) == int(input_fps)
    assert sim.data("FrameDropped") == []


def test_reads_due_on_a_fast_grid_do_not_keep_the_captures_out():
    """At 30 environment frames a second a read of 0.5 s finds the next read due when it answers.
    The operations due take the environment in the order they came due, so the capture due since
    the start goes first at 0.5, and each capture due at a whole second goes after the read
    then out."""
    sim = _run(World(check_s=0.5), 3.0, env_fps=30.0)
    assert sim.captures == [0.5, 1.5, 2.5]
    assert sim.started == [(1, 0.5)]


def test_reads_slower_than_the_frames_take_turns_with_the_captures():
    """Each read takes 1.5 s, longer than a frame. Once a read answers, the capture waiting goes
    before the read due next, so that a frame is captured after every read."""
    sim = _run(World(check_s=1.5), 12.0)
    assert sim.checks == [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5]
    assert sim.captures == [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5]


# ---- captures during an action (decision 50) ---------------------------------------------------


def test_frames_keep_coming_while_a_long_action_executes():
    """The click executes from 1 to 6. A capture and an action share the environment, so every
    grid point captures meanwhile; the reads due wait for the click, and the read right after it
    stands for them."""
    sim = _run(World(replies=[Reply(10.0, calls=(Call(1.0, CLICK, execute_s=5.0),))]), 8.0)
    assert sim.captures == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert sim.data("FrameDropped") == []
    assert sim.checks == [0.0, 6.0, 7.0]


def test_an_action_starts_while_a_capture_runs():
    """The capture at 1 takes until 1.75; the click arriving at 1.25 executes at once. The read
    after it waits for the capture."""
    world = World(replies=[Reply(10.0, calls=(Call(1.25, CLICK),))], capture_times={2: 0.75})
    sim = _run(world, 3.0)
    assert sim.executed == [(1.25, CLICK)]
    assert sim.checks == [0.0, 1.0, 1.75, 2.0]


def test_a_read_that_waits_for_a_capture_holds_back_an_action_due_after_it():
    """At 2 environment frames a second the reads due at 1 and 1.5 wait for the capture running
    from 1 to 1.75, which came due first; the click arriving at 1.625 waits behind them, so reads
    cannot be starved. The read after the click follows it."""
    world = World(replies=[Reply(10.0, calls=(Call(1.625, CLICK),))], capture_times={2: 0.75})
    sim = _run(world, 3.0, env_fps=2.0)
    assert sim.executed == [(1.75, CLICK)]
    assert sim.checks == [0.0, 0.5, 1.75, 1.75, 2.0, 2.5]


def test_the_snapshot_waits_for_the_capture_and_the_action_out():
    """The click executes from 2 to 3.5 and the capture at 3 takes until 3.75; the deadline at
    3.25 freezes the snapshot once both have let the environment go."""
    calls = (Call(2.0, CLICK, execute_s=1.5),)
    sim = _run(World(replies=[Reply(10.0, calls=calls)], capture_times={4: 0.75}), 3.25)
    assert sim.snapshots == [3.75]
    assert sim.resolved()[0]["terminal_after_action"] is True
