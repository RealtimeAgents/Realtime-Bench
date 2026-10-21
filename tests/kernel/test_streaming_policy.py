"""The streaming policy's rules, one at a time, on the discrete-event simulation.

Every expected value is worked out by hand from phase2.md (native streaming) and the decisions of
10-20: frames on the input-frame grid, end checks on the environment grid, one push, one reply, one
execution and one end check at a time, actions handed over as they appear, and every outcome
committed to the agent. Times are seconds since the episode origin; both frame rates are 1 and
every operation takes no time unless a test says otherwise.
"""

from __future__ import annotations

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.evaluation.kernel.events import (
    Capture,
    CheckEnd,
    Clock,
    Commit,
    Execute,
    Invalidated,
    Outcome,
    Record,
    SetTimer,
    StartGeneration,
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
    beats the check due with it."""
    sim = _run(World(), 3.0, env_fps=2.0)
    commands = [c for c in sim.commands[:9] if not isinstance(c, Record)]
    assert commands == [
        SetTimer(Clock.SEMANTIC, 3.0),
        SetTimer(Clock.SEMANTIC, 0.5),
        SetTimer(Clock.SEMANTIC, 1.0),
        CheckEnd(1),
        Capture(),
    ]
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
    """The click's feedback holds the shell from 0.25 to 2.75. The capture due at 1 is made then,
    and the one due at 2 is skipped."""
    world = World(replies=[Reply(5.0, calls=(Call(0.25, CLICK),))], commit_s=2.5)
    sim = _run(world)
    assert sim.captures == [0.0, 2.75, 3.0, 4.0]
    assert sim.data("FrameDropped") == [{"observation_id": None, "reason": "late", "dropped": 1}]
    assert sim.checks == [0.0, 2.75, 3.0, 4.0]


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
    """The third capture runs from 2 to 3.5, across the deadline; the termination waits for it."""
    sim = _run(World(capture_times={3: 1.5}, capture_errors={3: "screenshot failed"}), 3.0)
    assert _ended(sim) == ("invalid_run", "deadline", 3.0, 3.0)
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


def test_a_push_not_answered_gives_the_reply_up_and_the_next_one_starts():
    """The second push runs from 1 to 6. Its watchdog fires at 3: the reply in flight is given up
    and the next starts on the newest frame pushed, the first."""
    world = World(push_times={2: 5.0})
    sim = _run(world, 7.0, push_watchdog_s=2.0)
    assert sim.started == [(1, 0.0), (2, 3.0)]
    assert sim.data("GenerationAbandoned")[0] == {"generation": 1, "reason": "push_watchdog"}
    assert sim.data("InferenceStarted")[1]["observation_id"] == "obs-0"
    assert sim.aborted == [(1, 3.0), (2, 7.0)]
    assert sim.discarded[0] == 1
    # Captures at 2 to 5 wait for the push; each newer one drops the one before.
    assert [d["observation_id"] for d in sim.data("FrameDropped")] == ["obs-2", "obs-3", "obs-4"]
    assert sim.pushes[2] == ("obs-5", 3)


def test_a_push_stuck_before_the_first_reply_gives_nothing_up():
    sim = _run(World(push_times={1: 12.0}), 5.0, push_watchdog_s=2.0)
    assert sim.started == []
    assert "GenerationAbandoned" not in sim.kinds()
    assert _ended(sim) == ("time_limit", "deadline", 5.0, 5.0)


# ---- end checks ---------------------------------------------------------------------------------


def test_one_end_check_is_out_at_a_time_and_one_due_meanwhile_is_asked_when_it_answers():
    """The first read takes 2.5 s; the checks due at 1 and 2 are asked once, when it answers."""
    sim = _run(World(check_times={1: 2.5}))
    assert sim.checks == [0.0, 2.5, 3.0, 4.0]


def test_the_environment_end_is_dated_by_the_check_that_saw_it():
    sim = _run(World(ends_at=2.0, end_reason="success", check_s=0.5))
    assert sim.checks == [0.0, 1.0, 2.0]
    assert _ended(sim) == ("success", "environment", 2.0, 2.5)


def test_a_read_failing_after_the_end_still_counts():
    """The reads at 0 and 1 fail; the one asked at 2 is answered at 3.5, after the deadline, and is
    the third failure in a row."""
    errors = {1: "cdp timeout", 2: "cdp timeout", 3: "cdp timeout"}
    sim = _run(World(check_errors=errors, check_times={3: 1.5}), 3.0)
    assert _ended(sim) == ("invalid_run", "deadline", 3.0, 3.0)
    assert sim.termination.invalid == Invalidated(
        "environment", "terminal_state_check_failed", "cdp timeout"
    )


# ---- the end ------------------------------------------------------------------------------------


def test_the_deadline_beats_an_arrival_filed_before_it():
    """The click ends at 4.75 as the typing arrives; its feedback holds the shell until 5.75, and
    then the typing, filed at 4.75, is handled before the deadline timer. The deadline ends the
    episode first, and the typing is not applied."""
    world = World(
        replies=[
            Reply(1.0, calls=(Call(0.5, CLICK, execute_s=4.25),)),
            Reply(10.0, calls=(Call(3.75, TYPE),)),
        ],
        commit_s=1.0,
    )
    sim = _run(world)
    assert sim.statuses() == ["applied", "not_applied_time_limit"]
    assert sim.executed == [(0.5, CLICK)]
    ended = sim.entries("EpisodeEnded")[0]
    assert (ended.semantic, ended.data) == (5.75, {"cause": "deadline", "reason": None, "at": 5.0})
    assert _ended(sim) == ("time_limit", "deadline", 5.0, 5.75)


def test_the_deadline_beats_an_execution_answered_before_it():
    """The click ends at 4.75 as the refused action arrives; the refusal's feedback holds the shell
    until 5.75, and the click's answer, filed at 4.75, is handled then. The deadline ends the
    episode first, and the click is its last action, as on the turn-based clock."""
    calls = (Call(0.5, CLICK, execute_s=4.25), Call(4.75, TYPE, grounding=MISSING_FRAME))
    sim = _run(World(replies=[Reply(10.0, calls=calls)], commit_s=1.0))
    assert sim.statuses() == ["applied", INVALID_GROUNDING_FRAME]
    assert sim.resolved()[0]["terminal_after_action"] is True
    assert _ended(sim) == ("time_limit", "deadline", 5.0, 5.75)


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
    and the deadline settles the typing first."""
    calls = (
        Call(0.5, CLICK, "a", execute_s=6.0),
        Call(1.0, TYPE, "b"),
        Call(4.5, CLICK, "c", grounding=MISSING_FRAME),
    )
    reply = Reply(10.0, calls=calls, cancellations=((4.5, ("b",)),))
    sim = _run(World(replies=[reply], commit_s=1.0))
    assert sim.statuses() == ["applied", "not_applied_time_limit", INVALID_GROUNDING_FRAME]
    assert sim.data("CallCancelled") == [
        {"index": 1, "generation": 1, "call_id": "b", "status": "not_applied_time_limit"}
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
    sim = _run(World(replies=[Reply(3.0, calls=calls)], invalidations=((1.5, failure),)))
    assert sim.statuses() == ["applied", "invalid_run"]
    applied = sim.resolved()[0]
    assert (applied["finished_at"], applied["terminal_after_action"]) == (2.5, True)
    assert _ended(sim) == ("invalid_run", "invalid", 1.5, 2.5)


def test_no_generation_starts_after_the_last_one_before_the_deadline():
    sim = _run(World(replies=[Reply(1.0)] * 10), 3.0)
    starts = [c for c in sim.commands if isinstance(c, StartGeneration)]
    assert [s.decision for s in starts] == [0, 1, 2]
    assert sim.started == [(1, 0.0), (2, 1.0), (3, 2.0)]
