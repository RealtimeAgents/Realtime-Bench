"""The streaming scenarios the native-streaming plan lists, and seeded random scripts, on the
discrete-event simulation of the streaming policy.

Every expectation is worked out from the rules (phase2.md, native streaming; the decisions of
10-20): one action executes at a time and the newest one to arrive meanwhile waits; a reply is
given up only at the end or by a watchdog, and what it hands over late is refused; a cancelled call
is withdrawn while it waits and recorded as it ran otherwise; the snapshot waits for the action
executing; three failed reads in a row invalidate the run. The simulation checks every command as
it runs and every episode as a whole (tests/kernel/streaming_simulation.py ``check``); the random
scripts check only that. Both frame rates are 1 and every operation takes no time unless a test
says otherwise.
"""

from __future__ import annotations

import random

import pytest

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.evaluation.kernel.events import Invalidated, Termination
from realtime_framework.evaluation.kernel.streaming import StreamingConfig
from realtime_framework.harnesses.default.frames import MISSING_FRAME
from tests.kernel.streaming_simulation import Call, Reply, Simulation, World

CLICK = Action.click(10, 20)
TYPE = Action(ActionType.TYPE, {"text": "hi"})
SCROLL = Action(ActionType.SCROLL, {"x": 5, "y": 5, "direction": "down"})
KEY = Action(ActionType.KEY, {"key": "enter"})


def _run(world: World, duration_s: float = 5.0, **fields) -> Simulation:
    return Simulation(StreamingConfig(duration_s, **fields), world).run()


def _from(sim: Simulation, kind: str) -> list[str]:
    """The record kinds from the first ``kind`` on."""
    kinds = sim.kinds()
    return kinds[kinds.index(kind) :]


def _told(sim: Simulation) -> list[tuple]:
    """(status, call id) of each outcome committed, in order."""
    return [(outcome.status, outcome.call_id) for outcome in sim.outcomes]


# ---- 长动作执行期间连到三个动作 ----------------------------------------------------------------------


def test_three_actions_arriving_during_a_long_one_leave_only_the_newest():
    """The click executes from 0.25 to 4.25. The second reply hands over three actions at 1.5,
    2 and 2.5: each takes the waiting slot from the one before, and the last executes when the
    click ends. The model is told of both that were superseded."""
    second = Reply(3.0, calls=(Call(0.5, TYPE, "b"), Call(1.0, SCROLL, "c"), Call(1.5, KEY, "d")))
    world = World(
        replies=[Reply(1.0, calls=(Call(0.25, CLICK, "a", execute_s=4.0),)), second],
        execute_s=0.5,
    )
    sim = _run(world, 6.0)
    assert sim.executed == [(0.25, CLICK), (4.25, KEY)]
    assert sim.statuses() == ["applied", "superseded", "superseded", "applied"]
    resolved = sim.resolved()
    assert [resolved[i]["superseded_by"] for i in range(4)] == [None, 2, 3, None]
    assert _told(sim) == [
        ("superseded", "b"),
        ("superseded", "c"),
        ("applied", "a"),
        ("applied", "d"),
    ]
    # The replies go on meanwhile: the third starts when the second ends, at 4.
    assert sim.started == [(1, 0.0), (2, 1.0), (3, 4.0)]
    superseded = sim.outcomes[0]
    assert (superseded.decision, superseded.arrived_at, superseded.action_clock_position) == (
        1,
        1.5,
        2.0,
    )


# ---- 推帧积压 -----------------------------------------------------------------------------------


def test_frames_captured_while_one_is_pushed_keep_only_the_newest():
    """Each push takes 2.5 s. The frame captured at 1 waits and the one at 2 drops it; that one
    is pushed at 2.5, and so on. The frame pushed when the episode ends is not waited for, and the
    one waiting then is dropped."""
    sim = _run(World(push_s=2.5), 6.0)
    assert sim.pushes == [("obs-0", 1), ("obs-2", 2), ("obs-4", 3)]
    assert sim.data("FrameDropped") == [
        {"observation_id": "obs-1", "reason": "superseded", "dropped": 1},
        {"observation_id": "obs-3", "reason": "superseded", "dropped": 2},
        {"observation_id": "obs-5", "reason": "ended", "dropped": 3},
    ]
    assert sim.started == [(1, 2.5)]
    assert sim.policy.frames_pushed == 2
    assert sim.records[-1].semantic == 6.0


# ---- 截止落在执行中途 ------------------------------------------------------------------------------


def test_a_deadline_during_an_execution_freezes_once_the_action_finishes():
    """The click executes from 1 to 4. The deadline at 3 gives the reply in flight up at once;
    the snapshot waits for the click, which lands as the episode's last action."""
    world = World(replies=[Reply(2.0, calls=(Call(1.0, CLICK, execute_s=3.0),))])
    sim = _run(world, 3.0)
    assert _from(sim, "EpisodeEnded") == [
        "EpisodeEnded",
        "GenerationAbandoned",
        "GenerationFinished",
        "ResultDiscarded",
        "ActionExecuted",
        "ActionResolved",
        "ActionApplied",
        "SnapshotFrozen",
        "OutcomeCommitted",
        "Termination",
    ]
    assert sim.aborted == [(2, 3.0)]
    assert sim.snapshots == [4.0]
    (resolved,) = sim.data("ActionResolved")
    assert (resolved["status"], resolved["finished_at"], resolved["terminal_after_action"]) == (
        "applied",
        4.0,
        True,
    )
    assert sim.termination == Termination("time_limit", "deadline", 3.0, 4.0, 3.0, 4.0, None)


def test_an_end_the_task_reports_during_an_execution_settles_the_waiting_action():
    """The task reports its end from 2, while the click executes from 1 to 4. The read due at 2
    waits for the click and goes before the typing waiting behind it: it sees the end at 4, and
    the typing is not applied."""
    calls = (Call(1.0, CLICK, execute_s=3.0), Call(1.5, TYPE))
    sim = _run(World(replies=[Reply(3.0, calls=calls)], ends_at=2.0))
    assert sim.statuses() == ["applied", "not_applied_terminal"]
    assert sim.executed == [(1.0, CLICK)]
    assert sim.checks == [0.0, 4.0]
    assert sim.snapshots == [4.0]
    assert sim.termination == Termination("game_decided", "environment", 4.0, 4.0, 4.0, 4.0, None)


# ---- 生成被看门狗放弃以后迟到的结果和动作 ------------------------------------------------------------


def test_a_reply_the_watchdog_gave_up_hands_over_late_and_is_refused():
    """The first reply would run 4 s; its watchdog gives it up at 2 and the next reply starts.
    Its click, executing since 1, runs to its end; its typing, handed over at 3, is refused, and
    its result at 4 is discarded. The capture due at 1 waits for the click, so the next reply
    starts on the first frame."""
    first = Reply(4.0, calls=(Call(1.0, CLICK, "a"), Call(3.0, TYPE, "b")), interruptible=False)
    second = Reply(1.0, calls=(Call(0.5, SCROLL, "c"),))
    sim = _run(World(replies=[first, second], execute_s=1.5), 4.5, reply_watchdog_s=2.0)
    assert sim.started == [(1, 0.0), (2, 2.0), (3, 3.0)]
    assert sim.data("GenerationAbandoned")[0] == {"generation": 1, "reason": "reply_watchdog"}
    assert sim.data("InferenceStarted")[1]["observation_id"] == "obs-0"
    assert sim.captures == [0.0, 2.5, 4.0]
    assert (sim.accepted, sim.discarded) == ([2], [1])
    assert sim.executed == [(1.0, CLICK), (2.5, SCROLL)]
    assert sim.statuses() == ["applied", "applied", "cancelled_generation"]
    assert _told(sim) == [("applied", "a"), ("cancelled_generation", "b"), ("applied", "c")]
    refused = sim.outcomes[1]
    assert (refused.decision, refused.arrived_at) == (0, 3.0)


def test_an_action_waiting_when_its_reply_is_given_up_is_refused_then():
    """The typing waits behind the click when the watchdog gives its reply up at 2."""
    first = Reply(4.0, calls=(Call(0.5, CLICK, execute_s=2.0), Call(1.0, TYPE)))
    sim = _run(World(replies=[first]), 4.0, reply_watchdog_s=2.0)
    assert sim.statuses() == ["applied", "cancelled_generation"]
    assert sim.resolved()[1]["superseded_by"] is None
    assert sim.outcomes[0].action_clock_position == 2.0
    assert sim.executed == [(0.5, CLICK)]


# ---- 工具调用取消在执行前和执行后各一次 ----------------------------------------------------------------


def test_a_cancelled_call_is_withdrawn_before_it_executes_and_recorded_after():
    """The click executes from 0.5 to 2; the typing waits. The provider cancels the typing's call
    at 1.5 and the click's at 2.5."""
    reply = Reply(
        4.0,
        calls=(Call(0.5, CLICK, "a"), Call(1.0, TYPE, "b")),
        cancellations=((1.5, ("b",)), (2.5, ("a",))),
    )
    sim = _run(World(replies=[reply], execute_s=1.5))
    assert sim.executed == [(0.5, CLICK)]
    assert sim.statuses() == ["applied", "cancelled_call"]
    assert _told(sim) == [("cancelled_call", "b"), ("applied", "a")]
    assert sim.data("CallCancelled") == [
        {"index": 0, "generation": 1, "call_id": "a", "status": "applied"}
    ]


def test_a_call_cancelled_while_its_action_executes_lets_it_finish():
    reply = Reply(4.0, calls=(Call(0.5, CLICK, "a"),), cancellations=((1.0, ("a", "x")),))
    sim = _run(World(replies=[reply], execute_s=1.5))
    assert sim.statuses() == ["applied"]
    assert sim.resolved()[0]["finished_at"] == 2.0
    # The executing click is recorded as such; no action arrived in call x.
    assert sim.data("CallCancelled") == [
        {"index": 0, "generation": 1, "call_id": "a", "status": None}
    ]


# ---- 连续 3 次读失败 ----------------------------------------------------------------------------


def test_three_failed_reads_in_a_row_invalidate_the_run():
    error = "CDP Runtime.evaluate timed out"
    sim = _run(World(check_errors={2: error, 3: error, 4: error}))
    assert sim.checks == [0.0, 1.0, 2.0, 3.0]
    assert sim.termination == Termination(
        "invalid_run",
        "invalid",
        3.0,
        3.0,
        3.0,
        3.0,
        Invalidated("environment", "terminal_state_check_failed", error),
    )


def test_a_read_that_answers_starts_the_count_over():
    error = "CDP Runtime.evaluate timed out"
    sim = _run(World(check_errors={2: error, 3: error, 5: error, 6: error}), 6.0)
    assert sim.termination.stop_reason == "time_limit"


# ---- 回复在截止那一刻结束 --------------------------------------------------------------------------


def test_a_reply_that_ends_at_the_deadline_is_discarded():
    """The deadline is a control entry and is handled first: the reply is given up, and its result,
    due at the same instant, is discarded. Its click at 2 landed."""
    sim = _run(World(replies=[Reply(3.0, calls=(Call(2.0, CLICK),))]), 3.0)
    assert _from(sim, "EpisodeEnded") == [
        "EpisodeEnded",
        "GenerationAbandoned",
        "GenerationFinished",
        "ResultDiscarded",
        "SnapshotFrozen",
        "Termination",
    ]
    assert (sim.accepted, sim.discarded) == ([], [1])
    assert sim.statuses() == ["applied"]
    assert sim.termination == Termination("time_limit", "deadline", 3.0, 3.0, 3.0, 3.0, None)


# ---- seeded random scripts ----------------------------------------------------------------------

ACTIONS = (
    CLICK,
    TYPE,
    KEY,
    Action.wait(1.0),
    Action.hold_key("a", 0.75),
    Action(ActionType.DONE),
)
# Environment frame rate and input frame rate: equal, apart, and a fast grid.
RATES = ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (30.0, 1.0), (30.0, 7.0))


def _random_config(rng: random.Random) -> StreamingConfig:
    env_fps, input_fps = rng.choice(RATES)
    return StreamingConfig(
        rng.choice((5.0, 8.0, 12.0)),
        env_fps=env_fps,
        input_fps=input_fps,
        reply_watchdog_s=rng.choice((1.5, 3.0, 30.0)),
        push_watchdog_s=rng.choice((1.0, 2.5, 10.0)),
    )


def _random_world(rng: random.Random, config: StreamingConfig) -> World:
    """Captures and pushes are cheap, so replies start and hand actions over most of the time;
    faults, slow operations and stuck pushes are rare."""
    duration_s = config.duration_s
    reads = int(duration_s * config.env_fps) + 8

    def rare(value, p=0.1):
        return value if rng.random() < p else None

    def faults(n: int, p: float) -> dict:
        return {i: f"fault {i}" for i in range(1, n) if rng.random() < p}

    def slow(n: int, p: float, times) -> dict:
        return {i: rng.choice(times) for i in range(1, n) if rng.random() < p}

    # Now and then the environment wedges for three reads in a row.
    first = rng.randrange(1, reads)
    stretch = {i: "wedged" for i in range(first, first + 3)} if rng.random() < 0.1 else {}
    replies, calls_made = [], 0
    for _ in range(rng.randrange(0, 16)):
        latency = rng.choice((0.25, 0.5, 1.0, 2.0, 3.0, 5.0))
        calls = []
        for _ in range(rng.randrange(0, 5)):
            calls_made += 1
            calls.append(
                Call(
                    rng.uniform(0.0, latency),
                    rng.choice(ACTIONS),
                    rng.choice((None, f"c{calls_made}")),
                    grounding=rare(MISSING_FRAME),
                    execute_s=rng.choice((None, 0.25, 1.0, 2.5)),
                    rejection=rare("occluded"),
                    error=rare("guest down", 0.02),
                )
            )
        cancellations = []
        for _ in range(rng.randrange(0, 3)):
            # A cancellation names calls that came before it.
            at = rng.uniform(0.0, latency)
            ids = [call.call_id for call in calls if call.call_id is not None and call.at <= at]
            if ids:
                cancellations.append((at, tuple(rng.sample(ids, rng.randrange(1, len(ids) + 1)))))
        replies.append(
            Reply(
                latency,
                tuple(sorted(calls, key=lambda call: call.at)),
                tuple(cancellations),
                protocol_failure=rare("no_action_found", 0.2),
                invalid=rare(Invalidated("provider", "provider_request_failed"), 0.02),
                output_tokens=rng.choice((None, 40)),
                interruptible=rng.random() < 0.7,
            )
        )
    return World(
        replies=replies,
        ends_at=rare(rng.uniform(1.0, duration_s), 0.4),
        end_reason=rare("success", 0.5),
        start_s=rng.choice((0.0, 0.0, 0.3)),
        posts=((rng.uniform(0.0, duration_s), Invalidated("environment", "x")),)
        if rng.random() < 0.03
        else (),
        capture_s=rng.choice((0.0, 0.0, 0.125, 0.25)),
        push_s=rng.choice((0.0, 0.0, 0.125, 0.25)),
        check_s=rng.choice((0.0, 0.0, 0.125, 0.5)),
        execute_s=rng.choice((0.0, 0.25, 0.5, 1.0)),
        commit_s=rng.choice((0.0, 0.0, 0.25)),
        snapshot_s=rng.choice((0.0, 0.5)),
        capture_times=slow(reads, 0.02, (1.5, 2.5)),
        push_times=slow(reads, 0.03, (1.5, 12.0)),
        check_times=slow(reads, 0.04, (1.5, 2.5, 4.0)),
        capture_errors=faults(reads, 0.005),
        push_errors=faults(reads, 0.05),
        check_errors=faults(reads, 0.06) | stretch,
        stuck_pushes={rng.randrange(1, 12)} if rng.random() < 0.05 else set(),
        commit_errors=faults(reads, 0.01),
        release_errors={"a": "xdotool failed"} if rng.random() < 0.03 else {},
        snapshot_error=rare("snapshot failed", 0.03),
        frozen_from=rare(rng.randrange(0, 6), 0.03),
    )


def _random_run(seed: int) -> Simulation:
    rng = random.Random(seed)
    config = _random_config(rng)
    return Simulation(config, _random_world(rng, config)).run()


@pytest.mark.parametrize("seed", range(300))
def test_seeded_random_scripts_keep_the_invariants(seed):
    """The simulation checks every command as it runs and the whole episode once it ends."""
    assert _random_run(seed).termination is not None


def test_the_random_scripts_reach_the_states_that_matter():
    """How many of the 300 scripts reach each state; the floors are about half of what they
    reach, so that a change to the scripts that stops reaching one fails here."""
    reached: dict[str, int] = dict.fromkeys(COVERAGE, 0)
    for seed in range(300):
        sim = _random_run(seed)
        for state, test in COVERAGE.items():
            reached[state] += bool(test(sim))
    floors = {
        "a reply started": 250,
        "an action executed": 150,
        "an action superseded": 70,
        "a waiting call cancelled": 30,
        "an action from a reply given up": 30,
        "a frame that cannot be found": 50,
        "the task's end": 40,
        "three failed reads in a row": 4,
        "a reply's watchdog": 60,
        "a session rebuilt for its frames": 10,
        "an end handled past the deadline": 25,
        "an action that ended the task": 12,
        "an action the end kept from executing": 30,
    }
    assert {state: reached[state] >= floor for state, floor in floors.items()} == dict.fromkeys(
        floors, True
    ), reached


def _abandoned(sim: Simulation) -> list[str]:
    return [data["reason"] for data in sim.data("GenerationAbandoned")]


COVERAGE = {
    "a reply started": lambda sim: sim.started,
    "an action executed": lambda sim: sim.executed,
    "an action superseded": lambda sim: "superseded" in sim.statuses(),
    "a waiting call cancelled": lambda sim: "cancelled_call" in sim.statuses(),
    "an action from a reply given up": lambda sim: "cancelled_generation" in sim.statuses(),
    "a frame that cannot be found": lambda sim: "invalid_grounding_frame" in sim.statuses(),
    "the task's end": lambda sim: sim.termination.cause == "environment",
    "three failed reads in a row": lambda sim: (
        sim.termination.invalid is not None
        and sim.termination.invalid.reason == "terminal_state_check_failed"
    ),
    "a reply's watchdog": lambda sim: "reply_watchdog" in _abandoned(sim),
    "a session rebuilt for its frames": lambda sim: (
        {"push_watchdog", "push_failed"} & set(_abandoned(sim))
    ),
    "an end handled past the deadline": lambda sim: (
        sim.entries("EpisodeEnded")[0].semantic > sim.config.duration_s
    ),
    "an action that ended the task": lambda sim: (
        sim.termination.cause == "environment"
        and any(data["terminal_after_action"] for data in sim.data("ActionResolved"))
    ),
    "an action the end kept from executing": lambda sim: any(
        status.startswith("not_applied") for status in sim.statuses()
    ),
}
