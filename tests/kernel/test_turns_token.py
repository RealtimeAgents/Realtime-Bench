"""The turn-based policy on the token clock, with a finite B and paused, run in the discrete-event
simulation.

World time stands still while the model generates. The action lands at t_k + n/B after the clock
was advanced there in steps of at most one tick, each followed by an end check; the next frame is
the first boundary after both the observed frame and the action, never past the horizon. The skip
order, key leases, waits, the finite-B allowance and the operational watchdog decide the rest, and
the command stream keeps the four turn-based limits.

Every expected value is worked out by hand: from today's runner where the kernel keeps its rules
(evaluation/runner.py, sandbox/actions.py) and from phase2.md (事件内核; 帧率; 时间和结束判定) where it
changes them. Times are environment ticks; env_fps = input_fps = 1 and B = 256 unless a test says
otherwise, so a frame is a tick and n tokens are n/256 ticks.
"""

from __future__ import annotations

import pytest

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.evaluation.kernel.events import (
    AbortGeneration,
    AcceptResult,
    AdvanceClock,
    Capture,
    CheckEnd,
    Commit,
    Execute,
    Invalidated,
    Outcome,
    Record,
    ReleaseKeys,
    RenderFrame,
    StartGeneration,
)
from realtime_framework.evaluation.kernel.turns import TurnConfig
from tests.kernel.simulation import Reply, Simulation, World

B = 256
CLICK = Action.click(10, 20)
TAP = Action(ActionType.KEY, {"key": "space"})


def _token(duration_s: float, tokens_per_frame: int | None = B, **fields) -> TurnConfig:
    return TurnConfig("token", duration_s, tokens_per_frame=tokens_per_frame, **fields)


def _run(config: TurnConfig, *replies: Reply, **world) -> Simulation:
    return Simulation(config, World(replies=list(replies), **world)).run()


def _idle(frames: float) -> Reply:
    """No action; its output moves the clock ``frames`` frames."""
    return Reply(output_tokens=round(frames * B))


def _asked(sim: Simulation) -> list[float]:
    """The ticks at which the environment was asked whether the episode is over."""
    return [tick for tick, _ in sim.checks]


def _names(commands) -> list[str]:
    return [c.entry.kind if isinstance(c, Record) else type(c).__name__ for c in commands]


def _record(sim: Simulation, kind: str, decision: int, phase: str | None = None) -> dict:
    """The one trace record of ``kind`` (and clock ``phase``) for ``decision``."""
    (data,) = [
        entry.data
        for entry in sim.entries(kind)
        if entry.data.get("decision") == decision and entry.data.get("phase") == phase
    ]
    return data


def _approach(sim: Simulation, decision: int) -> dict:
    return _record(sim, "ClockAdvanced", decision, "inference_to_action")


def _pace(sim: Simulation, decision: int) -> dict:
    return _record(sim, "ClockAdvanced", decision, "action_to_observation_boundary")


def _inference_to_action(decision, from_tick, requested, to_tick, tokens, per_frame=B) -> dict:
    """The record of the move to the action time (runner.py:555-568)."""
    return {
        "decision": decision,
        "phase": "inference_to_action",
        "clock_mode": "token",
        "from_tick": from_tick,
        "requested_tick": requested,
        "to_tick": to_tick,
        "delta_ticks": to_tick - from_tick,
        "output_tokens": tokens,
        "tokens_per_frame": per_frame,
    }


def _to_boundary(decision, from_tick, next_tick, rendered, dropped) -> dict:
    """The record of a completed move to the next frame (runner.py:626-639)."""
    return {
        "decision": decision,
        "phase": "action_to_observation_boundary",
        "clock_mode": "token",
        "from_tick": from_tick,
        "requested_tick": next_tick,
        "to_tick": next_tick,
        "delta_ticks": next_tick - from_tick,
        "post_action_rendered": rendered,
        "dropped_intermediate_frames": dropped,
    }


def _outcome(
    decision: int,
    status: str,
    *,
    tau: float,
    delay: float,
    next_tick: float,
    dropped: int,
    tokens: int | None,
    rendered: bool = False,
    terminal: bool = False,
    detail: str = "",
    post_horizon: int = 0,
) -> Outcome:
    """An outcome as today's runner reports it. ``tau`` is the action clock position and ``delay``
    the token delay in frames, which at env_fps 1 is also the frame age (runner.py:648-655)."""
    return Outcome(
        decision=decision,
        status=status,
        detail=detail,
        terminal_after_action=terminal,
        action_clock_position=tau,
        next_observation_tick=next_tick,
        dropped_intermediate_frames=dropped,
        post_action_rendered=rendered,
        frame_age_s=delay,
        token_delay_frames=delay,
        action_output_tokens=tokens,
        post_horizon_output_tokens=post_horizon,
    )


def _clock(sim: Simulation) -> list:
    """Clock steps, key releases and end checks in the order the kernel asked for them."""
    steps: list = []
    for command in sim.commands:
        if isinstance(command, AdvanceClock):
            steps.append(("advance", command.step))
        elif isinstance(command, ReleaseKeys):
            steps.append(("release", command.keys))
        elif isinstance(command, CheckEnd):
            steps.append("check")
    return steps


# ---- A1 section 1.3, the worked examples ------------------------------------------------------


def test_an_action_crossing_a_boundary_lands_at_its_token_time_and_drops_the_crossed_frame():
    # t_k = 10 after an idle decision; n = 300: tau = 10 + 300/256 = 11.171875, reached in steps of
    # 1.0 and 0.171875. boundary = ceil(11.171875) = 12, next = min(max(10 + 1, 12), 20) = 12,
    # dropped = 12 - 10 - 1 = 1 (runner.py:600-625).
    sim = _run(_token(20), _idle(10), Reply(action=CLICK, output_tokens=300), _idle(8))
    assert _approach(sim, 1) == _inference_to_action(1, 10, 11.171875, 11.171875, 300)
    assert sim.executed == [(1, 11.171875, CLICK)]
    assert sim.advances == [1.0] * 10 + [1.0, 0.171875, 0.828125] + [1.0] * 8
    assert _pace(sim, 1) == _to_boundary(1, 11.171875, 12, False, 1)
    assert sim.renders == []
    assert sim.outcomes[1] == _outcome(
        1, "applied", tau=11.171875, delay=1.171875, next_tick=12, dropped=1, tokens=300
    )


def test_an_action_landing_exactly_on_a_boundary_renders_before_the_next_frame():
    # n = 256: tau = 11 = boundary, next = max(11, 11) = 11. The pace move is empty and the action
    # was applied, so the frame is rendered at 11 (runner.py:615-623); dropped = 11 - 10 - 1 = 0.
    sim = _run(_token(20), _idle(10), Reply(action=CLICK, output_tokens=256), _idle(9))
    assert sim.executed == [(1, 11, CLICK)]
    assert sim.renders == [11]
    assert _pace(sim, 1) == _to_boundary(1, 11, 11, True, 0)
    assert sim.outcomes[1] == _outcome(
        1, "applied", tau=11.0, delay=1.0, next_tick=11, dropped=0, tokens=256, rendered=True
    )


def test_a_key_tap_on_an_exact_boundary_moves_the_next_frame_without_a_render():
    # n = 256: tau = 11, but the tap holds the key 1/60 tick (sandbox/actions.py:513), so the
    # boundary of max(11, 11 + 1/60) is 12 and next = 12; dropped = 12 - 10 - 1 = 1.
    tapped = 11 + 1 / 60
    sim = _run(_token(20), _idle(10), Reply(action=TAP, output_tokens=256), _idle(8))
    assert sim.executed == [(1, 11, TAP)]
    assert _record(sim, "ActionResolved", 1)["environment_tick"] == tapped
    assert _pace(sim, 1) == _to_boundary(1, tapped, 12, False, 1)
    assert sim.renders == []
    assert sim.outcomes[1] == _outcome(
        1, "applied", tau=11.0, delay=1.0, next_tick=12, dropped=1, tokens=256
    )


@pytest.mark.parametrize("tokens", [256, 300, 1000], ids=["exactly-b", "over-b", "far-over-b"])
def test_an_action_reaching_the_horizon_is_clamped_and_not_applied(tokens):
    # T = 20, t_k = 19: remaining = floor(1 * 256 + 1e-9) = 256 (runner.py:867-870). Output past it
    # is billed, not charged (runner.py:983-993): tau = 19 + 256/256 = 20 = T. The check before
    # landing finds tick >= T (runner.py:528-531); nothing is paced, and the next loop top ends at
    # the horizon (runner.py:842, 802-803).
    sim = _run(_token(20), _idle(19), Reply(action=CLICK, output_tokens=tokens))
    assert sim.entries("InferenceStarted")[1].data["remaining_clock_tokens"] == 256
    assert _record(sim, "ActionProposed", 1)["action_output_tokens"] == 256
    assert _approach(sim, 1) == _inference_to_action(1, 19, 20, 20, 256)
    assert sim.executed == []
    assert _asked(sim) == [0, *range(1, 20), 19, 20, 20, 20]
    assert sim.outcomes[1] == _outcome(
        1,
        "not_applied_tick_limit",
        tau=20.0,
        delay=1.0,
        next_tick=20,
        dropped=0,
        tokens=256,
        post_horizon=tokens - 256,
    )
    termination = sim.termination
    assert (termination.stop_reason, termination.cause, termination.ended_at) == (
        "tick_limit",
        "horizon",
        20,
    )


def test_an_end_reported_at_the_horizon_beats_the_tick_limit():
    # The task ends from tick 20 = T on: the approach's check at 20 finds it before anything tests
    # the horizon, so the environment's reason is the stop reason (runner.py:526-529, 791-803).
    sim = _run(
        _token(20),
        _idle(19),
        Reply(action=CLICK, output_tokens=256),
        ends_at=20,
        end_reason="target_reached",
    )
    assert _asked(sim) == [0, *range(1, 20), 19, 20]
    assert sim.outcomes[1] == _outcome(
        1, "not_applied_terminal", tau=20.0, delay=1.0, next_tick=20, dropped=0, tokens=256
    )
    termination = sim.termination
    assert (termination.stop_reason, termination.cause, termination.ended_at) == (
        "target_reached",
        "environment",
        20,
    )


# ---- end checks and the next frame ------------------------------------------------------------


def test_the_end_is_checked_where_todays_loop_polls_it():
    # T = 5. Decision 0 at 0: n = 384, tau = 1.5, a click, next 2. Decision 1 at 2: n = 768, the
    # whole allowance, tau = 5 = T, not applied; the next loop top ends at the horizon.
    sim = _run(
        _token(5),
        Reply(action=CLICK, output_tokens=384),
        Reply(action=CLICK, output_tokens=768),
    )
    assert _asked(sim) == [
        0,  # loop top (runner.py:842)
        1,  # each approach step (sandbox/actions.py:248)
        1.5,
        1.5,  # before landing (runner.py:528)
        1.5,  # after landing (runner.py:545)
        2,  # each pace step (sandbox/actions.py:248)
        2,  # loop top
        3,  # approach
        4,
        5,
        5,  # before landing, which then finds tick >= T (runner.py:530)
        5,  # loop top, which then ends at the horizon
    ]
    assert [outcome.status for outcome in sim.outcomes] == ["applied", "not_applied_tick_limit"]
    assert sim.termination.stop_reason == "tick_limit"


@pytest.mark.parametrize(
    ("tokens", "tau", "next_tick", "dropped", "rendered"),
    [
        (0, 0.0, 1, 0, False),  # boundary 0, next = max(0 + 1, 0) = 1
        (128, 0.5, 1, 0, False),  # ceil(0.5) = 1
        (300, 1.171875, 2, 1, False),  # ceil(1.171875) = 2
        (768, 3.0, 3, 2, True),  # exactly 3: the pace move is empty
        (1000, 3.90625, 4, 3, False),  # ceil(3.90625) = 4
    ],
    ids=["no-delay", "half-frame", "crossing", "exact", "three-crossed"],
)
def test_the_next_frame_is_the_first_boundary_after_the_action(
    tokens, tau, next_tick, dropped, rendered
):
    # t_k = 0, T = 10: next = min(max(floor(t_k) + 1, boundary(max(tau, tick))), T) and
    # dropped = max(0, floor(next + 1e-9) - floor(t_k) - 1) (runner.py:600-625).
    sim = _run(_token(10), Reply(action=CLICK, output_tokens=tokens), _idle(10 - next_tick))
    assert sim.executed == [(0, tau, CLICK)]
    assert _pace(sim, 0) == _to_boundary(0, tau, next_tick, rendered, dropped)
    assert sim.outcomes[0] == _outcome(
        0,
        "applied",
        tau=tau,
        delay=tau,
        next_tick=next_tick,
        dropped=dropped,
        tokens=tokens,
        rendered=rendered,
    )


def test_the_next_frame_never_passes_the_horizon():
    # T = 5.5, n = 1344: tau = 5.25, boundary 6, next = min(max(1, 6), 5.5) = 5.5 (runner.py:606);
    # dropped = floor(5.5 + 1e-9) - 0 - 1 = 4. The loop top at 5.5 ends at the horizon.
    sim = _run(_token(5.5), Reply(action=CLICK, output_tokens=1344))
    assert sim.executed == [(0, 5.25, CLICK)]
    assert _asked(sim) == [0, 1, 2, 3, 4, 5, 5.25, 5.25, 5.25, 5.5, 5.5]
    assert sim.outcomes == [
        _outcome(0, "applied", tau=5.25, delay=5.25, next_tick=5.5, dropped=4, tokens=1344)
    ]
    assert (sim.termination.stop_reason, sim.termination.ended_at) == ("tick_limit", 5.5)


@pytest.mark.parametrize(
    ("tokens", "first_step"),
    [(10**10 + 1, 1.0), (10**10 - 1, (10**10 - 1) / 10**10)],
    ids=["just-above", "just-below"],
)
def test_an_action_within_1e9_of_a_boundary_counts_as_on_it(tokens, first_step):
    # B = 10**10 puts tau = n/B within 1e-10 of tick 1. The approach stops within 1e-9 of it and
    # the tick snaps to 1 (sandbox/actions.py:219, 245-246); a boundary within 1e-9 of an integer
    # is that integer (runner.py:601-605), so next = max(0 + 1, 1) = 1 and the frame is rendered.
    per_frame = 10**10
    tau = tokens / per_frame
    sim = _run(
        _token(10, per_frame),
        Reply(action=CLICK, output_tokens=tokens),
        Reply(output_tokens=9 * per_frame),
    )
    assert sim.advances[0] == first_step
    assert _approach(sim, 0) == _inference_to_action(0, 0, tau, 1, tokens, per_frame)
    assert sim.executed == [(0, 1, CLICK)]
    assert sim.renders == [1]
    assert sim.outcomes[0] == _outcome(
        0, "applied", tau=1.0, delay=tau, next_tick=1, dropped=0, tokens=tokens, rendered=True
    )


def test_a_decision_without_an_action_still_moves_the_clock_and_paces():
    # Status none asks nothing before or after landing (runner.py:522-523), but the clock still
    # goes to tau = 300/256 and on to next = 2, so an empty decision cannot repeat within one frame
    # (runner.py:592-608).
    sim = _run(_token(10), Reply(output_tokens=300), _idle(8))
    assert _asked(sim) == [0, 1, 1.171875, 2, 2, *range(3, 11), 10]
    assert sim.advances[:3] == [1.0, 0.171875, 0.828125]
    assert (sim.executed, sim.entries("ActionProposed")) == ([], [])
    assert _record(sim, "ActionResolved", 0) == {
        "decision": 0,
        "observation_id": "obs-0",
        "action": None,
        "status": "none",
        "terminal_after_action": False,
        "action_clock_position": 1.171875,
        "environment_tick": 1.171875,
        "execution_result": None,
    }
    assert _pace(sim, 0) == _to_boundary(0, 1.171875, 2, False, 1)
    assert sim.outcomes[0] == _outcome(
        0, "none", tau=1.171875, delay=1.171875, next_tick=2, dropped=1, tokens=300
    )


@pytest.mark.parametrize(
    ("action", "rejection", "status", "renders"),
    [
        (CLICK, None, "applied", [1]),
        (Action.wait(), None, "wait", []),
        (CLICK, "click outside the page", "rejected", []),
        (None, None, "none", []),
    ],
    ids=["applied", "wait", "rejected", "no-action"],
)
def test_only_an_applied_action_on_an_exact_boundary_renders(action, rejection, status, renders):
    # n = 256: tau = 1 = next, so every one of these has an empty pace move; only status applied
    # commits the frame (runner.py:615-623).
    sim = _run(
        _token(10),
        Reply(action=action, output_tokens=256),
        _idle(9),
        rejections={} if rejection is None else {0: rejection},
    )
    assert sim.outcomes[0].status == status
    assert sim.renders == renders
    assert sim.outcomes[0].post_action_rendered is bool(renders)
    assert _pace(sim, 0)["post_action_rendered"] is bool(renders)


@pytest.mark.parametrize(
    ("repeat", "next_tick", "dropped", "rendered"),
    [(1, 2, 1, False), (60, 2, 1, True), (90, 3, 2, False)],
    ids=["one", "sixty", "ninety"],
)
def test_each_key_tap_repeat_moves_the_clock_a_sixtieth_of_a_tick(
    repeat, next_tick, dropped, rendered
):
    # The tap lands at tau = 1 and holds the key 1/60 tick per repeat (sandbox/actions.py:503-514);
    # the boundary is taken from max(tau, tick) = 1 + repeat/60 (runner.py:600). Sixty repeats end
    # exactly on 2, so that pace move is empty and the frame is rendered.
    tap = Action(ActionType.KEY, {"key": "space", "repeat": repeat})
    sim = _run(_token(10), Reply(action=tap, output_tokens=256), _idle(10 - next_tick))
    assert sim.executed == [(0, 1, tap)]
    assert _record(sim, "ActionResolved", 0)["environment_tick"] == pytest.approx(
        1 + repeat / 60, abs=1e-9
    )
    assert sim.renders == ([2] if rendered else [])
    assert sim.outcomes[0] == _outcome(
        0,
        "applied",
        tau=1.0,
        delay=1.0,
        next_tick=next_tick,
        dropped=dropped,
        tokens=256,
        rendered=rendered,
    )


# ---- key leases -------------------------------------------------------------------------------


def test_a_lease_ending_inside_a_move_splits_the_step_and_releases_before_the_check():
    # Decision 0 holds "a" for 1.5 s at tick 0: the lease ends at 0 + 1.5 * env_fps = 1.5
    # (sandbox/actions.py:546-547). Decision 1 lands at tau = 2; its approach from 1 is cut at 1.5,
    # where the key is released before the end check (sandbox/actions.py:221-228, 244-249).
    sim = _run(
        _token(10),
        Reply(action=Action.hold_key("a", 1.5), output_tokens=0),
        Reply(action=CLICK, output_tokens=256),
        _idle(8),
    )
    assert sim.key_events == [(1.5, "up", ("a",))]
    assert _clock(sim) == [
        # decision 0: loop top, before and after landing, the pace move 0 -> 1
        "check",
        "check",
        "check",
        ("advance", 1.0),
        "check",
        # decision 1: loop top, the approach 1 -> 1.5 -> 2, before and after landing
        "check",
        ("advance", 0.5),
        ("release", ("a",)),
        "check",
        ("advance", 0.5),
        "check",
        "check",
        "check",
        # decision 2: loop top, the approach 2 -> 10, loop top
        "check",
        *([("advance", 1.0), "check"] * 8),
        "check",
    ]


def test_a_failed_key_release_invalidates_the_run():
    # Decision 0 holds "a" for 0.5 s at tick 0; its pace move is cut at 0.5, where the release
    # fails (sandbox/actions.py GuestEnvironment._release_clock_keys). The move stops there and the
    # decision is still reported.
    sim = _run(
        _token(10),
        Reply(action=Action.hold_key("a", 0.5), output_tokens=0),
        release_errors={1: "keyUp failed"},
    )
    assert sim.key_events == []
    assert sim.advances == [0.5]
    assert _pace(sim, 0)["to_tick"] == 0.5
    assert [outcome.status for outcome in sim.outcomes] == ["applied"]
    termination = sim.termination
    assert (termination.stop_reason, termination.ended_at) == ("invalid_run", 0.5)
    assert termination.invalid == Invalidated(
        "environment", "clock_key_release_failed", "keyUp failed"
    )


@pytest.mark.parametrize(("again", "released_at"), [(0.5, 2), (3.0, 4)], ids=["shorter", "longer"])
def test_holding_a_held_key_again_only_extends_its_lease(again, released_at):
    # A 2 s hold from tick 0 ends at 2; holding "a" again at tick 1 for ``again`` seconds leaves
    # max(2, 1 + again) (sandbox/actions.py:564-573).
    sim = _run(
        _token(10),
        Reply(action=Action.hold_key("a", 2.0), output_tokens=0),
        Reply(action=Action.hold_key("a", again), output_tokens=0),
        _idle(8),
    )
    assert sim.key_events == [(released_at, "up", ("a",))]
    assert sim.advances == [1.0] * 10


@pytest.mark.parametrize(
    "hold", [1 + 1 / 60, 1.01], ids=["ends-with-the-tap", "ends-inside-the-tap"]
)
def test_a_lease_that_ended_during_a_key_tap_is_released_without_a_check_of_its_own(hold):
    # Decision 0 holds "a" from tick 0 until ``hold``; decision 1 taps another key at tick 1, which
    # moves the clock to 1 + 1/60 inside the environment. The pace move then starts on a lease
    # that has already ended: it is released with no end check (sandbox/actions.py:226-228), and
    # the move goes on to next = 2.
    tapped = 1 + 1 / 60
    sim = _run(
        _token(10),
        Reply(action=Action.hold_key("a", hold), output_tokens=0),
        Reply(action=TAP, output_tokens=0),
        _idle(8),
    )
    assert sim.key_events == [(tapped, "up", ("a",))]
    assert _clock(sim) == [
        # decision 0: loop top, before and after landing, the pace move 0 -> 1
        "check",
        "check",
        "check",
        ("advance", 1.0),
        "check",
        # decision 1: loop top, before landing, after the tap, the pace move 1 + 1/60 -> 2
        "check",
        "check",
        "check",
        ("release", ("a",)),
        ("advance", 2 - tapped),
        "check",
        # decision 2
        "check",
        *([("advance", 1.0), "check"] * 8),
        "check",
    ]
    assert _asked(sim)[:8] == [0, 0, 0, 1, 1, 1, tapped, 2]


# ---- waits ------------------------------------------------------------------------------------


def test_a_wait_passes_its_time_in_checked_steps_without_reaching_the_environment():
    # n = 256: the wait starts at tau = 1 and lets 2.5 s pass, 2.5 ticks at env_fps 1, in steps of
    # at most one tick with a check after each (sandbox/actions.py:843-844, 214-249); next =
    # max(1, ceil(3.5)) = 4 and dropped = 4 - 0 - 1 = 3.
    sim = _run(_token(10), Reply(action=Action.wait(2.5), output_tokens=256), _idle(6))
    assert sim.executed == []
    assert sim.advances[:5] == [1.0, 1.0, 1.0, 0.5, 0.5]
    assert _asked(sim) == [0, 1, 1, 2, 3, 3.5, 3.5, 4, 4, *range(5, 11), 10]
    resolved = _record(sim, "ActionResolved", 0)
    assert (resolved["status"], resolved["environment_tick"]) == ("wait", 3.5)
    assert resolved["execution_result"] == {
        "duration_simulated_seconds": 2.5,
        "requested_duration_simulated_seconds": 2.5,
        "truncated_by_episode_horizon": False,
        "truncated_by_terminal_condition": False,
    }
    assert [entry.data["decision"] for entry in sim.entries("ActionApplied")] == [0]
    assert sim.outcomes[0] == _outcome(
        0, "wait", tau=1.0, delay=1.0, next_tick=4, dropped=3, tokens=256
    )


def test_a_wait_cut_short_by_the_end_reports_the_terminal_condition():
    # The task ends from tick 3 on: the wait from 1 stops at the check at 3, after 2 of its 2.5 s
    # (sandbox/actions.py:248-249, 848-866). The action ended the episode: nothing is paced and no
    # further check is asked.
    sim = _run(
        _token(10),
        Reply(action=Action.wait(2.5), output_tokens=256),
        ends_at=3,
        end_reason="target_reached",
    )
    assert _asked(sim) == [0, 1, 1, 2, 3]
    assert sim.advances == [1.0, 1.0, 1.0]
    assert _record(sim, "ActionResolved", 0)["execution_result"] == {
        "duration_simulated_seconds": 2.0,
        "requested_duration_simulated_seconds": 2.5,
        "truncated_by_episode_horizon": False,
        "truncated_by_terminal_condition": True,
    }
    assert sim.outcomes == [
        _outcome(0, "wait", tau=1.0, delay=1.0, next_tick=3, dropped=0, tokens=256, terminal=True)
    ]
    assert (sim.termination.stop_reason, sim.termination.ended_at) == ("target_reached", 3)


@pytest.mark.parametrize(
    ("replies", "fault", "reason", "failed_at"),
    [
        # The wait from 1 steps to 2; the step to 3, the third advance, fails.
        (
            [Reply(action=Action.wait(2.5), output_tokens=256)],
            {"advance_errors": {3: "renderer crashed"}},
            "clock_advance_failed",
            2,
        ),
        # "a" is held until 2.5; the wait from 1 is cut there, and the release fails.
        (
            [
                Reply(action=Action.hold_key("a", 2.5), output_tokens=0),
                Reply(action=Action.wait(3.0), output_tokens=0),
            ],
            {"release_errors": {1: "renderer crashed"}},
            "clock_key_release_failed",
            2.5,
        ),
    ],
    ids=["advance", "release"],
)
def test_a_wait_whose_own_step_fails_is_a_failed_action(replies, fault, reason, failed_at):
    # The environment lets a wait's time pass inside execute(), which reports a failed step as a
    # failed action: invalid_run, only the error as the execution record, nothing applied, not
    # paced, and still reported (sandbox/actions.py GuestEnvironment.execute; runner.py
    # _land_action).
    sim = _run(_token(10), *replies, **fault)
    wait = len(replies) - 1
    resolved = _record(sim, "ActionResolved", wait)
    assert (resolved["status"], resolved["terminal_after_action"]) == ("invalid_run", False)
    assert resolved["environment_tick"] == failed_at
    assert resolved["execution_result"] == {"error": "renderer crashed"}
    assert [entry.data["decision"] for entry in sim.entries("ActionApplied")] == list(range(wait))
    assert [entry.data["decision"] for entry in sim.entries("ClockAdvanced")].count(wait) == 1
    assert sim.outcomes[wait] == _outcome(
        wait,
        "invalid_run",
        tau=1.0,
        delay=1.0 if wait == 0 else 0.0,
        next_tick=failed_at,
        dropped=0,
        tokens=replies[-1].output_tokens,
    )
    termination = sim.termination
    assert (termination.stop_reason, termination.ended_at) == ("invalid_run", failed_at)
    assert termination.invalid == Invalidated("environment", reason, "renderer crashed")


def test_a_wait_cut_short_by_a_third_failed_read_stays_a_wait():
    # A failed read is no failed step: the wait stops as at an end it sees, and today's
    # environment reports it a wait cut short (sandbox/actions.py GuestEnvironment.done). The
    # check before the wait at 1 and those after its steps to 2 and 3 fail.
    errors = {check: f"check {check} failed" for check in (3, 4, 5)}
    sim = _run(_token(10), Reply(action=Action.wait(3.0), output_tokens=256), check_errors=errors)
    assert _asked(sim) == [0, 1, 1, 2, 3]
    resolved = _record(sim, "ActionResolved", 0)
    assert (resolved["status"], resolved["terminal_after_action"]) == ("wait", True)
    assert resolved["execution_result"] == {
        "duration_simulated_seconds": 2.0,
        "requested_duration_simulated_seconds": 3.0,
        "truncated_by_episode_horizon": False,
        "truncated_by_terminal_condition": True,
    }
    assert [entry.data["decision"] for entry in sim.entries("ActionApplied")] == [0]
    termination = sim.termination
    assert (termination.stop_reason, termination.ended_at) == ("invalid_run", 3)
    assert termination.invalid == Invalidated(
        "environment", "terminal_state_check_failed", "check 5 failed"
    )


def test_a_wait_cut_short_by_the_horizon_reports_the_horizon():
    # T = 5, n = 1024: the wait starts at tau = 4 and asks for 4 + 2.5 > T, so it stops at 5 after
    # 1 s (sandbox/actions.py:217-218, 837-857). The check after it still runs; nothing is paced
    # past T, and the next loop top ends at the horizon.
    sim = _run(_token(5), Reply(action=Action.wait(2.5), output_tokens=1024))
    assert _asked(sim) == [0, 1, 2, 3, 4, 4, 5, 5, 5]
    assert sim.advances == [1.0] * 5
    assert _record(sim, "ActionResolved", 0)["execution_result"] == {
        "duration_simulated_seconds": 1.0,
        "requested_duration_simulated_seconds": 2.5,
        "truncated_by_episode_horizon": True,
        "truncated_by_terminal_condition": False,
    }
    assert sim.outcomes == [
        _outcome(0, "wait", tau=4.0, delay=4.0, next_tick=5, dropped=0, tokens=1024)
    ]
    assert (sim.termination.stop_reason, sim.termination.ended_at) == ("tick_limit", 5)


@pytest.mark.parametrize("wait", [Action.wait(), Action.wait(0.0)], ids=["no-duration", "zero"])
def test_a_wait_of_zero_seconds_does_not_move_the_clock(wait):
    # A wait of 0 s lets no time pass (sandbox/actions.py:832, 842); tau = 1 = next, and a wait is
    # never rendered (runner.py:615-619).
    sim = _run(_token(10), Reply(action=wait, output_tokens=256), _idle(9))
    assert sim.advances == [1.0] * 10
    assert _asked(sim)[:5] == [0, 1, 1, 1, 1]
    assert _record(sim, "ActionResolved", 0)["execution_result"] == {
        "duration_simulated_seconds": 0.0,
        "requested_duration_simulated_seconds": 0.0,
        "truncated_by_episode_horizon": False,
        "truncated_by_terminal_condition": False,
    }
    assert sim.renders == []
    assert sim.outcomes[0] == _outcome(
        0, "wait", tau=1.0, delay=1.0, next_tick=1, dropped=0, tokens=256
    )


# ---- the skip order ---------------------------------------------------------------------------


def test_the_legacy_think_budget_skips_the_action_and_ends_after_its_outcome():
    # Budget 1 s; each decision deliberates 128/256 = 0.5 s (runner.py:490-492, 506). Decision 0
    # lands at 0.5. Decision 1 brings the spend to 1.0 >= 1: skipped_budget, asked nothing before
    # landing (runner.py:524-525), paced from 1.5 to 2; the episode then ends with no loop top
    # (runner.py:1170-1172).
    sim = _run(
        _token(10, think_budget_s=1.0),
        Reply(action=CLICK, output_tokens=128),
        Reply(action=CLICK, output_tokens=128),
    )
    assert _asked(sim) == [0, 0.5, 0.5, 0.5, 1, 1, 1.5, 2]
    assert sim.executed == [(0, 0.5, CLICK)]
    assert sim.outcomes[1] == _outcome(
        1, "skipped_budget", tau=1.5, delay=0.5, next_tick=2, dropped=0, tokens=128
    )
    termination = sim.termination
    assert (termination.stop_reason, termination.cause, termination.ended_at) == (
        "think_budget",
        "think_budget",
        2,
    )


@pytest.mark.parametrize(
    ("tokens", "ends_at", "end_check", "tau", "reached", "asked"),
    [
        (640, 1.0, None, 2.5, 1, [0, 1]),
        (384, None, 4, 1.5, 1.5, [0, 1, 1.5, 1.5]),
    ],
    ids=["during-the-approach", "at-the-check-before-landing"],
)
def test_an_end_found_before_landing_leaves_the_action_unapplied(
    tokens, ends_at, end_check, tau, reached, asked
):
    # The end stops the approach where it is found (sandbox/actions.py:248-249), or is found by the
    # check right before landing (runner.py:528-529): not_applied_terminal, nothing paced
    # (runner.py:595-599), and no check after the end.
    sim = _run(
        _token(10),
        Reply(action=CLICK, output_tokens=tokens),
        ends_at=ends_at,
        ends_from_check=end_check,
        end_reason="target_reached",
    )
    assert _asked(sim) == asked
    assert _approach(sim, 0) == _inference_to_action(0, 0, tau, reached, tokens)
    assert sim.executed == []
    assert sim.outcomes == [
        _outcome(
            0,
            "not_applied_terminal",
            tau=tau,
            delay=tau,
            next_tick=reached,
            dropped=0,
            tokens=tokens,
        )
    ]
    assert (sim.termination.stop_reason, sim.termination.ended_at) == ("target_reached", reached)


def test_a_rejected_action_skips_the_check_after_execution_but_still_paces():
    # n = 384: tau = 1.5. A refused action is not followed by a check (runner.py:533-537), is paced
    # to next = 2 and reports the guest's reason (runner.py:674).
    sim = _run(
        _token(10),
        Reply(action=CLICK, output_tokens=384),
        _idle(8),
        rejections={0: "click outside the page"},
    )
    assert _asked(sim) == [0, 1, 1.5, 1.5, 2, 2, *range(3, 11), 10]
    assert sim.executed == [(0, 1.5, CLICK)]
    assert sim.entries("ActionApplied") == []
    assert sim.outcomes[0] == _outcome(
        0,
        "rejected",
        tau=1.5,
        delay=1.5,
        next_tick=2,
        dropped=1,
        tokens=384,
        detail="click outside the page",
    )


@pytest.mark.parametrize("fault", ["rejections", "execute_errors"], ids=["refused", "failed"])
def test_a_refused_or_failed_action_keeps_the_environments_execution_record(fault):
    # The outcome record carries whatever execution record the environment left, a refusal or a
    # failure included (runner.py:582; sandbox/actions.py:817).
    sim = _run(_token(10), Reply(action=CLICK, output_tokens=256), _idle(9), **{fault: {0: "no"}})
    assert _record(sim, "ActionResolved", 0)["execution_result"] == {"action": "click"}


def test_an_execution_failure_invalidates_the_run_and_is_still_reported():
    # A failed execution invalidates the run (runner.py:538-541) with no check after it and no
    # pacing (runner.py:595-599); unlike an unusable result it is still committed (runner.py:674).
    sim = _run(
        _token(10),
        Reply(action=CLICK, output_tokens=384),
        execute_errors={0: "guest unreachable"},
    )
    assert _asked(sim) == [0, 1, 1.5, 1.5]
    assert sim.outcomes == [
        _outcome(0, "invalid_run", tau=1.5, delay=1.5, next_tick=1.5, dropped=0, tokens=384)
    ]
    termination = sim.termination
    assert (termination.stop_reason, termination.ended_at) == ("invalid_run", 1.5)
    assert termination.invalid == Invalidated(
        "environment", "action_execution_failed", "guest unreachable"
    )


def test_a_failed_clock_advance_invalidates_the_run_and_settles_the_decision():
    # The first approach step fails (sandbox/actions.py:235-241). The decision in hand is settled
    # by the skip order, whose first test is the invalid run (runner.py:520-521), and committed.
    sim = _run(
        _token(10),
        Reply(action=CLICK, output_tokens=384),
        advance_errors={1: "renderer crashed"},
    )
    assert _asked(sim) == [0]
    assert sim.executed == []
    assert _approach(sim, 0) == _inference_to_action(0, 0, 1.5, 0, 384)
    assert sim.outcomes == [
        _outcome(0, "invalid_run", tau=1.5, delay=1.5, next_tick=0, dropped=0, tokens=384)
    ]
    termination = sim.termination
    assert (termination.stop_reason, termination.ended_at) == ("invalid_run", 0)
    assert termination.invalid == Invalidated(
        "environment", "clock_advance_failed", "renderer crashed"
    )


def test_no_frame_is_rendered_after_a_failed_clock_advance():
    # n = 384: the click lands at 1.5 and the pace step to 2, the third advance, fails. The render
    # barrier only follows a pace move that succeeded without moving the clock (runner.py:608-623);
    # a failed one raises first (sandbox/actions.py:235-241), and nothing lands after the end.
    sim = _run(
        _token(10),
        Reply(action=CLICK, output_tokens=384),
        advance_errors={3: "renderer crashed"},
    )
    assert sim.termination.invalid.reason == "clock_advance_failed"
    assert sim.renders == []
    assert [outcome.post_action_rendered for outcome in sim.outcomes] == [False]


@pytest.mark.parametrize(
    ("tokens", "status", "next_tick", "dropped", "rendered", "executed"),
    [
        (256, "applied", 1, 0, True, [(0, 1, CLICK)]),
        (384, "skipped_timeout", 2, 1, False, []),
    ],
    ids=["at-the-cap", "over-the-cap"],
)
def test_a_move_over_the_per_move_cap_is_skipped_but_still_moves_the_clock(
    tokens, status, next_tick, dropped, rendered, executed
):
    # On the token clock the legacy cap is held against token-derived seconds n/B/env_fps
    # (runner.py:302-305, 422-432): 1.5 s > 1 s drops the action, which was still proposed, and the
    # clock still goes to tau = 1.5 and on to next = 2.
    tau = tokens / B
    sim = _run(
        _token(10, per_move_cap_s=1.0),
        Reply(action=CLICK, output_tokens=tokens),
        _idle(10 - next_tick),
    )
    assert [entry.data["action"] for entry in sim.entries("ActionProposed")] == [CLICK]
    assert sim.executed == executed
    assert _approach(sim, 0) == _inference_to_action(0, 0, tau, tau, tokens)
    assert sim.outcomes[0] == _outcome(
        0,
        status,
        tau=tau,
        delay=tau,
        next_tick=next_tick,
        dropped=dropped,
        tokens=tokens,
        rendered=rendered,
    )


# ---- failed state reads -----------------------------------------------------------------------


def _tap_episode(checks: tuple[int, ...] = (), taps: tuple[int, ...] = ()) -> Simulation:
    """A tap landing at tick 1 (n = 256), then a decision that idles to T = 10. End checks 1 to 6
    are the loop top at 0, after the approach step to 1, before the tap, after it at 1 + 1/60,
    after the pace step to 2 and the next loop top; the tap checks once inside, at 1 + 1/60.
    ``checks`` and ``taps`` name the reads that fail."""
    return _run(
        _token(10),
        Reply(action=TAP, output_tokens=256),
        _idle(8),
        check_errors={check: f"check {check} failed" for check in checks},
        tap_check_errors={tap: f"tap check {tap} failed" for tap in taps},
    )


@pytest.mark.parametrize(
    ("checks", "taps"),
    [((1,), ()), ((1, 2), ()), ((2, 3, 4, 5), ()), ((3,), (1,))],
    ids=["one", "two", "the-tap-answers", "one-in-the-tap"],
)
def test_fewer_than_three_failed_reads_in_a_row_leave_the_run_valid(checks, taps):
    # A failed read answers "not over" and the episode goes on as without it. The tap's own read
    # counts like any other: an answer starts the count over, a failure adds to it.
    sim = _tap_episode(checks, taps)
    assert _asked(sim) == _asked(_tap_episode())
    assert sim.tap_checks == [1 + 1 / 60]
    assert [outcome.status for outcome in sim.outcomes] == ["applied", "none"]
    assert sim.entries("RunInvalidated") == []
    assert (sim.termination.stop_reason, sim.termination.invalid) == ("tick_limit", None)


@pytest.mark.parametrize(
    ("checks", "taps", "asked", "outcomes", "ended_at", "detail"),
    [
        # Before the tap, then inside it: the run is invalid as the tap's result arrives, and the
        # tap is settled as applied and terminal without the check after it.
        ((2, 3), (1,), [0, 1, 1], [("applied", True)], 1 + 1 / 60, "tap check 1 failed"),
        # Before the tap, inside it and right after it.
        ((3, 4), (1,), [0, 1, 1, 1 + 1 / 60], [("applied", True)], 1 + 1 / 60, "check 4 failed"),
        # After the tap, after the pace step and at the next loop top, which settles nothing.
        ((4, 5, 6), (), [0, 1, 1, 1 + 1 / 60, 2, 2], [("applied", False)], 2, "check 6 failed"),
    ],
    ids=["in-the-tap", "around-the-tap", "after-the-tap"],
)
def test_the_third_failed_read_in_a_row_invalidates_the_run(
    checks, taps, asked, outcomes, ended_at, detail
):
    sim = _tap_episode(checks, taps)
    assert _asked(sim) == asked
    assert sim.executed == [(0, 1, TAP)]
    assert [(o.status, o.terminal_after_action) for o in sim.outcomes] == outcomes
    termination = sim.termination
    assert (termination.stop_reason, termination.ended_at) == ("invalid_run", ended_at)
    assert termination.invalid == Invalidated("environment", "terminal_state_check_failed", detail)


def test_a_tap_that_sees_the_end_stops_and_the_check_after_it_ends_the_episode():
    # Paused at 120 fps, a tap at tick 0 is two ticks. The task ends at tick 1, which the tap's
    # first check sees: the environment stops there (sandbox/actions.py:279-280), and the check
    # right after execution, at the same tick, reports the end. The loop top and the check before
    # the tap are at 0.
    sim = _run(
        _token(10, None, env_fps=120), Reply(action=TAP), ends_at=1, end_reason="target_reached"
    )
    assert sim.tap_checks == [1]
    assert _asked(sim) == [0, 0, 1]
    assert [(o.status, o.terminal_after_action) for o in sim.outcomes] == [("applied", True)]
    assert sim.termination.stop_reason == "target_reached"
    assert sim.termination.ended_at == 1


# ---- the finite-B allowance and missing usage -------------------------------------------------


@pytest.mark.parametrize(
    ("duration_s", "allowances"),
    [(10, [2560, 2048]), (10.5, [2688, 2176])],
    ids=["whole-horizon", "fractional-horizon"],
)
def test_the_allowance_is_the_frames_left_times_b(duration_s, allowances):
    # remaining = floor((T - t_k) * B + 1e-9) at t_k = 0 and, after a click paced to 2, at t_k = 2
    # (runner.py:867-870); the second decision spends exactly that and lands on T.
    sim = _run(
        _token(duration_s),
        Reply(action=CLICK, output_tokens=300),
        Reply(output_tokens=allowances[1]),
    )
    assert [e.data["remaining_clock_tokens"] for e in sim.entries("InferenceStarted")] == allowances
    starts = [c for c in sim.commands if isinstance(c, StartGeneration)]
    assert [start.remaining_clock_tokens for start in starts] == allowances
    assert (sim.outcomes[1].action_clock_position, sim.outcomes[1].post_horizon_output_tokens) == (
        duration_s,
        0,
    )


def test_with_no_allowance_left_the_clock_runs_to_the_horizon_without_a_generation():
    # T = 5 + 1/512. Decision 0 spends floor(T * 256 + 1e-9) = 1280 tokens and lands on 5, its next
    # frame; at t_k = 5, remaining = floor(256/512 + 1e-9) = 0, so that frame starts no generation
    # and the clock is advanced to T with a check (runner.py:871-873).
    horizon = 5 + 1 / 512
    sim = _run(_token(horizon), _idle(5))
    assert [tick for tick, _ in sim.captures] == [0, 5]
    assert [start.decision for start in sim.commands if isinstance(start, StartGeneration)] == [0]
    assert sim.advances == [1.0] * 5 + [1 / 512]
    assert _asked(sim) == [0, 1, 2, 3, 4, 5, 5, horizon]
    assert len(sim.outcomes) == 1
    termination = sim.termination
    assert (termination.stop_reason, termination.cause, termination.ended_at) == (
        "tick_limit",
        "horizon",
        horizon,
    )


@pytest.mark.parametrize("action", [CLICK, None], ids=["proposed", "no-action"])
def test_missing_output_usage_invalidates_a_finite_b_run_before_anything_is_paced(action):
    # Finite B needs the output count (runner.py:480-487): the run is invalid before the clock
    # moves, nothing is paced or committed, and a proposed action still gets its outcome record.
    sim = _run(_token(10), Reply(action=action, output_tokens=None))
    assert sim.termination.invalid == Invalidated(
        "provider",
        "missing_output_token_usage",
        "finite token-clock decisions require provider or deterministic local usage",
    )
    assert (sim.termination.stop_reason, sim.termination.ended_at) == ("invalid_run", 0)
    assert _asked(sim) == [0]
    assert (sim.advances, sim.outcomes) == ([], [])
    assert not any(isinstance(command, Commit) for command in sim.commands)
    resolved = [
        {
            "decision": 0,
            "observation_id": "obs-0",
            "action": CLICK,
            "status": "invalid_run",
            "terminal_after_action": False,
        }
    ]
    assert [entry.data for entry in sim.entries("ActionResolved")] == (resolved if action else [])


def test_missing_output_usage_is_allowed_when_paused():
    # Unlimited B charges no tokens, so it needs no count (runner.py:493-499).
    sim = _run(_token(10, None), Reply(action=CLICK), ends_at=1)
    assert sim.termination.invalid is None
    assert sim.outcomes == [
        _outcome(0, "applied", tau=0.0, delay=0.0, next_tick=1, dropped=0, tokens=None)
    ]


# ---- paused -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "status", "advances", "next_tick", "dropped"),
    [
        (CLICK, "applied", [1.0], 1, 0),
        (TAP, "applied", [1 - 1 / 60], 1, 0),
        (Action.wait(2.5), "wait", [1.0, 1.0, 0.5, 0.5], 3, 2),
    ],
    ids=["click", "key-tap", "wait"],
)
def test_a_paused_action_lands_on_the_observed_tick(action, status, advances, next_tick, dropped):
    # Unlimited B: tau = t_k whatever the output (runner.py:493-499), so 5000 tokens move nothing;
    # the next frame is max(t_k + 1, boundary(tick)): 1 after a click or a tap, 3 after a 2.5 s
    # wait, with 2 frames dropped (A1 section 1.4). The task ends once that frame is reached.
    sim = _run(_token(10, None), Reply(action=action, output_tokens=5000), ends_at=next_tick)
    assert sim.entries("InferenceStarted")[0].data["remaining_clock_tokens"] is None
    assert _approach(sim, 0) == _inference_to_action(0, 0, 0, 0, 5000, None)
    assert sim.advances == advances
    assert sim.outcomes == [
        _outcome(0, status, tau=0.0, delay=0.0, next_tick=next_tick, dropped=dropped, tokens=5000)
    ]


# ---- the watchdog and the proposal ------------------------------------------------------------


@pytest.mark.parametrize(
    ("latency", "stop_reason", "statuses"),
    [(299.0, "tick_limit", ["applied", "none"]), (301.0, "invalid_run", [])],
    ids=["inside", "past"],
)
def test_the_provider_watchdog_bounds_a_generation_at_300_operational_seconds(
    latency, stop_reason, statuses
):
    # The token clock bounds a call only by the provider watchdog, 300 s by default
    # (runner.py:310-311); past it the run is invalid (runner.py:396-404).
    sim = _run(_token(10), Reply(latency=latency, action=CLICK, output_tokens=256), _idle(9))
    timer = sim.entries("TimerSet")[0].data
    assert (timer["kind"], timer["clock"], timer["at"]) == ("generation_watchdog", "ops", 300.0)
    assert sim.termination.stop_reason == stop_reason
    assert [outcome.status for outcome in sim.outcomes] == statuses


def test_a_watchdog_timeout_abandons_the_generation_and_discards_its_late_result():
    # The result is due at 301 s. At 300 s the run is invalid (runner.py:396-404); the snapshot is
    # frozen and the generation written off before it is interrupted (phase2.md, 事件内核). The 5 s
    # snapshot lets the late result arrive, and it is discarded unseen.
    sim = _run(_token(10), Reply(latency=301.0, action=CLICK, output_tokens=256), snapshot_s=5.0)
    termination = sim.termination
    assert termination.invalid == Invalidated(
        "provider", "generation_watchdog_timeout", "generation exceeded 300s"
    )
    assert (termination.ended_at, termination.ops_ended_at) == (0, 300.0)
    assert sim.snapshots == [(0, 305.0)]
    names = _names(sim.commands)
    assert names.index("FreezeSnapshot") < names.index("AbortGeneration")
    assert names.index("GenerationAbandoned") < names.index("AbortGeneration")
    assert [generation for generation, _ in sim.aborted] == [1]
    assert (sim.accepted, sim.discarded) == ([], [1])
    assert [entry.data["generation"] for entry in sim.entries("ResultDiscarded")] == [1]
    assert (sim.entries("ActionProposed"), sim.outcomes, sim.executed) == ([], [], [])


@pytest.mark.parametrize("verb", [ActionType.DONE, ActionType.FAIL], ids=["done", "fail"])
def test_a_done_or_fail_proposal_becomes_a_wait_without_a_proposal_record(verb):
    # The loop has no terminal actions: the proposal becomes a wait tagged as a protocol failure,
    # which is not recorded as proposed (runner.py:1143-1163).
    sim = _run(_token(10), Reply(action=Action(verb), output_tokens=256), _idle(9))
    assert sim.entries("ActionProposed") == []
    resolved = _record(sim, "ActionResolved", 0)
    assert (resolved["action"], resolved["status"]) == (Action.wait(), "wait")
    assert sim.executed == []
    assert sim.outcomes[0] == _outcome(
        0, "wait", tau=1.0, delay=1.0, next_tick=1, dropped=0, tokens=256
    )


@pytest.mark.parametrize(
    ("reply", "cap", "status", "action", "protocol_failure"),
    [
        (Reply(action=CLICK, output_tokens=256), None, "applied", CLICK, None),
        (
            Reply(action=Action(ActionType.DONE), output_tokens=256),
            None,
            "wait",
            Action.wait(),
            "unsupported_terminal_action",
        ),
        (
            Reply(action=Action.wait(), output_tokens=256, protocol_failure="empty_action_output"),
            None,
            "wait",
            Action.wait(),
            "empty_action_output",
        ),
        (_idle(1), None, "none", None, None),
        # 1.5 s over a 1 s cap: no action, so no protocol failure either.
        (
            Reply(action=Action(ActionType.DONE), output_tokens=384),
            1.0,
            "skipped_timeout",
            None,
            None,
        ),
    ],
    ids=["click", "done", "unparsed", "none", "done-over-the-cap"],
)
def test_the_outcome_names_the_action_the_decision_came_to(
    reply, cap, status, action, protocol_failure
):
    # The agent's feedback reports what was executed: a DONE is reported as the wait it became,
    # with the protocol failure that made it one (runner.py:1218-1228).
    sim = _run(_token(10, per_move_cap_s=cap), reply, _idle(9))
    outcome = sim.outcomes[0]
    assert (outcome.status, outcome.action, outcome.protocol_failure) == (
        status,
        action,
        protocol_failure,
    )


def test_a_decisions_records_follow_todays_trace_order():
    # A1 H13 without the records the kernel does not write (runner.py:921-1165, 555-639).
    sim = _run(_token(10), Reply(action=CLICK, output_tokens=300), _idle(8))
    kinds = {
        "InferenceStarted",
        "ActionProposed",
        "ClockAdvanced",
        "ActionResolved",
        "ActionApplied",
    }
    trace = [
        (entry.kind, entry.data.get("phase"))
        for entry in sim.records
        if entry.kind in kinds and entry.data["decision"] == 0
    ]
    assert trace == [
        ("InferenceStarted", None),
        ("ActionProposed", None),
        ("ClockAdvanced", "inference_to_action"),
        ("ActionResolved", None),
        ("ActionApplied", None),
        ("ClockAdvanced", "action_to_observation_boundary"),
    ]


# ---- the four turn-based limits ---------------------------------------------------------------


def _mixed() -> Simulation:
    """Four decisions of different kinds with time-consuming operations, then a frame with no
    allowance left. T = 6 + 1/512; the click lands at 1.171875 and paces to 2, the wait from 2.5
    to 4, the tap at 4 paces to 5, and the empty decision lands on 6."""
    sim = _run(
        _token(6 + 1 / 512),
        Reply(latency=2.0, action=CLICK, output_tokens=300, records=((0.5, "ModelRequestSent"),)),
        Reply(latency=0.5, action=Action.wait(1.5), output_tokens=128),
        Reply(latency=3.0, action=TAP, output_tokens=0),
        Reply(latency=1.0, output_tokens=256),
        capture_s=0.25,
        check_s=0.125,
        execute_s=0.5,
        advance_s=0.0625,
        commit_s=0.25,
    )
    assert [outcome.status for outcome in sim.outcomes] == ["applied", "wait", "applied", "none"]
    assert [tick for tick, _ in sim.captures] == [0, 2, 4, 5, 6]
    return sim


def test_at_most_one_generation_is_in_flight():
    in_flight: set[int] = set()
    started = []
    for command in _mixed().commands:
        if isinstance(command, StartGeneration):
            assert in_flight == set()
            in_flight.add(command.generation)
            started.append(command.generation)
        elif isinstance(command, (AcceptResult, AbortGeneration)):
            in_flight.discard(command.generation)
    assert started == [1, 2, 3, 4]


def test_every_frame_is_handed_to_the_generation_started_on_it():
    commands = _mixed().commands
    world = (Capture, StartGeneration, Execute, AdvanceClock, ReleaseKeys, RenderFrame, Commit)
    following = [
        next(later for later in commands[index + 1 :] if isinstance(later, world))
        for index, command in enumerate(commands)
        if isinstance(command, Capture)
    ]
    # The last frame meets the horizon gate: no allowance, so the clock runs to T instead.
    assert _names(following) == ["StartGeneration"] * 4 + ["AdvanceClock"]
    assert [start.observation_id for start in following[:4]] == ["obs-0", "obs-1", "obs-2", "obs-3"]


def test_nothing_reaches_the_environment_or_the_agent_during_a_generation():
    commands = _mixed().commands
    in_flight = None
    during = []
    for command in commands:
        if isinstance(command, StartGeneration):
            in_flight = command.generation
        elif isinstance(command, (AcceptResult, AbortGeneration)):
            if command.generation == in_flight:
                in_flight = None
        elif in_flight is not None and isinstance(command, (Execute, Commit, Capture)):
            during.append(command)
    assert during == []
    names = _names(commands)
    assert (names.count("Capture"), names.count("Execute"), names.count("Commit")) == (5, 2, 4)


def test_a_generation_starts_only_once_the_previous_outcome_is_answered():
    committed: set[int] = set()
    answered: set[int] = set()
    starts = []
    for command in _mixed().commands:
        if isinstance(command, Commit):
            committed.add(command.outcome.decision)
        elif isinstance(command, Record) and command.entry.kind == "OutcomeCommitted":
            assert command.entry.data.decision in committed
            answered.add(command.entry.data.decision)
        elif isinstance(command, StartGeneration):
            assert answered == set(range(command.decision))
            starts.append(command.decision)
    assert starts == [0, 1, 2, 3]
