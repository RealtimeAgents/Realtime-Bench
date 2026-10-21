"""Frame rates (phase2.md, 帧率): the environment rate is the unit the world moves in, the horizon,
the token clock's step and the wall clock's end-check cadence; the input rate is the grid frames
are taken on and the unit of B. N tokens move the world N / (B * input_fps) seconds, which is
N * env_fps / (B * input_fps) environment ticks. Input frame j sits at j / input_fps seconds, at
the first environment tick at or after it. The allowance is the input frames left times B, and
the horizon is duration_s * env_fps ticks. With both rates equal every expression is today's.

Run at the four acceptance rates env/input: 1/1, 60/1, 30/7 and 2/2. Every expectation is worked
out by hand from those rules.
"""

from __future__ import annotations

import math

import pytest

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.evaluation.kernel.events import (
    AdvanceClock,
    CheckEnd,
    Execute,
    Record,
    ReleaseKeys,
    SetTimer,
    StartGeneration,
    Termination,
)
from realtime_framework.evaluation.kernel.turns import TurnConfig
from tests.kernel.simulation import Reply, Simulation, World
from tests.kernel.test_scenarios import _check_invariants, _semantics

CLICK = Action.click(10, 20)
# A call that never comes back: the token watchdog or the wall deadline ends the episode.
NEVER = Reply(latency=1e6)
IDS = ["1/1", "60/1", "30/7", "2/2"]


def _config(clock: str, rates: tuple[int, int], duration_s: float, tokens_per_frame=None):
    env_fps, input_fps = rates
    return TurnConfig(
        clock,
        duration_s,
        env_fps=env_fps,
        input_fps=input_fps,
        tokens_per_frame=tokens_per_frame,
    )


def _run(config: TurnConfig, replies: list[Reply], **world) -> Simulation:
    sim = Simulation(config, World(replies=replies, **world)).run()
    _check_invariants(sim)
    return sim


def _boundaries(sim: Simulation) -> list[dict]:
    return [
        entry.data
        for entry in sim.entries("ClockAdvanced")
        if entry.data["phase"] == "action_to_observation_boundary"
    ]


# ---- today's expressions, in frames that are also ticks -----------------------------------------


def _today_remaining(max_ticks: float, t_k: float, tokens_per_frame: int) -> int:
    return max(0, math.floor((max_ticks - t_k) * tokens_per_frame + 1e-9))  # runner.py _decide


def _today_tau(t_k: float, tokens: int, tokens_per_frame: int) -> float:
    return t_k + tokens / tokens_per_frame  # runner.py _apply


def _today_next(t_k: float, tau: float, tick: float, max_ticks: float) -> float:
    action_end = max(tau, tick)  # runner.py _apply
    rounded = round(action_end)
    boundary = int(rounded) if abs(action_end - rounded) < 1e-9 else math.ceil(action_end)
    return min(max(math.floor(t_k) + 1, boundary), max_ticks)


def _today_dropped(t_k: float, post_after: float) -> int:
    crossed = max(0, math.floor(post_after + 1e-9) - math.floor(t_k))  # runner.py _apply
    return max(0, crossed - 1)


@pytest.mark.parametrize(
    ("rates", "duration_s"), [((1, 1), 6.0), ((2, 2), 3.0)], ids=["1/1", "2/2"]
)
def test_at_equal_rates_the_rules_are_todays_expressions(rates, duration_s):
    """1/1 is today's loop, and 2/2 is the same in environment ticks (T = 6 both): the allowance,
    the action time, the next observation and the dropped frames are today's expressions; only
    the frame age, ticks / env_fps seconds, depends on the rate."""
    env_fps, _ = rates
    tokens_per_frame, max_ticks = 256, 6
    tokens = [300, 256, 10, 600]
    replies = [Reply(latency=1.0, action=CLICK, output_tokens=n) for n in tokens]
    sim = _run(_config("token", rates, duration_s, tokens_per_frame), replies)
    # (t_k, remaining, tau, next observation, dropped, post-horizon tokens, status)
    expected = [
        # 6 * 256 left; 300 / 256 = 1.171875; ceil(1.171875) = 2; frames 1 and 2 crossed.
        (0, 1536, 1.171875, 2, 1, 0, "applied"),
        # 4 * 256 left; 256 tokens are one frame, so the action is on the boundary at 3.
        (2, 1024, 3.0, 3, 0, 0, "applied"),
        # 3 * 256 left; 3 + 10 / 256.
        (3, 768, 3.0390625, 4, 0, 0, "applied"),
        # 2 * 256 left; 600 is clamped to 512, 88 past the horizon; the action time is T.
        (4, 512, 6.0, 6, 0, 88, "not_applied_tick_limit"),
    ]
    starts = [command for command in sim.commands if isinstance(command, StartGeneration)]
    assert [
        (
            tick,
            start.remaining_clock_tokens,
            outcome.action_clock_position,
            outcome.next_observation_tick,
            outcome.dropped_intermediate_frames,
            outcome.post_horizon_output_tokens,
            outcome.status,
        )
        for (tick, _), start, outcome in zip(sim.captures, starts, sim.outcomes, strict=True)
    ] == expected
    for (t_k, remaining, tau, next_tick, dropped, _, status), n in zip(
        expected, tokens, strict=True
    ):
        assert remaining == _today_remaining(max_ticks, t_k, tokens_per_frame)
        assert tau == _today_tau(t_k, min(n, remaining), tokens_per_frame)
        assert next_tick == _today_next(t_k, tau, min(tau, max_ticks), max_ticks)
        if status == "applied":
            assert dropped == _today_dropped(t_k, next_tick)
    assert [outcome.frame_age_s for outcome in sim.outcomes] == pytest.approx(
        [(tau - t_k) / env_fps for t_k, _, tau, *_ in expected]
    )
    assert [outcome.post_action_rendered for outcome in sim.outcomes] == [False, True, False, False]
    assert sim.renders == [3]
    assert max(sim.advances) <= 1.0
    assert sum(sim.advances) == pytest.approx(max_ticks)
    assert sim.termination == Termination("tick_limit", "horizon", 6.0, 6.0, 4.0, 4.0, None)


@pytest.mark.parametrize(
    ("rates", "tokens_per_frame", "tokens", "tau"),
    [
        ((1, 1), 100, 250, 2.5),  # 250 / 100
        ((60, 1), 100, 100, 60.0),  # one input frame: 1 s, 60 ticks
        ((30, 7), 7, 7, 30 / 7),  # one input frame: 1/7 s, 30/7 ticks
        ((2, 2), 100, 100, 1.0),  # one input frame: 1/2 s, 1 tick
    ],
    ids=IDS,
)
def test_the_action_time_is_in_environment_ticks(rates, tokens_per_frame, tokens, tau):
    """N tokens are N * env_fps / (B * input_fps) environment ticks; the clock steps there, at
    most one tick a step, before the action lands. The frame age is that time in seconds and the
    token delay N / B input frames, as today's loop records it (runner.py _apply)."""
    env_fps, input_fps = rates
    replies = [Reply(latency=1.0, action=CLICK, output_tokens=tokens), NEVER]
    sim = _run(_config("token", rates, 10.0, tokens_per_frame), replies)
    decision, executed_at, _ = sim.executed[0]
    assert (decision, executed_at) == (0, pytest.approx(tau))
    approach = next(
        entry.data
        for entry in sim.entries("ClockAdvanced")
        if entry.data["phase"] == "inference_to_action"
    )
    assert (approach["from_tick"], approach["requested_tick"]) == (0, pytest.approx(tau))
    outcome = sim.outcomes[0]
    assert outcome.action_clock_position == pytest.approx(tau, abs=1e-9)
    assert outcome.frame_age_s == pytest.approx(tokens / (tokens_per_frame * input_fps))
    assert outcome.frame_age_s == pytest.approx(tau / env_fps)
    assert outcome.token_delay_frames == pytest.approx(tokens / tokens_per_frame)
    # Whole ticks, then the fraction left: 1, 1, 0.5; 60 ones; 1, 1, 1, 1, 2/7; 1.
    execute = next(i for i, command in enumerate(sim.commands) if isinstance(command, Execute))
    steps = [
        command.step for command in sim.commands[:execute] if isinstance(command, AdvanceClock)
    ]
    whole = math.floor(tau + 1e-9)
    fraction = [tau - whole] if tau - whole > 1e-9 else []
    assert steps == pytest.approx([1.0] * whole + fraction)
    assert max(sim.advances) <= 1.0


@pytest.mark.parametrize(
    ("rates", "tokens_per_frame", "tokens", "next_tick", "dropped"),
    [
        # tau = 2.5; input frame 3 is tick 3; frames 1 and 2 dropped.
        ((1, 1), 100, 250, 3, 2),
        # tau = 150 * 60 / 100 = 90 ticks = 1.5 s; input frame 2 is at 2 s = tick 120; frame 1
        # dropped.
        ((60, 1), 100, 150, 120, 1),
        # tau = 21 * 30 / 49 = 90/7 ticks = exactly 3 input frames; frame 3 is at 3/7 s, tick
        # 12.86, so it is observed at tick 13; frames 1 and 2 dropped.
        ((30, 7), 7, 21, 13, 2),
        # tau = 250 * 2 / 200 = 2.5 ticks = 2.5 input frames; frame 3 is tick 3.
        ((2, 2), 100, 250, 3, 2),
    ],
    ids=IDS,
)
def test_the_next_observation_is_on_the_input_grid(
    rates, tokens_per_frame, tokens, next_tick, dropped
):
    """The next observation is input frame max(floor(t_k * input / env) + 1, ceil(tau * input /
    env)) at the first environment tick at or after it; the frames crossed but one are dropped."""
    replies = [Reply(latency=1.0, action=CLICK, output_tokens=tokens), NEVER]
    sim = _run(_config("token", rates, 10.0, tokens_per_frame), replies)
    outcome = sim.outcomes[0]
    assert (outcome.next_observation_tick, outcome.dropped_intermediate_frames) == (
        next_tick,
        dropped,
    )
    (boundary,) = _boundaries(sim)
    assert (boundary["requested_tick"], boundary["to_tick"]) == (next_tick, next_tick)
    assert boundary["dropped_intermediate_frames"] == dropped
    assert sim.captures[1][0] == next_tick
    assert sum(sim.advances) == pytest.approx(next_tick)
    assert max(sim.advances) <= 1.0
    assert sim.kernel.policy.dropped_frames == dropped


@pytest.mark.parametrize(
    ("rates", "tokens_per_frame", "duration_s", "first", "remaining", "status", "tau"),
    [
        # T = 5; 5 frames * 100, then 4 * 100 from tick 1; 400 tokens reach T.
        ((1, 1), 100, 5.0, 100, [500, 400], "not_applied_tick_limit", 5.0),
        # T = 120 ticks = 2 input frames; one frame is left at tick 60; 100 tokens reach T.
        ((60, 1), 100, 2.0, 100, [200, 100], "not_applied_tick_limit", 120.0),
        # T = 30 ticks = 7 input frames, 7 * 7 = 49. The first action (one frame, 30/7 ticks) is
        # observed at tick 5, which leaves (30 - 5) * 7/30 = 5.83 frames: floor(5.83 * 7) = 40
        # tokens, 40 * 30/49 = 24.49 ticks, short of T; the response runs past the horizon, so the
        # action lands at T all the same.
        ((30, 7), 7, 1.0, 7, [49, 40], "not_applied_tick_limit", 30.0),
        # T = 6 ticks; 6 frames * 100, then 5 * 100 from tick 1; 500 tokens reach T.
        ((2, 2), 100, 3.0, 100, [600, 500], "not_applied_tick_limit", 6.0),
    ],
    ids=IDS,
)
def test_the_allowance_is_the_remaining_input_frames_times_b(
    rates, tokens_per_frame, duration_s, first, remaining, status, tau
):
    """The allowance is floor((T - t_k) * input / env * B); output past it is clamped to it and
    billed past the horizon, the action lands at T, its delay the input frames left, and the
    horizon, in environment ticks, ends the episode."""
    env_fps, input_fps = rates
    max_ticks = duration_s * env_fps
    replies = [
        Reply(latency=1.0, action=CLICK, output_tokens=first),
        Reply(latency=1.0, action=CLICK, output_tokens=remaining[1] + 10),
    ]
    sim = _run(_config("token", rates, duration_s, tokens_per_frame), replies)
    starts = [command for command in sim.commands if isinstance(command, StartGeneration)]
    assert [start.remaining_clock_tokens for start in starts] == remaining
    second = sim.outcomes[1]
    assert (second.action_output_tokens, second.post_horizon_output_tokens) == (remaining[1], 10)
    assert second.status == status
    assert second.action_clock_position == pytest.approx(tau, abs=1e-9)
    t_k = sim.captures[1][0]
    assert second.token_delay_frames == pytest.approx((max_ticks - t_k) * input_fps / env_fps)
    assert second.next_observation_tick == max_ticks
    assert max(sim.advances) <= 1.0
    assert sim.termination == Termination(
        "tick_limit", "horizon", max_ticks, max_ticks, 2.0, 2.0, None
    )


@pytest.mark.parametrize(
    ("tokens_per_frame", "last"),
    [(None, "applied"), (100, "not_applied_tick_limit")],
    ids=["paused", "finite-b"],
)
def test_a_horizon_within_1e9_of_a_whole_tick_is_that_tick(tokens_per_frame, last):
    """0.17 min at 5 fps is 51.00000000000001 ticks. The environment's clock snaps to tick 51
    within 1e-9 (sandbox/actions.py GuestEnvironment.wait_until_tick), so the horizon is tick 51:
    the loop top there ends the episode instead of observing tick 51 again and again, paused, or
    once more with no allowance left, with finite B. Paused, each action lands on its observed
    tick; with B = 100, one input frame of output puts the last action at T, not applied."""
    config = _config("token", (5, 5), 0.17 * 60, tokens_per_frame)
    assert config.max_ticks == 51
    reply = Reply(latency=0.5, action=CLICK, output_tokens=tokens_per_frame)
    sim = _run(config, [reply] * 52)
    assert [tick for tick, _ in sim.captures] == list(range(51))
    assert [outcome.status for outcome in sim.outcomes] == ["applied"] * 50 + [last]
    assert sim.termination == Termination("tick_limit", "horizon", 51, 51, 25.5, 25.5, None)


@pytest.mark.parametrize(
    ("rates", "duration_s", "observed"),
    [
        ((1, 1), 3.0, [0, 1, 2]),  # T = 3
        ((60, 1), 2.0, [0, 60]),  # one input frame is 60 ticks; T = 120
        ((30, 7), 1.0, [0, 5, 9, 13, 18, 22, 26]),  # ceil(j * 30/7) for j = 0..6; T = 30
        ((2, 2), 2.0, [0, 1, 2, 3]),  # T = 4
    ],
    ids=IDS,
)
def test_paused_mode_advances_one_input_frame_per_decision(rates, duration_s, observed):
    """Paused (B unlimited), each action lands on its observed tick and the clock then moves to
    the next input frame, one tick a step with an end check after each, until T = duration_s *
    env_fps ticks."""
    env_fps, _ = rates
    max_ticks = duration_s * env_fps
    replies = [Reply(latency=0.5, action=CLICK) for _ in observed]
    sim = _run(_config("token", rates, duration_s), replies)
    assert [tick for tick, _ in sim.captures] == observed
    assert [semantic for _, semantic, _ in sim.executed] == observed
    next_ticks = observed[1:] + [max_ticks]
    assert [outcome.next_observation_tick for outcome in sim.outcomes] == next_ticks
    assert [outcome.dropped_intermediate_frames for outcome in sim.outcomes] == [0] * len(observed)
    assert [outcome.frame_age_s for outcome in sim.outcomes] == [0.0] * len(observed)
    assert sim.advances == [1.0] * int(max_ticks)
    # Loop top, before and after execution at t_k; one check per tick to the next frame; the
    # loop top that finds the horizon.
    checks = []
    for t_k, next_tick in zip(observed, next_ticks, strict=True):
        checks += [t_k] * 3 + list(range(t_k + 1, int(next_tick) + 1))
    assert _semantics(sim) == [*checks, max_ticks]
    ops = 0.5 * len(observed)
    assert sim.termination == Termination(
        "tick_limit", "horizon", max_ticks, max_ticks, ops, ops, None
    )


@pytest.mark.parametrize(
    ("rates", "next_tick", "dropped"),
    [
        ((1, 1), 1, 0),
        ((60, 1), 60, 0),  # 1 s is one input frame
        ((30, 7), 30, 6),  # 1 s is 7 input frames, frames 1 to 6 dropped
        ((2, 2), 2, 1),  # 1 s is 2 input frames, frame 1 dropped
    ],
    ids=IDS,
)
def test_a_one_second_wait_is_env_fps_ticks(rates, next_tick, dropped):
    """A 1 s WAIT moves the token clock env_fps ticks, one tick a step with an end check after
    each; the next observation is the first input frame at or after its end."""
    env_fps, _ = rates
    wait = Action.wait(1.0)
    sim = _run(_config("token", rates, 2.0), [Reply(latency=0.5, action=wait)] * 2)
    first = sim.entries("ActionResolved")[0]
    assert (first.data["status"], first.data["environment_tick"]) == ("wait", env_fps)
    assert first.data["execution_result"] == {
        "duration_simulated_seconds": 1.0,
        "requested_duration_simulated_seconds": 1.0,
        "truncated_by_episode_horizon": False,
        "truncated_by_terminal_condition": False,
    }
    outcome = sim.outcomes[0]
    assert (outcome.next_observation_tick, outcome.dropped_intermediate_frames) == (
        next_tick,
        dropped,
    )
    assert sim.advances == [1.0] * (2 * env_fps)
    # Each decision: loop top and before execution, a check per tick, after execution; the
    # second WAIT ends at T, where the next loop top finds the horizon.
    first_checks = [0, 0, *range(1, env_fps + 1), env_fps]
    second_checks = [env_fps, env_fps, *range(env_fps + 1, 2 * env_fps + 1), 2 * env_fps]
    assert _semantics(sim) == [*first_checks, *second_checks, 2 * env_fps]
    assert sim.termination == Termination(
        "tick_limit", "horizon", 2.0 * env_fps, 2.0 * env_fps, 1.0, 1.0, None
    )


@pytest.mark.parametrize(
    ("rates", "up_at"),
    [((1, 1), 0.5), ((60, 1), 30), ((30, 7), 15), ((2, 2), 1)],
    ids=IDS,
)
def test_a_half_second_hold_is_half_env_fps_ticks(rates, up_at):
    """A 0.5 s token-clock hold pressed at tick 0 is a lease of 0.5 * env_fps ticks: the clock
    steps to its end, releases the key and checks the end before it moves on."""
    hold = Action.hold_key("a", 0.5)
    replies = [Reply(latency=0.5, action=hold), *[Reply(latency=0.5, action=CLICK)] * 3, NEVER]
    sim = _run(_config("token", rates, 10.0), replies)
    assert sim.executed[0] == (0, 0, hold)
    assert sim.key_events == [(up_at, "up", ("a",))]
    commands = [command for command in sim.commands if not isinstance(command, Record | SetTimer)]
    release = next(i for i, command in enumerate(commands) if isinstance(command, ReleaseKeys))
    moved = sum(command.step for command in commands[:release] if isinstance(command, AdvanceClock))
    assert moved == pytest.approx(up_at)
    assert isinstance(commands[release + 1], CheckEnd)
    assert max(sim.advances) <= 1.0


@pytest.mark.parametrize(
    ("rates", "tap_checks"),
    [
        ((1, 1), [1 / 60]),
        ((60, 1), [1.0]),
        ((30, 7), [0.5]),
        ((2, 2), [1 / 30]),
        ((120, 1), [1.0, 2.0]),
    ],
    ids=[*IDS, "120/1"],
)
def test_a_key_tap_holds_its_key_a_sixtieth_of_a_second(rates, tap_checks):
    """A token-clock key tap keeps its key down for one 60 Hz frame at any rate: the environment
    moves the clock 1/60 * env_fps ticks inside the action, at most one tick a step with an end
    check after each, and hands the checks' reads to the kernel (sandbox/actions.py TAP_HOLD_S,
    GuestEnvironment._execute_clock_key)."""
    tap = Action(ActionType.KEY, {"key": "space"})
    sim = _run(_config("token", rates, 10.0), [Reply(latency=0.5, action=tap), NEVER])
    assert sim.executed[0] == (0, 0, tap)
    assert sim.tap_checks == pytest.approx(tap_checks, abs=1e-9)
    (executed,) = sim.entries("ActionExecuted")
    assert executed.data.end_checks == (None,) * len(tap_checks)
    tapped = tap_checks[-1]
    resolved = sim.entries("ActionResolved")[0]
    assert resolved.data["environment_tick"] == pytest.approx(tapped, abs=1e-9)
    (boundary,) = _boundaries(sim)
    assert boundary["from_tick"] == pytest.approx(tapped, abs=1e-9)


def test_a_key_tap_stops_at_the_horizon():
    """At 60/1 with B = 120, 119 tokens put a tap at tick 59.5 of T = 60. Its sixtieth of a second
    is one tick, but the environment stops it at the horizon (sandbox/actions.py
    GuestEnvironment.wait_until_tick); nothing is paced past T and the next loop top ends there."""
    tap = Action(ActionType.KEY, {"key": "space"})
    config = _config("token", (60, 1), 1.0, 120)
    sim = _run(config, [Reply(latency=0.5, action=tap, output_tokens=119)])
    assert sim.executed == [(0, 59.5, tap)]
    assert sim.tap_checks == [60]
    resolved = sim.entries("ActionResolved")[0]
    assert (resolved.data["status"], resolved.data["environment_tick"]) == ("applied", 60)
    assert _boundaries(sim) == []
    assert sim.outcomes[0].next_observation_tick == 60
    assert sim.termination == Termination("tick_limit", "horizon", 60, 60, 0.5, 0.5, None)


@pytest.mark.parametrize(
    ("rates", "duration_s", "checks"),
    [
        ((1, 1), 3.0, [0, 1, 2]),
        ((60, 1), 0.1, [k / 60 for k in range(6)]),
        ((30, 7), 0.2, [k / 30 for k in range(6)]),  # the environment rate, not the input rate
        ((2, 2), 2.0, [0, 0.5, 1.0, 1.5]),
    ],
    ids=IDS,
)
def test_wall_end_checks_run_at_the_environment_frame_rate(rates, duration_s, checks):
    """On the wall clock the end is checked every 1/env_fps s, during the generation too; the
    deadline at duration_s beats the check due at the same instant."""
    sim = _run(_config("wall", rates, duration_s), [NEVER])
    assert sim.checks == [(at, at) for at in checks]
    assert sim.termination == Termination(
        "time_limit", "deadline", duration_s, duration_s, duration_s, duration_s, None
    )


@pytest.mark.parametrize(
    ("rates", "next_tick", "dropped"),
    [
        ((1, 1), 1, 0),
        ((60, 1), 60, 0),  # 0.75 s is 0.75 input frames: frame 1 at 1 s = tick 60
        # 0.75 s is 5.25 input frames: frame 6 at 6/7 s = tick 25.71, observed at 26; frames 1 to
        # 5 dropped.
        ((30, 7), 26, 5),
        ((2, 2), 2, 1),  # 0.75 s is 1.5 input frames: frame 2 at 1 s = tick 2
    ],
    ids=IDS,
)
def test_the_wall_clock_observes_on_the_input_grid(rates, next_tick, dropped):
    """On the wall clock the action time is the clock position, 0.75 s = 0.75 * env_fps ticks,
    and the next observation is the first input frame after it, at the first environment tick at
    or after it."""
    env_fps, _ = rates
    sim = _run(_config("wall", rates, 5.0), [Reply(latency=0.75, action=CLICK), NEVER])
    outcome = sim.outcomes[0]
    assert outcome.action_clock_position == 0.75 * env_fps
    assert (outcome.next_observation_tick, outcome.dropped_intermediate_frames) == (
        next_tick,
        dropped,
    )
    assert (outcome.frame_age_s, outcome.token_delay_frames) == (0.75, None)
    assert sim.captures[1] == (next_tick, pytest.approx(next_tick / env_fps))
    assert sim.termination == Termination("time_limit", "deadline", 5.0, 5.0, 5.0, 5.0, None)


@pytest.mark.parametrize(
    ("rates", "second_frame"),
    [
        ((1, 1), 1.0),
        ((60, 1), 1.0),  # the hold lands at tick 15, input frame 0.25: frame 1 is tick 60
        ((30, 7), 0.3),  # tick 7.5 is input frame 1.75: frame 2 at 2/7 s = tick 8.57, so 9
        ((2, 2), 0.5),  # tick 0.5 is input frame 0.5: frame 1 is tick 1
    ],
    ids=IDS,
)
def test_wall_holds_and_waits_last_their_seconds(rates, second_frame):
    """On the wall clock durations are seconds at every rate: a 0.5 s hold pressed at 0.25 s is
    released at 0.75 s by the shell's timer, and a 1 s WAIT that starts 0.25 s after the next
    frame resolves 1 s later."""
    hold, wait = Action.hold_key("a", 0.5), Action.wait(1.0)
    replies = [Reply(latency=0.25, action=hold), Reply(latency=0.25, action=wait), NEVER]
    sim = _run(_config("wall", rates, 5.0), replies)
    (scheduled,) = sim.entries("KeyReleaseScheduled")
    assert scheduled.data == {"decision": 0, "at": 0.75}
    assert sim.key_events == [(0.75, "up", ("a",))]
    assert sim.captures[1][1] == pytest.approx(second_frame)
    resolved = sim.entries("ActionResolved")[1]
    assert resolved.data["status"] == "wait"
    assert resolved.semantic == pytest.approx(second_frame + 0.25 + 1.0)
    assert resolved.data["execution_result"]["duration_simulated_seconds"] == 1.0
