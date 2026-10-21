"""Replay baselines for run_stream when the environment and model-input frame rates are not 1/1.

main() passes run_stream the environment tick (delta_s = 1/env fps), the model-input frame in
environment ticks (frame_ticks = env fps / model-input fps) and the horizon in environment ticks
(evaluation/runner.py main). Frame j is due at j * frame_ticks and observed at the first whole
tick at or after it; on the finite token clock B is charged per model-input frame, so n output
tokens advance n / B * frame_ticks ticks and the remaining budget is the remaining model-input
frames times B (run_stream's _apply and _decide).

Token clock: the openrouter agent against the real GuestEnvironment on a task clock
(token_env.py), at 60/1, 30/7 and 2/2 with a finite B, and at 60/1 in pause mode. Wall clock: the
chat_completions agent against FakeEnvironment (fake_env.py) at the same three rates. Each case
asserts the observation ticks, where each action landed, the remaining budget and where the
horizon cut the episode, then compares the whole run with tests/replay/baselines/
frame_rates-<clock>-<env fps>_<model-input fps>.jsonl. Record with
``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay/test_runner_frame_rates.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation import runner

from . import baseline
from .build import build_episode
from ..fake_clock import FakeClock
from .clock_task import ClockTask
from .fake_env import FakeEnvironment
from .frames import CLOCK_ORIGIN, ReplayTask
from .loop import REAL_TIME_FIELDS, check_wall_clock_properties, commit_statuses, records, run_loop
from .scenario import CLICK_AGREE, CLICK_NAME, CLICK_SUBMIT, TASK_INSTRUCTION
from .stub import ChatCompletionsStub, Reply
from .stub_openrouter import OpenRouterStub
from .test_chat_completions import MODEL_KEY as CHAT_MODEL_KEY
from .test_chat_completions import chat_completions_config
from .test_openrouter import MODEL_KEY as OPENROUTER_MODEL_KEY
from .test_openrouter import NATIVE_OUTPUT_LIMIT, ROUTE, openrouter_env  # noqa: F401 (fixture)
from .token_env import TokenEnvironment

BASELINES = Path(__file__).parent / "baselines"
# On the token clock the decision latency is real time, but δ, the think clock and the capture
# time GuestEnvironment measures (GuestEnvironment.capture) are not masked by loop.py.
TOKEN_REAL_TIME_FIELDS = baseline.REAL_TIME_FIELDS | {
    "latency_s",
    "model_latency_s",
    "capture_latency_s",
}

TAB = '{"action": "key", "key": "tab"}'
TYPE_ADA = '{"action": "type", "text": "Ada"}'


def wait(seconds: float) -> str:
    return f'{{"action": "wait", "duration": {seconds}}}'


def reply(text: str, output_tokens: int) -> Reply:
    """An OpenAI-shaped usage whose completion count is the output the token clock charges."""
    usage = {
        "prompt_tokens": 1500,
        "completion_tokens": output_tokens,
        "total_tokens": 1500 + output_tokens,
    }
    return Reply(text, usage)


@dataclass(frozen=True)
class TokenCase:
    name: str
    env_fps: float
    model_input_fps: float
    tokens_per_frame: int | None
    max_minutes: float
    replies: list[Reply]
    # Expected, per decision:
    ticks: list  # the observation's tick
    remaining: list  # the remaining budget run_stream binds (None in pause mode)
    action_ticks: list  # tau_k, where the action lands
    statuses: list
    next_ticks: list  # the next observation's tick, or where the horizon stopped the clock
    dropped: list  # model-input frames the decision spanned without observing them
    stop_reason: str = "tick_limit"


TOKEN_CASES = [
    # 60/1, B=160, horizon 0.09375 min = 5.625 s = 337.5 ticks; one frame is 60 ticks. Decision 0:
    # 80 tokens are half a frame, 30 ticks; the tap holds the key for one tick (1/60 s). Decision
    # 1: 240 tokens, 90 ticks, land at 150, past frame 2 (tick 120): the next frame is 3. Decision
    # 2 lands exactly on frame 4. Decision 3's wait adds 0.5 s, 30 ticks, after its 12. Decision
    # 4 asks for 130 tokens with 37.5 ticks (0.625 frames, 100 tokens) left: clamped to 100, it
    # lands on the horizon and is not applied.
    TokenCase(
        "60_1",
        60.0,
        1.0,
        160,
        0.09375,
        [
            reply(TAB, 80),
            reply(CLICK_NAME, 240),
            reply(CLICK_AGREE, 160),
            reply(wait(0.5), 32),
            reply(CLICK_SUBMIT, 130),
        ],
        ticks=[0, 60, 180, 240, 300],
        remaining=[900, 740, 420, 260, 100],
        action_ticks=[30, 150, 240, 252, 337.5],
        statuses=["applied", "applied", "applied", "wait", "not_applied_tick_limit"],
        next_ticks=[60, 180, 240, 300, 337.5],
        dropped=[0, 1, 0, 0, 0],
    ),
    # 30/7, B=70, horizon 0.03125 min = 1.875 s = 56.25 ticks = 13.125 frames; frame j is due at
    # 30j/7 ticks and observed at 0, 5, 9, 13, 18, 22, 26, 30, 35, 39, ... Decision 0: 35 tokens,
    # half a frame, land at 15/7 ticks; frame 1 is observed at tick 5. Decision 1: two frames from
    # tick 5 land at 13.57, past frame 3 (12.86), and the tap adds 1/60 s, half a tick: frame 4,
    # tick 18. Decision 2 waits 0.1 s, 3 ticks, after a frame: 25.29, before frame 6 (25.71).
    # Decision 3: 139 tokens land at 34.51, past frame 8's due time 34.29 though frame 8 would be
    # observed at tick 35, so the next frame is 9. Decision 4: four frames land at 56.14, inside
    # the horizon; frame 14 (tick 60) is past it, so the clock stops at 56.25.
    TokenCase(
        "30_7",
        30.0,
        7.0,
        70,
        0.03125,
        [
            reply(CLICK_NAME, 35),
            reply(TAB, 140),
            reply(wait(0.1), 70),
            reply(CLICK_AGREE, 139),
            reply(CLICK_SUBMIT, 280),
        ],
        ticks=[0, 5, 18, 26, 39],
        remaining=[918, 837, 624, 494, 281],
        action_ticks=[15 / 7, 5 + 60 / 7, 18 + 30 / 7, 26 + 139 * 3 / 49, 39 + 120 / 7],
        statuses=["applied", "applied", "wait", "applied", "applied"],
        next_ticks=[5, 18, 26, 39, 56.25],
        dropped=[0, 2, 1, 2, 3],
    ),
    # 2/2, B=100, horizon 0.03125 min = 1.875 s = 3.75 ticks. Decision 0: one frame, then the tap
    # holds the key for 1/60 s, 1/30 tick, past frame 1: the next frame is 2. Decision 1: half a
    # frame and a 0.25 s wait end exactly on frame 3. Decision 2 asks for 200 tokens with 75 left
    # and lands on the horizon.
    TokenCase(
        "2_2",
        2.0,
        2.0,
        100,
        0.03125,
        [reply(TAB, 100), reply(wait(0.25), 50), reply(CLICK_NAME, 200)],
        ticks=[0, 2, 3],
        remaining=[375, 175, 75],
        action_ticks=[1, 2.5, 3.75],
        statuses=["applied", "wait", "not_applied_tick_limit"],
        next_ticks=[2, 3, 3.75],
        dropped=[1, 0, 0],
    ),
    # Pause mode at 60/1, horizon 0.0625 min = 3.75 s = 225 ticks: each action lands on its own
    # frame and the world then moves to the next model-input frame, 60 ticks on. Decision 1's
    # 1.5 s wait ends at tick 150, past frame 2: the next frame is 3. Decision 2's next frame,
    # tick 240, is past the horizon.
    TokenCase(
        "pause-60_1",
        60.0,
        1.0,
        None,
        0.0625,
        [reply(CLICK_NAME, 80), reply(wait(1.5), 40), reply(CLICK_AGREE, 500)],
        ticks=[0, 60, 180],
        remaining=[None, None, None],
        action_ticks=[0, 60, 180],
        statuses=["applied", "wait", "applied"],
        next_ticks=[60, 180, 225],
        dropped=[0, 1, 0],
    ),
]


def _decision_values(env, event: str, field_name: str) -> list:
    return [
        call["payload"][field_name]
        for call in env.calls
        if call["call"] == "trace" and call["event"] == event
    ]


def _lifecycle(env) -> list[dict]:
    return [call["record"] for call in env.calls if call["call"] == "env.record_lifecycle"]


def _compare(name: str, header: dict, actual: list[dict]) -> None:
    path = BASELINES / f"frame_rates-{name}.jsonl"
    if os.environ.get("REPLAY_RECORD") == "1":
        baseline.write(path, actual)
        return
    problems = baseline.compare(baseline.read(path), actual)
    assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
        problems[:60]
    )


@pytest.mark.usefixtures("openrouter_env")
@pytest.mark.parametrize("case", TOKEN_CASES, ids=[case.name for case in TOKEN_CASES])
def test_token_clock(case, monkeypatch, tmp_path):
    entry = catalog.get(OPENROUTER_MODEL_KEY)
    config = catalog.to_model_config(
        OPENROUTER_MODEL_KEY,
        fps=case.model_input_fps,
        reasoning_effort=entry.defaults["reasoning_effort"],
    )
    task = ClockTask(  # as main() configures the task clock
        case.env_fps,
        runner.task_clock_tokens_per_frame(
            case.tokens_per_frame, case.env_fps, case.model_input_fps
        ),
    )
    with OpenRouterStub(case.replies, route=ROUTE) as stub:
        monkeypatch.setenv("OPENROUTER_BASE_URL", stub.base_url)
        episode = build_episode(
            config,
            tmp_path,
            task=task,
            instruction=TASK_INSTRUCTION,
            model_key=OPENROUTER_MODEL_KEY,
            clock_mode="token",
            tokens_per_frame=case.tokens_per_frame,
            env_fps=case.env_fps,
        )
        env = TokenEnvironment(episode.frames, task, max_minutes=case.max_minutes)
        timeline = run_loop(episode, env)
    assert stub.unscripted == 0 and stub.unused_replies == 0

    max_ticks = case.max_minutes * 60 * case.env_fps
    observed = _decision_values(env, TraceEvent.OBSERVATION_CAPTURED.value, "tick")
    assert observed == case.ticks
    assert [t_k for _, t_k, _, _, _ in timeline] == case.ticks
    # Model-visible time: the observation's world seconds.
    stamps = _decision_values(env, TraceEvent.OBSERVATION_CAPTURED.value, "timestamp")
    assert stamps == pytest.approx([tick / case.env_fps for tick in case.ticks])
    lifecycle = _lifecycle(env)
    assert [r["remaining_clock_tokens"] for r in lifecycle] == case.remaining
    # Each request's cap is that budget, or the native limit in pause mode
    # (providers/litellm.py transport_output_token_cap).
    assert [json.loads(request.body)["max_tokens"] for request in stub.requests] == [
        NATIVE_OUTPUT_LIMIT if cap is None else cap for cap in case.remaining
    ]
    assert [tau for _, _, _, tau, _ in timeline] == pytest.approx(case.action_ticks)
    # The token delay is in model-input frames: the output charged over B, none in pause mode.
    assert [r["token_delay_frames"] for r in lifecycle] == [
        0.0 if case.tokens_per_frame is None else r["clock_output_tokens"] / case.tokens_per_frame
        for r in lifecycle
    ]
    assert [r["status"] for r in lifecycle] == case.statuses
    assert [r["next_observation_tick"] for r in lifecycle] == pytest.approx(case.next_ticks)
    assert [r["dropped_intermediate_frames"] for r in lifecycle] == case.dropped
    assert env._stop_reason == case.stop_reason
    assert env.tick_index() == pytest.approx(min(case.next_ticks[-1], max_ticks))
    # The task clock moved with the environment, never more than one tick per step
    # (GuestEnvironment.wait_until_tick): its world time at each capture is the observation's
    # tick in seconds, and at the end where the episode stopped.
    advances = [event for event in task.events if event["event"] == "advance"]
    assert all(event["frames"] <= 1.0 + 1e-9 for event in advances)
    task_s, at_capture = 0.0, []
    for call in env.calls:
        if call["call"] == "task.advance":
            task_s = call["to_s"]
        elif call["call"] == "env.capture":
            at_capture.append(task_s)
    assert at_capture == pytest.approx([tick / case.env_fps for tick in case.ticks])
    assert task.world_time() == pytest.approx(min(case.next_ticks[-1], max_ticks) / case.env_fps)

    header = {
        "record": "header",
        "api": "openrouter",
        "clock": "token",
        "env_fps": case.env_fps,
        "model_input_fps": case.model_input_fps,
        "tokens_per_frame": case.tokens_per_frame,
        "max_minutes": case.max_minutes,
        "config": config.model_dump(),
    }
    actual = baseline.Normalizer(TOKEN_REAL_TIME_FIELDS)(
        [header, *records(env, episode.agent, timeline), *baseline.trajectory(tmp_path)]
    )
    _compare(f"token-{case.name}", header, actual)


@pytest.mark.usefixtures("openrouter_env")
def test_a_failed_clock_advance_keeps_the_environment_s_reason(monkeypatch, tmp_path):
    """60/1, B=160: decision 0's 80 tokens land its tab at tick 30, and the task clock fails to
    advance from 0.75 s (tick 45) on, as the world moves to frame 1. GuestEnvironment latches
    clock_advance_failed and raises (GuestEnvironment.step_clock); the shell logs the exception
    and keeps it as the summary's error, as the legacy loop let it escape, and answers the step
    with it (shell.py _failure). The kernel gives it the environment's reason (kernel/turns.py
    _step_failed), so the run keeps it. The episode still ends through the snapshot, and then the
    decision in hand is reported."""
    entry = catalog.get(OPENROUTER_MODEL_KEY)
    config = catalog.to_model_config(
        OPENROUTER_MODEL_KEY, fps=1.0, reasoning_effort=entry.defaults["reasoning_effort"]
    )
    task = ClockTask(
        60.0,
        runner.task_clock_tokens_per_frame(160, 60.0, 1.0),
        advance_fails=(0.75, "page clock did not advance"),
    )
    with OpenRouterStub([reply(TAB, 80)], route=ROUTE) as stub:
        monkeypatch.setenv("OPENROUTER_BASE_URL", stub.base_url)
        episode = build_episode(
            config,
            tmp_path,
            task=task,
            instruction=TASK_INSTRUCTION,
            model_key=OPENROUTER_MODEL_KEY,
            clock_mode="token",
            tokens_per_frame=160,
            env_fps=60.0,
        )
        env = TokenEnvironment(episode.frames, task, max_minutes=0.09375)
        run_loop(episode, env)
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert env._invalid_run == {
        "category": "environment",
        "reason": "clock_advance_failed",
        "detail": "page clock did not advance",
    }
    assert [status for _, status, _ in commit_statuses(env)] == ["applied"]
    # The kernel freezes the snapshot before it reports the decision in hand.
    assert [
        call["call"] for call in env.calls if call["finalizing"] and call["call"] != "trace"
    ] == [
        "task.evaluation_snapshot",
        "agent.commit",
        "env.record_lifecycle",
        "trace.checkpoint",
        "runner.log",
        "env.release_inputs",
    ]
    assert env._termination_snapshot_captured
    assert env._termination == {
        "stop_reason": "invalid_run",
        "end_s": 0.75,
        "snapshot_s": 0.75,
        "snapshot_delay_s": 0.0,
    }
    logged = [call["line"] for call in env.calls if call["call"] == "runner.log"]
    assert any(
        line.startswith("[invalid-run] environment:clock_advance_failed")
        and line.rstrip().endswith("RuntimeError: page clock did not advance")
        for line in logged
    ), logged
    assert any(line.startswith("[default] decision loop raised:") for line in logged)
    assert "RuntimeError: page clock did not advance" in env._loop_error


@dataclass(frozen=True)
class WallCase:
    name: str
    env_fps: float
    model_input_fps: float
    max_minutes: float
    replies: list[str]
    latencies: list[float]
    ticks: list
    times: list  # the observation's seconds into the episode
    # Expected for each decision the shell reports: a generation the deadline cuts is not.
    action_ticks: list
    statuses: list
    next_ticks: list
    dropped: list
    stop_reason: str = "time_limit"
    env: dict = field(default_factory=dict)


WALL_CASES = [
    # 60/1, deadline 3.75 s: one frame a second, 60 ticks. Decision 1 replies after 1.25 s, at
    # tick 135, past frame 2: frame 3 at 3 s. Decision 2's reply would arrive after the deadline,
    # where the kernel gives the generation up.
    WallCase(
        "60_1",
        60.0,
        1.0,
        0.0625,
        [CLICK_NAME, TYPE_ADA, CLICK_SUBMIT],
        [0.5, 1.25, 0.875],
        ticks=[0, 60, 180],
        times=[0.0, 1.0, 3.0],
        action_ticks=[30, 135],
        statuses=["applied", "applied"],
        next_ticks=[60, 180],
        dropped=[0, 1],
    ),
    # 30/7, deadline 0.9375 s = 28.125 ticks: frames observed at ticks 0, 5, 9, 13, 18, 22, 26.
    # Decision 1 replies at tick 11, past frame 2 (8.57): frame 3, tick 13. Decision 3's reply
    # would arrive after the deadline.
    WallCase(
        "30_7",
        30.0,
        7.0,
        0.015625,
        [CLICK_NAME, TYPE_ADA, CLICK_AGREE, CLICK_SUBMIT],
        [0.1, 0.2, 0.05, 0.5],
        ticks=[0, 5, 13, 18],
        times=[0.0, 1 / 6, 13 / 30, 0.6],
        action_ticks=[3, 11, 14.5],
        statuses=["applied", "applied", "applied"],
        next_ticks=[5, 13, 18],
        dropped=[0, 1, 0],
    ),
    # 2/2, deadline 1.875 s = 3.75 ticks, one frame per tick. Decision 2's reply would arrive
    # after the deadline.
    WallCase(
        "2_2",
        2.0,
        2.0,
        0.03125,
        [CLICK_NAME, TYPE_ADA, CLICK_SUBMIT],
        [0.25, 0.75, 0.5],
        ticks=[0, 1, 3],
        times=[0.0, 0.5, 1.5],
        action_ticks=[0.5, 2.5],
        statuses=["applied", "applied"],
        next_ticks=[1, 3],
        dropped=[0, 1],
    ),
]


@pytest.mark.parametrize("case", WALL_CASES, ids=[case.name for case in WALL_CASES])
def test_wall_clock(case, tmp_path):
    clock = FakeClock(wall_origin=CLOCK_ORIGIN)
    env_box = []
    with (
        ChatCompletionsStub(
            [reply(text, 8) for text in case.replies],
            on_request=lambda index: env_box[0].model_request(index),
            clock=clock,
        ) as stub,
        clock.participant(),
    ):
        config = chat_completions_config(stub.base_url, stream=True).model_copy(
            update={"fps": case.model_input_fps}
        )
        episode = build_episode(
            config,
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=CHAT_MODEL_KEY,
            clock_mode="wall",
            clock=clock,
            env_fps=case.env_fps,
        )
        env = FakeEnvironment(
            episode.frames,
            clock,
            fps=case.env_fps,
            max_minutes=case.max_minutes,
            reply_latencies=case.latencies,
            **case.env,
        )
        env_box.append(env)
        timeline = run_loop(episode, env)
    assert stub.unscripted == 0 and stub.unused_replies == 0
    check_wall_clock_properties(env, case.stop_reason)
    assert [status for _, status, _ in commit_statuses(env)] == case.statuses

    captures = [call for call in env.calls if call["call"] == "env.capture"]
    assert [call["tick"] for call in captures] == case.ticks
    assert [call["time"] - env.t0 for call in captures] == pytest.approx(case.times)
    assert [tau for _, _, _, tau, _ in timeline] == pytest.approx(case.action_ticks)
    lifecycle = [call["record"] for call in env.calls if call["call"] == "env.record_lifecycle"]
    assert [r["status"] for r in lifecycle] == case.statuses
    assert [r["next_observation_tick"] for r in lifecycle] == pytest.approx(case.next_ticks)
    assert [r["dropped_intermediate_frames"] for r in lifecycle] == case.dropped

    header = {
        "record": "header",
        "api": "chat_completions",
        "clock": "wall",
        "env_fps": case.env_fps,
        "model_input_fps": case.model_input_fps,
        "max_minutes": case.max_minutes,
        "reply_latencies": case.latencies,
        "config": config.model_dump(),
    }
    actual = baseline.Normalizer(REAL_TIME_FIELDS)(
        [header, *records(env, episode.agent, timeline), *baseline.trajectory(tmp_path)]
    )
    _compare(f"wall-{case.name}", header, actual)
