"""run_stream past the deadline, on the fake clock (tests/fake_clock.py).

The kernel's deadline timer (evaluation/kernel/turns.py on_timer), the retry executor's backoff
(entity/model.py RequestRetry) and the kernel's timer for a wall-clock wait (kernel/turns.py
_start_wait) decide these cases; on the fake clock each episode runs the same way every time and
the assertions read the clock. The episode is test_runner_loop's short one: 3.75 s at 1 fps,
decisions at 0, 1, 2 and 3 s. At the deadline the kernel freezes the snapshot and gives up a
generation still running, whose decision is not reported.
"""

from __future__ import annotations

import json
import math

import realtime_framework.entity.model as model_module
from realtime_framework.clock import Clocks
from realtime_framework.entity.model import TRANSPORT_RETRY
from realtime_framework.evaluation import runner
from realtime_framework.sandbox.actions import GuestEnvironment

from ..fake_clock import FakeClock
from .build import build_episode
from .fake_env import FakeEnvironment
from .frames import CLOCK_ORIGIN, ReplayTask
from .loop import check_wall_clock_properties, commit_statuses, run_loop
from .scenario import CLICK_AGREE, CLICK_NAME, CLICK_SUBMIT, TASK_INSTRUCTION, replies
from .steps import Step
from .stub import ChatCompletionsStub, Failure
from .test_chat_completions import MODEL_KEY, chat_completions_config
from .test_runner_loop import SHORT_MAX_MINUTES, TYPE_ADA

FORM = [Step(CLICK_NAME), TYPE_ADA, Step(CLICK_SUBMIT), Step(CLICK_AGREE)]
DEADLINE_S = SHORT_MAX_MINUTES * 60  # 3.75 s after the start
UNAVAILABLE = Failure(503, {"error": {"message": "upstream down", "code": 503}})


def _episode(tmp_path, script, clock):
    config = chat_completions_config(script.base_url, stream=True)
    return build_episode(
        config,
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        model_key=MODEL_KEY,
        clock_mode="wall",
        clock=clock,
    )


def _run_on_fake_environment(tmp_path, script, latencies):
    """test_runner_loop's episode; returns the environment and when each request arrived."""
    clock = FakeClock(wall_origin=CLOCK_ORIGIN)
    environments, arrivals = [], []

    def on_request(index):
        arrivals.append(clock.time() - CLOCK_ORIGIN)
        return environments[0].model_request(index)

    with (
        ChatCompletionsStub(script, on_request=on_request, clock=clock) as stub,
        clock.participant(),
    ):
        episode = _episode(tmp_path, stub, clock)
        env = FakeEnvironment(
            episode.frames,
            clock,
            fps=1.0,
            max_minutes=SHORT_MAX_MINUTES,
            reply_latencies=latencies,
        )
        environments.append(env)
        run_loop(episode, env)
    assert stub.unscripted == 0 and stub.unused_replies == 0
    return env, arrivals


def _ended_at(env) -> float:
    """When the episode ended: the shell stamps it as it freezes the snapshot (shell.py
    _freeze)."""
    return env._active_end_unix_s - env.t0


def test_reply_after_the_deadline_is_dropped(tmp_path):
    """Decision 3's reply takes 3 s and would come at 6.0 s. At the deadline the kernel gives the
    generation up and the shell aborts the agent (shell.py _abort); the decision is not reported.
    The abort closes the request and cancels the decision's generation, so the abandoned worker
    does not send it again (DefaultHarness.abort; entity/model.py RequestRetry)."""
    env, arrivals = _run_on_fake_environment(tmp_path, replies(FORM), [0.5, 0.5, 0.5, 3.0])
    assert [status for _, status, _ in commit_statuses(env)] == ["applied"] * 3
    assert [e["action"]["action_type"] for e in env.executions] == ["click", "type", "click"]
    check_wall_clock_properties(env, "time_limit")
    assert _ended_at(env) == DEADLINE_S
    assert arrivals == [0.0, 1.0, 2.0, 3.0]


def test_retry_backoff_past_the_deadline(tmp_path, monkeypatch):
    """Decision 2's request fails twice with HTTP 503 before any output. The harness backs off
    1 s, then 3 s, each +-25% (entity/model.py TransportRetryPolicy). The second backoff would run
    past the deadline, where the kernel gives the generation up and the shell aborts the agent;
    the abort cancels the decision's generation, which ends the backoff then, and the decision is
    not reported, without a third attempt (entity/model.py RequestRetry)."""
    backoffs = []

    def backoff(clock, cancel, seconds):
        started = clock.monotonic()
        wait(clock, cancel, seconds)
        backoffs.append((seconds, started, clock.monotonic(), clock.time()))

    wait = model_module._backoff
    monkeypatch.setattr(model_module, "_backoff", backoff)
    script = [*replies(FORM[:2]), UNAVAILABLE, UNAVAILABLE]
    env, arrivals = _run_on_fake_environment(tmp_path, script, [0.5, 0.5, 0.0, 0.0])
    assert [status for _, status, _ in commit_statuses(env)] == ["applied"] * 2
    check_wall_clock_properties(env, "time_limit")
    assert _ended_at(env) == DEADLINE_S
    first, second = TRANSPORT_RETRY.backoff_s[:2]
    jitter = TRANSPORT_RETRY.jitter_ratio
    assert len(arrivals) == 4 and arrivals[:3] == [0.0, 1.0, 2.0]
    assert first * (1 - jitter) <= arrivals[3] - 2.0 <= first * (1 + jitter)
    assert arrivals[3] + second * (1 - jitter) > DEADLINE_S
    (delay_1, started_1, ended_1, _), (delay_2, started_2, ended_2, ended_at) = backoffs
    assert ended_1 == started_1 + delay_1  # the first backoff waits its delay out
    assert ended_at == env._active_end_unix_s  # the second ends at the deadline
    assert ended_2 < started_2 + delay_2


class _Guest:
    """The sandbox guest: executes nothing."""

    def execute_action(self, action):
        return {}


class _WallTask(ReplayTask):
    """The form task with a 1 fps task clock, which GuestEnvironment takes its tick from."""

    def clock_snapshot(self) -> dict:
        return {"fps": 1.0}


def test_wait_past_the_deadline_on_guest_environment(tmp_path):
    """GuestEnvironment, which main() hands run_stream on the wall clock. Decision 2 asks for a
    5 s wait when its reply arrives at 2.5 s. The kernel lets the wait's time pass on a timer of
    its own (kernel/turns.py _start_wait), so the wait never reaches GuestEnvironment.execute; the
    deadline cuts it, the decision is reported as a wait, and the episode ends there."""
    clock = FakeClock(wall_origin=CLOCK_ORIGIN)
    steps = [Step(CLICK_NAME), TYPE_ADA, Step('{"action": "wait", "duration": 5}', status="wait")]
    audit = tmp_path / "action_lifecycle.jsonl"
    executed = []
    with (
        ChatCompletionsStub(
            replies(steps), on_request=lambda index: clock.sleep(0.5), clock=clock
        ) as stub,
        clock.participant(),
    ):
        episode = _episode(tmp_path, stub, clock)
        deadline = CLOCK_ORIGIN + SHORT_MAX_MINUTES * 60
        env = GuestEnvironment(
            _Guest(),
            deadline,
            _WallTask(),
            scene_capture=episode.frames,
            action_audit_path=audit,
            clock_mode="wall",
            wall_start_time=CLOCK_ORIGIN,
            clocks=Clocks(clock, clock),
        )
        execute = env.execute

        def recorded_execute(action, **options):
            executed.append(action.action_type.value)
            return execute(action, **options)

        env.execute = recorded_execute
        try:
            timeline = runner.run_stream(
                episode.agent,
                env,
                clock_mode="wall",
                duration_s=SHORT_MAX_MINUTES * 60,
                env_fps=1.0,
                input_fps=1.0,
                log=lambda line: None,
                clocks=Clocks(clock, clock),
            )
        finally:
            env._drain_abandoned()
            episode.agent.close()
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert [action.action_type.value for *_, action in timeline] == ["click", "type", "wait"]
    assert executed == ["click", "type"]
    assert env._stop_reason == "time_limit"
    lifecycle = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [record["status"] for record in lifecycle] == ["applied", "applied", "wait"]
    waited = lifecycle[2]["execution_result"]
    assert waited["requested_duration_simulated_seconds"] == 5.0
    # The wait started when decision 2's reply arrived, 0.5 s after tick 2.
    _, t_k, _, tau_k, _ = timeline[2]
    assert t_k == 2 and tau_k == 2.5
    assert math.isclose(waited["duration_simulated_seconds"], 3.75 - tau_k, abs_tol=1e-6)
    assert math.isclose(env._active_end_unix_s, deadline, abs_tol=1e-6)
