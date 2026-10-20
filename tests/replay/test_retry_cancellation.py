"""An aborted decision's whole-turn retry still reaches the provider.

When run_stream's bounded wait expires it keeps the generate worker in ``abandoned`` and calls the
agent's abort (evaluation/runner.py:337-345). DefaultHarness.abort reaches only the runtime
(harnesses/default/harness.py:1689-1692), which bumps the generation epoch and schedules a
cleanup that closes the driver and clears ``_connected`` (harnesses/default/context.py:662-687).
Nothing reaches the retry executor stream_generate runs (harness.py:1338-1345): its backoff is a
plain ``time.sleep`` (entity/model.py:457) and every attempt calls run_turn, which reads the epoch
afresh (context.py:640). The next attempt therefore reconnects in _ensure_ready
(context.py:394-417), is current, and keeps its turn (context.py:577-590) and its action-history
entry (harness.py:1449-1466).

Each test pins that behaviour as it is today: after the abort, the worker sends one more request,
gets a reply that is not stale, and adds a turn and an action that nobody committed.
"""

from __future__ import annotations

import json
import threading
import time

import realtime_framework.entity.model as model_module
from realtime_framework.entity.model import TRANSIENT, TransportFailure
from realtime_framework.evaluation import runner

from .build import build_episode
from .fake_env import FakeClock, FakeEnvironment, Hang
from .frames import ReplayTask
from .loop import commit_statuses, run_loop
from .scenario import CLICK_NAME, TASK_INSTRUCTION, replies
from .steps import Step
from .stub import ChatCompletionsStub
from .stub_openrouter import Failure, OpenRouterStub
from .test_chat_completions import MODEL_KEY as CHAT_MODEL_KEY
from .test_chat_completions import chat_completions_config
from .test_openai_realtime import MAX_TICKS, TOKENS_PER_FRAME, realtime_config
from .test_openai_realtime import MODEL_KEY as REALTIME_MODEL_KEY
from .test_openrouter import MODEL_KEY as OPENROUTER_MODEL_KEY
from .test_openrouter import (
    ROUTE,
    TOKEN_CLOCK,
    openrouter_config,
    openrouter_env,  # noqa: F401 - fixture
)
from .ws_openai_realtime import RealtimeStub
from .ws_openai_realtime import replies as realtime_replies

WAIT_S = 10.0  # bound on every cross-thread wait; none is reached when the code behaves as pinned
STEP = Step(CLICK_NAME)
UNAVAILABLE = Failure(503, {"error": {"message": "upstream down", "code": 503}})


class _GatedSleep:
    """Stands in for the retry executor's time module. The backoff (entity/model.py:457) records
    its delay, sets ``entered`` and returns only once the test sets ``release``."""

    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def sleep(self, seconds):
        self.calls.append(seconds)
        self.entered.set()
        if not self.release.wait(WAIT_S):
            raise AssertionError("the test never released the backoff")

    def __getattr__(self, name):
        return getattr(time, name)


class _RecordedSleep:
    """Records the backoff and returns at once."""

    def __init__(self):
        self.calls = []

    def sleep(self, seconds):
        self.calls.append(seconds)

    def __getattr__(self, name):
        return getattr(time, name)


def _generate_in_worker(agent, observation, k=0):
    """run_stream's _step (runner.py:917-919) on a daemon thread, as _bounded_call runs it
    (runner.py:327-336)."""
    box, done = {}, threading.Event()

    def worker():
        try:
            agent.stream_prefill(observation)
            box["result"] = agent.stream_generate(tick=k)
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    return box, done


def _abort_like_the_runner(agent) -> int:
    """_bounded_call's timeout branch calls adapter.abort() (runner.py:337-345). Waiting for the
    scheduled cleanup (context.py:682-686) only makes the state below deterministic; the next
    attempt waits for it anyway (context.py:395). Returns the epoch the abort moved to."""
    runtime = agent._runtime
    before = runtime.generation_epoch()
    agent.abort()
    cleanup = runtime._cleanup_future
    assert cleanup is not None
    cleanup.result(timeout=WAIT_S)
    assert runtime.generation_epoch() == before + 1
    assert runtime._connected is False  # context.py:672
    return before + 1


def _finish(box, done):
    assert done.wait(WAIT_S), "the abandoned worker never returned"
    assert "error" not in box, repr(box.get("error"))
    return box["result"]


def _count_calls(target, name: str) -> list:
    """Wrap an async method; each call records its keyword arguments."""
    calls = []
    original = getattr(target, name)

    async def recorded(*args, **kwargs):
        calls.append(kwargs)
        return await original(*args, **kwargs)

    setattr(target, name, recorded)
    return calls


def _trace(run_dir, event: str) -> list[dict]:
    lines = (run_dir / "episode_trace.jsonl").read_text().splitlines()
    return [e["payload"] for e in map(json.loads, lines) if e["event"] == event]


def _attempts(run_dir) -> list[tuple]:
    return [
        (a["scope"]["attempt"], a["outcome"], a.get("stale"))
        for a in _trace(run_dir, "ModelAttemptFinished")
    ]


def _check_uncommitted_turn(agent, result, turn_epochs: list, epoch: int) -> None:
    """Attempt 1 ran under the epoch before the abort; the retry read the abort's epoch at
    context.py:640, so it is current: its turn and its action join the history although
    run_stream discarded the decision."""
    assert [call["epoch"] for call in turn_epochs] == [epoch - 1, epoch]
    assert result.get("stale") is None and result.get("invalid_reason") is None
    assert result["action"] is not None
    assert result["realtime_runtime"]["transport_reconnect_attempts"] == 1
    (turn,) = agent._runtime.turns
    assert turn.output == STEP.reply and turn.execution_feedback == ""
    (entry,) = agent.action_history
    assert "result" not in entry and "requested" not in entry  # commit_action never saw it


def test_abort_during_backoff_still_sends_the_retry_today(openrouter_env, tmp_path):  # noqa: F811
    """Stateless API. Attempt 1 gets HTTP 503 (TRANSIENT, entity/model.py:424-425) and the
    harness sleeps its backoff. The runner aborts the agent during that sleep. When the sleep
    ends the loop sends the same request again, on the abort's epoch, and keeps the reply."""
    sleeps = _GatedSleep()
    openrouter_env.setattr(model_module, "time", sleeps)
    with OpenRouterStub([UNAVAILABLE, *replies([STEP])], route=ROUTE) as stub:
        openrouter_env.setenv("OPENROUTER_BASE_URL", stub.base_url)
        episode = build_episode(
            openrouter_config(),
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=OPENROUTER_MODEL_KEY,
            **TOKEN_CLOCK,
        )
        agent = episode.agent
        try:
            connects = _count_calls(agent.driver, "connect")
            turn_epochs = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            agent.configure_decision(episode.budget(observation.tick_number))
            box, done = _generate_in_worker(agent, observation)
            assert sleeps.entered.wait(WAIT_S), "attempt 1 was not retried"
            assert len(stub.requests) == 1
            epoch = _abort_like_the_runner(agent)
            sleeps.release.set()
            result = _finish(box, done)
        finally:
            episode.finish()
    assert stub.unscripted == 0 and stub.unused_replies == 0
    # One request after the abort: the same body, rebuilt from canonical history.
    failed, retried = stub.requests
    assert failed.body == retried.body
    assert len(sleeps.calls) == 1
    # _ensure_ready reconnected because the abort cleared _connected (context.py:408-417).
    assert len(connects) == 1
    _check_uncommitted_turn(agent, result, turn_epochs, epoch)
    sent = [
        (r["scope"]["decision"], r["scope"]["attempt"])
        for r in _trace(tmp_path, "ModelRequestSent")
    ]
    assert sent == [(0, 1), (0, 2)]
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "ok", False)]


def test_abort_mid_request_is_classified_transient_and_retried_today(tmp_path, monkeypatch):
    """Stateless API, abort while the request is in flight. The stub holds request 0 until the
    client closes it; the abort's driver.close() does (providers/litellm.py:1033-1044). The
    failure arrives before any output, is classified TRANSIENT, and the loop retries it."""
    sleeps = _RecordedSleep()
    monkeypatch.setattr(model_module, "time", sleeps)
    arrived = threading.Event()

    def hold_first(index):
        if index == 0:
            arrived.set()
            return True
        return False

    with ChatCompletionsStub(replies([STEP]), on_request=hold_first) as stub:
        episode = build_episode(
            chat_completions_config(stub.base_url, stream=True),
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=CHAT_MODEL_KEY,
        )
        agent = episode.agent
        try:
            classified = []
            classify = agent.driver.classify_transport_error
            agent.driver.classify_transport_error = lambda exc: (
                classified.append((classify(exc), repr(exc)[:200])) or classified[-1][0]
            )
            turn_epochs = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            agent.configure_decision(episode.budget(observation.tick_number))
            box, done = _generate_in_worker(agent, observation)
            assert arrived.wait(WAIT_S)
            epoch = _abort_like_the_runner(agent)
            result = _finish(box, done)
        finally:
            episode.finish()
    assert stub.held == [True]  # the abort closed the held request
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert len(stub.requests) == 2
    ((failure, error),) = classified
    assert failure == TransportFailure(TRANSIENT), error
    assert len(sleeps.calls) == 1
    _check_uncommitted_turn(agent, result, turn_epochs, epoch)
    assert result["realtime_runtime"]["transport_reconnect_reason"].startswith("transient: ")
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "ok", False)]


class _ReleasingEnvironment(FakeEnvironment):
    """Releases the gated backoff from release_inputs, which _finalize calls after the episode's
    evaluation snapshot (runner.py:806-812), so the retry goes out once the episode is over."""

    def __init__(self, *args, release, **kwargs):
        super().__init__(*args, **kwargs)
        self._release = release

    def release_inputs(self):
        super().release_inputs()
        self._release.set()


def test_runner_timeout_leaves_a_worker_that_retries_after_the_episode_today(tmp_path, monkeypatch):
    """The real run_stream and _bounded_call on a 0.5 s wall-clock episode. Decision 0's request
    never comes back; the bounded wait (0.5 s, the grace set to 0 to keep it short) expires, the
    decision is committed as skipped_timeout and the episode ends. The abandoned worker's
    request was closed by the abort, so it retries; that request reaches the stub during the
    drain, after the evaluation snapshot, and its reply becomes a turn and an action."""
    monkeypatch.setattr(runner, "DEADLINE_GRACE_S", 0.0)
    sleeps = _GatedSleep()
    monkeypatch.setattr(model_module, "time", sleeps)
    env_box = []
    with ChatCompletionsStub(
        replies([STEP]), on_request=lambda index: env_box[0].model_request(index)
    ) as stub:
        config = chat_completions_config(stub.base_url, stream=True)
        episode = build_episode(
            config,
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=CHAT_MODEL_KEY,
            clock_mode="wall",
        )
        env = _ReleasingEnvironment(
            episode.frames,
            FakeClock(),
            fps=config.fps,
            max_minutes=0.5 / 60,
            reply_latencies=[Hang(0.5), 0.0],
            release=sleeps.release,
        )
        env_box.append(env)
        turn_epochs = _count_calls(episode.agent._runtime, "_run_turn")
        timeline = run_loop(episode, env)
    agent = episode.agent
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert stub.held == [True]
    assert len(stub.requests) == 2
    # The runner's abort moved the epoch from 0 to 1 between the two attempts.
    assert [call["epoch"] for call in turn_epochs] == [0, 1]
    assert commit_statuses(env) == [(0, "skipped_timeout", "")]
    assert [action for *_, action in timeline] == [None]
    requests = [
        (call["request"], call["hang"], call["finalizing"])
        for call in env.calls
        if call["call"] == "model.request"
    ]
    assert requests == [(0, True, False), (1, False, True)]  # the retry came after the snapshot
    assert any(
        call["call"] == "runner.log" and "draining 1 abandoned generate worker" in call["line"]
        for call in env.calls
    )
    assert len(sleeps.calls) == 1
    (turn,) = agent._runtime.turns
    assert turn.output == STEP.reply and turn.decision == 0
    (entry,) = agent.action_history
    assert entry["observation_tick"] == 0 and "result" not in entry
    assert agent.token_usage["responses"] == 1


def test_session_abort_during_backoff_reconnects_today(tmp_path, monkeypatch):
    """Session API (OpenAI Realtime). response 1 is dropped with close 1011 before any output,
    a TRANSIENT failure (providers/openai_realtime.py:233-250), and the harness sleeps its
    backoff. The runner aborts during that sleep. The next attempt finds _connected cleared,
    opens a second WebSocket through connect() (context.py:408-417) and runs a new response."""
    sleeps = _GatedSleep()
    monkeypatch.setattr(model_module, "time", sleeps)
    with RealtimeStub(realtime_replies([STEP]), drop_responses=(1,)) as stub:
        episode = build_episode(
            realtime_config(stub.base_url),
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=REALTIME_MODEL_KEY,
            clock_mode="token",
            tokens_per_frame=TOKENS_PER_FRAME,
            max_ticks=MAX_TICKS,
        )
        agent = episode.agent
        try:
            stub.wait_for_session_update()
            connects = _count_calls(agent.driver, "connect")
            syncs = _count_calls(agent.driver, "sync_history")
            turn_epochs = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            agent.configure_decision(episode.budget(observation.tick_number))
            box, done = _generate_in_worker(agent, observation)
            assert sleeps.entered.wait(WAIT_S), "the dropped response was not retried"
            assert len(stub.connections) == 1 and len(stub.responses) == 1
            epoch = _abort_like_the_runner(agent)
            sleeps.release.set()
            result = _finish(box, done)
        finally:
            episode.finish()
        connections = stub.finish()
        responses = list(stub.responses)
        second = [event["type"] for event in stub.connections[1].events]
    assert stub.violations == [] and stub.unscripted == 0 and stub.unused_replies == 0
    first, _ = connections
    assert first["close"] == {"received": 1011, "sent": 1011}
    # The new connection and response come after the abort.
    assert [r["connection"] for r in responses] == [1, 2]
    assert responses[1]["status"] == "completed"
    assert second == ["session.update", "conversation.item.create", "response.create"]
    assert len(connects) == 1 and len(syncs) == 0  # connect(), not the DIRTY-only recovery sync
    assert len(sleeps.calls) == 1
    _check_uncommitted_turn(agent, result, turn_epochs, epoch)
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "ok", False)]
