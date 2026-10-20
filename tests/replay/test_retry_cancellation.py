"""An aborted decision sends no further request.

When run_stream's bounded wait expires it keeps the generate worker in ``abandoned`` and calls the
agent's abort (evaluation/runner.py:337-345). DefaultHarness.abort reaches the runtime
(harnesses/default/harness.py:1700-1703), which ends the generation epoch and sets the epoch's
event (harnesses/default/context.py:687-691). stream_generate takes the epoch and its event when
the decision starts (harness.py:1159), runs every attempt under that epoch (context.py:640-651)
and its retries inside retry_cancellation (harness.py:1350). Once the event is set the backoff
ends, no attempt starts (entity/model.py:479-505), and an attempt already under way sends neither
its observation nor its request (context.py:482-484). RetryCancelled unwinds to stream_generate,
which returns a stale result without an action; nothing reads it, run_stream has moved on.

Before this, the next attempt read the epoch afresh, reconnected and kept its reply: a request
after the abort, and a turn and an action that nobody committed.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

import realtime_framework.entity.model as model_module
from realtime_framework.entity.model import (
    RATE_LIMITED,
    TRANSIENT,
    RequestRetry,
    RetryCancelled,
    RuntimeState,
    TransportFailure,
    retry_cancellation,
)
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

WAIT_S = 10.0  # bound on every cross-thread wait; none is reached when the code behaves
STEP = Step(CLICK_NAME)
UNAVAILABLE = Failure(503, {"error": {"message": "upstream down", "code": 503}})


class _GatedBackoff:
    """Wraps the retry executor's backoff (entity/model.py:457-458). It records the delay, sets
    ``entered`` and waits for ``release`` before the real wait, so the test aborts first."""

    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self._wait = model_module._backoff

    def __call__(self, clock, cancel, seconds):
        self.calls.append(seconds)
        self.entered.set()
        if not self.release.wait(WAIT_S):
            raise AssertionError("the test never released the backoff")
        self._wait(clock, cancel, seconds)


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
    scheduled cleanup (context.py:696-700) only makes the state below deterministic. Returns the
    epoch the abort ended."""
    runtime = agent._runtime
    before = runtime.generation_epoch()
    agent.abort()
    cleanup = runtime._cleanup_future
    assert cleanup is not None
    cleanup.result(timeout=WAIT_S)
    assert runtime.generation_epoch() == before + 1
    assert runtime._connected is False  # context.py:683
    return before


def _cancelled(box, done) -> dict:
    assert done.wait(WAIT_S), "the abandoned worker never returned"
    assert "error" not in box, repr(box.get("error"))
    result = box["result"]
    assert result["stale"] is True and result["action"] is None, result
    return result


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
        if a["kind"] == "generation"
    ]


def _check_nothing_kept(agent, turn_epochs: list, epoch: int) -> None:
    """Only attempt 1 ran, under the epoch the abort ended; no turn or action joins the history."""
    assert [call["epoch"] for call in turn_epochs] == [epoch]
    assert agent._runtime.turns == ()
    assert agent.action_history == []


@pytest.mark.parametrize("asynchronous", [False, True], ids=["run", "run_async"])
def test_cancel_ends_the_backoff(asynchronous):
    """A 25 s Retry-After backoff (entity/model.py:378-385) ends as soon as the epoch's event is
    set, and the unit is not sent again."""
    cancel, recorded = threading.Event(), threading.Event()
    sent, delays = [], []

    def send(attempt):
        sent.append(attempt)
        raise ConnectionError("dropped")

    async def send_async(attempt):
        send(attempt)

    def record(attempt, exc, failure, delay):
        delays.append(delay)
        recorded.set()

    retry = RequestRetry("unit", lambda exc: TransportFailure(RATE_LIMITED, 25.0), record)
    box, done = {}, threading.Event()

    def worker():
        try:
            with retry_cancellation(cancel):
                if asynchronous:
                    asyncio.run(retry.run_async(send_async))
                else:
                    retry.run(send)
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    assert recorded.wait(WAIT_S)
    cancel.set()
    assert done.wait(WAIT_S), "the backoff was not cancelled"
    assert isinstance(box.get("error"), RetryCancelled)
    assert sent == [1] and delays == [25.0]


def test_no_attempt_once_cancelled():
    cancel = threading.Event()
    cancel.set()
    sent = []
    retry = RequestRetry("unit", lambda exc: TransportFailure(TRANSIENT), lambda *args: None)
    with retry_cancellation(cancel), pytest.raises(RetryCancelled):
        retry.run(sent.append)
    assert sent == []


def test_abort_during_backoff_sends_no_retry(openrouter_env, tmp_path):  # noqa: F811
    """Stateless API. Attempt 1 gets HTTP 503 (TRANSIENT, entity/model.py:424-425) and the
    harness backs off. The runner aborts the agent during the backoff: the wait ends, and the
    same request is not sent again."""
    backoff = _GatedBackoff()
    openrouter_env.setattr(model_module, "_backoff", backoff)
    with OpenRouterStub([UNAVAILABLE], route=ROUTE) as stub:
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
            assert backoff.entered.wait(WAIT_S), "attempt 1 was not retried"
            assert len(stub.requests) == 1
            epoch = _abort_like_the_runner(agent)
            backoff.release.set()
            _cancelled(box, done)
        finally:
            episode.finish()
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert len(stub.requests) == 1
    assert len(backoff.calls) == 1
    assert connects == []  # no reconnect for the abandoned decision
    _check_nothing_kept(agent, turn_epochs, epoch)
    sent = [
        (r["scope"]["decision"], r["scope"]["attempt"])
        for r in _trace(tmp_path, "ModelRequestSent")
    ]
    assert sent == [(0, 1)]
    assert _attempts(tmp_path) == [(1, "retry", None)]


def test_abort_mid_request_is_not_retried(tmp_path, monkeypatch):
    """Stateless API, abort while the request is in flight. The stub holds request 0 until the
    client closes it; the abort's driver.close() does (providers/litellm.py:1033-1044). The
    failure arrives before any output and is classified TRANSIENT, but the epoch has ended: the
    backoff returns at once and nothing is sent again. The reply scripted for a retry stays
    unused."""
    backoff = _GatedBackoff()
    backoff.release.set()
    monkeypatch.setattr(model_module, "_backoff", backoff)
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
            _cancelled(box, done)
        finally:
            episode.finish()
    assert stub.held == [True]  # the abort closed the held request
    assert stub.unscripted == 0 and stub.unused_replies == 1
    assert len(stub.requests) == 1
    ((failure, error),) = classified
    assert failure == TransportFailure(TRANSIENT), error
    assert len(backoff.calls) == 1
    _check_nothing_kept(agent, turn_epochs, epoch)
    assert _attempts(tmp_path) == [(1, "retry", None)]


def test_runner_timeout_leaves_no_request_after_the_episode(tmp_path, monkeypatch):
    """The real run_stream and _bounded_call on a 0.5 s wall-clock episode. Decision 0's request
    never comes back; the bounded wait (0.5 s, the grace set to 0 to keep it short) expires, the
    decision is committed as skipped_timeout and the episode ends. The abort closes the request
    and ends the epoch, so the abandoned worker does not send it again: the stub sees one
    request, and neither a turn nor an action is left behind."""
    monkeypatch.setattr(runner, "DEADLINE_GRACE_S", 0.0)
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
        env = FakeEnvironment(
            episode.frames,
            FakeClock(),
            fps=config.fps,
            max_minutes=0.5 / 60,
            reply_latencies=[Hang(0.5)],
        )
        env_box.append(env)
        turn_epochs = _count_calls(episode.agent._runtime, "_run_turn")
        timeline = run_loop(episode, env)
    agent = episode.agent
    assert stub.unscripted == 0 and stub.unused_replies == 1
    assert stub.held == [True]
    assert len(stub.requests) == 1
    assert [call["epoch"] for call in turn_epochs] == [0]
    assert commit_statuses(env) == [(0, "skipped_timeout", "")]
    assert [action for *_, action in timeline] == [None]
    requests = [
        (call["request"], call["hang"], call["finalizing"])
        for call in env.calls
        if call["call"] == "model.request"
    ]
    assert requests == [(0, True, False)]
    assert agent._runtime.turns == ()
    assert agent.action_history == []
    assert agent.token_usage["responses"] == 0


def _session_episode(stub, tmp_path):
    return build_episode(
        realtime_config(stub.base_url),
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        model_key=REALTIME_MODEL_KEY,
        clock_mode="token",
        tokens_per_frame=TOKENS_PER_FRAME,
        max_ticks=MAX_TICKS,
    )


def test_session_abort_during_backoff_does_not_reconnect(tmp_path, monkeypatch):
    """Session API (OpenAI Realtime). Response 1 is dropped with close 1011 before any output, a
    TRANSIENT failure (providers/openai_realtime.py:233-250), and the harness backs off. The
    runner aborts during the backoff: no second WebSocket opens and no second response runs."""
    backoff = _GatedBackoff()
    monkeypatch.setattr(model_module, "_backoff", backoff)
    with RealtimeStub(realtime_replies([STEP]), drop_responses=(1,)) as stub:
        episode = _session_episode(stub, tmp_path)
        agent = episode.agent
        try:
            stub.wait_for_session_update()
            connects = _count_calls(agent.driver, "connect")
            turn_epochs = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            agent.configure_decision(episode.budget(observation.tick_number))
            box, done = _generate_in_worker(agent, observation)
            assert backoff.entered.wait(WAIT_S), "the dropped response was not retried"
            epoch = _abort_like_the_runner(agent)
            backoff.release.set()
            _cancelled(box, done)
        finally:
            episode.finish()
        connections = stub.finish()
        responses = list(stub.responses)
    assert stub.violations == [] and stub.unscripted == 0
    (first,) = connections
    assert first["close"] == {"received": 1011, "sent": 1011}
    assert [r["connection"] for r in responses] == [1]
    assert connects == [] and len(backoff.calls) == 1
    _check_nothing_kept(agent, turn_epochs, epoch)
    assert _attempts(tmp_path) == [(1, "retry", None)]


def test_abort_while_recovering_sends_no_request(openrouter_env, tmp_path):  # noqa: F811
    """Stateless API. Attempt 1 gets HTTP 503, which leaves the runtime DIRTY
    (context.py:665-671), so attempt 2 first recovers it (context.py:402-412). The abort lands
    during that recovery; the attempt then stops before it sends its observation or its request
    (context.py:482-484)."""
    openrouter_env.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    with OpenRouterStub([UNAVAILABLE], route=ROUTE) as stub:
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
            syncing, release = threading.Event(), threading.Event()
            sync_history = agent.driver.sync_history

            async def gated_sync(request):
                syncing.set()
                await asyncio.to_thread(release.wait, WAIT_S)
                return await sync_history(request)

            agent.driver.sync_history = gated_sync
            turn_epochs = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            agent.configure_decision(episode.budget(observation.tick_number))
            box, done = _generate_in_worker(agent, observation)
            assert syncing.wait(WAIT_S), "attempt 2 did not recover the runtime"
            epoch = _abort_like_the_runner(agent)
            release.set()
            _cancelled(box, done)
        finally:
            episode.finish()
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert len(stub.requests) == 1
    assert [call["epoch"] for call in turn_epochs] == [epoch, epoch]
    assert agent._runtime.turns == () and agent.action_history == []
    assert _attempts(tmp_path) == [(1, "retry", None)]


def test_session_abort_before_reconnecting_opens_no_session(tmp_path, monkeypatch):
    """Session API. Response 1 is dropped, and attempt 2 rebuilds the session through connect()
    (context.py:402-412; providers/openai_realtime.py:661-671). The abort lands before connect()
    opens it. connect()'s own attempts run under the decision's retry_cancellation
    (providers/openai_realtime.py:435-454), so no WebSocket opens and no response runs."""
    monkeypatch.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    with RealtimeStub(realtime_replies([STEP]), drop_responses=(1,)) as stub:
        episode = _session_episode(stub, tmp_path)
        agent = episode.agent
        try:
            stub.wait_for_session_update()
            connecting, release = threading.Event(), threading.Event()
            connect = agent.driver.connect

            async def gated_connect(request):
                connecting.set()
                await asyncio.to_thread(release.wait, WAIT_S)
                return await connect(request)

            agent.driver.connect = gated_connect
            turn_epochs = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            agent.configure_decision(episode.budget(observation.tick_number))
            box, done = _generate_in_worker(agent, observation)
            assert connecting.wait(WAIT_S), "attempt 2 did not reconnect"
            epoch = _abort_like_the_runner(agent)
            release.set()
            _cancelled(box, done)
            # abort()'s state: the abandoned rebuild records no failure and no session state.
            assert agent._runtime.state == RuntimeState.DIRTY
        finally:
            episode.finish()
        connections = stub.finish()
        responses = list(stub.responses)
    assert stub.violations == [] and stub.unscripted == 0
    assert len(connections) == 1
    assert [r["connection"] for r in responses] == [1]
    assert [call["epoch"] for call in turn_epochs] == [epoch, epoch]
    assert agent._runtime.turns == () and agent.action_history == []
    assert _attempts(tmp_path) == [(1, "retry", None)]
    connects = [a for a in _trace(tmp_path, "ModelAttemptFinished") if a["kind"] != "generation"]
    assert [a["scope"]["phase"] for a in connects] == ["initialize"]  # none for the decision
