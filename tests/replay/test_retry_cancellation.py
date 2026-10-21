"""An aborted generation sends no further request.

The kernel gives up on a generation with AbortGeneration, which the shell runs as the agent's
abort(generation) (evaluation/shell.py KernelAgent); the legacy run_stream calls abort() when its
bounded wait expires (evaluation/runner.py _bounded_call). DefaultHarness.abort sets the cancel
event of that generation, or of every generation that has not ended, then has the runtime close
the session (harnesses/default/context.py RealtimeRuntime.abort). Each generation runs every
attempt with its own event (RealtimeRuntime.run_turn's ``cancel``) and its retries inside
retry_cancellation. Once the event is set the backoff ends, no attempt starts (entity/model.py
RequestRetry), and an attempt already under way sends neither its observation nor its request
(RealtimeRuntime._run_turn). RetryCancelled unwinds to generate, which hands back a cancelled
result without an action. The kernel discards it, or stream_generate drops it for the legacy
loop, and nothing joins the history.

Before this, the next attempt read the epoch afresh, reconnected and kept its reply: a request
after the abort, and a turn and an action that nobody committed.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

import realtime_framework.entity.model as model_module
import realtime_framework.providers.decisions.selection as selection_module
import realtime_framework.providers.registry as catalog
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

from ..fake_clock import FakeClock
from .build import build_episode
from .fake_env import FakeEnvironment, Hang
from .frames import CLOCK_ORIGIN, ReplayTask
from .loop import commit_statuses, run_loop
from .scenario import CLICK_NAME, TASK_INSTRUCTION, replies
from .scenario_decisions import ADA, LOVELACE, SPACE, decision
from .steps import Step
from .steps_decisions import run_decisions
from .stub import ChatCompletionsStub
from .stub_decisions import DecisionsStub
from .stub_decisions import Failure as QuestionFailure
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
    """Wraps the retry executor's backoff (entity/model.py _backoff). It records every delay;
    backoffs before the ``gated``-th return at once. That one sets ``entered`` and waits for
    ``release``, so the test aborts first, then waits as the real backoff, but for longer than
    any test waits: only the cancellation ends it in time."""

    def __init__(self, gated: int = 1):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self._gated = gated
        self._wait = model_module._backoff

    def __call__(self, clock, cancel, seconds):
        self.calls.append(seconds)
        if len(self.calls) < self._gated:
            return
        self.entered.set()
        if not self.release.wait(WAIT_S):
            raise AssertionError("the test never released the backoff")
        self._wait(clock, cancel, 3 * WAIT_S)


GENERATION = 1  # the kernel's number of the generation each agent-level test aborts


def _generate_in_worker(episode, observation, decision=0):
    """The shell's StartGeneration for the decision: the agent's generate on a daemon thread."""
    box, done = {}, threading.Event()
    budget = episode.budget(observation.tick_number)

    def worker():
        try:
            box["pending"] = episode.agent.generate(GENERATION, decision, observation, budget)
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    return box, done


def _abort_like_the_kernel(agent) -> None:
    """The shell runs AbortGeneration as abort(generation). Waiting for the cleanup
    RealtimeRuntime.abort schedules only makes the state below deterministic."""
    runtime = agent._runtime
    agent.abort(GENERATION)
    cleanup = runtime._cleanup_future
    assert cleanup is not None
    cleanup.result(timeout=WAIT_S)
    assert runtime._connected is False  # RealtimeRuntime._abort_active


def _discarded(agent, box, done) -> None:
    """The abandoned generation's result is cancelled; the kernel discards it."""
    assert done.wait(WAIT_S), "the abandoned worker never returned"
    assert "error" not in box, repr(box.get("error"))
    pending = box["pending"]
    assert pending.cancelled, pending
    assert pending.result["stale"] is True and pending.result["action"] is None, pending.result
    agent.discard(GENERATION)


def _count_calls(target, name: str) -> list:
    """Wrap an async method; each call records its keyword arguments, and its positional ones
    under ``args``."""
    calls = []
    original = getattr(target, name)

    async def recorded(*args, **kwargs):
        calls.append({"args": args, **kwargs})
        return await original(*args, **kwargs)

    setattr(target, name, recorded)
    return calls


def _requests(calls: list) -> list[tuple]:
    """Each _run_turn call's kernel generation (its request's), and whether it is cancelled."""
    return [(call["args"][1].generation, call["cancel"].is_set()) for call in calls]


def _trace(run_dir, event: str) -> list[dict]:
    lines = (run_dir / "episode_trace.jsonl").read_text().splitlines()
    return [e["payload"] for e in map(json.loads, lines) if e["event"] == event]


def _attempts(run_dir) -> list[tuple]:
    return [
        (a["scope"]["attempt"], a["outcome"], a.get("cancelled"))
        for a in _trace(run_dir, "ModelAttemptFinished")
        if a["kind"] == "generation"
    ]


def _connects(run_dir) -> list[tuple]:
    return [
        (a["scope"]["phase"], a["scope"]["connect_attempt"], a["outcome"])
        for a in _trace(run_dir, "ModelAttemptFinished")
        if a["kind"] == "session_connect"
    ]


def _questions(run_dir) -> list[tuple]:
    return [
        (
            a["scope"]["decision"],
            a["scope"]["choice"]["question"],
            a["scope"]["choice"]["transport_attempt"],
            a["outcome"],
        )
        for a in _trace(run_dir, "ModelAttemptFinished")
        if a["kind"] == "choice_question"
    ]


def _inspection_outcomes(run_dir) -> list:
    """How each observation's inspection ended (sandbox/observation/som.py ToolRuntime.finish);
    one never finished is written at close without an outcome."""
    lines = (run_dir / "tool_calls.jsonl").read_text().splitlines()
    return [json.loads(line).get("outcome") for line in lines]


def _check_nothing_kept(agent, turns: list) -> None:
    """Only attempt 1 ran, for the generation the abort cancelled; no turn or action joins the
    history."""
    assert _requests(turns) == [(GENERATION, True)]
    assert agent._runtime.turns == ()
    assert agent.action_history == []


@pytest.mark.parametrize("asynchronous", [False, True], ids=["run", "run_async"])
def test_cancel_ends_the_backoff(asynchronous):
    """A 25 s Retry-After backoff (entity/model.py TransportRetryPolicy) ends as soon as the
    cancel event is set, and the unit is not sent again; it records attempt 2 as cancelled."""
    cancel, recorded = threading.Event(), threading.Event()
    sent, delays, stopped = [], [], []

    def send(attempt):
        sent.append(attempt)
        raise ConnectionError("dropped")

    async def send_async(attempt):
        send(attempt)

    def record(attempt, exc, failure, delay):
        delays.append(delay)
        recorded.set()

    retry = RequestRetry(
        "unit",
        lambda exc: TransportFailure(RATE_LIMITED, 25.0),
        record,
        cancelled=lambda attempt, exc: stopped.append(attempt),
    )
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
    assert sent == [1] and delays == [25.0] and stopped == [2]


def test_no_attempt_once_cancelled():
    cancel = threading.Event()
    cancel.set()
    sent, stopped = [], []
    retry = RequestRetry(
        "unit",
        lambda exc: TransportFailure(TRANSIENT),
        lambda *args: None,
        cancelled=lambda attempt, exc: stopped.append(attempt),
    )
    with retry_cancellation(cancel), pytest.raises(RetryCancelled):
        retry.run(sent.append)
    assert sent == [] and stopped == [1]


def test_a_nested_cancellation_stops_the_enclosing_attempt():
    """A unit sent inside another's attempt, as a session connect inside a turn: the abort comes
    while the inner unit's attempt fails. The inner unit records its retry as cancelled, and the
    outer one the attempt under way, which it neither records as failed nor retries."""
    cancel = threading.Event()
    stopped, failed = [], []

    def inner_send(attempt):
        cancel.set()
        raise ConnectionError("dropped")

    def unit(name):
        return RequestRetry(
            name,
            lambda exc: TransportFailure(TRANSIENT),
            lambda attempt, exc, failure, delay: failed.append((name, attempt)),
            cancelled=lambda attempt, exc: stopped.append((name, attempt)),
        )

    with retry_cancellation(cancel), pytest.raises(RetryCancelled):
        unit("outer").run(lambda attempt: unit("inner").run(inner_send))
    assert failed == [("inner", 1)]
    assert stopped == [("inner", 2), ("outer", 1)]


def test_abort_during_backoff_sends_no_retry(openrouter_env, tmp_path):  # noqa: F811
    """Stateless API. Attempt 1 gets HTTP 503 (TRANSIENT, entity/model.py classify_http_failure)
    and the harness backs off. The kernel aborts the generation during the backoff: the wait
    ends, and the same request is not sent again. The request was recorded under the kernel's
    generation number. Discarding the result ends the decision's inspection as
    superseded_by_timeout (harness.py DefaultHarness._drop)."""
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
            turns = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            box, done = _generate_in_worker(episode, observation)
            assert backoff.entered.wait(WAIT_S), "attempt 1 was not retried"
            assert len(stub.requests) == 1
            _abort_like_the_kernel(agent)
            backoff.release.set()
            _discarded(agent, box, done)
        finally:
            episode.finish()
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert len(stub.requests) == 1
    assert len(backoff.calls) == 1
    assert connects == []  # no reconnect for the abandoned decision
    _check_nothing_kept(agent, turns)
    sent = [
        (r["scope"]["decision"], r["scope"]["generation"], r["scope"]["attempt"])
        for r in _trace(tmp_path, "ModelRequestSent")
    ]
    assert sent == [(0, GENERATION, 1)]
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "cancelled", None)]
    assert _inspection_outcomes(tmp_path) == ["superseded_by_timeout"]


def _stateless_episode(tmp_path):
    return build_episode(
        openrouter_config(),
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        model_key=OPENROUTER_MODEL_KEY,
        **TOKEN_CLOCK,
    )


def test_next_decision_runs_after_an_abort(openrouter_env, tmp_path):  # noqa: F811
    """Stateless API. Generation 1 is abandoned in its backoff, as above. The abort cancels that
    generation's own event (DefaultHarness.abort); the next generation, for decision 1 on the
    same agent, starts with a new one (DefaultHarness._open), so it is not cancelled: it sends its
    request and returns its action."""
    backoff = _GatedBackoff()
    openrouter_env.setattr(model_module, "_backoff", backoff)
    with OpenRouterStub([UNAVAILABLE, *replies([STEP])], route=ROUTE) as stub:
        openrouter_env.setenv("OPENROUTER_BASE_URL", stub.base_url)
        episode = _stateless_episode(tmp_path)
        agent = episode.agent
        try:
            observation = episode.frames.capture(0)
            box, done = _generate_in_worker(episode, observation)
            assert backoff.entered.wait(WAIT_S), "attempt 1 was not retried"
            _abort_like_the_kernel(agent)
            backoff.release.set()
            _discarded(agent, box, done)
            observation = episode.frames.capture(1)
            budget = episode.budget(observation.tick_number)
            pending = agent.generate(GENERATION + 1, 1, observation, budget)
            agent.accept(GENERATION + 1)
        finally:
            episode.finish()
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert len(stub.requests) == 2
    assert not pending.cancelled and pending.result.get("invalid_reason") is None
    assert pending.result["action"] is not None
    assert len(agent.action_history) == 1
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "cancelled", None), (1, "ok", False)]


def test_close_during_backoff_sends_no_retry(openrouter_env, tmp_path):  # noqa: F811
    """Stateless API. Attempt 1 gets HTTP 503 and the harness backs off; the agent is closed, not
    aborted, during the backoff. close() cancels every generation that has not ended
    (DefaultHarness.close), so the wait ends and the request is not sent again."""
    backoff = _GatedBackoff()
    openrouter_env.setattr(model_module, "_backoff", backoff)
    with OpenRouterStub([UNAVAILABLE], route=ROUTE) as stub:
        openrouter_env.setenv("OPENROUTER_BASE_URL", stub.base_url)
        episode = _stateless_episode(tmp_path)
        agent = episode.agent
        try:
            observation = episode.frames.capture(0)
            box, done = _generate_in_worker(episode, observation)
            assert backoff.entered.wait(WAIT_S), "attempt 1 was not retried"
            agent.close()
            backoff.release.set()
            _discarded(agent, box, done)
        finally:
            episode.finish()
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert len(stub.requests) == 1 and len(backoff.calls) == 1
    assert agent._runtime.turns == () and agent.action_history == []
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "cancelled", None)]


def test_abort_mid_request_is_not_retried(tmp_path, monkeypatch):
    """Stateless API, abort while the request is in flight. The stub holds request 0 until the
    client closes it; the abort's driver.close() does (providers/litellm.py close). The
    failure arrives before any output and is classified TRANSIENT, but the generation is
    cancelled: the backoff returns at once and nothing is sent again. The reply scripted for a
    retry stays unused."""
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
            turns = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            box, done = _generate_in_worker(episode, observation)
            assert arrived.wait(WAIT_S)
            _abort_like_the_kernel(agent)
            _discarded(agent, box, done)
        finally:
            episode.finish()
    assert stub.held == [True]  # the abort closed the held request
    assert stub.unscripted == 0 and stub.unused_replies == 1
    assert len(stub.requests) == 1
    ((failure, error),) = classified
    assert failure == TransportFailure(TRANSIENT), error
    assert len(backoff.calls) == 1
    _check_nothing_kept(agent, turns)
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "cancelled", None)]


def test_runner_timeout_leaves_no_request_after_the_episode(tmp_path, monkeypatch):
    """The real run_stream and _bounded_call on a 0.5 s wall-clock episode. Decision 0's request
    never comes back; the bounded wait (0.5 s, the grace set to 0 to keep it short) expires, the
    decision is committed as skipped_timeout and the episode ends. The abort closes the request
    and cancels the decision's generation, so the abandoned worker does not send it again: the
    stub sees one request, and neither a turn nor an action is left behind."""
    monkeypatch.setattr(runner, "DEADLINE_GRACE_S", 0.0)
    clock = FakeClock(wall_origin=CLOCK_ORIGIN)
    env_box = []
    with (
        ChatCompletionsStub(
            replies([STEP]), on_request=lambda index: env_box[0].model_request(index), clock=clock
        ) as stub,
        clock.participant(),
    ):
        config = chat_completions_config(stub.base_url, stream=True)
        episode = build_episode(
            config,
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=CHAT_MODEL_KEY,
            clock_mode="wall",
            clock=clock,
        )
        env = FakeEnvironment(
            episode.frames,
            clock,
            fps=config.fps,
            max_minutes=0.5 / 60,
            reply_latencies=[Hang(0.5)],
        )
        env_box.append(env)
        turns = _count_calls(episode.agent._runtime, "_run_turn")
        timeline = run_loop(episode, env)
    agent = episode.agent
    assert stub.unscripted == 0 and stub.unused_replies == 1
    assert stub.held == [True]
    assert len(stub.requests) == 1
    assert _requests(turns) == [(None, True)]  # the legacy loop numbers no generation
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
    TRANSIENT failure (providers/openai_realtime.py classify_transport_error), and the harness
    backs off. The kernel aborts the generation during the backoff: no second WebSocket opens
    and no second response runs."""
    backoff = _GatedBackoff()
    monkeypatch.setattr(model_module, "_backoff", backoff)
    with RealtimeStub(realtime_replies([STEP]), drop_responses=(1,)) as stub:
        episode = _session_episode(stub, tmp_path)
        agent = episode.agent
        try:
            stub.wait_for_session_update()
            connects = _count_calls(agent.driver, "connect")
            turns = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            box, done = _generate_in_worker(episode, observation)
            assert backoff.entered.wait(WAIT_S), "the dropped response was not retried"
            _abort_like_the_kernel(agent)
            backoff.release.set()
            _discarded(agent, box, done)
        finally:
            episode.finish()
        connections = stub.finish()
        responses = list(stub.responses)
    assert stub.violations == [] and stub.unscripted == 0
    (first,) = connections
    assert first["close"] == {"received": 1011, "sent": 1011}
    assert [r["connection"] for r in responses] == [1]
    assert connects == [] and len(backoff.calls) == 1
    _check_nothing_kept(agent, turns)
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "cancelled", None)]


def test_abort_while_recovering_sends_no_request(openrouter_env, tmp_path):  # noqa: F811
    """Stateless API. Attempt 1 gets HTTP 503, which leaves the runtime DIRTY
    (RealtimeRuntime.run_turn), so attempt 2 first recovers it (RealtimeRuntime._ensure_ready).
    The abort lands during that recovery; the attempt then stops before it sends its observation
    or its request (RealtimeRuntime._run_turn)."""
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
            turns = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            box, done = _generate_in_worker(episode, observation)
            assert syncing.wait(WAIT_S), "attempt 2 did not recover the runtime"
            _abort_like_the_kernel(agent)
            release.set()
            _discarded(agent, box, done)
        finally:
            episode.finish()
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert len(stub.requests) == 1
    assert _requests(turns) == [(GENERATION, True)] * 2
    assert agent._runtime.turns == () and agent.action_history == []
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "cancelled", None)]


def test_session_abort_before_reconnecting_opens_no_session(tmp_path, monkeypatch):
    """Session API. Response 1 is dropped, and attempt 2 rebuilds the session through connect()
    (RealtimeRuntime._ensure_ready). The abort lands before connect() opens it. connect()'s own
    attempts run under the generation's retry_cancellation (providers/openai_realtime.py
    _connect), so no WebSocket opens and no response runs; the connect records its first attempt
    as cancelled."""
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
            turns = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            box, done = _generate_in_worker(episode, observation)
            assert connecting.wait(WAIT_S), "attempt 2 did not reconnect"
            _abort_like_the_kernel(agent)
            release.set()
            _discarded(agent, box, done)
            # abort()'s state: the abandoned rebuild records no failure and no session state.
            assert agent._runtime.state == RuntimeState.DIRTY
        finally:
            episode.finish()
        connections = stub.finish()
        responses = list(stub.responses)
    assert stub.violations == [] and stub.unscripted == 0
    assert len(connections) == 1
    assert [r["connection"] for r in responses] == [1]
    assert _requests(turns) == [(GENERATION, True)] * 2
    assert agent._runtime.turns == () and agent.action_history == []
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "cancelled", None)]
    assert _connects(tmp_path) == [("initialize", 1, "ok"), ("decision", 1, "cancelled")]


def test_session_abort_during_connect_backoff(tmp_path, monkeypatch):
    """Session API. Response 1 is dropped; attempt 2 rebuilds the session through connect(),
    whose first upgrade is refused (HTTP 503), and connect() backs off
    (providers/openai_realtime.py _connect). The abort lands during that backoff: no further
    upgrade is tried, and after the attempt it meant to retry the connect records the next one as
    cancelled."""
    backoff = _GatedBackoff(gated=2)  # the turn's backoff passes, the connect's is held
    monkeypatch.setattr(model_module, "_backoff", backoff)
    with RealtimeStub(
        realtime_replies([STEP]), drop_responses=(1,), reject_upgrades=(2,)
    ) as stub:
        episode = _session_episode(stub, tmp_path)
        agent = episode.agent
        try:
            stub.wait_for_session_update()
            observation = episode.frames.capture(0)
            box, done = _generate_in_worker(episode, observation)
            assert backoff.entered.wait(WAIT_S), "the refused upgrade was not retried"
            _abort_like_the_kernel(agent)
            backoff.release.set()
            _discarded(agent, box, done)
        finally:
            episode.finish()
        connections = stub.finish()
    assert stub.violations == [] and stub.unscripted == 0
    assert [c["rejected"] for c in connections] == [None, 503]
    assert len(backoff.calls) == 2
    assert agent._runtime.turns == () and agent.action_history == []
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "cancelled", None)]
    assert _connects(tmp_path) == [
        ("initialize", 1, "ok"),
        ("decision", 1, "retry"),
        ("decision", 2, "cancelled"),
    ]


def test_reconnect_that_spends_its_retries_is_not_retried_again(tmp_path, monkeypatch):
    """Session API. An abort closes the session, and the next decision reconnects
    (RealtimeRuntime._ensure_ready); every upgrade is refused (HTTP 503). connect() makes the
    shared policy's four attempts and raises TransportFailureError with its retries spent, which
    classify_transport_error passes on (providers/openai_realtime.py), so the turn is not retried
    on top of it: four connects in all, not four per turn attempt."""
    monkeypatch.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    refused = tuple(range(2, 20))  # more than four turn attempts of four connects would take
    with RealtimeStub(realtime_replies([STEP]), reject_upgrades=refused) as stub:
        episode = _session_episode(stub, tmp_path)
        agent = episode.agent
        try:
            stub.wait_for_session_update()
            agent.abort()  # no generation is under way: it only closes the session
            agent._runtime._cleanup_future.result(timeout=WAIT_S)
            observation = episode.frames.capture(0)
            budget = episode.budget(observation.tick_number)
            pending = agent.generate(GENERATION, 0, observation, budget)
            agent.discard(GENERATION)
        finally:
            episode.finish()
        connections = stub.finish()
    assert stub.violations == [] and stub.unscripted == 0
    assert [c["rejected"] for c in connections] == [None] + [503] * 4
    assert not pending.cancelled
    assert pending.result["invalid_reason"] == "provider_unavailable"
    assert _attempts(tmp_path) == [(1, "failed", None)]
    assert _connects(tmp_path) == [
        ("initialize", 1, "ok"),
        ("decision", 1, "retry"),
        ("decision", 2, "retry"),
        ("decision", 3, "retry"),
        ("decision", 4, "failed"),
    ]


LUNA = "gpt-6-luna-decisions"  # configs/models/gpt-6-luna-decisions.yaml


def _decision_episode(stub, tmp_path, monkeypatch):
    """The Luna catalog row at the stub, as tests/replay/test_decisions.py runs it."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "replay-dummy-key")
    config = catalog.to_model_config(LUNA, fps=1.0, reasoning_effort="fixed")
    return build_episode(
        config.model_copy(update={"api_base_url": stub.base_url}),
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        model_key=LUNA,
    )


def test_decision_abort_during_question_backoff(tmp_path, monkeypatch):
    """Decision API. The first question gets HTTP 503 (TRANSIENT) and backs off in its own retry
    (providers/decisions/selection.py _decide_with_retries), which runs under the generation's
    retry_cancellation. The abort lands during the backoff: the question is not asked again, and
    after the attempt it meant to retry the question records the next one as cancelled."""
    backoff = _GatedBackoff()
    monkeypatch.setattr(model_module, "_backoff", backoff)
    with DecisionsStub(upstream="openai") as stub:
        episode = _decision_episode(stub, tmp_path, monkeypatch)
        agent = episode.agent
        try:
            stub.load(decision(QuestionFailure(503, "Provider returned error"), "click_P1").script)
            observation = episode.frames.capture(0)
            box, done = _generate_in_worker(episode, observation)
            assert backoff.entered.wait(WAIT_S), "the question was not retried"
            _abort_like_the_kernel(agent)
            backoff.release.set()
            _discarded(agent, box, done)
        finally:
            episode.finish()
    assert len(stub.requests) == 1 and stub.unscripted == 0 and stub.unload() == 1
    assert len(backoff.calls) == 1
    assert agent._runtime.turns == () and agent.action_history == []
    assert _questions(tmp_path) == [(0, 0, 1, "retry"), (0, 0, 2, "cancelled")]
    assert _attempts(tmp_path) == [(1, "cancelled", None)]  # the turn the question was in


def test_decision_abort_between_questions(tmp_path, monkeypatch):
    """Decision API. Decision 0 clicks the name field. Decision 1's first question picks
    compose_text, and the abort lands before its first text-edit question is asked
    (selection.py _ask). That question sends nothing, and records its first attempt as
    cancelled."""
    ask = selection_module.ChoiceDecision._ask
    asked, entered, release = [], threading.Event(), threading.Event()

    async def gated_ask(self, question, **kwargs):
        asked.append(kwargs["phase"])
        if len(asked) == 3:
            entered.set()
            await asyncio.to_thread(release.wait, WAIT_S)
        return await ask(self, question, **kwargs)

    monkeypatch.setattr(selection_module.ChoiceDecision, "_ask", gated_ask)
    with DecisionsStub(upstream="openai") as stub:
        episode = _decision_episode(stub, tmp_path, monkeypatch)
        agent = episode.agent
        try:
            run_decisions(episode, [decision("click_P1")], stub)
            stub.load(decision("compose_text", ADA, SPACE, LOVELACE, "finish").script)
            observation = episode.frames.capture(1)
            box, done = _generate_in_worker(episode, observation, decision=1)
            assert entered.wait(WAIT_S), "the text-edit question was not reached"
            _abort_like_the_kernel(agent)
            release.set()
            _discarded(agent, box, done)
        finally:
            release.set()
            episode.finish()
    assert asked == ["gui_action", "gui_action", "text_edit"]
    assert len(stub.requests) == 2 and stub.unscripted == 0 and stub.unload() == 4
    assert len(agent._runtime.turns) == len(agent.action_history) == 1
    assert _questions(tmp_path) == [(0, 0, 1, "ok"), (1, 0, 1, "ok"), (1, 1, 1, "cancelled")]
    assert _attempts(tmp_path) == [(1, "ok", False), (1, "cancelled", None)]
