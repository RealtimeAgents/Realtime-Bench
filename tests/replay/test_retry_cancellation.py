"""An aborted generation sends no further request.

The kernel gives up on a generation with AbortGeneration, which the shell runs as the agent's
abort(generation) (evaluation/shell.py KernelAgent). DefaultHarness.abort sets the cancel event of
that generation, then has the runtime close the session (harnesses/default/context.py
RealtimeRuntime.abort). Each generation runs every attempt with its own event
(RealtimeRuntime.run_turn's ``cancel``) and its retries inside retry_cancellation. Once the event is
set the backoff ends, no attempt starts (entity/model.py RequestRetry), and an attempt already under
way sends neither its observation nor its request (RealtimeRuntime._run_turn). RetryCancelled
unwinds to generate, which hands back a cancelled result without an action. The kernel discards
it, and nothing joins the history.

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
import realtime_framework.providers.litellm as litellm_driver
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
from realtime_framework.evaluation.clock import token_record

from ..fake_clock import FakeClock
from .build import build_episode, wait_for_abort
from .fake_env import FakeEnvironment, Hang
from .frames import CLOCK_ORIGIN, ReplayTask
from .loop import commit_statuses, run_loop
from .scenario import CLICK_NAME, TASK_INSTRUCTION, replies
from .scenario_decisions import ADA, LOVELACE, SPACE, decision
from .steps import Step, run_steps
from .steps_decisions import run_decisions
from .stub import ChatCompletionsStub
from .stub import Failure as ChatFailure
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
    """The shell runs AbortGeneration as abort(generation). Waiting for the cleanup the abort
    runs (providers/session_lifecycle.py AbortCleanup) only makes the state below
    deterministic."""
    agent.abort(GENERATION)
    wait_for_abort(agent, WAIT_S)
    assert agent._runtime.state is RuntimeState.DIRTY  # closed, or a stateless driver's


def _discarded(agent, box, done) -> None:
    """The abandoned generation's result is cancelled; the kernel discards it."""
    assert done.wait(WAIT_S), "the abandoned worker never returned"
    assert "error" not in box, repr(box.get("error"))
    pending = box["pending"]
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
    """Each _run_turn call's kernel generation (its request's), and whether its cancel event is
    set by now. That a later generation has its own event is tested on the agent
    (tests/harness/test_generations.py)."""
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


def _cancelled(run_dir, kind: str) -> list[tuple]:
    """The error and the latency recorded for each attempt of ``kind`` a cancellation stopped."""
    return [
        (a.get("error"), a.get("latency_s"))
        for a in _trace(run_dir, "ModelAttemptFinished")
        if a["kind"] == kind and a["outcome"] == "cancelled"
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
    """Only attempt 1 ran; its request names the aborted generation, whose cancel event is set.
    No turn or action joins the history."""
    assert _requests(turns) == [(GENERATION, True)]
    assert agent.history.turns == ()
    assert agent.history.actions == ()


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


@pytest.mark.parametrize("asynchronous", [False, True], ids=["run", "run_async"])
def test_an_attempt_that_fails_once_cancelled_is_cancelled(asynchronous):
    """The abort sets the cancel event, then closes the connection under the attempt under way,
    which fails with a retryable error: the attempt is neither recorded as a failure nor sent
    again, and ``cancelled`` sees it, with that error as its RetryCancelled's cause."""
    cancel = threading.Event()
    failed, stopped = [], []
    dropped = ConnectionError("dropped")

    def send(attempt):
        cancel.set()
        raise dropped

    async def send_async(attempt):
        send(attempt)

    retry = RequestRetry(
        "unit",
        lambda exc: TransportFailure(TRANSIENT),
        lambda attempt, exc, failure, delay: failed.append(attempt),
        cancelled=lambda attempt, exc: stopped.append((attempt, exc)),
    )
    with retry_cancellation(cancel), pytest.raises(RetryCancelled) as raised:
        if asynchronous:
            asyncio.run(retry.run_async(send_async))
        else:
            retry.run(send)
    assert failed == [] and stopped == [(1, raised.value)]
    assert str(raised.value) == "generation abandoned during attempt 1"
    assert raised.value.__cause__ is dropped


def test_an_error_no_transport_failure_propagates_once_cancelled_too():
    """``classify`` returns None for it, as the selection flow does for a context limit, whose
    question records it itself (providers/decisions/selection.py): it propagates unrecorded,
    and is not taken for a cancellation either."""
    cancel = threading.Event()
    recorded = []

    def send(attempt):
        cancel.set()
        raise ValueError("beyond the context")

    retry = RequestRetry(
        "unit",
        lambda exc: None,
        lambda *args: recorded.append(args),
        cancelled=lambda *args: recorded.append(args),
    )
    with retry_cancellation(cancel), pytest.raises(ValueError):
        retry.run(send)
    assert recorded == []


def test_a_nested_cancellation_stops_the_enclosing_attempt():
    """A unit sent inside another's attempt, as a session connect inside a turn: the abort comes
    while the inner unit's attempt fails. The inner unit records that attempt as cancelled, and
    the outer one the attempt under way; neither records a failure or retries."""
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
    assert failed == []
    assert stopped == [("inner", 1), ("outer", 1)]


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
    _check_nothing_kept(agent, turns)
    sent = [
        (r["scope"]["decision"], r["scope"]["generation"], r["scope"]["attempt"])
        for r in _trace(tmp_path, "ModelRequestSent")
    ]
    assert sent == [(0, GENERATION, 1)]
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "cancelled", None)]
    assert _cancelled(tmp_path, "generation") == [
        ("RetryCancelled('generation abandoned before attempt 2')", None)
    ]
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
    same agent, starts with a new one (DefaultHarness._generation), so it is not cancelled: it
    sends its request and returns its action."""
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
    assert "stale" not in pending.result and pending.result.get("invalid_reason") is None
    assert pending.result["action"] is not None
    assert len(agent.history.actions) == 1
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
    assert agent.history.turns == () and agent.history.actions == ()
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "cancelled", None)]


class _WaitEntered:
    """``clock``, but ``entered`` is set as each wait on it starts."""

    def __init__(self, clock):
        self._clock = clock
        self.entered = threading.Event()

    def __getattr__(self, name):
        return getattr(self._clock, name)

    def wait(self, waitable, timeout):
        self.entered.set()
        return self._clock.wait(waitable, timeout)


def test_abort_during_request_pacing_sends_no_request(openrouter_env, tmp_path):  # noqa: F811
    """Stateless API. The request pacing interval (providers/litellm.py pace_request) keeps
    attempt 1 waiting for a minute, longer than any test waits, a wait recorded before it
    starts. The kernel aborts the generation during it: the wait ends, the request is never
    sent, and the attempt is recorded as cancelled."""
    openrouter_env.setenv("STREAM_MIN_REQUEST_INTERVAL", "60")
    with OpenRouterStub([], route=ROUTE) as stub:
        openrouter_env.setenv("OPENROUTER_BASE_URL", stub.base_url)
        episode = _stateless_episode(tmp_path)
        agent = episode.agent
        try:
            clock = agent.driver.clock = _WaitEntered(agent.driver.clock)
            openrouter_env.setattr(litellm_driver, "_LAST_REQUEST_AT", [clock.time()])
            turns = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            box, done = _generate_in_worker(episode, observation)
            assert clock.entered.wait(WAIT_S), "attempt 1 was not paced"
            _abort_like_the_kernel(agent)
            _discarded(agent, box, done)
        finally:
            episode.finish()
    assert stub.unscripted == 0 and stub.requests == []
    _check_nothing_kept(agent, turns)
    (wait,) = _trace(tmp_path, "SessionWait")
    assert (wait["wait"], wait["scope"]["generation"], wait["scope"]["attempt"]) == (
        "request_pacing",
        GENERATION,
        1,
    )
    assert 50 < wait["seconds"] <= 60
    assert _attempts(tmp_path) == [(1, "cancelled", None)]
    assert _inspection_outcomes(tmp_path) == ["superseded_by_timeout"]


def test_abort_mid_request_is_not_retried(tmp_path, monkeypatch):
    """Stateless API, abort while the request is in flight. The stub holds request 0 until the
    client closes it; the abort's driver.close() does (providers/litellm.py close). The
    failure arrives before any output and is classified TRANSIENT, but the generation is
    cancelled by then: the attempt is recorded once, as cancelled, not as a retry; no backoff
    starts and nothing is sent again. The reply scripted for a retry stays unused."""
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
    assert backoff.calls == []
    _check_nothing_kept(agent, turns)
    assert _attempts(tmp_path) == [(1, "cancelled", None)]
    assert _cancelled(tmp_path, "generation") == [
        ("RetryCancelled('generation abandoned during attempt 1')", None)
    ]


def test_a_failure_while_the_episode_runs_is_retried_and_the_aborts_is_not(tmp_path, monkeypatch):
    """Stateless API. Decision 0's first request gets HTTP 503 while the episode runs, as a
    gateway's does: that attempt is recorded as a retry, with its failure kind and delay, and
    the request is sent again. Decision 1's request is in flight when the kernel aborts its
    generation, and the abort's close fails it: that attempt is recorded once, as cancelled,
    with neither, and nothing is sent again."""
    monkeypatch.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    arrived = threading.Event()

    def hold_decision_1(index):
        if index == 2:
            arrived.set()
            return True
        return False

    unavailable = ChatFailure(503, {"error": {"message": "upstream down", "code": 503}})
    with ChatCompletionsStub(
        [unavailable, *replies([STEP, STEP])], on_request=hold_decision_1
    ) as stub:
        episode = build_episode(
            chat_completions_config(stub.base_url, stream=True),
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=CHAT_MODEL_KEY,
        )
        agent = episode.agent
        try:
            run_steps(episode, [STEP], stub)
            observation = episode.frames.capture(1)
            box, done = _generate_in_worker(episode, observation, decision=1)
            assert arrived.wait(WAIT_S)
            _abort_like_the_kernel(agent)
            _discarded(agent, box, done)
        finally:
            episode.finish()
    assert stub.held == [True] and len(stub.requests) == 3
    assert stub.unscripted == 0 and stub.unused_replies == 1
    attempts = [
        (
            a["scope"]["decision"],
            a["scope"]["attempt"],
            a["outcome"],
            a.get("failure_kind"),
            a.get("retry_delay_s") is not None,
        )
        for a in _trace(tmp_path, "ModelAttemptFinished")
        if a["kind"] == "generation"
    ]
    assert attempts == [
        (0, 1, "retry", TRANSIENT, True),
        (0, 2, "ok", None, False),
        (1, 1, "cancelled", None, False),
    ]


def test_runner_timeout_leaves_no_request_after_the_episode(tmp_path):
    """The real run_stream on a 0.5 s wall-clock episode. Decision 0's request never comes back;
    at the deadline the kernel gives the generation up, the shell aborts the agent (shell.py
    _abort) and the episode ends without reporting the decision. The abort closes the request
    and sets the generation's cancel event, so the abandoned worker does not send it again: the
    stub sees one request, and neither a turn nor an action is left behind."""
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
        result = run_loop(episode, env)
    agent = episode.agent
    assert stub.unscripted == 0 and stub.unused_replies == 1
    assert stub.held == [True]
    assert len(stub.requests) == 1
    assert _requests(turns) == [(1, True)]
    assert commit_statuses(env) == []
    assert result.timeline == []
    requests = [
        (call["request"], call["hang"], call["finalizing"])
        for call in env.calls
        if call["call"] == "model.request"
    ]
    assert requests == [(0, True, False)]
    assert agent.history.turns == ()
    assert agent.history.actions == ()
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


def test_abort_while_readying_after_a_failure_sends_no_request(openrouter_env, tmp_path):  # noqa: F811
    """Stateless API. Attempt 1 gets HTTP 503, which leaves the runtime dirty
    (RealtimeRuntime.run_turn); a stateless driver has no session to recover, so attempt 2 only
    readies the runtime (RealtimeRuntime._ready_stateless). The abort lands while it does, and
    the attempt stops at its next check: it sends no request. _run_turn's own checks before each
    are tested on the agent (tests/harness/test_generations.py)."""
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
            readying, release = threading.Event(), threading.Event()
            readied = []
            ready = agent._runtime._ready_stateless

            async def gated_ready():
                readied.append(None)
                if len(readied) == 2:  # attempt 2's
                    readying.set()
                    await asyncio.to_thread(release.wait, WAIT_S)
                return await ready()

            agent._runtime._ready_stateless = gated_ready
            turns = _count_calls(agent._runtime, "_run_turn")
            observation = episode.frames.capture(0)
            box, done = _generate_in_worker(episode, observation)
            assert readying.wait(WAIT_S), "attempt 2 did not ready the runtime"
            _abort_like_the_kernel(agent)
            release.set()
            _discarded(agent, box, done)
        finally:
            episode.finish()
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert len(stub.requests) == 1  # attempt 1's
    assert _requests(turns) == [(GENERATION, True)] * 2
    assert agent.history.turns == () and agent.history.actions == ()
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "cancelled", None)]


def test_session_abort_before_reconnecting_opens_no_session(tmp_path, monkeypatch):
    """Session API. Response 1 is dropped, and attempt 2 rebuilds the session through connect()
    (the session's ensure_ready). The abort lands before connect() opens it. connect()'s own
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
    assert agent.history.turns == () and agent.history.actions == ()
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
    assert agent.history.turns == () and agent.history.actions == ()
    assert _attempts(tmp_path) == [(1, "retry", None), (2, "cancelled", None)]
    assert _connects(tmp_path) == [
        ("initialize", 1, "ok"),
        ("decision", 1, "retry"),
        ("decision", 2, "cancelled"),
    ]
    assert _cancelled(tmp_path, "session_connect") == [
        ("RetryCancelled('generation abandoned before attempt 2')", None)
    ]


def test_reconnect_that_spends_its_retries_is_not_retried_again(tmp_path, monkeypatch):
    """Session API. The kernel aborts generation 1 before its worker starts generate(), which
    closes the session, and the next generation reconnects (the session's ensure_ready); every
    upgrade is refused (HTTP 503). connect() makes the
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
            agent.abort(GENERATION)
            wait_for_abort(agent, WAIT_S)
            observation = episode.frames.capture(1)
            budget = episode.budget(observation.tick_number)
            pending = agent.generate(GENERATION + 1, 1, observation, budget)
            agent.discard(GENERATION + 1)
        finally:
            episode.finish()
        connections = stub.finish()
    assert stub.violations == [] and stub.unscripted == 0
    assert [c["rejected"] for c in connections] == [None] + [503] * 4
    assert "stale" not in pending.result
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
    assert agent.history.turns == () and agent.history.actions == ()
    assert _questions(tmp_path) == [(0, 0, 1, "retry"), (0, 0, 2, "cancelled")]
    assert _attempts(tmp_path) == [(1, "cancelled", None)]  # the turn the question was in
    # The question never went out again, so its cancelled attempt took no time.
    assert _cancelled(tmp_path, "choice_question") == [
        ("RetryCancelled('generation abandoned before attempt 2')", None)
    ]
    # No question was answered: the generation given up on spent no usage it knows of.
    assert not (tmp_path / "provider-usage.jsonl").exists()


def test_decision_abort_between_questions(tmp_path, monkeypatch):
    """Decision API. Decision 0 clicks the name field. Decision 1's first question picks
    compose_text, and the abort lands before its first text-edit question is asked
    (selection.py _ask). That question sends nothing, and records its first attempt as
    cancelled. The question answered before is the usage of the generation given up on, kept
    as for any other model's (decision 37): in its cancelled attempt's record and in a stale
    provider-usage row, not in the agent's usage. The cancelled result the worker hands the shell
    carries it too, as what the call spent, without the question it was cut at."""
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
    cancelled = box["pending"].result
    assert asked == ["gui_action", "gui_action", "text_edit"]
    assert len(stub.requests) == 2 and stub.unscripted == 0 and stub.unload() == 4
    assert len(agent.history.turns) == len(agent.history.actions) == 1
    assert _questions(tmp_path) == [(0, 0, 1, "ok"), (1, 0, 1, "ok"), (1, 1, 1, "cancelled")]
    assert _attempts(tmp_path) == [(1, "ok", False), (1, "cancelled", None)]
    attempts = _trace(tmp_path, "ModelAttemptFinished")
    accepted, answered = (
        a["usage"] for a in attempts if a["kind"] == "choice_question" and a["outcome"] == "ok"
    )
    assert [a["usage"] for a in attempts if a["kind"] == "generation"] == [accepted, answered]
    lines = (tmp_path / "provider-usage.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [(r["tick"], r["stale"], r["status"], r["usage"]) for r in rows] == [
        (0, False, "reported", accepted),
        (1, True, "reported", answered),
    ]
    # The stale row names the generation its cancelled attempt was of.
    (given_up,) = [a for a in attempts if (a["kind"], a["outcome"]) == ("generation", "cancelled")]
    assert rows[1]["generation_id"] == given_up["scope"]["generation_id"] is not None
    assert agent.token_usage["responses"] == 1
    assert agent.token_usage["input_tokens"] == accepted["input_tokens"]
    assert (cancelled["usage"], cancelled["provider_usage_missing_reason"]) == (
        answered,
        "cancelled_request_usage_not_reported",
    )
    assert token_record(cancelled)["provider_usage_status"] == "partial"


def test_decision_closed_between_questions(tmp_path, monkeypatch):
    """Decision API. As in test_decision_abort_between_questions, but the agent is closed while
    decision 1's text-edit question is about to be asked, as main() closes it once the drain
    has given up on that worker: the close cancels the generation and stops the execution
    thread, which cancels the decision under way (harnesses/default/execution_thread.py stop).
    The question answered before is the usage of the generation given up on all the same
    (decision 37): its cancelled attempt's record and a stale provider-usage row carry it."""
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
            agent.close()
            assert done.wait(WAIT_S), "the closed worker never returned"
        finally:
            release.set()
            episode.finish()
    assert "error" not in box, repr(box.get("error"))
    assert box["pending"].result["stale"] is True
    assert asked == ["gui_action", "gui_action", "text_edit"] and len(stub.requests) == 2
    attempts = _trace(tmp_path, "ModelAttemptFinished")
    accepted, answered = (
        a["usage"] for a in attempts if a["kind"] == "choice_question" and a["outcome"] == "ok"
    )
    assert [a["usage"] for a in attempts if a["kind"] == "generation"] == [accepted, answered]
    lines = (tmp_path / "provider-usage.jsonl").read_text().splitlines()
    assert [(row["tick"], row["stale"], row["usage"]) for row in map(json.loads, lines)] == [
        (0, False, accepted),
        (1, True, answered),
    ]


def test_decision_aborted_during_its_last_question_attempt(tmp_path, monkeypatch):
    """Decision API. Decision 1's first question picks compose_text; its text-edit question gets
    HTTP 503 three times, each recorded as a retry, and the stub holds its fourth, last allowed
    attempt when the kernel aborts the generation. The abort closes the client, which ends that
    attempt with an error: though the question has no retry left, the attempt is cancelled, not
    failed, and so is the generation. The question answered before is the usage of the
    generation given up on (decision 37): the cancelled result the shell gets back carries it,
    and discarding that result records it in a stale provider-usage row."""
    monkeypatch.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    arrived, release = threading.Event(), threading.Event()
    with DecisionsStub(upstream="openai") as stub:
        respond = stub._respond

        def held(path, headers, body):
            # Request 0 is decision 0's, 1 the first question's, 2 to 4 the 503s.
            if len(stub.requests) == 5:
                arrived.set()
                release.wait(WAIT_S)
            return respond(path, headers, body)

        monkeypatch.setattr(stub, "_respond", held)
        episode = _decision_episode(stub, tmp_path, monkeypatch)
        agent = episode.agent
        try:
            run_decisions(episode, [decision("click_P1")], stub)
            fail = QuestionFailure(503, "Provider returned error")
            stub.load(
                decision("compose_text", fail, fail, fail, ADA, SPACE, LOVELACE, "finish").script
            )
            observation = episode.frames.capture(1)
            box, done = _generate_in_worker(episode, observation, decision=1)
            assert arrived.wait(WAIT_S), "the last attempt of the text-edit question never came"
            _abort_like_the_kernel(agent)
            assert done.wait(WAIT_S), "the abandoned worker never returned"
            agent.discard(GENERATION)
        finally:
            release.set()
            episode.finish()
    assert "error" not in box, repr(box.get("error"))
    result = box["pending"].result
    attempts = _trace(tmp_path, "ModelAttemptFinished")
    accepted, answered = (
        a["usage"] for a in attempts if a["kind"] == "choice_question" and a["outcome"] == "ok"
    )
    assert _questions(tmp_path) == [
        (0, 0, 1, "ok"),
        (1, 0, 1, "ok"),
        *[(1, 1, attempt, "retry") for attempt in (1, 2, 3)],
        (1, 1, 4, "cancelled"),
    ]
    assert result["stale"] is True and "invalid_reason" not in result
    assert (result["usage"], result["provider_usage_missing_reason"]) == (
        answered,
        "cancelled_request_usage_not_reported",
    )
    assert [(a["outcome"], a["usage"]) for a in attempts if a["kind"] == "generation"] == [
        ("ok", accepted),
        ("cancelled", answered),
    ]
    lines = (tmp_path / "provider-usage.jsonl").read_text().splitlines()
    assert [(row["tick"], row["stale"], row["usage"]) for row in map(json.loads, lines)] == [
        (0, False, accepted),
        (1, True, answered),
    ]
    assert agent.token_usage["input_tokens"] == accepted["input_tokens"]
