"""Replay baselines for the openrouter API: LiteLLMDriver's OpenRouterDriver subclass
(providers/litellm.py OpenRouterDriver).

The model is the catalog entry claude-opus-5 (configs/models/claude-opus-5.yaml:1-6): model
anthropic/claude-opus-5 pinned to upstream anthropic, reasoning effort low, and the native output
limit 128000 from LiteLLM's bundled model map (providers/registry.py _litellm_model_map,
litellm_output_limit, resolve_output_limit). The entry has no base_url, so the driver takes
OPENROUTER_BASE_URL (providers/litellm.py LiteLLMDriver.__init__), here the stub.

Record with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it the run is
compared with tests/replay/baselines/.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

import realtime_framework.entity.model as model_module
import realtime_framework.providers.registry as catalog
from realtime_framework.entity.events import GenerationBoundary, TextDelta, UsageSnapshot
from realtime_framework.entity.model import RATE_LIMITED, TRANSIENT, TRANSPORT_RETRY
from realtime_framework.entity.model import TransportFailure, exception_chain, retry_after_seconds

from . import baseline
from .build import build_episode
from .frames import ReplayTask
from .scenario import LONG_SCENARIO, TASK_INSTRUCTION, replies
from .steps import run_steps
from .stub import Reply
from .stub_openrouter import Cut, Failure, Held, OpenRouterStub, route_metadata
from .test_chat_completions import _check_history_changes, _check_rejection_feedback

BASELINES = Path(__file__).parent / "baselines"
MODEL_KEY = "claude-opus-5"
DUMMY_KEY = "replay-dummy-openrouter-key"
NATIVE_OUTPUT_LIMIT = 128000
# OpenRouter names the upstream "Anthropic"; its slug is the pinned upstream_provider "anthropic"
# (providers/litellm.py LiteLLMDriver._provider_slug, LiteLLMDriver._check_route).
ROUTE = route_metadata("anthropic/claude-opus-5", "Anthropic")

# The finite token clock: B charged tokens per frame over an N-tick episode (N is two minutes at
# 1 fps, runner.py main and kernel/turns.py TurnConfig.max_ticks). Decision k observes tick k
# (steps.py), and its cap min(128000, (N - k) * B) (kernel/turns.py _captured, episode_records.py
# decision_budget) is the native limit through k = 40 (exactly 128000 at k = 40) and the remaining
# clock from k = 41 on (126400, down to 97600 at k = 59), never zero in the scenario.
TOKEN_CLOCK = {"clock_mode": "token", "tokens_per_frame": 1600, "max_ticks": 120}

# Request headers that change how OpenRouter serves or attributes the request: the auth scheme,
# the body type, the route-metadata switch, and the app attribution the driver adds
# (providers/litellm.py LiteLLMDriver._transport_kwargs). Names are compared lowercased. Not
# compared: host (it carries the stub's port), content-length, accept, accept-encoding, connection
# and user-agent.
COMPARED_HEADERS = frozenset(
    {"authorization", "content-type", "http-referer", "x-openrouter-metadata", "x-title"}
)


@pytest.fixture
def openrouter_env(monkeypatch):
    """The driver reads its key when it is built (providers/litellm.py LiteLLMDriver._resolve_key)
    and the two attribution header values on every request (providers/litellm.py
    LiteLLMDriver._transport_kwargs)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", DUMMY_KEY)
    monkeypatch.delenv("OPENROUTER_REFERER", raising=False)
    monkeypatch.delenv("OPENROUTER_TITLE", raising=False)
    return monkeypatch


def openrouter_config():
    """The ModelConfig runner.main() builds for ``--model claude-opus-5`` (runner.py main)."""
    entry = catalog.get(MODEL_KEY)
    return catalog.to_model_config(
        MODEL_KEY, fps=1.0, reasoning_effort=entry.defaults["reasoning_effort"]
    )


@contextmanager
def openrouter_episode(monkeypatch, tmp_path, script, **clock):
    with OpenRouterStub(script, route=ROUTE) as stub:
        monkeypatch.setenv("OPENROUTER_BASE_URL", stub.base_url)
        config = openrouter_config()
        if clock.get("tokens_per_frame") is not None:  # runner.py main
            assert catalog.adapter_class(config.adapter_type).capabilities.finite_frame_budget
        episode = build_episode(
            config,
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=MODEL_KEY,
            **clock,
        )
        try:
            yield stub, config, episode
        finally:
            episode.agent.close()
    assert stub.unscripted == 0, "the agent sent requests beyond the script"
    assert stub.unused_replies == 0, "scripted replies were left unused"


def with_headers(records: list[dict], received) -> list[dict]:
    """run_steps stores each request's path and body; add its compared headers, in arrival
    order."""
    arrivals = iter(received)
    for record in records:
        for request in record["requests"]:
            headers = next(arrivals).headers
            request["headers"] = {
                name.lower(): value for name, value in headers if name.lower() in COMPARED_HEADERS
            }
    assert next(arrivals, None) is None
    return records


def header_record(config, scenario: str, clock: dict) -> dict:
    return {
        "record": "header",
        "api": "openrouter",
        "model_key": MODEL_KEY,
        "scenario": scenario,
        "clock": clock,
        "config": config.model_dump(),
    }


def check_baseline(name: str, header: dict, records: list[dict], run_dir: Path) -> list[dict]:
    actual = baseline.Normalizer()([header, *records, *baseline.trajectory(run_dir)])
    path = BASELINES / name
    if os.environ.get("REPLAY_RECORD") == "1":
        baseline.write(path, actual)
        return actual
    problems = baseline.compare(baseline.read(path), actual)
    assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
        problems[:60]
    )
    return actual


def check_one_session(actual: list[dict]) -> None:
    """Every request of one driver carries its one session id (providers/litellm.py
    LiteLLMDriver.__init__, LiteLLMDriver._request_kwargs), which the baseline stores as a random
    id."""
    sessions = {
        request["body"]["session_id"]
        for record in actual
        if record["record"] == "step"
        for request in record["requests"]
    }
    assert sessions == {"<session-1>"}


def check_cache_breakpoints(records: list[dict]) -> None:
    """anthropic/ models get one ephemeral breakpoint, on the last content part of the newest
    message (providers/litellm.py _mark_cache_breakpoint, LiteLLMDriver._request_kwargs)."""
    for record in records:
        for request in record["requests"]:
            messages = request["body"]["messages"]
            marked = [
                (i, j, part["cache_control"])
                for i, message in enumerate(messages)
                if isinstance(message["content"], list)
                for j, part in enumerate(message["content"])
                if "cache_control" in part
            ]
            newest = len(messages) - 1
            assert marked == [(newest, len(messages[newest]["content"]) - 1, {"type": "ephemeral"})]


def test_long_scenario(openrouter_env, tmp_path):
    """LONG_SCENARIO on the finite token clock. The first screenshot fold and the first archive
    fall on the same decisions as for chat_completions: both APIs are stateless and share the
    history policies (harnesses/default/context.py RealtimeRuntime._retained). The scripted statuses
    are the wall-clock ones; the last, not_applied_time_limit, cannot occur on the token clock
    (kernel/turns.py start, _past_deadline), but it reaches no request."""
    with openrouter_episode(openrouter_env, tmp_path, replies(LONG_SCENARIO), **TOKEN_CLOCK) as (
        stub,
        config,
        episode,
    ):
        records = run_steps(episode, LONG_SCENARIO, stub)
    with_headers(records, stub.requests)
    assert all(len(r["requests"]) == 1 for r in records)
    assert all(r["result"]["inspection"]["valid"] for r in records)
    _check_history_changes(records)
    _check_rejection_feedback(records)
    check_cache_breakpoints(records)
    # Each request sends the decision's cap as max_tokens (providers/litellm.py
    # LiteLLMDriver.transport_output_token_cap, LiteLLMDriver._request_kwargs).
    caps = [r["requests"][0]["body"]["max_tokens"] for r in records]
    rate, horizon = TOKEN_CLOCK["tokens_per_frame"], TOKEN_CLOCK["max_ticks"]
    assert caps == [
        min(NATIVE_OUTPUT_LIMIT, (horizon - r["observation"]["tick"]) * rate) for r in records
    ]
    assert caps[40] == NATIVE_OUTPUT_LIMIT and caps[41] < NATIVE_OUTPUT_LIMIT and min(caps) > 0
    actual = check_baseline(
        "openrouter-token-long.jsonl",
        header_record(config, "long", TOKEN_CLOCK),
        records,
        tmp_path,
    )
    check_one_session(actual)


def test_pause_mode_snapshot(openrouter_env, tmp_path):
    """The token clock in pause mode (tokens_per_frame=None, runner.py main). Nothing bounds
    a decision but the native limit, which the driver sends because the harness asks for no
    provider cap (harnesses/default/harness.py DefaultHarness.supports_unbounded_output;
    providers/litellm.py LiteLLMDriver.transport_output_token_cap), and the clock prompt says that B
    is unlimited (harnesses/default/prompts.py budget_clock_prompt)."""
    steps = LONG_SCENARIO[:3]
    clock = {"clock_mode": "token", "tokens_per_frame": None, "max_ticks": 120}
    with openrouter_episode(openrouter_env, tmp_path, replies(steps), **clock) as (
        stub,
        config,
        episode,
    ):
        records = run_steps(episode, steps, stub)
    with_headers(records, stub.requests)
    assert all(len(r["requests"]) == 1 for r in records)
    assert [r["requests"][0]["body"]["max_tokens"] for r in records] == [NATIVE_OUTPUT_LIMIT] * 3
    assert "B is unlimited" in records[0]["requests"][0]["body"]["messages"][0]["content"]
    actual = check_baseline(
        "openrouter-pause-short.jsonl",
        header_record(config, "pause-short", clock),
        records,
        tmp_path,
    )
    check_one_session(actual)


class _Sleeps:
    """Stands in for the retry executor's backoff and records its delays: it waits them out for
    real (entity/model.py _backoff)."""

    def __init__(self):
        self.calls = []

    def __call__(self, clock, cancel, seconds):
        self.calls.append(seconds)


def _spy_classifier(driver) -> list[tuple[TransportFailure, list]]:
    """Record each classify_transport_error result (providers/litellm.py
    LiteLLMDriver.classify_transport_error) with the Retry-After of every exception in the failure's
    chain. The harness looks the classifier up on the driver for every decision
    (harnesses/default/harness.py DefaultHarness._generate)."""
    calls = []
    classify = driver.classify_transport_error

    def spy(exc):
        failure = classify(exc)
        carried = [
            retry_after_seconds(getattr(getattr(current, "response", None), "headers", None))
            for current in exception_chain(exc)
        ]
        calls.append((failure, carried))
        return failure

    driver.classify_transport_error = spy
    return calls


def _retry_scenario(monkeypatch, tmp_path, failure: Failure, scenario: str):
    """One failed attempt before any output, then two decisions. The harness classifies the error,
    sleeps the shared backoff and sends the same request again, rebuilt from canonical history
    (harnesses/default/harness.py DefaultHarness._generate); the failed attempt leaves no trace in
    later requests."""
    steps = LONG_SCENARIO[:2]
    sleeps = _Sleeps()
    monkeypatch.setattr(model_module, "_backoff", sleeps)
    with openrouter_episode(monkeypatch, tmp_path, [failure, *replies(steps)], **TOKEN_CLOCK) as (
        stub,
        config,
        episode,
    ):
        classified = _spy_classifier(episode.agent.driver)
        records = run_steps(episode, steps, stub)
    failed, retried, _ = stub.requests
    assert failed.body == retried.body
    with_headers(records, stub.requests)
    assert [len(r["requests"]) for r in records] == [2, 1]
    roles = [m["role"] for m in records[1]["requests"][0]["body"]["messages"]]
    assert roles == ["system", "user", "assistant", "user", "user"]
    assert all(r["result"].get("invalid_reason") is None for r in records)
    (classification,) = classified
    runtime = records[0]["result"]["realtime_runtime"]
    assert runtime["transport_reconnect_attempts"] == 1
    # The first retry waits 1 s +-25%, jittered by the decision's observation
    # (entity/model.py TransportRetryPolicy).
    key = f"{records[0]['observation']['observation_id']}:turn"
    assert sleeps.calls == runtime["transport_reconnect_backoff_s"]
    assert sleeps.calls == [TRANSPORT_RETRY.delay_s(1, key, classification[0])]
    assert 0.75 <= sleeps.calls[0] <= 1.25
    actual = check_baseline(
        f"openrouter-{scenario}.jsonl",
        header_record(config, scenario, TOKEN_CLOCK),
        records,
        tmp_path,
    )
    check_one_session(actual)
    return classification, key


def test_whole_turn_retry_503(openrouter_env, tmp_path):
    """HTTP 503 before any output is TRANSIENT (entity/model.py:424-425) and retried once."""
    failure = Failure(503, {"error": {"message": "upstream down", "code": 503}})
    (kind, _), _ = _retry_scenario(openrouter_env, tmp_path, failure, "retry-503")
    assert kind == TransportFailure(TRANSIENT)


def test_whole_turn_429_with_a_long_retry_after(openrouter_env, tmp_path):
    """HTTP 429 with ``Retry-After: 120``. LiteLLM's RateLimitError, the first exception in the
    chain, has no Retry-After on its response; the HTTP error it wraps does, and
    classify_transport_error reads it from there (providers/litellm.py _retry_after). The shared
    policy does not wait out a Retry-After above 30 s (entity/model.py TransportRetryPolicy): the
    request is not sent again, and the decision fails as provider_rate_limited with nothing in
    the history."""
    sleeps = _Sleeps()
    openrouter_env.setattr(model_module, "_backoff", sleeps)
    failure = Failure(
        429, {"error": {"message": "rate limited", "code": 429}}, (("Retry-After", "120"),)
    )
    with openrouter_episode(openrouter_env, tmp_path, [failure], **TOKEN_CLOCK) as (
        stub,
        config,
        episode,
    ):
        classified = _spy_classifier(episode.agent.driver)
        agent = episode.agent
        observation = episode.frames.capture(0)
        agent.configure_decision(episode.budget(observation.tick_number))
        agent.stream_prefill(observation)
        result = agent.stream_generate(tick=0)
        agent.finalize_decision_audit(result)
    ((kind, carried),) = classified
    assert kind == TransportFailure(RATE_LIMITED, 120.0)
    assert carried[0] is None and 120.0 in carried
    assert sleeps.calls == []
    assert result["invalid_reason"] == "provider_rate_limited"
    assert agent.action_history == []
    records = [
        {
            "record": "step",
            "step": 0,
            "observation": {
                "observation_id": observation.metadata["observation_id"],
                "tick": observation.tick_number,
                "timestamp": observation.timestamp,
                "screenshot_sha256": observation.metadata["screenshot_sha256"],
            },
            "requests": [
                {"path": received.path, "body": json.loads(received.body)}
                for received in stub.requests
            ],
            "result": result,
            "token_usage": agent.token_usage,
        }
    ]
    with_headers(records, stub.requests)
    actual = check_baseline(
        "openrouter-retry-429.jsonl",
        header_record(config, "retry-429", TOKEN_CLOCK),
        records,
        tmp_path,
    )
    check_one_session(actual)


def test_an_error_recording_a_retry_propagates(openrouter_env, tmp_path):
    """Attempt 1 gets HTTP 503, and recording that failure raises (harnesses/default/harness.py
    _attempt_failed, standing in for a broken trace write). The retry executor passes the error
    on (entity/model.py RequestRetry), and the decision raises it: only the request failure that
    record() kept as final becomes an invalid decision."""
    openrouter_env.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    unavailable = Failure(503, {"error": {"message": "upstream down", "code": 503}})
    with openrouter_episode(openrouter_env, tmp_path, [unavailable], **TOKEN_CLOCK) as (
        stub,
        _,
        episode,
    ):
        agent = episode.agent

        def broken(*args, **kwargs):
            raise RuntimeError("the trace refused the record")

        agent._attempt_failed = broken
        observation = episode.frames.capture(0)
        agent.configure_decision(episode.budget(observation.tick_number))
        agent.stream_prefill(observation)
        with pytest.raises(RuntimeError, match="the trace refused the record"):
            agent.stream_generate(tick=0)
    assert len(stub.requests) == 1


def test_an_error_recording_the_final_failure_propagates(openrouter_env, tmp_path):
    """Attempt 1 gets HTTP 401, which no retry follows, and recording that failure raises. The
    decision raises the record's error, which the shell ends the run with as generation_error
    (evaluation/shell.py), instead of ending as the invalid decision of the request failure that
    a swallowed record left before the shared retry executor."""
    refused = Failure(401, {"error": {"message": "bad key", "code": 401}})
    with openrouter_episode(openrouter_env, tmp_path, [refused], **TOKEN_CLOCK) as (
        stub,
        _,
        episode,
    ):
        agent = episode.agent

        def broken(*args, **kwargs):
            raise RuntimeError("the trace refused the record")

        agent._attempt_failed = broken
        observation = episode.frames.capture(0)
        agent.configure_decision(episode.budget(observation.tick_number))
        agent.stream_prefill(observation)
        with pytest.raises(RuntimeError, match="the trace refused the record"):
            agent.stream_generate(tick=0)
    assert len(stub.requests) == 1


def test_partial_output_is_not_retried(openrouter_env, tmp_path):
    """The stream breaks off after its first text chunk: LiteLLM raises MidStreamFallbackError
    over httpx.ReadError, which the classifier takes as TRANSIENT (providers/litellm.py
    classify_transport_error). But the turn already received output, and the harness never sends
    such a turn again (harnesses/default/harness.py stream_generate, ``replayable``): a retry
    could repeat an action. The decision fails as provider_stream_interrupted after one
    request."""
    openrouter_env.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    (reply,) = replies(LONG_SCENARIO[:1])
    with openrouter_episode(openrouter_env, tmp_path, [Cut(reply)], **TOKEN_CLOCK) as (
        stub,
        _,
        episode,
    ):
        classified = _spy_classifier(episode.agent.driver)
        agent = episode.agent
        observation = episode.frames.capture(0)
        agent.configure_decision(episode.budget(observation.tick_number))
        agent.stream_prefill(observation)
        result = agent.stream_generate(tick=0)
    assert len(stub.requests) == 1
    ((failure, _),) = classified
    assert failure == TransportFailure(TRANSIENT)
    assert result["invalid_reason"] == "provider_stream_interrupted"
    assert result["output_chunks"]
    assert result["_trajectory_entry"]["transport_reconnect_attempts"] == 0


def _record_stream(driver, stub, action_text: str) -> tuple[list[list[dict]], list, list]:
    """Wrap the driver so the test sees each request's clock mode and every event the runtime
    reads (harnesses/default/context.py RealtimeRuntime._run_turn). Once the runtime has read the
    complete action text, the stub releases the held usage chunk."""
    streams, clock_modes, released = [], [], []
    start_generation, events = driver.start_generation, driver.events

    async def recorded_start(request):
        clock_modes.append(request.clock_mode)
        return await start_generation(request)

    async def recorded_events(generation):
        stream, visible = [], []
        streams.append(stream)
        async for event in events(generation):
            entry = {"event": type(event).__name__, "sequence": event.sequence}
            entry["raw_type"] = event.raw_type
            if isinstance(event, TextDelta):
                entry["channel"] = event.channel
                if event.channel == "visible":
                    visible.append(event.text)
            if isinstance(event, GenerationBoundary):
                entry["kind"] = event.kind
            stream.append(entry)
            yield event
            if "".join(visible) == action_text and not released:
                released.append([dict(item) for item in stream])
                stub.release_last_event()

    driver.start_generation, driver.events = recorded_start, recorded_events
    return streams, clock_modes, released


@pytest.mark.parametrize("clock_mode", ["token", "wall"])
def test_usage_only_with_last_event(clock_mode, openrouter_env, tmp_path):
    """Usage arrives only with the last event, after the whole action text.

    The openrouter path declares terminal_output_usage=True (providers/litellm.py _STREAMING_CHAT),
    but nothing reads that flag, and nothing on this path lets the clock mode decide whether
    terminal usage may delay a wall-clock action: the clock mode reaches the driver only as
    GenerationRequest.clock_mode (harnesses/registry.py configure_agent;
    harnesses/default/harness.py DefaultHarness._generate), which no driver reads (the
    non-streaming litellm gateway's check that did is now the pre-run check_wall_clock,
    providers/preflight.py check_wall_clock). On either clock the runtime reads every event before
    the decision returns (harnesses/default/context.py RealtimeRuntime._run_turn), and the
    collector keeps the last usage (harnesses/default/harness.py
    _GenerationOutputCollector.consume). The usage matters to the runner (kernel/turns.py _accept):
    on the finite token clock the output count places the action and its absence invalidates the
    run; on the wall clock the action lands where the clock is when the decision returns, so the
    wait delays it.

    Decision 0: the stub holds the usage chunk until the runtime has read the complete action
    text; the decision returns only after it, with that usage. Decision 1: the stream ends without
    a usage chunk, which leaves provider_usage_missing_reason set (harness.py
    VisionAgentBase._protocol_token_fields).
    """
    steps = LONG_SCENARIO[:2]
    held, unreported = replies(steps)
    script = [Held(held), Reply(unreported.content, None, unreported.reasoning)]
    clock = TOKEN_CLOCK if clock_mode == "token" else {"clock_mode": "wall"}
    with openrouter_episode(openrouter_env, tmp_path, script, **clock) as (stub, config, episode):
        streams, clock_modes, released = _record_stream(episode.agent.driver, stub, held.content)
        records = run_steps(episode, steps, stub)
    with_headers(records, stub.requests)
    assert clock_modes == [clock_mode, clock_mode]
    # The release came from the runtime reading the action text, not from the stub's timeout,
    # and no usage had been read by then.
    assert not stub.hold_timed_out
    (seen,) = released
    assert all(item["event"] == "TextDelta" for item in seen)
    assert streams[0][: len(seen)] == seen
    assert [item["event"] for item in streams[0][len(seen) :]] == [
        UsageSnapshot.__name__,
        GenerationBoundary.__name__,
    ]
    first, second = (record["result"] for record in records)
    assert first["usage"] == held.usage
    assert first["provider_output_tokens"] == held.usage["completion_tokens"]
    assert first["provider_usage_missing_reason"] is None
    assert UsageSnapshot.__name__ not in [item["event"] for item in streams[1]]
    assert second["usage"] is None and second["provider_output_tokens"] is None
    assert second["provider_usage_missing_reason"] == "backend_did_not_return_usage"
    for record, stream in zip(records, streams):
        record["events"] = stream
    actual = check_baseline(
        f"openrouter-{clock_mode}-usage-last.jsonl",
        header_record(config, "usage-last", clock),
        records,
        tmp_path,
    )
    check_one_session(actual)


def test_a_route_to_another_upstream_fails_the_decision(openrouter_env, tmp_path):
    """OpenRouter names the provider that served the request in the route metadata that comes
    with the usage (LiteLLMDriver._route_metadata). A provider other than the pinned upstream
    fails the attempt (LiteLLMDriver._check_route), though the reply came complete, and the
    classifier takes the error as FATAL. The action text was already read, so the decision is
    provider_stream_interrupted (DefaultHarness stream_generate, ``received_output``): it is not
    sent again and takes no action."""
    (reply,) = replies(LONG_SCENARIO[:1])
    bedrock = route_metadata("anthropic/claude-opus-5", "Amazon Bedrock")
    with OpenRouterStub([reply], route=bedrock) as stub:
        openrouter_env.setenv("OPENROUTER_BASE_URL", stub.base_url)
        episode = build_episode(
            openrouter_config(),
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=MODEL_KEY,
            **TOKEN_CLOCK,
        )
        try:
            agent = episode.agent
            observation = episode.frames.capture(0)
            agent.configure_decision(episode.budget(observation.tick_number))
            agent.stream_prefill(observation)
            result = agent.stream_generate(tick=0)
        finally:
            episode.finish()
    assert len(stub.requests) == 1 and stub.unused_replies == 0
    error = repr(RuntimeError("OpenRouter selected 'Amazon Bedrock'; expected pinned 'anthropic'"))
    assert result["action"] is None and result["output_chunks"]
    assert result["invalid_reason"] == "provider_stream_interrupted"
    assert result["invalid_detail"] == error
    events = map(json.loads, (tmp_path / "episode_trace.jsonl").read_text().splitlines())
    (attempt,) = [
        e["payload"]
        for e in events
        if e["event"] == "ModelAttemptFinished" and e["payload"]["kind"] == "generation"
    ]
    assert (attempt["outcome"], attempt["failure_kind"], attempt["error"]) == (
        "failed",
        "fatal",
        error,
    )
    assert attempt["received_output"] is True


@pytest.mark.parametrize(
    "key, provider, marked",
    [("gemini-3.8-flash", "Google AI Studio", True), ("gpt-5.6-sol", "OpenAI", False)],
    ids=["google", "openai"],
)
def test_cache_breakpoint_by_model_prefix(key, provider, marked, openrouter_env, tmp_path):
    """Behind OpenRouter, google/ models get the explicit breakpoint anthropic/ ones get, on the
    last content part of the newest message (ChatCompat cache_breakpoint "model_prefix",
    LiteLLMDriver._request_kwargs); openai/ models cache implicitly and get none. The stub
    reports each catalog entry's pinned upstream as the provider that served it."""
    steps = LONG_SCENARIO[:2]
    entry = catalog.get(key)
    route = route_metadata(entry.model_name, provider)
    with OpenRouterStub(replies(steps), route=route) as stub:
        openrouter_env.setenv("OPENROUTER_BASE_URL", stub.base_url)
        config = catalog.to_model_config(
            key, fps=1.0, reasoning_effort=entry.defaults["reasoning_effort"]
        )
        episode = build_episode(
            config, tmp_path, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key=key
        )
        try:
            records = run_steps(episode, steps, stub)
        finally:
            episode.agent.close()
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert all(r["result"]["inspection"]["valid"] for r in records)
    assert [r["requests"][0]["body"]["model"] for r in records] == [entry.model_name] * 2
    if marked:
        check_cache_breakpoints(records)
    else:
        parts = [
            part
            for record in records
            for message in record["requests"][0]["body"]["messages"]
            if isinstance(message["content"], list)
            for part in message["content"]
        ]
        assert parts and not any("cache_control" in part for part in parts)
