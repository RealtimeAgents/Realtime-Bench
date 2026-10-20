"""Replay baselines for the openrouter API: LiteLLMDriver with the openrouter wire policy
(providers/litellm.py:346-356).

The model is the catalog entry claude-opus-5 (configs/models/claude-opus-5.yaml:1-6): model
anthropic/claude-opus-5 pinned to upstream anthropic, reasoning effort low, and the native output
limit 128000 from LiteLLM's bundled model map (providers/registry.py:190-233). The entry has no
base_url, so the driver takes OPENROUTER_BASE_URL (providers/litellm.py:472-477), here the stub.

Record with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it the run is
compared with tests/replay/baselines/.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.events import GenerationBoundary, TextDelta, UsageSnapshot
from realtime_framework.entity.model import RATE_LIMITED, TRANSIENT, TRANSPORT_RETRY
from realtime_framework.entity.model import TransportFailure, exception_chain, retry_after_seconds
from realtime_framework.harnesses.default import harness as harness_module

from . import baseline
from .build import build_episode
from .frames import ReplayTask
from .scenario import LONG_SCENARIO, TASK_INSTRUCTION, replies
from .steps import run_steps
from .stub import Reply
from .stub_openrouter import Failure, Held, OpenRouterStub, route_metadata
from .test_chat_completions import _check_history_changes, _check_rejection_feedback

BASELINES = Path(__file__).parent / "baselines"
MODEL_KEY = "claude-opus-5"
DUMMY_KEY = "replay-dummy-openrouter-key"
NATIVE_OUTPUT_LIMIT = 128000
# OpenRouter names the upstream "Anthropic"; its slug is the pinned upstream_provider "anthropic"
# (providers/litellm.py:722-724, 744-752).
ROUTE = route_metadata("anthropic/claude-opus-5", "Anthropic")

# The finite token clock: B charged tokens per frame over an N-tick episode (N is two minutes at
# 1 fps, runner.py:1773). Decision k observes tick k (steps.py), and its cap
# min(128000, (N - k) * B) (runner.py:864-879) is the native limit through k = 40 (exactly 128000
# at k = 40) and the remaining clock from k = 41 on (126400, down to 97600 at k = 59), never zero
# in the scenario.
TOKEN_CLOCK = {"clock_mode": "token", "tokens_per_frame": 1600, "max_ticks": 120}

# Request headers that change how OpenRouter serves or attributes the request: the auth scheme,
# the body type, the route-metadata switch, and the app attribution the driver adds
# (providers/litellm.py:618-625). Names are compared lowercased. Not compared: host (it carries the
# stub's port), content-length, accept, accept-encoding, connection and user-agent.
COMPARED_HEADERS = frozenset(
    {"authorization", "content-type", "http-referer", "x-openrouter-metadata", "x-title"}
)


@pytest.fixture
def openrouter_env(monkeypatch):
    """The driver reads its key when it is built (providers/litellm.py:496-500) and the two
    attribution header values on every request (providers/litellm.py:621-624)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", DUMMY_KEY)
    monkeypatch.delenv("OPENROUTER_REFERER", raising=False)
    monkeypatch.delenv("OPENROUTER_TITLE", raising=False)
    return monkeypatch


def openrouter_config():
    """The ModelConfig runner.main() builds for ``--model claude-opus-5`` (runner.py:1422-1428,
    1539-1544)."""
    entry = catalog.get(MODEL_KEY)
    return catalog.to_model_config(
        MODEL_KEY, fps=1.0, reasoning_effort=entry.defaults["reasoning_effort"]
    )


@contextmanager
def openrouter_episode(monkeypatch, tmp_path, script, **clock):
    with OpenRouterStub(script, route=ROUTE) as stub:
        monkeypatch.setenv("OPENROUTER_BASE_URL", stub.base_url)
        config = openrouter_config()
        if clock.get("tokens_per_frame") is not None:  # runner.py:1449-1456
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
    """Every request of one driver carries its one session id (providers/litellm.py:483, 565),
    which the baseline stores as a random id."""
    sessions = {
        request["body"]["session_id"]
        for record in actual
        if record["record"] == "step"
        for request in record["requests"]
    }
    assert sessions == {"<session-1>"}


def check_cache_breakpoints(records: list[dict]) -> None:
    """anthropic/ models get one ephemeral breakpoint, on the last content part of the newest
    message (providers/litellm.py:293-302, 547-552)."""
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
    history policies (harnesses/default/context.py:251-293). The scripted statuses are the
    wall-clock ones; the last, not_applied_time_limit, cannot occur on the token clock
    (runner.py:524-525; sandbox/actions.py:274-275), but it reaches no request."""
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
    # Each request sends the decision's cap as max_tokens (providers/litellm.py:527-536, 561-563).
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
    """The token clock in pause mode (tokens_per_frame=None, runner.py:1401-1402). Nothing bounds
    a decision but the native limit, which the driver sends because the harness asks for no
    provider cap (harnesses/default/harness.py:920; providers/litellm.py:527-536), and the clock
    prompt says that B is unlimited (harnesses/default/prompts.py:182-187)."""
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
    """Stands in for the time module the harness uses and records its sleeps: the retry loop
    sleeps its backoff for real (harnesses/default/harness.py:1271)."""

    def __init__(self):
        self.calls = []

    def sleep(self, seconds):
        self.calls.append(seconds)

    def __getattr__(self, name):
        return getattr(time, name)


def _spy_classifier(driver) -> list[tuple[TransportFailure, list]]:
    """Record each classify_transport_error result (providers/litellm.py:1042-1064) with the
    Retry-After of every exception in the failure's chain. The harness looks the classifier up on
    the driver for every decision (harnesses/default/harness.py:1225)."""
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
    (harnesses/default/harness.py:1231-1273); the failed attempt leaves no trace in later
    requests."""
    steps = LONG_SCENARIO[:2]
    sleeps = _Sleeps()
    monkeypatch.setattr(harness_module, "time", sleeps)
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
    # The first retry waits 1 s +-25%, jittered by the generation id (entity/model.py:318-336).
    assert sleeps.calls == runtime["transport_reconnect_backoff_s"]
    assert sleeps.calls == [TRANSPORT_RETRY.delay_s(1, runtime["generation_id"], classification[0])]
    assert 0.75 <= sleeps.calls[0] <= 1.25
    actual = check_baseline(
        f"openrouter-{scenario}.jsonl",
        header_record(config, scenario, TOKEN_CLOCK),
        records,
        tmp_path,
    )
    check_one_session(actual)
    return classification, runtime


def test_whole_turn_retry_503(openrouter_env, tmp_path):
    """HTTP 503 before any output is TRANSIENT (entity/model.py:375-376) and retried once."""
    failure = Failure(503, {"error": {"message": "upstream down", "code": 503}})
    (kind, _), _ = _retry_scenario(openrouter_env, tmp_path, failure, "retry-503")
    assert kind == TransportFailure(TRANSIENT)


def test_whole_turn_retry_429_with_retry_after(openrouter_env, tmp_path):
    """HTTP 429 with ``Retry-After: 120``, as today's code handles it: retried after the normal
    first backoff. The policy would not wait out a Retry-After above 30 s; it would end the
    decision instead (entity/model.py:323-331). The header is lost because
    classify_transport_error stops at the first exception in the chain that has a status
    (providers/litellm.py:1047-1051): LiteLLM's RateLimitError, whose response has no Retry-After,
    while exceptions further down the chain carry the 120 s."""
    failure = Failure(
        429, {"error": {"message": "rate limited", "code": 429}}, (("Retry-After", "120"),)
    )
    (kind, carried), runtime = _retry_scenario(openrouter_env, tmp_path, failure, "retry-429")
    assert kind == TransportFailure(RATE_LIMITED)  # retry_after_s is None
    assert carried[0] is None and 120.0 in carried
    honored = TransportFailure(RATE_LIMITED, 120.0)
    assert TRANSPORT_RETRY.delay_s(1, runtime["generation_id"], honored) is None


def _record_stream(driver, stub, action_text: str) -> tuple[list[list[dict]], list, list]:
    """Wrap the driver so the test sees each request's clock mode and every event the runtime
    reads (harnesses/default/context.py:518-533). Once the runtime has read the complete action
    text, the stub releases the held usage chunk."""
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

    The openrouter path declares terminal_output_usage=True (providers/litellm.py:306-310), but
    nothing reads that flag, and no code on this path does what the runner comment says adapters
    do with _clock_mode, deciding whether terminal usage may delay a committed wall-clock action
    (runner.py:1571-1573): the harness passes _clock_mode on only as GenerationRequest.clock_mode
    (harnesses/default/harness.py:1205), which the streaming driver never reads (its one reader,
    providers/litellm.py:899, is the non-streaming litellm gateway). On either clock the runtime
    reads every event before the decision returns (harnesses/default/context.py:519-533), and the
    collector keeps the last usage (harnesses/default/harness.py:804-806). The usage matters to
    the runner: on the finite token clock the output count places the action (runner.py:486-490)
    and its absence invalidates the run (runner.py:477-485); on the wall clock the action lands
    where the clock is when the decision returns (runner.py:507-510), so the wait delays it.

    Decision 0: the stub holds the usage chunk until the runtime has read the complete action
    text; the decision returns only after it, with that usage. Decision 1: the stream ends without
    a usage chunk, which leaves provider_usage_missing_reason set (harness.py:82-84).
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
