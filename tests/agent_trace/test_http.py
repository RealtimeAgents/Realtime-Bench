"""T3: every HTTP outbound path records the request it sent and the bytes it received.

Each path runs against a local stub (tests/agent_trace/stubs.py). For the hooked paths the
recorded method, URL, header names, allow-listed header values and body equal what the stub
received, and the recorded response equals what the stub served. No API key reaches the trace or
the pack, and no cookie or account id reaches the trace.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest

import realtime_framework.entity.model as model_module
from realtime_framework.clock import SystemClock
from realtime_framework.entity.actions import Observation
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import CONTEXT_POLICY_VERSION, path_policy
from realtime_framework.entity.model import CanonicalTurn, GenerationRequest, ProviderObservation
from realtime_framework.entity.records import TraceEvent, trace_scope, validate_episode_trace
from realtime_framework.entity.request import (
    AnswerRequirements,
    CallParameters,
    Candidate,
    ChoiceRequirements,
    SelectionMaterial,
)
from realtime_framework.evaluation.agent_trace import AgentTraceSink, close_agent_trace, load_body
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.harnesses.default.request import build_request, ledger_block, system_block
from realtime_framework.providers.capture import REDACTED, header_kept
from realtime_framework.providers.decisions.questions import GUI_INSTRUCTIONS
from realtime_framework.providers.decisions.selection import ChoiceDecision
from realtime_framework.providers.jev.decisions import JevDecisionsClient
from realtime_framework.providers.litellm import ChatCompletionsDriver
from realtime_framework.providers.openai_decisions import OpenAIDecisionsClient
from realtime_framework.sandbox.observation.som import FRAME_INTERACTABLES_HEADER

from ..context import row_context
from ..replay.build import build_episode
from ..replay.frames import ReplayTask
from ..replay.scenario import LONG_SCENARIO, TASK_INSTRUCTION, replies
from ..replay.steps import decide, run_steps
from ..replay.stub import _completion, _sse
from .stubs import DROP, Answer, HttpStub
from .test_pack import assert_no_secret

KEY = "sk-replay-SECRET-0123456789abcdef"  # a fake key; it must never reach the trace or pack
STEPS = LONG_SCENARIO[:2]
REPLIES = replies(STEPS)


def _events(run_dir: Path) -> list[dict]:
    validate_episode_trace(run_dir / "episode_trace.jsonl")
    lines = (run_dir / "episode_trace.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _payloads(events, name) -> list[dict]:
    return [e["payload"] for e in events if e["event"] == name]


def _assert_no_key(run_dir: Path, *private: str) -> None:
    """No key in the trace or the pack's bodies, and no request content in the (exported) trace."""
    assert_no_secret(run_dir, KEY)
    trace = (run_dir / "episode_trace.jsonl").read_text()
    for text in private:
        assert text not in trace


def _openai_answer(index, received):
    request = json.loads(received.body)
    reply = REPLIES[index]
    if request.get("stream"):
        return Answer(200, _sse(reply, request["model"], index), "text/event-stream")
    return Answer(200, _completion(reply, request["model"], index))


def _anthropic_answer(index, received):
    text = REPLIES[index].content
    events = [
        {
            "type": "message_start",
            "message": {
                "id": f"msg_replay_{index}",
                "type": "message",
                "role": "assistant",
                "model": json.loads(received.body)["model"],
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 1500, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 12},
        },
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return Answer(200, body.encode(), "text/event-stream")


def _serving(answer, headers):
    """``answer`` with ``headers`` added to every response."""
    return lambda index, received: replace(answer(index, received), headers=headers)


# api -> (channel, answer, ModelConfig fields; api_base_url is relative to the stub origin). The
# model ids are the catalog's (configs/models/): LiteLLM accepts reasoning and thinking parameters
# only for models its bundled map lists.
HOOKED = {
    "openrouter": (
        "litellm.httpx",
        _openai_answer,
        {
            "adapter_type": "openrouter",
            "model_name": "anthropic/claude-opus-5",
            "api_base_url": "/api/v1",
            "upstream_provider": "anthropic",
        },
    ),
    "anthropic_messages": (
        "litellm.httpx",
        _anthropic_answer,
        {"adapter_type": "anthropic_messages", "model_name": "claude-opus-5", "api_base_url": ""},
    ),
    "moonshot": (
        "litellm.openai_sdk",
        _openai_answer,
        {
            "adapter_type": "moonshot",
            "model_name": "kimi-k3",
            "api_base_url": "/v1",
            # K3 runs only with its past reasoning back (its YAML's context).
            "context": row_context("moonshot", "kimi-k3", model_layer={"reasoning_replay": True}),
        },
    ),
    "chat_completions-stream": (
        "litellm.openai_sdk",
        _openai_answer,
        {"adapter_type": "chat_completions", "model_name": "replay-vlm", "api_base_url": "/v1"},
    ),
    "chat_completions-nonstream": (
        "litellm.openai_sdk",
        _openai_answer,
        {
            "adapter_type": "chat_completions",
            "model_name": "replay-vlm",
            "api_base_url": "/v1",
            "chat_stream": False,
        },
    ),
}
# The generic API, which LiteLLM's logger callback records instead of a hooked client.
GENERIC = {
    "adapter_type": "litellm",
    "model_name": "openai/replay-vlm",
    "api_base_url": "/v1",
    "chat_stream": False,
}


def _config(stub: HttpStub, fields: dict, monkeypatch) -> ModelConfig:
    monkeypatch.setenv("REPLAY_TRACE_KEY", KEY)
    stream = fields.get("chat_stream", True)
    return ModelConfig(
        **{**fields, "api_base_url": stub.origin + fields["api_base_url"]},
        api_key_env="REPLAY_TRACE_KEY",
        chat_include_usage=stream,
        native_output_limit=4096,
        native_output_limit_source="endpoint",
        reasoning_effort="low",
        fps=1.0,
    )


def _run(config: ModelConfig, stub: HttpStub, run_dir: Path, steps=STEPS) -> list[dict]:
    episode = build_episode(
        config, run_dir, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key="replay-model"
    )
    try:
        run_steps(episode, steps, stub)
    finally:
        assert episode.finish() is None
    return _events(run_dir)


@pytest.mark.parametrize("api", sorted(HOOKED))
def test_hooked_paths_record_the_wire(api, tmp_path, monkeypatch):
    channel, answer, fields = HOOKED[api]
    with HttpStub(answer) as stub:
        events = _run(_config(stub, fields, monkeypatch), stub, tmp_path)
    pack = tmp_path / "episode_blobs.pack"
    sent = _payloads(events, "ModelRequestSent")
    received = _payloads(events, "ModelResponseReceived")
    assert len(sent) == len(received) == len(stub.requests) == len(STEPS)
    for k, (request, response, wire, served) in enumerate(
        zip(sent, received, stub.requests, stub.served)
    ):
        assert request["channel"] == channel and request["exact_wire"]
        assert request["scope"] == {
            "phase": "decision",
            "decision": k,
            "generation": k + 1,
            "generation_id": request["scope"]["generation_id"],
            "attempt": 1,
        }
        assert request["method"] == wire.method
        url = urlsplit(request["url"])
        assert url.path + (f"?{url.query}" if url.query else "") == wire.path
        recorded = {name.lower(): value for name, value in request["headers"]}
        on_wire = {name.lower(): value for name, value in wire.headers}
        assert recorded == {
            name: value if header_kept(name) else REDACTED for name, value in on_wire.items()
        }
        (credential,) = {"authorization", "x-api-key"} & on_wire.keys()
        assert KEY in on_wire[credential]
        assert load_body(pack, request["body"]) == wire.body
        assert response["request_seq"] == request["capture_seq"]
        assert response["status_code"] == 200 and not response["error"]
        # A streaming SDK stops reading at its end-of-stream event, before the transport's EOF.
        assert response["complete"] or json.loads(wire.body).get("stream")
        assert load_body(pack, response["body"]) == served
    _assert_no_key(tmp_path, TASK_INSTRUCTION, REPLIES[0].content)


def test_generic_litellm_api_is_recorded_from_the_logger(tmp_path, monkeypatch):
    with HttpStub(_openai_answer) as stub:
        events = _run(_config(stub, GENERIC, monkeypatch), stub, tmp_path)
    pack = tmp_path / "episode_blobs.pack"
    sent = _payloads(events, "ModelRequestSent")
    received = _payloads(events, "ModelResponseReceived")
    assert len(sent) == len(received) == len(stub.requests) == len(STEPS)
    for k, (request, response, wire) in enumerate(zip(sent, received, stub.requests)):
        assert request["channel"] == "litellm.logger_fn" and request["exact_wire"] is False
        assert request["scope"]["decision"] == k
        assert request["url"].rstrip("/") == stub.origin + "/v1"  # LiteLLM's masked api_base
        # LiteLLM's final request dict, not the wire bytes: the OpenAI SDK still merges its
        # extra_body into the JSON it sends.
        logged = json.loads(load_body(pack, request["body"]))
        extra = logged.pop("extra_body", {})
        assert {**logged, **extra} == json.loads(wire.body)
        assert response["request_seq"] == request["capture_seq"]
        reply = json.loads(load_body(pack, response["body"]))
        assert reply["choices"][0]["message"]["content"] == REPLIES[k].content
    _assert_no_key(tmp_path, TASK_INSTRUCTION, REPLIES[0].content)


def test_connection_error_resend_is_a_second_request_in_the_same_attempt(tmp_path, monkeypatch):
    """LiteLLM's AsyncHTTPHandler resends once after RemoteProtocolError or ConnectError, on a
    new client that carries the same hooks (http_handler.py:851-866).

    DISABLE_AIOHTTP_TRANSPORT, which the handler reads each time it builds a client
    (http_handler.py:1249-1253), puts it on httpx's own transport, which reports a connection
    closed without a response as RemoteProtocolError every time; aiohttp reports it as a read
    error or a connect error depending on timing. The hooks sit on the httpx client either way.

    LiteLLM closes the resend client as soon as the response headers arrive (http_handler.py:
    865-866), so the streamed body sometimes fails with ReadError and the harness retries the
    whole turn; the trace then holds that failure and a second attempt.
    """
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    _, _, fields = HOOKED["openrouter"]

    def answer(index, received):
        return DROP if index == 0 else _openai_answer(index - 1, received)

    with HttpStub(answer) as stub:
        events = _run(_config(stub, fields, monkeypatch), stub, tmp_path, steps=STEPS[:1])
    sent = _payloads(events, "ModelRequestSent")
    assert len(stub.requests) == len(sent)
    first, resent = [r for r in sent if r["scope"]["attempt"] == 1]
    assert first["body"] == resent["body"]
    attempts = _payloads(events, "ModelAttemptFinished")
    assert attempts[-1]["outcome"] == "ok"
    if len(attempts) == 2:  # the resent stream was cut by LiteLLM's closed client
        (cut,) = [r for r in _payloads(events, "ModelResponseReceived") if r["error"]]
        assert cut["request_seq"] == resent["capture_seq"] and not cut["complete"]
        assert attempts[0]["outcome"] == "retry" and len(sent) == 3
    _assert_no_key(tmp_path)


@pytest.mark.parametrize("api", sorted([*HOOKED, "litellm"]))
def test_a_retry_after_reaches_the_retry_policy(api, tmp_path, monkeypatch):
    """HTTP 429 with ``Retry-After: 120`` on every LiteLLM path. The exception LiteLLM raises has
    no Retry-After on its response; the HTTP error it wraps does, and the driver's classifier
    reads it from there (providers/litellm.py _retry_after). The shared policy does not wait out
    more than 30 s (entity/model.py TransportRetryPolicy), so the first attempt is final."""
    monkeypatch.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    fields = GENERIC if api == "litellm" else HOOKED[api][2]
    body = json.dumps({"error": {"message": "rate limited", "code": 429}}).encode()
    limited = Answer(429, body, headers=(("Retry-After", "120"),))
    with HttpStub(lambda index, received: limited) as stub:
        episode = build_episode(
            _config(stub, fields, monkeypatch),
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key="replay-model",
        )
        try:
            observation = episode.frames.capture(0)
            budget = episode.budget(observation.tick_number)
            result = decide(episode.agent, 0, observation, budget)
        finally:
            assert episode.finish() is None
    assert len(stub.requests) == 1
    assert result["invalid_reason"] == "provider_rate_limited"
    (attempt,) = _payloads(_events(tmp_path), "ModelAttemptFinished")
    assert (attempt["outcome"], attempt["failure_kind"], attempt["retry_delay_s"]) == (
        "failed",
        "rate_limited",
        None,
    )


@pytest.mark.parametrize("api", sorted([*HOOKED, "litellm"]))
def test_a_fixed_reasoning_effort_sends_no_reasoning_setting(api, tmp_path, monkeypatch):
    """``fixed`` names a model whose reasoning is not configurable (providers/registry.py
    api_reasoning_efforts): no LiteLLM path sends a reasoning parameter for it."""
    answer, fields = (
        (_openai_answer, GENERIC) if api == "litellm" else (HOOKED[api][1], HOOKED[api][2])
    )
    with HttpStub(answer) as stub:
        config = _config(stub, fields, monkeypatch).model_copy(update={"reasoning_effort": "fixed"})
        _run(config, stub, tmp_path, steps=STEPS[:1])
    (wire,) = stub.requests
    sent = json.loads(wire.body)
    assert not {"reasoning_effort", "reasoning", "thinking", "output_config"} & sent.keys()


def _tokenize_driver(origin: str, monkeypatch, run_dir: Path):
    monkeypatch.setenv("REPLAY_TRACE_KEY", KEY)
    config = ModelConfig(
        adapter_type="chat_completions",
        model_name="replay-vlm",
        api_base_url=origin + "/v1",
        api_key_env="REPLAY_TRACE_KEY",
        context_window_tokens=100000,
        context_counter="vllm",
        native_output_limit=4096,
        fps=1.0,
    )
    clock = SystemClock()
    trace = EpisodeTrace(run_dir / "episode_trace.jsonl", run_id="tokenize", clock=clock)
    sink = AgentTraceSink(trace, run_dir / "episode_blobs.pack", clock=clock)
    driver = ChatCompletionsDriver(config)
    driver.bind_trace(sink)
    canonical = build_request(
        path_policy(config),
        system_block("count me", None),
        (),
        None,
        AnswerRequirements("text_json", "standard_vlm"),
        CallParameters(config.model_name, None, None),
    )
    request = GenerationRequest(
        generation_id="turn-0-tokenize",
        transport_output_token_cap=None,
        canonical=canonical,
    )
    return driver, request, trace, sink


@pytest.mark.parametrize("status", [200, 500])
def test_vllm_tokenize_is_recorded(status, tmp_path, monkeypatch):
    body = json.dumps({"count": 42, "max_model_len": 100000}).encode()
    with HttpStub(lambda index, received: Answer(status, body)) as stub:
        driver, request, trace, sink = _tokenize_driver(stub.origin, monkeypatch, tmp_path)
        with trace_scope(phase="decision", decision=0):
            if status == 200:
                assert asyncio.run(driver.request_context_status(request))["pressure"] is False
            else:
                with pytest.raises(Exception, match="500"):
                    asyncio.run(driver.request_context_status(request))
    assert close_agent_trace(sink) is None
    trace.close()
    events = _events(tmp_path)
    (sent,) = _payloads(events, "ModelRequestSent")
    (received,) = _payloads(events, "ModelResponseReceived")
    (wire,) = stub.requests
    assert sent["channel"] == "vllm.tokenize" and sent["purpose"] == "context_count"
    assert urlsplit(sent["url"]).path == wire.path == "/tokenize"
    assert dict(sent["headers"])["authorization"] == REDACTED
    assert load_body(tmp_path / "episode_blobs.pack", sent["body"]) == wire.body
    assert received["status_code"] == status and received["request_seq"] == sent["capture_seq"]
    assert load_body(tmp_path / "episode_blobs.pack", received["body"]) == body
    _assert_no_key(tmp_path)


def test_vllm_tokenize_connect_error_is_recorded(tmp_path, monkeypatch):
    with HttpStub(lambda index, received: Answer(200, b"{}")) as stub:
        origin = stub.origin  # nothing listens here once the stub is closed
    driver, request, trace, sink = _tokenize_driver(origin, monkeypatch, tmp_path)
    with pytest.raises(Exception, match="onnect"):
        asyncio.run(driver.request_context_status(request))
    assert close_agent_trace(sink) is None
    trace.close()
    events = _events(tmp_path)
    (sent,) = _payloads(events, "ModelRequestSent")
    assert sent["channel"] == "vllm.tokenize" and "ConnectError" in sent["error"]
    assert not _payloads(events, "ModelResponseReceived")
    _assert_no_key(tmp_path)


# ---- Decisions -----------------------------------------------------------------------------


def _decisions_answer(script, model, provider):
    """Answer each question with its first criterion unless the script names an error status."""

    def answer(index, received):
        status = script[index] if index < len(script) else 200
        if status == 413:
            return Answer(413, b'{"error": {"message": "request too large"}}')
        if status == 400:
            return Answer(400, b'{"error": {"message": "context_length_exceeded"}}')
        if status != 200:
            return Answer(status, b'{"error": {"message": "upstream unavailable"}}')
        criteria = list(json.loads(received.body)["questions"]["next_action"]["criteria"])
        choice = criteria[0]
        return Answer(
            200,
            json.dumps(
                {
                    "provider": provider,
                    "model": model,
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                    "answers": {
                        "next_action": {
                            "type": "choice",
                            "choice": choice,
                            "probabilities": {c: (1.0 if c == choice else 0.0) for c in criteria},
                            "confidence": 1.0,
                        }
                    },
                }
            ).encode(),
        )

    return answer


CANDIDATES = {
    "a": {"description": "click Submit", "action": {"action": "click", "target": "P3"}},
    "b": {"description": "click Name", "action": {"action": "click", "target": "P1"}},
    "c": {"description": "click I agree", "action": {"action": "click", "target": "P2"}},
}


def _turn(k, prompt, **recorded) -> CanonicalTurn:
    observation = ProviderObservation(
        observation=Observation(b"raw", 100.0 + k, float(k), {"observation_id": f"obs-{k}"}),
        prompt=prompt,
        image_bytes=b"png",
        mime_type="image/png",
        image_detail="auto",
        geometry={},
    )
    return CanonicalTurn(observation=observation, decision=k, **recorded)


def _request(config):
    """One earlier turn and an archived action before the current frame: the JEV digest state is
    then four messages (system, ledger, digest, current observation)."""
    current = "Choose." + FRAME_INTERACTABLES_HEADER + json.dumps({"observation_id": "obs-1"})
    turns = (
        _turn(0, "older observation", output='{"action": "wait"}', execution_feedback="applied"),
        _turn(1, current),
    )
    choice = ChoiceRequirements(
        GUI_INSTRUCTIONS,
        tuple((key, Candidate(c["description"], c["action"])) for key, c in CANDIDATES.items()),
        "replay_candidates",
    )
    return build_request(
        path_policy(config),
        system_block("system", None),
        turns,
        ledger_block('{"decision": 0}', 1, None),
        AnswerRequirements("choice", "replay_choice", choice=choice),
        CallParameters(config.model_name, "fixed", None),
        SelectionMaterial("obs-1", False, None, None),
    )


def _decide(client_type, model, provider, script, run_dir, monkeypatch, *, served_headers=()):
    monkeypatch.setenv("REPLAY_TRACE_KEY", KEY)
    clock = SystemClock()
    trace = EpisodeTrace(run_dir / "episode_trace.jsonl", run_id="decisions", clock=clock)
    sink = AgentTraceSink(trace, run_dir / "episode_blobs.pack", clock=clock)
    with HttpStub(_serving(_decisions_answer(script, model, provider), served_headers)) as stub:
        config = ModelConfig(
            adapter_type="jev_decisions"
            if client_type is JevDecisionsClient
            else "openai_decisions",
            model_name=model,
            api_base_url=stub.origin + "/api/alpha/",
            api_key_env="REPLAY_TRACE_KEY",
            upstream_provider=provider,
            context=row_context(
                "jev_decisions" if client_type is JevDecisionsClient else "openai_decisions",
                model,
            ),
        )
        client = client_type(config)
        client.bind_trace(sink)
        request = _request(config)

        async def run():
            try:
                with trace_scope(phase="decision", decision=0, generation_id="turn-0", attempt=1):
                    return await ChoiceDecision(client, request, trace=sink).run()
            finally:
                await client.close()

        result = asyncio.run(run())
    assert close_agent_trace(sink) is None
    trace.close()
    _assert_no_key(run_dir)
    return result, _events(run_dir), stub


def test_decisions_retry_records_each_transport_attempt(tmp_path, monkeypatch):
    result, events, stub = _decide(
        OpenAIDecisionsClient, "openai/gpt-6-luna-decisions", "openai", [503], tmp_path, monkeypatch
    )
    assert result["action"] == CANDIDATES["a"]["action"]
    sent = _payloads(events, "ModelRequestSent")
    assert len(sent) == len(stub.requests) == 2
    assert [r["channel"] for r in sent] == ["decisions.http"] * 2
    assert [r["scope"]["choice"]["transport_attempt"] for r in sent] == [1, 2]
    assert {r["scope"]["choice"]["question"] for r in sent} == {0}
    for request, wire in zip(sent, stub.requests):
        assert load_body(tmp_path / "episode_blobs.pack", request["body"]) == wire.body
    received = _payloads(events, "ModelResponseReceived")
    assert [r["status_code"] for r in received] == [503, 200]
    attempts = _payloads(events, "ModelAttemptFinished")
    assert [a["outcome"] for a in attempts] == ["retry", "ok"]
    assert attempts[0]["failure_kind"] == "transient" and attempts[0]["retry_delay_s"] > 0
    assert attempts[1]["answer"]["choice"] == "a" and attempts[1]["usage"]["output_tokens"] == 1


@pytest.mark.parametrize(
    "client_type, model, provider, status",
    [
        (OpenAIDecisionsClient, "openai/gpt-6-luna-decisions", "openai", 413),
        (JevDecisionsClient, "typesafe/jev-1.13", "typesafe", 400),
    ],
    ids=["luna", "jev"],
)
def test_decisions_context_limit_splits_the_candidates(
    client_type, model, provider, status, tmp_path, monkeypatch
):
    _, events, stub = _decide(client_type, model, provider, [status], tmp_path, monkeypatch)
    attempts = _payloads(events, "ModelAttemptFinished")
    # All three are rejected; the half of one wins without a question, the other half is asked
    # alone, then the two winners meet.
    assert [a["outcome"] for a in attempts] == ["context_limit", "ok", "ok"]
    assert [a["criteria_count"] for a in attempts] == [3, 2, 2]
    (split,) = _payloads(events, "HistoryChanged")
    assert split["kind"] == "choice_candidate_split" and split["trigger"] == "fallback"
    assert split["policy_item"] is None  # the context policy has no item for it yet
    assert split["policy_version"] == CONTEXT_POLICY_VERSION  # as the runtime's records carry
    assert split["choice"]["candidates"] == ["a", "b", "c"]
    assert len(_payloads(events, "ModelRequestSent")) == len(stub.requests) == 3


def test_jev_estimates_the_history_it_sends(tmp_path, monkeypatch):
    """The runtime archives against JEV's estimate (harnesses/default/context.py _run_turn): the
    UTF-8 bytes of the history each question sends, not of the chat rendering Luna's estimate
    counts."""
    _, _, stub = _decide(
        JevDecisionsClient, "typesafe/jev-1.13", "typesafe", [], tmp_path, monkeypatch
    )
    (sent,) = stub.requests
    history = json.loads(sent.body)["state"]["history"]
    size = len(json.dumps(history, ensure_ascii=False).encode())
    config = ModelConfig(
        adapter_type="jev_decisions",
        model_name="typesafe/jev-1.13",
        api_key_env="REPLAY_TRACE_KEY",
        upstream_provider="typesafe",
        context=row_context("jev_decisions", "typesafe/jev-1.13"),
    )
    request = GenerationRequest("turn-0", None, canonical=_request(config))

    def status(window, reserve):
        client = JevDecisionsClient(
            config.model_copy(
                update={"context_window_tokens": window, "context_reserve_tokens": reserve}
            )
        )
        try:
            return asyncio.run(client.request_context_status(request))
        finally:
            asyncio.run(client.close())

    assert status(size + 100, 100) == {
        "input_tokens": size,
        "native_limit": size + 100,
        "reserve_tokens": 100,
        "soft_limit": size,
        "pressure": False,
        "counter": "estimated_utf8_bytes_plus_image_allowance",
    }
    assert status(size + 99, 100)["pressure"] is True
    assert status(None, 100) == {"pressure": False, "counter": "unconfigured"}


# ---- Headers --------------------------------------------------------------------------------

COOKIE_VALUE = "COOKIE-VALUE-0123456789"
ORGANIZATION = "org-ACCOUNT-0123456789"
PROJECT = "proj_ACCOUNT_0123456789"
ACCOUNT = (COOKIE_VALUE, ORGANIZATION, PROJECT)
# A session cookie and the account a key belongs to, as a provider behind a CDN serves them,
# beside headers whose values the trace keeps.
SERVED_HEADERS = (
    ("Set-Cookie", f"__cf_bm={COOKIE_VALUE}; Path=/"),
    ("openai-organization", ORGANIZATION),
    ("x-request-id", "req_replay_0123"),
    ("x-ratelimit-remaining-requests", "4999"),
)


def _check_account_headers(events, stub, run_dir: Path) -> list[list[str]]:
    """Recorded headers keep their names and only allow-listed values, so no cookie or account id
    reaches the trace. Returns, per request, the headers that carried one on the wire."""
    for request, wire in zip(_payloads(events, "ModelRequestSent"), stub.requests, strict=True):
        assert {name.lower(): value for name, value in request["headers"]} == {
            name.lower(): value if header_kept(name) else REDACTED for name, value in wire.headers
        }
    responses = _payloads(events, "ModelResponseReceived")
    assert len(responses) == len(stub.requests)
    for response in responses:
        recorded = {name.lower(): value for name, value in response["headers"]}
        assert recorded["set-cookie"] == recorded["openai-organization"] == REDACTED
        assert recorded["x-request-id"] == "req_replay_0123"
        assert recorded["x-ratelimit-remaining-requests"] == "4999"
    trace = (run_dir / "episode_trace.jsonl").read_text()
    assert [value for value in ACCOUNT if value in trace] == []
    return [
        sorted(name.lower() for name, value in wire.headers if any(v in value for v in ACCOUNT))
        for wire in stub.requests
    ]


@pytest.mark.parametrize("api", sorted(HOOKED))
def test_cookies_and_account_ids_stay_out_of_the_trace(api, tmp_path, monkeypatch):
    channel, answer, fields = HOOKED[api]
    monkeypatch.setenv("OPENAI_ORG_ID", ORGANIZATION)
    monkeypatch.setenv("OPENAI_PROJECT_ID", PROJECT)
    with HttpStub(_serving(answer, SERVED_HEADERS)) as stub:
        events = _run(_config(stub, fields, monkeypatch), stub, tmp_path)
    carried = _check_account_headers(events, stub, tmp_path)
    if channel == "litellm.openai_sdk":
        # The OpenAI SDK sends the account from the environment; its httpx client replays the
        # served cookie.
        account = ["openai-organization", "openai-project"]
        assert carried == [account, ["cookie", *account]]
    else:
        # LiteLLM's aiohttp transport keeps no cookie for an IP-address host such as the stub's.
        assert carried == [[], []]


def test_decisions_cookie_stays_out_of_the_trace(tmp_path, monkeypatch):
    _, events, stub = _decide(
        OpenAIDecisionsClient,
        "openai/gpt-6-luna-decisions",
        "openai",
        [503],
        tmp_path,
        monkeypatch,
        served_headers=SERVED_HEADERS,
    )
    # The Decisions httpx client replays the cookie the rejected attempt was served.
    assert _check_account_headers(events, stub, tmp_path) == [[], ["cookie"]]


# A generation's end records the responses it left open (capture.py HttpCapture.record_open).


class _Responses:
    """The trace's capture, keeping each response record."""

    def __init__(self):
        self.records = []

    def capture(self, event, payload, *, body=None, classify=None):
        self.records.append((event, payload, body))
        return len(self.records)

    def of(self, request_seq) -> list[tuple[bool, bytes]]:
        return [
            (payload["complete"], b"".join(body))
            for event, payload, body in self.records
            if event is TraceEvent.MODEL_RESPONSE_RECEIVED and payload["request_seq"] == request_seq
        ]


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        pass


class _HeldStream:
    """LiteLLM's stream as the driver reads it, which ends once released. Its aclose does not reach
    the HTTP response, as that of LiteLLM's Anthropic iterator does not."""

    def __init__(self, released):
        self.released = released
        self.completion_stream = SimpleNamespace(chunk_parser=lambda chunk: chunk)

    def __aiter__(self):
        return self._events()

    async def _events(self):
        await self.released.wait()
        return
        yield

    async def aclose(self):
        pass


async def _served(capture, chunks):
    """One response through a capture's hooks; its request's record and its recording stream."""
    request = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions", content=b"{}")
    await capture._on_request(request)
    response = httpx.Response(200, stream=_Chunks(chunks), request=request)
    await capture._on_response(response)
    return capture._sent[request], response.stream


def test_an_abandoned_generation_records_only_the_responses_it_left_open():
    """The abort closes the client (session_lifecycle.py _interrupt_and_close), and the next
    generation's client has its own capture. The abandoned generation's events end later: they
    record the response it left open, and leave the next generation's, still being read, to be
    recorded once its stream closes."""

    async def scenario():
        driver = ChatCompletionsDriver(
            ModelConfig(
                adapter_type="chat_completions",
                model_name="replay-vlm",
                api_base_url="http://127.0.0.1:9/v1",
            )
        )
        trace = _Responses()
        driver.bind_trace(trace)
        released = asyncio.Event()
        stream = _HeldStream(released)

        async def acompletion(**kwargs):
            return stream

        driver._sdk = SimpleNamespace(acompletion=acompletion)
        driver._request_kwargs = lambda request: {}
        abandoned = await driver.start_generation(SimpleNamespace(generation_id="gen-a"))
        left_open, open_stream = await _served(driver._http_capture, [b"a1", b"a2"])
        assert await anext(open_stream.__aiter__()) == b"a1"

        async def read():
            return [event async for event in driver.events(abandoned)]

        reader = asyncio.create_task(read())
        await asyncio.sleep(0)
        await driver.interrupt(abandoned)
        await driver.close()

        driver._client()
        next_seq, next_stream = await _served(driver._http_capture, [b"b1", b"b2"])
        chunks = next_stream.__aiter__()
        assert await anext(chunks) == b"b1"
        released.set()
        await reader
        early = trace.of(next_seq)
        assert [chunk async for chunk in chunks] == [b"b2"]
        await next_stream.aclose()
        await driver.close()
        return trace.of(left_open), early, trace.of(next_seq)

    left_open, early, next_generation = asyncio.run(scenario())
    assert left_open == [(False, b"a1")]
    assert early == []
    assert next_generation == [(True, b"b1b2")]


def test_a_generation_aborted_during_its_request_records_only_its_own_clients_responses():
    """An abort before a generation's request returns closes the client
    (session_lifecycle.py _interrupt_and_close with no generation), and the next generation may
    build its own before the request returns: the generation's end still leaves the next one's
    responses alone."""

    async def scenario():
        driver = ChatCompletionsDriver(
            ModelConfig(
                adapter_type="chat_completions",
                model_name="replay-vlm",
                api_base_url="http://127.0.0.1:9/v1",
            )
        )
        trace = _Responses()
        driver.bind_trace(trace)
        requested, answered, released = asyncio.Event(), asyncio.Event(), asyncio.Event()
        released.set()

        async def acompletion(**kwargs):
            requested.set()
            await answered.wait()
            return _HeldStream(released)

        driver._sdk = SimpleNamespace(acompletion=acompletion)
        driver._request_kwargs = lambda request: {}
        starting = asyncio.create_task(
            driver.start_generation(SimpleNamespace(generation_id="gen-a"))
        )
        await requested.wait()
        await driver.close()

        driver._client()
        next_seq, next_stream = await _served(driver._http_capture, [b"b1", b"b2"])
        chunks = next_stream.__aiter__()
        assert await anext(chunks) == b"b1"
        answered.set()
        abandoned = await starting
        assert [event async for event in driver.events(abandoned)]
        early = trace.of(next_seq)
        assert [chunk async for chunk in chunks] == [b"b2"]
        await next_stream.aclose()
        await driver.close()
        return early, trace.of(next_seq)

    early, next_generation = asyncio.run(scenario())
    assert early == []
    assert next_generation == [(True, b"b1b2")]
