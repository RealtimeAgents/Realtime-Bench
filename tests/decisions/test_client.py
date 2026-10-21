"""One Decisions question of each Choice API, sent by its client straight to a local stub.

What goes on the wire: the path, every header in order, the body's bytes. What comes back: a
ChoiceReply holding the reply as received, a context limit the selection flow splits on, or an
error with the failure kind the per-question retry gives it (classify_question_error). The replay
baselines compare bodies with key order ignored, and their stub only sends valid replies.
"""

from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest

from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.model import (
    FATAL,
    QUOTA_EXHAUSTED,
    RATE_LIMITED,
    REFUSED,
    TRANSIENT,
    ChoiceContextLimitError,
    TransportFailure,
)
from realtime_framework.providers.jev.decisions import JevDecisionsClient
from realtime_framework.providers.openai_decisions import OpenAIDecisionsClient

from ..agent_trace.stubs import DROP, Answer, HttpStub

KEY = "decisions-test-key"
# api -> (client, model id, upstream provider): the catalog's (configs/models/,
# providers/jev/decisions.py model_defaults).
APIS = {
    "luna": (OpenAIDecisionsClient, "openai/gpt-6-luna-decisions", "openai"),
    "jev": (JevDecisionsClient, "typesafe/jev-1.13", "typesafe"),
}
QUESTION = {"type": "choice", "instructions": "Pick one.", "criteria": {"b": "Bé", "a": "A"}}
LUNA_STATE = [
    "System é\nObservation: x",
    {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo=", "detail": "auto"},
    },
    'Chosen action: {"action": "wait"}',
]
JEV_STATE = {
    "history": [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Current observation: {}"},
    ],
    "text_composition": {"draft": "é", "step": 1},
}


class _Trace:
    """What the HttpCapture hooks call (providers/capture.py)."""

    def __init__(self):
        self.events = []

    def capture(self, event, payload, body=None):
        self.events.append(event)
        return len(self.events)


def _reply(api="luna", **changes) -> dict:
    _, model, upstream = APIS[api]
    reply = {
        "provider": upstream,
        "model": model,
        "usage": {"input_tokens": 1000, "output_tokens": 1},
        "answers": {
            "next_action": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"b": 0.25, "a": 0.75},
                "confidence": 0.75,
            }
        },
    }
    reply.update(changes)
    return reply


def _answer(**changes) -> dict:
    return _reply(answers={"next_action": {**_reply()["answers"]["next_action"], **changes}})


def _serve(status, body, headers=(), content_type="application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    return lambda index, received: Answer(status, data, content_type, headers)


def _decide(api, answer, *, question=QUESTION):
    """Ask one question; return the reply or the exception, and the stub."""
    client_type, model, upstream = APIS[api]
    state = LUNA_STATE if api == "luna" else JEV_STATE
    with HttpStub(answer) as stub:
        client = client_type(
            ModelConfig(
                adapter_type="openai_decisions" if api == "luna" else "jev_decisions",
                model_name=model,
                api_key=KEY,
                api_base_url=stub.origin + "/api/alpha",
                upstream_provider=upstream,
            )
        )
        client.bind_trace(_Trace())

        async def run():
            try:
                return await client.decide(state, question)
            except Exception as exc:  # noqa: BLE001 - the failure under test
                return exc
            finally:
                await client.close()

        return asyncio.run(run()), stub


# ---- The request -------------------------------------------------------------------------------

LUNA_BODY = (
    '{"model":"openai/gpt-6-luna-decisions",'
    '"state":["System é\\nObservation: x",'
    '{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo=","detail":"auto"}},'
    '"Chosen action: {\\"action\\": \\"wait\\"}"],'
    '"questions":{"next_action":{"type":"choice","instructions":"Pick one.",'
    '"criteria":{"b":"Bé","a":"A"}}},'
    '"provider":{"only":["openai"],"allow_fallbacks":false,"require_parameters":true}}'
).encode()
JEV_BODY = (
    '{"model":"typesafe/jev-1.13",'
    '"state":{"history":[{"role":"system","content":"Sys"},'
    '{"role":"user","content":"Current observation: {}"}],'
    '"text_composition":{"draft":"é","step":1}},'
    '"questions":{"next_action":{"type":"choice","instructions":"Pick one.",'
    '"criteria":{"b":"Bé","a":"A"}}}}'
).encode()


@pytest.mark.parametrize("api, body", [("luna", LUNA_BODY), ("jev", JEV_BODY)], ids=list(APIS))
def test_one_post_with_these_bytes(api, body):
    """Compact JSON, UTF-8 as is, keys in the order built, and Luna's route block last."""
    reply, stub = _decide(api, _serve(200, _reply(api)))
    assert reply.answer["choice"] == "a"
    (received,) = stub.requests
    assert received.method == "POST" and received.path == "/api/alpha/decisions"
    assert received.body == body
    port = stub.origin.rpartition(":")[2]
    assert received.headers == [
        ("Host", f"127.0.0.1:{port}"),
        ("Accept", "*/*"),
        ("Accept-Encoding", "gzip, deflate"),
        ("Connection", "keep-alive"),
        ("User-Agent", f"python-httpx/{httpx.__version__}"),
        ("Authorization", f"Bearer {KEY}"),
        ("Content-Length", str(len(body))),
        ("Content-Type", "application/json"),
    ]


@pytest.mark.parametrize(
    "question",
    [
        {**QUESTION, "type": "score"},
        {**QUESTION, "criteria": {}},
        {**QUESTION, "criteria": ["a", "b"]},
        {**QUESTION, "criteria": {str(i): "" for i in range(256)}},
    ],
    ids=["not_choice", "no_options", "options_not_a_mapping", "256_options"],
)
def test_a_question_the_api_cannot_take_is_not_sent(question):
    error, stub = _decide("luna", _serve(200, _reply()), question=question)
    assert type(error) is ValueError
    assert stub.requests == []


# ---- The reply ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        _reply(),
        _reply(provider="OpenAI"),
        _reply(model="openai/gpt-6-luna-decisions-2026-10-01"),
        _reply(usage={"input_tokens": 0, "output_tokens": 0, "cost": 0.5}),
        _reply(id="dec-1"),
        _answer(probabilities={"b": 0, "a": 1}, confidence=1),
        _answer(probabilities={"b": 0.9, "a": 0.9}),
        _answer(probabilities={"b": 0.0, "a": 0.0}, extra="kept"),
    ],
    ids=[
        "plain",
        "provider_in_other_case",
        "dated_model",
        "zero_usage_and_more_keys",
        "more_fields",
        "integer_numbers",
        "probabilities_need_not_sum_to_one",
        "more_answer_fields",
    ],
)
def test_a_valid_reply_comes_back_as_received(reply):
    result, _ = _decide("luna", _serve(200, reply))
    assert result.response == reply
    assert result.answer == reply["answers"]["next_action"]
    assert result.usage == reply["usage"]


def test_a_reply_is_parsed_as_json_loads_parses_it():
    """A UTF-8 byte order mark is skipped (json.loads detects the encoding of bytes)."""
    reply = _reply()
    result, _ = _decide("luna", _serve(200, b"\xef\xbb\xbf" + json.dumps(reply).encode()))
    assert result.response == reply


def test_jev_reply_names_its_upstream():
    result, _ = _decide("jev", _serve(200, _reply("jev")))
    assert result.answer["choice"] == "a"
    error, _ = _decide("jev", _serve(200, _reply("jev", provider="openai")))
    assert type(error) is ValueError


INVALID = {
    "not_an_object": [],
    "other_provider": _reply(provider="anthropic"),
    "no_provider": {k: v for k, v in _reply().items() if k != "provider"},
    "other_model": _reply(model="openai/gpt-6-luna"),
    "no_model": {k: v for k, v in _reply().items() if k != "model"},
    "no_usage": {k: v for k, v in _reply().items() if k != "usage"},
    "usage_without_output": _reply(usage={"input_tokens": 1}),
    "usage_as_float": _reply(usage={"input_tokens": 1.0, "output_tokens": 1}),
    "usage_as_bool": _reply(usage={"input_tokens": 1, "output_tokens": True}),
    "usage_as_text": _reply(usage={"input_tokens": "1", "output_tokens": 1}),
    "negative_usage": _reply(usage={"input_tokens": -1, "output_tokens": 1}),
    "no_answers": {k: v for k, v in _reply().items() if k != "answers"},
    "no_answer": _reply(answers={}),
    "answers_not_a_mapping": _reply(answers=[]),
    "refusal": _reply(answers={"next_action": {"type": "refusal", "refusal": "no"}}),
    "choice_not_offered": _answer(choice="c"),
    "choice_not_text": _answer(choice=1),
    "probability_missing": _answer(probabilities={"a": 1.0}),
    "probability_extra": _answer(probabilities={"a": 1.0, "b": 0.0, "c": 0.0}),
    "probability_above_one": _answer(probabilities={"a": 1.5, "b": 0.0}),
    "probability_below_zero": _answer(probabilities={"a": 1.0, "b": -0.1}),
    "probability_nan": _answer(probabilities={"a": math.nan, "b": 0.0}),
    "probability_as_bool": _answer(probabilities={"a": True, "b": 0.0}),
    "probability_as_text": _answer(probabilities={"a": "1", "b": 0.0}),
    "no_confidence": {
        **_reply(),
        "answers": {
            "next_action": {
                k: v for k, v in _reply()["answers"]["next_action"].items() if k != "confidence"
            }
        },
    },
    "confidence_infinite": _answer(confidence=math.inf),
    "confidence_as_bool": _answer(confidence=True),
    "confidence_null": _answer(confidence=None),
}


@pytest.mark.parametrize("reply", list(INVALID.values()), ids=list(INVALID))
def test_an_invalid_reply_fails_the_question(reply):
    """Rejected before the selection flow reads it; a retry would not change it."""
    error, _ = _decide("luna", _serve(200, reply))
    assert isinstance(error, ValueError)
    assert OpenAIDecisionsClient.classify_question_error(error) == TransportFailure(FATAL)


def test_a_body_that_is_not_json_fails_the_question():
    error, _ = _decide("luna", _serve(200, b"<html>", content_type="text/html"))
    assert isinstance(error, ValueError)
    assert OpenAIDecisionsClient.classify_question_error(error) == TransportFailure(FATAL)


# ---- Errors ------------------------------------------------------------------------------------


def _error(message, code, **fields):
    """OpenRouter's error body."""
    return {"error": {"code": code, "message": message, **fields}}


LIMIT = "Decisions request exceeds its context limit: "
INTERNAL = "Internal Server Error"
LONG = "x" * 600
REFUSAL = 'OpenAI Refused to answer question "next_action"'
ERRORS = {
    # name: (status, body, response headers), (error type, message, failure kind)
    "413": (
        (413, _error("Too large", 413), ()),
        (ChoiceContextLimitError, LIMIT + "Too large", None),
    ),
    "400_context_length": (
        (400, _error("Too long", 400, metadata={"raw": "context_length_exceeded"}), ()),
        (ChoiceContextLimitError, LIMIT + "Too long", None),
    ),
    "400_max_tokens": (
        (400, _error("Over", "max_tokens_exceeded"), ()),
        (ChoiceContextLimitError, LIMIT + "Over", None),
    ),
    "400": ((400, _error("Bad", 400), ()), ("DecisionsHTTPError", "Bad", FATAL)),
    "502_refusal": ((502, _error(REFUSAL, 502), ()), ("DecisionsRefusalError", REFUSAL, REFUSED)),
    "502": ((502, _error("Upstream", 502), ()), ("DecisionsHTTPError", "Upstream", TRANSIENT)),
    "503_retry_after": (
        (503, _error("Busy", 503), (("Retry-After", "2"),)),
        ("DecisionsHTTPError", "Busy", TransportFailure(TRANSIENT, 2.0)),
    ),
    "408": ((408, _error("Late", 408), ()), ("DecisionsHTTPError", "Late", TRANSIENT)),
    "429": (
        (429, _error("Slow down", 429), (("Retry-After", "5"),)),
        ("DecisionsHTTPError", "Slow down", TransportFailure(RATE_LIMITED, 5.0)),
    ),
    "429_credits": (
        (429, _error("Insufficient credits", 429), ()),
        ("DecisionsHTTPError", "Insufficient credits", QUOTA_EXHAUSTED),
    ),
    "402": ((402, _error("Pay", 402), ()), ("DecisionsHTTPError", "Pay", QUOTA_EXHAUSTED)),
    "500_text": ((500, b"oops", ()), ("DecisionsHTTPError", INTERNAL, TRANSIENT)),
    "500_error_text": ((500, {"error": "boom"}, ()), ("DecisionsHTTPError", INTERNAL, TRANSIENT)),
    "500_no_message": ((500, {"error": {}}, ()), ("DecisionsHTTPError", INTERNAL, TRANSIENT)),
    "500_list": ((500, [1, 2], ()), ("DecisionsHTTPError", INTERNAL, TRANSIENT)),
    "500_long": ((500, _error(LONG, 500), ()), ("DecisionsHTTPError", LONG[:500], TRANSIENT)),
    # httpx does not follow redirects; a redirect is an error even with a reply's body.
    "307": ((307, _reply(), ()), ("DecisionsHTTPError", "Temporary Redirect", FATAL)),
}


@pytest.mark.parametrize("served, expected", list(ERRORS.values()), ids=list(ERRORS))
def test_an_error_status_is_classified(served, expected):
    """The replay baselines hold these messages (jev_transport_retries detail, invalid_detail)."""
    error_type, message, failure = expected
    status = served[0]
    error, _ = _decide("luna", _serve(*served))
    if error_type is ChoiceContextLimitError:
        assert type(error) is ChoiceContextLimitError
        assert str(error) == message
        return
    assert type(error).__name__ == error_type
    assert str(error) == f"Decisions HTTP {status}: {message}"
    if isinstance(failure, str):
        failure = TransportFailure(failure)
    assert OpenAIDecisionsClient.classify_question_error(error) == failure


def test_a_dropped_connection_is_transient():
    error, _ = _decide("luna", lambda index, received: DROP)
    assert isinstance(error, httpx.TransportError)
    assert OpenAIDecisionsClient.classify_question_error(error) == TransportFailure(TRANSIENT)
