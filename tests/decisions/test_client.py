"""One Decisions question of each Choice API, sent by its client straight to a local stub.

What goes on the wire: the path, every header in order, the body's bytes. What comes back: a
ChoiceReply holding the reply as received, a context limit the selection flow splits on, or an
error with the failure kind the per-question retry gives it (classify_question_error). The replay
baselines compare bodies with key order ignored, and every 200 their stub sends is a valid reply.
"""

from __future__ import annotations

import asyncio
import json
import math
import socket

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
from realtime_framework.providers.capture import HttpCapture
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


def _reply(api, **changes) -> dict:
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


def _answer(api, **changes) -> dict:
    answer = _reply(api)["answers"]["next_action"]
    return _reply(api, answers={"next_action": {**answer, **changes}})


def _without(reply: dict, key: str) -> dict:
    return {name: value for name, value in reply.items() if name != key}


def _serve(status, body, headers=(), content_type="application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    return lambda index, received: Answer(status, data, content_type, headers)


class _Endpoint:
    """The file's stub; each question sets what it answers."""

    server: HttpStub
    answer = None


@pytest.fixture(scope="module")
def endpoint():
    """One stub for the file: stopping one waits out its half-second poll."""
    endpoint = _Endpoint()
    with HttpStub(lambda index, received: endpoint.answer(index, received)) as server:
        endpoint.server = server
        yield endpoint


def _client(api, base_url):
    client_type, model, upstream = APIS[api]
    client = client_type(
        ModelConfig(
            adapter_type="openai_decisions" if api == "luna" else "jev_decisions",
            model_name=model,
            api_key=KEY,
            api_base_url=base_url,
            upstream_provider=upstream,
        )
    )
    client.bind_trace(_Trace())
    return client


def _ask(client, api, question=QUESTION):
    """The client's reply, or the exception it raised."""
    state = LUNA_STATE if api == "luna" else JEV_STATE

    async def run():
        try:
            return await client.decide(state, question)
        except Exception as exc:  # noqa: BLE001 - the failure under test
            return exc
        finally:
            await client.close()

    return asyncio.run(run())


def _decide(endpoint, api, answer, *, question=QUESTION):
    """Ask one question; return the reply or the exception, and the requests it sent."""
    endpoint.answer = answer
    first = len(endpoint.server.requests)
    result = _ask(_client(api, endpoint.server.origin + "/api/alpha"), api, question)
    return result, endpoint.server.requests[first:]


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
def test_one_post_with_these_bytes(endpoint, api, body):
    """Compact JSON, UTF-8 as is, keys in the order built, and Luna's route block last."""
    reply, requests = _decide(endpoint, api, _serve(200, _reply(api)))
    assert reply.answer["choice"] == "a"
    (received,) = requests
    assert received.method == "POST" and received.path == "/api/alpha/decisions"
    assert received.body == body
    port = endpoint.server.origin.rpartition(":")[2]
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


@pytest.mark.parametrize("api", list(APIS))
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
def test_a_question_the_api_cannot_take_is_not_sent(endpoint, question, api):
    error, requests = _decide(endpoint, api, _serve(200, _reply(api)), question=question)
    assert type(error) is ValueError
    assert requests == []


# ---- The reply ---------------------------------------------------------------------------------

VALID = {
    "plain": lambda api: _reply(api),
    "provider_in_other_case": lambda api: _reply(api, provider=APIS[api][2].upper()),
    "dated_model": lambda api: _reply(api, model=APIS[api][1] + "-2026-10-01"),
    "zero_usage_and_more_keys": lambda api: _reply(
        api, usage={"input_tokens": 0, "output_tokens": 0, "cost": 0.5}
    ),
    "more_fields": lambda api: _reply(api, id="dec-1"),
    "integer_numbers": lambda api: _answer(api, probabilities={"b": 0, "a": 1}, confidence=1),
    "probabilities_need_not_sum_to_one": lambda api: _answer(
        api, probabilities={"b": 0.9, "a": 0.9}
    ),
    "more_answer_fields": lambda api: _answer(
        api, probabilities={"b": 0.0, "a": 0.0}, extra="kept"
    ),
    "zero_confidence": lambda api: _answer(api, confidence=0),
}


@pytest.mark.parametrize("api", list(APIS))
@pytest.mark.parametrize("case", list(VALID))
def test_a_valid_reply_comes_back_as_received(endpoint, case, api):
    """As parsed: compared as JSON text, so key order and 1 against 1.0 count."""
    reply = VALID[case](api)
    result, _ = _decide(endpoint, api, _serve(200, reply))
    assert json.dumps(result.response) == json.dumps(reply)
    assert json.dumps(result.answer) == json.dumps(reply["answers"]["next_action"])
    assert json.dumps(result.usage) == json.dumps(reply["usage"])


@pytest.mark.parametrize("api", list(APIS))
def test_a_reply_is_parsed_as_json_loads_parses_it(endpoint, api):
    """A UTF-8 byte order mark is skipped (json.loads detects the encoding of bytes)."""
    reply = _reply(api)
    body = b"\xef\xbb\xbf" + json.dumps(reply).encode()
    result, _ = _decide(endpoint, api, _serve(200, body))
    assert result.response == reply


INVALID = {
    "not_an_object": lambda api: [],
    "other_provider": lambda api: _reply(api, provider="anthropic"),
    "no_provider": lambda api: _without(_reply(api), "provider"),
    "shorter_model_name": lambda api: _reply(api, model=APIS[api][1].rpartition("-")[0]),
    "model_name_running_on": lambda api: _reply(api, model=APIS[api][1] + "2"),
    "no_model": lambda api: _without(_reply(api), "model"),
    "no_usage": lambda api: _without(_reply(api), "usage"),
    "usage_without_output": lambda api: _reply(api, usage={"input_tokens": 1}),
    "usage_as_float": lambda api: _reply(api, usage={"input_tokens": 1.0, "output_tokens": 1}),
    "usage_as_bool": lambda api: _reply(api, usage={"input_tokens": 1, "output_tokens": True}),
    "usage_as_text": lambda api: _reply(api, usage={"input_tokens": "1", "output_tokens": 1}),
    "negative_input_tokens": lambda api: _reply(
        api, usage={"input_tokens": -1, "output_tokens": 1}
    ),
    "negative_output_tokens": lambda api: _reply(
        api, usage={"input_tokens": 1, "output_tokens": -1}
    ),
    "no_answers": lambda api: _without(_reply(api), "answers"),
    "no_answer": lambda api: _reply(api, answers={}),
    "answers_not_a_mapping": lambda api: _reply(api, answers=[]),
    "answer_not_asked": lambda api: _reply(
        api, answers={**_reply(api)["answers"], "other": _reply(api)["answers"]["next_action"]}
    ),
    "refusal": lambda api: _reply(
        api, answers={"next_action": {"type": "refusal", "refusal": "no"}}
    ),
    "answer_of_another_type": lambda api: _answer(api, type="score"),
    "choice_not_offered": lambda api: _answer(api, choice="c"),
    "choice_not_text": lambda api: _answer(api, choice=1),
    "probability_missing": lambda api: _answer(api, probabilities={"a": 1.0}),
    "probability_extra": lambda api: _answer(api, probabilities={"a": 1.0, "b": 0.0, "c": 0.0}),
    "probability_above_one": lambda api: _answer(api, probabilities={"a": 1.5, "b": 0.0}),
    "probability_below_zero": lambda api: _answer(api, probabilities={"a": 1.0, "b": -0.1}),
    "probability_nan": lambda api: _answer(api, probabilities={"a": math.nan, "b": 0.0}),
    "probability_as_bool": lambda api: _answer(api, probabilities={"a": True, "b": 0.0}),
    "probability_as_text": lambda api: _answer(api, probabilities={"a": "1", "b": 0.0}),
    "no_confidence": lambda api: _reply(
        api, answers={"next_action": _without(_reply(api)["answers"]["next_action"], "confidence")}
    ),
    "confidence_above_one": lambda api: _answer(api, confidence=1.5),
    "confidence_infinite": lambda api: _answer(api, confidence=math.inf),
    "confidence_as_bool": lambda api: _answer(api, confidence=True),
    "confidence_null": lambda api: _answer(api, confidence=None),
}


@pytest.mark.parametrize("api", list(APIS))
@pytest.mark.parametrize("case", list(INVALID))
def test_an_invalid_reply_fails_the_question(endpoint, case, api):
    """Rejected before the selection flow reads it; a retry would not change it."""
    error, _ = _decide(endpoint, api, _serve(200, INVALID[case](api)))
    assert isinstance(error, ValueError)
    assert APIS[api][0].classify_question_error(error) == TransportFailure(FATAL)


@pytest.mark.parametrize("api", list(APIS))
def test_a_body_that_is_not_json_fails_the_question(endpoint, api):
    error, _ = _decide(endpoint, api, _serve(200, b"<html>", content_type="text/html"))
    assert isinstance(error, ValueError)
    assert APIS[api][0].classify_question_error(error) == TransportFailure(FATAL)


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
    # Only a 400 or a 413 is a context limit.
    "500_context_length": (
        (500, _error("Too long", 500, metadata={"raw": "context_length_exceeded"}), ()),
        ("DecisionsHTTPError", "Too long", TRANSIENT),
    ),
    "502_not_utf8": ((502, b"\xff<html>", ()), ("DecisionsHTTPError", "Bad Gateway", TRANSIENT)),
    # httpx does not follow redirects; a redirect is an error whatever its body.
    "307": ((307, _reply("luna"), ()), ("DecisionsHTTPError", "Temporary Redirect", FATAL)),
}


@pytest.mark.parametrize("api", list(APIS))
@pytest.mark.parametrize("served, expected", list(ERRORS.values()), ids=list(ERRORS))
def test_an_error_status_is_classified(endpoint, served, expected, api):
    """The records take these messages (ModelAttemptFinished, invalid_detail); the replay
    baselines hold the 503's (openai_decisions-retry) and the 413's (the overflow baselines)."""
    error_type, message, failure = expected
    status = served[0]
    error, _ = _decide(endpoint, api, _serve(*served))
    if error_type is ChoiceContextLimitError:
        assert type(error) is ChoiceContextLimitError
        assert str(error) == message
        return
    assert type(error).__name__ == error_type
    assert str(error) == f"Decisions HTTP {status}: {message}"
    if isinstance(failure, str):
        failure = TransportFailure(failure)
    assert APIS[api][0].classify_question_error(error) == failure


@pytest.mark.parametrize("api", list(APIS))
def test_a_dropped_connection_is_transient(endpoint, api):
    error, _ = _decide(endpoint, api, lambda index, received: DROP)
    assert isinstance(error, httpx.TransportError)
    assert APIS[api][0].classify_question_error(error) == TransportFailure(TRANSIENT)


@pytest.mark.parametrize("api", list(APIS))
def test_a_refused_connection_is_transient(api):
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    error = _ask(_client(api, f"http://127.0.0.1:{port}/api/alpha"), api)
    assert isinstance(error, httpx.ConnectError)
    assert APIS[api][0].classify_question_error(error) == TransportFailure(TRANSIENT)


@pytest.mark.parametrize("api", list(APIS))
@pytest.mark.parametrize("error", [httpx.ReadTimeout("slow"), httpx.PoolTimeout("busy")])
def test_a_timeout_is_transient(error, api):
    assert APIS[api][0].classify_question_error(error) == TransportFailure(TRANSIENT)


@pytest.mark.parametrize("api", list(APIS))
def test_the_client_and_the_one_after_an_abort(api):
    """Neither model YAML names a base URL (configs/models/), so a run posts to OpenRouter's."""
    client = _client(api, None)

    async def clients():
        first = client.client
        await client.close()
        await client.connect(None)
        await client.close()
        return first, client.client

    first, rebuilt = asyncio.run(clients())
    assert rebuilt is not first
    for http in (first, rebuilt):
        assert str(http.base_url) == "https://openrouter.ai/api/alpha/"
        assert http.timeout == httpx.Timeout(30.0)
        assert http.headers["Authorization"] == f"Bearer {KEY}"
        for kind in ("request", "response"):
            assert [type(hook.__self__) for hook in http.event_hooks[kind]] == [HttpCapture]
