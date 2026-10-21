# Adapted from Pydantic AI, https://github.com/pydantic/pydantic-ai at commit
# 57d8610448b9a6b2b0d3f70a85bf134b59c4b7eb: SystemOneModel.decide, its response schema and
# _error_body in pydantic_ai_slim/pydantic_ai/models/system_one.py, and ChoiceAnswer in
# pydantic_ai_slim/pydantic_ai/models/decision.py. Its license:
#
# The MIT License (MIT)
#
# Copyright (c) Pydantic Services Inc. 2024 to present
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""One Decisions request: build it, send it, and read its answers or its error.

OpenRouter's Decisions endpoint speaks the protocol of Pydantic AI's System One model: a state
and named questions go out, the model and an answer per question come back with the usage. So
the request is built and sent, and the reply validated, the way that model does. What differs:
- The endpoint is ``decisions`` below the client's base URL, and httpx sends it, not httpx2.
- The body keeps the key order these clients have always sent, model before state, and each
  question goes out as the selection flow wrote it: how the server renders the JSON for the model
  is not known, so the bytes stay as they were.
- Of the model's settings only ``extra_body`` is taken (Luna's route); the client carries the
  key, headers and timeout. The state may hold images: Luna's screenshots travel inside it.
- Only Choice questions are asked, so only Choice answers are read.
- The reply is parsed as ``json.loads`` parses it and validated in strict mode (a count must be
  a JSON integer, a probability a JSON number); its usage counts have no default.
- The reply also names its upstream, which must be the pinned one, and its model, which must be
  the requested model or that name followed by a hyphen and more.
- The probabilities are not required to sum to one.
- Any status outside 2xx is an error. OpenRouter's context-limit and refusal statuses raise
  their own errors, which the selection flow reads; every other error, and a transport error as
  httpx raises it, is left to the shared retry policy (entity/model.py classify_http_failure).
  An invalid reply raises ValueError.
- The caller gets the reply's JSON as received, not converted answers.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Annotated, Any, Literal, TypedDict

import httpx
from pydantic import Field, TypeAdapter, ValidationError

from realtime_framework.entity.model import ChoiceContextLimitError


class DecisionsHTTPError(RuntimeError):
    def __init__(self, response, detail):
        super().__init__(f"Decisions HTTP {response.status_code}: {detail}")
        self.status_code = response.status_code
        self.response = response


class DecisionsRefusalError(RuntimeError):
    """The upstream model declined to answer the question."""


async def decide(
    client: httpx.AsyncClient,
    model: str,
    state: Any,
    questions: dict[str, dict],
    *,
    upstream: str,
    extra_body: Mapping[str, Any] | None = None,
) -> dict:
    """Send one request to the ``decisions`` endpoint; return its reply, as received, once it
    answers every question."""
    body: dict[str, Any] = {"model": model, "state": state, "questions": questions}
    if extra_body is not None:
        body.update(extra_body)
    response = await client.send(client.build_request("POST", "decisions", json=body))
    if not response.is_success:
        raise _error(response)
    reply = response.json()
    try:
        parsed = _response_adapter.validate_python(reply, strict=True)
    except ValidationError as e:
        raise ValueError(f"Invalid response from the Decisions API: {e}") from e
    if parsed["provider"].lower() != upstream.lower():
        raise ValueError("Decisions response did not come from the configured upstream provider")
    if parsed["model"] != model and not parsed["model"].startswith(model + "-"):
        raise ValueError("Decisions response model differs from the requested model family")
    if parsed["answers"].keys() != questions.keys():
        raise ValueError(
            "Invalid response from the Decisions API: answer names do not match the questions"
        )
    for name, question in questions.items():
        answer = parsed["answers"][name]
        probabilities = answer["probabilities"]
        valid = (
            answer["choice"] in question["criteria"]
            and probabilities.keys() == question["criteria"].keys()
            and isfinite(answer["confidence"])
            and 0 <= answer["confidence"] <= 1
            and all(isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
        )
        if not valid:
            raise ValueError(
                f"Invalid response from the Decisions API: answer {name!r} does not match its "
                f"question: {answer!r}"
            )
    return reply


def _error(response: httpx.Response) -> Exception:
    error_body = _error_body(response)
    error = error_body.get("error", {}) if isinstance(error_body, dict) else None
    detail = response.reason_phrase
    if isinstance(error, dict):
        detail = str(error.get("message", detail))[:500]
    # Context rejection lets the selection flow split the question's candidates.
    if response.status_code == 413 or (
        response.status_code == 400
        and any(
            code in response.text for code in ("max_tokens_exceeded", "context_length_exceeded")
        )
    ):
        return ChoiceContextLimitError(f"Decisions request exceeds its context limit: {detail}")
    # OpenRouter reports an upstream refusal as a gateway error; it is not transient.
    # Observed body: HTTP 502, error.message 'OpenAI refused to answer question "<key>"'.
    if response.status_code == 502 and "refused to answer question" in detail.lower():
        return DecisionsRefusalError(f"Decisions HTTP {response.status_code}: {detail}")
    return DecisionsHTTPError(response, detail)


# TypedDicts rather than Pydantic AI's dataclasses: strict mode takes a dict only for these.
class _Usage(TypedDict):
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]


class _ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class _DecisionsResponse(TypedDict):
    """The body the API answers ``decisions`` with."""

    provider: str
    model: str
    answers: dict[str, _ChoiceAnswer]
    usage: _Usage


_response_adapter = TypeAdapter(_DecisionsResponse)


def _error_body(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text
