"""OpenAI Decisions models through OpenRouter's Decisions transport.

Only wire encoding, route validation, context estimation, failure classification and native
usage live here. Candidate actions, composition, history and retry policies belong to the harness.
"""

from __future__ import annotations

import json
import math
import os

import httpx

from realtime_framework.entity.context_policy import LUNA_ACTION_LABEL, LUNA_OBSERVATION_LABEL
from realtime_framework.entity.model import (
    FATAL,
    FOLDED_VISUAL_PLACEHOLDER,
    REFUSED,
    TRANSIENT,
    ChoiceContextLimitError,
    ChoiceReply,
    RealtimeCapabilities,
    TransportFailure,
    classify_http_failure,
    exception_chain,
    retry_after_seconds,
)
from realtime_framework.providers.capture import HttpCapture
from realtime_framework.providers.context import (
    context_status,
    estimated_request_tokens,
    validate_context_reserve,
)
from realtime_framework.providers.legacy import chat_messages

QUESTION_KEY = "next_action"


class DecisionsHTTPError(RuntimeError):
    def __init__(self, response, detail):
        super().__init__(f"Decisions HTTP {response.status_code}: {detail}")
        self.status_code = response.status_code
        self.response = response


class DecisionsRefusalError(RuntimeError):
    """The upstream model declined to answer the question."""


class OpenAIDecisionsClient:
    capabilities = RealtimeCapabilities(
        protocol_family="openai_decisions",
        persistent_session=False,
        action_channel="choice",
        observation_modality="screenshot",
    )
    # Choices per question; the harness groups larger candidate sets.
    max_choices = 255
    pin_route = True

    def __init__(self, config):
        self.config = config
        key = config.api_key or os.environ.get(config.api_key_env or "OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(f"Decisions requires {config.api_key_env or 'OPENROUTER_API_KEY'}")
        if not config.upstream_provider:
            raise ValueError("Decisions requires a pinned upstream_provider")
        self.client = httpx.AsyncClient(
            base_url=config.api_base_url or "https://openrouter.ai/api/alpha/",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30.0,
            event_hooks=HttpCapture(self, "decisions.http", "choice_question").hooks,
        )

    def bind_trace(self, trace) -> None:
        self._trace = trace

    async def decide(self, state, question):
        criteria = question.get("criteria")
        if question.get("type") != "choice" or not isinstance(criteria, dict) or not criteria:
            raise ValueError("Decisions requires a nonempty choice question")
        if len(criteria) > self.max_choices:
            raise ValueError(f"Decisions accepts at most {self.max_choices} choices per call")
        payload = {
            "model": self.config.model_name,
            "state": self._wire_state(state),
            "questions": {QUESTION_KEY: question},
        }
        if self.pin_route:
            payload["provider"] = {
                "only": [self.config.upstream_provider],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        body = await self._post(payload)
        answers = body.get("answers")
        answer = answers.get(QUESTION_KEY) if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("Decisions response is missing a choice answer (or was refused)")
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in criteria:
            raise ValueError("Decisions returned a choice outside the requested criteria")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
            raise ValueError("Decisions returned an incomplete probability distribution")
        values = [*probabilities.values(), answer.get("confidence")]
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in values
        ):
            raise ValueError("Decisions returned invalid probabilities or confidence")
        return ChoiceReply(
            answer=answer,
            usage=body["usage"],
            response=body,
        )

    async def request_context_status(self, request):
        """Estimate the canonical request; the generation reserve covers each question's criteria."""
        if self.config.context_window_tokens is None:
            return {"pressure": False, "counter": "unconfigured"}
        validate_context_reserve(self.config)
        count = estimated_request_tokens(
            chat_messages(request.canonical), self.config.context_image_tokens
        )
        return context_status(self.config, count, "estimated_utf8_bytes_plus_image_allowance")

    def _wire_state(self, state):
        if self.capabilities.history_replay == "canonical":
            return self._wire_transcript(state)
        images = state.get("images", [])
        if not images:
            return state
        if self.capabilities.observation_modality != "screenshot":
            raise ValueError("This Decisions model does not accept images")
        # Image parts must be at the top level of OpenRouter's state array.
        context = {key: value for key, value in state.items() if key != "images"}
        return [json.dumps(context, ensure_ascii=False), *images]

    @staticmethod
    def _wire_transcript(state):
        """Render canonical messages in their order, with each image at its causal position.

        Decisions state has no roles, and image parts must be top-level entries of the state array.
        The canonical order keeps older turns a stable prefix for prompt caching; only the current
        question's state follows the newest observation. Observations and chosen actions carry
        plain labels: role-tagged turns make OpenAI refuse many-option questions more often.
        """
        entries, text = [], []
        for message in state["history"]:
            content = message.get("content")
            if message["role"] == "assistant":
                text.append(LUNA_ACTION_LABEL + content)
            elif isinstance(content, list) or content == FOLDED_VISUAL_PLACEHOLDER:
                parts = content if isinstance(content, list) else [{"text": content}]
                text.append(LUNA_OBSERVATION_LABEL + "".join(
                    part.get("text") or "" for part in parts if part.get("type") != "image_url"
                ))
                images = [part for part in parts if part.get("type") == "image_url"]
                if images:
                    entries.append("\n".join(text))
                    entries.extend(images)
                    text = []
            else:
                text.append(content)
        extra = {key: value for key, value in state.items() if key != "history"}
        if extra:
            text.append(json.dumps(extra, ensure_ascii=False))
        if text:
            entries.append("\n".join(text))
        return entries

    @staticmethod
    def classify_transport_error(exc: BaseException) -> TransportFailure:
        """Classify one failed question for the harness's shared retry policy."""
        for current in exception_chain(exc):
            if isinstance(current, DecisionsRefusalError):
                return TransportFailure(REFUSED)
            if isinstance(current, DecisionsHTTPError):
                return classify_http_failure(
                    current.status_code,
                    str(current),
                    retry_after_seconds(current.response.headers),
                )
            if isinstance(current, httpx.TransportError):
                return TransportFailure(TRANSIENT)
        return TransportFailure(FATAL)

    async def _post(self, payload):
        # The harness retries a failed question with the shared policy; the observation ages
        # during backoff and no extra frame is delivered.
        response = await self.client.post("decisions", json=payload)
        if not response.is_success:
            detail = response.reason_phrase
            try:
                detail = str(response.json().get("error", {}).get("message", detail))[:500]
            except (ValueError, AttributeError):
                pass
            # Context rejection lets the harness split candidates or reduce retained history.
            if response.status_code == 413 or (
                response.status_code == 400
                and any(code in response.text for code in (
                    "max_tokens_exceeded", "context_length_exceeded"
                ))
            ):
                raise ChoiceContextLimitError(f"Decisions request exceeds its context limit: {detail}")
            # OpenRouter reports an upstream refusal as a gateway error; it is not transient.
            # Observed body: HTTP 502, error.message 'OpenAI refused to answer question "<key>"'.
            if response.status_code == 502 and "refused to answer question" in detail.lower():
                raise DecisionsRefusalError(f"Decisions HTTP {response.status_code}: {detail}")
            raise DecisionsHTTPError(response, detail)
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Decisions response must be an object")  # noqa: TRY004
        if str(body.get("provider", "")).lower() != self.config.upstream_provider.lower():
            raise ValueError("Decisions response did not come from the configured upstream provider")
        model = str(body.get("model", ""))
        if model != self.config.model_name and not model.startswith(self.config.model_name + "-"):
            raise ValueError("Decisions response model differs from the requested model family")
        usage = body.get("usage")
        if not isinstance(usage, dict) or any(
            type(usage.get(key)) is not int or usage[key] < 0
            for key in ("input_tokens", "output_tokens")
        ):
            raise ValueError("Decisions response is missing valid native token usage")
        return body

    async def close(self):
        await self.client.aclose()
