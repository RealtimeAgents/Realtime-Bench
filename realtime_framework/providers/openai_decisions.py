"""OpenAI Decisions models through OpenRouter's Decisions transport.

Only the HTTP client (its key, base URL, timeout and capture hooks, built again after an abort),
the question limits, the route pin, context estimation and the retry kind of a failed question
live here; the request is built, sent and read in providers/decisions/protocol.py, which also
tells a context limit and a refusal from other error statuses. The registered class adds them to
DecisionEncoder, which makes it a complete driver. Candidate actions come with the canonical
request; the questions, their state, composition and retries belong to the selection flow
(providers/decisions).

The transport is httpx rather than LiteLLM's Decisions call (litellm.asystemone, 1.106.1): that
call cannot send the provider routing block Luna pins its upstream with, takes no client to carry
the HttpCapture hooks, and reports context limits, refusals and Retry-After in other shapes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

import httpx

from realtime_framework.clock import SystemClock
from realtime_framework.entity.model import (
    FATAL,
    REFUSED,
    TRANSIENT,
    ChoiceReply,
    Compat,
    Deployment,
    RealtimeCapabilities,
    TransportFailure,
    classify_http_failure,
    exception_chain,
    retry_after_seconds,
)
from realtime_framework.providers.capture import HttpCapture
from realtime_framework.providers.context import context_status, estimated_request_tokens
from realtime_framework.providers.decisions import protocol
from realtime_framework.providers.decisions.driver import DecisionEncoder
from realtime_framework.providers.decisions.protocol import (
    DecisionsHTTPError,
    DecisionsRefusalError,
)
from realtime_framework.providers.litellm import chat_messages

QUESTION_KEY = "next_action"
DEFAULT_DECISIONS_BASE_URL = "https://openrouter.ai/api/alpha/"


@dataclass(frozen=True, kw_only=True)
class DecisionsDeployment(Deployment):
    """Where one Decisions API's questions go: address and route."""

    default_base_url: str
    # Questions carry OpenRouter's provider routing block, pinning the model's upstream_provider.
    pin_route: bool


class OpenAIDecisionsClient(DecisionEncoder):
    capabilities = RealtimeCapabilities(
        protocol_family="openai_decisions",
        persistent_session=False,
        action_channel="choice",
        observation_modality="screenshot",
        # Decisions state has no roles (providers/decisions/history.py render_transcript).
        labels_instead_of_roles=True,
        feedback_placements=("separate_message",),
        server_context="none",
        max_choices=255,
        finite_frame_budget=False,
        wall_clock_only=True,
        accepts_reasoning_effort=False,
        classifies_transport_errors=True,
        request_size_estimation=True,
    )
    deployment = DecisionsDeployment(
        family="vlm",
        provider="openrouter",
        key_env="OPENROUTER_API_KEY",
        default_base_url=DEFAULT_DECISIONS_BASE_URL,
        pin_route=True,
        requires_upstream_provider=True,
    )
    compat = Compat()
    # Per-model catalog defaults (providers/registry.py).
    model_defaults: ClassVar[dict[str, dict]] = {
        "openai/gpt-6-luna-decisions": {
            # OpenRouter's published window. A 768 px screenshot measured 337 input tokens; the
            # reserve covers one question's criteria (at most 255 candidates) and instructions.
            "context_window_tokens": 1050000,
            "context_image_tokens": 512,
            "context_reserve_tokens": 32768,
        },
    }

    def __init__(self, config, *, clock=None):
        self.config = config
        self.clock = clock or SystemClock()
        variable = config.api_key_env or self.deployment.key_env
        key = config.api_key or os.environ.get(variable)
        if not key:
            raise RuntimeError(f"Decisions requires {variable}")
        if not config.upstream_provider:
            raise ValueError("Decisions requires a pinned upstream_provider")
        self.api_key = key
        self.client = self._new_client()

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.config.api_base_url or self.deployment.default_base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
            event_hooks=HttpCapture(self, "decisions.http", "choice_question").hooks,
        )

    async def connect(self, request):
        # An abort closes the client (close below); the decision after it connects again.
        if self.client.is_closed:
            self.client = self._new_client()

    def bind_trace(self, trace) -> None:
        self._trace = trace

    async def decide(self, state, question):
        criteria = question.get("criteria")
        if question.get("type") != "choice" or not isinstance(criteria, dict) or not criteria:
            raise ValueError("Decisions requires a nonempty choice question")
        if len(criteria) > self.capabilities.max_choices:
            raise ValueError(
                f"Decisions accepts at most {self.capabilities.max_choices} choices per call"
            )
        extra_body = None
        if self.deployment.pin_route:
            extra_body = {
                "provider": {
                    "only": [self.config.upstream_provider],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }
            }
        # The harness retries a failed question with the shared policy; the observation ages
        # during backoff and no extra frame is delivered.
        body = await protocol.decide(
            self.client,
            self.config.model_name,
            state,
            {QUESTION_KEY: question},
            upstream=self.config.upstream_provider,
            extra_body=extra_body,
        )
        return ChoiceReply(
            answer=body["answers"][QUESTION_KEY],
            usage=body["usage"],
            response=body,
        )

    async def request_context_status(self, request):
        """Estimate the canonical request; the generation reserve covers each question's criteria."""
        if self.config.context_window_tokens is None:
            return {"pressure": False, "counter": "unconfigured"}
        # Counts the request's chat-message rendering, not the transcript sent; the pressure
        # archive points were calibrated on this count.
        count = estimated_request_tokens(
            chat_messages(request.canonical), self.config.context_image_tokens
        )
        return context_status(self.config, count, "estimated_utf8_bytes_plus_image_allowance")

    @staticmethod
    def classify_question_error(exc: BaseException) -> TransportFailure:
        """Classify one failed question for the selection flow's shared retry policy."""
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

    async def close(self):
        await self.client.aclose()
