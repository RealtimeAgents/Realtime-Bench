"""TypeSafe JEV's text-only capability profile for OpenRouter Decisions."""

from typing import ClassVar

from realtime_framework.entity.model import Deployment, RealtimeCapabilities
from realtime_framework.providers.context import context_status, estimated_request_tokens
from realtime_framework.providers.decisions.history import render_digest
from realtime_framework.providers.openai_decisions import OpenAIDecisionsClient


class JevDecisionsClient(OpenAIDecisionsClient):
    capabilities = RealtimeCapabilities(
        protocol_family="jev_decisions",
        persistent_session=False,
        action_channel="choice",
        observation_modality="text",
        # TypeSafe caps prompts at 32k tokens, below a few retained SoM turns.
        history_replay="action_digest",
        # The digest's role strings are state data, not protocol roles; its blocks are told
        # apart by their place and prefixes (providers/decisions/history.py render_digest).
        labels_instead_of_roles=True,
        feedback_placements=("digest",),
        server_context="none",
        max_choices=255,
        finite_frame_budget=False,
        wall_clock_only=True,
        accepts_reasoning_effort=False,
        classifies_transport_errors=True,
        request_size_estimation=True,
    )
    deployment = Deployment(family="streaming", provider="openrouter", key_env="OPENROUTER_API_KEY")
    # Requests carry no provider routing block; responses still validate the TypeSafe route.
    pin_route = False
    model_defaults: ClassVar[dict[str, dict]] = {
        "typesafe/jev-1.13": {
            "upstream_provider": "typesafe",
            # TypeSafe caps prompts at 32768 tokens. The estimate counts UTF-8 bytes, so both
            # values are bytes: 2.25 a token, fewer than the 3.1-3.7 the replay requests take in
            # o200k tokens; check against TypeSafe's tokenizer once JEV runs for real. The
            # reserve holds what each question adds outside the estimate, its criteria and
            # instructions (13.4 KB on the replay page's 255 options), and the answer; a
            # question that still does not fit is split by the selection flow.
            "context_window_tokens": 73728,
            "context_reserve_tokens": 16384,
        },
    }

    async def request_context_status(self, request):
        """Estimate the digest state that every question of the decision sends."""
        if self.config.context_window_tokens is None:
            return {"pressure": False, "counter": "unconfigured"}
        count = estimated_request_tokens(
            render_digest(request.canonical), self.config.context_image_tokens
        )
        return context_status(self.config, count, "estimated_utf8_bytes_plus_image_allowance")
