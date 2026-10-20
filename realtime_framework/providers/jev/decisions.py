"""TypeSafe JEV's text-only capability profile for OpenRouter Decisions."""

from typing import ClassVar

from realtime_framework.entity.model import Deployment, RealtimeCapabilities
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
    )
    deployment = Deployment(family="streaming", provider="openrouter", key_env="OPENROUTER_API_KEY")
    # Requests carry no provider routing block; responses still validate the TypeSafe route.
    pin_route = False
    model_defaults: ClassVar[dict[str, dict]] = {
        "typesafe/jev-1.13": {"upstream_provider": "typesafe"},
    }
