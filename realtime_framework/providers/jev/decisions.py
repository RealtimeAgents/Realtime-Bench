"""TypeSafe JEV's text-only capability profile for OpenRouter Decisions."""

from realtime_framework.entity.model import RealtimeCapabilities
from realtime_framework.providers.openai_decisions import OpenAIDecisionsClient


class JevDecisionsClient(OpenAIDecisionsClient):
    capabilities = RealtimeCapabilities(
        protocol_family="jev_decisions",
        persistent_session=False,
        action_channel="choice",
        observation_modality="text",
        # TypeSafe caps prompts at 32k tokens, below a few retained SoM turns.
        history_replay="action_digest",
    )
    max_choices = 255
    # Requests carry no provider routing block; responses still validate the TypeSafe route.
    pin_route = False
    model_defaults = {"typesafe/jev-1.13": {"upstream_provider": "typesafe"}}
