"""Context settings and policies for the tests that build a model configuration or a policy without
a catalog model (entity/context_policy.py resolve_context, path_policy)."""

from __future__ import annotations

from types import SimpleNamespace

from realtime_framework.entity.context_policy import (
    ContextSettings,
    PathPolicy,
    path_policy,
    resolve_context,
)

# What the Decision rows took from their presets before the context settings, which their models'
# YAMLs now set (configs/models/gpt-6-luna-decisions.yaml, jev-1.13.yaml).
ROW_VALUES = {
    "openai_decisions": {"observation": "observation_json_screenshot"},
    "jev_decisions": {
        "observation": "observation_json_text",
        "execution_feedback": "digest",
        "decision_history": "action_digest",
    },
}


def row_context(api: str, model: str = "test-model", **layers) -> ContextSettings:
    """The context settings of a model of API ``api`` with the values the API's row stood for:
    the standard ones, or Luna's or JEV's on a Decision row; then the given ``layers``."""
    layers.setdefault("model_layer", ROW_VALUES.get(api, {}))
    return resolve_context(api, model, **layers)


def api_policy(api: str, model: str = "test-model", **layers) -> PathPolicy:
    """The policy of row_context(api, model, **layers) on API ``api``."""
    return path_policy(SimpleNamespace(adapter_type=api, context=row_context(api, model, **layers)))
