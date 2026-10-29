"""Pin the context policy to the code paths that applied each value before it existed, and each
later version's change to what it changed.

Every check reads class-level capabilities or static tables, so no key or connection is needed.
Line numbers cite those code paths as they were on 7912231. A row stands for the models of its
API without a YAML of their own (tests/context.py api_policy): the standard values, and Luna's or
JEV's on the Decision rows, which their YAMLs set.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.context_policy import (
    CONTEXT_POLICY_VERSION,
    FIXED_TEXTS,
    PROMPT_CHANNELS,
    ROW_PROMPTS,
    PathPolicy,
    context_defaults,
    path_policy,
)
from realtime_framework.entity.model import implements
from realtime_framework.harnesses.default.harness import INTERACTION_PROFILES
from realtime_framework.providers.registry import APIS, adapter_class

from ..context import api_policy

PATHS = {api: api_policy(api) for api in APIS}


def capabilities(api: str):
    return adapter_class(api).capabilities


def test_one_row_per_registered_api():
    assert set(ROW_PROMPTS) == set(APIS)
    assert all(path.version == CONTEXT_POLICY_VERSION for path in PATHS.values())


def test_policy_fields():
    """The items carry the names of their configuration keys; the two retentions are each a
    count kept and a block."""
    assert [field.name for field in fields(PathPolicy)] == [
        "version",
        "api",
        "screenshots_kept",
        "screenshot_fold_block",
        "turns_kept",
        "turn_archive_block",
        "execution_feedback",
        "ledger_role",
        "texts",
        "reasoning_replay",
        "archive_over_window",
        "decision_history",
        "tool_call_history",
        "prompts",
        "observation",
    ]
    assert list(PATHS["chat_completions"].to_record()) == [
        field.name for field in fields(PathPolicy)
    ]


def test_retention_and_texts_are_shared():
    # The defaults file's (realtime_framework/context_defaults.yaml); hermetic.py clears the
    # STREAM_* overrides.
    standard = context_defaults()["standard"]
    for path in PATHS.values():
        assert (path.screenshots_kept, path.screenshot_fold_block) == (
            standard["screenshots_kept"],
            standard["screenshot_fold_block"],
        )
        assert (path.turns_kept, path.turn_archive_block) == (
            standard["turns_kept"],
            standard["turn_archive_block"],
        )
        assert path.texts == FIXED_TEXTS


def test_retention_follows_the_run(monkeypatch):
    """The run's settings, which the STREAM_* variables gave before."""
    path = api_policy(
        "chat_completions",
        run={"screenshots_kept": 8, "screenshot_fold_block": 3, "turns_kept": 12,
             "turn_archive_block": 4},
    )  # fmt: skip
    assert (
        path.screenshots_kept,
        path.screenshot_fold_block,
        path.turns_kept,
        path.turn_archive_block,
    ) == (
        8,
        3,
        12,
        4,
    )


@pytest.mark.parametrize("api", sorted(APIS))
def test_feedback_placement(api):
    """Each action's feedback is its own block on every path, the live sessions' included; a
    digest carries its own."""
    caps = capabilities(api)
    expected = "digest" if caps.history_replay == "action_digest" else "separate_message"
    assert PATHS[api].execution_feedback == expected


def test_model_values():
    """A model's own values are in its YAML's context section, those its row's preset (Luna, JEV),
    its YAML's context_preset (Kimi K3) or its provider_options (Gemini Live) gave before."""
    models = {key: catalog.get(key).context for key in catalog.CATALOG}
    assert {key: values for key, values in models.items() if values} == {
        "kimi-k3": {"reasoning_replay": True},
        "gemini-live": {"tool_call_history": "tool_transcript"},
        "gemini-3.8-live": {"tool_call_history": "tool_transcript"},
        "gemini-3.8-live-extended-thinking": {"tool_call_history": "tool_transcript"},
        "gpt-6-luna-decisions": {"observation": "observation_json_screenshot"},
        "jev-1.13": {
            "observation": "observation_json_text",
            "execution_feedback": "digest",
            "decision_history": "action_digest",
        },
    }


def test_replay_and_pressure_sets():
    paths = PATHS
    # Version 2: reasoning goes back only where the provider requires it, Kimi K3, whose YAML
    # sets it (test_model_values); no row and no API requires it. Every reply is rebuilt from the
    # model output.
    assert not any(path.reasoning_replay for path in paths.values())
    assert not any(capabilities(api).reasoning_replay == "required" for api in APIS)
    # Version 5: one pressure-archive rule, which the runtime applies where the API estimates its
    # requests (harnesses/default/context.py _run_turn). That is where version 4 had it on, plus
    # JEV, which estimates its digest instead of dropping history for one rejected option.
    assert all(path.archive_over_window for path in paths.values())
    assert {api for api in APIS if implements(adapter_class(api), "request_context_status")} == {
        "chat_completions",
        "openai_decisions",
        "jev_decisions",
    }


@pytest.mark.parametrize("api", sorted(APIS))
def test_decision_history(api):
    caps = capabilities(api)
    history = PATHS[api].decision_history
    if caps.action_channel != "choice":
        assert history is None
        return
    assert history.form == ("turns" if caps.history_replay == "canonical" else "action_digest")


def test_ledger_roles():
    # Version 3: an input everywhere. Version 1 had OpenAI Realtime's system item
    # (openai_realtime_session.py:23, 205-221) and Luna's role-less state
    # (openai_decisions.py:132-163), which still places the ledger by position and header.
    assert {path.ledger_role for path in PATHS.values()} == {"input"}


def test_gemini_tool_history_format():
    """Gemini Live's standard is native_tools; each maintained model's YAML sets tool_transcript,
    which its provider_options set before. No other row replays tool calls."""
    assert PATHS["gemini_live"].tool_call_history == "native_tools"
    keys = [
        key
        for key in catalog.keys(provider="gemini")
        if catalog.get(key).adapter_type == "gemini_live"
    ]
    assert keys
    for key in keys:
        assert path_policy(catalog.to_model_config(key, fps=1.0)).tool_call_history == (
            "tool_transcript"
        )
    assert all(
        path.tool_call_history is None for api, path in PATHS.items() if api != "gemini_live"
    )


# The profile each agent picked before item 11 existed: by action channel, and for choice models
# by observation modality and scenario. JEV rejects games (providers/preflight.py).
CHANNEL_PROFILES = {"text_json": "standard_vlm", "function_call": "function_call"}


@pytest.mark.parametrize("api", sorted(APIS))
def test_prompts_and_observation_form(api):
    caps = capabilities(api)
    path = PATHS[api]
    if caps.action_channel == "choice":
        if caps.observation_modality == "screenshot":
            expected = {
                "realtime": "multimodal_decisions",
                "browser_game": "multimodal_game_decisions",
            }
        else:
            expected = {"realtime": "jev_decisions", "browser_game": None}
    else:
        profile = CHANNEL_PROFILES[caps.action_channel]
        expected = {"realtime": profile, "browser_game": profile}
    assert dict(path.prompts) == expected
    assert {name for name in path.prompts.values() if name is not None} <= set(INTERACTION_PROFILES)
    # The rule by which the runner switched choice and text models to the observation JSON
    # before it read item 12 (evaluation/runner.py).
    text_observation = caps.observation_modality == "text" or caps.action_channel == "choice"
    assert path.observation == (
        "som_screenshot"
        if not text_observation
        else "observation_json_screenshot"
        if caps.observation_modality == "screenshot"
        else "observation_json_text"
    )


def test_prompt_channels():
    """Every profile has the action channel it asks for, and each row's profiles ask for its API's
    (providers/preflight.py check_context_policy)."""
    assert set(PROMPT_CHANNELS) == set(INTERACTION_PROFILES)
    for api, path in PATHS.items():
        channels = {PROMPT_CHANNELS[name] for name in path.prompts.values() if name is not None}
        assert channels == {capabilities(api).action_channel}, api


def test_policy_record_is_json():
    import json

    record = PATHS["jev_decisions"].to_record()
    assert json.loads(json.dumps(record)) == record
    assert record["decision_history"] == {"form": "action_digest"}
