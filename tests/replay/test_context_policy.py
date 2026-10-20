"""Pin the context policy to the code paths that applied each value before it existed, and each
later version's change to what it changed.

Every check reads class-level capabilities or static tables, so no key or connection is needed.
Line numbers cite those code paths as they were on 7912231.
"""

from __future__ import annotations

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.config import (
    DEFAULT_ACTIVE_VISUAL_BUDGET,
    DEFAULT_DETAILED_HISTORY_DROP_BLOCK_SIZE,
    DEFAULT_DETAILED_HISTORY_TURNS,
    DEFAULT_VISUAL_FOLD_BLOCK_SIZE,
)
from realtime_framework.entity.context_policy import (
    FIXED_TEXTS,
    PRESETS,
    current_context_policy,
)
from realtime_framework.harnesses.default.harness import INTERACTION_PROFILES
from realtime_framework.providers.registry import APIS, adapter_class

POLICY = current_context_policy()


def capabilities(api: str):
    return adapter_class(api).capabilities


def test_one_path_per_registered_api():
    assert set(POLICY.paths) == set(APIS)
    assert all(path.version == POLICY.version for path in POLICY.paths.values())


def test_retention_and_texts_are_shared():
    # entity/config.py:35-38; hermetic.py clears the STREAM_* overrides.
    for path in POLICY.paths.values():
        assert (path.visuals.keep, path.visuals.block) == (
            DEFAULT_ACTIVE_VISUAL_BUDGET,
            DEFAULT_VISUAL_FOLD_BLOCK_SIZE,
        )
        assert (path.turns.keep, path.turns.block) == (
            DEFAULT_DETAILED_HISTORY_TURNS,
            DEFAULT_DETAILED_HISTORY_DROP_BLOCK_SIZE,
        )
        assert path.texts == FIXED_TEXTS


def test_retention_follows_the_environment(monkeypatch):
    monkeypatch.setenv("STREAM_ACTIVE_VISUAL_BUDGET", "8")
    monkeypatch.setenv("STREAM_VISUAL_FOLD_BLOCK_SIZE", "3")
    monkeypatch.setenv("STREAM_DETAILED_HISTORY_TURNS", "12")
    monkeypatch.setenv("STREAM_DETAILED_HISTORY_DROP_BLOCK_SIZE", "4")
    path = current_context_policy().paths["chat_completions"]
    assert (path.visuals.keep, path.visuals.block, path.turns.keep, path.turns.block) == (
        8,
        3,
        12,
        4,
    )


@pytest.mark.parametrize("api", sorted(APIS))
def test_feedback_placement(api):
    """Session paths prefix the next frame (harness.py:1011); a digest carries its own."""
    caps = capabilities(api)
    expected = (
        "frame_prefix"
        if caps.persistent_session
        else "digest"
        if caps.history_replay == "action_digest"
        else "separate_message"
    )
    assert POLICY.paths[api].feedback == expected


def test_presets():
    assert {api: path.preset for api, path in POLICY.paths.items() if path.preset} == {
        "openai_decisions": "luna",
        "jev_decisions": "jev",
        "moonshot": "kimi_k3",
    }
    for path in POLICY.paths.values():
        for item, value in PRESETS[path.preset].items() if path.preset else ():
            assert getattr(path, item) == value, (path.api, item)


def test_replay_and_pressure_sets():
    paths = POLICY.paths
    # Version 2: reasoning goes back only where the provider requires it (the Kimi K3 preset),
    # and every reply is rebuilt from the model output.
    assert {api for api, path in paths.items() if path.reasoning_replay} == {
        api for api in APIS if capabilities(api).reasoning_replay == "required"
    } == {"moonshot"}
    assert not any(path.verbatim_replay for path in paths.values())
    # Version 5: one pressure-archive rule, which the runtime applies where the API estimates its
    # requests (harnesses/default/context.py _run_turn). That is where version 4 had it on, plus
    # JEV, which estimates its digest instead of dropping history for one rejected option.
    assert all(path.pressure_archive for path in paths.values())
    assert {api for api in APIS if capabilities(api).request_size_estimation} == {
        "chat_completions",
        "openai_decisions",
        "jev_decisions",
    }


@pytest.mark.parametrize("api", sorted(APIS))
def test_decision_history(api):
    caps = capabilities(api)
    history = POLICY.paths[api].decision_history
    if caps.action_channel != "choice":
        assert history is None
        return
    assert history.form == ("turns" if caps.history_replay == "canonical" else "action_digest")


def test_ledger_roles():
    # Version 3: an input everywhere. Version 1 had OpenAI Realtime's system item
    # (openai_realtime_session.py:23, 205-221) and Luna's role-less state
    # (openai_decisions.py:132-163), which still places the ledger by position and header.
    assert {path.ledger_role for path in POLICY.paths.values()} == {"input"}


def test_gemini_tool_history_format():
    from realtime_framework.providers.gemini.live import LIVE_OPTIONS, live_options

    assert POLICY.paths["gemini_live"].tool_call_replay == LIVE_OPTIONS["history_format"][0]
    assert POLICY.for_path("gemini_live", {}).tool_call_replay == "native_tools"
    keys = [
        key
        for key in catalog.keys(provider="gemini")
        if catalog.get(key).adapter_type == "gemini_live"
    ]
    assert keys
    for key in keys:
        config = catalog.to_model_config(key, fps=1.0)
        resolved = POLICY.for_path(config.adapter_type, config.provider_options)
        assert resolved.tool_call_replay == live_options(config)["history_format"]
    assert all(
        path.tool_call_replay is None for api, path in POLICY.paths.items() if api != "gemini_live"
    )


# The profile each agent picked before item 11 existed: by action channel, and for choice models
# by observation modality and scenario. JEV rejects games (providers/preflight.py).
CHANNEL_PROFILES = {"text_json": "standard_vlm", "function_call": "function_call"}


@pytest.mark.parametrize("api", sorted(APIS))
def test_prompts_and_observation_form(api):
    caps = capabilities(api)
    path = POLICY.paths[api]
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


def test_policy_record_is_json():
    import json

    record = POLICY.paths["jev_decisions"].to_record()
    assert json.loads(json.dumps(record)) == record
    assert record["decision_history"] == {"form": "action_digest"}
