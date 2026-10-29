"""The context settings: their definitions (entity/context_policy.py SETTINGS), the standard
values and approved deviations, which come from the package's defaults file alone
(realtime_framework/context_defaults.yaml), and each model configuration's settings, resolved once
(resolve_context) into an immutable record that its dumps carry whole."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import (
    DEFAULTS_FILE,
    SETTINGS,
    Deviation,
    Overridden,
    PathPolicy,
    context_defaults,
    load_context_defaults,
    path_policy,
    resolve_context,
)
from realtime_framework.providers.registry import CONTEXT_PRESET_REMOVED

STANDARD = {
    "screenshots_kept": 20,
    "screenshot_fold_block": 10,
    "turns_kept": 50,
    "turn_archive_block": 10,
    "execution_feedback": "separate_message",
    "observation": "som_screenshot",
    "reasoning_replay": False,
    "archive_over_window": True,
    "tool_call_history": "native_tools",
    "decision_history": "turns",
}


def fails(action, message):
    with pytest.raises(ValueError) as caught:
        action()
    assert str(caught.value) == message


def write(tmp_path, standard, **sections) -> Path:
    path = tmp_path / "context_defaults.yaml"
    path.write_text(yaml.safe_dump({"standard": standard, **sections}, sort_keys=False))
    return path


def test_the_defaults_file_holds_the_standard_values():
    """The values of the policy before the file existed (entity/config.py DEFAULT_* and
    context_policy.py STANDARD at 8bec841b), in the order SETTINGS lists them."""
    assert context_defaults()["standard"] == STANDARD
    assert list(context_defaults()["standard"]) == list(SETTINGS)


def test_each_setting_is_a_policy_item_of_its_name():
    """The configuration keys are the policy's field names (commit 1); the policy's other
    fields are what the code decides."""
    names = [field.name for field in fields(PathPolicy)]
    assert set(SETTINGS) <= set(names)
    assert set(names) - set(SETTINGS) == {
        "version", "api", "ledger_role", "texts", "prompts",
    }  # fmt: skip


def test_the_defaults_file_is_part_of_the_harness_fingerprint():
    from realtime_framework.evaluation.config import _source_paths

    assert DEFAULTS_FILE.resolve() in {path.resolve() for path in _source_paths()}


def test_a_missing_defaults_file(tmp_path):
    fails(
        lambda: load_context_defaults(tmp_path / "context_defaults.yaml"),
        f"The context defaults file is missing: {tmp_path / 'context_defaults.yaml'}",
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (
            {"screenshots_kept": None},
            "context_defaults.yaml: standard lacks screenshots_kept",
        ),
        (
            {"screenshot_budget": 20},
            "context_defaults.yaml: standard has unknown settings ['screenshot_budget']",
        ),
        (
            {"turns_kept": "50"},
            "context_defaults.yaml: standard turns_kept must be a positive integer, got '50'",
        ),
        (
            {"turns_kept": 0},
            "context_defaults.yaml: standard turns_kept must be a positive integer, got 0",
        ),
        (
            {"reasoning_replay": 0},
            "context_defaults.yaml: standard reasoning_replay must be true or false, got 0",
        ),
        (
            {"observation": "screenshot"},
            (
                "context_defaults.yaml: standard observation must be one of ['som_screenshot', "
                "'observation_json_screenshot', 'observation_json_text'], got 'screenshot'"
            ),
        ),
        (
            {"screenshot_fold_block": 21},
            (
                "context_defaults.yaml: standard screenshot_fold_block (21) must be at most "
                "screenshots_kept (20)"
            ),
        ),
        (
            {"turn_archive_block": 51},
            (
                "context_defaults.yaml: standard turn_archive_block (51) must be at most "
                "turns_kept (50)"
            ),
        ),
    ],
    ids=["missing", "unknown", "type", "zero", "bool", "value", "fold-block", "archive-block"],
)
def test_a_defaults_file_that_does_not_hold(tmp_path, change, message):
    standard = {**STANDARD, **change}
    standard = {key: value for key, value in standard.items() if value is not None}
    fails(lambda: load_context_defaults(write(tmp_path, standard)), message)


def test_a_defaults_file_with_another_section(tmp_path):
    fails(
        lambda: load_context_defaults(write(tmp_path, STANDARD, defaults={})),
        "context_defaults.yaml: the file holds standard and approved_deviations; got ['defaults']",
    )


def _deviation(**fields):
    return {"api": "moonshot", "model": "kimi-k3", "settings": {"reasoning_replay": True},
            "reason": "Kimi requires it.", **fields}  # fmt: skip


@pytest.mark.parametrize(
    "entries, message",
    [
        (
            [_deviation(settings={"reasoning": True})],
            (
                "context_defaults.yaml: the approved deviation of moonshot kimi-k3: settings has "
                "unknown settings ['reasoning']"
            ),
        ),
        (
            [_deviation(settings={"reasoning_replay": "yes"})],
            (
                "context_defaults.yaml: the approved deviation of moonshot kimi-k3: settings "
                "reasoning_replay must be true or false, got 'yes'"
            ),
        ),
        (
            [_deviation(reason="")],
            "context_defaults.yaml: the approved deviation of moonshot kimi-k3: give its reason",
        ),
        (
            [_deviation(), _deviation()],
            "context_defaults.yaml: moonshot kimi-k3 is approved twice",
        ),
    ],
    ids=["unknown", "value", "reason", "twice"],
)
def test_approved_deviations_that_do_not_hold(tmp_path, entries, message):
    path = write(tmp_path, STANDARD, approved_deviations=entries)
    fails(lambda: load_context_defaults(path), message)


# The values each catalog model took before its YAML set them, from its API row's preset (Luna,
# JEV), its YAML's context_preset (Kimi K3) or its provider_options (Gemini Live):
# baseline/context-8bec841b.json. Each is an approved deviation of the model.
MODEL_VALUES = {
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


def test_each_catalog_model_takes_the_values_it_took_before():
    for key in catalog.CATALOG:
        config = catalog.to_model_config(key, fps=1.0)
        settings = config.context
        assert settings.values() == {**STANDARD, **MODEL_VALUES.get(key, {})}, key
        assert (settings.api, settings.model) == (config.adapter_type, config.model_name)
        assert settings.standard, key
        assert {(d.key, d.category) for d in settings.deviations} == {
            (item, "approved") for item in MODEL_VALUES.get(key, {})
        }, key
        assert {item.source for item in settings.items} <= {"default", "model"}, key


def _yaml(tmp_path, text):
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    (models / "m.yaml").write_text(text)
    return models


CHAT = "endpoint: {api: chat_completions, model: m, base_url: 'http://127.0.0.1:9/v1'}\n"


@pytest.mark.parametrize(
    "context, message",
    [
        (
            "{screenshots_kept: 8}",
            (
                "m.yaml: context a model may set only ['execution_feedback', 'observation', "
                "'reasoning_replay', 'tool_call_history', 'decision_history'], not screenshots_kept"
            ),
        ),
        (
            "{archive_over_window: false}",
            (
                "m.yaml: context a model may set only ['execution_feedback', 'observation', "
                "'reasoning_replay', 'tool_call_history', 'decision_history'], not "
                "archive_over_window"
            ),
        ),
        (
            "{tool_call_history: tool_transcript}",
            "m.yaml: context tool_call_history does not apply to the chat_completions API",
        ),
        (
            "{reasoning_replay: yes please}",
            "m.yaml: context reasoning_replay must be true or false, got 'yes please'",
        ),
        ("{reasoning: true}", "m.yaml: context has unknown settings ['reasoning']"),
        ("[reasoning_replay]", "m.yaml: context must be a mapping of settings to values"),
    ],
    ids=["retention", "archive", "not-applicable", "value", "unknown", "list"],
)
def test_a_model_sets_only_its_own_settings(tmp_path, context, message):
    models = _yaml(tmp_path, CHAT + f"context: {context}\n")
    fails(lambda: catalog.load_definitions(models), message)


def test_a_model_sets_the_settings_it_may(tmp_path):
    models = _yaml(tmp_path, CHAT + "context: {reasoning_replay: true}\n")
    ((_, definition),) = catalog.load_definitions(models)
    assert definition["context"] == {"reasoning_replay": True}


def test_the_removed_ways_to_set_context_values(tmp_path):
    """A YAML's context_preset and a Gemini Live YAML's provider_options.history_format are
    refused with where the setting went."""
    models = _yaml(tmp_path, CHAT + "context_preset: kimi_k3\n")
    fails(lambda: catalog.load_definitions(models), f"m.yaml: {CONTEXT_PRESET_REMOVED}")
    models = _yaml(
        tmp_path,
        "endpoint: {api: gemini_live, model: g}\nprovider_options: {history_format: native_tools}\n",
    )
    fails(
        lambda: catalog.load_definitions(models),
        "m.yaml: provider_options.history_format was removed; set tool_call_history in the "
        "context section instead (context: {tool_call_history: tool_transcript})",
    )


def test_the_resolution_records_where_each_value_came_from():
    settings = resolve_context(
        "gemini_live",
        "gemini-3.8-live",
        model_layer={"tool_call_history": "tool_transcript"},
        run={"screenshots_kept": 20, "turns_kept": 12},
    )
    item = settings.item("tool_call_history")
    assert (item.value, item.source, item.overridden, item.applies, item.category) == (
        "tool_transcript", "model", (Overridden(source="default", value="native_tools"),), True,
        "approved",
    )  # fmt: skip
    # Set to the standard value, a setting is recorded as given and stays standard.
    item = settings.item("screenshots_kept")
    assert (item.source, item.overridden, item.category) == (
        "run", (Overridden(source="default", value=20),), "standard",
    )  # fmt: skip
    item = settings.item("decision_history")
    assert (item.value, item.source, item.applies, item.category) == (
        "turns", "default", False, None,
    )  # fmt: skip
    assert not settings.standard
    assert settings.deviations == (
        Deviation(key="turns_kept", standard=50, value=12, source="run", category="nonstandard"),
        Deviation(
            key="tool_call_history", standard="native_tools", value="tool_transcript",
            source="model", category="approved",
        ),
    )  # fmt: skip


def test_a_value_the_model_is_not_approved_for_is_nonstandard():
    settings = resolve_context("chat_completions", "m", model_layer={"reasoning_replay": True})
    assert settings.item("reasoning_replay").category == "nonstandard"
    assert not settings.standard


def test_a_setting_that_does_not_apply_takes_no_part():
    """Kept, recorded as not applying, and left out of the policy (None) and of the verdict."""
    settings = resolve_context(
        "openrouter", "m", run={"tool_call_history": "tool_transcript", "decision_history": "turns"}
    )
    assert settings.item("tool_call_history").value == "tool_transcript"
    assert (settings.standard, settings.deviations) == (True, ())
    config = ModelConfig(adapter_type="openrouter", model_name="m", context=settings)
    policy = path_policy(config)
    assert (policy.tool_call_history, policy.decision_history) == (None, None)


def test_a_block_larger_than_what_it_folds():
    fails(
        lambda: resolve_context("chat_completions", "m", run={"screenshots_kept": 8}),
        "The context settings of m: screenshot_fold_block (10) must be at most "
        "screenshots_kept (8)",
    )


def test_a_configuration_built_without_context_takes_the_standard_values():
    """As a test builds a Decisions configuration without its YAML: the standard values, which
    the pre-run checks refuse for a choice API."""
    from realtime_framework.providers.preflight import check_context_policy

    config = ModelConfig(adapter_type="openai_decisions", model_name="openai/x")
    assert config.context.values() == STANDARD
    driver = catalog.adapter_class("openai_decisions")
    fails(
        lambda: check_context_policy(driver, path_policy(config), config),
        "The openai_decisions API chooses among the candidates of an observation JSON, not of "
        "the som_screenshot observation",
    )


def test_the_resolved_settings_cannot_change_even_read_back():
    config = catalog.to_model_config("kimi-k3", fps=1.0)
    read_back = ModelConfig.model_validate(json.loads(json.dumps(config.model_dump())))
    assert read_back.context == config.context
    for settings in (config.context, read_back.context):
        assert type(settings.items) is tuple and type(settings.deviations) is tuple
        assert all(type(item.overridden) is tuple for item in settings.items)
        with pytest.raises(ValidationError, match="frozen"):
            settings.standard = False
        with pytest.raises(ValidationError, match="frozen"):
            settings.items[0].value = 8
        with pytest.raises(ValidationError, match="frozen"):
            settings.deviations[0].value = False
        with pytest.raises(TypeError):
            settings.items[0] = settings.items[1]


def test_a_dump_is_read_back_as_it_was_and_never_resolved_again():
    config = catalog.to_model_config("kimi-k3", fps=1.0)
    dump = json.loads(json.dumps(config.model_dump()))
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({**dump, "context": config.context.values()})
    # A value of another type is refused, not converted.
    dump["context"]["items"][0]["value"] = 20.0
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(dump)
    dump["context"]["items"][0]["value"] = 20
    # A record changed by hand is read back as it is.
    replay = next(item for item in dump["context"]["items"] if item["key"] == "reasoning_replay")
    replay.update(source="run", category="nonstandard")
    item = ModelConfig.model_validate(dump).context.item("reasoning_replay")
    assert (item.value, item.source, item.category) == (True, "run", "nonstandard")


def test_the_configuration_hashes_take_the_values_alone():
    """configuration_sha256 (over the launch file) and the runtime-config hash take the settings'
    values: not where each came from, what it replaced, or how it stands."""
    from realtime_framework.evaluation.config import (
        config_fingerprint,
        configuration_sha256,
        run_spec_config,
    )
    from realtime_framework.evaluation.run_spec import run_spec

    spec = run_spec("trial")
    config = catalog.to_model_config("my-model", fps=1.0)

    def hashes(config):
        launch = {"model_config": config.model_dump(exclude={"api_key"}), "model_revision": None}
        return configuration_sha256(launch), config_fingerprint(run_spec_config(spec, config))

    def with_context(**layers):
        context = resolve_context(config.adapter_type, config.model_name, **layers)
        return config.model_copy(update={"context": context})

    same_values = with_context(run={"reasoning_replay": False, "observation": "som_screenshot"})
    assert same_values.context != config.context
    assert hashes(same_values) == hashes(config)
    dump = config.model_dump()
    dump["context"]["items"][0]["category"] = "nonstandard"
    dump["context"]["standard"] = False
    assert hashes(ModelConfig.model_validate(dump)) == hashes(config)
    changed = hashes(with_context(run={"reasoning_replay": True}))
    assert changed[0] != hashes(config)[0] and changed[1] != hashes(config)[1]
