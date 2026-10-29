"""The context settings: their definitions (entity/context_policy.py SETTINGS), the standard
values and approved deviations, which come from the package's defaults file alone
(realtime_framework/context_defaults.yaml), and each model configuration's settings, resolved once
(resolve_context) into an immutable record that its dumps carry whole."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import (
    CONTEXT_PRESET_OPTION_REMOVED,
    DEFAULTS_FILE,
    LEGACY_CONTEXT_VARIABLES,
    SETTINGS,
    ContextLayers,
    Deviation,
    ExperimentFile,
    Overridden,
    PathPolicy,
    context_defaults,
    load_context_defaults,
    path_policy,
    read_experiment_file,
    resolve_context,
)
from realtime_framework.providers.registry import CONTEXT_PRESET_REMOVED

WEB = "rt_waitlist_fit"
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


# The layers a run adds: an experiment file and the trial worker's command line


EVERY = {
    "screenshots_kept": 8,
    "screenshot_fold_block": 3,
    "turns_kept": 12,
    "turn_archive_block": 4,
    "execution_feedback": "digest",
    "observation": "observation_json_text",
    "reasoning_replay": True,
    "archive_over_window": False,
    "tool_call_history": "tool_transcript",
    "decision_history": "action_digest",
}


def test_each_layer_takes_the_place_of_the_one_below():
    """The defaults file, then the experiment file, then the model's YAML, then the run's command
    line; each value records the ones it replaced."""
    file = ExperimentFile(path="/experiments/a.yaml", sha256="0" * 64)
    settings = resolve_context(
        "moonshot",
        "kimi-k3",
        experiment={"reasoning_replay": False, "screenshots_kept": 12, "turns_kept": 30},
        experiment_file=file,
        model_layer={"reasoning_replay": True},
        run={"turns_kept": 40},
    )
    replay = settings.item("reasoning_replay")
    assert (replay.value, replay.source, replay.overridden) == (
        True, "model",
        (Overridden(source="default", value=False), Overridden(source="experiment", value=False)),
    )  # fmt: skip
    turns = settings.item("turns_kept")
    assert (turns.value, turns.source, turns.overridden) == (
        40, "run",
        (Overridden(source="default", value=50), Overridden(source="experiment", value=30)),
    )  # fmt: skip
    assert settings.item("screenshots_kept").source == "experiment"
    assert settings.experiment_file == file


@pytest.mark.parametrize("layer", ["experiment", "run"])
def test_an_experiment_file_and_the_command_line_set_any_setting(layer):
    """Every setting, those that do not apply to the API among them, which are kept and recorded
    as not applying."""
    settings = resolve_context("chat_completions", "m", **{layer: EVERY})
    assert settings.values() == EVERY
    assert {item.key for item in settings.items if not item.applies} == {
        "tool_call_history", "decision_history",
    }  # fmt: skip


def test_the_hashes_ignore_where_the_values_came_from(tmp_path):
    """An experiment file's path and digest, and what each value replaced, are recorded, not
    hashed: the same values from two files, from the command line or as the standard ones give
    the same configuration and runtime-config hashes."""
    first = tmp_path / "first.yaml"
    first.write_text("reasoning_replay: false\n")
    second = tmp_path / "elsewhere" / "second.yaml"
    second.parent.mkdir()
    second.write_text("# The standard value, from another file.\nreasoning_replay: false\n")
    config = catalog.to_model_config("my-model", fps=1.0)
    configs = [config]
    for path in (first, second):
        experiment_file, experiment = read_experiment_file(path)
        layers = ContextLayers(experiment=experiment, experiment_file=experiment_file)
        configs.append(catalog.to_model_config("my-model", fps=1.0, context=layers))
    layers = ContextLayers(run={"reasoning_replay": False})
    configs.append(catalog.to_model_config("my-model", fps=1.0, context=layers))
    assert len({config.context for config in configs}) == len(configs)
    assert len({_hashes(config) for config in configs}) == 1


def _hashes(config):
    from realtime_framework.evaluation.config import (
        config_fingerprint,
        configuration_sha256,
        run_spec_config,
    )
    from realtime_framework.evaluation.run_spec import run_spec

    launch = {"model_config": config.model_dump(exclude={"api_key"}), "model_revision": None}
    spec = run_spec("trial")
    return configuration_sha256(launch), config_fingerprint(run_spec_config(spec, config))


@pytest.fixture
def environment():
    """The trial worker writes os.environ (evaluation/_trial.py main), from which the catalog
    takes an inline endpoint's model (providers/registry.py _load_entries): a fresh catalog for
    each test, as each trial worker is a fresh process."""
    saved = dict(os.environ)
    catalog._catalog_entries.cache_clear()
    yield
    os.environ.clear()
    os.environ.update(saved)
    catalog._catalog_entries.cache_clear()


def trial(capsys, *argv) -> dict:
    """The settings the trial worker resolves (its dry run prints them)."""
    from realtime_framework.evaluation._trial import main

    assert main(["--task", WEB, "--dry-run", *argv]) == 0
    return json.loads(capsys.readouterr().out)


def usage_error(capsys, argv, message):
    """An argparse error: exit status 2 and the message as the last stderr line."""
    from realtime_framework.evaluation._trial import main

    with pytest.raises(SystemExit) as caught:
        main(["--task", WEB, "--dry-run", *argv])
    assert caught.value.code == 2
    assert capsys.readouterr().err.strip().splitlines()[-1].endswith("error: " + message)


def _items(settings) -> dict:
    return {item["key"]: item for item in settings["model_config"]["context"]["items"]}


def test_the_trial_worker_takes_an_experiment_file_and_settings(environment, tmp_path, capsys):
    path = tmp_path / "experiment.yaml"
    path.write_text("screenshots_kept: 8\nscreenshot_fold_block: 3\nreasoning_replay: false\n")
    settings = trial(
        capsys, "--catalog-model", "my-model", "--context-file", str(path),
        "--context", "screenshots_kept=12", "--context", "observation=som_screenshot",
    )  # fmt: skip
    items = _items(settings)
    assert items["screenshots_kept"] == {
        "key": "screenshots_kept", "value": 12, "source": "run",
        "overridden": [{"source": "default", "value": 20}, {"source": "experiment", "value": 8}],
        "applies": True, "category": "nonstandard",
    }  # fmt: skip
    assert (items["screenshot_fold_block"]["value"], items["screenshot_fold_block"]["source"]) == (
        3, "experiment",
    )  # fmt: skip
    assert items["observation"]["source"] == "run"
    assert settings["model_config"]["context"]["experiment_file"] == {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--context", "screenshots_kept"], "--context takes KEY=VALUE, got 'screenshots_kept'"),
        (
            ["--context", "screenshots_kept=20.0"],
            (
                "The run context settings of YOUR_MODEL: screenshots_kept must be a positive "
                "integer, got 20.0"
            ),
        ),
        (
            ["--context", "reasoning_replay=1.0"],
            (
                "The run context settings of YOUR_MODEL: reasoning_replay must be true or false, "
                "got 1.0"
            ),
        ),
        (
            ["--context", "reasoning=true"],
            "The run context settings of YOUR_MODEL: has unknown settings ['reasoning']",
        ),
    ],
    ids=["no-value", "float-count", "float-bool", "unknown"],
)
def test_the_trial_worker_refuses_a_setting_it_cannot_take(environment, capsys, argv, message):
    usage_error(capsys, ["--catalog-model", "my-model", *argv], message)


def test_the_trial_worker_refuses_an_experiment_file_it_cannot_take(environment, tmp_path, capsys):
    path = tmp_path / "experiment.yaml"
    path.write_text("[screenshots_kept]\n")
    usage_error(
        capsys,
        ["--catalog-model", "my-model", "--context-file", str(path)],
        f"{path.resolve()}: a context file is a mapping of context settings to values",
    )
    path.write_text("screenshots_kept: many\n")
    usage_error(
        capsys,
        ["--catalog-model", "my-model", "--context-file", str(path)],
        "The experiment context settings of YOUR_MODEL: screenshots_kept must be a positive "
        "integer, got 'many'",
    )


@pytest.mark.parametrize("variable", sorted(LEGACY_CONTEXT_VARIABLES))
def test_the_settings_context_settings_replaced_are_refused(environment, capsys, variable):
    """Not ignored: each removed variable stops the trial, naming the setting in its place."""
    os.environ[variable] = "8"
    usage_error(
        capsys,
        ["--catalog-model", "my-model"],
        f"Unset {variable}: context settings replaced them ({variable} by "
        f"{LEGACY_CONTEXT_VARIABLES[variable]}); give them with --context KEY=VALUE or in an "
        "experiment file (--context-file)",
    )


def test_the_context_preset_option_is_refused(environment, capsys):
    usage_error(
        capsys,
        ["--model", "m", "--base-url", "http://127.0.0.1:9/v1", "--context-preset", "kimi_k3"],
        CONTEXT_PRESET_OPTION_REMOVED,
    )


def _harbor_command(**options):
    """The trial worker's command line Harbor runs for these agent options
    (evaluation/harbor.py ApptainerEnvironment._runner_command)."""
    import types

    from realtime_framework.evaluation.harbor import ApptainerEnvironment, RealtimeBenchOptions

    options = RealtimeBenchOptions(**options)
    request = {"model": "my-model", "mode": "default", **options.model_dump()}
    request["catalog_model"] = "my-model"
    task = {
        "suite_id": "public", "task_id": WEB,
        **{
            field: request[field]
            for field in ("fps", "model_input_fps", "interaction_mode", "duration_minutes", "seed")
        },
    }  # fmt: skip
    environment_ = types.SimpleNamespace(python="python", lane=1, task=task)
    return ApptainerEnvironment._runner_command(environment_, {**request, "task": task})


def test_harbor_passes_the_context_settings_to_the_trial_worker(
    environment, tmp_path, capsys, monkeypatch
):
    """--ak context_file=PATH (made absolute) and --ak context='{...}' reach the trial worker as
    --context-file and --context, the values as given."""
    from realtime_framework.evaluation import _trial

    path = tmp_path / "experiment.yaml"
    path.write_text("turns_kept: 30\n")
    monkeypatch.chdir(tmp_path)
    command = _harbor_command(
        context_file="experiment.yaml",
        context={"screenshots_kept": 12, "reasoning_replay": True, "observation": "som_screenshot"},
    )
    assert command[command.index("--context-file") + 1] == str(path.resolve())
    assert [command[i + 1] for i, part in enumerate(command) if part == "--context"] == [
        "screenshots_kept=12", "reasoning_replay=true", 'observation="som_screenshot"',
    ]  # fmt: skip
    assert _trial.main([*command[3:], "--dry-run"]) == 0
    items = _items(json.loads(capsys.readouterr().out))
    assert [items[key]["value"] for key in ("screenshots_kept", "turns_kept", "reasoning_replay")] == [
        12, 30, True,
    ]  # fmt: skip


@pytest.mark.parametrize(
    "context, message",
    [
        ({"screenshots_kept": 20.0}, "screenshots_kept must be a positive integer, got 20.0"),
        ({"reasoning_replay": 1.0}, "reasoning_replay must be true or false, got 1.0"),
        ({"reasoning_replay": "true"}, "reasoning_replay must be true or false, got 'true'"),
    ],
    ids=["float-count", "float-bool", "string-bool"],
)
def test_harbor_refuses_a_setting_of_another_type(context, message):
    """Harbor's options keep the values as given, so a value of another type is refused, as the
    trial worker refuses it, not converted to one that passes."""
    from realtime_framework.evaluation.harbor import RealtimeBenchOptions

    with pytest.raises(ValidationError) as caught:
        RealtimeBenchOptions(context=context)
    assert f"context {message}" in str(caught.value)


def test_harbor_finds_an_experiment_file_changed_after_resolution(
    environment, tmp_path, capsys, monkeypatch
):
    """Harbor resolves a trial, then runs it, each time through the trial worker; a value changed
    in the experiment file in between changes the configuration hash, which verify compares
    (evaluation/harbor.py resolve, verify). A change that leaves the values alone does not."""
    from realtime_framework.evaluation import _trial
    from realtime_framework.evaluation.config import configuration_sha256

    path = tmp_path / "experiment.yaml"
    path.write_text("turns_kept: 30\n")
    command = [*_harbor_command(context_file=str(path))[3:], "--dry-run"]

    def resolved():
        assert _trial.main(command) == 0
        return configuration_sha256(json.loads(capsys.readouterr().out))

    before = resolved()
    path.write_text("# A comment changes the file, not its values.\nturns_kept: 30\n")
    assert resolved() == before
    path.write_text("turns_kept: 40\n")
    assert resolved() != before


# Generic LiteLLM: past reasoning goes back only to the providers whose conversion keeps it


def test_generic_litellm_sends_past_reasoning_to_a_provider_that_keeps_it(environment, capsys):
    """`--model provider/model-id` takes reasoning replay from the command line, for a provider
    whose LiteLLM conversion keeps an assistant message's reasoning_content."""
    settings = trial(capsys, "--model", "hosted_vllm/m", "--context", "reasoning_replay=true")
    config = ModelConfig.model_validate(settings["model_config"])
    item = config.context.item("reasoning_replay")
    assert (config.adapter_type, item.value, item.source) == ("litellm", True, "run")
    assert path_policy(config).reasoning_replay is True


@pytest.mark.parametrize("model", ["openai/m", "anthropic/m", "gemini/m", "vertex_ai/m"])
def test_generic_litellm_refuses_past_reasoning_elsewhere(environment, capsys, model):
    usage_error(
        capsys,
        ["--model", model, "--context", "reasoning_replay=true"],
        "The litellm API sends past reasoning back only to ['deepseek', 'moonshot', "
        f"'together_ai', 'hosted_vllm'] models, whose LiteLLM conversions keep it, not to {model}",
    )


# Kimi K3: its documentation requires its past reasoning back


def _check(config):
    """The pre-run checks of the trial worker (providers/preflight.py check_episode)."""
    import realtime_tasks.registry as tasks
    from realtime_framework.providers.preflight import check_episode

    check_episode(config, clock_mode="wall", task=tasks.get(WEB), env_fps=config.fps)


K3_REQUIRES = (
    "kimi-k3 on the moonshot API runs only with reasoning_replay: true (Kimi's thinking-model "
    "guide requires K3's past reasoning_content back in multi-turn conversations and tool-call "
    "loops.); set it in the model YAML's context section or with --context KEY=VALUE"
)


def test_kimi_k3_runs_only_with_its_past_reasoning(environment, tmp_path, monkeypatch, capsys):
    """An approved deviation the defaults file marks required: K3's YAML without it, or a run
    that turns it off, is refused before the run; another moonshot model runs without it."""
    models = tmp_path / "models"
    models.mkdir()
    text = (catalog.MODELS_DIR / "kimi-k3.yaml").read_text()
    (models / "kimi-k3.yaml").write_text(text.replace("context:\n  reasoning_replay: true\n", ""))
    monkeypatch.setattr(catalog, "MODELS_DIR", models)
    catalog._catalog_entries.cache_clear()
    fails(lambda: _check(catalog.to_model_config("kimi-k3", fps=1.0)), K3_REQUIRES)
    monkeypatch.undo()
    catalog._catalog_entries.cache_clear()
    usage_error(
        capsys, ["--catalog-model", "kimi-k3", "--context", "reasoning_replay=false"], K3_REQUIRES
    )
    _check(catalog.to_model_config("kimi-k3", fps=1.0))
    _check(ModelConfig(adapter_type="moonshot", model_name="kimi-other", reasoning_effort=None))


def test_kimi_k3_through_generic_litellm_runs_only_with_its_past_reasoning(environment, capsys):
    usage_error(
        capsys,
        ["--model", "moonshot/kimi-k3"],
        K3_REQUIRES.replace("kimi-k3 on the moonshot API", "moonshot/kimi-k3 on the litellm API"),
    )
    catalog._catalog_entries.cache_clear()
    settings = trial(capsys, "--model", "moonshot/kimi-k3", "--context", "reasoning_replay=true")
    context = ModelConfig.model_validate(settings["model_config"]).context
    assert (context.item("reasoning_replay").category, context.standard) == ("approved", True)


def test_a_required_deviation_is_true_or_false(tmp_path):
    path = write(tmp_path, STANDARD, approved_deviations=[_deviation(required="yes")])
    fails(
        lambda: load_context_defaults(path),
        "context_defaults.yaml: the approved deviation of moonshot kimi-k3: required must be true "
        "or false, got 'yes'",
    )
