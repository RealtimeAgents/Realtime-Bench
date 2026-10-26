"""What a chat_completions endpoint is set up with, as a model YAML, the inline endpoint
(remote-vlm, which has none) and the Harbor trial worker resolve it."""

from __future__ import annotations

import json
import os

import pytest

import realtime_framework.providers.registry as catalog
import realtime_tasks.registry as tasks
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.providers.preflight import check_episode

WEB = "rt_waitlist_fit"
ENDPOINT = "http://127.0.0.1:9/v1"
CATALOG_OPTIONS = (
    "Endpoint options apply to --model; catalog models use their own transport configuration"
)


@pytest.fixture(autouse=True)
def environment():
    """The trial worker writes os.environ (evaluation/_trial.py main)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


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


# Stream usage: requested unless the endpoint is declared to reject stream_options


@pytest.mark.parametrize(
    "setting, expected", [("", True), (", include_usage: false", False)], ids=["default", "off"]
)
def test_yaml_stream_usage(tmp_path, monkeypatch, setting, expected):
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.yaml").write_text(
        f"endpoint: {{api: chat_completions, model: m, base_url: {ENDPOINT}{setting}}}\n"
    )
    monkeypatch.setattr(catalog, "MODELS_DIR", models)
    catalog._catalog_entries.cache_clear()
    try:
        assert catalog.to_model_config("m", fps=1.0).chat_include_usage is expected
    finally:
        catalog._catalog_entries.cache_clear()


def test_built_config_stream_usage():
    """A chat_completions config built without a YAML, as the runner's --adapter path builds one
    (evaluation/runner.py main), asks for the usage as a YAML's does; no other API reads it."""
    assert ModelConfig(adapter_type="chat_completions", model_name="m").chat_include_usage is True
    assert ModelConfig(adapter_type="openrouter", model_name="m").chat_include_usage is False


@pytest.mark.parametrize(
    "variable, expected", [(None, True), ("1", True), ("0", False)], ids=["unset", "on", "off"]
)
def test_inline_endpoint_stream_usage(monkeypatch, variable, expected):
    if variable is None:
        monkeypatch.delenv("STREAM_CHAT_INCLUDE_USAGE", raising=False)
    else:
        monkeypatch.setenv("STREAM_CHAT_INCLUDE_USAGE", variable)
    assert catalog.to_model_config("remote-vlm", fps=1.0).chat_include_usage is expected


@pytest.mark.parametrize(
    "flags, variable, expected",
    [((), "1", True), (("--include-usage",), "1", True), (("--no-include-usage",), "0", False)],
    ids=["default", "on", "off"],
)
def test_trial_stream_usage(capsys, flags, variable, expected):
    settings = trial(capsys, "--model", "m", "--base-url", ENDPOINT, *flags)
    assert os.environ["STREAM_CHAT_INCLUDE_USAGE"] == variable
    assert settings["model_config"]["chat_include_usage"] is expected


@pytest.mark.parametrize("flag", ["--include-usage", "--no-include-usage"])
def test_trial_stream_usage_of_a_catalog_model(capsys, flag):
    usage_error(capsys, ["--catalog-model", "my-model", flag], CATALOG_OPTIONS)


@pytest.mark.parametrize("flag", ["--include-usage", "--no-include-usage"])
def test_trial_stream_usage_of_a_litellm_model(capsys, flag):
    usage_error(
        capsys,
        ["--model", "openai/m", flag],
        "LiteLLM collects response usage automatically; omit --include-usage",
    )


# Past reasoning: sent back where the model names the kimi_k3 preset, as Kimi requires for K3


def check(config):
    """The pre-run checks of the runner (evaluation/runner.py main) and the trial worker."""
    check_episode(config, clock_mode="wall", task=tasks.get(WEB), env_fps=config.fps)


def replays_reasoning(config) -> bool:
    policy = current_context_policy().for_path(
        config.adapter_type, config.provider_options, config.context_preset
    )
    return policy.reasoning_replay


def test_yaml_context_preset(tmp_path, monkeypatch):
    """K3 served behind a chat_completions endpoint names the kimi_k3 preset in its YAML, as its
    moonshot preset does (configs/models/kimi-k3.yaml): it sends past reasoning back, and the
    pre-run checks accept it. A chat_completions model without a preset sends none."""
    models = tmp_path / "models"
    models.mkdir()
    endpoint = f"endpoint: {{api: chat_completions, model: m, base_url: {ENDPOINT}}}\n"
    (models / "k3.yaml").write_text(endpoint + "context_preset: kimi_k3\n")
    (models / "plain.yaml").write_text(endpoint)
    monkeypatch.setattr(catalog, "MODELS_DIR", models)
    catalog._catalog_entries.cache_clear()
    try:
        for key, preset, replayed in (("k3", "kimi_k3", True), ("plain", None, False)):
            config = catalog.to_model_config(key, fps=1.0)
            assert (config.context_preset, replays_reasoning(config)) == (preset, replayed)
            check(config)
    finally:
        catalog._catalog_entries.cache_clear()


@pytest.mark.parametrize(
    "flags, preset", [((), None), (("--context-preset", "kimi_k3"), "kimi_k3")], ids=["none", "k3"]
)
def test_trial_context_preset(capsys, flags, preset):
    """An endpoint given on the command line has no YAML to name a preset in; the trial worker
    takes it instead, and the runner's inline endpoint (remote-vlm) applies it as a YAML's
    context_preset."""
    settings = trial(capsys, "--model", "m", "--base-url", ENDPOINT, *flags)
    assert os.environ["BENCHMARK_CONTEXT_PRESET"] == (preset or "")
    assert settings["model_config"]["context_preset"] == preset
    # What the runner builds from the environment the trial worker hands it.
    config = catalog.to_model_config("remote-vlm", fps=1.0)
    assert (config.context_preset, replays_reasoning(config)) == (preset, preset is not None)
    check(config)


def test_inline_endpoint_context_preset_must_be_a_model_preset(monkeypatch):
    monkeypatch.setenv("BENCHMARK_CONTEXT_PRESET", "jev")
    with pytest.raises(ValueError) as caught:
        catalog.to_model_config("remote-vlm", fps=1.0)
    assert str(caught.value) == "BENCHMARK_CONTEXT_PRESET must be one of ['kimi_k3']"


def test_trial_context_preset_choices(capsys):
    from realtime_framework.evaluation._trial import main

    with pytest.raises(SystemExit) as caught:
        main(["--task", WEB, "--model", "m", "--base-url", ENDPOINT, "--context-preset", "jev"])
    assert caught.value.code == 2
    assert "argument --context-preset: invalid choice: 'jev'" in capsys.readouterr().err


def test_trial_context_preset_of_a_catalog_model(capsys):
    usage_error(
        capsys,
        ["--catalog-model", "my-model", "--context-preset", "kimi_k3"],
        "A catalog model's YAML names its context_preset; omit --context-preset",
    )


def test_trial_context_preset_of_a_litellm_model(capsys):
    usage_error(
        capsys,
        ["--model", "openai/m", "--context-preset", "kimi_k3"],
        "--context-preset applies to an endpoint (--model DEPLOYMENT_ID --base-url URL), not a "
        "LiteLLM model",
    )
