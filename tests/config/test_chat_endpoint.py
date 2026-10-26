"""What a chat_completions endpoint is set up with, as a model YAML, the inline endpoint
(remote-vlm, which has none) and the Harbor trial worker resolve it."""

from __future__ import annotations

import json
import os

import pytest

import realtime_framework.providers.registry as catalog

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


def yaml_defaults(tmp_path, text: str) -> dict:
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.yaml").write_text(text)
    ((_, definition),) = catalog.load_definitions(models)
    return definition["defaults"]


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
def test_yaml_stream_usage(tmp_path, setting, expected):
    defaults = yaml_defaults(
        tmp_path, f"endpoint: {{api: chat_completions, model: m, base_url: {ENDPOINT}{setting}}}\n"
    )
    assert defaults["chat_include_usage"] is expected


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
