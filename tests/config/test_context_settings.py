"""The context settings: their definitions (entity/context_policy.py SETTINGS) and the standard
values, which come from the package's defaults file alone (realtime_framework/context_defaults.yaml)."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from realtime_framework.entity.context_policy import (
    DEFAULTS_FILE,
    SETTINGS,
    PathPolicy,
    context_defaults,
    load_context_defaults,
)

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
        "version", "api", "preset", "ledger_role", "texts", "prompts",
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
        "context_defaults.yaml: the file holds standard alone; got ['defaults']",
    )
