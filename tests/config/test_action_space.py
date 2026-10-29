"""The action space, gui or gui+cli, is a run setting like the interaction mode.

gui+cli adds one action, run_script (entity/actions.py), which only the parsers told the run's
action space admit; under gui it is an unknown verb and an unknown function, as before. The
setting reaches the prompts (PromptOptions.action_space), the pre-run checks (run_spec.py RunSpec,
providers/preflight.py check_action_space, which admit it on the turn-based wall clock only) and
the run identity: a gui+cli run records it in its effective configuration, so its runtime-config
hash differs, while a GUI-only run's configuration, hash and Harbor trial command line are as
they were. check_episode admits gui+cli only for a task whose prompt answers a script: text
JSON or a function call, not a choice.
"""

from __future__ import annotations

import types

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.actions import (
    DEFAULT_SCRIPT_TIMEOUT,
    MAX_SCRIPT_CODE_CHARS,
    MAX_SCRIPT_TIMEOUT,
    Action,
    ActionTranslator,
    ActionType,
    canonical_function_action,
    cli_function_declarations,
    gui_function_declarations,
    validate_action,
)
from realtime_framework.evaluation.config import config_fingerprint, effective_run_config
from realtime_framework.evaluation.run_spec import PLAN_OPTIONS, RunSpec, run_spec
from realtime_framework.harnesses.default.harness import action_text
from realtime_framework.providers.preflight import check_action_space, check_episode

from ..replay.frames import ReplayTask

MODEL = "claude-opus-5"
SCRIPT = {"language": "python", "code": "import rtact\nrtact.key('Up')\n"}


def script(**parameters) -> Action:
    return Action(ActionType.RUN_SCRIPT, {**SCRIPT, **parameters})


# The action


def test_a_script_needs_a_known_language_and_some_code():
    assert validate_action(script()) == (True, "")
    assert validate_action(script(language="bash", code="echo hi", timeout_s=5)) == (True, "")
    assert validate_action(script(language="ruby"))[0] is False
    assert validate_action(script(code="   \n"))[0] is False
    assert validate_action(script(code=None))[0] is False
    assert validate_action(Action(ActionType.RUN_SCRIPT, {"code": "x"}))[0] is False


def test_a_script_is_bounded_by_its_size_and_timeout():
    assert validate_action(script(code="x" * MAX_SCRIPT_CODE_CHARS))[0] is True
    assert validate_action(script(code="x" * (MAX_SCRIPT_CODE_CHARS + 1)))[0] is False
    assert validate_action(script(timeout_s=MAX_SCRIPT_TIMEOUT))[0] is True
    for timeout in (0, -1, MAX_SCRIPT_TIMEOUT + 1, float("inf"), True, "30"):
        assert validate_action(script(timeout_s=timeout))[0] is False, timeout
    ok, reason = validate_action(script(duration=1))
    assert not ok and "unexpected parameters: duration" in reason


def test_text_json_admits_a_script_only_under_gui_cli():
    data = {"action": "run_script", **SCRIPT, "timeout_s": 10}
    assert ActionTranslator.from_json(dict(data)) is None
    assert ActionTranslator.from_json(dict(data), action_space="gui") is None
    action = ActionTranslator.from_json(dict(data), action_space="gui+cli")
    assert action.action_type == ActionType.RUN_SCRIPT
    assert action.parameters == {**SCRIPT, "timeout_s": 10}
    # An invalid script is no action, as an invalid GUI action is.
    assert ActionTranslator.from_json({"action": "run_script"}, action_space="gui+cli") is None
    # GUI actions translate as before under either action space.
    for space in ("gui", "gui+cli"):
        key = ActionTranslator.from_json({"action": "key", "key": "enter"}, action_space=space)
        assert key.action_type == ActionType.KEY


def test_the_function_channel_admits_a_script_only_under_gui_cli():
    with pytest.raises(ValueError, match="unknown GUI function"):
        canonical_function_action("run_script", dict(SCRIPT))
    assert canonical_function_action("run_script", dict(SCRIPT), action_space="gui+cli") == {
        "action": "run_script",
        **SCRIPT,
    }
    with pytest.raises(ValueError, match="missing"):
        canonical_function_action("run_script", {"language": "python"}, action_space="gui+cli")
    with pytest.raises(ValueError, match="unexpected"):
        canonical_function_action("run_script", {**SCRIPT, "cwd": "/"}, action_space="gui+cli")
    with pytest.raises(ValueError, match="language"):
        canonical_function_action(
            "run_script", {**SCRIPT, "language": "ruby"}, action_space="gui+cli"
        )


def test_gui_cli_adds_one_function_beside_the_gui_functions():
    (declaration,) = cli_function_declarations()
    assert declaration["name"] == "run_script"
    schema = declaration["parameters_json_schema"]
    assert schema["required"] == ["language", "code"]
    assert set(schema["properties"]) == {"language", "code", "timeout_s"}
    assert schema["additionalProperties"] is False
    gui = {item["name"] for item in gui_function_declarations(clock_mode="wall")}
    assert "run_script" not in gui


def test_a_script_is_named_by_its_digest_not_its_source():
    text = action_text(script())
    assert text.startswith("run_script ")
    assert "rtact" not in text
    assert "'code_lines': 2" in text
    assert f"'timeout_s': {DEFAULT_SCRIPT_TIMEOUT}" in text
    assert action_text(script()) == text
    assert action_text(script(code="print(1)")) != text


# The run setting


def test_the_action_space_is_a_run_option_that_defaults_to_gui():
    assert "action_space" in PLAN_OPTIONS
    assert run_spec("runner").action_space == "gui"
    assert run_spec("harbor").action_space == "gui"
    spec = run_spec("trial", action_space="gui+cli")
    assert (spec.action_space, spec.clock_mode, spec.interaction_mode) == (
        "gui+cli",
        "wall",
        "turn_based",
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"action_space": "cli"}, "action_space must be 'gui' or 'gui\\+cli'"),
        ({"clock_mode": "token"}, "requires a wall clock"),
        ({"interaction_mode": "streaming"}, "requires turn_based interaction"),
    ],
)
def test_gui_cli_runs_only_turn_based_on_the_wall_clock(options, message):
    given = {"clock_mode": "wall", "action_space": "gui+cli", **options}
    with pytest.raises(ValueError, match=message):
        RunSpec(**given)
    with pytest.raises(ValueError, match=message):
        check_action_space(
            given["action_space"],
            clock_mode=given["clock_mode"],
            interaction_mode=given.get("interaction_mode", "turn_based"),
        )


def test_a_run_never_runs_scripts_unisolated(monkeypatch):
    """The local backend runs scripts as plain subprocesses; only the tests construct it."""
    monkeypatch.setenv("RTBENCH_WORKSPACE_BACKEND", "local")
    with pytest.raises(ValueError, match="the local backend is for tests only"):
        run_spec("trial", action_space="gui+cli")
    with pytest.raises(ValueError, match="the local backend is for tests only"):
        check_episode(
            catalog.to_model_config(MODEL, fps=1.0),
            clock_mode="wall",
            task=ReplayTask(),
            env_fps=1.0,
            action_space="gui+cli",
        )
    # A GUI-only run has no scripts to run.
    assert run_spec("trial").action_space == "gui"


def test_gui_cli_episodes_need_a_prompt_that_answers_a_script():
    config = catalog.to_model_config(MODEL, fps=1.0)
    check_episode(config, clock_mode="wall", task=ReplayTask(), env_fps=1.0)
    check_episode(config, clock_mode="wall", task=ReplayTask(), env_fps=1.0, action_space="gui+cli")
    # A choice API chooses among its observation's candidates, which hold no script.
    jev = catalog.to_model_config("jev-1.13", fps=1.0)
    check_episode(jev, clock_mode="wall", task=ReplayTask(), env_fps=1.0)
    with pytest.raises(ValueError, match="needs a text JSON or function-call prompt"):
        check_episode(
            jev, clock_mode="wall", task=ReplayTask(), env_fps=1.0, action_space="gui+cli"
        )


def _run_config(**options) -> dict:
    return effective_run_config(
        model_config=catalog.to_model_config(MODEL, fps=1.0),
        fps=1.0,
        max_minutes=5.0,
        protocol="wall-clock",
        think_budget_s=None,
        per_move_cap_s=None,
        mode="default",
        **options,
    )


def test_only_a_gui_cli_run_records_its_action_space(monkeypatch):
    monkeypatch.delenv("RTBENCH_WORKSPACE_BACKEND", raising=False)
    before = _run_config()
    gui = _run_config(action_space="gui")
    cli = _run_config(action_space="gui+cli")
    assert "action_space" not in gui
    assert config_fingerprint(gui) == config_fingerprint(before)
    assert cli["action_space"] == "gui+cli"
    assert config_fingerprint(cli) != config_fingerprint(gui)
    # What its scripts ran in is part of its identity too.
    assert len(cli["workspace_image_sources_sha256"]) == 64


def test_the_prompts_are_composed_for_the_action_space():
    from realtime_framework.harnesses.registry import configure_agent

    for given, expected in (({}, "gui"), ({"action_space": "gui+cli"}, "gui+cli")):
        agent = types.SimpleNamespace()
        configure_agent(
            agent,
            fps=1.0,
            tokens_per_frame=None,
            clock_mode="wall",
            category="realtime",
            interaction_mode="turn_based",
            **given,
        )
        assert agent.prompt_options.action_space == expected


def test_only_a_gui_cli_trial_names_its_action_space():
    from realtime_framework.evaluation.harbor import ApptainerEnvironment, RealtimeBenchOptions

    def trial_argv(action_space):
        options = RealtimeBenchOptions(action_space=action_space)
        request = {"model": MODEL, "mode": "default", **options.model_dump()}
        request["catalog_model"] = MODEL
        task = {
            "suite_id": "public",
            "task_id": "rt_waitlist_fit",
            **{
                field: request[field]
                for field in (
                    "fps",
                    "model_input_fps",
                    "interaction_mode",
                    "action_space",
                    "duration_minutes",
                    "seed",
                )
            },
        }
        environment = types.SimpleNamespace(python="python", lane=1, task=task)
        return ApptainerEnvironment._runner_command(environment, {**request, "task": task})[3:]

    assert "--action-space" not in trial_argv("gui")
    argv = trial_argv("gui+cli")
    assert argv[argv.index("--action-space") + 1] == "gui+cli"


def test_a_leaderboard_submission_is_gui_only():
    from realtime_framework.evaluation.submission import REQUIRED

    assert REQUIRED["action_space"] == "gui"
