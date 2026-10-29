"""The harness in the gui+cli action space: the prompt tells the model how to run a script, and
the run's answer forms admit run_script, while a GUI-only run's prompts and parsers are as before.

The text JSON and function-call profiles carry the SCRIPTS section (InteractionProfile.cli_rules)
only under gui+cli; the text parser (VisionAgentBase._parse_action), a structured action and a
native function call (FunctionCallTurnCollector) read run_script by the run's action space; the
function declarations add run_script beside the GUI functions; and the script's report reaches
the model in its EXECUTION RESULT (DefaultHarness._report).
"""

from __future__ import annotations

import json
import types

import pytest

from realtime_framework.entity.actions import ActionType
from realtime_framework.harnesses.default.harness import (
    DefaultHarness,
    interaction_profile,
    script_answer_text,
    structured_action,
)
from realtime_framework.harnesses.default.prompts import (
    FUNCTION_CALL_CLI_RULES,
    TEXT_JSON_CLI_RULES,
)
from realtime_framework.harnesses.registry import configure_agent

SCRIPT = {"action": "run_script", "language": "python", "code": "import rtact\nrtact.key('a')"}


def _agent(action_space: str):
    agent = types.SimpleNamespace(task_instruction="Win the game.")
    configure_agent(
        agent,
        fps=1.0,
        tokens_per_frame=None,
        clock_mode="wall",
        category="browser_game",
        interaction_mode="turn_based",
        action_space=action_space,
    )
    return agent


@pytest.mark.parametrize(
    ("profile", "rules"),
    [("standard_vlm", TEXT_JSON_CLI_RULES), ("function_call", FUNCTION_CALL_CLI_RULES)],
)
def test_only_a_gui_cli_prompt_tells_how_to_run_a_script(profile, rules):
    cli = _agent("gui+cli")
    gui = _agent("gui")
    with_scripts = interaction_profile(profile, cli.prompt_options).system_instruction(cli)
    without = interaction_profile(profile, gui.prompt_options).system_instruction(gui)
    assert rules in with_scripts
    assert "SCRIPTS" not in without and "rtact" not in without
    # The section comes before the task, and says what a script cannot do.
    assert with_scripts.index("SCRIPTS") < with_scripts.index("TASK:")
    assert "without network, display or browser" in rules
    assert without == with_scripts.replace("\n\n" + rules, "")


def test_text_json_reads_a_script_only_under_gui_cli():
    agent = DefaultHarness.__new__(DefaultHarness)
    body = json.dumps(SCRIPT)
    agent.prompt_options = _agent("gui").prompt_options
    assert agent._parse_action(body) is None
    agent.prompt_options = _agent("gui+cli").prompt_options
    action = agent._parse_action(body)
    assert action.action_type is ActionType.RUN_SCRIPT
    assert action.parameters["code"] == SCRIPT["code"]
    # A timeout given as a string is read as the number it names, as a duration is.
    action = agent._parse_action(json.dumps({**SCRIPT, "timeout_s": "12"}))
    assert action.parameters["timeout_s"] == 12


def test_a_script_written_with_raw_newlines_is_read_under_gui_cli():
    """Models write the code into the JSON string as it is, which strict JSON refuses."""
    raw = '{"action": "run_script", "language": "python", "code": "import rtact\n\tx = 1\n"}'
    with pytest.raises(ValueError):
        json.loads(raw)
    for text in (raw, f"```json\n{raw}\n```"):
        assert json.loads(script_answer_text(text))["code"] == "import rtact\n\tx = 1\n"
    agent = DefaultHarness.__new__(DefaultHarness)
    agent.prompt_options = _agent("gui+cli").prompt_options
    action = agent._parse_action(script_answer_text(raw))
    assert action.action_type is ActionType.RUN_SCRIPT
    # Valid JSON and text that is no JSON object reach the parser as they came.
    for text in (json.dumps(SCRIPT), "click the button", '["a\nb"]'):
        assert script_answer_text(text) == text


def test_a_structured_action_reads_a_script_only_under_gui_cli():
    body = json.dumps(SCRIPT)
    assert structured_action(body) == (None, None)
    action, frame = structured_action(body, "gui+cli")
    assert action.action_type is ActionType.RUN_SCRIPT and frame is None


def test_the_function_channel_declares_run_script_only_under_gui_cli():
    for action_space, expected in (("gui", False), ("gui+cli", True)):
        agent = DefaultHarness.__new__(DefaultHarness)
        agent.prompt_options = _agent(action_space).prompt_options
        agent.driver = types.SimpleNamespace(
            capabilities=types.SimpleNamespace(action_channel="function_call")
        )
        agent.interaction_profile = interaction_profile("function_call", agent.prompt_options)
        names = [item["name"] for item in agent._answer_requirements().function_declarations]
        assert ("run_script" in names) is expected
        assert "click" in names


def test_a_native_function_call_is_read_by_the_run_s_action_space():
    from realtime_framework.harnesses.default.parsing import FunctionCallTurnCollector

    arguments = {key: value for key, value in SCRIPT.items() if key != "action"}

    def finish(action_space):
        agent = types.SimpleNamespace(prompt_options=_agent(action_space).prompt_options)
        collector = FunctionCallTurnCollector(agent)
        collector.calls = [
            types.SimpleNamespace(call_id="c1", name="run_script", arguments=arguments)
        ]
        collector.usage = {
            "output_accounting": "native",
            "native_response_usage_exact": True,
            "output_tokens": 7,
        }
        collector.complete = collector.terminal_usage_received = True
        return collector.finish()

    output = finish("gui+cli")
    assert json.loads(output.text) == SCRIPT
    assert output.token_fields["protocol_failure"] is None
    refused = finish("gui")
    assert refused.token_fields["protocol_failure"] == "invalid_action_output"
    assert "unknown GUI function" in refused.token_fields["function_validation_error"]


def test_the_scripts_report_is_the_models_execution_result():
    record = {"parse_failed": False, "requested": "run_script {...}"}
    detail = "run_script exit_code=0 after 1.20s; gateway actions=2 (rejected 0).\nstdout:\nok"
    result, feedback = DefaultHarness._report(record, "applied", detail)
    assert result["applied"] is True
    assert feedback.startswith("EXECUTION RESULT: actual=run_script {...}")
    assert feedback.endswith("Detail: " + detail)
