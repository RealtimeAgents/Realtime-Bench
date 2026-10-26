"""Under streaming the action format asks for the frame each action is based on: a frame field in
the text JSON action, a required frame argument of every GUI function, and a FRAME REFERENCE
block in the system instruction; the action's marks and coordinates are that frame's. Turn-based
declarations and prompts are unchanged."""

from __future__ import annotations

import copy
import types

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.actions import (
    GUI_FUNCTION_ACTION_DESCRIPTION,
    GUI_FUNCTION_FRAME_ACTION_DESCRIPTION,
    GUI_FUNCTION_FRAME_DESCRIPTION,
    GUI_FUNCTION_FRAME_TARGET_DESCRIPTION,
    GUI_FUNCTION_TARGET_DESCRIPTION,
    canonical_function_action,
    gui_function_declarations,
)
from realtime_framework.entity.records import UNTRACED
from realtime_framework.harnesses.default.harness import interaction_profile
from realtime_framework.harnesses.default.prompts import (
    FRAME_REFERENCE_RULES,
    FUNCTION_CALL_FRAME_REFERENCE_RULES,
)
from realtime_framework.harnesses.registry import configure_agent
from realtime_framework.harnesses.streaming import agent as agent_module
from realtime_framework.harnesses.streaming.agent import NativeStreamingAgent
from tests.fake_clock import FakeClock
from tests.streaming.fake_live import LIVE, REALTIME, FakeLiveDriver, TextLiveDriver, observation


def _schemas(declarations):
    for item in declarations:
        schema = item["parameters_json_schema"]
        yield schema
        yield from schema.get("anyOf", ())


def test_streaming_functions_require_a_frame_argument():
    turn_based = gui_function_declarations()
    streaming = gui_function_declarations(frame_reference=True)
    assert not any("frame" in schema["properties"] for schema in _schemas(turn_based))
    for item in streaming:
        schema = item["parameters_json_schema"]
        assert schema["properties"]["frame"] == {
            "type": "integer",
            "minimum": 1,
            "description": GUI_FUNCTION_FRAME_DESCRIPTION,
        }
    for schema in _schemas(streaming):
        assert "frame" in schema["required"]
    # Nothing else differs but the marks and coordinates a call uses: its frame's.
    references = ("target", "start", "end")
    for item in streaming:
        assert item["description"].startswith(GUI_FUNCTION_FRAME_ACTION_DESCRIPTION)
        properties = item["parameters_json_schema"]["properties"]
        assert all(
            properties[name]["description"] == GUI_FUNCTION_FRAME_TARGET_DESCRIPTION
            for name in references
            if name in properties
        )
    stripped = copy.deepcopy(streaming)
    for item in stripped:
        item["description"] = item["description"].replace(
            GUI_FUNCTION_FRAME_ACTION_DESCRIPTION, GUI_FUNCTION_ACTION_DESCRIPTION
        )
        properties = item["parameters_json_schema"]["properties"]
        for name in references:
            if name in properties:
                properties[name]["description"] = GUI_FUNCTION_TARGET_DESCRIPTION
    for schema in _schemas(stripped):
        del schema["properties"]["frame"]
        schema["required"].remove("frame")
    assert stripped == turn_based


def test_a_streaming_call_keeps_its_frame_for_grounding():
    arguments = {"target": "P1", "frame": 3}
    assert canonical_function_action("click", arguments, frame_reference=True) == {
        "action": "click",
        **arguments,
    }
    with pytest.raises(ValueError, match="missing GUI function arguments"):
        canonical_function_action("click", {"target": "P1"}, frame_reference=True)
    with pytest.raises(ValueError, match="move requires numeric x and y"):
        canonical_function_action("move", {"frame": 3}, frame_reference=True)
    with pytest.raises(ValueError, match="unexpected GUI function arguments"):
        canonical_function_action("click", arguments)


# (a turn's text, a streaming run's) of each sentence streaming says otherwise (decision 6): an
# action's marks and coordinates are its frame's, nothing runs for no valid action, and a streaming
# run has replies, not decisions.
STREAMING_SENTENCES = [
    (
        "Never reuse a mark from an earlier screenshot.",
        "An action's marks are read from the screenshot its frame names.",
    ),
    (
        "supplied with the current screenshot.",
        "supplied with the screenshot the action's frame names.",
    ),
    (
        "multiple different actions, the harness executes WAIT.",
        "multiple different actions, nothing is executed for it.",
    ),
    ("Produce at most one action per decision", "Produce at most one action per reply"),
]


def _configure(agent, mode):
    configure_agent(
        agent,
        fps=1.0,
        tokens_per_frame=None,
        clock_mode="wall",
        category="realtime",
        interaction_mode=mode,
    )
    return agent


@pytest.mark.parametrize(
    ("driver", "model", "rules"),
    [
        (TextLiveDriver, REALTIME, FRAME_REFERENCE_RULES),
        (FakeLiveDriver, LIVE, FUNCTION_CALL_FRAME_REFERENCE_RULES),
    ],
    ids=["text_json", "function_call"],
)
def test_the_streaming_instruction_differs_from_a_turns_only_where_streaming_does(
    monkeypatch, driver, model, rules
):
    """The native-streaming agent's, configured as the runner configures it, against the
    instruction a turn of the same model composes on the wall clock: it gains its frame reference
    block and says otherwise only what streaming does otherwise. OpenAI Realtime answers in text
    JSON, Gemini Live in function calls."""
    monkeypatch.setattr(agent_module, "adapter_class", lambda adapter_type: driver)
    config = catalog.to_model_config(model, fps=1.0, reasoning_effort="fixed")
    clock = FakeClock(wall_origin=1000.0)
    with clock.participant():
        agent = _configure(
            NativeStreamingAgent(config, None, trace=UNTRACED, clock=clock), "streaming"
        )
        agent.initialize("Do the task.", observation(clock, 0))
        agent.close()
    streaming = agent._seed.system_instruction
    name = agent.interaction_profile.name
    turn = _configure(types.SimpleNamespace(task_instruction="Do the task."), "turn_based")
    turn_based = interaction_profile(name, turn.prompt_options).system_instruction(turn)
    assert rules not in turn_based
    assert streaming.count(rules) == 1
    expected = streaming.replace("\n\n" + rules, "")
    for turn_text, streaming_text in STREAMING_SENTENCES:
        expected = expected.replace(streaming_text, turn_text)
    assert expected == turn_based
