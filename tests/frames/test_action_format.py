"""Under streaming the action format asks for the frame each action is based on: a frame field in
the text JSON action, a required frame argument of every GUI function, and a FRAME REFERENCE
block in the system instruction. Turn-based declarations and prompts are unchanged."""

from __future__ import annotations

import copy
import types

import pytest

from realtime_framework.entity.actions import (
    GUI_FUNCTION_FRAME_DESCRIPTION,
    canonical_function_action,
    gui_function_declarations,
)
from realtime_framework.harnesses.default.harness import FUNCTION_CALL_PROFILE
from realtime_framework.harnesses.default.prompts import (
    FRAME_REFERENCE_RULES,
    FUNCTION_CALL_FRAME_REFERENCE_RULES,
    gui_system_prompt,
)
from tests.fake_clock import FakeClock
from tests.streaming.fake_live import LIVE, REALTIME, FakeLiveDriver, TextLiveDriver, make_agent


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
    # Nothing else differs.
    stripped = copy.deepcopy(streaming)
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


@pytest.mark.parametrize(("driver", "model"), [(TextLiveDriver, REALTIME), (FakeLiveDriver, LIVE)])
def test_the_streaming_instruction_gains_its_frame_reference_block(monkeypatch, driver, model):
    """The native-streaming agent's, against the instruction a turn composes of the same parts:
    no scenario prompt or clock note is configured here (tests/streaming/fake_live.py
    make_agent). OpenAI Realtime answers in text JSON, Gemini Live in function calls."""
    clock = FakeClock(wall_origin=1000.0)
    with clock.participant():
        agent = make_agent(monkeypatch, clock, driver, model)
        agent.close()
    streaming = agent._seed.system_instruction
    if model == REALTIME:
        rules, turn_based = FRAME_REFERENCE_RULES, gui_system_prompt("Click the button.")
    else:
        rules = FUNCTION_CALL_FRAME_REFERENCE_RULES
        turn_based = FUNCTION_CALL_PROFILE.system_instruction(
            types.SimpleNamespace(
                interaction_mode="turn_based",
                task_category="realtime",
                streaming_note=None,
                task_instruction="Click the button.",
            )
        )
    assert rules not in turn_based
    assert streaming.count(rules) == 1
    assert streaming.replace("\n\n" + rules, "") == turn_based


def test_the_function_call_instruction_gains_its_frame_reference_block():
    def agent(mode):
        return types.SimpleNamespace(
            interaction_mode=mode,
            task_category="realtime",
            streaming_note="CLOCK NOTE",
            task_instruction="Do the task.",
        )

    turn_based = FUNCTION_CALL_PROFILE.system_instruction(agent("turn_based"))
    streaming = FUNCTION_CALL_PROFILE.system_instruction(agent("streaming"))
    assert FUNCTION_CALL_FRAME_REFERENCE_RULES not in turn_based
    assert streaming.replace("\n\n" + FUNCTION_CALL_FRAME_REFERENCE_RULES, "") == turn_based
