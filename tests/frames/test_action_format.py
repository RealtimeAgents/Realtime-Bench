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
from realtime_framework.harnesses.default.frames import INVALID_GROUNDING_FRAME, MISSING_FRAME
from realtime_framework.harnesses.default.harness import FUNCTION_CALL_PROFILE
from realtime_framework.harnesses.default.prompts import (
    FRAME_REFERENCE_RULES,
    FUNCTION_CALL_FRAME_REFERENCE_RULES,
)
from tests.replay.build import build_episode
from tests.replay.frames import ReplayTask
from tests.replay.scenario import TASK_INSTRUCTION, replies
from tests.replay.steps import Step
from tests.replay.stub import ChatCompletionsStub
from tests.replay.test_gemini_live import (  # noqa: F401 - live_env is a fixture
    API_KEY,
    MAX_TICKS,
    TOKENS_PER_FRAME,
    TOOL_TRANSCRIPT,
    catalog_config,
    live_env,
    turn_script,
)
from tests.replay.ws_gemini_live import LiveStub

from .episode import open_episode
from .test_grounding import SCENE


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


def test_the_text_json_instruction_gains_the_frame_reference_block(tmp_path):
    texts = {}
    with ChatCompletionsStub(replies([])) as stub:
        for mode in ("turn_based", "streaming"):
            episode = open_episode(stub, tmp_path / mode, interaction_mode=mode)
            texts[mode] = episode.agent._system_block.text
            episode.finish()
    assert FRAME_REFERENCE_RULES not in texts["turn_based"]
    assert texts["streaming"].count(FRAME_REFERENCE_RULES) == 1
    assert texts["streaming"].replace("\n\n" + FRAME_REFERENCE_RULES, "") == texts["turn_based"]


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


def test_gemini_live_calls_pass_their_frame(live_env, tmp_path):  # noqa: F811
    steps = [
        Step('{"action": "wait", "frame": 1}', status="wait"),
        Step('{"action": "click", "target": "P3", "frame": 1}'),
        Step('{"action": "click", "target": "P3"}'),
    ]
    preset = TOOL_TRANSCRIPT
    turns = [turn_script(k, step, preset) for k, step in enumerate(steps)]
    with LiveStub(
        turns,
        cert_dir=tmp_path / "tls",
        api_key=API_KEY,
        history_format=preset.history_format,
        ack_options=preset.ack_options,
    ) as stub:
        live_env.setenv("GOOGLE_GEMINI_BASE_URL", stub.base_url)
        live_env.setenv("SSL_CERT_FILE", str(stub.cert_file))
        episode = build_episode(
            catalog_config(preset.model),
            tmp_path / "run",
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=preset.model,
            clock_mode="token",
            tokens_per_frame=TOKENS_PER_FRAME,
            max_ticks=MAX_TICKS,
            before_initialize=lambda agent: setattr(agent, "interaction_mode", "streaming"),
            scene=SCENE,
        )
        agent, results = episode.agent, []
        try:
            for k, step in enumerate(steps):
                observation = episode.frames.capture(k)
                agent.configure_decision(episode.budget(observation.tick_number))
                agent.stream_prefill(observation)
                results.append(agent.stream_generate(tick=k))
                agent.finalize_decision_audit(results[-1])
                agent.commit_action(step.status, step.detail)
                stub.wait_until_idle()
        finally:
            agent.close()
    assert stub.violations == [] and stub.unused_turns == 0
    (setup,) = [event["msg"]["setup"] for event in stub.events if "setup" in event["msg"]]
    assert FUNCTION_CALL_FRAME_REFERENCE_RULES in setup["systemInstruction"]["parts"][0]["text"]
    for declaration in setup["tools"][0]["functionDeclarations"]:
        assert "frame" in declaration["parameters_json_schema"]["required"]

    first = agent.frames.locate(1)[0]
    click = results[1]
    assert click["function_validation_error"] is None
    assert click["action_protocol"]["frame_reference"]["frame"] == 1
    assert [click["action"].parameters["x"], click["action"].parameters["y"]] == first.marks[
        "P3"
    ]["point"]
    assert click["action"].coordinate_transform == first.geometry

    unframed = results[2]
    assert unframed["function_validation_error"] == "missing GUI function arguments"
    assert unframed["action"] is None
    assert unframed["action_protocol"]["rejection"] == INVALID_GROUNDING_FRAME
    assert unframed["action_protocol"]["frame_reference"]["error"] == MISSING_FRAME
