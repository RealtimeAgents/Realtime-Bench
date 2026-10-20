"""A streaming action is interpreted against the frame it names: that frame's marks and geometry,
however many frames came after it and wherever its elements are now. An action whose frame is
missing, not an integer or not in the frame table is rejected, never executed. Every grounded
action records its frame's age. Turn-based actions keep their format and the newest frame."""

from __future__ import annotations

import io
import json

from PIL import Image

from realtime_framework.entity.actions import normalize_action
from realtime_framework.harnesses.default.frames import (
    FRAME_NOT_IN_TABLE,
    INVALID_FRAME_SYNTAX,
    INVALID_GROUNDING_FRAME,
    MISSING_FRAME,
)
from tests.replay import frames as replay_frames
from tests.replay.scenario import replies
from tests.replay.steps import Step
from tests.replay.stub import ChatCompletionsStub

from .episode import decide, open_episode

SUBMIT = "dom-0003"


def moving_probe(frame: int, *, text_observation: bool = False) -> dict:
    """The replay form, with its Submit button (mark P3) 120 px further right in every frame."""
    value = replay_frames.probe(frame, text_observation=text_observation)
    (submit,) = [item for item in value["actionable_now"] if item["id"] == SUBMIT]
    shift = 120 * frame
    submit["bbox"] = [700 + shift, 450, 900 + shift, 510]
    submit["click_point"] = [800 + shift, 480]
    return value


def first_frame_taller(frame: int) -> bytes:
    """Frame 1, the first decision's, is 4:3; its model image and geometry differ from the rest."""
    if frame != 1:
        return replay_frames.screenshot(frame)
    buffer = io.BytesIO()
    Image.new("RGB", (1600, 1200), (245, 245, 245)).save(buffer, format="PNG")
    return buffer.getvalue()


SCENE = {"screenshot": first_frame_taller, "probe": moving_probe}


def _run(tmp_path, steps, *, interaction_mode="streaming"):
    with ChatCompletionsStub(replies(steps)) as stub:
        episode = open_episode(stub, tmp_path, interaction_mode=interaction_mode, scene=SCENE)
        try:
            results = [decide(episode, k, step.status)[1] for k, step in enumerate(steps)]
        finally:
            episode.finish()
    trajectory = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
        if json.loads(line).get("type") == "decision"
    ]
    return episode.agent.frames, results, trajectory


def test_an_action_uses_the_marks_and_geometry_of_the_frame_it_names(tmp_path):
    steps = [
        Step('{"action": "wait", "frame": 1}', status="wait"),
        Step('{"action": "wait", "frame": 2}', status="wait"),
        Step('{"action": "click", "target": "P3", "frame": 1}'),
        Step('{"action": "click", "x": 300, "y": 200, "frame": 1}'),
    ]
    table, results, _ = _run(tmp_path, steps)
    first, newest = table.locate(1)[0], table.latest()
    assert first.marks["P3"]["point"] != newest.marks["P3"]["point"]
    assert first.geometry != newest.geometry

    by_mark = results[2]["action"]
    assert [by_mark.parameters["x"], by_mark.parameters["y"]] == first.marks["P3"]["point"]
    assert by_mark.coordinate_transform == first.geometry
    assert results[2]["action_protocol"]["grounding"]["resolutions"][0]["mark"] == "P3"

    by_coordinates = results[3]["action"]
    assert by_coordinates.coordinate_transform == first.geometry
    desktop, _ = normalize_action(by_coordinates)
    assert (desktop.parameters["x"], desktop.parameters["y"]) == (750, 375)  # scale 2.5 x 1.875


def test_each_grounded_action_records_its_frame_and_age(tmp_path):
    steps = [
        Step('{"action": "wait", "frame": 1}', status="wait"),
        Step('{"action": "wait", "frame": 2}', status="wait"),
        Step('{"action": "wait", "frame": 1}', status="wait"),
    ]
    table, results, trajectory = _run(tmp_path, steps)
    references = [result["action_protocol"]["frame_reference"] for result in results]
    assert [(r["frame"], r["age_frames"]) for r in references] == [(1, 0), (2, 0), (1, 2)]
    for reference in references:
        frame = table.locate(reference["frame"])[0]
        assert reference["observation_id"] == frame.observation_id
        assert reference["age_s"] > 0
    # Frame 1 was captured first, so an action reading it later finds it older.
    assert references[2]["age_s"] > references[0]["age_s"]
    assert [row["action_protocol"]["frame_reference"] for row in trajectory] == references
    assert results[0]["action_protocol"]["parse_mode"] == "canonical_json"
    assert results[0]["action_protocol"]["source"] == "model_action"


def test_an_action_without_a_usable_frame_is_rejected(tmp_path):
    steps = [
        Step('{"action": "click", "target": "P3"}'),
        Step('{"action": "click", "target": "P3", "frame": "one"}'),
        Step('{"action": "click", "target": "P3", "frame": 1.0}'),
        Step('{"action": "click", "target": "P3", "frame": 9}'),
        Step('{"action": "click", "target": "P3", "frame": 0}'),
    ]
    _, results, trajectory = _run(tmp_path, steps)
    expected = [
        (None, MISSING_FRAME),
        ("one", INVALID_FRAME_SYNTAX),
        (1.0, INVALID_FRAME_SYNTAX),
        (9, FRAME_NOT_IN_TABLE),
        (0, FRAME_NOT_IN_TABLE),
    ]
    for result, row, (value, error) in zip(results, trajectory, expected, strict=True):
        assert result["action"] is None
        protocol = result["action_protocol"]
        assert protocol["rejection"] == INVALID_GROUNDING_FRAME
        assert protocol["frame_reference"] == {"frame": value, "error": error}
        assert protocol["protocol_failure"] is None
        assert row["parsed_action"] is None
        assert row["action_protocol"]["rejection"] == INVALID_GROUNDING_FRAME


def test_an_invalid_action_is_a_protocol_failure_not_a_rejection(tmp_path):
    steps = [
        Step('{"action": "wait", "frame": 1}', status="wait"),
        Step('{"action": "click", "target": "P9", "frame": 1}', status="wait"),
    ]
    _, results, _ = _run(tmp_path, steps)
    protocol = results[1]["action_protocol"]
    assert protocol["protocol_failure"] == "invalid_action_output"
    assert "rejection" not in protocol and "frame_reference" not in protocol
    assert results[1]["action"].action_type.value == "wait"


def test_turn_based_actions_name_no_frame(tmp_path):
    steps = [
        Step('{"action": "click", "target": "P3"}'),
        Step('{"action": "click", "target": "P3", "frame": 1}', status="wait"),
    ]
    table, results, _ = _run(tmp_path, steps, interaction_mode=None)
    newest = table.latest()
    grounded = results[0]["action_protocol"]
    assert "frame_reference" not in grounded and "rejection" not in grounded
    assert results[0]["action"].coordinate_transform == table.locate(1)[0].geometry
    # The frame field is not part of the turn-based action format.
    assert results[1]["action_protocol"]["protocol_failure"] == "invalid_action_output"
    assert results[1]["action"].coordinate_transform == newest.geometry
