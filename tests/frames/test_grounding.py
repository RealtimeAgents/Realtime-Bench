"""A streaming action is interpreted against the frame it names: that frame's marks and geometry,
however many frames came after it. An action whose frame is missing, not an integer or not in the
frame table is refused, never executed. Turn-based actions keep their format and the newest frame.
A model-image point becomes the screen pixel nearest it, half to even, kept on the screen."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from realtime_framework.entity.actions import FrameGeometry, Observation, normalize_action
from realtime_framework.harnesses.default.frames import (
    FRAME_NOT_IN_TABLE,
    GROUNDING_FAILED,
    INVALID_FRAME_SYNTAX,
    INVALID_GROUNDING_FRAME,
    MISSING_FRAME,
)
from tests.fake_clock import FakeClock
from tests.replay import frames as replay_frames
from tests.replay.scenario import replies
from tests.replay.steps import Step
from tests.replay.stub import ChatCompletionsStub
from tests.streaming.fake_live import (
    BUDGET,
    REALTIME,
    Call,
    RecordingToolRuntime,
    Reply,
    Text,
    TextEnd,
    TextLiveDriver,
    make_agent,
    observation,
)
from tests.streaming.test_agent import ORIGIN, _two_frames

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


@pytest.fixture
def clock():
    clock = FakeClock(wall_origin=1000.0)
    with clock.participant():
        yield clock


class _MovingMarks(RecordingToolRuntime):
    """P3 120 px further right in every frame; an action's references resolve against the marks
    of the frame it names (harnesses/default/harness.py parse_grounded_action)."""

    def inspect(self, observation, **fields) -> dict:
        marks = {"P3": {"kind": "point", "point": [100 + 120 * observation.tick_number, 200]}}
        return {**super().inspect(observation, **fields), "marks": marks}


def test_an_action_uses_the_marks_and_geometry_of_the_frame_it_names(monkeypatch, clock):
    agent = make_agent(monkeypatch, clock)
    agent.bind_tool_runtime(_MovingMarks())
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    agent.push(Observation(first_frame_taller(1), clock.time(), 1, {"observation_id": "obs-1"}))
    clock.sleep(1.0)
    newest = observation(clock, 2)
    agent.push(newest)
    by_mark = Call(0.25, "c1", "click", {"target": "P3", "frame": 1})
    by_coordinates = Call(0.5, "c2", "click", {"x": 300, "y": 200, "frame": 1})
    agent.driver.replies.append(Reply((by_mark, by_coordinates), ends_at=1.0))
    agent.generate(0, 0, newest, BUDGET)
    agent.accept(0)
    agent.close()
    first, second = (agent.frames.locate(n)[0] for n in (1, 2))
    assert first.marks["P3"]["point"] != second.marks["P3"]["point"]
    assert first.geometry != second.geometry

    clicked, placed = (arrival.action for arrival in posted)
    assert [clicked.parameters["x"], clicked.parameters["y"]] == first.marks["P3"]["point"]
    assert clicked.coordinate_transform == placed.coordinate_transform == first.geometry
    desktop, _ = normalize_action(placed)
    assert (desktop.parameters["x"], desktop.parameters["y"]) == (750, 375)  # scale 2.5 x 1.875


def test_a_model_point_maps_to_a_screen_pixel_rounded_half_to_even_and_clamped():
    """FrameGeometry.image_to_screen_point on a 768x432 model image of the 1920x1080 desktop
    (scale 2.5): a point is rounded with Python's round, half to even, and kept on the screen.
    The image's own far edge (768, 432) and anything past it land on the last pixel,
    (1919, 1079), not one past it; a negative coordinate lands on 0."""
    geometry = FrameGeometry((768, 432), (1920, 1080))
    assert geometry.image_to_screen_point(0, 0) == (0, 0)
    assert geometry.image_to_screen_point(1, 3) == (2, 8)  # 2.5 and 7.5, half to even
    assert geometry.image_to_screen_point(0.3, 0.3) == (1, 1)  # 0.75 rounds up
    assert geometry.image_to_screen_point(767, 431) == (1918, 1078)  # 1917.5 and 1077.5
    assert geometry.image_to_screen_point(768, 432) == (1919, 1079)
    assert geometry.image_to_screen_point(5000, -40) == (1919, 0)


@pytest.mark.parametrize(
    ("frame", "error"),
    [
        (None, MISSING_FRAME),
        ('"one"', INVALID_FRAME_SYNTAX),
        ("1.0", INVALID_FRAME_SYNTAX),
        ("9", FRAME_NOT_IN_TABLE),
        ("0", FRAME_NOT_IN_TABLE),
    ],
)
def test_an_action_without_a_usable_frame_is_refused(monkeypatch, clock, frame, error):
    """In text, where the frame field is not checked before the action is parsed."""
    agent = make_agent(monkeypatch, clock, TextLiveDriver, REALTIME)
    posted = []
    agent.bind_inbox(posted.append, ORIGIN)
    newest = _two_frames(agent, clock)
    named = "" if frame is None else f', "frame": {frame}'
    action = '{"action": "click", "target": "P3"' + named + "}"
    agent.driver.replies.append(Reply((Text(0.25, action), TextEnd(0.5)), 1.0))
    result = agent.generate(0, 0, newest, BUDGET).result
    agent.close()
    [arrival] = posted
    assert (arrival.action, arrival.rejection, arrival.detail) == (
        None,
        INVALID_GROUNDING_FRAME,
        error,
    )
    assert result["action_protocol"]["protocol_failure"] is None


def _request_texts(received) -> str:
    texts = []
    for message in json.loads(received.body)["messages"]:
        content = message["content"]
        if isinstance(content, str):
            texts.append(content)
        else:
            texts += [item["text"] for item in content if item["type"] == "text"]
    return "\n".join(texts)


def test_an_action_naming_a_mark_its_frame_lacks_is_a_grounding_failure(tmp_path):
    """Its format is right, so the model is told that the mark is not in that screenshot, not
    that its response held no valid action; WAIT runs, and the record keeps what it asked for."""
    steps = [
        Step('{"action": "click", "target": "P9"}', status="wait"),
        Step('{"action": "wait"}', status="wait"),
    ]
    with ChatCompletionsStub(replies(steps)) as stub:
        episode = open_episode(stub, tmp_path)
        try:
            results = [decide(episode, k, step.status)[1] for k, step in enumerate(steps)]
        finally:
            episode.finish()
    protocol = results[0]["action_protocol"]
    assert protocol["protocol_failure"] == GROUNDING_FAILED
    assert protocol["grounding"]["resolutions"] == [
        {"error": "mark_not_in_observation", "mark": "P9", "reference": "P9"}
    ]
    assert results[0]["action"].action_type.value == "wait"
    entry = episode.agent.history.actions[0]
    assert (entry["requested"], entry["executed"]) == ("click {'target': 'P9'}", "wait()")
    assert (
        "EXECUTION RESULT: actual=wait(); requested=click {'target': 'P9'}; status=wait. "
        "NOTE: your previous action names P9, which is not a mark of the screenshot it was based "
        "on, so WAIT was executed. (grounding_failed)"
    ) in _request_texts(stub.requests[1])


def test_a_turn_based_inspection_row_keeps_the_last_resolution_of_its_output(tmp_path):
    """The tool trace row of a decision's inspection keeps the last resolution its output made,
    while its protocol keeps the first naming a mark the frame lacks, as the action asked for: of
    P9 then P8, the row keeps P8 and the protocol P9. Of two clicks the last, P2, is both; output
    that parsed to no action made none."""
    missing = '{"action": "click", "target": "%s"}'
    steps = [
        Step(missing % "P9", status="wait"),
        Step(f"{missing % 'P9'} {missing % 'P8'}", status="wait"),
        Step(f"{missing % 'P1'} {missing % 'P2'}", status="wait"),
        Step("no action", status="wait"),
    ]
    with ChatCompletionsStub(replies(steps)) as stub:
        episode = open_episode(stub, tmp_path)
        try:
            results = [decide(episode, k, step.status)[1] for k, step in enumerate(steps)]
        finally:
            episode.finish()
    rows = [json.loads(line) for line in (tmp_path / "tool_calls.jsonl").read_text().splitlines()]
    resolutions = [row["reference_resolution"] for row in rows]
    grounding = [result["action_protocol"]["grounding"] for result in results]

    def unknown(mark):
        failure = {"error": "mark_not_in_observation", "mark": mark, "reference": mark}
        return {"mode": "som", "valid": False, "resolutions": [failure]}

    assert resolutions[:2] == [unknown("P9"), unknown("P8")]
    assert grounding[:2] == [unknown("P9"), unknown("P9")]
    assert resolutions[2] == grounding[2]
    assert resolutions[2]["resolutions"][0]["mark"] == "P2"
    assert resolutions[3] is grounding[3] is None


def test_turn_based_actions_name_no_frame(tmp_path):
    steps = [
        Step('{"action": "click", "target": "P3"}'),
        Step('{"action": "click", "target": "P3", "frame": 1}', status="wait"),
    ]
    with ChatCompletionsStub(replies(steps)) as stub:
        episode = open_episode(stub, tmp_path, scene=SCENE)
        try:
            results = [decide(episode, k, step.status)[1] for k, step in enumerate(steps)]
        finally:
            episode.finish()
    frames = episode.agent.frames._frames
    grounded = results[0]["action_protocol"]
    assert "frame_reference" not in grounded and "rejection" not in grounded
    assert results[0]["action"].coordinate_transform == frames[1].geometry
    # The frame field is not part of the turn-based action format.
    assert results[1]["action_protocol"]["protocol_failure"] == "invalid_action_output"
    assert results[1]["action"].coordinate_transform == frames[2].geometry
