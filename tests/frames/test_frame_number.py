"""Under streaming each screenshot carries its frame number in its top-left corner (the frames the
native-streaming agent pushes: tests/streaming/test_streaming_e2e.py), and the SoM marks keep
clear of it; turn-based screenshots are the SoM images, unchanged."""

from __future__ import annotations

import base64
import hashlib
import io
import json

import pytest
from PIL import Image, ImageChops

from realtime_framework.sandbox.observation import som
from realtime_framework.sandbox.observation.som import render_frame_number
from tests.fake_clock import FakeClock
from tests.replay import frames as replay_frames
from tests.replay.frames import FrameSource, ReplayTask
from tests.replay.scenario import replies
from tests.replay.steps import Step
from tests.replay.stub import ChatCompletionsStub
from tests.streaming.fake_live import make_agent

from .episode import decide, open_episode

STEPS = [Step('{"action": "wait"}', status="wait")] * 3


@pytest.fixture
def clock():
    clock = FakeClock(wall_origin=1000.0)
    with clock.participant():
        yield clock


def _png(size=(768, 432), color=(200, 210, 220)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _changed_box(before: bytes, after: bytes):
    with Image.open(io.BytesIO(before)) as a, Image.open(io.BytesIO(after)) as b:
        assert a.size == b.size
        return ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox()


def test_the_label_is_drawn_in_the_top_left_corner_only():
    source = _png()
    labelled = render_frame_number(source, 7)
    assert labelled[:8] == b"\x89PNG\r\n\x1a\n"
    left, top, right, bottom = _changed_box(source, labelled)
    assert (left, top) == (2, 2)
    assert right < 768 // 4 and bottom < 432 // 8


def test_each_frame_number_draws_a_different_label():
    source = _png()
    labels = {render_frame_number(source, frame) for frame in (1, 2, 12, 123)}
    assert len(labels) == 4


def _corner_probe(frame: int, *, text_observation: bool = False) -> dict:
    """The replay form with its Name field (mark P1) at the left, just below the browser toolbar,
    where the first place of its badge is under the frame number."""
    value = replay_frames.probe(frame, text_observation=text_observation)
    (name,) = [item for item in value["actionable_now"] if item["id"] == "dom-0001"]
    name["bbox"] = [10, 90, 100, 120]
    name["click_point"] = [55, 105]
    return value


def test_the_mark_badges_keep_clear_of_the_frame_number(monkeypatch, clock):
    badges = {}
    draw_badge = som._draw_badge

    def recording(draw, box, text, font, color):
        badges[text] = box
        draw_badge(draw, box, text, font, color)

    monkeypatch.setattr(som, "_draw_badge", recording)
    agent = make_agent(monkeypatch, clock)
    agent.bind_tool_runtime(som.ToolRuntime())
    source = FrameSource(ReplayTask(), text_observation=False, fps=1.0, probe=_corner_probe)
    agent.push(source.capture(0))
    agent.close()
    frame = badges.pop("FRAME 1")
    assert sorted(badges) == ["P1", "P2", "P3", "R1", "U1"]
    for box in badges.values():
        assert box[2] <= frame[0] or box[0] >= frame[2] or box[3] <= frame[1] or box[1] >= frame[3]


def _sent_images(stub) -> list[bytes]:
    """The screenshot of each request's newest frame, as the model received it."""
    images = []
    for received in stub.requests:
        content = json.loads(received.body)["messages"][-1]["content"]
        (url,) = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
        images.append(base64.b64decode(url.partition(",")[2]))
    return images


def test_turn_based_screenshots_are_the_som_images(tmp_path):
    with ChatCompletionsStub(replies(STEPS)) as stub:
        episode = open_episode(stub, tmp_path)
        try:
            proofs = [decide(episode, k, "wait")[1]["inspection"] for k in range(len(STEPS))]
        finally:
            episode.finish()
    for image, proof in zip(_sent_images(stub), proofs, strict=True):
        assert hashlib.sha256(image).hexdigest() == proof["marked_image_sha256"]
