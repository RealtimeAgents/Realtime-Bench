"""Under streaming each screenshot carries its frame number in its top-left corner, drawn over
the SoM marks; turn-based screenshots are the SoM images, unchanged."""

from __future__ import annotations

import base64
import hashlib
import io
import json

from PIL import Image, ImageChops

from realtime_framework.sandbox.observation.som import render_frame_number
from tests.replay.scenario import replies
from tests.replay.steps import Step
from tests.replay.stub import ChatCompletionsStub

from .episode import decide, open_episode

STEPS = [Step('{"action": "wait"}', status="wait")] * 3


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


def _sent_images(stub) -> list[bytes]:
    """The screenshot of each request's newest frame, as the model received it."""
    images = []
    for received in stub.requests:
        content = json.loads(received.body)["messages"][-1]["content"]
        (url,) = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
        images.append(base64.b64decode(url.partition(",")[2]))
    return images


def _run(tmp_path, interaction_mode):
    with ChatCompletionsStub(replies(STEPS)) as stub:
        episode = open_episode(stub, tmp_path, interaction_mode=interaction_mode)
        try:
            proofs = [decide(episode, k, "wait")[1]["inspection"] for k in range(len(STEPS))]
        finally:
            episode.finish()
    return _sent_images(stub), proofs


def test_turn_based_screenshots_are_the_som_images(tmp_path):
    images, proofs = _run(tmp_path, None)
    for image, proof in zip(images, proofs, strict=True):
        assert hashlib.sha256(image).hexdigest() == proof["marked_image_sha256"]


def test_streaming_screenshots_carry_their_frame_number(tmp_path):
    turn_based, _ = _run(tmp_path / "turn_based", None)
    streaming, proofs = _run(tmp_path / "streaming", "streaming")
    for frame, (som_image, image, proof) in enumerate(
        zip(turn_based, streaming, proofs, strict=True), 1
    ):
        assert hashlib.sha256(som_image).hexdigest() == proof["marked_image_sha256"]
        assert image == render_frame_number(som_image, frame)
