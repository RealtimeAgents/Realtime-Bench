"""Deterministic synthetic frames, captured through the runner's own BrowserSceneCapture.

The runner builds every observation with BrowserSceneCapture.capture (evaluation/runner.py:1606,
1630; sandbox/actions.py:190-203). Here that class runs unchanged with its probe override
(sandbox/observation/capture.py:100, 144-145), a guest that returns synthetic PNGs, and a fixed
clock in place of the time.time() that stamps each capture (capture.py:267). Observation ids,
screenshot hashes, frame geometry and the canonical interactables therefore come from
production code, built from inputs that are a pure function of the frame index.

The probe returns what the CDP extractor returns for a screenshot model (scene, binding,
controls, canvases, metrics: sandbox/browser/cdp_interactables.py:424-440, 470-484, 569-583;
window assembly: sandbox/browser/cdp.py:244-283, 357-366). Text-observation (choice) models
also receive accessibility and field-state fields, which this probe does not produce yet.
"""

from __future__ import annotations

import hashlib
import io
import json
import types
from unittest import mock

from PIL import Image, ImageDraw

from realtime_framework.sandbox.observation import capture as capture_module
from realtime_framework.sandbox.observation.capture import BrowserSceneCapture

DESKTOP_SIZE = (1920, 1080)
CLOCK_ORIGIN = 1_760_000_000.0  # wall-clock time of frame 0; frame n is n/fps seconds later

URL = "http://localhost:18080/"
TITLE = "Replay form"
TARGET_ID = "replay-target"
WINDOW_ID = "replay-window"

NAME_BOX = (700, 300, 1220, 350)
AGREE_BOX = (700, 380, 724, 404)
SUBMIT_BOX = (700, 450, 900, 510)
NEXT_BOX = (940, 450, 1140, 510)
MAP_BOX = (100, 300, 600, 800)


class ReplayTask:
    """The task attributes the runner and BrowserSceneCapture read for a web task."""

    task_id = "replay-form"
    category = "realtime"


def agree_checked(frame: int) -> bool:
    return frame > 10


def next_enabled(frame: int) -> bool:
    """The Next button turns from unavailable (U1) into an actionable mark (P4)."""
    return frame > 30


def screenshot(frame: int) -> bytes:
    image = Image.new("RGB", DESKTOP_SIZE, (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1919, 79), fill=(60, 60, 60))  # browser toolbar, outside the page
    draw.rectangle(NAME_BOX, fill=(255, 255, 255), outline=(120, 120, 120), width=2)
    agree_fill = (30, 110, 220) if agree_checked(frame) else (255, 255, 255)
    draw.rectangle(AGREE_BOX, fill=agree_fill, outline=(120, 120, 120), width=2)
    draw.rectangle(SUBMIT_BOX, fill=(30, 110, 220))
    draw.rectangle(NEXT_BOX, fill=(40, 160, 80) if next_enabled(frame) else (190, 190, 190))
    draw.rectangle(MAP_BOX, fill=(220, 235, 220))
    x, y = 120 + (37 * frame) % 460, 320 + (53 * frame) % 460
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(200, 60, 40))  # a marker moving on the map
    draw.rectangle((100, 1000, 100 + 17 * (frame % 100), 1030), fill=(90, 90, 200))  # progress
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _control(
    item_id,
    order,
    role,
    tag,
    label,
    box,
    actions,
    *,
    input_type="",
    enabled=True,
    editable=False,
    checked=None,
    unavailable_reasons=(),
) -> dict:
    """One DOM control as the extractor reports it (cdp_interactables.py:424-440)."""
    return {
        "id": item_id,
        "dom_order": order,
        "source": "dom",
        "frame_path": "",
        "role": role,
        "label": label,
        "context": "",
        "semantic_label": label,
        "tag": tag,
        "input_type": input_type,
        "bbox": list(box),
        "click_point": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2],
        "actions": list(actions),
        "enabled": enabled,
        "readonly": False,
        "editable": editable,
        "checked": checked,
        "selected": None,
        "expanded": None,
        "modal_scope": False,
        "occluded": False,
        "unavailable_reasons": list(unavailable_reasons),
    }


def probe(frame: int) -> dict:
    name = _control(
        "dom-0001",
        0,
        "textbox",
        "input",
        "Name",
        NAME_BOX,
        ["click", "type"],
        input_type="text",
        editable=True,
        checked=False,
    )
    agree = _control(
        "dom-0002",
        1,
        "checkbox",
        "input",
        "I agree to the terms",
        AGREE_BOX,
        ["click"],
        input_type="checkbox",
        checked=agree_checked(frame),
    )
    submit = _control("dom-0003", 2, "button", "button", "Submit", SUBMIT_BOX, ["click"])
    if next_enabled(frame):
        following = _control("dom-0004", 3, "button", "button", "Next", NEXT_BOX, ["click"])
    else:
        following = _control(
            "dom-0004",
            3,
            "button",
            "button",
            "Next",
            NEXT_BOX,
            [],
            enabled=False,
            unavailable_reasons=["disabled"],
        )
    route_map = {
        "id": "canvas-1",
        "dom_order": 4,
        "source": "dom",
        "frame_path": "",
        "role": "canvas",
        "label": "Route map",
        "context": "",
        "semantic_label": "Route map",
        "tag": "canvas",
        "bbox": list(MAP_BOX),
        "click_point": [350, 550],
        "css_size": [500, 500],
        "backing_size": [500, 500],
        "device_pixel_ratio": 1,
        "actions": ["click"],
        "enabled": True,
        "occluded": False,
        "unavailable_reasons": [],
        # The spec default for a task without declared keys (interactables.py:868-869).
        "internal_target_coverage": "unsupported",
        "interaction": "",
        "internal_target_count": 0,
    }
    actionable = [name, agree, submit] + ([following] if following["enabled"] else [])
    unavailable = [] if following["enabled"] else [following]
    readiness = {"document_ready_state": "complete", "selector_present": False, "selector_text": ""}
    payload = json.dumps(
        {
            "actionable_now": actionable,
            "visible_but_unavailable": unavailable,
            "canvas_surfaces": [route_map],
            "readiness": readiness,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    window = {
        "scene": {
            "status": "ok",
            "kind": "browser_dom",
            "application": "google-chrome",
            "url": URL,
            "title": TITLE,
            "active": True,
            "visible": True,
            "target_id": TARGET_ID,
        },
        # Equal fingerprints before and after the screenshot make the binding stable
        # (capture.py:272-276).
        "binding": {
            "stable": True,
            "revision": "",
            "fingerprint": hashlib.sha256(payload.encode()).hexdigest(),
        },
        "actionable_now": actionable,
        "visible_but_unavailable": unavailable,
        "canvas_surfaces": [route_map],
        "limitations": ["browser_chrome_not_inspected", "canvas_internal_targets_unsupported"],
        "readiness": readiness,
        "metrics": {
            "screenX": 0,
            "screenY": 0,
            "outerWidth": 1920,
            "outerHeight": 1080,
            "innerWidth": 1920,
            "innerHeight": 1000,
            "devicePixelRatio": 1,
            "screenWidth": 1920,
            "screenHeight": 1080,
            "contentLeft": 0,
            "contentTop": 80,
            "inspectedFrameDocuments": 1,
            "inaccessibleFrames": 0,
        },
        "window_id": WINDOW_ID,
        "title": TITLE,
        "active": True,
    }
    return {**window, "binding": dict(window["binding"]), "windows": [window]}


class _Guest:
    def __init__(self, source: FrameSource):
        self._source = source

    def screenshot(self) -> bytes:
        return screenshot(self._source.frame)


class FrameSource:
    """Each capture() returns the next synthetic frame; frame 0 is the runner's initial capture."""

    def __init__(self, task, *, text_observation: bool, fps: float):
        self.frame = -1
        self._fps = fps
        self._capture = BrowserSceneCapture(
            _Guest(self),
            task,
            probe=lambda payload: probe(self.frame),
            desktop_size=DESKTOP_SIZE,
            text_observation=text_observation,
        )

    def capture(self, tick: float):
        self.frame += 1
        clock = types.SimpleNamespace(time=lambda: CLOCK_ORIGIN + self.frame / self._fps)
        with mock.patch.object(capture_module, "time", clock):
            observation = self._capture.capture(tick)
        # GuestEnvironment.capture adds its measured capture time (sandbox/actions.py:203).
        observation.metadata["capture_latency_s"] = 0.0
        return observation
