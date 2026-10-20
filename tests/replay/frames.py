"""Deterministic synthetic frames, captured through the runner's own BrowserSceneCapture.

The runner builds every observation with BrowserSceneCapture.capture (evaluation/runner.py:1606,
1630; sandbox/actions.py:190-203). Here that class runs unchanged with its probe override
(sandbox/observation/capture.py:100, 144-145), a guest that returns synthetic PNGs, and a fixed
clock in place of the time.time() that stamps each capture (capture.py:267). Observation ids,
screenshot hashes, frame geometry and the canonical interactables therefore come from
production code, built from inputs that are a pure function of the frame index.

The probe returns what the CDP extractor returns for a screenshot model (scene, binding,
controls, canvases, metrics: sandbox/browser/cdp_interactables.py:424-440, 470-484, 569-583;
window assembly: sandbox/browser/cdp.py:244-283, 357-366). With ``text_observation`` it also
returns what a text-observation (choice) capture adds: the rendered field state of every DOM
control (cdp_interactables.py:389-423) and the accessibility tree, read by the production
capture_ax_frames from synthetic CDP nodes (cdp.py:262-270). Without it the output is unchanged.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import types
from unittest import mock

from PIL import Image, ImageDraw

from realtime_framework.sandbox.browser.ax import capture_ax_frames
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


# The Name field's state, seen only by text-observation models. The screenshot does not draw it,
# so the screenshot-model baselines do not depend on it. It follows the Decisions scenario
# (scenario_decisions.py): a click on the field (decision 0, frame 1), composed text (frame 2),
# Tab (frame 3), a click (frame 14), select all (frame 15), Backspace (frame 17), Tab (frame 18),
# a click (frame 39) and a typed suggestion (frame 40). Decision k sees frame k + 1.
NAME_PLACEHOLDER = "Full name"
_FIRST_NAMES = (
    "Ada", "Alan", "Barbara", "Claude", "Donald", "Edsger", "Frances", "Grace",
    "Hedy", "John", "Katherine", "Linus", "Margaret", "Niklaus", "Radia", "Tim",
)  # fmt: skip
_LAST_NAMES = (
    "Allen", "Babbage", "Dijkstra", "Hamilton", "Hopper", "Johnson", "Kay", "Knuth",
    "Lamarr", "Liskov", "Lovelace", "Perlman", "Shannon", "Turing", "Wirth",
)  # fmt: skip
# The field's datalist while the page offers suggestions: 240 names, enough for a candidate set
# above one Choice question's 255 options (providers/openai_decisions.py:56).
NAME_SUGGESTIONS = tuple(f"{first} {last}" for first in _FIRST_NAMES for last in _LAST_NAMES)


def name_focused(frame: int) -> bool:
    return frame in (2, 3) or 15 <= frame <= 18 or frame in (40, 41)


def name_value(frame: int) -> str:
    return "Ada Lovelace" if 3 <= frame <= 17 or frame >= 41 else ""


def name_text_selected(frame: int) -> bool:
    return frame in (16, 17)


def name_suggestions(frame: int) -> tuple[str, ...]:
    return NAME_SUGGESTIONS if frame == 40 else ()


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


def _rgb(color: tuple[int, int, int]) -> str:
    return f"rgb({color[0]}, {color[1]}, {color[2]})"


def _text_observation(control: dict, frame: int, background: tuple[int, int, int]) -> dict:
    """The fields the extractor adds to each DOM control for a text-observation model
    (cdp_interactables.py:389-423). Inputs render no inner text; a button shows its label. Only
    the Name textbox is an editable field, so only it has input constraints and a value."""
    editable = control["role"] == "textbox"  # editableField (cdp_interactables.py:392)
    textbox = {"border": f"2px solid {_rgb((120, 120, 120))}", "foreground": _rgb((0, 0, 0))}
    button = {"border": f"0px none {_rgb((255, 255, 255))}", "foreground": _rgb((255, 255, 255))}
    style = button if control["tag"] == "button" else textbox
    return {
        "visible_content": {
            "source": "rendered_dom_and_accessibility",
            "text": control["label"] if control["tag"] == "button" else "",
            "image_labels": [],
            "background_color": _rgb(background),
            "foreground_color": style["foreground"],
            "border": style["border"],
        },
        "select_options": [],
        "select_multiple": False,
        "input_constraints": (
            {
                "placeholder": NAME_PLACEHOLDER,
                "pattern": "",
                "min": None,
                "max": None,
                "step": None,
                "maxlength": None,
                "options": [{"value": name, "label": name} for name in name_suggestions(frame)],
            }
            if editable
            else {}
        ),
        "focused": editable and name_focused(frame),
        "value": name_value(frame) if editable else "",
        "text_selected": editable and name_text_selected(frame),
    }


AX_FRAME_ID = "replay-main-frame"


class _AxSession:
    """Answers the CDP calls capture_ax_frames makes for a page without child frames
    (sandbox/browser/ax.py:16-20, 25, 77, 101-105)."""

    def __init__(self, nodes: list[dict]):
        self._results = {
            "Page.getFrameTree": {"frameTree": {"frame": {"id": AX_FRAME_ID}}},
            "DOM.getDocument": {"root": {"nodeName": "#document", "children": []}},
            "Accessibility.getFullAXTree": {"nodes": nodes},
        }

    def transaction(self):
        return contextlib.nullcontext()

    def call(self, method, params=None, *, timeout, session_id=None):
        return {"result": self._results[method]}


def ax_node(node_id: int, role: str, name: str = "", *, value=None, properties=(), children=()):
    """One Accessibility.getFullAXTree node in the shape ax.py reads (ax.py:123-135, 145-197)."""
    node = {
        "nodeId": str(node_id),
        "ignored": False,
        "role": {"type": "role", "value": role},
        "childIds": [str(child) for child in children],
    }
    if name:
        node["name"] = {"type": "computedString", "value": name}
    if value is not None:
        node["value"] = {"type": "string", "value": value}
    if properties:
        node["properties"] = [{"name": key, "value": prop} for key, prop in properties]
    return node


def ax_fields(nodes: list[dict]) -> dict:
    """ax_tree, ax_text and ax_capture, as the capture adds them to a window (cdp.py:262-270)."""
    return capture_ax_frames(_AxSession(nodes))


def _form_ax_nodes(frame: int) -> list[dict]:
    focused = (("focused", {"type": "booleanOrUndefined", "value": True}),)
    checked = {"type": "tristate", "value": "true" if agree_checked(frame) else "false"}
    disabled = (("disabled", {"type": "boolean", "value": True}),)
    return [
        ax_node(1, "RootWebArea", TITLE, children=(2, 3, 4, 5, 6, 7)),
        ax_node(2, "StaticText", "Name"),
        ax_node(
            3,
            "textbox",
            "Name",
            value=name_value(frame),
            properties=focused if name_focused(frame) else (),
        ),
        ax_node(4, "checkbox", "I agree to the terms", properties=(("checked", checked),)),
        ax_node(5, "button", "Submit"),
        ax_node(6, "button", "Next", properties=() if next_enabled(frame) else disabled),
        ax_node(7, "Canvas", "Route map"),
    ]


def probe(frame: int, *, text_observation: bool = False) -> dict:
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
    if text_observation:
        fills = {  # the backgrounds screenshot() draws
            "dom-0001": (255, 255, 255),
            "dom-0002": (30, 110, 220) if agree_checked(frame) else (255, 255, 255),
            "dom-0003": (30, 110, 220),
            "dom-0004": (40, 160, 80) if next_enabled(frame) else (190, 190, 190),
        }
        for control in (name, agree, submit, following):
            control.update(_text_observation(control, frame, fills[control["id"]]))
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
    if text_observation:
        window.update(ax_fields(_form_ax_nodes(frame)))
    return {**window, "binding": dict(window["binding"]), "windows": [window]}


class _Guest:
    def __init__(self, source: FrameSource):
        self._source = source

    def screenshot(self) -> bytes:
        return self._source.screenshot(self._source.frame)


class FrameSource:
    """Each capture() returns the next synthetic frame; frame 0 is the runner's initial capture.

    ``screenshot`` and ``probe`` give another page (scenario_decisions.py has a game page). The
    capture reads the accessibility tree only in its first probe (capture.py:253, 268); this probe
    adds it to both, and of the second probe only the target id and fingerprint are read
    (capture.py:272-274).
    """

    def __init__(
        self, task, *, text_observation: bool, fps: float, screenshot=screenshot, probe=probe
    ):
        self.frame = -1
        self._fps = fps
        self.screenshot = screenshot
        self._capture = BrowserSceneCapture(
            _Guest(self),
            task,
            probe=lambda payload: probe(self.frame, text_observation=text_observation),
            desktop_size=DESKTOP_SIZE,
            text_observation=text_observation,
        )

    def capture(self, tick: float, at: float | None = None):
        """``at`` is the capture time; by default frame n is stamped CLOCK_ORIGIN + n/fps."""
        self.frame += 1
        stamp = CLOCK_ORIGIN + self.frame / self._fps if at is None else at
        clock = types.SimpleNamespace(time=lambda: stamp)
        with mock.patch.object(capture_module, "time", clock):
            observation = self._capture.capture(tick)
        # GuestEnvironment.capture adds its measured capture time (sandbox/actions.py:203).
        observation.metadata["capture_latency_s"] = 0.0
        return observation
