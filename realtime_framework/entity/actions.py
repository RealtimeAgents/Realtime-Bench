from __future__ import annotations

# Actions
"""Canonical observation and action contracts shared by the entire runtime."""
import math as math
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from enum import Enum as Enum
from itertools import product
from typing import Optional as Optional

# A single action is deliberately bounded: long holds make the synchronous guest
# endpoint unresponsive and can leave a game key stuck if the caller times out.
MAX_ACTION_DURATION = 10.0


class ActionType(str, Enum):
    CLICK = "click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    TRIPLE_CLICK = "triple_click"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    TYPE = "type"
    KEY = "key"
    HOLD_KEY = "hold_key"
    SCROLL = "scroll"
    DRAG = "drag"
    MOVE = "move"
    WAIT = "wait"
    DONE = "done"
    FAIL = "fail"


@dataclass
class Action:
    action_type: ActionType
    parameters: dict = field(default_factory=dict)
    # Harness metadata. Adapters state the coordinates they emitted; only the guest execution
    # boundary converts them to desktop pixels. Non-pointer actions safely keep the defaults.
    coordinate_space: str = "desktop_px"
    coordinate_transform: Optional[dict] = None

    @classmethod
    def wait(cls, duration: float | None = None) -> "Action":
        parameters = {} if duration is None else {"duration": duration}
        return cls(action_type=ActionType.WAIT, parameters=parameters)

    @classmethod
    def click(cls, x: int, y: int, button: str = "left") -> "Action":
        return cls(action_type=ActionType.CLICK, parameters={"x": x, "y": y, "button": button})

    @classmethod
    def hold_key(cls, key: str, duration: float = 1.0) -> "Action":
        return cls(
            action_type=ActionType.HOLD_KEY,
            parameters={"key": key, "duration": duration},
        )


def validate_action(action: Optional["Action"]) -> tuple[bool, str]:
    """Validate the action schema and return ``(ok, reason)``.

    Invalid verbs or parameters are reported to the caller as parse errors,
    so the model receives feedback instead of silently performing a wait.
    The schema matches the guest action dispatcher."""
    if action is None:
        return False, "no action"
    t = action.action_type
    if action.parameters is None:
        p = {}
    elif not isinstance(action.parameters, dict):
        return False, "action parameters must be an object"
    else:
        p = action.parameters
    allowed_buttons = {"left", "middle", "right"}
    allowed_modifiers = {
        "shift",
        "ctrl",
        "control",
        "alt",
        "super",
        "meta",
        "cmd",
        "command",
        "win",
    }

    allowed_parameters = {
        ActionType.CLICK: {"x", "y", "button", "modifier"},
        ActionType.RIGHT_CLICK: {"x", "y", "modifier"},
        ActionType.DOUBLE_CLICK: {"x", "y", "modifier"},
        ActionType.TRIPLE_CLICK: {"x", "y", "modifier"},
        ActionType.MOUSE_DOWN: {"x", "y", "button", "duration"},
        ActionType.MOUSE_UP: {"x", "y", "button", "duration"},
        ActionType.MOVE: {"x", "y", "duration"},
        ActionType.TYPE: {"text"},
        ActionType.KEY: {"key", "keys", "repeat"},
        ActionType.HOLD_KEY: {"key", "keys", "duration"},
        ActionType.SCROLL: {"x", "y", "dx", "dy", "modifier"},
        ActionType.DRAG: {"x1", "y1", "x2", "y2", "duration", "button", "modifier"},
        ActionType.WAIT: {"duration"},
        ActionType.DONE: set(),
        ActionType.FAIL: set(),
    }
    if t not in allowed_parameters:
        return False, f"unsupported action type: {t!r}"
    unexpected = set(p) - allowed_parameters[t]
    if unexpected:
        return False, f"{t.value} received unexpected parameters: {', '.join(sorted(unexpected))}"

    def _num(k) -> bool:
        v = p.get(k)
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    def _valid_duration() -> tuple[bool, str]:
        if "duration" not in p:
            return True, ""
        if not _num("duration"):
            return False, "duration must be a finite number"
        if not 0 <= p["duration"] <= MAX_ACTION_DURATION:
            return False, f"duration must be between 0 and {MAX_ACTION_DURATION:g} seconds"
        return True, ""

    def _valid_key() -> bool:
        key = p.get("key")
        keys = p.get("keys")
        if key is not None and keys is not None:
            return False
        raw_items = [key] if key is not None else keys
        if not isinstance(raw_items, list) or not raw_items:
            return False
        for item in raw_items:
            if not isinstance(item, str) or not item.strip():
                return False
            if any(not part.strip() for part in item.split("+")):
                return False
        return True

    if t in (
        ActionType.CLICK,
        ActionType.RIGHT_CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.TRIPLE_CLICK,
    ):
        has_pointer = "x" in p or "y" in p
        if has_pointer and not (_num("x") and _num("y")):
            return False, f"{t.value} x and y must both be numeric when provided"
    elif t in (ActionType.MOUSE_DOWN, ActionType.MOUSE_UP):
        has_pointer = "x" in p or "y" in p
        if has_pointer and not (_num("x") and _num("y")):
            return False, f"{t.value} x and y must both be numeric when provided"
        ok, reason = _valid_duration()
        if not ok:
            return False, reason
    elif t == ActionType.MOVE:
        if not (_num("x") and _num("y")):
            return False, "move requires numeric x and y"
        ok, reason = _valid_duration()
        if not ok:
            return False, reason
    elif t == ActionType.DRAG:
        if not (_num("x2") and _num("y2")):
            return False, "drag requires numeric x2 and y2"
        has_start = "x1" in p or "y1" in p
        if has_start and not (_num("x1") and _num("y1")):
            return False, "drag x1 and y1 must both be numeric when provided"
        ok, reason = _valid_duration()
        if not ok:
            return False, reason
    elif t == ActionType.TYPE:
        if not isinstance(p.get("text"), str):
            return False, "type requires string text"
    elif t in (ActionType.KEY, ActionType.HOLD_KEY):
        if not _valid_key():
            return False, f"{t.value} requires a non-empty key or keys list"
        ok, reason = _valid_duration()
        if not ok:
            return False, reason
        if "repeat" in p and (
            t != ActionType.KEY
            or not isinstance(p["repeat"], int)
            or isinstance(p["repeat"], bool)
            or not 1 <= p["repeat"] <= 100
        ):
            return False, "key repeat must be an integer between 1 and 100"
    elif t == ActionType.SCROLL:
        has_pointer = "x" in p or "y" in p
        if has_pointer and not (_num("x") and _num("y")):
            return False, "scroll x and y must both be numeric when provided"
        if not (_num("dx") and _num("dy")):
            return False, "scroll requires numeric dx and dy"
        if p["dx"] == 0 and p["dy"] == 0:
            return False, "scroll dx and dy cannot both be zero"
    elif t == ActionType.WAIT:
        ok, reason = _valid_duration()
        if not ok:
            return False, reason
    if "button" in p and (not isinstance(p["button"], str) or p["button"] not in allowed_buttons):
        return False, "button must be left, middle, or right"
    if "modifier" in p:
        modifier = p["modifier"]
        parts = (
            [part.strip().lower() for part in modifier.split("+")]
            if isinstance(modifier, str)
            else []
        )
        if not parts or any(not part or part not in allowed_modifiers for part in parts):
            return False, "modifier must be a '+'-joined combination of shift, ctrl, alt, or super"
    return True, ""


@dataclass
class Observation:
    screenshot: bytes
    timestamp: float
    tick_number: float
    metadata: dict = field(default_factory=dict)


# Geometry
"""Frame geometry and action-coordinate normalization contracts."""

import hashlib as hashlib
import io as io
import json as json
from dataclasses import dataclass as dataclass
from typing import Any as Any


DESKTOP_SIZE = (1920, 1080)


def _positive_pair(value: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            pair = (int(value[0]), int(value[1]))
            if pair[0] > 0 and pair[1] > 0:
                return pair
        except (TypeError, ValueError):
            pass
    return fallback


@dataclass(frozen=True)
class FrameGeometry:
    """Exact affine mapping between model image pixels and executor screen pixels."""

    image_size: tuple[int, int]
    screen_size: tuple[int, int] = DESKTOP_SIZE
    capture_bbox_screen: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        iw, ih = _positive_pair(self.image_size, DESKTOP_SIZE)
        sw, sh = _positive_pair(self.screen_size, DESKTOP_SIZE)
        bbox = self.capture_bbox_screen or (0.0, 0.0, float(sw), float(sh))
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("capture_bbox_screen must be a non-empty x1,y1,x2,y2 box")
        object.__setattr__(self, "image_size", (iw, ih))
        object.__setattr__(self, "screen_size", (sw, sh))
        object.__setattr__(self, "capture_bbox_screen", tuple(float(v) for v in bbox))

    @property
    def scale(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.capture_bbox_screen
        return ((x2 - x1) / self.image_size[0], (y2 - y1) / self.image_size[1])

    @property
    def transform_id(self) -> str:
        payload = json.dumps(self.to_dict(include_id=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def image_to_screen_point(self, x: float, y: float) -> tuple[int, int]:
        sx, sy = self.scale
        left, top, _, _ = self.capture_bbox_screen
        sw, sh = self.screen_size
        return (
            max(0, min(sw - 1, round(left + float(x) * sx))),
            max(0, min(sh - 1, round(top + float(y) * sy))),
        )

    def screen_to_image_point(self, x: float, y: float) -> tuple[int, int]:
        sx, sy = self.scale
        left, top, _, _ = self.capture_bbox_screen
        iw, ih = self.image_size
        return (
            max(0, min(iw - 1, round((float(x) - left) / sx))),
            max(0, min(ih - 1, round((float(y) - top) / sy))),
        )

    def screen_to_image_rect(self, rect: list[int] | tuple[int, ...]) -> list[int]:
        sx, sy = self.scale
        left, top, _, _ = self.capture_bbox_screen
        iw, ih = self.image_size
        return [
            max(0, min(iw, round((float(rect[0]) - left) / sx))),
            max(0, min(ih, round((float(rect[1]) - top) / sy))),
            max(0, min(iw, round((float(rect[2]) - left) / sx))),
            max(0, min(ih, round((float(rect[3]) - top) / sy))),
        ]

    def to_dict(self, *, include_id: bool = True) -> dict:
        sx, sy = self.scale
        left, top, _, _ = self.capture_bbox_screen
        value = {
            "image_size": list(self.image_size),
            "screen_size": list(self.screen_size),
            "capture_bbox_screen": [round(v, 6) for v in self.capture_bbox_screen],
            "coordinate_transform": {
                "from": "model_image_px",
                "to": "desktop_px",
                "scale_x": sx,
                "scale_y": sy,
                "offset_x": left,
                "offset_y": top,
            },
        }
        if include_id:
            value["transform_id"] = self.transform_id
        return value

    @classmethod
    def from_dict(cls, value: dict | None) -> FrameGeometry:
        value = value if isinstance(value, dict) else {}
        return cls(
            image_size=_positive_pair(value.get("image_size"), DESKTOP_SIZE),
            screen_size=_positive_pair(value.get("screen_size"), DESKTOP_SIZE),
            capture_bbox_screen=tuple(value.get("capture_bbox_screen") or ()) or None,
        )


@dataclass(frozen=True)
class PreparedFrame:
    data: bytes
    mime_subtype: str
    detail: str
    geometry: FrameGeometry


def bind_action(
    action: Action | None, geometry: FrameGeometry, *, coordinate_space: str = "model_image_px"
) -> Action | None:
    """Attach a provider's output coordinate contract without converting it yet."""
    if action is None or action.coordinate_space != "desktop_px":
        return action
    return Action(
        action.action_type,
        dict(action.parameters or {}),
        coordinate_space=coordinate_space,
        coordinate_transform=geometry.to_dict(),
    )


def normalize_action(action: Action) -> tuple[Action, dict]:
    """Convert an adapter-authored action into desktop pixels and return an audit trace."""
    source = dict(action.parameters or {})
    space = action.coordinate_space or "desktop_px"
    if space == "desktop_px":
        return action, {
            "source_space": "desktop_px",
            "source_parameters": source,
            "desktop_parameters": dict(source),
            "transform_id": None,
        }

    geometry = FrameGeometry.from_dict(action.coordinate_transform)
    params = dict(source)
    if space in {"model_image_px", "model_tool_px"}:
        for x_key, y_key in (("x", "y"), ("x1", "y1"), ("x2", "y2")):
            if _finite(params.get(x_key)) and _finite(params.get(y_key)):
                params[x_key], params[y_key] = geometry.image_to_screen_point(
                    params[x_key], params[y_key]
                )
    elif space in {"normalized_0_999", "normalized_0_1000"}:
        maximum = 999.0 if space.endswith("999") else 1000.0
        normalized_geometry = FrameGeometry(
            (round(maximum), round(maximum)), geometry.screen_size, geometry.capture_bbox_screen
        )
        for x_key, y_key in (("x", "y"), ("x1", "y1"), ("x2", "y2")):
            if _finite(params.get(x_key)) and _finite(params.get(y_key)):
                params[x_key], params[y_key] = normalized_geometry.image_to_screen_point(
                    params[x_key], params[y_key]
                )
    else:
        raise ValueError(f"unsupported action coordinate space: {space}")

    desktop = Action(action.action_type, params)
    return desktop, {
        "source_space": space,
        "source_parameters": source,
        "desktop_parameters": dict(params),
        "transform_id": geometry.transform_id,
    }


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_pair(value: str) -> tuple[int, int]:
    import argparse as argparse

    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT")
    return int(parts[0]), int(parts[1])


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    import argparse as argparse

    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected X1,Y1,X2,Y2")
    return tuple(float(item) for item in parts)


def main() -> None:
    """Print deterministic landmark mappings for a GUI coordinate contract."""
    import argparse as argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--image-size", type=_parse_pair, required=True)
    parser.add_argument("--screen-size", type=_parse_pair, default=DESKTOP_SIZE)
    parser.add_argument("--capture-bbox-screen", type=_parse_bbox, default=None)
    args = parser.parse_args()

    geometry = FrameGeometry(args.image_size, args.screen_size, args.capture_bbox_screen)
    width, height = geometry.image_size
    landmarks = {
        "top_left": (0, 0),
        "center": (width / 2, height / 2),
        "bottom_right": (width - 1, height - 1),
    }
    print(
        json.dumps(
            {
                **geometry.to_dict(),
                "landmarks": {
                    name: {
                        "image": list(point),
                        "screen": list(geometry.image_to_screen_point(*point)),
                    }
                    for name, point in landmarks.items()
                },
            },
            indent=2,
        )
    )


# Grounding
"""Dependency-free contracts for observation-bound GUI target references.

Target references are an input convenience only.  They are resolved by ``realtime_framework.sandbox.observation`` against
the exact observation that supplied them before the canonical action reaches the executor.
"""

import re as re
from dataclasses import dataclass as dataclass

SOM_REPRESENTATION_VERSION = "observation_bound_hybrid_som_coordinate_with_region_center_v2"
_TARGET_RE = re.compile(
    r"^(?P<kind>[PR])(?P<index>[1-9]\d*)(?:@(?P<u>\d{1,4}),(?P<v>\d{1,4}))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TargetReference:
    mark: str
    kind: str
    u: int | None = None
    v: int | None = None


def parse_target_reference(value: object) -> TargetReference | None:
    """Parse ``P7`` or ``R2@500,250`` without consulting an observation."""
    if not isinstance(value, str):
        return None
    match = _TARGET_RE.fullmatch(value.strip())
    if match is None:
        return None
    kind = match.group("kind").upper()
    mark = f"{kind}{int(match.group('index'))}"
    u_raw, v_raw = match.group("u"), match.group("v")
    if (u_raw is None) != (v_raw is None):
        return None
    if u_raw is None:
        return TargetReference(mark=mark, kind=kind)
    u, v = int(u_raw), int(v_raw)
    if kind != "R" or not (0 <= u <= 1000 and 0 <= v <= 1000):
        return None
    return TargetReference(mark=mark, kind=kind, u=u, v=v)


def action_has_target_references(action: Action | None) -> bool:
    return bool(
        action is not None
        and isinstance(action.parameters, dict)
        and any(key in action.parameters for key in ("target", "start", "end"))
    )


def validate_action_intent(action: Action | None) -> tuple[bool, str]:
    """Validate a canonical action or its still-unresolved SoM target form.

    The ordinary canonical schema stays closed.  This function substitutes validated references
    with harmless coordinates and then delegates all remaining checks to ``validate_action``.
    """
    if action is None:
        return False, "no action"
    if not isinstance(action.parameters, dict):
        return validate_action(action)
    if not action_has_target_references(action):
        return validate_action(action)

    params = dict(action.parameters)
    pointer_types = {
        ActionType.CLICK,
        ActionType.RIGHT_CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.TRIPLE_CLICK,
        ActionType.MOUSE_DOWN,
        ActionType.MOUSE_UP,
        ActionType.MOVE,
        ActionType.SCROLL,
    }
    if "target" in params:
        if action.action_type not in pointer_types:
            return False, f"{action.action_type.value} does not accept target"
        if "x" in params or "y" in params:
            return False, "target cannot be combined with x or y"
        if parse_target_reference(params.pop("target")) is None:
            return False, "target must be Pn, Rn, or Rn@u,v with u,v in 0..1000"
        params.update({"x": 0, "y": 0})

    if "start" in params or "end" in params:
        if action.action_type != ActionType.DRAG:
            return False, f"{action.action_type.value} does not accept start/end references"
        if "start" in params:
            if "x1" in params or "y1" in params:
                return False, "start cannot be combined with x1 or y1"
            if parse_target_reference(params.pop("start")) is None:
                return False, "start must be Pn, Rn, or Rn@u,v"
            params.update({"x1": 0, "y1": 0})
        if "end" in params:
            if "x2" in params or "y2" in params:
                return False, "end cannot be combined with x2 or y2"
            if parse_target_reference(params.pop("end")) is None:
                return False, "end must be Pn, Rn, or Rn@u,v"
            params.update({"x2": 0, "y2": 0})

    probe = Action(
        action.action_type,
        params,
        coordinate_space=action.coordinate_space,
        coordinate_transform=action.coordinate_transform,
    )
    return validate_action(probe)


# Translation
"""Translate generic JSON model output into canonical GUI actions."""

import logging as logging

logger = logging.getLogger(__name__)


class ActionTranslator:
    """Convert the shared JSON action vocabulary to the core action schema."""

    @staticmethod
    def from_json(data: dict) -> Action | None:
        action_name = data.get("action", "wait")
        type_map = {
            "click": ActionType.CLICK,
            "left_click": ActionType.CLICK,
            "right_click": ActionType.RIGHT_CLICK,
            "double_click": ActionType.DOUBLE_CLICK,
            "triple_click": ActionType.TRIPLE_CLICK,
            "mouse_down": ActionType.MOUSE_DOWN,
            "mouse_up": ActionType.MOUSE_UP,
            "type": ActionType.TYPE,
            "key": ActionType.KEY,
            "press": ActionType.KEY,
            "hotkey": ActionType.KEY,
            "keypress": ActionType.KEY,
            "hold_key": ActionType.HOLD_KEY,
            "scroll": ActionType.SCROLL,
            "drag": ActionType.DRAG,
            "move": ActionType.MOVE,
            "mouse_move": ActionType.MOVE,
            "wait": ActionType.WAIT,
            "done": ActionType.DONE,
            "fail": ActionType.FAIL,
            "give_up": ActionType.FAIL,
        }

        action_type = type_map.get(action_name)
        if action_type is None:
            logger.warning("Unknown action verb %r", action_name)
            return None
        params = {key: value for key, value in data.items() if key != "action"}

        if (
            action_type
            in (
                ActionType.CLICK,
                ActionType.RIGHT_CLICK,
                ActionType.DOUBLE_CLICK,
                ActionType.TRIPLE_CLICK,
                ActionType.MOUSE_DOWN,
                ActionType.MOUSE_UP,
                ActionType.MOVE,
            )
            and "coordinate" in params
        ):
            coordinate = params.pop("coordinate")
            if isinstance(coordinate, (list, tuple)) and len(coordinate) >= 2:
                params["x"], params["y"] = coordinate[0], coordinate[1]

        if action_type == ActionType.DRAG:
            start = params.pop("start_coordinate", None)
            end = params.pop("end_coordinate", None)
            coordinate = params.pop("coordinate", None)
            if end is None:
                end = coordinate
            if isinstance(start, (list, tuple)) and len(start) >= 2:
                params["x1"], params["y1"] = start[0], start[1]
            if isinstance(end, (list, tuple)) and len(end) >= 2:
                params["x2"], params["y2"] = end[0], end[1]
            if "end_x" in params:
                params["x2"] = params.pop("end_x")
            if "end_y" in params:
                params["y2"] = params.pop("end_y")
            if "x" in params and "x1" not in params:
                params["x1"] = params.pop("x")
            if "y" in params and "y1" not in params:
                params["y1"] = params.pop("y")

        if (
            action_type in (ActionType.KEY, ActionType.HOLD_KEY)
            and "keys" not in params
            and isinstance(params.get("key"), list)
        ):
            params["keys"] = params.pop("key")

        action = Action(action_type=action_type, parameters=params)
        valid, _ = validate_action_intent(action)
        return action if valid else None


# GUI functions
"""Canonical GUI actions as function declarations for drivers with a native function channel."""

GUI_FUNCTION_TARGET_DESCRIPTION = (
    "Current-screenshot Pn point mark or Rn@u,v region-local mark. For scroll, Rn means "
    "the region center. Never reuse an older screenshot's mark. Choose this reference OR its "
    "raw coordinate pair, never both; omit the unused fields."
)

GUI_FUNCTION_MODIFIER_DESCRIPTION = (
    "Optional: one modifier or a '+'-joined combination of shift, ctrl, alt, and super "
    "(for example ctrl+shift). Omit this field when no modifier is needed. "
    "Never send 'none', an empty string, or null."
)

GUI_FUNCTION_ACTION_DESCRIPTION = (
    "Submit this GUI action using current P/R marks when supplied, or screenshot coordinates "
    "when no suitable mark exists. It is queued for execution according to the active clock setting."
)


def _parameters(properties, required=(), choices=()):
    """Describe mutually exclusive parameter forms without changing native tool names.

    The root carries field types/descriptions; closed anyOf branches select a reference,
    a complete coordinate pair, or (where allowed) the current cursor. Unlike merely listing
    optional properties, these branches reject partial pairs and redundant coordinates.
    Avoid oneOf/not/conditionals so the declaration stays within Gemini's schema vocabulary.
    The shared action validator remains authoritative even if the provider ignores a constraint.
    """
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }
    if choices:
        exclusive = {key for group in choices for option in group for key in option}
        common = [key for key in properties if key not in exclusive]
        branches = []
        for combination in product(*choices):
            selected = [key for option in combination for key in option]
            branches.append(
                {
                    "type": "object",
                    # Constraints are inherited from the root; don't repeat long descriptions.
                    "properties": {
                        key: {"type": properties[key]["type"]} for key in [*common, *selected]
                    },
                    "required": [*required, *selected],
                    "additionalProperties": False,
                }
            )
        schema["anyOf"] = branches
    return schema


def gui_function_declarations():
    coordinate = {
        "type": "number",
        "description": (
            "Screenshot coordinate, not desktop coordinates. Supply both coordinates for this "
            "point and omit its P/R reference. Omit coordinates when using a reference."
        ),
    }
    string = {"type": "string"}
    duration = {"type": "number", "minimum": 0, "maximum": MAX_ACTION_DURATION}
    pointer_duration = {
        **duration,
        "description": (
            "Optional executor-side pointer interpolation duration in seconds; not a key hold. "
            "GUI execution latency is excluded from simulated time."
        ),
    }
    target = {
        "type": "string",
        "description": GUI_FUNCTION_TARGET_DESCRIPTION,
        "pattern": r"^(P[1-9][0-9]*|R[1-9][0-9]*(@(1000|[0-9]{1,3}),(1000|[0-9]{1,3}))?)$",
    }
    pointer = {"target": target, "x": coordinate, "y": coordinate}
    button = {"type": "string", "enum": ["left", "middle", "right"]}
    modifier = {
        "type": "string",
        "description": GUI_FUNCTION_MODIFIER_DESCRIPTION,
        "pattern": r"^(shift|ctrl|alt|super)(\+(shift|ctrl|alt|super))*$",
    }
    repeat = {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "description": "Optional short-tap repetition count; defaults to 1.",
    }
    key = {
        "type": "string",
        "pattern": r"^\s*[^+\s][^+]*(\+\s*[^+\s][^+]*)*$",
        "description": "Non-empty key or '+'-joined chord, such as enter or ctrl+s.",
    }
    optional_point = (("target",), ("x", "y"), ())
    required_point = (("target",), ("x", "y"))
    specs = {
        "click": ({**pointer, "button": button, "modifier": modifier}, (), (optional_point,)),
        "right_click": ({**pointer, "modifier": modifier}, (), (optional_point,)),
        "double_click": ({**pointer, "modifier": modifier}, (), (optional_point,)),
        "triple_click": ({**pointer, "modifier": modifier}, (), (optional_point,)),
        "mouse_down": (
            {**pointer, "button": button, "duration": pointer_duration},
            (),
            (optional_point,),
        ),
        "mouse_up": (
            {**pointer, "button": button, "duration": pointer_duration},
            (),
            (optional_point,),
        ),
        "move": ({**pointer, "duration": pointer_duration}, (), (required_point,)),
        "type": ({"text": string}, ("text",), ()),
        "key": ({"key": key, "repeat": repeat}, ("key",), ()),
        "hold_key": (
            {
                "key": key,
                "duration": {
                    **duration,
                    "description": (
                        "Optional hold duration in seconds, defaults to 1. Holds the entire chord; later "
                        "actions do not cancel it. Under the token clock this is simulated world time."
                    ),
                },
            },
            ("key",),
            (),
        ),
        "scroll": (
            {
                **pointer,
                "dx": {"type": "integer", "description": "Horizontal delta; positive is right."},
                "dy": {"type": "integer", "description": "Vertical delta; positive is up."},
                "modifier": modifier,
            },
            ("dx", "dy"),
            (optional_point,),
        ),
        "drag": (
            {
                "x1": coordinate,
                "y1": coordinate,
                "x2": coordinate,
                "y2": coordinate,
                "start": target,
                "end": target,
                "duration": pointer_duration,
                "button": button,
                "modifier": modifier,
            },
            (),
            ((("start",), ("x1", "y1"), ()), (("end",), ("x2", "y2"))),
        ),
        "wait": (
            {
                "duration": {
                    **duration,
                    "description": (
                        "Optional wait duration in seconds. Omit to wait for the next observation. "
                        "Under the token clock this is simulated world time."
                    ),
                }
            },
            (),
            (),
        ),
    }
    action_notes = {
        "click": "Omit all target/x/y fields only to click at the current cursor.",
        "right_click": "Right-click; omit all target/x/y fields only for the current cursor.",
        "double_click": "Double-click; omit all target/x/y fields only for the current cursor.",
        "triple_click": "Triple-click; omit all target/x/y fields only for the current cursor.",
        "mouse_down": "Press and keep the mouse button down until mouse_up; prefer drag for a complete drag.",
        "mouse_up": "Release the matching held mouse button.",
        "move": "Requires target OR both x and y; an omitted destination is invalid.",
        "key": "Short key tap; use hold_key for continuous control.",
        "hold_key": "Hold for the declared duration; auto-repeat begins after 0.5s, every 0.05s.",
        "scroll": "dx and dy are required and must not both be zero. Omit the point for the current cursor.",
        "drag": "Requires end OR x2/y2. Start may be omitted for the current cursor. Choose one form per endpoint.",
    }
    return [
        {
            "name": name,
            "description": " ".join(
                filter(None, (GUI_FUNCTION_ACTION_DESCRIPTION, action_notes.get(name)))
            ),
            "parameters_json_schema": _parameters(props, required, choices),
        }
        for name, (props, required, choices) in specs.items()
    ]


def canonical_function_action(name: str, arguments: dict) -> dict:
    declarations = {item["name"]: item for item in gui_function_declarations()}
    if name not in declarations or not isinstance(arguments, dict):
        raise ValueError("unknown GUI function or invalid arguments")
    schema = declarations[name]["parameters_json_schema"]
    if set(arguments) - set(schema["properties"]):
        raise ValueError("unexpected GUI function arguments")
    if set(schema["required"]) - set(arguments):
        raise ValueError("missing GUI function arguments")
    action = Action(ActionType(name), dict(arguments))
    valid, reason = validate_action_intent(action)
    if not valid:
        raise ValueError(reason)
    return {"action": name, **arguments}


if __name__ == "__main__":
    main()
