import base64
import io
import math
import os
import subprocess
import time

import pyautogui
from flask import Flask, jsonify, request

app = Flask(__name__)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05
# A tapped key stays down this long: two 60 Hz frames, so at least one frame boundary falls
# inside the press even with frame jitter. The token clock holds a tap for one rendered frame.
TAP_HOLD_S = 1.0 / 30.0

_MAX_ACTION_DURATION = 10.0
_MOUSE_BUTTONS = {"left", "middle", "right"}


_KEY_ALIASES = {
    "arrowleft": "left",
    "arrowright": "right",
    "arrowup": "up",
    "arrowdown": "down",
    "space": "space",
    "enter": "enter",
    "return": "enter",
    "backspace": "backspace",
    "delete": "delete",
    "escape": "escape",
    "tab": "tab",
    "shift": "shift",
    "control": "ctrl",
    "ctrl": "ctrl",
    "alt": "alt",
    "pageup": "pageup",
    "pagedown": "pagedown",
    "meta": "win",
    "super": "win",
    "cmd": "win",
    "command": "win",
}


def _is_supported_key(key):
    mapping = getattr(getattr(pyautogui, "platformModule", None), "keyboardMapping", None)
    if isinstance(mapping, dict):
        return key in mapping and mapping[key] is not None
    return key in getattr(pyautogui, "KEYBOARD_KEYS", ())


def _normalize_key_name(key):
    if not isinstance(key, str):
        raise TypeError(f"key name must be a string, got {type(key).__name__}")
    value = _KEY_ALIASES.get(key.lower(), key)
    if len(value) == 1 and value.isalpha():
        value = value.lower()
    if len(value) > 1:
        value = value.lower()
    if not _is_supported_key(value):
        raise ValueError(f"unsupported key for pyautogui: {key!r} (normalized: {value!r})")
    return value


def _normalized_keys(params):
    raw = params.get("keys")
    if raw is None:
        raw = [params.get("key", "")]
    if not isinstance(raw, list) or not raw:
        raise ValueError("key action requires a non-empty 'key' string or 'keys' list")
    parts = []
    for item in raw:
        if not isinstance(item, str):
            raise TypeError("key action 'keys' must contain only strings")
        parts.extend(part.strip() for part in item.split("+") if part.strip())
    if not parts:
        raise ValueError("key action requires at least one key")
    return [_normalize_key_name(part) for part in parts]


def _duration(params, default=0.0):
    value = params.get("duration", default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("duration must be a finite number")
    if not 0 <= value <= _MAX_ACTION_DURATION:
        raise ValueError(f"duration must be between 0 and {_MAX_ACTION_DURATION:g} seconds")
    return float(value)


def _button(params):
    value = params.get("button", "left")
    if not isinstance(value, str) or value not in _MOUSE_BUTTONS:
        raise ValueError(f"unsupported mouse button: {value!r}")
    return value


def _with_modifier(params, callback):
    modifier = params.get("modifier")
    if not modifier:
        return callback()
    if not isinstance(modifier, str):
        raise TypeError("modifier must be a string")
    keys = [_normalize_key_name(part.strip()) for part in modifier.split("+") if part.strip()]
    if not keys:
        raise ValueError("modifier must contain at least one key")
    pressed = []
    try:
        for key in keys:
            pyautogui.keyDown(key)
            pressed.append(key)
        return callback()
    finally:
        for key in reversed(pressed):
            pyautogui.keyUp(key)


def _repeat(params):
    value = params.get("repeat", 1)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("repeat must be an integer between 1 and 100")
    return value


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})


@app.route("/screenshot", methods=["GET"])
def screenshot():
    img = pyautogui.screenshot()
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return jsonify(
        {
            "image": base64.b64encode(img_bytes).decode("utf-8"),
            "width": img.width,
            "height": img.height,
            "timestamp": time.time(),
        }
    )


@app.route("/action", methods=["POST"])
def execute_action():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "request body must be a JSON object"}), 400
    action_type = data.get("action_type")
    params = data.get("parameters", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return jsonify({"status": "error", "message": "parameters must be a JSON object"}), 400
    if not action_type:
        return jsonify({"status": "error", "message": "missing 'action_type'"}), 400

    # Keep the sandbox boundary as strict as the host-side canonical validator. Malformed actions
    # fail visibly instead of degrading to an accidental no-op.
    _ALLOWED = {
        "click": {"x", "y", "button", "modifier"},
        "right_click": {"x", "y", "modifier"},
        "double_click": {"x", "y", "modifier"},
        "triple_click": {"x", "y", "modifier"},
        "mouse_down": {"x", "y", "button", "duration"},
        "mouse_up": {"x", "y", "button", "duration"},
        "move": {"x", "y", "duration"},
        "type": {"text"},
        "key": {"key", "keys", "repeat"},
        "hold_key": {"key", "keys", "duration"},
        "scroll": {"x", "y", "dx", "dy", "modifier"},
        "drag": {"x1", "y1", "x2", "y2", "duration", "button", "modifier"},
        "wait": {"duration"},
    }
    if action_type not in _ALLOWED:
        return jsonify({"status": "error", "message": f"Unknown action: {action_type}"}), 400
    unexpected = set(params) - _ALLOWED[action_type]
    if unexpected:
        return jsonify(
            {
                "status": "error",
                "message": f"{action_type}: unexpected parameters {sorted(unexpected)}",
            }
        ), 400

    required = {
        "move": ("x", "y"),
        "drag": ("x2", "y2"),
        "type": ("text",),
        "scroll": ("dx", "dy"),
    }.get(action_type, ())
    for name in required:
        if name not in params or params[name] is None:
            return jsonify(
                {
                    "status": "error",
                    "message": f"{action_type}: missing required parameter '{name}'",
                }
            ), 400
    if action_type == "type" and not isinstance(params.get("text"), str):
        return jsonify({"status": "error", "message": "type: text must be a string"}), 400
    if action_type in ("key", "hold_key") and "key" not in params and "keys" not in params:
        return jsonify({"status": "error", "message": f"{action_type}: missing key/keys"}), 400
    if action_type in ("key", "hold_key") and "key" in params and "keys" in params:
        return jsonify(
            {
                "status": "error",
                "message": f"{action_type}: supply either key or keys, not both",
            }
        ), 400

    pointer_actions = {
        "click",
        "right_click",
        "double_click",
        "triple_click",
        "mouse_down",
        "mouse_up",
        "scroll",
    }
    if action_type in pointer_actions and (("x" in params) != ("y" in params)):
        return jsonify(
            {
                "status": "error",
                "message": f"{action_type}: x and y must be supplied together",
            }
        ), 400
    for name in ("x", "y", "x1", "y1", "x2", "y2", "dx", "dy"):
        if name in params and (
            (value := params[name]) is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return jsonify(
                {"status": "error", "message": f"{action_type}: parameter '{name}' must be numeric"}
            ), 400
    if action_type == "scroll" and params["dx"] == 0 and params["dy"] == 0:
        return jsonify(
            {
                "status": "error",
                "message": "scroll: dx and dy cannot both be zero",
            }
        ), 400
    if action_type == "drag" and (("x1" in params) != ("y1" in params)):
        return jsonify(
            {"status": "error", "message": "drag: x1 and y1 must be supplied together"}
        ), 400
    if (
        action_type == "drag"
        and "x1" in params
        and any(
            isinstance(params[name], bool)
            or not isinstance(params[name], (int, float))
            or not math.isfinite(params[name])
            for name in ("x1", "y1")
        )
    ):
        return jsonify({"status": "error", "message": "drag: x1 and y1 must be numeric"}), 400

    try:
        if "duration" in params:
            _duration(params)
        if "button" in params:
            _button(params)
        if "modifier" in params:
            if not isinstance(params["modifier"], str):
                raise TypeError("modifier must be a string")
            for part in params["modifier"].split("+"):
                if not part.strip():
                    raise ValueError("modifier contains an empty key")
                if part.strip().lower() not in {
                    "shift",
                    "ctrl",
                    "control",
                    "alt",
                    "super",
                    "meta",
                    "cmd",
                    "command",
                    "win",
                }:
                    raise ValueError("modifier contains a non-modifier key")
                _normalize_key_name(part.strip())
        if "repeat" in params:
            _repeat(params)
    except (TypeError, ValueError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    try:
        if action_type == "click":
            _with_modifier(
                params,
                lambda: pyautogui.click(
                    x=params.get("x"), y=params.get("y"), button=_button(params)
                ),
            )
        elif action_type == "right_click":
            _with_modifier(
                params, lambda: pyautogui.rightClick(x=params.get("x"), y=params.get("y"))
            )
        elif action_type == "double_click":
            _with_modifier(
                params, lambda: pyautogui.doubleClick(x=params.get("x"), y=params.get("y"))
            )
        elif action_type == "triple_click":
            _with_modifier(
                params,
                lambda: pyautogui.click(
                    x=params.get("x"), y=params.get("y"), clicks=3, interval=0.1, button="left"
                ),
            )
        elif action_type == "mouse_down":
            if "x" in params:
                pyautogui.moveTo(params["x"], params["y"], duration=_duration(params))
            pyautogui.mouseDown(button=_button(params))
        elif action_type == "mouse_up":
            if "x" in params:
                pyautogui.moveTo(params["x"], params["y"], duration=_duration(params))
            pyautogui.mouseUp(button=_button(params))
        elif action_type == "type":
            text = params["text"]
            # One xdotool call injects the whole string. A per-character pyautogui.write loop
            # pays PAUSE per character, so a multi-KB script outlasts the client timeout and
            # later keystrokes interleave with it. xdotool keeps newlines, tabs and shell
            # metacharacters verbatim; passing text as an argv element (after --) avoids any
            # shell re-parsing.
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "2", "--", text],
                check=True,
            )
        elif action_type in ("key", "hold_key"):
            try:
                keys = _normalized_keys(params)
            except (TypeError, ValueError) as exc:
                return jsonify({"status": "error", "message": str(exc)}), 400
            if action_type == "hold_key" or "duration" in params:
                pressed = []
                try:
                    for key in keys:
                        pyautogui.keyDown(key)
                        pressed.append(key)
                    time.sleep(_duration(params, default=1.0))
                finally:
                    for key in reversed(pressed):
                        pyautogui.keyUp(key)
            else:
                # pyautogui.press/hotkey release at once, so games that read key state once per
                # frame rarely see the tap. Keep each tap down, and up again, across a frame.
                for index in range(_repeat(params)):
                    if index:
                        time.sleep(TAP_HOLD_S)
                    pressed = []
                    try:
                        for key in keys:
                            pyautogui.keyDown(key, _pause=False)
                            pressed.append(key)
                        time.sleep(TAP_HOLD_S)
                    finally:
                        for key in reversed(pressed):
                            pyautogui.keyUp(key, _pause=False)
        elif action_type == "scroll":

            def scroll():
                pyautogui.scroll(params.get("dy", 0), x=params.get("x"), y=params.get("y"))
                if params.get("dx", 0):
                    pyautogui.hscroll(params["dx"], x=params.get("x"), y=params.get("y"))

            _with_modifier(params, scroll)
        elif action_type == "move":
            pyautogui.moveTo(x=params["x"], y=params["y"], duration=_duration(params))
        elif action_type == "drag":
            if "x1" in params:
                pyautogui.moveTo(params["x1"], params["y1"])
            _with_modifier(
                params,
                lambda: pyautogui.dragTo(
                    params["x2"],
                    params["y2"],
                    duration=_duration(params, default=0.5),
                    button=_button(params),
                ),
            )
        elif action_type == "wait":
            time.sleep(_duration(params))

        return jsonify({"status": "ok", "timestamp": time.time()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/execute", methods=["POST"])
def execute_command():
    data = request.json
    cmd = data["command"]
    timeout = data.get("timeout", 30)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return jsonify(
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "timestamp": time.time(),
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify(
            {
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1,
                "timestamp": time.time(),
            }
        )
    except Exception as e:
        return jsonify(
            {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "timestamp": time.time(),
            }
        ), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("GUEST_AGENT_PORT", "5000")))
