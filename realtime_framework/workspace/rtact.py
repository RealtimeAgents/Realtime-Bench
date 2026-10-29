"""rtact: a run_script's only way to observe and act on the task (gui+cli action space).

This module is copied into the workspace and imported by the model's scripts; it uses the standard
library alone, since the workspace has no network to install anything with. Every call is one
request to the action gateway over the unix socket RTACT_SOCKET names, which the runner executes
exactly as it executes the model's own GUI actions: the same validation, refusals, timing and
trace. The gateway takes nothing else, no browser, page, clock or state request.

    import rtact
    shot = rtact.screenshot()            # PNG of the desktop, saved under /workspace/obs/
    if shot.done: ...                     # the episode is over: stop
    rtact.key("Up")
    rtact.hold_key("Right", 0.3)
    rtact.click(640, 360)                 # desktop pixels of the screenshot

From bash, ``rtact screenshot`` prints the reply as JSON, and ``rtact key Up``, ``rtact click 640
360``, ``rtact act '{"action": "scroll", "x": 10, "y": 10, "dx": 0, "dy": -3}'`` act.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from dataclasses import dataclass, field

SOCKET_ENV = "RTACT_SOCKET"


@dataclass
class Reply:
    """The gateway's answer. ``done`` is True once the episode is over or the script's time is
    up: every later call answers the same, so a script should stop. ``status`` is the action's
    outcome ("applied", "rejected"), ``detail`` why one was refused; a screenshot's ``path`` is
    the PNG it wrote."""

    ok: bool
    done: bool = False
    status: str = ""
    detail: str = ""
    path: str | None = None
    width: int | None = None
    height: int | None = None
    raw: dict = field(default_factory=dict)


class _Connection:
    def __init__(self) -> None:
        self._socket: socket.socket | None = None
        self._reader = None

    def request(self, payload: dict) -> Reply:
        if self._socket is None:
            path = os.environ.get(SOCKET_ENV)
            if not path:
                raise RuntimeError(f"{SOCKET_ENV} is not set: run inside run_script")
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(path)
            self._reader = self._socket.makefile("r", encoding="utf-8")
        self._socket.sendall((json.dumps(payload) + "\n").encode())
        line = self._reader.readline()
        if not line:
            # The gateway closed: the script's time is up or the episode is over.
            return Reply(ok=False, done=True, status="closed")
        data = json.loads(line)
        return Reply(
            ok=bool(data.get("ok")),
            done=bool(data.get("done")),
            status=data.get("status", ""),
            detail=data.get("detail", ""),
            path=data.get("path"),
            width=data.get("width"),
            height=data.get("height"),
            raw=data,
        )


_connection = _Connection()


def act(action: dict) -> Reply:
    """Submit one GUI action object, as the model's text JSON names it."""
    return _connection.request({"op": "act", "action": action})


def screenshot() -> Reply:
    return _connection.request({"op": "screenshot"})


def click(x: float, y: float, button: str = "left", modifier: str | None = None) -> Reply:
    action = {"action": "click", "x": x, "y": y, "button": button}
    if modifier:
        action["modifier"] = modifier
    return act(action)


def double_click(x: float, y: float) -> Reply:
    return act({"action": "double_click", "x": x, "y": y})


def right_click(x: float, y: float) -> Reply:
    return act({"action": "right_click", "x": x, "y": y})


def move(x: float, y: float, duration: float | None = None) -> Reply:
    action = {"action": "move", "x": x, "y": y}
    if duration is not None:
        action["duration"] = duration
    return act(action)


def mouse_down(x: float | None = None, y: float | None = None, button: str = "left") -> Reply:
    action = {"action": "mouse_down", "button": button}
    if x is not None and y is not None:
        action.update(x=x, y=y)
    return act(action)


def mouse_up(x: float | None = None, y: float | None = None, button: str = "left") -> Reply:
    action = {"action": "mouse_up", "button": button}
    if x is not None and y is not None:
        action.update(x=x, y=y)
    return act(action)


def drag(x1: float, y1: float, x2: float, y2: float, duration: float | None = None) -> Reply:
    action = {"action": "drag", "x1": x1, "y1": y1, "x2": x2, "y2": y2}
    if duration is not None:
        action["duration"] = duration
    return act(action)


def type_text(text: str) -> Reply:
    return act({"action": "type", "text": text})


def key(name: str, repeat: int | None = None) -> Reply:
    action = {"action": "key", "key": name}
    if repeat is not None:
        action["repeat"] = repeat
    return act(action)


def hold_key(name: str, duration: float = 1.0) -> Reply:
    return act({"action": "hold_key", "key": name, "duration": duration})


def scroll(dx: int, dy: int, x: float | None = None, y: float | None = None) -> Reply:
    action = {"action": "scroll", "dx": dx, "dy": dy}
    if x is not None and y is not None:
        action.update(x=x, y=y)
    return act(action)


def _number(text: str) -> float:
    value = float(text)
    return int(value) if value.is_integer() else value


def main(argv: list[str] | None = None) -> int:
    """The bash form: ``rtact screenshot``, ``rtact key Up``, ``rtact hold_key Right 0.5``,
    ``rtact click X Y``, ``rtact type TEXT``, ``rtact act JSON``. Prints the reply as JSON and
    exits 0 when it was ok, 1 when refused, 2 once the episode is over."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(main.__doc__, file=sys.stderr)
        return 64
    verb, rest = args[0], args[1:]
    if verb == "screenshot":
        reply = screenshot()
    elif verb == "act" and len(rest) == 1:
        reply = act(json.loads(rest[0]))
    elif verb == "key" and len(rest) in (1, 2):
        reply = key(rest[0], int(rest[1]) if len(rest) == 2 else None)
    elif verb == "hold_key" and len(rest) in (1, 2):
        reply = hold_key(rest[0], float(rest[1]) if len(rest) == 2 else 1.0)
    elif verb in ("click", "double_click", "right_click", "move") and len(rest) == 2:
        x, y = (_number(value) for value in rest)
        reply = {"click": click, "double_click": double_click, "right_click": right_click,
                 "move": move}[verb](x, y)  # fmt: skip
    elif verb == "type" and len(rest) == 1:
        reply = type_text(rest[0])
    else:
        print(f"rtact: unknown command {' '.join(args)!r}", file=sys.stderr)
        return 64
    print(json.dumps(reply.raw or {"ok": reply.ok, "done": reply.done, "status": reply.status}))
    if reply.done:
        return 2
    return 0 if reply.ok else 1


if __name__ == "__main__":
    sys.exit(main())
