"""The action gateway: the one channel from a run_script to the task (gui+cli action space).

A unix socket in a directory of its own, which only the workspace is given. It reads one JSON
request per line, answers one JSON line, and accepts two requests alone:

- ``{"op": "act", "action": {...}}``: one GUI action object, as the model's text JSON names it,
  in desktop pixels. It must be a valid canonical action (entity/actions.py validate_action):
  no mark references, which belong to the model's own screenshots, and no wait, done, fail or
  run_script; a script waits by sleeping.
- ``{"op": "screenshot"}``: a PNG of the desktop, written into the workspace.

There is no request for the page, the browser, its devtools, the task's state or its clock. The
gateway hands every valid request to ``submit(kind, action)``, which the shell answers on its own
thread with the same environment calls a model action makes (evaluation/shell.py), and holds
requests to one at a time across connections. Once ``shut`` (the script's time is up, the episode
is over), every request is answered ``done``.
"""

from __future__ import annotations

import json
import os
import socketserver
import threading
from collections.abc import Callable
from pathlib import Path

from realtime_framework.entity.actions import ActionTranslator, ActionType, validate_action

SOCKET_NAME = "gateway.sock"
# Bytes of one request line; a GUI action object is far shorter.
MAX_REQUEST_BYTES = 64 * 1024
_NOT_ACTIONS = (ActionType.WAIT, ActionType.DONE, ActionType.FAIL, ActionType.RUN_SCRIPT)

DONE = {"ok": False, "done": True, "status": "done", "detail": "the script may act no more"}


def _refused(detail: str) -> dict:
    return {"ok": False, "done": False, "status": "invalid", "detail": detail}


def parse_request(line: str):
    """``(kind, action)`` of a request line, or ``(None, refusal)``: kind is "act" with its
    canonical Action, or "screenshot" with None."""
    try:
        request = json.loads(line)
    except ValueError:
        return None, _refused("request is not JSON")
    if not isinstance(request, dict):
        return None, _refused("request must be a JSON object")
    op = request.get("op")
    if op == "screenshot" and set(request) == {"op"}:
        return "screenshot", None
    if op != "act" or set(request) != {"op", "action"}:
        return None, _refused("request must be {'op': 'screenshot'} or {'op': 'act', 'action'}")
    data = request["action"]
    if not isinstance(data, dict) or data.get("action") in (None, "run_script"):
        return None, _refused("action must be a GUI action object")
    action = ActionTranslator.from_json(dict(data))
    if action is None:
        return None, _refused(f"invalid GUI action: {data!r}"[:500])
    if action.action_type in _NOT_ACTIONS:
        return None, _refused(f"{action.action_type.value} is not a gateway action")
    # Strict: P/R marks are the model's screenshots', which a script never sees.
    ok, reason = validate_action(action)
    if not ok:
        return None, _refused(reason)
    return "act", action


class ActionGateway:
    """The socket server, on threads of its own. ``submit(kind, action) -> dict`` answers one
    valid request; it is called on the connection's thread, one call at a time."""

    def __init__(self, directory: Path, submit: Callable[[str, object], dict]) -> None:
        self.directory = Path(directory)
        self.path = self.directory / SOCKET_NAME
        self._submit = submit
        self._one_at_a_time = threading.Lock()
        self._open = threading.Event()  # set while a script may act
        self._server: socketserver.ThreadingUnixStreamServer | None = None
        self._thread: threading.Thread | None = None
        self.requests = 0
        self.refused = 0

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        gateway = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                while True:
                    line = self.rfile.readline(MAX_REQUEST_BYTES + 1)
                    if not line:
                        return
                    if len(line) > MAX_REQUEST_BYTES:
                        reply = _refused("request too long")
                    else:
                        reply = gateway.answer(line.decode("utf-8", "replace"))
                    try:
                        self.wfile.write((json.dumps(reply) + "\n").encode())
                        self.wfile.flush()
                    except OSError:
                        return
                    if reply.get("done"):
                        return

        server = socketserver.ThreadingUnixStreamServer(str(self.path), Handler)
        server.daemon_threads = True
        os.chmod(self.path, 0o666)  # the workspace's user may differ from the runner's
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="action-gateway")
        self._thread.daemon = True
        self._thread.start()

    def open(self) -> None:
        """A script starts: its requests are answered until ``shut``."""
        self._open.set()

    def shut(self) -> None:
        self._open.clear()

    def answer(self, line: str) -> dict:
        self.requests += 1
        kind, action = parse_request(line)
        if kind is None:
            self.refused += 1
            return action
        with self._one_at_a_time:
            if not self._open.is_set():
                return dict(DONE)
            return self._submit(kind, action)

    def close(self) -> None:
        self.shut()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self.path.exists():
            self.path.unlink()
