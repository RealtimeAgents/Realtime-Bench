"""Offline doubles for the sandbox's browser debugging port and guest agent, on 127.0.0.1.

``FakeCDP`` stands in for Chrome's remote debugging port as realtime_framework/sandbox/browser/
cdp.py uses it: ``page_targets`` reads ``GET /json``, and ``PersistentCDPClient`` opens the listed
``webSocketDebuggerUrl``, sends ``{"id", "method", "params"}`` commands and reads messages until
the one carrying its id, keeping the rest as events. ``FakeGuest`` answers the HTTP API that
``GuestClient`` (realtime_framework/sandbox/guest.py) calls. Either fake can delay or withhold its
answers so timeout paths run against a real socket; every delay ends as soon as the fake stops.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Self

from PIL import Image
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

FLOOD_INTERVAL_S = 0.01  # gap between unrelated events in flood mode
CLOSE_TIMEOUT_S = 1.0  # bounds each WebSocket closing handshake once the fake stops

_FRAME_NAVIGATED = json.dumps({"method": "Page.frameNavigated", "params": {}})
_BUDGET_EXPIRED = json.dumps({"method": "Emulation.virtualTimeBudgetExpired", "params": {}})


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    return buffer.getvalue()


SCREENSHOT_PNG = _png()  # GET /screenshot returns it base64-encoded


class FakeCDP:
    """One page target on 127.0.0.1: GET /json lists it and its WebSocket answers CDP commands.

    In ``"reply"`` mode each command gets one unrelated event, then a reply shaped like a
    ``Runtime.evaluate`` result whose value is ``values[params["expression"]]``. In ``"flood"``
    mode no command is answered: unrelated events stream from the moment a connection opens until
    it closes. ``mode`` applies to connections opened after it is set.
    """

    def __init__(
        self,
        *,
        mode: str = "reply",
        json_delay: float = 0.0,
        handshake_delay: float = 0.0,
        values: dict | None = None,
    ):
        self.target_id = "page-1"
        self.mode = mode
        self.json_delay = json_delay
        self.handshake_delay = handshake_delay
        self.commands: list[dict] = []
        self.json_requests = 0
        self.connections = 0
        self._values = values or {}
        self._stop = threading.Event()
        self._open = set()
        self._server = serve(
            self._serve,
            "127.0.0.1",
            0,
            process_request=self._process_request,
            close_timeout=CLOSE_TIMEOUT_S,
        )
        self._port = self._server.socket.getsockname()[1]  # shutdown() closes the socket
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._port

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._server.shutdown()
        # shutdown() leaves each connection's non-daemon thread waiting on its client.
        for websocket in list(self._open):
            websocket.close()
        self._thread.join()

    def drop(self) -> None:
        """Close every open connection from the page's side, as Chrome drops a DevTools client;
        each close returns once its closing handshake is done."""
        for websocket in list(self._open):
            websocket.close()

    def _process_request(self, connection, request):
        if request.path == "/json":
            self._stop.wait(self.json_delay)
            self.json_requests += 1
            return connection.respond(200, json.dumps([self._target()]))
        self._stop.wait(self.handshake_delay)
        return None

    def _target(self) -> dict:
        return {
            "id": self.target_id,
            "type": "page",
            "title": "fake",
            "url": "http://localhost:8930/game",
            "webSocketDebuggerUrl": f"ws://127.0.0.1:{self.port}/devtools/page/{self.target_id}",
        }

    def _serve(self, websocket):
        self.connections += 1
        self._open.add(websocket)
        try:
            if self._stop.is_set():  # accepted after __exit__ closed the open connections
                return
            if self.mode == "flood":
                self._flood(websocket)
                return
            for message in websocket:
                command = json.loads(message)
                self.commands.append(command)
                self._reply(websocket, command)
        except ConnectionClosed:
            pass
        finally:
            self._open.discard(websocket)

    def _flood(self, websocket):
        while not self._stop.is_set():
            try:
                self.commands.append(json.loads(websocket.recv(timeout=FLOOD_INTERVAL_S)))
            except TimeoutError:
                websocket.send(_FRAME_NAVIGATED)

    def _reply(self, websocket, command: dict):
        params = command.get("params") or {}
        value = self._values.get(params.get("expression"))
        websocket.send(_FRAME_NAVIGATED)
        websocket.send(
            json.dumps(
                {"id": command["id"], "result": {"result": {"type": "object", "value": value}}}
            )
        )
        if (
            command.get("method") == "Emulation.setVirtualTimePolicy"
            and params.get("policy") == "advance"
        ):
            websocket.send(_BUDGET_EXPIRED)


class FakeGuest:
    """The guest agent's HTTP API on 127.0.0.1, with optional per-endpoint delays. With
    ``refusal`` set, /action refuses every action as the guest server refuses one it validates
    and rejects (sandbox/guest_server/server.py): 400 with the refusal as its message."""

    def __init__(
        self,
        *,
        screenshot_delay: float = 0.0,
        action_delay: float = 0.0,
        command_delay: float = 0.0,
        refusal: str | None = None,
    ):
        self.screenshot_delay = screenshot_delay
        self.action_delay = action_delay
        self.command_delay = command_delay
        self.refusal = refusal
        self.actions: list[dict] = []
        self.commands: list[dict] = []
        self._stop = threading.Event()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._server.block_on_close = False
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def _handler(self):
        guest = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self._send({"status": "healthy"})
                elif self.path == "/screenshot":
                    guest._stop.wait(guest.screenshot_delay)
                    self._send({"image": base64.b64encode(SCREENSHOT_PNG).decode()})
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/action":
                    guest.actions.append(body)
                    guest._stop.wait(guest.action_delay)
                    if guest.refusal is not None:
                        self._send({"status": "error", "message": guest.refusal}, 400)
                    else:
                        self._send({"status": "ok", "timestamp": time.time()})
                elif self.path == "/execute":
                    guest.commands.append(body)
                    guest._stop.wait(guest.command_delay)
                    self._send({"returncode": 0, "stdout": "", "stderr": ""})
                else:
                    self.send_error(404)

            def _send(self, payload: dict, status: int = 200):
                body = json.dumps(payload).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client gave up during the delay

            def log_message(self, *args):
                pass

        return Handler
