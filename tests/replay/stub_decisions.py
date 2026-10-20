"""Scripted OpenRouter Decisions endpoint on 127.0.0.1 with an OS-assigned port.

Both Decisions clients post each Choice question attempt to ``{base_url}/decisions``, one after
another (providers/openai_decisions.py:184; harnesses/jev/selection.py:163-172, 246). The test loads
one decision's script before the decision. Each request takes the next entry: an Answer is
returned in the smallest body the client accepts (openai_decisions.py:90-105, 204-217), a Failure
as the error status and body it names. Every request is kept in arrival order, with its path,
lower-cased headers and raw body.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

QUESTION_KEY = "next_action"  # providers/openai_decisions.py:34


@dataclass(frozen=True)
class Answer:
    """The model picks ``choice``; it must be one of the question's criteria keys."""

    choice: str
    usage: dict


@dataclass(frozen=True)
class Failure:
    """An error response; the client reads ``error.message`` (openai_decisions.py:185-203)."""

    status: int
    message: str


@dataclass(frozen=True)
class Received:
    path: str
    headers: dict[str, str]
    body: bytes


class DecisionsStub:
    def __init__(self, *, upstream: str):
        self._upstream = upstream
        self._script: list[Answer | Failure] = []
        self.requests: list[Received] = []
        self.unscripted = 0
        self.off_script: list[tuple[int, str]] = []  # (request index, choice not in criteria)
        self._server = HTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        """A model YAML base_url as the catalog stores it, without a trailing slash
        (providers/registry.py:279); httpx joins ``decisions`` below it."""
        return f"http://127.0.0.1:{self._server.server_address[1]}/api/alpha"

    def load(self, script) -> None:
        self._script = list(script)

    def unload(self) -> int:
        """Drop what the last decision left unused and return how many entries that was."""
        unused, self._script = len(self._script), []
        return unused

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def _respond(self, path: str, headers: dict, body: bytes) -> tuple[int, dict]:
        self.requests.append(Received(path, headers, body))
        index = len(self.requests) - 1
        if not self._script:
            self.unscripted += 1
            # HTTP 400 is FATAL for the per-question retry (entity/model.py:369-377), so a
            # question beyond the script fails its decision at once instead of being retried.
            return 400, _error(400, "request beyond the script")
        entry = self._script.pop(0)
        if isinstance(entry, Failure):
            return entry.status, _error(entry.status, entry.message)
        request = json.loads(body)
        criteria = request["questions"][QUESTION_KEY]["criteria"]
        if entry.choice not in criteria:
            self.off_script.append((index, entry.choice))
            return 400, _error(400, f"scripted choice {entry.choice!r} is not offered")
        return 200, {
            "provider": self._upstream,
            "model": request["model"],
            "usage": entry.usage,
            "answers": {
                QUESTION_KEY: {
                    "type": "choice",
                    "choice": entry.choice,
                    "probabilities": {key: float(key == entry.choice) for key in criteria},
                    "confidence": 1.0,
                }
            },
        }

    def _handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                headers = {name.lower(): value for name, value in self.headers.items()}
                status, payload = stub._respond(self.path, headers, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        return Handler


def _error(status: int, message: str) -> dict:
    return {"error": {"code": status, "message": message}}
