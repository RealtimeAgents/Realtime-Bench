"""Scripted OpenRouter chat completions endpoint on 127.0.0.1 with an OS-assigned port.

The openrouter API posts through LiteLLM's OpenRouter provider with LiteLLM's own HTTP handler
(providers/litellm.py:1123-1136, 605-610) to ``{OPENROUTER_BASE_URL}/chat/completions``; the driver
reads OPENROUTER_BASE_URL because the catalog entries carry no base_url (providers/litellm.py:
470-475). Each request takes the next script entry: a Reply is streamed as OpenRouter server-sent
events, a Failure is answered with its HTTP status, headers and JSON error body, a Held reply
is streamed up to its last event, which waits until the test releases it, and a Cut reply up to
its first text chunk, where the connection closes. Every request is kept with its method, path,
headers and raw body, in arrival order.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .stub import CREATED, Failure, Reply, _error, _pieces

HOLD_TIMEOUT_S = 30.0  # a Held reply's last event is sent after this even without a release


@dataclass(frozen=True)
class Held:
    """A reply whose last event (the usage chunk) is sent only after ``release_last_event``."""

    reply: Reply


@dataclass(frozen=True)
class Cut:
    """A reply whose stream breaks off after its first text chunk: the connection closes short of
    the length the response announced."""

    reply: Reply


@dataclass(frozen=True)
class Received:
    method: str
    path: str
    headers: tuple[tuple[str, str], ...]
    body: bytes


def route_metadata(model: str, provider: str) -> dict:
    """The route OpenRouter reports when X-OpenRouter-Metadata is enabled, in the form the driver
    reads (providers/litellm.py:732-734, 740-756): exactly one selected endpoint."""
    return {
        "requested": model,
        "strategy": "ordered",
        "attempt": 1,
        "endpoints": {"available": [{"provider": provider, "model": model, "selected": True}]},
    }


class OpenRouterStub:
    def __init__(self, script: list[Reply | Failure | Held], *, route: dict):
        self._script = list(script)
        self._route = route
        self.requests: list[Received] = []
        self.unscripted = 0
        self.hold_timed_out = False
        self._release = threading.Event()
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        """The OPENROUTER_BASE_URL form: the API prefix, as in https://openrouter.ai/api/v1
        (providers/litellm.py:59)."""
        return f"http://127.0.0.1:{self._server.server_address[1]}/api/v1"

    @property
    def unused_replies(self) -> int:
        return len(self._script)

    def release_last_event(self) -> None:
        self._release.set()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._release.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def _next(self, received: Received):
        with self._lock:
            self.requests.append(received)
            if not self._script:
                self.unscripted += 1
                return None
            return self._script.pop(0)

    def _handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            # LiteLLM's aiohttp transport keeps connections alive; every reply carries its length.
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received = Received(self.command, self.path, tuple(self.headers.items()), body)
                entry = stub._next(received)
                if entry is None:
                    # HTTP 400 is FATAL for the shared retry policy (entity/model.py:418-426), so
                    # an extra request fails its decision at once instead of being retried.
                    self._start(400, (), "application/json", _error("request beyond the script"))
                    return
                if isinstance(entry, Failure):
                    payload = json.dumps(entry.body).encode()
                    self._start(entry.status, entry.headers, "application/json", payload)
                    return
                model = json.loads(body)["model"]
                index = len(stub.requests)
                reply = entry.reply if isinstance(entry, Held | Cut) else entry
                events = _events(reply, model, index, stub._route)
                lines = [f"data: {json.dumps(event)}\n\n".encode() for event in events]
                lines.append(b"data: [DONE]\n\n")
                if isinstance(entry, Cut):
                    first_text = next(
                        i
                        for i, event in enumerate(events)
                        if event["choices"][0]["delta"].get("content")
                    )
                    sent = b"".join(lines[: first_text + 1])
                    self._start(200, (), "text/event-stream", sent, length=len(b"".join(lines)))
                    self.close_connection = True
                    return
                if not isinstance(entry, Held):
                    self._start(200, (), "text/event-stream", b"".join(lines))
                    return
                head, last = b"".join(lines[:-2]), b"".join(lines[-2:])
                self._start(200, (), "text/event-stream", head, length=len(head) + len(last))
                stub.hold_timed_out = not stub._release.wait(HOLD_TIMEOUT_S)
                self.wfile.write(last)

            def _start(self, status, headers, content_type, payload, *, length=None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload) if length is None else length))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        return Handler


def _events(reply: Reply, model: str, index: int, route: dict) -> list[dict]:
    """OpenRouter's stream: every chunk carries id, created, model and choices (LiteLLM's parser
    raises KeyError without them, litellm/llms/openrouter/chat/transformation.py:273-286),
    reasoning arrives in ``delta.reasoning`` (copied to reasoning_content at transformation.py:
    270-272), and the usage, with the route metadata, follows the finish chunk with no choices.
    A reply without usage (``usage=None``) ends at the finish chunk."""
    head = {
        "id": f"gen-replay-{index:04d}",
        "object": "chat.completion.chunk",
        "created": CREATED,
        "model": model,
    }

    def chunk(delta: dict, finish_reason: str | None = None) -> dict:
        choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
        return {**head, "choices": [choice]}

    events = [chunk({"role": "assistant", "content": ""})]
    events += [chunk({"reasoning": piece}) for piece in _pieces(reply.reasoning)]
    events += [chunk({"content": piece}) for piece in _pieces(reply.content)]
    events.append(chunk({}, "stop"))
    if reply.usage is not None:
        events.append({**head, "choices": [], "usage": reply.usage, "openrouter_metadata": route})
    return events
