"""Scripted OpenAI chat completions endpoint on 127.0.0.1 with an OS-assigned port.

The chat_completions API reaches it through LiteLLM's OpenAI provider with an AsyncOpenAI client
(providers/litellm.py ChatCompletionsDriver, LiteLLMDriver._client), which posts to
``{base_url}/chat/completions``. Each request is answered with the next scripted reply: a JSON body
when the request sets ``stream: false``, otherwise server-sent events ending in a usage chunk and
``[DONE]``; a Failure is answered with its HTTP status, headers and JSON error body. Every raw
request body is kept in arrival order.

Given a fake clock (tests/fake_clock.py), the stub serves on one of the clock's threads, so a
reply's scripted latency is a sleep on that clock, and it waits on the clock for a held request's
client to close.
"""

from __future__ import annotations

import json
import select
import socket
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..fake_clock import wait_delivered

CREATED = 1_760_000_000  # fixed response timestamp
STREAM_PIECE_CHARS = 12  # visible and reasoning text arrive in pieces of this size


@dataclass(frozen=True)
class Reply:
    content: str
    usage: dict
    reasoning: str = ""  # sent as reasoning_content, as OpenAI-compatible reasoning servers do


@dataclass(frozen=True)
class Failure:
    """An HTTP error answered before any output is streamed."""

    status: int
    body: dict
    headers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Received:
    path: str
    body: bytes


HOLD_LIMIT_S = 30.0  # a held request is released after this long even if the client stays
_HELD = object()


class ChatCompletionsStub:
    def __init__(self, replies: list[Reply], on_request=None, *, clock=None):
        """``on_request(index)``, when given, runs on the server thread after request ``index``
        (0-based) has been recorded and before it is answered. A true return holds the request:
        nothing is sent and no reply is used until the client closes the connection, and
        ``held`` records whether it did."""
        self._replies = list(replies)
        self._on_request = on_request
        self._clock = clock
        self._closing = False
        self.requests: list[Received] = []
        self.unscripted = 0
        self.held: list[bool] = []
        self._server = HTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    @property
    def unused_replies(self) -> int:
        return len(self._replies)

    def __enter__(self):
        if self._clock is None:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
        else:
            self._thread = self._clock.start_thread(self._serve_on_clock)
        return self

    def __exit__(self, *exc_info):
        if self._clock is None:
            self._server.shutdown()
        else:
            self._closing = True
            socket.create_connection(self._server.server_address).close()  # wakes the server
        self._thread.join()
        self._server.server_close()

    def _serve_on_clock(self):
        while self._clock.wait_readable(self._server.socket) and not self._closing:
            self._server.handle_request()

    def _next(self, path: str, body: bytes):
        self.requests.append(Received(path, body))
        if self._on_request is not None and self._on_request(len(self.requests) - 1):
            return _HELD
        if not self._replies:
            self.unscripted += 1
            return None
        return self._replies.pop(0)

    def _handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                reply = stub._next(self.path, body)
                if reply is _HELD:
                    stub.held.append(self._wait_for_close())
                    return
                if reply is None:
                    # HTTP 400 is FATAL for the shared retry policy (entity/model.py
                    # classify_http_failure), so an extra request fails its decision at once instead
                    # of being retried.
                    self._send(400, "application/json", _error("request beyond the script"))
                    return
                if isinstance(reply, Failure):
                    payload = json.dumps(reply.body).encode()
                    self._send(reply.status, "application/json", payload, reply.headers)
                    return
                request = json.loads(body)
                index = len(stub.requests)
                if request.get("stream"):  # a non-streaming body carries no "stream" field
                    self._send(200, "text/event-stream", _sse(reply, request["model"], index))
                else:
                    self._send(200, "application/json", _completion(reply, request["model"], index))

            def _wait_for_close(self) -> bool:
                """A client waiting for its response sends nothing, so the socket turns readable
                only when the client closes it, and then reads end-of-file."""
                if stub._clock is None:
                    readable, _, _ = select.select([self.connection], [], [], HOLD_LIMIT_S)
                else:
                    readable = stub._clock.wait_readable(self.connection, HOLD_LIMIT_S)
                self.close_connection = True
                return bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)

            def _send(self, status: int, content_type: str, payload: bytes, headers=()):
                self.send_response(status)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                if stub._clock is not None:
                    # The reply must not be in flight while the server waits on the clock again.
                    wait_delivered(self.connection)

            def log_message(self, *args):
                pass

        return Handler


def _error(message: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": "invalid_request_error"}}).encode()


def _pieces(text: str) -> list[str]:
    return [text[i : i + STREAM_PIECE_CHARS] for i in range(0, len(text), STREAM_PIECE_CHARS)]


def _completion(reply: Reply, model: str, index: int) -> bytes:
    message = {"role": "assistant", "content": reply.content}
    if reply.reasoning:
        message["reasoning_content"] = reply.reasoning
    return json.dumps(
        {
            "id": f"chatcmpl-replay-{index:04d}",
            "object": "chat.completion",
            "created": CREATED,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": reply.usage,
        }
    ).encode()


def _sse(reply: Reply, model: str, index: int) -> bytes:
    head = {
        "id": f"chatcmpl-replay-{index:04d}",
        "object": "chat.completion.chunk",
        "created": CREATED,
        "model": model,
    }

    def chunk(delta: dict, finish_reason: str | None = None) -> dict:
        choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
        return {**head, "choices": [choice]}

    events = [chunk({"role": "assistant", "content": ""})]
    events += [chunk({"reasoning_content": piece}) for piece in _pieces(reply.reasoning)]
    events += [chunk({"content": piece}) for piece in _pieces(reply.content)]
    events.append(chunk({}, "stop"))
    events.append({**head, "choices": [], "usage": reply.usage})
    lines = [f"data: {json.dumps(event)}\n\n" for event in events] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()
