"""Scripted OpenAI chat completions endpoint on 127.0.0.1 with an OS-assigned port.

The chat_completions API reaches it through LiteLLM's OpenAI provider with an AsyncOpenAI client
(providers/litellm.py:385-401, 589-593), which posts to ``{base_url}/chat/completions``. Each
request is answered with the next scripted reply: a JSON body when the request sets
``stream: false``, otherwise server-sent events ending in a usage chunk and ``[DONE]``. Every raw
request body is kept in arrival order.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

CREATED = 1_760_000_000  # fixed response timestamp
STREAM_PIECE_CHARS = 12  # visible and reasoning text arrive in pieces of this size


@dataclass(frozen=True)
class Reply:
    content: str
    usage: dict
    reasoning: str = ""  # sent as reasoning_content, as OpenAI-compatible reasoning servers do


@dataclass(frozen=True)
class Received:
    path: str
    body: bytes


class ChatCompletionsStub:
    def __init__(self, replies: list[Reply]):
        self._replies = list(replies)
        self.requests: list[Received] = []
        self.unscripted = 0
        self._server = HTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    @property
    def unused_replies(self) -> int:
        return len(self._replies)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def _next(self, path: str, body: bytes) -> Reply | None:
        self.requests.append(Received(path, body))
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
                if reply is None:
                    # HTTP 400 is FATAL for the shared retry policy (entity/model.py:369-377), so
                    # an extra request fails its decision at once instead of being retried.
                    self._send(400, "application/json", _error("request beyond the script"))
                    return
                request = json.loads(body)
                index = len(stub.requests)
                if request.get("stream"):  # a non-streaming body carries no "stream" field
                    self._send(200, "text/event-stream", _sse(reply, request["model"], index))
                else:
                    self._send(200, "application/json", _completion(reply, request["model"], index))

            def _send(self, status: int, content_type: str, payload: bytes):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

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
