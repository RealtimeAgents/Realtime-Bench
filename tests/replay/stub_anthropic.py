"""Scripted Anthropic Messages endpoint on 127.0.0.1 with an OS-assigned port.

The anthropic_messages API sends through LiteLLM's Anthropic provider on LiteLLM's own
AsyncHTTPHandler (providers/litellm.py LiteLLMDriver._client, LiteLLMDriver._transport_kwargs),
which posts to ``{api_base}/v1/messages`` (litellm/main.py:2914-2927). Each request is answered with
the next scripted reply as server-sent events in the order the Messages API streams them:
message_start carrying the input usage, a thinking block when the step has reasoning, a text block
when it has visible output, message_delta carrying the stop reason and the output usage, and
message_stop. The driver takes usage only from message_start and message_delta (providers/litellm.py
LiteLLMDriver._observe_raw, LiteLLMDriver.events). Every request is kept in arrival order with its
path, headers and raw body.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

from .steps import Step
from .stub import _pieces

# Request headers that change how the server treats a request: the API version, beta features
# and the credential scheme. Names are compared lowercased; the Host header (the stub's port), the
# user agent and the transport's connection headers are not compared.
COMPARED_HEADERS = ("anthropic-beta", "anthropic-version", "authorization", "x-api-key")
SIGNATURE = "c2lnbmF0dXJlLXJlcGxheQ=="  # an opaque thinking-block signature; never sent back
UNCACHED_INPUT_TOKENS = 20  # input after the newest cache breakpoint


@dataclass(frozen=True)
class Reply:
    content: str
    start_usage: dict  # message_start.message.usage
    delta_usage: dict  # message_delta.usage
    reasoning: str = ""  # streamed as a summarized thinking block


@dataclass(frozen=True)
class Received:
    path: str
    headers: tuple[tuple[str, str], ...]
    body: bytes


def compared_headers(received: Received) -> dict:
    return {name.lower(): v for name, v in received.headers if name.lower() in COMPARED_HEADERS}


def replies(steps: list[Step]) -> list[Reply]:
    """The stub's reply for each step, with usage split as the Messages API reports it.

    The prompt totals scenario.replies' 1500 + 260 k tokens: a fixed uncached remainder, 1280
    tokens read from the cache after the first decision, and the rest written to the 5-minute
    cache that an ephemeral breakpoint without a ttl creates. message_start also carries the
    placeholder output count the API sends before generation; message_delta carries the final
    output count, which includes thinking, and the thinking share when the step has reasoning.
    The fields are the ones LiteLLM parses (litellm/types/llms/anthropic.py:559-564, 619-641;
    litellm/llms/anthropic/chat/transformation.py:2330-2338, 2424-2436) and the driver merges.
    """
    out = []
    for k, step in enumerate(steps):
        thinking_tokens = len(step.reasoning) // 4
        prompt_tokens = 1500 + 260 * k
        cache_read = 0 if k == 0 else 1280
        cache_write = prompt_tokens - UNCACHED_INPUT_TOKENS - cache_read
        start_usage = {
            "input_tokens": UNCACHED_INPUT_TOKENS,
            "cache_creation_input_tokens": cache_write,
            "cache_read_input_tokens": cache_read,
            "cache_creation": {
                "ephemeral_5m_input_tokens": cache_write,
                "ephemeral_1h_input_tokens": 0,
            },
            "output_tokens": 1,
        }
        delta_usage = {"output_tokens": 4 + len(step.reply) // 4 + thinking_tokens}
        if step.reasoning:
            delta_usage["output_tokens_details"] = {"thinking_tokens": thinking_tokens}
        out.append(Reply(step.reply, start_usage, delta_usage, step.reasoning))
    return out


class AnthropicMessagesStub:
    def __init__(self, replies: list[Reply]):
        self._replies = list(replies)
        self.requests: list[Received] = []
        self.unscripted = 0
        self._server = HTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        """The API base without a path; LiteLLM appends /v1/messages (litellm/main.py:2925-2927)."""
        return f"http://127.0.0.1:{self._server.server_address[1]}"

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

    def _next(self, received: Received) -> Reply | None:
        self.requests.append(received)
        if not self._replies:
            self.unscripted += 1
            return None
        return self._replies.pop(0)

    def _handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                reply = stub._next(Received(self.path, tuple(self.headers.items()), body))
                if reply is None:
                    # HTTP 400 is FATAL for the shared retry policy (entity/model.py
                    # classify_http_failure), so an extra request fails its decision at once instead
                    # of being retried.
                    self._send(400, "application/json", _error("request beyond the script"))
                    return
                model = json.loads(body)["model"]
                self._send(200, "text/event-stream", _sse(reply, model, len(stub.requests)))

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
    return json.dumps(
        {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    ).encode()


def _sse(reply: Reply, model: str, index: int) -> bytes:
    message = {
        "id": f"msg_replay_{index:04d}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": reply.start_usage,
    }
    blocks = []
    if reply.reasoning:
        deltas = [{"type": "thinking_delta", "thinking": p} for p in _pieces(reply.reasoning)]
        deltas.append({"type": "signature_delta", "signature": SIGNATURE})
        blocks.append(({"type": "thinking", "thinking": "", "signature": ""}, deltas))
    if reply.content:
        deltas = [{"type": "text_delta", "text": p} for p in _pieces(reply.content)]
        blocks.append(({"type": "text", "text": ""}, deltas))
    events = [{"type": "message_start", "message": message}]
    for position, (block, deltas) in enumerate(blocks):
        events.append({"type": "content_block_start", "index": position, "content_block": block})
        events += [{"type": "content_block_delta", "index": position, "delta": d} for d in deltas]
        events.append({"type": "content_block_stop", "index": position})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": reply.delta_usage,
        }
    )
    events.append({"type": "message_stop"})
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
