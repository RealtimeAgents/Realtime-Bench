"""Scripted Kimi endpoint for the moonshot API, reached through the official base URL.

The moonshot API sends with the driver's own AsyncOpenAI client (providers/litellm.py
MoonshotDriver, LiteLLMDriver._client) to the catalog's base URL https://api.moonshot.ai/v1
(configs/models/kimi-k3.yaml:5, providers/litellm.py LiteLLMDriver.__init__). The base URL stays
official, so each request is addressed as a run addresses it, and redirect_to_stub() moves every
request to this stub at the HTTP transport.

Replies are server-sent events: reasoning arrives as ``reasoning_content`` deltas, which the
driver streams on the reasoning channel (providers/litellm.py LiteLLMDriver.events,
LiteLLMDriver._delta_reasoning), and usage comes on the finishing chunk's choice instead of a
separate usage chunk; the driver reads it there when a chunk has no top-level usage
(providers/litellm.py LiteLLMDriver._observe_raw).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler

import httpx
import openai
from openai import DefaultAsyncHttpxClient

from .scenario import replies
from .steps import Step
from .stub import CREATED, ChatCompletionsStub, Received, Reply, _error, _pieces

# Request headers that change what the server does; names are compared lowercased. Host is
# covered by the recorded request URL. The others are transport details (Content-Length,
# Connection, Accept-Encoding) or SDK telemetry (User-Agent and X-Stainless-*, among them the
# machine-dependent OS, Arch and Runtime-Version).
COMPARED_HEADERS = ("accept", "authorization", "content-type")


@dataclass(frozen=True)
class ReceivedWithHeaders(Received):
    headers: tuple[tuple[str, str], ...]


class KimiStub(ChatCompletionsStub):
    """ChatCompletionsStub answering with the stream above and keeping each request's headers.

    The moonshot driver always streams (providers/litellm.py ChatProtocol, MoonshotDriver), so every
    request is answered with server-sent events.
    """

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def _next(self, path: str, body: bytes, headers: tuple[tuple[str, str], ...]) -> Reply | None:
        self.requests.append(ReceivedWithHeaders(path, body, headers))
        if not self._replies:
            self.unscripted += 1
            return None
        return self._replies.pop(0)

    def _handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                reply = stub._next(self.path, body, tuple(self.headers.items()))
                if reply is None:
                    # HTTP 400 is FATAL for the shared retry policy (entity/model.py
                    # classify_http_failure).
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


def compared_headers(received: ReceivedWithHeaders) -> dict:
    return {
        name.lower(): value for name, value in received.headers if name.lower() in COMPARED_HEADERS
    }


def _sse(reply: Reply, model: str, index: int) -> bytes:
    head = {
        "id": f"chatcmpl-replay-{index:04d}",
        "object": "chat.completion.chunk",
        "created": CREATED,
        "model": model,
    }

    def chunk(delta: dict, finish_reason: str | None = None, **fields) -> dict:
        choice = {"index": 0, "delta": delta, "finish_reason": finish_reason, **fields}
        return {**head, "choices": [choice]}

    events = [chunk({"role": "assistant", "content": ""})]
    events += [chunk({"reasoning_content": piece}) for piece in _pieces(reply.reasoning)]
    events += [chunk({"content": piece}) for piece in _pieces(reply.content)]
    events.append(chunk({}, "stop", usage=reply.usage))
    lines = [f"data: {json.dumps(event)}\n\n" for event in events] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def kimi_replies(steps: list[Step]) -> list[Reply]:
    """scenario.replies plus a cache-write count, the prompt beyond the cached prefix, which
    kimi_usage reports as 5-minute cache writes (providers/litellm.py kimi_usage)."""
    out = []
    for reply in replies(steps):
        usage = copy.deepcopy(reply.usage)
        details = usage["prompt_tokens_details"]
        details["cache_write_tokens"] = usage["prompt_tokens"] - details["cached_tokens"]
        out.append(replace(reply, usage=usage))
    return out


class StubTransport(httpx.AsyncBaseTransport):
    """Deliver every request to the stub. The request keeps the URL and Host header the client
    built; only the connection goes to 127.0.0.1."""

    def __init__(self, port: int):
        self.urls: list[str] = []  # each request's URL as the client addressed it
        self._port = port
        self._inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        request.url = request.url.copy_with(scheme="http", host="127.0.0.1", port=self._port)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def redirect_to_stub(monkeypatch, port: int) -> StubTransport:
    """Send the driver's requests to the stub while its base URL stays official.

    On its first request the driver builds its AsyncOpenAI client over the HTTP client
    openai.DefaultAsyncHttpxClient(event_hooks=...) returns, with the capture hooks
    (providers/litellm.py LiteLLMDriver._client), and LiteLLM sends through that client
    (litellm/llms/openai/openai.py:381-472). Here that HTTP client also gets this transport, so
    the hooks record every request as in a run. An httpx client given a transport mounts no
    proxies from the environment (httpx/_client.py:1399), so every request goes through the
    transport.
    """
    transport = StubTransport(port)
    monkeypatch.setattr(
        openai,
        "DefaultAsyncHttpxClient",
        lambda **kwargs: DefaultAsyncHttpxClient(transport=transport, **kwargs),
    )
    return transport
