"""Record what each outbound transport sends and receives, for the agent's trace sink.

Every hook reads ``owner._trace`` (set by the driver's ``bind_trace``) at I/O time and hands the
sink references only; hashing, serialization and storage happen on the sink's writer thread.
Hooks only read: what is sent and received is unchanged. Credentials travel in headers, so the two
headers real paths send them in are redacted at capture, and WebSocket handshakes are not
recorded at all.
"""

from __future__ import annotations

import itertools
import time
import weakref

import httpx

from realtime_framework.entity.records import TraceEvent

# Decisions, OpenRouter, AsyncOpenAI and vLLM tokenize send Authorization; Anthropic x-api-key.
SECRET_HEADERS = frozenset({"authorization", "x-api-key"})
REDACTED = "<redacted>"


def redacted_headers(headers: httpx.Headers) -> list[list[str]]:
    return [
        [name, REDACTED if name.lower() in SECRET_HEADERS else value]
        for name, value in headers.multi_items()
    ]


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _request_payload(channel: str, purpose: str, request: httpx.Request) -> dict:
    return {
        "channel": channel,
        "purpose": purpose,
        "exact_wire": True,
        "method": request.method,
        "url": str(request.url),
        "headers": redacted_headers(request.headers),
    }


def _response_payload(channel, purpose, request_seq, response, **fields) -> dict:
    return {
        "channel": channel,
        "purpose": purpose,
        "request_seq": request_seq,
        "status_code": response.status_code,
        "headers": redacted_headers(response.headers),
        "content_encoding": response.headers.get("content-encoding"),
        **fields,
    }


class _RecordingStream(httpx.AsyncByteStream):
    """Pass response chunks through unchanged; record them once, when the response is closed.

    httpx closes the stream after the last chunk, SDKs close it when they stop reading, and the
    drivers close it on interrupt, so error bodies the SDK reads and cut-short streams are covered.
    ``complete`` means the transport reported the end of the body.
    """

    def __init__(self, capture: HttpCapture, request_seq, response):
        self._capture = capture
        self._stream = response.stream
        self._request_seq = request_seq
        self._response = response
        self._chunks: list[bytes] = []
        self._headers_ns = time.monotonic_ns()
        self._first_byte_ns = None
        self._last_byte_ns = None
        self._complete = False
        self._error = None
        self._recorded = False

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                self._last_byte_ns = time.monotonic_ns()
                if self._first_byte_ns is None:
                    self._first_byte_ns = self._last_byte_ns
                self._chunks.append(chunk)
                yield chunk
        except Exception as exc:
            self._error = _error_text(exc)
            raise
        self._complete = True

    async def aclose(self) -> None:
        try:
            await self._stream.aclose()
        finally:
            self.record()

    def record(self) -> None:
        # Closed by its SDK, or recorded at the end of the generation that left it open.
        if self._recorded:
            return
        self._recorded = True
        self._capture.open.discard(self)
        capture = self._capture
        capture.owner._trace.capture(
            TraceEvent.MODEL_RESPONSE_RECEIVED,
            _response_payload(
                capture.channel,
                capture.purpose,
                self._request_seq,
                self._response,
                complete=self._complete,
                error=self._error,
                headers_monotonic_ns=self._headers_ns,
                first_byte_monotonic_ns=self._first_byte_ns,
                last_byte_monotonic_ns=self._last_byte_ns,
            ),
            body=self._chunks,
        )


class HttpCapture:
    """httpx event hooks for one client: each request as sent, each response's raw bytes.

    The request hook runs on the final request right before it is sent, with its body loaded; the
    response hook wraps the response stream before any byte is read.
    """

    def __init__(self, owner, channel: str, purpose: str):
        self.owner = owner
        self.channel = channel
        self.purpose = purpose
        self.open: set[_RecordingStream] = set()
        self._sent: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self.hooks = {"request": [self._on_request], "response": [self._on_response]}

    async def _on_request(self, request: httpx.Request) -> None:
        self._sent[request] = self.owner._trace.capture(
            TraceEvent.MODEL_REQUEST_SENT,
            _request_payload(self.channel, self.purpose, request),
            body=request.content,
        )

    async def _on_response(self, response: httpx.Response) -> None:
        stream = _RecordingStream(self, self._sent.get(response.request), response)
        self.open.add(stream)
        response.stream = stream

    def record_open(self) -> None:
        """Record responses still open when a generation ends: LiteLLM never closes its
        Anthropic stream (its ModelResponseIterator has no aclose)."""
        for stream in list(self.open):
            stream.record()


def traced_post(owner, channel: str, purpose: str):
    """``httpx.post`` that records, after it returns or fails, the request httpx actually sent
    and the response it read."""

    def post(*args, **kwargs):
        try:
            response = httpx.post(*args, **kwargs)
        except httpx.RequestError as exc:
            owner._trace.capture(
                TraceEvent.MODEL_REQUEST_SENT,
                {**_request_payload(channel, purpose, exc.request), "error": _error_text(exc)},
                body=exc.request.content,
            )
            raise
        request_seq = owner._trace.capture(
            TraceEvent.MODEL_REQUEST_SENT,
            _request_payload(channel, purpose, response.request),
            body=response.request.content,
        )
        # The synchronous API has already read and decoded the body.
        owner._trace.capture(
            TraceEvent.MODEL_RESPONSE_RECEIVED,
            _response_payload(channel, purpose, request_seq, response, complete=True, error=None),
            body=response.content,
        )
        return response

    return post


class LoggerCapture:
    """Record the generic LiteLLM API from LiteLLM's logger callback.

    That API has no HTTP client to hook (LiteLLM builds its own), so the request is LiteLLM's
    final request dict and the reply its logged response text: not the bytes on the wire
    (``exact_wire`` is false). The callback also carries the plaintext key (``api_key`` and the
    logged request headers); neither is copied. The API base is LiteLLM's masked copy.
    """

    channel = "litellm.logger_fn"

    def __init__(self, owner):
        self.owner = owner
        self.request_seq = None

    def __call__(self, details: dict) -> None:
        kind = details.get("log_event_type")
        if kind == "pre_api_call":
            self.request_seq = self.owner._trace.capture(
                TraceEvent.MODEL_REQUEST_SENT,
                {
                    "channel": self.channel,
                    "purpose": "generation",
                    "exact_wire": False,
                    "method": "POST",
                    "url": (details.get("litellm_params") or {}).get("api_base"),
                    "headers": None,
                },
                body=(details.get("additional_args") or {}).get("complete_input_dict"),
            )
        elif kind == "post_api_call":
            # LiteLLM logs the provider reply as text (it dumps a dict reply to JSON first).
            self.owner._trace.capture(
                TraceEvent.MODEL_RESPONSE_RECEIVED,
                {
                    "channel": self.channel,
                    "purpose": "generation",
                    "request_seq": self.request_seq,
                    "exact_wire": False,
                    "status_code": None,
                    "complete": True,
                    "error": None,
                },
                body=details.get("original_response"),
            )


def _session_capture(connection, direction: str, message=None, **fields) -> int | None:
    payload = {
        "channel": connection.trace_channel,
        "connection_id": connection.trace_connection_id,
        "direction": direction,
        **fields,
    }
    return connection.trace_owner._trace.capture(
        TraceEvent.SESSION_MESSAGE, payload, body=message, classify=connection.trace_classify
    )


def _connection_class(base, owner, channel: str, classify):
    numbers = itertools.count(1)

    class Recording(base):
        trace_owner = owner
        trace_channel = channel
        trace_classify = staticmethod(classify)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace_connection_id = f"{channel}-{next(numbers)}"

    return Recording


def sync_ws_connection(owner, channel: str, classify):
    """A ``websockets.sync`` client connection class that records every message.

    A client message is captured just before it is written, so it always precedes its
    acknowledgement in capture order; a failed write adds a lifecycle record whose state is
    unknown. A server message is captured when the existing code receives it.
    """
    from websockets.sync.client import ClientConnection

    class Connection(_connection_class(ClientConnection, owner, channel, classify)):
        def handshake(self, *args, **kwargs):
            try:
                return super().handshake(*args, **kwargs)
            except Exception as exc:
                _session_capture(
                    self, "lifecycle", message_type="handshake_failed", error=_error_text(exc)
                )
                raise

        def send(self, message, text=None):
            sequence = _session_capture(self, "client", message)
            try:
                return super().send(message, text)
            except Exception as exc:
                _session_capture(
                    self,
                    "lifecycle",
                    message_type="send_failed",
                    state="unknown",
                    refers_to=sequence,
                    error=_error_text(exc),
                )
                raise

        def recv(self, timeout=None, decode=None):
            message = super().recv(timeout, decode)
            _session_capture(self, "server", message)
            return message

    return Connection


def async_ws_connection(owner, channel: str, classify):
    """The ``websockets.asyncio`` counterpart of :func:`sync_ws_connection`."""
    from websockets.asyncio.client import ClientConnection

    class Connection(_connection_class(ClientConnection, owner, channel, classify)):
        async def handshake(self, *args, **kwargs):
            try:
                return await super().handshake(*args, **kwargs)
            except Exception as exc:
                _session_capture(
                    self, "lifecycle", message_type="handshake_failed", error=_error_text(exc)
                )
                raise

        async def send(self, message, text=None):
            sequence = _session_capture(self, "client", message)
            try:
                return await super().send(message, text)
            except Exception as exc:
                _session_capture(
                    self,
                    "lifecycle",
                    message_type="send_failed",
                    state="unknown",
                    refers_to=sequence,
                    error=_error_text(exc),
                )
                raise

        async def recv(self, decode=None):
            message = await super().recv(decode)
            _session_capture(self, "server", message)
            return message

    return Connection
