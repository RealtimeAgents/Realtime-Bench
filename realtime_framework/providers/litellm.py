from __future__ import annotations

# LiteLLM
"""One driver for every HTTP chat API; each registered API is a subclass declaring its settings.

LiteLLM supplies the provider transports. Each subclass declares, for one registered API, its
protocol (request and response shape), deployment (gateway, address, key, route) and compat flags
(provider restrictions), preserving the benchmark's request shaping and accounting rules: pinned
OpenRouter routes, explicit cache breakpoints, provider quirks, and provider-reported usage only.
The pinned SDK can synthesize usage at stream completion; the driver observes raw provider
payloads before LiteLLM transforms them, and only those counters reach the token clock.
"""

import asyncio as asyncio
import base64 as base64
import copy as copy
import json as json
import os as os
import re as re
import threading as threading
import uuid as uuid
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from dataclasses import replace as replace
from importlib.metadata import version as version
from typing import ClassVar as ClassVar
from urllib.parse import urlsplit as urlsplit

from realtime_framework.clock import SystemClock
from realtime_framework.entity.events import GenerationBoundary
from realtime_framework.entity.events import ProviderHistoryState
from realtime_framework.entity.events import TextDelta
from realtime_framework.entity.events import UsageSnapshot
from realtime_framework.entity.model import FATAL
from realtime_framework.entity.model import TRANSIENT
from realtime_framework.entity.model import RetryCancelled
from realtime_framework.entity.model import TransportFailure
from realtime_framework.entity.model import classify_http_failure
from realtime_framework.entity.model import exception_chain
from realtime_framework.entity.model import retry_after_seconds
from realtime_framework.entity.model import retry_cancel_event
from realtime_framework.entity.model import Compat
from realtime_framework.entity.model import Deployment
from realtime_framework.entity.model import GenerationHandle
from realtime_framework.entity.model import GenerationRequest
from realtime_framework.entity.model import RealtimeCapabilities
from realtime_framework.entity.model import StatelessEncoder
from realtime_framework.entity.records import TraceEvent
from realtime_framework.providers.capture import HttpCapture
from realtime_framework.providers.capture import LoggerCapture
from realtime_framework.providers.capture import replayed_reasoning
from realtime_framework.providers.capture import response_statuses
from realtime_framework.providers.capture import traced_post
from realtime_framework.providers.context import context_status
from realtime_framework.providers.context import estimated_request_tokens

LITELLM_VERSION = "1.106.1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# reasoning_effort values anthropic_thinking maps.
ANTHROPIC_REASONING_EFFORTS = (None, "fixed", "none", *ANTHROPIC_EFFORT_LEVELS)
_REQUEST_GATE = threading.Lock()
_LAST_REQUEST_AT = [0.0]


# Helpers
def import_litellm():
    """Import LiteLLM without the two things its import does by default: load the first .env
    found above its package directory (litellm/__init__.py:29-30, skipped unless LITELLM_MODE is
    DEV) and replace its bundled model map with one fetched over the network
    (litellm/litellm_core_utils/get_model_cost_map.py:646, skipped when
    LITELLM_LOCAL_MODEL_COST_MAP is true). LiteLLM reads both settings at its first import and
    checks requests against the map (whether a Moonshot model takes reasoning_effort, for one), so
    a run then depends only on its own environment and the locked version. Nor does it print its
    help banner as it maps an error (litellm_core_utils/exception_mapping_utils.py
    exception_type), the abort's close of a request included: the trace records each failed
    attempt with its error."""
    os.environ["LITELLM_MODE"] = "PRODUCTION"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"
    import litellm as litellm

    litellm.suppress_debug_info = True
    return litellm


def pace_request(owner) -> None:
    """Apply the optional process-wide minimum interval between provider requests.

    ``owner`` is the driver. A wait runs on its clock and is written to its trace before it
    starts, as the session encoders record theirs (providers/session_lifecycle.py _record_wait).
    Only a wait reads the trace, which the harness binds as it builds the driver
    (harnesses/default/harness.py _setup_backend). Like RequestRetry's backoff, the wait ends
    once the generation is abandoned (the enclosing retry_cancellation), and RetryCancelled stops
    the request unsent.
    """
    interval = float(os.environ.get("STREAM_MIN_REQUEST_INTERVAL", "0"))
    if interval <= 0:
        return
    clock = owner.clock
    with _REQUEST_GATE:
        wait = _LAST_REQUEST_AT[0] + interval - clock.time()
        if wait > 0:
            owner._trace.capture(
                TraceEvent.SESSION_WAIT,
                {
                    "wait": "request_pacing",
                    "seconds": wait,
                    "until_monotonic_ns": clock.monotonic_ns() + round(wait * 1e9),
                },
            )
            if clock.wait(retry_cancel_event(), wait):
                raise RetryCancelled("generation abandoned during the request_pacing wait")
        _LAST_REQUEST_AT[0] = clock.time()


def object_dict(value) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    try:
        return dict(value)
    except (TypeError, ValueError):
        return None


def validate_endpoint(url: str | None) -> str:
    parsed = urlsplit(url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Set a valid http(s) API base URL, including its API prefix (usually /v1)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "The base URL must not contain credentials, query parameters, or fragments"
        )
    return str(url).rstrip("/")


# Usage
def kimi_usage(usage: dict) -> dict:
    """Expose Kimi's default 5m cache and undivided input to cost accounting."""
    usage = copy.deepcopy(usage)
    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        return usage
    prompt_tokens = usage.get("prompt_tokens")
    if isinstance(prompt_tokens, int):
        # Kimi reports a single input total for text and image tokens, priced identically.
        details.setdefault("text_tokens", prompt_tokens)
    cached = details.get("cached_tokens")
    if isinstance(cached, int):
        details.setdefault("cached_tokens_details", {"text_tokens": cached})
    cache_write = details.get("cache_write_tokens")
    if isinstance(cache_write, int):
        details.setdefault(
            "cache_write_tokens_details",
            {
                "ephemeral_5m_tokens": cache_write,
                "ephemeral_1h_tokens": 0,
            },
        )
    usage["prompt_tokens_details"] = details
    output_details = usage.get("completion_tokens_details") or {}
    if isinstance(output_details, dict):
        # K3 Chat Completions returns text, with no audio output modality.
        output_details.setdefault("audio_tokens", 0)
        usage["completion_tokens_details"] = output_details
    return usage


def _usage_field(usage, name: str) -> int | None:
    if usage is None:
        return None
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _usage_detail_field(usage, container: str, name: str) -> int | None:
    if usage is None:
        return None
    details = usage.get(container) if isinstance(usage, dict) else getattr(usage, container, None)
    return _usage_field(details, name) if details is not None else None


def _first_reported(*values: int | None, default: int = 0) -> int:
    for value in values:
        if value is not None:
            return value
    return default


def anthropic_usage(start_usage, delta_usage, *, stop_reason: str | None = None) -> dict:
    """Merge ``message_start`` and terminal ``message_delta`` usage into the harness usage dict.

    Anthropic reports ``input_tokens`` excluding cache reads and writes; the harness (like OpenAI
    usage) expects the complete prompt count with the cached share nested under
    ``input_token_details.cached_tokens``. Anthropic bills text and image input at the same rate
    and never splits them, so the undivided count is reported as ``text_tokens`` to let the
    catalog rate card reconcile exactly. ``output_tokens`` already includes thinking; when the API
    reports ``output_tokens_details.thinking_tokens`` it is exposed as the reasoning split, otherwise
    the split stays unknown rather than being invented.
    """
    uncached_input = _first_reported(
        _usage_field(start_usage, "input_tokens"),
        _usage_field(delta_usage, "input_tokens"),
    )
    cache_read = _first_reported(
        _usage_field(delta_usage, "cache_read_input_tokens"),
        _usage_field(start_usage, "cache_read_input_tokens"),
    )
    cache_write = _first_reported(
        _usage_field(delta_usage, "cache_creation_input_tokens"),
        _usage_field(start_usage, "cache_creation_input_tokens"),
    )
    cache_write_5m_reported = _first_reported(
        _usage_detail_field(delta_usage, "cache_creation", "ephemeral_5m_input_tokens"),
        _usage_detail_field(start_usage, "cache_creation", "ephemeral_5m_input_tokens"),
        default=None,
    )
    cache_write_1h_reported = _first_reported(
        _usage_detail_field(delta_usage, "cache_creation", "ephemeral_1h_input_tokens"),
        _usage_detail_field(start_usage, "cache_creation", "ephemeral_1h_input_tokens"),
        default=None,
    )
    if cache_write_5m_reported is None and cache_write_1h_reported is None:
        # The driver sets an explicit ephemeral breakpoint without a ttl, whose documented
        # default is 5 minutes. Older responses may omit the newer ttl breakdown.
        cache_write_5m = cache_write
        cache_write_1h = 0
        cache_write_details_source = "derived:request_cache_control_default_5m"
    else:
        cache_write_5m = cache_write_5m_reported or 0
        cache_write_1h = cache_write_1h_reported or 0
        if cache_write_5m + cache_write_1h != cache_write:
            raise RuntimeError(
                "Anthropic cache-creation ttl details do not reconcile with the total"
            )
        cache_write_details_source = "provider_usage.cache_creation"
    output_tokens = _first_reported(
        _usage_field(delta_usage, "output_tokens"),
        _usage_field(start_usage, "output_tokens"),
    )
    thinking_tokens = _first_reported(
        _usage_detail_field(delta_usage, "output_tokens_details", "thinking_tokens"),
        _usage_detail_field(start_usage, "output_tokens_details", "thinking_tokens"),
        default=None,
    )
    # No audio modality exists on this transport; stating it lets billing derive the text output
    # count. The thinking split is reported only when the provider reported it.
    output_details = {"audio_tokens": 0}
    if thinking_tokens is not None:
        output_details["reasoning_tokens"] = thinking_tokens
        output_details["thinking_tokens"] = thinking_tokens
    input_tokens = uncached_input + cache_read + cache_write
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "uncached_input_tokens": uncached_input,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
        "input_token_details": {
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "cache_write_tokens_details": {
                "ephemeral_5m_tokens": cache_write_5m,
                "ephemeral_1h_tokens": cache_write_1h,
            },
            "cache_write_tokens_details_source": cache_write_details_source,
            "text_tokens": input_tokens,
            "cached_tokens_details": {"text_tokens": cache_read},
        },
        "output_token_details": output_details,
        "stop_reason": stop_reason,
        "output_accounting": "anthropic_messages_output_tokens_include_thinking_v2",
        "raw_native_usage": {
            "message_start": copy.deepcopy(start_usage),
            "message_delta": copy.deepcopy(delta_usage),
        },
    }


def anthropic_thinking(reasoning_effort: str | None) -> dict:
    """Map the catalog's model-native ``reasoning_effort`` onto Messages API parameters.

    ``none`` explicitly disables thinking and sends no effort. ``fixed``, a model whose reasoning
    is not configurable, sends neither parameter. Every other accepted value keeps adaptive
    thinking and sets ``output_config.effort``. Omitting the effort keeps the provider's default
    adaptive thinking. ``display`` only controls whether a thinking summary is returned for the
    audit trail; billing and the token clock are unaffected.
    """
    adaptive = {"type": "adaptive", "display": "summarized"}
    if reasoning_effort is None:
        return {"thinking": adaptive}
    if reasoning_effort == "fixed":
        return {}
    if reasoning_effort == "none":
        return {"thinking": {"type": "disabled"}}
    if reasoning_effort in ANTHROPIC_EFFORT_LEVELS:
        return {"thinking": adaptive, "output_config": {"effort": reasoning_effort}}
    raise ValueError(
        f"Anthropic reasoning_effort must be one of {('none', 'fixed', *ANTHROPIC_EFFORT_LEVELS)}, "
        f"got {reasoning_effort!r}"
    )


def _strip_image_detail(messages: list[dict]) -> None:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    part["image_url"].pop("detail", None)


def _mark_cache_breakpoint(messages: list[dict]) -> None:
    """Cache through the newest message; only unchanged earlier history is a reusable prefix."""
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
            message["content"] = content
        if isinstance(content, list) and content:
            content[-1]["cache_control"] = {"type": "ephemeral"}
            break


# Chat protocol mapping
# The canonical request's neutral roles (entity/request.py Role) as chat roles.
CHAT_ROLES = {"instruction": "system", "input": "user", "model": "assistant"}


def frame_text(prefix: str, prompt: str) -> str:
    """A session frame's text: the pending execution feedback, then the frame prompt."""
    return f"{prefix}\n{prompt}" if prefix else prompt


def chat_messages(request) -> tuple[dict, ...]:
    """The canonical request as the OpenAI-format message list, before the compat flags apply.

    Luna's size estimate counts it and the trajectory summary shows it. The ledger, at index 1,
    takes the role the policy gives it. Live tool exchanges have no message form here.
    """
    messages = []
    prefix = ""
    for item in request.items:
        block = item.block
        if block.kind == "system_instruction":
            messages.append({"role": "system", "content": block.text})
        elif block.kind == "action_ledger":
            messages.append(
                {
                    "role": CHAT_ROLES[item.intent.role],
                    "content": request.policy.texts.ledger_header + "\n" + block.text,
                }
            )
        elif item.intent.position == "frame_prefix":
            prefix = block.text
        elif block.kind == "frame":
            if block.folded:
                messages.append({"role": "user", "content": frame_text(prefix, block.text)})
            else:
                encoded = base64.b64encode(block.image.data).decode("ascii")
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": frame_text(prefix, block.text)},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block.image.mime_type};base64,{encoded}",
                                    "detail": block.image.detail,
                                },
                            },
                        ],
                    }
                )
            prefix = ""
        elif block.kind == "model_output":
            message = {"role": "assistant", "content": block.text}
            if block.reasoning:
                message["reasoning_content"] = block.reasoning
            messages.append(message)
        elif block.kind == "execution_feedback":
            messages.append({"role": "user", "content": block.text})
    return tuple(messages)


# Per-API declarations
_STREAMING_CHAT = RealtimeCapabilities(
    protocol_family="openai_chat",
    roles=("instruction", "input", "model"),
    feedback_placements=("separate_message",),
    server_context="none",
)


@dataclass(frozen=True)
class ChatProtocol:
    """How one API's requests are built and its responses parsed."""

    # LiteLLM provider prefix for model_name; None means model_name is already provider/model.
    sdk_provider: str | None
    # "httpx": LiteLLM's own handler; "openai": an AsyncOpenAI client; None: SDK default.
    client: str | None = "httpx"
    # None follows ModelConfig.chat_stream / chat_include_usage.
    stream: bool | None = True
    include_usage: bool | None = True
    # "reasoning_effort": top-level parameter; "openrouter": extra_body.reasoning when a route is
    # pinned, else top-level; "anthropic_thinking": Messages API thinking and output_config.
    reasoning_param: str = "reasoning_effort"
    usage: str = "raw"  # "raw" | "kimi" | "anthropic"
    # Keep streamed tool-call deltas as visible text (complete responses always keep them).
    surface_tool_calls: bool = False


@dataclass(frozen=True, kw_only=True)
class ChatDeployment(Deployment):
    """Where one API's requests go: gateway, address, key and route."""

    gateway: str
    default_base_url: str | None = None
    base_url_env: str | None = None
    # OpenRouter session id, provider pin, attribution headers and response route checks.
    pin_route: bool = False
    require_api_key: bool = True


@dataclass(frozen=True)
class ChatCompat(Compat):
    """Named restrictions and special requirements of one API."""

    # Cache through the newest message on every request. Elsewhere a model whose provider caches
    # only at explicit breakpoints (Anthropic's, Gemini's behind OpenRouter) sets the
    # cache_breakpoint provider option its API declares; OpenAI caches implicitly.
    cache_breakpoint: bool = False
    strip_image_detail: bool = False
    # Each request sends the decision cap, else the declared native limit, else omits the
    # parameter (provider default); an API that requires it (output_limit_required) refuses an
    # undeclared limit. The name is the request field on the wire, which the run configuration
    # records.
    output_limit_param: str = "max_tokens"


class _AuditedSDKStream:
    """Observe OpenAI-compatible SDK chunks before LiteLLM's usage reconstruction."""

    def __init__(self, stream, observe):
        self.stream = stream
        self.iterator = stream.__aiter__()
        self.observe = observe

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await anext(self.iterator)
        self.observe(chunk.model_dump(exclude_unset=True))
        return chunk

    async def aclose(self):
        await self.stream.close()


class RequestEvidence:
    """One request's evidence, from its send to the end of its events: its wire record, and the
    usage and route the provider itself reported.

    The pinned LiteLLM rebuilds usage, and answers some refused requests with a reply of its own
    (complete), so the driver reports usage and route only as this object saw them in the
    provider's reply: a stream's raw chunks, observed before LiteLLM parses them, or a complete
    reply's body, from LiteLLM's logger callback, which must back the output count of its
    normalized usage. The wire record is the HttpCapture of the client that sent the request; the
    generic API, which has no client to hook, is recorded from the logger callback instead
    (LoggerCapture).
    """

    def __init__(self, driver: LiteLLMDriver, kwargs: dict):
        self.driver = driver
        self.kwargs = kwargs
        self.route = driver._backend_route()
        if driver.protocol.client is None:
            self.capture, self.logger = None, LoggerCapture(driver)
        else:
            # Kept as the request goes: an abort can replace the client, and its capture, before
            # the request or its events end.
            self.capture, self.logger = driver._http_capture, None
        self.interrupted = False
        # A stream: LiteLLM's response, and what its raw chunks reported.
        self.response = None
        self.raw_usage = None
        self.usage_revision = 0
        self.taken_revision = 0
        self.raw_route = None
        self.anthropic_start = None
        self.anthropic_delta = None
        self.anthropic_message = None
        self.stop_reason = None
        # A complete reply: its message, the provider's body and the usage that body backs.
        self.reply = None
        self.raw_reply = {}
        self.reply_usage = None

    async def stream(self) -> None:
        """Send the request for a stream whose raw chunks are observed before LiteLLM parses
        them. A hooked client records with the request the past reasonings it carries back
        (capture.py replayed_reasoning)."""
        with replayed_reasoning(self.kwargs["messages"]):
            response = await self.driver._sdk.acompletion(**self.kwargs)
        try:
            # Fail closed if LiteLLM no longer exposes the provider payload before parsing it.
            raw_stream = response.completion_stream
            parse = getattr(raw_stream, "chunk_parser", None)
            if callable(parse):

                def parse_observed(chunk):
                    self._observe(chunk)
                    return parse(chunk)

                raw_stream.chunk_parser = parse_observed
            else:
                from openai import AsyncStream as AsyncStream

                if not isinstance(raw_stream, AsyncStream):
                    raise RuntimeError("LiteLLM stream has no auditable provider SSE parser")
                response.completion_stream = _AuditedSDKStream(raw_stream, self._observe)
        except BaseException:
            await response.aclose()
            raise
        with self.driver._lock:
            self.driver._responses[id(response)] = response
        self.response = response

    def _observe(self, chunk: dict) -> None:
        if self.driver.protocol.usage == "anthropic":
            kind = chunk.get("type")
            if kind == "message_start":
                message = chunk.get("message") or {}
                self.anthropic_start = copy.deepcopy(message.get("usage"))
                self.anthropic_message = {"model": message.get("model"), "id": message.get("id")}
            elif kind == "message_delta":
                self.anthropic_delta = copy.deepcopy(chunk.get("usage"))
                self.stop_reason = (chunk.get("delta") or {}).get("stop_reason")
            return
        raw_usage = chunk.get("usage")
        if raw_usage is None:
            choices = chunk.get("choices") or []
            raw_usage = choices[0].get("usage") if choices else None
        if isinstance(raw_usage, dict):
            self.raw_usage = copy.deepcopy(raw_usage)
            self.usage_revision += 1
        route = chunk.get("openrouter_metadata")
        if route is not None:
            self.raw_route = copy.deepcopy(route)

    def check_route(self) -> None:
        """Keep the route OpenRouter reported for a pinned request, and refuse a provider other
        than the pinned one."""
        if not self.driver.deployment.pin_route:
            return
        route = self._route_metadata(self.raw_route)
        if route is None:
            return
        self.route["response_metadata"] = route
        selected = route.get("selected_provider")
        expected = self.driver.config.upstream_provider
        if selected is not None and expected is not None and self._provider_slug(selected) != expected:
            raise RuntimeError(f"OpenRouter selected {selected!r}; expected pinned {expected!r}")

    @staticmethod
    def _provider_slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")

    @staticmethod
    def _route_metadata(value) -> dict | None:
        value = object_dict(value)
        if value is None:
            return None
        endpoints = value.get("endpoints") or {}
        available = endpoints.get("available") or []
        selected = [endpoint for endpoint in available if endpoint.get("selected")]
        endpoint = selected[0] if len(selected) == 1 else None
        return {
            "requested_model": value.get("requested"),
            "strategy": value.get("strategy"),
            "attempt": value.get("attempt"),
            "selected_provider": endpoint.get("provider") if endpoint else None,
            "selected_model": endpoint.get("model") if endpoint else None,
            "selected_endpoint_count": len(selected),
        }

    def revised_usage(self) -> dict | None:
        """The stream's usage as its raw chunks last reported it, normalized, when one revised it
        since the last call; never LiteLLM's reconstructed or estimated usage. An Anthropic
        stream has none here (terminal_usage)."""
        if self.usage_revision == self.taken_revision:
            return None
        self.taken_revision = self.usage_revision
        return self.driver._normalize_usage(self.raw_usage)

    def terminal_usage(self) -> dict | None:
        """An Anthropic stream's usage. Only the terminal message_delta carries authoritative
        output usage; an interrupted stream has no complete count and reports none."""
        if self.anthropic_delta is None:
            return None
        return anthropic_usage(
            self.anthropic_start, self.anthropic_delta, stop_reason=self.stop_reason
        )

    def boundary_metadata(self) -> dict:
        if self.driver.protocol.usage != "anthropic":
            return {"backend_route": self.route}
        message = self.anthropic_message or {}
        self.route.update(response_model=message.get("model"), response_id=message.get("id"))
        return {"stop_reason": self.stop_reason, "backend_route": dict(self.route)}

    async def interrupt(self) -> None:
        self.interrupted = True
        if self.response is not None:
            await self.response.aclose()
            with self.driver._lock:
                self.driver._responses.pop(id(self.response), None)

    async def finish(self) -> None:
        """Close the stream, and record the responses still open on its client's capture."""
        try:
            await self.driver.clock.wait_for(self.response.aclose(), 1.0)
        finally:
            with self.driver._lock:
                self.driver._responses.pop(id(self.response), None)
            self.capture.record_open()

    async def complete(self) -> None:
        """Send the request for a complete reply, keeping its message and the usage the
        provider's body backs."""
        kwargs = self.kwargs
        kwargs["logger_fn"] = self._logged
        with response_statuses() as statuses, replayed_reasoning(kwargs["messages"]):
            response = await self.driver._sdk.acompletion(**kwargs)
        if not response.choices:
            raise RuntimeError("LiteLLM returned no response choices")
        self.reply = response.choices[0].message.model_dump(exclude_none=True)
        # LiteLLM answers a 400 saying the output limit was reached with an empty completion of
        # its own, whose prompt count it estimated (llms/openai/common_utils.py
        # build_output_token_limit_response): a reply the provider refused reported no usage. Only
        # a hooked client sees the status; the generic API takes the estimate.
        refused = any(status >= 400 for status in statuses)
        self.reply_usage = None if refused else self._reported_usage(response)

    def _logged(self, details: dict) -> None:
        if self.logger is not None:
            self.logger(details)
        if details.get("log_event_type") != "post_api_call":
            return
        value = details.get("original_response")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                return
        if isinstance(value, dict) and ("usage" in value or "usageMetadata" in value):
            # Do not retain callback credentials, request headers, or prompt logging fields.
            self.raw_reply.update(copy.deepcopy(value))

    def _reported_usage(self, response):
        """Accept normalized counters only when backed by a raw provider output count."""
        raw = self.raw_reply
        raw_usage = raw.get("usage") or raw.get("usageMetadata") or {}
        output = next(
            (
                raw_usage[k]
                for k in ("completion_tokens", "output_tokens")
                if type(raw_usage.get(k)) is int
            ),
            None,
        )
        if output is None and type(raw_usage.get("candidatesTokenCount")) is int:
            output = raw_usage["candidatesTokenCount"] + (raw_usage.get("thoughtsTokenCount") or 0)
        normalized = getattr(response, "usage", None)
        if normalized is None or output is None:
            return None
        usage = normalized.model_dump(exclude_none=True)
        if usage.get("completion_tokens") != output:
            raise RuntimeError("LiteLLM output usage disagrees with the provider response")
        usage.pop("cost", None)
        if type(raw_usage.get("cost")) in (int, float):
            usage["cost"] = raw_usage["cost"]
        return usage


# Driver
class LiteLLMDriver(StatelessEncoder):
    """Stateless HTTP driver: canonical history is rendered into every request.

    Consequently there is no provider-side conversation to repair after a wall-clock early return.
    Each registered API (ModelConfig.adapter_type) is a subclass declaring its capabilities,
    protocol, deployment and compat flags.
    """

    capabilities: RealtimeCapabilities
    protocol: ChatProtocol
    deployment: ChatDeployment
    compat: ChatCompat
    model_defaults: ClassVar[dict[str, dict]] = {}

    def __init__(self, config, *, clock=None):
        self.config = config
        self.clock = clock or SystemClock()
        deployment = self.deployment
        if deployment.gateway == "openai_compatible":
            self.base_url = validate_endpoint(config.api_base_url)
        else:
            self.base_url = (
                config.api_base_url
                or (os.environ.get(deployment.base_url_env) if deployment.base_url_env else None)
                or deployment.default_base_url
            )
        self._sdk = self._load_sdk()
        self.api_key = self._resolve_key()
        self._http = None
        # Streams still open, which close() ends; each RequestEvidence adds and removes its own.
        self._responses: dict[int, object] = {}
        self._lock = threading.Lock()
        self._session_id = "rtb-" + uuid.uuid4().hex

    @staticmethod
    def _load_sdk():
        # Import lazily; constructing a native realtime driver does not load LiteLLM.
        litellm = import_litellm()
        if version("litellm") != LITELLM_VERSION:
            raise RuntimeError(
                f"Use litellm=={LITELLM_VERSION}; raw usage auditing is version-validated"
            )
        return litellm

    def _resolve_key(self) -> str | None:
        variable = self.config.api_key_env or self.deployment.key_env
        key = self.config.api_key or (os.environ.get(variable) if variable else None)
        if key:
            return key
        if self.deployment.require_api_key:
            raise RuntimeError(f"The {self.config.adapter_type} API needs {variable}")
        if self.config.api_key_env and self.deployment.gateway == "litellm":
            raise ValueError(f"Credential variable {self.config.api_key_env!r} is empty or unset")
        return "not-required" if self.protocol.client == "openai" else None

    def bind_trace(self, trace) -> None:
        self._trace = trace

    # ---- request shaping -------------------------------------------------------------------

    def transport_output_token_cap(self, provider_cap: int | None) -> int | None:
        """Return the completion limit each request sends; None omits it (provider default).

        Omitting the limit is unsafe on some hosts: Kimi-compatible gateways have defaulted to 128
        tokens (all spent on reasoning) or to the whole context window (rejected as overflow). A
        declared native limit is therefore always sent explicitly.
        """
        if provider_cap is not None:
            return int(provider_cap)
        return self.config.native_output_limit

    def _stream_enabled(self) -> bool:
        stream = self.protocol.stream
        return self.config.chat_stream if stream is None else stream

    def _request_kwargs(self, request: GenerationRequest) -> dict:
        protocol, compat = self.protocol, self.compat
        # A fresh list per call: the steps below edit it in place.
        messages = list(chat_messages(request.canonical))
        if compat.strip_image_detail:
            _strip_image_detail(messages)
        if compat.cache_breakpoint or self.config.provider_options.get("cache_breakpoint"):
            _mark_cache_breakpoint(messages)
        stream = self._stream_enabled()
        kwargs = {"model": self.config.model_name, "messages": messages, "stream": stream}
        include_usage = protocol.include_usage
        if include_usage is None:
            include_usage = self.config.chat_include_usage
        if stream and include_usage:
            kwargs["stream_options"] = {"include_usage": True}
        # The recorded transport cap is the value sent; the request never substitutes another.
        limit = request.transport_output_token_cap
        if limit is not None:
            kwargs[compat.output_limit_param] = int(limit)
        effort = self.config.reasoning_effort
        pin_route = self.deployment.pin_route
        extra_body = {"session_id": self._session_id} if pin_route else {}
        upstream = self.config.upstream_provider
        if pin_route and upstream:
            extra_body["provider"] = {
                "order": [upstream],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        if protocol.reasoning_param == "anthropic_thinking":
            kwargs.update(anthropic_thinking(effort))
        elif effort not in (None, "fixed"):  # `fixed`: the model's reasoning is not configurable
            if protocol.reasoning_param == "openrouter" and upstream:
                extra_body["reasoning"] = {"effort": effort, "exclude": False}
            else:
                kwargs["reasoning_effort"] = effort
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _client(self):
        if self._http is not None or self.protocol.client is None:
            return self._http
        if self.protocol.client == "openai":
            from openai import AsyncOpenAI as AsyncOpenAI
            from openai import DefaultAsyncHttpxClient as DefaultAsyncHttpxClient

            # The harness owns retries (entity.model.TRANSPORT_RETRY) so every attempt is recorded.
            # DefaultAsyncHttpxClient keeps the SDK's own client defaults and adds the hooks.
            self._http_capture = HttpCapture(self, "litellm.openai_sdk", "generation")
            self._http = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=0,
                http_client=DefaultAsyncHttpxClient(event_hooks=self._http_capture.hooks),
            )
        else:
            from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler as AsyncHTTPHandler

            # LiteLLM passes the hooks on to the client of its connection-error resend.
            self._http_capture = HttpCapture(self, "litellm.httpx", "generation")
            self._http = AsyncHTTPHandler(event_hooks=self._http_capture.hooks)
        return self._http

    def _transport_kwargs(self) -> dict:
        sdk_provider = self.protocol.sdk_provider
        model = self.config.model_name
        kwargs = {
            "model": f"{sdk_provider}/{model}" if sdk_provider else model,
            # The harness owns retries; never silently retry or fall back to another backend.
            "num_retries": 0,
            "max_retries": 0,
            "drop_params": False,
            "caching": False,
        }
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = str(self.base_url)
        client = self._client()
        if client is not None:
            kwargs["client"] = client
        if self.deployment.pin_route:
            kwargs["extra_headers"] = {
                "X-OpenRouter-Metadata": "enabled",
                "HTTP-Referer": os.environ.get(
                    "OPENROUTER_REFERER", "https://github.com/RealtimeAgents/Realtime-Bench"
                ),
                "X-Title": os.environ.get("OPENROUTER_TITLE", "Realtime-Bench"),
            }
        return kwargs

    def _backend_route(self) -> dict:
        gateway = self.deployment.gateway
        route = {
            "gateway": gateway,
            "base_url": self.base_url,
            "transport": "litellm",
            "transport_version": LITELLM_VERSION,
            "requested_model": self.config.model_name,
        }
        if self.deployment.pin_route or gateway == "kimi_compatible":
            pinned = self.deployment.pin_route
            route.update(
                configured_upstream_provider=self.config.upstream_provider,
                allow_fallbacks=False if pinned else None,
                require_parameters=True if pinned else None,
                response_metadata=None,
            )
        if self.protocol.usage == "anthropic":
            route.update(response_model=None, response_id=None)
        return route

    # ---- generation ------------------------------------------------------------------------

    async def start_generation(self, request: GenerationRequest) -> GenerationHandle:
        await asyncio.to_thread(pace_request, self)
        kwargs = self._request_kwargs(request)
        kwargs.update(self._transport_kwargs())
        evidence = RequestEvidence(self, kwargs)
        if self._stream_enabled():
            await evidence.stream()
        else:
            await evidence.complete()
        return GenerationHandle(request.generation_id, opaque=evidence)

    async def events(self, generation: GenerationHandle):
        evidence = generation.opaque
        if evidence.response is None:
            async for event in self._complete_response_events(generation):
                yield event
            return
        sequence = 0
        calls: dict = {}
        try:
            events = evidence.response.__aiter__()
            # The runtime can interrupt while consuming a yielded delta. Check before
            # requesting another SDK event: interrupt() has already closed its socket.
            while not evidence.interrupted:
                try:
                    event = await anext(events)
                except StopAsyncIteration:
                    break
                raw_type = type(event).__name__
                evidence.check_route()
                usage = evidence.revised_usage()
                if usage is not None:
                    sequence += 1
                    yield UsageSnapshot(generation.generation_id, sequence, raw_type, usage)
                    if evidence.interrupted:
                        break
                choices = getattr(event, "choices", None) or []
                if not choices:
                    continue
                delta = choices[0].delta
                if self.protocol.surface_tool_calls:
                    for call in getattr(delta, "tool_calls", None) or []:
                        saved = calls.setdefault(call.index, {"name": "", "arguments": ""})
                        function = getattr(call, "function", None)
                        if function:
                            saved["name"] += function.name or ""
                            saved["arguments"] += function.arguments or ""
                reasoning = self._delta_reasoning(delta)
                if reasoning:
                    sequence += 1
                    yield TextDelta(
                        generation.generation_id, sequence, raw_type, reasoning, "reasoning"
                    )
                    if evidence.interrupted:
                        break
                text = getattr(delta, "content", None) or ""
                if text:
                    sequence += 1
                    yield TextDelta(generation.generation_id, sequence, raw_type, text, "visible")
            interrupted = evidence.interrupted
            usage = None if interrupted else evidence.revised_usage()
            if usage is not None:
                sequence += 1
                yield UsageSnapshot(generation.generation_id, sequence, "usage", usage)
            if calls and not interrupted:
                # No tools were supplied: preserve unexpected tool output for protocol validation.
                sequence += 1
                yield TextDelta(
                    generation.generation_id,
                    sequence,
                    "tool_calls",
                    "\n" + json.dumps({"tool_calls": [{"function": v} for v in calls.values()]}),
                    "visible",
                )
            metadata = evidence.boundary_metadata()
            usage = evidence.terminal_usage()
            if usage is not None:
                sequence += 1
                yield UsageSnapshot(generation.generation_id, sequence, "message_delta", usage)
            sequence += 1
            yield GenerationBoundary(
                generation.generation_id,
                sequence,
                "chat.completion.done",
                "interrupted" if interrupted else "turn_complete",
                ProviderHistoryState.EXACT,
                metadata,
            )
        finally:
            await evidence.finish()

    @staticmethod
    def _delta_reasoning(delta) -> str:
        reasoning = next(
            (
                value
                for name in ("reasoning", "reasoning_content")
                if isinstance((value := getattr(delta, name, None)), str) and value
            ),
            "",
        )
        if reasoning:
            return reasoning
        values = []
        for detail in getattr(delta, "reasoning_details", None) or []:
            detail = object_dict(detail) or {}
            value = detail.get("text") or detail.get("summary")
            if value:
                values.append(value)
        return "".join(values)

    def _normalize_usage(self, usage: dict) -> dict:
        if self.protocol.usage == "kimi":
            return kimi_usage(usage)
        return usage

    # ---- complete responses ----------------------------------------------------------------

    async def _complete_response_events(self, generation):
        evidence = generation.opaque
        sequence = 0
        if evidence.interrupted:
            yield GenerationBoundary(
                generation.generation_id,
                1,
                "litellm.completion",
                "interrupted",
                ProviderHistoryState.EXACT,
            )
            return
        usage = evidence.reply_usage
        if usage is not None:
            sequence += 1
            yield UsageSnapshot(
                generation.generation_id, sequence, "litellm.completion", usage
            )
        message = evidence.reply
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if message.get("tool_calls") or message.get("function_call"):
            # No tools were supplied: preserve unexpected tool output for protocol validation.
            content += "\n" + json.dumps(
                {k: message[k] for k in ("tool_calls", "function_call") if message.get(k)}
            )
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        for channel, value in (("reasoning", reasoning), ("visible", content)):
            if isinstance(value, str) and value:
                sequence += 1
                yield TextDelta(
                    generation.generation_id, sequence, "litellm.completion", value, channel
                )
        yield GenerationBoundary(
            generation.generation_id,
            sequence + 1,
            "litellm.completion",
            "turn_complete",
            ProviderHistoryState.EXACT,
            evidence.boundary_metadata(),
        )

    # ---- interruption and cleanup ----------------------------------------------------------

    async def interrupt(self, generation: GenerationHandle) -> None:
        await generation.opaque.interrupt()

    async def close(self) -> None:
        with self._lock:
            responses = list(self._responses.values())
            self._responses.clear()
        for response in responses:
            try:
                await response.aclose()
            except Exception:  # noqa: BLE001, S110 - best-effort stream cleanup
                pass
        if self._http is not None:
            http, self._http = self._http, None
            await http.close()
        # LiteLLM's process-wide logging worker binds to the loop that last used it. Each runtime
        # owns a private loop that closes with the session; stop the worker if it is ours.
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        if getattr(GLOBAL_LOGGING_WORKER, "_bound_loop", None) is asyncio.get_running_loop():
            try:
                await self.clock.wait_for(GLOBAL_LOGGING_WORKER.stop(), 1.0)
            except Exception:  # noqa: BLE001, S110 - logging cleanup must not mask shutdown
                pass

    # Provider error types read from the error's text when no exception in the chain has an HTTP
    # status. LiteLLM gives one to every error it maps, a stream's SSE error event included, but
    # not to a complete response whose 200 body is an error object without a numeric code, as a
    # gateway forwards an upstream error.
    _TRANSIENT_ERROR_TYPES = ("overloaded_error", "api_error", "timeout_error")

    def classify_transport_error(self, exc: BaseException) -> TransportFailure:
        """Classify a failed request for the harness's shared retry policy."""
        import httpx as httpx

        litellm = self._sdk
        for current in exception_chain(exc):
            status = getattr(current, "status_code", None)
            if isinstance(status, int) and status >= 400:
                return classify_http_failure(status, str(current), _retry_after(exc))
        for current in exception_chain(exc):
            if isinstance(current, litellm.RateLimitError):
                return classify_http_failure(429, str(current))
            if isinstance(
                current, (litellm.APIConnectionError, litellm.Timeout, httpx.TransportError)
            ):
                return TransportFailure(TRANSIENT)
        detail = str(exc)
        if "rate_limit_error" in detail:
            return classify_http_failure(429, detail)
        if any(kind in detail for kind in self._TRANSIENT_ERROR_TYPES):
            return TransportFailure(TRANSIENT)
        return TransportFailure(FATAL)


def _retry_after(exc: BaseException) -> float | None:
    """The first Retry-After along the exception chain. The exception LiteLLM raises keeps none
    on its response; the HTTP client's error it wraps, further down the chain, does."""
    for current in exception_chain(exc):
        headers = getattr(getattr(current, "response", None), "headers", None)
        retry_after = retry_after_seconds(headers)
        if retry_after is not None:
            return retry_after
    return None


# Registered chat APIs
class OpenRouterDriver(LiteLLMDriver):
    capabilities = replace(
        _STREAMING_CHAT,
        # The encoder sends past reasoning as plain reasoning_content. OpenRouter's reasoning
        # guide takes it as `reasoning`, for models that return raw reasoning strings; signed,
        # encrypted or summarized reasoning (its Anthropic, OpenAI and Gemini formats) goes back
        # as reasoning_details, which the driver does not keep.
        reasoning_replay="unsupported",
    )
    protocol = ChatProtocol(sdk_provider="openrouter", reasoning_param="openrouter")
    deployment = ChatDeployment(
        family="vlm",
        provider="openrouter",
        key_env="OPENROUTER_API_KEY",
        gateway="openrouter",
        default_base_url=DEFAULT_OPENROUTER_BASE_URL,
        base_url_env="OPENROUTER_BASE_URL",
        pin_route=True,
        requires_upstream_provider=True,
    )
    compat = ChatCompat(provider_options={"cache_breakpoint": (False, (False, True))})


class MoonshotDriver(LiteLLMDriver):
    capabilities = replace(
        _STREAMING_CHAT,
        protocol_family="moonshot_chat",
        # Kimi's thinking-model guide: for K3, passing reasoning_content back "is required in
        # multi-turn conversations and tool-call loops". That is K3's requirement, not the API's:
        # K3's YAML sets reasoning_replay, which sends it (configs/models/kimi-k3.yaml).
        # LiteLLM's Moonshot config sends the encoder's reasoning_content unchanged.
        reasoning_replay="optional",
    )
    protocol = ChatProtocol(sdk_provider="moonshot", client="openai", usage="kimi")
    deployment = ChatDeployment(
        family="vlm",
        provider="moonshot",
        key_env="MOONSHOT_API_KEY",
        vendor="moonshot",
        gateway="kimi_compatible",
        default_base_url=DEFAULT_MOONSHOT_BASE_URL,
        base_url_env="MOONSHOT_BASE_URL",
    )
    compat = ChatCompat(strip_image_detail=True)


class AnthropicMessagesDriver(LiteLLMDriver):
    capabilities = replace(
        _STREAMING_CHAT,
        protocol_family="anthropic_messages",
        # LiteLLM's Anthropic converter drops reasoning_content. Anthropic takes past thinking
        # only as signed thinking blocks passed back unmodified; the driver keeps their summary.
        reasoning_replay="unsupported",
    )
    protocol = ChatProtocol(
        sdk_provider="anthropic",
        include_usage=False,
        reasoning_param="anthropic_thinking",
        usage="anthropic",
    )
    deployment = ChatDeployment(
        family="vlm",
        provider="anthropic",
        key_env="ANTHROPIC_API_KEY",
        vendor="anthropic",
        gateway="anthropic",
        default_base_url=DEFAULT_ANTHROPIC_BASE_URL,
    )
    compat = ChatCompat(
        cache_breakpoint=True,
        output_limit_required=True,
        reasoning_efforts=ANTHROPIC_REASONING_EFFORTS,
    )


class ChatCompletionsDriver(LiteLLMDriver):
    capabilities = replace(
        _STREAMING_CHAT,
        protocol_family="compatible_chat",
        finite_frame_budget=False,
        wall_clock_only=True,
        # OpenAI's chat format has no field for past reasoning, but LiteLLM passes the encoder's
        # reasoning_content through unchanged, and vLLM and SGLang hand it to the model's chat
        # template. A model that needs it back sets reasoning_replay in its YAML's context (K3).
        reasoning_replay="optional",
    )
    protocol = ChatProtocol(
        sdk_provider="openai",
        client="openai",
        stream=None,
        include_usage=None,
        surface_tool_calls=True,
    )
    deployment = ChatDeployment(
        family="vlm",
        provider="compatible",
        key_env="OPENAI_API_KEY",
        gateway="openai_compatible",
        require_api_key=False,
    )
    compat = ChatCompat(strip_image_detail=True)

    # The only LiteLLM API whose window an endpoint declares (BENCHMARK_CONTEXT_*).
    async def request_context_status(self, request):
        """Count the actual pending request, including the newest image and action ledger."""
        limit = self.config.context_window_tokens
        if limit is None:
            return {"pressure": False, "counter": "unconfigured"}
        messages = self._request_kwargs(request)["messages"]
        if self.config.context_counter == "vllm":
            endpoint = self.base_url.removesuffix("/v1") + "/tokenize"
            response = await asyncio.to_thread(
                traced_post(self, "vllm.tokenize", "context_count"),
                endpoint,
                json={
                    "model": self.config.model_name,
                    "messages": messages,
                    "add_generation_prompt": True,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            count = result.get("count")
            actual_limit = result.get("max_model_len")
            if type(count) is not int or count < 0:
                raise RuntimeError("Token counter did not return a valid multimodal input count")
            if actual_limit != limit:
                raise RuntimeError(
                    f"Serving capacity changed: configured {limit}, reported {actual_limit}; start a new run with matching settings"
                )
            source = "vllm_multimodal_tokenize"
        else:
            count = estimated_request_tokens(messages, self.config.context_image_tokens)
            source = "estimated_utf8_bytes_plus_image_allowance"
        return context_status(self.config, count, source)


class DirectLiteLLMDriver(LiteLLMDriver):
    capabilities = replace(
        _STREAMING_CHAT,
        protocol_family="compatible_chat",
        finite_frame_budget=False,
        wall_clock_only=True,
        # The model id picks LiteLLM's provider, and with it what becomes of reasoning_content:
        # sent unchanged to OpenAI-format APIs, an unsigned thought part for Gemini, dropped for
        # Anthropic. It may go back to the providers compat names.
        reasoning_replay="optional",
    )
    protocol = ChatProtocol(sdk_provider=None, client=None, stream=False, surface_tool_calls=True)
    deployment = ChatDeployment(
        family="vlm",
        provider="litellm",
        key_env=None,
        litellm_provider="",
        gateway="litellm",
        require_api_key=False,
    )
    # The providers whose conversion (LiteLLM 1.106.1) keeps an assistant message's
    # reasoning_content: llms/deepseek/chat/transformation.py (which fills a missing one with a
    # space in thinking mode), llms/moonshot/chat/transformation.py, llms/together_ai/chat/
    # transformation.py (which says Together consumes it) and llms/hosted_vllm/chat/
    # transformation.py (which keeps a string one).
    compat = ChatCompat(
        reasoning_replay_providers=("deepseek", "moonshot", "together_ai", "hosted_vllm")
    )
