from __future__ import annotations
from realtime_framework.entity.events import ProviderHistoryState, ProviderEvent

# Contracts
"""Provider-neutral contracts for the frame-driven realtime model runtime.

The benchmark owns time and action execution.  These types describe only model interaction:
provider input, normalized generation events, canonical model history, and synchronization of a
provider/native session with that history.
"""

from collections.abc import AsyncIterator as AsyncIterator
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from enum import Enum as Enum
from typing import Any as Any
from typing import Literal as Literal
from typing import Protocol as Protocol
from typing import runtime_checkable as runtime_checkable

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.request import AnswerRequirements
from realtime_framework.entity.request import CanonicalRequest


class RuntimeState(str, Enum):
    NEW = "new"
    CONNECTING = "connecting"
    READY = "ready"
    GENERATING = "generating"
    INTERRUPTING = "interrupting"
    SYNCHRONIZING = "synchronizing"
    DIRTY = "dirty"
    BROKEN = "broken"
    CLOSED = "closed"


class SyncDisposition(str, Enum):
    EXACT = "exact"
    DEFERRED = "deferred"
    REBUILT = "rebuilt"
    UNSUPPORTED = "unsupported"
    BROKEN = "broken"


@dataclass(frozen=True)
class RealtimeCapabilities:
    """Static protocol facts used for validation, never as turn-result substitutes.

    Every registered driver class declares them; the catalog, the pre-run checks
    (providers/preflight.py), the harness and the runtime read only this declaration.
    """

    protocol_family: str
    persistent_session: bool
    can_compact_history_in_place: bool = False
    max_video_fps: float | None = None
    terminal_output_usage: bool = True
    # How the model submits a GUI action: one JSON action in visible text, exactly one call
    # to the canonical GUI functions (entity.actions.gui_function_declarations), or one choice
    # among harness-enumerated candidate actions. The harness selects its output contract and
    # collector from this value; the observation encoding is context policy item 12.
    action_channel: Literal["text_json", "function_call", "choice"] = "text_json"
    # What one observation carries to the model: the screenshot, or (for models without image
    # input) text only — the accessibility tree plus rendered control state and appearance.
    observation_modality: Literal["screenshot", "text"] = "screenshot"
    # How a stateless request replays history: the canonical retained turns (observations with
    # folded visuals, actions, execution feedback and the archived ledger) shared by every model,
    # or, for prompt windows too small to retain SoM turns, only a digest of actions and feedback.
    # Context policy items 3 and 9 decide what is sent; tests/replay/test_context_policy.py pins it.
    history_replay: Literal["canonical", "action_digest"] = "canonical"
    # What the encoder can render of the context policy; the pre-run checks hold the policy
    # against it (providers/preflight.py check_context_policy). Undeclared means unsupported.
    # The request roles (entity/request.py Role) the encoder maps onto protocol roles.
    roles: tuple[str, ...] = ()
    # The protocol has no roles: blocks are told apart by their position and labels.
    labels_instead_of_roles: bool = False
    # Item 3: where execution feedback can go (PathPolicy.feedback values).
    feedback_placements: tuple[str, ...] = ()
    # Item 6: whether past reasoning can be sent back, and whether the provider's documentation
    # requires it in multi-turn conversations.
    reasoning_replay: Literal["unsupported", "optional", "required"] = "unsupported"
    # What the provider keeps of the conversation besides what the client sent: nothing, or a
    # session's own items the client never sees (reasoning, thoughts, audio).
    server_context: Literal["none", "unknown"] = "unknown"
    # Item 10: the forms past tool calls can be replayed in (PathPolicy.tool_call_replay values).
    tool_call_replay: tuple[str, ...] = ()
    # Choice channel: options per question; the selection flow groups larger candidate sets.
    max_choices: int | None = None
    # Reports complete usage for a finite token clock.
    finite_frame_budget: bool = True
    wall_clock_only: bool = False
    # False: the driver sends no reasoning setting, so `fixed` is the only truthful effort label.
    accepts_reasoning_effort: bool = True
    # Optional driver methods the harness and runtime call.
    classifies_transport_errors: bool = False  # classify_transport_error(exc)
    connection_renewal: bool = False  # connection_renewal_reason()
    # request_context_status(request): counts a request against ModelConfig.context_window_tokens.
    request_size_estimation: bool = False
    maps_output_cap: bool = False  # transport_output_token_cap(provider_cap)
    # How the driver can interact: one request per frame ("turn_based"), or native streaming.
    interaction_modes: frozenset[str] = frozenset({"turn_based"})


@dataclass(frozen=True, kw_only=True)
class Deployment:
    """Where a registered API's requests go: catalog labels and the default credential."""

    family: str  # catalog family: streaming | vlm
    provider: str  # display/grouping label, e.g. openai | openrouter | gemini
    key_env: str | None  # default credential variable
    vendor: str | None = None  # model maker, when the API's model ids carry no vendor prefix
    # LiteLLM provider whose bundled model map supplies native output limits; "" means the model
    # id carries its own provider prefix, None means the endpoint must declare any limit.
    litellm_provider: str | None = None


@dataclass(frozen=True)
class SessionSeed:
    system_instruction: str
    model_name: str
    native_output_limit: int | None
    fps: float
    # The answer requirements a live session declares at setup, before any request exists: the
    # function declarations.
    answer: AnswerRequirements


@dataclass
class ProviderObservation:
    observation: Observation
    prompt: str
    image_bytes: bytes
    mime_type: str
    image_detail: str
    geometry: dict
    inspection: dict | None = None
    # Session paths send the pending execution feedback before the frame prompt; it is kept
    # apart from the prompt and dropped when the screenshot folds.
    feedback_prefix: str = ""

    @property
    def observation_id(self) -> str:
        metadata = getattr(self.observation, "metadata", {}) or {}
        return str(metadata.get("observation_id") or "missing-observation-id")


@dataclass
class CanonicalTurn:
    """One complete model turn, independent of provider-side representation."""

    observation: ProviderObservation
    output: str = ""
    execution_feedback: str = ""
    action_history_index: int | None = None
    visual_folded: bool = False
    provider_refs: dict[str, Any] = field(default_factory=dict)
    # The runner's decision index; a retry rebuilds the turn with the same index.
    decision: int = field(kw_only=True)
    # Always stored; the context policy decides whether a request replays it.
    output_reasoning: str = ""
    output_wall_ns: int | None = None
    feedback_wall_ns: int | None = None

    def fold_visual(self) -> None:
        """Release one old screenshot and its observation-bound structured payload."""
        if self.visual_folded:
            return
        source = self.observation
        raw = source.observation
        metadata = getattr(raw, "metadata", {}) or {}
        self.observation = ProviderObservation(
            observation=Observation(
                screenshot=b"",
                timestamp=raw.timestamp,
                tick_number=raw.tick_number,
                metadata={
                    "observation_id": metadata.get("observation_id"),
                    "visual_folded": True,
                },
            ),
            prompt="",
            image_bytes=b"",
            mime_type=source.mime_type,
            image_detail=source.image_detail,
            geometry={},
            inspection=None,
        )
        self.visual_folded = True

    def retention_record(self) -> dict:
        return {
            "observation": self.observation.observation,
            "prompt": self.observation.prompt,
            "output": self.output,
            "execution_feedback": self.execution_feedback,
            "visual_folded": self.visual_folded,
        }


@dataclass(frozen=True)
class ConnectRequest:
    seed: SessionSeed
    resume_handle: Any = None
    # The retained history as the request builder renders it; None opens an empty session.
    history: CanonicalRequest | None = None


@dataclass(frozen=True)
class GenerationRequest:
    generation_id: str
    provider_output_token_cap: int | None
    transport_output_token_cap: int | str | None
    clock_mode: Literal["token", "wall"]
    timeout_s: float | None = None
    # Filled by the runtime from the retained history before each provider call.
    canonical: CanonicalRequest | None = None
    # The event kernel's number of this generation, for an agent that keys a generation's state by
    # it; None when the agent does not pass it on, as the default harness's requests do not.
    generation: int | None = None


@dataclass
class GenerationHandle:
    generation_id: str
    opaque: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterruptResult:
    history_state: ProviderHistoryState
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HistorySyncRequest:
    seed: SessionSeed
    reason: str
    # The retained turns and the archived-action ledger, as the request builder renders them.
    history: CanonicalRequest
    current_generation: GenerationHandle | None = None
    force_rebuild: bool = False


@dataclass(frozen=True)
class HistorySyncResult:
    disposition: SyncDisposition
    history_state: ProviderHistoryState
    detail: str = ""
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    usage: dict | None
    token_fields: dict
    output_chunks: tuple[dict, ...]
    reasoning_text: str = ""


@dataclass(frozen=True)
class RuntimeTurnResult:
    output: GenerationOutput
    generation_id: str
    history_sync: HistorySyncResult
    terminal_history_state: ProviderHistoryState
    interrupted: bool
    cancelled: bool  # the generation was aborted before its result came back
    latency_s: float
    provider_call_latency_s: float
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PendingDecision:
    """One generation's result, held by the agent until the event kernel accepts or discards it."""

    generation: int  # the kernel's generation number
    decision: int
    result: dict  # the decision result, as stream_generate returns it
    cancelled: bool = False  # the generation was aborted before its result came back
    # What accepting the result writes, kept by the agent; the shell does not read it.
    staged: Any = None


@runtime_checkable
class OutputCollector(Protocol):
    def consume(self, event: ProviderEvent) -> None: ...

    def finish(self) -> GenerationOutput: ...


@runtime_checkable
class RealtimeDriver(Protocol):
    capabilities: RealtimeCapabilities

    async def connect(self, request: ConnectRequest) -> None: ...

    # ``history`` is the retained history plus this observation's frame, as the request builder
    # renders it.
    async def send_observation(
        self, observation: ProviderObservation, history: CanonicalRequest
    ) -> dict[str, Any]: ...

    async def start_generation(self, request: GenerationRequest) -> GenerationHandle: ...

    def events(self, generation: GenerationHandle) -> AsyncIterator[ProviderEvent]: ...

    async def interrupt(self, generation: GenerationHandle) -> InterruptResult: ...

    async def sync_history(self, request: HistorySyncRequest) -> HistorySyncResult: ...

    async def close(self) -> None: ...


class ChoiceContextLimitError(RuntimeError):
    """The provider rejected one state plus question as beyond the model's context."""


@dataclass(frozen=True)
class ChoiceReply:
    """One answered choice question: the native answer, usage and raw response."""

    answer: dict[str, Any]
    usage: dict[str, int]
    response: dict[str, Any]


@runtime_checkable
class ChoiceClient(Protocol):
    """A stateless choice-model API (action_channel="choice"): one question per call.

    The client owns only transport: authentication, failure classification
    (classify_question_error), context-limit detection and response validation. Questions, their
    state, candidates, history and retries belong to the selection flow (providers/decisions).
    A client that declares request_size_estimation reports context pressure for the runtime
    through ``request_context_status(request)``.
    """

    capabilities: RealtimeCapabilities  # max_choices: options per question

    async def decide(self, state: dict[str, Any] | list, question: dict[str, Any]) -> ChoiceReply:
        """Raise ChoiceContextLimitError when state plus question exceed the context."""

    async def close(self) -> None: ...


# Transport failures
"""One retry policy for every model API, run by one executor.

Drivers only classify a failed request; RequestRetry decides whether and when to send it again,
per request unit: the harness's turn, a Decision question, a session connect. A request is
retried only while no model output, usage, or call has been received, and always from canonical
history, so a retry can never replay or duplicate a model action. SDK-level retries stay disabled
so every attempt is recorded. Retry time is part of the decision: on the wall clock the world keeps
running; on the token clock a request without output charges no tokens. Once the runner abandons
a decision, none of its request units is sent again.
"""

TRANSIENT = "transient"  # connection loss, timeout, 408, 5xx, overloaded
RATE_LIMITED = "rate_limited"  # 429 without a quota/billing signal
QUOTA_EXHAUSTED = "quota_exhausted"  # credits or quota: a retry cannot succeed
FATAL = "fatal"  # rejected request, protocol error, anything unclassified
REFUSED = "refused"  # the model declined to answer: the same request cannot succeed on retry

TRANSPORT_FAILURE_REASONS = {
    TRANSIENT: "provider_unavailable",
    RATE_LIMITED: "provider_rate_limited",
    QUOTA_EXHAUSTED: "provider_quota_exhausted",
    FATAL: "provider_request_failed",
    REFUSED: "provider_refused",
}
_QUOTA_MARKERS = (
    "insufficient_quota",
    "exceeded_current_quota",
    "exceeded your current quota",
    "quota exceeded",
    "insufficient credits",
    "insufficient balance",
    "billing",
)


@dataclass(frozen=True)
class TransportFailure:
    kind: str
    retry_after_s: float | None = None
    # Set once a nested request already spent the shared retry budget.
    exhausted: bool = False

    @property
    def retryable(self) -> bool:
        return self.kind in (TRANSIENT, RATE_LIMITED) and not self.exhausted

    @property
    def invalid_reason(self) -> str:
        return TRANSPORT_FAILURE_REASONS[self.kind]


@dataclass(frozen=True)
class TransportRetryPolicy:
    max_retries: int = 3
    backoff_s: tuple[float, ...] = (1.0, 3.0, 8.0)
    jitter_ratio: float = 0.25
    # A provider asking for a longer pause than this is treated as exhausted, not waited out.
    max_retry_after_s: float = 30.0

    def delay_s(self, retry: int, key: str, failure: TransportFailure) -> float | None:
        """Delay before 1-based ``retry``, or None when the request must not be retried."""
        if not failure.retryable or retry > self.max_retries:
            return None
        if failure.retry_after_s is not None and failure.retry_after_s > self.max_retry_after_s:
            return None
        delay = self.backoff_s[min(retry, len(self.backoff_s)) - 1]
        # Stable jitter from the request key keeps parallel lanes from retrying in lockstep.
        unit = sum((i + 1) * ord(c) for i, c in enumerate(f"{key}:{retry}")) % 1001 / 1000.0
        delay *= 1.0 + self.jitter_ratio * (2.0 * unit - 1.0)
        return round(max(delay, failure.retry_after_s or 0.0), 3)


TRANSPORT_RETRY = TransportRetryPolicy()


class TransportFailureError(RuntimeError):
    """A request that failed after the shared retry policy was applied to it."""

    def __init__(self, failure: TransportFailure, cause: BaseException, retries: int):
        super().__init__(f"{failure.kind} after {retries} transport retries: {cause!r}"[:500])
        self.failure = TransportFailure(failure.kind, failure.retry_after_s, exhausted=True)
        self.retries = retries


def exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def retry_after_seconds(headers) -> float | None:
    """Parse a numeric Retry-After header; HTTP-date values are ignored."""
    try:
        value = headers.get("retry-after") if headers is not None else None
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def classify_http_failure(status: int, detail: str = "", retry_after_s=None) -> TransportFailure:
    detail = detail.lower()
    if status == 402 or (status == 429 and any(m in detail for m in _QUOTA_MARKERS)):
        return TransportFailure(QUOTA_EXHAUSTED)
    if status == 429:
        return TransportFailure(RATE_LIMITED, retry_after_s)
    if status == 408 or status >= 500:
        return TransportFailure(TRANSIENT, retry_after_s)
    return TransportFailure(FATAL)


import asyncio
import contextlib
import contextvars
import threading
from collections.abc import Callable

from realtime_framework.clock import Clock
from realtime_framework.clock import SystemClock


class RetryCancelled(BaseException):
    """The generation a request unit belongs to was abandoned before the unit was sent again.

    Like asyncio.CancelledError it is not an Exception, so the provider and session error handlers
    it unwinds through neither record it as a failure nor turn it into session state.
    """


_NEVER_CANCELLED = threading.Event()
_RETRY_CANCEL: contextvars.ContextVar[threading.Event] = contextvars.ContextVar(
    "retry_cancel", default=_NEVER_CANCELLED
)


@contextlib.contextmanager
def retry_cancellation(cancel: threading.Event):
    """Cancel every request unit sent in this context once ``cancel`` is set. Like trace_scope,
    the ContextVar follows the call into the runtime's event loop."""
    token = _RETRY_CANCEL.set(cancel)
    try:
        yield
    finally:
        _RETRY_CANCEL.reset(token)


def _backoff(clock: Clock, cancel: threading.Event, delay: float) -> None:
    clock.wait(cancel, delay)


@dataclass(frozen=True)
class RequestRetry:
    """Run one request unit under TRANSPORT_RETRY: a turn, a Decision question, or a session
    connect.

    ``send(attempt)`` makes one 1-based attempt. A failed attempt is classified, recorded and, if
    the policy allows, sent again after its delay. ``classify`` returns None for an exception that
    is not a transport failure; it propagates unrecorded. ``replayable`` is False once the failed
    attempt received model output. ``record(attempt, exc, failure, delay)`` sees every classified
    failure before any wait; delay is None when the request is not sent again, and the exception
    then propagates unless ``record`` raises its own. Once the context's retry_cancellation is
    set, the backoff ends, no attempt starts, and RetryCancelled passes enclosing units unrecorded.
    The backoff waits on ``clock``.
    """

    key: str  # the request unit's stable jitter key
    classify: Callable[[BaseException], TransportFailure | None]
    record: Callable[[int, BaseException, TransportFailure, float | None], None]
    replayable: Callable[[BaseException], bool] = lambda exc: True
    clock: Clock = field(default_factory=SystemClock)

    def run(self, send):
        cancel = _RETRY_CANCEL.get()
        attempt = 1
        while not cancel.is_set():
            try:
                return send(attempt)
            except Exception as exc:
                delay = self._failed(attempt, exc)
                if delay is None:
                    raise
            _backoff(self.clock, cancel, delay)
            attempt += 1
        raise RetryCancelled(f"generation abandoned before attempt {attempt}")

    async def run_async(self, send):
        cancel = _RETRY_CANCEL.get()
        attempt = 1
        while not cancel.is_set():
            try:
                return await send(attempt)
            except Exception as exc:
                delay = self._failed(attempt, exc)
                if delay is None:
                    raise
            await asyncio.to_thread(_backoff, self.clock, cancel, delay)
            attempt += 1
        raise RetryCancelled(f"generation abandoned before attempt {attempt}")

    def _failed(self, attempt: int, exc: Exception) -> float | None:
        failure = self.classify(exc)
        if failure is None:
            return None
        replayable = self.replayable(exc)
        delay = TRANSPORT_RETRY.delay_s(attempt, self.key, failure) if replayable else None
        self.record(attempt, exc, failure, delay)
        return delay


# Says only what the fold removed: whether this step's reply and feedback are still there differs
# by path and by step.
FOLDED_VISUAL_PLACEHOLDER = (
    "[Earlier screenshot and its interactables were folded from active visual context.]"
)

ARCHIVED_ACTION_LEDGER_HEADER = (
    "ARCHIVED EXECUTED ACTIONS (older than the verbatim detailed-history window):"
)

import sys
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str  # stable CLI id, e.g. "gpt-5.2" (primary identifier)
    family: str  # streaming | vlm
    provider: str  # display/grouping label, e.g. openai | openrouter | gemini
    adapter_type: str  # registered API key (providers.registry.APIS) or package.module:Class
    model_name: str  # backend deployment/model id
    native_output_limit: Optional[int] = None  # physical model/provider cap; never benchmark B
    native_output_limit_source: Optional[str] = None  # endpoint | litellm | None (unknown)
    python_env: str = (
        sys.executable
    )  # Python interpreter for model subprocesses
    key_env: Optional[str] = None  # API-key env var; None for local models
    api_base_url: Optional[str] = None  # override provider default base_url (usually None)
    reasoning_efforts: tuple[str, ...] = ()  # model-native accepted reasoning values
    defaults: dict = field(default_factory=dict)  # ModelConfig field overrides
    pricing: Optional[dict] = None  # billing-only rate card; not runtime identity
    benchmark_eligible: bool = True  # false => excluded from maintained leaderboard sweeps
    comparison_scope: str = "model"  # model | experimental_system | system_reference


ModelClient = RealtimeDriver
