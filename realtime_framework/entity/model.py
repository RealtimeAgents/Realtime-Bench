from __future__ import annotations
from realtime_framework.entity.events import ProviderHistoryState, ProviderEvent

# Contracts
"""Provider-neutral contracts for the frame-driven realtime model runtime.

The benchmark owns time and action execution.  These types describe only model interaction:
provider input, normalized generation events, canonical model history, and synchronization of a
provider/native session with that history.
"""

import abc
from collections.abc import AsyncIterator as AsyncIterator
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from enum import Enum as Enum
from typing import Any as Any
from typing import ClassVar
from typing import Literal as Literal
from typing import Protocol as Protocol
from typing import runtime_checkable as runtime_checkable

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.request import AnswerRequirements
from realtime_framework.entity.request import CanonicalRequest


class RuntimeState(str, Enum):
    """Whether the session can take the next generation (providers/session_lifecycle.py), or
    the runtime is closed."""

    READY = "ready"
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
    can_compact_history_in_place: bool = False
    max_video_fps: float | None = None
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
    # Context policy item 9 must name this form (providers/preflight.py check_context_policy);
    # under the digest every turn's feedback is in it, whatever item 3 says.
    history_replay: Literal["canonical", "action_digest"] = "canonical"
    # What the encoder can render of the context policy: the roles, feedback placements,
    # reasoning replay and tool-call replay forms below, which the pre-run checks hold the
    # policy against (providers/preflight.py check_context_policy). Undeclared means unsupported.
    # The request roles (entity/request.py Role) the encoder maps onto protocol roles.
    roles: tuple[str, ...] = ()
    # The protocol has no roles: blocks are told apart by their position and labels.
    labels_instead_of_roles: bool = False
    # Item 3: where execution feedback can go (PathPolicy.execution_feedback values).
    feedback_placements: tuple[str, ...] = ()
    # Item 6: whether past reasoning can be sent back, and whether the provider's documentation
    # requires it in multi-turn conversations.
    reasoning_replay: Literal["unsupported", "optional", "required"] = "unsupported"
    # Not checked, but recorded in the run configuration (evaluation/config.py
    # effective_run_config); undeclared means unknown. What the provider keeps of the
    # conversation besides what the client sent: nothing, or a session's own items the client
    # never sees (reasoning, thoughts, audio).
    server_context: Literal["none", "unknown"] = "unknown"
    # Item 10: the forms past tool calls can be replayed in (PathPolicy.tool_call_history values).
    tool_call_replay: tuple[str, ...] = ()
    # Not a context-policy item. Choice channel: options per question; the selection flow groups
    # larger candidate sets.
    max_choices: int | None = None
    # Reports complete usage for a finite token clock.
    finite_frame_budget: bool = True
    wall_clock_only: bool = False
    # False: the driver sends no reasoning setting, so `fixed` is the only truthful effort label.
    accepts_reasoning_effort: bool = True
    # How the driver can interact: one request per frame ("turn_based"), or native streaming.
    interaction_modes: frozenset[str] = frozenset({"turn_based"})
    # The highest model-input rate a native-streaming session takes, where frames go in as
    # items rather than as video (max_video_fps bounds that in either mode); None: no other cap.
    max_streaming_fps: float | None = None


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
    # Every model of the API names its upstream provider (ModelConfig.upstream_provider), in its
    # YAML or the driver's model_defaults, for its requests to pin or its responses to be checked
    # against; a model of any other API names none (providers/registry.py).
    requires_upstream_provider: bool = False


@dataclass(frozen=True, kw_only=True)
class Compat:
    """Named restrictions and special requirements of a registered API: those the pre-run checks
    (providers/preflight.py check_config, check_session_config) read of every API. A driver
    family's compat adds its own."""

    # Every request sends an output limit, so a model without a declared native limit cannot run.
    output_limit_required: bool = False
    # The ModelConfig.provider_options the encoder reads: each name, its default and its allowed
    # values. Any other option is refused.
    provider_options: dict[str, tuple[str | bool, tuple[str | bool, ...]]] = field(
        default_factory=dict
    )
    # The options a model must set (in its YAML): their defaults stand in for no model.
    required_provider_options: tuple[str, ...] = ()
    # An option value that holds only with another option's value: (name, value) -> (name, value).
    provider_option_requires: dict[tuple[str, str], tuple[str, str]] = field(
        default_factory=dict
    )
    # The reasoning_effort values the encoder maps (None: the provider default); None takes any.
    reasoning_efforts: tuple[str | None, ...] | None = None
    # Where the model id names the provider (provider/model-id), the providers past reasoning may
    # go back to (context setting reasoning_replay); None: any the API takes it for.
    reasoning_replay_providers: tuple[str, ...] | None = None


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
    # Its frame opens with the feedback on decisions of which nothing was accepted, which no
    # turn carried as its generation started; the fold keeps it.
    unattached_feedback: str = ""
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
    transport_output_token_cap: int | str | None
    # Filled by the runtime from the retained history before each provider call.
    canonical: CanonicalRequest | None = None
    # The event kernel's number of this generation, which the agent keys the generation's state by.
    generation: int | None = None


@dataclass(frozen=True)
class PushReceipt:
    """What became of a frame native streaming pushed (decision N43): when it was written on
    the wire, and when the server confirmed it holds it, None where the protocol confirms no
    frame; both read on the driver's operational clock (Clock.monotonic)."""

    written_at: float
    confirmed_at: float | None = None


@dataclass
class GenerationHandle:
    generation_id: str
    opaque: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    # The error that broke a broken sync, where the driver keeps it: a session operation that
    # needs the sync exact raises its failure from it (providers/session_lifecycle.py).
    error: BaseException | None = None


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
    # The generation's turn with its output; the history takes it once the result is accepted.
    turn: CanonicalTurn
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
    result: dict  # the decision result


@runtime_checkable
class OutputCollector(Protocol):
    def consume(self, event: ProviderEvent) -> None: ...

    def finish(self) -> GenerationOutput: ...


class ModelEncoder(abc.ABC):
    """What the runtime asks of every model API's encoder: a generation, its events, its
    interrupt and the close; and bind_trace, which the harness calls with the episode trace on
    every encoder it builds. An encoder is a StatelessEncoder or a SessionEncoder. Its class
    declares capabilities, and deployment, compat and model_defaults beside them, which the
    catalog checks it for (providers/registry.py). An optional method has a default, which an
    encoder overrides to take it up (implements)."""

    capabilities: ClassVar[RealtimeCapabilities]
    # The episode's operational clock, which the harness passes as Class(config, clock=clock).
    # The HTTP capture hooks (providers/capture.py) time each response on it.
    clock: Clock

    @abc.abstractmethod
    def bind_trace(self, trace) -> None: ...

    @abc.abstractmethod
    async def start_generation(self, request: GenerationRequest) -> GenerationHandle: ...

    @abc.abstractmethod
    def events(self, generation: GenerationHandle) -> AsyncIterator[ProviderEvent]: ...

    @abc.abstractmethod
    async def interrupt(self, generation: GenerationHandle) -> None:
        """Cut ``generation`` short, as far as the API can, ahead of the abort's close."""

    @abc.abstractmethod
    async def close(self) -> None: ...

    def classify_transport_error(self, exc: BaseException) -> TransportFailure:
        """The failure ``exc`` of a request, by which RequestRetry decides whether and when to
        send it again; by default every failure is fatal, and none is retried."""
        return TransportFailure(FATAL)

    def transport_output_token_cap(self, provider_cap: int | None) -> int | str | None:
        """The output limit a request sends for the decision's provider cap; by default that
        cap."""
        return None if provider_cap is None else int(provider_cap)


class StatelessEncoder(ModelEncoder):
    """An encoder that leaves nothing at the provider: each generation sends the whole canonical
    request (GenerationRequest.canonical) in the API's own format, and the runtime asks it
    nothing else. An abort interrupts the generation under way and closes the encoder; the
    generations after it reopen what they need."""

    async def request_context_status(self, request: GenerationRequest) -> dict | None:
        """Count ``request`` against ModelConfig.context_window_tokens; None when the encoder
        does not estimate its requests, and the runtime then archives nothing for context
        pressure."""
        return None


class SessionEncoder(ModelEncoder):
    """An encoder that keeps one provider session open across generations, holding the native
    history. The runtime connects it, gives it each observation, and has it manage the session:
    ready it before each generation, bring it to the retained history after the history changed,
    renew its connection, repair it after each generation, abort it and mark it dirty; and it
    reads the session's state. Each operation that takes ``record`` calls it as
    ``record(sync, result=None, error=None)``: once for each sync it runs, with the sync request
    and its result or error, and with ``sync=None`` once it reopened the session.

    SessionLifecycle (providers/session_lifecycle.py) implements each of them but
    send_observation, and close, over the encoder's connection operations; the encoder adds
    send_observation and the rest of ModelEncoder's."""

    @abc.abstractmethod
    async def connect(self, request: ConnectRequest) -> None: ...

    @abc.abstractmethod
    async def send_observation(
        self, observation: ProviderObservation, history: CanonicalRequest
    ) -> dict[str, Any]:
        """Send ``observation``; ``history`` is the retained history plus this observation's
        frame, as the request builder renders it."""

    @abc.abstractmethod
    async def ensure_ready(
        self, seed: SessionSeed, history: CanonicalRequest, record: Callable[..., None]
    ) -> HistorySyncResult | None: ...

    @abc.abstractmethod
    async def synchronize(
        self, request: HistorySyncRequest, record: Callable[..., None]
    ) -> HistorySyncResult: ...

    @abc.abstractmethod
    async def renew(
        self, seed: SessionSeed, history: CanonicalRequest, record: Callable[..., None]
    ) -> HistorySyncResult | None: ...

    @abc.abstractmethod
    async def repair(
        self,
        request: HistorySyncRequest,
        terminal_history: ProviderHistoryState,
        reconnect_required: bool,
        record: Callable[..., None],
    ) -> HistorySyncResult | None: ...

    @abc.abstractmethod
    async def abort(self, generation: GenerationHandle | None) -> None: ...

    @abc.abstractmethod
    def mark_dirty(self) -> None: ...

    @property
    @abc.abstractmethod
    def state(self) -> RuntimeState: ...

    def connection_renewal_reason(self) -> str | None:
        """Why the provider asks for a new connection; None when it asks for none."""
        return None


def interface_methods(kind: type) -> tuple[str, ...]:
    """The abstract members of ``kind`` and its bases, a base's before its subclass's and each
    class's in the order it defines them: what an encoder of that kind implements."""
    names = (
        name
        for base in reversed(kind.__mro__)
        for name, member in vars(base).items()
        if getattr(member, "__isabstractmethod__", False)
    )
    return tuple(dict.fromkeys(names))


def implements(driver: type, name: str) -> bool:
    """Whether the ``driver`` class overrides the default ModelEncoder, StatelessEncoder or
    SessionEncoder gives its optional method ``name``: how the pre-run checks know, before any
    request, whether a driver estimates its requests or maps the output cap."""
    default = next(
        vars(base)[name]
        for base in (ModelEncoder, StatelessEncoder, SessionEncoder)
        if name in vars(base)
    )
    return getattr(driver, name, default) is not default


class ChoiceContextLimitError(RuntimeError):
    """The provider rejected one state plus question as beyond the model's context."""


@dataclass(frozen=True)
class ChoiceReply:
    """One answered choice question: the native answer, usage and raw response."""

    answer: dict[str, Any]
    usage: dict[str, int]
    response: dict[str, Any]


class ChoiceClient(abc.ABC):
    """A stateless choice-model API (action_channel="choice"): one question per call, with at
    most its capabilities' max_choices options. A choice encoder has these methods, which the
    selection flow (providers/decisions/selection.py) asks each question and classifies its
    failures with.

    The client owns only transport: authentication, failure classification
    (classify_question_error), context-limit detection and response validation. Questions, their
    state, candidates, history and retries belong to the selection flow (providers/decisions).
    A client that overrides ``request_context_status(request)`` reports context pressure for the
    runtime through it.
    """

    @abc.abstractmethod
    async def decide(self, state: dict[str, Any] | list, question: dict[str, Any]) -> ChoiceReply:
        """Raise ChoiceContextLimitError when state plus question exceed the context."""

    @abc.abstractmethod
    def classify_question_error(self, exc: BaseException) -> TransportFailure:
        """The failure ``exc`` of one question, by which the selection flow retries it."""


# Transport failures
"""One retry policy for every model API, run by one executor.

Drivers only classify a failed request; RequestRetry decides whether and when to send it again,
per request unit: the harness's turn, a Decision question, a session connect. A request is
retried only while no model output, usage, or call has been received, and always from canonical
history, so a retry can never replay or duplicate a model action. SDK-level retries stay disabled
so every attempt is recorded. Retry time is part of the decision: on the wall clock the world keeps
running; on the token clock a request without output charges no tokens. Once the runner abandons
a decision, none of its request units is sent again, and each records the attempt that stopped.
"""

import hashlib

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
        # Stable jitter from the request key, which names the episode attempt, keeps parallel
        # episodes from retrying in lockstep.
        digest = hashlib.sha256(f"{key}:{retry}".encode()).digest()
        unit = int.from_bytes(digest[:8], "big") / 2**64
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
    the ContextVar follows the call onto the agent's execution thread."""
    token = _RETRY_CANCEL.set(cancel)
    try:
        yield
    finally:
        _RETRY_CANCEL.reset(token)


def retry_cancel_event() -> threading.Event:
    """The cancel event of the enclosing retry_cancellation; outside one, an event never set."""
    return _RETRY_CANCEL.get()


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
    set, the backoff ends and no attempt starts. ``cancelled(attempt, exc)`` then sees the attempt
    the cancellation stopped: the next one, the one under way when a nested unit raised
    RetryCancelled, or the one that failed once it was set, as the abort's close fails the
    attempt under way, which ``record`` then never sees. The RetryCancelled propagates, and
    enclosing units take it the same way, never as a failure. The backoff waits on ``clock``.
    """

    key: str  # the request unit's stable jitter key, led by the episode attempt (AgentTrace)
    classify: Callable[[BaseException], TransportFailure | None]
    record: Callable[[int, BaseException, TransportFailure, float | None], None]
    cancelled: Callable[[int, RetryCancelled], None]
    replayable: Callable[[BaseException], bool] = lambda exc: True
    clock: Clock = field(default_factory=SystemClock)

    def run(self, send):
        cancel = _RETRY_CANCEL.get()
        attempt = 1
        try:
            while not cancel.is_set():
                try:
                    return send(attempt)
                except Exception as exc:
                    delay = self._failed(attempt, exc, cancel)
                    if delay is None:
                        raise
                _backoff(self.clock, cancel, delay)
                attempt += 1
            raise RetryCancelled(f"generation abandoned before attempt {attempt}")
        except RetryCancelled as exc:
            self.cancelled(attempt, exc)
            raise

    async def run_async(self, send):
        cancel = _RETRY_CANCEL.get()
        attempt = 1
        try:
            while not cancel.is_set():
                try:
                    return await send(attempt)
                except Exception as exc:
                    delay = self._failed(attempt, exc, cancel)
                    if delay is None:
                        raise
                await asyncio.to_thread(_backoff, self.clock, cancel, delay)
                attempt += 1
            raise RetryCancelled(f"generation abandoned before attempt {attempt}")
        except RetryCancelled as exc:
            self.cancelled(attempt, exc)
            raise

    def _failed(self, attempt: int, exc: Exception, cancel: threading.Event) -> float | None:
        failure = self.classify(exc)
        if failure is None:
            return None
        if cancel.is_set():
            raise RetryCancelled(f"generation abandoned during attempt {attempt}") from exc
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
    # The context settings the model's YAML sets (its context section), over the standard values
    context: dict = field(default_factory=dict)
