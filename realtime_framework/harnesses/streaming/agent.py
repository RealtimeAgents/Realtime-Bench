"""The native-streaming agent: the streaming shell's agent (evaluation/streaming_shell.py
StreamingKernelAgent) for a live session that frames go into as they are captured, and whose
replies hand their actions over as they come off the wire (phase2.md, native streaming).

The agent keeps the content blocks the session should hold, in the order they came to it, and
gives every push and connection operation the view they make (history.py), so that a session
rebuilt from the view holds what the live one did. Every driver operation runs on the agent's one
execution thread; the session's lifecycle is the driver's (providers/session_lifecycle.py). Each
push and each reply readies the session first, and renews the connection the provider asked to
renew, except a push while a reply is on the session: readying it then would close the reply's
connection, or rebuild the session without the reply's output. A push the session refuses until
it is renewed, as Gemini Live's after a goAway, waits for that reply instead, at a gate of its
own; whether it waits is decided on the execution thread, where the goAway is read. While a push
or a reply prepares the session, attempts and backoffs alike, or a push waits at its gate for the
renewal, the kernel is told the session is being prepared, and once nothing prepares it any more
(decision 13): a push waiting for the session meanwhile is not stuck in it.

A push enters the frame in the frame table, which draws its number on the screenshot, and once the
driver put it into the session records it as sent and confirmed when the driver's receipt says it
was written and confirmed, or its confirmation unknown where the protocol confirms no frame
(decision N43). A frame whose push failed keeps its number and stays out of the view (frames.py).
As a turn-based history does, the agent holds a screenshot only while the view keeps it, and a
frame only while a reply may start on it or is on it.

A reply starts on the view. Each function call is parsed as its message is read, against the frame
it names, and posted to the kernel with the time its message came off the wire, in semantic
seconds from the episode's origin; one whose frame cannot be found arrives refused. One that is no
action is the reply's protocol failure. Where the model answers in text, the action is parsed once
the text is complete, and arrived then; once the reply ended where no text completed. A call the
provider cancels, of this reply or an earlier one, is withdrawn under the generation it came in,
once its action was posted. A reply ends as a turn does: a provider fault fails it, but for one
the encoder takes as transient that came before the model said anything, which has the reply
asked for again under the shared retry policy; a session it leaves other than exact is rebuilt
before it is used again, and a function call reply must complete.

An accepted reply's output joins the view where the output began, so that a rebuilt session shows
the model its actions before their results; a discarded one adds nothing. Each outcome the kernel
commits joins it as its execution feedback (feedback.py), where the policy places it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from realtime_framework.clock import SystemClock
from realtime_framework.entity.actions import (
    Observation,
    canonical_function_action,
    gui_function_declarations,
)
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.events import (
    FunctionCall,
    FunctionCallsCancelled,
    GenerationBoundary,
    ProviderError,
    ProviderHistoryState,
    TextDelta,
    TextDone,
    UsageSnapshot,
)
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationHandle,
    GenerationRequest,
    PendingDecision,
    ProviderObservation,
    RequestRetry,
    RetryCancelled,
    SessionSeed,
    SyncDisposition,
    TransportFailureError,
    retry_cancellation,
)
from realtime_framework.entity.records import TraceEvent, trace_scope
from realtime_framework.entity.request import (
    AnswerRequirements,
    CallParameters,
    CanonicalRequest,
    ContentBlock,
)
from realtime_framework.evaluation.kernel.streaming import (
    ActionArrived,
    CallsCancelled,
    SessionPrepared,
    SessionPreparing,
)
from realtime_framework.harnesses.default.execution_thread import ExecutionThread
from realtime_framework.harnesses.default.frames import FrameRecord, FrameTable
from realtime_framework.harnesses.default.harness import (
    INTERACTION_PROFILES,
    VisionAgentBase,
    action_text,
    bind_observation,
    parse_grounded_action,
)
from realtime_framework.harnesses.default.request import system_block
from realtime_framework.harnesses.streaming.feedback import execution_feedback
from realtime_framework.harnesses.streaming.history import (
    feedback_block,
    frame_block,
    history_view,
    output_block,
)
from realtime_framework.providers.registry import adapter_class
from realtime_framework.providers.session_lifecycle import RenewalRequired, session_preparing

if TYPE_CHECKING:
    from realtime_framework.evaluation.kernel.events import Outcome

logger = logging.getLogger(__name__)

# The call id the agent gives the one action of a reply in text (ActionArrived.call_id).
TEXT_CALL = "text"
# How long the session may take to open as the episode starts, its retries included, as a turn's
# first connect (RealtimeRuntime.start).
_OPEN_TIMEOUT_S = 60.0
# How long an abort or close waits on the execution thread: longer than an abort takes, within
# its interrupt's bound and its close's, and than the session waits for an abort's cleanup, with
# time for a close after (RealtimeRuntime.close).
_SESSION_TIMEOUT_S = 15.0
# The boundaries that end a reply, whose provider history state is the session's after it.
_TERMINAL = {"generation_complete", "turn_complete", "interrupted"}


@dataclass
class _Pushed:
    """A frame in the session: its frame-table entry, as prepared for the model, and its
    inspection, if the agent has a tool runtime; and the replies started on it, for its
    inspection's one record: how many are on, the outputs of those that ended, in order, their
    time, and how the last ended."""

    record: FrameRecord
    value: ProviderObservation
    execution: dict | None
    running: int = 0
    outputs: list[str] = field(default_factory=list)
    latency: float = 0.0
    outcome: str = "no_reply"


@dataclass
class _Reply:
    """One reply, from its StartGeneration until the kernel accepts or discards it."""

    generation: int
    decision: int
    own: _Pushed  # the frame it started on
    cancel: threading.Event = field(default_factory=threading.Event)  # set once it is aborted
    failed: bool = False  # generate() raised for it
    handle: GenerationHandle | None = None
    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)  # each function call, as its JSON action
    usage: dict | None = None
    # Its terminal boundaries, as a turn's (RealtimeRuntime._run_turn): the provider history
    # state of the last, and whether a turn_complete, with its usage, or an interrupted came.
    terminal: ProviderHistoryState = ProviderHistoryState.UNKNOWN
    complete: bool = False
    terminal_usage: bool = False
    interrupted: bool = False
    text_done: bool = False  # its text action was parsed, when its text was completed
    protocol_failure: str | None = None  # the first
    # Each parsed call's id and grounding (parse_grounded_action), in order; the grounding is
    # None where nothing was resolved against a frame.
    grounding: list[dict] = field(default_factory=list)
    based_on: str | None = None  # the observation id of the frame its first action names
    # Where in the blocks its output began, which its output goes to once accepted: before the
    # feedback on its actions and the frames pushed since.
    position: int | None = None

    @property
    def output(self) -> str:
        return "\n".join(self.calls) if self.calls else "".join(self.text)

    @property
    def said_anything(self) -> bool:
        """Text, reasoning, a call or usage came, as a turn's received output
        (harnesses/default/harness.py)."""
        return bool(
            self.text or self.reasoning or self.calls or self.usage is not None or self.text_done
        )


class NativeStreamingAgent(VisionAgentBase):
    """The agent of a native-streaming episode on a live driver that takes ``streaming=True``."""

    supports_unbounded_output = True
    interaction_mode = "streaming"

    def __init__(self, config, log_dir=None, *, trace, clock=None):
        # The episode's operational clock; on the wall clock it is the semantic one as well.
        self.clock = clock or SystemClock()
        self._lock = threading.Lock()  # the blocks and the replies; the shell's threads share them
        # Held, on the execution thread, while a push readies the session, while a reply
        # readies it and is asked for, and while close() closes it: no push reopens the session
        # between a reply's two steps, or once it is closed.
        self._readying = asyncio.Lock()
        self._closed = False
        # Set once close() starts: a push readying the session then gives up, its backoff
        # included, instead of holding the lock close() waits for.
        self._closing = self.clock.event()
        self._session_known: str | None = None  # the last SessionStateChanged's state
        self._blocks: list[ContentBlock] = []
        self._pushed: dict[str | None, _Pushed] = {}  # by observation id
        self._replies: dict[int, _Reply] = {}  # by the kernel's generation number
        # The replies on the session: asked for, and not yet accepted, discarded or aborted.
        # Readying the session under one would close its connection, or rebuild the session
        # without its output. Each push waiting for them holds a gate, opened once none is.
        self._on_session: set[int] = set()
        self._gates: list[threading.Event] = []
        # What prepares the session: a readying ("ready"), a push waiting for the renewal
        # ("renewal").
        self._preparing: set[str] = set()
        self._aborted_early: set[int] = set()  # aborted before their worker started generate()
        # What each action asked for (action_text), by its outcome's (decision, call id).
        self._requested: dict[tuple[int, str], str] = {}
        # The generation of each call an action was posted for, which a cancellation names.
        self._generation_of: dict[str, int] = {}
        super().__init__(config, log_dir, trace=trace)

    def _setup_backend(self):
        self.driver = adapter_class(self.config.adapter_type)(
            self.config, clock=self.clock, streaming=True
        )
        self.driver.bind_trace(self._trace)
        self._context_policy = current_context_policy().for_path(
            self.config.adapter_type, self.config.provider_options, self.config.context_preset
        )
        self._execution = ExecutionThread(
            self.clock, f"execution-thread-{self.driver.capabilities.protocol_family}"
        )

    def bind_inbox(self, post, origin: float) -> None:
        self._post, self._origin = post, origin

    def initialize(self, task_instruction: str, initial_observation: Observation):
        super().initialize(task_instruction, initial_observation)
        self.frames = FrameTable()
        self.interaction_profile = INTERACTION_PROFILES[
            self._context_policy.prompts[self.task_category]  # context policy item 11
        ]
        fixed = self.interaction_profile.system_instruction(self)
        self._system = system_block(fixed, self.clock.time_ns())
        channel = self.driver.capabilities.action_channel
        self._answer = AnswerRequirements(
            action_channel=channel,
            format_profile=self.interaction_profile.name,
            function_declarations=(
                tuple(gui_function_declarations(frame_reference=True))
                if channel == "function_call"
                else ()
            ),
        )
        self._call = CallParameters(
            model=self.config.model_name,
            reasoning_effort=self.config.reasoning_effort,
            output_token_cap=None,
        )
        self._seed = SessionSeed(
            system_instruction=fixed,
            model_name=self.model_name,
            native_output_limit=self.config.native_output_limit,
            fps=float(self.config.fps),
            answer=self._answer,
        )
        # The encoder's streaming setting (ModelConfig.streaming over its defaults); one that
        # keeps no frame count of its own, as Gemini Live leaves its context to the server, takes
        # the policy's.
        self._retained = self.driver.streaming_settings.get(
            "retained_frames", self._context_policy.visuals.keep
        )
        self._execution.start()
        with self._lock:
            view = self._view()
        with trace_scope(phase="initialize"):
            self._execution.run(self._open(view), timeout=_OPEN_TIMEOUT_S)

    def discard_pending_observations(self):
        """No frame waits in the agent: a frame goes into the session only when pushed."""

    def transport_output_token_cap(self, provider_cap):
        if self.driver.capabilities.maps_output_cap:
            return self.driver.transport_output_token_cap(provider_cap)
        return None if provider_cap is None else int(provider_cap)

    def _view(self, blocks: list[ContentBlock] | None = None) -> CanonicalRequest:
        """The session ``blocks`` make, the accepted ones by default; the caller holds the
        lock."""
        return history_view(
            self._blocks if blocks is None else blocks,
            policy=self._context_policy,
            system=self._system,
            retained_frames=self._retained,
            answer=self._answer,
            call=self._call,
        )

    def _record_session(self, sync, result=None, error=None) -> None:
        """The ``record`` of a session operation (SessionLifecycle), as a turn's
        (RealtimeRuntime._record_reconnect, _record_sync): a HistoryChanged for a reopened
        session, and for each sync, of the kind the driver's result reports; and what that left
        of the provider's context."""
        if sync is None:
            self._history_changed("rebuild", "recovery", {"reason": "reconnect"})
            self._session_state("exact", "reconnect", ProviderHistoryState.EXACT)
            return
        metadata = result.provider_metadata if result is not None else {}
        trigger = "recovery" if sync.reason == "runtime_recovery" else "provider"
        if metadata.get("session_resumed"):
            kind = "session_resumed"
        elif metadata.get("resume_error"):
            kind, trigger = "resume_fallback", "fallback"
        elif metadata.get("history_in_place_edit_fallback"):
            kind, trigger = "in_place_edit_fallback", "fallback"
        elif result is not None and (
            result.disposition == SyncDisposition.REBUILT
            or metadata.get("history_session_reconnected")
        ):
            kind = "rebuild"
        else:
            kind = "sync"
        self._history_changed(
            kind,
            trigger,
            {
                "reason": sync.reason,
                "force_rebuild": sync.force_rebuild,
                "disposition": result.disposition.value if result is not None else None,
                "history_state": result.history_state.value if result is not None else None,
                "detail": result.detail if result is not None else None,
                "provider_metadata": metadata,
            },
            error=repr(error)[:500] if error is not None else None,
        )
        if result is None:
            self._session_state("unknown", f"sync_error:{sync.reason}", None)
        elif kind == "session_resumed":
            # Exact for the driver, but restored from the resume handle, not sent on this
            # connection.
            self._session_state("unknown", "session_resumed", result.history_state)
        elif result.history_state == ProviderHistoryState.EXACT:
            self._session_state("exact", f"sync:{sync.reason}", result.history_state)
        else:
            self._session_state("unknown", f"sync:{sync.reason}", result.history_state)

    def _history_changed(self, kind: str, trigger: str, sync: dict, **fields) -> None:
        self._trace.capture(
            TraceEvent.HISTORY_CHANGED,
            {
                "kind": kind,
                "trigger": trigger,
                "policy_item": None,
                "policy_version": self._context_policy.version,
                "sync": sync,
                **fields,
            },
        )

    def _session_state(self, state, cause, provider_state, generation_id=None) -> None:
        """Record when the provider's context becomes unknown, or exact again, as a turn's
        session does (RealtimeRuntime._record_session_state)."""
        if state == self._session_known:
            return
        self._session_known = state
        self._trace.capture(
            TraceEvent.SESSION_STATE_CHANGED,
            {
                "protocol_family": self.driver.capabilities.protocol_family,
                "state": state,
                "cause": cause,
                "provider_history_state": provider_state.value if provider_state else None,
                "generation_id": generation_id,
            },
        )

    # ---- frames -------------------------------------------------------------------------------

    def push(self, observation: Observation) -> None:
        """Enter the frame in the frame table and put it into the session, readied first, and
        record it sent. Raises when it did not go in."""
        value, execution, record = bind_observation(
            self, observation, self.interaction_profile.frame_prompt, observation.tick_number
        )
        pushed = _Pushed(record, value, execution)
        key = observation.metadata.get("observation_id")
        try:
            with (
                trace_scope(phase="push", observation_id=value.observation_id),
                retry_cancellation(self._closing),
            ):
                while (gate := self._execution.run(self._push(key, pushed))) is not None:
                    if gate.is_set():
                        continue
                    # The session waits for its renewal, which is no push stuck in it, until
                    # the push has readied it after the gate (_push).
                    self._preparing_session("renewal", True)
                    self.clock.wait(gate, None)
        except BaseException:
            self._preparing_session("renewal", False)
            self._finish_inspection(pushed, "push_failed")
            raise

    async def _push(self, key: str | None, pushed: _Pushed) -> threading.Event | None:
        """Put the frame in, or, when it has to wait, the gate to wait at before trying again;
        one already open to try again at once.

        A session the provider asked to renew takes no frame before it is renewed, as Gemini
        Live's after a goAway, and renewing it would end the reply on it: the frame waits until
        that reply is off the session. That is decided here, on the execution thread, where the
        reader applies a goAway, and the gate opens once the reply is off (_off), whatever goes
        on the session after it; a goAway read while the push waits for the session, which
        refuses the frame unwritten (RenewalRequired), is taken the same way. The frames
        captured meanwhile wait in the policy's slot, not counted as failed pushes."""
        async with self._readying:
            self._stop_if_closed()
            with self._lock:
                view = self._view()
                on_session = bool(self._on_session)
                if on_session and self._renewal_due():
                    gate = self.clock.event()
                    self._gates.append(gate)
                    return gate
            if not on_session:
                await self._ready(view, renew=True)
        self._preparing_session("renewal", False)
        block = frame_block(pushed.record, pushed.value)
        with self._lock:
            view = self._view([*self._blocks, block])
        try:
            receipt = await self.driver.push_frame(pushed.value, view)
        except RenewalRequired:
            # A goAway read while the push waited for the session, after the renewal check:
            # nothing went out. Try again, once the reply renewing would end is off the session.
            gate = self.clock.event()
            with self._lock:
                if self._on_session:
                    self._gates.append(gate)
                else:
                    gate.set()
            return gate
        frame = pushed.record.frame
        # On the execution thread, where the reply's messages are read: an action read from
        # now on finds the frame sent, when the receipt says it was written (decision N43), and
        # counts it in an older frame's age.
        with self._lock:
            self.frames.record_sent(frame, self._wall(receipt.written_at))
            if receipt.confirmed_at is None:
                self.frames.record_ack_unknown(frame)
            else:
                self.frames.record_acked(frame, self._wall(receipt.confirmed_at))
            self._blocks.append(block)
            self._pushed[key] = pushed
            self._release_folded()
        # The kernel's records name the frame by its observation id, the session's by its block:
        # this ties the two, with the row's capture, send and confirmation on one clock
        # (evaluation/reporting.py summarize_streaming).
        record = pushed.record
        self._trace.capture(
            TraceEvent.FRAME_SENT,
            {
                "frame": frame,
                "observation_id": key,
                "captured_unix_s": record.captured_at,
                "sent_unix_s": record.sent_unix_s,
                "acked_unix_s": record.acked_unix_s,
            },
        )
        self._retire()
        return None

    def _wall(self, monotonic: float) -> float:
        """A reading of the clock's monotonic time, the drivers' (FunctionCall.received_at,
        PushReceipt), as wall seconds, as Clock.time() reads them."""
        return self.clock.time() - (self.clock.monotonic() - monotonic)

    def _renewal_due(self) -> bool:
        return bool(
            self.driver.capabilities.connection_renewal and self.driver.connection_renewal_reason()
        )

    def _release_folded(self) -> None:
        """Let go of the screenshot of the frame the newest push folded, as a turn-based history
        folds one (CanonicalTurn.fold_visual): the view folds it already. The caller holds the
        lock."""
        frames = [index for index, block in enumerate(self._blocks) if block.kind == "frame"]
        if len(frames) > self._retained:
            index = frames[-self._retained - 1]
            self._blocks[index] = replace(self._blocks[index], image=None)

    def _retire(self) -> None:
        """Finish the inspection of each frame no reply can start on any more, once the replies
        on it ended, and let go of the frame. The kernel starts a reply on the newest frame it
        was told was pushed: the newest, or the one before while it is not told yet."""
        with self._lock:
            past = list(self._pushed)[:-2]
            retired = [self._pushed.pop(key) for key in past if not self._pushed[key].running]
        for pushed in retired:
            self._finish_frame(pushed)

    def _finish_frame(self, pushed: _Pushed) -> None:
        """The one record of the frame's inspection, with every reply on it."""
        self._finish_inspection(
            pushed,
            pushed.outcome,
            action_latency_s=pushed.latency,
            raw_action_output="\n".join(pushed.outputs),
        )

    # ---- replies ------------------------------------------------------------------------------

    def generate(
        self, generation: int, decision: int, observation: Observation, budget: DecisionBudget
    ) -> PendingDecision:
        """StartGeneration: one reply, on the worker thread, until it ended. ``observation`` is
        the newest frame pushed: the result's, and the one an unparsable action falls back to."""
        with self._lock:
            reply = _Reply(
                generation, decision, self._pushed[observation.metadata.get("observation_id")]
            )
            self._replies[generation] = reply
            reply.own.running += 1
            if generation in self._aborted_early:
                self._aborted_early.remove(generation)
                reply.cancel.set()
        started = self.clock.monotonic()
        outcome = "completed"
        try:
            with (
                trace_scope(phase="decision", decision=decision, generation=generation),
                retry_cancellation(reply.cancel),
            ):
                self._execution.run(self._reply(reply, budget))
        except BaseException:
            if not reply.cancel.is_set():
                reply.failed = True
                outcome = "reply_failed"
                raise
            outcome = "cancelled"
        finally:
            latency = self.clock.monotonic() - started
            with self._lock:
                own = reply.own
                own.running -= 1
                own.outputs += [reply.output] if reply.output else []
                own.latency += latency
                own.outcome = outcome
            self._retire()
        return PendingDecision(generation, decision, self._result(reply, latency))

    async def _ready(self, view: CanonicalRequest, *, renew: bool) -> None:
        """Ready the session from ``view``, and renew it if the provider asked for that and
        ``renew``, as _retried tries it. Each attempt starts over from the session's state, which
        a failed one left closed or broken, so the next opens it again from ``view``.

        From the first attempt that starts preparing the session (session_preparing), through
        the backoffs and the attempts after it, until this returns or raises, the session is
        being prepared (_preparing_session)."""

        async def attempt() -> None:
            await self.driver.ensure_ready(self._seed, view, self._record_session)
            if renew:
                await self.driver.renew(self._seed, view, self._record_session)

        try:
            with session_preparing(lambda: self._preparing_session("ready", True)):
                await self._retried(view, attempt)
        finally:
            self._preparing_session("ready", False)

    async def _open(self, view: CanonicalRequest) -> None:
        """Open the session from ``view`` as the episode starts, as _retried tries it."""
        await self._retried(
            view, lambda: self.driver.connect(ConnectRequest(seed=self._seed, history=view))
        )

    async def _retried(self, view: CanonicalRequest, attempt) -> None:
        """Run ``attempt()``, which opens or readies the session from ``view``, and try it again
        on a failure the driver takes as transient, under the shared retry policy (decision
        39), as a turn's connect is: nothing was generated on the session yet. This is the only
        retry of opening a native-streaming session: neither encoder retries its streaming
        connect itself (decision 13)."""

        async def tried(number: int) -> None:
            await attempt()
            if number > 1:
                self._ready_finished(number, "ok")

        def record(number, exc, failure, delay):
            self._ready_finished(
                number, "failed" if delay is None else "retry", exc, failure, delay
            )
            if delay is None:
                raise TransportFailureError(failure, exc, number - 1) from exc

        frames = [item.block.id for item in view.items if item.block.kind == "frame"]
        await RequestRetry(
            f"{self._trace.attempt_id}:{frames[-1] if frames else 'empty'}:ready",
            self.driver.classify_transport_error,
            record,
            cancelled=lambda number, exc: self._ready_finished(number, "cancelled", exc),
            clock=self.clock,
        ).run_async(tried)

    def _preparing_session(self, who: str, preparing: bool) -> None:
        """``who``, a readying or a push waiting for the renewal, starts or stops preparing the
        session: the kernel hears when the session starts being prepared, and when nothing
        prepares it any more (SessionPreparing, SessionPrepared), in that order."""
        with self._lock:
            was = bool(self._preparing)
            if preparing:
                self._preparing.add(who)
            else:
                self._preparing.discard(who)
            if was != bool(self._preparing):
                self._post(SessionPreparing() if preparing else SessionPrepared())

    def _ready_finished(self, number, outcome, error=None, failure=None, delay=None) -> None:
        """One ModelAttemptFinished per attempt to open or ready the session that failed
        (_retried), for the one that succeeded after them, and for one a cancellation stopped.
        One that succeeds at once leaves none."""
        with trace_scope(ready_attempt=number):
            self._trace.capture(
                TraceEvent.MODEL_ATTEMPT_FINISHED,
                {
                    "kind": "session_ready",
                    "outcome": outcome,
                    "failure_kind": failure.kind if failure is not None else None,
                    "error": repr(error)[:500] if error is not None else None,
                    "retry_delay_s": delay,
                },
            )

    def _stop_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("the streaming agent is closed")

    async def _reply(self, reply: _Reply, budget: DecisionBudget) -> None:
        """Ask for the reply and read it to its end. One that fails before the model said
        anything, as when its connection drops, is asked for again on the session readied afresh
        from the view, under the shared retry policy (decision 39), as a turn is
        (harnesses/default/harness.py, ``replayable``): none of its actions went to the kernel.
        Each failed attempt is recorded, and one that succeeded or was given up after them."""

        async def attempt(number: int) -> None:
            # Each attempt reads a stream of its own, to its own boundaries.
            reply.handle = None
            reply.terminal, reply.complete = ProviderHistoryState.UNKNOWN, False
            reply.terminal_usage = reply.interrupted = False
            try:
                await self._ask(reply, budget)
            except Exception:
                # The session may hold some of the reply, which the view does not: the next push
                # or reply rebuilds it (SessionLifecycle.ensure_ready). The encoder's goAway
                # deadline passing during the reply raises too: an incomplete reply is no result.
                self.driver.mark_dirty()
                with self._lock:
                    self._off(reply.generation)
                if reply.cancel.is_set():
                    # The abort closed the connection under it: no failure of the reply's.
                    raise RetryCancelled("generation abandoned during its reply") from None
                raise
            if number > 1:
                self._reply_finished(number, "ok")

        def classify(exc):
            # Readying the session spent the shared retry budget already (_ready).
            if isinstance(exc, TransportFailureError):
                return exc.failure
            return self.driver.classify_transport_error(exc)

        def record(number, exc, failure, delay):
            self._reply_finished(
                number, "failed" if delay is None else "retry", exc, failure, delay
            )

        def cancelled(number, exc):
            if number > 1:
                self._reply_finished(number, "cancelled", exc)

        await RequestRetry(
            f"{self._trace.attempt_id}:{reply.own.value.observation_id}:reply",
            classify,
            record,
            cancelled=cancelled,
            replayable=lambda exc: not reply.said_anything,
            clock=self.clock,
        ).run_async(attempt)

    async def _ask(self, reply: _Reply, budget: DecisionBudget) -> None:
        """One attempt: ask for the reply on the session readied from the view, and take its
        messages until it ended."""
        async with self._readying:
            self._stop_if_closed()
            while reply.handle is None:
                with self._lock:
                    view = self._view()
                await self._ready(view, renew=True)
                # Given up while its session was readied, which ensure_ready looks at only as it
                # reopens or rebuilds the session (SessionLifecycle): asked for then, it would
                # run beside the next one (context.py _run_turn).
                if reply.cancel.is_set():
                    raise RetryCancelled("generation abandoned before its reply")
                try:
                    reply.handle = await self.driver.request_reply(
                        GenerationRequest(
                            generation_id=f"streaming-{reply.generation}",
                            provider_output_token_cap=budget.provider_output_tokens,
                            transport_output_token_cap=budget.transport_output_tokens,
                            clock_mode="wall",
                            canonical=view,
                            generation=reply.generation,
                        )
                    )
                except RenewalRequired:
                    # A goAway read while the trigger waited for the session, after the
                    # renewal check: nothing went out, so renew and ask again.
                    continue
            with self._lock:
                self._on_session.add(reply.generation)
        async for event in self.driver.events(reply.handle):
            self._take(reply, event)
        if not reply.cancel.is_set():
            self._ended(reply)

    def _reply_finished(self, number, outcome, error=None, failure=None, delay=None) -> None:
        """One ModelAttemptFinished per attempt of a reply that failed, as a readying attempt's
        (_ready_finished)."""
        with trace_scope(reply_attempt=number):
            self._trace.capture(
                TraceEvent.MODEL_ATTEMPT_FINISHED,
                {
                    "kind": "reply",
                    "outcome": outcome,
                    "failure_kind": failure.kind if failure is not None else None,
                    "error": repr(error)[:500] if error is not None else None,
                    "retry_delay_s": delay,
                },
            )

    def _ended(self, reply: _Reply) -> None:
        """The reply's stream ended, as a turn's does (RealtimeRuntime._run_turn): a session it
        left other than exact is rebuilt from the view before it is used again, and a function
        call reply that did not complete is no result (FunctionCallTurnCollector.finish). A text
        reply's action is parsed whatever the end, from the text that came."""
        if reply.terminal != ProviderHistoryState.EXACT:
            self.driver.mark_dirty()
            self._session_state(
                "unknown", "generation_terminal_state", reply.terminal, reply.handle.generation_id
            )
        if self._answer.action_channel == "function_call":
            # A call already handed over fixed its action: an interruption after it stands once
            # the terminal usage came.
            if not reply.complete or (
                reply.interrupted and not (reply.calls and reply.terminal_usage)
            ):
                raise RuntimeError("native tool accounting requires a complete reply")
        elif self._answer.action_channel == "text_json" and not reply.text_done:
            # No text was completed, as when the response was cut short: the reply's end stands
            # in for when it was.
            self._arrived(reply, reply.output, TEXT_CALL, self.clock.monotonic())

    def _take(self, reply: _Reply, event) -> None:
        """One message of the reply, on the execution thread as it is read."""
        if reply.position is None and (
            isinstance(event, FunctionCall)
            or (isinstance(event, TextDelta) and event.channel == "visible")
        ):
            with self._lock:
                reply.position = len(self._blocks)
        if isinstance(event, TextDelta):
            (reply.reasoning if event.channel == "reasoning" else reply.text).append(event.text)
        elif isinstance(event, UsageSnapshot):
            reply.usage = event.usage
        elif isinstance(event, TextDone):
            # The reply's action, in the text so far, arrived when the text was completed
            # (decisions 19 and 4.17); text after it is the reply's output only.
            if self._answer.action_channel == "text_json" and not reply.text_done:
                reply.text_done = True
                self._arrived(reply, reply.output, TEXT_CALL, event.received_at)
        elif isinstance(event, ProviderError):
            # A provider fault, not the model's: the run cannot be scored, as a turn's.
            if event.fatal:
                raise RuntimeError(event.detail)
        elif isinstance(event, GenerationBoundary):
            if event.kind in _TERMINAL:
                reply.terminal = event.history_state
            reply.complete |= event.kind == "turn_complete"
            reply.interrupted |= event.kind == "interrupted"
            if event.kind == "turn_complete":
                reply.terminal_usage = bool(event.metadata.get("terminal_usage_received"))
        elif isinstance(event, FunctionCall):
            action = {"action": event.name, **event.arguments}
            reply.calls.append(json.dumps(action, ensure_ascii=False))
            # A call without its frame is refused for that (frames.MISSING_FRAME); one its
            # function's schema rejects is no action, as a turn's call (FunctionCallTurnCollector).
            if "frame" in action:
                try:
                    canonical_function_action(event.name, event.arguments, frame_reference=True)
                except ValueError:
                    reply.protocol_failure = reply.protocol_failure or "invalid_action_output"
                    return
            self._arrived(reply, reply.calls[-1], event.call_id, event.received_at)
        elif isinstance(event, FunctionCallsCancelled):
            # By the generation each call came in, which may be an earlier reply's.
            withdrawn: dict[int, list[str]] = {}
            with self._lock:
                for call_id in event.call_ids:
                    if call_id in self._generation_of:
                        withdrawn.setdefault(self._generation_of[call_id], []).append(call_id)
            for generation, call_ids in withdrawn.items():
                self._post(CallsCancelled(generation, tuple(call_ids)))

    def _arrived(self, reply: _Reply, text: str, call_id: str, received_at: float) -> None:
        """Parse one action against the frame it names and post it, with the time its message
        came off the wire, ``received_at`` on the clock's monotonic reading."""
        arrived = self._wall(received_at)
        action, protocol = parse_grounded_action(
            self, text, reply.own.execution, arrived, reply.own.record
        )
        reply.grounding.append({"call_id": call_id, "grounding": protocol.get("grounding")})
        if protocol["protocol_failure"] is not None:
            reply.protocol_failure = reply.protocol_failure or protocol["protocol_failure"]
            return
        reference = protocol["frame_reference"]
        if action is None:
            arrival = ActionArrived(
                reply.generation,
                None,
                arrived - self._origin,
                call_id,
                rejection=protocol["rejection"],
                detail=reference["error"],
            )
        else:
            reply.based_on = reply.based_on or reference["observation_id"]
            arrival = ActionArrived(
                reply.generation,
                action,
                arrived - self._origin,
                call_id,
                reference["frame"],
                reference["age_s"],
                reference["age_frames"],
            )
        with self._lock:
            self._requested[(reply.decision, call_id)] = (
                "none" if action is None else action_text(action)
            )
            self._generation_of[call_id] = reply.generation
        self._post(arrival)

    def abort(self, generation: int) -> None:
        """AbortGeneration: give the reply up, interrupt it and close the session, and return
        once that is done, which fails a push stuck in the session. The next push or reply opens
        the session again from the view."""
        with self._lock:
            reply = self._replies.get(generation)
            if reply is None:
                self._aborted_early.add(generation)
            else:
                reply.cancel.set()
        with trace_scope(phase="abort", generation=generation):
            self._execution.run(
                self.driver.abort(None if reply is None else reply.handle),
                timeout=_SESSION_TIMEOUT_S,
            )
        with self._lock:
            self._off(generation)

    def _off(self, generation: int) -> None:
        """The reply is no longer on the session: once none is, open the gates of the pushes
        that waited for it. The caller holds the lock."""
        self._on_session.discard(generation)
        if not self._on_session:
            self._open_gates()

    def _open_gates(self) -> None:
        """The caller holds the lock."""
        for gate in self._gates:
            gate.set()
        self._gates.clear()

    def accept(self, generation: int) -> None:
        """AcceptResult: the reply's output joins the view where it began, and its usage the
        episode's. A failed reply, which the kernel accepts too, joins nothing, as a failed
        turn."""
        with self._lock:
            reply = self._replies.pop(generation)
            self._off(generation)
            if reply.output and not reply.failed:
                self._blocks.insert(
                    reply.position,
                    output_block(
                        reply.decision, reply.output, reply.based_on, self.clock.time_ns()
                    ),
                )
        if reply.failed:
            self._log_reply(reply, "reply_failed")
            return
        self._accumulate_usage(reply.usage)
        self._log_reply(reply, "reply")

    def discard(self, generation: int) -> None:
        """DiscardResult: a reply given up adds nothing."""
        with self._lock:
            reply = self._replies.pop(generation)
            self._off(generation)
        self._log_reply(reply, "reply_discarded")

    def commit(self, outcome: Outcome) -> None:
        """Commit: the outcome's feedback joins the view, on the action its call handed over, or
        on the reply when it came to no action."""
        with self._lock:
            requested = self._requested.pop((outcome.decision, outcome.call_id), "none")
            feedback = execution_feedback(outcome, requested)
            self._blocks.append(
                feedback_block(outcome.decision, outcome.call_id, feedback, self.clock.time_ns())
            )
        self.action_history.append(
            {
                "decision": outcome.decision,
                "call_id": outcome.call_id,
                "requested": requested,
                "executed": action_text(outcome.action) if outcome.action else "none",
                "result": outcome.status,
                "detail": outcome.detail,
            }
        )

    def _log_reply(self, reply: _Reply, kind: str) -> None:
        self._log_entry(
            {
                "type": kind,
                "decision": reply.decision,
                "generation": reply.generation,
                "observation_id": reply.own.value.observation_id,
                "model_output": reply.output,
                "thinking": "".join(reply.reasoning),
                "protocol_failure": reply.protocol_failure,
                "grounding": reply.grounding,
                "usage": reply.usage,
            }
        )

    def _result(self, reply: _Reply, latency: float) -> dict:
        """What the shell and the records read of a reply that ended; its actions went to the
        kernel as they arrived."""
        return {
            "decision": reply.decision,
            "generation": reply.generation,
            "observation_id": reply.own.value.observation_id,
            "inspection": reply.own.value.inspection,
            "raw": reply.output,
            "thinking": "".join(reply.reasoning),
            "usage": reply.usage,
            "latency": latency,
            "action_protocol": {"protocol_failure": reply.protocol_failure},
            **self._protocol_token_fields(reply.usage),
        }

    def _finish_inspection(self, pushed: _Pushed, outcome: str, **fields) -> None:
        if self.tool_runtime is not None and pushed.execution is not None:
            self.tool_runtime.finish(pushed.execution, outcome=outcome, **fields)

    # ---- the end ------------------------------------------------------------------------------

    def close(self):
        """Give up the replies left, which the kernel neither accepts nor discards once the
        episode ended, as DefaultHarness.close discards them; close the session once no push is
        readying it, and take no push or reply after; then stop the execution thread."""
        self._closing.set()
        with self._lock:
            left = list(self._replies.values())
            self._replies.clear()
        for reply in left:
            reply.cancel.set()
            self._log_reply(reply, "reply_discarded")
        try:
            # The runner closes an agent whose initialize failed, before or after the execution
            # thread started; one that never started has no session to close.
            if self._execution.running:
                try:
                    with trace_scope(phase="close"):
                        self._execution.run(self._close(), timeout=_SESSION_TIMEOUT_S)
                except Exception:
                    # As a turn's close (RealtimeRuntime.close): the episode's outcome is settled
                    # by now, and stopping the execution thread below ends what is left.
                    logger.warning("closing the streaming session failed", exc_info=True)
        finally:
            try:
                self._execution.stop()
                # A reply's worker still in generate(), which stopping the execution thread
                # wakes, retires its frame meanwhile: it finds none left.
                with self._lock:
                    left = list(self._pushed.values())
                    self._pushed.clear()
                for pushed in left:
                    self._finish_frame(pushed)
            finally:
                super().close()

    async def _close(self) -> None:
        async with self._readying:
            self._closed = True
            with self._lock:
                self._open_gates()  # a push waiting at one tries again and finds the agent closed
            await self.driver.close()
