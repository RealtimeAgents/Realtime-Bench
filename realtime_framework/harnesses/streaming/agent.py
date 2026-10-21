"""The native-streaming agent: the streaming shell's agent (evaluation/streaming_shell.py
StreamingKernelAgent) for a live session that frames go into as they are captured, and whose
replies hand their actions over as they come off the wire (phase2.md, native streaming).

The agent keeps the content blocks the session should hold, in the order it accepted them, and
gives every push and connection operation the view they make (history.py), so that a session
rebuilt from the view holds what the live one did. Every driver operation runs on the agent's one
execution thread; the session's lifecycle is the driver's (providers/session_lifecycle.py). Each
push and each reply readies the session first, and renews the connection the provider asked to
renew, except a push while a reply is on the session: readying it then would close the reply's
connection, or rebuild the session without the reply's output. A push the session refuses until
it is renewed, as Gemini Live's after a goAway, waits for that reply instead.

A push enters the frame in the frame table, which draws its number on the screenshot, and records
it as sent once the driver put it into the session. A frame whose push failed keeps its number and
stays out of the view (frames.py).

A reply starts on the view. Each function call is parsed as its message is read, against the frame
it names, and posted to the kernel with the time its message came off the wire, in semantic
seconds from the episode's origin; one whose frame cannot be found arrives refused. One that is no
action is the reply's protocol failure. Where the model answers in text, the action is parsed once
the reply ended. A call the provider cancels, of this reply or an earlier one, is withdrawn under
the generation it came in, once its action was posted. A reply ends as a turn does: a provider
fault fails it, a session it leaves other than exact is rebuilt before it is used again, and a
function call reply must complete.

An accepted reply's output joins the view where the output began, so that a rebuilt session shows
the model its actions before their results; a discarded one adds nothing. Each outcome the kernel
commits joins it as its execution feedback (feedback.py), where the policy places it.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field

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
    UsageSnapshot,
)
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationHandle,
    GenerationRequest,
    PendingDecision,
    ProviderObservation,
    RetryCancelled,
    SessionSeed,
    retry_cancellation,
)
from realtime_framework.entity.records import TraceEvent, trace_scope
from realtime_framework.entity.request import (
    AnswerRequirements,
    CallParameters,
    CanonicalRequest,
    ContentBlock,
)
from realtime_framework.evaluation.kernel.events import Outcome
from realtime_framework.evaluation.kernel.streaming import ActionArrived, CallsCancelled
from realtime_framework.harnesses.default.execution_thread import ExecutionThread
from realtime_framework.harnesses.default.frames import FrameRecord, FrameTable
from realtime_framework.harnesses.default.harness import (
    INTERACTION_PROFILES,
    VisionAgentBase,
    bind_observation,
    parse_grounded_action,
)
from realtime_framework.harnesses.default.request import system_block
from realtime_framework.harnesses.streaming.feedback import action_text, execution_feedback
from realtime_framework.harnesses.streaming.history import (
    feedback_block,
    frame_block,
    history_view,
    output_block,
)
from realtime_framework.providers.registry import adapter_class

# The call id the agent gives the one action of a reply in text (ActionArrived.call_id).
TEXT_CALL = "text"
# The boundaries that end a reply, whose provider history state is the session's after it.
_TERMINAL = {"generation_complete", "turn_complete", "interrupted"}


@dataclass(frozen=True)
class _Pushed:
    """A frame in the session: its frame-table entry, as prepared for the model, and its
    inspection, if the agent has a tool runtime."""

    record: FrameRecord
    value: ProviderObservation
    execution: dict | None


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
    protocol_failure: str | None = None  # the first
    based_on: str | None = None  # the observation id of the frame its first action names
    # Where in the blocks its output began, which its output goes to once accepted: before the
    # feedback on its actions and the frames pushed since.
    position: int | None = None

    @property
    def output(self) -> str:
        return "\n".join(self.calls) if self.calls else "".join(self.text)


class NativeStreamingAgent(VisionAgentBase):
    """The agent of a native-streaming episode on a live driver that takes ``streaming=True``."""

    supports_unbounded_output = True
    interaction_mode = "streaming"

    def __init__(self, config, log_dir=None, *, trace, clock=None):
        # The episode's operational clock; on the wall clock it is the semantic one as well.
        self.clock = clock or SystemClock()
        self._lock = threading.Lock()  # the blocks and the replies; the shell's threads share them
        # Held, on the execution thread, while a push readies the session and while a reply
        # readies it and is asked for: no push reopens the session between the two.
        self._readying = asyncio.Lock()
        self._blocks: list[ContentBlock] = []
        self._pushed: dict[str | None, _Pushed] = {}  # by observation id
        self._replies: dict[int, _Reply] = {}  # by the kernel's generation number
        # The replies on the session: asked for, and not yet accepted, discarded or aborted.
        # Readying the session under one would close its connection, or rebuild the session
        # without its output. ``_off_session`` is set while there is none.
        self._on_session: set[int] = set()
        self._off_session = self.clock.event()
        self._off_session.set()
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
            self._execution.run(self.driver.connect(ConnectRequest(seed=self._seed, history=view)))

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
        """The ``record`` of a session operation (SessionLifecycle): a HistoryChanged for each
        sync it ran and for a reopened session."""
        self._trace.capture(
            TraceEvent.HISTORY_CHANGED,
            {
                "kind": "reconnect" if sync is None else "sync",
                "reason": None if sync is None else sync.reason,
                "force_rebuild": None if sync is None else sync.force_rebuild,
                "disposition": None if result is None else result.disposition.value,
                "history_state": None if result is None else result.history_state.value,
                "error": None if error is None else repr(error)[:500],
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
        try:
            with trace_scope(phase="push", observation_id=value.observation_id):
                self._wait_for_renewal()
                self._execution.run(self._push(observation.metadata.get("observation_id"), pushed))
        except BaseException:
            self._finish_inspection(pushed, "push_failed")
            raise

    def _wait_for_renewal(self) -> None:
        """A session the provider asked to renew takes no frame before it is renewed, as Gemini
        Live's after a goAway, and renewing it would end the reply on it: the frame waits, on
        the push lane, until that reply is accepted, discarded or aborted. The frames captured
        meanwhile wait in the policy's slot, not counted as failed pushes."""
        while (
            self.driver.capabilities.connection_renewal and self.driver.connection_renewal_reason()
        ):
            with self._lock:
                if not self._on_session:
                    return
            self.clock.wait(self._off_session, None)

    async def _push(self, key: str | None, pushed: _Pushed) -> None:
        async with self._readying:
            with self._lock:
                view = self._view()
                on_session = bool(self._on_session)
            if not on_session:
                await self._ready(view, renew=True)
        block = frame_block(pushed.record, pushed.value)
        with self._lock:
            view = self._view([*self._blocks, block])
        await self.driver.push_frame(pushed.value, view)
        # On the execution thread, where the reply's messages are read: an action read from
        # now on finds the frame sent, and counts it in an older frame's age.
        with self._lock:
            self.frames.record_sent(pushed.record.frame, self.clock.time())
            self._blocks.append(block)
            self._pushed[key] = pushed
        # The kernel's records name the frame by its observation id, the session's by its block:
        # this ties the two (evaluation/reporting.py summarize_streaming).
        self._trace.capture(
            TraceEvent.FRAME_SENT, {"frame": pushed.record.frame, "observation_id": key}
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
                self._finish_inspection(reply.own, "reply_failed", raw_action_output=reply.output)
                raise
            outcome = "cancelled"
        latency = self.clock.monotonic() - started
        self._finish_inspection(
            reply.own, outcome, action_latency_s=latency, raw_action_output=reply.output
        )
        return PendingDecision(
            generation, decision, self._result(reply, latency), reply.cancel.is_set()
        )

    async def _ready(self, view: CanonicalRequest, *, renew: bool) -> None:
        """Ready the session from ``view``, and renew it if the provider asked for that and
        ``renew``."""
        await self.driver.ensure_ready(self._seed, view, self._record_session)
        if renew:
            await self.driver.renew(self._seed, view, self._record_session)

    async def _reply(self, reply: _Reply, budget: DecisionBudget) -> None:
        try:
            async with self._readying:
                with self._lock:
                    view = self._view()
                await self._ready(view, renew=True)
                # A push may have reopened the session for the reply after it was given up:
                # asked for then, it would run beside the next one (context.py _run_turn).
                if reply.cancel.is_set():
                    raise RetryCancelled("generation abandoned before its reply")
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
                with self._lock:
                    self._on_session.add(reply.generation)
                    self._off_session.clear()
            async for event in self.driver.events(reply.handle):
                self._take(reply, event)
            if not reply.cancel.is_set():
                self._ended(reply)
        except Exception:
            # The session may hold some of the reply, which the view does not: the next push or
            # reply rebuilds it (SessionLifecycle.ensure_ready). The encoder's goAway deadline
            # passing during the reply raises too: an incomplete reply is no result.
            self.driver.mark_dirty()
            with self._lock:
                self._off(reply.generation)
            raise

    def _ended(self, reply: _Reply) -> None:
        """The reply's stream ended, as a turn's does (RealtimeRuntime._run_turn): a session it
        left other than exact is rebuilt from the view before it is used again, and a function
        call reply that did not complete is no result (FunctionCallTurnCollector.finish). A text
        reply's action is parsed whatever the end, from the text that came."""
        if reply.terminal != ProviderHistoryState.EXACT:
            self.driver.mark_dirty()
        if self._answer.action_channel == "function_call":
            # A call already handed over fixed its action: an interruption after it stands once
            # the terminal usage came.
            if not reply.complete or (
                reply.interrupted and not (reply.calls and reply.terminal_usage)
            ):
                raise RuntimeError("native tool accounting requires a complete reply")
        elif self._answer.action_channel == "text_json":
            # The encoder does not tell when the text was completed: the reply's end stands in.
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
            try:
                action = canonical_function_action(
                    event.name, event.arguments, frame_reference=True
                )
            except ValueError:
                # The parser takes it as no action, as a turn's call (FunctionCallTurnCollector).
                action = {"action": event.name, **event.arguments}
            reply.calls.append(json.dumps(action, ensure_ascii=False))
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
        arrived = self.clock.time() - (self.clock.monotonic() - received_at)
        action, protocol = parse_grounded_action(
            self, text, reply.own.execution, arrived, reply.own.record
        )
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
            self._execution.run(self.driver.abort(None if reply is None else reply.handle))
        with self._lock:
            self._off(generation)

    def _off(self, generation: int) -> None:
        """The reply is no longer on the session; the caller holds the lock."""
        self._on_session.discard(generation)
        if not self._on_session:
            self._off_session.set()

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
        try:
            with trace_scope(phase="close"):
                self._execution.run(self.driver.close())
            self._execution.stop()
        finally:
            for pushed in self._pushed.values():
                self._finish_inspection(pushed, "no_reply")
            super().close()
