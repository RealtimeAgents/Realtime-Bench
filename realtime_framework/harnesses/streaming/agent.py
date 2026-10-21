"""The native-streaming agent: the streaming shell's agent (evaluation/streaming_shell.py
StreamingKernelAgent) for a live session that frames go into as they are captured, and whose
replies hand their actions over as they come off the wire (phase2.md, native streaming).

The agent keeps the content blocks the session should hold, in the order it accepted them, and
gives every push and connection operation the view they make (history.py), so that a session
rebuilt from the view holds what the live one did. Every driver operation runs on the agent's one
execution thread; the session's lifecycle is the driver's (providers/session_lifecycle.py). Each
push and each reply readies the session first, and renews the connection the provider asked to
renew, except a push while a reply is on: renewing would end it.

A push enters the frame in the frame table, which draws its number on the screenshot, and records
it as sent once the driver put it into the session. A frame whose push failed keeps its number and
stays out of the view (frames.py).

A reply starts on the view. Each function call is parsed as its message is read, against the frame
it names, and posted to the kernel with the time its message came off the wire, in semantic
seconds from the episode's origin; one whose frame cannot be found arrives refused. One that is no
action is the reply's protocol failure. Where the model answers in text, the action is parsed once
the reply ended. An accepted reply's output joins the view; a discarded one adds nothing. Each
outcome the kernel commits joins it as its execution feedback (feedback.py), where the policy
places it.
"""

from __future__ import annotations

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
from realtime_framework.entity.events import FunctionCall, TextDelta, UsageSnapshot
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationHandle,
    GenerationRequest,
    PendingDecision,
    ProviderObservation,
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
from realtime_framework.evaluation.kernel.streaming import ActionArrived
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
    handle: GenerationHandle | None = None
    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)  # each function call, as its JSON action
    usage: dict | None = None
    protocol_failure: str | None = None  # the first
    based_on: str | None = None  # the observation id of the frame its first action names

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
        self._blocks: list[ContentBlock] = []
        self._pushed: dict[str | None, _Pushed] = {}  # by observation id
        self._replies: dict[int, _Reply] = {}  # by the kernel's generation number
        self._aborted_early: set[int] = set()  # aborted before their worker started generate()
        # What each action asked for (action_text), by its outcome's (decision, call id).
        self._requested: dict[tuple[int, str], str] = {}
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
        # Until the frame count is a setting of its own (stream-plan 4.12): the encoder's, else
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
        """Enter the frame in the frame table and put it into the session, readied first. Raises
        when it did not go in."""
        value, execution, record = bind_observation(
            self, observation, self.interaction_profile.frame_prompt, observation.tick_number
        )
        pushed = _Pushed(record, value, execution)
        try:
            with trace_scope(phase="push", observation_id=value.observation_id):
                self._execution.run(self._push(observation.metadata.get("observation_id"), pushed))
        except BaseException:
            self._finish_inspection(pushed, "push_failed")
            raise
        self.frames.record_sent(record.frame, self.clock.time())

    async def _push(self, key: str | None, pushed: _Pushed) -> None:
        with self._lock:
            view = self._view()
            # Renewing the connection would end a reply on it, and one that ended is renewed
            # from once its output is accepted: meanwhile a session that asked to be renewed
            # refuses the frame.
            replying = bool(self._replies)
        await self._ready(view, renew=not replying)
        block = frame_block(pushed.record, pushed.value)
        with self._lock:
            view = self._view([*self._blocks, block])
        await self.driver.push_frame(pushed.value, view)
        with self._lock:
            self._blocks.append(block)
            self._pushed[key] = pushed

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
        with self._lock:
            view = self._view()
        await self._ready(view, renew=True)
        request = GenerationRequest(
            generation_id=f"streaming-{reply.generation}",
            provider_output_token_cap=budget.provider_output_tokens,
            transport_output_token_cap=budget.transport_output_tokens,
            clock_mode="wall",
            canonical=view,
            generation=reply.generation,
        )
        try:
            reply.handle = await self.driver.request_reply(request)
            async for event in self.driver.events(reply.handle):
                self._take(reply, event)
        except Exception:
            # The session may hold some of the reply, which the view does not: the next push or
            # reply rebuilds it (SessionLifecycle.ensure_ready). The encoder's goAway deadline
            # passing during the reply raises too: an incomplete reply is no result.
            self.driver.mark_dirty()
            raise
        if self._answer.action_channel == "text_json":
            # The encoder does not tell when the text was completed: the reply's end stands in.
            self._arrived(reply, reply.output, TEXT_CALL, self.clock.monotonic())

    def _take(self, reply: _Reply, event) -> None:
        """One message of the reply, on the execution thread as it is read."""
        if isinstance(event, TextDelta):
            (reply.reasoning if event.channel == "reasoning" else reply.text).append(event.text)
        elif isinstance(event, UsageSnapshot):
            reply.usage = event.usage
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

    def accept(self, generation: int) -> None:
        """AcceptResult: the reply's output joins the view, and its usage the episode's."""
        with self._lock:
            reply = self._replies.pop(generation)
            if reply.output:
                self._blocks.append(
                    output_block(reply.decision, reply.output, reply.based_on, self.clock.time_ns())
                )
        self._accumulate_usage(reply.usage)
        self._log_reply(reply, "reply")

    def discard(self, generation: int) -> None:
        """DiscardResult: a reply given up adds nothing."""
        with self._lock:
            reply = self._replies.pop(generation)
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
