"""The native-streaming agent: the streaming shell's agent (evaluation/streaming_shell.py
StreamingKernelAgent) for a live session that frames go into as they are captured, and whose
replies hand their actions over as they come off the wire (phase2.md, native streaming).

The agent keeps the content blocks the session should hold, in the order it accepted them, and
gives every push and connection operation the view they make (history.py), so that a session
rebuilt from the view holds what the live one did. Every driver operation runs on the agent's one
execution thread; the session's lifecycle is the driver's (providers/session_lifecycle.py).

A push enters the frame in the frame table, which draws its number on the screenshot, and records
it as sent once the driver put it into the session. A frame whose push failed keeps its number and
stays out of the view (frames.py).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from realtime_framework.clock import SystemClock
from realtime_framework.entity.actions import Observation, gui_function_declarations
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.model import ConnectRequest, ProviderObservation, SessionSeed
from realtime_framework.entity.records import TraceEvent, trace_scope
from realtime_framework.entity.request import (
    AnswerRequirements,
    CallParameters,
    CanonicalRequest,
    ContentBlock,
)
from realtime_framework.harnesses.default.execution_thread import ExecutionThread
from realtime_framework.harnesses.default.frames import FrameRecord, FrameTable
from realtime_framework.harnesses.default.harness import (
    INTERACTION_PROFILES,
    VisionAgentBase,
    bind_observation,
)
from realtime_framework.harnesses.default.request import system_block
from realtime_framework.harnesses.streaming.history import frame_block, history_view
from realtime_framework.providers.registry import adapter_class


@dataclass(frozen=True)
class _Pushed:
    """A frame in the session: its frame-table entry, as prepared for the model, and its
    inspection, if the agent has a tool runtime."""

    record: FrameRecord
    value: ProviderObservation
    execution: dict | None


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
                self._execution.run(self._push(pushed))
        except BaseException:
            self._finish_inspection(pushed, "push_failed")
            raise
        self.frames.record_sent(record.frame, self.clock.time())

    async def _push(self, pushed: _Pushed) -> None:
        with self._lock:
            view = self._view()
        await self.driver.ensure_ready(self._seed, view, self._record_session)
        block = frame_block(pushed.record, pushed.value)
        with self._lock:
            view = self._view([*self._blocks, block])
        await self.driver.push_frame(pushed.value, view)
        with self._lock:
            self._blocks.append(block)
            self._pushed[pushed.value.observation_id] = pushed

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
