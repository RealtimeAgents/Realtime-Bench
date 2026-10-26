from __future__ import annotations

# Base
from abc import ABC as ABC
from abc import abstractmethod as abstractmethod

from realtime_framework.entity.actions import Observation


class StreamingAgent(ABC):
    """Unified interface for all model adapters.

    All models, regardless of their API type (streaming video, chat-based image,
    tool-use), implement this interface. The framework treats every model as a
    streaming agent that receives frames and emits actions.
    """

    @abstractmethod
    def initialize(self, task_instruction: str, initial_observation: Observation):
        """Called once at task start with the instruction and first frame.

        The task_instruction is the problem statement text. The agent should
        understand what it needs to accomplish from this + the visual context.
        """

    @abstractmethod
    def discard_pending_observations(self):
        """Drop frames captured before the scored wall-clock episode starts.

        The maintained runner warms the adapter before restarting a wall-clock episode; that
        provisional frame must not leak into the first scored decision.
        """

    def abort(self, generation: int) -> None:
        """Interrupt the event kernel's generation ``generation`` in flight. An adapter with
        nothing to interrupt does nothing."""


from dataclasses import dataclass as dataclass

from realtime_framework.entity.actions import Action


@dataclass(frozen=True)
class Decision:
    action: Action
    action_text: str | None
    valid: bool
    protocol_failure: str | None = None
    parse_mode: str | None = None
    candidate_count: int = 0

    def metadata(self) -> dict:
        return {
            "source": (
                "model_action"
                if self.valid and self.parse_mode == "canonical_json"
                else "recovered_model_action"
                if self.valid
                else "invalid_action_fallback_wait"
            ),
            "valid_action": self.valid,
            "action_text": self.action_text,
            "protocol_failure": self.protocol_failure,
            "parse_mode": self.parse_mode,
            "candidate_count": self.candidate_count,
        }
