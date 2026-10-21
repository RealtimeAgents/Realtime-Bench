from __future__ import annotations

# Base
from abc import ABC as ABC
from abc import abstractmethod as abstractmethod
from typing import TYPE_CHECKING

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.entity.model import PendingDecision

if TYPE_CHECKING:  # the entity layer does not import the evaluation package at run time
    from realtime_framework.evaluation.kernel.events import Outcome


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

    def configure_decision(self, budget: DecisionBudget) -> None:
        """Bind model-native generation limits for the next decision."""
        self._provider_output_token_cap = budget.provider_output_tokens
        self._transport_output_token_cap = budget.transport_output_tokens
        self._clock_mode = budget.clock_mode

    def stream_prefill(self, observation: Observation) -> None:
        raise NotImplementedError(f"{type(self).__name__} is not a frame-driven adapter")

    def stream_generate(self, tick: int = 0) -> dict:
        raise NotImplementedError(f"{type(self).__name__} is not a frame-driven adapter")

    # The event kernel's agent (evaluation/shell.py KernelAgent). These defaults run the legacy
    # turn: stream_generate writes the history itself and marks a result stale once the runtime's
    # epoch has moved on, so accepting and discarding have nothing left to do.

    def generate(
        self, generation: int, decision: int, observation: Observation, budget: DecisionBudget
    ) -> PendingDecision:
        self.configure_decision(budget)
        self.stream_prefill(observation)
        result = self.stream_generate(tick=decision)
        return PendingDecision(generation, decision, result, cancelled=bool(result.get("stale")))

    def accept(self, generation: int) -> None:
        pass

    def discard(self, generation: int) -> None:
        pass

    def commit(self, outcome: Outcome) -> None:
        self.commit_action(outcome.status, outcome.detail)


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
