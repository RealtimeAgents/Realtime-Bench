"""StreamingAgent's kernel agent methods (evaluation/shell.py KernelAgent) run the legacy turn: a
generation configures the decision, prefills the frame and generates on the decision index, the
legacy runtime's stale mark says whether it was cancelled, and the outcome is commit_action's
feedback. Accepting and discarding leave the agent alone."""

from __future__ import annotations

import pytest

from realtime_framework.entity.actions import Action, Observation
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.entity.harness import StreamingAgent
from realtime_framework.entity.model import PendingDecision
from realtime_framework.evaluation.kernel.events import Outcome

BUDGET = DecisionBudget(
    runtime_output_tokens=512,
    provider_output_tokens=512,
    transport_output_tokens=512,
    clock_mode="token",
)
OBSERVATION = Observation(screenshot=b"", timestamp=2.0, tick_number=2)
OUTCOME = Outcome(
    decision=3,
    status="rejected",
    detail="lands outside the page",
    terminal_after_action=False,
    action_clock_position=2.0,
    next_observation_tick=3,
    dropped_intermediate_frames=0,
    post_action_rendered=False,
    frame_age_s=0.0,
    token_delay_frames=0.0,
    action_output_tokens=0,
    post_horizon_output_tokens=0,
)


class LegacyAgent(StreamingAgent):
    """A frame-driven adapter that records the legacy calls it receives."""

    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list = []

    def initialize(self, task_instruction, initial_observation):
        pass

    def discard_pending_observations(self):
        pass

    def configure_decision(self, budget):
        super().configure_decision(budget)
        self.calls.append(("configure_decision", budget))

    def stream_prefill(self, observation):
        self.calls.append(("stream_prefill", observation))

    def stream_generate(self, tick=0):
        self.calls.append(("stream_generate", tick))
        return self.result

    def commit_action(self, status, detail=""):
        self.calls.append(("commit_action", status, detail))


@pytest.mark.parametrize(
    ("result", "cancelled"),
    [({"action": Action.wait(), "latency": 0.5}, False), ({"action": None, "stale": True}, True)],
    ids=["current", "stale"],
)
def test_generate_runs_the_legacy_turn_on_the_decision_index(result, cancelled):
    agent = LegacyAgent(result)
    pending = agent.generate(7, 3, OBSERVATION, BUDGET)
    assert pending == PendingDecision(7, 3, result, cancelled=cancelled)
    assert pending.result is result
    assert agent.calls == [
        ("configure_decision", BUDGET),
        ("stream_prefill", OBSERVATION),
        ("stream_generate", 3),
    ]
    assert agent._provider_output_token_cap == 512


def test_accept_and_discard_leave_the_agent_alone_and_commit_reports_the_outcome():
    agent = LegacyAgent({})
    agent.accept(7)
    agent.discard(8)
    assert agent.calls == []
    agent.commit(OUTCOME)
    assert agent.calls == [("commit_action", "rejected", "lands outside the page")]


class FrameAgent(StreamingAgent):
    """A frame-driven adapter with only the methods the legacy loop requires: it reports no
    outcome and has nothing to interrupt (runner.py ``_commit`` and ``_bounded_call`` skip a
    missing commit_action and abort)."""

    def initialize(self, task_instruction, initial_observation):
        pass

    def discard_pending_observations(self):
        pass


def test_the_kernel_agent_methods_run_on_an_agent_with_only_the_required_ones():
    """StreamingAgent is a KernelAgent by itself: commit and abort, with a generation number or
    without, do nothing on an adapter that reports and interrupts nothing."""
    agent = FrameAgent()
    agent.commit(OUTCOME)
    agent.abort(7)
    agent.abort()
