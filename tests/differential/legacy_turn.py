"""The agent as today's loop calls it (legacy_loop.py), over DefaultHarness's kernel methods.

The loop binds the decision's budget, prefills its frame and generates on it, and the result is
written into the history at once, unless an abort cancelled the generation; then it reports the
outcome on the newest action record and turn, whichever decision they belong to. Its abort names
no generation: it cancels every one that has not ended and closes the session. The agent had
these calls itself until the kernel's shell was its only caller in production (decision 23).
"""

from __future__ import annotations

import itertools

from realtime_framework.entity.records import trace_scope


class LegacyTurn:
    """Wraps the agent; whatever else the loop reads of it is the agent's. The generations take
    numbers of their own."""

    def __init__(self, agent):
        self._agent = agent
        self._numbers = itertools.count(1)
        self._budget = None
        self._observation = None
        self._decision = None  # the newest generation's, for abort()'s trace

    def __getattr__(self, name):
        return getattr(self._agent, name)

    def configure_decision(self, budget) -> None:
        self._budget = budget

    def stream_prefill(self, observation) -> None:
        self._observation = observation

    def stream_generate(self, tick: int = 0) -> dict:
        agent, number = self._agent, next(self._numbers)
        self._decision = tick
        result = agent.generate(number, tick, self._observation, self._budget).result
        with agent._generations_lock:
            # close() discards a generation still running; it was dropped as generate() returned.
            held = any(generation.number == number for generation in agent._generations)
        if held:
            (agent.discard if result.get("stale") else agent.accept)(number)
        return result

    def commit_action(self, status: str, detail: str = "") -> None:
        agent = self._agent
        newest = agent.action_history[-1] if agent.action_history else None
        feedback = agent._report(newest, status, detail)
        turns = agent._runtime.turns
        if turns:
            agent._runtime.record_execution_feedback(feedback, turns[-1].decision)

    def abort(self) -> None:
        agent = self._agent
        agent._cancel()
        with trace_scope(phase="abort", decision=self._decision):
            agent._runtime.abort()
