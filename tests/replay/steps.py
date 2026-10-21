"""Drive scripted decisions straight into an agent, in the order the legacy turn loop calls it.

Per decision the loop (tests/differential/legacy_loop.py _run_stream_legacy) captures a frame, binds
the budget, prefills and generates (its _decide), hands the result to the agent's trajectory writer
(episode_records.py finalize_decision_audit) and reports the execution result (its _apply and
_commit). Its own clock, deadline and action execution are not reproduced: each step's status and
detail are scripted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .build import Episode
from .stub import ChatCompletionsStub


@dataclass(frozen=True)
class Step:
    """One decision: the model's reply, and the execution result the runner reports for it."""

    reply: str
    reasoning: str = ""
    status: str = "applied"
    detail: str = ""


def run_steps(episode: Episode, steps: list[Step], stub: ChatCompletionsStub) -> list[dict]:
    agent = episode.agent
    records = []
    for k, step in enumerate(steps):
        observation = episode.frames.capture(k)
        first_request = len(stub.requests)
        agent.configure_decision(episode.budget(observation.tick_number))  # t_k: runner.py _decide
        agent.stream_prefill(observation)
        result = agent.stream_generate(tick=k)
        agent.finalize_decision_audit(result)  # consumes the trajectory hand-off entry
        agent.commit_action(step.status, step.detail)
        records.append(
            {
                "record": "step",
                "step": k,
                "observation": {
                    "observation_id": observation.metadata["observation_id"],
                    "tick": observation.tick_number,
                    "timestamp": observation.timestamp,
                    "screenshot_sha256": observation.metadata["screenshot_sha256"],
                },
                "requests": [
                    {"path": received.path, "body": json.loads(received.body)}
                    for received in stub.requests[first_request:]
                ],
                "result": result,
                "commit": {"status": step.status, "detail": step.detail},
                "action_history_entry": dict(agent.action_history[-1]),
                "token_usage": agent.token_usage,
            }
        )
    return records
