"""Drive scripted decisions straight into an agent, in the order the kernel's shell calls it.

Per decision the shell (evaluation/shell.py) captures a frame, binds the budget and runs generate()
(its _start_generation), writes the result's trajectory row as it settles it (its _settle;
episode_records.py finalize_decision_audit), lets the agent accept it (AcceptResult) and reports the
execution result (its _commit). The kernel's clock, deadline and action execution are not
reproduced: each step's status and detail are scripted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from realtime_framework.evaluation.kernel.events import Outcome

from .build import Episode
from .stub import ChatCompletionsStub


@dataclass(frozen=True)
class Step:
    """One decision: the model's reply, and the execution result the runner reports for it."""

    reply: str
    reasoning: str = ""
    status: str = "applied"
    detail: str = ""


def decide(agent, k: int, observation, budget) -> dict:
    """Decision k on ``observation``: its result, accepted. The kernel numbers its generations
    from 1 and accepts the result of the one in flight, an invalid result too (kernel/core.py
    _dispatch)."""
    result = agent.generate(k + 1, k, observation, budget).result
    agent.finalize_decision_audit(result)  # consumes the trajectory hand-off entry
    agent.accept(k + 1)
    return result


def commit(agent, k: int, status: str, detail: str = "") -> None:
    """Report decision k's scripted outcome. Of its fields DefaultHarness.commit reads the
    decision, status and detail, and the action only for a DONE or FAIL the kernel ran as a wait,
    which no script replies."""
    agent.commit(Outcome(k, status, detail, False, 0.0, 0.0, 0, False, None, None, None, 0))


def run_steps(episode: Episode, steps: list[Step], stub: ChatCompletionsStub) -> list[dict]:
    agent = episode.agent
    records = []
    for k, step in enumerate(steps):
        observation = episode.frames.capture(k)
        first_request = len(stub.requests)
        # The budget at t_k (kernel/turns.py _captured, shell.py _start_generation).
        result = decide(agent, k, observation, episode.budget(observation.tick_number))
        commit(agent, k, step.status, step.detail)
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
