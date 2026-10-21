"""Drive scripted Choice Decisions into an agent, in the order the legacy turn loop calls it.

As in steps.py: per decision the loop (runner.py _run_stream_legacy) captures a frame, binds the
budget, prefills and generates (its _decide), hands the result to the trajectory writer
(episode_records.py finalize_decision_audit) and reports the execution result (its _apply). In
addition each decision loads its own script into the stub, every question attempt is recorded
with an allow-list of headers, and an invalid decision ends the episode: run_stream invalidates
the run without committing an action and stops deciding (kernel/turns.py _unusable).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .build import Episode
from .stub_decisions import DecisionsStub

# Headers that change how the endpoint treats a request. The user agent, transfer headers and the
# host port are machine- or version-dependent and are not compared; names are lower-cased.
HEADER_ALLOW_LIST = ("authorization", "content-type")


@dataclass(frozen=True)
class Decision:
    """One decision: the stub entries for its question attempts, and its execution result."""

    script: tuple
    status: str = "applied"
    detail: str = ""


def request_record(received) -> dict:
    """The baseline ignores key order, but a question's criteria order is the order of its
    options, taken from the candidate order (providers/decisions/selection.py:165-177). It is
    kept as a list beside the body."""
    body = json.loads(received.body)
    return {
        "path": received.path,
        "headers": {name: received.headers.get(name) for name in HEADER_ALLOW_LIST},
        "body": body,
        "criteria_order": list(body["questions"]["next_action"]["criteria"]),
    }


def run_decisions(
    episode: Episode, decisions: list[Decision], stub: DecisionsStub
) -> tuple[list[dict], list[int]]:
    """Return the step records and, per decision, how many script entries were left unused."""
    agent = episode.agent
    records, unused = [], []
    for k, step in enumerate(decisions):
        observation = episode.frames.capture(k)
        first_request = len(stub.requests)
        history_length = len(agent.action_history)
        stub.load(step.script)
        agent.configure_decision(episode.budget(observation.tick_number))  # t_k: runner.py _decide
        agent.stream_prefill(observation)
        result = agent.stream_generate(tick=k)
        agent.finalize_decision_audit(result)  # consumes the trajectory hand-off entry
        unused.append(stub.unload())
        invalid = bool(result.get("invalid_reason"))
        if not invalid:
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
                "requests": [request_record(r) for r in stub.requests[first_request:]],
                "result": result,
                "commit": None if invalid else {"status": step.status, "detail": step.detail},
                # A failed decision adds no action (harnesses/default/harness.py:1357-1397).
                "action_history_entry": (
                    dict(agent.action_history[-1])
                    if len(agent.action_history) > history_length
                    else None
                ),
                "token_usage": agent.token_usage,
            }
        )
        if invalid:
            break
    return records, unused
