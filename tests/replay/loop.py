"""Drive the real run_stream with an agent from build_episode against a FakeEnvironment.

run_stream gets the arguments main() passes on the wall clock (evaluation/runner.py:1771-1782).
Its own time.time() reads (runner.py:365-367, 382-415, 571, 756, 774) are pointed at the fake
clock, so action timestamps, frame ages and the deadline all read one clock. After the loop the
abandoned workers are drained (runner.py:1875-1880) and the agent is closed (runner.py:1881-1888),
in main()'s order. main() would also call the agent's start_episode (runner.py:1752-1754), which
no agent defines.

Any agent build_episode returns can be driven, given a stub that calls ``env.model_request`` as
its ``on_request`` hook (see fake_env.py).
"""

from __future__ import annotations

import re
import types
from collections import Counter
from unittest import mock

from realtime_framework.entity.records import TraceEvent, _jsonable
from realtime_framework.evaluation import runner

from . import baseline
from .build import Episode
from .fake_env import FakeEnvironment

# Values derived from the agent's real-time decision latency (harnesses/default/harness.py:1168,
# 1500), on top of the fields baseline.REAL_TIME_FIELDS already masks.
REAL_TIME_FIELDS = baseline.REAL_TIME_FIELDS | {
    "latency_s",  # InferenceCompleted copies the decision latency (runner.py:1062)
    "model_latency_s",  # so does the lifecycle record (runner.py:699)
    "deliberation_ticks",  # the timeline's latency / delta_s (runner.py:502-503, 764)
    "think_spent",  # the sum of decision latencies (runner.py:431, 777)
}
LOGGED_DELIBERATION = re.compile(r"δ=\S+")  # that latency in the decision log (runner.py:766)

# Lifecycle statuses of a decision whose action reached env.execute (runner.py:532-546).
EXECUTED = {"applied", "wait", "rejected"}


class RecordingTrace:
    """Stands in for EpisodeTrace (runner.py:1466) and keeps each payload as EpisodeTrace
    serializes it (evaluation/reporting.py:2028-2040)."""

    def __init__(self, env: FakeEnvironment):
        self._env = env

    def emit(self, event, payload=None):
        name = event.value if isinstance(event, TraceEvent) else str(event)
        self._env.log("trace", event=name, payload=_jsonable(payload or {}))

    def checkpoint(self):
        self._env.log("trace.checkpoint")


def run_loop(episode: Episode, env: FakeEnvironment) -> list:
    """Run one episode and return run_stream's timeline; ``env.calls`` holds everything else."""
    if episode.clock_mode != "wall":
        raise ValueError("FakeEnvironment has the wall clock only")
    agent = episode.agent
    commit_action = agent.commit_action

    def recording_commit(status, detail=""):
        env.log("agent.commit_action", status=status, detail=detail)
        return commit_action(status, detail)

    agent.commit_action = recording_commit  # what run_stream reports (runner.py:434-443, 674)

    def log(line):
        env.log("runner.log", line=LOGGED_DELIBERATION.sub("δ=<real time>", line))

    with mock.patch.object(runner, "time", types.SimpleNamespace(time=env.clock.time)):
        try:
            return runner.run_stream(
                agent,
                env,
                delta_s=1.0 / env.fps,
                max_ticks=env.max_minutes * 60 * env.fps,
                log=log,
                think_budget_s=None,
                per_move_cap_s=None,
                clock_mode="wall",
                tokens_per_frame=None,
                trace=RecordingTrace(env),
            )
        finally:
            env._drain_abandoned()
            agent.close()


def records(env: FakeEnvironment, agent, timeline: list) -> list[dict]:
    """One record per decision with its calls in order, then the episode outcome. Calls made
    from the evaluation snapshot on, including those of abandoned workers, go to the outcome."""
    decisions: dict[int, list[dict]] = {}
    for call in env.calls:
        if not call["finalizing"]:
            decisions.setdefault(call["decision"], []).append(_strip(call))
    out = [{"record": "decision", "decision": k, "calls": calls} for k, calls in decisions.items()]
    out.append(
        {
            "record": "outcome",
            # Latched by _finalize (runner.py:772-812).
            "stop_reason": env._stop_reason,
            "invalid_run": getattr(env, "_invalid_run", None),
            "termination_snapshot_captured": env._termination_snapshot_captured,
            "termination_snapshot": getattr(env, "_termination_snapshot", None),
            "active_end_unix_s": env._active_end_unix_s,
            "think_spent": env._think_spent,
            "frame_ages": env.frame_ages,
            "token_accounting": env._token_accounting.summary(),
            "timeline": [
                {
                    "decision": k,
                    "t_k": t_k,
                    "deliberation_ticks": d_ticks,
                    "tau_k": tau_k,
                    "action": action,
                }
                for k, t_k, d_ticks, tau_k, action in timeline
            ],
            "finalize_calls": [_strip(call) for call in env.calls if call["finalizing"]],
            # What the agent holds about each decision once the episode is over.
            "action_history": agent.action_history,
        }
    )
    return out


def _strip(call: dict) -> dict:
    return {name: value for name, value in call.items() if name not in {"decision", "finalizing"}}


def resolved(env: FakeEnvironment) -> dict[int, dict]:
    """Each decision's ActionResolved payload (runner.py:573-585)."""
    return {
        call["decision"]: call["payload"]
        for call in env.calls
        if call["call"] == "trace" and call["event"] == TraceEvent.ACTION_RESOLVED.value
    }


def commit_statuses(env: FakeEnvironment) -> list[tuple[int, str, str]]:
    return [
        (call["decision"], call["status"], call["detail"])
        for call in env.calls
        if call["call"] == "agent.commit_action"
    ]


def check_wall_clock_properties(env: FakeEnvironment, stop_reason: str) -> None:
    """phase1.md, 回放测试: on the wall clock no action runs after the deadline, no action runs
    twice, and the episode ends for the right reason. An action that arrives once the task has
    decided the episode counts as late too."""
    for execution in env.executions:
        assert execution["time"] < env.deadline, f"executed after the deadline: {execution}"
        assert execution["episode_over"] is None, f"executed after the episode: {execution}"
    per_decision = Counter(execution["decision"] for execution in env.executions)
    assert all(count == 1 for count in per_decision.values()), f"executed twice: {per_decision}"
    after_loop = [c for c in env.calls if c["call"] == "env.execute" and c["finalizing"]]
    assert not after_loop, f"executed after the loop ended: {after_loop}"
    for k, payload in resolved(env).items():
        ran = payload["status"] in EXECUTED or (
            payload["status"] == "invalid_run" and k in env.failures
        )
        assert per_decision[k] == int(ran), f"decision {k} is {payload['status']}"
    assert env._stop_reason == stop_reason
