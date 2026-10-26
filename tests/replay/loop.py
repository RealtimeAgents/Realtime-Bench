"""Drive the real run_stream with an agent from build_episode against a FakeEnvironment on the
wall clock or a TokenEnvironment on the token clock.

run_stream gets the arguments main() passes (evaluation/runner.py main), with the environment's
clock as its clocks: on the wall clock the fake clock (tests/fake_clock.py), so the kernel's shell,
its workers, action timestamps, frame ages and the deadline all run on one clock. After the loop
the abandoned workers are drained and the agent and its records are closed, in main()'s order,
and how the episode came out is read off what run_stream hands back, as main() reads it. main()
would also call the agent's start_episode, which no agent defines.

On the wall clock any agent build_episode returns on the same clock can be driven, given a stub
that serves on that clock and calls ``env.model_request`` as its ``on_request`` hook (see
fake_env.py); the calling thread takes part in the clock. On the token clock run_stream's
operational clock is TokenEnvironment's, whose recorded times stay at the origin and whose
readings are the world's seconds (token_env.py _RunnerClock).
"""

from __future__ import annotations

import re
from collections import Counter

from realtime_framework.clock import Clocks
from realtime_framework.entity.records import TraceEvent, _jsonable
from realtime_framework.evaluation import runner
from realtime_framework.evaluation.clock import TokenAccounting
from realtime_framework.evaluation.episode_end import EpisodeResult

from . import baseline
from .build import Episode
from .fake_env import FakeEnvironment

# Values derived from the agent's real-time decision latency (harnesses/default/harness.py
# DefaultHarness._generate), on top of the fields baseline.REAL_TIME_FIELDS already masks. On the token
# clock the last two come from the output tokens instead.
REAL_TIME_FIELDS = baseline.REAL_TIME_FIELDS | {
    "latency_s",  # InferenceCompleted copies the decision latency (episode_records.py)
    "model_latency_s",  # so does the lifecycle record (episode_records.py lifecycle_record)
    "deliberation_ticks",  # the timeline's latency times the env fps (shell.py _commit)
    "think_spent",  # the sum of decision latencies (kernel/turns.py _accept)
}
LOGGED_DELIBERATION = re.compile(r"δ=\S+")  # that latency in the decision log (shell.py _commit)

# Lifecycle statuses of a decision whose action reached env.execute. A wait is not one: the
# kernel lets its time pass (kernel/turns.py _start_wait).
EXECUTED = {"applied", "rejected"}


class RecordingTrace:
    """Stands in for EpisodeTrace (runner.py main) and keeps each record as EpisodeTrace
    serializes it (evaluation/reporting.py EpisodeTrace.emit): the payload and the kernel's
    fields beside it."""

    def __init__(self, env: FakeEnvironment):
        self._env = env

    def emit(self, event, payload=None, **fields):
        name = event.value if isinstance(event, TraceEvent) else str(event)
        self._env.log("trace", event=name, payload=_jsonable(payload or {}), **_jsonable(fields))

    def checkpoint(self):
        self._env.log("trace.checkpoint")


def run_loop(episode: Episode, env) -> EpisodeResult:
    """Run one episode and return how it came out, as run_stream hands it back; ``env.calls``
    holds everything else.

    ``env`` is a FakeEnvironment on the wall clock or a TokenEnvironment (token_env.py) on the
    token clock. Its ``fps`` is the environment's rate and the agent's config.fps the
    model-input rate, as main() passes them."""
    if episode.clock_mode != env.clock_mode:
        raise ValueError(f"a {env.clock_mode}-clock environment for a {episode.clock_mode} episode")
    agent = episode.agent
    commit = agent.commit

    def recording_commit(outcome):
        env.log("agent.commit", status=outcome.status, detail=outcome.detail)
        return commit(outcome)

    agent.commit = recording_commit  # what the shell reports (shell.py _commit)

    def log(line):
        if episode.clock_mode == "wall":  # the token clock's δ comes from the output tokens
            line = LOGGED_DELIBERATION.sub("δ=<real time>", line)
        env.log("runner.log", line=line)

    result = None
    try:
        result = runner.run_stream(
            agent,
            env,
            clock_mode=episode.clock_mode,
            duration_s=env.max_minutes * 60,
            env_fps=env.fps,
            input_fps=agent.config.fps,
            tokens_per_frame=episode.tokens_per_frame,
            think_budget_s=None,
            per_move_cap_s=None,
            log=log,
            trace=RecordingTrace(env),
            clocks=Clocks(env.clock, env.clock),
        )
        return result
    finally:
        if result is not None:
            result.drain_abandoned()
        episode.finish()


def records(env: FakeEnvironment, agent, result: EpisodeResult) -> list[dict]:
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
            # What the shell hands main() (evaluation/episode_end.py EpisodeResult).
            "stop_reason": result.stop_reason,
            "termination": result.termination_record,
            "invalid_run": result.invalid_run,
            "termination_snapshot_captured": result.snapshot_captured,
            "termination_snapshot": result.snapshot,
            "active_end_unix_s": result.active_end_unix_s,
            "think_spent": result.think_spent,
            "frame_ages": env.frame_ages,
            "token_accounting": TokenAccounting(result.settlements).summary(),
            "timeline": [
                {
                    "decision": k,
                    "t_k": t_k,
                    "deliberation_ticks": d_ticks,
                    "tau_k": tau_k,
                    "action": action,
                }
                for k, t_k, d_ticks, tau_k, action in result.timeline
            ],
            "finalize_calls": [_strip(call) for call in env.calls if call["finalizing"]],
            # What the agent holds about each decision once the episode is over.
            "action_history": agent.history.actions,
        }
    )
    return out


def _strip(call: dict) -> dict:
    return {name: value for name, value in call.items() if name not in {"decision", "finalizing"}}


def resolved(env: FakeEnvironment) -> dict[int, dict]:
    """Each decision's ActionResolved payload (episode_records.py action_resolved)."""
    return {
        call["decision"]: call["payload"]
        for call in env.calls
        if call["call"] == "trace" and call["event"] == TraceEvent.ACTION_RESOLVED.value
    }


def commit_statuses(env: FakeEnvironment) -> list[tuple[int, str, str]]:
    return [
        (call["decision"], call["status"], call["detail"])
        for call in env.calls
        if call["call"] == "agent.commit"
    ]


def check_wall_clock_properties(
    env: FakeEnvironment, result: EpisodeResult, stop_reason: str
) -> None:
    """phase1.md, 回放测试: on the wall clock no action runs after the deadline, no action runs
    twice, and the episode ends for the right reason. The task's end is known once a state read
    sees it, and the kernel reads right before landing an action unless it read at that same
    reading (kernel/turns.py _read_already): no action runs once the task has decided, nor after
    a read has seen the end."""
    for execution in env.executions:
        assert execution["time"] < env.deadline, f"executed after the deadline: {execution}"
        assert execution["episode_over"] is None, f"executed after the episode: {execution}"
    seen = next(
        (i for i, c in enumerate(env.calls) if c["call"] == "env.check_end" and "failed" not in c),
        len(env.calls),
    )
    late = [c for c in env.calls[seen:] if c["call"] == "env.execute"]
    assert not late, f"executed after a state read saw the end: {late}"
    per_decision = Counter(execution["decision"] for execution in env.executions)
    assert all(count == 1 for count in per_decision.values()), f"executed twice: {per_decision}"
    after_loop = [c for c in env.calls if c["call"] == "env.execute" and c["finalizing"]]
    assert not after_loop, f"executed after the loop ended: {after_loop}"
    for k, payload in resolved(env).items():
        ran = payload["status"] in EXECUTED or (
            payload["status"] == "invalid_run" and k in env.failures
        )
        assert per_decision[k] == int(ran), f"decision {k} is {payload['status']}"
    assert result.stop_reason == stop_reason
