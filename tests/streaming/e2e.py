"""One native-streaming episode end to end on the fake clock: a live session replayed from a
cassette (tests/streaming/clocked_ws.py CassetteServer), the replay's wall-clock environment
(tests/replay/fake_env.py) and the streaming shell (evaluation/streaming_shell.py), which the
runner's loop runs a streaming episode on (evaluation/runner.py run_stream).

``run_e2e`` takes the agent from a factory, ``agent_factory(clock, port, trace=trace)``, built
against the cassette's server on ``port`` with ``trace``, its write entry into the episode trace,
as runner.set_up_agent gives it (evaluation/agent_trace.py AgentTraceSink). The shell hands the
agent the environment's t0 (bind_inbox), from which an action's arrival is read in semantic
seconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from realtime_framework.clock import Clocks
from realtime_framework.entity.records import validate_episode_trace
from realtime_framework.evaluation import runner
from realtime_framework.evaluation.agent_trace import AgentTraceSink, close_agent_trace
from realtime_framework.evaluation.reporting import EpisodeTrace
from tests.fake_clock import FakeClock
from tests.replay.fake_env import FakeEnvironment
from tests.replay.frames import FrameSource, ReplayTask
from tests.streaming.clocked_ws import (
    REAL_LIMIT_S,
    CassetteServer,
    real_time_limit,
)
from tests.streaming.ws_cassette import RealtimeNormalizer

LIVE_PATH = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
SILENCE = bytes(3200)  # 100 ms of 16 kHz PCM, a Gemini Live activity's audio (gemini/live.py)


@dataclass
class Episode:
    """How the episode ended is left on ``env``, as main() reads it: ``_stop_reason``,
    ``_termination`` and ``_invalid_run``."""

    env: FakeEnvironment
    agent: object
    trace_path: object
    attempt_id: str  # the episode's in its trace (EpisodeTrace.attempt_id)
    records: list[dict]
    problems: list[str]  # the cassette server's: [] when the session matched it to the end
    logs: list[str]

    def kernel(self, kind: str, *fields) -> list:
        """The data of each kernel record of ``kind``, or the tuple of its ``fields``."""
        data = [
            r["payload"]["data"]
            for r in self.records
            if r["event"] == "KernelRecord" and r["payload"]["kind"] == kind
        ]
        return [tuple(d[f] for f in fields) for d in data] if fields else data


def run_e2e(agent_factory, cassette, tmp_path, *, duration_s: float, normalizer=None) -> Episode:
    """One streaming episode on a fresh fake clock: the cassette's server, the agent the factory
    builds against it, the replay's wall-clock environment at 1 fps and the runner's loop. Then
    the agent closes, its trace is closed as main() closes it, and the server reports what its
    cursor found."""
    clock = FakeClock()
    frames = FrameSource(ReplayTask(), text_observation=False, fps=1.0)
    # A snapshot that takes time: the reply the deadline gives up ends before the episode
    # finishes, so its result is discarded rather than racing the Finish (kernel/core.py).
    env = FakeEnvironment(
        frames, clock, fps=1.0, max_minutes=duration_s / 60, reply_latencies=[],
        snapshot_seconds=0.5,
    )  # fmt: skip
    path = tmp_path / "episode_trace.jsonl"
    trace, logs = EpisodeTrace(path, run_id="e2e", clock=clock), []
    sink = AgentTraceSink(trace, tmp_path / "episode_blobs.pack", clock=clock)
    server = CassetteServer(cassette, clock, normalizer=normalizer or RealtimeNormalizer())
    with server, real_time_limit(REAL_LIMIT_S, server.drop, clock.monotonic), clock.participant():
        agent = agent_factory(clock, server.port, trace=sink)
        try:
            timeline = runner.run_stream(
                agent, env, clock_mode="wall", duration_s=duration_s, env_fps=1.0, input_fps=1.0,
                log=logs.append, trace=trace, clocks=Clocks(clock, clock),
                interaction_mode="streaming",
            )  # fmt: skip
            assert timeline == []  # no turns
        finally:
            agent.close()
    assert close_agent_trace(sink) is None
    trace.close()
    validate_episode_trace(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return Episode(env, agent, path, trace.attempt_id, records, server.finish(), logs)
