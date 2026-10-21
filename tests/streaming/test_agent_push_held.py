"""A push the agent holds for the session's renewal (harnesses/streaming/agent.py ``push``), in a
whole native-streaming episode on the fake live encoder and the fake clock, as
test_agent_episode.py runs one: the kernel hears that it is held (kernel/streaming.py PushHeld),
so its watchdog gives no reply up meanwhile."""

from __future__ import annotations

from realtime_framework.clock import Clocks
from realtime_framework.evaluation.kernel.streaming import StreamingConfig
from realtime_framework.evaluation.streaming_shell import _StreamingShell, run_streaming_episode
from tests.fake_clock import FakeClock

from .fake_live import Call, GoAway, Reply, make_agent
from .test_agent_episode import _Env


def test_a_reply_longer_than_the_push_watchdog_after_a_go_away_runs_to_its_end(monkeypatch):
    """Gemini Live's goAway comes 1 s into a reply that goes on until 14 s, past the push watchdog
    of the frame that waits for the renewal since then. That frame is held, not stuck: the reply is
    not given up, its click executes, and the connection is renewed after it."""
    clock, records = FakeClock(), []
    record = _StreamingShell._record

    def recording(shell, entry):
        records.append(entry)
        record(shell, entry)

    monkeypatch.setattr(_StreamingShell, "_record", recording)
    click = Call(0.5, "c1", "click", {"x": 10, "y": 20, "button": "left", "frame": 1})
    with clock.participant():
        agent = make_agent(monkeypatch, clock)
        agent.driver.replies += [Reply((click, GoAway(1.0)), ends_at=14.0)]
        agent.driver.replies += [Reply(ends_at=1.0)] * 20
        env = _Env(clock)
        env.seconds["execute"] = 0.25
        run_streaming_episode(
            agent,
            env,
            StreamingConfig(duration_s=20.0),
            clocks=Clocks(clock, clock),
            log=lambda message: None,
        )
        agent.close()
    assert not env._invalid_run
    abandoned = [e.data for e in records if e.kind == "GenerationAbandoned"]
    assert all(data["reason"] == "deadline" for data in abandoned), abandoned
    assert [e.kind for e in records if e.kind in ("PushHeld", "PushResumed")][:2] == [
        "PushHeld",
        "PushResumed",
    ]
    assert ("c1", "applied") in [(a["call_id"], a["result"]) for a in agent.action_history]
    calls = [c[:2] for c in agent.driver.calls]
    assert ("interrupt", "streaming-1") not in calls
    assert calls.index(("go_away", "streaming-1")) < calls.index(("sync", "go_away"))
