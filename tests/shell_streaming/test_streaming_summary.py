"""The streaming section of a run summary (evaluation/reporting.py summarize_streaming), read off
the episode trace of a streaming episode the shell ran on the fake clock (tests/shell_streaming/
fakes.py, whose default world tests/shell_streaming/test_streaming_shell.py describes)."""

from __future__ import annotations

import itertools

from realtime_framework.clock import Clocks
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.kernel.streaming import StreamingConfig
from realtime_framework.evaluation.reporting import EpisodeTrace, summarize_streaming
from realtime_framework.evaluation.streaming_shell import run_streaming_episode
from tests.fake_clock import FakeClock

from .fakes import Call, FakeAgent, FakeEnv, Reply

CLICK = Action(ActionType.CLICK, {"x": 1, "y": 2})
OPENAI = {"retained_frames": 20}
GEMINI = {"compression_trigger_tokens": 65_536, "compression_target_tokens": 32_768}


class ConfirmingAgent(FakeAgent):
    """A session that records each frame it pushes as OpenAI Realtime's session core does, through
    the agent's trace (providers/session_lifecycle.py _record_delivery): sent as the push starts,
    confirmed by the server as it returns. Then, as the native-streaming agent does, the frame-table
    number the frame went in under (harnesses/streaming/agent.py push)."""

    def __init__(self, clock, replies, trace):
        super().__init__(clock, replies)
        self.trace = trace
        self._sequence = itertools.count(1)

    def push(self, observation):
        frame = len(self.pushed) + 1
        block = f"f{frame}.frame"
        self._deliver("sent", block)
        super().push(observation)
        self._deliver("confirmed", block)
        observation_id = observation.metadata["observation_id"]
        self._capture(TraceEvent.FRAME_SENT, frame=frame, observation_id=observation_id)

    def _deliver(self, state, block):
        contents = [{"key": ["create", f"item-{block}"], "block": block}]
        self._capture(
            TraceEvent.SESSION_DELIVERY, state=state, connection_id="conn-1", contents=contents
        )

    def _capture(self, event, **fields):
        # What evaluation/agent_trace.py AgentTraceSink writes for a capture.
        self.trace.emit(
            event,
            {
                "source": "agent",
                "capture_seq": next(self._sequence),
                "captured_wall_time_ns": self.clock.time_ns(),
                "captured_monotonic_ns": self.clock.monotonic_ns(),
                "scope": None,
                **fields,
            },
        )


def _summary(tmp_path, replies, settings, *, confirming=False, duration_s=8.0):
    clock = FakeClock()
    path = tmp_path / "trace.jsonl"
    trace = EpisodeTrace(path, run_id="s", clock=clock)
    env = FakeEnv(clock)
    agent = ConfirmingAgent(clock, replies, trace) if confirming else FakeAgent(clock, replies)
    with clock.participant():
        run_streaming_episode(
            agent, env, StreamingConfig(duration_s=duration_s), clocks=Clocks(clock, clock),
            log=lambda line: None, trace=trace,
        )
    trace.close()
    return summarize_streaming(path, settings), env, agent


def test_the_summary_counts_frames_actions_and_replies(tmp_path):
    """Three clicks of one reply. The first executes from 1.775 s to 3.775 s; the second waits,
    and the third takes its place and executes once the read after the first is done."""
    replies = [Reply(3.0, (Call(0.9, CLICK, "a"), Call(1.2, CLICK, "b"), Call(1.5, CLICK, "c")))]
    summary, env, agent = _summary(tmp_path, replies, OPENAI, confirming=True)
    assert summary["schema"] == "realtime_bench.streaming_summary.v1"
    assert summary["context_control"] == "exact"
    frames = summary["frames"]
    assert (frames["captured"], frames["pushed"], frames["push_failed"]) == (
        env.captures,
        len(agent.pushed),
        0,
    )
    assert frames["dropped"] == sum(frames["dropped_by_reason"].values())
    assert frames["dropped"] == frames["captured"] - frames["pushed"]
    # Every frame the session pushed was confirmed as its push returned, 0.125 s after it went.
    assert (frames["confirmed"], frames["refused"]) == (len(agent.pushed), 0)
    confirm = frames["send_to_confirm_s"]
    assert (confirm["sample_count"], confirm["min"], confirm["max"]) == (
        len(agent.pushed),
        0.125,
        0.125,
    )
    assert frames["capture_to_push_s"]["sample_count"] == len(agent.pushed)
    assert frames["capture_to_push_s"]["min"] == 0.125  # captured at 0.75 s, pushed at 0.875 s

    actions = summary["actions"]
    assert actions["arrived"] == 3
    assert actions["outcomes"] == {"applied": 2, "superseded": 1}
    # Each click named the newest frame pushed when it arrived: the first frame, pushed 0.9 s
    # before the first click, then the frames pushed at 1.875 s and 2.375 s.
    age = actions["frame_age_s"]
    assert (age["sample_count"], age["min"], age["max"]) == (3, 0.0, 0.9)
    assert actions["frame_age_frames"]["max"] == 0
    # The first click executed as it arrived; each ran the fake's 2 s.
    assert actions["arrival_to_start_s"]["sample_count"] == 2
    assert actions["arrival_to_start_s"]["min"] == 0.0
    assert (actions["execution_s"]["min"], actions["execution_s"]["max"]) == (2.0, 2.0)

    replies = summary["replies"]
    assert replies["started"] == len(agent.started)
    assert replies["abandoned"] == {"deadline": 1}


def test_a_session_that_confirms_no_frame_summarizes_none(tmp_path):
    """Gemini Live's compression makes the context control approximate, and its session confirms
    no frame (providers/gemini/session.py Delivery)."""
    summary, _, agent = _summary(tmp_path, [], GEMINI, duration_s=4.0)
    assert summary["context_control"] == "approximate"
    frames = summary["frames"]
    assert frames["pushed"] == len(agent.pushed) > 0
    assert (frames["confirmed"], frames["send_to_confirm_s"]["sample_count"]) == (0, 0)
    assert summary["actions"]["arrived"] == 0
    assert summary["actions"]["outcomes"] == {}


def test_a_run_without_its_configuration_has_no_context_control(tmp_path):
    """A run that failed before its configuration was resolved, and its trace holds no episode."""
    path = tmp_path / "trace.jsonl"
    EpisodeTrace(path, run_id="s", clock=FakeClock()).close()
    summary = summarize_streaming(path, None)
    assert summary["context_control"] is None
    assert (summary["frames"]["captured"], summary["replies"]["started"]) == (0, 0)
