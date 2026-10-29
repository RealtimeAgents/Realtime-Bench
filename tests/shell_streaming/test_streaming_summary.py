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


class ReceiptAgent(FakeAgent):
    """A session whose push returns once the server confirmed the frame, as OpenAI Realtime's
    does, or, with ``confirms`` False, once it is written, as Gemini Live's does; and that records
    each frame it pushed as the native-streaming agent does (harnesses/streaming/agent.py push):
    its frame-table number, and its row's capture, send and confirmation on the wall clock, the
    send being when the push wrote it (decision N43)."""

    def __init__(self, clock, replies, trace, *, confirms=True):
        super().__init__(clock, replies)
        self.trace, self.confirms = trace, confirms
        self._sequence = itertools.count(1)

    def push(self, observation):
        frame, sent = len(self.pushed) + 1, self.clock.time()
        if self.confirms:
            super().push(observation)
        else:
            self.pushed.append((observation.metadata["observation_id"], sent))
        capture(
            self.trace, self.clock, self._sequence, TraceEvent.FRAME_SENT, frame=frame,
            observation_id=observation.metadata["observation_id"],
            captured_unix_s=observation.timestamp, sent_unix_s=sent,
            acked_unix_s=self.clock.time() if self.confirms else None,
        )  # fmt: skip


def capture(trace, clock, sequence, event, **fields):
    """What evaluation/agent_trace.py AgentTraceSink writes for an agent's record."""
    trace.emit(
        event,
        {
            "source": "agent",
            "capture_seq": next(sequence),
            "captured_wall_time_ns": clock.time_ns(),
            "captured_monotonic_ns": clock.monotonic_ns(),
            "scope": None,
            **fields,
        },
    )


def _summary(tmp_path, replies, settings, *, confirms=True, duration_s=8.0):
    clock = FakeClock()
    path = tmp_path / "trace.jsonl"
    trace = EpisodeTrace(path, run_id="s", clock=clock)
    env = FakeEnv(clock)
    agent = ReceiptAgent(clock, replies, trace, confirms=confirms)
    with clock.participant():
        run_streaming_episode(
            agent, env, StreamingConfig(duration_s=duration_s), clocks=Clocks(clock, clock),
            log=lambda line: None, trace=trace,
        )
    trace.close()
    return summarize_streaming(path, settings, attempt_id=trace.attempt_id), env, agent


def test_the_summary_counts_frames_actions_and_replies(tmp_path):
    """Three clicks of one reply. The first executes from 2.275 s, once the read asked for it
    answered, to 4.275 s; the second waits, and the third takes its place and executes once the
    read after the first is done."""
    replies = [Reply(3.0, (Call(0.9, CLICK, "a"), Call(1.2, CLICK, "b"), Call(1.5, CLICK, "c")))]
    summary, env, agent = _summary(tmp_path, replies, OPENAI)
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
    # Every frame the session pushed was confirmed as its push returned, 0.125 s after it was
    # written. A push starts as its frame is captured, at 0.75 s for the first: the seconds from
    # capture to push count up to the write, and the confirmation's wait counts only from it.
    assert (frames["confirmed"], frames["refused"]) == (len(agent.pushed), 0)
    confirm = frames["send_to_confirm_s"]
    assert (confirm["sample_count"], confirm["min"], confirm["max"]) == (
        len(agent.pushed),
        0.125,
        0.125,
    )
    push = frames["capture_to_push_s"]
    assert (push["sample_count"], push["max"]) == (len(agent.pushed), 0.0)

    actions = summary["actions"]
    assert actions["arrived"] == 3
    assert actions["outcomes"] == {"applied": 2, "superseded": 1}
    # Each click named the newest frame pushed when it arrived: the first frame, pushed 0.9 s
    # before the first click, then the frame pushed at 1.875 s, 0.2 s and 0.5 s before the other
    # two.
    age = actions["frame_age_s"]
    assert (age["sample_count"], age["min"], age["max"]) == (3, 0.2, 0.9)
    assert actions["frame_age_frames"]["max"] == 0
    # The first click executed once the read asked for it answered, 0.5 s after it arrived; each
    # ran the fake's 2 s.
    assert actions["arrival_to_start_s"]["sample_count"] == 2
    assert actions["arrival_to_start_s"]["min"] == 0.5
    assert (actions["execution_s"]["min"], actions["execution_s"]["max"]) == (2.0, 2.0)

    replies = summary["replies"]
    assert replies["started"] == len(agent.started)
    assert replies["abandoned"] == {"deadline": 1}


def test_a_session_that_confirms_no_frame_summarizes_none(tmp_path):
    """Gemini Live's compression makes the context control approximate, and its session confirms
    no frame (providers/gemini/session.py Delivery)."""
    summary, _, agent = _summary(tmp_path, [], GEMINI, confirms=False, duration_s=4.0)
    assert summary["context_control"] == "approximate"
    frames = summary["frames"]
    assert frames["pushed"] == len(agent.pushed) > 0
    assert (frames["confirmed"], frames["send_to_confirm_s"]["sample_count"]) == (0, 0)
    assert frames["capture_to_push_s"]["sample_count"] == len(agent.pushed)
    assert summary["actions"]["arrived"] == 0
    assert summary["actions"]["outcomes"] == {}


def test_a_run_without_its_configuration_has_no_context_control(tmp_path):
    """A run that failed before its configuration was resolved, and its trace holds no episode."""
    path = tmp_path / "trace.jsonl"
    trace = EpisodeTrace(path, run_id="s", clock=FakeClock())
    trace.close()
    summary = summarize_streaming(path, None, attempt_id=trace.attempt_id)
    assert summary["context_control"] is None
    assert (summary["frames"]["captured"], summary["replies"]["started"]) == (0, 0)


def test_the_summary_pairs_each_frame_with_its_own_push(tmp_path):
    """Twelve frames, the eleventh's push refused: frames are matched by the number each went in
    under, not by the order they were pushed in, so the refusal counts the eleventh and a refusal
    of the second's placeholder as the session folded it counts nothing (low 4). The age of a
    frame is taken over the actions whose frame was found."""
    clock, sequence = FakeClock(), itertools.count(1)
    path = tmp_path / "trace.jsonl"
    trace = EpisodeTrace(path, run_id="s", clock=clock)

    def kernel(kind, **data):
        trace.emit("KernelRecord", {"kind": kind, "data": data})

    def refused(block):
        contents = [{"key": ["create", f"item-{block}"], "block": block}]
        capture(
            trace, clock, sequence, TraceEvent.SESSION_DELIVERY, state="refused",
            connection_id="conn-1", contents=contents,
        )  # fmt: skip

    for n in range(1, 13):
        observation_id = f"obs-{n}"
        kernel("FrameCaptured", observation_id=observation_id, timestamp=float(n), error=None)
        if n == 11:
            refused("f11.frame")
            kernel("FramePushed", observation_id=observation_id, sent_at=None, error="refused")
            kernel("FrameDropped", observation_id=observation_id, reason="push_failed", dropped=1)
            continue
        capture(
            trace, clock, sequence, TraceEvent.FRAME_SENT, frame=n, observation_id=observation_id,
            captured_unix_s=1000.0 + n, sent_unix_s=1000.1 + n, acked_unix_s=1000.3 + n,
        )  # fmt: skip
        kernel("FramePushed", observation_id=observation_id, sent_at=n + 0.3, error=None)
    refused("f2.frame")
    for age, rejection in ((0.5, None), (1.5, None), (None, "invalid_grounding_frame")):
        kernel("ActionArrived", frame_age_s=age, frame_age_frames=0, rejection=rejection)
    trace.close()

    summary = summarize_streaming(path, OPENAI, attempt_id=trace.attempt_id)
    frames = summary["frames"]
    assert (frames["captured"], frames["pushed"], frames["push_failed"]) == (12, 11, 1)
    assert (frames["confirmed"], frames["refused"]) == (11, 1)
    push, confirm = frames["capture_to_push_s"], frames["send_to_confirm_s"]
    assert (push["sample_count"], push["min"], push["max"]) == (11, 0.1, 0.1)
    assert (confirm["sample_count"], confirm["complete"], confirm["min"]) == (11, True, 0.2)
    age = summary["actions"]["frame_age_s"]
    assert (age["sample_count"], age["total_decisions"], age["complete"]) == (2, 2, True)


def test_a_push_the_end_s_abort_failed_is_no_failed_push(tmp_path):
    """The kernel drops a frame whose push failed as push_failed, and one whose push failed once
    the episode had ended, as the end's abort closing the session fails it, as ended
    (kernel/streaming.py _pushed): only the first counts as a failed push."""
    path = tmp_path / "trace.jsonl"
    trace = EpisodeTrace(path, run_id="s", clock=FakeClock())

    def kernel(kind, **data):
        trace.emit("KernelRecord", {"kind": kind, "data": data})

    for n, reason in ((1, "push_failed"), (2, "ended")):
        observation_id = f"obs-{n}"
        kernel("FrameCaptured", observation_id=observation_id, timestamp=float(n), error=None)
        kernel("FramePushed", observation_id=observation_id, sent_at=None, error="closed")
        kernel("FrameDropped", observation_id=observation_id, reason=reason, dropped=n)
    trace.close()

    frames = summarize_streaming(path, OPENAI, attempt_id=trace.attempt_id)["frames"]
    assert (frames["captured"], frames["pushed"], frames["push_failed"]) == (2, 0, 1)
    assert (frames["dropped"], frames["dropped_by_reason"]) == (2, {"ended": 1, "push_failed": 1})


def test_the_summary_reads_its_own_run_of_a_trace_written_twice(tmp_path):
    """A run directory used again appends a new attempt to its trace (reporting.py EpisodeTrace):
    a run's section counts only the records of its own attempt (low 1)."""
    path = tmp_path / "trace.jsonl"
    attempts = []
    for frames in (3, 2):
        trace = EpisodeTrace(path, run_id="s", clock=FakeClock())
        for n in range(frames):
            data = {"observation_id": f"obs-{n}", "timestamp": float(n), "error": None}
            trace.emit("KernelRecord", {"kind": "FrameCaptured", "data": data})
        trace.close()
        attempts.append(trace.attempt_id)
    counts = [
        summarize_streaming(path, OPENAI, attempt_id=attempt)["frames"]["captured"]
        for attempt in attempts
    ]
    assert counts == [3, 2]
