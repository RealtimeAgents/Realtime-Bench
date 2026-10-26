"""The agent's records through the kernel (evaluation/shell.py ``agent_trace``): during the episode
its AgentTraceSink hands each to the shell, the shell gives the kernel one a step, the kernel
numbers it into the episode log and the shell writes it; once the episode is finished the sink
writes them itself again."""

from __future__ import annotations

import time
from typing import ClassVar

import pytest

from realtime_framework.entity.records import TraceEvent, trace_scope
from realtime_framework.evaluation.agent_trace import close_agent_trace
from realtime_framework.evaluation.kernel.core import Kernel
from realtime_framework.evaluation.kernel.events import AgentRecord

from .world import CLICK, Agent, Reply, Trace, World, run


def _handed(sink, count: int) -> None:
    """Wait, in real time, until the sink's writer has passed on ``count`` records. The fake clock
    stands still meanwhile: the waiting thread takes part in it and is busy."""
    while sink._written < count:
        time.sleep(0.001)


class RecordingAgent(Agent):
    """Records one request for each generation on its worker, in the generation's scope, and
    returns only once the sink has handed it over."""

    def __init__(self, clock, replies):
        super().__init__(clock, replies)
        self.captured = 0

    def generate(self, generation, decision, observation, budget):
        with trace_scope(phase="decision", decision=decision, generation=generation):
            self.sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"decision": decision})
        self.captured += 1
        _handed(self.sink, self.captured)
        return super().generate(generation, decision, observation, budget)


class BurstAgent(Agent):
    """Records a burst of session messages on the shell's thread as it is told an outcome,
    without waiting for the sink."""

    made: ClassVar[list[BurstAgent]] = []

    def __init__(self, clock, replies):
        super().__init__(clock, replies)
        BurstAgent.made.append(self)

    def commit(self, outcome) -> None:
        super().commit(outcome)
        for index in range(200):
            self.sink.capture(TraceEvent.SESSION_STATE_CHANGED, {"index": index})


class AbortRecordingAgent(RecordingAgent):
    """RecordingAgent that also records each abort, on the shell's thread, and returns only once
    the sink has passed it on."""

    def abort(self, generation: int | None = None) -> None:
        super().abort(generation)
        with trace_scope(phase="abort", generation=generation):
            self.sink.capture(TraceEvent.SESSION_STATE_CHANGED, {"aborted": generation})
        self.captured += 1
        _handed(self.sink, self.captured)


class SlowTrace(Trace):
    """A trace whose write of an agent record takes 2 ms of fake time, as a long session log's
    writes take time in the world."""

    clock: ClassVar = None

    def emit(self, event, payload=None, **fields) -> None:
        super().emit(event, payload, **fields)
        if payload is not None and payload.get("source") == "agent":
            self.clock.sleep(0.002)


class BacklogAgent(Agent):
    """Its second generation records 500 session messages at 2.1 s, while the shell is busy, and
    returns only once the sink has handed them all over."""

    def __init__(self, clock, replies):
        super().__init__(clock, replies)
        SlowTrace.clock = clock

    def generate(self, generation, decision, observation, budget):
        if decision == 1:
            self.clock.sleep(1.1)
            for index in range(500):
                self.sink.capture(TraceEvent.SESSION_STATE_CHANGED, {"index": index})
            _handed(self.sink, 500)
        return super().generate(generation, decision, observation, budget)


def _agent_records(trace) -> list[dict]:
    return [record for record in trace.records if record["payload"].get("source") == "agent"]


def test_the_agent_s_records_are_numbered_by_the_kernel_and_written_by_the_shell(tmp_path):
    """Each generation's request record joins the episode log where the kernel took it, between
    the generation's start and its result, with the agent's own capture number and scope."""
    episode = run(
        "wall",
        2.0,
        [Reply(0.5), Reply(0.5)],
        agent_type=RecordingAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    records = episode.trace.records
    requests = [record for record in records if record["event"] == "ModelRequestSent"]
    assert [record["payload"]["decision"] for record in requests] == [0, 1]
    assert [record["payload"]["capture_seq"] for record in requests] == [1, 2]
    assert [record["payload"]["scope"]["generation"] for record in requests] == [1, 2]
    assert [(record["semantic"], record["ops"]) for record in requests] == [(0.0, 0.0), (1.0, 1.0)]
    # The trace is in the kernel's order, the agent's records with the rest.
    numbers = [record["kernel_seq"] for record in records]
    assert numbers == sorted(numbers)
    events = [record["event"] for record in records]
    started = [i for i, event in enumerate(events) if event == "InferenceStarted"]
    completed = [i for i, event in enumerate(events) if event == "InferenceCompleted"]
    at = [records.index(record) for record in requests]
    assert all(s < a < c for s, a, c in zip(started, at, completed, strict=True))


def test_once_the_episode_is_finished_the_sink_writes_its_records_itself(tmp_path):
    episode = run(
        "wall",
        1.0,
        [Reply(0.5)],
        agent_type=RecordingAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    sink = episode.agent.sink
    with trace_scope(phase="close"):
        sink.capture(TraceEvent.SESSION_STATE_CHANGED, {"state": "closed"})
    summary = sink.finish(5.0)
    last = episode.trace.records[-1]
    assert (last["event"], last["payload"]["state"]) == ("SessionStateChanged", "closed")
    assert "kernel_seq" not in last
    assert (summary["captures"], summary["written"], summary["error"]) == (2, 2, None)


def test_records_handed_over_as_the_episode_finishes_are_written_before_it_ends(tmp_path):
    """The task ends at 0.5 s and decision 0's outcome, told after the snapshot, records a burst.
    The shell releases the inputs after the termination record and before it finishes; by then
    the sink has handed the whole burst to the kernel, which has not taken it yet. The shell
    writes it, numbered after the termination, before the sink writes again."""
    BurstAgent.made.clear()
    world = World(ends_at=0.5, on_release=lambda: _handed(BurstAgent.made[0].sink, 200))
    episode = run(
        "wall",
        1.0,
        [Reply(0.5)],
        world,
        agent_type=BurstAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    sink = episode.agent.sink
    sink.capture(TraceEvent.SESSION_STATE_CHANGED, {"index": 200})
    summary = sink.finish(5.0)
    assert (summary["captures"], summary["written"], summary["error"]) == (201, 201, None)
    records = _agent_records(episode.trace)
    assert [record["payload"]["index"] for record in records] == list(range(201))
    assert ["kernel_seq" in record for record in records] == [True] * 200 + [False]
    (termination,) = [
        record["kernel_seq"]
        for record in episode.trace.records
        if record["event"] == "KernelRecord" and record["payload"]["kind"] == "Termination"
    ]
    assert all(record["kernel_seq"] > termination for record in records[:200])
    numbers = [record["kernel_seq"] for record in episode.trace.records if "kernel_seq" in record]
    assert numbers == sorted(numbers)


def test_a_kernel_that_raises_leaves_the_agent_s_records_to_the_sink(tmp_path, monkeypatch):
    """N11: the kernel raises at the end check at 2 s, during the second call, which never
    answers. The sink takes the records back before the shell wraps the episode up without the
    kernel, so the abort that interrupts the call is written by the sink itself rather than handed
    to a kernel that no longer steps; the requests the kernel took before stay numbered."""
    step = Kernel.step

    def broken(kernel, semantic, ops=None):
        if semantic >= 2.0:
            raise RuntimeError("step broke")
        return step(kernel, semantic, ops)

    monkeypatch.setattr(Kernel, "step", broken)
    episode = run(
        "wall",
        3.0,
        [Reply(0.5), Reply(hang=True)],
        agent_type=AbortRecordingAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    assert episode.invalid["reason"] == "run_exception"
    summary = episode.agent.sink.finish(5.0)
    assert (summary["captures"], summary["written"], summary["error"]) == (3, 3, None)
    records = _agent_records(episode.trace)
    assert [record["event"] for record in records] == [
        "ModelRequestSent",
        "ModelRequestSent",
        "SessionStateChanged",
    ]
    assert ["kernel_seq" in record for record in records] == [True, True, False]
    assert records[2]["payload"]["aborted"] == 2


def test_a_backlog_of_agent_records_does_not_hold_back_a_result_that_came_back(tmp_path):
    """The end check at 2.0 s takes until 2.5 s. Meanwhile the second call records 500 session
    messages, each 2 ms to write, and comes back at 2.2 s. The kernel takes the result at 2.5 s,
    before the records, though they came first. It is given one record a step: the four steps
    that land the action each write one first, so it lands at 2.508 s, however many more the
    agent logged."""
    world = World(slow_checks=[(2.0, 2.1, 0.5)])
    episode = run(
        "wall",
        3.0,
        [Reply(0.5), Reply(0.1)],
        world,
        trace=SlowTrace(),
        agent_type=BacklogAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    assert episode.statuses() == ["applied", "applied"]
    completed = [r for r in episode.trace.records if r["event"] == "InferenceCompleted"]
    assert completed[1]["semantic"] == 2.5
    assert completed[1]["kernel_seq"] < _agent_records(episode.trace)[0]["kernel_seq"]
    resolved = [record for record in episode.trace.records if record["event"] == "ActionResolved"]
    assert resolved[1]["semantic"] == pytest.approx(2.508)
    summary = episode.agent.sink.finish(5.0)
    assert (summary["captures"], summary["written"], summary["error"]) == (500, 500, None)
    assert len(_agent_records(episode.trace)) == 500
    numbers = [record["kernel_seq"] for record in episode.trace.records if "kernel_seq" in record]
    assert numbers == sorted(numbers)


class HeldRecordsAgent(Agent):
    """Its second generation records three session messages at 2.1 s, while the shell is busy, and
    returns only once the sink has handed them all over."""

    def generate(self, generation, decision, observation, budget):
        if decision == 1:
            self.clock.sleep(1.1)
            for index in range(3):
                self.sink.capture(TraceEvent.SESSION_STATE_CHANGED, {"index": index})
            _handed(self.sink, 3)
        return super().generate(generation, decision, observation, budget)


def test_records_the_kernel_never_took_are_written_when_it_raises(tmp_path, monkeypatch):
    """The end check at 2.0 s takes until 2.5 s; meanwhile the agent records three session
    messages. The kernel raises in its next step, before it took them, and the shell wraps the
    episode up without it (N11): the three are still written, unnumbered, as the sink writes
    once it has them back."""
    step = Kernel.step

    def broken(kernel, semantic, ops=None):
        if semantic >= 2.5:
            raise RuntimeError("step broke")
        return step(kernel, semantic, ops)

    monkeypatch.setattr(Kernel, "step", broken)
    episode = run(
        "wall",
        3.0,
        [Reply(0.5), Reply(0.1)],
        World(slow_checks=[(2.0, 2.1, 0.5)]),
        agent_type=HeldRecordsAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    assert episode.invalid["reason"] == "run_exception"
    summary = episode.agent.sink.finish(5.0)
    assert (summary["captures"], summary["written"], summary["error"]) == (3, 3, None)
    records = _agent_records(episode.trace)
    assert [record["payload"]["index"] for record in records] == [0, 1, 2]
    assert ["kernel_seq" in record for record in records] == [False] * 3


class SameNameAgent(Agent):
    """Records, on its worker, a record under a name the kernel's records use too."""

    def generate(self, generation, decision, observation, budget):
        with trace_scope(phase="decision", decision=decision, generation=generation):
            self.sink.capture(TraceEvent.ACTION_PROPOSED, {"note": "the agent's own"})
        _handed(self.sink, decision + 1)
        return super().generate(generation, decision, observation, budget)


def test_an_agent_record_is_written_as_the_agent_s_whatever_its_name(tmp_path):
    """An agent record under a name the kernel's records also use is the agent's: it is written
    as the sink gave it, not read as the kernel's ActionProposed."""
    episode = run(
        "wall",
        2.0,
        [Reply(0.5), Reply(0.5)],
        agent_type=SameNameAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    assert episode.invalid is None
    assert episode.statuses() == ["applied", "applied"]
    notes = [record["payload"].get("note") for record in _agent_records(episode.trace)]
    assert notes == ["the agent's own"] * 2


class AcceptRecordingAgent(Agent):
    """Records a history change on the shell's thread as it accepts each result, as DefaultHarness
    writes its history then, and returns only once the sink has handed it over."""

    def __init__(self, clock, replies):
        super().__init__(clock, replies)
        self.captured = 0

    def accept(self, generation: int) -> None:
        super().accept(generation)
        with trace_scope(phase="accept", generation=generation):
            self.sink.capture(TraceEvent.SESSION_STATE_CHANGED, {"accepted": generation})
        self.captured += 1
        _handed(self.sink, self.captured)


def _seqs(trace, event: str) -> list[int]:
    return [record["kernel_seq"] for record in trace.records if record["event"] == event]


def _termination(trace) -> int:
    (seq,) = [
        record["kernel_seq"]
        for record in trace.records
        if record["event"] == "KernelRecord" and record["payload"]["kind"] == "Termination"
    ]
    return seq


def test_agent_records_are_numbered_while_every_read_takes_a_frame(tmp_path):
    """Each end check takes 1 s, a whole frame, so a timer or an answer is due at every step and
    the kernel is never idle. Each request record is still numbered after its generation starts
    and before the next one does, and none after the termination."""
    episode = run(
        "wall",
        10.0,
        [Reply(0.5)] * 6,
        World(check_s=1.0),
        agent_type=RecordingAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    starts = _seqs(episode.trace, "InferenceStarted")
    requests = _seqs(episode.trace, "ModelRequestSent")
    assert len(requests) == len(starts) >= 2
    assert all(start < request for start, request in zip(starts, requests))
    assert all(request < start for request, start in zip(requests, starts[1:]))
    assert max(requests) < _termination(episode.trace)


def test_a_record_made_at_an_accept_is_numbered_before_the_next_generation_starts(tmp_path):
    """Token clock, B = 100: each reply's 2000 tokens move the world 20 frames, every step of
    which the kernel paces, reads and has answered at once, so it is idle only while a generation
    runs. The history change the agent records as it accepts each result is still numbered before
    the next generation starts."""
    episode = run(
        "token",
        60.0,
        [Reply(0.5, CLICK, output_tokens=2000)] * 4,
        tokens_per_frame=100,
        agent_type=AcceptRecordingAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    starts = _seqs(episode.trace, "InferenceStarted")
    accepts = [r["kernel_seq"] for r in _agent_records(episode.trace)]
    assert len(accepts) >= 2
    assert all(accept < start for accept, start in zip(accepts, starts[1:]))
    assert max(accepts) < _termination(episode.trace)


def test_a_backlog_handed_while_the_shell_is_idle_holds_back_no_result(tmp_path):
    """As test_a_backlog_of_agent_records_does_not_hold_back_a_result_that_came_back, but the
    shell is idle when the 500 records, 2 ms each to write, arrive at 2.1 s. The second result is
    back at 2.2 s, while they are being written: it lands then, not after the rest of them."""
    episode = run(
        "wall",
        3.0,
        [Reply(0.5), Reply(0.1)],
        trace=SlowTrace(),
        agent_type=BacklogAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    assert episode.statuses() == ["applied", "applied"]
    resolved = [record for record in episode.trace.records if record["event"] == "ActionResolved"]
    assert resolved[1]["semantic"] < 2.3


def test_the_record_the_kernel_was_given_is_written_when_it_raises_taking_it(tmp_path, monkeypatch):
    """The kernel raises in the step that would take an agent record, the first of three the
    agent made at 2.1 s (N11). That one and the two still held are written, unnumbered."""
    step = Kernel.step

    def broken(kernel, semantic, ops=None):
        if any(isinstance(event, AgentRecord) for event in kernel.inbox._events):
            raise RuntimeError("step broke on an agent record")
        return step(kernel, semantic, ops)

    monkeypatch.setattr(Kernel, "step", broken)
    episode = run(
        "wall",
        3.0,
        [Reply(0.5), Reply(0.5)],
        agent_type=HeldRecordsAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    assert episode.invalid["reason"] == "run_exception"
    summary = episode.agent.sink.finish(5.0)
    assert (summary["captures"], summary["written"], summary["error"]) == (3, 3, None)
    records = _agent_records(episode.trace)
    assert [record["payload"]["index"] for record in records] == [0, 1, 2]


class FailingOnceAfterTheEnd(Trace):
    """Fails to write the first agent record numbered after the termination, once."""

    def __init__(self):
        super().__init__()
        self._ended = self._failed = False

    def emit(self, event, payload=None, **fields) -> None:
        name = event.value if isinstance(event, TraceEvent) else event
        if name == "KernelRecord" and payload["kind"] == "Termination":
            self._ended = True
        agent = payload is not None and payload.get("source") == "agent"
        if self._ended and agent and "kernel_seq" in fields and not self._failed:
            self._failed = True
            raise OSError("trace disk full")
        super().emit(event, payload, **fields)


def test_a_kernel_that_raises_numbering_the_last_records_leaves_the_finished_episode_as_it_was(
    tmp_path, monkeypatch
):
    """As test_records_handed_over_as_the_episode_finishes_are_written_before_it_ends, but the
    kernel raises as the shell steps it through the burst after its Finish. The episode is over
    and its verdict stands: it is not wrapped up again, and the burst is written unnumbered."""
    original = Kernel.step

    def broken(kernel, semantic, ops=None):
        if kernel.termination is not None:
            raise RuntimeError("step broke")
        return original(kernel, semantic, ops)

    monkeypatch.setattr(Kernel, "step", broken)
    BurstAgent.made.clear()
    world = World(ends_at=0.5, on_release=lambda: _handed(BurstAgent.made[0].sink, 200))
    episode = run(
        "wall",
        1.0,
        [Reply(0.5)],
        world,
        agent_type=BurstAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    assert episode.invalid is None
    assert episode.result.stop_reason == "game_decided"
    assert [call for call, _, _ in episode.env.calls].count("release_inputs") == 1
    assert episode.result.loop_error is None
    records = _agent_records(episode.trace)
    assert [record["payload"]["index"] for record in records] == list(range(200))
    assert [record["kernel_seq"] for record in records] == [None] * 200
    assert any("numbering the agent's last records failed" in line for line in episode.logs)
    assert close_agent_trace(episode.agent.sink) is None


def test_a_record_that_fails_to_be_written_after_the_end_invalidates_the_run(tmp_path):
    """As test_records_handed_over_as_the_episode_finishes_are_written_before_it_ends, but the
    first of the burst written after the termination fails. The kernel handles nothing more, so
    the sink invalidates the run as it closes."""
    BurstAgent.made.clear()
    world = World(ends_at=0.5, on_release=lambda: _handed(BurstAgent.made[0].sink, 200))
    episode = run(
        "wall",
        1.0,
        [Reply(0.5)],
        world,
        trace=FailingOnceAfterTheEnd(),
        agent_type=BurstAgent,
        agent_trace_pack=tmp_path / "episode_blobs.pack",
    )
    assert episode.invalid is None
    invalid = close_agent_trace(episode.agent.sink)
    assert (invalid["category"], invalid["reason"]) == ("harness", "agent_trace_write_failed")
    assert "OSError: trace disk full" in invalid["detail"]
