"""The agent's records through the kernel (evaluation/shell.py ``agent_trace``): during the episode
its AgentTraceSink hands each to the shell, the kernel numbers it into the episode log once nothing
else is due, and the shell writes it; once the episode is finished the sink writes them itself
again."""

from __future__ import annotations

import time
from typing import ClassVar

from realtime_framework.entity.records import TraceEvent, trace_scope
from realtime_framework.evaluation.kernel.core import Kernel

from .world import Agent, Reply, Trace, World, run


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
    messages, each 2 ms to write, and comes back at 2.2 s. The kernel takes the result at 2.5 s
    and its action lands then; the records are written after it, though they came first. How
    much the agent logs does not decide whether or when its action lands."""
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
    resolved = [record for record in episode.trace.records if record["event"] == "ActionResolved"]
    assert resolved[1]["semantic"] == 2.5
    summary = episode.agent.sink.finish(5.0)
    assert (summary["captures"], summary["written"], summary["error"]) == (500, 500, None)
    assert len(_agent_records(episode.trace)) == 500
    numbers = [record["kernel_seq"] for record in episode.trace.records if "kernel_seq" in record]
    assert numbers == sorted(numbers)
