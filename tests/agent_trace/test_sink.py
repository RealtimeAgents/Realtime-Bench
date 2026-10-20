"""T1: the agent trace sink (evaluation/agent_trace.py).

A capture only queues references; one writer thread emits them through EpisodeTrace, interleaved
with the runner's own records in one chain, and close_agent_trace finishes them before the runner
writes EpisodeFinished.
"""

from __future__ import annotations

import hashlib
import json
import threading
import types

from realtime_framework.entity.records import TraceEvent, validate_episode_trace
from realtime_framework.evaluation import agent_trace
from realtime_framework.evaluation.agent_trace import AgentTraceSink, close_agent_trace
from realtime_framework.evaluation.reporting import EpisodeTrace, _episode_trace_ok

SCREENSHOT = b"iVBORw0KGgoAAAANSUhEUgAA" * 400  # long enough to be stored as a span
BODY = b'{"image_url":{"url":"data:image/png;base64,' + SCREENSHOT + b'"}}'


def _open(tmp_path):
    trace = EpisodeTrace(tmp_path / "episode_trace.jsonl", run_id="sink-test")
    return trace, AgentTraceSink(trace, tmp_path / "episode_blobs.pack")


def _events(tmp_path) -> list[dict]:
    lines = (tmp_path / "episode_trace.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _block_writer(sink):
    """Hold the writer before it stores a body, outside the lock it emits under."""
    entered, release = threading.Event(), threading.Event()
    store = sink._store

    def blocked(data):
        entered.set()
        release.wait()
        return store(data)

    sink._store = blocked
    return entered, release


class _RecordingOs:
    """The os module, recording which threads write."""

    def __init__(self, real, threads: set):
        self._real = real
        self._threads = threads

    def __getattr__(self, name):
        return getattr(self._real, name)

    def write(self, fd, data):
        self._threads.add(threading.get_ident())
        return self._real.write(fd, data)


def test_capture_does_no_io_or_hashing_on_the_caller_thread(tmp_path, monkeypatch):
    seen = {"hash": set(), "write": set(), "emit": set()}

    def sha256(data=b""):
        seen["hash"].add(threading.get_ident())
        return hashlib.sha256(data)

    monkeypatch.setattr(agent_trace, "hashlib", types.SimpleNamespace(sha256=sha256))
    monkeypatch.setattr(agent_trace, "os", _RecordingOs(agent_trace.os, seen["write"]))
    trace, sink = _open(tmp_path)
    emit = trace.emit

    def recorded_emit(event, payload=None, **fields):
        if event != TraceEvent.AGENT_TRACE_CLOSED:
            seen["emit"].add(threading.get_ident())
        return emit(event, payload, **fields)

    trace.emit = recorded_emit
    for index in range(5):
        sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"index": index}, body=BODY)
    assert close_agent_trace(sink, None) is None
    caller = threading.get_ident()
    assert seen["hash"] and seen["write"] and seen["emit"]
    assert caller not in seen["hash"] | seen["write"] | seen["emit"]


def test_concurrent_captures_and_runner_emits_form_one_chain(tmp_path):
    trace, sink = _open(tmp_path)
    trace.emit(TraceEvent.EPISODE_STARTED, {})
    producers, per_producer = 8, 200

    def produce(producer):
        for index in range(per_producer):
            sink.capture(
                TraceEvent.HISTORY_CHANGED,
                {"producer": producer, "index": index},
                body=f"{producer}:{index}",
            )

    threads = [threading.Thread(target=produce, args=(n,)) for n in range(producers)]
    for thread in threads:
        thread.start()
    for index in range(per_producer):
        trace.emit(TraceEvent.OBSERVATION_CAPTURED, {"decision": index})
    for thread in threads:
        thread.join()
    assert close_agent_trace(sink, None) is None
    trace.emit(TraceEvent.EPISODE_FINISHED, {})
    trace.close()

    verified = validate_episode_trace(tmp_path / "episode_trace.jsonl")
    events = _events(tmp_path)
    agent = [e["payload"] for e in events if e["event"] == "HistoryChanged"]
    assert verified["event_count"] == 1 + per_producer + producers * per_producer + 2
    assert [record["capture_seq"] for record in agent] == list(range(1, len(agent) + 1))
    for producer in range(producers):
        mine = [record["index"] for record in agent if record["producer"] == producer]
        assert mine == list(range(per_producer))


def test_close_drops_late_captures_and_precedes_episode_finished(tmp_path):
    trace, sink = _open(tmp_path)
    trace.emit(TraceEvent.EPISODE_STARTED, {})
    entered, release = _block_writer(sink)
    sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"channel": "test"}, body=BODY)
    entered.wait()
    trace.emit(TraceEvent.SNAPSHOT_FROZEN, {})
    trace.emit(TraceEvent.SCORE_PRODUCED, {})
    closed = []
    closing = threading.Thread(target=lambda: closed.append(close_agent_trace(sink, None)))
    closing.start()
    while not sink._sealed:
        threading.Event().wait(0.01)
    assert sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"channel": "late"}) is None
    release.set()
    closing.join()
    assert closed == [None]
    assert sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"channel": "later"}) is None

    trace.emit(TraceEvent.EPISODE_FINISHED, {})
    summary = {"episode_trace": trace.describe()}
    trace.close()
    summary_path = tmp_path / "run_summary.json"
    summary_path.write_text(json.dumps(summary))
    assert _episode_trace_ok(summary_path, summary)
    events = _events(tmp_path)
    assert [e["event"] for e in events][-2:] == ["AgentTraceClosed", "EpisodeFinished"]
    (close,) = [e["payload"] for e in events if e["event"] == "AgentTraceClosed"]
    assert close["captures"] == close["written"] == 1
    assert close["dropped_after_seal"] == 1
    assert close["error"] is None and close["pending_at_timeout"] is None
    assert close["pack"]["fsynced"]


def test_writer_error_invalidates_the_run(tmp_path):
    _, sink = _open(tmp_path)

    def fail(data):
        raise OSError("disk full")

    sink._store = fail
    sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"channel": "test"}, body=b"{}")
    sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"channel": "test"}, body=b"{}")
    invalid_run = close_agent_trace(sink, None)
    assert invalid_run["category"] == "harness"
    assert invalid_run["reason"] == "agent_trace_write_failed"
    assert "disk full" in invalid_run["detail"]
    (close,) = [e["payload"] for e in _events(tmp_path) if e["event"] == "AgentTraceClosed"]
    assert close["written"] == 0 and close["captures"] == 2 and "disk full" in close["error"]

    earlier = {"category": "environment", "reason": "evaluation_snapshot_failed"}
    _, other = _open(tmp_path / "earlier")
    other._store = fail
    other.capture(TraceEvent.MODEL_REQUEST_SENT, {"channel": "test"}, body=b"{}")
    assert close_agent_trace(other, earlier) is earlier


def test_a_long_write_error_keeps_its_exception_line(tmp_path):
    """The invalid-run detail is the end of the writer's traceback, which names the exception."""
    _, sink = _open(tmp_path)

    def fail(data):
        try:
            raise ValueError("cannot encode " + "x" * 3000)
        except ValueError as exc:
            raise OSError("disk full") from exc

    sink._store = fail
    sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"channel": "test"}, body=b"{}")
    invalid_run = close_agent_trace(sink, None)
    (close,) = [e["payload"] for e in _events(tmp_path) if e["event"] == "AgentTraceClosed"]
    assert len(close["error"]) > 2000
    assert invalid_run["detail"] == close["error"][-2000:]
    assert invalid_run["detail"].endswith("OSError: disk full\n")


def test_blocked_writer_times_out_and_never_emits_after_the_fence(tmp_path):
    trace, sink = _open(tmp_path)
    entered, release = _block_writer(sink)
    sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"channel": "test"}, body=b"a")
    sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"channel": "test"}, body=b"b")
    entered.wait()
    # The writer waits until released, so the bound is reached however fast the machine is.
    invalid_run = close_agent_trace(sink, None, timeout=0.2)
    assert invalid_run["reason"] == "agent_trace_flush_timeout"
    trace.emit(TraceEvent.EPISODE_FINISHED, {})
    release.set()
    sink._writer.join(10)
    assert not sink._writer.is_alive()
    trace.close()
    events = _events(tmp_path)
    assert [e["event"] for e in events] == ["AgentTraceClosed", "EpisodeFinished"]
    assert events[0]["payload"]["pending_at_timeout"] == 2
