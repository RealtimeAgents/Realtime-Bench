"""An episode trace's live session as a cassette (ws_cassette.from_trace), replayed on the fake
clock.

The OpenAI Realtime driver runs two turns in real time against a WebSocket stub
(tests/agent_trace/stubs.py), with an agent trace sink as the runner gives it. The cassette
converted from that trace then serves the same two turns to the same driver on the fake clock,
through clocked_connection, and the strict cursor finds no difference.
"""

from __future__ import annotations

import json
import time

import pytest

from realtime_framework.clock import SystemClock
from realtime_framework.entity.records import UNTRACED, TraceEvent
from realtime_framework.evaluation.agent_trace import AgentTraceSink, close_agent_trace
from realtime_framework.evaluation.reporting import EpisodeTrace
from tests.agent_trace.stubs import WsStub
from tests.fake_clock import FakeClock
from tests.replay import baseline

from . import ws_cassette
from .clocked_ws import CassetteServer, clocked_connection
from .openai_session import driver_for, run_on_fake_clock, run_turns

REPLY_TEXTS = ('{"action": "wait"}', '{"action": "click", "x": 1, "y": 2}')


def _realtime(stub, connection, message):
    """Answer like the Realtime API, one scripted reply per response.create."""
    if message is None:
        return [{"type": "session.created", "event_id": "srv_created", "session": {}}]
    kind = message["type"]
    if kind == "session.update":
        return [{"type": "session.updated", "event_id": "srv_updated", "session": {}}]
    if kind == "conversation.item.create":
        item = {"id": message["item"]["id"]}
        return [
            {"type": "conversation.item.added", "item": item},
            {"type": "conversation.item.done", "item": item},
        ]
    n = sum(json.loads(m)["type"] == "response.create" for m in connection.received)
    response_id, item_id = f"resp_{n}", f"srv_item_{n}"
    text = REPLY_TEXTS[n - 1]
    return [
        {"type": "response.created", "response": {"id": response_id}},
        {"type": "response.output_item.added", "response_id": response_id, "item": {"id": item_id}},
        {"type": "response.output_text.delta", "response_id": response_id, "delta": text},
        {
            "type": "response.done",
            "response": {
                "id": response_id,
                "status": "completed",
                "output": [{"id": item_id}],
                "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
            },
        },
    ]


def _record_session(run_dir):
    """Two turns in real time against the stub, traced; the stub's one connection and the text
    deltas the driver read."""
    trace = EpisodeTrace(run_dir / "episode_trace.jsonl", run_id="cassette")
    # The runner records the episode origin before the agent's first frame (runner.py:1410-1419).
    trace.emit("ActiveLoopStarted", {"start_unix_s": time.time(), "clock_mode": "wall"})
    sink = AgentTraceSink(trace, run_dir / "episode_blobs.pack")
    with WsStub(_realtime) as stub:
        clock = SystemClock()
        loop = clock.new_event_loop()
        try:
            deltas = loop.run_until_complete(run_turns(driver_for(stub.port, clock, sink), 2))
        finally:
            loop.close()
    assert close_agent_trace(sink, None) is None
    trace.close()
    (connection,) = stub.connections
    return connection, deltas


def test_a_traced_session_becomes_a_cassette_that_replays_on_the_fake_clock(tmp_path):
    connection, recorded = _record_session(tmp_path)
    records = ws_cassette.from_trace(tmp_path)

    # One connection: its open, then every message the driver sent and each one it read, in the
    # order they were captured. The acknowledgements carry the client's ids, as placeholders.
    # Only the server's messages keep their times: the client's are compared in order only.
    assert records[0]["dir"] == "open"
    assert {record["conn"] for record in records} == {1}
    assert all((r["at"] is None) == (r["dir"] != "s2c") for r in records)
    normalize = baseline.Normalizer()
    sent = [r["msg"] for r in records if r["dir"] == "c2s"]
    read = [r["msg"] for r in records if r["dir"] == "s2c"]
    assert sent == [normalize(json.loads(message)) for message in connection.received]
    assert read == [normalize(json.loads(message)) for message in connection.sent[: len(read)]]
    assert read[2]["item"]["id"] == sent[1]["item"]["id"] == "<rtb_user-1>"
    assert [m["type"] for m in sent] == [
        "session.update",
        "conversation.item.create",
        "response.create",
        "conversation.item.create",
        "response.create",
    ]
    assert read[-1]["type"] == "response.done"
    times = [r["at"] for r in records if r["dir"] == "s2c"]
    assert 0 <= times[0] and times[-1] < 30.0

    clock = FakeClock()
    with CassetteServer(records, clock) as server:
        driver = driver_for(server.port, clock, UNTRACED)
        connection = driver._ws_connection = clocked_connection(driver._ws_connection, clock)
        replayed = run_on_fake_clock(clock, run_turns(driver, 2))
    assert connection.opened == 1
    assert server.finish() == []
    assert server.paths == ["/v1/realtime?model=gpt-realtime-2"]
    assert [text for text, _ in replayed] == [text for text, _ in recorded] == list(REPLY_TEXTS)
    # Each reply arrives when the clock reads the time the recording read it.
    deltas = [r["at"] for r in records if r["dir"] == "s2c" and r["msg"]["type"].endswith("delta")]
    assert [at for _, at in replayed] == pytest.approx(deltas, abs=1e-6)


def test_a_trace_off_the_wall_clock_is_refused(tmp_path):
    trace = EpisodeTrace(tmp_path / "episode_trace.jsonl", run_id="cassette")
    trace.emit("ActiveLoopStarted", {"start_unix_s": 0.0, "clock_mode": "token"})
    trace.close()
    with pytest.raises(ValueError, match="wall clock"):
        ws_cassette.from_trace(tmp_path)


def test_a_failed_send_has_no_cassette_form_yet(tmp_path):
    trace = EpisodeTrace(tmp_path / "episode_trace.jsonl", run_id="cassette")
    trace.emit("ActiveLoopStarted", {"start_unix_s": time.time(), "clock_mode": "wall"})
    sink = AgentTraceSink(trace, tmp_path / "episode_blobs.pack")
    # As providers/capture.py:318-325 records a write that raised.
    sink.capture(
        TraceEvent.SESSION_MESSAGE,
        {
            "channel": "openai_realtime.ws",
            "connection_id": "openai_realtime.ws-1",
            "direction": "lifecycle",
            "message_type": "send_failed",
            "state": "unknown",
        },
    )
    assert close_agent_trace(sink, None) is None
    trace.close()
    with pytest.raises(ValueError, match="send_failed"):
        ws_cassette.from_trace(tmp_path)
