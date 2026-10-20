"""T2: request and reply bodies in episode_blobs.pack (evaluation/agent_trace.py).

Long base64 spans are stored verbatim once per hash, the rest of a body is its template, and every
body rebuilds byte for byte from the locator its event carries. No body content reaches the trace,
which is exported and uploaded.
"""

from __future__ import annotations

import base64
import json
import os

import pytest

from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.agent_trace import (
    SPAN_MIN_CHARS,
    AgentTraceSink,
    close_agent_trace,
    load_body,
)
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.providers.openai_realtime import classify_realtime_message

IMAGE = os.urandom(6000)
PRIVATE = "Enter the private account number 4417 and the caller's home address"


def _record(run_dir, bodies, *, session=()) -> tuple[list[dict], dict]:
    """Capture each body in one sink; return the request records and the close summary.

    ``session`` holds Realtime client messages, captured as a live session would capture them.
    """
    trace = EpisodeTrace(run_dir / "episode_trace.jsonl", run_id="pack-test")
    sink = AgentTraceSink(trace, run_dir / "episode_blobs.pack")
    for body in bodies:
        sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"channel": "test"}, body=body)
    for message in session:
        sink.capture(
            TraceEvent.SESSION_MESSAGE,
            {"channel": "test.ws", "connection_id": "test.ws-1", "direction": "client"},
            body=message,
            classify=classify_realtime_message,
        )
    assert close_agent_trace(sink, None) is None
    trace.close()
    lines = (run_dir / "episode_trace.jsonl").read_text().splitlines()
    events = [e for e in map(json.loads, lines) if e["attempt_id"] == trace.attempt_id]
    requests = [e["payload"] for e in events if e["event"] == "ModelRequestSent"]
    (closed,) = [e["payload"] for e in events if e["event"] == "AgentTraceClosed"]
    return requests, closed


BODIES = {
    # A chat data URL (harnesses/default/harness.py), standard alphabet, json.dumps separators.
    "data_url": json.dumps(
        {"image_url": {"url": "data:image/png;base64," + base64.b64encode(IMAGE).decode()}}
    ).encode(),
    # google-genai inlineData: URL-safe base64 after '"data": "'.
    "genai_inline_data": json.dumps(
        {
            "inline_data": {
                "data": base64.urlsafe_b64encode(IMAGE).decode(),
                "mime_type": "image/png",
            }
        }
    ).encode(),
    # An Anthropic image source as httpx encodes JSON: compact, '"data":"'.
    "compact_data": json.dumps(
        {
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(IMAGE).decode(),
            }
        },
        separators=(",", ":"),
    ).encode(),
}


@pytest.mark.parametrize("name", sorted(BODIES))
def test_bodies_rebuild_byte_exact(tmp_path, name):
    body = BODIES[name]
    (request,), _ = _record(tmp_path, [body])
    assert len(request["body"]["spans"]) == 1
    assert request["body"]["template"]["encoding"] == "zlib"
    assert load_body(tmp_path / "episode_blobs.pack", request["body"]) == body


def test_a_screenshot_is_stored_once_across_requests(tmp_path):
    bodies = [
        BODIES["data_url"].replace(b'{"image_url"', b'{"turn": %d, "image_url"' % k)
        for k in range(20)
    ]
    requests, closed = _record(tmp_path, bodies)
    spans = {json.dumps(span["ref"]) for r in requests for span in r["body"]["spans"]}
    assert len(spans) == 1
    assert closed["pack"]["records"] == len(spans) + len(bodies)  # one template per request
    for request, body in zip(requests, bodies):
        assert load_body(tmp_path / "episode_blobs.pack", request["body"]) == body


@pytest.mark.parametrize("length,spans", [(SPAN_MIN_CHARS - 1, 0), (SPAN_MIN_CHARS, 1)])
def test_span_threshold(tmp_path, length, spans):
    body = b'{"url":"data:image/png;base64,' + b"A" * length + b'"}'
    (request,), _ = _record(tmp_path, [body])
    assert len(request["body"].get("spans", [])) == spans
    assert load_body(tmp_path / "episode_blobs.pack", request["body"]) == body


def test_a_template_is_compressed_in_the_pack(tmp_path):
    body = json.dumps({"messages": [{"content": f"observation {k}"} for k in range(800)]}).encode()
    (request,), _ = _record(tmp_path, [body])
    assert set(request["body"]) == {"sha256", "bytes", "template"}
    assert request["body"]["template"]["encoding"] == "zlib"
    assert request["body"]["template"]["length"] < len(body)
    assert load_body(tmp_path / "episode_blobs.pack", request["body"]) == body


def test_no_body_content_reaches_the_trace(tmp_path):
    small = json.dumps({"messages": [{"role": "user", "content": PRIVATE}]}).encode()
    large = json.dumps({"messages": [{"role": "user", "content": PRIVATE * 100}]}).encode()
    message = json.dumps(
        {
            "type": "conversation.item.create",
            "event_id": "rtb_create_1",
            "item": {"id": "rtb_user_1", "content": [{"type": "input_text", "text": PRIVATE}]},
        }
    )
    requests, _ = _record(tmp_path, [small, large], session=[message])
    trace = (tmp_path / "episode_trace.jsonl").read_text()
    assert PRIVATE not in trace and "private account" not in trace
    (session,) = [
        json.loads(line)["payload"]
        for line in trace.splitlines()
        if json.loads(line)["event"] == "SessionMessage"
    ]
    assert session["message_type"] == "conversation.item.create"
    assert session["keys"] == ["event_id", "item", "type"]
    pack = tmp_path / "episode_blobs.pack"
    assert load_body(pack, requests[0]["body"]) == small
    assert load_body(pack, session["body"]) == message.encode()


def test_a_resumed_attempt_appends_to_the_pack(tmp_path):
    (first,), closed = _record(tmp_path, [BODIES["data_url"]])
    (second,), _ = _record(tmp_path, [BODIES["genai_inline_data"]])
    assert second["body"]["spans"][0]["ref"]["offset"] == closed["pack"]["bytes"]
    for request, name in ((first, "data_url"), (second, "genai_inline_data")):
        assert load_body(tmp_path / "episode_blobs.pack", request["body"]) == BODIES[name]
