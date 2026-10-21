"""Native-streaming episodes end to end over replayed live sessions (tests/streaming/e2e.py).

Each cassette scripts the same episode. The environment captures a frame at 0, 1 and 2 s, and
the agent pushes each as it comes. Reply 1 starts on the first frame at 0 s; its click, based on
frame 1, comes off the wire at 0.4 s and executes at once, and the reply ends at 0.6 s. Reply 2
starts then and never ends: the deadline at 3 s gives it up, and the snapshot takes 0.5 s.
"""

from __future__ import annotations

import hashlib
import json

from realtime_framework.evaluation.kernel.events import Termination
from realtime_framework.evaluation.reporting import summarize_streaming
from tests.replay import baseline
from tests.replay.frames import screenshot
from tests.replay.ws_gemini_live import WireNormalizer
from tests.streaming.e2e import (
    LIVE_PATH,
    SILENCE,
    ScriptedLiveAgent,
    ScriptedRealtimeAgent,
    run_e2e,
)

FRAMES = [f"replay-form:{n:06d}" for n in (1, 2, 3)]  # tests/replay/frames.py ids, captures 0-2
CLICK = {"x": 1, "y": 2, "frame": 1}


def _record(direction, msg=None, at=None):
    return {"conn": 1, "dir": direction, "at": at, "msg": msg}


def _image(n: int) -> str:
    return baseline.image_digest(screenshot(n))


def _realtime_push(n: int, at: float) -> list[dict]:
    """Frame ``n`` (from 0) going in at ``at``, and the server's confirmation."""
    item = {
        "id": f"<rtb_user-{n + 1}>", "type": "message", "role": "user",
        "content": [
            {"type": "input_text", "text": f"FRAME {n + 1}"},
            {"type": "input_image", "image_url": f"data:image/png;{_image(n)}"},
        ],
    }  # fmt: skip
    create = {"type": "conversation.item.create", "event_id": f"<rtb_push-{n + 1}>", "item": item}
    created = {"type": "conversation.item.created", "item": {"id": f"<rtb_user-{n + 1}>"}}
    return [_record("c2s", create, at), _record("s2c", created)]


def realtime_cassette(frame_2_at: float = 1.0) -> list[dict]:
    create = {"type": "response.create", "response": {"max_output_tokens": "inf"}}
    click = {
        "type": "response.output_text.done", "response_id": "resp_1", "item_id": "item_1",
        "text": json.dumps({"action": "click", **CLICK}),
    }  # fmt: skip
    return [
        _record("open", at=0.0),
        _record("s2c", {"type": "session.created", "session": {}}),
        _record("c2s", {"type": "session.update", "session": {"type": "realtime"}}, 0.0),
        _record("s2c", {"type": "session.updated", "session": {}}),
        *_realtime_push(0, 0.0),
        _record("c2s", create, 0.0),
        _record("s2c", {"type": "response.created", "response": {"id": "resp_1"}}, 0.2),
        _record("s2c", click, 0.4),
        _record("s2c", {"type": "response.done", "response": {"id": "resp_1"}}, 0.6),
        _record("c2s", create, 0.6),
        _record("s2c", {"type": "response.created", "response": {"id": "resp_2"}}, 0.8),
        *_realtime_push(1, frame_2_at),
        *_realtime_push(2, 2.0),
        _record(
            "c2s",
            {"type": "response.cancel", "event_id": "<rtb_cancel-1>", "response_id": "resp_2"},
            3.0,
        ),
        _record("s2c", {"type": "response.done", "response": {"id": "resp_2"}}),
        _record("close", {"by": "client", "code": 1000}),
    ]


def _live_push(n: int, at: float) -> list[dict]:
    video = {"data": _image(n), "mime_type": "image/png"}
    return [
        _record("c2s", {"realtime_input": {"video": video}}, at),
        _record("c2s", {"realtime_input": {"text": f"FRAME {n + 1}"}}, at),
    ]


def _live_trigger(at: float) -> list[dict]:
    # WireNormalizer keeps an audio blob as the SHA-256 of its bytes.
    silence = {
        "data": f"sha256:{hashlib.sha256(SILENCE).hexdigest()}",
        "mime_type": "audio/pcm;rate=16000",
    }
    return [
        _record("c2s", {"realtime_input": {"activityStart": {}}}, at),
        _record("c2s", {"realtime_input": {"audio": silence}}, at),
        _record("c2s", {"realtime_input": {"activityEnd": {}}}, at),
    ]


def live_cassette() -> list[dict]:
    call = {"id": "call_1", "name": "click", "args": CLICK}
    queued = {"id": "call_1", "name": "click", "response": {"status": "queued_not_executed"}}
    return [
        _record("open", {"path": LIVE_PATH}, 0.0),
        _record("c2s", ScriptedLiveAgent.SETUP, 0.0),
        _record("s2c", {"setupComplete": {}}),
        *_live_push(0, 0.0),
        *_live_trigger(0.0),
        _record("s2c", {"toolCall": {"functionCalls": [call]}}, 0.4),
        _record("c2s", {"tool_response": {"functionResponses": [queued]}}, 0.4),
        _record("s2c", {"serverContent": {"turnComplete": True}}, 0.6),
        *_live_trigger(0.6),
        *_live_push(1, 1.0),
        *_live_push(2, 2.0),
        # Gemini Live names no reply to cancel: giving one up closes its connection.
        _record("close", {"by": "client", "code": 1000}),
    ]


def _the_scripted_episode(episode, call_id):
    """What the episode both cassettes script leaves, whichever API it ran on."""
    env, agent = episode.env, episode.agent
    assert episode.logs == []
    assert episode.problems == []  # every message at its time, and the cassette used up
    assert episode.termination == Termination("time_limit", "deadline", 3.0, 3.5, 3.0, 3.5, None)
    assert [(p.observation_id, p.sent_at) for p in agent.pushed] == list(
        zip(FRAMES, (0.0, 1.0, 2.0), strict=True)
    )
    assert agent.started == [(1, FRAMES[0], 0.0), (2, FRAMES[0], 0.6)]
    assert (agent.accepted, agent.aborted, agent.discarded) == ([1], [(2, 3.0)], [2])
    ((committed_at, outcome),) = agent.committed
    assert (committed_at, outcome.status, outcome.call_id) == (0.4, "applied", call_id)
    assert (outcome.arrived_at, outcome.frame_age_s) == (0.4, 0.4)
    (execution,) = env.executions
    assert (execution["time"], execution["status"]) == (0.4, "applied")
    assert execution["action"]["parameters"] == {"x": 1, "y": 2}
    assert [c["call"] for c in env.calls] == [
        "env.capture", "env.execute", "env.capture", "env.capture",
        "task.evaluation_snapshot", "env.release_inputs",
    ]  # fmt: skip
    assert env._termination == {
        "stop_reason": "time_limit", "end_s": 3.0, "snapshot_s": 3.5, "snapshot_delay_s": 0.5,
    }  # fmt: skip
    assert episode.kernel("FrameCaptured", "observation_id", "timestamp") == list(
        zip(FRAMES, (0.0, 1.0, 2.0), strict=True)
    )
    assert episode.kernel("FramePushed", "observation_id", "sent_at", "error") == [
        (FRAMES[0], 0.0, None), (FRAMES[1], 1.0, None), (FRAMES[2], 2.0, None),
    ]  # fmt: skip
    assert episode.kernel("ActionArrived", "call_id", "frame", "frame_age_frames") == [
        (call_id, 1, 0)
    ]
    assert episode.kernel("ActionResolved", "status", "started_at", "finished_at") == [
        ("applied", 0.4, 0.4)
    ]
    assert episode.kernel("ResultDiscarded", "generation") == [(2,)]
    assert episode.kernel("FrameDropped") == []
    assert episode.records[-1]["payload"]["kind"] == "Termination"


def test_an_episode_over_a_replayed_openai_realtime_session(tmp_path):
    episode = run_e2e(ScriptedRealtimeAgent, realtime_cassette(), tmp_path, duration_s=3.0)
    _the_scripted_episode(episode, "item_1")
    summary = summarize_streaming(episode.trace_path, {"retained_frames": 20})
    assert summary["context_control"] == "exact"
    assert (summary["frames"]["captured"], summary["frames"]["pushed"]) == (3, 3)
    assert summary["actions"]["outcomes"] == {"applied": 1}
    assert summary["replies"] == {"started": 2, "abandoned": {"deadline": 1}}


def test_an_episode_over_a_replayed_gemini_live_session(tmp_path):
    episode = run_e2e(
        ScriptedLiveAgent, live_cassette(), tmp_path, duration_s=3.0, normalizer=WireNormalizer()
    )
    _the_scripted_episode(episode, "call_1")
    settings = {"compression_trigger_tokens": 65_536, "compression_target_tokens": 32_768}
    summary = summarize_streaming(episode.trace_path, settings)
    assert summary["context_control"] == "approximate"
    assert (summary["frames"]["pushed"], summary["frames"]["confirmed"]) == (3, 0)


def test_a_session_that_departs_from_its_cassette_ends_the_episode_invalid(tmp_path):
    """The cassette has frame 2 going in at 1.5 s; the episode pushes it as it is captured, at
    1 s. The server closes the connection over it, the reply running fails with it, and the run
    is invalid at that time."""
    episode = run_e2e(
        ScriptedRealtimeAgent, realtime_cassette(frame_2_at=1.5), tmp_path, duration_s=3.0
    )
    assert episode.problems[0] == (
        "conn 1 record 12: c2s conversation.item.create at t=1, recorded at t=1.5"
    )
    termination = episode.termination
    assert (termination.stop_reason, termination.ended_at) == ("invalid_run", 1.0)
    assert termination.invalid.reason == "generation_error"
    assert "ConnectionError: the session ended during reply 2" in termination.invalid.detail
    assert episode.kernel("FrameDropped", "observation_id", "reason") == [
        (FRAMES[1], "push_failed")
    ]
