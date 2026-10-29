"""Native-streaming episodes end to end on the real agent and drivers (tests/streaming/e2e.py).

NativeStreamingAgent runs on OpenAIRealtimeDriver or GeminiLiveDriver, each connected to the
cassette's server on the fake clock. The environment captures a frame at 0, 1 and 2 s, and the
agent pushes each as it comes. Reply 1 starts on the first frame at 0 s; its click, based on
frame 1, comes off the wire at 0.4 s and takes 0.1 s to execute. Reply 2 starts when reply 1
ends and never ends: the deadline at 3 s gives it up, and the snapshot takes 0.5 s.

The opening message's system instruction, and Gemini's function declarations, are placeholders
in the cassettes. Each test checks the values they stand for: the instruction against the agent's
seed, and the declarations against the GUI functions that take the frame an action is based on.
"""

from __future__ import annotations

import hashlib
import json
from typing import ClassVar

import pytest

from realtime_framework.entity.actions import Observation, gui_function_declarations
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.reporting import summarize_streaming
from realtime_framework.harnesses.default.prompts import (
    STREAMING_FUNCTION_CALL_FRAME_PROMPT,
    STREAMING_TEXT_ACTION_FRAME_PROMPT,
)
from realtime_framework.harnesses.registry import configure_agent, create_agent
from realtime_framework.providers import registry as catalog
from realtime_framework.sandbox.observation.capture import prepare_for_model
from realtime_framework.sandbox.observation.som import (
    FRAME_INTERACTABLES_HEADER,
    observation_prompt_with_interactables,
    render_frame_number,
)
from tests.replay import baseline
from tests.replay.fake_env import FakeEnvironment
from tests.replay.frames import screenshot
from tests.replay.ws_gemini_live import QUEUED_RESPONSE, WireNormalizer
from tests.streaming.e2e import LIVE_PATH, SILENCE, run_e2e
from tests.streaming.fake_live import RecordingToolRuntime
from tests.streaming.ws_cassette import RealtimeNormalizer

FRAMES = [f"replay-form:{n:06d}" for n in (1, 2, 3)]  # tests/replay/frames.py ids, captures 0-2
TASK = "Click the button."
CLICK = {"x": 100, "y": 200, "frame": 1}
REQUESTED = "click {'x': 100, 'y': 200}"
FEEDBACK = f"EXECUTION RESULT: actual={REQUESTED}; requested={REQUESTED}; status=applied."
INSTRUCTION = "<system instruction>"
TOOLS = "<tools>"
EXECUTION_S = 0.1


@pytest.fixture(autouse=True)
def _slow_click(monkeypatch):
    """The click takes 0.1 s, so that its feedback is committed at 0.5 s: after OpenAI's reply 2
    started (0.45 s), so that it goes in with the next frame, and before Gemini's (0.6 s), whose
    trigger carries it. Neither then turns on which of two threads goes first."""
    execute = FakeEnvironment.execute

    def slow(self, action, *, release_at=None):
        status = execute(self, action, release_at=release_at)
        self.clock.sleep(EXECUTION_S)
        return status

    monkeypatch.setattr(FakeEnvironment, "execute", slow)


class _Placeholders:
    """Replaces the value under each key in PLACEHOLDERS with its placeholder, and keeps it."""

    PLACEHOLDERS: ClassVar[dict[str, str]] = {}

    def __init__(self):
        super().__init__()
        self.hidden: dict[str, object] = {}

    def __call__(self, value, key=None):
        if key in self.PLACEHOLDERS:
            self.hidden[key] = value
            return self.PLACEHOLDERS[key]
        return super().__call__(value, key)


class RealtimeWire(_Placeholders, RealtimeNormalizer):
    PLACEHOLDERS: ClassVar[dict[str, str]] = {"instructions": INSTRUCTION}


class LiveWire(_Placeholders, WireNormalizer):
    PLACEHOLDERS: ClassVar[dict[str, str]] = {"systemInstruction": INSTRUCTION, "tools": TOOLS}


def _initialized(agent, clock):
    configure_agent(
        agent,
        fps=1.0,
        tokens_per_frame=None,
        clock_mode="wall",
        category="realtime",
        interaction_mode="streaming",
    )
    agent.initialize(TASK, Observation(b"", clock.time(), 0, {}))
    return agent


def _agent(config, log_dir, trace, clock):
    """The agent as the runner sets it up, with a tool runtime whose inspections find no element
    (runner.py set_up_agent)."""
    agent = create_agent(config, log_dir, trace=trace, clock=clock, interaction_mode="streaming")
    agent.bind_tool_runtime(RecordingToolRuntime())
    return agent


def _frame_text(prompt: str, n: int) -> str:
    """Capture ``n``'s (from 0) text: its prompt, naming its frame, and the interactables its
    inspection found, none."""
    return observation_prompt_with_interactables(
        prompt.format(frame=n + 1), "", FRAME_INTERACTABLES_HEADER
    )


def realtime_agent(log_dir):
    def build(clock, port, *, trace):
        config = catalog.to_model_config(
            "gpt-realtime-2", fps=1.0, reasoning_effort="low", api_key="sk-replay"
        ).model_copy(update={"api_base_url": f"http://127.0.0.1:{port}/v1"})
        agent = _agent(config, log_dir, trace, clock)
        return _initialized(agent, clock)

    return build


def live_agent(log_dir):
    def build(clock, port, *, trace):
        config = catalog.to_model_config(
            "gemini-3.8-live", fps=1.0, reasoning_effort="fixed", api_key="replay-key"
        )
        agent = _agent(config, log_dir, trace, clock)
        # google-genai connects over wss whenever it has an API key; the server is plain ws.
        api_client = agent.driver.client._api_client
        api_client._websocket_base_url = lambda: f"ws://127.0.0.1:{port}"
        del api_client._websocket_ssl_ctx["ssl"]
        return _initialized(agent, clock)

    return build


def _record(direction, msg=None, at=None):
    return {"conn": 1, "dir": direction, "at": at, "msg": msg}


def _frame(n: int) -> str:
    """Capture ``n`` (from 0) as the agent sends it: scaled for the model, its number drawn."""
    return baseline.image_digest(render_frame_number(prepare_for_model(screenshot(n)).data, n + 1))


def _realtime_push(n: int, content: list[dict], at: float, previous: str | None) -> list[dict]:
    """Item ``n`` (from 1) going in at ``at``, and the server's confirmation of it."""
    item = {"id": f"<rtb_user-{n}>", "type": "message", "role": "user", "content": content}
    create = {"type": "conversation.item.create", "event_id": f"<rtb_push-{n}>", "item": item}
    done = {
        "type": "conversation.item.done", "previous_item_id": previous,
        "item": {"id": f"<rtb_user-{n}>"},
    }  # fmt: skip
    return [_record("c2s", create, at), _record("s2c", done)]


def _realtime_frame(n: int) -> list[dict]:
    return [
        {"type": "input_text", "text": _frame_text(STREAMING_TEXT_ACTION_FRAME_PROMPT, n)},
        {"type": "input_image", "image_url": f"data:image/png;{_frame(n)}"},
    ]


def realtime_cassette(frame_2_at: float = 1.0) -> list[dict]:
    """``frame_2_at``: when the cassette has the second frame go in."""
    session = {
        "type": "realtime", "instructions": INSTRUCTION, "output_modalities": ["text"],
        "truncation": "disabled",
    }  # fmt: skip
    create = {
        "type": "response.create",
        "response": {"max_output_tokens": "inf", "reasoning": {"effort": "low"}},
    }
    text = json.dumps({"action": "click", **CLICK})
    added = {"type": "response.output_item.added", "response_id": "resp_1"}
    reply = {"response_id": "resp_1", "item_id": "item_1", "output_index": 0, "content_index": 0}
    usage = {
        "input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
        "input_token_details": {
            "cached_tokens": 0, "text_tokens": 10, "image_tokens": 10, "audio_tokens": 0,
        },
        "output_token_details": {"text_tokens": 5, "audio_tokens": 0},
    }  # fmt: skip
    done = {
        "id": "resp_1", "status": "completed", "output": [{"id": "item_1"}], "usage": usage,
    }  # fmt: skip
    return [
        _record("open", {"path": "/v1/realtime?model=gpt-realtime-2"}, 0.0),
        _record("s2c", {"type": "session.created", "session": {}}),
        _record("c2s", {"type": "session.update", "session": session}, 0.0),
        _record("s2c", {"type": "session.updated", "session": {}}),
        *_realtime_push(1, _realtime_frame(0), 0.0, None),
        _record("c2s", {**create, "event_id": "<rtb_reply-1>"}, 0.0),
        _record("s2c", {"type": "response.created", "response": {"id": "resp_1"}}, 0.2),
        _record("s2c", {**added, "item": {"id": "item_1"}}, 0.2),
        _record("s2c", {"type": "response.output_text.delta", **reply, "delta": text}, 0.3),
        # The click is handed over as its text is done, before its reply is.
        _record("s2c", {"type": "response.output_text.done", **reply, "text": text}, 0.4),
        _record("s2c", {"type": "response.done", "response": done}, 0.45),
        _record("c2s", {**create, "event_id": "<rtb_reply-2>"}, 0.45),
        _record("s2c", {"type": "response.created", "response": {"id": "resp_2"}}, 0.6),
        # The click's feedback, committed at 0.5 s, goes in before the next frame.
        *_realtime_push(2, [{"type": "input_text", "text": FEEDBACK}], 1.0, "item_1"),
        *_realtime_push(3, _realtime_frame(1), frame_2_at, "<rtb_user-2>"),
        *_realtime_push(4, _realtime_frame(2), 2.0, "<rtb_user-3>"),
        _record(
            "c2s",
            {"type": "response.cancel", "event_id": "<rtb_cancel-1>", "response_id": "resp_2"},
            3.0,
        ),
        _record("close", {"by": "client", "code": 1000}, 3.0),
    ]


def _live_input(at: float, **realtime_input) -> dict:
    return _record("c2s", {"realtime_input": realtime_input}, at)


def _live_push(n: int, at: float) -> list[dict]:
    """Capture ``n`` (from 0) going in at ``at``: its image, then its text."""
    return [
        _live_input(at, video={"data": _frame(n), "mime_type": "image/png"}),
        _live_input(at, text=_frame_text(STREAMING_FUNCTION_CALL_FRAME_PROMPT, n)),
    ]


def _live_trigger(at: float, *text: str) -> list[dict]:
    """A reply's start: one activity, the text it carries and the silence that opens it."""
    silence = {
        "data": "sha256:" + hashlib.sha256(SILENCE).hexdigest(), "mime_type": "audio/pcm;rate=16000",
    }  # fmt: skip
    return [
        _live_input(at, activityStart={}),
        *(_live_input(at, text=part) for part in text),
        _live_input(at, audio=silence),
        _live_input(at, activityEnd={}),
    ]


def live_cassette() -> list[dict]:
    setup = {
        "model": "models/gemini-3.8-live",
        "generationConfig": {"responseModalities": ["AUDIO"]},
        "systemInstruction": INSTRUCTION,
        "tools": TOOLS,
        "realtimeInputConfig": {
            "activity_handling": "NO_INTERRUPTION",
            "automatic_activity_detection": {"disabled": True},
            "turn_coverage": "TURN_INCLUDES_ALL_INPUT",
        },
        "contextWindowCompression": {
            "sliding_window": {"target_tokens": 32768}, "trigger_tokens": 65536,
        },
        "outputAudioTranscription": {},
        "sessionResumption": {},
    }  # fmt: skip
    call = {"id": "call_1", "name": "click", "args": CLICK}
    response = {
        "id": "call_1", "name": "click", "response": QUEUED_RESPONSE, "scheduling": "SILENT",
        "will_continue": False,
    }  # fmt: skip
    usage = {"promptTokenCount": 20, "responseTokenCount": 3, "totalTokenCount": 23}
    return [
        _record("open", {"path": LIVE_PATH}, 0.0),
        _record("c2s", {"setup": setup}, 0.0),
        _record("s2c", {"setupComplete": {}}),
        *_live_push(0, 0.0),
        *_live_trigger(0.0),
        _record("s2c", {"toolCall": {"functionCalls": [call]}}, 0.4),
        _record("c2s", {"tool_response": {"functionResponses": [response]}}, 0.4),
        _record("s2c", {"serverContent": {"turnComplete": True}, "usageMetadata": usage}, 0.6),
        *_live_trigger(0.6, FEEDBACK),
        *_live_push(1, 1.0),
        *_live_push(2, 2.0),
        _record("close", {"by": "client", "code": 1000}, 3.0),
    ]


def _agent_records(episode, event: TraceEvent) -> list[dict]:
    return [row["payload"] for row in episode.records if row["event"] == event.value]


def _delivered(episode, state: str) -> list[str | None]:
    return [
        content["block"]
        for payload in _agent_records(episode, TraceEvent.SESSION_DELIVERY)
        if payload["state"] == state
        for content in payload["contents"]
    ]


def _the_episode(episode, call_id: str) -> dict:
    """What either cassette's episode leaves in the kernel, the environment and the agent; and
    the episode's streaming summary."""
    env, agent = episode.env, episode.agent
    assert episode.logs == []
    assert episode.problems == []  # every message at its time, and the cassette used up
    # As main() reads them (runner.py main).
    result = episode.result
    assert (result.stop_reason, result.invalid_run) == ("time_limit", None)
    assert result.termination_record == {
        "stop_reason": "time_limit", "end_s": 3.0, "snapshot_s": 3.5, "snapshot_delay_s": 0.5,
    }  # fmt: skip
    (execution,) = env.executions
    assert (execution["time"], execution["status"]) == (0.4, "applied")
    assert execution["action"]["parameters"] == {"x": 100, "y": 200}
    assert execution["action"]["coordinate_space"] == "model_image_px"
    assert [c["call"] for c in env.calls] == [
        "env.capture", "env.execute", "env.capture", "env.capture",
        "task.evaluation_snapshot", "env.release_inputs",
    ]  # fmt: skip
    assert episode.kernel("FramePushed", "observation_id", "sent_at", "error") == [
        (FRAMES[0], 0.0, None), (FRAMES[1], 1.0, None), (FRAMES[2], 2.0, None),
    ]  # fmt: skip
    assert episode.kernel(
        "ActionArrived", "call_id", "frame", "frame_age_s", "frame_age_frames", "generation"
    ) == [(call_id, 1, 0.4, 0, 1)]
    assert episode.kernel("ActionResolved", "status", "started_at", "finished_at") == [
        ("applied", 0.4, 0.4 + EXECUTION_S)
    ]
    assert episode.kernel("ResultDiscarded", "generation") == [(2,)]
    assert episode.kernel("FrameDropped") == []
    assert agent.action_history == [
        {
            "decision": 0, "call_id": call_id, "requested": REQUESTED, "executed": REQUESTED,
            "result": "applied", "detail": "",
        }
    ]  # fmt: skip
    assert [
        (row.frame, row.observation_id, row.send_status, row.sent_unix_s)
        for row in agent.frames._frames.values()
    ] == [(1, FRAMES[0], "sent", 0.0), (2, FRAMES[1], "sent", 1.0), (3, FRAMES[2], "sent", 2.0)]
    assert [
        (sent["frame"], sent["observation_id"])
        for sent in _agent_records(episode, TraceEvent.FRAME_SENT)
    ] == [(1, FRAMES[0]), (2, FRAMES[1]), (3, FRAMES[2])]
    trajectory = episode.trace_path.parent / "agent" / "trajectory.jsonl"
    rows = [json.loads(line) for line in trajectory.read_text().splitlines()]
    assert [(row["type"], row["generation"], row["observation_id"]) for row in rows] == [
        ("reply", 1, FRAMES[0]), ("reply_discarded", 2, FRAMES[0]),
    ]  # fmt: skip
    summary = summarize_streaming(
        episode.trace_path, agent.driver.streaming_settings, attempt_id=episode.attempt_id
    )
    assert (summary["frames"]["captured"], summary["frames"]["pushed"]) == (3, 3)
    assert summary["actions"]["outcomes"] == {"applied": 1}
    assert summary["replies"] == {"started": 2, "abandoned": {"deadline": 1}}
    return summary


def test_the_agent_on_openai_realtime_over_a_replayed_session(tmp_path):
    wire = RealtimeWire()
    episode = run_e2e(
        realtime_agent(tmp_path / "agent"), realtime_cassette(), tmp_path, duration_s=3.0,
        normalizer=wire,
    )  # fmt: skip
    summary = _the_episode(episode, "text")
    assert wire.hidden == {"instructions": episode.agent._seed.system_instruction}
    # The server confirms every frame and the feedback as it takes them.
    blocks = ["f1.frame", "d0.text.feedback", "f2.frame", "f3.frame"]
    assert _delivered(episode, "sent") == blocks
    assert _delivered(episode, "confirmed") == blocks
    assert summary["context_control"] == "exact"
    assert summary["frames"]["confirmed"] == 3
    assert summary["frames"]["send_to_confirm_s"]["max"] == 0.0


def test_the_agent_on_gemini_live_over_a_replayed_session(tmp_path):
    wire = LiveWire()
    episode = run_e2e(
        live_agent(tmp_path / "agent"), live_cassette(), tmp_path, duration_s=3.0,
        normalizer=wire,
    )  # fmt: skip
    summary = _the_episode(episode, "call_1")
    agent = episode.agent
    assert wire.hidden["systemInstruction"] == {"parts": [{"text": agent._seed.system_instruction}]}
    # Each GUI function takes the frame its action is based on, as an integer it must give
    # (phase2.md, 动作的定位依据); the session sets how a call runs.
    (tools,) = wire.hidden["tools"]
    declarations = [
        {key: value for key, value in tool.items() if key != "behavior"}
        for tool in tools["functionDeclarations"]
    ]
    assert declarations == list(gui_function_declarations(clock_mode="wall", frame_reference=True))
    assert all("frame" in d["parameters_json_schema"]["required"] for d in declarations)
    # Gemini Live confirms nothing but the setup.
    assert _delivered(episode, "sent") == [
        "f1.frame", None, "d0.call_1.feedback", "f2.frame", "f3.frame",
    ]  # fmt: skip
    assert _delivered(episode, "confirmed") == ["system"]
    assert summary["context_control"] == "approximate"
    assert summary["frames"]["confirmed"] == 0


def test_a_session_that_departs_from_its_cassette_is_closed_with_the_departure(tmp_path):
    """The cassette has the second frame go in at 1.5 s; the agent pushes it as it is captured,
    at 1 s. The server closes the connection over it with the departure, which fails that push
    and the reply running. That reply had said nothing yet, so it is asked for again, as a turn
    would be; the server takes no other connection, and the deadline ends the episode while the
    session is being reopened."""
    episode = run_e2e(
        realtime_agent(tmp_path / "agent"), realtime_cassette(frame_2_at=1.5), tmp_path,
        duration_s=3.0, normalizer=RealtimeWire(),
    )  # fmt: skip
    assert episode.problems == [
        "conn 1 record 16: c2s conversation.item.create at t=1, recorded at t=1.5"
    ]
    assert episode.kernel("FrameDropped", "observation_id", "reason") == [
        (FRAMES[1], "push_failed")
    ]
    retried = next(
        payload
        for payload in _agent_records(episode, TraceEvent.MODEL_ATTEMPT_FINISHED)
        if payload["kind"] == "reply"
    )
    assert (retried["outcome"], retried["failure_kind"]) == ("retry", "transient")
    assert "code=1011, reason='conn 1 record 16" in retried["error"]
    result = episode.result
    assert (result.stop_reason, result.termination_record["end_s"]) == ("time_limit", 3.0)
