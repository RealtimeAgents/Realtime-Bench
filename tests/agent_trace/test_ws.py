"""T4: live sessions record every WebSocket message both ways, with acknowledgements linked.

The drivers run in the shared RealtimeRuntime against local WebSocket stubs
(tests/agent_trace/stubs.py). Every frame a stub received is one client SessionMessage with the
same bytes, and the server messages the driver read are the stub's frames in order. History
changes and session-state markers come from the runtime.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from realtime_framework.clock import SystemClock
from realtime_framework.entity.actions import Observation, gui_function_declarations
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.events import TextDelta, UsageSnapshot
from realtime_framework.entity.model import (
    TRANSIENT,
    TRANSPORT_RETRY,
    GenerationOutput,
    GenerationRequest,
    ProviderObservation,
    SessionSeed,
    TransportFailure,
)
from realtime_framework.entity.records import trace_scope, validate_episode_trace
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.evaluation.agent_trace import AgentTraceSink, close_agent_trace, load_body
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.harnesses.default.context import RealtimeRuntime
from realtime_framework.harnesses.default.execution_thread import ExecutionThread
from realtime_framework.harnesses.default.request import build_request, system_block
from realtime_framework.providers.gemini import live
from realtime_framework.providers.openai_realtime import OpenAIRealtimeDriver

from .stubs import WsStub, self_signed_tls
from .test_pack import assert_no_secret

KEY = "sk-replay-SECRET-0123456789abcdef"  # a fake key; it must never reach the trace or pack
PROMPT = "frame {k}: the private page text of the replay form"
IMAGE = b"\x89PNG\r\n\x1a\n" + os.urandom(6000)  # the stubs never decode it


class _Collector:
    def __init__(self):
        self.text: list[str] = []
        self.usage = None

    def consume(self, event):
        if isinstance(event, TextDelta):
            self.text.append(event.text)
        elif isinstance(event, UsageSnapshot):
            self.usage = event.usage

    def finish(self):
        return GenerationOutput("".join(self.text), self.usage, {}, ())


def _observation(k: int) -> ProviderObservation:
    return ProviderObservation(
        observation=Observation(b"", float(k), k, {"observation_id": f"obs-{k}"}),
        prompt=PROMPT.format(k=k),
        image_bytes=IMAGE,
        mime_type="image/png",
        image_detail="auto",
        geometry={},
    )


def _run(driver, run_dir: Path, turns: int):
    """Start a session and run ``turns`` decisions the way DefaultHarness drives the runtime."""
    clock = SystemClock()
    trace = EpisodeTrace(run_dir / "episode_trace.jsonl", run_id="ws", clock=clock)
    sink = AgentTraceSink(trace, run_dir / "episode_blobs.pack", clock=clock)
    driver.bind_trace(sink)
    config = driver.config
    policy = current_context_policy().for_path(
        config.adapter_type, config.provider_options, config.context_preset
    )
    execution = ExecutionThread(SystemClock(), "execution-thread-ws")
    runtime = RealtimeRuntime(driver, policy, trace=sink, execution=execution)
    history: list[dict] = []
    channel = driver.capabilities.action_channel
    # As DefaultHarness._answer_requirements declares them (harnesses/default/harness.py).
    declarations = tuple(gui_function_declarations()) if channel == "function_call" else ()
    answer = AnswerRequirements(channel, "standard_vlm", declarations)
    seed = SessionSeed("system instruction", config.model_name, None, 1.0, answer)
    system = system_block(seed.system_instruction, None)
    call = CallParameters(config.model_name, config.reasoning_effort, None)

    def request_builder(turns, ledger):
        return build_request(policy, system, turns, ledger, answer, call)

    execution.start()
    try:
        with trace_scope(phase="initialize"):
            runtime.start(seed)
        for k in range(turns):
            request = GenerationRequest(f"turn-{k}", None, "inf", "wall")
            with trace_scope(phase="decision", decision=k, generation_id=f"turn-{k}", attempt=1):
                result = runtime.run_turn(
                    [_observation(k)],
                    request,
                    _Collector(),
                    action_history=history,
                    timeout=60,
                    request_builder=request_builder,
                    decision=k,
                    cancel=runtime.clock.event(),
                )
            runtime.accept_turn(result.turn)
            history.append({"tick": k + 1, "executed": "wait", "result": "wait"})
    finally:
        with trace_scope(phase="close"):
            runtime.close()
        execution.stop()
    assert close_agent_trace(sink, None) is None
    trace.close()
    validate_episode_trace(run_dir / "episode_trace.jsonl")
    assert_no_secret(run_dir, KEY)
    text = (run_dir / "episode_trace.jsonl").read_text()
    assert "private page text" not in text and "system instruction" not in text
    return [json.loads(line) for line in text.splitlines()]


def _messages(events, connection_id: str) -> list[dict]:
    return [
        e["payload"]
        for e in events
        if e["event"] == "SessionMessage" and e["payload"]["connection_id"] == connection_id
    ]


def _check_frames(events, stub: WsStub, channel: str, pack: Path) -> list[str]:
    """Client records equal the frames each stub connection received; server records are the
    frames it sent, in order. Returns the connection ids that carried messages."""
    ids = []
    for connection in stub.connections:
        ids.append(f"{channel}-{len(ids) + 1 + stub.rejected}")
        messages = _messages(events, ids[-1])
        client = [m for m in messages if m["direction"] == "client"]
        server = [m for m in messages if m["direction"] == "server"]
        assert [load_body(pack, m["body"]).decode() for m in client] == connection.received
        assert [load_body(pack, m["body"]).decode() for m in server] == connection.sent[
            : len(server)
        ]
    return ids


def _by_type(messages, direction, message_type) -> list[dict]:
    return [
        m for m in messages if m["direction"] == direction and m["message_type"] == message_type
    ]


# ---- OpenAI Realtime -----------------------------------------------------------------------


def _realtime_server():
    """Answer like the Realtime API; ``state["fail_delete"]`` rejects the next item delete."""
    state = {"fail_delete": False, "responses": 0}

    def handle(stub, connection, message):
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
        if kind == "conversation.item.delete":
            if state["fail_delete"]:
                state["fail_delete"] = False
                error = {"type": "invalid_request_error", "message": "no such item"}
                return [{"type": "error", "error": {**error, "event_id": message["event_id"]}}]
            return [{"type": "conversation.item.deleted", "item_id": message["item_id"]}]
        if kind == "response.create":
            state["responses"] += 1
            response_id, item_id = f"resp_{state['responses']}", f"srv_item_{state['responses']}"
            return [
                {"type": "response.created", "response": {"id": response_id}},
                {
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "item": {"id": item_id},
                },
                {
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "delta": '{"action": "wait"}',
                },
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
        return []

    return state, handle


def test_openai_realtime_session(tmp_path, monkeypatch):
    # Fold one screenshot per turn, so turn 1 edits history in place and turn 2's edit fails.
    monkeypatch.setenv("STREAM_ACTIVE_VISUAL_BUDGET", "1")
    monkeypatch.setenv("STREAM_VISUAL_FOLD_BLOCK_SIZE", "1")
    monkeypatch.setenv("REPLAY_TRACE_KEY", KEY)
    state, handle = _realtime_server()
    with WsStub(handle, reject=1) as stub:
        driver = OpenAIRealtimeDriver(
            ModelConfig(
                adapter_type="openai_realtime",
                model_name="gpt-realtime-replay",
                api_base_url=f"http://127.0.0.1:{stub.port}/v1",
                api_key_env="REPLAY_TRACE_KEY",
            )
        )
        original = driver.sync_history

        async def sync_history(request):
            frames = [item.block for item in request.history.items if item.block.kind == "frame"]
            state["fail_delete"] = frames[-1].observation_id == "obs-1"
            return await original(request)

        driver.sync_history = sync_history
        events = _run(driver, tmp_path, turns=3)

    pack = tmp_path / "episode_blobs.pack"
    assert stub.rejected == 1
    (failed,) = _messages(events, "openai_realtime.ws-1")
    assert failed["direction"] == "lifecycle" and failed["message_type"] == "handshake_failed"
    assert failed["scope"] == {"phase": "initialize", "connect_attempt": 1}
    # connect() retries the refused upgrade (HTTP 503) under the shared policy, and records each
    # attempt; turn 2's rebuild connects once more.
    attempts = [
        e["payload"]
        for e in events
        if e["event"] == "ModelAttemptFinished" and e["payload"]["kind"] == "session_connect"
    ]
    connects = [
        (a["scope"].get("decision"), a["scope"]["connect_attempt"], a["outcome"], a["failure_kind"])
        for a in attempts
    ]
    assert connects == [(None, 1, "retry", "transient"), (None, 2, "ok", None), (2, 1, "ok", None)]
    assert attempts[0]["retry_delay_s"] == TRANSPORT_RETRY.delay_s(
        1, "empty:connect", TransportFailure(TRANSIENT)
    )
    first, rebuilt = _check_frames(events, stub, "openai_realtime.ws", pack)
    messages = _messages(events, first)

    def acked_by(server_type, client_type, *, repeat=False, outcome="ok"):
        (client,) = _by_type(messages, "client", client_type)[:1]
        replies = [m for m in _by_type(messages, "server", server_type) if m["repeat"] == repeat]
        assert replies and replies[0]["acknowledges"] == [client["capture_seq"]]
        assert replies[0]["ack_outcome"] == outcome and replies[0]["origin"] == "ack"
        assert client["state"] == "sent"

    acked_by("session.updated", "session.update")
    acked_by("conversation.item.added", "conversation.item.create")
    acked_by("conversation.item.done", "conversation.item.create", repeat=True)
    acked_by("response.created", "response.create")
    acked_by("conversation.item.deleted", "conversation.item.delete")
    (created,) = _by_type(messages, "server", "session.created")
    assert created["origin"] == "unsolicited" and created["acknowledges"] == []
    assert {m["origin"] for m in _by_type(messages, "server", "response.output_item.added")} == {
        "response"
    }
    (error,) = _by_type(messages, "server", "error")
    deletes = _by_type(messages, "client", "conversation.item.delete")
    assert error["acknowledges"] == [deletes[-1]["capture_seq"]]
    assert error["ack_outcome"] == "error"

    changes = [e["payload"] for e in events if e["event"] == "HistoryChanged"]
    assert [(c["scope"]["decision"], c["kind"]) for c in changes] == [
        (1, "visual_fold"),
        (1, "sync"),
        (2, "visual_fold"),
        (2, "in_place_edit_fallback"),
    ]
    assert changes[1]["sync"]["disposition"] == "exact"
    assert changes[1]["policy_item"] == ["visuals"]
    fallback = changes[3]
    assert fallback["trigger"] == "fallback" and fallback["replayed_turns"] == 2
    assert "no such item" in fallback["sync"]["provider_metadata"]["history_in_place_edit_error"]
    # The rebuild replays the session on a new connection.
    assert _by_type(_messages(events, rebuilt), "client", "session.update")
    (closed,) = [e["payload"] for e in events if e["event"] == "AgentTraceClosed"]
    sessions = {s["connection_id"]: s for s in closed["sessions"]}
    assert sessions[first]["ack_errors"] == 1 and sessions[first]["client_acked"] > 0


# ---- Gemini Live ---------------------------------------------------------------------------


def _live_server():
    """Answer like Gemini Live; each turn ends with a GoAway and a resumption handle.

    ``state["fail_resume"]`` closes a setup that resumes a handle, as a failed resumption does.
    """
    state = {"fail_resume": False, "turns": 0}

    def handle(stub, connection, message):
        if message is None:
            return []
        if "setup" in message:
            setup = json.dumps(message["setup"])
            if "handle-" in setup and state["fail_resume"]:
                return None
            connection.history = "istory" in setup  # history_config: seeded turns follow
            return [{"setupComplete": {}}]
        content = message.get("client_content") or message.get("clientContent")
        if content is None:
            return []  # tool_response and realtime_input have no reply
        if connection.history:
            connection.history = not content.get("turn_complete", content.get("turnComplete"))
            return []
        state["turns"] += 1
        n = state["turns"]
        return [
            {"toolCall": {"functionCalls": [{"id": f"call-{n}", "name": "wait", "args": {}}]}},
            {"sessionResumptionUpdate": {"newHandle": f"handle-{n}", "resumable": True}},
            {"goAway": {"timeLeft": "30s"}},
            {"serverContent": {"generationComplete": True}},
            {
                "serverContent": {"turnComplete": True},
                "usageMetadata": {
                    "promptTokenCount": 20,
                    "responseTokenCount": 3,
                    "totalTokenCount": 23,
                },
            },
        ]

    return state, handle


def test_gemini_live_session(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "_SESSION_RELEASE_GRACE_S", 0.0)
    monkeypatch.setenv("REPLAY_TRACE_KEY", KEY)
    tls, certificate = self_signed_tls(tmp_path)
    # google-genai reads the CA file when the client is built (genai/_api_client.py:1257-1260).
    monkeypatch.setenv("SSL_CERT_FILE", str(certificate))
    state, handle = _live_server()
    model = "gemini-3.1-flash-live-preview"
    with WsStub(handle, tls=tls) as stub:
        driver = live.GeminiLiveDriver(
            ModelConfig(
                adapter_type="gemini_live",
                model_name=model,
                api_base_url=f"https://127.0.0.1:{stub.port}",
                api_key_env="REPLAY_TRACE_KEY",
                **live.GeminiLiveDriver.model_defaults[model],
            )
        )
        original = driver.sync_history

        async def sync_history(request):
            state["fail_resume"] = request.current_generation.generation_id == "turn-1"
            return await original(request)

        driver.sync_history = sync_history
        events = _run(driver, tmp_path, turns=2)

    pack = tmp_path / "episode_blobs.pack"
    first, _, refused, replayed = _check_frames(events, stub, "gemini_live.ws", pack)
    messages = _messages(events, first)
    (setup,) = _by_type(messages, "client", "setup")
    (complete,) = _by_type(messages, "server", "setupComplete")
    assert complete["acknowledges"] == [setup["capture_seq"]] and complete["origin"] == "ack"
    acknowledged = {
        seq for m in messages if m["direction"] == "server" for seq in m["acknowledges"]
    }
    for kind in ("client_content", "tool_response"):
        (sent,) = _by_type(messages, "client", kind)
        assert sent["state"] == "sent" and sent["capture_seq"] not in acknowledged
    for kind in ("goAway", "sessionResumptionUpdate"):
        assert {m["origin"] for m in _by_type(messages, "server", kind)} == {"unsolicited"}
    assert {m["origin"] for m in _by_type(messages, "server", "toolCall")} == {"response"}

    changes = [e["payload"] for e in events if e["event"] == "HistoryChanged"]
    assert [(c["scope"]["decision"], c["kind"]) for c in changes] == [
        (0, "session_resumed"),
        (1, "resume_fallback"),
    ]
    assert changes[0]["trigger"] == "provider" and changes[0]["sync"]["reason"] == "go_away"
    assert changes[1]["trigger"] == "fallback" and changes[1]["replayed_turns"] == 2
    assert changes[1]["policy_item"] == ["tool_call_replay"]
    assert "resume_error" in changes[1]["sync"]["provider_metadata"]
    states = [e["payload"] for e in events if e["event"] == "SessionStateChanged"]
    assert [(s["state"], s["cause"]) for s in states] == [
        ("unknown", "session_resumed"),
        ("exact", "sync:go_away"),
    ]
    # The refused resumption sent its setup and was never acknowledged.
    assert [m["message_type"] for m in _messages(events, refused)] == ["setup"]
    # The replay seeds both earlier turns on the new connection, screenshots included.
    (seeded,) = _by_type(_messages(events, replayed), "client", "client_content")
    history = json.loads(load_body(pack, seeded["body"]))["client_content"]
    frames = [turn["parts"][0]["text"] for turn in history["turns"] if turn["role"] == "user"]
    assert frames == [PROMPT.format(k=0), PROMPT.format(k=1)]
    assert len(seeded["body"]["spans"]) == 2  # the two screenshots, stored in the pack
