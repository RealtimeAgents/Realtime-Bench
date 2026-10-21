"""One native-streaming episode end to end on the fake clock: a live session replayed from a
cassette (tests/streaming/clocked_ws.py CassetteServer), the replay's wall-clock environment
(tests/replay/fake_env.py) and the streaming shell (evaluation/streaming_shell.py).

``run_e2e`` takes the agent from a factory, ``agent_factory(clock, port, origin)``, built against
the cassette's server on ``port``; ``origin`` is the environment's t0, from which an action's
arrival is read in semantic seconds. Until the native-streaming agent exists, the agents here are
scripted: each speaks its API's wire on one clocked connection, as its driver's streaming
methods do, and hands an action over as the message carrying it comes off the wire.

- ScriptedRealtimeAgent, for OpenAI Realtime: a push writes the frame's item and returns once
  the server confirmed it; a reply asks with response.create and ends at response.done, or at
  its abort, which cancels the response; each completed text is one JSON action.
- ScriptedLiveAgent, for Gemini Live: a push writes the frame as realtime_input video and its
  text, which the server confirms none of; a reply is one activity, carrying 100 ms of silence,
  and ends at turnComplete, or at its abort, which closes the connection; each function call is
  one action, acknowledged as queued.

An action names its frame by the number the agent pushed it under, from 1; its age is read off
the frames pushed so far, as the frame table reads it (harnesses/default/frames.py age).

One thread reads each connection and wakes whoever waits: websockets' sync recv takes one reader.
What a push or a reply waits for is registered before the message that asks for it goes out, as
an answer may come at the same fake instant.
"""

from __future__ import annotations

import base64
import json
import threading
import uuid
from dataclasses import dataclass
from typing import ClassVar

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from realtime_framework.clock import Clocks
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.entity.model import PendingDecision
from realtime_framework.entity.records import validate_episode_trace
from realtime_framework.evaluation.kernel.events import Termination
from realtime_framework.evaluation.kernel.streaming import ActionArrived, StreamingConfig
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.evaluation.streaming_shell import run_streaming_episode
from tests.fake_clock import FakeClock
from tests.replay.fake_env import FakeEnvironment
from tests.replay.frames import FrameSource, ReplayTask
from tests.streaming.clocked_ws import (
    REAL_LIMIT_S,
    CassetteServer,
    clocked_connection,
    real_time_limit,
)
from tests.streaming.ws_cassette import RealtimeNormalizer

LIVE_PATH = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
SILENCE = bytes(3200)  # 100 ms of 16 kHz PCM, the activity's audio (providers/gemini/live.py)


def new_id(kind: str) -> str:
    """One of our ids, which the cassette's normalizer numbers (ws_cassette.RealtimeNormalizer)."""
    return f"rtb_{kind}_{uuid.uuid4().hex[:16]}"


@dataclass
class _Pushed:
    observation_id: str
    captured_at: float  # semantic seconds
    sent_at: float


class _Reply:
    def __init__(self, clock, generation: int):
        self.generation, self.ended = generation, clock.event()
        self.done = self.aborted = False
        self.response_id: str | None = None  # OpenAI Realtime's


class _ScriptedAgent:
    """What both scripted agents share: the StreamingKernelAgent bookkeeping, the reader thread
    and the frames pushed. A subclass connects, pushes, triggers and reads its own wire."""

    def __init__(self, clock, url: str, origin: float):
        self.clock, self.origin, self.post = clock, origin, None
        self.pushed: list[_Pushed] = []
        self.started, self.aborted = [], []
        self.accepted, self.discarded, self.committed = [], [], []
        self._replies: dict[int, _Reply] = {}
        self._lock = threading.Lock()
        self._closed = False
        self.ws = connect(url, create_connection=clocked_connection(ClientConnection, clock))
        self._reader = clock.start_thread(self._read, name="e2e-reader")

    def now(self) -> float:
        return self.clock.time() - self.origin

    def send(self, message: dict) -> None:
        self.ws.send(json.dumps(message))

    def _read(self) -> None:
        try:
            while True:
                self.received(json.loads(self.ws.recv()))
        except ConnectionClosed:
            pass
        finally:  # the session is over: wake every waiter
            with self._lock:
                self._closed = True
                waiting = [*self.waiters(), *(r.ended for r in self._replies.values())]
            for event in waiting:
                event.set()

    def waiters(self) -> list:
        return []

    def received(self, message: dict) -> None:
        raise NotImplementedError

    def arrived(self, generation: int, action: Action, call_id: str, frame) -> None:
        """Hand an action over as its message came off the wire, with its frame's age."""
        now = self.now()
        age_s = age_frames = None
        if type(frame) is int and 0 < frame <= len(self.pushed):
            age_s = round(now - self.pushed[frame - 1].captured_at, 9)
            age_frames = len(self.pushed) - frame
        self.post(ActionArrived(generation, action, now, call_id, frame, age_s, age_frames))

    def record_push(self, observation) -> None:
        self.pushed.append(
            _Pushed(
                observation.metadata["observation_id"],
                round(observation.timestamp - self.origin, 9),
                self.now(),
            )
        )

    # ---- StreamingKernelAgent -----------------------------------------------------------------

    def bind_inbox(self, post) -> None:
        self.post = post

    def generate(self, generation, decision, observation, budget) -> PendingDecision:
        reply = _Reply(self.clock, generation)
        with self._lock:
            self._replies[generation] = reply
            self.requested(reply)
        self.started.append((generation, observation.metadata["observation_id"], self.now()))
        self.trigger()
        self.clock.wait(reply.ended, None)
        if not (reply.done or reply.aborted):
            raise ConnectionError(f"the session ended during reply {generation}")
        return PendingDecision(generation, decision, {"action": None, "latency": 0.0})

    def requested(self, reply: _Reply) -> None:
        """Under the lock, before the reply's trigger goes out."""

    def trigger(self) -> None:
        raise NotImplementedError

    def abort(self, generation=None) -> None:
        reply = self._replies[generation]
        reply.aborted = True
        self.aborted.append((generation, self.now()))
        try:
            self.interrupt(reply)
        finally:
            reply.ended.set()

    def interrupt(self, reply: _Reply) -> None:
        raise NotImplementedError

    def accept(self, generation) -> None:
        self.accepted.append(generation)

    def discard(self, generation) -> None:
        self.discarded.append(generation)

    def commit(self, outcome) -> None:
        self.committed.append((self.now(), outcome))

    def close(self) -> None:
        self.ws.close()
        self._reader.join(5.0)


def _image_url(observation) -> str:
    return "data:image/png;base64," + base64.b64encode(observation.screenshot).decode("ascii")


class ScriptedRealtimeAgent(_ScriptedAgent):
    """OpenAI Realtime's wire, as the driver's push_frame and request_reply write it."""

    def __init__(self, clock, port: int, origin: float = 0.0):
        self._acks: dict[str, threading.Event] = {}
        self._confirmed: set[str] = set()
        self._requested: list[_Reply] = []  # waiting for their response.created
        self._by_response: dict[str, _Reply] = {}
        super().__init__(clock, f"ws://127.0.0.1:{port}/v1/realtime?model=gpt-realtime-2", origin)
        self.send({"type": "session.update", "session": {"type": "realtime"}})

    def waiters(self) -> list:
        return list(self._acks.values())

    def received(self, message: dict) -> None:
        kind = message["type"]
        with self._lock:
            if kind in ("conversation.item.created", "conversation.item.done"):
                item = message["item"]["id"]
                self._confirmed.add(item)
                if item in self._acks:
                    self._acks[item].set()
            elif kind == "response.created":
                reply = self._requested.pop(0)
                reply.response_id = message["response"]["id"]
                self._by_response[reply.response_id] = reply
            elif kind == "response.output_text.done":
                reply = self._by_response[message["response_id"]]
                fields = json.loads(message["text"])
                action = Action(ActionType(fields.pop("action")), fields)
                frame = action.parameters.pop("frame", None)
                self.arrived(reply.generation, action, message["item_id"], frame)
            elif kind == "response.done":
                reply = self._by_response[message["response"]["id"]]
                reply.done = True
                reply.ended.set()

    def push(self, observation) -> None:
        number = len(self.pushed) + 1
        item = new_id("user")
        with self._lock:
            acked = self._acks[item] = self.clock.event()
            if self._closed:
                acked.set()
        content = [
            {"type": "input_text", "text": f"FRAME {number}"},
            {"type": "input_image", "image_url": _image_url(observation)},
        ]
        self.send(
            {
                "type": "conversation.item.create",
                "event_id": new_id("push"),
                "item": {"id": item, "type": "message", "role": "user", "content": content},
            }
        )
        self.clock.wait(acked, None)
        if item not in self._confirmed:
            raise ConnectionError(f"the session ended before it confirmed frame {number}")
        self.record_push(observation)

    def requested(self, reply: _Reply) -> None:
        self._requested.append(reply)

    def trigger(self) -> None:
        self.send({"type": "response.create", "response": {"max_output_tokens": "inf"}})

    def interrupt(self, reply: _Reply) -> None:
        cancel = {"type": "response.cancel", "event_id": new_id("cancel")}
        if reply.response_id is not None:
            cancel["response_id"] = reply.response_id
        self.send(cancel)


def _blob(data: bytes, mime_type: str) -> dict:
    """A realtime_input blob as google-genai writes it: URL-safe base64."""
    return {"data": base64.urlsafe_b64encode(data).decode("ascii"), "mime_type": mime_type}


class ScriptedLiveAgent(_ScriptedAgent):
    """Gemini Live's wire, as the driver's native-streaming push_frame and request_reply write it,
    over plain WebSocket: the cassette's server takes no TLS."""

    SETUP: ClassVar[dict] = {
        "setup": {
            "model": "models/gemini-3.8-live",
            "realtimeInputConfig": {
                "activity_handling": "NO_INTERRUPTION",
                "automatic_activity_detection": {"disabled": True},
                "turn_coverage": "TURN_INCLUDES_ALL_INPUT",
            },
            "contextWindowCompression": {
                "trigger_tokens": 65536,
                "sliding_window": {"target_tokens": 32768},
            },
        }
    }

    def __init__(self, clock, port: int, origin: float = 0.0):
        self._reply: _Reply | None = None  # the reply in flight: Gemini Live names none
        self._set_up = clock.event()
        super().__init__(clock, f"ws://127.0.0.1:{port}{LIVE_PATH}", origin)
        self.send(self.SETUP)
        clock.wait(self._set_up, None)

    def waiters(self) -> list:
        return [self._set_up]

    def received(self, message: dict) -> None:
        with self._lock:
            if "setupComplete" in message:
                self._set_up.set()
            elif "toolCall" in message:
                calls = message["toolCall"]["functionCalls"]
                for call in calls:
                    args = dict(call["args"])
                    frame = args.pop("frame", None)
                    action = Action(ActionType(call["name"]), args)
                    self.arrived(self._reply.generation, action, call["id"], frame)
                responses = [
                    {"id": c["id"], "name": c["name"], "response": {"status": "queued_not_executed"}}
                    for c in calls
                ]
                self.send({"tool_response": {"functionResponses": responses}})
            elif message.get("serverContent", {}).get("turnComplete"):
                reply, self._reply = self._reply, None
                reply.done = True
                reply.ended.set()

    def push(self, observation) -> None:
        number = len(self.pushed) + 1
        self.send({"realtime_input": {"video": _blob(observation.screenshot, "image/png")}})
        self.send({"realtime_input": {"text": f"FRAME {number}"}})
        self.record_push(observation)

    def requested(self, reply: _Reply) -> None:
        self._reply = reply

    def trigger(self) -> None:
        self.send({"realtime_input": {"activityStart": {}}})
        self.send({"realtime_input": {"audio": _blob(SILENCE, "audio/pcm;rate=16000")}})
        self.send({"realtime_input": {"activityEnd": {}}})

    def interrupt(self, reply: _Reply) -> None:
        self.ws.close()


@dataclass
class Episode:
    termination: Termination
    env: FakeEnvironment
    agent: object
    trace_path: object
    records: list[dict]
    problems: list[str]  # the cassette server's: [] when the session matched it to the end
    logs: list[str]

    def kernel(self, kind: str, *fields) -> list:
        """The data of each kernel record of ``kind``, or the tuple of its ``fields``."""
        data = [
            r["payload"]["data"]
            for r in self.records
            if r["event"] == "KernelRecord" and r["payload"]["kind"] == kind
        ]
        return [tuple(d[f] for f in fields) for d in data] if fields else data


def run_e2e(agent_factory, cassette, tmp_path, *, duration_s: float, normalizer=None) -> Episode:
    """One streaming episode on a fresh fake clock: the cassette's server, the agent the factory
    builds against it, the replay's wall-clock environment at 1 fps and the shell. Then the agent
    closes, and the server reports what its cursor found."""
    clock = FakeClock()
    frames = FrameSource(ReplayTask(), text_observation=False, fps=1.0)
    # A snapshot that takes time: the reply the deadline gives up ends before the episode
    # finishes, so its result is discarded rather than racing the Finish (kernel/core.py).
    env = FakeEnvironment(
        frames, clock, fps=1.0, max_minutes=duration_s / 60, reply_latencies=[],
        snapshot_seconds=0.5,
    )  # fmt: skip
    path = tmp_path / "episode_trace.jsonl"
    trace, logs = EpisodeTrace(path, run_id="e2e", clock=clock), []
    server = CassetteServer(cassette, clock, normalizer=normalizer or RealtimeNormalizer())
    with server, real_time_limit(REAL_LIMIT_S, server.drop), clock.participant():
        agent = agent_factory(clock, server.port, env.t0)
        try:
            termination = run_streaming_episode(
                agent, env, StreamingConfig(duration_s=duration_s),
                clocks=Clocks(clock, clock), log=logs.append, trace=trace,
            )  # fmt: skip
        finally:
            agent.close()
    trace.close()
    validate_episode_trace(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return Episode(termination, env, agent, path, records, server.finish(), logs)
