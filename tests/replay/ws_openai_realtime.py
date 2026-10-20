"""Scripted OpenAI Realtime WebSocket server on 127.0.0.1, and the replay step loop for it.

The openai_realtime driver reaches the server through the OpenAI SDK's synchronous realtime
client (providers/openai_realtime.py:62-68, 264-275). With ``ModelConfig.api_base_url`` set to
``base_url``, the SDK connects to ``ws://127.0.0.1:<port>/v1/realtime?model=<model>``
(openai/resources/realtime/realtime.py:1151-1200). The server runs on the websockets library the
SDK uses, without a message size limit: client frames carry base64 screenshots.

Each connection keeps an ordered item list and answers the client events the driver sends:
- session.update gets session.updated; the driver never reads either (openai_realtime.py:264-275).
- conversation.item.create inserts the item at the end, at the start (``previous_item_id``
  "root") or right after its anchor, and gets conversation.item.added and
  conversation.item.done. The in-place compaction waits for ``.done`` (openai_realtime.py:245-246).
- conversation.item.delete removes the item and gets conversation.item.deleted (247-248).
- response.create gets the next scripted reply as one assistant text item. The usage always
  carries both token-detail objects, which the harness reads with ``.get``
  (harnesses/default/harness.py:118-124).
Acks carry only the item id and role: the driver reads nothing else from them (246, 248).

Before a create or delete is applied it is checked: a unique item id of at most 32 characters
(openai_realtime.py:48), an existing anchor, content parts the SDK allows for the role
(openai/types/realtime/realtime_conversation_item_{user,system,assistant}_message.py) minus
audio, since the session is text only (openai_realtime.py:268-275), and an existing delete
target. A failed check is a violation; the tests assert there are none.

Errors always echo the client event they answer. The driver treats an error without one as fatal
for the generation that reads it, and as a failure of any in-place edit it is waiting for
(openai_realtime.py:239-244, 357-371). After any error the stub sends nothing more on that
connection but keeps recording. The client sends a whole edit batch before it reads; acks it
never reads would fill its receive queue, stop its reader and stall its close handshake for the
close timeout (websockets/sync/connection.py:87-95, 811, 989-993).

Scripted faults:
- ``drop_responses``: answer these response.create calls (1-based, over the session) with
  response.created and then close the connection with 1011, before any output or usage.
- ``edit_error_on_delete``: answer the n-th delete that carries an event_id with an error.
  Only the in-place compaction sends deletes with an event_id (openai_realtime.py:555-591).
- ``reject_upgrades``: refuse the first n WebSocket upgrades with HTTP 503.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from dataclasses import dataclass, field

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from .build import Episode
from .steps import Step

CLIENT_ID_MAX_LENGTH = 32  # the driver's client id limit (providers/openai_realtime.py:48)
CONTENT_TYPES = {
    "user": {"input_text", "input_image"},
    "system": {"input_text"},
    "assistant": {"output_text"},
}
TEXT_DELTA_CHARS = 12  # visible text arrives in pieces of this size
IMAGE_TOKENS = 780  # input image tokens reported per request
SETTLE_TIMEOUT_S = 10.0  # longest wait for the stub's connection threads before a test fails


@dataclass(frozen=True)
class Reply:
    text: str
    usage: dict


@dataclass
class Connection:
    """One WebSocket upgrade: its handshake, the client events in arrival order, the
    conversation as the model would see it, and how the connection was closed."""

    number: int
    url: str
    header_names: list[str]
    authorization: str | None
    rejected: int | None = None
    events: list[dict] = field(default_factory=list)
    # For each event, the item index a create or delete took effect at (None otherwise).
    positions: list[int | None] = field(default_factory=list)
    # The conversation, one (item id, role, text, image sha256 or None) per item.
    items: list[tuple] = field(default_factory=list)
    muted: bool = False
    close: dict | None = None
    finished: bool = False

    def record(self) -> dict:
        return {
            "connection": self.number,
            "url": self.url,
            "header_names": self.header_names,
            "authorization": self.authorization,
            "rejected": self.rejected,
            "close": self.close,
        }


def replies(steps: list[Step]) -> list[Reply]:
    """The stub's reply for each step, with usage sized like scenario.replies. Output tokens
    include reasoning (entity/usage.py:55-58, 99-100). The SDK's usage model has no
    reasoning_tokens; the key survives its model_dump (providers/litellm.py:91-101), and the
    harness and the usage normalization read it (harness.py:124; entity/usage.py:66)."""
    out = []
    for k, step in enumerate(steps):
        reasoning_tokens = len(step.reasoning) // 4
        output_tokens = 4 + len(step.reply) // 4 + reasoning_tokens
        input_tokens = 1500 + 260 * k
        usage = {
            "total_tokens": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_token_details": {
                "text_tokens": input_tokens - IMAGE_TOKENS,
                "image_tokens": IMAGE_TOKENS,
                "audio_tokens": 0,
                "cached_tokens": 0 if k == 0 else 1280,
            },
            "output_token_details": {
                "text_tokens": output_tokens,
                "audio_tokens": 0,
                "reasoning_tokens": reasoning_tokens,
            },
        }
        out.append(Reply(step.reply, usage))
    return out


def _item_view(item: dict) -> tuple:
    text, image = None, None
    for part in item["content"]:
        if part["type"] == "input_image":
            data = part["image_url"].split(";base64,", 1)[1]
            image = hashlib.sha256(base64.b64decode(data)).hexdigest()
        elif text is None:
            text = part["text"]
    return (item["id"], item["role"], text, image)


class RealtimeStub:
    def __init__(
        self,
        replies: list[Reply],
        *,
        drop_responses: tuple[int, ...] = (),
        edit_error_on_delete: int | None = None,
        reject_upgrades: int = 0,
    ):
        self._replies = list(replies)
        self._drop_responses = set(drop_responses)
        self._edit_error_on_delete = edit_error_on_delete
        self._reject_upgrades = reject_upgrades
        self.connections: list[Connection] = []
        # One entry per response.create: what the conversation held and what the stub produced.
        self.responses: list[dict] = []
        # The full (id, role, text, image) items behind each entry's "context".
        self.views: list[list[tuple]] = []
        self.violations: list[str] = []
        self.unscripted = 0
        self.edit_errors = 0
        self._deletes_with_event_id = 0
        self._server_items = 0
        self._server_events = 0
        self._lock = threading.Condition()
        self._websockets: dict[object, Connection] = {}
        self._server = serve(
            self._serve,
            "127.0.0.1",
            0,
            max_size=None,
            ping_interval=None,
            process_request=self._upgrade,
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.socket.getsockname()[1]}/v1"

    @property
    def unused_replies(self) -> int:
        return len(self._replies)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        for websocket in list(self._websockets):
            websocket.close()  # returns at once when the client already closed
        self._thread.join()
        with self._lock:
            self._lock.wait_for(
                lambda: all(c.finished for c in self.connections), timeout=SETTLE_TIMEOUT_S
            )

    # ---- what the tests read -------------------------------------------------------------

    def mark(self) -> tuple[dict[int, int], int]:
        with self._lock:
            return {c.number: len(c.events) for c in self.connections}, len(self.responses)

    def since(self, mark) -> tuple[list[dict], list[dict]]:
        """Client events per connection and response entries after ``mark``.

        Called after a decision. The newest connection has recorded everything the client sent
        on it by then: the stub records a frame before answering it, and the decision ended with
        the response to its last response.create. An earlier connection can still be draining
        frames the client sent before it gave up on it, so wait until it has closed.
        """
        counts, responses = mark
        with self._lock:
            settled = self._lock.wait_for(
                lambda: all(c.finished for c in self.connections[:-1]), timeout=SETTLE_TIMEOUT_S
            )
            assert settled, "an abandoned connection did not close"
            events = [
                {"connection": c.number, "events": c.events[counts.get(c.number, 0) :]}
                for c in self.connections
                if len(c.events) > counts.get(c.number, 0)
            ]
            return events, self.responses[responses:]

    def wait_for_session_update(self) -> None:
        """Each connect sends session.update and reads no reply (openai_realtime.py:264-275),
        so initialize() can return before the stub has recorded it."""
        with self._lock:
            recorded = self._lock.wait_for(
                lambda: any(e["type"] == "session.update" for e in self.connections[-1].events),
                timeout=SETTLE_TIMEOUT_S,
            )
        assert recorded, "no session.update arrived"

    def finish(self) -> list[dict]:
        """The connection records, once the client has closed every connection."""
        with self._lock:
            closed = self._lock.wait_for(
                lambda: all(c.finished for c in self.connections), timeout=SETTLE_TIMEOUT_S
            )
            assert closed, "a connection is still open"
            return [c.record() for c in self.connections]

    # ---- server side -----------------------------------------------------------------------

    def _upgrade(self, websocket, request):
        headers = request.headers
        with self._lock:
            connection = Connection(
                number=len(self.connections) + 1,
                url=f"ws://{headers['Host']}{request.path}",
                header_names=[name for name, _ in headers.raw_items()],
                authorization=headers.get("Authorization"),
            )
            self.connections.append(connection)
            if connection.number <= self._reject_upgrades:
                connection.rejected = 503
                connection.finished = True
                self._lock.notify_all()
                return websocket.respond(503, "stub unavailable\n")
            self._websockets[websocket] = connection
        return None

    def _serve(self, websocket):
        connection = self._websockets[websocket]
        try:
            with self._lock:
                created = self._event({"type": "session.created", "session": {}})
            websocket.send(json.dumps(created))
            while True:
                try:
                    frame = json.loads(websocket.recv())
                except ConnectionClosed as closed:
                    connection.close = {
                        "received": int(closed.rcvd.code) if closed.rcvd else None,
                        "sent": int(closed.sent.code) if closed.sent else None,
                    }
                    return
                with self._lock:
                    connection.events.append(frame)
                    connection.positions.append(None)
                    answers, close_code = (
                        ([], None) if connection.muted else self._answer(connection, frame)
                    )
                    self._lock.notify_all()
                for answer in answers:
                    websocket.send(json.dumps(answer))
                if close_code is not None:
                    websocket.close(close_code, "scripted close")
        finally:
            with self._lock:
                connection.finished = True
                self._lock.notify_all()

    def _event(self, body: dict) -> dict:
        self._server_events += 1
        return {"event_id": f"event_srv_{self._server_events:05d}", **body}

    def _error(self, connection: Connection, event_id: str, code: str, message: str) -> dict:
        connection.muted = True
        error = {
            "type": "invalid_request_error",
            "code": code,
            "message": message,
            "param": None,
            "event_id": event_id,
        }
        return self._event({"type": "error", "error": error})

    def _violation(self, connection: Connection, frame: dict, problem: str):
        self.violations.append(f"connection {connection.number}: {problem}")
        event_id = frame.get("event_id")
        if event_id is None:  # nothing waits for an answer to it
            return [], None
        return [self._error(connection, event_id, "stub_violation", problem)], None

    def _answer(self, connection: Connection, frame: dict):
        kind = frame.get("type")
        if kind == "session.update":
            return [self._event({"type": "session.updated", "session": frame["session"]})], None
        if kind == "conversation.item.create":
            return self._create(connection, frame)
        if kind == "conversation.item.delete":
            return self._delete(connection, frame)
        if kind == "response.create":
            return self._respond(connection)
        return self._violation(connection, frame, f"unexpected client event {kind!r}")

    def _create(self, connection: Connection, frame: dict):
        item = frame["item"]
        anchor = frame.get("previous_item_id")
        ids = [existing[0] for existing in connection.items]
        item_id = item.get("id")
        allowed = CONTENT_TYPES.get(item.get("role"), set())
        if not isinstance(item_id, str) or not 0 < len(item_id) <= CLIENT_ID_MAX_LENGTH:
            return self._violation(connection, frame, f"bad item id {item_id!r}")
        if item_id in ids:
            return self._violation(connection, frame, f"duplicate item id {item_id}")
        if anchor not in (None, "root") and anchor not in ids:
            return self._violation(connection, frame, f"unknown anchor {anchor}")
        if not item.get("content") or any(p.get("type") not in allowed for p in item["content"]):
            return self._violation(connection, frame, f"bad content for {item.get('role')!r}")
        index = len(ids) if anchor is None else 0 if anchor == "root" else ids.index(anchor) + 1
        connection.items.insert(index, _item_view(item))
        connection.positions[-1] = index
        ack = {
            "previous_item_id": ids[index - 1] if index else None,
            "item": {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": item["role"],
                "status": "completed",
            },
        }
        return [
            self._event({"type": "conversation.item.added", **ack}),
            self._event({"type": "conversation.item.done", **ack}),
        ], None

    def _delete(self, connection: Connection, frame: dict):
        event_id = frame.get("event_id")
        if event_id is not None:
            self._deletes_with_event_id += 1
            if self._deletes_with_event_id == self._edit_error_on_delete:
                self.edit_errors += 1
                return [
                    self._error(connection, event_id, "stub_scripted_edit_error", "scripted")
                ], None
        ids = [existing[0] for existing in connection.items]
        if frame.get("item_id") not in ids:
            return self._violation(
                connection, frame, f"unknown delete target {frame.get('item_id')}"
            )
        index = ids.index(frame["item_id"])
        del connection.items[index]
        connection.positions[-1] = index
        return [
            self._event({"type": "conversation.item.deleted", "item_id": frame["item_id"]})
        ], None

    def _respond(self, connection: Connection):
        ordinal = len(self.responses) + 1
        entry = {"connection": connection.number, "context": [item[0] for item in connection.items]}
        self.responses.append(entry)
        self.views.append(list(connection.items))
        response_id = f"resp_srv_{ordinal:04d}"
        created = self._event(
            {
                "type": "response.created",
                "response": {
                    "object": "realtime.response",
                    "id": response_id,
                    "status": "in_progress",
                    "output": [],
                },
            }
        )
        if ordinal in self._drop_responses:
            entry["closed"] = 1011
            return [created], 1011
        if not self._replies:
            # Like the chat stub's HTTP 400: a FATAL failure ends the decision without a retry
            # (openai_realtime.py:97-105, 408-425; entity/model.py:318-339).
            self.unscripted += 1
            entry["status"] = "failed"
            failed = {
                "object": "realtime.response",
                "id": response_id,
                "status": "failed",
                "output": [],
                "status_details": {
                    "type": "failed",
                    "error": {"type": "invalid_request_error", "code": "response_beyond_script"},
                },
            }
            return [created, self._event({"type": "response.done", "response": failed})], None
        reply = self._replies.pop(0)
        self._server_items += 1
        item_id = f"item_srv_{self._server_items:04d}"
        entry.update(output_item_ids=[item_id], status="completed")
        head = {"response_id": response_id, "output_index": 0}
        part = {**head, "item_id": item_id, "content_index": 0}
        item = {"id": item_id, "object": "realtime.item", "type": "message", "role": "assistant"}
        done_item = {
            **item,
            "status": "completed",
            "content": [{"type": "output_text", "text": reply.text}],
        }
        pieces = [
            reply.text[i : i + TEXT_DELTA_CHARS]
            for i in range(0, len(reply.text), TEXT_DELTA_CHARS)
        ]
        stream = [
            {
                "type": "response.output_item.added",
                **head,
                "item": {**item, "status": "in_progress", "content": []},
            },
            {"type": "response.content_part.added", **part, "part": {"type": "text", "text": ""}},
            *({"type": "response.output_text.delta", **part, "delta": p} for p in pieces),
            {"type": "response.output_text.done", **part, "text": reply.text},
            {
                "type": "response.content_part.done",
                **part,
                "part": {"type": "text", "text": reply.text},
            },
            {"type": "response.output_item.done", **head, "item": done_item},
            {
                "type": "response.done",
                "response": {
                    "object": "realtime.response",
                    "id": response_id,
                    "status": "completed",
                    "output": [done_item],
                    "usage": reply.usage,
                },
            },
        ]
        connection.items.append(_item_view(done_item))
        return [created, *map(self._event, stream)], None


def run_steps(episode: Episode, steps: list[Step], stub: RealtimeStub) -> list[dict]:
    """steps.run_steps for a WebSocket session: each decision records the client events it
    caused on every connection and the stub's response entries, in place of request bodies."""
    agent = episode.agent
    records = []
    for k, step in enumerate(steps):
        observation = episode.frames.capture(k)
        mark = stub.mark()
        budget = episode.budget(observation.tick_number)  # t_k: runner.py:851
        agent.configure_decision(budget)
        agent.stream_prefill(observation)
        result = agent.stream_generate(tick=k)
        agent.finalize_decision_audit(result)  # consumes the trajectory hand-off entry
        agent.commit_action(step.status, step.detail)
        client_events, responses = stub.since(mark)
        records.append(
            {
                "record": "step",
                "step": k,
                "observation": {
                    "observation_id": observation.metadata["observation_id"],
                    "tick": observation.tick_number,
                    "timestamp": observation.timestamp,
                    "screenshot_sha256": observation.metadata["screenshot_sha256"],
                },
                "budget": {
                    "runtime_output_tokens": budget.runtime_output_tokens,
                    "provider_output_tokens": budget.provider_output_tokens,
                    "transport_output_tokens": budget.transport_output_tokens,
                },
                "client_events": client_events,
                "server_responses": responses,
                "result": result,
                "commit": {"status": step.status, "detail": step.detail},
                "action_history_entry": dict(agent.action_history[-1]),
                "token_usage": agent.token_usage,
            }
        )
    return records
