"""WebSocket cassettes for the native streaming tests: the recording format and its strict cursor.

A cassette holds one live session's messages, both ways, in the order the client saw them. It
verifies our side only, such as frames sent, cancels, reconnects and late actions; it does not
reproduce the server's state or the model's output (phase2.md, native streaming). One JSON object
per line:

    {"conn": 1, "dir": "open" | "c2s" | "s2c" | "close", "at": 1.25 | null, "msg": ...}

``conn`` numbers the connections in the order the client opened them. ``at`` is semantic seconds
since the episode origin. ``msg`` is a message as JSON, normalized like the replay baselines
(tests/replay/baseline.py): an image becomes its pixel digest and each of our random ids a
placeholder numbered by first appearance. An open has no message; a close is
``{"by": "client" | "server", "code": int}``, with the ``reason`` a server close sends.

The cursor replays a cassette strictly, each connection's records in order. A server record is
released once the client has done everything recorded before it on that connection: at once when
its ``at`` is null (a reactive message), otherwise once the clock reads ``at`` (a timed message).
It is sent with this session's ids in place of the placeholders. What the client does next is
matched against the next client record: a message must equal it once normalized, and the client
must do it when the clock reads the record's ``at``. A null ``at`` there is not compared.
A connection whose recording ends without a close may be closed by the client any time after its
last record: the episode trace, which recordings are converted from, has no closes.

Cassettes come two ways (stream-plan 4.19). A hand-written script runs on the fake clock, where
our side's times are as exact as the script's, so it gives every client record its time. A
recording converted from a real run by from_trace compares the order only: the client times of a
real run include the time our side took to compute, which passes for nothing on the fake clock.
So when the client does what such a record says before the server messages recorded ahead of it
are due, those are released then: the client went ahead, it did not send early.

The first difference is a problem, and after it nothing more is released:
- a message sent early: the next record is a server message that is not due yet, and the client
  record after it has a time;
- a message that differs from the recording, or one beyond it, or one sent at another time;
- a connection the recording does not have, or a close where it has more records;
- at the end, records left unused.

To turn a smoke test's run into a recording:

    .venv/bin/python -m tests.streaming.ws_cassette <run dir> <cassette.jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path

from realtime_framework.evaluation.agent_trace import load_body
from tests.replay import baseline
from tests.replay.ws_gemini_live import WireNormalizer

DIRECTIONS = ("open", "c2s", "s2c", "close")


class RealtimeNormalizer(baseline.Normalizer):
    """baseline.Normalizer, and two of our OpenAI Realtime ids the replay baselines never meet: the
    event ids of a pushed frame (providers/openai_realtime_session.py:290) and of a cancel
    (providers/openai_realtime.py:548)."""

    IDS = (
        ("rtb_push", re.compile(r"rtb_push_[0-9a-f]+")),
        ("rtb_cancel", re.compile(r"rtb_cancel_[0-9a-f]+")),
    )

    def _string(self, value: str) -> str:
        value = super()._string(value)
        for kind, pattern in self.IDS:
            value = pattern.sub(lambda match, kind=kind: self._placeholder(kind, match[0]), value)
        return value


# The normalizer of each live channel's messages: the drivers' trace channels
# (providers/openai_realtime.py:188-190, providers/gemini/live.py:271).
NORMALIZERS = {"openai_realtime.ws": RealtimeNormalizer, "gemini_live.ws": WireNormalizer}
_PLACEHOLDER = re.compile(r"<[a-z_]+-\d+>")
# Times on the fake clock agree to within this: a wait until ``at`` ends at ``now + (at - now)``,
# which can round one ulp short of ``at``.
_SLACK_S = 1e-9


def write(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        assert record.keys() == {"conn", "dir", "at", "msg"}, record
        assert record["dir"] in DIRECTIONS, record
    return records


def from_trace(run_dir: Path) -> list[dict]:
    """The cassette of the live session in ``run_dir``'s episode trace.

    The agent trace records each message a driver sends just before it writes it, and each one it
    receives when it reads it, with the connection that carried it (providers/capture.py:266-370),
    in the order they were captured (evaluation/agent_trace.py:179-196); the bodies are in the
    blob pack. A server message's ``at`` is its capture's wall time less the episode origin the
    runner records in ActiveLoopStarted (evaluation/runner.py:1407-1419), which on the wall clock
    is semantic time; the client's records have none, so only their order is compared. A retried
    run appends attempts; the last one with that record is taken.

    The trace records no handshake (providers/capture.py:8-9) and no close: the recording
    connection captures handshake failures, sends and receives only (capture.py:303-331). So a
    connection opens before its first message, and the cassette has no closes. The failed
    handshakes and sends the trace does record have no cassette form yet and raise.
    """
    events = [
        json.loads(line)
        for line in (run_dir / "episode_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    started = [event for event in events if event["event"] == "ActiveLoopStarted"][-1]
    if started["payload"]["clock_mode"] != "wall":
        raise ValueError("capture times are semantic time on the wall clock only")
    origin_ns = round(started["payload"]["start_unix_s"] * 1e9)
    messages = [
        event["payload"]
        for event in events
        if event["event"] == "SessionMessage" and event["attempt_id"] == started["attempt_id"]
    ]
    (channel,) = {message["channel"] for message in messages}
    normalize = NORMALIZERS[channel]()
    numbers: dict[str, int] = {}
    records = []
    for message in messages:
        if message["direction"] == "lifecycle":
            raise ValueError(f"no cassette form for a {message['message_type']} record")
        conn = numbers.get(message["connection_id"])
        if conn is None:
            conn = numbers[message["connection_id"]] = len(numbers) + 1
            records.append({"conn": conn, "dir": "open", "at": None, "msg": None})
        body = json.loads(load_body(run_dir / "episode_blobs.pack", message["body"]))
        if message["direction"] == "client":
            direction, at = "c2s", None
        else:
            direction = "s2c"
            at = round((message["captured_wall_time_ns"] - origin_ns) / 1e9, 6)
        records.append({"conn": conn, "dir": direction, "at": at, "msg": normalize(body)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn a run's live session into a cassette.")
    parser.add_argument("run_dir", type=Path, help="holds episode_trace.jsonl, episode_blobs.pack")
    parser.add_argument("cassette", type=Path)
    args = parser.parse_args()
    write(args.cassette, from_trace(args.run_dir))


def _server_side(record: dict) -> bool:
    return record["dir"] == "s2c" or (record["dir"] == "close" and record["msg"]["by"] == "server")


def _label(direction: str, message) -> str:
    if direction == "open":
        return "open"
    if direction == "close":
        return f"{message['by']} close {message['code']}"
    # OpenAI Realtime names each event in "type"; a Gemini Live message is its one top-level key.
    return f"{direction} {message.get('type') or next(iter(message), '')}"


class Cursor:
    """The strict cursor over one cassette. ``normalizer`` is this session's
    baseline.Normalizer (or a subclass): one mapping for every message the client sends, and the
    one the server's placeholders are restored from. Times are semantic seconds since the origin.
    """

    def __init__(self, records: list[dict], normalizer):
        self._pending: dict[int, deque] = {}
        for index, record in enumerate(records):
            self._pending.setdefault(record["conn"], deque()).append((index, record))
        self._normalize = normalizer
        self._opened = 0
        # Server records released ahead of their time, which ``release`` hands out first.
        self._ahead: dict[int, deque] = {}
        self.problems: list[str] = []

    @property
    def failed(self) -> bool:
        return bool(self.problems)

    # ---- what the client does ----------------------------------------------------------------

    def open(self, now: float) -> int | None:
        """The client opened its next connection; its number, or None when it is not recorded."""
        self._opened += 1
        conn = self._opened
        head = self._head(conn)
        if head is None or head[1]["dir"] != "open":
            self._problem(f"conn {conn}: opened, but the recording {self._next(conn)}")
            return None
        self._take(conn, now)
        return conn

    def receive(self, conn: int, message: dict, now: float) -> None:
        """A message the client sent. The server releases what is due before it reads one, so a
        server record still next was not due: the client sent this one early, unless the record
        it is matched against has no time."""
        actual = self._normalize(message)
        head = self._head(conn)
        if head is not None and _server_side(head[1]) and self._go_ahead(conn):
            head = self._head(conn)
        if head is not None and _server_side(head[1]):
            index, record = head
            self._problem(
                f"conn {conn} record {index}: {_label('c2s', actual)} sent at t={now:g}, before"
                f" the recorded {_label(record['dir'], record['msg'])} at t={record['at']:g}"
            )
            return
        if head is None or head[1]["dir"] != "c2s":
            self._problem(
                f"conn {conn}: {_label('c2s', actual)} sent, but the recording {self._next(conn)}"
            )
            return
        index, record = head
        differences = [
            "msg" + problem.removeprefix("record[0]")
            for problem in baseline.compare([record["msg"]], [actual])
        ]
        if differences:
            self._problem(
                f"conn {conn} record {index}: {_label('c2s', record['msg'])} differs:\n  "
                + "\n  ".join(differences[:20])
            )
            return
        self._take(conn, now)

    def client_closed(self, conn: int, code: int | None, now: float) -> None:
        """The client closed ``conn`` with ``code`` (None for a connection dropped without a
        close frame)."""
        head = self._head(conn)
        if head is None:
            return
        index, record = head
        if record["dir"] != "close" or record["msg"]["by"] != "client":
            self._problem(
                f"conn {conn}: the client closed it ({code}), but the recording {self._next(conn)}"
            )
        elif record["msg"]["code"] != code:
            self._problem(
                f"conn {conn} record {index}: the client closed it with {code}, recorded"
                f" {record['msg']['code']}"
            )
        else:
            self._take(conn, now)

    # ---- what the server does ----------------------------------------------------------------

    def release(self, conn: int, now: float) -> list[dict]:
        """The server records due on ``conn``, in order: messages with this session's ids, and a
        close as recorded."""
        if self.failed:
            return []
        released = list(self._ahead.pop(conn, ()))
        while True:
            head = self._head(conn)
            if head is None or not _server_side(head[1]):
                break
            record = head[1]
            if record["at"] is not None and record["at"] > now + _SLACK_S:
                break
            self._pending[conn].popleft()
            released.append(self._restored(record))
        return released

    def next_due(self, conn: int) -> float | None:
        """When the timed server record next on ``conn`` comes due; None when none is next."""
        head = self._head(conn)
        if head is None or self.failed or not _server_side(head[1]):
            return None
        return head[1]["at"]

    def finish(self) -> list[str]:
        """The problems, with the records left unused once the client is done."""
        if not self.failed:
            for conn, pending in sorted(self._pending.items()):
                if pending:
                    index, record = pending[0]
                    self.problems.append(
                        f"conn {conn}: {len(pending)} record(s) left unused, from record {index}"
                        f" ({_label(record['dir'], record['msg'])})"
                    )
        return self.problems

    # ---- internals ---------------------------------------------------------------------------

    def _head(self, conn: int):
        pending = self._pending.get(conn)
        return pending[0] if pending else None

    def _next(self, conn: int) -> str:
        head = self._head(conn)
        if head is None:
            return "has nothing more on it"
        index, record = head
        return f"has {_label(record['dir'], record['msg'])} next (record {index})"

    def _take(self, conn: int, now: float) -> None:
        index, record = self._pending[conn].popleft()
        at = record["at"]
        if at is not None and abs(now - at) > _SLACK_S:
            self._problem(
                f"conn {conn} record {index}: {_label(record['dir'], record['msg'])} at"
                f" t={now:g}, recorded at t={at:g}"
            )

    def _go_ahead(self, conn: int) -> bool:
        """Release the server records ahead of the next client record, when that has no time: the
        client went ahead of them in an order-only recording. False when it has one."""
        pending = self._pending[conn]
        following = next((record for _, record in pending if not _server_side(record)), None)
        if following is None or following["at"] is not None:
            return False
        ahead = self._ahead.setdefault(conn, deque())
        while _server_side(pending[0][1]):
            ahead.append(self._restored(pending.popleft()[1]))
        return True

    def _restored(self, record: dict) -> dict:
        """A server record as it is sent: a message with this session's ids."""
        if record["dir"] != "s2c":
            return record
        return {**record, "msg": self._restore(record["msg"])}

    def _problem(self, problem: str) -> None:
        if not self.failed:
            self.problems.append(problem)

    def _restore(self, message):
        """``message`` with each placeholder of an id the client sent replaced by that id. An id
        the server made up keeps its placeholder, which the client then echoes as it is."""
        raw = {placeholder: original for original, placeholder in self._normalize._ids.items()}

        def restore(value):
            if isinstance(value, dict):
                return {name: restore(item) for name, item in value.items()}
            if isinstance(value, list):
                return [restore(item) for item in value]
            if isinstance(value, str):
                return _PLACEHOLDER.sub(lambda match: raw.get(match[0], match[0]), value)
            return value

        return restore(message)


if __name__ == "__main__":
    main()
