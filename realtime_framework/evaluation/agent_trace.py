"""The agent's write entry into the runner's episode trace.

The runner opens and closes ``episode_trace.jsonl``; the agent only borrows its write entry. Every
outbound request, raw reply, history change and live-session message is captured on the thread
that observed it, without I/O, hashing or serialization, and queued for one writer thread. That
thread is the only agent-side caller of ``EpisodeTrace.emit``, whose lock already serializes it
with the runner, so agent and runner records form one hash chain.

Every body (request, reply, live-session message, the canonical request a generation was prepared
from) goes to ``episode_blobs.pack`` next to the trace, never into the trace: the trace is
exported and uploaded, the pack is neither, and bodies carry private task content. Long base64
spans are stored verbatim and once per hash; the rest of a body (its template) is stored
zlib-compressed. Events carry only the locators, which the chain protects, and non-content
metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import traceback
import zlib
from pathlib import Path

from realtime_framework.clock import Clock
from realtime_framework.entity.records import TraceEvent, current_trace_scope
from realtime_framework.entity.request import CanonicalRequest
from realtime_framework.evaluation.kernel.events import AgentRecord

# Bound on writing the queued agent records at episode end, sized like ABANDONED_DRAIN_S
# (runner.py). A run whose agent records are incomplete is invalid.
AGENT_TRACE_FLUSH_S = 30.0
# Base64 spans at least this long are stored verbatim and once per hash.
SPAN_MIN_CHARS = 4096
# Data URLs (chat, Realtime and Decisions images), and the JSON "data" field of google-genai
# inlineData and of Anthropic image sources as httpx encodes them.
SPAN_PREFIXES = (b";base64,", b'"data":"', b'"data": "')
_STOP = object()


class BlobPack:
    """Append-only byte file. A record is located by ``{sha256, offset, length, encoding}``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        # A retried run appends after the earlier attempt's bytes, as EpisodeTrace resumes.
        self.bytes = os.fstat(self._fd).st_size
        self.records = 0
        self._stored: dict[str, dict] = {}

    def put(self, data: bytes, *, compress: bool = False) -> dict:
        digest = hashlib.sha256(data).hexdigest()
        stored = self._stored.get(digest)
        if stored is not None:
            return stored
        payload = zlib.compress(data, 1) if compress else data
        remaining = memoryview(payload)
        while remaining:
            written = os.write(self._fd, remaining)
            if written <= 0:
                raise OSError("blob pack append made no progress")
            remaining = remaining[written:]
        ref = {
            "sha256": digest,
            "offset": self.bytes,
            "length": len(payload),
            "encoding": "zlib" if compress else "identity",
        }
        self.bytes += len(payload)
        self.records += 1
        self._stored[digest] = ref
        return ref

    def fsync(self) -> None:
        os.fsync(self._fd)

    def close(self) -> None:
        os.close(self._fd)


def base64_spans(data: bytes) -> list[tuple[int, int]]:
    """Byte ranges of long base64 strings: each runs from a span prefix to the closing quote."""
    spans = []
    following = {prefix: data.find(prefix) for prefix in SPAN_PREFIXES}
    while True:
        found = [(index, prefix) for prefix, index in following.items() if index >= 0]
        if not found:
            return spans
        index, prefix = min(found)
        start = index + len(prefix)
        end = data.find(b'"', start)
        if end < 0:
            return spans
        # A quote escaped inside a JSON string does not end it; keep such text in the template so
        # the template stays valid JSON for session classification.
        if end - start >= SPAN_MIN_CHARS and data[end - 1] != ord("\\"):
            spans.append((start, end))
        for name, position in following.items():
            if 0 <= position <= end:
                following[name] = data.find(name, end + 1)


def load_body(pack_path: Path, body: dict) -> bytes:
    """Rebuild a recorded body from its event locator and check it against ``body.sha256``."""
    with Path(pack_path).open("rb") as pack:

        def read(ref: dict) -> bytes:
            pack.seek(ref["offset"])
            data = pack.read(ref["length"])
            return zlib.decompress(data) if ref["encoding"] == "zlib" else data

        template = read(body["template"])
        pieces, previous = [], 0
        for span in body.get("spans", ()):
            pieces += [template[previous : span["at"]], read(span["ref"])]
            previous = span["at"]
        pieces.append(template[previous:])
    data = b"".join(pieces)
    if hashlib.sha256(data).hexdigest() != body["sha256"]:
        raise ValueError("recorded body does not match its sha256")
    return data


def _json_bytes(value) -> bytes:
    """Compact JSON for a request dict, encoded piece by piece.

    One json.dumps call holds the GIL for the whole body (23 ms for an 8 MB chat request); per
    value, the longest hold is one screenshot string.
    """
    pieces = []

    def encode(item):
        if isinstance(item, dict):
            pieces.append(b"{")
            for index, (key, member) in enumerate(item.items()):
                separator = b"," if index else b""
                pieces.append(
                    separator + json.dumps(key, ensure_ascii=False).encode("utf-8") + b":"
                )
                encode(member)
            pieces.append(b"}")
        elif isinstance(item, (list, tuple)):
            pieces.append(b"[")
            for index, member in enumerate(item):
                if index:
                    pieces.append(b",")
                encode(member)
            pieces.append(b"]")
        else:
            pieces.append(json.dumps(item, ensure_ascii=False).encode("utf-8"))
            if isinstance(item, str) and len(item) >= SPAN_MIN_CHARS:
                # Yields the GIL; waits no time.
                time.sleep(0)  # noqa: TID251

    encode(value)
    return b"".join(pieces)


class AgentTraceSink:
    """Queue agent records on any thread; one writer thread emits them in capture order.

    While the kernel's shell runs an episode, the writer hands each record to the shell as an
    AgentRecord instead (``hand_over``): the shell gives the kernel one a step, the kernel numbers
    it into the episode log and the shell writes it, so agent and kernel records share one order.
    The blobs and the session links are still made here, before the record is handed over. A
    capture's times are read on ``clock``, the episode's operational clock."""

    def __init__(self, trace, pack_path: Path, *, clock: Clock):
        self.trace = trace
        self.clock = clock
        self.pack = BlobPack(pack_path)
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._captures = 0
        self._sealed = False
        self._dropped = 0
        self._written = 0
        self._error: str | None = None
        # The writer emits only while holding this lock; finish() takes it to fence the writer.
        self._emit_lock = threading.Lock()
        self._fenced = False
        self._post = None  # the shell's hand_over while records are handed over
        self._sessions: dict[str, dict] = {}
        self._writer = threading.Thread(target=self._run, name="agent-trace-writer", daemon=True)
        self._writer.start()

    @property
    def attempt_id(self) -> str:
        return self.trace.attempt_id

    def capture(self, event, payload, *, body=None, classify=None) -> int | None:
        """Queue one record; ``body`` is bytes, str, a list of byte chunks, a request dict or a
        canonical request, which the writer stores in its audit form (CanonicalRequest.to_record:
        images by length and SHA-256).

        ``classify`` marks a live-session message and names its acknowledgement keys.
        """
        wall_time_ns = self.clock.time_ns()
        monotonic_ns = self.clock.monotonic_ns()
        scope = current_trace_scope()
        with self._lock:
            if self._sealed:
                self._dropped += 1
                return None
            self._captures += 1
            sequence = self._captures
            self._queue.put(
                (sequence, event, payload, body, classify, wall_time_ns, monotonic_ns, scope)
            )
        return sequence

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            if self._error is not None:
                continue  # the run is already invalid; later records are counted, not written
            try:
                event, record = self._record(*item)
                with self._emit_lock:
                    if self._fenced:
                        return
                    if self._post is None:
                        self.trace.emit(event, record)
                    else:
                        scope = record["scope"] or {}
                        self._post(AgentRecord(event.value, record, scope.get("generation")))
                    self._written += 1
            except Exception:  # noqa: BLE001 - any write failure invalidates the run
                self._error = traceback.format_exc()

    def hand_over(self, post) -> None:
        """From the next record on, hand each to the shell with ``post`` instead of writing it."""
        with self._emit_lock:
            self._post = post

    def fail(self, error: str) -> None:
        """A record handed over could not be written: the run is invalid as the sink closes
        (close_agent_trace), and later records are counted, not written."""
        self._error = self._error or error

    def take_back(self, write_handed) -> None:
        """Write the records again. ``write_handed`` writes those already handed over first; the
        writer waits for it."""
        with self._emit_lock:
            self._post = None
            write_handed()

    def _record(self, sequence, event, payload, body, classify, wall_time_ns, monotonic_ns, scope):
        record = {
            "source": "agent",
            "capture_seq": sequence,
            "captured_wall_time_ns": wall_time_ns,
            "captured_monotonic_ns": monotonic_ns,
            "scope": scope,
            **payload,
        }
        template = None
        if body is not None:
            if isinstance(body, str):
                body = body.encode("utf-8")
            elif isinstance(body, list):
                body = b"".join(body)
            elif isinstance(body, dict):
                body = _json_bytes(body)
            elif isinstance(body, CanonicalRequest):
                body = _json_bytes(body.to_record())
            record["body"], template = self._store(body)
        if classify is not None:
            record.update(self._link(sequence, classify, record, template))
        return event, record

    def _store(self, data: bytes) -> tuple[dict, bytes]:
        """Store one body in the pack: each long base64 span once, then the template.

        Returns the locator for the event (no body content) and the template, from which a
        session message is classified.
        """
        described = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        pieces, spans, previous, length = [], [], 0, 0
        for start, end in base64_spans(data):
            pieces.append(data[previous:start])
            length += start - previous
            spans.append({"at": length, "ref": self.pack.put(data[start:end])})
            previous = end
            # Yields the GIL so the execution thread's loop runs between screenshots; waits no
            # time.
            time.sleep(0)  # noqa: TID251
        pieces.append(data[previous:])
        template = b"".join(pieces)
        described["template"] = self.pack.put(template, compress=True)
        if spans:
            described["spans"] = spans
        return described, template

    def _link(self, sequence: int, classify, record: dict, template: bytes | None) -> dict:
        """Name a live-session message and link server acknowledgements to client messages.

        Only messages the existing code sends or reads are seen, so no wait is added. The first
        server message that answers a client message is its acknowledgement; later ones repeat it.
        """
        state = self._sessions.setdefault(
            record["connection_id"],
            {
                "channel": record["channel"],
                "waiting": {},
                "answered": {},
                "client_sent": 0,
                "client_acked": set(),
                "ack_errors": 0,
                "client_unknown": 0,
                "server_messages": 0,
            },
        )
        direction = record["direction"]
        if direction == "lifecycle":
            if record.get("state") == "unknown":
                state["client_unknown"] += 1
            return {}
        message = json.loads(template) if template else {}
        message_type, keys, origin = classify(direction, message)
        fields = {"message_type": message_type, "keys": sorted(message) if message else []}
        if direction == "client":
            state["client_sent"] += 1
            for key in keys:
                state["waiting"].setdefault(key, []).append(sequence)
            return {**fields, "state": "sent"}
        state["server_messages"] += 1
        acknowledges, repeat = [], False
        for key in keys:
            waiting = state["waiting"].get(key)
            if waiting:
                acknowledged = waiting.pop(0)
                state["answered"][key] = acknowledged
                acknowledges.append(acknowledged)
            elif key in state["answered"]:
                acknowledges.append(state["answered"][key])
                repeat = True
        outcome = None
        if acknowledges:
            origin = "ack"
            outcome = "error" if message_type == "error" else "ok"
            if outcome == "error":
                state["ack_errors"] += 1
            elif not repeat:
                state["client_acked"].update(acknowledges)
        return {
            **fields,
            "acknowledges": acknowledges,
            "ack_outcome": outcome,
            "origin": origin,
            "repeat": repeat,
        }

    def finish(self, timeout: float) -> dict:
        """Seal, flush within ``timeout``, fsync the pack and fence the writer; return the close
        summary. After this call the writer never emits."""
        with self._lock:
            self._sealed = True
            self._queue.put(_STOP)
        self._writer.join(timeout)
        timed_out = self._writer.is_alive()
        with self._emit_lock:
            self._fenced = True
        error = self._error
        fsynced = False
        try:
            self.pack.fsync()
            fsynced = True
            if not timed_out:
                self.pack.close()
        except OSError:
            error = error or traceback.format_exc()
        with self._lock:
            dropped = self._dropped
        return {
            "captures": self._captures,
            "written": self._written,
            "dropped_after_seal": dropped,
            "pending_at_timeout": self._captures - self._written if timed_out else None,
            "error": error,
            "pack": {
                "artifact": self.pack.path.name,
                "bytes": self.pack.bytes,
                "records": self.pack.records,
                "fsynced": fsynced,
            },
            "sessions": [
                {
                    "connection_id": connection_id,
                    "channel": state["channel"],
                    "client_sent": state["client_sent"],
                    "client_acked": len(state["client_acked"]),
                    "ack_errors": state["ack_errors"],
                    "client_unknown": state["client_unknown"],
                    "server_messages": state["server_messages"],
                }
                for connection_id, state in list(self._sessions.items())
            ],
        }


def close_agent_trace(sink: AgentTraceSink, *, timeout=AGENT_TRACE_FLUSH_S) -> dict | None:
    """Finish the agent's records before the runner checkpoints and writes EpisodeFinished.

    Captures after this point are only counted. The runner thread writes AgentTraceClosed after
    the writer is fenced, so no agent record follows it. A writer error or a flush that does not
    finish within ``timeout`` invalidates the run: the invalid run is returned, None otherwise.
    """
    summary = sink.finish(timeout)
    try:
        sink.trace.emit(TraceEvent.AGENT_TRACE_CLOSED, summary)
    except Exception:  # noqa: BLE001 - the chain cannot record the close
        summary["error"] = summary["error"] or traceback.format_exc()
    if summary["error"]:
        return {
            "category": "harness",
            "reason": "agent_trace_write_failed",
            # The end of the traceback, which names the exception; the whole text is in
            # AgentTraceClosed.
            "detail": summary["error"][-2000:],
        }
    if summary["pending_at_timeout"] is not None:
        return {
            "category": "harness",
            "reason": "agent_trace_flush_timeout",
            "detail": (
                f"{summary['pending_at_timeout']} agent trace records unwritten after {timeout:g}s"
            ),
        }
    return None
