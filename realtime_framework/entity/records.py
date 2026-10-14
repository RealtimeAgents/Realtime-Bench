from __future__ import annotations

# Trace
"""Append-only episode audit trace.

The trace is the canonical record of what crossed the runtime boundaries.  It is deliberately
*not* a state store: the environment remains imperative, snapshots are read from that environment,
and scorers consume those frozen snapshots directly.  A hash chain makes accidental truncation,
reordering, or mutation detectable without introducing event reducers into the execution path.
"""

from dataclasses import asdict as asdict
from dataclasses import is_dataclass as is_dataclass
from enum import Enum as Enum
import hashlib as hashlib
import json as json
import math as math
from pathlib import Path as Path


EPISODE_TRACE_SCHEMA = "realtime_bench.episode_trace.v1"
GENESIS_HASH = "0" * 64


class TraceEvent(str, Enum):
    EPISODE_STARTED = "EpisodeStarted"
    OBSERVATION_CAPTURED = "ObservationCaptured"
    INFERENCE_STARTED = "InferenceStarted"
    OUTPUT_CHUNK_RECEIVED = "OutputChunkReceived"
    INFERENCE_COMPLETED = "InferenceCompleted"
    ACTION_PROPOSED = "ActionProposed"
    CLOCK_ADVANCED = "ClockAdvanced"
    ACTION_APPLIED = "ActionApplied"
    ACTION_RESOLVED = "ActionResolved"
    SNAPSHOT_FROZEN = "SnapshotFrozen"
    SCORE_PRODUCED = "ScoreProduced"
    RUN_INVALIDATED = "RunInvalidated"
    EPISODE_FINISHED = "EpisodeFinished"


def _jsonable(value):
    """Convert runtime values to a stable JSON surface without serializing binary observations."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"byte_length": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_episode_trace(path: str | Path) -> dict:
    """Validate an episode trace without opening it for append.

    Returns the verified chain head and a few structural facts used by resumability checks. The
    function deliberately does not reconstruct environment state; validation protects the audit
    artifact, not an event-sourced runtime.
    """
    path = Path(path)
    sequence = 0
    head_hash = GENESIS_HASH
    last_event = None
    events = set()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                raise ValueError(f"blank line in episode trace at line {line_number}")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid episode trace JSON at line {line_number}: {exc}"
                ) from exc
            expected_sequence = sequence + 1
            if record.get("schema") != EPISODE_TRACE_SCHEMA:
                raise ValueError(f"episode trace schema mismatch at line {line_number}")
            if record.get("sequence") != expected_sequence:
                raise ValueError(f"episode trace sequence mismatch at line {line_number}")
            if record.get("previous_event_sha256") != head_hash:
                raise ValueError(f"episode trace chain mismatch at line {line_number}")
            claimed = record.get("event_sha256")
            unhashed = dict(record)
            unhashed.pop("event_sha256", None)
            actual = canonical_sha256(unhashed)
            if claimed != actual:
                raise ValueError(f"episode trace digest mismatch at line {line_number}")
            sequence = expected_sequence
            head_hash = claimed
            last_event = record.get("event")
            events.add(last_event)
    return {
        "event_count": sequence,
        "head_event_sha256": head_hash,
        "last_event": last_event,
        "events": events,
    }
