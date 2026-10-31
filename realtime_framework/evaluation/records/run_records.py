"""A run directory's records: which file holds what, which rows of them go together, and how
strictly each reader takes them.

runner.main() leaves an episode's records side by side in its run directory; an episode's ATIF
export keeps copies under the same names (harbor_viewer.export_episode). A run retried in its
directory leaves them mixed: the episode trace and the decision lifecycle are appended to, each
trace record naming its attempt and no lifecycle record naming one, while the run summary, the
trajectory and the inspections are the last attempt's. The trace's rows are therefore taken by
attempt: a run summary's sections read their own attempt's, the recording timeline refuses a trace
several attempts wrote, and the export takes every attempt's.

Scoring reads the decision lifecycle strictly, a line that is not a whole object failing a run's
metrics and its report, and counts a trace only once its hash chain verifies against what the run
summary declares. Everything else reads leniently: a writer killed mid-line leaves an incomplete
final line, which is skipped, and a record the run never wrote reads as empty.
"""

import json
from pathlib import Path

from realtime_framework.entity.records import EPISODE_TRACE_SCHEMA
from realtime_framework.entity.records import TraceEvent
from realtime_framework.entity.records import validate_episode_trace

SUMMARY = "run_summary.json"
TRACE = "episode_trace.jsonl"
LIFECYCLE = "action_lifecycle.jsonl"
TRAJECTORY = "trajectory.jsonl"
INSPECTIONS = "tool_calls.jsonl"
ACTIVE_LOOP = "active-loop.json"
RECORDING_TIMING = "recording-timing.json"


def read_json(path) -> dict:
    """A JSON record, or {} when the run did not write it."""
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def read_rows(path, events=None) -> list:
    """A JSONL record's rows, those of ``events`` only when given; none without the file. A writer
    killed mid-line leaves an incomplete final line, which is skipped; an incomplete line with
    another after it fails."""
    if not Path(path).is_file():
        return []
    result = []
    with Path(path).open() as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if handle.read().strip():
                    raise
                break
            if events is None or value.get("event") in events:
                result.append(value)
    return result


def lifecycle_records(path) -> list[dict]:
    """The decision lifecycle as scoring reads it, one object per line, every line whole. A run
    that made no decision wrote no lifecycle: the sandbox creates the file with the first record
    (sandbox/actions.py record_lifecycle)."""
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: lifecycle record is not an object")
            records.append(record)
    return records


def attempt_rows(trace_path, events, attempt_id: str) -> list[dict]:
    """The episode trace's rows of ``events`` that the attempt ``attempt_id`` wrote
    (EpisodeTrace.attempt_id)."""
    return [row for row in read_rows(trace_path, events) if row["attempt_id"] == attempt_id]


def sole_attempt_rows(trace_path, events) -> list[dict]:
    """The episode trace's rows of ``events``, ``TraceEvent.EPISODE_STARTED`` among them, when
    one attempt wrote it. A trace several attempts wrote fails: nothing in the presentation
    tells which of them the recording shows."""
    trace = read_rows(trace_path, events)
    started = TraceEvent.EPISODE_STARTED.value
    attempts = {e.get("attempt_id") for e in trace if e["event"] == started}
    if len(attempts) > 1:
        raise ValueError("Multiple trace attempts require explicit recording-to-attempt selection")
    return trace


def trace_matches_summary(run_dir, declared) -> bool:
    """Whether the episode trace a run summary declares (``declared``, EpisodeTrace.describe) is
    a whole, scored episode's: a file of ``run_dir`` whose hash chain verifies, with the declared
    record count and head, that ends with EpisodeFinished after a frozen snapshot and a score."""
    if not isinstance(declared, dict) or declared.get("schema") != EPISODE_TRACE_SCHEMA:
        return False
    artifact = declared.get("artifact")
    if not isinstance(artifact, str) or not artifact or Path(artifact).name != artifact:
        return False
    try:
        verified = validate_episode_trace(Path(run_dir) / artifact)
    except Exception:
        return False
    required = {
        TraceEvent.SNAPSHOT_FROZEN.value,
        TraceEvent.SCORE_PRODUCED.value,
        TraceEvent.EPISODE_FINISHED.value,
    }
    return (
        verified["event_count"] == declared.get("event_count")
        and verified["head_event_sha256"] == declared.get("head_event_sha256")
        and verified["last_event"] == TraceEvent.EPISODE_FINISHED.value
        and required.issubset(verified["events"])
    )


def decision_of(row):
    """The decision a trajectory row records. The shell names it (shell.py _settle); the rows of
    a step-driven replay name only their tick, which is the decision's."""
    return row.get("decision", row.get("tick"))


def model_input_fps(record: dict):
    """The rate a run fed the model frames at, from its run summary or the identity its report
    expected. Runs recorded before the frame rates were split have none: they fed the model at the
    environment rate, ``fps``."""
    rate = record.get("model_input_fps")
    return record.get("fps") if rate is None else rate
