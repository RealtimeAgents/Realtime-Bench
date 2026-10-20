"""JSONL storage of Choice Decisions baselines; normalization and comparison are baseline.py's.

Every question attempt resends the whole state, and one decision can ask many questions. So each
state entry (Luna: the transcript's strings and image parts, providers/openai_decisions.py:132-163;
JEV: the history messages, harnesses/jev/selection.py:223-227), each question and each criteria
order is stored once, on a "blob" line before its first use, and referenced as {"$blob": id}. The
decision metadata repeats the questions (jev_calls and jev_question, harnesses/jev/driver.py:90-103)
and references the same blobs, in the step's result and in its trajectory row. This is storage
only: read() restores the records before comparison.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _dumps(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


class _Blobs:
    def __init__(self, lines: list[str]):
        self._lines = lines
        self._stored: set[str] = set()

    def ref(self, value) -> dict:
        text = _dumps(value)
        blob_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        if blob_id not in self._stored:
            self._stored.add(blob_id)
            self._lines.append(_dumps({"record": "blob", "id": blob_id, "value": value}))
        return {"$blob": blob_id}


def _pack_request(request: dict, blobs: _Blobs) -> dict:
    body = request["body"]
    state = body["state"]
    if isinstance(state, list):  # Luna: the canonical transcript array
        state = [blobs.ref(entry) for entry in state]
    else:  # JEV: {"history": [...], "text_composition": ...}
        state = {**state, "history": [blobs.ref(message) for message in state["history"]]}
    questions = {key: blobs.ref(question) for key, question in body["questions"].items()}
    return {
        **request,
        "body": {**body, "state": state, "questions": questions},
        "criteria_order": blobs.ref(request["criteria_order"]),
    }


def _pack_runtime(runtime: dict, blobs: _Blobs) -> dict:
    return {
        **runtime,
        "jev_calls": [
            {**call, "question": blobs.ref(call["question"])} for call in runtime["jev_calls"]
        ],
        "jev_question": blobs.ref(runtime["jev_question"]),
    }


def _pack_step(record: dict, blobs: _Blobs) -> dict:
    record = {**record, "requests": [_pack_request(r, blobs) for r in record["requests"]]}
    runtime = record["result"].get("realtime_runtime")
    if runtime is not None:  # a failed decision returns no runtime metadata (harness.py:1317-1329)
        record["result"] = {**record["result"], "realtime_runtime": _pack_runtime(runtime, blobs)}
    return record


def _pack_trajectory(record: dict, blobs: _Blobs) -> dict:
    """A trajectory row repeats its decision's runtime metadata (harness.py:1489)."""
    runtime = record["row"].get("realtime_runtime")
    if runtime is None:  # nor does a failed decision's row (harness.py:1294-1312)
        return record
    return {**record, "row": {**record["row"], "realtime_runtime": _pack_runtime(runtime, blobs)}}


def write(path: Path, records: list[dict]) -> None:
    lines: list[str] = []
    blobs = _Blobs(lines)
    for record in records:
        if record["record"] == "step":
            record = _pack_step(record, blobs)
        elif record["record"] == "trajectory":
            record = _pack_trajectory(record, blobs)
        lines.append(_dumps(record))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read(path: Path) -> list[dict]:
    blobs: dict[str, object] = {}
    records = []

    def expand(value):
        if isinstance(value, dict):
            if value.keys() == {"$blob"}:
                return blobs[value["$blob"]]
            return {name: expand(item) for name, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value["record"] == "blob":
            blobs[value["id"]] = value["value"]
        else:
            records.append(expand(value))
    return records
