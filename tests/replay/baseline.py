"""Replay baselines: normalization, JSONL storage and field-level comparison.

Only non-semantic differences are normalized. Key order never counts. Random ids are replaced by
placeholders numbered in order of first appearance, one mapping for the whole session, so equal
ids stay equal. Fields that hold real elapsed time or real clock readings keep their key and
their null-ness but not their value. Everything else, model-visible times included, is compared
exactly. Images are compared by the SHA-256 of their bytes rather than stored.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

from realtime_framework.entity.actions import Action

REAL_TIME = "<real time>"
REAL_TIME_FIELDS = frozenset(
    {
        # DefaultHarness.stream_generate (harnesses/default/harness.py:1125, 1432, 1447, 1493-1497)
        "latency",
        "decision_wall_latency_s",
        "provider_call_wall_latency_s",
        "runtime_turn_wall_latency_s",
        "provider_ttft_s",
        # the output collector's receive stamps (harnesses/default/harness.py:802-803)
        "received_wall_time_ns",
        "received_monotonic_ns",
        # RealtimeRuntime around the provider call (harnesses/default/context.py:477-489)
        "provider_call_started_monotonic_ns",
        "provider_call_started_wall_time_ns",
        "provider_stream_started_monotonic_ns",
        # ToolRuntime.inspect (sandbox/observation/som.py:677, 713, 728)
        "tool_execution_latency_s",
        # the retry backoff the harness sleeps for real (harnesses/default/harness.py:1248-1253,
        # 1270-1271), jittered by the uuid4 generation id (entity/model.py:332-336)
        "transport_reconnect_backoff_s",
    }
)
RANDOM_IDS = (
    ("generation", re.compile(r"turn-\d+-[0-9a-f]{12}")),  # uuid4: harness.py:1202
    ("inspection", re.compile(r"inspect-\d{6}-[0-9a-f]{8}")),  # uuid4: som.py:660
    ("stub", re.compile(r"http://127\.0\.0\.1:\d+")),  # the stub's OS-assigned port
    ("session", re.compile(r"rtb-[0-9a-f]{32}")),  # uuid4 per driver: providers/litellm.py:483
)
DATA_URL = re.compile(r"data:([^;,]+);base64,(.*)", re.DOTALL)  # harness.py:1050


def action_record(action: Action) -> dict:
    """The fields run_stream records for an action (runner.py:180-188)."""
    return {
        "action_type": action.action_type.value,
        "parameters": action.parameters,
        "coordinate_space": action.coordinate_space,
        "coordinate_transform": action.coordinate_transform,
    }


class Normalizer:
    """Normalize records of one session; random ids keep one mapping across all of them."""

    def __init__(self):
        self._ids: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def __call__(self, value, key: str | None = None):
        if key in REAL_TIME_FIELDS and value is not None:
            return REAL_TIME
        if isinstance(value, Action):
            value = action_record(value)
        if isinstance(value, dict):
            return {name: self(value[name], name) for name in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [self(item) for item in value]
        if isinstance(value, str):
            return self._string(value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        raise TypeError(f"no baseline form for {type(value).__name__} at {key!r}")

    def _string(self, value: str) -> str:
        image = DATA_URL.fullmatch(value)
        if image:
            digest = hashlib.sha256(base64.b64decode(image.group(2))).hexdigest()
            return f"data:{image.group(1)};sha256,{digest}"
        for kind, pattern in RANDOM_IDS:
            value = pattern.sub(lambda match, kind=kind: self._placeholder(kind, match[0]), value)
        return value

    def _placeholder(self, kind: str, raw: str) -> str:
        if raw not in self._ids:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            self._ids[raw] = f"<{kind}-{self._counts[kind]}>"
        return self._ids[raw]


def _dumps(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def write(path: Path, records: list[dict]) -> None:
    """One JSON object per line. A stateless request resends its whole history, so each distinct
    request message is stored once, on a "message" line before its first use, and requests list
    message ids. This is storage only: read() restores the full requests before comparison.
    """
    stored = set()
    lines = []
    for record in records:
        requests = []
        for request in record.get("requests", ()):
            ids = []
            for message in request["body"]["messages"]:
                text = _dumps(message)
                message_id = hashlib.sha256(text.encode()).hexdigest()[:16]
                if message_id not in stored:
                    stored.add(message_id)
                    lines.append(
                        _dumps({"record": "message", "id": message_id, "message": message})
                    )
                ids.append(message_id)
            requests.append({**request, "body": {**request["body"], "messages": ids}})
        lines.append(_dumps({**record, "requests": requests} if "requests" in record else record))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read(path: Path) -> list[dict]:
    messages = {}
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value["record"] == "message":
            messages[value["id"]] = value["message"]
            continue
        for request in value.get("requests", ()):
            request["body"]["messages"] = [messages[i] for i in request["body"]["messages"]]
        records.append(value)
    return records


def compare(expected: list[dict], actual: list[dict]) -> list[str]:
    """Field-level differences in order; baseline records left unused are a difference too."""
    problems = []
    for index, (old, new) in enumerate(zip(expected, actual)):
        problems += _diff(old, new, f"record[{index}]")
    if len(expected) > len(actual):
        problems.append(
            f"{len(expected) - len(actual)} baseline record(s) left unused,"
            f" from record[{len(actual)}]"
        )
    if len(actual) > len(expected):
        problems.append(
            f"{len(actual) - len(expected)} record(s) missing from the baseline,"
            f" from record[{len(expected)}]"
        )
    return problems


def _diff(old, new, path: str) -> list[str]:
    if type(old) is not type(new):
        return [f"{path}: {_show(old)} -> {_show(new)}"]
    if isinstance(old, dict):
        problems = []
        for name in sorted(old.keys() | new.keys()):
            if name not in new:
                problems.append(f"{path}.{name}: removed (was {_show(old[name])})")
            elif name not in old:
                problems.append(f"{path}.{name}: added {_show(new[name])}")
            else:
                problems += _diff(old[name], new[name], f"{path}.{name}")
        return problems
    if isinstance(old, list):
        problems = []
        for index, (a, b) in enumerate(zip(old, new)):
            problems += _diff(a, b, f"{path}[{index}]")
        if len(old) != len(new):
            problems.append(f"{path}: length {len(old)} -> {len(new)}")
        return problems
    if old == new:
        return []
    if isinstance(old, str) and len(old) + len(new) > 160:
        at = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
        start = max(0, at - 40)
        return [
            f"{path}: differs at char {at}: {old[start : at + 80]!r} -> {new[start : at + 80]!r}"
        ]
    return [f"{path}: {_show(old)} -> {_show(new)}"]


def _show(value) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= 160 else text[:157] + "..."
