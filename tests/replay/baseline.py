"""Replay baselines: normalization, JSONL storage and field-level comparison.

Only non-semantic differences are normalized. Key order never counts. Random ids are replaced by
placeholders numbered in order of first appearance, one mapping for the whole session, so equal
ids stay equal. Fields that hold real elapsed time or real clock readings keep their key and
their null-ness but not their value. Everything else, model-visible times included, is compared
exactly. A trajectory row's input_prompt, a text summary of a whole request, is compared by the
SHA-256 and length of its normalized text.

Images are compared by their decoded pixels rather than stored: the same pixels encode to
different PNG bytes under different zlib builds (Pillow's macOS wheels bundle zlib-ng, the Linux
ones zlib). Where a record holds only the SHA-256 of an image's bytes, that image must have been
registered with image_digest where the test produced it (frames.py, build.py).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from pathlib import Path

from PIL import Image

from realtime_framework.entity.actions import Action

REAL_TIME = "<real time>"
REAL_TIME_FIELDS = frozenset(
    {
        # DefaultHarness.stream_generate (harnesses/default/harness.py:1168, 1500, 1515, 1569-1573)
        "latency",
        "decision_wall_latency_s",
        "provider_call_wall_latency_s",
        "runtime_turn_wall_latency_s",
        "provider_ttft_s",
        # the output collector's receive stamps (harnesses/default/harness.py:864-865)
        "received_wall_time_ns",
        "received_monotonic_ns",
        # RealtimeRuntime around the provider call (harnesses/default/context.py:516-528)
        "provider_call_started_monotonic_ns",
        "provider_call_started_wall_time_ns",
        "provider_stream_started_monotonic_ns",
        # ToolRuntime.inspect (sandbox/observation/som.py:677, 713, 728)
        "tool_execution_latency_s",
        # ChoiceDecision's candidate splits, question calls and text composition
        # (providers/decisions/selection.py:198, 216, 413)
        "latency_s",
    }
)
RANDOM_IDS = (
    ("generation", re.compile(r"turn-\d+-[0-9a-f]{12}")),  # uuid4: DefaultHarness._generate
    ("inspection", re.compile(r"inspect-\d{6}-[0-9a-f]{8}")),  # uuid4: som.py _next_call_id
    ("stub", re.compile(r"(?:http|ws)://127\.0\.0\.1:\d+")),  # the stub's OS-assigned port
    ("session", re.compile(r"rtb-[0-9a-f]{32}")),  # uuid4 per driver: LiteLLMDriver.__init__
    # OpenAI Realtime client ids "rtb_<kind>_" + uuid4 hex (providers/openai_realtime.py
    # _new_client_id), of the kinds the replayed sessions use (openai_realtime_session.py _new_id)
    ("rtb_user", re.compile(r"rtb_user_[0-9a-f]+")),
    ("rtb_assistant", re.compile(r"rtb_assistant_[0-9a-f]+")),
    ("rtb_ledger", re.compile(r"rtb_ledger_[0-9a-f]+")),
    ("rtb_feedback", re.compile(r"rtb_feedback_[0-9a-f]+")),
    ("rtb_create", re.compile(r"rtb_create_[0-9a-f]+")),
    ("rtb_delete", re.compile(r"rtb_delete_[0-9a-f]+")),
    # output item ids of the Realtime stub (tests/replay/ws_openai_realtime.py _respond)
    ("server_item", re.compile(r"item_srv_\d+")),
)
DATA_URL = re.compile(r"data:([^;,]+);base64,(.*)", re.DOTALL)  # providers/litellm.py:348
# The trajectory's summary of the retained history (harnesses/default/harness.py:1557-1559).
DIGESTED_FIELDS = frozenset({"input_prompt"})
# Fields that hold the SHA-256 of an image's bytes.
IMAGE_SHA256_FIELDS = frozenset(
    {
        # the screenshot (sandbox/observation/capture.py:285, 301), as the step records copy it
        # and the runner traces it (evaluation/episode_records.py observation_captured)
        "screenshot_sha256",
        # the SoM-marked frame (sandbox/observation/som.py:448, 725)
        "marked_image_sha256",
    }
)

# Each registered image's pixel digest, by the SHA-256 of its encoded bytes.
_IMAGES: dict[str, str] = {}


def image_digest(data: bytes) -> str:
    """The pixel digest "rgba:<width>x<height>:<SHA-256 of the RGBA pixels>" of an encoded image,
    which this also registers under the SHA-256 of its bytes."""
    key = hashlib.sha256(data).hexdigest()
    if key not in _IMAGES:
        with Image.open(io.BytesIO(data)) as image:
            pixels = image.convert("RGBA")
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        _IMAGES[key] = f"rgba:{pixels.width}x{pixels.height}:{digest}"
    return _IMAGES[key]


def registered_image(sha256: str) -> str:
    """The pixel digest of the registered image whose bytes have this SHA-256."""
    if sha256 not in _IMAGES:
        raise LookupError(f"no registered image has bytes with SHA-256 {sha256}")
    return _IMAGES[sha256]


def action_record(action: Action) -> dict:
    """The fields run_stream records for an action (episode_records.py action_record)."""
    return {
        "action_type": action.action_type.value,
        "parameters": action.parameters,
        "coordinate_space": action.coordinate_space,
        "coordinate_transform": action.coordinate_transform,
    }


class Normalizer:
    """Normalize records of one session; random ids keep one mapping across all of them."""

    def __init__(self, real_time_fields: frozenset[str] = REAL_TIME_FIELDS):
        self._real_time_fields = real_time_fields
        self._ids: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def __call__(self, value, key: str | None = None):
        if key in self._real_time_fields and value is not None:
            return REAL_TIME
        if key in DIGESTED_FIELDS and isinstance(value, str):
            text = self._string(value)
            return {"length": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}
        if key in IMAGE_SHA256_FIELDS:
            return registered_image(value)
        if isinstance(value, Action):
            value = action_record(value)
        if isinstance(value, dict):
            # The image bytes of a canonical request record (entity/request.py:42, 146), as
            # entity/records.py:61 describes bytes.
            if value.keys() == {"byte_length", "sha256"}:
                return registered_image(value["sha256"])
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
            return f"data:{image.group(1)};{image_digest(base64.b64decode(image.group(2)))}"
        for kind, pattern in RANDOM_IDS:
            value = pattern.sub(lambda match, kind=kind: self._placeholder(kind, match[0]), value)
        return value

    def _placeholder(self, kind: str, raw: str) -> str:
        if raw not in self._ids:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            self._ids[raw] = f"<{kind}-{self._counts[kind]}>"
        return self._ids[raw]


def trajectory(run_dir: Path) -> list[dict]:
    """The agent's trajectory rows, one record each in the order written
    (harnesses/default/harness.py:163-223). The viewer and the reports read them
    (evaluation/harbor_viewer.py:130, evaluation/reporting.py:1176)."""
    lines = (run_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    return [{"record": "trajectory", "row": json.loads(line)} for line in lines]


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
