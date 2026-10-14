from __future__ import annotations

# Som
"""Observation-bound hybrid Set-of-Marks rendering and target resolution.

Discrete controls receive point marks (``P#``). Continuous surfaces receive region marks
(``R#``) whose local coordinates are normalized to 0..1000. The returned map is valid only for
the observation from which it was produced.
"""

import hashlib as hashlib
import io as io
import json as json
import math as math
import re as re
from typing import Any as Any

from PIL import Image as Image
from PIL import ImageDraw as ImageDraw
from PIL import ImageFont as ImageFont

from realtime_framework.entity.actions import Action
from realtime_framework.entity.actions import validate_action
from realtime_framework.entity.actions import SOM_REPRESENTATION_VERSION
from realtime_framework.entity.actions import action_has_target_references
from realtime_framework.entity.actions import parse_target_reference

_MAX_LEGEND_LABEL = 96
# Point marks whose box is this small (in marked-image pixels) get an outside outline and no dot.
SMALL_CONTROL_PX = 14


def _clean(value: Any, limit: int = _MAX_LEGEND_LABEL) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _area(box: list[int] | tuple[int, ...]) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _overlap_ratio(box: list[int], other: list[int]) -> float:
    area = _area(box)
    if area <= 0:
        return 0.0
    width = max(0, min(box[2], other[2]) - max(box[0], other[0]))
    height = max(0, min(box[3], other[3]) - max(box[1], other[1]))
    return width * height / area


def _active_items(result: dict) -> tuple[list[dict], list[dict], list[dict]]:
    windows = result.get("windows") if isinstance(result.get("windows"), list) else []
    active = next((window for window in windows if window.get("active")), None)
    if active is not None:
        items = active.get("interactables") if isinstance(active.get("interactables"), list) else []
        return (
            [item for item in items if item.get("availability") == "actionable_now"],
            [item for item in items if item.get("availability") == "visible_but_unavailable"],
            [item for item in items if item.get("availability") == "canvas_surfaces"],
        )
    return (
        list(result.get("actionable_now") or []),
        list(result.get("visible_but_unavailable") or []),
        list(result.get("canvas_surfaces") or []),
    )


def _region_actions(canvas: dict, actionable: list[dict]) -> list[str]:
    actions = set(canvas.get("actions") or [])
    box = canvas.get("bbox")
    if isinstance(box, list):
        for item in actionable:
            item_box = item.get("bbox")
            if (
                item.get("role") == "region"
                and isinstance(item_box, list)
                and _overlap_ratio(item_box, box) >= 0.9
            ):
                actions.update(item.get("actions") or [])
    if canvas.get("internal_target_coverage") in {"continuous_coordinate_space", "unsupported"}:
        actions.update(("drag", "scroll"))
    return sorted(actions)


def _is_canvas_wrapper(item: dict, canvases: list[dict]) -> bool:
    box = item.get("bbox")
    return bool(
        item.get("role") == "region"
        and isinstance(box, list)
        and any(
            isinstance(canvas.get("bbox"), list) and _overlap_ratio(box, canvas["bbox"]) >= 0.9
            for canvas in canvases
        )
    )


def _mark_map(
    result: dict, *, text_observation: bool = False
) -> tuple[dict[str, dict], list[str]]:
    actionable, unavailable, canvases = _active_items(result)
    regions = list(canvases)
    for item in actionable:
        box = item.get("bbox")
        actions = set(item.get("actions") or [])
        if (
            item.get("role") == "region"
            and isinstance(box, list)
            and actions.intersection({"drag", "scroll"})
            and not any(
                isinstance(region.get("bbox"), list) and _overlap_ratio(box, region["bbox"]) >= 0.9
                for region in regions
            )
        ):
            region = dict(item)
            region["internal_target_coverage"] = "continuous_coordinate_space"
            regions.append(region)
    marks: dict[str, dict] = {}
    lines: list[str] = []

    points = [item for item in actionable if not _is_canvas_wrapper(item, regions)]
    for index, item in enumerate(points, 1):
        point = item.get("click_point")
        box = item.get("bbox")
        if not (isinstance(point, list) and len(point) == 2 and isinstance(box, list)):
            continue
        mark = f"P{index}"
        marks[mark] = {
            "kind": "point",
            "point": [int(point[0]), int(point[1])],
            "bbox": [int(value) for value in box],
            "source_id": item.get("id"),
            "role": item.get("role"),
            "label": item.get("semantic_label") or item.get("label") or "",
            "actions": list(item.get("actions") or []),
            "availability": "actionable_now",
        }
        state = item.get("state") or {}
        selected = " selected" if state.get("selected") else ""
        # Include checkbox/radio state in the legend because the mark can obscure the control.
        checked = ""
        if isinstance(state.get("checked"), bool):
            checked = " checked" if state["checked"] else " unchecked"
        field = ""
        if text_observation:
            # Without pixels, field focus/value/content must be stated explicitly.
            marks[mark].update(
                focused=bool(state.get("focused")),
                value=str(state.get("value") or "")[:160],
                text_selected=bool(state.get("text_selected")),
                input_type=item.get("input_type", ""),
                input_constraints=item.get("input_constraints") or {},
                state=state,
                visible_content=item.get("visible_content") or {},
                appearance=item.get("appearance") or {},
                select_options=item.get("select_options") or [],
                select_multiple=bool(item.get("select_multiple")),
            )
            field = (" focused" if state.get("focused") else "") + (
                f" value={_json(_clean(state['value'], 80))}" if state.get("value") else ""
            )
        lines.append(
            f"{mark} {_clean(item.get('role') or 'control', 32)} "
            f"{_json(_clean(item.get('semantic_label') or item.get('label')))} "
            f"a={','.join(item.get('actions') or []) or 'none'}{checked}{selected}{field}"
        )
        details = {
            key: item[key]
            for key in ("visible_content", "appearance", "select_options")
            if text_observation and item.get(key)
        }
        if details:
            lines.append(f"{mark} VISIBLE " + _json(details))
        if text_observation and item.get("select_options"):
            lines.append(
                f"{mark} native option selected flags are committed DOM values; an open "
                "menu may have a different keyboard highlight until Enter commits it."
            )

    region_lines = []
    for index, item in enumerate(regions, 1):
        box = item.get("bbox")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        mark = f"R{index}"
        actions = _region_actions(item, actionable)
        entry = {
            "kind": "region",
            "bbox": [int(value) for value in box],
            "source_id": item.get("id"),
            "role": item.get("role") or "canvas",
            "label": item.get("semantic_label") or item.get("label") or "canvas",
            "actions": actions,
            "availability": "actionable_now",
            "coverage": item.get("internal_target_coverage") or "unsupported",
        }
        target_map = item.get("internal_target_map")
        grid_text = ""
        if isinstance(target_map, dict) and target_map.get("kind") == "grid":
            entry["grid"] = target_map
            origin = target_map.get("origin") or []
            step = target_map.get("step") or []
            width, height = max(1, box[2] - box[0]), max(1, box[3] - box[1])
            if len(origin) == 2 and len(step) == 2:
                local_origin = [
                    round((origin[0] - box[0]) * 1000 / width),
                    round((origin[1] - box[1]) * 1000 / height),
                ]
                # Rounded to 0.1: an integer step drifts by up to half a cell across 15 columns.
                local_step = [
                    round(step[0] * 1000 / width, 1),
                    round(step[1] * 1000 / height, 1),
                ]
                grid_text = (
                    f" grid_cols={_json(target_map.get('column_labels') or [])}"
                    f" grid_rows={_json(target_map.get('row_labels') or [])}"
                    f" grid_origin=@{local_origin[0]},{local_origin[1]}"
                    f" grid_step={local_step[0]},{local_step[1]}"
                )
        marks[mark] = entry
        region_lines.append(
            f"{mark} {_clean(entry['role'], 32)} {_json(_clean(entry['label']))} "
            f"a={','.join(actions) or 'none'} coverage={entry['coverage']}{grid_text}"
        )

    unavailable_lines = []
    for index, item in enumerate(unavailable, 1):
        unavailable_lines.append(
            f"U{index} {_clean(item.get('role') or 'control', 32)} "
            f"{_json(_clean(item.get('semantic_label') or item.get('label')))} "
            f"reason={','.join(item.get('unavailable_reasons') or []) or 'unavailable'}"
        )

    if region_lines:
        lines.append("REGIONS")
        lines.extend(region_lines)
    if unavailable_lines:
        lines.append("UNAVAILABLE (U marks are informational and cannot be targeted)")
        lines.extend(unavailable_lines)
    keys = (
        result.get("keyboard_actions") if isinstance(result.get("keyboard_actions"), list) else []
    )
    if keys:
        lines.append(
            "KEYS "
            + "; ".join(
                f"{_clean(item.get('key'), 32)}={_json(_clean(item.get('label')))}" for item in keys
            )
        )
    return marks, lines


def _to_image_point(
    point: list[int], sx: float, sy: float, size: tuple[int, int]
) -> tuple[int, int]:
    return (
        max(0, min(size[0] - 1, round(point[0] * sx))),
        max(0, min(size[1] - 1, round(point[1] * sy))),
    )


def _to_image_box(box: list[int], sx: float, sy: float, size: tuple[int, int]) -> list[int]:
    left, top = _to_image_point([box[0], box[1]], sx, sy, size)
    right = max(left + 1, min(size[0] - 1, round(box[2] * sx)))
    bottom = max(top + 1, min(size[1] - 1, round(box[3] * sy)))
    return [left, top, right, bottom]


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _badge_position(
    anchor: tuple[int, int],
    badge_size: tuple[int, int],
    image_size: tuple[int, int],
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    x, y = anchor
    width, height = badge_size
    candidates = (
        (x + 6, y - height - 6),
        (x + 6, y + 6),
        (x - width - 6, y - height - 6),
        (x - width - 6, y + 6),
    )
    best = None
    for rank, (left, top) in enumerate(candidates):
        left = max(0, min(image_size[0] - width, left))
        top = max(0, min(image_size[1] - height, top))
        box = (left, top, left + width, top + height)
        score = sum(_intersection_area(box, prior) for prior in occupied) * 100 + rank
        if best is None or score < best[0]:
            best = (score, box)
    return best[1]


def _render_marks(
    image_bytes: bytes,
    marks: dict[str, dict],
    unavailable: list[dict],
    *,
    action_width: int,
    action_height: int,
) -> tuple[bytes, str]:
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(14, min(24, round(min(image.size) / 27)))
    font = ImageFont.load_default(size=font_size)
    sx, sy = image.width / action_width, image.height / action_height
    occupied: list[tuple[int, int, int, int]] = []

    def badge(
        mark: str,
        anchor: tuple[int, int],
        color: tuple[int, int, int, int],
        keep_clear: tuple[int, int, int, int] | None = None,
    ):
        text_box = draw.textbbox((0, 0), mark, font=font, stroke_width=1)
        width = text_box[2] - text_box[0] + 8
        height = text_box[3] - text_box[1] + 6
        box = _badge_position(anchor, (width, height), image.size, occupied)
        occupied.append(box)
        centre = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        start = anchor
        if keep_clear is not None:
            # The leader line starts at the control's outer outline, not at its centre, so the
            # pixels of a small control stay visible.
            corners = (
                (keep_clear[0], keep_clear[1]),
                (keep_clear[2], keep_clear[1]),
                (keep_clear[0], keep_clear[3]),
                (keep_clear[2], keep_clear[3]),
            )
            start = min(corners, key=lambda c: (c[0] - centre[0]) ** 2 + (c[1] - centre[1]) ** 2)
        draw.line((start[0], start[1], centre[0], centre[1]), fill=color, width=2)
        draw.rounded_rectangle(box, radius=3, fill=color, outline=(255, 255, 255, 245), width=1)
        draw.text(
            (box[0] + 4, box[1] + 3),
            mark,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 230),
        )

    for mark, entry in marks.items():
        if entry["kind"] == "point":
            box = _to_image_box(entry["bbox"], sx, sy, image.size)
            point = _to_image_point(entry["point"], sx, sy, image.size)
            color = (230, 72, 46, 235)
            if min(box[2] - box[0], box[3] - box[1]) <= SMALL_CONTROL_PX:
                # A 20 px checkbox is 8 px in a 768x432 frame, where the 7 px dot, an inward 2 px
                # outline and the leader line would cover its check mark. Outline small controls
                # from the outside, skip the dot, and keep the badge and its leader off the control.
                outer = (
                    max(0, box[0] - 3),
                    max(0, box[1] - 3),
                    min(image.width - 1, box[2] + 3),
                    min(image.height - 1, box[3] + 3),
                )
                draw.rectangle(outer, outline=(230, 72, 46, 175), width=2)
                # Badge candidates start 6 px off the anchor, which already clears a small control;
                # reserving the outline as well would push badges onto the neighbouring control.
                badge(mark, point, color, keep_clear=outer)
                continue
            draw.rectangle(box, outline=(230, 72, 46, 175), width=2)
            draw.ellipse(
                (point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3),
                fill=(255, 255, 255, 255),
                outline=color,
                width=2,
            )
            badge(mark, point, color)
        else:
            box = _to_image_box(entry["bbox"], sx, sy, image.size)
            color = (0, 137, 178, 235)
            draw.rectangle(box, outline=color, width=3)
            # A sparse local-coordinate guide makes canvas/map positions expressible without
            # replacing the underlying pixels or turning the region into a discrete grid.
            for fraction in (0.25, 0.5, 0.75):
                x = round(box[0] + (box[2] - box[0]) * fraction)
                y = round(box[1] + (box[3] - box[1]) * fraction)
                draw.line((x, box[1], x, box[3]), fill=(0, 137, 178, 70), width=1)
                draw.line((box[0], y, box[2], y), fill=(0, 137, 178, 70), width=1)
            badge(mark, (box[0], box[1]), color)

    # Unavailable controls are outlined but deliberately have no resolvable target entry.
    for index, item in enumerate(unavailable, 1):
        box = item.get("bbox")
        if not isinstance(box, list):
            continue
        image_box = _to_image_box(box, sx, sy, image.size)
        point = item.get("click_point") or [box[0], box[1]]
        anchor = _to_image_point(point, sx, sy, image.size)
        draw.rectangle(image_box, outline=(105, 105, 105, 150), width=1)
        badge(f"U{index}", anchor, (105, 105, 105, 220))

    rendered = Image.alpha_composite(image, overlay).convert("RGB")
    output = io.BytesIO()
    rendered.save(output, format="PNG", optimize=True)
    return output.getvalue(), "png"


def build_hybrid_som(
    result: dict,
    image_bytes: bytes | None,
    *,
    action_width: int,
    action_height: int,
    text_observation: bool = False,
) -> dict:
    """Build the model-facing legend, mark map, and optional marked screenshot."""
    marks, item_lines = _mark_map(result, text_observation=text_observation)
    obs = result.get("observation") or {}
    binding = result.get("binding") or {}
    header = (
        f"HYBRID_SOM v1 observation={obs.get('observation_id', '')} "
        f"space={action_width}x{action_height} stable={str(bool(binding.get('stable'))).lower()}"
    )
    instructions = [
        header,
        "REF current-only: Pn=point; Rn@u,v=region-local 0..1000; raw coordinates allowed.",
    ]
    text = "\n".join(instructions + item_lines)
    unavailable = _active_items(result)[1]
    marked_bytes, mime = (None, None)
    if image_bytes:
        marked_bytes, mime = _render_marks(
            image_bytes,
            marks,
            unavailable,
            action_width=action_width,
            action_height=action_height,
        )
    mark_payload = json.dumps(marks, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "representation": SOM_REPRESENTATION_VERSION,
        "marks": marks,
        "text": text,
        "image_bytes": marked_bytes,
        "image_mime_subtype": mime,
        "marked_image_sha256": hashlib.sha256(marked_bytes).hexdigest() if marked_bytes else None,
        "mark_map_sha256": hashlib.sha256(mark_payload.encode("utf-8")).hexdigest(),
    }


def _resolve_reference(
    raw: object,
    marks: dict[str, dict],
    *,
    allow_region_center: bool = False,
) -> tuple[tuple[int, int] | None, dict]:
    reference = parse_target_reference(raw)
    if reference is None:
        return None, {"reference": raw, "error": "invalid_reference_syntax"}
    entry = marks.get(reference.mark)
    if not isinstance(entry, dict):
        return None, {"reference": raw, "mark": reference.mark, "error": "mark_not_in_observation"}
    if reference.kind == "P":
        if reference.u is not None or entry.get("kind") != "point":
            return None, {"reference": raw, "mark": reference.mark, "error": "mark_kind_mismatch"}
        point = entry.get("point")
    else:
        if entry.get("kind") != "region":
            return None, {"reference": raw, "mark": reference.mark, "error": "mark_kind_mismatch"}
        box = entry.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            return None, {"reference": raw, "mark": reference.mark, "error": "invalid_region"}
        if reference.u is None:
            if not allow_region_center:
                return None, {
                    "reference": raw,
                    "mark": reference.mark,
                    "error": "region_point_requires_local_coordinates",
                }
            u = v = 500
        else:
            u, v = reference.u, reference.v
        right = max(box[0], box[2] - 1)
        bottom = max(box[1], box[3] - 1)
        point = [
            max(box[0], min(right, round(box[0] + (box[2] - box[0]) * u / 1000))),
            max(box[1], min(bottom, round(box[1] + (box[3] - box[1]) * v / 1000))),
        ]
    if (
        not isinstance(point, list)
        or len(point) != 2
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
    ):
        return None, {"reference": raw, "mark": reference.mark, "error": "invalid_mark_geometry"}
    resolved = (round(point[0]), round(point[1]))
    return resolved, {
        "reference": raw,
        "mark": reference.mark,
        "kind": entry.get("kind"),
        "source_id": entry.get("source_id"),
        "resolved": list(resolved),
    }


def resolve_action_references(
    action: Action | None, marks: dict[str, dict]
) -> tuple[Action | None, dict]:
    """Resolve an action against one observation's immutable mark map."""
    if action is None:
        return None, {"mode": "invalid", "error": "no_action"}
    if not action_has_target_references(action):
        ok, reason = validate_action(action)
        return (action if ok else None), {
            "mode": "raw_coordinates",
            "valid": ok,
            "error": reason or None,
        }

    params = dict(action.parameters or {})
    resolutions = []
    if "target" in params:
        point, trace = _resolve_reference(
            params.pop("target"),
            marks,
            # A bare region target requests the region itself. Use its center just as ordinary
            # element targeting does; Rn@u,v remains available for precise points.
            allow_region_center=True,
        )
        resolutions.append(trace)
        if point is None:
            return None, {"mode": "som", "valid": False, "resolutions": resolutions}
        params.update({"x": point[0], "y": point[1]})
    if "start" in params:
        point, trace = _resolve_reference(params.pop("start"), marks)
        resolutions.append(trace)
        if point is None:
            return None, {"mode": "som", "valid": False, "resolutions": resolutions}
        params.update({"x1": point[0], "y1": point[1]})
    if "end" in params:
        point, trace = _resolve_reference(params.pop("end"), marks)
        resolutions.append(trace)
        if point is None:
            return None, {"mode": "som", "valid": False, "resolutions": resolutions}
        params.update({"x2": point[0], "y2": point[1]})
    resolved = Action(
        action.action_type,
        params,
        coordinate_space=action.coordinate_space,
        coordinate_transform=action.coordinate_transform,
    )
    ok, reason = validate_action(resolved)
    return (resolved if ok else None), {
        "mode": "som",
        "valid": ok,
        "error": reason or None,
        "resolutions": resolutions,
        "resolved_parameters": params if ok else None,
    }


# Protocol
"""Provider-neutral injection of observation-bound, non-mutating interactables."""

OBSERVATION_INTERACTABLES_HEADER = (
    "\n\nCURRENT-SCREEN GROUNDING (read-only; no inspection round):\n"
)


def without_observation_interactables(prompt: str) -> str:
    """Remove the appended SoM legend while keeping the frame prompt and execution feedback."""
    return prompt.partition(OBSERVATION_INTERACTABLES_HEADER)[0]


def observation_prompt_with_interactables(base_prompt: str, text: str) -> str:
    """Attach environment-provided controls to a visual observation.

    This is observation data, not a model-invoked tool turn: the harness calls the read-only
    runtime for each frame and asks the model for its GUI action once.
    """
    return (
        base_prompt
        + OBSERVATION_INTERACTABLES_HEADER
        + (text or "No browser interactables were available for this screenshot.")
        + "\nUse only these marks; otherwise use raw coordinates."
    )


def valid_inspection_proof(proof: dict | None, observation_id: str | None) -> bool:
    return bool(
        proof
        and proof.get("valid")
        and observation_id
        and proof.get("observation_id") == observation_id
        and proof.get("requested_observation_id") == observation_id
    )


# Runtime
"""Project observation-bound interactables and persist an independent audit trace."""

import threading as threading
import time as time
import uuid as uuid
from pathlib import Path as Path

from realtime_framework.sandbox.observation.interactables import project_result
from realtime_framework.sandbox.observation.interactables import result_digest


class ToolRuntime:
    """Thread-safe runtime for environment-provided ``inspect_interactables`` data."""

    def __init__(
        self,
        trace_path: str | Path | None = None,
        *,
        task_id: str | None = None,
        adapter: str | None = None,
        model: str | None = None,
        text_observation: bool = False,
    ):
        self.task_id = task_id
        self.adapter = adapter
        self.model = model
        # Text-observation models (no screenshot input) get control appearance and field state.
        self.text_observation = text_observation
        self._lock = threading.Lock()
        self._sequence = 0
        self._pending = {}
        self._trace = None
        if trace_path is not None:
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._trace = path.open("w", encoding="utf-8")

    def close(self):
        with self._lock:
            if self._trace is not None:
                for event in self._pending.values():
                    event["incomplete"] = True
                    self._trace.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                self._pending.clear()
                self._trace.flush()
                self._trace.close()
                self._trace = None

    def _write(self, event: dict):
        with self._lock:
            if self._trace is None:
                return
            self._trace.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self._trace.flush()

    def _next_call_id(self) -> tuple[int, str]:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        return sequence, f"inspect-{sequence:06d}-{uuid.uuid4().hex[:8]}"

    def inspect(
        self,
        observation,
        *,
        requested_observation_id: str,
        tick: float,
        model_width: int = 1920,
        model_height: int = 1080,
        model_image_bytes: bytes | None = None,
        model_image_mime_subtype: str | None = None,
        raw_model_call: str = "",
        inspection_latency_s: float = 0.0,
        protocol_error: str | None = None,
    ) -> dict:
        """Return the cached result attached to ``observation`` and a runner-verifiable proof."""
        started = time.time()
        metadata = getattr(observation, "metadata", {}) or {}
        expected = metadata.get("observation_id")
        canonical = metadata.get("interactables")
        sequence, call_id = self._next_call_id()
        error = protocol_error
        if not expected:
            error = error or "observation_missing_identity"
        elif requested_observation_id != expected:
            error = error or "observation_id_mismatch"
        elif not isinstance(canonical, dict):
            error = error or "observation_missing_interactables"
        elif canonical.get("scene", {}).get("status") != "ok":
            error = error or "interactables_scene_unavailable"
        elif not canonical.get("binding", {}).get("stable"):
            error = error or "interactables_binding_unstable"

        delivered = None
        text = ""
        som = None
        if error is None:
            delivered = project_result(canonical, model_width, model_height)
            if self.text_observation:
                from realtime_framework.sandbox.observation.appearance import attach_web_control_appearance

                attach_web_control_appearance(
                    delivered, observation.screenshot, width=model_width, height=model_height
                )
            som = build_hybrid_som(
                delivered,
                model_image_bytes,
                action_width=model_width,
                action_height=model_height,
                text_observation=self.text_observation,
            )
            text = som["text"]
        elapsed = time.time() - started
        proof = {
            "valid": error is None,
            "call_id": call_id,
            "observation_id": expected,
            "requested_observation_id": requested_observation_id,
            "tool_name": "inspect_interactables",
            "status": (
                canonical.get("scene", {}).get("status") if isinstance(canonical, dict) else "error"
            ),
            "protocol_error": error,
            "representation": som.get("representation") if som else None,
            "marked_image_sha256": som.get("marked_image_sha256") if som else None,
            "mark_map_sha256": som.get("mark_map_sha256") if som else None,
            "inspection_round_latency_s": round(float(inspection_latency_s or 0.0), 6),
            "tool_execution_latency_s": round(elapsed, 6),
        }
        event = {
            "schema_version": "tool_trace.v1",
            "sequence": sequence,
            "call_id": call_id,
            "tick": tick,
            "task_id": self.task_id,
            "adapter": self.adapter,
            "model": self.model,
            "tool_name": "inspect_interactables",
            "arguments": {"observation_id": requested_observation_id},
            "observation_id": expected,
            "screenshot_sha256": metadata.get("screenshot_sha256"),
            "raw_model_call": (raw_model_call or "")[:4000],
            "protocol_error": error,
            "latency": {
                "inspection_round_model_s": round(float(inspection_latency_s or 0.0), 6),
                "tool_execution_s": round(elapsed, 6),
            },
            "canonical_result_sha256": result_digest(canonical)
            if isinstance(canonical, dict)
            else None,
            "canonical_result": canonical,
            **({"delivered_result": delivered} if self.text_observation else {}),
            "delivered_coordinate_space": (
                {"width": model_width, "height": model_height} if delivered is not None else None
            ),
            "model_image_mime_subtype": model_image_mime_subtype,
            "model_image_sha256": (
                hashlib.sha256(model_image_bytes).hexdigest() if model_image_bytes else None
            ),
            "representation": som.get("representation") if som else None,
            "marked_image_sha256": som.get("marked_image_sha256") if som else None,
            "mark_map_sha256": som.get("mark_map_sha256") if som else None,
            "delivered_mark_map": som.get("marks") if som else None,
            "delivered_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text
            else None,
            "delivered_text": text,
        }
        with self._lock:
            self._pending[call_id] = event
        return {
            "proof": proof,
            "canonical": canonical,
            "delivered": delivered,
            "text": text,
            "marks": som.get("marks") if som else {},
            "image_bytes": som.get("image_bytes") if som else None,
            "image_mime_subtype": som.get("image_mime_subtype") if som else None,
        }

    def resolve_action(self, action, execution: dict | None):
        """Compile current-observation mark references into the canonical coordinate action."""
        marks = execution.get("marks") if isinstance(execution, dict) else {}
        resolved, trace = resolve_action_references(action, marks or {})
        if isinstance(execution, dict):
            execution["reference_resolution"] = trace
        return resolved

    def finish(
        self,
        execution: dict,
        *,
        action_latency_s: float = 0.0,
        raw_action_output: str = "",
        outcome: str = "completed",
    ) -> None:
        """Finish and persist one observation injection after its action round."""
        call_id = (execution.get("proof") or {}).get("call_id")
        if not call_id:
            return
        with self._lock:
            event = self._pending.pop(call_id, None)
        if event is None:
            return
        event["latency"]["action_round_model_s"] = round(float(action_latency_s or 0.0), 6)
        event["raw_action_output"] = (raw_action_output or "")[:8000]
        event["outcome"] = outcome
        event["reference_resolution"] = execution.get("reference_resolution")
        self._write(event)
