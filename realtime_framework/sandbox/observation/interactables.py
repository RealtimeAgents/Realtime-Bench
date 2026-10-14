from __future__ import annotations

# Interactables
"""Canonical schema and deterministic model rendering for browser interactables.

Runtime geometry is always stored in the executor's 1920x1080 desktop coordinate space.  The
model-facing copy is projected into the adapter's coordinate space.
The canonical object is what is persisted in ``tool_calls.jsonl``.
"""

from copy import deepcopy as deepcopy
import hashlib as hashlib
import json as json
import math as math
import re as re
from typing import Any as Any

from realtime_framework.entity.actions import DESKTOP_SIZE
from realtime_framework.entity.actions import FrameGeometry


INTERACTABLES_SCHEMA_VERSION = "inspect_interactables.v1"
DESKTOP_WIDTH = 1920
DESKTOP_HEIGHT = 1080
MAX_ITEMS_PER_SECTION = 120
MAX_LABEL_CHARS = 160
MAX_CONTEXT_CHARS = 240
MAX_SEMANTIC_LABEL_CHARS = 320
MAX_WINDOWS = 16
CANVAS_COVERAGE = {
    "complete",
    "continuous_coordinate_space",
    "not_applicable",
    "unsupported",
}


def _clean_text(value: Any, limit: int = MAX_LABEL_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        text = text[: max(0, limit - 1)] + "…"
    return text


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return default


def _rect(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    nums = [_number(v, float("nan")) for v in value]
    if not all(math.isfinite(v) for v in nums):
        return None
    x1, y1, x2, y2 = nums
    x1, x2 = sorted((max(0.0, min(DESKTOP_WIDTH, x1)), max(0.0, min(DESKTOP_WIDTH, x2))))
    y1, y2 = sorted((max(0.0, min(DESKTOP_HEIGHT, y1)), max(0.0, min(DESKTOP_HEIGHT, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1), round(y1), round(x2), round(y2)]


def _point(value: Any, box: list[int] | None = None) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        x, y = (_number(value[0], float("nan")), _number(value[1], float("nan")))
        if math.isfinite(x) and math.isfinite(y):
            return [
                round(max(0, min(DESKTOP_WIDTH - 1, x))),
                round(max(0, min(DESKTOP_HEIGHT - 1, y))),
            ]
    if box:
        return [round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)]
    return None


# Roles whose ``checked`` state is meaningful. The DOM extractor reads ``el.checked`` from every
# input element, and that property is a boolean (False) on text/search/number/submit inputs too.
CHECKABLE_ROLES = frozenset({"checkbox", "radio", "switch", "menuitemcheckbox", "menuitemradio"})


def _item(raw: dict, index: int, section: str) -> dict | None:
    box = _rect(raw.get("bbox"))
    if box is None:
        return None
    role = _clean_text(raw.get("role") or raw.get("tag") or "control", 48)
    input_type = _clean_text(raw.get("input_type"), 32)
    checkable = role in CHECKABLE_ROLES or input_type in {"checkbox", "radio"}
    actions = sorted({_clean_text(action, 32) for action in raw.get("actions", []) if action})
    reasons = sorted(
        {_clean_text(reason, 64) for reason in raw.get("unavailable_reasons", []) if reason}
    )
    item_id = _clean_text(raw.get("id"), 96) or f"dom-{index + 1:04d}"
    label = _clean_text(raw.get("label"))
    context = _clean_text(raw.get("context"), MAX_CONTEXT_CHARS)
    semantic_label = _clean_text(
        raw.get("semantic_label") or label or context, MAX_SEMANTIC_LABEL_CHARS
    )
    item = {
        "id": item_id,
        "dom_order": int(_number(raw.get("dom_order"), index)),
        "source": _clean_text(raw.get("source") or "dom", 32),
        "role": role,
        "label": label,
        "context": context,
        "semantic_label": semantic_label,
        "tag": _clean_text(raw.get("tag"), 32),
        "input_type": input_type,
        "bbox": box,
        "click_point": _point(raw.get("click_point"), box),
        "actions": actions,
        "state": {
            "enabled": bool(raw.get("enabled", section != "visible_but_unavailable")),
            "readonly": bool(raw.get("readonly", False)),
            "editable": bool(raw.get("editable", False)),
            "checked": (
                raw.get("checked") if checkable and isinstance(raw.get("checked"), bool) else None
            ),
            "selected": raw.get("selected") if isinstance(raw.get("selected"), bool) else None,
            "expanded": raw.get("expanded") if isinstance(raw.get("expanded"), bool) else None,
            "modal_scope": bool(raw.get("modal_scope", False)),
            "occluded": bool(raw.get("occluded", False)),
        },
        "unavailable_reasons": reasons,
        "availability": section,
    }
    # Present only when the extractor ran for a text-observation model.
    for key in ("visible_content", "select_options", "select_multiple"):
        if key in raw:
            item[key] = raw[key]
    if isinstance(raw.get("input_constraints"), dict):
        item["input_constraints"] = raw["input_constraints"]
    if "focused" in raw:
        item["state"].update(
            focused=bool(raw.get("focused")),
            value=_clean_text(raw.get("value"), 160),
            text_selected=bool(raw.get("text_selected")),
        )
    frame_path = _clean_text(raw.get("frame_path"), 128)
    if frame_path:
        item["frame_path"] = frame_path
    canvas_id = _clean_text(raw.get("canvas_id"), 128)
    if canvas_id:
        item["canvas_id"] = canvas_id
    if section == "canvas_surfaces":
        item["canvas_bbox_screen"] = list(box)
        item["css_size"] = _size(raw.get("css_size"))
        item["backing_size"] = _size(raw.get("backing_size"))
        item["device_pixel_ratio"] = _number(raw.get("device_pixel_ratio"), 1.0)
        backing = item["backing_size"] or item["css_size"]
        if backing:
            item["canvas_coordinate_transform"] = {
                "from": "canvas_backing_px" if item["backing_size"] else "canvas_css_px",
                "to": "desktop_px",
                "scale_x": (box[2] - box[0]) / backing[0],
                "scale_y": (box[3] - box[1]) / backing[1],
                "offset_x": box[0],
                "offset_y": box[1],
            }
        else:
            item["canvas_coordinate_transform"] = None
        coverage = _clean_text(raw.get("internal_target_coverage"), 48)
        item["internal_target_coverage"] = (
            coverage if coverage in CANVAS_COVERAGE else "unsupported"
        )
        item["internal_target_count"] = max(0, int(_number(raw.get("internal_target_count"))))
        interaction = _clean_text(raw.get("interaction"), 240)
        if interaction:
            item["interaction"] = interaction
        target_map = _canvas_target_map(raw.get("internal_target_map"))
        if target_map:
            item["internal_target_map"] = target_map
    return item


def _size(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    width, height = round(_number(value[0])), round(_number(value[1]))
    return [width, height] if width > 0 and height > 0 else None


def _canvas_target_map(value: Any) -> dict | None:
    if not isinstance(value, dict) or value.get("kind") != "grid":
        return None
    columns = [_clean_text(label, 32) for label in value.get("column_labels", [])[:40] if label]
    rows = [_clean_text(label, 32) for label in value.get("row_labels", [])[:40] if label]
    origin = _point(value.get("origin"))
    step = value.get("step")
    if (
        not columns
        or not rows
        or origin is None
        or not isinstance(step, (list, tuple))
        or len(step) != 2
    ):
        return None
    step_x = _number(step[0], float("nan"))
    step_y = _number(step[1], float("nan"))
    if not math.isfinite(step_x) or not math.isfinite(step_y):
        return None
    return {
        "kind": "grid",
        "column_labels": columns,
        "row_labels": rows,
        "origin": origin,
        "step": [step_x, step_y],
        "target_count": len(columns) * len(rows),
    }


def _items(raw: dict, section: str) -> tuple[list[dict], bool]:
    values = raw.get(section) if isinstance(raw.get(section), list) else []
    normalized = []
    for index, value in enumerate(values[:MAX_ITEMS_PER_SECTION]):
        if isinstance(value, dict):
            item = _item(value, index, section)
            if item:
                normalized.append(item)
    normalized.sort(key=lambda item: (item["dom_order"], item["id"]))
    return normalized, len(values) > MAX_ITEMS_PER_SECTION


def _window(raw: dict, index: int) -> tuple[dict, bool]:
    actionable, actionable_truncated = _items(raw, "actionable_now")
    unavailable, unavailable_truncated = _items(raw, "visible_but_unavailable")
    canvases, canvases_truncated = _items(raw, "canvas_surfaces")
    interactables = actionable + unavailable + canvases
    interactables.sort(key=lambda item: (item["dom_order"], item["id"], item["availability"]))
    scene = raw.get("scene") if isinstance(raw.get("scene"), dict) else {}
    return (
        {
            "window_id": _clean_text(raw.get("window_id") or scene.get("target_id"), 128)
            or f"window-{index + 1}",
            "title": _clean_text(raw.get("title") or scene.get("title"), 200),
            "active": bool(raw.get("active", False)),
            # Accessibility capture exists only for text-observation models.
            **({"ax_tree": raw["ax_tree"]} if isinstance(raw.get("ax_tree"), str) else {}),
            **({"ax_text": raw["ax_text"]} if isinstance(raw.get("ax_text"), list) else {}),
            **(
                {"ax_capture": raw["ax_capture"]}
                if isinstance(raw.get("ax_capture"), dict)
                else {}
            ),
            "interactables": interactables,
        },
        actionable_truncated or unavailable_truncated or canvases_truncated,
    )


def canonical_result(
    raw: dict | None,
    *,
    observation_id: str,
    screenshot_sha256: str,
    captured_at: float,
    task_id: str | None = None,
    frame_geometry: FrameGeometry | None = None,
) -> dict:
    """Normalize an extractor result into the versioned, bounded canonical contract."""
    raw = raw if isinstance(raw, dict) else {}
    scene_raw = raw.get("scene") if isinstance(raw.get("scene"), dict) else {}
    binding_raw = raw.get("binding") if isinstance(raw.get("binding"), dict) else {}
    status = (
        scene_raw.get("status")
        if scene_raw.get("status") in {"ok", "unsupported", "error"}
        else "error"
    )
    geometry = frame_geometry or FrameGeometry(DESKTOP_SIZE, DESKTOP_SIZE)
    screen_width, screen_height = geometry.screen_size
    result = {
        "schema_version": INTERACTABLES_SCHEMA_VERSION,
        "task_id": task_id,
        "observation": {
            "observation_id": observation_id,
            "screenshot_sha256": screenshot_sha256,
            "captured_at": float(captured_at),
        },
        "scene": {
            "status": status,
            "kind": _clean_text(scene_raw.get("kind") or "unknown", 48),
            "application": _clean_text(scene_raw.get("application"), 80),
            "target_id": _clean_text(scene_raw.get("target_id"), 128),
            "url": _clean_text(scene_raw.get("url"), 500),
            "title": _clean_text(scene_raw.get("title"), 200),
            "reason": _clean_text(scene_raw.get("reason"), 160),
        },
        "binding": {
            "stable": bool(binding_raw.get("stable", status != "ok")),
            "unstable": bool(binding_raw.get("unstable", False)),
            "revision": _clean_text(binding_raw.get("revision"), 128),
            "fingerprint": _clean_text(binding_raw.get("fingerprint"), 128),
            "reason_codes": sorted(
                {_clean_text(v, 64) for v in binding_raw.get("reason_codes", []) if v}
            ),
        },
        "coordinate_space": {
            "name": "desktop_px",
            "origin": "top_left",
            "width": screen_width,
            "height": screen_height,
        },
        "frame_geometry": geometry.to_dict(),
        "actionable_now": [],
        "visible_but_unavailable": [],
        "keyboard_actions": [],
        "canvas_surfaces": [],
        "windows": [],
        "warnings": sorted({_clean_text(v, 160) for v in raw.get("warnings", []) if v}),
        "limitations": sorted({_clean_text(v, 160) for v in raw.get("limitations", []) if v}),
        "truncated": False,
    }
    raw_windows = raw.get("windows") if isinstance(raw.get("windows"), list) else []
    if not raw_windows and status == "ok":
        raw_windows = [
            {
                **raw,
                "window_id": scene_raw.get("target_id"),
                "title": scene_raw.get("title"),
                "active": True,
            }
        ]
    for index, value in enumerate(raw_windows[:MAX_WINDOWS]):
        if not isinstance(value, dict):
            continue
        window, truncated = _window(value, index)
        result["windows"].append(window)
        result["truncated"] = result["truncated"] or truncated
    if len(raw_windows) > MAX_WINDOWS:
        result["truncated"] = True

    active_window = next((window for window in result["windows"] if window["active"]), None)
    if active_window is None and result["windows"]:
        active_window = result["windows"][0]
    for window in result["windows"]:
        window["active"] = window is active_window
    if active_window is not None:
        result["actionable_now"] = [
            deepcopy(item)
            for item in active_window["interactables"]
            if item["availability"] == "actionable_now"
        ]
        result["visible_but_unavailable"] = [
            deepcopy(item)
            for item in active_window["interactables"]
            if item["availability"] == "visible_but_unavailable"
        ]
        result["canvas_surfaces"] = [
            deepcopy(item)
            for item in active_window["interactables"]
            if item["availability"] == "canvas_surfaces"
        ]
        # Accessibility wrappers around WebGL/map canvases are commonly exposed as a large
        # tabindex-enabled ``region``. The DOM extractor can only infer a generic click for such a
        # wrapper, but a plain map/canvas click often has no visual effect. Advertise exploratory
        # viewport gestures instead so models do not treat the absence of click feedback as a bug.
        canvases = result["canvas_surfaces"]
        for item in result["actionable_now"]:
            if item.get("role") != "region" or item.get("actions") != ["click"]:
                continue
            box = item.get("bbox")
            if not box:
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area <= 0:
                continue
            for canvas in canvases:
                canvas_box = canvas.get("bbox")
                if not canvas_box:
                    continue
                overlap_width = max(0, min(box[2], canvas_box[2]) - max(box[0], canvas_box[0]))
                overlap_height = max(0, min(box[3], canvas_box[3]) - max(box[1], canvas_box[1]))
                if overlap_width * overlap_height >= area * 0.9:
                    item["actions"] = ["drag", "scroll"]
                    for window_item in active_window["interactables"]:
                        if window_item.get("availability") == "actionable_now" and window_item.get(
                            "id"
                        ) == item.get("id"):
                            window_item["actions"] = ["drag", "scroll"]
                    break
    keys = raw.get("keyboard_actions") if isinstance(raw.get("keyboard_actions"), list) else []
    for value in keys[:MAX_ITEMS_PER_SECTION]:
        if not isinstance(value, dict) or not value.get("key"):
            continue
        result["keyboard_actions"].append(
            {
                "id": _clean_text(value.get("id") or f"key:{value['key']}", 96),
                "key": _clean_text(value["key"], 48),
                "label": _clean_text(value.get("label") or value["key"]),
                "source": _clean_text(value.get("source") or "scenario_declared", 32),
            }
        )
    result["keyboard_actions"].sort(key=lambda item: (item["id"], item["key"]))
    if len(keys) > MAX_ITEMS_PER_SECTION:
        result["truncated"] = True
    return result


def project_result(result: dict, width: int, height: int) -> dict:
    """Return a model-facing copy projected from desktop pixels to ``width`` × ``height``."""
    if width <= 0 or height <= 0:
        raise ValueError("coordinate dimensions must be positive")
    projected = deepcopy(result)
    canonical_geometry = FrameGeometry.from_dict(result.get("frame_geometry"))
    geometry = FrameGeometry(
        (width, height), canonical_geometry.screen_size, canonical_geometry.capture_bbox_screen
    )
    projected["coordinate_space"] = {
        "name": "model_tool_px",
        "origin": "top_left",
        "width": width,
        "height": height,
        "canonical_width": canonical_geometry.screen_size[0],
        "canonical_height": canonical_geometry.screen_size[1],
    }
    projected["frame_geometry"] = geometry.to_dict()
    for section in ("actionable_now", "visible_but_unavailable", "canvas_surfaces"):
        for item in projected.get(section, []):
            box = item["bbox"]
            if section == "canvas_surfaces":
                item["canvas_bbox_screen"] = list(item.get("canvas_bbox_screen") or box)
            item["bbox"] = geometry.screen_to_image_rect(box)
            if item.get("click_point"):
                item["click_point"] = list(geometry.screen_to_image_point(*item["click_point"]))
            if section == "canvas_surfaces":
                item["canvas_bbox_model"] = list(item["bbox"])
                _project_canvas_target_map(item, geometry)
    for window in projected.get("windows", []):
        for item in window.get("interactables", []):
            box = item["bbox"]
            is_canvas = item.get("availability") == "canvas_surfaces"
            if is_canvas:
                item["canvas_bbox_screen"] = list(item.get("canvas_bbox_screen") or box)
            item["bbox"] = geometry.screen_to_image_rect(box)
            if item.get("click_point"):
                item["click_point"] = list(geometry.screen_to_image_point(*item["click_point"]))
            if is_canvas:
                item["canvas_bbox_model"] = list(item["bbox"])
                _project_canvas_target_map(item, geometry)
    return projected


def _project_canvas_target_map(item: dict, geometry: FrameGeometry) -> None:
    target_map = item.get("internal_target_map")
    if not isinstance(target_map, dict) or target_map.get("kind") != "grid":
        return
    origin = target_map.get("origin")
    step = target_map.get("step")
    if not isinstance(origin, list) or not isinstance(step, list):
        return
    projected_origin = geometry.screen_to_image_point(origin[0], origin[1])
    sx, sy = geometry.scale
    target_map["origin_screen"] = list(origin)
    target_map["origin"] = list(projected_origin)
    # Keep the step fractional: a whole-pixel step accumulates its rounding over every column.
    target_map["step"] = [step[0] / sx, step[1] / sy]


# Full-text rendering of a projected result, used by text-observation models.
def _quoted(text: str, limit: int = MAX_LABEL_CHARS) -> str:
    return json.dumps(_clean_text(text, limit), ensure_ascii=False)


def _format_items_inline(lines: list[str], items: list[dict], section: str) -> None:
    for item in items:
        box = item["bbox"]
        point = item.get("click_point")
        label = item.get("label", "")
        semantic_label = item.get("semantic_label") or label
        context = item.get("context", "")
        suffix = f" actions={','.join(item.get('actions') or []) or 'none'}"
        if semantic_label != label:
            suffix += f" label={_quoted(label)}"
        if context:
            suffix += f" context={_quoted(context, MAX_CONTEXT_CHARS)}"
        if section == "visible_but_unavailable":
            suffix += f" unavailable={','.join(item.get('unavailable_reasons') or []) or 'unavailable'}"
        state = item.get("state") or {}
        selected = " selected=true" if state.get("selected") else ""
        checked = ""
        if isinstance(state.get("checked"), bool):
            checked = f" checked={'true' if state['checked'] else 'false'}"
        lines.append(
            f"{item['id']} {item['role']} "
            f"{_quoted(semantic_label, MAX_SEMANTIC_LABEL_CHARS)} "
            f"box=({box[0]},{box[1]},{box[2]},{box[3]})"
            + (f" point=({point[0]},{point[1]})" if point else "")
            + suffix + checked + selected
        )


def _grid_table(records: list[dict]) -> dict | None:
    """Recognize an exact Cartesian grid; never approximate coordinates or invent cells."""
    if len(records) < 4 or any(not row.get("point") for row in records):
        return None
    xs = sorted({(row["box"][0], row["box"][2], row["point"][0]) for row in records})
    ys = sorted({(row["box"][1], row["box"][3], row["point"][1]) for row in records})
    if min(len(xs), len(ys)) < 2 or len(xs) * len(ys) != len(records):
        return None
    cells = []
    for row in records:
        c = xs.index((row["box"][0], row["box"][2], row["point"][0]))
        r = ys.index((row["box"][1], row["box"][3], row["point"][1]))
        cells.append((r, c))
    # Preserve original element order as well as membership. Irregular/missing/duplicate
    # cells fall back to the ordinary table, including per-cell off-centre click points.
    if cells != [(r, c) for r in range(len(ys)) for c in range(len(xs))]:
        return None
    templates = {}
    for key in ("id", "name"):
        values = [row[key] for row in records]
        if any("{" in value or "}" in value for value in values):
            return None
        parts = [re.split(r"([0-9]+)", value) for value in values]
        if any(len(value) != len(parts[0]) for value in parts):
            return None
        template = []
        for tokens in zip(*parts):
            if len(set(tokens)) == 1:
                template.append(tokens[0])
            elif list(tokens) == [str(r) for r, c in cells]:
                template.append("{r}")
            elif list(tokens) == [str(c) for r, c in cells]:
                template.append("{c}")
            else:
                return None
        value = "".join(template)
        if "{r}" not in value or "{c}" not in value:
            return None
        templates[key] = value
    shared = {
        key: value for key, value in records[0].items()
        if key not in {"id", "name", "box", "point"}
        and all(key in row and row[key] == value for row in records[1:])
    }
    extras = []
    for (r, c), row in zip(cells, records):
        fields = {key: value for key, value in row.items()
                  if key not in {"id", "name", "box", "point"} and key not in shared}
        if fields:
            extras.append([r, c, fields])
    return {"templates": templates, "shared": shared,
            "columns": [[c, *values] for c, values in enumerate(xs)],
            "rows": [[r, *values] for r, values in enumerate(ys)], "extras": extras}


def _format_items(lines: list[str], items: list[dict], section: str) -> None:
    """Factor repeated fields within this observation, without cross-frame references."""
    if len(items) < 4:
        _format_items_inline(lines, items, section)
        return
    groups = []
    for item in items:
        if not groups or groups[-1][0]["role"] != item["role"]:
            groups.append([])
        groups[-1].append(item)
    if 1 < len(groups) <= len(items) // 4:
        for group in groups:
            _format_items(lines, group, section)
        return
    records = []
    for item in items:
        label = item.get("label", "")
        semantic_label = item.get("semantic_label") or label
        record = {
            "id": item["id"],
            "role": item["role"],
            "name": _clean_text(semantic_label, MAX_SEMANTIC_LABEL_CHARS),
            "box": item["bbox"],
            "actions": ','.join(item.get('actions') or []) or 'none',
        }
        if item.get("click_point"):
            record["point"] = item["click_point"]
        if semantic_label != label:
            record["label"] = _clean_text(label, MAX_LABEL_CHARS)
        if item.get("context"):
            record["context"] = _clean_text(item["context"], MAX_CONTEXT_CHARS)
        if section == "visible_but_unavailable":
            record["unavailable"] = ','.join(item.get('unavailable_reasons') or []) or 'unavailable'
        if isinstance((item.get("state") or {}).get("checked"), bool):
            record["checked"] = item["state"]["checked"]
        if (item.get("state") or {}).get("selected"):
            record["selected"] = True
        records.append(record)
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    grid = _grid_table(records)
    if grid is not None:
        lines.extend([
            "GRID templates=" + encode(grid["templates"]) + "; r,c are zero-based.",
            "All row/column pairs exist in row-major order; shared=" + encode(grid["shared"]),
            "COLUMNS [c,left,right,click_x]=" + encode(grid["columns"]),
            "ROWS [r,top,bottom,click_y]=" + encode(grid["rows"]),
            "Cell (r,c): point=[click_x[c],click_y[r]], "
            "box=[left[c],top[r],right[c],bottom[r]]; name=semantic label.",
        ])
        if grid["extras"]:
            lines.append("Per-cell extra fields [r,c,fields]=" + encode(grid["extras"]))
        return
    shared = {
        key: value for key, value in records[0].items()
        if len(records) > 1 and key != "id"
        and all(key in record and record[key] == value for record in records[1:])
    }
    columns = [key for key in (
        "id", "role", "name", "box", "point", "actions", "label", "context",
        "unavailable", "checked", "selected",
    ) if key not in shared and any(key in record for record in records)]
    lines.append("TABLE shared=" + encode(shared) + " columns=" + encode(columns))
    lines.extend(encode([record.get(key) for key in columns]) for record in records)


def format_interactables(result: dict) -> str:
    """Render a compact, deterministic tool result for every model family."""
    obs = result.get("observation", {})
    scene = result.get("scene", {})
    binding = result.get("binding", {})
    space = result.get("coordinate_space", {})
    lines = [
        f"INSPECT_INTERACTABLES v1 observation={obs.get('observation_id', '')} "
        f"status={scene.get('status', 'error')} space={space.get('width')}x{space.get('height')} "
        f"stable={str(bool(binding.get('stable'))).lower()} "
        f"unstable={str(bool(binding.get('unstable'))).lower()}"
    ]
    if scene.get("status") != "ok":
        lines.append(f"SCENE {scene.get('kind') or 'unknown'} reason={_quoted(scene.get('reason') or 'unavailable')}")
    windows = result.get("windows") if isinstance(result.get("windows"), list) else []
    lines.append(f"WINDOWS {len(windows)}")
    for window in windows:
        lines.append(
            f"WINDOW id={_quoted(window.get('window_id'), 128)} "
            f"title={_quoted(window.get('title'), 200)} "
            f"active={str(bool(window.get('active'))).lower()}"
        )
        interactables = window.get("interactables", [])
        for section, heading in (("actionable_now", "ACTIONABLE_NOW"),
                                 ("visible_but_unavailable", "VISIBLE_BUT_UNAVAILABLE")):
            items = [item for item in interactables if item.get("availability") == section]
            lines.append(f"{heading} {len(items)}")
            _format_items(lines, items, section)
        canvases = [item for item in interactables if item.get("availability") == "canvas_surfaces"]
        lines.append(f"CANVAS_SURFACES {len(canvases)}")
        for item in canvases:
            box = item["bbox"]
            semantic_label = item.get("semantic_label") or item.get("label", "canvas")
            context = item.get("context", "")
            coverage = item.get("internal_target_coverage", "unsupported")
            target_map = item.get("internal_target_map") or {}
            target_detail = f"internal_targets={coverage}"
            if target_map.get("kind") == "grid":
                origin = target_map["origin"]
                step = target_map["step"]
                target_detail = (
                    "internal_targets=grid_complete "
                    f"columns={','.join(target_map['column_labels'])} "
                    f"rows={','.join(target_map['row_labels'])} "
                    f"origin=({origin[0]},{origin[1]}) "
                    f"step=({step[0]:.3g},{step[1]:.3g}) "
                    f"target_count={target_map['target_count']}"
                )
            elif coverage == "complete":
                target_detail += f" target_count={item.get('internal_target_count', 0)}"
            if item.get("interaction"):
                target_detail += f" interaction={_quoted(item['interaction'], 240)}"
            if coverage == "unsupported":
                target_detail += " explore_with=drag,scroll plain_click_no_change_is_not_bug_evidence"
            lines.append(
                f"{item['id']} {_quoted(semantic_label, MAX_SEMANTIC_LABEL_CHARS)} "
                f"box=({box[0]},{box[1]},{box[2]},{box[3]}) {target_detail}"
                + (f" screen_box=({','.join(str(v) for v in item['canvas_bbox_screen'])})"
                   if item.get("canvas_bbox_screen") and item["canvas_bbox_screen"] != box else "")
                + (f" css_size={item['css_size'][0]}x{item['css_size'][1]}"
                   if item.get("css_size") else "")
                + (f" backing_size={item['backing_size'][0]}x{item['backing_size'][1]}"
                   if item.get("backing_size") else "")
                + (f" context={_quoted(context, MAX_CONTEXT_CHARS)}" if context else "")
            )
    if not windows:
        for section, heading in (("actionable_now", "ACTIONABLE_NOW"),
                                 ("visible_but_unavailable", "VISIBLE_BUT_UNAVAILABLE")):
            items = result.get(section, [])
            lines.append(f"{heading} {len(items)}")
            _format_items(lines, items, section)
        lines.append("CANVAS_SURFACES 0")
    keys = result.get("keyboard_actions", [])
    lines.append(f"KEYBOARD_ACTIONS {len(keys)}")
    for item in keys:
        lines.append(f"{item['id']} key={_quoted(item['key'])} label={_quoted(item['label'])}")
    if result.get("limitations"):
        lines.append("LIMITATIONS " + "; ".join(result["limitations"]))
    if result.get("truncated"):
        lines.append("TRUNCATED true")
    if any(line.startswith("TABLE shared=") for line in lines):
        lines.insert(1, "TABLE rows inherit shared fields; columns name array entries; null=absent. "
                     "name=semantic label; box=[left,top,right,bottom]; point=[x,y].")
    return "\n".join(lines)


def result_digest(result: dict) -> str:
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Specs
"""Load optional task-owned interactable identity and enrichment specifications."""

from pathlib import Path as Path

import tomllib as tomllib


_ROOT = Path(__file__).resolve().parents[3]
_CATEGORY_DIR = {
    "browser_game": "game",
    "realtime": "web",
}


class ToolSpecError(ValueError):
    pass


_OBJECT_PATH = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*$")
_CANVAS_COVERAGE = {"complete", "continuous_coordinate_space", "not_applicable", "unsupported"}


def _validate_entry(entry: Any, path: Path) -> dict:
    if not isinstance(entry, dict):
        raise ToolSpecError(f"{path}: each element rule must be a mapping")
    selector = entry.get("selector")
    if not isinstance(selector, str) or not selector.strip():
        raise ToolSpecError(f"{path}: element rule selector must be a non-empty string")
    out = {"selector": selector.strip()}
    for name in (
        "id_attribute",
        "entity_attribute",
        "label_attribute",
        "id_template",
        "label_template",
        "label",
        "role",
    ):
        value = entry.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ToolSpecError(f"{path}: {name} must be a non-empty string")
        if value:
            out[name] = value.strip()
    actions = entry.get("actions")
    if actions is not None:
        if not isinstance(actions, list) or any(not isinstance(v, str) or not v for v in actions):
            raise ToolSpecError(f"{path}: actions must be a list of strings")
        out["actions"] = list(actions)
    return out


def _validate_string_list(value: Any, *, name: str, path: Path) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ToolSpecError(f"{path}: {name} must be a non-empty list of strings")
    return [item.strip() for item in value]


def _validate_canvas_rule(entry: Any, path: Path) -> dict:
    if not isinstance(entry, dict):
        raise ToolSpecError(f"{path}: each canvas rule must be a mapping")
    selector = entry.get("selector")
    if not isinstance(selector, str) or not selector.strip():
        raise ToolSpecError(f"{path}: canvas rule selector must be a non-empty string")
    out: dict[str, Any] = {"selector": selector.strip()}
    for name in ("label", "interaction"):
        value = entry.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ToolSpecError(f"{path}: canvas {name} must be a non-empty string")
        if value:
            out[name] = value.strip()
    actions = entry.get("actions")
    if actions is not None:
        out["actions"] = _validate_string_list(actions, name="canvas actions", path=path)

    coverage = entry.get("coverage")
    if coverage is not None and coverage not in _CANVAS_COVERAGE:
        raise ToolSpecError(f"{path}: canvas coverage must be one of {sorted(_CANVAS_COVERAGE)}")
    if coverage:
        out["coverage"] = coverage

    grid = entry.get("grid")
    if grid is not None:
        if not isinstance(grid, dict):
            raise ToolSpecError(f"{path}: canvas grid must be a mapping")
        columns = _validate_string_list(
            grid.get("column_labels"), name="grid column_labels", path=path
        )
        rows = _validate_string_list(grid.get("row_labels"), name="grid row_labels", path=path)
        grid_out: dict[str, Any] = {"column_labels": columns, "row_labels": rows}
        for name in ("inset_x", "inset_y"):
            value = grid.get(name, 0.0)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 <= float(value) < 0.5
            ):
                raise ToolSpecError(f"{path}: grid {name} must be in [0, 0.5)")
            grid_out[name] = float(value)
        out["grid"] = grid_out

    targets = entry.get("targets")
    if targets is not None:
        if not isinstance(targets, dict):
            raise ToolSpecError(f"{path}: canvas targets must be a mapping")
        targets_out: dict[str, Any] = {}
        required_paths = ("root_path", "x_path", "y_path")
        optional_paths = (
            "offset_x_path",
            "offset_y_path",
            "radius_path",
            "visible_path",
            "enabled_path",
            "id_path",
        )
        for name in required_paths + optional_paths:
            value = targets.get(name)
            if name in required_paths and not value:
                raise ToolSpecError(f"{path}: canvas targets {name} is required")
            if value is not None:
                if not isinstance(value, str) or not _OBJECT_PATH.fullmatch(value):
                    raise ToolSpecError(f"{path}: canvas targets {name} is not a safe object path")
                targets_out[name] = value
        logical_size = targets.get("logical_size")
        if (
            not isinstance(logical_size, list)
            or len(logical_size) != 2
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
                for value in logical_size
            )
        ):
            raise ToolSpecError(f"{path}: canvas targets logical_size must contain two positives")
        targets_out["logical_size"] = [float(value) for value in logical_size]
        radius = targets.get("radius", 12)
        if not isinstance(radius, (int, float)) or isinstance(radius, bool) or radius <= 0:
            raise ToolSpecError(f"{path}: canvas targets radius must be positive")
        targets_out["radius"] = float(radius)
        for name in ("id_prefix", "label"):
            value = targets.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ToolSpecError(f"{path}: canvas targets {name} must be non-empty")
            if value:
                targets_out[name] = value.strip()
        out["targets"] = targets_out

    if grid is not None and targets is not None:
        raise ToolSpecError(f"{path}: a canvas rule cannot define both grid and targets")
    if (grid is not None or targets is not None) and coverage not in (None, "complete"):
        raise ToolSpecError(f"{path}: canvas grid/targets require complete coverage")
    if grid is not None or targets is not None:
        out["coverage"] = "complete"
    return out


def load_tool_spec(task) -> dict:
    """Return a validated task spec, or the generic DOM default when none exists."""
    category = getattr(task, "category", "") if task is not None else ""
    family = _CATEGORY_DIR.get(category)
    task_id = getattr(task, "task_id", None) if task is not None else None
    keys = getattr(task, "keys", ()) if task is not None else ()
    canvas_default_coverage = "not_applicable" if keys else "unsupported"
    default = {
        "schema_version": 1,
        "task_id": task_id,
        "elements": [],
        "canvases": [],
        "canvas_default_coverage": canvas_default_coverage,
    }
    if not family or not task_id:
        return default
    path = _ROOT / "realtime_tasks" / family / task_id / "task.toml"
    if not path.is_file():
        return default
    data = tomllib.loads(path.read_text(encoding="utf-8")).get("metadata", {}).get("interactables")
    if data is None:
        return default
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ToolSpecError(f"{path}: schema_version must be 1")
    if data.get("task_id") != task_id:
        raise ToolSpecError(f"{path}: task_id must be {task_id!r}")
    elements = data.get("elements", [])
    if not isinstance(elements, list):
        raise ToolSpecError(f"{path}: elements must be a list")
    canvases = data.get("canvases", [])
    if not isinstance(canvases, list):
        raise ToolSpecError(f"{path}: canvases must be a list")
    configured_default = data.get("canvas_default_coverage", canvas_default_coverage)
    if configured_default not in _CANVAS_COVERAGE:
        raise ToolSpecError(
            f"{path}: canvas_default_coverage must be one of {sorted(_CANVAS_COVERAGE)}"
        )
    return {
        "schema_version": 1,
        "task_id": task_id,
        "elements": [_validate_entry(entry, path) for entry in elements],
        "canvases": [_validate_canvas_rule(entry, path) for entry in canvases],
        "canvas_default_coverage": configured_default,
    }
