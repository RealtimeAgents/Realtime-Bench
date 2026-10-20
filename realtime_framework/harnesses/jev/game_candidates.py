"""Game controls from the current observation, without task-specific action policies."""

from realtime_framework.entity.actions import FrameGeometry

GAME_CANDIDATE_POLICY = "screenshot_game_keys_composed_pairs_grid_pointer_v1"
POINTER_COMPOSITION_POLICY = "normalized_coordinate_hundreds_then_units_v1"


def add_game_candidates(candidates, marks, delivered, geometry):
    """Extend ordinary visible controls with keyboard and continuous pointer actions.

    Declared keys and grid geometry are read-only observation data. Never infer a winning
    move, inspect game state, or filter grid cells by whether a move would succeed.
    """
    def add(name, description, action, **audit):
        candidates[name] = {"description": description, "action": action, "audit": audit}

    keys = list(dict.fromkeys(
        item["key"] for item in delivered.get("keyboard_actions", []) if item.get("key")
    ))
    for index, key in enumerate(keys):
        add(f"game_key_{index}", f"Tap {key} once", {"action": "key", "key": key},
            kind="game_keyboard", keys=[key])
    for index, key in enumerate(keys):
        for duration in (0.1, 0.5, 1.0):
            add(f"game_hold_{index}_{duration}",
                f"Hold {key} for {duration:g} seconds; then release",
                {"action": "hold_key", "key": key, "duration": duration},
                kind="game_keyboard", keys=[key], duration=duration)
    # Compose pairs instead of expanding O(keys²) options (e.g. a full alphabet keyboard).
    if len(keys) >= 2:
        add("game_hold_pair", "Choose two declared keys and a duration, then hold them together",
            {"action": "compose_hold", "keys": keys}, kind="game_keyboard_pair")

    for mark, item in marks.items():
        if item.get("kind") != "region":
            continue
        grid = item.get("grid") or {}
        box = item.get("bbox") or []
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            continue
        if grid and "click" in item.get("actions", []):
            origin, step = grid.get("origin", []), grid.get("step", [])
            if len(origin) == len(step) == 2:
                for row, row_label in enumerate(grid.get("row_labels", [])):
                    for col, col_label in enumerate(grid.get("column_labels", [])):
                        x, y = origin[0] + col * step[0], origin[1] + row * step[1]
                        if not (box[0] <= x < box[2] and box[1] <= y < box[3]):
                            continue
                        u = round((x - box[0]) * 1000 / (box[2] - box[0]))
                        v = round((y - box[1]) * 1000 / (box[3] - box[1]))
                        add(f"game_grid_{mark}_{row}_{col}",
                            f"Click {mark} grid column {col_label}, row {row_label} (@{u},{v})",
                            {"action": "click", "target": f"{mark}@{u},{v}"},
                            kind="observed_grid", mark=mark, row=row, column=col)
        _pointer_candidates(add, mark, item.get("label") or mark, region=mark)

    # Canvas controls need not have a DOM/SoM mark. The shared GUI protocol permits image
    # coordinates; expose the same fallback with the current screenshot's geometry.
    width, height = FrameGeometry.from_dict(geometry).image_size
    _pointer_candidates(add, "screen", "current screenshot", image_size=[width, height])
    add("game_mouse_release", "Release the left mouse button at the current cursor",
        {"action": "mouse_up"}, kind="game_pointer_release")


def _pointer_candidates(add, name, label, **target):
    for operation in ("click", "right_click", "double_click", "move", "mouse_down", "mouse_up"):
        add(f"game_pointer_{name}_{operation}",
            f"{operation} at a point you will choose within {label}; "
            "choose both coordinates before executing one action",
            {"action": "compose_pointer", "operation": operation, **target},
            kind="game_pointer", operation=operation, **target)


def coordinate_choices(base=None):
    """Cover every integer 0..1000 without a coarse grid or a >255-option question."""
    if base is None:
        return {str(value): {
            "description": f"Coordinate {'1000' if value == 1000 else f'{value}..{value + 99}'}",
            "value": value,
        } for value in range(0, 1001, 100)}
    return {str(value): {"description": f"Coordinate {value}", "value": value}
            for value in range(base, min(base + 100, 1001))}
