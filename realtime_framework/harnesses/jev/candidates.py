"""Candidate GUI actions of one observation, including field-aware literal typing choices."""
from __future__ import annotations

import ast
import calendar
import datetime
import json
import re
from decimal import Decimal, InvalidOperation

from realtime_framework.harnesses.jev.composition import text_inventory

CANDIDATE_POLICY = "jev_web_controls_native_select_compose_v9"
# Whole page-text rows and "label: value" tails longer than this are prose, not field values.
_MAX_PAGE_LITERAL = 80

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTHS = {name.lower(): month for month in range(1, 13)
           for name in (calendar.month_name[month], calendar.month_abbr[month])}
_MONTH_NAME = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"
_NAMED_DATE = re.compile(rf"\b({_MONTH_NAME})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I)
_DAY_FIRST_DATE = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_NAME})\.?\s+(\d{{4}})\b", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_URL = re.compile(r"https?://[^\s<>\"']+")
_CODE = re.compile(r"\b[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)+\b")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?(?![\w.])")
# Single quotes must open and close at word boundaries so apostrophes ("Ortiz's") never pair.
_QUOTED = re.compile(r'''"(?:\\.|[^"\\\n])*"|(?<!\w)'(?:\\.|[^'\\\n])*'(?!\w)|“[^”\n]+”|'''
                     r'''(?<!\w)‘[^’\n]+’(?!\w)|`[^`\n]+`''')


def text_options(instruction: str, marks: dict, ax_tree: str, field: dict,
                 ax_text: list[dict] | None = None) -> list[dict]:
    """Enumerate finite observed literals, not arbitrary generated strings or hidden state.

    Only explicit field formats filter values. Source order never imposes a count cap;
    the driver handles large candidate sets using bounded Choice tournaments.
    """
    sources = [(instruction, "task", False)]
    sources += [(item.get("label", ""), f"mark:{mark}", False)
                for mark, item in marks.items()]
    if ax_text is not None:
        sources += [(row["text"], f"AX:{row['frame_id']}:{row['node_id']}:{row['field']}", True)
                    for row in ax_text if isinstance(row.get("text"), str)]
    else:
        sources.append((ax_tree, "AX", False))
    options: dict[str, dict] = {}
    calendar_months: dict[tuple[int, int], str] = {}
    constraints = field.get("input_constraints") or {}
    input_type = field.get("input_type", "").lower()
    date_field = input_type == "date" or "yyyy-mm-dd" in (
        str(field.get("label", "")) + " " + str(constraints.get("placeholder", ""))
    ).lower()

    def accepts(value):
        if constraints.get("maxlength") is not None:
            try:
                if len(value) > int(constraints["maxlength"]):
                    return False
            except (ValueError, TypeError):
                pass
        if date_field:
            if not _DATE.fullmatch(value):
                return False
            try:
                datetime.date.fromisoformat(value)
            except ValueError:
                return False
            return not ((constraints.get("min") and value < constraints["min"]) or
                        (constraints.get("max") and value > constraints["max"]))
        if input_type == "email":
            return bool(_EMAIL.fullmatch(value))
        if input_type == "number":
            try:
                number = Decimal(value)
                if not number.is_finite():
                    return False
                for key, bound in (("min", -1), ("max", 1)):
                    if constraints.get(key) not in (None, ""):
                        limit = Decimal(constraints[key])
                        if (bound == -1 and number < limit) or (bound == 1 and number > limit):
                            return False
            except InvalidOperation:
                return False
        return True

    def add(value, source):
        value = value.strip()
        if date_field and _DATE.fullmatch(value):
            try:
                date = datetime.date.fromisoformat(value)
                calendar_months.setdefault((date.year, date.month), source)
            except ValueError:
                pass
        if value and accepts(value):
            if value not in options:
                options[value] = {"text": value, "sources": []}
            if source not in options[value]["sources"]:
                options[value]["sources"].append(source)

    for option in constraints.get("options", []):
        add(str(option.get("value", "")), "field:datalist")
    for text, source, whole in sources:
        if whole and len(text.strip()) <= _MAX_PAGE_LITERAL:
            add(text, source)
        # Extract complete structured values before considering numeric subspans. Do not
        # produce bogus year/day fragments from dates or parts of IDs and email addresses.
        covered = []
        for pattern in (_NAMED_DATE, _DAY_FIRST_DATE):
            for match in pattern.finditer(text):
                first, second, year = match.groups()
                month, day = (_MONTHS[first.lower()], int(second)) if pattern is _NAMED_DATE else (
                    _MONTHS[second.lower()], int(first))
                try:
                    add(datetime.date(int(year), month, day).isoformat(), source + ":date_format")
                except ValueError:
                    pass
                covered.append(match.span())
        for pattern in (_DATE, _EMAIL, _URL, _CODE):
            for match in pattern.finditer(text):
                add(match.group(), source)
                covered.append(match.span())
        for match in _QUOTED.finditer(text):
            token = match.group()
            try:
                value = ast.literal_eval(token) if token[0] in "\"'" else token[1:-1]
            except (ValueError, SyntaxError):
                value = token[1:-1]
            if isinstance(value, str):
                add(value, source)
        for line in text.splitlines():
            # Preserve full visible field/value text such as "Name: Jane Doe" or addresses.
            if ": " in line:
                value = line.split(": ", 1)[1]
                if len(value.strip()) <= _MAX_PAGE_LITERAL:
                    add(value, source)
        for match in _NUMBER.finditer(text):
            if not any(start < match.end() and match.start() < end for start, end in covered):
                add(match.group().replace(",", ""), source)
    # A date picker offers dates beyond already printed values. Give an equivalent finite
    # calendar around each observed month, without computing which date solves the task.
    # This is deliberately not claimed to enumerate every possible date or free-form string.
    for (year, month), source in list(calendar_months.items()):
        for shift in (-1, 0, 1):
            month_index = year * 12 + month - 1 + shift
            y, m = month_index // 12, month_index % 12 + 1
            if not 1 <= y <= 9999:
                continue
            for day in range(1, calendar.monthrange(y, m)[1] + 1):
                add(datetime.date(y, m, day).isoformat(), source + ":calendar_adjacent_months")
    return list(options.values())


def _compact_label(value: object, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def observation_candidates(instruction: str, marks: dict, active_window: dict):
    """Return (candidates, focused_input, text_composition) for one bound observation.

    Candidates are executable actions of the current observation only.
    """
    ax_tree = active_window.get("ax_tree") or ""
    candidates = {"wait": {
        "description": "Wait for the next scheduled observation without clicking.",
        "action": {"action": "wait"},
    }}

    def add(name: str, description: str, action: dict, *, audit: dict | None = None) -> None:
        candidate = {"description": description, "action": action}
        if audit:
            candidate["audit"] = audit
        candidates[name] = candidate

    for mark, item in marks.items():
        options = item.get("select_options") or []
        if not options or not item.get("focused") or item.get("select_multiple"):
            continue
        enabled = [o for o in options if not o.get("disabled") and not o.get("hidden")]
        selected = next((o["index"] for o in options if o.get("selected")), -1)
        add(f"confirm_select_{mark}",
            f"Press Enter in {mark} to commit the currently highlighted native menu option. "
            "After arrow navigation an open menu's highlight may differ from the DOM's "
            "committed selected value until Enter is pressed.",
            {"action": "key", "key": "Return"})
        for option in enabled:
            # Standard keyboard navigation only. No DOM setter or compound GUI macro.
            # A disabled placeholder can be selected; Down then reaches the first option.
            target = option["index"]
            if target > selected:
                delta = sum(selected < o["index"] <= target for o in enabled)
            elif target < selected:
                delta = -sum(target <= o["index"] < selected for o in enabled)
            else:
                delta = 0
            if delta == 0:
                continue  # Enter is represented once by the independent confirmation choice.
            action = {"action": "key", "key": "Down" if delta > 0 else "Up",
                      "repeat": min(abs(delta), 100)}
            description = (f"Move selection {'toward' if abs(delta) > 100 else 'to'} "
                           f"option {option['label']!r} in {mark} using "
                           f"{action['key']} {action['repeat']} times; confirm separately")
            add(f"option_{mark}_{option['index']}", description, action,
                audit={"kind": "native_select_keyboard", "option_index": option["index"],
                       "option_label": option["label"], "source_id": item.get("source_id")})

    sources = [(mark, item) for mark, item in marks.items()
               if item.get("kind") == "point" and "drag" in item.get("actions", [])]
    targets = [(mark, item) for mark, item in marks.items()
               if item.get("kind") == "point" and "drop" in item.get("actions", [])]
    for source_mark, source in sources:
        for target_mark, target in targets:
            if source_mark == target_mark:
                continue
            add(f"drag_{source_mark}_{target_mark}",
                f"Drag {source_mark} {source.get('label')} to "
                f"{target_mark} {target.get('label')}",
                {"action": "drag", "start": source_mark,
                 "end": target_mark, "duration": 0.5})

    focused = [
        (mark, item) for mark, item in marks.items()
        if item.get("focused") and "type" in item.get("actions", [])
    ]
    focused_input = None
    text_composition = None
    if focused:
        focused_mark, focused_field = focused[0]
        field = focused_field.get("label") or "focused field"
        current_value = focused_field.get("value") or ""
        may_type = not current_value or focused_field.get("text_selected")
        focused_input = {
            "mark": focused_mark,
            "label": field,
            "input_type": focused_field.get("input_type") or "text",
            "input_constraints": focused_field.get("input_constraints") or {},
            "current_value": current_value,
            "text_selected": bool(focused_field.get("text_selected")),
        }
        literals = text_options(instruction, marks, ax_tree, focused_field,
                                active_window.get("ax_text"))
        text_composition = text_inventory(
            instruction, marks, ax_tree, focused_field, active_window.get("ax_text"), literals,
        )
        # A private output-encoding choice: the decision resolves it into a single
        # ordinary type action. It never reaches the shared action executor.
        add("compose_text", "Compose new text for the focused field by choosing exact "
            "words or characters, then finish to type it. Inserts at the caret or "
            "replaces the current selection; does not automatically select existing text.",
            {"action": "compose_text"},
            audit={"kind": "text_composition", "focused_mark": focused_mark})
        if current_value and not focused_field.get("text_selected"):
            add("select_all", "Select existing text in the currently focused field",
                {"action": "key", "keys": ["ctrl", "a"]})
        if current_value:
            add("delete_text", "Delete selected text in the currently focused field",
                {"action": "key", "key": "Backspace"})
        add("press_enter", "Press Enter in the currently focused field",
            {"action": "key", "key": "Return"})
        add("press_tab", "Press Tab to leave the currently focused field",
            {"action": "key", "key": "Tab"})
        if may_type:
            for index, option in enumerate(literals, 1):
                value = option["text"]
                # The current state carries the focused field once. Exact provenance stays
                # in audit metadata instead of being repeated in every native Choice label.
                add(f"type_{index}", f"Type exact value {value!r}",
                    {"action": "type", "text": value},
                    audit={"kind": "type", "focused_mark": focused_mark,
                           "sources": option["sources"]})

    for mark, item in marks.items():
        actions = item.get("actions", [])
        label = item.get("label") or ""
        content = item.get("visible_content") or {}
        details = {k: v for k, v in (item.get("state") or {}).items()
                   if k in {"checked", "selected", "expanded", "value"}
                   and v is not None and v != ""}
        if content.get("text"):
            details["visible_text"] = _compact_label(content["text"])
        if content.get("image_labels"):
            details["image_labels"] = content["image_labels"]
        if item.get("appearance"):
            details["pixel_colors"] = item["appearance"].get("colors")
        state_label = " " + json.dumps(details, ensure_ascii=False) if details else ""
        if item.get("kind") == "point":
            # Some applications explicitly say "click" on elements whose DOM also
            # advertises drag/drop; a plain click remains a valid GUI action there.
            if "click" in actions or (("drag" in actions or "drop" in actions)
                                      and "click" in label.lower()):
                add(f"click_{mark}",
                    f"Click {mark}: {item.get('role')} {_compact_label(label)}{state_label}",
                    {"action": "click", "target": mark})
        elif item.get("kind") == "region":
            if "click" in actions:
                add(f"click_{mark}", f"Click center of {mark}: {_compact_label(label)}",
                    {"action": "click", "target": mark})
            if "scroll" in actions:
                for direction, amount in (("up", 600), ("down", -600)):
                    add(f"scroll_{direction}_{mark}",
                        f"Scroll {direction} in {mark}: {label}",
                        {"action": "scroll", "target": mark, "dx": 0, "dy": amount})
            if "drag" in actions:
                for direction, end in (("left", "250,500"), ("right", "750,500"),
                                       ("up", "500,250"), ("down", "500,750")):
                    add(f"pan_{direction}_{mark}",
                        f"Drag {mark} {label} toward {direction}",
                        {"action": "drag", "start": f"{mark}@500,500",
                         "end": f"{mark}@{end}", "duration": 0.5})
    return candidates, focused_input, text_composition
