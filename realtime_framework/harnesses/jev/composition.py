"""Exact text composition (output encoding) for the choice-only JEV model.

These are output encodings, not a task solver. No candidate depends on previous
choices or progress. Spaces, casing and punctuation are never inserted implicitly.
"""
from __future__ import annotations

import json
import re
import string
import unicodedata


TEXT_COMPOSITION_POLICY = "jev_exact_fragments_ascii_observed_unicode_scalar_v1"


def text_inventory(instruction: str, marks: dict, ax_tree: str, field: dict,
                   ax_text: list[dict] | None, literals: list[dict]) -> dict:
    """Keep observed words/literals and every observed Unicode scalar, without a cap."""
    sources = [(instruction, "task")]
    for mark, item in marks.items():
        for name in ("label", "value"):
            sources.append((item.get(name, ""), f"mark:{mark}:{name}"))
        sources.append(((item.get("visible_content") or {}).get("text", ""),
                        f"mark:{mark}:visible_text"))
    if ax_text is not None:
        sources.extend((row.get("text", ""),
                        f"AX:{row.get('frame_id')}:{row.get('node_id')}:{row.get('field')}")
                       for row in ax_text)
    else:
        sources.append((ax_tree, "AX"))
    constraints = field.get("input_constraints") or {}
    sources.append((constraints.get("placeholder", ""), "field:placeholder"))
    for option in constraints.get("options", []):
        for name in ("label", "value"):
            sources.append((option.get(name, ""), f"field:datalist:{name}"))
    # Literal extraction also decodes explicitly quoted escapes. Do not lose those
    # characters when the source spells a newline or a Unicode character as an escape.
    for literal in literals:
        sources.append((literal["text"], "literal:" + ",".join(literal["sources"])))

    characters = dict.fromkeys(string.ascii_letters + string.digits + string.punctuation
                               + " \t\n\r")
    fragments = {}

    def add_fragment(value, source):
        if len(value) <= 1:
            return
        entry = fragments.setdefault(value, {"text": value, "sources": []})
        if source not in entry["sources"]:
            entry["sources"].append(source)

    for text, source in sources:
        if not isinstance(text, str):
            continue
        for char in text:
            if not 0xD800 <= ord(char) <= 0xDFFF:
                characters.setdefault(char, None)
        for match in re.finditer(r"\w+|[^\w\s]+", text):
            add_fragment(match.group(), source)
    for literal in literals:
        for source in literal["sources"]:
            add_fragment(literal["text"], source)
    return {"characters": list(characters), "fragments": list(fragments.values())}


def composition_candidates(inventory: dict) -> dict[str, dict]:
    """The same edit candidates remain available after every edit, even a repeated one."""
    candidates = {
        "finish": {"description": "Finish composing and type the exact draft into the focused field",
                   "operation": "finish"},
        "cancel": {"description": "Discard the draft and wait without typing",
                   "operation": "cancel"},
        "delete_last": {"description": "Delete the last Unicode code point from the draft",
                        "operation": "delete"},
        "clear": {"description": "Clear the draft only; do not change the page",
                  "operation": "clear"},
        "unicode": {"description": "Append another Unicode character by choosing its six hexadecimal code-point digits",
                    "operation": "unicode"},
    }
    for char in inventory["characters"]:
        code = ord(char)
        name = {" ": "SPACE", "\t": "TAB", "\n": "LINE FEED", "\r": "CARRIAGE RETURN"}.get(
            char, unicodedata.name(char, "UNNAMED CHARACTER"))
        candidates[f"char_{code:06X}"] = {
            "description": f"Append character {json.dumps(char, ensure_ascii=False)} (U+{code:04X} {name})",
            "operation": "append", "text": char,
        }
    for index, fragment in enumerate(inventory["fragments"]):
        candidates[f"fragment_{index}"] = {
            "description": "Append exact fragment " + json.dumps(fragment["text"], ensure_ascii=False),
            "operation": "append", "text": fragment["text"], "sources": fragment["sources"],
        }
    return candidates


def unicode_candidates(prefix: str) -> dict[str, dict]:
    """Encode any Unicode scalar in six hex digits; exclude only invalid code points."""
    candidates = {"back": {"description": "Return to the draft without appending a character",
                           "operation": "back"}}
    for digit in "0123456789ABCDEF":
        candidate = prefix + digit
        lower = int(candidate.ljust(6, "0"), 16)
        upper = int(candidate.ljust(6, "F"), 16)
        if not any(lower <= end and upper >= start
                   for start, end in ((0, 0xD7FF), (0xE000, 0x10FFFF))):
            continue
        candidates[f"hex_{digit}"] = {
            "description": f"Choose hex digit {digit}: U+{candidate}{'?' * (6 - len(candidate))}",
            "operation": "hex", "text": digit,
        }
    return candidates
