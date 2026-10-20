"""Instructions and options of the Choice questions one decision asks.

The text and pointer compositions are output encodings, not a task solver: no option depends on
previous choices or progress, and spaces, casing and punctuation are never inserted implicitly.
"""
from __future__ import annotations

import json
import unicodedata

TEXT_COMPOSITION_POLICY = "jev_exact_fragments_ascii_observed_unicode_scalar_v1"
POINTER_COMPOSITION_POLICY = "normalized_coordinate_hundreds_then_units_v1"

GUI_INSTRUCTIONS = (
    "Choose exactly one next GUI action to accomplish the task. Use the newest "
    "AX tree, SoM observation, task constraints, and execution feedback. "
    "For compact typing choices, the current state's focused_input identifies the field; "
    "the exact value remains in the Choice criterion. "
    "Click a text field to focus it, then select a suitable literal typing action or "
    "compose_text to build new input from exact words/characters. Composition inserts "
    "at the caret or replaces the current selection, like a normal type action. "
    "To replace a nonempty value, select all, then type to replace the selection. "
    "Clicking the field repeatedly does not enter text. Use drag/drop choices for "
    "Web drag targets. Wait only when timing matters "
    "or no offered action advances the task. Earlier marks are stale. Candidate "
    "descriptions identify actions, not recommendations."
)
GAME_INSTRUCTIONS = (
    "Choose exactly one next GUI action toward the game objective using the newest screenshot, "
    "current marks, declared controls and execution feedback. Choose a tap for one press or "
    "a short hold for continuous control; paired keys are held simultaneously. "
    "Use visible point/grid choices or compose a pointer location. For a drag, press the mouse, "
    "move on later observations, then release it. Do not reload or use browser shortcuts. "
    "Candidates describe available inputs, not recommended moves or hidden game state."
)
COMPOSE_INSTRUCTIONS = (
    "Compose the exact text to insert into focused_input to accomplish the user's task. "
    "Choose one edit of text_composition.draft, or finish when it is ready to type. "
    "Append choices add their exact text, with NO automatic spaces, capitalization, "
    "punctuation or correction. Select space/newline/tab explicitly when needed. "
    "The draft is not yet on the page. Finish inserts it at the existing caret or "
    "replaces the current selection; it does not clear the field or submit the form. "
    "You may delete, clear, or cancel the draft. Field constraints are context, not "
    "a restriction on intermediate drafts. Every choice remains available after repeats."
)
UNICODE_INSTRUCTIONS = (
    "Choose the next hexadecimal digit of the Unicode scalar to append to "
    "the draft. Use exactly six digits including leading zeros. Back returns "
    "to editing without adding a character."
)
GROUP_NOTE = (
    " This is one group of a larger candidate set. Select the best action "
    "in this group; it will be compared with the other group winners."
)


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


def coordinate_choices(base=None):
    """Cover every integer 0..1000 without a coarse grid or a >255-option question."""
    if base is None:
        return {str(value): {
            "description": f"Coordinate {'1000' if value == 1000 else f'{value}..{value + 99}'}",
            "value": value,
        } for value in range(0, 1001, 100)}
    return {str(value): {"description": f"Coordinate {value}", "value": value}
            for value in range(base, min(base + 100, 1001))}
