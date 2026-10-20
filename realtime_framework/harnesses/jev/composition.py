"""The inventory of exact text composition (output encoding) for choice models.

The composition questions themselves are in providers/decisions/questions.py.
"""
from __future__ import annotations

import re
import string


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
