from __future__ import annotations
from realtime_framework.entity.events import EmptyInterruptedToolTurnError

# Protocol
"""Recover one unambiguous GUI action from a model response.

Models are asked for the canonical JSON form, but this boundary accepts common provider/model
wrappers without making the executor guess between distinct actions.
"""

import ast as ast
import json as json
import logging as logging
import re as re
import shlex as shlex
import xml.etree.ElementTree as ET
from collections.abc import Callable as Callable
from dataclasses import dataclass as dataclass
from typing import Any as Any

from realtime_framework.entity.actions import Action
from realtime_framework.entity.actions import ActionTranslator

logger = logging.getLogger(__name__)

_ACTION_KEYS = frozenset(
    {
        "action",
        "action_name",
        "action_type",
        "name",
        "tool",
        "tool_name",
        "command",
        "operation",
        "function",
        "function_call",
        "tool_call",
    }
)
_CANONICAL_ACTION_NAMES = frozenset(
    {
        "click",
        "right_click",
        "double_click",
        "triple_click",
        "mouse_down",
        "mouse_up",
        "move",
        "type",
        "key",
        "hold_key",
        "scroll",
        "drag",
        "wait",
        "done",
        "fail",
    }
)
_CANONICAL_FIELDS = frozenset(
    {
        "action",
        "target",
        "start",
        "end",
        "x",
        "y",
        "x1",
        "y1",
        "x2",
        "y2",
        "button",
        "modifier",
        "duration",
        "text",
        "key",
        "keys",
        "repeat",
        "dx",
        "dy",
    }
)
# A streaming action also names the frame it is based on.
_FRAMED_CANONICAL_FIELDS = _CANONICAL_FIELDS | {"frame"}
_FENCE_RE = re.compile(
    r"^\s*```(?:json|javascript|js|python|yaml|yml|xml)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_LABEL_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:(?:the\s+)?action(?:\s+is)?|tool|command|output|answer)\b"
    r"\s*[:=-]?\s*",
    re.IGNORECASE,
)
_INTENT_PREFIX_RE = re.compile(
    r"^\s*(?:i(?:'ll|\s+will)|we(?:'ll|\s+will|\s+should)|let'?s)\s+"
    r"(?:(?:now\s+)?(?:execute|perform|use|do|output|choose)\s+)?",
    re.IGNORECASE,
)
_NUMBER_RE = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
_TARGET_RE = r"(?:P[1-9]\d*|R[1-9]\d*(?:@\d{1,4},\d{1,4})?)"
_DURATION_RE = rf"({_NUMBER_RE})(?:\s*(?:s|sec|secs|second|seconds))?"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate action field: {key}")
        result[key] = value
    return result


def _literal_object(body):
    tree = ast.parse(body, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [ast.literal_eval(key) for key in node.keys]
            if len(set(keys)) != len(keys):
                raise ValueError("Duplicate literal action field")
    return ast.literal_eval(tree)


def _unwrap_action(value: object, depth: int = 0) -> dict | None:
    """Recognized transport envelopes only; never select the first of several actions."""
    if depth > 8:
        return None
    if isinstance(value, str):
        try:
            return _unwrap_action(json.loads(value, object_pairs_hook=unique_object), depth + 1)
        except ValueError:
            return None
    if isinstance(value, list):
        return _unwrap_action(value[0], depth + 1) if len(value) == 1 else None
    if not isinstance(value, dict):
        return None
    if _looks_like_action_object(value):
        return value
    if isinstance(value.get("type"), str) and value["type"] in _CANONICAL_ACTION_NAMES:
        return {"action": value["type"], **{k: v for k, v in value.items() if k != "type"}}
    containers = [
        key
        for key in ("actions", "tool_calls", "calls", "output", "response", "result", "content")
        if key in value
    ]
    if len(containers) == 1:
        return _unwrap_action(value[containers[0]], depth + 1)
    return None


@dataclass(frozen=True)
class _Candidate:
    text: str
    parse_mode: str


def _balanced_object_strings(text: str):
    """Yield balanced brace substrings while ignoring braces inside quoted strings."""
    stack: list[int] = []
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if not stack:
            if character == "{":
                stack.append(index)
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == "{":
            stack.append(index)
        elif character == "}" and stack:
            start = stack.pop()
            if not stack:
                yield text[start : index + 1]


def _looks_like_action_object(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    keys = {
        re.sub(
            r"[\s-]+",
            "_",
            re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key).strip()),
        ).lower()
        for key in value
    }
    return bool(keys & (_ACTION_KEYS | _CANONICAL_ACTION_NAMES))


def _is_canonical_json_action(value: object, fields: frozenset) -> bool:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("action"), str)
        or value["action"] not in _CANONICAL_ACTION_NAMES
    ):
        return False
    if not set(value) <= fields:
        return False
    for key in ("target", "start", "end"):
        if key in value and not isinstance(value[key], str):
            return False
    return True


def _structured_object_candidate(
    body: str,
    *,
    json_mode: str,
    literal_mode: str,
) -> _Candidate | None:
    try:
        value = json.loads(body, object_pairs_hook=unique_object)
    except (TypeError, ValueError):
        try:
            value = _literal_object(body)
        except (SyntaxError, ValueError, TypeError, RecursionError):
            return None
        mode = literal_mode
    else:
        mode = json_mode
    original = value
    value = _unwrap_action(original)
    if value is None:
        if isinstance(original, list) or (
            isinstance(original, dict)
            and set(original)
            & {"actions", "tool_calls", "calls", "output", "response", "result", "content"}
        ):
            return _Candidate(json.dumps(original, ensure_ascii=False), "invalid_envelope")
        return None
    try:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return _Candidate(payload, mode)


def _structured_candidates(text: str):
    # A whole structured payload is authoritative. Do not salvage one valid action from a
    # malformed/multi-action envelope by scanning its nested braces.
    whole = text.strip()
    fenced = _FENCE_RE.fullmatch(whole)
    if fenced:
        whole = fenced.group(1).strip()
    try:
        parsed = json.loads(whole, object_pairs_hook=unique_object)
    except ValueError:
        try:
            parsed = _literal_object(whole)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            parsed = None
    if isinstance(parsed, (dict, list, str)):
        candidate = _structured_object_candidate(
            whole, json_mode="structured_envelope", literal_mode="python_literal"
        )
        if candidate:
            yield candidate
        return
    if re.match(r"^(?:action|action_type|name|tool|function|type)\s*:", whole) and "\n" in whole:
        # Explicit YAML mappings only, with no constructors, anchors or duplicate keys.
        import yaml as yaml

        class ActionLoader(yaml.SafeLoader):
            def compose_node(self, parent, index):
                if self.check_event(yaml.AliasEvent) or self.peek_event().anchor is not None:
                    raise ValueError("YAML aliases/anchors are not action syntax")
                return super().compose_node(parent, index)

            def construct_mapping(self, node, deep=False):
                result = {}
                for key_node, value_node in node.value:
                    key = self.construct_object(key_node, deep=deep)
                    if not isinstance(key, str) or key in result:
                        raise ValueError("Duplicate or non-string action field")
                    result[key] = self.construct_object(value_node, deep=deep)
                return result

        try:
            payload = _unwrap_action(yaml.load(whole, Loader=ActionLoader))
            if payload is not None:
                yield _Candidate(json.dumps(payload, ensure_ascii=False), "yaml_mapping")
        except (yaml.YAMLError, ValueError, TypeError):
            pass
        return
    # XML tool/action wrappers are serialization, not executable markup.
    if whole.startswith("<") and "<!" not in whole:
        try:
            element = ET.fromstring(whole)
        except ET.ParseError:
            element = None
        if element is not None and element.tag in {
            "action",
            "tool_call",
            "function_call",
            "response",
            "final",
        }:
            if element.attrib.get("name") and not list(element):
                try:
                    arguments = json.loads(element.text or "{}")
                    payload = {"name": element.attrib["name"], "arguments": arguments}
                except ValueError:
                    return
                yield _Candidate(json.dumps(payload), "xml_tool_call")
            elif not list(element):
                yield from _structured_candidates(element.text or "")
                yield from _text_candidates(element.text or "")
            return
    fenced = _FENCE_RE.fullmatch(text)
    if fenced is not None:
        body = fenced.group(1).strip()
        if body:
            candidate = _structured_object_candidate(
                body,
                json_mode="markdown_fence",
                literal_mode="markdown_fence",
            )
            if candidate is not None:
                yield candidate

    sources = [(text, "embedded_json", "python_literal")]
    # Some models escape Markdown-special braces even though the contents are otherwise valid JSON,
    # e.g. \{"action":"click",...\}. Remove only those structural delimiter escapes; quoted text
    # and JSON string escapes remain untouched.
    unescaped_delimiters = re.sub(r"\\([{}\[\]])", r"\1", text)
    if unescaped_delimiters != text:
        sources.append(
            (
                unescaped_delimiters,
                "escaped_json_delimiters",
                "escaped_literal_delimiters",
            )
        )

    for source, json_mode, literal_mode in sources:
        for body in _balanced_object_strings(source):
            candidate = _structured_object_candidate(
                body,
                json_mode=json_mode,
                literal_mode=literal_mode,
            )
            if candidate is not None:
                yield candidate


def _literal_value(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        if isinstance(node, ast.Name):
            return node.id
        raise


def _call_name(function: ast.AST) -> str | None:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _function_call_object(text: str) -> dict | None:
    """Translate a plain ``hold_key(key=..., duration=...)``-style response."""
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expression, ast.Call):
        return None
    action_name = _call_name(expression.func)
    if not action_name or any(keyword.arg is None for keyword in expression.keywords):
        return None
    try:
        positional = [_literal_value(value) for value in expression.args]
        parameters = {
            str(keyword.arg): _literal_value(keyword.value) for keyword in expression.keywords
        }
    except (TypeError, ValueError):
        return None

    normalized_name = re.sub(
        r"[\s-]+",
        "_",
        re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", action_name.strip()),
    ).lower()
    if (
        normalized_name == "hotkey"
        and positional
        and all(isinstance(value, str) for value in positional)
    ):
        return {"action": "key", "keys": positional} if not parameters else None
    if normalized_name in {"click", "left_click", "tap"}:
        if positional and isinstance(positional[0], (int, float)):
            fields = ("x", "y", "button")
        else:
            fields = ("target", "button")
    else:
        fields = {
            "right_click": ("target",),
            "double_click": ("target",),
            "triple_click": ("target",),
            "move": ("x", "y", "duration"),
            "mouse_move": ("x", "y", "duration"),
            "mouse_down": ("target", "button", "duration"),
            "mouse_up": ("target", "button", "duration"),
            "type": ("text",),
            "type_text": ("text",),
            "input_text": ("text",),
            "write": ("text",),
            "typewrite": ("text",),
            "press_key": ("key", "repeat"),
            "key_press": ("key", "repeat"),
            "mousedown": ("target", "button"),
            "mouseup": ("target", "button"),
            "move_to": ("x", "y", "duration"),
            "key": ("key", "repeat"),
            "press": ("key", "repeat"),
            "keypress": ("key", "repeat"),
            "hotkey": ("key", "repeat"),
            "hold_key": ("key", "duration"),
            "holdkey": ("key", "duration"),
            "scroll": ("dx", "dy"),
            "drag": ("x1", "y1", "x2", "y2", "duration"),
            "wait": ("duration",),
            "done": (),
            "fail": (),
        }.get(normalized_name)
    if fields is None or len(positional) > len(fields):
        return None
    for field, value in zip(fields, positional):
        if field in parameters:
            return None
        parameters[field] = value
    return {"action": action_name, **parameters}


def _clean_text_candidate(value: str) -> str:
    value = value.strip().strip("`").strip()
    value = _LABEL_RE.sub("", value, count=1)
    value = _INTENT_PREFIX_RE.sub("", value, count=1)
    return (
        value.strip()
        if re.match(r"type\s", value, re.IGNORECASE)
        else value.strip().rstrip(".;").strip()
    )


def _shorthand_object(text: str) -> dict | None:
    """Translate conservative, whole-line action shorthands used by text models."""
    text = _clean_text_candidate(text)
    if not text:
        return None

    call = _function_call_object(text)
    if call is not None:
        return call

    # Tokenize quoted references/keys without changing text-to-type bytes. Assignment syntax
    # supports future clients without executing their code or inventing omitted arguments.
    try:
        words = shlex.split(text)
    except ValueError:
        words = []
    if words and words[0].lower() != "type":
        verb = words[0].lower().replace("-", "_")
        aliases = {
            "mousedown": "mouse_down",
            "mouseup": "mouse_up",
            "mousemove": "move",
            "press_key": "key",
            "key_press": "key",
        }
        verb = aliases.get(verb, verb)
        if verb in _CANONICAL_ACTION_NAMES or verb in {"press", "keypress", "hotkey", "tap"}:
            assignments = words[1:]
            if assignments and all("=" in word for word in assignments):
                params = {}
                for word in assignments:
                    key, value = word.split("=", 1)
                    if key in params or key not in _CANONICAL_FIELDS or key == "action":
                        return None
                    try:
                        params[key] = json.loads(value)
                    except ValueError:
                        params[key] = value
                return {"action": verb, **params}
            if (
                verb in {"mouse_down", "mouse_up", "move"}
                and len(words) == 2
                and re.fullmatch(_TARGET_RE, words[1], re.IGNORECASE)
            ):
                return {"action": verb, "target": words[1]}
            # Quotes are legal around a single reference or key; the subsequent whole-line
            # grammar still enforces argument count and rejects extra prose/actions.
            text = " ".join([verb, *words[1:]])

    terminal = re.fullmatch(r"(?:done|finish|finished|complete|completed)", text, re.IGNORECASE)
    if terminal:
        return {"action": "done"}
    if re.fullmatch(r"(?:fail|failed|give[ _-]?up)", text, re.IGNORECASE):
        return {"action": "fail"}

    match = re.fullmatch(rf"wait(?:\s+(?:for\s+)?{_DURATION_RE})?", text, re.IGNORECASE)
    if match:
        result = {"action": "wait"}
        if match.group(1) is not None:
            result["duration"] = float(match.group(1))
        return result

    match = re.fullmatch(
        rf"(click|left[ _-]?click|tap)(?:\s+(?:target\s*[=:]?\s*)?)"
        rf"({_TARGET_RE})(?:\s+(?:(?:with|using)\s+(?:the\s+)?(?:button\s+)?)?"
        rf"(left|middle|right)(?:\s+button)?)?",
        text,
        re.IGNORECASE,
    )
    if match:
        result = {"action": "click", "target": match.group(2)}
        if match.group(3):
            result["button"] = match.group(3).lower()
        return result

    match = re.fullmatch(
        rf"(right[ _-]?click|double[ _-]?click|triple[ _-]?click)\s+({_TARGET_RE})",
        text,
        re.IGNORECASE,
    )
    if match:
        return {
            "action": re.sub(r"[ -]+", "_", match.group(1).lower()),
            "target": match.group(2),
        }

    match = re.fullmatch(
        rf"(?:click|left[ _-]?click|tap)(?:\s+at)?\s*[([]?\s*({_NUMBER_RE})"
        rf"\s*[, ]\s*({_NUMBER_RE})\s*[)\]]?(?:\s+(left|middle|right))?",
        text,
        re.IGNORECASE,
    )
    if match:
        result = {
            "action": "click",
            "x": float(match.group(1)),
            "y": float(match.group(2)),
        }
        if match.group(3):
            result["button"] = match.group(3).lower()
        return result

    match = re.fullmatch(
        rf"(press\s+and\s+hold|hold[ _-]?key|holdkey|hold|key|keypress|press|hotkey)\s+"
        rf"(?:with\s+)?(?:key\s*[=:]\s*)?([A-Za-z0-9_+\-]+)"
        rf"(?:\s+(?:for\s+)?{_DURATION_RE})?",
        text,
        re.IGNORECASE,
    )
    if match:
        verb = match.group(1).lower().replace("-", "_").replace(" ", "_")
        duration = match.group(3)
        action_name = (
            "hold_key"
            if verb in {"press_and_hold", "hold_key", "holdkey", "hold"} or duration
            else "key"
        )
        result = {"action": action_name, "key": match.group(2)}
        if duration:
            result["duration"] = float(duration)
        return result

    match = re.fullmatch(r"type\s+(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        raw_value = match.group(1).strip()
        if raw_value.startswith(('"', "'")):
            try:
                value = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError):
                return None
            return {"action": "type", "text": value} if isinstance(value, str) else None
        if "\n" in raw_value:
            return None
        return {"action": "type", "text": raw_value}
    return None


def _text_candidates(text: str):
    seen: set[str] = set()
    values = [text]
    fence = _FENCE_RE.fullmatch(text)
    if fence:
        text = fence.group(1).strip()
        values = [text]
    # Structured objects, quoted payloads and XML must not also be interpreted as prose.
    if text.lstrip().startswith(("{", "[", '"', "<")):
        return
    if re.match(r"^(?:action|action_type|name|tool|function|type)\s*:", text) and "\n" in text:
        return
    values.extend(line for line in text.splitlines() if line.strip())
    values.extend(fragment for fragment in re.split(r"(?<=[.!?])\s+", text) if fragment.strip())
    for value in values:
        cleaned = _clean_text_candidate(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        parsed = _shorthand_object(cleaned)
        if parsed is not None:
            yield _Candidate(
                json.dumps(parsed, ensure_ascii=False, separators=(",", ":")),
                "function_call" if "(" in cleaned else "text_shorthand",
            )


_KNOWN_ACTION_NAMES = frozenset(
    {
        "click",
        "left_click",
        "right_click",
        "double_click",
        "triple_click",
        "mouse_down",
        "mouse_up",
        "type",
        "key",
        "press",
        "hotkey",
        "keypress",
        "hold_key",
        "scroll",
        "drag",
        "move",
        "mouse_move",
        "wait",
        "done",
        "fail",
        "give_up",
    }
)


def _load_action_object(value: str) -> dict | None:
    try:
        parsed = json.loads(value, object_pairs_hook=unique_object)
    except (ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalized_action_name(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = re.split(r"[./]", value.strip())[-1]
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    name = re.sub(r"[\s-]+", "_", name).lower()
    aliases = {
        "tap": "click",
        "click_at": "click",
        "leftclick": "left_click",
        "rightclick": "right_click",
        "doubleclick": "double_click",
        "tripleclick": "triple_click",
        "mousedown": "mouse_down",
        "mouseup": "mouse_up",
        "mousemove": "mouse_move",
        "press_key": "key",
        "key_press": "key",
        "send_keys": "key",
        "type_text": "type",
        "input_text": "type",
        "typewrite": "type",
        "write": "type",
        "move_to": "move",
        "holdkey": "hold_key",
        "press_and_hold": "hold_key",
        "finish": "done",
        "finished": "done",
        "complete": "done",
        "completed": "done",
    }
    return aliases.get(name, name)


def _normalized_field_name(value: object) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value).strip())
    return re.sub(r"[\s-]+", "_", name).lower()


def _normalize_action_key_shorthand(data: dict) -> dict | None:
    """Expand shorthand ``{"click": "P1"}`` action objects.

    Some text checkpoints follow the one-action constraint but serialize the verb as the
    object's key instead of putting it in an ``action`` field.  Accept that form only when
    exactly one known verb key is present, then translate its value conservatively into the
    canonical field for that verb.  Keeping this at the adapter boundary leaves the core
    action and executor contracts unchanged.
    """
    lowered = {_normalized_field_name(key): value for key, value in data.items()}
    if _normalized_action_name(lowered.get("type")) in _KNOWN_ACTION_NAMES and len(lowered) > 1:
        return data
    selector_keys = {
        "action",
        "action_name",
        "action_type",
        "name",
        "tool",
        "tool_name",
        "command",
        "operation",
        "function",
        "function_call",
        "tool_call",
    }
    if set(lowered) & selector_keys:
        return data

    action_keys = [key for key in lowered if _normalized_action_name(key) in _KNOWN_ACTION_NAMES]
    if len(action_keys) != 1:
        return data

    action_key = action_keys[0]
    action_name = _normalized_action_name(action_key)
    payload = lowered.pop(action_key)
    parameters: dict = {}

    if isinstance(payload, dict):
        parameters = {_normalized_field_name(key): value for key, value in payload.items()}
    elif action_name in {
        "click",
        "left_click",
        "right_click",
        "double_click",
        "triple_click",
        "mouse_down",
        "mouse_up",
        "move",
        "mouse_move",
    }:
        if isinstance(payload, str):
            parameters["target"] = payload
        elif isinstance(payload, (list, tuple)) and len(payload) == 2:
            parameters["coordinate"] = list(payload)
        elif payload is not None:
            return data
    elif action_name == "type":
        if not isinstance(payload, str):
            return data
        parameters["text"] = payload
    elif action_name in {"key", "press", "hotkey", "keypress", "hold_key"}:
        if isinstance(payload, str):
            parameters["key"] = payload
        elif isinstance(payload, (list, tuple)):
            parameters["keys"] = list(payload)
        else:
            return data
    elif action_name == "scroll":
        if isinstance(payload, (list, tuple)) and len(payload) == 2:
            parameters.update({"dx": payload[0], "dy": payload[1]})
        elif payload is not None:
            return data
    elif action_name == "drag":
        if isinstance(payload, (list, tuple)) and len(payload) == 2:
            parameters.update({"start": payload[0], "end": payload[1]})
        elif payload is not None:
            return data
    elif action_name == "wait":
        if isinstance(payload, (int, float)) and not isinstance(payload, bool):
            parameters["duration"] = payload
        elif payload not in (None, True):
            return data
    elif action_name in {"done", "fail", "give_up"}:
        if payload not in (None, True):
            return data
    elif payload is not None:
        return data

    if set(parameters) & set(lowered):
        return data
    return {"action": action_name, **lowered, **parameters}


def _normalize_action_object(data: dict) -> dict | None:
    """Flatten common JSON/tool envelopes into the canonical action object."""
    data = _normalize_action_key_shorthand(data)
    if data is None:
        return None
    lowered = {_normalized_field_name(key): value for key, value in data.items()}
    if len(lowered) != len(data):
        return None
    if "type" in lowered and _normalized_action_name(lowered["type"]) in _KNOWN_ACTION_NAMES:
        if "action" in lowered and _normalized_action_name(
            lowered["action"]
        ) != _normalized_action_name(lowered["type"]):
            return None
        lowered.setdefault("action", lowered.pop("type"))
    selectors = [
        lowered[k]
        for k in (
            "action",
            "action_name",
            "action_type",
            "name",
            "tool",
            "tool_name",
            "command",
            "operation",
        )
        if isinstance(lowered.get(k), str)
    ]
    if len({_normalized_action_name(value) for value in selectors}) > 1:
        return None
    nested = None
    if (
        sum(
            isinstance(lowered.get(key), dict) for key in ("function_call", "function", "tool_call")
        )
        > 1
    ):
        return None
    for key in ("function_call", "function", "tool_call"):
        if isinstance(lowered.get(key), dict):
            nested = lowered[key]
            break
    action_value = next(
        (
            lowered[key]
            for key in (
                "action",
                "action_name",
                "action_type",
                "name",
                "tool",
                "tool_name",
                "command",
                "operation",
            )
            if key in lowered
        ),
        None,
    )
    if isinstance(action_value, dict):
        nested = action_value
        action_value = None
    top_action_name = _normalized_action_name(action_value)
    if nested is not None:
        nested_lowered = {_normalized_field_name(key): value for key, value in nested.items()}
        nested_selectors = [
            nested_lowered[key]
            for key in (
                "action",
                "action_name",
                "action_type",
                "name",
                "tool",
                "tool_name",
                "command",
                "operation",
            )
            if isinstance(nested_lowered.get(key), str)
        ]
        if len({_normalized_action_name(value) for value in nested_selectors}) > 1:
            return None
        nested_action_value = next(
            (
                nested_lowered[key]
                for key in (
                    "action",
                    "action_name",
                    "action_type",
                    "name",
                    "tool",
                    "tool_name",
                    "command",
                    "operation",
                )
                if key in nested_lowered
            ),
            None,
        )
        nested_action_name = _normalized_action_name(nested_action_value)
        if (
            top_action_name in _KNOWN_ACTION_NAMES
            and nested_action_name in _KNOWN_ACTION_NAMES
            and top_action_name != nested_action_name
        ):
            return None
        action_name = (
            nested_action_name
            if nested_action_name in _KNOWN_ACTION_NAMES
            else top_action_name or nested_action_name
        )
    else:
        nested_lowered = {}
        action_name = top_action_name

    if action_name is None:
        return None

    reserved = {
        "action",
        "action_name",
        "action_type",
        "name",
        "tool",
        "tool_name",
        "command",
        "operation",
        "type",
        "id",
        "call_id",
        "function",
        "function_call",
        "tool_call",
        "arguments",
        "parameters",
        "params",
        "args",
        "input",
        "tool_input",
    }
    parameters = {key: value for key, value in lowered.items() if key not in reserved}
    nested_parameters = {key: value for key, value in nested_lowered.items() if key not in reserved}
    if any(
        key in parameters and parameters[key] != value for key, value in nested_parameters.items()
    ):
        return None
    parameters.update(nested_parameters)
    containers = [
        source[key]
        for source in (nested_lowered, lowered)
        for key in ("arguments", "parameters", "params", "args", "input", "tool_input")
        if key in source
    ]
    if len(containers) > 1:
        return None
    parameter_container = next(
        (
            source[key]
            for source in (nested_lowered, lowered)
            for key in (
                "arguments",
                "parameters",
                "params",
                "args",
                "input",
                "tool_input",
            )
            if key in source
        ),
        None,
    )
    if isinstance(parameter_container, str):
        try:
            parameter_container = json.loads(parameter_container, object_pairs_hook=unique_object)
        except (TypeError, ValueError):
            try:
                parameter_container = _literal_object(parameter_container)
            except (SyntaxError, ValueError, TypeError, RecursionError):
                return None
    if parameter_container is not None:
        if not isinstance(parameter_container, dict):
            return None
        supplied = {
            _normalized_field_name(key): value for key, value in parameter_container.items()
        }
        if any(key in parameters and parameters[key] != value for key, value in supplied.items()):
            return None
        parameters.update(supplied)
    if "action" in parameters:
        inner_name = _normalized_action_name(parameters.pop("action"))
        if action_name in _KNOWN_ACTION_NAMES and inner_name != action_name:
            return None
        action_name = inner_name

    field_aliases = {
        "coordinates": "coordinate",
        "coords": "coordinate",
        "position": "coordinate",
        "location": "coordinate",
        "point": "coordinate",
        "target_id": "target",
        "reference": "target",
        "ref": "target",
        "duration_seconds": "duration",
        "seconds": "duration",
        "text_to_type": "text",
        "delta_x": "dx",
        "delta_y": "dy",
    }
    normalized_parameters = {}
    for key, value in parameters.items():
        key = field_aliases.get(key, key)
        if key in normalized_parameters and normalized_parameters[key] != value:
            return None
        normalized_parameters[key] = value
    parameters = normalized_parameters
    duration_ms = parameters.pop("duration_ms", None)
    if "duration" not in parameters and isinstance(duration_ms, (int, float, str)):
        try:
            parameters["duration"] = float(duration_ms) / 1000
        except ValueError:
            parameters["duration"] = duration_ms
    if action_name == "type" and "text" not in parameters and "value" in parameters:
        parameters["text"] = parameters.pop("value")
    if isinstance(parameters.get("keys"), str) and "key" not in parameters:
        parameters["key"] = parameters.pop("keys")
    coordinate = parameters.get("coordinate")
    if isinstance(coordinate, dict):
        coordinate = {_normalized_field_name(key): value for key, value in coordinate.items()}
        if "x" in coordinate and "y" in coordinate:
            parameters.pop("coordinate")
            parameters.setdefault("x", coordinate["x"])
            parameters.setdefault("y", coordinate["y"])
    if action_name == "drag":
        for field, x_field, y_field in (
            ("start", "x1", "y1"),
            ("end", "x2", "y2"),
        ):
            point = parameters.get(field)
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                parameters.pop(field)
                parameters.setdefault(x_field, point[0])
                parameters.setdefault(y_field, point[1])
            elif isinstance(point, dict):
                point = {_normalized_field_name(key): value for key, value in point.items()}
                if "x" in point and "y" in point:
                    parameters.pop(field)
                    parameters.setdefault(x_field, point["x"])
                    parameters.setdefault(y_field, point["y"])

    for key in (
        "x",
        "y",
        "x1",
        "y1",
        "x2",
        "y2",
        "dx",
        "dy",
        "duration",
        "repeat",
    ):
        value = parameters.get(key)
        if not isinstance(value, str):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        parameters[key] = int(number) if number.is_integer() else number
    for key in ("button", "modifier"):
        if isinstance(parameters.get(key), str):
            parameters[key] = parameters[key].strip().lower()
    return {"action": action_name, **parameters}


def _action_object(content: str) -> dict | None:
    if not content:
        return None
    loaded = _load_action_object(content)
    data = _normalize_action_object(loaded) if loaded is not None else None
    if data is None or data.get("action") not in _KNOWN_ACTION_NAMES:
        return None
    return data


def text_action(body: str, *, framed: bool) -> tuple[Action | None, object]:
    """The action one body of text output reads as and, ``framed`` (a streaming action), the value
    of its frame field: the frame it is based on. Unframed, a frame field stays one of the action's
    fields, which no action takes."""
    data = _action_object(body)
    if data is None:
        return None, None
    reference = data.pop("frame", None) if framed else None
    return ActionTranslator.from_json(data), reference


def _action_key(action: Action) -> str:
    parameters = dict(action.parameters)
    defaults = {
        "click": {"button": "left"},
        "hold_key": {"duration": 1},
        "key": {"repeat": 1},
    }.get(action.action_type.value, {})
    for key, value in defaults.items():
        parameters.setdefault(key, value)
    for key, value in tuple(parameters.items()):
        if isinstance(value, float) and value.is_integer():
            parameters[key] = int(value)
    return json.dumps(
        {"action": action.action_type.value, "parameters": parameters},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


from realtime_framework.entity.harness import Decision as ActionOutputResolution


@dataclass(frozen=True)
class ResolvedOutput:
    """The one action resolved from model output, and what the parse made of each body it was
    given that parsed, in the order they were tried: the one the action is, and all of them."""

    decision: ActionOutputResolution
    chosen: Any = None  # the parse whose action the decision holds; None unless it is valid
    parses: tuple = ()


def resolve_action_output(
    content: str,
    parse: Callable[[str], Any],
    *,
    framed: bool = False,
) -> ResolvedOutput:
    """Resolve one unambiguous action from canonical or commonly wrapped model output.
    ``parse(body)`` is None for a body that does not parse, otherwise what it made of the body,
    whose ``action`` is the action, None when the body parsed to none, as an action naming marks
    its frame lacks. ``framed``: the canonical form also carries the frame field of a streaming
    action."""
    text = (content or "").strip()
    if len(text) > 1_000_000:
        return ResolvedOutput(
            ActionOutputResolution(
                action=Action.wait(),
                action_text=None,
                valid=False,
                protocol_failure="action_output_too_large",
            )
        )
    if not text:
        return ResolvedOutput(
            ActionOutputResolution(
                action=Action.wait(),
                action_text=None,
                valid=False,
                protocol_failure="empty_action_output",
            )
        )
    parses = []

    def safe_parse(body):
        """The parse of a body that parsed to an action, else None."""
        try:
            parsed = parse(body)
        except (ValueError, TypeError, KeyError, RecursionError):
            return None
        if parsed is None:
            return None
        parses.append(parsed)
        return parsed if parsed.action is not None else None

    chosen = safe_parse(text)
    if chosen is not None:
        try:
            complete_value = json.loads(text)
        except (TypeError, ValueError):
            complete_value = None
        canonical = _is_canonical_json_action(
            complete_value, _FRAMED_CANONICAL_FIELDS if framed else _CANONICAL_FIELDS
        )
        decision = ActionOutputResolution(
            action=chosen.action,
            action_text=text,
            valid=True,
            parse_mode="canonical_json" if canonical else "json_envelope",
            candidate_count=1,
        )
        return ResolvedOutput(decision, chosen, tuple(parses))

    # A complete literal call/shorthand is authoritative, including JSON *inside* text to type.
    whole_action = _shorthand_object(text)
    if whole_action is not None:
        serialized = json.dumps(whole_action, ensure_ascii=False)
        chosen = safe_parse(serialized)
        if chosen is not None:
            decision = ActionOutputResolution(
                action=chosen.action,
                action_text=serialized,
                valid=True,
                parse_mode="function_call"
                if _function_call_object(_clean_text_candidate(text)) is not None
                else "text_shorthand",
                candidate_count=1,
            )
            return ResolvedOutput(decision, chosen, tuple(parses))

    candidates: list[tuple[_Candidate, Any]] = []
    seen_text: set[str] = set()
    try:
        decoded = (*_structured_candidates(text), *_text_candidates(text))
    except (ValueError, TypeError, RecursionError):
        decoded = ()
    rejected_candidate = False
    for candidate in decoded:
        if candidate.text == text or candidate.text in seen_text:
            continue
        seen_text.add(candidate.text)
        recovered = safe_parse(candidate.text)
        if recovered is not None:
            candidates.append((candidate, recovered))
        else:
            rejected_candidate = True

    unique: dict[str, tuple[_Candidate, Any]] = {}
    for candidate, recovered in candidates:
        unique.setdefault(_action_key(recovered.action), (candidate, recovered))
    if len(unique) == 1 and not rejected_candidate:
        candidate, chosen = next(iter(unique.values()))
        decision = ActionOutputResolution(
            action=chosen.action,
            action_text=candidate.text,
            valid=True,
            parse_mode=candidate.parse_mode,
            candidate_count=len(unique),
        )
        return ResolvedOutput(decision, chosen, tuple(parses))

    failure = (
        "ambiguous_action_output"
        if len(unique) > 1 or (unique and rejected_candidate)
        else "invalid_action_output"
    )
    logger.warning(
        "Could not recover one unambiguous action (%s): %s",
        failure,
        text[:200],
    )
    decision = ActionOutputResolution(
        action=Action.wait(),
        action_text=None,
        valid=False,
        protocol_failure=failure,
        candidate_count=len(unique),
    )
    return ResolvedOutput(decision, None, tuple(parses))


def resolve_structured_output(content: str, parse: Callable[[str], Any]) -> ResolvedOutput:
    """The one action of a function call or a Decision's choice, whose channel hands over the
    action object itself: ``parse(content)`` as resolve_action_output's ``parse`` reads it, and
    nothing is recovered from content that is no action."""
    try:
        parsed = parse(content)
    except (ValueError, TypeError, KeyError, RecursionError):
        parsed = None
    parses = () if parsed is None else (parsed,)
    if parsed is None or parsed.action is None:
        decision = ActionOutputResolution(
            action=Action.wait(),
            action_text=None,
            valid=False,
            protocol_failure="invalid_action_output",
        )
        return ResolvedOutput(decision, None, parses)
    decision = ActionOutputResolution(
        action=parsed.action,
        action_text=content,
        valid=True,
        parse_mode="canonical_json",
        candidate_count=1,
    )
    return ResolvedOutput(decision, parsed, parses)


# Native tool output
"""Native usage clock for an acknowledged GUI function call.

The tool is acknowledged as queued, never falsely as executed. Only terminal native usage
authorizes the action. Audio transcription cannot supply an action.
"""

from realtime_framework.entity.events import FunctionCall
from realtime_framework.entity.events import GenerationBoundary
from realtime_framework.entity.model import GenerationOutput
from realtime_framework.entity.events import TextDelta
from realtime_framework.entity.events import UsageSnapshot
from realtime_framework.entity.actions import canonical_function_action
from realtime_framework.entity.usage import TOKEN_NORMALIZATION_SCHEMA


class FunctionCallTurnCollector:
    """Collect one native function-call turn for drivers with action_channel="function_call".

    The driver's terminal UsageSnapshot must declare ``output_accounting`` and
    ``native_response_usage_exact`` with an integer ``output_tokens`` clock total. Audio and
    response-detail fields are optional; Gemini Live reports all of them.
    """

    def __init__(self, agent):
        self.agent = agent
        self.visible = []
        self.calls = []
        self.reasoning = []
        self.usage = None
        self.chunks = []
        self.complete = False
        self.interrupted = False
        self.received_native_output = False
        self.terminal_usage_received = False
        self.submission_boundary = "native_turn_complete"

    def consume(self, event):
        received = {
            "received_wall_time_ns": self.agent.clock.time_ns(),
            "received_monotonic_ns": self.agent.clock.monotonic_ns(),
            "receive_timestamp_source": "collector_consume",
        }
        if isinstance(event, UsageSnapshot):
            self.usage = event.usage
            self.chunks.append(
                {
                    "sequence": event.sequence,
                    "usage": event.usage,
                    "transport_event": event.raw_type,
                    **received,
                }
            )
        elif isinstance(event, FunctionCall):
            self.calls.append(event)
            self.chunks.append(
                {
                    "sequence": event.sequence,
                    "function_call": {
                        "id": event.call_id,
                        "name": event.name,
                        "args": event.arguments,
                    },
                    "transport_event": event.raw_type,
                    **received,
                }
            )
        elif isinstance(event, TextDelta) and event.channel == "visible":
            self.visible.append(event.text)
            self.chunks.append(
                {
                    "sequence": event.sequence,
                    "content": event.text,
                    "channel": event.channel,
                    "transport_event": event.raw_type,
                    "source": "output_audio_transcription",
                    **received,
                }
            )
        elif isinstance(event, GenerationBoundary):
            self.received_native_output |= bool(event.metadata.get("native_output_received"))
            self.complete |= event.kind == "turn_complete"
            self.interrupted |= event.kind == "interrupted"
            if event.kind == "turn_complete":
                self.terminal_usage_received = bool(event.metadata.get("terminal_usage_received"))
                self.submission_boundary = event.metadata.get(
                    "submission_boundary", self.submission_boundary
                )

    def finish(self):
        if (
            self.complete
            and self.interrupted
            and self.terminal_usage_received
            and not self.calls
            and not self.visible
            and not self.reasoning
            and not self.received_native_output
            and self.usage
            and self.usage.get("output_accounting")
            and self.usage.get("native_response_usage_exact")
            and type(self.usage.get("output_tokens")) is int
            and self.usage["output_tokens"] == 0
            and self.usage.get("native_audio_response_tokens", 0) == 0
        ):
            # The provider sometimes interrupts before starting a response, then reports only
            # prompt usage. Reconnecting cannot discard a decision in this narrowly proven case.
            raise EmptyInterruptedToolTurnError("Provider interrupted before generating any output")
        # A delivered function call already fixes the GUI decision. An interruption after that
        # call need not invalidate it if the terminal message supplies exact final usage. The
        # driver rejects explicit tool cancellation and rebuilds interrupted provider history.
        # Never accept a partial turn, an ambiguous interrupted choice, or stale usage.
        if not self.complete or (
            self.interrupted and (len(self.calls) != 1 or not self.terminal_usage_received)
        ):
            raise RuntimeError("native tool accounting requires a complete uninterrupted turn")
        if not self.usage or not self.usage.get("output_accounting"):
            raise RuntimeError(
                "missing terminal native tool output usage; transcript estimates are forbidden"
            )
        if not self.usage.get("native_response_usage_exact"):
            raise RuntimeError(
                "provider response token count is neither reported nor exactly derivable"
            )
        # The driver has already removed native audio from this clock total. Native audio is
        # forced transport output, not semantic GUI-action generation.
        total = self.usage["output_tokens"]
        audio = self.usage.get("native_audio_response_tokens", 0)
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise RuntimeError("invalid native tool output usage")
        call_count = len(self.calls)
        calls = [
            {"id": call.call_id, "name": call.name, "arguments": call.arguments}
            for call in self.calls
        ]
        call = self.calls[0] if call_count == 1 else None
        function_validation_error = None
        protocol_failure = None
        if call is None:
            # The benchmark accepts one top-level GUI action per observation. Live itself permits
            # zero or multiple tool calls, so this is model protocol behavior, not a provider-run
            # failure. Preserve the charged turn and let the shared harness execute WAIT.
            protocol_failure = "native_function_call_count"
            text = ""
        else:
            try:
                action = canonical_function_action(call.name, call.arguments)
            except ValueError as exc:
                # A call its function's schema rejects is no action: the turn stays, complete and
                # charged, and the shared harness executes WAIT. A malformed action is model
                # behavior, not a transport/accounting failure that should invalidate the
                # benchmark cell.
                function_validation_error = str(exc)
                action = {
                    "action": call.name,
                    **(call.arguments if isinstance(call.arguments, dict) else {}),
                }
                protocol_failure = "invalid_action_output"
            text = json.dumps(action, ensure_ascii=False)
        fields = {
            "provider_token_normalization_schema": TOKEN_NORMALIZATION_SCHEMA,
            "provider_token_semantics": self.usage["output_accounting"],
            "native_received_output_tokens": total,
            "native_received_output_tokens_source": self.usage.get(
                "native_output_tokens_source", "provider_usage.output_tokens"
            ),
            "native_output_total_exact": True,
            "provider_output_tokens": total,
            "provider_billed_output_tokens": total + audio,
            "provider_text_output_tokens": total,
            "provider_reasoning_output_tokens": (self.usage.get("output_token_details") or {}).get(
                "reasoning_tokens"
            ),
            "provider_reasoning_output_tokens_source": (
                "provider_usage.output_token_details.reasoning_tokens"
            ),
            "provider_visible_output_tokens": self.usage.get("native_non_audio_response_tokens"),
            "provider_visible_output_tokens_source": (
                "provider_usage.native_non_audio_response_tokens"
            ),
            "provider_native_response_output_tokens": self.usage.get("native_response_tokens"),
            "provider_native_response_output_tokens_source": self.usage.get(
                "native_response_tokens_source"
            ),
            "provider_audio_output_tokens": audio,
            "provider_audio_output_tokens_source": self.usage.get(
                "native_audio_tokens_source", "provider_usage.native_audio_response_tokens"
            ),
            "provider_native_total_tokens": self.usage.get("native_total_tokens"),
            "submission_boundary": self.submission_boundary,
            "native_turn_interrupted": self.interrupted,
            "action_serialization": (
                "native_function_call" if call_count == 1 else "native_function_calls"
            ),
            "function_call_id": call.call_id if call is not None else None,
            "function_call_count": call_count,
            "function_call_ids": [value["id"] for value in calls],
            "function_call_names": [value["name"] for value in calls],
            "thinking_visibility": "provider_internal",
            "function_call_received": bool(calls),
            "function_validation_error": function_validation_error,
            "protocol_failure": protocol_failure,
            "provider_usage_missing_reason": None,
        }
        return GenerationOutput(
            text=text,
            usage=self.usage,
            token_fields=fields,
            output_chunks=tuple(self.chunks),
        )
