"""Read-only, frame-scoped accessibility tree capture for text-observation models."""

from __future__ import annotations

import re

MAX_AX_NODES = 350
MAX_AX_CHARS = 16000


def capture_ax_frames(client) -> dict:
    frames, texts, warnings, sessions = [], [], [], []
    seen = set()
    frame_children = {}

    def call(method, params=None, session=None):
        response = client.call(method, params, timeout=8, session_id=session)
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        return response.get("result", {})

    def discover_frames(root_frame_id, session):
        # Page.getFrameTree omits remote children on some Chrome versions. DOM exposes
        # their frame IDs on the iframe owners, including owners inside local subframes.
        root = call("DOM.getDocument", {"depth": -1, "pierce": True}, session)["root"]

        def walk(node, owner):
            child_id = node.get("frameId") if node.get("nodeName") in {"IFRAME", "FRAME"} else None
            if child_id:
                frame_children.setdefault(owner, []).append(child_id)
                if node.get("contentDocument"):
                    walk(node["contentDocument"], child_id)
                return
            for child in node.get("children", []) + node.get("shadowRoots", []):
                walk(child, owner)
        walk(root, root_frame_id)

    def visible_owner(frame_id, session):
        owner = call("DOM.getFrameOwner", {"frameId": frame_id}, session)
        remote = call("DOM.resolveNode", {"backendNodeId": owner["backendNodeId"]}, session)
        object_id = remote["object"]["objectId"]
        try:
            result = call("Runtime.callFunctionOn", {
                "objectId": object_id, "returnByValue": True,
                "functionDeclaration": """function() {
                  for (let el = this; el; el = el.parentElement) {
                    const s = el.ownerDocument.defaultView.getComputedStyle(el);
                    if (el.hidden || el.getAttribute('aria-hidden') === 'true' ||
                        s.display === 'none' || s.visibility !== 'visible' || s.opacity === '0')
                      return false;
                  }
                  const r = this.getBoundingClientRect(), w = this.ownerDocument.defaultView;
                  return r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 &&
                         r.top < w.innerHeight && r.left < w.innerWidth;
                }""",
            }, session)
            if result.get("exceptionDetails"):
                raise RuntimeError("frame visibility evaluation failed")
            return result.get("result", {}).get("value") is True
        finally:
            call("Runtime.releaseObject", {"objectId": object_id}, session)

    def visit(tree, session=None, parent_id=None):
        frame_id = tree["frame"]["id"]
        if frame_id in seen:
            return
        seen.add(frame_id)
        if parent_id:
            try:
                if not visible_owner(frame_id, session):
                    return
            except Exception as exc:
                warnings.append(f"ax_frame_visibility_unavailable:{frame_id}:{str(exc)[:160]}")
                return
        try:
            try:
                nodes = call("Accessibility.getFullAXTree", {"frameId": frame_id}, session)["nodes"]
            except RuntimeError:
                # Out-of-process iframes have their own target/session. Their target ID is
                # the frame ID; keep all node identities scoped to that frame.
                session = call("Target.attachToTarget", {
                    "targetId": frame_id, "flatten": True,
                })["sessionId"]
                sessions.append(session)
                tree = call("Page.getFrameTree", session=session)["frameTree"]
                discover_frames(frame_id, session)
                nodes = call("Accessibility.getFullAXTree", {"frameId": frame_id}, session)["nodes"]
            rendered = render_ax_tree(nodes)
            frames.append({"frame_id": frame_id, "parent_frame_id": parent_id,
                           "node_count": len(nodes), "text": rendered,
                           "truncated": "[AX tree truncated]" in rendered})
            texts.extend(ax_text_records(nodes, frame_id))
        except Exception as exc:
            warnings.append(f"ax_frame_unavailable:{frame_id}:{str(exc)[:160]}")
        children = {child["frame"]["id"]: child for child in tree.get("childFrames", [])}
        for child_id in frame_children.get(frame_id, []):
            children.setdefault(child_id, {"frame": {"id": child_id}})
        for child in children.values():
            visit(child, session, frame_id)

    with client.transaction():
        try:
            tree = call("Page.getFrameTree")["frameTree"]
            discover_frames(tree["frame"]["id"], None)
            visit(tree)
        finally:
            for session in reversed(sessions):
                try:
                    call("Target.detachFromTarget", {"sessionId": session})
                except Exception:
                    pass
    return {
        "ax_tree": "\n".join(f"[Frame {f['frame_id']} parent={f['parent_frame_id']}]\n{f['text']}"
                             for f in frames),
        # Full accessible strings are captured before display rendering limits are applied.
        "ax_text": texts,
        "ax_capture": {"frames": [{k: v for k, v in f.items() if k != "text"} for f in frames],
                       "warnings": warnings, "complete": not warnings and
                       not any(f["truncated"] for f in frames)},
    }


def ax_text_records(nodes: list[dict], frame_id: str) -> list[dict]:
    """Preserve complete accessible text for literal input choices, before rendering caps."""
    records = []
    for node in nodes:
        role = (node.get("role") or {}).get("value", "")
        if node.get("ignored") or role in {"InlineTextBox", "generic", "none", "Ignored"}:
            continue
        for field in ("name", "value", "description"):
            value = (node.get(field) or {}).get("value")
            if isinstance(value, str) and value.strip():
                records.append({"text": value, "role": role, "field": field,
                                "frame_id": frame_id, "node_id": str(node.get("nodeId"))})
    return records


def _value(node: dict, field: str) -> str:
    value = node.get(field)
    if isinstance(value, dict):
        value = value.get("value")
    return re.sub(r"\s+", " ", str(value or "")).strip()[:240]


def render_ax_tree(nodes: list[dict]) -> str:
    """Keep accessible names, values and state in tree order, without DOM internals."""
    by_id = {str(node.get("nodeId")): node for node in nodes if isinstance(node, dict)}
    if not by_id:
        return ""
    children = {str(child) for node in by_id.values() for child in node.get("childIds", [])}
    roots = [key for key in by_id if key not in children] or list(by_id)[:1]
    lines: list[str] = []
    seen: set[str] = set()
    length = 0
    truncated = False

    def visit(node_id: str, depth: int) -> None:
        nonlocal length, truncated
        if node_id in seen or len(lines) >= MAX_AX_NODES or length >= MAX_AX_CHARS:
            return
        node = by_id.get(node_id)
        if node is None:
            return
        seen.add(node_id)
        role = _value(node, "role")
        name = _value(node, "name")
        value = _value(node, "value")
        properties = {
            str(prop.get("name")): _value(prop, "value")
            for prop in node.get("properties", []) if isinstance(prop, dict)
        }
        state = " ".join(
            f"{key}={properties[key]}" for key in
            ("focused", "disabled", "checked", "selected", "expanded", "required")
            if properties.get(key) not in (None, "", "false")
        )
        meaningful = bool(not node.get("ignored") and role not in
                          {"InlineTextBox", "generic", "none", "Ignored"}
                          and (name or value or state or role in {"RootWebArea", "dialog"}))
        if meaningful:
            line = "  " * min(depth, 12) + role
            if name:
                line += f" {name!r}"
            if value and value != name:
                line += f" value={value!r}"
            if state:
                line += f" {state}"
            if length + len(line) + 1 <= MAX_AX_CHARS:
                lines.append(line)
                length += len(line) + 1
            else:
                truncated = True
            if any(len(str((node.get(key) or {}).get("value", ""))) > 240
                   for key in ("name", "value")):
                truncated = True
        for child in node.get("childIds", []):
            visit(str(child), depth + int(meaningful))

    for root in roots:
        visit(root, 0)
    if truncated or len(seen) < len(by_id):
        lines.append("[AX tree truncated]")
    return "\n".join(lines)
