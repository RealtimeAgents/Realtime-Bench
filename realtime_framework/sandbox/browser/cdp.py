from __future__ import annotations

# Cdp
"""Persistent, process-local Chrome DevTools Protocol connections.

Chrome in the sandbox uses host networking, so the harness can keep one WebSocket open for the
episode instead of launching an ``apptainer exec`` helper for every clock, input, and DOM probe.
"""

import hashlib as hashlib
import json as json
import threading as threading
import time as time
import urllib.request
from collections import deque as deque
from contextlib import contextmanager as contextmanager
from typing import Any as Any

from websockets.sync.client import connect as connect

from realtime_framework.sandbox.budget import Deadline as Deadline
from realtime_framework.sandbox.budget import bounded as bounded


def page_targets(port: int, timeout: float = 3) -> list[dict]:
    url = f"http://127.0.0.1:{int(port)}/json"
    with urllib.request.urlopen(url, timeout=bounded(timeout)) as response:
        values = json.load(response)
    return [
        value
        for value in values
        if value.get("type") == "page" and value.get("webSocketDebuggerUrl")
    ]


class PersistentCDPClient:
    """A serialized CDP session bound to one page target."""

    def __init__(self, port: int, *, target_id: str | None = None, target_hint: str = ""):
        self.port = int(port)
        self.target_id = target_id
        self.target_hint = str(target_hint or "")
        self._socket = None
        self._message_id = 0
        self._events: deque[dict] = deque()
        self._lock = threading.RLock()

    def _select_target(self, deadline: Deadline) -> dict:
        targets = page_targets(self.port, timeout=min(3, deadline.remaining()))
        if self.target_id:
            target = next(
                (value for value in targets if str(value.get("id") or "") == self.target_id),
                None,
            )
            if target is not None:
                return target
            raise RuntimeError(f"CDP target {self.target_id} disappeared; refusing to switch pages")
        if self.target_hint:
            preferred = [
                value for value in targets if self.target_hint in str(value.get("url") or "")
            ]
            if preferred:
                return preferred[0]
        if not targets:
            raise RuntimeError(f"CDP port {self.port} has no page target")
        return targets[0]

    def _ensure_connected(self, deadline: Deadline) -> None:
        if self._socket is not None:
            return
        target = self._select_target(deadline)
        self.target_id = str(target.get("id") or "") or self.target_id
        self._socket = connect(
            target["webSocketDebuggerUrl"],
            open_timeout=min(5, deadline.remaining()),
            close_timeout=2,
            max_size=None,
        )
        self._events.clear()

    def _recv(self, deadline: Deadline) -> dict:
        if self._socket is None:
            raise RuntimeError("CDP WebSocket is not connected")
        try:
            raw = self._socket.recv(timeout=deadline.remaining())
        except TimeoutError:
            raise deadline.exceeded() from None
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("CDP returned a non-object message")
        return value

    @contextmanager
    def _locked(self, deadline: Deadline):
        """Hold the session lock; waiting for it spends the call's deadline."""
        left = deadline.remaining()
        if not self._lock.acquire(timeout=-1 if left is None else left):
            raise deadline.exceeded()
        try:
            yield
        finally:
            self._lock.release()

    def call(
        self,
        method: str,
        params: dict | None = None,
        *,
        timeout: float = 120,
        session_id: str | None = None,
    ) -> dict:
        """Send one command and return its reply; ``timeout`` bounds the lock wait, the connect
        and the wait for the reply, not the send."""
        deadline = Deadline(timeout, f"CDP {method}")
        with self._locked(deadline):
            self._ensure_connected(deadline)
            self._message_id += 1
            message_id = self._message_id
            assert self._socket is not None
            try:
                self._socket.send(
                    json.dumps(
                        {
                            "id": message_id,
                            "method": method,
                            "params": params or {},
                            **({"sessionId": session_id} if session_id else {}),
                        }
                    )
                )
                while True:
                    message = self._recv(deadline)
                    if message.get("id") == message_id:
                        return message
                    self._events.append(message)
            except BaseException:
                self._close_locked()
                raise

    def wait_event(self, method: str, *, timeout: float = 120) -> dict:
        deadline = Deadline(timeout, f"CDP wait for {method}")
        with self._locked(deadline):
            for index, event in enumerate(self._events):
                if event.get("method") == method:
                    del self._events[index]
                    return event
            self._ensure_connected(deadline)
            try:
                while True:
                    event = self._recv(deadline)
                    if event.get("method") == method:
                        return event
                    self._events.append(event)
            except BaseException:
                self._close_locked()
                raise

    def evaluate(self, expression: str, *, await_promise: bool = True, timeout: float = 120) -> Any:
        response = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": bool(await_promise),
            },
            timeout=timeout,
        )
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        remote = response.get("result", {}).get("result", {})
        if remote.get("subtype") == "error":
            raise RuntimeError(remote.get("description") or "browser evaluation failed")
        return remote.get("value")

    @contextmanager
    def transaction(self):
        """Prevent another harness thread from interleaving a multi-command operation."""
        with self._locked(Deadline(None, "CDP transaction")):
            yield self

    def _close_locked(self) -> None:
        socket, self._socket = self._socket, None
        self._events.clear()
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            self._close_locked()


_REGISTRY_LOCK = threading.Lock()
_CLIENTS: dict[tuple[int, str], PersistentCDPClient] = {}
_ALIASES: dict[tuple[int, str], str] = {}


def client_for(
    port: int, *, target_id: str | None = None, target_hint: str = ""
) -> PersistentCDPClient:
    port = int(port)
    explicit_target = target_id is not None
    alias = (port, str(target_hint or ""))
    with _REGISTRY_LOCK:
        target_id = target_id or _ALIASES.get(alias)
    if not target_id:
        targets = page_targets(port)
        if target_hint:
            targets = sorted(
                targets,
                key=lambda value: target_hint not in str(value.get("url") or ""),
            )
        if not targets:
            raise RuntimeError(f"CDP port {port} has no page target")
        target_id = str(targets[0].get("id") or "")
    key = (port, target_id)
    with _REGISTRY_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = PersistentCDPClient(
                port,
                target_id=target_id,
                target_hint=target_hint,
            )
            _CLIENTS[key] = client
        # Inspecting an unrelated popup must not redirect the scorer/clock's default client.
        if not explicit_target:
            _ALIASES[alias] = target_id
        return client


def ensure_page_focus(port: int, *, target_hint: str = "", timeout: float = 1.0) -> bool:
    """Give the page Chrome's keyboard focus; return True when it had to be restored.

    Desktop key input goes wherever Chrome's focus is, including the omnibox, which keeps focus
    after a CDP navigation from ``about:blank``.  ``Page.bringToFront`` returns focus to the page
    without dispatching input to it.  Focus emulation is never used: it makes
    ``document.hasFocus()`` true while real key events still reach the omnibox.
    """
    client = client_for(port, target_hint=target_hint)
    if client.evaluate("document.hasFocus()") is True:
        return False
    response = client.call("Page.bringToFront")
    if response.get("error"):
        raise RuntimeError(f"Page.bringToFront failed: {response['error']}")
    deadline = time.monotonic() + timeout
    while client.evaluate("document.hasFocus()") is not True:
        if time.monotonic() >= deadline:
            raise RuntimeError("page did not regain keyboard focus from the browser toolbar")
        time.sleep(0.02)
    return True


def close_port(port: int) -> None:
    with _REGISTRY_LOCK:
        keys = [key for key in _CLIENTS if key[0] == int(port)]
        clients = [_CLIENTS.pop(key) for key in keys]
        aliases = [key for key in _ALIASES if key[0] == int(port)]
        for key in aliases:
            _ALIASES.pop(key, None)
    for client in clients:
        client.close()


def _probe_one(port: int, target: dict, expression: str, *, include_ax_tree=False) -> dict:
    target_id = str(target.get("id") or "")
    client = client_for(port, target_id=target_id)
    value = client.evaluate(expression, timeout=8)
    if not isinstance(value, dict):
        raise RuntimeError("browser extractor returned no object")
    scene = value.setdefault("scene", {})
    scene["target_id"] = target_id
    try:
        response = client.call("Browser.getWindowForTarget", {"targetId": target_id}, timeout=8)
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        window_id = str(response.get("result", {}).get("windowId") or target_id)
    except Exception:
        window_id = target_id
    value["window_id"] = window_id
    value["title"] = str(scene.get("title") or target.get("title") or "")
    value["active"] = bool(scene.get("active"))
    if include_ax_tree:
        from realtime_framework.sandbox.browser.ax import capture_ax_frames

        try:
            value.update(capture_ax_frames(client))
        except Exception as exc:
            value.setdefault("warnings", []).append(
                f"ax_tree_unavailable:{type(exc).__name__}:{str(exc)[:120]}"
            )
    payload = json.dumps(
        {
            "actionable_now": value.get("actionable_now", []),
            "visible_but_unavailable": value.get("visible_but_unavailable", []),
            "canvas_surfaces": value.get("canvas_surfaces", []),
            "readiness": value.get("readiness", {}),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    value.setdefault("binding", {})["fingerprint"] = hashlib.sha256(payload.encode()).hexdigest()
    return value


def probe_interactables(
    port: int,
    *,
    target_hint: str,
    spec: dict,
    desktop_size: tuple[int, int],
    include_ax_tree: bool = False,
) -> dict:
    """Evaluate the observation-bound DOM extractor over persistent page sessions."""
    from realtime_framework.sandbox.browser.cdp_interactables import build_expression as build_expression

    pages = page_targets(port)
    if target_hint:
        preferred = [value for value in pages if target_hint in str(value.get("url") or "")]
        preferred_ids = {value.get("id") for value in preferred}
        pages = preferred + [value for value in pages if value.get("id") not in preferred_ids]
    if not pages:
        return {
            "scene": {
                "status": "unsupported",
                "kind": "browser_without_page",
                "application": "google-chrome",
                "reason": "no_matching_page_target",
            },
            "binding": {"stable": True, "reason_codes": ["no_page_target"]},
            "actionable_now": [],
            "visible_but_unavailable": [],
            "canvas_surfaces": [],
        }

    expression = build_expression(spec, desktop_size)
    windows = []
    warnings = []
    for page in pages:
        try:
            windows.append(_probe_one(port, page, expression, include_ax_tree=include_ax_tree))
        except Exception as exc:
            warnings.append(f"window_probe_failed:{type(exc).__name__}:{str(exc)[:160]}")
    warnings.extend(warning for value in windows for warning in value.get("warnings", []))
    if not windows:
        raise RuntimeError("all browser window probes failed")

    visible_by_window = {}
    for value in windows:
        window_id = value["window_id"]
        current = visible_by_window.get(window_id)
        if current is None or (
            value["scene"].get("visible") and not current["scene"].get("visible")
        ):
            visible_by_window[window_id] = value
    windows = list(visible_by_window.values())
    active = next((value for value in windows if value.get("active")), windows[0])
    for value in windows:
        value["active"] = value is active

    aggregate = json.dumps(
        sorted(
            [
                {
                    "window_id": value["window_id"],
                    "active": value["active"],
                    "fingerprint": value.get("binding", {}).get("fingerprint", ""),
                }
                for value in windows
            ],
            key=lambda value: value["window_id"],
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    result = dict(active)
    result["windows"] = windows
    result["binding"] = dict(active.get("binding") or {})
    result["binding"]["fingerprint"] = hashlib.sha256(aggregate.encode()).hexdigest()
    result["limitations"] = sorted(
        {limitation for value in windows for limitation in value.get("limitations", [])}
    )
    if warnings:
        result["warnings"] = warnings
    return result
