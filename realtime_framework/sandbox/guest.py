from __future__ import annotations

# Guest
"""HTTP execution boundary for the guest agent inside the sandbox."""
import base64 as base64
import time as time

import requests as requests

from realtime_framework.entity.actions import Action
from realtime_framework.entity.actions import ActionType
from realtime_framework.entity.actions import validate_action
from realtime_framework.entity.actions import normalize_action
from realtime_framework.sandbox.budget import bounded


def action_timeout(action: Action, timeout: float) -> float:
    """How long the guest may take to execute ``action`` when a plain request has ``timeout``."""
    # Long text input and timed holds run synchronously on the guest; extend the HTTP timeout.
    if action.action_type == ActionType.TYPE:
        return max(timeout, 120.0)
    if action.action_type == ActionType.HOLD_KEY or "duration" in action.parameters:
        duration = action.parameters.get("duration", 1.0)
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            return max(timeout, float(duration) + 5.0)
    return timeout


class GuestClient:
    """HTTP client for communicating with the guest agent server inside the sandbox."""

    def __init__(self, host: str = "localhost", port: int = 5000, timeout: float = 10.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def screenshot(self) -> bytes:
        resp = requests.get(f"{self.base_url}/screenshot", timeout=bounded(self.timeout))
        resp.raise_for_status()
        data = resp.json()
        return base64.b64decode(data["image"])

    def execute_action(self, action: Action) -> dict:
        valid, reason = validate_action(action)
        if not valid:
            raise ValueError(f"invalid GUI action: {reason}")
        # Not in an episode: GuestEnvironment.execute (sandbox/actions.py) applies WAIT, DONE and
        # FAIL itself and never sends them here.
        if action.action_type == ActionType.WAIT:
            duration = float(action.parameters.get("duration", 0.0))
            if duration:
                time.sleep(duration)  # noqa: TID251
            return {
                "status": "ok",
                "timestamp": time.time(),  # noqa: TID251
                "duration_seconds": duration,
            }
        if action.action_type in (ActionType.DONE, ActionType.FAIL):
            return {"status": "ok", "timestamp": time.time()}  # noqa: TID251

        desktop_action, coordinate_trace = normalize_action(action)
        payload = {
            "action_type": desktop_action.action_type.value,
            "parameters": desktop_action.parameters,
        }
        resp = requests.post(
            f"{self.base_url}/action",
            json=payload,
            timeout=bounded(action_timeout(desktop_action, self.timeout)),
        )
        if 400 <= resp.status_code < 500:
            # The guest validated the body and refused it (unsupported key name, unexpected
            # parameter, ...). Keep its message: the runtime reports it back to the model.
            try:
                message = (resp.json() or {}).get("message")
            except ValueError:
                message = None
            raise requests.HTTPError(
                f"{resp.status_code} guest rejected the action"
                + (f": {message}" if message else ""),
                response=resp,
            )
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, dict):
            result.setdefault("coordinate_trace", coordinate_trace)
        return result

    def execute_command(self, command: str, timeout: int = 30) -> dict:
        payload = {"command": command, "timeout": timeout}
        resp = requests.post(
            f"{self.base_url}/execute",
            json=payload,
            timeout=bounded(max(self.timeout, timeout + 5)),
        )
        resp.raise_for_status()
        return resp.json()

    def wait_until_ready(self, max_wait: float = 60.0, poll_interval: float = 2.0) -> bool:
        # Before the episode: the runner waits for the guest server while it sets up.
        start = time.time()  # noqa: TID251
        while time.time() - start < max_wait:  # noqa: TID251
            if self.health_check():
                return True
            time.sleep(poll_interval)  # noqa: TID251
        return False
