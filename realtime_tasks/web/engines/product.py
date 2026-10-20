"""Neutral episode transport only; products own all state, decisions, UI, and scoring."""

from copy import deepcopy
import hashlib
import json
import secrets


class ProductEpisode:
    def __init__(self, provider, clock, wall_clock):
        self.provider = provider
        self.clock = clock
        self.wall_clock = wall_clock
        self.template = f"scenarios/{provider.TASK_ID}.html"

    def ui_context(self):
        return {}

    def reset(self, generation=1, seed=42):
        episode_id = secrets.token_hex(16)
        return {
            "episode": episode_id,
            "generation": generation,
            "seed": seed,
            "started": self.clock(),
            "wall_started": self.wall_clock(),
            "opened": False,
            "world": self.provider.create_world(str(seed)),
            "version": 0,
            "view_secret": secrets.token_hex(16),
            "projection_hash": "",
            "receipts": {},
        }

    def _advance(self, episode, now=None):
        elapsed = max(0.0, (self.clock() if now is None else now) - episode["started"])
        self.provider.advance(episode["world"], elapsed)
        return elapsed

    def _projection(self, episode):
        public = self.provider.public_world(episode["world"])
        # Providers declare top-level display clocks separately from business data.
        # Keep those clocks in observations without invalidating actions as time passes.
        # Timed business changes and timestamps inside records remain versioned.
        clock_fields = getattr(self.provider, "CLOCK_FIELDS", ())
        versioned = {key: value for key, value in public.items() if key not in clock_fields}
        digest = hashlib.sha256(json.dumps(versioned, sort_keys=True).encode()).hexdigest()
        if digest != episode["projection_hash"]:
            episode["version"] += 1
            episode["projection_hash"] = digest
        return public

    def _token(self, episode):
        return hashlib.sha256(
            f"{episode['view_secret']}:{episode['generation']}:{episode['version']}".encode()
        ).hexdigest()[:32]

    def state(self, episode):
        elapsed = self._advance(episode)
        public = self._projection(episode)
        evaluation = self.provider.evaluate(episode["world"])
        terminal = self.provider.terminal_reason(episode["world"])
        resolved = evaluation["resolved"]
        return {
            "ok": True,
            "episode": episode["episode"],
            "generation": episode["generation"],
            "version": episode["version"],
            "view_token": self._token(episode),
            "server_now_ms": self.wall_clock() * 1000,
            "elapsed_seconds": elapsed,
            "resolved": resolved,
            "terminal_reason": terminal,
            "phase": "resolved" if resolved else "expired" if terminal else "active",
            "can_act": not (resolved or terminal),
            "world": public,
        }

    def action(self, episode, body, now=None):
        elapsed = self._advance(episode, now)
        self._projection(episode)
        if type(body.get("generation")) is not int or body["generation"] != episode["generation"]:
            return {"ok": False, "error": "This session has changed. Reload the page."}, 409
        command = body.get("command")
        if not isinstance(command, dict):
            return {"ok": False, "error": "Invalid request."}, 400
        receipt_id = body.get("command_id")
        if receipt_id is not None and (not isinstance(receipt_id, str) or len(receipt_id) > 128):
            return {"ok": False, "error": "Invalid request identifier."}, 400
        if receipt_id and receipt_id in episode["receipts"]:
            saved = episode["receipts"][receipt_id]
            if saved["command"] != command:
                return {"ok": False, "error": "Request identifier already used."}, 409
            return deepcopy(saved["response"]), saved["status"]
        if (
            type(body.get("version")) is not int
            or body["version"] != episode["version"]
            or body.get("view_token") != self._token(episode)
        ):
            return {
                "ok": False,
                "error": getattr(
                    self.provider, "STALE_MESSAGE", "The order changed. Refresh and try again."
                ),
            }, 409
        if (
            self.provider.terminal_reason(episode["world"])
            or self.provider.evaluate(episode["world"])["resolved"]
        ):
            return {"ok": False, "error": "This operation is no longer available."}, 409
        response = self.provider.apply_command(episode["world"], command, elapsed)
        self._projection(episode)
        response["resolved"] = self.provider.evaluate(episode["world"])["resolved"]
        status = 200 if response.get("ok") else 409
        if receipt_id:
            episode["receipts"][receipt_id] = {
                "command": deepcopy(command),
                "response": deepcopy(response),
                "status": status,
            }
        return response, status

    def evaluation(self, episode):
        # Scoring is read-only. Clock advancement belongs to state/action boundaries.
        return {**self.provider.evaluate(episode["world"]), "generation": episode["generation"]}
