"""A game task's token clock without the browser, for GuestEnvironment on the token clock.

The runner configures the task clock with the environment fps before anything else
(evaluation/runner.py:1536), and GuestEnvironment takes its tick from it (sandbox/actions.py:130).
On the token clock GuestEnvironment pushes time into the task with advance_clock and sends key
input with clock_key_event (sandbox/actions.py:214-249, 495-605), as it does for a game
(realtime_tasks/game/runtime.py:177-214). Here the frame counter of realtime_tasks/types.py
advances unchanged, and every advance and key event is logged with the task's world time.
"""

from __future__ import annotations

from realtime_tasks.types import Task


class ClockTask(Task):
    task_id = "replay-clock"
    category = "browser_game"

    def __init__(self, fps: float, tokens_per_frame: int | None = None, *, snapshot=None):
        self.events: list[dict] = []
        self._snapshot = snapshot
        self.configure_clock("token", fps, tokens_per_frame)
        self.activate_clock()

    def world_time(self) -> float:
        return self.clock_snapshot()["world_time_s"]

    def advance_clock(self, frames: float = 1, key_events=()):
        before = self.world_time()
        snapshot = super().advance_clock(frames)
        self.events.append(
            {
                "event": "advance",
                "frames": frames,
                "from_s": before,
                "to_s": snapshot["world_time_s"],
                "key_events": [dict(event) for event in key_events],
            }
        )
        return snapshot

    def clock_key_event(self, event_type, key, auto_repeat=False, held_modifiers=()):
        self.events.append(
            {
                "event": event_type,
                "key": key,
                "auto_repeat": auto_repeat,
                "held_modifiers": list(held_modifiers),
                "at_s": self.world_time(),
            }
        )

    def evaluation_snapshot(self) -> dict:
        """run_stream freezes this when the loop ends (runner.py:778-788)."""
        return self._snapshot(self) if self._snapshot is not None else {}
