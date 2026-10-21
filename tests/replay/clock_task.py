"""A game task's token clock without the browser, for GuestEnvironment on the token clock.

The runner configures the task clock with the environment fps before anything else
(evaluation/runner.py main), and GuestEnvironment takes its tick from it (sandbox/actions.py:130).
On the token clock GuestEnvironment pushes time into the task with advance_clock and sends key
input with clock_key_event (sandbox/actions.py:214-249, 495-605), as it does for a game
(realtime_tasks/game/runtime.py:177-214). Here the frame counter of realtime_tasks/types.py
advances unchanged, and every advance, key event and evaluation snapshot is appended to
``events`` with the task's world time, and passed to ``on_event`` when one is set.
"""

from __future__ import annotations

from realtime_tasks.types import Task


class ClockTask(Task):
    """``advance_fails`` is ``(world seconds, message)``: from that time on the clock fails to
    advance, as a game page's can."""

    task_id = "replay-clock"
    category = "browser_game"

    def __init__(
        self,
        fps: float,
        tokens_per_frame: int | None = None,
        *,
        advance_fails: tuple[float, str] | None = None,
    ):
        self.events: list[dict] = []
        self.on_event = None
        self.advance_fails = advance_fails
        self.configure_clock("token", fps, tokens_per_frame)
        self.activate_clock()

    def world_time(self) -> float:
        return self.clock_snapshot()["world_time_s"]

    def _event(self, event: dict) -> None:
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def advance_clock(self, frames: float = 1, key_events=()):
        before = self.world_time()
        if self.advance_fails is not None and before >= self.advance_fails[0] - 1e-9:
            self._event({"event": "advance_failed", "at_s": before})
            raise RuntimeError(self.advance_fails[1])
        snapshot = super().advance_clock(frames)
        self._event(
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
        self._event(
            {
                "event": event_type,
                "key": key,
                "auto_repeat": auto_repeat,
                "held_modifiers": list(held_modifiers),
                "at_s": self.world_time(),
            }
        )

    def evaluation_snapshot(self) -> dict:
        """run_stream freezes this when the loop ends (shell.py _freeze)."""
        snapshot = {"kind": "replay_clock", **self.clock_snapshot()}
        self._event({"event": "evaluation_snapshot", "snapshot": snapshot})
        return snapshot
