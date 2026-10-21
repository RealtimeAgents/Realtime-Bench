"""GuestEnvironment's wall-clock half that only the legacy loop (legacy_loop.py) reads, moved here
from sandbox/actions.py unchanged (decision 11): the deadline the environment was built with and
compared itself against, in its waits, its WAIT, done() and the optional deadline hooks. The
kernel judges the deadline itself (kernel/turns.py), so production's GuestEnvironment takes none.
The token half the legacy loop reads, done() and the advance's end check without the kernel
counting reads, is still in sandbox/actions.py."""

from __future__ import annotations

from realtime_framework.entity.actions import ActionType
from realtime_framework.sandbox.actions import GuestEnvironment


class LegacyWallClock:
    """Mixed in before GuestEnvironment; takes the deadline back as the second argument, as main()
    passed it."""

    def __init__(self, guest, deadline, task, *args, **kwargs):
        super().__init__(guest, task, *args, **kwargs)
        self.deadline = deadline

    def wait_until_tick(self, target):
        if self.clock_mode == "token":
            return super().wait_until_tick(target)
        clock = self.clocks.semantic
        while self.tick_index() < target and clock.time() < self.deadline:
            clock.sleep(0.03)
        return None

    def time_remaining(self):
        return None if self.clock_mode == "token" else self.deadline - self.clocks.semantic.time()

    def past_deadline(self):
        return False if self.clock_mode == "token" else self.clocks.semantic.time() >= self.deadline

    def execute(self, action, *, release_at=None):
        if self.clock_mode == "token" or action.action_type != ActionType.WAIT:
            return super().execute(action, release_at=release_at)
        self.last_held_keys = ()
        self.last_end_checks = ()
        try:
            duration = float(action.parameters.get("duration", 0.0))
            started_at = self.clock_position()
            if duration > 0:
                remaining = max(0.0, self.deadline - self.clocks.semantic.time())
                self.clocks.semantic.sleep(min(duration, remaining))
            elapsed = round(
                max(0.0, (self.clock_position() - started_at) * self.d),
                9,
            )
            self._last_execution_result = {
                "duration_simulated_seconds": elapsed,
                "requested_duration_simulated_seconds": duration,
                "truncated_by_episode_horizon": False,
                "truncated_by_terminal_condition": False,
                "active_holds": self._active_hold_snapshot(),
                "active_mouse_buttons": sorted(self._held_mouse_buttons),
            }
            return "wait"
        except Exception as exc:  # noqa: BLE001 - a failed wait fails the action
            return self._execution_failed(exc)

    def done(self):
        if self.clock_mode != "token" and self.clocks.semantic.time() >= self.deadline:
            return True
        return super().done()


class LegacyGuestEnvironment(LegacyWallClock, GuestEnvironment):
    """GuestEnvironment as the legacy loop had it."""
