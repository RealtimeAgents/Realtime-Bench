"""What of GuestEnvironment only the legacy loop (legacy_loop.py) reads, moved here from
sandbox/actions.py unchanged (decision 11).

LegacyWallClock is the wall-clock half: the deadline the environment was built with and compared
itself against, in its waits, its WAIT, done() and the optional deadline hooks. The kernel judges
the deadline itself (kernel/turns.py), so production's GuestEnvironment takes none.

LegacyStateReads is the rest: done(), the legacy loop's end check, and the end check after each
token step through it, which invalidates the run at the first failed read, while the kernel's
shell, which counts the reads itself, has not set kernel_counts_reads. Production's
GuestEnvironment answers check_end() only.
"""

from __future__ import annotations

import traceback

from realtime_framework.entity.actions import ActionType
from realtime_framework.sandbox import actions
from realtime_framework.sandbox.actions import GuestEnvironment
from realtime_framework.sandbox.budget import operation_budget


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


class LegacyStateReads:
    """Mixed in before GuestEnvironment, after LegacyWallClock if both are."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set by the kernel's shell, which counts failed state reads: the end checks a key tap
        # makes go to last_end_checks, each (ended, reason, error) as check_end() answered or
        # failed, (False, None, None) also when the task has no end check, instead of done(),
        # which invalidates the run at the first failure.
        self.kernel_counts_reads = False

    def _ended_after_step(self):
        if not self.kernel_counts_reads:
            return self.done()
        return super()._ended_after_step()

    def done(self):
        if getattr(self, "_invalid_run", None):
            return True
        if self._done:
            return True
        if self.stop_check is not None:
            try:
                with operation_budget(
                    "state read",
                    actions.ENVIRONMENT_BUDGET.state_read_s,
                    clock=self.clocks.operational,
                ):
                    terminal = self.stop_check()
                if terminal:
                    self._episode_over = True
                    if isinstance(terminal, str):
                        self._terminal_reason = terminal
                    return True
            except Exception:  # noqa: BLE001 - a failed read invalidates the run
                self._invalid_run = {
                    "category": "environment",
                    "reason": "terminal_state_check_failed",
                    "detail": traceback.format_exc(),
                }
                return True
        return False


class LegacyGuestEnvironment(LegacyWallClock, LegacyStateReads, GuestEnvironment):
    """GuestEnvironment as the legacy loop had it."""
