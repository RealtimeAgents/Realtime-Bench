"""GuestEnvironment itself on the token clock, logging the runner's calls into it.

On the token clock main() hands run_stream a GuestEnvironment (evaluation/runner.py:1678-1690).
Here that class runs unchanged: its tick, horizon, key leases and waits (sandbox/actions.py) move a
ClockTask (clock_task.py) in place of a game's task clock, captures come from the synthetic frames
of frames.py, and pointer and text actions reach a guest that only records them. Each call
run_stream makes, and each event of the task clock, is appended to ``calls`` tagged with the
decision it belongs to, as FakeEnvironment does, so loop.records reads both.
"""

from __future__ import annotations

from realtime_framework.sandbox.actions import GuestEnvironment

from .baseline import action_record
from .clock_task import ClockTask
from .fake_env import FakeClock
from .frames import FrameSource


class _Guest:
    """The guest agent's action endpoint (sandbox/guest.py:50-81), which answers every action."""

    def __init__(self, env: TokenEnvironment):
        self._env = env

    def execute_action(self, action):
        self._env.log("guest.execute_action", action=action_record(action))
        return {"status": "ok"}


class TokenEnvironment(GuestEnvironment):
    """``task`` is the task the agent was set up with; ``max_minutes`` sets the horizon.
    ``stop_check()`` stands for the task's episode_over, which main() passes (runner.py:1767)."""

    def __init__(
        self, frames: FrameSource, task: ClockTask, *, max_minutes: float, stop_check=None
    ):
        self.calls: list[dict] = []
        self.decision = -1  # run_stream captures once per decision (runner.py:846)
        self.finalizing = False
        super().__init__(
            _Guest(self),
            None,
            task,
            stop_check=stop_check,
            scene_capture=frames,
            clock_mode="token",
        )
        self.fps = task.clock_snapshot()["fps"]
        self.max_minutes = max_minutes
        # run_stream's own time.time() reads (loop.py); token time never moves it.
        self.clock = FakeClock()
        task.on_event = self._task_event

    def log(self, call: str, **fields) -> None:
        self.calls.append(
            {"call": call, "decision": self.decision, "finalizing": self.finalizing, **fields}
        )

    def _task_event(self, event: dict) -> None:
        if event["event"] == "evaluation_snapshot":
            self.finalizing = True
        self.log("task." + event["event"], **{k: v for k, v in event.items() if k != "event"})

    def capture(self):
        self.decision += 1
        observation = super().capture()
        self.log(
            "env.capture",
            tick=observation.tick_number,
            timestamp=observation.timestamp,
            observation_id=observation.metadata["observation_id"],
        )
        return observation

    def wait_until_tick(self, target) -> None:
        before = self._logical_tick
        super().wait_until_tick(target)
        self.log("env.wait_until_tick", target=target, from_tick=before, to_tick=self._logical_tick)

    def execute(self, action):
        before = self._logical_tick
        status = super().execute(action)
        self.log(
            "env.execute",
            action=action_record(action),
            from_tick=before,
            to_tick=self._logical_tick,
            status=status,
            execution_result=self._last_execution_result,
        )
        return status

    def done(self) -> bool:
        """Only a true answer is logged."""
        over = super().done()
        if over:
            self.log("env.done")
        return over

    def record_lifecycle(self, record: dict) -> None:
        self.log("env.record_lifecycle", record=record)

    def release_inputs(self) -> None:
        self.log("env.release_inputs")
        super().release_inputs()
