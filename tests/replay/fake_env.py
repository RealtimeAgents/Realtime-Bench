"""A fake wall-clock environment that takes the place of GuestEnvironment under run_stream.

run_stream (evaluation/runner.py:102-1180) needs capture, execute, tick_index, done and
wait_until_tick, and reads the optional time_remaining and past_deadline hooks
(runner.py:15-18). On the wall clock main() hands it a GuestEnvironment (runner.py:1678-1690,
sandbox/actions.py:112-967). This class follows GuestEnvironment's wall-clock branch call for
call on the episode's fake clock (tests/fake_clock.py), serves the frames of frames.py, and
executes nothing: each call is appended to ``calls`` in order, tagged with the decision it belongs
to. The kernel's shell asks check_end instead of done (evaluation/shell.py); what it calls only on
the token clock (set_episode_horizon, step_clock, release_keys, render_current_frame) has no
wall-clock part.

Model latency is scripted here too, since it is wall time: a stub calls ``model_request(index)``
as its ``on_request`` hook before it answers request ``index`` and sleeps that request's scripted
seconds on the clock, and a Hang tells the stub to hold the request.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass

from realtime_framework.entity.actions import ActionType

from .baseline import action_record
from .frames import FrameSource


@dataclass(frozen=True)
class Hang:
    """A request the model never answers. The stub holds it until the client closes the
    connection, which the agent's abort does: runner.py:339-344 calls
    harnesses/default/harness.py:1683-1686, whose runtime closes the driver
    (harnesses/default/context.py:662-687), and the driver closes its HTTP client
    (providers/litellm.py:1033-1044). ``seconds`` is the fake time the runner's bounded wait
    (runner.py:337) lets pass before it gives up; the clock gets there by itself."""

    seconds: float


class FakeEnvironment:
    """GuestEnvironment's wall-clock behavior on a fake clock. ``fps`` is the environment's rate.

    ``rejections`` maps a decision to the guest's refusal of its action, which run_stream turns
    into "rejected" (runner.py:571-575). ``failures`` maps a decision to an environment fault,
    which invalidates the run (runner.py:576-579). ``advance_failures`` maps a decision to a
    failure of its post-action wait, which the environment latches and raises again. It stands in
    for GuestEnvironment's token-clock advance (actions.py:235-241); the wall-clock wait only
    polls (actions.py:251-252). run_stream catches the exception and still finalizes
    (runner.py:1246-1258). ``stop_check(env)`` stands for the task's episode_over
    (runner.py:1773): a true value ends the episode, a string names the reason
    (actions.py:952-958). ``reply_latencies[i]`` is the fake time request i takes, and
    ``snapshot_seconds`` the time the evaluation snapshot's read of the task state takes.
    """

    clock_mode = "wall"

    def __init__(
        self,
        frames: FrameSource,
        clock,
        *,
        fps: float,
        max_minutes: float,
        reply_latencies: list,
        rejections: dict[int, str] | None = None,
        failures: dict[int, str] | None = None,
        advance_failures: dict[int, str] | None = None,
        stop_check=None,
        snapshot_seconds: float = 0.0,
    ):
        self.frames = frames
        self.clock = clock
        self.fps = fps
        self.max_minutes = max_minutes
        self.d = 1.0 / fps  # actions.py:130
        # main() starts the wall clock and sets the deadline from it (runner.py:1670-1674);
        # GuestEnvironment measures ticks from that start (actions.py:131).
        self.t0 = clock.time()
        self.deadline = self.t0 + max_minutes * 60
        self.reply_latencies = list(reply_latencies)
        self.rejections = dict(rejections or {})
        self.failures = dict(failures or {})
        self.advance_failures = dict(advance_failures or {})
        self.stop_check = stop_check
        self.snapshot_seconds = snapshot_seconds
        # The task hook run_stream reads through env.task (runner.py:778-788).
        self.task = types.SimpleNamespace(evaluation_snapshot=self._evaluation_snapshot)
        self.frame_ages = []  # actions.py:166; run_stream appends to it (runner.py:656-658)
        self.calls: list[dict] = []
        self.decision = -1  # run_stream captures once per decision (runner.py:846)
        self.finalizing = False
        self.executions: list[dict] = []
        # What the kernel's shell reads and sets (evaluation/shell.py). On the wall clock an
        # execution leases no token-clock key and makes no end check, and here no key-release
        # timer runs that could invalidate the run.
        self.last_held_keys: tuple[str, ...] = ()
        self.last_end_checks: tuple[str | None, ...] = ()
        self.kernel_counts_reads = False
        self.on_invalid = None

    def log(self, call: str, **fields) -> None:
        self.calls.append(
            {"call": call, "decision": self.decision, "finalizing": self.finalizing, **fields}
        )

    # ---- clock (actions.py:179-188, 214-252, 271-275) ----

    def clock_position(self) -> float:
        return max(0.0, (self.clock.time() - self.t0) / self.d)

    def tick_index(self) -> int:
        return int(self.clock_position())

    def time_remaining(self) -> float:
        return self.deadline - self.clock.time()

    def past_deadline(self) -> bool:
        return self.clock.time() >= self.deadline

    def wait_until_tick(self, target) -> None:
        """GuestEnvironment polls until tick_index() reaches ``target`` or the deadline passes
        (actions.py:251-252); tick_index() is an integer, so a fractional target waits for the
        next whole tick. The fake lands on the tick exactly, without the 30 ms polling step: on
        the first clock reading whose tick_index() reaches it, which for a tick that is not a
        whole number of seconds can lie a few ulps past t0 + tick * d."""
        before = self.clock.time()
        if self.decision in self.advance_failures:
            message = self.advance_failures[self.decision]
            self._invalid_run = {
                "category": "environment",
                "reason": "clock_advance_failed",
                "detail": message,
            }
            self.log("env.wait_until_tick", target=target, failed=message)
            raise RuntimeError(message)
        if self.tick_index() < target and before < self.deadline:
            self.clock.sleep(min(self.t0 + math.ceil(target) * self.d, self.deadline) - before)
            while self.tick_index() < target and self.clock.time() < self.deadline:
                now = self.clock.time()
                self.clock.sleep(math.nextafter(now, math.inf) - now)
        self.log("env.wait_until_tick", target=target, from_time=before, to_time=self.clock.time())

    # ---- observation ----

    def capture(self):
        """actions.py:190-212. BrowserSceneCapture already sets the frame geometry and frames.py
        a zero capture latency, which GuestEnvironment would measure."""
        self.decision += 1
        observation = self.frames.capture(self.tick_index(), at=self.clock.time())
        self.log(
            "env.capture",
            tick=observation.tick_number,
            time=self.clock.time(),
            observation_id=observation.metadata["observation_id"],
        )
        return observation

    # ---- terminal state (actions.py:947-967) ----

    def _over(self):
        """Why the episode is over now, or None; with the task's answer."""
        if getattr(self, "_invalid_run", None):
            return "invalid_run", None
        if self.clock.time() >= self.deadline:
            return "deadline", None
        if self.stop_check is not None:
            terminal = self.stop_check(self)
            if terminal:
                return "task", terminal
        return None, None

    def done(self) -> bool:
        """Only a true answer is logged, with its cause."""
        cause, terminal = self._over()
        if cause is None:
            return False
        if cause == "task":
            self._episode_over = True
            if isinstance(terminal, str):
                self._terminal_reason = terminal
        self.log("env.done", cause=cause)
        return True

    def check_end(self) -> tuple[bool, str | None]:
        """GuestEnvironment.check_end, for the kernel's shell: only the task is asked. Only an end
        is logged."""
        terminal = self.stop_check(self) if self.stop_check is not None else None
        if not terminal:
            return False, None
        self._episode_over = True
        reason = terminal if isinstance(terminal, str) else None
        if reason is not None:
            self._terminal_reason = reason
        self.log("env.check_end", reason=reason)
        return True, reason

    # ---- actions (actions.py:805-938) ----

    def execute(self, action, *, release_at=None):
        """``release_at`` is the release time the kernel's shell gives a hold_key; it is recorded
        when given, since no key is held here."""
        execution = {
            "action": action_record(action),
            "time": self.clock.time(),
            "position": self.clock_position(),
            # Set when the action arrives after the episode is over, whether or not the runner
            # has asked done() since.
            "episode_over": self._over()[0],
        }
        if release_at is not None:
            execution["release_at"] = release_at
        self.executions.append({"decision": self.decision, **execution})
        if action.action_type == ActionType.WAIT:
            # The wall clock sleeps out the wait, cut short at the deadline (actions.py:830-870);
            # of its execution result, the fields that carry the duration.
            duration = float(action.parameters.get("duration", 0.0))
            started_at = self.clock_position()
            if duration > 0:
                self.clock.sleep(min(duration, max(0.0, self.deadline - self.clock.time())))
            elapsed = round(max(0.0, (self.clock_position() - started_at) * self.d), 9)
            self._last_execution_result = {
                "duration_simulated_seconds": elapsed,
                "requested_duration_simulated_seconds": duration,
            }
            status = "wait"
        elif self.decision in self.rejections:
            status = self._execution_failed(self.rejections[self.decision], by_guest=True)
        elif self.decision in self.failures:
            status = self._execution_failed(self.failures[self.decision], by_guest=False)
        else:
            self._last_execution_result = {"execution": len(self.executions) - 1}
            status = "applied"
        execution["status"] = status
        self.executions[-1]["status"] = status
        self.log("env.execute", **execution)
        return status

    def _execution_failed(self, message: str, *, by_guest: bool) -> str:
        """actions.py:805-819: a guest refusal is the model's action failing; any other fault
        latches the run invalid in the environment itself."""
        self._last_exec_error = message
        self._last_rejected_by_guest = by_guest
        if not by_guest and not getattr(self, "_invalid_run", None):
            self._invalid_run = {
                "category": "environment",
                "reason": "action_execution_failed",
                "detail": message,
            }
        self._last_execution_result = {"error": message}
        self._last_coordinate_trace = None
        return "failed"

    def record_lifecycle(self, record: dict) -> None:
        self.log("env.record_lifecycle", record=record)

    def release_inputs(self) -> None:
        self.log("env.release_inputs")

    def _evaluation_snapshot(self) -> dict:
        self.finalizing = True
        self.log("task.evaluation_snapshot")
        self.clock.sleep(self.snapshot_seconds)
        return {
            "kind": "replay_form",
            "applied_actions": sum(e["status"] == "applied" for e in self.executions),
            "environment_tick": self.tick_index(),
        }

    # ---- model latency ----

    def model_request(self, index: int) -> bool:
        """Stub hook, on the stub's thread: request ``index`` takes its scripted fake time before
        it is answered. True asks the stub to hold the request (Hang). A request beyond the
        script takes no time, so the stub's own refusal of it (stub.py) fails the decision at
        once."""
        if index >= len(self.reply_latencies):
            self.log("model.request", request=index, seconds=None, hang=False)
            return False
        latency = self.reply_latencies[index]
        hang = isinstance(latency, Hang)
        seconds = latency.seconds if hang else latency
        self.log("model.request", request=index, seconds=seconds, hang=hang)
        if not hang:
            self.clock.sleep(seconds)
        return hang
