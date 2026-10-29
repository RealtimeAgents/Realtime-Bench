"""A fake wall-clock environment that takes the place of GuestEnvironment under run_stream.

run_stream, the kernel's shell (evaluation/shell.py), needs capture, execute, check_end and
tick_index, and the parts of the environment its ShellEnvironment protocol names; what it calls only
on the token clock (set_episode_horizon, step_clock, release_keys, render_current_frame) has no
wall-clock part. The legacy loop (tests/differential/legacy_loop.py _run_stream_legacy) asks done
instead of check_end, paces with wait_until_tick and reads the optional time_remaining and
past_deadline hooks. On the wall clock main() hands it a GuestEnvironment (sandbox/actions.py), the
legacy loop one with its deadline (tests/differential/legacy_environment.py). This class follows
their wall-clock branch call for call on the episode's fake clock (tests/fake_clock.py), serves the
frames of frames.py, and executes nothing: each call is appended to ``calls`` in order, tagged with
the decision it belongs to.

Model latency is scripted here too, since it is wall time: a stub calls ``model_request(index)``
as its ``on_request`` hook before it answers request ``index`` and sleeps that request's scripted
seconds on the clock, and a Hang tells the stub to hold the request.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass

from realtime_framework.entity.actions import ActionType
from realtime_framework.sandbox.actions import Execution

from .baseline import action_record
from .frames import FrameSource


@dataclass(frozen=True)
class Hang:
    """A request the model never answers. The stub holds it until the client closes the
    connection, which the agent's abort does: the shell calls it when the kernel gives the
    generation up (shell.py _abort), DefaultHarness.abort cancels the generation and aborts the
    runtime, which closes the driver (harnesses/default/context.py), and the driver closes its
    HTTP client (providers/litellm.py). ``seconds`` is the fake time until the kernel gives up, at
    the deadline; the clock gets there by itself."""

    seconds: float


class FakeEnvironment:
    """GuestEnvironment's wall-clock behavior on a fake clock. ``fps`` is the environment's rate.

    ``rejections`` maps a decision to the guest's refusal of its action, which run_stream turns
    into "rejected" (shell.py _execute). ``failures`` maps a decision to an environment fault,
    which invalidates the run. ``stop_check(env)`` stands for the task's episode_over, which
    main() passes: a true value ends the episode, a string names the reason
    (GuestEnvironment.check_end). ``read_failures`` are the end checks, counted from 1, whose
    read of the task state fails. ``reply_latencies[i]`` is the fake time request i takes, and
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
        stop_check=None,
        read_failures: frozenset[int] = frozenset(),
        snapshot_seconds: float = 0.0,
    ):
        self.frames = frames
        self.clock = clock
        self.fps = fps
        self.max_minutes = max_minutes
        self.d = 1.0 / fps  # as GuestEnvironment's
        # main() starts the wall clock and sets the deadline from it (runner.py main);
        # GuestEnvironment measures ticks from that start (t0).
        self.t0 = clock.time()
        self.deadline = self.t0 + max_minutes * 60
        self.reply_latencies = list(reply_latencies)
        self.rejections = dict(rejections or {})
        self.failures = dict(failures or {})
        self.stop_check = stop_check
        self.read_failures = read_failures
        self.end_checks = 0
        self.snapshot_seconds = snapshot_seconds
        # The task hook run_stream reads through env.task (shell.py _read_snapshot).
        self.task = types.SimpleNamespace(evaluation_snapshot=self._evaluation_snapshot)
        self.frame_ages = []  # run_stream appends to it (shell.py _commit)
        self.calls: list[dict] = []
        self.decision = -1  # run_stream captures once per decision
        self.finalizing = False
        self.executions: list[dict] = []
        # What the kernel's shell sets (evaluation/shell.py). Here no key-release timer runs that
        # could invalidate the run.
        self.on_invalid = None
        # The streaming lanes point a browser task's page capture at a CDP session of its own
        # (evaluation/env_lanes.py). Here the frames come from frames.py, with no page capture, as
        # in a desktop task.
        self.scene_capture = None

    def log(self, call: str, **fields) -> None:
        self.calls.append(
            {"call": call, "decision": self.decision, "finalizing": self.finalizing, **fields}
        )

    # ---- clock (actions.py tick_index; the deadline's, legacy_environment.py) ----

    def clock_position(self) -> float:
        return max(0.0, (self.clock.time() - self.t0) / self.d)

    def tick_index(self) -> int:
        return int(self.clock_position())

    def time_remaining(self) -> float:
        return self.deadline - self.clock.time()

    def past_deadline(self) -> bool:
        return self.clock.time() >= self.deadline

    def wait_until_tick(self, target) -> None:
        """The legacy loop's pace. GuestEnvironment polls until tick_index() reaches ``target`` or
        the deadline passes; tick_index() is an integer, so a fractional target waits for the next
        whole tick. The fake lands on the tick exactly, without the 30 ms polling step: on the
        first clock reading whose tick_index() reaches it, which for a tick that is not a whole
        number of seconds can lie a few ulps past t0 + tick * d."""
        before = self.clock.time()
        if self.tick_index() < target and before < self.deadline:
            self.clock.sleep(min(self.t0 + math.ceil(target) * self.d, self.deadline) - before)
            while self.tick_index() < target and self.clock.time() < self.deadline:
                now = self.clock.time()
                self.clock.sleep(math.nextafter(now, math.inf) - now)
        self.log("env.wait_until_tick", target=target, from_time=before, to_time=self.clock.time())

    # ---- observation ----

    def capture(self):
        """As GuestEnvironment's. BrowserSceneCapture already sets the frame geometry and
        frames.py a zero capture latency, which GuestEnvironment would measure."""
        self.decision += 1
        observation = self.frames.capture(self.tick_index(), at=self.clock.time())
        self.log(
            "env.capture",
            tick=observation.tick_number,
            time=self.clock.time(),
            observation_id=observation.metadata["observation_id"],
        )
        return observation

    # ---- terminal state (actions.py done, check_end) ----

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
        and a failed read are logged."""
        self.end_checks += 1
        if self.end_checks in self.read_failures:
            self.log("env.check_end", failed=self.end_checks)
            raise RuntimeError(f"state read {self.end_checks} failed")
        terminal = self.stop_check(self) if self.stop_check is not None else None
        if not terminal:
            return False, None
        self._episode_over = True
        reason = terminal if isinstance(terminal, str) else None
        if reason is not None:
            self._terminal_reason = reason
        self.log("env.check_end", reason=reason)
        return True, reason

    # ---- actions (actions.py execute) ----

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
            # The wall clock sleeps out the wait, cut short at the deadline
            # (legacy_environment.py LegacyWallClock.execute); of its execution result, the fields
            # that carry the duration.
            duration = float(action.parameters.get("duration", 0.0))
            started_at = self.clock_position()
            if duration > 0:
                self.clock.sleep(min(duration, max(0.0, self.deadline - self.clock.time())))
            elapsed = round(max(0.0, (self.clock_position() - started_at) * self.d), 9)
            executed = Execution(
                "wait",
                {
                    "duration_simulated_seconds": elapsed,
                    "requested_duration_simulated_seconds": duration,
                },
            )
        elif self.decision in self.rejections:
            executed = self._execution_failed(self.rejections[self.decision], by_guest=True)
        elif self.decision in self.failures:
            executed = self._execution_failed(self.failures[self.decision], by_guest=False)
        else:
            executed = Execution("applied", {"execution": len(self.executions) - 1})
        execution["status"] = executed.status
        self.executions[-1]["status"] = executed.status
        self.log("env.execute", **execution)
        return executed

    def _execution_failed(self, message: str, *, by_guest: bool) -> Execution:
        """As GuestEnvironment's: a guest refusal is the model's action failing; any other fault
        latches the run invalid in the environment itself."""
        if not by_guest and not getattr(self, "_invalid_run", None):
            self._invalid_run = {
                "category": "environment",
                "reason": "action_execution_failed",
                "detail": message,
            }
        return Execution(
            "failed",
            {"error": message},
            rejection=message if by_guest else None,
            error=None if by_guest else message,
        )

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
        it is answered. A reply that comes due at the same instant as another wait, such as one
        landing on a frame boundary, comes back first (FakeClock.sleep). True asks the stub to
        hold the request (Hang). A request beyond the script takes no time, so the stub's own
        refusal of it (stub.py) fails the decision at once."""
        if index >= len(self.reply_latencies):
            self.log("model.request", request=index, seconds=None, hang=False)
            return False
        latency = self.reply_latencies[index]
        hang = isinstance(latency, Hang)
        seconds = latency.seconds if hang else latency
        self.log("model.request", request=index, seconds=seconds, hang=hang)
        if not hang:
            self.clock.sleep(seconds, first=True)
        return hang
