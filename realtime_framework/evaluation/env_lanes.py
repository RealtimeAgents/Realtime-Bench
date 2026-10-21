"""The lanes native streaming runs the environment's operations on, and the shell's lock.

Under native streaming the shell neither captures frames nor executes actions on its own thread:
each runs on a lane, which posts its answer to the kernel's inbox, so that while a long action
runs the frames and generations that finish are still handled (phase2.md, native streaming). The
turn-based shell runs both on its own thread (evaluation/shell.py).

A lane is one thread, started on the operational clock, that runs the commands it is given one at
a time and posts each one's answer. It takes one command at a time: another for a lane still
running one is the policy's bug and fails an assertion. Which commands to keep, replace or drop
meanwhile is the policy's to decide, not the lane's. A lane is free again before it posts the
answer, so a command that answer brings finds it free.

The gate is the shell's one lock on the environment. The task code is not thread-safe and does
not change in phase 2: a game's end check folds the state it reads into its peak score unlocked
(realtime_tasks/game/runtime.py:536-543, 587), and the environment adapter latches its end and
its invalid run unlocked (GuestEnvironment.check_end, _execution_failed). A capture enters the
task too (sandbox/observation/capture.py:228, 261), as an action does (sandbox/actions.py:931-932),
so every operation holds the gate and no two run at once: a capture waits for an action that holds
it. The gate goes to its waiters in the order they asked, and waiting for it is a wait on the
clock, so a fake clock moves on while an operation that holds it waits.

Like the turn-based shell, the gate compares ``_invalid_run`` before and after each operation and
posts what the environment latched itself before the operation's answer; it posts each record
once. The wall clock's key-release timer latches one on its own thread and reports it through
``on_invalid`` (GuestEnvironment._wall_release_due), which the shell points at ``latched``.

Answers. A capture's is what the shell makes of the Observation: the shell keeps the frame for the
agent and knows its identity for the frozen-stream check. An action's is ActionExecuted, read from
the environment right after the action and still under the gate, before any other operation can
change what it read (``_last_exec_error``, ``_last_rejected_by_guest``, ``_last_execution_result``,
``last_held_keys``, ``last_end_checks``); its coordinate trace and when it started and finished go
to the shell as Executed before the answer. An end check's is EndChecked. A command whose run
raised is answered with the error, as the turn-based shell answers one.

End checks run on a lane of their own (stream-plan 4.9): the policy may send an Execute while a
state read, up to 10 s on the wall clock, is out, and the two take the gate in turn.

At the end the snapshot waits for the action executing to finish (stream-plan 4.8): the lanes say
which one executes (``executing``) and when it is done (``action_done``).
"""

from __future__ import annotations

import threading
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from realtime_framework.entity.actions import Observation
from realtime_framework.evaluation.kernel.events import (
    ActionExecuted,
    CheckEnd,
    EndChecked,
    Execute,
    FrameCaptured,
    Invalidated,
)


class EnvGate:
    """The shell's one lock on the environment, and the invalid runs the environment latched."""

    def __init__(self, env, post: Callable[[object], None], clock) -> None:
        self._env, self._post, self._clock = env, post, clock
        # Guards the fields below and is never held across a wait.
        self._mutex = threading.Lock()
        self._held = False
        self._waiting: deque[threading.Event] = deque()  # one per waiter, in the order it asked
        self._posted: dict | None = None

    def call(self, operation: Callable[..., Any], *args, **kwargs):
        """Run one environment operation under the gate. An invalid run the environment latched
        itself, with its own reason, goes to the kernel before the operation returns."""
        self._acquire()
        try:
            latched = getattr(self._env, "_invalid_run", None)
            try:
                return operation(*args, **kwargs)
            finally:
                invalid = getattr(self._env, "_invalid_run", None)
                if invalid is not None and invalid is not latched:
                    self.latched(invalid)
        finally:
            self._release()

    def latched(self, record: dict) -> None:
        """Post the invalid-run record the environment latched itself, unless it was posted."""
        with self._mutex:
            if record is self._posted:
                return
            self._posted = record
        self._post(Invalidated(record["category"], record["reason"], record.get("detail")))

    def _acquire(self) -> None:
        with self._mutex:
            if not self._held:
                self._held = True
                return
            turn = self._clock.event()
            self._waiting.append(turn)
        # The holder hands the gate on by setting it.
        self._clock.wait(turn, None)

    def _release(self) -> None:
        with self._mutex:
            if self._waiting:
                self._waiting.popleft().set()
            else:
                self._held = False


class Lane:
    """One thread that runs one command at a time and posts the answer each one returns."""

    def __init__(self, name: str, post: Callable[[object], None], clock) -> None:
        self.name = name
        self._post, self._clock = post, clock
        self._mutex = threading.Lock()
        self._command: Callable[[], object] | None = None
        self._closed = False
        self._ready = clock.event()
        clock.start_thread(self._run, name=f"env-lane-{name}")

    def submit(self, command: Callable[[], object]) -> None:
        """Run ``command`` next; it returns the answer to post and never raises."""
        with self._mutex:
            assert not self._closed, f"the {self.name} lane is closed"
            assert self._command is None, f"the {self.name} lane is still running a command"
            self._command = command
        self._ready.set()

    def close(self) -> None:
        """End the thread once the command it runs, if any, is answered."""
        with self._mutex:
            self._closed = True
        self._ready.set()

    def _run(self) -> None:
        while True:
            self._clock.wait(self._ready, None)
            with self._mutex:
                self._ready.clear()
                command, closed = self._command, self._closed
            if command is not None:
                answer = command()
                with self._mutex:
                    self._command = None
                self._post(answer)
            elif closed:
                return


@dataclass(frozen=True)
class Executed:
    """What the action lane read of one action besides its answer: its coordinate trace, and the
    operational clock's wall time once it held the gate and once it returned."""

    decision: int
    coordinate_trace: Any
    started_at: float
    finished_at: float


class EnvLanes:
    """The capture, action and end-check lanes of one streaming episode, behind one gate.
    ``clock`` is the operational clock; ``post`` hands an event to the kernel (Kernel.post)."""

    def __init__(self, env, post: Callable[[object], None], clock):
        self.env = env
        self._clock = clock
        self.gate = EnvGate(env, post, clock)
        self._capture = Lane("capture", post, clock)
        self._action = Lane("action", post, clock)
        self._end_check = Lane("end-check", post, clock)
        self.executing: Execute | None = None  # the command the action lane runs
        self._idle = clock.event()  # set while no action executes
        self._idle.set()

    def capture(self, captured: Callable[[Observation], FrameCaptured]) -> None:
        """Capture: answer with ``captured(observation)``, which the lane calls."""

        def run():
            try:
                return captured(self.gate.call(self.env.capture))
            except Exception:  # noqa: BLE001 - the kernel invalidates the run for it
                return FrameCaptured(error=traceback.format_exc())

        self._capture.submit(run)

    def execute(self, command: Execute, executed: Callable[[Executed], None]) -> None:
        """Execute: hand ``executed`` what the action lane read, then answer with
        ActionExecuted."""

        def run():
            try:
                answer, record = self.gate.call(self._act, command)
                executed(record)
                return answer
            except Exception:  # noqa: BLE001 - the kernel invalidates the run for it
                return ActionExecuted(command.decision, error=traceback.format_exc())
            finally:
                self.executing = None
                self._idle.set()

        self.executing = command
        self._idle.clear()
        self._action.submit(run)

    def action_done(self, wait: float) -> bool:
        """Whether no action executes, or the one that does finishes within ``wait`` seconds."""
        return self._clock.wait(self._idle, wait)

    def check_end(self, command: CheckEnd) -> None:
        def run():
            try:
                ended, reason = self.gate.call(self.env.check_end)
            except Exception:  # noqa: BLE001 - the kernel counts the failed read
                return EndChecked(command.request, error=traceback.format_exc())
            return EndChecked(command.request, ended, reason)

        self._end_check.submit(run)

    def close(self) -> None:
        for lane in (self._capture, self._action, self._end_check):
            lane.close()

    def _act(self, command: Execute) -> tuple[ActionExecuted, Executed]:
        """Run the action and read what it left on the environment; under the gate."""
        env, clock = self.env, self._clock
        env._last_execution_result = None
        env._last_coordinate_trace = None
        started_at = clock.time()
        status = env.execute(command.action, release_at=command.release_at)
        finished_at = clock.time()
        rejection = error = None
        if status == "failed":
            # A guest refusal is the model's action failing, not the environment's.
            if env._last_rejected_by_guest:
                rejection = env._last_exec_error or ""
            else:
                error = env._last_exec_error
        answer = ActionExecuted(
            command.decision,
            result=env._last_execution_result,
            held_keys=env.last_held_keys,
            rejection=rejection,
            error=error,
            end_checks=env.last_end_checks,
        )
        record = Executed(command.decision, env._last_coordinate_trace, started_at, finished_at)
        return answer, record
