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

The gate is the shell's lock on the environment. The task code is not thread-safe and does not
change in phase 2: a game's end check folds the state it reads into its peak score unlocked, as its
snapshot does (realtime_tasks/game/runtime.py:536-543, 548, 587), and the environment adapter
latches its end and its invalid run unlocked (GuestEnvironment.check_end, _execution_failed). A
capture only reads the task (sandbox/observation/capture.py:228, 261); an action's calls into it
are its page barriers (sandbox/actions.py:1022-1025, 1046), and only an action writes what the
action lane reads after it (actions.py:930-941, 978-1063). So a capture and an action share the
gate and run at once, and new frames come in while a long action runs (phase2.md, native
streaming). An end check, the snapshot and the release of the inputs hold it alone.

Sharing the gate is not sharing the browser's connection. A CDP call's timeout spends the wait for
its session's lock (sandbox/browser/cdp.py PersistentCDPClient), which the action's page barriers
and focus check hold on the session every caller shares; so the capture's DOM and AX probes run on
a session of their own on the same page (decision N33), which the lanes open. Still one at a time:
the page runs the two sessions' scripts in turn, and the task's own check of a captured frame
(``captured_frame_is_current``) stays on the shared session.

A lane takes its place in the gate's line when it is given the command, so the environment runs
the kernel's commands in the order the kernel sent them, also those it sent in one step to two
lanes. One kind goes out of turn: while an action holds the gate, a capture goes past what waits to
hold it alone, which waits for that action anyway. What waits to hold the gate alone so waits for
the action and for the captures that run alongside it, each within its own budget, and frames
keep coming meanwhile. Waiting for the gate is a wait on the clock, so a fake clock moves on while
an operation that holds it waits.

Like the turn-based shell, the gate compares ``_invalid_run`` before and after each operation and
posts what the environment latched itself before the operation's answer; it posts each record
once. The wall clock's key-release timer latches one on its own thread and reports it through
``on_invalid`` (GuestEnvironment._wall_release_due), which the shell points at ``latched``.

Answers. A capture's is what the shell makes of the Observation: the shell keeps the frame for the
agent and knows its identity for the frozen-stream check. An action's is ActionExecuted, read from
the environment right after the action, before the next action can change what it read
(``_last_exec_error``, ``_last_rejected_by_guest``, ``_last_execution_result``, ``last_held_keys``,
``last_end_checks``) with when it held the gate and when it returned; its coordinate trace and
those times go to the shell as Executed before the answer. An end check's is EndChecked. An
environment operation that raised is answered with the error, as the turn-based shell answers one.
What the shell's own code does with an answer (``captured``, ``executed``) raising is the shell's
failure, not the environment's: it goes to the shell's ``raised`` first, which makes it a
run_exception as the turn-based shell does (shell.py ``_raised``). A capture is then answered as
failed; an action with its answer all the same, since it ran.

End checks run on a lane of their own (stream-plan 4.9). The streaming policy sends one only
while no capture or action is out, and neither while one is out (kernel/streaming.py ``_proceed``),
so an end check does not wait behind them for the gate; the gate holds it alone all the same.

At the end the snapshot waits for the action executing to finish (stream-plan 4.8) and goes
right after it: ``freeze_turn`` gives it the place in the gate's line next after that action,
ahead of the others that were waiting, as the turn-based shell takes it right after the action on
its one thread. Its wait is bounded by whoever takes it: the streaming shell's bound is the larger
of the action's budget and a capture's, for a capture alongside the action (N32).
"""

from __future__ import annotations

import functools
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

# What an operation needs of the gate: a capture and an action share it, the rest hold it alone.
CAPTURE, ACTION, ALONE = "capture", "action", "alone"
# The CDP session of the captures' probes (sandbox/browser/cdp.py client_for).
CAPTURE_CDP_SESSION = "capture"


class EnvGate:
    """The shell's lock on the environment, and the invalid runs the environment latched."""

    def __init__(self, env, post: Callable[[object], None], clock) -> None:
        self._env, self._post, self._clock = env, post, clock
        # Guards the fields below and is never held across a wait.
        self._mutex = threading.Lock()
        self._holders: dict[threading.Event, str] = {}  # turn -> its kind
        self._line: deque[tuple[threading.Event, str]] = deque()  # in the order they asked
        self._posted: dict | None = None

    def reserve(
        self, kind: str = ALONE, *, after: threading.Event | None = None, ahead=False
    ) -> threading.Event:
        """A place in the gate's line for an operation of ``kind``, which ``call`` waits for: at
        the line's end, or ``ahead`` of every other waiter but ``after`` while that one waits."""
        turn = self._clock.event()
        with self._mutex:
            turns = [waiting for waiting, _ in self._line]
            if not ahead:
                self._line.append((turn, kind))
            elif after in turns:
                self._line.insert(turns.index(after) + 1, (turn, kind))
            else:
                self._line.appendleft((turn, kind))
            self._grant()
        return turn

    def call(
        self,
        operation: Callable[..., Any],
        *args,
        turn: threading.Event | None = None,
        wait: float | None = None,
    ):
        """Run one environment operation under the gate once its ``turn`` comes, a place it takes
        now unless one is given; TimeoutError when it does not come within ``wait`` seconds. An
        invalid run the environment latched itself, with its own reason, goes to the kernel
        before the operation returns."""
        turn = self.reserve() if turn is None else turn
        # The holder hands the gate on by setting the next turn.
        if not self._clock.wait(turn, wait) and not self._leave(turn):
            raise TimeoutError(f"the environment was still busy {wait:g}s later")
        try:
            latched = getattr(self._env, "_invalid_run", None)
            try:
                return operation(*args)
            finally:
                invalid = getattr(self._env, "_invalid_run", None)
                if invalid is not None and invalid is not latched:
                    self.latched(invalid)
        finally:
            with self._mutex:
                del self._holders[turn]
                self._grant()

    def latched(self, record: dict) -> None:
        """Post the invalid-run record the environment latched itself, unless it was posted.
        Posted under the mutex: an operation that finds the record claimed must not post its
        answer before the claim's post. The kernel's inbox takes no gate lock, so no lock is
        taken the other way round."""
        with self._mutex:
            if record is self._posted:
                return
            self._posted = record
            self._post(Invalidated(record["category"], record["reason"], record.get("detail")))

    def _grant(self) -> None:
        """Give the gate to the waiters that may have it now, in line order (caller holds the
        mutex)."""
        alone_waits = False
        for entry in list(self._line):
            turn, kind = entry
            holding = set(self._holders.values())
            if ALONE in holding:
                return
            if kind == ALONE:
                if self._holders or alone_waits:
                    alone_waits = True
                    continue
            elif alone_waits and not (kind == CAPTURE and ACTION in holding):
                continue
            self._line.remove(entry)
            self._holders[turn] = kind
            turn.set()

    def _leave(self, turn: threading.Event) -> bool:
        """Leave the line after a wait ran out; True when the gate came to ``turn`` meanwhile."""
        with self._mutex:
            if turn.is_set():
                return True
            self._line.remove(next(entry for entry in self._line if entry[0] is turn))
            self._grant()
            return False


class Lane:
    """One thread that runs one command at a time and posts the answer each one returns."""

    def __init__(
        self,
        name: str,
        post: Callable[[object], None],
        clock,
        gate: EnvGate | None = None,
        kind: str = ALONE,
    ) -> None:
        self.name = name
        self._post, self._clock, self._gate, self._kind = post, clock, gate, kind
        self._mutex = threading.Lock()
        self._command: Callable[[], object] | None = None
        self._closed = False
        self._ready = clock.event()
        self._thread = clock.start_thread(self._run, name=f"env-lane-{name}")

    def submit(self, command: Callable[..., object]) -> threading.Event | None:
        """Run ``command`` next; it returns the answer to post and never raises. On the gate, the
        lane takes its place in the gate's line now, passes it as ``command(turn)`` and returns
        it."""
        turn = None
        with self._mutex:
            assert not self._closed, f"the {self.name} lane is closed"
            assert self._command is None, f"the {self.name} lane is still running a command"
            if self._gate is not None:
                turn = self._gate.reserve(self._kind)
                command = functools.partial(command, turn)
            self._command = command
        self._ready.set()
        return turn

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
                command = self._command
            if command is not None:
                answer = command()
                with self._mutex:
                    self._command = None
                self._post(answer)
            # A close that came with the command or while it ran woke the thread only once.
            with self._mutex:
                if self._closed and self._command is None:
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
    ``clock`` is the operational clock; ``post`` is the shell's, which hands an event to the
    kernel (Kernel.post) and wakes the shell; ``raised(failure)`` takes a traceback of the shell's
    own code that raised on a lane."""

    def __init__(
        self, env, post: Callable[[object], None], clock, *, raised: Callable[[str], None]
    ):
        self.env = env
        self._clock = clock
        self._raised = raised
        if env.scene_capture is not None:  # None for a desktop task
            env.scene_capture.cdp_session = CAPTURE_CDP_SESSION
        self.gate = EnvGate(env, post, clock)
        self._capture = Lane("capture", post, clock, self.gate, CAPTURE)
        self._action = Lane("action", post, clock, self.gate, ACTION)
        self._end_check = Lane("end-check", post, clock, self.gate, ALONE)
        self.executing: Execute | None = None  # the command the action lane runs
        self._executing_turn: threading.Event | None = None  # its place at the gate

    def capture(self, captured: Callable[[Observation], FrameCaptured]) -> None:
        """Capture: answer with ``captured(observation)``, which the lane calls."""

        def run(turn):
            try:
                observation = self.gate.call(self.env.capture, turn=turn)
            # Also a BaseException (entity/model.py RetryCancelled): the answer must come back,
            # or the kernel would wait for it.
            except BaseException:  # noqa: BLE001 - the kernel invalidates the run for it
                return FrameCaptured(error=traceback.format_exc())
            try:
                return captured(observation)
            except BaseException:  # noqa: BLE001 - the shell's own failure
                failure = traceback.format_exc()
                self._raised(failure)
                return FrameCaptured(error=failure)

        self._capture.submit(run)

    def execute(self, command: Execute, executed: Callable[[Executed], None]) -> None:
        """Execute: hand ``executed`` what the action lane read, then answer with
        ActionExecuted."""

        def run(turn):
            try:
                try:
                    answer, record = self.gate.call(self._act, command, turn=turn)
                except BaseException:  # noqa: BLE001 - the kernel invalidates the run for it
                    return ActionExecuted(command.decision, error=traceback.format_exc())
                try:
                    executed(record)
                except BaseException:  # noqa: BLE001 - the shell's own failure; the action ran
                    self._raised(traceback.format_exc())
                return answer
            finally:
                self.executing = self._executing_turn = None

        self.executing = command
        self._executing_turn = self._action.submit(run)

    def freeze_turn(self) -> tuple[threading.Event, Execute | None]:
        """The snapshot's place at the gate, alone and next after the action executing, ahead of
        every other operation waiting; and that action, None when none executes."""
        executing, turn = self.executing, self._executing_turn
        return self.gate.reserve(ALONE, after=turn, ahead=True), executing

    def check_end(self, command: CheckEnd) -> None:
        def run(turn):
            try:
                ended, reason = self.gate.call(self.env.check_end, turn=turn)
            except BaseException:  # noqa: BLE001 - the kernel counts the failed read
                return EndChecked(command.request, error=traceback.format_exc())
            return EndChecked(command.request, ended, reason)

        self._end_check.submit(run)

    def close(self) -> None:
        for lane in (self._capture, self._action, self._end_check):
            lane.close()

    def _act(self, command: Execute) -> tuple[ActionExecuted, Executed]:
        """Run the action and read what it left on the environment, which only an action
        writes."""
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
            # Semantic seconds: on the wall clock the operational clock is the semantic one.
            started_at=started_at - env.t0,
            finished_at=finished_at - env.t0,
        )
        record = Executed(command.decision, env._last_coordinate_trace, started_at, finished_at)
        return answer, record
