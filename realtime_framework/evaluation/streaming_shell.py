"""The shell of a native-streaming episode: the event kernel with the streaming policy.

``run_streaming_episode`` drives ``Kernel(StreamingPolicy(config))`` (evaluation/kernel/streaming.py)
as evaluation/shell.py drives the turn-based policy: it reads the clock, runs the commands each
``step`` returns, posts their answers, and in between waits for a post or for the next timer the
kernel set. What differs is where the commands run: the shell's own thread never waits for the
environment or for the session (phase2.md, native streaming).

- Captures, actions and end checks run on the environment lanes, behind the shell's gate
  (evaluation/env_lanes.py): a capture and an action share it, so frames keep coming during a
  long action, and an end check, on a lane of its own (stream-plan 4.9), holds it alone.
- A frame goes into the session on the push lane. A reply runs on a worker until it ends; the agent
  posts its actions, and the calls the provider cancels, as they appear.
- Feedback, accepting or discarding a reply and giving one up run on the shell's thread. Giving up
  returns once the agent has cleaned the session up, so the next reply rebuilds it.
- A frame the capture cannot keep up with is skipped and counted by the policy (4.10), which never
  hands the capture lane a second capture.
- At the end the snapshot is taken on a thread of its own, next after the action executing and
  ahead of the reads waiting for the gate, as the turn-based shell takes it right after the action
  on its one thread (4.8). It waits for that action at most the action's budget; an action still
  running then fails the snapshot, and the run is invalid. It is read in its own budget.

Wall clock only: ``semantic`` is ``clocks.semantic.time() - env.t0``, and the two clocks are one.

Records. Every kernel record goes into the episode trace with ``kernel_seq``, ``semantic`` and
``ops``: RunInvalidated under its name, as the turn-based shell writes it, and every other one as
``KernelRecord`` ``{"kind", "data"}``. When each action held the gate and when it returned, with
its coordinate trace, as the action lane read them, are written as ActionExecution.
"""

from __future__ import annotations

import heapq
import threading
import traceback
from collections.abc import Callable
from typing import Protocol

from realtime_framework.clock import Clocks
from realtime_framework.entity.actions import Observation
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.env_lanes import EnvLanes, Executed, Lane
from realtime_framework.evaluation.kernel.core import Kernel
from realtime_framework.evaluation.kernel.events import (
    AbortGeneration,
    AcceptResult,
    ActionExecuted,
    Capture,
    CheckEnd,
    Commit,
    DiscardResult,
    EndChecked,
    Entry,
    Execute,
    Finish,
    FrameCaptured,
    FreezeSnapshot,
    GenerationFinished,
    GenerationResult,
    Invalidated,
    Outcome,
    OutcomeCommitted,
    Record,
    ReleaseInputs,
    SetTimer,
    SnapshotFrozen,
    StartGeneration,
    Termination,
)
from realtime_framework.evaluation.kernel.streaming import (
    FramePushed,
    PushFrame,
    StreamingConfig,
    StreamingPolicy,
)
from realtime_framework.evaluation.shell import KernelAgent, ShellEnvironment, _invalid_record
from realtime_framework.sandbox.actions import ENVIRONMENT_BUDGET
from realtime_framework.sandbox.budget import operation_budget
from realtime_framework.sandbox.guest import action_timeout


class StreamingKernelAgent(KernelAgent, Protocol):
    """The agent of a native-streaming episode: frames go into its live session as they are
    captured, and a reply hands its actions over as they appear. ``generate`` starts one reply on
    the newest frame pushed, which it takes for its records only, and returns once the reply
    ended; ``commit`` reports one action's outcome, or a reply's that came to none."""

    def bind_inbox(self, post: Callable[[object], None]) -> None:
        """Before the episode: where a reply's ActionArrived and CallsCancelled go
        (kernel/streaming.py), each posted as its message comes and before ``generate``
        returns."""

    def push(self, observation: Observation) -> None:
        """PushFrame: put the frame into the session. Raises when it did not go out; only that
        frame is lost."""


def run_streaming_episode(
    agent: StreamingKernelAgent,
    env: ShellEnvironment,
    config: StreamingConfig,
    *,
    clocks: Clocks,
    log: Callable[[str], object] = print,
    trace=None,
) -> Termination:
    """Run one streaming episode; return how it ended. What main() reads afterwards is left on
    ``env`` as the turn-based shell leaves it: the stop reason and the termination record
    (``_stop_reason``, ``_termination``), the frozen snapshot (``_termination_snapshot*``,
    ``_active_end_unix_s``), ``_invalid_run`` and ``_loop_error``."""
    return _StreamingShell(agent, env, config, clocks, log, trace).run()


class _StreamingShell:
    def __init__(self, agent, env, config: StreamingConfig, clocks: Clocks, log, trace) -> None:
        self.agent, self.env, self.clocks = agent, env, clocks
        self.log, self.trace = log, trace
        self.kernel = Kernel(StreamingPolicy(config))
        clock = clocks.operational
        self.lanes = EnvLanes(env, self.post, clock, raised=self._raised)
        self._push_lane = Lane("push", self.post, clock)
        self._wake = clock.event()
        self._now = 0.0  # the reading of the last step
        self._timers: list[float] = []  # heap of when the timers the kernel set come due
        # The frames captured from the newest one a reply started on, in capture order.
        self._frames: dict[str | None, Observation] = {}
        self._frames_lock = threading.Lock()
        self._replies: dict[int, threading.Event] = {}  # generation -> set once its worker ended
        self._snapshot_read = False
        self._trace_failed = False
        self._loop_error: str | None = None
        self._loop_error_lock = threading.Lock()  # lanes raise on threads of their own
        self._termination: Termination | None = None

    def run(self) -> Termination:
        env = self.env
        env.kernel_counts_reads = True
        env.on_invalid = self.lanes.gate.latched
        self.agent.bind_inbox(self.post)
        self._now = self._read()
        self._run_all(self.kernel.start(self._now))
        while self._termination is None:
            # Cleared before the step takes the inbox: a post the step misses sets it again.
            self._wake.clear()
            self._now = self._read()
            commands = self.kernel.step(self._now)
            if commands is None:
                self._sleep()
            else:
                self._run_all(commands)
        return self._termination

    def post(self, event) -> None:
        """Hand the kernel an event, from any thread, and wake the shell."""
        self.kernel.post(event)
        self._wake.set()

    def _read(self) -> float:
        return self.clocks.semantic.time() - self.env.t0

    def _sleep(self) -> None:
        """Wait for a post or for the next timer. A timer the kernel cancelled stays in the heap
        and wakes the shell for nothing."""
        while self._timers and self._timers[0] <= self._now:
            heapq.heappop(self._timers)
        timeout = self._timers[0] - self._now if self._timers else None
        self.clocks.operational.wait(self._wake, timeout)

    # ---- commands -----------------------------------------------------------------------------

    def _run_all(self, commands: list) -> None:
        for command in commands:
            self._run(command)

    def _run(self, command) -> None:
        try:
            self._handle(command)
        except Exception:  # noqa: BLE001 - the run is invalid and the episode still finalizes
            failure = traceback.format_exc()
            self._raised(failure)
            answer = self._unanswered(command, failure)
            if answer is not None:
                self.post(answer)

    def _raised(self, failure: str) -> None:
        """The shell's own code raised, on its thread or on a lane: the run is invalid and the
        episode still finalizes; the first traceback is the loop error (runner.py main)."""
        self.log(f"[streaming] shell raised:\n{failure}")
        with self._loop_error_lock:
            if self._loop_error is None:
                self._loop_error = self.env._loop_error = failure
        self.post(Invalidated("runner", "run_exception", failure))

    def _handle(self, command) -> None:
        match command:
            case Record(entry=entry):
                self._record(entry)
            case SetTimer(at=at):
                heapq.heappush(self._timers, at)
            case CheckEnd():
                self.lanes.check_end(command)
            case Capture():
                self.lanes.capture(self._captured)
            case PushFrame(observation_id=observation_id):
                self._push(observation_id)
            case StartGeneration():
                self._start_reply(command)
            case AbortGeneration(generation=generation):
                self._abort(generation)
            case AcceptResult(generation=generation):
                self.agent.accept(generation)
            case DiscardResult(generation=generation):
                self.agent.discard(generation)
            case Execute():
                self.lanes.execute(command, self._executed)
            case Commit(outcome=outcome):
                self._commit(outcome)
            case FreezeSnapshot():
                self.clocks.operational.start_thread(self._freeze, name="snapshot")
            case ReleaseInputs():
                try:
                    self.lanes.gate.call(self.env.release_inputs)
                except Exception:  # noqa: BLE001 - the snapshot is frozen; cleanup is best effort
                    self.log(
                        "[cleanup] failed to release stateful inputs:\n" + traceback.format_exc()
                    )
            case Finish(termination=termination):
                self._finish(termination)

    def _unanswered(self, command, failure: str):
        """The answer to a command whose run raised, so that the kernel, which waits for it, can
        settle; the run_exception before it decides the reason."""
        match command:
            case CheckEnd(request=request):
                return EndChecked(request, error=failure)
            case Capture():
                return FrameCaptured(error=failure)
            case PushFrame(observation_id=observation_id):
                return FramePushed(observation_id, error=failure)
            case Execute(decision=decision):
                return ActionExecuted(decision, error=failure)
            case Commit(outcome=outcome):
                return OutcomeCommitted(outcome.decision, failure)
            case FreezeSnapshot():
                return SnapshotFrozen(self._now, self._now, failure)
        return None

    # ---- frames -------------------------------------------------------------------------------

    def _captured(self, observation: Observation) -> FrameCaptured:
        """On the capture lane: keep the frame for its push and answer with its identity."""
        observation_id = observation.metadata.get("observation_id")
        with self._frames_lock:
            self._frames[observation_id] = observation
        return FrameCaptured(
            observation_id,
            observation.tick_number,
            observation.timestamp - self.env.t0,
            episode_records.observation_fingerprint(observation),
        )

    def _push(self, observation_id: str | None) -> None:
        with self._frames_lock:
            observation = self._frames[observation_id]

        def run():
            try:
                self.agent.push(observation)
            # Also a BaseException (entity/model.py RetryCancelled): the answer must come back,
            # or the kernel would wait for it.
            except BaseException:  # noqa: BLE001 - that frame is lost; the kernel goes on
                return FramePushed(observation_id, error=traceback.format_exc())
            return FramePushed(observation_id, self._read())

        self._push_lane.submit(run)

    # ---- replies ------------------------------------------------------------------------------

    def _start_reply(self, command: StartGeneration) -> None:
        # The newest frame pushed. Frames captured before it are never pushed or started on again.
        with self._frames_lock:
            for observation_id in list(self._frames):
                if observation_id == command.observation_id:
                    break
                del self._frames[observation_id]
            observation = self._frames[command.observation_id]
        _, budget = episode_records.decision_budget(self.agent, None, "wall")
        clock = self.clocks.operational
        done = self._replies[command.generation] = clock.event()
        clock.start_thread(
            lambda: self._reply(command, observation, budget, done),
            name=f"reply-{command.generation}",
        )

    def _reply(self, command: StartGeneration, observation, budget, done) -> None:
        """The worker: one reply, and what the policy needs of its result."""
        clock = self.clocks.operational
        started = clock.time()
        # Also a cancellation that escaped the agent (entity/model.py RetryCancelled): the result
        # must come back, or the kernel would wait for it.
        try:
            r = self.agent.generate(
                command.generation, command.decision, observation, budget
            ).result
            result = GenerationResult(
                output_tokens=episode_records.frame_output_tokens(r),
                protocol_failure=(r.get("action_protocol") or {}).get("protocol_failure"),
                invalid=episode_records.result_invalid(r, self.agent, "wall"),
            )
        except BaseException:  # noqa: BLE001 - a failed reply invalidates the run
            failure = traceback.format_exc()
            self.log(f"[streaming] reply {command.generation} failed:\n{failure}")
            r = episode_records.generation_error(round(clock.time() - started, 2), failure)
            result = GenerationResult(invalid=episode_records.result_invalid(r, self.agent, "wall"))
        done.set()
        self.post(GenerationFinished(command.generation, result))

    def _abort(self, generation: int) -> None:
        done = self._replies.get(generation)  # None when its StartGeneration raised
        if done is None or done.is_set():
            return  # nothing runs to interrupt; the kernel discards the result
        try:
            self.agent.abort(generation)
        except Exception:  # noqa: BLE001 - interrupting is best effort, as in the runner
            self.log(f"[streaming] abort of reply {generation} failed:\n{traceback.format_exc()}")

    # ---- actions ------------------------------------------------------------------------------

    def _executed(self, record: Executed) -> None:
        """On the action lane, before its answer."""
        self._write(
            "ActionExecution",
            {
                "decision": record.decision,
                "started_unix_s": record.started_at,
                "finished_unix_s": record.finished_at,
                "coordinate_trace": record.coordinate_trace,
            },
        )

    def _commit(self, outcome: Outcome) -> None:
        try:
            self.agent.commit(outcome)
        except Exception:  # noqa: BLE001 - the kernel invalidates the run for it
            failure = traceback.format_exc()
            self.log(f"[streaming] commit({outcome.status}) failed:\n{failure}")
            self.post(OutcomeCommitted(outcome.decision, failure))
        else:
            self.post(OutcomeCommitted(outcome.decision))
        self._checkpoint()

    # ---- the end ------------------------------------------------------------------------------

    def _freeze(self) -> None:
        """On a thread of its own: the snapshot, next after the action executing (4.8)."""
        env, clock = self.env, self.clocks.operational
        # The presentation boundary precedes the scorer's snapshot.
        env._active_end_unix_s = clock.time()
        env._termination_snapshot_captured = False
        # The Task base class has none (realtime_tasks/types.py).
        snapshotter = getattr(env.task, "evaluation_snapshot", None)
        error = None
        if snapshotter is not None:
            turn, executing = self.lanes.freeze_turn()
            wait = (
                None
                if executing is None
                else action_timeout(executing.action, ENVIRONMENT_BUDGET.action_s)
            )
            try:
                snapshot = self.lanes.gate.call(
                    self._read_snapshot, snapshotter, turn=turn, wait=wait
                )
            except Exception:  # noqa: BLE001 - the kernel invalidates the run for it
                error = traceback.format_exc()
            else:
                env._termination_snapshot = snapshot
                env._termination_snapshot_captured = True
                self._snapshot_read = True
        env._termination_snapshot_error = error
        at = self._read()  # when the read returned
        self.post(SnapshotFrozen(at, at, error))

    def _read_snapshot(self, snapshotter):
        clock = self.clocks.operational
        with operation_budget("snapshot read", ENVIRONMENT_BUDGET.snapshot_read_s, clock=clock):
            return snapshotter()

    def _finish(self, termination: Termination) -> None:
        """The termination record main() and the report read, in seconds to nine places."""
        env = self.env
        end_s = termination.ended_at
        snapshot_s = termination.snapshot_at if self._snapshot_read else None
        env._termination = {
            "stop_reason": termination.stop_reason,
            "end_s": round(end_s, 9),
            "snapshot_s": None if snapshot_s is None else round(snapshot_s, 9),
            "snapshot_delay_s": None if snapshot_s is None else round(snapshot_s - end_s, 9),
        }
        env._stop_reason = termination.stop_reason
        if termination.invalid is not None:
            env._invalid_run = _invalid_record(termination.invalid)
        self.lanes.close()
        self._push_lane.close()
        self._termination = termination

    # ---- the episode trace --------------------------------------------------------------------

    def _record(self, entry: Entry) -> None:
        if entry.kind == TraceEvent.RUN_INVALIDATED.value:
            invalid = Invalidated(**entry.data)
            self.log(
                f"[invalid-run] {invalid.category}:{invalid.reason}"
                + (f" — {invalid.detail}" if invalid.detail else "")
            )
            payload = _invalid_record(invalid)
            self._write(TraceEvent.RUN_INVALIDATED, payload, entry)
        else:
            self._write("KernelRecord", {"kind": entry.kind, "data": entry.data}, entry)

    def _write(self, event: TraceEvent | str, payload: dict, entry: Entry | None = None) -> None:
        if self.trace is None or self._trace_failed:
            return
        fields = (
            {}
            if entry is None
            else {"kernel_seq": entry.seq, "semantic": entry.semantic, "ops": entry.ops}
        )
        try:
            self.trace.emit(event, payload, **fields)
        except Exception:  # noqa: BLE001 - any write failure invalidates the run
            self._trace_failed = True
            self.post(Invalidated("harness", "episode_trace_write_failed", traceback.format_exc()))

    def _checkpoint(self) -> None:
        """Make the trace durable at each outcome, as the turn-based shell does at each decision."""
        if self.trace is None or self._trace_failed:
            return
        try:
            self.trace.checkpoint()
        except Exception:  # noqa: BLE001 - any write failure invalidates the run
            self._trace_failed = True
            self.post(
                Invalidated("harness", "episode_trace_checkpoint_failed", traceback.format_exc())
            )
