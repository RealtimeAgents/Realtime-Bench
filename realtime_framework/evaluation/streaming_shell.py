"""The shell of a native-streaming episode: the event kernel with the streaming policy.

``run_streaming_episode`` drives ``Kernel(StreamingPolicy(config))`` (evaluation/kernel/streaming.py)
as evaluation/shell.py drives the turn-based policy: it reads the clock, runs the commands each
``step`` returns, posts their answers, and in between waits for a post or for the next timer the
kernel set. What differs is where the commands run: the shell's own thread never waits for the
environment or for the session (phase2.md, native streaming). One command waits for the gate on
it, ReleaseInputs, which the kernel sends once every operation it sent is answered: the gate is
free by then.

- Captures, actions and end checks run on the environment lanes, behind the shell's gate
  (evaluation/env_lanes.py): a capture and an action share it, so frames keep coming during a
  long action, and an end check, on a lane of its own (stream-plan 4.9), holds it alone.
- A frame goes into the session on the push lane. A reply runs on a worker until it ends; the agent
  posts its actions, and the calls the provider cancels, as they appear.
- Feedback and accepting or discarding a reply run on the shell's thread. Giving a reply up runs
  on the abort lane, for as long as the agent takes to clean the session up and close it, and is
  answered then (GenerationAborted), so that the next reply rebuilds the session; the episode goes
  on meanwhile.
- A frame the capture cannot keep up with is skipped and counted by the policy (4.10), which never
  hands the capture lane a second capture.
- At the end the snapshot is taken on a thread of its own, next after the action executing and
  ahead of the reads waiting for the gate, as the turn-based shell takes it right after the action
  on its one thread (4.8). It holds the gate alone, so it also waits for a capture that runs
  alongside that action: at most the larger of the action's budget and a capture's (N32). An
  operation still running then fails the snapshot, and the run is invalid. It is read in its own
  budget.

Wall clock only: ``semantic`` is ``clocks.semantic.time() - env.t0``, and the two clocks are one.

Records. Every kernel record goes into the episode trace with ``kernel_seq``, ``semantic`` and
``ops``: RunInvalidated under its name, as the turn-based shell writes it, and every other one as
``KernelRecord`` ``{"kind", "data"}``. When each action held the gate and when it returned, with
its coordinate trace, as the action lane read them, are written as ActionExecution, on the
shell's thread ahead of the next kernel record: the lane posts the action's answer without
waiting for the trace, whose writers share a lock and which a checkpoint holds while it syncs,
so that an action finished just before the deadline is not taken in after it.
"""

from __future__ import annotations

import contextlib
import heapq
import threading
import traceback
from collections import deque
from collections.abc import Callable
from typing import Protocol

from realtime_framework.clock import Clocks
from realtime_framework.entity.actions import Observation
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.env_lanes import EnvLanes, Executed, Lane
from realtime_framework.evaluation.episode_end import (
    EpisodeEnd,
    EpisodeResult,
    TraceWriter,
    escaped_termination,
    invalid_line,
    invalid_record,
    only_logged,
    record_run_exception,
    with_unheard,
)
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
    GenerationAborted,
    PushFrame,
    StreamingConfig,
    StreamingPolicy,
)
from realtime_framework.evaluation.shell import KernelAgent, ShellEnvironment
from realtime_framework.sandbox.actions import ENVIRONMENT_BUDGET
from realtime_framework.sandbox.budget import operation_budget
from realtime_framework.sandbox.guest import action_timeout


class StreamingKernelAgent(KernelAgent, Protocol):
    """The agent of a native-streaming episode: frames go into its live session as they are
    captured, and a reply hands its actions over as they appear. ``generate`` starts one reply on
    the newest frame pushed, which it takes for its records only, and returns once the reply
    ended; ``commit`` reports one action's outcome, or a reply's that came to none."""

    def bind_inbox(self, post: Callable[[object], None], origin: float) -> None:
        """Before the episode: where a reply's ActionArrived and CallsCancelled go
        (kernel/streaming.py), each posted as its message comes and before ``generate``
        returns, SessionPreparing and SessionPrepared as the session's preparation starts
        and ends, and FrameFoundInvalid before a push raises for it; and the episode's origin on
        the wall clock (``env.t0``), from which the semantic seconds the actions carry count."""

    def push(self, observation: Observation) -> None:
        """PushFrame: put the frame into the session. Raises when it did not go out; only that
        frame is lost."""

    def end_pushes(self) -> None:
        """The episode ended, before the end's abort closes the session: a push still out gives
        up, and one that abort fails is no failed push."""


def run_streaming_episode(
    agent: StreamingKernelAgent,
    env: ShellEnvironment,
    config: StreamingConfig,
    *,
    clocks: Clocks,
    log: Callable[[str], object] = print,
    trace=None,
) -> EpisodeResult:
    """Run one streaming episode; return how it came out (episode_end.py)."""
    return _StreamingShell(agent, env, config, clocks, log, trace).run()


class _StreamingShell:
    def __init__(self, agent, env, config: StreamingConfig, clocks: Clocks, log, trace) -> None:
        self.agent, self.env, self.clocks = agent, env, clocks
        self.log = log
        self.writer = TraceWriter(trace, self._trace_lost)
        self.kernel = Kernel(StreamingPolicy(config))
        clock = clocks.operational
        self.lanes = EnvLanes(env, self.post, clock, raised=self._raised)
        self._push_lane = Lane("push", self.post, clock)
        self._abort_lane = Lane("abort", self.post, clock)
        self._wake = clock.event()
        self._now = 0.0  # the reading of the last step
        self._timers: list[float] = []  # heap of when the timers the kernel set come due
        # The frames captured from the newest one a reply started on, in capture order.
        self._frames: dict[str | None, Observation] = {}
        self._frames_lock = threading.Lock()
        self._executions: deque[Executed] = deque()  # from the action lane, not written yet
        self._replies: dict[int, threading.Event] = {}  # generation -> set once its worker ended
        self._frozen: threading.Event | None = None  # set once the snapshot FreezeSnapshot took
        self._frozen_at: float | None = None  # the reading it was frozen at
        self._snapshot_read = False
        self._unheard: Invalidated | None = None  # one found once the kernel had finished
        self.end = EpisodeEnd()
        self._loop_error_lock = threading.Lock()  # lanes raise on threads of their own

    def run(self) -> EpisodeResult:
        env = self.env
        env.on_invalid = self.lanes.gate.latched
        self.agent.bind_inbox(self.post, env.t0)
        try:
            self._now = self._read()
            self._run_all(self.kernel.start(self._now))
            while self.end.termination is None:
                # Cleared before the step takes the inbox: a post the step misses sets it again.
                self._wake.clear()
                self._now = self._read()
                commands = self.kernel.step(self._now)
                if commands is None:
                    self._sleep()
                else:
                    self._run_all(commands)
        except Exception:  # noqa: BLE001 - the run is invalid and the episode still finalizes
            self._wrap_up(traceback.format_exc())
        return self.end.result(env, [])

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
        with self._loop_error_lock:
            if self.end.loop_error is None:
                self.end.loop_error = failure
        self.post(Invalidated("runner", "run_exception", failure))
        self._say(f"[streaming] shell raised:\n{failure}")

    def _say(self, message: str) -> None:
        """Log from a thread that owes the kernel an answer: a log that raises (its stdout gone)
        loses the line, not the answer."""
        with contextlib.suppress(Exception):
            self.log(message)

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
                self._abort_lane.submit(lambda: self._abort(generation))
            case AcceptResult(generation=generation):
                self.agent.accept(generation)
            case DiscardResult(generation=generation):
                self.agent.discard(generation)
            case Execute():
                self.lanes.execute(command, self._executed)
            case Commit(outcome=outcome):
                self._commit(outcome)
            case FreezeSnapshot():
                # The end's first command: its AbortGeneration follows in this batch.
                self.agent.end_pushes()
                self._frozen = self.clocks.operational.event()
                self.clocks.operational.start_thread(self._freezing, name="snapshot")
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
            case AbortGeneration(generation=generation):
                return GenerationAborted(generation)
            case Execute(decision=decision):
                return ActionExecuted(decision, error=failure)
            case Commit(outcome=outcome):
                return OutcomeCommitted(outcome.decision, failure)
            case FreezeSnapshot():
                if self._frozen is not None:
                    self._frozen.set()  # its thread never ran: nothing is left to wait for
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
            self._say(f"[streaming] reply {command.generation} failed:\n{failure}")
            r = episode_records.generation_error(round(clock.time() - started, 2), failure)
            result = GenerationResult(invalid=episode_records.result_invalid(r, self.agent, "wall"))
        done.set()
        self.post(GenerationFinished(command.generation, result))

    def _abort(self, generation: int) -> GenerationAborted:
        """Give the reply up, on the abort lane (on the shell's thread in the wrap-up), and answer
        whatever happens, as a command is answered (_run). Also a reply whose worker just ended:
        the abort closes the session, which the policy gives a reply up for (kernel/streaming.py
        ``_session_stuck``), and fails a push stuck in it."""
        if generation in self._replies:  # not when its StartGeneration raised
            # Also a BaseException (entity/model.py RetryCancelled): the answer must come back,
            # or the kernel would wait for it.
            try:
                self.agent.abort(generation)
            except BaseException:  # noqa: BLE001 - interrupting is best effort, as in the runner
                self._say(
                    f"[streaming] abort of reply {generation} failed:\n{traceback.format_exc()}"
                )
        return GenerationAborted(generation)

    # ---- actions ------------------------------------------------------------------------------

    def _executed(self, record: Executed) -> None:
        """On the action lane, before its answer: kept for the shell's thread to write."""
        self._executions.append(record)

    def _write_executions(self) -> None:
        """The ActionExecution rows the action lane handed over, on the shell's thread: before
        the kernel record of an answer, which the lane posts after handing its row over."""
        while self._executions:
            record = self._executions.popleft()
            self.writer.emit(
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
        self.writer.checkpoint()

    # ---- the end ------------------------------------------------------------------------------

    def _freeze(self) -> None:
        """The snapshot, next after the action executing (4.8), on its own thread (_freezing). A
        read that raises fails the snapshot, a cancellation's too."""
        end = self.end
        # The presentation boundary precedes the scorer's snapshot; it moves to when the read
        # begins, once the gate is the snapshot's.
        end.snapshot_begins(self.clocks.operational.time())
        # The Task base class has none (realtime_tasks/types.py).
        snapshotter = getattr(self.env.task, "evaluation_snapshot", None)
        error = None
        if snapshotter is not None:
            turn, executing = self.lanes.freeze_turn()
            # The action, and a capture alongside it (N32). No valid action's budget is past a
            # capture's, so the wait is capture_s: text input's equals it, and a timed action's is
            # at most 20 s (sandbox/guest.py action_timeout). The action's counts if capture_s is
            # ever below text input's.
            wait = (
                None
                if executing is None
                else max(
                    action_timeout(executing.action, ENVIRONMENT_BUDGET.action_s),
                    ENVIRONMENT_BUDGET.capture_s,
                )
            )
            # Also a BaseException (entity/model.py RetryCancelled).
            try:
                snapshot = self.lanes.gate.call(
                    self._read_snapshot, snapshotter, turn=turn, wait=wait
                )
            except BaseException:  # noqa: BLE001 - the kernel invalidates the run for it
                error = traceback.format_exc()
                end.snapshot_read(None, error)
            else:
                end.snapshot_read(snapshot, None)
                self._snapshot_read = True
        self.lanes.gate.freeze()
        at = self._read()  # when the read returned
        self._frozen_at = at
        self.post(SnapshotFrozen(at, at, error))

    def _freezing(self) -> None:
        """The snapshot's thread. Its answer comes back whatever happens, as a command's does
        (_run), and whoever waits for the snapshot is let go."""
        try:
            self._freeze()
        except Exception:  # noqa: BLE001 - the run is invalid and the episode still finalizes
            failure = traceback.format_exc()
            self._raised(failure)
            self.post(SnapshotFrozen(self._now, self._now, failure))
        finally:
            self._frozen.set()

    def _read_snapshot(self, snapshotter):
        clock = self.clocks.operational
        # After the action, as on the turn-based shell.
        self.end.active_end_unix_s = clock.time()
        with operation_budget("snapshot read", ENVIRONMENT_BUDGET.snapshot_read_s, clock=clock):
            return snapshotter()

    def _finish(self, termination: Termination) -> None:
        """The termination record main() and the report read, in seconds to nine places, and a
        failure the shell found in this last batch goes into it (episode_end.py with_unheard).
        The episode is over whatever comes of writing it, and its verdict goes first: main() is
        handed it whatever else could be written."""
        termination = with_unheard(termination, self._unheard, self.log)
        late = None
        try:
            late = self.end.terminate(termination, self.env, self._snapshot_read)
        finally:
            self.lanes.close()
            self._push_lane.close()
            self._abort_lane.close()
        if late is not None:
            self.log(only_logged(late))

    def _wrap_up(self, failure: str) -> None:
        """The kernel raised (N11): the episode is wrapped up without it, as the turn-based shell
        does (evaluation/shell.py _Shell._wrap_up). The pushes and the replies running are given
        up, the snapshot taken unless FreezeSnapshot took it, the inputs released and the
        termination written. A failure on the way is only logged."""
        self._raised(failure)
        # What the environment latched by now came first; a failure during the wrap-up after.
        latched = self.end.latched(self.env)
        first = self.kernel.invalid is None and latched is None
        try:
            self.agent.end_pushes()
            for generation, done in list(self._replies.items()):
                if not done.is_set():
                    self._abort(generation)
            if self._frozen is None:
                self._frozen = self.clocks.operational.event()
                self._freezing()
            else:
                self.clocks.operational.wait(self._frozen, None)
            self._write_executions()  # of an action the kernel did not take the answer of
            self._handle(ReleaseInputs())
            termination = escaped_termination(
                self.kernel, self._now, self._now, self._frozen_at, self.end.loop_error, latched
            )
            self._finish(termination)
            if first:
                record_run_exception(
                    self.writer, termination.invalid, self._now, self._now, self.log
                )
        except Exception:  # noqa: BLE001 - the episode is over either way
            self.log("[streaming] wrapping up the episode failed:\n" + traceback.format_exc())
        if self.end.termination is None:  # how it ended, as far as the wrap-up got
            self.end.termination = escaped_termination(
                self.kernel, self._now, self._now, self._frozen_at, self.end.loop_error, latched
            )

    # ---- the episode trace --------------------------------------------------------------------

    def _record(self, entry: Entry) -> None:
        self._write_executions()
        if entry.kind == TraceEvent.RUN_INVALIDATED.value:
            invalid = Invalidated(**entry.data)
            if invalid.reason == "observation_stream_frozen":
                # What main() reads for the run summary, as the turn-based shell counts it.
                self.end.stream_freeze_count += 1
            self.log(invalid_line(invalid))
            self.writer.emit(TraceEvent.RUN_INVALIDATED, invalid_record(invalid), entry)
        else:
            self.writer.emit("KernelRecord", {"kind": entry.kind, "data": entry.data}, entry)

    def _trace_lost(self, invalid: Invalidated) -> None:
        """Post the trace's failure, which the shell finds on its own thread. A kernel that
        finished in the step whose batch this is takes no more events: the first such failure
        goes with its Finish (_finish)."""
        if self.kernel.termination is None:
            self.post(invalid)
        elif self._unheard is None:
            self._unheard = invalid
