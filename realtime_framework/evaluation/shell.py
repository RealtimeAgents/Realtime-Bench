"""The shell that runs the event kernel, and what it needs from the environment and the agent.

The shell drives ``Kernel(TurnPolicy(config))`` (evaluation/kernel): it reads the clocks, runs the
commands each ``step`` returns against a ShellEnvironment and a KernelAgent, and answers them with
the events in evaluation/kernel/events.py. ``run_episode`` is the shell of the wall clock and of
the token clock and pause mode alike; only its clock readings differ. A batch of commands runs in
order on the shell's thread before the next ``step``; a generation runs on a worker.

Waiting. Every event for the kernel goes through the shell's ``post``, which also sets the shell's
wake-up Event. The shell steps the kernel until nothing is due, then waits for that Event or for
the earliest timer the kernel set. It never polls: the kernel files an event at the reading of the
step that takes it (kernel/core.py ``step``), so polling would file a result that came just before
the deadline after it.

Clock readings. On the wall clock ``semantic`` is ``clocks.semantic.time() - env.t0``, seconds
since the episode origin, and ``ops`` is None: the two clocks are one. On the token clock, and
paused, ``semantic`` is ``env.tick_index()``, environment ticks, and ``ops`` is operational seconds
since the kernel started; it must be passed, or the kernel would time its watchdogs in ticks
(kernel/core.py ``_read``). ``Execute.release_at`` and, on the wall clock,
``FrameCaptured.timestamp`` are semantic seconds since the episode origin; on the token clock
``FrameCaptured.timestamp`` is the world's seconds (sandbox/actions.py ``capture``), not ticks.

Episode-trace names. The records the kernel writes under a TraceEvent name (InferenceStarted,
ActionProposed, ClockAdvanced, ActionResolved, ActionApplied, RunInvalidated) keep it, with the
payload the runner writes (evaluation/episode_records.py) instead of the kernel's data, which only
ClockAdvanced keeps: a generation's number is in no record but the KernelRecords naming it. Of the
echoes of the events it handled, FrameCaptured of a frame that was captured is written as
ObservationCaptured (a failed capture writes none, nor does the frame that froze the stream when
that is the run's failure), GenerationFinished of a result the shell settled as
OutputChunkReceived and InferenceCompleted (GenerationAbandoned is, for one the kernel gave up on
first), and an agent's AgentRecord under its own kind with the data its trace sink gave.
Every other record (TimerSet, EpisodeEnded, Termination and the other echoes, SnapshotFrozen's too)
is written as ``KernelRecord`` with payload ``{"kind", "data"}``, so the trace holds at most one
SnapshotFrozen, the one main() writes for a snapshot it took. Every record carries ``kernel_seq``,
``semantic`` and ``ops`` at the top level.

Failed state reads. The run is invalid once three state reads in a row have failed; a read that
answers starts the count over. The kernel keeps the count (kernel/turns.py READ_FAILURE_LIMIT),
over every read that decides whether the episode is over: ``check_end()``, which answers CheckEnd,
and on the token clock the reads a key tap makes inside ``execute()``, which the shell passes on as
``ActionExecuted.end_checks``. The environment invalidates the run for a failed read on neither
path; the legacy loop's environment (tests/differential/legacy_environment.py) invalidates the run
at the first failed read.
"""

from __future__ import annotations

import heapq
import json
import threading
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from realtime_framework.clock import Clocks
from realtime_framework.entity.actions import Action, ActionType, Observation
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.entity.model import PendingDecision
from realtime_framework.entity.records import TraceEvent, _jsonable
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.clock import Settlement, token_record
from realtime_framework.evaluation.episode_end import (
    EpisodeEnd,
    EpisodeResult,
    TraceWriter,
    escaped_termination,
    invalid_line,
    invalid_record,
    invalidated,
    only_logged,
    record_run_exception,
    with_unheard,
)
from realtime_framework.evaluation.kernel.core import Kernel
from realtime_framework.evaluation.kernel.events import (
    AbortGeneration,
    AcceptResult,
    ActionExecuted,
    AdvanceClock,
    AgentRecord,
    Capture,
    CheckEnd,
    ClockStepped,
    Commit,
    DiscardResult,
    EndChecked,
    Entry,
    Execute,
    Finish,
    FrameCaptured,
    FrameRendered,
    FreezeSnapshot,
    GenerationFinished,
    GenerationResult,
    Invalidated,
    KeysReleased,
    Outcome,
    OutcomeCommitted,
    Record,
    ReleaseInputs,
    ReleaseKeys,
    RenderFrame,
    SetTimer,
    SnapshotFrozen,
    StartGeneration,
    Termination,
)
from realtime_framework.evaluation.kernel.turns import TurnConfig, TurnPolicy
from realtime_framework.sandbox.actions import ENVIRONMENT_BUDGET
from realtime_framework.sandbox.budget import operation_budget


class ShellEnvironment(Protocol):
    """The environment adapter as the shell runs the kernel's commands on it, all on the shell's
    thread. An operation that fails raises, except ``execute()``: it returns ``"failed"`` and leaves
    the error in ``_last_exec_error``, whether the guest refused the action in
    ``_last_rejected_by_guest`` and the execution record in ``_last_execution_result``, which the
    shell reads as the runner does (legacy_loop.py ``_land_action``, ``_apply``). The environment
    may also latch the run invalid itself (``_invalid_run``) with a more precise reason, such as
    ``clock_advance_failed`` from a tap; for an operation that returns, the shell posts that as
    Invalidated before the command's answer, after it when the answer is an action's that
    carries that same failure (``_execute``). One that raises is answered with its error, and the
    run is invalid for the kernel's reason for that failure (kernel/turns.py ``_step_failed``,
    ``_rendered``)."""

    t0: float  # wall clock: the episode origin, which semantic readings count from
    d: float  # seconds per environment tick
    task: Any  # the task runtime; the shell reads its evaluation_snapshot() at the freeze
    # Token clock: the key identities, as the environment names them, that the last execute()
    # holds with a hold_key; () otherwise. The shell passes them as ActionExecuted.held_keys.
    last_held_keys: tuple[str, ...]
    # The state reads of the end checks the last execute() made inside the action, in order, each
    # (ended, reason, error): what check_end() answered with no error, or (False, None, the error)
    # for one that failed; () when it made none. Only a token-clock key tap makes them, after each
    # step of its advance; one that sees the end stops that advance, a failed one does not. The
    # shell passes them as ActionExecuted.end_checks.
    last_end_checks: tuple[tuple[bool, str | None, str | None], ...]
    # What the last execute() left for the shell: its execution record and coordinate trace,
    # which the shell clears before each, and when it returned "failed" its error and whether
    # the guest refused the action.
    _last_execution_result: Any
    _last_coordinate_trace: Any
    _last_exec_error: str | None
    _last_rejected_by_guest: bool
    frame_ages: list[float]  # each reported decision's frame age, which the shell appends
    # Called with the invalid-run record ({"category", "reason", "detail"}) when the wall clock's
    # key-release timer, on its own thread, invalidates the run. Once the snapshot is frozen, an
    # invalid run the environment latches is only logged, and the run's record is the
    # termination's (decision 30).
    on_invalid: Callable[[dict], None] | None

    def capture(self) -> Observation: ...

    def execute(self, action: Action, *, release_at: float | None = None) -> str:
        """Run one action and return its status as today. Never a wait: the kernel lets a wait's
        time pass itself. On the wall clock a hold_key's keys are released at ``release_at``,
        semantic seconds since the origin, unless a key is already held until later; a press
        keeps its keys down at least min(duration, 1/60 s) after its keydown returned, past
        ``release_at`` when the press came that late."""

    def check_end(self) -> tuple[bool, str | None]:
        """Ask the task whether the episode is over: ``(ended, reason)``. Only the task is asked:
        no deadline is compared and ``_invalid_run`` is neither read nor written. A failed read
        raises, and the kernel counts it.

        One read is bounded by the environment budget. On the wall clock its own item bounds it,
        10 s, instead of the 120 s ``state_read_s``, so a wedged environment ends the episode in
        about 30 s; on the token clock ``state_read_s`` bounds it as today."""

    def step_clock(self, step: float) -> None:
        """Token clock: advance ``step`` ticks, never past the episode horizon. Leases are not
        cut, no key is released and the end is not checked: the kernel does each."""

    def release_keys(self, keys: tuple[str, ...]) -> None:
        """Token clock: release these held keys, by identity, whose lease ended."""

    def render_current_frame(self) -> Any:
        """Token clock: commit the action's pixels without advancing time; None when the task
        committed no frame."""

    def release_inputs(self) -> None:
        """Release every held key and mouse button, after the snapshot."""

    def record_lifecycle(self, record: dict) -> None:
        """Write one reported decision's action-lifecycle record."""

    def tick_index(self) -> float: ...

    def set_episode_horizon(self, max_tick: float) -> None: ...


class KernelAgent(Protocol):
    """The agent as the shell runs the kernel's generation commands on it. ``generate`` runs on a
    worker thread; the rest run on the shell's thread, ``abort`` while the generation it names may
    still be running."""

    def generate(
        self, generation: int, decision: int, observation: Observation, budget: DecisionBudget
    ) -> PendingDecision:
        """StartGeneration: one generation on ``observation``. It writes no history: the result is
        held until the kernel accepts or discards it."""

    def accept(self, generation: int) -> None:
        """AcceptResult: write the held result into the history, the action record and the
        usage."""

    def discard(self, generation: int) -> None:
        """DiscardResult: drop the held result of a stale generation without writing history."""

    def commit(self, outcome: Outcome) -> None:
        """Commit: report the outcome of decision ``outcome.decision`` as its feedback."""

    def abort(self, generation: int) -> None:
        """AbortGeneration: interrupt the generation. The shell aborts every generation the
        kernel gives up on, one whose worker has returned too, before it discards that one's
        result."""


# ---- the shell --------------------------------------------------------------------------------


def run_episode(
    agent: KernelAgent,
    env: ShellEnvironment,
    config: TurnConfig,
    *,
    clocks: Clocks,
    log: Callable[[str], object] = print,
    trace=None,
    agent_trace=None,
    inputs_log: Path | None = None,
) -> EpisodeResult:
    """Run one episode on the kernel; return how it came out (episode_end.py), with its timeline,
    ``(k, t_k, d_ticks, tau_k, action)`` for each decision it reported. Build ``config`` with
    ``timing=RUNNER_TIMING``: TurnConfig's default timing reads no STREAM_* variable.

    ``agent_trace``, the agent's AgentTraceSink on ``trace``, hands the agent's records to the
    kernel during the episode, and the shell writes them in the kernel's order. Once the episode
    is finished the sink writes them itself again, after the shell has written those it handed
    over by then.

    ``inputs_log`` records what the kernel was given, one JSON line for each call that handled
    something: its readings and the events the step took from the inbox. A new kernel fed the
    same reproduces the episode's log (phase2.md, 怎么验证). A step the kernel raised in is
    recorded too, marked ``raised``; the log of an episode the kernel finished ends with an
    ``end`` line, so one without it is incomplete: the loop raised, a write failed or the process
    was stopped."""
    return _Shell(agent, env, config, clocks, log, trace, agent_trace, inputs_log).run()


@dataclass
class _Generation:
    """A generation the shell started, and its result once the worker hands it back."""

    generation: int
    decision: int
    observation: Observation
    remaining: int | None  # the finite-B allowance left before the horizon
    native_output_limit: int | None
    budget: DecisionBudget
    usage_before: dict | None
    started_at: float  # operational time when it started
    started_semantic: float  # the readings it started at
    started_ops: float | None
    done: threading.Event  # set once the worker has handed its result back
    # When its own bound falls due: the wall clock's per-move cap, in semantic seconds, or the
    # token clock's watchdog, in operational seconds; and the wall clock's deadline. None
    # without one (StartGeneration).
    bound_at: float | None
    deadline: float | None
    returned_at: float | None = None  # the reading on the bound's clock when it came back
    result: dict | None = None  # as the worker handed it back
    # The OutputChunkReceived and InferenceCompleted payloads of the result once it is settled, on
    # the shell's thread before the kernel gets it; None for a result the kernel had given up on.
    inference: tuple[list[dict], dict] | None = None
    settlement: Settlement | None = None  # of the result settled, its own or one synthesized
    invalid: Invalidated | None = None  # why the settled result is unusable, if it is
    echo: Entry | None = None  # the kernel's record of the result, which those are written under
    given_up: bool = False  # the kernel gave it up, and its records are written (_given_up)


@dataclass
class _Decision:
    """What the shell keeps of the decision in hand for its records."""

    index: int
    observation: Observation | None = None
    # Its settled result: the one the kernel accepted, or the one of the move it gave up on at the
    # cap: the call given up on, or the result settled past the cap.
    result: dict | None = None
    # When its action came to land, Unix seconds: the landing's reading on the wall clock,
    # operational time on the token clock.
    landed_at: float | None = None
    coordinate_trace: Any = None
    resolved: dict | None = None  # its ActionResolved payload


class _Shell:
    def __init__(
        self, agent, env, config: TurnConfig, clocks: Clocks, log, trace, agent_trace, inputs_log
    ):
        self.agent, self.env, self.config, self.clocks = agent, env, config, clocks
        self.log, self.agent_trace = log, agent_trace
        self.writer = TraceWriter(trace, self._trace_lost)
        self._inputs_log = inputs_log
        self.token = config.clock_mode == "token"
        self.policy = TurnPolicy(config)
        self.kernel = Kernel(self.policy)
        self.timeline: list[tuple] = []
        self._origin = clocks.operational.monotonic()
        self._now: tuple[float, float | None] = (0.0, None)  # the readings of the last step
        self._wake = clocks.operational.event()
        # A post and the step that takes it from the inbox, in one order.
        self._inbox = threading.Lock()
        self._posted: list = []  # posted since the last step
        # The agent's records not given to the kernel yet (_hand_over), and the one given to it
        # until it is written.
        self._held_records: deque[AgentRecord] = deque()
        self._given: AgentRecord | None = None
        self._inputs = None  # the open inputs log
        self._timers: list[float] = []  # heap of when the timers the kernel set come due
        self._returned: deque[_Generation] = deque()  # results the workers handed back
        self._generations: dict[int, _Generation] = {}
        self._abandoned: list[_Generation] = []
        self._settlements: list[Settlement] = []  # one for each result settled, in order
        # The generations whose settlement the shell synthesized, given up on before a result of
        # their own was settled.
        self._synthesized: list[_Generation] = []
        self._decision: _Decision | None = None
        self._captures = 0
        self._started: Entry | None = None  # the InferenceStarted record StartGeneration writes
        self._frozen_at: float | None = None  # the semantic reading the snapshot was frozen at
        self._accepting: set[int] = set()  # the generations the batch being run accepts
        self._aborting: dict[int, str] = {}  # and those it aborts, with why
        self._batch_invalid = False  # the shell posted an invalid run during the batch being run
        self._unheard: Invalidated | None = None  # one found once the kernel had finished
        self._snapshot_read = False
        self.end = EpisodeEnd()
        self._drained = False
        self._finished = False

    def run(self) -> EpisodeResult:
        env = self.env
        if self.token:
            # Under the kernel's own horizon, a safety net for every advance.
            env.set_episode_horizon(self.config.max_ticks)
        env.on_invalid = self._timer_latched
        # A replay aid, opened and closed apart from the episode, which goes on without it.
        inputs = None
        if self._inputs_log is not None:
            try:
                inputs = self._inputs = Path(self._inputs_log).open("w", encoding="utf-8")  # noqa: SIM115 - closed below, whatever the episode does
            except Exception:  # noqa: BLE001 - see above
                self.log("[default] kernel inputs log write failed:\n" + traceback.format_exc())
        try:
            try:
                self._episode()
            except Exception:  # noqa: BLE001 - the run is invalid and the episode still finalizes
                self._wrap_up(traceback.format_exc())
            else:
                self._record_inputs("end", [])
        finally:
            if inputs is not None:
                try:
                    inputs.close()
                except Exception:  # noqa: BLE001 - see above
                    self.log("[default] kernel inputs log write failed:\n" + traceback.format_exc())
        return self.end.result(env, self.timeline, self._settlements, self._drain_abandoned)

    def _episode(self) -> None:
        if self.agent_trace is None:
            return self._loop()
        self.agent_trace.hand_over(self._hand_over)
        try:
            self._loop()
        finally:
            self.agent_trace.take_back(self._write_handed if self._finished else self._write_held)

    def _hand_over(self, record: AgentRecord) -> None:
        """An agent record from the sink's writer, given to the kernel in its turn (_give)."""
        with self._inbox:
            self._held_records.append(record)
        self._wake.set()

    def _write_held(self) -> None:
        """The loop raised before the end and the shell wraps the episode up without the kernel
        (_wrap_up): the agent records the kernel had not taken are written unnumbered, as the sink
        writes them once it has them back."""
        given = [] if self._given is None else [self._given]
        for record in given + list(self._held_records):
            writer = self.writer
            if writer.trace is None or writer.failed:
                return
            try:
                writer.trace.emit(record.kind, record.data)
            except Exception:  # noqa: BLE001 - the run is invalid already
                writer.failed = True

    def _write_handed(self) -> None:
        """Once the episode is finished: hand the kernel the agent records it has not taken, step
        it through what is still due, the records among it, and write them. The kernel handles
        nothing after the end, so any other event's echo is only a KernelRecord. Should the
        kernel raise meanwhile, the episode's verdict stands: that is logged, and the agent
        records not written yet are written unnumbered."""
        unwritten = ([] if self._given is None else [self._given]) + list(self._held_records)
        while self._held_records:
            self.post(self._held_records.popleft())
        now = self._now
        try:
            now = self._read()
            while True:
                commands = self._step(now)
                if commands is None:
                    return
                for command in commands:
                    entry = command.entry
                    if isinstance(entry.data, AgentRecord):
                        unwritten = [record for record in unwritten if record is not entry.data]
                        self.writer.emit(entry.kind, entry.data.data, entry)
                    else:
                        self.writer.emit(
                            "KernelRecord", {"kind": entry.kind, "data": entry.data}, entry
                        )
        except Exception:  # noqa: BLE001 - the episode finished; only its last records are left
            self.log(
                "[default] numbering the agent's last records failed:\n" + traceback.format_exc()
            )
            for record in unwritten:
                self.writer.emit(record.kind, record.data, Entry(None, *now, record.kind, record))

    def _loop(self) -> None:
        self._now = self._read()
        self._record_inputs("start", [])
        self._run_all(self.kernel.start(*self._now))
        while not self._finished:
            # Cleared before the step takes the inbox: a post the step misses sets it again.
            self._wake.clear()
            # The results back before the reading are settled first; one back later waits for
            # the next reading. On the wall clock the kernel gets them at the reading after the
            # settling, which takes time in the world too: a deadline that came meanwhile goes
            # first, and nothing lands after it (phase2.md, 时间和结束判定). The token clock's
            # world stands still meanwhile; a new reading would only let the operational
            # watchdog overtake a result back before it.
            returned = len(self._returned)
            now = self._read()
            if returned:
                self._settle_returned(returned)
                if not self.token:
                    now = self._read()
            self._give()
            commands = self._step(now)
            if commands is not None:
                self._run_all(commands)
            else:
                self._sleep()

    def _give(self) -> None:
        """Give the kernel the next agent record once the last is written: one a round, after
        the results settled in it. A backlog of records, however long they take to write, holds
        back no result and no timer by more than one, and a kernel with something due at every
        step still numbers each a few steps after it came, not after the end."""
        if self._given is not None or not self._held_records:
            return
        with self._inbox:
            self._given = self._held_records.popleft()
        self.post(self._given)

    def post(self, event) -> None:
        """Hand the kernel an event, from any thread, and wake the shell."""
        with self._inbox:
            self.kernel.post(event)
            self._posted.append(event)
        self._wake.set()

    def _timer_latched(self, record: dict) -> None:
        """The wall clock's key-release timer latched an invalid run, on its own thread. Once the
        snapshot is frozen the kernel is not told: the failure is only logged (_terminate)."""
        if self._frozen_at is None:
            self.post(invalidated(record))

    def _record_inputs(self, call: str, events: list, *, raised: bool = False) -> None:
        if self._inputs is None:
            return
        semantic, ops = self._now
        try:
            line = {
                "call": call,
                "semantic": semantic,
                "ops": ops,
                "events": [{"event": type(event).__name__, **_jsonable(event)} for event in events],
            }
            if raised:
                line["raised"] = True
            self._inputs.write(json.dumps(line) + "\n")
            self._inputs.flush()
        except Exception:  # noqa: BLE001 - a replay aid; the episode goes on without it
            self.log("[default] kernel inputs log write failed:\n" + traceback.format_exc())
            self._inputs = None

    # ---- clocks -------------------------------------------------------------------------------

    def _read(self) -> tuple[float, float | None]:
        if self.token:
            return self.env.tick_index(), self.clocks.operational.monotonic() - self._origin
        return self.clocks.semantic.time() - self.env.t0, None

    def _step(self, now: tuple[float, float | None]):
        with self._inbox:
            self._now = now
            taken, self._posted = self._posted, []
            try:
                commands = self.kernel.step(*now)
            except BaseException:
                self._record_inputs("step", taken, raised=True)
                raise
        if commands is not None:  # a step that took an event always handles something
            self._record_inputs("step", taken)
        return commands

    def _sleep(self) -> None:
        """Wait for a post or for the next timer. A timer the kernel cancelled stays in the heap
        and wakes the shell for nothing."""
        semantic, ops = self._now
        # The wall clock's two schedules are one; the token clock's timers are all operational
        # (kernel/turns.py sets only its watchdog there).
        now = ops if self.token else semantic
        while self._timers and self._timers[0] <= now:
            heapq.heappop(self._timers)
        timeout = self._timers[0] - now if self._timers else None
        self.clocks.operational.wait(self._wake, timeout)

    # ---- commands -----------------------------------------------------------------------------

    def _run_all(self, commands: list) -> None:
        self._accepting = {c.generation for c in commands if isinstance(c, AcceptResult)}
        # An abort comes right after its generation's GenerationAbandoned record (kernel/core.py
        # abandon_generation), whose handling records why the generation was given up (_given_up).
        self._aborting = {
            c.generation: c.reason for c in commands if isinstance(c, AbortGeneration)
        }
        self._batch_invalid = False
        for command in commands:
            self._run(command)

    def _run(self, command) -> None:
        if self._batch_invalid and isinstance(command, Execute | AdvanceClock | RenderFrame):
            # The kernel takes the invalid run the shell posted only at its next step: what is
            # left of the batch acts on the world no more, as the legacy loop executed nothing
            # once _invalid_run was latched (legacy_loop.py _land_action).
            return self.post(self._not_run(command))
        try:
            self._handle(command)
        except Exception:  # noqa: BLE001 - the run is invalid and the episode still finalizes
            failure = traceback.format_exc()
            self._raised(failure)
            answer = self._unanswered(command, failure)
            if answer is not None:
                self.post(answer)

    def _handle(self, command) -> None:
        match command:
            case Record(entry=entry):
                self._record(entry)
            case SetTimer(at=at):
                heapq.heappush(self._timers, at)
            case CheckEnd(request=request):
                self._check_end(request)
            case Capture():
                self._capture()
            case StartGeneration():
                self._start_generation(command)
            case AbortGeneration(generation=generation):
                self._abort(generation)
            # Its frame and result go; the records of a settled result follow the agent's accept.
            case AcceptResult(generation=generation):
                held = self._generations.pop(generation)
                self._decision.result = held.result
                try:
                    self.agent.accept(generation)
                finally:
                    self._write_inference(held)
            case DiscardResult(generation=generation):
                del self._generations[generation]
                self.agent.discard(generation)
            case Execute():
                self._execute(command)
            # A failed step is answered with its error, and the kernel gives it the environment's
            # reason (kernel/turns.py _step_failed, _rendered).
            case AdvanceClock(step=step, escapes=escapes):
                self.post(ClockStepped(self._failure(escapes, self.env.step_clock, step)))
            case ReleaseKeys(keys=keys, escapes=escapes):
                self.post(KeysReleased(keys, self._failure(escapes, self.env.release_keys, keys)))
            case RenderFrame():
                try:
                    rendered = self._env(self.env.render_current_frame)
                except Exception:  # noqa: BLE001 - the kernel invalidates the run for it
                    failure = traceback.format_exc()
                    self._escaped(failure)
                    self.post(FrameRendered(error=failure))
                else:
                    self.post(FrameRendered(rendered is not None))
            case Commit(outcome=outcome):
                self._commit(outcome)
            case FreezeSnapshot():
                self._freeze()
            case ReleaseInputs():
                self._release_inputs()
            case Finish():
                self._finish(command)

    def _raised(self, failure: str) -> None:
        """An exception the legacy loop let escape it: the run is invalid and the episode still
        finalizes."""
        self._escaped(failure)
        self._invalidate(Invalidated("runner", "run_exception", failure))

    def _invalidate(self, invalid: Invalidated) -> None:
        """Post an invalid run the shell found on its own thread; the rest of the batch it came
        in runs nothing in the world (_run). A kernel that finished in the step whose batch this
        is takes no more events: the first such failure goes with its Finish (_finish)."""
        self._batch_invalid = True
        if self.kernel.termination is None:
            self.post(invalid)
        elif self._unheard is None:
            self._unheard = invalid

    def _escaped(self, failure: str) -> None:
        """An exception that escaped the legacy loop: it is logged, and the first traceback is
        the summary's error (tests/differential/legacy_loop.py _run_stream_legacy, main)."""
        self.log(f"[default] decision loop raised:\n{failure}")
        if self.end.loop_error is None:
            self.end.loop_error = failure

    def _unanswered(self, command, failure: str):
        """The answer to a command whose run raised, so that the kernel, which waits for it, can
        settle the decision; the run_exception before it decides the reason."""
        semantic, ops = self._now
        match command:
            case Capture():
                return FrameCaptured(error=failure)
            case Execute(decision=decision):
                return ActionExecuted(decision, error=failure)
            case Commit(outcome=outcome):
                return OutcomeCommitted(outcome.decision, failure)
            case FreezeSnapshot():
                return SnapshotFrozen(semantic, semantic if ops is None else ops, failure)
        return None

    @staticmethod
    def _not_run(command):
        """The answer to a command left unrun for an invalid run the kernel has not taken yet. The
        kernel takes that run first; an action left unrun, or one whose move to the action time
        was, settles as invalid_run (kernel/turns.py _executed, _step_failed)."""
        error = "not run: the run was invalidated earlier in this batch of commands"
        match command:
            case Execute(decision=decision):
                return ActionExecuted(decision, error=error)
            case AdvanceClock():
                return ClockStepped(error)
            case RenderFrame():
                return FrameRendered(error=error)

    def _env(self, operation, *args, **kwargs):
        """Run one synchronous environment operation. An invalid run the environment latched
        itself in an operation that returned goes to the kernel with its own reason before the
        command's answer (an action's, see _execute), unless the snapshot is frozen: then it is
        only logged (_terminate). One that raised raises here, with no post: its caller answers
        with the error."""
        latched = getattr(self.env, "_invalid_run", None)
        result = operation(*args, **kwargs)
        invalid = getattr(self.env, "_invalid_run", None)
        if invalid is not None and invalid is not latched and self._frozen_at is None:
            self._invalidate(invalidated(invalid))
        return result

    def _failure(self, escapes: bool, operation, *args) -> str | None:
        """Run one synchronous environment operation; its error, None when it returned. A
        failure the legacy loop let escape, as the command says, is logged as one
        (events.py AdvanceClock)."""
        try:
            self._env(operation, *args)
        except Exception:  # noqa: BLE001 - the command's answer carries it
            failure = traceback.format_exc()
            if escapes:
                self._escaped(failure)
            return failure
        return None

    def _check_end(self, request: int) -> None:
        try:
            ended, reason = self._env(self.env.check_end)
        except Exception:  # noqa: BLE001 - the kernel counts the failed read
            self.post(EndChecked(request, error=traceback.format_exc()))
        else:
            self.post(EndChecked(request, ended, reason))

    def _capture(self) -> None:
        # A decision starts with its frame, and only the last capture ends none.
        self._decision = _Decision(self._captures)
        self._captures += 1
        try:
            observation = self._env(self.env.capture)
        except Exception:  # noqa: BLE001 - the kernel invalidates the run for it
            return self.post(FrameCaptured(error=traceback.format_exc()))
        self._decision.observation = observation
        timestamp = observation.timestamp
        self.post(
            FrameCaptured(
                observation.metadata.get("observation_id"),
                observation.tick_number,
                timestamp if self.token else timestamp - self.env.t0,
                episode_records.observation_fingerprint(observation),
            )
        )

    # ---- generations --------------------------------------------------------------------------

    def _start_generation(self, command: StartGeneration) -> None:
        decision, remaining = command.decision, command.remaining_clock_tokens
        observation = self._decision.observation
        native_output_limit, budget = episode_records.decision_budget(
            self.agent, remaining, self.config.clock_mode
        )
        usage_before = episode_records.usage_snapshot(self.agent)
        entry, self._started = self._started, None
        self.writer.emit(
            TraceEvent.INFERENCE_STARTED,
            episode_records.inference_started(
                observation, decision, usage_before, native_output_limit, budget
            ),
            entry,
        )
        clock = self.clocks.operational
        held = _Generation(
            command.generation,
            decision,
            observation,
            remaining,
            native_output_limit,
            budget,
            usage_before,
            clock.time(),
            *self._now,
            clock.event(),
            command.bound_at,
            command.deadline,
        )
        self._generations[held.generation] = held
        clock.start_thread(lambda: self._generate(held), name=f"generation-{held.generation}")

    def _generate(self, held: _Generation) -> None:
        """The worker: one generation, handed back to the shell's thread as it came, whatever
        happens on the way: the kernel waits for it."""
        try:
            pending = self.agent.generate(
                held.generation, held.decision, held.observation, held.budget
            )
            held.result = pending.result
        # Also a cancellation that escaped the agent's retries (entity/model.py RetryCancelled).
        except BaseException:  # noqa: BLE001 - a failed generation invalidates the run
            failure = traceback.format_exc()
            latency = round(self.clocks.operational.time() - held.started_at, 2)
            held.result = episode_records.generation_error(latency, failure)
            self.log(f"[default] k={held.decision} generation failed, recovering:\n{failure}")
        finally:
            # The reading on the bound's clock, taken from the clock itself: the environment is
            # the shell thread's (ShellEnvironment).
            held.returned_at = (
                self.clocks.operational.monotonic() - self._origin
                if self.token
                else self.clocks.semantic.time() - self.env.t0
            )
            self._returned.append(held)
            held.done.set()
            self._wake.set()

    def _settle_returned(self, count: int) -> None:
        """Settle the first ``count`` results the workers handed back, on this thread, and hand
        them to the kernel. One back after a bound on its generation fell due waits for the
        kernel to give the generation up, as the runner's bounded wait did (legacy_loop.py
        _bounded_call). The late result of a generation the kernel gave up on (_given_up) goes
        back as it came, to be discarded: the runner never handled one either. One back before a
        bound the kernel then handled first, busy as the shell was, is settled all the same: the
        kernel discards it, and its records are its own."""
        for _ in range(count):
            held = self._returned.popleft()
            if held.given_up:
                self.post(GenerationFinished(held.generation, GenerationResult()))
                continue
            if self._late(held):
                self._returned.append(held)
                continue
            try:
                result, held.inference = self._settle(held, held.result)
            except Exception:  # noqa: BLE001 - as an exception that escaped the legacy loop
                self._raised(traceback.format_exc())
                # Recorded as far as settling it got: the kernel gives it up for the
                # run_exception, and nothing is recorded for it again.
                del self._generations[held.generation]
            else:
                held.invalid = result.invalid
                self.post(GenerationFinished(held.generation, result))

    def _late(self, held: _Generation) -> bool:
        """Whether a bound on the generation, its own or the wall deadline, fell due before its
        result came back; the kernel handles one due at the same time first too."""
        bounds = (held.bound_at, held.deadline)
        return any(at is not None and at <= held.returned_at for at in bounds)

    def _settle(
        self, held: _Generation, r: dict, outcome: str = "reported"
    ) -> tuple[GenerationResult, tuple]:
        """The result's annotations, usage, token accounting, cap audit and trajectory row, in
        the runner's order (legacy_loop.py ``_decide``); what the kernel needs of it, and its
        OutputChunkReceived and InferenceCompleted payloads as they were before the trajectory
        row. Its settlement goes into the episode's, with ``outcome`` (clock.py Settlement): the
        decision of a result the kernel takes is reported unless the shell marks it otherwise
        (_given_up, _record_unfinished)."""
        budget, clock_mode, agent = held.budget, self.config.clock_mode, self.agent
        episode_records.annotate_result(
            r,
            decision=held.decision,
            observation=held.observation,
            native_output_limit=held.native_output_limit,
            budget=budget,
            tokens_per_frame=self.config.tokens_per_frame,
        )
        # The clock's count is the result's own: an agent that holds its result adds its usage
        # only as it accepts it (_write_inference), so what the usage grew by until now is no count.
        # A result that names none has none: the agents create_agent makes name theirs
        # (harnesses/registry.py).
        held.settlement = episode_records.account_tokens(r, held.remaining)
        held.settlement.outcome = outcome
        self._settlements.append(held.settlement)
        usage_after = episode_records.usage_snapshot(agent)
        if held.usage_before is not None and usage_after is not None:
            r["frame_usage"] = episode_records.frame_usage(held.usage_before, usage_after)
        observation_id = r["observation_id"]
        inference = (
            episode_records.output_chunks(r, held.decision, observation_id),
            episode_records.inference_completed(r, held.decision, observation_id),
        )
        episode_records.audit_output_caps(agent, r, budget)
        failure = episode_records.finalize_decision_audit(agent, r)
        if failure is not None:
            self.log(f"[default] trajectory audit write failed:\n{failure}")
        # The kernel clamps the count to the allowance itself (kernel/turns.py _accept).
        unclamped = r["unclamped_clock_output_tokens"]
        result = GenerationResult(
            action=r.get("action"),
            output_tokens=r["clock_output_tokens"] if unclamped is None else unclamped,
            latency=r["latency"],
            protocol_failure=(r.get("action_protocol") or {}).get("protocol_failure"),
            invalid=episode_records.result_invalid(r, agent, clock_mode),
        )
        return result, inference

    def _write_inference(self, held: _Generation) -> None:
        """The OutputChunkReceived and InferenceCompleted records of a settled result, under the
        kernel's record of it. An agent that holds its result adds its usage as it accepts it
        (harnesses/default/harness.py), so the frame's usage is what the usage grew by until now."""
        if held.echo is None:
            return
        chunks, completed = held.inference
        usage_after = episode_records.usage_snapshot(self.agent)
        if held.usage_before is not None and usage_after is not None:
            completed = {
                **completed,
                "frame_usage": episode_records.frame_usage(held.usage_before, usage_after),
            }
        self._emit_inference((chunks, completed), held.echo)

    def _abort(self, generation: int) -> None:
        """Interrupt a generation the kernel gave up on. One whose worker came back is aborted
        too, before its result is discarded: a session that holds its reply rebuilds from the
        accepted history, as the runner aborted every call it gave up on (legacy_loop.py
        _bounded_call)."""
        held = self._generations.get(generation)  # None when its StartGeneration raised
        if held is None:
            return
        if not held.done.is_set():
            self._abandoned.append(held)
        try:
            self.agent.abort(generation)
        except Exception:  # noqa: BLE001 - interrupting is best effort, as in the runner
            self.log(
                f"[default] abort of generation {generation} failed:\n{traceback.format_exc()}"
            )

    def _drain_abandoned(self) -> None:
        """Wait a bounded time for the workers of abandoned generations before main() closes the
        agent, after the snapshot is frozen; workers still inside a native call need the runner's
        hard exit. Then a generation whose settlement was synthesized takes the result its worker
        handed back, if it did, as what its call spent (clock.py Settlement.late). Runs once."""
        if self._drained:
            return
        self._drained = True
        pending = [held for held in self._abandoned if not held.done.is_set()]
        if pending:
            seconds, clock = self.config.timing.abandoned_drain_s, self.clocks.operational
            self.log(
                f"[default] draining {len(pending)} abandoned generate worker(s), <= {seconds:g}s"
            )
            limit = clock.time() + seconds
            for held in pending:
                clock.wait(held.done, max(0.0, limit - clock.time()))
            live = sum(1 for held in pending if not held.done.is_set())
            if live:
                self.log(
                    f"[default] {live} abandoned worker(s) STILL inside a native call after "
                    f"{seconds:g}s — teardown continues; the runner hard-exits so interpreter "
                    f"finalization cannot abort on them"
                )
        for held in self._synthesized:
            if held.done.is_set():
                held.settlement.late = token_record(held.result)

    # ---- the action ---------------------------------------------------------------------------

    def _execute(self, command: Execute) -> None:
        env, decision = self.env, self._decision
        env._last_execution_result = None
        env._last_coordinate_trace = None
        latched = getattr(env, "_invalid_run", None)
        status = env.execute(command.action, release_at=command.release_at)
        rejection = error = None
        if status == "failed":
            # A guest refusal is the model's action failing, not the environment's.
            if env._last_rejected_by_guest:
                rejection = env._last_exec_error or ""
            else:
                error = env._last_exec_error
        decision.coordinate_trace = env._last_coordinate_trace
        answer = ActionExecuted(
            command.decision,
            # As the environment returned it; a guest reply's "timestamp" is real time inside the
            # sandbox (sandbox/guest.py execute_action).
            result=env._last_execution_result,
            held_keys=env.last_held_keys,
            rejection=rejection,
            error=error,
            end_checks=env.last_end_checks,
        )
        invalid = getattr(env, "_invalid_run", None)
        if invalid is None or invalid is latched:
            self.post(answer)
        elif invalid["reason"] == "action_execution_failed":
            # The answer carries this failure and the kernel names it the same (kernel/turns.py
            # _executed): first, it settles the action where it ran, not after the snapshot read
            # the end asks for.
            self.post(answer)
            self._invalidate(invalidated(invalid))
        else:
            # Latched for a step inside the action, such as a key tap's failed advance: first, it
            # keeps its reason, which the kernel would read from the answer as
            # action_execution_failed.
            self._invalidate(invalidated(invalid))
            self.post(answer)

    def _commit(self, outcome: Outcome) -> None:
        env, decision, config = self.env, self._decision, self.config
        r = decision.result
        # The loop has no terminal actions: the record of a proposed DONE or FAIL says so, also
        # when the move went over the per-move cap and the kernel named no protocol failure
        # (kernel/turns.py _accept).
        proposed = None if r is None else r.get("action")
        if proposed is not None and proposed.action_type in (ActionType.DONE, ActionType.FAIL):
            r["action_protocol"] = episode_records.terminal_wait_protocol(
                r.get("action_protocol") or {}
            )
        if outcome.frame_age_s is not None:
            env.frame_ages.append(outcome.frame_age_s)
        try:
            self.agent.commit(outcome)
        except Exception:  # noqa: BLE001 - the kernel invalidates the run for it
            failure = traceback.format_exc()
            self.log(f"[default] commit({outcome.status}) failed:\n{failure}")
            self.post(OutcomeCommitted(outcome.decision, failure))
        else:
            self.post(OutcomeCommitted(outcome.decision))
        if r is None:
            # A move given up on at the cap whose StartGeneration raised has no result to record;
            # that raise invalidated the run.
            return
        resolved = decision.resolved
        try:
            self._env(
                env.record_lifecycle,
                episode_records.lifecycle_record(
                    r,
                    decision=outcome.decision,
                    clock_mode=config.clock_mode,
                    frame_age_s=outcome.frame_age_s,
                    action_clock_position=outcome.action_clock_position,
                    action_resolved_unix_s=resolved["action_resolved_unix_s"],
                    frame_output_tokens=outcome.action_output_tokens,
                    action_output_tokens=outcome.action_output_tokens if self.token else None,
                    token_delay_frames=outcome.token_delay_frames,
                    next_observation_tick=outcome.next_observation_tick,
                    post_action_rendered=outcome.post_action_rendered,
                    dropped_intermediate_frames=outcome.dropped_intermediate_frames,
                    status=outcome.status,
                    terminal_after_action=outcome.terminal_after_action,
                    action=outcome.action,
                    execution_result=resolved["execution_result"],
                    coordinate_trace=resolved["coordinate_trace"],
                    clock_advance=(
                        getattr(env.task, "_last_clock_advance", None) if self.token else None
                    ),
                    recorded_unix_s=self.clocks.operational.time(),
                ),
            )
        except Exception:  # noqa: BLE001 - any write failure invalidates the run
            failure = traceback.format_exc()
            self.log(f"[default] lifecycle audit write failed:\n{failure}")
            self._invalidate(Invalidated("harness", "action_lifecycle_write_failed", failure))
        self.writer.checkpoint()
        t_k, tau, action = r["observation_tick"], outcome.action_clock_position, outcome.action
        # The token delay is in input frames; on the wall clock the delay is the decision's
        # latency (kernel/turns.py _accept).
        d_ticks = (
            outcome.token_delay_frames * config.env_fps / config.input_fps
            if self.token
            else max(0.0, r.get("latency") or 0.0) * config.env_fps
        )
        self.timeline.append((outcome.decision, t_k, d_ticks, tau, action))
        act = None if action is None else (action.action_type.value, action.parameters)
        self.log(
            f"[default] k={outcome.decision} t_k={t_k} δ={d_ticks:g} τ={tau:g} act={act} -> "
            f"{outcome.status}"
            + (f" tokens={outcome.action_output_tokens}" if self.token else "")
            + (" TIMEOUT" if outcome.status == "skipped_timeout" else "")
        )

    # ---- the end ------------------------------------------------------------------------------

    def _freeze(self) -> None:
        error = self._read_snapshot()
        # Should the reading after it raise, a wrap-up does not read the snapshot again; only the
        # token clock's reading can raise, and its world stood still meanwhile.
        self._frozen_at = self._now[0]
        semantic, ops = self._read()  # when the read returned
        self._frozen_at = semantic
        self.post(SnapshotFrozen(semantic, semantic if ops is None else ops, error))

    def _read_snapshot(self) -> str | None:
        """Read the scorer's snapshot; the error of a read that failed."""
        # The presentation boundary precedes the scorer's snapshot.
        self.end.snapshot_begins(self.clocks.operational.time())
        # The Task base class has none (realtime_tasks/types.py).
        snapshotter = getattr(self.env.task, "evaluation_snapshot", None)
        error = None
        if snapshotter is not None:
            try:
                with operation_budget(
                    "snapshot read",
                    ENVIRONMENT_BUDGET.snapshot_read_s,
                    clock=self.clocks.operational,
                ):
                    snapshot = self._env(snapshotter)
            except Exception:  # noqa: BLE001 - the kernel invalidates the run for it
                error = traceback.format_exc()
                self.end.snapshot_read(None, error)
            else:
                self.end.snapshot_read(snapshot, None)
                self._snapshot_read = True
        return error

    def _release_inputs(self) -> None:
        try:
            self.env.release_inputs()
        except Exception:  # noqa: BLE001 - the snapshot is frozen; cleanup is best effort
            self.log("[cleanup] failed to release stateful inputs:\n" + traceback.format_exc())

    def _finish(self, finish: Finish) -> None:
        """The termination record main() and the report read, in world seconds to nine places
        (legacy_loop.py ``_termination``); the token clock's are ticks, and a failure the shell
        found in this last batch goes into it (episode_end.py with_unheard)."""
        termination = with_unheard(finish.termination, self._unheard, self.log)
        try:
            self._terminate(termination, finish.think_spent, finish.dropped_intermediate_frames)
        finally:
            self._finished = True  # nothing more will be asked (kernel/core.py)

    def _terminate(
        self, termination: Termination, think_spent: float, dropped_intermediate_frames: int
    ) -> None:
        end = self.end
        late = end.terminate(
            termination, self.env, self._snapshot_read, self.config.env_fps if self.token else 1.0
        )
        end.think_spent = think_spent
        end.dropped_intermediate_frames = dropped_intermediate_frames
        if late is not None:
            self.log(only_logged(late))

    def _wrap_up(self, failure: str) -> None:
        """The kernel, or the shell's reading of the clocks, raised (N11). The kernel's state no
        longer holds, so the shell wraps the episode up without it, as the runner wrapped up a loop
        that raised (tests/differential/legacy_loop.py _run_stream_legacy, _finalize): the
        call running is interrupted, the snapshot frozen unless it was, the inputs released
        and the termination record written. A failure on the way is only logged."""
        self._escaped(failure)
        # What the environment latched by now came first; a failure during the wrap-up after.
        latched = self.end.latched(self.env)
        try:
            for generation in list(self._generations):
                self._abort(generation)
            semantic, ops = self._now  # the reading the failure came at, or the last good one
            if self._frozen_at is None:
                self._read_snapshot()
                # The token clock's world stands still meanwhile; its reading may be what raised.
                self._frozen_at = semantic if self.token else self._read()[0]
            self._release_inputs()
            first = self.kernel.invalid is None and latched is None
            termination = escaped_termination(
                self.kernel,
                semantic,
                semantic if ops is None else ops,
                self._frozen_at,
                self.end.loop_error,  # this one's, unless a failure the kernel never took came first
                latched,
            )
            # No Finish came to carry the policy's counts: they are as far as they got.
            self._terminate(termination, **self.policy.totals)
            if first:
                record_run_exception(self.writer, termination.invalid, semantic, ops, self.log)
            self._record_unfinished(termination, semantic, ops)
        except Exception:  # noqa: BLE001 - the episode is over either way
            self.log("[default] wrapping up the episode failed:\n" + traceback.format_exc())
        self._finished = True

    def _record_unfinished(self, termination: Termination, semantic: float, ops) -> None:
        """The records of the generations the kernel neither accepted nor discarded, unnumbered,
        as for one an invalidation gave up on (_given_up, N10): a settled result keeps its own,
        out of the reported decisions' subtotal; any other is recorded as a call given up on for
        the invalid run, from its start to the episode's end. The agent's close() settles them,
        as it settles the generation an episode the kernel finished ended on."""
        for held in list(self._generations.values()):
            if held.given_up:
                continue
            data = {"generation": held.generation, "reason": "invalid"}
            entry = Entry(None, semantic, ops, "GenerationAbandoned", data)
            if held.inference is not None:
                self._emit_inference(held.inference, entry)
                held.settlement.outcome = "unreported"
                continue
            seconds = (
                termination.ops_ended_at - held.started_ops
                if self.token
                else termination.ended_at - held.started_semantic
            )
            r = episode_records.abandoned_result(round(max(0.0, seconds), 2), "invalid_run")
            self._emit_inference(self._settle(held, r, "abandoned")[1], entry)
            self._synthesized.append(held)

    # ---- the episode trace --------------------------------------------------------------------

    def _record(self, entry: Entry) -> None:
        """Write one kernel record (D6). An agent's record is the agent's, whatever its name. One
        under a TraceEvent name keeps it, with the payload the runner writes; a frame's echo is
        ObservationCaptured and the echo of a settled result OutputChunkReceived and
        InferenceCompleted; any other is a KernelRecord."""
        kind, data, decision = entry.kind, entry.data, self._decision
        if isinstance(data, AgentRecord):
            self._given = None
            self.writer.emit(kind, data.data, entry)
        elif kind == TraceEvent.INFERENCE_STARTED.value:
            self._started = entry  # StartGeneration, which follows it, writes it with its caps
        elif kind == "FrameCaptured" and data.error is None and not self._froze():
            self.writer.emit(
                TraceEvent.OBSERVATION_CAPTURED,
                episode_records.observation_captured(decision.observation, decision.index),
                entry,
            )
        elif kind == "GenerationFinished" and self._generations[data.generation].inference:
            held = self._generations[data.generation]
            held.echo = entry
            # The kernel takes a result in the step that echoes it (kernel/core.py _dispatch),
            # whose next command is AcceptResult: the records wait for the agent to have it. A
            # result the kernel discards adds no usage.
            if data.generation not in self._accepting:
                self._write_inference(held)
        elif kind == TraceEvent.ACTION_PROPOSED.value:
            self.writer.emit(
                TraceEvent.ACTION_PROPOSED,
                episode_records.action_proposed(
                    decision.result, data["decision"], data["observation_id"]
                ),
                entry,
            )
        elif kind == TraceEvent.ACTION_RESOLVED.value:
            # The record of an unusable result has no landing, clock position or tick
            # (kernel/turns.py _unusable).
            landed = decision.landed_at is not None
            decision.resolved = episode_records.action_resolved(
                decision=data["decision"],
                observation_id=data["observation_id"],
                action=data["action"],
                status=data["status"],
                terminal_after_action=data["terminal_after_action"],
                action_clock_position=data.get("action_clock_position"),
                action_resolved_unix_s=(
                    decision.landed_at if landed else self.clocks.operational.time()
                ),
                environment_tick=data.get("environment_tick"),
                execution_result=data.get("execution_result"),
                coordinate_trace=decision.coordinate_trace,
            )
            self.writer.emit(TraceEvent.ACTION_RESOLVED, decision.resolved, entry)
        elif kind == TraceEvent.ACTION_APPLIED.value:
            self.writer.emit(TraceEvent.ACTION_APPLIED, decision.resolved, entry)
        elif kind == TraceEvent.CLOCK_ADVANCED.value:
            if data["phase"] == "inference_to_action":
                # The action lands: the runner times its resolution here, whatever comes of it
                # (legacy_loop.py _apply). On the wall clock that is the reading the kernel landed
                # it at, which its frame age counts from too (kernel/turns.py _approached); on the
                # token clock it is when the clock got to the action time.
                decision.landed_at = (
                    self.clocks.operational.time() if self.token else self.env.t0 + entry.semantic
                )
            self.writer.emit(TraceEvent.CLOCK_ADVANCED, data, entry)
        elif kind == TraceEvent.RUN_INVALIDATED.value:
            invalid = Invalidated(**data)
            if invalid.reason == "observation_stream_frozen":
                self.end.stream_freeze_count += 1
                self.log(
                    "[default] observation stream FROZEN (same id/tick/hash x"
                    f"{self.config.timing.observation_freeze_n}); invalidating run"
                )
            self.log(invalid_line(invalid))
            self.writer.emit(TraceEvent.RUN_INVALIDATED, invalid_record(invalid), entry)
        else:
            self.writer.emit("KernelRecord", {"kind": kind, "data": data}, entry)
            if kind == "GenerationAbandoned":
                self._given_up(entry)
            elif kind == "EpisodeEnded" and data["cause"] == "think_budget":
                self.log(
                    f"[think] deliberation budget exhausted: {self.policy.spent:.1f}s >= "
                    f"{self.config.think_budget_s:g}s -> forfeit"
                )

    def _given_up(self, entry: Entry) -> None:
        """A generation the kernel gave up on. One whose result came back before and was settled
        keeps the call's own records, written now, ahead of the decision's, and what made that
        result unusable still invalidates the run, as the runner's check of every result did
        (legacy_loop.py _apply): the shell posts that reason, as the kernel discards the result
        without it. Otherwise one given up on at the per-move cap, the wall deadline or the
        provider watchdog timed out: it is logged and recorded as the runner records a call it
        gave up on (legacy_loop.py _run_step), with no usage, which leaves the episode's cost
        incomplete (decision 48). One the task's end or an invalidation gave up on is recorded
        so too, named for what ended the episode (N10), as a synthesized record the token
        totals and the model timing leave out (N34). Only the move over the cap is reported, as
        skipped (kernel/turns.py on_timer), its record naming the cap also when it keeps a result
        settled before the cap was handled; the others end the episode and their decision goes
        unreported (K-a): the settlement of a result settled before, or of one the totals count,
        says so, which keeps it out of the subtotal of reported decisions the report checks the
        lifecycle against (reporting.py report)."""
        reason, held = entry.data["reason"], self._generations.get(entry.data["generation"])
        if held is None:
            return  # its StartGeneration raised
        held.given_up = True
        bound = {
            "per_move_cap": "per_move_cap",
            "deadline": "wall_deadline",
            "generation_watchdog_timeout": "provider_watchdog",
        }.get(self._aborting[held.generation])
        if held.inference is not None:
            self._emit_inference(held.inference, entry)
            held.inference = None  # its echo writes nothing more
            if reason == "per_move_cap":
                held.result["harness_timeout_reason"] = "per_move_cap"
                self._decision.result = held.result
            else:
                held.settlement.outcome = "unreported"
            if bound is not None:
                self.log(
                    f"[default] k={held.decision} generation back in "
                    f"{held.result['latency']:g}s, settled past its {bound} — discarded"
                )
            if held.invalid is not None and self.kernel.invalid is None:
                self._invalidate(held.invalid)
            return
        if bound is None:
            # It ran until the episode ended (decision 32), however late the shell got to it. A
            # read's failure ends it when the read was asked (N44), which can be before a
            # generation started while that read's answer waited: that one ran no time.
            ended = {"environment": "task_end", "invalid": "invalid_run"}.get(reason, reason)
            ending = self.kernel.ending
            seconds = (
                ending.ops - held.started_ops if self.token else ending.at - held.started_semantic
            )
            r = episode_records.abandoned_result(round(max(0.0, seconds), 2), ended)
        elif bound == "provider_watchdog":
            seconds = held.bound_at - held.started_ops
            r = episode_records.watchdog_result(round(seconds, 2), seconds)
            timeout = self.config.timing.provider_watchdog_s
        else:
            # The bound, however late the shell got to it: the kernel charges the move that much.
            due = held.bound_at if bound == "per_move_cap" else held.deadline
            seconds = due - held.started_semantic
            r = episode_records.timed_out_result(round(seconds, 2), bound)
            capped = bound == "per_move_cap"
            timeout = max(0.1, self.config.per_move_cap_s) if capped else round(seconds, 2)
        if bound is not None:
            # The cap and the watchdog as the runner names them (legacy_loop.py _move_timeout); the
            # deadline from the call's start, where the legacy loop gave the time left plus a grace.
            self.log(
                f"[default] k={held.decision} generation TIMED OUT (>{timeout}s, {bound}) "
                "— discarded"
            )
        if reason == "per_move_cap":
            outcome = "reported"
        else:
            outcome = "abandoned" if bound is None else "unreported"
        result, inference = self._settle(held, r, outcome)
        self._synthesized.append(held)
        if reason == "per_move_cap":
            self._decision.result = r
        self._emit_inference(inference, entry)
        if result.invalid is not None and self.kernel.invalid is None:
            self._invalidate(result.invalid)  # its trajectory row could not be written

    def _emit_inference(self, inference: tuple[list[dict], dict], entry: Entry) -> None:
        chunks, completed = inference
        for chunk in chunks:
            self.writer.emit(TraceEvent.OUTPUT_CHUNK_RECEIVED, chunk, entry)
        self.writer.emit(TraceEvent.INFERENCE_COMPLETED, completed, entry)

    def _froze(self) -> bool:
        """Whether the frame whose echo this is froze the stream: the kernel judged it in the
        step that handled it (kernel/turns.py _captured), and the runner writes no frame its
        check stops (legacy_loop.py _decide)."""
        invalid = self.kernel.invalid
        return invalid is not None and invalid.reason == "observation_stream_frozen"

    def _trace_lost(self, invalid: Invalidated) -> None:
        """A write to the episode trace failed: the run is invalid, which the agent's trace sink
        is told once the kernel handles nothing more (_write_handed)."""
        if self._finished:
            self.agent_trace.fail(invalid.detail)
        else:
            self._invalidate(invalid)
