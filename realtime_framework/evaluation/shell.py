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
``FrameCaptured.timestamp`` are semantic seconds since the episode origin.

Episode-trace names. The records the kernel writes under a TraceEvent name (InferenceStarted,
ActionProposed, ClockAdvanced, ActionResolved, ActionApplied, RunInvalidated) keep it, their
payload completed with the fields the runner writes today. Of the echoes of the events it handled,
FrameCaptured is written as ObservationCaptured, GenerationFinished as OutputChunkReceived and
InferenceCompleted, and an agent's AgentRecord under its own kind with the data its trace sink gave.
Every other record (TimerSet, EpisodeEnded, Termination and the other echoes, SnapshotFrozen's too)
is written as ``KernelRecord`` with payload ``{"kind", "data"}``, so the one SnapshotFrozen in the
trace is the snapshot main() writes, where reporting ends the active interval. Every record carries
``kernel_seq``, ``semantic`` and ``ops`` at the top level.

Failed state reads. The run is invalid once three state reads in a row have failed; a read that
answers starts the count over. The kernel keeps the count (kernel/turns.py READ_FAILURE_LIMIT),
over every read that decides whether the episode is over: ``check_end()``, which answers CheckEnd,
and on the token clock the reads a key tap makes inside ``execute()``, which the shell passes on as
``ActionExecuted.end_checks``. The shell sets ``kernel_counts_reads``, and then the environment
invalidates the run for a failed read on neither path; under the legacy loop it stays unset, and
the first failed read invalidates the run as today.
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
from realtime_framework.entity.actions import Action, Observation
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.entity.model import PendingDecision
from realtime_framework.entity.records import TraceEvent, _jsonable
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.kernel.core import Kernel
from realtime_framework.evaluation.kernel.events import (
    AbortGeneration,
    AcceptResult,
    ActionExecuted,
    AdvanceClock,
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
    thread. An operation that fails raises, except ``execute()``: it returns ``"failed"`` and
    leaves the error in ``_last_exec_error``, whether the guest refused the action in
    ``_last_rejected_by_guest`` and the execution record in ``_last_execution_result``, which the
    shell reads as the runner does (runner.py:576-585, 626). The environment may also latch the run
    invalid itself (``_invalid_run``) with a more precise reason, such as ``clock_advance_failed``
    from a tap; the shell posts that as Invalidated before the command's answer."""

    t0: float  # wall clock: the episode origin, which semantic readings count from
    d: float  # seconds per environment tick
    task: Any  # the task runtime; the shell reads its evaluation_snapshot() at the freeze
    # Token clock: the key identities, as the environment names them, that the last execute()
    # holds with a hold_key; () otherwise. The shell passes them as ActionExecuted.held_keys.
    last_held_keys: tuple[str, ...]
    # The state reads of the end checks the last execute() made inside the action, in order: None
    # for one that answered, the error of one that failed; () when it made none. Only a token-clock
    # key tap makes them, after each step of its advance; one that sees the end stops the tap, a
    # failed one does not. The shell passes them as ActionExecuted.end_checks.
    last_end_checks: tuple[str | None, ...]
    # Set by the shell before the episode: the kernel counts failed state reads, so a failed read
    # inside execute() goes to last_end_checks instead of latching the run invalid. The legacy
    # loop leaves it False.
    kernel_counts_reads: bool
    # Called with the invalid-run record ({"category", "reason", "detail"}) when the wall clock's
    # key-release timer, on its own thread, invalidates the run.
    on_invalid: Callable[[dict], None] | None

    def capture(self) -> Observation: ...

    def execute(self, action: Action, *, release_at: float | None = None) -> str:
        """Run one action and return its status as today. Never a wait: the kernel lets a wait's
        time pass itself. On the wall clock a hold_key's keys are released at ``release_at``,
        semantic seconds since the origin, unless a key is already held until later."""

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
        held until the kernel accepts or discards it. StreamingAgent's default runs the legacy
        turn, which writes it at once (entity/harness.py)."""

    def accept(self, generation: int) -> None:
        """AcceptResult: write the held result into the history, the action record and the
        usage."""

    def discard(self, generation: int) -> None:
        """DiscardResult: drop the held result of a stale generation without writing history."""

    def commit(self, outcome: Outcome) -> None:
        """Commit: report the outcome of decision ``outcome.decision`` as its feedback."""

    def abort(self, generation: int | None = None) -> None:
        """AbortGeneration: interrupt the generation; None interrupts whichever is active."""


# ---- the shell --------------------------------------------------------------------------------


def run_episode(
    agent: KernelAgent,
    env: ShellEnvironment,
    config: TurnConfig,
    *,
    clocks: Clocks,
    log: Callable[[str], object] = print,
    trace=None,
    inputs_log: Path | None = None,
) -> list[tuple]:
    """Run one episode on the kernel; return its timeline, ``(k, t_k, d_ticks, tau_k, action)``
    for each decision it reported. Build ``config`` with ``timing=RUNNER_TIMING``: TurnConfig's
    default timing reads no STREAM_* variable.

    What main() reads after the loop is left on ``env`` as the legacy loop leaves it: the stop
    reason and the termination record in world seconds (``_stop_reason``, ``_termination``), the
    frozen snapshot (``_termination_snapshot*``, ``_active_end_unix_s``), ``_invalid_run``,
    ``_loop_error``, ``_think_spent``, ``_dropped_intermediate_frames``, ``_stream_frozen`` and
    the drain of abandoned workers (``_drain_abandoned``).

    ``inputs_log`` records what the kernel was given, one JSON line for each call that handled
    something: its readings and the events the step took from the inbox. A new kernel fed the
    same reproduces the episode's log (phase2.md, 怎么验证)."""
    return _Shell(agent, env, config, clocks, log, trace, inputs_log).run()


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
    done: threading.Event  # set once the worker has handed its result back
    result: dict | None = None  # settled on the shell's thread before the kernel gets it


@dataclass
class _Decision:
    """What the shell keeps of the decision in hand for its records."""

    index: int
    observation: Observation | None = None
    generation: _Generation | None = None
    accepted: dict | None = None  # the settled result the kernel accepted
    executed_at: float | None = None  # operational time right before its Execute
    coordinate_trace: Any = None
    resolved: dict | None = None  # its ActionResolved payload


def _invalidated(record: dict) -> Invalidated:
    """An invalid-run record the environment latched, as the kernel's event."""
    return Invalidated(record["category"], record["reason"], record.get("detail"))


def _invalid_record(invalid: Invalidated) -> dict:
    """An invalid run as the runner keeps it on ``env._invalid_run`` and writes it as
    RunInvalidated: the detail only when there is one (runner.py ``_invalidate``)."""
    record = {"category": invalid.category, "reason": invalid.reason}
    if invalid.detail:
        record["detail"] = invalid.detail
    return record


class _Shell:
    def __init__(self, agent, env, config: TurnConfig, clocks: Clocks, log, trace, inputs_log):
        self.agent, self.env, self.config, self.clocks = agent, env, config, clocks
        self.log, self.trace = log, trace
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
        self._inputs = None  # the open inputs log
        self._timers: list[float] = []  # heap of when the timers the kernel set come due
        self._returned: deque[_Generation] = deque()  # results the workers handed back
        self._generations: dict[int, _Generation] = {}
        self._abandoned: list[_Generation] = []
        self._decision: _Decision | None = None
        self._captures = 0
        self._started: Entry | None = None  # the InferenceStarted record StartGeneration writes
        self._snapshot_read = False
        self._trace_failed = False
        self._loop_error: str | None = None
        self._drained = False
        self._finished = False

    def run(self) -> list[tuple]:
        env = self.env
        if self.token:
            # Under the kernel's own horizon, a safety net for every advance.
            env.set_episode_horizon(self.config.max_ticks)
        self.agent._clock_mode = self.config.clock_mode
        env.kernel_counts_reads = True
        env.on_invalid = lambda record: self.post(_invalidated(record))
        env._drain_abandoned = self._drain_abandoned
        if self._inputs_log is None:
            self._loop()
        else:
            with Path(self._inputs_log).open("w", encoding="utf-8") as self._inputs:
                self._loop()
        return self.timeline

    def _loop(self) -> None:
        self._now = self._read()
        commands = self.kernel.start(*self._now)
        self._record_inputs("start", [])
        self._run_all(commands)
        while not self._finished:
            # Cleared before the step takes the inbox: a post the step misses sets it again.
            self._wake.clear()
            commands = self._step()
            if commands is None:
                self._sleep()
            else:
                self._run_all(commands)

    def post(self, event) -> None:
        """Hand the kernel an event, from any thread, and wake the shell."""
        with self._inbox:
            self.kernel.post(event)
            self._posted.append(event)
        self._wake.set()

    def _record_inputs(self, call: str, events: list) -> None:
        if self._inputs is None:
            return
        semantic, ops = self._now
        line = {
            "call": call,
            "semantic": semantic,
            "ops": ops,
            "events": [{"event": type(event).__name__, **_jsonable(event)} for event in events],
        }
        try:
            self._inputs.write(json.dumps(line) + "\n")
        except Exception:  # noqa: BLE001 - a replay aid; the episode goes on without it
            self.log("[default] kernel inputs log write failed:\n" + traceback.format_exc())
            self._inputs = None

    # ---- clocks -------------------------------------------------------------------------------

    def _read(self) -> tuple[float, float | None]:
        if self.token:
            return self.env.tick_index(), self.clocks.operational.monotonic() - self._origin
        return self.clocks.semantic.time() - self.env.t0, None

    def _step(self):
        self._settle_returned()
        with self._inbox:
            self._now = self._read()
            commands = self.kernel.step(*self._now)
            taken, self._posted = self._posted, []
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
            case AcceptResult(generation=generation):
                self.agent.accept(generation)
                self._decision.accepted = self._generations[generation].result
            case DiscardResult(generation=generation):
                self.agent.discard(generation)
            case Execute():
                self._execute(command)
            case AdvanceClock(step=step):
                self._env(self.env.step_clock, step)
                self.post(ClockStepped())
            case ReleaseKeys(keys=keys):
                self._env(self.env.release_keys, keys)
                self.post(KeysReleased(keys))
            case RenderFrame():
                self.post(FrameRendered(self._env(self.env.render_current_frame) is not None))
            case Commit(outcome=outcome):
                self._commit(outcome)
            case FreezeSnapshot():
                self._freeze()
            case ReleaseInputs():
                try:
                    self.env.release_inputs()
                except Exception:  # noqa: BLE001 - the snapshot is frozen; cleanup is best effort
                    self.log(
                        "[cleanup] failed to release stateful inputs:\n" + traceback.format_exc()
                    )
            case Finish(termination=termination):
                self._finish(termination)

    def _raised(self, failure: str) -> None:
        """An exception the legacy loop let escape it: the run is invalid and the episode still
        finalizes; the first traceback is the summary's error (runner.py main)."""
        self.log(f"[default] decision loop raised:\n{failure}")
        if self._loop_error is None:
            self._loop_error = self.env._loop_error = failure
        self.post(Invalidated("runner", "run_exception", failure))

    def _unanswered(self, command, failure: str):
        """The answer to a command whose run raised, so that the kernel, which waits for it, can
        settle the decision; the run_exception before it decides the reason."""
        semantic, ops = self._now
        match command:
            case CheckEnd(request=request):
                return EndChecked(request, error=failure)
            case Capture():
                return FrameCaptured(error=failure)
            case Execute(decision=decision):
                return ActionExecuted(decision, error=failure)
            case AdvanceClock():
                return ClockStepped(failure)
            case ReleaseKeys(keys=keys):
                return KeysReleased(keys, failure)
            case RenderFrame():
                return FrameRendered(error=failure)
            case Commit(outcome=outcome):
                return OutcomeCommitted(outcome.decision, failure)
            case FreezeSnapshot():
                return SnapshotFrozen(semantic, semantic if ops is None else ops, failure)
        return None

    def _env(self, operation, *args, **kwargs):
        """Run one synchronous environment operation. An invalid run the environment latched
        itself, with its own reason, goes to the kernel before the command's answer."""
        latched = getattr(self.env, "_invalid_run", None)
        try:
            return operation(*args, **kwargs)
        finally:
            invalid = getattr(self.env, "_invalid_run", None)
            if invalid is not None and invalid is not latched:
                self.post(_invalidated(invalid))

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
        self._emit(
            TraceEvent.INFERENCE_STARTED,
            {
                **episode_records.inference_started(
                    observation, decision, usage_before, native_output_limit, budget
                ),
                "generation": command.generation,
                "remaining_clock_tokens": remaining,
            },
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
            clock.event(),
        )
        self._generations[held.generation] = self._decision.generation = held
        clock.start_thread(lambda: self._generate(held), name=f"generation-{held.generation}")

    def _generate(self, held: _Generation) -> None:
        """The worker: one generation, handed back to the shell's thread as it came."""
        try:
            pending = self.agent.generate(
                held.generation, held.decision, held.observation, held.budget
            )
            held.result = pending.result
        except Exception:  # noqa: BLE001 - a failed generation invalidates the run
            failure = traceback.format_exc()
            self.log(f"[default] k={held.decision} generation failed, recovering:\n{failure}")
            latency = round(self.clocks.operational.time() - held.started_at, 2)
            held.result = episode_records.generation_error(latency, failure)
        self._returned.append(held)
        held.done.set()
        self._wake.set()

    def _settle_returned(self) -> None:
        """Settle the results the workers handed back, on this thread, and hand them to the
        kernel: the next step files them."""
        while self._returned:
            held = self._returned.popleft()
            try:
                result = self._settle(held)
            except Exception:  # noqa: BLE001 - as an exception that escaped the legacy loop
                self._raised(traceback.format_exc())
            else:
                self.post(GenerationFinished(held.generation, result))

    def _settle(self, held: _Generation) -> GenerationResult:
        """The result's annotations, usage, token accounting, cap audit and trajectory row, in
        the runner's order (runner.py ``_decide``), and what the kernel needs of it."""
        r, budget, clock_mode, agent = held.result, held.budget, self.config.clock_mode, self.agent
        episode_records.annotate_result(
            r,
            decision=held.decision,
            observation=held.observation,
            native_output_limit=held.native_output_limit,
            budget=budget,
            remaining_clock_tokens=held.remaining,
            clock_mode=clock_mode,
            tokens_per_frame=self.config.tokens_per_frame,
        )
        usage_after = episode_records.usage_snapshot(agent)
        if held.usage_before is not None and usage_after is not None:
            r["frame_usage"] = episode_records.frame_usage(held.usage_before, usage_after)
        episode_records.account_tokens(r, held.remaining, self.env)
        episode_records.audit_output_caps(agent, r, budget)
        failure = episode_records.finalize_decision_audit(agent, r)
        if failure is not None:
            self.log(f"[default] trajectory audit write failed:\n{failure}")
        # The kernel clamps the count to the allowance itself (kernel/turns.py _accept).
        unclamped = r["unclamped_clock_output_tokens"]
        return GenerationResult(
            action=r.get("action"),
            output_tokens=r["clock_output_tokens"] if unclamped is None else unclamped,
            latency=r["latency"],
            protocol_failure=(r.get("action_protocol") or {}).get("protocol_failure"),
            invalid=episode_records.result_invalid(r, agent, clock_mode),
        )

    def _abort(self, generation: int) -> None:
        held = self._generations.get(generation)  # None when its StartGeneration raised
        if held is None or held.done.is_set():
            return  # nothing is running to interrupt; the kernel discards the result
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
        hard exit. Runs once."""
        if self._drained:
            return
        self._drained = True
        pending = [held for held in self._abandoned if not held.done.is_set()]
        if not pending:
            return
        seconds, clock = self.config.timing.abandoned_drain_s, self.clocks.operational
        self.log(f"[default] draining {len(pending)} abandoned generate worker(s), <= {seconds:g}s")
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

    # ---- the action ---------------------------------------------------------------------------

    def _execute(self, command: Execute) -> None:
        env, decision = self.env, self._decision
        env._last_execution_result = None
        env._last_coordinate_trace = None
        decision.executed_at = self.clocks.operational.time()
        status = self._env(env.execute, command.action, release_at=command.release_at)
        rejection = error = None
        if status == "failed":
            # A guest refusal is the model's action failing, not the environment's.
            if env._last_rejected_by_guest:
                rejection = env._last_exec_error or ""
            else:
                error = env._last_exec_error
        decision.coordinate_trace = env._last_coordinate_trace
        self.post(
            ActionExecuted(
                command.decision,
                result=env._last_execution_result,
                held_keys=env.last_held_keys,
                rejection=rejection,
                error=error,
                end_checks=env.last_end_checks,
            )
        )

    def _commit(self, outcome: Outcome) -> None:
        env, decision, config = self.env, self._decision, self.config
        r = decision.accepted
        if r is None:
            # A move over the per-move cap reports no result (kernel/turns.py on_timer): its
            # record has what the decision was given.
            held, r = decision.generation, {}
            episode_records.annotate_result(
                r,
                decision=held.decision,
                observation=held.observation,
                native_output_limit=held.native_output_limit,
                budget=held.budget,
                remaining_clock_tokens=held.remaining,
                clock_mode=config.clock_mode,
                tokens_per_frame=config.tokens_per_frame,
            )
        if outcome.protocol_failure == "unsupported_terminal_action":
            r["action_protocol"] = episode_records.terminal_wait_protocol(
                r.get("action_protocol") or {}
            )
        if outcome.frame_age_s is not None:
            env.frame_ages.append(outcome.frame_age_s)
        try:
            self.agent.commit(outcome)
        except Exception:  # noqa: BLE001 - the kernel invalidates the run for it
            failure = traceback.format_exc()
            self.log(f"[default] commit_action({outcome.status}) failed:\n{failure}")
            self.post(OutcomeCommitted(outcome.decision, failure))
        else:
            self.post(OutcomeCommitted(outcome.decision))
        resolved = decision.resolved
        try:
            self._env(
                env.record_lifecycle,
                episode_records.lifecycle_record(
                    r,
                    decision=outcome.decision,
                    observation_tick=r["observation_tick"],
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
            self.post(Invalidated("harness", "action_lifecycle_write_failed", failure))
        self._checkpoint()
        t_k, tau, action = r["observation_tick"], outcome.action_clock_position, outcome.action
        # On the wall clock the delay is the decision's latency (kernel/turns.py _accept).
        d_ticks = (
            outcome.token_delay_frames
            if self.token
            else max(0.0, r.get("latency") or 0.0) * config.env_fps
        )
        self.timeline.append((outcome.decision, t_k, d_ticks, tau, action))
        act = None if action is None else (action.action_type.value, action.parameters)
        self.log(
            f"[default] k={outcome.decision} t_k={t_k} δ={d_ticks:g} τ={tau:g} act={act} -> "
            f"{outcome.status}" + (f" tokens={outcome.action_output_tokens}" if self.token else "")
        )

    # ---- the end ------------------------------------------------------------------------------

    def _freeze(self) -> None:
        env = self.env
        # The presentation boundary precedes the scorer's snapshot.
        env._active_end_unix_s = self.clocks.operational.time()
        env._termination_snapshot_captured = False
        snapshotter = env.task.evaluation_snapshot
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
                env._termination_snapshot_error = error
            else:
                env._termination_snapshot = snapshot
                env._termination_snapshot_captured = True
                env._termination_snapshot_error = None
                self._snapshot_read = True
        semantic, ops = self._read()  # when the read returned
        self.post(SnapshotFrozen(semantic, semantic if ops is None else ops, error))

    def _finish(self, termination: Termination) -> None:
        """The termination record main() and the report read, in world seconds to nine places
        (runner.py ``_termination``); the token clock's are ticks."""
        env = self.env
        ticks_per_s = self.config.env_fps if self.token else 1.0
        end_s = termination.ended_at / ticks_per_s
        snapshot_s = termination.snapshot_at / ticks_per_s if self._snapshot_read else None
        env._termination = {
            "stop_reason": termination.stop_reason,
            "end_s": round(end_s, 9),
            "snapshot_s": None if snapshot_s is None else round(snapshot_s, 9),
            "snapshot_delay_s": None if snapshot_s is None else round(snapshot_s - end_s, 9),
        }
        env._stop_reason = termination.stop_reason
        env._think_spent = self.policy.spent
        env._dropped_intermediate_frames = self.policy.dropped_frames
        if termination.invalid is not None:
            env._invalid_run = _invalid_record(termination.invalid)
        self._finished = True

    # ---- the episode trace --------------------------------------------------------------------

    def _record(self, entry: Entry) -> None:
        """Write one kernel record (D6). One under a TraceEvent name keeps it, with the payload
        the runner writes; a frame's echo is ObservationCaptured and a generation's is
        OutputChunkReceived and InferenceCompleted; any other is a KernelRecord."""
        kind, data, decision = entry.kind, entry.data, self._decision
        if kind == TraceEvent.INFERENCE_STARTED.value:
            self._started = entry  # StartGeneration, which follows it, writes it with its caps
        elif kind == "FrameCaptured" and data.error is None:
            self._emit(
                TraceEvent.OBSERVATION_CAPTURED,
                episode_records.observation_captured(decision.observation, decision.index),
                entry,
            )
        elif kind == "GenerationFinished":
            held = self._generations[data.generation]
            r, observation_id = held.result, held.result["observation_id"]
            for chunk in episode_records.output_chunks(r, held.decision, observation_id):
                self._emit(TraceEvent.OUTPUT_CHUNK_RECEIVED, chunk, entry)
            self._emit(
                TraceEvent.INFERENCE_COMPLETED,
                episode_records.inference_completed(r, held.decision, observation_id),
                entry,
            )
        elif kind == TraceEvent.ACTION_PROPOSED.value:
            self._emit(
                TraceEvent.ACTION_PROPOSED,
                episode_records.action_proposed(
                    decision.accepted, data["decision"], data["observation_id"]
                ),
                entry,
            )
        elif kind == TraceEvent.ACTION_RESOLVED.value:
            executed = decision.executed_at is not None
            decision.resolved = episode_records.action_resolved(
                decision=data["decision"],
                observation_id=data["observation_id"],
                action=data["action"],
                status=data["status"],
                terminal_after_action=data["terminal_after_action"],
                # The record of an unusable result has neither (kernel/turns.py _unusable).
                action_clock_position=data.get("action_clock_position"),
                action_resolved_unix_s=(
                    decision.executed_at if executed else self.clocks.operational.time()
                ),
                environment_tick=data.get("environment_tick"),
                execution_result=data.get("execution_result"),
                coordinate_trace=decision.coordinate_trace,
            )
            self._emit(TraceEvent.ACTION_RESOLVED, decision.resolved, entry)
        elif kind == TraceEvent.ACTION_APPLIED.value:
            self._emit(TraceEvent.ACTION_APPLIED, decision.resolved, entry)
        elif kind == TraceEvent.CLOCK_ADVANCED.value:
            self._emit(TraceEvent.CLOCK_ADVANCED, data, entry)
        elif kind == TraceEvent.RUN_INVALIDATED.value:
            invalid = Invalidated(**data)
            if invalid.reason == "observation_stream_frozen":
                env = self.env
                env._stream_freeze_count = getattr(env, "_stream_freeze_count", 0) + 1
                env._stream_frozen = True
                self.log(
                    "[default] observation stream FROZEN (same id/tick/hash x"
                    f"{self.config.timing.observation_freeze_n}); invalidating run"
                )
            self.log(
                f"[invalid-run] {invalid.category}:{invalid.reason}"
                + (f" — {invalid.detail}" if invalid.detail else "")
            )
            self._emit(TraceEvent.RUN_INVALIDATED, _invalid_record(invalid), entry)
        else:
            self._emit("KernelRecord", {"kind": kind, "data": data}, entry)

    def _emit(self, event: TraceEvent | str, payload: dict, entry: Entry) -> None:
        if self.trace is None or self._trace_failed:
            return
        try:
            self.trace.emit(
                event, payload, kernel_seq=entry.seq, semantic=entry.semantic, ops=entry.ops
            )
        except Exception:  # noqa: BLE001 - any write failure invalidates the run
            self._trace_failed = True
            self.post(Invalidated("harness", "episode_trace_write_failed", traceback.format_exc()))

    def _checkpoint(self) -> None:
        """Make the trace durable at the end of every decision."""
        if self.trace is None or self._trace_failed:
            return
        try:
            self.trace.checkpoint()
        except Exception:  # noqa: BLE001 - any write failure invalidates the run
            self._trace_failed = True
            self.post(
                Invalidated("harness", "episode_trace_checkpoint_failed", traceback.format_exc())
            )
