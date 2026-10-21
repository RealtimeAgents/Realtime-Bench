"""What the shell that runs the event kernel needs from the environment and from the agent.

The shell drives ``Kernel(TurnPolicy(config))`` (evaluation/kernel): it reads the clocks, runs the
commands each ``step`` returns against a ShellEnvironment and a KernelAgent, and answers them with
the events in evaluation/kernel/events.py.

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

from collections.abc import Callable
from typing import Any, Protocol

from realtime_framework.entity.actions import Action, Observation
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.entity.model import PendingDecision
from realtime_framework.evaluation.kernel.events import Outcome


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
