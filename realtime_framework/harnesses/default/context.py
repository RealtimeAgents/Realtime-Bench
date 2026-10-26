from __future__ import annotations

# Session
"""Model-session helpers without token estimation or token-based eviction."""

import threading as threading

from realtime_framework.harnesses.default.prompts import ACTION_RESULT_NOTES


def native_output_limit(native_limit: int | None, provider_limit: int | None = None) -> int | None:
    """Resolve the unavoidable physical output ceiling for a provider request; None is unknown."""
    limits = [max(1, int(limit)) for limit in (native_limit, provider_limit) if limit is not None]
    return min(limits) if limits else None


def action_result_note(status: str, detail: str = "") -> str:
    """Return model feedback when a proposed action was not applied."""
    note = ACTION_RESULT_NOTES.get(status, "")
    if note and detail:
        note += f" ({detail})"
    return note


# Runtime
"""Common session and generation state machine for every catalog model."""

from copy import deepcopy as deepcopy
from collections.abc import Callable as Callable
from collections.abc import Iterable as Iterable
from dataclasses import replace as replace

from realtime_framework.clock import Clock
from realtime_framework.clock import SystemClock
from realtime_framework.entity.context_policy import PathPolicy
from realtime_framework.entity.model import CanonicalTurn
from realtime_framework.entity.model import ConnectRequest
from realtime_framework.entity.events import GenerationBoundary
from realtime_framework.entity.model import GenerationHandle
from realtime_framework.entity.model import GenerationRequest
from realtime_framework.entity.model import HistorySyncRequest
from realtime_framework.entity.model import HistorySyncResult
from realtime_framework.entity.model import OutputCollector
from realtime_framework.entity.events import ProviderError
from realtime_framework.entity.events import ProviderHistoryState
from realtime_framework.entity.model import ProviderObservation
from realtime_framework.entity.model import RealtimeDriver
from realtime_framework.entity.model import RetryCancelled
from realtime_framework.entity.model import RuntimeState
from realtime_framework.entity.model import RuntimeTurnResult
from realtime_framework.entity.model import SessionSeed
from realtime_framework.entity.model import SyncDisposition
from realtime_framework.entity.model import retry_cancellation
from realtime_framework.entity.records import TraceEvent
from realtime_framework.entity.request import CanonicalRequest
from realtime_framework.entity.request import ContentBlock
from realtime_framework.harnesses.default.execution_thread import ExecutionThread
from realtime_framework.harnesses.default.history import Accept
from realtime_framework.harnesses.default.history import ArchiveBlock
from realtime_framework.harnesses.default.history import History
from realtime_framework.harnesses.default.history import OmitBlock
from realtime_framework.harnesses.default.history import Retain
from realtime_framework.harnesses.default.history import Retained
from realtime_framework.providers.session_lifecycle import SessionStates
from realtime_framework.providers.session_lifecycle import StatelessSession
from realtime_framework.providers.session_lifecycle import replay_items
from realtime_framework.providers.session_lifecycle import session_record
from realtime_framework.sandbox.observation.som import without_observation_interactables

# Builds the canonical request from the turns to send and the archived-action ledger block.
RequestBuilder = Callable[[tuple[CanonicalTurn, ...], ContentBlock | None], CanonicalRequest]


class RequestExceedsContext(RuntimeError):
    """The request did not fit the declared context window, so it was never sent."""

    invalid_reason = "context_exceeded_before_request"


_EVERY_ACTION_OMITTED = (
    "Task instructions and current observation exceed context capacity with every archived "
    "action omitted; use a larger endpoint window"
)


class RealtimeRuntime:
    """Own the protocol-neutral generation flow around one provider driver: the retained history,
    the requests built from it, and each generation's run.

    The session the driver keeps, its connect, recovery, renewal, repair and abort, belongs to the
    session encoder (providers/session_lifecycle.py): a live driver is its own, a stateless one is
    presented as a StatelessSession. The runtime tells it when and from which history.

    The synchronous harness may call the agent from disposable worker threads. Every driver
    operation runs on the agent's execution thread instead (``execution``), which the agent starts
    and stops around the runtime.
    """

    def __init__(
        self,
        driver: RealtimeDriver,
        policy: PathPolicy,
        *,
        history: History,
        trace,
        execution: ExecutionThread,
        clock: Clock | None = None,
    ):
        if not isinstance(driver, RealtimeDriver):
            raise TypeError("realtime driver does not implement the required protocol")
        self.driver = driver
        self.capabilities = driver.capabilities
        self.policy = policy
        self.history = history
        self._trace = trace
        self.execution = execution
        # The operational clock, which also made the execution thread's loop.
        self.clock = clock or SystemClock()
        self.session = (
            driver
            if self.capabilities.persistent_session
            else StatelessSession(driver, clock=self.clock, trace=trace)
        )
        self._states = SessionStates(trace, self.capabilities.protocol_family)
        # The context-policy items behind this turn's history_block_compaction sync.
        self._compaction_items: list[str] = []
        self._lock = threading.Lock()
        self._closed = False
        # The cancel event of each generation under way.
        self._running: set[threading.Event] = set()
        self._active_generation: GenerationHandle | None = None
        self._seed: SessionSeed | None = None

    @property
    def state(self) -> RuntimeState:
        """The session encoder's state, until the runtime is closed."""
        with self._lock:
            if self._closed:
                return RuntimeState.CLOSED
        return self.session.state

    def start(self, seed: SessionSeed, *, timeout: float = 60.0) -> None:
        self._seed = seed
        self.execution.run(self.session.connect(ConnectRequest(seed=seed)), timeout=timeout)

    def _abandoned(self, cancel: threading.Event) -> bool:
        """Whether the generation of this cancel event was aborted, or the runtime closed."""
        with self._lock:
            return cancel.is_set() or self._closed

    def _stop_if_abandoned(self, cancel: threading.Event, before: str) -> None:
        """Stop an aborted generation. _run_turn checks before each change it writes into the
        history (_write), before it sends its observation, before it counts a request with every
        archived action left out, and before it starts its reply."""
        if self._abandoned(cancel):
            raise RetryCancelled(f"generation abandoned before {before}")

    def _write(self, cancel: threading.Event, before: str, change):
        """Write ``change`` into the history, unless the generation of this cancel event was
        abandoned: it still runs, and the history may be the next generation's by now."""
        self._stop_if_abandoned(cancel, before)
        return self.history.write(change)

    def accept(self, accepted: Accept) -> None:
        """Write an accepted generation into the history, on the execution thread, where a
        generation that was abandoned but still runs reads the history."""

        async def write() -> None:
            self.history.write(accepted)

        self.execution.run(write())

    def history_view(
        self, request_builder: RequestBuilder, incoming: Iterable[CanonicalTurn] = ()
    ) -> CanonicalRequest:
        """The retained history (and the turn of the generation under way) where a session
        encoder needs it before or after the request exists: the same builder and blocks as the
        request."""
        return request_builder(self.history.turns + tuple(incoming), self.history.ledger_block())

    def _recorder(
        self,
        removed_turns: tuple[CanonicalTurn, ...] = (),
        incoming: tuple[CanonicalTurn, ...] = (),
    ) -> Callable[..., None]:
        """The ``record`` of one session operation: a HistoryChanged for each sync it ran, with
        the turns it removed and the incoming turns it carried, and for a reopened session."""

        def record(sync, result=None, error=None) -> None:
            if sync is None:
                self._record_reconnect()
                return
            self._record_sync(sync, removed_turns, incoming, result=result, error=error)

        return record

    async def _run_turn(
        self,
        observations: tuple[ProviderObservation, ...],
        request: GenerationRequest,
        collector: OutputCollector,
        *,
        unattached: str,
        cancel: threading.Event,
        request_builder: RequestBuilder,
        decision: int,
    ) -> RuntimeTurnResult:
        if len(observations) != 1:
            raise ValueError("each generation accepts exactly one observation")
        seed = self._seed
        history = self.history
        await self.session.ensure_ready(seed, self.history_view(request_builder), self._recorder())
        incoming = [
            CanonicalTurn(observation=value, decision=decision, unattached_feedback=unattached)
            for value in observations
        ]
        # The policy archive and fold apply before the generation, accepted or not: a discarded
        # generation leaves them, and their HistoryChanged records, in place.
        retention = self._write(
            cancel, "its history", Retain(tuple(incoming), self.clock.time_ns())
        )
        self._record_retention(retention)
        sync = HistorySyncResult(
            SyncDisposition.EXACT,
            ProviderHistoryState.EXACT,
            detail="already synchronized",
        )
        if retention.changed:
            sync = await self.session.synchronize(
                HistorySyncRequest(
                    seed,
                    "history_block_compaction",
                    self.history_view(request_builder),
                    force_rebuild=(
                        self.capabilities.persistent_session
                        and not self.capabilities.can_compact_history_in_place
                    ),
                ),
                self._recorder(retention.archived),
            )
        physical_metadata = {}
        renewal = await self.session.renew(
            seed, self.history_view(request_builder), self._recorder()
        )
        if renewal is not None:
            sync = renewal

        if not incoming:
            raise RuntimeError("no current observation")
        self._stop_if_abandoned(cancel, "its request")
        view = self.history_view(request_builder, incoming)
        input_refs = [
            await self.driver.send_observation(turn.observation, view) for turn in incoming
        ]
        input_refs = [refs or {} for refs in input_refs]

        compactions = 0
        archived_count = 0
        omitted_blocks = 0
        omitted_before = history.omitted
        while True:
            canonical = request_builder(history.turns + tuple(incoming), history.ledger_block())
            request = replace(request, canonical=canonical)
            if not (self.policy.pressure_archive and self.capabilities.request_size_estimation):
                break
            status = dict(await self.driver.request_context_status(request))
            physical_metadata.setdefault("request_context_before", status)
            physical_metadata.update(
                request_context=status,
                request_context_compactions=compactions,
                request_context_archived_turns=archived_count,
            )
            if not status.get("pressure"):
                break
            # The count can suspend (a "vllm" counter): by now the history may be the next
            # generation's, which an aborted one must not archive or cut (_write).
            if not history.turns:
                # Every turn is archived: leave out the earliest archived actions, a block at a
                # time, before the request is given up. Once one block was not enough, a count
                # with every archived action left out tells whether more blocks can help.
                if omitted_blocks == 1 and history.omitted < history.archived:
                    self._stop_if_abandoned(cancel, "its pressure archive")
                    if not await self._fits_with_every_action_omitted(
                        request, request_builder, incoming
                    ):
                        self._stop_if_abandoned(cancel, "its pressure archive")
                        raise RequestExceedsContext(_EVERY_ACTION_OMITTED)
                omitted = self._write(cancel, "its pressure archive", OmitBlock())
                if not omitted:
                    raise RequestExceedsContext(_EVERY_ACTION_OMITTED)
                self._record_omission(omitted, status)
                omitted_blocks += 1
                physical_metadata["request_context_omitted_actions"] = (
                    history.omitted - omitted_before
                )
                continue
            archived, archived_actions = self._write(
                cancel, "its pressure archive", ArchiveBlock(self.clock.time_ns())
            )
            self._record_pressure_archive(archived, archived_actions, status)
            compactions += 1
            archived_count += len(archived)

        self._stop_if_abandoned(cancel, "its reply")
        self._record_generation_prepared(incoming, physical_metadata, request)
        provider_call_started_monotonic_ns = self.clock.monotonic_ns()
        provider_call_started_wall_time_ns = self.clock.time_ns()
        generation = await self.driver.start_generation(request)
        stream_started_monotonic_ns = self.clock.monotonic_ns()
        generation.metadata.setdefault("input_refs", input_refs)
        with self._lock:
            self._active_generation = generation
        started = self.clock.monotonic()
        terminal_history = ProviderHistoryState.UNKNOWN
        # The totals and turn counts below are read from the history once the request has
        # started: a generation aborted while it starts may find them as the next one has left
        # them. Its GenerationPrepared holds them as its request went.
        provider_metadata: dict = {
            "provider_call_started_monotonic_ns": provider_call_started_monotonic_ns,
            "provider_call_started_wall_time_ns": provider_call_started_wall_time_ns,
            "provider_stream_started_monotonic_ns": stream_started_monotonic_ns,
            "history_folded_visuals": len(retention.folded),
            "history_archived_turns": len(retention.archived),
            "history_archived_actions": retention.archived_actions,
            "history_archived_actions_total": history.archived,
            # Only once the ledger leaves actions out: a run that never does records no more.
            **(
                {"history_omitted_actions_total": history.omitted} if history.omitted else {}
            ),
            "history_detailed_turns": len(history.turns),
            "history_current_observations": len(incoming),
            "history_prompt_turns": len(history.turns) + len(incoming),
            "history_active_visuals": sum(
                not turn.visual_folded for turn in history.turns + tuple(incoming)
            ),
            "history_folded_turns": sum(
                turn.visual_folded for turn in history.turns + tuple(incoming)
            ),
            **physical_metadata,
        }
        provider_metadata.update(
            {
                "history_som_retained_blocks": sum(
                    without_observation_interactables(turn.observation.prompt)
                    != turn.observation.prompt
                    for turn in history.turns
                ),
                "history_som_current_blocks": sum(
                    without_observation_interactables(turn.observation.prompt)
                    != turn.observation.prompt
                    for turn in incoming
                ),
            }
        )
        reconnect_required = False
        events = self.driver.events(generation)
        try:
            async for event in events:
                if isinstance(event, ProviderError):
                    provider_metadata.setdefault("errors", []).append(event.detail)
                    if event.fatal:
                        raise RuntimeError(event.detail)
                if isinstance(event, GenerationBoundary):
                    provider_metadata.update(event.metadata)
                    reconnect_required = reconnect_required or bool(
                        event.metadata.get("reconnect_required")
                    )
                    if event.kind in {"generation_complete", "turn_complete", "interrupted"}:
                        terminal_history = event.history_state
                # Every generation runs to its terminal event so provider usage is complete.
                collector.consume(event)
        finally:
            with self._lock:
                if self._active_generation is generation:
                    self._active_generation = None

        cancelled = self._abandoned(cancel)
        if terminal_history != ProviderHistoryState.EXACT:
            self._states.record(
                "unknown", "generation_terminal_state", terminal_history, request.generation_id
            )
        output = collector.finish()
        turn = incoming[-1]
        turn.output = output.text
        turn.output_wall_ns = self.clock.time_ns()
        turn.output_reasoning = output.reasoning_text
        turn.provider_refs.update(generation.metadata.get("completed_turn_refs") or {})

        # The turn joins the history only once the result is accepted (accept). A repair
        # after the generation already includes it: a result that is discarded was aborted first,
        # and the session that held it is closed, by the abort or by the repair it came during,
        # so the next generation rebuilds from the accepted history.
        repair = None
        if not cancelled:
            try:
                repair = await self.session.repair(
                    HistorySyncRequest(
                        seed,
                        provider_metadata.get("reconnect_reason", "provider_history_changed"),
                        self.history_view(request_builder, (turn,)),
                        current_generation=generation,
                    ),
                    terminal_history,
                    reconnect_required,
                    self._recorder(incoming=(turn,)),
                )
            except RetryCancelled:
                # Aborted during the repair: the generation itself came back, usage and all.
                cancelled = True
        if cancelled:
            # abort() already dirtied the session and scheduled its cleanup.
            sync = HistorySyncResult(
                SyncDisposition.REBUILT,
                ProviderHistoryState.UNKNOWN,
                detail="generation superseded by harness timeout",
            )
        elif repair is not None:
            sync = repair

        finished_monotonic_ns = self.clock.monotonic_ns()
        return RuntimeTurnResult(
            output=output,
            turn=turn,
            generation_id=request.generation_id,
            history_sync=sync,
            terminal_history_state=terminal_history,
            interrupted=False,
            cancelled=cancelled,
            latency_s=self.clock.monotonic() - started,
            provider_call_latency_s=(finished_monotonic_ns - provider_call_started_monotonic_ns)
            / 1_000_000_000,
            provider_metadata=provider_metadata,
        )

    def run_turn(
        self,
        observations: list[ProviderObservation],
        request: GenerationRequest,
        collector: OutputCollector,
        *,
        unattached: str,
        timeout: float | None,
        request_builder: RequestBuilder,
        decision: int,
        cancel: threading.Event,
    ) -> RuntimeTurnResult:
        """One generation, whose frame opens with the ``unattached`` feedback. ``cancel`` is its
        own event: once it is set, the generation stops at its next check (_stop_if_abandoned)
        and its retries end, the session encoder's included (they run in its
        retry_cancellation). Closing the runtime sets it. Its turn comes back in the result, for
        accept once it is accepted."""
        with self._lock:
            self._running.add(cancel)
        try:
            with retry_cancellation(cancel):
                return self.execution.run(
                    self._run_turn(
                        tuple(observations),
                        request,
                        collector,
                        unattached=unattached,
                        cancel=cancel,
                        request_builder=request_builder,
                        decision=decision,
                    ),
                    timeout=timeout,
                )
        except Exception:
            if not self._abandoned(cancel):
                self.session.mark_dirty()
            raise
        finally:
            with self._lock:
                self._running.discard(cancel)

    def abort(self) -> None:
        """Interrupt whichever generation is generating, close the session and mark it dirty:
        the next generation reconnects. The caller has already set the cancel event of each
        generation it aborts, which stops that generation's requests."""
        with self._lock:
            if self._closed:
                return
            generation = self._active_generation
        self.session.mark_dirty()
        self.execution.submit(self.session.abort(generation))

    def close(self) -> None:
        """Abandon the generations under way, then close the session, once the last abort's
        cleanup is done or has run for a while. The agent stops the execution thread after
        this."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for cancel in self._running:
                cancel.set()
        # The runner closes an agent whose initialize failed, before or after the execution
        # thread started; one that never started has no session to close.
        if self.execution.running:
            try:
                # Longer than the session waits for an abort's cleanup, with time to close after.
                self.execution.run(self.session.close(), timeout=15.0)
            except Exception:  # noqa: BLE001, S110 - destructor-style best effort
                pass

    # ---- agent trace -------------------------------------------------------------------------
    # Each change to the model-visible history is recorded where it happens, so the history at any
    # step follows from the trace. policy_item names the PathPolicy fields behind the change
    # (entity/context_policy.py), and policy_values their values for this API.

    def _turn_refs(self, turns: Iterable[CanonicalTurn]) -> list[dict]:
        return [
            {
                "decision": turn.decision,
                "observation_id": turn.observation.observation_id,
                "tick": turn.observation.observation.tick_number,
                "action_history_index": self.history.index(turn.decision),
            }
            for turn in turns
        ]

    def _history_changed(self, kind: str, trigger: str, policy_item, **fields) -> None:
        self._trace.capture(
            TraceEvent.HISTORY_CHANGED,
            {
                "kind": kind,
                "trigger": trigger,
                "policy_item": policy_item,
                "policy_version": self.policy.version,
                **fields,
            },
        )

    def _ledger_change(self, archived_actions: int) -> dict:
        total = self.history.archived
        return {"archived_actions_before": total - archived_actions, "archived_actions_after": total}

    def _record_retention(self, retention: Retained) -> None:
        """The policy archive and screenshot fold applied before this turn's request."""
        self._compaction_items = []
        if retention.archived:
            self._compaction_items.append("turns")
            self._history_changed(
                "archive",
                "policy",
                "turns",
                policy_values={"keep": self.policy.turns.keep, "block": self.policy.turns.block},
                turns=self._turn_refs(retention.archived),
                ledger=self._ledger_change(retention.archived_actions),
            )
        if retention.folded:
            self._compaction_items.append("visuals")
            self._history_changed(
                "visual_fold",
                "policy",
                "visuals",
                policy_values={
                    "keep": self.policy.visuals.keep,
                    "block": self.policy.visuals.block,
                },
                turns=self._turn_refs(retention.folded),
            )

    def _record_pressure_archive(self, archived, archived_actions, status) -> None:
        self._history_changed(
            "pressure_archive",
            "policy",
            "pressure_archive",
            policy_values={"block": self.policy.turns.block},
            turns=self._turn_refs(archived),
            ledger=self._ledger_change(archived_actions),
            request_context=status,
        )

    async def _fits_with_every_action_omitted(self, request, request_builder, incoming) -> bool:
        """Count the request as it would go with every archived action left out, leaving the
        history as it is."""
        block = self.history.ledger_block(omitted=self.history.archived)
        canonical = request_builder(self.history.turns + tuple(incoming), block)
        status = await self.driver.request_context_status(replace(request, canonical=canonical))
        return not status.get("pressure")

    def _record_omission(self, omitted: int, status) -> None:
        """The ``omitted`` earliest archived actions an OmitBlock just left out of the ledger."""
        self._history_changed(
            "ledger_omission",
            "policy",
            "pressure_archive",
            policy_values={"block": self.policy.turns.block},
            ledger={
                "archived_actions": self.history.archived,
                "omitted_actions_before": self.history.omitted - omitted,
                "omitted_actions_after": self.history.omitted,
            },
            request_context=status,
        )

    def _record_sync(self, sync, removed_turns, incoming=(), *, result=None, error=None):
        """One HistoryChanged per sync call (session_record), with the policy items behind it
        and the turns it removed and replayed."""
        record = session_record(sync, result, error)
        compaction = sync.reason == "history_block_compaction"
        policy_item = list(self._compaction_items) if compaction else []
        if record.rebuilt:
            policy_item += replay_items(self.policy)
        self._history_changed(
            record.kind,
            record.trigger,
            policy_item or None,
            turns=self._turn_refs(removed_turns),
            replayed_turns=len(self.history.turns) + len(incoming) if record.rebuilt else None,
            **record.fields,
        )
        self._states.record(*record.state)

    def _record_reconnect(self) -> None:
        """The session's ensure_ready replayed the canonical turns on a new session after an
        abort or failure."""
        if not self.capabilities.persistent_session:
            return  # a stateless connect sends nothing; the next request carries the history
        record = session_record(None)
        self._history_changed(
            record.kind,
            record.trigger,
            replay_items(self.policy) or None,
            turns=[],
            replayed_turns=len(self.history.turns),
            **record.fields,
        )
        self._states.record(*record.state)

    def _record_generation_prepared(self, incoming, physical_metadata, request) -> None:
        """The history the generation's request went with, and the canonical request itself as
        its body."""
        history = self.history
        turns = history.turns + tuple(incoming)
        self._trace.capture(
            TraceEvent.GENERATION_PREPARED,
            {
                "generation": request.generation,
                "observation_id": incoming[-1].observation.observation_id,
                "persistent_session": self.capabilities.persistent_session,
                "history": {
                    "archived_actions_total": history.archived,
                    # As history_omitted_actions_total in the step's provider metadata.
                    **({"omitted_actions_total": history.omitted} if history.omitted else {}),
                    "detailed_turns": len(history.turns),
                    "current_observations": len(incoming),
                    "prompt_turns": len(turns),
                    "active_visuals": sum(not turn.visual_folded for turn in turns),
                    "folded_turns": sum(turn.visual_folded for turn in turns),
                    "som_retained_blocks": sum(
                        without_observation_interactables(turn.observation.prompt)
                        != turn.observation.prompt
                        for turn in history.turns
                    ),
                    "som_current_blocks": sum(
                        without_observation_interactables(turn.observation.prompt)
                        != turn.observation.prompt
                        for turn in incoming
                    ),
                },
                "request_context": physical_metadata or None,
            },
            body=request.canonical,
        )
