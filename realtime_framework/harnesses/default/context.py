from __future__ import annotations

# Session
"""Model-session history helpers without token estimation or token-based eviction."""

import json as json
import threading as threading

from realtime_framework.harnesses.default.prompts import ACTION_RESULT_NOTES


def _mark_oldest_active_visuals(turns: list[dict], count: int) -> int:
    marked = 0
    for turn in turns:
        if turn.get("visual_folded"):
            continue
        turn["visual_folded"] = True
        marked += 1
        if marked >= count:
            break
    return marked


def apply_active_visual_budget(turns: list[dict], budget: int, block_size: int) -> int:
    """Fold old screenshots in blocks without making any text-retention decision."""
    active = sum(not turn.get("visual_folded") for turn in turns)
    folded = 0
    while active > budget:
        count = min(block_size, active - 1)
        folded += _mark_oldest_active_visuals(turns, count)
        active -= count
    return folded


def executed_action_ledger(history: list[dict], *, limit: int | None = None) -> str:
    """Serialize actual old actions as compact deterministic JSON Lines."""
    items = list(history or [])
    if limit is not None:
        items = items[: max(0, int(limit))]
    lines = []
    for item in items:
        record = {
            "decision": item.get("tick", "?"),
            "observed_at_tick": item.get("observation_tick", item.get("tick", "?")),
            "requested": item.get("requested", item.get("executed", "?")),
            "actual": item.get("executed", "?"),
            "status": item.get("result", item.get("status", "pending")),
        }
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return "\n".join(lines)


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
from realtime_framework.harnesses.default.request import ledger_block
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
        trace,
        execution: ExecutionThread,
        clock: Clock | None = None,
    ):
        if not isinstance(driver, RealtimeDriver):
            raise TypeError("realtime driver does not implement the required protocol")
        self.driver = driver
        self.capabilities = driver.capabilities
        self.policy = policy
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
        self._turns: list[CanonicalTurn] = []
        self._archived_action_count = 0
        # The earliest archived actions the ledger leaves out: a request that did not fit with
        # every turn archived dropped them. As archived turns do, they stay out for the rest of
        # the episode, though a later request would fit with them, so that the history the model
        # sees does not change back.
        self._omitted_action_count = 0
        self._ledger_wall_ns: int | None = None  # when the archived-action ledger last grew

    @property
    def state(self) -> RuntimeState:
        """The session encoder's state, until the runtime is closed."""
        with self._lock:
            if self._closed:
                return RuntimeState.CLOSED
        return self.session.state

    @property
    def turns(self) -> tuple[CanonicalTurn, ...]:
        return tuple(self._turns)

    @property
    def archived_action_count(self) -> int:
        return self._archived_action_count

    def start(self, seed: SessionSeed, *, timeout: float = 60.0) -> None:
        self._seed = seed
        self.execution.run(self.session.connect(ConnectRequest(seed=seed)), timeout=timeout)

    def _abandoned(self, cancel: threading.Event) -> bool:
        """Whether the generation of this cancel event was aborted, or the runtime closed."""
        with self._lock:
            return cancel.is_set() or self._closed

    def _stop_if_abandoned(self, cancel: threading.Event, before: str) -> None:
        """Stop an aborted generation. _run_turn checks before it applies the retention policy,
        before it sends its observation, before each pressure archive and before it starts its
        reply."""
        if self._abandoned(cancel):
            raise RetryCancelled(f"generation abandoned before {before}")

    def record_execution_feedback(self, feedback: str, decision: int) -> None:
        """Attach an action's execution feedback to the turn of its decision."""
        turns = [turn for turn in self._turns if turn.decision == decision]
        if turns:
            turns[-1].execution_feedback = feedback
            turns[-1].feedback_wall_ns = self.clock.time_ns()

    def accept_turn(self, turn: CanonicalTurn) -> None:
        """Write an accepted generation's turn into the retained history. The write runs on the
        execution thread, where a generation that was abandoned but still runs reads the history."""

        async def write() -> None:
            self._turns.append(turn)

        self.execution.run(write())

    def _action_ledger(self, history: list[dict]) -> str:
        return executed_action_ledger(history, limit=self._archived_action_count)

    def ledger_block(self, ledger: str, omitted: int | None = None) -> ContentBlock | None:
        """The ledger as requests carry it: ``ledger`` holds one line per archived action, and
        the ``omitted`` earliest ones (by default the ones left out so far) give way to a note
        that counts them."""
        omitted = self._omitted_action_count if omitted is None else omitted
        if omitted:
            kept = ledger.split("\n")[omitted:]
            note = self.policy.texts.ledger_omission.format(count=omitted)
            ledger = "\n".join([note, *kept])
        return ledger_block(ledger, self._archived_action_count, self._ledger_wall_ns)

    def history_view(
        self,
        request_builder: RequestBuilder,
        ledger: str,
        incoming: Iterable[CanonicalTurn] = (),
    ) -> CanonicalRequest:
        """The retained history (and the turn of the generation under way) where a session
        encoder needs it before or after the request exists: the same builder and blocks as the
        request."""
        return request_builder(tuple(self._turns) + tuple(incoming), self.ledger_block(ledger))

    def _record_archived_turns(self, turns: Iterable[CanonicalTurn]) -> int:
        indexes = [
            turn.action_history_index for turn in turns if turn.action_history_index is not None
        ]
        before = self._archived_action_count
        if indexes:
            self._archived_action_count = max(before, max(indexes) + 1)
        if self._archived_action_count > before:
            self._ledger_wall_ns = self.clock.time_ns()
        return self._archived_action_count - before

    def _retained(
        self,
        turns: Iterable[CanonicalTurn],
        incoming: Iterable[CanonicalTurn],
    ) -> tuple[
        list[CanonicalTurn],
        list[CanonicalTurn],
        tuple[CanonicalTurn, ...],
        dict[str, int],
    ]:
        existing = list(turns)
        additions = list(incoming)
        window, block_size = self.policy.turns.keep, self.policy.turns.block
        archive_count = 0
        while len(existing) - archive_count + len(additions) > window:
            removable = len(existing) - archive_count
            if removable <= 0:
                break
            archive_count += min(block_size, removable)
        archived = existing[:archive_count]
        existing = existing[archive_count:]
        archived_actions = self._record_archived_turns(archived)
        candidates = existing + additions
        previously_folded = {id(turn): turn.visual_folded for turn in candidates}
        records = [turn.retention_record() for turn in candidates]
        apply_active_visual_budget(records, self.policy.visuals.keep, self.policy.visuals.block)
        folded_visuals = 0
        for turn, record in zip(candidates, records):
            if record.get("visual_folded"):
                turn.fold_visual()
            if turn.visual_folded and not previously_folded[id(turn)]:
                folded_visuals += 1
        return (
            existing,
            additions,
            tuple(archived),
            {
                "folded_visuals": folded_visuals,
                "archived_turns": len(archived),
                "archived_actions": archived_actions,
                "existing_history_changed": int(bool(archived) or folded_visuals > 0),
                # The turns behind these counts, for the agent trace.
                "folded_turn_refs": self._turn_refs(
                    turn
                    for turn in candidates
                    if turn.visual_folded and not previously_folded[id(turn)]
                ),
                "archived_turn_refs": self._turn_refs(archived),
            },
        )

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
        action_history: list[dict],
        cancel: threading.Event,
        request_builder: RequestBuilder,
        decision: int,
    ) -> RuntimeTurnResult:
        if len(observations) != 1:
            raise ValueError("each generation accepts exactly one observation")
        seed = self._seed
        executed_action_ledger = self._action_ledger(action_history)
        await self.session.ensure_ready(
            seed, self.history_view(request_builder, executed_action_ledger), self._recorder()
        )
        self._stop_if_abandoned(cancel, "its history")
        incoming = [CanonicalTurn(observation=value, decision=decision) for value in observations]
        # The policy archive and fold apply before the generation, accepted or not: a discarded
        # generation leaves them, and their HistoryChanged records, in place.
        retained, incoming, removed_turns, retention = self._retained(self._turns, incoming)
        self._turns[:] = retained
        self._record_retention(retention)
        executed_action_ledger = self._action_ledger(action_history)
        sync = HistorySyncResult(
            SyncDisposition.EXACT,
            ProviderHistoryState.EXACT,
            detail="already synchronized",
        )
        if retention["existing_history_changed"]:
            sync = await self.session.synchronize(
                HistorySyncRequest(
                    seed,
                    "history_block_compaction",
                    self.history_view(request_builder, executed_action_ledger),
                    force_rebuild=(
                        self.capabilities.persistent_session
                        and not self.capabilities.can_compact_history_in_place
                    ),
                ),
                self._recorder(removed_turns),
            )
        physical_metadata = {}
        executed_action_ledger = self._action_ledger(action_history)
        renewal = await self.session.renew(
            seed, self.history_view(request_builder, executed_action_ledger), self._recorder()
        )
        if renewal is not None:
            sync = renewal

        if not incoming:
            raise RuntimeError("no current observation")
        self._stop_if_abandoned(cancel, "its request")
        history = self.history_view(request_builder, executed_action_ledger, incoming)
        input_refs = [
            await self.driver.send_observation(turn.observation, history) for turn in incoming
        ]
        input_refs = [refs or {} for refs in input_refs]

        compactions = 0
        archived_count = 0
        omitted_blocks = 0
        omitted_before = self._omitted_action_count
        while True:
            executed_action_ledger = self._action_ledger(action_history)
            canonical = request_builder(
                tuple(self._turns + incoming), self.ledger_block(executed_action_ledger)
            )
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
            # generation's, which an aborted one must not archive or cut.
            self._stop_if_abandoned(cancel, "its pressure archive")
            if not self._turns:
                # Every turn is archived: leave out the earliest archived actions, a block at a
                # time, before the request is given up. Once one block was not enough, a count
                # with every archived action left out tells whether more blocks can help.
                hopeless = (
                    omitted_blocks == 1
                    and self._omitted_action_count < self._archived_action_count
                    and not await self._fits_with_every_action_omitted(
                        request, request_builder, incoming, executed_action_ledger
                    )
                )
                self._stop_if_abandoned(cancel, "its pressure archive")
                if hopeless or not self._omit_archived_actions(status):
                    raise RequestExceedsContext(
                        "Task instructions and current observation exceed context capacity with "
                        "every archived action omitted; use a larger endpoint window"
                    )
                omitted_blocks += 1
                physical_metadata["request_context_omitted_actions"] = (
                    self._omitted_action_count - omitted_before
                )
                continue
            archived = self._turns[: self.policy.turns.block]
            del self._turns[: len(archived)]
            archived_actions = self._record_archived_turns(archived)
            self._record_pressure_archive(archived, archived_actions, status)
            compactions += 1
            archived_count += len(archived)

        self._stop_if_abandoned(cancel, "its reply")
        self._record_generation_prepared(incoming, physical_metadata, request.generation)
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
            "history_folded_visuals": retention["folded_visuals"],
            "history_archived_turns": retention["archived_turns"],
            "history_archived_actions": retention["archived_actions"],
            "history_archived_actions_total": self._archived_action_count,
            # Only once the ledger leaves actions out: a run that never does records no more.
            **(
                {"history_omitted_actions_total": self._omitted_action_count}
                if self._omitted_action_count
                else {}
            ),
            "history_detailed_turns": len(self._turns),
            "history_current_observations": len(incoming),
            "history_prompt_turns": len(self._turns) + len(incoming),
            "history_active_visuals": sum(
                not turn.visual_folded for turn in self._turns + incoming
            ),
            "history_folded_turns": sum(turn.visual_folded for turn in self._turns + incoming),
            **physical_metadata,
        }
        provider_metadata.update(
            {
                "history_som_retained_blocks": sum(
                    without_observation_interactables(turn.observation.prompt)
                    != turn.observation.prompt
                    for turn in self._turns
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
        turn.action_history_index = len(action_history)
        turn.provider_refs.update(generation.metadata.get("completed_turn_refs") or {})

        # The turn joins the history only once the result is accepted (accept_turn). A repair
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
                        self.history_view(request_builder, executed_action_ledger, (turn,)),
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
        action_history: list[dict],
        timeout: float | None,
        request_builder: RequestBuilder,
        decision: int,
        cancel: threading.Event,
    ) -> RuntimeTurnResult:
        """One generation. ``cancel`` is its own event: once it is set, the generation stops at
        its next check (_stop_if_abandoned) and its retries end, the session encoder's included
        (they run in its retry_cancellation). Closing the runtime sets it. Its turn comes back in
        the result, for accept_turn once it is accepted."""
        with self._lock:
            self._running.add(cancel)
        try:
            with retry_cancellation(cancel):
                return self.execution.run(
                    self._run_turn(
                        tuple(observations),
                        request,
                        collector,
                        action_history=action_history,
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

    @staticmethod
    def _turn_refs(turns: Iterable[CanonicalTurn]) -> list[dict]:
        return [
            {
                "decision": turn.decision,
                "observation_id": turn.observation.observation_id,
                "tick": turn.observation.observation.tick_number,
                "action_history_index": turn.action_history_index,
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
        total = self._archived_action_count
        return {"archived_actions_before": total - archived_actions, "archived_actions_after": total}

    def _record_retention(self, retention: dict) -> None:
        """The policy archive and screenshot fold applied before this turn's request."""
        self._compaction_items = []
        if retention["archived_turn_refs"]:
            self._compaction_items.append("turns")
            self._history_changed(
                "archive",
                "policy",
                "turns",
                policy_values={"keep": self.policy.turns.keep, "block": self.policy.turns.block},
                turns=retention["archived_turn_refs"],
                ledger=self._ledger_change(retention["archived_actions"]),
            )
        if retention["folded_turn_refs"]:
            self._compaction_items.append("visuals")
            self._history_changed(
                "visual_fold",
                "policy",
                "visuals",
                policy_values={
                    "keep": self.policy.visuals.keep,
                    "block": self.policy.visuals.block,
                },
                turns=retention["folded_turn_refs"],
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

    async def _fits_with_every_action_omitted(
        self, request, request_builder, incoming, ledger
    ) -> bool:
        """Count the request as it would go with every archived action left out, leaving the
        history as it is."""
        block = self.ledger_block(ledger, omitted=self._archived_action_count)
        canonical = request_builder(tuple(self._turns + incoming), block)
        status = await self.driver.request_context_status(replace(request, canonical=canonical))
        return not status.get("pressure")

    def _omit_archived_actions(self, status) -> int:
        """Leave the next block of the earliest archived actions out of the ledger; returns how
        many, none once every one is out."""
        before = self._omitted_action_count
        count = min(self.policy.turns.block, self._archived_action_count - before)
        if count:
            self._omitted_action_count += count
            self._history_changed(
                "ledger_omission",
                "policy",
                "pressure_archive",
                policy_values={"block": self.policy.turns.block},
                ledger={
                    "archived_actions": self._archived_action_count,
                    "omitted_actions_before": before,
                    "omitted_actions_after": self._omitted_action_count,
                },
                request_context=status,
            )
        return count

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
            replayed_turns=len(self._turns) + len(incoming) if record.rebuilt else None,
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
            replayed_turns=len(self._turns),
            **record.fields,
        )
        self._states.record(*record.state)

    def _record_generation_prepared(self, incoming, physical_metadata, generation) -> None:
        turns = self._turns + incoming
        self._trace.capture(
            TraceEvent.GENERATION_PREPARED,
            {
                "generation": generation,
                "observation_id": incoming[-1].observation.observation_id,
                "persistent_session": self.capabilities.persistent_session,
                "history": {
                    "archived_actions_total": self._archived_action_count,
                    # As history_omitted_actions_total in the step's provider metadata.
                    **(
                        {"omitted_actions_total": self._omitted_action_count}
                        if self._omitted_action_count
                        else {}
                    ),
                    "detailed_turns": len(self._turns),
                    "current_observations": len(incoming),
                    "prompt_turns": len(turns),
                    "active_visuals": sum(not turn.visual_folded for turn in turns),
                    "folded_turns": sum(turn.visual_folded for turn in turns),
                    "som_retained_blocks": sum(
                        without_observation_interactables(turn.observation.prompt)
                        != turn.observation.prompt
                        for turn in self._turns
                    ),
                    "som_current_blocks": sum(
                        without_observation_interactables(turn.observation.prompt)
                        != turn.observation.prompt
                        for turn in incoming
                    ),
                },
                "request_context": physical_metadata or None,
            },
        )
