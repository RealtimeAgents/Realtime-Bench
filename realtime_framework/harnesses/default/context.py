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


class CatchupBuffer:
    """Thread-safe single-observation slot: prefill replaces it and generate consumes it."""

    def __init__(self):
        self._pending = None
        self._lock = threading.Lock()
        self._last = None

    def enqueue(self, observation):
        with self._lock:
            self._pending = observation
            self._last = observation

    def drain(self) -> list:
        """Return the pending observation, falling back to the latest known frame."""
        with self._lock:
            observation, self._pending = self._pending, None
            observation = observation if observation is not None else self._last
            return [] if observation is None else [observation]


def action_result_note(status: str, detail: str = "") -> str:
    """Return model feedback when a proposed action was not applied."""
    note = ACTION_RESULT_NOTES.get(status, "")
    if note and detail:
        note += f" ({detail})"
    return note


# Runtime
"""Common session and generation state machine for every catalog model."""

import asyncio as asyncio
import concurrent.futures
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
from realtime_framework.entity.records import TraceEvent
from realtime_framework.entity.request import CanonicalRequest
from realtime_framework.entity.request import ContentBlock
from realtime_framework.harnesses.default.request import ledger_block
from realtime_framework.sandbox.observation.som import without_observation_interactables

# Builds the canonical request from the turns to send and the archived-action ledger block.
RequestBuilder = Callable[[tuple[CanonicalTurn, ...], ContentBlock | None], CanonicalRequest]


def session_frame_text(observation) -> str:
    """The prompt the live drivers send for one ProviderObservation, after its pending feedback."""
    prefix = observation.feedback_prefix
    return f"{prefix}\n{observation.prompt}" if prefix else observation.prompt


class RealtimeRuntime:
    """Own the protocol-neutral lifecycle around one provider driver.

    The synchronous harness may call the agent from disposable worker threads.  One private asyncio
    loop owns all driver operations, so async WebSocket SDKs and synchronous transports bridged by a
    driver follow the same generation state machine.
    """

    def __init__(
        self, driver: RealtimeDriver, policy: PathPolicy, *, trace, clock: Clock | None = None
    ):
        if not isinstance(driver, RealtimeDriver):
            raise TypeError("realtime driver does not implement the required protocol")
        self.driver = driver
        self.capabilities = driver.capabilities
        self.policy = policy
        self._trace = trace
        # The operational clock; it makes the loop, so the drivers' waits run on its time.
        self.clock = clock or SystemClock()
        # What the agent trace last recorded about the provider-side context: exact or unknown.
        self._session_state = "exact"
        # The context-policy items behind this turn's history_block_compaction sync.
        self._compaction_items: list[str] = []
        self._state = RuntimeState.NEW
        self._state_lock = threading.RLock()
        self._active_generation: GenerationHandle | None = None
        self._connected = False
        self._seed: SessionSeed | None = None
        self._turns: list[CanonicalTurn] = []
        self._archived_action_count = 0
        self._ledger_wall_ns: int | None = None  # when the archived-action ledger last grew
        self._buffer = CatchupBuffer()
        self._cleanup_future: concurrent.futures.Future | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @property
    def state(self) -> RuntimeState:
        with self._state_lock:
            return self._state

    @property
    def turns(self) -> tuple[CanonicalTurn, ...]:
        return tuple(self._turns)

    @property
    def archived_action_count(self) -> int:
        return self._archived_action_count

    def _set_state(self, state: RuntimeState) -> None:
        with self._state_lock:
            if self._state == RuntimeState.CLOSED and state != RuntimeState.CLOSED:
                return
            self._state = state

    def _start_loop(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            return
        loop = self.clock.new_event_loop()
        ready = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=run,
            name=f"realtime-runtime-{self.capabilities.protocol_family}",
            daemon=True,
        )
        thread.start()
        ready.wait()
        self._loop = loop
        self._thread = thread

    def _submit(self, coroutine, timeout: float | None = None):
        loop = self._loop
        if loop is None or loop.is_closed():
            coroutine.close()
            raise RuntimeError("realtime runtime event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        if not self.clock.wait(future, timeout):
            if not future.done():
                future.cancel()
            raise TimeoutError(f"realtime runtime operation exceeded {timeout:g}s") from None
        return future.result()

    def start(self, seed: SessionSeed, initial_observation=None, *, timeout: float = 60.0) -> None:
        self._start_loop()
        self._seed = seed
        if initial_observation is not None:
            self._buffer.enqueue(initial_observation)
        self._set_state(RuntimeState.CONNECTING)
        try:
            self._submit(
                self.driver.connect(ConnectRequest(seed=seed)),
                timeout=timeout,
            )
        except Exception:
            self._set_state(RuntimeState.BROKEN)
            raise
        self._connected = True
        self._set_state(RuntimeState.READY)

    def enqueue(self, observation) -> None:
        self._buffer.enqueue(observation)

    def discard_pending_observations(self) -> None:
        self._buffer = CatchupBuffer()

    def drain_observations(self) -> list:
        return self._buffer.drain()

    def _abandoned(self, cancel: threading.Event) -> bool:
        """Whether the generation of this cancel event was aborted, or the runtime closed."""
        with self._state_lock:
            return cancel.is_set() or self._state == RuntimeState.CLOSED

    def _stop_if_abandoned(self, cancel: threading.Event, before: str) -> None:
        """Stop an aborted generation. _run_turn checks before it applies the retention policy,
        before it sends its observation and before it starts its reply."""
        if self._abandoned(cancel):
            raise RetryCancelled(f"generation abandoned before {before}")

    def record_execution_feedback(self, feedback: str, decision: int | None = None) -> None:
        """Attach an action's execution feedback to the turn of its decision; with no decision,
        to the newest turn, as the legacy loop reports."""
        turns = [turn for turn in self._turns if decision is None or turn.decision == decision]
        if turns:
            turns[-1].execution_feedback = feedback
            turns[-1].feedback_wall_ns = self.clock.time_ns()

    def accept_turn(self, turn: CanonicalTurn) -> None:
        """Write an accepted generation's turn into the retained history. The write runs on the
        loop, where a generation that was abandoned but still runs reads the history."""

        async def write() -> None:
            self._turns.append(turn)

        self._submit(write())

    def _action_ledger(self, history: list[dict]) -> str:
        return executed_action_ledger(history, limit=self._archived_action_count)

    def ledger_block(self, ledger: str) -> ContentBlock | None:
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

    async def _synchronize(
        self,
        *,
        reason: str,
        ledger: str,
        request_builder: RequestBuilder,
        current_generation: GenerationHandle | None = None,
        force_rebuild: bool = False,
        removed_turns: tuple[CanonicalTurn, ...] = (),
        incoming: tuple[CanonicalTurn, ...] = (),
    ) -> HistorySyncResult:
        seed = self._seed
        if seed is None:
            raise RuntimeError("realtime runtime has no session seed")
        self._set_state(RuntimeState.SYNCHRONIZING)
        try:
            result = await self.driver.sync_history(
                HistorySyncRequest(
                    seed=seed,
                    reason=reason,
                    history=self.history_view(request_builder, ledger, incoming),
                    current_generation=current_generation,
                    force_rebuild=force_rebuild,
                )
            )
        except Exception as exc:
            self._connected = False
            self._set_state(RuntimeState.BROKEN)
            self._record_sync(reason, force_rebuild, removed_turns, incoming, error=exc)
            raise
        self._record_sync(reason, force_rebuild, removed_turns, incoming, result=result)
        if result.history_state == ProviderHistoryState.EXACT:
            self._connected = True
            self._set_state(RuntimeState.READY)
        elif result.disposition in {SyncDisposition.UNSUPPORTED, SyncDisposition.BROKEN}:
            self._connected = False
            self._set_state(RuntimeState.BROKEN)
        else:
            self._set_state(RuntimeState.DIRTY)
        return result

    async def _wait_for_cleanup(self) -> None:
        # Background wall-clock cleanup runs on this same loop.  The task reference is stored by the
        # coroutine, whereas the cross-thread Future is only used by close/diagnostics.
        task = getattr(self, "_cleanup_task", None)
        if task is not None:
            try:
                await task
            except Exception:  # noqa: BLE001, S110 - cleanup failure forces later recovery
                pass
            self._cleanup_task = None
        future = self._cleanup_future
        if future is not None:
            try:
                await asyncio.wrap_future(future)
            except Exception:  # noqa: BLE001, S110 - cleanup failure forces later recovery
                pass
            if self._cleanup_future is future:
                self._cleanup_future = None

    async def _ensure_ready(self, ledger: str, request_builder: RequestBuilder) -> None:
        await self._wait_for_cleanup()
        if self._connected and self.state == RuntimeState.READY:
            return
        if self._connected:
            result = await self._synchronize(
                reason="runtime_recovery",
                ledger=ledger,
                request_builder=request_builder,
                force_rebuild=True,
            )
            if result.history_state != ProviderHistoryState.EXACT:
                raise RuntimeError(f"provider history recovery failed: {result.detail}")
            return
        seed = self._seed
        if seed is None:
            raise RuntimeError("realtime runtime has not been started")
        self._set_state(RuntimeState.CONNECTING)
        await self.driver.connect(
            ConnectRequest(seed=seed, history=self.history_view(request_builder, ledger))
        )
        self._record_reconnect()
        self._connected = True
        self._set_state(RuntimeState.READY)

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
        executed_action_ledger = self._action_ledger(action_history)
        await self._ensure_ready(executed_action_ledger, request_builder)
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
            sync = await self._synchronize(
                reason="history_block_compaction",
                ledger=executed_action_ledger,
                request_builder=request_builder,
                force_rebuild=(
                    self.capabilities.persistent_session
                    and not self.capabilities.can_compact_history_in_place
                ),
                removed_turns=removed_turns,
            )
            if sync.history_state != ProviderHistoryState.EXACT:
                raise RuntimeError(f"history synchronization failed: {sync.detail}")
        physical_metadata = {}
        executed_action_ledger = self._action_ledger(action_history)

        reason = (
            self.driver.connection_renewal_reason()
            if self.capabilities.connection_renewal
            else None
        )
        if reason:
            sync = await self._synchronize(
                reason=reason, ledger=executed_action_ledger, request_builder=request_builder
            )
            if sync.history_state != ProviderHistoryState.EXACT:
                raise RuntimeError(f"connection renewal failed: {sync.detail}")

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
            if self.capabilities.persistent_session:
                raise RuntimeError(
                    "Request exceeds physical context; driver must implement native context compaction"
                )
            if not self._turns:
                raise RuntimeError(
                    "Task instructions, execution ledger and current observation exceed context capacity; use a larger endpoint window"
                )
            archived = self._turns[: self.policy.turns.block]
            del self._turns[: len(archived)]
            archived_actions = self._record_archived_turns(archived)
            self._record_pressure_archive(archived, archived_actions, status)
            compactions += 1
            archived_count += len(archived)

        self._stop_if_abandoned(cancel, "its reply")
        self._record_generation_prepared(incoming, physical_metadata, request.generation)
        self._set_state(RuntimeState.GENERATING)
        provider_call_started_monotonic_ns = self.clock.monotonic_ns()
        provider_call_started_wall_time_ns = self.clock.time_ns()
        generation = await self.driver.start_generation(request)
        stream_started_monotonic_ns = self.clock.monotonic_ns()
        generation.metadata.setdefault("input_refs", input_refs)
        with self._state_lock:
            self._active_generation = generation
        started = self.clock.monotonic()
        terminal_history = ProviderHistoryState.UNKNOWN
        provider_metadata: dict = {
            "provider_call_started_monotonic_ns": provider_call_started_monotonic_ns,
            "provider_call_started_wall_time_ns": provider_call_started_wall_time_ns,
            "provider_stream_started_monotonic_ns": stream_started_monotonic_ns,
            "history_folded_visuals": retention["folded_visuals"],
            "history_archived_turns": retention["archived_turns"],
            "history_archived_actions": retention["archived_actions"],
            "history_archived_actions_total": self._archived_action_count,
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
                    without_observation_interactables(session_frame_text(turn.observation))
                    != session_frame_text(turn.observation)
                    for turn in self._turns
                ),
                "history_som_current_blocks": sum(
                    without_observation_interactables(session_frame_text(turn.observation))
                    != session_frame_text(turn.observation)
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
            with self._state_lock:
                if self._active_generation is generation:
                    self._active_generation = None

        cancelled = self._abandoned(cancel)
        if terminal_history != ProviderHistoryState.EXACT:
            self._record_session_state(
                "unknown", "generation_terminal_state", terminal_history, request.generation_id
            )
        output = collector.finish()
        turn = incoming[-1]
        turn.output = output.text
        turn.output_wall_ns = self.clock.time_ns()
        turn.output_reasoning = output.reasoning_text
        turn.action_history_index = len(action_history)
        turn.provider_refs.update(generation.metadata.get("completed_turn_refs") or {})

        # The turn joins the history only once the result is accepted (accept_turn). A sync after
        # the generation already includes it: a result that is discarded was aborted first, and
        # the abort's close makes the next generation rebuild from the accepted history.
        needs_sync = not cancelled and (
            terminal_history != ProviderHistoryState.EXACT or reconnect_required
        )
        if needs_sync:
            sync = await self._synchronize(
                reason=provider_metadata.get("reconnect_reason", "provider_history_changed"),
                ledger=executed_action_ledger,
                request_builder=request_builder,
                current_generation=generation,
                force_rebuild=False,
                incoming=(turn,),
            )
        elif cancelled:
            # abort() already dirtied and scheduled cleanup for this generation. A newer turn may
            # have recovered by the time this abandoned worker drains its last event, so it must
            # not overwrite the newer lifecycle state.
            sync = HistorySyncResult(
                SyncDisposition.REBUILT,
                ProviderHistoryState.UNKNOWN,
                detail="generation superseded by harness timeout",
            )
        else:
            self._set_state(RuntimeState.READY)

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
        its next check (_stop_if_abandoned) and its retries end. Its turn comes back in the
        result, for accept_turn once it is accepted."""
        try:
            return self._submit(
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
            if not self._abandoned(cancel) and self.state not in {
                RuntimeState.BROKEN,
                RuntimeState.CLOSED,
            }:
                self._set_state(RuntimeState.DIRTY)
            raise

    async def _abort_active(self, generation: GenerationHandle | None) -> None:
        try:
            if generation is not None:
                await self.driver.interrupt(generation)
        except Exception:  # noqa: BLE001, S110 - abort remains best-effort
            pass
        try:
            await self.driver.close()
        except Exception:  # noqa: BLE001, S110 - close below still marks the session dirty
            pass
        self._connected = False
        if self.state != RuntimeState.CLOSED:
            self._set_state(RuntimeState.DIRTY)

    def abort(self) -> None:
        """Interrupt whichever generation is generating, close the session and mark it dirty:
        the next generation reconnects. The caller has already set the cancel event of each
        generation it aborts, which stops that generation's requests."""
        with self._state_lock:
            generation = self._active_generation
            if self._state != RuntimeState.CLOSED:
                self._state = RuntimeState.DIRTY
        loop = self._loop
        if loop is not None and not loop.is_closed():
            self._cleanup_future = asyncio.run_coroutine_threadsafe(
                self._abort_active(generation),
                loop,
            )

    def close(self) -> None:
        with self._state_lock:
            if self._state == RuntimeState.CLOSED:
                return
            self._state = RuntimeState.CLOSED
        loop = self._loop
        if loop is not None and not loop.is_closed():
            cleanup = self._cleanup_future
            if cleanup is not None:
                self.clock.wait(cleanup, 5.0)
            try:
                self._submit(self._wait_for_cleanup(), timeout=5.0)
            except Exception:  # noqa: BLE001, S110 - final close retries independently
                pass
            try:
                self._submit(self.driver.close(), timeout=5.0)
            except Exception:  # noqa: BLE001, S110 - destructor-style best effort
                pass
            loop.call_soon_threadsafe(loop.stop)
            thread = self._thread
            if thread is not None:
                thread.join(timeout=5.0)
            if not loop.is_running():
                loop.close()
        self._connected = False

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

    def _replay_items(self) -> list[str]:
        # A replayed live session re-sends past tool calls in the policy's format.
        return ["tool_call_replay"] if self.policy.tool_call_replay is not None else []

    def _record_sync(
        self, reason, force_rebuild, removed_turns, incoming=(), *, result=None, error=None
    ):
        """One HistoryChanged per sync call, of the kind the driver's result reports.

        The fallback kinds come from provider_metadata keys the drivers already return.
        """
        metadata = result.provider_metadata if result is not None else {}
        trigger = {"history_block_compaction": "policy", "runtime_recovery": "recovery"}.get(
            reason, "provider"
        )
        rebuilt = result is not None and (
            result.disposition == SyncDisposition.REBUILT
            or bool(metadata.get("history_session_reconnected"))
        )
        if metadata.get("session_resumed"):
            kind, rebuilt = "session_resumed", False
        elif metadata.get("resume_error"):
            kind, trigger = "resume_fallback", "fallback"
        elif metadata.get("history_in_place_edit_fallback"):
            kind, trigger = "in_place_edit_fallback", "fallback"
        else:
            kind = "rebuild" if rebuilt else "sync"
        policy_item = list(self._compaction_items) if reason == "history_block_compaction" else []
        if rebuilt:
            policy_item += self._replay_items()
        self._history_changed(
            kind,
            trigger,
            policy_item or None,
            turns=self._turn_refs(removed_turns),
            replayed_turns=len(self._turns) + len(incoming) if rebuilt else None,
            sync={
                "reason": reason,
                "force_rebuild": force_rebuild,
                "disposition": result.disposition.value if result is not None else None,
                "history_state": result.history_state.value if result is not None else None,
                "detail": result.detail if result is not None else None,
                "provider_metadata": metadata,
            },
            error=repr(error)[:500] if error is not None else None,
        )
        if result is None:
            self._record_session_state("unknown", f"sync_error:{reason}", None)
        elif kind == "session_resumed":
            # Exact for the driver, but restored from the resume handle, not sent on this
            # connection.
            self._record_session_state("unknown", "session_resumed", result.history_state)
        elif result.history_state == ProviderHistoryState.EXACT:
            self._record_session_state("exact", f"sync:{reason}", result.history_state)
        else:
            self._record_session_state("unknown", f"sync:{reason}", result.history_state)

    def _record_reconnect(self) -> None:
        """_ensure_ready replayed the canonical turns on a new session after an abort or failure."""
        if not self.capabilities.persistent_session:
            return  # a stateless connect sends nothing; the next request carries the history
        self._history_changed(
            "rebuild",
            "recovery",
            self._replay_items() or None,
            turns=[],
            replayed_turns=len(self._turns),
            sync={"reason": "reconnect"},
        )
        self._record_session_state("exact", "reconnect", ProviderHistoryState.EXACT)

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

    def _record_session_state(self, state, cause, provider_state, generation_id=None) -> None:
        """Record when the provider-side context becomes unknown, or exact again."""
        if state == self._session_state:
            return
        self._session_state = state
        self._trace.capture(
            TraceEvent.SESSION_STATE_CHANGED,
            {
                "protocol_family": self.capabilities.protocol_family,
                "state": state,
                "cause": cause,
                "provider_history_state": provider_state.value if provider_state else None,
                "generation_id": generation_id,
            },
        )
