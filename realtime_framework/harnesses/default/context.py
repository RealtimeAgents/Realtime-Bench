from __future__ import annotations
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER

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
import time as time
from copy import deepcopy as deepcopy
from collections.abc import Callable as Callable
from collections.abc import Iterable as Iterable
from dataclasses import replace as replace

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
from realtime_framework.entity.model import RuntimeState
from realtime_framework.entity.model import RuntimeTurnResult
from realtime_framework.entity.model import SessionSeed
from realtime_framework.entity.model import SyncDisposition
from realtime_framework.sandbox.observation.som import without_observation_interactables


class RealtimeRuntime:
    """Own the protocol-neutral lifecycle around one provider driver.

    The synchronous harness may call the agent from disposable worker threads.  One private asyncio
    loop owns all driver operations, so async WebSocket SDKs and synchronous transports bridged by a
    driver follow the same generation state machine.
    """

    def __init__(self, driver: RealtimeDriver, policy: PathPolicy):
        if not isinstance(driver, RealtimeDriver):
            raise TypeError("realtime driver does not implement the required protocol")
        self.driver = driver
        self.capabilities = driver.capabilities
        self.policy = policy
        self._state = RuntimeState.NEW
        self._state_lock = threading.RLock()
        self._epoch = 0
        self._active_generation: GenerationHandle | None = None
        self._connected = False
        self._seed: SessionSeed | None = None
        self._turns: list[CanonicalTurn] = []
        self._archived_action_count = 0
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
        loop = asyncio.new_event_loop()
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
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            if not future.done():
                future.cancel()
            raise TimeoutError(f"realtime runtime operation exceeded {timeout:g}s") from None

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

    def generation_epoch(self) -> int:
        with self._state_lock:
            return self._epoch

    def is_current(self, epoch: int) -> bool:
        with self._state_lock:
            return epoch == self._epoch and self._state != RuntimeState.CLOSED

    def record_execution_feedback(self, feedback: str) -> None:
        if self._turns:
            self._turns[-1].execution_feedback = feedback

    def _action_ledger(self, history: list[dict]) -> str:
        return executed_action_ledger(history, limit=self._archived_action_count)

    def _record_archived_turns(self, turns: Iterable[CanonicalTurn]) -> int:
        indexes = [
            turn.action_history_index for turn in turns if turn.action_history_index is not None
        ]
        before = self._archived_action_count
        if indexes:
            self._archived_action_count = max(before, max(indexes) + 1)
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
            },
        )

    async def _synchronize(
        self,
        *,
        reason: str,
        ledger: str,
        current_generation: GenerationHandle | None = None,
        force_rebuild: bool = False,
        removed_turns: tuple[CanonicalTurn, ...] = (),
    ) -> HistorySyncResult:
        seed = self._seed
        if seed is None:
            raise RuntimeError("realtime runtime has no session seed")
        self._set_state(RuntimeState.SYNCHRONIZING)
        try:
            result = await self.driver.sync_history(
                HistorySyncRequest(
                    seed=seed,
                    turns=tuple(self._turns),
                    executed_action_ledger=ledger,
                    reason=reason,
                    current_generation=current_generation,
                    force_rebuild=force_rebuild,
                    removed_turns=removed_turns,
                )
            )
        except Exception:
            self._connected = False
            self._set_state(RuntimeState.BROKEN)
            raise
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

    async def _ensure_ready(self, ledger: str) -> None:
        await self._wait_for_cleanup()
        if self._connected and self.state == RuntimeState.READY:
            return
        if self._connected:
            result = await self._synchronize(
                reason="runtime_recovery",
                ledger=ledger,
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
            ConnectRequest(
                seed=seed,
                turns=tuple(self._turns),
                executed_action_ledger=ledger,
            )
        )
        self._connected = True
        self._set_state(RuntimeState.READY)

    async def _run_turn(
        self,
        observations: tuple[ProviderObservation, ...],
        request: GenerationRequest,
        collector: OutputCollector,
        *,
        action_history: list[dict],
        epoch: int,
        message_builder: Callable[[tuple[CanonicalTurn, ...]], tuple[dict, ...]] | None,
    ) -> RuntimeTurnResult:
        if len(observations) != 1:
            raise ValueError("each generation accepts exactly one observation")
        executed_action_ledger = self._action_ledger(action_history)
        await self._ensure_ready(executed_action_ledger)
        incoming = [CanonicalTurn(observation=value) for value in observations]
        retained, incoming, removed_turns, retention = self._retained(self._turns, incoming)
        self._turns[:] = retained
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

        renewal_reason = getattr(self.driver, "connection_renewal_reason", None)
        reason = renewal_reason() if callable(renewal_reason) else None
        if reason:
            sync = await self._synchronize(reason=reason, ledger=executed_action_ledger)
            if sync.history_state != ProviderHistoryState.EXACT:
                raise RuntimeError(f"connection renewal failed: {sync.detail}")

        if not incoming:
            raise RuntimeError("no current observation")
        input_refs = [await self.driver.send_observation(turn.observation) for turn in incoming]
        input_refs = [refs or {} for refs in input_refs]
        for turn, refs in zip(incoming, input_refs):
            turn.provider_refs.update(refs or {})

        if message_builder is not None:
            compactions = 0
            archived_count = 0
            while True:
                messages = list(message_builder(tuple(self._turns + incoming)))
                executed_action_ledger = self._action_ledger(action_history)
                if executed_action_ledger:
                    messages.insert(
                        1,
                        {
                            "role": "user",
                            "content": ARCHIVED_ACTION_LEDGER_HEADER
                            + "\n"
                            + executed_action_ledger,
                        },
                    )
                request = replace(request, messages=tuple(messages))
                if not self.policy.pressure_archive:
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
                self._record_archived_turns(archived)
                compactions += 1
                archived_count += len(archived)

        self._set_state(RuntimeState.GENERATING)
        provider_call_started_monotonic_ns = time.monotonic_ns()
        provider_call_started_wall_time_ns = time.time_ns()
        generation = await self.driver.start_generation(request)
        stream_started_monotonic_ns = time.monotonic_ns()
        generation.metadata.setdefault("input_refs", input_refs)
        with self._state_lock:
            self._active_generation = generation
        started = time.monotonic()
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
            with self._state_lock:
                if self._active_generation is generation:
                    self._active_generation = None

        stale = not self.is_current(epoch)
        output = collector.finish()
        if not stale:
            incoming[-1].output = output.text
            if output.reasoning_text and getattr(self.driver, "preserve_reasoning_history", False):
                incoming[-1].provider_refs["reasoning_content"] = output.reasoning_text
            incoming[-1].action_history_index = len(action_history)
            incoming[-1].provider_refs.update(generation.metadata.get("completed_turn_refs") or {})
            self._turns.extend(incoming)

        needs_sync = not stale and (
            terminal_history != ProviderHistoryState.EXACT or reconnect_required
        )
        if needs_sync:
            sync = await self._synchronize(
                reason=provider_metadata.get("reconnect_reason", "provider_history_changed"),
                ledger=executed_action_ledger,
                current_generation=generation,
                force_rebuild=False,
            )
        elif stale:
            # abort() already dirtied and scheduled cleanup for this epoch. A newer turn may have
            # recovered by the time this abandoned worker drains its last event, so it must not
            # overwrite the newer lifecycle state.
            sync = HistorySyncResult(
                SyncDisposition.REBUILT,
                ProviderHistoryState.UNKNOWN,
                detail="generation superseded by harness timeout",
            )
        else:
            self._set_state(RuntimeState.READY)

        finished_monotonic_ns = time.monotonic_ns()
        return RuntimeTurnResult(
            output=output,
            generation_id=request.generation_id,
            history_sync=sync,
            terminal_history_state=terminal_history,
            interrupted=False,
            stale=stale,
            latency_s=time.monotonic() - started,
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
        message_builder: Callable[[tuple[CanonicalTurn, ...]], tuple[dict, ...]] | None = None,
    ) -> RuntimeTurnResult:
        epoch = self.generation_epoch()
        try:
            return self._submit(
                self._run_turn(
                    tuple(observations),
                    request,
                    collector,
                    action_history=action_history,
                    epoch=epoch,
                    message_builder=message_builder,
                ),
                timeout=timeout,
            )
        except Exception:
            if self.is_current(epoch) and self.state not in {
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
        with self._state_lock:
            self._epoch += 1
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
            self._epoch += 1
            self._state = RuntimeState.CLOSED
        loop = self._loop
        if loop is not None and not loop.is_closed():
            cleanup = self._cleanup_future
            if cleanup is not None:
                try:
                    cleanup.result(timeout=5.0)
                except Exception:  # noqa: BLE001, S110 - final close retries independently
                    pass
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
