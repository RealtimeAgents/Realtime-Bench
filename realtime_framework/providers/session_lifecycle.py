"""What both live session encoders do around their connection, whatever the protocol.

A session encoder keeps one provider session open across generations. Whether that session can
take the next generation, and the cleanup an abort leaves behind for the next use to wait for, do
not depend on the protocol: SessionLifecycle keeps both, and readies, renews, repairs or aborts the
session through the encoder's own connection operations. It also records the encoder's waits on
the clock, and takes the ones the encoder makes of its own accord. StatelessSession presents a
stateless driver the same way, so the runtime drives every driver alike. How a session operation
reads in an agent's trace, and what it leaves of the provider's context, are session_record's and
SessionStates', which the turn-based runtime and the native-streaming agent share.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
from collections.abc import Callable
from dataclasses import dataclass

from realtime_framework.entity.context_policy import PathPolicy
from realtime_framework.entity.events import ProviderHistoryState
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationHandle,
    HistorySyncRequest,
    HistorySyncResult,
    RetryCancelled,
    RuntimeState,
    SessionSeed,
    SyncDisposition,
    retry_cancel_event,
)
from realtime_framework.entity.records import TraceEvent
from realtime_framework.entity.request import CanonicalRequest

# How long ensure_ready and close() wait for an abort's cleanup before they cancel it and close
# the connection all the same (decision 14).
_CLEANUP_WAIT_S = 10.0

_PREPARING: contextvars.ContextVar[Callable[[], None]] = contextvars.ContextVar(
    "session_preparing", default=lambda: None
)


@contextlib.contextmanager
def session_preparing(observe: Callable[[], None]):
    """Have ensure_ready and renew, run in this context, call ``observe()`` as each starts
    preparing the session: waiting for an abort's cleanup, opening, rebuilding or renewing it. One
    that leaves a ready session as it is calls nothing. A native-streaming reply readying the
    session tells the kernel the session is being prepared from then on (decision 13). Like
    retry_cancellation, the ContextVar follows the call onto the execution thread."""
    token = _PREPARING.set(observe)
    try:
        yield
    finally:
        _PREPARING.reset(token)


class RenewalRequired(RuntimeError):
    """The provider asked for a new connection (connection_renewal_reason), and the session
    takes no input until renew() made it: a frame or a reply refused for that is held back, not
    lost to a broken session, and nothing of it was written or started. A reply in flight goes
    on to its end on the old connection first; renew() after it resumes the session where the
    encoder can, and what was refused can be put in again."""


class SessionLifecycle:
    """A live encoder's session: connected or not, dirty or not, and the last abort's cleanup.

    The session is connected once connect() opened it or a sync left it exact, and no longer once
    close() or abort() closed it or a sync raised an Exception or broke it; the last two leave it
    broken until it is opened again or aborted. A connected session is dirty when the provider
    may hold what the canonical history does not: a generation on it failed or was aborted
    (mark_dirty), a sync left it neither exact nor broken, or a sync was cut short
    (RetryCancelled, a cancelled task).
    A connection that an operation closes on its way (the one a connect replaces, one past its
    goAway deadline) changes neither.

    The runtime readies the session before each generation (ensure_ready), brings it to the
    retained history after the history changed (synchronize), renews the connection the provider
    asked to (renew), repairs it after a generation (repair), and aborts it (abort). One of these
    operations runs at a time. Each runs for one generation, the one whose retry_cancellation it
    runs in: once that generation is abandoned it does not start, and what it opened or rebuilt
    meanwhile is closed again, because the abort that came during it closed what was open before.
    ``record(sync, result=None, error=None)`` sees what each did before it returns or raises: the
    sync's HistorySyncRequest with its result or the exception it raised, or None for a reopened
    session.

    Every timed wait the encoder takes on the clock is written to the trace before it starts
    (_record_wait), once: what it waits for, how long, and until which time on the operational
    clock. A wait that would last no time, for a cleanup already done or for a generation already
    abandoned is not recorded, and _wait does not take it.
    The encoder's own waits (_wait) end early once their generation is abandoned. A bound on an
    operation (a goAway deadline, a close) ends with the operation, and an abort's close ends the
    operation.

    Whether the session can take the next generation is its state. What became of each content
    the encoder wrote, sent, confirmed or refused by the server, or unknown, its session core
    records, and the encoder writes each change to the trace (_record_delivery). What the
    provider keeps beside them is not known: the capability server_context says so.

    The encoder provides the connection operations: ``_connect(request)``,
    ``_sync_history(request)``, ``_close_current()``, which closes the open connection if there
    is one, and ``interrupt(generation)``; ``capabilities``, with
    ``connection_renewal_reason()`` where they declare connection_renewal; ``clock``, the
    operational clock; and ``_trace``, which bind_trace sets. ``interrupt`` and
    ``_close_current`` must each end within a bound, whatever a write in progress on the
    connection does: the close fails a write stuck in it, and the interrupt, which the close
    follows, does not wait for one. That is what ends an abort's cleanup before ensure_ready and
    close stop waiting for it.
    """

    _connected = False
    _dirty = False
    _broken = False
    _cleanup: asyncio.Task | None = None
    _operation_lock: asyncio.Lock | None = None

    @property
    def state(self) -> RuntimeState:
        """READY when the session can take the next generation as it is; BROKEN when the last
        sync broke it; otherwise DIRTY, to be rebuilt or reopened first."""
        if self._broken:
            return RuntimeState.BROKEN
        if self._connected and not self._dirty:
            return RuntimeState.READY
        return RuntimeState.DIRTY

    async def connect(self, request: ConnectRequest) -> None:
        await self._connect(request)
        self._connected, self._dirty, self._broken = True, False, False

    async def sync_history(self, request: HistorySyncRequest) -> HistorySyncResult:
        # Dirty until the sync reports: one cut short may have closed the connection already.
        self._dirty, self._broken = True, False
        try:
            result = await self._sync_history(request)
        except Exception:
            self._connected, self._broken = False, True
            raise
        if result.history_state == ProviderHistoryState.EXACT:
            self._connected, self._dirty = True, False
        elif result.disposition in {SyncDisposition.UNSUPPORTED, SyncDisposition.BROKEN}:
            self._connected, self._broken = False, True
        else:
            self._dirty = True
        return result

    async def close(self) -> None:
        """Close the connection once the last abort's cleanup is done, or has run for
        _CLEANUP_WAIT_S, or the caller stops waiting, whichever comes first."""
        try:
            await self._settle()
        finally:
            self._connected = False
            await self._close_current()

    def mark_dirty(self) -> None:
        """A generation on the session failed, or is being aborted: the next ensure_ready
        rebuilds the session if it is still connected."""
        self._dirty = True

    async def ensure_ready(
        self,
        seed: SessionSeed,
        history: CanonicalRequest,
        record: Callable[..., None],
    ) -> HistorySyncResult | None:
        """Make the session ready for the next generation, once the last abort's cleanup is done,
        or has run for _CLEANUP_WAIT_S, as close() waits for it.

        A ready session is left as it is. A closed one is reopened with ``history``, as is one
        whose abort's cleanup was given up on: the connection that cleanup was to close is
        closed first. A dirty one is rebuilt from ``history`` by a runtime_recovery sync, whose
        result is returned; a recovery that does not leave it exact raises. A failed reopen
        raises unrecorded.
        """
        if self._cleanup is not None and not self._cleanup.done():
            self._preparing()
        if await self._settle():
            await self._interrupt_and_close(None)
        async with self._session_lock():
            if self._connected and not self._dirty:
                return None
            self._preparing()
            self._stop_if_abandoned("its session")
            if self._connected:
                result = await self._sync(
                    HistorySyncRequest(seed, "runtime_recovery", history, force_rebuild=True),
                    record,
                )
                if result.history_state != ProviderHistoryState.EXACT:
                    raise RuntimeError(f"provider history recovery failed: {result.detail}")
                return result
            await self.connect(ConnectRequest(seed=seed, history=history))
            record(None)
            await self._release_if_abandoned()
            return None

    async def synchronize(
        self, request: HistorySyncRequest, record: Callable[..., None]
    ) -> HistorySyncResult:
        """Bring the session to the retained history after the history changed (a fold or an
        archive); a sync that does not leave it exact raises."""
        async with self._session_lock():
            result = await self._sync(request, record)
        if result.history_state != ProviderHistoryState.EXACT:
            raise RuntimeError(f"history synchronization failed: {result.detail}")
        return result

    async def renew(
        self, seed: SessionSeed, history: CanonicalRequest, record: Callable[..., None]
    ) -> HistorySyncResult | None:
        """Renew the connection once the provider asked for it (connection_renewal_reason), from
        ``history``; a renewal that does not leave the session exact raises. None when no renewal
        is due. The caller readies the session first (ensure_ready): one closed or dirty since,
        which may hold what the history does not, is rebuilt then, and owes no renewal."""
        reason = (
            self.connection_renewal_reason() if self.capabilities.connection_renewal else None
        )
        if not reason:
            return None
        self._preparing()
        async with self._session_lock():
            result = await self._sync(HistorySyncRequest(seed, reason, history), record)
        if result.history_state != ProviderHistoryState.EXACT:
            raise RuntimeError(f"connection renewal failed: {result.detail}")
        return result

    async def repair(
        self,
        request: HistorySyncRequest,
        terminal_history: ProviderHistoryState,
        reconnect_required: bool,
        record: Callable[..., None],
    ) -> HistorySyncResult | None:
        """After a generation that came back: when its end left the provider's history other than
        exact, or it asked to reconnect, sync the session to ``request.history``, which already
        holds the generation's turn. None when there is nothing to repair."""
        if terminal_history == ProviderHistoryState.EXACT and not reconnect_required:
            return None
        async with self._session_lock():
            return await self._sync(request, record)

    async def abort(self, generation: GenerationHandle | None) -> None:
        """Interrupt ``generation`` (None when none had started), then close the connection, as
        far as either works, once the cleanup of the abort before is done. The next ensure_ready
        and close wait for this to finish, for at most _CLEANUP_WAIT_S; a later abort takes its
        place as the one they wait for."""
        self._cleanup = asyncio.ensure_future(self._clean_up(self._cleanup, generation))
        await asyncio.wait({self._cleanup})

    async def _clean_up(
        self, before: asyncio.Task | None, generation: GenerationHandle | None
    ) -> None:
        """An abort's cleanup, once ``before``, the one it takes the place of, is done:
        ensure_ready waits for the last cleanup alone, and the close of one still running would
        close the session it opens. Cancelled while it waits, it cancels ``before`` too."""
        if before is not None:
            try:
                await asyncio.wait({before})
            except asyncio.CancelledError:
                before.cancel()
                raise
        await self._interrupt_and_close(generation)

    def _session_lock(self) -> asyncio.Lock:
        # Made on first use, on the execution thread's loop, which every operation runs on.
        if self._operation_lock is None:
            self._operation_lock = asyncio.Lock()
        return self._operation_lock

    async def _sync(
        self, request: HistorySyncRequest, record: Callable[..., None]
    ) -> HistorySyncResult:
        """One sync, under the session lock its caller holds."""
        self._stop_if_abandoned("its sync")
        try:
            result = await self.sync_history(request)
        except Exception as exc:
            record(request, error=exc)
            raise
        record(request, result)
        await self._release_if_abandoned()
        return result

    @staticmethod
    def _stop_if_abandoned(before: str) -> None:
        if _abandoned():
            raise RetryCancelled(f"generation abandoned before {before}")

    async def _release_if_abandoned(self) -> None:
        """An abort that came while this connected or synced closed what was open then, not the
        session this opened or rebuilt: close that too, so the next generation opens its own."""
        if _abandoned():
            await self._interrupt_and_close(None)
            raise RetryCancelled("generation abandoned while it readied its session")

    async def _interrupt_and_close(self, generation: GenerationHandle | None) -> None:
        if generation is not None:
            try:
                await self.interrupt(generation)
            except Exception:  # noqa: BLE001, S110 - abort remains best-effort
                pass
        try:
            await self._close_current()
        except Exception:  # noqa: BLE001, S110 - the session is closed below all the same
            pass
        self._connected, self._broken = False, False

    async def _settle(self) -> bool:
        """Wait for the last abort's cleanup; one still running after _CLEANUP_WAIT_S is
        cancelled. Whether it was."""
        cleanup = self._cleanup
        if cleanup is None:
            return False
        if not cleanup.done():
            self._record_wait("abort_cleanup", _CLEANUP_WAIT_S)
        cancelled = False
        try:
            await self.clock.wait_for(asyncio.wait({cleanup}), _CLEANUP_WAIT_S)
        except TimeoutError:
            cleanup.cancel()
            cancelled = True
        if self._cleanup is cleanup:
            self._cleanup = None
        return cancelled

    async def _wait(self, wait: str, seconds: float) -> None:
        """Wait ``seconds`` on the operational clock, on an executor thread as RequestRetry's
        backoff does, and like it on the cancel event of the enclosing retry_cancellation: once
        the generation the wait belongs to is abandoned, RetryCancelled ends the wait. A wait of
        no time, or for a generation already abandoned, is neither recorded nor taken."""
        if seconds <= 0:
            return
        cancel = retry_cancel_event()
        if cancel.is_set():
            raise RetryCancelled(f"generation abandoned before the {wait} wait")
        self._record_wait(wait, seconds)
        if await asyncio.to_thread(self.clock.wait, cancel, seconds):
            raise RetryCancelled(f"generation abandoned during the {wait} wait")

    @staticmethod
    def _preparing() -> None:
        """ensure_ready or renew starts preparing the session: tell the context's
        session_preparing."""
        _PREPARING.get()()

    def _record_delivery(self, state: str, contents, connection: str | None) -> None:
        """What became of contents the session core wrote on the trace's ``connection``, which
        the connection's SessionMessage records name as their connection_id: sent once the
        write returned, confirmed or refused by the server, or unknown once the write failed.
        Each content is its key in the core's delivery record and the history block it carries,
        None where the key names the content instead (an OpenAI Realtime delete, whose create
        named the block; a Gemini Live tool response, by its call id). Only what a session core
        plans is recorded: Gemini Live's setup (as the "system" block), seed, observations,
        pushes, triggers and tool responses; OpenAI Realtime's item creates and deletes, not
        the session.update that carries its system instruction."""
        self._trace.capture(
            TraceEvent.SESSION_DELIVERY,
            {
                "state": state,
                "connection_id": connection,
                "contents": [{"key": list(key), "block": block} for key, block in contents],
            },
        )

    def _record_wait(self, wait: str, seconds: float, until: float | None = None) -> None:
        """``until`` is the end of a wait bounded by a deadline, on the operational clock; a
        wait without one ends ``seconds`` from now."""
        if until is None:
            until_ns = self.clock.monotonic_ns() + round(seconds * 1e9)
        else:
            until_ns = round(until * 1e9)
        self._trace.capture(
            TraceEvent.SESSION_WAIT,
            {"wait": wait, "seconds": seconds, "until_monotonic_ns": until_ns},
        )


@dataclass(frozen=True)
class SessionRecord:
    """A session operation's ``record`` call as an agent's trace has it: the HistoryChanged's
    kind and trigger and its sync fields, whether the operation replayed the history on the
    session, and the SessionStateChanged it leaves, as (state, cause, provider history state)."""

    kind: str
    trigger: str
    rebuilt: bool
    fields: dict
    state: tuple[str, str, ProviderHistoryState | None]


def session_record(
    sync: HistorySyncRequest | None,
    result: HistorySyncResult | None = None,
    error: BaseException | None = None,
) -> SessionRecord:
    """The record of one ``record(sync, result=None, error=None)`` call, alike for the
    turn-based runtime and the native-streaming agent: a reopened session when ``sync`` is None,
    otherwise one sync, of the kind the driver's result reports. The fallback kinds come from
    provider_metadata keys the drivers already return."""
    if sync is None:
        return SessionRecord(
            "rebuild",
            "recovery",
            True,
            {"sync": {"reason": "reconnect"}},
            ("exact", "reconnect", ProviderHistoryState.EXACT),
        )
    reason = sync.reason
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
    if result is None:
        state = ("unknown", f"sync_error:{reason}", None)
    elif kind == "session_resumed":
        # Exact for the driver, but restored from the resume handle, not sent on this
        # connection.
        state = ("unknown", "session_resumed", result.history_state)
    elif result.history_state == ProviderHistoryState.EXACT:
        state = ("exact", f"sync:{reason}", result.history_state)
    else:
        state = ("unknown", f"sync:{reason}", result.history_state)
    fields = {
        "sync": {
            "reason": reason,
            "force_rebuild": sync.force_rebuild,
            "disposition": result.disposition.value if result is not None else None,
            "history_state": result.history_state.value if result is not None else None,
            "detail": result.detail if result is not None else None,
            "provider_metadata": metadata,
        },
        "error": repr(error)[:500] if error is not None else None,
    }
    return SessionRecord(kind, trigger, rebuilt, fields, state)


def replay_items(policy: PathPolicy) -> list[str]:
    """The context policy items a session rebuilt from the history view replays it by: item 10,
    past tool calls in the policy's format."""
    return ["tool_call_replay"] if policy.tool_call_replay is not None else []


class SessionStates:
    """An agent's SessionStateChanged records: one each time the provider-side context becomes
    unknown, or exact again, which it is before anything says otherwise. What became of each
    content the encoder wrote is its SessionDelivery records; what the provider keeps beside it,
    the capability server_context. The native-streaming agent records from the execution thread
    and from its abort's."""

    def __init__(self, trace, protocol_family: str) -> None:
        self._trace, self._family = trace, protocol_family
        self._state = "exact"
        self._lock = threading.Lock()

    def record(
        self,
        state: str,
        cause: str,
        provider_state: ProviderHistoryState | None = None,
        generation_id: str | None = None,
    ) -> None:
        with self._lock:
            if state == self._state:
                return
            self._state = state
            self._trace.capture(
                TraceEvent.SESSION_STATE_CHANGED,
                {
                    "protocol_family": self._family,
                    "state": state,
                    "cause": cause,
                    "provider_history_state": provider_state.value if provider_state else None,
                    "generation_id": generation_id,
                },
            )


def _abandoned() -> bool:
    """Whether the generation this context's operations belong to was abandoned."""
    return retry_cancel_event().is_set()


class StatelessSession(SessionLifecycle):
    """A stateless driver as a session: its connect, sync_history and close stand for the
    connection operations. Nothing on the provider side outlives a request, but the runtime
    readies, aborts and closes it as it does a live session's; a driver whose abort closed its
    client makes a new one when the next generation connects again."""

    def __init__(self, driver, *, clock, trace):
        self.driver = driver
        self.capabilities = driver.capabilities
        self.clock = clock
        self._trace = trace

    async def _connect(self, request: ConnectRequest) -> None:
        await self.driver.connect(request)

    async def _sync_history(self, request: HistorySyncRequest) -> HistorySyncResult:
        return await self.driver.sync_history(request)

    async def _close_current(self) -> None:
        await self.driver.close()

    async def interrupt(self, generation: GenerationHandle):
        return await self.driver.interrupt(generation)

    def connection_renewal_reason(self) -> str | None:
        return self.driver.connection_renewal_reason()
