"""What both live session encoders do around their connection, whatever the protocol.

A session encoder keeps one provider session open across generations. Whether that session can
take the next generation, and the cleanup an abort leaves behind for the next use to wait for, do
not depend on the protocol: SessionLifecycle keeps both, and readies or aborts the session through
the encoder's own connection operations. The encoder's own waits run through it too.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

from realtime_framework.entity.events import ProviderHistoryState
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationHandle,
    HistorySyncRequest,
    HistorySyncResult,
    RetryCancelled,
    SessionSeed,
    SyncDisposition,
)
from realtime_framework.entity.request import CanonicalRequest

_NEVER_CANCELLED = threading.Event()


class SessionLifecycle:
    """A live encoder's session: connected or not, dirty or not, and the last abort's cleanup.

    The session is connected once connect() opened it or a sync left it exact, and no longer once
    close() or abort() closed it or a sync raised or broke it. A connected session is dirty when
    the provider may hold what the canonical history does not: a generation on it failed
    (mark_dirty) or a sync left it neither exact nor broken. A connection that an operation
    closes on its way (the one a connect replaces, one past its goAway deadline) changes neither.

    The encoder provides the connection operations: ``_connect(request)``,
    ``_sync_history(request)``, ``_close_current()``, which closes the open connection if there
    is one and does not raise, and ``interrupt(generation)``; and ``clock``, the operational
    clock.
    """

    _connected = False
    _dirty = False
    _cleanup: asyncio.Task | None = None

    async def connect(self, request: ConnectRequest) -> None:
        self._connected = False
        await self._connect(request)
        self._connected, self._dirty = True, False

    async def sync_history(self, request: HistorySyncRequest) -> HistorySyncResult:
        try:
            result = await self._sync_history(request)
        except Exception:
            self._connected = False
            raise
        if result.history_state == ProviderHistoryState.EXACT:
            self._connected, self._dirty = True, False
        elif result.disposition in {SyncDisposition.UNSUPPORTED, SyncDisposition.BROKEN}:
            self._connected = False
        else:
            self._dirty = True
        return result

    async def close(self) -> None:
        await self._settle()
        self._connected = False
        await self._close_current()

    def mark_dirty(self) -> None:
        """A generation on the session failed: the next ensure_ready rebuilds it."""
        self._dirty = True

    async def ensure_ready(
        self,
        seed: SessionSeed,
        history: CanonicalRequest,
        record: Callable[..., None],
    ) -> HistorySyncResult | None:
        """Make the session ready for the next generation, once the last abort's cleanup is done.

        A ready session is left as it is. A closed one is reopened with ``history``. A dirty one
        is rebuilt from ``history`` by a runtime_recovery sync, whose result is returned.
        ``record(sync, result=None, error=None)`` sees what was done before this returns or
        raises: the recovery's HistorySyncRequest with its result or the exception it raised, or
        None for a reopened session. A failed reopen raises unrecorded.
        """
        await self._settle()
        if self._connected and not self._dirty:
            return None
        if not self._connected:
            await self.connect(ConnectRequest(seed=seed, history=history))
            record(None)
            return None
        sync = HistorySyncRequest(seed, "runtime_recovery", history, force_rebuild=True)
        try:
            result = await self.sync_history(sync)
        except Exception as exc:
            record(sync, error=exc)
            raise
        record(sync, result)
        return result

    async def abort(self, generation: GenerationHandle | None) -> None:
        """Interrupt ``generation`` (None when none had started), then close the connection, as
        far as either works. The next ensure_ready, or close, waits for this to finish."""
        self._cleanup = asyncio.ensure_future(self._interrupt_and_close(generation))
        await asyncio.wait({self._cleanup})

    async def _interrupt_and_close(self, generation: GenerationHandle | None) -> None:
        if generation is not None:
            try:
                await self.interrupt(generation)
            except Exception:  # noqa: BLE001, S110 - abort remains best-effort
                pass
        await self._close_current()
        self._connected = False

    async def _settle(self) -> None:
        cleanup = self._cleanup
        if cleanup is None:
            return
        await asyncio.wait({cleanup})
        if self._cleanup is cleanup:
            self._cleanup = None

    async def _wait(
        self, wait: str, seconds: float, cancel: threading.Event = _NEVER_CANCELLED
    ) -> None:
        """Wait ``seconds`` on the operational clock, as RequestRetry's backoff does. Once
        ``cancel`` is set the generation the wait belongs to is abandoned, and RetryCancelled
        ends the wait."""
        if await asyncio.to_thread(self.clock.wait, cancel, seconds):
            raise RetryCancelled(f"generation abandoned during the {wait} wait")
