"""The thread on which an agent's driver operations run."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading

from realtime_framework.clock import Clock
from realtime_framework.entity.model import RetryCancelled, retry_cancel_event


class ExecutionThread:
    """One private asyncio event loop, on its own thread, for every operation of the driver.

    The synchronous harness calls the agent from the runner's main thread and from a worker thread
    per decision. Driver operations submitted here run on this one loop instead, whatever the
    protocol, so async WebSocket SDKs and the synchronous transports a driver bridges with
    asyncio.to_thread follow one order. The operational clock makes the loop, so the drivers'
    waits run on its time.
    """

    def __init__(self, clock: Clock, name: str):
        self.clock = clock
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """Started and not stopped: submit takes coroutines."""
        return self._loop is not None and not self._loop.is_closed()

    def start(self) -> None:
        if self.running:
            return
        loop = self.clock.new_event_loop()
        ready = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(target=run, name=self._name, daemon=True)
        thread.start()
        ready.wait()
        self._loop = loop
        self._thread = thread

    def submit(self, coroutine) -> concurrent.futures.Future:
        """Schedule ``coroutine`` on the thread and return at once. Coroutines start in the
        order they are submitted."""
        if not self.running:
            coroutine.close()
            raise RuntimeError("the execution thread is not running")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def run(self, coroutine, timeout: float | None = None):
        """Run ``coroutine`` on the thread and return its result. After ``timeout`` seconds on
        the clock it is cancelled and TimeoutError raised. One that stop() cancelled raises
        RetryCancelled once the generation it runs for was abandoned (retry_cancellation), as
        the harness abandons those under way before it stops the thread."""
        future = self.submit(coroutine)
        if not self.clock.wait(future, timeout):
            if not future.done():
                future.cancel()
            raise TimeoutError(
                f"an operation on the execution thread exceeded {timeout:g}s"
            ) from None
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            if retry_cancel_event().is_set():
                raise RetryCancelled("generation abandoned while its operation ran") from None
            raise

    def stop(self, timeout: float = 5.0) -> None:
        """Cancel the coroutines still pending, let their cleanup end on the loop within
        ``timeout`` seconds, as asyncio.run ends its loop, then stop the loop and, once its
        thread has ended within another ``timeout``, close it. A coroutine left pending on the
        closed loop would be finalized wherever the garbage collector frees it, and the cleanup
        in its finally clauses would run on that thread: there an asyncio.to_thread can take the
        executor lock that thread already holds, and the thread waits forever. A cleanup that
        does not end in time is left pending. A loop still blocked in a call after both timeouts
        stops once the call returns, and the thread takes no coroutine meanwhile."""
        if not self.running:
            return
        loop = self._loop
        asyncio.run_coroutine_threadsafe(_end_pending(), loop)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=timeout)
        if loop.is_running():
            self._loop = None
        else:
            loop.close()


async def _end_pending() -> None:
    """Cancel every other task on the running loop, wait for them, then stop the loop."""
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending)
    for task in pending:
        if not task.cancelled():
            task.exception()  # retrieved, so the loop logs no unretrieved exception
    asyncio.get_running_loop().stop()
