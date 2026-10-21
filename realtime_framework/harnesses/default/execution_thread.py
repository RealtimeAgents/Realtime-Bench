"""The thread on which an agent's driver operations run."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading

from realtime_framework.clock import Clock


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
        the clock it is cancelled and TimeoutError raised."""
        future = self.submit(coroutine)
        if not self.clock.wait(future, timeout):
            if not future.done():
                future.cancel()
            raise TimeoutError(
                f"an operation on the execution thread exceeded {timeout:g}s"
            ) from None
        return future.result()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the loop and, once its thread has ended within ``timeout`` seconds, close it.
        Coroutines still pending on it never finish."""
        if not self.running:
            return
        loop = self._loop
        loop.call_soon_threadsafe(loop.stop)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        if not loop.is_running():
            loop.close()
