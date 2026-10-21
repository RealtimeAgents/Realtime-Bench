"""The OpenAI Realtime driver in native streaming (stream-plan 2.4): its reader reads every event
as it arrives, and the session takes pushed frames and replies.

The reader tests drive a session on the fake connection of tests/session/test_openai_realtime_receive.py,
which counts the reads in flight. Streaming views come from the streaming history view
(harnesses/streaming/history.py), as the session core tests build them.
"""

from __future__ import annotations

import asyncio
import itertools

from realtime_framework.entity.events import ProviderHistoryState
from realtime_framework.entity.model import HistorySyncRequest, HistorySyncResult, SyncDisposition
from realtime_framework.providers.openai_realtime import _Session
from tests.session.test_openai_realtime_receive import (
    SEED,
    _clean_close,
    _FakeConnection,
    _offline_driver,
    _within,
)
from tests.session.test_openai_realtime_session import (
    done,
    new_core,
    response_events,
    sent,
    server,
    stream,
    streaming_view,
)


class Ticks:
    """A clock whose every reading is one second after the one before."""

    def __init__(self) -> None:
        self._readings = itertools.count(1.0)

    def monotonic(self) -> float:
        return next(self._readings)


async def until(condition, timeout: float = 5.0) -> None:
    """Wait in real time until ``condition()`` holds."""
    for _ in range(round(timeout / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the condition never held")


def text_done(event_id: str, text: str):
    return server(
        type="response.output_text.done",
        event_id=event_id,
        response_id="resp_1",
        item_id="srv_1",
        output_index=0,
        content_index=0,
        text=text,
    )


def test_a_streaming_session_reads_each_event_as_it_arrives():
    """A native-streaming session reads every event as it arrives, though nothing waits for it,
    and hands each to the core with the time the clock read then; a waiter only waits for those
    reads, one at a time. A clean close ends the reader, and the waiter with it."""

    async def scenario():
        connection, core = _FakeConnection(), new_core()
        (pushed,) = sent(core, core.push(streaming_view(stream("F"))))
        session = _Session(1, connection, core)
        session.read_continuously(Ticks())
        reply = core.start_reply()
        created, added, _, finished = response_events()
        for event in (done(pushed.item_id), created, added):
            connection.queue.put(event)
        await until(lambda: len(reply.events) == 2)
        waiter = asyncio.create_task(session.receive_until(lambda: len(reply.events) == 4))
        for event in (text_done("event_7", "{}"), finished, _clean_close()):
            connection.queue.put(event)
        await waiter
        await session.receive_until(lambda: False)
        return connection, core, session, reply, pushed

    connection, core, session, reply, pushed = _within(scenario())
    assert connection.most_in_flight == 1
    assert core.acknowledged_at == {pushed.key: 1.0}
    assert reply.received_at == {"event_7": 4.0}
    assert [event.type for event in reply.events] == [
        "response.created",
        "response.output_item.added",
        "response.output_text.done",
        "response.done",
    ]
    assert session.closed is not None and session.failure is None


def test_a_failed_read_ends_the_streaming_reader_and_raises_in_every_waiter():
    failure = ConnectionError("read failed")

    async def scenario():
        connection, core = _FakeConnection(), new_core()
        session = _Session(1, connection, core)
        session.read_continuously(Ticks())
        reply = core.start_reply()
        waiters = [
            asyncio.create_task(session.receive_until(lambda: reply.events)) for _ in range(2)
        ]
        connection.queue.put(failure)
        return await asyncio.gather(*waiters, return_exceptions=True)

    assert all(raised is failure for raised in _within(scenario()))


class _ClosingManager:
    """Stands in for the SDK's connection manager: leaving it closes the connection, which ends
    the read in flight with a clean close."""

    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection

    def __exit__(self, *exc_info) -> None:
        self.connection.queue.put(_clean_close())


def test_closing_a_streaming_session_lets_its_reader_end():
    """Closing the connection ends the reader's read with a clean close; the close returns once
    the reader has ended, and nothing is left reading."""

    async def scenario():
        driver, connection = _offline_driver(), _FakeConnection()
        session = _Session(1, connection, new_core())
        session.read_continuously(Ticks())
        driver._session, driver._cm = session, _ClosingManager(connection)
        await until(lambda: connection.in_flight == 1)
        await driver._close_current()
        return connection, session

    connection, session = _within(scenario())
    assert connection.in_flight == 0
    assert session.closed is not None and session.failure is None


def test_a_streaming_sync_rebuilds_the_session():
    """Native streaming edits its session as frames are pushed; a sync, whatever its reason,
    rebuilds the session from the streaming view (decisions 4.7, 4.14)."""
    driver = _offline_driver()
    driver.interaction_mode = "streaming"
    driver._conn = object()
    rebuilt = []

    async def rebuild(request):
        rebuilt.append(request.reason)
        return HistorySyncResult(SyncDisposition.REBUILT, ProviderHistoryState.EXACT)

    driver._rebuild = rebuild
    history = streaming_view(stream("FF"))
    for reason in ("history_block_compaction", "provider_history_changed"):
        _within(driver._sync_history(HistorySyncRequest(SEED, reason, history)))
    assert rebuilt == ["history_block_compaction", "provider_history_changed"]
