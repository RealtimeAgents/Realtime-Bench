"""The OpenAI Realtime driver in native streaming (stream-plan 2.4): its reader reads every event
as it arrives, and the session takes pushed frames and replies.

The reader tests drive a session on the fake connection of tests/session/test_openai_realtime_receive.py,
which counts the reads in flight. Streaming views come from the streaming history view
(harnesses/streaming/history.py), as the session core tests build them.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from functools import partial
from types import SimpleNamespace

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from realtime_framework.clock import SystemClock
from realtime_framework.entity.events import (
    GenerationBoundary,
    ProviderHistoryState,
    TextDelta,
    TextDone,
    UsageSnapshot,
)
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationHandle,
    GenerationRequest,
    HistorySyncRequest,
    HistorySyncResult,
    PushReceipt,
    RuntimeState,
    SessionSeed,
    SyncDisposition,
)
from realtime_framework.entity.records import UNTRACED, current_trace_scope, trace_scope
from realtime_framework.entity.request import AnswerRequirements
from realtime_framework.harnesses.default.execution_thread import ExecutionThread
from realtime_framework.harnesses.default.frames import FrameTable
from realtime_framework.harnesses.streaming.history import frame_block
from realtime_framework.providers import openai_realtime
from realtime_framework.providers.openai_realtime import OpenAIRealtimeDriver, _Session
from realtime_framework.providers.openai_realtime_session import Delete
from tests.fake_clock import FakeClock
from tests.replay import baseline
from tests.session.test_openai_realtime_receive import (
    SEED,
    _clean_close,
    _FakeConnection,
    _offline_driver,
    _within,
)
from tests.session.test_openai_realtime_session import (
    TEXTS,
    Server,
    done,
    error,
    new_core,
    response_events,
    sent,
    server,
    stream,
    streaming_view,
)

from .clocked_ws import CassetteServer, clocked_connection
from .openai_session import (
    MODEL_KEY,
    SYSTEM_INSTRUCTION,
    _observation,
    driver_for,
    frame_png,
    run_on_fake_clock,
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


class _ScopedConnection(_FakeConnection):
    """The fake connection, which keeps the trace scope each read runs in."""

    def __init__(self) -> None:
        super().__init__()
        self.scopes: list[dict | None] = []

    def recv(self):
        self.scopes.append(current_trace_scope())
        return super().recv()


def test_the_streaming_reader_reads_in_no_generation_scope():
    """The reader reads for the connection, whatever reply is in progress: the trace records
    what it reads in no scope, not in that of the call that opened the connection, often a
    rebuild for an earlier generation."""

    async def scenario():
        connection = _ScopedConnection()
        session = _Session(1, connection, new_core())
        with trace_scope(decision=2, generation=2):
            session.read_continuously(Ticks())
        connection.queue.put(_clean_close())
        await session.receive_until(lambda: False)
        return connection.scopes

    assert _within(scenario()) == [None]


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
        raised = await asyncio.gather(*waiters, return_exceptions=True)
        ended = session._reader.done()
        connection.queue.put(_clean_close())  # ends a reader that went on reading
        return raised, ended, session

    raised, ended, session = _within(scenario())
    assert all(each is failure for each in raised)
    assert ended
    assert session.failure is failure


def test_an_event_the_core_fails_on_ends_the_streaming_reader_and_raises_in_every_waiter():
    """The session core failing on an event (here a second error about one push) ends the reader
    as a failed read does: every waiter raises the core's error instead of waiting for good."""

    async def scenario():
        connection, core = _FakeConnection(), new_core()
        (pushed,) = sent(core, core.push(streaming_view(stream("F"))))
        session = _Session(1, connection, core)
        session.read_continuously(Ticks())
        reply = core.start_reply()
        waiter = asyncio.create_task(session.receive_until(lambda: reply.events))
        for event in (error(pushed.event_id), error(pushed.event_id)):
            connection.queue.put(event)
        (raised,) = await asyncio.gather(waiter, return_exceptions=True)
        return session, raised

    session, raised = _within(scenario(), timeout=5.0)
    assert isinstance(raised, KeyError)
    assert session.failure is raised
    assert session._reader.done()


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
        return connection, session, session._reader.done()

    connection, session, ended = _within(scenario())
    assert ended
    assert connection.in_flight == 0
    assert session.closed is not None and session.failure is None


class _FailingManager:
    """The SDK's connection manager whose close raises, which may leave the connection open."""

    def __exit__(self, *exc_info) -> None:
        raise RuntimeError("close failed")


class _StuckConnection:
    """An SDK connection whose writes block on a socket that does not drain, until it is closed;
    its close_socket() closes it, which fails every write stuck in it."""

    def __init__(self) -> None:
        self._open = threading.Event()
        self._connection = self
        self.writing = threading.Event()
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=self.write))
        self.response = SimpleNamespace(cancel=self.write)

    def write(self, **kwargs) -> None:
        self.writing.set()
        if not self._open.wait(timeout=5):
            raise AssertionError("the socket was never closed")
        raise ConnectionClosedError(None, None)

    def close_socket(self) -> None:
        self._open.set()


class _StuckManager:
    """The SDK's connection manager whose close blocks on the same undraining socket."""

    def __init__(self, connection: _StuckConnection) -> None:
        self.connection = connection

    def __exit__(self, *exc_info) -> None:
        self.connection._open.wait(timeout=5)


@pytest.mark.parametrize("on_reply", [False, True])
def test_an_abort_fails_a_push_stuck_in_a_write_and_frees_the_session(monkeypatch, on_reply):
    """A push stuck in a write that does not drain holds the session and the connection's
    writes. An abort of the reply on the session, as the kernel's push watchdog gives one up,
    writes its cancel behind that write, and gives the cancel up at its bound (review-encoders
    中 1); the abort's close then waits on the same write until its own bound and drops the
    socket under it: the push fails, the close returns, and the session is freed for the next
    ensure_ready to rebuild."""
    monkeypatch.setattr(openai_realtime, "_CONNECTION_CLOSE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(openai_realtime, "_INTERRUPT_TIMEOUT_S", 0.05)

    async def scenario():
        driver, connection = _offline_driver(streaming=True), _StuckConnection()
        session = _Session(1, connection, new_core())
        driver._session, driver._conn, driver._cm = session, connection, _StuckManager(connection)
        view = streaming_view(stream("F"))
        push = asyncio.ensure_future(driver.push_frame(_observation(1), view))
        await until(connection.writing.is_set)
        reply = session.core.start_reply() if on_reply else None
        handle = GenerationHandle("streaming-1", opaque=(session, reply)) if on_reply else None
        await asyncio.wait_for(driver.abort(handle), 2)
        (failure,) = await asyncio.gather(push, return_exceptions=True)
        return failure, driver._session, driver._connected

    failure, left_session, connected = _within(scenario())
    assert isinstance(failure, ConnectionClosedError)
    assert left_session is None and connected is False


class _SocketConnection(_FakeConnection):
    """The fake connection as the SDK holds websockets' (``_connection``), whose
    close_socket() ends the read in flight, as websockets' does."""

    @property
    def _connection(self):
        return self

    def close_socket(self) -> None:
        self.queue.put(ConnectionClosedError(None, None))


def test_a_failed_close_cancels_the_reader_and_ends_its_waiters():
    """A close that raises may leave the connection open, so the reader is cancelled instead of
    waited for; whoever waits on the session then raises rather than waiting for good. The
    cancelled reader closes the socket, which ends the read its thread is blocked in."""

    async def scenario():
        driver, connection = _offline_driver(), _SocketConnection()
        session = _Session(1, connection, new_core())
        session.read_continuously(Ticks())
        reply = session.core.start_reply()
        waiter = asyncio.create_task(session.receive_until(lambda: reply.events))
        driver._session, driver._cm = session, _FailingManager()
        await until(lambda: connection.in_flight == 1)
        await driver._close_current()
        cancelled = session._reader.cancelled()
        raised = await asyncio.gather(waiter, return_exceptions=True)
        await until(lambda: connection.in_flight == 0, timeout=1.0)
        return cancelled, raised

    cancelled, (raised,) = _within(scenario())
    assert cancelled
    assert isinstance(raised, ConnectionError)


def test_stopping_the_loop_under_a_streaming_reader_closes_its_socket():
    """An execution thread stopped with the connection still open cancels the reader, which
    closes the socket: no thread is left blocked in a read."""
    connection, execution = _SocketConnection(), ExecutionThread(SystemClock(), "reader")
    execution.start()

    async def start():
        _Session(1, connection, new_core()).read_continuously(Ticks())

    execution.submit(start()).result()
    _within(until(lambda: connection.in_flight == 1))
    execution.stop()
    _within(until(lambda: connection.in_flight == 0, timeout=1.0))


def test_a_model_yaml_sets_the_frames_the_session_keeps():
    """The model YAML's streaming settings replace the driver's defaults they name
    (ModelConfig.streaming); the native-streaming agent reads the frame count off the driver."""
    driver = _offline_driver(streaming=True)
    assert driver.streaming_settings == OpenAIRealtimeDriver.streaming_settings
    config = driver.config.model_copy(update={"streaming": {"retained_frames": 5}})
    assert OpenAIRealtimeDriver(config, streaming=True).streaming_settings == {"retained_frames": 5}
    assert OpenAIRealtimeDriver.streaming_settings == {"retained_frames": 20}


def test_a_streaming_sync_rebuilds_the_session():
    """Native streaming edits its session as frames are pushed; a sync, whatever its reason,
    rebuilds the session from the streaming view (decisions 4.7, 4.14)."""
    driver = _offline_driver(streaming=True)
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


class _Answers(Server):
    """The session core tests' server model, which also refuses the create of each block in
    ``refused`` and fails every delete when ``deletes_fail``."""

    def __init__(self, refused=(), *, deletes_fail: bool = False) -> None:
        super().__init__()
        self.refused, self.deletes_fail = set(refused), deletes_fail

    def answer(self, operation):
        if operation.block in self.refused or (self.deletes_fail and isinstance(operation, Delete)):
            return error(operation.event_id)
        return super().answer(operation)


async def answering(server: Server, blocks):
    """An offline driver on a native-streaming session over a fake connection, behind which
    ``server`` answers each operation the driver sends; the session holds the replay of
    ``blocks``. A clean close queued on the connection ends its reader."""
    driver, connection = _offline_driver(), _FakeConnection()
    session = _Session(1, connection, new_core())
    session.read_continuously(Ticks())
    driver._session, driver._conn = session, connection
    driver._send = lambda conn, operation: conn.queue.put(server.answer(operation))
    replay = session.core.replay_streaming(streaming_view(list(blocks), frames=5))
    await driver._send_all(session, replay)
    await session.receive_until(lambda: all(map(session.core.settled, replay)))
    return driver, connection


def test_a_refused_frame_raises_though_the_feedback_pushed_before_it_went_in():
    """Under separate_message the feedback the session lacks goes in as an item of its own before
    the frame; the server refusing the frame raises, and the feedback stays."""

    async def scenario():
        driver, connection = await answering(_Answers({"f2.frame"}), stream("FR"))
        with pytest.raises(RuntimeError, match="^OpenAI Realtime refused f2.frame: "):
            await driver.push_frame(_observation(2), streaming_view(stream("FRBF"), frames=5))
        connection.queue.put(_clean_close())
        return driver._session.core

    assert sorted(_within(scenario()).frames) == ["d1.feedback", "f1.frame"]


def test_a_fold_the_server_fails_leaves_the_session_dirty():
    """A fold the server fails leaves the session dirty, for the next ensure_ready to rebuild it
    from the view; the frame pushed with it is in all the same."""

    async def scenario():
        driver, connection = await answering(_Answers(deletes_fail=True), stream("F"))
        await driver.push_frame(_observation(2), streaming_view(stream("FF"), frames=1))
        connection.queue.put(_clean_close())
        return driver

    driver = _within(scenario())
    assert driver._dirty
    assert "f2.frame" in driver._session.core.frames


def test_a_push_of_a_frame_the_session_holds_waits_for_its_confirmation():
    """A push writes only what the session lacks: a frame the session holds already is not
    pushed again, and push_frame returns once it is confirmed, and raises when it was refused.
    The agent pushes each frame before any view holds it, so this is the core's planning alone."""

    async def scenario():
        driver, connection = await answering(Server(), stream("FF"))
        await driver.push_frame(_observation(2), streaming_view(stream("FF"), frames=5))
        connection.queue.put(_clean_close())
        driver, connection = await answering(_Answers({"f2.frame"}), stream("F"))
        view = streaming_view(stream("FF"), frames=5)
        await driver._send_all(driver._session, driver._session.core.push(view))
        with pytest.raises(RuntimeError, match="^OpenAI Realtime refused f2.frame: "):
            await driver.push_frame(_observation(2), view)
        connection.queue.put(_clean_close())

    _within(scenario())


def test_a_connection_that_closes_before_the_frame_is_confirmed_fails_the_push():
    async def scenario():
        driver, _ = await answering(Server(), stream("F"))
        driver._send = lambda conn, operation: conn.queue.put(_clean_close())
        with pytest.raises(ConnectionClosedOK):
            await driver.push_frame(_observation(2), streaming_view(stream("FF"), frames=5))

    _within(scenario())


# ---- a streaming session against a cassette, on the fake clock -------------------------------

PLACEHOLDER = TEXTS.folded_visual
SESSION_UPDATE = {
    "type": "session.update",
    "session": {
        "type": "realtime",
        "output_modalities": ["text"],
        "instructions": SYSTEM_INSTRUCTION,
        "truncation": "disabled",
    },
}
RESPONSE_CREATE = {
    "type": "response.create",
    "response": {"max_output_tokens": "inf", "reasoning": {"effort": "low"}},
    "event_id": "<rtb_reply-1>",
}
USAGE = {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}


def c2s(msg: dict) -> dict:
    return {"conn": 1, "dir": "c2s", "at": None, "msg": msg}


def s2c(msg: dict, at: float | None = None) -> dict:
    return {"conn": 1, "dir": "s2c", "at": at, "msg": msg}


def push(user: int, frame: int, push_event: int) -> dict:
    image = f"data:image/png;{baseline.image_digest(frame_png(frame))}"
    content = [
        {"type": "input_text", "text": f"frame {frame}"},
        {"type": "input_image", "image_url": image},
    ]
    item = {"id": f"<rtb_user-{user}>", "type": "message", "role": "user", "content": content}
    return c2s(
        {"type": "conversation.item.create", "item": item, "event_id": f"<rtb_push-{push_event}>"}
    )


def fold(deleted: int, delete_event: int, placeholder: int, create_event: int, after: str):
    item = {
        "id": f"<rtb_user-{placeholder}>",
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": PLACEHOLDER}],
    }
    return [
        c2s(
            {
                "type": "conversation.item.delete",
                "item_id": f"<rtb_user-{deleted}>",
                "event_id": f"<rtb_delete-{delete_event}>",
            }
        ),
        c2s(
            {
                "type": "conversation.item.create",
                "item": item,
                "previous_item_id": after,
                "event_id": f"<rtb_create-{create_event}>",
            }
        ),
    ]


def item_done(user: int, previous: str | None) -> dict:
    item = {"id": f"<rtb_user-{user}>"}
    return s2c({"type": "conversation.item.done", "item": item, "previous_item_id": previous})


def item_deleted(user: int) -> dict:
    return s2c({"type": "conversation.item.deleted", "item_id": f"<rtb_user-{user}>"})


def reply_event(kind: str, at: float, **fields) -> dict:
    return s2c({"type": kind, "response_id": "resp_1", **fields}, at)


# One reply runs from 0.2 s to 2.0 s while frames 2 and 3 are pushed, at 1.0 s and 1.5 s, each
# folding the frame before it (K = 1). The server made the reply's item before frame 2 came,
# which so follows it; frame 2's placeholder goes there too. Frame 3 is refused.
RECORDS = [
    {"conn": 1, "dir": "open", "at": 0.0, "msg": None},
    c2s(SESSION_UPDATE),
    push(1, 1, 1),
    item_done(1, None),
    c2s(RESPONSE_CREATE),
    s2c({"type": "response.created", "response": {"id": "resp_1"}}, 0.2),
    reply_event("response.output_item.added", 0.2, item={"id": "srv_item_1"}),
    reply_event("response.output_text.delta", 0.5, delta='{"action": '),
    push(2, 2, 2),
    *fold(1, 1, 3, 1, "root"),
    item_done(2, "srv_item_1"),
    item_deleted(1),
    item_done(3, None),
    reply_event("response.output_text.delta", 1.2, delta='"wait"}'),
    push(4, 3, 3),
    *fold(2, 2, 5, 2, "srv_item_1"),
    s2c(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "refused",
                "event_id": "<rtb_push-3>",
            },
        }
    ),
    item_deleted(2),
    item_done(5, "srv_item_1"),
    reply_event(
        "response.output_text.done",
        1.9,
        event_id="evt_text",
        item_id="srv_item_1",
        output_index=0,
        content_index=0,
        text='{"action": "wait"}',
    ),
    s2c(
        {
            "type": "response.done",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [{"id": "srv_item_1"}],
                "usage": USAGE,
            },
        },
        2.0,
    ),
    {"conn": 1, "dir": "close", "at": None, "msg": {"by": "client", "code": 1000}},
]


def test_a_streaming_session_pushes_frames_folds_old_ones_and_keeps_its_reply():
    """Frames go in while a reply streams, each fold of an older frame confirmed between the
    reply's events, none of which is lost; a refused frame is lost alone; closing the session
    ends its reader. The cassette checks every message the driver sent, in order."""
    clock, table, blocks = FakeClock(), FrameTable(), []
    answer = AnswerRequirements("text_json", "standard_vlm")
    seed = SessionSeed(SYSTEM_INSTRUCTION, MODEL_KEY, None, 1.0, answer)

    def frame(k: int):
        observation = _observation(k)
        record = table.register(observation.observation_id, 0.0, {}, None)
        blocks.append(frame_block(record, observation))
        return observation, streaming_view(blocks, frames=1)

    async def scenario(driver):
        await driver.connect(ConnectRequest(seed=seed))
        observation, view = frame(1)
        await driver.push_frame(observation, view)
        handle = await driver.request_reply(
            GenerationRequest("reply-1", None, "inf", "wall", canonical=view)
        )
        reply = asyncio.create_task(_collect(driver.events(handle)))
        await asyncio.sleep(1.0)
        await driver.push_frame(*frame(2))
        await asyncio.sleep(0.5)
        with pytest.raises(RuntimeError, match="^OpenAI Realtime refused f3.frame: "):
            await driver.push_frame(*frame(3))
        events = await reply
        session = driver._session
        await driver.close()
        return events, handle, session

    with CassetteServer(RECORDS, clock) as server:
        driver = driver_for(server.port, clock, UNTRACED, streaming=True)
        driver._ws_connection = clocked_connection(driver._ws_connection, clock)
        events, handle, session = run_on_fake_clock(clock, scenario(driver))
    assert server.finish() == []

    assert [type(event) for event in events] == [
        GenerationBoundary,
        TextDelta,
        TextDelta,
        TextDone,
        UsageSnapshot,
        GenerationBoundary,
    ]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == '{"action": "wait"}'
    _, reply = handle.opaque
    assert reply.received_at == {"evt_text": 1.9}
    # The completed text comes with the time it came off the wire (decisions 19 and 4.17).
    (completed,) = (event for event in events if isinstance(event, TextDone))
    assert (completed.text, completed.received_at) == ('{"action": "wait"}', 1.9)
    core = session.core
    assert {k: (f.folded, f.text) for k, f in core.frames.items()} == {
        "f1.frame": (True, PLACEHOLDER),
        "f2.frame": (True, PLACEHOLDER),
    }
    assert session.closed is not None and session.failure is None
    assert session._reader.done() and session._reader.exception() is None


async def _collect(events) -> list:
    return [event async for event in events]


# A rebuilt session replays frames 1 and 2, which the server confirms only at 0.3 s; frame 3,
# pushed next, folds frame 1 after the item the server put it after, which only a confirmation
# names.
REBUILT = [
    {"conn": 1, "dir": "open", "at": 0.0, "msg": None},
    c2s(SESSION_UPDATE),
    push(1, 1, 1),
    push(2, 2, 2),
    item_done(1, None) | {"at": 0.3},
    item_done(2, "<rtb_user-1>") | {"at": 0.3},
    push(3, 3, 3),
    *fold(1, 1, 4, 1, "root"),
    item_done(3, "<rtb_user-2>"),
    item_deleted(1),
    item_done(4, None),
    {"conn": 1, "dir": "close", "at": None, "msg": {"by": "client", "code": 1000}},
]


def test_a_rebuilt_streaming_session_takes_pushes_once_its_replay_is_confirmed():
    """connect() replays the streaming view as pushes and returns once the server confirmed
    them, so the first fold after it knows where the server put each replayed frame."""
    clock, table, blocks = FakeClock(), FrameTable(), []
    answer = AnswerRequirements("text_json", "standard_vlm")
    seed = SessionSeed(SYSTEM_INSTRUCTION, MODEL_KEY, None, 1.0, answer)

    def frame(k: int):
        observation = _observation(k)
        record = table.register(observation.observation_id, 0.0, {}, None)
        blocks.append(frame_block(record, observation))
        return observation, streaming_view(blocks, frames=2)

    async def scenario(driver):
        frame(1)
        _, history = frame(2)
        await driver.connect(ConnectRequest(seed=seed, history=history))
        connected = clock.monotonic()
        await driver.push_frame(*frame(3))
        session = driver._session
        await driver.close()
        return connected, session

    with CassetteServer(REBUILT, clock) as server:
        driver = driver_for(server.port, clock, UNTRACED, streaming=True)
        driver._ws_connection = clocked_connection(driver._ws_connection, clock)
        connected, session = run_on_fake_clock(clock, scenario(driver))
    assert server.finish() == []
    assert connected == 0.3
    assert {k: f.folded for k, f in session.core.frames.items()} == {
        "f1.frame": True,
        "f2.frame": False,
        "f3.frame": False,
    }


def _closed_by_server(code: int) -> list[dict]:
    """The server closes a native-streaming session's connection at 0.3 s with ``code``, no reply
    in progress; the session is rebuilt at 0.5 s on a second connection from a view of one
    frame."""
    rebuilt = [c2s(SESSION_UPDATE), push(1, 1, 1), item_done(1, None)]
    return [
        {"conn": 1, "dir": "open", "at": 0.0, "msg": None},
        c2s(SESSION_UPDATE),
        {
            "conn": 1,
            "dir": "close",
            "at": 0.3,
            "msg": {"by": "server", "code": code, "reason": "x"},
        },
        {"conn": 2, "dir": "open", "at": 0.5, "msg": None},
        *({**record, "conn": 2} for record in rebuilt),
        {"conn": 2, "dir": "close", "at": None, "msg": {"by": "client", "code": 1000}},
    ]


@pytest.mark.parametrize("code", [1000, 1011])
def test_a_connection_the_server_ends_leaves_the_session_to_be_rebuilt(code):
    """A native-streaming connection the server closes, normally or not, leaves the session
    dirty, as Gemini Live's: the next ensure_ready rebuilds it from the view on a new connection
    instead of handing on the dead one."""
    clock, table = FakeClock(), FrameTable()
    answer = AnswerRequirements("text_json", "standard_vlm")
    seed = SessionSeed(SYSTEM_INSTRUCTION, MODEL_KEY, None, 1.0, answer)
    observation = _observation(1)
    record = table.register(observation.observation_id, 0.0, {}, None)
    history = streaming_view([frame_block(record, observation)], frames=2)

    async def scenario(driver):
        await driver.connect(ConnectRequest(seed=seed))
        await asyncio.sleep(0.5)
        ended = driver.state
        await driver.ensure_ready(seed, history, lambda *args, **kwargs: None)
        rebuilt = driver.state
        await driver.close()
        return ended, rebuilt

    with CassetteServer(_closed_by_server(code), clock) as server:
        driver = driver_for(server.port, clock, UNTRACED, streaming=True)
        driver._ws_connection = clocked_connection(driver._ws_connection, clock)
        ended, rebuilt = run_on_fake_clock(clock, scenario(driver))
    assert server.finish() == []
    assert (ended, rebuilt) == (RuntimeState.DIRTY, RuntimeState.READY)


# The server closes a rebuilt session's connection before it confirmed the replay.
CLOSED_BEFORE_CONFIRMED = [
    {"conn": 1, "dir": "open", "at": 0.0, "msg": None},
    c2s(SESSION_UPDATE),
    push(1, 1, 1),
    push(2, 2, 2),
    {"conn": 1, "dir": "close", "at": 0.3, "msg": {"by": "server", "code": 1000, "reason": "bye"}},
]


def test_a_rebuild_whose_connection_closes_before_its_replay_is_confirmed_raises():
    """connect() hands over no session the server closed before it confirmed the replay: it
    raises that close."""
    clock, table, blocks = FakeClock(), FrameTable(), []
    answer = AnswerRequirements("text_json", "standard_vlm")
    seed = SessionSeed(SYSTEM_INSTRUCTION, MODEL_KEY, None, 1.0, answer)
    for k in (1, 2):
        observation = _observation(k)
        record = table.register(observation.observation_id, 0.0, {}, None)
        blocks.append(frame_block(record, observation))

    async def scenario(driver):
        with pytest.raises(ConnectionClosedOK):
            await driver.connect(
                ConnectRequest(seed=seed, history=streaming_view(blocks, frames=2))
            )
        await driver.close()

    with CassetteServer(CLOSED_BEFORE_CONFIRMED, clock) as server:
        driver = driver_for(server.port, clock, UNTRACED, streaming=True)
        driver._ws_connection = clocked_connection(driver._ws_connection, clock)
        run_on_fake_clock(clock, scenario(driver))
    assert server.finish() == []


def test_a_connect_retries_under_the_newest_frame_of_its_history(monkeypatch):
    """The connect retry's jitter key names the session the connect resumes: the newest frame
    of its history."""
    keys = []

    class Stop(Exception):
        pass

    def retry(key, *args, **kwargs):
        keys.append(key)
        raise Stop

    monkeypatch.setattr(openai_realtime, "RequestRetry", retry)
    request = ConnectRequest(seed=SEED, history=streaming_view(stream("FRF")))
    with pytest.raises(Stop):
        _within(_offline_driver()._connect(request))
    assert keys == ["f2.frame:connect"]


def test_a_push_during_a_rebuild_goes_out_once_the_rebuilt_session_took_its_replay():
    """push_frame holds the session as a connection operation does: called at 0.1 s, while
    ensure_ready opens a session and waits for its replay's confirmations, it plans and writes on
    that session once they came at 0.3 s, as REBUILT records."""
    clock, table, blocks = FakeClock(), FrameTable(), []
    answer = AnswerRequirements("text_json", "standard_vlm")
    seed = SessionSeed(SYSTEM_INSTRUCTION, MODEL_KEY, None, 1.0, answer)

    def frame(k: int):
        observation = _observation(k)
        record = table.register(observation.observation_id, 0.0, {}, None)
        blocks.append(frame_block(record, observation))
        return observation, streaming_view(blocks, frames=2)

    async def scenario(driver):
        frame(1)
        _, history = frame(2)
        ready = asyncio.create_task(
            driver.ensure_ready(seed, history, lambda *args, **kwargs: None)
        )
        await asyncio.sleep(0.1)
        await driver.push_frame(*frame(3))
        pushed = clock.monotonic()
        await ready
        session = driver._session
        await driver.close()
        return pushed, session

    with CassetteServer(REBUILT, clock) as server:
        driver = driver_for(server.port, clock, UNTRACED, streaming=True)
        driver._ws_connection = clocked_connection(driver._ws_connection, clock)
        pushed, session = run_on_fake_clock(clock, scenario(driver))
    assert server.finish() == []
    assert pushed == 0.3
    assert {k: f.folded for k, f in session.core.frames.items()} == {
        "f1.frame": True,
        "f2.frame": False,
        "f3.frame": False,
    }


def test_a_reply_asked_for_during_a_rebuild_goes_to_the_rebuilt_session():
    """request_reply holds the session as a connection operation does: called while a rebuild
    holds it, it pushes, registers its reply and writes response.create on the session the
    rebuild leaves, whose events its handle reads."""

    async def scenario():
        driver, old, new = _offline_driver(), _Wire(), _Wire()
        driver._session, driver._conn = _Session(1, old, new_core()), old
        lock = driver._session_lock()
        await lock.acquire()
        view = streaming_view(stream("FRB"))
        reply = asyncio.create_task(
            driver.request_reply(GenerationRequest("reply-1", None, "inf", "wall", canonical=view))
        )
        await asyncio.sleep(0)
        rebuilt = _Session(2, new, new_core())
        sent(rebuilt.core, rebuilt.core.replay_streaming(streaming_view(stream("FR"))))
        driver._session, driver._conn = rebuilt, new
        lock.release()
        return old, new, rebuilt, await reply

    old, new, rebuilt, handle = _within(scenario())
    assert old.written == []
    assert [(kind, kwargs.get("item", {}).get("content")) for kind, kwargs in new.written] == [
        ("item.create", [{"type": "input_text", "text": "fb1"}]),
        ("response.create", None),
    ]
    assert handle.opaque[0] is rebuilt


class _Wire:
    """Stands in for the SDK connection's writes: keeps each item create and delete and each
    response create and cancel, in the order they were written."""

    def __init__(self) -> None:
        self.written: list[tuple[str, dict]] = []
        item = SimpleNamespace(
            create=partial(self._write, "item.create"), delete=partial(self._write, "item.delete")
        )
        self.conversation = SimpleNamespace(item=item)
        self.response = SimpleNamespace(
            create=partial(self._write, "response.create"),
            cancel=partial(self._write, "response.cancel"),
        )

    def _write(self, kind: str, **kwargs) -> None:
        self.written.append((kind, kwargs))


def test_an_interrupt_cancels_a_reply_on_its_own_session_only():
    """A reply whose session a rebuild replaced was closed with it: interrupting it sends nothing,
    as a cancel on the new connection would name a response it never had, whose error fails the
    reply in progress there. A reply on the session as it is is cancelled on its connection."""
    driver, old, new = _offline_driver(), _Wire(), _Wire()
    replaced = _Session(1, old, new_core())
    reply = replaced.core.start_reply()
    reply.response_id = "resp_1"
    driver._session, driver._conn = _Session(2, new, new_core()), new
    result = _within(driver.interrupt(GenerationHandle("reply-1", opaque=(replaced, reply))))
    assert (old.written, new.written) == ([], [])
    assert result.history_state == ProviderHistoryState.UNKNOWN
    driver._session, driver._conn = replaced, old
    result = _within(driver.interrupt(GenerationHandle("reply-1", opaque=(replaced, reply))))
    assert old.written == [
        ("response.cancel", {"response_id": "resp_1", "event_id": reply.cancel_event_id})
    ]
    assert result.history_state == ProviderHistoryState.CONTAMINATED


def test_a_reply_goes_out_after_the_feedback_the_session_lacks():
    """Feedback accepted after the newest frame, an item of its own under separate_message, is
    pushed before the reply's response.create, so the reply sees it; the reply answers no frame."""
    driver, core, wire = _offline_driver(), new_core(), _Wire()
    sent(core, core.push(streaming_view(stream("FR"))))
    driver._session, driver._conn = _Session(1, wire, core), wire
    request = GenerationRequest(
        "reply-2", None, "inf", "wall", canonical=streaming_view(stream("FRB"))
    )
    handle = _within(driver.request_reply(request))
    _, reply = handle.opaque
    (_, feedback), reply_create = wire.written
    assert feedback["item"]["content"] == [{"type": "input_text", "text": "fb1"}]
    assert reply_create == (
        "response.create",
        {
            "response": {"max_output_tokens": "inf", "reasoning": {"effort": "low"}},
            "event_id": reply.request_event_id,
        },
    )
    assert reply.answered is None


def test_a_reply_asked_for_while_one_is_in_progress_raises():
    """A connection carries one reply at a time: request_reply raises while the reply before it
    has not ended, and writes nothing."""
    driver, core, wire = _offline_driver(), new_core(), _Wire()
    sent(core, core.push(streaming_view(stream("F"))))
    driver._session, driver._conn = _Session(1, wire, core), wire
    view = streaming_view(stream("F"))

    async def scenario():
        await driver.request_reply(
            GenerationRequest("reply-1", None, "inf", "wall", canonical=view)
        )
        with pytest.raises(RuntimeError, match="^OpenAI Realtime reply already in progress$"):
            await driver.request_reply(
                GenerationRequest("reply-2", None, "inf", "wall", canonical=view)
            )

    _within(scenario())
    assert [kind for kind, _ in wire.written] == ["response.create"]


# The server confirms the pushed frame 0.4 s after it went out.
CONFIRMED_LATER = [
    {"conn": 1, "dir": "open", "at": 0.0, "msg": None},
    c2s(SESSION_UPDATE),
    push(1, 1, 1),
    item_done(1, None) | {"at": 0.4},
    {"conn": 1, "dir": "close", "at": None, "msg": {"by": "client", "code": 1000}},
]


def test_a_push_tells_when_its_frame_went_out_and_when_the_server_confirmed_it():
    """The receipt's times are the driver clock's monotonic readings (decision N43): the frame
    was written at 0 s, and its conversation.item.done came at 0.4 s."""
    clock, table = FakeClock(), FrameTable()
    answer = AnswerRequirements("text_json", "standard_vlm")
    seed = SessionSeed(SYSTEM_INSTRUCTION, MODEL_KEY, None, 1.0, answer)

    async def scenario(driver):
        await driver.connect(ConnectRequest(seed=seed))
        observation = _observation(1)
        record = table.register(observation.observation_id, 0.0, {}, None)
        receipt = await driver.push_frame(
            observation, streaming_view([frame_block(record, observation)], frames=1)
        )
        await driver.close()
        return receipt

    with CassetteServer(CONFIRMED_LATER, clock) as server:
        driver = driver_for(server.port, clock, UNTRACED, streaming=True)
        driver._ws_connection = clocked_connection(driver._ws_connection, clock)
        receipt = run_on_fake_clock(clock, scenario(driver))
    assert server.finish() == []
    assert receipt == PushReceipt(written_at=0.0, confirmed_at=0.4)
