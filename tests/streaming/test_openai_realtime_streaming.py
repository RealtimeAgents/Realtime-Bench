"""The OpenAI Realtime driver in native streaming (stream-plan 2.4): its reader reads every event
as it arrives, and the session takes pushed frames and replies.

The reader tests drive a session on the fake connection of tests/session/test_openai_realtime_receive.py,
which counts the reads in flight. Streaming views come from the streaming history view
(harnesses/streaming/history.py), as the session core tests build them.
"""

from __future__ import annotations

import asyncio
import itertools

import pytest
from websockets.exceptions import ConnectionClosedOK

from realtime_framework.entity.events import (
    GenerationBoundary,
    ProviderHistoryState,
    TextDelta,
    TextDone,
    UsageSnapshot,
)
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationRequest,
    HistorySyncRequest,
    HistorySyncResult,
    SessionSeed,
    SyncDisposition,
)
from realtime_framework.entity.records import UNTRACED
from realtime_framework.entity.request import AnswerRequirements
from realtime_framework.harnesses.default.frames import FrameTable
from realtime_framework.harnesses.streaming.history import frame_block
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
        return connection, session, session._reader.done()

    connection, session, ended = _within(scenario())
    assert ended
    assert connection.in_flight == 0
    assert session.closed is not None and session.failure is None


class _FailingManager:
    """The SDK's connection manager whose close raises, which may leave the connection open."""

    def __exit__(self, *exc_info) -> None:
        raise RuntimeError("close failed")


def test_a_failed_close_cancels_the_reader_and_ends_its_waiters():
    """A close that raises may leave the connection open, so the reader is cancelled instead of
    waited for; whoever waits on the session then raises rather than waiting for good."""

    async def scenario():
        driver, connection = _offline_driver(), _FakeConnection()
        session = _Session(1, connection, new_core())
        session.read_continuously(Ticks())
        reply = session.core.start_reply()
        waiter = asyncio.create_task(session.receive_until(lambda: reply.events))
        driver._session, driver._cm = session, _FailingManager()
        await until(lambda: connection.in_flight == 1)
        await driver._close_current()
        cancelled = session._reader.cancelled()
        raised = await asyncio.gather(waiter, return_exceptions=True)
        connection.queue.put(_clean_close())  # ends the read the cancel left blocked
        return cancelled, raised

    cancelled, (raised,) = _within(scenario())
    assert cancelled
    assert isinstance(raised, ConnectionError)


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
    driver._session, driver._conn = session, object()
    driver._send = lambda operation: connection.queue.put(server.answer(operation))
    replay = session.core.replay_streaming(streaming_view(list(blocks), frames=5))
    await driver._send_all(replay)
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


def test_a_connection_that_closes_before_the_frame_is_confirmed_fails_the_push():
    async def scenario():
        driver, connection = await answering(Server(), stream("F"))
        driver._send = lambda operation: connection.queue.put(_clean_close())
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


class _Responses:
    """Stands in for the SDK connection's response resource: keeps each response.create."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.response = self

    def create(self, **kwargs) -> None:
        self.created.append(kwargs)


def test_a_reply_goes_out_after_the_feedback_the_session_lacks():
    """Feedback accepted after the newest frame, an item of its own under separate_message, is
    pushed before the reply's response.create, so the reply sees it; the reply answers no frame."""
    driver, core = _offline_driver(), new_core()
    sent(core, core.push(streaming_view(stream("FR"))))
    driver._session = _Session(1, _FakeConnection(), core)
    driver._conn = responses = _Responses()
    pushed = []
    driver._send = pushed.append
    request = GenerationRequest(
        "reply-2", None, "inf", "wall", canonical=streaming_view(stream("FRB"))
    )
    handle = _within(driver.request_reply(request))
    assert [(op.block, op.text) for op in pushed] == [("d1.feedback", "fb1")]
    assert responses.created == [
        {"response": {"max_output_tokens": "inf", "reasoning": {"effort": "low"}}}
    ]
    _, reply = handle.opaque
    assert reply.answered is None
