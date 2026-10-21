"""One receiver per OpenAI Realtime connection: an in-place edit made while a response streams
loses neither the response's events nor the edit's acknowledgements.

Most tests reach a local WebSocket stub (tests/agent_trace/stubs.py) through the OpenAI SDK. The
stub can hold a response after its first delta and send the rest only as an edit arrives: the
delete's acknowledgement is followed by a delta, the create's by the last delta and
response.done. Before the session core split the edit wait and the response each read the
connection, and websockets refused the second concurrent recv. The tests at the end drive the
receiver on a fake connection, which counts the reads in flight: one read for every waiter, a
clean close, a failed read.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading

import pytest
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.events import (
    GenerationBoundary,
    ProviderError,
    ProviderHistoryState,
    TextDelta,
    UsageSnapshot,
)
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationHandle,
    GenerationRequest,
    HistorySyncRequest,
    SessionSeed,
    SyncDisposition,
)
from realtime_framework.entity.records import UNTRACED
from realtime_framework.providers.openai_realtime import OpenAIRealtimeDriver, _Session
from realtime_framework.providers.openai_realtime_session import Delivery
from tests.agent_trace.stubs import WsStub

from .test_openai_realtime_session import (
    ANSWER,
    deleted,
    done,
    error,
    opened,
    response_events,
    turn,
    view,
)

SEED = SessionSeed("system instruction", "gpt-realtime-replay", None, 1.0, ANSWER)
USAGE = {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}


def _server(held: int | None = None, status: str = "completed"):
    """Answer like the Realtime API. Response ``held`` stops after its first delta; an edit's
    delete and create each bring more of it after their acknowledgement, and response.cancel ends
    it as cancelled. The others end with ``status``."""
    responses: list[str] = []

    def text(piece: str) -> dict:
        return {"type": "response.output_text.delta", "response_id": responses[-1], "delta": piece}

    def end(status: str = status) -> list[dict]:
        response_id = responses[-1]
        item = {"id": f"srv_item_{len(responses)}", "type": "message", "role": "assistant"}
        response = {"id": response_id, "status": status, "output": [item], "usage": USAGE}
        return [
            {"type": "response.output_item.done", "response_id": response_id, "item": item},
            {"type": "response.done", "response": response},
        ]

    def handle(stub, connection, message):
        if message is None:
            return [{"type": "session.created", "session": {}}]
        kind = message["type"]
        if kind == "session.update":
            return [{"type": "session.updated", "session": {}}]
        if kind == "response.create":
            responses.append(f"resp_{len(responses) + 1}")
            item = {"id": f"srv_item_{len(responses)}"}
            started = [
                {"type": "response.created", "response": {"id": responses[-1]}},
                {"type": "response.output_item.added", "response_id": responses[-1], "item": item},
            ]
            if len(responses) == held:
                return [*started, text('{"action"')]
            return [*started, text('{"action": "wait"}'), *end()]
        if kind == "response.cancel":
            return end("cancelled")
        if kind == "conversation.item.delete":
            deleted = {"type": "conversation.item.deleted", "item_id": message["item_id"]}
            # Only the in-place edit names its client events.
            return [deleted, text(': "wait"')] if "event_id" in message else [deleted]
        if kind == "conversation.item.create":
            item = {"id": message["item"]["id"]}
            acks = [
                {"type": "conversation.item.added", "item": item},
                {"type": "conversation.item.done", "item": item},
            ]
            # Only the in-place edit names its client events.
            return [*acks, text("}"), *end()] if "event_id" in message else acks
        return []

    return handle


def _driver(stub: WsStub) -> OpenAIRealtimeDriver:
    driver = OpenAIRealtimeDriver(
        ModelConfig(
            adapter_type="openai_realtime",
            model_name="gpt-realtime-replay",
            api_base_url=f"http://127.0.0.1:{stub.port}/v1",
            api_key="sk-receive-stub",
        )
    )
    assert driver.client.base_url.host == "127.0.0.1"
    driver.bind_trace(UNTRACED)
    return driver


async def _start(driver: OpenAIRealtimeDriver, turns):
    """Send the newest turn's observation and start the response to it."""
    history = view(turns)
    await driver.send_observation(turns[-1].observation, history)
    request = GenerationRequest(f"turn-{len(turns)}", None, "inf", "wall", canonical=history)
    return await driver.start_generation(request)


async def _generate(driver: OpenAIRealtimeDriver, turns):
    """The event stream of the response to the newest turn."""
    return driver.events(await _start(driver, turns))


def _text(events) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


def _within(coroutine, timeout: float = 30.0):
    """Run the coroutine, failing instead of hanging if it never ends."""
    return asyncio.run(asyncio.wait_for(coroutine, timeout))


async def _edit_during_response(driver: OpenAIRealtimeDriver):
    """Two decisions; the first screenshot folds in place during the second response."""
    await driver.connect(ConnectRequest(seed=SEED))
    turns = [turn(0)]
    try:
        turns[0].output = _text([event async for event in await _generate(driver, turns)])
        turns.append(turn(1))
        stream = await _generate(driver, turns)
        received = [await anext(stream), await anext(stream)]
        turns[0].fold_visual()
        edit = asyncio.create_task(
            driver.sync_history(HistorySyncRequest(SEED, "history_block_compaction", view(turns)))
        )
        received += [event async for event in stream]
        return received, await edit, driver._session.core
    finally:
        await driver.close()


def test_an_edit_during_a_response_keeps_both():
    with WsStub(_server(held=2)) as stub:
        received, sync, core = _within(_edit_during_response(_driver(stub)))
    (connection,) = stub.connections

    # The response: every delta, its usage and its end, in order.
    assert [type(e) for e in received] == [
        GenerationBoundary,
        TextDelta,
        TextDelta,
        TextDelta,
        UsageSnapshot,
        GenerationBoundary,
    ]
    assert _text(received) == '{"action": "wait"}'
    assert (received[0].kind, received[-1].kind) == ("started", "turn_complete")
    assert received[-2].usage["output_tokens"] == 5

    # The edit: confirmed in place, on the same connection.
    assert sync.disposition == SyncDisposition.EXACT
    assert sync.provider_metadata["history_items_created"] == 1
    assert sync.provider_metadata["history_items_deleted"] == 1
    sent = [json.loads(message)["type"] for message in connection.received]
    assert sent[-2:] == ["conversation.item.delete", "conversation.item.create"]
    assert core.frames["d0.frame"].folded
    assert set(core.delivery.values()) == {Delivery.ACKNOWLEDGED}
    # The response's output item joins the frame it answered, whichever of the two ended first.
    assert core.frames["d1.frame"].assistant_item_ids == ["srv_item_2"]


async def _two_connections(driver: OpenAIRealtimeDriver):
    """Two responses on one connection, then a rebuild on a second and one response there."""
    await driver.connect(ConnectRequest(seed=SEED))
    turns, responses = [], []
    try:
        for k in range(3):
            if k == 2:
                await driver.connect(ConnectRequest(seed=SEED, history=view(turns)))
            turns.append(turn(k))
            responses.append([event async for event in await _generate(driver, turns)])
            turns[-1].output = _text(responses[-1])
        return responses
    finally:
        await driver.close()


def test_events_carry_their_connection_and_generation():
    with WsStub(_server()) as stub:
        responses = _within(_two_connections(_driver(stub)))
    assert len(stub.connections) == 2
    assert [{(e.connection, e.generation) for e in events} for events in responses] == [
        {(1, 1)},
        {(1, 2)},
        {(2, 1)},
    ]
    assert all(len(events) == 4 for events in responses)  # started, text, usage, turn_complete


class _ServerConnections:
    """An agent trace that keeps the connection id of each server message the driver read."""

    def __init__(self):
        self.ids: list[str] = []

    def capture(self, event, payload, *, body=None, classify=None):
        if payload.get("direction") == "server":
            self.ids.append(payload["connection_id"])


def test_connection_numbers_follow_the_trace_after_a_refused_handshake():
    """The trace numbers every WebSocket connection it sees, a refused upgrade included
    (providers/capture.py); the events' connection numbers are the same numbers."""
    trace = _ServerConnections()
    with WsStub(_server(), reject=1) as stub:
        driver = _driver(stub)
        driver.bind_trace(trace)
        responses = _within(_two_connections(driver))
    assert stub.rejected == 1 and len(stub.connections) == 2
    assert [{(e.connection, e.generation) for e in events} for events in responses] == [
        {(2, 1)},
        {(2, 2)},
        {(3, 1)},
    ]
    assert sorted(set(trace.ids)) == ["openai_realtime.ws-2", "openai_realtime.ws-3"]


async def _interrupted(driver: OpenAIRealtimeDriver):
    """A response cancelled after its first delta, as RealtimeRuntime.abort does it."""
    await driver.connect(ConnectRequest(seed=SEED))
    try:
        generation = await _start(driver, [turn(0)])
        stream = driver.events(generation)
        received = [await anext(stream), await anext(stream)]
        result = await driver.interrupt(generation)
        received += [event async for event in stream]
        return received, result
    finally:
        await driver.close()


def test_interrupt_cancels_the_response_by_its_id():
    with WsStub(_server(held=1)) as stub:
        received, result = _within(_interrupted(_driver(stub)))
    (connection,) = stub.connections
    cancel = json.loads(connection.received[-1])
    assert cancel["type"] == "response.cancel" and cancel["response_id"] == "resp_1"
    assert cancel["event_id"].startswith("rtb_cancel_")
    assert result.history_state == ProviderHistoryState.CONTAMINATED
    assert [e.kind for e in received if isinstance(e, GenerationBoundary)] == [
        "started",
        "interrupted",
    ]
    assert _text(received) == '{"action"'


async def _repaired(driver: OpenAIRealtimeDriver):
    """An incomplete response, then the sync the runtime makes after it."""
    await driver.connect(ConnectRequest(seed=SEED))
    turns = [turn(0)]
    try:
        generation = await _start(driver, turns)
        received = [event async for event in driver.events(generation)]
        turns[0].output = _text(received)
        request = HistorySyncRequest(
            SEED, "provider_history_changed", view(turns), current_generation=generation
        )
        sync = await driver.sync_history(request)
        # The repair does not wait for its acknowledgements; the stub records a message before
        # it answers it, so once they are in, the stub has the delete and the create.
        session = driver._session
        acknowledged = {Delivery.ACKNOWLEDGED}
        await session.receive_until(lambda: set(session.core.delivery.values()) == acknowledged)
        return received, sync, session.core
    finally:
        await driver.close()


def test_an_incomplete_response_is_replaced_by_its_collected_output():
    with WsStub(_server(status="incomplete")) as stub:
        received, sync, core = _within(_repaired(_driver(stub)))
    (connection,) = stub.connections
    assert (received[-1].kind, received[-1].history_state) == (
        "interrupted",
        ProviderHistoryState.CONTAMINATED,
    )
    assert (sync.disposition, sync.provider_metadata) == (
        SyncDisposition.EXACT,
        {"deleted_output_items": 1},
    )
    delete, create = map(json.loads, connection.received[-2:])
    assert delete == {"type": "conversation.item.delete", "item_id": "srv_item_1"}
    assert set(create) == {"type", "item"} and create["item"]["role"] == "assistant"
    assert create["item"]["content"] == [{"type": "output_text", "text": '{"action": "wait"}'}]
    assert core.frames["d0.frame"].assistant_item_ids == [create["item"]["id"]]


# ---- the receiver on a fake connection ----------------------------------------------------


class _FakeConnection:
    """Stands in for the SDK connection the receiver reads: recv() hands out what the test
    queued and raises a queued exception (a ConnectionClosedOK is a clean close). It counts how
    many recv calls run at once."""

    def __init__(self):
        self.queue: queue.Queue = queue.Queue()
        self.in_flight = 0
        self.most_in_flight = 0
        self._lock = threading.Lock()

    def recv(self):
        with self._lock:
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            item = self.queue.get(timeout=5)
            if isinstance(item, Exception):
                raise item
            return item
        finally:
            with self._lock:
                self.in_flight -= 1


def _clean_close() -> ConnectionClosedOK:
    return ConnectionClosedOK(Close(1000, "bye"), Close(1000, "bye"), True)


def _waiting_edit():
    """A core with a response in progress for decision 1 and, sent, an in-place edit that folds
    decision 0."""
    turns = [turn(0, "a0"), turn(1)]
    core = opened(turns)
    generation = core.start_generation("d1.frame")
    turns[0].fold_visual()
    batch = core.begin_edit(view(turns))
    for operation in batch.operations:
        core.sending(operation)
        core.sent(operation)
    return core, generation, batch


def test_a_clean_close_during_an_edit_fails_it_with_that_close():
    """The edit's fallback records the close itself, as it did before the session core split
    (which read with conn.recv())."""
    connection, closed = _FakeConnection(), _clean_close()
    core, _, batch = _waiting_edit()
    session = _Session(1, connection, core)
    connection.queue.put(deleted(batch.operations[0].item_id))
    connection.queue.put(closed)
    with pytest.raises(ConnectionClosedOK) as raised:
        _within(OpenAIRealtimeDriver._wait_for_item_edits(session))
    assert raised.value is closed


async def _waiting(connection: _FakeConnection) -> None:
    """Return once a read is in, with time for a second one to start if anything would."""
    for _ in range(500):
        if connection.in_flight:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)


def test_two_waiters_share_one_read():
    """An edit and a response both wait before anything arrives. One recv serves both, and each
    gets its own events."""

    async def scenario():
        connection = _FakeConnection()
        core, generation, batch = _waiting_edit()
        session = _Session(1, connection, core)
        edit = asyncio.create_task(OpenAIRealtimeDriver._wait_for_item_edits(session))
        response = asyncio.create_task(session.receive_until(lambda: len(generation.events) == 4))
        await _waiting(connection)
        assert connection.in_flight == 1
        delete, create = batch.operations
        created, added, text, finished = response_events()
        for event in (
            deleted(delete.item_id),
            created,
            added,
            done(create.item_id),
            text,
            finished,
        ):
            connection.queue.put(event)
        await asyncio.gather(edit, response)
        return connection, core, generation, batch

    connection, core, generation, batch = _within(scenario())
    assert connection.most_in_flight == 1
    assert core.finish_edit() is batch
    assert [event.type for event in generation.events] == [
        "response.created",
        "response.output_item.added",
        "response.output_text.delta",
        "response.done",
    ]


def test_a_failed_read_raises_in_every_waiter():
    failure = ConnectionError("read failed")

    async def scenario():
        connection = _FakeConnection()
        core, generation, _ = _waiting_edit()
        session = _Session(1, connection, core)
        edit = asyncio.create_task(OpenAIRealtimeDriver._wait_for_item_edits(session))
        response = asyncio.create_task(session.receive_until(lambda: generation.events))
        await _waiting(connection)
        connection.queue.put(failure)
        raised = await asyncio.gather(edit, response, return_exceptions=True)
        # A waiter that comes later raises it too, without reading again.
        try:
            await session.receive_until(lambda: generation.events)
        except ConnectionError as later:
            raised.append(later)
        return raised

    assert all(raised is failure for raised in _within(scenario()))


def _offline_driver(*, streaming: bool = False) -> OpenAIRealtimeDriver:
    driver = OpenAIRealtimeDriver(
        ModelConfig(
            adapter_type="openai_realtime",
            model_name="gpt-realtime-replay",
            api_base_url="http://127.0.0.1:9/v1",  # never connected to
            api_key="sk-receive-stub",
        ),
        streaming=streaming,
    )
    driver.bind_trace(UNTRACED)
    return driver


def test_a_clean_close_mid_reply_ends_the_generation_with_a_fatal_error():
    """A connection the server closes normally before the response.done leaves a half reply: the
    generation ends with a fatal ProviderError, as a response that ends with status=failed does,
    not with a quiet return that would pass the partial text off as the whole answer."""
    connection = _FakeConnection()
    core = opened([turn(0)])
    generation = core.start_generation("d0.frame")
    session = _Session(3, connection, core)
    created, added, text, _ = response_events()
    for event in (created, added, text, _clean_close()):
        connection.queue.put(event)
    handle = GenerationHandle("turn-0", opaque=(session, generation))

    async def collect():
        return [event async for event in _offline_driver().events(handle)]

    events = _within(collect())
    assert [type(event) for event in events] == [GenerationBoundary, TextDelta, ProviderError]
    assert events[-1].fatal
    assert {(event.connection, event.generation) for event in events} == {(3, 1)}


def test_events_end_when_the_connection_closes():
    """A clean close ends the response's events; an error on the way is reported with the
    connection and generation numbers."""
    connection = _FakeConnection()
    core = opened([turn(0)])
    generation = core.start_generation("d0.frame")
    session = _Session(3, connection, core)
    created, *_ = response_events()
    for event in (created, error(None), _clean_close()):
        connection.queue.put(event)
    handle = GenerationHandle("turn-0", opaque=(session, generation))

    async def collect():
        return [event async for event in _offline_driver().events(handle)]

    events = _within(collect())
    assert [type(event) for event in events] == [GenerationBoundary, ProviderError]
    assert events[1].fatal
    assert {(event.connection, event.generation) for event in events} == {(3, 1)}


def test_an_operation_whose_send_fails_is_unknown():
    driver = _offline_driver()
    core = opened([turn(0, "a0")])
    observation = core.observe(view([turn(0, "a0"), turn(1)]))
    session = _Session(1, _FakeConnection(), core)

    def send(conn, operation):
        raise ConnectionError("send failed")

    driver._send = send
    with pytest.raises(ConnectionError):
        _within(driver._send_all(session, [observation]))
    assert core.delivery[observation.key] == Delivery.UNKNOWN


def test_a_closed_session_reports_it_is_not_connected():
    """Once abort() closed the connection, an abandoned worker that goes on is told so."""
    driver = _offline_driver()
    history = view([turn(0)])
    request = GenerationRequest("turn-0", None, "inf", "wall", canonical=history)
    for step in (
        driver.send_observation(turn(0).observation, history),
        driver.start_generation(request),
    ):
        with pytest.raises(RuntimeError, match="^OpenAI Realtime session is not connected$"):
            _within(step)
