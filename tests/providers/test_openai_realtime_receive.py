"""One receiver per OpenAI Realtime connection: an in-place edit made while a response streams
loses neither the response's events nor the edit's acknowledgements.

The driver reaches a local WebSocket stub (tests/agent_trace/stubs.py) through the OpenAI SDK.
The stub can hold a response after its first delta and send the rest only as an edit arrives:
the delete's acknowledgement is followed by a delta, the create's by the last delta and
response.done. Two readers on the connection would race for these messages, and websockets
refuses a second concurrent recv outright.
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
    ProviderHistoryState,
    TextDelta,
    UsageSnapshot,
)
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationRequest,
    HistorySyncRequest,
    SessionSeed,
    SyncDisposition,
)
from realtime_framework.entity.records import UNTRACED
from realtime_framework.providers.openai_realtime import OpenAIRealtimeDriver, _Session
from realtime_framework.providers.openai_realtime_session import Delivery
from tests.agent_trace.stubs import WsStub

from .test_openai_realtime_session import ANSWER, deleted, opened, turn, view

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
            return [deleted, text(': "wait"')]
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
        received, sync, core = asyncio.run(_edit_during_response(_driver(stub)))
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
    assert set(core.delivery.values()) == {Delivery.CONFIRMED}
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
        responses = asyncio.run(_two_connections(_driver(stub)))
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
        responses = asyncio.run(_two_connections(driver))
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
        received, result = asyncio.run(_interrupted(_driver(stub)))
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
        return received, await driver.sync_history(request), driver._session.core
    finally:
        await driver.close()


def test_an_incomplete_response_is_replaced_by_its_collected_output():
    with WsStub(_server(status="incomplete")) as stub:
        received, sync, core = asyncio.run(_repaired(_driver(stub)))
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
    return core, generation, batch


def _within(coroutine, timeout: float = 10.0):
    """Run the coroutine, failing instead of hanging if it never ends."""
    return asyncio.run(asyncio.wait_for(coroutine, timeout))


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
