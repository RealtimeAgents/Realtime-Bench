"""One receiver per OpenAI Realtime connection: an in-place edit made while a response streams
loses neither the response's events nor the edit's acknowledgements.

The driver reaches a local WebSocket stub (tests/agent_trace/stubs.py) through the OpenAI SDK.
The stub holds the second response after its first delta and sends the rest only as the edit
arrives: the delete's acknowledgement is followed by a delta, the create's by the last delta and
response.done. Two readers on the connection would race for these messages, and websockets
refuses a second concurrent recv outright.
"""

from __future__ import annotations

import asyncio
import json

from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.events import GenerationBoundary, TextDelta, UsageSnapshot
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationRequest,
    HistorySyncRequest,
    SessionSeed,
    SyncDisposition,
)
from realtime_framework.entity.records import UNTRACED
from realtime_framework.providers.openai_realtime import OpenAIRealtimeDriver
from realtime_framework.providers.openai_realtime_session import Delivery
from tests.agent_trace.stubs import WsStub

from .test_openai_realtime_session import ANSWER, turn, view

USAGE = {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}


def _server():
    """Answer like the Realtime API. The second response stops after its first delta; the edit's
    delete and create each bring more of it after their acknowledgement."""
    responses: list[str] = []

    def text(piece: str) -> dict:
        return {"type": "response.output_text.delta", "response_id": responses[-1], "delta": piece}

    def end() -> list[dict]:
        response_id = responses[-1]
        item = {"id": f"srv_item_{len(responses)}", "type": "message", "role": "assistant"}
        response = {"id": response_id, "status": "completed", "output": [item], "usage": USAGE}
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
            if len(responses) == 1:
                return [*started, text('{"action": "wait"}'), *end()]
            return [*started, text('{"action"')]
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


async def _scenario(driver: OpenAIRealtimeDriver):
    """Two decisions; the first screenshot folds in place during the second response."""
    seed = SessionSeed("system instruction", "gpt-realtime-replay", None, 1.0, ANSWER)
    await driver.connect(ConnectRequest(seed=seed))
    turns = [turn(0)]

    async def generate(k: int):
        history = view(turns)
        await driver.send_observation(turns[-1].observation, history)
        request = GenerationRequest(f"turn-{k}", None, "inf", "wall", canonical=history)
        return driver.events(await driver.start_generation(request))

    try:
        first = [event async for event in await generate(0)]
        turns[0].output = "".join(e.text for e in first if isinstance(e, TextDelta))
        turns.append(turn(1))
        stream = await generate(1)
        received = [await anext(stream), await anext(stream)]
        turns[0].fold_visual()
        edit = asyncio.create_task(
            driver.sync_history(HistorySyncRequest(seed, "history_block_compaction", view(turns)))
        )
        received += [event async for event in stream]
        return received, await edit, driver._session.core
    finally:
        await driver.close()


def test_an_edit_during_a_response_keeps_both():
    with WsStub(_server()) as stub:
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
        received, sync, core = asyncio.run(_scenario(driver))
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
    assert "".join(e.text for e in received if isinstance(e, TextDelta)) == '{"action": "wait"}'
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
