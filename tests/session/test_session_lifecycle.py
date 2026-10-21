"""The live encoders ready and abort their own session (providers/session_lifecycle.py), on fake
connections, without a WebSocket stub.

The first tests drive SessionLifecycle on an encoder whose connection operations only log, on the
execution thread and the fake clock. The others drive the two live drivers on fake SDK
connections: OpenAI Realtime's RealtimeConnection, whose client events are logged, and Gemini
Live's session over a scripted socket (test_gemini_live_receive.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from types import SimpleNamespace

import pytest

from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.events import ProviderHistoryState
from realtime_framework.entity.model import (
    ConnectRequest,
    GenerationHandle,
    GenerationRequest,
    HistorySyncRequest,
    HistorySyncResult,
    SessionSeed,
    SyncDisposition,
)
from realtime_framework.entity.records import UNTRACED
from realtime_framework.harnesses.default.execution_thread import ExecutionThread
from realtime_framework.providers.openai_realtime import OpenAIRealtimeDriver
from realtime_framework.providers.session_lifecycle import SessionLifecycle
from tests.fake_clock import FakeClock

from .test_gemini_live_receive import HANG, TURN_COMPLETE, Rig, drain, frame_turn
from .test_openai_realtime_session import ANSWER, turn, view

SEED = SessionSeed("system instruction", "replay-model", None, 1.0, ANSWER)
HISTORY = object()  # the history view; the logging encoder only passes it on
GENERATION = GenerationHandle("turn-0")
EXACT = HistorySyncResult(SyncDisposition.REBUILT, ProviderHistoryState.EXACT, "rebuilt")


class _Encoder(SessionLifecycle):
    """Connection operations that log what they are asked, at the clock's time. ``syncs`` holds
    what each sync returns, or raises; an interrupt takes ``interrupt_s`` and then raises
    ``interrupt_error``, if set."""

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.log: list[tuple] = []
        self.syncs: list = []
        self.interrupt_s = 0.0
        self.interrupt_error: Exception | None = None

    async def _connect(self, request):
        self.log.append(("connect", request.history, self.clock.monotonic()))

    async def _sync_history(self, request):
        self.log.append(("sync", request.reason, request.force_rebuild, request.history))
        outcome = self.syncs.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def _close_current(self):
        self.log.append(("close", self.clock.monotonic()))

    async def interrupt(self, generation):
        self.log.append(("interrupt", generation.generation_id))
        await self.clock.sleep_async(self.interrupt_s)
        if self.interrupt_error is not None:
            raise self.interrupt_error


class _Records:
    """The ``record`` callback of ensure_ready: each call's arguments."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, sync, result=None, error=None):
        self.calls.append((sync, result, error))


class _Rig:
    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.encoder = _Encoder(clock)
        self.records = _Records()
        self.thread = ExecutionThread(clock, "lifecycle-test")

    def run(self, coroutine):
        return self.thread.run(coroutine, timeout=60.0)

    def ensure_ready(self):
        return self.encoder.ensure_ready(SEED, HISTORY, self.records)


@contextlib.contextmanager
def _rig():
    """An _Encoder that connect() opened, with its execution thread on the fake clock."""
    clock = FakeClock()
    with clock.participant():
        rig = _Rig(clock)
        rig.thread.start()
        try:
            rig.run(rig.encoder.connect(ConnectRequest(seed=SEED)))
            rig.encoder.log.clear()
            yield rig
        finally:
            rig.thread.stop()


def test_a_ready_session_is_left_alone():
    with _rig() as rig:
        assert rig.run(rig.ensure_ready()) is None
    assert rig.encoder.log == [] and rig.records.calls == []


def test_the_next_use_reopens_the_session_once_the_abort_is_done():
    """The abort is submitted first and its interrupt takes 1 s: the next ensure_ready, submitted
    right after it, reconnects only once the abort has closed the connection."""
    with _rig() as rig:
        rig.encoder.interrupt_s = 1.0
        abort = rig.thread.submit(rig.encoder.abort(GENERATION))
        assert rig.run(rig.ensure_ready()) is None
        assert abort.done()
    assert rig.encoder.log == [
        ("interrupt", "turn-0"),
        ("close", 1.0),
        ("connect", HISTORY, 1.0),
    ]
    assert rig.records.calls == [(None, None, None)]


@pytest.mark.parametrize(
    ("generation", "error", "log"),
    [
        (None, None, [("close", 0.0)]),
        (GENERATION, RuntimeError("cancel refused"), [("interrupt", "turn-0"), ("close", 0.0)]),
    ],
    ids=["before-the-request", "interrupt-fails"],
)
def test_an_abort_closes_the_session_whatever_the_interrupt_does(generation, error, log):
    with _rig() as rig:
        rig.encoder.interrupt_error = error
        rig.run(rig.encoder.abort(generation))
        assert rig.encoder.log == log
        rig.run(rig.ensure_ready())
    assert rig.encoder.log[-1] == ("connect", HISTORY, 0.0)


def test_close_waits_for_the_abort():
    with _rig() as rig:
        rig.encoder.interrupt_s = 1.0
        rig.thread.submit(rig.encoder.abort(GENERATION))
        rig.run(rig.encoder.close())
    assert rig.encoder.log == [("interrupt", "turn-0"), ("close", 1.0), ("close", 1.0)]


def test_a_dirty_session_is_rebuilt_by_a_recovery_sync():
    """A failed generation leaves the session dirty; the recovery sync rebuilds it from the
    history and returns what the sync did, which ``record`` saw first."""
    with _rig() as rig:
        rig.encoder.mark_dirty()
        rig.encoder.syncs = [EXACT]
        assert rig.run(rig.ensure_ready()) is EXACT
        assert rig.run(rig.ensure_ready()) is None  # exact: ready again
    assert rig.encoder.log == [("sync", "runtime_recovery", True, HISTORY)]
    ((sync, result, error),) = rig.records.calls
    assert sync == HistorySyncRequest(SEED, "runtime_recovery", HISTORY, force_rebuild=True)
    assert (result, error) == (EXACT, None)


def test_a_recovery_that_raises_is_recorded_and_the_session_reopened_next():
    failure = ConnectionError("dropped")
    with _rig() as rig:
        rig.encoder.mark_dirty()
        rig.encoder.syncs = [failure]
        with pytest.raises(ConnectionError) as raised:
            rig.run(rig.ensure_ready())
        assert raised.value is failure
        rig.run(rig.ensure_ready())
    assert rig.encoder.log[-1] == ("connect", HISTORY, 0.0)
    (sync, result, error), reopened = rig.records.calls
    assert (sync.reason, result, error) == ("runtime_recovery", None, failure)
    assert reopened == (None, None, None)


@pytest.mark.parametrize(
    ("disposition", "state", "next_use"),
    [
        (SyncDisposition.EXACT, ProviderHistoryState.EXACT, None),
        (SyncDisposition.BROKEN, ProviderHistoryState.UNKNOWN, "connect"),
        (SyncDisposition.UNSUPPORTED, ProviderHistoryState.UNKNOWN, "connect"),
        (SyncDisposition.REBUILT, ProviderHistoryState.CONTAMINATED, "sync"),
        (SyncDisposition.DEFERRED, ProviderHistoryState.UNKNOWN, "sync"),
    ],
)
def test_what_a_sync_leaves_decides_the_next_use(disposition, state, next_use):
    """As the runtime maps a sync's result (harnesses/default/context.py:377-384): exact is
    ready, broken or unsupported is closed, anything else is dirty."""
    with _rig() as rig:
        rig.encoder.syncs = [HistorySyncResult(disposition, state), EXACT]
        request = HistorySyncRequest(SEED, "history_block_compaction", HISTORY)
        rig.run(rig.encoder.sync_history(request))
        rig.encoder.log.clear()
        rig.run(rig.ensure_ready())
    assert [entry[0] for entry in rig.encoder.log] == ([next_use] if next_use else [])


# ---- OpenAI Realtime on a fake SDK connection ---------------------------------------------


class _FakeRealtime:
    """client.realtime of the OpenAI SDK: each connect opens a connection, numbered from 1, that
    logs each client event the driver sends as (connection, type, arguments), and its close."""

    def __init__(self):
        self.log: list[tuple] = []
        self._numbers = itertools.count(1)

    def connect(self, *, model, websocket_connection_options):
        return _FakeConnectionManager(next(self._numbers), self.log)


class _FakeConnectionManager:
    def __init__(self, number: int, log: list):
        self.number, self.log = number, log

    def __enter__(self):
        number, log = self.number, self.log

        def client_event(kind):
            return lambda **arguments: log.append((number, kind, arguments))

        return SimpleNamespace(
            _connection=SimpleNamespace(trace_connection_number=number),
            session=SimpleNamespace(update=client_event("session.update")),
            conversation=SimpleNamespace(
                item=SimpleNamespace(
                    create=client_event("conversation.item.create"),
                    delete=client_event("conversation.item.delete"),
                )
            ),
            response=SimpleNamespace(
                create=client_event("response.create"), cancel=client_event("response.cancel")
            ),
        )

    def __exit__(self, *exc):
        self.log.append((self.number, "close", {}))


def _openai() -> tuple[OpenAIRealtimeDriver, list]:
    driver = OpenAIRealtimeDriver(
        ModelConfig(
            adapter_type="openai_realtime",
            model_name="gpt-realtime-replay",
            api_base_url="http://127.0.0.1:9/v1",  # never connected to
            api_key="sk-lifecycle",
        )
    )
    driver.bind_trace(UNTRACED)
    realtime = _FakeRealtime()
    driver.client = SimpleNamespace(realtime=realtime)
    return driver, realtime.log


def _sent(log) -> list[tuple]:
    return [(number, kind) for number, kind, _ in log]


def test_openai_realtime_abort_cancels_the_response_and_the_next_use_replays():
    driver, log = _openai()
    records = _Records()
    turns = [turn(0)]

    async def run():
        await driver.connect(ConnectRequest(seed=SEED))
        await driver.send_observation(turns[0].observation, view(turns))
        request = GenerationRequest("turn-0", None, "inf", "wall", canonical=view(turns))
        await driver.abort(await driver.start_generation(request))
        # The decision was discarded: the session reopens with the history before it.
        await driver.ensure_ready(SEED, view([turn(0, "a0")]), records)

    asyncio.run(run())
    assert _sent(log) == [
        (1, "session.update"),
        (1, "conversation.item.create"),
        (1, "response.create"),
        (1, "response.cancel"),
        (1, "close"),
        (2, "session.update"),
        (2, "conversation.item.create"),
        (2, "conversation.item.create"),
    ]
    assert log[3][2]["event_id"].startswith("rtb_cancel_")
    assert [entry[2]["item"]["role"] for entry in log[-2:]] == ["user", "assistant"]
    assert records.calls == [(None, None, None)]


def test_openai_realtime_rebuilds_a_dirty_session():
    driver, log = _openai()
    records = _Records()

    async def run():
        await driver.connect(ConnectRequest(seed=SEED))
        assert await driver.ensure_ready(SEED, view([]), records) is None
        driver.mark_dirty()
        return await driver.ensure_ready(SEED, view([turn(0, "a0")]), records)

    result = asyncio.run(run())
    assert (result.disposition, result.history_state, result.detail) == (
        SyncDisposition.REBUILT,
        ProviderHistoryState.EXACT,
        "runtime_recovery",
    )
    assert _sent(log) == [
        (1, "session.update"),
        (1, "close"),
        (2, "session.update"),
        (2, "conversation.item.create"),
        (2, "conversation.item.create"),
    ]
    ((sync, recorded, _),) = records.calls
    assert (sync.reason, sync.force_rebuild, recorded) == ("runtime_recovery", True, result)


# ---- Gemini Live on google-genai's session over a scripted socket ---------------------------


@pytest.mark.parametrize(
    ("model", "interrupt"),
    [
        # client_content: the interrupt closes the connection.
        ("gemini-3.8-live", []),
        # realtime_input (the driver defaults): it opens an activity, then the abort closes.
        ("replay-gemini-live", ["realtime_input"]),
    ],
    ids=["client-content", "realtime-input"],
)
def test_gemini_live_abort_closes_and_the_next_use_replays(model, interrupt):
    rig = Rig(model, [[HANG], []])
    records = _Records()

    async def run():
        await rig.connect()
        await rig.driver.abort(await rig.start("turn-0"))
        assert rig.driver._session is None
        await rig.driver.ensure_ready(rig.seed, rig.view([frame_turn(0)]), records)

    asyncio.run(run())
    first, second = rig.live.sockets
    assert [next(iter(message)) for message in first.sent] == interrupt
    assert [next(iter(message)) for message in second.sent] == ["client_content"]  # the seed
    assert rig.log == [("connect", None), ("sleep", 2.0), ("connect", None)]
    assert records.calls == [(None, None, None)]


def test_gemini_live_rebuilds_a_dirty_session():
    """A recovery replays the history even where a resumption handle would have done."""
    update = {"sessionResumptionUpdate": {"newHandle": "handle-0", "resumable": True}}
    rig = Rig("gemini-3.8-live", [[update, TURN_COMPLETE], []])
    records = _Records()

    async def run():
        await rig.connect()
        await drain(rig.driver.events(await rig.start("turn-0")))
        assert rig.driver._core.resume_handle == "handle-0"
        rig.driver.mark_dirty()
        return await rig.driver.ensure_ready(rig.seed, rig.view([frame_turn(0)]), records)

    result = asyncio.run(run())
    assert (result.disposition, result.history_state, result.detail) == (
        SyncDisposition.REBUILT,
        ProviderHistoryState.EXACT,
        "runtime_recovery",
    )
    assert result.provider_metadata["session_connect_mode"] == "canonical_history_replay"
    assert rig.log == [("connect", None), ("sleep", 2.0), ("connect", None)]
    ((sync, recorded, _),) = records.calls
    assert (sync.reason, sync.force_rebuild, recorded) == ("runtime_recovery", True, result)
