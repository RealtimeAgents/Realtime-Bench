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
import threading
from types import SimpleNamespace

import pytest

from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.events import ProviderHistoryState
from realtime_framework.entity.model import (
    TRANSIENT,
    TRANSPORT_RETRY,
    ConnectRequest,
    Deployment,
    GenerationHandle,
    GenerationRequest,
    HistorySyncRequest,
    HistorySyncResult,
    InterruptResult,
    RealtimeCapabilities,
    RetryCancelled,
    RuntimeState,
    SessionSeed,
    SyncDisposition,
    TransportFailure,
    retry_cancellation,
)
from realtime_framework.entity.records import UNTRACED, TraceEvent, current_trace_scope
from realtime_framework.harnesses.default.execution_thread import ExecutionThread
from realtime_framework.providers import registry as catalog
from realtime_framework.providers.gemini import live
from realtime_framework.providers.openai_realtime import OpenAIRealtimeDriver
from realtime_framework.providers.session_lifecycle import SessionLifecycle, StatelessSession
from tests.fake_clock import FakeClock

from .test_gemini_live_receive import (
    HANG,
    TURN_COMPLETE,
    FakeLive,
    RecordingClock,
    Rig,
    drain,
    frame_turn,
)
from .test_openai_realtime_session import ANSWER, turn, view

SEED = SessionSeed("system instruction", "replay-model", None, 1.0, ANSWER)
HISTORY = object()  # the history view; the logging encoder only passes it on
GENERATION = GenerationHandle("turn-0")
EXACT = HistorySyncResult(SyncDisposition.REBUILT, ProviderHistoryState.EXACT, "rebuilt")


class _Encoder(SessionLifecycle):
    """Connection operations that log what they are asked, at the clock's time. ``syncs`` holds
    what each sync returns, or raises; a connect takes ``connect_s``; an interrupt takes
    ``interrupt_s`` and then raises ``interrupt_error``, if set; a close raises ``close_error``
    and a connect ``connect_error``, if set. ``during`` runs in the middle of each connect and
    sync. ``renewal`` is what connection_renewal_reason returns."""

    capabilities = RealtimeCapabilities(
        "replay_session", persistent_session=True, connection_renewal=True
    )

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self._trace = UNTRACED
        self.log: list[tuple] = []
        self.syncs: list = []
        self.connect_s = 0.0
        self.interrupt_s = 0.0
        self.interrupt_error: Exception | None = None
        self.close_error: Exception | None = None
        self.connect_error: Exception | None = None
        self.during = lambda: None
        self.renewal: str | None = None

    async def _connect(self, request):
        self.log.append(("connect", request.history, self.clock.monotonic()))
        await self.clock.sleep_async(self.connect_s)
        self.during()
        if self.connect_error is not None:
            raise self.connect_error

    async def _sync_history(self, request):
        self.log.append(("sync", request.reason, request.force_rebuild, request.history))
        self.during()
        outcome = self.syncs.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def connection_renewal_reason(self):
        return self.renewal

    async def _close_current(self):
        self.log.append(("close", self.clock.monotonic()))
        if self.close_error is not None:
            raise self.close_error

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
def _rig(*, connected: bool = True):
    """An _Encoder, which connect() opened unless not ``connected``, with its execution thread
    on the fake clock."""
    clock = FakeClock()
    with clock.participant():
        rig = _Rig(clock)
        rig.thread.start()
        try:
            if connected:
                rig.run(rig.encoder.connect(ConnectRequest(seed=SEED)))
                rig.encoder.log.clear()
            yield rig
        finally:
            rig.thread.stop()


def test_a_ready_session_is_left_alone():
    with _rig() as rig:
        assert rig.run(rig.ensure_ready()) is None
    assert rig.encoder.log == [] and rig.records.calls == []


@pytest.mark.parametrize("before", ["never-connected", "closed"])
def test_a_session_not_open_is_opened(before):
    with _rig(connected=before == "closed") as rig:
        if before == "closed":
            rig.run(rig.encoder.close())
        rig.encoder.log.clear()
        rig.run(rig.ensure_ready())
    assert rig.encoder.log == [("connect", HISTORY, 0.0)]
    assert rig.records.calls == [(None, None, None)]


def test_a_failed_reopen_raises_unrecorded():
    with _rig() as rig:
        rig.run(rig.encoder.abort(None))
        rig.encoder.connect_error = ConnectionError("refused")
        with pytest.raises(ConnectionError, match="^refused$"):
            rig.run(rig.ensure_ready())
    assert rig.records.calls == []


def test_a_reopened_session_is_clean():
    """A session that failed and was then aborted is reopened, not rebuilt again after that."""
    with _rig() as rig:
        rig.encoder.mark_dirty()
        rig.run(rig.encoder.abort(None))
        rig.run(rig.ensure_ready())
        assert rig.run(rig.ensure_ready()) is None
    assert [entry[0] for entry in rig.encoder.log] == ["close", "connect"]


def test_an_abort_its_caller_gives_up_on_still_cleans_up():
    """The cleanup runs on its own: cancelling the abort that started it does not stop it, and
    the next ensure_ready waits for it."""
    with _rig() as rig:
        rig.encoder.interrupt_s = 5.0
        with pytest.raises(TimeoutError):
            rig.thread.run(rig.encoder.abort(GENERATION), timeout=2.0)
        rig.run(rig.ensure_ready())
    assert rig.encoder.log == [
        ("interrupt", "turn-0"),
        ("close", 5.0),
        ("connect", HISTORY, 5.0),
    ]


def test_a_second_abort_cleans_up_once_the_first_is_done():
    """An abort whose caller gave up on it goes on (its interrupt takes 5 s), and a second abort
    comes meanwhile: its cleanup starts once the first one's is done, so that the next
    ensure_ready, which waits for the last cleanup, reopens the session after both, and the
    first one's close does not close what it opened (review-encoders 低 15)."""
    with _rig() as rig:
        rig.encoder.interrupt_s = 5.0
        with pytest.raises(TimeoutError):
            rig.thread.run(rig.encoder.abort(GENERATION), timeout=2.0)
        rig.run(rig.encoder.abort(None))
        rig.run(rig.ensure_ready())
        rig.run(rig.clock.sleep_async(10.0))
        state = rig.encoder.state
    assert rig.encoder.log == [
        ("interrupt", "turn-0"),
        ("close", 5.0),
        ("close", 5.0),
        ("connect", HISTORY, 5.0),
    ]
    assert state == RuntimeState.READY


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
    ("generation", "interrupt_error", "close_error", "log"),
    [
        (None, None, None, [("close", 0.0)]),
        (
            GENERATION,
            RuntimeError("cancel refused"),
            None,
            [("interrupt", "turn-0"), ("close", 0.0)],
        ),
        (GENERATION, None, RuntimeError("close failed"), [("interrupt", "turn-0"), ("close", 0.0)]),
    ],
    ids=["before-the-request", "interrupt-fails", "close-fails"],
)
def test_an_abort_leaves_the_session_closed_whatever_fails(
    generation, interrupt_error, close_error, log
):
    with _rig() as rig:
        rig.encoder.interrupt_error = interrupt_error
        rig.encoder.close_error = close_error
        rig.run(rig.encoder.abort(generation))
        assert rig.encoder.log == log
        rig.encoder.close_error = None
        rig.run(rig.ensure_ready())
    assert rig.encoder.log[-1] == ("connect", HISTORY, 0.0)


@pytest.mark.parametrize(
    ("interrupt_s", "log"),
    [
        (1.0, [("interrupt", "turn-0"), ("close", 1.0), ("close", 1.0)]),
        # A cleanup still running after 10 s is cancelled, and close closes all the same.
        (60.0, [("interrupt", "turn-0"), ("close", 10.0)]),
    ],
    ids=["cleanup-done", "cleanup-hangs"],
)
def test_close_waits_for_the_abort_for_a_while(interrupt_s, log):
    with _rig() as rig:
        rig.encoder.interrupt_s = interrupt_s
        abort = rig.thread.submit(rig.encoder.abort(GENERATION))
        rig.run(rig.encoder.close())
        assert rig.clock.wait(abort, 1.0)
        assert rig.clock.monotonic() == min(interrupt_s, 10.0)
    assert rig.encoder.log == log


def test_a_close_its_caller_gives_up_on_still_closes():
    """A caller that gives close less time than close waits for a stuck abort: cancelled while it
    waits, close still closes the connection."""
    with _rig() as rig:
        rig.encoder.interrupt_s = 60.0
        abort = rig.thread.submit(rig.encoder.abort(GENERATION))
        with pytest.raises(TimeoutError):
            rig.thread.run(rig.encoder.close(), timeout=5.0)
        rig.run(asyncio.sleep(0))  # the loop takes the cancellation first
        assert rig.encoder.log == [("interrupt", "turn-0"), ("close", 5.0)]
        assert not rig.encoder._connected
        assert rig.clock.wait(abort, 100.0)  # the abort ends when its interrupt does


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
    """As the runtime mapped a sync's result before the session moved into the encoder: exact is
    ready, broken or unsupported is closed, anything else is dirty."""
    with _rig() as rig:
        rig.encoder.syncs = [HistorySyncResult(disposition, state), EXACT]
        request = HistorySyncRequest(SEED, "history_block_compaction", HISTORY)
        rig.run(rig.encoder.sync_history(request))
        rig.encoder.log.clear()
        rig.run(rig.ensure_ready())
    assert [entry[0] for entry in rig.encoder.log] == ([next_use] if next_use else [])


@pytest.mark.parametrize(
    "interruption",
    [RetryCancelled("generation abandoned"), asyncio.CancelledError()],
    ids=["retry-cancelled", "task-cancelled"],
)
def test_a_sync_cut_short_leaves_the_session_to_be_recovered(interruption):
    """A sync may close the connection before an abandoned generation's cancellation cuts it
    short (Gemini's resume does). The session is left dirty then, as the runtime was left
    SYNCHRONIZING before the move, so its next use rebuilds the session."""
    with _rig() as rig:
        rig.encoder.syncs = [interruption, EXACT]

        async def interrupted_sync():
            try:
                await rig.encoder.sync_history(HistorySyncRequest(SEED, "go_away", HISTORY))
            except BaseException as exc:  # noqa: BLE001 - the interruption under test
                assert exc is interruption
            return await rig.ensure_ready()

        assert rig.run(interrupted_sync()) is EXACT
    assert [entry[:2] for entry in rig.encoder.log] == [
        ("sync", "go_away"),
        ("sync", "runtime_recovery"),
    ]


def test_a_wait_runs_on_the_clock_and_its_generation_s_cancel_ends_it():
    with _rig() as rig:
        rig.run(rig.encoder._wait("session_release", 2.0))
        assert rig.clock.monotonic() == 2.0
        cancel = rig.clock.event()
        with retry_cancellation(cancel):
            waiting = rig.thread.submit(rig.encoder._wait("session_release", 2.0))
        rig.clock.sleep(0.5)
        cancel.set()
        assert rig.clock.wait(waiting, 10.0)
        with pytest.raises(RetryCancelled, match="during the session_release wait$"):
            waiting.result()
        assert rig.clock.monotonic() == 2.5


# ---- what the runtime asks of the session around each generation --------------------------

CONTAMINATED = HistorySyncResult(
    SyncDisposition.REBUILT, ProviderHistoryState.CONTAMINATED, "replayed"
)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("recovery", "provider history recovery failed: replayed"),
        ("synchronize", "history synchronization failed: replayed"),
        ("renew", "connection renewal failed: replayed"),
    ],
)
def test_a_session_that_must_come_out_exact_raises_otherwise(operation, message):
    """The three operations before a generation's input raise, with the runtime's words, when
    their sync leaves the session other than exact (harnesses/default/context.py before the
    session moved into the encoder). Each sync is recorded first."""
    with _rig() as rig:
        rig.encoder.syncs = [CONTAMINATED]
        rig.encoder.renewal = "go_away"
        if operation == "recovery":
            rig.encoder.mark_dirty()
            running = rig.ensure_ready()
        elif operation == "synchronize":
            request = HistorySyncRequest(SEED, "history_block_compaction", HISTORY)
            running = rig.encoder.synchronize(request, rig.records)
        else:
            running = rig.encoder.renew(SEED, HISTORY, rig.records)
        with pytest.raises(RuntimeError, match=f"^{message}$"):
            rig.run(running)
    ((sync, result, error),) = rig.records.calls
    assert (result, error) == (CONTAMINATED, None)
    assert sync.reason == {"recovery": "runtime_recovery", "renew": "go_away"}.get(
        operation, "history_block_compaction"
    )


def test_a_renewal_runs_only_once_the_provider_asks_for_it():
    with _rig() as rig:
        assert rig.run(rig.encoder.renew(SEED, HISTORY, rig.records)) is None
        rig.encoder.renewal, rig.encoder.syncs = "go_away", [EXACT]
        assert rig.run(rig.encoder.renew(SEED, HISTORY, rig.records)) is EXACT
    assert rig.encoder.log == [("sync", "go_away", False, HISTORY)]


@pytest.mark.parametrize(
    ("terminal", "reconnect_required", "repaired"),
    [
        (ProviderHistoryState.EXACT, False, False),
        (ProviderHistoryState.EXACT, True, True),
        (ProviderHistoryState.CONTAMINATED, False, True),
        (ProviderHistoryState.UNKNOWN, False, True),
    ],
)
def test_a_generation_is_repaired_when_its_end_leaves_the_history_unknown_or_asks_to_reconnect(
    terminal, reconnect_required, repaired
):
    """As the runtime decided after a generation (context.py before the session moved): a
    repair that does not leave the session exact does not raise; its result says so."""
    request = HistorySyncRequest(SEED, "provider_history_changed", HISTORY, GENERATION)
    with _rig() as rig:
        rig.encoder.syncs = [CONTAMINATED]
        result = rig.run(rig.encoder.repair(request, terminal, reconnect_required, rig.records))
    assert result is (CONTAMINATED if repaired else None)
    assert rig.records.calls == ([(request, CONTAMINATED, None)] if repaired else [])


def test_an_abandoned_generation_neither_readies_nor_syncs_the_session():
    abandoned = threading.Event()
    abandoned.set()
    with _rig() as rig:
        encoder, records = rig.encoder, rig.records
        encoder.renewal, encoder.syncs = "go_away", [EXACT]
        encoder.mark_dirty()
        operations = [
            (rig.ensure_ready, "its session"),
            (
                lambda: encoder.synchronize(HistorySyncRequest(SEED, "c", HISTORY), records),
                "its sync",
            ),
            (lambda: encoder.renew(SEED, HISTORY, records), "its sync"),
            (
                lambda: encoder.repair(
                    HistorySyncRequest(SEED, "r", HISTORY, GENERATION),
                    ProviderHistoryState.UNKNOWN,
                    False,
                    records,
                ),
                "its sync",
            ),
        ]
        with retry_cancellation(abandoned):
            for operation, before in operations:
                with pytest.raises(RetryCancelled, match=f"^generation abandoned before {before}$"):
                    rig.run(operation())
        assert rig.run(rig.ensure_ready()) is EXACT  # a generation that goes on rebuilds it
    assert encoder.log == [("sync", "runtime_recovery", True, HISTORY)]


@pytest.mark.parametrize("operation", ["reopen", "recovery", "synchronize"])
def test_what_an_abandoned_generation_opened_or_rebuilt_is_closed_again(operation):
    """The abort came while the generation connected or synced, and closed what was open then.
    What the connect or sync then opened is recorded, closed again, and the session is left for
    the next generation to open."""
    cancel = threading.Event()
    with _rig(connected=operation != "reopen") as rig:
        rig.encoder.during = cancel.set
        rig.encoder.syncs = [EXACT]
        if operation == "recovery":
            rig.encoder.mark_dirty()
        with retry_cancellation(cancel):
            running = (
                rig.encoder.synchronize(HistorySyncRequest(SEED, "c", HISTORY), rig.records)
                if operation == "synchronize"
                else rig.ensure_ready()
            )
            with pytest.raises(
                RetryCancelled, match="^generation abandoned while it readied its session$"
            ):
                rig.run(running)
        assert rig.encoder.state == RuntimeState.DIRTY and not rig.encoder._connected
    assert [entry[0] for entry in rig.encoder.log] == [
        "connect" if operation == "reopen" else "sync",
        "close",
    ]
    ((sync, result, _),) = rig.records.calls
    assert (sync is None, result) == ((True, None) if operation == "reopen" else (False, EXACT))


def test_one_session_operation_runs_at_a_time():
    """A reopen that takes 1 s holds a sync submitted while it runs until it is done."""
    with _rig(connected=False) as rig:
        rig.encoder.connect_s, rig.encoder.syncs = 1.0, [EXACT]
        reopening = rig.thread.submit(rig.ensure_ready())
        request = HistorySyncRequest(SEED, "history_block_compaction", HISTORY)
        rig.run(rig.encoder.synchronize(request, rig.records))
        assert reopening.done()
        rig.encoder.log.append(("synced", rig.clock.monotonic()))
    assert rig.encoder.log == [
        ("connect", HISTORY, 0.0),
        ("sync", "history_block_compaction", False, HISTORY),
        ("synced", 1.0),
    ]


@pytest.mark.parametrize("operation", ["renew", "repair"])
def test_a_renewal_or_a_repair_waits_for_the_session_operation_under_way(operation):
    """A reopen that takes 1 s holds a renewal or a repair submitted while it runs until it is
    done, as it holds a sync: a generation repaired after it was abandoned holds the next one's
    session operations the same way."""
    with _rig(connected=False) as rig:
        rig.encoder.connect_s, rig.encoder.syncs, rig.encoder.renewal = 1.0, [EXACT], "go_away"
        reopening = rig.thread.submit(rig.ensure_ready())
        if operation == "renew":
            rig.run(rig.encoder.renew(SEED, HISTORY, rig.records))
        else:
            request = HistorySyncRequest(SEED, "provider_history_changed", HISTORY)
            unknown = ProviderHistoryState.UNKNOWN
            rig.run(rig.encoder.repair(request, unknown, False, rig.records))
        assert reopening.done()
        rig.encoder.log.append(("synced", rig.clock.monotonic()))
    reason = "go_away" if operation == "renew" else "provider_history_changed"
    assert rig.encoder.log == [
        ("connect", HISTORY, 0.0),
        ("sync", reason, False, HISTORY),
        ("synced", 1.0),
    ]


def test_the_state_says_whether_the_session_can_take_the_next_generation():
    with _rig(connected=False) as rig:
        states = [rig.encoder.state]
        rig.run(rig.encoder.connect(ConnectRequest(seed=SEED)))
        states.append(rig.encoder.state)
        rig.encoder.mark_dirty()
        states.append(rig.encoder.state)
        broken = HistorySyncResult(SyncDisposition.BROKEN, ProviderHistoryState.UNKNOWN)
        rig.encoder.syncs = [broken]
        rig.run(
            rig.encoder.sync_history(HistorySyncRequest(SEED, "provider_history_changed", HISTORY))
        )
        states.append(rig.encoder.state)
        rig.run(rig.encoder.abort(None))
        states.append(rig.encoder.state)
        rig.run(rig.ensure_ready())
        states.append(rig.encoder.state)
    assert states == [
        RuntimeState.DIRTY,  # never opened
        RuntimeState.READY,
        RuntimeState.DIRTY,
        RuntimeState.BROKEN,
        RuntimeState.DIRTY,  # an abort closes a broken session too
        RuntimeState.READY,
    ]


class _StatelessDriver:
    """A stateless driver's protocol operations, logged."""

    capabilities = RealtimeCapabilities("replay_stateless", persistent_session=False)

    def __init__(self):
        self.log: list[tuple] = []

    async def connect(self, request):
        self.log.append(("connect", request.history))

    async def sync_history(self, request):
        self.log.append(("sync", request.reason))
        return HistorySyncResult(SyncDisposition.EXACT, ProviderHistoryState.EXACT, "stateless")

    async def interrupt(self, generation):
        self.log.append(("interrupt", generation.generation_id))
        return InterruptResult(ProviderHistoryState.EXACT)

    async def close(self):
        self.log.append(("close",))


def test_a_stateless_driver_is_readied_and_aborted_as_a_session():
    """StatelessSession: the driver's connect, sync_history and close stand for the
    connection operations. After an abort the next generation connects again, which is where a
    Decisions client remakes the HTTP client the abort closed."""
    driver = _StatelessDriver()
    clock = FakeClock()
    with clock.participant():
        session = StatelessSession(driver, clock=clock, trace=UNTRACED)
        thread = ExecutionThread(clock, "stateless-test")
        thread.start()
        try:
            thread.run(session.connect(ConnectRequest(seed=SEED)))
            thread.run(session.abort(GENERATION))
            thread.run(session.ensure_ready(SEED, HISTORY, _Records()))
            session.mark_dirty()
            thread.run(session.ensure_ready(SEED, HISTORY, _Records()))
            thread.run(session.close())
        finally:
            thread.stop()
    assert driver.log == [
        ("connect", None),
        ("interrupt", "turn-0"),
        ("close",),
        ("connect", HISTORY),
        ("sync", "runtime_recovery"),
        ("close",),
    ]


def test_a_persistent_session_must_manage_itself():
    """The catalog's check of a driver class (providers/registry.py): one that declares
    persistent_session without any of the operations the runtime calls is refused."""
    attributes = {
        "__init__": lambda self, config, *, clock=None: None,
        "capabilities": RealtimeCapabilities("replay_session", persistent_session=True),
        "deployment": Deployment(family="streaming", provider="replay", key_env=None),
        "model_defaults": {},
        **{name: lambda self, *args: None for name in catalog._DRIVER_METHODS},
    }
    lifecycle = ("ensure_ready", "synchronize", "renew", "repair", "abort", "mark_dirty")
    for missing in lifecycle:
        methods = {name: getattr(SessionLifecycle, name) for name in lifecycle if name != missing}
        driver = type("SessionDriver", (), {**attributes, **methods})
        with pytest.raises(ValueError, match=rf"declares persistent_session but has no {missing}"):
            catalog._check_driver("rtb_ext.session:SessionDriver", driver)
    # The runtime reads the session's state too (RealtimeRuntime.state), a property.
    methods = {name: getattr(SessionLifecycle, name) for name in lifecycle}
    driver = type("SessionDriver", (), {**attributes, **methods})
    with pytest.raises(ValueError, match=r"declares persistent_session but has no state$"):
        catalog._check_driver("rtb_ext.session:SessionDriver", driver)
    managed = type("SessionDriver", (), {**attributes, **methods, "state": SessionLifecycle.state})
    catalog._check_driver("rtb_ext.session:SessionDriver", managed)


# ---- OpenAI Realtime on a fake SDK connection ---------------------------------------------


class _FakeRealtime:
    """client.realtime of the OpenAI SDK: each connect opens a connection, numbered from 1, that
    logs each client event the driver sends as (connection, type, arguments), and its close. The
    first ``refuse`` connects fail as a dropped connection does; a client event of a type in
    ``fail`` fails as a write on a dropped connection does."""

    def __init__(self, refuse: int = 0, fail: frozenset = frozenset()):
        self.log: list[tuple] = []
        self._numbers = itertools.count(1)
        self._refuse = refuse
        self._fail = fail

    def connect(self, *, model, websocket_connection_options):
        if self._refuse:
            self._refuse -= 1
            raise ConnectionError("connection refused")
        return _FakeConnectionManager(next(self._numbers), self.log, self._fail)


class _FakeConnectionManager:
    def __init__(self, number: int, log: list, fail: frozenset = frozenset()):
        self.number, self.log, self.fail = number, log, fail

    def __enter__(self):
        number, log, fail = self.number, self.log, self.fail

        def client_event(kind):
            def send(**arguments):
                if kind in fail:
                    raise ConnectionError("connection dropped")
                log.append((number, kind, arguments))

            return send

        return SimpleNamespace(
            _connection=SimpleNamespace(
                trace_connection_number=number, trace_connection_id=f"openai_realtime.ws-{number}"
            ),
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


def _openai(refuse: int = 0, fail: frozenset = frozenset()) -> tuple[OpenAIRealtimeDriver, list]:
    driver = OpenAIRealtimeDriver(
        ModelConfig(
            adapter_type="openai_realtime",
            model_name="gpt-realtime-replay",
            api_base_url="http://127.0.0.1:9/v1",  # never connected to
            api_key="sk-lifecycle",
        )
    )
    driver.bind_trace(UNTRACED)
    realtime = _FakeRealtime(refuse, fail)
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


def test_openai_realtime_records_a_write_that_failed_as_unknown():
    """A client event whose write raised leaves the item it carried unknown in the event log, on
    the connection it was written on: the server may or may not have it."""
    driver, _ = _openai(fail=frozenset({"conversation.item.create"}))
    trace = _Trace()
    driver.bind_trace(trace)
    turns = [turn(0)]

    async def run():
        await driver.connect(ConnectRequest(seed=SEED))
        with pytest.raises(ConnectionError, match="^connection dropped$"):
            await driver.send_observation(turns[0].observation, view(turns))

    asyncio.run(run())
    deliveries = [p for event, p, _ in trace.events if event == TraceEvent.SESSION_DELIVERY]
    assert [
        (d["state"], d["connection_id"], [c["block"] for c in d["contents"]]) for d in deliveries
    ] == [("unknown", "openai_realtime.ws-1", ["d0.frame"])]


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


def test_openai_realtime_recovers_after_an_abandoned_rebuild():
    """An abandoned generation's rebuild closes the connection, then its connect attempt is
    cancelled (RequestRetry under a set retry_cancellation). The next use reopens the session."""
    driver, log = _openai()
    records = _Records()
    abandoned = threading.Event()
    abandoned.set()

    async def run():
        await driver.connect(ConnectRequest(seed=SEED))
        with retry_cancellation(abandoned), pytest.raises(RetryCancelled):
            await driver.sync_history(
                HistorySyncRequest(SEED, "history_block_compaction", view([]), force_rebuild=True)
            )
        return await driver.ensure_ready(SEED, view([]), records)

    result = asyncio.run(run())
    assert (result.disposition, result.history_state) == (
        SyncDisposition.REBUILT,
        ProviderHistoryState.EXACT,
    )
    assert _sent(log) == [(1, "session.update"), (1, "close"), (2, "session.update")]
    assert records.calls[0][0].reason == "runtime_recovery"


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


# ---- each wait in the trace, before it starts -------------------------------------------------

S = 1_000_000_000  # nanoseconds


class _Trace:
    """An agent trace that keeps what it is given, and the scope: (event, payload, scope). Each
    SessionWait also goes into ``log`` as ("record", wait), beside what the test logs there."""

    def __init__(self, log: list | None = None):
        self.events: list[tuple] = []
        self._log = log

    def capture(self, event, payload, *, body=None, classify=None):
        self.events.append((event, payload, current_trace_scope()))
        if event == "SessionWait" and self._log is not None:
            self._log.append(("record", payload["wait"]))

    @property
    def waits(self) -> list[tuple]:
        return [
            (payload["wait"], payload["seconds"], payload["until_monotonic_ns"])
            for event, payload, _ in self.events
            if event == "SessionWait"  # the name the episode trace carries
        ]


class _FrozenClock(RecordingClock):
    """Waits are logged, not taken, and the monotonic time stays at 100 s."""

    def monotonic(self) -> float:
        return 100.0

    def monotonic_ns(self) -> int:
        return 100 * S


def _log_closes(live: FakeLive) -> None:
    """Each connection ``live`` opens also logs its close, as ("exit",)."""
    connect, log = live.connect, live._log

    def logged_connect(*, model, config):
        manager = connect(model=model, config=config)

        class Logged:
            async def __aenter__(self):
                return await manager.__aenter__()

            async def __aexit__(self, *exc):
                log.append(("exit",))
                return await manager.__aexit__(*exc)

        return Logged()

    live.connect = logged_connect


def _traced(rig: Rig) -> _Trace:
    """Records go into the rig's log too, beside its connects, closes and waits."""
    trace = _Trace(rig.log)
    rig.driver.clock = _FrozenClock(rig.log)
    rig.driver.bind_trace(trace)
    _log_closes(rig.live)
    return trace


def test_a_wait_is_recorded_with_the_time_it_ends_before_it_starts():
    """On a clock that moves, a record written after the wait would end 2 s later."""
    with _rig() as rig:
        trace = rig.encoder._trace = _Trace()
        rig.clock.sleep(5.0)
        rig.run(rig.encoder._wait("video_ingestion", 2.0))
        assert rig.clock.monotonic() == 7.0
    assert trace.waits == [("video_ingestion", 2.0, 7 * S)]


def test_gemini_live_records_the_go_away_deadline_closes_and_the_release_grace():
    update = {"sessionResumptionUpdate": {"newHandle": "handle-0", "resumable": True}}
    rig = Rig("gemini-3.8-live", [[update, {"goAway": {"timeLeft": "30s"}}, TURN_COMPLETE], []])
    trace = _traced(rig)

    async def run():
        await rig.connect()
        generation = await rig.start("turn-0")
        await drain(rig.driver.events(generation))  # the last read is bounded by the goAway
        renewal = HistorySyncRequest(rig.seed, "go_away", rig.history, generation)
        await rig.driver.sync_history(renewal)  # closes, waits the grace, resumes
        await rig.driver.close()

    asyncio.run(run())
    assert trace.waits == [
        ("go_away_deadline", 28.0, 128 * S),  # the goAway's 30 s, less the 2 s close margin
        ("connection_close", 1.0, 101 * S),
        ("session_release", 2.0, 102 * S),
        ("connection_close", 1.0, 101 * S),
    ]
    # Each record comes before what it waits for: the close, the release grace.
    assert rig.log == [
        ("connect", None),
        ("record", "go_away_deadline"),
        ("record", "connection_close"),
        ("exit",),
        ("record", "session_release"),
        ("sleep", 2.0),
        ("connect", "handle-0"),
        ("record", "connection_close"),
        ("exit",),
    ]


def test_gemini_live_records_the_video_waits():
    rig = Rig("replay-gemini-live", [[]])  # the driver defaults: realtime_input, at 1 fps
    trace = _traced(rig)

    async def run():
        await rig.connect()
        for k in range(2):
            await rig.driver.send_observation(None, rig.view([frame_turn(j) for j in range(k + 1)]))

    asyncio.run(run())
    assert trace.waits == [
        ("video_ingestion", 1.0, 101 * S),
        ("video_interval", 1.0, 101 * S),  # no time passed since the first video
        ("video_ingestion", 1.0, 101 * S),
    ]


def test_openai_realtime_records_the_connect_backoff():
    """The record comes after the failed attempt's ModelAttemptFinished and before RequestRetry
    waits the delay."""
    driver, _ = _openai(refuse=1)
    log = []
    trace = _Trace(log)
    driver.clock = _FrozenClock(log)
    driver.bind_trace(trace)
    asyncio.run(driver.connect(ConnectRequest(seed=SEED)))
    delay = TRANSPORT_RETRY.delay_s(1, "empty:connect", TransportFailure(TRANSIENT))
    assert [(event, payload.get("outcome"), scope) for event, payload, scope in trace.events] == [
        (TraceEvent.MODEL_ATTEMPT_FINISHED, "retry", {"connect_attempt": 1}),
        (TraceEvent.SESSION_WAIT, None, {"connect_attempt": 1}),
        (TraceEvent.MODEL_ATTEMPT_FINISHED, "ok", {"connect_attempt": 2}),
    ]
    assert trace.waits == [("connect_backoff", delay, 100 * S + round(delay * S))]
    assert log == [("record", "connect_backoff"), ("sleep", delay)]


def test_openai_realtime_records_no_connect_backoff_once_its_generation_is_abandoned():
    """The generation is abandoned during the failed attempt: RequestRetry does not wait the
    delay, so no wait is recorded, and no second attempt runs."""
    driver, _ = _openai(refuse=1)
    trace = _Trace()
    driver.clock = _FrozenClock([])
    driver.bind_trace(trace)
    abandoned = threading.Event()
    realtime = driver.client.realtime
    connect = realtime.connect

    def abandoning_connect(**arguments):
        abandoned.set()
        return connect(**arguments)

    realtime.connect = abandoning_connect

    async def run():
        with retry_cancellation(abandoned), pytest.raises(RetryCancelled):
            await driver.connect(ConnectRequest(seed=SEED))

    asyncio.run(run())
    # The attempt the cancellation stops is recorded as such, and no wait before it.
    assert [(event, payload.get("outcome")) for event, payload, _ in trace.events] == [
        (TraceEvent.MODEL_ATTEMPT_FINISHED, "retry"),
        (TraceEvent.MODEL_ATTEMPT_FINISHED, "cancelled"),
    ]


def test_gemini_live_an_abandoned_generation_neither_waits_the_release_grace_nor_reconnects():
    """After an abort the next connect waits 2 s for the old session's release first. The
    generation was abandoned by then (its abort came while it waited for the session lock):
    it takes no wait, records none and opens no session."""
    rig = Rig("gemini-3.8-live", [[HANG], []])
    trace = _Trace()
    rig.driver.bind_trace(trace)
    abandoned = threading.Event()
    abandoned.set()

    async def run():
        await rig.connect()
        await rig.driver.abort(await rig.start("turn-0"))
        history = rig.view([frame_turn(0)])
        with (
            retry_cancellation(abandoned),
            pytest.raises(RetryCancelled, match="before the session_release wait$"),
        ):
            await rig.driver.connect(ConnectRequest(seed=rig.seed, history=history))

    asyncio.run(run())
    assert rig.log == [("connect", None)]  # neither a grace sleep nor a second connect
    assert [wait for wait, _, _ in trace.waits] == ["connection_close"]  # the abort's close


def test_close_records_its_wait_for_an_abort():
    with _rig() as rig:
        trace = rig.encoder._trace = _Trace()
        rig.encoder.interrupt_s = 1.0
        rig.thread.submit(rig.encoder.abort(GENERATION))
        rig.run(rig.encoder.close())
        rig.run(rig.encoder.close())  # no abort left to wait for
    assert trace.waits == [("abort_cleanup", 10.0, 10 * S)]


def test_close_records_no_wait_for_an_abort_already_done():
    with _rig() as rig:
        trace = rig.encoder._trace = _Trace()
        rig.run(rig.encoder.abort(GENERATION))
        rig.run(rig.encoder.close())
    assert trace.waits == []


class _TickingClock(RecordingClock):
    """Waits are logged, not taken; each read of the monotonic time finds it 1 ms later."""

    def __init__(self, log: list):
        super().__init__(log)
        self._reads = itertools.count()

    def monotonic(self) -> float:
        return 100.0 + next(self._reads) / 1000

    def monotonic_ns(self) -> int:
        return round(self.monotonic() * S)


def test_gemini_live_records_a_go_away_deadline_once():
    """The reads after a goAway all wait until its deadline: one wait, recorded before the first
    of them, which ends at the deadline itself."""
    transcript = {"serverContent": {"outputTranscription": {"text": "partial"}}}
    rig = Rig("gemini-3.8-live", [[{"goAway": {"timeLeft": "30s"}}, transcript, TURN_COMPLETE]])
    trace = _Trace()
    rig.driver.clock = _TickingClock(rig.log)
    rig.driver.bind_trace(trace)

    async def run():
        await rig.connect()
        await drain(rig.driver.events(await rig.start("turn-0")))

    asyncio.run(run())
    deadline = round(rig.driver._core.go_away_deadline * S)
    assert [(wait, until) for wait, _, until in trace.waits] == [("go_away_deadline", deadline)]


def test_gemini_live_records_no_wait_for_a_go_away_deadline_already_past():
    """A goAway with less time left than the close margin leaves nothing to wait for: the next
    read fails at once and the driver closes the connection, which is the only wait recorded."""
    rig = Rig("gemini-3.8-live", [[{"goAway": {"timeLeft": "1s"}}, HANG]])
    trace = _traced(rig)

    async def run():
        await rig.connect()
        with pytest.raises(live.GoAwayDeadlineExceeded):
            await drain(rig.driver.events(await rig.start("turn-0")))

    asyncio.run(run())
    assert trace.waits == [("connection_close", 1.0, 101 * S)]


class _AdvancingClock(RecordingClock):
    """Waits are logged, not taken, and move the monotonic time on by what they wait."""

    def __init__(self, log: list):
        super().__init__(log)
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now

    def monotonic_ns(self) -> int:
        return round(self.now * S)

    def wait(self, waitable, timeout):
        self.now += timeout
        return super().wait(waitable, timeout)


def test_gemini_live_neither_records_nor_takes_a_wait_of_no_time():
    """At 1 fps the second video is due once the first one's 1 s ingestion wait is over: its
    interval wait would last no time, so there is none."""
    rig = Rig("replay-gemini-live", [[]])
    trace = _Trace()
    rig.driver.clock = _AdvancingClock(rig.log)
    rig.driver.bind_trace(trace)

    async def run():
        await rig.connect()
        for k in range(2):
            await rig.driver.send_observation(None, rig.view([frame_turn(j) for j in range(k + 1)]))

    asyncio.run(run())
    assert trace.waits == [("video_ingestion", 1.0, 101 * S), ("video_ingestion", 1.0, 102 * S)]
    assert rig.log == [("connect", None), ("sleep", 1.0), ("sleep", 1.0)]
