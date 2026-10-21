"""RealtimeRuntime (harnesses/default/context.py) on a live session encoder: what the runtime asks
the session (providers/session_lifecycle.py) and what it records of it. The encoder's connection
operations only log; the runtime runs on a real execution thread, as DefaultHarness runs it."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

from realtime_framework.clock import SystemClock
from realtime_framework.entity.actions import Observation
from realtime_framework.entity.context_policy import Retention, current_context_policy
from realtime_framework.entity.events import GenerationBoundary, ProviderHistoryState
from realtime_framework.entity.model import (
    GenerationHandle,
    GenerationOutput,
    GenerationRequest,
    HistorySyncResult,
    InterruptResult,
    ProviderObservation,
    RealtimeCapabilities,
    SessionSeed,
    SyncDisposition,
)
from realtime_framework.entity.records import TraceEvent
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.context import RealtimeRuntime
from realtime_framework.harnesses.default.execution_thread import ExecutionThread
from realtime_framework.harnesses.default.request import build_request, system_block
from realtime_framework.providers.session_lifecycle import SessionLifecycle

WAIT_S = 10.0
ANSWER = AnswerRequirements("text_json", "standard_vlm", ())
SEED = SessionSeed("system instruction", "replay-model", None, 1.0, ANSWER)
REBUILT = HistorySyncResult(SyncDisposition.REBUILT, ProviderHistoryState.EXACT, "rebuilt")


class _Trace:
    def __init__(self):
        self.events: list[tuple] = []

    def capture(self, event, payload, **_) -> None:
        self.events.append((event, payload))

    def history_changes(self) -> list[dict]:
        return [payload for event, payload in self.events if event == TraceEvent.HISTORY_CHANGED]


class _LiveDriver(SessionLifecycle):
    """A persistent session whose operations log; a reply ends exact. While ``hold`` is set, a
    reply waits for it to clear once it has started (``generating``)."""

    capabilities = RealtimeCapabilities("replay_session", persistent_session=True)

    def __init__(self, clock, trace):
        self.clock = clock
        self._trace = trace
        self.log: list[tuple] = []
        self.hold: threading.Event | None = None
        self.generating = threading.Event()

    def bind_trace(self, trace) -> None:
        self._trace = trace

    async def _connect(self, request):
        self.log.append(("connect", request.history is not None))

    async def _sync_history(self, request):
        self.log.append(("sync", request.reason))
        return REBUILT

    async def _close_current(self):
        self.log.append(("close",))

    async def interrupt(self, generation):
        self.log.append(("interrupt", generation.generation_id))
        return InterruptResult(ProviderHistoryState.UNKNOWN)

    async def send_observation(self, observation, history):
        return {}

    async def start_generation(self, request):
        self.log.append(("start_generation", request.generation_id))
        return GenerationHandle(request.generation_id)

    async def events(self, generation):
        self.generating.set()
        if self.hold is not None:
            await asyncio.to_thread(self.hold.wait, WAIT_S)
        yield GenerationBoundary(
            generation.generation_id, 0, history_state=ProviderHistoryState.EXACT
        )


class _Collector:
    def consume(self, event) -> None:
        pass

    def finish(self) -> GenerationOutput:
        return GenerationOutput('{"action": "wait"}', None, {}, ())


def _observation(k: int) -> ProviderObservation:
    return ProviderObservation(
        observation=Observation(b"", float(k), k, {"observation_id": f"obs-{k}"}),
        prompt=f"frame {k}",
        image_bytes=b"png %d" % k,
        mime_type="image/png",
        image_detail="auto",
        geometry={},
    )


class _Rig:
    def __init__(self, turns: Retention | None = None):
        clock = SystemClock()
        self.trace = _Trace()
        self.driver = _LiveDriver(clock, self.trace)
        policy = current_context_policy().for_path("openai_realtime", {}, None)
        self.policy = policy if turns is None else replace(policy, turns=turns)
        self.execution = ExecutionThread(clock, "runtime-session-test")
        self.runtime = RealtimeRuntime(
            self.driver, self.policy, trace=self.trace, execution=self.execution, clock=clock
        )
        self.history: list[dict] = []

    def __enter__(self):
        self.execution.start()
        self.runtime.start(SEED)
        return self

    def __exit__(self, *exc_info):
        self.runtime.close()
        self.execution.stop()

    def turn(self, k: int, cancel: threading.Event | None = None):
        system = system_block(SEED.system_instruction, None)
        call = CallParameters(SEED.model_name, None, None)
        return self.runtime.run_turn(
            [_observation(k)],
            GenerationRequest(f"turn-{k}", None, "inf", "wall"),
            _Collector(),
            action_history=self.history,
            timeout=WAIT_S,
            request_builder=lambda turns, ledger: build_request(
                self.policy, system, turns, ledger, ANSWER, call
            ),
            decision=k,
            cancel=cancel or self.runtime.clock.event(),
        )

    def accept(self, k: int) -> None:
        result = self.turn(k)
        self.runtime.accept_turn(result.turn)
        self.history.append({"tick": k + 1, "executed": "wait", "result": "wait"})


def test_an_abort_interrupts_the_generation_under_way_then_closes_the_session():
    """RealtimeRuntime.abort hands the session the generation that is generating: the encoder
    interrupts it, then closes the connection."""
    with _Rig() as rig:
        rig.driver.hold = threading.Event()
        cancel = rig.runtime.clock.event()
        outcome = []
        worker = threading.Thread(target=lambda: outcome.append(rig.turn(0, cancel)))
        worker.start()
        assert rig.driver.generating.wait(WAIT_S), "the generation did not start"
        cancel.set()  # as DefaultHarness._cancel does before it aborts
        rig.runtime.abort()
        rig.driver.hold.set()
        worker.join(WAIT_S)
        (result,) = outcome
    assert result.cancelled
    started = rig.driver.log.index(("start_generation", "turn-0"))
    assert rig.driver.log[started + 1 : started + 3] == [("interrupt", "turn-0"), ("close",)]


def test_the_session_reopened_after_an_abort_is_recorded_as_a_recovery_rebuild():
    """The next generation reopens the session the abort closed, from the retained history, and
    the runtime records it: a rebuild for recovery that replayed the retained turns."""
    with _Rig() as rig:
        rig.accept(0)
        before = len(rig.driver.log)
        rig.trace.events.clear()
        rig.runtime.abort()
        rig.turn(1)
    assert rig.driver.log[before : before + 2] == [("close",), ("connect", True)]
    (reopened,) = [c for c in rig.trace.history_changes() if c["sync"] == {"reason": "reconnect"}]
    assert (reopened["kind"], reopened["trigger"], reopened["replayed_turns"]) == (
        "rebuild",
        "recovery",
        1,
    )
    assert reopened["turns"] == []


def test_the_sync_after_an_archive_records_the_turns_it_removed():
    """With one verbatim turn kept, decision 1's request archives decision 0's turn and the
    session is brought to the history without it; that sync's HistoryChanged names the turn."""
    with _Rig(turns=Retention(keep=1, block=1)) as rig:
        rig.accept(0)
        rig.trace.events.clear()
        rig.turn(1)
    (synced,) = [
        c
        for c in rig.trace.history_changes()
        if (c.get("sync") or {}).get("reason") == "history_block_compaction"
    ]
    assert synced["kind"] == "rebuild"
    assert [turn["decision"] for turn in synced["turns"]] == [0]
    assert ("sync", "history_block_compaction") in rig.driver.log
