"""An abort that lands while a live session is being rebuilt or synced (OpenAI Realtime on the
replay WebSocket stub): what the abandoned generation leaves to the next one."""

from __future__ import annotations

import json
import threading

import realtime_framework.entity.model as model_module
from tests.replay.build import build_episode, wait_for_abort
from tests.replay.frames import ReplayTask
from tests.replay.scenario import CLICK_NAME, TASK_INSTRUCTION
from tests.replay.steps import Step
from tests.replay.test_openai_realtime import MODEL_KEY, realtime_config
from tests.replay.ws_openai_realtime import Connection, RealtimeStub
from tests.replay.ws_openai_realtime import replies as realtime_replies

WAIT_S = 10.0  # bound on every cross-thread wait; none is reached when the code behaves
STEP = Step(CLICK_NAME)


class _FaultStub(RealtimeStub):
    """Refuses the upgrades numbered in ``refused`` with HTTP 503, and ends the responses
    numbered in ``incomplete`` as ``incomplete`` with no output items, which leaves the session
    for a rebuild (providers/openai_realtime.py ``_sync_history``)."""

    def __init__(self, *args, refused=(), incomplete=(), **kwargs):
        super().__init__(*args, **kwargs)
        self._refused, self._incomplete = set(refused), set(incomplete)

    def _upgrade(self, websocket, request):
        with self._lock:
            number = len(self.connections) + 1
            if number in self._refused:
                headers = request.headers
                self.connections.append(
                    Connection(
                        number=number,
                        url=f"ws://{headers['Host']}{request.path}",
                        header_names=[name for name, _ in headers.raw_items()],
                        authorization=headers.get("Authorization"),
                        rejected=503,
                        finished=True,
                    )
                )
                self._lock.notify_all()
                return websocket.respond(503, "stub unavailable\n")
        return super()._upgrade(websocket, request)

    def _respond(self, connection):
        number = len(self.responses) + 1
        answers, close = super()._respond(connection)
        if number in self._incomplete:
            added = ("response.output_item.added", "response.output_item.done")
            answers = [answer for answer in answers if answer["type"] not in added]
            done = answers[-1]["response"]
            done["status"] = "incomplete"
            done["status_details"] = {"type": "incomplete", "reason": "max_output_tokens"}
            done["output"] = []
        return answers, close


def _episode(stub, run_dir):
    return build_episode(
        realtime_config(stub.base_url),
        run_dir,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        model_key=MODEL_KEY,
    )


def _generate(episode, generation: int, decision: int):
    observation = episode.frames.capture(decision)
    budget = episode.budget(observation.tick_number)
    return episode.agent.generate(generation, decision, observation, budget)


def _in_worker(target):
    box, done = {}, threading.Event()

    def worker():
        try:
            box["value"] = target()
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    return box, done


def _joined(box, done):
    assert done.wait(WAIT_S), "the worker never returned"
    assert "error" not in box, repr(box.get("error"))
    return box["value"]


def _hold_the_next_open(realtime):
    """Hold the next session the SDK opens before its connection exists, until ``release``."""
    entered, release = threading.Event(), threading.Event()
    connect = realtime.connect

    def held(**kwargs):
        realtime.connect = connect
        manager = connect(**kwargs)

        class Held:
            def __enter__(self):
                entered.set()
                assert release.wait(WAIT_S)
                return manager.__enter__()

            def __exit__(self, *exc):
                return manager.__exit__(*exc)

        return Held()

    realtime.connect = held
    return entered, release


def _after_the_abort_runs(runtime) -> None:
    """Return once the execution thread has run the abort's cleanup as far as it goes without
    waiting: the cleanup was scheduled first, and the thread runs what it is given in order."""

    async def barrier():
        pass

    runtime.execution.run(barrier())


def _trace(run_dir, event: str) -> list[dict]:
    lines = (run_dir / "episode_trace.jsonl").read_text().splitlines()
    return [e["payload"] for e in map(json.loads, lines) if e["event"] == event]


def test_an_abort_in_the_sync_after_a_generation_keeps_its_usage(tmp_path, monkeypatch):
    """Response 2 completes, but as incomplete without output items, so the sync after the
    generation rebuilds the session; the stub refuses that upgrade and the connect backs off.
    The kernel aborts generation 2 in that backoff. The generation itself ran: its attempt is
    recorded as cancelled with its usage, and its provider usage as stale, and nothing of it
    joins the history."""
    entered, release = threading.Event(), threading.Event()
    backoff = model_module._backoff

    def held(clock, cancel, seconds):
        entered.set()
        assert release.wait(WAIT_S)
        backoff(clock, cancel, seconds)

    monkeypatch.setattr(model_module, "_backoff", held)
    replies = realtime_replies([STEP, STEP])
    with _FaultStub(replies, refused=(2,), incomplete=(2,)) as stub:
        episode = _episode(stub, tmp_path)
        agent = episode.agent
        try:
            stub.wait_for_session_update()
            _generate(episode, 1, 0)
            agent.accept(1)
            box, done = _in_worker(lambda: _generate(episode, 2, 1))
            assert entered.wait(WAIT_S), "the sync after generation 2 did not back off"
            agent.abort(2)
            release.set()
            abandoned = _joined(box, done)
            agent.discard(2)
            turns, history = agent.history.turns, list(agent.history.actions)
            usage = agent.token_usage
        finally:
            episode.finish()
        connections = stub.finish()
    assert abandoned.result["stale"] and abandoned.result["action"] is None
    assert [c["rejected"] for c in connections] == [None, 503]
    generation = [
        (a["scope"]["generation"], a["outcome"], a["usage"] is not None)
        for a in _trace(tmp_path, "ModelAttemptFinished")
        if a["kind"] == "generation"
    ]
    assert generation == [(1, "ok", True), (2, "cancelled", True)]
    provider = [
        (u["tick"], u["stale"], u["status"])
        for u in map(json.loads, (tmp_path / "provider-usage.jsonl").read_text().splitlines())
    ]
    assert provider == [(0, False, "reported"), (1, True, "reported")]
    assert [turn.decision for turn in turns] == [0] and len(history) == 1
    assert usage["responses"] == 1


def test_a_repair_whose_rebuild_spends_its_connect_retries_keeps_the_generation(
    tmp_path, monkeypatch
):
    """Decision 35: response 1 ends incomplete without output items, so the repair after the
    generation rebuilds the session, and the stub refuses every upgrade after the first. The
    repair comes back broken instead of raising the spent retries: the turn keeps its output and
    fails as history_sync_failed, not as a stream interrupted after its output, which drops it."""
    monkeypatch.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    replies = realtime_replies([STEP])
    with _FaultStub(replies, refused=range(2, 12), incomplete=(1,)) as stub:
        episode = _episode(stub, tmp_path)
        try:
            stub.wait_for_session_update()
            result = _generate(episode, 1, 0).result
        finally:
            episode.finish()
        rejected = [connection.rejected for connection in stub.connections]
    assert rejected == [None] + [503] * 4
    assert result["invalid_reason"] == "history_sync_failed"
    assert result["raw"] == replies[0].text
    assert result["realtime_runtime"]["history_sync"]["disposition"] == "broken"


def test_closing_the_runtime_abandons_a_generation_that_reopens_the_session(tmp_path):
    """The runtime closes while generation 2's second attempt reopens the session, before the
    agent cancelled anything: the generation is abandoned all the same, and the connection it
    opened after the close is closed again."""
    with RealtimeStub(realtime_replies([STEP, STEP]), drop_responses=(2,)) as stub:
        episode = _episode(stub, tmp_path)
        agent, runtime = episode.agent, episode.agent._runtime
        try:
            stub.wait_for_session_update()
            _generate(episode, 1, 0)
            agent.accept(1)
            entered, release = _hold_the_next_open(agent.driver.client.realtime)
            box, done = _in_worker(lambda: _generate(episode, 2, 1))
            assert entered.wait(WAIT_S), "the second attempt opened no session"
            runtime.close()
            release.set()
            abandoned = _joined(box, done)
            left = (agent.driver._connected, agent.driver._conn)
        finally:
            episode.finish()
        connections = stub.finish()
    assert abandoned.result["stale"]
    # The driver holds no connection: the one the reopen made is closed.
    assert left == (False, None)
    assert len(connections) == 2 and stub.violations == []


def test_an_abort_during_a_rebuild_leaves_the_rebuilt_session_closed(tmp_path):
    """Response 2 is dropped before any output, so generation 2's second attempt rebuilds the
    session (the driver's ensure_ready). The kernel aborts generation 2 before the rebuild's
    connection exists, so the abort's close finds nothing to close; the rebuild then opens
    connection 2 and replays the history on it. The abandoned generation must not leave the
    runtime ready on that connection: it is closed, and decision 2 opens connection 3."""
    with RealtimeStub(realtime_replies([STEP, STEP, STEP]), drop_responses=(2,)) as stub:
        episode = _episode(stub, tmp_path)
        agent, runtime = episode.agent, episode.agent._runtime
        try:
            stub.wait_for_session_update()
            _generate(episode, 1, 0)
            agent.accept(1)
            entered, release = _hold_the_next_open(agent.driver.client.realtime)
            box, done = _in_worker(lambda: _generate(episode, 2, 1))
            assert entered.wait(WAIT_S), "the second attempt opened no session"
            agent.abort(2)
            _after_the_abort_runs(runtime)
            release.set()
            abandoned = _joined(box, done)
            agent.discard(2)
            wait_for_abort(agent, WAIT_S)
            # The driver holds no connection: the one the rebuild opened is closed.
            left = (agent.driver._connected, runtime.state.value, agent.driver._conn)
            _generate(episode, 3, 2)
            agent.accept(3)
        finally:
            episode.finish()
        connections = stub.finish()
        responses = [r["connection"] for r in stub.responses]
    assert abandoned.result["stale"]
    assert left == (False, "dirty", None)
    assert len(connections) == 3 and responses == [1, 1, 3]
    assert stub.violations == []
