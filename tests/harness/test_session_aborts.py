"""An abort that lands while a live session is being rebuilt or synced (OpenAI Realtime on the
replay WebSocket stub): what the abandoned generation leaves to the next one."""

from __future__ import annotations

import threading

from tests.replay.build import build_episode
from tests.replay.frames import ReplayTask
from tests.replay.scenario import CLICK_NAME, TASK_INSTRUCTION
from tests.replay.steps import Step
from tests.replay.test_openai_realtime import MODEL_KEY, realtime_config
from tests.replay.ws_openai_realtime import RealtimeStub
from tests.replay.ws_openai_realtime import replies as realtime_replies

WAIT_S = 10.0  # bound on every cross-thread wait; none is reached when the code behaves
STEP = Step(CLICK_NAME)


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
    """Return once the loop has run the abort's cleanup as far as it goes without waiting:
    the cleanup was scheduled first, and the loop runs what it is given in order."""

    async def barrier():
        pass

    runtime._submit(barrier())


def test_an_abort_during_a_rebuild_leaves_the_rebuilt_session_closed(tmp_path):
    """Response 2 is dropped before any output, so generation 2's second attempt rebuilds the
    session (RealtimeRuntime._ensure_ready). The kernel aborts generation 2 before the rebuild's
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
            runtime._cleanup_future.result(timeout=WAIT_S)
            # The driver holds no connection: the one the rebuild opened is closed.
            left = (runtime._connected, runtime.state.value, agent.driver._conn)
            _generate(episode, 3, 2)
            agent.accept(3)
        finally:
            episode.finish()
        connections = stub.finish()
        responses = [r["connection"] for r in stub.responses]
    assert abandoned.cancelled
    assert left == (False, "dirty", None)
    assert len(connections) == 3 and responses == [1, 1, 3]
    assert stub.violations == []
