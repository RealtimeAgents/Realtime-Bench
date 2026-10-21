"""DefaultHarness generations on the replay stubs: what an abort leaves behind."""

from __future__ import annotations

import asyncio
import threading

import realtime_framework.providers.registry as catalog
from tests.replay import scenario_decisions
from tests.replay.build import build_episode
from tests.replay.frames import ReplayTask
from tests.replay.scenario import TASK_INSTRUCTION
from tests.replay.stub_decisions import DecisionsStub

WAIT_S = 10.0  # bound on every cross-thread wait; none is reached when the code behaves
LUNA = "gpt-6-luna-decisions"  # configs/models/gpt-6-luna-decisions.yaml; upstream "openai"


def _luna_episode(stub, run_dir):
    config = catalog.to_model_config(LUNA, fps=1.0, reasoning_effort="fixed").model_copy(
        update={"api_base_url": stub.base_url}
    )
    return build_episode(
        config, run_dir, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key=LUNA
    )


def _hold_questions(driver):
    """Hold each Choice question before it is posted until ``release`` is set; ``entered`` is
    set once one waits."""
    entered, release = threading.Event(), threading.Event()
    decide = driver.decide

    async def held(state, question):
        if not release.is_set():
            entered.set()
            await asyncio.to_thread(release.wait, WAIT_S)
        return await decide(state, question)

    driver.decide = held
    return entered, release


def test_a_decision_after_an_abort_sends_its_questions(tmp_path, monkeypatch):
    """Luna, aborted while decision 0's question is under way, as the legacy loop's per-move cap
    does. The abort closes the Decisions HTTP client (providers/openai_decisions.py close), and
    the held question then fails on it. Decision 1 connects again first
    (RealtimeRuntime._ensure_ready), which opens a new client, so its questions are sent and
    answered."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "replay-dummy-key")
    with DecisionsStub(upstream="openai") as stub:
        episode = _luna_episode(stub, tmp_path)
        agent = episode.agent
        try:
            entered, release = _hold_questions(agent.driver)
            observation = episode.frames.capture(0)
            agent.configure_decision(episode.budget(observation.tick_number))
            agent.stream_prefill(observation)
            done = threading.Event()
            worker = threading.Thread(
                target=lambda: (agent.stream_generate(tick=0), done.set()), daemon=True
            )
            worker.start()
            assert entered.wait(WAIT_S), "decision 0 asked no question"
            closed = agent.driver.client
            agent.abort()
            agent._runtime._cleanup_future.result(timeout=WAIT_S)
            assert closed.is_closed
            release.set()
            assert done.wait(WAIT_S), "decision 0 never returned"
            assert stub.requests == []
            observation = episode.frames.capture(1)
            stub.load(scenario_decisions.WEB_LONG[1].script)
            agent.configure_decision(episode.budget(observation.tick_number))
            agent.stream_prefill(observation)
            result = agent.stream_generate(tick=1)
        finally:
            episode.finish()
    assert not result.get("invalid_reason"), result.get("invalid_detail")
    assert result["raw"] == '{"action": "type", "text": "Ada Lovelace"}'
    assert len(stub.requests) == len(scenario_decisions.WEB_LONG[1].script)
    assert stub.unscripted == 0 and not stub.off_script
