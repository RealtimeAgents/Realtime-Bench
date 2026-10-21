"""DefaultHarness as the event kernel's agent (evaluation/shell.py KernelAgent), on the replay
stubs: a generation writes no history, AcceptResult writes it, DiscardResult drops it, and an
abort cancels only the generation it names."""

from __future__ import annotations

import asyncio
import json
import threading

import realtime_framework.providers.registry as catalog
from tests.replay import scenario_decisions
from tests.replay.build import build_episode
from tests.replay.frames import ReplayTask
from tests.replay.scenario import CLICK_NAME, LONG_SCENARIO, TASK_INSTRUCTION, replies
from tests.replay.stub import ChatCompletionsStub
from tests.replay.stub_decisions import DecisionsStub
from tests.replay.test_chat_completions import MODEL_KEY as CHAT_MODEL_KEY
from tests.replay.test_chat_completions import chat_completions_config

WAIT_S = 10.0  # bound on every cross-thread wait; none is reached when the code behaves
LUNA = "gpt-6-luna-decisions"  # configs/models/gpt-6-luna-decisions.yaml; upstream "openai"


def _luna_episode(stub, run_dir):
    config = catalog.to_model_config(LUNA, fps=1.0, reasoning_effort="fixed").model_copy(
        update={"api_base_url": stub.base_url}
    )
    return build_episode(
        config, run_dir, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key=LUNA
    )


def _chat_episode(stub, run_dir):
    return build_episode(
        chat_completions_config(stub.base_url, stream=True),
        run_dir,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        model_key=CHAT_MODEL_KEY,
    )


def _generate(episode, generation: int, decision: int):
    """The shell's StartGeneration for ``decision`` on its frame, on the calling thread."""
    observation = episode.frames.capture(decision)
    budget = episode.budget(observation.tick_number)
    return episode.agent.generate(generation, decision, observation, budget)


def _in_worker(target):
    """Run ``target`` on a daemon thread; the box gets its value or its exception."""
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


def _jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


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


def test_a_generation_writes_nothing_until_it_is_accepted(tmp_path):
    """generate() hands back the decision result and holds what it would write; accept() writes
    the turn, the action record, the usage, the provider usage and the tool call's end."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            pending = _generate(episode, 1, 0)
            held = (agent._runtime.turns, list(agent.action_history), agent.token_usage)
            written = (tmp_path / "provider-usage.jsonl").exists()
            tool_calls = _jsonl(tmp_path / "tool_calls.jsonl")
            agent.accept(1)
        finally:
            episode.finish()
    assert (pending.generation, pending.decision, pending.cancelled) == (1, 0, False)
    assert pending.result["raw"] == CLICK_NAME and pending.result["action"] is not None
    turns, history, usage = held
    assert (turns, history, usage["responses"], usage["output_tokens"]) == ((), [], 0, 0)
    assert not written and tool_calls == []
    (turn,) = agent._runtime.turns
    assert (turn.decision, turn.output, turn.action_history_index) == (0, CLICK_NAME, 0)
    (record,) = agent.action_history
    assert (record["tick"], record["action"]) == (1, CLICK_NAME)
    assert agent.token_usage["responses"] == 1
    assert [u["stale"] for u in _jsonl(tmp_path / "provider-usage.jsonl")] == [False]
    assert [t["outcome"] for t in _jsonl(tmp_path / "tool_calls.jsonl")] == ["completed"]


def test_a_discarded_result_writes_no_history(tmp_path):
    """Generation 2's result is back before the kernel gives up on it, as when the per-move cap
    fires as the reply lands: the abort and DiscardResult follow. Nothing of it joins the
    history; its provider usage is recorded as stale and its tool call as superseded."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:2])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            _generate(episode, 1, 0)
            agent.accept(1)
            pending = _generate(episode, 2, 1)
            agent.abort(2)
            agent.discard(2)
            turns, history = agent._runtime.turns, list(agent.action_history)
            usage = agent.token_usage
        finally:
            episode.finish()
    assert not pending.cancelled and pending.result["action"] is not None
    assert len(stub.requests) == 2
    assert [turn.decision for turn in turns] == [0]
    assert [record["tick"] for record in history] == [1]
    assert usage["responses"] == 1
    assert [u["stale"] for u in _jsonl(tmp_path / "provider-usage.jsonl")] == [False, True]
    outcomes = [t["outcome"] for t in _jsonl(tmp_path / "tool_calls.jsonl")]
    assert outcomes == ["completed", "superseded_by_timeout"]


def test_an_abort_before_its_generation_starts_cancels_it(tmp_path):
    """The shell runs StartGeneration on a worker, and the AbortGeneration that follows can come
    before the worker calls generate(). The generation then starts cancelled: it sends nothing,
    and its discarded result leaves no history."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            agent.abort(1)
            pending = _generate(episode, 1, 0)
            agent.discard(1)
        finally:
            episode.finish()
    assert pending.cancelled
    assert pending.result["stale"] is True and pending.result["action"] is None
    assert stub.requests == [] and stub.unused_replies == 1
    assert agent._runtime.turns == () and agent.action_history == []
    assert [t["outcome"] for t in _jsonl(tmp_path / "tool_calls.jsonl")] == [
        "superseded_by_timeout"
    ]


def test_an_abandoned_generation_keeps_its_own_candidates(tmp_path, monkeypatch):
    """Luna. Generation 1 (decision 0) is aborted while its observation is being sent, and it is
    still running when generation 2 (decision 1) prepares its own frame and is accepted. When
    generation 1 then builds its request, it offers decision 0's candidates, not decision 1's;
    the abort cancelled it, so its question is never sent."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "replay-dummy-key")
    with DecisionsStub(upstream="openai") as stub:
        episode = _luna_episode(stub, tmp_path)
        agent = episode.agent
        driver = agent.driver
        try:
            first, second = episode.frames.capture(0), episode.frames.capture(1)
            entered, release = threading.Event(), threading.Event()
            send_observation, start_generation = driver.send_observation, driver.start_generation

            async def held(observation, history):
                if observation.observation_id == first.metadata["observation_id"]:
                    entered.set()
                    await asyncio.to_thread(release.wait, WAIT_S)
                return await send_observation(observation, history)

            offered = []

            async def offering(request):
                offered.append((request.generation, request.canonical.selection.observation_id))
                return await start_generation(request)

            driver.send_observation, driver.start_generation = held, offering
            budget = episode.budget(first.tick_number)
            box, done = _in_worker(lambda: agent.generate(1, 0, first, budget))
            assert entered.wait(WAIT_S), "generation 1 sent no observation"
            agent.abort(1)
            agent._runtime._cleanup_future.result(timeout=WAIT_S)
            stub.load(scenario_decisions.WEB_LONG[1].script)
            accepted = agent.generate(2, 1, second, episode.budget(second.tick_number))
            agent.accept(2)
            release.set()
            abandoned = _joined(box, done)
            agent.discard(1)
        finally:
            episode.finish()
    ids = [frame.metadata["observation_id"] for frame in (first, second)]
    assert offered == [(2, ids[1]), (1, ids[0])]
    assert abandoned.cancelled and not accepted.cancelled
    assert stub.unload() == 0 and stub.unscripted == 0 and not stub.off_script
    assert len(stub.requests) == len(scenario_decisions.WEB_LONG[1].script)
    assert [turn.decision for turn in agent._runtime.turns] == [1]


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
            box, done = _in_worker(lambda: agent.stream_generate(tick=0))
            assert entered.wait(WAIT_S), "decision 0 asked no question"
            closed = agent.driver.client
            agent.abort()
            agent._runtime._cleanup_future.result(timeout=WAIT_S)
            assert closed.is_closed
            release.set()
            _joined(box, done)
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
