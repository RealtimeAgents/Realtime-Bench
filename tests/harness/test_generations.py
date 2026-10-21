"""DefaultHarness as the event kernel's agent (evaluation/shell.py KernelAgent), on the replay
stubs: a generation writes no history, AcceptResult writes it, DiscardResult drops it, and an
abort cancels only the generation it names."""

from __future__ import annotations

import asyncio
import json
import threading

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.actions import Action
from realtime_framework.evaluation.kernel.events import Outcome
from realtime_framework.harnesses.default.prompts import (
    ACTION_RESULT_NOTES,
    UNSUPPORTED_TERMINAL_ACTION_FEEDBACK,
)
from tests.replay import scenario_decisions
from tests.replay.build import build_episode
from tests.replay.frames import ReplayTask
from tests.replay.scenario import (
    CLICK_NAME,
    LONG_SCENARIO,
    OUTSIDE_PAGE,
    TASK_INSTRUCTION,
    replies,
)
from tests.replay.steps import Step
from tests.replay.stub import ChatCompletionsStub
from tests.replay.stub_decisions import DecisionsStub
from tests.replay.test_chat_completions import MODEL_KEY as CHAT_MODEL_KEY
from tests.replay.test_chat_completions import chat_completions_config
from tests.replay.test_openai_realtime import MODEL_KEY as REALTIME_MODEL_KEY
from tests.replay.test_openai_realtime import realtime_config
from tests.replay.ws_openai_realtime import RealtimeStub
from tests.replay.ws_openai_realtime import replies as realtime_replies

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


def _outcome(decision: int, status: str, detail: str = "", **fields) -> Outcome:
    """The kernel's Commit for ``decision``; the clock fields are not part of the feedback."""
    return Outcome(
        decision=decision,
        status=status,
        detail=detail,
        terminal_after_action=False,
        action_clock_position=float(decision),
        next_observation_tick=decision + 1,
        dropped_intermediate_frames=0,
        post_action_rendered=False,
        frame_age_s=0.0,
        token_delay_frames=None,
        action_output_tokens=None,
        post_horizon_output_tokens=0,
        **fields,
    )


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
    before the worker calls generate(). The generation then starts cancelled: it inspects,
    enters and sends nothing, and its discarded result leaves no history."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            agent.abort(1)
            pending = _generate(episode, 1, 0)
            agent.discard(1)
            frame = agent.frames.latest()
        finally:
            episode.finish()
    assert pending.cancelled
    assert pending.result["stale"] is True and pending.result["action"] is None
    assert stub.requests == [] and stub.unused_replies == 1
    assert agent._runtime.turns == () and agent.action_history == []
    assert frame is None and _jsonl(tmp_path / "tool_calls.jsonl") == []


def test_an_abort_before_the_reply_sends_no_request(tmp_path):
    """Stateless chat. The kernel aborts generation 1 after its observation step and before its
    request goes out. The abort closes the HTTP client; the generation does not open a new one
    for its request but stops before its reply, so the stub sees no request."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            entered, release = threading.Event(), threading.Event()
            send_observation = agent.driver.send_observation

            async def held(observation, history):
                refs = await send_observation(observation, history)
                entered.set()
                await asyncio.to_thread(release.wait, WAIT_S)
                return refs

            agent.driver.send_observation = held
            observation = episode.frames.capture(0)
            budget = episode.budget(observation.tick_number)
            box, done = _in_worker(lambda: agent.generate(1, 0, observation, budget))
            assert entered.wait(WAIT_S), "generation 1 reached no observation step"
            agent.abort(1)
            agent._runtime._cleanup_future.result(timeout=WAIT_S)
            release.set()
            pending = _joined(box, done)
            agent.discard(1)
        finally:
            episode.finish()
    assert pending.cancelled
    assert stub.requests == [] and stub.unused_replies == 1
    assert agent._runtime.turns == () and agent.action_history == []


def test_an_outcome_is_reported_on_its_accepted_decision(tmp_path):
    """AcceptResult writes decision 0, then Commit reports it: its own action record and turn
    take the outcome, and the next request carries the feedback after decision 0's reply."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:2])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            _generate(episode, 1, 0)
            agent.accept(1)
            agent.commit(_outcome(0, "rejected", OUTSIDE_PAGE))
            _generate(episode, 2, 1)
        finally:
            episode.finish()
    record = agent.action_history[0]
    assert (record["applied"], record["result"]) == (False, "rejected")
    assert record["executed"] == "wait()" and record["requested"].startswith("click ")
    feedback = (
        f"EXECUTION RESULT: actual=wait(); requested={record['requested']}; status=rejected."
        f" Detail: {OUTSIDE_PAGE}"
    )
    assert agent._runtime.turns[0].execution_feedback == feedback
    messages = json.loads(stub.requests[1].body)["messages"]
    replied = [m["content"] for m in messages].index(CLICK_NAME)
    assert messages[replied + 1] == {"role": "user", "content": feedback}


def test_a_done_is_reported_as_the_wait_it_ran(tmp_path):
    """The model answers done, which the loop does not support: the kernel runs a wait in its
    place and reports it (kernel/turns.py _accept). The record and the feedback name the wait as
    what ran and the done as what was asked, and a note says why."""
    with ChatCompletionsStub(replies([Step('{"action": "done"}')])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            _generate(episode, 1, 0)
            agent.accept(1)
            agent.commit(
                _outcome(
                    0,
                    "wait",
                    action=Action.wait(),
                    protocol_failure="unsupported_terminal_action",
                )
            )
        finally:
            episode.finish()
    (record,) = agent.action_history
    assert (record["requested"], record["executed"], record["result"]) == ("done", "wait", "wait")
    feedback = (
        "EXECUTION RESULT: actual=wait; requested=done; status=wait. "
        + UNSUPPORTED_TERMINAL_ACTION_FEEDBACK
    )
    assert agent._runtime.turns[0].execution_feedback == feedback


def test_a_move_over_the_cap_opens_the_next_frame_with_its_note(tmp_path):
    """Chat, whose feedback follows each reply. Decision 1's request is still out when its move
    runs over the per-move cap: the kernel aborts the generation, reports skipped_timeout and
    discards it. Decision 0 keeps its action record and its feedback; the note on decision 1
    has no turn to follow, so decision 2's frame opens with it."""
    arrived = threading.Event()

    def hold_second(index):
        if index == 1:
            arrived.set()
            return True
        return False

    steps = [LONG_SCENARIO[0], LONG_SCENARIO[2]]
    with ChatCompletionsStub(replies(steps), on_request=hold_second) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            _generate(episode, 1, 0)
            agent.accept(1)
            agent.commit(_outcome(0, "applied"))
            record = dict(agent.action_history[0])
            feedback = agent._runtime.turns[0].execution_feedback
            box, done = _in_worker(lambda: _generate(episode, 2, 1))
            assert arrived.wait(WAIT_S), "decision 1 sent no request"
            agent.abort(2)
            skipped = _joined(box, done)
            agent.commit(_outcome(1, "skipped_timeout"))
            agent.discard(2)
            _generate(episode, 3, 2)
            agent.accept(3)
        finally:
            episode.finish()
    assert skipped.cancelled and stub.held == [True]
    assert agent.action_history[0] == record and record["result"] == "applied"
    first, last = agent._runtime.turns
    assert (first.decision, first.execution_feedback, last.decision) == (0, feedback, 2)
    note = (
        "EXECUTION RESULT: actual=wait(); requested=none; status=skipped_timeout. "
        + ACTION_RESULT_NOTES["skipped_timeout"]
    )
    assert last.observation.feedback_prefix == note
    messages = json.loads(stub.requests[2].body)["messages"]
    replied = [m["content"] for m in messages].index(CLICK_NAME)
    assert messages[replied + 1] == {"role": "user", "content": feedback}
    assert messages[-1]["content"][0]["text"].startswith(note + "\n")


def test_a_move_over_the_cap_changes_no_earlier_record(tmp_path):
    """OpenAI Realtime, whose feedback opens the next frame. Decision 1's move runs over the
    per-move cap: the kernel aborts and discards its generation, and reports skipped_timeout for
    a decision of which nothing was accepted. Decision 0's action record and turn keep their
    own outcome. The next frame opens with decision 0's feedback, which only the discarded frame
    carried, then the note that decision 1's move was skipped."""
    steps = [LONG_SCENARIO[0], LONG_SCENARIO[2]]
    with RealtimeStub(realtime_replies(steps)) as stub:
        episode = build_episode(
            realtime_config(stub.base_url),
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=REALTIME_MODEL_KEY,
        )
        agent = episode.agent
        try:
            stub.wait_for_session_update()
            _generate(episode, 1, 0)
            agent.accept(1)
            agent.commit(_outcome(0, "applied"))
            record = dict(agent.action_history[0])
            feedback = agent._runtime.turns[0].execution_feedback
            agent.abort(2)
            skipped = _generate(episode, 2, 1)
            agent.commit(_outcome(1, "skipped_timeout"))
            agent.discard(2)
            _generate(episode, 3, 2)
            agent.accept(3)
        finally:
            episode.finish()
        stub.finish()
    assert skipped.cancelled
    assert agent.action_history[0] == record and record["result"] == "applied"
    assert [r["tick"] for r in agent.action_history] == [1, 2]
    first, last = agent._runtime.turns
    assert (first.decision, first.execution_feedback, last.decision) == (0, feedback, 2)
    note = (
        "EXECUTION RESULT: actual=wait(); requested=none; status=skipped_timeout. "
        + ACTION_RESULT_NOTES["skipped_timeout"]
    )
    prefix = f"{feedback}\n{note}"
    assert last.observation.feedback_prefix == prefix
    frames = [text for _, role, text, _ in stub.views[-1] if role == "user" and text]
    assert frames[-1].startswith(prefix)


def test_an_abandoned_generation_keeps_its_own_candidates(tmp_path, monkeypatch):
    """Luna. Generation 1 (decision 0) is aborted while its observation is being sent, and it is
    still running when generation 2 (decision 1) prepares its own frame and is accepted. When
    generation 1 then builds its request, which the runtime sizes against the context window, it
    offers decision 0's candidates, not decision 1's. The abort cancelled it, so it stops before
    its reply: no question of it is asked."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "replay-dummy-key")
    with DecisionsStub(upstream="openai") as stub:
        episode = _luna_episode(stub, tmp_path)
        agent = episode.agent
        driver = agent.driver
        try:
            first, second = episode.frames.capture(0), episode.frames.capture(1)
            entered, release = threading.Event(), threading.Event()
            send_observation, sized = driver.send_observation, driver.request_context_status
            start_generation = driver.start_generation

            async def held(observation, history):
                if observation.observation_id == first.metadata["observation_id"]:
                    entered.set()
                    await asyncio.to_thread(release.wait, WAIT_S)
                return await send_observation(observation, history)

            offered, started = [], []

            async def offering(request):
                offered.append((request.generation, request.canonical.selection.observation_id))
                return await sized(request)

            async def starting(request):
                started.append(request.generation)
                return await start_generation(request)

            driver.send_observation, driver.request_context_status = held, offering
            driver.start_generation = starting
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
    assert started == [2]
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
