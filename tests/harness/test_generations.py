"""DefaultHarness as the event kernel's agent (evaluation/shell.py KernelAgent), on the replay
stubs: a generation writes no history, AcceptResult writes it, DiscardResult drops it, and an
abort cancels only the generation it names (test_an_abort_names_one_generation)."""

from __future__ import annotations

import asyncio
import json
import threading
from copy import deepcopy

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.actions import Action
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.entity.context_policy import (
    JEV_CURRENT_PREFIX,
    JEV_HISTORY_PREFIX,
    LUNA_OBSERVATION_LABEL,
)
from realtime_framework.entity.model import FOLDED_VISUAL_PLACEHOLDER
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.agent_trace import load_body
from realtime_framework.evaluation.kernel.events import Outcome
from realtime_framework.harnesses.default.context import native_output_limit
from realtime_framework.harnesses.default.history import Retain
from realtime_framework.harnesses.default.prompts import (
    ACTION_RESULT_NOTES,
    UNSUPPORTED_TERMINAL_ACTION_FEEDBACK,
)
from tests.frames.test_grounding import moving_probe
from tests.replay import scenario_decisions
from tests.replay.build import build_episode, wait_for_abort
from tests.replay.frames import ReplayTask
from tests.replay.scenario import (
    CLICK_NAME,
    CLICK_SUBMIT,
    LONG_SCENARIO,
    OUTSIDE_PAGE,
    TASK_INSTRUCTION,
    replies,
)
from tests.replay.steps import Step, commit, decide
from tests.replay.stub import ChatCompletionsStub
from tests.replay.stub_decisions import DecisionsStub
from tests.replay.test_chat_completions import LEDGER_WINDOW, chat_completions_config
from tests.replay.test_chat_completions import MODEL_KEY as CHAT_MODEL_KEY
from tests.replay.test_openai_realtime import MODEL_KEY as REALTIME_MODEL_KEY
from tests.replay.test_openai_realtime import realtime_config
from tests.replay.ws_openai_realtime import RealtimeStub
from tests.replay.ws_openai_realtime import replies as realtime_replies

WAIT_S = 10.0  # bound on every cross-thread wait; none is reached when the code behaves
LUNA = "gpt-6-luna-decisions"  # configs/models/gpt-6-luna-decisions.yaml; upstream "openai"
JEV = "jev-1.13"  # configs/models/jev-1.13.yaml; upstream "typesafe"
# The feedback on a move over the per-move cap, of which nothing was accepted.
NOTE = (
    "EXECUTION RESULT: status=skipped_timeout. NOTE: a step timed out at the per-move cap before"
    " your answer arrived; no action was executed."
)


def _decisions_episode(stub, run_dir, model=LUNA):
    config = catalog.to_model_config(model, fps=1.0, reasoning_effort="fixed").model_copy(
        update={"api_base_url": stub.base_url}
    )
    return build_episode(
        config, run_dir, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key=model
    )


def _chat_episode(stub, run_dir, scene=None):
    return build_episode(
        chat_completions_config(stub.base_url, stream=True),
        run_dir,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        model_key=CHAT_MODEL_KEY,
        scene=scene,
    )


def _generate(episode, generation: int, decision: int):
    """The shell's StartGeneration for ``decision`` on its frame, on the calling thread."""
    observation = episode.frames.capture(decision)
    budget = episode.budget(observation.tick_number)
    return episode.agent.generate(generation, decision, observation, budget)


def _cancelled(pending) -> bool:
    """generate() hands a cancelled generation's result back stale (DefaultHarness._generate)."""
    return pending.result.get("stale", False)


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


def _trace(run_dir, event: str) -> list[dict]:
    rows = map(json.loads, (run_dir / "episode_trace.jsonl").read_text().splitlines())
    return [row["payload"] for row in rows if row["event"] == event]


def _hold_after_the_reply(driver):
    """Let each reply stream to its end, then hold it until ``release`` is set: an abort can
    land once the reply is back and before the runtime reads its cancel event. ``entered`` is
    set once one waits."""
    entered, release = threading.Event(), threading.Event()
    events = driver.events

    def held(generation):
        async def stream():
            async for event in events(generation):
                yield event
            if not release.is_set():
                entered.set()
                await asyncio.to_thread(release.wait, WAIT_S)

        return stream()

    driver.events = held
    return entered, release


def _hold_the_frame(agent, decision: int):
    """Hold ``decision``'s generation once it has prepared its frame, until ``release`` is set;
    ``entered`` is set once it waits."""
    entered, release = threading.Event(), threading.Event()
    prepare = agent._prepare_observations

    def held(observations, tick):
        prepared = prepare(observations, tick)
        if tick == decision:
            entered.set()
            release.wait(WAIT_S)
        return prepared

    agent._prepare_observations = held
    return entered, release


def _count_calls(target, name: str) -> list:
    """Wrap an async method; each call records its arguments."""
    calls, original = [], getattr(target, name)

    async def recorded(*args, **kwargs):
        calls.append((args, kwargs))
        return await original(*args, **kwargs)

    setattr(target, name, recorded)
    return calls


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


@pytest.mark.parametrize("api", ["stateless", "session"])
def test_the_agent_runs_its_driver_on_its_execution_thread(tmp_path, api):
    """DefaultHarness keeps the execution thread, whatever the driver: initialize() started it
    and the runtime runs on it; close() stops it once the runtime has closed the session."""
    stub_type, scripted = (
        (ChatCompletionsStub, replies([Step(CLICK_NAME)]))
        if api == "stateless"
        else (RealtimeStub, realtime_replies([Step(CLICK_NAME)]))
    )
    with stub_type(scripted) as stub:
        episode = (
            _chat_episode(stub, tmp_path)
            if api == "stateless"
            else build_episode(
                realtime_config(stub.base_url),
                tmp_path,
                task=ReplayTask(),
                instruction=TASK_INSTRUCTION,
                model_key=REALTIME_MODEL_KEY,
            )
        )
        thread = episode.agent._execution_thread
        try:
            assert episode.agent._runtime.execution is thread and thread.running
        finally:
            episode.finish()
        assert not thread.running


def test_a_generation_writes_nothing_until_it_is_accepted(tmp_path):
    """generate() hands back the decision result and holds what it would write; accept() writes
    the turn, the action record, the usage, the provider usage and the tool call's end."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            pending = _generate(episode, 1, 0)
            held = (agent.history.turns, list(agent.history.actions), agent.token_usage)
            written = (tmp_path / "provider-usage.jsonl").exists()
            tool_calls = _jsonl(tmp_path / "tool_calls.jsonl")
            agent.accept(1)
        finally:
            episode.finish()
    assert (pending.generation, pending.decision, _cancelled(pending)) == (1, 0, False)
    assert pending.result["raw"] == CLICK_NAME and pending.result["action"] is not None
    turns, history, usage = held
    assert (turns, history, usage["responses"], usage["output_tokens"]) == ((), [], 0, 0)
    assert not written and tool_calls == []
    (turn,) = agent.history.turns
    assert (turn.decision, turn.output, agent.history.index(0)) == (0, CLICK_NAME, 0)
    (record,) = agent.history.actions
    assert (record["tick"], record["action"]) == (1, CLICK_NAME)
    assert agent.token_usage["responses"] == 1
    assert [u["stale"] for u in _jsonl(tmp_path / "provider-usage.jsonl")] == [False]
    assert [t["outcome"] for t in _jsonl(tmp_path / "tool_calls.jsonl")] == ["completed"]


def test_the_agents_usage_counts_a_chat_replys_cached_and_reasoning_tokens(tmp_path):
    """The agent's cumulative usage (run_summary.json's adapter_token_usage, and the frame_usage
    of each InferenceCompleted) reads the cached and reasoning tokens of chat-format usage, under
    the names billing reads them by (evaluation/clock.py token_record)."""
    steps = LONG_SCENARIO[:2]
    scripted = [reply.usage for reply in replies(steps)]
    with ChatCompletionsStub(replies(steps)) as stub:
        episode = _chat_episode(stub, tmp_path)
        try:
            for k, step in enumerate(steps):
                observation = episode.frames.capture(k)
                decide(episode.agent, k, observation, episode.budget(observation.tick_number))
                commit(episode.agent, k, step.status, step.detail)
            usage = episode.agent.token_usage
        finally:
            episode.finish()
    cached = sum(u["prompt_tokens_details"]["cached_tokens"] for u in scripted)
    reasoning = sum(u["completion_tokens_details"]["reasoning_tokens"] for u in scripted)
    assert cached > 0 and reasoning > 0
    assert (usage["cached_input_tokens"], usage["reasoning_output_tokens"]) == (cached, reasoning)
    assert usage["input_tokens"] == sum(u["prompt_tokens"] for u in scripted)


def test_a_discarded_result_writes_no_history(tmp_path):
    """Generation 2's result is back and settled before the kernel gives up on it, as when the
    per-move cap fires as the reply lands (tests/shell/test_timing.py
    test_a_result_settled_across_the_cap_is_the_moves_only_record): the shell writes each
    trajectory row as it settles the result, before the kernel's verdict (shell.py
    _Shell._settle), and DiscardResult follows. Nothing of it joins the history; its provider
    usage is recorded as stale and its tool call as superseded, while its row, written by then,
    stays a decision's (U13, tests/differential/rules.py)."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:2])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            accepted = _generate(episode, 1, 0)
            agent.finalize_decision_audit(accepted.result)
            agent.accept(1)
            pending = _generate(episode, 2, 1)
            agent.finalize_decision_audit(pending.result)
            agent.discard(2)
            turns, history = agent.history.turns, list(agent.history.actions)
            usage = agent.token_usage
        finally:
            episode.finish()
    assert not _cancelled(pending) and pending.result["action"] is not None
    assert len(stub.requests) == 2
    assert [turn.decision for turn in turns] == [0]
    assert [record["tick"] for record in history] == [1]
    assert usage["responses"] == 1
    assert [u["stale"] for u in _jsonl(tmp_path / "provider-usage.jsonl")] == [False, True]
    outcomes = [t["outcome"] for t in _jsonl(tmp_path / "tool_calls.jsonl")]
    assert outcomes == ["completed", "superseded_by_timeout"]
    rows = [(r["tick"], r["type"]) for r in _jsonl(tmp_path / "trajectory.jsonl")]
    assert rows == [(0, "decision"), (1, "decision")]


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
        finally:
            episode.finish()
    assert _cancelled(pending)
    assert pending.result["stale"] is True and pending.result["action"] is None
    assert stub.requests == [] and stub.unused_replies == 1
    assert agent.history.turns == () and agent.history.actions == ()
    assert agent.frames._frames == {} and _jsonl(tmp_path / "tool_calls.jsonl") == []


def test_an_abort_before_the_reply_sends_no_request(tmp_path):
    """Stateless chat. The kernel aborts generation 1 after its observation step and before its
    request goes out. The abort closes the HTTP client; the generation does not open a new one
    for its request but stops before its reply, so the stub sees no request. Its tool call ends
    as superseded."""
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
            wait_for_abort(agent, WAIT_S)
            release.set()
            pending = _joined(box, done)
            agent.discard(1)
        finally:
            episode.finish()
    assert _cancelled(pending)
    assert stub.requests == [] and stub.unused_replies == 1
    assert agent.history.turns == () and agent.history.actions == ()
    outcomes = [t["outcome"] for t in _jsonl(tmp_path / "tool_calls.jsonl")]
    assert outcomes == ["superseded_by_timeout"]


@pytest.mark.parametrize("where", ["ready", "retention"])
def test_an_abandoned_generation_leaves_the_history_alone(tmp_path, where):
    """The kernel aborts generation 1 once its session is ready, or while the retention policy
    runs for it (RealtimeRuntime._run_turn). Aborted first, it applies no retention policy to
    the shared history; in either case it sends no observation and no request."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent, runtime = episode.agent, episode.agent._runtime
        entered, release = threading.Event(), threading.Event()
        ensure_ready, write = runtime.session.ensure_ready, agent.history.write
        sends = _count_calls(agent.driver, "send_observation")
        retentions = []

        async def ready(*args):
            await ensure_ready(*args)
            if where == "ready":
                entered.set()
                await asyncio.to_thread(release.wait, WAIT_S)

        def writing(change):
            if isinstance(change, Retain):
                retentions.append(change)
                if where == "retention":
                    entered.set()
                    release.wait(WAIT_S)
            return write(change)

        runtime.session.ensure_ready, agent.history.write = ready, writing
        try:
            box, done = _in_worker(lambda: _generate(episode, 1, 0))
            assert entered.wait(WAIT_S), f"generation 1 never reached {where}"
            agent.abort(1)
            release.set()
            pending = _joined(box, done)
            agent.discard(1)
        finally:
            episode.finish()
    assert _cancelled(pending) and stub.requests == [] and sends == []
    assert len(retentions) == (where == "retention")


def test_an_abort_after_the_reply_keeps_nothing(tmp_path):
    """Generation 1's reply is back when the kernel aborts it, before the runtime has read its
    cancel event: generate() reports it cancelled and without an action. Its attempt is
    recorded cancelled, its provider usage stale, its tool call superseded and its trajectory
    row discarded; the frame it prepared names its generation."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            entered, release = _hold_after_the_reply(agent.driver)
            box, done = _in_worker(lambda: _generate(episode, 1, 0))
            assert entered.wait(WAIT_S), "generation 1 got no reply"
            agent.abort(1)
            wait_for_abort(agent, WAIT_S)
            release.set()
            pending = _joined(box, done)
            agent.discard(1)
            agent.finalize_decision_audit(pending.result)
        finally:
            episode.finish()
    assert len(stub.requests) == 1
    assert _cancelled(pending) and pending.result["action"] is None
    assert agent.history.turns == () and agent.history.actions == ()
    attempts = [a for a in _trace(tmp_path, "ModelAttemptFinished") if a["kind"] == "generation"]
    assert [(a["outcome"], a["cancelled"]) for a in attempts] == [("cancelled", True)]
    assert [p["generation"] for p in _trace(tmp_path, "GenerationPrepared")] == [1]
    assert [u["stale"] for u in _jsonl(tmp_path / "provider-usage.jsonl")] == [True]
    outcomes = [t["outcome"] for t in _jsonl(tmp_path / "tool_calls.jsonl")]
    assert outcomes == ["superseded_by_timeout"]
    assert [r["type"] for r in _jsonl(tmp_path / "trajectory.jsonl")] == ["decision_discarded"]


def test_a_generation_records_the_request_it_was_prepared_from(tmp_path):
    """GenerationPrepared holds the canonical request the generation sent as its body, in the
    blob pack, in its audit form: the screenshot by its length and SHA-256. The trace holds only
    the body's locator."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        started = _count_calls(episode.agent.driver, "start_generation")
        try:
            decide(episode.agent, 0, episode.frames.capture(0), episode.budget(0))
        finally:
            episode.finish()
    (((request,), _),) = started
    (prepared,) = _trace(tmp_path, "GenerationPrepared")
    assert set(prepared["body"]) <= {"sha256", "bytes", "template", "spans"}
    record = json.loads(load_body(tmp_path / "episode_blobs.pack", prepared["body"]))
    assert record == json.loads(json.dumps(request.canonical.to_record()))
    (frame,) = [item["block"] for item in record["items"] if item["block"]["kind"] == "frame"]
    assert set(frame["image"]["data"]) == {"byte_length", "sha256"}


def test_the_result_export_shows_each_step_the_request_it_was_prepared_from(tmp_path):
    """The ATIF export of an episode (harbor_viewer.export_episode) gives each agent step the
    canonical request its generation was prepared from, read from the run's pack."""
    from realtime_framework.evaluation.harbor_viewer import export_episode

    steps = LONG_SCENARIO[:2]
    with ChatCompletionsStub(replies(steps)) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        started = _count_calls(agent.driver, "start_generation")
        try:
            for k, step in enumerate(steps):
                observation = episode.frames.capture(k)
                budget = episode.budget(observation.tick_number)
                result = agent.generate(k + 1, k, observation, budget).result
                # The shell names the row's decision before it is written (shell.py _settle).
                episode_records.annotate_result(
                    result,
                    decision=k,
                    observation=observation,
                    native_output_limit=agent.config.native_output_limit,
                    budget=budget,
                    tokens_per_frame=None,
                )
                agent.finalize_decision_audit(result)
                agent.accept(k + 1)
                commit(agent, k, step.status, step.detail)
        finally:
            episode.finish()
    task = {"task_id": "replay", "suite_id": "replay", "interaction_mode": "turn_based"}
    request = {"task": task, "mode": "default", "model": CHAT_MODEL_KEY}
    export_episode(tmp_path, tmp_path / "logs", request)
    trajectory = json.loads((tmp_path / "logs" / "trajectory.json").read_text())
    shown = [step["extra"]["canonical_request"] for step in trajectory["steps"][2::2]]
    sent = [json.loads(json.dumps(args[0].canonical.to_record())) for args, _ in started]
    assert len(sent) == len(steps) and shown == sent


def test_a_generation_requests_with_its_budget(tmp_path):
    """generate() hands its budget's provider and transport caps to the request; the transport
    cap is the request's max_tokens."""
    seen = []
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        start = agent.driver.start_generation

        async def recorded(request):
            seen.append((request.provider_output_token_cap, request.transport_output_token_cap))
            return await start(request)

        agent.driver.start_generation = recorded
        try:
            observation = episode.frames.capture(0)
            budget = DecisionBudget(None, 777, 778, "wall")
            assert episode.budget(observation.tick_number) != budget
            agent.generate(1, 0, observation, budget)
            agent.accept(1)
        finally:
            episode.finish()
    assert seen == [(777, 778)]
    assert json.loads(stub.requests[0].body)["max_tokens"] == 778


def test_an_abort_names_one_generation(tmp_path):
    """Generation 1 is aborted while it prepares its frame, and it is still running when the
    kernel aborts generation 2, whose worker has not started generate() yet. That abort names
    generation 2 alone, so generation 2 starts cancelled, and generation 1 stops before its
    request: the stub sees none."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:2])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            entered, release = _hold_the_frame(agent, 0)
            box, done = _in_worker(lambda: _generate(episode, 1, 0))
            assert entered.wait(WAIT_S), "generation 1 prepared no frame"
            agent.abort(1)
            agent.abort(2)
            second = _generate(episode, 2, 1)
            release.set()
            first = _joined(box, done)
            agent.discard(2)
            agent.discard(1)
        finally:
            episode.finish()
    assert _cancelled(first) and _cancelled(second)
    assert stub.requests == [] and stub.unused_replies == 2


def test_close_cancels_a_generation_in_flight(tmp_path):
    """close() at the end of the episode cancels and discards the generations that have not
    ended: one still preparing its frame then stops before its request and comes back
    cancelled, and the agent holds no generation after it. Its tool call, still open when the
    agent closed, is recorded incomplete."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            entered, release = _hold_the_frame(agent, 0)
            box, done = _in_worker(lambda: _generate(episode, 1, 0))
            assert entered.wait(WAIT_S), "generation 1 prepared no frame"
            agent.close()
            release.set()
            pending = _joined(box, done)
        finally:
            episode.finish()
    assert _cancelled(pending) and pending.result["action"] is None
    assert stub.requests == [] and agent._generations == {}
    calls = _jsonl(tmp_path / "tool_calls.jsonl")
    assert [(call.get("outcome"), call.get("incomplete")) for call in calls] == [(None, True)]


def test_an_accepted_reply_that_is_no_action_ends_its_tool_call_as_such(tmp_path):
    """A reply that is not one valid action is accepted as the wait the loop runs for it; its
    record is marked a parse failure and its tool call ends as an action protocol error."""
    with ChatCompletionsStub(replies([LONG_SCENARIO[6]])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            _generate(episode, 1, 0)
            agent.accept(1)
        finally:
            episode.finish()
    (record,) = agent.history.actions
    assert record["parse_failed"] and record["protocol_failure"]
    outcomes = [t["outcome"] for t in _jsonl(tmp_path / "tool_calls.jsonl")]
    assert outcomes == ["action_protocol_error"]


def _usage_rows(run_dir) -> list[tuple]:
    return [(u["tick"], u["stale"]) for u in _jsonl(run_dir / "provider-usage.jsonl")]


def _tool_outcomes(run_dir) -> list:
    return [call.get("outcome") for call in _jsonl(run_dir / "tool_calls.jsonl")]


def test_the_generation_an_episode_ends_on_is_discarded_with_its_records(tmp_path):
    """The episode ends as generation 2's reply lands: its result is back and settled before
    the deadline, and the kernel gets it only after (tests/shell/test_timing.py
    test_a_result_settled_across_the_deadline_does_not_land). The shell has written its
    trajectory row as it settled it (shell.py _Shell._settle); the kernel gives the generation
    up and discards the result. Generation 2's provider usage is recorded stale and its tool call
    superseded, its tokens stay out of token_usage, and its row stays a decision's (U13,
    tests/differential/rules.py)."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:2])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            accepted = _generate(episode, 1, 0)
            agent.finalize_decision_audit(accepted.result)
            agent.accept(1)
            usage = deepcopy(agent.token_usage)
            last = _generate(episode, 2, 1)
            agent.finalize_decision_audit(last.result)
            agent.discard(2)
        finally:
            episode.finish()
    assert not _cancelled(last) and len(stub.requests) == 2
    assert _usage_rows(tmp_path) == [(0, False), (1, True)]
    assert _tool_outcomes(tmp_path) == ["completed", "superseded_by_timeout"]
    rows = [(r["tick"], r["type"]) for r in _jsonl(tmp_path / "trajectory.jsonl")]
    assert rows == [(0, "decision"), (1, "decision")]
    assert agent.token_usage == usage


def test_the_row_of_a_call_the_episodes_end_gave_up_on_names_why(tmp_path):
    """N34: the shell's record of a call the task's end gave up on is synthesized and has no
    reply of the harness's; its trajectory row names why, which the model timing reads to
    leave it out (evaluation/clock.py summarize_model_timing)."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        try:
            abandoned = episode_records.abandoned_result(0.9, "task_end")
            episode.agent.finalize_decision_audit({**abandoned, "decision": 0})
            episode.agent.finalize_decision_audit({"decision": 1, "latency": 0.5})
        finally:
            episode.finish()
    rows = _jsonl(tmp_path / "trajectory.jsonl")
    assert [(row["decision"], row.get("harness_abort_reason")) for row in rows] == [
        (0, "task_end"),
        (1, None),
    ]
    assert "harness_abort_reason" not in rows[1]


def test_close_discards_the_generation_the_kernel_gave_no_verdict(tmp_path):
    """The episode ends as generation 2's reply lands: the shell settles its result, writing its
    trajectory row (shell.py _Shell._settle), and the kernel gets it only after Termination,
    which dispatches nothing more (kernel/core.py), so neither AcceptResult nor DiscardResult
    comes for it. Closing the agent discards it: its provider usage is recorded stale and its
    tool call superseded, its tokens stay out of token_usage, and its row stays a decision's
    (U13, tests/differential/rules.py)."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:2])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            accepted = _generate(episode, 1, 0)
            agent.finalize_decision_audit(accepted.result)
            agent.accept(1)
            usage = deepcopy(agent.token_usage)
            last = _generate(episode, 2, 1)
            agent.finalize_decision_audit(last.result)
        finally:
            episode.finish()
    assert not _cancelled(last) and len(stub.requests) == 2
    assert _usage_rows(tmp_path) == [(0, False), (1, True)]
    assert _tool_outcomes(tmp_path) == ["completed", "superseded_by_timeout"]
    rows = [(r["tick"], r["type"]) for r in _jsonl(tmp_path / "trajectory.jsonl")]
    assert rows == [(0, "decision"), (1, "decision")]
    assert agent.token_usage == usage and agent._generations == {}


def test_a_generation_discarded_while_it_runs_is_dropped_when_it_returns(tmp_path):
    """The episode ends while generation 2's worker is still running, its reply back: the kernel
    aborts it, and the shell discards it before the worker returns. Nothing is written then.
    The generation is dropped as generate() returns: its provider usage is recorded stale, its
    tool call superseded and its trajectory row discarded; its tokens stay out of
    token_usage, and the agent holds no generation after it."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:2])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            _generate(episode, 1, 0)
            agent.accept(1)
            usage = deepcopy(agent.token_usage)
            entered, release = _hold_after_the_reply(agent.driver)
            box, done = _in_worker(lambda: _generate(episode, 2, 1))
            assert entered.wait(WAIT_S), "generation 2 got no reply"
            agent.abort(2)
            agent.discard(2)
            written = (_usage_rows(tmp_path), _tool_outcomes(tmp_path))
            release.set()
            last = _joined(box, done)
            agent.finalize_decision_audit(last.result)
            generations = list(agent._generations)
        finally:
            episode.finish()
    assert _cancelled(last) and len(stub.requests) == 2
    assert written == ([(0, False)], ["completed"])
    assert _usage_rows(tmp_path) == [(0, False), (1, True)]
    assert _tool_outcomes(tmp_path) == ["completed", "superseded_by_timeout"]
    assert [r["type"] for r in _jsonl(tmp_path / "trajectory.jsonl")] == ["decision_discarded"]
    assert agent.token_usage == usage and generations == []


def test_a_generation_discarded_before_it_starts_keeps_nothing(tmp_path):
    """The episode ends right after StartGeneration, before the worker has called generate():
    the kernel aborts the generation and the shell discards it. It then starts cancelled, sends
    and writes nothing, and the agent holds no generation after it."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            agent.abort(1)
            agent.discard(1)
            pending = _generate(episode, 1, 0)
            generations = list(agent._generations)
        finally:
            episode.finish()
    assert _cancelled(pending) and stub.requests == [] and generations == []
    assert agent.history.actions == () and agent.history.turns == ()
    assert not (tmp_path / "provider-usage.jsonl").exists()
    assert _jsonl(tmp_path / "tool_calls.jsonl") == []


def test_a_late_abandoned_frame_leaves_the_next_decision_its_own(tmp_path):
    """Turn-based, on a page whose Submit button (P3) moves 120 px right in every frame.
    Generation 1 is aborted while it still inspects its frame, and enters that frame in the
    table only after generation 2 has entered its own and sent its request. The newest frame is
    then generation 1's, yet generation 2's click on P3 is grounded on generation 2's own
    frame."""
    inspecting, inspected, asked, answer = (threading.Event() for _ in range(4))

    def hold_the_reply(index):
        asked.set()
        assert answer.wait(WAIT_S)
        return False

    with ChatCompletionsStub(replies([Step(CLICK_SUBMIT)]), on_request=hold_the_reply) as stub:
        episode = _chat_episode(stub, tmp_path, scene={"probe": moving_probe})
        agent = episode.agent
        try:
            late, own = episode.frames.capture(0), episode.frames.capture(1)
            inspect = agent.tool_runtime.inspect

            def held(observation, **kwargs):
                if observation is late:
                    inspecting.set()
                    assert inspected.wait(WAIT_S)
                return inspect(observation, **kwargs)

            agent.tool_runtime.inspect = held
            late_budget, own_budget = (episode.budget(o.tick_number) for o in (late, own))
            box, done = _in_worker(lambda: agent.generate(1, 0, late, late_budget))
            assert inspecting.wait(WAIT_S), "generation 1 inspected no frame"
            agent.abort(1)
            wait_for_abort(agent, WAIT_S)
            box2, done2 = _in_worker(lambda: agent.generate(2, 1, own, own_budget))
            assert asked.wait(WAIT_S), "generation 2 sent no request"
            inspected.set()
            abandoned = _joined(box, done)
            answer.set()
            pending = _joined(box2, done2)
            agent.discard(1)
            agent.accept(2)
            table = agent.frames
        finally:
            episode.finish()
    own_frame, late_frame = table._frames[1], table._frames[2]
    assert (own_frame.observation_id, late_frame.observation_id) == (
        own.metadata["observation_id"],
        late.metadata["observation_id"],
    )
    assert len(table._frames) == 2 and _cancelled(abandoned)
    assert own_frame.marks["P3"]["point"] != late_frame.marks["P3"]["point"]
    click = pending.result["action"]
    assert [click.parameters["x"], click.parameters["y"]] == own_frame.marks["P3"]["point"]
    assert click.coordinate_transform == own_frame.geometry


def test_a_late_abandoned_generation_leaves_the_next_its_action_cap(tmp_path):
    """The runner audits the agent's action cap once it has a result
    (episode_records.py audit_output_caps), and only for the newest generation. Generation 1 is
    aborted while it prepares its frame and returns after generation 2, whose cap differs. The
    agent keeps generation 2's cap."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            entered, release = _hold_the_frame(agent, 0)
            first, second = episode.frames.capture(0), episode.frames.capture(1)
            box, done = _in_worker(
                lambda: agent.generate(1, 0, first, DecisionBudget(None, 111, None, "wall"))
            )
            assert entered.wait(WAIT_S), "generation 1 prepared no frame"
            agent.abort(1)
            accepted = agent.generate(2, 1, second, DecisionBudget(None, 222, None, "wall"))
            agent.accept(2)
            release.set()
            abandoned = _joined(box, done)
            agent.discard(1)
        finally:
            episode.finish()
    assert _cancelled(abandoned) and not _cancelled(accepted)
    limit = agent.config.native_output_limit
    assert native_output_limit(limit, 111) != native_output_limit(limit, 222)
    assert agent._frame_action_max_tokens == native_output_limit(limit, 222)


def test_a_generation_aborted_before_it_starts_leaves_the_next_its_action_cap(tmp_path):
    """Generation 1 is aborted before its worker calls generate(); generation 2 opens and
    returns, and only then does generation 1's worker call generate(), with a different cap. The
    runner audits generation 2's result after that (episode_records.py audit_output_caps): the
    agent still keeps generation 2's cap."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            first, second = episode.frames.capture(0), episode.frames.capture(1)
            agent.abort(1)
            accepted = agent.generate(2, 1, second, DecisionBudget(None, 222, None, "wall"))
            late = agent.generate(1, 0, first, DecisionBudget(None, 111, None, "wall"))
            kept = agent._frame_action_max_tokens
            agent.accept(2)
            agent.discard(1)
        finally:
            episode.finish()
    assert _cancelled(late) and not _cancelled(accepted)
    limit = agent.config.native_output_limit
    assert native_output_limit(limit, 111) != native_output_limit(limit, 222)
    assert kept == native_output_limit(limit, 222)


def test_an_outcome_is_reported_on_its_accepted_decision(tmp_path):
    """AcceptResult writes decisions 0 and 1 before Commit reports either. Each outcome goes to
    its own decision's action record and turn, not to the newest, and the next request carries
    each feedback after its own reply."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:3])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            _generate(episode, 1, 0)
            agent.accept(1)
            _generate(episode, 2, 1)
            agent.accept(2)
            agent.commit(_outcome(0, "rejected", OUTSIDE_PAGE))
            agent.commit(_outcome(1, "applied"))
            _generate(episode, 3, 2)
        finally:
            episode.finish()
    first, second = agent.history.actions
    assert (first["applied"], first["result"]) == (False, "rejected")
    assert first["executed"] == "wait()" and first["requested"].startswith("click ")
    assert (second["applied"], second["result"]) == (True, "applied")
    rejected = f"requested={first['requested']}; status=rejected. Detail: {OUTSIDE_PAGE}"
    applied = f"actual={second['executed']}; requested={second['executed']}; status=applied."
    feedback = [
        f"EXECUTION RESULT: actual=wait(); {rejected}",
        f"EXECUTION RESULT: {applied}",
    ]
    assert [turn.execution_feedback for turn in agent.history.turns] == feedback
    texts = [m["content"] for m in json.loads(stub.requests[2].body)["messages"]]
    for reply, text in zip((CLICK_NAME, LONG_SCENARIO[1].reply), feedback, strict=True):
        assert texts[texts.index(reply) + 1] == text


@pytest.mark.parametrize(
    ("status", "executed", "note"),
    [
        ("wait", "wait", UNSUPPORTED_TERMINAL_ACTION_FEEDBACK),
        ("skipped_budget", "wait()", ACTION_RESULT_NOTES["skipped_budget"]),
    ],
    ids=["ran", "skipped"],
)
def test_a_done_is_reported_as_the_wait_it_ran(tmp_path, status, executed, note):
    """The model answers done, which the loop does not support: the kernel runs a wait in its
    place and reports it (kernel/turns.py _accept). The record and the feedback name the wait as
    what ran and the done as what was asked, and a note says why. A wait that did not run, as
    when the deliberation budget ran out, is reported as any skipped action is."""
    with ChatCompletionsStub(replies([Step('{"action": "done"}')])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            _generate(episode, 1, 0)
            agent.accept(1)
            agent.commit(
                _outcome(
                    0,
                    status,
                    action=Action.wait(),
                    protocol_failure="unsupported_terminal_action",
                )
            )
        finally:
            episode.finish()
    (record,) = agent.history.actions
    assert (record["requested"], record["executed"], record["result"]) == ("done", executed, status)
    feedback = f"EXECUTION RESULT: actual={executed}; requested=done; status={status}. {note}"
    assert agent.history.turns[0].execution_feedback == feedback


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
            record = dict(agent.history.actions[0])
            feedback = agent.history.turns[0].execution_feedback
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
    assert _cancelled(skipped) and stub.held == [True]
    assert agent.history.actions[0] == record and record["result"] == "applied"
    first, last = agent.history.turns
    assert (first.decision, first.execution_feedback, last.decision) == (0, feedback, 2)
    assert last.unattached_feedback == NOTE
    messages = json.loads(stub.requests[2].body)["messages"]
    replied = [m["content"] for m in messages].index(CLICK_NAME)
    assert messages[replied + 1] == {"role": "user", "content": feedback}
    assert messages[-1]["content"][0]["text"].startswith(NOTE + "\n")


def test_a_decision_with_no_accepted_answer_reports_no_action(tmp_path):
    """Commit for decisions of which nothing was accepted names no action and changes no record:
    a move over the per-move cap says that it timed out, any other status that no action ran."""
    with ChatCompletionsStub(replies([])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        try:
            agent.commit(_outcome(0, "skipped_timeout"))
            agent.commit(_outcome(1, "none", "no current observation"))
        finally:
            episode.finish()
    other = (
        "EXECUTION RESULT: status=none. Detail: no current observation NOTE: no answer was"
        " accepted for a step; no action was executed."
    )
    assert agent.history.unattached == f"{NOTE}\n{other}"
    assert (agent.history.actions, agent.history.turns, stub.requests) == ((), (), [])


@pytest.mark.parametrize("api", ["chat", "luna"])
def test_the_note_outlasts_its_frame_s_screenshot(tmp_path, monkeypatch, api):
    """Decision 1's move runs over the per-move cap, so decision 2's frame opens with the note.
    The episode goes on until that frame's screenshot folds (20 are kept; the 21st folds the
    oldest 10). The note stays before the placeholder, as the feedback after each reply stays."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "replay-dummy-key")
    last = 22
    if api == "chat":
        stub = ChatCompletionsStub(replies([LONG_SCENARIO[k] for k in range(last + 1) if k != 1]))
    else:
        stub = DecisionsStub(upstream="openai")
    with stub:
        episode = (
            _chat_episode(stub, tmp_path) if api == "chat" else _decisions_episode(stub, tmp_path)
        )
        agent = episode.agent
        try:
            for generation, decision in enumerate(range(last + 1), 1):
                if decision == 1:
                    agent.abort(generation)
                    assert _cancelled(_generate(episode, generation, decision))
                    agent.commit(_outcome(decision, "skipped_timeout"))
                    agent.discard(generation)
                    continue
                if api == "luna":
                    stub.load(scenario_decisions.WEB_LONG[decision].script)
                _generate(episode, generation, decision)
                agent.accept(generation)
                agent.commit(_outcome(decision, "applied"))
            (turn,) = [turn for turn in agent.history.turns if turn.decision == 2]
        finally:
            episode.finish()
    assert turn.visual_folded
    body = json.loads(stub.requests[-1].body)
    if api == "chat":
        texts = [message["content"] for message in body["messages"]]
        assert texts.count(f"{NOTE}\n{FOLDED_VISUAL_PLACEHOLDER}") == 1
        assert json.dumps(texts).count(NOTE) == 1
    else:
        text = "\n".join(entry for entry in body["state"] if isinstance(entry, str))
        assert text.count(NOTE) == 1
        assert f"{NOTE}\n{LUNA_OBSERVATION_LABEL}{FOLDED_VISUAL_PLACEHOLDER}" in text


def test_jev_opens_the_next_observation_with_the_note(tmp_path, monkeypatch):
    """JEV, whose feedback is part of its action digest. Decision 1's generation is aborted
    before it starts, reported skipped_timeout and discarded. Decision 2's current observation
    opens with the note. On decision 3 the digest keeps it before decision 2's action, and
    decision 0's entry keeps its own feedback."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "replay-dummy-key")
    with DecisionsStub(upstream="typesafe") as stub:
        episode = _decisions_episode(stub, tmp_path, JEV)
        agent = episode.agent
        try:
            for generation, decision in enumerate(range(4), 1):
                if decision == 1:
                    agent.abort(generation)
                    assert _cancelled(_generate(episode, generation, decision))
                    agent.commit(_outcome(decision, "skipped_timeout"))
                    agent.discard(generation)
                    continue
                stub.load(scenario_decisions.WEB_LONG[decision].script)
                _generate(episode, generation, decision)
                agent.accept(generation)
                agent.commit(_outcome(decision, "applied"))
            turns = agent.history.turns
        finally:
            episode.finish()
    ids = [turn.observation.observation_id for turn in turns]
    assert [turn.decision for turn in turns] == [0, 2, 3]
    _, second, third = (json.loads(r.body)["state"]["history"] for r in stub.requests)
    assert second[-1]["content"].startswith(f"{NOTE}\n{JEV_CURRENT_PREFIX}")
    assert third[-1]["content"].startswith(JEV_CURRENT_PREFIX)
    digest = json.loads(third[-2]["content"].removeprefix(JEV_HISTORY_PREFIX))
    assert [entry["observation_id"] for entry in digest] == [ids[0], ids[1], ids[1]]
    assert digest[0]["execution_feedback"] == turns[0].execution_feedback
    assert digest[1] == {"observation_id": ids[1], "execution_feedback": NOTE}
    assert digest[2]["execution_feedback"] == turns[1].execution_feedback


def test_a_move_over_the_cap_changes_no_earlier_record(tmp_path):
    """OpenAI Realtime, whose feedback is an item of its own after each reply. Decision 1's
    reply is back when its move runs over the per-move cap: the kernel aborts and discards its
    generation, and reports skipped_timeout for a decision of which nothing was accepted.
    Decision 0's action record and turn keep their own outcome, and its feedback item follows
    its reply, ahead of decision 1's frame and in the replay after the abort. The note on
    decision 1 has no turn to follow, so the next frame opens with it."""
    steps = [LONG_SCENARIO[0], LONG_SCENARIO[2], LONG_SCENARIO[3]]
    with RealtimeStub(realtime_replies(steps)) as stub:
        episode = build_episode(
            realtime_config(stub.base_url),
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=REALTIME_MODEL_KEY,
        )
        agent = episode.agent
        opened, run_turn = {}, agent._runtime.run_turn

        def recorded(*args, **kwargs):
            opened[kwargs["decision"]] = kwargs["unattached"]
            return run_turn(*args, **kwargs)

        agent._runtime.run_turn = recorded
        try:
            stub.wait_for_session_update()
            _generate(episode, 1, 0)
            agent.accept(1)
            agent.commit(_outcome(0, "applied"))
            record = dict(agent.history.actions[0])
            feedback = agent.history.turns[0].execution_feedback
            entered, release = _hold_after_the_reply(agent.driver)
            box, done = _in_worker(lambda: _generate(episode, 2, 1))
            assert entered.wait(WAIT_S), "decision 1 got no reply"
            agent.abort(2)
            wait_for_abort(agent, WAIT_S)
            release.set()
            skipped = _joined(box, done)
            agent.commit(_outcome(1, "skipped_timeout"))
            agent.discard(2)
            _generate(episode, 3, 2)
            agent.accept(3)
        finally:
            episode.finish()
        stub.finish()
    assert _cancelled(skipped)
    # Its reply was back: its usage is recorded, stale.
    usage = [(u["tick"], u["stale"]) for u in _jsonl(tmp_path / "provider-usage.jsonl")]
    assert usage == [(0, False), (1, True), (2, False)]
    # Decision 0's feedback went as its own item: decision 1's frame opens with none.
    assert opened[1] == ""
    assert agent.history.actions[0] == record and record["result"] == "applied"
    assert [r["tick"] for r in agent.history.actions] == [1, 2]
    first, last = agent.history.turns
    assert (first.decision, first.execution_feedback, last.decision) == (0, feedback, 2)
    assert last.unattached_feedback == NOTE
    for view in (stub.views[1], stub.views[-1]):
        items = [(role, text) for _, role, text, _ in view]
        replied = items.index(("assistant", CLICK_NAME))
        assert items[replied + 1] == ("user", feedback)
    assert items[-1][1].startswith(NOTE + "\n")


def test_an_abandoned_generation_keeps_its_own_candidates(tmp_path, monkeypatch):
    """Luna. Generation 1 (decision 0) is aborted while its observation is being sent, and it is
    still running when generation 2 (decision 1) prepares its own frame and is accepted. When
    generation 1 then builds its request, which the runtime sizes against the context window, it
    offers decision 0's candidates, not decision 1's. The abort cancelled it, so it stops before
    its reply: no question of it is asked."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "replay-dummy-key")
    with DecisionsStub(upstream="openai") as stub:
        episode = _decisions_episode(stub, tmp_path)
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
            wait_for_abort(agent, WAIT_S)
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
    assert _cancelled(abandoned) and not _cancelled(accepted)
    assert stub.unload() == 0 and stub.unscripted == 0 and not stub.off_script
    assert len(stub.requests) == len(scenario_decisions.WEB_LONG[1].script)
    assert [turn.decision for turn in agent.history.turns] == [1]


def test_an_abandoned_generation_archives_nothing_under_pressure(tmp_path):
    """Stateless chat. Generation 1 (decision 0) is aborted while the runtime sizes its request,
    which suspends where an endpoint counts the request itself (context_counter "vllm",
    providers/litellm.py ChatCompletionsDriver.request_context_status). Generation 2 (decision 1)
    runs and is accepted meanwhile. Generation 1's count then reports pressure: the history it
    would archive is the next generation's, so it stops and leaves the turn, the ledger and the
    trace as they are."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[1:2])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        driver = agent.driver
        try:
            first, second = episode.frames.capture(0), episode.frames.capture(1)
            entered, release = threading.Event(), threading.Event()
            sized = driver.request_context_status

            async def counted(request):
                status = dict(await sized(request))
                if request.generation == 1 and not entered.is_set():
                    entered.set()
                    await asyncio.to_thread(release.wait, WAIT_S)
                    status["pressure"] = True
                return status

            driver.request_context_status = counted
            budget = episode.budget(first.tick_number)
            box, done = _in_worker(lambda: agent.generate(1, 0, first, budget))
            assert entered.wait(WAIT_S), "generation 1 sized no request"
            agent.abort(1)
            wait_for_abort(agent, WAIT_S)
            accepted = agent.generate(2, 1, second, episode.budget(second.tick_number))
            agent.accept(2)
            release.set()
            abandoned = _joined(box, done)
            agent.discard(1)
            turns = [turn.decision for turn in agent.history.turns]
            archived = agent.history.archived
        finally:
            episode.finish()
    assert _cancelled(abandoned) and not _cancelled(accepted)
    assert len(stub.requests) == 1
    assert (turns, archived) == ([1], 0)
    changes = [e["payload"] for e in _jsonl(tmp_path / "episode_trace.jsonl")]
    assert not [c for c in changes if c.get("kind") in {"pressure_archive", "ledger_omission"}]


def test_an_abandoned_generation_omits_nothing_under_pressure(tmp_path):
    """As above, with every turn archived when the aborted generation's count comes back.
    Generation 1 (decision 0) is accepted. Generation 2 (decision 1) is aborted while its request
    is sized. Generation 3 (decision 2) archives decision 0's turn to fit a small window
    (tests/replay/test_chat_completions.py LEDGER_WINDOW) and waits before its reply. Generation
    2's count then reports pressure with no turn left: it would cut the ledger, which is the next
    generation's, so it stops and leaves no action out."""
    with ChatCompletionsStub(replies([LONG_SCENARIO[0], LONG_SCENARIO[2]])) as stub:
        config = chat_completions_config(stub.base_url, stream=True).model_copy(
            update=dict(zip(("context_window_tokens", "context_reserve_tokens"), LEDGER_WINDOW))
        )
        episode = build_episode(
            config,
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=CHAT_MODEL_KEY,
        )
        agent = episode.agent
        driver = agent.driver
        try:
            frames = [episode.frames.capture(k) for k in range(3)]
            budgets = [episode.budget(frame.tick_number) for frame in frames]
            agent.generate(1, 0, frames[0], budgets[0])
            agent.accept(1)
            sizing, sized_release = threading.Event(), threading.Event()
            replying, reply_release = threading.Event(), threading.Event()
            sized, start_generation = driver.request_context_status, driver.start_generation

            async def counted(request):
                status = dict(await sized(request))
                if request.generation == 2 and not sizing.is_set():
                    sizing.set()
                    await asyncio.to_thread(sized_release.wait, WAIT_S)
                    status["pressure"] = True
                return status

            async def starting(request):
                if request.generation == 3:
                    replying.set()
                    await asyncio.to_thread(reply_release.wait, WAIT_S)
                return await start_generation(request)

            driver.request_context_status, driver.start_generation = counted, starting
            box2, done2 = _in_worker(lambda: agent.generate(2, 1, frames[1], budgets[1]))
            assert sizing.wait(WAIT_S), "generation 2 sized no request"
            agent.abort(2)
            wait_for_abort(agent, WAIT_S)
            box3, done3 = _in_worker(lambda: agent.generate(3, 2, frames[2], budgets[2]))
            assert replying.wait(WAIT_S), "generation 3 reached no reply"
            history = (
                [turn.decision for turn in agent.history.turns],
                agent.history.archived,
            )
            sized_release.set()
            abandoned = _joined(box2, done2)
            reply_release.set()
            accepted = _joined(box3, done3)
            agent.discard(2)
            agent.accept(3)
        finally:
            episode.finish()
    assert history == ([], 1)  # generation 3 archived decision 0's turn
    assert _cancelled(abandoned) and not _cancelled(accepted)
    assert len(stub.requests) == 2
    changes = [e["payload"] for e in _jsonl(tmp_path / "episode_trace.jsonl")]
    assert not [c for c in changes if c.get("kind") == "ledger_omission"]
    assert [c["scope"]["generation"] for c in changes if c.get("kind") == "pressure_archive"] == [3]


def test_an_abandoned_generation_omits_no_second_block(tmp_path):
    """Stateless chat, with pressure reported by the counts the test chooses. Generations 1-12
    are accepted, each archiving the turn before it, so twelve actions are archived. Generation
    13 archives decision 11's turn, leaves ten actions out and is still over, so it counts the
    request with every action left out. It is aborted while that count is under way, and the
    count fits: the ledger may be the next generation's by then, so it stops with the ten
    left out and no more."""
    steps = [LONG_SCENARIO[0]] * 12
    with ChatCompletionsStub(replies(steps)) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = episode.agent
        driver = agent.driver
        try:
            probing, release = threading.Event(), threading.Event()
            sized, counts = driver.request_context_status, {}

            async def counted(request):
                status = dict(await sized(request))
                count = counts[request.generation] = counts.get(request.generation, 0) + 1
                if request.generation == 13 and count == 4:
                    probing.set()
                    await asyncio.to_thread(release.wait, WAIT_S)
                over = 3 if request.generation == 13 else 1 if request.generation > 1 else 0
                status["pressure"] = count <= over
                return status

            driver.request_context_status = counted
            for generation, decision in enumerate(range(len(steps)), 1):
                _generate(episode, generation, decision)
                agent.accept(generation)
                agent.commit(_outcome(decision, "applied"))
            box, done = _in_worker(lambda: _generate(episode, 13, 12))
            assert probing.wait(WAIT_S), "generation 13 counted no request with every action out"
            agent.abort(13)
            wait_for_abort(agent, WAIT_S)
            release.set()
            abandoned = _joined(box, done)
            agent.discard(13)
            history = (agent.history.archived, agent.history.omitted)
        finally:
            episode.finish()
    assert _cancelled(abandoned)
    assert len(stub.requests) == len(steps)
    assert history == (12, 10)
    changes = [e["payload"] for e in _jsonl(tmp_path / "episode_trace.jsonl")]
    omissions = [
        (c["scope"]["generation"], c["ledger"]["omitted_actions_after"])
        for c in changes
        if c.get("kind") == "ledger_omission"
    ]
    assert omissions == [(13, 10)]


def test_a_decision_after_an_abort_sends_its_questions(tmp_path, monkeypatch):
    """Luna, aborted while decision 0's question is under way, as the kernel's per-move cap
    does. The abort closes the Decisions HTTP client (providers/openai_decisions.py close), and
    the held question then fails on it. Decision 1 connects again first
    (the session's ensure_ready), which opens a new client, so its questions are sent and
    answered."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "replay-dummy-key")
    with DecisionsStub(upstream="openai") as stub:
        episode = _decisions_episode(stub, tmp_path)
        agent = episode.agent
        try:
            entered, release = _hold_questions(agent.driver)
            box, done = _in_worker(lambda: _generate(episode, 1, 0))
            assert entered.wait(WAIT_S), "decision 0 asked no question"
            closed = agent.driver.client
            agent.abort(1)
            wait_for_abort(agent, WAIT_S)
            assert closed.is_closed
            release.set()
            _joined(box, done)
            agent.discard(1)
            assert stub.requests == []
            stub.load(scenario_decisions.WEB_LONG[1].script)
            result = _generate(episode, 2, 1).result
            agent.accept(2)
        finally:
            episode.finish()
    assert not result.get("invalid_reason"), result.get("invalid_detail")
    assert result["raw"] == '{"action": "type", "text": "Ada Lovelace"}'
    assert len(stub.requests) == len(scenario_decisions.WEB_LONG[1].script)
    assert stub.unscripted == 0 and not stub.off_script
