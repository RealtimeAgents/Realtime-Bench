"""Replay baselines for the Choice Decisions APIs: openai_decisions (Luna) and jev_decisions (JEV).

Record with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it each run is
compared with tests/replay/baselines/.

Both APIs run through the one agent (harnesses/registry.py) on the wall clock only
(providers/preflight.py, check_wall_clock). Each decision is a sequence of Choice questions, one
HTTP request per attempt, and every request is recorded. JEV is never run on the game page: the
pre-run checks reject it (providers/preflight.py, check_choice_task).

Both catalog rows are experimental (benchmark_eligible False, providers/registry.py:54-64). The
runner refuses them unless ALLOW_EXPERIMENTAL_MODEL=1 (evaluation/runner.py main), but that
is an argparse-level check that build_episode does not mirror, and nothing called here reads the
variable, so it is not set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import realtime_framework.entity.model as model_module
import realtime_framework.providers.registry as catalog
from realtime_framework.entity.context_policy import FIXED_TEXTS, LEDGER_OMISSION
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.clock import TokenAccounting
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER
from realtime_framework.entity.model import FOLDED_VISUAL_PLACEHOLDER
from realtime_framework.harnesses.default.prompts import FRAME_PROMPT
from realtime_framework.providers.decisions.questions import GAME_INSTRUCTIONS
from realtime_framework.providers.decisions.questions import GROUP_NOTE
from realtime_framework.providers.decisions.questions import GUI_INSTRUCTIONS
from realtime_framework.providers.jev.decisions import JevDecisionsClient
from realtime_framework.sandbox.observation.som import FRAME_INTERACTABLES_HEADER

from . import baseline, baseline_decisions
from . import scenario_decisions as script
from .build import build_episode
from .frames import ReplayTask
from .scenario import OUTSIDE_PAGE, TASK_INSTRUCTION
from .steps_decisions import run_decisions
from .stub_decisions import DecisionsStub, Failure

BASELINES = Path(__file__).parent / "baselines"
DUMMY_KEY = "replay-dummy-key"
# api -> (catalog key = configs/models/<key>.yaml, upstream provider the response must name).
# Luna's upstream is the YAML's upstream_provider; JEV's comes from its driver's model_defaults
# (providers/jev/decisions.py model_defaults; registry.py load_definitions;
# decisions/protocol.py decide).
APIS = {
    "openai_decisions": ("gpt-6-luna-decisions", "openai"),
    "jev_decisions": ("jev-1.13", "typesafe"),
}

# With the default policies (entity/config.py effective_visual_fold_policy,
# effective_detailed_history_policy) a request holding 21 screenshots folds the oldest 10
# (harnesses/default/history.py apply_active_visual_budget), and one holding 51 turns archives the
# oldest 10 (history.py History._retain). Decision k carries k earlier turns plus its own
# frame.
FIRST_FOLD, FOLDED = 20, 10
FIRST_ARCHIVE, ARCHIVED = 50, 10
# entity/context_policy.py JEV_LEDGER_HEADER
JEV_LEDGER_HEADER = "ARCHIVED EXECUTED ACTIONS (JEV lossless consecutive runs;"


@pytest.fixture(autouse=True)
def dummy_key(monkeypatch):
    """The YAMLs name OPENROUTER_API_KEY (OpenAIDecisionsClient.__init__); create_agent's .env
    loading only fills unset variables (providers/registry.py:38), so this dummy wins."""
    monkeypatch.setenv("OPENROUTER_API_KEY", DUMMY_KEY)


def _play(api, decisions, run_dir, *, task, instruction, scene=None, update=None):
    key, upstream = APIS[api]
    with DecisionsStub(upstream=upstream) as stub:
        # The runner's config (runner.py main), with the YAML's base_url at the stub.
        config = catalog.to_model_config(key, fps=1.0, reasoning_effort="fixed").model_copy(
            update={"api_base_url": stub.base_url, **(update or {})}
        )
        episode = build_episode(
            config, run_dir, task=task, instruction=instruction, model_key=key, scene=scene
        )
        try:
            records, unused = run_decisions(episode, decisions, stub)
        finally:
            episode.finish()
    assert stub.unscripted == 0, "the agent asked questions beyond the script"
    assert not stub.off_script, f"scripted choices were not offered: {stub.off_script}"
    assert not any(unused), f"script entries left unused per decision: {unused}"
    assert len(records) == len(decisions)
    header = {
        "record": "header",
        "api": api,
        "model_key": key,
        "config": config.model_dump(),
    }
    return header, records


def _settle(name: str, header: dict, records: list[dict], run_dir: Path) -> None:
    actual = baseline.Normalizer()(
        [{**header, "scenario": name}, *records, *baseline.trajectory(run_dir)]
    )
    path = BASELINES / f"{header['api']}-{name}.jsonl"
    if os.environ.get("REPLAY_RECORD") == "1":
        baseline_decisions.write(path, actual)
        return
    problems = baseline.compare(baseline_decisions.read(path), actual)
    assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
        problems[:60]
    )


def _bodies(record: dict) -> list[dict]:
    return [request["body"] for request in record["requests"]]


def _criteria_sizes(record: dict) -> list[int]:
    return [len(body["questions"]["next_action"]["criteria"]) for body in _bodies(record)]


def _runtime(record: dict) -> dict:
    return record["result"]["realtime_runtime"]


def _check_requests(api: str, records: list[dict]) -> None:
    """One POST per question attempt (providers/decisions/protocol.py decide), the route block
    only for Luna (deployment.pin_route: providers/openai_decisions.py,
    providers/jev/decisions.py)."""
    for record in records:
        for request in record["requests"]:
            assert request["path"] == "/api/alpha/decisions"
            assert request["headers"] == {
                "authorization": f"Bearer {DUMMY_KEY}",
                "content-type": "application/json",
            }
            assert ("provider" in request["body"]) == (api == "openai_decisions")


def _check_composition(records: list[dict]) -> None:
    """Decision 1 composes the name; decision 15 uses every other edit and cancels
    (selection.py ChoiceDecision._compose_text)."""
    typed = records[1]
    assert typed["result"]["raw"] == '{"action": "type", "text": "Ada Lovelace"}'
    composed = _runtime(typed)["jev_text_composition"]
    assert composed["status"] == "finish"
    assert [edit["choice"] for edit in composed["edits"]] == [
        script.ADA,
        script.SPACE,
        script.LOVELACE,
        "finish",
    ]
    assert [call["phase"] for call in _runtime(typed)["jev_calls"]] == ["gui_action"] + 4 * [
        "text_edit"
    ]
    cancelled = records[15]
    assert cancelled["result"]["raw"] == '{"action": "wait"}'
    edits = _runtime(cancelled)["jev_text_composition"]["edits"]
    assert [(edit["operation"], edit.get("unicode_hex")) for edit in edits] == [
        ("append", None),
        ("unicode", "0000E9"),
        ("delete", None),
        ("unicode", ""),
        ("clear", None),
        ("cancel", None),
    ]
    assert edits[1]["text"] == "é"


def _check_grouping(records: list[dict]) -> None:
    """261 candidates: groups of 255 and 6 with the group note, then their two winners
    (selection.py ChoiceDecision._choose)."""
    grouped = records[script.GROUPED_DECISION]
    assert len(grouped["requests"]) == 3
    assert _criteria_sizes(grouped) == [255, 6, 2]
    instructions = [body["questions"]["next_action"]["instructions"] for body in _bodies(grouped)]
    assert instructions == [GUI_INSTRUCTIONS + GROUP_NOTE] * 2 + [GUI_INSTRUCTIONS]
    calls = [(c["phase"], c["round"], c["group"]) for c in _runtime(grouped)["jev_calls"]]
    assert calls == [("gui_action", 0, 0), ("gui_action", 0, 1), ("gui_action", 1, 0)]
    assert grouped["result"]["raw"] == '{"action": "type", "text": "Ada Lovelace"}'
    assert grouped["result"]["usage"] == {"input_tokens": 3030, "output_tokens": 3}


def _check_history_counters(records: list[dict]) -> None:
    folded = [_runtime(r)["history_folded_visuals"] for r in records]
    archived = [_runtime(r)["history_archived_turns"] for r in records]
    assert next(k for k, count in enumerate(folded) if count) == FIRST_FOLD
    assert folded[FIRST_FOLD] == FOLDED
    assert next(k for k, count in enumerate(archived) if count) == FIRST_ARCHIVE
    assert archived[FIRST_ARCHIVE] == ARCHIVED


def _check_luna_history(records: list[dict]) -> None:
    """The transcript drops a folded turn's image and keeps a placeholder observation; the ledger
    joins the first string (decisions/history.py render_transcript; default/request.py
    build_request, _frame)."""
    states = [_bodies(r)[0]["state"] for r in records]
    images = [sum(isinstance(entry, dict) for entry in state) for state in states]
    placeholders = [
        sum(
            entry.count("Observation: " + FOLDED_VISUAL_PLACEHOLDER)
            for entry in state
            if isinstance(entry, str)
        )
        for state in states
    ]
    assert images[FIRST_FOLD - 1] == FIRST_FOLD and placeholders[FIRST_FOLD - 1] == 0
    assert images[FIRST_FOLD] == FIRST_FOLD + 1 - FOLDED and placeholders[FIRST_FOLD] == FOLDED
    ledgers = [state[0].partition(ARCHIVED_ACTION_LEDGER_HEADER + "\n")[2] for state in states]
    assert next(k for k, ledger in enumerate(ledgers) if ledger) == FIRST_ARCHIVE
    lines = ledgers[FIRST_ARCHIVE].split("\n")
    assert all(json.loads(line)["decision"] == n for n, line in enumerate(lines[:ARCHIVED], 1))
    assert not lines[ARCHIVED].startswith("{")  # the retained transcript follows the ledger


def _check_luna_frames(records: list[dict]) -> None:
    """Each observation in a transcript is the frame of its own decision, folded or not, and
    says nothing that stops being true once newer frames follow (context policy, since 6)."""
    for record in records:
        for body in _bodies(record):
            text = "".join(entry for entry in body["state"] if isinstance(entry, str))
            assert text.count("Observation: ") == text.count(
                "Observation: " + FRAME_PROMPT + FRAME_INTERACTABLES_HEADER
            ) + text.count("Observation: " + FOLDED_VISUAL_PLACEHOLDER)


def _check_jev_history(records: list[dict]) -> None:
    """JEV's digest reads only each older turn's observation id, action and feedback
    (default/request.py _digest_items; decisions/history.py render_digest), and a fold keeps the
    observation id (entity/model.py CanonicalTurn.fold_visual). A fold therefore changes no request,
    and only the runtime counter shows it. The ledger is compacted into its own message
    (decisions/history.py render_digest)."""
    for record in records:
        for body in _bodies(record):
            text = json.dumps(body, ensure_ascii=False)
            assert FOLDED_VISUAL_PLACEHOLDER not in text and "image_url" not in text
    histories = [_bodies(r)[0]["state"]["history"] for r in records]
    assert [len(h) for h in histories[FIRST_ARCHIVE - 1 : FIRST_ARCHIVE + 1]] == [3, 4]
    assert histories[FIRST_ARCHIVE][1]["content"].startswith(JEV_LEDGER_HEADER)
    assert not any(h[1]["content"].startswith(JEV_LEDGER_HEADER) for h in histories[:FIRST_ARCHIVE])


@pytest.mark.parametrize("api", list(APIS))
def test_long_scenario(api, tmp_path):
    """60 decisions on the web form: text composition, a grouped decision, the first fold and the
    first archive. Pointer composition is not in it: only the game prompt offers pointer
    candidates (harnesses/jev/observation.py; game_candidates.py:55-71), which JEV refuses,
    so it is in the Luna game snapshot."""
    header, records = _play(
        api, script.WEB_LONG, tmp_path, task=ReplayTask(), instruction=TASK_INSTRUCTION
    )
    assert [len(r["requests"]) for r in records] == [len(d.script) for d in script.WEB_LONG]
    assert all(r["result"]["inspection"]["valid"] for r in records)
    assert not any(r["result"].get("invalid_reason") for r in records)
    _check_requests(api, records)
    _check_composition(records)
    _check_grouping(records)
    _check_history_counters(records)
    if api == "openai_decisions":
        _check_luna_history(records)
        _check_luna_frames(records)
    else:
        _check_jev_history(records)
    _settle("long", header, records, tmp_path)


def test_luna_game_snapshot(tmp_path):
    """The game prompt (context policy item 11) and game candidates
    (harnesses/jev/observation.py:43, 70) with keys, pointer, hold and release choices."""
    header, records = _play(
        "openai_decisions",
        script.GAME_DECISIONS,
        tmp_path,
        task=script.ReplayGameTask(),
        instruction=script.GAME_INSTRUCTION,
        scene=script.GAME_SCENE,
    )
    _check_requests("openai_decisions", records)
    _check_luna_frames(records)
    assert [len(r["requests"]) for r in records] == [5, 5, 4, 5, 1, 4, 1, 1, 1, 1]
    assert all(
        _bodies(r)[0]["questions"]["next_action"]["instructions"] == GAME_INSTRUCTIONS
        for r in records
    )
    phases = [[call["phase"] for call in _runtime(r)["jev_calls"]] for r in records]
    assert phases[0] == ["gui_action", "pointer_x_range", "pointer_x_value"] + [
        "pointer_y_range",
        "pointer_y_value",
    ]
    assert phases[2] == ["gui_action", "pointer_x_range", "pointer_y_range", "pointer_y_value"]
    assert phases[5] == ["gui_action", "hold_key_1", "hold_key_2", "hold_duration"]
    raws = [json.loads(r["result"]["raw"]) for r in records]
    assert raws[:3] == [
        {"action": "click", "x": 336, "y": 108},
        {"action": "click", "x": 393, "y": 13},
        {"action": "mouse_down", "target": "R1@1000,512"},
    ]
    assert raws[5] == {"action": "hold_key", "key": "ArrowLeft+Space", "duration": 0.5}
    # The environment's refusal reaches the next request (harnesses/default/harness.py
    # DefaultHarness._report).
    assert records[1]["commit"]["detail"] == OUTSIDE_PAGE
    assert OUTSIDE_PAGE in json.dumps(_bodies(records[2])[0]["state"], ensure_ascii=False)
    _settle("game", header, records, tmp_path)


def test_luna_question_retry(tmp_path):
    """A transient failure is retried for that question alone (selection.py
    ChoiceDecision._decide_with_retries); the decision-level retry never runs for Decisions
    (providers/decisions/driver.py DecisionEncoder.classify_transport_error)."""
    header, records = _play(
        "openai_decisions", script.RETRY, tmp_path, task=ReplayTask(), instruction=TASK_INSTRUCTION
    )
    retried = records[1]
    bodies = _bodies(retried)
    assert len(bodies) == 6 and bodies[1] == bodies[2]
    (retry,) = _runtime(retried)["jev_transport_retries"]
    assert (retry["call"], retry["phase"], retry["kind"]) == (1, "text_edit", "transient")
    assert retried["result"]["raw"] == '{"action": "type", "text": "Ada Lovelace"}'
    assert "transport_reconnect_attempts" not in _runtime(retried)
    _settle("retry", header, records, tmp_path)


def test_luna_question_retries_spent(tmp_path, monkeypatch):
    """A question that fails four times (HTTP 503) spends the shared policy's three retries
    (selection.py _decide_with_retries), and the decision fails with an error that names the
    question and the retries spent."""
    monkeypatch.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    unavailable = Failure(503, "Provider returned error")
    _, (failed,) = _play(
        "openai_decisions",
        [script.decision(*[unavailable] * 4)],
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
    )
    assert len(failed["requests"]) == 4
    assert failed["commit"] is None and failed["action_history_entry"] is None
    assert failed["result"]["invalid_reason"] == "provider_unavailable"
    detail = failed["result"]["invalid_detail"]
    assert detail.startswith("TransportFailureError('transient after 3 transport retries: ")
    assert 'RuntimeError("gui_action question 0: DecisionsHTTPError(' in detail


def test_luna_decision_that_fails_after_a_question_keeps_its_usage(tmp_path):
    """Decision 19: decision 1's first question picks compose_text, and its first text-edit
    question gets HTTP 400 (fatal, not retried). The failed step's usage is that of the question
    it answered, marked incomplete, as the failed question's is not known; the episode's token
    totals take it as known but not complete. A failed step is not accepted, so neither the
    agent's usage nor the provider-usage audit counts it a second time."""
    key, upstream = APIS["openai_decisions"]
    with DecisionsStub(upstream=upstream) as stub:
        config = catalog.to_model_config(key, fps=1.0, reasoning_effort="fixed").model_copy(
            update={"api_base_url": stub.base_url}
        )
        episode = build_episode(
            config, tmp_path, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key=key
        )
        try:
            (accepted, failed), _ = run_decisions(
                episode,
                [
                    script.decision("click_P1"),
                    script.decision("compose_text", Failure(400, "bad request")),
                ],
                stub,
            )
        finally:
            episode.finish()
    result = failed["result"]
    assert result["invalid_reason"] == "provider_request_failed"
    lines = (tmp_path / "episode_trace.jsonl").read_text().splitlines()
    attempts = [
        e["payload"] for e in map(json.loads, lines) if e["event"] == "ModelAttemptFinished"
    ]
    questions = [a for a in attempts if a["kind"] == "choice_question"]
    assert [(a["scope"]["decision"], a["outcome"]) for a in questions] == [
        (0, "ok"),
        (1, "ok"),
        (1, "failed"),
    ]
    answered = questions[1]["usage"]
    assert result["usage"] == answered
    assert result["provider_usage_missing_reason"] == "failed_request_usage_not_reported"
    rows = [json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    assert [row["usage"]["provider_usage"] for row in rows if row["tick"] == 1] == [answered]
    generations = [a for a in attempts if a["kind"] == "generation"]
    assert [(a["outcome"], a["usage"]) for a in generations][-1] == ("failed", answered)
    assert failed["token_usage"] == accepted["token_usage"]
    rows = (tmp_path / "provider-usage.jsonl").read_text().splitlines()
    assert [(row["tick"], row["stale"]) for row in map(json.loads, rows)] == [(0, False)]
    # Settled as the runner settles each result (evaluation/episode_records.py account_tokens).
    accounting = TokenAccounting([episode_records.account_tokens(result, None)])
    record = result["token_accounting"]
    assert record["provider_usage_status"] == "partial"
    assert record["provider_usage_missing_reason"] == "failed_request_usage_not_reported"
    inputs = accounting.summary()["counts"]["provider_input_tokens"]
    assert (inputs["total"], inputs["known_total"], inputs["complete"]) == (
        None,
        answered["input_tokens"],
        False,
    )


def test_luna_screen_pointer_at_1000_is_the_last_column(tmp_path):
    """A whole-screen pointer composed at x = 1000 is the model image's last column, 767 of 768
    (ChoiceDecision._compose_pointer keeps a whole-screen point below the image size), not one
    past it; y = 512 becomes 221, round(432 * 512 / 1000)."""
    _, (record,) = _play(
        "openai_decisions",
        [script.decision("game_pointer_screen_click", "1000", "500", "512")],
        tmp_path,
        task=script.ReplayGameTask(),
        instruction=script.GAME_INSTRUCTION,
        scene=script.GAME_SCENE,
    )
    phases = [call["phase"] for call in _runtime(record)["jev_calls"]]
    assert phases == ["gui_action", "pointer_x_range", "pointer_y_range", "pointer_y_value"]
    assert json.loads(record["result"]["raw"]) == {"action": "click", "x": 767, "y": 221}


@pytest.mark.parametrize("api", list(APIS))
def test_candidate_overflow(api, tmp_path):
    """Too many candidates: a rejected group is bisected (selection.py _choose); a rejected
    one-option question ends the decision, since no history is dropped at question time."""
    header, records = _play(
        api, script.CANDIDATE_OVERFLOW, tmp_path, task=ReplayTask(), instruction=TASK_INSTRUCTION
    )
    split, failed = records
    assert _criteria_sizes(split) == [11, 5, 6, 2]
    assert [len(s["candidates"]) for s in _runtime(split)["jev_context_splits"]] == [11]
    assert _criteria_sizes(failed) == [20, 10, 5, 2, 1]
    assert len({json.dumps(body["state"]) for body in _bodies(failed)}) == 1
    assert failed["result"]["invalid_category"] == "provider"
    assert failed["result"]["invalid_reason"] == "provider_request_failed"
    assert failed["commit"] is None and failed["action_history_entry"] is None
    _settle("overflow", header, records, tmp_path)


# (context_window_tokens, context_reserve_tokens) small enough that the estimate of decision 12
# of the long script, and of no earlier one, exceeds the window less the reserve.
PRESSURE_WINDOWS = {"openai_decisions": (56000, 4000), "jev_decisions": (8192, 512)}
# JEV's declared window and reserve, in the UTF-8 bytes its estimate counts: TypeSafe's 32768
# tokens at 2.25 bytes a token (providers/jev/decisions.py model_defaults).
JEV_WINDOW, JEV_RESERVE = 73728, 16384


def _has_ledger(api: str, body: dict) -> bool:
    if api == "openai_decisions":
        return ARCHIVED_ACTION_LEDGER_HEADER in body["state"][0]
    return body["state"]["history"][1]["content"].startswith(JEV_LEDGER_HEADER)


def _first_decision(line: str) -> int:
    """The decision of a ledger line's first action: its own, or the first of a run the JEV
    digest compacts (providers/decisions/history.py compact_repeated_records)."""
    record = json.loads(line)
    if "decision" in record:
        return record["decision"]
    decisions = record["occurrences"]["decision"]
    return decisions["start"] if isinstance(decisions, dict) else decisions[0]


@pytest.mark.parametrize("api", list(APIS))
def test_history_overflow(api, tmp_path):
    """History too long: the runtime estimates each request against the declared window and,
    while it is over, archives the oldest turns into the ledger before the first question
    (harnesses/default/context.py _run_turn), once per block, each a HistoryChanged."""
    window, reserve = PRESSURE_WINDOWS[api]
    header, records = _play(
        api,
        script.HISTORY_OVERFLOW,
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        update={"context_window_tokens": window, "context_reserve_tokens": reserve},
    )
    k = script.HISTORY_OVERFLOW_ARCHIVE
    archived = [_runtime(r)["request_context_archived_turns"] for r in records]
    assert archived == [0] * k + [10] + [0] * (len(records) - k - 1)
    runtime = _runtime(records[k])
    before, after = runtime["request_context_before"], runtime["request_context"]
    assert before["pressure"] and before["soft_limit"] == window - reserve
    assert not any(_runtime(r)["request_context"]["pressure"] for r in records)
    assert after["input_tokens"] < before["input_tokens"]
    assert runtime["history_archived_actions_total"] == 10
    assert [_has_ledger(api, _bodies(r)[0]) for r in records] == [False] * k + [True] * (
        len(records) - k
    )
    events = [
        json.loads(line) for line in (tmp_path / "episode_trace.jsonl").read_text().splitlines()
    ]
    (change,) = [
        e["payload"]
        for e in events
        if e["event"] == "HistoryChanged" and e["payload"]["kind"] == "pressure_archive"
    ]
    assert change["scope"]["decision"] == k and change["policy_item"] == "pressure_archive"
    assert change["ledger"] == {"archived_actions_before": 0, "archived_actions_after": 10}
    _settle("history_overflow", header, records, tmp_path)


def test_jev_large_observation(tmp_path):
    """A 25-32 KB observation fits JEV's window. The estimate counts UTF-8 bytes, and the window
    is declared in them (providers/jev/decisions.py model_defaults): read as bytes, the 32768
    tokens it was declared as before would have left no room for this frame, and the decision
    would have been invalid before any request."""
    header, records = _play(
        "jev_decisions",
        script.LARGE_OBSERVATION,
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        scene=script.terms_scene(script.LARGE_OBSERVATION_DECISION + 1),
    )
    assert not any(r["result"].get("invalid_reason") for r in records)
    k = script.LARGE_OBSERVATION_DECISION
    current = _bodies(records[k])[0]["state"]["history"][-1]
    assert 25_000 <= len(json.dumps(current, ensure_ascii=False).encode()) <= 32_000
    contexts = [_runtime(r)["request_context"] for r in records]
    assert contexts[k]["soft_limit"] == JEV_WINDOW - JEV_RESERVE
    assert contexts[k]["input_tokens"] > 32768 - 1024  # the window less reserve declared before
    assert not any(context["pressure"] for context in contexts)
    assert not any(_runtime(r)["request_context_archived_turns"] for r in records)
    _settle("large_observation", header, records, tmp_path)


# JEV windows for decision 24 of the terms scripts, which estimates 38560 bytes with its 24 turns,
# 37743 with 14 and 36666 with 4 (each archived turn adds its action to the ledger): the first
# needs two archived blocks; the second is below the system text and the terms alone; the third
# holds the terms only once every turn is archived and ten actions are left out.
ARCHIVE_WINDOWS = {
    "two_blocks": (40000, 3000),
    "exhausted": (32000, 4000),
    "omission": (38000, 2000),
}


def _trace_events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "episode_trace.jsonl").read_text().splitlines()]


def _pressure_archives(run_dir: Path, kind: str = "pressure_archive") -> list[dict]:
    return [
        e["payload"]
        for e in _trace_events(run_dir)
        if e["event"] == "HistoryChanged" and e["payload"]["kind"] == kind
    ]


def _play_terms(decisions, window, run_dir):
    return _play(
        "jev_decisions",
        decisions,
        run_dir,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        scene=script.terms_scene(script.TERMS_ARCHIVE_DECISION + 1),
        update=dict(zip(("context_window_tokens", "context_reserve_tokens"), window)),
    )


def test_history_overflow_two_blocks(tmp_path):
    """While the request is over the window, the runtime archives the next block: decision 24
    sends its request after two (harnesses/default/context.py _run_turn)."""
    header, records = _play_terms(script.TERMS_ARCHIVE, ARCHIVE_WINDOWS["two_blocks"], tmp_path)
    k = script.TERMS_ARCHIVE_DECISION
    assert not any(r["result"].get("invalid_reason") for r in records)
    archived = [_runtime(r)["request_context_archived_turns"] for r in records]
    assert archived == [0] * k + [20] + [0] * (len(records) - k - 1)
    runtime = _runtime(records[k])
    assert runtime["request_context_compactions"] == 2
    assert runtime["request_context_before"]["pressure"]
    assert not runtime["request_context"]["pressure"]
    assert runtime["history_archived_actions_total"] == 20
    assert _has_ledger("jev_decisions", _bodies(records[k])[0])
    changes = _pressure_archives(tmp_path)
    assert [change["scope"]["decision"] for change in changes] == [k, k]
    assert [change["ledger"] for change in changes] == [
        {"archived_actions_before": 0, "archived_actions_after": 10},
        {"archived_actions_before": 10, "archived_actions_after": 20},
    ]
    _settle("history_overflow_two_blocks", header, records, tmp_path)


# (window, reserve) under which decision 24 of the terms script, with every turn archived, fits
# only once more than one block of its 24 archived actions is left out: it estimates 34643 with
# ten out, 33157 with twenty and 32470 with all 24. The first fits after two blocks; the second
# only after a third, short block of four.
OMISSION_BLOCKS = {
    "two": ((36000, 2000), [10, 20]),
    "short_last": ((34800, 2000), [10, 20, 24]),
}


@pytest.mark.parametrize("blocks", list(OMISSION_BLOCKS))
def test_history_overflow_omission_blocks(blocks, tmp_path):
    """One block of archived actions left out is not enough, but all of them would be: the count
    with every action left out lets the runtime go on, a block at a time, until the request
    fits, and the decision is sent. The step records the actions this request left out."""
    window, after = OMISSION_BLOCKS[blocks]
    _, records = _play_terms(script.TERMS_OMISSION, window, tmp_path)
    k = script.TERMS_ARCHIVE_DECISION
    assert not any(r["result"].get("invalid_reason") for r in records)
    omissions = _pressure_archives(tmp_path, "ledger_omission")
    assert [(o["scope"]["decision"], o["ledger"]["omitted_actions_after"]) for o in omissions] == [
        (k, n) for n in after
    ]
    assert _runtime(records[k])["request_context_omitted_actions"] == after[-1]
    assert not _runtime(records[k])["request_context"]["pressure"]
    for record in records[k:]:
        assert _runtime(record)["history_omitted_actions_total"] == after[-1]
        ledger = _bodies(record)[0]["state"]["history"][1]["content"].split("\n")
        assert ledger[1] == LEDGER_OMISSION.format(count=after[-1])


def test_history_overflow_exhausted_in_one_block(tmp_path, monkeypatch):
    """Decision 5 sees the terms with only five actions archived: one block leaves them all out,
    so no count with every action left out is needed before the request is given up. It counts
    its request three times: as it came, after its one archived block and after its one omitted
    block."""
    counts = []
    sized = JevDecisionsClient.request_context_status

    async def counted(self, request):
        counts.append(request.canonical.items[-1].block.observation_id)
        return await sized(self, request)

    monkeypatch.setattr(JevDecisionsClient, "request_context_status", counted)
    _, records = _play(
        "jev_decisions",
        script.TERMS_EXHAUSTED_EARLY,
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        scene=script.terms_scene(len(script.TERMS_EXHAUSTED_EARLY)),
        update=dict(zip(("context_window_tokens", "context_reserve_tokens"), (32000, 4000))),
    )
    *_, failed = records
    assert failed["result"]["invalid_reason"] == "context_exceeded_before_request"
    omissions = _pressure_archives(tmp_path, "ledger_omission")
    assert [o["ledger"]["omitted_actions_after"] for o in omissions] == [5]
    assert counts.count(failed["observation"]["observation_id"]) == 3


def test_history_overflow_omission_stays(tmp_path):
    """Actions left out of the ledger stay out for the rest of the episode, as archived turns
    stay archived, so the history the model sees stays stable: after decision 24 leaves ten out
    to fit the terms, decisions 25-27 still leave them out, though with the full ledger they
    would be far below the window."""
    _, records = _play_terms(script.TERMS_OMISSION, ARCHIVE_WINDOWS["omission"], tmp_path)
    k = script.TERMS_ARCHIVE_DECISION
    window, reserve = ARCHIVE_WINDOWS["omission"]
    assert not any(r["result"].get("invalid_reason") for r in records)
    omissions = _pressure_archives(tmp_path, "ledger_omission")
    assert [(o["scope"]["decision"], o["ledger"]["omitted_actions_after"]) for o in omissions] == [
        (k, 10)
    ]
    for record in records[k + 1 :]:
        runtime = _runtime(record)
        assert runtime["history_omitted_actions_total"] == 10
        assert runtime["request_context"]["input_tokens"] < (window - reserve) / 3
        ledger = _bodies(record)[0]["state"]["history"][1]["content"].split("\n")
        assert ledger[1] == LEDGER_OMISSION.format(count=10)
        assert _first_decision(ledger[2]) == 11  # decisions count from 1 in the ledger


def test_history_overflow_exhausted(tmp_path, monkeypatch):
    """A request still over the window once every turn is archived and every archived action
    omitted is not sent: the decision is invalid, a provider failure of its own name, and the
    episode ends (harnesses/default/context.py _run_turn). Once one block of actions is not
    enough, one count with all 24 left out shows that no block can help, so decision 24 counts
    its request six times: as it came, after each of three archived blocks, after the omitted
    block, and with every action omitted."""
    counts = []
    sized = JevDecisionsClient.request_context_status

    async def counted(self, request):
        counts.append(request.canonical.items[-1].block.observation_id)
        return await sized(self, request)

    monkeypatch.setattr(JevDecisionsClient, "request_context_status", counted)
    header, records = _play_terms(script.TERMS_EXHAUSTED, ARCHIVE_WINDOWS["exhausted"], tmp_path)
    *_, failed = records
    assert failed["step"] == script.TERMS_ARCHIVE_DECISION and failed["requests"] == []
    assert failed["result"]["invalid_category"] == "provider"
    assert failed["result"]["invalid_reason"] == "context_exceeded_before_request"
    assert failed["result"]["invalid_detail"] == (
        "RequestExceedsContext('Task instructions and current observation exceed context "
        "capacity with every archived action omitted; use a larger endpoint window')"
    )
    omissions = _pressure_archives(tmp_path, "ledger_omission")
    assert [o["ledger"]["omitted_actions_after"] for o in omissions] == [10]
    assert counts.count(failed["observation"]["observation_id"]) == 6
    assert failed["commit"] is None and failed["action_history_entry"] is None
    (attempt,) = [
        e["payload"]
        for e in map(json.loads, (tmp_path / "episode_trace.jsonl").read_text().splitlines())
        if e["event"] == "ModelAttemptFinished"
        and e["payload"]["scope"]["decision"] == script.TERMS_ARCHIVE_DECISION
    ]
    assert attempt["invalid_reason"] == "context_exceeded_before_request"
    changes = _pressure_archives(tmp_path)
    assert [change["ledger"]["archived_actions_after"] for change in changes] == [10, 20, 24]
    assert {change["scope"]["decision"] for change in changes} == {script.TERMS_ARCHIVE_DECISION}
    _settle("history_overflow_exhausted", header, records, tmp_path)


# (window, reserve) that the long script's ledger outgrows, and the decisions that then leave out
# the next ten archived actions. JEV's is the review's: from decision 15 the runtime archives turns
# to fit, from decision 24 none is left, and from decision 28 the ledger alone outgrows it, since
# each archived action adds its record to the ledger and takes less than twice that from the
# digest. Luna's holds the system text and the current frame and few archived actions, so from
# decision 1 every earlier turn is archived.
LEDGER_RUNS = {
    "jev_decisions": ((9986, 1024), (28, 34)),
    "openai_decisions": ((12939, 4000), (25, 31)),
}


def _ledger_lines(api: str, body: dict) -> tuple[str, list[str]]:
    """The ledger's header, and its note and records: JEV's own message, or the lines after the
    header in Luna's first transcript entry, up to the transcript that follows them."""
    if api == "jev_decisions":
        header, *lines = body["state"]["history"][1]["content"].split("\n")
        return header, lines
    lines = body["state"][0].partition(ARCHIVED_ACTION_LEDGER_HEADER + "\n")[2].split("\n")
    count = next(i for i, line in enumerate(lines) if not line.startswith(("{", "[")))
    return ARCHIVED_ACTION_LEDGER_HEADER, lines[:count]


@pytest.mark.parametrize("api", list(APIS))
def test_history_overflow_ledger(api, tmp_path):
    """With every turn archived and the request still over the window, the earliest archived
    actions give way to a note, ten at a time, and the episode goes on (context policy item 8).
    Without the note, JEV's decision 28 was invalid without a request."""
    window, omitting = LEDGER_RUNS[api]
    header, records = _play(
        api,
        script.LEDGER_OVERFLOW,
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        update=dict(zip(("context_window_tokens", "context_reserve_tokens"), window)),
    )
    assert not any(r["result"].get("invalid_reason") for r in records)
    assert not any(_runtime(r)["request_context"]["pressure"] for r in records)
    omissions = _pressure_archives(tmp_path, "ledger_omission")
    assert [o["scope"]["decision"] for o in omissions] == list(omitting)
    assert [o["ledger"]["omitted_actions_after"] for o in omissions] == [10, 20]
    assert all(o["request_context"]["pressure"] for o in omissions)
    first = omitting[0]
    omitted = [10 * sum(k >= decision for decision in omitting) for k in range(len(records))]
    # Each step records how many actions the ledger leaves out, once any is.
    totals = [_runtime(r).get("history_omitted_actions_total") for r in records]
    assert totals == [n or None for n in omitted]
    omitted_now = [_runtime(r).get("request_context_omitted_actions") for r in records]
    assert {k: n for k, n in enumerate(omitted_now) if n} == dict.fromkeys(omitting, 10)
    prepared = [
        e["payload"]["history"].get("omitted_actions_total")
        for e in _trace_events(tmp_path)
        if e["event"] == "GenerationPrepared"
    ]
    assert prepared == totals
    header_line, kept = _ledger_lines(api, _bodies(records[first - 1])[0])
    assert kept and all(line.startswith("{") for line in kept)
    if api == "jev_decisions":
        assert header_line == FIXED_TEXTS.jev_ledger_header
    for k in range(first, len(records)):
        header_line, (note, *kept) = _ledger_lines(api, _bodies(records[k])[0])
        if api == "jev_decisions":  # the header no longer says that every action is kept
            assert header_line == FIXED_TEXTS.jev_ledger_header_omitted
        assert note == LEDGER_OMISSION.format(count=omitted[k])
        assert kept and all(line.startswith("{") for line in kept)
        # The earliest actions are the ones left out; decisions count from 1 in the ledger.
        assert _first_decision(kept[0]) == omitted[k] + 1
    _settle("history_overflow_ledger", header, records, tmp_path)
