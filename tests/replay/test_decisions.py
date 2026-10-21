"""Replay baselines for the Choice Decisions APIs: openai_decisions (Luna) and jev_decisions (JEV).

Record with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it each run is
compared with tests/replay/baselines/.

Both APIs run through the one agent (harnesses/registry.py) on the wall clock only
(providers/preflight.py, check_wall_clock). Each decision is a sequence of Choice questions, one
HTTP request per attempt, and every request is recorded. JEV is never run on the game page: the
pre-run checks reject it (providers/preflight.py, check_choice_task).

Both catalog rows are experimental (benchmark_eligible False, providers/registry.py:54-64). The
runner refuses them unless ALLOW_EXPERIMENTAL_MODEL=1 (evaluation/runner.py:1444-1452), but that
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
from realtime_framework.entity.context_policy import LEDGER_OMISSION
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER
from realtime_framework.entity.model import FOLDED_VISUAL_PLACEHOLDER
from realtime_framework.providers.decisions.questions import GAME_INSTRUCTIONS
from realtime_framework.providers.decisions.questions import GROUP_NOTE
from realtime_framework.providers.decisions.questions import GUI_INSTRUCTIONS

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
# (providers/jev/decisions.py:31-33; registry.py:277-278, 299-300; decisions/protocol.py decide).
APIS = {
    "openai_decisions": ("gpt-6-luna-decisions", "openai"),
    "jev_decisions": ("jev-1.13", "typesafe"),
}

# With the default policies (entity/config.py:35-38) a request holding 21 screenshots folds the
# oldest 10 (harnesses/default/context.py:24-32), and one holding 51 turns archives the oldest 10
# (context.py:295-305). Decision k carries k earlier turns plus its own frame.
FIRST_FOLD, FOLDED = 20, 10
FIRST_ARCHIVE, ARCHIVED = 50, 10
# entity/context_policy.py:30
JEV_LEDGER_HEADER = "ARCHIVED EXECUTED ACTIONS (JEV lossless consecutive runs;"


@pytest.fixture(autouse=True)
def dummy_key(monkeypatch):
    """The YAMLs name OPENROUTER_API_KEY (openai_decisions.py:82); create_agent's .env loading
    only fills unset variables (providers/registry.py:38), so this dummy wins."""
    monkeypatch.setenv("OPENROUTER_API_KEY", DUMMY_KEY)


def _play(api, decisions, run_dir, *, task, instruction, scene=None, update=None):
    key, upstream = APIS[api]
    with DecisionsStub(upstream=upstream) as stub:
        # The runner's config (runner.py:1548-1553), with the YAML's base_url at the stub.
        config = catalog.to_model_config(key, fps=1.0, reasoning_effort="fixed").model_copy(
            update={"api_base_url": stub.base_url, **(update or {})}
        )
        episode = build_episode(
            config, run_dir, task=task, instruction=instruction, model_key=key, scene=scene
        )
        try:
            records, unused = run_decisions(episode, decisions, stub)
        finally:
            episode.agent.close()
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
    only for Luna (pin_route: providers/openai_decisions.py, providers/jev/decisions.py)."""
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
    (selection.py:356-414)."""
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
    (selection.py:165-229)."""
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
    joins the first string (decisions/history.py:54-79; default/request.py:53-55, 155-161)."""
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


def _check_jev_history(records: list[dict]) -> None:
    """JEV's digest reads only each older turn's observation id, action and feedback
    (default/request.py:138-141; decisions/history.py:113-126), and a fold keeps the observation id
    (entity/model.py:146-170). A fold therefore changes no request, and only the runtime counter
    shows it. The ledger is compacted into its own message (decisions/history.py:102-112)."""
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
    # The environment's refusal reaches the next request (harnesses/default/harness.py:1612-1614).
    assert records[1]["commit"]["detail"] == OUTSIDE_PAGE
    assert OUTSIDE_PAGE in json.dumps(_bodies(records[2])[0]["state"], ensure_ascii=False)
    _settle("game", header, records, tmp_path)


def test_luna_question_retry(tmp_path):
    """A transient failure is retried for that question alone (selection.py:313-354); the
    decision-level retry never runs for Decisions (providers/decisions/driver.py:45-51)."""
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
# needs two archived blocks; the second is below the system text and the terms alone.
ARCHIVE_WINDOWS = {"two_blocks": (40000, 3000), "exhausted": (32000, 4000)}


def _pressure_archives(run_dir: Path, kind: str = "pressure_archive") -> list[dict]:
    events = [
        json.loads(line) for line in (run_dir / "episode_trace.jsonl").read_text().splitlines()
    ]
    return [
        e["payload"]
        for e in events
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


def test_history_overflow_exhausted(tmp_path):
    """A request still over the window once every turn is archived and every archived action
    omitted is not sent: the decision is invalid, for its context rather than its provider, and
    the episode ends (harnesses/default/context.py _run_turn)."""
    header, records = _play_terms(script.TERMS_EXHAUSTED, ARCHIVE_WINDOWS["exhausted"], tmp_path)
    *_, failed = records
    assert failed["step"] == script.TERMS_ARCHIVE_DECISION and failed["requests"] == []
    assert failed["result"]["invalid_category"] == "harness"
    assert failed["result"]["invalid_reason"] == "context_exceeded_before_request"
    assert failed["result"]["invalid_detail"] == (
        "RequestExceedsContext('Task instructions and current observation exceed context "
        "capacity with every archived action omitted; use a larger endpoint window')"
    )
    omissions = _pressure_archives(tmp_path, "ledger_omission")
    assert [o["ledger"]["omitted_actions_after"] for o in omissions] == [10, 20, 24]
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


# The review's JEV window: from decision 15 the runtime archives turns to fit, from decision 24
# none is left, and from decision 28 the ledger alone outgrows it, since each archived action adds
# its record to the ledger and takes less than twice that from the digest.
LEDGER_WINDOW = (10024, 1024)


def test_history_overflow_ledger(tmp_path):
    """With every turn archived and the request still over the window, the earliest archived
    actions give way to a note, ten at a time, and the episode goes on (context policy item 8).
    Without the note, decision 28 was invalid without a request."""
    header, records = _play(
        "jev_decisions",
        script.LEDGER_OVERFLOW,
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        update=dict(zip(("context_window_tokens", "context_reserve_tokens"), LEDGER_WINDOW)),
    )
    assert not any(r["result"].get("invalid_reason") for r in records)
    assert not any(_runtime(r)["request_context"]["pressure"] for r in records)
    omissions = _pressure_archives(tmp_path, "ledger_omission")
    assert [o["scope"]["decision"] for o in omissions] == list(script.LEDGER_OMISSIONS)
    assert [o["ledger"]["omitted_actions_after"] for o in omissions] == [10, 20]
    assert all(o["request_context"]["pressure"] for o in omissions)

    def ledger(k: int) -> list[str]:
        return _bodies(records[k])[0]["state"]["history"][1]["content"].split("\n")

    first = script.LEDGER_OMISSIONS[0]
    assert all(line.startswith("{") for line in ledger(first - 1)[1:])
    for k in range(first, len(records)):
        header_line, note, *kept = ledger(k)
        omitted = 10 * sum(k >= decision for decision in script.LEDGER_OMISSIONS)
        assert header_line.startswith(JEV_LEDGER_HEADER)
        assert note == LEDGER_OMISSION.format(count=omitted)
        assert kept and all(line.startswith("{") for line in kept)
    _settle("history_overflow_ledger", header, records, tmp_path)
