"""Replay baselines for the Choice Decisions APIs: openai_decisions (Luna) and jev_decisions (JEV).

Record with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it each run is
compared with tests/replay/baselines/.

Both APIs run through JevHarness (harnesses/registry.py:13-16) on the wall clock only
(providers/registry.py:164-177, 411-416; harnesses/jev/harness.py:52-53, 71-73;
harnesses/jev/driver.py:62-63). Each decision is a sequence of Choice questions, one HTTP request
per attempt, and every request is recorded. JEV is never run on the game page: it rejects the
game prompt (harnesses/jev/harness.py:59-61).

Both catalog rows are experimental (benchmark_eligible False, providers/registry.py:76-94). The
runner refuses them unless ALLOW_EXPERIMENTAL_MODEL=1 (evaluation/runner.py:1435-1443), but that
is an argparse-level check that build_episode does not mirror, and nothing called here reads the
variable, so it is not set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import realtime_framework.providers.registry as catalog
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
from .stub_decisions import DecisionsStub

BASELINES = Path(__file__).parent / "baselines"
DUMMY_KEY = "replay-dummy-key"
# api -> (catalog key = configs/models/<key>.yaml, upstream provider the response must name).
# Luna's upstream is the YAML's upstream_provider; JEV's comes from its model profile
# (providers/registry.py:78-79, 363-364; providers/openai_decisions.py:207-208).
APIS = {
    "openai_decisions": ("gpt-6-luna-decisions", "openai"),
    "jev_decisions": ("jev-1.13", "typesafe"),
}

# With the default policies (entity/config.py:35-38) a request holding 21 screenshots folds the
# oldest 10 (harnesses/default/context.py:27-36), and one holding 51 turns archives the oldest 10
# (context.py:262-272). Decision k carries k earlier turns plus its own frame.
FIRST_FOLD, FOLDED = 20, 10
FIRST_ARCHIVE, ARCHIVED = 50, 10
JEV_LEDGER_HEADER = "ARCHIVED EXECUTED ACTIONS (JEV lossless consecutive runs;"  # jev/context.py:48


@pytest.fixture(autouse=True)
def dummy_key(monkeypatch):
    """The YAMLs name OPENROUTER_API_KEY (openai_decisions.py:61); create_agent's .env loading
    only fills unset variables (providers/registry.py:38), so this dummy wins."""
    monkeypatch.setenv("OPENROUTER_API_KEY", DUMMY_KEY)


def _play(api, decisions, run_dir, *, task, instruction, scene=None):
    key, upstream = APIS[api]
    with DecisionsStub(upstream=upstream) as stub:
        # The runner's config (runner.py:1539-1544), with the YAML's base_url at the stub.
        config = catalog.to_model_config(key, fps=1.0, reasoning_effort="fixed").model_copy(
            update={"api_base_url": stub.base_url}
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
    """One POST per question attempt (openai_decisions.py:184), the route block only for Luna
    (openai_decisions.py:83-88; providers/jev/decisions.py:17-18)."""
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
    (selection.py:268-326)."""
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
    (selection.py:156-212)."""
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
    """The transcript drops a folded turn's image and keeps a placeholder observation; the
    ledger joins the first string (openai_decisions.py:141-163; default/context.py:437-449)."""
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
    (jev/context.py:66-77), and a fold keeps the observation id (entity/model.py:102-126). A fold
    therefore changes no request, and only the runtime counter shows it. The ledger is compacted
    into its own message (jev/context.py:38-55)."""
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
    candidates (harnesses/jev/harness.py:59, 90-91; game_candidates.py:56-72), which JEV refuses,
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
    """The game prompt and game candidates (harnesses/jev/harness.py:58-68, 90-91;
    selection.py:59) with keys, pointer, hold and release choices."""
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
    # The environment's refusal reaches the next request (harnesses/default/harness.py:1536-1538).
    assert records[1]["commit"]["detail"] == OUTSIDE_PAGE
    assert OUTSIDE_PAGE in json.dumps(_bodies(records[2])[0]["state"], ensure_ascii=False)
    _settle("game", header, records, tmp_path)


def test_luna_question_retry(tmp_path):
    """A transient failure is retried for that question alone (selection.py:240-266); the
    decision-level retry never runs for Decisions (harnesses/jev/driver.py:46-52)."""
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


def test_luna_context_overflow(tmp_path):
    """A rejected group is bisected (selection.py:173-186); a rejected one-option question ends
    the decision, since canonical history is never reduced (selection.py:230-233)."""
    header, records = _play(
        "openai_decisions",
        script.LUNA_OVERFLOW,
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
    )
    split, failed = records
    assert _criteria_sizes(split) == [11, 5, 6, 2]
    assert [len(s["candidates"]) for s in _runtime(split)["jev_context_splits"]] == [11]
    assert _runtime(split)["jev_history_reductions"] == []
    assert _criteria_sizes(failed) == [20, 10, 5, 2, 1]
    assert failed["result"]["invalid_category"] == "provider"
    assert failed["result"]["invalid_reason"] == "provider_request_failed"
    assert failed["commit"] is None and failed["action_history_entry"] is None
    _settle("overflow", header, records, tmp_path)


def test_jev_context_overflow(tmp_path):
    """JEV bisects too; a rejected one-option question drops the oldest messages after the system
    message, for that question only (selection.py:214-238)."""
    header, records = _play(
        "jev_decisions",
        script.JEV_OVERFLOW,
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
    )
    reduced, failed = records[3], records[4]
    assert _criteria_sizes(reduced) == [11, 5, 2, 1, 1, 1, 3, 6, 4]
    assert [len(b["state"]["history"]) for b in _bodies(reduced)] == [3, 3, 3, 3, 2, 3, 3, 3, 3]
    assert [len(s["candidates"]) for s in _runtime(reduced)["jev_context_splits"]] == [11, 5, 2]
    assert _runtime(reduced)["jev_history_reductions"] == [
        {"phase": "gui_action", "removed_messages": 1, "remaining": 2}
    ]
    assert reduced["result"]["raw"] == '{"action": "click", "target": "P3"}'
    assert [len(b["state"]["history"]) for b in _bodies(failed)] == [3, 3, 3, 3, 2]
    assert failed["result"]["invalid_reason"] == "provider_request_failed"
    _settle("overflow", header, records, tmp_path)
