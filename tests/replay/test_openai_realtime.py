"""Replay baselines for the openai_realtime API, one WebSocket session per script.

This API has no request body. A baseline holds the client events of each decision in the order
they were sent, per connection; the stub's conversation at each response.create; and the parsed
action, usage and reported execution result of each decision. GenerationRequest.canonical is
built every turn (harnesses/default/context.py:480-487) but the driver takes only its newest
frame's id from it (providers/openai_realtime.py:355-357), and the trajectory's input_prompt
shows the archived action ledger as a user message (harnesses/default/harness.py:1127-1133;
providers/litellm.py:329-332), the role the wire item has had since context policy version 3
(openai_realtime_session.py:205-221, 292-297; version 1 sent a system item). The canonical
request is not recorded; the trajectory rows are, with input_prompt as a digest (baseline.py).

All scripts run on the finite token clock with B=1024 and 61 ticks: decision k may emit
min(32000, (61 - k) * 1024) tokens (evaluation/runner.py:866-881). That is the native limit up
to decision 29 and less from decision 30 on. The driver sends any cap above 4096 as "inf"
(openai_realtime.py:214-220), so the last three response.create calls carry 4096, 3072, 2048.

Record a baseline with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it
the run is compared with tests/replay/baselines/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER
from realtime_framework.entity.model import FOLDED_VISUAL_PLACEHOLDER
from realtime_framework.harnesses.default.prompts import TEXT_ACTION_FRAME_PROMPT
from realtime_framework.providers import registry as catalog

from . import baseline
from .build import build_episode
from .frames import ReplayTask
from .scenario import LONG_SCENARIO, TASK_INSTRUCTION
from .ws_openai_realtime import RealtimeStub, replies, run_steps

BASELINES = Path(__file__).parent / "baselines"
MODEL_KEY = "gpt-realtime-2"  # configs/models/gpt-realtime-2.yaml
STUB_API_KEY = "sk-replay-stub"  # a placeholder; every handshake must carry it
TOKENS_PER_FRAME, MAX_TICKS = 1024, 61

# With the default policies (entity/config.py:35-38) a decision that would hold 21 screenshots
# folds the oldest 10 (harnesses/default/context.py:24-32), and one that would hold 51 turns
# archives the oldest 10 (context.py:295-305). Decision k holds k earlier turns plus its own, so
# decision 50 archives turns 0-9 and also folds turns 30-39. Counted from 1, these are steps 21,
# 31, 41 and 51.
FOLDS = {20: range(0, 10), 30: range(10, 20), 40: range(20, 30), 50: range(30, 40)}
ARCHIVE, ARCHIVED = 50, range(0, 10)


def realtime_config(base_url: str) -> ModelConfig:
    """The runner's ModelConfig for the catalog key (evaluation/runner.py:1549-1553), pointed at
    the stub. The placeholder key keeps any OPENAI_API_KEY in the environment or the .env from
    being used (providers/openai_realtime.py:161-162)."""
    config = catalog.to_model_config(
        MODEL_KEY, fps=1.0, reasoning_effort="low", api_key=STUB_API_KEY
    )
    return config.model_copy(update={"api_base_url": base_url})


def session_waits(run_dir: Path) -> list[str]:
    """What the driver waited for, in order: the SessionWait records of the episode trace,
    written before each wait (providers/session_lifecycle.py _record_wait)."""
    lines = (run_dir / "episode_trace.jsonl").read_text(encoding="utf-8").splitlines()
    return [e["payload"]["wait"] for e in map(json.loads, lines) if e["event"] == "SessionWait"]


def _run(stub: RealtimeStub, steps, tmp_path: Path, scenario: str) -> list[dict]:
    """Run the steps in one episode; return the records a baseline holds."""
    config = realtime_config(stub.base_url)

    def points_at_stub(agent):
        # The SDK derives the WebSocket URL from base_url (openai/resources/realtime/
        # realtime.py:1191-1200); unset, it is api.openai.com (openai/_client.py:250-253).
        assert str(agent.driver.client.base_url) == stub.base_url + "/"

    start = stub.mark()
    episode = build_episode(
        config,
        tmp_path,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        model_key=MODEL_KEY,
        clock_mode="token",
        tokens_per_frame=TOKENS_PER_FRAME,
        max_ticks=MAX_TICKS,
        before_initialize=points_at_stub,
    )
    try:
        stub.wait_for_session_update()
        initialize_events, _ = stub.since(start)
        records = run_steps(episode, steps, stub)
    finally:
        episode.finish()
    connections = stub.finish()

    assert stub.violations == [], stub.violations
    assert stub.unscripted == 0, "the agent sent response.create beyond the script"
    assert stub.unused_replies == 0, "scripted replies were left unused"
    accepted = [c for c in connections if c["rejected"] is None]
    assert all(c["authorization"] == f"Bearer {STUB_API_KEY}" for c in accepted)
    assert accepted[-1]["close"] == {"received": 1000, "sent": 1000}  # agent.close()
    for record in records:
        result = record["result"]
        assert result.get("invalid_reason") is None, result.get("invalid_detail")
        assert result["inspection"]["valid"]
        # Each decision ends with response.create; its cap is the transport cap the runner
        # bound (openai_realtime.py:364-371; runner.py:890-915).
        response = record["client_events"][-1]["events"][-1]
        assert response["type"] == "response.create"
        assert response["response"] == {
            "max_output_tokens": record["budget"]["transport_output_tokens"],
            "reasoning": {"effort": "low"},
        }

    header = {
        "record": "header",
        "api": "openai_realtime",
        "scenario": scenario,
        "clock": {"mode": "token", "tokens_per_frame": TOKENS_PER_FRAME, "max_ticks": MAX_TICKS},
        "config": config.model_dump(),
    }
    return [
        header,
        {"record": "initialize", "client_events": initialize_events},
        *records,
        {"record": "connections", "connections": connections},
    ]


def _compare_with_baseline(name: str, records: list[dict], trajectory: list[dict]) -> None:
    actual = baseline.Normalizer()([*records, *trajectory])
    path = BASELINES / f"openai_realtime-{name}.jsonl"
    if os.environ.get("REPLAY_RECORD") == "1":
        baseline.write(path, actual)
        return
    problems = baseline.compare(baseline.read(path), actual)
    assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
        problems[:60]
    )


def _steps(records: list[dict]) -> list[dict]:
    return [r for r in records if r["record"] == "step"]


def _connections_used(record: dict) -> list[int]:
    return [segment["connection"] for segment in record["client_events"]]


def _split(events: list[dict]) -> tuple[list[dict], dict]:
    """A decision's events on one connection: any edit batch, then the observation item from
    send_observation (openai_realtime_session.py:245-250; openai_realtime.py:346-349), then
    response.create (351-371)."""
    *batch, observation, response = events
    assert response["type"] == "response.create"
    assert observation["type"] == "conversation.item.create"
    assert set(observation) == {"type", "item"} and observation["item"]["role"] == "user"
    return batch, observation


def _is_placeholder(item: dict) -> bool:
    """A folded turn's user item carries only the placeholder text
    (harnesses/default/request.py:155-161; openai_realtime.py:245-258)."""
    return item["content"] == [{"type": "input_text", "text": FOLDED_VISUAL_PLACEHOLDER}]


def _is_ledger(item: dict) -> bool:
    """The archived-action ledger item: a user item opened by the ledger header (context policy
    version 3; openai_realtime_session.py:205-221)."""
    if not item["content"][0]["text"].startswith(ARCHIVED_ACTION_LEDGER_HEADER + "\n"):
        return False
    assert item["role"] == "user"
    return True


def _applied_fold_pairs(stub: RealtimeStub) -> int:
    """Count in-place folds the stub applied, asserting each one put the placeholder where the
    deleted item was. A fold deletes the user item, then creates the placeholder after the item
    before it (openai_realtime_session.py:308-318); ids are not reused, so it is a new item."""
    pairs = 0
    for connection in stub.connections:
        events, positions = connection.events, connection.positions
        for i, (delete, create) in enumerate(zip(events, events[1:])):
            if (
                delete["type"] == "conversation.item.delete"
                and create["type"] == "conversation.item.create"
                and "event_id" in create
                and _is_placeholder(create["item"])
                and positions[i] is not None  # a batch the stub refused was never applied
            ):
                assert positions[i + 1] == positions[i]
                pairs += 1
    return pairs


def test_long_scenario(tmp_path):
    with RealtimeStub(replies(LONG_SCENARIO)) as stub:
        records = _run(stub, LONG_SCENARIO, tmp_path, "long")
        last_view = stub.views[-1]
        fold_pairs = _applied_fold_pairs(stub)
    steps = _steps(records)
    assert len(records[-1]["connections"]) == 1  # one handshake for the whole episode
    assert all(_connections_used(r) == [1] for r in steps)
    assert fold_pairs == sum(len(turns) for turns in FOLDS.values())

    # Follow every turn's items on the server through the edit batches the driver sends.
    user_items, assistant_items, prompts, ledger = [], [], [], None
    for k, record in enumerate(steps):
        (segment,) = record["client_events"]
        batch, observation = _split(segment["events"])
        expected = []
        if k == ARCHIVE:
            for j in ARCHIVED:
                expected += [("delete", user_items[j]), ("delete", assistant_items[j])]
            expected.append(("ledger", "root"))
        for j in FOLDS.get(k, ()):
            anchor = assistant_items[j - 1] if j else "root"
            expected += [("delete", user_items[j]), ("placeholder", anchor)]
        actual = []
        for event in batch:
            # in-place edits are waited for by id (openai_realtime.py:284-289, 549;
            # openai_realtime_session.py:263-270, 388-422)
            assert "event_id" in event
            if event["type"] == "conversation.item.delete":
                actual.append(("delete", event["item_id"]))
            elif _is_ledger(event["item"]):
                actual.append(("ledger", event["previous_item_id"]))
                ledger = event["item"]["content"][0]["text"]
            else:
                assert _is_placeholder(event["item"])
                actual.append(("placeholder", event["previous_item_id"]))
        assert actual == expected, f"decision {k}"
        placeholders = [
            event["item"]["id"]
            for event in batch
            if event["type"] == "conversation.item.create" and _is_placeholder(event["item"])
        ]
        for j, item_id in zip(FOLDS.get(k, ()), placeholders, strict=True):
            user_items[j] = item_id
        user_items.append(observation["item"]["id"])
        (response,) = record["server_responses"]
        assistant_items.append(response["output_item_ids"][0])
        prompts.append(observation["item"]["content"][0]["text"])

        sync = record["result"]["realtime_runtime"]["history_sync"]
        if k in FOLDS:
            assert sync == {
                "disposition": "exact",
                "history_state": "exact",
                "detail": "observation history updated in the existing Realtime session",
                "history_session_reconnected": False,
                "history_items_created": len(FOLDS[k]) + (k == ARCHIVE),
                "history_items_deleted": len(FOLDS[k]) + (k == ARCHIVE) * 2 * len(ARCHIVED),
                "history_visuals_folded_in_place": len(FOLDS[k]),
                "history_prompts_edited_in_place": len(FOLDS[k]),
            }
        else:
            assert sync["detail"] == "already synchronized"
        archived = record["result"]["realtime_runtime"]["history_archived_turns"]
        assert archived == (len(ARCHIVED) if k == ARCHIVE else 0)

    # The archive puts the ledger first, as a user item at root, one JSON line per archived
    # action (context.py:35-50; openai_realtime_session.py:288-297).
    lines = ledger.split("\n")
    assert lines[0] == ARCHIVED_ACTION_LEDGER_HEADER and len(lines) == 1 + len(ARCHIVED)

    # Execution feedback opens the next observation's prompt (harness.py:1071-1079, 1602-1622).
    assert prompts[0].startswith(TEXT_ACTION_FRAME_PROMPT)
    for prompt in prompts[1:]:
        feedback, frame_prompt = prompt.split("\n", 1)
        assert feedback.startswith("EXECUTION RESULT: ")
        assert frame_prompt.startswith(TEXT_ACTION_FRAME_PROMPT)
    # The rejected actions' feedback: two of the items carrying it are folded later, and the
    # batch above deleted them for a bare placeholder (harnesses/default/request.py:155-161;
    # entity/model.py:154-170); the last is kept.
    rejected = [k for k, step in enumerate(LONG_SCENARIO) if step.status == "rejected"]
    folded_at = {j: k for k, turns in FOLDS.items() for j in turns}
    assert [folded_at.get(k + 1) for k in rejected] == [20, 40, None]
    for k in rejected:
        assert prompts[k + 1].startswith("EXECUTION RESULT: actual=wait(); requested=")
        assert f"; status=rejected. Detail: {LONG_SCENARIO[k].detail}\n" in prompts[k + 1]

    # What the model sees: the conversation size at each response.create, and at the last
    # decision the ledger, turns 10-39 folded, turns 40-58 with their screenshots, then the newest
    # observation. Each turn's assistant item is the stub's own output item.
    sizes = [len(r["server_responses"][0]["context"]) for r in steps]
    assert [sizes[k] for k in (0, 19, 20, 49, 50, 59)] == [1, 39, 41, 99, 82, 100]
    assert last_view[0][1:] == ("user", ledger, None)
    for j, user, assistant in zip(range(10, 59), last_view[1:-1:2], last_view[2:-1:2]):
        assert assistant[1:] == ("assistant", LONG_SCENARIO[j].reply, None)
        assert user[1] == "user"
        assert (user[2] == FOLDED_VISUAL_PLACEHOLDER) == (j < 40) == (user[3] is None)
        if j >= 40:
            assert user[2] == prompts[j]
    assert last_view[-1][1:3] == ("user", prompts[-1]) and len(last_view) == 100

    # The runtime cap falls below the native limit at decision 30; the wire shows a number once
    # the cap is at most 4096 (openai_realtime.py:214-220).
    native = records[0]["config"]["native_output_limit"]
    runtime_caps = [r["budget"]["runtime_output_tokens"] for r in steps]
    assert runtime_caps[29] == native > runtime_caps[30]
    wire_caps = [r["budget"]["transport_output_tokens"] for r in steps]
    assert wire_caps[:57] == ["inf"] * 57 and wire_caps[57:] == [4096, 3072, 2048]

    _compare_with_baseline("token-long", records, baseline.trajectory(tmp_path))
    assert session_waits(tmp_path) == []  # edits in place; nothing to wait for


def test_dropped_response_reconnects(tmp_path):
    """A close with 1011 after response.created, before any output or usage, is a transient
    failure (openai_realtime.py:182-199). The harness retries the decision because nothing was
    received (harness.py:1284-1340). The runtime rebuilds the session on a new connection through
    connect(), replays canonical history (context.py:394-407; openai_realtime.py:301-344;
    openai_realtime_session.py:223-243) and sends the same observation again under a new item id."""
    steps = LONG_SCENARIO[:4]
    with RealtimeStub(replies(steps), drop_responses=(3,)) as stub:
        records = _run(stub, steps, tmp_path, "dropped-response")
    steps_records = _steps(records)
    first, _ = records[-1]["connections"]
    assert first["close"] == {"received": 1011, "sent": 1011}  # the client echoes the code
    assert [_connections_used(r) for r in steps_records] == [[1], [1], [1, 2], [2]]
    assert [r.get("closed") for r in steps_records[2]["server_responses"]] == [1011, None]

    dropped, rebuilt = steps_records[2]["client_events"]
    _, dropped_observation = _split(dropped["events"])
    session, *replay = rebuilt["events"]
    assert session["type"] == "session.update"
    replay, observation = _split(replay)
    assert observation["item"]["content"] == dropped_observation["item"]["content"]
    assert observation["item"]["id"] != dropped_observation["item"]["id"]
    # The replay is turns 0 and 1, each user item as first created, then its output.
    originals = [_split(r["client_events"][0]["events"])[1]["item"] for r in steps_records[:2]]
    assert all(set(event) == {"type", "item"} for event in replay)
    assert [e["item"]["role"] for e in replay] == ["user", "assistant"] * 2
    assert [e["item"]["content"] for e in replay[::2]] == [o["content"] for o in originals]
    assert [e["item"]["content"][0]["text"] for e in replay[1::2]] == [s.reply for s in steps[:2]]

    runtime = steps_records[2]["result"]["realtime_runtime"]
    assert runtime["transport_reconnect_attempts"] == 1
    assert runtime["transport_reconnect_reason"].startswith("transient: ConnectionClosedError(")
    assert runtime["transport_retry_usage"] == [None]
    _compare_with_baseline("token-dropped-response", records, baseline.trajectory(tmp_path))
    # The turn's retry backs off in the harness (RequestRetry), not in the session.
    assert session_waits(tmp_path) == []


def test_dropped_response_after_archive_replays_ledger(tmp_path):
    """The same drop at decision 51, after the first archive. The rebuild replays the ledger
    first, as a user item without an anchor (openai_realtime_session.py:229-231), then the
    retained turns 10-50, the folded ones as placeholders (233-236). Decisions 0-50 repeat the long
    scenario and are compared there; this baseline records decisions 51 and 52."""
    steps = LONG_SCENARIO[:53]
    with RealtimeStub(replies(steps), drop_responses=(52,)) as stub:
        records = _run(stub, steps, tmp_path, "dropped-response-after-archive")
    steps_records = _steps(records)
    first, _ = records[-1]["connections"]
    assert first["close"] == {"received": 1011, "sent": 1011}
    assert [_connections_used(r) for r in steps_records] == [[1]] * 51 + [[1, 2], [2]]

    (archive,) = steps_records[ARCHIVE]["client_events"]
    creates = [e["item"] for e in archive["events"] if e["type"] == "conversation.item.create"]
    (ledger,) = [item for item in creates if _is_ledger(item)]
    dropped, rebuilt = steps_records[51]["client_events"]
    session, replayed_ledger, *replay = rebuilt["events"]
    assert session["type"] == "session.update"
    assert set(replayed_ledger) == {"type", "item"} and _is_ledger(replayed_ledger["item"])
    assert replayed_ledger["item"]["content"] == ledger["content"]
    replay, observation = _split(replay)
    _, dropped_observation = _split(dropped["events"])
    assert observation["item"]["content"] == dropped_observation["item"]["content"]
    users = [e["item"] for e in replay if e["item"]["role"] == "user"]
    outputs = [e["item"]["content"][0]["text"] for e in replay if e["item"]["role"] == "assistant"]
    assert [_is_placeholder(item) for item in users] == [True] * 30 + [False] * 11
    # Turn 14's empty output is an item on the first connection but not in the replay
    # (harnesses/default/request.py:91-93; openai_realtime_session.py:195-198, 437), so the rebuilt
    # conversation is one item shorter.
    assert outputs == [step.reply for step in steps[10:51] if step.reply]
    assert [len(r["context"]) for r in steps_records[51]["server_responses"]] == [84, 83]
    _compare_with_baseline(
        "token-dropped-response-after-archive",
        [records[0], records[1], *steps_records[51:], records[-1]],
        baseline.trajectory(tmp_path)[51:],
    )


def test_rejected_upgrade_retries_connect(tmp_path):
    """connect()'s retry: a refused upgrade (HTTP 503, TRANSIENT through its response's status,
    providers/openai_realtime.py:257-263) fails one open_session attempt, and connect() tries a
    new connection after the shared backoff (providers/openai_realtime.py:435-454)."""
    steps = LONG_SCENARIO[:2]
    with RealtimeStub(replies(steps), reject_upgrades=1) as stub:
        records = _run(stub, steps, tmp_path, "rejected-upgrade")
    first, second = records[-1]["connections"]
    assert first["rejected"] == 503 and first["close"] is None
    assert second["rejected"] is None and second["url"] == first["url"]
    (initialize,) = records[1]["client_events"]
    assert initialize["connection"] == 2
    assert [event["type"] for event in initialize["events"]] == ["session.update"]
    assert all(_connections_used(r) == [2] for r in _steps(records))
    _compare_with_baseline("token-rejected-upgrade", records, baseline.trajectory(tmp_path))
    assert session_waits(tmp_path) == ["connect_backoff"]


def test_session_edit_error_rebuilds(tmp_path):
    """An error that echoes the first delete of the first in-place edit fails that edit
    (openai_realtime_session.py:366-367, 397-406). The driver falls back to a rebuild on a second
    connection (openai_realtime.py:517-532, 551-552) that replays the retained turns with the new
    folds (openai_realtime_session.py:229-242). Later edits anchor on the item ids that replay
    created (235-242, 319-321)."""
    steps = LONG_SCENARIO[:31]
    with RealtimeStub(replies(steps), edit_error_on_delete=1) as stub:
        records = _run(stub, steps, tmp_path, "edit-error")
        edit_errors = stub.edit_errors
        fold_pairs = _applied_fold_pairs(stub)
    assert edit_errors == 1
    assert fold_pairs == len(FOLDS[30])  # the refused batch on connection 1 never took effect
    steps_records = _steps(records)
    first, _ = records[-1]["connections"]
    # the rebuild closes it first (openai_realtime.py:291-299, 302)
    assert first["close"] == {"received": 1000, "sent": 1000}
    used = [_connections_used(r) for r in steps_records]
    assert used == [[1]] * 20 + [[1, 2]] + [[2]] * 10

    abandoned, rebuilt = steps_records[20]["client_events"]
    batch = abandoned["events"]  # sent whole before any ack is read (openai_realtime.py:547-549)
    assert [event["type"] for event in batch] == [
        "conversation.item.delete",
        "conversation.item.create",
    ] * len(FOLDS[20])
    sync = steps_records[20]["result"]["realtime_runtime"]["history_sync"]
    assert {k: v for k, v in sync.items() if k != "history_in_place_edit_error"} == {
        "disposition": "rebuilt",
        "history_state": "exact",
        "detail": "history_block_compaction",
        "history_session_reconnected": True,
        "history_in_place_edit_fallback": True,
    }
    error = sync["history_in_place_edit_error"]
    assert error.startswith('RuntimeError("Realtime history edit failed: RealtimeError(')
    assert f"event_id='{batch[0]['event_id']}'" in error

    session, *replay = rebuilt["events"]
    assert session["type"] == "session.update"
    replay, _ = _split(replay)
    users = [e["item"] for e in replay if e["item"]["role"] == "user"]
    outputs = [e["item"]["content"][0]["text"] for e in replay if e["item"]["role"] == "assistant"]
    assert [_is_placeholder(item) for item in users] == [True] * 10 + [False] * 10
    # A rebuild skips empty outputs (harnesses/default/request.py:91-93;
    # openai_realtime_session.py:195-198); decision 14 replied "".
    assert outputs == [step.reply for step in steps[:20] if step.reply]

    # Decision 30 folds turns 10-19 on connection 2. Each placeholder follows the previous turn's
    # output as the replay created it; turn 14 has none, so turn 15's placeholder follows the
    # placeholder this batch just created for turn 14 (openai_realtime_session.py:318-321).
    (segment,) = steps_records[30]["client_events"]
    fold, _ = _split(segment["events"])
    creates = [event for event in fold if event["type"] == "conversation.item.create"]
    origin = {event["item"]["id"]: event["item"]["role"] for event in replay}
    origin.update({event["item"]["id"]: "this batch" for event in creates})
    anchors = [origin.get(event["previous_item_id"]) for event in creates]
    assert anchors == ["assistant"] * 5 + ["this batch"] + ["assistant"] * 4
    assert creates[5]["previous_item_id"] == creates[4]["item"]["id"]
    _compare_with_baseline("token-edit-error", records, baseline.trajectory(tmp_path))
