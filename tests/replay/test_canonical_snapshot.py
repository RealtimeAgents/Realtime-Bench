"""Canonical-request snapshots: the request each decision hands its encoder, recorded as the
baseline for the encoder changes of phase 1 steps 4 to 6.

Requests are captured where the runtime calls the encoder, after its context-pressure loop
(harnesses/default/context.py, RealtimeRuntime._run_turn). Record with
``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay/test_canonical_snapshot.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import httpx
from PIL import Image

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.actions import FrameGeometry, Observation
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.harnesses.registry import configure_agent, create_agent
from realtime_framework.sandbox.observation.interactables import canonical_result

from . import baseline
from .build import ToolRuntime, build_episode
from .frames import ReplayTask
from .scenario import LONG_SCENARIO, TASK_INSTRUCTION, replies
from .steps import commit, decide, run_steps
from .stub import ChatCompletionsStub
from .test_chat_completions import MODEL_KEY, chat_completions_config

BASELINES = Path(__file__).parent / "baselines"
WALL_TIME = "<wall time>"


def capture_requests(agent) -> list:
    """Record every canonical request the runtime passes to the encoder."""
    driver = agent._runtime.driver
    start_generation = driver.start_generation
    captured = []

    async def recording(request):
        captured.append(request.canonical)
        return await start_generation(request)

    driver.start_generation = recording
    return captured


def snapshot(requests) -> list[dict]:
    """Each distinct block once, then per request the ordered (block, intent) list. A block is
    keyed by its normalized form, which holds images by their pixel digests."""
    normalize = baseline.Normalizer()
    records, stored = [], set()
    for index, request in enumerate(requests):
        record = request.to_record()
        items = []
        for item in record.pop("items"):
            block = item["block"]
            if block["time"]["wall_ns"] is not None:  # the agent's wall clock
                block["time"]["wall_ns"] = WALL_TIME
            block = normalize(block)
            text = json.dumps(block, sort_keys=True, ensure_ascii=False)
            key = hashlib.sha256(text.encode()).hexdigest()[:16]
            if key not in stored:
                stored.add(key)
                records.append({"record": "block", "key": key, "block": block})
            items.append({"block": key, "intent": item["intent"]})
        records.append(normalize({"record": "request", "index": index, **record, "items": items}))
    return records


def check(name: str, records: list[dict]) -> None:
    path = BASELINES / f"canonical-{name}.jsonl"
    if os.environ.get("REPLAY_RECORD") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = (json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    expected = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    problems = baseline.compare(expected, records)
    assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
        problems[:60]
    )


def test_chat_completions_long(tmp_path):
    with ChatCompletionsStub(replies(LONG_SCENARIO)) as stub:
        config = chat_completions_config(stub.base_url, stream=True)
        episode = build_episode(
            config, tmp_path, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key=MODEL_KEY
        )
        requests = capture_requests(episode.agent)
        try:
            run_steps(episode, LONG_SCENARIO, stub)
        finally:
            episode.agent.close()
    assert len(requests) == len(LONG_SCENARIO)
    kinds = [{item.block.kind for item in request.items} for request in requests]
    assert "action_ledger" in kinds[50] and "action_ledger" not in kinds[49]
    check("chat_completions-long", snapshot(requests))


# A focused text field, so that the Decision state carries focused_input and the composition
# inventory (decisions.md section 5 has the fixture this follows).
PNG = io.BytesIO()
Image.new("RGB", (1920, 1080), (245, 245, 245)).save(PNG, format="PNG")
PNG = PNG.getvalue()
LUNA_STEPS = [
    ("", ["compose_text", "char_000048", "char_000069", "finish"], "applied"),
    ("Hi", ["click_P2"], "applied"),
    ("Hi", ["wait"], "wait"),
]


def luna_frame(seq: int, value: str) -> Observation:
    observation_id = f"replay-luna:{seq:06d}"
    sha = hashlib.sha256(PNG).hexdigest()
    geometry = FrameGeometry((1920, 1080), (1920, 1080))
    window = {
        "window_id": "page-1",
        "title": "Stub form",
        "active": True,
        "ax_tree": 'RootWebArea "Stub form"\n  textbox "Name" focused\n  button "Submit"',
        "actionable_now": [
            {
                "id": "name",
                "role": "textbox",
                "label": "Name",
                "input_type": "text",
                "bbox": [700, 400, 1220, 460],
                "actions": ["click", "type"],
                "dom_order": 1,
                "focused": True,
                "value": value,
                "text_selected": False,
            },
            {
                "id": "submit",
                "role": "button",
                "label": "Submit",
                "bbox": [860, 520, 1060, 580],
                "actions": ["click"],
                "dom_order": 2,
            },
        ],
    }
    raw = {
        "scene": {"status": "ok", "kind": "browser_dom", "target_id": "page-1"},
        "binding": {"stable": True, "fingerprint": f"fp-{seq}"},
        "windows": [window],
        "keyboard_actions": [],
    }
    interactables = canonical_result(
        raw,
        observation_id=observation_id,
        screenshot_sha256=sha,
        captured_at=1000.0 + seq,
        task_id="replay-luna",
        frame_geometry=geometry,
    )
    return Observation(
        PNG,
        1000.0 + seq,
        float(seq - 1),
        {
            "observation_id": observation_id,
            "screenshot_sha256": sha,
            "frame_geometry": geometry.to_dict(),
            "interactables": interactables,
        },
    )


def test_luna_choice_parts(tmp_path):
    script = [choice for _, choices, _ in LUNA_STEPS for choice in choices]

    def answer(request):
        body = json.loads(request.content)
        criteria = body["questions"]["next_action"]["criteria"]
        choice = script.pop(0)
        assert choice in criteria
        return httpx.Response(
            200,
            json={
                "provider": "openai",
                "model": body["model"],
                "usage": {"input_tokens": 1000, "output_tokens": 1},
                "answers": {
                    "next_action": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": 1.0,
                        "probabilities": {key: float(key == choice) for key in criteria},
                    }
                },
            },
        )

    config = catalog.to_model_config(
        "gpt-6-luna-decisions", fps=1.0, reasoning_effort="fixed"
    ).model_copy(update={"api_key": "stub-key", "api_base_url": "http://127.0.0.1:9/api/alpha"})
    agent = create_agent(config, log_dir=tmp_path)
    client = agent.driver
    client.client = httpx.AsyncClient(
        base_url=client.client.base_url,
        headers=client.client.headers,
        timeout=30.0,
        transport=httpx.MockTransport(answer),
    )
    configure_agent(
        agent,
        fps=1.0,
        tokens_per_frame=None,
        clock_mode="wall",
        category="realtime",
        interaction_mode="turn_based",
    )
    agent.bind_tool_runtime(
        ToolRuntime(
            tmp_path / "tool_calls.jsonl",
            task_id="replay-luna",
            adapter=config.adapter_type,
            model="gpt-6-luna-decisions",
            text_observation=True,
        )
    )
    agent.initialize(
        "Type the greeting Hi into the Name field, then press Submit.", luna_frame(1, "")
    )
    agent.discard_pending_observations()
    requests = capture_requests(agent)
    used = []
    driver_start = agent.driver.start_generation

    async def remember_choice(request):
        (generation,) = agent._generations
        used.append(generation.choice)
        return await driver_start(request)

    agent.driver.start_generation = remember_choice
    try:
        for k, (value, _, status) in enumerate(LUNA_STEPS):
            budget = DecisionBudget(None, None, None, "wall")
            result = decide(agent, k, luna_frame(k + 2, value), budget)
            assert not result.get("invalid_reason"), result.get("invalid_detail")
            commit(agent, k, status)
    finally:
        agent.close()
    assert not script
    # Each request carries the choice parts the agent prepared for its own observation.
    for request, (choice, selection) in zip(requests, used, strict=True):
        assert (request.answer.choice, request.selection) == (choice, selection)
    assert requests[0].selection.text_composition and requests[0].selection.focused_input
    check("openai_decisions-compose", snapshot(requests))
