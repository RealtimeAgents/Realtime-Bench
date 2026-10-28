"""A model's click past the screenshot is refused as the environment refuses any action
(sandbox/actions.py GuestEnvironment._reject_outside_screenshot): the decision counts, the world
moves on, nothing reaches the guest, and the model's next request says why. A click on the
screenshot's last pixel runs. The episode runs the openrouter agent in pause mode against the real
GuestEnvironment on a task clock (token_env.py), as test_termination.py does."""

from __future__ import annotations

import json

import pytest

import realtime_framework.providers.registry as catalog

from .build import build_episode
from .clock_task import ClockTask
from .loop import commit_statuses, run_loop
from .scenario import CLICK_SUBMIT, TASK_INSTRUCTION
from .stub_openrouter import OpenRouterStub
from .test_openrouter import MODEL_KEY, ROUTE, openrouter_env  # noqa: F401 (fixture)
from .test_runner_frame_rates import reply
from .token_env import TokenEnvironment

# 3 s at 1 fps: in pause mode decisions observe ticks 0, 1 and 2.
MAX_MINUTES = 0.05
OUTSIDE = '{"action": "click", "x": 768, "y": 100}'
REFUSAL = "click point (768, 100) lands outside the screenshot, which is 768x432 pixels"


def _request_texts(received) -> str:
    texts = []
    for message in json.loads(received.body)["messages"]:
        content = message["content"]
        if isinstance(content, str):
            texts.append(content)
        else:
            texts += [item["text"] for item in content if item["type"] == "text"]
    return "\n".join(texts)


@pytest.mark.usefixtures("openrouter_env")
def test_a_click_past_the_screenshot_is_refused_and_the_model_told(monkeypatch, tmp_path):
    entry = catalog.get(MODEL_KEY)
    config = catalog.to_model_config(
        MODEL_KEY, fps=1.0, reasoning_effort=entry.defaults["reasoning_effort"]
    )
    task = ClockTask(1.0)
    replies = [
        reply(OUTSIDE, 80),
        reply('{"action": "click", "x": 767, "y": 431}', 80),
        reply(CLICK_SUBMIT, 80),
    ]
    with OpenRouterStub(replies, route=ROUTE) as stub:
        monkeypatch.setenv("OPENROUTER_BASE_URL", stub.base_url)
        episode = build_episode(
            config,
            tmp_path,
            task=task,
            instruction=TASK_INSTRUCTION,
            model_key=MODEL_KEY,
            clock_mode="token",
            tokens_per_frame=None,
        )
        env = TokenEnvironment(episode.frames, task, max_minutes=MAX_MINUTES)
        result = run_loop(episode, env)
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert result.invalid_run is None

    assert commit_statuses(env) == [
        (0, "rejected", REFUSAL),
        (1, "applied", ""),
        (2, "applied", ""),
    ]
    assert [call["tick"] for call in env.calls if call["call"] == "env.capture"][-3:] == [0, 1, 2]
    sent = [
        (call["decision"], call["action"]["parameters"])
        for call in env.calls
        if call["call"] == "guest.execute_action"
    ]
    assert sent == [(1, {"x": 767, "y": 431}), (2, {"x": 320, "y": 192})]

    refused = episode.agent.history.actions[0]
    assert (refused["requested"], refused["executed"], refused["result"]) == (
        "click {'x': 768, 'y': 100}",
        "wait()",
        "rejected",
    )
    assert (
        "EXECUTION RESULT: actual=wait(); requested=click {'x': 768, 'y': 100}; "
        f"status=rejected. Detail: {REFUSAL}"
    ) in _request_texts(stub.requests[1])
