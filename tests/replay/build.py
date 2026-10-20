"""Set up an agent the way evaluation/runner.py main() does, without containers or a task server.

Only the wall clock is mirrored: chat_completions accepts no other (providers/registry.py:141-144,
411-416). The finite token clock adds a per-decision budget (runner.py:864-887) and a different
clock prompt (harnesses/default/prompts.py:177-201).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from realtime_framework.entity.config import DecisionBudget, ModelConfig
from realtime_framework.harnesses.default.prompts import budget_clock_prompt
from realtime_framework.harnesses.default.prompts import system_prompt_for_category
from realtime_framework.harnesses.registry import create_agent
from realtime_framework.providers.registry import validate_model_interaction
from realtime_framework.sandbox.observation.som import ToolRuntime

from .frames import FrameSource
from .hermetic import HISTORY_ENV

CLOCK_MODE = "wall"


@dataclass
class Episode:
    agent: object
    frames: FrameSource
    budget: DecisionBudget


def build_episode(
    config: ModelConfig, run_dir: Path, *, task, instruction: str, model_key: str
) -> Episode:
    """``model_key`` stands for the catalog key the runner records as the model identity."""
    fps = config.fps
    # Pre-run API check (runner.py:1554-1559); tokens_per_frame is None on the wall clock
    # (runner.py:1401-1402).
    validate_model_interaction(
        config.adapter_type, clock_mode=CLOCK_MODE, fps=fps, tokens_per_frame=None
    )
    agent = create_agent(config, log_dir=run_dir)  # runner.py:1560
    # create_agent loads the repository .env (harnesses/registry.py:11 calls
    # providers/registry.py:15-39), which could set history settings hermetic.py cleared.
    injected = sorted(name for name in HISTORY_ENV if name in os.environ)
    if injected:
        raise RuntimeError(
            f"the repository .env sets {injected}; replay baselines use the defaults"
        )
    agent.streaming_note = budget_clock_prompt(fps, None, CLOCK_MODE)  # runner.py:1564-1568
    agent.system_prompt_override = system_prompt_for_category(task.category)  # runner.py:1570
    agent._clock_mode = CLOCK_MODE  # runner.py:1573; run_stream sets it again (runner.py:126)

    capabilities = agent.driver.capabilities  # runner.py:1601-1605
    text_observation = (
        capabilities.observation_modality == "text" or capabilities.action_channel == "choice"
    )
    frames = FrameSource(task, text_observation=text_observation, fps=fps)  # runner.py:1606-1613
    tool_runtime = ToolRuntime(  # runner.py:1614-1620
        run_dir / "tool_calls.jsonl",
        task_id=task.task_id,
        adapter=config.adapter_type,
        model=model_key,
        text_observation=text_observation,
    )
    agent.bind_tool_runtime(tool_runtime)  # runner.py:1621-1622

    initial_observation = frames.capture(0)  # runner.py:1630
    interactables = initial_observation.metadata.get("interactables") or {}
    # The runner refuses to start without a bound, stable first frame (runner.py:1631-1639).
    assert interactables.get("scene", {}).get("status") == "ok"
    assert interactables.get("binding", {}).get("stable")
    agent.initialize(instruction, initial_observation)  # runner.py:1644
    agent.discard_pending_observations()  # runner.py:1647

    # The per-decision budget (runner.py:861-913). Without a token clock nothing bounds the
    # decision except the native limit; an agent with unbounded output sends no provider cap.
    native_output_limit = agent.config.native_output_limit
    provider_cap = None if agent.supports_unbounded_output else native_output_limit
    budget = DecisionBudget(
        runtime_output_tokens=native_output_limit,
        provider_output_tokens=provider_cap,
        transport_output_tokens=agent.transport_output_token_cap(provider_cap),
        clock_mode=CLOCK_MODE,
    )
    return Episode(agent=agent, frames=frames, budget=budget)
