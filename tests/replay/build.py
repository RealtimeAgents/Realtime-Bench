"""Set up an agent the way evaluation/runner.py main() does, without containers or a task server.

The clock settings are the runner's: ``clock_mode`` is ``--clock``, ``tokens_per_frame`` is the
frame budget B (None on the wall clock and in pause mode, runner.py:1401-1402), and ``max_ticks``
is the episode length run_stream receives (runner.py:1773). The finite token clock changes the
per-decision budget (runner.py:864-887) and the clock prompt (harnesses/default/prompts.py:177-201).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from realtime_framework.entity.config import DecisionBudget, ModelConfig
from realtime_framework.evaluation.agent_trace import AgentTraceSink, close_agent_trace
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.harnesses.default.prompts import budget_clock_prompt
from realtime_framework.harnesses.default.prompts import system_prompt_for_category
from realtime_framework.harnesses.registry import create_agent
from realtime_framework.providers.preflight import check_episode
from realtime_framework.sandbox.observation.som import ToolRuntime

from .frames import FrameSource
from .hermetic import HISTORY_ENV


@dataclass
class Episode:
    agent: object
    frames: FrameSource
    clock_mode: str
    tokens_per_frame: int | None
    max_ticks: int | None
    trace: EpisodeTrace
    agent_trace: AgentTraceSink

    def finish(self):
        """Close the agent, then its records, in the order of runner.main()'s finally block.

        Returns the invalid-run value closing the agent records produced (None when complete).
        """
        self.agent.close()
        invalid_run = close_agent_trace(self.agent_trace, None)
        self.trace.close()
        return invalid_run

    def budget(self, t_k: int) -> DecisionBudget:
        """The budget run_stream binds before the decision at clock position ``t_k``
        (runner.py:851, 861-913). Without a finite token clock nothing bounds the decision except
        the native limit, and an agent with unbounded output sends no provider cap."""
        agent = self.agent
        native_output_limit = getattr(getattr(agent, "config", None), "native_output_limit", None)
        remaining_clock_tokens = None
        if self.clock_mode == "token" and self.tokens_per_frame is not None:
            remaining_clock_tokens = max(
                0, math.floor((float(self.max_ticks) - float(t_k)) * self.tokens_per_frame + 1e-9)
            )
            # run_stream ends the episode here instead of deciding (runner.py:869-871).
            assert remaining_clock_tokens > 0, "the scenario outlasts the token clock"
        runtime_output_cap = (
            native_output_limit
            if remaining_clock_tokens is None
            else remaining_clock_tokens
            if native_output_limit is None
            else min(native_output_limit, remaining_clock_tokens)
        )
        if remaining_clock_tokens is not None:
            provider_output_token_cap = runtime_output_cap
        else:
            unbounded_output = bool(getattr(agent, "supports_unbounded_output", False))
            provider_output_token_cap = None if unbounded_output else runtime_output_cap
        transport_cap_mapper = getattr(agent, "transport_output_token_cap", None)
        transport_output_cap = (
            transport_cap_mapper(provider_output_token_cap)
            if callable(transport_cap_mapper)
            else provider_output_token_cap
        )
        if not (  # runner.py:894-905
            transport_output_cap is None
            or transport_output_cap == "inf"
            or (
                isinstance(transport_output_cap, int)
                and not isinstance(transport_output_cap, bool)
                and transport_output_cap > 0
            )
        ):
            raise ValueError(
                "adapter transport output cap must be None, a positive integer, or 'inf'"
            )
        return DecisionBudget(
            runtime_output_tokens=runtime_output_cap,
            provider_output_tokens=provider_output_token_cap,
            transport_output_tokens=transport_output_cap,
            clock_mode=self.clock_mode,
        )


def build_episode(
    config: ModelConfig,
    run_dir: Path,
    *,
    task,
    instruction: str,
    model_key: str,
    clock_mode: str = "wall",
    tokens_per_frame: int | None = None,
    max_ticks: int | None = None,
    before_initialize=None,
    scene: dict | None = None,
) -> Episode:
    """``model_key`` stands for the catalog key the runner records as the model identity.
    ``before_initialize(agent)`` runs just before initialize(), which may open a connection.
    ``scene`` holds FrameSource's ``screenshot`` and ``probe`` for a page other than the form."""
    fps = config.fps
    check_episode(config, clock_mode=clock_mode, task=task, env_fps=fps)  # runner.py:1565
    # runner.main() opens the episode trace, then hands the agent its write entry.
    trace = EpisodeTrace(run_dir / "episode_trace.jsonl", run_id=run_dir.name)
    agent_trace = AgentTraceSink(trace, run_dir / "episode_blobs.pack")
    agent = create_agent(config, log_dir=run_dir, trace=agent_trace)  # runner.py:1563
    # create_agent loads the repository .env (harnesses/registry.py:11 calls
    # providers/registry.py:15-39), which could set history settings hermetic.py cleared.
    injected = sorted(name for name in HISTORY_ENV if name in os.environ)
    if injected:
        raise RuntimeError(
            f"the repository .env sets {injected}; replay baselines use the defaults"
        )
    # runner.py:1564-1568
    agent.streaming_note = budget_clock_prompt(fps, tokens_per_frame, clock_mode)
    agent.system_prompt_override = system_prompt_for_category(task.category)  # runner.py:1570
    agent._clock_mode = clock_mode  # runner.py:1573; run_stream sets it again (runner.py:126)

    capabilities = agent.driver.capabilities  # runner.py:1601-1605
    text_observation = (
        capabilities.observation_modality == "text" or capabilities.action_channel == "choice"
    )
    frames = FrameSource(  # runner.py:1606-1613
        task, text_observation=text_observation, fps=fps, **(scene or {})
    )
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
    if before_initialize is not None:
        before_initialize(agent)
    agent.initialize(instruction, initial_observation)  # runner.py:1644
    agent.discard_pending_observations()  # runner.py:1647
    return Episode(
        agent=agent,
        frames=frames,
        clock_mode=clock_mode,
        tokens_per_frame=tokens_per_frame,
        max_ticks=max_ticks,
        trace=trace,
        agent_trace=agent_trace,
    )
