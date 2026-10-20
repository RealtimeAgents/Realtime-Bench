"""Set up an agent with the runner's set_up_agent, without containers or a task server.

runner.main() and build_episode both call set_up_agent (evaluation/runner.py); in place of a
BrowserSceneCapture on the container's browser, build_episode hands it the synthetic frames.
The clock settings are the runner's: ``clock_mode`` is ``--clock``, ``tokens_per_frame`` is the
frame budget B (None on the wall clock and in pause mode, runner.py:1407-1408), and ``max_ticks``
is the episode length run_stream receives (runner.py:1714). The finite token clock changes the
per-decision budget (runner.py:866-889) and the clock prompt (harnesses/default/prompts.py:177-201).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from realtime_framework.entity.config import DecisionBudget, ModelConfig
from realtime_framework.evaluation import runner
from realtime_framework.evaluation.agent_trace import AgentTraceSink, close_agent_trace
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.harnesses.registry import create_agent

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
        (runner.py:853, 863-915). Without a finite token clock nothing bounds the decision except
        the native limit, and an agent with unbounded output sends no provider cap. Step-driven
        episodes observe one frame per tick, so run_stream's frame_ticks is 1 here."""
        agent = self.agent
        native_output_limit = getattr(getattr(agent, "config", None), "native_output_limit", None)
        remaining_clock_tokens = None
        if self.clock_mode == "token" and self.tokens_per_frame is not None:
            remaining_clock_tokens = max(
                0, math.floor((float(self.max_ticks) - float(t_k)) * self.tokens_per_frame + 1e-9)
            )
            # run_stream ends the episode here instead of deciding (runner.py:871-873).
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
        if not (  # runner.py:896-907
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
    ``before_initialize(agent)`` runs on the new agent, before initialize() may open a
    connection. ``scene`` holds FrameSource's ``screenshot`` and ``probe`` for a page other
    than the form."""
    fps = config.fps  # main() passes --fps as both rates (runner.py:1421)

    def checked_create_agent(*args, **kwargs):
        agent = create_agent(*args, **kwargs)
        # create_agent loads the repository .env (harnesses/registry.py:10 calls
        # providers/registry.py:15-39), which could set history settings hermetic.py cleared.
        injected = sorted(name for name in HISTORY_ENV if name in os.environ)
        if injected:
            raise RuntimeError(
                f"the repository .env sets {injected}; replay baselines use the defaults"
            )
        if before_initialize is not None:
            before_initialize(agent)
        return agent

    # runner.main() opens the episode trace before it sets up the agent.
    trace = EpisodeTrace(run_dir / "episode_trace.jsonl", run_id=run_dir.name)
    setup = runner.AgentSetup()
    with mock.patch.object(runner, "create_agent", checked_create_agent):
        runner.set_up_agent(
            setup,
            config,
            task,
            instruction,
            clock_mode=clock_mode,
            env_fps=fps,
            tokens_per_frame=tokens_per_frame,
            run_dir=run_dir,
            episode_trace=trace,
            model=model_key,
            open_capture=lambda text_observation: FrameSource(
                task, text_observation=text_observation, fps=fps, **(scene or {})
            ),
        )
    return Episode(
        agent=setup.agent,
        frames=setup.scene_capture,
        clock_mode=clock_mode,
        tokens_per_frame=tokens_per_frame,
        max_ticks=max_ticks,
        trace=trace,
        agent_trace=setup.agent_trace,
    )
