"""Set up an agent with the runner's set_up_agent, without containers or a task server.

runner.main() and build_episode both call set_up_agent (evaluation/runner.py); in place of a
BrowserSceneCapture on the container's browser, build_episode hands it the synthetic frames.
The clock settings are the runner's: ``clock_mode`` is ``--clock``, ``tokens_per_frame`` is the
frame budget B (None on the wall clock and in pause mode, runner.py main), and ``max_ticks`` is
the episode length in ticks, the duration_s run_stream receives times its env_fps (runner.py main,
kernel/turns.py TurnConfig.max_ticks). The finite token clock changes the per-decision budget
(kernel/turns.py _captured, episode_records.py decision_budget) and the clock prompt
(harnesses/default/prompts.py budget_clock_prompt).
"""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from realtime_framework.clock import SystemClock
from realtime_framework.entity.config import DecisionBudget, ModelConfig
from realtime_framework.evaluation import runner
from realtime_framework.evaluation.agent_trace import AgentTraceSink, close_agent_trace
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.harnesses.registry import create_agent
from realtime_framework.sandbox.observation import som

from . import baseline
from .frames import FrameSource
from .hermetic import HISTORY_ENV


def wait_for_abort(agent, timeout: float) -> None:
    """Return once the cleanup of the agent's last abort (providers/session_lifecycle.py abort)
    is done. The abort was submitted first, and the execution thread starts what it is given in
    order."""
    session = agent._runtime.session

    async def cleanup_done():
        await asyncio.wait({session._cleanup})

    agent._runtime.execution.run(cleanup_done(), timeout=timeout)


class ToolRuntime(som.ToolRuntime):
    """The runner's tool runtime. The inspection proof keeps only the SHA-256 of the SoM-marked
    image (sandbox/observation/som.py:448, 725), so each one is registered for the baselines."""

    def inspect(self, observation, **kwargs) -> dict:
        execution = super().inspect(observation, **kwargs)
        if execution["image_bytes"]:
            baseline.image_digest(execution["image_bytes"])
        return execution


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
        (kernel/turns.py _captured, shell.py _start_generation, episode_records.py
        decision_budget). Without a finite token clock nothing bounds the decision except the
        native limit, and an agent with unbounded output sends no provider cap. Step-driven
        episodes observe one frame per tick, so run_stream's env_fps and input_fps are equal
        here."""
        agent = self.agent
        native_output_limit = getattr(getattr(agent, "config", None), "native_output_limit", None)
        remaining_clock_tokens = None
        if self.clock_mode == "token" and self.tokens_per_frame is not None:
            remaining_clock_tokens = max(
                0, math.floor((float(self.max_ticks) - float(t_k)) * self.tokens_per_frame + 1e-9)
            )
            # run_stream ends the episode here instead of deciding (kernel/turns.py _captured).
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
        if not (  # episode_records.py decision_budget
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
    clock=None,
    env_fps: float | None = None,
) -> Episode:
    """``model_key`` stands for the catalog key the runner records as the model identity.
    ``before_initialize(agent)`` runs on the new agent, before initialize() may open a
    connection. ``scene`` holds FrameSource's ``screenshot`` and ``probe`` for a page other
    than the form. ``clock`` is the agent's clock (real time when omitted). ``env_fps`` is the
    environment frame rate main() passes set_up_agent, when it is not ``config.fps``."""

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

    clock = clock or SystemClock()
    # runner.main() opens the episode trace before it sets up the agent. Its attempt id leads the
    # retry jitter keys (entity/model.py RequestRetry); a fixed one keeps the baselines' backoffs.
    trace = EpisodeTrace(
        run_dir / "episode_trace.jsonl", run_id=run_dir.name, attempt_id="replay", clock=clock
    )
    setup = runner.AgentSetup()
    with (
        mock.patch.object(runner, "create_agent", checked_create_agent),
        mock.patch.object(runner, "ToolRuntime", ToolRuntime),
    ):
        runner.set_up_agent(
            setup,
            config,
            task,
            instruction,
            clock_mode=clock_mode,
            interaction_mode="turn_based",
            tokens_per_frame=tokens_per_frame,
            run_dir=run_dir,
            episode_trace=trace,
            model=model_key,
            open_capture=lambda text_observation: FrameSource(
                task, text_observation=text_observation, fps=config.fps, **(scene or {})
            ),
            clock=clock,
            env_fps=config.fps if env_fps is None else env_fps,
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
