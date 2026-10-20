"""Build the agent for a model config and apply the run's settings."""

from realtime_framework.entity.records import UNTRACED
from realtime_framework.providers.registry import load_project_secrets


def create_agent(config, log_dir=None, *, trace=UNTRACED, clock=None):
    """``trace`` is the agent's write entry into the episode trace: the runner passes its
    AgentTraceSink; an agent built without one (a test or a tool) records nothing. ``clock`` is
    the episode's operational clock (realtime_framework/clock.py), real time when omitted.

    The configuration rules run before this, in providers/preflight.py check_episode, and
    configure_agent must run before initialize(), which selects the prompts by task category.
    """
    load_project_secrets()
    from realtime_framework.harnesses.default.harness import DefaultHarness

    return DefaultHarness(config, log_dir=log_dir, trace=trace, clock=clock)


def configure_agent(agent, *, fps, tokens_per_frame, clock_mode, category):
    """Apply the run's clock note, task category, scenario prompt and clock mode before
    initialize(). ``fps`` is the model-input frame rate, the unit of ``tokens_per_frame``."""
    from realtime_framework.harnesses.default.prompts import budget_clock_prompt
    from realtime_framework.harnesses.default.prompts import system_prompt_for_category

    agent.streaming_note = budget_clock_prompt(
        fps,
        tokens_per_frame,
        clock_mode,
    )
    # Role/rules vary by scenario family; the task-specific objective is injected separately.
    agent.system_prompt_override = system_prompt_for_category(
        category, finite_budget=clock_mode == "token" and tokens_per_frame is not None
    )
    # The context policy's prompts (item 11) are keyed by it.
    agent.task_category = category
    # Copied into every GenerationRequest.clock_mode; no driver reads it any more (the wall-clock
    # rules are pre-run checks, providers/preflight.py).
    agent._clock_mode = clock_mode
