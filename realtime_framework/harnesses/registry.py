"""Select the model decision strategy independently of provider transport."""

from realtime_framework.entity.records import UNTRACED
from realtime_framework.providers.registry import (
    adapter_class,
    load_project_secrets,
)


def create_agent(config, log_dir=None, *, trace=UNTRACED):
    """``trace`` is the agent's write entry into the episode trace: the runner passes its
    AgentTraceSink; an agent built without one (a test or a tool) records nothing."""
    load_project_secrets()
    if adapter_class(config.adapter_type).capabilities.action_channel == "choice":
        from realtime_framework.harnesses.jev.harness import JevHarness

        return JevHarness(config, log_dir=log_dir, trace=trace)
    from realtime_framework.harnesses.default.harness import DefaultHarness

    return DefaultHarness(config, log_dir=log_dir, trace=trace)


def configure_agent(agent, *, fps, tokens_per_frame, clock_mode, category):
    """Apply the run's clock note, task category, scenario prompt and clock mode before
    initialize()."""
    from realtime_framework.harnesses.default.prompts import budget_clock_prompt
    from realtime_framework.harnesses.default.prompts import system_prompt_for_category

    agent.streaming_note = budget_clock_prompt(
        fps,
        tokens_per_frame,
        clock_mode,
    )
    # Role/rules vary by scenario family; the task-specific objective is injected separately.
    agent.system_prompt_override = system_prompt_for_category(category)
    # The context policy's prompts (item 11) are keyed by it.
    agent.task_category = category
    # Adapters use this before run_stream() starts to decide whether terminal provider usage is
    # allowed to delay a committed wall-clock action.
    agent._clock_mode = clock_mode
