"""Pre-run checks: every rule that rejects a run configuration before an episode starts.

Each rule reads only the registered driver class's declarations (capabilities, deployment,
compat flags, option tables), the configuration and, for check_context_policy, the context
policy the run resolves. Every entry point keeps the rules it applied before the checks were
collected here:

- runner argparse (evaluation/runner.py) and identity_main (evaluation/config.py):
  check_token_budget;
- run plans, effective_run_config and the Harbor trial worker: validate_model_interaction
  (check_frame_rates, check_wall_clock, check_interaction_mode);
- Harbor's agent options (evaluation/harbor.py RealtimeBenchOptions): check_frame_rates;
- the runner, once, before it builds the agent: check_episode.

Construction-time failures (credentials, SDK pin, endpoint URL, Decisions route) stay in the
drivers, so they now come after every check_episode rule. When a run has two faults, the first
one reported can therefore differ from before: the choice-task, frame-rate and Gemini session
checks used to fire in initialize or at the session connect, after the constructor had looked
for the API key, and are now reported ahead of a missing key.
"""

from __future__ import annotations

from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.providers.gemini.live import (
    LIVE_OPTIONS,
    LIVE_REASONING_EFFORTS,
    GeminiLiveDriver,
    live_options,
)
from realtime_framework.providers.litellm import LiteLLMDriver
from realtime_framework.providers.registry import adapter_class

_GEMINI_EXTENDED_THINKING = "gemini-3.8-live-extended-thinking"
_GEMINI_FLASH_LIVE = "gemini-3.1-flash-live-preview"


class UnsupportedTokenBudget(ValueError):
    """The API cannot report complete usage for a finite token clock."""


def check_token_budget(adapter_type, *, clock_mode, tokens_per_frame) -> None:
    """A finite token clock charges provider-reported usage, so the API must report it all.

    Raises UnsupportedTokenBudget, which the argparse entry points report as a usage error; an
    unknown adapter_type still raises the lookup's ValueError.
    """
    if clock_mode != "token" or tokens_per_frame is None:
        return
    if not adapter_class(adapter_type).capabilities.finite_frame_budget:
        raise UnsupportedTokenBudget(
            f"adapter {adapter_type!r} cannot report complete usage for a finite token clock"
        )


def check_wall_clock(driver, adapter_type, clock_mode) -> None:
    if driver.capabilities.wall_clock_only and clock_mode != "wall":
        raise ValueError(f"The {adapter_type} API requires a wall clock")


def check_interaction_mode(
    driver, adapter_type, *, interaction_mode, clock_mode, model_input_fps
) -> None:
    """Native streaming pushes frames into a live session at the model-input rate while the model
    generates, so it needs a live API that declares it, the wall clock (neither the token nor the
    pause clock) and a model-input rate within the model's video cap."""
    if interaction_mode == "turn_based":
        return
    capabilities = driver.capabilities
    if not capabilities.persistent_session:
        raise ValueError(
            f"Streaming requires a live API (openai_realtime or gemini_live); {adapter_type} "
            "is not one"
        )
    if interaction_mode not in capabilities.interaction_modes:
        raise ValueError(f"The {adapter_type} API does not support streaming")
    if clock_mode != "wall":
        raise ValueError("Streaming requires a wall clock")
    check_fps(driver, model_input_fps)


def check_frame_rates(fps, model_input_fps) -> None:
    """The loop observes at most one frame per environment tick, so a model-input rate above the
    environment rate (``fps``) would be stated to the model but never fed to it."""
    if float(model_input_fps) > float(fps):
        raise ValueError(
            f"The model-input frame rate ({model_input_fps:g} fps) exceeds the environment frame "
            f"rate ({fps:g} fps)"
        )


def validate_model_interaction(
    adapter_type,
    *,
    clock_mode,
    fps,
    model_input_fps,
    interaction_mode="turn_based",
    tokens_per_frame=None,
):
    """Validate API interaction requirements before starting an episode. ``fps`` is the
    environment frame rate."""
    del tokens_per_frame
    check_frame_rates(fps, model_input_fps)
    driver = adapter_class(adapter_type)
    check_wall_clock(driver, adapter_type, clock_mode)
    check_interaction_mode(
        driver,
        adapter_type,
        interaction_mode=interaction_mode,
        clock_mode=clock_mode,
        model_input_fps=model_input_fps,
    )


def check_config(driver, config) -> None:
    """Static model-config rules, in the order the driver's constructor applied them."""
    if issubclass(driver, LiteLLMDriver):
        compat = driver.compat
        if compat.output_limit_required and config.native_output_limit is None:
            raise ValueError(
                f"The {config.adapter_type} API requires {compat.output_limit_param}; "
                "declare endpoint.native_output_limit in the model YAML"
            )
        if (
            driver.deployment.gateway == "openai_compatible"
            and config.context_window_tokens is not None
            and config.context_reserve_tokens >= config.context_window_tokens
        ):
            raise ValueError("Context capacity must exceed generation reserve")
    elif issubclass(driver, GeminiLiveDriver):
        _check_live_options(config)
        _check_live_model_contract(config)


def _check_live_options(config) -> None:
    options = config.provider_options
    unknown = set(options) - set(LIVE_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown Gemini Live provider_options: {sorted(unknown)}")
    for name, (default, allowed) in LIVE_OPTIONS.items():
        value = options.get(name, default)
        if value not in allowed:
            raise ValueError(f"Gemini Live {name} must be one of {list(allowed)}, got {value!r}")


def _check_live_model_contract(config) -> None:
    """Reject setup combinations that violate the selected Live model's API contract."""
    options = live_options(config)
    if (
        options["tool_response_scheduling"] == "silent"
        and options["tool_behavior"] != "non_blocking"
    ):
        raise ValueError("Silent Gemini tool responses require non-blocking tools")
    model_name = str(config.model_name).rsplit("/", 1)[-1]
    if model_name == _GEMINI_FLASH_LIVE and options["tool_behavior"] != "blocking":
        raise ValueError(
            f"{_GEMINI_FLASH_LIVE} calls functions synchronously only; use tool_behavior=blocking"
        )
    if model_name == "gemini-3.8-live" and config.reasoning_effort not in {None, "fixed"}:
        raise ValueError(
            "gemini-3.8-live requires omitting thinking_config; use reasoning_effort=fixed"
        )
    if model_name != _GEMINI_EXTENDED_THINKING:
        return
    invalid = []
    if config.reasoning_effort not in {"low", "medium", "high"}:
        invalid.append("reasoning_effort must be low, medium, or high")
    if options["tool_behavior"] != "non_blocking":
        invalid.append("tool_behavior must be non_blocking")
    if options["completion_signal"] != "interaction_idle":
        invalid.append("completion_signal must be interaction_idle")
    if options["tool_response_scheduling"] == "silent":
        invalid.append("function scheduling is unsupported; use tool_response_scheduling=omit")
    if invalid:
        raise ValueError(f"{_GEMINI_EXTENDED_THINKING} requires " + "; ".join(invalid))


def check_choice_task(driver, task) -> None:
    """A text-only choice model cannot play browser games."""
    capabilities = driver.capabilities
    if (
        capabilities.action_channel == "choice"
        and capabilities.observation_modality != "screenshot"
        and task.category == "browser_game"
    ):
        raise ValueError(f"Choice Decisions supports Web task evaluation only: {task.task_id}")


def check_fps(driver, model_input_fps) -> None:
    """A video cap bounds the model-input rate (ModelConfig.fps); the environment may run faster."""
    capabilities = driver.capabilities
    if (
        capabilities.max_video_fps is not None
        and float(model_input_fps) > capabilities.max_video_fps
    ):
        raise ValueError(
            f"{capabilities.protocol_family} supports at most "
            f"{capabilities.max_video_fps:g} fps; requested {model_input_fps:g}"
        )


def check_session_config(driver, config) -> None:
    """Settings a driver applied when its session connected, after the frame-rate check."""
    if (
        issubclass(driver, GeminiLiveDriver)
        and config.reasoning_effort not in LIVE_REASONING_EFFORTS
    ):
        raise ValueError(
            "Gemini Live reasoning_effort must be one of "
            "none, fixed, minimal, low, medium, or high"
        )


_HISTORY_FORMS = {"canonical": "turns", "action_digest": "action_digest"}


def check_context_policy(driver, policy, config) -> None:
    """The context policy's items, held against what the encoder declares it can render."""
    capabilities = driver.capabilities
    api = config.adapter_type
    if policy.feedback not in capabilities.feedback_placements:
        raise ValueError(f"The {api} API cannot place execution feedback as {policy.feedback}")
    if policy.ledger_role not in capabilities.roles and not capabilities.labels_instead_of_roles:
        raise ValueError(f"The {api} API has no {policy.ledger_role} role for the action ledger")
    if capabilities.reasoning_replay == "required" and not policy.reasoning_replay:
        raise ValueError(
            f"The {api} API requires past reasoning in later requests; "
            "its context policy sends none"
        )
    if capabilities.reasoning_replay == "unsupported" and policy.reasoning_replay:
        raise ValueError(f"The {api} API cannot send past reasoning back")
    if (
        policy.pressure_archive
        and config.context_window_tokens is not None
        and not capabilities.request_size_estimation
    ):
        raise ValueError(
            f"The {api} API declares a context window but no estimate of its requests"
        )
    form = "turns" if policy.decision_history is None else policy.decision_history.form
    if form != _HISTORY_FORMS[capabilities.history_replay]:
        raise ValueError(f"The {api} API cannot replay history as {form}")
    if (policy.observation == "observation_json_text") != (
        capabilities.observation_modality == "text"
    ):
        raise ValueError(
            f"The {api} API takes {capabilities.observation_modality} input, "
            f"not the {policy.observation} observation"
        )


def check_episode(
    model_config, *, clock_mode, task, interaction_mode="turn_based", env_fps=None
) -> None:
    """The runner's rules, applied once before it builds the agent.

    Among themselves the rules keep the order in which they fired before: the clock check, then
    the driver's constructor, then initialize (the choice-task and frame-rate checks), then the
    session connect. All of them now precede the driver's construction, so the initialize and
    connect rules are reported before a missing API key, which the constructor reports. The
    interaction-mode rules follow the clock check, as in validate_model_interaction. The context
    policy is checked last. ``task`` is the registered task (its category and task_id).
    ``env_fps`` is the environment frame rate, checked first against the model-input rate
    ``model_config.fps`` as in validate_model_interaction; without it that check is left to
    effective_run_config.
    """
    if env_fps is not None:
        check_frame_rates(env_fps, model_config.fps)
    adapter_type = model_config.adapter_type
    driver = adapter_class(adapter_type)
    check_wall_clock(driver, adapter_type, clock_mode)
    check_interaction_mode(
        driver,
        adapter_type,
        interaction_mode=interaction_mode,
        clock_mode=clock_mode,
        model_input_fps=model_config.fps,
    )
    check_config(driver, model_config)
    check_choice_task(driver, task)
    check_fps(driver, model_config.fps)
    check_session_config(driver, model_config)
    policy = current_context_policy().for_path(
        adapter_type, model_config.provider_options, model_config.context_preset
    )
    check_context_policy(driver, policy, model_config)
