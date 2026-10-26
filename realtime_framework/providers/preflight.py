"""Pre-run checks: every rule that rejects a run configuration before an episode starts.

Each rule reads only the registered driver class's declarations (capabilities, deployment,
compat flags, option tables), the configuration and, for check_context_policy, the context
policy the run resolves. Every entry point keeps the rules it applied before the checks were
collected here:

- runner argparse (evaluation/runner.py) and identity_main (evaluation/config.py):
  check_token_budget and check_streaming_options, which effective_run_config applies too;
- run plans, effective_run_config and the Harbor trial worker: validate_model_interaction
  (check_frame_rates, check_wall_clock, check_interaction_mode);
- Harbor's agent options (evaluation/harbor.py RealtimeBenchOptions): check_frame_rates;
- the runner's argparse, before it starts a container, and the Harbor trial worker, whose dry run
  Harbor's resolve makes (evaluation/_trial.py): check_episode; the trial worker applies
  check_token_budget too.

Construction-time failures (credentials, SDK pin, endpoint URL, Decisions route) stay in the
drivers, so they now come after every check_episode rule. When a run has two faults, the first
one reported can therefore differ from before: the choice-task, frame-rate and Gemini session
checks used to fire in initialize or at the session connect, after the constructor had looked
for the API key, and are now reported ahead of a missing key.
"""

from __future__ import annotations

from realtime_framework.entity.context_policy import PROMPT_CHANNELS, current_context_policy
from realtime_framework.providers.registry import adapter_class


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
    pause clock) and a model-input rate within the model's video cap and its streaming cap."""
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
    cap = capabilities.max_streaming_fps
    if cap is not None and float(model_input_fps) > cap:
        raise ValueError(
            f"{capabilities.protocol_family} streams at most {cap:g} fps; "
            f"requested {model_input_fps:g}"
        )


def check_streaming_options(interaction_mode, *, think_budget_s, per_move_cap_s) -> None:
    """The turn-based deliberation caps bound a model's turns, and a streaming episode has none
    (evaluation/runner.py run_stream), so a streaming run refuses them rather than run without
    them. No other run option is turn-based only: the token clock's, tokens_per_frame and the
    time-aware protocol, go with the token clock, which streaming refuses."""
    if interaction_mode == "turn_based":
        return
    given = [
        option
        for option, value in (
            ("--think-budget-s", think_budget_s),
            ("--per-move-cap-s", per_move_cap_s),
        )
        if value is not None
    ]
    if given:
        raise ValueError(f"Streaming does not support {' or '.join(given)}")


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
    """Static model-config rules, in the order the driver's constructor applied them; the
    context reserve, which the estimating APIs also checked before each request, comes last,
    then the reserve against the output limit each request sends."""
    compat = driver.compat
    if compat.output_limit_required and config.native_output_limit is None:
        raise ValueError(
            f"The {config.adapter_type} API requires an output limit on every request; "
            "declare endpoint.native_output_limit in the model YAML"
        )
    check_provider_options(compat, config)
    if (
        driver.capabilities.request_size_estimation
        and config.context_window_tokens is not None
        and config.context_reserve_tokens >= config.context_window_tokens
    ):
        raise ValueError("Context capacity must exceed generation reserve")
    # An API that maps the decision's cap onto its request (maps_output_cap; LiteLLMDriver
    # transport_output_token_cap) sends the native limit when no decision cap applies, so a
    # request the window admits must leave room for that output.
    limit = config.native_output_limit
    if (
        driver.capabilities.request_size_estimation
        and driver.capabilities.maps_output_cap
        and config.context_window_tokens is not None
        and limit is not None
        and config.context_reserve_tokens < limit
    ):
        raise ValueError(
            f"The context reserve ({config.context_reserve_tokens} tokens) is below the output "
            f"limit each request sends ({limit} tokens); set BENCHMARK_CONTEXT_RESERVE_TOKENS to "
            f"at least {limit}, or lower endpoint.native_output_limit in the model YAML"
        )


def check_provider_options(compat, config) -> None:
    """The model's provider_options against the option table its API's compat declares. What
    one model of the API takes and another does not is in each model's YAML: its options, and
    the reasoning efforts the catalog takes for it (providers/registry.py load_definitions)."""
    api, options, table = config.adapter_type, config.provider_options, compat.provider_options
    unknown = set(options) - set(table)
    if unknown:
        raise ValueError(f"Unknown {api} provider_options: {sorted(unknown)}")
    resolved = {name: options.get(name, default) for name, (default, _) in table.items()}
    for name, (_, allowed) in table.items():
        if resolved[name] not in allowed:
            raise ValueError(
                f"{api} {name} must be one of {list(allowed)}, got {resolved[name]!r}"
            )
    missing = [name for name in compat.required_provider_options if name not in options]
    if missing:
        raise ValueError(
            f"The {api} API requires provider_options {missing}; set them in the model YAML"
        )
    for (name, value), (other, required) in compat.provider_option_requires.items():
        if resolved[name] == value and resolved[other] != required:
            raise ValueError(f"{api} {name}={value} requires {other}={required}")


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
    efforts = driver.compat.reasoning_efforts
    if efforts is not None and config.reasoning_effort not in efforts:
        named = [effort for effort in efforts if effort is not None]
        raise ValueError(
            f"{config.adapter_type} reasoning_effort must be one of {named}, "
            f"got {config.reasoning_effort!r}"
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
    if (
        policy.tool_call_replay is not None
        and policy.tool_call_replay not in capabilities.tool_call_replay
    ):
        raise ValueError(
            f"The {api} API cannot replay past tool calls as {policy.tool_call_replay}"
        )
    if (policy.observation == "observation_json_text") != (
        capabilities.observation_modality == "text"
    ):
        raise ValueError(
            f"The {api} API takes {capabilities.observation_modality} input, "
            f"not the {policy.observation} observation"
        )
    # The observation JSON carries a choice's candidates (harnesses/jev/observation.py
    # choice_observation); the other channels answer from the SoM screenshot.
    choice = capabilities.action_channel == "choice"
    if choice and policy.observation == "som_screenshot":
        raise ValueError(
            f"The {api} API chooses among the candidates of an observation JSON, "
            f"not of the {policy.observation} observation"
        )
    if not choice and policy.observation != "som_screenshot":
        raise ValueError(
            f"The {api} API answers in {capabilities.action_channel}; "
            f"the {policy.observation} observation is for choice APIs"
        )
    # The harness picks the system prompt by the profile, and the function declarations and the
    # reply's collector by the driver's channel (harnesses/default/harness.py).
    for category, prompt in policy.prompts.items():
        if prompt is not None and PROMPT_CHANNELS[prompt] != capabilities.action_channel:
            raise ValueError(
                f"The {api} API answers in {capabilities.action_channel}; its context policy "
                f"prompts {category} tasks with {prompt}, which asks for {PROMPT_CHANNELS[prompt]}"
            )


def check_episode(
    model_config, *, clock_mode, task, env_fps, interaction_mode="turn_based"
) -> None:
    """The runner's rules, applied before it starts a container (and by the Harbor trial worker).

    Among themselves the rules keep the order in which they fired before: the clock check, then
    the driver's constructor, then initialize (the choice-task and frame-rate checks), then the
    session connect. All of them now precede the driver's construction, so the initialize and
    connect rules are reported before a missing API key, which the constructor reports. The
    interaction-mode rules follow the clock check, as in validate_model_interaction. The context
    policy is checked last. ``task`` is the registered task (its category and task_id).
    ``env_fps`` is the environment frame rate, checked first against the model-input rate
    ``model_config.fps`` as in validate_model_interaction.
    """
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
