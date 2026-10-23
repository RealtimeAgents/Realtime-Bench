"""Every pre-run check that can reject a configuration, at each entry point that applies it.

One test per condition, asserting the exact message each check produced before the checks were
collected in providers/preflight.py (recorded on 8319a10, unchanged on 9ada3b0); the context-policy
checks came later and are pinned as written. The runner's own rules run in check_episode
(evaluation/runner.py set_up_agent); its argparse checks and those of identity_main and the
Harbor trial worker are exercised through their main functions, which fail before any container
or network is used.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import pytest

import realtime_framework.providers.registry as catalog
import realtime_tasks.registry as tasks
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import (
    PRESETS,
    STANDARD,
    DecisionHistory,
    current_context_policy,
)
from realtime_framework.evaluation.config import effective_run_config, load_benchmark
from realtime_framework.providers.preflight import (
    UnsupportedTokenBudget,
    check_context_policy,
    check_episode,
    check_token_budget,
    validate_model_interaction,
)
from realtime_framework.providers.session_lifecycle import SessionLifecycle

GAME, WEB = "gomoku", "rt_waitlist_fit"


@pytest.fixture(autouse=True)
def environment():
    """The entry points write os.environ (runner.py main; _trial.py:101-102)."""
    saved = dict(os.environ)
    os.environ.pop("ALLOW_EXPERIMENTAL_MODEL", None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def preset(key, **overrides):
    """What the runner builds for a catalog key (runner.py main)."""
    entry = catalog.get(key)
    return catalog.to_model_config(
        key, **{"fps": 1.0, "reasoning_effort": entry.defaults["reasoning_effort"], **overrides}
    )


def episode(
    model_config, *, clock_mode="wall", task=WEB, interaction_mode="turn_based", env_fps=None
):
    """check_episode with the environment at the model-input rate unless ``env_fps`` is given."""
    check_episode(
        model_config, clock_mode=clock_mode, task=tasks.get(task),
        interaction_mode=interaction_mode,
        env_fps=model_config.fps if env_fps is None else env_fps,
    )


def fails(call, message, kind=ValueError):
    with pytest.raises(kind) as caught:
        call()
    assert type(caught.value) is kind
    assert str(caught.value) == message


def usage_error(capsys, call, message):
    """An argparse error: exit status 2 and the message as the last stderr line."""
    with pytest.raises(SystemExit) as caught:
        call()
    assert caught.value.code == 2
    assert capsys.readouterr().err.strip().splitlines()[-1].endswith("error: " + message)


def runner_main(*argv):
    def call():
        from realtime_framework.evaluation.runner import main

        sys.argv = ["runner", *argv]
        main()

    return call


def plan(tmp_path, text):
    path = tmp_path / "plan.yaml"
    path.write_text(text)
    return lambda: load_benchmark(path)


# Wall clock only (runner, run plans, effective_run_config, Harbor trial worker)


@pytest.mark.parametrize(
    "key, api",
    [
        ("my-model", "chat_completions"),
        ("litellm-vlm", "litellm"),
        ("jev-1.13", "jev_decisions"),
        ("gpt-6-luna-decisions", "openai_decisions"),
    ],
)
def test_wall_clock_only_runner(key, api):
    fails(lambda: episode(preset(key), clock_mode="token"), f"The {api} API requires a wall clock")


def test_wall_clock_only_plan(tmp_path):
    fails(
        plan(tmp_path, "models: [my-model]\nruntime: {clock_mode: token, tokens_per_frame: 64}\n"),
        "The chat_completions API requires a wall clock",
    )


def test_wall_clock_only_run_config():
    fails(
        lambda: effective_run_config(
            model_config=preset("jev-1.13"), fps=1.0, max_minutes=5.0,
            protocol="time-aware-token-v1", think_budget_s=None, per_move_cap_s=None,
            clock_mode="token", tokens_per_frame=None,
        ),
        "The jev_decisions API requires a wall clock",
    )


def test_wall_clock_only_trial(capsys):
    from realtime_framework.evaluation._trial import main

    usage_error(
        capsys,
        lambda: main(["--catalog-model", "my-model", "--task", WEB, "--clock", "token",
                      "--dry-run"]),
        "The chat_completions API requires a wall clock",
    )


# A model-input rate above the environment rate (runner, run plans, effective_run_config,
# identity_main, Harbor agent options and trial worker)

FASTER_MODEL_INPUT = "The model-input frame rate (2 fps) exceeds the environment frame rate (1 fps)"


def test_faster_model_input_runner():
    fails(lambda: episode(preset("my-model", fps=2.0), env_fps=1.0), FASTER_MODEL_INPUT)
    episode(preset("my-model", fps=2.0), env_fps=2.0)


def test_faster_model_input_plan(tmp_path):
    fails(
        plan(tmp_path, "models: [my-model]\nruntime: {clock_mode: wall, fps: 1, "
                       "model_input_fps: 2}\n"),
        FASTER_MODEL_INPUT,
    )


def test_faster_model_input_run_config():
    fails(
        lambda: effective_run_config(
            model_config=preset("my-model", fps=2.0), fps=1.0, max_minutes=5.0,
            protocol="wall-clock", think_budget_s=None, per_move_cap_s=None,
        ),
        FASTER_MODEL_INPUT,
    )


def test_faster_model_input_identity():
    from realtime_framework.evaluation.config import identity_main

    fails(
        lambda: identity_main(["--model", "my-model", "--task", WEB, "--fps", "1",
                               "--model-input-fps", "2", "--max-minutes", "5",
                               "--protocol", "wall-clock", "--clock", "wall"]),
        FASTER_MODEL_INPUT,
    )


def test_faster_model_input_harbor_options():
    """Harbor checks the agent's options once --ak is merged in, so an --ak fps that takes a
    plan's environment rate below its model-input rate is refused too."""
    from pydantic import ValidationError

    from realtime_framework.evaluation.harbor import RealtimeBenchOptions

    with pytest.raises(ValidationError) as caught:
        RealtimeBenchOptions(fps=1, model_input_fps=2)
    (error,) = caught.value.errors()
    assert error["msg"] == "Value error, " + FASTER_MODEL_INPUT
    RealtimeBenchOptions(fps=2, model_input_fps=2)


def test_faster_model_input_trial(capsys):
    from realtime_framework.evaluation._trial import main

    usage_error(
        capsys,
        lambda: main(["--catalog-model", "my-model", "--task", WEB, "--fps", "1",
                      "--model-input-fps", "2", "--dry-run"]),
        FASTER_MODEL_INPUT,
    )


# Finite token clock (runner and identity_main argparse)

FINITE_BUDGET = "adapter 'chat_completions' cannot report complete usage for a finite token clock"


def test_finite_budget_runner(capsys):
    usage_error(
        capsys,
        runner_main("--model", "my-model", "--task", WEB, "--clock", "token",
                    "--tokens-per-frame", "64"),
        FINITE_BUDGET,
    )


def test_finite_budget_identity(capsys):
    from realtime_framework.evaluation.config import identity_main

    usage_error(
        capsys,
        lambda: identity_main(["--model", "my-model", "--task", WEB, "--fps", "1",
                               "--max-minutes", "5", "--protocol", "time-aware-token-v1",
                               "--clock", "token", "--tokens-per-frame", "64"]),
        FINITE_BUDGET,
    )


def test_finite_budget_unknown_adapter_is_not_a_usage_error():
    """A raw model id without --adapter, on the runner's default token clock.

    The lookup error is raised as before, not turned into a usage error (runner.py main).
    """
    fails(
        lambda: check_token_budget(None, clock_mode="token", tokens_per_frame=256),
        "Unknown adapter type: None",
    )
    fails(
        lambda: check_token_budget("chat_completions", clock_mode="token", tokens_per_frame=64),
        FINITE_BUDGET,
        UnsupportedTokenBudget,
    )


# Inline endpoints in a run plan


def test_plan_endpoint_requires_wall_clock(tmp_path):
    fails(
        plan(
            tmp_path,
            "models: [{key: remote-vlm, endpoint: {api: chat_completions, model: m, "
            "base_url: 'http://127.0.0.1:9/v1'}}]\n"
            "runtime: {clock_mode: token, tokens_per_frame: 64}\n",
        ),
        "endpoint settings require the wall clock",
    )


# The runner's episode rules (check_episode)


def test_frame_rate_cap():
    fails(
        lambda: episode(preset("gemini-live", fps=2.0)),
        "gemini_live supports at most 1 fps; requested 2",
    )


def test_text_only_choice_model_on_game():
    fails(
        lambda: episode(preset("jev-1.13"), task=GAME),
        "Choice Decisions supports Web task evaluation only: gomoku",
    )
    episode(preset("jev-1.13"), task=WEB)
    episode(preset("gpt-6-luna-decisions"), task=GAME)


def test_output_limit_required():
    # An anthropic_messages YAML for a model LiteLLM's map does not list, without
    # endpoint.native_output_limit (providers/registry.py resolve_output_limit).
    config = ModelConfig(
        adapter_type="anthropic_messages", model_name="claude-unlisted",
        api_key_env="ANTHROPIC_API_KEY", reasoning_effort="low",
    )
    fails(
        lambda: episode(config),
        "The anthropic_messages API requires max_tokens; declare endpoint.native_output_limit "
        "in the model YAML",
    )


def test_context_reserve():
    os.environ["BENCHMARK_CONTEXT_WINDOW_TOKENS"] = "4096"  # providers/registry.py to_model_config
    fails(lambda: episode(preset("my-model")), "Context capacity must exceed generation reserve")


@pytest.mark.parametrize("key", ["jev-1.13", "gpt-6-luna-decisions"])
def test_context_reserve_of_a_choice_api(key):
    """Every API that estimates its requests against a window, not only chat_completions."""
    config = preset(key)
    fails(
        lambda: episode(preset(key, context_reserve_tokens=config.context_window_tokens)),
        "Context capacity must exceed generation reserve",
    )


def gemini(**fields):
    return ModelConfig(adapter_type="gemini_live", api_key_env="GEMINI_API_KEY", **fields)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"bogus": 1}, "Unknown Gemini Live provider_options: ['bogus']"),
        (
            {"tool_behavior": "sometimes"},
            (
                "Gemini Live tool_behavior must be one of ['blocking', 'non_blocking'], "
                "got 'sometimes'"
            ),
        ),
    ],
)
def test_gemini_options(options, message):
    fails(
        lambda: episode(gemini(model_name="gemini-3.1-flash-live-preview",
                               provider_options=options)),
        message,
    )


EXTENDED = "gemini-3.8-live-extended-thinking"


@pytest.mark.parametrize(
    "config, message",
    [
        (
            gemini(model_name="gemini-3.8-live", reasoning_effort="fixed",
                   provider_options={"tool_response_scheduling": "silent"}),
            "Silent Gemini tool responses require non-blocking tools",
        ),
        (
            gemini(model_name="gemini-3.1-flash-live-preview",
                   provider_options={"tool_behavior": "non_blocking"}),
            "gemini-3.1-flash-live-preview calls functions synchronously only; "
            "use tool_behavior=blocking",
        ),
        # Both 3.1 faults in one message, so fixing one does not raise the other.
        (
            gemini(model_name="gemini-3.1-flash-live-preview",
                   provider_options={"tool_behavior": "non_blocking",
                                     "tool_response_scheduling": "silent"}),
            (
                "gemini-3.1-flash-live-preview calls functions synchronously only; "
                "use tool_behavior=blocking and tool_response_scheduling=omit"
            ),
        ),
        (
            gemini(model_name="gemini-3.1-flash-live-preview",
                   provider_options={"tool_response_scheduling": "silent"}),
            (
                "gemini-3.1-flash-live-preview calls functions synchronously only; "
                "use tool_response_scheduling=omit"
            ),
        ),
        (
            preset("gemini-3.8-live", reasoning_effort="low"),
            "gemini-3.8-live requires omitting thinking_config; use reasoning_effort=fixed",
        ),
        (
            preset(EXTENDED, reasoning_effort="none"),
            f"{EXTENDED} requires reasoning_effort must be low, medium, or high",
        ),
        (
            gemini(model_name=EXTENDED, reasoning_effort="low"),
            (
                f"{EXTENDED} requires tool_behavior must be non_blocking; "
                "completion_signal must be interaction_idle"
            ),
        ),
        (
            gemini(model_name=EXTENDED, reasoning_effort="low", provider_options={
                "tool_behavior": "non_blocking", "completion_signal": "interaction_idle",
                "tool_response_scheduling": "silent",
            }),
            (
                f"{EXTENDED} requires function scheduling is unsupported; "
                "use tool_response_scheduling=omit"
            ),
        ),
    ],
    ids=["silent-blocking", "3.1-non-blocking", "3.1-non-blocking-silent", "3.1-silent",
         "3.8-live-effort", "extended-effort", "extended-options", "extended-silent"],
)
def test_gemini_model_contract(config, message):
    fails(lambda: episode(config), message)


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_gemini_reasoning_effort(effort):
    fails(
        lambda: episode(preset("gemini-live", reasoning_effort=effort)),
        "Gemini Live reasoning_effort must be one of none, fixed, minimal, low, medium, or high",
    )


@pytest.mark.parametrize(
    "key, effort, message",
    [
        # The model contract came from the driver's constructor, before initialize's fps check.
        ("gemini-3.8-live", "low",
         "gemini-3.8-live requires omitting thinking_config; use reasoning_effort=fixed"),
        # The fps check came from initialize, before the session connect mapped the effort.
        ("gemini-live", "xhigh", "gemini_live supports at most 1 fps; requested 2"),
    ],
)
def test_episode_rule_order(key, effort, message):
    fails(lambda: episode(preset(key, fps=2.0, reasoning_effort=effort)), message)


# The context policy against the encoder's declarations (check_context_policy, the last rule of
# check_episode). No catalog model breaks a rule, so each case changes one item of its policy.


def test_context_policy_fits_every_catalog_model():
    for key in catalog.CATALOG:
        episode(preset(key))


def policy_check(config, **values):
    driver = catalog.adapter_class(config.adapter_type)
    policy = current_context_policy().for_path(
        config.adapter_type, config.provider_options, config.context_preset
    )
    return lambda: check_context_policy(driver, replace(policy, **values), config)


@pytest.mark.parametrize(
    "key, values, message",
    [
        (
            "gpt-realtime-2",
            {"feedback": "frame_prefix"},
            "The openai_realtime API cannot place execution feedback as frame_prefix",
        ),
        (
            "jev-1.13",
            {"feedback": "separate_message"},
            "The jev_decisions API cannot place execution feedback as separate_message",
        ),
        (
            "gemini-live",
            {"ledger_role": "instruction"},
            "The gemini_live API has no instruction role for the action ledger",
        ),
        (
            "claude-opus-5",
            {"reasoning_replay": True},
            "The openrouter API cannot send past reasoning back",
        ),
        (
            "jev-1.13",
            {"decision_history": DecisionHistory("turns")},
            "The jev_decisions API cannot replay history as turns",
        ),
        (
            "gpt-6-luna-decisions",
            {"observation": "observation_json_text"},
            "The openai_decisions API takes screenshot input, not the observation_json_text "
            "observation",
        ),
        (
            "jev-1.13",
            {"observation": "observation_json_screenshot"},
            "The jev_decisions API takes text input, not the observation_json_screenshot "
            "observation",
        ),
        # Without the observation JSON a choice API has no candidates (harnesses/jev/
        # observation.py choice_observation), and the other channels have no use for them.
        (
            "gpt-6-luna-decisions",
            {"observation": "som_screenshot"},
            (
                "The openai_decisions API chooses among the candidates of an observation JSON, "
                "not of the som_screenshot observation"
            ),
        ),
        (
            "claude-opus-5",
            {"observation": "observation_json_screenshot"},
            (
                "The openrouter API answers in text_json; the observation_json_screenshot "
                "observation is for choice APIs"
            ),
        ),
        (
            "claude-opus-5",
            {"tool_call_replay": "tool_transcript"},
            "The openrouter API cannot replay past tool calls as tool_transcript",
        ),
        (
            "gpt-realtime-2",
            {"tool_call_replay": "native_tools"},
            "The openai_realtime API cannot replay past tool calls as native_tools",
        ),
    ],
    ids=["live-feedback", "digest-feedback", "ledger-role", "reasoning-unsupported",
         "history-form", "screenshot-input", "text-input", "choice-observation",
         "chat-observation", "tool-transcript", "native-tools"],
)
def test_context_policy_item(key, values, message):
    fails(policy_check(preset(key), **values), message)


def test_kimi_k3_preset_comes_from_its_model_config():
    """Kimi requires K3's past reasoning back (entity/context_policy.py PRESETS): K3's YAML names
    the kimi_k3 preset. Another moonshot model runs under the standard values, which send none."""
    policy = current_context_policy()
    k3 = preset("kimi-k3")
    assert k3.context_preset == "kimi_k3"
    resolved = policy.for_path(k3.adapter_type, k3.provider_options, k3.context_preset)
    assert (resolved.preset, resolved.reasoning_replay) == ("kimi_k3", True)
    episode(k3)
    other = k3.model_copy(update={"model_name": "kimi-other", "context_preset": None})
    resolved = policy.for_path(other.adapter_type, other.provider_options, other.context_preset)
    assert resolved == policy.paths["moonshot"]
    assert {item: getattr(resolved, item) for item in PRESETS["kimi_k3"]} == {
        item: STANDARD[item] for item in PRESETS["kimi_k3"]
    }
    episode(other)


def test_model_preset_the_api_cannot_render():
    fails(
        lambda: episode(preset("claude-opus-5", context_preset="kimi_k3")),
        "The openrouter API cannot send past reasoning back",
    )


def test_context_window_needs_a_request_estimate():
    # Luna and JEV estimate their requests against their declared windows; OpenRouter
    # declares no estimate.
    policy_check(preset("gpt-6-luna-decisions"))()
    policy_check(preset("jev-1.13"))()
    fails(
        policy_check(preset("claude-opus-5", context_window_tokens=200000)),
        "The openrouter API declares a context window but no estimate of its requests",
    )


# Native streaming (check_interaction_mode): rules added with the interaction mode, so their
# messages are new rather than recorded on 8319a10.

NOT_LIVE = (
    "Streaming requires a live API (openai_realtime or gemini_live); chat_completions is not one"
)
STREAMING_CLOCK = "Streaming requires a wall clock"


def undeclared(monkeypatch, api: str) -> None:
    """The live ``api``'s driver without streaming among its interaction modes. Both live drivers
    declare it, so the rule that refuses a live API without it is held to one this way."""
    driver = catalog.adapter_class(api)
    capabilities = replace(driver.capabilities, interaction_modes=frozenset({"turn_based"}))
    monkeypatch.setattr(driver, "capabilities", capabilities)


def test_the_live_apis_declare_streaming():
    for api in ("openai_realtime", "gemini_live"):
        assert "streaming" in catalog.adapter_class(api).capabilities.interaction_modes


def test_streaming_needs_a_live_api():
    fails(lambda: episode(preset("my-model"), interaction_mode="streaming"), NOT_LIVE)


def test_streaming_needs_a_live_api_run_config():
    fails(
        lambda: effective_run_config(
            model_config=preset("my-model"), fps=1.0, max_minutes=5.0, protocol="wall-clock",
            think_budget_s=None, per_move_cap_s=None, clock_mode="wall",
            interaction_mode="streaming",
        ),
        NOT_LIVE,
    )


def test_streaming_needs_a_live_api_plan(tmp_path):
    fails(
        plan(tmp_path, "models: [my-model]\nruntime: {clock_mode: wall, "
                       "interaction_mode: streaming}\n"),
        NOT_LIVE,
    )


def test_streaming_plan_names_a_mode(tmp_path):
    fails(
        plan(tmp_path, "models: [my-model]\nruntime: {interaction_mode: live}\n"),
        "runtime.interaction_mode must be 'turn_based' or 'streaming'",
    )


def test_streaming_trial(monkeypatch, capsys):
    from realtime_framework.evaluation._trial import main

    undeclared(monkeypatch, "gemini_live")

    usage_error(
        capsys,
        lambda: main(["--catalog-model", "gemini-live", "--task", WEB, "--interaction-mode",
                      "streaming", "--dry-run"]),
        "The gemini_live API does not support streaming",
    )


def test_streaming_identity(monkeypatch):
    from realtime_framework.evaluation.config import identity_main

    undeclared(monkeypatch, "gemini_live")

    fails(
        lambda: identity_main(["--model", "gemini-live", "--task", WEB, "--fps", "1",
                               "--max-minutes", "5", "--protocol", "wall-clock", "--clock",
                               "wall", "--interaction-mode", "streaming"]),
        "The gemini_live API does not support streaming",
    )


def test_the_live_apis_are_the_persistent_sessions():
    """The APIs the message names are the drivers that declare a persistent session."""
    drivers = {api: catalog.adapter_class(api) for api in catalog.APIS}
    live = {api for api, driver in drivers.items() if driver.capabilities.persistent_session}
    assert live == {"openai_realtime", "gemini_live"}


@pytest.mark.parametrize(
    "key, api", [("gpt-realtime-2", "openai_realtime"), ("gemini-live", "gemini_live")]
)
def test_streaming_needs_the_driver_to_declare_it(monkeypatch, key, api):
    undeclared(monkeypatch, api)
    fails(
        lambda: episode(preset(key), interaction_mode="streaming"),
        f"The {api} API does not support streaming",
    )


def test_streaming_needs_the_wall_clock():
    fails(
        lambda: episode(preset("gemini-live"), clock_mode="token", interaction_mode="streaming"),
        STREAMING_CLOCK,
    )


@pytest.mark.parametrize("tokens_per_frame", ["64", "unlimited"], ids=["token", "pause"])
def test_streaming_needs_the_wall_clock_plan(tmp_path, tokens_per_frame):
    fails(
        plan(tmp_path, "models: [gemini-live]\nruntime: {clock_mode: token, tokens_per_frame: "
                       f"{tokens_per_frame}, interaction_mode: streaming}}\n"),
        STREAMING_CLOCK,
    )


def test_streaming_needs_the_wall_clock_trial(capsys):
    from realtime_framework.evaluation._trial import main

    usage_error(
        capsys,
        lambda: main(["--catalog-model", "gemini-live", "--task", WEB, "--clock", "token",
                      "--interaction-mode", "streaming", "--dry-run"]),
        STREAMING_CLOCK,
    )


def test_streaming_frame_rate_cap():
    """The run plans and the Harbor trial worker apply the cap to a streaming run too; on a turn-
    based run only the runner does (check_fps)."""
    cap = "gemini_live supports at most 1 fps; requested 2"
    fails(
        lambda: validate_model_interaction(
            "gemini_live", clock_mode="wall", fps=2.0, model_input_fps=2.0,
            interaction_mode="streaming",
        ),
        cap,
    )
    validate_model_interaction("gemini_live", clock_mode="wall", fps=2.0, model_input_fps=2.0)
    fails(lambda: episode(preset("gemini-live", fps=2.0), interaction_mode="streaming"), cap)
    episode(preset("gemini-live"), interaction_mode="streaming")


OPENAI_STREAMING_CAP = "openai_realtime streams at most 1 fps; requested 2"


def test_streaming_frame_rate_cap_of_openai_realtime():
    """OpenAI Realtime's streaming cap (max_streaming_fps): validate_model_interaction and
    check_episode here, and the trial worker below, apply it to a streaming run only; a
    turn-based run at 2 fps goes on."""
    fails(
        lambda: validate_model_interaction(
            "openai_realtime", clock_mode="wall", fps=2.0, model_input_fps=2.0,
            interaction_mode="streaming",
        ),
        OPENAI_STREAMING_CAP,
    )
    validate_model_interaction("openai_realtime", clock_mode="wall", fps=2.0, model_input_fps=2.0)
    fails(
        lambda: episode(
            preset("gpt-realtime-2", fps=2.0), interaction_mode="streaming", env_fps=2.0
        ),
        OPENAI_STREAMING_CAP,
    )
    episode(preset("gpt-realtime-2", fps=2.0), env_fps=2.0)
    episode(preset("gpt-realtime-2"), interaction_mode="streaming")


def test_streaming_frame_rate_cap_of_openai_realtime_trial(capsys):
    from realtime_framework.evaluation._trial import main

    usage_error(
        capsys,
        lambda: main(["--catalog-model", "gpt-realtime-2", "--task", WEB, "--fps", "2",
                      "--interaction-mode", "streaming", "--dry-run"]),
        OPENAI_STREAMING_CAP,
    )


# The turn-based deliberation caps, which a streaming episode has no turns for (N41).

CAPS = [
    (["--think-budget-s", "5"], "Streaming does not support --think-budget-s"),
    (["--per-move-cap-s", "2"], "Streaming does not support --per-move-cap-s"),
    (
        ["--think-budget-s", "5", "--per-move-cap-s", "2"],
        "Streaming does not support --think-budget-s or --per-move-cap-s",
    ),
]
CAP_IDS = ["think-budget", "per-move-cap", "both"]


@pytest.mark.parametrize("caps, message", CAPS, ids=CAP_IDS)
def test_streaming_refuses_deliberation_caps_runner(capsys, caps, message):
    usage_error(
        capsys,
        runner_main("--model", "gemini-live", "--task", WEB, "--clock", "wall",
                    "--interaction-mode", "streaming", *caps),
        message,
    )


@pytest.mark.parametrize("caps, message", CAPS, ids=CAP_IDS)
def test_streaming_refuses_deliberation_caps_identity(capsys, caps, message):
    from realtime_framework.evaluation.config import identity_main

    usage_error(
        capsys,
        lambda: identity_main(["--model", "gemini-live", "--task", WEB, "--fps", "1",
                               "--max-minutes", "5", "--protocol", "wall-clock", "--clock",
                               "wall", "--interaction-mode", "streaming", *caps]),
        message,
    )


@pytest.mark.parametrize(
    "think_budget_s, per_move_cap_s, message",
    [(5.0, None, CAPS[0][1]), (None, 2.0, CAPS[1][1]), (5.0, 2.0, CAPS[2][1])],
    ids=CAP_IDS,
)
def test_streaming_refuses_deliberation_caps_run_config(think_budget_s, per_move_cap_s, message):
    """effective_run_config, which reporting.prepare_episode_report computes the expected hash
    with; a turn-based wall-clock run keeps its caps."""

    def run_config(interaction_mode):
        return effective_run_config(
            model_config=preset("gemini-live"), fps=1.0, max_minutes=5.0, protocol="wall-clock",
            think_budget_s=think_budget_s, per_move_cap_s=per_move_cap_s, clock_mode="wall",
            interaction_mode=interaction_mode,
        )

    fails(lambda: run_config("streaming"), message)
    config = run_config("turn_based")
    assert (config["think_budget_s"], config["per_move_cap_s"]) == (think_budget_s, per_move_cap_s)


def test_harbor_carries_no_deliberation_caps(capsys):
    """Harbor's agent options and the trial worker it runs take neither cap, so no Harbor request
    hands one to a streaming run."""
    from pydantic import ValidationError

    from realtime_framework.evaluation._trial import main
    from realtime_framework.evaluation.harbor import RealtimeBenchOptions

    for field in ("think_budget_s", "per_move_cap_s"):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RealtimeBenchOptions(interaction_mode="streaming", **{field: 5.0})
    usage_error(
        capsys,
        lambda: main(["--catalog-model", "gemini-live", "--task", WEB, "--interaction-mode",
                      "streaming", "--think-budget-s", "5", "--dry-run"]),
        "unrecognized arguments: --think-budget-s 5",
    )


def test_streaming_frame_rate_cap_plan(tmp_path):
    """The plan's model-input rate defaults to its environment rate."""
    fails(
        plan(tmp_path, "models: [gemini-live]\nruntime: {clock_mode: wall, fps: 2, "
                       "interaction_mode: streaming}\n"),
        "gemini_live supports at most 1 fps; requested 2",
    )


# Each streaming rule at the entry points the tests above leave out.

STREAMING_CAP = "gemini_live supports at most 1 fps; requested 2"


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--catalog-model", "my-model"], NOT_LIVE),
        (["--catalog-model", "gemini-live", "--clock", "token", "--tokens-per-frame", "unlimited"],
         STREAMING_CLOCK),
        (["--catalog-model", "gemini-live", "--fps", "2"], STREAMING_CAP),
    ],
    ids=["not-live", "pause-clock", "frame-rate-cap"],
)
def test_streaming_rules_trial(capsys, argv, message):
    from realtime_framework.evaluation._trial import main

    usage_error(
        capsys,
        lambda: main([*argv, "--task", WEB, "--interaction-mode", "streaming", "--dry-run"]),
        message,
    )


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--model", "my-model", "--fps", "1", "--clock", "wall"], NOT_LIVE),
        (["--model", "gemini-live", "--fps", "1", "--clock", "token", "--tokens-per-frame", "64"],
         STREAMING_CLOCK),
        (["--model", "gemini-live", "--fps", "1", "--clock", "token", "--tokens-per-frame",
          "unlimited"], STREAMING_CLOCK),
        # The runner's default clock too (runner.py main).
        (["--model", "gemini-live", "--fps", "1", "--tokens-per-frame", "unlimited"],
         STREAMING_CLOCK),
        (["--model", "gemini-live", "--fps", "2", "--clock", "wall"], STREAMING_CAP),
    ],
    ids=["not-live", "token-clock", "pause-clock", "default-clock", "frame-rate-cap"],
)
def test_streaming_rules_identity(argv, message):
    from realtime_framework.evaluation.config import identity_main

    fails(
        lambda: identity_main([*argv, "--task", WEB, "--max-minutes", "5", "--protocol",
                               "wall-clock", "--interaction-mode", "streaming"]),
        message,
    )


@pytest.mark.parametrize(
    "key, clock_mode, tokens_per_frame, message",
    [
        ("gpt-realtime-2", "wall", None, "The openai_realtime API does not support streaming"),
        ("gemini-live", "token", 64, STREAMING_CLOCK),
        ("gemini-live", "token", None, STREAMING_CLOCK),
    ],
    ids=["undeclared", "token-clock", "pause-clock"],
)
def test_streaming_rules_run_config(monkeypatch, key, clock_mode, tokens_per_frame, message):
    if key == "gpt-realtime-2":
        undeclared(monkeypatch, "openai_realtime")
    fails(
        lambda: effective_run_config(
            model_config=preset(key), fps=1.0, max_minutes=5.0, protocol="wall-clock",
            think_budget_s=None, per_move_cap_s=None, clock_mode=clock_mode,
            tokens_per_frame=tokens_per_frame, interaction_mode="streaming",
        ),
        message,
    )


def test_streaming_undeclared_plan(monkeypatch, tmp_path):
    undeclared(monkeypatch, "gemini_live")
    fails(
        plan(tmp_path, "models: [gemini-live]\nruntime: {clock_mode: wall, "
                       "interaction_mode: streaming}\n"),
        "The gemini_live API does not support streaming",
    )


# Catalog status and reasoning effort (unchanged entry points)


def test_experimental_cell_runner(capsys):
    usage_error(
        capsys,
        runner_main("--model", "jev-1.13", "--task", WEB, "--clock", "wall"),
        "catalog cell 'jev-1.13' is excluded from maintained benchmark results; "
        "set ALLOW_EXPERIMENTAL_MODEL=1 only for explicitly non-comparable diagnostics",
    )


def test_experimental_cell_trial(capsys):
    from realtime_framework.evaluation._trial import main

    usage_error(
        capsys,
        lambda: main(["--catalog-model", "gpt-6-luna-decisions", "--task", WEB, "--dry-run"]),
        "gpt-6-luna-decisions is not eligible for the maintained benchmark",
    )


def test_reasoning_effort_of_a_choice_api(capsys, tmp_path):
    fails(
        lambda: catalog.validate_reasoning_effort("jev-1.13", "low"),
        "jev-1.13 reasoning_effort must be one of ['fixed'], got 'low'",
    )
    usage_error(
        capsys,
        runner_main("--model", "jev-1.13", "--task", WEB, "--clock", "wall",
                    "--reasoning-effort", "low"),
        "jev-1.13 reasoning_effort must be one of ['fixed'], got 'low'",
    )
    fails(
        plan(tmp_path, "models: [{key: jev-1.13, reasoning_effort: low}]\n"),
        "models[0].reasoning_effort for 'jev-1.13' must be one of ['fixed']",
    )
    models = tmp_path / "models"
    models.mkdir()
    (models / "luna.yaml").write_text(
        "endpoint: {api: openai_decisions, model: openai/x, upstream_provider: openai}\n"
        "reasoning_effort: low\n"
    )
    fails(
        lambda: catalog.load_definitions(models),
        "luna.yaml: reasoning_effort must be one of ['fixed']",
    )


@pytest.mark.parametrize(
    "endpoint, preset",
    [
        ("{api: moonshot, model: kimi-k3}", "kimi-k3"),  # the catalog key's spelling
        ("{api: moonshot, model: kimi-k3}", "[kimi_k3]"),
        # Luna's and JEV's presets belong to their API rows, in their own models' YAMLs too.
        ("{api: openrouter, model: openai/x, upstream_provider: openai}", "luna"),
        ("{api: openai_decisions, model: openai/x, upstream_provider: openai}", "luna"),
        ("{api: jev_decisions, model: typesafe/x}", "jev"),
    ],
    ids=["misspelt", "list", "luna-on-chat", "luna", "jev"],
)
def test_context_preset_of_a_model_yaml(tmp_path, endpoint, preset):
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.yaml").write_text(f"endpoint: {endpoint}\ncontext_preset: {preset}\n")
    fails(
        lambda: catalog.load_definitions(models),
        "m.yaml: context_preset must be one of ['kimi_k3']",
    )


@pytest.mark.parametrize(
    "endpoint, streaming, message",
    [
        (
            "{api: openai_realtime, model: gpt-realtime-2}",
            "[20]",
            "m.yaml: streaming must be a mapping",
        ),
        (
            "{api: openai_realtime, model: gpt-realtime-2}",
            "{compression_trigger_tokens: 65536}",
            (
                "m.yaml: unknown streaming settings ['compression_trigger_tokens']; the "
                "openai_realtime API takes ['retained_frames']"
            ),
        ),
        (
            "{api: openai_realtime, model: gpt-realtime-2}",
            "{retained_frames: 0}",
            "m.yaml: streaming.retained_frames must be a positive integer",
        ),
        (
            "{api: openai_realtime, model: gpt-realtime-2}",
            "{retained_frames: true}",
            "m.yaml: streaming.retained_frames must be a positive integer",
        ),
        (
            "{api: gemini_live, model: gemini-3.8-live}",
            "{compression_trigger_tokens: 32768.0}",
            "m.yaml: streaming.compression_trigger_tokens must be a positive integer",
        ),
        # Against the default target of 32768.
        (
            "{api: gemini_live, model: gemini-3.8-live}",
            "{compression_trigger_tokens: 32768}",
            "m.yaml: streaming.compression_target_tokens must be below compression_trigger_tokens",
        ),
        (
            "{api: moonshot, model: kimi-k3}",
            "{retained_frames: 20}",
            "m.yaml: the moonshot API takes no streaming settings",
        ),
    ],
    ids=["list", "other-api-key", "zero", "bool", "float", "target-not-below", "not-live"],
)
def test_streaming_settings_of_a_model_yaml(tmp_path, endpoint, streaming, message):
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.yaml").write_text(f"endpoint: {endpoint}\nstreaming: {streaming}\n")
    fails(lambda: catalog.load_definitions(models), message)


def test_streaming_settings_compare_the_thresholds_a_driver_has_both_of(tmp_path, monkeypatch):
    """A driver may take one compression threshold and not the other."""
    from realtime_framework.providers.gemini.live import GeminiLiveDriver

    monkeypatch.setattr(
        GeminiLiveDriver, "streaming_settings", {"compression_target_tokens": 32_768}
    )
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.yaml").write_text(
        "endpoint: {api: gemini_live, model: gemini-3.8-live}\n"
        "streaming: {compression_target_tokens: 16384}\n"
    )
    ((_, definition),) = catalog.load_definitions(models)
    assert definition["defaults"]["streaming"] == {"compression_target_tokens": 16_384}


def test_reasoning_effort_of_other_apis():
    fails(
        lambda: catalog.validate_reasoning_effort("gemini-live", "bogus"),
        "gemini-live reasoning_effort must be one of ['provider_default', 'none', 'minimal', "
        "'low', 'medium', 'high', 'xhigh', 'max', 'fixed'], got 'bogus'",
    )


# Driver resolution


def test_unknown_api():
    fails(lambda: catalog.adapter_class("nope"), "Unknown adapter type: 'nope'")
    fails(lambda: episode(ModelConfig(adapter_type="nope")), "Unknown adapter type: 'nope'")
    fails(
        lambda: catalog.validate_connection({"api": "nope", "model": "m"}),
        "endpoint.api must be one of ['litellm', 'chat_completions', 'openrouter', "
        "'anthropic_messages', 'moonshot', 'openai_realtime', 'gemini_live', 'jev_decisions', "
        "'openai_decisions']",
    )


def test_registered_target_is_not_a_class(monkeypatch):
    target = "realtime_framework.providers.litellm:pace_request"
    monkeypatch.setitem(catalog.APIS, "chat_completions", target)
    fails(
        lambda: catalog.adapter_class("chat_completions"),
        f"Registered driver {target!r} is not a class",
        TypeError,
    )


# Construction-time checks that stay in the drivers


def test_decisions_route_is_pinned():
    from realtime_framework.providers.openai_decisions import OpenAIDecisionsClient

    os.environ["OPENROUTER_API_KEY"] = "preflight-test"
    fails(
        lambda: OpenAIDecisionsClient(
            ModelConfig(adapter_type="openai_decisions", model_name="openai/x")
        ),
        "Decisions requires a pinned upstream_provider",
    )


# The declarations the harness and runtime read instead of probing for methods


def test_declared_methods_exist():
    methods = {
        "classifies_transport_errors": "classify_transport_error",
        "connection_renewal": "connection_renewal_reason",
        "request_size_estimation": "request_context_status",
        "maps_output_cap": "transport_output_token_cap",
    }
    for api in catalog.APIS:
        driver = catalog.adapter_class(api)
        capabilities = driver.capabilities
        for flag, method in methods.items():
            if getattr(capabilities, flag):
                assert callable(getattr(driver, method, None)), (api, method)
        assert capabilities.request_size_estimation == callable(
            getattr(driver, "request_context_status", None)
        ), api
        live = api in {"openai_realtime", "gemini_live"}
        assert capabilities.interaction_modes == (
            {"turn_based", "streaming"} if live else {"turn_based"}
        ), api


def test_context_status_needs_its_declaration():
    """The runtime asks request_context_status only of an API that declares
    request_size_estimation (harnesses/default/context.py _run_turn). A driver that has the
    method without the flag, as the method alone once sufficed, is refused at load instead of
    never archiving under context pressure."""
    driver = catalog.adapter_class("chat_completions")
    undeclared = type(
        "Undeclared",
        (driver,),
        {"capabilities": replace(driver.capabilities, request_size_estimation=False)},
    )
    fails(
        lambda: catalog._check_driver("ext.chat:Undeclared", undeclared),
        "'ext.chat:Undeclared' has request_context_status() but does not declare "
        "request_size_estimation",
    )


def test_a_session_declares_no_request_estimate():
    """The runtime archives the history a stateless request carries when it does not fit; a
    persistent session holds its own, so a driver that declares both is refused at load. The
    driver manages its session (SessionLifecycle), as persistent_session requires."""
    driver = catalog.adapter_class("chat_completions")
    session = type(
        "Session",
        (driver, SessionLifecycle),
        {"capabilities": replace(driver.capabilities, persistent_session=True)},
    )
    fails(
        lambda: catalog._check_driver("ext.chat:Session", session),
        "'ext.chat:Session' declares request_size_estimation on a persistent session, whose "
        "context the runtime cannot archive",
    )


# The protocol role each encoder sends the archived-action ledger in, by the ledger's intent role
# (entity/request.py Role): a chat message (providers/litellm.py chat_messages), a Realtime
# conversation item (openai_realtime_session.py) or a Live content (gemini/session.py).
CHAT_ROLES = {"instruction": "system", "input": "user", "model": "assistant"}
LEDGER_SENT_AS = {
    "litellm": CHAT_ROLES,
    "chat_completions": CHAT_ROLES,
    "openrouter": CHAT_ROLES,
    "anthropic_messages": CHAT_ROLES,
    "moonshot": CHAT_ROLES,
    "openai_realtime": CHAT_ROLES,
    "gemini_live": {"input": "user", "model": "model"},
}


def sent_ledger_roles(api, role):
    """The roles the API's encoder sends the ledger in, for a request built under the API's
    policy with that ledger role: the messages the driver hands LiteLLM, or what the session core
    plans."""
    from realtime_framework.entity.actions import Observation
    from realtime_framework.entity.model import (
        CanonicalTurn,
        GenerationRequest,
        ProviderObservation,
    )
    from realtime_framework.entity.request import AnswerRequirements, CallParameters
    from realtime_framework.harnesses.default.request import (
        build_request,
        ledger_block,
        system_block,
    )
    from realtime_framework.providers.gemini.live import GeminiLiveDriver, live_options
    from realtime_framework.providers.gemini.session import SessionCore
    from realtime_framework.providers.litellm import LiteLLMDriver
    from realtime_framework.providers.openai_realtime import OpenAIRealtimeDriver
    from realtime_framework.providers.openai_realtime_session import RealtimeSessionCore

    key = next(key for key in catalog.CATALOG if catalog.get(key).adapter_type == api)
    config = preset(key, api_key="preflight-test")
    policy = current_context_policy().for_path(api, config.provider_options, config.context_preset)
    frame = ProviderObservation(
        observation=Observation(b"raw", 100.0, 0.0, {"observation_id": "o0"}),
        prompt="FRAME",
        image_bytes=b"png",
        mime_type="image/png",
        image_detail="auto",
        geometry={},
    )
    ledger = ledger_block('{"decision": 0}', 1, None)
    request = build_request(
        replace(policy, ledger_role=role),
        system_block("SYSTEM", None),
        (CanonicalTurn(observation=frame, decision=1),),
        ledger,
        AnswerRequirements("text_json", "standard_vlm"),
        CallParameters(config.model_name, None, None),
    )
    text = policy.texts.ledger_header + "\n" + ledger.text
    driver = catalog.adapter_class(api)
    if issubclass(driver, LiteLLMDriver):
        generation = GenerationRequest("turn-1", None, None, "wall", canonical=request)
        messages = driver(config)._request_kwargs(generation)["messages"]
        return [message["role"] for message in messages if message["content"] == text]
    if issubclass(driver, OpenAIRealtimeDriver):
        creates = RealtimeSessionCore(lambda kind: f"rtb_{kind}").replay(request)
        return [create.role for create in creates if create.text == text]
    if issubclass(driver, GeminiLiveDriver):
        seed = SessionCore(live_options(config)).connect(request, None, replacing=False).seed
        contents = [content for batch in seed for content in batch]
        return [content.role for content in contents if content.parts[0].text == text]
    raise AssertionError(f"{api} declares roles but has no ledger rendering here")


@pytest.mark.parametrize(
    "api, role", [(api, role) for api, sent_as in LEDGER_SENT_AS.items() for role in sent_as]
)
def test_ledger_is_sent_in_its_intent_role(api, role):
    """check_context_policy accepts any ledger role the API declares, so the encoder sends the
    ledger in each one."""
    assert sent_ledger_roles(api, role) == [LEDGER_SENT_AS[api][role]]


def test_declared_roles_are_the_encoders_roles():
    """The roles each encoder renders (LEDGER_SENT_AS); Decisions state has none."""
    declaring = {api for api in catalog.APIS if catalog.adapter_class(api).capabilities.roles}
    assert declaring == set(LEDGER_SENT_AS)
    for api, sent_as in LEDGER_SENT_AS.items():
        assert set(catalog.adapter_class(api).capabilities.roles) == set(sent_as), api
    for api in catalog.APIS:
        capabilities = catalog.adapter_class(api).capabilities
        assert capabilities.labels_instead_of_roles == (capabilities.action_channel == "choice")
        assert bool(capabilities.roles) != capabilities.labels_instead_of_roles, api
        assert (capabilities.max_choices is not None) == (capabilities.action_channel == "choice")
        assert capabilities.server_context == (
            "unknown" if capabilities.persistent_session else "none"
        ), api
