"""Every pre-run check that can reject a configuration, at each entry point that applies it.

One test per condition, asserting the exact message each check produced before the checks were
collected in providers/preflight.py (recorded on 8319a10, unchanged on 9ada3b0); the context-policy
checks came later and are pinned as written. The runner's own rules run in check_episode
(evaluation/runner.py:2242); its argparse checks and those of identity_main and the Harbor trial
worker are exercised through their main functions, which fail before any container or network is
used.
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

GAME, WEB = "gomoku", "rt_waitlist_fit"


@pytest.fixture(autouse=True)
def environment():
    """The entry points write os.environ (runner.py:1403-1404; _trial.py:101-102)."""
    saved = dict(os.environ)
    os.environ.pop("ALLOW_EXPERIMENTAL_MODEL", None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def preset(key, **overrides):
    """What the runner builds for a catalog key (runner.py:1548-1553)."""
    entry = catalog.get(key)
    return catalog.to_model_config(
        key, **{"fps": 1.0, "reasoning_effort": entry.defaults["reasoning_effort"], **overrides}
    )


def episode(model_config, *, clock_mode="wall", task=WEB, interaction_mode="turn_based"):
    check_episode(
        model_config, clock_mode=clock_mode, task=tasks.get(task),
        interaction_mode=interaction_mode,
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

    The lookup error is raised as before, not turned into a usage error (runner.py:1453-1460).
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
    # endpoint.native_output_limit (providers/registry.py:157-166).
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
    os.environ["BENCHMARK_CONTEXT_WINDOW_TOKENS"] = "4096"  # providers/registry.py:621-629
    fails(lambda: episode(preset("my-model")), "Context capacity must exceed generation reserve")


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
            gemini(model_name="gemini-3.1-flash-live-preview",
                   provider_options={"tool_response_scheduling": "silent"}),
            "Silent Gemini tool responses require non-blocking tools",
        ),
        (
            gemini(model_name="gemini-3.1-flash-live-preview",
                   provider_options={"tool_behavior": "non_blocking"}),
            "gemini-3.1-flash-live-preview calls functions synchronously only; "
            "use tool_behavior=blocking",
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
    ids=["silent-blocking", "3.1-non-blocking", "3.8-live-effort", "extended-effort",
         "extended-options", "extended-silent"],
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
            {"feedback": "separate_message"},
            "The openai_realtime API cannot place execution feedback as separate_message",
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
    ],
    ids=["live-feedback", "digest-feedback", "ledger-role", "reasoning-unsupported",
         "history-form", "screenshot-input", "text-input"],
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


@pytest.fixture
def streaming_gemini(monkeypatch):
    """Gemini Live declaring native streaming, as it will once its streaming agent exists; no
    driver declares it yet."""
    from dataclasses import replace

    from realtime_framework.providers.gemini.live import GeminiLiveDriver

    capabilities = replace(
        GeminiLiveDriver.capabilities, interaction_modes=frozenset({"turn_based", "streaming"})
    )
    monkeypatch.setattr(GeminiLiveDriver, "capabilities", capabilities)


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


def test_streaming_trial(capsys):
    from realtime_framework.evaluation._trial import main

    usage_error(
        capsys,
        lambda: main(["--catalog-model", "gemini-live", "--task", WEB, "--interaction-mode",
                      "streaming", "--dry-run"]),
        "The gemini_live API does not support streaming",
    )


def test_streaming_identity():
    from realtime_framework.evaluation.config import identity_main

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
def test_streaming_needs_the_driver_to_declare_it(key, api):
    fails(
        lambda: episode(preset(key), interaction_mode="streaming"),
        f"The {api} API does not support streaming",
    )


def test_streaming_needs_the_wall_clock(streaming_gemini):
    fails(
        lambda: episode(preset("gemini-live"), clock_mode="token", interaction_mode="streaming"),
        STREAMING_CLOCK,
    )


@pytest.mark.parametrize("tokens_per_frame", ["64", "unlimited"], ids=["token", "pause"])
def test_streaming_needs_the_wall_clock_plan(streaming_gemini, tmp_path, tokens_per_frame):
    fails(
        plan(tmp_path, "models: [gemini-live]\nruntime: {clock_mode: token, tokens_per_frame: "
                       f"{tokens_per_frame}, interaction_mode: streaming}}\n"),
        STREAMING_CLOCK,
    )


def test_streaming_needs_the_wall_clock_trial(streaming_gemini, capsys):
    from realtime_framework.evaluation._trial import main

    usage_error(
        capsys,
        lambda: main(["--catalog-model", "gemini-live", "--task", WEB, "--clock", "token",
                      "--interaction-mode", "streaming", "--dry-run"]),
        STREAMING_CLOCK,
    )


def test_streaming_frame_rate_cap(streaming_gemini):
    """The run plans and the Harbor trial worker apply the cap to a streaming run too; on a turn-
    based run only the runner does (check_fps)."""
    cap = "gemini_live supports at most 1 fps; requested 2"
    fails(
        lambda: validate_model_interaction(
            "gemini_live", clock_mode="wall", model_input_fps=2.0, interaction_mode="streaming"
        ),
        cap,
    )
    validate_model_interaction("gemini_live", clock_mode="wall", model_input_fps=2.0)
    fails(lambda: episode(preset("gemini-live", fps=2.0), interaction_mode="streaming"), cap)
    episode(preset("gemini-live"), interaction_mode="streaming")


def test_streaming_frame_rate_cap_plan(streaming_gemini, tmp_path):
    """The plan's model-input rate defaults to its environment rate."""
    fails(
        plan(tmp_path, "models: [gemini-live]\nruntime: {clock_mode: wall, fps: 2, "
                       "interaction_mode: streaming}\n"),
        "gemini_live supports at most 1 fps; requested 2",
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
        # Luna's and JEV's presets belong to their API rows, in their own models' YAMLs too.
        ("{api: openrouter, model: openai/x, upstream_provider: openai}", "luna"),
        ("{api: openai_decisions, model: openai/x, upstream_provider: openai}", "luna"),
        ("{api: jev_decisions, model: typesafe/x}", "jev"),
    ],
    ids=["misspelt", "luna-on-chat", "luna", "jev"],
)
def test_context_preset_of_a_model_yaml(tmp_path, endpoint, preset):
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.yaml").write_text(f"endpoint: {endpoint}\ncontext_preset: {preset}\n")
    fails(
        lambda: catalog.load_definitions(models),
        "m.yaml: context_preset must be one of ['kimi_k3']",
    )


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
        assert capabilities.interaction_modes == {"turn_based"}


def test_declared_roles_are_the_encoders_roles():
    """The roles each encoder maps a request intent onto (openai_realtime_session.py ROLES,
    gemini/session.py _ROLES); Decisions state has none."""
    from realtime_framework.providers import openai_realtime_session
    from realtime_framework.providers.gemini import session as gemini_session

    for api, roles in (
        ("openai_realtime", openai_realtime_session.ROLES),
        ("gemini_live", gemini_session._ROLES),
    ):
        assert set(catalog.adapter_class(api).capabilities.roles) == set(roles), api
    for api in catalog.APIS:
        capabilities = catalog.adapter_class(api).capabilities
        assert capabilities.labels_instead_of_roles == (capabilities.action_channel == "choice")
        assert bool(capabilities.roles) != capabilities.labels_instead_of_roles, api
        assert (capabilities.max_choices is not None) == (capabilities.action_channel == "choice")
        assert capabilities.server_context == (
            "unknown" if capabilities.persistent_session else "none"
        ), api
