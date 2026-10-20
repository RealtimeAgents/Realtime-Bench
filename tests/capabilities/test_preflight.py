"""Every pre-run check that can reject a configuration, at each entry point that applies it.

One test per condition, asserting the exact message each check produced before the checks were
collected in providers/preflight.py (recorded on 8319a10, unchanged on 9ada3b0). The runner's own
rules run in check_episode (evaluation/runner.py:1560); its argparse checks and those of
identity_main and the Harbor trial worker are exercised through their main functions, which fail
before any container or network is used.
"""

from __future__ import annotations

import os
import sys

import pytest

import realtime_framework.providers.registry as catalog
import realtime_tasks.registry as tasks
from realtime_framework.entity.config import ModelConfig
from realtime_framework.evaluation.config import effective_run_config, load_benchmark
from realtime_framework.providers.preflight import (
    UnsupportedTokenBudget,
    check_episode,
    check_token_budget,
)

GAME, WEB = "gomoku", "rt_waitlist_fit"


@pytest.fixture(autouse=True)
def environment():
    """The entry points write os.environ (runner.py:1402-1403; _trial.py:101-102)."""
    saved = dict(os.environ)
    os.environ.pop("ALLOW_EXPERIMENTAL_MODEL", None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def preset(key, **overrides):
    """What the runner builds for a catalog key (runner.py:1544-1549)."""
    entry = catalog.get(key)
    return catalog.to_model_config(
        key, **{"fps": 1.0, "reasoning_effort": entry.defaults["reasoning_effort"], **overrides}
    )


def episode(model_config, *, clock_mode="wall", task=WEB):
    check_episode(model_config, clock_mode=clock_mode, task=tasks.get(task))


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

    The lookup error is raised as before, not turned into a usage error (runner.py:1449-1456).
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
    # endpoint.native_output_limit (providers/registry.py:147-156).
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
    os.environ["BENCHMARK_CONTEXT_WINDOW_TOKENS"] = "4096"  # providers/registry.py:533-541
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
    ids=["silent-blocking", "3.8-live-effort", "extended-effort", "extended-options",
         "extended-silent"],
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
