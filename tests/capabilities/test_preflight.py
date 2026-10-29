"""Every pre-run check that can reject a configuration, at each entry point that applies it.

One test per condition, asserting the exact message each check produced before the checks were
collected in providers/preflight.py (recorded on 8319a10, unchanged on 9ada3b0); the context-policy
checks came later and are pinned as written. The runner's own rules run in check_episode, which
the Harbor trial worker applies with its argparse checks before it writes the launch file the
runner runs (evaluation/_trial.py main); those are exercised through its main function, which
fails before any container or network is used.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import replace

import pytest

import realtime_framework.providers.registry as catalog
import realtime_tasks.registry as tasks
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import (
    DecisionHistory,
    path_policy,
    resolve_context,
)
from realtime_framework.entity.model import SessionEncoder, implements
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
    """The trial worker writes os.environ (evaluation/_trial.py main)."""
    saved = dict(os.environ)
    os.environ.pop("ALLOW_EXPERIMENTAL_MODEL", None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def preset(key, **overrides):
    """What the trial worker resolves for a catalog key and the runner runs (_trial.py main)."""
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


def plan(tmp_path, text):
    path = tmp_path / "plan.yaml"
    path.write_text(text)
    return lambda: load_benchmark(path)


# Wall clock only (check_episode, run plans, effective_run_config, Harbor trial worker)


@pytest.mark.parametrize(
    "key, api",
    [
        ("my-model", "chat_completions"),
        ("litellm-vlm", "litellm"),
        ("jev-1.13", "jev_decisions"),
        ("gpt-6-luna-decisions", "openai_decisions"),
    ],
)
def test_wall_clock_only_episode(key, api):
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


# A model-input rate above the environment rate (check_episode, run plans, effective_run_config,
# Harbor agent options and trial worker)

FASTER_MODEL_INPUT = "The model-input frame rate (2 fps) exceeds the environment frame rate (1 fps)"


def test_faster_model_input_episode():
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


# Finite token clock (trial worker argparse)

FINITE_BUDGET = "adapter 'chat_completions' cannot report complete usage for a finite token clock"


def test_finite_budget():
    """The trial worker applies it after the wall-clock rule (evaluation/_trial.py main), which
    every API that cannot report complete usage also has, so its error is pinned here alone."""
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


# The runner's episode rules (check_episode), which the trial worker applies


def test_frame_rate_cap():
    fails(
        lambda: episode(preset("gemini-live", fps=2.0)),
        "gemini_live supports at most 1 fps; requested 2",
    )


CHOICE_ON_GAME = "Choice Decisions supports Web task evaluation only: gomoku"


def test_text_only_choice_model_on_game():
    fails(lambda: episode(preset("jev-1.13"), task=GAME), CHOICE_ON_GAME)
    episode(preset("jev-1.13"), task=WEB)
    episode(preset("gpt-6-luna-decisions"), task=GAME)


def test_text_only_choice_model_on_game_trial(capsys, monkeypatch):
    """A rule only check_episode applies is the Harbor trial worker's argument error, on its dry
    run too, which Harbor's resolve makes (harbor.py resolve)."""
    from realtime_framework.evaluation._trial import main

    monkeypatch.setenv("ALLOW_EXPERIMENTAL_MODEL", "1")
    usage_error(
        capsys, lambda: main(["--catalog-model", "jev-1.13", "--task", GAME, "--dry-run"]),
        CHOICE_ON_GAME,
    )


def test_output_limit_required():
    # An anthropic_messages YAML without endpoint.native_output_limit (providers/registry.py
    # resolve_output_limit).
    config = ModelConfig(
        adapter_type="anthropic_messages", model_name="claude-unlisted",
        api_key_env="ANTHROPIC_API_KEY", reasoning_effort="low",
    )
    fails(
        lambda: episode(config),
        "The anthropic_messages API requires an output limit on every request; declare "
        "endpoint.native_output_limit in the model YAML",
    )


def test_context_reserve():
    os.environ["BENCHMARK_CONTEXT_WINDOW_TOKENS"] = "4096"  # providers/registry.py to_model_config
    fails(lambda: episode(preset("my-model")), "Context capacity must exceed generation reserve")


def test_context_reserve_below_the_output_limit():
    """With a context window, chat_completions reserves at least the output limit each request
    sends, so a request the window admits leaves room for the reply."""
    os.environ["BENCHMARK_CONTEXT_WINDOW_TOKENS"] = "100000"  # providers/registry.py to_model_config
    os.environ["BENCHMARK_CONTEXT_RESERVE_TOKENS"] = "4096"
    declared = {"native_output_limit": 8192, "native_output_limit_source": "endpoint"}
    config = preset("my-model").model_copy(update=declared)
    fails(
        lambda: episode(config),
        "The context reserve (4096 tokens) is below the output limit each request sends "
        "(8192 tokens); set BENCHMARK_CONTEXT_RESERVE_TOKENS to at least 8192, or lower "
        "endpoint.native_output_limit in the model YAML",
    )
    episode(config.model_copy(update={"context_reserve_tokens": 8192}))
    episode(config.model_copy(update={"context_window_tokens": None}))
    episode(preset("my-model"))


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
        ({"bogus": 1}, "Unknown gemini_live provider_options: ['bogus']"),
        (
            {"tool_behavior": "sometimes"},
            (
                "gemini_live tool_behavior must be one of ['blocking', 'non_blocking'], "
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


@pytest.mark.parametrize(
    "options, message",
    [
        ({"bogus": True}, "Unknown openrouter provider_options: ['bogus']"),
        (
            {"cache_breakpoint": "yes"},
            "openrouter cache_breakpoint must be one of [False, True], got 'yes'",
        ),
    ],
)
def test_openrouter_options(options, message):
    """OpenRouter takes the one option its compat declares, cache_breakpoint, as the presets that
    cache at explicit breakpoints set it."""
    assert preset("claude-opus-5").provider_options == {"cache_breakpoint": True}
    episode(preset("claude-opus-5"))
    fails(lambda: episode(preset("gpt-5.6-sol", provider_options=options)), message)


# Every wire option of Gemini Live, as a model YAML sets them (GeminiLiveDriver.compat).
GEMINI_OPTIONS = {
    "tool_behavior": "non_blocking",
    "completion_signal": "turn_complete",
    "input_transport": "client_content",
    "tool_response_scheduling": "when_idle",
}


@pytest.mark.parametrize(
    "options",
    [
        {},
        *[
            {name: value for name, value in GEMINI_OPTIONS.items() if name != left_out}
            for left_out in ("input_transport",)
        ],
    ],
    ids=["none", "one-left-out"],
)
def test_gemini_model_without_its_options(options):
    """No driver default stands in for a Gemini Live model's wire options: a model that does not
    set each one, a raw model id (which has no YAML) or a YAML without them, is refused."""
    missing = [name for name in GEMINI_OPTIONS if name not in options]
    fails(
        lambda: episode(gemini(model_name="gemini-other", provider_options=options)),
        f"The gemini_live API requires provider_options {missing}; set them in the model YAML",
    )
    episode(gemini(model_name="gemini-other", provider_options=GEMINI_OPTIONS))


def test_gemini_silent_responses_need_non_blocking_tools():
    options = {**GEMINI_OPTIONS, "tool_behavior": "blocking", "tool_response_scheduling": "silent"}
    fails(
        lambda: episode(gemini(model_name="gemini-other", provider_options=options)),
        "gemini_live tool_response_scheduling=silent requires tool_behavior=non_blocking",
    )


EXTENDED = "gemini-3.8-live-extended-thinking"


@pytest.mark.parametrize(
    "key, effort, supported",
    [("gemini-3.8-live", "low", ["provider_default", "fixed"]),
     (EXTENDED, "none", ["low", "medium", "high"])],
    ids=["3.8-live", "extended"],
)
def test_reasoning_efforts_of_a_gemini_model(capsys, tmp_path, key, effort, supported):
    """3.8 Live rejects thinking_config, and 3.8 Extended Thinking takes low, medium or high
    only: each YAML names the efforts its model takes (reasoning_efforts), and every entry point
    holds a run to them before it starts."""
    message = f"{key} reasoning_effort must be one of {supported}, got {effort!r}"
    fails(lambda: catalog.validate_reasoning_effort(key, effort), message)
    from realtime_framework.evaluation._trial import main

    usage_error(
        capsys,
        lambda: main(["--catalog-model", key, "--task", WEB, "--reasoning-effort", effort,
                      "--dry-run"]),
        message,
    )
    fails(
        plan(tmp_path, f"models: [{{key: {key}, reasoning_effort: {effort}}}]\n"),
        f"models[0].reasoning_effort for {key!r} must be one of {supported}",
    )


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_gemini_reasoning_effort(effort):
    fails(
        lambda: episode(preset("gemini-live", reasoning_effort=effort)),
        "gemini_live reasoning_effort must be one of ['fixed', 'none', 'minimal', 'low', "
        f"'medium', 'high'], got {effort!r}",
    )


def test_anthropic_reasoning_effort():
    """The Messages API maps none, fixed and its effort levels (litellm.py anthropic_thinking), not
    every effort the catalog lists for an API that takes one."""
    fails(
        lambda: episode(preset("claude-opus-5-anthropic", reasoning_effort="minimal")),
        "anthropic_messages reasoning_effort must be one of ['fixed', 'none', 'low', 'medium', "
        "'high', 'xhigh', 'max'], got 'minimal'",
    )
    for effort in (None, "fixed", "none", "low", "max"):
        episode(preset("claude-opus-5-anthropic", reasoning_effort=effort))


@pytest.mark.parametrize(
    "config, message",
    [
        # The provider options are checked with the model config, before initialize's fps check.
        (
            gemini(model_name="gemini-other", fps=2.0),
            f"The gemini_live API requires provider_options {list(GEMINI_OPTIONS)}; set them in "
            "the model YAML",
        ),
        # The fps check came from initialize, before the session connect mapped the effort.
        (
            preset("gemini-live", fps=2.0, reasoning_effort="xhigh"),
            "gemini_live supports at most 1 fps; requested 2",
        ),
    ],
    ids=["options", "effort"],
)
def test_episode_rule_order(config, message):
    fails(lambda: episode(config), message)


# The context policy against the encoder's declarations (check_context_policy, the last rule of
# check_episode). No catalog model breaks a rule, so each case changes one item of its policy.


def test_context_policy_fits_every_catalog_model():
    for key in catalog.CATALOG:
        episode(preset(key))


def policy_check(config, **values):
    driver = catalog.adapter_class(config.adapter_type)
    policy = path_policy(config)
    return lambda: check_context_policy(driver, replace(policy, **values), config)


@pytest.mark.parametrize(
    "key, values, message",
    [
        (
            "gpt-realtime-2",
            {"execution_feedback": "digest"},
            "The openai_realtime API cannot place execution feedback as digest",
        ),
        (
            "jev-1.13",
            {"execution_feedback": "separate_message"},
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
            {"tool_call_history": "tool_transcript"},
            "The openrouter API cannot replay past tool calls as tool_transcript",
        ),
        (
            "gpt-realtime-2",
            {"tool_call_history": "native_tools"},
            "The openai_realtime API cannot replay past tool calls as native_tools",
        ),
    ],
    ids=["live-feedback", "digest-feedback", "ledger-role", "reasoning-unsupported",
         "history-form", "screenshot-input", "text-input", "choice-observation",
         "chat-observation", "tool-transcript", "native-tools"],
)
def test_context_policy_item(key, values, message):
    fails(policy_check(preset(key), **values), message)


def test_kimi_k3_reasoning_replay_comes_from_its_model_yaml():
    """Kimi requires K3's past reasoning back: K3's YAML sets reasoning_replay
    (configs/models/kimi-k3.yaml), an approved deviation. Another moonshot model runs under the
    standard values, which send none."""
    k3 = preset("kimi-k3")
    item = k3.context.item("reasoning_replay")
    assert (item.value, item.source, item.category) == (True, "model", "approved")
    assert path_policy(k3).reasoning_replay is True
    episode(k3)
    other = k3.model_copy(
        update={"model_name": "kimi-other", "context": resolve_context("moonshot", "kimi-other")}
    )
    assert path_policy(other).reasoning_replay is False
    episode(other)


def test_reasoning_replay_the_api_cannot_render():
    config = preset("claude-opus-5")
    context = resolve_context(
        config.adapter_type, config.model_name, model_layer={"reasoning_replay": True}
    )
    fails(
        lambda: episode(config.model_copy(update={"context": context})),
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


def test_the_live_apis_are_the_session_encoders():
    """The APIs the message names are the drivers that are session encoders."""
    drivers = {api: catalog.adapter_class(api) for api in catalog.APIS}
    live = {api for api, driver in drivers.items() if issubclass(driver, SessionEncoder)}
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
    based run only check_episode does (check_fps), which the trial worker applies."""
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
    """The trial worker refuses with the reason alone: a Harbor trial reports its stderr, where
    argparse's usage text would bury the reason (harbor.py resolve)."""
    from realtime_framework.evaluation._trial import main

    with pytest.raises(SystemExit) as caught:
        main(["--catalog-model", "gpt-realtime-2", "--task", WEB, "--fps", "2",
              "--interaction-mode", "streaming", "--dry-run"])
    assert caught.value.code == 2
    (line,) = capsys.readouterr().err.strip().splitlines()
    assert line.endswith("error: " + OPENAI_STREAMING_CAP)


# The turn-based deliberation caps, which a streaming episode has no turns for (N41).

@pytest.mark.parametrize(
    "think_budget_s, per_move_cap_s, message",
    [
        (5.0, None, "Streaming does not support --think-budget-s"),
        (None, 2.0, "Streaming does not support --per-move-cap-s"),
        (5.0, 2.0, "Streaming does not support --think-budget-s or --per-move-cap-s"),
    ],
    ids=["think-budget", "per-move-cap", "both"],
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


# Catalog status and reasoning effort


def test_experimental_cell_trial(capsys):
    from realtime_framework.evaluation._trial import main

    usage_error(
        capsys,
        lambda: main(["--catalog-model", "gpt-6-luna-decisions", "--task", WEB, "--dry-run"]),
        "gpt-6-luna-decisions is not eligible for the maintained benchmark",
    )


def test_reasoning_effort_of_a_choice_api(capsys, tmp_path, monkeypatch):
    from realtime_framework.evaluation._trial import main

    fails(
        lambda: catalog.validate_reasoning_effort("jev-1.13", "low"),
        "jev-1.13 reasoning_effort must be one of ['fixed'], got 'low'",
    )
    # The trial worker holds the model to its catalog status first.
    monkeypatch.setenv("ALLOW_EXPERIMENTAL_MODEL", "1")
    usage_error(
        capsys,
        lambda: main(["--catalog-model", "jev-1.13", "--task", WEB, "--reasoning-effort", "low",
                      "--dry-run"]),
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


def test_upstream_provider_of_a_model_yaml(tmp_path, monkeypatch):
    """Whether a model names an upstream provider is its API's deployment's declaration
    (Deployment.requires_upstream_provider). A new JEV model names it in its YAML, as an OpenRouter
    or a Luna model does, and the shipped JEV keeps its driver's; a model of an API that routes to
    no upstream names none."""
    models = tmp_path / "models"
    models.mkdir()
    shutil.copy(catalog.MODELS_DIR / "jev-1.13.yaml", models)
    monkeypatch.setattr(catalog, "MODELS_DIR", models)

    def load(text):
        (models / "m.yaml").write_text(text)
        catalog._catalog_entries.cache_clear()
        return lambda: catalog.get("m")

    try:
        jev = "endpoint: {api: jev_decisions, model: typesafe/jev-2.0, upstream_provider: typesafe}\n"
        assert load(jev)().defaults["upstream_provider"] == "typesafe"
        assert catalog.get("jev-1.13").defaults["upstream_provider"] == "typesafe"
        for api, model in (
            ("jev_decisions", "typesafe/jev-2.0"),
            ("openai_decisions", "openai/x"),
            ("openrouter", "openai/x"),
        ):
            fails(
                load(f"endpoint: {{api: {api}, model: {model}}}\n"),
                f"m.yaml: the {api} API requires endpoint.upstream_provider",
            )
        fails(
            load("endpoint: {api: moonshot, model: kimi-k3, upstream_provider: moonshot}\n"),
            "The moonshot API takes no upstream_provider",
        )
    finally:
        catalog._catalog_entries.cache_clear()


@pytest.mark.parametrize(
    "endpoint, preset",
    [
        ("{api: moonshot, model: kimi-k3}", "kimi_k3"),
        ("{api: openai_decisions, model: openai/x, upstream_provider: openai}", "luna"),
    ],
    ids=["kimi_k3", "luna"],
)
def test_context_preset_of_a_model_yaml(tmp_path, endpoint, preset):
    """A YAML's context_preset, which context settings replaced, is refused with where they
    went."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.yaml").write_text(f"endpoint: {endpoint}\ncontext_preset: {preset}\n")
    fails(
        lambda: catalog.load_definitions(models), f"m.yaml: {catalog.CONTEXT_PRESET_REMOVED}"
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


EFFORTS = [*catalog.DEFAULT_EFFORTS, "fixed"]


@pytest.mark.parametrize(
    "fields, message",
    [
        (
            "endpoint: {api: moonshot, model: kimi-k3}\nprovider_options: {tool_behavior: x}\n",
            "m.yaml: the moonshot API takes no provider_options",
        ),
        (
            "endpoint: {api: gemini_live, model: g}\nprovider_options: [blocking]\n",
            "m.yaml: provider_options must be a mapping",
        ),
        *[
            (
                f"endpoint: {{api: gemini_live, model: g}}\nreasoning_efforts: {efforts}\n",
                f"m.yaml: reasoning_efforts must be distinct values among {EFFORTS}",
            )
            for efforts in ("low", "[]", "[ultra]", "[low, low]", "[[low]]")
        ],
        # The default effort, provider_default, is not among them.
        (
            "endpoint: {api: gemini_live, model: g}\nreasoning_efforts: [fixed]\n",
            "m.yaml: reasoning_effort must be one of ['fixed']",
        ),
    ],
    ids=["options-not-taken", "options-list", "efforts-string", "efforts-empty",
         "efforts-unknown", "efforts-repeated", "efforts-nested", "default-effort-left-out"],
)
def test_options_and_efforts_of_a_model_yaml(tmp_path, fields, message):
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.yaml").write_text(fields)
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


# What the harness and runtime ask of a driver besides its abstract methods


def test_optional_methods_the_registered_drivers_override():
    """The optional methods each registered driver overrides (entity/model.py implements):
    every one classifies its failures, Gemini Live's alone renews its connection, three estimate
    their requests, and all but Gemini Live's and the Decisions APIs' map the output cap."""
    names = (
        "classify_transport_error",
        "connection_renewal_reason",
        "request_context_status",
        "transport_output_token_cap",
    )
    overriding = {
        name: {api for api in catalog.APIS if implements(catalog.adapter_class(api), name)}
        for name in names
    }
    assert overriding == {
        "classify_transport_error": {
            "litellm",
            "chat_completions",
            "openrouter",
            "anthropic_messages",
            "moonshot",
            "openai_realtime",
            "gemini_live",
            "jev_decisions",
            "openai_decisions",
        },
        "connection_renewal_reason": {"gemini_live"},
        "request_context_status": {"chat_completions", "openai_decisions", "jev_decisions"},
        "transport_output_token_cap": {
            "litellm",
            "chat_completions",
            "openrouter",
            "anthropic_messages",
            "moonshot",
            "openai_realtime",
        },
    }


def test_the_live_apis_stream():
    for api in catalog.APIS:
        live = api in {"openai_realtime", "gemini_live"}
        assert catalog.adapter_class(api).capabilities.interaction_modes == (
            {"turn_based", "streaming"} if live else {"turn_based"}
        ), api


def test_a_driver_is_stateless_or_a_session():
    """The runtime manages a SessionEncoder's session and only aborts a StatelessEncoder, so a
    driver that subclasses both is refused at load, though it takes each observation and manages
    its session (SessionLifecycle) as a session encoder does."""
    driver = catalog.adapter_class("chat_completions")
    session = type(
        "Session",
        (driver, SessionLifecycle),
        {"send_observation": lambda self, observation, history: {}},
    )
    fails(
        lambda: catalog._check_driver("ext.chat:Session", session),
        "'ext.chat:Session' subclasses both StatelessEncoder and SessionEncoder",
    )


def test_a_session_declares_no_request_estimate():
    """The runtime archives the history a stateless request carries when it does not fit; a
    session holds its own, so a session encoder with request_context_status is refused at
    load."""
    driver = catalog.adapter_class("gemini_live")
    session = type(
        "Session",
        (driver,),
        {"request_context_status": lambda self, request: {"pressure": False}},
    )
    fails(
        lambda: catalog._check_driver("ext.live:Session", session),
        "'ext.live:Session' has request_context_status() on a session encoder, whose context "
        "the runtime cannot archive",
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
    policy = path_policy(config)
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
        generation = GenerationRequest("turn-1", None, canonical=request)
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
        driver = catalog.adapter_class(api)
        capabilities = driver.capabilities
        assert capabilities.labels_instead_of_roles == (capabilities.action_channel == "choice")
        assert bool(capabilities.roles) != capabilities.labels_instead_of_roles, api
        assert (capabilities.max_choices is not None) == (capabilities.action_channel == "choice")
        assert capabilities.server_context == (
            "unknown" if issubclass(driver, SessionEncoder) else "none"
        ), api
