"""A driver kept outside the repository, selected by one model YAML.

The package in external_template/ is copied into a temporary directory that is put on sys.path, as
installing it into the run's environment would, and its YAML goes into a temporary models
directory in place of configs/models (providers/registry.py MODELS_DIR). Nothing in the repository
names the driver.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.records import validate_episode_trace
from realtime_framework.evaluation.agent_trace import load_body
from realtime_framework.evaluation.config import (
    configuration_sha256,
    effective_run_config,
    external_driver_identity,
    harness_fingerprint,
)
from realtime_framework.evaluation.harbor_cli import describe_agent
from realtime_framework.providers.preflight import (
    UnsupportedTokenBudget,
    check_episode,
    check_token_budget,
)

from ..fake_clock import FakeClock
from . import baseline
from .build import build_episode
from .fake_env import FakeEnvironment
from .frames import CLOCK_ORIGIN, ReplayTask
from .loop import check_wall_clock_properties, commit_statuses, run_loop
from .scenario import LONG_SCENARIO, TASK_INSTRUCTION, replies
from .steps import run_steps
from .stub import ChatCompletionsStub
from .test_chat_completions import BASELINES, _check_history_changes, _check_rejection_feedback
from .test_runner_loop import LONG_LATENCIES, LONG_MAX_MINUTES

TEMPLATE = Path(__file__).with_name("external_template")
TARGET = "rtb_ext.chat:ExternalChatDriver"
MODEL_KEY = "ext-replay"
KEY_ENV = "REPLAY_EXT_API_KEY"
WEB_TASK = "rt_waitlist_fit"  # a public task, for the Harbor trial worker
FORM = '"package.module:Class"'


@pytest.fixture
def external(tmp_path, monkeypatch):
    """The driver package on sys.path and an empty models directory for its YAML."""
    saved = dict(os.environ)  # the trial worker writes os.environ (evaluation/_trial.py:101-102)
    site = tmp_path / "site"
    shutil.copytree(TEMPLATE, site)
    monkeypatch.syspath_prepend(str(site))
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(catalog, "MODELS_DIR", models)
    monkeypatch.setenv(KEY_ENV, "replay-external-key")
    catalog._catalog_entries.cache_clear()
    yield SimpleNamespace(site=site, models=models)
    catalog._catalog_entries.cache_clear()
    unload()
    os.environ.clear()
    os.environ.update(saved)


def unload():
    """Forget the imported driver modules, as a new interpreter would."""
    for name in [name for name in sys.modules if name.partition(".")[0] in ("rtb_ext", "rtb_one")]:
        del sys.modules[name]


def write_model(models: Path, api: str = TARGET, key: str = MODEL_KEY, **endpoint) -> Path:
    fields = {
        "api": api,
        "model": "replay-vlm",
        "base_url": "http://127.0.0.1:9/v1",
        "api_key_env": KEY_ENV,
        "native_output_limit": 4096,
        **endpoint,
    }
    path = models / f"{key}.yaml"
    path.write_text(yaml.safe_dump({"endpoint": fields, "reasoning_effort": "low"}))
    return path


def fails(call, message, kind=ValueError):
    with pytest.raises(kind) as caught:
        call()
    assert type(caught.value) is kind
    assert str(caught.value) == message


# The acceptance: load, pass the pre-run checks and run a whole episode, the repository unchanged.


def _check_trace(run_dir: Path, stub: ChatCompletionsStub) -> None:
    """The driver's own requests reach the episode trace through the capture hooks."""
    validate_episode_trace(run_dir / "episode_trace.jsonl")
    text = (run_dir / "episode_trace.jsonl").read_text()
    assert "replay-external-key" not in text
    events = [json.loads(line) for line in text.splitlines()]
    closed = events[-1]["payload"]
    assert events[-1]["event"] == "AgentTraceClosed" and closed["error"] is None
    sent = [e["payload"] for e in events if e["event"] == "ModelRequestSent"]
    assert len(sent) == len(stub.requests) == len(LONG_SCENARIO)
    for k, (request, received) in enumerate(zip(sent, stub.requests)):
        assert request["channel"] == "rtb_ext.chat" and request["exact_wire"]
        assert request["scope"]["decision"] == k and request["scope"]["attempt"] == 1
        assert ["authorization", "<redacted>"] in request["headers"]
        assert load_body(run_dir / "episode_blobs.pack", request["body"]) == received.body
    responses = [e["payload"] for e in events if e["event"] == "ModelResponseReceived"]
    assert [r["complete"] for r in responses] == [True] * len(LONG_SCENARIO)
    attempts = [e["payload"]["outcome"] for e in events if e["event"] == "ModelAttemptFinished"]
    assert attempts == ["ok"] * len(LONG_SCENARIO)


def steps(records: list[dict]) -> list[dict]:
    return [record for record in records if record["record"] == "step"]


def _without_detail(messages: list[dict]) -> list[dict]:
    """chat_completions strips the image detail its endpoints reject (providers/litellm.py)."""
    messages = copy.deepcopy(messages)
    for message in messages:
        for part in message["content"] if isinstance(message["content"], list) else ():
            part.get("image_url", {}).pop("detail", None)
    return messages


def test_external_driver_full_episode(external, tmp_path, capsys):
    repository = catalog.PROJECT_ROOT
    fingerprint = harness_fingerprint()
    apis = dict(catalog.APIS)
    presets = sorted(path.name for path in (repository / "configs" / "models").iterdir())

    with ChatCompletionsStub(replies(LONG_SCENARIO)) as stub:
        yaml_path = write_model(external.models, base_url=stub.base_url)
        # Load, as every entry point's first catalog lookup does.
        entry = catalog.get(MODEL_KEY)
        assert entry.adapter_type == TARGET
        agent = {"model_name": MODEL_KEY}
        describe_agent(agent)  # what rtbench run -m ext-replay shows and passes on
        assert agent["model_name"] == "replay/ext-replay"
        assert agent["kwargs"]["catalog_model"] == MODEL_KEY

        # The Harbor trial worker's dry run, which Harbor's resolve hashes.
        settings = trial_settings(capsys, MODEL_KEY)
        assert settings["model_config"]["adapter_type"] == TARGET
        assert settings["credential_variable"] == KEY_ENV

        # The runner's pre-run checks (evaluation/runner.py): effort, finite budget, then the
        # episode rules, which build_episode applies before it builds the agent.
        catalog.validate_reasoning_effort(MODEL_KEY, "low")
        check_token_budget(entry.adapter_type, clock_mode="wall", tokens_per_frame=None)
        config = catalog.to_model_config(MODEL_KEY, fps=1.0, reasoning_effort="low")
        identity = run_config(config)
        assert identity["model_config"]["adapter_type"] == TARGET
        assert identity["context_policy"]["api"] == "chat_completions"
        assert identity["external_driver"] == settings["external_driver"]

        run_dir = tmp_path / "run"
        episode = build_episode(
            config, run_dir, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key=MODEL_KEY
        )
        driver = type(episode.agent.driver)
        assert driver.__module__ == "rtb_ext.chat"
        assert Path(sys.modules[driver.__module__].__file__).is_relative_to(external.site)
        try:
            records = run_steps(episode, LONG_SCENARIO, stub)
        finally:
            invalid_run = episode.finish()

    assert invalid_run is None
    assert stub.unscripted == 0 and stub.unused_replies == 0
    assert all(len(r["requests"]) == 1 for r in records)
    assert {r["requests"][0]["path"] for r in records} == {"/v1/chat/completions"}
    assert all(r["result"]["inspection"]["valid"] for r in records)
    # The first fold (decision 20) and archive (decision 50) reach the driver's requests.
    _check_history_changes(records)
    _check_rejection_feedback(records)
    _check_trace(run_dir, stub)

    header = {"record": "header", "api": TARGET, "scenario": "long", "config": config.model_dump()}
    actual = baseline.Normalizer()([header, *records, *baseline.trajectory(run_dir)])
    # The same agent and context-policy row with a registered encoder on the same replies: what
    # the agent saw, did and was told, and its history, are identical, and so is each request but
    # for the image detail. Only the encoder's own labels differ.
    expected = baseline.read(BASELINES / "chat_completions-nonstream-long.jsonl")
    labels = {"backend_route", "protocol_family", "history_sync"}
    for old, new in zip(steps(expected), steps(actual), strict=True):
        for name in ("observation", "commit", "action_history_entry", "token_usage"):
            assert new[name] == old[name], (new["step"], name)
        for name in ("action", "action_protocol", "raw", "thinking", "usage", "inspection"):
            assert new["result"][name] == old["result"][name], (new["step"], name)
        runtime = [
            {key: value for key, value in record["result"]["realtime_runtime"].items()
             if key not in labels}
            for record in (old, new)
        ]
        assert runtime[1] == runtime[0], new["step"]
        assert _without_detail(new["requests"][0]["body"]["messages"]) == (
            old["requests"][0]["body"]["messages"]
        )

    path = BASELINES / "external_driver-long.jsonl"
    if os.environ.get("REPLAY_RECORD") == "1":
        baseline.write(path, actual)
    else:
        problems = baseline.compare(baseline.read(path), actual)
        assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
            problems[:60]
        )

    # Nothing in the repository changed or names the driver.
    assert harness_fingerprint() == fingerprint
    assert catalog.APIS == apis
    assert sorted(path.name for path in (repository / "configs" / "models").iterdir()) == presets
    assert not yaml_path.is_relative_to(repository)
    assert not external.site.is_relative_to(repository)


def test_external_driver_through_run_stream(external, tmp_path):
    """The same episode through the runner's own decision loop (evaluation/runner.py run_stream)
    on the wall clock, against the fake environment of test_runner_loop.py on its fake clock."""
    clock = FakeClock(wall_origin=CLOCK_ORIGIN)
    environments = []
    with (
        ChatCompletionsStub(
            replies(LONG_SCENARIO),
            on_request=lambda index: environments[0].model_request(index),
            clock=clock,
        ) as stub,
        clock.participant(),
    ):
        write_model(external.models, base_url=stub.base_url)
        config = catalog.to_model_config(MODEL_KEY, fps=1.0, reasoning_effort="low")
        episode = build_episode(
            config,
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=MODEL_KEY,
            clock=clock,
        )
        env = FakeEnvironment(
            episode.frames,
            clock,
            fps=config.fps,
            max_minutes=LONG_MAX_MINUTES,
            reply_latencies=LONG_LATENCIES,
            rejections={k: s.detail for k, s in enumerate(LONG_SCENARIO) if s.status == "rejected"},
        )
        environments.append(env)
        run_loop(episode, env)
    assert stub.unscripted == 0 and stub.unused_replies == 0
    check_wall_clock_properties(env, "time_limit")
    assert commit_statuses(env) == [
        (k, step.status, step.detail) for k, step in enumerate(LONG_SCENARIO)
    ]


# Loading and the pre-run checks read the class's own declarations.


def test_preflight_reads_the_external_declaration(external, capsys):
    write_model(external.models)
    entry = catalog.get(MODEL_KEY)
    assert (entry.adapter_type, entry.family, entry.provider, entry.key_env) == (
        TARGET, "vlm", "replay-external", KEY_ENV,
    )
    assert (entry.native_output_limit, entry.native_output_limit_source) == (4096, "endpoint")
    assert entry.reasoning_efforts == (*catalog.DEFAULT_EFFORTS, "fixed")
    assert entry.benchmark_eligible and entry.comparison_scope == "model"
    assert catalog.vendor(MODEL_KEY) == "replay"
    config = catalog.to_model_config(MODEL_KEY, fps=1.0, reasoning_effort="low")
    assert config.adapter_type == TARGET and config.api_key_env == KEY_ENV

    fails(
        lambda: check_token_budget(TARGET, clock_mode="token", tokens_per_frame=256),
        f"adapter {TARGET!r} cannot report complete usage for a finite token clock",
        UnsupportedTokenBudget,
    )
    fails(
        lambda: check_episode(config, clock_mode="token", task=ReplayTask()),
        f"The {TARGET} API requires a wall clock",
    )
    check_episode(config, clock_mode="wall", task=ReplayTask())

    from realtime_framework.evaluation._trial import main

    with pytest.raises(SystemExit) as caught:
        main(["--catalog-model", MODEL_KEY, "--task", WEB_TASK, "--clock", "token", "--dry-run"])
    assert caught.value.code == 2
    assert capsys.readouterr().err.strip().endswith(f"The {TARGET} API requires a wall clock")


# Refused at load: catalog.get is the first lookup of every entry point.


@pytest.mark.parametrize(
    "api, message",
    [
        (
            "rtb_missing.chat:ExternalChatDriver",
            (
                "'rtb_missing.chat:ExternalChatDriver': cannot import module 'rtb_missing.chat': "
                "ModuleNotFoundError: No module named 'rtb_missing'"
            ),
        ),
        (
            "rtb_ext.chat:Missing",
            "'rtb_ext.chat:Missing': module 'rtb_ext.chat' has no attribute 'Missing'",
        ),
        ("rtb_ext.broken:not_a_class", "'rtb_ext.broken:not_a_class' is not a class"),
        (
            "rtb_ext.broken:NotAnEncoder",
            (
                "'rtb_ext.broken:NotAnEncoder' is not a model encoder: missing connect, "
                "send_observation, start_generation, events, interrupt, sync_history, close, "
                "bind_trace"
            ),
        ),
        # The template declares request_size_estimation: the runtime estimates every request.
        (
            "rtb_ext.broken:NoContextStatus",
            (
                "'rtb_ext.broken:NoContextStatus' declares request_size_estimation but has no "
                "request_context_status()"
            ),
        ),
        (
            "rtb_ext.broken:NoCapabilities",
            (
                "'rtb_ext.broken:NoCapabilities' declares no capabilities "
                "(a RealtimeCapabilities class attribute)"
            ),
        ),
        (
            "rtb_ext.broken:NoDeployment",
            "'rtb_ext.broken:NoDeployment' declares no deployment (a Deployment class attribute)",
        ),
        (
            "rtb_ext.broken:NoModelDefaults",
            "'rtb_ext.broken:NoModelDefaults' declares no model_defaults (a dict class attribute)",
        ),
        (
            "rtb_ext.broken:ClassifierFlagOnly",
            (
                "'rtb_ext.broken:ClassifierFlagOnly' declares classifies_transport_errors but has "
                "no classify_transport_error()"
            ),
        ),
        (
            "rtb_ext.broken:RenewalFlagOnly",
            (
                "'rtb_ext.broken:RenewalFlagOnly' declares connection_renewal but has no "
                "connection_renewal_reason()"
            ),
        ),
        (
            "rtb_ext.broken:OutputCapFlagOnly",
            (
                "'rtb_ext.broken:OutputCapFlagOnly' declares maps_output_cap but has no "
                "transport_output_token_cap()"
            ),
        ),
    ],
    ids=["no-module", "no-attribute", "function", "not-an-encoder", "no-context-status",
         "no-capabilities", "no-deployment", "no-model-defaults", "classifier-flag",
         "renewal-flag", "output-cap-flag"],
)
def test_external_driver_rejected(external, api, message):
    write_model(external.models, api, key="ext-bad")
    fails(lambda: catalog.get("ext-bad"), f"ext-bad.yaml: endpoint.api {message}")
    # The runner's --adapter path (evaluation/runner.py:1548-1554) gets the same message.
    fails(lambda: catalog.adapter_class(api), message)


@pytest.mark.parametrize(
    "api, endpoint, message",
    [
        *[
            (api, {}, f"endpoint.api {api!r} is neither a registered API nor a {FORM} target")
            for api in ("rtb_ext.chat:", "rtb_ext/chat:ExternalChatDriver", "a:b:c")
        ],
        (
            "realtime_framework.providers.litellm:OpenRouterDriver",
            {},
            (
                "endpoint.api 'realtime_framework.providers.litellm:OpenRouterDriver' is a "
                "framework class; use endpoint.api: openrouter"
            ),
        ),
        (
            "realtime_framework.providers.litellm:LiteLLMDriver",
            {},
            (
                "endpoint.api 'realtime_framework.providers.litellm:LiteLLMDriver' is a "
                f"framework class; use one of {list(catalog.APIS)}"
            ),
        ),
        # The rules of registered names stay theirs.
        (TARGET, {"upstream_provider": "x"}, "upstream_provider is only valid for OpenRouter APIs"),
        (
            TARGET,
            {"stream": False},
            "stream/include_usage are transport options for chat_completions only",
        ),
    ],
    ids=["empty-class", "path", "two-colons", "registered", "unregistered", "upstream", "stream"],
)
def test_external_endpoint_rejected(external, api, endpoint, message):
    write_model(external.models, api, key="ext-bad", **endpoint)
    fails(lambda: catalog.get("ext-bad"), message)


def test_registered_drivers_meet_the_contract():
    """External drivers are checked against the contract every registered driver meets."""
    for api, target in catalog.APIS.items():
        catalog._check_driver(target, catalog.adapter_class(api))


def test_context_policy_of_an_external_driver():
    """An external driver gets the chat_completions row, the plain stateless chat values."""
    policy = current_context_policy()
    assert policy.for_path(TARGET, {}) is policy.paths["chat_completions"]


def test_registered_api_needs_its_own_row(monkeypatch):
    """Only an external driver falls back to the chat_completions row. A registered API without
    a row of its own is refused when its policy is resolved: by the run identity Harbor and the
    trial worker compute before a run, and by the agent the runner builds."""
    from realtime_framework.entity.config import ModelConfig
    from realtime_framework.harnesses.registry import create_agent

    policy = current_context_policy()
    assert policy.paths.keys() == catalog.APIS.keys()
    monkeypatch.setitem(catalog.APIS, "my_api", catalog.APIS["chat_completions"])
    monkeypatch.setenv(KEY_ENV, "replay-external-key")
    message = (
        "API 'my_api' has no context-policy row; add one in current_context_policy() "
        "(realtime_framework/entity/context_policy.py)"
    )
    fails(lambda: policy.for_path("my_api", {}), message)
    config = ModelConfig(
        adapter_type="my_api",
        model_name="replay-vlm",
        api_base_url="http://127.0.0.1:9/v1",
        api_key_env=KEY_ENV,
        native_output_limit=4096,
        fps=1.0,
    )
    fails(lambda: run_config(config), message)
    fails(lambda: create_agent(config), message)


# The driver's code is part of the run identity.


def trial_settings(capsys, key: str) -> dict:
    """What the Harbor trial worker resolves for a catalog key (evaluation/_trial.py:221-232)."""
    from realtime_framework.evaluation._trial import main

    assert main(["--catalog-model", key, "--task", WEB_TASK, "--dry-run"]) == 0
    return json.loads(capsys.readouterr().out)


def run_config(config) -> dict:
    return effective_run_config(
        model_config=config, fps=1.0, max_minutes=5.0, protocol="wall-clock",
        think_budget_s=None, per_move_cap_s=None, mode="default",
    )


def test_run_identity_records_the_driver_code(external, tmp_path, monkeypatch, capsys):
    write_model(external.models)
    config = catalog.to_model_config(MODEL_KEY, fps=1.0, reasoning_effort="low")
    identity = run_config(config)["external_driver"]
    assert identity.keys() == {"module", "class", "source_sha256", "context_policy_row"}
    assert (identity["module"], identity["class"]) == ("rtb_ext.chat", "ExternalChatDriver")
    assert identity["context_policy_row"] == "chat_completions"
    settings = trial_settings(capsys, MODEL_KEY)
    assert settings["external_driver"] == identity
    configuration = configuration_sha256(settings)

    # The same sources installed elsewhere: the digest does not depend on the location.
    moved = tmp_path / "elsewhere"
    shutil.copytree(external.site, moved)
    unload()
    monkeypatch.syspath_prepend(str(moved))
    assert run_config(config)["external_driver"] == identity
    assert Path(sys.modules["rtb_ext"].__file__).is_relative_to(moved)

    # An edited source changes the run config and the configuration Harbor checks.
    with (moved / "rtb_ext" / "chat.py").open("a") as source:
        source.write("# edited\n")
    edited = run_config(config)["external_driver"]
    assert edited["source_sha256"] != identity["source_sha256"]
    assert configuration_sha256({**settings, "external_driver": edited}) != configuration

    # So does any other file of the package, such as a prompt the driver reads.
    (moved / "rtb_ext" / "system_prompt.txt").write_text("Act on the page.\n")
    assert run_config(config)["external_driver"]["source_sha256"] != edited["source_sha256"]

    # A driver module outside any package is hashed alone.
    module = moved / "rtb_one.py"
    module.write_text("from rtb_ext.chat import ExternalChatDriver as Driver\n")
    one = external_driver_identity("rtb_one:Driver")
    assert (one["module"], one["class"]) == ("rtb_one", "Driver")
    module.write_text("from rtb_ext.chat import ExternalChatDriver as Driver  # edited\n")
    assert external_driver_identity("rtb_one:Driver")["source_sha256"] != one["source_sha256"]


def test_effective_config_export_hashes_the_driver_code(external, tmp_path, monkeypatch, capsys):
    """The effective-config export reports the configuration hash Harbor's resolve computes for
    the same trial dry run, external driver identity included."""
    from realtime_framework.evaluation import effective_config

    write_model(external.models)
    settings = trial_settings(capsys, MODEL_KEY)
    harbor = resolved_request(monkeypatch, tmp_path, MODEL_KEY)["configuration_sha256"]

    def probe(kind, environ, request, *, scratch, python=sys.executable):
        """The trial probe's record of the cell (effective_config._probe_trial); the runner
        probe adds nothing the hash reads."""
        cell = {}
        if kind == "trial":
            cell = {
                "error": None,
                "settings": settings,
                "runner_argv": [],
                "runner_environ": {},
                "attribution": {},
                "python": python,
            }
        return {"models": {MODEL_KEY: {WEB_TASK: cell}}}

    monkeypatch.setattr(effective_config, "_probe", probe)
    launch = {((), (), sys.executable): [(MODEL_KEY, WEB_TASK, [])]}
    exported = effective_config._launched_entry(launch, env_file=None, scratch=tmp_path)
    assert exported["models"][MODEL_KEY][WEB_TASK]["configuration_sha256"] == harbor
    without_driver = {key: value for key, value in settings.items() if key != "external_driver"}
    assert configuration_sha256(without_driver) != harbor


def test_configuration_hash_of_registered_models(capsys):
    """A registered model hashes only its model config and revision, as before."""
    settings = trial_settings(capsys, "my-model")
    assert "external_driver" not in settings
    configuration = {key: settings[key] for key in ("model_config", "model_revision")}
    assert configuration_sha256(settings) == hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# External results are unofficial: the mark follows the YAML form to the upload check.


def resolved_request(monkeypatch, tmp_path, key: str) -> dict:
    """The request Harbor's environment resolves for a catalog key, running the trial worker's
    dry run in this interpreter (evaluation/harbor.py ApptainerEnvironment.resolve)."""
    from harbor.environments.base import ExecResult

    from realtime_framework.evaluation import harbor
    from realtime_framework.evaluation._trial import main

    images = {kind: tmp_path / f"{kind}.sif" for kind in ("sandbox", "realtime")}
    for image in images.values():
        image.touch()
    suite = {
        "suite_id": "public", "suite_version": "1.0", "suite_hash": "0" * 64,
        "harness_version": "1.0", "harness_hash": "1" * 64,
        "images": {kind: str(path) for kind, path in images.items()},
        "tasks": [{"task_id": WEB_TASK, "category": "realtime"}],
    }
    monkeypatch.setattr(harbor, "runtime", lambda root, python, suite_id: suite)

    async def run(command, **options):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(command[3:])  # after "python -m realtime_framework.evaluation._trial"
        return ExecResult(return_code=code, stdout=out.getvalue(), stderr="")

    environment = object.__new__(harbor.ApptainerEnvironment)
    environment.root, environment.python, environment.lane = catalog.PROJECT_ROOT, "python", 1
    environment.task_id = WEB_TASK
    environment._run = run
    environment._runner_environment = lambda extra_env: {}
    request = {
        "model": key, "mode": "default", "catalog_model": key, "suite": "public", "fps": 1,
        "model_input_fps": 1, "interaction_mode": "turn_based", "duration_minutes": 5.0,
        "seed": 42, "clock": "wall", "stream": True,
    }
    return asyncio.run(environment.resolve(request))


def submit(tmp_path, request: dict):
    """rtbench upload's check of three jobs, one trial each, before any upload."""
    from realtime_framework.evaluation.submission import check_runs

    jobs = []
    for run in (1, 2, 3):
        trial = tmp_path / f"run{run}" / f"{WEB_TASK}__trial"
        (trial / "agent").mkdir(parents=True)
        (trial / "agent" / "request.json").write_text(json.dumps(request))
        jobs.append(trial.parent)
    return check_runs(jobs)


def test_upload_refuses_external_driver_results(external, tmp_path, monkeypatch, capsys):
    from realtime_framework.evaluation.submission import SubmissionError

    write_model(external.models)
    shutil.copy(catalog.PROJECT_ROOT / "configs" / "models" / "my-model.yaml", external.models)
    identity = trial_settings(capsys, MODEL_KEY)["external_driver"]
    request = resolved_request(monkeypatch, tmp_path, MODEL_KEY)
    assert request["external_driver"] == identity
    with pytest.raises(SubmissionError) as caught:
        submit(tmp_path / "external", request)
    assert str(caught.value) == (
        f"{tmp_path / 'external' / 'run1'}: rtb_ext.chat:ExternalChatDriver is an external "
        "driver, so its results are not official; drivers are registered by contributing them "
        "to the repository"
    )

    # The same jobs of a registered model pass (their trial errors count as failures).
    request = resolved_request(monkeypatch, tmp_path, "my-model")
    assert "external_driver" not in request
    reports = submit(tmp_path / "registered", request)
    assert [report["outcomes"]["error"] for report in reports] == [1, 1, 1]
    assert all("external_driver" not in report for report in reports)
