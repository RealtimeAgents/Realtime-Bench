"""Export the effective configuration a run would use, per entry point, catalog model and task.

    python -m realtime_framework.evaluation.effective_config [--entry E] [--model M] [--task T]

Entry points (each with its own defaults):

- ``runner``: ``python -m realtime_framework.evaluation.runner --model M --task T``.
- ``trial``: ``python -m realtime_framework.evaluation._trial --catalog-model M --task T``, which
  loads ``.env`` and execs the runner (evaluation/_trial.py:99-325).
- ``harbor``: ``rtbench run -m M``; ``harbor:<file>`` adds ``-c <file>`` (a run plan or a Harbor
  job file). Harbor resolves the trial (evaluation/harbor.py:265-534) and starts ``_trial``.
- ``identity``: ``python -m realtime_framework.evaluation.config`` with the runner's defaults; it
  prints the hash the runner should record.

Each process an entry point starts is reproduced in a fresh interpreter with the environment it
would receive, so values read at import are the values that process would use. The runner process
runs its own main() up to the decision loop. Only its outside calls are replaced: containers, the
guest, the browser, the task server, the model driver and the recorder. The replaced driver keeps
the repository .env load it performs today (harnesses/registry.py:10).

The first catalog lookup of every entry point imports LiteLLM (providers/registry.py:133-134),
which loads a .env and fetches LiteLLM's model cost map (litellm/__init__.py:29-30). The probes set
LITELLM_LOCAL_MODEL_COST_MAP=true so nothing leaves the machine; no exported value reads that map.

Only non-secret setting variables are exported. Each value names the layer that set it last:
``shell`` (the environment given to the entry point), ``.env.local`` (Harbor), ``harbor``,
``.env``, ``_trial``, ``runner`` or ``litellm .env``. Machine paths in arguments are replaced with
placeholders; image paths and interpreters stay, as the runtime-config hash records them.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "rtbench-effective-config.v1"
CLOCK_ORIGIN = 1_760_000_000.0  # fixed task origin for the replaced task server

# Non-secret variables the framework or its SDKs read, besides the ones the hash records
# (evaluation/config.py:423-445).
_EXTRA_VARIABLES = frozenset(
    {
        # Lane and child-process settings written by _trial (evaluation/_trial.py:273-291).
        "PYTHONPATH", "TASK_SUITE", "SEED", "CLOCK_MODE", "TOKENS_PER_FRAME", "TASK_PORT_OFFSET",
        "SANDBOX_DISPLAY_NUMBER", "SANDBOX_CONTAINER", "RT_SUITE_CONTAINER", "GUEST_AGENT_PORT",
        "VNC_PORT", "NOVNC_PORT", "RT_SUITE_PORT", "CUDA_VISIBLE_DEVICES",
        # Interpreters, roots and gates.
        "REALTIMEBENCH_PYTHON", "RTBENCH_PYTHON", "RTBENCH_ROOT", "ALLOW_EXPERIMENTAL_MODEL",
        # Model bridges (providers/registry.py:494-504, 612-639).
        "BENCHMARK_MODEL", "BENCHMARK_BASE_URL", "BENCHMARK_MODEL_REVISION",
        "BENCHMARK_LITELLM_MODEL", "BENCHMARK_LITELLM_API_BASE", "BENCHMARK_LITELLM_KEY_ENV",
        "BENCHMARK_CONTEXT_WINDOW_TOKENS", "BENCHMARK_CONTEXT_COUNTER",
        "BENCHMARK_CONTEXT_IMAGE_TOKENS", "BENCHMARK_CONTEXT_RESERVE_TOKENS",
        "STREAM_CHAT_STREAM", "STREAM_CHAT_INCLUDE_USAGE", "STREAM_CHAT_API_KEY_ENV",
        # Read at call time, not recorded (providers/litellm.py:634-637; gemini/live.py:204-212).
        "OPENROUTER_REFERER", "OPENROUTER_TITLE", "GEMINI_API_KEY_START_SLOT",
        "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
        # Whether importing LiteLLM loads a .env (litellm/__init__.py:29-30).
        "LITELLM_MODE", "LITELLM_DEV_ENV_HOT_RELOAD",
    }
)

# Settings the runner process reads once, at import: (group, field, variable, parse, default).
_IMPORT_SETTINGS = (
    ("frame_encoding", "max_dimension", "FRAME_MAX_DIM", int, 768),
    ("frame_encoding", "jpeg_quality", "FRAME_JPEG_QUALITY", int, 85),
    ("frame_encoding", "image_detail", "FRAME_IMAGE_DETAIL", str, "auto"),
    ("runner_timing", "deadline_grace_s", "STREAM_DEADLINE_GRACE_S", float, 2.0),
    ("runner_timing", "abandoned_drain_s", "STREAM_ABANDONED_DRAIN_S", float, 30.0),
    ("runner_timing", "observation_freeze_n", "STREAM_OBSERVATION_FREEZE_N",
     lambda value: max(2, int(value)), 3),
    ("runner_timing", "provider_watchdog_s", "STREAM_PROVIDER_WATCHDOG_S", float, 300.0),
)


def _import_values() -> dict:
    """The values the runner process uses for _IMPORT_SETTINGS, read from the imported modules."""
    import realtime_framework.evaluation.runner as runner
    from realtime_framework.sandbox.observation import capture

    return {
        "frame_encoding": {
            "max_dimension": capture.FRAME_ENCODING.max_dimension,
            "jpeg_quality": capture.FRAME_ENCODING.jpeg_quality,
            "image_detail": capture.FRAME_ENCODING.image_detail,
        },
        "runner_timing": {
            "deadline_grace_s": runner.DEADLINE_GRACE_S,
            "abandoned_drain_s": runner.ABANDONED_DRAIN_S,
            "observation_freeze_n": runner.OBSERVATION_FREEZE_N,
            "provider_watchdog_s": runner.PROVIDER_WATCHDOG_S,
        },
    }


# Probes: each runs in a fresh interpreter (`--probe KIND REQUEST RESULT`).


class _Stop(BaseException):
    """Ends runner main() at its decision loop; main()'s own handlers catch only Exception."""


class _Rejected(Exception):
    """An argument error the entry point would report with parser.error."""


def _reject(parser, message):
    raise _Rejected(message)


def _variables() -> frozenset:
    from realtime_framework.evaluation.config import _RUNTIME_ENV_KEYS

    return _RUNTIME_ENV_KEYS | _EXTRA_VARIABLES


def _snapshot() -> dict:
    variables = _variables()
    return {name: value for name, value in os.environ.items() if name in variables}


def _attribute(attribution: dict, before: dict, layer: str, claimed=frozenset()) -> set:
    """Credit ``layer`` with every setting variable changed since ``before``, except ``claimed``."""
    after = _snapshot()
    changed = {
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    } - set(claimed)
    for name in changed:
        attribution[name] = layer if name in after else None
    return changed


def _environment(attribution: dict) -> dict:
    return {
        name: {"value": value, "set_by": attribution.get(name)}
        for name, value in sorted(_snapshot().items())
    }


@contextlib.contextmanager
def _restored_environ():
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _sha256(text) -> str | None:
    return None if text is None else hashlib.sha256(str(text).encode()).hexdigest()


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _load_env_file(path, attribution: dict) -> set:
    """The repository .env load (providers/registry.py:15-39) of the file the export names."""
    from realtime_framework.providers.registry import load_project_secrets

    before = _snapshot()
    if path:
        load_project_secrets(Path(path))
    return _attribute(attribution, before, ".env")


def _fresh_catalog(state: dict) -> None:
    """Make the next catalog lookup load it as a fresh process does.

    That lookup imports LiteLLM, whose import loads a .env (providers/registry.py:123-137). The
    module stays imported after the first cell, so the load is repeated here for the later cells.
    """
    import realtime_framework.providers.registry as registry

    if not hasattr(registry._litellm_model_map, "cache_clear"):  # already wrapped
        registry._catalog_entries.cache_clear()
        return
    model_map = registry._litellm_model_map

    def litellm_model_map():
        before = _snapshot()
        value = model_map()
        _litellm_env_file(state["attribution"])
        state.setdefault("claimed", set()).update(
            _attribute(state["attribution"], before, "litellm .env")
        )
        return value

    registry._litellm_model_map = litellm_model_map
    registry._catalog_entries.cache_clear()


def _litellm_env_file(attribution: dict) -> set:
    """What importing LiteLLM loads (litellm/__init__.py:29-30): python-dotenv's find_dotenv() walks
    up from the litellm package directory and loads the first .env, without overriding."""
    if os.getenv("LITELLM_MODE", "DEV") != "DEV":
        return set()
    import importlib.util

    import dotenv

    spec = importlib.util.find_spec("litellm")
    if spec is None or spec.origin is None:
        return set()
    start = Path(spec.origin).resolve().parent
    for directory in (start, *start.parents):
        if (directory / ".env").is_file():
            before = _snapshot()
            dotenv.load_dotenv(
                directory / ".env",
                override=os.getenv("LITELLM_DEV_ENV_HOT_RELOAD") == "True",
            )
            return _attribute(attribution, before, "litellm .env")
    return set()


class _Agent:
    """Stands in for the harness create_agent() returns; main() only sets and calls these."""

    driver = None
    token_usage = None

    def __init__(self, config):
        self.config = config

    def bind_tool_runtime(self, runtime):
        pass

    def initialize(self, task_instruction, initial_observation):
        pass

    def discard_pending_observations(self):
        pass

    def close(self):
        pass


class _Guest:
    def __init__(self, *args, **kwargs):
        pass

    def wait_until_ready(self, max_wait=None):
        return True

    def screenshot(self):
        return b""


class _Scene:
    """Stands in for BrowserSceneCapture: a bound, stable first frame (runner.py:2297-2306, in
    set_up_agent)."""

    def __init__(self, guest, task=None, **kwargs):
        pass

    def ensure_page_focus(self):
        return False

    def capture(self, tick_number):
        from realtime_framework.entity.actions import Observation

        interactables = {"scene": {"status": "ok"}, "binding": {"stable": True}}
        return Observation(b"", CLOCK_ORIGIN, tick_number, {"interactables": interactables})

    def close(self):
        pass


class _Task:
    """The registered task with its task-server calls replaced (runner.py:1533-1539 and
    1638-1657 in main, 2258-2274 in set_up_agent)."""

    def __init__(self, task):
        self._task = task

    def __getattr__(self, name):
        from realtime_tasks.types import Task

        value = getattr(self._task, name)
        if name == "provision":
            return lambda: None
        if name == "reset_episode":
            return dict
        if name == "start_episode":
            return lambda: {"clock_origin": CLOCK_ORIGIN, "start_hold": True}
        if name == "activate_clock":
            return lambda: Task.activate_clock(self._task)
        return value


def _probe_runner(request: dict) -> dict:
    """Run runner main() for each (model, task) up to its decision loop."""
    import realtime_framework.evaluation.runner as runner  # first: the runner's import-time reads

    attribution = dict(request["attribution"])
    imported = _snapshot()
    used = _import_values()
    import time

    import realtime_framework.evaluation.reporting as reporting
    import realtime_framework.sandbox.apptainer as apptainer
    import realtime_framework.sandbox.desktop as desktop
    import realtime_tasks.registry as tasks
    from realtime_framework.entity.config import EnvironmentConfig
    from realtime_framework.entity.context_policy import phase1_context_policy

    settings = {}
    for group, field, variable, parse, default in _IMPORT_SETTINGS:
        settings.setdefault(group, {})[field] = {
            "value": used[group][field],
            "variable": variable,
            "set_by": attribution.get(variable) if variable in imported else None,
        }
    settings["environment"] = {
        "environment_config": EnvironmentConfig().model_dump(),
        "sandbox_container": desktop.CONTAINER_NAME,
    }
    process = {
        "environment_at_import": _environment(attribution),
        "settings_at_import": settings,
    }

    argparse.ArgumentParser.error = _reject
    time.sleep = lambda seconds: None
    for name in ("build_image_if_needed", "start_container", "_start_screensaver_keepalive"):
        setattr(desktop, name, lambda *args, **kwargs: None)
    desktop._start_screen_recording = lambda run_dir=None: False
    apptainer.build_image = lambda *args, **kwargs: None
    reporting.finalize_video = lambda run_dir: None
    runner.GuestClient = _Guest
    runner.BrowserSceneCapture = _Scene
    registered = tasks.get
    tasks.get = lambda task_id: _Task(registered(task_id))
    state = {}

    def create_agent(config, log_dir=None, *, trace=None):
        state["claimed"] |= _load_env_file(request.get("env_file"), state["attribution"])
        # The harness reads its context policy once, after building the driver
        # (harnesses/default/harness.py:1003).
        policy = phase1_context_policy().for_path(config.adapter_type, config.provider_options)
        state["settings_at_agent_setup"] = {
            "active_visual_budget": policy.visuals.keep,
            "visual_fold_block_size": policy.visuals.block,
            "detailed_history_turns": policy.turns.keep,
            "detailed_history_drop_block_size": policy.turns.block,
        }
        return _Agent(config)

    def run_stream(adapter, env, **loop):
        _attribute(state["attribution"], state["start"], "runner", claimed=state["claimed"])
        state["loop"] = {
            "run_stream": {key: loop[key] for key in loop if key not in {"log", "trace"}},
            "guest_environment": {"tick_seconds": env.d, "clock_mode": env.clock_mode},
            "clock_prompt": adapter.streaming_note,
            "settings_at_agent_setup": state["settings_at_agent_setup"],
            "settings_at_call": {
                # providers/litellm.py:71, read before each LiteLLM-driver request.
                "min_request_interval_s": float(
                    os.environ.get("STREAM_MIN_REQUEST_INTERVAL", "0")
                ),
            },
            "environment_at_loop": _environment(state["attribution"]),
            "task": {
                "task_id": env.task.task_id,
                "category": env.task.category,
                "seed": env.task.seed,
                "honor_fail": env.honor_fail,
                "task_clock": env.task.clock_snapshot(),
                "system_prompt_sha256": _sha256(adapter.system_prompt_override),
                "instruction_sha256": _sha256(env.task.description()),
            },
        }
        raise _Stop

    runner.create_agent = create_agent
    runner.run_stream = run_stream
    scratch = Path(request["scratch"])
    results = {}
    for index, (model, task, argv) in enumerate(request["runs"]):
        results_dir = scratch / f"run-{index}"
        argv = [results_dir.as_posix() if arg == "<results>" else arg for arg in argv]
        state.clear()
        state.update(attribution=dict(attribution), claimed=set(), start=_snapshot())
        _fresh_catalog(state)
        with _restored_environ():
            sys.argv = ["runner", *argv, "--results-dir", str(results_dir), "--run-id", "run"]
            record = {"error": None}
            try:
                runner.main()
                record["error"] = "main() returned before the decision loop"
            except _Stop:
                pass
            except (_Rejected, SystemExit, Exception) as exc:
                record["error"] = _error(exc)
        run_dir = results_dir / "run"
        summary = run_dir / "run_summary.json"
        if summary.is_file():
            error = json.loads(summary.read_text()).get("error")
            record["error"] = error.strip().splitlines()[-1] if error else record["error"]
        trace = run_dir / "episode_trace.jsonl"
        if trace.is_file():
            started = json.loads(trace.read_text().splitlines()[0])
            record["run"] = {
                key: value for key, value in started["payload"].items() if key != "task_id"
            }
        run_config = run_dir / "run_config.json"
        if run_config.is_file():
            config = json.loads(run_config.read_text())
            for key in ("harness_version", "harness_hash"):
                config.pop(key)  # source identity, not configuration
            record["runtime_config_hash"] = config.pop("runtime_config_hash")
            record["runtime_config"] = config
            record["recorded_vs_used"] = _recorded_vs_used(
                config, settings, state.get("settings_at_agent_setup", {})
            )
        expected = results_dir / "summary.json"
        if expected.is_file():
            (run,) = json.loads(expected.read_text())["runs"]
            record["expected_runtime_config_hash"] = run["runtime_config_hash"]
        record.update(state.get("loop", {}))
        if "loop" in state and "runtime_config" in record:
            record["frame_rates"] = {  # as the task clock and ModelConfig received them
                "env_fps": record["task"]["task_clock"]["fps"],
                "model_input_fps": record["runtime_config"]["model_config"]["fps"],
            }
        results.setdefault(model, {})[task] = record
    return {**process, "models": results}


def _recorded_vs_used(config: dict, settings: dict, agent_setup: dict) -> list[dict]:
    """Settings whose recorded value implies something other than the value used.

    The hash records the import-time variables as found when it is computed
    (evaluation/config.py:678-680, 720), not the values the modules read at import.
    """
    recorded = config["environment_overrides"]
    differences = [
        {"setting": f"context_policy.{field}", "used": used, "recorded": config[field]}
        for field, used in agent_setup.items()
        if config[field] != used
    ]
    for group, field, variable, parse, default in _IMPORT_SETTINGS:
        used = settings[group][field]["value"]
        implied = parse(recorded[variable]) if variable in recorded else default
        extra = {}
        if field == "max_dimension" and config["frame_max_dimension"] != used:
            extra["frame_max_dimension"] = config["frame_max_dimension"]
        if implied != used or extra:
            differences.append(
                {
                    "setting": f"{group}.{field}",
                    "variable": variable,
                    "used": used,
                    "recorded": recorded.get(variable),
                    **extra,
                }
            )
    return differences


def _probe_trial(request: dict) -> dict:
    """_trial --dry-run for each (model, task), and the runner command and environment it execs."""
    import realtime_framework.evaluation._trial as trial
    import realtime_framework.providers.registry as registry

    argparse.ArgumentParser.error = _reject
    load = registry.load_project_secrets
    state = {}

    def load_project_secrets(path=None):  # called by _trial.main (evaluation/_trial.py:99-101)
        before = _snapshot()
        if request.get("env_file"):
            load(Path(request["env_file"]))
        state["loaded"] |= _attribute(state["attribution"], before, ".env")

    registry.load_project_secrets = load_project_secrets
    results = {}
    for model, task, argv in request["runs"]:
        state.update(attribution=dict(request["attribution"]), loaded=set(), claimed=set())
        _fresh_catalog(state)
        record = {"error": None}
        with _restored_environ():
            before = _snapshot()
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    trial.main([*argv, "--dry-run"])
            except (_Rejected, SystemExit, Exception) as exc:
                record["error"] = _error(exc)
            if record["error"] is None:
                settings = json.loads(output.getvalue())
                record["settings"] = settings
                record["runner_argv"], record["python"] = _trial_command(settings["plan"])
                claimed = state["loaded"] | state["claimed"]
                _attribute(state["attribution"], before, "_trial", claimed=claimed)
                record["runner_environ"] = dict(os.environ)
                record["attribution"] = state["attribution"]
        results.setdefault(model, {})[task] = record
    return {"models": results}


def _trial_command(plan: dict) -> tuple[list[str], str]:
    """The runner argv and interpreter _trial execs, after writing the runner's environment into
    os.environ (evaluation/_trial.py:272-325)."""
    import realtime_framework.providers.registry as catalog

    runtime = plan["runtime"]
    lane = plan["execution"]["lane"]
    offset = lane * 1000
    tokens_per_frame = runtime["tokens_per_frame"]
    os.environ.update(
        {
            "PYTHONPATH": str(ROOT),
            "TASK_SUITE": plan["suite"],
            "SEED": str(runtime["seed"]),
            "CLOCK_MODE": runtime["clock_mode"],
            "TOKENS_PER_FRAME": (
                str(tokens_per_frame) if tokens_per_frame is not None else "unlimited"
            ),
            "SANDBOX_GRAPHICS_MODE": runtime["graphics_mode"],
            "SCREEN_RECORDING_ENCODER": runtime["recording_encoder"],
            "TASK_PORT_OFFSET": str(offset),
            "SANDBOX_DISPLAY_NUMBER": str(99 + lane),
            "SANDBOX_CONTAINER": f"streaming-agent-sandbox-L{lane}",
            "RT_SUITE_CONTAINER": f"rt-realtime-suite-L{lane}",
            "GUEST_AGENT_PORT": str(6010 + offset),
            "VNC_PORT": str(6011 + offset),
            "NOVNC_PORT": str(6012 + offset),
            "RT_SUITE_PORT": str(6014 + offset),
        }
    )
    (model,) = plan["models"]
    (task,) = plan["tasks"]
    argv = [
        "--model", model["key"],
        "--task", task,
        "--clock", runtime["clock_mode"],
        "--tokens-per-frame", os.environ["TOKENS_PER_FRAME"],
        "--fps", str(runtime["fps"]),
        "--model-input-fps", str(runtime["model_input_fps"]),
        "--max-minutes", str(runtime["duration_minutes"]),
        "--seed", str(runtime["seed"]),
        "--reasoning-effort", model["reasoning_effort"],
        "--protocol", runtime["protocol"],
        "--report-root", "<results>",
    ]  # --results-dir and --run-id only name output paths
    return argv, catalog.get(model["key"]).python_env


def _probe_harbor(request: dict) -> dict:
    """``rtbench run -m M [-c FILE]``: the Harbor request and the _trial command each trial runs."""
    import types

    import dotenv
    import yaml

    import realtime_framework.evaluation.harbor_cli as harbor_cli
    from realtime_framework.evaluation.harbor import ApptainerEnvironment
    from realtime_framework.evaluation.harbor import RealtimeBenchAgent
    from realtime_framework.evaluation.harbor import runtime

    # One process serves every cell, as one rtbench process serves its job: the catalog, and the
    # LiteLLM .env its first lookup loads, are read once.
    state = {"attribution": dict(request["attribution"])}
    attribution = state["attribution"]
    _fresh_catalog(state)
    env_local = Path.cwd() / ".env.local"
    scratch = Path(request["scratch"])
    # suite_dataset links task folders under temp/; the job only needs the selected names.
    harbor_cli.suite_dataset = lambda suite, selected: {"path": suite, "task_names": selected}
    harbor_cli.tempfile = types.SimpleNamespace(
        NamedTemporaryFile=lambda *args, **kwargs: tempfile.NamedTemporaryFile(
            *args, **{**kwargs, "dir": scratch}
        )
    )
    metadata = {}
    results = {}
    for model, task in request["runs"]:
        record = {"error": None}
        try:
            arguments = ["-m", model, *(["-c", request["config"]] if request["config"] else [])]
            job = yaml.safe_load(Path(harbor_cli.run_arguments(arguments)[1]).read_text())
            if env_local is not None:
                # Harbor's CLI loads it after rtbench rewrote the job (harbor/cli/jobs.py:1436-1437;
                # evaluation/harbor_cli.py:414-415).
                before = _snapshot()
                if env_local.exists():
                    dotenv.load_dotenv(env_local, override=False)
                _attribute(attribution, before, ".env.local")
                env_local = None
            (agent,) = job["agents"]
            options = RealtimeBenchAgent.parse_options(dict(agent.get("kwargs") or {}), {})
            # RealtimeBenchAgent.run (evaluation/harbor.py:267-273)
            name = agent["model_name"]
            harbor_request = {"model": name, "mode": "default", **options.model_dump()}
            if not (harbor_request["catalog_model"] or options.base_url or "/" in name):
                harbor_request["catalog_model"] = name
            # ApptainerEnvironment.__init__ and resolve (evaluation/harbor.py:323-326, 490-518)
            environment = types.SimpleNamespace(
                root=Path(os.environ.get("RTBENCH_ROOT", ".")).resolve(),
                python=os.environ.get("RTBENCH_PYTHON") or sys.executable,
                lane=int(job["environment"].get("kwargs", {}).get("lane", 1)),
                _merge_env=lambda env: None,
            )
            suite = harbor_request["suite"]
            if suite not in metadata:
                metadata[suite] = runtime(environment.root, environment.python, suite)
            described = metadata[suite]
            categories = {entry["task_id"]: entry["category"] for entry in described["tasks"]}
            environment.images = {kind: Path(path) for kind, path in described["images"].items()}
            environment.task = {
                "suite_id": described["suite_id"],
                "suite_version": described["suite_version"],
                "suite_hash": described["suite_hash"],
                "task_id": task,
                "category": categories[task],
                "fps": harbor_request["fps"],
                "model_input_fps": harbor_request["model_input_fps"],
                "duration_minutes": harbor_request["duration_minutes"],
                "seed": harbor_request["seed"],
            }
            record["harbor_request"] = harbor_request
            record["task_json"] = environment.task
            record["override_timeout_sec"] = agent.get("override_timeout_sec")
            record["trial_argv"] = ApptainerEnvironment._runner_command(
                environment, {**harbor_request, "task": environment.task}
            )[3:]
            record["trial_environ"] = ApptainerEnvironment._runner_environment(environment, None)
            record["trial_python"] = environment.python
        except (_Rejected, SystemExit, Exception) as exc:
            record["error"] = _error(exc)
        results.setdefault(model, {})[task] = record
    return {"models": results, "attribution": attribution}


def _probe_identity(request: dict) -> dict:
    """The hash identity_main prints for the runner's defaults (evaluation/config.py:794-862)."""
    from realtime_framework.evaluation.config import identity_main

    argparse.ArgumentParser.error = _reject
    state = {}
    results = {}
    for model, task, argv in request["runs"]:
        state.update(attribution={})
        _fresh_catalog(state)
        record = {"error": None}
        with _restored_environ():
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    identity_main(argv)
                record["runtime_config_hash"] = output.getvalue().strip()
            except (_Rejected, SystemExit, Exception) as exc:
                record["error"] = _error(exc)
        results.setdefault(model, {})[task] = record
    return {"models": results}


_PROBES = {
    "runner": _probe_runner,
    "trial": _probe_trial,
    "harbor": _probe_harbor,
    "identity": _probe_identity,
}


# Export


def _probe(kind: str, environ: dict, request: dict, *, scratch: Path, python=sys.executable):
    """Run one probe in a fresh interpreter started like the entry point's own process."""
    directory = Path(tempfile.mkdtemp(prefix=f"{kind}-", dir=scratch))
    request = {**request, "scratch": str(directory)}
    (directory / "request.json").write_text(json.dumps(request))
    completed = subprocess.run(
        [str(python), "-B", "-m", "realtime_framework.evaluation.effective_config",
         "--probe", kind, str(directory / "request.json"), str(directory / "result.json")],
        cwd=ROOT,
        env={**environ, "LITELLM_LOCAL_MODEL_COST_MAP": "true"},  # see the module docstring
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"{kind} probe failed:\n{completed.stderr[-4000:]}")
    return json.loads((directory / "result.json").read_text())


def catalog_models() -> list[str]:
    from realtime_framework.providers.registry import MODELS_DIR

    return sorted(path.stem for path in MODELS_DIR.glob("*.yaml"))


def suite_tasks(suite: str = "public") -> list[str]:
    import yaml

    manifest = yaml.safe_load((ROOT / "realtime_tasks" / "sets" / f"{suite}.yaml").read_text())
    return [task for group in manifest["categories"] for task in group["task_ids"]]


def default_entries() -> list[str]:
    files = sorted((ROOT / "configs" / "runs").glob("*.yaml"))
    files += sorted((ROOT / "configs" / "jobs").glob("*.yaml"))
    return [
        "runner",
        "trial",
        "harbor",
        *(f"harbor:{path.relative_to(ROOT).as_posix()}" for path in files),
        "identity",
    ]


def _cells(models, tasks):
    return [(model, task) for model in models for task in tasks]


def _runner_entry(environ, attribution, runs, *, env_file, scratch, python=sys.executable):
    return _probe(
        "runner",
        environ,
        {"runs": runs, "env_file": env_file, "attribution": attribution},
        scratch=scratch,
        python=python,
    )


def _launched_entry(launch: dict, *, env_file, scratch) -> dict:
    """Run _trial for each cell of ``launch``, then the runner it would exec."""
    from realtime_framework.evaluation.config import configuration_sha256

    models = {}
    runs_by_environment = {}
    for (environ, attribution, python), runs in launch.items():
        trial = _probe(
            "trial",
            dict(environ),
            {"runs": runs, "env_file": env_file, "attribution": dict(attribution)},
            scratch=scratch,
            python=python,
        )
        for model, tasks in trial["models"].items():
            for task, record in tasks.items():
                cell = models.setdefault(model, {}).setdefault(task, {})
                cell["trial"] = {key: record.get(key) for key in ("error", "settings")}
                if record["error"] is not None:
                    cell["error"] = record["error"]
                    continue
                cell["runner_argv"] = record["runner_argv"]
                # Harbor hashes the same dry-run output (evaluation/harbor.py resolve).
                cell["configuration_sha256"] = configuration_sha256(record["settings"])
                key = (
                    tuple(sorted(record["runner_environ"].items())),
                    tuple(sorted((k, v) for k, v in record["attribution"].items() if v)),
                    record["python"],
                )
                runs_by_environment.setdefault(key, []).append(
                    (model, task, record["runner_argv"])
                )
    process = {}
    for (environ, attribution, python), runs in runs_by_environment.items():
        result = _runner_entry(
            dict(environ), dict(attribution), runs, env_file=env_file, scratch=scratch,
            python=python,
        )
        if process and {k: v for k, v in result.items() if k != "models"} != process:
            raise RuntimeError("cells of one entry point reached runners with different settings")
        process = {k: v for k, v in result.items() if k != "models"}
        for model, tasks in result["models"].items():
            for task, record in tasks.items():
                models[model][task].update(record)
    return {**process, "models": models}


def _entry(entry: str, *, environ, env_file, models, tasks, scratch) -> dict:
    shell = {name: "shell" for name in environ}
    if entry == "identity":
        runs = [
            (model, task, ["--model", model, "--task", task, "--fps", "1.0",
                           "--max-minutes", "5.0", "--protocol", "time-aware-token-v1"])
            for model, task in _cells(models, tasks)
        ]
        return _probe("identity", environ, {"runs": runs}, scratch=scratch)
    if entry == "runner":
        runs = [(model, task, ["--model", model, "--task", task])
                for model, task in _cells(models, tasks)]
        return _runner_entry(environ, shell, runs, env_file=env_file, scratch=scratch)
    if entry == "trial":
        launch = {
            (tuple(sorted(environ.items())), tuple(sorted(shell.items())), sys.executable): [
                (model, task, ["--catalog-model", model, "--task", task])
                for model, task in _cells(models, tasks)
            ]
        }
        return _launched_entry(launch, env_file=env_file, scratch=scratch)
    config = entry.partition(":")[2] or None
    harbor = _probe(
        "harbor",
        environ,
        {"runs": _cells(models, tasks), "config": config, "attribution": shell},
        scratch=scratch,
    )
    launch = {}
    failed = {}
    for model, cells in harbor["models"].items():
        for task, record in cells.items():
            if record["error"] is not None:
                failed.setdefault(model, {})[task] = {"error": record["error"], "harbor": record}
                continue
            environ_ = record["trial_environ"]
            attribution = dict(harbor["attribution"])
            for name in ("PYTHONPATH", "SANDBOX_IMAGE", "RT_SUITE_IMAGE"):
                attribution[name] = "harbor"
            key = (
                tuple(sorted(environ_.items())),
                tuple(sorted((k, v) for k, v in attribution.items() if v)),
                record["trial_python"],
            )
            launch.setdefault(key, []).append((model, task, record["trial_argv"]))
    result = _launched_entry(launch, env_file=env_file, scratch=scratch)
    for model, cells in harbor["models"].items():
        for task, record in cells.items():
            if record["error"] is None:
                result["models"][model][task]["harbor"] = {
                    key: record[key]
                    for key in ("harbor_request", "task_json", "override_timeout_sec", "trial_argv")
                }
    for model, cells in failed.items():
        result["models"].setdefault(model, {}).update(cells)
    return result


_MISSING = "<missing>"


def _difference(value, base):
    if isinstance(value, dict) and isinstance(base, dict):
        keys = set(value) | set(base)
        difference = {
            key: _difference(value.get(key, _MISSING), base.get(key, _MISSING))
            for key in sorted(keys)
            if value.get(key, _MISSING) != base.get(key, _MISSING)
        }
        return difference
    return value


def _fold(entry: dict) -> dict:
    """Keep each model's first task whole and only the differences for its other tasks."""
    models = {}
    for model, cells in entry["models"].items():
        (first, base), *rest = cells.items()
        models[model] = {
            "first_task": first,
            **base,
            "tasks": {task: _difference(record, base) for task, record in rest},
        }
    return {**entry, "models": models}


def _redact_paths(value, replacements):
    if isinstance(value, dict):
        return {key: _redact_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_paths(item, replacements) for item in value]
    if isinstance(value, str):
        for path, placeholder in replacements:
            value = value.replace(path, placeholder)
    return value


def export(
    *,
    environ: dict | None = None,
    env_file: str | Path | None = None,
    entries: list[str] | None = None,
    models: list[str] | None = None,
    tasks: list[str] | None = None,
    jobs: int = 4,
) -> dict:
    """Return the effective configuration of every entry point, catalog model and task.

    ``environ`` is the environment each entry point is started with (default: this process's).
    ``env_file`` is the repository ``.env`` the entry points load (default: ``<repo>/.env``).
    """
    from realtime_framework.providers.registry import DEFAULT_SECRETS_FILE

    environ = dict(os.environ if environ is None else environ)
    env_file = Path(env_file) if env_file is not None else DEFAULT_SECRETS_FILE
    env_file = str(env_file) if env_file.is_file() else None
    entries = entries or default_entries()
    models = models or catalog_models()
    tasks = tasks or suite_tasks()
    with tempfile.TemporaryDirectory(prefix="rtbench-effective-config-") as scratch:
        scratch = Path(scratch)
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = {
                entry: pool.submit(
                    _entry,
                    entry,
                    environ=environ,
                    env_file=env_file,
                    models=models,
                    tasks=tasks,
                    scratch=scratch,
                )
                for entry in entries
            }
            results = {entry: _fold(future.result()) for entry, future in futures.items()}
        results = _redact_paths(results, [(str(scratch), "<scratch>")])
    return {
        "schema": SCHEMA,
        "env_file": env_file,
        "models": models,
        "tasks": tasks,
        "entries": results,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entry", action="append", help="entry point (default: all)")
    parser.add_argument("--model", action="append", help="catalog key (default: all)")
    parser.add_argument("--task", action="append", help="task id (default: the public suite)")
    parser.add_argument("--env-file", help="the repository .env (default: <repo>/.env)")
    parser.add_argument("--output", type=Path, help="write the JSON here instead of stdout")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--probe", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe:
        kind, request, result = args.probe
        payload = _PROBES[kind](json.loads(Path(request).read_text()))
        Path(result).write_text(json.dumps(payload, default=str))
        return 0
    exported = export(
        env_file=args.env_file,
        entries=args.entry,
        models=args.model,
        tasks=args.task,
        jobs=args.jobs,
    )
    text = json.dumps(exported, indent=1, sort_keys=True, default=str) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
