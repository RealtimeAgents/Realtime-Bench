"""Internal Harbor trial worker: resolve a model, then exec the evaluation runner."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from realtime_framework.entity.config import merge_streaming_settings
from realtime_framework.entity.context_policy import MODEL_PRESETS
from realtime_framework.evaluation.run_spec import ENTRY_DEFAULTS
from realtime_framework.evaluation.run_spec import INTERACTION_MODES
from realtime_framework.evaluation.run_spec import PLAN_OPTIONS
from realtime_framework.evaluation.run_spec import PROTOCOLS
from realtime_framework.evaluation.run_spec import run_spec

ROOT = Path(__file__).resolve().parents[2]


LAUNCH_FILE = "wall_clock_launch.json"


def runner_arguments(result_root: Path) -> list[str]:
    """The runner arguments for the launch file in ``result_root``, whose plan gives the runner
    its run options, model and task, except the episode's output paths."""
    return ["--launch", str(result_root / LAUNCH_FILE), "--report-root", str(result_root)]


def runner_environment(plan) -> dict:
    """The variables the runner's fresh interpreter receives before any module-level defaults are
    loaded: the suite, the desktop's graphics mode and recording encoder, and the lane's."""
    lane = plan["execution"]["lane"]
    offset = lane * 1000
    return {
        "PYTHONPATH": str(ROOT),
        "TASK_SUITE": plan["suite"],
        "SANDBOX_GRAPHICS_MODE": plan["runtime"]["graphics_mode"],
        "SCREEN_RECORDING_ENCODER": plan["runtime"]["recording_encoder"],
        "TASK_PORT_OFFSET": str(offset),
        "SANDBOX_DISPLAY_NUMBER": str(99 + lane),
        "SANDBOX_CONTAINER": f"streaming-agent-sandbox-L{lane}",
        "RT_SUITE_CONTAINER": f"rt-realtime-suite-L{lane}",
        "GUEST_AGENT_PORT": str(6010 + offset),
        "VNC_PORT": str(6011 + offset),
        "NOVNC_PORT": str(6012 + offset),
        "RT_SUITE_PORT": str(6014 + offset),
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--model", help="LiteLLM provider/model-id (no model YAML required)")
    model.add_argument(
        "--catalog-model", help="Existing model/driver key, including realtime models"
    )
    parser.add_argument(
        "--model-revision", help="Immutable screenshot endpoint deployment revision"
    )
    parser.add_argument(
        "--api-base", "--base-url", dest="base_url", help="Optional custom API base URL"
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Optional credential variable override; otherwise LiteLLM uses provider credentials",
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="Use a complete Chat Completions response"
    )
    parser.add_argument(
        "--include-usage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Request terminal usage from streaming endpoints (default: on; "
        "--no-include-usage for endpoints that reject stream_options)",
    )
    parser.add_argument(
        "--context-preset",
        choices=MODEL_PRESETS,
        help="Context-policy preset of the endpoint, as a model YAML's context_preset "
        "(kimi_k3: send Kimi K3 its past reasoning)",
    )
    parser.add_argument("--reasoning-effort", help="Only set if the selected endpoint supports it")
    parser.add_argument(
        "--suite", default="public", help="Task suite NAME or NAME@MAJOR.MINOR (default: public)"
    )
    parser.add_argument("--task", required=True, help="Task ID supplied by Harbor")
    # A run option not given is None, which takes the trial worker's default
    # (evaluation/run_spec.py).
    parser.add_argument(
        "--clock",
        choices=tuple(PROTOCOLS),
        help=f"(default: {ENTRY_DEFAULTS['trial']['clock_mode']})",
    )
    parser.add_argument(
        "--tokens-per-frame", help=f"(default: {ENTRY_DEFAULTS['trial']['tokens_per_frame']})"
    )
    parser.add_argument(
        "--fps", type=float, help="Environment frame rate in frames/second (default: 1)"
    )
    parser.add_argument(
        "--model-input-fps", type=float, help="Frames per second fed to the model (default: --fps)"
    )
    parser.add_argument(
        "--interaction-mode",
        choices=INTERACTION_MODES,
        help="One frame per model generation, or frames streamed to a live model (default: "
        "turn_based)",
    )
    parser.add_argument(
        "--duration-minutes",
        "--max-minutes",
        type=float,
        help="Scored episode horizon in minutes (default: 5)",
    )
    parser.add_argument("--seed", help="Task seed (default: SEED where it is set)")
    parser.add_argument("--lane", type=int, default=0, help="Port/display isolation lane")
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Also atomically write the validated report here (for local Harbor or maintainer evaluation)",
    )
    parser.add_argument(
        "--run-root-output",
        type=Path,
        help="Write the allocated run directory for external artifact collectors",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print the plan; no API calls or files"
    )
    args = parser.parse_args(argv)
    try:
        spec = run_spec(
            "trial",
            clock_mode=args.clock,
            tokens_per_frame=args.tokens_per_frame,
            fps=args.fps,
            model_input_fps=args.model_input_fps,
            interaction_mode=args.interaction_mode,
            duration_minutes=args.duration_minutes,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    from realtime_framework.providers.registry import load_project_secrets

    load_project_secrets()
    os.environ.setdefault("REALTIMEBENCH_PYTHON", sys.executable)
    if args.lane < 0 or 9292 + args.lane * 1000 > 65535:
        parser.error("--lane must be between 0 and 56")
    if args.catalog_model:
        if (
            args.base_url
            or args.no_stream
            or args.include_usage is not None
            or args.api_key_env is not None
        ):
            parser.error(
                "Endpoint options apply to --model; catalog models use their own transport configuration"
            )
        if args.context_preset is not None:
            parser.error("A catalog model's YAML names its context_preset; omit --context-preset")
        key = args.catalog_model
    elif args.model and "/" in args.model:
        from realtime_framework.providers.litellm import validate_endpoint

        if any(c.isspace() for c in args.model) or not all(args.model.split("/", 1)):
            parser.error("Use a LiteLLM model name in provider/model-id format")
        if args.include_usage is not None:
            parser.error("LiteLLM collects response usage automatically; omit --include-usage")
        if args.context_preset is not None:
            parser.error(
                "--context-preset applies to an endpoint (--model DEPLOYMENT_ID --base-url URL), "
                "not a LiteLLM model"
            )
        try:
            base_url = validate_endpoint(args.base_url) if args.base_url else ""
        except ValueError as exc:
            parser.error(str(exc))
        os.environ.update(
            BENCHMARK_LITELLM_MODEL=args.model,
            BENCHMARK_LITELLM_API_BASE=base_url,
            BENCHMARK_LITELLM_KEY_ENV=args.api_key_env or "",
            BENCHMARK_MODEL_REVISION=args.model_revision or "",
        )
        key = "litellm-vlm"
    else:
        from realtime_framework.providers.litellm import validate_endpoint

        deployment = args.model or os.environ.get("BENCHMARK_MODEL")
        if not deployment or any(c in deployment for c in "\r\n\t"):
            parser.error("Supply --model DEPLOYMENT_ID or BENCHMARK_MODEL")
        try:
            base_url = validate_endpoint(args.base_url or os.environ.get("BENCHMARK_BASE_URL"))
        except ValueError as exc:
            parser.error(str(exc))
        os.environ.update(
            BENCHMARK_MODEL=deployment,
            BENCHMARK_MODEL_REVISION=args.model_revision or "",
            BENCHMARK_BASE_URL=base_url,
            STREAM_CHAT_STREAM="0" if args.no_stream else "1",
            STREAM_CHAT_INCLUDE_USAGE="0" if args.include_usage is False else "1",
            BENCHMARK_CONTEXT_WINDOW_TOKENS="",
            BENCHMARK_CONTEXT_COUNTER="estimate",
            BENCHMARK_CONTEXT_IMAGE_TOKENS="4096",
            BENCHMARK_CONTEXT_RESERVE_TOKENS="4096",
            BENCHMARK_CONTEXT_PRESET=args.context_preset or "",
            STREAM_CHAT_API_KEY_ENV=args.api_key_env or "OPENAI_API_KEY",
        )
        if args.api_key_env and not os.environ.get(args.api_key_env):
            parser.error(f"Credential variable {args.api_key_env!r} is empty or unset")
        key = "remote-vlm"

    # Import after resolving environment-backed endpoint/model settings.
    import realtime_framework.evaluation.config as _yaml
    import realtime_framework.providers.registry as catalog
    from realtime_framework.providers.preflight import check_episode
    from realtime_framework.providers.preflight import check_token_budget
    from realtime_framework.providers.preflight import validate_model_interaction

    try:
        entry = catalog.get(key)
        if not entry.benchmark_eligible and os.environ.get("ALLOW_EXPERIMENTAL_MODEL") != "1":
            parser.error(f"{key} is not eligible for the maintained benchmark")
        effort = args.reasoning_effort or entry.defaults["reasoning_effort"]
        catalog.validate_reasoning_effort(key, effort)
        validate_model_interaction(
            entry.adapter_type,
            clock_mode=spec.clock_mode,
            fps=spec.fps,
            model_input_fps=spec.model_input_fps,
            interaction_mode=spec.interaction_mode,
            tokens_per_frame=spec.tokens_per_frame,
        )
        check_token_budget(
            entry.adapter_type, clock_mode=spec.clock_mode, tokens_per_frame=spec.tokens_per_frame
        )
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.suite):
        parser.error("Invalid suite identifier")
    from realtime_tasks.registry import suite_path

    suite_path = suite_path(ROOT, args.suite)
    if not suite_path.is_file():
        parser.error(f"Suite is not installed: {args.suite}")
    suite = _yaml.load_yaml(suite_path)
    all_tasks = [task for category in suite["categories"] for task in category["task_ids"]]
    selected = [args.task]
    if selected[0] not in all_tasks:
        parser.error("Task ID must be a member of the selected suite")
    plan = {
        "benchmark_id": "harbor-trial",
        "suite": args.suite,
        "runtime": {
            **{option: getattr(spec, option) for option in PLAN_OPTIONS},
            "graphics_mode": os.environ.get("SANDBOX_GRAPHICS_MODE", "cpu"),
            "recording_encoder": os.environ.get("SCREEN_RECORDING_ENCODER", "cpu"),
        },
        "execution": {"results_dir": "temp", "lane": args.lane},
        "models": [{"key": key, "reasoning_effort": effort}],
        "tasks": selected,
    }
    if key == "litellm-vlm":
        endpoint = {"api": "litellm", "model": args.model}
        for field, value in (
            ("base_url", args.base_url),
            ("api_key_env", args.api_key_env),
            ("revision", args.model_revision),
        ):
            if value:
                endpoint[field] = value
        plan["models"][0]["endpoint"] = endpoint
    # The runner looks its task up in the suite TASK_SUITE names (realtime_tasks/registry.py).
    os.environ["TASK_SUITE"] = args.suite
    import realtime_tasks.registry as tasks

    try:
        resolved_model = catalog.to_model_config(
            key, fps=spec.model_input_fps, reasoning_effort=effort
        )
        # The runner's pre-run rules, so that Harbor's resolve refuses what the runner would.
        check_episode(
            resolved_model,
            clock_mode=spec.clock_mode,
            task=tasks.get(args.task),
            env_fps=spec.fps,
            interaction_mode=spec.interaction_mode,
        )
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    settings = {
        "plan": plan,
        "model_config": resolved_model.model_dump(exclude={"api_key"}),
        "credential_variable": resolved_model.api_key_env or entry.key_env,
        "model_revision": resolved_model.model_revision,
    }
    if catalog.is_external_driver(entry.adapter_type):
        # The driver's code, part of the configuration hash (evaluation/config.py).
        settings["external_driver"] = _yaml.external_driver_identity(entry.adapter_type)
    if spec.interaction_mode == "streaming":
        # The settings the run records (evaluation/config.py effective_run_config), part of the
        # configuration hash.
        driver = catalog.adapter_class(entry.adapter_type)
        settings["streaming"] = merge_streaming_settings(driver, resolved_model.streaming)
    if args.dry_run:
        print(json.dumps(settings, indent=2))
        return 0

    failures = []
    if not shutil.which("apptainer"):
        failures.append("Apptainer is missing; install it or load your cluster module")
    if not Path(entry.python_env).is_file():
        failures.append(f"Model Python is missing: {entry.python_env}")
    for kind, variable in (("sandbox", "SANDBOX_IMAGE"), ("realtime", "RT_SUITE_IMAGE")):
        path = Path(os.environ.get(variable, str(ROOT / f".cache/apptainer/{kind}.sif")))
        if not path.is_file() or not os.access(path, os.R_OK):
            failures.append(f"Missing/unreadable {kind} image: {path}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(
            "Run `rtbench setup`, or configure prebuilt images.",
            file=sys.stderr,
        )
        return 2
    results_base = Path(plan["execution"]["results_dir"]).expanduser()
    if not results_base.is_absolute():
        results_base = ROOT / results_base
    results_base = results_base.resolve()
    results_base.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        result_root = results_base / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        try:
            result_root.mkdir()
            break
        except FileExistsError:
            if attempt == 4:
                parser.error("Unable to allocate a unique run timestamp; retry shortly")
            time.sleep(1)
    (result_root / LAUNCH_FILE).write_text(json.dumps(settings, indent=2) + "\n")
    if args.run_root_output:
        locator = args.run_root_output.resolve()
        locator.parent.mkdir(parents=True, exist_ok=True)
        locator.write_text(json.dumps({"run_root": str(result_root)}) + "\n")

    env = {**os.environ, **runner_environment(plan)}
    command = [
        entry.python_env,
        "-m",
        "realtime_framework.evaluation.runner",
        *runner_arguments(result_root),
        "--results-dir",
        str(result_root / key),
        "--run-id",
        selected[0].split("__", 1)[-1].replace("/", "_"),
    ]
    if args.report_output:
        command += ["--report-output", str(args.report_output.resolve())]
    print(f"Episode: {entry.model_name}; {selected[0]}; {result_root}", flush=True)
    os.chdir(ROOT)
    os.execve(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
