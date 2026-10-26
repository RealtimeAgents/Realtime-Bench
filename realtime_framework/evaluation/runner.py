from __future__ import annotations

# Loop
"""The decision loop main() runs: run_stream drives one episode on the event kernel's shell
(evaluation/shell.py) with the run's frame rates, horizon and timing, or a native-streaming
episode on the streaming shell (evaluation/streaming_shell.py). The turn loop it replaced is kept
only for the side-by-side runs (tests/differential/legacy_loop.py)."""
import traceback as traceback

from realtime_framework.clock import SystemClock
from realtime_framework.clock import episode_clocks
from realtime_framework.evaluation.config import DEFAULT_MODE
from realtime_framework.evaluation.kernel.streaming import StreamingConfig
from realtime_framework.evaluation.kernel.turns import TurnConfig
from realtime_framework.evaluation.shell import run_episode
from realtime_framework.evaluation.streaming_shell import run_streaming_episode

from realtime_framework.entity.config import RunnerTimingConfig
from realtime_framework.entity.records import TraceEvent

RUNNER_TIMING = RunnerTimingConfig.from_environ()  # read once, at import


def run_stream(
    adapter,
    env,
    *,
    clock_mode="token",
    duration_s=60.0,
    env_fps=1.0,
    input_fps=1.0,
    tokens_per_frame=None,
    think_budget_s=None,
    per_move_cap_s=None,
    log=print,
    trace=None,
    agent_trace=None,
    clocks=None,
    inputs_log=None,
    interaction_mode="turn_based",
):
    """Run one episode on the event kernel's shell (shell.run_episode) and return its timeline.
    The horizon in world seconds and the two frame rates are main()'s own values: the kernel's
    frame arithmetic takes the rates, which tick lengths would give back only to within an ulp.
    ``agent_trace`` is the agent's AgentTraceSink on ``trace``, whose records the kernel numbers
    during the episode. ``clocks`` are the episode's (realtime_framework/clock.py), real ones when
    omitted. ``inputs_log`` is where the shell records what the kernel was given, to replay it.

    A streaming episode (``interaction_mode`` streaming, on the native-streaming agent) runs on
    the streaming shell instead, which takes neither the token clock nor deliberation caps, writes
    the agent's records as they come rather than through the kernel, and keeps no inputs log. It
    has no turns, so its timeline is empty; its run summary's streaming section stands in."""
    if interaction_mode == "streaming":
        streaming = StreamingConfig(
            duration_s, env_fps, input_fps, timing=RUNNER_TIMING, clock_mode=clock_mode
        )
        run_streaming_episode(
            adapter,
            env,
            streaming,
            clocks=clocks or episode_clocks(clock_mode, env_fps),
            log=log,
            trace=trace,
        )
        return []
    config = TurnConfig(
        clock_mode,
        duration_s,
        env_fps,
        input_fps,
        tokens_per_frame,
        think_budget_s,
        per_move_cap_s,
        timing=RUNNER_TIMING,
    )
    return run_episode(
        adapter,
        env,
        config,
        clocks=clocks or episode_clocks(clock_mode, env_fps),
        log=log,
        trace=trace,
        agent_trace=agent_trace,
        inputs_log=inputs_log,
    )


# Episode runner
"""End-to-end runner for persistent-session adapters on the real desktop container.

Token-clock decisions are charged by output-token consumption and wall-clock decisions by latency;
the world advances by that delay before the proposed action can land. Desktop lifecycle and task
scoring use their shared owners.

Usage: python -m realtime_framework.evaluation.runner --model gpt-realtime-2 --task <task-id>
       [--fps 1] [--max-minutes 5] --run-id NAME --results-dir DIR
"""
import argparse as argparse
import json as json
import logging as logging
import os as os
import sys as sys
from datetime import datetime as datetime
from pathlib import Path as Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import realtime_framework.sandbox.desktop as R
import realtime_framework.providers.registry as catalog
from realtime_framework.harnesses.registry import configure_agent
from realtime_framework.harnesses.registry import create_agent
from realtime_framework.entity.config import DEFAULT_TOKENS_PER_FRAME
from realtime_framework.entity.config import EnvironmentConfig
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.providers.preflight import UnsupportedTokenBudget
from realtime_framework.providers.preflight import check_episode
from realtime_framework.providers.preflight import check_streaming_options
from realtime_framework.providers.preflight import check_token_budget
from realtime_framework.sandbox.observation.capture import BrowserSceneCapture
from realtime_framework.sandbox.guest import GuestClient
from realtime_framework.sandbox.actions import GuestEnvironment
from realtime_framework.evaluation.reporting import capture_generation_artifact
from realtime_framework.evaluation.reporting import invalid_evaluation
from realtime_framework.evaluation.reporting import resolve_run_identity
from realtime_framework.evaluation.reporting import save_evaluation
from realtime_framework.evaluation.reporting import summarize_cost
from realtime_framework.evaluation.config import OUTPUT_CLOCK_POLICY
from realtime_framework.evaluation.config import INTERACTABLES_PROTOCOL
from realtime_framework.evaluation.config import INTERACTABLES_REPRESENTATION
from realtime_framework.evaluation.config import INTERACTABLES_SCHEMA
from realtime_framework.evaluation.config import apply_run_defaults
from realtime_framework.evaluation.config import config_fingerprint
from realtime_framework.evaluation.config import effective_run_config
from realtime_framework.evaluation.config import harness_fingerprint, harness_version
from realtime_framework.evaluation.reporting import summarize_run_metrics
from realtime_framework.evaluation.reporting import summarize_streaming
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.evaluation.reporting import episode_outcome
from realtime_framework.evaluation.agent_trace import AgentTraceSink
from realtime_framework.evaluation.agent_trace import close_agent_trace
from realtime_framework.entity.records import canonical_sha256
import realtime_tasks.registry as custom_tasks
from realtime_framework.sandbox.observation.som import ToolRuntime


def _model_identity(requested_model, requested_adapter=None):
    """Separate the stable catalog key from the actual backend model and adapter."""
    if catalog.has(requested_model):
        entry = catalog.get(requested_model)
        return {
            "adapter": entry.adapter_type,
            "model": requested_model,
            "backend_model": entry.model_name,
        }
    return {
        "adapter": requested_adapter,
        "model": requested_model,
        "backend_model": requested_model,
    }


def _distinct(failures: list[dict]) -> list[dict]:
    """The failures a run found, each once: a failed snapshot read is found by the kernel and
    again by main(), with the same category and reason."""
    found, distinct = set(), []
    for failure in failures:
        if (failure["category"], failure["reason"]) not in found:
            found.add((failure["category"], failure["reason"]))
            distinct.append(failure)
    return distinct


def task_clock_tokens_per_frame(tokens_per_frame, env_fps, model_input_fps):
    """B in the task clock's frames, which are environment ticks. The task clock records
    tokens_per_frame times its fps, the environment's, as the tokens per world second
    (realtime_tasks/types.py clock_snapshot), which is then B times the model-input fps; at
    equal rates it gets B as given."""
    if tokens_per_frame is None or env_fps == model_input_fps:
        return tokens_per_frame
    return tokens_per_frame * model_input_fps / env_fps


def _metric_avg(env, name):
    values = list(getattr(env, name, []) or []) if env is not None else []
    return round(sum(values) / len(values), 6) if values else None


def _metric_max(env, name):
    values = list(getattr(env, name, []) or []) if env is not None else []
    return round(max(values), 6) if values else None


def _effective_capture_fps(env):
    values = list(getattr(env, "capture_times", []) or []) if env is not None else []
    if len(values) < 2 or values[-1] <= values[0]:
        return None
    return round((len(values) - 1) / (values[-1] - values[0]), 6)


def _atomic_write_json(path, obj):
    """Write and fsync a sibling temporary file, then atomically replace the target.

    Interrupted writes preserve the previous complete result for resume checks."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class EnvironmentInvalidError(RuntimeError):
    """Startup health checks failed; the result must not be attributed to the model."""


def _invalid_snapshot(snapshot):
    if not isinstance(snapshot, dict):
        return "snapshot_not_an_object"
    state = snapshot.get("state")
    if not isinstance(state, dict) or not state:
        return "snapshot_missing_environment_state"
    if state.get("ok") is False:
        return "snapshot_environment_unhealthy"
    evaluation = snapshot.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("error"):
        return "snapshot_evaluator_error"
    return None


# Capture, simulated-time advancement, and action execution live in realtime_framework.sandbox.actions.
from realtime_framework.entity.config import task_seed


def _parser() -> argparse.ArgumentParser:
    """The runner's arguments, as main() parses them and the emergency summary reads them."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adapter",
        default=None,
        help="adapter_type; optional when --model is a catalog key (then it is derived).",
    )
    p.add_argument(
        "--model",
        required=True,
        help="a catalog key (realtime_framework/providers/registry.py) OR a raw backend id / local model path",
    )
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument(
        "--model-input-fps",
        type=float,
        default=None,
        help="frames per second fed to the model (default: --fps, the environment's rate)",
    )
    p.add_argument(
        "--interaction-mode",
        choices=("turn_based", "streaming"),
        default="turn_based",
        help="turn_based: one frame per model generation (the leaderboard); streaming: frames "
        "pushed into a live session while the model generates.",
    )
    p.add_argument(
        "--clock",
        choices=["wall", "token"],
        default=None,
        help="wall uses deployment time; token places actions at n/B frame positions.",
    )
    p.add_argument(
        "--tokens-per-frame",
        default=str(DEFAULT_TOKENS_PER_FRAME),
        help="positive token-to-time rate B per frame (default: %(default)s); "
        "'unlimited' acts at the observed tick, then advances one frame.",
    )
    p.add_argument(
        "--reasoning-effort",
        default=None,
        help="model-native reasoning effort; catalog models validate their own range.",
    )
    p.add_argument("--task", required=True, help="registered task id")
    p.add_argument("--seed", type=task_seed, default=os.environ.get("SEED", "42"))
    p.add_argument("--max-minutes", type=float, default=5.0)
    p.add_argument(
        "--think-budget-s",
        type=float,
        default=None,
        help="Deliberation clock: cumulative token-derived simulated seconds in token "
        "mode, wall seconds in wall mode. Legacy diagnostic only; the maintained "
        "protocol leaves this unset and terminates at T. Turn-based only.",
    )
    p.add_argument(
        "--per-move-cap-s",
        type=float,
        default=None,
        help="Per-move deliberation cap in the active clock's seconds. A move over it "
        "times out (no action that turn; costs exactly the cap from the "
        "deliberation clock). Legacy diagnostic only; the maintained protocol "
        "leaves this unset. Turn-based only.",
    )
    p.add_argument(
        "--protocol", default=None, help="Identity label for the executable evaluation protocol."
    )
    p.add_argument("--run-id", default=None)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--report-root", type=Path, help="Episode launch directory to validate and report")
    p.add_argument("--report-output", type=Path, help="Also write the episode report here")
    p.add_argument(
        "--graphics-mode",
        choices=("cpu", "gpu"),
        default=os.environ.get("SANDBOX_GRAPHICS_MODE", "cpu"),
        help="Desktop browser renderer: cpu uses SwiftShader; gpu uses NVIDIA Vulkan.",
    )
    p.add_argument(
        "--recording-encoder",
        choices=("cpu", "gpu"),
        default=os.environ.get("SCREEN_RECORDING_ENCODER", "cpu"),
        help="Screen recording encoder: cpu uses libx264; gpu uses NVENC.",
    )
    return p


def main():
    p = _parser()
    a = p.parse_args()
    a.mode = DEFAULT_MODE  # fixed identity field; the runner has one decision loop
    # prepare_episode_report reads the run's identity from the arguments
    apply_run_defaults(a)
    if a.clock == "wall" or a.tokens_per_frame == "unlimited":
        tokens_per_frame = None
    else:
        try:
            tokens_per_frame = int(a.tokens_per_frame)
        except ValueError:
            p.error("--tokens-per-frame must be a positive integer or 'unlimited'")
        if tokens_per_frame <= 0:
            p.error("--tokens-per-frame must be positive")
    protocol = a.protocol
    if not 0 < a.fps < float("inf") or not 0 < a.max_minutes < float("inf"):
        p.error("--fps and --max-minutes must be finite and positive")
    if not 0 < a.model_input_fps < float("inf"):
        p.error("--model-input-fps must be finite and positive")
    # --fps is the environment's frame rate (task clock, ticks, loop, scored world time);
    # --model-input-fps the rate of the frames the model is fed (ModelConfig.fps, B, the prompt).
    env_fps, model_input_fps = a.fps, a.model_input_fps
    if protocol == "time-aware-token-v1":
        if a.clock != "token":
            p.error("time-aware-token-v1 requires --clock token")
        if a.think_budget_s is not None or a.per_move_cap_s is not None:
            p.error(
                "time-aware-token-v1 uses only the task horizon T; deliberation caps are "
                "legacy diagnostics"
            )
    try:
        check_streaming_options(
            a.interaction_mode, think_budget_s=a.think_budget_s, per_move_cap_s=a.per_move_cap_s
        )
    except ValueError as exc:
        p.error(str(exc))
    requested_model_identity = _model_identity(a.model, a.adapter)
    catalog_entry = catalog.get(a.model) if catalog.has(a.model) else None
    reasoning_effort = a.reasoning_effort
    if catalog_entry is not None:
        try:
            catalog.validate_reasoning_effort(a.model, reasoning_effort)
        except ValueError as exc:
            p.error(str(exc))
    benchmark_cell_eligible = bool(
        catalog_entry.benchmark_eligible if catalog_entry is not None else True
    )
    comparison_scope = catalog_entry.comparison_scope if catalog_entry is not None else "ad_hoc"
    if not benchmark_cell_eligible and os.environ.get("ALLOW_EXPERIMENTAL_MODEL") != "1":
        p.error(
            f"catalog cell {a.model!r} is excluded from maintained benchmark results; "
            "set ALLOW_EXPERIMENTAL_MODEL=1 only for explicitly non-comparable diagnostics"
        )
    try:
        check_token_budget(
            requested_model_identity["adapter"],
            clock_mode=a.clock,
            tokens_per_frame=tokens_per_frame,
        )
    except UnsupportedTokenBudget as exc:
        p.error(str(exc))
    clocks = episode_clocks(a.clock, env_fps)
    run_id = a.run_id or datetime.fromtimestamp(clocks.operational.time()).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(a.results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # A run retried in its run directory: the summary there is moved aside, so that the one in
    # the directory is this run's, the emergency summary's too.
    if (run_dir / "run_summary.json").exists():
        (run_dir / "run_summary.json").replace(run_dir / "run_summary.pre_retry.json")
    episode_trace = EpisodeTrace(
        run_dir / "episode_trace.jsonl", run_id=run_id, clock=clocks.operational
    )
    episode_trace.emit(
        TraceEvent.EPISODE_STARTED,
        {
            "task_id": a.task,
            "requested_model": requested_model_identity,
            "clock_mode": a.clock,
            "mode": a.mode,
            "fps": env_fps,
            "model_input_fps": model_input_fps,
            "interaction_mode": a.interaction_mode,
            "tokens_per_frame": tokens_per_frame,
            "reasoning_effort": reasoning_effort,
            "max_minutes": a.max_minutes,
            "seed": a.seed,
            "protocol": protocol,
            "benchmark_eligible": benchmark_cell_eligible,
            "comparison_scope": comparison_scope,
            "graphics_mode": a.graphics_mode,
            "recording_encoder": a.recording_encoder,
        },
    )

    env_config = EnvironmentConfig()
    R.build_image_if_needed()
    if a.report_root:
        from realtime_framework.sandbox.apptainer import build_image
        from realtime_framework.evaluation.reporting import prepare_episode_report

        build_image("realtime")
        prepare_episode_report(a)
    R.start_container(env_config)
    guest = GuestClient(host="localhost", port=env_config.guest_agent_port)
    if not guest.wait_until_ready(max_wait=60):
        raise RuntimeError("guest not ready")
    R._start_screensaver_keepalive()
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Initialize bookkeeping before setup so early failures produce a summary
    # and the finally block can release resources.
    ctask = None
    task = None
    agent = None
    agent_trace = None
    cfg = None
    env = None
    think_budget = None
    per_move_cap = None
    frozen_snapshot = None
    timeline = []
    verification = None
    solved = None
    error = None
    generation_artifact = None
    artifact_error = None
    evaluation_path = None
    environment_invalid = False
    invalid_runs = []  # every failure main() finds, in the order found
    scene_capture = None
    tool_runtime = None
    loop_time_s = None
    _recorder_on = False
    active_loop = None
    harness_hash = harness_fingerprint()
    runtime_config = None
    runtime_config_hash = None
    t0 = clocks.operational.time()
    run_stage = "environment_setup"
    try:
        # -------- provision the registered task --------
        ctask = custom_tasks.get(a.task)
        ctask.configure_seed(a.seed)
        task = ctask.description()
        ctask.configure_clock(
            a.clock, env_fps, task_clock_tokens_per_frame(tokens_per_frame, env_fps, model_input_fps)
        )
        url = ctask.provision()
        if url:
            R.open_local_url_in_container(url, debug_port=getattr(ctask, "dbg_port", None))
        clocks.operational.sleep(5)

        # -------- resolve the model + build the agent --------
        # A catalog key (e.g. gpt-realtime-2, claude-opus-5) sets the adapter +
        # backend id/path + per-model defaults from realtime_framework/providers/registry.py; a raw id/path still works
        # if --adapter is given. B controls simulated time rather than response length; only the
        # remaining episode horizon and native model limits can bound a decision.
        run_stage = "provider_setup"
        if catalog.has(a.model):
            cfg = catalog.to_model_config(
                a.model,
                fps=model_input_fps,
                reasoning_effort=reasoning_effort,
            )
        else:
            if not a.adapter:
                p.error("--adapter is required when --model is not a catalog key")
            cfg = ModelConfig(
                adapter_type=a.adapter,
                model_name=a.model,
                fps=model_input_fps,
                reasoning_effort=reasoning_effort,
            )
        setup = AgentSetup()
        try:
            set_up_agent(
                setup,
                cfg,
                ctask,
                task,
                clock_mode=a.clock,
                interaction_mode=a.interaction_mode,
                tokens_per_frame=tokens_per_frame,
                run_dir=run_dir,
                episode_trace=episode_trace,
                model=requested_model_identity["model"],
                open_capture=lambda text_observation: BrowserSceneCapture(
                    guest,
                    ctask,
                    debug_port=getattr(ctask, "dbg_port", None),
                    container=R.CONTAINER_NAME,
                    desktop_size=(env_config.screen_width, env_config.screen_height),
                    text_observation=text_observation,
                    clock=clocks.operational,
                ),
                clock=clocks.operational,
                env_fps=env_fps,
            )
        finally:
            # The cleanup below closes what setup made, also when one of its steps failed.
            run_stage, environment_invalid = setup.stage, setup.environment_invalid
            agent, agent_trace = setup.agent, setup.agent_trace
            scene_capture, tool_runtime = setup.scene_capture, setup.tool_runtime

        # Optional legacy diagnostics may add deliberation caps explicitly. The maintained method
        # uses only the shared episode horizon T and therefore does not inherit task-local caps.
        think_budget = a.think_budget_s
        if think_budget is not None:
            print(
                f"[think] deliberation budget = {think_budget:g}s of cumulative model thinking",
                flush=True,
            )
        per_move_cap = a.per_move_cap_s
        if per_move_cap is not None:
            print(
                f"[think] per-move cap = {per_move_cap:g}s (a slower move times out -> no action that turn)",
                flush=True,
            )
        runtime_config = effective_run_config(
            seed=a.seed,
            model_config=cfg,
            fps=env_fps,
            mode=a.mode,
            max_minutes=a.max_minutes,
            protocol=protocol,
            think_budget_s=think_budget,
            per_move_cap_s=per_move_cap,
            clock_mode=a.clock,
            tokens_per_frame=tokens_per_frame,
            interaction_mode=a.interaction_mode,
        )
        runtime_config_hash = config_fingerprint(runtime_config)
        _atomic_write_json(
            run_dir / "run_config.json",
            {
                **runtime_config,
                "harness_version": harness_version(harness_hash),
                "harness_hash": harness_hash,
                "runtime_config_hash": runtime_config_hash,
            },
        )

        # initialize() may open a realtime session or prefill a local model.  Restart time-sensitive
        # wall-clock tasks only after that work is complete.  The restart is two-phase: reset_episode()
        # prepares a held episode (reload, baseline, semantics, browser render) and start_episode()
        # releases it.  The origin start_episode() reports is the scored wall-clock origin, so no
        # framework setup consumes the episode.  Record only after provider/environment
        # initialization; the first-frame PTS and explicit active boundaries remove the small
        # recorder startup margin.
        _recorder_on = R._start_screen_recording(run_dir)
        wall_start_time = None
        episode_start = None
        if (
            a.clock == "wall"
            and ctask is not None
            and hasattr(ctask, "reset_episode")
            and getattr(ctask, "category", None) in {"browser_game", "realtime"}
        ):
            category = getattr(ctask, "category", None)
            try:
                _rb = ctask.reset_episode()
                print(
                    f"[{category}] scored wall episode reset after adapter initialization; "
                    f"baseline={_rb}",
                    flush=True,
                )
                if category == "realtime":
                    # Let the open page render the held, restarted state before the clock starts.
                    clocks.operational.sleep(0.45)
                episode_start = ctask.start_episode()
                wall_start_time = float(episode_start["clock_origin"])
                print(f"[{category}] scored wall episode started; {episode_start}", flush=True)
            except Exception:
                detail = traceback.format_exc()
                environment_invalid = True
                print(
                    f"[{category}] scored wall restart FAILED; run is invalid:\n" + detail,
                    flush=True,
                )
                raise EnvironmentInvalidError(
                    f"{category} episode restart failed after adapter initialization\n" + detail
                )

        # -------- environment budget --------
        if a.clock == "token":
            deadline = None
        else:
            wall_start_time = wall_start_time or clocks.semantic.time()
            deadline = wall_start_time + a.max_minutes * 60
        # Any task may expose episode_over for a server-owned terminal state; realtime tasks use it
        # for irreversible failures.
        env = GuestEnvironment(
            guest,
            ctask,
            frames_dir=frames_dir,
            stop_check=(getattr(ctask, "episode_over", None) if ctask else None),
            scene_capture=scene_capture,
            action_audit_path=run_dir / "action_lifecycle.jsonl",
            clock_mode=a.clock,
            wall_start_time=wall_start_time,
            clocks=clocks,
        )

        # -------- run the streaming loop --------
        _loop_t0 = clocks.operational.time()
        run_stage = "inference"
        _active_start = wall_start_time if a.clock == "wall" else _loop_t0
        episode_trace.emit(
            "ActiveLoopStarted",
            {
                "start_unix_s": _active_start,
                "loop_entered_unix_s": _loop_t0,
                "deadline_unix_s": deadline,
                "clock_mode": a.clock,
                "episode_start": episode_start,
            },
        )
        try:
            timeline = run_stream(
                agent,
                env,
                clock_mode=a.clock,
                duration_s=a.max_minutes * 60,
                env_fps=env_fps,
                input_fps=model_input_fps,
                tokens_per_frame=tokens_per_frame,
                think_budget_s=think_budget,
                per_move_cap_s=per_move_cap,
                log=lambda m: print(m, flush=True),
                trace=episode_trace,
                agent_trace=agent_trace,
                clocks=clocks,
                inputs_log=run_dir / "kernel_inputs.jsonl",
                interaction_mode=a.interaction_mode,
            )
            error = getattr(env, "_loop_error", None)
        finally:
            # The agent's run window is recorded even when the loop ends in an error.
            # Physical diagnostic only; the token-clock horizon is simulated and does not bound wall time.
            loop_time_s = round(clocks.operational.time() - _loop_t0, 1)
            _active_end = getattr(env, "_active_end_unix_s", clocks.operational.time())
            active_loop = {
                "start_unix_s": _active_start,
                "end_unix_s": _active_end,
                "loop_entered_unix_s": _loop_t0,
                "deadline_unix_s": deadline,
                "episode_start": episode_start,
                # Setup between the task's scored origin and the scheduler's first observation.
                "start_overhead_s": (_loop_t0 - wall_start_time) if a.clock == "wall" else 0.0,
                "active_wall_time_s": _active_end - _active_start,
                "clock_mode": a.clock,
                "horizon_s": a.max_minutes * 60,
            }
            _atomic_write_json(run_dir / "active-loop.json", active_loop)
            episode_trace.emit("ActiveLoopEnded", active_loop)
        R._stop_screen_recording(run_dir, frames_dir, model_input_fps)
        _recorder_on = False
        # INFERENCE/EVAL BOUNDARY: run_stream freezes live scorer input at the latched terminal tick,
        # before it releases held keys or mouse buttons. Consume that exact snapshot here, before
        # agent close or verifier work. The same value is stored in the artifact
        # so python -m realtime_framework.evaluation.reporting reevaluate agrees with the summary.
        try:
            snapshot_error = getattr(env, "_termination_snapshot_error", None)
            if snapshot_error:
                raise RuntimeError(snapshot_error)
            if getattr(env, "_termination_snapshot_captured", False):
                frozen_snapshot = getattr(env, "_termination_snapshot", None)
            snapshot_invalid_reason = _invalid_snapshot(frozen_snapshot)
            if snapshot_invalid_reason:
                invalid_runs.append(
                    {
                        "category": "environment",
                        "reason": snapshot_invalid_reason,
                        "detail": json.dumps(frozen_snapshot, default=str)[:2000],
                    }
                )
            episode_trace.emit(
                TraceEvent.SNAPSHOT_FROZEN,
                {
                    "kind": (frozen_snapshot or {}).get("kind"),
                    "snapshot_sha256": canonical_sha256(frozen_snapshot),
                    "artifact": "generation_artifact/task_state.json",
                    "valid": snapshot_invalid_reason is None,
                    "invalid_reason": snapshot_invalid_reason,
                    "environment_tick": env.tick_index() if env is not None else None,
                },
            )
            episode_trace.checkpoint()
        except Exception:
            detail = traceback.format_exc()
            failure = {
                "category": "environment",
                "reason": "evaluation_snapshot_failed",
                "detail": detail,
            }
            invalid_runs.append(failure)
            episode_trace.emit(TraceEvent.RUN_INVALIDATED, failure)
            print("[invalid-run] evaluation snapshot failed:\n" + detail, flush=True)
        try:
            generation_artifact = capture_generation_artifact(
                run_dir,
                clock=clocks.operational,
                ctask=ctask,
                adapter=cfg.adapter_type,
                model=requested_model_identity["model"],
                backend_model=cfg.model_name,
                mode=a.mode,
                cap=a.max_minutes,
                protocol=protocol,
                harness_hash=harness_hash,
                fps=env_fps,
                model_input_fps=model_input_fps,
                interaction_mode=a.interaction_mode,
                runtime_config=runtime_config,
                runtime_config_hash=runtime_config_hash,
                snapshot=frozen_snapshot,
            )
            print("[artifact] generation state saved before eval", flush=True)
        except Exception:
            artifact_error = traceback.format_exc()
            print(
                "[artifact] capture failed (evaluation will continue):\n" + artifact_error,
                flush=True,
            )

        # A per-move generate the runner abandoned on timeout is still running (daemon, un-killable).
        # Wait it out HERE — after the evaluation snapshot and artifact are frozen, so the drain adds
        # no drift to the score, and before close(), which would otherwise tear the model/session down
        # underneath a live native call. Bounded; _hard_exit covers a worker that never returns.
        try:
            drain = getattr(env, "_drain_abandoned", None)
            if drain is not None:
                drain()
        except Exception:
            pass
        try:
            agent.close()
        except Exception:
            invalid_runs.append(
                {
                    "category": "provider",
                    "reason": "provider_close_failed",
                    "detail": traceback.format_exc(),
                }
            )

        run_stage = "evaluation"
        if frozen_snapshot is None:
            # The run is invalid for its snapshot already, and nothing read since is its state.
            evaluation_path = str(save_evaluation(run_dir, None, error="no frozen snapshot"))
        else:
            try:
                # score the FROZEN snapshot (== the artifact), not a fresh post-cleanup live read
                verification = ctask.evaluate_snapshot(frozen_snapshot)
                evaluation_invalid_reason = invalid_evaluation(verification)
                if evaluation_invalid_reason:
                    invalid_runs.append(
                        {
                            "category": "evaluator",
                            "reason": evaluation_invalid_reason,
                            "detail": json.dumps(verification, default=str)[:2000],
                        }
                    )
                    solved = None
                else:
                    solved = verification["resolved"]
                evaluation_path = str(save_evaluation(run_dir, verification))
                episode_trace.emit(
                    TraceEvent.SCORE_PRODUCED,
                    {
                        "valid": evaluation_invalid_reason is None,
                        "invalid_reason": evaluation_invalid_reason,
                        "resolved": solved,
                        "verification": verification,
                        "artifact": "evaluation.json",
                    },
                )
                episode_trace.checkpoint()
            except Exception:
                _eval_error = traceback.format_exc()
                failure = {
                    "category": "evaluator",
                    "reason": "evaluation_exception",
                    "detail": _eval_error,
                }
                invalid_runs.append(failure)
                episode_trace.emit(TraceEvent.RUN_INVALIDATED, failure)
                evaluation_path = str(save_evaluation(run_dir, None, error=_eval_error))
                print("[invalid-run] evaluator exception:\n" + _eval_error, flush=True)
    except Exception:
        error = traceback.format_exc()
        invalid_runs.append(
            {
                "category": (
                    "environment"
                    if isinstance(sys.exc_info()[1], EnvironmentInvalidError)
                    or run_stage == "environment_setup"
                    else "provider"
                    if run_stage == "provider_setup"
                    else "tool"
                    if run_stage == "tool_setup"
                    else "evaluator"
                    if run_stage == "evaluation"
                    else "runner"
                ),
                "reason": (
                    "environment_preflight_failed" if environment_invalid else "run_exception"
                ),
                "detail": error,
            }
        )
        print("RUN ERROR\n" + error, flush=True)
        if evaluation_path is None:
            try:
                evaluation_path = str(save_evaluation(run_dir, None, error=error))
            except Exception:
                pass
    finally:
        # Stop recording and close model resources even when setup or scoring fails.
        # The shared container remains available for subsequent runs.
        if _recorder_on:
            try:
                R._stop_screen_recording(run_dir, frames_dir, model_input_fps)
            except Exception:
                pass
        try:
            # Same reason as the main path: never close() under a live abandoned native call.
            # Idempotent, so this is a no-op when the main path already drained. env is None if we
            # failed before the loop existed, in which case there is nothing to drain.
            drain = getattr(env, "_drain_abandoned", None)
            if drain is not None:
                drain()
        except Exception:
            pass
        try:
            if agent is not None:
                agent.close()
        except Exception:
            invalid_runs.append(
                {
                    "category": "provider",
                    "reason": "provider_close_failed",
                    "detail": traceback.format_exc(),
                }
            )
        # After the agent is closed: stop new agent records, write the queued ones and fence the
        # writer before the checkpoint below and EpisodeFinished.
        if agent_trace is not None:
            failure = close_agent_trace(agent_trace)
            if failure is not None:
                invalid_runs.append(failure)
        try:
            if tool_runtime is not None:
                tool_runtime.close()
        except Exception:
            invalid_runs.append(
                {
                    "category": "tool",
                    "reason": "tool_trace_close_failed",
                    "detail": traceback.format_exc(),
                }
            )
        try:
            if scene_capture is not None:
                scene_capture.close()
        except Exception:
            invalid_runs.append(
                {
                    "category": "environment",
                    "reason": "cdp_close_failed",
                    "detail": traceback.format_exc(),
                }
            )
        try:
            episode_trace.checkpoint()
        except Exception:
            invalid_runs.append(
                {
                    "category": "harness",
                    "reason": "episode_trace_checkpoint_failed",
                    "detail": traceback.format_exc(),
                }
            )
    deltas = [x[2] for x in timeline]
    # Billing uses complete provider-reported cost when available (OpenRouter), otherwise a
    # versioned catalog rate card with modality and cache-read splits (direct providers).
    tok = getattr(agent, "token_usage", None) if agent is not None else None
    adapter_tok = tok
    tok = {"input_tokens": None, "output_tokens": None, "total_tokens": None, "responses": 0}
    accounting = getattr(env, "_token_accounting", None) if env is not None else None
    token_accounting = accounting.summary() if accounting is not None else None
    if token_accounting is not None:
        counts = token_accounting["counts"]
        tok = {
            name: counts["provider_" + name]["total"]
            for name in ("input_tokens", "output_tokens", "total_tokens")
        }
        tok["responses"] = token_accounting["responses"]
    cost_accounting = summarize_cost(
        token_accounting,
        pricing=(catalog_entry.pricing if catalog_entry is not None else None),
        provider=(catalog_entry.provider if catalog_entry is not None else None),
        # The streaming shell accounts no reply's tokens (episode_records.account_tokens).
        unaccounted="not_accounted_streaming" if a.interaction_mode == "streaming" else None,
    )
    from realtime_framework.evaluation.clock import summarize_model_timing

    model_timing = summarize_model_timing(run_dir, interaction_mode=a.interaction_mode)
    cost_usd = cost_accounting.get("total_usd") if cost_accounting.get("complete") else None
    cost_known_usd = cost_accounting.get("known_cost_usd")
    # The environment latches a failure only while the episode runs, before any main() finds.
    latched_invalid = getattr(env, "_invalid_run", None) if env is not None else None
    if latched_invalid is not None:
        invalid_runs.insert(0, latched_invalid)
    invalid_run = invalid_runs[0] if invalid_runs else None
    if invalid_run is not None:
        solved = None
    run_status = (
        "invalid"
        if invalid_run is not None
        else ("scored" if benchmark_cell_eligible else "experimental")
    )
    # The stop reason and the end time are LATCHED by the runner at loop exit (env._stop_reason,
    # env._termination), so cleanup that crosses the deadline rewrites neither a real environment
    # terminal state into time_limit nor the episode's length. Any invalid run reports
    # "invalid_run"; "setup_error" if we never reached the loop.
    latched = getattr(env, "_stop_reason", None) if env is not None else None
    termination = getattr(env, "_termination", None) if env is not None else None
    stop_reason = (
        "invalid_run"
        if invalid_run is not None
        else (latched or ("setup_error" if env is None else "unknown"))
    )
    world_time_s = termination["end_s"] if termination is not None else None
    # The trace holds every record the episode wrote, checkpointed above. A streaming section
    # that cannot be read leaves the run unscored, as the run metrics below then count it.
    streaming = None
    if a.interaction_mode == "streaming":
        try:
            streaming = summarize_streaming(
                run_dir / "episode_trace.jsonl",
                (runtime_config or {}).get("streaming"),
                attempt_id=episode_trace.attempt_id,
            )
        # A record written in part has the next one after it on its line (EpisodeTrace.emit).
        except json.JSONDecodeError:
            invalid_runs.append(
                {
                    "category": "harness",
                    "reason": "streaming_summary_failed",
                    "detail": traceback.format_exc(),
                }
            )
            invalid_run = invalid_runs[0]
            solved = None
            run_status = "invalid"
            stop_reason = "invalid_run"
    try:
        run_metrics = summarize_run_metrics(
            run_dir / "action_lifecycle.jsonl",
            total_decisions=len(timeline),
            solved=solved,
            clock_mode=a.clock,
            tokens_per_frame=tokens_per_frame,
            fps=model_input_fps,
            termination=termination,
            horizon_s=a.max_minutes * 60,
        )
    except Exception:
        metrics_error = traceback.format_exc()
        invalid_runs.append(
            {
                "category": "harness",
                "reason": "run_metrics_aggregation_failed",
                "detail": metrics_error,
            }
        )
        invalid_run = invalid_runs[0]
        solved = None
        run_status = "invalid"
        stop_reason = "invalid_run"
        run_metrics = summarize_run_metrics(
            run_dir / "missing-action-lifecycle.jsonl",
            total_decisions=len(timeline),
            solved=None,
            clock_mode=a.clock,
            tokens_per_frame=tokens_per_frame,
            fps=model_input_fps,
            termination=None,
            horizon_s=a.max_minutes * 60,
            validate_decision_count=False,
        )
        run_metrics["run_metrics_error"] = metrics_error
    effective_model_identity = {
        "adapter": cfg.adapter_type if cfg else requested_model_identity["adapter"],
        "model": requested_model_identity["model"],
        "backend_model": cfg.model_name if cfg else requested_model_identity["backend_model"],
    }
    run_identity = resolve_run_identity(
        task_id=a.task,
        ctask=ctask,
        **effective_model_identity,
        mode=a.mode,
        cap=a.max_minutes,
        protocol=protocol,
        harness_hash=harness_hash,
        fps=env_fps,
        model_input_fps=model_input_fps,
        interaction_mode=a.interaction_mode,
        runtime_config_hash=runtime_config_hash,
    )
    failures = _distinct(invalid_runs)
    summ = {
        **run_identity,
        "instance_id": a.task,
        "task_description": task,
        "reasoning_effort": reasoning_effort,
        "benchmark_eligible": benchmark_cell_eligible,
        "comparison_scope": comparison_scope,
        "interactables_protocol": (INTERACTABLES_PROTOCOL if tool_runtime is not None else None),
        "interactables_schema": (INTERACTABLES_SCHEMA if tool_runtime is not None else None),
        "interactables_representation": (
            INTERACTABLES_REPRESENTATION if tool_runtime is not None else None
        ),
        "interactables_artifact": ("tool_calls.jsonl" if tool_runtime is not None else None),
        "run_status": run_status,
        "invalid_run": invalid_run,
        # Each failure after the run's too, which a re-evaluation keeps (reporting.reevaluate_main).
        **({"invalid_runs": failures} if len(failures) > 1 else {}),
        "solved": solved,
        "decisions": len(timeline),
        "actions_executed": getattr(env, "actions", 0) if env is not None else 0,
        "actions_failed": getattr(env, "failed", 0) if env is not None else 0,
        "avg_delta_ticks": round(sum(deltas) / len(deltas), 2) if deltas else None,
        "token_usage": tok,
        "cost_usd": cost_usd,
        "cost_known_usd": cost_known_usd,
        "cost_status": cost_accounting.get("status"),
        "cost_source": cost_accounting.get("source"),
        "cost_accounting": cost_accounting,
        "token_usage_source": "provider_reported_inference_responses",
        "token_accounting": token_accounting,
        "token_clock_sources": (
            token_accounting.get("clock_sources") if token_accounting is not None else None
        ),
        "adapter_token_usage": adapter_tok,
        "adapter_token_usage_note": "Adapter-native cumulative counters; not used for the action clock or billing.",
        "token_accounting_source": (
            ",".join(token_accounting.get("clock_sources") or []) or None
            if token_accounting is not None
            else None
        ),
        "stop_reason": stop_reason,
        "failure_attribution": ((invalid_run or {}).get("category") if invalid_run else None),
        "environment_valid": (
            False
            if (invalid_run or {}).get("category") == "environment" or environment_invalid
            else None
        ),
        "loop_stop_reason": latched,
        "termination": termination,
        "max_minutes": a.max_minutes,
        "seed": a.seed,
        "applied_seed": getattr(ctask, "applied_seed", None),
        "clock_mode": a.clock,
        "tokens_per_frame": tokens_per_frame,
        "dropped_intermediate_frames": int(getattr(env, "_dropped_intermediate_frames", 0) or 0)
        if env is not None
        else 0,
        "output_clock_policy": OUTPUT_CLOCK_POLICY,
        "tokens_per_world_second": (
            tokens_per_frame * model_input_fps
            if a.clock == "token" and tokens_per_frame is not None
            else None
        ),
        "world_time_s": world_time_s,
        "last_clock_advance": (
            getattr(ctask, "_last_clock_advance", None) if ctask is not None else None
        ),
        "think_budget_s": think_budget,
        "per_move_cap_s": per_move_cap,
        "think_spent_s": round(getattr(env, "_think_spent", 0.0), 1) if env is not None else 0.0,
        "runtime_config": runtime_config,
        "observation_stream_frozen": bool(getattr(env, "_stream_frozen", False))
        if env is not None
        else False,
        "observation_stream_freeze_count": int(getattr(env, "_stream_freeze_count", 0) or 0)
        if env is not None
        else 0,
        "loop_time_s": loop_time_s,
        "active_wall_time_s": (active_loop or {}).get("active_wall_time_s"),
        "active_loop_artifact": "active-loop.json"
        if (run_dir / "active-loop.json").is_file()
        else None,
        "captures": len(getattr(env, "capture_times", [])) if env is not None else 0,
        "effective_capture_fps": _effective_capture_fps(env),
        "avg_capture_latency_s": _metric_avg(env, "capture_latencies"),
        "max_capture_latency_s": _metric_max(env, "capture_latencies"),
        "avg_frame_age_at_action_s": _metric_avg(env, "frame_ages"),
        "max_frame_age_at_action_s": _metric_max(env, "frame_ages"),
        "action_lifecycle_artifact": (
            "action_lifecycle.jsonl" if (run_dir / "action_lifecycle.jsonl").is_file() else None
        ),
        "wall_time_s": model_timing["wall_time_s"],
        "model_timing": model_timing,
        "total_run_time_s": round(clocks.operational.time() - t0, 1),
        "verification": verification,
        "error": error,
        "generation_artifact": (
            "generation_artifact/manifest.json" if generation_artifact is not None else None
        ),
        "artifact_error": artifact_error,
        "evaluation_artifact": ("evaluation.json" if evaluation_path is not None else None),
        # Partial-credit progress headline (None for games without partial credit).
        "progress": (verification or {}).get("progress"),
        **run_metrics,
    }
    if invalid_run is not None:
        episode_trace.emit(TraceEvent.RUN_INVALIDATED, invalid_run)
    episode_trace.emit(
        TraceEvent.EPISODE_FINISHED,
        {
            "run_status": run_status,
            **episode_outcome(summ),
            "stop_reason": stop_reason,
            "resolved": solved,
            "decisions": len(timeline),
            "actions_executed": getattr(env, "actions", 0) if env is not None else 0,
            "world_time_s": world_time_s,
        },
    )
    summ.update(episode_outcome(summ))
    summ["episode_trace"] = episode_trace.describe()
    episode_trace.close()
    if a.interaction_mode == "streaming":
        summ["streaming"] = streaming
    _atomic_write_json(run_dir / "run_summary.json", summ)
    from realtime_framework.evaluation.reporting import finalize_video

    finalize_video(run_dir)
    print(
        "SUMMARY",
        json.dumps(
            {
                k: summ[k]
                for k in [
                    "adapter",
                    "mode",
                    "run_status",
                    "outcome",
                    "error_reason",
                    "solved",
                    "progress",
                    "decisions",
                    "actions_executed",
                    "actions_failed",
                    "avg_delta_ticks",
                    "stop_reason",
                ]
            }
        ),
    )
    # Infrastructure-invalid runs are never successful process results and are never resumable as
    # scored cells. The final tasks suite additionally requires a Boolean score for every valid run.
    accepted = run_status == "scored" or (
        run_status == "experimental" and os.environ.get("ALLOW_EXPERIMENTAL_MODEL") == "1"
    )
    if not accepted or (custom_tasks.suite_id() == "tasks" and not isinstance(solved, bool)):
        raise SystemExit(1)


class AgentSetup:
    """What set_up_agent() has made so far. main() closes these after the episode, also when a
    later setup step fails, so each is recorded as soon as it exists. ``stage`` is main()'s
    run_stage, which attributes a failure, and ``environment_invalid`` marks a failed check of
    the environment."""

    def __init__(self):
        self.stage = "provider_setup"
        self.environment_invalid = False
        self.agent_trace = None
        self.agent = None
        self.scene_capture = None
        self.tool_runtime = None


def set_up_agent(
    setup: AgentSetup,
    cfg: ModelConfig,
    ctask,
    instruction: str,
    *,
    clock_mode: str,
    interaction_mode: str,
    tokens_per_frame: int | None,
    run_dir: Path,
    episode_trace: EpisodeTrace,
    model: str,
    open_capture,
    env_fps: float,
    clock=None,
) -> None:
    """Turn a model config and a provisioned task into an agent initialized on the first frame.

    ``instruction`` is the task description and ``model`` the model identity the tool trace
    records. ``open_capture(text_observation)`` returns the scene capture for the observation
    form the model's context policy names. ``clock`` is the episode's operational clock, which
    the agent reads (real time when omitted). ``env_fps`` is the environment frame rate, which may
    not be below the model-input rate ``cfg.fps`` and which the clock note describes where the two
    differ. What this makes is recorded in ``setup``.
    """
    clock = clock or SystemClock()
    # Every pre-run rule on the model, clock, interaction mode and task, before any is built.
    check_episode(
        cfg, clock_mode=clock_mode, task=ctask, env_fps=env_fps, interaction_mode=interaction_mode
    )
    # The agent borrows the episode trace's write entry; the runner still opens and closes it.
    setup.agent_trace = AgentTraceSink(episode_trace, run_dir / "episode_blobs.pack", clock=clock)
    agent = setup.agent = create_agent(
        cfg,
        log_dir=run_dir,
        trace=setup.agent_trace,
        clock=clock,
        interaction_mode=interaction_mode,
    )
    configure_agent(
        agent,
        fps=cfg.fps,
        tokens_per_frame=tokens_per_frame,
        clock_mode=clock_mode,
        category=getattr(ctask, "category", None),
        env_fps=env_fps,
    )
    setup.stage = "environment_setup"

    # Prepare a valid provisional episode/frame for adapter initialization. main() restarts
    # wall-clock tasks once more after initialize(), so none of this warmup consumes their scored
    # action window.
    if hasattr(ctask, "reset_episode"):
        category = getattr(ctask, "category", None)
        if category in {"browser_game", "realtime"}:
            try:
                _rb = ctask.reset_episode()
                print(f"[{category}] fresh episode reset; baseline={_rb}", flush=True)
                if category == "realtime" and clock_mode != "token":
                    clock.sleep(0.45)
            except Exception:
                detail = traceback.format_exc()
                setup.environment_invalid = True
                print(f"[{category}] fresh restart FAILED; run is invalid:\n" + detail, flush=True)
                raise EnvironmentInvalidError(
                    f"{category} episode restart failed before tick 0\n" + detail
                )
    ctask.activate_clock()
    print(f"[clock] {ctask.clock_snapshot()}", flush=True)
    setup.stage = "tool_setup"
    # Choice models need focused fields and control state to enumerate actions,
    # including when they also receive screenshots (context policy item 12).
    context_policy = current_context_policy().for_path(
        cfg.adapter_type, cfg.provider_options, cfg.context_preset
    )
    text_observation = context_policy.observation != "som_screenshot"
    scene_capture = setup.scene_capture = open_capture(text_observation)
    setup.tool_runtime = ToolRuntime(
        run_dir / "tool_calls.jsonl",
        task_id=ctask.task_id,
        adapter=cfg.adapter_type,
        model=model,
        text_observation=text_observation,
        clock=clock,
    )
    if hasattr(agent, "bind_tool_runtime"):
        agent.bind_tool_runtime(setup.tool_runtime)
    setup.stage = "environment_setup"
    # Chrome can start with keyboard focus in its address bar; the model acts on the page.
    try:
        scene_capture.ensure_page_focus()
    except Exception as exc:
        setup.environment_invalid = True
        raise EnvironmentInvalidError(f"page did not take keyboard focus: {exc}") from exc
    initial_observation = scene_capture.capture(0)
    initial_interactables = initial_observation.metadata.get("interactables") or {}
    if initial_interactables.get("scene", {}).get(
        "status"
    ) != "ok" or not initial_interactables.get("binding", {}).get("stable"):
        setup.environment_invalid = True
        raise EnvironmentInvalidError(
            "initial observation has unavailable or unstable interactables: "
            + json.dumps(initial_interactables, default=str)[:1000]
        )
    setup.stage = "provider_setup"
    agent.initialize(instruction, initial_observation)
    # Initialization is allowed to open a native session or warm a local KV cache, but the
    # scored first turn receives exactly one freshly bound observation from the frame clock.
    agent.discard_pending_observations()
    setup.stage = "environment_setup"


def _write_emergency_setup_summary(error: str):
    """Persist setup/provision failures that happen before main() enters its episode try/finally,
    from the arguments main() parsed."""
    a = _parser().parse_args()
    if not a.run_id:
        return
    run_dir = Path(a.results_dir) / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_summary.json"
    if path.exists():
        return
    model_identity = _model_identity(a.model, a.adapter)
    mode = DEFAULT_MODE
    clock_mode = a.clock or "token"
    protocol = a.protocol or ("wall-clock" if clock_mode == "wall" else "time-aware-token-v1")
    cap_minutes = a.max_minutes
    env_fps = a.fps
    model_input_fps = env_fps if a.model_input_fps is None else a.model_input_fps
    run_identity = resolve_run_identity(
        task_id=a.task,
        **model_identity,
        mode=mode,
        cap=cap_minutes,
        protocol=protocol,
        harness_hash=harness_fingerprint(),
        fps=env_fps,
        model_input_fps=model_input_fps,
        interaction_mode=a.interaction_mode,
    )
    tokens_per_frame = (
        None
        if clock_mode == "wall" or a.tokens_per_frame == "unlimited"
        else int(a.tokens_per_frame)
    )
    # A run that failed to set up made no decision; a lifecycle in a run directory used before is
    # an earlier run's.
    run_metrics = summarize_run_metrics(
        run_dir / "missing-action-lifecycle.jsonl",
        total_decisions=0,
        solved=None,
        clock_mode=clock_mode,
        tokens_per_frame=tokens_per_frame,
        fps=model_input_fps,
        termination=None,
        horizon_s=cap_minutes * 60,
    )
    cost_accounting = summarize_cost(None)
    _atomic_write_json(
        path,
        {
            **run_identity,
            "instance_id": a.task,
            "max_minutes": cap_minutes,
            "seed": a.seed,
            "clock_mode": clock_mode,
            "tokens_per_frame": tokens_per_frame,
            "world_time_s": None,
            "run_status": "invalid",
            "invalid_run": {"category": "runner", "reason": "setup_error", "detail": error},
            "outcome": "error",
            "error_category": "runner",
            "error_reason": "setup_error",
            "failure_attribution": "runner",
            "solved": None,
            "decisions": 0,
            "actions_executed": 0,
            "stop_reason": "setup_error",
            "loop_stop_reason": None,
            "termination": None,
            "verification": None,
            "error": error,
            "cost_usd": None,
            "cost_known_usd": None,
            "cost_status": cost_accounting["status"],
            "cost_source": cost_accounting["source"],
            "cost_accounting": cost_accounting,
            **run_metrics,
        },
    )


def _hard_exit(code: int):
    """Exit without finalizing Python while timed-out native workers may still run.

    The normal teardown first drains workers within a bounded interval and
    persists results. This fallback avoids unwinding live native calls during
    interpreter shutdown; flush remaining output before exiting."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    try:
        logging.shutdown()  # flush configured log handlers
    except Exception:
        pass
    os._exit(code)


if __name__ == "__main__":
    # The episode entry point execs this process, so interrupts reach the runner directly.
    _report_parser = argparse.ArgumentParser(add_help=False)
    _report_parser.add_argument("--report-root", type=Path)
    _report_parser.add_argument("--report-output", type=Path)
    _report_args, _ = _report_parser.parse_known_args()
    _interrupted_signal = 2
    if _report_args.report_root:
        import signal

        def _interrupt_episode(signum, _frame):
            global _interrupted_signal
            _interrupted_signal = signum
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _interrupt_episode)
    _code = 0
    try:
        main()
    except SystemExit as e:  # main() raises SystemExit(1) for an unscored benchmark cell
        _code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except KeyboardInterrupt:
        # main()'s cleanup closed the recorder, the agent and its trace, but the episode froze no
        # snapshot, released no held input and wrote no run summary: an interrupted episode is
        # never a scored result.
        print(
            "INTERRUPTED — episode did not complete",
            flush=True,
        )
        _code = 128 + _interrupted_signal
    except Exception:
        _error = traceback.format_exc()
        print("FATAL SETUP/RUN ERROR\n" + _error, flush=True)
        try:
            _write_emergency_setup_summary(_error)
        except Exception:
            print("Could not write emergency run summary:\n" + traceback.format_exc(), flush=True)
        _code = 1
    finally:
        if _report_args.report_root:
            try:
                R.stop_containers()
            except Exception:
                print("DESKTOP CLEANUP ERROR\n" + traceback.format_exc(), flush=True)
                _code = _code or 1
            try:
                from realtime_framework.evaluation.reporting import finish_episode_report

                _report_code = finish_episode_report(
                    _report_args.report_root, _report_args.report_output
                )
                _code = _code or _report_code
            except Exception:
                print("REPORT ERROR\n" + traceback.format_exc(), flush=True)
                _code = _code or 1
    # os._exit rather than re-raising: see _hard_exit. The traceback above is already printed in
    # full, so routing it through the default excepthook would only duplicate it.
    _hard_exit(_code)
