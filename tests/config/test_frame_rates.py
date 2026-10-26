"""The environment and model-input frame rates may differ; a model's video cap bounds only the
model-input rate, ModelConfig.fps (providers/preflight.py check_fps).

runner.main() takes the two as --fps and --model-input-fps. The effective-config export's runner
probe (evaluation/effective_config.py) runs main() in a fresh interpreter up to its decision loop
and records where each rate went.
"""

from __future__ import annotations

import json
import os

import pytest

import realtime_framework.providers.registry as catalog
import realtime_tasks.registry as tasks
from realtime_framework.evaluation.effective_config import _runner_entry, export
from realtime_framework.providers.preflight import check_episode

GEMINI = "gemini-3.8-live"
WEB, GAME, CAPPED = "rt_waitlist_fit", "gh_timed_chess", "rt_accessible_trip_dispatch"
CAP_MESSAGE = "gemini_live supports at most 1 fps; requested 2"


def gemini_config(model_input_fps: float):
    effort = catalog.get(GEMINI).defaults["reasoning_effort"]
    return catalog.to_model_config(GEMINI, fps=model_input_fps, reasoning_effort=effort)


def test_video_cap_reads_the_model_input_rate():
    check_episode(gemini_config(1.0), clock_mode="wall", task=tasks.get(WEB), env_fps=2.0)
    with pytest.raises(ValueError) as caught:
        check_episode(gemini_config(2.0), clock_mode="wall", task=tasks.get(WEB), env_fps=2.0)
    assert str(caught.value) == CAP_MESSAGE


@pytest.fixture(scope="module")
def runner_cells(tmp_path_factory):
    """Gemini Live through runner.main(): on the wall clock at 2/1, on the token clock at 60/1
    with B=100, and at 2/2, which its cap refuses. Each cell is one task of the probe's run; the
    arguments are those _trial passes (evaluation/_trial.py main)."""
    wall = ["--clock", "wall", "--protocol", "wall-clock"]
    token = ["--clock", "token", "--tokens-per-frame", "100", "--protocol", "time-aware-token-v1"]
    runs = [
        (WEB, [*wall, "--fps", "2", "--model-input-fps", "1"]),
        (GAME, [*token, "--fps", "60", "--model-input-fps", "1"]),
        (CAPPED, [*wall, "--fps", "2"]),
    ]
    effort = catalog.get(GEMINI).defaults["reasoning_effort"]
    environ = {"HOME": os.environ.get("HOME", "/"), "PATH": "/usr/bin:/bin"}
    exported = _runner_entry(
        environ,
        {},
        [
            (
                GEMINI,
                task,
                ["--model", GEMINI, "--task", task, *argv, "--reasoning-effort", effort]
                + ["--report-root", "<results>"],
            )
            for task, argv in runs
        ],
        env_file=None,
        scratch=tmp_path_factory.mktemp("runner"),
    )
    return exported["models"][GEMINI]


def test_environment_rate_above_the_cap_starts(runner_cells):
    cell = runner_cells[WEB]
    assert cell["error"] is None
    assert (cell["run"]["fps"], cell["run"]["model_input_fps"]) == (2.0, 1.0)
    # The environment: the task clock, GuestEnvironment's tick, and the loop's two rates and its
    # horizon in world seconds.
    assert cell["task"]["task_clock"]["fps"] == 2.0
    assert cell["guest_environment"]["tick_seconds"] == 0.5
    loop = cell["run_stream"]
    assert (loop["env_fps"], loop["input_fps"], loop["duration_s"]) == (2.0, 1.0, 300.0)
    # The model: its config, and the rate its prompt names.
    assert cell["frame_rates"] == {"env_fps": 2.0, "model_input_fps": 1.0}
    assert cell["runtime_config"]["fps"] == 2.0
    assert cell["clock_prompt"].startswith("WALL CLOCK: observations are scheduled at 1 fps. ")
    # Where the rates differ the note also says what the ticks it shows count.
    assert "at the first environment tick (2 per second) at or after" in cell["clock_prompt"]
    # The identity the report expects is computed from the same two rates.
    assert cell["expected_runtime_config_hash"] == cell["runtime_config_hash"]


def test_token_clock_is_set_up_with_b_per_model_input_frame(runner_cells):
    """What main() hands the loop, the prompt and the task clock; the charging itself is
    test_a_token_run_summary_charges_b_per_model_input_frame's."""
    cell = runner_cells[GAME]
    assert cell["error"] is None
    loop = cell["run_stream"]
    assert (loop["env_fps"], loop["input_fps"], loop["tokens_per_frame"]) == (60.0, 1.0, 100)
    assert cell["frame_rates"] == {"env_fps": 60.0, "model_input_fps": 1.0}
    prompt = cell["clock_prompt"]
    assert "observations are scheduled at f=1 frames per simulated second" in prompt
    assert "at the first environment tick (60 per simulated second) at or after" in prompt
    assert "kappa=B*f=100 charged tokens per simulated second" in prompt
    # The task clock counts environment ticks: B per tick, the same kappa.
    task_clock = cell["task"]["task_clock"]
    assert (task_clock["fps"], task_clock["tokens_per_world_second"]) == (60.0, 100.0)
    assert cell["expected_runtime_config_hash"] == cell["runtime_config_hash"]


def test_pause_mode_steps_to_the_next_frame_where_the_rates_differ():
    """At 30/7 frames are due every 30/7 ticks and observed at whole ticks, so after an action
    the world moves 5 or 4 ticks, never exactly 1/7 s; at equal rates the note is unchanged."""
    from realtime_framework.harnesses.default.prompts import PromptOptions, budget_clock_prompt

    pause = PromptOptions("turn_based", "token", False)
    split = budget_clock_prompt(7.0, None, pause, env_fps=30.0)
    assert "The environment then advances to the next scheduled frame, and" in split
    assert "1/f seconds" not in split
    equal = budget_clock_prompt(7.0, None, pause, env_fps=7.0)
    assert equal == budget_clock_prompt(7.0, None, pause)
    assert "advances exactly one frame interval (1/f seconds), and" in equal


def test_model_input_rate_above_the_cap_is_refused(runner_cells):
    """--model-input-fps defaults to --fps; the runner refuses the rate as an argument."""
    cell = runner_cells[CAPPED]
    assert cell["error"] == f"_Rejected: {CAP_MESSAGE}"


WALL_2_1 = ["--clock", "wall", "--fps", "2", "--model-input-fps", "1"]
TOKEN_60_1 = ["--clock", "token", "--tokens-per-frame", "100", "--protocol", "time-aware-token-v1"]
TOKEN_60_1 += ["--fps", "60", "--model-input-fps", "1"]


def _main_in_process(monkeypatch, tmp_path, argv, run_stream) -> list:
    """Set up runner.main() to run in this interpreter on ``argv`` with ``run_stream`` as its
    decision loop and the export probe's stand-ins for the container, the guest, the scene
    capture, the agent and the task server (evaluation/effective_config.py _Guest, _Scene, _Agent,
    _Task). There is no screen recording to copy out, so main() composes the video from the
    frames captured once per decision (sandbox/desktop.py _stop_screen_recording); the returned
    list collects the rates it is played at. The run writes under ``tmp_path / "run"``."""
    import sys
    import types

    from realtime_framework.clock import Clocks, FrameClock, SystemClock
    from realtime_framework.evaluation import effective_config as probe
    from realtime_framework.evaluation import reporting, runner
    from realtime_framework.sandbox import desktop

    for name in ("build_image_if_needed", "start_container", "_start_screensaver_keepalive"):
        monkeypatch.setattr(desktop, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(desktop, "_start_screen_recording", lambda run_dir=None: True)

    def missing_recording(*args, **kwargs):
        raise FileNotFoundError("no container")

    monkeypatch.setattr(desktop, "subprocess", types.SimpleNamespace(run=missing_recording))
    videos = []
    monkeypatch.setattr(
        desktop, "make_video", lambda frames_dir, output_path, fps: videos.append(fps)
    )
    monkeypatch.setattr(reporting, "finalize_video", lambda run_dir: None)

    class NoSleeps(SystemClock):
        """main()'s setup waits take no time here."""

        def sleep(self, seconds):
            pass

    clock = NoSleeps()

    def clocks(clock_mode, fps):
        """As clock.episode_clocks, with no setup waits."""
        return Clocks(FrameClock(fps) if clock_mode == "token" else clock, clock)

    monkeypatch.setattr(runner, "episode_clocks", clocks)
    monkeypatch.setattr(runner, "GuestClient", probe._Guest)
    monkeypatch.setattr(runner, "BrowserSceneCapture", probe._Scene)
    monkeypatch.setattr(runner, "create_agent", lambda config, **kwargs: probe._Agent(config))
    registered = tasks.get
    monkeypatch.setattr(tasks, "get", lambda task_id: probe._Task(registered(task_id)))
    monkeypatch.setattr(runner, "run_stream", run_stream)
    for name in ("SANDBOX_GRAPHICS_MODE", "SCREEN_RECORDING_ENCODER"):
        monkeypatch.setenv(name, "cpu")  # main() sets both
    argv = [*argv, "--results-dir", str(tmp_path), "--run-id", "run"]
    monkeypatch.setattr(sys, "argv", ["runner", *argv])
    return videos


def _run_summary(tmp_path) -> dict:
    return json.loads((tmp_path / "run" / "run_summary.json").read_text())


def test_fallback_recording_plays_at_the_model_input_rate(tmp_path, monkeypatch):
    """A run error in the decision loop reaches main()'s cleanup, which stops the screen
    recording; the composed video plays at the model-input rate."""
    from realtime_framework.evaluation import runner

    def run_stream(adapter, env, **loop):
        raise RuntimeError("decision loop failed")

    argv = ["--model", GEMINI, "--task", WEB, *WALL_2_1]
    videos = _main_in_process(monkeypatch, tmp_path, argv, run_stream)
    with pytest.raises(SystemExit) as exited:  # an invalid run is not a scored result
        runner.main()
    assert exited.value.code == 1
    summary = _run_summary(tmp_path)
    assert summary["error"].strip().endswith("RuntimeError: decision loop failed")
    assert (summary["fps"], summary["model_input_fps"]) == (2.0, 1.0)
    assert videos == [1.0]


def test_fallback_recording_after_a_finished_loop_plays_at_the_model_input_rate(
    tmp_path, monkeypatch
):
    """When the decision loop returns, main() stops the screen recording right after it, not in
    its cleanup; that video too plays at the model-input rate."""
    from realtime_framework.evaluation import runner

    argv = ["--model", GEMINI, "--task", WEB, *WALL_2_1]
    videos = _main_in_process(monkeypatch, tmp_path, argv, lambda adapter, env, **loop: [])
    with pytest.raises(SystemExit):  # the stand-in task has no snapshot to score
        runner.main()
    assert _run_summary(tmp_path)["error"] is None
    assert videos == [1.0]


def test_a_token_run_summary_charges_b_per_model_input_frame(tmp_path, monkeypatch):
    """60/1 with B=100: one decision of 50 tokens is half a model-input frame, 0.5 s, 30 ticks.
    The run summary's kappa and simulated latency are per model-input frame."""
    from realtime_framework.evaluation import runner

    def run_stream(adapter, env, **loop):
        env.record_lifecycle({"decision": 0, "clock_output_tokens": 50})
        env._termination_snapshot_captured = True
        env._termination_snapshot = {"kind": "realtime", "state": {"generation": 0}}
        d_ticks = 50 / 100 * loop["env_fps"] / loop["input_fps"]
        return [(0, 0, d_ticks, d_ticks, None)]

    argv = ["--model", GEMINI, "--task", WEB, *TOKEN_60_1]
    videos = _main_in_process(monkeypatch, tmp_path, argv, run_stream)
    runner.main()
    summary = _run_summary(tmp_path)
    assert summary["invalid_run"] is None
    assert (summary["fps"], summary["model_input_fps"]) == (60.0, 1.0)
    assert summary["tokens_per_world_second"] == 100
    assert summary["avg_delta_ticks"] == 30.0
    latency = summary["decision_simulated_latency_s"]
    assert (latency["sample_count"], latency["mean"]) == (1, 0.5)
    assert videos == [1.0]


def test_main_hands_the_loop_a_model_input_frame_between_two_ticks(tmp_path, monkeypatch):
    """At 30/7 a model-input frame is 30/7 environment ticks. main() hands run_stream the two
    rates as they are, and the kernel computes frame times from them (kernel/turns.py _frames,
    _ticks); the ratios of runner_cells are whole. Claude has no video cap, unlike Gemini Live."""
    from realtime_framework.evaluation import runner

    loops = []

    def run_stream(adapter, env, **loop):
        loops.append(loop)
        return []

    token = ["--clock", "token", "--tokens-per-frame", "100", "--protocol", "time-aware-token-v1"]
    argv = ["--model", "claude-opus-5", "--task", WEB, *token, "--fps", "30"]
    _main_in_process(monkeypatch, tmp_path, [*argv, "--model-input-fps", "7"], run_stream)
    with pytest.raises(SystemExit):  # the stand-in task has no snapshot to score
        runner.main()
    (loop,) = loops
    assert (loop["env_fps"], loop["input_fps"], loop["duration_s"]) == (30.0, 7.0, 300.0)


def test_main_refuses_a_model_input_rate_above_the_environment_rate(
    tmp_path, monkeypatch, capsys
):
    """The pre-run check's error (providers/preflight.py check_frame_rates), which main() makes
    before it starts the container (runner.py main), is an argument error: no container is
    started, no agent is made, the decision loop never starts and no run summary is written."""
    from realtime_framework.evaluation import runner
    from realtime_framework.sandbox import desktop

    loops = []

    def run_stream(adapter, env, **loop):
        loops.append(loop)
        return []

    argv = ["--model", "claude-opus-5", "--task", WEB, "--clock", "wall", "--fps", "1"]
    _main_in_process(monkeypatch, tmp_path, [*argv, "--model-input-fps", "2"], run_stream)
    started, created = [], []
    monkeypatch.setattr(desktop, "start_container", started.append)
    create_agent = runner.create_agent

    def recording_create_agent(config, **kwargs):
        created.append(config)
        return create_agent(config, **kwargs)

    monkeypatch.setattr(runner, "create_agent", recording_create_agent)
    with pytest.raises(SystemExit) as exited:
        runner.main()
    assert exited.value.code == 2
    assert capsys.readouterr().err.strip().endswith(
        "error: The model-input frame rate (2 fps) exceeds the environment frame rate (1 fps)"
    )
    assert not (tmp_path / "run").exists()
    assert started == created == loops == []


def test_a_setup_failure_summary_records_both_rates(tmp_path, monkeypatch):
    """A setup error main() raises reaches the entry point's emergency summary
    (runner._write_emergency_setup_summary), which reads the two rates off the command line."""
    import sys

    from realtime_framework.evaluation import runner

    argv = ["--model", GEMINI, "--task", WEB, *TOKEN_60_1]
    argv += ["--results-dir", str(tmp_path), "--run-id", "run"]
    monkeypatch.setattr(sys, "argv", ["runner", *argv])
    runner._write_emergency_setup_summary("RuntimeError: guest not ready")
    summary = _run_summary(tmp_path)
    assert (summary["stop_reason"], summary["run_status"]) == ("setup_error", "invalid")
    assert (summary["fps"], summary["model_input_fps"]) == (60.0, 1.0)
    assert (summary["clock_mode"], summary["tokens_per_frame"]) == ("token", 100)


PLAN = """\
suite: public
models:
  - gemini-3.8-live
runtime:
  seed: 42
  mode: default
  clock_mode: wall
  fps: 2
  model_input_fps: 1
  duration_minutes: 5
  protocol: wall-clock
"""


def test_a_run_plan_carries_both_rates_to_the_runner(tmp_path):
    """``rtbench run -c PLAN`` as the effective-config export follows it: the plan's runtime
    becomes the Harbor agent's options (evaluation/harbor_cli.py plan_job), the export's copies of
    the trial's identity and runner arguments (evaluation/effective_config.py), the trial's plan
    and model config (evaluation/_trial.py), and the runner's arguments, which reach its decision
    loop. test_a_harbor_trial_carries_both_rates_to_the_runner runs Harbor's own copies."""
    plan = tmp_path / "split.yaml"
    plan.write_text(PLAN)
    entry = f"harbor:{plan}"
    environ = {"HOME": os.environ.get("HOME", "/"), "PATH": "/usr/bin:/bin"}
    exported = export(
        environ=environ, env_file="/nonexistent/.env", entries=[entry], models=[GEMINI],
        tasks=[WEB],
    )
    cell = exported["entries"][entry]["models"][GEMINI]
    assert cell["error"] is None
    harbor = cell["harbor"]
    assert (harbor["harbor_request"]["fps"], harbor["harbor_request"]["model_input_fps"]) == (2, 1)
    assert (harbor["task_json"]["fps"], harbor["task_json"]["model_input_fps"]) == (2, 1)
    argv = harbor["trial_argv"]
    assert argv[argv.index("--fps") + 1] == "2.0"
    assert argv[argv.index("--model-input-fps") + 1] == "1.0"
    runtime = cell["trial"]["settings"]["plan"]["runtime"]
    assert (runtime["fps"], runtime["model_input_fps"]) == (2, 1)
    assert cell["trial"]["settings"]["model_config"]["fps"] == 1.0
    assert cell["frame_rates"] == {"env_fps": 2.0, "model_input_fps": 1.0}
    assert (cell["run_stream"]["env_fps"], cell["run_stream"]["input_fps"]) == (2.0, 1.0)
    assert cell["expected_runtime_config_hash"] == cell["runtime_config_hash"]


@pytest.fixture
def environment():
    """_trial.main writes os.environ (evaluation/_trial.py)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_a_harbor_trial_carries_both_rates_to_the_runner(environment, tmp_path, monkeypatch):
    """``rtbench run -c PLAN`` through Harbor's environment: it fixes the trial's identity
    (ApptainerEnvironment.resolve), writes it as the verifier's task.json and launches the trial
    worker (run_benchmark), which execs the runner (evaluation/_trial.py main). The worker runs in
    this interpreter, as in tests/replay/test_external_driver.py resolved_request; the exec is
    captured."""
    import asyncio
    import contextlib
    import io

    from harbor.environments.base import ExecResult

    from realtime_framework.evaluation import _trial, harbor, harbor_cli

    plan = tmp_path / "split.yaml"
    plan.write_text(PLAN)
    # suite_dataset links task folders under temp/; the job only needs the selected names.
    monkeypatch.setattr(
        harbor_cli, "suite_dataset", lambda suite, selected: {"path": suite, "task_names": selected}
    )
    (agent,) = harbor_cli.plan_job(plan)["agents"]
    options = harbor.RealtimeBenchOptions(**agent["kwargs"])
    # RealtimeBenchAgent.run: a preset named only by -m becomes the catalog model.
    request = {"model": GEMINI, "mode": "default", **options.model_dump(), "catalog_model": GEMINI}

    images = {kind: tmp_path / f"{kind}.sif" for kind in ("sandbox", "realtime")}
    for variable, image in zip(("SANDBOX_IMAGE", "RT_SUITE_IMAGE"), images.values()):
        image.touch()
        monkeypatch.setenv(variable, str(image))
    suite = {
        "suite_id": "public", "suite_version": "1.0", "suite_hash": "0" * 64,
        "harness_version": "1.0", "harness_hash": "1" * 64,
        "images": {kind: str(path) for kind, path in images.items()},
        "tasks": [{"task_id": WEB, "category": "realtime"}],
    }  # fmt: skip
    monkeypatch.setattr(harbor, "runtime", lambda root, python, suite_id: suite)
    # The worker allocates its run directory under its ROOT, and execs from there.
    suite_path = tasks.suite_path
    monkeypatch.setattr(
        tasks, "suite_path", lambda root, name: suite_path(catalog.PROJECT_ROOT, name)
    )
    monkeypatch.setattr(_trial, "ROOT", tmp_path / "trial-root")
    monkeypatch.chdir(os.getcwd())
    monkeypatch.setattr(_trial.shutil, "which", lambda name: "/usr/bin/" + name)

    class Exec(Exception):
        pass

    def execve(path, argv, env):
        raise Exec(argv)

    monkeypatch.setattr(_trial.os, "execve", execve)
    runner_argv = []

    async def run(command, **options):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                code = _trial.main(command[3:])  # after "python -m ...evaluation._trial"
            except Exec as exc:
                runner_argv.append(exc.args[0])
                code = 0
        return ExecResult(return_code=code, stdout=out.getvalue(), stderr="")

    trial = object.__new__(harbor.ApptainerEnvironment)
    trial.root, trial.python, trial.lane = catalog.PROJECT_ROOT, "python", 1
    trial.task_id = WEB
    trial.staging = (tmp_path / "staging").resolve()
    for folder in ("workspace", "tests", "logs/agent", "logs/artifacts"):
        (trial.staging / folder).mkdir(parents=True)
    trial._launched = False
    trial._run = run
    trial._runner_environment = lambda extra_env: {}
    resolved = asyncio.run(trial.resolve(request))
    assert (resolved["task"]["fps"], resolved["task"]["model_input_fps"]) == (2.0, 1.0)
    asyncio.run(trial.run_benchmark(resolved))
    task = json.loads((trial.staging / "tests" / "task.json").read_text())
    assert (task["fps"], task["model_input_fps"]) == (2.0, 1.0)
    (argv,) = runner_argv
    assert argv[argv.index("--fps") + 1] == "2.0"
    assert argv[argv.index("--model-input-fps") + 1] == "1.0"


def test_ak_fps_on_a_run_plan_without_a_model_input_rate_sets_both(tmp_path, monkeypatch):
    """``rtbench run -c PLAN --ak fps=N``: Harbor merges --ak into each agent's kwargs key by key
    (harbor/cli/jobs.py:1575-1577). A plan that does not set model_input_fps leaves it to the
    agent's options, which default it to the fps they end up with, as without a plan; a plan
    that sets it keeps it."""
    from harbor.cli.utils import parse_kwargs

    from realtime_framework.evaluation import harbor_cli
    from realtime_framework.evaluation.harbor import RealtimeBenchOptions

    # suite_dataset links task folders under temp/; the job only needs the selected names.
    monkeypatch.setattr(
        harbor_cli, "suite_dataset", lambda suite, selected: {"path": suite, "task_names": selected}
    )

    def rates(plan, *ak):
        (agent,) = harbor_cli.plan_job(plan)["agents"]
        agent["kwargs"].update(parse_kwargs(list(ak)))
        options = RealtimeBenchOptions(**agent["kwargs"])
        return options.fps, options.model_input_fps

    default = harbor_cli.ROOT / "configs/runs/default.yaml"
    assert rates(default) == (1.0, 1.0)
    assert rates(default, "fps=2") == (2.0, 2.0)
    split = tmp_path / "split.yaml"
    split.write_text(PLAN)
    assert rates(split) == (2.0, 1.0)
    assert rates(split, "fps=4") == (4.0, 1.0)


def _trial(model_input_fps, fps=2.0) -> tuple[dict, dict, dict]:
    """A trial's identity (evaluation/harbor.py ApptainerEnvironment.resolve), its request, and
    the one-task local report whose group carries the identity reporting.report found in the run
    summary."""
    task = {
        "suite_id": "public", "suite_version": "1.0", "suite_hash": "0" * 64,
        "harness_version": "1.0", "harness_hash": "1" * 64, "task_id": WEB,
        "category": "web", "fps": fps, "duration_minutes": 5.0, "seed": 42,
    }
    if model_input_fps is not None:
        task["model_input_fps"] = model_input_fps
    request = {
        "model": GEMINI, "mode": "default", "model_revision": None, "clock": "wall",
        "task": task, "suite_tasks": {WEB: "web"},
    }
    group = {
        **{field: task[field] for field in ("suite_id", "suite_version", "suite_hash")},
        "harness_hash": task["harness_hash"], "fps": task["fps"],
        "model_input_fps": model_input_fps, "seed": 42, "public_category": "web",
        "mode": "default", "protocol": "wall-clock", "clock_mode": "wall",
        "tokens_per_frame": None, "cap": 5.0, "expected": 1, "scored": 1, "invalid": 0,
        "missing": 0, "partial": False, "runtime_config_hash": "2" * 64,
        "tasks": [{"task_id": WEB, "status": "scored", "solved": True, "outcome": "pass"}],
    }
    report = {
        "integration": {
            "model": GEMINI, "model_revision": None, "mode": "default",
            "configuration_sha256": "3" * 64,
        },
        "groups": [group],
    }
    return task, request, report


def test_harbor_verify_compares_the_model_input_rate():
    from realtime_framework.evaluation.harbor import verify

    task, request, report = _trial(1.0)
    assert verify(report, task, request) == {"reward": 1.0}
    report["groups"][0]["model_input_fps"] = 2.0
    with pytest.raises(ValueError, match="Report model_input_fps mismatch"):
        verify(report, task, request)


def test_the_identity_a_local_report_expects_has_both_rates(tmp_path):
    """runner.main() with --report-root records the identity its report expects before the run
    (reporting.prepare_episode_report), from the run options main() runs with."""
    from realtime_framework.evaluation.reporting import prepare_episode_report
    from realtime_framework.evaluation.run_spec import RunSpec

    prepare_episode_report(
        RunSpec(clock_mode="wall", fps=2.0, model_input_fps=1.0), model=GEMINI, task=WEB,
        reasoning_effort=catalog.get(GEMINI).defaults["reasoning_effort"],
        results_dir=str(tmp_path),
    )
    (run,) = json.loads((tmp_path / "summary.json").read_text())["runs"]
    assert (run["fps"], run["model_input_fps"]) == (2.0, 1.0)


def test_a_run_summary_check_compares_the_model_input_rate(tmp_path):
    """reporting.summary_ok, as tests/config/test_interaction_mode.py checks the mode."""
    from realtime_framework.evaluation.reporting import summary_ok

    path = tmp_path / "run_summary.json"
    path.write_text(json.dumps({"stop_reason": "terminal", "fps": 2.0, "model_input_fps": 1.0}))
    assert summary_ok(path, expected_fps=2.0, expected_model_input_fps=1.0)
    assert not summary_ok(path, expected_fps=2.0, expected_model_input_fps=2.0)


def test_a_local_report_checks_each_run_for_both_rates(tmp_path, monkeypatch):
    """reporting.report hands each run summary's check the two rates the run expected."""
    from realtime_framework.evaluation import reporting

    checks = []
    monkeypatch.setattr(
        reporting, "summary_ok", lambda path, **expected: checks.append(expected) or False
    )
    cell = tmp_path / GEMINI
    (cell / WEB).mkdir(parents=True)
    expected = {"task_id": WEB, "fps": 2.0, "model_input_fps": 1.0}
    (cell / "summary.json").write_text(json.dumps({"runs": [expected]}))
    (cell / WEB / "run_summary.json").write_text(json.dumps({"stop_reason": "terminal"}))
    (group,) = reporting.report(tmp_path)["groups"]
    assert (group["fps"], group["model_input_fps"]) == (2.0, 1.0)
    (check,) = checks
    assert (check["expected_fps"], check["expected_model_input_fps"]) == (2.0, 1.0)


def test_a_reevaluation_compares_the_model_input_rate(tmp_path, monkeypatch):
    """reevaluate --replace-summary refuses a run summary whose identity is not the saved
    artifact's (reporting.reevaluate_main); the re-evaluation itself is stubbed."""
    import sys

    from realtime_framework.evaluation import reporting

    identity = {"suite_id": "public", "task_id": WEB, "fps": 2.0, "model_input_fps": 1.0}
    evaluation = {"resolved": True, "evaluation_version": 1}
    monkeypatch.setattr(reporting, "reeval", lambda run_dir, **options: (evaluation, identity))
    monkeypatch.setattr(sys, "argv", ["reevaluate", str(tmp_path), "--replace-summary"])
    summary = tmp_path / "run_summary.json"
    summary.write_text(json.dumps({**identity, "model_input_fps": 2.0, "stop_reason": "terminal"}))
    with pytest.raises(RuntimeError, match="identity mismatch for model_input_fps"):
        reporting.reevaluate_main()
    summary.write_text(
        json.dumps({**identity, "stop_reason": "terminal", "benchmark_eligible": True})
    )
    reporting.reevaluate_main()
    assert json.loads(summary.read_text())["solved"] is True


@pytest.mark.parametrize(
    "rates", [["--fps", "2", "--model-input-fps", "1"], ["--fps", "1"]], ids=["2/1", "1"]
)
def test_identity_main_prints_the_hash_runner_main_records(tmp_path, monkeypatch, capsys, rates):
    """identity_main prints the runtime-config hash runner.main() records in run_config.json for
    the same arguments, with neither the protocol, the reasoning effort, the model-input rate
    nor the desktop's graphics mode and recording encoder given: both take the runner's
    defaults (config.apply_run_defaults)."""
    from realtime_framework.evaluation import runner
    from realtime_framework.evaluation.config import identity_main

    argv = ["--model", GEMINI, "--task", WEB, "--clock", "wall", "--max-minutes", "5", *rates]
    _main_in_process(monkeypatch, tmp_path, argv, lambda adapter, env, **loop: [])
    for name in ("SANDBOX_GRAPHICS_MODE", "SCREEN_RECORDING_ENCODER"):
        monkeypatch.delenv(name)
    assert identity_main(argv) == 0
    printed = capsys.readouterr().out.strip()
    with pytest.raises(SystemExit):  # the stand-in task has no snapshot to score
        runner.main()
    recorded = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert printed == recorded["runtime_config_hash"]


def test_the_hash_a_local_report_expects_is_the_one_the_run_records(tmp_path, monkeypatch):
    """runner.main() with --report-root records the identity its report expects before the run
    (reporting.prepare_episode_report) from its arguments as it runs them: without --protocol
    and --reasoning-effort the expected hash is the one the run records in run_config.json."""
    from realtime_framework.evaluation import runner
    from realtime_framework.sandbox import apptainer

    monkeypatch.setattr(apptainer, "build_image", lambda name: None)
    argv = ["--model", GEMINI, "--task", WEB, *WALL_2_1, "--report-root", str(tmp_path)]
    _main_in_process(monkeypatch, tmp_path, argv, lambda adapter, env, **loop: [])
    with pytest.raises(SystemExit):  # the stand-in task has no snapshot to score
        runner.main()
    (expected,) = json.loads((tmp_path / "summary.json").read_text())["runs"]
    recorded = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert expected["runtime_config_hash"] == recorded["runtime_config_hash"]
    effort = catalog.get(GEMINI).defaults["reasoning_effort"]
    assert (expected["protocol"], expected["reasoning_effort"]) == ("wall-clock", effort)


def _job(path, request):
    trial = path / f"{WEB}__trial" / "agent"
    trial.mkdir(parents=True)
    (trial / "request.json").write_text(json.dumps(request))
    return path


def test_jobs_from_before_the_split_report_the_environment_rate(tmp_path):
    """A job recorded with one --fps has no model_input_fps; its model ran at that rate."""
    from realtime_framework.evaluation.harbor import aggregate

    report = aggregate(_job(tmp_path, _trial(None)[1]))
    assert (report["fps"], report["model_input_fps"]) == (2.0, 2.0)


def test_local_reports_on_runs_from_before_the_split_read_the_environment_rate(tmp_path):
    """The same for a local report: a run recorded with one --fps is grouped with runs recorded
    since at that rate for both, and its run summary passes a check that expects both."""
    from realtime_framework.evaluation.reporting import report, summary_ok

    cell = tmp_path / GEMINI
    cell.mkdir()
    runs = [{"task_id": WEB, "fps": 2.0}, {"task_id": GAME, "fps": 2.0, "model_input_fps": 2.0}]
    (cell / "summary.json").write_text(json.dumps({"runs": runs}))
    (group,) = report(tmp_path)["groups"]
    assert (group["fps"], group["model_input_fps"]) == (2.0, 2.0)
    path = tmp_path / "run_summary.json"
    path.write_text(json.dumps({"stop_reason": "terminal", "fps": 2.0}))
    assert summary_ok(path, expected_fps=2.0, expected_model_input_fps=2.0)
    assert not summary_ok(path, expected_fps=2.0, expected_model_input_fps=1.0)


def test_a_leaderboard_submission_needs_both_rates_at_one(tmp_path):
    """rtbench upload's check of three jobs (evaluation/submission.py check_runs); each one's trial
    has no report, which counts as a failed task."""
    from realtime_framework.evaluation.submission import SubmissionError, check_runs

    def jobs(name, model_input_fps):
        request = _trial(model_input_fps, fps=1.0)[1]
        return [_job(tmp_path / name / f"run{run}", request) for run in (1, 2, 3)]

    assert len(check_runs(jobs("one", 1.0))) == 3
    assert len(check_runs(jobs("before-the-split", None))) == 3
    with pytest.raises(SubmissionError) as caught:
        check_runs(jobs("split", 2.0))
    assert str(caught.value) == f"{tmp_path / 'split' / 'run1'}: model_input_fps is 2.0, expected 1"
