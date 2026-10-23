"""The interaction mode, turn_based or streaming, is a run setting like the frame rates.

runner.main() takes it as --interaction-mode, records it in the run identity (the episode start,
run_config.json and its hash, the expected summary and the run summary) and passes it to the
pre-run checks (providers/preflight.py check_interaction_mode). Run plans, the Harbor agent's
options, each trial's task.json and the trial worker carry it to the runner. Both live drivers
declare streaming (OpenAI Realtime, Gemini Live), and a streaming run of either runs on the
native-streaming agent; any other API, or the token clock, refuses it before the agent is built.
"""

from __future__ import annotations

import hashlib
import json
import os
import types

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.evaluation.effective_config import _runner_entry, export

GEMINI = "gemini-3.8-live"
WEB, GAME = "rt_waitlist_fit", "gh_timed_chess"


@pytest.fixture(scope="module")
def runner_cells(tmp_path_factory):
    """Gemini Live through runner.main() on the wall clock: by default, and asking for streaming.
    Each cell is one task of the probe's run, as in tests/config/test_frame_rates.py."""
    wall = ["--clock", "wall", "--protocol", "wall-clock"]
    runs = [(WEB, wall), (GAME, [*wall, "--interaction-mode", "streaming"])]
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


def test_the_runner_records_turn_based_by_default(runner_cells):
    cell = runner_cells[WEB]
    assert cell["error"] is None
    assert cell["run"]["interaction_mode"] == "turn_based"
    assert cell["runtime_config"]["interaction_mode"] == "turn_based"
    # The hash the report expects (reporting.prepare_episode_report) covers the same mode.
    assert cell["expected_runtime_config_hash"] == cell["runtime_config_hash"]


def test_the_runner_records_streaming(runner_cells):
    """Gemini Live declares streaming: the run records it, with the driver's streaming settings,
    and the report expects the same hash."""
    from realtime_framework.providers.gemini.live import GeminiLiveDriver

    cell = runner_cells[GAME]
    assert cell["error"] is None
    assert cell["run"]["interaction_mode"] == "streaming"
    assert cell["runtime_config"]["interaction_mode"] == "streaming"
    assert cell["runtime_config"]["streaming"] == GeminiLiveDriver.streaming_settings
    assert cell["expected_runtime_config_hash"] == cell["runtime_config_hash"]


def run_main(
    tmp_path, monkeypatch, *argv, agents=None, loops=None, clock_mode="wall", loop=None
) -> dict:
    """runner.main() in this interpreter with the export probe's stand-ins for the container, the
    guest, the scene capture, the agent and the task server (evaluation/effective_config.py), and a
    decision loop that fails at once, so the run ends invalid, or that is ``loop(env, trace)``
    and returns its timeline. The interaction mode each agent is
    built for is appended to ``agents``, and the one the loop is run in to ``loops``;
    ``clock_mode`` None leaves --clock out. Returns the run summary."""
    import sys

    import realtime_tasks.registry as tasks
    from realtime_framework.clock import Clocks, SystemClock
    from realtime_framework.evaluation import effective_config as probe
    from realtime_framework.evaluation import reporting, runner
    from realtime_framework.sandbox import apptainer, desktop

    for name in ("build_image_if_needed", "start_container", "_start_screensaver_keepalive"):
        monkeypatch.setattr(desktop, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(desktop, "_start_screen_recording", lambda run_dir=None: False)
    monkeypatch.setattr(apptainer, "build_image", lambda name: None)  # main() with --report-root
    monkeypatch.setattr(reporting, "finalize_video", lambda run_dir: None)
    agents = [] if agents is None else agents
    loops = [] if loops is None else loops

    def create_agent(config, **kwargs):
        agents.append(kwargs["interaction_mode"])
        return probe._Agent(config)

    class NoSleeps(SystemClock):
        """main()'s setup waits take no time here."""

        def sleep(self, seconds):
            pass

    clock = NoSleeps()
    monkeypatch.setattr(runner, "episode_clocks", lambda clock_mode, fps: Clocks(clock, clock))
    monkeypatch.setattr(runner, "GuestClient", probe._Guest)
    monkeypatch.setattr(runner, "BrowserSceneCapture", probe._Scene)
    monkeypatch.setattr(runner, "create_agent", create_agent)
    registered = tasks.get
    monkeypatch.setattr(tasks, "get", lambda task_id: probe._Task(registered(task_id)))

    def run_stream(adapter, env, **kwargs):
        loops.append(kwargs["interaction_mode"])
        if loop is None:
            raise RuntimeError("decision loop failed")
        return loop(env, kwargs["trace"])

    monkeypatch.setattr(runner, "run_stream", run_stream)
    for name in ("SANDBOX_GRAPHICS_MODE", "SCREEN_RECORDING_ENCODER"):
        monkeypatch.setenv(name, "cpu")  # main() sets both
    clock_argv = ["--clock", clock_mode] if clock_mode else []
    argv = ["--model", GEMINI, "--task", WEB, *clock_argv, *argv]
    argv += ["--results-dir", str(tmp_path), "--run-id", "run"]
    monkeypatch.setattr(sys, "argv", ["runner", *argv])
    with pytest.raises(SystemExit) as exited:  # an invalid run is not a scored result
        runner.main()
    assert exited.value.code == 1
    return json.loads((tmp_path / "run" / "run_summary.json").read_text())


def test_the_run_summary_records_the_mode(tmp_path, monkeypatch):
    summary = run_main(tmp_path, monkeypatch)
    assert summary["error"].strip().endswith("RuntimeError: decision loop failed")
    assert summary["interaction_mode"] == "turn_based"
    assert summary["runtime_config"]["interaction_mode"] == "turn_based"
    assert "streaming" not in summary


def test_a_streaming_run_summary_records_the_mode(tmp_path, monkeypatch):
    """The run's summary, of a loop that failed before any frame went in."""
    summary = run_main(tmp_path, monkeypatch, "--interaction-mode", "streaming")
    assert summary["error"].strip().endswith("RuntimeError: decision loop failed")
    assert summary["interaction_mode"] == "streaming"
    assert summary["runtime_config"]["interaction_mode"] == "streaming"
    assert summary["streaming"]["context_control"] == "approximate"
    assert summary["streaming"]["frames"]["pushed"] == 0


def test_a_streaming_trace_its_summary_cannot_read_invalidates_the_run_alone(
    tmp_path, monkeypatch
):
    """A trace record written in part, with the next one after it on its line
    (reporting.py EpisodeTrace.emit), keeps the trace's rows from being read (reporting.rows): the
    run is invalid for it, its run metrics count it unscored although its snapshot is solved, and
    the rest of its summary stands."""
    termination = {"stop_reason": "time_limit", "end_s": 300.0, "snapshot_s": 300.5,
                   "snapshot_delay_s": 0.5}  # fmt: skip

    def loop(env, trace):
        with trace.path.open("a") as cut:
            cut.write('{"event": "KernelRecord", "payload": ')
        env._stop_reason, env._termination = "time_limit", termination
        # The snapshot the shell froze for the scorer (streaming_shell.py _freeze).
        snapshot = {
            "state": {"ok": True}, "evaluation": {"resolved": True, "evaluation_version": 1},
        }  # fmt: skip
        env._termination_snapshot, env._termination_snapshot_captured = snapshot, True
        return []

    summary = run_main(tmp_path, monkeypatch, "--interaction-mode", "streaming", loop=loop)
    invalid = summary["invalid_run"]
    assert (invalid["category"], invalid["reason"]) == ("harness", "streaming_summary_failed")
    assert "JSONDecodeError" in invalid["detail"]
    assert (summary["stop_reason"], summary["termination"]) == ("invalid_run", termination)
    assert summary["verification"]["resolved"] is True and summary["solved"] is None
    assert (summary["task_success_rate"], summary["normalized_time_to_success"]) == (None, None)
    assert summary["streaming"] is None


@pytest.mark.parametrize(
    "interaction_mode, reason",
    [("turn_based", "no_inference_responses"), ("streaming", "not_accounted_streaming")],
)
def test_a_run_summary_says_why_its_cost_is_unknown(
    tmp_path, monkeypatch, interaction_mode, reason
):
    """The streaming shell accounts no reply's tokens (episode_records.account_tokens runs on the
    turn-based shell's), so a streaming run's cost is unknown for that, not for a lack of replies
    (low 2)."""
    summary = run_main(tmp_path, monkeypatch, "--interaction-mode", interaction_mode)
    assert (summary["cost_usd"], summary["cost_accounting"]["reason"]) == (None, reason)


@pytest.mark.parametrize(
    "interaction_mode, status", [("turn_based", "unavailable"), ("streaming", "not_applicable")]
)
def test_a_streaming_run_times_no_model_call(tmp_path, monkeypatch, interaction_mode, status):
    """A streaming session generates throughout, and its replies are no model calls to time
    (evaluation/clock.py summarize_model_timing); a turn-based run whose loop failed timed none
    (low 5)."""
    summary = run_main(tmp_path, monkeypatch, "--interaction-mode", interaction_mode)
    timing = summary["model_timing"]
    assert (timing["status"], timing["call_count"], summary["wall_time_s"]) == (status, 0, None)


def test_the_runner_refuses_streaming_on_its_default_clock(tmp_path, monkeypatch):
    """Without --clock the runner runs the token clock (runner.py main), on which streaming is
    refused. Without --report-root, which refuses it earlier (reporting.prepare_episode_report),
    the runner's own check refuses the run before it builds the agent (providers/preflight.py
    check_episode), so no live session is opened. A summary refused before its configuration was
    resolved has no context control."""
    agents = []
    summary = run_main(
        tmp_path, monkeypatch, "--interaction-mode", "streaming", agents=agents, clock_mode=None
    )
    assert summary["error"].strip().endswith("ValueError: Streaming requires a wall clock")
    assert (summary["clock_mode"], agents) == ("token", [])
    assert summary["streaming"]["context_control"] is None
    assert summary["streaming"]["frames"]["pushed"] == 0


@pytest.mark.parametrize("interaction_mode", ["turn_based", "streaming"])
def test_the_runner_builds_the_agent_and_runs_the_loop_of_its_mode(
    tmp_path, monkeypatch, interaction_mode
):
    """runner.set_up_agent builds the agent of the run's mode (harnesses/registry.py
    create_agent) and main() runs the loop in it (runner.py run_stream)."""
    agents, loops = [], []
    run_main(
        tmp_path, monkeypatch, "--interaction-mode", interaction_mode, agents=agents, loops=loops
    )
    assert (agents, loops) == ([interaction_mode], [interaction_mode])


@pytest.mark.parametrize(
    "interaction_mode, agent_class",
    [("turn_based", "DefaultHarness"), ("streaming", "NativeStreamingAgent")],
)
def test_each_mode_has_its_agent(interaction_mode, agent_class):
    """The agent's own interaction mode is the run's: the native-streaming agent's session
    driver is built for streaming."""
    from realtime_framework.harnesses.registry import create_agent

    config = catalog.to_model_config(GEMINI, fps=1.0, reasoning_effort="fixed").model_copy(
        update={"api_key": "test-key"}
    )
    agent = create_agent(config, interaction_mode=interaction_mode)
    assert (type(agent).__name__, agent.interaction_mode) == (agent_class, interaction_mode)
    if interaction_mode == "streaming":
        assert agent.driver.streaming is True


def test_a_streaming_episode_runs_on_the_streaming_shell(monkeypatch):
    """runner.run_stream hands a streaming episode its agent, environment, horizon, frame rates,
    frozen-stream timing and clock to the streaming shell, with the episode's clocks, log and
    trace, whose records the run's streaming section is read off; the shell's config refuses any
    clock but the wall clock. The episode has no turns."""
    from realtime_framework.evaluation import runner
    from realtime_framework.evaluation.kernel.streaming import StreamingConfig

    episodes = []
    monkeypatch.setattr(
        runner,
        "run_streaming_episode",
        lambda agent, env, config, **kwargs: episodes.append((agent, env, config, kwargs)),
    )
    clocks, trace = object(), object()

    def streaming(clock_mode):
        return runner.run_stream(
            "agent", "env", clock_mode=clock_mode, duration_s=300.0, env_fps=2.0, input_fps=1.0,
            log=print, trace=trace, clocks=clocks, interaction_mode="streaming",
        )

    assert streaming("wall") == []
    config = StreamingConfig(300.0, 2.0, 1.0, timing=runner.RUNNER_TIMING, clock_mode="wall")
    assert episodes == [
        ("agent", "env", config, {"clocks": clocks, "log": print, "trace": trace})
    ]
    with pytest.raises(ValueError, match="native streaming runs on the wall clock, not 'token'"):
        streaming("token")


def test_a_streaming_run_keeps_the_hash_it_expected(tmp_path, monkeypatch):
    """Where a driver declares streaming, the runner's run configuration hashes to the one
    reporting.prepare_episode_report expected before the run, the model YAML's streaming settings
    included: either side leaving the mode or the settings out would fail the run's report."""
    models = tmp_path / "models"
    models.mkdir()
    (models / f"{GEMINI}.yaml").write_text(
        (catalog.MODELS_DIR / f"{GEMINI}.yaml").read_text()
        + "streaming:\n  compression_trigger_tokens: 100000\n"
    )
    monkeypatch.setattr(catalog, "MODELS_DIR", models)
    catalog._catalog_entries.cache_clear()
    try:
        # The arguments the expected identity reads as given, as Harbor's trial passes them.
        summary = run_main(
            tmp_path, monkeypatch, "--interaction-mode", "streaming", "--protocol", "wall-clock",
            "--reasoning-effort", "fixed", "--report-root", str(tmp_path / "report"),
        )
    finally:
        catalog._catalog_entries.cache_clear()
    (expected,) = json.loads((tmp_path / "summary.json").read_text())["runs"]
    run_config = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert (expected["interaction_mode"], run_config["interaction_mode"]) == ("streaming",) * 2
    assert run_config["streaming"]["compression_trigger_tokens"] == 100_000
    assert expected["runtime_config_hash"] == run_config["runtime_config_hash"]
    assert summary["runtime_config_hash"] == run_config["runtime_config_hash"]


def test_a_setup_failure_records_the_mode(tmp_path, monkeypatch):
    """A failure before main() enters its episode leaves the summary runner.py's entry point
    writes from the arguments (_write_emergency_setup_summary)."""
    import sys

    from realtime_framework.evaluation import runner

    argv = ["--model", GEMINI, "--task", WEB, "--clock", "wall", "--interaction-mode", "streaming"]
    argv += ["--results-dir", str(tmp_path), "--run-id", "run"]
    monkeypatch.setattr(sys, "argv", ["runner", *argv])
    runner._write_emergency_setup_summary("setup failed")
    summary = json.loads((tmp_path / "run" / "run_summary.json").read_text())
    assert (summary["stop_reason"], summary["interaction_mode"]) == ("setup_error", "streaming")


def test_the_report_compares_the_mode(tmp_path):
    """reporting.report checks each run summary against the identity expected before the run."""
    from realtime_framework.evaluation.reporting import summary_ok

    path = tmp_path / "run_summary.json"
    path.write_text(json.dumps({"stop_reason": "terminal", "interaction_mode": "turn_based"}))
    assert summary_ok(path, expected_interaction_mode="turn_based")
    assert not summary_ok(path, expected_interaction_mode="streaming")


PLAN = """\
suite: public
models:
  - gemini-3.8-live
runtime:
  seed: 42
  mode: default
  clock_mode: wall
  fps: 1
  interaction_mode: {mode}
  duration_minutes: 5
  protocol: wall-clock
"""


def test_a_run_plan_carries_the_mode_to_the_runner(tmp_path):
    """``rtbench run -c PLAN``, as tests/config/test_frame_rates.py follows the frame rates: the
    plan's runtime becomes the Harbor agent's options, the trial's task.json and command, the
    trial's plan, and the runner's arguments, which its episode start records."""
    plan = tmp_path / "turn_based.yaml"
    plan.write_text(PLAN.format(mode="turn_based"))
    entry = f"harbor:{plan}"
    environ = {"HOME": os.environ.get("HOME", "/"), "PATH": "/usr/bin:/bin"}
    exported = export(
        environ=environ, env_file="/nonexistent/.env", entries=[entry], models=[GEMINI],
        tasks=[WEB],
    )
    cell = exported["entries"][entry]["models"][GEMINI]
    assert cell["error"] is None
    harbor = cell["harbor"]
    assert harbor["harbor_request"]["interaction_mode"] == "turn_based"
    assert harbor["task_json"]["interaction_mode"] == "turn_based"
    argv = harbor["trial_argv"]
    assert argv[argv.index("--interaction-mode") + 1] == "turn_based"
    assert cell["trial"]["settings"]["plan"]["runtime"]["interaction_mode"] == "turn_based"
    argv = cell["runner_argv"]
    assert argv[argv.index("--interaction-mode") + 1] == "turn_based"
    assert cell["run"]["interaction_mode"] == "turn_based"
    assert cell["runtime_config"]["interaction_mode"] == "turn_based"
    assert cell["expected_runtime_config_hash"] == cell["runtime_config_hash"]


@pytest.fixture
def environment():
    """_trial.main writes os.environ (providers/registry.py load_project_secrets)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_each_step_passes_streaming_on(
    environment, monkeypatch, tmp_path, capsys
):
    """The same steps in this interpreter, where a driver can declare streaming: the value each
    step receives is the one it hands on, not its default."""
    from realtime_framework.evaluation import _trial, harbor_cli
    from realtime_framework.evaluation.effective_config import _trial_command
    from realtime_framework.evaluation.harbor import ApptainerEnvironment, RealtimeBenchOptions

    plan = tmp_path / "streaming.yaml"
    plan.write_text(PLAN.format(mode="streaming"))
    # suite_dataset links task folders under temp/; the job only needs the selected names.
    monkeypatch.setattr(
        harbor_cli, "suite_dataset",
        lambda suite, selected: {"path": suite, "task_names": selected},
    )
    (agent,) = harbor_cli.plan_job(plan)["agents"]
    options = RealtimeBenchOptions(**agent["kwargs"])
    assert options.interaction_mode == "streaming"
    # RealtimeBenchAgent.run: a preset named only by -m becomes the catalog model.
    request = {"model": GEMINI, "mode": "default", **options.model_dump(), "catalog_model": GEMINI}
    task = {
        "suite_id": "public", "task_id": WEB, "fps": request["fps"],
        "model_input_fps": request["model_input_fps"],
        "interaction_mode": request["interaction_mode"],
        "duration_minutes": request["duration_minutes"], "seed": request["seed"],
    }
    environment_ = types.SimpleNamespace(python="python", lane=1, task=task)
    command = ApptainerEnvironment._runner_command(environment_, {**request, "task": task})
    trial_argv = command[3:]
    assert trial_argv[trial_argv.index("--interaction-mode") + 1] == "streaming"
    assert _trial.main([*trial_argv, "--dry-run"]) == 0
    settings = json.loads(capsys.readouterr().out)
    assert settings["plan"]["runtime"]["interaction_mode"] == "streaming"
    runner_argv, _ = _trial_command(settings["plan"])
    assert runner_argv[runner_argv.index("--interaction-mode") + 1] == "streaming"


def _trial_launch(capsys, interaction_mode) -> dict:
    """The trial worker's settings for a Harbor request in this mode (evaluation/harbor.py
    resolve), which the configuration hash is of."""
    from realtime_framework.evaluation import _trial
    from realtime_framework.evaluation.harbor import ApptainerEnvironment, RealtimeBenchOptions

    options = RealtimeBenchOptions(interaction_mode=interaction_mode)
    request = {"model": GEMINI, "mode": "default", **options.model_dump(), "catalog_model": GEMINI}
    task = {
        "suite_id": "public", "task_id": WEB,
        **{
            field: request[field]
            for field in ("fps", "model_input_fps", "interaction_mode", "duration_minutes", "seed")
        },
    }
    environment_ = types.SimpleNamespace(python="python", lane=1, task=task)
    command = ApptainerEnvironment._runner_command(environment_, {**request, "task": task})
    assert _trial.main([*command[3:], "--dry-run"]) == 0
    return json.loads(capsys.readouterr().out)


def test_a_streaming_trial_hashes_the_streaming_settings_its_run_records(
    environment, monkeypatch, tmp_path, capsys
):
    """Harbor's verify and aggregate compare the configuration hash (evaluation/config.py
    configuration_sha256): a streaming trial's holds the settings its run records
    (effective_run_config), so a job that runs two of them shows it; a turn-based trial's launch
    and hash are as they were (N47)."""
    from realtime_framework.evaluation.config import configuration_sha256
    from realtime_framework.evaluation.harbor import aggregate, verify

    turn_based = _trial_launch(capsys, "turn_based")
    assert "streaming" not in turn_based
    before = {"model_config": turn_based["model_config"], "model_revision": None}
    assert configuration_sha256(turn_based) == hashlib.sha256(
        json.dumps(before, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    streaming = _trial_launch(capsys, "streaming")
    assert streaming["streaming"] == {
        "compression_trigger_tokens": 65_536, "compression_target_tokens": 32_768,
    }
    resolve = catalog.to_model_config
    monkeypatch.setattr(
        catalog, "to_model_config",
        lambda *args, **kwargs: resolve(*args, **kwargs).model_copy(
            update={"streaming": {"compression_target_tokens": 16_384}}
        ),
    )
    edited = _trial_launch(capsys, "streaming")
    assert edited["streaming"] == {**streaming["streaming"], "compression_target_tokens": 16_384}
    assert edited["model_config"] == streaming["model_config"] == turn_based["model_config"]
    assert configuration_sha256(edited) != configuration_sha256(streaming)

    task, request, report = _trial_identity("streaming")
    report["integration"]["configuration_sha256"] = configuration_sha256(edited)
    request["configuration_sha256"] = configuration_sha256(edited)
    assert verify(report, task, request) == {"reward": 1.0}
    resolved = {**request, "configuration_sha256": configuration_sha256(streaming)}
    with pytest.raises(ValueError, match="Model configuration changed after resolution"):
        verify(report, task, resolved)
    suite_tasks = {WEB: "web", GAME: "game"}
    _job(tmp_path, {**request, "suite_tasks": suite_tasks})
    _job(tmp_path, {**resolved, "task": {**task, "task_id": GAME}, "suite_tasks": suite_tasks})
    with pytest.raises(ValueError, match="Cannot aggregate different .* configurations"):
        aggregate(tmp_path)


def _trial_identity(interaction_mode) -> tuple[dict, dict, dict]:
    """A trial's task.json (evaluation/harbor.py resolve), its request, and the one-task local
    report whose group carries the identity reporting.report found in the run summary. The request
    holds the Harbor agent's options as RealtimeBenchAgent.run records them, which a job recorded
    before the interaction mode existed has no interaction_mode among."""
    from realtime_framework.evaluation.harbor import RealtimeBenchOptions

    options = RealtimeBenchOptions(interaction_mode=interaction_mode or "turn_based").model_dump()
    if interaction_mode is None:
        del options["interaction_mode"]
    task = {
        "suite_id": "public", "suite_version": "1.0", "suite_hash": "0" * 64,
        "harness_version": "1.0", "harness_hash": "1" * 64, "task_id": WEB,
        "category": "web", "fps": 1.0, "model_input_fps": 1.0, "duration_minutes": 5.0,
        "seed": 42,
    }
    if interaction_mode is not None:
        task["interaction_mode"] = interaction_mode
    request = {
        "model": GEMINI, "mode": "default", **options, "task": task, "suite_tasks": {WEB: "web"},
    }
    group = {
        **{field: task[field] for field in ("suite_id", "suite_version", "suite_hash")},
        "harness_hash": task["harness_hash"], "fps": 1.0, "model_input_fps": 1.0,
        "interaction_mode": interaction_mode, "seed": 42, "public_category": "web",
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


# What a short streaming episode leaves in its trace (evaluation/streaming_shell.py,
# harnesses/streaming/agent.py): three frames captured, two pushed and the third's push failed, on
# Gemini Live, which confirms none; one click applied; two replies, the second given up at the
# deadline. Each is (event, kind or None, data).
EPISODE = [
    ("KernelRecord", "FrameCaptured", {"observation_id": "obs-1", "timestamp": 0.0, "error": None}),
    ("FrameSent", None, {
        "frame": 1, "observation_id": "obs-1", "captured_unix_s": 1000.0, "sent_unix_s": 1000.0,
        "acked_unix_s": None,
    }),
    ("KernelRecord", "FramePushed", {"observation_id": "obs-1", "sent_at": 0.0, "error": None}),
    ("KernelRecord", "InferenceStarted", {"decision": 0, "generation": 1}),
    ("KernelRecord", "ActionArrived", {
        "generation": 1, "frame": 1, "frame_age_s": 0.4, "frame_age_frames": 0, "rejection": None,
    }),
    ("KernelRecord", "ActionResolved", {
        "status": "applied", "arrived_at": 0.4, "started_at": 0.4, "finished_at": 0.5,
    }),
    ("KernelRecord", "InferenceStarted", {"decision": 1, "generation": 2}),
    ("KernelRecord", "FrameCaptured", {"observation_id": "obs-2", "timestamp": 1.0, "error": None}),
    ("FrameSent", None, {
        "frame": 2, "observation_id": "obs-2", "captured_unix_s": 1001.0, "sent_unix_s": 1001.0,
        "acked_unix_s": None,
    }),
    ("KernelRecord", "FramePushed", {"observation_id": "obs-2", "sent_at": 1.0, "error": None}),
    ("KernelRecord", "FrameCaptured", {"observation_id": "obs-3", "timestamp": 2.0, "error": None}),
    ("KernelRecord", "FramePushed", {"observation_id": "obs-3", "sent_at": None, "error": "lost"}),
    ("KernelRecord", "FrameDropped", {"observation_id": "obs-3", "reason": "push_failed"}),
    ("KernelRecord", "GenerationAbandoned", {"generation": 2, "reason": "deadline"}),
]  # fmt: skip
TRACE = object()  # _report_root: the streaming section its trace gives


def _trace(run, interaction_mode) -> dict:
    """The episode trace of a finished, scored episode, which passes the report's own check of
    it (reporting._episode_trace_ok); a streaming one holds EPISODE. Its description."""
    from realtime_framework.clock import SystemClock
    from realtime_framework.entity.records import TraceEvent
    from realtime_framework.evaluation.reporting import EpisodeTrace

    trace = EpisodeTrace(run / "episode_trace.jsonl", run_id=WEB, clock=SystemClock())
    trace.emit(TraceEvent.EPISODE_STARTED, {"interaction_mode": interaction_mode})
    for event, kind, data in EPISODE if interaction_mode == "streaming" else []:
        trace.emit(event, {"kind": kind, "data": data} if kind else data)
    for event in (
        TraceEvent.SNAPSHOT_FROZEN, TraceEvent.SCORE_PRODUCED, TraceEvent.EPISODE_FINISHED
    ):
        trace.emit(event, {})
    description = trace.describe()
    trace.close()
    return description


def _report_root(root, interaction_mode, streaming=TRACE, cell="cell"):
    """A local report root as runner.main() leaves it with --report-root, with one task in
    ``cell``: the identity reporting.prepare_episode_report expects before the run, and a scored
    run summary of a solved task with that identity and its episode trace (_trace). A turn-based
    run made one decision and wrote its lifecycle; a streaming run makes none and writes none
    (runner.py run_stream), and its summary has the streaming section ``streaming``: by default
    the one its trace gives (reporting.summarize_streaming), none for None. Returns the root."""
    from realtime_framework.evaluation.reporting import prepare_episode_report, summarize_streaming
    from realtime_framework.providers.gemini.live import GeminiLiveDriver

    args = types.SimpleNamespace(
        model=GEMINI, model_input_fps=1.0, reasoning_effort="fixed", fps=1.0, max_minutes=5.0,
        mode="default", clock="wall", protocol="wall-clock", seed=42, tokens_per_frame="64",
        think_budget_s=None, per_move_cap_s=None, interaction_mode=interaction_mode, task=WEB,
        results_dir=str(root / cell),
    )
    prepare_episode_report(args)
    (expected,) = json.loads((root / cell / "summary.json").read_text())["runs"]
    run = root / cell / WEB
    run.mkdir()
    runtime_config = {"interaction_mode": interaction_mode}
    if interaction_mode == "streaming":
        runtime_config["streaming"] = dict(GeminiLiveDriver.streaming_settings)
    summary = {
        **expected,
        "stop_reason": "game_decided", "run_status": "scored", "invalid_run": None,
        "error": None, "solved": True, "outcome": "pass",
        "decisions": 0 if interaction_mode == "streaming" else 1,
        "verification": {"resolved": True, "evaluation_version": 1},
        "interactables_protocol": "environment_injected_each_observation",
        "token_accounting": {"counts": {"clock_output_tokens": {"responses_with_count": 0}}},
        "cost_accounting": {}, "runtime_config": runtime_config,
        "episode_trace": _trace(run, interaction_mode),
    }  # fmt: skip
    if streaming is TRACE and interaction_mode == "streaming":
        trace = summary["episode_trace"]
        streaming = summarize_streaming(
            run / trace["artifact"], runtime_config["streaming"], attempt_id=trace["attempt_id"]
        )
    if streaming is not None and streaming is not TRACE:
        summary["streaming"] = streaming
    (run / "run_summary.json").write_text(json.dumps(summary))
    if interaction_mode == "turn_based":
        (run / "action_lifecycle.jsonl").write_text(json.dumps({"decision": 0}) + "\n")
    return root


def test_only_a_streaming_report_has_a_streaming_section(tmp_path):
    """reporting.report carries each task's streaming section (reporting.summarize_streaming) and
    sums a streaming group's counts; a turn-based group and its tasks have none."""
    from realtime_framework.evaluation.reporting import report

    (group,) = report(_report_root(tmp_path / "turn_based", "turn_based"))["groups"]
    assert group["scored"] == 1
    assert "streaming" not in group and "streaming" not in group["tasks"][0]

    root = _report_root(tmp_path / "streaming", "streaming")
    (group,) = report(root)["groups"]
    assert group["scored"] == 1
    summary = json.loads((root / "cell" / WEB / "run_summary.json").read_text())
    assert group["tasks"][0]["streaming"] == summary["streaming"]
    assert group["streaming"] == {
        "context_control": "approximate",
        "frames": {
            "captured": 3, "pushed": 2, "push_failed": 1, "dropped": 1, "confirmed": 0,
            "refused": 0, "dropped_by_reason": {"push_failed": 1},
        },
        "actions": {"arrived": 1, "outcomes": {"applied": 1}},
        "replies": {"started": 2, "abandoned": {"deadline": 1}},
    }  # fmt: skip


def test_a_streaming_run_summary_without_its_section_is_not_scored(tmp_path):
    from realtime_framework.evaluation.reporting import report

    (group,) = report(_report_root(tmp_path, "streaming", None))["groups"]
    assert (group["scored"], group["invalid"]) == (0, 1)
    (task,) = group["tasks"]
    assert (task["status"], task["error_reason"]) == ("invalid", "result_validation_failed")
    assert "streaming" not in task
    assert group["streaming"]["context_control"] is None


@pytest.mark.parametrize("interaction_mode", ["turn_based", "streaming"])
def test_a_trial_verifies_its_mode_from_its_run_to_its_reward(
    environment, tmp_path, capsys, interaction_mode
):
    """A Harbor trial over a run summary made as runner.main() leaves one (_report_root), with
    no run: the request's options become the trial worker's command and settings, which the
    report's integration reads (wall_clock_launch.json); the runner's expected identity
    (reporting.prepare_episode_report) and the report over the run summary (reporting.report)
    carry the mode to Harbor's verifier, which compares it with task.json."""
    from realtime_framework.evaluation import _trial
    from realtime_framework.evaluation.config import configuration_sha256
    from realtime_framework.evaluation.harbor import (
        ApptainerEnvironment,
        RealtimeBenchOptions,
        verify,
    )
    from realtime_framework.evaluation.reporting import report

    root = _report_root(tmp_path / "report", interaction_mode)
    (expected,) = json.loads((root / "cell" / "summary.json").read_text())["runs"]
    options = RealtimeBenchOptions(interaction_mode=interaction_mode)
    request = {"model": GEMINI, "mode": "default", **options.model_dump(), "catalog_model": GEMINI}
    # What harbor.py resolve writes as the trial's task.json.
    task = {
        **{field: expected[field] for field in ("suite_id", "suite_version", "suite_hash")},
        "harness_version": expected["harness_version"], "harness_hash": expected["harness_hash"],
        "task_id": WEB, "category": expected["public_category"], "fps": request["fps"],
        "model_input_fps": request["model_input_fps"],
        "interaction_mode": request["interaction_mode"],
        "duration_minutes": request["duration_minutes"], "seed": request["seed"],
    }
    environment_ = types.SimpleNamespace(python="python", lane=1, task=task)
    command = ApptainerEnvironment._runner_command(environment_, {**request, "task": task})
    assert _trial.main([*command[3:], "--dry-run"]) == 0
    settings = json.loads(capsys.readouterr().out)
    (root / "wall_clock_launch.json").write_text(json.dumps(settings))
    request.update(task=task, configuration_sha256=configuration_sha256(settings))

    result = report(root)
    (group,) = result["groups"]
    assert (group["interaction_mode"], group["scored"]) == (interaction_mode, 1)
    assert verify(result, task, request) == {"reward": 1.0}
    other = "turn_based" if interaction_mode == "streaming" else "streaming"
    with pytest.raises(ValueError, match="Report interaction_mode mismatch"):
        verify(result, {**task, "interaction_mode": other}, {**request, "task": {
            **task, "interaction_mode": other,
        }})
    # A run summary of the other mode is not the result the trial expected (reporting.summary_ok).
    path = root / "cell" / WEB / "run_summary.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "interaction_mode": other}))
    (group,) = report(root)["groups"]
    assert (group["scored"], group["invalid"]) == (0, 1)


def _tamper(section: dict) -> dict:
    return {**section, "frames": {**section["frames"], "pushed": 10_000}}


@pytest.mark.parametrize(
    "section, decisions, error",
    [
        (lambda s: {**s, "schema": "realtime_bench.streaming_summary.v0"}, 0, "trace"),
        (_tamper, 0, "trace"),
        # The run is Gemini Live's, whose context control is approximate.
        (lambda s: {**s, "context_control": "exact"}, 0, "trace"),
        (lambda s: {"schema": s["schema"]}, 0, "trace"),
        (lambda s: None, 0, "trace"),
        (lambda s: s, 1, "Decision count mismatch"),
    ],
    ids=["another-schema", "counts", "context-control", "no-counts", "null", "decisions"],
)
def test_a_streaming_run_summary_is_held_to_its_trace(tmp_path, section, decisions, error):
    """A streaming row stands on its streaming section, which must be the one its own trace gives
    (reporting.summarize_streaming), and on no decision: the turn-based lifecycle has none to
    compare (reporting.report). A section that is not is no score, whatever its form."""
    from realtime_framework.evaluation.reporting import report

    root = _report_root(tmp_path, "streaming")
    path = root / "cell" / WEB / "run_summary.json"
    summary = json.loads(path.read_text())
    summary.update(streaming=section(summary["streaming"]), decisions=decisions)
    path.write_text(json.dumps(summary))
    (group,) = report(root)["groups"]
    assert (group["scored"], group["invalid"]) == (0, 1)
    (task,) = group["tasks"]
    assert task["error_reason"] == "result_validation_failed"
    if error == "trace":
        error = "Streaming section does not match the episode trace"
    assert task["error"] == error


def test_a_streaming_section_of_no_form_leaves_the_report_to_the_other_groups(tmp_path):
    """A section with nothing but its schema invalidates its row; the report goes on, and the
    turn-based group beside it is scored (reporting.report)."""
    from realtime_framework.evaluation.reporting import report

    root = tmp_path / "report"
    _report_root(root, "turn_based", cell="turn_based")
    _report_root(root, "streaming", {"schema": "realtime_bench.streaming_summary.v1"})
    groups = {group["interaction_mode"]: group for group in report(root)["groups"]}
    assert (groups["turn_based"]["scored"], groups["streaming"]["invalid"]) == (1, 1)


def test_harbor_verify_compares_the_mode():
    from realtime_framework.evaluation.harbor import verify

    task, request, report = _trial_identity("turn_based")
    assert verify(report, task, request) == {"reward": 1.0}
    report["groups"][0]["interaction_mode"] = "streaming"
    with pytest.raises(ValueError, match="Report interaction_mode mismatch"):
        verify(report, task, request)


def test_harbor_verify_compares_the_harness():
    from realtime_framework.evaluation.harbor import verify

    task, request, report = _trial_identity("turn_based")
    assert verify(report, task, request) == {"reward": 1.0}
    report["groups"][0]["harness_hash"] = "9" * 64
    with pytest.raises(ValueError, match="Report harness_hash mismatch"):
        verify(report, task, request)


def _job(path, request):
    trial = path / f"{request['task']['task_id']}__trial" / "agent"
    trial.mkdir(parents=True)
    (trial / "request.json").write_text(json.dumps(request))
    return path


def test_a_job_on_two_harnesses_is_not_scored(tmp_path):
    """Each trial's task.json records the harness it ran on (evaluation/harbor.py resolve); a job
    whose two trials ran on two harnesses is refused, not scored (harbor.aggregate)."""
    from realtime_framework.evaluation.harbor import aggregate

    task, request, _ = _trial_identity("turn_based")
    request["suite_tasks"] = {WEB: "web", GAME: "game"}
    _job(tmp_path, request)
    _job(tmp_path, {**request, "task": {**task, "task_id": GAME, "harness_hash": "9" * 64}})
    with pytest.raises(ValueError, match="The job mixes benchmark versions, suites or runtime"):
        aggregate(tmp_path)


@pytest.mark.parametrize(
    "recorded, status, outcome",
    [(1.0, "scored", "pass"), (0.0, "error", "error")],
    ids=["as-reported", "unlike-the-report"],
)
def test_a_trial_is_scored_on_the_reward_its_verifier_recorded(tmp_path, recorded, status, outcome):
    """harbor.aggregate grades each trial's report again (harbor.verify) and scores the trial only
    if its verifier recorded that reward: a report worth 1 beside a recorded 0 is an error."""
    from realtime_framework.evaluation.harbor import aggregate

    _, request, report = _trial_identity("turn_based")
    trial = _job(tmp_path, request) / f"{WEB}__trial"
    (trial / "artifacts/logs/artifacts").mkdir(parents=True)
    (trial / "artifacts/logs/artifacts/report.json").write_text(json.dumps(report))
    (trial / "verifier").mkdir()
    (trial / "verifier/reward.json").write_text(json.dumps({"reward": recorded}))
    (trial / "result.json").write_text(json.dumps({"exception_info": None}))
    (row,) = aggregate(tmp_path)["tasks"]
    assert (row["status"], row["outcome"]) == (status, outcome)


@pytest.mark.parametrize(
    "recorded, reported", [("streaming", "streaming"), (None, "turn_based")],
    ids=["recorded", "before-the-mode"],
)
def test_a_job_reports_its_mode(tmp_path, recorded, reported):
    """A job recorded before the interaction mode existed has none in its task.json."""
    from realtime_framework.evaluation.harbor import aggregate

    report = aggregate(_job(tmp_path, _trial_identity(recorded)[1]))
    assert report["interaction_mode"] == reported


def test_a_streaming_run_records_the_live_api_settings():
    """OpenAI Realtime's retained frames and Gemini Live's compression thresholds apply only to
    streaming, so only a streaming run's configuration (and its hash) records them."""
    from realtime_framework.evaluation.config import effective_run_config
    from realtime_framework.providers.gemini.live import GeminiLiveDriver
    from realtime_framework.providers.openai_realtime import OpenAIRealtimeDriver

    def run_config(key, interaction_mode):
        effort = catalog.get(key).defaults["reasoning_effort"]
        return effective_run_config(
            model_config=catalog.to_model_config(key, fps=1.0, reasoning_effort=effort),
            fps=1.0, max_minutes=5.0, protocol="wall-clock", think_budget_s=None,
            per_move_cap_s=None, mode="default", clock_mode="wall",
            interaction_mode=interaction_mode,
        )

    openai = run_config("gpt-realtime-2", "streaming")["streaming"]
    assert openai == OpenAIRealtimeDriver.streaming_settings
    gemini = run_config(GEMINI, "streaming")["streaming"]
    assert gemini == GeminiLiveDriver.streaming_settings
    # SlidingWindow.target_tokens must be below ContextWindowCompressionConfig.trigger_tokens.
    assert 0 < gemini["compression_target_tokens"] < gemini["compression_trigger_tokens"]
    for key in ("gpt-realtime-2", GEMINI):
        assert "streaming" not in run_config(key, "turn_based")


def test_a_model_yaml_sets_the_streaming_settings(
    monkeypatch, tmp_path
):
    """A model YAML's streaming settings replace the driver's they name. A streaming run records
    them; a turn-based run's configuration, its model configuration included, is the one without
    them."""
    from realtime_framework.evaluation.config import effective_run_config
    from realtime_framework.providers.gemini.live import GeminiLiveDriver

    def run_config(key, interaction_mode):
        return effective_run_config(
            model_config=catalog.to_model_config(key, fps=1.0, reasoning_effort="fixed"),
            fps=1.0, max_minutes=5.0, protocol="wall-clock", think_budget_s=None,
            per_move_cap_s=None, mode="default", clock_mode="wall",
            interaction_mode=interaction_mode,
        )

    turn_based = run_config(GEMINI, "turn_based")
    models = tmp_path / "models"
    models.mkdir()
    (models / f"{GEMINI}.yaml").write_text(
        (catalog.MODELS_DIR / f"{GEMINI}.yaml").read_text()
        + "streaming:\n  compression_trigger_tokens: 100000\n"
    )
    monkeypatch.setattr(catalog, "MODELS_DIR", models)
    catalog._catalog_entries.cache_clear()
    try:
        assert run_config(GEMINI, "streaming")["streaming"] == {
            **GeminiLiveDriver.streaming_settings,
            "compression_trigger_tokens": 100_000,
        }
        assert run_config(GEMINI, "turn_based") == turn_based
    finally:
        catalog._catalog_entries.cache_clear()


def test_a_leaderboard_submission_is_turn_based(tmp_path):
    """rtbench upload's check of three jobs (evaluation/submission.py check_runs); each one's
    trial has no report, which counts as a failed task. Streaming results go to a separate
    leaderboard."""
    from realtime_framework.evaluation.submission import SubmissionError, check_runs

    def jobs(name, interaction_mode):
        request = _trial_identity(interaction_mode)[1]
        return [_job(tmp_path / name / f"run{run}", request) for run in (1, 2, 3)]

    assert len(check_runs(jobs("turn_based", "turn_based"))) == 3
    assert len(check_runs(jobs("before-the-mode", None))) == 3
    # The mode is one of the agent options every request records, so a job from before it existed
    # has another integration hash (evaluation/harbor.py aggregate) and is submitted on its own.
    mixed = [*jobs("old", None)[:1], *jobs("new", "turn_based")[1:]]
    with pytest.raises(SubmissionError, match="The runs differ in integration_hash"):
        check_runs(mixed)
    with pytest.raises(SubmissionError) as caught:
        check_runs(jobs("streaming", "streaming"))
    assert str(caught.value) == (
        f"{tmp_path / 'streaming' / 'run1'}: interaction_mode is 'streaming', expected 'turn_based'"
    )


def test_a_leaderboard_submission_refuses_an_unreleased_harness(tmp_path):
    """A run records the version of the harness hash it ran on, +dev for any hash pyproject.toml
    does not lock (evaluation/config.py harness_version); rtbench upload takes only a release."""
    from realtime_framework.evaluation.config import harness_version
    from realtime_framework.evaluation.submission import SubmissionError, check_runs

    task, request, _ = _trial_identity("turn_based")
    task["harness_version"] = harness_version(task["harness_hash"])
    jobs = [_job(tmp_path / f"run{run}", request) for run in (1, 2, 3)]
    with pytest.raises(SubmissionError) as caught:
        check_runs(jobs)
    assert str(caught.value) == f"{tmp_path / 'run1'}: harness_version '1.0+dev' is not a release"


def test_a_leaderboard_submission_refuses_one_job_given_three_times(tmp_path):
    from realtime_framework.evaluation.submission import SubmissionError, check_runs

    job = _job(tmp_path, _trial_identity("turn_based")[1])
    with pytest.raises(SubmissionError, match="Give 3 different job directories, one per run"):
        check_runs([job, job, job])


def test_a_leaderboard_submission_refuses_the_token_clock(tmp_path):
    from realtime_framework.evaluation.submission import SubmissionError, check_runs

    request = {**_trial_identity("turn_based")[1], "clock": "token"}
    jobs = [_job(tmp_path / f"run{run}", request) for run in (1, 2, 3)]
    with pytest.raises(SubmissionError) as caught:
        check_runs(jobs)
    assert str(caught.value) == f"{tmp_path / 'run1'}: clock is 'token', expected 'wall'"


def test_an_upload_carries_no_request_or_reply_body(tmp_path):
    """Every body goes to episode_blobs.pack next to the trace, which is exported and uploaded
    while the pack is neither (evaluation/agent_trace.py): Harbor downloads the trial's export into
    the job (harbor_viewer.export_episode), and rtbench upload archives the job
    (submission.build_archive)."""
    import tarfile

    from realtime_framework.evaluation.harbor_viewer import export_episode
    from realtime_framework.evaluation.submission import build_archive

    run = _report_root(tmp_path / "report", "turn_based") / "cell" / WEB
    (run / "episode_blobs.pack").write_bytes(bytes(64))
    job = tmp_path / "job"
    export_episode(run, job / f"{WEB}__trial" / "agent", _trial_identity("turn_based")[1])
    archive = tmp_path / "submission.tar.gz"
    build_archive([job], [{}], archive)
    with tarfile.open(archive) as members:
        names = {os.path.basename(name) for name in members.getnames()}
    assert "episode_trace.jsonl" in names
    assert "episode_blobs.pack" not in names
