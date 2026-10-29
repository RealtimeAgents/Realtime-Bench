"""An episode's run options are declared, defaulted and checked once (evaluation/run_spec.py):
every entry point builds a RunSpec from what it is given, with its own defaults only where the
entry points differ on purpose (ENTRY_DEFAULTS)."""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.config import DEFAULT_TASK_SEED
from realtime_framework.evaluation.run_spec import RunSpec, run_spec

from ..launch import trial_launch

GEMINI, WEB = "gemini-3.8-live", "rt_waitlist_fit"


def test_a_spec_is_frozen_and_refuses_an_option_it_does_not_declare():
    spec = run_spec("harbor")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.fps = 2.0
    with pytest.raises(TypeError, match="frame_rate"):
        RunSpec(clock_mode="wall", frame_rate=2.0)


@pytest.mark.parametrize(
    "entry, clock_mode, tokens_per_frame, protocol",
    [
        ("trial", "wall", None, "wall-clock"),
        ("harbor", "wall", None, "wall-clock"),
    ],
)
def test_each_entry_point_s_defaults(monkeypatch, entry, clock_mode, tokens_per_frame, protocol):
    """The trial worker and Harbor's options and plans take the wall clock, and every other
    default is the same for both."""
    monkeypatch.delenv("SEED", raising=False)
    spec = run_spec(entry)
    assert (spec.clock_mode, spec.tokens_per_frame, spec.protocol) == (
        clock_mode, tokens_per_frame, protocol,
    )  # fmt: skip
    assert (spec.mode, spec.fps, spec.model_input_fps, spec.interaction_mode) == (
        "default", 1.0, 1.0, "turn_based",
    )  # fmt: skip
    assert (spec.duration_minutes, spec.seed) == (5.0, DEFAULT_TASK_SEED)
    assert (spec.think_budget_s, spec.per_move_cap_s) == (None, None)


def test_the_token_clock_without_b():
    """The trial worker and Harbor run unlimited."""
    for entry in ("trial", "harbor"):
        assert run_spec(entry, clock_mode="token").tokens_per_frame is None


def test_b_on_the_command_lines_and_on_the_wall_clock():
    """The command line gives B as text; on the wall clock there is none, whatever was given."""
    token = {"clock_mode": "token"}
    assert run_spec("trial", **token, tokens_per_frame="64").tokens_per_frame == 64
    assert run_spec("trial", **token, tokens_per_frame="unlimited").tokens_per_frame is None
    assert run_spec("trial", tokens_per_frame="64").tokens_per_frame is None
    assert run_spec("harbor", clock_mode="wall", tokens_per_frame=64).tokens_per_frame is None


def test_only_the_command_line_takes_the_seed_variable(monkeypatch):
    monkeypatch.setenv("SEED", "7")
    assert run_spec("trial").seed == 7
    assert run_spec("harbor").seed == DEFAULT_TASK_SEED
    assert run_spec("trial", seed="9").seed == 9


def test_an_option_given_as_none_is_one_not_given():
    spec = run_spec("trial", fps=2.0, model_input_fps=None, clock_mode=None)
    assert (spec.fps, spec.model_input_fps, spec.clock_mode) == (2.0, 2.0, "wall")


@pytest.mark.parametrize(
    "options, message",
    [
        ({"mode": "fast"}, "mode must be 'default'"),
        ({"clock_mode": "pause"}, "clock_mode must be 'token' or 'wall'"),
        ({"tokens_per_frame": 0}, "tokens_per_frame must be a positive integer or 'unlimited'"),
        ({"tokens_per_frame": "many"}, "tokens_per_frame must be a positive integer or 'unlimited'"),
        ({"tokens_per_frame": True}, "tokens_per_frame must be a positive integer or 'unlimited'"),
        ({"fps": 0}, "fps must be positive"),
        ({"fps": float("inf")}, "fps must be positive"),
        ({"fps": True}, "fps must be positive"),
        ({"model_input_fps": -1.0}, "model_input_fps must be positive"),
        ({"interaction_mode": "live"}, "interaction_mode must be 'turn_based' or 'streaming'"),
        ({"duration_minutes": 0}, "duration_minutes must be positive"),
        ({"seed": -1}, "seed must be an integer between 0 and 4294967295"),
        (
            {"clock_mode": "wall", "protocol": "time-aware-token-v1"},
            "time-aware-token-v1 requires --clock token",
        ),
        (
            {"think_budget_s": 5.0},
            "time-aware-token-v1 uses only the task horizon T; deliberation caps are legacy "
            "diagnostics",
        ),
        (
            {"clock_mode": "wall", "interaction_mode": "streaming", "per_move_cap_s": 2.0},
            "Streaming does not support --per-move-cap-s",
        ),
    ],
    ids=[
        "mode", "clock", "b-zero", "b-text", "b-bool", "fps-zero", "fps-inf", "fps-bool",
        "model-input-fps", "interaction-mode", "duration", "seed", "protocol-clock",
        "protocol-caps", "streaming-caps",
    ],
)  # fmt: skip
def test_each_option_is_checked_where_it_is_declared(options, message):
    """On the token clock unless the case names another."""
    with pytest.raises(ValueError) as caught:
        run_spec("trial", **{"clock_mode": "token", **options})
    assert str(caught.value) == message


def test_a_legacy_protocol_label_keeps_its_caps():
    spec = RunSpec(clock_mode="wall", protocol="legacy", think_budget_s=5.0)
    assert (spec.protocol, spec.think_budget_s) == ("legacy", 5.0)


def test_the_runner_runs_the_options_the_trial_worker_wrote(tmp_path, monkeypatch):
    """The trial worker parses its options once and writes them into its launch file; the runner
    it execs reads them, the model, its reasoning effort and the task from there
    (evaluation/_trial.py read_launch), not from a command line rebuilt for it."""
    from realtime_framework.evaluation import _trial

    for name in ("SEED", "SANDBOX_GRAPHICS_MODE", "SCREEN_RECORDING_ENCODER"):
        monkeypatch.delenv(name, raising=False)
    path = trial_launch(
        monkeypatch, tmp_path, "--catalog-model", GEMINI, "--task", WEB, "--fps", "2",
        "--model-input-fps", "1", "--duration-minutes", "3", "--seed", "7",
    )  # fmt: skip
    launch = _trial.read_launch(path)
    assert launch.spec == run_spec(
        "trial", fps=2.0, model_input_fps=1.0, duration_minutes=3.0, seed=7
    )
    effort = catalog.get(GEMINI).defaults["reasoning_effort"]
    assert (launch.model, launch.task, launch.reasoning_effort) == (GEMINI, WEB, effort)
    assert (launch.graphics_mode, launch.recording_encoder) == ("cpu", "cpu")


def test_the_runner_takes_its_launch_file_and_where_to_write_alone(capsys):
    """Everything the runner runs is in the trial worker's launch file: it needs one, and takes
    no option of what to run (runner.py _parser)."""
    from realtime_framework.evaluation import runner

    parser = runner._parser()
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["--results-dir", "out", "--run-id", "run"])
    assert caught.value.code == 2
    assert capsys.readouterr().err.strip().endswith(
        "error: the following arguments are required: --launch"
    )
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["--launch", "launch.json", "--model", GEMINI, "--fps", "2"])
    assert caught.value.code == 2
    assert capsys.readouterr().err.strip().endswith(
        f"error: unrecognized arguments: --model {GEMINI} --fps 2"
    )


def _launched_run(tmp_path, edit, *argv) -> dict:
    """The export probes' record (evaluation/effective_config.py) of the runner run up to its
    decision loop on the launch file the trial worker writes for Gemini Live on the web task with
    ``argv``, after ``edit(settings)`` changed it; each in a fresh interpreter, with the
    environment the trial worker execs the runner with."""
    from realtime_framework.evaluation.effective_config import _probe, _runner_entry

    environ = {"HOME": os.environ.get("HOME", "/"), "PATH": "/usr/bin:/bin"}
    argv = ["--catalog-model", GEMINI, "--task", WEB, *argv]
    request = {"runs": [(GEMINI, WEB, argv)], "env_file": None, "attribution": {}}
    trial = _probe("trial", environ, request, scratch=tmp_path)["models"][GEMINI][WEB]
    assert trial["error"] is None
    settings = trial["settings"]
    edit(settings)
    runs = [(GEMINI, WEB, trial["runner_argv"], settings)]
    exported = _runner_entry(
        trial["runner_environ"], {}, runs, env_file=None, scratch=tmp_path, python=trial["python"]
    )
    record = exported["models"][GEMINI][WEB]
    assert record["error"] is None
    return record


def test_the_runner_runs_the_model_configuration_its_launch_file_holds(tmp_path):
    """The runner runs the model configuration in its launch file, the one the trial worker
    checked and Harbor hashed (evaluation/config.py configuration_sha256), not one it rebuilds
    from the catalog: a value changed there is the one its run identity, the identity its report
    expects (reporting.prepare_episode_report) and its run configuration record."""

    def edit(settings):
        settings["model_config"].update(model_name="edited-backend", context_reserve_tokens=8192)

    record = _launched_run(tmp_path, edit)
    assert record["run"]["requested_model"] == {
        "adapter": "gemini_live", "model": GEMINI, "backend_model": "edited-backend",
    }  # fmt: skip
    model_config = record["runtime_config"]["model_config"]
    assert (model_config["model_name"], model_config["context_reserve_tokens"]) == (
        "edited-backend", 8192,
    )  # fmt: skip
    assert record["expected_runtime_config_hash"] == record["runtime_config_hash"]


def test_the_runner_runs_the_context_settings_its_launch_file_holds(tmp_path, monkeypatch):
    """The model configuration's context settings travel whole in the launch file
    (ModelConfig.context): the runner reads them back as they are, never resolving them again, and
    runs their values."""
    from realtime_framework.evaluation import _trial

    path = trial_launch(monkeypatch, tmp_path / "trial", "--catalog-model", GEMINI, "--task", WEB)
    settings = json.loads(path.read_text())
    items = {item["key"]: item for item in settings["model_config"]["context"]["items"]}
    items["tool_call_history"].update(source="run", category="nonstandard")
    path.write_text(json.dumps(settings))
    item = _trial.read_launch(path).model_config.context.item("tool_call_history")
    assert (item.value, item.source, item.category) == ("tool_transcript", "run", "nonstandard")

    def edit(settings):
        items = {item["key"]: item for item in settings["model_config"]["context"]["items"]}
        items["screenshots_kept"]["value"], items["screenshot_fold_block"]["value"] = 8, 3

    (tmp_path / "run").mkdir()
    record = _launched_run(tmp_path / "run", edit)
    config = record["runtime_config"]
    assert (config["active_visual_budget"], config["visual_fold_block_size"]) == (8, 3)
    assert record["expected_runtime_config_hash"] == record["runtime_config_hash"]


def test_a_streaming_runner_runs_the_streaming_settings_its_launch_file_holds(tmp_path):
    """A streaming launch file holds the model's streaming settings merged over its driver's
    (evaluation/_trial.py main), which ModelConfig leaves out of its dump; the runner runs those."""

    def edit(settings):
        settings["streaming"]["compression_target_tokens"] = 16_384

    record = _launched_run(tmp_path, edit, "--interaction-mode", "streaming")
    assert record["runtime_config"]["streaming"] == {
        "compression_trigger_tokens": 65_536, "compression_target_tokens": 16_384,
    }  # fmt: skip
    assert record["expected_runtime_config_hash"] == record["runtime_config_hash"]


def test_the_emergency_summary_names_the_model_its_launch_file_holds(tmp_path, monkeypatch):
    """A run that fails to set up is summarized from its launch file
    (runner._write_emergency_setup_summary): the backend model and API of the model
    configuration there, and the catalog key the trial worker resolved it from."""
    import json
    import sys

    from realtime_framework.evaluation import runner

    path = trial_launch(monkeypatch, tmp_path / "trial", "--catalog-model", GEMINI, "--task", WEB)
    settings = json.loads(path.read_text())
    settings["model_config"]["model_name"] = "edited-backend"
    path.write_text(json.dumps(settings))
    argv = ["--launch", str(path), "--results-dir", str(tmp_path), "--run-id", "run"]
    monkeypatch.setattr(sys, "argv", ["runner", *argv])
    runner._write_emergency_setup_summary("RuntimeError: guest not ready")
    summary = json.loads((tmp_path / "run" / "run_summary.json").read_text())
    assert (summary["model"], summary["adapter"], summary["backend_model"]) == (
        GEMINI, "gemini_live", "edited-backend",
    )  # fmt: skip


ENDPOINT = "http://127.0.0.1:9/v1"
# The inline endpoints, whose configuration the trial worker resolves from the variables it sets
# (evaluation/_trial.py main, providers/registry.py _load_entries and to_model_config).
INLINE = [
    ("remote-vlm", ["--model", "dep-1", "--base-url", ENDPOINT, "--model-revision", "r1"]),
    ("remote-vlm", ["--model", "dep-1", "--base-url", ENDPOINT, "--no-stream",
                    "--context", "reasoning_replay=true"]),
    ("litellm-vlm", ["--model", "openai/vlm-1", "--base-url", ENDPOINT, "--model-revision", "r2"]),
    ("litellm-vlm", ["--model", "openai/vlm-1"]),
]  # fmt: skip


def _launches():
    """Every catalog model in each mode its API takes, and the inline endpoints."""
    cases = []
    presets = catalog.keys()
    for key in presets:
        driver = catalog.adapter_class(catalog.get(key).adapter_type)
        modes = ["turn_based"]
        if "streaming" in driver.capabilities.interaction_modes:
            modes.append("streaming")
        cases += [(key, ["--catalog-model", key], mode) for mode in modes]
    return cases + [(key, argv, "turn_based") for key, argv in INLINE]


@pytest.mark.parametrize("key, argv, interaction_mode", _launches())
def test_a_launch_file_carries_the_model_configuration_the_trial_worker_resolved(
    tmp_path, key, argv, interaction_mode
):
    """The runner's model configuration, read from the launch file (evaluation/_trial.py
    read_launch), is the one the trial worker resolved from the catalog: every field but the
    key, which no launch file holds and which the driver reads from its variable, and the
    streaming settings as the driver merges them, and so the same run configuration and hash."""
    from realtime_framework.entity.config import ModelConfig, merge_streaming_settings
    from realtime_framework.entity.context_policy import ContextLayers
    from realtime_framework.evaluation import _trial
    from realtime_framework.evaluation.config import config_fingerprint, run_spec_config

    catalog._catalog_entries.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("ALLOW_EXPERIMENTAL_MODEL", "1")
            path = trial_launch(
                patch, tmp_path, *argv, "--task", WEB, "--interaction-mode", interaction_mode
            )
            launch = _trial.read_launch(path)
            # As the trial worker resolved it, in the environment the runner gets, with the
            # settings its command line gave.
            catalog._catalog_entries.cache_clear()
            run = _trial.context_settings(
                [argv[index + 1] for index, part in enumerate(argv) if part == "--context"]
            )
            resolved = catalog.to_model_config(
                key,
                context=ContextLayers(run=run),
                fps=launch.spec.model_input_fps,
                reasoning_effort=launch.reasoning_effort,
            )
            launched = launch.model_config
            for field in ModelConfig.model_fields:
                if field not in ("api_key", "streaming"):
                    assert getattr(launched, field) == getattr(resolved, field), field
            if interaction_mode == "streaming":
                driver = catalog.adapter_class(resolved.adapter_type)
                assert merge_streaming_settings(driver, launched.streaming) == (
                    merge_streaming_settings(driver, resolved.streaming)
                )
            assert config_fingerprint(run_spec_config(launch.spec, launched)) == (
                config_fingerprint(run_spec_config(launch.spec, resolved))
            )
    finally:
        catalog._catalog_entries.cache_clear()
