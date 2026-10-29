"""An episode's run options are declared, defaulted and checked once (evaluation/run_spec.py):
every entry point builds a RunSpec from what it is given, with its own defaults only where the
entry points differ on purpose (ENTRY_DEFAULTS)."""

from __future__ import annotations

import dataclasses

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
