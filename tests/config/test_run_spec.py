"""An episode's run options are declared, defaulted and checked once (evaluation/run_spec.py):
every entry point builds a RunSpec from what it is given, with its own defaults only where the
entry points differ on purpose (ENTRY_DEFAULTS)."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.config import DEFAULT_TASK_SEED, DEFAULT_TOKENS_PER_FRAME
from realtime_framework.evaluation.run_spec import RunSpec, run_spec

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
        ("runner", "token", DEFAULT_TOKENS_PER_FRAME, "time-aware-token-v1"),
        ("trial", "wall", None, "wall-clock"),
        ("harbor", "wall", None, "wall-clock"),
    ],
)
def test_each_entry_point_s_defaults(monkeypatch, entry, clock_mode, tokens_per_frame, protocol):
    """Run directly, the runner takes the token clock at the default B; the trial worker and
    Harbor's options and plans the wall clock. Every other default is the same for all."""
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
    """The runner takes the default B; the trial worker and Harbor run unlimited."""
    assert run_spec("runner", clock_mode="token").tokens_per_frame == DEFAULT_TOKENS_PER_FRAME
    for entry in ("trial", "harbor"):
        assert run_spec(entry, clock_mode="token").tokens_per_frame is None


def test_b_on_the_command_lines_and_on_the_wall_clock():
    """The command lines give B as text; on the wall clock there is none, whatever was given."""
    assert run_spec("runner", tokens_per_frame="64").tokens_per_frame == 64
    assert run_spec("runner", tokens_per_frame="unlimited").tokens_per_frame is None
    assert run_spec("runner", clock_mode="wall").tokens_per_frame is None
    assert run_spec("harbor", clock_mode="wall", tokens_per_frame=64).tokens_per_frame is None


def test_only_the_command_lines_take_the_seed_variable(monkeypatch):
    monkeypatch.setenv("SEED", "7")
    assert (run_spec("runner").seed, run_spec("trial").seed) == (7, 7)
    assert run_spec("harbor").seed == DEFAULT_TASK_SEED
    assert run_spec("runner", seed="9").seed == 9


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
    with pytest.raises(ValueError) as caught:
        run_spec("runner", **options)
    assert str(caught.value) == message


def test_a_legacy_protocol_label_keeps_its_caps():
    spec = run_spec("runner", clock_mode="wall", protocol="legacy", think_budget_s=5.0)
    assert (spec.protocol, spec.think_budget_s) == ("legacy", 5.0)


@pytest.fixture
def environment():
    """_trial.main writes os.environ (evaluation/_trial.py)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _launch(tmp_path, capsys, *argv) -> Path:
    """The launch file of a trial worker's dry run on ``argv``, where it writes it."""
    from realtime_framework.evaluation import _trial

    assert _trial.main([*argv, "--dry-run"]) == 0
    (tmp_path / _trial.LAUNCH_FILE).write_text(capsys.readouterr().out)
    return tmp_path


def test_the_runner_runs_the_options_the_trial_worker_wrote(environment, tmp_path, capsys):
    """The trial worker parses its options once and writes them into its launch file; the runner
    it execs reads them, the model, its reasoning effort and the task from there (runner.py
    --launch), not from a command line rebuilt for it."""
    from realtime_framework.evaluation import _trial, runner

    os.environ.pop("SEED", None)
    root = _launch(
        tmp_path, capsys, "--catalog-model", GEMINI, "--task", WEB, "--fps", "2",
        "--model-input-fps", "1", "--duration-minutes", "3", "--seed", "7",
    )  # fmt: skip
    parser = runner._parser()
    arguments = parser.parse_args(_trial.runner_arguments(root))
    spec = runner._run_spec(parser, arguments)
    assert spec == run_spec("trial", fps=2.0, model_input_fps=1.0, duration_minutes=3.0, seed=7)
    effort = catalog.get(GEMINI).defaults["reasoning_effort"]
    assert (arguments.model, arguments.task, arguments.reasoning_effort) == (GEMINI, WEB, effort)


@pytest.mark.parametrize(
    "argv, given", [(["--fps", "2"], "fps"), (["--model", GEMINI, "--seed", "1"], "model, seed")]
)
def test_the_runner_refuses_what_its_launch_file_gives(
    environment, tmp_path, capsys, argv, given
):
    from realtime_framework.evaluation import _trial, runner

    root = _launch(tmp_path, capsys, "--catalog-model", GEMINI, "--task", WEB)
    parser = runner._parser()
    arguments = parser.parse_args([*_trial.runner_arguments(root), *argv])
    with pytest.raises(SystemExit) as caught:
        runner._run_spec(parser, arguments)
    assert caught.value.code == 2
    assert capsys.readouterr().err.strip().endswith(f"error: --launch gives {given}; omit them")


def test_the_runner_needs_a_model_and_a_task_without_a_launch_file(capsys):
    from realtime_framework.evaluation import runner

    parser = runner._parser()
    with pytest.raises(SystemExit) as caught:
        runner._run_spec(parser, parser.parse_args(["--model", GEMINI]))
    assert caught.value.code == 2
    assert capsys.readouterr().err.strip().endswith(
        "error: --model and --task are required without --launch"
    )
