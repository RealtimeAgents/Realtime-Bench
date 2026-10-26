"""The options one episode runs with, each declared, defaulted and checked here alone.

Every entry point builds a RunSpec from what it is given: the runner's command line and
identity_main's, which takes the same options (evaluation/runner.py, evaluation/config.py), the
Harbor trial worker's (evaluation/_trial.py), Harbor's agent options (evaluation/harbor.py
RealtimeBenchOptions) and a run plan's runtime (evaluation/config.py load_benchmark). An option an
entry point is not given takes that entry point's default (ENTRY_DEFAULTS), else the field's. The
rules that read the model's API, its driver's declarations, are the pre-run checks'
(providers/preflight.py), and stay there.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from realtime_framework.entity.config import DEFAULT_TASK_SEED
from realtime_framework.entity.config import DEFAULT_TOKENS_PER_FRAME
from realtime_framework.entity.config import task_seed
from realtime_framework.providers.preflight import check_streaming_options

DEFAULT_MODE = "default"
INTERACTION_MODES = ("turn_based", "streaming")
# The protocol each clock runs, unless the runner is given another label for it (--protocol).
PROTOCOLS = {"wall": "wall-clock", "token": "time-aware-token-v1"}

# The defaults that differ between entry points, on purpose. Run directly, the runner takes the
# token clock at the default B, as identity_main does, which prints the hash the runner records.
# The trial worker, and Harbor's agent options and run plans, from which Harbor starts it, take the
# wall clock, and an unlimited B where the token clock is asked for without one; a run plan on the
# token clock names its B (load_benchmark).
ENTRY_DEFAULTS = {
    "runner": {"clock_mode": "token", "tokens_per_frame": DEFAULT_TOKENS_PER_FRAME},
    "trial": {"clock_mode": "wall", "tokens_per_frame": "unlimited"},
    "harbor": {"clock_mode": "wall", "tokens_per_frame": "unlimited"},
}
# The two command lines take an option they are not given from this variable, where it is set.
ENVIRONMENT_DEFAULTS = {
    "runner": {"seed": "SEED"},
    "trial": {"seed": "SEED"},
}


def _positive(name: str, value) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, kw_only=True)
class RunSpec:
    """One episode's run options, checked as they are made; an unknown one is a TypeError.

    ``fps`` is the environment frame rate and ``model_input_fps`` that of the frames the model is
    fed, ``fps`` when left out. ``tokens_per_frame`` is B per model-input frame, a positive
    integer, or None for unlimited, which it is on the wall clock whatever it was given; it takes
    "unlimited" and, as the command lines give it, a decimal string. ``protocol`` left out is its
    clock's (PROTOCOLS). The deliberation caps are the runner's legacy diagnostics, which
    streaming refuses."""

    mode: str = DEFAULT_MODE
    clock_mode: str
    tokens_per_frame: int | None = None
    fps: float = 1.0
    model_input_fps: float | None = None
    interaction_mode: str = "turn_based"
    duration_minutes: float = 5.0
    seed: int = DEFAULT_TASK_SEED
    protocol: str | None = None
    think_budget_s: float | None = None
    per_move_cap_s: float | None = None

    def __post_init__(self):
        if self.mode != DEFAULT_MODE:
            raise ValueError(f"mode must be {DEFAULT_MODE!r}")
        if self.clock_mode not in PROTOCOLS:
            raise ValueError("clock_mode must be 'token' or 'wall'")
        tokens_per_frame = self.tokens_per_frame
        if tokens_per_frame == "unlimited":
            tokens_per_frame = None
        elif (
            isinstance(tokens_per_frame, str)
            and tokens_per_frame.isascii()
            and tokens_per_frame.isdecimal()
        ):
            tokens_per_frame = int(tokens_per_frame)
        if tokens_per_frame is not None and (
            type(tokens_per_frame) is not int or tokens_per_frame <= 0
        ):
            raise ValueError("tokens_per_frame must be a positive integer or 'unlimited'")
        if self.clock_mode == "wall":
            tokens_per_frame = None
        object.__setattr__(self, "tokens_per_frame", tokens_per_frame)
        _positive("fps", self.fps)
        if self.model_input_fps is None:
            object.__setattr__(self, "model_input_fps", self.fps)
        _positive("model_input_fps", self.model_input_fps)
        if self.interaction_mode not in INTERACTION_MODES:
            raise ValueError("interaction_mode must be 'turn_based' or 'streaming'")
        _positive("duration_minutes", self.duration_minutes)
        object.__setattr__(self, "seed", task_seed(self.seed))
        if self.protocol is None:
            object.__setattr__(self, "protocol", PROTOCOLS[self.clock_mode])
        if self.protocol == PROTOCOLS["token"]:
            if self.clock_mode != "token":
                raise ValueError("time-aware-token-v1 requires --clock token")
            if self.think_budget_s is not None or self.per_move_cap_s is not None:
                raise ValueError(
                    "time-aware-token-v1 uses only the task horizon T; deliberation caps are "
                    "legacy diagnostics"
                )
        check_streaming_options(
            self.interaction_mode,
            think_budget_s=self.think_budget_s,
            per_move_cap_s=self.per_move_cap_s,
        )


def run_spec(entry: str, **given) -> RunSpec:
    """The RunSpec of the options ``given`` to ``entry``, a row of ENTRY_DEFAULTS; an option given
    as None is one not given."""
    options = dict(ENTRY_DEFAULTS[entry])
    for option, variable in ENVIRONMENT_DEFAULTS.get(entry, {}).items():
        if variable in os.environ:
            options[option] = os.environ[variable]
    options.update((option, value) for option, value in given.items() if value is not None)
    return RunSpec(**options)
