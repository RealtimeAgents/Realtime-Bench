from __future__ import annotations

# Config
"""Side-effect-free runtime configuration models."""
import os as os
from dataclasses import dataclass as dataclass
from typing import Literal as Literal
from typing import Optional as Optional

from pydantic import BaseModel as BaseModel
from pydantic import ConfigDict as ConfigDict
from pydantic import Field as Field

SANDBOX_CONTAINER = os.environ.get("SANDBOX_CONTAINER", "streaming-agent-sandbox")
SANDBOX_DISPLAY_NUMBER = int(os.environ.get("SANDBOX_DISPLAY_NUMBER", "99"))
if not 0 <= SANDBOX_DISPLAY_NUMBER <= 65535:
    raise ValueError("SANDBOX_DISPLAY_NUMBER must be between 0 and 65535")
SANDBOX_DISPLAY = f":{SANDBOX_DISPLAY_NUMBER}"
DEFAULT_TASK_SEED = 42


def task_seed(value) -> int:
    """Validate a task seed without silently truncating floats or accepting booleans."""
    if isinstance(value, str):
        if not value.isascii() or not value.isdecimal():
            raise ValueError("seed must be an integer between 0 and 4294967295")
        value = int(value)
    if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    return value


DEFAULT_TOKENS_PER_FRAME = 256
DEFAULT_FRAME_MAX_DIM = 768
DEFAULT_ACTIVE_VISUAL_BUDGET = 20
DEFAULT_VISUAL_FOLD_BLOCK_SIZE = 10
DEFAULT_DETAILED_HISTORY_TURNS = 50
DEFAULT_DETAILED_HISTORY_DROP_BLOCK_SIZE = 10


def effective_detailed_history_policy() -> tuple[int, int]:
    """Return the verbatim text-history window and block-archive size.

    At ``window + 1`` turns, the oldest ``drop_block_size`` complete detailed turns are archived
    together. With the maintained defaults, verbatim text/action context therefore cycles between
    41 and 50 turns while older executed actions remain available through the compact ledger,
    unless a request that does not fit leaves the earliest of them out.
    """
    window = int(
        os.environ.get(
            "STREAM_DETAILED_HISTORY_TURNS",
            str(DEFAULT_DETAILED_HISTORY_TURNS),
        )
    )
    drop_block_size = int(
        os.environ.get(
            "STREAM_DETAILED_HISTORY_DROP_BLOCK_SIZE",
            str(DEFAULT_DETAILED_HISTORY_DROP_BLOCK_SIZE),
        )
    )
    if window <= 0:
        raise ValueError("detailed history turns must be positive")
    if drop_block_size <= 0 or drop_block_size > window:
        raise ValueError("detailed history drop block size must be in [1, detailed history turns]")
    return window, drop_block_size


def effective_visual_fold_policy() -> tuple[int, int]:
    """Return the independent active-image budget and visual folding block size.

    At ``budget + 1`` active visual observations, the oldest ``block_size`` screenshots and their
    bound interactables are replaced together by fixed text. The matching text/action turns remain
    until the separate detailed-history policy archives them.
    """
    budget = int(
        os.environ.get(
            "STREAM_ACTIVE_VISUAL_BUDGET",
            str(DEFAULT_ACTIVE_VISUAL_BUDGET),
        )
    )
    block_size = int(
        os.environ.get(
            "STREAM_VISUAL_FOLD_BLOCK_SIZE",
            str(DEFAULT_VISUAL_FOLD_BLOCK_SIZE),
        )
    )
    if budget <= 0:
        raise ValueError("active visual budget must be positive")
    if block_size <= 0 or block_size > budget:
        raise ValueError("visual fold block size must be in [1, active visual budget]")
    return budget, block_size


@dataclass(frozen=True)
class DecisionBudget:
    """Physical generation limits for one frame decision.

    The native model limit is additionally bounded by the remaining episode capacity under a finite
    token clock. This is a terminal horizon bound, not the nominal per-observation rate ``B``: a
    decision may still consume more than ``B`` and cross intermediate observation boundaries.
    """

    runtime_output_tokens: int | None
    provider_output_tokens: int | None
    transport_output_tokens: int | str | None
    clock_mode: Literal["token", "wall"]


class EnvironmentConfig(BaseModel):
    # Host ports default from env so parallel lanes can each bind a distinct set
    # (configured by the Harbor trial worker).
    vm_image: str = os.environ.get("SANDBOX_IMAGE", ".cache/apptainer/sandbox.sif")
    screen_width: int = 1920
    screen_height: int = 1080
    guest_agent_port: int = int(os.environ.get("GUEST_AGENT_PORT", "5000"))
    vnc_port: int = int(os.environ.get("VNC_PORT", "5900"))
    novnc_port: int = int(os.environ.get("NOVNC_PORT", "5910"))
    display_number: int = SANDBOX_DISPLAY_NUMBER


@dataclass(frozen=True)
class EnvironmentBudgetConfig:
    """Overall time budget in seconds of each environment operation (sandbox/budget.py).

    The loop's operations are in sandbox/actions.py; the snapshot read at the freeze is in the
    shells (evaluation/shell.py, evaluation/streaming_shell.py) and the legacy loop
    (evaluation/runner.py). Running past a budget fails the operation and, as a rule,
    invalidates the run. The exceptions: a refusal of the model's action that comes after the
    action's budget is still a refusal, the guest's or the runtime's own after a wait for the
    wall lock; a state read the kernel counts fails only that read, and three failed reads in a
    row invalidate the run; at the cleanup after the outcome is latched (release_inputs), where
    only a game's key releases under the token clock have a budget, a failure is only logged.
    """

    # Execute one action through the guest or xdotool, from its checks to the page's action
    # barrier and the hold snapshot; the wall clock's key lock is waited for whole and judged at
    # the end. Text input gets the guest's longer request timeout (sandbox/guest.py
    # action_timeout); a valid timed action, at most 10 s, stays within 20 s. A game's key action
    # under the token clock goes over CDP and is timed per key event instead.
    action_s: float = 20.0
    # Send one key event of a game under the token clock over CDP, a press or a release: an
    # action's, a lease's end or the cleanup's. A tap's frame between its press and its release is
    # a clock advance. The same 10 s the wall clock gives the guest's request for a key.
    key_event_s: float = 10.0
    # Capture one observation: the page probes, the guest's screenshots and the wait for the render
    # fence. As long as one CDP call may take by default.
    capture_s: float = 120.0
    state_read_s: float = 120.0  # ask the task whether the episode ended; the CDP call default
    # The same read when the kernel's shell asks on the wall clock, at the environment rate. Three
    # failed reads in a row invalidate the run, so a wedged environment ends it in about 30 s.
    wall_state_read_s: float = 10.0
    clock_advance_s: float = 120.0  # advance or render one token-clock step; the CDP call default
    # Read the scored snapshot at the freeze. The snapshot is valid when the whole read ends within
    # 16 s; no single read of it has a hard 8 s cap. How late it was is in the termination record.
    snapshot_read_s: float = 16.0


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter_type: str = "openai_realtime"
    model_name: str = "gpt-realtime-2"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    model_revision: str | None = None
    chat_stream: bool = True
    chat_include_usage: bool = False
    # Physical input capacity, independent of the benchmark's token clock. An estimated
    # counter is explicitly approximate; provider counters can include visual/template tokens.
    context_window_tokens: int | None = Field(default=None, gt=0)
    context_counter: Literal["estimate", "vllm"] = "estimate"
    context_image_tokens: int = Field(default=4096, gt=0)
    context_reserve_tokens: int = Field(default=4096, gt=0)
    # Physical provider/model capability, not a benchmark token budget. None means undeclared:
    # requests omit the completion limit and the provider default applies.
    native_output_limit: Optional[int] = Field(default=None, gt=0)
    native_output_limit_source: Optional[Literal["endpoint", "litellm"]] = None
    fps: float = 1.0  # model-input frame rate (runner.py --model-input-fps)
    # Retain the effective contract in runtime identity, but never send a value.
    temperature_policy: Literal["provider_default"] = "provider_default"
    temperature: None = None
    reasoning_effort: Optional[str] = "low"
    # Driver-specific wire options from the driver class's model_defaults (for example Gemini Live
    # tool scheduling). Each driver validates its own keys; the context policy reads history_format.
    provider_options: dict = Field(default_factory=dict)
    # The context-policy preset the model takes in place of its API row's (one of
    # entity/context_policy.py MODEL_PRESETS), from its model YAML's context_preset.
    context_preset: str | None = None
    # OpenRouter's concrete upstream provider slug. When set, the transport restricts routing to
    # this provider, disables fallbacks, and requires support for every requested parameter.
    # Keeping it in ModelConfig makes the route part of the effective runtime-config hash.
    upstream_provider: Optional[str] = None


# Settings read from the environment. Each field names its variable and parser; an unset variable
# keeps the field default. The module that uses a group reads it once, at its import, so the
# process environment at that moment decides the value (the runtime-config hash reads the same
# variables again when it is computed: evaluation/config.py:678-680, 720).
from typing import ClassVar as ClassVar


def _environ_fields(fields: dict, environ) -> dict:
    return {
        field: parse(environ[variable])
        for field, (variable, parse) in fields.items()
        if variable in environ
    }


def _at_least_two(value: str) -> int:
    return max(2, int(value))


@dataclass(frozen=True)
class FrameEncodingConfig:
    """How every model's screenshots are encoded (sandbox/observation/capture.py)."""

    max_dimension: int = DEFAULT_FRAME_MAX_DIM  # longest side; larger frames are scaled down
    jpeg_quality: int = 85  # JPEG quality of a scaled frame
    image_detail: str = "auto"  # image detail sent with each screenshot

    ENVIRONMENT: ClassVar[dict] = {
        "max_dimension": ("FRAME_MAX_DIM", int),
        "jpeg_quality": ("FRAME_JPEG_QUALITY", int),
        "image_detail": ("FRAME_IMAGE_DETAIL", str),
    }

    @classmethod
    def from_environ(cls, environ=os.environ) -> FrameEncodingConfig:
        return cls(**_environ_fields(cls.ENVIRONMENT, environ))


@dataclass(frozen=True)
class RunnerTimingConfig:
    """Operational bounds of the decision loop (evaluation/runner.py)."""

    abandoned_drain_s: float = 30.0  # wait for per-move workers a timeout abandoned
    observation_freeze_n: int = 3  # identical observations that make a frozen stream, at least 2
    provider_watchdog_s: float = 300.0  # token-clock provider watchdog; off when <= 0

    ENVIRONMENT: ClassVar[dict] = {
        "abandoned_drain_s": ("STREAM_ABANDONED_DRAIN_S", float),
        "observation_freeze_n": ("STREAM_OBSERVATION_FREEZE_N", _at_least_two),
        "provider_watchdog_s": ("STREAM_PROVIDER_WATCHDOG_S", float),
    }

    @classmethod
    def from_environ(cls, environ=os.environ) -> RunnerTimingConfig:
        return cls(**_environ_fields(cls.ENVIRONMENT, environ))
