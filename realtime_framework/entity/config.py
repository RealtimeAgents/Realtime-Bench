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
    41 and 50 turns while older executed actions remain available through the compact ledger.
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
    fps: float = 1.0
    # Retain the effective contract in runtime identity, but never send a value.
    temperature_policy: Literal["provider_default"] = "provider_default"
    temperature: None = None
    reasoning_effort: Optional[str] = "low"
    # Driver-specific wire options from the maintained model profile (for example Gemini Live
    # tool scheduling). Each driver validates its own keys; the harness never reads them.
    provider_options: dict = Field(default_factory=dict)
    # OpenRouter's concrete upstream provider slug. When set, the transport restricts routing to
    # this provider, disables fallbacks, and requires support for every requested parameter.
    # Keeping it in ModelConfig makes the route part of the effective runtime-config hash.
    upstream_provider: Optional[str] = None
