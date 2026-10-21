"""Provider-independent stream event types."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class ProviderHistoryState(str, Enum):
    """What is known about the provider/native context after an operation."""

    EXACT = "exact"
    CONTAMINATED = "contaminated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderEvent:
    generation_id: str
    sequence: int
    raw_type: str = ""
    # For a driver that numbers them: the connection the event arrived on, and which generation
    # on that connection it belongs to.
    connection: int | None = field(default=None, kw_only=True)
    generation: int | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class TextDelta(ProviderEvent):
    text: str = ""
    channel: Literal["visible", "reasoning"] = "visible"
    native_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class FunctionCall(ProviderEvent):
    call_id: str = ""
    name: str = ""
    arguments: dict = field(default_factory=dict)
    # For a driver that reads every message as it arrives: when it read the call's message, on
    # its operational clock, which is when the call came off the wire.
    received_at: float | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class UsageSnapshot(ProviderEvent):
    usage: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationBoundary(ProviderEvent):
    kind: Literal[
        "started",
        "generation_complete",
        "turn_complete",
        "interrupted",
        "go_away",
    ] = "generation_complete"
    history_state: ProviderHistoryState = ProviderHistoryState.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderError(ProviderEvent):
    detail: str = ""
    fatal: bool = True


class EmptyInterruptedToolTurnError(RuntimeError):
    """A provider interruption with exactly zero generated native output."""
