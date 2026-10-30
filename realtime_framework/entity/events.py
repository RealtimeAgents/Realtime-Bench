"""Provider-independent stream event types."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # entity.model imports this module
    from realtime_framework.entity.model import TransportFailure


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
class TextDone(ProviderEvent):
    """One output text is complete, all of it in ``text``: an action in it arrived then
    (decisions 19 and 4.17). For a driver that reads every message as it arrives, when it read
    the message, as for FunctionCall."""

    text: str = ""
    received_at: float = field(kw_only=True)


@dataclass(frozen=True)
class FunctionCall(ProviderEvent):
    call_id: str = ""
    name: str = ""
    arguments: dict = field(default_factory=dict)
    # For a driver that reads every message as it arrives: when it read the call's message, on
    # its operational clock, which is when the call came off the wire.
    received_at: float | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class FunctionCallsCancelled(ProviderEvent):
    """The provider cancelled function calls, which are not to be executed; the reply goes on.
    They are told by call id: they may be an earlier reply's than the one this event comes in."""

    call_ids: tuple[str, ...] = ()
    received_at: float | None = field(default=None, kw_only=True)  # as for FunctionCall


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
    # What the error means, as the driver read it off the provider's own error: the failure that
    # ends the generation, which the shared retry takes as it is (entity/model.py raised_as); None
    # for one that leaves the generation going, as an error about a cancel that came too late.
    failure: TransportFailure | None = field(kw_only=True)


class EmptyInterruptedToolTurnError(RuntimeError):
    """A provider interruption with exactly zero generated native output."""
