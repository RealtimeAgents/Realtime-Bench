"""What a differential run compares, in one shape for every side (wave2-plan.md, P4 投影).

Decisions are keyed by their number, never zipped by position, so one decision more or less on a
side does not shift every later one. ``compared`` holds what must agree once the expected
differences are applied: per decision its observation, allowance, proposed action, the two clock
moves, the resolution, what the environment executed, the commit and the lifecycle outcome; per
episode the captures, the token clock's end checks, the key releases, the aborts, the
invalidation, the termination and where the snapshot fell among the commits. ``beyond`` holds what
only the two replayed sides have: the whole requests, the agent's usage and the trace records the
decisions leave out, compared between the shell and today's loop. ``facts`` holds what only the
rules read: the times things happened and side-specific detail, never compared.

Times are world seconds from the episode origin, ticks are environment ticks. Sequence numbers,
real time, record ids and fields only one side writes are left out.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

# Floats agree within this: the runner and the kernel spell the frame arithmetic differently
# (wave2-plan.md, X4), so the same tick can differ by a few ulps.
TOLERANCE = 1e-9

DECISION_FIELDS = (
    "observed",  # the observation's tick
    "remaining",  # the finite-B allowance bound to the generation
    "proposed",  # the action proposed, as script.action_key spells it
    "approach",  # ClockAdvanced(inference_to_action)
    "resolved",  # ActionResolved
    "wait_record",  # a resolved wait's execution result
    "applied",  # whether ActionApplied was written
    "executed",  # what the environment was asked to execute, and when
    "boundary",  # ClockAdvanced(action_to_observation_boundary)
    "commit",  # what the agent was told
    "outcome",  # the lifecycle record
)


def decision() -> dict:
    return {name: None for name in DECISION_FIELDS} | {"applied": False, "executed": []}


@dataclass
class Projection:
    side: str
    compared: dict
    facts: dict = field(default_factory=dict)
    beyond: dict | None = None

    @property
    def decisions(self) -> dict[int, dict]:
        return self.compared["decisions"]

    @property
    def episode(self) -> dict:
        return self.compared["episode"]

    def copy(self) -> Projection:
        return Projection(
            self.side, copy.deepcopy(self.compared), self.facts, copy.deepcopy(self.beyond)
        )


def earlier(a: float, b: float) -> bool:
    """``a`` comes before ``b`` by more than the tolerance: the replay side's wall times are
    clock readings less the origin, a few ulps off the script's own arithmetic."""
    return a < b - TOLERANCE


def ends_the_run(in_tap: bool, answer: str | None) -> bool:
    """Whether the kernel's episode is over after a token state read that answered ``answer``
    (None, "task", "failed" or "invalid_run"): an end check that saw the end or did not read; a
    read inside a tap only when it failed for the last time, since one that sees the end only
    stops the tap and the end check after it tells the kernel (turns.py:471-474)."""
    return answer == "failed" if in_tap else bool(answer)


def key_releases(released: list[dict], until: float) -> list[dict]:
    """One entry per key released up to the episode's end ``until``, in time order: a chord's
    keys can go up in one call or one call each, and what goes up later belongs to the wind-down,
    which each side runs for its own time."""
    return sorted(
        (
            {"at": item["at"], "key": key}
            for item in released
            for key in item["keys"]
            if not earlier(until, item["at"])
        ),
        key=lambda item: (item["at"], item["key"]),
    )


def differences(old, new, path: str = "") -> tuple[list[str], list[str]]:
    """Paths where ``old`` and ``new`` differ, and paths where two floats differ only within
    TOLERANCE."""
    problems: list[str] = []
    close: list[str] = []
    _walk(old, new, path, problems, close)
    return problems, close


def _walk(old, new, path: str, problems: list[str], close: list[str]) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new), key=str):
            where = f"{path}.{key}" if path else str(key)
            if key not in old:
                problems.append(f"{where}: missing on the old side, new {_short(new[key])}")
            elif key not in new:
                problems.append(f"{where}: missing on the new side, old {_short(old[key])}")
            else:
                _walk(old[key], new[key], where, problems, close)
        return
    if isinstance(old, list | tuple) and isinstance(new, list | tuple):
        if len(old) != len(new):
            problems.append(
                f"{path}: {len(old)} item(s) -> {len(new)}: {_short(old)} -> {_short(new)}"
            )
            return
        for index, (a, b) in enumerate(zip(old, new, strict=True)):
            _walk(a, b, f"{path}[{index}]", problems, close)
        return
    if _number(old) and _number(new) and (isinstance(old, float) or isinstance(new, float)):
        if old == new:
            return
        if math.isclose(old, new, rel_tol=0.0, abs_tol=TOLERANCE):
            close.append(f"{path}: {old!r} ~ {new!r}")
            return
    if old != new or type(old) is not type(new) and not (_number(old) and _number(new)):
        problems.append(f"{path}: {_short(old)} -> {_short(new)}")


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _short(value, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
