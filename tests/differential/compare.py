"""One differential run: a script through a replay driver and through the simulation, compared
under the expected differences."""

from __future__ import annotations

import warnings
from pathlib import Path

from . import drivers
from .replayed import replay
from .rules import Verdict, check
from .script import Script
from .simulated import simulate

# Verdicts of the session, for the summary conftest.py prints.
VERDICTS: list[tuple[Verdict, bool]] = []


class ExpectationNotShown(UserWarning):
    """A named script expects a difference it did not show."""


def compare(script: Script, run_dir: Path, driver=drivers.legacy) -> Verdict:
    return check(script, replay(script, run_dir, driver), simulate(script))


def record(verdict: Verdict, *, named: bool) -> None:
    """Keep the verdict for the summary. A random script may show any expected difference, so
    only a named one warns about those it expects and did not show."""
    VERDICTS.append((verdict, named))
    if named and verdict.unused:
        warnings.warn(
            f"{verdict.script.name}: expected {', '.join(verdict.unused)}, not shown",
            ExpectationNotShown,
            stacklevel=2,
        )


def describe(verdict: Verdict, limit: int = 40) -> str:
    lines = [f"{verdict.script.name or verdict.script!r}"]
    if verdict.unexpected:
        lines.append(f"differences the script does not expect: {', '.join(verdict.unexpected)}")
    if verdict.problems:
        lines.append(f"{len(verdict.problems)} difference(s) no rule explains:")
        lines += [f"  {problem}" for problem in verdict.problems[:limit]]
    return "\n".join(lines)
