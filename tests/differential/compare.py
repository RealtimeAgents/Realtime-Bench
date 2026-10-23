"""One differential run: a script played three ways, on the kernel's simulation, through the
kernel's shell and through today's loop (wave2-plan.md, P4b). The shell and the simulation agree
exactly; today's loop agrees with the shell once the expected differences the script names are
applied, in its requests, usage and trace records too."""

from __future__ import annotations

from pathlib import Path

from . import drivers
from .projection import Projection, differences
from .replayed import replay
from .rules import Verdict, check, x12_plan, x12_probe
from .script import Script
from .simulated import simulate

# Verdicts of the session, for the summary conftest.py prints.
VERDICTS: list[tuple[Verdict, bool]] = []


def compare(script: Script, run_dir: Path) -> Verdict:
    shell = replay(script, _subdir(run_dir, "shell"), drivers.shell)
    verdict = check(script, _legacy(script, run_dir), shell)
    simulated = simulate(script)
    verdict.apart, _ = differences(shell.compared, simulated.compared)
    verdict.apart += differences(shell.kernel, simulated.kernel, "kernel")[0]
    return verdict


def _legacy(script: Script, run_dir: Path) -> Projection:
    """A script with failed state reads first plays today's loop once with every failure taken
    as a read that saw no end, to learn the order of its reads; then today's loop plays what X12
    says the kernel does with them (rules.x12_plan)."""
    if not script.fails_reads:
        return replay(script, _subdir(run_dir, "legacy"), drivers.legacy)
    probe = replay(x12_probe(script), _subdir(run_dir, "x12-probe"), drivers.legacy)
    plan = x12_plan(script, probe)
    old = replay(plan.script, _subdir(run_dir, "x12-plan"), drivers.legacy)
    old.facts["x12_end"], old.facts["x12_shown"] = plan.end_at, plan.shown
    return old


def _subdir(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    path.mkdir()
    return path


def record(verdict: Verdict, *, named: bool) -> None:
    """Keep the verdict for the summary."""
    VERDICTS.append((verdict, named))


def as_expected(verdict: Verdict) -> bool:
    """A named script also shows every difference it expects; a random one may show any."""
    return verdict.ok and not verdict.unused


def describe(verdict: Verdict, limit: int = 40, *, named: bool = False) -> str:
    lines = [f"{verdict.script.name or verdict.script!r}"]
    if verdict.apart:
        lines.append(f"{len(verdict.apart)} difference(s) between the shell and the simulation:")
        lines += [f"  {problem}" for problem in verdict.apart[:limit]]
    if verdict.unexpected:
        lines.append(f"differences the script does not expect: {', '.join(verdict.unexpected)}")
    if named and verdict.unused:
        lines.append(f"differences the script expects, not shown: {', '.join(verdict.unused)}")
    if verdict.problems:
        lines.append(f"{len(verdict.problems)} difference(s) no rule explains:")
        lines += [f"  {problem}" for problem in verdict.problems[:limit]]
    return "\n".join(lines)
