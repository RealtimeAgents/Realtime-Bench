"""After the differential tests, say which expected differences each script showed, and which
rules a named script expects that it did not show."""

from __future__ import annotations

from .compare import VERDICTS


def pytest_terminal_summary(terminalreporter):
    if not VERDICTS:
        return
    write = terminalreporter.write_line
    terminalreporter.section("expected differences")
    for verdict, named in VERDICTS:
        name = verdict.script.name or "(random)"
        line = f"{name}: {', '.join(verdict.triggered) or 'none'}"
        if named and verdict.unused:
            line += f"  [expected, not shown: {', '.join(verdict.unused)}]"
        write(line)
    shown = sorted({rule for verdict, _ in VERDICTS for rule in verdict.triggered})
    write(f"shown at least once: {', '.join(shown)}")
