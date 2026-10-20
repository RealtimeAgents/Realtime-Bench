"""The clock rule: framework code reads the time and waits only through the episode clocks
(realtime_framework/clock.py). pyproject.toml bans the module functions with ruff's TID251 and
lists the files it exempts; ruff cannot see a call on an event loop object, so loop.time(),
loop.call_later() and loop.call_at() are found here, outside the same exempt files."""

from __future__ import annotations

import ast
import fnmatch
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUFF = Path(sys.executable).with_name("ruff")
LINT = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["ruff"]["lint"]
EXEMPT = [pattern for pattern, rules in LINT["per-file-ignores"].items() if "TID251" in rules]
LOOP_METHODS = {"time", "call_later", "call_at"}


def _ruff(*args: str, source: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(RUFF), "check", "--select", "TID251", "--no-cache", *args],
        cwd=ROOT,
        input=source,
        capture_output=True,
        text=True,
    )


def test_framework_reads_the_time_only_through_the_clock():
    result = _ruff("realtime_framework", "realtime_tasks", "tests")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_rule_reports_a_direct_call_in_framework_code():
    source = (
        "import asyncio, threading, time\n\n\n"
        "async def f():\n"
        "    time.sleep(1)\n"
        "    threading.Timer(1, print)\n"
        "    await asyncio.sleep(1)\n"
    )
    result = _ruff("--stdin-filename", "realtime_framework/sandbox/actions.py", "-", source=source)
    assert result.returncode == 1
    assert result.stdout.count("TID251") == 3, result.stdout


def _is_loop(node: ast.expr) -> bool:
    """A name or attribute that ends in "loop", or asyncio.get_running_loop()/get_event_loop()."""
    if isinstance(node, ast.Name):
        return node.id.endswith("loop")
    if isinstance(node, ast.Attribute):
        return node.attr.endswith("loop")
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"get_running_loop", "get_event_loop"}
    )


def loop_time_calls(source: str) -> list[int]:
    """Lines that call time(), call_later() or call_at() on an event loop."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in LOOP_METHODS
        and _is_loop(node.func.value)
    ]


def test_loop_time_calls_are_found():
    source = (
        "import asyncio\n"
        "loop = asyncio.get_running_loop()\n"
        "loop.time()\n"
        "asyncio.get_running_loop().call_later(1, print)\n"
        "self._loop.call_at(1, print)\n"
        "clock.time()\n"
    )
    assert loop_time_calls(source) == [3, 4, 5]


def test_no_loop_time_outside_the_clock():
    offenders = []
    for path in sorted((ROOT / "realtime_framework").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in EXEMPT):
            continue
        offenders += [f"{relative}:{line}" for line in loop_time_calls(path.read_text())]
    assert not offenders, offenders
