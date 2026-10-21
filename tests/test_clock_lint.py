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

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUFF = Path(sys.executable).with_name("ruff")
LINT = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["ruff"]["lint"]
EXEMPT = [pattern for pattern, rules in LINT["per-file-ignores"].items() if "TID251" in rules]
LOOP_METHODS = {"time", "call_later", "call_at"}
# Calls that return an event loop: asyncio's, a Clock's, and a future's or task's get_loop().
LOOP_CALLS = {"get_running_loop", "get_event_loop", "new_event_loop", "get_loop"}

needs_ruff = pytest.mark.skipif(not RUFF.exists(), reason=f"ruff is not installed at {RUFF}")


def _ruff(*args: str, source: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(RUFF), "check", "--select", "TID251", "--no-cache", *args],
        cwd=ROOT,
        input=source,
        capture_output=True,
        text=True,
    )


@needs_ruff
def test_framework_reads_the_time_only_through_the_clock():
    result = _ruff("realtime_framework", "realtime_tasks", "tests")
    assert result.returncode == 0, result.stdout + result.stderr


@needs_ruff
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


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _makes_loop(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and _name(node.func) in LOOP_CALLS


def loop_time_calls(source: str) -> list[int]:
    """Lines that call time(), call_later() or call_at() on an event loop: a name or attribute
    that ends in "loop" or is assigned a call in LOOP_CALLS, or such a call itself."""
    tree = ast.parse(source)
    bound = {
        _name(target)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and _makes_loop(node.value)
        for target in node.targets
    }

    def is_loop(node: ast.expr) -> bool:
        name = _name(node)
        if name is not None:
            return name.endswith("loop") or name in bound
        return _makes_loop(node)

    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in LOOP_METHODS
        and is_loop(node.func.value)
    ]


def test_loop_time_calls_are_found():
    source = (
        "import asyncio\n"
        "loop = asyncio.get_running_loop()\n"
        "loop.time()\n"
        "asyncio.get_running_loop().call_later(1, print)\n"
        "self._loop.call_at(1, print)\n"
        "clock.time()\n"
        "lp = asyncio.get_running_loop()\n"
        "lp.time()\n"
        "lp.call_later(1, print)\n"
        "future.get_loop().time()\n"
        "self.lp = clock.new_event_loop()\n"
        "self.lp.call_at(1, print)\n"
    )
    assert loop_time_calls(source) == [3, 4, 5, 8, 9, 10, 12]


def test_no_loop_time_outside_the_clock():
    offenders = []
    for path in sorted((ROOT / "realtime_framework").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in EXEMPT):
            continue
        offenders += [f"{relative}:{line}" for line in loop_time_calls(path.read_text())]
    assert not offenders, offenders
