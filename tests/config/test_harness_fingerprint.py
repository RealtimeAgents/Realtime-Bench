"""The harness hash covers the framework's source (phase2.md: harness 的哈希覆盖框架源码): a change to
any of its modules makes a run +dev until the next release locks the new hash (decision 36 relies
on it for the environment budgets). It covers nothing else: tests, replay recordings and the
legacy loop archived under tests/ (decision 11) can change while a released harness stays
released."""

from __future__ import annotations

from pathlib import Path

import realtime_framework
from realtime_framework.evaluation.config import _source_paths

PACKAGE = Path(realtime_framework.__file__).resolve().parent


def test_every_framework_module_is_in_the_harness_fingerprint():
    modules = {path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts}
    assert modules
    assert sorted(modules - set(_source_paths())) == []


def test_the_harness_fingerprint_takes_in_nothing_outside_the_framework():
    assert [path for path in _source_paths() if PACKAGE not in path.parents] == []
