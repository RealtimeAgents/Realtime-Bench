"""The harness hash covers the framework's source (phase2.md: harness 的哈希覆盖框架源码): a change to
any of its modules makes a run +dev until the next release locks the new hash (decision 36 relies
on it for the environment budgets). Of the rest of the repository it covers only the runtime
dependencies uv.lock pins: tests, replay recordings and the legacy loop archived under tests/
(decision 11) can change while a released harness stays released."""

from __future__ import annotations

from pathlib import Path

import realtime_framework
from realtime_framework.evaluation import config
from realtime_framework.evaluation.config import _source_paths

PACKAGE = Path(realtime_framework.__file__).resolve().parent


def test_every_framework_module_is_in_the_harness_fingerprint():
    modules = {path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts}
    assert modules
    assert sorted(modules - set(_source_paths())) == []


def test_the_harness_fingerprint_takes_in_nothing_outside_the_framework():
    assert [path for path in _source_paths() if PACKAGE not in path.parents] == []


def test_the_sandbox_shell_scripts_are_in_the_harness_fingerprint():
    """The scripts the sandbox image runs (sandbox/images/scripts) are framework code too."""
    scripts = set(PACKAGE.rglob("*.sh"))
    assert scripts
    assert sorted(scripts - set(_source_paths())) == []


def test_a_runtime_dependency_upgrade_is_another_harness(monkeypatch):
    """Another version of a package uv.lock resolves for the framework (config.runtime_dependencies)
    changes the harness hash as a change to a module does."""
    before = config.harness_fingerprint()
    first, *rest = config.runtime_dependencies()
    upgraded = {**first, "version": first["version"] + ".post1"}
    monkeypatch.setattr(config, "runtime_dependencies", lambda: [upgraded, *rest])
    assert config.harness_fingerprint() != before


def test_only_the_locked_hash_is_the_released_harness():
    """pyproject.toml locks the released version's hash (config.harness_lock); any other hash, as
    a change to a framework module gives, is that version's +dev (config.harness_version)."""
    version, locked = config.harness_lock()
    assert config.harness_version(locked) == version
    assert config.harness_version("0" * 64) == version + config.HARNESS_UNRELEASED
