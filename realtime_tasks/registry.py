from __future__ import annotations

# Suite paths
"""Public manifests are the default; maintainer manifests require explicit selection."""
import re as re
import tomllib as tomllib
from pathlib import Path as Path

TASK_CONFIG = "task.toml"
# Task-owned success criterion. It sits with Harbor's tests/test.sh but runs in the runner,
# against the live snapshot frozen at the deadline; Harbor never executes it.
EVALUATION = "tests/evaluation.py"


def task_metadata(path):
    """Return the benchmark definition kept in a Harbor ``task.toml`` ``[metadata]`` table."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8")).get("metadata")
    if not isinstance(data, dict):
        raise ValueError(f"{path}: [metadata] must be a table")
    return data


# A released benchmark version is MAJOR.MINOR and is tagged in git as RELEASE_TAG. Older versions
# are reproduced from their tag; a checkout carries only its own version of every suite.
VERSION_PATTERN = r"[0-9]+\.[0-9]+"
RELEASE_TAG = "rtbench-v{version}"
# Appended to the declared version when task content no longer matches the locked suite_hash.
UNRELEASED = "+dev"


def parse_suite(selector):
    """Split ``NAME`` or ``NAME@MAJOR.MINOR`` into the suite id and the pinned version (or None)."""
    match = (
        re.fullmatch(rf"([A-Za-z0-9][A-Za-z0-9_.-]*)(?:@({VERSION_PATTERN}))?", selector)
        if isinstance(selector, str)
        else None
    )
    if match is None:
        raise ValueError(f"Invalid suite {selector!r}; use NAME or NAME@MAJOR.MINOR")
    return match.group(1), match.group(2)


def suite_path(root, selector):
    """Return the manifest of ``NAME[@VERSION]``; a pin must match this checkout's version."""
    import yaml

    suite_id, pinned = parse_suite(selector)
    path = Path(root) / "realtime_tasks" / "sets" / f"{suite_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Suite not found: {path}")
    if pinned is not None:
        declared = (yaml.safe_load(path.read_text()) or {}).get("version")
        if declared != pinned:
            raise ValueError(
                f"This checkout has {suite_id}@{declared}, not {suite_id}@{pinned}; "
                f"check out git tag {RELEASE_TAG.format(version=pinned)}"
            )
    return path


# Suites
"""Versioned suite-manifest loading, validation, and semantic fingerprinting.

A suite manifest selects an ordered subset of the task definitions registered by the scenario
modules. Public categories are presentation groupings; execution kinds are stable runner routing
keys. Keeping those concepts separate lets future suites reorganize their public taxonomy without
changing each task object's runtime ``category``.
"""

from dataclasses import dataclass as dataclass
import hashlib as hashlib
import json as json
from typing import Mapping as Mapping

import yaml


_ROOT = Path(__file__).resolve().parents[1]
_SUITE_DIR = _ROOT / "realtime_tasks" / "sets"
_TASK_DIR = _ROOT / "realtime_tasks"

# A routing kind has exactly one task runtime category. The manifest is public metadata, not an
# escape hatch for sending a task through an incompatible runner path.
_EXECUTION_RUNTIME_CATEGORIES = {
    "game": "browser_game",
    "web": "realtime",
}

# Source provenance is intentionally read as files rather than imported. In particular, loading a
# suite must never initialize model/provider adapters. Source bytes are newline-normalized so the
# fingerprint stays identical across operating systems and Python interpreter versions.
_COMMON_EXECUTION_SOURCES = ("realtime_tasks/types.py", "realtime_tasks/registry.py")
_EXECUTION_SOURCES = {
    "game": ("realtime_tasks/game/runtime.py", "realtime_tasks/game/verifier.py"),
    "web": ("realtime_tasks/web/runtime.py", "realtime_tasks/web/verifier.py", "realtime_tasks/web/app.py"),
}
_EXECUTION_SOURCE_DIRS = {
    "game": ("realtime_tasks/game/assets",),
    "web": ("realtime_tasks/web/static", "realtime_tasks/web/engines"),
}


class SuiteManifestError(ValueError):
    """A suite manifest is missing, malformed, or inconsistent with the task registry."""


@dataclass(frozen=True)
class SuiteCategory:
    """One ordered task group sharing public and execution metadata."""

    public_category: str
    execution_kind: str
    task_ids: tuple[str, ...]


@dataclass(frozen=True)
class SuiteManifest:
    """Validated, ordered suite selection."""

    suite_id: str
    fingerprint: str
    expected_total: int
    expected_counts: tuple[tuple[str, int], ...]
    categories: tuple[SuiteCategory, ...]
    # The declared release, or ``<release>+dev`` when content differs from its locked suite_hash.
    version: str

    @property
    def hash(self) -> str:
        """Alias of ``fingerprint``, published as ``suite_hash`` in run metadata."""
        return self.fingerprint

    @property
    def released(self) -> bool:
        return not self.version.endswith(UNRELEASED)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task_id for category in self.categories for task_id in category.task_ids)

    @property
    def bucket_order(self) -> tuple[str, ...]:
        """Public categories in order of first appearance in ``categories``."""
        return tuple(dict.fromkeys(group.public_category for group in self.categories))

    @property
    def buckets(self) -> dict[str, list[str]]:
        buckets = {category: [] for category in self.bucket_order}
        for group in self.categories:
            buckets[group.public_category].extend(group.task_ids)
        return buckets

    def public_category(self, task_id: str) -> str:
        for group in self.categories:
            if task_id in group.task_ids:
                return group.public_category
        raise ValueError(f"Task {task_id!r} is not in suite {self.suite_id!r}")

    def execution_kind(self, task_id: str) -> str:
        for group in self.categories:
            if task_id in group.task_ids:
                return group.execution_kind
        raise ValueError(f"Task {task_id!r} is not in suite {self.suite_id!r}")


def _nonempty_string(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SuiteManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _expected_counts(data: dict, suite_id: str) -> dict[str, int]:
    raw = data.get("expected_counts")
    if not isinstance(raw, dict) or not raw:
        raise SuiteManifestError(f"Suite {suite_id!r} expected_counts must be a non-empty mapping")

    counts = {}
    for category, count in raw.items():
        category = _nonempty_string(category, "expected_counts category")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise SuiteManifestError(
                f"Suite {suite_id!r} expected count for {category!r} must be a non-negative integer"
            )
        if category in counts:
            raise SuiteManifestError(
                f"Suite {suite_id!r} repeats expected_counts category {category!r}"
            )
        counts[category] = count
    return counts


def _canonical_json(value) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SuiteManifestError(f"Suite metadata is not canonically serializable: {exc}") from exc
    return text.encode("utf-8")


def _task_definition_paths(task_ids, task_catalog_dir):
    result = {}
    for task_id in task_ids:
        found = [
            task_catalog_dir / family / task_id / TASK_CONFIG
            for family in ("game", "web")
            if (task_catalog_dir / family / task_id / TASK_CONFIG).is_file()
        ]
        if len(found) > 1:
            raise SuiteManifestError(f"Ambiguous task package: {task_id}")
        if found:
            result[task_id] = found[0]
    return result


def _normalized_source(path: Path) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SuiteManifestError(f"Could not read execution source {path}: {exc}") from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # Task assets can legitimately be binary. Hash them byte-for-byte while retaining newline
        # normalization for textual execution sources across operating systems.
        return data
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _semantic_fingerprint(
    manifest: dict,
    task_ids: tuple[str, ...],
    execution_kinds: tuple[str, ...],
    *,
    task_catalog_dir: Path,
    source_root: Path,
) -> str:
    digest = hashlib.sha256()

    def add(kind: str, name: str, content: bytes) -> None:
        # Length delimiters make this unambiguous even when names/content contain separators.
        for part in (kind.encode(), name.encode(), content):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)

    add("manifest", "normalized", _canonical_json(manifest))

    definitions = _task_definition_paths(task_ids, task_catalog_dir)
    selected_definitions = {}
    for task_id in task_ids:
        path = definitions.get(task_id)
        if path is None:
            # Synthetic callers may not have a task catalog; production tasks all do. Keep the
            # selected identity in the fingerprint anyway.
            add("task", task_id, _canonical_json({"task_id": task_id}))
            continue
        try:
            definition = task_metadata(path)
        except Exception as exc:
            raise SuiteManifestError(
                f"Could not load selected task definition {path}: {exc}"
            ) from exc
        if not isinstance(definition, dict):
            raise SuiteManifestError(f"Selected task definition {path} must contain a mapping")
        declared_id = _nonempty_string(definition.get("task_id"), f"{path} task_id")
        if declared_id != task_id:
            raise SuiteManifestError(
                f"Selected task definition {path} declares {declared_id!r}, expected {task_id!r}"
            )
        selected_definitions[task_id] = definition
        add("task", task_id, _canonical_json(definition))

    source_names = set(_COMMON_EXECUTION_SOURCES)
    selected_package_dirs = {
        str(path.parent.relative_to(source_root)) for path in definitions.values()
    }
    source_dirs = {
        relative_dir
        for execution_kind in execution_kinds
        for relative_dir in _EXECUTION_SOURCE_DIRS.get(execution_kind, ())
    }
    source_dirs.update(selected_package_dirs)
    for execution_kind in execution_kinds:
        source_names.update(_EXECUTION_SOURCES[execution_kind])
    if "game" in execution_kinds:
        # A game's vendored HTML/JS/assets are executable task content, not harness packaging.  Hash
        # each selected bundle so changing game physics, rendering, or its gameAPI shim necessarily
        # changes suite_hash and invalidates stale resumable results.
        for task_id, definition in selected_definitions.items():
            game_key = definition.get("game_key")
            if isinstance(game_key, str) and game_key.strip():
                source_dirs.add(f"realtime_tasks/game/{task_id}")
    for relative in sorted(source_names):
        path = source_root / relative
        if not path.is_file():
            raise SuiteManifestError(f"Configured execution source is missing: {path}")
        add("source", relative, _normalized_source(path))
    for relative_dir in sorted(source_dirs):
        directory = source_root / relative_dir
        if not directory.is_dir():
            raise SuiteManifestError(
                f"Configured execution source directory is missing: {directory}"
            )
        for path in sorted(directory.rglob("*")):
            relative_path = path.relative_to(source_root)
            if (
                not path.is_file()
                or any(
                    part.startswith(".") or part == "__pycache__" for part in relative_path.parts
                )
                or path.suffix == ".pyc"
            ):
                continue
            add("source", relative_path.as_posix(), _normalized_source(path))

    return digest.hexdigest()


def load_suite(
    suite_id: str,
    all_tasks: Mapping[str, object],
    *,
    catalog_dir: str | Path | None = None,
    task_catalog_dir: str | Path | None = None,
    source_root: str | Path | None = None,
) -> SuiteManifest:
    """Load ``realtime_tasks/sets/<suite_id>.yaml`` and validate it against ``all_tasks``.

    Category and task ordering is taken from the YAML. Multiple category records may use the same
    public category with different execution kinds. The returned hash is a semantic fingerprint of
    the normalized manifest, selected task definitions, and routing-relevant execution source.
    ``suite_id`` may pin a release as ``NAME@MAJOR.MINOR``; unreleased content then fails to load.
    """
    try:
        suite_id, pinned = parse_suite(suite_id)
    except ValueError as exc:
        raise SuiteManifestError(str(exc)) from exc

    suite_dir = Path(catalog_dir) if catalog_dir is not None else _SUITE_DIR
    path = suite_dir / f"{suite_id}.yaml"
    if not path.is_file():
        raise SuiteManifestError(f"Suite manifest not found: {path}")

    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:
        raise SuiteManifestError(f"Could not parse suite manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SuiteManifestError(f"Suite manifest {path} must contain a mapping")

    declared_id = _nonempty_string(data.get("suite_id"), "manifest suite_id")
    if declared_id != suite_id:
        raise SuiteManifestError(
            f"Suite manifest {path} declares {declared_id!r}, expected {suite_id!r}"
        )
    version = data.get("version")
    if not isinstance(version, str) or not re.fullmatch(VERSION_PATTERN, version):
        raise SuiteManifestError(f"Suite {suite_id!r} version must be a quoted MAJOR.MINOR string")
    locked_hash = data.get("suite_hash")
    if locked_hash is not None and (
        not isinstance(locked_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", locked_hash)
    ):
        raise SuiteManifestError(f"Suite {suite_id!r} suite_hash must be a SHA-256 hex digest")
    expected_total = data.get("expected_total")
    if (
        not isinstance(expected_total, int)
        or isinstance(expected_total, bool)
        or expected_total < 0
    ):
        raise SuiteManifestError(
            f"Suite {suite_id!r} expected_total must be a non-negative integer"
        )
    expected_by_category = _expected_counts(data, suite_id)

    raw_categories = data.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise SuiteManifestError(f"Suite {suite_id!r} categories must be a non-empty list")

    categories = []
    category_order = []
    seen_ids = set()
    actual_counts = {category: 0 for category in expected_by_category}
    for index, raw_category in enumerate(raw_categories):
        label = f"Suite {suite_id!r} categories[{index}]"
        if not isinstance(raw_category, dict):
            raise SuiteManifestError(f"{label} must be a mapping")
        public_category = _nonempty_string(
            raw_category.get("public_category"), f"{label}.public_category"
        )
        execution_kind = _nonempty_string(
            raw_category.get("execution_kind"), f"{label}.execution_kind"
        )
        if execution_kind not in _EXECUTION_RUNTIME_CATEGORIES:
            raise SuiteManifestError(
                f"{label} has unknown execution_kind {execution_kind!r}; expected one of "
                f"{sorted(_EXECUTION_RUNTIME_CATEGORIES)}"
            )
        if public_category not in expected_by_category:
            raise SuiteManifestError(f"{label} uses undeclared public category {public_category!r}")
        if public_category not in category_order:
            category_order.append(public_category)
        raw_ids = raw_category.get("task_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise SuiteManifestError(f"{label}.task_ids must be a non-empty list")

        task_ids = []
        required_runtime = _EXECUTION_RUNTIME_CATEGORIES[execution_kind]
        for task_id in raw_ids:
            task_id = _nonempty_string(task_id, f"{label}.task_ids entry")
            if task_id in seen_ids:
                raise SuiteManifestError(f"Suite {suite_id!r} has duplicate task id {task_id!r}")
            if task_id not in all_tasks:
                task_ids.append(task_id)
                seen_ids.add(task_id)
                continue
            runtime_category = getattr(all_tasks[task_id], "category", None)
            if runtime_category != required_runtime:
                raise SuiteManifestError(
                    f"Suite {suite_id!r} task {task_id!r} execution_kind {execution_kind!r} "
                    f"requires runtime category {required_runtime!r}, found {runtime_category!r}"
                )
            seen_ids.add(task_id)
            task_ids.append(task_id)
        actual_counts[public_category] += len(task_ids)
        categories.append(SuiteCategory(public_category, execution_kind, tuple(task_ids)))

    unknown = sorted(seen_ids - set(all_tasks))
    if unknown:
        raise SuiteManifestError(f"Suite {suite_id!r} has unknown task ids: {unknown}")

    missing_categories = sorted(set(expected_by_category) - set(category_order))
    if missing_categories:
        raise SuiteManifestError(
            f"Suite {suite_id!r} has expected_counts categories with no task group: {missing_categories}"
        )
    expected_counts = tuple(
        (category, expected_by_category[category]) for category in category_order
    )
    for category, expected in expected_counts:
        actual = actual_counts[category]
        if actual != expected:
            raise SuiteManifestError(
                f"Suite {suite_id!r} category {category!r} expected {expected} tasks, found {actual}"
            )

    actual_total = len(seen_ids)
    if actual_total != expected_total:
        raise SuiteManifestError(
            f"Suite {suite_id!r} expected {expected_total} tasks, found {actual_total}"
        )
    counts_total = sum(count for _category, count in expected_counts)
    if counts_total != expected_total:
        raise SuiteManifestError(
            f"Suite {suite_id!r} expected_counts sum to {counts_total}, not {expected_total}"
        )

    normalized_manifest = {
        "suite_id": suite_id,
        "expected_total": expected_total,
        "expected_counts": list(expected_counts),
        "categories": [
            {
                "public_category": group.public_category,
                "execution_kind": group.execution_kind,
                "task_ids": list(group.task_ids),
            }
            for group in categories
        ],
    }
    task_ids = tuple(task_id for group in categories for task_id in group.task_ids)
    execution_kinds = tuple(dict.fromkeys(group.execution_kind for group in categories))
    fingerprint = _semantic_fingerprint(
        normalized_manifest,
        task_ids,
        execution_kinds,
        task_catalog_dir=Path(task_catalog_dir) if task_catalog_dir is not None else _TASK_DIR,
        source_root=Path(source_root) if source_root is not None else _ROOT,
    )

    # The version and its lock are release labels, not content, so they stay out of the fingerprint.
    if fingerprint != locked_hash:
        version += UNRELEASED
    if pinned is not None and version != pinned:
        raise SuiteManifestError(
            f"{suite_id}@{pinned} was requested, but this checkout has {suite_id}@{version}; "
            f"check out git tag {RELEASE_TAG.format(version=pinned)}"
        )

    return SuiteManifest(
        suite_id=suite_id,
        fingerprint=fingerprint,
        expected_total=expected_total,
        expected_counts=expected_counts,
        categories=tuple(categories),
        version=version,
    )


"""Discover task packages and load their declared content without starting environments."""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "realtime_tasks"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def task_directory(family: str, task_id: str) -> Path:
    if family not in {"game", "web"} or not _ID.fullmatch(task_id):
        raise ValueError(f"Invalid task package: {family}/{task_id}")
    return TASK_ROOT / family / task_id


def load_definition(path: Path) -> dict:
    """Read metadata and the single authoritative instruction file."""
    data = task_metadata(path)
    if not isinstance(data, dict) or data.get("task_id") != path.parent.name:
        raise ValueError(f"{path}: task_id must match the package directory")
    evaluator = data.get("evaluation")
    if evaluator != EVALUATION or not (path.parent / evaluator).is_file():
        raise ValueError(f"{path}: evaluation must name the task-owned {EVALUATION}")
    if "prompt" in data:
        raise ValueError(f"{path}: put the prompt in the declared instruction file")
    instruction = data.get("instruction")
    if not isinstance(instruction, str) or not instruction:
        raise ValueError(f"{path}: instruction must name a package-local file")
    target = (path.parent / instruction).resolve()
    if not target.is_relative_to(path.parent.resolve()):
        raise ValueError(f"{path}: instruction escapes the task package")
    prompt = target.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"{target}: instruction must not be empty")
    return {**data, "prompt": prompt}


@lru_cache(maxsize=1)
def web_definitions() -> dict[str, dict]:
    definitions = {}
    for path in sorted((TASK_ROOT / "web").glob("*/" + TASK_CONFIG)):
        data = load_definition(path)
        if data.get("backend") not in {"product", "configured", "bespoke"}:
            raise ValueError(f"{path}: unknown web backend {data.get('backend')!r}")
        definitions[data["task_id"]] = data
    return definitions


@lru_cache(maxsize=None)
def load_task_module(task_id: str, relative: str, family: str = "web"):
    """Load a task-owned Python module by file, with a stable per-file module identity."""
    directory = task_directory(family, task_id).resolve()
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory) or path.suffix != ".py":
        raise ValueError(f"Invalid task module path: {relative!r}")
    name = "_realtimebench_task_" + re.sub(
        r"[^A-Za-z0-9_]", "_", family + "_" + task_id + "_" + relative
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load task module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


# Registry
"""Aggregate every scenario task, then select the active versioned suite.

Scenario modules own executable task definitions.  ``ALL_TASKS`` contains every definition they
export, while ``TASKS`` and ``BUCKETS`` contain only the ordered tasks selected by the manifest
named by ``TASK_SUITE`` (default: ``public``; ``NAME@MAJOR.MINOR`` pins a release).
"""
import importlib as importlib
import os as os

from realtime_tasks.types import Task


# Every top-level scenario module is loaded regardless of which suite is active.  A manifest selects
# from the resulting complete registry, so adding a future suite never changes task-definition code.
_SCENARIO_MODULES = [
    "realtime_tasks.game.runtime",
    "realtime_tasks.web.runtime",
]


def _merge_tasks(target: dict, source: dict, module_name: str) -> None:
    for task_id, task in source.items():
        if task_id in target:
            raise ValueError(f"Duplicate registered task id {task_id!r} from {module_name}")
        target[task_id] = task


def _load_all_tasks() -> dict:
    """Return every task the scenario modules export.

    Scenario imports remain task-only: neither suite discovery nor fingerprinting imports model
    adapters/provider SDKs.
    """
    all_tasks = {}
    for module_name in _SCENARIO_MODULES:
        module = importlib.import_module(module_name)
        _merge_tasks(all_tasks, module.TASKS, module_name)
    return all_tasks


_REGISTRY_READY = False


def _initialize_registry():
    global _REGISTRY_READY, ALL_TASKS, _ACTIVE_SUITE, ACTIVE_SUITE
    global SUITE_ID, SUITE_HASH, SUITE_VERSION, TASKS, BUCKET_ORDER, BUCKETS
    if _REGISTRY_READY:
        return
    ALL_TASKS = _load_all_tasks()
    _suite_name = (os.environ.get("TASK_SUITE") or "").strip() or "public"
    _ACTIVE_SUITE = load_suite(_suite_name, ALL_TASKS)
    ACTIVE_SUITE = _ACTIVE_SUITE
    SUITE_ID = _ACTIVE_SUITE.suite_id
    SUITE_HASH = _ACTIVE_SUITE.hash
    SUITE_VERSION = _ACTIVE_SUITE.version

    TASKS = {task_id: ALL_TASKS[task_id] for task_id in _ACTIVE_SUITE.task_ids}
    BUCKET_ORDER = list(_ACTIVE_SUITE.bucket_order)
    BUCKETS = _ACTIVE_SUITE.buckets

    _REGISTRY_READY = True


def __getattr__(name):
    if name in {
        "ALL_TASKS",
        "ACTIVE_SUITE",
        "SUITE_ID",
        "SUITE_HASH",
        "SUITE_VERSION",
        "TASKS",
        "BUCKET_ORDER",
        "BUCKETS",
    }:
        _initialize_registry()
        return globals()[name]
    raise AttributeError(name)


def suite_id() -> str:
    """Identifier of the active suite manifest."""
    _initialize_registry()
    return SUITE_ID


def suite_hash() -> str:
    """Semantic SHA-256 fingerprint of the active suite and its execution definitions."""
    _initialize_registry()
    return SUITE_HASH


def suite_version() -> str:
    """Released version of the active suite, or ``<version>+dev`` for modified task content."""
    _initialize_registry()
    return SUITE_VERSION


def public_category(task_id: str) -> str:
    """Public category assigned to an active task by the suite manifest."""
    _initialize_registry()
    return _ACTIVE_SUITE.public_category(task_id)


def execution_kind(task_id: str) -> str:
    """Stable runner-routing kind assigned to an active task by the suite manifest."""
    _initialize_registry()
    return _ACTIVE_SUITE.execution_kind(task_id)


def get(task_id: str) -> Task:
    _initialize_registry()
    if task_id not in TASKS:
        raise ValueError(f"Unknown task {task_id!r} in suite {SUITE_ID!r} (have: {sorted(TASKS)})")
    return TASKS[task_id]
