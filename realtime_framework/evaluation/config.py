from __future__ import annotations

from realtime_framework.entity.config import DEFAULT_TASK_SEED, task_seed

#  yaml
"""Tiny YAML load helper for benchmark configs, suite manifests and run plans."""

from pathlib import Path as Path

import yaml as yaml


def load_yaml(path) -> dict:
    """Load one YAML file into a dict (empty dict if the file is empty)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Benchmark
"""Load and validate the maintained top-level benchmark configuration."""

import argparse as argparse
import re as re
import sys as sys

from realtime_framework.evaluation.run_spec import PLAN_OPTIONS
from realtime_framework.evaluation.run_spec import run_spec


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "runs" / "default.yaml"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _required_mapping(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")  # noqa: TRY004 - configuration validation error
    return value


def _required_id(value, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"{field} must be a non-empty identifier")
    return value


def _required_id_list(value, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result = [_required_id(item, f"{field} entry") for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicate entries")
    return result


def _connection(item, *, directory: Path) -> dict:
    """Validate deployment settings shared by public and internal benchmark plans."""
    from realtime_framework.providers.registry import validate_connection

    result = dict(item)
    if "endpoint" in item:
        endpoint = validate_connection(item["endpoint"])
        result["endpoint"] = endpoint
        key = "litellm-vlm" if endpoint.get("api") == "litellm" else "remote-vlm"
        if item["key"] != key:
            raise ValueError(f"endpoint settings require model key {key!r}")
        if endpoint.get("api", "chat_completions") not in {"chat_completions", "litellm"}:
            raise ValueError(
                "Put provider API settings in configs/models/<name>.yaml and select that name"
            )
    return result


def _required_model_list(value, model_catalog: dict[str, dict], *, directory: Path) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ValueError("models must be a non-empty list")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            item = {"key": item}
        item = _required_mapping(item, f"models[{index}]")
        unknown_fields = set(item) - {"key", "reasoning_effort", "endpoint"}
        if unknown_fields:
            raise ValueError(f"models[{index}] has unknown fields: {sorted(unknown_fields)}")
        inferred = "remote-vlm" if "endpoint" in item else None
        if isinstance(item.get("endpoint"), dict) and item["endpoint"].get("api") == "litellm":
            inferred = "litellm-vlm"
        key = _required_id(item.get("key", inferred), f"models[{index}].key")
        if key not in model_catalog:
            raise ValueError(f"models[{index}].key is not present in configs/models: {key!r}")
        effort = _required_id(
            item.get("reasoning_effort", model_catalog[key]["defaults"]["reasoning_effort"]),
            f"models[{index}].reasoning_effort",
        )
        supported = model_catalog[key].get("reasoning_efforts")
        if not isinstance(supported, list) or effort not in supported:
            raise ValueError(
                f"models[{index}].reasoning_effort for {key!r} must be one of {supported!r}"
            )
        result.append(
            _connection({**item, "key": key, "reasoning_effort": effort}, directory=directory)
        )
    keys = [item["key"] for item in result]
    if len(set(keys)) != len(keys):
        raise ValueError("models contains duplicate keys")
    return result


def _required_tokens_per_frame(value, field: str) -> int | str:
    if value != "unlimited" and (type(value) is not int or value <= 0):
        raise ValueError(f"{field} must be a positive integer or 'unlimited'")
    return value


def _required_reasoning_effort(
    value,
    field: str,
    *,
    key: str,
    model_catalog: dict[str, dict],
) -> str:
    effort = _required_id(value, field)
    supported = model_catalog[key].get("reasoning_efforts")
    if not isinstance(supported, list) or effort not in supported:
        raise ValueError(f"{field} for {key!r} must be one of {supported!r}")
    return effort


def _required_cell_list(value, model_catalog: dict[str, dict], *, wall_clock=False) -> list[dict]:
    """Validate named runtime cells, allowing one model to appear more than once."""
    if not isinstance(value, list) or not value:
        raise ValueError("cells must be a non-empty list")
    result = []
    for index, item in enumerate(value):
        field = f"cells[{index}]"
        item = _required_mapping(item, field)
        unknown_fields = set(item) - {
            "id",
            "model",
            "reasoning_effort",
            "tokens_per_frame",
            "device",
        }
        if unknown_fields:
            raise ValueError(f"{field} has unknown fields: {sorted(unknown_fields)}")
        cell_id = _required_id(item.get("id"), f"{field}.id")
        model = _required_id(item.get("model"), f"{field}.model")
        if model not in model_catalog:
            raise ValueError(f"{field}.model is not present in configs/models: {model!r}")
        effort = _required_reasoning_effort(
            item.get("reasoning_effort"),
            f"{field}.reasoning_effort",
            key=model,
            model_catalog=model_catalog,
        )
        tokens_per_frame = (
            None
            if wall_clock
            else _required_tokens_per_frame(
                item.get("tokens_per_frame"),
                f"{field}.tokens_per_frame",
            )
        )
        device = item.get("device")
        if device is not None and (type(device) is not int or device < 0):
            raise ValueError(f"{field}.device must be a non-negative integer")
        result.append(
            {
                "id": cell_id,
                "model": model,
                "reasoning_effort": effort,
                "tokens_per_frame": tokens_per_frame,
                "device": device,
            }
        )
    cell_ids = [item["id"] for item in result]
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cells contains duplicate ids")
    return result


def load_benchmark(path: str | Path = DEFAULT_CONFIG) -> dict:
    """A run plan, as ``rtbench run -c`` hands it to Harbor (harbor_cli.py plan_job): a field the
    job does not carry is refused, not dropped."""
    path = Path(path).expanduser().resolve()
    raw = _required_mapping(load_yaml(path), str(path))
    unknown = set(raw) - {"suite", "runtime", "models", "cells", "tasks"}
    if unknown:
        raise ValueError(f"Unknown benchmark fields: {sorted(unknown)}")
    runtime = _required_mapping(raw.get("runtime", {}), "runtime")
    unknown = set(runtime) - set(PLAN_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown runtime fields: {sorted(unknown)}")
    try:
        spec = run_spec(
            "harbor", **{name: value for name, value in runtime.items() if name != "protocol"}
        )
    except ValueError as exc:
        raise ValueError(f"runtime.{exc}") from None
    # The trial runs the protocol its clock implies (evaluation/_trial.py), and a plan on the
    # token clock names its B.
    if runtime.get("protocol", spec.protocol) != spec.protocol:
        raise ValueError(
            f"runtime.protocol must be {spec.protocol!r} on the {spec.clock_mode} clock"
        )
    if spec.clock_mode == "token" and runtime.get("tokens_per_frame") is None:
        raise ValueError("runtime.tokens_per_frame must be a positive integer or 'unlimited'")

    suite_id = raw.get("suite", "public")
    if not isinstance(suite_id, str):
        raise ValueError("suite must be NAME or NAME@MAJOR.MINOR")
    from realtime_framework.providers.registry import MODELS_DIR
    from realtime_framework.providers.registry import bridge_definition
    from realtime_framework.providers.registry import load_definitions

    model_files = load_definitions(MODELS_DIR)
    model_files += [
        (key, bridge_definition(key)) for key in ("remote-vlm", "litellm-vlm")
    ]
    model_catalog = {entry.get("key"): entry for _, entry in model_files}
    has_models = raw.get("models") is not None
    has_cells = raw.get("cells") is not None
    if has_models == has_cells:
        raise ValueError("benchmark config must define exactly one of models or cells")
    models = (
        _required_model_list(raw.get("models"), model_catalog, directory=path.parent)
        if has_models
        else []
    )
    cells = (
        _required_cell_list(
            raw.get("cells"),
            model_catalog,
            wall_clock=spec.clock_mode == "wall",
        )
        if has_cells
        else []
    )

    from realtime_tasks.registry import suite_path

    try:
        suite_path = suite_path(ROOT, suite_id)
    except FileNotFoundError as exc:
        raise ValueError(f"unknown suite {suite_id!r}: {exc}") from exc
    suite = _required_mapping(load_yaml(suite_path), str(suite_path))
    suite_tasks = [
        task_id
        for category in suite.get("categories", [])
        for task_id in category.get("task_ids", [])
    ]
    tasks = _required_id_list(raw.get("tasks", suite_tasks), "tasks")
    unknown_tasks = [task for task in tasks if task not in set(suite_tasks)]
    if unknown_tasks:
        raise ValueError(f"tasks not present in suite {suite_id!r}: {unknown_tasks}")

    from realtime_framework.providers.preflight import validate_model_interaction

    for item in cells or models:
        model = item.get("model", item.get("key"))
        if "endpoint" in item and spec.clock_mode != "wall":
            raise ValueError("endpoint settings require the wall clock")
        validate_model_interaction(
            model_catalog[model]["adapter_type"],
            clock_mode=spec.clock_mode,
            fps=spec.fps,
            model_input_fps=spec.model_input_fps,
            interaction_mode=spec.interaction_mode,
            tokens_per_frame=item.get("tokens_per_frame", spec.tokens_per_frame),
        )

    return {
        "suite": suite_id,
        "models": models,
        "cells": cells,
        "tasks": tasks,
        "runtime": {
            "mode": spec.mode,
            "clock_mode": spec.clock_mode,
            "fps": spec.fps,
            # Unset, it stays None for the Harbor agent's options to default to fps, after any
            # --ak fps.
            "model_input_fps": runtime.get("model_input_fps"),
            "interaction_mode": spec.interaction_mode,
            "tokens_per_frame": (
                None
                if spec.clock_mode == "wall"
                else "unlimited"
                if spec.tokens_per_frame is None
                else spec.tokens_per_frame
            ),
            "duration_minutes": spec.duration_minutes,
            "seed": spec.seed,
            "protocol": spec.protocol,
        },
    }


def benchmark_main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    import json

    print(json.dumps(load_benchmark(args.config), indent=2))
    return 0


# Identity
"""Deterministic identities for the executable model harness and one effective run policy."""

import hashlib as hashlib
import json as json
import os as os
from pathlib import Path as Path

from realtime_framework.entity.config import DEFAULT_FRAME_MAX_DIM
from realtime_framework.entity.config import FrameEncodingConfig
from realtime_framework.entity.config import RunnerTimingConfig
from realtime_framework.entity.config import merge_streaming_settings
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.evaluation.run_spec import DEFAULT_MODE
from realtime_framework.evaluation.run_spec import INTERACTION_MODES
from realtime_framework.evaluation.run_spec import PROTOCOLS
from realtime_framework.evaluation.run_spec import RunSpec
from realtime_framework.providers.preflight import check_streaming_options
from realtime_framework.providers.preflight import validate_model_interaction
from realtime_framework.entity.actions import SOM_REPRESENTATION_VERSION
from realtime_framework.entity.records import EPISODE_TRACE_SCHEMA

identity_ROOT = Path(__file__).resolve().parents[2]
INTERACTABLES_PROTOCOL = "environment_injected_each_observation"
INTERACTABLES_SCHEMA = "inspect_interactables.v1"
INTERACTABLES_REPRESENTATION = SOM_REPRESENTATION_VERSION


OUTPUT_CLOCK_POLICY = "finite_provider_native_usage_horizon_capped_or_unlimited_act_then_frame_v16"
GUI_ACTION_PROTOCOL = "single_unambiguous_gui_action_transport_envelopes_quoted_shorthand_v7"

# Non-secret environment settings that change the effective model transport, visual input, or
# runtime policy: those the frame encoding and the runner's timing read at import, and the rest.
# Secrets are deliberately excluded, but endpoint selection is identity-bearing: two services
# advertising the same model id are not necessarily the same backend.
_RUNTIME_ENV_KEYS = frozenset(
    {
        *(variable for variable, _ in FrameEncodingConfig.ENVIRONMENT.values()),
        *(variable for variable, _ in RunnerTimingConfig.ENVIRONMENT.values()),
        "SANDBOX_IMAGE",
        "RT_SUITE_IMAGE",
        "SANDBOX_GRAPHICS_MODE",
        "SCREEN_RECORDING_ENCODER",
        "OPENAI_BASE_URL",
        "OPENROUTER_BASE_URL",
        "MOONSHOT_BASE_URL",
        "STREAM_ACTIVE_VISUAL_BUDGET",
        "STREAM_DETAILED_HISTORY_DROP_BLOCK_SIZE",
        "STREAM_DETAILED_HISTORY_TURNS",
        "STREAM_MIN_REQUEST_INTERVAL",
        "STREAM_VISUAL_FOLD_BLOCK_SIZE",
    }
)

# The framework package, top-level modules and all: its sources can change what a model sees, what
# it may emit, how actions execute, or how a run is recorded. Task/scorer content is covered by the
# suite fingerprint, and each run's model and runtime settings by runtime_config_hash, so neither
# configs/ nor task packages belong here.
_HARNESS_PACKAGE = "realtime_framework"
_SOURCE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".json", ".sh", ".toml"})
# The released harness version and its locked hash live in pyproject.toml [tool.rtbench.harness],
# outside the fingerprinted sources, like each suite manifest's version/suite_hash lock.
HARNESS_UNRELEASED = "+dev"


def _source_paths() -> list[Path]:
    return sorted(
        path
        for path in (identity_ROOT / _HARNESS_PACKAGE).rglob("*")
        if path.is_file() and path.suffix in _SOURCE_SUFFIXES and "__pycache__" not in path.parts
    )


def _normalized_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    try:
        return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    except UnicodeDecodeError:
        return data


def runtime_dependencies() -> list[dict]:
    """Locked packages reachable from the project's runtime dependencies (no dev group).

    Only resolved name/version (and any non-registry source) are identity: lockfile layout,
    artifact URLs, dev tooling and the project's own version do not change model I/O.
    """
    import tomllib

    path = identity_ROOT / "uv.lock"
    if not path.is_file():
        raise FileNotFoundError("Harness fingerprint requires uv.lock")
    packages = tomllib.loads(path.read_text())["package"]
    by_name: dict[str, list[dict]] = {}
    for package in packages:
        by_name.setdefault(package["name"], []).append(package)
    projects = [p for p in packages if p.get("source", {}).get("editable") == "."]
    if len(projects) != 1:
        raise ValueError("uv.lock must contain exactly one editable project package")

    def resolve(dependency):
        candidates = by_name.get(dependency["name"], [])
        if "version" in dependency:
            candidates = [c for c in candidates if c["version"] == dependency["version"]]
        if len(candidates) != 1:
            raise ValueError(f"uv.lock cannot resolve dependency {dependency!r}")
        return candidates[0]

    seen: dict[tuple[str, str], dict] = {}
    pending = [(d, tuple(d.get("extra", ()))) for d in projects[0].get("dependencies", [])]
    expanded: set[tuple[str, str, str | None]] = set()
    while pending:
        dependency, extras = pending.pop()
        package = resolve(dependency)
        key = (package["name"], package["version"])
        for extra in (None, *extras):
            if (*key, extra) in expanded:
                continue
            expanded.add((*key, extra))
            seen[key] = package
            edges = (
                package.get("dependencies", [])
                if extra is None
                else package.get("optional-dependencies", {}).get(extra, [])
            )
            pending.extend((d, tuple(d.get("extra", ()))) for d in edges)
    return [
        {
            "name": package["name"],
            "version": package["version"],
            **(
                {}
                if "registry" in package.get("source", {})
                else {"source": package.get("source")}
            ),
        }
        for _, package in sorted(seen.items())
    ]


def _entries_sha256(entries: list[tuple[bytes, bytes]]) -> str:
    """SHA-256 over (name, content) pairs, each part prefixed with its length."""
    digest = hashlib.sha256()
    for name, content in entries:
        for part in (name, content):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def harness_fingerprint() -> str:
    """Hash framework sources and locked runtime dependencies that can affect model I/O or actions."""
    entries = [
        (path.relative_to(identity_ROOT).as_posix().encode(), _normalized_bytes(path))
        for path in _source_paths()
    ]
    entries.append(
        (
            b"<runtime-dependencies>",
            json.dumps(runtime_dependencies(), sort_keys=True, separators=(",", ":")).encode(),
        )
    )
    return _entries_sha256(entries)


def external_driver_identity(target: str) -> dict:
    """The code an external driver ("package.module:Class") runs, which harness_fingerprint does
    not cover: the module and class the model YAML names, and a digest of every file of the
    module's top-level package but its __pycache__ (or of the module alone, outside a package):
    prompts, templates and compiled modules are driver code too. Paths are relative to the
    directory holding the package, so its install location is not part of the digest. The record
    also names the context-policy row the driver gets (EXTERNAL_DRIVER_ROW)."""
    from importlib import import_module

    module_name, _, class_name = target.partition(":")
    top = import_module(module_name.partition(".")[0])
    entries = []
    for location in map(Path, getattr(top, "__path__", None) or [top.__file__]):
        entries += [
            (path.relative_to(location.parent).as_posix().encode(), _normalized_bytes(path))
            for path in (location.rglob("*") if location.is_dir() else [location])
            if path.is_file() and "__pycache__" not in path.parts
        ]
    return {
        "module": module_name,
        "class": class_name,
        "source_sha256": _entries_sha256(sorted(entries)),
        "context_policy_row": current_context_policy().for_path(target, {}, None).api,
    }


def configuration_sha256(launch: dict) -> str:
    """Hash of the model configuration a Harbor trial resolved (the trial worker's settings).

    Harbor's verify and aggregate and the upload compare it between resolution and the run and
    across runs; an external driver's code identity and a streaming run's settings are part of it.
    """
    configuration = {
        "model_config": launch["model_config"],
        "model_revision": launch.get("model_revision"),
    }
    for part in ("external_driver", "streaming"):
        if part in launch:
            configuration[part] = launch[part]
    return hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def harness_lock() -> tuple[str, str]:
    """Return the released (version, hash) locked in pyproject.toml [tool.rtbench.harness]."""
    import tomllib

    lock = (
        tomllib.loads((identity_ROOT / "pyproject.toml").read_text())
        .get("tool", {})
        .get("rtbench", {})
        .get("harness", {})
    )
    version, locked = lock.get("version"), lock.get("hash")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+", version):
        raise ValueError("pyproject.toml [tool.rtbench.harness] version must be a MAJOR.MINOR string")
    if not isinstance(locked, str) or not re.fullmatch(r"[0-9a-f]{64}", locked):
        raise ValueError("pyproject.toml [tool.rtbench.harness] hash must be a SHA-256 hex digest")
    return version, locked


def harness_version(harness_hash: str | None = None) -> str:
    """Released harness version, or ``<version>+dev`` when the hash differs from the lock."""
    version, locked = harness_lock()
    current = harness_hash or harness_fingerprint()
    return version if current == locked else version + HARNESS_UNRELEASED


def effective_run_config(
    *,
    model_config,
    fps: float,
    max_minutes: float,
    protocol: str,
    think_budget_s,
    per_move_cap_s,
    mode: str | None = None,
    clock_mode: str = "wall",
    tokens_per_frame: int | None = None,
    seed: int = DEFAULT_TASK_SEED,
    execution_feedback: bool | None = None,
    interaction_mode: str = "turn_based",
) -> dict:
    """Capture non-secret effective settings that materially change a run."""
    streaming = getattr(model_config, "streaming", {})  # never in the dump (ModelConfig.streaming)
    if hasattr(model_config, "model_dump"):
        model_config = model_config.model_dump()
    if clock_mode == "wall":
        tokens_per_frame = None
    validate_model_interaction(
        model_config.get("adapter_type"),
        clock_mode=clock_mode,
        fps=fps,
        model_input_fps=model_config["fps"],
        interaction_mode=interaction_mode,
        tokens_per_frame=tokens_per_frame,
    )
    if protocol == "time-aware-token-v1":
        if clock_mode != "token":
            raise ValueError("time-aware-token-v1 requires a token clock")
        if think_budget_s is not None or per_move_cap_s is not None:
            raise ValueError("time-aware-token-v1 does not use separate deliberation caps")
    check_streaming_options(
        interaction_mode, think_budget_s=think_budget_s, per_move_cap_s=per_move_cap_s
    )
    model_config = dict(model_config)
    model_config.pop("api_key", None)
    environment = {
        key: value for key, value in sorted(os.environ.items()) if key in _RUNTIME_ENV_KEYS
    }
    from realtime_framework.sandbox.apptainer import runtime_identity

    adapter_type = model_config.get("adapter_type")
    context_policy = current_context_policy().for_path(
        adapter_type, model_config["provider_options"], model_config["context_preset"]
    )
    visual_budget = context_policy.visuals.keep
    visual_fold_block_size = context_policy.visuals.block
    detailed_turns = context_policy.turns.keep
    detailed_drop_block_size = context_policy.turns.block
    config = {
        "version": 47,
        "seed": task_seed(seed),
        "history_som_policy": "retained",
        "history_som_transport": "provider_retained_v1",
        "history_window_policy": "independent_visual_20_10_text_50_10_plus_ledger_v4",
        "history_transport_compaction_policy": (
            "openai_in_place_else_native_tool_aware_block_replay_v2"
        ),
        "stateless_execution_feedback_policy": "chronological_turn_feedback_v1",
        "token_clock_output_accounting": (
            "usage_audit_only_not_charged_v1"
            if clock_mode == "wall" or tokens_per_frame is None
            else "provider_or_native_response_capped_at_remaining_episode_capacity_v2"
        ),
        "realtime_business_action_barrier": "browser_input_to_server_and_render_completion_v2",
        "termination_snapshot_policy": "freeze_before_input_release_v2",
        "browser_raf_clock_policy": "persistent_60hz_grid_with_zero_time_action_commit_v3",
        "active_visual_budget": visual_budget,
        "visual_fold_block_size": visual_fold_block_size,
        "minimum_active_visuals_after_fold": visual_budget - visual_fold_block_size + 1,
        "detailed_history_turns": detailed_turns,
        "detailed_history_drop_block_size": detailed_drop_block_size,
        "minimum_detailed_turns_after_drop": detailed_turns - detailed_drop_block_size + 1,
        "archived_history_policy": "actual_action_jsonl_v1",
        # Every context-policy item and its version, as this model API applies them.
        "context_policy": context_policy.to_record(),
        # entity.model.TRANSPORT_RETRY, identical for every model API.
        "transport_retry_policy": "harness_shared_3_retries_1_3_8s_jitter25_retry_after_30s_v2",
        "frame_max_dimension": int(os.environ.get("FRAME_MAX_DIM", str(DEFAULT_FRAME_MAX_DIM))),
        "container_runtime": runtime_identity(),
        "model_config": model_config,
        "fps": float(fps),
        "max_minutes": float(max_minutes),
        "protocol": protocol,
        "think_budget_s": think_budget_s,
        "per_move_cap_s": per_move_cap_s,
        "clock_mode": clock_mode,
        "interaction_mode": interaction_mode,
        "tokens_per_frame": tokens_per_frame,
        "token_clock_timing_policy": (
            "fixed_action_at_observation_then_one_frame_per_decision"
            if clock_mode == "token" and tokens_per_frame is None
            else "output_tokens_divided_by_tokens_per_frame"
            if clock_mode == "token"
            else None
        ),
        "native_output_limit": model_config.get("native_output_limit"),
        "native_output_limit_source": model_config.get("native_output_limit_source"),
        "interactables_protocol": INTERACTABLES_PROTOCOL,
        "interactables_schema": INTERACTABLES_SCHEMA,
        "interactables_representation": INTERACTABLES_REPRESENTATION,
        "episode_trace_schema": EPISODE_TRACE_SCHEMA,
        "gui_action_protocol": GUI_ACTION_PROTOCOL,
        "request_context_policy": "pending_multimodal_request_block_archive_with_ledger_v1",
        "cdp_transport": "persistent_websocket_explicit_probe_preserves_default_target_v2",
        "environment_overrides": environment,
    }
    from realtime_framework.providers.litellm import LiteLLMDriver
    from realtime_framework.providers.registry import adapter_class, is_external_driver

    driver = adapter_class(adapter_type)
    # A live session's own items stay with the provider, unseen; recorded, never refused.
    config["server_context"] = driver.capabilities.server_context
    if issubclass(driver, LiteLLMDriver):
        from importlib.metadata import version as version

        streamed = driver.protocol.stream
        if streamed is None:
            streamed = model_config.get("chat_stream", True)
        config["provider_transport"] = (
            "litellm_raw_provider_usage_v1"
            if streamed
            else "litellm_complete_response_raw_usage_v1"
        )
        config["provider_sdk_versions"] = {
            package: version(package) for package in ("litellm", "openai")
        }
        # Requests without a decision cap send the native limit, or omit the parameter.
        config["provider_output_limit"] = {
            "parameter": driver.compat.output_limit_param,
            "uncapped_value": model_config.get("native_output_limit") or "provider_default",
        }
    if is_external_driver(adapter_type):
        config["external_driver"] = external_driver_identity(adapter_type)
    if interaction_mode == "streaming":
        # validate_model_interaction admits only a live API that declares streaming, and a driver
        # that declares it declares its settings (providers/registry.py _check_driver).
        config["streaming"] = merge_streaming_settings(driver, streaming)
    if mode is not None:
        config["mode"] = mode
    if execution_feedback is not None:
        config["execution_feedback"] = bool(execution_feedback)
    if mode == DEFAULT_MODE:
        config["output_clock_policy"] = OUTPUT_CLOCK_POLICY
        config["provider_output_cap_policy"] = "no_benchmark_derived_provider_cap"
    return config


def config_fingerprint(config: dict) -> str:
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def run_spec_config(spec: RunSpec, model_config) -> dict:
    """effective_run_config of an episode with the run options ``spec``."""
    return effective_run_config(
        model_config=model_config,
        fps=spec.fps,
        max_minutes=spec.duration_minutes,
        mode=spec.mode,
        clock_mode=spec.clock_mode,
        protocol=spec.protocol,
        seed=spec.seed,
        tokens_per_frame=spec.tokens_per_frame,
        think_budget_s=spec.think_budget_s,
        per_move_cap_s=spec.per_move_cap_s,
        interaction_mode=spec.interaction_mode,
    )


def runner_run_spec(args) -> RunSpec:
    """The run options of the runner's arguments ``args`` (runner.py _parser), with the runner's
    defaults (run_spec.py ENTRY_DEFAULTS); identity_main takes the same options."""
    return run_spec(
        "runner",
        clock_mode=args.clock,
        tokens_per_frame=args.tokens_per_frame,
        fps=args.fps,
        model_input_fps=args.model_input_fps,
        interaction_mode=args.interaction_mode,
        duration_minutes=args.max_minutes,
        seed=args.seed,
        protocol=args.protocol,
        think_budget_s=args.think_budget_s,
        per_move_cap_s=args.per_move_cap_s,
    )


def apply_run_defaults(args) -> None:
    """Fill in what the runner's arguments ``args`` leave to its defaults besides its run options
    (runner_run_spec), as runner.main() runs and records the run: the model's reasoning effort
    (its catalog default, "low" for a raw backend id) and, in the environment the runtime-config
    hash reads (_RUNTIME_ENV_KEYS), the desktop's graphics mode and recording encoder. The hash
    the run's report expects (reporting.prepare_episode_report) and the one identity_main prints
    are then the one the run records."""
    import realtime_framework.providers.registry as catalog

    if catalog.has(args.model):
        if args.reasoning_effort is None:
            args.reasoning_effort = catalog.get(args.model).defaults["reasoning_effort"]
    else:
        args.reasoning_effort = args.reasoning_effort or "low"
    os.environ["SANDBOX_GRAPHICS_MODE"] = args.graphics_mode
    os.environ["SCREEN_RECORDING_ENCODER"] = args.recording_encoder


def identity_main(argv=None) -> int:
    """Print the runtime-config hash runner.main() records for the same arguments."""
    import argparse as argparse

    parser = argparse.ArgumentParser(description=identity_main.__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--model-input-fps", type=float, help="default: --fps")
    parser.add_argument("--interaction-mode", choices=INTERACTION_MODES)
    parser.add_argument("--max-minutes", type=float, required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--think-budget-s", type=float)
    parser.add_argument("--per-move-cap-s", type=float)
    parser.add_argument("--clock", choices=tuple(PROTOCOLS))
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--tokens-per-frame")
    parser.add_argument(
        "--graphics-mode",
        choices=("cpu", "gpu"),
        default=os.environ.get("SANDBOX_GRAPHICS_MODE", "cpu"),
    )
    parser.add_argument(
        "--recording-encoder",
        choices=("cpu", "gpu"),
        default=os.environ.get("SCREEN_RECORDING_ENCODER", "cpu"),
    )
    args = parser.parse_args(argv)
    apply_run_defaults(args)
    try:
        spec = runner_run_spec(args)
    except ValueError as exc:
        parser.error(str(exc))

    import realtime_framework.providers.registry as catalog
    import realtime_tasks.registry as registry

    registry.get(args.task)  # validate membership; task-local deliberation caps are not inherited
    entry = catalog.get(args.model)
    try:
        catalog.validate_reasoning_effort(args.model, args.reasoning_effort)
    except ValueError as exc:
        parser.error(str(exc))
    from realtime_framework.providers.preflight import UnsupportedTokenBudget, check_token_budget

    try:
        check_token_budget(
            entry.adapter_type, clock_mode=spec.clock_mode, tokens_per_frame=spec.tokens_per_frame
        )
    except UnsupportedTokenBudget as exc:
        parser.error(str(exc))
    config = catalog.to_model_config(
        args.model, fps=spec.model_input_fps, reasoning_effort=args.reasoning_effort
    )
    print(config_fingerprint(run_spec_config(spec, config)))
    return 0


if __name__ == "__main__":
    import sys

    if "--model" in sys.argv and "--task" in sys.argv:
        identity_main()
    else:
        raise SystemExit(benchmark_main())
