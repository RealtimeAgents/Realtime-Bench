from __future__ import annotations

# Secrets
"""Environment configuration and lazy provider registration."""

import os as os
import re as re
from pathlib import Path as Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECRETS_FILE = PROJECT_ROOT / ".env"
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_project_secrets(path: Path = DEFAULT_SECRETS_FILE) -> bool:
    """Load simple ``KEY=VALUE`` entries without overriding the process environment.

    Direct Python entry points use the same ``.env`` configuration as the launcher.
    Values are treated as data; shell substitutions are not evaluated.
    """
    if not path.is_file():
        return False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not _ENV_KEY.fullmatch(key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return True


# Catalog policy
"""Leaderboard status of catalog cells, independent of user connection YAMLs.

Per-model settings no external source supplies (wire protocol presets, routes, measured context
accounting) are declared by each API's driver class as ``model_defaults``. As in Harbor's LiteLLM
agents, nothing restates what LiteLLM's bundled model map knows: the native output limit comes
from that map (see litellm_output_limit), and reasoning_effort is the connection YAML's value,
passed through for the driver or provider to accept or reject. Pricing is not inferred here;
unreported cost remains unknown.
"""

CATALOG_POLICY = {
    # JEV is a text-only choice model, and its scores are not pooled with the screenshot
    # leaderboard.
    ("jev_decisions", "typesafe/jev-1.13"): {
        "benchmark_eligible": False,
        "comparison_scope": "experimental_system",
    },
    # Candidate enumeration and composition are a system-level policy.
    ("openai_decisions", "openai/gpt-6-luna-decisions"): {
        "benchmark_eligible": False,
        "comparison_scope": "experimental_system",
    },
}


# Model definitions
"""One connection schema for every built-in model and user model."""

import re as re
from copy import deepcopy as deepcopy
from pathlib import Path as Path
from urllib.parse import urlsplit as urlsplit

import yaml

from functools import lru_cache


_LITELLM = "realtime_framework.providers.litellm:"

# The adapter_type of a ModelConfig is one of these keys, each naming its driver class
# ("package.module:Class"), imported when the API is first looked up (loading the catalog looks
# up every API a model YAML names), or an external driver (below).
# The class declares the API's capabilities, deployment and per-model defaults; adding a protocol
# means adding a driver module and one row here.
APIS = {
    "litellm": _LITELLM + "DirectLiteLLMDriver",
    "chat_completions": _LITELLM + "ChatCompletionsDriver",
    "openrouter": _LITELLM + "OpenRouterDriver",
    "anthropic_messages": _LITELLM + "AnthropicMessagesDriver",
    "moonshot": _LITELLM + "MoonshotDriver",
    "openai_realtime": "realtime_framework.providers.openai_realtime:OpenAIRealtimeDriver",
    "gemini_live": "realtime_framework.providers.gemini.live:GeminiLiveDriver",
    # OpenRouter Decisions API (choice questions), wall clock only.
    "jev_decisions": "realtime_framework.providers.jev.decisions:JevDecisionsClient",
    # OpenAI Decisions models through OpenRouter's Decisions API, wall clock only.
    "openai_decisions": "realtime_framework.providers.openai_decisions:OpenAIDecisionsClient",
}
DEFAULT_EFFORTS = ["provider_default", "none", "minimal", "low", "medium", "high", "xhigh", "max"]
# A model YAML may instead name a driver class kept outside the repository, as endpoint.api
# "package.module:Class"; registered names contain no colon.
_EXTERNAL_DRIVER = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*"
)


def is_external_driver(api) -> bool:
    """Whether an API names an external driver class ("package.module:Class")."""
    return isinstance(api, str) and _EXTERNAL_DRIVER.fullmatch(api) is not None


def api_reasoning_efforts(api: str) -> list[str]:
    """Effort labels an API accepts; model-specific support is left to the driver and provider.

    `fixed` records a model whose reasoning is not configurable, so no setting is sent.
    """
    if adapter_class(api).capabilities.accepts_reasoning_effort:
        return [*DEFAULT_EFFORTS, "fixed"]
    return ["fixed"]


@lru_cache(maxsize=1)
def _litellm_model_map() -> dict:
    """LiteLLM's bundled model map, read from the pinned package's own file.

    Locating that file imports litellm (importlib.resources imports the package), and the import
    fetches a newer map over the network unless LITELLM_LOCAL_MODEL_COST_MAP is set. The limits
    read here come from the bundled file either way, so they stay fixed by the locked version.
    """
    import json
    from importlib.resources import files

    try:
        text = files("litellm").joinpath("model_prices_and_context_window_backup.json").read_text()
    except (ModuleNotFoundError, FileNotFoundError):
        return {}
    return json.loads(text)


def litellm_output_limit(api: str, model: str) -> int | None:
    """Return LiteLLM's max_output_tokens for an API's model id, or None when unlisted."""
    provider = adapter_class(api).deployment.litellm_provider
    if provider is None:
        return None
    if not provider:
        provider = model.partition("/")[0]
    model_map = _litellm_model_map()
    for name in (model if model.startswith(provider + "/") else f"{provider}/{model}", model):
        info = model_map.get(name)
        if isinstance(info, dict) and info.get("litellm_provider") == provider:
            limit = info.get("max_output_tokens")
            if type(limit) is int and limit > 0:
                return limit
    return None


def resolve_output_limit(definition: dict, declared: int | None = None) -> None:
    """Set the native output limit: the endpoint's declaration, else LiteLLM's map."""
    if declared is not None:
        definition["native_output_limit"] = declared
        definition["native_output_limit_source"] = "endpoint"
        return
    limit = litellm_output_limit(definition["adapter_type"], definition["model_name"])
    if limit is not None:
        definition["native_output_limit"] = limit
        definition["native_output_limit_source"] = "litellm"


def validate_connection(connection):
    if not isinstance(connection, dict):
        raise ValueError("endpoint must be a mapping")  # noqa: TRY004
    settings = dict(connection)
    allowed = {
        "api", "model", "base_url", "api_key_env", "stream", "include_usage",
        "revision", "upstream_provider", "native_output_limit",
    }
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"Unknown endpoint fields: {sorted(unknown)}; keep API keys in .env")
    if "model" not in settings:
        raise ValueError("endpoint requires model")
    for name, value in settings.items():
        if name in {"stream", "include_usage"}:
            if type(value) is not bool:
                raise ValueError(f"endpoint.{name} must be true or false")
        elif name == "native_output_limit":
            if type(value) is not int or value <= 0:
                raise ValueError("endpoint.native_output_limit must be a positive integer")
        elif not isinstance(value, str) or not value.strip() or any(c in value for c in "\n\r\t\0"):
            raise ValueError(f"endpoint.{name} must be a nonempty single-line string")
    api = settings.get("api", "chat_completions")
    if is_external_driver(api):
        # A framework class is reached by its registered name, which keeps the name's rules.
        module = api.partition(":")[0]
        if module.partition(".")[0] == "realtime_framework":
            names = [name for name, target in APIS.items() if target == api]
            use = f"endpoint.api: {names[0]}" if names else f"one of {list(APIS)}"
            raise ValueError(f"endpoint.api {api!r} is a framework class; use {use}")
    elif ":" in api:
        raise ValueError(
            f'endpoint.api {api!r} is neither a registered API nor a "package.module:Class" target'
        )
    elif api not in APIS:
        raise ValueError(f"endpoint.api must be one of {list(APIS)}")
    if api == "litellm" and not all(settings["model"].partition("/")[::2]):
        raise ValueError("LiteLLM models use provider/model-id names")
    if api == "chat_completions" and "base_url" not in settings:
        raise ValueError("endpoint.base_url is required for chat_completions")
    if "base_url" in settings:
        url = urlsplit(settings["base_url"])
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "endpoint.base_url must be an HTTP(S) API prefix without credentials"
            )
        settings["base_url"] = settings["base_url"].rstrip("/")
    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", settings.get("api_key_env", "OPENAI_API_KEY")
    ):
        raise ValueError("endpoint.api_key_env must name an environment variable")
    if api != "chat_completions" and {"stream", "include_usage"} & settings.keys():
        raise ValueError("stream/include_usage are transport options for chat_completions only")
    if api in {"openrouter", "openai_decisions"} and not settings.get("upstream_provider"):
        raise ValueError(f"{api} requires upstream_provider to keep the route fixed")
    if api not in {"openrouter", "openai_decisions"} and "upstream_provider" in settings:
        raise ValueError("upstream_provider is only valid for OpenRouter APIs")
    return settings


def bridge_definition(key):
    """Placeholder entry for the launcher's inline-endpoint keys (remote-vlm, litellm-vlm)."""
    api = "litellm" if key == "litellm-vlm" else "chat_completions"
    deployment = adapter_class(api).deployment
    return {
        "key": key,
        "family": deployment.family,
        "provider": deployment.provider,
        "adapter_type": api,
        "model_name": key,
        "key_env": deployment.key_env,
        "reasoning_efforts": api_reasoning_efforts(api),
        "defaults": {
            "temperature_policy": "provider_default",
            "reasoning_effort": "provider_default",
        },
    }


def load_definitions(directory):
    result = []
    seen = {"remote-vlm", "litellm-vlm"}
    for filename, value in [
        (p.name, yaml.safe_load(p.read_text())) for p in sorted(Path(directory).glob("*.yaml"))
    ]:
        if not isinstance(value, dict):
            raise ValueError(f"{filename}: model definition must be a mapping")  # noqa: TRY004
        if set(value) - {"endpoint", "reasoning_effort"} or "endpoint" not in value:
            raise ValueError(f"{filename}: use an endpoint block, plus optional reasoning_effort")
        key = Path(filename).stem
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key) or key in seen:
            raise ValueError(f"{filename}: invalid, duplicate or reserved model name {key!r}")
        seen.add(key)
        settings = validate_connection(value["endpoint"])
        definition = bridge_definition("remote-vlm")
        definition.update(key=key)
        api = settings.get("api", "chat_completions")
        try:
            driver = adapter_class(api)
        except ValueError as exc:  # an external driver that cannot be imported or is incomplete
            raise ValueError(f"{filename}: endpoint.api {exc}") from exc
        model_defaults = driver.model_defaults.get(settings["model"])
        profile = {} if model_defaults is None else {"defaults": deepcopy(model_defaults)}
        profile.update(CATALOG_POLICY.get((api, settings["model"]), {}))
        definition.update(
            adapter_type=api,
            family=driver.deployment.family,
            provider=driver.deployment.provider,
            model_name=settings["model"],
            api_base_url=settings.get("base_url"),
            key_env=settings.get("api_key_env", driver.deployment.key_env),
            reasoning_efforts=api_reasoning_efforts(api),
            **profile,
        )
        resolve_output_limit(definition, settings.get("native_output_limit"))
        definition.setdefault("defaults", {})
        definition["defaults"].setdefault("temperature_policy", "provider_default")
        definition["defaults"]["model_revision"] = settings.get("revision")
        if api == "chat_completions":
            definition["defaults"].update(
                chat_stream=settings.get("stream", True),
                chat_include_usage=settings.get("include_usage", False),
            )
        if api in {"openrouter", "openai_decisions"}:
            definition["defaults"]["upstream_provider"] = settings["upstream_provider"]
        effort = value.get(
            "reasoning_effort",
            "provider_default" if driver.capabilities.accepts_reasoning_effort else "fixed",
        )
        if not isinstance(effort, str) or effort not in definition["reasoning_efforts"]:
            raise ValueError(
                f"{filename}: reasoning_effort must be one of {definition['reasoning_efforts']}"
            )
        definition["defaults"]["reasoning_effort"] = effort
        result.append((filename, definition))
    return result


# Factory
"""Lazy driver resolution: the one lookup of an API's driver class, registered or external.

Only the selected implementation is imported, so optional SDKs never affect other APIs.
"""

from importlib import import_module as import_module

from realtime_framework.entity.model import Deployment, ModelClient, RealtimeCapabilities

# RealtimeDriver's methods, plus bind_trace, which the harness calls on every driver it builds.
_DRIVER_METHODS = (
    "connect",
    "send_observation",
    "start_generation",
    "events",
    "interrupt",
    "sync_history",
    "close",
    "bind_trace",
)
_DRIVER_DECLARATIONS = (
    ("capabilities", RealtimeCapabilities),
    ("deployment", Deployment),
    ("model_defaults", dict),
)
# Capability flags whose method the harness or runtime calls. The runtime calls
# request_context_status before each request of an API that declares request_size_estimation.
_DECLARED_METHODS = {
    "classifies_transport_errors": "classify_transport_error",
    "connection_renewal": "connection_renewal_reason",
    "request_size_estimation": "request_context_status",
    "maps_output_cap": "transport_output_token_cap",
}


def adapter_class(adapter_type: str) -> type[ModelClient]:
    """Resolve one driver without importing unrelated provider stacks."""
    if adapter_type in APIS:
        return _driver_class(APIS[adapter_type])
    if is_external_driver(adapter_type):
        return _external_class(adapter_type)
    raise ValueError(f"Unknown adapter type: {adapter_type!r}")


def _driver_class(target: str) -> type:
    module_name, class_name = target.split(":", 1)
    implementation = getattr(import_module(module_name), class_name)
    if not isinstance(implementation, type):
        raise TypeError(f"Registered driver {target!r} is not a class")
    return implementation


def _external_class(target: str) -> type:
    """Import an external driver from the current environment and check its declarations."""
    module_name, _, class_name = target.partition(":")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"{target!r}: cannot import module {module_name!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if not hasattr(module, class_name):
        raise ValueError(f"{target!r}: module {module_name!r} has no attribute {class_name!r}")
    driver = getattr(module, class_name)
    if not isinstance(driver, type):
        raise ValueError(f"{target!r} is not a class")  # noqa: TRY004
    _check_driver(target, driver)
    return driver


def _check_driver(target: str, driver: type) -> None:
    """Check that a driver class has what the catalog, the pre-run checks, the harness and the
    runtime read from it; registered drivers meet the same contract."""
    missing = [name for name in _DRIVER_METHODS if not callable(getattr(driver, name, None))]
    if missing:
        raise ValueError(f"{target!r} is not a model encoder: missing {', '.join(missing)}")
    for name, kind in _DRIVER_DECLARATIONS:
        if not isinstance(getattr(driver, name, None), kind):
            declaration = f"a {kind.__name__} class attribute"
            raise ValueError(f"{target!r} declares no {name} ({declaration})")  # noqa: TRY004
    for flag, method in _DECLARED_METHODS.items():
        if getattr(driver.capabilities, flag) and not callable(getattr(driver, method, None)):
            raise ValueError(f"{target!r} declares {flag} but has no {method}()")


# Catalog
"""Load model connection YAMLs and resolve API facts from the driver classes.

Every file in configs/models uses the same endpoint schema. Its filename is the
catalog identifier. Per-model wire presets are the drivers' model_defaults; users configure
connections and reasoning effort. The loader imports the driver module of each API it reads,
an external driver's included, with what those modules import (the OpenAI SDK, websockets and
Pillow among them), and reading LiteLLM's model map imports litellm (_litellm_model_map). The
Gemini SDK is imported only when its driver is constructed.
"""

import difflib as difflib
import sys as sys
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from typing import Optional as Optional
from realtime_framework.entity.config import ModelConfig


# All maintained runs omit temperature and use the provider default.
TEMPERATURE_POLICIES = frozenset({"provider_default"})
# The connection YAMLs, the user's own (external drivers included) beside the built-in presets.
MODELS_DIR = PROJECT_ROOT / "configs" / "models"


from realtime_framework.entity.model import ModelSpec as ModelEntry


# --- the catalog ------------------------------------------------------------
def _load_entries() -> list[ModelEntry]:
    """Load the shared connection schema; filenames are model identifiers."""
    import sys as sys
    from pathlib import Path as Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    out: list[ModelEntry] = []
    definitions = load_definitions(MODELS_DIR)
    definitions += [
        (key, bridge_definition(key)) for key in ("remote-vlm", "litellm-vlm")
    ]
    for _fname, d in definitions:
        d = dict(d)
        defaults = dict(d.get("defaults") or {})
        if "max_output_tokens" in defaults:
            raise ValueError(
                f"{_fname}: max_output_tokens was removed; benchmark output is governed by B"
            )
        if "native_output_limit" in defaults:
            raise ValueError(f"{_fname}: native_output_limit must be top-level capability metadata")
        if "temperature" in defaults:
            raise ValueError(
                f"{_fname}: temperature must be omitted; all models use the provider default"
            )
        reasoning_efforts = d.get("reasoning_efforts")
        if (
            not isinstance(reasoning_efforts, list)
            or not reasoning_efforts
            or any(not isinstance(value, str) or not value for value in reasoning_efforts)
            or len(set(reasoning_efforts)) != len(reasoning_efforts)
        ):
            raise ValueError(
                f"{_fname}: reasoning_efforts must be a non-empty list of unique strings"
            )
        default_effort = defaults.get("reasoning_effort")
        if default_effort not in reasoning_efforts:
            raise ValueError(
                f"{_fname}: defaults.reasoning_effort={default_effort!r} is not in "
                f"reasoning_efforts={reasoning_efforts!r}"
            )
        d["reasoning_efforts"] = tuple(reasoning_efforts)
        temperature_policy = defaults.get("temperature_policy")
        if temperature_policy not in TEMPERATURE_POLICIES:
            raise ValueError(
                f"{_fname}: defaults.temperature_policy must be one of "
                f"{sorted(TEMPERATURE_POLICIES)}"
            )
        defaults["temperature"] = None
        upstream_provider = defaults.get("upstream_provider")
        if d.get("provider") == "openrouter":
            if not isinstance(upstream_provider, str) or not upstream_provider.strip():
                raise ValueError(
                    f"{_fname}: OpenRouter entries must pin defaults.upstream_provider"
                )
            defaults["upstream_provider"] = upstream_provider.strip()
        elif upstream_provider is not None:
            raise ValueError(
                f"{_fname}: defaults.upstream_provider is only valid for OpenRouter entries"
            )
        d["defaults"] = defaults
        d["python_env"] = os.environ.get("REALTIMEBENCH_PYTHON") or sys.executable
        # Bridge keys are placeholders; the trial launcher supplies the actual endpoint.
        if d["key"] == "remote-vlm":
            d["key_env"] = os.environ.get("STREAM_CHAT_API_KEY_ENV", d["key_env"])
            d["model_name"] = os.environ.get("BENCHMARK_MODEL") or d["model_name"]
            d["api_base_url"] = os.environ.get("BENCHMARK_BASE_URL") or None
        if d["key"] == "litellm-vlm":
            d["key_env"] = os.environ.get("BENCHMARK_LITELLM_KEY_ENV") or None
            d["model_name"] = os.environ.get("BENCHMARK_LITELLM_MODEL") or d["model_name"]
            d["api_base_url"] = os.environ.get("BENCHMARK_LITELLM_API_BASE") or None
        if d["key"] in {"remote-vlm", "litellm-vlm"}:
            resolve_output_limit(d)
        out.append(ModelEntry(**d))
    return out


from collections.abc import Mapping
from functools import lru_cache


@lru_cache(maxsize=1)
def _catalog_entries():
    # Resolve endpoint overrides and secrets before the first model lookup.
    return {entry.key: entry for entry in _load_entries()}


class _LazyCatalog(Mapping):
    def __getitem__(self, key):
        return _catalog_entries()[key]

    def __iter__(self):
        return iter(_catalog_entries())

    def __len__(self):
        return len(_catalog_entries())


CATALOG = _LazyCatalog()


# --- helper API the rest of the codebase consumes ---------------------------
def get(key: str) -> ModelEntry:
    """Look up a catalog entry by key. Raises KeyError with a 'did you mean' hint."""
    if key in CATALOG:
        return CATALOG[key]
    hint = difflib.get_close_matches(key, CATALOG.keys(), n=3)
    suffix = f" — did you mean {hint}?" if hint else f" — known keys: {sorted(CATALOG)}"
    raise KeyError(f"unknown model key {key!r}{suffix}")


def has(key: str) -> bool:
    return key in CATALOG


def keys(family: Optional[str] = None, provider: Optional[str] = None) -> list[str]:
    """List catalog keys, optionally filtered by family and/or provider (insertion order)."""
    out = []
    for e in CATALOG.values():
        if e.key in {"remote-vlm", "litellm-vlm"}:
            continue
        if family is not None and e.family != family:
            continue
        if provider is not None and e.provider != provider:
            continue
        out.append(e.key)
    return out


def vendor(key: str) -> str:
    """Return who makes the entry's model: its model id prefix, else its API's vendor."""
    entry = get(key)
    if "/" in entry.model_name:
        return entry.model_name.split("/", 1)[0]
    return adapter_class(entry.adapter_type).deployment.vendor or entry.provider


def python_env(key: str) -> str:
    """Return the Python interpreter configured for model subprocesses."""
    return get(key).python_env


def benchmark_eligible(key: str) -> bool:
    """Whether a catalog cell may run in the maintained leaderboard workflow."""
    return bool(get(key).benchmark_eligible)


def reasoning_efforts(key: str) -> tuple[str, ...]:
    """Return the model-native reasoning effort values accepted by a catalog entry."""
    return get(key).reasoning_efforts


def validate_reasoning_effort(key: str, value: str) -> str:
    """Return a supported model-native effort value or raise a model-specific error."""
    supported = reasoning_efforts(key)
    if value not in supported:
        raise ValueError(f"{key} reasoning_effort must be one of {list(supported)}, got {value!r}")
    return value


def to_model_config(key: str, **overrides) -> ModelConfig:
    """Build a ModelConfig for a catalog key, layering: entry.defaults < overrides.

    Runtime overrides such as fps win over entry defaults. Provider-default temperature
    and each entry's upstream provider are
    fixed capability/transport metadata. The native output limit is model capability metadata rather
    than a run budget. adapter_type/model_name/api_base_url come from the entry; api_key stays None
    (each adapter resolves its own key_env at backend setup).
    """
    e = get(key)
    if "temperature" in overrides:
        raise ValueError("temperature must be omitted; all models use the provider default")
    if "temperature_policy" in overrides:
        raise ValueError("temperature_policy is fixed by the catalog entry")
    if "upstream_provider" in overrides:
        raise ValueError("upstream_provider is fixed by the catalog entry")
    if "native_output_limit" in overrides:
        raise ValueError("native_output_limit is endpoint capability metadata, not a run override")
    fields = {**e.defaults, **overrides}
    if key == "litellm-vlm":
        fields["model_revision"] = os.environ.get("BENCHMARK_MODEL_REVISION") or None
    fields["api_key_env"] = e.key_env
    if fields.get("reasoning_effort") == "provider_default":
        fields["reasoning_effort"] = None
    if e.adapter_type == "chat_completions":
        fields["api_key_env"] = e.key_env
        if key == "remote-vlm":
            fields["model_revision"] = os.environ.get("BENCHMARK_MODEL_REVISION") or None
        for field in (
            "context_window_tokens",
            "context_counter",
            "context_image_tokens",
            "context_reserve_tokens",
        ):
            value = os.environ.get("BENCHMARK_" + field.upper())
            if value:
                fields[field] = value
        if fields.get("reasoning_effort") == "provider_default":
            fields["reasoning_effort"] = None
        for field, env_name in (
            ("chat_stream", "STREAM_CHAT_STREAM"),
            ("chat_include_usage", "STREAM_CHAT_INCLUDE_USAGE"),
        ):
            if key == "remote-vlm" and env_name in os.environ:
                if os.environ[env_name] not in {"0", "1"}:
                    raise ValueError(f"{env_name} must be 0 or 1")
                fields[field] = os.environ[env_name] == "1"
    fields.setdefault("native_output_limit", e.native_output_limit)
    fields.setdefault("native_output_limit_source", e.native_output_limit_source)
    return ModelConfig(
        adapter_type=e.adapter_type,
        model_name=e.model_name,
        api_base_url=e.api_base_url,
        **fields,
    )
