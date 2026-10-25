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


# Model profiles
"""Per-model settings no external source supplies, independent of user connection YAMLs.

As in Harbor's LiteLLM agents, nothing here restates what LiteLLM's bundled model map knows: the
native output limit comes from that map (see litellm_output_limit), and reasoning_effort is the
connection YAML's value, passed through for the driver or provider to accept or reject. Entries
only carry wire protocol options, experimental-system status and measured context accounting.
Pricing is not inferred here; unreported cost remains unknown.
"""

_LIVE_TOOL_TRANSCRIPT = {
    "tool_behavior": "non_blocking",
    "completion_signal": "turn_complete",
    "input_transport": "client_content",
    "tool_response_scheduling": "silent",
    "history_format": "tool_transcript",
}

MODEL_PROFILES = {
    ("gemini_live", "gemini-3.1-flash-live-preview"): {
        "defaults": {"provider_options": _LIVE_TOOL_TRANSCRIPT},
    },
    ("gemini_live", "gemini-3.8-live"): {
        "defaults": {"provider_options": _LIVE_TOOL_TRANSCRIPT},
    },
    ("gemini_live", "gemini-3.8-live-extended-thinking"): {
        "defaults": {
            "provider_options": {
                **_LIVE_TOOL_TRANSCRIPT,
                "completion_signal": "interaction_idle",
                "tool_response_scheduling": "omit",
            },
        },
    },
    # JEV is a text-only choice model, and its scores are not pooled with the screenshot
    # leaderboard.
    ("jev_decisions", "typesafe/jev-1.13"): {
        "defaults": {"upstream_provider": "typesafe"},
        "benchmark_eligible": False,
        "comparison_scope": "experimental_system",
    },
    ("openai_decisions", "openai/gpt-6-luna-decisions"): {
        "defaults": {
            # OpenRouter's published window. A 768 px screenshot measured 337 input tokens; the
            # reserve covers one question's criteria (at most 255 candidates) and instructions.
            "context_window_tokens": 1050000,
            "context_image_tokens": 512,
            "context_reserve_tokens": 32768,
        },
        # Candidate enumeration and composition are a system-level policy.
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

from dataclasses import dataclass as dataclass
from functools import lru_cache


@dataclass(frozen=True)
class ApiSpec:
    """One registered model API: its lazy driver target and catalog/runner capabilities."""

    driver: str  # "package.module:Class"; imported only when the API is selected
    family: str
    provider: str
    key_env: str | None  # default credential variable
    finite_frame_budget: bool = True  # reports complete usage for a finite token clock
    wall_clock_only: bool = False
    endpoint: bool = True  # selectable as endpoint.api in a model YAML
    action_channel: str = "text_json"  # static driver capability used for harness selection
    vendor: str | None = None  # model maker, when the API's model ids carry no vendor prefix
    # LiteLLM provider whose bundled model map supplies native output limits; "" means the model
    # id carries its own provider prefix, None means the endpoint must declare any limit.
    litellm_provider: str | None = None
    # False: the driver sends no reasoning setting, so `fixed` is the only truthful effort label.
    reasoning_effort: bool = True


LITELLM_DRIVER = "realtime_framework.providers.litellm:LiteLLMDriver"

# The adapter_type of every ModelConfig is one of these keys. Adding a protocol means adding a
# driver module and one row here; the harness selects behavior from driver capabilities.
APIS = {
    "litellm": ApiSpec(
        LITELLM_DRIVER, "vlm", "litellm", None,
        finite_frame_budget=False, wall_clock_only=True, litellm_provider="",
    ),
    "chat_completions": ApiSpec(
        LITELLM_DRIVER, "vlm", "compatible", "OPENAI_API_KEY",
        finite_frame_budget=False, wall_clock_only=True,
    ),
    "openrouter": ApiSpec(
        LITELLM_DRIVER, "vlm", "openrouter", "OPENROUTER_API_KEY", litellm_provider="openrouter"
    ),
    "anthropic_messages": ApiSpec(
        LITELLM_DRIVER, "vlm", "anthropic", "ANTHROPIC_API_KEY", vendor="anthropic",
        litellm_provider="anthropic",
    ),
    "moonshot": ApiSpec(
        LITELLM_DRIVER, "vlm", "moonshot", "MOONSHOT_API_KEY", vendor="moonshot",
        litellm_provider="moonshot",
    ),
    "openai_realtime": ApiSpec(
        "realtime_framework.providers.openai_realtime:OpenAIRealtimeDriver", "streaming", "openai",
        "OPENAI_API_KEY", vendor="openai", litellm_provider="openai",
    ),
    "gemini_live": ApiSpec(
        "realtime_framework.providers.gemini.live:GeminiLiveDriver", "streaming", "gemini", "GEMINI_API_KEY",
        action_channel="function_call", vendor="google", litellm_provider="gemini",
    ),
    # OpenRouter Decisions API (choice questions), wall clock only.
    "jev_decisions": ApiSpec(
        "realtime_framework.providers.jev.decisions:JevDecisionsClient", "streaming", "openrouter",
        "OPENROUTER_API_KEY",
        finite_frame_budget=False, wall_clock_only=True,
        action_channel="choice", reasoning_effort=False,
    ),
    # OpenAI Decisions models through OpenRouter's Decisions API, wall clock only.
    "openai_decisions": ApiSpec(
        "realtime_framework.providers.openai_decisions:OpenAIDecisionsClient",
        "vlm", "openrouter", "OPENROUTER_API_KEY",
        finite_frame_budget=False, wall_clock_only=True,
        action_channel="choice", reasoning_effort=False,
    ),
}
DEFAULT_EFFORTS = ["provider_default", "none", "minimal", "low", "medium", "high", "xhigh", "max"]


def api_reasoning_efforts(api: str) -> list[str]:
    """Effort labels an API accepts; model-specific support is left to the driver and provider.

    `fixed` records a model whose reasoning is not configurable, so no setting is sent.
    """
    return [*DEFAULT_EFFORTS, "fixed"] if APIS[api].reasoning_effort else ["fixed"]


@lru_cache(maxsize=1)
def _litellm_model_map() -> dict:
    """LiteLLM's bundled model map, read from the pinned package without importing it.

    Importing litellm may fetch a newer map over the network; the bundled file keeps a run's
    limits fixed by the locked LiteLLM version.
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
    provider = APIS[api].litellm_provider
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
    if api not in APIS or not APIS[api].endpoint:
        choices = [name for name, spec in APIS.items() if spec.endpoint]
        raise ValueError(f"endpoint.api must be one of {choices}")
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
    if key == "litellm-vlm":
        return {
            "key": key,
            "family": "vlm",
            "provider": "litellm",
            "adapter_type": "litellm",
            "model_name": key,
            "key_env": None,
            "reasoning_efforts": api_reasoning_efforts("litellm"),
            "defaults": {
                "temperature_policy": "provider_default",
                "reasoning_effort": "provider_default",
            },
        }
    return {
        "key": key,
        "family": "vlm",
        "provider": "compatible",
        "adapter_type": "chat_completions",
        "model_name": key,
        "key_env": "OPENAI_API_KEY",
        "reasoning_efforts": api_reasoning_efforts("chat_completions"),
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
        spec = APIS[api]
        profile = deepcopy(MODEL_PROFILES.get((api, settings["model"]), {}))
        definition.update(
            adapter_type=api,
            family=spec.family,
            provider=spec.provider,
            model_name=settings["model"],
            api_base_url=settings.get("base_url"),
            key_env=settings.get("api_key_env", spec.key_env),
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
                chat_include_usage=settings.get("include_usage", True),
            )
        if api in {"openrouter", "openai_decisions"}:
            definition["defaults"]["upstream_provider"] = settings["upstream_provider"]
        effort = value.get(
            "reasoning_effort", "provider_default" if spec.reasoning_effort else "fixed"
        )
        if not isinstance(effort, str) or effort not in definition["reasoning_efforts"]:
            raise ValueError(
                f"{filename}: reasoning_effort must be one of {definition['reasoning_efforts']}"
            )
        definition["defaults"]["reasoning_effort"] = effort
        result.append((filename, definition))
    return result


# Factory
"""Lazy driver resolution and API-level interaction rules.

Only the selected implementation is imported, so optional SDKs never affect other APIs.
"""

from importlib import import_module as import_module

from realtime_framework.entity.model import ModelClient


def api_spec(adapter_type: str) -> ApiSpec:
    try:
        return APIS[adapter_type]
    except KeyError as exc:
        raise ValueError(f"Unknown adapter type: {adapter_type!r}") from exc


def adapter_class(adapter_type: str) -> type[ModelClient]:
    """Resolve one registered driver without importing unrelated provider stacks."""
    return _driver_class(api_spec(adapter_type).driver)


def _driver_class(target: str) -> type:
    module_name, class_name = target.split(":", 1)
    implementation = getattr(import_module(module_name), class_name)
    if not isinstance(implementation, type):
        raise TypeError(f"Registered driver {target!r} is not a class")
    return implementation


adapter_capabilities = api_spec


def validate_model_interaction(adapter_type, *, clock_mode, fps, tokens_per_frame=None):
    """Validate API interaction requirements before starting an episode."""
    del fps, tokens_per_frame
    spec = api_spec(adapter_type)
    if spec.wall_clock_only and clock_mode != "wall":
        raise ValueError(f"The {adapter_type} API requires a wall clock")


# Catalog
"""Load model connection YAMLs and resolve internal API capabilities.

Every file in configs/models uses the same endpoint schema. Its filename is the
catalog identifier. Wire policies live in MODEL_PROFILES above; users configure
connections and reasoning effort. No optional model SDK is imported by this loader.
"""

import difflib as difflib
import sys as sys
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from typing import Optional as Optional
from realtime_framework.entity.config import ModelConfig


# All maintained runs omit temperature and use the provider default.
TEMPERATURE_POLICIES = frozenset({"provider_default"})


from realtime_framework.entity.model import ModelSpec as ModelEntry


# --- the catalog ------------------------------------------------------------
def _load_entries() -> list[ModelEntry]:
    """Load the shared connection schema; filenames are model identifiers."""
    import sys as sys
    from pathlib import Path as Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    models_dir = root / "configs" / "models"
    out: list[ModelEntry] = []
    definitions = load_definitions(models_dir)
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
    return APIS[entry.adapter_type].vendor or entry.provider


def python_env(key: str) -> str:
    """Return the Python interpreter configured for model subprocesses."""
    return get(key).python_env


def supports_finite_frame_budget(adapter_type: str) -> bool:
    return adapter_capabilities(adapter_type).finite_frame_budget


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
            # Same default as YAML presets; STREAM_CHAT_INCLUDE_USAGE below can still override.
            fields.setdefault("chat_include_usage", True)
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
