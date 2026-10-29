"""The effective configuration of every catalog model must not change when the API registry and
the driver declarations are reorganized.

effective_config_3831f39.json was recorded on commit 3831f39, before the change (the exports on
77145b9 and 9ada3b0 are byte-identical), by running this file as a script from the root of that
tree:

    .venv/bin/python tests/capabilities/test_effective_config.py OUT

Deliberate changes since were re-exported the same way, offline (LITELLM_LOCAL_MODEL_COST_MAP=true):
- deepseek-v4.1-flash declares its output limit on its endpoint: its native_output_limit_source
  and its run hashes changed.
- LiteLLM 1.104.1 -> 1.106.1: provider_sdk_versions, the run hashes over it, and the
  deepseek-v4.1-flash row of LiteLLM's map (litellm_output_limit) changed.
- The context-policy commits: each one that records a capability in the run identity or changes
  a context-policy value or version re-exports it; only the keys it changes differ.
- Every run configuration records its interaction_mode (turn_based here), so every run hash
  changed.
- Every ModelConfig records its context_preset (kimi_k3 for kimi-k3, which its YAML names, and in
  its catalog defaults; None elsewhere), so every run hash changed.
- jev-1.13 declares its context window and reserve in the UTF-8 bytes its estimate counts: its
  context_window_tokens, context_reserve_tokens and run hash changed.
- The shared retry executor changed how it retries (stable jitter keys, OpenAI Realtime connects
  under the shared policy, no retry once the decision is abandoned), so transport_retry_policy
  went to v2 and every run hash changed.
- chat_completions presets and the inline endpoint request stream usage unless they opt out:
  chat_include_usage of my-model and remote-vlm, and their run hashes, changed.
- gemini-3.8-live and gemini-3.8-live-extended-thinking name the reasoning efforts they take in
  their YAMLs: their entries' reasoning_efforts changed, and no run hash.
- The moonshot API names the output limit as sent on the wire, max_tokens: kimi-k3's
  provider_output_limit.parameter and run hashes changed.
- Every preset declares its output limit on its endpoint, with the value LiteLLM's map gave it, and
  only the generic litellm API still looks one up: the native_output_limit_source of the eleven
  presets that took it from the map, their run hashes, and litellm_output_limit of every preset
  changed.
- The OpenRouter presets that cache at explicit breakpoints set the cache_breakpoint provider
  option in their YAMLs: provider_options of claude-opus-5, claude-fable-5.1 and
  gemini-3.8-flash, and their run hashes, changed.
- chat_include_usage takes its default from ModelConfig alone, which a YAML that sets
  include_usage overrides: my-model's catalog defaults no longer list it, and its model config
  and run hashes did not change. (Edited in place: that key was the one difference.)
- gemini-live goes back to v1.0's non-blocking functions with silent responses: its
  provider_options and run hashes changed.
- kimi-k3 declares an output limit of 131072, not its whole window: its native_output_limit,
  provider_output_limit.uncapped_value and run hashes changed.
- The context-policy items take the names of their configuration keys, and each retention is two
  counts: every context_policy record has screenshots_kept, screenshot_fold_block, turns_kept and
  turn_archive_block for visuals and turns, execution_feedback for feedback,
  archive_over_window for pressure_archive and tool_call_history for tool_call_replay, with the
  same values, so every run hash changed.
- The verbatim-replay item, which no encoder read, is gone: every context_policy record lost
  verbatim_replay (false everywhere), so every run hash changed.

The export uses only functions that exist before and after the change. It covers, per catalog key,
the catalog entry, the ModelConfig the runner builds (providers/registry.py:609-663 there), the
API-level lookups, and the effective run configuration and its hash (evaluation/config.py:596-739
there) on the wall clock, a finite token clock and pause mode, or the error that configuration
raises.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

SNAPSHOT = Path(__file__).with_name("effective_config_3831f39.json")
# Keys the reorganization moves out of the export (none: Gemini's history_format stays in
# provider_options, where ContextPolicy.for_path reads it).
MOVED: frozenset[str] = frozenset()
# Variables the catalog, to_model_config or effective_run_config read (providers/registry.py
# _load_entries, to_model_config; evaluation/config.py _RUNTIME_ENV_KEYS, effective_run_config).
# The export runs without them.
ENVIRONMENT = (
    "BENCHMARK_MODEL",
    "BENCHMARK_BASE_URL",
    "BENCHMARK_MODEL_REVISION",
    "BENCHMARK_LITELLM_KEY_ENV",
    "BENCHMARK_LITELLM_MODEL",
    "BENCHMARK_LITELLM_API_BASE",
    "BENCHMARK_CONTEXT_WINDOW_TOKENS",
    "BENCHMARK_CONTEXT_COUNTER",
    "BENCHMARK_CONTEXT_IMAGE_TOKENS",
    "BENCHMARK_CONTEXT_RESERVE_TOKENS",
    "BENCHMARK_CONTEXT_PRESET",
    "STREAM_CHAT_API_KEY_ENV",
    "STREAM_CHAT_STREAM",
    "STREAM_CHAT_INCLUDE_USAGE",
    "REALTIMEBENCH_PYTHON",
)
CONTAINER_RUNTIME = {
    "runtime": "apptainer", "network": "host", "images": {}, "image_provenance": {},
}
RUNS = (("wall", None, "wall-clock"), ("token", 64, "time-aware-token-v1"),
        ("token", None, "time-aware-token-v1"))


def export() -> dict:
    """Effective configuration of every catalog key; the caller clears the variables above."""
    import realtime_framework.providers.registry as catalog
    from realtime_framework.evaluation.config import config_fingerprint, effective_run_config
    from realtime_framework.sandbox import apptainer

    # Image identities hash the host's SIF files; they are outside this change.
    apptainer.runtime_identity = lambda: CONTAINER_RUNTIME
    catalog._catalog_entries.cache_clear()
    result = {}
    for key in sorted(catalog.CATALOG):
        entry = catalog.get(key)
        spec = {
            field.name: getattr(entry, field.name)
            for field in dataclasses.fields(entry)
            if field.name != "python_env"  # the interpreter path
        }
        effort = entry.defaults["reasoning_effort"]
        model_config = catalog.to_model_config(key, fps=1.0, reasoning_effort=effort)
        runs = {}
        for clock_mode, tokens_per_frame, protocol in RUNS:
            name = f"{clock_mode}/{tokens_per_frame}"
            try:
                config = effective_run_config(
                    model_config=model_config, fps=1.0, max_minutes=5.0, protocol=protocol,
                    think_budget_s=None, per_move_cap_s=None, mode="default",
                    clock_mode=clock_mode, tokens_per_frame=tokens_per_frame,
                )
            except ValueError as exc:
                runs[name] = {"error": [type(exc).__name__, str(exc)]}
                continue
            runs[name] = {"config": config, "hash": config_fingerprint(config)}
        result[key] = {
            "entry": spec,
            "model_config": model_config.model_dump(),
            "vendor": catalog.vendor(key),
            "benchmark_eligible": catalog.benchmark_eligible(key),
            "api_reasoning_efforts": catalog.api_reasoning_efforts(entry.adapter_type),
            "litellm_output_limit": catalog.litellm_output_limit(
                entry.adapter_type, entry.model_name
            ),
            "runs": runs,
        }
    result["<keys>"] = {
        "all": catalog.keys(),
        "streaming": catalog.keys(family="streaming"),
        "vlm": catalog.keys(family="vlm"),
        "openrouter": catalog.keys(provider="openrouter"),
    }
    catalog._catalog_entries.cache_clear()
    return json.loads(json.dumps(result, default=list))


def _differences(before, after, path="") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        out = []
        for key in sorted(set(before) | set(after)):
            where = f"{path}.{key}" if path else key
            if key in MOVED:
                continue
            if key not in after:
                out.append(f"{where}: missing after")
            elif key not in before:
                out.append(f"{where}: new after")
            else:
                out += _differences(before[key], after[key], where)
        return out
    return [] if before == after else [f"{path}: {before!r} != {after!r}"]


def test_effective_configuration_unchanged(monkeypatch):
    for name in ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    from realtime_framework.evaluation.config import _RUNTIME_ENV_KEYS

    for name in _RUNTIME_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    from realtime_framework.sandbox import apptainer

    # export() replaces the image identity; restore it after the test.
    monkeypatch.setattr(apptainer, "runtime_identity", apptainer.runtime_identity)
    before = json.loads(SNAPSHOT.read_text())
    problems = _differences(before, export())
    assert not problems, f"{len(problems)} difference(s):\n" + "\n".join(problems[:40])


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    from realtime_framework.evaluation.config import _RUNTIME_ENV_KEYS

    for name in (*ENVIRONMENT, *_RUNTIME_ENV_KEYS):
        os.environ.pop(name, None)
    Path(sys.argv[1]).write_text(json.dumps(export(), indent=1, sort_keys=True) + "\n")
