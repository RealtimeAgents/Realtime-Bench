from __future__ import annotations

# Artifacts
"""Persist model-produced state before evaluation so inference and scoring stay decoupled."""

import hashlib as hashlib
import json as json
from pathlib import Path as Path
import time as time

from realtime_framework.evaluation.config import configuration_sha256
from realtime_framework.evaluation.config import harness_fingerprint, harness_version


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_suite_metadata(*, task_id=None, ctask=None) -> dict:
    """Describe the active suite and, when possible, one of its registered tasks.

    Runs outside the active registry still receive the suite identity and hash but intentionally omit
    task-level metadata.
    """
    import realtime_tasks.registry as registry

    resolved_id = task_id if task_id in registry.TASKS else None
    if resolved_id is None and ctask is not None:
        declared_id = getattr(ctask, "task_id", None)
        if declared_id in registry.TASKS:
            resolved_id = declared_id
        else:
            resolved_id = next(
                (
                    registered_id
                    for registered_id, registered in registry.TASKS.items()
                    if registered is ctask
                ),
                None,
            )
    metadata = {
        "suite_id": registry.suite_id(),
        "suite_version": registry.suite_version(),
        "suite_hash": registry.suite_hash(),
    }
    if resolved_id is not None:
        metadata.update(
            {
                "task_id": resolved_id,
                "public_category": registry.public_category(resolved_id),
                "execution_kind": registry.execution_kind(resolved_id),
            }
        )
    return metadata


def resolve_run_identity(
    *,
    task_id=None,
    ctask=None,
    adapter=None,
    model=None,
    backend_model=None,
    mode=None,
    cap=None,
    protocol=None,
    harness_hash=None,
    fps=None,
    model_input_fps=None,
    interaction_mode=None,
    runtime_config_hash=None,
) -> dict:
    """Return the canonical fields that distinguish one executable run from another.

    ``fps`` is the environment's frame rate and ``model_input_fps`` the model's;
    ``interaction_mode`` is turn_based or streaming."""
    harness_hash = harness_hash or harness_fingerprint()
    identity = {
        **resolve_suite_metadata(task_id=task_id, ctask=ctask),
        "adapter": adapter,
        "model": model,
        "backend_model": backend_model,
        "cap": cap,
        "protocol": protocol,
        "harness_version": harness_version(harness_hash),
        "harness_hash": harness_hash,
        "fps": fps,
        "model_input_fps": model_input_fps,
        "interaction_mode": interaction_mode,
        "seed": getattr(ctask, "seed", None),
        "runtime_config_hash": runtime_config_hash,
    }
    if mode is not None:
        identity["mode"] = mode
    return identity


def capture_generation_artifact(
    run_dir: Path,
    *,
    clock: Clock,
    ctask=None,
    adapter=None,
    model=None,
    backend_model=None,
    mode=None,
    cap=None,
    protocol=None,
    harness_hash=None,
    fps=None,
    model_input_fps=None,
    interaction_mode=None,
    runtime_config=None,
    runtime_config_hash=None,
    snapshot=None,
) -> dict:
    """Capture the model's output/state before verifier code runs.

    `snapshot`, when given, is the evaluation snapshot ALREADY taken by the caller at the exact
    inference boundary; it is written verbatim so the saved artifact and the run summary score the
    identical frozen state (re-sampling here would fold a second, later live read and let the two
    disagree). Returns the manifest that is also atomically written to
    generation_artifact/manifest.json; its capture time is read on ``clock``, the episode's
    operational clock.
    """
    artifact_dir = Path(run_dir) / "generation_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_identity = resolve_run_identity(
        task_id=getattr(ctask, "task_id", None),
        ctask=ctask,
        adapter=adapter,
        model=model,
        backend_model=backend_model,
        mode=mode,
        cap=cap,
        protocol=protocol,
        harness_hash=harness_hash,
        fps=fps,
        model_input_fps=model_input_fps,
        interaction_mode=interaction_mode,
        runtime_config_hash=runtime_config_hash,
    )
    manifest = {
        "artifact_version": 2,
        "captured_unix_s": clock.time(),
        **run_identity,
        "category": getattr(ctask, "category", None),
        "instance_id": getattr(ctask, "task_id", None),
        "runtime_config": runtime_config,
    }

    if ctask is not None and hasattr(ctask, "evaluation_snapshot"):
        snap = snapshot if snapshot is not None else ctask.evaluation_snapshot()
        state_file = artifact_dir / "task_state.json"
        state_file.write_text(json.dumps(snap, indent=2, default=str))
        manifest["payload"] = {
            "kind": "task_state",
            "state": state_file.name,
            "state_sha256": _sha256(state_file),
        }
    else:
        manifest["payload"] = {"kind": "none"}

    tmp = artifact_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(artifact_dir / "manifest.json")
    return manifest


def save_evaluation(run_dir: Path, evaluation, *, error=None) -> Path:
    """Persist evaluation separately from the generation artifact/run summary."""
    path = Path(run_dir) / "evaluation.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"evaluation": evaluation, "error": error}, indent=2, default=str))
    tmp.replace(path)
    return path


# Cost accounting
"""Turn provider usage into an auditable USD billing summary."""

from decimal import Decimal as Decimal
from decimal import InvalidOperation as InvalidOperation

_MODALITY_COUNT = {
    "text": "provider_text_input_tokens",
    "image": "provider_image_input_tokens",
    "video": "provider_video_input_tokens",
    "audio": "provider_audio_input_tokens",
}
_CACHED_MODALITY_COUNT = {
    "text": "provider_cached_text_input_tokens",
    "image": "provider_cached_image_input_tokens",
    "video": "provider_cached_video_input_tokens",
    "audio": "provider_cached_audio_input_tokens",
}
_OUTPUT_MODALITY_COUNT = {
    "text": "provider_text_output_tokens",
    "audio": "provider_audio_output_tokens",
}


def _amount(value) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() and result >= 0 else None


def _json_amount(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000000001")))


def _complete_count(summary: dict, name: str) -> int | None:
    value = (summary.get("counts") or {}).get(name) or {}
    total = value.get("total") if value.get("complete") else None
    return total if isinstance(total, int) and not isinstance(total, bool) and total >= 0 else None


def _result(
    summary: dict,
    *,
    source: str,
    status: str,
    reason: str | None,
    total: Decimal | None = None,
    known: Decimal | None = None,
    rate_card=None,
    components=None,
    assumptions=None,
) -> dict:
    reported = summary.get("provider_reported_cost") or {}
    return {
        "schema": "realtime_bench.cost_accounting.v2",
        "currency": "USD",
        "source": source,
        "status": status,
        "complete": status == "complete",
        "total_usd": _json_amount(total) if total is not None else None,
        "known_cost_usd": _json_amount(known) if known is not None else None,
        "reason": reason,
        "provider_reported_cost": reported,
        "rate_card": rate_card,
        "components": components or [],
        "assumptions": assumptions or [],
    }


def _unavailable(summary: dict, reason: str, *, rate_card=None, status="unavailable") -> dict:
    reported = summary.get("provider_reported_cost") or {}
    known = _amount(reported.get("known_total_usd"))
    return _result(
        summary,
        source="provider_reported_usage_cost" if known is not None else "unavailable",
        status="partial" if known is not None else status,
        reason=reason,
        known=known,
        rate_card=rate_card,
    )


def summarize_cost(
    token_accounting: dict | None, *, pricing: dict | None = None, provider: str | None = None
) -> dict:
    """Prefer a complete provider invoice amount, otherwise apply a versioned rate card.

    A complete rate-card result requires complete provider totals and cache splits. Modality
    details must reconcile unless the catalog explicitly declares a billing modality for the
    unclassified positive remainder; missing data is never silently priced as text.
    """
    if not isinstance(token_accounting, dict) or not token_accounting.get("responses"):
        return _unavailable(token_accounting or {}, "no_inference_responses")

    reported = token_accounting.get("provider_reported_cost") or {}
    reported_total = _amount(reported.get("known_total_usd"))
    if reported.get("complete") and reported_total is not None:
        return _result(
            token_accounting,
            source="provider_reported_usage_cost",
            status="complete",
            reason=None,
            total=reported_total,
            known=reported_total,
        )

    if provider == "local":
        return _unavailable(
            token_accounting,
            "local_compute_cost_not_monetized",
            rate_card=pricing,
            status="unmonetized",
        )
    if not isinstance(pricing, dict):
        reason = (
            "provider_reported_cost_incomplete"
            if reported.get("responses_with_cost")
            else "no_provider_cost_or_rate_card"
        )
        return _unavailable(token_accounting, reason)
    if pricing.get("currency") != "USD" or pricing.get("unit") != "per_million_tokens":
        return _unavailable(token_accounting, "unsupported_rate_card", rate_card=pricing)

    rates = pricing.get("rates") or {}
    input_rates = rates.get("input") or {}
    cached_rates = rates.get("cached_input") or {}
    cache_write_rates = {
        "5m": rates.get("cache_write_5m_input") or {},
        "1h": rates.get("cache_write_1h_input") or {},
    }
    output_rates = rates.get("output") or {}
    provider_input = _complete_count(token_accounting, "provider_input_tokens")
    provider_output = _complete_count(token_accounting, "provider_billed_output_tokens")
    provider_cached = _complete_count(token_accounting, "provider_cached_input_tokens")
    if provider_input is None or provider_output is None or provider_cached is None:
        return _unavailable(
            token_accounting,
            "incomplete_provider_token_totals",
            rate_card=pricing,
        )

    input_counts = {}
    for modality in input_rates:
        count = _complete_count(token_accounting, _MODALITY_COUNT.get(modality, ""))
        if count is None:
            return _unavailable(
                token_accounting,
                f"missing_{modality}_input_tokens",
                rate_card=pricing,
            )
        input_counts[modality] = count
    classified_input = sum(input_counts.values())
    if classified_input > provider_input:
        return _unavailable(
            token_accounting,
            "input_modalities_exceed_provider_total",
            rate_card=pricing,
        )
    unclassified_input = provider_input - classified_input
    unclassified_modality = pricing.get("unclassified_input_modality")
    if unclassified_input and unclassified_modality not in input_rates:
        return _unavailable(
            token_accounting,
            "input_modalities_do_not_reconcile",
            rate_card=pricing,
        )

    cached_counts = {modality: 0 for modality in input_counts}
    if provider_cached:
        for modality in input_counts:
            if modality not in cached_rates:
                return _unavailable(
                    token_accounting,
                    f"missing_{modality}_cached_input_rate",
                    rate_card=pricing,
                )
            count = _complete_count(
                token_accounting,
                _CACHED_MODALITY_COUNT.get(modality, ""),
            )
            if count is None:
                return _unavailable(
                    token_accounting,
                    f"missing_cached_{modality}_input_tokens",
                    rate_card=pricing,
                )
            cached_counts[modality] = count
        if sum(cached_counts.values()) != provider_cached:
            return _unavailable(
                token_accounting,
                "cached_modalities_do_not_reconcile",
                rate_card=pricing,
            )

    cache_write_counts = {"5m": 0, "1h": 0}
    provider_cache_write = _complete_count(
        token_accounting,
        "provider_cache_write_input_tokens",
    )
    has_cache_write_rates = any(cache_write_rates.values())
    if has_cache_write_rates:
        if provider_cache_write is None:
            return _unavailable(
                token_accounting,
                "missing_cache_write_input_tokens",
                rate_card=pricing,
            )
        for ttl in cache_write_counts:
            count = _complete_count(
                token_accounting,
                f"provider_cache_write_{ttl}_input_tokens",
            )
            if count is None:
                return _unavailable(
                    token_accounting,
                    f"missing_cache_write_{ttl}_input_tokens",
                    rate_card=pricing,
                )
            cache_write_counts[ttl] = count
        if sum(cache_write_counts.values()) != provider_cache_write:
            return _unavailable(
                token_accounting,
                "cache_write_ttls_do_not_reconcile",
                rate_card=pricing,
            )
    elif provider_cache_write:
        return _unavailable(
            token_accounting,
            "missing_cache_write_input_rate",
            rate_card=pricing,
        )

    cache_write_modality = pricing.get("cache_write_input_modality")
    if provider_cache_write and cache_write_modality not in input_counts:
        return _unavailable(
            token_accounting,
            "missing_cache_write_input_modality",
            rate_card=pricing,
        )
    for ttl, count in cache_write_counts.items():
        if count and cache_write_modality not in cache_write_rates[ttl]:
            return _unavailable(
                token_accounting,
                f"missing_cache_write_{ttl}_input_rate",
                rate_card=pricing,
            )

    components = []
    total = Decimal(0)

    def add_component(kind: str, modality: str, tokens: int, rate_value) -> bool:
        nonlocal total
        rate = _amount(rate_value)
        if rate is None:
            return False
        cost = Decimal(tokens) * rate / Decimal(1_000_000)
        total += cost
        components.append(
            {
                "kind": kind,
                "modality": modality,
                "tokens": tokens,
                "rate_per_million_tokens_usd": float(rate),
                "cost_usd": _json_amount(cost),
            }
        )
        return True

    for modality, count in input_counts.items():
        cached = cached_counts[modality]
        cache_written = sum(cache_write_counts.values()) if modality == cache_write_modality else 0
        if cached + cache_written > count:
            return _unavailable(
                token_accounting,
                f"cached_or_cache_written_{modality}_exceeds_input",
                rate_card=pricing,
            )
        if not add_component(
            "input",
            modality,
            count - cached - cache_written,
            input_rates[modality],
        ):
            return _unavailable(token_accounting, "invalid_input_rate", rate_card=pricing)
        if cached and not add_component("cached_input", modality, cached, cached_rates[modality]):
            return _unavailable(token_accounting, "invalid_cached_rate", rate_card=pricing)

    for ttl, count in cache_write_counts.items():
        if not count:
            continue
        if not add_component(
            "cache_write_input",
            cache_write_modality,
            count,
            cache_write_rates[ttl][cache_write_modality],
        ):
            return _unavailable(
                token_accounting,
                f"invalid_cache_write_{ttl}_rate",
                rate_card=pricing,
            )
        components[-1]["cache_ttl"] = ttl

    assumptions = []
    if unclassified_input:
        if not add_component(
            "input",
            "unclassified",
            unclassified_input,
            input_rates[unclassified_modality],
        ):
            return _unavailable(token_accounting, "invalid_input_rate", rate_card=pricing)
        components[-1]["rate_basis_modality"] = unclassified_modality
        assumptions.append(
            {
                "type": "unclassified_input_priced_as",
                "modality": unclassified_modality,
                "tokens": unclassified_input,
            }
        )

    output_counts = {}
    for modality in output_rates:
        count = _complete_count(
            token_accounting,
            _OUTPUT_MODALITY_COUNT.get(modality, ""),
        )
        if count is None:
            return _unavailable(
                token_accounting,
                f"missing_{modality}_output_tokens",
                rate_card=pricing,
            )
        output_counts[modality] = count
    if sum(output_counts.values()) != provider_output:
        return _unavailable(
            token_accounting,
            "output_modalities_do_not_reconcile",
            rate_card=pricing,
        )
    for modality, count in output_counts.items():
        if not add_component("output", modality, count, output_rates[modality]):
            return _unavailable(token_accounting, "invalid_output_rate", rate_card=pricing)

    return _result(
        token_accounting,
        source=(
            "catalog_rate_card_with_declared_fallback"
            if unclassified_input
            else "catalog_rate_card"
        ),
        status="complete",
        reason=None,
        total=total,
        known=total,
        rate_card=pricing,
        components=components,
        assumptions=assumptions,
    )


# Run metrics
"""Aggregate publication-facing metrics for one completed benchmark run."""

from collections import Counter as Counter
import math as math
import re as re


RUN_METRICS_SCHEMA = "realtime_bench.run_metrics.v1"


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _percentile(values: list[float], quantile: float):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _rounded(value):
    return None if value is None else round(float(value), 9)


def _distribution(
    values,
    *,
    total_decisions: int,
    unit: str,
    source: str,
    source_counts=None,
    missing_reason_counts=None,
):
    values = [value for raw in values if (value := _number(raw)) is not None]
    sample_count = len(values)
    result = {
        "unit": unit,
        "source": source,
        "sample_count": sample_count,
        "total_decisions": total_decisions,
        "coverage": round(sample_count / total_decisions, 9) if total_decisions else None,
        "complete": bool(total_decisions and sample_count == total_decisions),
        "mean": _rounded(sum(values) / sample_count) if values else None,
        "p50": _rounded(_percentile(values, 0.50)),
        "p90": _rounded(_percentile(values, 0.90)),
        "p95": _rounded(_percentile(values, 0.95)),
        "min": _rounded(min(values)) if values else None,
        "max": _rounded(max(values)) if values else None,
    }
    if source_counts:
        result["source_counts"] = dict(sorted(source_counts.items()))
    if missing_reason_counts:
        result["missing_reason_counts"] = dict(sorted(missing_reason_counts.items()))
    return result


def _load_lifecycle(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: lifecycle record is not an object")
            records.append(record)
    return records


def summarize_run_metrics(
    lifecycle_path,
    *,
    total_decisions: int,
    solved,
    clock_mode: str,
    tokens_per_frame: int | None,
    fps: float,
    termination: dict | None,
    horizon_s: float,
    validate_decision_count: bool = True,
) -> dict:
    """Return run-level metrics derived from the authoritative decision lifecycle.

    Missing provider measurements remain missing. Rates use only decisions for which the relevant
    predicate is known and expose their denominator explicitly. ``fps`` is the model-input frame
    rate, whose frames ``tokens_per_frame`` is charged per.
    """
    records = _load_lifecycle(Path(lifecycle_path))
    if validate_decision_count and len(records) != total_decisions:
        raise ValueError(
            f"lifecycle decision count {len(records)} != timeline count {total_decisions}"
        )

    charged_tokens = []
    charged_sources = Counter()
    simulated_latencies = []
    wall_latencies = []
    wall_sources = Counter()
    ttfts = []
    ttft_sources = Counter()
    ttft_missing_reasons = Counter()
    cross_frame_decisions = 0
    cross_frame_eligible = 0
    parse_errors = 0
    parse_eligible = 0
    parse_error_reasons = Counter()

    finite_token_clock = (
        clock_mode == "token"
        and isinstance(tokens_per_frame, int)
        and not isinstance(tokens_per_frame, bool)
        and tokens_per_frame > 0
        and _number(fps) not in {None, 0}
    )
    tokens_per_world_second = tokens_per_frame * fps if finite_token_clock else None
    simulated_latency_source = (
        "clock_output_tokens/(tokens_per_frame*fps)"
        if finite_token_clock
        else "paused_environment_zero_generation_delay"
        if clock_mode == "token" and tokens_per_frame is None
        else "not_applicable_outside_token_clock"
    )

    for record in records:
        tokens = _number(record.get("clock_output_tokens"))
        if tokens is not None:
            charged_tokens.append(tokens)
            charged_sources[str(record.get("clock_output_tokens_source") or "unspecified")] += 1
            if finite_token_clock:
                simulated_latencies.append(tokens / tokens_per_world_second)
                cross_frame_eligible += 1
                cross_frame_decisions += tokens > tokens_per_frame
        if clock_mode == "token" and tokens_per_frame is None:
            simulated_latencies.append(0.0)

        wall_latency = _number(record.get("decision_wall_latency_s"))
        wall_source = "decision_wall_latency_s"
        if wall_latency is None:
            wall_latency = _number(record.get("model_latency_s"))
            wall_source = "model_latency_s_compatibility"
        if wall_latency is not None:
            wall_latencies.append(wall_latency)
            wall_sources[wall_source] += 1

        ttft = _number(record.get("provider_ttft_s"))
        if ttft is not None:
            ttfts.append(ttft)
            ttft_sources[str(record.get("provider_ttft_source") or "unspecified")] += 1
        else:
            ttft_missing_reasons[
                str(record.get("provider_ttft_missing_reason") or "not_recorded")
            ] += 1

        protocol = record.get("action_protocol")
        valid_action = protocol.get("valid_action") if isinstance(protocol, dict) else None
        if isinstance(valid_action, bool):
            parse_eligible += 1
            if not valid_action:
                parse_errors += 1
                parse_error_reasons[str(protocol.get("protocol_failure") or "unspecified")] += 1

    task_success_rate = float(solved) if isinstance(solved, bool) else None
    horizon = _number(horizon_s)
    # A solved run always has a termination record: only a finalized loop is scored.
    normalized_time_to_success = (
        termination["end_s"] / horizon
        if solved is True and horizon not in {None, 0}
        else None
    )

    return {
        "run_metrics_schema": RUN_METRICS_SCHEMA,
        # A run_summary describes one task instance, so its task-success rate is the Bernoulli
        # observation 1/0. Suite/category success rates are aggregated across these values.
        "task_success_rate": task_success_rate,
        "decision_simulated_latency_s": _distribution(
            simulated_latencies,
            total_decisions=total_decisions,
            unit="seconds",
            source=simulated_latency_source,
        ),
        "decision_wall_latency_s": _distribution(
            wall_latencies,
            total_decisions=total_decisions,
            unit="seconds",
            source="observation_preparation_through_complete_response_processing",
            source_counts=wall_sources,
        ),
        "cross_frame_rate": (
            round(cross_frame_decisions / cross_frame_eligible, 9) if cross_frame_eligible else None
        ),
        "cross_frame_decisions": cross_frame_decisions,
        "cross_frame_eligible_decisions": cross_frame_eligible,
        "cross_frame_definition": "clock_output_tokens > tokens_per_frame",
        "charged_tokens_per_decision": _distribution(
            charged_tokens,
            total_decisions=total_decisions,
            unit="tokens",
            source="clock_output_tokens",
            source_counts=charged_sources,
        ),
        "action_parse_error_rate": (
            round(parse_errors / parse_eligible, 9) if parse_eligible else None
        ),
        "action_parse_errors": parse_errors,
        "action_parse_eligible_decisions": parse_eligible,
        "action_parse_error_reasons": dict(sorted(parse_error_reasons.items())),
        "action_parse_error_definition": "action_protocol.valid_action == false",
        "normalized_time_to_success": _rounded(normalized_time_to_success),
        "normalized_time_to_success_definition": (
            "termination.end_s/(max_minutes*60), defined only when solved == true"
        ),
        "provider_ttft_s": _distribution(
            ttfts,
            total_decisions=total_decisions,
            unit="seconds",
            source="provider_call_start_to_first_timestamped_output_event",
            source_counts=ttft_sources,
            missing_reason_counts=ttft_missing_reasons,
        ),
    }


STREAMING_SUMMARY_SCHEMA = "realtime_bench.streaming_summary.v1"
_FRAME_BLOCK = re.compile(r"f\d+\.frame")  # harnesses/streaming/history.py frame_block


def summarize_streaming(trace_path, settings: dict | None) -> dict:
    """The streaming section of a native-streaming run's summary, from its episode trace.

    Frames: those captured and pushed into the session, from the kernel's FrameCaptured and
    FramePushed (seconds from capture to push), those dropped by reason (FrameDropped), and those
    the provider confirmed or refused, from the session's delivery records of frame blocks
    (seconds from sent to confirmed). The kernel's records name a frame by its observation id and
    the session's by its frame-table number: the agent's FrameSent ties the two, so a frame pushed
    counts as confirmed when the block it went in under was. Gemini Live confirms no frame.
    Actions: those that arrived, the age of the frame each names when it arrived, and each one's
    outcome (ActionResolved), with the seconds an applied action waited from its arrival and ran.
    Replies: those started and those given up, by reason. Each distribution's total_decisions
    counts the frames or actions it is taken over. ``settings`` is the run configuration's
    streaming settings: the newest frames OpenAI Realtime keeps, which the harness deletes past,
    make the context control exact; Gemini Live's compression thresholds, past which the server
    compresses the session unseen, approximate (phase2.md, 上下文怎么管)."""
    kernel = {
        kind: []
        for kind in (
            "FrameCaptured",
            "FramePushed",
            "FrameDropped",
            "ActionArrived",
            "ActionResolved",
            TraceEvent.INFERENCE_STARTED.value,
            "GenerationAbandoned",
        )
    }
    sent, confirmed, refused = {}, {}, set()
    numbers = {}  # observation id -> the frame-table number it went in under
    events = {"KernelRecord", TraceEvent.SESSION_DELIVERY.value, TraceEvent.FRAME_SENT.value}
    for row in rows(trace_path, events):
        payload = row["payload"]
        if row["event"] == "KernelRecord":
            if payload["kind"] in kernel:
                kernel[payload["kind"]].append(payload["data"])
            continue
        if row["event"] == TraceEvent.FRAME_SENT.value:
            numbers[payload["observation_id"]] = payload["frame"]
            continue
        at = payload["captured_monotonic_ns"] / 1e9
        for content in payload["contents"]:
            block = content["block"]
            if block is None or not _FRAME_BLOCK.fullmatch(block):
                continue
            if payload["state"] == "sent":
                sent.setdefault(block, at)
            elif payload["state"] == "confirmed":
                confirmed.setdefault(block, at)
            elif payload["state"] == "refused":
                refused.add(block)
    captured = {
        d["observation_id"]: d["timestamp"] for d in kernel["FrameCaptured"] if d["error"] is None
    }
    pushed = [d for d in kernel["FramePushed"] if d["error"] is None]
    dropped = Counter(d["reason"] for d in kernel["FrameDropped"])
    arrived = kernel["ActionArrived"]
    applied = [d for d in kernel["ActionResolved"] if d["status"] == "applied"]

    def seconds(values, total):
        return _distribution(
            values, total_decisions=total, unit="seconds", source="episode_trace.jsonl"
        )

    return {
        "schema": STREAMING_SUMMARY_SCHEMA,
        "context_control": (
            None
            if settings is None
            else "exact"
            if "retained_frames" in settings
            else "approximate"
        ),
        "frames": {
            "captured": len(captured),
            "pushed": len(pushed),
            "push_failed": len(kernel["FramePushed"]) - len(pushed),
            "dropped": sum(dropped.values()),
            "dropped_by_reason": dict(sorted(dropped.items())),
            "confirmed": sum(
                f"f{numbers.get(d['observation_id'])}.frame" in confirmed for d in pushed
            ),
            "refused": len(refused),
            "capture_to_push_s": seconds(
                [d["sent_at"] - captured[d["observation_id"]] for d in pushed], len(pushed)
            ),
            "send_to_confirm_s": seconds(
                [at - sent[block] for block, at in confirmed.items() if block in sent],
                len(sent),
            ),
        },
        "actions": {
            "arrived": len(arrived),
            "outcomes": dict(
                sorted(Counter(d["status"] for d in kernel["ActionResolved"]).items())
            ),
            "frame_age_s": seconds([d["frame_age_s"] for d in arrived], len(arrived)),
            "frame_age_frames": _distribution(
                [d["frame_age_frames"] for d in arrived],
                total_decisions=len(arrived),
                unit="frames",
                source="episode_trace.jsonl",
            ),
            "arrival_to_start_s": seconds(
                [d["started_at"] - d["arrived_at"] for d in applied], len(applied)
            ),
            "execution_s": seconds(
                [d["finished_at"] - d["started_at"] for d in applied], len(applied)
            ),
        },
        "replies": {
            "started": len(kernel[TraceEvent.INFERENCE_STARTED.value]),
            "abandoned": dict(
                sorted(Counter(d["reason"] for d in kernel["GenerationAbandoned"]).items())
            ),
        },
    }


# Summary
"""Validation helpers and CLI for complete benchmark run summaries.

With only ``RUN_SUMMARY``, valid JSON with a stop reason is complete, including an error
summary. Versioned callers can additionally require a scored result and
an exact run identity so a stale summary cannot be resumed or aggregated into another cell.

Exit 0 means complete/matched; exit 1 means missing, partial, unscored, or mismatched.
"""
import argparse as argparse
import sys as sys

from realtime_framework.entity.records import EPISODE_TRACE_SCHEMA
from realtime_framework.entity.records import TraceEvent
from realtime_framework.entity.records import validate_episode_trace

# Every episode has exactly one outcome; only pass scores 1. An unsolved episode that reached the
# scored horizon (max_minutes, 5 by default) is a timeout, whatever the task reported at its end;
# any earlier, task-specific end is a fail. The end time is the termination record's, latched
# before teardown, so teardown never turns a fail into a timeout. The scorer input is frozen at
# the horizon, so a solved episode always passes.
OUTCOMES = ("pass", "fail", "timeout", "error")
HORIZON_STOP_REASONS = frozenset({"time_limit", "tick_limit"})


def _reached_horizon(summary: dict) -> bool:
    """Whether the episode ran the whole scored horizon: the horizon stopped it, or it ended at
    or after the horizon whatever the task reported. Summaries written before the termination
    record are judged by their stop reason and world_time_s, as they were when written."""
    termination = summary.get("termination")
    if termination is None:
        if summary.get("stop_reason") in HORIZON_STOP_REASONS:
            return True
        world, minutes = summary.get("world_time_s"), summary.get("max_minutes")
        numbers = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in (world, minutes)
        )
        return numbers and world >= 60 * minutes
    # The record keeps nine decimals (legacy_loop.py _termination).
    horizon_s = round(60 * summary["max_minutes"], 9)
    return termination["stop_reason"] in HORIZON_STOP_REASONS or termination["end_s"] >= horizon_s


def episode_outcome(summary: dict) -> dict:
    """Return ``{"outcome", "error_category", "error_reason"}`` for one run summary."""
    invalid = summary.get("invalid_run")
    if isinstance(invalid, dict):
        return {
            "outcome": "error",
            "error_category": invalid.get("category") or "unknown",
            "error_reason": invalid.get("reason") or "unknown",
        }
    solved = summary.get("solved")
    if type(solved) is not bool:
        return {"outcome": "error", "error_category": "evaluator", "error_reason": "missing_score"}
    if solved:
        outcome = "pass"
    elif _reached_horizon(summary):
        outcome = "timeout"
    else:
        outcome = "fail"
    return {"outcome": outcome, "error_category": None, "error_reason": None}


def _episode_trace_ok(summary_path, summary) -> bool:
    declared = summary.get("episode_trace")
    if not isinstance(declared, dict) or declared.get("schema") != EPISODE_TRACE_SCHEMA:
        return False
    artifact = declared.get("artifact")
    if not isinstance(artifact, str) or not artifact or Path(artifact).name != artifact:
        return False
    try:
        verified = validate_episode_trace(Path(summary_path).parent / artifact)
    except Exception:
        return False
    required = {
        TraceEvent.SNAPSHOT_FROZEN.value,
        TraceEvent.SCORE_PRODUCED.value,
        TraceEvent.EPISODE_FINISHED.value,
    }
    return (
        verified["event_count"] == declared.get("event_count")
        and verified["head_event_sha256"] == declared.get("head_event_sha256")
        and verified["last_event"] == TraceEvent.EPISODE_FINISHED.value
        and required.issubset(verified["events"])
    )


def summary_ok(
    path,
    *,
    expected_suite_id=None,
    expected_suite_version=None,
    expected_suite_hash=None,
    expected_task_id=None,
    expected_adapter=None,
    expected_model=None,
    expected_backend_model=None,
    expected_mode=None,
    expected_cap=None,
    expected_protocol=None,
    expected_harness_hash=None,
    expected_fps=None,
    expected_model_input_fps=None,
    expected_interaction_mode=None,
    expected_seed=None,
    expected_runtime_config_hash=None,
    expected_interactables_protocol=None,
    require_scored=False,
    allow_experimental=False,
) -> bool:
    try:
        with open(path) as f:
            summary = json.load(f)
    except Exception:
        return False
    if not isinstance(summary, dict) or not summary.get("stop_reason"):
        return False
    if require_scored:
        # Experimental systems are scored identically but only accepted on explicit opt-in.
        experimental = (
            allow_experimental
            and summary.get("run_status") == "experimental"
            and summary.get("benchmark_eligible") is False
            and summary.get("comparison_scope") in {"experimental_system", "system_reference"}
        )
        if summary.get("run_status") != "scored" and not experimental:
            return False
        if summary.get("invalid_run") is not None or summary.get("error"):
            return False
        if type(summary.get("solved")) is not bool:
            return False
        outcome = episode_outcome(summary)
        if outcome["outcome"] == "error" or summary.get("outcome") != outcome["outcome"]:
            return False
        verification = summary.get("verification")
        if (
            not isinstance(verification, dict)
            or verification.get("resolved") is not summary.get("solved")
            or type(verification.get("evaluation_version")) is not int
            or verification.get("error")
            or verification.get("evaluation_valid") is False
            or verification.get("ok") is False
            or verification.get("invalid_reasons")
            or verification.get("baseline_suspect") is True
        ):
            return False
        if not _episode_trace_ok(path, summary):
            return False
    expected = {
        "suite_id": expected_suite_id,
        "suite_version": expected_suite_version,
        "suite_hash": expected_suite_hash,
        "task_id": expected_task_id,
        "adapter": expected_adapter,
        "model": expected_model,
        "backend_model": expected_backend_model,
        "mode": expected_mode,
        "cap": expected_cap,
        "protocol": expected_protocol,
        "harness_hash": expected_harness_hash,
        "fps": expected_fps,
        "model_input_fps": expected_model_input_fps,
        "interaction_mode": expected_interaction_mode,
        "seed": expected_seed,
        "runtime_config_hash": expected_runtime_config_hash,
        "interactables_protocol": expected_interactables_protocol,
    }
    # Runs recorded before the frame rates were split fed the model at the environment rate.
    summary.setdefault("model_input_fps", summary.get("fps"))
    return all(value is None or summary.get(field) == value for field, value in expected.items())


def summary_main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--expected-suite-id")
    parser.add_argument("--expected-suite-version")
    parser.add_argument("--expected-suite-hash")
    parser.add_argument("--expected-task-id")
    parser.add_argument("--expected-adapter")
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-backend-model")
    parser.add_argument("--expected-mode")
    parser.add_argument("--expected-cap", type=json.loads)
    parser.add_argument("--expected-protocol")
    parser.add_argument("--expected-harness-hash")
    parser.add_argument("--expected-fps", type=float)
    parser.add_argument("--expected-model-input-fps", type=float)
    parser.add_argument("--expected-interaction-mode")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-runtime-config-hash")
    parser.add_argument("--expected-interactables-protocol")
    parser.add_argument("--require-scored", action="store_true")
    parser.add_argument(
        "--allow-experimental",
        action="store_true",
        help="Accept explicitly ineligible experimental results with full score checks",
    )
    args = parser.parse_args(argv)
    return (
        0
        if summary_ok(
            args.path,
            expected_suite_id=args.expected_suite_id,
            expected_suite_version=args.expected_suite_version,
            expected_suite_hash=args.expected_suite_hash,
            expected_task_id=args.expected_task_id,
            expected_adapter=args.expected_adapter,
            expected_model=args.expected_model,
            expected_backend_model=args.expected_backend_model,
            expected_mode=args.expected_mode,
            expected_cap=args.expected_cap,
            expected_protocol=args.expected_protocol,
            expected_harness_hash=args.expected_harness_hash,
            expected_fps=args.expected_fps,
            expected_model_input_fps=args.expected_model_input_fps,
            expected_interaction_mode=args.expected_interaction_mode,
            expected_seed=args.expected_seed,
            expected_runtime_config_hash=args.expected_runtime_config_hash,
            expected_interactables_protocol=args.expected_interactables_protocol,
            require_scored=args.require_scored,
            allow_experimental=args.allow_experimental,
        )
        else 1
    )


# Video
"""Evidence-aligned active-episode recordings, shared by runners and Harbor imports.

Postprocessing never edits scores, trajectories, or the trace hash chain. Recordings align on
the first X11 input PTS; without it, the recorder launch log gives an approximate alignment.
"""

import os as os
import re as re
import uuid as uuid
from datetime import datetime as datetime
from datetime import timezone as timezone
from statistics import median as median

VERSION = "active-overlay-v3"


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def rows(path, events=None):
    if not Path(path).is_file():
        return []
    result = []
    with Path(path).open() as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if handle.read().strip():
                    raise
                break  # An interrupted writer may leave an incomplete final line.
            if events is None or value.get("event") in events:
                result.append(value)
    return result


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending." + uuid.uuid4().hex)
    try:
        with pending.open("w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def recording_origin(run_dir, trace):
    metadata = read_json(run_dir / "recording-timing.json")
    if finite(metadata.get("first_frame_unix_s")):
        return {
            "recording_start_unix_s": metadata["first_frame_unix_s"],
            "basis": "x11_first_input_pts",
            "precision": "ffmpeg input timestamp",
        }
    log = (
        (run_dir / "runner.out").read_text(errors="replace")
        if (run_dir / "runner.out").exists()
        else ""
    )
    pattern = r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*Screen recording "
    starts = re.findall(pattern + "started", log, re.M)
    ends = re.findall(pattern + "saved", log, re.M)
    if len(starts) != 1 or len(ends) != 1 or "falling back to frame-composited" in log:
        raise ValueError(
            "Missing/ambiguous screen-recorder timestamps or frame-composited recording"
        )
    parse = lambda value: (
        datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc).timestamp()
    )
    # Recording is saved immediately before score/final trace events. Infer the log
    # timezone in 15-minute increments, then require a close independent match.
    anchors = [
        e["wall_time_ns"] / 1e9 for e in trace if e["event"] in {"ScoreProduced", "EpisodeFinished"}
    ]
    matches = []
    for anchor in anchors:
        shift = round((anchor - parse(ends[0])) / 900) * 900
        residual = anchor - parse(ends[0]) - shift
        if abs(shift) <= 14 * 3600 and -1 <= residual <= 5:
            matches.append((abs(residual), shift, residual))
    if not matches:
        raise ValueError("Recorder log timezone cannot be verified against final trace events")
    _, shift, residual = min(matches)
    return {
        "recording_start_unix_s": parse(starts[0]) + shift,
        "basis": "legacy_recorder_launch_log",
        "log_to_utc_s": shift,
        "end_check_residual_s": residual,
        "precision": "Approximate: legacy recorder launch log, first-frame PTS unavailable",
    }


def _active_trace(run_dir):
    trace = rows(
        run_dir / "episode_trace.jsonl",
        {
            "EpisodeStarted",
            "EpisodeFinished",
            "ScoreProduced",
            "ObservationCaptured",
            "InferenceStarted",
            "InferenceCompleted",
            "ActionProposed",
            "ActionResolved",
            "SnapshotFrozen",
            "ActiveLoopStarted",
            "ActiveLoopEnded",
        },
    )
    attempts = {e.get("attempt_id") for e in trace if e["event"] == "EpisodeStarted"}
    if len(attempts) > 1:
        raise ValueError("Multiple trace attempts require explicit recording-to-attempt selection")
    return trace


def active_interval(run_dir, trace=None):
    """Return the agent's scored run as ``(start_unix_s, end_unix_s, boundary)``.

    Runs record it in ``active-loop.json``; runs without that file derive it from their traces.
    """
    run_dir = Path(run_dir)
    explicit = read_json(run_dir / "active-loop.json")
    if finite(explicit.get("start_unix_s")) and finite(explicit.get("end_unix_s")):
        start, end = explicit["start_unix_s"], explicit["end_unix_s"]
        boundary = "explicit_scored_episode_boundaries"
    else:
        summary = read_json(run_dir / "run_summary.json")
        trace = _active_trace(run_dir) if trace is None else trace
        life = rows(run_dir / "action_lifecycle.jsonl")
        snapshots = [e["wall_time_ns"] / 1e9 for e in trace if e["event"] == "SnapshotFrozen"]
        observations = [
            e["payload"].get("timestamp") for e in trace if e["event"] == "ObservationCaptured"
        ]
        observations = [v for v in observations if finite(v)]
        if not snapshots or not observations:
            raise ValueError("No reliable active-loop boundaries (setup-only/incomplete episode)")
        end = snapshots[0]
        if summary.get("clock_mode") == "wall":
            fps = summary.get("fps") or 1
            origins = [
                r["action_resolved_unix_s"] - r["action_clock_position"] / fps
                for r in life
                if finite(r.get("action_resolved_unix_s"))
                and finite(r.get("action_clock_position"))
            ]
            if origins and max(origins) - min(origins) < 0.25:
                start = median(origins)
                boundary = "legacy_wall_origin_from_action_clock_to_frozen_snapshot"
            else:
                start = min(observations)
                boundary = "legacy_first_observation_to_frozen_snapshot"
        else:
            # Token-clock observation.timestamp is simulated time (often 0),
            # whereas the screen recorder and trace envelope use physical time.
            start = min(
                e["wall_time_ns"] / 1e9 - max(0, e["payload"].get("capture_latency_s") or 0)
                for e in trace
                if e["event"] == "ObservationCaptured"
            )
            boundary = "legacy_first_capture_wall_time_to_frozen_snapshot"
    if not (finite(start) and finite(end) and end > start):
        raise ValueError("Invalid active-loop interval")
    return start, end, boundary


def active_timeline(run_dir):
    run_dir = Path(run_dir)
    summary = read_json(run_dir / "run_summary.json")
    trace = _active_trace(run_dir)
    life = rows(run_dir / "action_lifecycle.jsonl")
    start, end, boundary = active_interval(run_dir, trace)
    alignment = recording_origin(run_dir, trace)
    begin = start - alignment["recording_start_unix_s"]
    finish = end - alignment["recording_start_unix_s"]
    if begin < -0.5 or finish <= 0:
        raise ValueError("Active loop is outside recording clock")
    inference = []
    for e in trace:
        if e["event"] == "InferenceStarted":
            inference.append(
                {"decision": e["payload"].get("decision"), "start": e["wall_time_ns"] / 1e9}
            )
    decisions = {r.get("decision", r.get("tick")): r for r in rows(run_dir / "trajectory.jsonl")}
    thinking = {
        key: r.get("thinking") for key, r in decisions.items() if isinstance(r.get("thinking"), str)
    }
    inspections = rows(run_dir / "tool_calls.jsonl")
    inspections_by_call = {r.get("call_id"): r for r in inspections if r.get("call_id")}
    inspections_by_observation = {}
    for inspection in inspections:
        inspections_by_observation.setdefault(inspection.get("observation_id"), []).append(
            inspection
        )
    timestamps = {
        kind: {
            e["payload"].get("decision"): e["wall_time_ns"] / 1e9
            for e in trace
            if e["event"] == kind
        }
        for kind in ("InferenceCompleted", "ActionProposed", "ActionResolved")
    }
    events = []
    for r in life:
        at = r.get("action_resolved_unix_s")
        if not finite(at):
            continue
        action = r.get("action") or {}
        proof = decisions.get(r.get("decision"), {}).get("inspection") or {}
        inspection, proof = _action_inspection(
            r, proof, inspections_by_call, inspections_by_observation
        )
        events.append(
            {
                "decision": r.get("decision"),
                "time": at - start,
                "proposed_time": timestamps["ActionProposed"].get(r.get("decision"), at) - start,
                "resolved_time": timestamps["ActionResolved"].get(r.get("decision"), at) - start,
                "status": r.get("status", "unknown"),
                "action": action,
                "text": (r.get("action_protocol") or {}).get("action_text")
                or json.dumps(
                    {"action": action.get("action_type"), **(action.get("parameters") or {})},
                    ensure_ascii=False,
                ),
                "thinking": thinking.get(r.get("decision")),
                "points": action_points(r),
                "targets": action_targets(r, inspection, proof),
            }
        )
    resolved = {r.get("decision"): r.get("action_resolved_unix_s") for r in life}
    spans = [
        {
            "decision": i["decision"],
            "start": max(0, i["start"] - start),
            "end": min(
                end,
                timestamps["InferenceCompleted"].get(i["decision"])
                or resolved.get(i["decision"])
                or end,
            )
            - start,
        }
        for i in inference
    ]
    return {
        "schema": VERSION,
        "clock_mode": summary.get("clock_mode"),
        "start_unix_s": start,
        "end_unix_s": end,
        "duration_s": end - start,
        "trim_start_s": max(0, begin),
        "trim_end_s": finish,
        "boundary_basis": boundary,
        "alignment": alignment,
        "inferences": spans,
        "events": events,
        "annotation_policy": {
            "action_anchor": "action_resolved_unix_s (dispatch entry, not completion)",
            "preview_lead_s": 0,
            "preview_is_retrospective": False,
            "targets_start_at": "dispatch entry",
            "targets_end_at": "min(ActionResolved + 1 s, next dispatch entry)",
            "target_boxes_version": 1,
            "target_box_basis": "action observation's delivered mark map; not live element tracking",
            "thinking_end": "InferenceCompleted, with dispatch entry as legacy fallback",
        },
        "note": "Thinking labels denote an outstanding model call, not verified internal reasoning. "
        "Actions are shown from dispatch entry without any preview or time offset. "
        "Only recorded reasoning text is shown after resolution. "
        "Historical end uses the frozen-state trace "
        "boundary and may include brief snapshot/cleanup overhead. Token-clock video "
        "plays physical active time; its scored horizon remains simulated time.",
    }


def _grounded_observation(event):
    """The observation whose marks resolved the action: the frame a streaming action names,
    which its grounding records, or else the action's own."""
    grounding = (event.get("action_protocol") or {}).get("grounding") or {}
    return grounding.get("observation_id") or event.get("observation_id")


def _action_inspection(event, proof, inspections_by_call, inspections_by_observation):
    """The inspection row an action's targets are drawn from, and the proof that checks it: the
    decision's own row, or for a streaming action on an older frame that frame's row, which the
    decision's proof does not describe."""
    grounded = _grounded_observation(event)
    if grounded != event.get("observation_id"):
        candidates = inspections_by_observation.get(grounded, [])
        return (candidates[0] if len(candidates) == 1 else None), {}
    inspection = inspections_by_call.get(proof.get("call_id"))
    if inspection is None and not proof.get("call_id"):
        candidates = inspections_by_observation.get(event.get("observation_id"), [])
        if len(candidates) == 1:
            inspection = candidates[0]
    return inspection, proof


def action_targets(event, inspection, proof):
    """Recover only explicitly resolved marks from the observation whose marks resolved them."""
    observation_id = _grounded_observation(event)
    if not inspection or not observation_id:
        return []
    if inspection.get("observation_id") != observation_id:
        return []
    if proof.get("mark_map_sha256") and proof["mark_map_sha256"] != inspection.get(
        "mark_map_sha256"
    ):
        return []
    grounding = (event.get("action_protocol") or {}).get("grounding") or {}
    if grounding.get("valid") is not True:
        return []
    transform = (event.get("action") or {}).get("coordinate_transform") or {}
    size, affine = transform.get("screen_size"), transform.get("coordinate_transform") or {}
    if not isinstance(size, list) or len(size) != 2 or not all(finite(v) and v > 0 for v in size):
        return []
    if not all(finite(affine.get(k)) for k in ("scale_x", "scale_y", "offset_x", "offset_y")):
        return []
    delivered = inspection.get("delivered_coordinate_space") or {}
    if transform.get("image_size") != [delivered.get("width"), delivered.get("height")]:
        return []
    targets = []
    seen = set()
    for resolution in grounding.get("resolutions") or []:
        mark = resolution.get("mark")
        entry = (inspection.get("delivered_mark_map") or {}).get(mark) or {}
        if not isinstance(mark, str) or not re.fullmatch(r"[PR]\d+", mark) or mark in seen:
            continue
        if entry.get("source_id") != resolution.get("source_id"):
            continue
        box = entry.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or not all(finite(v) for v in box):
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        normalized = [
            (x1 * affine["scale_x"] + affine["offset_x"]) / size[0],
            (y1 * affine["scale_y"] + affine["offset_y"]) / size[1],
            (x2 * affine["scale_x"] + affine["offset_x"]) / size[0],
            (y2 * affine["scale_y"] + affine["offset_y"]) / size[1],
        ]
        normalized = [max(0, min(1, v)) for v in normalized]
        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
            continue
        targets.append(
            {
                "mark": mark,
                "bbox": normalized,
                "label": entry.get("label") or "",
                "observation_id": observation_id,
            }
        )
        seen.add(mark)
    return targets


def action_points(event):
    action = event.get("action") or {}
    transform = action.get("coordinate_transform") or {}
    size = transform.get("screen_size")
    if not size or len(size) != 2 or not all(finite(v) and v > 0 for v in size):
        return []
    coordinates = (
        event.get("coordinate_trace")
        or (event.get("execution_result") or {}).get("coordinate_trace")
        or {}
    )
    desktop = coordinates.get("desktop_parameters")
    parameters = desktop if desktop is not None else action.get("parameters") or {}
    affine = transform.get("coordinate_transform") or {}
    result = []
    keys = [("x1", "y1"), ("x2", "y2")] if action.get("action_type") == "drag" else [("x", "y")]
    for xkey, ykey in keys:
        x, y = parameters.get(xkey), parameters.get(ykey)
        if not finite(x) or not finite(y):
            continue
        if desktop is None and action.get("coordinate_space") != "desktop_px":
            if not all(
                finite(affine.get(k)) for k in ("scale_x", "scale_y", "offset_x", "offset_y")
            ):
                continue
            x, y = (
                x * affine["scale_x"] + affine["offset_x"],
                y * affine["scale_y"] + affine["offset_y"],
            )
        if 0 <= x <= size[0] and 0 <= y <= size[1]:
            result.append([x / size[0], y / size[1]])
    return result


def browser_metadata(run_dir, *, processed=None):
    """Describe a playback window and overlays without opening an encoder or editing media.

    ``processed`` describes legacy *displayed* media whose annotations are already
    embedded. Disable dynamic overlays there instead of annotating it twice.
    Event times remain relative to the scored start; clip bounds use MP4 time.
    """
    if processed is not None:
        duration = processed.get("output_duration_s")
        return {
            "schema": "browser-overlay-v1",
            "state": "legacy_processed",
            "clip": {"start_s": 0, "end_s": duration},
            "timeline_offset_s": 0,
            "baked_annotations": True,
            "events": [],
            "inferences": [],
            "note": "This historical video already contains embedded annotations; live overlays are disabled.",
        }
    try:
        timeline = active_timeline(run_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {
            "schema": "browser-overlay-v1",
            "state": "unavailable",
            "clip": None,
            "events": [],
            "inferences": [],
            "baked_annotations": False,
            "note": str(exc),
        }
    return {
        "schema": "browser-overlay-v1",
        "state": "ready",
        "clip": {"start_s": timeline["trim_start_s"], "end_s": timeline["trim_end_s"]},
        "timeline_offset_s": timeline["start_unix_s"]
        - timeline["alignment"]["recording_start_unix_s"],
        "baked_annotations": False,
        "events": timeline["events"],
        "inferences": timeline["inferences"],
        "alignment": timeline["alignment"],
        "annotation_policy": timeline["annotation_policy"],
        "note": timeline["note"],
    }


def finalize_video(run_dir):
    """Publish only browser playback metadata; never crop, encode, or replace a video."""
    try:
        if (Path(run_dir) / "video.mp4").is_file():
            metadata = browser_metadata(run_dir)
            atomic_json(Path(run_dir) / "recording-timeline.json", metadata)
    except Exception as exc:
        atomic_json(
            Path(run_dir) / "recording-timeline-error.json",
            {
                "error": str(exc),
                "schema": "browser-overlay-v1",
                "original_retained": True,
            },
        )
        print(f"[video] playback metadata unavailable; original retained: {exc}", flush=True)


# Token usage
"""Self-contained provider/native token details for one trajectory decision."""

TRAJECTORY_USAGE_SCHEMA = "trajectory.token_usage.v2"


def trajectory_usage(entry):
    """Build one self-contained, finalized trajectory usage record."""
    raw = entry.get("usage")
    if isinstance(raw, dict) and raw.get("schema") == TRAJECTORY_USAGE_SCHEMA:
        return raw
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    provider = raw if isinstance(raw, dict) else None
    accounting = entry.get("_token_accounting")
    accounting = accounting if isinstance(accounting, dict) else {}
    values = provider or {}

    def final(name):
        return accounting.get(name, entry.get(name))

    provider_input = final("provider_input_tokens")
    if provider_input is None:
        provider_input = values.get("input_tokens", values.get("prompt_tokens"))
    provider_total = final("provider_total_tokens")
    if provider_total is None:
        provider_total = values.get("total_tokens")
    return {
        "schema": TRAJECTORY_USAGE_SCHEMA,
        "provider_token_normalization_schema": final("provider_token_normalization_schema"),
        "provider_token_semantics": final("provider_token_semantics"),
        "clock_output_tokens": final("clock_output_tokens"),
        "clock_output_tokens_source": final("clock_output_tokens_source"),
        "unclamped_clock_output_tokens": final("unclamped_clock_output_tokens"),
        "post_horizon_output_tokens": final("post_horizon_output_tokens"),
        "native_output_tokens": final("native_received_output_tokens"),
        "native_output_count_source": final("native_received_output_tokens_source"),
        "provider_input_tokens": provider_input,
        "provider_output_tokens": final("provider_output_tokens"),
        "provider_output_tokens_source": final("provider_output_tokens_source"),
        "provider_reasoning_output_tokens": final("provider_reasoning_output_tokens"),
        "provider_reasoning_output_tokens_source": final("provider_reasoning_output_tokens_source"),
        "provider_visible_output_tokens": final("provider_visible_output_tokens"),
        "provider_visible_output_tokens_source": final("provider_visible_output_tokens_source"),
        "provider_visible_output_tokens_missing_reason": final(
            "provider_visible_output_tokens_missing_reason"
        ),
        "provider_native_response_output_tokens": final("provider_native_response_output_tokens"),
        "provider_audio_output_tokens": final("provider_audio_output_tokens"),
        "provider_audio_output_tokens_source": final("provider_audio_output_tokens_source"),
        "provider_audio_output_tokens_missing_reason": final(
            "provider_audio_output_tokens_missing_reason"
        ),
        "provider_total_tokens": provider_total,
        "provider_native_total_tokens": final("provider_native_total_tokens"),
        "provider_protocol_family": final("provider_protocol_family"),
        "provider_upstream": final("provider_upstream"),
        "provider_usage": raw,
        "provider_usage_missing_reason": (
            final("provider_usage_missing_reason") if raw is None else None
        ),
    }


# Report wall clock
"""Export validated wall-clock results, grouped by configuration and public category."""

from collections import defaultdict as defaultdict


def distribution(values, unit, total):
    return _distribution(
        values, total_decisions=total, unit=unit, source="validated action_lifecycle.jsonl"
    )


_FRAME_COUNTS = ("captured", "pushed", "push_failed", "dropped", "confirmed", "refused")


def _streaming_totals(scored) -> dict:
    """A streaming group's counts, summed over its scored tasks' streaming sections
    (summarize_streaming), and its context control, which its runtime configuration fixes."""
    sections = [row["streaming"] for row in scored]
    frames, dropped, outcomes, abandoned = Counter(), Counter(), Counter(), Counter()
    arrived = started = 0
    for section in sections:
        frames.update({name: section["frames"][name] for name in _FRAME_COUNTS})
        dropped.update(section["frames"]["dropped_by_reason"])
        arrived += section["actions"]["arrived"]
        outcomes.update(section["actions"]["outcomes"])
        started += section["replies"]["started"]
        abandoned.update(section["replies"]["abandoned"])
    return {
        "context_control": sections[0]["context_control"] if sections else None,
        "frames": {
            **{name: frames[name] for name in _FRAME_COUNTS},
            "dropped_by_reason": dict(sorted(dropped.items())),
        },
        "actions": {"arrived": arrived, "outcomes": dict(sorted(outcomes.items()))},
        "replies": {"started": started, "abandoned": dict(sorted(abandoned.items()))},
    }


def prepare_episode_report(args):
    """Record expected identity before execution, independently of the resulting summary."""
    from realtime_framework.evaluation.config import config_fingerprint, effective_run_config
    from realtime_framework.providers import registry as catalog

    model_config = catalog.to_model_config(
        args.model, fps=args.model_input_fps, reasoning_effort=args.reasoning_effort
    )
    runtime_hash = config_fingerprint(effective_run_config(
        model_config=model_config, fps=args.fps, max_minutes=args.max_minutes,
        mode=args.mode, clock_mode=args.clock, protocol=args.protocol, seed=args.seed,
        tokens_per_frame=None if args.tokens_per_frame == "unlimited" else int(args.tokens_per_frame),
        think_budget_s=args.think_budget_s, per_move_cap_s=args.per_move_cap_s,
        interaction_mode=args.interaction_mode,
    ))
    expected = {
        **resolve_run_identity(
            task_id=args.task, model=args.model, adapter=model_config.adapter_type,
            backend_model=model_config.model_name, mode=args.mode, cap=args.max_minutes,
            protocol=args.protocol, fps=args.fps, model_input_fps=args.model_input_fps,
            interaction_mode=args.interaction_mode, runtime_config_hash=runtime_hash,
        ),
        "seed": args.seed,
        "benchmark_cell": args.model,
        "reasoning_effort": args.reasoning_effort,
        "clock_mode": args.clock,
        "tokens_per_frame": None if args.clock == "wall" or args.tokens_per_frame == "unlimited" else int(args.tokens_per_frame),
    }
    path = Path(args.results_dir) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": [expected]}, indent=2) + "\n")


def finish_episode_report(root, output=None):
    """Validate one episode without repairing or stamping its runner-owned summary."""
    result = report(root)
    destinations = [Path(root) / "report.json"]
    if output:
        destinations.append(Path(output))
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(path)
    print(f"Validated report: {destinations[0]}", flush=True)
    return int(not result["groups"] or any(
        group["invalid"] or group["missing"] for group in result["groups"]
    ))


def report(root):
    root = Path(root)
    groups = defaultdict(list)
    for cell_path in sorted(root.glob("*/summary.json")):
        cell = json.loads(cell_path.read_text())
        for expected in cell["runs"]:
            path = (
                cell_path.parent
                / expected["task_id"].split("__", 1)[-1].replace("/", "_")
                / "run_summary.json"
            )
            identity = {
                field: expected.get(field, cell.get(field))
                for field in (
                    "suite_id",
                    "suite_version",
                    "suite_hash",
                    "harness_hash",
                    "model",
                    "benchmark_cell",
                    "mode",
                    "protocol",
                    "fps",
                    "model_input_fps",
                    "interaction_mode",
                    "seed",
                    "cap",
                    "reasoning_effort",
                    "runtime_config_hash",
                    "public_category",
                )
            }
            if identity["model_input_fps"] is None:
                # Runs recorded before the frame rates were split fed the model at the
                # environment rate.
                identity["model_input_fps"] = identity["fps"]
            identity["clock_mode"] = expected.get("clock_mode", cell.get("clock_mode", "wall"))
            identity["tokens_per_frame"] = expected.get("tokens_per_frame", cell.get("tokens_per_frame"))
            row = {
                "task_id": expected["task_id"],
                "summary": str(path),
                "status": "missing",
                "outcome": "error",
                "error_category": "runner",
                "error_reason": "missing_result",
            }
            if path.is_file():
                row["status"] = "invalid"
                try:
                    summary = json.loads(path.read_text())
                    row.update(episode_outcome(summary))
                    cost = (summary.get("cost_accounting") or {})
                    row["known_cost_usd"] = _number(cost.get("known_cost_usd"))
                    row["cost_status"] = cost.get("status", "unknown")
                    matched = summary_ok(
                        path,
                        require_scored=True,
                        allow_experimental=os.environ.get("ALLOW_EXPERIMENTAL_MODEL") == "1",
                        **{
                            f"expected_{field}": identity[field]
                            for field in (
                                "suite_id",
                                "suite_version",
                                "suite_hash",
                                "harness_hash",
                                "model",
                                "mode",
                                "protocol",
                                "fps",
                                "model_input_fps",
                                "interaction_mode",
                                "seed",
                                "cap",
                                "runtime_config_hash",
                            )
                        },
                        expected_task_id=row["task_id"],
                        expected_interactables_protocol="environment_injected_each_observation",
                    )
                    if (not matched or summary.get("clock_mode") != identity["clock_mode"]
                            or summary.get("tokens_per_frame") != identity["tokens_per_frame"]):
                        raise ValueError("Unscored result or identity mismatch")
                    records = [
                        json.loads(line)
                        for line in (path.parent / "action_lifecycle.jsonl")
                        .read_text()
                        .splitlines()
                        if line
                    ]
                    if len(records) != summary["decisions"] or len(
                        {r["decision"] for r in records}
                    ) != len(records):
                        raise ValueError("Decision count mismatch")
                    tokens = [record.get("clock_output_tokens") for record in records]
                    if any(
                        value is not None and (type(value) is not int or value < 0)
                        for value in tokens
                    ):
                        raise ValueError("Invalid charged-token value")
                    known = [value for value in tokens if value is not None]
                    counts = ((summary.get("token_accounting") or {}).get("counts") or {}).get(
                        "clock_output_tokens", {}
                    )
                    complete = bool(tokens and len(known) == len(tokens))
                    if (
                        counts.get("responses_with_count", 0) != len(known)
                        or (known and sum(known) != counts.get("known_total", counts.get("total")))
                        or (not known and counts.get("known_total") not in (None, 0))
                        or (counts.get("complete", False) != complete)
                        or (complete and sum(known) != counts.get("total"))
                    ):
                        raise ValueError("Charged-token accounting mismatch")
                    latency = [_number(r.get("decision_wall_latency_s")) for r in records]
                    cost = (summary.get("cost_accounting") or {})
                    row.update(
                        status="scored",
                        solved=summary["solved"],
                        decisions=len(records),
                        backend_model=summary["backend_model"],
                        cost_usd=_number(cost.get("total_usd")) if cost.get("complete") else None,
                        cost_source=cost.get("source"),
                        cost_status=cost.get("status", "unknown"),
                        world_time_s=summary.get("world_time_s"),
                        wall_latency_s=[value for value in latency if value is not None],
                        # Exclude partially recorded tasks from pooled token distributions.
                        charged_tokens=known if len(known) == len(tokens) else [],
                        charged_tokens_complete=complete,
                        # summarize_streaming's, which every streaming run summary has.
                        **(
                            {"streaming": summary["streaming"]}
                            if identity["interaction_mode"] == "streaming"
                            else {}
                        ),
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row["error"] = str(exc)
                    if row["outcome"] != "error":
                        # A summary that fails validation is never scored, whatever it claims.
                        row.update(
                            outcome="error",
                            error_category="evaluator",
                            error_reason="result_validation_failed",
                        )
            groups[json.dumps(identity, sort_keys=True)].append(row)
    # Outside any episode: over finished runs, after runner main() has returned.
    output = {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "groups": []}  # noqa: TID251
    for identity_json, rows in groups.items():
        scored = [row for row in rows if row["status"] == "scored"]
        passed = sum(row["solved"] for row in scored)
        costs = [row["cost_usd"] for row in scored if row["cost_usd"] is not None]
        tokens = [value for row in scored for value in row["charged_tokens"]]
        total_decisions = sum(row["decisions"] for row in scored)
        token_distribution = distribution(tokens, "tokens", total_decisions)
        token_distribution["budget_relative_bins"] = (
            "not_applicable_wall_clock" if json.loads(identity_json)["clock_mode"] == "wall"
            else "absolute_token_counts"
        )
        token_distribution["bins"] = []
        for lower, upper in ((-1, 64), (64, 128), (128, 256), (256, 512), (512, float("inf"))):
            count = sum(lower < value <= upper for value in tokens)
            token_distribution["bins"].append(
                {
                    "range": f"{max(0, lower + 1)}..{upper}",
                    "count": count,
                    "percentage": 100 * count / len(tokens) if tokens else None,
                }
            )
        output["groups"].append(
            {
                **json.loads(identity_json),
                "expected": len(rows),
                "scored": len(scored),
                "passed": passed,
                # Errored and missing runs count as unsolved.
                "success_percentage": 100 * passed / len(rows) if rows else None,
                "outcomes": {
                    name: sum(row["outcome"] == name for row in rows) for name in OUTCOMES
                },
                "error_reasons": dict(
                    sorted(
                        Counter(
                            row["error_reason"] for row in rows if row["outcome"] == "error"
                        ).items()
                    )
                ),
                "invalid": sum(row["status"] == "invalid" for row in rows),
                "invalid_known_spend_usd": sum(
                    row["known_cost_usd"]
                    for row in rows
                    if row["status"] == "invalid" and row.get("known_cost_usd") is not None
                )
                if any(
                    row["status"] == "invalid" and row.get("known_cost_usd") is not None
                    for row in rows
                )
                else None,
                "missing": sum(row["status"] == "missing" for row in rows),
                "partial": len(scored) != len(rows),
                "mean_cost_usd": sum(costs) / len(costs) if costs else None,
                "cost_tasks": len(costs),
                "cost_unknown_or_partial_tasks": len(scored) - len(costs),
                "wall_latency_s": distribution(
                    [v for row in scored for v in row["wall_latency_s"]], "seconds", total_decisions
                ),
                "charged_tokens": token_distribution,
                **(
                    {"streaming": _streaming_totals(scored)}
                    if json.loads(identity_json)["interaction_mode"] == "streaming"
                    else {}
                ),
                "tasks": rows,
            }
        )
    launch_path = root / "wall_clock_launch.json"
    if launch_path.is_file():
        launch = json.loads(launch_path.read_text())
        output["integration"] = {
            "model": launch["model_config"]["model_name"],
            "model_revision": launch.get("model_revision"),
            "mode": launch["plan"]["runtime"]["mode"],
            "configuration_sha256": configuration_sha256(launch),
        }
    # Archived attempts are spend, never additional scored tasks.
    output["retry_attempts"] = []
    for path in sorted(root.glob("*/*.stale.*/run_summary.json")):
        try:
            summary = json.loads(path.read_text())
            cost = (summary.get("cost_accounting") or {})
            output["retry_attempts"].append(
                {
                    "summary": str(path),
                    "task_id": summary.get("task_id"),
                    "cost_status": cost.get("status", "unknown"),
                    "known_cost_usd": cost.get("known_cost_usd"),
                    "total_cost_usd": cost.get("total_usd"),
                }
            )
        except (OSError, ValueError):
            output["retry_attempts"].append({"summary": str(path), "cost_status": "unknown"})
    return output


def report_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args(argv)
    result = report(args.result_root)
    if not result["groups"]:
        parser.error(
            "No completed cell summaries found; wait for the launcher to finalize its summaries"
        )
    print(json.dumps(result, indent=2))
    return 0


# Reevaluate
"""Re-evaluate a saved generation artifact without rerunning the model.

Usage:
  python -m realtime_framework.evaluation.reporting reevaluate RUN_DIR [--replace-summary]
"""

import shutil as shutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _invalid_evaluation(result):
    if not isinstance(result, dict):
        return "evaluation_not_an_object"
    if result.get("error"):
        return "evaluation_error"
    if result.get("evaluation_valid") is False or result.get("ok") is False:
        return "evaluation_declared_invalid"
    if result.get("invalid_reasons"):
        return "evaluation_invalid_reasons"
    if result.get("baseline_suspect") is True:
        return "evaluation_baseline_suspect"
    if type(result.get("resolved")) is not bool:
        return "evaluation_missing_boolean_resolved"
    if type(result.get("evaluation_version")) is not int:
        return "evaluation_missing_version"
    return None


def _registry_for_manifest(manifest, *, allow_suite_hash_mismatch=False):
    """Select and validate the artifact's suite before importing its registry."""
    suite_id = manifest.get("suite_id")
    suite_hash = manifest.get("suite_hash")
    if suite_hash and not suite_id:
        raise RuntimeError("generation artifact has suite_hash but no suite_id")
    if suite_id:
        os.environ["TASK_SUITE"] = suite_id

    # This import must stay below TASK_SUITE assignment. A cached registry from another suite cannot
    # safely be switched in-process; normal CLI use starts with no registry imported.
    import realtime_tasks.registry as registry

    if suite_id and registry.suite_id() != suite_id:
        raise RuntimeError(
            f"artifact requires suite {suite_id!r}, but registry {registry.suite_id()!r} "
            "was already loaded; re-run python -m realtime_framework.evaluation.reporting reevaluate in a fresh process"
        )
    actual_hash = registry.suite_hash()
    if suite_id and suite_hash != actual_hash and not allow_suite_hash_mismatch:
        raise RuntimeError(
            f"suite hash mismatch for {suite_id}: artifact={suite_hash!r}, current={actual_hash!r}; "
            "pass --allow-suite-hash-mismatch to override explicitly"
        )
    return registry


def reeval(run_dir: Path, *, allow_suite_hash_mismatch=False):
    artifact_dir = run_dir / "generation_artifact"
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    registry = _registry_for_manifest(manifest, allow_suite_hash_mismatch=allow_suite_hash_mismatch)
    payload = manifest["payload"]
    if payload["kind"] == "task_state":
        task = registry.get(manifest["task_id"])
        state_path = artifact_dir / payload["state"]
        if sha(state_path) != payload["state_sha256"]:
            raise RuntimeError("state checksum mismatch")
        state = json.loads(state_path.read_text())
        if not hasattr(task, "evaluate_snapshot"):
            raise RuntimeError(f"task {manifest['task_id']} cannot evaluate saved state")
        return task.evaluate_snapshot(state), manifest

    raise RuntimeError(f"artifact kind {payload['kind']!r} is not re-evaluable")


def reevaluate_main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--replace-summary", action="store_true")
    p.add_argument(
        "--allow-suite-hash-mismatch",
        action="store_true",
        help="re-evaluate with the current suite even when its hash differs from the artifact",
    )
    a = p.parse_args()
    if a.replace_summary and a.allow_suite_hash_mismatch:
        p.error(
            "--allow-suite-hash-mismatch is diagnostic-only and cannot be combined with "
            "--replace-summary; a score produced by different task/scorer content must not inherit "
            "the artifact's old suite_hash"
        )
    result, manifest = reeval(a.run_dir, allow_suite_hash_mismatch=a.allow_suite_hash_mismatch)
    import realtime_tasks.registry as registry

    invalid_reason = _invalid_evaluation(result)
    reported_invalid_reason = invalid_reason
    record = {
        # Outside any episode: a finished run, re-evaluated from its saved state.
        "reevaluated_unix_s": time.time(),  # noqa: TID251
        "source_manifest": "generation_artifact/manifest.json",
        "suite_id": manifest.get("suite_id"),
        "suite_version": manifest.get("suite_version"),
        "suite_hash": manifest.get("suite_hash"),
        "evaluator_suite_id": registry.suite_id(),
        "evaluator_suite_version": registry.suite_version(),
        "evaluator_suite_hash": registry.suite_hash(),
        "suite_hash_mismatch_allowed": a.allow_suite_hash_mismatch,
        "evaluation": result,
    }
    out = a.run_dir / "evaluation.reeval.json"
    out.write_text(json.dumps(record, indent=2, default=str))
    if a.replace_summary:
        sp = a.run_dir / "run_summary.json"
        backup = a.run_dir / "run_summary.pre_reeval.json"
        summary = json.loads(sp.read_text()) if sp.exists() else {}
        if manifest.get("suite_id"):
            if not sp.exists():
                raise RuntimeError("cannot replace a missing versioned run_summary.json")
            for field in (
                "suite_id",
                "suite_version",
                "suite_hash",
                "task_id",
                "adapter",
                "model",
                "backend_model",
                "mode",
                "cap",
                "protocol",
                "harness_hash",
                "fps",
                "model_input_fps",
                "interaction_mode",
                "seed",
                "runtime_config_hash",
            ):
                if summary.get(field) != manifest.get(field):
                    raise RuntimeError(
                        f"run summary/artifact identity mismatch for {field}: "
                        f"summary={summary.get(field)!r}, artifact={manifest.get(field)!r}"
                    )
        if sp.exists() and not backup.exists():
            shutil.copy2(sp, backup)
        prior_invalid = summary.get("invalid_run")
        persistent_invalid = (
            prior_invalid
            if isinstance(prior_invalid, dict) and prior_invalid.get("category") != "evaluator"
            else None
        )
        replacement_invalid = (
            {
                "category": "evaluator",
                "reason": invalid_reason,
                "detail": json.dumps(result, default=str)[:2000],
            }
            if invalid_reason
            else persistent_invalid
        )
        if replacement_invalid:
            reported_invalid_reason = replacement_invalid.get("reason")
        summary.update(
            {
                "run_status": "invalid" if replacement_invalid else "scored",
                "invalid_run": replacement_invalid,
                "failure_attribution": (
                    (replacement_invalid or {}).get("category") if replacement_invalid else None
                ),
                "stop_reason": "invalid_run"
                if replacement_invalid
                else summary.get("loop_stop_reason") or summary.get("stop_reason"),
                "solved": None if replacement_invalid else result.get("resolved"),
                "verification": result,
                "evaluation_artifact": out.name,
                "reevaluated_from_saved_artifact": True,
            }
        )
        summary.update(episode_outcome(summary))
        sp.write_text(json.dumps(summary, indent=2, default=str))
    print(
        json.dumps(
            {
                "run_dir": str(a.run_dir),
                "run_status": "invalid" if reported_invalid_reason else "scored",
                "solved": None if reported_invalid_reason else result.get("resolved"),
                "invalid_reason": reported_invalid_reason,
                "output": str(out),
            }
        )
    )


import threading
from realtime_framework.clock import Clock
from realtime_framework.entity.records import (
    GENESIS_HASH,
    _jsonable,
    canonical_json_bytes,
    canonical_sha256,
)


class EpisodeTrace:
    """Thread-safe JSONL writer with monotonic sequence numbers and a SHA-256 hash chain.

    Opening an existing file validates it and resumes at the next sequence number.  This preserves
    append-only behavior when a crashed run is retried in the same run directory; a new ``attempt_id``
    distinguishes the retry without rewriting prior evidence. Each record's wall and monotonic
    times are read on ``clock``, the episode's operational clock.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Clock,
        run_id: str | None = None,
        attempt_id: str | None = None,
        durable: bool = False,
    ):
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.attempt_id = attempt_id or uuid.uuid4().hex
        self.durable = bool(durable)
        self._lock = threading.Lock()
        self._closed = False
        self._sequence, self._head_hash = self._validate_existing()
        self._attempt_start_sequence = self._sequence
        self._needs_separator = False
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                self._needs_separator = handle.read(1) != b"\n"
        self._fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)

    def _validate_existing(self) -> tuple[int, str]:
        if not self.path.exists():
            return 0, GENESIS_HASH
        verified = validate_episode_trace(self.path)
        return verified["event_count"], verified["head_event_sha256"]

    def emit(self, event: TraceEvent | str, payload: dict | None = None, **fields) -> dict:
        event_name = event.value if isinstance(event, TraceEvent) else str(event)
        with self._lock:
            if self._closed:
                raise RuntimeError("episode trace is closed")
            record = {
                "schema": EPISODE_TRACE_SCHEMA,
                "sequence": self._sequence + 1,
                "event_id": uuid.uuid4().hex,
                "event": event_name,
                "run_id": self.run_id,
                "attempt_id": self.attempt_id,
                "wall_time_ns": self.clock.time_ns(),
                "monotonic_ns": self.clock.monotonic_ns(),
                "previous_event_sha256": self._head_hash,
                "payload": _jsonable(payload or {}),
                **_jsonable(fields),
            }
            record["event_sha256"] = canonical_sha256(record)
            encoded = (
                (b"\n" if self._needs_separator else b"") + canonical_json_bytes(record) + b"\n"
            )
            # Append under the process-local lock to prevent event interleaving. High-volume output
            # chunks stay in the page cache; decision boundaries call checkpoint(), and close()
            # always fsyncs the canonical audit trail.
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(self._fd, remaining)
                if written <= 0:
                    raise OSError("episode trace append made no progress")
                remaining = remaining[written:]
            if self.durable:
                os.fsync(self._fd)
            self._needs_separator = False
            self._sequence = record["sequence"]
            self._head_hash = record["event_sha256"]
            return record

    def describe(self) -> dict:
        with self._lock:
            return {
                "schema": EPISODE_TRACE_SCHEMA,
                "artifact": self.path.name,
                "attempt_id": self.attempt_id,
                "event_count": self._sequence,
                "attempt_event_count": self._sequence - self._attempt_start_sequence,
                "head_event_sha256": self._head_hash,
            }

    def checkpoint(self) -> None:
        """Durably flush all events written so far without closing the append-only stream."""
        with self._lock:
            if self._closed:
                raise RuntimeError("episode trace is closed")
            os.fsync(self._fd)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            os.fsync(self._fd)
            os.close(self._fd)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command in {"report", "reevaluate"}:
        del sys.argv[1]
        raise SystemExit(report_main() if command == "report" else reevaluate_main())
    raise SystemExit(summary_main())
