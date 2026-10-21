from __future__ import annotations

# Token accounting
"""Model-independent token audit records; unknown usage is never treated as zero."""
from collections import Counter as Counter
from decimal import Decimal as Decimal
from decimal import InvalidOperation as InvalidOperation

from realtime_framework.entity.usage import TOKEN_NORMALIZATION_SCHEMA
from realtime_framework.entity.usage import normalize_provider_token_usage

COUNTS = (
    "clock_output_tokens",
    "unclamped_clock_output_tokens",
    "post_horizon_output_tokens",
    "native_received_output_tokens",
    "provider_input_tokens",
    "provider_output_tokens",
    "provider_total_tokens",
    "provider_billed_output_tokens",
    "provider_text_output_tokens",
    "provider_cached_input_tokens",
    "provider_cache_write_input_tokens",
    "provider_cache_write_5m_input_tokens",
    "provider_cache_write_1h_input_tokens",
    "provider_reasoning_output_tokens",
    "provider_text_input_tokens",
    "provider_image_input_tokens",
    "provider_video_input_tokens",
    "provider_audio_input_tokens",
    "provider_cached_text_input_tokens",
    "provider_cached_image_input_tokens",
    "provider_cached_video_input_tokens",
    "provider_cached_audio_input_tokens",
    "provider_visible_output_tokens",
    "provider_native_response_output_tokens",
    "provider_audio_output_tokens",
    "provider_native_total_tokens",
)


def _number(data, *keys):
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _decimal(data, *keys):
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError):
            continue
        if amount.is_finite() and amount >= 0:
            return amount
    return None


def token_record(result):
    """Keep the original provider object alongside explicitly sourced output counts."""
    usage = result.get("usage")
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    usage = usage if isinstance(usage, dict) else None
    normalized = normalize_provider_token_usage(
        usage,
        protocol_family=result.get("provider_protocol_family"),
        upstream_provider=result.get("provider_upstream"),
        existing=result,
    )
    record = {key: result.get(key) for key in COUNTS}
    record.update(normalized)
    record.update(
        {
            "schema": "realtime_bench.token_accounting.v2",
            "scope": "inference_response",
            "remaining_clock_tokens": result.get("remaining_clock_tokens"),
            "provider_usage": usage,
            "provider_usage_status": "reported" if usage is not None else "unavailable",
            "provider_usage_missing_reason": None
            if usage is not None
            else result.get("provider_usage_missing_reason", "backend_did_not_return_usage"),
            "clock_output_tokens_source": result.get("clock_output_tokens_source"),
            "provider_visible_output_tokens_source": result.get(
                "provider_visible_output_tokens_source"
            )
            or normalized.get("provider_visible_output_tokens_source"),
            "provider_visible_output_tokens_missing_reason": result.get(
                "provider_visible_output_tokens_missing_reason"
            )
            or normalized.get("provider_visible_output_tokens_missing_reason"),
            "provider_reasoning_output_tokens_source": result.get(
                "provider_reasoning_output_tokens_source"
            )
            or normalized.get("provider_reasoning_output_tokens_source"),
            "provider_output_tokens_source": result.get("provider_output_tokens_source")
            or normalized.get("provider_output_tokens_source"),
            "provider_token_normalization_schema": normalized.get(
                "provider_token_normalization_schema"
            ),
            "provider_token_semantics": normalized.get("provider_token_semantics"),
            "provider_protocol_family": normalized.get("provider_protocol_family"),
            "provider_upstream": normalized.get("provider_upstream"),
            "provider_audio_output_tokens_missing_reason": result.get(
                "provider_audio_output_tokens_missing_reason"
            )
            or normalized.get("provider_audio_output_tokens_missing_reason"),
            "provider_response_status": result.get("provider_response_status"),
            "provider_response_status_reason": result.get("provider_response_status_reason"),
            "decision": result.get("decision"),
            "observation_id": result.get("observation_id"),
            "decision_wall_latency_s": result.get("decision_wall_latency_s"),
            "provider_call_wall_latency_s": result.get("provider_call_wall_latency_s"),
            "runtime_turn_wall_latency_s": result.get("runtime_turn_wall_latency_s"),
            "provider_ttft_s": result.get("provider_ttft_s"),
            "provider_ttft_source": result.get("provider_ttft_source"),
            "provider_ttft_missing_reason": result.get("provider_ttft_missing_reason"),
            "latency_clock": "wall_monotonic",
            "clock_delay_clock": result.get("clock_delay_clock"),
            "clock_delay_definition": result.get("clock_delay_definition"),
            "provider_audio_output_tokens_source": result.get("provider_audio_output_tokens_source")
            or normalized.get("provider_audio_output_tokens_source"),
            "native_received_output_tokens_source": result.get(
                "native_received_output_tokens_source"
            ),
            "provider_reported_cost_usd": None,
            "provider_cost_details": None,
        }
    )
    # Adapters may expose normalized top-level counts without a provider ``usage`` object. Never
    # erase those explicit counts merely because ``usage`` is unavailable.
    for field, aliases in (
        ("provider_input_tokens", ("input_tokens", "prompt_tokens")),
        ("provider_output_tokens", ("output_tokens", "completion_tokens")),
        ("provider_total_tokens", ("total_tokens",)),
    ):
        reported = _number(usage, *aliases)
        if reported is not None:
            record[field] = reported
    if usage is not None:
        inp = usage.get("input_token_details") or usage.get("prompt_tokens_details") or {}
        out = (
            usage.get("output_token_details")
            or usage.get("output_tokens_details")
            or usage.get("completion_tokens_details")
            or {}
        )
        cached_total = _number(inp, "cached_tokens")
        record["provider_cached_input_tokens"] = cached_total
        record["provider_cache_write_input_tokens"] = _number(
            inp,
            "cache_write_tokens",
            "cache_creation_input_tokens",
        )
        cache_write = inp.get("cache_write_tokens_details") or {}
        cache_write_total = record["provider_cache_write_input_tokens"]
        for ttl in ("5m", "1h"):
            count = _number(
                cache_write,
                f"ephemeral_{ttl}_tokens",
                f"ephemeral_{ttl}_input_tokens",
            )
            record[f"provider_cache_write_{ttl}_input_tokens"] = (
                0 if count is None and cache_write_total == 0 else count
            )
        cached = inp.get("cached_tokens_details") or inp.get("cached_token_details") or {}
        for modality in ("text", "image", "video", "audio"):
            record[f"provider_{modality}_input_tokens"] = _number(
                inp,
                f"{modality}_tokens",
                f"{modality}_token_count",
            )
            cached_modality = _number(
                cached,
                f"{modality}_tokens",
                f"{modality}_token_count",
            )
            record[f"provider_cached_{modality}_input_tokens"] = (
                0 if cached_modality is None and cached_total == 0 else cached_modality
            )
        reasoning = _number(out, "reasoning_tokens", "thinking_tokens")
        if reasoning is not None:
            record["provider_reasoning_output_tokens"] = reasoning
        audio = _number(out, "audio_tokens")
        if audio is not None:
            record["provider_audio_output_tokens"] = audio
        # Ordinary provider usage reports output_tokens across every output modality. Native
        # adapters whose benchmark clock deliberately excludes a transport-only modality provide
        # these two billing fields explicitly in ``result`` instead.
        if record["provider_billed_output_tokens"] is None:
            record["provider_billed_output_tokens"] = record["provider_output_tokens"]
        if record["provider_text_output_tokens"] is None:
            provider_output = record["provider_output_tokens"]
            if provider_output is not None and audio is not None and audio <= provider_output:
                record["provider_text_output_tokens"] = provider_output - audio
        reported_cost = _decimal(usage, "cost")
        if reported_cost is not None:
            record["provider_reported_cost_usd"] = str(reported_cost)
        details = usage.get("cost_details")
        if isinstance(details, dict):
            record["provider_cost_details"] = details
    if result.get("harness_abort_reason"):
        record["harness_abort_reason"] = result["harness_abort_reason"]
    return record


class TokenAccounting:
    def __init__(self):
        self.responses = 0
        self.sums = Counter()
        self.coverage = Counter()
        self.missing_reasons = Counter()
        self.clock_sources = set()
        self.normalization_schemas = Counter()
        self.normalization_semantics = Counter()
        self.normalization_sources = {
            field: Counter()
            for field in (
                "provider_output_tokens",
                "provider_reasoning_output_tokens",
                "provider_visible_output_tokens",
                "provider_audio_output_tokens",
            )
        }
        self.reported_cost_usd = Decimal(0)
        self.reported_cost_coverage = 0
        self.cost_detail_sums: dict[str, Decimal] = {}
        self.cost_detail_coverage = Counter()

    def add(self, record):
        # A call given up on at the episode's end has a synthesized record and no usage: the
        # totals stay those of the calls made (N34). One given up on at a bound counts, and
        # leaves them incomplete (decision 48).
        if record.get("harness_abort_reason"):
            return
        self.responses += 1
        for key in COUNTS:
            value = _number(record, key)
            if value is not None:
                self.sums[key] += value
                self.coverage[key] += 1
        if record.get("provider_usage_missing_reason"):
            self.missing_reasons[record["provider_usage_missing_reason"]] += 1
        if record.get("clock_output_tokens_source"):
            self.clock_sources.add(record["clock_output_tokens_source"])
        schema = record.get("provider_token_normalization_schema")
        if schema:
            self.normalization_schemas[str(schema)] += 1
        semantics = record.get("provider_token_semantics")
        if semantics:
            self.normalization_semantics[str(semantics)] += 1
        for field, counts in self.normalization_sources.items():
            source = record.get(f"{field}_source")
            if source:
                counts[str(source)] += 1
        try:
            reported_cost = Decimal(str(record.get("provider_reported_cost_usd")))
        except (InvalidOperation, ValueError):
            reported_cost = None
        if reported_cost is not None and reported_cost.is_finite() and reported_cost >= 0:
            self.reported_cost_usd += reported_cost
            self.reported_cost_coverage += 1
        for key, value in (record.get("provider_cost_details") or {}).items():
            try:
                amount = Decimal(str(value))
            except (InvalidOperation, ValueError):
                continue
            if not amount.is_finite() or amount < 0:
                continue
            self.cost_detail_sums[key] = self.cost_detail_sums.get(key, Decimal(0)) + amount
            self.cost_detail_coverage[key] += 1

    def summary(self):
        counts = {}
        for key in COUNTS:
            source_counts = self.normalization_sources.get(key)
            counts[key] = {
                "total": (
                    self.sums[key]
                    if self.responses and self.coverage[key] == self.responses
                    else None
                ),
                "known_total": self.sums[key] if self.coverage[key] else None,
                "responses_with_count": self.coverage[key],
                # complete is numeric coverage only. The separate source flag shows that every
                # normalized value also carries an auditable derivation.
                "complete": bool(self.responses and self.coverage[key] == self.responses),
                "normalization_source_complete": (
                    bool(self.responses and sum(source_counts.values()) == self.responses)
                    if source_counts is not None
                    else None
                ),
            }
        return {
            "schema": "realtime_bench.token_accounting.v2",
            "scope": "inference_responses_excludes_session_prefill_and_replay",
            "responses": self.responses,
            "clock_sources": sorted(self.clock_sources),
            "normalization": {
                "current_schema": TOKEN_NORMALIZATION_SCHEMA,
                "responses_with_schema": sum(self.normalization_schemas.values()),
                "complete": bool(
                    self.responses
                    and self.normalization_schemas[TOKEN_NORMALIZATION_SCHEMA] == self.responses
                ),
                "schema_response_counts": dict(self.normalization_schemas),
                "semantics_response_counts": dict(self.normalization_semantics),
                "field_source_response_counts": {
                    field: dict(counts) for field, counts in self.normalization_sources.items()
                },
            },
            "provider_usage_missing_reasons": dict(self.missing_reasons),
            "provider_reported_cost": {
                "currency": "USD",
                "known_total_usd": (
                    str(self.reported_cost_usd) if self.reported_cost_coverage else None
                ),
                "responses_with_cost": self.reported_cost_coverage,
                "complete": bool(self.responses and self.reported_cost_coverage == self.responses),
                "cost_details_known_totals": {
                    key: str(value) for key, value in sorted(self.cost_detail_sums.items())
                },
                "cost_details_response_coverage": dict(self.cost_detail_coverage),
            },
            "counts": counts,
        }


# Model timing
"""Model wall span and union of busy intervals, independent of process lifetime."""
import json as json
from pathlib import Path as Path


def _rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_model_timing(directory):
    directory = Path(directory)
    intervals, missing, errors = [], 0, []
    source = "provider_runtime"
    try:
        for row in _rows(directory / "trajectory.jsonl"):
            if row.get("harness_abort_reason"):
                continue  # synthesized for a call given up on at the episode's end (N34)
            runtime = row.get("realtime_runtime") or {}
            start = runtime.get("provider_call_started_monotonic_ns")
            duration = row.get("provider_call_wall_latency_s")
            if start is None or duration is None:
                missing += 1
            else:
                intervals.append((start, start + round(duration * 1e9)))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        errors.append(str(exc))
    intervals.sort()
    if any(end < start for start, end in intervals):
        errors.append("Negative model interval")
    complete = bool(intervals) and not missing and not errors
    span = busy = None
    if intervals and not errors:
        span = (max(end for _, end in intervals) - intervals[0][0]) / 1e9
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        busy = sum(end - start for start, end in merged) / 1e9
    return {
        "schema": "model-timing-v1",
        "source": source,
        "definition": "first_model_call_start_to_last_model_call_end_all_phases",
        "status": "complete" if complete else "partial" if intervals else "unavailable",
        "complete": complete,
        "call_count": len(intervals) + missing,
        "unfinished_calls": missing,
        "errors": errors,
        "wall_time_s": span if complete else None,
        "busy_time_s": busy if complete else None,
        "known_wall_span_s": span,
        "known_busy_time_s": busy,
    }
