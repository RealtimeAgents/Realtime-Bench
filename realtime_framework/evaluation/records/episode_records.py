"""What one decision leaves on the record: the result of a generation that raised or was given up
on at a bound, by the watchdog or at the episode's end, its output caps and their audit, what its
result is annotated with, the output tokens its clock is charged with and its token accounting,
whether its result is usable, the protocol of a DONE or FAIL landed as a wait, its frame's
identity for the frozen-stream check, the payloads of its trace events and its action lifecycle
record.

The kernel's shell (shell.py) and the legacy turn loop the tests keep build their records with
these. Nothing here paces the episode, orders the skips
or waits: each function takes what its caller measured, so the pacing on the two sides stays
separate code and running them side by side still compares it (phase2.md, 怎么验证).

A generation's result ``r`` is settled in this order, each step reading what the ones before it
wrote: ``annotate_result`` (the observation id the inspection proof is checked against), the
``frame_usage`` between two ``usage_snapshot`` readings, ``account_tokens``, the
``output_chunks`` and ``inference_completed`` payloads, ``audit_output_caps``,
``finalize_decision_audit`` and last ``result_invalid``. A failed trajectory write is recorded on
``r`` over any failure already there, so it takes precedence. The kernel's shell measures
``frame_usage`` after ``account_tokens``: an agent that holds its result adds its usage only as it
accepts it, after the clock's count is needed, and the shell measures the frame's usage again then.
So the clock's count is the result's own, never what the usage grew by: a result that names no
count has none.

What the kernel's GenerationResult takes from a settled ``r``: ``action`` is ``r["action"]``, a
DONE or FAIL as it came; ``output_tokens`` the count before the allowance clamped it, which the
kernel clamps itself: ``r["unclamped_clock_output_tokens"]`` when ``account_tokens`` clamped it,
else ``r["clock_output_tokens"]``; ``latency`` is ``r["latency"]``;
``protocol_failure`` comes from ``r["action_protocol"]``, which a result the loop made up for a
failed or abandoned call has none of; ``invalid`` is ``result_invalid``.
"""

from __future__ import annotations

import hashlib
import traceback

from realtime_framework.entity.actions import Action, Observation
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.evaluation.token_accounting import Settlement, token_record
from realtime_framework.evaluation.kernel.events import Invalidated
from realtime_framework.sandbox.observation.som import valid_inspection_proof


def generation_error(latency: float, detail: str) -> dict:
    """The result of a generation that raised: no action, and a provider failure that makes it
    unusable."""
    return {
        "action": None,
        "latency": latency,
        "error": True,
        "invalid_category": "provider",
        "invalid_reason": "generation_error",
        "invalid_detail": detail,
    }


def timed_out_result(latency: float, reason: str) -> dict:
    """The result of a call the loop gave up on at a bound (``reason``): no action, and no usage
    reported for it."""
    return {
        "action": None,
        "latency": latency,
        "timed_out": True,
        "harness_timeout_reason": reason,
        "provider_usage_missing_reason": "generation_aborted_at_wall_timeout",
    }


def abandoned_result(latency: float, reason: str) -> dict:
    """The result of a call given up on when the episode ended by ``reason`` (``task_end`` or
    ``invalid_run``) before the call came back: no action, and no usage reported for it. The
    runner never gave one up so: it checked nothing during a generation."""
    return {
        "action": None,
        "latency": latency,
        "stale": True,
        "harness_abort_reason": reason,
        "provider_usage_missing_reason": f"generation_aborted_at_{reason}",
    }


def watchdog_result(latency: float, seconds: float) -> dict:
    """The result of a token-clock call the provider watchdog gave up on after ``seconds``: no
    action, no usage reported for it, and a provider failure that makes the run invalid."""
    return {
        "action": None,
        "latency": latency,
        "invalid_category": "provider",
        "invalid_reason": "generation_watchdog_timeout",
        "invalid_detail": f"generation exceeded {seconds:g}s",
        "provider_usage_missing_reason": "generation_aborted_at_provider_watchdog",
    }


def decision_budget(
    adapter, remaining_clock_tokens: int | None, clock_mode: str
) -> tuple[int | None, DecisionBudget]:
    """The model's native output limit and one decision's output caps. ``remaining_clock_tokens``
    is the finite token clock's allowance left before the horizon, None without one."""
    native_output_limit = getattr(getattr(adapter, "config", None), "native_output_limit", None)
    # None when the model declares no native limit and no token clock bounds the episode.
    runtime_output_cap = (
        native_output_limit
        if remaining_clock_tokens is None
        else remaining_clock_tokens
        if native_output_limit is None
        else min(native_output_limit, remaining_clock_tokens)
    )
    # Bound transports that support a per-turn cap by the episode's remaining charged-token
    # capacity. Drivers without an exact per-turn cap may still return terminal usage beyond T;
    # that excess is retained for billing audit but excluded from the episode clock.
    if remaining_clock_tokens is not None:
        provider_output_token_cap = runtime_output_cap
    else:
        unbounded_output = bool(getattr(adapter, "supports_unbounded_output", False))
        provider_output_token_cap = None if unbounded_output else runtime_output_cap
    transport_cap_mapper = getattr(adapter, "transport_output_token_cap", None)
    transport_output_cap = (
        transport_cap_mapper(provider_output_token_cap)
        if callable(transport_cap_mapper)
        else provider_output_token_cap
    )
    if not (
        transport_output_cap is None
        or transport_output_cap == "inf"
        or (
            isinstance(transport_output_cap, int)
            and not isinstance(transport_output_cap, bool)
            and transport_output_cap > 0
        )
    ):
        raise ValueError("adapter transport output cap must be None, a positive integer, or 'inf'")
    return native_output_limit, DecisionBudget(
        runtime_output_tokens=runtime_output_cap,
        provider_output_tokens=provider_output_token_cap,
        transport_output_tokens=transport_output_cap,
        clock_mode=clock_mode,
    )


def usage_snapshot(adapter) -> dict | None:
    """The adapter's cumulative token usage, or None when it reports none or reading it fails."""
    try:
        value = getattr(adapter, "token_usage", None)
        value = value() if callable(value) else value
        return dict(value) if isinstance(value, dict) else None
    except Exception:  # noqa: BLE001 - usage that cannot be read is usage not reported
        return None


def _observation_batch(observation: Observation) -> list[dict]:
    """The frames a decision observes, as its records list them: the one it captured."""
    metadata = getattr(observation, "metadata", {}) or {}
    return [
        {
            "observation_id": metadata.get("observation_id"),
            "tick": observation.tick_number,
            "timestamp": observation.timestamp,
        }
    ]


def annotate_result(
    r: dict,
    *,
    decision: int,
    observation: Observation,
    native_output_limit: int | None,
    budget: DecisionBudget,
    tokens_per_frame: int | None,
) -> None:
    """Add to a generation result what its decision was given: the observation, the output caps
    (provider and transport caps the adapter reports stay) and how the clock delays the action;
    and the provider's response status from the runtime metadata when the result has none. The
    clock allowance is account_tokens'."""
    metadata = getattr(observation, "metadata", {}) or {}
    clock_mode = budget.clock_mode
    r.update(
        {
            "decision": decision,
            "observation_id": metadata.get("observation_id"),
            "observation_tick": observation.tick_number,
            "observation_batch": _observation_batch(observation),
            "observation_timestamp": observation.timestamp,
            "decision_output_cap": budget.runtime_output_tokens,
            "native_output_limit": native_output_limit,
            "runtime_output_token_cap": budget.runtime_output_tokens,
            "provider_output_token_cap": r.get(
                "provider_output_token_cap", budget.provider_output_tokens
            ),
            "transport_output_token_cap": r.get(
                "transport_output_token_cap", budget.transport_output_tokens
            ),
            "clock_delay_clock": (
                "simulated_token_clock" if clock_mode == "token" else "wall_clock"
            ),
            "clock_delay_definition": (
                "complete_clock_output_tokens_divided_by_tokens_per_frame"
                if clock_mode == "token" and tokens_per_frame is not None
                else "paused_environment_no_generation_delay"
                if clock_mode == "token"
                else "complete_decision_wall_latency"
            ),
        }
    )
    runtime_metadata = r.get("realtime_runtime")
    if isinstance(runtime_metadata, dict):
        for name in ("provider_response_status", "provider_response_status_reason"):
            if r.get(name) is None and runtime_metadata.get(name) is not None:
                r[name] = runtime_metadata[name]


def frame_usage(before: dict, after: dict) -> dict:
    """What one generation added to the adapter's cumulative usage, counter by counter."""
    keys = set(before) | set(after)
    return {
        name: max(0, int(after.get(name, 0) or 0) - int(before.get(name, 0) or 0))
        for name in keys
        if isinstance(before.get(name, 0), (int, float))
        and isinstance(after.get(name, 0), (int, float))
    }


def _usage_number(usage, *names):
    if not isinstance(usage, dict):
        return None
    for name in names:
        value = usage.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return int(value)
    return None


def clock_output_tokens(r: dict) -> tuple[int | None, str | None]:
    """Return the complete native output count and its clock source.

    Remote cells must use the API's response-level output usage. Local native runtimes use the
    token IDs emitted by that runtime. The count is not clamped to the episode's remaining
    allowance.
    """
    provider_missing = r.get("provider_usage_missing_reason")
    if not provider_missing:
        value = r.get("provider_output_tokens")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return int(value), "provider_reported_output_usage"
        value = _usage_number(r.get("usage"), "output_tokens", "completion_tokens")
        if value is not None:
            return value, "provider_reported_output_usage"
    value = r.get("native_received_output_tokens")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return int(value), (
            r.get("native_received_output_tokens_source") or "native_runtime_output_tokens"
        )
    return None, None


def account_tokens(r: dict, remaining_clock_tokens: int | None) -> Settlement:
    """Settle the result's clock output tokens and its token record, once per result, and return
    its settlement, which the episode's token accounting and model timing take (clock.py
    Settlement). Output past the remaining allowance is billed but not charged to the clock: the
    clock count stops at the allowance and the excess is kept as post-horizon tokens."""
    count, source = clock_output_tokens(r)
    if remaining_clock_tokens is not None and count is not None and count > remaining_clock_tokens:
        r["unclamped_clock_output_tokens"] = count
        r["post_horizon_output_tokens"] = count - remaining_clock_tokens
        count = remaining_clock_tokens
        source = f"episode_horizon_clamped:{source or 'unknown'}"
    r["clock_output_tokens"] = count
    r["clock_output_tokens_source"] = source
    r["action_output_tokens"] = count
    r["remaining_clock_tokens"] = remaining_clock_tokens
    r.setdefault("unclamped_clock_output_tokens", None)
    r.setdefault("post_horizon_output_tokens", 0)
    r["token_accounting"] = token_record(r)
    for name in (
        "provider_input_tokens",
        "provider_output_tokens",
        "provider_total_tokens",
        "provider_reasoning_output_tokens",
        "provider_visible_output_tokens",
        "provider_audio_output_tokens",
        "provider_output_tokens_source",
        "provider_reasoning_output_tokens_source",
        "provider_visible_output_tokens_source",
        "provider_visible_output_tokens_missing_reason",
        "provider_audio_output_tokens_source",
        "provider_audio_output_tokens_missing_reason",
        "provider_token_normalization_schema",
        "provider_token_semantics",
        "provider_protocol_family",
        "provider_upstream",
    ):
        r[name] = r["token_accounting"].get(name)
    return Settlement(
        r["token_accounting"],
        (r.get("realtime_runtime") or {}).get("provider_call_started_monotonic_ns"),
        r.get("provider_call_wall_latency_s"),
    )


def audit_output_caps(adapter, r: dict, budget: DecisionBudget) -> None:
    """Record the adapter's action cap for the decision on ``r``. No adapter hands a part of its
    output off, so the handoff cap is 0."""
    action_cap = int(
        getattr(
            adapter,
            "_frame_action_max_tokens",
            budget.runtime_output_tokens,
        )
        or 0
    )
    r.update(
        {
            "handoff_token_cap": 0,
            "action_token_cap": action_cap,
            "unclamped_clock_output_tokens": r.get("unclamped_clock_output_tokens"),
            "post_horizon_output_tokens": r.get("post_horizon_output_tokens", 0),
        }
    )


def finalize_decision_audit(adapter, r: dict) -> str | None:
    """Hand the decision's settled result to the adapter's trajectory audit. A failed write makes
    the result unusable: harness:trajectory_write_failed goes on ``r`` over any failure already
    there, and the traceback is returned for the caller to log and latch."""
    finalize = getattr(adapter, "finalize_decision_audit", None)
    if not callable(finalize):
        return None
    try:
        finalize(r)
    except Exception:  # noqa: BLE001 - any write failure invalidates the run
        failure = traceback.format_exc()
        r.update(
            {
                "invalid_category": "harness",
                "invalid_reason": "trajectory_write_failed",
                "invalid_detail": failure,
            }
        )
        return failure
    return None


def result_invalid(r: dict, adapter, clock_mode: str) -> Invalidated | None:
    """Why a generation result cannot be used, or None when it can: first the failure on ``r``
    (the generation's own, or a failed trajectory write recorded over it), then a missing or
    mismatched inspection proof from an adapter with a tool runtime. A GenerationResult carries
    this as ``invalid`` (kernel/events.py)."""
    proof = r.get("inspection")
    inspection_required = getattr(adapter, "tool_runtime", None) is not None
    inspection_ok = not inspection_required or valid_inspection_proof(
        proof,
        r.get("observation_id"),
    )
    if r.get("invalid_reason"):
        return Invalidated(
            r.get("invalid_category") or "provider",
            r["invalid_reason"],
            r.get("invalid_detail"),
        )
    # A harness-aborted wall-clock call has no returned model decision to inspect. Keep
    # charging its elapsed time and recording unknown usage, but never mistake this synthetic
    # timeout for a model response missing its observation-bound inspection proof.
    harness_wall_timeout = (
        clock_mode == "wall"
        and r.get("timed_out")
        and r.get("action") is None
        and r.get("harness_timeout_reason") in {"wall_deadline", "per_move_cap"}
    )
    # Nor has a call given up on at the episode's end (abandoned_result).
    harness_abort = r.get("harness_abort_reason") is not None
    if not inspection_ok and not harness_wall_timeout and not harness_abort:
        return Invalidated(
            "tool",
            "inspection_protocol_error",
            (proof or {}).get("protocol_error") or (proof or {}).get("status"),
        )
    return None


def terminal_wait_protocol(action_protocol: dict) -> dict:
    """The action protocol of a DONE or FAIL that landed as a wait: the loop has no terminal
    actions."""
    return {
        **action_protocol,
        "source": "invalid_action_fallback_wait",
        "valid_action": False,
        "protocol_failure": "unsupported_terminal_action",
    }


def frame_output_tokens(r: dict) -> int | None:
    """The output tokens the legacy loop places the decision's action with: the clock count
    already settled on ``r``, which the allowance has clamped, else the count its sources give
    now. The kernel takes the count before the clamp."""
    value = r.get("clock_output_tokens")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return clock_output_tokens(r)[0]


def observation_fingerprint(observation: Observation) -> tuple:
    """The producer's identity of a frame: its observation id, tick and screenshot digest. The
    same identity repeated while time moves on is a frozen stream."""
    metadata = getattr(observation, "metadata", {}) or {}
    digest = metadata.get("screenshot_sha256")
    if not digest:
        shot = getattr(observation, "screenshot", b"") or b""
        digest = hashlib.sha256(shot).hexdigest()
    return (
        metadata.get("observation_id"),
        getattr(observation, "tick_number", None),
        digest,
    )


# ---- trace payloads ---------------------------------------------------------------------------


def action_record(action: Action | None) -> dict | None:
    """An action as the trace and the lifecycle record keep it."""
    if action is None:
        return None
    return {
        "action_type": action.action_type.value,
        "parameters": action.parameters,
        "coordinate_space": action.coordinate_space,
        "coordinate_transform": action.coordinate_transform,
    }


def observation_captured(observation: Observation, decision: int) -> dict:
    """The ObservationCaptured payload of a captured frame."""
    metadata = getattr(observation, "metadata", {}) or {}
    return {
        "decision": decision,
        "observation_id": metadata.get("observation_id"),
        "tick": observation.tick_number,
        "timestamp": observation.timestamp,
        "screenshot_sha256": metadata.get("screenshot_sha256")
        or hashlib.sha256(observation.screenshot or b"").hexdigest(),
        "capture_latency_s": metadata.get("capture_latency_s"),
        "interactables_status": (metadata.get("interactables") or {})
        .get("scene", {})
        .get("status"),
    }


def inference_started(
    observation: Observation,
    decision: int,
    usage_before: dict | None,
    native_output_limit: int | None,
    budget: DecisionBudget,
) -> dict:
    """The InferenceStarted payload: the frame, the adapter's usage before the generation and the
    decision's output caps."""
    metadata = getattr(observation, "metadata", {}) or {}
    return {
        "decision": decision,
        "observation_id": metadata.get("observation_id"),
        "observation_tick": observation.tick_number,
        "observation_batch": _observation_batch(observation),
        "usage_before": usage_before,
        "runtime_output_token_cap": budget.runtime_output_tokens,
        "provider_output_token_cap": budget.provider_output_tokens,
        "transport_output_token_cap": budget.transport_output_tokens,
        "native_output_limit": native_output_limit,
    }


def output_chunks(r: dict, decision: int, observation_id: str | None) -> list[dict]:
    """One OutputChunkReceived payload per chunk of the result's output; the raw reply is one final
    chunk when the adapter reports no chunks."""
    chunks = r.get("output_chunks")
    if not isinstance(chunks, list):
        chunks = (
            [{"content": r.get("raw"), "final": True}] if r.get("raw") not in {None, ""} else []
        )
    payloads = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_record = chunk if isinstance(chunk, dict) else {"content": str(chunk)}
        payloads.append(
            {
                "decision": decision,
                "observation_id": observation_id,
                "chunk_index": chunk_index,
                "final": bool(chunk_record.get("final", chunk_index == len(chunks) - 1)),
                "content": chunk_record.get("content", ""),
                "channel": chunk_record.get("channel"),
                "output_tokens": chunk_record.get("output_tokens"),
                "native_token_count": chunk_record.get("native_token_count"),
                "function_call": chunk_record.get("function_call"),
                "transport_event": chunk_record.get("transport_event"),
                # The trace event itself is appended after inference completes. These fields are
                # captured synchronously when the collector receives the provider event.
                "received_wall_time_ns": chunk_record.get("received_wall_time_ns"),
                "received_monotonic_ns": chunk_record.get("received_monotonic_ns"),
                "receive_timestamp_source": chunk_record.get(
                    "receive_timestamp_source", "unavailable"
                ),
                "trace_recorded_after_inference": True,
            }
        )
    return payloads


def inference_completed(r: dict, decision: int, observation_id: str | None) -> dict:
    """The InferenceCompleted payload: the result's latencies, usage, token accounting, caps,
    response status and failure flags."""
    return {
        "decision": decision,
        "observation_id": observation_id,
        "latency_s": r.get("latency"),
        "latency_compatibility_scope": (
            "observation_preparation_through_complete_response_processing"
        ),
        "decision_wall_latency_s": r.get("decision_wall_latency_s"),
        "provider_call_wall_latency_s": r.get("provider_call_wall_latency_s"),
        "runtime_turn_wall_latency_s": r.get("runtime_turn_wall_latency_s"),
        "provider_ttft_s": r.get("provider_ttft_s"),
        "provider_ttft_source": r.get("provider_ttft_source"),
        "provider_ttft_missing_reason": r.get("provider_ttft_missing_reason"),
        "latency_clock": "wall_monotonic",
        "usage": r.get("usage"),
        "frame_usage": r.get("frame_usage"),
        "token_accounting": r.get("token_accounting"),
        "provider_output_tokens": r.get("provider_output_tokens"),
        "provider_reasoning_output_tokens": r.get("provider_reasoning_output_tokens"),
        "provider_visible_output_tokens": r.get("provider_visible_output_tokens"),
        "provider_audio_output_tokens": r.get("provider_audio_output_tokens"),
        "provider_token_normalization_schema": r.get("provider_token_normalization_schema"),
        "provider_token_semantics": r.get("provider_token_semantics"),
        "clock_output_tokens": r.get("clock_output_tokens"),
        "clock_output_tokens_source": r.get("clock_output_tokens_source"),
        "remaining_clock_tokens": r.get("remaining_clock_tokens"),
        "unclamped_clock_output_tokens": r.get("unclamped_clock_output_tokens"),
        "post_horizon_output_tokens": r.get("post_horizon_output_tokens", 0),
        "provider_output_token_cap": r.get("provider_output_token_cap"),
        "transport_output_token_cap": r.get("transport_output_token_cap"),
        "provider_response_status": r.get("provider_response_status"),
        "provider_response_status_reason": r.get("provider_response_status_reason"),
        "stale": bool(r.get("stale")),
        "timed_out": bool(r.get("timed_out")),
        "error": bool(r.get("error")),
        "invalid_reason": r.get("invalid_reason"),
        # Only the synthesized record of a call given up on at the episode's end has it (N34).
        **({"harness_abort_reason": reason} if (reason := r.get("harness_abort_reason")) else {}),
    }


def action_proposed(r: dict, decision: int, observation_id: str | None) -> dict:
    """The ActionProposed payload: the result's action with the model's thinking and the action
    protocol it was parsed under."""
    return {
        "decision": decision,
        "observation_id": observation_id,
        "action": action_record(r.get("action")),
        "thinking": r.get("thinking"),
        "protocol": r.get("action_protocol") or {},
        "action_output_tokens": r.get("action_output_tokens"),
    }


def action_resolved(
    *,
    decision: int,
    observation_id: str | None,
    action: Action | None,
    status: str,
    terminal_after_action: bool,
    action_clock_position: float,
    action_resolved_unix_s: float,
    environment_tick: float,
    execution_result,
    coordinate_trace,
) -> dict:
    """The ActionResolved payload, which ActionApplied repeats for an action that landed."""
    return {
        "decision": decision,
        "observation_id": observation_id,
        "action": action_record(action),
        "status": status,
        "terminal_after_action": terminal_after_action,
        "action_clock_position": action_clock_position,
        "action_resolved_unix_s": action_resolved_unix_s,
        "environment_tick": environment_tick,
        # Recorded as the environment returned it: a guest reply's "timestamp" is real time
        # inside the sandbox (sandbox/guest.py execute_action).
        "execution_result": execution_result,
        "coordinate_trace": coordinate_trace,
    }


# ---- the action lifecycle record --------------------------------------------------------------


def lifecycle_record(
    r: dict,
    *,
    decision: int,
    clock_mode: str,
    frame_age_s: float | None,
    action_clock_position: float,
    action_resolved_unix_s: float,
    frame_output_tokens: int | None,
    action_output_tokens: int | None,
    token_delay_frames: float | None,
    next_observation_tick: float,
    post_action_rendered: bool,
    dropped_intermediate_frames: int,
    status: str,
    terminal_after_action: bool,
    action: Action | None,
    execution_result,
    coordinate_trace,
    clock_advance,
    recorded_unix_s: float,
) -> dict:
    """One decision's action lifecycle record: its frame, latencies, token accounting and caps from
    the settled result ``r``, and what became of its action."""
    return {
        "decision": decision,
        "generation_id": (r.get("realtime_runtime") or {}).get("generation_id"),
        "observation_id": r.get("observation_id"),
        "observation_batch": r.get("observation_batch"),
        "observation_tick": r["observation_tick"],
        "observation_timestamp": r.get("observation_timestamp"),
        "frame_age_at_action_s": round(frame_age_s, 6) if frame_age_s is not None else None,
        "frame_age_clock": "simulated_token_clock" if clock_mode == "token" else "wall_clock",
        "frame_age_definition": (
            "action_simulated_time_minus_observation_simulated_time"
            if clock_mode == "token"
            else "action_resolution_wall_time_minus_observation_wall_time"
        ),
        # Compatibility field: this is the complete adapter decision duration, not pure provider
        # compute and not time to first token.
        "model_latency_s": r.get("latency"),
        "harness_timeout_reason": r.get("harness_timeout_reason"),
        "model_latency_scope": "observation_preparation_through_complete_response_processing",
        "decision_wall_latency_s": r.get("decision_wall_latency_s"),
        "provider_call_wall_latency_s": r.get("provider_call_wall_latency_s"),
        "runtime_turn_wall_latency_s": r.get("runtime_turn_wall_latency_s"),
        "provider_ttft_s": r.get("provider_ttft_s"),
        "provider_ttft_source": r.get("provider_ttft_source"),
        "provider_ttft_missing_reason": r.get("provider_ttft_missing_reason"),
        "latency_clock": "wall_monotonic",
        "action_clock_position": action_clock_position,
        "action_resolved_unix_s": action_resolved_unix_s,
        "frame_output_tokens": frame_output_tokens,
        "token_accounting": r.get("token_accounting"),
        "action_output_tokens": action_output_tokens,
        "provider_output_tokens": r.get("provider_output_tokens"),
        "provider_reasoning_output_tokens": r.get("provider_reasoning_output_tokens"),
        "provider_visible_output_tokens": r.get("provider_visible_output_tokens"),
        "provider_audio_output_tokens": r.get("provider_audio_output_tokens"),
        "provider_token_normalization_schema": r.get("provider_token_normalization_schema"),
        "provider_token_semantics": r.get("provider_token_semantics"),
        "clock_output_tokens": r.get("clock_output_tokens"),
        "clock_output_tokens_source": r.get("clock_output_tokens_source"),
        "remaining_clock_tokens": r.get("remaining_clock_tokens"),
        "unclamped_clock_output_tokens": r.get("unclamped_clock_output_tokens"),
        "post_horizon_output_tokens": r.get("post_horizon_output_tokens", 0),
        "provider_response_status": r.get("provider_response_status"),
        "provider_response_status_reason": r.get("provider_response_status_reason"),
        "action_protocol": r.get("action_protocol"),
        "token_accounting_source": r.get("clock_output_tokens_source"),
        "token_delay_frames": token_delay_frames,
        "next_observation_tick": next_observation_tick,
        "post_action_rendered": post_action_rendered,
        "dropped_intermediate_frames": dropped_intermediate_frames,
        "decision_output_cap": r.get("decision_output_cap"),
        "native_output_limit": r.get("native_output_limit"),
        "runtime_output_token_cap": r.get("runtime_output_token_cap"),
        "provider_output_token_cap": r.get("provider_output_token_cap"),
        "transport_output_token_cap": r.get("transport_output_token_cap"),
        "handoff_token_cap": r.get("handoff_token_cap"),
        "action_token_cap": r.get("action_token_cap"),
        "status": status,
        "terminal_after_action": terminal_after_action,
        "action": action_record(action),
        # As in action_resolved: a guest reply's "timestamp" is real time inside the sandbox.
        "execution_result": execution_result,
        "coordinate_trace": coordinate_trace,
        "clock_advance": clock_advance,
        "recorded_unix_s": recorded_unix_s,
    }
