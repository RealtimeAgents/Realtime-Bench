"""What one decision leaves on the record: its output caps and their audit, the output tokens its
clock is charged with, whether its result is usable, and the action as the trace keeps it.

run_stream's turn loop builds its records with these, and the kernel's shell is to build the same
records with them. Nothing here paces the episode, orders the skips or waits: each function takes
what its caller measured, so the pacing on the two sides stays separate code and running them side
by side still compares it (phase2.md, 怎么验证).
"""

from __future__ import annotations

import traceback

from realtime_framework.entity.actions import Action
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.evaluation.kernel.events import Invalidated
from realtime_framework.sandbox.observation.som import valid_inspection_proof


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


def audit_output_caps(adapter, r: dict, budget: DecisionBudget) -> None:
    """Record the adapter's handoff and action caps for the decision on ``r``. Raises RuntimeError
    when together they exceed the provider allowance."""
    handoff_cap = int(getattr(adapter, "_frame_handoff_max_tokens", 0) or 0)
    action_cap = int(
        getattr(
            adapter,
            "_frame_action_max_tokens",
            budget.runtime_output_tokens,
        )
        or 0
    )
    # A decision that was invalidated before generation (e.g. inspection_protocol_error)
    # never recomputed the adapter's per-decision caps, so they still describe the previous
    # decision; auditing them against this decision's allowance would raise spuriously.
    if (
        budget.provider_output_tokens is not None
        and r.get("invalid_reason") is None
        and handoff_cap + action_cap > budget.provider_output_tokens
    ):
        raise RuntimeError(
            f"adapter decision output caps exceed provider transport allowance: "
            f"handoff={handoff_cap} + action={action_cap} > {budget.provider_output_tokens}"
        )
    r.update(
        {
            "handoff_token_cap": handoff_cap,
            "action_token_cap": action_cap,
            "unclamped_clock_output_tokens": r.get("unclamped_clock_output_tokens"),
            "post_horizon_output_tokens": r.get("post_horizon_output_tokens", 0),
        }
    )


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


def usage_snapshot(adapter) -> dict | None:
    """The adapter's cumulative token usage, or None when it reports none or reading it fails."""
    try:
        value = getattr(adapter, "token_usage", None)
        value = value() if callable(value) else value
        return dict(value) if isinstance(value, dict) else None
    except Exception:  # noqa: BLE001 - usage that cannot be read is usage not reported
        return None


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
        value = _usage_number(r.get("frame_usage"), "output_tokens", "completion_tokens")
        if value is not None:
            return value, "provider_reported_output_usage_delta"
    value = r.get("native_received_output_tokens")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return int(value), (
            r.get("native_received_output_tokens_source") or "native_runtime_output_tokens"
        )
    return None, None


def frame_output_tokens(r: dict) -> int | None:
    """The output tokens that place the decision's action: the clock count already settled on
    ``r``, else the count its sources give now."""
    value = r.get("clock_output_tokens")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return clock_output_tokens(r)[0]


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
    if not inspection_ok and not harness_wall_timeout:
        return Invalidated(
            "tool",
            "inspection_protocol_error",
            (proof or {}).get("protocol_error") or (proof or {}).get("status"),
        )
    return None
