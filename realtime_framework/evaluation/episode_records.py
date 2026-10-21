"""What one decision leaves on the record: the output tokens its clock is charged with, whether
its result is usable, and the action as the trace keeps it.

run_stream's turn loop builds its records with these, and the kernel's shell is to build the same
records with them. Nothing here paces the episode, orders the skips or waits: each function takes
what its caller measured, so the pacing on the two sides stays separate code and running them side
by side still compares it (phase2.md, 怎么验证).
"""

from __future__ import annotations

import traceback

from realtime_framework.entity.actions import Action
from realtime_framework.evaluation.kernel.events import Invalidated
from realtime_framework.sandbox.observation.som import valid_inspection_proof


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
