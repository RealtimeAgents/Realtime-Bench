"""What one decision leaves on the record: the output tokens its clock is charged with, and the
action as the trace keeps it.

run_stream's turn loop builds its records with these, and the kernel's shell is to build the same
records with them. Nothing here paces the episode, orders the skips or waits: each function takes
what its caller measured, so the pacing on the two sides stays separate code and running them side
by side still compares it (phase2.md, 怎么验证).
"""

from __future__ import annotations

from realtime_framework.entity.actions import Action


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
