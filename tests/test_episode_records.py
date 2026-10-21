"""The records one decision leaves (evaluation/episode_records.py), called directly rather than
through run_stream: the output caps and their audit, the output-token source the clock is charged
with, whether a result is usable and the action as the trace keeps it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realtime_framework.entity.actions import Action
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.kernel.events import Invalidated


def _adapter(native_output_limit=None, **attributes):
    return SimpleNamespace(
        config=SimpleNamespace(native_output_limit=native_output_limit), **attributes
    )


def test_the_runtime_cap_is_the_native_limit_bounded_by_the_remaining_clock():
    budget = episode_records.decision_budget(_adapter(128_000), None, "wall")
    assert budget == (128_000, DecisionBudget(128_000, 128_000, 128_000, "wall"))
    budget = episode_records.decision_budget(_adapter(128_000), 2_560, "token")
    assert budget == (128_000, DecisionBudget(2_560, 2_560, 2_560, "token"))
    budget = episode_records.decision_budget(_adapter(), 2_560, "token")
    assert budget == (None, DecisionBudget(2_560, 2_560, 2_560, "token"))
    budget = episode_records.decision_budget(_adapter(), None, "token")
    assert budget == (None, DecisionBudget(None, None, None, "token"))


def test_unbounded_output_drops_the_provider_cap_only_without_a_token_clock():
    adapter = _adapter(65_536, supports_unbounded_output=True)
    _, budget = episode_records.decision_budget(adapter, None, "wall")
    assert budget == DecisionBudget(65_536, None, None, "wall")
    _, budget = episode_records.decision_budget(adapter, 1_024, "token")
    assert budget == DecisionBudget(1_024, 1_024, 1_024, "token")


def test_the_adapter_maps_the_provider_cap_onto_its_transport():
    adapter = _adapter(4_096, transport_output_token_cap=lambda cap: "inf")
    _, budget = episode_records.decision_budget(adapter, None, "wall")
    assert budget == DecisionBudget(4_096, 4_096, "inf", "wall")
    adapter.transport_output_token_cap = lambda cap: cap * 2
    _, budget = episode_records.decision_budget(adapter, 100, "token")
    assert budget == DecisionBudget(100, 100, 200, "token")


@pytest.mark.parametrize("mapped", [0, -1, True, 1.5, "unbounded"])
def test_a_transport_cap_that_is_not_a_positive_integer_or_inf_is_refused(mapped):
    adapter = _adapter(4_096, transport_output_token_cap=lambda cap: mapped)
    with pytest.raises(ValueError, match="transport output cap"):
        episode_records.decision_budget(adapter, None, "wall")


def test_the_cap_audit_records_the_handoff_and_action_caps():
    budget = DecisionBudget(1_000, 1_000, 1_000, "token")
    adapter = SimpleNamespace(_frame_handoff_max_tokens=200, _frame_action_max_tokens=800)
    r = {"unclamped_clock_output_tokens": 1_200, "post_horizon_output_tokens": 200}
    episode_records.audit_output_caps(adapter, r, budget)
    assert r == {
        "unclamped_clock_output_tokens": 1_200,
        "post_horizon_output_tokens": 200,
        "handoff_token_cap": 200,
        "action_token_cap": 800,
    }
    r = {}
    episode_records.audit_output_caps(SimpleNamespace(), r, budget)
    assert r == {
        "handoff_token_cap": 0,
        "action_token_cap": 1_000,
        "unclamped_clock_output_tokens": None,
        "post_horizon_output_tokens": 0,
    }


def test_caps_past_the_provider_allowance_fail_the_audit_of_a_usable_result():
    budget = DecisionBudget(1_000, 1_000, 1_000, "token")
    adapter = SimpleNamespace(_frame_handoff_max_tokens=201, _frame_action_max_tokens=800)
    with pytest.raises(RuntimeError, match=r"handoff=201 \+ action=800 > 1000"):
        episode_records.audit_output_caps(adapter, {}, budget)
    r = {"invalid_reason": "inspection_protocol_error"}
    episode_records.audit_output_caps(adapter, r, budget)
    assert (r["handoff_token_cap"], r["action_token_cap"]) == (201, 800)
    episode_records.audit_output_caps(adapter, {}, DecisionBudget(1_000, None, None, "wall"))


def test_action_record_keeps_the_type_parameters_and_coordinates():
    action = Action.click(10, 20)
    assert episode_records.action_record(action) == {
        "action_type": "click",
        "parameters": action.parameters,
        "coordinate_space": action.coordinate_space,
        "coordinate_transform": action.coordinate_transform,
    }
    assert episode_records.action_record(None) is None


class _Usage:
    def __init__(self, token_usage):
        self.token_usage = token_usage


def test_usage_snapshot_copies_a_mapping_or_reports_none():
    usage = {"output_tokens": 5}
    snapshot = episode_records.usage_snapshot(_Usage(usage))
    assert snapshot == usage and snapshot is not usage
    assert episode_records.usage_snapshot(_Usage(lambda: {"output_tokens": 7})) == {
        "output_tokens": 7
    }
    assert episode_records.usage_snapshot(_Usage([("output_tokens", 5)])) is None
    assert episode_records.usage_snapshot(object()) is None

    def broken():
        raise RuntimeError("usage unavailable")

    assert episode_records.usage_snapshot(_Usage(broken)) is None


def test_clock_output_tokens_prefers_the_response_usage():
    r = {
        "provider_output_tokens": 40,
        "usage": {"output_tokens": 30},
        "frame_usage": {"output_tokens": 20},
        "native_received_output_tokens": 10,
    }
    assert episode_records.clock_output_tokens(r) == (40, "provider_reported_output_usage")
    del r["provider_output_tokens"]
    assert episode_records.clock_output_tokens(r) == (30, "provider_reported_output_usage")
    r["usage"] = {"completion_tokens": 31}
    assert episode_records.clock_output_tokens(r) == (31, "provider_reported_output_usage")
    del r["usage"]
    assert episode_records.clock_output_tokens(r) == (20, "provider_reported_output_usage_delta")
    del r["frame_usage"]
    assert episode_records.clock_output_tokens(r) == (10, "native_runtime_output_tokens")
    r["native_received_output_tokens_source"] = "local_token_ids"
    assert episode_records.clock_output_tokens(r) == (10, "local_token_ids")
    del r["native_received_output_tokens"]
    assert episode_records.clock_output_tokens(r) == (None, None)


def test_clock_output_tokens_skips_provider_counts_the_provider_disowns():
    r = {
        "provider_usage_missing_reason": "generation_aborted_at_wall_timeout",
        "provider_output_tokens": 40,
        "usage": {"output_tokens": 30},
        "native_received_output_tokens": 10,
    }
    assert episode_records.clock_output_tokens(r) == (10, "native_runtime_output_tokens")
    del r["native_received_output_tokens"]
    assert episode_records.clock_output_tokens(r) == (None, None)


def test_clock_output_tokens_takes_only_non_negative_numbers():
    r = {
        "provider_output_tokens": True,
        "usage": {"output_tokens": -1, "completion_tokens": "12"},
        "frame_usage": {"output_tokens": 7.0},
    }
    assert episode_records.clock_output_tokens(r) == (7, "provider_reported_output_usage_delta")


def test_frame_output_tokens_takes_the_settled_clock_count_first():
    r = {"clock_output_tokens": 256, "provider_output_tokens": 300}
    assert episode_records.frame_output_tokens(r) == 256
    r["clock_output_tokens"] = None
    assert episode_records.frame_output_tokens(r) == 300


class _Audited:
    def __init__(self, error=None):
        self.error = error
        self.results = []

    def finalize_decision_audit(self, result):
        self.results.append(result)
        if self.error is not None:
            raise self.error


def test_a_written_trajectory_leaves_the_result_as_it_was():
    adapter, r = _Audited(), {"action": None}
    assert episode_records.finalize_decision_audit(adapter, r) is None
    assert adapter.results == [r] and r == {"action": None}
    assert episode_records.finalize_decision_audit(object(), r) is None


def test_a_failed_trajectory_write_replaces_the_result_failure():
    r = {
        "invalid_category": "provider",
        "invalid_reason": "generation_error",
        "invalid_detail": "x",
    }
    failure = episode_records.finalize_decision_audit(_Audited(OSError("disk full")), r)
    assert failure.startswith("Traceback") and failure.rstrip().endswith("OSError: disk full")
    assert r == {
        "invalid_category": "harness",
        "invalid_reason": "trajectory_write_failed",
        "invalid_detail": failure,
    }


class _Tools:
    tool_runtime = object()


PROOF = {"valid": True, "observation_id": "o1", "requested_observation_id": "o1"}


def test_the_result_failure_comes_before_the_inspection_proof():
    r = {"invalid_reason": "generation_error", "invalid_detail": "boom", "observation_id": "o1"}
    assert episode_records.result_invalid(r, _Tools(), "token") == Invalidated(
        "provider", "generation_error", "boom"
    )
    r["invalid_category"] = "harness"
    assert episode_records.result_invalid(r, object(), "wall") == Invalidated(
        "harness", "generation_error", "boom"
    )


def test_an_adapter_with_tools_must_prove_it_inspected_the_observation():
    r = {"observation_id": "o1", "inspection": PROOF}
    assert episode_records.result_invalid(r, _Tools(), "token") is None
    r["inspection"] = {**PROOF, "requested_observation_id": "o0"}
    assert episode_records.result_invalid(r, object(), "token") is None
    r["inspection"] = {"valid": False, "status": "stale", "protocol_error": "wrong frame"}
    assert episode_records.result_invalid(r, _Tools(), "token") == Invalidated(
        "tool", "inspection_protocol_error", "wrong frame"
    )
    r["inspection"] = {"valid": False, "status": "stale"}
    assert episode_records.result_invalid(r, _Tools(), "token") == Invalidated(
        "tool", "inspection_protocol_error", "stale"
    )
    del r["inspection"]
    assert episode_records.result_invalid(r, _Tools(), "token") == Invalidated(
        "tool", "inspection_protocol_error", None
    )


@pytest.mark.parametrize("reason", ["wall_deadline", "per_move_cap"])
def test_a_wall_call_the_harness_aborted_needs_no_proof(reason):
    r = {"action": None, "timed_out": True, "harness_timeout_reason": reason}
    assert episode_records.result_invalid(r, _Tools(), "wall") is None
    assert episode_records.result_invalid(r, _Tools(), "token") == Invalidated(
        "tool", "inspection_protocol_error", None
    )
    r["action"] = Action.wait()
    assert episode_records.result_invalid(r, _Tools(), "wall") is not None
