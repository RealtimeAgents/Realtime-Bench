"""The records one decision leaves (evaluation/episode_records.py), called directly rather than
through run_stream: the output caps and their audit, what a result is annotated with, the
output-token source the clock is charged with and the token accounting, whether a result is usable,
the payloads of its trace events and its action lifecycle record."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from realtime_framework.entity.actions import Action, Observation
from realtime_framework.entity.config import DecisionBudget
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.clock import TokenAccounting
from realtime_framework.evaluation.kernel.events import Invalidated

OBSERVATION = Observation(b"png", 1_000.5, 3.0, {"observation_id": "o3"})
BATCH = [{"observation_id": "o3", "tick": 3.0, "timestamp": 1_000.5}]


def _exactly(record: dict, expected: dict) -> None:
    """Equal, and in the same key order, which the lifecycle file keeps
    (GuestEnvironment.record_lifecycle writes each record as built). The episode trace writes
    its payloads with sorted keys, and the replay baselines compare them so."""
    assert record == expected
    assert list(record) == list(expected)


# ---- output caps ------------------------------------------------------------------------------


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


# ---- usage and the result's annotations -------------------------------------------------------


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


def _annotate(r, budget, tokens_per_frame):
    episode_records.annotate_result(
        r,
        decision=3,
        observation=OBSERVATION,
        native_output_limit=128_000,
        budget=budget,
        tokens_per_frame=tokens_per_frame,
    )
    return r


def test_a_result_is_annotated_with_its_observation_caps_and_clock():
    budget = DecisionBudget(2_560, 2_560, "inf", "token")
    r = _annotate({"action": None, "latency": 1.25}, budget, 256)
    _exactly(
        r,
        {
            "action": None,
            "latency": 1.25,
            "decision": 3,
            "observation_id": "o3",
            "observation_tick": 3.0,
            "observation_batch": BATCH,
            "observation_timestamp": 1_000.5,
            "decision_output_cap": 2_560,
            "native_output_limit": 128_000,
            "runtime_output_token_cap": 2_560,
            "provider_output_token_cap": 2_560,
            "transport_output_token_cap": "inf",
            "clock_delay_clock": "simulated_token_clock",
            "clock_delay_definition": "complete_clock_output_tokens_divided_by_tokens_per_frame",
        },
    )


def test_the_clock_definitions_follow_the_clock_mode():
    budget = DecisionBudget(None, None, None, "token")
    r = _annotate({}, budget, None)
    assert (r["clock_delay_clock"], r["clock_delay_definition"]) == (
        "simulated_token_clock",
        "paused_environment_no_generation_delay",
    )
    r = _annotate({}, DecisionBudget(None, None, None, "wall"), None)
    assert (r["clock_delay_clock"], r["clock_delay_definition"]) == (
        "wall_clock",
        "complete_decision_wall_latency",
    )


def test_caps_and_a_response_status_the_adapter_reports_are_kept():
    budget = DecisionBudget(4_096, 4_096, 4_096, "wall")
    runtime = {"provider_response_status": "incomplete", "provider_response_status_reason": "max"}
    r = {
        "provider_output_token_cap": 2_048,
        "transport_output_token_cap": None,
        "provider_response_status_reason": "length",
        "realtime_runtime": runtime,
    }
    _annotate(r, budget, None)
    assert r["provider_output_token_cap"] == 2_048 and r["transport_output_token_cap"] is None
    assert r["provider_response_status"] == "incomplete"
    assert r["provider_response_status_reason"] == "length"


def test_frame_usage_is_the_growth_of_each_numeric_counter():
    before = {"output_tokens": 100, "input_tokens": 50, "model": "m", "cached_tokens": 9}
    after = {"output_tokens": 160, "input_tokens": 40, "model": "m", "reasoning_tokens": 7}
    assert episode_records.frame_usage(before, after) == {
        "output_tokens": 60,
        "input_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 7,
    }


# ---- the clock's output tokens and the token accounting ----------------------------------------


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


class _Env:
    pass


def test_tokens_within_the_allowance_are_charged_whole():
    env, r = _Env(), {"provider_output_tokens": 300, "remaining_clock_tokens": None}
    episode_records.account_tokens(r, 2_560, env)
    assert r["clock_output_tokens"] == r["action_output_tokens"] == 300
    assert r["clock_output_tokens_source"] == "provider_reported_output_usage"
    assert r["remaining_clock_tokens"] == 2_560
    assert r["unclamped_clock_output_tokens"] is None and r["post_horizon_output_tokens"] == 0
    accounting = r["token_accounting"]
    assert (
        accounting["clock_output_tokens"] == 300 and accounting["remaining_clock_tokens"] == 2_560
    )
    assert r["provider_output_tokens"] == accounting["provider_output_tokens"] == 300
    assert "provider_upstream" in r and "provider_token_semantics" in r
    assert env._token_accounting.responses == 1
    assert env._token_accounting.sums["clock_output_tokens"] == 300


def test_output_past_the_allowance_is_billed_but_not_charged():
    env = _Env()
    env._token_accounting = totals = TokenAccounting()
    r = {"native_received_output_tokens": 500, "native_received_output_tokens_source": "ids"}
    episode_records.account_tokens(r, 256, env)
    assert (r["clock_output_tokens"], r["action_output_tokens"]) == (256, 256)
    assert r["clock_output_tokens_source"] == "episode_horizon_clamped:ids"
    assert (r["unclamped_clock_output_tokens"], r["post_horizon_output_tokens"]) == (500, 244)
    assert env._token_accounting is totals and totals.responses == 1
    assert totals.sums["post_horizon_output_tokens"] == 244
    r = {"usage": {"output_tokens": 300}}
    episode_records.account_tokens(r, 0, env)
    assert r["clock_output_tokens"] == 0 and r["post_horizon_output_tokens"] == 300


def test_a_result_without_a_count_charges_nothing():
    r = {"provider_usage_missing_reason": "generation_aborted_at_wall_timeout"}
    episode_records.account_tokens(r, None, _Env())
    assert r["clock_output_tokens"] is None and r["clock_output_tokens_source"] is None
    assert r["token_accounting"]["provider_usage_status"] == "unavailable"
    r = {"native_received_output_tokens": 40}
    episode_records.account_tokens(r, None, _Env())
    assert r["clock_output_tokens"] == 40 and r["post_horizon_output_tokens"] == 0


def test_frame_output_tokens_takes_the_settled_clock_count_first():
    r = {"clock_output_tokens": 256, "provider_output_tokens": 300}
    assert episode_records.frame_output_tokens(r) == 256
    r["clock_output_tokens"] = None
    assert episode_records.frame_output_tokens(r) == 300


# ---- audits and whether the result is usable --------------------------------------------------


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
    assert episode_records.result_invalid(r, _Tools(), "token") == Invalidated(
        "tool", "inspection_protocol_error", None
    )
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


def test_a_frame_is_known_by_its_observation_id_tick_and_screenshot_digest():
    observation = Observation(b"png", 1_000.5, 3.0, {"observation_id": "o3"})
    assert episode_records.observation_fingerprint(observation) == (
        "o3",
        3.0,
        hashlib.sha256(b"png").hexdigest(),
    )
    observation.metadata["screenshot_sha256"] = "digest"
    assert episode_records.observation_fingerprint(observation) == ("o3", 3.0, "digest")
    blank = Observation(b"", 0.0, 0.0, {})
    assert episode_records.observation_fingerprint(blank) == (
        None,
        0.0,
        hashlib.sha256(b"").hexdigest(),
    )


# ---- trace payloads ---------------------------------------------------------------------------

CLICK = Action.click(10, 20)


def test_action_record_keeps_the_type_parameters_and_coordinates():
    _exactly(
        episode_records.action_record(CLICK),
        {
            "action_type": "click",
            "parameters": CLICK.parameters,
            "coordinate_space": CLICK.coordinate_space,
            "coordinate_transform": CLICK.coordinate_transform,
        },
    )
    assert episode_records.action_record(None) is None


def test_observation_captured_names_the_frame_and_its_screenshot():
    metadata = {
        "observation_id": "o3",
        "capture_latency_s": 0.02,
        "interactables": {"scene": {"status": "complete"}},
    }
    observation = Observation(b"png", 1_000.5, 3.0, metadata)
    _exactly(
        episode_records.observation_captured(observation, 3),
        {
            "decision": 3,
            "observation_id": "o3",
            "tick": 3.0,
            "timestamp": 1_000.5,
            "screenshot_sha256": hashlib.sha256(b"png").hexdigest(),
            "capture_latency_s": 0.02,
            "interactables_status": "complete",
        },
    )
    metadata["screenshot_sha256"] = "digest"
    assert episode_records.observation_captured(observation, 3)["screenshot_sha256"] == "digest"
    payload = episode_records.observation_captured(Observation(b"", 0.0, 0.0), 0)
    assert payload["observation_id"] is None and payload["interactables_status"] is None
    assert payload["screenshot_sha256"] == hashlib.sha256(b"").hexdigest()


def test_inference_started_carries_the_frame_the_usage_before_and_the_caps():
    budget = DecisionBudget(2_560, 2_048, "inf", "token")
    _exactly(
        episode_records.inference_started(OBSERVATION, 3, {"output_tokens": 9}, 128_000, budget),
        {
            "decision": 3,
            "observation_id": "o3",
            "observation_tick": 3.0,
            "observation_batch": BATCH,
            "usage_before": {"output_tokens": 9},
            "runtime_output_token_cap": 2_560,
            "provider_output_token_cap": 2_048,
            "transport_output_token_cap": "inf",
            "native_output_limit": 128_000,
        },
    )


def test_each_output_chunk_has_a_payload_and_the_last_is_final_unless_it_says_otherwise():
    chunks = [
        {
            "content": "thinking",
            "channel": "reasoning",
            "final": True,
            "output_tokens": 3,
            "received_wall_time_ns": 5,
            "received_monotonic_ns": 6,
            "receive_timestamp_source": "collector",
        },
        "click(1, 2)",
        {"native_token_count": 4},
    ]
    payloads = episode_records.output_chunks({"output_chunks": chunks, "raw": "x"}, 3, "o3")
    _exactly(
        payloads[0],
        {
            "decision": 3,
            "observation_id": "o3",
            "chunk_index": 0,
            "final": True,
            "content": "thinking",
            "channel": "reasoning",
            "output_tokens": 3,
            "native_token_count": None,
            "function_call": None,
            "transport_event": None,
            "received_wall_time_ns": 5,
            "received_monotonic_ns": 6,
            "receive_timestamp_source": "collector",
            "trace_recorded_after_inference": True,
        },
    )
    assert [(p["chunk_index"], p["content"], p["final"]) for p in payloads] == [
        (0, "thinking", True),
        (1, "click(1, 2)", False),
        (2, "", True),
    ]
    assert payloads[1]["receive_timestamp_source"] == "unavailable"


def test_a_raw_reply_without_chunks_is_one_final_chunk():
    [payload] = episode_records.output_chunks({"raw": "click(1, 2)"}, 0, None)
    assert (payload["chunk_index"], payload["content"], payload["final"]) == (
        0,
        "click(1, 2)",
        True,
    )
    assert episode_records.output_chunks({"raw": ""}, 0, None) == []
    assert episode_records.output_chunks({"output_chunks": None}, 0, None) == []
    assert episode_records.output_chunks({"output_chunks": [], "raw": "x"}, 0, None) == []


def test_inference_completed_reports_the_result_and_its_flags():
    r = {
        "latency": 1.5,
        "provider_ttft_s": 0.25,
        "usage": {"output_tokens": 300},
        "frame_usage": {"output_tokens": 300},
        "token_accounting": {"schema": "realtime_bench.token_accounting.v2"},
        "provider_output_tokens": 300,
        "clock_output_tokens": 256,
        "clock_output_tokens_source": "episode_horizon_clamped:provider_reported_output_usage",
        "remaining_clock_tokens": 256,
        "unclamped_clock_output_tokens": 300,
        "post_horizon_output_tokens": 44,
        "provider_output_token_cap": 256,
        "stale": 1,
        "timed_out": 1,
        "error": 0,
    }
    payload = episode_records.inference_completed(r, 3, "o3")
    _exactly(
        payload,
        {
            "decision": 3,
            "observation_id": "o3",
            "latency_s": 1.5,
            "latency_compatibility_scope": (
                "observation_preparation_through_complete_response_processing"
            ),
            "decision_wall_latency_s": None,
            "provider_call_wall_latency_s": None,
            "runtime_turn_wall_latency_s": None,
            "provider_ttft_s": 0.25,
            "provider_ttft_source": None,
            "provider_ttft_missing_reason": None,
            "latency_clock": "wall_monotonic",
            "usage": {"output_tokens": 300},
            "frame_usage": {"output_tokens": 300},
            "token_accounting": {"schema": "realtime_bench.token_accounting.v2"},
            "provider_output_tokens": 300,
            "provider_reasoning_output_tokens": None,
            "provider_visible_output_tokens": None,
            "provider_audio_output_tokens": None,
            "provider_token_normalization_schema": None,
            "provider_token_semantics": None,
            "clock_output_tokens": 256,
            "clock_output_tokens_source": "episode_horizon_clamped:provider_reported_output_usage",
            "remaining_clock_tokens": 256,
            "unclamped_clock_output_tokens": 300,
            "post_horizon_output_tokens": 44,
            "provider_output_token_cap": 256,
            "transport_output_token_cap": None,
            "provider_response_status": None,
            "provider_response_status_reason": None,
            "stale": True,
            "timed_out": True,
            "error": False,
            "invalid_reason": None,
        },
    )
    assert (payload["stale"], payload["timed_out"], payload["error"]) == (True, True, False)
    assert all(type(payload[flag]) is bool for flag in ("stale", "timed_out", "error"))
    assert episode_records.inference_completed({}, 0, None)["post_horizon_output_tokens"] == 0


def test_action_proposed_keeps_the_thinking_and_the_protocol():
    r = {
        "action": CLICK,
        "thinking": "the button",
        "action_protocol": {"source": "tool_call", "valid_action": True},
        "action_output_tokens": 12,
    }
    _exactly(
        episode_records.action_proposed(r, 3, "o3"),
        {
            "decision": 3,
            "observation_id": "o3",
            "action": episode_records.action_record(CLICK),
            "thinking": "the button",
            "protocol": {"source": "tool_call", "valid_action": True},
            "action_output_tokens": 12,
        },
    )
    r["action_protocol"] = None
    assert episode_records.action_proposed(r, 3, "o3")["protocol"] == {}


def test_action_resolved_records_what_became_of_the_action():
    _exactly(
        episode_records.action_resolved(
            decision=3,
            observation_id="o3",
            action=CLICK,
            status="applied",
            terminal_after_action=False,
            action_clock_position=4.5,
            action_resolved_unix_s=1_004.5,
            environment_tick=4,
            execution_result={"dispatched": True},
            coordinate_trace={"space": "screen"},
        ),
        {
            "decision": 3,
            "observation_id": "o3",
            "action": episode_records.action_record(CLICK),
            "status": "applied",
            "terminal_after_action": False,
            "action_clock_position": 4.5,
            "action_resolved_unix_s": 1_004.5,
            "environment_tick": 4,
            "execution_result": {"dispatched": True},
            "coordinate_trace": {"space": "screen"},
        },
    )
    payload = episode_records.action_resolved(
        decision=0,
        observation_id=None,
        action=None,
        status="none",
        terminal_after_action=False,
        action_clock_position=1.0,
        action_resolved_unix_s=1_001.0,
        environment_tick=1.0,
        execution_result=None,
        coordinate_trace=None,
    )
    assert payload["action"] is None


# ---- the action lifecycle record --------------------------------------------------------------

CLAMPED = "episode_horizon_clamped:provider_reported_output_usage"


def _lifecycle(r, clock_mode="token", **outcome):
    fields = {
        "decision": 3,
        "clock_mode": clock_mode,
        "frame_age_s": 1.23456789,
        "action_clock_position": 4.5,
        "action_resolved_unix_s": 1_004.5,
        "frame_output_tokens": 383,
        "action_output_tokens": 382,
        "token_delay_frames": 1.5,
        "next_observation_tick": 5.0,
        "post_action_rendered": False,
        "dropped_intermediate_frames": 0,
        "status": "applied",
        "terminal_after_action": False,
        "action": CLICK,
        "execution_result": {"dispatched": True},
        "coordinate_trace": {"space": "screen"},
        "clock_advance": {"ticks": 1.5},
        "recorded_unix_s": 1_004.75,
    }
    return episode_records.lifecycle_record(r, **{**fields, **outcome})


def test_the_lifecycle_record_joins_the_settled_result_and_the_outcome():
    r = {
        "realtime_runtime": {"generation_id": "turn-3-0123456789ab"},
        "observation_id": "o3",
        "observation_batch": BATCH,
        "observation_tick": 3.0,
        "observation_timestamp": 1_000.5,
        "latency": 2.5,
        "provider_ttft_s": 0.25,
        "token_accounting": {"schema": "realtime_bench.token_accounting.v2"},
        "provider_output_tokens": 400,
        "clock_output_tokens": 384,
        "clock_output_tokens_source": CLAMPED,
        "remaining_clock_tokens": 384,
        "unclamped_clock_output_tokens": 400,
        "post_horizon_output_tokens": 16,
        "action_protocol": {"source": "tool_call"},
        "decision_output_cap": 2_816,
        "native_output_limit": 128_000,
        "runtime_output_token_cap": 2_560,
        "provider_output_token_cap": 2_048,
        "transport_output_token_cap": 2_304,
        "handoff_token_cap": 256,
        "action_token_cap": 1_792,
    }
    _exactly(
        _lifecycle(r),
        {
            "decision": 3,
            "generation_id": "turn-3-0123456789ab",
            "observation_id": "o3",
            "observation_batch": BATCH,
            "observation_tick": 3.0,
            "observation_timestamp": 1_000.5,
            "frame_age_at_action_s": 1.234568,
            "frame_age_clock": "simulated_token_clock",
            "frame_age_definition": "action_simulated_time_minus_observation_simulated_time",
            "model_latency_s": 2.5,
            "harness_timeout_reason": None,
            "model_latency_scope": "observation_preparation_through_complete_response_processing",
            "decision_wall_latency_s": None,
            "provider_call_wall_latency_s": None,
            "runtime_turn_wall_latency_s": None,
            "provider_ttft_s": 0.25,
            "provider_ttft_source": None,
            "provider_ttft_missing_reason": None,
            "latency_clock": "wall_monotonic",
            "action_clock_position": 4.5,
            "action_resolved_unix_s": 1_004.5,
            "frame_output_tokens": 383,
            "token_accounting": {"schema": "realtime_bench.token_accounting.v2"},
            "action_output_tokens": 382,
            "provider_output_tokens": 400,
            "provider_reasoning_output_tokens": None,
            "provider_visible_output_tokens": None,
            "provider_audio_output_tokens": None,
            "provider_token_normalization_schema": None,
            "provider_token_semantics": None,
            "clock_output_tokens": 384,
            "clock_output_tokens_source": CLAMPED,
            "remaining_clock_tokens": 384,
            "unclamped_clock_output_tokens": 400,
            "post_horizon_output_tokens": 16,
            "provider_response_status": None,
            "provider_response_status_reason": None,
            "action_protocol": {"source": "tool_call"},
            "token_accounting_source": CLAMPED,
            "token_delay_frames": 1.5,
            "next_observation_tick": 5.0,
            "post_action_rendered": False,
            "dropped_intermediate_frames": 0,
            "decision_output_cap": 2_816,
            "native_output_limit": 128_000,
            "runtime_output_token_cap": 2_560,
            "provider_output_token_cap": 2_048,
            "transport_output_token_cap": 2_304,
            "handoff_token_cap": 256,
            "action_token_cap": 1_792,
            "status": "applied",
            "terminal_after_action": False,
            "action": episode_records.action_record(CLICK),
            "execution_result": {"dispatched": True},
            "coordinate_trace": {"space": "screen"},
            "clock_advance": {"ticks": 1.5},
            "recorded_unix_s": 1_004.75,
        },
    )


def test_the_wall_clock_lifecycle_record_keeps_the_frame_age_in_wall_time():
    """On the wall clock the frame age is wall time, kept to the microsecond, and no output
    tokens place the action: only the frame's count is recorded."""
    record = _lifecycle(
        {"observation_tick": 3.0, "observation_timestamp": 1_000.0},
        "wall",
        frame_age_s=0.43750049,
        action=None,
        frame_output_tokens=12,
        action_output_tokens=None,
        token_delay_frames=None,
        clock_advance=None,
        status="skipped_timeout",
    )
    assert (record["frame_age_clock"], record["frame_age_definition"]) == (
        "wall_clock",
        "action_resolution_wall_time_minus_observation_wall_time",
    )
    assert record["frame_age_at_action_s"] == 0.4375
    assert (record["frame_output_tokens"], record["action_output_tokens"]) == (12, None)
    assert record["action"] is None and record["token_delay_frames"] is None
    assert record["generation_id"] is None and record["post_horizon_output_tokens"] == 0
