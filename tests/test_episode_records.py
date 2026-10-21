"""The records one decision leaves (evaluation/episode_records.py), called directly rather than
through run_stream: the output-token source the clock is charged with and the action as the trace
keeps it."""

from __future__ import annotations

from realtime_framework.entity.actions import Action
from realtime_framework.evaluation import episode_records


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
