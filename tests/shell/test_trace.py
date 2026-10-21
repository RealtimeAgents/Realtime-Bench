"""The episode trace run_episode writes from the kernel's records (evaluation/shell.py, D6).

A record under a TraceEvent name keeps it, with the payload the runner writes today; the echo of
a captured frame is ObservationCaptured, that of a generation's result OutputChunkReceived and
InferenceCompleted; every other record is a KernelRecord, the kernel's echo of SnapshotFrozen
too, so the only SnapshotFrozen is the one main() writes. Each record carries the kernel's
sequence number and readings.
"""

from __future__ import annotations

from realtime_framework.entity.actions import Action
from realtime_framework.entity.records import validate_episode_trace
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.reporting import EpisodeTrace

from .world import CLICK, ORIGIN, Reply, Trace, World, run

ONE_DECISION = [
    Reply(0.5, CLICK, fields={"thinking": "click it", "action_protocol": {"source": "json"}})
]


def _kernel_kinds(trace: Trace) -> list[str]:
    return [r["payload"]["kind"] for r in trace.records if r["event"] == "KernelRecord"]


def test_kernel_records_keep_the_trace_event_names_with_the_runners_payloads():
    episode = run("wall", 1.0, ONE_DECISION)
    trace = episode.trace
    assert [event for event in trace.events() if event != "KernelRecord"] == [
        "ObservationCaptured",
        "InferenceStarted",
        "OutputChunkReceived",
        "InferenceCompleted",
        "ActionProposed",
        "ClockAdvanced",
        "ActionResolved",
        "ActionApplied",
        "ClockAdvanced",
    ]
    observation = episode.agent.generated[0]
    (captured,) = trace.payloads("ObservationCaptured")
    assert captured == {
        "decision": 0,
        "observation_id": "obs-0",
        "tick": 0,
        "timestamp": ORIGIN,
        "screenshot_sha256": captured["screenshot_sha256"],
        "capture_latency_s": None,
        "interactables_status": None,
    }
    (started,) = trace.payloads("InferenceStarted")
    assert started == {
        "decision": 0,
        "observation_id": "obs-0",
        "observation_tick": 0,
        "observation_batch": [{"observation_id": "obs-0", "tick": 0, "timestamp": ORIGIN}],
        "usage_before": None,
        "runtime_output_token_cap": None,
        "provider_output_token_cap": None,
        "transport_output_token_cap": None,
        "native_output_limit": None,
        "generation": 1,
        "remaining_clock_tokens": None,
    }
    assert observation[0] == started["generation"]
    (chunk,) = trace.payloads("OutputChunkReceived")
    assert (chunk["decision"], chunk["content"], chunk["final"]) == (0, "a reply", True)
    (completed,) = trace.payloads("InferenceCompleted")
    assert (completed["decision"], completed["latency_s"], completed["stale"]) == (0, 0.5, False)
    (proposed,) = trace.payloads("ActionProposed")
    assert proposed == {
        "decision": 0,
        "observation_id": "obs-0",
        "action": episode_records.action_record(CLICK),
        "thinking": "click it",
        "protocol": {"source": "json"},
        "action_output_tokens": None,
    }
    (resolved,) = trace.payloads("ActionResolved")
    assert resolved == {
        "decision": 0,
        "observation_id": "obs-0",
        "action": episode_records.action_record(CLICK),
        "status": "applied",
        "terminal_after_action": False,
        "action_clock_position": 0.5,
        "action_resolved_unix_s": ORIGIN + 0.5,
        "environment_tick": 0,
        "execution_result": {"executed": 0},
        "coordinate_trace": {"decision": 0},
    }
    assert trace.payloads("ActionApplied") == [resolved]
    lifecycle = episode.env.lifecycle[0]
    assert lifecycle["action_resolved_unix_s"] == resolved["action_resolved_unix_s"]
    assert lifecycle["coordinate_trace"] == {"decision": 0}
    assert [p["phase"] for p in trace.payloads("ClockAdvanced")] == [
        "inference_to_action",
        "action_to_observation_boundary",
    ]


def test_every_record_carries_the_kernels_sequence_and_readings():
    episode = run("token", 2.0, [Reply(0.25, CLICK, output_tokens=50)] * 2, tokens_per_frame=100)
    records = episode.trace.records
    sequence = [record["kernel_seq"] for record in records]
    assert sequence == sorted(sequence) and sequence[0] == 1
    # Both echoes of one generation's result carry its record's number.
    finished = [r for r in records if r["event"] in {"OutputChunkReceived", "InferenceCompleted"}]
    assert len({r["kernel_seq"] for r in finished}) == 2
    assert all(isinstance(record["ops"], float) for record in records)
    # The first decision lands at tick 0.5 and moves on to the next frame; the token clock's
    # operational reading is the 0.25 s the first generation took.
    started = [r for r in records if r["event"] == "InferenceStarted"]
    assert [(r["semantic"], r["ops"]) for r in started] == [(0, 0.0), (1, 0.25)]


def test_the_only_snapshot_frozen_record_is_main_s():
    """reporting.active_interval ends the active interval at the first SnapshotFrozen in the
    trace, which main() writes after the loop; the kernel's echo of the freeze is a KernelRecord."""
    episode = run("wall", 1.0, ONE_DECISION)
    assert "SnapshotFrozen" not in episode.trace.events()
    assert "SnapshotFrozen" in _kernel_kinds(episode.trace)
    assert _kernel_kinds(episode.trace)[-1] == "Termination"


def test_an_invalidation_without_a_detail_is_written_as_the_runner_writes_it():
    episode = run("wall", 1.0, [Reply(0.5, fields={"invalid_reason": "history_sync_failed"})])
    assert episode.trace.payloads("RunInvalidated") == [
        {"category": "provider", "reason": "history_sync_failed"}
    ]
    assert episode.invalid == {"category": "provider", "reason": "history_sync_failed"}
    assert "[invalid-run] provider:history_sync_failed" in episode.logs


def test_the_records_go_into_an_episode_trace_file(tmp_path):
    """Every payload, the kernel's own records and echoes included, is written as JSON and the
    trace's hash chain holds."""
    path = tmp_path / "episode_trace.jsonl"
    with EpisodeTrace(path, run_id="shell") as trace:
        replies = [Reply(0.5, Action.hold_key("a", 0.25)), Reply(0.5, Action.wait(0.25))]
        episode = run("wall", 2.0, replies, World(), trace=trace)
    verified = validate_episode_trace(path)
    assert verified["event_count"] > 0
    assert episode.statuses() == ["applied", "wait"]


def test_a_failed_trace_write_invalidates_the_run():
    episode = run("wall", 3.0, [Reply(0.5), Reply(0.5)], trace=Trace(fail_on="ActionProposed"))
    assert episode.invalid["category"] == "harness"
    assert episode.invalid["reason"] == "episode_trace_write_failed"
    assert "OSError: trace disk full" in episode.invalid["detail"]
    assert episode.env._stop_reason == "invalid_run"
    assert "ActionApplied" not in episode.trace.events()  # nothing is written after it


def test_a_failed_trace_checkpoint_invalidates_the_run():
    episode = run("wall", 3.0, [Reply(0.5), Reply(0.5)], trace=Trace(fail_checkpoint=True))
    assert episode.invalid["reason"] == "episode_trace_checkpoint_failed"
    assert episode.statuses() == ["applied"]


def test_without_a_trace_the_episode_runs_the_same():
    with_trace = run("wall", 2.0, [Reply(0.5), Reply(0.5)])
    without = run("wall", 2.0, [Reply(0.5), Reply(0.5)], trace=None)
    assert without.statuses() == with_trace.statuses()
    assert without.env.lifecycle == with_trace.env.lifecycle
