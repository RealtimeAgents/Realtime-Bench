"""The kernel numbers every record in handling order, discards stale generation results, and on the
first end signal fixes the cause, freezes the snapshot and interrupts the generation before it
writes the termination record. A stand-in policy drives it."""

from __future__ import annotations

import threading

import pytest

from realtime_framework.evaluation.kernel.core import Kernel
from realtime_framework.evaluation.kernel.events import (
    AbortGeneration,
    AcceptResult,
    AgentRecord,
    Clock,
    DiscardResult,
    EndChecked,
    Finish,
    FreezeSnapshot,
    GenerationFinished,
    GenerationResult,
    Invalidated,
    Priority,
    Record,
    ReleaseInputs,
    SetTimer,
    SnapshotFrozen,
    StartGeneration,
)


class Policy:
    """Records every callback; ``settled`` is set by the test."""

    def __init__(self, merged_clocks: bool = True) -> None:
        self.merged_clocks = merged_clocks
        self.settled = True
        self.totals: dict = {}
        self.calls: list = []
        self.kernel = None

    def start(self, kernel) -> None:
        self.kernel = kernel
        self.calls.append("start")

    def on_event(self, event) -> None:
        self.calls.append(("event", event))

    def on_result(self, result) -> None:
        self.calls.append(("result", result))

    def on_timer(self, handle, timer) -> None:
        self.calls.append(("timer", timer.kind))

    def on_end(self) -> None:
        self.calls.append(("end", self.kernel.ending.cause))


def _kernel(merged_clocks: bool = True) -> tuple[Kernel, Policy]:
    policy = Policy(merged_clocks)
    kernel = Kernel(policy)
    kernel.start(0.0)
    return kernel, policy


def _records(commands) -> list:
    return [command.entry for command in commands if isinstance(command, Record)]


def _kinds(commands) -> list[str]:
    return [entry.kind for entry in _records(commands)]


def _others(commands) -> list:
    return [command for command in commands if not isinstance(command, Record)]


def test_step_returns_none_when_nothing_is_due():
    kernel, _ = _kernel()
    kernel.set_timer("next_frame", Clock.SEMANTIC, 2.0, Priority.START)
    assert kernel.step(1.0) is None


def test_posted_events_are_stamped_and_numbered_when_handled():
    kernel, policy = _kernel(merged_clocks=False)
    event = EndChecked(1)
    kernel.post(event)
    commands = kernel.step(4.0, 17.5)
    (entry,) = _records(commands)
    assert (entry.seq, entry.semantic, entry.ops, entry.kind, entry.data) == (
        1,
        4.0,
        17.5,
        "EndChecked",
        event,
    )
    assert policy.calls[-1] == ("event", event)


def test_sequence_follows_handling_order_not_posting_order():
    kernel, _ = _kernel()
    kernel.post(AgentRecord("ModelRequestSent"))
    kernel.post(Invalidated("environment", "wall_key_release_failed"))
    seen = []
    while (commands := kernel.step(1.0)) is not None:
        seen += _records(commands)
    # The control event is handled first, so it gets the first number.
    assert [entry.seq for entry in seen] == list(range(1, len(seen) + 1))
    assert seen[0].kind == "Invalidated"
    assert "ModelRequestSent" in [entry.kind for entry in seen]


def test_events_posted_from_many_threads_get_one_contiguous_sequence():
    kernel, _ = _kernel()

    def post(name):
        for index in range(200):
            kernel.post(AgentRecord("SessionMessage", {"thread": name, "index": index}))

    workers = [threading.Thread(target=post, args=(name,)) for name in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    entries = []
    while (commands := kernel.step(0.0)) is not None:
        entries += _records(commands)
    assert [entry.seq for entry in entries] == list(range(1, 801))
    for name in range(4):
        indices = [e.data.data["index"] for e in entries if e.data.data["thread"] == name]
        assert indices == list(range(200))


def test_agent_records_are_numbered_but_not_handed_to_the_policy():
    kernel, policy = _kernel()
    kernel.post(AgentRecord("SessionMessage", {"kind": "tool_ack"}, generation=1))
    commands = kernel.step(0.0)
    assert _kinds(commands) == ["SessionMessage"]
    assert _records(commands)[0].data.generation == 1
    assert policy.calls == ["start"]


def test_a_timer_records_its_intent_before_asking_for_the_wake_up():
    kernel, policy = _kernel()
    handle = kernel.set_timer("next_frame", Clock.SEMANTIC, 3.0, Priority.START)
    commands = kernel._flush()
    assert _kinds(commands) == ["TimerSet"]
    assert _records(commands)[0].data == {
        "timer": handle,
        "kind": "next_frame",
        "clock": "semantic",
        "at": 3.0,
    }
    assert _others(commands) == [SetTimer(Clock.SEMANTIC, 3.0)]
    fired = kernel.step(3.0)
    assert _kinds(fired) == ["TimerFired"]
    assert policy.calls[-1] == ("timer", "next_frame")


def test_the_result_of_the_generation_in_flight_is_accepted():
    kernel, policy = _kernel()
    generation = kernel.start_generation(0, "obs-0", 0, None)
    commands = kernel._flush()
    assert _kinds(commands) == ["InferenceStarted"]
    assert _others(commands) == [StartGeneration(generation, 0, "obs-0", None)]
    result = GenerationResult(output_tokens=5)
    kernel.post(GenerationFinished(generation, result))
    commands = kernel.step(1.0)
    assert _others(commands) == [AcceptResult(generation)]
    assert policy.calls[-1] == ("result", result)
    assert kernel.generation is None


def test_an_abandoned_generation_is_written_off_before_it_is_interrupted():
    kernel, policy = _kernel()
    generation = kernel.start_generation(0, "obs-0", 0, None)
    kernel._flush()
    kernel.abandon_generation("per_move_cap")
    commands = kernel._flush()
    assert isinstance(commands[0], Record)
    assert commands[0].entry.kind == "GenerationAbandoned"
    assert commands[0].entry.data == {"generation": generation, "reason": "per_move_cap"}
    assert commands[1:] == [AbortGeneration(generation)]
    # Its late result is stale: discarded, never given to the policy.
    kernel.post(GenerationFinished(generation, GenerationResult(output_tokens=9)))
    commands = kernel.step(5.0)
    assert _kinds(commands) == ["GenerationFinished", "ResultDiscarded"]
    assert _others(commands) == [DiscardResult(generation)]
    assert all(call[0] != "result" for call in policy.calls[1:])


def test_a_result_for_an_older_generation_is_stale():
    kernel, _ = _kernel()
    first = kernel.start_generation(0, "obs-0", 0, None)
    kernel.abandon_generation("per_move_cap")
    second = kernel.start_generation(1, "obs-1", 1, None)
    kernel._flush()
    kernel.post(GenerationFinished(first, GenerationResult()))
    kernel.post(GenerationFinished(second, GenerationResult()))
    assert _others(kernel.step(2.0)) == [DiscardResult(first)]
    assert _others(kernel.step(2.0)) == [AcceptResult(second)]


def test_the_first_end_freezes_the_snapshot_then_interrupts_the_generation():
    kernel, policy = _kernel()
    generation = kernel.start_generation(0, "obs-0", 0, None)
    kernel.set_timer("next_frame", Clock.SEMANTIC, 9.0, Priority.START)
    kernel._flush()
    kernel.end("deadline", 300.0)
    commands = kernel._flush()
    assert _kinds(commands) == ["EpisodeEnded", "GenerationAbandoned"]
    assert _records(commands)[0].data == {"cause": "deadline", "reason": None, "at": 300.0}
    assert _others(commands) == [FreezeSnapshot(), AbortGeneration(generation)]
    kinds = [type(c).__name__ if not isinstance(c, Record) else c.entry.kind for c in commands]
    assert kinds == ["EpisodeEnded", "FreezeSnapshot", "GenerationAbandoned", "AbortGeneration"]
    assert policy.calls[-1] == ("end", "deadline")
    # Timers stop with the episode.
    assert kernel.step(10.0) is None


def test_later_end_signals_change_nothing():
    kernel, policy = _kernel()
    kernel.end("environment", 12.0, reason="target_reached")
    kernel._flush()
    kernel.end("deadline", 300.0)
    assert kernel._flush() == []
    assert kernel.ending.cause == "environment"
    assert policy.calls.count(("end", "environment")) == 1


def test_the_first_invalidation_is_latched_and_ends_the_episode():
    kernel, _ = _kernel()
    # A long detail keeps its end, where a traceback names the exception (legacy_loop.py
    # _invalidate).
    traceback = "Traceback (most recent call last):\n" + "x" * 5000 + "\nRuntimeError: boom"
    kernel.invalidate("environment", "clock_advance_failed", traceback)
    kernel.invalidate("provider", "generation_error")
    commands = kernel._flush()
    assert _kinds(commands) == ["RunInvalidated", "EpisodeEnded"]
    data = _records(commands)[0].data
    assert (data["category"], data["reason"], data["detail"]) == (
        "environment",
        "clock_advance_failed",
        traceback[-2000:],
    )
    assert data["detail"].endswith("\nRuntimeError: boom")
    assert kernel.invalid.reason == "clock_advance_failed"
    assert kernel.ending.cause == "invalid"


def _finish(kernel, now: float) -> list:
    kernel.post(SnapshotFrozen(now, now))
    return kernel.step(now)


@pytest.mark.parametrize(
    ("cause", "reason", "stop_reason"),
    [
        ("deadline", None, "time_limit"),
        ("horizon", None, "tick_limit"),
        ("think_budget", None, "think_budget"),
        ("environment", "task_success", "task_success"),
        ("environment", None, "game_decided"),
    ],
    ids=["deadline", "horizon", "think-budget", "task-reason", "game-decided"],
)
def test_the_stop_reason_comes_from_the_first_end(cause, reason, stop_reason):
    kernel, policy = _kernel()
    policy.totals = {"think_spent": 2.5, "dropped_intermediate_frames": 3}
    kernel.end(cause, 30.0, reason=reason)
    kernel._flush()
    commands = _finish(kernel, 30.5)
    assert kernel.termination.stop_reason == stop_reason
    assert kernel.termination.cause == cause
    # Finish carries what the policy counted beside the termination.
    assert _others(commands)[-2:] == [ReleaseInputs(), Finish(kernel.termination, 2.5, 3)]
    assert _kinds(commands)[-1] == "Termination"


def test_termination_records_the_rule_time_and_the_snapshot_time():
    kernel, _ = _kernel(merged_clocks=False)
    kernel.step(12.0, 40.0)
    kernel.end("environment", 11.0, reason="target_reached")
    kernel._flush()
    # The shell took the snapshot at 41.5 s and ran a commit before the kernel stepped again.
    kernel.post(SnapshotFrozen(12.0, 41.5))
    kernel.step(12.0, 41.75)
    termination = kernel.termination
    assert (termination.ended_at, termination.snapshot_at) == (11.0, 12.0)
    assert (termination.ops_ended_at, termination.ops_snapshot_at) == (40.0, 41.5)
    assert termination.invalid is None


@pytest.mark.parametrize(
    ("cause", "reason"),
    [
        ("deadline", None),
        ("environment", "task_success"),
        ("horizon", None),
        ("think_budget", None),
    ],
    ids=["deadline", "environment", "horizon", "think-budget"],
)
def test_an_invalidation_after_the_end_still_makes_the_run_invalid(cause, reason):
    # The invalid run comes before every other stop reason, the task's own included (legacy_loop.py
    # _finalize).
    kernel, _ = _kernel()
    kernel.end(cause, 300.0, reason=reason)
    kernel._flush()
    kernel.post(Invalidated("provider", "action_commit_feedback_failed"))
    kernel.step(300.0)
    _finish(kernel, 300.1)
    assert kernel.termination.stop_reason == "invalid_run"
    assert kernel.termination.cause == cause
    assert kernel.termination.invalid.reason == "action_commit_feedback_failed"


def test_a_failed_snapshot_invalidates_and_has_no_time():
    kernel, _ = _kernel()
    kernel.end("deadline", 300.0)
    kernel._flush()
    kernel.post(SnapshotFrozen(300.2, 300.2, error="state read failed"))
    kernel.step(300.2)
    assert kernel.termination.stop_reason == "invalid_run"
    assert kernel.termination.invalid.reason == "evaluation_snapshot_failed"
    assert kernel.termination.snapshot_at is None


def test_termination_waits_for_the_policy_to_settle():
    kernel, policy = _kernel()
    policy.settled = False
    kernel.end("deadline", 300.0)
    kernel._flush()
    _finish(kernel, 300.0)
    assert kernel.termination is None
    policy.settled = True
    kernel.post(AgentRecord("ModelResponseReceived"))
    kernel.step(300.5)
    assert kernel.termination is not None


def test_nothing_is_handed_to_the_policy_after_termination():
    kernel, policy = _kernel()
    kernel.end("deadline", 300.0)
    kernel._flush()
    _finish(kernel, 300.0)
    calls = list(policy.calls)
    kernel.post(EndChecked(5, ended=True))
    commands = kernel.step(301.0)
    assert _kinds(commands) == ["EndChecked"]
    assert policy.calls == calls
