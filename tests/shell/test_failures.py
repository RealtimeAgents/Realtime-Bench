"""How run_episode carries failures to the kernel (evaluation/shell.py), on the fake clock.

An exception the legacy loop let escape it (runner.py run_stream) invalidates the run as
runner:run_exception and the episode still finalizes. An invalid run the environment latches
itself keeps its own reason: the shell posts it before the command's answer, and the wall
clock's key-release timer posts it through ``on_invalid``. Failed state reads are counted by the
kernel, three in a row invalidate the run, the ones a token key tap makes inside execute() too
(X12).
"""

from __future__ import annotations

import pytest

from realtime_framework.entity.actions import Action, ActionType

from .world import CLICK, Agent, Reply, World, run

TAP = Action(ActionType.KEY, {"key": "a"})


class _RaisingAccept(Agent):
    def accept(self, generation: int) -> None:
        super().accept(generation)
        raise RuntimeError("accept broke")


class _NoTransportCap(Agent):
    def transport_output_token_cap(self, cap):
        return 0


class _RaisingCommit(Agent):
    def commit(self, outcome) -> None:
        super().commit(outcome)
        raise RuntimeError("feedback broke")


def _finalized(episode, at: float) -> None:
    """The snapshot was frozen and the inputs released after it; the run ended at ``at``."""
    env = episode.env
    assert [call for call, _, _ in env.calls][-2:] == ["snapshot", "release_inputs"]
    assert env._termination_snapshot_captured is True
    assert env._stop_reason == "invalid_run"
    assert env._termination["stop_reason"] == "invalid_run"
    assert env._termination["end_s"] == at


def test_an_exception_a_command_raises_invalidates_the_run_and_still_finalizes():
    """The agent's accept raises at 0.5 s. The decision in hand is settled as invalid_run and
    reported, the snapshot is frozen, and the whole traceback is the summary's error."""
    episode = run("wall", 3.0, [Reply(0.5), Reply(0.5)], agent_type=_RaisingAccept)
    _finalized(episode, 0.5)
    assert episode.invalid["category"] == "runner"
    assert episode.invalid["reason"] == "run_exception"
    assert "RuntimeError: accept broke" in episode.invalid["detail"]
    assert episode.env._loop_error.startswith("Traceback")
    assert "accept broke" in episode.env._loop_error
    raised = [line for line in episode.logs if line.startswith("[default] decision loop raised:")]
    assert len(raised) == 1
    assert episode.statuses() == ["invalid_run"]
    assert episode.env.lifecycle[0]["model_latency_s"] == 0.5  # the result it accepted


def test_a_decision_budget_the_adapter_refuses_invalidates_the_run_before_the_generation():
    episode = run("wall", 3.0, [Reply(0.5)], agent_type=_NoTransportCap)
    _finalized(episode, 0.0)
    assert episode.invalid["reason"] == "run_exception"
    assert "adapter transport output cap" in episode.env._loop_error
    assert episode.agent.generated == []
    assert "InferenceStarted" not in episode.trace.events()


def test_a_generation_that_raises_is_an_unusable_result():
    episode = run("wall", 3.0, [Reply(0.5, raises="provider down")])
    _finalized(episode, 0.5)
    assert episode.invalid["category"] == "provider"
    assert episode.invalid["reason"] == "generation_error"
    assert "RuntimeError: provider down" in episode.invalid["detail"]
    assert getattr(episode.env, "_loop_error", None) is None
    assert any("k=0 generation failed, recovering" in line for line in episode.logs)
    assert episode.statuses() == []
    completed = episode.trace.payloads("InferenceCompleted")
    assert [(p["error"], p["invalid_reason"]) for p in completed] == [(True, "generation_error")]


class _Cancelled(BaseException):
    """As entity/model.py RetryCancelled, which is not an Exception."""


class _CancelledGeneration(Agent):
    def generate(self, generation, decision, observation, budget):
        self.clock.sleep(0.5)
        raise _Cancelled("retries cancelled")


def test_a_generation_that_raises_any_exception_comes_back_failed():
    """Not only an Exception: the kernel waits for the result, so the worker always hands one
    back, as the runner's bounded call did (runner.py _bounded_call)."""
    episode = run("wall", 3.0, [Reply(0.5)], agent_type=_CancelledGeneration)
    _finalized(episode, 0.5)
    assert episode.invalid["reason"] == "generation_error"
    assert "_Cancelled: retries cancelled" in episode.invalid["detail"]


def test_the_environment_keeps_its_reason_for_a_tap_whose_advance_failed():
    """GuestEnvironment latches clock_advance_failed when a token key tap cannot advance, and its
    execute() returns "failed" (sandbox/actions.py). Without the bridge the kernel would read
    the failure as action_execution_failed."""
    latched = {"category": "environment", "reason": "clock_advance_failed", "detail": "no tick"}
    episode = run("token", 3.0, [Reply(0.0, TAP)], World(latches={0: latched}))
    _finalized(episode, 0.0)
    assert episode.invalid == latched
    assert episode.statuses() == ["invalid_run"]
    assert getattr(episode.env, "_loop_error", None) is None


def test_the_environment_keeps_its_reason_for_an_advance_that_raised():
    """The environment latches clock_advance_failed and raises, which escaped the legacy loop:
    the reason is the environment's and the traceback is the summary's error."""
    world = World(advance_errors={1: "advance failed"})
    episode = run("token", 3.0, [Reply(0.0, CLICK, output_tokens=100)], world, tokens_per_frame=100)
    _finalized(episode, 0.0)
    assert episode.invalid == {
        "category": "environment",
        "reason": "clock_advance_failed",
        "detail": "advance failed",
    }
    assert "RuntimeError: advance failed" in episode.env._loop_error


def test_a_failed_wall_key_release_invalidates_the_run_when_it_happens():
    """The hold executed at 0.5 s is released 0.75 s later by the environment's own timer, on
    its own thread; the release fails and the environment calls on_invalid. The run ends then,
    during the next generation."""
    world = World(release_failures={0})
    episode = run("wall", 3.0, [Reply(0.5, Action.hold_key("a", 0.75)), Reply(1.0)], world)
    _finalized(episode, 1.25)
    assert episode.invalid == {
        "category": "environment",
        "reason": "wall_key_release_failed",
        "detail": "keyup failed",
    }
    assert episode.statuses() == ["applied"]
    assert episode.agent.calls[-1] == ("abort", 2)


@pytest.mark.parametrize(
    ("tap_checks", "invalid"),
    [
        (("tap read 1", "tap read 2"), True),
        (("tap read 1", None), False),
    ],
    ids=["three-in-a-row", "a-read-between"],
)
def test_the_reads_a_tap_makes_count_toward_three_failed_reads_in_a_row(tap_checks, invalid):
    """Paused at tick 0: the loop top and the check before execution answer, the tap's two reads
    are passed on as ActionExecuted.end_checks, and the check right after execution fails. Two
    failed tap reads and that one are three in a row; a tap read that answered starts over."""
    world = World(tap_checks={0: tap_checks}, check_errors={3: "state read 3"})
    episode = run("token", 1.0, [Reply(0.0, TAP)], world)
    assert episode.env.kernel_counts_reads is True
    if invalid:
        assert episode.invalid["reason"] == "terminal_state_check_failed"
        assert "RuntimeError: state read 3" in episode.invalid["detail"]
        assert episode.statuses() == ["applied"]
        assert episode.agent.outcomes[0].terminal_after_action is True
    else:
        assert episode.invalid is None
        assert episode.env._stop_reason == "tick_limit"


def test_a_failed_end_check_alone_does_not_end_the_episode():
    episode = run("wall", 2.0, [Reply(0.5), Reply(0.5)], World(check_errors={2: "slow read"}))
    assert episode.invalid is None
    assert episode.statuses() == ["applied", "applied"]


def test_a_failed_capture_invalidates_the_run():
    episode = run("wall", 3.0, [Reply(0.5)], World(capture_errors={1: "no frame"}))
    _finalized(episode, 1.0)
    assert episode.invalid["reason"] == "observation_capture_failed"
    assert "RuntimeError: no frame" in episode.invalid["detail"]
    assert episode.statuses() == ["applied"]


def test_an_execution_failure_invalidates_the_run():
    episode = run("wall", 3.0, [Reply(0.5)], World(failures={0: "guest down"}))
    _finalized(episode, 0.5)
    assert episode.invalid == {
        "category": "environment",
        "reason": "action_execution_failed",
        "detail": "guest down",
    }
    assert episode.statuses() == ["invalid_run"]


def test_failed_feedback_and_a_failed_lifecycle_write_invalidate_the_run():
    episode = run("wall", 3.0, [Reply(0.5), Reply(0.5)], agent_type=_RaisingCommit)
    assert episode.invalid["reason"] == "action_commit_feedback_failed"
    assert "feedback broke" in episode.invalid["detail"]
    assert episode.statuses() == ["applied"]
    episode = run("wall", 3.0, [Reply(0.5), Reply(0.5)], World(lifecycle_errors={0}))
    assert episode.invalid["category"] == "harness"
    assert episode.invalid["reason"] == "action_lifecycle_write_failed"
    assert episode.statuses() == ["applied"]


def test_a_failed_snapshot_read_invalidates_the_run_and_leaves_no_snapshot_time():
    episode = run("wall", 1.0, [Reply(0.5)], World(snapshot_error="state gone"))
    env = episode.env
    assert episode.invalid["reason"] == "evaluation_snapshot_failed"
    assert "RuntimeError: state gone" in env._termination_snapshot_error
    assert env._termination_snapshot_captured is False
    assert env._termination == {
        "stop_reason": "invalid_run",
        "end_s": 1.0,
        "snapshot_s": None,
        "snapshot_delay_s": None,
    }
