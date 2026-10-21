"""How run_episode carries failures to the kernel (evaluation/shell.py), on the fake clock.

An exception the legacy loop let escape it (runner.py run_stream) invalidates the run as
runner:run_exception and the episode still finalizes. An invalid run the environment latches
itself in an operation that returns keeps its own reason: the shell posts it before the
command's answer, and the wall clock's key-release timer posts it through ``on_invalid``. Failed
state reads are counted by the kernel, three in a row invalidate the run, the ones a token key
tap makes inside execute() too (X12).
"""

from __future__ import annotations

import threading

import pytest

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.evaluation.kernel.core import Kernel
from realtime_framework.evaluation.kernel.turns import TurnPolicy

from .world import CLICK, Agent, Environment, Reply, World, run

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
    """The agent's accept raises at 0.5 s. The result lands in frame 0, read already, so the
    execution follows the accept in the same batch of commands and runs: the decision is settled
    as applied with the episode over and reported, the snapshot is frozen, and the whole
    traceback is the summary's error."""
    episode = run("wall", 3.0, [Reply(0.5), Reply(0.5)], agent_type=_RaisingAccept)
    _finalized(episode, 0.5)
    assert episode.invalid["category"] == "runner"
    assert episode.invalid["reason"] == "run_exception"
    assert "RuntimeError: accept broke" in episode.invalid["detail"]
    assert episode.env._loop_error.startswith("Traceback")
    assert "accept broke" in episode.env._loop_error
    raised = [line for line in episode.logs if line.startswith("[default] decision loop raised:")]
    assert len(raised) == 1
    assert episode.statuses() == ["applied"]
    assert episode.agent.outcomes[0].terminal_after_action is True
    assert episode.env.lifecycle[0]["model_latency_s"] == 0.5  # the result it accepted


def test_a_decision_budget_the_adapter_refuses_invalidates_the_run_before_the_generation():
    episode = run("wall", 3.0, [Reply(0.5)], agent_type=_NoTransportCap)
    _finalized(episode, 0.0)
    assert episode.invalid["reason"] == "run_exception"
    assert "adapter transport output cap" in episode.env._loop_error
    assert episode.agent.generated == []
    assert "InferenceStarted" not in episode.trace.events()


class _SlowNoTransportCap(Agent):
    """The adapter takes 0.2 s to refuse the decision budget."""

    def transport_output_token_cap(self, cap):
        self.clock.sleep(0.2)
        return 0


@pytest.mark.parametrize(
    ("duration", "config"),
    [(0.1, {}), (3.0, {"per_move_cap_s": 0.1})],
    ids=["deadline", "cap"],
)
def test_a_bound_past_a_start_that_raised_gives_up_a_generation_that_never_started(
    duration, config
):
    """StartGeneration raises 0.2 s in, past the bound, which the kernel handles before the
    run_exception: it gives up a generation that never started. (Over the cap, the kernel then
    reports the move skipped, and that commit has no result to record.)"""
    episode = run("wall", duration, [Reply(0.5)], agent_type=_SlowNoTransportCap, **config)
    assert episode.invalid["reason"] == "run_exception"
    raised = [line for line in episode.logs if line.startswith("[default] decision loop raised:")]
    assert "adapter transport output cap" in raised[0]
    assert not any("in _given_up" in line for line in raised)
    if not config:
        assert len(raised) == 1


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


class _FirstRowFails(Agent):
    def finalize_decision_audit(self, result: dict) -> None:
        if result["decision"] == 0:
            raise OSError("trajectory disk full")
        super().finalize_decision_audit(result)


def test_a_failed_trajectory_row_of_a_move_over_the_cap_invalidates_the_run():
    """The record of the call given up on at the cap is written like any result's: a failed
    write is a harness failure (runner.py _decide)."""
    episode = run(
        "wall", 3.0, [Reply(hang=True), Reply(0.5)], per_move_cap_s=1.0, agent_type=_FirstRowFails
    )
    assert episode.invalid["category"] == "harness"
    assert episode.invalid["reason"] == "trajectory_write_failed"
    assert "OSError: trajectory disk full" in episode.invalid["detail"]
    assert episode.env._stop_reason == "invalid_run"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_generation_that_raises_comes_back_failed_though_logging_it_fails():
    """The worker hands its result back whatever happens on the way: here the log line about the
    failure raises (runner.py _bounded_call hands back in a finally)."""
    lines = []

    def log(line: str) -> None:
        if "generation failed" in line:
            raise BrokenPipeError("stdout closed")
        lines.append(line)

    episode = run("wall", 3.0, [Reply(0.5, raises="provider down")], log=log)
    # The worker ends with the log's error once it has handed back; it ends within this test.
    for thread in threading.enumerate():
        if thread.name == "generation-1":
            thread.join(5)
    assert episode.invalid["reason"] == "generation_error"
    assert not any("STILL inside a native call" in line for line in lines)


class _UnfinishableEnvironment(Environment):
    """Refuses the termination record."""

    def __setattr__(self, name, value):
        if name == "_termination":
            raise RuntimeError("read-only")
        super().__setattr__(name, value)


def test_the_shell_stops_at_the_finish_though_writing_the_termination_raises():
    """Nothing more is asked after the finish (kernel/core.py), so the shell stops there."""
    episode = run("wall", 1.0, [Reply(0.5)], env_type=_UnfinishableEnvironment)
    assert [call for call, _, _ in episode.env.calls][-2:] == ["snapshot", "release_inputs"]
    assert any(line.startswith("[default] decision loop raised:") for line in episode.logs)


def test_a_failed_trajectory_row_of_a_call_the_deadline_cut_invalidates_the_run():
    """The record of the call the deadline gave up on is written like any result's (decision
    48): a failed write is a harness failure, after the end too (runner.py _decide)."""
    episode = run("wall", 2.0, [Reply(hang=True)], agent_type=_FirstRowFails)
    assert episode.invalid["reason"] == "trajectory_write_failed"
    assert episode.env._stop_reason == "invalid_run"
    assert episode.env._termination["end_s"] == 2.0


class _SlowRows(Agent):
    """Writing a trajectory row takes 10 ms; with ``fails`` the first one fails."""

    fails = False

    def finalize_decision_audit(self, result: dict) -> None:
        self.clock.sleep(0.01)
        if self.fails and result["decision"] == 0:
            raise OSError("trajectory disk full")
        super().finalize_decision_audit(result)


class _SlowFailingRows(_SlowRows):
    fails = True


class _SlowToolRows(_SlowRows):
    """With a tool runtime a result needs its inspection proof (episode_records.result_invalid)."""

    tool_runtime = object()


@pytest.mark.parametrize("bound", ["cap", "deadline"])
@pytest.mark.parametrize(
    ("agent_type", "raises", "reason"),
    [
        (_SlowFailingRows, None, "trajectory_write_failed"),
        (_SlowRows, "provider down", "generation_error"),
        (_SlowToolRows, None, "inspection_protocol_error"),
    ],
    ids=["row", "generation", "inspection"],
)
def test_a_result_settled_past_its_bound_still_invalidates_the_run(
    bound, agent_type, raises, reason
):
    """The result comes back 5 ms before the bound and settling it ends 5 ms after it: the
    kernel gives the generation up first and discards it. What made the result unusable still
    invalidates the run, as the runner's check of every result it settled did (runner.py
    _apply). (The next call never answers: the record of a call given up on needs no proof.)"""
    if bound == "cap":
        replies = [Reply(1.495, raises=raises), Reply(hang=True)]
        config = {"per_move_cap_s": 1.5}
    else:
        replies, config = [Reply(2.995, raises=raises)], {}
    episode = run("wall", 3.0, replies, agent_type=agent_type, **config)
    assert episode.invalid["reason"] == reason
    assert episode.env._stop_reason == "invalid_run"


def _breaks(monkeypatch, owner, name: str, when) -> None:
    """``owner.name`` raises from the first call ``when(self, *args)`` holds for on."""
    original = getattr(owner, name)

    def broken(self, *args, **kwargs):
        if when(self, *args):
            raise RuntimeError(f"{name} broke")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(owner, name, broken)


def _wrapped_up(episode, at: float, snapshot_at: float | None = None) -> None:
    """The shell wrapped the episode up without the kernel: the snapshot frozen once and the
    inputs released after it, the run invalid and its termination record written."""
    env = episode.env
    calls = [call for call, _, _ in env.calls if call != "check_end"]
    assert calls.count("snapshot") == 1 and calls[-2:] == ["snapshot", "release_inputs"]
    assert env._termination_snapshot_captured is True
    snapshot_at = at if snapshot_at is None else snapshot_at
    assert env._termination == {
        "stop_reason": "invalid_run",
        "end_s": at,
        "snapshot_s": snapshot_at,
        "snapshot_delay_s": round(snapshot_at - at, 9),
    }
    assert env._stop_reason == "invalid_run"
    assert any(line.startswith("[default] decision loop raised:") for line in episode.logs)


def test_a_kernel_that_raises_leaves_the_episode_wrapped_up_by_the_shell(monkeypatch):
    """N11: the kernel raises at the end check at 2 s, during the second call, which never
    answers. Nothing can be asked of the kernel any more: the shell interrupts the call, freezes
    the snapshot, releases the inputs and writes the termination record itself, as the runner
    wrapped up a loop that raised (runner.py run_stream). The run is invalid: run_exception,
    its traceback the summary's error, recorded as the runner records it."""
    _breaks(monkeypatch, Kernel, "step", lambda kernel, semantic, ops=None: semantic >= 2.0)
    episode = run("wall", 3.0, [Reply(0.5), Reply(hang=True)])
    _wrapped_up(episode, 2.0)
    assert episode.statuses() == ["applied"]
    assert ("abort", 2) in episode.agent.calls
    invalid = episode.invalid
    assert (invalid["category"], invalid["reason"]) == ("runner", "run_exception")
    assert "RuntimeError: step broke" in invalid["detail"]
    assert "RuntimeError: step broke" in episode.env._loop_error
    assert any(line.startswith("[invalid-run] runner:run_exception") for line in episode.logs)
    (written,) = episode.trace.payloads("RunInvalidated")
    assert written["reason"] == "run_exception"


def test_a_policy_that_raises_leaves_the_episode_wrapped_up_by_the_shell(monkeypatch):
    """The policy raises as it takes the second result, back at 1.5 s, inside the kernel's
    step: the same as the kernel raising."""
    _breaks(monkeypatch, TurnPolicy, "_accept", lambda policy, *_: policy.decisions == 1)
    episode = run("wall", 3.0, [Reply(0.5), Reply(0.5)])
    _wrapped_up(episode, 1.5)
    assert episode.invalid["reason"] == "run_exception"
    assert "RuntimeError: _accept broke" in episode.env._loop_error


class _BrokenTicks(Environment):
    """The environment's tick reading raises from tick 2 on; its snapshot does not read it."""

    def tick_index(self) -> float:
        if self.semantic() >= 2:
            raise RuntimeError("tick_index broke")
        return super().tick_index()

    def _snapshot(self) -> dict:
        self.log("snapshot")
        return {"tick": self.semantic()}


def test_a_failed_clock_reading_leaves_the_episode_wrapped_up_by_the_shell():
    """The token clock comes to tick 2 after the second action, and the shell's next reading of
    it raises: the episode ends at the last reading the shell took, tick 1."""
    replies = [Reply(0.0, CLICK, output_tokens=100)] * 3
    episode = run("token", 5.0, replies, tokens_per_frame=100, env_type=_BrokenTicks)
    _wrapped_up(episode, 1.0)
    assert episode.invalid["reason"] == "run_exception"
    assert "RuntimeError: tick_index broke" in episode.env._loop_error


def test_a_kernel_that_raises_after_the_end_keeps_its_end_and_its_snapshot(monkeypatch):
    """The deadline ended the episode at 3 s and the snapshot, frozen then, took until 3.5 s;
    the kernel raises as it takes the snapshot's answer. The end stays the deadline, the first
    end signal (decision 32), and the snapshot is not taken again."""
    _breaks(monkeypatch, Kernel, "step", lambda kernel, *_: kernel.ending is not None)
    episode = run("wall", 3.0, [Reply(hang=True)], World(snapshot_s=0.5))
    _wrapped_up(episode, 3.0, snapshot_at=3.5)
    assert episode.invalid["reason"] == "run_exception"


def test_a_kernel_that_raises_after_an_invalidation_keeps_the_first_failure(monkeypatch):
    """The generation fails at 0.5 s, and the kernel raises at its next step: the run stays
    invalid for the first failure, as under the runner (runner.py _invalidate)."""
    _breaks(monkeypatch, Kernel, "step", lambda kernel, *_: kernel.invalid is not None)
    episode = run("wall", 3.0, [Reply(0.5, raises="provider down")])
    _wrapped_up(episode, 0.5)
    assert episode.invalid["reason"] == "generation_error"
    assert "RuntimeError: step broke" in episode.env._loop_error
    assert [p["reason"] for p in episode.trace.payloads("RunInvalidated")] == ["generation_error"]


def test_a_kernel_that_raises_before_taking_the_environments_failure_keeps_it(monkeypatch):
    """The key held from 0.5 s fails to come up at 1.25 s, and the kernel raises at the step that
    would take the environment's report of it: that failure came first, and the run is invalid
    for it, its record the environment's."""
    _breaks(monkeypatch, Kernel, "step", lambda kernel, semantic, ops=None: semantic >= 1.25)
    world = World(release_failures={0})
    episode = run("wall", 3.0, [Reply(0.5, Action.hold_key("a", 0.75)), Reply(hang=True)], world)
    _wrapped_up(episode, 1.25)
    assert episode.invalid == {
        "category": "environment",
        "reason": "wall_key_release_failed",
        "detail": "keyup failed",
    }
    assert episode.trace.payloads("RunInvalidated") == []


def test_a_wrap_up_that_raises_is_only_logged(monkeypatch):
    """The kernel raises, and the environment refuses the termination record: run_episode
    still returns, and the failure is logged."""
    _breaks(monkeypatch, Kernel, "step", lambda kernel, semantic, ops=None: semantic >= 1.0)
    episode = run("wall", 3.0, [Reply(0.5), Reply(0.5)], env_type=_UnfinishableEnvironment)
    assert [call for call, _, _ in episode.env.calls][-2:] == ["snapshot", "release_inputs"]
    assert any(
        line.startswith("[default] wrapping up the episode failed:") for line in episode.logs
    )


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


def test_a_token_wait_whose_own_step_fails_is_a_failed_action():
    """Paused at tick 0, a 3 s wait: its second step fails and the environment latches
    clock_advance_failed. The wait is a failed action, as the environment reports it
    (sandbox/actions.py GuestEnvironment.execute; kernel/turns.py _step_failed): invalid_run,
    an error for its execution record, no ActionApplied, and nothing escaped the loop."""
    world = World(advance_errors={2: "advance failed"})
    episode = run("token", 5.0, [Reply(0.0, Action.wait(3.0))], world)
    _finalized(episode, 1.0)
    assert episode.invalid == {
        "category": "environment",
        "reason": "clock_advance_failed",
        "detail": "advance failed",
    }
    assert episode.statuses() == ["invalid_run"]
    (record,) = episode.env.lifecycle
    assert "RuntimeError: advance failed" in record["execution_result"]["error"]
    assert "ActionApplied" not in episode.trace.events()
    assert getattr(episode.env, "_loop_error", None) is None
    assert not any("decision loop raised" in line for line in episode.logs)


def test_the_first_failure_is_the_record_though_the_environment_latched_another_after_it():
    """The lifecycle write of decision 0 fails at 1 s and the snapshot read takes until 2 s; the
    key held from 0.25 s fails to come up at 1.25 s, meanwhile, and the environment latches
    that. The run is invalid for the first failure, on env too (runner.py _invalidate)."""
    world = World(release_failures={0}, lifecycle_errors={0}, snapshot_s=1.0)
    episode = run("wall", 4.0, [Reply(0.25, Action.hold_key("a", 1.0)), Reply(0.5)], world)
    assert [call for call, _, _ in episode.env.calls if call != "check_end"][-3:] == [
        "wall_release",
        "snapshot",
        "release_inputs",
    ]
    assert episode.invalid["reason"] == "action_lifecycle_write_failed"


def test_the_environment_keeps_its_own_record_of_the_invalid_run_it_latched():
    """The kernel keeps the end of a long detail (kernel/core.py invalidate); the record main()
    reads stays the environment's, whole, as under the runner (runner.py _invalidate)."""
    detail = "x" * 3000
    latched = {"category": "environment", "reason": "clock_advance_failed", "detail": detail}
    episode = run("token", 3.0, [Reply(0.0, TAP)], World(latches={0: latched}))
    assert episode.invalid == latched
    assert episode.trace.payloads("RunInvalidated")[0]["detail"] == detail[-2000:]


def test_the_environment_keeps_its_reason_for_an_advance_that_raised():
    """The environment latches clock_advance_failed and raises while the clock moves to the
    action time. The step is answered with its error and the kernel gives it the environment's
    reason (kernel/turns.py _step_failed); the record stays the environment's. The traceback is
    the summary's error, as under the runner, whose loop the exception escaped."""
    world = World(advance_errors={1: "advance failed"})
    episode = run("token", 3.0, [Reply(0.0, CLICK, output_tokens=100)], world, tokens_per_frame=100)
    _finalized(episode, 0.0)
    assert episode.invalid == {
        "category": "environment",
        "reason": "clock_advance_failed",
        "detail": "advance failed",
    }
    assert "RuntimeError: advance failed" in episode.trace.payloads("RunInvalidated")[0]["detail"]
    assert "RuntimeError: advance failed" in episode.env._loop_error
    assert any(line.startswith("[default] decision loop raised:") for line in episode.logs)


@pytest.mark.parametrize(
    ("world", "action", "at", "reason"),
    [
        (
            World(release_errors={1: "keyup failed"}),
            Action.hold_key("a", 0.5),
            1.5,
            "clock_key_release_failed",
        ),
        (World(render_errors={1: "render failed"}), CLICK, 1.0, "frame_render_failed"),
    ],
    ids=["release", "render"],
)
def test_a_token_release_or_render_that_raises_invalidates_the_run(world, action, at, reason):
    """Each action lands at tick 1. The hold's lease ends at tick 1.5 on the way to the next
    frame, and its release raises; or the frame the click committed fails to render. The kernel
    gives the step the environment's reason (kernel/turns.py _released, _rendered). The
    traceback is the summary's error, as under the runner, whose loop the exception escaped."""
    replies = [Reply(0.0, action, output_tokens=100)] * 2
    episode = run("token", 3.0, replies, world, tokens_per_frame=100)
    _finalized(episode, at)
    assert (episode.invalid["category"], episode.invalid["reason"]) == ("environment", reason)
    assert "RuntimeError" in episode.env._loop_error
    assert any(line.startswith("[default] decision loop raised:") for line in episode.logs)


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
        (((False, None, "tap read 1"), (False, None, "tap read 2")), True),
        (((False, None, "tap read 1"), (False, None, None)), False),
    ],
    ids=["three-in-a-row", "a-read-between"],
)
def test_the_reads_a_tap_makes_count_toward_three_failed_reads_in_a_row(tap_checks, invalid):
    """Paused at tick 0: the loop top answers, and tick 0 is not read again before execution; the
    tap's two reads are passed on as ActionExecuted.end_checks, and the check right after
    execution fails. Two failed tap reads and that one are three in a row; a tap read that
    answered starts over."""
    world = World(tap_checks={0: tap_checks}, check_errors={2: "state read 2"})
    episode = run("token", 1.0, [Reply(0.0, TAP)], world)
    assert episode.env.kernel_counts_reads is True
    if invalid:
        assert episode.invalid["reason"] == "terminal_state_check_failed"
        assert "RuntimeError: state read 2" in episode.invalid["detail"]
        assert episode.statuses() == ["applied"]
        assert episode.agent.outcomes[0].terminal_after_action is True
    else:
        assert episode.invalid is None
        assert episode.env._stop_reason == "tick_limit"


class _ExecuteRaises(Environment):
    """Breaks the protocol: execute() raises instead of returning "failed"."""

    def execute(self, action, *, release_at=None):
        raise RuntimeError("execute broke")


def test_a_command_the_kernel_waits_for_is_answered_though_it_raised():
    """The kernel settles a decision by the answer to its Execute; one whose run raised is
    answered with its failure, so the episode still finalizes (kernel/core.py)."""
    episode = run("wall", 3.0, [Reply(0.5)], env_type=_ExecuteRaises)
    _finalized(episode, 0.5)
    assert episode.invalid["reason"] == "run_exception"
    assert "RuntimeError: execute broke" in episode.env._loop_error
    assert episode.statuses() == ["invalid_run"]


def test_a_tap_read_that_saw_the_end_ends_the_episode_by_the_tasks_reason():
    """Paused at tick 0: the tap's read saw the end, passed on as ActionExecuted.end_checks. The
    episode ends there by the task's reason; the check that would follow execution, scripted to
    fail, is never asked."""
    world = World(
        tap_checks={0: ((True, "target_reached", None),)}, check_errors={2: "state read 2"}
    )
    episode = run("token", 1.0, [Reply(0.0, TAP)], world)
    assert [call for call, _, _ in episode.env.calls].count("check_end") == 1
    assert episode.invalid is None
    assert episode.env._stop_reason == "target_reached"
    assert episode.statuses() == ["applied"]
    assert episode.agent.outcomes[0].terminal_after_action is True


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
