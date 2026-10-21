"""Each kernel command as run_episode runs it against the environment and the agent
(evaluation/shell.py), on the fake clock.

Times are semantic: seconds since the episode origin on the wall clock, environment ticks on the
token clock. Every expected value follows from the kernel's turn rules (kernel/turns.py) and the
scripted world (tests/shell/world.py).
"""

from __future__ import annotations

import gc

import pytest

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.entity.config import DecisionBudget, RunnerTimingConfig

from .world import CLICK, ORIGIN, Agent, Reply, World, run

HOLD = Action.hold_key("a", 1.25)
# Chat-completions usage, which the token accounting counts whole.
USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 20,
    "total_tokens": 1020,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 0},
}
WAIT = Action.wait(0.5)


def _calls(episode) -> list[tuple]:
    return [(call, time) for call, time, _ in episode.env.calls]


def test_a_token_delay_is_recorded_in_input_frames_and_timed_in_environment_ticks():
    """At 4 fps with the model fed at 1 fps and B = 100, 50 tokens are half an input frame, two
    environment ticks. The lifecycle records the delay in input frames and the timeline main()
    reads keeps it in ticks, as the runner does (legacy_loop.py _apply); the second decision's 100
    tokens are the whole allowance and land on the horizon."""
    replies = [Reply(0.0, CLICK, output_tokens=50), Reply(0.0, CLICK, output_tokens=100)]
    episode = run("token", 2.0, replies, fps=4.0, input_fps=1.0, tokens_per_frame=100)
    assert [entry[1:4] for entry in episode.timeline] == [(0, 2.0, 2.0), (4, 4.0, 8.0)]
    assert [record["token_delay_frames"] for record in episode.env.lifecycle] == [0.5, 1.0]
    assert episode.statuses() == ["applied", "not_applied_tick_limit"]


def test_a_wall_episode_runs_each_decision_on_the_environment_and_the_agent():
    """Three decisions of 0.5 s each at 1 fps, then the 3 s deadline. The environment is asked
    whether the episode ended by the periodic check once each frame, at the loop top only in a
    frame not read yet, right before an action unless it was asked at that same reading, and
    right after every execution.
    A wait is the kernel's own timer and never reaches the environment; the hold's keys are
    released by the environment's timer at the time the kernel gave."""
    episode = run("wall", 3.0, [Reply(0.5), Reply(0.5, HOLD), Reply(0.5, WAIT)])
    env, agent = episode.env, episode.agent
    assert _calls(episode) == [
        ("check_end", 0.0),  # the loop top
        ("capture", 0.0),
        ("check_end", 0.5),  # before execution
        ("execute", 0.5),
        ("check_end", 0.5),  # after execution
        ("check_end", 1.0),  # the periodic check; the loop top in frame 1 reads no more
        ("capture", 1.0),
        ("check_end", 1.5),
        ("execute", 1.5),
        ("check_end", 1.5),
        ("check_end", 2.0),
        ("capture", 2.0),
        ("check_end", 2.5),  # before the wait
        ("wall_release", 2.75),
        ("snapshot", 3.0),
        ("release_inputs", 3.0),
    ]
    assert [details for call, _, details in env.calls if call == "execute"] == [
        {"action": "click", "release_at": None},
        {"action": "hold_key", "release_at": 2.75},  # executed at 1.5, held 1.25 s
    ]
    assert agent.generated == [
        (1, 0, "obs-0", DecisionBudget(None, None, None, "wall")),
        (2, 1, "obs-1", DecisionBudget(None, None, None, "wall")),
        (3, 2, "obs-2", DecisionBudget(None, None, None, "wall")),
    ]
    assert agent.calls == [
        ("accept", 1),
        ("commit", 0),
        ("accept", 2),
        ("commit", 1),
        ("accept", 3),
        ("commit", 2),
    ]
    assert episode.statuses() == ["applied", "applied", "wait"]
    assert episode.timeline == [
        (0, 0, 0.5, 0.5, CLICK),
        (1, 1, 0.5, 1.5, HOLD),
        (2, 2, 0.5, 2.5, WAIT),
    ]
    assert [record["status"] for record in env.lifecycle] == ["applied", "applied", "wait"]
    assert env.frame_ages == [0.5, 0.5, 0.5]
    assert env._termination == {
        "stop_reason": "time_limit",
        "end_s": 3.0,
        "snapshot_s": 3.0,
        "snapshot_delay_s": 0.0,
    }
    assert env._stop_reason == "time_limit"
    assert env._termination_snapshot_captured is True
    assert env._termination_snapshot == {"tick": 3}
    assert env._termination_snapshot_error is None
    assert env._active_end_unix_s == ORIGIN + 3.0
    assert env._think_spent == 1.5
    assert env._dropped_intermediate_frames == 0
    assert getattr(env, "_invalid_run", None) is None
    assert env.kernel_counts_reads is True
    assert agent._clock_mode == "wall"
    assert env.horizon is None
    assert episode.trace.checkpoints == 3


def test_a_frames_check_and_the_loop_top_after_it_read_the_frame_once():
    """Reads take 1 ms of the fake clock; the times are when each read ends. The periodic check
    reads frames 1 and 2 when they begin, and the loop top a millisecond later, in the same
    frame, reads no more; the check before an action reads, the clock having moved on since."""
    episode = run("wall", 3.0, [Reply(0.5), Reply(0.5), Reply(0.5)], World(check_s=0.001))
    checks = [round(time, 3) for call, time, _ in episode.env.calls if call == "check_end"]
    assert checks == [0.001, 0.502, 0.503, 1.001, 1.502, 1.503, 2.001, 2.502, 2.503]


def test_a_token_episode_advances_releases_and_renders_through_the_environment():
    """2 fps, B = 100 per frame, 2 s: the horizon is tick 4. A decision's n tokens are n / 100
    frames, one tick each; the clock moves in steps of at most one tick, cut where the 0.5 s
    hold's lease ends (tick 2), with an end check after each step and after each execution, and
    none at a tick read already. A landing on the next frame commits its pixels with a render.
    The last decision's action would land at the horizon."""
    replies = [
        Reply(0.0, Action.hold_key("a", 0.5), output_tokens=100),
        Reply(0.0, CLICK, output_tokens=50),
        Reply(0.0, CLICK, output_tokens=100),
        Reply(0.0, CLICK, output_tokens=100),
    ]
    episode = run("token", 2.0, replies, fps=2.0, tokens_per_frame=100)
    env = episode.env
    assert [
        (call, time, details.get("step") or details.get("keys"))
        for call, time, details in env.calls
    ] == [
        ("check_end", 0, None),
        ("capture", 0, None),
        ("step_clock", 0, 1.0),
        ("check_end", 1, None),
        ("execute", 1, None),
        ("check_end", 1, None),
        ("render", 1, None),
        ("capture", 1, None),
        ("step_clock", 1, 0.5),
        ("check_end", 1.5, None),
        ("execute", 1.5, None),
        ("check_end", 1.5, None),
        ("step_clock", 1.5, 0.5),
        ("release_keys", 2, ("a",)),
        ("check_end", 2, None),
        ("capture", 2, None),
        ("step_clock", 2, 1.0),
        ("check_end", 3, None),
        ("execute", 3, None),
        ("check_end", 3, None),
        ("render", 3, None),
        ("capture", 3, None),
        ("step_clock", 3, 1.0),
        ("check_end", 4, None),
        ("snapshot", 4, None),
        ("release_inputs", 4, None),
    ]
    assert episode.statuses() == ["applied", "applied", "applied", "not_applied_tick_limit"]
    assert [entry[2:4] for entry in episode.timeline] == [
        (1.0, 1.0),
        (0.5, 1.5),
        (1.0, 3.0),
        (1.0, 4.0),
    ]
    assert [record["post_action_rendered"] for record in env.lifecycle] == [
        True,
        False,
        True,
        False,
    ]
    assert [record["action_output_tokens"] for record in env.lifecycle] == [100, 50, 100, 100]
    assert [record["token_delay_frames"] for record in env.lifecycle] == [1.0, 0.5, 1.0, 1.0]
    assert [budget for _, _, _, budget in episode.agent.generated] == [
        DecisionBudget(400, 400, 400, "token"),
        DecisionBudget(300, 300, 300, "token"),
        DecisionBudget(200, 200, 200, "token"),
        DecisionBudget(100, 100, 100, "token"),
    ]
    # Ticks to world seconds: the horizon's 4 ticks at 2 fps.
    assert env._termination == {
        "stop_reason": "tick_limit",
        "end_s": 2.0,
        "snapshot_s": 2.0,
        "snapshot_delay_s": 0.0,
    }
    assert env.horizon == 4.0
    assert episode.agent._clock_mode == "token"


def test_an_action_is_timed_when_it_lands_whatever_comes_of_it():
    """The runner times an action's resolution when it lands, before the check right before
    execution (legacy_loop.py _apply). With end checks of 125 ms the frame age and the resolution
    time still agree, for an executed action and for a wait the kernel lets pass."""
    episode = run("wall", 4.0, [Reply(0.5), Reply(0.5, Action.wait(1.0))], World(check_s=0.125))
    env = episode.env
    assert episode.statuses() == ["applied", "wait"]
    resolved = [p["action_resolved_unix_s"] for p in episode.trace.payloads("ActionResolved")]
    assert resolved == [record["action_resolved_unix_s"] for record in env.lifecycle]
    for record in env.lifecycle:
        age = record["action_resolved_unix_s"] - record["observation_timestamp"]
        assert round(age, 6) == record["frame_age_at_action_s"] == 0.5
    (executed,) = [time for call, time, _ in env.calls if call == "execute"]
    assert resolved[0] - env.t0 == executed - 0.125  # before the check right before execution


class _SlowTrajectory(Agent):
    """Writing a trajectory row takes 0.125 s."""

    def finalize_decision_audit(self, result: dict) -> None:
        super().finalize_decision_audit(result)
        self.clock.sleep(0.125)


def test_a_wall_action_is_timed_at_the_reading_it_lands_at_though_settling_it_took_time():
    """The result comes back at 0.5 s and settling it takes 0.125 s; the kernel gets it after,
    as the runner landed it after settling it (legacy_loop.py _decide): it lands at 0.625 s, and its
    record's resolution time and frame age agree, as the definition the record gives says they
    must."""
    episode = run("wall", 2.0, [Reply(0.5), Reply(0.5)], agent_type=_SlowTrajectory)
    for record in episode.env.lifecycle:
        assert record["frame_age_at_action_s"] == 0.625
        age = record["action_resolved_unix_s"] - record["observation_timestamp"]
        assert round(age, 6) == 0.625


class _SlowAccept(Agent):
    """Taking a result into the history takes 0.125 s, after the kernel's reading."""

    def accept(self, generation: int) -> None:
        super().accept(generation)
        self.clock.sleep(0.125)


def test_a_wall_action_is_timed_at_the_reading_it_lands_at_though_accepting_it_took_time():
    """The result is back at 0.5 s and the agent takes 0.125 s to accept it. The action lands
    at the reading after the accept, 0.625 s (kernel/turns.py _accept), and is executed then;
    the record's resolution time is that landing's reading, which its frame age counts from.
    The action time stays where the result was handled."""
    episode = run("wall", 2.0, [Reply(0.5), Reply(0.5)], agent_type=_SlowAccept)
    assert [time for call, time, _ in episode.env.calls if call == "execute"] == [0.625, 1.625]
    assert [o.action_clock_position for o in episode.agent.outcomes] == [0.5, 1.5]
    for record in episode.env.lifecycle:
        assert record["frame_age_at_action_s"] == 0.625
        age = record["action_resolved_unix_s"] - record["observation_timestamp"]
        assert round(age, 6) == 0.625


def test_a_token_action_is_timed_when_the_clock_got_to_it():
    """The call takes 0.25 s and its 100 tokens put the action at tick 1; the clock gets there at
    once. The runner times the resolution then (legacy_loop.py _apply): 0.25 s in, operational."""
    episode = run("token", 3.0, [Reply(0.25, CLICK, output_tokens=100)], tokens_per_frame=100)
    (resolved, *_) = episode.trace.payloads("ActionResolved")
    assert resolved["action_resolved_unix_s"] == ORIGIN + 0.25
    assert episode.env.lifecycle[0]["action_resolved_unix_s"] == ORIGIN + 0.25


def test_an_action_the_guest_refuses_is_reported_with_its_reason():
    episode = run("wall", 2.0, [Reply(0.5), Reply(0.5)], World(rejections={0: "bad key"}))
    assert episode.statuses() == ["rejected", "applied"]
    assert episode.agent.outcomes[0].detail == "bad key"
    assert episode.env.lifecycle[0]["execution_result"] == {"error": "bad key"}
    assert episode.invalid is None


@pytest.mark.parametrize("terminal", [ActionType.DONE, ActionType.FAIL])
def test_a_done_reply_lands_as_a_wait_and_its_record_says_why(terminal):
    """The loop has no terminal actions: the kernel lands a DONE or a FAIL as a wait, and the
    lifecycle record keeps the fallback protocol the runner writes (legacy_loop.py _decide)."""
    episode = run(
        "wall",
        1.0,
        [Reply(0.5, Action(terminal), fields={"action_protocol": {"source": "json"}})],
    )
    outcome = episode.agent.outcomes[0]
    assert (outcome.status, outcome.action, outcome.protocol_failure) == (
        "wait",
        Action.wait(),
        "unsupported_terminal_action",
    )
    assert episode.env.lifecycle[0]["action_protocol"] == {
        "source": "invalid_action_fallback_wait",
        "valid_action": False,
        "protocol_failure": "unsupported_terminal_action",
    }
    assert "ActionProposed" not in episode.trace.events()


@pytest.mark.parametrize("terminal", [ActionType.DONE, ActionType.FAIL])
def test_a_done_reply_over_the_cap_still_records_why_it_was_not_a_done(terminal):
    """Its 100 tokens are one second at B = 100, over the 0.5 s cap: the kernel places no action
    and so names no protocol failure (kernel/turns.py _accept). The runner rewrites the protocol
    of every DONE or FAIL it was given (legacy_loop.py _decide), and the record says so here too."""
    reply = Reply(0.0, Action(terminal), 100, fields={"action_protocol": {"source": "json"}})
    episode = run("token", 2.0, [reply, reply], tokens_per_frame=100, per_move_cap_s=0.5)
    assert episode.statuses()[0] == "skipped_timeout"
    assert episode.agent.outcomes[0].protocol_failure is None
    assert episode.env.lifecycle[0]["action_protocol"] == {
        "source": "invalid_action_fallback_wait",
        "valid_action": False,
        "protocol_failure": "unsupported_terminal_action",
    }


def test_a_move_over_the_per_move_cap_is_recorded_as_the_runner_records_a_timed_out_call():
    """The legacy wall cap is a timer: at 1 s the kernel gives the generation up, the agent is
    asked to abort it and the decision is reported skipped. Its records are the runner's of the
    call it gave up on (legacy_loop.py _run_step): the cap's latency and no usage. The aborted call
    hands back what it had; the kernel discards it and nothing records it."""
    episode = run("wall", 2.0, [Reply(2.0), Reply(0.5)], per_move_cap_s=1.0)
    agent = episode.agent
    assert episode.statuses() == ["skipped_timeout", "applied"]
    assert ("abort", 1) in agent.calls and ("discard", 1) in agent.calls
    assert ("accept", 1) not in agent.calls
    skipped = episode.env.lifecycle[0]
    assert skipped["status"] == "skipped_timeout"
    assert (skipped["observation_id"], skipped["model_latency_s"]) == ("obs-0", 1.0)
    assert skipped["harness_timeout_reason"] == "per_move_cap"
    assert skipped["token_accounting"] is not None
    assert episode.timeline[0][2] == 1.0  # the delay is the call's latency, in ticks
    assert [(row["decision"], row.get("timed_out")) for row in agent.trajectory] == [
        (0, True),
        (1, None),
    ]
    completed = episode.trace.payloads("InferenceCompleted")
    assert [(p["decision"], p["timed_out"], p["stale"]) for p in completed] == [
        (0, True, False),
        (1, False, False),
    ]
    echoes = [
        r["payload"]["data"]
        for r in episode.trace.records
        if r["event"] == "KernelRecord" and r["payload"]["kind"] == "GenerationFinished"
    ]
    assert [echo.generation for echo in echoes] == [1]
    assert episode.env._token_accounting.summary()["responses"] == 2
    reported = next(line for line in episode.logs if " -> " in line)
    assert reported.endswith("-> skipped_timeout TIMEOUT")


def _decisions(payloads: list[dict]) -> list:
    return [(p["decision"], p["timed_out"]) for p in payloads]


def test_a_result_back_after_the_cap_while_the_shell_is_busy_is_never_settled():
    """The end check asked at 1 s takes 1 s; the reply comes back at 1.7 s, during it and after
    the 1.5 s cap. When the check is over, the cap has given the generation up first: the move's
    only records are those of the call given up on, and the late result leaves none."""
    world = World(slow_checks=[(1.0, 1.5, 1.0)])
    episode = run("wall", 3.0, [Reply(1.7), Reply(0.5)], world, per_move_cap_s=1.5)
    agent = episode.agent
    assert episode.statuses() == ["skipped_timeout", "applied"]
    assert _decisions(episode.trace.payloads("InferenceCompleted")) == [(0, True), (1, False)]
    assert [(row["decision"], row.get("timed_out")) for row in agent.trajectory] == [
        (0, True),
        (1, None),
    ]
    assert episode.env._token_accounting.summary()["responses"] == 2
    assert ("accept", 1) not in agent.calls and ("discard", 1) in agent.calls


def test_a_result_back_before_the_cap_while_the_shell_is_busy_is_filed_after_it():
    """Decision 47: the end check asked at 1 s takes 1 s, and the reply comes back at 1.4 s,
    before the 1.5 s cap. It is filed at the reading after the check, and the cap, due by then,
    goes first: the move is skipped. The call did come back in time, so its records are its
    own, as when the shell was idle. (The runner, which checked nothing during a generation,
    applied it.)"""
    world = World(slow_checks=[(1.0, 1.5, 1.0)])
    episode = run("wall", 3.0, [Reply(1.4), Reply(0.5)], world, per_move_cap_s=1.5)
    assert episode.statuses() == ["skipped_timeout", "applied"]
    assert _decisions(episode.trace.payloads("InferenceCompleted")) == [(0, False), (1, False)]
    assert episode.env.lifecycle[0]["model_latency_s"] == 1.4
    assert ("discard", 1) in episode.agent.calls


def test_a_move_over_the_cap_took_the_cap_though_the_shell_saw_it_late():
    """The 1.5 s cap falls due during the end check that runs from 1 s to 2 s. The move is
    charged the cap (kernel/turns.py on_timer), and its record says so, as the runner's bounded
    wait did (legacy_loop.py _run_step)."""
    world = World(slow_checks=[(1.0, 1.5, 1.0)])
    episode = run("wall", 3.0, [Reply(hang=True), Reply(0.5)], world, per_move_cap_s=1.5)
    assert episode.statuses()[0] == "skipped_timeout"
    assert episode.env.lifecycle[0]["model_latency_s"] == 1.5
    assert episode.timeline[0][2] == 1.5
    (timed_out,) = [p for p in episode.trace.payloads("InferenceCompleted") if p["decision"] == 0]
    assert timed_out["latency_s"] == 1.5
    assert "[default] k=0 generation TIMED OUT (>1.5s, per_move_cap) — discarded" in episode.logs


def test_a_later_move_over_the_cap_took_the_cap():
    """The second move starts at 1 s and its 1.5 s cap falls due at 2.5 s: the move is charged
    the cap from its start."""
    episode = run("wall", 4.0, [Reply(0.5), Reply(hang=True), Reply(0.5)], per_move_cap_s=1.5)
    assert episode.statuses()[:2] == ["applied", "skipped_timeout"]
    assert episode.env.lifecycle[1]["model_latency_s"] == 1.5
    (timed_out,) = [p for p in episode.trace.payloads("InferenceCompleted") if p["decision"] == 1]
    assert timed_out["latency_s"] == 1.5


def test_a_generation_given_up_on_a_bound_and_a_spent_think_budget_are_logged():
    """The runner's lines for them (legacy_loop.py _run_step, _mark_budget_exhausted). The deadline
    line gives the seconds from the call's start; the runner's gave the time left plus its
    grace (K-a)."""
    capped = run("wall", 3.0, [Reply(hang=True), Reply(0.5)], per_move_cap_s=1.0)
    assert "[default] k=0 generation TIMED OUT (>1.0s, per_move_cap) — discarded" in capped.logs
    cut = run("wall", 3.0, [Reply(0.5), Reply(hang=True)])
    assert "[default] k=1 generation TIMED OUT (>2.0s, wall_deadline) — discarded" in cut.logs
    spent = run("wall", 5.0, [Reply(1.0), Reply(1.0)], think_budget_s=1.5)
    assert spent.env._stop_reason == "think_budget"
    assert "[think] deliberation budget exhausted: 2.0s >= 1.5s -> forfeit" in spent.logs


def test_a_result_back_at_the_cap_itself_came_too_late():
    """The reply comes back at 1.5 s, as the 1.5 s cap falls due, during the end check asked at
    1 s, which takes 1 s. The kernel handles a bound due at the same time first, so the result
    came too late: the move's records are those of the call given up on."""
    world = World(slow_checks=[(1.0, 1.5, 1.0)])
    episode = run("wall", 3.0, [Reply(1.5), Reply(0.5)], world, per_move_cap_s=1.5)
    assert episode.statuses() == ["skipped_timeout", "applied"]
    assert _decisions(episode.trace.payloads("InferenceCompleted")) == [(0, True), (1, False)]


def test_the_timed_out_lines_give_the_bounds_as_the_runner_takes_them():
    """The runner's cap is at least 0.1 s, its watchdog the configured seconds, whatever float
    the start and the due time make of them (legacy_loop.py _move_timeout); the deadline line gives
    the seconds from the call's start, to two places."""
    capped = run("wall", 3.0, [Reply(hang=True), Reply(0.5)], per_move_cap_s=0.05)
    assert "[default] k=0 generation TIMED OUT (>0.1s, per_move_cap) — discarded" in capped.logs
    replies = [Reply(0.1, CLICK, output_tokens=100), Reply(hang=True)]
    timing = RunnerTimingConfig(provider_watchdog_s=0.3)
    watched = run("token", 20.0, replies, tokens_per_frame=100, timing=timing)
    line = "[default] k=1 generation TIMED OUT (>0.3s, provider_watchdog) — discarded"
    assert line in watched.logs
    cut = run("wall", 1.0, [Reply(0.25), Reply(hang=True)], fps=10.0)
    assert "[default] k=1 generation TIMED OUT (>0.7s, wall_deadline) — discarded" in cut.logs


def test_a_call_the_tasks_end_gave_up_on_is_timed_to_two_places():
    """The end checks take 70 ms each, so the readings are not round; the end, seen by the check
    asked at 1.93 s, is at 1.93 s, and the call that started at 1 s ran 0.93 s by then."""
    world = World(ends_at=1.45, end_reason="won", check_s=0.07)
    episode = run("wall", 5.0, [Reply(0.5, CLICK), Reply(hang=True)], world)
    completed = episode.trace.payloads("InferenceCompleted")
    assert [(p["decision"], p["latency_s"]) for p in completed] == [(0, 0.5), (1, 0.93)]


def test_a_generation_given_up_on_at_the_tasks_end_did_not_time_out():
    """The task ends at 1.5 s and the end check at 2 s sees it: the call is given up on, but no
    bound came, so nothing is logged or recorded as timed out."""
    episode = run("wall", 3.0, [Reply(hang=True)], World(ends_at=1.5, end_reason="won"))
    assert episode.env._termination["stop_reason"] == "won"
    assert not any("TIMED OUT" in line for line in episode.logs)
    completed = episode.trace.payloads("InferenceCompleted")
    assert [(p["decision"], p["stale"], p["timed_out"]) for p in completed] == [(0, True, False)]


class _ToolAgent(Agent):
    """With a tool runtime a result needs its inspection proof (episode_records.result_invalid)."""

    tool_runtime = object()


@pytest.mark.parametrize(
    ("world", "action", "reason", "latency"),
    [
        (World(ends_at=1.5, end_reason="won"), CLICK, "task_end", 1.0),
        (World(release_failures={0}), Action.hold_key("a", 0.75), "invalid_run", 0.25),
    ],
    ids=["task-end", "invalid"],
)
def test_a_generation_the_episode_ended_on_is_recorded_as_a_call_given_up_on(
    world, action, reason, latency
):
    """N10: the second call starts at 1 s and never answers. The task's end, seen by the end
    check at 2 s, or the key held from 0.5 s failing to come up at 1.25 s ends the episode
    during it. The call leaves a record of one given up on: an InferenceCompleted and a
    trajectory row, named for what ended the episode. N34: that record is synthesized, so the
    episode's token totals leave it out and stay those of the call accepted, complete. The
    decision is not reported, and the agent is asked to abort the call and nothing more:
    closing it discards the generation no one settled (harness.py close)."""
    usage = {"usage": USAGE, "provider_protocol_family": "chat_completions"}
    replies = [Reply(0.5, action, fields=usage), Reply(hang=True)]
    episode = run("wall", 3.0, replies, world)
    assert episode.statuses() == ["applied"]
    completed = episode.trace.payloads("InferenceCompleted")
    assert [(p["decision"], p["stale"], p["timed_out"], p["latency_s"]) for p in completed] == [
        (0, False, False, 0.5),
        (1, True, False, latency),
    ]
    assert "harness_abort_reason" not in completed[0]
    assert completed[1]["harness_abort_reason"] == reason
    rows = [(row["decision"], row.get("harness_abort_reason")) for row in episode.agent.trajectory]
    assert rows == [(0, None), (1, reason)]
    summary = episode.env._token_accounting.summary()
    assert summary["responses"] == 1 and summary["provider_usage_missing_reasons"] == {}
    assert summary["counts"]["provider_output_tokens"]["complete"] is True
    assert summary["counts"]["provider_output_tokens"]["total"] == 20
    calls = [call for call in episode.agent.calls if call[1] == 2]
    assert calls == [("abort", 2)]


def test_a_call_the_tasks_end_gave_up_on_ran_until_the_end_signal():
    """The end check asked at 2 s takes 0.3 s and sees the task over: the episode ended at 2 s
    (decision 32), and the call that started at 1 s ran 1 s by then, however late the shell got
    to it."""
    world = World(ends_at=1.5, end_reason="won", slow_checks=[(2.0, 2.5, 0.3)])
    episode = run("wall", 3.0, [Reply(0.5), Reply(hang=True)], world)
    assert episode.env._termination["end_s"] == 2.0
    completed = episode.trace.payloads("InferenceCompleted")
    assert [(p["decision"], p["latency_s"]) for p in completed] == [(0, 0.5), (1, 1.0)]


def test_a_settled_result_the_tasks_end_gave_up_on_is_not_logged_as_timed_out():
    """The result comes back at 1.9 s, during the end check asked at 1 s, which takes 1 s and
    sees the task over. Settled, its records are its own; no bound came, so nothing is
    logged about one."""
    world = World(ends_at=0.5, end_reason="won", slow_checks=[(1.0, 1.5, 1.0)])
    episode = run("wall", 3.0, [Reply(1.9)], world)
    assert episode.env._termination["stop_reason"] == "won"
    completed = episode.trace.payloads("InferenceCompleted")
    assert [(p["decision"], p["stale"]) for p in completed] == [(0, False)]
    assert not any("settled past" in line or "TIMED OUT" in line for line in episode.logs)


def test_a_generation_the_tasks_end_gave_up_on_needs_no_inspection_proof():
    """The record of a call given up on has no model decision to inspect: the episode the task
    ended stays valid, as one the deadline ended does (episode_records.result_invalid)."""
    world = World(ends_at=0.5, end_reason="won")
    episode = run("wall", 3.0, [Reply(hang=True)], world, agent_type=_ToolAgent)
    assert episode.invalid is None
    assert episode.env._termination["stop_reason"] == "won"
    assert [row.get("harness_abort_reason") for row in episode.agent.trajectory] == ["task_end"]


@pytest.mark.parametrize(
    ("seconds", "row", "completed"),
    [(2.9, (0, None), (0, False, 2.9)), (3.2, (0, True), (0, True, 3.0))],
    ids=["before-the-deadline", "after-it"],
)
def test_a_result_back_while_the_shell_was_busy_at_the_deadline_is_recorded_as_it_came(
    seconds, row, completed
):
    """The end check asked at 2 s takes 1.5 s, and the reply comes back during it; when the
    check is over, the 3 s deadline goes first (decision 47), ends the episode and the result is
    discarded. One back before the deadline is settled and its records are its own, as when the
    shell was idle; one back after it is never settled, and the records are those of the call
    the deadline gave up on (decision 48). (That one reaches the kernel only after the end.)"""
    world = World(slow_checks=[(2.0, 2.5, 1.5)])
    episode = run("wall", 3.0, [Reply(seconds)], world)
    assert episode.statuses() == []
    rows = episode.agent.trajectory
    assert [(r["decision"], r.get("timed_out")) for r in rows] == [row]
    payloads = episode.trace.payloads("InferenceCompleted")
    assert [(p["decision"], p["timed_out"], p["latency_s"]) for p in payloads] == [completed]
    if seconds < 3.0:
        assert episode.agent.calls == [("discard", 1)]
    assert episode.env._termination["stop_reason"] == "time_limit"


def test_a_generation_the_deadline_gave_up_on_is_recorded_as_the_runner_records_it():
    """Decision 48: the call the 3 s deadline cut leaves the runner's record of a call it gave up on
    (legacy_loop.py _run_step): an InferenceCompleted, a trajectory row and a token record without
    usage, so the episode's output tokens are not complete. The decision is not reported (K-a), and
    the agent's own usage is not touched (decision 37)."""
    episode = run("wall", 3.0, [Reply(0.5, output_tokens=40), Reply(hang=True)])
    assert episode.statuses() == ["applied"]
    assert len(episode.env.lifecycle) == 1
    completed = episode.trace.payloads("InferenceCompleted")
    assert [(p["decision"], p["timed_out"], p["latency_s"]) for p in completed] == [
        (0, False, 0.5),
        (1, True, 2.0),
    ]
    rows = [
        (row["decision"], row.get("timed_out"), row.get("harness_timeout_reason"))
        for row in episode.agent.trajectory
    ]
    assert rows == [(0, None, None), (1, True, "wall_deadline")]
    summary = episode.env._token_accounting.summary()
    assert summary["responses"] == 2
    assert summary["provider_usage_missing_reasons"]["generation_aborted_at_wall_timeout"] == 1
    assert summary["counts"]["provider_output_tokens"]["complete"] is False
    assert summary["counts"]["provider_output_tokens"]["known_total"] == 40


def test_a_paused_decision_lands_on_its_frame_and_moves_one_frame_on():
    episode = run("token", 2.0, [Reply(0.25), Reply(0.25)])
    assert episode.statuses() == ["applied", "applied"]
    assert episode.timeline == [(0, 0, 0.0, 0.0, CLICK), (1, 1, 0.0, 1.0, CLICK)]
    assert episode.env._termination["stop_reason"] == "tick_limit"
    assert episode.env._termination["end_s"] == 2.0


def test_a_frame_and_a_result_are_let_go_once_the_decision_is_over():
    """Only the decision in hand keeps its frame: one a long episode had kept would grow with
    it."""
    episode = run("wall", 5.0, [Reply(0.5)] * 5)
    assert episode.statuses() == ["applied"] * 5
    gc.collect()
    assert [ref() is not None for ref in episode.env.observations] == [False] * 4 + [True]


def test_without_a_snapshot_reader_the_termination_has_no_snapshot_time():
    episode = run("wall", 1.0, [Reply(0.5)], World(snapshotter=False))
    assert episode.env._termination == {
        "stop_reason": "time_limit",
        "end_s": 1.0,
        "snapshot_s": None,
        "snapshot_delay_s": None,
    }
    assert episode.env._termination_snapshot_captured is False
    assert episode.invalid is None


def test_the_snapshot_is_timed_when_its_read_returned():
    """The read at the 1 s deadline takes 0.5 s; the termination record has when it returned and
    how late that was (legacy_loop.py _finalize)."""
    episode = run("wall", 1.0, [Reply(0.5)], World(snapshot_s=0.5))
    assert episode.env._termination == {
        "stop_reason": "time_limit",
        "end_s": 1.0,
        "snapshot_s": 1.5,
        "snapshot_delay_s": 0.5,
    }
    assert episode.env._active_end_unix_s == ORIGIN + 1.0


def test_a_snapshot_read_over_its_budget_invalidates_the_run():
    """The freeze reads the snapshot within 16 s (decision 3); a read of 17 s fails."""
    episode = run("wall", 1.0, [Reply(0.5)], World(snapshot_s=17.0))
    assert episode.invalid["reason"] == "evaluation_snapshot_failed"
    assert "snapshot read took longer than 16 s" in episode.env._termination_snapshot_error
    assert episode.env._termination["snapshot_s"] is None


def test_the_token_end_time_is_kept_to_nine_places():
    """At 29.97 fps a 15 s horizon is 449.55 ticks, 14.999999999999998 s back in seconds; to
    nine places it is the time limit, which reporting counts as a timeout (decision 4)."""
    reply = Reply(0.0, CLICK, output_tokens=10**6)
    episode = run("token", 15.0, [reply], fps=29.97, tokens_per_frame=100)
    assert episode.statuses() == ["not_applied_tick_limit"]
    assert episode.env._termination["end_s"] == 15.0


def test_the_wall_delay_is_the_latency_in_environment_ticks():
    episode = run("wall", 2.0, [Reply(0.5), Reply(0.25)], fps=2.0)
    assert [entry[2] for entry in episode.timeline] == [1.0, 0.5]


def test_the_frames_a_token_decision_crossed_are_counted_as_dropped():
    """250 tokens at B = 100 are 2.5 frames: the next frame is the third after the observed one,
    and the two in between were dropped."""
    replies = [Reply(0.0, CLICK, output_tokens=250), Reply(0.0, CLICK, output_tokens=100)]
    episode = run("token", 5.0, replies, tokens_per_frame=100)
    assert episode.env.lifecycle[0]["dropped_intermediate_frames"] == 2
    assert episode.env._dropped_intermediate_frames == sum(
        record["dropped_intermediate_frames"] for record in episode.env.lifecycle
    )


def test_a_render_that_commits_no_frame_is_recorded_as_none():
    """A landing on the next frame asks for a render; a task without a renderer commits none."""
    episode = run(
        "token",
        2.0,
        [Reply(0.0, CLICK, output_tokens=100)],
        World(renders=False),
        tokens_per_frame=100,
    )
    assert "render" in [call for call, _, _ in episode.env.calls]
    assert episode.env.lifecycle[0]["post_action_rendered"] is False


class _Uninterruptible(Agent):
    """Its call runs its course whatever the abort, as a Decision call does."""

    def abort(self, generation=None) -> None:
        self.calls.append(("abort", generation))


def test_main_waits_for_the_worker_of_a_generation_given_up_on():
    """The deadline gives up a 5 s call at 2 s; main()'s drain waits for its worker before it
    closes the agent (runner.py main; shell.py _drain_abandoned)."""
    episode = run("wall", 2.0, [Reply(5.0)], agent_type=_Uninterruptible)
    assert "[default] draining 1 abandoned generate worker(s), <= 30s" in episode.logs
    assert not any("STILL inside" in line for line in episode.logs)
    assert episode.clock.monotonic() == 5.0
