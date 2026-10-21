"""Each kernel command as run_episode runs it against the environment and the agent
(evaluation/shell.py), on the fake clock.

Times are semantic: seconds since the episode origin on the wall clock, environment ticks on the
token clock. Every expected value follows from the kernel's turn rules (kernel/turns.py) and the
scripted world (tests/shell/world.py).
"""

from __future__ import annotations

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.entity.config import DecisionBudget

from .world import CLICK, ORIGIN, Reply, World, run

HOLD = Action.hold_key("a", 1.25)
WAIT = Action.wait(0.5)


def _calls(episode) -> list[tuple]:
    return [(call, time) for call, time, _ in episode.env.calls]


def test_a_wall_episode_runs_each_decision_on_the_environment_and_the_agent():
    """Three decisions of 0.5 s each at 1 fps, then the 3 s deadline. Between decisions the
    environment is asked whether the episode ended on the frame grid, besides the loop top and
    right before and after execution; a wait is the kernel's own timer and never reaches the
    environment; the hold's keys are released by the environment's timer at the time the kernel
    gave."""
    episode = run("wall", 3.0, [Reply(0.5), Reply(0.5, HOLD), Reply(0.5, WAIT)])
    env, agent = episode.env, episode.agent
    assert _calls(episode) == [
        ("check_end", 0.0),
        ("capture", 0.0),
        ("check_end", 0.5),
        ("execute", 0.5),
        ("check_end", 0.5),
        ("check_end", 1.0),  # the periodic check
        ("check_end", 1.0),  # the loop top
        ("capture", 1.0),
        ("check_end", 1.5),
        ("execute", 1.5),
        ("check_end", 1.5),
        ("check_end", 2.0),
        ("check_end", 2.0),
        ("capture", 2.0),
        ("check_end", 2.5),  # right before the wait
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


def test_a_token_episode_advances_releases_and_renders_through_the_environment():
    """2 fps, B = 100 per frame, 2 s: the horizon is tick 4. A decision's n tokens are n / 100
    frames, one tick each; the clock moves in steps of at most one tick, cut where the 0.5 s
    hold's lease ends (tick 2), with an end check after each step. A landing on the next frame
    commits its pixels with a render. The last decision's action would land at the horizon."""
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
        ("check_end", 1, None),
        ("execute", 1, None),
        ("check_end", 1, None),
        ("render", 1, None),
        ("check_end", 1, None),
        ("capture", 1, None),
        ("step_clock", 1, 0.5),
        ("check_end", 1.5, None),
        ("check_end", 1.5, None),
        ("execute", 1.5, None),
        ("check_end", 1.5, None),
        ("step_clock", 1.5, 0.5),
        ("release_keys", 2, ("a",)),
        ("check_end", 2, None),
        ("check_end", 2, None),
        ("capture", 2, None),
        ("step_clock", 2, 1.0),
        ("check_end", 3, None),
        ("check_end", 3, None),
        ("execute", 3, None),
        ("check_end", 3, None),
        ("render", 3, None),
        ("check_end", 3, None),
        ("capture", 3, None),
        ("step_clock", 3, 1.0),
        ("check_end", 4, None),
        ("check_end", 4, None),
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


def test_an_action_the_guest_refuses_is_reported_with_its_reason():
    episode = run("wall", 2.0, [Reply(0.5), Reply(0.5)], World(rejections={0: "bad key"}))
    assert episode.statuses() == ["rejected", "applied"]
    assert episode.agent.outcomes[0].detail == "bad key"
    assert episode.env.lifecycle[0]["execution_result"] == {"error": "bad key"}
    assert episode.invalid is None


def test_a_done_reply_lands_as_a_wait_and_its_record_says_why():
    """The loop has no terminal actions: the kernel lands a DONE as a wait, and the lifecycle
    record keeps the fallback protocol the runner writes (runner.py _decide)."""
    episode = run(
        "wall",
        1.0,
        [Reply(0.5, Action(ActionType.DONE), fields={"action_protocol": {"source": "json"}})],
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


def test_a_move_over_the_per_move_cap_is_skipped_and_its_late_result_discarded():
    """The legacy wall cap is a timer: at 1 s the kernel gives the generation up, the agent is
    asked to abort it and the decision is reported skipped. The aborted call hands back what it
    had; the kernel discards it and the trace still has its InferenceCompleted."""
    episode = run("wall", 3.0, [Reply(2.0), Reply(0.5)], per_move_cap_s=1.0)
    agent = episode.agent
    assert episode.statuses() == ["skipped_timeout", "applied"]
    assert ("abort", 1) in agent.calls and ("discard", 1) in agent.calls
    assert ("accept", 1) not in agent.calls
    skipped = episode.env.lifecycle[0]
    assert (skipped["status"], skipped["observation_id"], skipped["model_latency_s"]) == (
        "skipped_timeout",
        "obs-0",
        None,
    )
    late = [
        payload
        for payload in episode.trace.payloads("InferenceCompleted")
        if payload["decision"] == 0
    ]
    assert [(payload["stale"], payload["latency_s"]) for payload in late] == [(True, 1.0)]


def test_a_paused_decision_lands_on_its_frame_and_moves_one_frame_on():
    episode = run("token", 2.0, [Reply(0.25), Reply(0.25)])
    assert episode.statuses() == ["applied", "applied"]
    assert episode.timeline == [(0, 0, 0.0, 0.0, CLICK), (1, 1, 0.0, 1.0, CLICK)]
    assert episode.env._termination["stop_reason"] == "tick_limit"
    assert episode.env._termination["end_s"] == 2.0


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
