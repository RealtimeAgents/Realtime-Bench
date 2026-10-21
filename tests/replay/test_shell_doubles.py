"""The replay environments' primitives for the kernel's shell (evaluation/shell.py), before any
shell calls them: FakeEnvironment (fake_env.py) and TokenEnvironment (token_env.py) answer as
GuestEnvironment does and log the calls the way their legacy ones are logged."""

from __future__ import annotations

from realtime_framework.entity.actions import Action

from ..fake_clock import FakeClock
from .clock_task import ClockTask
from .fake_env import FakeEnvironment
from .frames import FrameSource
from .token_env import TokenEnvironment


def _answers(*answers):
    answers = iter(answers)
    return lambda *env: next(answers)


FIELDS_OF_EVERY_CALL = {"call", "decision", "finalizing"}


def _logged(env, call):
    return [
        {key: value for key, value in entry.items() if key not in FIELDS_OF_EVERY_CALL}
        for entry in env.calls
        if entry["call"] == call
    ]


def _token(**kwargs):
    task = ClockTask(1.0)
    frames = FrameSource(task, text_observation=False, fps=1.0)
    return TokenEnvironment(frames, task, max_minutes=1.0, **kwargs), task


def test_the_token_environment_s_end_check_logs_only_an_end():
    env, _ = _token(stop_check=_answers(False, "task_success"))
    assert env.check_end() == (False, None)
    assert env.check_end() == (True, "task_success")
    assert _logged(env, "env.check_end") == [{"reason": "task_success"}]


def test_the_token_environment_logs_a_clock_step_and_a_key_release():
    env, task = _token()
    env.execute(Action.hold_key("a", 5.0))
    env.step_clock(1.0)
    env.release_keys(("a",))
    assert _logged(env, "env.step_clock") == [{"step": 1.0, "from_tick": 0, "to_tick": 1}]
    assert _logged(env, "env.release_keys") == [{"keys": ["a"]}]
    assert [event["event"] for event in task.events] == ["keyDown", "advance", "keyUp"]


def test_the_wall_environment_s_end_check_asks_the_task_and_logs_only_an_end():
    clock = FakeClock()
    task = ClockTask(1.0)
    env = FakeEnvironment(
        FrameSource(task, text_observation=False, fps=1.0),
        clock,
        fps=1.0,
        max_minutes=1.0,
        reply_latencies=[],
        stop_check=_answers(False, "task_success"),
    )
    assert env.check_end() == (False, None)
    assert env.check_end() == (True, "task_success")
    assert env._episode_over and env._terminal_reason == "task_success"
    assert _logged(env, "env.check_end") == [{"reason": "task_success"}]
