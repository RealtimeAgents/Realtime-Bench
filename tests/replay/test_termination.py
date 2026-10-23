"""How the outcome reads the termination record run_stream latches (evaluation/shell.py _finish;
evaluation/reporting.py episode_outcome).

An unsolved episode that reached the scored horizon is a timeout, whatever the task reported at
its end; one that ended earlier is a fail. On the token clock run_stream asks the task after every
clock step (kernel/turns.py _clock_stepped, shell.py _check_end), so a task whose own clock ends
at the horizon reports its reason before run_stream's tick limit, and that reason is the stop
reason (kernel/core.py Kernel.end, Kernel._finish). The episodes run the openrouter agent in pause
mode against the real GuestEnvironment on a task clock (token_env.py).

An invalid run is an error, whatever its score says.
"""

from __future__ import annotations

import pytest

import realtime_framework.providers.registry as catalog
from realtime_framework.evaluation.reporting import episode_outcome

from .build import build_episode
from .clock_task import ClockTask
from .loop import run_loop
from .scenario import CLICK_AGREE, CLICK_NAME, CLICK_SUBMIT, TASK_INSTRUCTION
from .stub_openrouter import OpenRouterStub
from .test_openrouter import MODEL_KEY, ROUTE, openrouter_env  # noqa: F401 (fixture)
from .test_runner_frame_rates import reply
from .token_env import TokenEnvironment

# 3 s at 1 fps: in pause mode decisions observe ticks 0, 1 and 2, and each moves the world on
# one frame after its action.
MAX_MINUTES = 0.05
# A web task whose own clock closes the episode reports this (e.g.
# realtime_tasks/web/rt_accessible_trip_dispatch/environment/engine.py:405-409).
HANDOVER = "Dispatch handover completed."
CLICKS = [CLICK_NAME, CLICK_AGREE, CLICK_SUBMIT]


@pytest.mark.usefixtures("openrouter_env")
@pytest.mark.parametrize(("ends_at", "outcome"), [(3.0, "timeout"), (2.0, "fail")])
def test_task_end_at_the_horizon_is_a_timeout(monkeypatch, tmp_path, ends_at, outcome):
    """The task ends itself once its world time reaches ``ends_at``: at the 3 s horizon the
    unsolved run is a timeout, a frame earlier a fail."""
    entry = catalog.get(MODEL_KEY)
    config = catalog.to_model_config(
        MODEL_KEY, fps=1.0, reasoning_effort=entry.defaults["reasoning_effort"]
    )
    task = ClockTask(1.0)

    def episode_over():
        return HANDOVER if task.world_time() >= ends_at else False

    decisions = int(ends_at)
    with OpenRouterStub(
        [reply(click, 80) for click in CLICKS[:decisions]], route=ROUTE
    ) as stub:
        monkeypatch.setenv("OPENROUTER_BASE_URL", stub.base_url)
        episode = build_episode(
            config,
            tmp_path,
            task=task,
            instruction=TASK_INSTRUCTION,
            model_key=MODEL_KEY,
            clock_mode="token",
            tokens_per_frame=None,
        )
        env = TokenEnvironment(
            episode.frames, task, max_minutes=MAX_MINUTES, stop_check=episode_over
        )
        run_loop(episode, env)
    assert stub.unscripted == 0 and stub.unused_replies == 0

    assert env._termination == {
        "stop_reason": HANDOVER,
        "end_s": ends_at,
        "snapshot_s": ends_at,
        "snapshot_delay_s": 0.0,
    }
    # The fields of run_summary.json that episode_outcome reads (runner.py main).
    summary = {
        "invalid_run": None,
        "solved": False,
        "stop_reason": env._stop_reason,
        "termination": env._termination,
        "world_time_s": env._termination["end_s"],
        "max_minutes": MAX_MINUTES,
    }
    assert episode_outcome(summary)["outcome"] == outcome


def test_task_end_at_the_horizon_is_a_timeout_when_its_seconds_are_inexact():
    """The horizon is compared with the record's end at the nine decimals the record keeps
    (reporting._reached_horizon): 60 * 0.13 minutes is 7.800000000000001 in floating point, and a
    task that ends itself at 7.8 s reached the horizon."""
    termination = {
        "stop_reason": HANDOVER,
        "end_s": 7.8,
        "snapshot_s": 7.8,
        "snapshot_delay_s": 0.0,
    }
    summary = {
        "invalid_run": None,
        "solved": False,
        "stop_reason": HANDOVER,
        "termination": termination,
        "world_time_s": 7.8,
        "max_minutes": 0.13,
    }
    assert episode_outcome(summary)["outcome"] == "timeout"


@pytest.mark.parametrize(
    ("stop_reason", "world_time_s", "outcome"),
    [
        ("time_limit", 3.0, "timeout"),
        ("deadline_expired", 3.0, "timeout"),
        ("deadline_expired", 4.0, "timeout"),
        ("deadline_expired", 2.5, "fail"),
    ],
)
def test_a_summary_without_a_termination_record(stop_reason, world_time_s, outcome):
    """A summary written before the record keeps the outcome it was written with: the horizon's
    stop reason, or a world_time_s at or past the horizon, is a timeout."""
    summary = {
        "invalid_run": None,
        "solved": False,
        "stop_reason": stop_reason,
        "world_time_s": world_time_s,
        "max_minutes": MAX_MINUTES,
    }
    assert episode_outcome(summary)["outcome"] == outcome


@pytest.mark.parametrize("solved", [None, True, False])
def test_an_invalid_run_is_an_error_of_its_own_category_whatever_its_score(solved):
    """runner.main clears the score of an invalid run; the outcome does not rely on that, and
    names what invalidated the run."""
    summary = {
        "invalid_run": {"category": "provider", "reason": "generation_error", "detail": "boom"},
        "solved": solved,
        "stop_reason": "invalid_run",
        "termination": None,
        "world_time_s": 3.0,
        "max_minutes": MAX_MINUTES,
    }
    assert episode_outcome(summary) == {
        "outcome": "error",
        "error_category": "provider",
        "error_reason": "generation_error",
    }
