"""runner.main() on the wall clock writes the termination its decision loop latched.

The real main(), run_stream and GuestEnvironment run on a fake wall clock; the desktop, the guest,
the task and the agent are fakes. The task's end check and its evaluation snapshot take fake
time, and closing the agent (runner.py:1912) moves the clock past the deadline, so a world time
read after teardown is past the horizon although the episode ended early. main() takes
world_time_s from the termination record (runner.py:2095), hands the record to the run metrics
(runner.py:2104) and writes it into run_summary.json (runner.py:2192).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import types
from unittest import mock

import pytest

from realtime_framework.clock import Clocks, SystemClock
from realtime_framework.entity.actions import Action
from realtime_framework.entity.harness import StreamingAgent
from realtime_framework.evaluation import runner

FPS = 1
MAX_MINUTES = 0.0625
HORIZON_S = MAX_MINUTES * 60
QUERY_S = 0.25  # one call of the task's episode_over()
LATENCY_S = 0.5  # the decision's generation
SNAPSHOT_S = 0.5  # one call of the task's evaluation_snapshot()
CLOSE_S = HORIZON_S  # one agent.close()

# The kernel asks the task at most once a frame and once after every action (kernel/turns.py
# _check_end): before the decision and after the click lands, but not before the click is
# dispatched, since this frame was read. The one after the landing reports the end, which is when
# it was asked; the snapshot is frozen once it returned.
END_S = QUERY_S + LATENCY_S
FINALIZED_S = END_S + QUERY_S
SNAPSHOT_AT_S = FINALIZED_S + SNAPSHOT_S  # when evaluation_snapshot() returned


class FakeClock(SystemClock):
    """The episode clocks: time() reads the fake clock and sleep() advances it; anything else is
    the system clock."""

    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeGuest:
    def __init__(self):
        self.clicks = []

    def wait_until_ready(self, max_wait):
        return True

    def screenshot(self):
        return b""

    def execute_action(self, action):
        self.clicks.append(action)
        return {}


class FakeTask:
    """A realtime task whose end check reports ``reason`` once the guest has been clicked."""

    category = "realtime"  # main() takes the wall origin from start_episode() (runner.py:1744)

    def __init__(self, clock, guest, reason, solved):
        self.clock, self.guest, self.reason, self.solved = clock, guest, reason, solved
        self.origin = None

    def configure_seed(self, seed):
        pass

    def description(self):
        return "Click once."

    def configure_clock(self, clock_mode, fps, tokens_per_frame):
        pass

    def provision(self):
        return None

    def clock_snapshot(self):
        return {"fps": FPS}

    def reset_episode(self):
        return {}

    def start_episode(self):
        self.origin = self.clock.now
        return {"clock_origin": self.origin}

    def episode_over(self):
        self.clock.now += QUERY_S
        return self.reason if self.guest.clicks else False

    def evaluation_snapshot(self):
        self.clock.now += SNAPSHOT_S
        return {"kind": "fake", "state": {"clicks": len(self.guest.clicks)}}

    def evaluate_snapshot(self, snapshot):
        return {"resolved": self.solved, "evaluation_version": 1}


class FakeAgent(StreamingAgent):
    def __init__(self, clock):
        self.clock = clock

    def initialize(self, task_instruction, initial_observation):
        pass

    def discard_pending_observations(self):
        pass

    def stream_prefill(self, observation):
        pass

    def stream_generate(self, tick):
        self.clock.now += LATENCY_S
        return {"action": Action.click(100, 100), "latency": LATENCY_S}

    def close(self):
        self.clock.now += CLOSE_S


DESKTOP = types.SimpleNamespace(
    build_image_if_needed=lambda: None,
    start_container=lambda config: None,
    _start_screensaver_keepalive=lambda: None,
    _start_screen_recording=lambda run_dir: False,
    _stop_screen_recording=lambda run_dir, frames_dir, fps: None,
)


def run_main(tmp_path, task, agent, clock, guest) -> dict:
    """Run main() for ``task`` and return its run_summary.json."""

    def set_up_agent(setup, *args, **kwargs):
        setup.agent = agent
        setup.stage = "environment_setup"

    argv = [
        "runner", "--task", "fake_task", "--model", "fake-model", "--adapter", "anthropic_messages",
        "--clock", "wall", "--fps", str(FPS), "--max-minutes", str(MAX_MINUTES),
        "--run-id", "run", "--results-dir", str(tmp_path),
    ]  # fmt: skip
    with (
        mock.patch.object(sys, "argv", argv),
        # main() writes SANDBOX_GRAPHICS_MODE and SCREEN_RECORDING_ENCODER into os.environ.
        mock.patch.dict(os.environ),
        mock.patch.object(runner, "episode_clocks", lambda clock_mode, fps: Clocks(clock, clock)),
        mock.patch.object(runner, "R", DESKTOP),
        mock.patch.object(runner, "GuestClient", lambda **_: guest),
        mock.patch.object(runner.custom_tasks, "get", lambda task_id: task),
        mock.patch.object(runner, "set_up_agent", set_up_agent),
        # An unscored cell exits 1 after the summary is written (runner.py:2297-2298).
        contextlib.suppress(SystemExit),
    ):
        runner.main()
    return json.loads((tmp_path / "run" / "run_summary.json").read_text())


@pytest.mark.parametrize(
    ("reason", "solved", "outcome"),
    [("deadline_expired", False, "fail"), ("task_success", True, "pass")],
)
def test_main_reports_the_termination_the_loop_latched(tmp_path, reason, solved, outcome):
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, reason, solved)
    summary = run_main(tmp_path, task, FakeAgent(clock), clock, guest)

    assert clock.now - task.origin >= HORIZON_S  # teardown ran past the deadline
    assert summary["invalid_run"] is None
    assert summary["termination"] == {
        "stop_reason": reason,
        "end_s": END_S,
        "snapshot_s": SNAPSHOT_AT_S,
        "snapshot_delay_s": SNAPSHOT_AT_S - END_S,
    }
    assert summary["world_time_s"] == END_S
    assert "run_metrics_error" not in summary
    assert summary["outcome"] == outcome
    assert summary["normalized_time_to_success"] == (
        round(END_S / HORIZON_S, 9) if solved else None
    )


# Two functions in turn: a traceback folds a run of identical frames into one line.
def _raise_deep(depth):
    if depth == 0:
        raise RuntimeError("budget rejected")
    return _raise_deeper(depth - 1)


def _raise_deeper(depth):
    return _raise_deep(depth)


class FailingAgent(FakeAgent):
    """Mapping its output cap to the transport's, which the shell does on its own thread as it
    starts a generation (episode_records.decision_budget), raises from deep in a call stack, as a
    driver's can: the traceback runs far past the invalid-run detail's 2000 characters and names
    the exception last."""

    def transport_output_token_cap(self, provider_cap):
        _raise_deep(40)


def test_an_exception_that_escapes_the_loop_keeps_its_cause(tmp_path):
    """The first decision raises after the loop's first end check: the run is invalid, its detail
    ends with the exception and the summary's error holds the whole traceback."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "deadline_expired", False)
    summary = run_main(tmp_path, task, FailingAgent(clock), clock, guest)

    invalid = summary["invalid_run"]
    assert (invalid["category"], invalid["reason"]) == ("runner", "run_exception")
    error = summary["error"]
    assert error.startswith("Traceback (most recent call last):") and len(error) > 2000
    assert error.rstrip().endswith("RuntimeError: budget rejected")
    assert invalid["detail"] == error[-2000:]
    assert summary["outcome"] == "error"
    assert summary["termination"] == {
        "stop_reason": "invalid_run",
        "end_s": QUERY_S,
        "snapshot_s": QUERY_S + SNAPSHOT_S,
        "snapshot_delay_s": SNAPSHOT_S,
    }
