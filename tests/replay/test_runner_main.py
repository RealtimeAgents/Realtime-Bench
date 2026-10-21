"""runner.main() on the wall clock writes the termination its decision loop latched.

The real main(), run_stream and GuestEnvironment run on a fake wall clock; the desktop, the guest,
the task and the agent are fakes. The task's end check and its evaluation snapshot take fake
time, and closing the agent (runner.py main) moves the clock past the deadline, so a world time
read after teardown is past the horizon although the episode ended early. main() takes
world_time_s from the termination record, hands the record to the run metrics and writes it into
run_summary.json (runner.py main).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import types
from unittest import mock

import pytest

from realtime_framework.clock import Clocks, FrameClock, SystemClock
from realtime_framework.entity.actions import Action
from realtime_framework.entity.harness import StreamingAgent
from realtime_framework.entity.records import TraceEvent, validate_episode_trace
from realtime_framework.evaluation import runner
from realtime_framework.evaluation.agent_trace import AgentTraceSink
from realtime_framework.evaluation.kernel.turns import TurnConfig

from ..shell.test_inputs import replay_inputs

S = 1_000_000_000  # nanoseconds
FPS = 1
MAX_MINUTES = 0.0625
HORIZON_S = MAX_MINUTES * 60
QUERY_S = 0.25  # one call of the task's episode_over()
LATENCY_S = 0.5  # the decision's generation
SNAPSHOT_S = 0.5  # one call of the task's evaluation_snapshot()
CLOSE_S = HORIZON_S  # one agent.close()

# The kernel asks the task at the loop top once a frame, right before an action lands when the
# clock has moved on since the last read, and once after every action (kernel/turns.py
# _read_already): before the decision, before the click is dispatched and after it. The one
# after the click reports the end, which is when it was asked; the snapshot is frozen once it
# returned.
END_S = QUERY_S + LATENCY_S + QUERY_S
FINALIZED_S = END_S + QUERY_S
SNAPSHOT_AT_S = FINALIZED_S + SNAPSHOT_S  # when evaluation_snapshot() returned


class FakeClock(SystemClock):
    """The episode clocks: time() and time_ns() read the fake clock and sleep() advances it;
    anything else is the system clock."""

    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now

    def time_ns(self):
        return round(self.now * S)

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

    category = "realtime"  # main() takes the wall origin from start_episode() (runner.py main)

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


def run_main(
    tmp_path, task, agent, clock, guest, *, agent_trace=False, seen=None, clock_mode="wall"
) -> dict:
    """Run main() for ``task`` and return its run_summary.json. ``clock`` is the operational
    clock; on the wall clock it is world time too, and under the token clock (in pause mode)
    world time is a FrameClock. With ``agent_trace`` the agent gets an AgentTraceSink on the
    episode trace, as ``agent.sink``, as set_up_agent makes one. ``seen`` gets the episode clocks
    main() makes, the keyword arguments it calls set_up_agent with, and the environment and
    keyword arguments it calls run_stream with."""
    seen = {} if seen is None else seen

    def episode_clocks(clock_mode, fps):
        semantic = FrameClock(fps) if clock_mode == "token" else clock
        seen.setdefault("clocks", []).append(Clocks(semantic, clock))
        return seen["clocks"][-1]

    def set_up_agent(setup, *args, **kwargs):
        seen["set_up_agent"] = kwargs
        setup.agent = agent
        if agent_trace:
            pack = kwargs["run_dir"] / "episode_blobs.pack"
            sink = AgentTraceSink(kwargs["episode_trace"], pack, clock=kwargs["clock"])
            setup.agent_trace = agent.sink = sink
        setup.stage = "environment_setup"

    def run_stream(adapter, env, **kwargs):
        seen["run_stream"] = {"env": env, **kwargs}
        return real_run_stream(adapter, env, **kwargs)

    real_run_stream = runner.run_stream
    clock_args = ["--clock", clock_mode]
    if clock_mode == "token":
        clock_args += ["--tokens-per-frame", "unlimited"]
    argv = [
        "runner", "--task", "fake_task", "--model", "fake-model", "--adapter", "anthropic_messages",
        *clock_args, "--fps", str(FPS), "--max-minutes", str(MAX_MINUTES),
        "--run-id", "run", "--results-dir", str(tmp_path),
    ]  # fmt: skip
    with (
        mock.patch.object(sys, "argv", argv),
        # main() writes SANDBOX_GRAPHICS_MODE and SCREEN_RECORDING_ENCODER into os.environ.
        mock.patch.dict(os.environ),
        mock.patch.object(runner, "episode_clocks", episode_clocks),
        mock.patch.object(runner, "R", DESKTOP),
        mock.patch.object(runner, "GuestClient", lambda **_: guest),
        mock.patch.object(runner.custom_tasks, "get", lambda task_id: task),
        mock.patch.object(runner, "set_up_agent", set_up_agent),
        mock.patch.object(runner, "run_stream", run_stream),
        # An unscored cell exits 1 after the summary is written (runner.py main).
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


def test_main_runs_the_episode_on_the_clocks_it_makes(tmp_path):
    """main() makes the episode clocks once and hands the same ones to run_stream and the
    environment. On the wall clock they are one clock, which set_up_agent, the scene capture it
    opens for it and the episode trace take; which of two clocks each takes is told apart under
    the token clock (below)."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "task_success", True)
    seen = {}
    run_main(tmp_path, task, FakeAgent(clock), clock, guest, seen=seen)
    (clocks,) = seen["clocks"]
    assert seen["run_stream"]["clocks"] is clocks
    assert seen["run_stream"]["env"].clocks is clocks
    assert seen["set_up_agent"]["clock"] is clock
    assert seen["set_up_agent"]["open_capture"](False).clock is clock
    started = json.loads((tmp_path / "run" / "episode_trace.jsonl").read_text().splitlines()[0])
    assert (started["event"], started["wall_time_ns"]) == ("EpisodeStarted", 1_000_000 * S)


def test_the_token_clock_s_records_and_agent_read_the_operational_clock(tmp_path):
    """Under the token clock world time is the environment's frames (FrameClock), and main()
    hands the operational clock to set_up_agent, the scene capture it opens for it, the episode
    trace and the generation artifact. The FrameClock has no wall time: an artifact stamped on it
    would fail, and main() would score the run without its saved state."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "task_success", True)
    seen = {}
    summary = run_main(
        tmp_path, task, FakeAgent(clock), clock, guest, seen=seen, clock_mode="token"
    )
    (clocks,) = seen["clocks"]
    assert isinstance(clocks.semantic, FrameClock) and clocks.operational is clock
    assert seen["run_stream"]["clocks"] is clocks
    assert seen["set_up_agent"]["clock"] is clock
    assert seen["set_up_agent"]["open_capture"](False).clock is clock
    run_dir = tmp_path / "run"
    started = json.loads((run_dir / "episode_trace.jsonl").read_text().splitlines()[0])
    assert (started["event"], started["wall_time_ns"]) == ("EpisodeStarted", 1_000_000 * S)
    assert (summary["invalid_run"], summary["artifact_error"]) == (None, None)
    assert summary["generation_artifact"] is not None
    manifest = json.loads((run_dir / "generation_artifact" / "manifest.json").read_text())
    assert 1_000_000.0 < manifest["captured_unix_s"] <= clock.now


class RecordingAgent(FakeAgent):
    """Records a request as it generates, and returns once its sink has handed the record over."""

    def stream_generate(self, tick: int) -> dict:
        self.sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"decision": tick})
        while self.sink._written < 1:
            time.sleep(0.001)
        return super().stream_generate(tick)


def test_main_has_the_kernel_number_the_agent_s_records(tmp_path):
    """main() gives the shell the agent's sink: the request record is written in the kernel's
    order, with its number, and the agent's records are closed after the episode."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "task_success", True)
    summary = run_main(tmp_path, task, RecordingAgent(clock), clock, guest, agent_trace=True)
    assert summary["invalid_run"] is None
    path = tmp_path / "run" / "episode_trace.jsonl"
    validate_episode_trace(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    events = [record["event"] for record in records]
    (request,) = [record for record in records if record["event"] == "ModelRequestSent"]
    assert request["payload"]["source"] == "agent"
    assert request["kernel_seq"] > records[events.index("InferenceStarted")]["kernel_seq"]
    assert events.index("ModelRequestSent") < events.index("InferenceCompleted")
    assert events.index("ModelRequestSent") < events.index("AgentTraceClosed")
    (closed,) = [record["payload"] for record in records if record["event"] == "AgentTraceClosed"]
    assert (closed["captures"], closed["written"], closed["error"]) == (1, 1, None)


def test_main_records_the_kernel_s_inputs(tmp_path):
    """main() has the shell record what the kernel was given, the agent's records among it, next
    to the episode trace. Fed to a new kernel they reproduce the kernel's records in the trace,
    with their numbers and readings (phase2.md, 怎么验证)."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "task_success", True)
    run_main(tmp_path, task, RecordingAgent(clock), clock, guest, agent_trace=True)
    run_dir = tmp_path / "run"
    inputs = (run_dir / "kernel_inputs.jsonl").read_text()
    lines = [json.loads(line) for line in inputs.splitlines()]
    assert any(event["event"] == "AgentRecord" for line in lines for event in line["events"])
    config = TurnConfig("wall", HORIZON_S, FPS, FPS, timing=runner.RUNNER_TIMING)
    kernel, entries = replay_inputs(config, lines)
    assert kernel.termination.stop_reason == "task_success"
    trace = (run_dir / "episode_trace.jsonl").read_text()
    records = [json.loads(line) for line in trace.splitlines()]
    written = {}
    for record in records:
        if record.get("kernel_seq") is not None:
            written.setdefault(record["kernel_seq"], (record["semantic"], record["ops"]))
    assert [(e.seq, e.semantic, e.ops) for e in entries] == sorted(
        (seq, *readings) for seq, readings in written.items()
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
