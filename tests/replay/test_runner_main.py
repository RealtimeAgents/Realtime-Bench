"""runner.main() on the wall clock writes the termination its decision loop latched.

The real main(), run_stream and GuestEnvironment run on a fake wall clock; the desktop, the guest,
the task and the agent are fakes. The task's end check and its evaluation snapshot take fake
time, and closing the agent (runner.py main) moves the clock past the deadline, so a world time
read after teardown is past the horizon although the episode ended early. main() takes
world_time_s from the termination record, hands the record to the run metrics and writes it into
run_summary.json (runner.py main).
"""

from __future__ import annotations

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
from realtime_framework.entity.model import PendingDecision
from realtime_framework.entity.records import TraceEvent, validate_episode_trace
from realtime_framework.evaluation import reporting, runner
from realtime_framework.evaluation.agent_trace import AgentTraceSink
from realtime_framework.evaluation.kernel.turns import TurnConfig

from ..shell.test_inputs import kernel_log, replay_inputs

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

    def generate(self, generation, decision, observation, budget):
        return PendingDecision(generation, decision, self.reply(decision))

    def reply(self, decision: int) -> dict:
        self.clock.now += LATENCY_S
        return {"action": Action.click(100, 100), "latency": LATENCY_S}

    def accept(self, generation):
        pass

    def discard(self, generation):
        pass

    def commit(self, outcome):
        pass

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
    main() makes, the keyword arguments it calls set_up_agent with, the environment and keyword
    arguments it calls run_stream with, and the code main() exits with (None when it returns)."""
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
    ):
        seen["exit_code"] = None
        try:
            runner.main()
        except SystemExit as exc:  # an unscored cell, after the summary is written
            seen["exit_code"] = exc.code
    return json.loads((tmp_path / "run" / "run_summary.json").read_text())


def scored(tmp_path) -> bool:
    """Whether the report's own check scores the run main() wrote (reporting.py summary_ok)."""
    return reporting.summary_ok(tmp_path / "run" / "run_summary.json", require_scored=True)


@pytest.mark.parametrize(
    ("reason", "solved", "outcome"),
    [("deadline_expired", False, "fail"), ("task_success", True, "pass")],
)
def test_main_reports_the_termination_the_loop_latched(tmp_path, reason, solved, outcome):
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, reason, solved)
    seen = {}
    summary = run_main(tmp_path, task, FakeAgent(clock), clock, guest, seen=seen)

    assert clock.now - task.origin >= HORIZON_S  # teardown ran past the deadline
    assert summary["invalid_run"] is None
    assert seen["exit_code"] is None and scored(tmp_path)
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

    def reply(self, decision: int) -> dict:
        self.sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"decision": decision})
        while self.sink._written < 1:
            time.sleep(0.001)
        return super().reply(decision)


def test_main_has_the_kernel_number_the_agent_s_records(tmp_path):
    """main() gives the shell the agent's sink: the request record is written in the kernel's
    order, with its number, and the agent's records are closed after the episode."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "task_success", True)
    summary = run_main(tmp_path, task, RecordingAgent(clock), clock, guest, agent_trace=True)
    assert summary["invalid_run"] is None and scored(tmp_path)
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


class TraceFailingAgent(FakeAgent):
    """Its one request record cannot be stored, so its records fail only once the episode is
    over, as main() closes them (agent_trace.close_agent_trace)."""

    def reply(self, decision: int) -> dict:
        def fail(data):
            raise OSError("disk full")

        self.sink._store = fail
        self.sink.capture(TraceEvent.MODEL_REQUEST_SENT, {"decision": decision}, body=b"{}")
        return super().reply(decision)


def test_agent_records_that_fail_to_be_written_invalidate_a_solved_run(tmp_path):
    """The task was solved, but the agent's records could not all be written: main() alone finds
    the run invalid, so it is no scored result and exits 1, its summary says why, and the trace
    has the one RunInvalidated that says so, before EpisodeFinished."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "task_success", True)
    seen = {}
    summary = run_main(
        tmp_path, task, TraceFailingAgent(clock), clock, guest, agent_trace=True, seen=seen
    )
    invalid = summary["invalid_run"]
    assert (invalid["category"], invalid["reason"]) == ("harness", "agent_trace_write_failed")
    assert (summary["solved"], summary["stop_reason"], summary["outcome"]) == (
        None, "invalid_run", "error",
    )  # fmt: skip
    assert seen["exit_code"] == 1 and not scored(tmp_path)
    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "episode_trace.jsonl").read_text().splitlines()
    ]
    events = [record["event"] for record in records]
    invalidated = [record["payload"] for record in records if record["event"] == "RunInvalidated"]
    assert [payload["reason"] for payload in invalidated] == ["agent_trace_write_failed"]
    assert events.index("RunInvalidated") < events.index("EpisodeFinished")
    finished = records[events.index("EpisodeFinished")]["payload"]
    assert (finished["stop_reason"], finished["resolved"]) == ("invalid_run", None)


class FailingSnapshotTask(FakeTask):
    """The snapshot read at the freeze fails; any later read would succeed, as after a passing
    failure of the state backend."""

    def __init__(self, *args):
        super().__init__(*args)
        self.reads = 0

    def evaluation_snapshot(self):
        self.reads += 1
        if self.reads == 1:
            self.clock.now += SNAPSHOT_S
            raise RuntimeError("snapshot read failed")
        return super().evaluation_snapshot()

    def verify(self):
        self.reads += 1
        return self.evaluate_snapshot(super().evaluation_snapshot())


def test_a_run_whose_snapshot_failed_scores_no_state_read_after_it(tmp_path):
    """The freeze's snapshot read fails: the run is invalid for it, and main() reads the task's
    state no more once the inputs are released. Nothing is scored or saved as the run's state,
    and the evaluation says why."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FailingSnapshotTask(clock, guest, "task_success", True)
    summary = run_main(tmp_path, task, FakeAgent(clock), clock, guest)
    assert summary["invalid_run"]["reason"] == "evaluation_snapshot_failed"
    assert (task.reads, summary["verification"]) == (1, None)
    run_dir = tmp_path / "run"
    trace = (run_dir / "episode_trace.jsonl").read_text().splitlines()
    assert "ScoreProduced" not in [json.loads(line)["event"] for line in trace]
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    assert evaluation == {"evaluation": None, "error": "no frozen snapshot"}
    manifest = json.loads((run_dir / "generation_artifact" / "manifest.json").read_text())
    assert manifest["payload"] == {"kind": "none"}


class UnhealthyTask(FakeTask):
    """Its evaluator answers, but says its backend was not healthy."""

    def evaluate_snapshot(self, snapshot):
        return {**super().evaluate_snapshot(snapshot), "ok": False}


def test_a_re_evaluation_judges_an_unscorable_evaluation_as_the_run_did(tmp_path, monkeypatch):
    """The first evaluation (runner.py main) and a re-evaluation of the saved state
    (reporting.reevaluate_main, its evaluator stubbed) give the same reason for not scoring it."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = UnhealthyTask(clock, guest, "task_success", True)
    summary = run_main(tmp_path, task, FakeAgent(clock), clock, guest)
    assert summary["invalid_run"]["reason"] == "evaluation_backend_unhealthy"
    run_dir = tmp_path / "run"
    verification = summary["verification"]
    monkeypatch.setattr(reporting, "reeval", lambda run_dir, **options: (verification, {}))
    monkeypatch.setattr(sys, "argv", ["reevaluate", str(run_dir), "--replace-summary"])
    reporting.reevaluate_main()
    replaced = json.loads((run_dir / "run_summary.json").read_text())
    assert replaced["invalid_run"]["reason"] == "evaluation_backend_unhealthy"


class TwoClickTask(FakeTask):
    """Ends once the guest has been clicked twice."""

    def episode_over(self):
        self.clock.now += QUERY_S
        return self.reason if len(self.guest.clicks) >= 2 else False


class HalfReportingAgent(FakeAgent):
    """Its first reply carries the provider's usage, its second none."""

    def reply(self, decision: int) -> dict:
        reply = super().reply(decision)
        if not getattr(self, "reported", False):
            self.reported = True
            reply["usage"] = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return reply


def test_token_usage_is_unknown_while_a_reply_reported_none(tmp_path):
    """The summary's token usage is the token accounting's totals, which a reply without usage
    leaves unknown; the part known is in the accounting alone."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = TwoClickTask(clock, guest, "task_success", True)
    summary = run_main(tmp_path, task, HalfReportingAgent(clock), clock, guest)
    accounting = summary["token_accounting"]
    assert accounting["responses"] == summary["token_usage"]["responses"] == 2
    assert accounting["counts"]["provider_input_tokens"]["known_total"] == 10
    assert summary["token_usage"]["input_tokens"] is None
    assert summary["token_usage"]["output_tokens"] is None


def test_main_records_the_kernel_s_inputs(tmp_path):
    """main() has the shell record what the kernel was given, the agent's records among it, next
    to the episode trace. Fed to a new kernel they reproduce the kernel's log, and so its records
    in the trace with their numbers and readings (phase2.md, 怎么验证)."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "task_success", True)
    with kernel_log() as logged:
        run_main(tmp_path, task, RecordingAgent(clock), clock, guest, agent_trace=True)
    run_dir = tmp_path / "run"
    inputs = (run_dir / "kernel_inputs.jsonl").read_text()
    lines = [json.loads(line) for line in inputs.splitlines()]
    assert any(event["event"] == "AgentRecord" for line in lines for event in line["events"])
    assert lines[-1]["call"] == "end"
    config = TurnConfig("wall", HORIZON_S, FPS, FPS, timing=runner.RUNNER_TIMING)
    kernel, entries = replay_inputs(config, lines)
    assert entries == logged
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
        raise RuntimeError("agent failed")
    return _raise_deeper(depth - 1)


def _raise_deeper(depth):
    return _raise_deep(depth)


class FailingAgent(FakeAgent):
    """Accepting its result raises from deep in a call stack, as an SDK's can: the traceback runs
    far past the invalid-run detail's 2000 characters and names the exception last."""

    def accept(self, generation):
        _raise_deep(40)


class FailingGenerationAgent(FakeAgent):
    """Its generation raises, as deep, before it took any time."""

    def generate(self, generation, decision, observation, budget):
        _raise_deep(40)


def test_an_exception_that_escapes_the_loop_keeps_its_cause(tmp_path):
    """The first decision's result is accepted on the shell's thread, after the loop's first end
    check and the generation, and accepting it raises (U18): the run is invalid as the runner's,
    its detail ends with the exception and the summary's error holds the whole traceback."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "deadline_expired", False)
    seen = {}
    summary = run_main(tmp_path, task, FailingAgent(clock), clock, guest, seen=seen)

    invalid = summary["invalid_run"]
    assert (invalid["category"], invalid["reason"]) == ("runner", "run_exception")
    assert seen["exit_code"] == 1
    error = summary["error"]
    assert error.startswith("Traceback (most recent call last):") and len(error) > 2000
    assert error.rstrip().endswith("RuntimeError: agent failed")
    assert invalid["detail"] == error[-2000:]
    assert summary["outcome"] == "error"
    assert summary["termination"] == {
        "stop_reason": "invalid_run",
        "end_s": QUERY_S + LATENCY_S,
        "snapshot_s": QUERY_S + LATENCY_S + SNAPSHOT_S,
        "snapshot_delay_s": SNAPSHOT_S,
    }


def test_a_generation_that_raises_is_its_generation_error(tmp_path):
    """The first decision's generation raises on its worker (evaluation/shell.py _generate): the
    run is invalid as the provider's generation error, and the summary has no error."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "deadline_expired", False)
    seen = {}
    summary = run_main(tmp_path, task, FailingGenerationAgent(clock), clock, guest, seen=seen)

    invalid = summary["invalid_run"]
    assert (invalid["category"], invalid["reason"]) == ("provider", "generation_error")
    assert seen["exit_code"] == 1
    assert summary["error"] is None
    assert summary["outcome"] == "error"
    assert summary["termination"] == {
        "stop_reason": "invalid_run",
        "end_s": QUERY_S,
        "snapshot_s": QUERY_S + SNAPSHOT_S,
        "snapshot_delay_s": SNAPSHOT_S,
    }
