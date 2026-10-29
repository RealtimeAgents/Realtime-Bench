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
import sys
import time
import traceback
import types
from unittest import mock

import pytest

from realtime_framework.clock import Clocks, FrameClock, SystemClock
from realtime_framework.entity.actions import Action, Observation
from realtime_framework.entity.harness import StreamingAgent
from realtime_framework.entity.model import (
    QUOTA_EXHAUSTED,
    PendingDecision,
    TransportFailure,
    TransportFailureError,
)
from realtime_framework.entity.records import TraceEvent, validate_episode_trace
from realtime_framework.evaluation import reporting, runner
from realtime_framework.evaluation.agent_trace import AgentTraceSink
from realtime_framework.evaluation.kernel.turns import TurnConfig
from realtime_framework.evaluation.reporting import REASONING_REPLAY_DEFINITION

from ..launch import trial_launch
from ..shell.test_inputs import kernel_log, replay_inputs

S = 1_000_000_000  # nanoseconds
# The task the trial worker writes into the launch file; the runner runs the fake task in its place.
WEB = "rt_waitlist_fit"
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
    tmp_path,
    task,
    agent,
    clock,
    guest,
    *,
    agent_trace=False,
    seen=None,
    clock_mode="wall",
    emergency=False,
    scene_capture=None,
    model_args=("--catalog-model", "claude-opus-5-anthropic"),
    setup_error=None,
) -> dict:
    """Run main() for ``task`` and return its run_summary.json. ``clock`` is the operational
    clock; on the wall clock it is world time too, and under the token clock (in pause mode)
    world time is a FrameClock. With ``agent_trace`` the agent gets an AgentTraceSink on the
    episode trace, as ``agent.sink``, as set_up_agent makes one. ``seen`` gets the episode clocks
    main() makes, the keyword arguments it calls set_up_agent with, the environment and keyword
    arguments it calls run_stream with, and the code main() exits with (None when it returns).
    With ``emergency`` an exception main() lets out goes to the emergency summary, as the
    __main__ block hands it over, and ``seen["escaped"]`` gets its traceback. ``scene_capture``
    is the one set_up_agent opens, and ``model_args`` name the model to the trial worker, which
    writes the launch file main() runs. ``setup_error`` is raised by set_up_agent where
    agent.initialize() raises, in the provider's setup."""
    seen = {} if seen is None else seen

    def episode_clocks(clock_mode, fps):
        semantic = FrameClock(fps) if clock_mode == "token" else clock
        seen.setdefault("clocks", []).append(Clocks(semantic, clock))
        return seen["clocks"][-1]

    def set_up_agent(setup, *args, **kwargs):
        seen["set_up_agent"] = kwargs
        setup.agent = agent
        setup.scene_capture = scene_capture
        if agent_trace:
            pack = kwargs["run_dir"] / "episode_blobs.pack"
            sink = AgentTraceSink(kwargs["episode_trace"], pack, clock=kwargs["clock"])
            setup.agent_trace = agent.sink = sink
        if setup_error is not None:
            raise setup_error
        setup.stage = "environment_setup"

    def run_stream(adapter, env, **kwargs):
        seen["run_stream"] = {"env": env, **kwargs}
        return real_run_stream(adapter, env, **kwargs)

    real_run_stream = runner.run_stream
    clock_args = ["--clock", clock_mode]
    if clock_mode == "token":
        clock_args += ["--tokens-per-frame", "unlimited"]
    with pytest.MonkeyPatch.context() as patch:
        launch = trial_launch(
            patch, tmp_path, *model_args, "--task", WEB, *clock_args, "--fps", str(FPS),
            "--duration-minutes", str(MAX_MINUTES),
        )  # fmt: skip
        argv = ["--launch", str(launch), "--run-id", "run", "--results-dir", str(tmp_path)]
        patch.setattr(sys, "argv", ["runner", *argv])
        patch.setattr(runner, "episode_clocks", episode_clocks)
        patch.setattr(runner, "R", DESKTOP)
        patch.setattr(runner, "GuestClient", lambda **_: guest)
        patch.setattr(runner.custom_tasks, "get", lambda task_id: task)
        patch.setattr(runner, "set_up_agent", set_up_agent)
        patch.setattr(runner, "run_stream", run_stream)
        seen["exit_code"] = None
        try:
            runner.main()
        except SystemExit as exc:  # an unscored cell, after the summary is written
            seen["exit_code"] = exc.code
        except Exception:
            if not emergency:
                raise
            seen["escaped"] = traceback.format_exc()
            runner._write_emergency_setup_summary(seen["escaped"])
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
    assert (summary["run_status"], summary["solved"], summary["stop_reason"]) == (
        "scored", solved, reason,
    )  # fmt: skip
    assert (summary["failure_attribution"], summary["environment_valid"]) == (None, None)
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


class ReplayingAgent(FakeAgent):
    """Records a generation request that carried two past reasonings back, as a LiteLLM driver's
    capture does (providers/capture.py HttpCapture)."""

    def reply(self, decision: int) -> dict:
        self.sink.capture(
            TraceEvent.MODEL_REQUEST_SENT,
            {"channel": "litellm.openai_sdk", "purpose": "generation", "exact_wire": True,
             "replayed_reasoning": 2},
        )  # fmt: skip
        while self.sink._written < 1:
            time.sleep(0.001)
        return super().reply(decision)


def test_main_summarizes_the_past_reasonings_its_requests_carried(tmp_path):
    """The run summary's reasoning_replay: the generation requests of this attempt that recorded
    the past reasonings they carried (reporting.summarize_reasoning_replay); none where no
    request recorded them."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "task_success", True)
    summary = run_main(tmp_path, task, ReplayingAgent(clock), clock, guest, agent_trace=True)
    assert summary["invalid_run"] is None and scored(tmp_path)
    assert summary["reasoning_replay"] == {
        "requests": 1,
        "requests_with_reasoning": 1,
        "replayed_reasoning": 2,
        "exact_wire": True,
        "definition": REASONING_REPLAY_DEFINITION,
    }
    plain = tmp_path / "plain"
    plain.mkdir()
    clock = FakeClock(1_000_000.0)
    task = FakeTask(clock, guest, "task_success", True)
    assert run_main(plain, task, FakeAgent(clock), clock, guest)["reasoning_replay"] is None


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
    assert "invalid_runs" not in summary  # the kernel and main() find the one failure
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


def test_a_re_evaluation_that_scores_a_run_invalid_for_its_evaluator(tmp_path, monkeypatch):
    """The run is invalid only for its evaluator's unhealthy backend; a re-evaluation that scores
    the saved state solved makes it scored and passed, stopped as its loop stopped, with the
    task-success rate and the time to success main() writes of a run solved when it ended."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = UnhealthyTask(clock, guest, "task_success", True)
    summary = run_main(tmp_path, task, FakeAgent(clock), clock, guest)
    assert (summary["run_status"], summary["solved"], summary["task_success_rate"]) == (
        "invalid", None, None,
    )  # fmt: skip
    replaced = _reevaluated(
        tmp_path / "run", {"resolved": True, "evaluation_version": 1}, monkeypatch
    )
    assert (replaced["run_status"], replaced["invalid_run"], replaced["solved"]) == (
        "scored", None, True,
    )  # fmt: skip
    assert (replaced["stop_reason"], replaced["outcome"], replaced["failure_attribution"]) == (
        "task_success", "pass", None,
    )  # fmt: skip
    assert (replaced["task_success_rate"], replaced["normalized_time_to_success"]) == (
        1.0, round(END_S / HORIZON_S, 9),
    )  # fmt: skip
    assert scored(tmp_path)


@pytest.mark.parametrize(
    ("verification", "concluded"),
    [
        ({"resolved": False, "evaluation_version": 1}, ("scored", False, "fail", 0.0)),
        (
            {"resolved": True, "evaluation_version": 1, "ok": False},
            ("invalid", None, "error", None),
        ),
    ],
    ids=["unsolved", "not safe to score"],
)
def test_a_re_evaluation_of_a_solved_run_counts_the_new_score(
    tmp_path, monkeypatch, verification, concluded
):
    """A run scored solved, re-evaluated unsolved or not safe to score: its task-success rate
    and time to success follow the new score."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    summary = run_main(
        tmp_path, FakeTask(clock, guest, "task_success", True), FakeAgent(clock), clock, guest
    )
    assert (summary["task_success_rate"], summary["normalized_time_to_success"]) == (
        1.0, round(END_S / HORIZON_S, 9),
    )  # fmt: skip
    replaced = _reevaluated(tmp_path / "run", verification, monkeypatch)
    assert (
        replaced["run_status"], replaced["solved"], replaced["outcome"], replaced["task_success_rate"],
    ) == concluded  # fmt: skip
    assert replaced["normalized_time_to_success"] is None


def _reevaluate(run_dir, verification, monkeypatch, capsys) -> tuple[dict, object]:
    """reevaluate without --replace-summary, the saved state's evaluation stubbed: the line it
    prints and what it returns, which ``python -m ...reporting reevaluate`` exits with."""
    monkeypatch.setattr(reporting, "reeval", lambda run_dir, **options: (verification, {}))
    monkeypatch.setattr(sys, "argv", ["reevaluate", str(run_dir)])
    capsys.readouterr()
    returned = reporting.reevaluate_main()
    return json.loads(capsys.readouterr().out.splitlines()[-1]), returned


@pytest.mark.parametrize(
    ("verification", "printed"),
    [
        ({"resolved": True, "evaluation_version": 1}, ("scored", True, None)),
        (
            {"resolved": True, "evaluation_version": 1, "ok": False},
            ("invalid", None, "evaluation_backend_unhealthy"),
        ),
    ],
)
def test_a_re_evaluation_without_replacing_the_summary_reports_the_evaluation_alone(
    tmp_path, monkeypatch, capsys, verification, printed
):
    """Without --replace-summary a re-evaluation judges the new evaluation alone: it prints whether
    that evaluation scores, as a diagnostic, writes it to evaluation.reeval.json and leaves the run
    summary as it was, whatever failure made the run invalid."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "task_success", True)
    run_main(
        tmp_path, task, FakeAgent(clock), clock, guest, scene_capture=FailingCloseCapture(clock)
    )
    run_dir = tmp_path / "run"
    written = (run_dir / "run_summary.json").read_text()
    out, returned = _reevaluate(run_dir, verification, monkeypatch, capsys)
    assert returned is None
    assert (out["run_status"], out["solved"], out["invalid_reason"]) == printed
    assert (run_dir / "run_summary.json").read_text() == written
    assert not (run_dir / "run_summary.pre_reeval.json").exists()
    record = json.loads((run_dir / "evaluation.reeval.json").read_text())
    assert record["evaluation"] == verification


class FailingCloseCapture:
    """The scene capture set_up_agent opens, whose close() fails as a CDP socket's can."""

    def __init__(self, clock):
        self.clock = clock

    def capture(self, tick):
        return Observation(b"", self.clock.time(), tick)

    def close(self):
        raise RuntimeError("CDP socket gone")


def _reevaluated(run_dir, verification, monkeypatch) -> dict:
    """reevaluate --replace-summary, the saved state's evaluation stubbed."""
    monkeypatch.setattr(reporting, "reeval", lambda run_dir, **options: (verification, {}))
    monkeypatch.setattr(sys, "argv", ["reevaluate", str(run_dir), "--replace-summary"])
    reporting.reevaluate_main()
    return json.loads((run_dir / "run_summary.json").read_text())


def test_a_failure_after_the_first_is_kept_and_keeps_a_re_evaluated_run_invalid(
    tmp_path, monkeypatch
):
    """The evaluator says its backend was unhealthy, and then the scene capture fails to close.
    The run is invalid for the first, and its summary lists both. A re-evaluation whose evaluator
    answers healthy leaves it invalid for the other, its environment's, which it says is not
    valid."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = UnhealthyTask(clock, guest, "task_success", True)
    summary = run_main(
        tmp_path, task, FakeAgent(clock), clock, guest, scene_capture=FailingCloseCapture(clock)
    )
    assert summary["invalid_run"]["reason"] == "evaluation_backend_unhealthy"
    assert [(r["category"], r["reason"]) for r in summary["invalid_runs"]] == [
        ("evaluator", "evaluation_backend_unhealthy"),
        ("environment", "cdp_close_failed"),
    ]
    assert summary["environment_valid"] is None
    replaced = _reevaluated(
        tmp_path / "run", {"resolved": True, "evaluation_version": 1}, monkeypatch
    )
    assert (replaced["run_status"], replaced["invalid_run"]["reason"]) == (
        "invalid", "cdp_close_failed",
    )  # fmt: skip
    assert (replaced["stop_reason"], replaced["outcome"]) == ("invalid_run", "error")
    assert replaced["environment_valid"] is False
    assert not scored(tmp_path)


class FailingActionGuest(FakeGuest):
    """Its action endpoint fails as a dead guest's does: the environment latches the run invalid
    (sandbox/actions.py _execution_failed)."""

    def execute_action(self, action):
        raise RuntimeError("guest action endpoint down")


def test_the_failure_the_environment_latched_comes_before_those_main_finds_after_it(tmp_path):
    """The click fails and the environment latches the run invalid; after the episode the scene
    capture fails to close. The environment latched its failure while the episode ran, before
    main() found the other: the run is invalid for it, and its summary lists both."""
    clock = FakeClock(1_000_000.0)
    guest = FailingActionGuest()
    task = FakeTask(clock, guest, "task_success", True)
    summary = run_main(
        tmp_path, task, FakeAgent(clock), clock, guest, scene_capture=FailingCloseCapture(clock)
    )
    assert summary["invalid_run"] == {
        "category": "environment",
        "reason": "action_execution_failed",
        "detail": "guest action endpoint down",
    }
    assert [(r["category"], r["reason"]) for r in summary["invalid_runs"]] == [
        ("environment", "action_execution_failed"),
        ("environment", "cdp_close_failed"),
    ]
    assert (summary["failure_attribution"], summary["environment_valid"]) == ("environment", False)


def test_a_re_evaluation_of_the_same_evaluation_keeps_the_failure_found_first(
    tmp_path, monkeypatch
):
    """The environment latched the run invalid as the click failed, and the evaluator then said
    its backend was unhealthy: the run is invalid for the environment's failure, found first. A
    re-evaluation with the very same evaluation decides the evaluator's failure alone: the run
    stays invalid for the environment's, and keeps the run's list of failures."""
    clock = FakeClock(1_000_000.0)
    guest = FailingActionGuest()
    summary = run_main(
        tmp_path, UnhealthyTask(clock, guest, "task_success", True), FakeAgent(clock), clock, guest
    )
    found = [
        ("environment", "action_execution_failed"),
        ("evaluator", "evaluation_backend_unhealthy"),
    ]
    assert (summary["invalid_run"]["reason"], summary["environment_valid"]) == (
        "action_execution_failed", False,
    )  # fmt: skip
    assert [(r["category"], r["reason"]) for r in summary["invalid_runs"]] == found
    replaced = _reevaluated(tmp_path / "run", summary["verification"], monkeypatch)
    assert replaced["invalid_run"] == summary["invalid_run"]
    assert (replaced["run_status"], replaced["failure_attribution"]) == ("invalid", "environment")
    assert (replaced["error_category"], replaced["error_reason"]) == (
        "environment", "action_execution_failed",
    )  # fmt: skip
    assert (replaced["stop_reason"], replaced["solved"], replaced["environment_valid"]) == (
        "invalid_run", None, False,
    )  # fmt: skip
    assert [(r["category"], r["reason"]) for r in replaced["invalid_runs"]] == found


class UnhealthySnapshotTask(FakeTask):
    """Its frozen snapshot says the environment was not healthy (runner.py _invalid_snapshot)."""

    def evaluation_snapshot(self):
        self.clock.now += SNAPSHOT_S
        return {"kind": "fake", "state": {"ok": False}}


class RaisingEvaluatorTask(UnhealthySnapshotTask):
    def evaluate_snapshot(self, snapshot):
        raise RuntimeError("evaluator crashed")


def test_an_evaluator_that_raises_writes_its_own_failure_to_the_trace(tmp_path):
    """The frozen snapshot says the environment was unhealthy, and then the evaluator raises.
    The run is invalid for the first failure, which main() writes last, as the run's verdict;
    the RunInvalidated it writes as the evaluator raises is the evaluator's own."""
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = RaisingEvaluatorTask(clock, guest, "task_success", True)
    summary = run_main(tmp_path, task, FakeAgent(clock), clock, guest)
    assert summary["invalid_run"]["reason"] == "snapshot_environment_unhealthy"
    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "episode_trace.jsonl").read_text().splitlines()
    ]
    invalidated = [record["payload"] for record in records if record["event"] == "RunInvalidated"]
    assert [(payload["category"], payload["reason"]) for payload in invalidated] == [
        ("evaluator", "evaluation_exception"),
        ("environment", "snapshot_environment_unhealthy"),
    ]
    assert invalidated[0]["detail"].rstrip().endswith("RuntimeError: evaluator crashed")


def test_a_snapshot_judged_invalid_keeps_its_reason_though_its_record_then_fails(
    tmp_path, monkeypatch
):
    """The frozen snapshot says the environment was unhealthy, and then its trace record fails,
    its hash raising. The run is invalid for the first of the two, as for any failures main()
    finds, and its summary lists both."""

    def broken(value):
        raise RuntimeError("snapshot hash failed")

    monkeypatch.setattr(runner, "canonical_sha256", broken)
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = UnhealthySnapshotTask(clock, guest, "task_success", True)
    summary = run_main(tmp_path, task, FakeAgent(clock), clock, guest)
    assert summary["invalid_run"]["reason"] == "snapshot_environment_unhealthy"
    assert [(r["category"], r["reason"]) for r in summary["invalid_runs"]] == [
        ("environment", "snapshot_environment_unhealthy"),
        ("environment", "evaluation_snapshot_failed"),
    ]


def test_a_re_evaluated_run_of_an_experimental_model_stays_experimental(tmp_path, monkeypatch):
    """jev-1.13 is excluded from maintained results and runs only with ALLOW_EXPERIMENTAL_MODEL:
    its run is experimental, and so is it re-evaluated, with the progress of the new
    evaluation."""
    monkeypatch.setenv("ALLOW_EXPERIMENTAL_MODEL", "1")
    clock = FakeClock(1_000_000.0)
    guest = FakeGuest()
    task = FakeTask(clock, guest, "task_success", True)
    seen = {}
    summary = run_main(
        tmp_path, task, FakeAgent(clock), clock, guest, seen=seen,
        model_args=("--catalog-model", "jev-1.13"),
    )  # fmt: skip
    assert (summary["run_status"], summary["benchmark_eligible"]) == ("experimental", False)
    assert seen["exit_code"] is None  # accepted, as ALLOW_EXPERIMENTAL_MODEL is set
    verification = {"resolved": True, "evaluation_version": 1, "progress": 0.5}
    replaced = _reevaluated(tmp_path / "run", verification, monkeypatch)
    assert (replaced["run_status"], replaced["progress"]) == ("experimental", 0.5)
    assert not scored(tmp_path)


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


def test_a_live_session_that_fails_to_open_is_invalid_for_its_failure_s_kind(tmp_path):
    """agent.initialize() opens the live session, and its connect raises the failure it was
    classified as, as OpenAI Realtime's retried connect and a streaming session's readying do,
    here an exhausted quota: the run is invalid for that failure, as a generation's would be,
    not as the runner's run_exception."""
    clock, guest = FakeClock(1_000_000.0), FakeGuest()
    cause = RuntimeError("1011 None. You exceeded your current quota")
    failure = TransportFailureError(TransportFailure(QUOTA_EXHAUSTED), cause, 0)
    seen = {}
    summary = run_main(
        tmp_path,
        FakeTask(clock, guest, "task_success", True),
        FakeAgent(clock),
        clock,
        guest,
        seen=seen,
        setup_error=failure,
    )
    invalid = summary["invalid_run"]
    assert (invalid["category"], invalid["reason"]) == ("provider", "provider_quota_exhausted")
    assert "quota_exhausted after 0 transport retries" in invalid["detail"]
    assert seen["exit_code"] == 1 and not scored(tmp_path)


def test_a_summary_main_fails_to_build_is_written_as_the_run_s_failure(tmp_path, monkeypatch):
    """The episode plays its decision and is scored, and then building the summary raises: the
    run is invalid for it, its summary says so with the decision counted, and main() exits 1."""
    from realtime_framework.evaluation import clock as clock_module

    def broken(*args, **kwargs):
        raise RuntimeError("model timing broke")

    monkeypatch.setattr(clock_module, "summarize_model_timing", broken)
    clock, guest = FakeClock(1_000_000.0), FakeGuest()
    seen = {}
    task = FakeTask(clock, guest, "task_success", True)
    summary = run_main(tmp_path, task, FakeAgent(clock), clock, guest, seen=seen)
    invalid = summary["invalid_run"]
    assert (invalid["category"], invalid["reason"]) == ("harness", "run_summary_failed")
    assert "RuntimeError: model timing broke" in invalid["detail"]
    assert (summary["stop_reason"], summary["decisions"], summary["outcome"]) == (
        "invalid_run", 1, "error",
    )  # fmt: skip
    assert seen["exit_code"] == 1 and not scored(tmp_path)


def test_a_summary_that_fails_after_an_earlier_failure_is_invalid_for_the_earlier(
    tmp_path, monkeypatch
):
    """The scene capture fails to close, and then building the summary raises: the summary main()
    writes in its place is invalid for the first failure and lists both, with the outcome and
    attribution of the first."""
    from realtime_framework.evaluation import clock as clock_module

    def broken(*args, **kwargs):
        raise RuntimeError("model timing broke")

    monkeypatch.setattr(clock_module, "summarize_model_timing", broken)
    clock, guest = FakeClock(1_000_000.0), FakeGuest()
    seen = {}
    task = FakeTask(clock, guest, "task_success", True)
    summary = run_main(
        tmp_path, task, FakeAgent(clock), clock, guest, seen=seen,
        scene_capture=FailingCloseCapture(clock),
    )  # fmt: skip
    assert [(r["category"], r["reason"]) for r in summary["invalid_runs"]] == [
        ("environment", "cdp_close_failed"),
        ("harness", "run_summary_failed"),
    ]
    assert summary["invalid_run"] == summary["invalid_runs"][0]
    assert (summary["run_status"], summary["solved"], summary["stop_reason"]) == (
        "invalid", None, "invalid_run",
    )  # fmt: skip
    assert (summary["outcome"], summary["error_category"], summary["error_reason"]) == (
        "error", "environment", "cdp_close_failed",
    )  # fmt: skip
    assert summary["failure_attribution"] == "environment"
    assert (summary["loop_stop_reason"], summary["decisions"]) == (None, 1)
    assert "environment_valid" not in summary and "progress" not in summary
    assert seen["exit_code"] == 1 and not scored(tmp_path)


def test_a_run_whose_run_metrics_fail_is_invalid_for_them(tmp_path, monkeypatch):
    """The decision lifecycle cannot be aggregated: the run, solved before, is invalid for it,
    its run metrics are those of no lifecycle and say why, and main() exits 1. The run metrics are
    first asked with the run's score, then with none."""
    real = runner.summarize_run_metrics
    asked = []

    def failing_once(*args, **kwargs):
        asked.append(kwargs["solved"])
        if len(asked) == 1:
            raise ValueError("lifecycle broke")
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "summarize_run_metrics", failing_once)
    clock, guest = FakeClock(1_000_000.0), FakeGuest()
    seen = {}
    task = FakeTask(clock, guest, "task_success", True)
    summary = run_main(tmp_path, task, FakeAgent(clock), clock, guest, seen=seen)
    assert asked == [True, None]
    invalid = summary["invalid_run"]
    assert (invalid["category"], invalid["reason"]) == ("harness", "run_metrics_aggregation_failed")
    assert "ValueError: lifecycle broke" in summary["run_metrics_error"]
    assert "invalid_runs" not in summary
    assert (summary["run_status"], summary["solved"], summary["stop_reason"]) == (
        "invalid", None, "invalid_run",
    )  # fmt: skip
    assert (summary["outcome"], summary["failure_attribution"]) == ("error", "harness")
    assert (summary["loop_stop_reason"], summary["task_success_rate"]) == ("task_success", None)
    assert summary["verification"]["resolved"] is True
    assert seen["exit_code"] == 1 and not scored(tmp_path)


class NotReadyGuest(FakeGuest):
    def wait_until_ready(self, max_wait):
        return False


def test_a_run_that_fails_to_set_up_in_a_used_run_dir_leaves_its_own_summary(tmp_path):
    """The first run scores; the second, with the same run id, fails before its episode, as the
    guest is not ready, and main() lets the failure out to the emergency summary. The run
    directory's summary is the second run's setup error, which reads none of the first run's
    decisions left in the directory; the first run's summary is moved aside."""
    clock, guest = FakeClock(1_000_000.0), FakeGuest()
    first = run_main(
        tmp_path, FakeTask(clock, guest, "task_success", True), FakeAgent(clock), clock, guest
    )
    assert first["run_status"] == "scored"
    assert first["decision_wall_latency_s"]["sample_count"] == 1
    run_dir = tmp_path / "run"
    written = (run_dir / "run_summary.json").read_text()

    clock, guest = FakeClock(2_000_000.0), NotReadyGuest()
    seen = {}
    summary = run_main(
        tmp_path,
        FakeTask(clock, guest, "task_success", True),
        FakeAgent(clock),
        clock,
        guest,
        seen=seen,
        emergency=True,
    )
    assert "RuntimeError: guest not ready" in seen["escaped"]
    assert (summary["run_status"], summary["stop_reason"]) == ("invalid", "setup_error")
    assert summary["decision_wall_latency_s"]["sample_count"] == 0
    assert (run_dir / "run_summary.pre_retry.json").read_text() == written


def test_the_emergency_summary_reads_the_arguments_as_main_parses_them(tmp_path, monkeypatch):
    """A failure main() lets out is written as a setup error from the run's arguments and the
    launch file they name (runner.py, the __main__ block): an option written --name=value is the
    one written --name value."""
    monkeypatch.chdir(tmp_path)  # where a summary under ./results would go
    launch = trial_launch(
        monkeypatch, tmp_path / "trial", "--catalog-model", "claude-opus-5-anthropic", "--task",
        WEB, "--clock", "wall", "--fps", "2", "--duration-minutes", "1",
    )  # fmt: skip
    argv = ["runner", f"--launch={launch}", "--run-id=run", "--results-dir=out"]
    with mock.patch.object(sys, "argv", argv):
        runner._write_emergency_setup_summary("boom")
    summary = json.loads((tmp_path / "out" / "run" / "run_summary.json").read_text())
    assert (summary["clock_mode"], summary["protocol"], summary["tokens_per_frame"]) == (
        "wall", "wall-clock", None,
    )  # fmt: skip
    assert (summary["fps"], summary["max_minutes"], summary["stop_reason"]) == (
        2.0, 1.0, "setup_error",
    )  # fmt: skip
    assert (summary["task_id"], summary["model"]) == (WEB, "claude-opus-5-anthropic")
