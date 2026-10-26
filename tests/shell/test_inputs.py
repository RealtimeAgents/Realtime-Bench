"""The kernel's inputs as run_episode records them (evaluation/shell.py ``inputs_log``): the
readings of every kernel call that handled something and the events the step took from the inbox.
Fed to a new kernel they reproduce the episode's log record for record, which is what replaying a
real run's inputs after a change needs (phase2.md, 怎么验证). A step the kernel raised in is the
last line, marked; the log of an episode the kernel finished ends with an end line."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import threading

import pytest

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.evaluation.kernel import events
from realtime_framework.evaluation.kernel.core import Kernel
from realtime_framework.evaluation.kernel.events import Record
from realtime_framework.evaluation.kernel.turns import TurnPolicy

from .world import CLICK, Environment, Reply, World, run

TAP = Action(ActionType.KEY, {"key": "a"})
# Fields the JSON line holds as lists.
TUPLES = {"fingerprint", "held_keys", "end_checks", "keys"}
# Kernel records the trace has under another name than a KernelRecord (D6).
MAPPED = {
    "InferenceStarted",
    "FrameCaptured",
    "GenerationFinished",
    "ActionProposed",
    "ActionResolved",
    "ActionApplied",
    "ClockAdvanced",
    "RunInvalidated",
}


def _action(record: dict | None) -> Action | None:
    if record is None:
        return None
    return Action(
        ActionType(record["action_type"]),
        record["parameters"],
        record["coordinate_space"],
        record["coordinate_transform"],
    )


def recorded_event(record: dict):
    """One recorded event, as the kernel got it."""
    fields = dict(record)
    kind = fields.pop("event")
    if kind == "GenerationFinished":
        result = fields["result"]
        invalid = result["invalid"]
        return events.GenerationFinished(
            fields["generation"],
            events.GenerationResult(
                _action(result["action"]),
                result["output_tokens"],
                result["latency"],
                result["protocol_failure"],
                None if invalid is None else events.Invalidated(**invalid),
            ),
        )
    if kind == "ActionExecuted":
        # Each of a tap's end checks is an (ended, reason, error) list in the line.
        fields["end_checks"] = [tuple(read) for read in fields["end_checks"]]
    return getattr(events, kind)(
        **{name: tuple(value) if name in TUPLES else value for name, value in fields.items()}
    )


def replay_inputs(config, lines: list[dict]) -> tuple[Kernel, list]:
    """Feed the recorded inputs to a new kernel, up to a step the kernel raised in; return it and
    the log it wrote."""
    kernel = Kernel(TurnPolicy(config))
    start, *steps = lines
    assert (start["call"], start["events"]) == ("start", [])
    out = kernel.start(start["semantic"], start["ops"])
    entries = [command.entry for command in out if isinstance(command, Record)]
    for line in steps:
        if line["call"] == "end" or line.get("raised"):
            assert line is lines[-1]
            break
        assert line["call"] == "step"
        for record in line["events"]:
            kernel.post(recorded_event(record))
        out = kernel.step(line["semantic"], line["ops"])
        assert out is not None  # only calls that handled something are recorded
        entries += [command.entry for command in out if isinstance(command, Record)]
    return kernel, entries


@contextlib.contextmanager
def kernel_log():
    """The entries the kernels write meanwhile, in order."""
    entries: list = []
    log = Kernel._log

    def spied(kernel, kind, data):
        log(kernel, kind, data)
        entries.append(kernel._out[-1].entry)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Kernel, "_log", spied)
        yield entries


@pytest.mark.parametrize(
    ("clock_mode", "replies", "world", "fields"),
    [
        (
            "wall",
            [
                Reply(0.5),
                Reply(0.5, Action.hold_key("a", 0.75)),
                Reply(0.25, Action.wait(0.5)),
                Reply(2.0),
            ],
            World(rejections={0: "bad key"}, check_errors={2: "slow read"}),
            {},
        ),
        (
            "token",
            [
                Reply(0.25, Action.hold_key("a", 0.5), output_tokens=100),
                Reply(0.25, TAP, output_tokens=50),
                Reply(0.25, CLICK, output_tokens=150),
                Reply(0.25, CLICK, output_tokens=100),
            ],
            World(tap_checks={1: ((False, None, "tap read"),)}),
            {"fps": 2.0, "tokens_per_frame": 100},
        ),
    ],
)
def test_replaying_the_recorded_inputs_reproduces_the_episode_log(
    tmp_path, clock_mode, replies, world, fields
):
    path = tmp_path / "kernel_inputs.jsonl"
    with kernel_log() as logged:
        episode = run(clock_mode, 3.0, replies, world, inputs_log=path, **fields)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[-1]["call"] == "end"
    kernel, entries = replay_inputs(episode.config, lines)
    assert entries == logged
    assert kernel.termination is not None
    assert kernel.termination.stop_reason == episode.result.stop_reason
    written = {}
    for record in episode.trace.records:
        written.setdefault(record["kernel_seq"], (record["semantic"], record["ops"]))
    assert [(e.seq, e.semantic, e.ops) for e in entries] == [
        (seq, *readings) for seq, readings in sorted(written.items())
    ]
    kinds = [r["payload"]["kind"] for r in episode.trace.records if r["event"] == "KernelRecord"]
    assert kinds == [e.kind for e in entries if e.kind not in MAPPED]


class _LockedResult(Environment):
    """An execution record that cannot be copied."""

    def execute(self, action, *, release_at=None):
        status = super().execute(action, release_at=release_at)
        self._last_execution_result = {"lock": threading.Lock()}
        return status


def test_an_input_the_log_cannot_hold_leaves_the_episode_to_finish(tmp_path):
    episode = run(
        "wall", 2.0, [Reply(0.5)] * 2, inputs_log=tmp_path / "inputs.jsonl", env_type=_LockedResult
    )
    assert episode.statuses() == ["applied", "applied"]
    assert episode.result.stop_reason == "time_limit"
    assert any(line.startswith("[default] kernel inputs log write failed") for line in episode.logs)
    lines = (tmp_path / "inputs.jsonl").read_text().splitlines()
    assert "end" not in [json.loads(line)["call"] for line in lines]


def test_a_step_the_kernel_raised_in_is_the_log_s_last_line(tmp_path, monkeypatch):
    """N11: the kernel raises in the step that takes the end check at 2 s. That step is the last
    line, marked, with its readings and the events it took; there is no end line, and the lines
    before it replay the kernel's log as far as it got."""
    step = Kernel.step

    def broken(kernel, semantic, ops=None):
        taken = kernel.inbox._events
        if semantic >= 2.0 and any(isinstance(event, events.EndChecked) for event in taken):
            raise RuntimeError("step broke")
        return step(kernel, semantic, ops)

    monkeypatch.setattr(Kernel, "step", broken)
    path = tmp_path / "kernel_inputs.jsonl"
    with kernel_log() as logged:
        episode = run("wall", 3.0, [Reply(0.5), Reply(0.5)], inputs_log=path)
    assert episode.invalid["reason"] == "run_exception"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    last = lines[-1]
    assert (last["call"], last["raised"], last["semantic"]) == ("step", True, 2.0)
    assert "EndChecked" in [event["event"] for event in last["events"]]
    _, entries = replay_inputs(episode.config, lines)
    assert entries == logged


class _Disk(io.StringIO):
    """A log file that keeps apart what was flushed to it."""

    def __init__(self):
        super().__init__()
        self.flushed = ""
        self.behind = []  # for each write, whether what came before was not flushed yet

    def write(self, text: str) -> int:
        self.behind.append(self.getvalue() != self.flushed)
        return super().write(text)

    def flush(self) -> None:
        super().flush()
        self.flushed = self.getvalue()


class _Interrupted(Environment):
    """The second end check is interrupted, as SIGTERM interrupts main() (runner.py)."""

    def check_end(self):
        if "check_end" in self.names():
            raise KeyboardInterrupt
        return super().check_end()


def test_an_interrupted_episode_leaves_the_lines_it_wrote_on_disk(monkeypatch, tmp_path):
    """Each line is flushed as it is written and the log is closed whatever stops the episode."""
    disk = _Disk()
    _opens(monkeypatch, lambda: disk)
    with pytest.raises(KeyboardInterrupt):
        run("wall", 3.0, [Reply(0.5)], inputs_log=tmp_path / "in.jsonl", env_type=_Interrupted)
    assert disk.closed
    calls = [json.loads(line)["call"] for line in disk.flushed.splitlines()]
    assert calls[0] == "start" and calls[1:] and set(calls[1:]) == {"step"}
    assert disk.behind == [False] * len(calls)


class _FullDisk(io.StringIO):
    """A log file whose last flush, on closing, finds the disk full."""

    def close(self) -> None:
        if not self.closed:
            super().close()
            raise OSError(28, "No space left on device")


def _opens(monkeypatch, opened) -> None:
    monkeypatch.setattr(pathlib.Path, "open", lambda path, *args, **kwargs: opened())


def test_an_inputs_log_that_fails_to_close_leaves_the_episode_as_it_ended(monkeypatch, tmp_path):
    """The log is a replay aid: the deadline episode it failed to close for is still a valid
    episode that ended at the deadline, its inputs released once."""
    _opens(monkeypatch, _FullDisk)
    episode = run("wall", 3.0, [Reply(0.5), Reply(hang=True)], inputs_log=tmp_path / "in.jsonl")
    assert episode.invalid is None
    assert episode.result.termination_record["stop_reason"] == "time_limit"
    assert episode.env.names().count("release_inputs") == 1
    assert any(line.startswith("[default] kernel inputs log write failed") for line in episode.logs)


def test_an_inputs_log_that_cannot_be_opened_leaves_the_episode_to_run(monkeypatch, tmp_path):
    def refuse():
        raise OSError(13, "Permission denied")

    _opens(monkeypatch, refuse)
    episode = run("wall", 2.0, [Reply(0.5)] * 2, inputs_log=tmp_path / "in.jsonl")
    assert episode.invalid is None
    assert episode.statuses() == ["applied", "applied"]
    assert any(line.startswith("[default] kernel inputs log write failed") for line in episode.logs)
