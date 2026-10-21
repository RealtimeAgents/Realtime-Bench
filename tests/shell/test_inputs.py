"""The kernel's inputs as run_episode records them (evaluation/shell.py ``inputs_log``): the
readings of every kernel call that handled something and the events the step took from the inbox.
Fed to a new kernel they reproduce the episode's log record for record, which is what replaying a
real run's inputs after a change needs (phase2.md, 怎么验证)."""

from __future__ import annotations

import json
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


def _event(record: dict):
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
    return getattr(events, kind)(
        **{name: tuple(value) if name in TUPLES else value for name, value in fields.items()}
    )


def _replay(config, lines: list[dict]) -> tuple[Kernel, list]:
    """Feed the recorded inputs to a new kernel; return it and the log it wrote."""
    kernel = Kernel(TurnPolicy(config))
    start, *steps = lines
    assert (start["call"], start["events"]) == ("start", [])
    out = kernel.start(start["semantic"], start["ops"])
    entries = [command.entry for command in out if isinstance(command, Record)]
    for line in steps:
        assert line["call"] == "step"
        for record in line["events"]:
            kernel.post(_event(record))
        out = kernel.step(line["semantic"], line["ops"])
        assert out is not None  # only calls that handled something are recorded
        entries += [command.entry for command in out if isinstance(command, Record)]
    return kernel, entries


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
            World(tap_checks={1: ("tap read",)}),
            {"fps": 2.0, "tokens_per_frame": 100},
        ),
    ],
)
def test_replaying_the_recorded_inputs_reproduces_the_episode_log(
    tmp_path, clock_mode, replies, world, fields
):
    path = tmp_path / "kernel_inputs.jsonl"
    episode = run(clock_mode, 3.0, replies, world, inputs_log=path, **fields)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    kernel, entries = _replay(episode.config, lines)
    assert kernel.termination is not None
    assert kernel.termination.stop_reason == episode.env._stop_reason
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
    assert episode.env._stop_reason == "time_limit"
    assert any(line.startswith("[default] kernel inputs log write failed") for line in episode.logs)
