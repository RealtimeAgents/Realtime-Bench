"""The scripted world both sides of a differential run play (phase2.md, 事件内核, 怎么验证).

A script fixes what the world does and how long everything takes, and nothing about the rules:
the replay side renders it into stub replies and a fake environment for today's loop, the
simulation side into a TurnConfig and a World for the kernel. Times are world seconds from the
episode origin; on the token clock and paused the adapters turn them into environment ticks.

Actions come from one vocabulary, so that neither side calls the other's parser:

    click           a click on the Name field
    type:<text>     typing
    key:<k>[:<n>]   a tap of key k, n times
    hold:<k>:<s>    holding key k for s seconds
    wait[:<s>]      waiting s seconds (none: the default)
    done            the model declaring the task done
    garbage         a reply with no action in it
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from realtime_framework.entity.actions import Action, ActionType

CLOCKS = ("wall", "token", "pause")


@dataclass(frozen=True)
class Attempt:
    """One HTTP request of a turn: answered after ``latency_s``, or failed with ``status``
    before any output, or never answered (``hang``) until the client gives up on it."""

    latency_s: float = 0.5
    status: int | None = None
    hang: bool = False


@dataclass(frozen=True)
class Turn:
    """One generation: its action, the output tokens its usage reports (None: no usage), its
    requests in order, and how the environment takes the action."""

    action: str
    output_tokens: int | None = 8
    attempts: tuple[Attempt, ...] = (Attempt(),)
    rejected: str | None = None  # the guest refuses the action with this message
    fault: str | None = None  # executing the action fails with this message

    @property
    def latency_s(self) -> float:
        """Seconds from the generation's start to its result, when it has one."""
        return sum(attempt.latency_s for attempt in self.attempts)

    @property
    def hangs(self) -> bool:
        return any(attempt.hang for attempt in self.attempts)


@dataclass(frozen=True)
class Script:
    clock: str  # "wall", "token" or "pause"
    duration_s: float
    turns: tuple[Turn, ...]
    env_fps: float = 1.0
    input_fps: float = 1.0
    tokens_per_frame: int | None = None  # B on the token clock; None on the wall clock and paused
    # Wall clock: the seconds each environment operation and the agent's commit take.
    capture_s: float = 0.0
    check_s: float = 0.0
    execute_s: float = 0.0
    commit_s: float = 0.0
    snapshot_s: float = 0.0
    ends_at_s: float | None = None  # the task reports its end from then on
    ends_after: int | None = None  # ... or once this many actions have taken effect
    end_reason: str | None = None
    frozen_from: int | None = None  # from this capture (0-based) on every frame repeats it
    # Token clock, counted 1-based as the kernel asks them: end checks that fail to read, and
    # clock steps that fail.
    check_errors: tuple[int, ...] = ()
    advance_errors: tuple[int, ...] = ()
    snapshot_error: str | None = None
    think_budget_s: float | None = None
    per_move_cap_s: float | None = None
    expect: frozenset[str] = frozenset()  # the expected differences (rules.py) it may show
    name: str = ""

    def __post_init__(self) -> None:
        if self.clock not in CLOCKS:
            raise ValueError(f"unsupported clock {self.clock!r}")
        if (self.tokens_per_frame is not None) != (self.clock == "token"):
            raise ValueError("B is set exactly on the token clock")
        timed = (self.capture_s, self.check_s, self.execute_s, self.commit_s, self.snapshot_s)
        if self.clock != "wall" and any(timed):
            raise ValueError("operation times are world time only on the wall clock")
        if self.clock != "wall" and any(turn.hangs for turn in self.turns):
            raise ValueError("a hang is only bounded on the wall clock")
        if (self.check_errors or self.advance_errors) and self.clock == "wall":
            raise ValueError("wall end checks and waits are not comparable one by one")
        # The guest only executes pointer and text actions on the token clock (key input goes to
        # the task clock), and a wait is never the guest's to refuse.
        refusable = ("click", "type") if self.clock != "wall" else ("click", "type", "key", "hold")
        for turn in self.turns:
            if (turn.rejected or turn.fault) and parse(turn.action).kind not in refusable:
                raise ValueError(f"the environment cannot refuse {turn.action!r} here")

    @property
    def mode(self) -> str:
        """The runner's --clock: pause mode is the token clock without B."""
        return "wall" if self.clock == "wall" else "token"

    @property
    def max_ticks(self) -> float:
        return self.duration_s * self.env_fps

    def but(self, **changes) -> Script:
        return replace(self, **changes)


# ---- the vocabulary ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Word:
    kind: str
    key: str | None = None
    repeat: int = 1
    seconds: float | None = None
    text: str | None = None


def parse(word: str) -> Word:
    kind, *rest = word.split(":")
    if kind in ("click", "done", "garbage") and not rest:
        return Word(kind)
    if kind == "type" and len(rest) == 1:
        return Word(kind, text=rest[0])
    if kind == "key" and len(rest) in (1, 2):
        return Word(kind, key=rest[0], repeat=int(rest[1]) if len(rest) == 2 else 1)
    if kind == "hold" and len(rest) == 2:
        return Word(kind, key=rest[0], seconds=float(rest[1]))
    if kind == "wait" and len(rest) <= 1:
        return Word(kind, seconds=float(rest[0]) if rest else None)
    raise ValueError(f"not in the vocabulary: {word!r}")


def reply_text(word: str) -> str:
    """The model's text_json reply for the agent's own parser."""
    w = parse(word)
    if w.kind == "click":
        return '{"action": "click", "target": "P1"}'  # the Name field, on every frame
    if w.kind == "type":
        return json.dumps({"action": "type", "text": w.text})
    if w.kind == "key":
        body = {"action": "key", "key": w.key}
        return json.dumps(body if w.repeat == 1 else {**body, "repeat": w.repeat})
    if w.kind == "hold":
        return json.dumps({"action": "hold_key", "key": w.key, "duration": w.seconds})
    if w.kind == "wait":
        body = {"action": "wait"}
        return json.dumps(body if w.seconds is None else {**body, "duration": w.seconds})
    if w.kind == "done":
        return '{"action": "done"}'
    return "Let me look at the form again."


def sim_action(word: str) -> tuple[Action, str | None]:
    """The action the kernel is handed, with only what timing reads, and its protocol failure.
    A reply without an action is a wait the agent marks as a protocol failure."""
    w = parse(word)
    if w.kind == "click":
        return Action.click(10, 20), None
    if w.kind == "type":
        return Action(ActionType.TYPE, {"text": w.text}), None
    if w.kind == "key":
        return Action(ActionType.KEY, {"key": w.key, "repeat": w.repeat}), None
    if w.kind == "hold":
        return Action.hold_key(w.key, w.seconds), None
    if w.kind == "wait":
        return Action.wait(w.seconds), None
    if w.kind == "done":
        return Action(ActionType.DONE), None
    return Action.wait(), "no_action"


def action_key(action_type: str, parameters: dict) -> str:
    """An action as both sides can compare it: its type and what timing reads."""
    parameters = parameters or {}
    if action_type == "key":
        return f"key:{str(parameters.get('key')).lower()}:{int(parameters.get('repeat', 1))}"
    if action_type == "hold_key":
        return f"hold:{str(parameters.get('key')).lower()}:{float(parameters['duration']):g}"
    if action_type == "wait":
        return f"wait:{float(parameters.get('duration', 0.0)):g}"
    return action_type


def turns(*words: str, **fields) -> tuple[Turn, ...]:
    return tuple(Turn(word, **fields) for word in words)
