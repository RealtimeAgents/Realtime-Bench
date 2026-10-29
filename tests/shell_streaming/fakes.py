"""A fake environment and a fake streaming agent on the fake clock, for the streaming shell's tests.

Every environment operation takes a set time on the clock and records when it ran, so a test sees
whether two ever overlapped and in which order they held the gate. The agent's session takes a
set time for each push, and its replies follow a script: when each action appears, when the reply
ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from realtime_framework.entity.actions import Action, Observation
from realtime_framework.entity.model import PendingDecision
from realtime_framework.evaluation.kernel.streaming import ActionArrived
from realtime_framework.sandbox.actions import Execution

# The operations that hold the gate alone (evaluation/env_lanes.py).
ALONE = {"check_end", "snapshot"}


class FakeTask:
    def __init__(self, env: FakeEnv):
        self._env = env

    def evaluation_snapshot(self) -> dict:
        self._env._operate("snapshot")
        return {"score": 1}


class FakeEnv:
    """The ShellEnvironment the lanes and the shell use, its operations each taking a set time:
    ``seconds[name]``, or ``durations[(name, n)]`` for its nth time (1-based)."""

    def __init__(self, clock, *, capture_s=0.25, execute_s=2.0, check_s=0.5, snapshot_s=0.5):
        self.clock = clock
        self.t0 = 0.0
        self.d = 1.0
        self.task = FakeTask(self)
        self.scene_capture = None  # a desktop task's
        self.seconds = {
            "capture": capture_s,
            "execute": execute_s,
            "check_end": check_s,
            "snapshot": snapshot_s,
        }
        self.durations: dict[tuple[str, int], float] = {}
        self.ends_at: float | None = None  # the task reports its end from this time on
        self.intervals: list[tuple[str, float, float]] = []
        self.counts: dict[str, int] = {}
        self.running: list[str] = []
        self.concurrent = False  # two operations ran at once
        self.overlapped = False  # one that must run alone ran with another
        self.captures = 0
        self.fail: set[str] = set()
        self.during: dict = {}  # name -> what an operation does halfway through
        self.released_at: float | None = None
        self.on_invalid = None
        self._invalid_run = None

    def _operate(self, name: str) -> None:
        if self.running:
            self.concurrent = True
            self.overlapped |= bool(ALONE & {name, *self.running})
        self.running.append(name)
        self.counts[name] = self.counts.get(name, 0) + 1
        seconds = self.durations.get((name, self.counts[name]), self.seconds[name])
        start = self.clock.monotonic()
        try:
            self.clock.sleep(seconds / 2)
            self.during.pop(name, lambda: None)()
            self.clock.sleep(seconds / 2)
            if name in self.fail:
                raise RuntimeError(f"{name} failed")
        finally:
            self.running.remove(name)
            self.intervals.append((name, start, self.clock.monotonic()))

    def capture(self) -> Observation:
        self._operate("capture")
        self.captures += 1
        return Observation(b"", self.clock.time(), 0, {"observation_id": f"obs-{self.captures}"})

    def execute(self, action, *, release_at=None) -> Execution:
        self._operate("execute")
        return Execution(
            "applied", {"executed": action.action_type.value}, {"x": action.parameters.get("x")}
        )

    def check_end(self) -> tuple[bool, str | None]:
        asked = self.clock.monotonic()
        self._operate("check_end")
        ended = self.ends_at is not None and asked >= self.ends_at
        return ended, "goal" if ended else None

    def release_inputs(self) -> None:
        self.released_at = self.clock.monotonic()


class Inbox:
    """The kernel's inbox as the lanes see it: each event posted, with the time it came."""

    def __init__(self, clock):
        self.clock = clock
        self.events: list[tuple[float, object]] = []
        self._arrived = clock.event()

    def post(self, event) -> None:
        self.events.append((self.clock.monotonic(), event))
        self._arrived.set()

    def wait_for(self, count: int) -> list[tuple[float, object]]:
        while len(self.events) < count:
            self._arrived.clear()
            if len(self.events) < count:
                assert self.clock.wait(self._arrived, 60.0), f"only {len(self.events)} events"
        return self.events


@dataclass(frozen=True)
class Call:
    """One action a reply hands over ``at`` seconds after it starts."""

    at: float
    action: Action
    call_id: str | None = None


@dataclass(frozen=True)
class Reply:
    latency: float = math.inf  # seconds from its start until it ends; inf: until it is aborted
    calls: tuple[Call, ...] = ()


class FakeAgent:
    """A StreamingKernelAgent whose session takes ``push_s`` per frame and whose replies follow
    ``replies``; once they run out, a reply never ends. An action names the newest frame pushed
    when it appears."""

    def __init__(self, clock, replies: list[Reply], *, push_s=0.125):
        self.clock = clock
        self.push_s = push_s
        self._replies = iter(replies)
        self.post = None
        self.origin: float | None = None
        self.pushed: list[tuple[str, float]] = []  # (observation id, when it went out)
        self.started: list[tuple[int, str, float]] = []  # (generation, frame, when)
        self.ended: list[tuple[int, float]] = []
        self.aborted: list[tuple[int, float]] = []
        self.accepted: list[int] = []
        self.discarded: list[int] = []
        self.committed: list = []
        self.pushes_ended: list[float] = []
        self._cancel: dict[int, object] = {}

    def bind_inbox(self, post, origin) -> None:
        self.post, self.origin = post, origin

    def end_pushes(self) -> None:
        self.pushes_ended.append(self.clock.monotonic())

    def push(self, observation: Observation) -> None:
        self.clock.sleep(self.push_s)
        self.pushed.append((observation.metadata["observation_id"], self.clock.monotonic()))

    def generate(self, generation, decision, observation, budget) -> PendingDecision:
        reply = next(self._replies, Reply())
        cancel = self._cancel[generation] = self.clock.event()
        start = self.clock.monotonic()
        self.started.append((generation, observation.metadata["observation_id"], start))
        for call in reply.calls:
            if self.clock.wait(cancel, start + call.at - self.clock.monotonic()):
                break
            frame = len(self.pushed)
            age = round(self.clock.monotonic() - self.pushed[-1][1], 9)
            self.post(
                ActionArrived(
                    generation, call.action, self.clock.monotonic(), call.call_id, frame, age, 0
                )
            )
        else:
            remaining = start + reply.latency - self.clock.monotonic()
            self.clock.wait(cancel, None if math.isinf(remaining) else remaining)
        self.ended.append((generation, self.clock.monotonic()))
        return PendingDecision(generation, decision, {"action": None, "latency": 0.0})

    def abort(self, generation=None) -> None:
        self.aborted.append((generation, self.clock.monotonic()))
        self._cancel[generation].set()

    def accept(self, generation) -> None:
        self.accepted.append(generation)

    def discard(self, generation) -> None:
        self.discarded.append(generation)

    def commit(self, outcome) -> None:
        self.committed.append((self.clock.monotonic(), outcome))
