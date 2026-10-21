"""A discrete-event simulation shell for the streaming policy on the event kernel (phase2.md,
native streaming).

No threads. A scripted world says how long each environment operation, frame push and reply takes,
when a reply hands over each of its actions and cancels calls, and from when the task reports its
end. Time is simulated wall time: ``ops`` is both clocks. Capturing, pushing, end checks and
executing each run on a lane of their own, standing for the shell's other threads, and the answer
is posted when the command's time is up, so the kernel goes on handling events meanwhile. A lane
takes one command at a time; one more while it is busy is a policy bug.

As in the streaming shell (evaluation/env_lanes.py), captures, end checks, actions and the snapshot
share one gate, first come first served: an operation reads the task, presses keys or takes its
frame once the gate is its, not when it was asked for. The snapshot runs on a thread of its own and
goes next after the operation holding the gate, ahead of those waiting (``EnvLanes.freeze_turn``);
the shell bounds that wait by the action's budget, which no script here runs into. Commit runs on
the shell's own thread and occupies it.

Every command is checked as it runs (one generation in flight at most; nothing captured, pushed,
started, asked or executed at or after the deadline or after the end), and every finished episode
against what the policy keeps over a whole run (``check``).
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace

from realtime_framework.entity.actions import Action
from realtime_framework.entity.records import TraceEvent
from realtime_framework.evaluation.kernel.core import Kernel
from realtime_framework.evaluation.kernel.events import (
    AbortGeneration,
    AcceptResult,
    ActionExecuted,
    Capture,
    CheckEnd,
    Commit,
    DiscardResult,
    EndChecked,
    Entry,
    Execute,
    Finish,
    FrameCaptured,
    FreezeSnapshot,
    GenerationFinished,
    GenerationResult,
    Invalidated,
    KeysReleased,
    OutcomeCommitted,
    Record,
    ReleaseInputs,
    SetTimer,
    SnapshotFrozen,
    StartGeneration,
)
from realtime_framework.evaluation.kernel.streaming import (
    CANCELLED_CALL,
    CANCELLED_GENERATION,
    SUPERSEDED,
    ActionArrived,
    CallsCancelled,
    FramePushed,
    PushFrame,
    StreamingConfig,
    StreamingPolicy,
)
from realtime_framework.harnesses.default.frames import INVALID_GROUNDING_FRAME
from tests.kernel.simulation import key_identities

# Commands that act in the world, which nothing may run at or after the deadline or the end.
IN_WORLD = (Capture, PushFrame, StartGeneration, CheckEnd, Execute)
ENDED = {"invalid_run", "not_applied_time_limit", "not_applied_terminal"}
REFUSED_LATE = ENDED | {CANCELLED_GENERATION}  # an action its reply hands over once given up
WITHDRAWN = ENDED | {CANCELLED_CALL, CANCELLED_GENERATION}  # a cancelled call's action not run
# The records written past the deadline before it ends the episode, besides the events' echoes.
LATE_RECORDS = {"TimerFired", TraceEvent.ACTION_PROPOSED.value, "ResultDiscarded", "RunInvalidated"}


@dataclass(frozen=True)
class Call:
    """One action a reply hands over ``at`` seconds after it starts."""

    at: float
    action: Action | None
    call_id: str | None = None
    frame: int | None = None  # the frame it names; None: the newest frame sent when it arrives
    grounding: str | None = None  # why its frame cannot be found (frames.MISSING_FRAME, ...)
    execute_s: float | None = None  # how long it executes; None: World.execute_s
    rejection: str | None = None  # the guest refuses it
    error: str | None = None  # the environment fails on it


@dataclass(frozen=True)
class Reply:
    """One reply of the live session. As the agent posts them, its actions come before its end,
    and each before a cancellation of its call."""

    latency: float  # seconds from its start until it ends
    calls: tuple[Call, ...] = ()
    cancellations: tuple[tuple[float, tuple[str, ...]], ...] = ()  # (at, call ids it cancels)
    protocol_failure: str | None = None
    invalid: Invalidated | None = None
    output_tokens: int | None = None
    # False when its messages keep coming after an abort, as from the old connection.
    interruptible: bool = True


@dataclass
class World:
    """The scripted environment and live session of one episode. Once the replies run out, a
    reply never ends."""

    replies: list[Reply] = field(default_factory=list)
    ends_at: float | None = None  # the task reports its end from this time on
    end_reason: str | None = None
    start_s: float = 0.0  # the policy starts this long after the episode origin
    invalidations: tuple[tuple[float, Invalidated], ...] = ()  # posted by other threads
    capture_s: float = 0.0
    push_s: float = 0.0
    check_s: float = 0.0
    execute_s: float = 0.0
    commit_s: float = 0.0
    snapshot_s: float = 0.0
    # The nth (1-based) capture, push or end check -> its own duration, or its fault.
    capture_times: dict[int, float] = field(default_factory=dict)
    push_times: dict[int, float] = field(default_factory=dict)
    check_times: dict[int, float] = field(default_factory=dict)
    capture_errors: dict[int, str] = field(default_factory=dict)
    push_errors: dict[int, str] = field(default_factory=dict)
    check_errors: dict[int, str] = field(default_factory=dict)
    commit_errors: dict[int, str] = field(default_factory=dict)  # the nth Commit -> fault
    # Key -> the fault that makes the environment's release timer invalidate the run, which the
    # shell passes on from on_invalid.
    release_errors: dict[str, str] = field(default_factory=dict)
    snapshot_error: str | None = None
    frozen_from: int | None = None  # from this capture (0-based) on every frame repeats itself


class Simulation:
    """Runs one streaming episode against a World and keeps everything it saw."""

    def __init__(self, config: StreamingConfig, world: World) -> None:
        self.config = config
        self.world = world
        self.policy = StreamingPolicy(config)
        self.kernel = Kernel(self.policy)
        self.ops = world.start_s
        self.records: list[Entry] = []
        self.commands: list = []
        self.checks: list[float] = []  # when each end check read the task
        self.captures: list[float] = []  # when each capture took its frame
        # (operation, asked, took the gate, let it go) of every operation on the gate.
        self.operations: list[tuple[str, float, float, float]] = []
        self.pushes: list[tuple[str | None, int]] = []  # (observation, frame) of each push
        self.sent: dict[int, float] = {}  # frame -> its capture time, once it went out
        self.executed: list[tuple[float, Action]] = []  # (start, action)
        self.key_events: list[tuple[float, tuple[str, ...]]] = []  # (time, keys released)
        self.outcomes: list = []
        self.started: list[tuple[int, float]] = []  # (generation, time)
        self.aborted: list[tuple[int, float]] = []
        self.accepted: list[int] = []
        self.discarded: list[int] = []
        self.snapshots: list[float] = []
        self.released_inputs = False
        self.termination = None
        self._ended = False
        self._queue: deque = deque()
        self._pending: list = []  # (time, order, work) of other threads
        self._order = itertools.count()
        self._wakes: list[float] = []
        self._replies = iter(world.replies)
        self._reply_of: dict[int, Reply] = {}
        self._posts: dict[int, list] = {}  # generation -> its pending posts
        self._finished: set[int] = set()  # generations whose result was posted
        self._in_flight: int | None = None
        self._busy: set[str] = set()  # the lanes running a command
        self._holder: str | None = None  # the operation holding the gate
        self._line: deque[Callable] = deque()  # the operations waiting for it, in order
        self._observed: dict[str | None, float] = {}  # observation -> capture time
        self._calls: dict[int, tuple[Action, Call]] = {}  # id(action) -> (action, its call)
        self._releases: dict[str, float] = {}  # key -> its latest release time
        self._frozen_tick: float | None = None

    # ---- driving ----------------------------------------------------------------------------

    def run(self, max_steps: int = 100_000) -> Simulation:
        for at, event in self.world.invalidations:
            self._later(at, lambda event=event: self.kernel.post(event))
        self._queue.extend(self.kernel.start(self.ops))
        for _ in range(max_steps):
            if self._queue:
                self._run(self._queue.popleft())
                continue
            if self.termination is not None:
                self.check()
                return self
            out = self.kernel.step(self.ops)
            if out is not None:
                self._queue.extend(out)
            else:
                self._wait()
        raise RuntimeError("the simulated episode did not finish")

    def kinds(self) -> list[str]:
        return [entry.kind for entry in self.records]

    def entries(self, kind: str) -> list[Entry]:
        return [entry for entry in self.records if entry.kind == kind]

    def data(self, kind: str) -> list:
        return [entry.data for entry in self.entries(kind)]

    def resolved(self) -> dict[int, dict]:
        """Each action's outcome record, by its index."""
        return {data["index"]: data for data in self.data("ActionResolved")}

    def statuses(self) -> list[str]:
        """The outcome of each action, in arrival order."""
        resolved = self.resolved()
        return [resolved[index]["status"] for index in sorted(resolved)]

    def _wait(self) -> None:
        """Nothing is due: move the time to the next thing that will happen."""
        self._wakes = [at for at in self._wakes if at > self.ops]
        times = list(self._wakes)
        if self._pending:
            times.append(self._pending[0][0])
        if not times:
            raise RuntimeError("the simulated episode stalled")
        self._spend_until(min(times))

    def _spend(self, seconds: float) -> None:
        self._spend_until(self.ops + seconds)

    def _spend_until(self, until: float) -> None:
        while self._pending and self._pending[0][0] <= until:
            entry = heapq.heappop(self._pending)
            self.ops = max(self.ops, entry[0])
            entry[2]()
        self.ops = max(self.ops, until)

    def _later(self, at: float, work: Callable) -> list:
        entry = [at, next(self._order), work]
        heapq.heappush(self._pending, entry)
        return entry

    def _lane(self, lane: str, seconds: float, answer: Callable) -> None:
        """Run a command off the gate on its lane; ``answer`` posts its result when it is done."""
        assert lane not in self._busy, f"the {lane} lane is busy"
        self._busy.add(lane)

        def finish() -> None:
            self._busy.discard(lane)
            answer()

        self._later(self.ops + seconds, finish)

    def _gated(self, lane: str, seconds: float, run: Callable, *, ahead: bool = False) -> None:
        """Run an operation on its lane once it holds the gate: ``run()`` acts on the world then
        and returns what posts its answer once the operation lets the gate go, ``seconds`` later.
        ``ahead`` puts it next after the holder, ahead of those waiting."""
        assert lane not in self._busy, f"the {lane} lane is busy"
        self._busy.add(lane)
        asked = self.ops

        def take() -> None:
            self._holder, started = lane, self.ops
            answer = run()

            def let_go() -> None:
                self.operations.append((lane, asked, started, self.ops))
                self._busy.discard(lane)
                self._holder = None
                if self._line:
                    self._line.popleft()()
                answer()

            self._later(started + seconds, let_go)

        if self._holder is None:
            take()
        elif ahead:
            self._line.appendleft(take)
        else:
            self._line.append(take)

    # ---- commands ---------------------------------------------------------------------------

    def _run(self, command) -> None:
        self.commands.append(command)
        if isinstance(command, IN_WORLD):
            assert not self._ended, f"{command!r} after the end"
            assert self.ops < self.config.duration_s, (
                f"{command!r} at {self.ops}, past the deadline"
            )
        world, kernel = self.world, self.kernel
        match command:
            case Record(entry=entry):
                self.records.append(entry)
                self._ended = self._ended or entry.kind == "EpisodeEnded"
            case SetTimer(at=at):
                self._wakes.append(at)
            case CheckEnd(request=request):
                self._check(request)
            case Capture():
                self._capture()
            case PushFrame(observation_id=observation_id):
                self._push(observation_id)
            case StartGeneration(generation=generation):
                self._start(generation)
            case AbortGeneration(generation=generation):
                self._abort(generation)
            case AcceptResult(generation=generation):
                self.accepted.append(generation)
                self._in_flight = None
            case DiscardResult(generation=generation):
                self.discarded.append(generation)
            case Execute(decision=decision, action=action, release_at=release_at):
                self._execute(decision, action, release_at)
            case Commit(outcome=outcome):
                self._spend(world.commit_s)
                self.outcomes.append(outcome)
                error = world.commit_errors.get(len(self.outcomes))
                kernel.post(OutcomeCommitted(outcome.decision, error))
            case FreezeSnapshot():
                self._gated("snapshot", world.snapshot_s, lambda: self._frozen, ahead=True)
            case ReleaseInputs():
                self.released_inputs = True
            case Finish(termination=termination):
                self.termination = termination
            case _:
                raise AssertionError(f"a streaming episode does not run {command!r}")

    def _frozen(self) -> None:
        """The snapshot's answer, once its read returned."""
        self.snapshots.append(self.ops)
        self.kernel.post(SnapshotFrozen(self.ops, self.ops, self.world.snapshot_error))

    def _check(self, request: int) -> None:
        """The task answers what holds once the read has the gate."""
        world, index = self.world, sum(name == "check" for name, *_ in self.operations) + 1

        def read() -> Callable:
            self.checks.append(self.ops)
            error = world.check_errors.get(index)
            ended = error is None and world.ends_at is not None and self.ops >= world.ends_at
            reason = world.end_reason if ended else None
            return lambda: self.kernel.post(EndChecked(request, ended, reason, error))

        self._gated("check", world.check_times.get(index, world.check_s), read)

    def _capture(self) -> None:
        """The frame is the one on screen once the capture has the gate."""
        world, index = self.world, len(self._observed)
        observation_id = f"obs-{index}"
        self._observed[observation_id] = None

        def take() -> Callable:
            self.captures.append(self.ops)
            tick = math.floor(self.ops * self.config.env_fps + 1e-9)
            timestamp = self._observed[observation_id] = self.ops
            error = world.capture_errors.get(index + 1)
            frame = index
            if world.frozen_from is not None and index >= world.frozen_from:
                frame = world.frozen_from
                self._frozen_tick = tick if self._frozen_tick is None else self._frozen_tick
                tick = self._frozen_tick
            fingerprint = (f"frame-{frame}", tick, f"sha-{frame}")
            event = FrameCaptured(observation_id, tick, timestamp, fingerprint, error)
            return lambda: self.kernel.post(event)

        self._gated("capture", world.capture_times.get(index + 1, world.capture_s), take)

    def _push(self, observation_id: str | None) -> None:
        """The frame gets the next frame-table number, as the agent registers it to push it, and
        is marked sent if the push goes through."""
        world = self.world
        frame = len(self.pushes) + 1
        self.pushes.append((observation_id, frame))
        error = world.push_errors.get(frame)

        def pushed() -> None:
            if error is None:
                self.sent[frame] = self._observed[observation_id]
            self.kernel.post(FramePushed(observation_id, self.ops, error))

        self._lane("push", world.push_times.get(frame, world.push_s), pushed)

    def _start(self, generation: int) -> None:
        assert self._in_flight is None, f"generation {generation} starts with another in flight"
        self._in_flight = generation
        self.started.append((generation, self.ops))
        reply = next(self._replies, None) or Reply(latency=math.inf)
        arrivals = {call.call_id: call.at for call in reply.calls if call.call_id is not None}
        assert all(call.at <= reply.latency for call in reply.calls), "an action after its end"
        for at, call_ids in reply.cancellations:
            assert all(arrivals.get(c, at) <= at for c in call_ids), "a cancel before its call"
        self._reply_of[generation] = reply
        start, kernel = self.ops, self.kernel
        posts = self._posts[generation] = []
        for call in reply.calls:
            posts.append(
                self._later(start + call.at, lambda call=call: self._arrive(generation, call))
            )
        for at, call_ids in reply.cancellations:
            posts.append(
                self._later(
                    start + at,
                    lambda call_ids=call_ids: kernel.post(CallsCancelled(generation, call_ids)),
                )
            )
        if math.isfinite(reply.latency):
            result = GenerationResult(
                output_tokens=reply.output_tokens,
                protocol_failure=reply.protocol_failure,
                invalid=reply.invalid,
            )
            posts.append(
                self._later(start + reply.latency, lambda: self._finish(generation, result))
            )

    def _finish(self, generation: int, result: GenerationResult) -> None:
        self._finished.add(generation)
        self.kernel.post(GenerationFinished(generation, result))

    def _arrive(self, generation: int, call: Call) -> None:
        """Post the call's action with the frame it names and that frame's age now, as the agent
        does with FrameTable.locate and FrameTable.age: the frames sent after it, not those
        whose push failed."""
        if call.grounding is not None:
            event = ActionArrived(
                generation,
                None,
                self.ops,
                call.call_id,
                rejection=INVALID_GROUNDING_FRAME,
                detail=call.grounding,
            )
        else:
            newest = max(self.sent)
            frame = newest if call.frame is None else call.frame
            # One object per arrival, so that Execute names the call it came in.
            action = replace(call.action)
            self._calls[id(action)] = (action, call)
            event = ActionArrived(
                generation,
                action,
                self.ops,
                call.call_id,
                frame,
                round(self.ops - self.sent[frame], 9),
                sum(sent > frame for sent in self.sent),
            )
        self.kernel.post(event)

    def _abort(self, generation: int) -> None:
        """An interruptible reply stops handing anything over and hands back its result now."""
        self.aborted.append((generation, self.ops))
        if self._in_flight == generation:
            self._in_flight = None
        if not self._reply_of[generation].interruptible:
            return
        for entry in self._posts[generation]:
            entry[2] = lambda: None
        if generation not in self._finished:
            self._later(self.ops, lambda: self._finish(generation, GenerationResult()))

    def _execute(self, decision: int, action: Action, release_at: float | None) -> None:
        """The action presses once it has the gate; its answer says when it took the gate and
        when it let it go."""
        world = self.world
        _, call = self._calls[id(action)]
        seconds = world.execute_s if call.execute_s is None else call.execute_s

        def press() -> Callable:
            started = self.ops
            self.executed.append((started, action))
            if call.error is None and call.rejection is None and release_at is not None:
                # The release timer is armed right after the press; a key already held until
                # later keeps its later release.
                for key in key_identities(action):
                    at = max(self._releases.get(key, release_at), release_at)
                    self._releases[key] = at
                    self._later(at, lambda key=key, at=at: self._release(key, at))
            result = {"action": action.action_type.value}

            def answer() -> None:
                self.kernel.post(
                    ActionExecuted(
                        decision,
                        result,
                        rejection=call.rejection,
                        error=call.error,
                        started_at=started,
                        finished_at=self.ops,
                    )
                )

            return answer

        self._gated("execute", seconds, press)

    def _release(self, key: str, at: float) -> None:
        if self._releases.get(key) != at:
            return
        del self._releases[key]
        error = self.world.release_errors.get(key)
        if error is not None:
            self.kernel.post(Invalidated("environment", "wall_key_release_failed", error))
            return
        self.key_events.append((self.ops, (key,)))
        self.kernel.post(KeysReleased((key,)))

    # ---- what every run keeps ---------------------------------------------------------------

    def check(self) -> None:
        records, commands = self.records, self.commands
        assert [entry.seq for entry in records] == list(range(1, len(records) + 1))
        # The termination record is written last; then inputs are released and the episode ends.
        assert records[-1].kind == "Termination"
        assert records[-1].data == asdict(self.termination)
        assert commands[-2:] == [ReleaseInputs(), Finish(self.termination)]
        assert self.kinds().count("EpisodeEnded") == 1
        ended = next(
            index
            for index, command in enumerate(commands)
            if isinstance(command, Record) and command.entry.kind == "EpisodeEnded"
        )
        assert isinstance(commands[ended + 1], FreezeSnapshot)
        # An interruption is written off before it is sent, and a stale result never lands.
        for index, command in enumerate(commands):
            if isinstance(command, AbortGeneration):
                written = commands[index - 1].entry
                assert (written.kind, written.data["generation"]) == (
                    "GenerationAbandoned",
                    command.generation,
                )
        assert not set(self.accepted) & set(self.discarded)
        # The policy keeps one environment operation out at a time: each one it asks for takes
        # the gate when it is asked, so it acts in the world before the deadline, as asked.
        for name, asked, started, _ in self.operations:
            if name != "snapshot":
                assert started == asked, f"the {name} asked at {asked} waited until {started}"
        # Every action that arrived has exactly one outcome, recorded and committed; the nth
        # arrival is action n. Only a real action is a proposal, as on the turn-based clock.
        arrivals = self.entries("ActionArrived")
        outcomes = self.resolved()
        assert len(self.data("ActionResolved")) == len(arrivals)
        assert sorted(outcomes) == list(range(len(arrivals)))
        assert len([o for o in self.outcomes if o.arrived_at is not None]) == len(arrivals)
        proposed = {data["index"] for data in self.data("ActionProposed")}
        for index, data in outcomes.items():
            assert (index in proposed) == (
                data["action"] is not None and data["protocol_failure"] is None
            )
        # Actions that Execute ran did so in arrival order, one after another; a superseded one
        # names the later action that took its place.
        executed = sorted(
            (data["started_at"], index)
            for index, data in outcomes.items()
            if data["started_at"] is not None and data["status"] != "wait"
        )
        assert len(executed) == len(self.executed)
        assert [index for _, index in executed] == sorted(index for _, index in executed)
        for (_, before), (_, after) in itertools.pairwise(executed):
            assert outcomes[before]["finished_at"] <= outcomes[after]["started_at"]
        for index, data in outcomes.items():
            assert (data["status"] == SUPERSEDED) == (data["superseded_by"] is not None)
            if data["status"] == SUPERSEDED:
                assert data["superseded_by"] > index
        # A reply reported as having come to no action handed none over, and whatever a reply
        # handed over after it was given up was refused.
        generation_of = {
            data["decision"]: data["generation"] for data in self.data("InferenceStarted")
        }
        handed = {entry.data.generation for entry in arrivals}
        for outcome in self.outcomes:
            if outcome.status == "none":
                assert generation_of[outcome.decision] not in handed
        given_up = {
            entry.data["generation"]: entry.seq for entry in self.entries("GenerationAbandoned")
        }
        for index, entry in enumerate(arrivals):
            if entry.seq > given_up.get(entry.data.generation, math.inf):
                assert outcomes[index]["status"] in REFUSED_LATE
        # A cancellation withdraws the action of its call unless it executes already.
        resolved_at = {entry.data["index"]: entry.seq for entry in self.entries("ActionResolved")}
        calls = {(entry.data.generation, entry.data.call_id): i for i, entry in enumerate(arrivals)}
        for entry in self.entries("CallsCancelled"):
            for call_id in entry.data.call_ids:
                index = calls.get((entry.data.generation, call_id))
                if index is None or resolved_at[index] < entry.seq:
                    continue
                data = outcomes[index]
                assert data["status"] in WITHDRAWN or (
                    data["started_at"] is not None and data["started_at"] <= entry.semantic
                )
        # Past the deadline nothing is decided before the deadline ends the episode, but an
        # arrival's record, a stale result and an invalidation that comes first.
        end = self.entries("EpisodeEnded")[0].seq
        for entry in records[: end - 1]:
            if entry.semantic >= self.config.duration_s and isinstance(entry.data, dict):
                assert entry.kind in LATE_RECORDS, entry
