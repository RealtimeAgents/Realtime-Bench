"""A discrete-event simulation shell for the streaming policy on the event kernel (phase2.md,
native streaming).

No threads. A scripted world says how long each environment operation, frame push and reply takes,
when a reply hands over each of its actions and cancels calls, and from when the task reports its
end. Time is simulated wall time: ``ops`` is both clocks. Capturing, pushing, end checks and
executing each run on a lane of their own, standing for the shell's other threads, and the answer
is posted when the command's time is up, so the kernel goes on handling events meanwhile. A lane
takes one command at a time; one more while it is busy is a policy bug.

The environment has a lock by resource, as the streaming shell's gate has it (evaluation/
env_lanes.py, decision 50): a capture and an action may hold it together, an end check and the
snapshot only alone, first come first served. An operation reads the task, presses keys or takes
its frame once it holds the lock, not when it was asked for. The snapshot runs on a thread of its
own and goes ahead of those waiting, after those holding the lock (``EnvLanes.freeze_turn``); the
shell bounds that wait by the larger of the action's budget and the capture's (N32), which no
script here runs into. Commit runs on the shell's own thread and occupies it; an abort runs on a
lane of its own, off the lock.

Every command is checked as it runs (one generation in flight at most; nothing captured, pushed,
started, asked or executed at or after the deadline or after the end, but the read after an action
that finished before the deadline), and every finished episode against what the policy keeps over
a whole run (``check``).
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace

from realtime_framework.entity.actions import Action, ActionType
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
    GenerationAborted,
    PushFrame,
    SessionPrepared,
    SessionPreparing,
    StreamingConfig,
    StreamingPolicy,
)
from realtime_framework.harnesses.default.frames import INVALID_GROUNDING_FRAME
from tests.kernel.simulation import key_identities

# Commands that act in the world, which nothing may run at or after the deadline or the end.
IN_WORLD = (Capture, PushFrame, StartGeneration, CheckEnd, Execute)
# The operations that hold the environment's lock alone; a capture and an action share it.
ALONE = {"check", "snapshot"}
ENDED = {"invalid_run", "not_applied_time_limit", "not_applied_terminal"}
REFUSED_LATE = ENDED | {CANCELLED_GENERATION}  # an action its reply hands over once given up
WITHDRAWN = ENDED | {CANCELLED_CALL, CANCELLED_GENERATION}  # a cancelled call's action not run
# The records written past the deadline before it ends the episode, besides the events' echoes:
# none decides anything, also while the end waits for a read, and the reply in flight is given up.
# A frame is dropped then only as a push failed.
LATE_RECORDS = {
    "TimerFired",
    "GenerationAbandoned",
    TraceEvent.ACTION_PROPOSED.value,
    "CallCancelled",
    "FrameDropped",
    "ResultDiscarded",
    "RunInvalidated",
}
OUTCOME_RECORDS = {TraceEvent.ACTION_RESOLVED.value, TraceEvent.ACTION_APPLIED.value}


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
    # (time, event) other threads post, such as an invalidation.
    posts: tuple[tuple[float, object], ...] = ()
    capture_s: float = 0.0
    push_s: float = 0.0
    check_s: float = 0.0
    execute_s: float = 0.0
    commit_s: float = 0.0
    # An abort runs this long on its lane, until the agent closed the session.
    abort_s: float = 0.0
    snapshot_s: float = 0.0
    # The nth (1-based) capture, push or end check -> its own duration, or its fault.
    capture_times: dict[int, float] = field(default_factory=dict)
    push_times: dict[int, float] = field(default_factory=dict)
    check_times: dict[int, float] = field(default_factory=dict)
    capture_errors: dict[int, str] = field(default_factory=dict)
    push_errors: dict[int, str] = field(default_factory=dict)
    # The nth push hangs until the next abort closes the session, which fails it.
    stuck_pushes: set[int] = field(default_factory=set)
    # The nth push waits for the session's renewal, which waits for the reply on it to end or be
    # given up, and the session is being prepared meanwhile (SessionPreparing, then
    # SessionPrepared), as the agent says after a goAway.
    renewal_pushes: set[int] = field(default_factory=set)
    # The nth push prepares the session this long first, as through the backoff between attempts
    # to reopen it, which no abort ends, and says so (agent.py _ready).
    backoff_pushes: dict[int, float] = field(default_factory=dict)
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
        # (operation, asked, took the lock, let it go, the order it took the lock in) of every
        # operation on the lock.
        self.operations: list[tuple[str, float, float, float, int]] = []
        self._takes = itertools.count()
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
        self._stuck_push: str | None = None  # the frame a stuck push holds
        # (the reply it waits for, what goes on then) of a push held for the session's renewal
        self._held_push: tuple[int, Callable] | None = None
        self._holders: set[str] = set()  # the operations holding the lock
        self._line: deque[tuple[bool, Callable]] = deque()  # (alone, take) waiting, in order
        self._observed: dict[str | None, float] = {}  # observation -> capture time
        self._calls: dict[int, tuple[Action, Call]] = {}  # id(action) -> (action, its call)
        self._releases: dict[str, float] = {}  # key -> its latest release time
        self._frozen_tick: float | None = None
        self._read_owed = False  # an action finished before the deadline, and no read since

    # ---- driving ----------------------------------------------------------------------------

    def run(self, max_steps: int = 100_000) -> Simulation:
        for at, event in self.world.posts:
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
        """Run an operation on its lane once it holds the lock: ``run()`` acts on the world then
        and returns what posts its answer once the operation lets the lock go, ``seconds`` later.
        An end check or the snapshot holds it alone, a capture or an action with each other; an
        operation waits behind those that asked before it, or with ``ahead`` before them all."""
        assert lane not in self._busy, f"the {lane} lane is busy"
        self._busy.add(lane)
        asked, alone = self.ops, lane in ALONE

        def take() -> None:
            self._holders.add(lane)
            started, order = self.ops, next(self._takes)
            answer = run()

            def let_go() -> None:
                self.operations.append((lane, asked, started, self.ops, order))
                self._busy.discard(lane)
                self._holders.discard(lane)
                while self._line and self._may_take(self._line[0][0]):
                    self._line.popleft()[1]()
                answer()

            self._later(started + seconds, let_go)

        if not self._line and self._may_take(alone):
            take()
        elif ahead:
            self._line.appendleft((alone, take))
        else:
            self._line.append((alone, take))

    def _may_take(self, alone: bool) -> bool:
        if alone:
            return not self._holders
        return not self._holders & ALONE

    # ---- commands ---------------------------------------------------------------------------

    def _run(self, command) -> None:
        self.commands.append(command)
        if isinstance(command, IN_WORLD):
            assert not self._ended, f"{command!r} after the end"
            # Past the deadline only the read after an action that finished before it (N31).
            owed = isinstance(command, CheckEnd) and self._read_owed
            assert owed or self.ops < self.config.duration_s, (
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
            self._read_owed = False
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
            self._busy.discard("push")
            if error is None:
                self.sent[frame] = self._observed[observation_id]
            self.kernel.post(FramePushed(observation_id, self.ops, error))

        def go() -> None:
            if frame in world.stuck_pushes:
                self._stuck_push = observation_id
            else:
                self._later(self.ops + world.push_times.get(frame, world.push_s), pushed)

        def resume() -> None:
            self.kernel.post(SessionPrepared())
            go()

        assert "push" not in self._busy, "the push lane is busy"
        self._busy.add("push")
        if frame in world.renewal_pushes and self._in_flight is not None:
            self.kernel.post(SessionPreparing())
            self._held_push = (self._in_flight, resume)
            return
        if frame in world.backoff_pushes:
            self.kernel.post(SessionPreparing())
            self._later(self.ops + world.backoff_pushes[frame], resume)
            return
        go()

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
        self._off(generation)
        self.kernel.post(GenerationFinished(generation, result))

    def _off(self, generation: int) -> None:
        """The reply is off the session: a push held for its renewal goes on."""
        if self._held_push is not None and self._held_push[0] == generation:
            _, resume = self._held_push
            self._held_push = None
            resume()

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
        """An interruptible reply stops handing anything over at once, and hands back its result
        once the abort returned, ``abort_s`` later on its lane. Closing the session at the end of
        it fails a push stuck in it; then the abort is answered."""
        self.aborted.append((generation, self.ops))
        if self._in_flight == generation:
            self._in_flight = None
        interruptible = self._reply_of[generation].interruptible
        if interruptible:
            for entry in self._posts[generation]:
                entry[2] = lambda: None

        def returned() -> None:
            if self._stuck_push is not None:
                observation_id, self._stuck_push = self._stuck_push, None
                self._busy.discard("push")
                self.kernel.post(FramePushed(observation_id, self.ops, "the session was closed"))
            self._off(generation)
            if interruptible and generation not in self._finished:
                self._later(self.ops, lambda: self._finish(generation, GenerationResult()))
            self.kernel.post(GenerationAborted(generation))

        self._lane("abort", self.world.abort_s, returned)

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
                self._read_owed = self.ops < self.config.duration_s
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
        """The environment's own release timer. It tells the kernel nothing of a release that
        worked; one that failed for good goes through on_invalid (sandbox/actions.py
        GuestEnvironment._wall_release_due)."""
        if self._releases.get(key) != at:
            return
        del self._releases[key]
        error = self.world.release_errors.get(key)
        if error is not None:
            self.kernel.post(Invalidated("environment", "wall_key_release_failed", error))
            return
        self.key_events.append((self.ops, (key,)))

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
        # Every abort is answered before the episode finishes.
        aborts = [c.generation for c in commands if isinstance(c, AbortGeneration)]
        assert [entry.data.generation for entry in self.entries("GenerationAborted")] == aborts
        # The policy sends a read only while no capture or action is out, and neither while a
        # read is out: each takes the lock when it is asked, so it acts in the world before the
        # deadline, as asked, and no read overlaps a capture or an action.
        gated = sorted((op for op in self.operations if op[0] != "snapshot"), key=lambda op: op[4])
        for name, asked, started, *_ in gated:
            assert started == asked, f"the {name} asked at {asked} waited until {started}"
        reads = [op for op in gated if op[0] == "check"]
        for read in reads:
            for op in gated:
                if op[0] != "check":
                    assert not (op[2] < read[3] and read[2] < op[3]), f"{op} overlaps {read}"
        # The state is read at most once a frame, but right before an action that arrived since
        # the read before (decision 12) and right after one; an action applied that finished
        # before the deadline is followed by that read before another action executes
        # (decision 31), also when the deadline comes while it waits (N31), unless the run turns
        # invalid first.
        end = self.entries("EpisodeEnded")[0].seq
        executions = [op for op in gated if op[0] == "execute"]
        asks = [i for i, command in enumerate(commands) if isinstance(command, CheckEnd)]
        acts = [
            i
            for i, command in enumerate(commands)
            if isinstance(command, Execute)
            or isinstance(command, Record)
            and command.entry.kind == "ActionArrived"
        ]
        for (before, asked), (after, then) in itertools.pairwise(zip(reads, asks, strict=True)):
            frame = math.floor(after[2] * self.config.env_fps + 1e-9)
            if frame == math.floor(before[2] * self.config.env_fps + 1e-9):
                assert any(asked < i < then for i in acts), f"frame {frame} read twice"
        applied = {
            data["started_at"]: data
            for data in self.resolved().values()
            if data["status"] == "applied" and data["action"].action_type is not ActionType.WAIT
        }
        for execution, following in itertools.zip_longest(executions, executions[1:]):
            data = applied.get(execution[2])
            if (
                data is None
                or data["terminal_after_action"]
                and (execution[3] >= self.config.duration_s or self.termination.cause == "invalid")
            ):
                continue
            until = math.inf if following is None else following[4]
            assert any(execution[4] < read[4] < until for read in reads), (
                f"no read after {execution}"
            )
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
        # The agent is told of each action as it was recorded, in the order it was settled: the
        # call and decision it came in, when it arrived, its status, whether it was the last, and
        # where it landed, in environment ticks.
        decision_of = {
            data["generation"]: data["decision"] for data in self.data("InferenceStarted")
        }
        told = [outcome for outcome in self.outcomes if outcome.arrived_at is not None]
        for outcome, entry in zip(told, self.entries("ActionResolved"), strict=True):
            data = entry.data
            landed = entry.semantic if data["started_at"] is None else data["started_at"]
            assert (
                outcome.decision,
                outcome.call_id,
                outcome.arrived_at,
                outcome.status,
                outcome.terminal_after_action,
                outcome.action_clock_position,
            ) == (
                decision_of[data["generation"]],
                data["call_id"],
                data["arrived_at"],
                data["status"],
                data["terminal_after_action"],
                round(landed * self.config.env_fps, 9),
            )
        # Every frame captured goes in or is dropped, as the counts say, but the one that froze
        # the stream and one in a push still out when the run finishes.
        answered = self.entries("FramePushed")
        went_in = [entry.data.observation_id for entry in answered if entry.data.error is None]
        dropped = self.data("FrameDropped")
        assert (self.policy.frames_pushed, self.policy.dropped_frames) == (
            len(went_in),
            len(dropped),
        )
        frames = {
            entry.data.observation_id
            for entry in self.entries("FrameCaptured")
            if entry.data.error is None
        }
        still_out = {observation for observation, _ in self.pushes} - {
            entry.data.observation_id for entry in answered
        }
        invalid = self.termination.invalid
        if invalid is None or invalid.reason != "observation_stream_frozen":
            gone = set(went_in) | {data["observation_id"] for data in dropped} | still_out
            assert frames <= gone, f"frames {sorted(frames - gone)} neither went in nor dropped"
        # A hold's keys are released its duration after it was pressed.
        for data in self.data("KeyReleaseScheduled"):
            action = outcomes[data["index"]]
            duration = float(action["action"].parameters.get("duration", 1.0))
            assert data["at"] == action["started_at"] + duration
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
        # A reply reported as having come to no action, and to no protocol failure, handed none
        # over, and whatever a reply handed over after it was given up was refused.
        generation_of = {
            data["decision"]: data["generation"] for data in self.data("InferenceStarted")
        }
        handed = {entry.data.generation for entry in arrivals}
        for outcome in self.outcomes:
            if outcome.status == "none" and outcome.protocol_failure is None:
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
        # arrival's record, a stale result, an invalidation that comes first, and the outcome of
        # an action that finished before the deadline.
        deadline = self.config.duration_s
        for entry in records[: end - 1]:
            if entry.semantic < deadline or not isinstance(entry.data, dict):
                continue
            ran = (
                entry.data.get("finished_at") is not None and entry.data["finished_at"] <= deadline
            )
            if entry.kind in OUTCOME_RECORDS and ran:
                continue
            assert entry.kind in LATE_RECORDS, entry
