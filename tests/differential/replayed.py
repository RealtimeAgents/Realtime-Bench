"""The replay side: a script played by a real agent on the replay stubs, under a driver
(drivers.py), projected for comparison.

Wall clock: the chat_completions agent against ChatCompletionsStub and a FakeEnvironment
(tests/replay/fake_env.py), all on one fake clock per episode (tests/fake_clock.py), so every
reply, operation and wait takes exactly the script's time. Token clock and paused: the openrouter
agent against OpenRouterStub and GuestEnvironment itself on a task clock
(tests/replay/token_env.py), whose operational clock reads the world's seconds. The agent parses
each reply with its own parser; nothing here knows the loop's rules.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from unittest import mock

import realtime_framework.providers.registry as catalog
from realtime_framework.clock import Clocks
from realtime_framework.entity.actions import ActionType
from realtime_framework.entity.records import TraceEvent, _jsonable
from realtime_framework.evaluation import episode_records
from realtime_framework.evaluation.clock import TokenAccounting
from realtime_framework.evaluation.episode_end import EpisodeResult
from realtime_framework.sandbox.actions import TAP_HOLD_S, ActionRejected, GuestEnvironment
from tests.fake_clock import FakeClock
from tests.replay.baseline import REAL_TIME_FIELDS, Normalizer, action_record
from tests.replay.build import build_episode
from tests.replay.clock_task import ClockTask
from tests.replay.fake_env import FakeEnvironment, Hang
from tests.replay.frames import ReplayTask
from tests.replay.scenario import TASK_INSTRUCTION
from tests.replay.stub import ChatCompletionsStub, Failure, Reply
from tests.replay.stub_openrouter import OpenRouterStub
from tests.replay.test_chat_completions import MODEL_KEY as CHAT_MODEL_KEY
from tests.replay.test_chat_completions import chat_completions_config
from tests.replay.test_openrouter import DUMMY_KEY, ROUTE
from tests.replay.test_openrouter import MODEL_KEY as OPENROUTER_MODEL_KEY
from tests.replay.token_env import TokenEnvironment

from . import drivers
from .legacy_environment import LegacyStateReads
from .legacy_turn import LegacyTurn
from .projection import (
    ONCE,
    Once,
    Projection,
    decision,
    earlier,
    ends_the_run,
    key_releases,
    wait_record,
)
from .script import Script, action_key, reply_text
from .simulated import STATE_READ_FAILED, turn_config

# The fake wall clock starts here: a whole binary number of seconds, so that a time since the
# origin is exact to ~1e-13 s. At the replay baselines' CLOCK_ORIGIN (1.76e9 s) one ulp is
# 2.4e-7 s, which would swamp the comparison's tolerance.
ORIGIN = 1024.0
PROMPT_TOKENS = 1500


def _usage(output_tokens: int | None) -> dict | None:
    if output_tokens is None:
        return None
    return {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": output_tokens,
        "total_tokens": PROMPT_TOKENS + output_tokens,
    }


def _replies(script: Script) -> list:
    """One stub entry per request, in order: a turn's failed attempts, then its reply. A held
    request uses none (stub.py ChatCompletionsStub._next)."""
    out = []
    for turn in script.turns:
        for attempt in turn.attempts:
            if attempt.status is not None:
                out.append(Failure(attempt.status, {"error": {"message": "scripted failure"}}))
            elif not attempt.hang:
                out.append(Reply(reply_text(turn.action), _usage(turn.output_tokens)))
    return out


class _Trace:
    """Stands in for EpisodeTrace: keeps each event's payload as EpisodeTrace serializes it, and
    the world time it was written at. What the kernel's shell adds to each record (kernel_seq,
    semantic, ops) is dropped (wave2-plan.md, P4 投影)."""

    def __init__(self, env, now):
        self._env, self._now = env, now

    def emit(self, event, payload=None, **_):
        name = event.value if isinstance(event, TraceEvent) else str(event)
        self._env.log("trace", event=name, payload=_jsonable(payload or {}), at=self._now())

    def checkpoint(self):
        pass


def _ended(script: Script, reached: bool, applied: int):
    """The task's answer once its end time is ``reached`` or ``applied`` actions took effect:
    its reason, True for a task that names none, or None while it goes on."""
    if reached or (script.ends_after is not None and applied >= script.ends_after):
        return script.end_reason or True
    return None


class _Latched:
    """Logs when the run's invalid-run value is first latched, by the environment itself or by
    today's loop (legacy_loop.py _invalidate)."""

    @property
    def _invalid_run(self):
        return self.__dict__.get("_latched_invalid_run")

    @_invalid_run.setter
    def _invalid_run(self, value):
        if value and self.__dict__.get("_latched_invalid_run") is None:
            self.log("diff.invalid", reason=value["reason"], at=self.now())
        self.__dict__["_latched_invalid_run"] = value


class WallEnvironment(_Latched, FakeEnvironment):
    """FakeEnvironment with the script's operation times, end and frozen stream, and the parts
    of GuestEnvironment's wall clock the kernel sees: holds released by a timer when their lease
    ends (sandbox/actions.py GuestEnvironment._execute_wall_hold, _schedule_wall_release,
    _wall_release_due) and the wait record (legacy_environment.py LegacyWallClock.execute).

    A capture reads its tick when it starts and is stamped when it returns, an end check reads
    the task when it is asked and returns after its time, and an action takes its time before it
    takes effect, so a hold's lease starts after it
    (actions.py GuestEnvironment._execute_wall_hold). A wait, DONE and FAIL go to no guest and
    take none (actions.py GuestEnvironment.execute)."""

    def __init__(self, frames, clock, script: Script):
        super().__init__(
            frames,
            clock,
            fps=script.env_fps,
            max_minutes=script.duration_s / 60,
            reply_latencies=[
                Hang(0.0) if attempt.hang else attempt.latency_s
                for turn in script.turns
                for attempt in turn.attempts
            ],
            rejections={k: t.rejected for k, t in enumerate(script.turns) if t.rejected},
            failures={k: t.fault for k, t in enumerate(script.turns) if t.fault},
            stop_check=lambda env: _ended(
                script,
                script.ends_at_s is not None and env.clock.time() - env.t0 >= script.ends_at_s,
                sum(execution["status"] == "applied" for execution in env.executions),
            ),
            snapshot_seconds=script.snapshot_s,
        )
        self.deadline = self.t0 + script.duration_s
        self.script = script
        self.kernel_counts_reads = False  # drivers.shell sets it
        self._after_action = False  # the next read follows an executed action
        self._landing = False  # the next read is the one before landing: a frame was captured
        self._last_asked: float | None = None  # when the last state read was asked
        self._lock = threading.RLock()
        self._leases: dict[str, float] = {}
        self._release_timer = None
        self._closed = False

    def now(self) -> float:
        return self.clock.time() - self.t0

    def capture(self):
        self.decision += 1
        tick = self.tick_index()
        self.clock.sleep(self.script.capture_s)
        observation = self.frames.capture(tick, at=self.clock.time())
        self.log("env.capture", tick=tick, at=self.now())
        self._landing = True
        return observation

    def _kernels(self, asked: float) -> bool:
        """Whether the kernel makes this read too (decision 31, N26; turns.py
        TurnPolicy._check_end): every one under its shell; of today's, the one after an executed
        action, the one before landing on a reading not read at already, and the first at a
        frame's start, where the kernel's periodic read falls, the episode's start too. Today's
        others, at the loop top within a frame, fail nowhere."""
        landing, self._landing = self._landing, False
        first = self._last_asked is None or earlier(self._last_asked, asked)
        if self.kernel_counts_reads or self._after_action or (landing and first):
            return True
        ticks = asked * self.fps
        return abs(ticks - round(ticks)) <= 1e-9 and first

    def done(self) -> bool:
        asked = self.now()
        cause, _ = self._over()
        reads = cause not in ("invalid_run", "deadline")  # the task's state is read
        kernels = self._kernels(asked)
        self._last_asked = asked
        if reads and kernels and self.script.read_fails_at(asked):
            # Today's environment latches a read that failed, once it has taken its time, and
            # answers that the episode is over (legacy_environment.py LegacyStateReads.done).
            self.clock.sleep(self.script.check_s)
            self._invalid_run = {
                "category": "environment",
                "reason": "terminal_state_check_failed",
                "detail": STATE_READ_FAILED,
            }
            cause, over = "failed", True
        else:
            over = super().done()
            if reads:
                self.clock.sleep(self.script.check_s)
        # The read after an executed action is one the kernel makes too; it reads each frame
        # by its own timer instead of today's others (turns.py TurnPolicy._check_end).
        self.log("diff.check", at=asked, answer=cause, kernels=kernels)
        self._after_action = False
        return over

    def check_end(self) -> tuple[bool, str | None]:
        """The kernel's shell reads the task here; a failed read raises (sandbox/actions.py
        GuestEnvironment.check_end)."""
        asked = self.now()
        failed = self.script.read_fails_at(asked)
        if not failed:
            ended, reason = super().check_end()
        self.clock.sleep(self.script.check_s)
        self.log("diff.check", at=asked, answer="failed" if failed else "task" if ended else None)
        if failed:
            raise RuntimeError(STATE_READ_FAILED)
        return ended, reason

    def execute(self, action, *, release_at=None):
        self.log(
            "diff.execute",
            at=self.now(),
            action=action_key(action.action_type.value, action.parameters),
        )
        if action.action_type not in (ActionType.WAIT, ActionType.DONE, ActionType.FAIL):
            self.clock.sleep(self.script.execute_s)
        refused = self.decision in self.rejections or self.decision in self.failures
        if action.action_type is ActionType.HOLD_KEY and not refused:
            self._hold(action, release_at)
        execution = super().execute(action, release_at=release_at)
        self._after_action = execution.status in ("applied", "wait")
        if action.action_type is ActionType.WAIT:
            execution.result.update(
                {
                    "truncated_by_episode_horizon": False,
                    "truncated_by_terminal_condition": False,
                    "active_holds": self._active_holds(),
                    "active_mouse_buttons": [],
                }
            )
        return execution

    def model_request(self, index: int) -> bool:
        self.log("diff.request", request=index, at=self.now())
        return super().model_request(index)

    def _hold(self, action, release_at: float | None) -> None:
        """The lease lasts the hold's duration from the press, or to the kernel's ``release_at``,
        seconds since the origin, but at least as long as a tap (decision 38;
        sandbox/actions.py GuestEnvironment._execute_wall_hold)."""
        duration = float(action.parameters.get("duration", 1.0))
        if duration <= 0:
            return
        parts, _ = GuestEnvironment._key_chord(action)
        with self._lock:
            now = self.clock.time()
            expires_at = (
                now + duration
                if release_at is None
                else max(self.t0 + release_at, now + min(duration, TAP_HOLD_S))
            )
            for identity, _ in parts:
                self._leases[identity] = max(self._leases.get(identity, now), expires_at)
            self._schedule_release()

    def _schedule_release(self) -> None:
        if self._release_timer is not None:
            self._release_timer.cancel()
            self._release_timer = None
        if self._leases and not self._closed:
            earliest = min(self._leases.values())
            self._release_timer = self.clock.timer(
                max(0.0, earliest - self.clock.time()), self._release_due
            )

    def _release_due(self) -> None:
        with self._lock:
            self._release_timer = None
            if self._closed:
                return
            now = self.clock.time()
            expired = sorted(key for key, at in self._leases.items() if at <= now + 1e-3)
            for key in expired:
                del self._leases[key]
            if expired:
                self.log("diff.key_up", at=self.now(), keys=expired)
            self._schedule_release()

    def _active_holds(self) -> list[dict]:
        now = self.clock.time()
        return [
            {"key": key, "remaining_seconds": round(max(0.0, at - now), 3)}
            for key, at in self._leases.items()
        ]

    def release_inputs(self) -> None:
        with self._lock:
            self._closed = True
            if self._release_timer is not None:
                self._release_timer.cancel()
        super().release_inputs()

    def _evaluation_snapshot(self) -> dict:
        snapshot = super()._evaluation_snapshot()
        if self.script.snapshot_error is not None:
            raise RuntimeError(self.script.snapshot_error)
        return snapshot


class _Task(ClockTask):
    """ClockTask whose advance fails when the environment says so, and whose snapshot read fails
    with ``snapshot_error``."""

    def __init__(self, fps, tokens_per_frame, snapshot_error: str | None):
        self.failure = lambda: None
        self.snapshot_error = snapshot_error
        super().__init__(fps, tokens_per_frame)

    def advance_clock(self, frames: float = 1, key_events=()):
        message = self.failure()
        if message is not None:
            raise RuntimeError(message)
        return super().advance_clock(frames, key_events)

    def evaluation_snapshot(self) -> dict:
        snapshot = super().evaluation_snapshot()
        if self.snapshot_error is not None:
            raise RuntimeError(self.snapshot_error)
        return snapshot


class _Guest:
    """The guest's action endpoint: refuses or fails the action of a turn that says so."""

    def __init__(self, env: TokenWorld):
        self._env = env

    def execute_action(self, action):
        self._env.log("guest.execute_action", action=action_record(action))
        turn = self._env.turn()
        if turn.rejected is not None:
            raise ActionRejected(turn.rejected)
        if turn.fault is not None:
            raise RuntimeError(turn.fault)
        return {"status": "ok"}


class TokenWorld(_Latched, TokenEnvironment, LegacyStateReads, GuestEnvironment):
    """TokenEnvironment with the script's end, failed state reads and advances, refusals and
    faults, and a log of every end check and key release with the tick it happened at. The step,
    the end check and the releases a tap makes inside execute (sandbox/actions.py
    GuestEnvironment._execute_clock_key) are logged apart and not counted: the kernel's world moves
    the tick of a tap without them. LegacyStateReads comes after TokenEnvironment, whose log of the
    runner's calls takes in the legacy loop's WAIT too."""

    def __init__(self, frames, task: _Task, script: Script):
        self.script = script
        self._in_tap = False
        self._checks = 0
        self._tap_reads = 0
        self._saw_end = False
        self._tap_saw_end = False  # a read inside the last tap saw the end
        self._last_read: float | None = None  # the tick of the kernel's last read
        self._after_action = False  # the next read follows an executed action
        self._last_asked: float | None = None  # when the last state read was asked
        self._extra: str | None = None  # why the read in hand is not one the kernel makes
        self._frame_failed = False  # the kernel's last read failed, or stands for a failure
        self._advances = 0
        self._applied = 0
        super().__init__(frames, task, max_minutes=script.duration_s / 60, stop_check=self._stop)
        self.g = _Guest(self)
        task.failure = self._advance_failure

    def turn(self):
        return self.script.turns[self.decision]

    def now(self) -> float:
        return self._logical_tick / self.fps

    def _stop(self):
        script = self.script
        if self._in_tap:
            read, failing, blind = self._tap_reads, script.tap_check_errors, script.blind_tap_checks
        else:
            read, failing, blind = self._checks, script.check_errors, script.blind_checks
        if self._extra is None:
            self._frame_failed = read in failing or read in blind
        if self._extra is None and read in failing:
            raise RuntimeError(STATE_READ_FAILED)
        if (
            self._extra is None and read in blind or self._extra == "frame" and self._frame_failed
        ) and not self._saw_end:
            # A blind read stands for a failure the kernel takes as "not ended"; so does today's
            # read again in a frame whose read failed, since the kernel takes the frame as read
            # (turns.py TurnPolicy._check_end). Once a read has seen the end, the kernel reads no
            # more, and today's loop reads the task as it is.
            return None
        ends_at = self.script.ends_at_s
        reached = ends_at is not None and self._logical_tick >= ends_at * self.fps
        return _ended(self.script, reached, self._applied)

    def _advance_failure(self) -> str | None:
        if self._in_tap:
            return None
        self._advances += 1
        return (
            "page clock did not advance" if self._advances in self.script.advance_errors else None
        )

    def capture(self):
        tick = self._logical_tick
        observation = super().capture()
        self.log("diff.capture", tick=tick, timestamp=observation.timestamp)
        return observation

    def done(self) -> bool:
        tick, before = self._count_read(), self._invalid_run
        over = super().done()
        if not over:
            answer = None
        elif before is not None:
            answer = "invalid_run"
        else:
            answer = "task" if self._invalid_run is None else "failed"
        self._answered(tick, answer)
        return over

    def check_end(self) -> tuple[bool, str | None]:
        """The kernel's shell reads the task here, and so do a tap's steps under it
        (sandbox/actions.py GuestEnvironment._ended_after_step); a failed read raises."""
        tick = self._count_read()
        try:
            ended, reason = super().check_end()
        except Exception:
            self._answered(tick, "failed")
            raise
        self._answered(tick, "task" if ended else None)
        return ended, reason

    def _count_read(self) -> float:
        """Number the read as the kernel numbers its own (turns.py TurnPolicy._check_end): the
        reads inside a tap, at most one more a frame, a frame being the clock's position, and the
        one after every executed action, but none once the episode is over (TurnPolicy._executed
        ends it on a tap's read that saw the end). Under the kernel's shell every read is the
        kernel's. Today's loop reads where the kernel does not (KF5, KF4); such a read goes
        unnumbered, fails nowhere and reads the task as it is."""
        tick = self._logical_tick
        extra = None
        if self._in_tap:
            self._tap_reads += 1
        elif not self.kernel_counts_reads:
            if self._invalid_run is not None or self._saw_end:
                extra = "tap_end" if self._tap_saw_end and self._after_action else "ended"
            elif not self._after_action and self._last_read is not None:
                extra = "frame" if abs(tick - self._last_read) <= 1e-9 else None
        if not self._in_tap:
            self._after_action = False
            if extra is None:
                self._checks += 1
        if extra is None:
            self._last_read = tick
        self._extra = extra
        return tick

    def _answered(self, tick: float, answer: str | None) -> None:
        self._saw_end = self._saw_end or answer == "task"
        if self._in_tap:
            self._tap_saw_end = self._tap_saw_end or answer == "task"
        read = None if self._extra else self._tap_reads if self._in_tap else self._checks
        self.log(
            "diff.check",
            tick=tick,
            in_tap=self._in_tap,
            read=read,
            answer=answer,
            extra=self._extra,
        )

    def execute(self, action, *, release_at=None):
        self._in_tap = action.action_type is ActionType.KEY
        self._tap_saw_end = False
        try:
            execution = super().execute(action, release_at=release_at)
        finally:
            self._in_tap = False
        self._applied += execution.status == "applied"
        self._after_action = execution.status in ("applied", "wait")
        return execution

    def release_keys(self, keys) -> None:
        """The kernel's release of keys whose lease ended. Inside a tap the environment releases
        them itself, where the lease ends; the kernel's release comes after the tap and finds them
        gone (K-j). The shell side's key releases are these, as the kernel decides them."""
        self.log("diff.release_keys", tick=self._logical_tick, keys=sorted(keys))
        super().release_keys(keys)

    def _release_clock_keys(self, identities=None):
        selected = list(self._held_keys) if identities is None else identities
        held = [key for key in selected if key in self._held_keys]
        tick = self._logical_tick
        super()._release_clock_keys(identities)
        if held:
            self.log("diff.key_up", tick=tick, keys=sorted(held), in_tap=self._in_tap)


def replay(script: Script, run_dir: Path, driver=drivers.legacy) -> Projection:
    config = turn_config(script)
    if script.clock == "wall":
        env, result = _replay_wall(script, run_dir, driver, config)
    else:
        env, result = _replay_token(script, run_dir, driver, config)
    return project(script, env, result, driver.__name__)


def _replay_wall(script, run_dir, driver, config) -> tuple[WallEnvironment, EpisodeResult]:
    clock = FakeClock(wall_origin=ORIGIN)
    box = []
    with (
        ChatCompletionsStub(
            _replies(script), on_request=lambda index: box[0].model_request(index), clock=clock
        ) as stub,
        clock.participant(),
    ):
        model = chat_completions_config(stub.base_url, stream=True).model_copy(
            update={"fps": script.input_fps}
        )
        episode = build_episode(
            model,
            run_dir,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=CHAT_MODEL_KEY,
            clock_mode="wall",
            clock=clock,
            env_fps=script.env_fps,
        )
        env = WallEnvironment(_frames(episode.frames, script), clock, script)
        box.append(env)
        result = _drive(episode, env, script, driver, config, Clocks(clock, clock), env.now)
        env.requests = _requests(stub)
    return env, result


def _replay_token(script, run_dir, driver, config) -> tuple[TokenWorld, EpisodeResult]:
    entry = catalog.get(OPENROUTER_MODEL_KEY)
    model = catalog.to_model_config(
        OPENROUTER_MODEL_KEY,
        fps=script.input_fps,
        reasoning_effort=entry.defaults["reasoning_effort"],
    )
    task = _Task(script.env_fps, script.tokens_per_frame, script.snapshot_error)
    variables = {"OPENROUTER_API_KEY": DUMMY_KEY}
    with OpenRouterStub(_replies(script), route=ROUTE) as stub, mock.patch.dict(os.environ):
        # What test_openrouter's openrouter_env fixture sets, and the stub's address.
        for name in ("OPENROUTER_REFERER", "OPENROUTER_TITLE"):
            os.environ.pop(name, None)
        os.environ.update(variables | {"OPENROUTER_BASE_URL": stub.base_url})
        episode = build_episode(
            model,
            run_dir,
            task=task,
            instruction=TASK_INSTRUCTION,
            model_key=OPENROUTER_MODEL_KEY,
            clock_mode="token",
            tokens_per_frame=script.tokens_per_frame,
            env_fps=script.env_fps,
        )
        env = TokenWorld(_frames(episode.frames, script), task, script)
        result = _drive(
            episode, env, script, driver, config, Clocks(env.clocks.semantic, env.clock), env.now
        )
        env.requests = _requests(stub)
    return env, result


def _requests(stub) -> list[dict]:
    """What the stub received, as the replay baselines normalize it (tests/replay/baseline.py):
    one placeholder numbering of random ids per episode, images by their pixels."""
    normalize = Normalizer()
    return [
        normalize({"path": request.path, "body": json.loads(request.body)})
        for request in stub.requests
    ]


def _drive(episode, env, script, driver, config, clocks, now) -> EpisodeResult:
    """Run the episode under ``driver``, today's loop on the agent as it calls it
    (legacy_turn.py), recording what the agent is told, when a generation returns and when one
    is aborted; then drain the abandoned workers and close as main() does (evaluation/runner.py
    main). How the episode came out, as the driver hands it back."""
    agent = LegacyTurn(episode.agent) if driver is drivers.legacy else episode.agent
    commit, abort, generate = agent.commit, agent.abort, agent.generate

    def recording_generate(generation, *args, **kwargs):
        pending = generate(generation, *args, **kwargs)
        env.log("diff.generated", generation=generation, at=now())
        return pending

    def recording_commit(status, detail=""):
        env.log("agent.commit_action", status=status, detail=detail, at=now())
        if script.commit_s:
            clocks.operational.sleep(script.commit_s)
        return commit_action(status, detail)

    def recording_outcome(outcome):
        # The kernel's shell reports through commit(outcome) (evaluation/shell.py _Shell._commit).
        env.log("agent.commit_action", status=outcome.status, detail=outcome.detail, at=now())
        if script.commit_s:
            clocks.operational.sleep(script.commit_s)
        return commit(outcome)

    def recording_abort(*args, **kwargs):
        env.log("agent.abort", generation=args[0] if args else kwargs.get("generation"), at=now())
        return abort(*args, **kwargs)

    agent.generate = recording_generate
    agent.commit = recording_outcome
    agent.abort = recording_abort
    if isinstance(agent, LegacyTurn):
        commit_action = agent.commit_action
        agent.commit_action = recording_commit
    result = None
    try:
        result = driver(
            agent,
            env,
            config,
            clocks=clocks,
            log=lambda line: env.log("runner.log", line=line),
            trace=_Trace(env, now),
        )
        return result
    finally:
        if result is not None:
            result.drain_abandoned()
        env.usage = episode_records.usage_snapshot(agent)
        env.subtotal = _subtotal(() if result is None else result.settlements)
        episode.finish()


def _subtotal(settlements) -> dict | None:
    """The clock counts of the reported decisions' responses, which the report checks the
    lifecycle against (reporting.py report): the token accounting's own subtotal once a response
    goes unreported, else its totals (clock.py TokenAccounting.summary; N50)."""
    if not settlements:
        return None
    summary = TokenAccounting(settlements).summary()
    reported = summary.get("reported_decisions", summary)
    counts = reported["counts"]["clock_output_tokens"]
    return {
        "responses": reported["responses"],
        "clock_output_tokens": {
            name: counts[name]
            for name in ("total", "known_total", "responses_with_count", "complete")
        },
    }


class _FrozenFrames:
    """The frame source, repeating one frame's identity (observation id, tick, screenshot) from
    capture ``frozen_from`` on, each stamped when it is captured: a producer replaying one frame
    (legacy_loop.py ObservationStreamWatchdog)."""

    def __init__(self, frames, frozen_from: int):
        self._frames = frames
        self._frozen_from = frozen_from
        self._captures = 0
        self._frozen = None

    def __getattr__(self, name):
        return getattr(self._frames, name)

    def capture(self, tick, at=None):
        index, self._captures = self._captures, self._captures + 1
        observation = self._frames.capture(tick, at)
        if index < self._frozen_from:
            return observation
        if self._frozen is None:
            self._frozen = observation
            return observation
        frozen = self._frozen
        observation.tick_number = frozen.tick_number
        observation.screenshot = frozen.screenshot
        observation.metadata["observation_id"] = frozen.metadata["observation_id"]
        observation.metadata["screenshot_sha256"] = frozen.metadata["screenshot_sha256"]
        return observation


def _frames(frames, script: Script):
    return frames if script.frozen_from is None else _FrozenFrames(frames, script.frozen_from)


# ---- projection -------------------------------------------------------------------------------

# Trace records outside the projection's decisions (wave2-plan.md, P4 投影).
TRACED = ("ObservationCaptured", "InferenceStarted", "OutputChunkReceived", "InferenceCompleted")
# A capture's latency is measured on the operational clock, real time on the token clock
# (sandbox/actions.py GuestEnvironment.capture).
TRACE_REAL_TIME = REAL_TIME_FIELDS | {"capture_latency_s"}

OUTCOME_FIELDS = (
    "status",
    "terminal_after_action",
    "action_clock_position",
    "next_observation_tick",
    "dropped_intermediate_frames",
    "post_action_rendered",
    "token_delay_frames",
    "action_output_tokens",
    "post_horizon_output_tokens",
)


def _unaborted(abandoned, aborted: set, generated: dict) -> list[int]:
    """The generations the kernel gave up on that the shell left running: neither aborted nor
    returned by the reading the kernel gave them up at."""
    return [
        generation
        for generation, at in abandoned
        if generation not in aborted
        and (generation not in generated or earlier(at, generated[generation]))
    ]


def project(script: Script, env, episode_result: EpisodeResult, side: str) -> Projection:
    """The environment's log, in order: the runner's trace events and calls into the
    environment, what the agent was told, and what the subclasses above log (``diff.*``); and
    how the episode came out, as the driver handed it back."""
    token = script.mode == "token"
    seconds = 1.0 / script.env_fps
    decisions: dict[int, dict] = {}
    normalize = Normalizer(TRACE_REAL_TIME)

    def of(k: int) -> dict:
        return decisions.setdefault(k, decision())

    captures, checks, key_ups = [], [], []
    started_at, finished_at, resolved_at, committed_at, timeout_reasons = {}, {}, {}, {}, {}
    executed_at = {}
    taps, reads, records, aborts, requested, extras = [], [], 0, 0, [], []
    kernel_reads, once = [], Once()
    abandoned, aborted, generated = [], set(), {}
    commits, snapshot_after, aborted_at, output_tokens = 0, None, [], {}
    first_end, ended_before_commit = None, {}
    trace = {kind: {} for kind in TRACED}
    key_events, results = [], {}
    for call in env.calls:
        name = call["call"]
        ends = name == "diff.invalid" or (name == "diff.check" and ends_the_run(call["answer"]))
        if first_end is None and not call["finalizing"] and ends:
            first_end = call
        if name == "trace":
            event, payload, at = call["event"], call["payload"], call["at"]
            if event in TRACED:
                trace[event].setdefault(payload.get("decision"), []).append(normalize(payload))
            if event in ONCE:
                once(event, payload["decision"], payload.get("phase"))
            if event == "InferenceStarted":
                of(payload["decision"])["observed"] = payload["observation_tick"]
                started_at[payload["decision"]] = at
            elif event == "InferenceCompleted":
                of(payload["decision"])["remaining"] = payload["remaining_clock_tokens"]
                finished_at[payload["decision"]] = at
                output_tokens[payload["decision"]] = payload["clock_output_tokens"]
            elif event == "ActionProposed":
                action = payload["action"]
                of(payload["decision"])["proposed"] = action_key(
                    action["action_type"], action["parameters"]
                )
            elif event == "ClockAdvanced":
                d = of(payload["decision"])
                moved = {
                    "from": payload["from_tick"],
                    "requested": payload["requested_tick"],
                    "to": payload["to_tick"],
                }
                if payload["phase"] == "inference_to_action":
                    d["approach"] = moved
                else:
                    d["boundary"] = moved | {
                        "rendered": payload["post_action_rendered"],
                        "dropped": payload["dropped_intermediate_frames"],
                    }
            elif event == "ActionResolved":
                resolved_at[payload["decision"]] = at
                d = of(payload["decision"])
                d["resolved"] = {
                    "status": payload["status"],
                    "terminal_after_action": payload["terminal_after_action"],
                    "action_clock_position": payload["action_clock_position"],
                    "environment_tick": payload["environment_tick"],
                }
                action = payload["action"]
                if action is not None and action["action_type"] == "wait":
                    d["wait_record"] = wait_record(payload["execution_result"])
                else:
                    # A wait's is its record above.
                    result = normalize(payload["execution_result"])
                    if isinstance(result, dict) and not token:
                        # FakeEnvironment numbers its executions; "executed" says which ran.
                        result.pop("execution", None)
                    results[payload["decision"]] = result
            elif event == "ActionApplied":
                of(payload["decision"])["applied"] = True
            elif event == "RunInvalidated":
                records += 1
            elif event == "KernelRecord" and payload["kind"] == "GenerationAbandoned":
                # The kernel's own count: the shell interrupts only a generation still running
                # (evaluation/shell.py _Shell._abort), which at one instant is the scheduler's
                # call; _unaborted checks it did.
                aborts += 1
                abandoned.append((payload["data"]["generation"], at))
        elif name == "agent.commit_action":
            # The environment numbers its calls by capture, and each decision captures once.
            k = call["decision"]
            once("commit", k)
            of(k)["commit"] = {"status": call["status"], "detail": call["detail"]}
            committed_at[k] = call["at"]
            ended_before_commit[k] = first_end is not None
            commits += 1
        elif name == "env.record_lifecycle":
            record = call["record"]
            once("lifecycle", record["decision"])
            d = of(record["decision"])
            d["outcome"] = {field: record[field] for field in OUTCOME_FIELDS} | {
                "frame_age_s": record["frame_age_at_action_s"]
            }
            timeout_reasons[record["decision"]] = record["harness_timeout_reason"]
        elif name.startswith("task.") and "key" in call and not call["finalizing"]:
            # What the task clock is told of the keys (tests/replay/clock_task.py).
            key_events.append(
                {k: v for k, v in call.items() if k not in ("decision", "finalizing")}
            )
        elif name == "task.advance" and call["key_events"] and not call["finalizing"]:
            key_events.append(
                {"call": name, "at_s": call["from_s"], "key_events": call["key_events"]}
            )
        elif name == "task.evaluation_snapshot" and snapshot_after is None:
            snapshot_after = commits
        elif name == "diff.execute":
            of(call["decision"])["executed"].append({"at": call["at"], "action": call["action"]})
            executed_at.setdefault(call["decision"], call["at"])
        elif name == "env.execute" and token:
            action = call["action"]
            at = call["from_tick"] * seconds
            of(call["decision"])["executed"].append(
                {"at": at, "action": action_key(action["action_type"], action["parameters"])}
            )
            if action["action_type"] == "key":
                taps.append((call["from_tick"], call["to_tick"]))
        elif name == "diff.capture" or (name == "env.capture" and not token):
            captures.append(
                {"tick": call["tick"], "timestamp": call["timestamp"] if token else call["at"]}
            )
        elif name == "diff.check" and token:
            if call["extra"] is None:
                reads.append(("tap" if call["in_tap"] else "check", call["read"], call["answer"]))
            if not call["in_tap"]:
                checks.append(call["tick"])
                extras.append(call["extra"])
        elif name == "diff.check" and call["answer"] not in ("invalid_run", "deadline"):
            reads.append((call["at"], call["answer"], call.get("kernels", True)))
            if side == "shell" and not call["finalizing"]:
                kernel_reads.append({"at": call["at"], "answer": call["answer"]})
        elif name == "diff.release_keys" and side == "shell":
            key_ups.append({"at": call["tick"] * seconds, "keys": call["keys"], "in_tap": False})
        elif name == "diff.key_up" and not call["finalizing"] and not (token and side == "shell"):
            if token:
                key_ups.append(
                    {"at": call["tick"] * seconds, "keys": call["keys"], "in_tap": call["in_tap"]}
                )
            else:
                key_ups.append({"at": call["at"], "keys": call["keys"]})
        elif name == "diff.request":
            requested.append((call["decision"], call["at"]))
        elif name == "agent.abort":
            aborts += side == "legacy"
            aborted_at.append(call["at"])
            aborted.add(call["generation"])
        elif name == "diff.generated":
            generated[call["generation"]] = call["at"]
    invalid, termination = episode_result.invalid_run, episode_result.termination_record
    if first_end is None:
        ended_by, ended_at = None, None
    else:
        ended_by = "invalid_run" if first_end["call"] == "diff.invalid" else first_end["answer"]
        ended_at = first_end["at"] if "at" in first_end else first_end["tick"] * seconds
    episode = {
        "captures": captures,
        "checks": checks if token else None,
        "key_ups": key_releases(key_ups, termination["end_s"]),
        "aborts": aborts,
        "invalid": None
        if invalid is None
        else {"category": invalid["category"], "reason": invalid["reason"]},
        "termination": {
            "stop_reason": termination["stop_reason"],
            "end_s": termination["end_s"],
            "snapshot_s": termination["snapshot_s"],
        },
        "snapshot_after": snapshot_after,
        "invalidation_records": records,
        "duplicates": once.again,
    }
    facts = {
        "started_at": started_at,
        "finished_at": finished_at,
        "resolved_at": resolved_at,
        # Wall clock: when the environment was handed each decision's action, a wait too.
        "executed_at": executed_at,
        "committed_at": committed_at,
        "timeout_reasons": timeout_reasons,
        # The state reads before the snapshot, in order, with their answer (None, "task" or
        # "failed"): on the token clock the kernel's reads, (kind, number, answer), kind "check"
        # or "tap" numbered as Script.check_errors and Script.tap_check_errors; on the wall clock
        # today's, (seconds, answer, whether the kernel makes it too: WallEnvironment._kernels).
        "reads": reads,
        # Token clock: for each of today's end checks, why the kernel does not make it (None
        # when it does): "frame" read already, the episode "ended", or a tap's read saw the end,
        # "tap_end" (TokenWorld._count_read).
        "check_extras": extras,
        "taps": taps,
        "key_ups": key_ups,
        "aborted_at": aborted_at,
        "output_tokens": output_tokens,
        # The first end signal before the snapshot: what answered (an end check's answer, or
        # invalid_run for a failure latched otherwise) and when.
        "ended_by": ended_by,
        "ended_at": ended_at,
        "ended_before_commit": ended_before_commit,
        "loop_error": episode_result.loop_error,
        # Today's loop: whether the think budget ran out, which a failed snapshot read leaves out
        # of its stop reason (legacy_loop.py _finalize).
        "think_exhausted": getattr(env, "_think_exhausted", False),
        # Wall clock: the decision each request belongs to and when it reached the stub.
        "requested": requested,
    }
    # The trace records by decision, normalized as the replay baselines are.
    # "synthesized": what the records of a call the shell gave up on say, by decision
    # (rules._abandoned).
    # "key_events": token clock, what the task clock was told of the keys before the snapshot.
    # "results": each decision's execution_result but a wait's, as ActionResolved has it.
    # "subtotal": the token accounting's clock counts of the reported decisions (_subtotal).
    beyond = {
        "requests": env.requests,
        "usage": env.usage,
        "subtotal": env.subtotal,
        "trace": trace,
        "synthesized": {},
        "key_events": key_events if token else None,
        "results": results,
    }
    kernel = None
    if side == "shell":
        kernel = {"unaborted": _unaborted(abandoned, aborted, generated)}
        if not token:
            kernel["reads"] = kernel_reads
    return Projection(side, {"decisions": decisions, "episode": episode}, facts, beyond, kernel)
