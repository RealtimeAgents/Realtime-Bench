"""The replay side: a script played by a real agent on the replay stubs, under a driver
(drivers.py), projected for comparison.

Wall clock: the chat_completions agent against ChatCompletionsStub and a FakeEnvironment
(tests/replay/fake_env.py), all on one fake clock per episode (tests/fake_clock.py), so every
reply, operation and wait takes exactly the script's time. Token clock and paused: the openrouter
agent against OpenRouterStub and GuestEnvironment itself on a task clock
(tests/replay/token_env.py); operational time is real, which the token clock's world time does not
read. The agent parses each reply with its own parser; nothing here knows the loop's rules.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest import mock

import realtime_framework.providers.registry as catalog
from realtime_framework.clock import Clocks
from realtime_framework.entity.actions import ActionType
from realtime_framework.entity.records import TraceEvent, _jsonable
from realtime_framework.sandbox.actions import ActionRejected, GuestEnvironment
from tests.fake_clock import FakeClock
from tests.replay.baseline import action_record
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
from .projection import Projection, decision, key_releases
from .script import Script, action_key, reply_text
from .simulated import turn_config

# The fake wall clock starts here: a whole binary number of seconds, so that a time since the
# origin is exact to ~1e-13 s. At the replay baselines' CLOCK_ORIGIN (1.76e9 s) one ulp is
# 2.4e-7 s, which would swamp the comparison's tolerance.
ORIGIN = 1024.0
PROMPT_TOKENS = 1500
STATE_READ_FAILED = "state read failed"


def _usage(output_tokens: int | None) -> dict | None:
    if output_tokens is None:
        return None
    return {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": output_tokens,
        "total_tokens": PROMPT_TOKENS + output_tokens,
    }


def _replies(script: Script) -> list:
    """One stub entry per request, in order: a turn's failed attempts, then its reply."""
    out = []
    for turn in script.turns:
        for attempt in turn.attempts:
            if attempt.status is not None:
                out.append(Failure(attempt.status, {"error": {"message": "scripted failure"}}))
            else:
                out.append(Reply(reply_text(turn.action), _usage(turn.output_tokens)))
    return out


class _Trace:
    """Stands in for EpisodeTrace: keeps each event's payload as EpisodeTrace serializes it, and
    the world time it was written at."""

    def __init__(self, env, now):
        self._env, self._now = env, now

    def emit(self, event, payload=None):
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
    the runner (evaluation/runner.py:200-211)."""

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
    ends (sandbox/actions.py:667-775) and the wait record (actions.py:863-903).

    A capture reads its tick when it starts and is stamped when it returns, an end check reads
    the task when it is asked and returns after its time, and an action takes its time before it
    takes effect, so a hold's lease starts after it (actions.py:746)."""

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
        return observation

    def done(self) -> bool:
        asked = self.now()
        cause, _ = self._over()
        over = super().done()
        if cause not in ("invalid_run", "deadline"):  # the task's state was read
            self.clock.sleep(self.script.check_s)
        self.log("diff.check", at=asked, answer=cause)
        return over

    def execute(self, action):
        self.log(
            "diff.execute",
            at=self.now(),
            action=action_key(action.action_type.value, action.parameters),
        )
        self.clock.sleep(self.script.execute_s)
        refused = self.decision in self.rejections or self.decision in self.failures
        if action.action_type is ActionType.HOLD_KEY and not refused:
            self._hold(action)
        status = super().execute(action)
        if action.action_type is ActionType.WAIT:
            self._last_execution_result.update(
                {
                    "truncated_by_episode_horizon": False,
                    "truncated_by_terminal_condition": False,
                    "active_holds": self._active_holds(),
                    "active_mouse_buttons": [],
                }
            )
        return status

    def _hold(self, action) -> None:
        duration = float(action.parameters.get("duration", 1.0))
        if duration <= 0:
            return
        parts, _ = GuestEnvironment._key_chord(action)
        with self._lock:
            now = self.clock.time()
            for identity, _ in parts:
                self._leases[identity] = max(self._leases.get(identity, now), now + duration)
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


class TokenWorld(_Latched, TokenEnvironment):
    """TokenEnvironment with the script's end, failed state reads and advances, refusals and
    faults, and a log of every end check and key release with the tick it happened at. The step,
    the end check and the releases a tap makes inside execute (sandbox/actions.py:548) are logged
    apart and not counted: the kernel's world moves the tick of a tap without them."""

    def __init__(self, frames, task: _Task, script: Script):
        self.script = script
        self._in_tap = False
        self._checks = 0
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
        if not self._in_tap and self._checks in self.script.check_errors:
            raise RuntimeError(STATE_READ_FAILED)
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
        if not self._in_tap:
            self._checks += 1
        tick, before = self._logical_tick, self._invalid_run
        over = super().done()
        if not over:
            answer = None
        elif before is not None:
            answer = "invalid_run"
        else:
            answer = "task" if self._invalid_run is None else "failed"
        self.log("diff.check", tick=tick, in_tap=self._in_tap, answer=answer)
        return over

    def execute(self, action):
        self._in_tap = action.action_type is ActionType.KEY
        try:
            status = super().execute(action)
        finally:
            self._in_tap = False
        self._applied += status == "applied"
        return status

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
        env = _replay_wall(script, run_dir, driver, config)
    else:
        env = _replay_token(script, run_dir, driver, config)
    return project(script, env)


def _replay_wall(script, run_dir, driver, config) -> WallEnvironment:
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
        )
        env = WallEnvironment(_frames(episode.frames, script), clock, script)
        box.append(env)
        _drive(episode, env, script, driver, config, Clocks(clock, clock), env.now)
    return env


def _replay_token(script, run_dir, driver, config) -> TokenWorld:
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
        )
        env = TokenWorld(_frames(episode.frames, script), task, script)
        _drive(
            episode, env, script, driver, config, Clocks(env.clocks.semantic, env.clock), env.now
        )
    return env


def _drive(episode, env, script, driver, config, clocks, now) -> None:
    """Run the episode under ``driver``, recording what the agent is told and when it is
    aborted; then drain the abandoned workers and close as main() does (evaluation/runner.py:
    1932-1938)."""
    agent = episode.agent
    commit_action, abort = agent.commit_action, agent.abort

    def recording_commit(status, detail=""):
        env.log("agent.commit_action", status=status, detail=detail, at=now())
        if script.commit_s:
            clocks.operational.sleep(script.commit_s)
        return commit_action(status, detail)

    def recording_abort(*args, **kwargs):
        env.log("agent.abort", at=now())
        return abort(*args, **kwargs)

    agent.commit_action = recording_commit
    agent.abort = recording_abort
    try:
        driver(
            agent,
            env,
            config,
            clocks=clocks,
            log=lambda line: env.log("runner.log", line=line),
            trace=_Trace(env, now),
        )
    finally:
        env._drain_abandoned()
        episode.finish()


class _FrozenFrames:
    """The frame source, repeating one frame's identity (observation id, tick, screenshot) from
    capture ``frozen_from`` on, each stamped when it is captured: a producer replaying one frame
    (evaluation/runner.py:74-109)."""

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


def project(script: Script, env) -> Projection:
    """The environment's log, in order: the runner's trace events and calls into the
    environment, what the agent was told, and what the subclasses above log (``diff.*``)."""
    token = script.mode == "token"
    seconds = 1.0 / script.env_fps
    decisions: dict[int, dict] = {}

    def of(k: int) -> dict:
        return decisions.setdefault(k, decision())

    captures, checks, check_answers, key_ups = [], [], [], []
    started_at, finished_at, resolved_at, committed_at, timeout_reasons = {}, {}, {}, {}, {}
    taps, records, aborts = [], 0, 0
    commits, snapshot_after, aborted_at, output_tokens = 0, None, [], {}
    first_end, ended_before_commit = None, {}
    for call in env.calls:
        name = call["call"]
        if (
            first_end is None
            and not call["finalizing"]
            and (name == "diff.invalid" or (name == "diff.check" and call["answer"]))
        ):
            first_end = call
        if name == "trace":
            event, payload, at = call["event"], call["payload"], call["at"]
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
                    d["wait_record"] = payload["execution_result"]
            elif event == "ActionApplied":
                of(payload["decision"])["applied"] = True
            elif event == "RunInvalidated":
                records += 1
        elif name == "agent.commit_action":
            # The environment numbers its calls by capture, and each decision captures once.
            k = call["decision"]
            of(k)["commit"] = {"status": call["status"], "detail": call["detail"]}
            committed_at[k] = call["at"]
            ended_before_commit[k] = first_end is not None
            commits += 1
        elif name == "env.record_lifecycle":
            record = call["record"]
            d = of(record["decision"])
            d["outcome"] = {field: record[field] for field in OUTCOME_FIELDS} | {
                "frame_age_s": record["frame_age_at_action_s"]
            }
            timeout_reasons[record["decision"]] = record["harness_timeout_reason"]
        elif name == "task.evaluation_snapshot" and snapshot_after is None:
            snapshot_after = commits
        elif name == "diff.execute":
            of(call["decision"])["executed"].append({"at": call["at"], "action": call["action"]})
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
        elif name == "diff.check" and token and not call["in_tap"]:
            checks.append(call["tick"])
            check_answers.append(call["answer"])
        elif name == "diff.key_up" and not call["finalizing"]:
            if token:
                key_ups.append(
                    {"at": call["tick"] * seconds, "keys": call["keys"], "in_tap": call["in_tap"]}
                )
            else:
                key_ups.append({"at": call["at"], "keys": call["keys"]})
        elif name == "agent.abort" and not call["finalizing"]:
            aborts += 1
            aborted_at.append(call["at"])
    invalid = getattr(env, "_invalid_run", None)
    termination = env._termination
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
    }
    facts = {
        "started_at": started_at,
        "finished_at": finished_at,
        "resolved_at": resolved_at,
        "committed_at": committed_at,
        "timeout_reasons": timeout_reasons,
        "check_answers": check_answers,
        "taps": taps,
        "key_ups": key_ups,
        "aborted_at": aborted_at,
        "output_tokens": output_tokens,
        # The first end signal before the snapshot: what answered (an end check's answer, or
        # invalid_run for a failure latched otherwise) and when.
        "ended_by": ended_by,
        "ended_at": ended_at,
        "ended_before_commit": ended_before_commit,
        "loop_error": getattr(env, "_loop_error", None),
    }
    return Projection("legacy", {"decisions": decisions, "episode": episode}, facts)
