"""A whole native-streaming episode: the event kernel and the streaming shell
(evaluation/streaming_shell.py) on the fake environment of tests/shell_streaming, driving the
native-streaming agent (harnesses/streaming/agent.py) on a fake live encoder, on the fake clock.

Every command the kernel gives the agent comes to one agent call of its kind, in the kernel's
order, and every push and reply to one encoder operation."""

from __future__ import annotations

import hashlib

import pytest

from realtime_framework.clock import Clocks
from realtime_framework.entity.actions import FrameGeometry
from realtime_framework.evaluation.kernel.events import (
    AbortGeneration,
    AcceptResult,
    Commit,
    DiscardResult,
    StartGeneration,
)
from realtime_framework.evaluation.kernel.streaming import PushFrame, StreamingConfig
from realtime_framework.evaluation.streaming_shell import _StreamingShell, run_streaming_episode
from realtime_framework.sandbox.observation.interactables import canonical_result
from realtime_framework.sandbox.observation.som import ToolRuntime
from tests.fake_clock import FakeClock
from tests.shell_streaming.fakes import FakeEnv

from .fake_live import Call, GoAway, Reply, make_agent, screenshot


class _Env(FakeEnv):
    """The shell tests' environment, its captures real screenshots the agent can prepare."""

    def capture(self):
        observation = super().capture()
        observation.screenshot = screenshot()
        return observation


def _command(command) -> tuple | None:
    """A command the agent answers, as the call it comes to; None for the others."""
    match command:
        case PushFrame(observation_id=observation_id):
            return ("push", observation_id)
        case StartGeneration(generation=generation):
            return ("generate", generation)
        case AcceptResult(generation=generation):
            return ("accept", generation)
        case DiscardResult(generation=generation):
            return ("discard", generation)
        case Commit(outcome=outcome):
            return ("commit", outcome.decision, outcome.call_id, outcome.status)
        case AbortGeneration(generation=generation):
            return ("abort", generation)
    return None


def _recorded(agent, calls: list) -> None:
    """Write each kernel-facing call of ``agent`` to ``calls`` as _command names it."""
    keys = {
        "push": lambda observation: (observation.metadata["observation_id"],),
        "generate": lambda generation, *rest: (generation,),
        "accept": lambda generation: (generation,),
        "discard": lambda generation: (generation,),
        "commit": lambda outcome: (outcome.decision, outcome.call_id, outcome.status),
        "abort": lambda generation: (generation,),
    }
    for name, key in keys.items():
        method = getattr(agent, name)

        def call(*args, _name=name, _key=key, _method=method):
            calls.append((_name, *_key(*args)))
            return _method(*args)

        setattr(agent, name, call)


def test_every_kernel_command_comes_to_its_agent_call_and_encoder_operation(monkeypatch):
    """Four replies: the first and third click frame 1 and end, and each click executes; the
    second and fourth never end, and the reply watchdog gives each up, which discards it."""
    clock = FakeClock()
    commands, calls = [], []
    run = _StreamingShell._run

    def recording(shell, command):
        commands.append(command)
        run(shell, command)

    monkeypatch.setattr(_StreamingShell, "_run", recording)
    click = Call(0.5, "c1", "click", {"x": 10, "y": 20, "button": "left", "frame": 1})
    with clock.participant():
        agent = make_agent(monkeypatch, clock)
        agent.driver.replies += [Reply((click,), ends_at=1.0), Reply(ends_at=99.0)] * 2
        _recorded(agent, calls)
        env = _Env(clock)
        run_streaming_episode(
            agent,
            env,
            StreamingConfig(duration_s=6.0, reply_watchdog_s=2.0),
            clocks=Clocks(clock, clock),
            log=lambda message: None,
        )
        agent.close()
    assert not env._invalid_run
    expected = [entry for entry in map(_command, commands) if entry is not None]
    for kind in ("push", "generate", "accept", "discard", "commit", "abort"):
        assert [c for c in calls if c[0] == kind] == [c for c in expected if c[0] == kind], kind

    def of(kind):
        return [c[1:] for c in calls if c[0] == kind]

    assert of("generate") == [(1,), (2,), (3,), (4,)]
    assert (of("accept"), of("abort"), of("discard")) == ([(1,), (3,)], [(2,), (4,)], [(2,), (4,)])
    assert of("commit") == [(0, "c1", "applied"), (2, "c1", "applied")]
    assert env.counts["execute"] == 2
    for operation, kind in (("push", "push"), ("reply", "generate")):
        assert [c[1] for c in agent.driver.calls if c[0] == operation] == [
            c[1] for c in expected if c[0] == kind
        ], operation


def test_a_go_away_during_a_long_reply_leaves_it_to_finish_and_renews_after_it(monkeypatch):
    """Gemini Live's goAway comes 1 s into a reply that goes on 8 s more. The frames captured
    meanwhile wait for the reply instead of failing, so the reply is not given up and its click
    executes; once it is accepted the connection is renewed, by resuming."""
    clock, records = FakeClock(), []
    record = _StreamingShell._record

    def recording(shell, entry):
        records.append(entry)
        record(shell, entry)

    monkeypatch.setattr(_StreamingShell, "_record", recording)
    click = Call(0.5, "c1", "click", {"x": 10, "y": 20, "button": "left", "frame": 1})
    with clock.participant():
        agent = make_agent(monkeypatch, clock)
        agent.driver.replies += [Reply((click, GoAway(1.0)), ends_at=9.0)]
        agent.driver.replies += [Reply(ends_at=1.0)] * 20
        env = _Env(clock)
        env.seconds["execute"] = 0.25
        run_streaming_episode(
            agent,
            env,
            StreamingConfig(duration_s=14.0),
            clocks=Clocks(clock, clock),
            log=lambda message: None,
        )
        agent.close()
    assert not env._invalid_run
    abandoned = [e.data for e in records if e.kind == "GenerationAbandoned"]
    assert all(data["reason"] == "deadline" for data in abandoned), abandoned
    dropped = [e.data["reason"] for e in records if e.kind == "FrameDropped"]
    assert "push_failed" not in dropped, dropped
    assert ("c1", "applied") in [(a["call_id"], a["result"]) for a in agent.action_history]
    calls = [c[:2] for c in agent.driver.calls]
    assert ("interrupt", "streaming-1") not in calls
    renewal = calls.index(("sync", "go_away"))
    assert calls.index(("go_away", "streaming-1")) < renewal < calls.index(("reply", 2))


def test_a_push_whose_own_reopen_never_ends_before_the_first_reply_invalidates_the_run(
    monkeypatch,
):
    """The session is closed as the episode starts, and the first push's attempt to reopen it
    never ends. Nothing else bounds that attempt: the push watchdog times a push's own attempts
    (decision N48, which decision 13 keeps), and with no reply to give up the run is invalid 10 s
    after the push started at 0.75, behind the first read and capture, instead of running to its
    deadline with no reply."""
    clock = FakeClock()
    with clock.participant():
        agent = make_agent(monkeypatch, clock)
        agent.abort(99)
        agent.driver.connect_s = 100.0
        termination = run_streaming_episode(
            agent,
            _Env(clock),
            StreamingConfig(duration_s=20.0),
            clocks=Clocks(clock, clock),
            log=lambda message: None,
        ).termination
        agent.close()
    assert (termination.invalid.reason, termination.ended_at) == ("frame_push_stuck", 10.75)


def test_a_push_reopening_the_session_through_its_retries_is_not_taken_as_stuck(monkeypatch):
    """The session is closed as the episode starts, and the first push's attempts to reopen it
    fail three times: with their backoff (10.4 s for this view) and the attempts it takes over
    12 s, past the push watchdog, with no reply yet to give up. The session is being prepared
    while the push waits between attempts (decisions N48 and 13), so the run goes on instead of
    ending invalid (review-final-streaming 中-1)."""
    clock, records = FakeClock(), []
    record = _StreamingShell._record

    def recording(shell, entry):
        records.append(entry)
        record(shell, entry)

    monkeypatch.setattr(_StreamingShell, "_record", recording)
    with clock.participant():
        agent = make_agent(monkeypatch, clock)
        agent.abort(99)
        agent.driver.connect_s = 0.5
        agent.driver.connect_failures = [ConnectionError("1011 internal error")] * 3
        agent.driver.replies += [Reply(ends_at=1.0)] * 20
        env = _Env(clock)
        run_streaming_episode(
            agent,
            env,
            StreamingConfig(duration_s=20.0),
            clocks=Clocks(clock, clock),
            log=lambda message: None,
        )
        agent.close()
    assert not env._invalid_run
    pushed = [e.data for e in records if e.kind == "FramePushed"]
    assert pushed[0].error is None and pushed[0].sent_at > 12.0
    kinds = [e.kind for e in records if e.kind in ("SessionPreparing", "SessionPrepared")]
    assert kinds[:6] == ["SessionPreparing", "SessionPrepared"] * 3


def test_a_reply_given_up_while_its_click_runs_shows_in_the_sessions_rebuilt_once_it_ran(
    monkeypatch,
):
    """Decision 25: the first reply's click runs from 2.25, after the read asked for it, and its
    reply never ends: the watchdog gives it up at 2.75, while the read after the click runs, and
    the next reply reopens the session without it. Once the click is applied, the reply's output
    stands before the click's result in every session reopened after."""
    clock = FakeClock()
    click = Call(0.5, "c1", "click", {"x": 10, "y": 20, "button": "left", "frame": 1})
    with clock.participant():
        agent = make_agent(monkeypatch, clock)
        agent.driver.replies += [Reply((click,), ends_at=99.0)] + [Reply(ends_at=99.0)] * 5
        env = _Env(clock)
        env.seconds["execute"] = 0.25
        run_streaming_episode(
            agent,
            env,
            StreamingConfig(duration_s=6.0, reply_watchdog_s=2.0),
            clocks=Clocks(clock, clock),
            log=lambda message: None,
        )
        agent.close()
    assert ("c1", "applied") in [(a["call_id"], a["result"]) for a in agent.action_history]
    first, *later = [call[1] for call in agent.driver.calls if call[0] == "connect"][1:]
    assert "d0.output" not in first and "d0.c1.feedback" not in first
    assert later
    for ids in later:
        assert ids.index("d0.output") < ids.index("d0.c1.feedback")


class _InspectedEnv(_Env):
    """Captures with the canonical interactables the SoM runtime inspects; those of ``unstable``
    kept changing while it was captured."""

    def __init__(self, clock, unstable: str):
        super().__init__(clock)
        self.unstable = unstable

    def capture(self):
        observation = super().capture()
        observation_id = observation.metadata["observation_id"]
        digest = hashlib.sha256(observation.screenshot).hexdigest()
        raw = {
            "scene": {"status": "ok", "kind": "browser_dom", "target_id": "page-1"},
            "binding": {"stable": observation_id != self.unstable, "fingerprint": observation_id},
        }
        observation.metadata["screenshot_sha256"] = digest
        observation.metadata["interactables"] = canonical_result(
            raw,
            observation_id=observation_id,
            screenshot_sha256=digest,
            captured_at=observation.timestamp,
            frame_geometry=FrameGeometry((64, 48), (64, 48)),
        )
        return observation


@pytest.mark.parametrize("first_reply_s", [2.3, 3.3])
def test_a_frame_whose_inspection_is_invalid_invalidates_the_run_whenever_replies_start(
    monkeypatch, first_reply_s
):
    """Frame 3's elements kept changing while it was captured, at 2.75 s. The first reply,
    started on frame 1, clicks frame 3 at 2.95 s, and ends before frame 3 is the newest or after
    frame 4 is. Either way the run is invalid as a turn-based decision on that frame makes it,
    once the frame is inspected, and the frame never goes in, nor does the click run
    (review-merge bug #1: one timing left the run valid with the click applied, the other
    invalidated it for the tool)."""
    clock, records = FakeClock(), []
    record = _StreamingShell._record

    def recording(shell, entry):
        records.append(entry)
        record(shell, entry)

    monkeypatch.setattr(_StreamingShell, "_record", recording)
    click = Call(2.2, "c3", "click", {"x": 10, "y": 20, "button": "left", "frame": 3})
    with clock.participant():
        agent = make_agent(monkeypatch, clock)
        agent.bind_tool_runtime(ToolRuntime(None, clock=clock))
        agent.driver.replies += [Reply((click,), ends_at=first_reply_s)]
        agent.driver.replies += [Reply(ends_at=1.0)] * 10
        env = _InspectedEnv(clock, "obs-3")
        env.seconds["execute"] = 0.25
        termination = run_streaming_episode(
            agent,
            env,
            StreamingConfig(duration_s=8.0),
            clocks=Clocks(clock, clock),
            log=lambda message: None,
        ).termination
        agent.close()
    assert env._invalid_run == {
        "category": "harness",
        "reason": "inspection_protocol_error",
        "detail": "interactables_binding_unstable",
    }
    assert termination.ended_at == 2.75
    assert [c[1] for c in agent.driver.calls if c[0] == "push"] == ["obs-1", "obs-2"]
    assert ("c3", "applied") not in [(a["call_id"], a["result"]) for a in agent.action_history]
    assert env.counts.get("execute", 0) == 0
