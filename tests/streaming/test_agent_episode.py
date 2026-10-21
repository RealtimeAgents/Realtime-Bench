"""A whole native-streaming episode: the event kernel and the streaming shell
(evaluation/streaming_shell.py) on the fake environment of tests/shell_streaming, driving the
native-streaming agent (harnesses/streaming/agent.py) on a fake live encoder, on the fake clock.

Every command the kernel gives the agent comes to one agent call of its kind, in the kernel's
order, and every push and reply to one encoder operation."""

from __future__ import annotations

from realtime_framework.clock import Clocks
from realtime_framework.evaluation.kernel.events import (
    AbortGeneration,
    AcceptResult,
    Commit,
    DiscardResult,
    StartGeneration,
)
from realtime_framework.evaluation.kernel.streaming import PushFrame, StreamingConfig
from realtime_framework.evaluation.streaming_shell import _StreamingShell, run_streaming_episode
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
