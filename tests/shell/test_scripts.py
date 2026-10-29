"""run_script episodes on the kernel's shell (gui+cli action space): the real kernel, shell, action
gateway and workspace, with the scripted environment and agent of tests/shell/world.py.

A script is one decision's action that runs as a wait lets time pass (kernel/turns.py SCRIPT):
each action it takes through the gateway is executed by the environment as a model's action is,
on the shell's thread, with the check right after it; the decision settles once the script's
process is gone, with its report as the outcome's detail. The script's timeout, the task's end
and the deadline each kill it. Scripts are real subprocesses (the local workspace backend), so
these episodes run on real wall clocks, a few seconds each.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from realtime_framework.clock import Clocks, SystemClock
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.entity.config import RunnerTimingConfig
from realtime_framework.entity.records import _jsonable
from realtime_framework.evaluation.kernel.turns import TurnConfig
from realtime_framework.evaluation.shell import run_episode
from realtime_framework.workspace.runner import LocalBackend, Workspace

from .world import Agent, Environment, Reply, Trace, World

PNG = b"\x89PNG\r\n\x1a\n" + bytes(8) + (640).to_bytes(4, "big") + (360).to_bytes(4, "big")
WAIT = Action.wait(0.2)


class ScriptEnvironment(Environment):
    def script_screenshot(self) -> bytes:
        self.log("script_screenshot")
        return PNG


def script(code: str, timeout_s: float | None = None) -> Action:
    parameters = {"language": "python", "code": code}
    if timeout_s is not None:
        parameters["timeout_s"] = timeout_s
    return Action(ActionType.RUN_SCRIPT, parameters)


def episode(tmp_path, duration_s, replies, world=None):
    clock = SystemClock()
    clocks = Clocks(clock, clock)
    turn = TurnConfig("wall", duration_s, 1.0, 1.0, timing=RunnerTimingConfig())
    env = ScriptEnvironment(clocks, "wall", 1.0, world or World())
    agent = Agent(clock, replies)
    trace = Trace()
    gateway_dir = Path(tempfile.mkdtemp(prefix="rtg-shell-"))
    workspace = Workspace(tmp_path / "workspace", gateway_dir, LocalBackend(), clock)
    logs: list[str] = []
    result = run_episode(
        agent, env, turn, clocks=clocks, log=logs.append, trace=trace, workspace=workspace
    )
    result.drain_abandoned()
    assert not gateway_dir.exists(), "the gateway's directory outlived the episode"
    return env, agent, trace, result, workspace


def script_executes(env) -> list[tuple]:
    return [(call, details) for call, _, details in env.calls if call in ("execute",
            "script_screenshot")]  # fmt: skip


def finished(trace) -> list[dict]:
    """The ScriptFinished records, as the episode trace writes them."""
    return [
        _jsonable(record["payload"]["data"])
        for record in trace.records
        if record["event"] == "KernelRecord" and record["payload"]["kind"] == "ScriptFinished"
    ]


def test_a_script_acts_through_the_environment_and_reports(tmp_path):
    code = (
        "import rtact\n"
        "shot = rtact.screenshot()\n"
        "print(shot.width, shot.height, shot.path.endswith('obs/00001.png'))\n"
        "for name in ('Up', 'Down'):\n"
        "    print(rtact.key(name).status)\n"
        "print(rtact.hold_key('Right', 0.2).status)\n"
    )
    env, agent, trace, result, workspace = episode(
        tmp_path, 4.0, [Reply(action=script(code)), *[Reply(0.1, WAIT)] * 20]
    )
    assert result.invalid_run is None
    first = agent.outcomes[0]
    assert first.status == "applied" and first.action.action_type is ActionType.RUN_SCRIPT
    assert "run_script exit_code=0" in first.detail
    # Screenshots are no actions.
    assert "gateway actions=3 (rejected 0)" in first.detail
    assert "640 360 True\napplied\napplied\napplied" in first.detail
    calls = script_executes(env)
    assert calls[:4] == [
        ("script_screenshot", {}),
        ("execute", {"action": "key", "release_at": None}),
        ("execute", {"action": "key", "release_at": None}),
        ("execute", {"action": "hold_key", "release_at": pytest.approx(calls[3][1]["release_at"])}),
    ]
    assert calls[3][1]["release_at"] is not None
    (record,) = finished(trace)
    assert record["result"]["exit_code"] == 0 and record["result"]["killed"] is None
    assert (workspace.root / "obs" / "00001.png").read_bytes() == PNG
    # Each script action is checked for the end right after it, as a model action is.
    names = env.names()
    executes = [i for i, name in enumerate(names) if name == "execute"]
    assert all(names[i + 1] == "check_end" for i in executes[:3])
    resolved = [r["payload"] for r in trace.records if r["event"] == "ActionResolved"]
    assert resolved[0]["status"] == "applied"
    # The output is kept once, in ScriptFinished; the action's records hold its summary.
    summary = resolved[0]["execution_result"]["script"]
    assert summary["exit_code"] == 0 and summary["stdout_bytes"] > 0
    assert "stdout" not in summary and "stderr" not in summary
    assert env.lifecycle[0]["execution_result"] == resolved[0]["execution_result"]


def test_a_script_past_its_timeout_is_killed_and_the_episode_goes_on(tmp_path):
    code = "import rtact, time\nwhile True:\n    rtact.key('a')\n    time.sleep(0.05)\n"
    env, agent, trace, result, _ = episode(
        tmp_path, 4.0, [Reply(action=script(code, timeout_s=0.8)), *[Reply(0.1, WAIT)] * 20]
    )
    assert result.invalid_run is None
    first = agent.outcomes[0]
    assert first.status == "applied"
    assert "run_script killed (timeout)" in first.detail
    (record,) = finished(trace)
    assert record["result"]["killed"] == "timeout"
    taps = [d for c, d in script_executes(env) if c == "execute"]
    assert 3 <= len(taps)
    # The next decisions came after it.
    assert len(agent.outcomes) >= 2
    assert all(o.action.action_type is ActionType.WAIT for o in agent.outcomes[1:])


def test_the_tasks_end_stops_the_script(tmp_path):
    code = (
        "import rtact\n"
        "for i in range(1000):\n"
        "    reply = rtact.key('a')\n"
        "    if reply.done:\n"
        "        print('done after', i)\n"
        "        break\n"
        "else:\n"
        "    import time; time.sleep(30)\n"
    )
    world = World(ends_after=3, end_reason="game_won")
    env, agent, trace, result, _ = episode(tmp_path, 10.0, [Reply(action=script(code))], world)
    assert result.invalid_run is None
    (outcome,) = agent.outcomes
    assert outcome.status == "applied" and outcome.terminal_after_action
    assert env.applied == 3
    (record,) = finished(trace)
    # The script heard it was done, or was killed first; either way it acted no more.
    assert (
        record["result"]["killed"] == "episode_end" or "done after 3" in record["result"]["stdout"]
    )
    termination = [
        _jsonable(r["payload"]["data"])
        for r in trace.records
        if r["event"] == "KernelRecord" and r["payload"]["kind"] == "Termination"
    ]
    assert termination[0]["stop_reason"] == "game_won"


def test_the_deadline_kills_a_script(tmp_path):
    code = "import time\nprint('started', flush=True)\ntime.sleep(30)\n"
    _env, agent, trace, _result, _ = episode(tmp_path, 1.5, [Reply(action=script(code))])
    (outcome,) = agent.outcomes
    assert outcome.terminal_after_action
    assert "run_script killed (episode_end)" in outcome.detail
    assert "started" in outcome.detail
    (record,) = finished(trace)
    assert record["result"]["killed"] == "episode_end"
    assert record["result"]["duration_s"] < 5


def test_a_refused_action_is_the_scripts_to_handle(tmp_path):
    world = World(rejections={0: "outside the page"})
    code = "import rtact\nreply = rtact.click(5, 5)\nprint(reply.ok, reply.status, reply.detail)\n"
    _env, agent, _trace, result, _ = episode(
        tmp_path, 3.0, [Reply(action=script(code)), *[Reply(0.1, WAIT)] * 20], world
    )
    assert result.invalid_run is None
    first = agent.outcomes[0]
    assert "False rejected outside the page" in first.detail
    assert "gateway actions=1 (rejected 1)" in first.detail


def test_a_failed_environment_invalidates_the_run(tmp_path):
    world = World(failures={0: "guest unreachable"})
    code = "import rtact\nprint(rtact.key('a').done)\nimport time; time.sleep(30)\n"
    _env, agent, trace, result, _ = episode(tmp_path, 5.0, [Reply(action=script(code))], world)
    assert result.invalid_run["reason"] == "action_execution_failed"
    (outcome,) = agent.outcomes
    assert outcome.status == "invalid_run"
    (record,) = finished(trace)
    assert record["result"]["killed"] == "episode_end" or "True" in record["result"]["stdout"]


def test_a_script_cannot_start_without_a_workspace(tmp_path):
    clock = SystemClock()
    clocks = Clocks(clock, clock)
    turn = TurnConfig("wall", 3.0, 1.0, 1.0, timing=RunnerTimingConfig())
    env = ScriptEnvironment(clocks, "wall", 1.0, World())
    agent = Agent(clock, [Reply(action=script("print(1)"))])
    result = run_episode(agent, env, turn, clocks=clocks, log=lambda m: None, trace=Trace())
    assert result.invalid_run["reason"] == "script_start_failed"
