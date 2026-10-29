"""A gui+cli episode end to end, offline: the chat_completions agent that runner.set_up_agent
builds (build.py), a scripted OpenAI-compatible endpoint (stub.py), runner.run_stream with the
real kernel and shell, and a real script in a workspace on the local backend, acting through the
real action gateway on the replay environment (fake_env.py).

The model's system prompt tells it how to run a script; its first answer is a run_script whose
script takes a screenshot and presses keys through rtact; the environment executes those keys
as it executes model actions; and the model's next request carries the script's EXECUTION RESULT,
its exit code and output. The episode trace keeps the script's full record. Real clocks: the
script is a real process. The isolation of the apptainer backend is tests/workspace/
test_isolation.py's.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from realtime_framework.clock import Clocks, SystemClock
from realtime_framework.evaluation import runner
from realtime_framework.workspace.runner import LocalBackend, Workspace

from .build import build_episode
from .fake_env import FakeEnvironment
from .frames import ReplayTask
from .scenario import TASK_INSTRUCTION
from .stub import ChatCompletionsStub, Reply
from .test_chat_completions import MODEL_KEY, chat_completions_config

SCRIPT = (
    "import rtact\n"
    "shot = rtact.screenshot()\n"
    "print('screenshot', shot.width, shot.height)\n"
    "for name in ('Tab', 'Tab', 'Return'):\n"
    "    print(name, rtact.key(name).status)\n"
    "open('notes.md', 'w').write('pressed submit')\n"
)
USAGE = {"prompt_tokens": 1500, "completion_tokens": 40, "total_tokens": 1540}


class ScriptEnvironment(FakeEnvironment):
    def script_screenshot(self) -> bytes:
        self.log("env.script_screenshot")
        return self.frames.capture(self.tick_index(), at=self.clock.time()).screenshot


def _messages(body: bytes) -> list[dict]:
    return json.loads(body)["messages"]


def _text(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if part.get("type") == "text")


def test_a_gui_cli_episode_runs_a_script_and_reports_it_to_the_model(tmp_path):
    clock = SystemClock()
    # As models write it: the code's newlines raw in the JSON string (script_answer_text).
    run_script = json.dumps(
        {"action": "run_script", "language": "python", "code": SCRIPT, "timeout_s": 20}
    ).replace("\\n", "\n")
    wait = json.dumps({"action": "wait", "duration": 0.5})
    replies = [Reply(run_script, USAGE), *[Reply(wait, USAGE)] * 12]
    with ChatCompletionsStub(replies) as stub:
        config = chat_completions_config(stub.base_url, stream=False)
        episode = build_episode(
            config,
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=MODEL_KEY,
            clock_mode="wall",
            clock=clock,
            action_space="gui+cli",
        )
        env = ScriptEnvironment(
            episode.frames, clock, fps=1.0, max_minutes=4 / 60, reply_latencies=[]
        )
        gateway_dir = Path(tempfile.mkdtemp(prefix="rtg-e2e-"))
        workspace = Workspace(tmp_path / "workspace", gateway_dir, LocalBackend(), clock)
        result = None
        try:
            result = runner.run_stream(
                episode.agent,
                env,
                clock_mode="wall",
                duration_s=4.0,
                env_fps=1.0,
                input_fps=1.0,
                trace=episode.trace,
                agent_trace=episode.agent_trace,
                clocks=Clocks(clock, clock),
                log=lambda line: None,
                workspace=workspace,
            )
        finally:
            if result is not None:
                result.drain_abandoned()
            episode.finish()
        requests = [received.body for received in stub.requests]

    assert result.invalid_run is None, result.invalid_run
    # The prompt says how to run a script.
    system = _text(_messages(requests[0])[0])
    assert "SCRIPTS" in system and '"action": "run_script"' in system
    # The script's keys ran in the environment as model actions run, in desktop pixels.
    keys = [
        execution["action"]
        for execution in env.executions
        if execution["decision"] == 0 and execution["status"] == "applied"
    ]
    assert [(key["action_type"], key["parameters"]) for key in keys] == [
        ("key", {"key": "Tab"}),
        ("key", {"key": "Tab"}),
        ("key", {"key": "Return"}),
    ]
    assert {key["coordinate_space"] for key in keys} == {"desktop_px"}
    assert any(call["call"] == "env.script_screenshot" for call in env.calls)
    # The next request reports it to the model.
    second = "\n".join(_text(message) for message in _messages(requests[1]))
    assert "EXECUTION RESULT: actual=run_script" in second
    assert "status=applied" in second
    assert "run_script exit_code=0" in second and "gateway actions=3 (rejected 0)" in second
    assert "Return applied" in second and "screenshot " in second
    # Its files stay in the workspace; the trace keeps its whole record.
    assert (workspace.root / "notes.md").read_text() == "pressed submit"
    trace = [
        json.loads(line) for line in (tmp_path / "episode_trace.jsonl").read_text().splitlines()
    ]
    finished = [
        record["payload"]["data"]
        for record in trace
        if record.get("event") == "KernelRecord" and record["payload"]["kind"] == "ScriptFinished"
    ]
    assert len(finished) == 1
    assert finished[0]["result"]["exit_code"] == 0
    assert "Tab applied" in finished[0]["result"]["stdout"]
