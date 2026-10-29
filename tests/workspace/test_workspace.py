"""The workspace where a run_script runs and the action gateway it acts through.

The gateway (workspace/gateway.py) takes a valid GUI action or a screenshot request alone and
hands it to the runner one at a time; anything else is refused before it reaches the runner, and
once shut every request is answered done. The workspace (workspace/runner.py) runs each script
as a process group of its own from a file holding its source, keeps its files between scripts,
and kills the whole group when asked. These run real subprocesses on the local backend; the
apptainer backend's isolation is in the command it builds, which is checked here, and is
exercised on a host with Apptainer by tests/workspace/test_isolation.py.
"""

from __future__ import annotations

import json
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest

from realtime_framework.clock import SystemClock
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.workspace import rtact
from realtime_framework.workspace.gateway import SOCKET_NAME, ActionGateway, parse_request
from realtime_framework.workspace.runner import (
    TRACE_OUTPUT_BYTES,
    ApptainerBackend,
    LocalBackend,
    ScriptResult,
    Workspace,
    script_feedback,
)


def script(code: str, language: str = "python") -> Action:
    return Action(ActionType.RUN_SCRIPT, {"language": language, "code": code})


@pytest.fixture
def gateway_dir():
    # A unix socket path is limited to 107 bytes; pytest's tmp_path can be longer.
    directory = Path(tempfile.mkdtemp(prefix="rtg-test-"))
    yield directory
    shutil.rmtree(directory)


class Recorder:
    """A runner that applies every action and writes a 1x1 PNG for every screenshot."""

    def __init__(self, workspace: Workspace | None = None, *, reject: str | None = None):
        self.workspace = workspace
        self.reject = reject
        self.requests: list[tuple[str, Action | None]] = []

    def __call__(self, kind, action):
        self.requests.append((kind, action))
        if kind == "screenshot":
            path, inside = self.workspace.screenshot_path()
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(16))
            return {"ok": True, "done": False, "status": "captured", "path": inside}
        if self.reject and action.parameters.get("key") == self.reject:
            return {"ok": False, "done": False, "status": "rejected", "detail": "refused"}
        return {"ok": True, "done": False, "status": "applied", "detail": ""}


# The gateway's requests


@pytest.mark.parametrize(
    ("line", "kind", "action_type"),
    [
        ('{"op": "screenshot"}', "screenshot", None),
        ('{"op": "act", "action": {"action": "key", "key": "Up"}}', "act", ActionType.KEY),
        (
            '{"op": "act", "action": {"action": "hold_key", "key": "Right", "duration": 0.3}}',
            "act",
            ActionType.HOLD_KEY,
        ),
        ('{"op": "act", "action": {"action": "click", "x": 10, "y": 20}}', "act", ActionType.CLICK),
    ],
)
def test_the_gateway_takes_gui_actions_and_screenshots(line, kind, action_type):
    got, action = parse_request(line)
    assert got == kind
    assert (action.action_type if action is not None else None) == action_type


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[]",
        '{"op": "eval", "js": "document.title"}',
        '{"op": "state"}',
        '{"op": "screenshot", "frame": 1}',
        '{"op": "act", "action": {"action": "run_script", "language": "bash", "code": "id"}}',
        '{"op": "act", "action": {"action": "wait", "duration": 1}}',
        '{"op": "act", "action": {"action": "done"}}',
        '{"op": "act", "action": {"action": "click", "target": "P1"}}',
        '{"op": "act", "action": {"action": "key"}}',
        '{"op": "act", "action": {"action": "hold_key", "key": "a", "duration": 99}}',
        '{"op": "act", "action": "key Up"}',
    ],
)
def test_the_gateway_refuses_everything_else(line):
    kind, refusal = parse_request(line)
    assert kind is None
    assert refusal["ok"] is False and refusal["done"] is False


def test_requests_reach_the_runner_only_while_open(gateway_dir, monkeypatch):
    recorder = Recorder()
    gateway = ActionGateway(gateway_dir, recorder)
    gateway.start()
    try:
        monkeypatch.setenv(rtact.SOCKET_ENV, str(gateway_dir / SOCKET_NAME))
        connection = rtact._Connection()
        # Closed: no script runs.
        assert connection.request({"op": "act", "action": {"action": "key", "key": "a"}}).done
        assert recorder.requests == []
        gateway.open()
        connection = rtact._Connection()
        reply = connection.request({"op": "act", "action": {"action": "key", "key": "a"}})
        assert (reply.ok, reply.status) == (True, "applied")
        refused = connection.request({"op": "act", "action": {"action": "wait"}})
        assert (refused.ok, refused.done, refused.status) == (False, False, "invalid")
        assert [kind for kind, _ in recorder.requests] == ["act"]
        gateway.shut()
        assert connection.request({"op": "screenshot"}).done
        assert gateway.requests == 4 and gateway.refused == 1
    finally:
        gateway.close()
    assert not (gateway_dir / SOCKET_NAME).exists()


def test_requests_reach_the_runner_one_at_a_time(gateway_dir):
    inside = threading.Lock()
    overlaps = []

    def slow(kind, action):
        if not inside.acquire(blocking=False):
            overlaps.append(kind)
            return {"ok": True}
        try:
            time.sleep(0.05)
            return {"ok": True, "done": False, "status": "applied"}
        finally:
            inside.release()

    gateway = ActionGateway(gateway_dir, slow)
    gateway.start()
    gateway.open()
    try:

        def client():
            with socket.socket(socket.AF_UNIX) as sock:
                sock.connect(str(gateway.path))
                stream = sock.makefile("rw")
                for _ in range(3):
                    stream.write(json.dumps({"op": "screenshot"}) + "\n")
                    stream.flush()
                    assert json.loads(stream.readline())["ok"]

        threads = [threading.Thread(target=client) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
    finally:
        gateway.close()
    assert overlaps == []


# The workspace


@pytest.fixture
def workspace(tmp_path, gateway_dir):
    ws = Workspace(tmp_path / "workspace", gateway_dir, LocalBackend(), SystemClock())
    ws.prepare()
    return ws


def test_a_script_runs_from_its_source_and_keeps_its_files(workspace):
    first = workspace.run(script("open('notes.txt', 'w').write('seen 1')\nprint('hi')"))
    assert (first.exit_code, first.killed, first.stdout) == (0, None, "hi\n")
    second = workspace.run(script("cat notes.txt; echo; echo oops >&2; exit 3", "bash"))
    assert (second.exit_code, second.stdout, second.stderr) == (3, "seen 1\n", "oops\n")
    sources = sorted(p.name for p in (workspace.root / ".scripts").iterdir())
    assert sources[:2] == ["0001.py", "0001.stderr"]
    assert (workspace.root / ".scripts" / "0002.sh").read_text().startswith("cat notes.txt")


def test_a_script_acts_through_rtact(workspace):
    recorder = Recorder(workspace, reject="F12")
    gateway = ActionGateway(workspace.gateway_dir, recorder)
    gateway.start()
    gateway.open()
    try:
        result = workspace.run(
            script(
                "import rtact\n"
                "shot = rtact.screenshot()\n"
                "print(shot.status, shot.path.endswith('obs/00001.png'))\n"
                "print(rtact.hold_key('Right', 0.25).status)\n"
                "print(rtact.key('F12').detail)\n"
                "print(rtact.click(5, 6).ok)\n"
            )
        )
        bash = workspace.run(script("rtact key Up; echo exit=$?", "bash"))
    finally:
        gateway.close()
    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines() == ["captured True", "applied", "refused", "True"]
    assert '"status": "applied"' in bash.stdout and "exit=0" in bash.stdout
    kinds = [(kind, action and action.action_type) for kind, action in recorder.requests]
    assert kinds == [
        ("screenshot", None),
        ("act", ActionType.HOLD_KEY),
        ("act", ActionType.KEY),
        ("act", ActionType.CLICK),
        ("act", ActionType.KEY),
    ]
    assert recorder.requests[1][1].parameters == {"key": "Right", "duration": 0.25}


def test_a_killed_script_takes_its_children_with_it(workspace):
    marker = workspace.root / "child.pid"
    action = script(f"sleep 30 & echo $! > {marker}; wait", "bash")
    results = []
    worker = threading.Thread(target=lambda: results.append(workspace.run(action)))
    worker.start()
    for _ in range(100):
        if marker.exists() and marker.read_text().strip():
            break
        time.sleep(0.05)
    workspace.kill("timeout")
    worker.join(10)
    (result,) = results
    assert result.killed == "timeout" and result.exit_code == -9
    status = Path(f"/proc/{int(marker.read_text())}/status")
    for _ in range(50):
        try:
            # Gone, or a zombie its reaper has not collected yet.
            if "zombie" in status.read_text():
                break
        except (FileNotFoundError, ProcessLookupError):  # gone before or during the read
            break
        time.sleep(0.05)
    else:
        raise AssertionError("the script's child outlived it")


def test_output_is_capped_for_the_trace_and_the_feedback(workspace):
    result = workspace.run(script(f"print('x' * {TRACE_OUTPUT_BYTES * 2})"))
    assert result.stdout_bytes == TRACE_OUTPUT_BYTES * 2 + 1
    assert len(result.stdout) == TRACE_OUTPUT_BYTES
    feedback = script_feedback(result, actions=3, rejected=1)
    assert feedback.startswith("run_script exit_code=0 after ")
    assert "gateway actions=3 (rejected 1)" in feedback
    assert "characters cut]" in feedback and len(feedback) < 5000


def test_a_script_that_cannot_start_says_so():
    result = ScriptResult(None, None, 0.0, "", "", 0, 0, error="no such file")
    assert script_feedback(result, 0, 0) == "run_script could not start: no such file"


def test_the_apptainer_instance_is_cut_off_and_started_once(tmp_path, gateway_dir, monkeypatch):
    """The instance is started once, with every isolation option; a script only execs into it,
    and what it left is killed after it."""
    from realtime_framework.workspace import runner

    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **kwargs: calls.append(list(argv)))
    # Apptainer itself would take these: a mount, and a variable set in the script's environment.
    monkeypatch.setenv("APPTAINER_BIND", "/:/host")
    monkeypatch.setenv("APPTAINERENV_LEAK", "1")
    monkeypatch.setenv("SINGULARITY_BIND", "/:/host")
    backend = ApptainerBackend(Path("/img/workspace.sif"))
    ws = Workspace(tmp_path / "ws", gateway_dir, backend, SystemClock())
    ws.prepare()
    ws.prepare()  # once
    (start,) = calls
    name = backend.instance
    joined = " ".join(start)
    assert start[:3] == ["apptainer", "instance", "start"]
    for isolation in (
        "--containall",
        "--cleanenv",
        "--net --network none",
        "--no-mount hostfs,bind-paths,cwd,sys,devpts",
        f"--home {ws.root}:/workspace",
        f"--bind {gateway_dir}:/rtact:ro",
    ):
        assert isolation in joined
    # The task sandbox's display, devtools and services are never named.
    assert "DISPLAY" not in joined and start.count("--bind") == 1
    assert start[-2:] == ["/img/workspace.sif", name]

    argv, env = backend.command(ws, "python3", ws.root / ".scripts" / "0001.py")
    assert argv[:3] == ["apptainer", "exec", "--cleanenv"]
    assert argv[-3:] == [f"instance://{name}", "python3", "/workspace/.scripts/0001.py"]
    assert "PYTHONPATH=/rtact" in argv and "PATH=/rtact/bin:/usr/local/bin:/usr/bin:/bin" in argv
    assert "DISPLAY" not in " ".join(argv)
    # Apptainer keeps the host's session (rootless cgroups need its bus), none of its own config.
    assert "PATH" in env and not any(k.startswith(("APPTAINER", "SINGULARITY")) for k in env)

    backend.sweep(ws)
    assert calls[-1][-4:] == [f"instance://{name}", "sh", "-c", "kill -9 -1 2>/dev/null; true"]
    ws.close()
    ws.close()  # once
    assert calls[-1] == ["apptainer", "instance", "stop", "--force", name]
    assert len(calls) == 3 and not gateway_dir.exists()
    gateway_dir.mkdir()  # for the fixture's cleanup


def test_a_script_cannot_run_before_its_instance(tmp_path, gateway_dir):
    backend = ApptainerBackend(Path("/img/workspace.sif"))
    backend.start = lambda workspace: None  # an instance that never came up
    ws = Workspace(tmp_path / "ws", gateway_dir, backend, SystemClock())
    ws.prepare()
    result = ws.run(script("print(1)"))
    assert result.error == "the workspace instance is not running"
