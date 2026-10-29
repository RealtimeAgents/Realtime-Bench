"""The apptainer workspace's isolation, on a host with Apptainer and the workspace image.

The task sandbox shares the host network (sandbox/apptainer.py): the task's services, its state
API and Chrome's devtools port listen on the host's 127.0.0.1, and its X display serves there too.
A script must reach none of it, nor the host's files, and still reach the action gateway. Each
test starts a real script in the workspace image through ApptainerBackend; they are skipped where
Apptainer cannot run the workspace image (``RTBENCH_ISOLATION_TESTS=1`` makes that a failure instead).
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path

import pytest

from realtime_framework.clock import SystemClock
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.sandbox.apptainer import image_path
from realtime_framework.workspace.gateway import ActionGateway
from realtime_framework.workspace.runner import ApptainerBackend, Workspace


def _usable() -> str | None:
    """Why the apptainer workspace cannot run here, or None."""
    if shutil.which("apptainer") is None:
        return "apptainer is not installed"
    image = image_path("workspace")
    if not image.is_file():
        return f"no workspace image at {image}; build it with `rtbench setup`"
    probe = subprocess.run(
        ["apptainer", "exec", "--containall", "--net", "--network", "none", str(image), "true"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if probe.returncode:
        return f"apptainer cannot run the image: {probe.stderr.strip()[-300:]}"
    return None


REASON = _usable()
pytestmark = pytest.mark.skipif(
    REASON is not None and os.environ.get("RTBENCH_ISOLATION_TESTS") != "1",
    reason=REASON or "",
)


@pytest.fixture
def workspace(tmp_path):
    gateway_dir = Path(tempfile.mkdtemp(prefix="rtg-iso-"))
    ws = Workspace(
        tmp_path / "ws", gateway_dir, ApptainerBackend(image_path("workspace")), SystemClock()
    )
    ws.prepare()
    yield ws
    ws.close()


def run(ws: Workspace, code: str, language: str = "python"):
    return ws.run(Action(ActionType.RUN_SCRIPT, {"language": language, "code": code}))


@pytest.fixture
def host_service():
    """A TCP service on the host's loopback, where the task's state API and devtools listen."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    accepted = []

    def serve():
        try:
            while True:
                connection, _ = server.accept()
                accepted.append(connection)
                connection.sendall(b"HTTP/1.0 200 OK\r\n\r\nsecret state")
                connection.close()
        except OSError:
            return

    threading.Thread(target=serve, daemon=True).start()
    yield server.getsockname()[1], accepted
    server.close()


def test_a_script_cannot_reach_the_hosts_services(workspace, host_service):
    port, accepted = host_service
    code = (
        "import socket\n"
        "for host in ('127.0.0.1', 'localhost'):\n"
        "    try:\n"
        f"        socket.create_connection((host, {port}), timeout=2).recv(100)\n"
        "        print('REACHED', host)\n"
        "    except OSError as exc:\n"
        "        print('refused', host, type(exc).__name__)\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
        "    print('REACHED internet')\n"
        "except OSError:\n"
        "    print('no internet')\n"
    )
    result = run(workspace, code)
    assert result.exit_code == 0, result.stderr
    assert "REACHED" not in result.stdout
    assert accepted == []


def test_a_script_has_no_display_and_no_host_files(workspace, tmp_path):
    secret = tmp_path / "host-secret.txt"
    secret.write_text("task answer")
    code = (
        "import os\n"
        "print('DISPLAY', os.environ.get('DISPLAY'))\n"
        "print('x11', os.path.exists('/tmp/.X11-unix'))\n"
        f"print('secret', os.path.exists({str(secret)!r}))\n"
        "print('home', os.getcwd(), sorted(os.listdir('/workspace'))[:3])\n"
        "print('pids', len([p for p in os.listdir('/proc') if p.isdigit()]))\n"
    )
    result = run(workspace, code)
    assert result.exit_code == 0, result.stderr
    lines = dict(line.split(" ", 1) for line in result.stdout.splitlines())
    assert lines["DISPLAY"] == "None"
    assert lines["x11"] == "False"
    assert lines["secret"] == "False"
    assert lines["home"].startswith("/workspace")
    # Its own PID namespace: the script and its interpreter's parents alone.
    assert int(lines["pids"]) < 10


def test_a_script_gets_its_workspace_and_a_read_only_rtact_alone(workspace):
    code = (
        "import os\n"
        "def writable(path):\n"
        "    try:\n"
        "        open(os.path.join(path, 'probe'), 'w').close()\n"
        "        return True\n"
        "    except OSError:\n"
        "        return False\n"
        "print('workspace', writable('/workspace'))\n"
        "print('rtact', writable('/rtact'), sorted(os.listdir('/rtact')))\n"
        "print('sys', os.listdir('/sys') if os.path.isdir('/sys') else [])\n"
        "print('devpts', os.path.isdir('/dev/pts') and os.listdir('/dev/pts'))\n"
        "print('task', [p for p in ('/opt/google', '/opt/guest_agent', '/app/realtime_tasks',\n"
        "    '/usr/bin/xdotool', '/usr/bin/google-chrome') if os.path.exists(p)])\n"
    )
    result = run(workspace, code)
    assert result.exit_code == 0, result.stderr
    lines = dict(line.split(" ", 1) for line in result.stdout.splitlines())
    assert lines["workspace"] == "True"
    assert lines["rtact"] == "False ['bin', 'rtact.py']"
    assert lines["sys"] == "[]"
    assert lines["devpts"] in ("False", "[]")
    assert lines["task"] == "[]"


def test_a_script_still_reaches_the_gateway(workspace):
    requests = []

    def submit(kind, action):
        requests.append((kind, action))
        return {"ok": True, "done": False, "status": "applied", "detail": ""}

    gateway = ActionGateway(workspace.gateway_dir, submit)
    gateway.start()
    gateway.open()
    try:
        result = run(workspace, "import rtact\nprint(rtact.key('Up').status)\n")
        bash = run(workspace, "rtact key Down", "bash")
    finally:
        gateway.close()
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "applied"
    assert bash.exit_code == 0, bash.stderr
    assert [action.parameters["key"] for _, action in requests] == ["Up", "Down"]
