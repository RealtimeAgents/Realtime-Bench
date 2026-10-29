"""The workspace where a run_script runs (gui+cli action space).

One directory per episode, the script's /workspace: its scripts, the screenshots the gateway
writes, and whatever the scripts keep there between calls. Each run_script is one process group,
started from a file holding the script's source verbatim and killed whole when the kernel aborts
it; no process outlives its call.

The script is given its workspace and one read-only directory, /rtact: the gateway's socket and
the rtact client beside it. The ``apptainer`` backend is the benchmark's: the script runs in the
workspace image (sandbox/images/workspace.def: Python, Pillow, bash, nothing of the task's) with
``--containall`` (its own mounts, PIDs and IPC, a clean environment), ``--net --network none``
(loopback alone), so it reaches neither the task's services nor Chrome's devtools port nor the
X display, which the task sandbox serves on the host network, and no host filesystem, /sys or
/dev/pts; only /workspace and /rtact are mounted. Its instance starts before the episode's clock
and lives through the episode; each script is an exec into it. The ``local`` backend runs the
script as a plain subprocess, with nothing isolated: the tests' alone, which a run refuses.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from realtime_framework.entity.actions import Action
from realtime_framework.workspace import rtact
from realtime_framework.workspace.gateway import SOCKET_NAME

# Set to anything but apptainer, a run refuses it (providers/preflight.py check_action_space):
# the local backend is the tests' alone.
WORKSPACE_BACKEND_ENV = "RTBENCH_WORKSPACE_BACKEND"
# Bytes of each output stream the trace keeps, and characters the model's feedback quotes.
TRACE_OUTPUT_BYTES = 64 * 1024
FEEDBACK_OUTPUT_CHARS = 4096
_SUFFIX = {"python": ".py", "bash": ".sh"}
_BASH_RTACT = '#!/bin/sh\nexec python3 -m rtact "$@"\n'


@dataclass
class ScriptResult:
    """How one run_script came out. ``killed`` names why the kernel stopped it ("timeout",
    "episode_end"), None when it exited by itself; ``error`` that it could not be started."""

    exit_code: int | None
    killed: str | None
    duration_s: float
    stdout: str
    stderr: str
    stdout_bytes: int
    stderr_bytes: int
    error: str | None = None

    def record(self) -> dict:
        return asdict(self)

    def summary(self) -> dict:
        """The record without its output, which the episode trace keeps once, in the kernel's
        ScriptFinished record."""
        return {k: v for k, v in asdict(self).items() if k not in ("stdout", "stderr")}


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"[{len(text) - limit} characters cut]\n" + text[-limit:]


def script_feedback(result: ScriptResult, actions: int, rejected: int) -> str:
    """The script's report in the model's feedback (Outcome.detail): how it ended, what it did
    through the gateway, and the tail of each output stream."""
    if result.error is not None:
        return f"run_script could not start: {result.error}"
    ended = (
        f"killed ({result.killed})"
        if result.killed is not None
        else f"exit_code={result.exit_code}"
    )
    parts = [
        (
            f"run_script {ended} after {result.duration_s:.2f}s; gateway actions={actions} "
            f"(rejected {rejected})."
        )
    ]
    for name in ("stdout", "stderr"):
        text = getattr(result, name)
        if text:
            parts.append(f"{name}:\n{_tail(text, FEEDBACK_OUTPUT_CHARS)}")
    return "\n".join(parts)


def _read_capped(path: Path) -> tuple[str, int]:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            if size > TRACE_OUTPUT_BYTES:
                stream.seek(size - TRACE_OUTPUT_BYTES)
            data = stream.read(TRACE_OUTPUT_BYTES)
    except OSError:
        return "", 0
    return data.decode("utf-8", "replace"), size


class LocalBackend:
    """A plain subprocess, unisolated: tests alone construct it (providers/preflight.py
    check_action_space refuses it for a run)."""

    name = "local"

    def start(self, workspace: Workspace) -> None:
        pass

    def stop(self, workspace: Workspace) -> None:
        pass

    def sweep(self, workspace: Workspace) -> None:
        pass

    def inside(self, workspace: Workspace, path: Path) -> str:
        return str(path)

    def command(self, workspace: Workspace, interpreter: str, script: Path):
        client = workspace.gateway_dir
        env = {
            "PATH": f"{client / 'bin'}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(workspace.root),
            "PYTHONPATH": str(client),
            "PYTHONDONTWRITEBYTECODE": "1",
            "RTACT_SOCKET": str(workspace.gateway_dir / SOCKET_NAME),
            "LANG": "C.UTF-8",
        }
        program = sys.executable if interpreter == "python3" else interpreter
        return [program, str(script)], env


def _apptainer_env() -> dict[str, str]:
    """Apptainer's own environment: the host's, which rootless apptainer needs (its session bus
    and runtime directory, to join the instance's cgroup), but none of the variables that would
    configure it (APPTAINER_BIND adds mounts, APPTAINERENV_* sets the script's environment).
    ``--cleanenv`` keeps the rest out of the container."""
    return {k: v for k, v in os.environ.items() if not k.startswith(("APPTAINER", "SINGULARITY"))}


class ApptainerBackend:
    """The workspace image without network, display or the task's files (module docstring).

    One instance per episode, started before the episode's clock does (runner.py main), so a
    script pays only for joining it: each run_script is an exec into the instance, and every
    process the script left is killed when it is done (``sweep``)."""

    name = "apptainer"

    def __init__(self, image: Path) -> None:
        self.image = Path(image)
        self.instance: str | None = None

    def inside(self, workspace: Workspace, path: Path) -> str:
        return "/workspace/" + path.relative_to(workspace.root).as_posix()

    def isolation(self, workspace: Workspace) -> list[str]:
        """The instance's: what a script can reach is set here, once."""
        return [
            "--containall", "--cleanenv",
            "--net", "--network", "none",
            "--no-mount", "hostfs,bind-paths,cwd,sys,devpts",
            "--home", f"{workspace.root}:/workspace",
            # Read-only: connecting to the socket needs no write to its directory, and the
            # script cannot change the client.
            "--bind", f"{workspace.gateway_dir}:/rtact:ro",
        ]  # fmt: skip

    def start(self, workspace: Workspace) -> None:
        name = f"rtws-{uuid.uuid4().hex[:12]}"
        subprocess.run(
            ["apptainer", "instance", "start", *self.isolation(workspace), str(self.image), name],
            check=True,
            capture_output=True,
            env=_apptainer_env(),
            timeout=120,
        )
        self.instance = name

    def stop(self, workspace: Workspace) -> None:
        name, self.instance = self.instance, None
        if name is not None:
            subprocess.run(
                ["apptainer", "instance", "stop", "--force", name],
                check=False,
                capture_output=True,
                env=_apptainer_env(),
                timeout=60,
            )

    def _exec(self, *args: str) -> list[str]:
        # Exec rebuilds the image's environment, so the script's is given again each time.
        return [
            "apptainer", "exec", "--cleanenv", "--pwd", "/workspace",
            "--env", f"RTACT_SOCKET=/rtact/{SOCKET_NAME}",
            "--env", "PYTHONPATH=/rtact",
            "--env", "PATH=/rtact/bin:/usr/local/bin:/usr/bin:/bin",
            "--env", "LANG=C.UTF-8",
            f"instance://{self.instance}",
            *args,
        ]  # fmt: skip

    def command(self, workspace: Workspace, interpreter: str, script: Path):
        if self.instance is None:
            raise RuntimeError("the workspace instance is not running")
        argv = self._exec(interpreter, self.inside(workspace, script))
        return argv, _apptainer_env()

    def sweep(self, workspace: Workspace) -> None:
        """Kill every process in the instance but its init: kill(-1) spares the caller and the
        PID namespace's init, so what a script started, detached or not, ends with it."""
        if self.instance is not None:
            subprocess.run(
                self._exec("sh", "-c", "kill -9 -1 2>/dev/null; true"),
                check=False,
                capture_output=True,
                env=_apptainer_env(),
                timeout=30,
            )


class Workspace:
    """One episode's workspace. ``run`` blocks on the shell's script worker; ``kill`` is called
    from the shell's thread."""

    def __init__(self, root: Path, gateway_dir: Path, backend, clock) -> None:
        self.root = Path(root)
        self.gateway_dir = Path(gateway_dir)
        self.backend = backend
        self.clock = clock  # the episode's operational clock
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._killed: str | None = None
        self._screenshots = 0
        self.scripts = 0
        self._prepared = False
        self._closed = False

    def prepare(self) -> None:
        """Make the workspace and start its backend; once, before the episode's clock starts
        (runner.py main), or by the shell for a workspace given it unprepared."""
        if self._prepared:
            return
        self._prepared = True
        for directory in (self.root, self.root / "obs", self.root / ".scripts"):
            directory.mkdir(parents=True, exist_ok=True)
        # The client goes beside the gateway's socket, in the directory the script gets
        # read-only (/rtact), not in its workspace.
        client = self.gateway_dir
        (client / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rtact.__file__, client / "rtact.py")
        (client / "rtact.py").chmod(0o644)
        launcher = client / "bin" / "rtact"
        launcher.write_text(_BASH_RTACT)
        launcher.chmod(0o755)
        self.backend.start(self)

    def screenshot_path(self) -> tuple[Path, str]:
        """Where the next gateway screenshot goes: the host path, and the script's."""
        self._screenshots += 1
        path = self.root / "obs" / f"{self._screenshots:05d}.png"
        return path, self.backend.inside(self, path)

    def run(self, action: Action) -> ScriptResult:
        language = action.parameters["language"]
        self.scripts += 1
        stem = self.root / ".scripts" / f"{self.scripts:04d}"
        script = stem.with_suffix(_SUFFIX[language])
        script.write_text(action.parameters["code"])
        stdout, stderr = stem.with_suffix(".stdout"), stem.with_suffix(".stderr")
        interpreter = "python3" if language == "python" else "bash"
        started = self.clock.monotonic()
        try:
            argv, env = self.backend.command(self, interpreter, script)
            with stdout.open("wb") as out, stderr.open("wb") as err:
                with self._lock:
                    self._killed = None
                    self._process = subprocess.Popen(
                        argv,
                        cwd=self.root,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=err,
                        start_new_session=True,
                    )
                process = self._process
                exit_code = process.wait()
        except (OSError, RuntimeError) as exc:
            return ScriptResult(None, None, 0.0, "", "", 0, 0, error=str(exc))
        finally:
            with self._lock:
                self._process = None
        duration = self.clock.monotonic() - started
        # No process outlives its call, whatever it detached.
        self.backend.sweep(self)
        out_text, out_bytes = _read_capped(stdout)
        err_text, err_bytes = _read_capped(stderr)
        return ScriptResult(
            exit_code, self._killed, round(duration, 3), out_text, err_text, out_bytes, err_bytes
        )

    def close(self) -> None:
        """Stop the backend and remove the gateway's directory; the workspace stays with the
        run's records. Idempotent: the shell closes it, and main() again whatever happened."""
        if self._closed:
            return
        self._closed = True
        try:
            self.backend.stop(self)
        finally:
            shutil.rmtree(self.gateway_dir, ignore_errors=True)

    def kill(self, reason: str) -> None:
        """Stop the running script, its whole process group; the first reason stands."""
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return
            if self._killed is None:
                self._killed = reason
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def episode_workspace(run_dir: Path, clock) -> Workspace:
    """The workspace of one run's episode, on the apptainer backend: a run never has another
    (providers/preflight.py check_action_space). The gateway's socket lives in a short directory
    of its own: a unix socket path is limited to 107 bytes, and the script is given that
    directory alone."""
    from realtime_framework.sandbox.apptainer import image_path

    gateway_dir = Path(tempfile.mkdtemp(prefix="rtg-"))
    gateway_dir.chmod(0o755)
    backend = ApptainerBackend(image_path("workspace"))
    return Workspace(Path(run_dir) / "workspace", gateway_dir, backend, clock)
