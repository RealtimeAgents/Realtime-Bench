from __future__ import annotations

# Acceleration
"""Validated sandbox graphics and screen-recording acceleration settings."""

import os as os
import shlex as shlex


GRAPHICS_ENV = "SANDBOX_GRAPHICS_MODE"
RECORDING_ENV = "SCREEN_RECORDING_ENCODER"
_MODES = frozenset({"cpu", "gpu"})


def _mode(name: str, default: str = "cpu") -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in _MODES:
        raise ValueError(f"{name} must be cpu or gpu, got {value!r}")
    return value


def graphics_mode() -> str:
    """Return ``cpu`` (SwiftShader) or ``gpu`` (NVIDIA Vulkan through ANGLE)."""
    return _mode(GRAPHICS_ENV)


def recording_encoder() -> str:
    """Return ``cpu`` (libx264) or ``gpu`` (NVENC)."""
    return _mode(RECORDING_ENV)


def sandbox_needs_gpu() -> bool:
    """Whether the desktop instance needs host NVIDIA devices and driver libraries."""
    return graphics_mode() == "gpu" or recording_encoder() == "gpu"


def chrome_graphics_flags(*, explicit_cpu: bool = True) -> str:
    """Return shell-safe Chrome flags for the selected renderer.

    GPU mode disables the software rasterizer so a missing/inaccessible device fails visibly
    instead of silently producing results under a different renderer.
    """
    if graphics_mode() == "cpu":
        if not explicit_cpu:
            # Ordinary web tasks keep Chrome's default renderer. Browser-game callers request
            # the explicit SwiftShader baseline because they require WebGL.
            return ""
        flags = (
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--ignore-gpu-blocklist",
        )
    else:
        flags = (
            "--use-gl=angle",
            "--use-angle=vulkan",
            "--enable-features=Vulkan",
            "--ignore-gpu-blocklist",
            "--enable-gpu-rasterization",
            "--disable-software-rasterizer",
        )
    return " ".join(shlex.quote(flag) for flag in flags)


def recording_codec_flags() -> str:
    """Return shell-safe ffmpeg encoder arguments for the selected recording device."""
    flags = (
        ("-c:v", "libx264", "-preset", "ultrafast")
        if recording_encoder() == "cpu"
        else ("-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll")
    )
    return " ".join(shlex.quote(flag) for flag in flags)


# Apptainer
"""Apptainer image builds, isolated instance filesystems, and command/file transport.

Services use host networking. The benchmark launcher assigns every lane a disjoint port block,
and each sandbox holds a lane-local lifetime lock for that block.
"""

import argparse as argparse
import fcntl as fcntl
import hashlib as hashlib
import json as json
import logging as logging
from pathlib import Path as Path
import platform as platform
import re as re
import shutil as shutil
import subprocess as subprocess
import sys as sys
import tempfile as tempfile
import time as time

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / ".cache" / "apptainer"
STATE_DIR = ROOT / "temp" / "apptainer" / platform.node()
# Google publishes no arm64 Chrome; sandbox.def pins the amd64 base image and Chrome package.
ARM64_SANDBOX_BASE = (
    "debian:bookworm@sha256:2c037a04925515fdd6ea85ea14a682d0e79931f5e9f5d07b6dbfc6ba12f9e858"
)
# Optional mirror for the pinned Chrome .deb once Google drops it from its package pool.
CHROME_MIRROR_ENV = "RTBENCH_CHROME_DEB_URL"


def require_runtime() -> None:
    if shutil.which("apptainer") is None:
        raise RuntimeError(
            "Apptainer is unavailable. Load your cluster Apptainer module or install Apptainer, then retry."
        )


def image_path(kind: str = "sandbox") -> Path:
    override = os.environ.get("SANDBOX_IMAGE" if kind == "sandbox" else "RT_SUITE_IMAGE")
    return Path(override).expanduser().resolve() if override else IMAGE_DIR / f"{kind}.sif"


def source_fingerprint(kind: str) -> str:
    """Fingerprint the sources embedded in one image, independently of any cached SIF."""
    definition = ROOT / "realtime_framework/sandbox/images" / f"{kind}.def"
    sources = [definition]
    directories = (
        [
            ROOT / "realtime_framework/sandbox/guest_server",
            ROOT / "realtime_framework/sandbox/images" / "scripts",
        ]
        if kind == "sandbox"
        else [ROOT / "realtime_tasks"]
    )
    if kind == "realtime":
        sources.append(ROOT / "realtime_framework/sandbox/images/realtime-requirements.txt")
    for directory in directories:
        sources.extend(
            p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
    digest = hashlib.sha256(platform.machine().encode())
    for source in sorted(sources):
        digest.update(str(source.relative_to(ROOT)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


_APT_UNITS = {"B": 1, "kB": 1e3, "MB": 1e6, "GB": 1e9}
_APT_SIZE = r"([\d.]+) (B|kB|MB|GB)"


def _apt_bytes(value: str, unit: str) -> float:
    return float(value) * _APT_UNITS[unit]


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


class _BuildProgress:
    """Estimate image-build progress from apptainer/apt/pip output.

    apt downloads dominate a cold sandbox build, so the bar tracks downloaded bytes against apt's
    ``Need to get`` total, then dpkg unpack/configure steps against ``newly installed``.
    """

    def __init__(self, kind: str):
        self.kind = kind
        self.start = time.monotonic()
        self.phase = "starting"
        self.done = self.total = 0.0
        self.unit = ""
        self.phase_start = self.start
        self.downloading = False
        self.installs = 0

    def _enter(self, phase: str, total: float = 0.0, unit: str = "") -> None:
        self.phase, self.total, self.unit = phase, total, unit
        self.done = 0.0
        self.phase_start = time.monotonic()

    def feed(self, line: str) -> None:
        if line.startswith("INFO:    Fetching OCI image"):
            self._enter("fetching base image")
        elif line.startswith("INFO:    Running post scriptlet"):
            self._enter("running build script")
        elif line.startswith("INFO:    Creating SIF file"):
            self._enter("writing SIF file")
        elif match := re.match(r"Need to get " + _APT_SIZE, line):
            self._enter("downloading packages", _apt_bytes(*match.groups()), "bytes")
            self.downloading = True
        elif self.downloading and (match := re.match(r"Get:\d+ .*\[" + _APT_SIZE + r"\]$", line)):
            self.done = min(self.total, self.done + _apt_bytes(*match.groups()))
        elif self.downloading and line.startswith("Fetched "):
            # dpkg unpacks every package, then configures each one: two steps per package.
            self.downloading = False
            self._enter("installing packages", 2 * self.installs, "dpkg steps")
        elif match := re.match(r"\d+ upgraded, (\d+) newly installed", line):
            self.installs = int(match.group(1))
        elif line.startswith(("Unpacking ", "Setting up ")) and self.phase == "installing packages":
            self.done = min(self.total, self.done + 1)
        elif line.startswith("Collecting "):
            if self.phase != "installing Python packages":
                self._enter("installing Python packages")
            self.done += 1

    def render(self, width: int = 24) -> str:
        now = time.monotonic()
        text = f"[{self.kind} image] {self.phase}"
        if self.total:
            fraction = self.done / self.total
            filled = int(fraction * width)
            text += f" [{'#' * filled}{'.' * (width - filled)}] {fraction:4.0%}"
            if self.unit == "bytes":
                text += f" {self.done / 1e6:.0f}/{self.total / 1e6:.0f} MB"
            else:
                text += f" {self.done:.0f}/{self.total:.0f} {self.unit}"
            elapsed = now - self.phase_start
            if 0 < fraction < 1 and elapsed > 5:
                if self.unit == "bytes":
                    text += f" {self.done / elapsed / 1e3:.0f} kB/s"
                text += f" ETA {_duration(elapsed * (1 - fraction) / fraction)}"
        elif self.done:
            text += f" ({self.done:.0f})"
        return f"{text} | elapsed {_duration(now - self.start)}"


def _run_build(command: list[str], *, kind: str, env: dict[str, str]) -> None:
    """Run ``apptainer build`` with its output in a log file and a progress line on stderr."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = IMAGE_DIR / f"{kind}.build.log"
    progress = _BuildProgress(kind)
    interactive = sys.stderr.isatty()
    print(
        f"Building {kind} image (first build may take 30+ minutes); log: {log_path}",
        file=sys.stderr,
        flush=True,
    )
    last_render = 0.0
    with log_path.open("w") as log, subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            progress.feed(line.rstrip("\n"))
            now = time.monotonic()
            # A terminal gets a live bar; logs get a status line every 30 seconds.
            if interactive and now - last_render >= 0.5:
                print(f"\r\033[K{progress.render()}", end="", file=sys.stderr, flush=True)
                last_render = now
            elif not interactive and now - last_render >= 30:
                print(progress.render(), file=sys.stderr, flush=True)
                last_render = now
        returncode = process.wait()
    if interactive:
        print("\r\033[K", end="", file=sys.stderr, flush=True)
    if returncode:
        tail = log_path.read_text(errors="replace").splitlines()[-40:]
        print("\n".join(tail), file=sys.stderr, flush=True)
        raise subprocess.CalledProcessError(returncode, command)
    elapsed = _duration(time.monotonic() - progress.start)
    print(f"[{kind} image] built in {elapsed}", file=sys.stderr, flush=True)


def build_image(kind: str = "sandbox", image: Path | None = None) -> Path:
    """Build atomically under a process-shared lock; invalidate on source changes."""
    require_runtime()
    image = image or image_path(kind)
    image = (image if image.is_absolute() else ROOT / image).resolve()
    definition = ROOT / "realtime_framework/sandbox/images" / f"{kind}.def"
    fingerprint = source_fingerprint(kind)
    image.parent.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = image.with_suffix(".sources.sha256")
    with image.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Explicit images can be built on another machine and copied to this host, but a
        # sandbox must carry the pinned-build labels so its browser version is on record.
        if (
            image.is_file()
            and not stamp.exists()
            and os.environ.get("SANDBOX_IMAGE" if kind == "sandbox" else "RT_SUITE_IMAGE")
        ):
            if kind == "sandbox" and "browser" not in image_provenance(image):
                raise RuntimeError(
                    f"{image} has no {PROVENANCE_LABEL_PREFIX}browser label; rebuild it from "
                    "realtime_framework/sandbox/images/sandbox.def"
                )
            return image
        if image.is_file() and stamp.is_file() and stamp.read_text() == fingerprint:
            return image
        with tempfile.TemporaryDirectory(
            prefix="build-", dir=STATE_DIR, ignore_cleanup_errors=True
        ) as work:
            output = Path(work) / "image.sif"
            args = ["apptainer", "build", "--fakeroot"]
            if kind == "sandbox" and platform.machine() in {"aarch64", "arm64"}:
                args += ["--build-arg", f"BASE_IMAGE={ARM64_SANDBOX_BASE}"]
            if kind == "sandbox" and os.environ.get(CHROME_MIRROR_ENV):
                # The pinned SHA-256 still applies, so a mirror cannot change the browser.
                args += ["--build-arg", f"CHROME_DEB_URL={os.environ[CHROME_MIRROR_ENV]}"]
            build_env = dict(os.environ)
            build_env.setdefault("APPTAINER_TMPDIR", work)
            build_env.setdefault("APPTAINER_CACHEDIR", str(IMAGE_DIR / "cache"))
            _run_build([*args, str(output), str(definition)], kind=kind, env=build_env)
            # STATE_DIR and image may be on different filesystems.
            staged = image.with_suffix(".sif.partial")
            try:
                shutil.copyfile(output, staged)
                staged.replace(image)
            finally:
                staged.unlink(missing_ok=True)
            # Publish provenance with the image, before best-effort cleanup on network storage.
            stamp.write_text(fingerprint)
        if Path(work).exists():
            logging.warning("Image built successfully; temporary build files remain at %s", work)
    return image


def _state(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"Invalid Apptainer instance name: {name!r}")
    return STATE_DIR / name


def is_running(name: str) -> bool:
    result = subprocess.run(
        ["apptainer", "instance", "list", "--json", name],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return any(row["instance"] == name for row in json.loads(result.stdout).get("instances", []))


def exec_command(name: str, args: list[str], *, detached: bool = False) -> list[str]:
    """Run as the invoking host UID; all mutable task paths belong to that UID."""
    from realtime_framework.entity.config import SANDBOX_DISPLAY

    _state(name)
    if detached:
        args = ["sh", "-c", "nohup " + shlex.join(args) + " </dev/null >/dev/null 2>&1 &"]
    # Instance-start --env values apply to the startscript but Apptainer exec reconstructs the
    # image environment. Re-assert the lane display for every later GUI/task command.
    return [
        "apptainer",
        "exec",
        "--env",
        f"DISPLAY={SANDBOX_DISPLAY}",
        "--pwd",
        "/home/user",
        f"instance://{name}",
        *args,
    ]


def stop_instance(name: str) -> None:
    require_runtime()
    state = _state(name)
    if is_running(name):
        subprocess.run(
            ["apptainer", "instance", "stop", "--timeout", "10", name], check=True, timeout=30
        )
    if state.exists():
        shutil.rmtree(state)


def start_instance(
    name: str,
    image: Path,
    *,
    sandbox: bool = False,
    environment: dict[str, str] | None = None,
    gpu: bool | None = None,
) -> str:
    require_runtime()
    state = _state(name)
    stop_instance(name)
    state.mkdir(parents=True)
    for directory in ("home", "tmp", "run"):
        (state / directory).mkdir()
    args = [
        "apptainer",
        "instance",
        "start",
        "--containall",
        "--cleanenv",
        "--home",
        f"{state / 'home'}:/home/user",
        "--bind",
        f"{state / 'tmp'}:/tmp",
        "--bind",
        f"{state / 'run'}:/run",
        "--no-mount",
        "hostfs,bind-paths,cwd",
    ]
    if gpu is None:
        gpu = sandbox and sandbox_needs_gpu()
    if gpu:
        # Bind the host NVIDIA devices and matching driver libraries into this instance.  The
        # model itself remains a host process; this only accelerates the isolated desktop.
        args.insert(3, "--nv")
    if sandbox:
        # The startscript keeps this advisory lock for the complete instance lifetime. Derive its
        # name from host-visible lane resources so aliases cannot accidentally reuse one lane.
        lane_resources = {
            key: (environment or {}).get(key)
            for key in (
                "GUEST_AGENT_PORT",
                "VNC_PORT",
                "NOVNC_PORT",
                "DISPLAY",
                "TASK_PORT_OFFSET",
            )
        }
        lane_key = hashlib.sha256(json.dumps(lane_resources, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        network_lock = STATE_DIR / f"sandbox-network-{lane_key}.lock"
        network_lock.touch(exist_ok=True)
        args += ["--bind", f"{network_lock}:/opt/sandbox-network.lock"]
        subprocess.run(
            [
                "apptainer",
                "exec",
                "--containall",
                "--cleanenv",
                "--home",
                f"{state / 'home'}:/home/user",
                "--pwd",
                "/home/user",
                "--no-mount",
                "hostfs,bind-paths,cwd",
                str(image),
                "cp",
                "-r",
                "/opt/sandbox-home/.",
                "/home/user/",
            ],
            check=True,
            timeout=60,
        )
    for key, value in (environment or {}).items():
        args += ["--env", f"{key}={value}"]
    try:
        subprocess.run([*args, str(image), name], check=True, timeout=60)
        if sandbox:
            for _ in range(60):
                if not is_running(name):
                    raise RuntimeError(
                        "Sandbox exited during startup; inspect Apptainer instance logs."
                    )
                ready = subprocess.run(
                    exec_command(name, ["test", "-f", "/tmp/sandbox-ready"]),
                    capture_output=True,
                    timeout=10,
                )
                if ready.returncode == 0:
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("Sandbox did not become ready; inspect Apptainer instance logs.")
    except BaseException:
        stop_instance(name)
        raise
    return name


def copy_into(
    name: str, source: str | Path, destination: str, *, timeout: int = 60
) -> subprocess.CompletedProcess:
    """Stage through the instance's private /tmp; no host home or repo bind is needed."""
    state = _state(name)
    with tempfile.TemporaryDirectory(prefix="transfer-", dir=state / "tmp") as work:
        target = Path(work) / "payload"
        source = Path(source)
        if source.is_dir():
            shutil.copytree(source, target)
            inside = f"/tmp/{Path(work).name}/payload/."
        else:
            shutil.copy2(source, target)
            inside = f"/tmp/{Path(work).name}/payload"
        return subprocess.run(
            exec_command(name, ["cp", "-r", inside, destination]),
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def copy_out(name: str, source: str, destination: str | Path) -> None:
    with open(destination, "wb") as output:
        subprocess.run(exec_command(name, ["cat", source]), stdout=output, check=True, timeout=60)


PROVENANCE_LABEL_PREFIX = "org.rtbench."


def image_provenance(image: Path) -> dict:
    """Build-time labels (browser, OS, base digest, package snapshot) recorded in a SIF."""
    result = subprocess.run(
        ["apptainer", "inspect", "--labels", "--json", str(image)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    labels = json.loads(result.stdout).get("data", {}).get("attributes", {}).get("labels") or {}
    return {
        key.removeprefix(PROVENANCE_LABEL_PREFIX): value
        for key, value in sorted(labels.items())
        if key.startswith(PROVENANCE_LABEL_PREFIX)
    }


def runtime_identity() -> dict:
    """Content identities for SIFs, cached against file metadata between lane checks."""
    identities, provenance = {}, {}
    for kind in ("sandbox", "realtime"):
        path = image_path(kind)
        if not path.is_file():
            identities[kind] = provenance[kind] = None
            continue
        stat = path.stat()
        signature = [
            str(path.resolve()),
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ]
        cache_dir = IMAGE_DIR / "identities"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / (hashlib.sha256(str(path).encode()).hexdigest() + ".json")
        try:
            record = json.loads(cache.read_text())
        except (OSError, ValueError):
            record = {}
        if record.get("signature") != signature or "provenance" not in record:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            record = {
                "signature": signature,
                "sha256": digest.hexdigest(),
                "provenance": image_provenance(path),
            }
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            # Keep the temporary file on the destination filesystem for atomic replacement.
            with tempfile.NamedTemporaryFile("w", dir=cache_dir, delete=False) as output:
                json.dump(record, output)
            Path(output.name).replace(cache)
        identities[kind] = record["sha256"]
        provenance[kind] = record["provenance"]
    return {
        "runtime": "apptainer",
        "network": "host",
        "images": identities,
        "image_provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "stop", "start-realtime"])
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    if args.action == "build":
        for kind in args.names or ["sandbox", "realtime"]:
            if kind not in {"sandbox", "realtime"}:
                parser.error("build expects sandbox or realtime")
            print(build_image(kind))
    elif args.action == "stop":
        for name in args.names:
            stop_instance(name)
    else:
        name = (
            args.names[0]
            if args.names
            else os.environ.get("RT_SUITE_CONTAINER", "rt-realtime-suite")
        )
        start_instance(
            name,
            build_image("realtime"),
            environment={"RT_SUITE_PORT": os.environ.get("RT_SUITE_PORT", "5091")},
        )


# Sandbox
"""Lazy sandbox config aliases for task runtimes."""


def __getattr__(name):
    # Load task-runtime aliases without importing Pydantic until needed.
    if name in {"SANDBOX_CONTAINER", "SANDBOX_DISPLAY"}:
        from realtime_framework.entity import config

        return getattr(config, name)
    raise AttributeError(name)


# Base
"""Container execution and service-readiness helpers for tasks."""


def _dexec(
    script: str, timeout: int = 120, input: str = None, detached: bool = False
) -> subprocess.CompletedProcess:
    """Run inside the instance as the host UID."""
    from realtime_framework.entity.config import SANDBOX_CONTAINER

    cmd = exec_command(SANDBOX_CONTAINER, ["bash", "-c", script], detached=detached)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=input)


def _wait_http(port: int, path: str = "/", tries: int = 30):
    _dexec(
        f"for i in $(seq 1 {tries}); do curl -sf localhost:{port}{path} >/dev/null 2>&1 "
        f"&& exit 0; sleep 0.5; done; exit 1"
    )

if __name__ == "__main__":
    main()
