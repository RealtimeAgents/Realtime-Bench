"""Desktop startup, browser launch and recording lifecycle."""

from __future__ import annotations
import os as os
import sys as sys
import time as time
import subprocess as subprocess
import logging as logging
import re as re
from pathlib import Path as Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from realtime_framework.entity.config import EnvironmentConfig
from realtime_framework.entity.config import SANDBOX_DISPLAY
import realtime_framework.sandbox.apptainer as apptainer
from realtime_framework.sandbox.apptainer import GRAPHICS_ENV
from realtime_framework.sandbox.apptainer import RECORDING_ENV
from realtime_framework.sandbox.apptainer import chrome_graphics_flags
from realtime_framework.sandbox.apptainer import graphics_mode
from realtime_framework.sandbox.apptainer import recording_codec_flags
from realtime_framework.sandbox.apptainer import recording_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Container name defaults from env so parallel lanes each use a distinct sandbox and port block
# (configured by the Harbor trial worker).
CONTAINER_NAME = os.environ.get("SANDBOX_CONTAINER", "streaming-agent-sandbox")


def stop_containers():
    """Release the desktop and task service owned by this episode."""
    failures = []
    for name in (CONTAINER_NAME, os.environ.get("RT_SUITE_CONTAINER", "rt-realtime-suite")):
        try:
            apptainer.stop_instance(name)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    if failures:
        raise RuntimeError("Could not stop episode containers: " + "; ".join(failures))


def _run_retry(cmd, *, timeout, retries=2, **kw):
    """Run a host or container command, retrying transient timeouts during startup."""
    last = None
    for attempt in range(retries + 1):
        try:
            return subprocess.run(cmd, timeout=timeout, **kw)
        except subprocess.TimeoutExpired as e:
            last = e
            logger.warning(
                "apptainer exec timed out (attempt %d/%d): %s",
                attempt + 1,
                retries + 1,
                " ".join(map(str, cmd[:6])),
            )
    raise last


def build_image_if_needed():
    return apptainer.build_image()


def start_container(env_config: EnvironmentConfig) -> str:
    environment = {
        "GUEST_AGENT_PORT": str(env_config.guest_agent_port),
        "VNC_PORT": str(env_config.vnc_port),
        "NOVNC_PORT": str(env_config.novnc_port),
        "DISPLAY": f":{env_config.display_number}",
        "TASK_PORT_OFFSET": os.environ.get("TASK_PORT_OFFSET", "0"),
        GRAPHICS_ENV: graphics_mode(),
        RECORDING_ENV: recording_encoder(),
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    return apptainer.start_instance(
        CONTAINER_NAME,
        apptainer.build_image(image=Path(env_config.vm_image)),
        sandbox=True,
        environment=environment,
    )


def open_local_url_in_container(url: str, debug_port: int):
    """Open a task URL in ordinary Chrome with a CDP inspection port."""
    _launch_chrome(url, debug_port=debug_port)
    time.sleep(8)
    logger.info(f"Opened local URL in Chrome (CDP {debug_port}): {url}")


CHROME_PROFILE = "/home/user/.config/google-chrome-agent"


def _launch_chrome(url: str, profile: str = CHROME_PROFILE, debug_port: int | None = None):
    """Launch a Chrome window, optionally with a CDP inspection port."""

    def dx(cmd, detached=False, timeout=30):
        full = apptainer.exec_command(CONTAINER_NAME, ["bash", "-c", cmd], detached=detached)
        return _run_retry(full, capture_output=True, text=True, timeout=timeout)

    dx(
        "pkill -9 xfce4-screensaver 2>/dev/null; "
        f"DISPLAY={SANDBOX_DISPLAY} xset s off s noblank 2>/dev/null; "
        "xfconf-query -c xfce4-screensaver -p /saver/enabled -s false -n -t bool 2>/dev/null; "
        "xfconf-query -c xfce4-screensaver -p /lock/enabled -s false -n -t bool 2>/dev/null; true"
    )
    dx(
        f"pkill -f '[u]ser-data-dir={profile}' 2>/dev/null; "
        + (
            f"pkill -f '[r]emote-debugging-port={int(debug_port)}' 2>/dev/null; "
            if debug_port is not None
            else ""
        )
        + "true"
    )
    dx(f"rm -rf {profile}; mkdir -p {profile}")
    debug = (
        f"--remote-debugging-port={int(debug_port)} --remote-allow-origins=* "
        if debug_port is not None
        else ""
    )
    dx(
        f"DISPLAY={SANDBOX_DISPLAY} google-chrome-stable --no-sandbox --no-first-run "
        f"--no-default-browser-check --disable-features=Translate --start-maximized "
        f"{chrome_graphics_flags(explicit_cpu=False)} "
        f"--user-data-dir={profile} {debug}'{url}'",
        detached=True,
    )
    if debug_port is not None:
        dx(
            f"for i in $(seq 1 50); do curl -sf http://127.0.0.1:{int(debug_port)}/json "
            f">/dev/null 2>&1 && exit 0; sleep .2; done; exit 1",
            timeout=15,
        )


def _start_screensaver_keepalive():
    """Keep the X display awake for the whole run.

    Xvfb has no DPMS and xfce's xfsettingsd re-applies a 300s screen-blank timer,
    so during idle stretches (e.g. when the model is rate-limited) the desktop
    blanks to BLACK — which the agent then captures and 'sees', getting stuck. A
    detached loop disables and resets the screensaver timer every 60s so the
    screen never blanks. Dies with the container (next run starts fresh)."""
    try:
        subprocess.run(
            apptainer.exec_command(
                CONTAINER_NAME,
                [
                    "bash",
                    "-c",
                    f"while true; do DISPLAY={SANDBOX_DISPLAY} xset s off s noblank s 0 0 2>/dev/null; "
                    f"DISPLAY={SANDBOX_DISPLAY} xset s reset 2>/dev/null; sleep 60; done",
                ],
                detached=True,
            ),
            check=True,
            timeout=10,
        )
        logger.info("Screensaver keepalive started (prevents idle screen-blank)")
    except Exception as e:
        logger.warning(f"could not start screensaver keepalive: {e}")


def _start_screen_recording(run_dir=None):
    """Start a real screen recording of the lane-local desktop inside the container.

    ffmpeg must run in the container because the host can't reach the container's
    X display. Detached; finalized later by _stop_screen_recording(). Captures the
    true session (cursor, animations, windows) — unlike the frame-composited video.
    """
    try:
        launched_at = time.time()
        subprocess.run(
            apptainer.exec_command(
                CONTAINER_NAME,
                [
                    "bash",
                    "-c",
                    "ffmpeg -y -f x11grab -draw_mouse 1 -video_size 1920x1080 -framerate 12 "
                    f"-i {SANDBOX_DISPLAY} {recording_codec_flags()} "
                    "-pix_fmt yuv420p /tmp/recording.mp4 "
                    "> /tmp/ffmpeg.log 2>&1",
                ],
                detached=True,
            ),
            check=True,
            timeout=10,
        )
        # Wait until the X11 input has opened, before a time-sensitive task reset.
        # This is setup time and must never consume the task's scored horizon.
        ready = subprocess.run(
            apptainer.exec_command(
                CONTAINER_NAME,
                [
                    "bash",
                    "-c",
                    "for i in {1..50}; do grep -q 'Duration: N/A, start:' /tmp/ffmpeg.log && exit 0; "
                    "sleep 0.1; done; exit 1",
                ],
            ),
            capture_output=True,
            timeout=10,
        )
        if ready.returncode:
            logger.warning("Screen recorder did not report its first input timestamp")
        logger.info(
            "Screen recording started (ffmpeg x11grab %s, %s encoder -> /tmp/recording.mp4)",
            SANDBOX_DISPLAY,
            recording_encoder(),
        )
        if run_dir is not None:
            from realtime_framework.evaluation.reporting import atomic_json

            atomic_json(
                Path(run_dir) / "recording-timing.json",
                {
                    "launch_unix_s": launched_at,
                    "kind": "x11grab",
                    "fps": 12,
                },
            )
        return True
    except Exception as e:
        logger.warning(f"Could not start screen recording (is ffmpeg in the image?): {e}")
        return False


def _stop_screen_recording(run_dir: Path, frames_dir: Path, fps: float):
    """Stop the recording (SIGINT so the MP4 is finalized) and copy it out as
    video.mp4. Falls back to compositing the frames captured once per decision if the
    recording is missing or empty. ``fps`` is the model-input rate, a ceiling on the decision
    rate: the composite runs faster than the episode wherever decisions were further apart."""
    out = run_dir / "video.mp4"
    try:
        subprocess.run(
            apptainer.exec_command(CONTAINER_NAME, ["pkill", "-INT", "-f", "ffmpeg.*x11grab"]),
            capture_output=True,
            timeout=10,
        )
        # Wait for ffmpeg to flush the moov atom and exit.
        for _ in range(20):
            r = subprocess.run(
                apptainer.exec_command(CONTAINER_NAME, ["pgrep", "-f", "ffmpeg.*x11grab"]),
                capture_output=True,
                timeout=10,
            )
            if r.returncode != 0:
                break
            time.sleep(0.3)
        size = subprocess.run(
            apptainer.exec_command(
                CONTAINER_NAME,
                ["bash", "-c", "stat -c %s /tmp/recording.mp4 2>/dev/null || echo 0"],
            ),
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if size.isdigit() and int(size) > 1024:
            apptainer.copy_out(CONTAINER_NAME, "/tmp/recording.mp4", out)
            # X11's first input PTS is the recording origin; process-launch time is not.
            try:
                from realtime_framework.evaluation.reporting import atomic_json
                from realtime_framework.evaluation.run_records import read_json

                apptainer.copy_out(
                    CONTAINER_NAME, "/tmp/ffmpeg.log", run_dir / "recording-ffmpeg.log"
                )
                text = (run_dir / "recording-ffmpeg.log").read_text(errors="replace")
                match = re.search(r"Duration: N/A, start: ([0-9.]+)", text)
                metadata = read_json(run_dir / "recording-timing.json")
                if match and abs(float(match.group(1)) - metadata.get("launch_unix_s", 0)) < 30:
                    metadata["first_frame_unix_s"] = float(match.group(1))
                    metadata["saved_unix_s"] = time.time()
                    atomic_json(run_dir / "recording-timing.json", metadata)
            except Exception as exc:
                logger.warning("Could not retain recording clock: %s", exc)
            logger.info(f"Screen recording saved: {out} ({int(size) // 1024} KB)")
            return
        logger.warning("Screen recording empty/missing — falling back to frame-composited video")
    except Exception as e:
        logger.warning(f"Stop/copy recording failed ({e}) — falling back to frame-composited video")
    make_video(frames_dir, out, fps=fps)


def make_video(frames_dir: Path, output_path: Path, fps: float):
    """Assemble the continuously-captured frames into a REAL-TIME video:
    captured at `fps` across the session and played back at `fps`, so the
    video duration equals the run's wall-clock duration."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=1920:1080",
                str(output_path),
            ],
            capture_output=True,
            timeout=300,
        )
        logger.info(f"Real-time video saved: {output_path}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("ffmpeg not available, skipping video")
