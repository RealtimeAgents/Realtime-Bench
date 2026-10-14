from __future__ import annotations

# Realtime
"""Lifecycle and scoring bridge for the realtime SaaS simulation tasks.

The app runs in one sibling container.  Every task provision starts a fresh container,
resets one server-side episode, and stores its opaque episode id.  Actions taken in the
browser hit the server-side API; verify only reads the evaluator endpoint and never mutates
task state.
"""

import os as os
from pathlib import Path as Path
import time as time

import requests as requests

import realtime_framework.sandbox.apptainer as apptainer


# Container name + host port come from env so parallel lanes each run their own realtime service
# (the runner assigns per-lane values).
CONTAINER = os.environ.get("RT_SUITE_CONTAINER", "rt-realtime-suite")
HOST_PORT = int(os.environ.get("RT_SUITE_PORT", "5091"))
HOST_APP = f"http://localhost:{HOST_PORT}"
AGENT_APP = HOST_APP


def _ensure_image() -> str:
    return str(apptainer.build_image("realtime"))


def boot() -> bool:
    image = _ensure_image()
    apptainer.start_instance(CONTAINER, Path(image), environment={"RT_SUITE_PORT": str(HOST_PORT)})
    for _ in range(60):
        try:
            if requests.get(f"{HOST_APP}/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def reset(scenario: str, seed: int = 42) -> dict:
    """Open a fresh episode and return its full descriptor.

    The response carries ``episode``, ``generation`` and the per-episode ``control_token`` that
    authorizes the harness-only ``/api/restart`` re-arm. The token is surfaced ONLY here (the harness
    calls this from the host) and is never rendered into the page or exposed by ``/state``, so a model
    driving the browser cannot forge a restart to wipe its own failed attempts.
    """
    response = requests.post(f"{HOST_APP}/api/reset/{scenario}", json={"seed": seed}, timeout=8)
    response.raise_for_status()
    return response.json()


def restart(
    scenario: str, episode: str, control_token: str | None = None, hold: bool = False
) -> str:
    """Restart an already-open episode's clock/state while preserving its browser URL id.

    ``control_token`` is the per-episode authorization returned by :func:`reset`; the app rejects a
    restart without it (403), which is what stops a model from re-arming the clock itself.
    """
    payload = {"episode": episode, "hold": hold}
    if control_token is not None:
        payload["control_token"] = control_token
    if scenario.startswith("rt_"):
        generation = state(scenario, episode).get("generation")
        if type(generation) is not int:
            raise RuntimeError("Could not read the current realtime generation before restart")
        payload["generation"] = generation
    response = requests.post(
        f"{HOST_APP}/api/restart/{scenario}",
        json=payload,
        timeout=8,
    )
    response.raise_for_status()
    returned = response.json()["episode"]
    if returned != episode:
        raise RuntimeError(f"realtime restart changed episode id: {episode!r} -> {returned!r}")
    return returned


def start(scenario: str, episode: str, control_token: str) -> dict:
    """Start the world clock of an episode restarted with ``hold=True``."""
    generation = state(scenario, episode).get("generation")
    response = requests.post(
        f"{HOST_APP}/api/start/{scenario}",
        json={"episode": episode, "control_token": control_token, "generation": generation},
        timeout=8,
    )
    response.raise_for_status()
    return response.json()


def configure_clock(scenario: str, episode: str, control_token: str, fps: float) -> dict:
    response = requests.post(
        f"{HOST_APP}/api/clock/{scenario}/configure",
        json={"episode": episode, "control_token": control_token, "fps": fps},
        timeout=8,
    )
    response.raise_for_status()
    return response.json()


def advance_clock(scenario: str, episode: str, control_token: str, frames: float = 1) -> dict:
    response = requests.post(
        f"{HOST_APP}/api/clock/{scenario}/advance",
        json={"episode": episode, "control_token": control_token, "frames": frames},
        timeout=8,
    )
    response.raise_for_status()
    return response.json()


def page_url(scenario: str, episode: str) -> str:
    return f"{AGENT_APP}/{scenario}?episode={episode}"


def state(scenario: str, episode: str) -> dict:
    try:
        response = requests.get(
            f"{HOST_APP}/api/{scenario}/state",
            params={"episode": episode},
            timeout=8,
        )
        if response.status_code == 200:
            return response.json()
        return {"ok": False, "error": f"state returned HTTP {response.status_code}"}
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


def evaluation(scenario: str, episode: str, generation: int | None = None) -> dict:
    try:
        params = {"episode": episode}
        if generation is not None:
            params["generation"] = generation
        response = requests.get(
            f"{HOST_APP}/api/evaluation/{scenario}",
            params=params,
            timeout=8,
        )
        if response.status_code == 200:
            return response.json()
        return {"resolved": False, "error": f"evaluation returned HTTP {response.status_code}"}
    except requests.RequestException as exc:
        return {"resolved": False, "error": str(exc)}


# Tasks
"""Server-scored realtime browser interaction tasks.

Each task uses the shared sibling service in ``realtime_tasks.web.app``.  The service owns the clock and
world state while the browser continuously renders it, which makes stale screenshots and delayed
actions materially different from acting on the live page.

Each definition (task_id, scenario, reviewer description) lives in the ``[metadata]`` table of
``realtime_tasks/web/<task_id>/task.toml`` and the model prompt in its declared instruction file;
this module is a thin loader that wires each definition to the shared provision/verify logic.
"""

import logging as logging
import os as os
import sys as sys

from realtime_tasks.types import Task
import realtime_tasks.web.runtime as suite
from realtime_framework.sandbox.browser.clock import captured_frame_has_render_fence
from realtime_framework.sandbox.browser.clock import cdp_realtime_action_barrier_snapshot
from realtime_framework.sandbox.browser.clock import cdp_sync_realtime_frame
from realtime_framework.sandbox.browser.clock import cdp_wait_realtime_action_barrier
from realtime_framework.sandbox.actions import isolated_task_port

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from realtime_tasks.registry import TASK_CONFIG, load_definition, web_definitions

_TASK_DIR = os.path.join(_ROOT, "realtime_tasks", "web")

import json as json


class _RealtimeSuiteTask(Task):
    category = "realtime"
    # Ordinary task Chrome; every benchmark lane applies its task-port offset.
    dbg_port = isolated_task_port(9234)
    source_id = "local stateful realtime SaaS simulation"

    def __init__(self, task_id: str, scenario: str, prompt: str):
        self.task_id = task_id
        self.scenario = scenario
        self.prompt = prompt
        self._episode = None
        self._control_token = None

    def description(self):
        return self.prompt

    def provision(self):
        if not boot():
            raise RuntimeError("The realtime task service did not become ready")
        opened = reset(self.scenario, self.seed)
        if opened.get("seed") != self.seed:
            raise RuntimeError(f"web app applied seed {opened.get('seed')!r}, expected {self.seed}")
        self.applied_seed = opened["seed"]
        self._episode = opened["episode"]
        # Harness-only restart authorization (see reset). Held here so reset_episode()
        # can re-arm the clock; never surfaced to the model.
        self._control_token = opened.get("control_token")
        return page_url(self.scenario, self._episode)

    def reset_episode(self):
        """Restart a held episode immediately before the agent's first tick.

        provision() must open the page before model loading, but these tasks expose their decisive
        event after only 3–8 seconds. Restarting in-place after agent.initialize prevents model-load
        time from consuming the action window; the opaque episode id stays unchanged, so the already
        open browser automatically observes the fresh state on its next poll. World time stays at
        zero until start_episode() (wall clock) or the token clock is configured.
        """
        if not self._episode:
            raise RuntimeError("task has not been provisioned")
        restart(self.scenario, self._episode, self._control_token, hold=True)
        return {"episode": self._episode, "scenario": self.scenario}

    def start_episode(self):
        if (getattr(self, "_clock_config", {}) or {}).get("mode") == "token":
            return super().start_episode()
        if not self._episode:
            raise RuntimeError("task has not been provisioned")
        origin = time.time()
        started = start(self.scenario, self._episode, self._control_token)
        if started.get("already_started"):
            raise RuntimeError("realtime episode was already running before start_episode()")
        return {"clock_origin": origin, "start_hold": True}

    def _sync_browser_frame(self, benchmark_frame: float) -> dict:
        """Poll/render the exact server frame, then wait for a compositor commit."""
        last_error = None
        for _ in range(30):
            try:
                return cdp_sync_realtime_frame(
                    self.dbg_port,
                    benchmark_frame,
                    f"/{self.scenario}?episode={self._episode}",
                )
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.1)
        raise RuntimeError(
            f"realtime browser did not render benchmark frame {benchmark_frame}: {last_error}"
        )

    def begin_gui_action(self) -> dict:
        """Latch browser counters before the desktop input is dispatched."""
        return cdp_realtime_action_barrier_snapshot(
            self.dbg_port,
            f"/{self.scenario}?episode={self._episode}",
        )

    def complete_gui_action(self, snapshot: dict) -> dict:
        """Keep logical time frozen until any business request caused by the input completes."""
        return cdp_wait_realtime_action_barrier(
            self.dbg_port,
            f"/{self.scenario}?episode={self._episode}",
            snapshot,
        )

    def activate_clock(self):
        snapshot = super().activate_clock()
        if snapshot["mode"] == "token":
            configured = configure_clock(
                self.scenario,
                self._episode,
                self._control_token,
                snapshot["fps"],
            )
            self._last_clock_advance = {
                **configured,
                "browser": self._sync_browser_frame(configured["frame"]),
            }
        return self.clock_snapshot()

    def advance_clock(self, frames: float = 1):
        before = self.clock_snapshot()
        if before["mode"] == "token":
            advanced = advance_clock(
                self.scenario,
                self._episode,
                self._control_token,
                frames,
            )
            self._last_clock_advance = {
                **{key: value for key, value in advanced.items() if key != "state"},
                "browser": self._sync_browser_frame(advanced["frame"]),
            }
        return super().advance_clock(frames)

    def render_current_frame(self):
        """Poll and commit action-updated UI at the current benchmark frame."""
        snapshot = self.clock_snapshot()
        if snapshot["mode"] == "token":
            self._last_clock_advance = {
                **snapshot,
                "render_only": True,
                "browser": self._sync_browser_frame(snapshot["frame"]),
            }
        return self.clock_snapshot()

    def captured_frame_is_current(self, screenshot: bytes) -> bool:
        """Require the desktop capture to contain the latest realtime render fence."""
        browser = (getattr(self, "_last_clock_advance", {}) or {}).get("browser") or {}
        return captured_frame_has_render_fence(screenshot, browser.get("render_fence"))

    def evaluation_snapshot(self):
        if not self._episode:
            raise RuntimeError("task has not been provisioned")
        state = suite.state(self.scenario, self._episode)
        return {
            "kind": "realtime",
            "scenario": self.scenario,
            "episode": self._episode,
            "state": state,
            "evaluation": evaluation(
                self.scenario,
                self._episode,
                state.get("generation"),
            ),
        }

    def episode_over(self):
        """Return the server-owned reason for a completed or irreversibly failed task."""
        if not self._episode:
            return False
        state = suite.state(self.scenario, self._episode)
        if state.get("ok") is False:
            raise RuntimeError(f"realtime state backend unavailable: {state.get('error')}")
        if state.get("resolved") is True:
            return "task_success"
        return state.get("terminal_reason") or False

    def evaluate_snapshot(self, snapshot):
        result = dict(snapshot.get("evaluation") or {})
        result.setdefault("resolved", False)
        result["evaluation_version"] = 4
        result["evaluated_from_snapshot"] = True
        return result

    def verify(self):
        return self.evaluate_snapshot(self.evaluation_snapshot())


def _task_from_definition(path: Path, expected_id: str) -> _RealtimeSuiteTask:
    data = load_definition(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    task_id = data.get("task_id")
    if task_id != expected_id:
        raise ValueError(f"{path} declares task_id {task_id!r}, expected {expected_id!r}")
    scenario = data.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError(f"{path} scenario must be a non-empty string")
    if scenario != task_id:
        raise ValueError(f"{path} declares scenario {scenario!r}, expected {task_id!r}")
    if data.get("category") != "realtime":
        raise ValueError(f"{path} category must be 'realtime'")
    prompt = data.get("prompt")
    description = data.get("description")
    for field, value in (("prompt", prompt), ("description", description)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} {field} must be a non-empty string")
    return _RealtimeSuiteTask(task_id, scenario, prompt.strip())


TASKS = {
    tid: _task_from_definition(Path(_TASK_DIR) / tid / TASK_CONFIG, tid) for tid in web_definitions()
}
