#!/usr/bin/env python3
"""HTTP lifecycle for all stateful realtime browser scenarios.

Every task uses its public task id as the scenario id and implements one engine protocol.  Engines
own task-specific state and scoring; this module owns only transport, episode lifecycle, and page
rendering.
"""

from __future__ import annotations

import os
import math
import secrets
import threading
import time
from contextlib import contextmanager
from collections.abc import Mapping

from flask import (
    abort,
    send_from_directory,
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)


from pathlib import Path
from jinja2 import FunctionLoader
from realtime_tasks.registry import task_directory, load_task_module, web_definitions

_WEB_ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(_WEB_ROOT / "static"))


def _template(name):
    if not name.startswith("scenarios/") or not name.endswith(".html"):
        return None
    task_id = name.removeprefix("scenarios/").removesuffix(".html")
    if task_id not in web_definitions():
        return None
    return (task_directory("web", task_id) / "environment/index.html").read_text()


app.jinja_loader = FunctionLoader(_template)


@app.get("/assets/<task_id>/<path:filename>")
def task_asset(task_id, filename):
    if task_id not in web_definitions():
        abort(404)
    root = (task_directory("web", task_id) / "environment").resolve()
    path = (root / filename).resolve()
    if (
        not path.is_relative_to(root)
        or not path.is_file()
        or path.suffix.lower()
        not in {
            ".js",
            ".css",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".svg",
            ".gif",
            ".ico",
            ".woff",
            ".woff2",
            ".ttf",
            ".otf",
            ".mp3",
            ".wav",
            ".ogg",
        }
    ):
        abort(404)
    return send_from_directory(root, filename)


def build_engines(clock, wall_clock, agent_user):
    from realtime_tasks.web.engines.scenario import build_engines as configured
    from realtime_tasks.web.engines.product import ProductEpisode

    engines = configured(clock, wall_clock)
    for task_id, definition in web_definitions().items():
        if definition["backend"] == "product":
            engines[task_id] = ProductEpisode(
                load_task_module(task_id, definition["engine"]), clock, wall_clock
            )
        elif definition["backend"] == "bespoke":
            # A task-owned engine class with the configured-scenario interface.
            engines[task_id] = load_task_module(task_id, definition["engine"]).Engine(
                clock, wall_clock, agent_user
            )
    if set(engines) != set(web_definitions()):
        raise RuntimeError("Web engine registration differs from task metadata")
    return engines


app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
LOCK = threading.RLock()
EPISODES: dict[str, dict] = {}
_REQUEST_BODY = object()

AGENT_USER = "alex.morgan"


@app.after_request
def inject_semantic_accessibility(response):
    """Give every realtime HTML page live, context-unique accessible control names."""
    if response.direct_passthrough or response.status_code != 200 or not response.mimetype.startswith("text/html"):
        return response
    marker = b"semantic-accessibility.js"
    body = response.get_data()
    if marker in body:
        return response
    script = (
        b'<script src="/static/semantic-accessibility.js" '
        b'data-shared-semantic-accessibility="true"></script>'
    )
    closing = body.lower().rfind(b"</body>")
    body = body[:closing] + script + body[closing:] if closing >= 0 else body + script
    response.set_data(body)
    return response


def mono() -> float:
    return time.monotonic()


def episode_now(episode: dict) -> float:
    """Return server-owned world time, frozen between harness clock advances."""
    clock = episode.get("_benchmark_clock") or {}
    if clock.get("mode") == "token":
        return float(clock["now"])
    # A held episode (restarted but not yet started) stays at its origin.
    if not episode.get("opened"):
        return float(episode["started"])
    return mono()


@contextmanager
def episode_engine_clock(engine, episode: dict):
    """Point a shared engine's clocks at one episode's logical clock during an atomic call."""
    original = engine.clock
    original_wall = engine.wall_clock
    engine.clock = lambda: episode_now(episode)
    engine.wall_clock = lambda: episode["wall_started"] + episode_now(episode) - episode["started"]
    try:
        yield
    finally:
        engine.clock = original
        engine.wall_clock = original_wall


def rendered_state(engine, episode: dict) -> dict:
    """Public engine state plus a monotonic token-frame marker for browser render barriers."""
    state = engine.state(episode)
    clock = episode.get("_benchmark_clock") or {}
    if clock.get("mode") == "token":
        state["benchmark_frame"] = clock["frame"]
    return state


class _LazyScenarioEngines(Mapping):
    """Load all selected scenario engines only when the service receives a request."""

    def __init__(self):
        self._engines = None

    def _load(self):
        if self._engines is None:
            engines = build_engines(mono, time.time, AGENT_USER)
            self._engines = engines
        return self._engines

    def __getitem__(self, name):
        return self._load()[name]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())


SCENARIO_ENGINES = _LazyScenarioEngines()


# Absolute episode timestamps that must move with the episode origin.
_EPISODE_TIME_KEYS = (
    "live_at",
    "round_end",
    "shock_at",
    "event_at",
    "revision_at",
    "action_at",
    "deadline",
)


def activate_episode(episode: dict) -> None:
    """Start an episode clock on first page open without changing its relative schedule."""
    if episode.get("opened"):
        return
    if (episode.get("_benchmark_clock") or {}).get("mode") == "token":
        episode["opened"] = True
        return
    delta = mono() - episode["started"]
    episode["started"] += delta
    episode["wall_started"] = time.time()
    for key in _EPISODE_TIME_KEYS:
        if isinstance(episode.get(key), (int, float)):
            episode[key] += delta
    for item in episode.get("bot_schedule", ()):
        if isinstance(item.get("at"), (int, float)):
            item["at"] += delta
    episode["opened"] = True


def require_episode(name: str, body=_REQUEST_BODY) -> tuple[dict | None, tuple | None]:
    episode_id = request.args.get("episode")
    if body is _REQUEST_BODY and request.is_json:
        body = request.get_json(silent=True)
    if body is not _REQUEST_BODY:
        if not isinstance(body, dict):
            return None, (jsonify(ok=False, error="Request payload must be a JSON object."), 400)
        episode_id = body.get("episode", episode_id)
    episode = EPISODES.get(name)
    if not episode or episode_id != episode["episode"]:
        return None, (jsonify(ok=False, error="This session has expired. Reload the task."), 409)
    return episode, None


@app.get("/api/<name>/state")
def api_scenario_state(name: str):
    engine = SCENARIO_ENGINES.get(name)
    if engine is None:
        return jsonify(ok=False, error="Unknown scenario."), 404
    with LOCK:
        episode, error = require_episode(name)
        if error:
            return error
        with episode_engine_clock(engine, episode):
            return jsonify(rendered_state(engine, episode))


@app.post("/api/<name>/action")
def api_scenario_action(name: str):
    engine = SCENARIO_ENGINES.get(name)
    if engine is None:
        return jsonify(ok=False, error="Unknown scenario."), 404
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(ok=False, error="Action payload must be a JSON object."), 400
    with LOCK:
        episode, error = require_episode(name, body)
        if error:
            return error
        received_at = episode_now(episode)
        with episode_engine_clock(engine, episode):
            result, status = engine.action(episode, body, now=received_at)
        result.setdefault("episode", episode["episode"])
        result.setdefault("generation", episode["generation"])
        return jsonify(result), status


@app.post("/api/reset/<name>")
def api_reset(name: str):
    engine = SCENARIO_ENGINES.get(name)
    if engine is None:
        return jsonify(ok=False, error="Unknown scenario."), 404
    body = request.get_json(silent=True)
    if body is None:
        body = {}
    if not isinstance(body, dict):
        return jsonify(ok=False, error="Reset payload must be a JSON object."), 400
    seed = body.get("seed", 42)
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        return jsonify(ok=False, error="Seed must be an integer between 0 and 4294967295."), 400
    with LOCK:
        episode = engine.reset(generation=1, seed=seed)
        episode["control_token"] = secrets.token_hex(16)
        EPISODES[name] = episode
        return jsonify(
            ok=True,
            episode=episode["episode"],
            generation=episode["generation"],
            seed=episode["seed"],
            control_token=episode["control_token"],
            path=f"/{name}?episode={episode['episode']}",
        )


@app.post("/api/clock/<name>/configure")
def api_clock_configure(name: str):
    """Harness-only switch from wall time to the deterministic token clock."""
    if SCENARIO_ENGINES.get(name) is None:
        return jsonify(ok=False, error="Unknown scenario."), 404
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(ok=False, error="Clock payload must be a JSON object."), 400
    with LOCK:
        episode, error = require_episode(name, body)
        if error:
            return error
        if body.get("control_token") != episode.get("control_token"):
            return jsonify(ok=False, error="Clock control is not authorized."), 403
        fps = body.get("fps")
        if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
            return jsonify(ok=False, error="fps must be positive."), 400
        now = mono()
        episode["_benchmark_clock"] = {"mode": "token", "fps": float(fps), "frame": 0, "now": now}
        # Preserve every relative deadline while moving the episode origin to the
        # exact instant at which wall time stops.
        delta = now - episode["started"]
        episode["started"] += delta
        episode["wall_started"] = time.time()
        for key in _EPISODE_TIME_KEYS:
            if isinstance(episode.get(key), (int, float)):
                episode[key] += delta
        for item in episode.get("bot_schedule", ()):
            if isinstance(item.get("at"), (int, float)):
                item["at"] += delta
        episode["opened"] = True
        return jsonify(ok=True, mode="token", fps=float(fps), frame=0, world_time_s=0.0)


@app.post("/api/clock/<name>/advance")
def api_clock_advance(name: str):
    """Advance exactly N simulated frames; no wall-clock measurement enters."""
    engine = SCENARIO_ENGINES.get(name)
    if engine is None:
        return jsonify(ok=False, error="Unknown scenario."), 404
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(ok=False, error="Clock payload must be a JSON object."), 400
    with LOCK:
        episode, error = require_episode(name, body)
        if error:
            return error
        if body.get("control_token") != episode.get("control_token"):
            return jsonify(ok=False, error="Clock control is not authorized."), 403
        clock = episode.get("_benchmark_clock") or {}
        if clock.get("mode") != "token":
            return jsonify(ok=False, error="Token clock is not configured."), 409
        frames = body.get("frames", 1)
        if (
            not isinstance(frames, (int, float))
            or isinstance(frames, bool)
            or not math.isfinite(frames)
            or frames < 0
        ):
            return jsonify(ok=False, error="frames must be a non-negative finite number."), 400
        clock["frame"] += frames
        if abs(clock["frame"] - round(clock["frame"])) < 1e-9:
            clock["frame"] = int(round(clock["frame"]))
        clock["now"] += frames / clock["fps"]
        with episode_engine_clock(engine, episode):
            state = rendered_state(engine, episode)
        return jsonify(
            ok=True,
            mode="token",
            fps=clock["fps"],
            frame=clock["frame"],
            world_time_s=clock["frame"] / clock["fps"],
            state=state,
        )


@app.post("/api/restart/<name>")
def api_restart(name: str):
    """Re-arm the current browser episode using its harness-only control token."""
    engine = SCENARIO_ENGINES.get(name)
    if engine is None:
        return jsonify(ok=False, error="Unknown scenario."), 404
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(ok=False, error="Restart payload must be a JSON object."), 400
    with LOCK:
        current, error = require_episode(name, body)
        if error:
            return error
        token = current.get("control_token")
        if not token or body.get("control_token") != token:
            return jsonify(ok=False, error="Restart is not authorized for this session."), 403
        if type(body.get("generation")) is not int:
            return jsonify(ok=False, error="Restart requires the current generation."), 400
        if body["generation"] != current["generation"]:
            return jsonify(ok=False, error="This restart request is stale."), 409
        episode_id = current["episode"]
        episode = engine.reset(generation=current["generation"] + 1, seed=current["seed"])
        episode["episode"] = episode_id
        episode["control_token"] = token
        # A held restart keeps world time at zero until /api/start, so harness setup after the
        # restart never consumes the episode.
        if body.get("hold") is not True:
            activate_episode(episode)
        EPISODES[name] = episode
        return jsonify(
            ok=True,
            episode=episode_id,
            generation=episode["generation"],
            path=f"/{name}?episode={episode_id}",
        )


@app.post("/api/start/<name>")
def api_start(name: str):
    """Start the world clock of a held episode using its harness-only control token."""
    if SCENARIO_ENGINES.get(name) is None:
        return jsonify(ok=False, error="Unknown scenario."), 404
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(ok=False, error="Start payload must be a JSON object."), 400
    with LOCK:
        episode, error = require_episode(name, body)
        if error:
            return error
        token = episode.get("control_token")
        if not token or body.get("control_token") != token:
            return jsonify(ok=False, error="Start is not authorized for this session."), 403
        if body.get("generation") is not None and body["generation"] != episode["generation"]:
            return jsonify(ok=False, error="This start request is stale."), 409
        already_started = bool(episode.get("opened"))
        activate_episode(episode)
        return jsonify(
            ok=True,
            episode=episode["episode"],
            generation=episode["generation"],
            already_started=already_started,
            wall_started=episode["wall_started"],
        )


@app.get("/api/evaluation/<name>")
def api_evaluation(name: str):
    engine = SCENARIO_ENGINES.get(name)
    if engine is None:
        return jsonify(ok=False, error="Unknown scenario."), 404
    with LOCK:
        episode, error = require_episode(name)
        if error:
            return error
        try:
            expected_generation = int(request.args.get("generation", ""))
        except ValueError:
            return jsonify(ok=False, error="Evaluation requires the expected generation."), 400
        if expected_generation != episode["generation"]:
            return jsonify(ok=False, error="This evaluation request is stale."), 409
        with episode_engine_clock(engine, episode):
            return jsonify({**engine.evaluation(episode), "seed": episode["seed"]})


@app.get("/health")
def health():
    return jsonify(ok=True, service="realtime-suite", scenarios=len(SCENARIO_ENGINES))


@app.get("/")
def root():
    return redirect(url_for("page", name=next(iter(SCENARIO_ENGINES))))


@app.get("/<name>")
def page(name: str):
    engine = SCENARIO_ENGINES.get(name)
    if engine is None:
        return "Not found", 404
    with LOCK:
        episode = EPISODES.get(name)
        if not episode:
            episode = engine.reset(generation=1)
            EPISODES[name] = episode
            return redirect(url_for("page", name=name, episode=episode["episode"]))
        if request.args.get("episode") != episode["episode"]:
            return "This session has expired. Reload the task.", 409
        activate_episode(episode)
        return render_template(
            engine.template,
            episode=episode["episode"],
            scenario=name,
            agent_user=AGENT_USER,
            ui=engine.ui_context(),
        )


if __name__ == "__main__":
    host = os.environ.get("RT_SUITE_HOST", "127.0.0.1")
    port = int(os.environ.get("RT_SUITE_PORT", "5000"))
    app.run(host=host, port=port, threaded=True)
