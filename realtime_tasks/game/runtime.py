from __future__ import annotations
from realtime_framework.sandbox.browser.clock import (
    captured_frame_has_render_fence,
    cdp_advance_rendered_frame,
    cdp_eval,
    cdp_install_document_script,
    cdp_install_frame_clock,
    cdp_install_game_page_guard,
    cdp_key_event,
    cdp_pause_virtual_time,
    cdp_start_frame_clock,
)

# Browser game
"""Real third-party browser games, served + played in Chrome, with their live state read and
driven over the Chrome DevTools Protocol (CDP). Games preserve their pinned upstream source and
licenses; a game may also carry a minimal benchmark launch/build adapter that exposes native state.
We serve the bundle, open it in Chrome on the lane-local X11 desktop for keyboard/mouse play, and observe its
state over CDP for scoring. ``verify()`` is read-only.
"""
import base64 as base64
import io as io
import json as json
import math as math
import shlex as shlex
import time as time
from pathlib import Path as Path

from PIL import Image as Image

from realtime_framework.entity.config import task_seed
from realtime_framework.sandbox.browser.cdp import close_port
import realtime_framework.sandbox.apptainer as C
from realtime_tasks.types import Task
from realtime_framework.sandbox.actions import isolated_task_port

_SEMANTIC_ACCESSIBILITY_PATH = str(
    Path(__file__).resolve().parents[2] / "realtime_tasks" / "web" / "static" / "semantic-accessibility.js"
)
_SEEDED_RANDOM_PATH = Path(__file__).resolve().parent / "assets" / "benchmark_seed.js"
_SEED_PLACEHOLDER = "__BENCHMARK_SEED__"


def seeded_random_source(seed):
    """benchmark_seed.js bound to ``seed``: the only source of randomness for game documents."""
    source = _SEEDED_RANDOM_PATH.read_text(encoding="utf-8")
    if source.count(_SEED_PLACEHOLDER) != 1:
        raise RuntimeError(f"{_SEEDED_RANDOM_PATH} must contain {_SEED_PLACEHOLDER} exactly once")
    return source.replace(_SEED_PLACEHOLDER, str(task_seed(seed)))


# The browser clock module installs deterministic frame scheduling and suppresses
# native dialogs that would block CDP state reads. Same-origin child frames share
# the parent clock; game-specific right-click handlers still receive their events.


def _root(task_id):
    return f"/home/user/games/{task_id}"


def serve_and_launch(
    task_id, http_port, dbg_port, host_game_dir, *, seed, url_suffix="", logical_time=False
):
    """Copy the pinned game bundle in, serve it, and open it in Chrome (maximized) with remote
    debugging on the lane-local display. `url_suffix` (e.g. "?level=1", "?level=infinite") is appended to the
    served URL for games whose start mode/level is selected via the query/hash. Every document is
    seeded with `seed` before its own scripts run (assets/benchmark_seed.js). Returns
    None — Chrome already shows the game, so the harness must NOT reopen a browser."""
    root = _root(task_id)
    # Discard a connection retained for the previous Chrome process before reusing this lane port.
    close_port(dbg_port)
    # Bracket the first character so pkill's pattern cannot match the cleanup shell itself.
    C._dexec(
        f"pkill -f '[h]ttp.server {http_port}' 2>/dev/null; "
        f"pkill -f '[r]emote-debugging-port={dbg_port}' 2>/dev/null; true"
    )
    # Create a clean destination owned by the instance UID, then
    # copy the game's CONTENTS in ('/.' = contents-into-dir). Retry the cp defensively.
    for _ in range(4):
        C._dexec(
            f"rm -rf {shlex.quote(root)}; mkdir -p {shlex.quote(root)}; chmod 777 {shlex.quote(root)}"
        )
        if C._dexec(f"test -d {shlex.quote(root)} && echo ok").stdout.strip() != "ok":
            time.sleep(1)
            continue
        cp = C.copy_into(C.SANDBOX_CONTAINER, host_game_dir, root)
        if cp.returncode == 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"serve_and_launch: could not stage game into {root}")
    C._dexec(f"chmod -R a+rX {shlex.quote(root)}")
    C._dexec(
        f"cd {shlex.quote(root)} && python3 -m http.server {http_port} --bind 127.0.0.1 "
        f">/tmp/game_{task_id}.out 2>&1",
        detached=True,
    )
    C._wait_http(http_port, "/")
    # Do NOT use --disable-gpu: it disables WebGL entirely in the container, leaving WebGL games
    # black and forever `status=loading`. The configured backend is either deterministic
    # SwiftShader or the lane-visible NVIDIA GPU.
    game_url = f"http://localhost:{http_port}/{url_suffix}"
    # Chrome starts blank so the seeded-random (and token-clock) scripts are registered before the
    # game's first document exists.
    launch_url = "about:blank"
    # Chromium's aggregate --deterministic-mode also enables begin-frame-control.  That mode is
    # only externally drivable through HeadlessExperimental.beginFrame, which a headful X11 target
    # does not expose; enabling it here leaves game state advancing while the visible window keeps
    # its old surface.  Keep the deterministic subflags that are valid for headful Chrome and let
    # the compositor publish frames normally.
    deterministic = (
        "--run-all-compositor-stages-before-draw "
        "--disable-new-content-rendering-timeout "
        "--disable-threaded-animation --disable-threaded-scrolling "
        "--enable-surface-synchronization "
        if logical_time
        else ""
    )
    from realtime_framework.sandbox.apptainer import chrome_graphics_flags

    C._dexec(
        f"DISPLAY={C.SANDBOX_DISPLAY} google-chrome-stable --remote-debugging-port={dbg_port} --remote-allow-origins=* "
        f"--user-data-dir=/tmp/cdp_{task_id} --no-sandbox --no-first-run --no-default-browser-check "
        f"{chrome_graphics_flags()} --enable-webgl {deterministic}"
        f"--start-maximized {shlex.quote(launch_url)} "
        f">/tmp/chrome_{task_id}.out 2>&1",
        detached=True,
    )
    ready = C._dexec(
        f"for i in $(seq 1 40); do curl -sf localhost:{dbg_port}/json >/dev/null 2>&1 "
        f"&& exit 0; sleep 0.5; done; exit 1"
    )
    if ready.returncode != 0:
        detail = C._dexec(f"tail -n 30 /tmp/chrome_{shlex.quote(task_id)}.out").stdout
        raise RuntimeError(f"Chrome CDP did not start for {task_id}: {detail[-800:]}")
    cdp_install_document_script(
        dbg_port, seeded_random_source(seed), url=None if logical_time else game_url
    )
    if logical_time:
        cdp_install_frame_clock(dbg_port, game_url)
    cdp_install_game_page_guard(dbg_port)
    install_accessible_semantics(dbg_port)
    return None


def install_accessible_semantics(dbg_port):
    """Install the shared live accessible-name enhancer without modifying vendored game files."""
    try:
        with open(_SEMANTIC_ACCESSIBILITY_PATH, encoding="utf-8") as handle:
            source = handle.read()
        return cdp_eval(dbg_port, source)
    except (OSError, UnicodeError):
        return None


class _BrowserGame(Task):
    category = "browser_game"
    http_port = isolated_task_port(8930)
    dbg_port = isolated_task_port(9222)
    game_dir = None  # subclass: absolute path to realtime_tasks/game/<task-id>/environment/
    keys = ()  # public controls only; never include evaluator/game-state data

    def _freeze_token_clock(self):
        """Stop wall-time game progress without marking the logical clock active."""
        if self.clock_snapshot()["mode"] != "token":
            return None
        cdp_pause_virtual_time(self.dbg_port)
        self._last_clock_advance = cdp_start_frame_clock(self.dbg_port)
        return self._last_clock_advance

    def activate_clock(self):
        snapshot = super().activate_clock()
        if snapshot["mode"] == "token":
            self._freeze_token_clock()
        return self.clock_snapshot()

    def advance_clock(self, frames: float = 1, key_events=()):
        """Advance the token clock; ``key_events`` carry ``at_frames`` offsets into the advance."""
        before = self.clock_snapshot()
        if before["mode"] == "token" and (frames or key_events):
            fps = float(before["fps"])
            scheduled = [
                {**event, "at_seconds": float(event["at_frames"]) / fps}
                for event in (key_events or ())
            ]
            self._last_clock_advance = cdp_advance_rendered_frame(
                self.dbg_port,
                frames / fps,
                key_events=scheduled,
            )
            if not self._last_clock_advance.get("compositor_committed"):
                raise RuntimeError("browser frame advanced without a compositor commit")
        return super().advance_clock(frames)

    def render_current_frame(self):
        """Commit the current page surface without moving simulated browser time."""
        before = self.clock_snapshot()
        if before["mode"] == "token":
            self._last_clock_advance = cdp_advance_rendered_frame(self.dbg_port, 0.0)
            if not self._last_clock_advance.get("compositor_committed"):
                raise RuntimeError("browser frame rendered without a compositor commit")
        return self.clock_snapshot()

    def captured_frame_is_current(self, screenshot: bytes) -> bool:
        """Return whether a desktop capture contains the latest post-step render fence.

        The CDP compositor barrier proves the page surface is current; this second check closes
        the remaining compositor-to-X11 gap so the exact desktop image handed to the model cannot
        still be the prior frame.
        """
        fence = (getattr(self, "_last_clock_advance", {}) or {}).get("render_fence")
        return captured_frame_has_render_fence(screenshot, fence)

    def clock_key_event(self, event_type: str, key: str, auto_repeat=False, held_modifiers=()):
        """Dispatch a key without sleeping so it can remain held across virtual frames."""
        if event_type not in {"keyDown", "keyUp"}:
            raise ValueError("event_type must be keyDown or keyUp")
        return cdp_key_event(
            self.dbg_port,
            event_type,
            key,
            auto_repeat=auto_repeat,
            held_modifiers=held_modifiers,
        )


# Catalog games
"""Shared plumbing for vendored real-time browser games. Each game is a self-contained static
bundle under ``realtime_tasks/game/<task-id>/environment/`` carrying a read-only ``game_api.js`` observation
shim (loaded by the game's own index.html) that exposes a uniform
`window.gameAPI.getState()` (schema 2.0):

    { status, is_actionable, terminal:{isTerminal,outcome,reason},
      game_state:{score, level, completion_progress, ...}, metrics:{primary_score, ...} }

We serve the bundle, open it in Chrome on the lane-local desktop, and read its live state over CDP;
``verify()`` is read-only. Each game's success rule is a catalog-owned ``score_field`` dotted
path plus numeric ``target``. Some bundles ship their observation shim alongside pinned game
files; others keep upstream game code under ``upstream/`` behind a separate root adapter.

"""
import os as os
from pathlib import Path as Path

_GAMES_DIR = str(Path(__file__).resolve().parents[2] / "realtime_tasks" / "game")

# Read the read-only observation shim as JSON (null until the shim is loaded/initialised).
_GET_STATE = (
    "JSON.stringify((window.gameAPI && window.gameAPI.getState) ? window.gameAPI.getState() : null)"
)
# Release a deferred loader (deferred_start games start their countdown/launch here, not on load).
_ACTIVATE = "window.gameAPI&&window.gameAPI.activate?window.gameAPI.activate():null"


def _dig(state, path):
    """Follow a dotted path (e.g. 'game_state.level', 'metrics.kills') into the state dict."""
    cur = state
    for k in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def build_task_prompt(prompt: str, rules: str, keys) -> str:
    """Combine catalog-owned task objective, rules, and controls into user prompt content."""
    keyline = ", ".join(keys) if keys else "mouse clicks only"
    return (
        f"{prompt.strip()}\n\n"
        f"GAME-SPECIFIC RULES:\n{rules.strip()}\n\n"
        f"GAME-SPECIFIC CONTROLS: {keyline}. Use visible controls only."
    )


class _CatalogGame(_BrowserGame):
    """One catalog game as a streaming browser-game task. Category 'browser_game' so the runner
    ignores model failure declarations (honor_fail=False) and auto-stops at the game's own terminal
    (episode_over -> GuestEnv.stop_check)."""

    game_key = None  # catalog key "NN_name" from task.toml, e.g. "55_evolve"
    score_field = "metrics.primary_score"  # dotted path into gameAPI.getState()
    target_score = 1.0
    http_port = 8940
    dbg_port = 9240
    url_suffix = ""  # optional "?level=1" / "?level=infinite" the game needs to start in mode
    keys = ()  # public catalog controls; never evaluator/game-state data
    _desc = ""  # catalog-composed user prompt, set per-task by _make()
    # Progress accounting (reset per run in provision()): the first score seen and
    # the peak score seen, so partial progress survives a mid-run score drop/reset.
    _score_start = None
    _score_best = None
    deferred_start = False

    @property
    def game_dir(self):
        return os.path.join(_GAMES_DIR, self.task_id, "environment")

    def description(self):
        return self._desc

    def interactable_providers(self):
        return ({"type": "keyboard", "keys": tuple(self.keys)},) if self.keys else ()

    # ---- live state (read-only) --------------------------------------------
    def _state(self):
        raw = cdp_eval(self.dbg_port, _GET_STATE)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return None
        return raw if isinstance(raw, dict) else None

    def provision(self):
        # Fresh progress accounting for this run (score_start/score_best; the
        # instance is reused across runs, so reset the peaks here at the start of each run).
        self._score_start = None
        self._score_best = None
        # Serve the bundle + open it in Chrome (serve_and_launch returns None: Chrome
        # already shows the game, so the harness must NOT reopen a browser).
        serve_and_launch(
            self.task_id,
            self.http_port,
            self.dbg_port,
            self.game_dir,
            seed=self.seed,
            url_suffix=self.url_suffix or "",
            logical_time=(getattr(self, "_clock_config", {}) or {}).get("mode") == "token",
        )
        # Wait for the game_api.js shim, then call its init() once. The fresh, deterministic
        # episode is started later by reset_episode() (which the runner calls right before the
        # agent loop, so the baseline is sampled AFTER model warmup and pre-agent
        # auto-progression is excluded).
        for _ in range(40):
            if cdp_eval(
                self.dbg_port, "typeof window.gameAPI!=='undefined' && !!window.gameAPI.getState"
            ):
                cdp_eval(
                    self.dbg_port,
                    "try{window.gameAPI.init&&window.gameAPI.init({seed:%d})}catch(e){}" % self.seed,
                )
                break
            time.sleep(0.5)
        return None

    # ---- fresh-episode start + baseline pinning ----------------------------
    def reset_episode(self):
        """Start a FRESH, deterministic episode and pin the baseline score NOW.

        Called by the runner right before the agent loop (i.e. AFTER the container/model warmup),
        so the score the agent must beat is sampled from a freshly-restarted game rather than from a
        game that has been auto-running (uninstructed) during the seconds of model load + the 5s
        provisioning sleep. We call the shim's init() then reset() (the restart), wait for the game
        to become actionable again, then sample score_start/score_best from the fresh state.
        Failure to restore the page or its logical clock is an environment error; score sampling
        itself remains best-effort.
        """
        seed = self.seed
        # Fallback scored origin for a backend that cannot hold its episode (see start_episode).
        self._reset_started_at = time.time()
        js = (
            "(async function(){try{"
            "if(window.gameAPI&&window.gameAPI.init){await window.gameAPI.init({seed:%d});}"
            "if(window.gameAPI&&window.gameAPI.reset){await window.gameAPI.reset({seed:%d});}"
            "return 'ok';}catch(e){return 'err:'+String(e);}})()" % (seed, seed)
        )
        try:
            cdp_eval(self.dbg_port, js)
        except Exception:
            pass
        # A reset() may fall back to window.location.reload(); wait for the game API to come back.
        game_api_ready = False
        for _ in range(40):
            if cdp_eval(
                self.dbg_port, "typeof window.gameAPI!=='undefined' && !!window.gameAPI.getState"
            ):
                game_api_ready = True
                break
            time.sleep(0.5)
        if not game_api_ready:
            raise RuntimeError("game API did not return after episode reset")

        # The persistent CDP target normally carries the new-document registrations (seeded random
        # and, under the token clock, the frame clock) across reloads. If Chrome replaced the target
        # or lost a registration, reinstall them by navigating once more. This page load is itself
        # a clean episode, so do not call reset() again and trigger another uninstrumented reload.
        page_scripts_reinstalled = False
        clock_mode = (getattr(self, "_clock_config", {}) or {}).get("mode")
        seed_lost = not cdp_eval(self.dbg_port, "Number.isInteger(window.__benchmarkSeed)")
        clock_lost = clock_mode == "token" and not cdp_eval(
            self.dbg_port, "typeof window.__benchmarkClock !== 'undefined'"
        )
        if seed_lost or clock_lost:
            game_url = f"http://localhost:{self.http_port}/{self.url_suffix or ''}"
            cdp_install_document_script(
                self.dbg_port,
                seeded_random_source(self.seed),
                url=None if clock_mode == "token" else game_url,
            )
            if clock_mode == "token":
                cdp_install_frame_clock(self.dbg_port, game_url)
            page_scripts_reinstalled = True
            game_api_ready = False
            for _ in range(40):
                if cdp_eval(
                    self.dbg_port,
                    "typeof window.gameAPI!=='undefined' && !!window.gameAPI.getState",
                ):
                    game_api_ready = True
                    break
                time.sleep(0.5)
            if not game_api_ready:
                raise RuntimeError("game API did not return after page-script reinstallation")
            # Re-seed the observation wrapper's episode bookkeeping without restarting/reloading
            # the freshly navigated game document.
            cdp_eval(
                self.dbg_port,
                "try{window.gameAPI.init&&window.gameAPI.init({seed:%d})}catch(e){}" % seed,
            )
        # Give the fresh episode a moment to become actionable (score at/near its start value). In
        # token-clock mode, freeze virtual time and take ownership of RAF immediately when that
        # state is observed. Baseline reads, semantics installation, and later model/provider setup
        # must not consume an auto-running game's episode in physical time.
        token_clock_frozen = False
        for _ in range(16):
            st = self._state()
            if st and (
                _dig(st, "is_actionable") is True or _dig(st, "status") in ("playing", "ready")
            ):
                if clock_mode == "token":
                    self._freeze_token_clock()
                    token_clock_frozen = True
                break
            time.sleep(0.25)
        # Preserve the token-clock invariant even for a malformed/non-actionable state. The
        # caller can diagnose that state from the returned baseline without allowing more drift.
        if clock_mode == "token" and not token_clock_frozen:
            self._freeze_token_clock()
        # Some shim reset implementations reload the document, which removes the provision-
        # time observer. Reinstall it on the fresh, now-frozen page before the first observation.
        install_accessible_semantics(self.dbg_port)
        # Pin the baseline as the MINIMUM score over a short rapid burst right after reset. A freshly
        # restarted game is at its score floor (≈0) immediately after reset. Under the token clock,
        # every read now observes that same frozen instant; under the wall clock, taking the minimum
        # still captures the floor closest to the true fresh start.
        self._score_start = None
        self._score_best = None
        floor = None
        # A held (deferred_start) episode cannot move, so its burst costs no episode time. A
        # wall-clock game that runs during reset takes one read so setup stays short.
        burst = 1 if clock_mode != "token" and not self.deferred_start else 10
        for index in range(burst):
            st = self._state() or {}
            v = self._numeric_score(st)
            if v is not None:
                floor = v if floor is None else min(floor, v)
            if index + 1 < burst:
                time.sleep(0.12)
        if floor is not None:
            self._score_start = floor
            self._score_best = floor
        st = self._state() or {}
        # A token clock that is already active owns game time, so releasing the reloaded loader
        # here advances nothing. Wall-clock episodes stay held until start_episode().
        if (
            self.deferred_start
            and clock_mode == "token"
            and (getattr(self, "_clock_config", None) or {}).get("active")
        ):
            cdp_eval(self.dbg_port, _ACTIVATE)
        # benchmark_seed.js publishes the seed driving the page as window.__benchmarkSeed. A missing
        # value means the registration was lost and the game is drawing native randomness.
        applied = cdp_eval(
            self.dbg_port,
            "Number.isInteger(window.__benchmarkSeed) ? window.__benchmarkSeed : null",
        )
        if applied != self.seed:
            raise RuntimeError(f"game applied seed {applied!r}, expected {self.seed}")
        self.applied_seed = applied
        return {
            "seed": self.seed,
            "applied_seed": applied,
            "score_start": self._score_start,
            "score_best": self._score_best,
            "status": _dig(st, "status"),
            "is_actionable": _dig(st, "is_actionable"),
            "page_scripts_reinstalled": page_scripts_reinstalled,
        }

    def activate_clock(self):
        """Under the token clock, release any deferred loader and take ownership of game time.

        The wall clock releases the loader in start_episode(), after every setup step.
        """
        clock_mode = (getattr(self, "_clock_config", {}) or {}).get("mode")
        if self.deferred_start and clock_mode == "token":
            cdp_eval(self.dbg_port, _ACTIVATE)
        return super().activate_clock()

    def start_episode(self):
        """Release the held wall-clock episode once all framework setup is complete.

        The origin is taken before the release, so the game's own start sequence (title key press,
        launch countdown) counts against the agent and never in its favour. The first observation
        is captured only after the game reports an actionable state.
        """
        if (getattr(self, "_clock_config", {}) or {}).get("mode") == "token":
            return super().start_episode()
        if not self.deferred_start:
            return {
                "clock_origin": getattr(self, "_reset_started_at", None) or time.time(),
                "start_hold": False,
            }
        origin = time.time()
        cdp_eval(self.dbg_port, _ACTIVATE)
        actionable = False
        deadline = origin + 5.0
        while time.time() < deadline:
            st = self._state() or {}
            if _dig(st, "is_actionable") is True or _dig(st, "status") == "playing":
                actionable = True
                break
            time.sleep(0.02)
        if not actionable:
            raise RuntimeError("held game did not become actionable after start_episode()")
        return {
            "clock_origin": origin,
            "start_hold": True,
            "release_latency_s": round(time.time() - origin, 3),
        }

    # ---- progress accounting -----------------------------------------------
    def _numeric_score(self, st):
        """The game's current numeric score (score_field dotted-path), or None if absent/non-numeric."""
        try:
            return float(_dig(st or {}, self.score_field))
        except (TypeError, ValueError):
            return None

    def _sample(self, st):
        """Fold one live state read into score_start (first seen) / score_best (peak)."""
        val = self._numeric_score(st)
        if val is None:
            return
        if self._score_start is None:
            self._score_start = val
        self._score_best = val if self._score_best is None else max(self._score_best, val)

    def evaluation_snapshot(self):
        """Freeze every scorer input immediately when model inference ends."""
        st = self._state() or {}
        self._sample(st)
        return {
            "kind": "browser_game",
            "state": st,
            "score_start": self._score_start,
            "score_best": self._score_best,
            "score_field": self.score_field,
            "target_score": self.target_score,
        }

    def evaluate_snapshot(self, snapshot):
        return self._evaluate_state(
            snapshot.get("state") or {}, snapshot.get("score_start"), snapshot.get("score_best")
        )

    def verify(self):
        snapshot = self.evaluation_snapshot()
        return self.evaluate_snapshot(snapshot)

    def _evaluate_state(self, st, saved_start=None, saved_best=None):
        from realtime_tasks.game.verifier import verify

        return verify(
            self.task_id,
            st,
            score_field=self.score_field,
            target_score=self.target_score,
            saved_start=saved_start,
            saved_best=saved_best,
        )

    def episode_over(self):
        # Auto-stop the instant the game reports it is decided (won or lost) OR the benchmark target
        # is reached. This is polled continuously by the runner (GuestEnv.stop_check), so it doubles
        # as the progress sampler — one live read feeds the terminal check, the target check and
        # score_best (no extra CDP calls).
        st = self._state()
        if not st:
            raise RuntimeError("game state backend returned no state")
        self._sample(st)
        if _dig(st, "terminal.isTerminal") is True:
            return True
        # Success contract parity: a game whose score reaches `target` should end NOW rather than
        # keep running until the wall clock (some games continue past the target, and their score
        # could drift back below target before the final read). Reuse the same target_reached logic
        # verify()/evaluate_snapshot() use, against the peak sampled so far.
        ev = self._evaluate_state(st, self._score_start, self._score_best)
        return bool(ev.get("target_reached"))


# Catalog tasks
"""Vendored real-time browser games as streaming tasks (category ``browser_game``).

Source of truth = the ``[metadata]`` table of ``realtime_tasks/game/<task-id>/task.toml`` (one file per
game, generated from or wrapping pinned upstream browser-game repositories). Each definition becomes one
``_CatalogGame`` instance. The
game bundle lives at ``realtime_tasks/game/<task-id>/environment/``; success = its own
``gameAPI.getState()`` ``score_field`` reaching ``target``. Some bundles ship their state shim
alongside pinned game files; others keep upstream code isolated in a bundle subdirectory and use a
thin benchmark adapter at the bundle root. See
``realtime_tasks/game/verifier.py``.

Gomoku uses the same catalog fields, ``window.gameAPI`` state contract, loader, and scorer as every
other game; its original files remain isolated below its bundle's ``game/`` directory.
"""
import re as re
import sys as sys
from realtime_framework.sandbox.actions import isolated_task_port

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from realtime_tasks.registry import TASK_CONFIG, load_definition

_GAME_DIR = os.path.join(_ROOT, "realtime_tasks", "game")


def _make(d: dict, source: str) -> _CatalogGame:
    required_text = (
        "task_id",
        "game_key",
        "tier",
        "score_field",
        "prompt",
        "description",
        "rules",
    )
    missing = [
        field
        for field in required_text
        if not isinstance(d.get(field), str) or not d[field].strip()
    ]
    keys = d.get("keys")
    if missing or not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
        raise ValueError(f"invalid game task definition {d.get('task_id')!r}: {missing}")
    if (
        not isinstance(d.get("target"), (int, float))
        or isinstance(d.get("target"), bool)
        or not isinstance(d.get("idx"), int)
        or isinstance(d.get("idx"), bool)
        or not isinstance(d.get("http_port"), int)
        or not isinstance(d.get("dbg_port"), int)
    ):
        raise ValueError(f"invalid game numeric fields for {d.get('task_id')!r}")

    game_key = d["game_key"].strip()
    match = re.fullmatch(r"(\d{2})_.+", game_key)
    if not match:
        raise ValueError(f"game definition {source!r} game_key {game_key!r} must be NN_name")
    expected_tier = f"{match.group(1)}_01"
    if d["tier"].strip() != expected_tier:
        raise ValueError(f"game definition {source!r} tier must be {expected_tier!r}")
    expected_http_port = 8940 + d["idx"]
    expected_dbg_port = 9240 + d["idx"]
    if d["http_port"] != expected_http_port or d["dbg_port"] != expected_dbg_port:
        raise ValueError(
            f"game definition {source!r} ports must be "
            f"http={expected_http_port}, dbg={expected_dbg_port} for idx={d['idx']}"
        )
    optional_limits = ("think_budget_s", "per_move_cap_s")
    invalid_limits = [
        field
        for field in optional_limits
        if field in d
        and (not isinstance(d[field], (int, float)) or isinstance(d[field], bool) or d[field] <= 0)
    ]
    if invalid_limits:
        raise ValueError(
            f"invalid game task limits for {d.get('task_id')!r}: {invalid_limits}"
        )
    if "deferred_start" in d and not isinstance(d["deferred_start"], bool):
        raise ValueError(
            f"invalid game deferred_start for {d.get('task_id')!r}: expected boolean"
        )

    tid = d["task_id"].strip()
    cls = type(
        "Game_" + tid,
        (_CatalogGame,),
        {
            "task_id": tid,
            "game_key": game_key,
            "score_field": d["score_field"].strip(),
            "target_score": d["target"],
            # distinct base ports per game (8940/9240 + idx from task.toml) avoid stale-process
            # collisions; isolated_task_port() offsets them per lane.
            "http_port": isolated_task_port(d["http_port"]),
            "dbg_port": isolated_task_port(d["dbg_port"]),
            "url_suffix": d.get("url_suffix") or "",
            "keys": tuple(keys),
            "_desc": build_task_prompt(d["prompt"], d["rules"], keys),
            "source_id": d.get("source"),
            "deferred_start": d.get("deferred_start", False),
            **{field: d[field] for field in optional_limits if field in d},
        },
    )
    return cls()


TASKS = {}
_ALLOCATIONS = {"idx": {}, "http_port": {}, "dbg_port": {}}
for _path in sorted(Path(_GAME_DIR).glob("*/" + TASK_CONFIG)):
    _d = load_definition(_path)
    _source = str(_path.relative_to(_ROOT))
    _t = _make(_d, _source)
    if _t.task_id in TASKS:
        raise ValueError(f"duplicate game task_id {_t.task_id!r}")
    for _field, _seen in _ALLOCATIONS.items():
        _value = _d[_field]
        if _value in _seen:
            raise ValueError(
                f"duplicate game {_field} {_value!r}: {_seen[_value]!r} and {_source!r}"
            )
        _seen[_value] = _source
    TASKS[_t.task_id] = _t
