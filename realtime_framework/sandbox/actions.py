from __future__ import annotations

# Runtime
"""Episode environment boundary for desktop capture, simulated time, and action execution."""

import json as json
import math as math
import shlex as shlex
import string as string
import threading as threading
import traceback as traceback
from contextlib import contextmanager as contextmanager
from pathlib import Path as Path

from realtime_framework.clock import episode_clocks
from realtime_framework.entity.actions import Action
from realtime_framework.entity.actions import ActionType
from realtime_framework.entity.actions import Observation
from realtime_framework.entity.actions import normalize_action
from realtime_framework.entity.config import EnvironmentBudgetConfig
from realtime_framework.sandbox.budget import BudgetExceeded
from realtime_framework.sandbox.budget import Deadline
from realtime_framework.sandbox.budget import operation_budget
from realtime_framework.sandbox.guest import GuestRefused
from realtime_framework.sandbox.guest import action_timeout
from realtime_framework.sandbox.observation.capture import ensure_observation_geometry

ENVIRONMENT_BUDGET = EnvironmentBudgetConfig()

# A tapped key (KEY action under the token clock) stays down for this many seconds so that one
# rendered frame observes it: one 60 Hz rAF at any environment fps.
TAP_HOLD_S = 1.0 / 60.0

# Reloading a browser game discards episode state. Reject F5/BrowserRefresh and
# Ctrl/Meta+R for browser_game tasks; server-backed Web tasks may reload safely.
PAGE_RELOAD_KEYS = frozenset({"f5", "browserrefresh"})


# The model acts on the page only. These chords operate Chrome itself (address bar, tabs,
# windows, history, menus, find, zoom, developer tools) or the desktop, so they are rejected in
# every task. A chord matches when its keys are pressed, possibly with further modifiers held.
BROWSER_SHORTCUTS = tuple(
    frozenset(chord.split("+"))
    for chord in (
        "f3", "f6", "f10", "f11", "f12",
        "ctrl+l", "alt+d", "ctrl+k", "ctrl+e",
        "ctrl+t", "ctrl+n", "ctrl+w", "ctrl+f4", "ctrl+q", "alt+f4", "alt+space",
        "ctrl+tab", "ctrl+pageup", "ctrl+pagedown",
        *(f"ctrl+{digit}" for digit in "123456789"),
        "alt+arrowleft", "alt+arrowright", "alt+home", "alt+f", "alt+e", "alt+shift+t",
        "ctrl+f", "ctrl+g", "ctrl+h", "ctrl+j", "ctrl+d", "ctrl+p", "ctrl+s", "ctrl+o", "ctrl+u",
        "ctrl+shift+i", "ctrl+shift+j", "ctrl+shift+c", "ctrl+shift+a", "ctrl+shift+b",
        "ctrl+shift+m", "ctrl+shift+o", "ctrl+shift+delete",
        "ctrl+=", "ctrl+-", "ctrl+0", "ctrl+plus", "ctrl+minus", "ctrl+equal",
        "meta",  # desktop menus
    )
)
POINTER_ACTIONS = frozenset(
    {
        "click", "right_click", "double_click", "triple_click",
        "mouse_down", "mouse_up", "scroll", "drag", "move",
    }
)


class ActionRejected(ValueError):
    """The runtime refused a well-formed action; the model is told why and the run stays valid."""


# Match keyboard repeat behavior under both clocks: repeat only the newest held
# non-modifier key, and stop when it is released. The wall-clock path pins X11 to
# the same delay and interval used by the simulated clock.
AUTOREPEAT_DELAY_S = 0.5
AUTOREPEAT_INTERVAL_S = 0.05
MODIFIER_KEYS = ("ctrl", "shift", "alt", "meta")

# A wall-clock lease whose keyup the guest cannot execute is retried this often, this many times,
# and then reported as an environment failure (the key is physically stuck down).
WALL_RELEASE_RETRY_S = 0.5
WALL_RELEASE_MAX_ATTEMPTS = 4

# X keysyms for wall-clock holds, which xdotool presses and a timer releases so held keys do not
# block the action loop. Token-clock leases use the same press/release semantics.
_XDOTOOL_KEYSYMS = {
    "arrowleft": "Left",
    "arrowright": "Right",
    "arrowup": "Up",
    "arrowdown": "Down",
    "space": "space",
    "enter": "Return",
    "escape": "Escape",
    "tab": "Tab",
    "backspace": "BackSpace",
    "delete": "Delete",
    "insert": "Insert",
    "home": "Home",
    "end": "End",
    "pageup": "Prior",
    "pagedown": "Next",
    "ctrl": "Control_L",
    "shift": "Shift_L",
    "alt": "Alt_L",
    "meta": "Super_L",
    **{f"f{number}": f"F{number}" for number in range(1, 13)},
}


def _xdotool_keysym(identity):
    """Return the X keysym for a normalized key identity, or None when unsupported.

    Only ASCII letters and digits map to themselves: xdotool ignores an unknown keysym such as
    'é' with exit code 0, which would report a hold as applied without pressing anything.
    """
    if identity in _XDOTOOL_KEYSYMS:
        return _XDOTOOL_KEYSYMS[identity]
    if len(identity) == 1 and (identity in string.ascii_lowercase or identity in string.digits):
        return identity
    return None


class GuestEnvironment:
    """Imperative environment state used by the model-independent harness scheduler."""

    def __init__(
        self,
        guest,
        task,
        frames_dir=None,
        honor_fail=True,
        stop_check=None,
        scene_capture=None,
        action_audit_path=None,
        clock_mode="token",
        wall_start_time=None,
        clocks=None,
    ):
        self.g = guest
        # One environment tick, from the task clock the task and its service advance by.
        fps = task.clock_snapshot()["fps"]
        self.d = 1.0 / fps
        # Wall-clock time, waits and key releases read the semantic clock; under the token clock
        # it is the FrameClock that holds the logical tick. Records read the operational clock.
        self.clocks = clocks or episode_clocks(clock_mode, fps)
        if clock_mode == "token" and self.clocks.semantic.frame_seconds != self.d:
            raise ValueError(
                f"the frame clock counts {self.clocks.semantic.frame_seconds:g} s frames, "
                f"the task {self.d:g} s ticks"
            )
        if clock_mode != "token" and self.clocks.semantic is not self.clocks.operational:
            raise ValueError("the wall clock reads one clock for world and real time")
        self.t0 = (
            self.clocks.operational.time() if wall_start_time is None else float(wall_start_time)
        )
        self.clock_mode = clock_mode
        self.task = task
        self._episode_horizon_tick = None
        # Token-clock holds are leases on individual keys. This lets a movement hold survive later
        # actions and lets overlapping chords extend shared keys without an early keyUp.
        self._held_keys = {}
        # The one held key whose auto-repeat presses are currently scheduled (see AUTOREPEAT_*).
        self._repeat_key = None
        # The identities of the keys the last execute() leased with a token-clock hold_key, for
        # the kernel's shell, which ends the leases itself; () on the wall clock, whose release
        # timer ends them.
        self.last_held_keys = ()
        # The end checks the last key tap made, for the kernel's shell, which counts failed state
        # reads: each (ended, reason, error) as check_end() answered or failed, (False, None, None)
        # also when the task has no end check.
        self.last_end_checks = ()
        # Wall-clock leases: identity -> {"wire", "keysym", "expires_at" (semantic clock time)},
        # and "stuck" once the release timer gave up on the key, left for the cleanup to release.
        # _wall_lock guards them and the release timer and is never held over a guest request,
        # so the key checks and the hold snapshot do not wait for one; _wall_commands serializes
        # the xdotool commands that press and release leased keys with the lease changes they make.
        self._wall_holds = {}
        self._wall_lock = threading.RLock()
        self._wall_commands = threading.RLock()
        self._wall_timer = None
        self._wall_timer_generation = 0
        self._wall_closed = False
        self._wall_release_failures = 0
        # Set by the kernel's shell: called with the invalid-run record when the release timer
        # invalidates the run.
        self.on_invalid = None
        # Result of pinning the wall display's auto-repeat to AUTOREPEAT_* (set on first use).
        self._wall_autorepeat = None
        self._held_mouse_buttons = set()
        self.scene_capture = scene_capture
        self._done = False
        self.actions = 0
        self.frames_dir = frames_dir
        self.fi = 0
        self._fi_lock = threading.Lock()
        self.honor_fail = honor_fail
        self.stop_check = stop_check
        self._episode_over = False
        self._terminal_reason = None
        self._terminal_action = None
        self.failed = 0
        self._last_exec_error = None
        self.capture_latencies = []
        self.capture_times = []
        self.frame_ages = []
        self._last_execution_result = None
        self._last_coordinate_trace = None
        self.action_audit_path = Path(action_audit_path) if action_audit_path else None
        self._audit_lock = threading.Lock()

    def set_episode_horizon(self, max_tick):
        """Install the scheduler-owned hard stop for every token-clock advance."""
        value = float(max_tick)
        if not math.isfinite(value) or value < 0:
            raise ValueError("episode horizon must be a non-negative finite tick")
        self._episode_horizon_tick = value

    @property
    def _logical_tick(self):
        """The token clock's tick (frames advanced), kept by the semantic FrameClock."""
        return self.clocks.semantic.frame

    @_logical_tick.setter
    def _logical_tick(self, value):
        self.clocks.semantic.frame = value

    def tick_index(self):
        if self.clock_mode == "token":
            return self._logical_tick
        return int(self.clock_position())

    def clock_position(self):
        """Return the current clock in frames without quantizing wall time."""
        if self.clock_mode == "token":
            return self._logical_tick
        return max(0.0, (self.clocks.semantic.time() - self.t0) / self.d)

    def capture(self):
        started = self.clocks.operational.time()
        with operation_budget(
            "capture", ENVIRONMENT_BUDGET.capture_s, clock=self.clocks.operational
        ):
            obs = (
                self.scene_capture.capture(self.tick_index())
                if self.scene_capture is not None
                else Observation(
                    self.g.screenshot(), self.clocks.operational.time(), self.tick_index()
                )
            )
        ensure_observation_geometry(obs)
        if self.clock_mode == "token":
            obs.timestamp = self.clocks.semantic.monotonic()
        finished = self.clocks.operational.time()
        self.capture_latencies.append(finished - started)
        self.capture_times.append(getattr(obs, "timestamp", finished))
        obs.metadata["capture_latency_s"] = round(finished - started, 6)
        if self.frames_dir is not None:
            try:
                with self._fi_lock:
                    index = self.fi
                    self.fi += 1
                (self.frames_dir / f"frame_{index:05d}.png").write_bytes(obs.screenshot)
            except Exception:
                pass
        return obs

    def wait_until_tick(self, target):
        if self.clock_mode == "token":
            target = float(target)
            if self._episode_horizon_tick is not None:
                target = min(target, self._episode_horizon_tick)
            while self._logical_tick + 1e-9 < target:
                step = min(1.0, float(target) - self._logical_tick)
                if self._held_keys:
                    # Hold expiries split the advance so the keyUp lands at its exact tick; the
                    # repeating key's presses are delivered inside the advance instead.
                    next_expiry = min(item["expires_at_tick"] for item in self._held_keys.values())
                    step = min(step, max(0.0, next_expiry - self._logical_tick))
                    if step <= 1e-9:
                        self._release_expired_clock_keys()
                        continue
                self._advance_clock(step)
                self._release_expired_clock_keys()
                if self._ended_after_step():
                    break

    def _ended_after_step(self):
        """The end check after each token step of wait_until_tick: the kernel counts failed reads,
        so every check goes to last_end_checks as (ended, reason, error), and after a failed one,
        (False, None, its traceback), the advance goes on."""
        try:
            ended, reason = self.check_end()
        except Exception:  # noqa: BLE001 - the kernel counts the failed read
            self.last_end_checks += ((False, None, traceback.format_exc()),)
            return False
        self.last_end_checks += ((ended, reason, None),)
        return ended

    def step_clock(self, step):
        """Token clock, for the kernel's shell: advance ``step`` ticks, never past the episode
        horizon. Leases are not cut, no key is released and the end is not checked: the kernel
        does each (evaluation/shell.py)."""
        step = float(step)
        if self._episode_horizon_tick is not None:
            step = min(step, self._episode_horizon_tick - self._logical_tick)
        if step > 0:
            self._advance_clock(step)

    def _advance_clock(self, step):
        """Advance the task clock ``step`` ticks, with the repeating key's presses due meanwhile."""
        repeat_events, repeats_sent = self._pending_repeat_events(step)
        try:
            with operation_budget(
                "clock advance",
                ENVIRONMENT_BUDGET.clock_advance_s,
                clock=self.clocks.operational,
            ):
                if repeat_events:
                    self.task.advance_clock(step, key_events=repeat_events)
                else:
                    self.task.advance_clock(step)
        except Exception as exc:
            self._invalid_run = {
                "category": "environment",
                "reason": "clock_advance_failed",
                "detail": str(exc),
            }
            raise
        if repeat_events and self._repeat_key in self._held_keys:
            self._held_keys[self._repeat_key]["repeats_sent"] = repeats_sent
        self._logical_tick += step
        if abs(self._logical_tick - round(self._logical_tick)) < 1e-9:
            self._logical_tick = int(round(self._logical_tick))

    def render_current_frame(self):
        """Publish action-visible pixels without advancing the token clock."""
        if self.clock_mode != "token":
            return None
        renderer = getattr(self.task, "render_current_frame", None)
        if not callable(renderer):
            return None
        try:
            with operation_budget(
                "frame render", ENVIRONMENT_BUDGET.clock_advance_s, clock=self.clocks.operational
            ):
                return renderer()
        except Exception as exc:
            self._invalid_run = {
                "category": "environment",
                "reason": "frame_render_failed",
                "detail": str(exc),
            }
            raise

    @staticmethod
    def _key_identity(key):
        aliases = {
            "left": "arrowleft",
            "right": "arrowright",
            "up": "arrowup",
            "down": "arrowdown",
            "control": "ctrl",
            "return": "enter",
            "esc": "escape",
            "del": "delete",
            "super": "meta",
            "cmd": "meta",
            "command": "meta",
            "win": "meta",
        }
        value = str(key).strip().lower()
        return aliases.get(value, value)

    @classmethod
    def _key_chord(cls, action):
        raw_items = action.parameters.get("keys")
        if raw_items is None:
            raw_items = [action.parameters.get("key", "")]
        parts = []
        seen = set()
        for item in raw_items:
            for part in str(item).split("+"):
                wire = part.strip()
                identity = cls._key_identity(wire)
                if wire and identity not in seen:
                    parts.append((identity, wire))
                    seen.add(identity)
        if not parts:
            raise ValueError("key action requires at least one key")
        return parts, "+".join(wire for _, wire in parts)

    def _held_modifiers(self, exclude=()):
        """Modifier identities currently held under the token clock, minus ``exclude``."""
        return [
            identity
            for identity in MODIFIER_KEYS
            if identity in self._held_keys and identity not in exclude
        ]

    def _clock_key_event(self, event_type, chord, held_modifiers):
        """Send one token-clock key event within its own budget."""
        with operation_budget(
            "key press" if event_type == "keyDown" else "key release",
            ENVIRONMENT_BUDGET.key_event_s,
            clock=self.clocks.operational,
        ):
            self.task.clock_key_event(event_type, chord, held_modifiers=held_modifiers)

    def release_keys(self, keys):
        """Token clock, for the kernel's shell: release these held keys, by identity, whose lease
        ended; a key that is not held is skipped."""
        self._release_clock_keys(keys)

    def _clock_key_up(self, identities):
        """Release these held token-clock keys in one keyUp, in the order they were pressed; a key
        that is not held is skipped."""
        wanted = set(identities)
        selected = [identity for identity in self._held_keys if identity in wanted]
        if not selected:
            return
        chord = "+".join(self._held_keys[identity]["wire"] for identity in selected)
        self._clock_key_event("keyUp", chord, held_modifiers=self._held_modifiers(exclude=selected))
        for identity in selected:
            self._held_keys.pop(identity, None)
        if self._repeat_key in selected:
            self._repeat_key = None

    def _release_clock_keys(self, identities):
        """_clock_key_up, invalidating the run when it fails."""
        try:
            self._clock_key_up(identities)
        except Exception as exc:
            self._invalid_run = {
                "category": "environment",
                "reason": "clock_key_release_failed",
                "detail": str(exc),
            }
            raise

    def _release_expired_clock_keys(self):
        expired = [
            identity
            for identity, item in self._held_keys.items()
            if item["expires_at_tick"] <= self._logical_tick + 1e-9
        ]
        self._release_clock_keys(expired)

    def _start_autorepeat(self, identity):
        """A fresh press of a non-modifier key makes it the (only) auto-repeating key."""
        if identity in MODIFIER_KEYS or identity not in self._held_keys:
            self._repeat_key = None
            return
        self._repeat_key = identity
        self._held_keys[identity]["repeat_origin_tick"] = self._logical_tick
        self._held_keys[identity]["repeats_sent"] = 0

    def _pending_repeat_events(self, step):
        """Auto-repeat presses due within the next ``step`` ticks, and the repeat count after them.

        Repeat k fires at ``origin + delay + k * interval``; computing every tick from the press
        instead of accumulating an interval keeps a repeat that sits on an advance boundary from
        drifting a few ulps below the next advance's start (which the frame scheduler rejects).
        A repeat on the end of this advance goes with the next one. When that end is on the page's
        frame grid this advance renders the frame there, and the game sees the repeat a frame later
        than a keyboard would deliver it (sandbox/browser/clock.py cdp_advance_rendered_frame).
        """
        item = self._held_keys.get(self._repeat_key) if self._repeat_key else None
        if not item or "repeat_origin_tick" not in item:
            return [], None
        delay = AUTOREPEAT_DELAY_S / self.d
        interval = AUTOREPEAT_INTERVAL_S / self.d
        limit = min(self._logical_tick + step, item["expires_at_tick"])
        held_modifiers = self._held_modifiers(exclude=(self._repeat_key,))
        events = []
        sent = int(item.get("repeats_sent", 0))
        while True:
            repeat_tick = item["repeat_origin_tick"] + delay + sent * interval
            if repeat_tick >= limit - 1e-9:
                break
            events.append(
                {
                    "at_frames": max(0.0, repeat_tick - self._logical_tick),
                    "event_type": "keyDown",
                    "key": item["wire"],
                    "auto_repeat": True,
                    "held_modifiers": held_modifiers,
                }
            )
            sent += 1
        return events, sent

    def _active_hold_snapshot(self):
        if self.clock_mode != "token":
            now = self.clocks.semantic.time()
            with self._wall_lock:
                return [
                    {
                        "key": item["wire"],
                        "remaining_seconds": round(max(0.0, item["expires_at"] - now), 3),
                    }
                    for item in self._wall_holds.values()
                ]
        return [
            {
                "key": item["wire"],
                "remaining_simulated_seconds": round(
                    max(
                        0.0,
                        (item["expires_at_tick"] - self._logical_tick) * self.d,
                    ),
                    9,
                ),
            }
            for item in self._held_keys.values()
        ]

    def _reject_page_reload(self, action):
        """Refuse the page-reload shortcuts in game tasks (see PAGE_RELOAD_KEYS)."""
        if getattr(self.task, "category", None) != "browser_game":
            return
        try:
            parts, chord = self._key_chord(action)
        except ValueError:
            return  # the guest rejects an empty chord with HTTP 400
        identities = {identity for identity, _ in parts}
        with self._wall_lock:
            held = set(self._held_keys) | set(self._wall_holds)
        if identities & PAGE_RELOAD_KEYS or (
            "r" in identities and (identities | held) & {"ctrl", "meta"}
        ):
            raise ActionRejected(
                f"page reload shortcut {chord!r} is disabled in game tasks: reloading the page "
                "restarts the game and discards this episode's progress"
            )

    def _reject_browser_shortcut(self, action):
        """Refuse chords that operate the browser or desktop instead of the page."""
        try:
            parts, chord = self._key_chord(action)
        except ValueError:
            return
        pressed = {identity for identity, _ in parts}
        with self._wall_lock:
            held = set(self._held_keys) | set(self._wall_holds)
        active = pressed | (held & set(MODIFIER_KEYS))
        for shortcut in BROWSER_SHORTCUTS:
            if shortcut <= active and shortcut - set(MODIFIER_KEYS) <= pressed:
                raise ActionRejected(
                    f"browser shortcut {chord!r} is disabled: only the page can be used, "
                    "not the browser's address bar, tabs, menus or windows"
                )

    def _reject_zoom_scroll(self, action):
        """Refuse Ctrl/Meta+wheel: Chrome zooms the page and every mark is displaced afterwards."""
        if action.action_type != ActionType.SCROLL:
            return
        modifier = (action.parameters or {}).get("modifier")
        pressed = {self._key_identity(part) for part in str(modifier or "").split("+")}
        with self._wall_lock:
            held = set(self._held_keys) | set(self._wall_holds)
        if (pressed | held) & {"ctrl", "meta"}:
            raise ActionRejected(
                "ctrl/meta+scroll is disabled: it zooms the browser instead of scrolling the page"
            )

    def _reject_outside_page(self, action):
        """Refuse pointer input outside the page viewport (browser toolbar, tabs, desktop)."""
        rect = getattr(self.scene_capture, "page_rect", None)
        if rect is None:
            return
        desktop, _ = normalize_action(action)
        params = desktop.parameters or {}
        left, top, right, bottom = rect
        for x_key, y_key in (("x", "y"), ("x1", "y1"), ("x2", "y2")):
            x, y = params.get(x_key), params.get(y_key)
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                continue
            if not (left <= x < right and top <= y < bottom):
                raise ActionRejected(
                    f"{action.action_type.value} lands outside the page: only the page content can "
                    "be used, not the browser's address bar, tabs or the desktop around it"
                )

    def _reject(self, action):
        """Run the _reject_* checks that apply to the action."""
        if action.action_type in (ActionType.KEY, ActionType.HOLD_KEY):
            self._reject_page_reload(action)
            self._reject_browser_shortcut(action)
        elif action.action_type.value in POINTER_ACTIONS:
            self._reject_zoom_scroll(action)
            self._reject_outside_page(action)

    def _ensure_page_focus(self):
        """Keep desktop key input on the page; returns True when focus had to be restored."""
        ensure = getattr(self.scene_capture, "ensure_page_focus", None)
        return bool(ensure()) if callable(ensure) else False

    def _count_as_held(self, parts):
        """Keys whose press or release failed may still be down on the page: count them as held,
        so that the cleanup releases them (a key held already keeps its lease). A key that was up
        gets that keyUp as an event too, but the cleanup comes after the scored snapshot is
        frozen."""
        for identity, wire in parts:
            self._held_keys.setdefault(
                identity, {"wire": wire, "expires_at_tick": self._logical_tick}
            )

    def _execute_clock_key(self, action):
        parts, chord = self._key_chord(action)
        requested_identities = {identity for identity, _ in parts}
        if action.action_type == ActionType.KEY:
            repeat = int(action.parameters.get("repeat", 1))
            overlapping = [identity for identity, _ in parts if identity in self._held_keys]
            # Modifiers held by an earlier hold_key stay down: they only flag the tap's events.
            inherited_modifiers = self._held_modifiers(exclude=requested_identities)
            for _ in range(repeat):
                try:
                    self._clock_key_event("keyDown", chord, held_modifiers=inherited_modifiers)
                    # A new press of a repeatable key takes over auto-repeat on a real keyboard;
                    # the tap is released before its first repeat could fire, so nothing repeats
                    # afterwards. Modifier keys do not repeat and leave the repeating key alone.
                    if any(identity not in MODIFIER_KEYS for identity, _ in parts):
                        self._repeat_key = None
                    # Games that poll key state once per draw (reading a keys-down map in draw())
                    # never see a keyDown+keyUp delivered inside one frozen instant. Render
                    # one frame with the key down before releasing it (~17 ms of virtual time).
                    self.wait_until_tick(self._logical_tick + TAP_HOLD_S / self.d)
                    self._clock_key_event("keyUp", chord, held_modifiers=inherited_modifiers)
                except Exception:
                    self._count_as_held(parts)
                    raise
            # A tap of an already-held key must not terminate its outstanding lease. The lease may
            # have expired during the rendered frame above, so only keys still held are pressed
            # again.
            restore_identities = [item for item in overlapping if item in self._held_keys]
            if restore_identities:
                restore = "+".join(self._held_keys[item]["wire"] for item in restore_identities)
                self._clock_key_event(
                    "keyDown",
                    restore,
                    held_modifiers=self._held_modifiers(exclude=restore_identities),
                )
                for identity in restore_identities:
                    if identity not in MODIFIER_KEYS:
                        self._start_autorepeat(identity)
            return {
                "logical_key": chord,
                "effective_chord": "+".join(
                    [
                        *(
                            self._held_keys[item]["wire"]
                            for item in inherited_modifiers
                            if item in self._held_keys
                        ),
                        chord,
                    ]
                ),
                "repeat": repeat,
                "active_holds": self._active_hold_snapshot(),
                "active_mouse_buttons": sorted(self._held_mouse_buttons),
            }

        duration = float(action.parameters.get("duration", 1.0))
        expires_at = self._logical_tick + duration / self.d
        inherited_modifiers = self._held_modifiers(exclude=requested_identities)
        # Keys that are already down are only extended: a keyboard cannot press a held key again,
        # and grid games move one cell per keydown, so a second press would be an extra move.
        new_parts = [
            (identity, wire) for identity, wire in parts if identity not in self._held_keys
        ]
        extended = [identity for identity, _ in parts if identity in self._held_keys]
        new_chord = "+".join(wire for _, wire in new_parts)
        if new_parts:
            try:
                self._clock_key_event("keyDown", new_chord, held_modifiers=inherited_modifiers)
                if duration <= 1e-9:
                    self._clock_key_event("keyUp", new_chord, held_modifiers=inherited_modifiers)
            except Exception:
                self._count_as_held(new_parts)
                raise
        if duration <= 1e-9:
            if new_parts:
                # The transient press took over auto-repeat and was released at once.
                self._repeat_key = None
        else:
            for identity, wire in parts:
                current = self._held_keys.get(identity)
                self._held_keys[identity] = {
                    **(current or {}),
                    "wire": current["wire"] if current else wire,
                    "expires_at_tick": max(
                        current["expires_at_tick"] if current else self._logical_tick,
                        expires_at,
                    ),
                }
            for identity, _ in new_parts:
                if identity not in MODIFIER_KEYS:
                    self._start_autorepeat(identity)
            self.last_held_keys = tuple(identity for identity, _ in parts)
        return {
            "logical_key": chord,
            "effective_chord": "+".join(
                [
                    *(
                        self._held_keys[item]["wire"]
                        for item in inherited_modifiers
                        if item in self._held_keys
                    ),
                    chord,
                ]
            ),
            "pressed_keys": [wire for _, wire in new_parts],
            "extended_keys": [
                self._held_keys[item]["wire"] for item in extended if item in self._held_keys
            ],
            "hold_ticks": duration / self.d,
            "duration_simulated_seconds": duration,
            "auto_repeat": {
                "delay_seconds": AUTOREPEAT_DELAY_S,
                "interval_seconds": AUTOREPEAT_INTERVAL_S,
                "repeating_key": (
                    self._held_keys[self._repeat_key]["wire"]
                    if self._repeat_key in self._held_keys
                    else None
                ),
            },
            "active_holds": self._active_hold_snapshot(),
            "active_mouse_buttons": sorted(self._held_mouse_buttons),
        }

    # ---- wall-clock holds -------------------------------------------------------------------
    def _wall_command(self, argv):
        result = self.g.execute_command(shlex.join(argv), timeout=10)
        if not isinstance(result, dict) or result.get("returncode") != 0:
            detail = (result or {}).get("stderr") if isinstance(result, dict) else result
            raise RuntimeError(f"{argv[0]} {argv[1]} failed: {str(detail).strip()[:200]}")
        return result

    @contextmanager
    def _wall_commands_within_budget(self):
        """Hold ``_wall_commands``. The release timer holds it during its guest request, which an
        operation waits for no longer than its budget lasts."""
        deadline = Deadline(None, "wall key commands")
        left = deadline.remaining()
        if not self._wall_commands.acquire(timeout=-1 if left is None else left):
            raise deadline.exceeded()
        try:
            yield
        finally:
            self._wall_commands.release()

    def _wall_release(self, identities):
        """Release the given wall-clock leases (caller holds ``_wall_commands``)."""
        with self._wall_lock:
            selected = [identity for identity in identities if identity in self._wall_holds]
            keysyms = [self._wall_holds[identity]["keysym"] for identity in selected]
        if not selected:
            return
        self._wall_command(["xdotool", "keyup", "--", *keysyms])
        with self._wall_lock:
            for identity in selected:
                self._wall_holds.pop(identity, None)

    def _wall_release_due(self, generation):
        invalid = None
        with self._wall_commands:
            with self._wall_lock:
                if self._wall_closed or generation != self._wall_timer_generation:
                    return
                now = self.clocks.semantic.time()
                expired = [
                    identity
                    for identity, item in self._wall_holds.items()
                    if not item.get("stuck") and item["expires_at"] <= now + 1e-3
                ]
            try:
                self._wall_release(expired)
            except Exception as exc:
                # A key the guest cannot release stays physically down (and auto-repeating). Retry
                # briefly with a floor between attempts; then treat the stuck key as an
                # environment failure instead of hammering the guest from this timer forever, and
                # leave it to the cleanup.
                with self._wall_lock:
                    self._wall_release_failures += 1
                    if self._wall_release_failures < WALL_RELEASE_MAX_ATTEMPTS:
                        self._schedule_wall_release(min_delay=WALL_RELEASE_RETRY_S)
                        return
                    for identity in expired:
                        self._wall_holds[identity]["stuck"] = True
                    if not getattr(self, "_invalid_run", None):
                        self._invalid_run = invalid = {
                            "category": "environment",
                            "reason": "wall_key_release_failed",
                            "detail": str(exc),
                        }
            else:
                with self._wall_lock:
                    self._wall_release_failures = 0
            with self._wall_lock:
                self._schedule_wall_release()
        # On this timer's thread and outside the locks: the callback is the shell's.
        if invalid is not None and self.on_invalid is not None:
            self.on_invalid(invalid)

    def _schedule_wall_release(self, min_delay=0.0):
        """Arm one timer for the earliest outstanding wall lease (caller holds ``_wall_lock``).

        A timer that fired already cannot be cancelled and may be waiting for _wall_commands: the
        generation it was armed with tells it that it has been replaced."""
        if self._wall_timer is not None:
            self._wall_timer.cancel()
            self._wall_timer = None
        self._wall_timer_generation += 1
        due = [item["expires_at"] for item in self._wall_holds.values() if not item.get("stuck")]
        if self._wall_closed or not due:
            return
        clock = self.clocks.semantic
        delay = max(float(min_delay), min(due) - clock.time())
        generation = self._wall_timer_generation
        self._wall_timer = clock.timer(delay, lambda: self._wall_release_due(generation))

    def _ensure_wall_autorepeat(self):
        """Pin the wall display's auto-repeat to the values the token clock emulates (once).

        The sandbox desktop runs xfsettingsd, which applies XFCE's keyboard settings to the X
        server; pinning makes the wall clock independent of that daemon's defaults and records
        whether the pin took effect.
        """
        if self._wall_autorepeat is not None:
            return self._wall_autorepeat
        delay_ms = round(AUTOREPEAT_DELAY_S * 1000.0)
        rate_hz = round(1.0 / AUTOREPEAT_INTERVAL_S)
        try:
            self._wall_command(["xset", "r", "rate", str(delay_ms), str(rate_hz)])
            self._wall_autorepeat = {"pinned": True, "delay_ms": delay_ms, "rate_hz": rate_hz}
        except Exception as exc:
            self._wall_autorepeat = {
                "pinned": False,
                "delay_ms": delay_ms,
                "rate_hz": rate_hz,
                "detail": str(exc)[:200],
            }
        return self._wall_autorepeat

    def _execute_wall_hold(self, action, release_at=None):
        """Press the chord now and release it when the lease expires; never block the loop.

        The lease lasts ``duration`` from the press, or to ``release_at`` (semantic seconds since
        the origin) when the kernel's shell gives it, but at least min(duration, TAP_HOLD_S)
        after its keydown returned, past that time when the press came that late: TAP_HOLD_S is
        the token clock's tap, 1/60 s (the guest's wall-clock tap is 1/30 s), and a press never
        holds longer late than on time.
        """
        parts, chord = self._key_chord(action)
        duration = float(action.parameters.get("duration", 1.0))
        if not math.isfinite(duration) or duration <= 0:
            return None
        keysyms = {identity: _xdotool_keysym(identity) for identity, _ in parts}
        if any(value is None for value in keysyms.values()):
            return None
        with self._wall_commands_within_budget():
            autorepeat = self._ensure_wall_autorepeat()
            now = self.clocks.semantic.time()
            with self._wall_lock:
                new_parts = [
                    (identity, wire) for identity, wire in parts if identity not in self._wall_holds
                ]
            if new_parts:
                try:
                    self._wall_command(
                        [
                            "xdotool",
                            "keydown",
                            "--",
                            *(keysyms[identity] for identity, _ in new_parts),
                        ]
                    )
                except Exception:
                    self._count_as_held_on_the_wall(action)
                    raise
            pressed_at = self.clocks.semantic.time()
            expires_at = (
                now + duration
                if release_at is None
                else max(self.t0 + release_at, pressed_at + min(duration, TAP_HOLD_S))
            )
            with self._wall_lock:
                for identity, wire in parts:
                    current = self._wall_holds.get(identity)
                    self._wall_holds[identity] = {
                        "wire": current["wire"] if current else wire,
                        "keysym": keysyms[identity],
                        "expires_at": max(current["expires_at"] if current else now, expires_at),
                    }
                self._schedule_wall_release()
            result = {
                "status": "ok",
                "timestamp": now,
                "logical_key": chord,
                "pressed_keys": [wire for _, wire in new_parts],
                "extended_keys": [
                    wire for identity, wire in parts if identity not in dict(new_parts)
                ],
                "duration_seconds": duration,
                "non_blocking": True,
                "display_autorepeat": autorepeat,
                "active_holds": self._active_hold_snapshot(),
                "active_mouse_buttons": sorted(self._held_mouse_buttons),
            }
            if release_at is not None:
                # The lease this hold set, from its keydown request to the release the timer will
                # send: about how long its keys stay down, the keyup's trip to the desktop taking
                # about as long as the keydown's (longer on a retry). A key an earlier hold keeps
                # down longer goes up at that one's end.
                result["lease_seconds"] = round(expires_at - now, 3)
            return result

    def _count_as_held_on_the_wall(self, action):
        """The keys of an action whose press or release failed may still be down on the desktop:
        count those xdotool can name as held, so that the cleanup releases them (a key held
        already keeps its lease). The X server drops the release of a key that is up."""
        try:
            parts, _ = self._key_chord(action)
        except ValueError:
            return
        # The release timer may hold _wall_commands for its guest request: wait no longer than the
        # action's budget lasts, and record nothing without it (the run is invalid already).
        try:
            left = Deadline(None, "wall key commands").remaining()
        except BudgetExceeded:
            left = 0.0
        if not self._wall_commands.acquire(timeout=-1 if left is None else left):
            return
        try:
            with self._wall_lock:
                now = self.clocks.semantic.time()
                for identity, wire in parts:
                    keysym = _xdotool_keysym(identity)
                    if keysym is not None:
                        self._wall_holds.setdefault(
                            identity, {"wire": wire, "keysym": keysym, "expires_at": now}
                        )
        finally:
            self._wall_commands.release()

    def _restore_wall_holds(self, action):
        """Press wall-held keys again after a guest action released them.

        A `key` tap goes to the guest as a press and release, which ends the X hold of any key
        in its chord; the token path re-presses such keys so the lease survives, and the wall
        path must do the same. Returns the wires pressed again, if any.
        """
        try:
            parts, _ = self._key_chord(action)
        except ValueError:
            return []
        identities = [identity for identity, _ in parts]
        # Only actions add leases and the release timer only ends them: a chord with no key held
        # now has none to press again.
        with self._wall_lock:
            if not any(identity in self._wall_holds for identity in identities):
                return []
        with self._wall_commands_within_budget():
            with self._wall_lock:
                selected = [
                    self._wall_holds[identity]
                    for identity in identities
                    if identity in self._wall_holds
                ]
            if not selected:
                return []
            self._wall_command(
                ["xdotool", "keydown", "--", *(item["keysym"] for item in selected)]
            )
            return [item["wire"] for item in selected]

    def _release_wall_holds(self):
        with self._wall_lock:
            self._wall_closed = True
            if self._wall_timer is not None:
                self._wall_timer.cancel()
                self._wall_timer = None
        with self._wall_commands:
            self._wall_release(list(self._wall_holds))

    def _release_mouse_inputs(self):
        """Release each held mouse button on its own: one that fails stays held, and the failures
        are raised together once every button was tried."""
        errors = []
        for button in list(self._held_mouse_buttons):
            try:
                self.g.execute_action(Action(ActionType.MOUSE_UP, {"button": button}))
            except Exception as exc:  # noqa: BLE001 - the other buttons are still released
                errors.append(exc)
            else:
                self._held_mouse_buttons.discard(button)
        if errors:
            raise ExceptionGroup("held mouse buttons were not released", errors)

    def release_inputs(self):
        """Best-effort episode cleanup for every stateful input primitive. It runs once the
        outcome is latched: a failure raises for the caller to log and leaves the run as it was."""
        try:
            if self.clock_mode == "token" and hasattr(self.task, "clock_key_event"):
                self._clock_key_up(list(self._held_keys))
            if self.clock_mode != "token":
                self._release_wall_holds()
        finally:
            self._release_mouse_inputs()

    @staticmethod
    def _refusal(exc):
        """The refusal of the model's action, the runtime's (ActionRejected) or the guest agent's
        (GuestRefused), if ``exc`` is one.

        The guest validates every action body (unknown key names such as ``ctrl+plus``,
        unexpected parameters, ...) and answers 400 on /action. That is the model's action being
        rejected, not an environment failure, so it must not invalidate the run. The guest's
        refusal that comes after the action's budget ran out is one too, and the BudgetExceeded
        keeps it as its cause: the late answer still shows that the guest works. The runtime's own
        checks wait for nothing and refuse first. A 4xx from another endpoint, such as the
        /execute of the xdotool commands, does not show that the environment works, and the run
        is invalid.
        """
        if isinstance(exc, (ActionRejected, GuestRefused)):
            return exc
        if isinstance(exc.__cause__, GuestRefused):
            return exc.__cause__
        return None

    def _execution_failed(self, exc):
        self.failed += 1
        refusal = self._refusal(exc)
        # The model is told the refusal's own words; the execution result keeps the whole failure.
        self._last_exec_error = str(exc if refusal is None else refusal)
        self._last_rejected_by_guest = refusal is not None
        if not self._last_rejected_by_guest and not getattr(self, "_invalid_run", None):
            self._invalid_run = {
                "category": "environment",
                "reason": "action_execution_failed",
                "detail": str(exc),
            }
        self._last_execution_result = {"error": str(exc)}
        self._last_coordinate_trace = None
        return "failed"

    def execute(self, action, *, release_at=None):
        """Apply one action and return its lifecycle status to the scheduler. The kernel's shell
        gives a wall-clock hold_key the time to release its keys, ``release_at``."""
        self.last_held_keys = ()
        self.last_end_checks = ()
        action_type = action.action_type
        if action_type in (ActionType.DONE, ActionType.FAIL):
            if self.honor_fail:
                self._done = True
                self._terminal_action = action_type
                return "terminal"
            return "terminal_ignored"
        try:
            if (
                self.clock_mode == "token"
                and hasattr(self.task, "clock_key_event")
                and action_type in (ActionType.KEY, ActionType.HOLD_KEY)
            ):
                self._reject(action)
                self._last_execution_result = self._execute_clock_key(action)
                self.actions += 1
                return "applied"
            budget = action_timeout(action, ENVIRONMENT_BUDGET.action_s)
            with operation_budget(
                f"{action_type.value} action", budget, clock=self.clocks.operational
            ):
                # On the wall clock the hold, the re-press after a guest tap and counting the keys
                # of a failed guest request as held wait for the release timer's guest request in
                # flight only while the budget lasts; the key and scroll checks and the hold
                # snapshot below read the leases without waiting for it.
                self._reject(action)
                # Remaining keyboard input travels through the desktop to whatever Chrome focuses.
                focus_restored = action_type in (
                    ActionType.KEY,
                    ActionType.HOLD_KEY,
                    ActionType.TYPE,
                ) and self._ensure_page_focus()
                begin_barrier = getattr(self.task, "begin_gui_action", None)
                complete_barrier = getattr(self.task, "complete_gui_action", None)
                barrier_snapshot = (
                    begin_barrier()
                    if callable(begin_barrier) and callable(complete_barrier)
                    else None
                )
                result = None
                if self.clock_mode != "token" and action_type == ActionType.HOLD_KEY:
                    # Keys xdotool cannot name and zero-length holds fall through to the guest,
                    # which validates them (HTTP 400 for unsupported names).
                    result = self._execute_wall_hold(action, release_at)
                if result is None:
                    wall_key = self.clock_mode != "token" and action_type in (
                        ActionType.KEY,
                        ActionType.HOLD_KEY,
                    )
                    try:
                        result = self.g.execute_action(action)
                    except Exception as exc:
                        # The guest presses and releases these keys itself; a refusal pressed none.
                        if wall_key and self._refusal(exc) is None:
                            self._count_as_held_on_the_wall(action)
                        raise
                    if wall_key:
                        restored = self._restore_wall_holds(action)
                        if restored:
                            if not isinstance(result, dict):
                                result = {"guest_result": result}
                            result["restored_wall_holds"] = restored
                if barrier_snapshot is not None:
                    barrier_result = complete_barrier(barrier_snapshot)
                    if not isinstance(result, dict):
                        result = {"guest_result": result}
                    result["business_action_barrier"] = barrier_result
                active_holds = self._active_hold_snapshot()
            if focus_restored:
                if not isinstance(result, dict):
                    result = {"guest_result": result}
                result["page_focus_restored"] = True
            self._last_execution_result = result
            self._last_coordinate_trace = (result or {}).get("coordinate_trace")
            if action_type == ActionType.MOUSE_DOWN:
                self._held_mouse_buttons.add(action.parameters.get("button", "left"))
            elif action_type == ActionType.MOUSE_UP:
                self._held_mouse_buttons.discard(action.parameters.get("button", "left"))
            if isinstance(self._last_execution_result, dict):
                self._last_execution_result["active_holds"] = active_holds
                self._last_execution_result["active_mouse_buttons"] = sorted(
                    self._held_mouse_buttons
                )
            self.actions += 1
            return "applied"
        except Exception as exc:
            return self._execution_failed(exc)

    def record_lifecycle(self, record):
        if self.action_audit_path is None:
            return
        with self._audit_lock:
            with self.action_audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def check_end(self):
        """Ask the task whether the episode is over, for the kernel's shell: ``(ended, reason)``.

        ``_invalid_run`` is neither read nor written: a failed read raises, and the kernel counts
        it (evaluation/shell.py). An end is latched in ``_episode_over`` and ``_terminal_reason``.
        On the wall clock one read has its own, shorter budget.
        """
        if self.stop_check is None:
            return False, None
        budget = (
            ENVIRONMENT_BUDGET.state_read_s
            if self.clock_mode == "token"
            else ENVIRONMENT_BUDGET.wall_state_read_s
        )
        with operation_budget("state read", budget, clock=self.clocks.operational):
            terminal = self.stop_check()
        if not terminal:
            return False, None
        self._episode_over = True
        reason = terminal if isinstance(terminal, str) else None
        if reason is not None:
            self._terminal_reason = reason
        return True, reason


# Ports
"""Lane-local task port allocation for concurrent sandboxes on one host."""

import os as os


PORT_OFFSET_ENV = "TASK_PORT_OFFSET"


def task_port_offset() -> int:
    """Return the launcher-assigned port offset shared by one task process."""
    raw = os.environ.get(PORT_OFFSET_ENV, "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{PORT_OFFSET_ENV} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{PORT_OFFSET_ENV} must be non-negative, got {value}")
    return value


def isolated_task_port(base_port: int) -> int:
    """Map a catalog-owned base port into the current lane's host-port block."""
    if isinstance(base_port, bool) or not isinstance(base_port, int):
        raise TypeError(f"base task port must be an integer, got {base_port!r}")
    offset = task_port_offset()
    port = base_port + offset
    if not 1 <= port <= 65535:
        raise ValueError(
            f"isolated task port is outside 1..65535: base={base_port}, "
            f"offset={offset}, resolved={port}"
        )
    return port
