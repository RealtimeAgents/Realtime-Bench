"""Browser virtual-time clocks, frame fences and action synchronization."""

from __future__ import annotations
import base64, io, json, math, time
from PIL import Image
from realtime_framework.sandbox.browser.cdp import client_for

_RENDER_FENCE_EXPRESSION = r"""(() => {
  let marker = document.getElementById('__benchmark_render_fence');
  if (!marker) {
    marker = document.createElement('div');
    marker.id = '__benchmark_render_fence';
    marker.setAttribute('aria-hidden', 'true');
    Object.assign(marker.style, {
      position: 'fixed', left: '16px', top: '16px', width: '8px', height: '8px',
      zIndex: '2147483647', pointerEvents: 'none', margin: '0', padding: '0',
      border: '0', boxShadow: 'none'
    });
    document.documentElement.appendChild(marker);
  }
  const seq = (Number(marker.dataset.seq) || 0) + 1;
  const rgb = seq % 2 ? [255, 0, 255] : [0, 255, 255];
  marker.dataset.seq = String(seq);
  marker.style.backgroundColor = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
  void marker.offsetHeight;
  return {seq, rgb};
})()"""


def captured_frame_has_render_fence(screenshot: bytes, fence) -> bool:
    """Return whether a desktop screenshot contains the requested compositor fence."""
    if not isinstance(fence, dict) or not isinstance(fence.get("rgb"), list):
        return True
    try:
        image = Image.open(io.BytesIO(screenshot)).convert("RGB")
        rgb = tuple(int(channel) for channel in fence["rgb"])
        size = 8
        raw = image.tobytes()
        stride = image.width * 3
        needle = bytes(rgb) * size
        for y in range(image.height - size + 1):
            row = raw[y * stride : (y + 1) * stride]
            start = 0
            while True:
                offset = row.find(needle, start)
                if offset < 0:
                    break
                if offset % 3 == 0 and all(
                    raw[(y + dy) * stride + offset : (y + dy) * stride + offset + len(needle)]
                    == needle
                    for dy in range(size)
                ):
                    return True
                start = offset + 1
        return False
    except (OSError, ValueError, TypeError):
        return False


def _encoded_frame_has_render_fence(data: str, fence) -> bool:
    try:
        return captured_frame_has_render_fence(base64.b64decode(data), fence)
    except (ValueError, TypeError):
        return False


_RAF_CLOCK_SHIM = r"""
(() => {
  if (window.__benchmarkClock) return;
  const nativeRAF = window.requestAnimationFrame.bind(window);
  const nativeCancel = window.cancelAnimationFrame.bind(window);
  let controlled = false, nextId = 1, renderedFrames = 0, skippedEarlySteps = 0;
  let frameIntervalMs = 1000 / 60, nextFrameTimestamp = null;
  // Set once a parent frame's shim drives this frame; the parent's grid then decides when this
  // frame's steps are due, whoever requests them.
  let adopted = false;
  const pending = new Map();
  // A document created while the parent frame already holds the clock (the wrapper reloads its
  // iframe on reset, and some games reload themselves on death) must not run a single native
  // frame; the parent's next start()/step() gives it a schedule.  A cross-origin parent is
  // unreadable, so such a frame keeps native behaviour.
  try {
    if (window.parent !== window && window.parent.__benchmarkClock
        && window.parent.__benchmarkClock.state().controlled) {
      controlled = true;
      adopted = true;
    }
  } catch (_) {}
  window.requestAnimationFrame = function(callback) {
    const id = nextId++;
    const entry = {callback, nativeId: null};
    pending.set(id, entry);
    if (!controlled) {
      entry.nativeId = nativeRAF((timestamp) => {
        if (!pending.delete(id)) return;
        callback(timestamp);
      });
    }
    return id;
  };
  window.cancelAnimationFrame = function(id) {
    const entry = pending.get(id);
    if (!entry) return;
    pending.delete(id);
    if (entry.nativeId !== null) nativeCancel(entry.nativeId);
  };
  const takeControl = () => {
    if (controlled) return;
    controlled = true;
    for (const entry of pending.values()) {
      if (entry.nativeId !== null) nativeCancel(entry.nativeId);
      entry.nativeId = null;
    }
  };
  // Shims of the same-origin child frames that exist right now.  Looked up on every call so a
  // reloaded iframe is picked up and a navigated-away document is dropped; cross-origin frames
  // throw on property access and are skipped, frames without the shim are skipped.
  const childClocks = () => {
    const clocks = [];
    for (let index = 0; index < window.length; index++) {
      let clock = null;
      try { clock = window[index].__benchmarkClock; } catch (_) { continue; }
      if (clock && typeof clock.attach === 'function') clocks.push(clock);
    }
    return clocks;
  };
  const attachChildren = (delayMs) => {
    let count = 0;
    for (const clock of childClocks()) count += clock.attach(frameIntervalMs, delayMs);
    return count;
  };
  Object.defineProperty(window, '__benchmarkClock', {value: {
    start(rafHz = 60) {
      const hz = Number(rafHz);
      if (!Number.isFinite(hz) || hz <= 0) throw new Error('RAF frequency must be positive');
      frameIntervalMs = 1000 / hz;
      takeControl();
      if (nextFrameTimestamp === null) {
        nextFrameTimestamp = performance.now() + frameIntervalMs;
      }
      const controlledChildFrames = attachChildren(nextFrameTimestamp - performance.now());
      return {controlled, pending: pending.size, renderedFrames,
              frameIntervalMs, nextFrameTimestamp, controlledChildFrames};
    },
    // Called by the parent frame's shim.  Each frame's RAF timestamps stay on that frame's own
    // performance.now() timeline (the timeline its native RAF used); ``delayMs`` is the signed
    // distance from the parent's performance.now() to the parent's frame timestamp, so a newly
    // adopted schedule is the parent's grid translated onto this frame's timeline and the child
    // then advances exactly one interval per parent step.  A child that already holds a schedule
    // keeps it.  Returns the number of controlled frames in this subtree.
    attach(intervalMs, delayMs) {
      frameIntervalMs = intervalMs;
      adopted = true;
      takeControl();
      if (nextFrameTimestamp === null) {
        nextFrameTimestamp = performance.now() + delayMs;
      }
      return 1 + attachChildren(delayMs);
    },
    step() {
      if (!controlled) throw new Error('benchmark clock is not active');
      if (nextFrameTimestamp === null) throw new Error('benchmark RAF schedule is not initialized');
      const currentTimestamp = performance.now();
      const timestamp = nextFrameTimestamp;
      // CDP virtual-time expiry and performance.now() are quantized independently. Allow their
      // sub-millisecond skew while still rejecting an early step relative to the 60 Hz schedule.
      const early = currentTimestamp + 0.5 < timestamp;
      if (early && !adopted) {
        throw new Error(`benchmark RAF step requested before its scheduled timestamp: current=${currentTimestamp}, next=${timestamp}, earlyBy=${timestamp - currentTimestamp}`);
      }
      let callbacks = 0;
      if (early) {
        // The parent's shim steps an adopted frame once per parent frame, below, after the
        // parent's own callbacks.  Page code may step it too: a game_config.js that runs a
        // parent-frame RAF bridge calling the child shim's step() from inside the parent's
        // callback batch makes the parent's forwarding arrive one interval early for a frame
        // that has already run.  Such a request is a no-op for that frame.  Only the top frame,
        // which the harness steps on its own grid, treats an early step as an error.
        skippedEarlySteps += 1;
      } else {
        const batch = Array.from(pending.values());
        pending.clear();
        for (const entry of batch) entry.callback(timestamp);
        callbacks = batch.length;
        renderedFrames += 1;
        nextFrameTimestamp = timestamp + frameIntervalMs;
      }
      // Browsers run a page's animation frame callbacks parent first, then each child document.
      let childCallbacks = 0, controlledChildFrames = 0, subtreeSkipped = skippedEarlySteps;
      for (const clock of childClocks()) {
        clock.attach(frameIntervalMs, timestamp - currentTimestamp);
        const result = clock.step();
        childCallbacks += result.callbacks + result.childCallbacks;
        controlledChildFrames += 1 + result.controlledChildFrames;
        subtreeSkipped += result.skippedEarlySteps;
      }
      return {callbacks, childCallbacks, controlledChildFrames, skippedEarlySteps: subtreeSkipped,
              pending: pending.size, renderedFrames, timestamp: early ? null : timestamp,
              performanceNow: performance.now(), frameIntervalMs, nextFrameTimestamp};
    },
    state() {
      let controlledChildFrames = 0, subtreeSkipped = skippedEarlySteps;
      for (const clock of childClocks()) {
        const child = clock.state();
        if (child.controlled) controlledChildFrames += 1 + child.controlledChildFrames;
        subtreeSkipped += child.skippedEarlySteps;
      }
      return {controlled, pending: pending.size, renderedFrames,
              frameIntervalMs, nextFrameTimestamp, controlledChildFrames,
              skippedEarlySteps: subtreeSkipped};
    }
  }, configurable: false, enumerable: false, writable: false});
})();
"""

_GAME_PAGE_GUARD = r"""
(() => {
  const arm = (win) => {
    if (win.__benchmarkGamePageGuard) return;
    win.__benchmarkGamePageGuard = true;
    win.addEventListener('contextmenu', (event) => event.preventDefault(), true);
    win.alert = () => undefined;
    win.confirm = () => false;
    win.prompt = () => null;
    // Every same-origin frame reachable through window.frames, recursively; a cross-origin frame
    // throws on property access and keeps Chrome's default.
    for (let index = 0; index < win.frames.length; index++) {
      try { arm(win.frames[index]); } catch (_) {}
    }
  };
  arm(window);
})();
"""


def cdp_eval(dbg_port, expr):
    """Read-only: evaluate a JS expression in the live game page and return its value."""
    try:
        return client_for(dbg_port).evaluate(expr)
    except Exception:
        return None


def cdp_install_document_script(dbg_port, source, url=None):
    """Run ``source`` in every document created from now on, then optionally navigate to ``url``.

    The script executes before the document's own scripts, in each same-origin frame and again
    after every reload, for as long as the persistent target session keeps the registration.
    """
    client = client_for(dbg_port)
    with client.transaction():
        _cdp_result(client.call("Page.enable"), "Page.enable")
        _cdp_result(
            client.call("Page.addScriptToEvaluateOnNewDocument", {"source": source}),
            "Page.addScriptToEvaluateOnNewDocument",
        )
        if url is not None:
            _cdp_result(client.call("Page.navigate", {"url": url}), "Page.navigate")
            client.wait_event("Page.loadEventFired")
    return {"ok": True}


def cdp_install_frame_clock(dbg_port, url):
    """Install the RAF clock before navigating to ``url``.

    The registration and later clock/input/state commands share one persistent target session.
    Reload-reset games retain the new-document script; the defensive reset path reinstalls it only
    if Chrome replaces the target or loses the registration.
    """
    client = client_for(dbg_port)
    with client.transaction():
        _cdp_result(client.call("Page.enable"), "Page.enable")
        _cdp_result(
            client.call(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": _RAF_CLOCK_SHIM},
            ),
            "Page.addScriptToEvaluateOnNewDocument",
        )
        _cdp_result(client.call("Page.navigate", {"url": url}), "Page.navigate")
        client.wait_event("Page.loadEventFired")
    return {"ok": True}


def cdp_install_game_page_guard(dbg_port):
    """Arm _GAME_PAGE_GUARD (no context menu, no native dialogs) in every game document.

    Registered for every document created from now on (reloads, frames created later) and
    evaluated once in the document that is already loaded: both clocks have navigated to the game
    before this runs.
    """
    client = client_for(dbg_port)
    with client.transaction():
        _cdp_result(client.call("Page.enable"), "Page.enable")
        _cdp_result(
            client.call(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": _GAME_PAGE_GUARD},
            ),
            "Page.addScriptToEvaluateOnNewDocument",
        )
        client.evaluate(_GAME_PAGE_GUARD)
    return {"ok": True}


def cdp_pause_virtual_time(dbg_port):
    response = client_for(dbg_port).call(
        "Emulation.setVirtualTimePolicy",
        {"policy": "pause"},
    )
    _cdp_result(response, "virtual_pause")
    return {"ok": True}


def cdp_start_frame_clock(dbg_port):
    clock = client_for(dbg_port).evaluate(
        "window.__benchmarkClock && window.__benchmarkClock.start()"
    )
    if clock is None:
        raise RuntimeError("clock shim missing")
    return {"ok": True, "clock": clock}


def _sorted_key_events(key_events, seconds):
    """Validate scheduled key events for one frame advance and return them in time order."""
    events = []
    for index, event in enumerate(key_events or ()):
        if not isinstance(event, dict):
            raise TypeError("key_events must be dicts")
        at = event.get("at_seconds")
        if (
            not isinstance(at, (int, float))
            or isinstance(at, bool)
            or not math.isfinite(at)
            or at < -1e-6
            or at > seconds + 1e-9
        ):
            raise ValueError("key event at_seconds must lie inside the advanced interval")
        # A repeat scheduled on the previous advance's boundary may arrive a few ulps early.
        at = max(0.0, float(at))
        if event.get("event_type") not in {"keyDown", "keyUp"}:
            raise ValueError("key event event_type must be keyDown or keyUp")
        if not isinstance(event.get("key"), str) or not event["key"]:
            raise ValueError("key event requires a key chord")
        events.append((float(at), index, event))
    events.sort(key=lambda item: (item[0], item[1]))
    return events


def cdp_advance_rendered_frame(dbg_port, seconds, raf_hz=60.0, key_events=()):
    """Advance the page's virtual time by ``seconds`` on the persistent rAF grid.

    ``key_events`` are dispatched at their ``at_seconds`` offset inside the same advance: virtual
    time is moved to the event, the key event is sent, and frame stepping resumes. This is how the
    token clock delivers a held key's auto-repeat presses between frames without a compositor
    commit per press.
    """
    client = client_for(dbg_port)
    seconds = float(seconds)
    raf_hz = float(raf_hz)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be a non-negative finite number")
    if not math.isfinite(raf_hz) or raf_hz <= 0:
        raise ValueError("raf_hz must be a positive finite number")
    pending_events = _sorted_key_events(key_events, seconds)
    callback_count = 0
    last = None
    substeps = 0
    dispatched = 0
    fence = None
    with client.transaction():
        schedule = client.evaluate(
            f"window.__benchmarkClock.start({json.dumps(raf_hz)})",
            await_promise=False,
        )
        current = client.evaluate("performance.now()", await_promise=False)
        if not isinstance(schedule, dict) or not isinstance(current, (int, float)):
            raise RuntimeError("RAF schedule is unavailable")
        origin = float(current)
        target = origin + seconds * 1000.0
        next_frame = schedule.get("nextFrameTimestamp")
        if not isinstance(next_frame, (int, float)):
            raise RuntimeError("RAF schedule has no next-frame timestamp")

        def advance_virtual_time(milliseconds):
            if milliseconds <= 1e-7:
                return
            _cdp_result(
                client.call(
                    "Emulation.setVirtualTimePolicy",
                    {
                        "policy": "advance",
                        "budget": milliseconds,
                        "maxVirtualTimeTaskStarvationCount": 10000,
                    },
                ),
                "virtual_frame",
            )
            client.wait_event("Emulation.virtualTimeBudgetExpired")

        def read_now():
            value = client.evaluate("performance.now()", await_promise=False)
            if not isinstance(value, (int, float)):
                raise RuntimeError("virtual clock did not expose its current timestamp")
            return float(value)

        def dispatch_due_key_events(limit):
            """Deliver every scheduled key event at or before ``limit`` (a virtual timestamp)."""
            nonlocal current, dispatched
            while pending_events and origin + pending_events[0][0] * 1000.0 <= limit + 1e-7:
                _, _, event = pending_events.pop(0)
                advance_virtual_time(max(0.0, origin + event["at_seconds"] * 1000.0 - current))
                current = read_now()
                _dispatch_key_events(
                    client,
                    event["event_type"],
                    event["key"],
                    auto_repeat=bool(event.get("auto_repeat")),
                    held_modifiers=event.get("held_modifiers") or (),
                )
                dispatched += 1

        # Advance to each point on one persistent 60-Hz grid, then run that frame's callbacks at
        # the reached timestamp. Segmenting a simulated second around an action therefore cannot add,
        # remove, or phase-shift physics updates. Scheduled key events land before the first frame
        # whose timestamp is at or after them, so the game processes them in the frame a real
        # keyboard would have delivered them to.
        while float(next_frame) <= target + 1e-7:
            dispatch_due_key_events(float(next_frame))
            advance_virtual_time(max(0.0, float(next_frame) - float(current)))
            current = read_now()
            last = client.evaluate("window.__benchmarkClock.step()", await_promise=True)
            if not isinstance(last, dict):
                raise RuntimeError("RAF step failed")
            callback_count += int(last.get("callbacks", 0)) + int(last.get("childCallbacks", 0))
            substeps += 1
            next_frame = last.get("nextFrameTimestamp")
            if not isinstance(next_frame, (int, float)):
                raise RuntimeError("RAF step did not schedule the next frame")
        dispatch_due_key_events(target)
        advance_virtual_time(max(0.0, target - float(current)))
        _cdp_result(
            client.call(
                "Emulation.setVirtualTimePolicy",
                {"policy": "pause"},
            ),
            "virtual_pause",
        )

        fence = client.evaluate(_RENDER_FENCE_EXPRESSION, await_promise=False)
        if not isinstance(fence, dict):
            raise RuntimeError("render fence installation failed")
        if last is None:
            last = client.evaluate(
                "window.__benchmarkClock.state()",
                await_promise=False,
            )

        data = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            client.evaluate(
                "document.documentElement && document.documentElement.offsetHeight",
                await_promise=False,
            )
            result = _cdp_result(
                client.call(
                    "Page.captureScreenshot",
                    {
                        "format": "png",
                        "fromSurface": True,
                    },
                ),
                "Page.captureScreenshot",
            )
            candidate = result.get("data")
            if candidate and _encoded_frame_has_render_fence(candidate, fence):
                data = candidate
                break
            time.sleep(0.02)
    if not data:
        raise RuntimeError("compositor did not publish the post-step render fence")
    return {
        "ok": True,
        "advanced_ms": seconds * 1000.0,
        "substeps": substeps,
        "raf_callbacks": callback_count,
        "key_events_dispatched": dispatched,
        "clock": last,
        "compositor_committed": True,
        "compositor_frame_bytes": len(data),
        "render_fence": fence,
    }


def cdp_sync_realtime_frame(dbg_port, benchmark_frame, url_contains):
    client = client_for(dbg_port, target_hint=url_contains)
    expected = float(benchmark_frame)
    expression = """(async () => {
      const runtime = globalThis.__realtimeRuntime;
      if (!runtime) throw new Error('realtime runtime is not ready');
      await runtime.poll();
      const state = runtime.state || {};
      return {benchmark_frame: state.benchmark_frame,
              generation: state.generation, version: state.version};
    })()"""
    with client.transaction():
        observed = client.evaluate(expression)
        if not isinstance(observed, dict):
            raise RuntimeError("realtime poll barrier failed")
        observed_frame = observed.get("benchmark_frame")
        if not isinstance(observed_frame, (int, float)) or abs(observed_frame - expected) > 1e-9:
            raise RuntimeError(
                f"browser rendered benchmark frame {observed_frame!r}, expected {expected!r}"
            )
        fence = client.evaluate(_RENDER_FENCE_EXPRESSION, await_promise=False)
        if not isinstance(fence, dict):
            raise RuntimeError("realtime render fence installation failed")
        data = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            client.evaluate(
                "document.documentElement && document.documentElement.offsetHeight",
                await_promise=False,
            )
            result = _cdp_result(
                client.call(
                    "Page.captureScreenshot",
                    {
                        "format": "png",
                        "fromSurface": True,
                    },
                ),
                "Page.captureScreenshot",
            )
            candidate = result.get("data")
            if candidate and _encoded_frame_has_render_fence(candidate, fence):
                data = candidate
                break
            time.sleep(0.02)
        if not data:
            raise RuntimeError("realtime compositor did not publish the render fence")
    return {
        "ok": True,
        "benchmark_frame": expected,
        "observed": observed,
        "compositor_committed": True,
        "compositor_frame_bytes": len(data),
        "render_fence": fence,
    }


def cdp_realtime_action_barrier_snapshot(dbg_port, url_contains):
    """Capture browser-side input/business counters immediately before desktop GUI dispatch."""
    client = client_for(dbg_port, target_hint=url_contains)
    expression = """(() => {
      const runtime = globalThis.__realtimeRuntime;
      if (!runtime) throw new Error('realtime runtime is not ready');
      return runtime.actionBarrierSnapshot();
    })()"""
    snapshot = client.evaluate(expression, timeout=8)
    if not isinstance(snapshot, dict):
        raise TypeError("realtime action barrier did not return a snapshot")
    for key in (
        "input_event_sequence",
        "business_action_started",
        "business_action_completed",
    ):
        value = snapshot.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TypeError(f"realtime action barrier returned invalid {key}: {value!r}")
    return snapshot


def cdp_wait_realtime_action_barrier(dbg_port, url_contains, snapshot):
    """Wait for the dispatched desktop event and any resulting server mutation to finish."""
    client = client_for(dbg_port, target_hint=url_contains)
    baseline = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    expression = f"""(async () => {{
      const runtime = globalThis.__realtimeRuntime;
      if (!runtime) throw new Error('realtime runtime is not ready');
      return await runtime.waitForActionBarrier({baseline});
    }})()"""
    result = client.evaluate(expression, timeout=8)
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError(f"realtime action completion barrier failed: {result!r}")
    return result


def cdp_key_event(dbg_port, event_type, key, auto_repeat=False, held_modifiers=()):
    """Dispatch one key chord over CDP in its own transaction (see ``_dispatch_key_events``)."""
    client = client_for(dbg_port)
    with client.transaction():
        return _dispatch_key_events(
            client,
            event_type,
            key,
            auto_repeat=auto_repeat,
            held_modifiers=held_modifiers,
        )


def _dispatch_key_events(client, event_type, key, auto_repeat=False, held_modifiers=()):
    """Send ``event_type`` for every key in the '+'-joined chord.

    ``held_modifiers`` names modifier keys that are already down (held by an earlier action) and
    must be reflected in each event's ``modifiers`` bitmask without being pressed again.
    ``auto_repeat`` marks a keyDown as a keyboard auto-repeat (DOM ``event.repeat === true``).
    """
    key_map = {
        "arrowleft": ("ArrowLeft", "ArrowLeft", 37),
        "left": ("ArrowLeft", "ArrowLeft", 37),
        "arrowup": ("ArrowUp", "ArrowUp", 38),
        "up": ("ArrowUp", "ArrowUp", 38),
        "arrowright": ("ArrowRight", "ArrowRight", 39),
        "right": ("ArrowRight", "ArrowRight", 39),
        "arrowdown": ("ArrowDown", "ArrowDown", 40),
        "down": ("ArrowDown", "ArrowDown", 40),
        "space": (" ", "Space", 32),
        " ": (" ", "Space", 32),
        "enter": ("Enter", "Enter", 13),
        "return": ("Enter", "Enter", 13),
        "escape": ("Escape", "Escape", 27),
        "esc": ("Escape", "Escape", 27),
        "ctrl": ("Control", "ControlLeft", 17),
        "control": ("Control", "ControlLeft", 17),
        "shift": ("Shift", "ShiftLeft", 16),
        "alt": ("Alt", "AltLeft", 18),
        "super": ("Meta", "MetaLeft", 91),
        "meta": ("Meta", "MetaLeft", 91),
        "cmd": ("Meta", "MetaLeft", 91),
        "command": ("Meta", "MetaLeft", 91),
        "win": ("Meta", "MetaLeft", 91),
        "tab": ("Tab", "Tab", 9),
        "backspace": ("Backspace", "Backspace", 8),
        "pageup": ("PageUp", "PageUp", 33),
        "pagedown": ("PageDown", "PageDown", 34),
        "end": ("End", "End", 35),
        "home": ("Home", "Home", 36),
        "insert": ("Insert", "Insert", 45),
        "delete": ("Delete", "Delete", 46),
        "del": ("Delete", "Delete", 46),
        "capslock": ("CapsLock", "CapsLock", 20),
        "pause": ("Pause", "Pause", 19),
        **{f"f{number}": (f"F{number}", f"F{number}", 111 + number) for number in range(1, 13)},
    }
    raw_keys = [" "] if key == " " else [value.strip() for value in key.split("+") if value.strip()]
    modifier_bits = _MODIFIER_BITS
    modifiers = 0
    for value in held_modifiers:
        modifiers |= modifier_bits.get(str(value).lower(), 0)
    if event_type == "keyUp":
        for value in raw_keys:
            modifiers |= modifier_bits.get(value.lower(), 0)
        raw_keys.reverse()
    for value in raw_keys:
        normalized = value.lower()
        bit = modifier_bits.get(normalized, 0)
        if event_type == "keyDown":
            modifiers |= bit
        else:
            modifiers &= ~bit
        if normalized in key_map:
            actual, code, virtual_key = key_map[normalized]
        elif len(value) == 1 and value.isalpha():
            actual, code, virtual_key = value.lower(), "Key" + value.upper(), ord(value.upper())
        elif len(value) == 1 and value.isdigit():
            actual, code, virtual_key = value, "Digit" + value, ord(value)
        else:
            actual, code, virtual_key = value, value, 0
        params = {
            "type": event_type,
            "key": actual,
            "code": code,
            "windowsVirtualKeyCode": virtual_key,
            "nativeVirtualKeyCode": virtual_key,
            "modifiers": modifiers,
        }
        if auto_repeat and event_type == "keyDown":
            params["autoRepeat"] = True
        _cdp_result(client.call("Input.dispatchKeyEvent", params), "key_event")
    return {"ok": True}


def _cdp_result(response, operation):
    if response.get("error"):
        raise RuntimeError(f"CDP {operation} failed: {response['error']}")
    return response.get("result", {})


_MODIFIER_BITS = {"alt": 1, "ctrl": 2, "control": 2, "meta": 4, "super": 4, "shift": 8}
