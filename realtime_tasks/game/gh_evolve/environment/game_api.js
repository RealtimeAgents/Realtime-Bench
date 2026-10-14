(function () {
  "use strict";

  const config = window.OPEN_SOURCE_GAME_CONFIG || {};
  const frame = document.getElementById("game-frame");
  const sampleCanvas = document.createElement("canvas");
  sampleCanvas.width = 32;
  sampleCanvas.height = 18;
  const sampleContext = sampleCanvas.getContext("2d", { willReadFrequently: true });

  let generation = 0;
  // assets/benchmark_seed.js, registered by the runtime, seeds every document with the task seed.
  const seed = Number.isInteger(window.__benchmarkSeed) ? window.__benchmarkSeed : null;
  let loaded = false;
  let startedAt = performance.now();
  let lastPollAt = startedAt;
  let lastSignature = null;
  let lastDynamicAt = 0;
  let lastTrustedInputAt = 0;
  let dynamicChanges = 0;
  let trustedInputs = 0;
  let activePlayMs = 0;
  let terminal = false;
  let terminalReason = null;
  let startTimers = [];

  function childWindow() {
    try { return frame.contentWindow; } catch (_) { return null; }
  }

  function childDocument() {
    try { return frame.contentDocument || (frame.contentWindow && frame.contentWindow.document); }
    catch (_) { return null; }
  }

  function keyEvent(type, code, key, keyCode) {
    const w = childWindow();
    const d = childDocument();
    if (!w || !d) return;
    const dispatch = (target) => {
      if (!target) return;
      const event = new w.KeyboardEvent(type, {
        key: key || code, code: code, bubbles: true, cancelable: true
      });
      for (const field of ["keyCode", "which", "charCode"]) {
        try { Object.defineProperty(event, field, { get: () => keyCode || 0 }); } catch (_) {}
      }
      target.dispatchEvent(event);
    };
    dispatch(w);
    dispatch(d);
    dispatch(d.body);
  }

  function press(code, key, keyCode, duration) {
    keyEvent("keydown", code, key, keyCode);
    window.setTimeout(() => keyEvent("keyup", code, key, keyCode), duration || 90);
  }

  function clickSelector(selector) {
    const d = childDocument();
    const el = d && d.querySelector(selector);
    if (!el) return false;
    try { el.click(); } catch (_) {}
    return true;
  }

  function clickAt(xRatio, yRatio) {
    const w = childWindow();
    const d = childDocument();
    if (!w || !d) return false;
    const x = Math.max(1, Math.floor(w.innerWidth * xRatio));
    const y = Math.max(1, Math.floor(w.innerHeight * yRatio));
    const target = d.elementFromPoint(x, y) || d.querySelector("canvas") || d.body;
    if (!target) return false;
    for (const type of ["mousemove", "mousedown", "mouseup", "click"]) {
      target.dispatchEvent(new w.MouseEvent(type, {
        bubbles: true, cancelable: true, view: w, clientX: x, clientY: y,
        button: 0, buttons: type === "mousedown" ? 1 : 0
      }));
    }
    return true;
  }

  function call(name) {
    const w = childWindow();
    if (!w) return false;
    const parts = String(name).split(".");
    let receiver = w;
    let value = w;
    for (const part of parts) {
      receiver = value;
      value = value && value[part];
    }
    if (typeof value !== "function") return false;
    try { value.call(receiver); return true; } catch (_) { return false; }
  }

  function starterContext() {
    return {
      window: childWindow,
      document: childDocument,
      press,
      clickSelector,
      clickAt,
      call,
      later(delay, fn) {
        const id = window.setTimeout(() => {
          try { fn(starterContext()); } catch (_) {}
        }, delay);
        startTimers.push(id);
      }
    };
  }

  function noteTrustedInput(event) {
    if (!event || !event.isTrusted) return;
    trustedInputs += 1;
    lastTrustedInputAt = performance.now();
  }

  function attachInputObservers() {
    const w = childWindow();
    const d = childDocument();
    if (!w || !d) return;
    for (const type of ["keydown", "pointerdown", "mousedown", "touchstart"]) {
      w.addEventListener(type, noteTrustedInput, true);
      d.addEventListener(type, noteTrustedInput, true);
    }
    try { frame.focus(); w.focus(); if (d.body) d.body.focus(); } catch (_) {}
  }

  function hashBytes(bytes, seed) {
    let hash = seed >>> 0;
    for (let i = 0; i < bytes.length; i += 4) {
      hash ^= bytes[i] + (bytes[i + 1] << 8) + (bytes[i + 2] << 16) + (bytes[i + 3] << 24);
      hash = Math.imul(hash, 16777619) >>> 0;
    }
    return hash;
  }

  function visualSignature() {
    const d = childDocument();
    const w = childWindow();
    if (!d || !w || !d.body) return null;
    let hash = 2166136261;
    const text = (d.body.innerText || "").replace(/\s+/g, " ").slice(0, 12000);
    for (let i = 0; i < text.length; i += 3) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 16777619) >>> 0;
    }

    const canvases = Array.from(d.querySelectorAll("canvas"))
      .filter((canvas) => canvas.width > 0 && canvas.height > 0)
      .sort((a, b) => b.width * b.height - a.width * a.height)
      .slice(0, 3);
    for (const canvas of canvases) {
      try {
        sampleContext.clearRect(0, 0, 32, 18);
        sampleContext.drawImage(canvas, 0, 0, 32, 18);
        hash = hashBytes(sampleContext.getImageData(0, 0, 32, 18).data, hash);
      } catch (_) {}
    }

    const elements = Array.from(d.querySelectorAll("img, canvas, svg, [style]"))
      .filter((el) => {
        const rect = el.getBoundingClientRect();
        return rect.width > 3 && rect.height > 3 && rect.bottom > 0 && rect.right > 0;
      }).slice(0, 100);
    for (const el of elements) {
      const rect = el.getBoundingClientRect();
      const style = w.getComputedStyle(el);
      const values = [rect.left, rect.top, rect.width, rect.height, style.transform, style.opacity];
      const token = values.join("|");
      for (let i = 0; i < token.length; i += 2) {
        hash ^= token.charCodeAt(i);
        hash = Math.imul(hash, 16777619) >>> 0;
      }
    }

    // Older Phaser releases often use WebGL canvases without preserveDrawingBuffer, which makes
    // pixels unreadable between frames. Sample live display-object transforms as an additional
    // observation channel; this reads engine state but never changes it.
    try {
      const games = w.Phaser && w.Phaser.GAMES;
      const game = games && Array.from(games).find(Boolean);
      if (game && game.world) {
        const queue = [game.world];
        let seen = 0;
        while (queue.length && seen < 160) {
          const node = queue.shift();
          if (!node) continue;
          seen += 1;
          const token = [
            node.x, node.y, node.angle, node.rotation, node.frame,
            node.visible, node.alive, node.exists,
            node.tilePosition && node.tilePosition.x,
            node.tilePosition && node.tilePosition.y
          ].join("|");
          for (let i = 0; i < token.length; i += 2) {
            hash ^= token.charCodeAt(i);
            hash = Math.imul(hash, 16777619) >>> 0;
          }
          if (node.children) queue.push(...Array.from(node.children).slice(0, 80));
        }
      }
    } catch (_) {}

    if (typeof config.signal === "function") {
      try {
        const token = JSON.stringify(config.signal(starterContext()));
        for (let i = 0; i < token.length; i += 2) {
          hash ^= token.charCodeAt(i);
          hash = Math.imul(hash, 16777619) >>> 0;
        }
      } catch (_) {}
    }
    return hash.toString(16);
  }

  function terminalState() {
    if (typeof config.terminal !== "function") return null;
    try { return config.terminal(starterContext()) || null; } catch (_) { return null; }
  }

  function poll() {
    const now = performance.now();
    const elapsed = Math.min(1000, Math.max(0, now - lastPollAt));
    lastPollAt = now;
    if (!loaded) return;

    if (typeof config.tick === "function") {
      try { config.tick(starterContext()); } catch (_) {}
    }

    const signature = visualSignature();
    if (signature !== null && lastSignature !== null && signature !== lastSignature) {
      dynamicChanges += 1;
      lastDynamicAt = now;
    }
    lastSignature = signature;

    const result = terminalState();
    if (result) {
      terminal = true;
      terminalReason = typeof result === "string" ? result : "upstream_game_over";
    }

    const visualIsLive = dynamicChanges >= 2 && now - lastDynamicAt < 1800;
    const playerIsActive = now - lastTrustedInputAt < 3500;
    if (!terminal && visualIsLive && playerIsActive) activePlayMs += elapsed;
  }

  function clearStartTimers() {
    for (const id of startTimers) window.clearTimeout(id);
    startTimers = [];
  }

  function loadFresh() {
    generation += 1;
    clearStartTimers();
    loaded = false;
    startedAt = performance.now();
    lastPollAt = startedAt;
    lastSignature = null;
    lastDynamicAt = 0;
    lastTrustedInputAt = 0;
    dynamicChanges = 0;
    trustedInputs = 0;
    activePlayMs = 0;
    terminal = false;
    terminalReason = null;
    const separator = String(config.entry || "upstream/index.html").includes("?") ? "&" : "?";
    frame.src = (config.entry || "upstream/index.html") + separator + "benchmark_reset=" + generation;
  }

  frame.addEventListener("load", () => {
    loaded = true;
    startedAt = performance.now();
    lastPollAt = startedAt;
    attachInputObservers();
    if (typeof config.start === "function") {
      try { config.start(starterContext()); } catch (_) {}
    }
  });

  window.gameAPI = {
    schemaVersion: "2.0",
    init(options = {}) { return { ok: true, seed, source: config.source || null }; },
    reset(options = {}) {
      try {
        const w = childWindow();
        if (w && w.localStorage) w.localStorage.clear();
        if (w && w.sessionStorage) w.sessionStorage.clear();
      } catch (_) {}
      if (typeof config.reset === "function") {
        try { config.reset(starterContext()); } catch (_) {}
      }
      loadFresh();
      return new Promise((resolve) => {
        const deadline = performance.now() + 20000;
        const finishWhenReady = () => {
          let ready = loaded;
          if (ready && typeof config.ready === "function") {
            try { ready = Boolean(config.ready(starterContext())); } catch (_) { ready = false; }
          }
          if (ready || performance.now() >= deadline) {
            resolve({ ok: ready });
          } else {
            window.setTimeout(finishWhenReady, 100);
          }
        };
        window.setTimeout(finishWhenReady, 100);
      });
    },
    getState() {
      const now = performance.now();
      const visualIsLive = dynamicChanges >= 2 && now - lastDynamicAt < 1800;
      let nativeState = {};
      if (typeof config.state === "function") {
        try { nativeState = config.state(starterContext()) || {}; } catch (_) {}
      }
      const defaultActionable = loaded && dynamicChanges >= 2 && !terminal;
      const actionable = typeof nativeState.is_actionable === "boolean"
        ? nativeState.is_actionable && !terminal
        : defaultActionable;
      const seconds = Math.round(activePlayMs / 100) / 10;
      return {
        schema_version: "2.0",
        status: terminal
          ? "terminal"
          : (nativeState.status || (actionable ? "playing" : (loaded ? "loading" : "resetting"))),
        is_actionable: actionable,
        terminal: {
          isTerminal: terminal,
          outcome: terminal ? "game_over" : null,
          reason: terminalReason
        },
        game_state: {
          title: config.title || "Open-source browser game",
          active_play_seconds: seconds,
          environment_dynamic: visualIsLive,
          dynamic_changes: dynamicChanges,
          trusted_inputs: trustedInputs,
          elapsed_seconds: Math.round((now - startedAt) / 100) / 10,
          ...(nativeState.game_state || {})
        },
        metrics: {
          primary_score: seconds,
          active_play_seconds: seconds,
          dynamic_changes: dynamicChanges,
          ...(nativeState.metrics || {})
        },
        provenance: {
          repository: config.repository || null,
          commit: config.commit || null,
          license: config.license || null
        }
      };
    }
  };

  window.setInterval(poll, 250);
  loadFresh();
}());
