(function () {
  "use strict";

  // Benchmark rule: Black (the agent) has PER_MOVE_CAP_S seconds for each move. The move clock
  // runs only while it is Black's turn (not during the AI reply, not after game over). When it
  // reaches 0 the game is lost on time: getState() reports a fail terminal, the board is covered
  // and no further stone can be placed. The clock is shown in the top bar. The wrapper page runs
  // under the harness's virtual time during a benchmark episode, so this is simulated time there.
  // Debug override only (e.g. ?per_move_cap_s=5).
  const PER_MOVE_CAP_S = 10;
  const clockParams = new URLSearchParams(window.location.search);
  const perMoveCapS = positiveOr(Number(clockParams.get("per_move_cap_s")), PER_MOVE_CAP_S);
  const CLOCK_TICK_MS = 200;

  const frame = document.getElementById("game-frame");
  const timeupEl = document.getElementById("timeup");
  const clockEls = {
    turn: document.getElementById("clock-turn"),
    move: document.getElementById("clock-move"),
    moveBar: document.getElementById("clock-move-bar"),
    last: document.getElementById("clock-last")
  };
  let generation = 0;
  let loaded = false;
  let configured = false;
  let auxiliaryToolViolation = false;
  let resetResolve = null;

  // elapsed: seconds of the current Black move (Black to move, AI idle, game not over);
  // spent: total seconds Black has spent over finished moves (diagnostic only);
  // expired: the move clock reached 0 -> Black lost on time (latched until reset()).
  // moves: last seen upstream move count, used to notice a move-count drop.
  const clock = { elapsed: 0, spent: 0, expired: false, lastTick: null, lastMove: null, moves: 0 };

  function positiveOr(value, fallback) {
    return Number.isFinite(value) && value > 0 ? value : fallback;
  }

  function childWindow() {
    try { return frame.contentWindow; } catch (_) { return null; }
  }

  function childDocument() {
    try { return frame.contentDocument; } catch (_) { return null; }
  }

  function activeValue(documentRef, selector, attribute) {
    const node = documentRef && documentRef.querySelector(selector + " .opt.active");
    return node ? node.getAttribute(attribute) : null;
  }

  function clickOption(documentRef, selector) {
    const node = documentRef && documentRef.querySelector(selector);
    if (!node) return false;
    node.click();
    return true;
  }

  function cellLabel(x, y) {
    // Columns A–O left to right, rows numbered from the bottom (centre = H8).
    return String.fromCharCode(65 + x) + String(15 - y);
  }

  // Observe stone placements without editing the upstream file: `tryPlace` is a top-level function
  // declaration in game/index.html, so it is a property of the iframe window and every internal
  // call (canvas click handler, AI reply) resolves through that property.
  function installPlacementHook() {
    const w = childWindow();
    const original = w && w.tryPlace;
    if (typeof original !== "function" || original.__benchmarkWrapped) return;
    const wrapped = function (x, y, player) {
      const placed = original.apply(this, arguments);
      if (placed) onStonePlaced(x, y, player);
      return placed;
    };
    wrapped.__benchmarkWrapped = true;
    w.tryPlace = wrapped;
  }

  function onStonePlaced(x, y, player) {
    if (player === 1) {
      // Black (the agent) moved: the next move starts with a fresh clock.
      clock.spent += Math.min(clock.elapsed, perMoveCapS);
      clock.elapsed = 0;
      clock.lastMove = { player: "black", cell: cellLabel(x, y), x, y };
    } else if (player === 2) {
      clock.lastMove = { player: "white", cell: cellLabel(x, y), x, y };
    }
    renderClock(readState());
  }

  function configureGame() {
    const d = childDocument();
    if (!d) return false;
    // Fixed task setup: agent=Black, AI=White, Easy, PVE.
    const side = clickOption(d, "#sideSeg .opt[data-side='white']");
    const difficulty = clickOption(d, "#diffSeg .opt[data-diff='1']");
    const mode = clickOption(d, "#modeSeg .opt[data-mode='pve']");
    if (!side || !difficulty || !mode) return false;

    // Programmatic setup above is allowed. Subsequent trusted changes to these controls are not.
    d.addEventListener("click", (event) => {
      if (!event.isTrusted || !event.target || !event.target.closest) return;
      if (event.target.closest("#modeSeg, #diffSeg, #sideSeg")) {
        auxiliaryToolViolation = true;
      }
    }, true);
    // Upstream Undo / Restart would take stones back or wipe a decided game, and the Renju
    // "Forbidden moves" switch changes the rules mid-game. None are part of the task, so hide
    // them (upstream updateUI() only toggles `disabled`, never `display`, so they stay hidden).
    ["undoBtn", "restartBtn", "forbidSwitch"].forEach((id) => {
      const btn = d.getElementById(id);
      if (btn) btn.style.display = "none";
    });
    installPlacementHook();
    return true;
  }

  function readState() {
    const d = childDocument();
    if (!loaded || !configured || !d) return null;
    const status = d.getElementById("statusBox");
    const message = (d.getElementById("statusMsg") || {}).textContent || "";
    const turn = (d.getElementById("turnText") || {}).textContent || "";
    const moveText = (d.getElementById("moveCount") || {}).textContent || "";
    const gameOver = !!(status && status.classList.contains("over"));
    const winner = gameOver && /^Black wins/.test(message)
      ? 1
      : (gameOver && /^White wins/.test(message) ? 2 : 0);
    const mode = activeValue(d, "#modeSeg", "data-mode");
    const difficulty = Number(activeValue(d, "#diffSeg", "data-diff"));
    const aiSide = activeValue(d, "#sideSeg", "data-side");
    const setupValid = mode === "pve" && difficulty === 1 && aiSide === "white";
    const moves = Number((moveText.match(/\d+/) || [0])[0]);
    return {
      gameOver,
      winner,
      mode,
      difficulty,
      aiSide,
      setupValid,
      moves,
      aiThinking: /AI thinking/.test(turn),
      blackToMove: /^Black to move/.test(turn)
    };
  }

  /* ---------- Move clock ---------- */
  function clockRunning(state) {
    return !!(state && configured && !clock.expired && !state.gameOver && !state.aiThinking &&
              state.blackToMove);
  }

  function moveRemainingS() {
    return Math.max(0, perMoveCapS - clock.elapsed);
  }

  function showTimeUp(show) {
    if (timeupEl) timeupEl.classList.toggle("show", show);
  }

  // Book the time since the previous call. Called by the 200 ms ticker and by getState(), so a
  // state read is never behind the clock it reports.
  function accrue() {
    const now = performance.now();
    const dt = clock.lastTick === null ? 0 : (now - clock.lastTick) / 1000;
    clock.lastTick = now;
    const state = readState();
    if (state) {
      if (state.moves < clock.moves) clock.lastMove = null;
      clock.moves = state.moves;
    }
    if (clockRunning(state)) {
      clock.elapsed += dt;
      if (clock.elapsed >= perMoveCapS) {
        // Move clock ran out: Black loses on time. The overlay covers the board so no further
        // click reaches the canvas; the state is latched until the next reset().
        clock.elapsed = perMoveCapS;
        clock.expired = true;
        showTimeUp(true);
      }
    }
    return state;
  }

  function clockTick() {
    renderClock(accrue());
  }

  function formatClock(seconds) {
    const whole = Math.ceil(Math.max(0, seconds));
    const m = Math.floor(whole / 60);
    const s = whole % 60;
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  function setText(el, text) {
    if (el && el.textContent !== text) el.textContent = text;
  }

  function renderClock(state) {
    const running = clockRunning(state);
    let turnText = "Loading…";
    let turnKind = "idle";
    if (clock.expired) {
      turnText = "Black lost on time"; turnKind = "over";
    } else if (state && state.gameOver) {
      turnText = state.winner === 1 ? "Black wins" : (state.winner === 2 ? "White wins" : "Game over");
      turnKind = "over";
    } else if (state && state.aiThinking) {
      turnText = "White thinking…"; turnKind = "white";
    } else if (state && state.blackToMove) {
      turnText = "Black to move"; turnKind = "black";
    } else if (state) {
      turnText = "White to move"; turnKind = "white";
    }
    setText(clockEls.turn, turnText);
    if (clockEls.turn) clockEls.turn.dataset.kind = turnKind;

    const moveLeft = moveRemainingS();
    setText(clockEls.move, formatClock(moveLeft));
    if (clockEls.moveBar) clockEls.moveBar.style.width = (100 * moveLeft / perMoveCapS).toFixed(1) + "%";
    const bar = document.getElementById("clockbar");
    if (bar) {
      bar.classList.toggle("running", running);
      bar.classList.toggle("move-low", moveLeft <= 3 && !clock.expired);
      bar.classList.toggle("expired", clock.expired);
    }
    const last = clock.lastMove;
    setText(clockEls.last, last ? ((last.player === "white" ? "⚪ White" : "⚫ Black") + " · " + last.cell) : "—");
  }

  function resetClock() {
    clock.elapsed = 0;
    clock.spent = 0;
    clock.expired = false;
    clock.lastTick = null;
    clock.lastMove = null;
    clock.moves = 0;
    showTimeUp(false);
    renderClock(null);
  }

  function loadFresh() {
    generation += 1;
    loaded = false;
    configured = false;
    auxiliaryToolViolation = false;
    resetClock();
    frame.src = "game/index.html?benchmark_reset=" + generation;
    return new Promise((resolve) => { resetResolve = resolve; });
  }

  frame.addEventListener("load", () => {
    loaded = true;
    configured = configureGame();
    resetClock();
    try { frame.focus(); childWindow().focus(); } catch (_) {}
    if (resetResolve) {
      const resolve = resetResolve;
      resetResolve = null;
      resolve({ ok: configured, generation });
    }
  });

  window.setInterval(clockTick, CLOCK_TICK_MS);

  window.gameAPI = {
    schemaVersion: "2.0",
    init() { return { ok: true, source: "vajvip/gomoku" }; },
    reset() { return loadFresh(); },
    // deferred_start hook: the harness calls this the instant the agent's clock starts, so the
    // first move gets its full clock instead of the seconds spent on reset/baseline sampling.
    activate() {
      if (!clock.expired) { clock.elapsed = 0; clock.lastTick = null; }
      renderClock(readState());
      return { ok: true, per_move_cap_s: perMoveCapS };
    },
    getState() {
      accrue();
      const state = readState();
      const setupValid = !!(state && state.setupValid);
      const violation = auxiliaryToolViolation || !!(state && !setupValid);
      const legitimateBlackWin = !!(
        state && state.gameOver && state.winner === 1 && setupValid && !violation
      );
      const terminal = !!(state && (state.gameOver || clock.expired));
      const reason = !terminal ? null
        : (clock.expired ? "move_clock_expired" : (state.winner === 1 ? "black_five" : "white_five"));
      return {
        schema_version: "2.0",
        status: terminal ? "terminal" : (configured ? "playing" : (loaded ? "loading" : "resetting")),
        is_actionable: !!(state && !terminal && !state.aiThinking),
        terminal: {
          isTerminal: terminal,
          outcome: terminal ? (legitimateBlackWin ? "success" : "failure") : null,
          reason
        },
        game_state: {
          winner: state ? state.winner : 0,
          game_over: terminal,
          mode: state ? state.mode : null,
          difficulty: state ? state.difficulty : null,
          ai_side: state ? state.aiSide : null,
          moves: state ? state.moves : 0,
          black_to_move: !!(state && state.blackToMove),
          last_move: clock.lastMove ? { player: clock.lastMove.player, cell: clock.lastMove.cell } : null,
          move_clock_s: Math.round(moveRemainingS() * 10) / 10,
          move_clock_expired: clock.expired,
          think_spent_s: Math.round((clock.spent + clock.elapsed) * 10) / 10,
          per_move_cap_s: perMoveCapS
        },
        metrics: {
          primary_score: legitimateBlackWin ? 1 : 0,
          legitimate_black_win: legitimateBlackWin ? 1 : 0,
          auxiliary_tool_violation: violation
        },
        provenance: {
          repository: "vajvip/gomoku",
          license: "MIT"
        }
      };
    }
  };
  loadFresh();
}());
