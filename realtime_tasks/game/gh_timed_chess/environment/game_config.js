window.OPEN_SOURCE_GAME_CONFIG = {
  title: "Timed Chess",
  entry: "upstream/index.html",
  repository: "https://github.com/sagarbudhwani/classic-chess-js",
  commit: "28f90f911db2d3638ff064795d67a87f1142b98b",
  license: "MIT",

  start(c) {
    const d = c.document();
    const w = c.window();
    if (!d || !w) return;

    // Fix one benchmark configuration using the upstream game's own visible controls. Restarting
    // after selecting the timer is required because the upstream initializes its clocks before a
    // timer button is chosen. The upstream UI offers 3 minutes at minimum; the benchmark uses a
    // 2-minute White clock, so rewrite the button's data-time (script.js parses
    // `parseInt(button.dataset.time) * 60`) before clicking it.
    c.clickSelector("#bot-mode");
    c.clickSelector('#bot-difficulty button[data-difficulty="easy"]');
    c.clickSelector('#side-selection button[data-side="white"]');
    const timerButton = d.querySelector('#timer-selection button[data-time="3"]');
    if (timerButton) timerButton.dataset.time = "2";
    c.clickSelector('#timer-selection button[data-time="2"]');
    w.confirm = () => true;
    c.clickSelector("#restart-btn");

    // The benchmark is a single irreversible timed line: settings, undo, and restart cannot be
    // used to pause or replenish the deadline. Hide the now-fixed setup so the board fits on the
    // benchmark desktop without changing any upstream source file.
    const setup = d.querySelector(".game-setup");
    if (setup) setup.style.display = "none";
    for (const selector of ["#hint-btn", "#undo-btn", "#restart-btn"]) {
      const button = d.querySelector(selector);
      if (button) {
        button.disabled = true;
        button.style.display = "none";
      }
    }
    Array.from(d.querySelectorAll(".rank-labels span")).forEach((label, index) => {
      label.textContent = String(8 - index);
    });

    // Enter a fixed legal opening through the upstream UI. The upstream AI replies after 500 ms;
    // its reply starts White's 2-minute clock before the first model-visible observation.
    c.later(250, () => c.clickSelector('.square[data-row="6"][data-col="4"]'));
    c.later(400, () => c.clickSelector('.square[data-row="4"][data-col="4"]'));
  },

  signal(c) {
    const d = c.document();
    if (!d) return null;
    return {
      status: d.querySelector("#game-status")?.textContent || "",
      whiteClock: d.querySelector("#white-time")?.textContent || "",
      history: d.querySelector("#history-list")?.textContent || ""
    };
  },

  state(c) {
    const d = c.document();
    if (!d) return {};
    const whiteMoves = Array.from(d.querySelectorAll("#history-list .move-entry.white"))
      .map((node) => (node.textContent || "").trim());
    const clockText = (d.querySelector("#white-time")?.textContent || "0:00").trim();
    const match = /^(\d+):(\d{2})$/.exec(clockText);
    const clockSeconds = match ? Number(match[1]) * 60 + Number(match[2]) : 0;
    return {
      captures: whiteMoves.filter((move) => move.includes("x")).length,
      player_moves: whiteMoves.length,
      clock_seconds_remaining: clockSeconds,
      deadline_active: clockSeconds > 0,
      side_to_move: (d.querySelector("#game-status")?.textContent || "").startsWith("White")
        ? "white" : "black"
    };
  },

  terminal(c) {
    const status = c.document()?.querySelector("#game-status")?.textContent || "";
    return /^(Time out!|Checkmate!|Stalemate!)/.test(status) ? status : null;
  }
};
