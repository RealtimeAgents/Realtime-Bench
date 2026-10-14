(() => {
  "use strict";

  const TIME_LIMIT_SECONDS = 180;
  let deadlineAt = null;

  function instance(c) {
    const game = c.window();
    return game && game.__synthblastGame && game.__synthblastGame();
  }

  // The seed the page applied to arena construction, so records reflect the actual layout.
  function layoutSeed(c) {
    const game = c.window();
    const seed = game && game.__synthblastLayoutSeed;
    return Number.isInteger(seed) ? "0x" + seed.toString(16).padStart(8, "0") : null;
  }

  function remainingSeconds() {
    if (deadlineAt === null) return TIME_LIMIT_SECONDS;
    return Math.max(0, Math.ceil((deadlineAt - performance.now()) / 1000));
  }

  function updateTimer() {
    const timer = document.getElementById("task-timer");
    if (!timer) return;
    const remaining = remainingSeconds();
    timer.textContent = `TIME ${remaining}`;
    timer.classList.toggle("urgent", remaining > 0 && remaining <= 20);
    timer.classList.toggle("expired", remaining === 0);
  }

window.OPEN_SOURCE_GAME_CONFIG = {
  title: "SYNTH*BLAST", entry: "upstream/dist/index.html",
  repository: "https://github.com/brianrisk/SYNTHBLAST-threejs-game", commit: "8fd77ccc4e03bc9c6033e0738472351b0ab8f2a6", license: "Apache-2.0",
  timeLimitSeconds: TIME_LIMIT_SECONDS,
  layoutSeed: 0x12345678,
  start(c) {
    deadlineAt = null;
    updateTimer();
    for (const delay of [1200, 2500, 4500]) c.later(delay, () => c.press("Enter", "Enter", 13, 100));
  },
  ready(c) {
    const game = c.window();
    return Boolean(game && game.__synthblastGame && game.__synthblastGame());
  },
  terminal(c) {
    const game = instance(c);
    if (deadlineAt !== null && performance.now() >= deadlineAt) return "task_time_expired";
    if (game && game.levelNumber >= 3) return "reached_level_3";
    return game && game.isOver() ? "tank_destroyed" : null;
  },
  tick(c) {
    const game = instance(c);
    if (deadlineAt === null && game && game.level && game.level.inPlay) {
      deadlineAt = performance.now() + TIME_LIMIT_SECONDS * 1000;
    }
    updateTimer();
  },
  state(c) {
    const game = instance(c);
    const level = Number(game && game.levelNumber) || 0;
    const current = game && game.level;
    const padsRemaining = Number(current && current.padsRemaining) || 0;
    const padsTotal = Number(current && current.padsTotal) || 0;
    const hero = game && game.hero;
    return {
      status: game ? (current && current.inPlay ? "playing" : "loading") : "loading",
      is_actionable: Boolean(game && current && current.inPlay),
      game_state: {
        level,
        pads_activated: Math.max(0, padsTotal - padsRemaining),
        pads_remaining: padsRemaining,
        pads_total: padsTotal,
        hit_points: Number(hero && hero.hitPoints) || 0,
        player_x: Number(hero && hero.getX && hero.getX()) || 0,
        player_y: Number(hero && hero.getY && hero.getY()) || 0,
        layout_seed: layoutSeed(c),
        time_limit_seconds: TIME_LIMIT_SECONDS,
        time_remaining_seconds: remainingSeconds()
      },
      metrics: { primary_score: level }
    };
  }
};
})();
