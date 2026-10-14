// Iframe document loads since the last gameAPI.reset(): 1 is the episode itself. Upstream has no
// game-over screen; Player().onDeath reloads the page (src/objects.ts:31), so a second load means
// Norman died.
let benchmarkNormanLoads = 0;

window.OPEN_SOURCE_GAME_CONFIG = {
  title: "Norman the Necromancer", entry: "upstream/runtime/index.html",
  repository: "https://github.com/danprince/norman-the-necromancer", commit: "5db8ca88c8038b3d0888addaa521a16db3c259a8", license: "Unlicense",
  start(c) {
    benchmarkNormanLoads += 1;
    const w = c.window(); if (!w) return;
    // The game loop integrates dt from the requestAnimationFrame timestamp (src/engine.ts:140-144).
    // That timestamp comes from the compositor and ignores CDP virtual time: measured 3033 ms of
    // game time per 3.0 s wall under Emulation.setVirtualTimePolicy pause while performance.now()
    // stayed frozen. Hand the loop performance.now() instead, so the world stops while the harness
    // holds the token clock and advances by exactly the simulated interval. Under the wall clock
    // both timestamps agree.
    // A 1 s virtual advance reached the loop in only 10-15 native frames (dt up to 100 ms, an
    // 18 px skull step against 10-14 px hitboxes), so one native frame is split into game steps of
    // at most 1000/30 ms. The loop registers itself exactly once per invocation, so a callback that
    // is already pending gets its existing id back instead of a second native frame.
    if (!w.__benchmarkNormanClockAdapted) {
      w.__benchmarkNormanClockAdapted = true;
      const nativeRAF = w.requestAnimationFrame.bind(w);
      const MAX_STEP_MS = 1000 / 30;
      const pendingIds = new Map();
      let lastNow = null;
      w.requestAnimationFrame = (callback) => {
        if (pendingIds.has(callback)) return pendingIds.get(callback);
        const id = nativeRAF(() => {
          pendingIds.delete(callback);
          const now = w.performance.now();
          const elapsed = lastNow === null ? 0 : now - lastNow;
          const steps = Math.max(1, Math.ceil(elapsed / MAX_STEP_MS));
          for (let i = 1; i <= steps; i++) callback(now - elapsed + elapsed * i / steps);
          lastNow = now;
        });
        pendingIds.set(callback, id);
        return id;
      };
    }
    // After Norman's death the reloaded page stays on the intro screen; terminal() reports it.
    if (benchmarkNormanLoads > 1) return;
    for (const delay of [500, 1200, 2200]) c.later(delay, () => {
      const w = c.window(); const d = c.document(); if (!w || !d) return;
      if (w.__benchmarkNormanStarted) return;
      w.__benchmarkNormanStarted = true;
      const target = d.querySelector("canvas") || d.body;
      target.dispatchEvent(new w.PointerEvent("pointerup", { bubbles: true, clientX: w.innerWidth / 2, clientY: w.innerHeight / 2 }));
    });
  },
  reset() { benchmarkNormanLoads = 0; },
  ready(c) { const w = c.window(); const s = w && w.__benchmarkNormanState ? w.__benchmarkNormanState() : null; return Boolean(s && s[5] === 1); },
  state(c) {
    const w = c.window(); const s = w && w.__benchmarkNormanState ? w.__benchmarkNormanState() : null;
    if (!s) return { status: "loading", is_actionable: false };
    // Game states: 0 intro, 1 playing, 2 shop between levels (keyboard), 3 lose (unused upstream), 4 win.
    return { status: s[5] === 3 ? "terminal" : "playing", is_actionable: s[5] === 1 || s[5] === 2,
      game_state: { souls_collected: Math.floor(Number(s[0]) || 0), level: Number(s[1]) || 0,
        player_hp: Number(s[2]) || 0, casts_available: Number(s[3]) || 0,
        resurrection_ready: Boolean(s[4]), in_shop: s[5] === 2 }, metrics: { primary_score: Math.floor(Number(s[0]) || 0) } };
  },
  terminal(c) {
    if (benchmarkNormanLoads > 1) return "norman_defeated";
    const w = c.window(); const s = w && w.__benchmarkNormanState ? w.__benchmarkNormanState() : null;
    return s && s[5] === 3 ? "norman_defeated" : null;
  }
};
