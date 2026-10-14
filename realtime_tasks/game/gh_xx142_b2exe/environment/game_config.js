function xx142Window(context) {
  return context.window();
}

function xx142Snapshot(context) {
  const child = xx142Window(context);
  if (!child || typeof child.xx142BenchmarkState !== "function") return null;
  try { return child.xx142BenchmarkState(); } catch (_) { return null; }
}

window.OPEN_SOURCE_GAME_CONFIG = {
  title: "xx142-b2.exe",
  entry: "upstream/index.html",
  repository: "https://github.com/bencoder/js13k-2019",
  commit: "39bd729b5f18f82bda6c7b1e094e70521c09fbfd",
  license: "MIT",
  start(context) {
    // The upstream title and fade are presentation screens. Space is deliberately unmapped by
    // the game, so these presses begin the native timer without moving the player.
    context.later(250, () => context.press("Space", " ", 32, 80));
    context.later(1500, () => context.press("Space", " ", 32, 80));
  },
  ready(context) {
    const snapshot = xx142Snapshot(context);
    return Boolean(snapshot && snapshot.phase === "playing");
  },
  signal(context) {
    return xx142Snapshot(context);
  },
  state(context) {
    const snapshot = xx142Snapshot(context);
    if (!snapshot) {
      return { status: "loading", is_actionable: false, game_state: {}, metrics: {} };
    }
    const terminal = snapshot.phase === "complete";
    return {
      status: terminal ? "won" : snapshot.phase,
      is_actionable: !terminal && snapshot.phase !== "title",
      game_state: snapshot,
      metrics: { primary_score: snapshot.levels_completed },
    };
  },
  terminal(context) {
    const snapshot = xx142Snapshot(context);
    return snapshot && snapshot.phase === "complete" ? "memory_core_deactivated" : null;
  },
};
