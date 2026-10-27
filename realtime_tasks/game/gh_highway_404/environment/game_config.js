function highwaySnapshot(context) {
  const child = context.window();
  try { return child && child.highway404BenchmarkState ? child.highway404BenchmarkState() : null; }
  catch (_) { return null; }
}

window.OPEN_SOURCE_GAME_CONFIG = {
  title: "Highway 404", entry: "upstream/dist/index.html",
  repository: "https://github.com/herebefrogs/highway-404", commit: "9bbf0de17a49dc424f53b1f076573b6a789baa9d", license: "MIT",
  // The game loads asynchronously (images, voices, music) and reports phase "loading" until its
  // title screen exists; poll until then so Space starts the real level instead of a half-built one.
  start(context) {
    const tick = (left) => {
      const s = highwaySnapshot(context);
      if (s && s.phase === "playing") return;
      if (s && s.phase === "title") context.press("Space", " ", 32, 80);
      if (left > 0) context.later(250, () => tick(left - 1));
    };
    context.later(400, () => tick(60));
  },
  ready(context) { const s = highwaySnapshot(context); return Boolean(s && s.phase === "playing"); },
  signal(context) { return highwaySnapshot(context); },
  state(context) {
    const s = highwaySnapshot(context);
    if (!s) return { status: "loading", is_actionable: false };
    const completed = s.survival_seconds >= 30 && s.alive && s.teapots_collected >= 1;
    return { status: completed ? "won" : s.phase, is_actionable: s.phase === "playing" && s.alive,
      game_state: { ...s, mission_goal: completed ? 1 : 0 }, metrics: { primary_score: completed ? 1 : 0 } };
  },
  terminal(context) { const s = highwaySnapshot(context); return s && (s.survival_seconds >= 30 && s.alive && s.teapots_collected >= 1 ? "teapot_collected_and_survived" : (s.phase === "game_over" ? "car_crashed" : null)); }
};
