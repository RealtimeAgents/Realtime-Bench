function highwaySnapshot(context) {
  const child = context.window();
  try { return child && child.highway404BenchmarkState ? child.highway404BenchmarkState() : null; }
  catch (_) { return null; }
}

window.OPEN_SOURCE_GAME_CONFIG = {
  title: "Highway 404", entry: "upstream/dist/index.html",
  repository: "https://github.com/herebefrogs/highway-404", commit: "9bbf0de17a49dc424f53b1f076573b6a789baa9d", license: "MIT",
  start(context) { for (const delay of [400, 1000]) context.later(delay, () => { const s = highwaySnapshot(context); if (!s || s.phase === "title") context.press("Space", " ", 32, 80); }); },
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
