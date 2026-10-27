function readEvolveSave(c) {
  const w = c.window();
  try {
    const raw = w.localStorage.getItem("evolved");
    return raw && w.LZString ? JSON.parse(w.LZString.decompressFromUTF16(raw)) : null;
  } catch (_) { return null; }
}

window.OPEN_SOURCE_GAME_CONFIG = {
  title: "Evolve", entry: "upstream/index.html",
  repository: "https://github.com/pmotschmann/Evolve", commit: "3436358dcd03d9f9e071d51ea071e0a78c0322e4", license: "MPL-2.0",
  // Narrow embeds (frame width <= 768px, e.g. the user-study iframe): Bulma stacks the columns, which pushes
  // the main Evolve column below the fold of a page that cannot scroll. Keep the columns side by side there.
  // The rule is a no-op at wider widths, so the full-size layout is unchanged.
  start(c) {
    const d = c.document();
    if (!d || !d.head || d.getElementById("benchmark-narrow-columns")) return;
    const style = d.createElement("style");
    style.id = "benchmark-narrow-columns";
    style.textContent = "@media screen and (max-width:768px){" +
      ".main>.columns:not(.is-desktop){display:flex}" +
      ".main>.columns>.column.is-one-quarter{flex:none;width:38%}" +
      ".main>.columns>.column.is-three-quarters{flex:none;width:62%}}";
    d.head.appendChild(style);
  },
  ready(c) { const d = c.document(); return Boolean(d && d.body && !d.querySelector(".loading") && d.querySelector("button, .button")); },
  state(c) {
    const d = c.document(); const w = c.window();
    const save = (w && w.__benchmarkEvolveGlobal) || readEvolveSave(c) || {};
    const evolution = save.evolution || {};
    const upgrades = Object.values(evolution).reduce((n, value) => {
      return n + (value && typeof value === "object" && Number(value.count) > 0 ? 1 : 0);
    }, 0);
    const body = d && d.body ? d.body.innerText : "";
    return { status: body ? "playing" : "loading", is_actionable: Boolean(body),
      game_state: { evolution_upgrades: upgrades, rna: Number(save.resource && save.resource.RNA && save.resource.RNA.amount) || 0,
        dna: Number(save.resource && save.resource.DNA && save.resource.DNA.amount) || 0, evolution_stage: Number(save.race && save.race.stage) || 0 },
      metrics: { primary_score: upgrades } };
  }
};
