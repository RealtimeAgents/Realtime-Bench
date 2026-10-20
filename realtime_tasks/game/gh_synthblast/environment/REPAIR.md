# Benchmark adaptations

The task remains `gh_synthblast` at the pinned Apache-2.0 upstream commit.

- Arena layouts are fixed and do not follow the task seed: before each `Level` is constructed,
  `Math.random` restarts from a level-specific derivation of layout seed `0x12345678`, so the
  task seed, rendering duration and earlier-level play do not change either arena layout.
  Once the `Level` is built, the stream restarts from the same derivation of the task seed,
  which drives all later randomness (enemy wander, coin drops, particles, comets). Enemy
  placement and starting speeds are part of the fixed arena.
- The selected fixed Level 1 layout has two yellow objective pads and Level 2 has three. Neither
  level places a pad on the tank's initial straight-ahead line, so continuous forward movement
  cannot complete the route.
- A task-local, visible `TIME` HUD counts down 180 simulated seconds. It starts when Level 1 first
  becomes drivable, continues across the native level transition, terminates the task at zero,
  and resets to the full duration with every episode.
- Success still comes from the native level progression: driving over every yellow objective pad
  in Levels 1 and 2 constructs Level 3. Player coordinates and pad counts are exposed read-only
  for scoring and audit; no task adapter writes gameplay progress.
- The catalog rules document held movement/steering controls and the native one-second `LEVEL`
  banner followed by a gameplay key to launch the next arena.

## Distribution notices

The 2026-10-11 license review added modification comments to the adapted upstream
`.gitignore`, `index.html` and `js/classes/Game.js`, and notice comments to the existing
generated HTML and JavaScript. The executable content of those bundles was preserved;
this review did not rebuild or change gameplay.

`LICENSES.html` indexes the game and bundled-component notices. `license-sources.json`
records release versions, notice origins and hashes; `PROVENANCE.json` records the shipped
upstream files and the fonts/music omitted from the public distribution. Preserve these
notices when rebuilding or exporting the task, and refresh the records after changes.
