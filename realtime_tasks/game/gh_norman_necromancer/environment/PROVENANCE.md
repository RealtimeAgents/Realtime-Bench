# Norman the Necromancer provenance

- Upstream: https://github.com/danprince/norman-the-necromancer at `5db8ca88c8038b3d0888addaa521a16db3c259a8`.
- Popularity observed 2026-09-18: 200 GitHub stars and 40 forks. Popularity exception: GitHub's official js13kGames 2022 top-thirteen roundup selected Norman from 167 entries; the official entry has a 23.86/25.2 score.
- License: Unlicense at `upstream/LICENSE.md`; source, sprites, and procedural audio code remain together.
- Build: `npm install --ignore-scripts`, `npx tsc`, then `npx vite build --outDir runtime`; corresponding source and `upstream/runtime` are shipped.
- Local adaptation: a fixed seed is set in `index.html` and a read-only array accessor exposes native souls, level, health, casts, resurrection readiness, and state without being broken by upstream property mangling.
- Wrapper adaptation (`game_config.js`, upstream untouched): the requestAnimationFrame-driven loop is handed `performance.now()` in steps of at most 1000/30 ms so CDP virtual time governs game time (the compositor timestamp it reads natively ignores virtual time); Norman's death, which upstream implements as a page reload (`src/objects.ts:31`), is reported as the `norman_defeated` terminal and the reloaded page is left on its intro screen instead of auto-starting.
