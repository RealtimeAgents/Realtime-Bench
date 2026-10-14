# SYNTH*BLAST provenance

Selected for public-20, reviewed 2026-10-11.

- Upstream: https://github.com/brianrisk/SYNTHBLAST-threejs-game
- Pinned commit: `8fd77ccc4e03bc9c6033e0738472351b0ab8f2a6`
- License: Apache-2.0; the upstream license is preserved at `environment/upstream/LICENSE`.
- Vendoring: `environment/upstream/` is the GitHub source tree; its `dist/` is a Vite production build
  (`vite build --base=./`) with runtime media copied beside it. `upstream/index.html` carries the
  seeded-PRNG script described in `environment/REPAIR.md`. Root files are benchmark-only launch/observation
  adapters.
- Sound effects: `assets/audio/*.wav` ship with their Bfxr `.bfxrsound` project files and were made
  by the author; they are covered by the repository's Apache-2.0 grant.
- Images: `assets/img/` and the Drone model are the author's repository files.
- Music removed: `music/motel_bedroom.mp3` (never loaded) and `music/taketheworld-outro.mp3`
  (played only when music is toggled on; off by default and never enabled by the benchmark) carry
  no recorded author or license, so both were deleted from `upstream/` and `upstream/dist/`. The
  unchanged build still constructs the audio element; it simply has no source to play.
- Bundled libraries in `environment/upstream/dist/assets/`: three.js 0.185.1, pixi.js 8.19.0
  and stats.js 0.17.0 (MIT), plus the following dependencies verified in the generated chunks:

| Component | Version | License |
| --- | --- | --- |
| eventemitter3 | 5.0.4 | MIT |
| @pixi/colord | 2.9.6 | MIT |
| ismobilejs | 1.1.1 | MIT |
| earcut | 3.2.3 | ISC |
| parse-svg-path | 0.2.0 | MIT |
| tiny-lru | 11.4.7 | BSD-3-Clause |
| Vite runtime helpers | 8.1.5 | MIT |
| Rolldown runtime | 1.1.5 | MIT, with its upstream Rollup/esbuild notices |

The complete component notices are in `environment/licenses/`, indexed by
`environment/LICENSES.html`. `environment/license-sources.json` records their source URLs,
versions and SHA-256 values. The @pixi/colord package omits its license file; its MIT notice
comes from the exact Git commit identified by the package's npm `gitHead`.
Vite's notice preserves its core license for the emitted runtime helpers; the build tool
and its server-side dependencies are not shipped. Rolldown's separate third-party notices
are retained alongside its own license.

Modified upstream files carry modification notices, and generated JavaScript carries a
license pointer. `environment/PROVENANCE.json` identifies shipped files, upstream differences
and removed files. Task-local repair notes are in `environment/REPAIR.md`. Preserve the game
license, component notices and these records when redistributing the complete task package.
