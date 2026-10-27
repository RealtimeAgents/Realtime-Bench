# Highway 404 provenance

- Upstream: https://github.com/herebefrogs/highway-404
- Commit: `9bbf0de17a49dc424f53b1f076573b6a789baa9d`
- License: MIT; preserved in `upstream/LICENSE`.
- Build: `npm ci --ignore-scripts && npm run clean && npm run build:js && npm run build:html`; optional ZIP recompression is not part of the runtime.
- Adaptation: fixed pre-load RNG seed and a read-only snapshot of native countdown, car, collectibles, and screen state. The benchmark objective combines one native teapot with 30 seconds of survival to require both collection and sustained navigation. Startup is hardened: input before the asynchronous load finishes no longer starts a level that the title-screen setup then erases (which left the road empty and the teapot missing), and a browser with no speech voices plays silently instead of hanging on the loading screen. Road schedule, speed, hazards, and collision behavior are unchanged.

## Source and asset audit

`PROVENANCE.json` identifies every vendored file and its original path. The generated
`dist/` files and two adapted source files are distinguished from unchanged upstream
files. `assets/*.ase`, `src/img/*.png`, and the submission screenshots come from the
fixed game snapshot. The [upstream credits](upstream/README.md#special-thanks--credits)
identify Peters and flo- for the Glitch Hunter pixel font, Frank Force and Keith Clark
for sound synthesis, and Matt DesLauriers for interpolation helpers. The game draws text
only from the 5 px `src/img/charset.png` (`src/js/text.js`), which is part of the MIT game
tree. The unused 8 px atlas `src/img/charset.glitchhunter.png`, recoloured from Glitch
Hunters (p1100i and flo-, js13k 2016, declared MIT in that project's `package.json`), was
removed for the public release; the credit is preserved.

Additional component notices are preserved from their authors:

- [ZzFX MIT](licenses/ZZFX-LICENSE.txt): [Frank Force's repository](https://github.com/KilledByAPixel/ZzFX/blob/aab7e2b6b9086746b6e55fab75c75ce03e716c49/LICENSE).
- [ZzFXM MIT](licenses/ZZFXM-LICENSE.txt): [Keith Clark's repository](https://github.com/keithclark/ZzFXM/blob/cb07fa9ca36aefd67a0c8c656d2958b62f8ed9fe/LICENSE). The embedded header identifies v2.0.3.
- [canvas-sketch-util MIT](licenses/CANVAS-SKETCH-UTIL-LICENSE.txt): [Matt DesLauriers' repository](https://github.com/mattdesl/canvas-sketch-util/blob/852fee9564c85d535ef82cc59f9b857b1bf08a3d/LICENSE.md).

These links identify the notice sources; the implementation snapshot is the pinned
Highway 404 commit, not the latest version of those components.
