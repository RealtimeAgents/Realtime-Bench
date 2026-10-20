# JavaScript Pong rights review

Selected for public-20, reviewed 2026-10-08.
Source: https://github.com/jakesgordon/javascript-pong/tree/ca3240536e4f79ab7144388e56ed19de715b6662

The author's MIT grant (Jake Gordon and contributors, 2011–2016) is retained at
`environment/upstream/LICENSE`. All 37 vendored upstream files match the pinned
source, including the author's PNG UI labels and four WAV clips. No external
recording, commercial sprite set, bundled font or runtime framework is identified
in that distribution. The current benchmark launch disables sounds, but the WAVs
remain part of the licensed source package. `game.js` credits the reference material
used for its small ES5 compatibility routines; those original comments remain intact.

Benchmark-specific launcher, seed, observer and task content are MIT under
`environment/ADAPTER-LICENSE.txt`. The launch page's documented speed configuration
does not modify the pristine upstream source. Preserve the original source license,
attributions, modification record and adapter license in every export.
Source comparison: `environment/source-audit.json`; legal entry: `environment/LICENSES.html`.
