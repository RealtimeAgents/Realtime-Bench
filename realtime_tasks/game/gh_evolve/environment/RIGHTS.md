# Evolve rights review

Selected for public-20, reviewed 2026-10-08; inline icons re-reviewed 2026-10-10.
Source: https://github.com/pmotschmann/Evolve/tree/3436358dcd03d9f9e071d51ea071e0a78c0322e4

The game is MPL-2.0 (`environment/upstream/LICENSE`), not MIT. The package includes
editable `src/`, stylesheets, translation data, build scripts and package lockfile.
The modified source `src/vars.js` includes the same read-only benchmark accessor
as the generated `evolve/main.js`; the modified HTML localizes dependencies and
removes remote analytics/fonts. Those modified upstream files remain MPL-2.0.
`environment/source-audit.json` lists 111 author-repository files and distinguishes
the three modified files. Retain this complete source alongside generated code.

The project's own graphics, tutorial images and favicons are retained from the
MPL-licensed source distribution. The inline `svgIcons` in `src/functions.js` are not all
original: `heavy` is Google Material Icons `fitness_center` (Apache-2.0, text in
`licenses/Material-Icons-Apache-2.0.txt`), and `trash` and `martini` are Mapbox Maki
`waste-basket` and `bar` (CC0, no notice required). The remaining icon paths have no
recorded external source and are retained under the repository's MPL-2.0 grant.
The only bundled font is Weather Icons: its embedded OFL grant and designers Erik
Flowers / Lukas Bischoff are retained with full OFL terms. Weather Icons CSS is MIT.
System font names do not bundle any font binaries.

Separately licensed libraries include jQuery, Vue, Buefy, Popper, Sortable, Chart.js
(MIT), LZString 1.4.4 (WTFPL), CryptoJS 4.0.0 / HTML5Sortable 0.9.15 (MIT), and
Less 3.9.0 (Apache-2.0). The three previously unnamed minified files match the stated
published distributions after trimming outer whitespace. The existing vendor
notices remain, supplemented by Bulma 0.9.4, Chart.js's Kurkle Color, Vue's Snabbdom
and Microsoft TypeScript helper notices, plus Less's Promise, Asap, Clone, Browserify and
Sass notices. Copies/attribution methods are recorded in `environment/license-sources.json`.
Less's source also credits Michael Jackson's RGB/HSL/HSV conversion reference:
https://gist.github.com/mjackson/5311256 . This records the upstream credit, not a
separate permission invented for that reference. Matching lineage notices are not
a claim to have reconstructed every historical transitive dependency lock.

Original benchmark adapters and task content have `environment/ADAPTER-LICENSE.txt`.
Publish the component notices, full MPL source and modification record with the game;
a minified-runtime-only copy is not this reviewed distribution. No game mechanics
were changed for this review. See `environment/LICENSES.html`.
