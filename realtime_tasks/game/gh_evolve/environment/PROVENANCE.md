# Evolve provenance

- Upstream: https://github.com/pmotschmann/Evolve at `3436358dcd03d9f9e071d51ea071e0a78c0322e4`.
- License: MPL-2.0 at `upstream/LICENSE`. Vendored browser dependencies retain their notices in `upstream/vendor`.
- Build: upstream static browser release; `upstream/index.html` is served directly.
- Local adaptation: remote fonts, analytics, and CDN dependencies are removed or vendored, and a fixed seed loads before scripts. A read-only accessor exposes the current native state so the adapter reports resources and distinct acquired evolution records without waiting for an autosave.

## Source and asset audit

`PROVENANCE.json` records local hashes, source paths, and modified/generated files.
Game icons, PNG assets, and Weather Icons font files match the fixed Evolve snapshot.
The source tree does not provide individual artist records for its game icons; its
MPL-2.0 declaration is retained. The browser bundle and native source modifications
are shipped together, with the original MPL text.

The `upstream/vendor/` directory already preserves notices for Buefy 0.9.22,
Chart.js 3.8.2, jQuery 3.6.3, Popper 2.9.2, Sortable 1.10.2, and Vue 2.7.14.
The Buefy notice also applies to `upstream/lib/buefy.min.0.9.22.css`.
Additional notices for libraries in `upstream/lib/` are in `licenses/`:

| Component | Version / source evidence | Preserved notice |
| --- | --- | --- |
| Less | Header identifies 3.9.0; [upstream tag](https://github.com/less/less.js/tree/v3.9.0). | [Apache-2.0](licenses/LESS-LICENSE.txt) |
| CryptoJS | Minified bytes match the 4.0.0 distribution after trailing whitespace normalization; [source](https://github.com/brix/crypto-js/tree/4.0.0). | [MIT](licenses/CRYPTOJS-LICENSE.txt) |
| LZ-String | Minified bytes match 1.4.4 after trailing whitespace normalization; [fixed source](https://github.com/pieroxy/lz-string/tree/cf06e9a0e61daa8b120a474bbc80666f959ff7d4). | [WTFPL v2](licenses/LZ-STRING-LICENSE.txt) |
| HTML5 Sortable | Minified bytes match v0.9.16 (also shared by older tags); [source](https://github.com/lukasoppermann/html5sortable/tree/v0.9.16). | [MIT](licenses/HTML5SORTABLE-LICENSE.md) |
| Weather Icons | CSS identifies 2.0 and Erik Flowers; exact fonts/CSS are pinned by the Evolve snapshot. | [Font: SIL OFL 1.1](licenses/Weather-Icons-OFL.txt); CSS declares MIT in its retained header. |

Weather Icons' [upstream license declaration](https://github.com/erikflowers/weather-icons/tree/a85ba96d252d6729eb9a07aa2f0544049b291163)
distinguishes fonts, code, and documentation. Its font terms are not replaced by
Evolve's MPL. License texts are copied as notices; implementation files remain unchanged.
