# Timed Chess rights

Selected for public-20, reviewed 2026-10-10.

- Upstream: https://github.com/sagarbudhwani/classic-chess-js at
  `28f90f911db2d3638ff064795d67a87f1142b98b`.
- Game code: MIT (Sagar Budhwani), preserved at `upstream/LICENSE`.
- Chess pieces: the twelve PNGs in `upstream/pieces/` are byte-identical to
  Cburnett's `Chess_*t60.png` files on Wikimedia Commons, CC BY-SA 3.0, used unmodified. The
  attribution is in `licenses/CHESS-PIECES-ATTRIBUTION.md` and the license text in
  `licenses/CC-BY-SA-3.0.txt`. The upstream screenshot shows the same pieces.
- No fonts or audio are bundled.
- Vendoring: `upstream/` is the pinned GitHub source tree, unmodified. Sibling
  `index.html`, `game_config.js` and `game_api.js` are benchmark integration files.
