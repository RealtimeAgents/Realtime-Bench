# Norman the Necromancer rights

Selected for public-20, reviewed 2026-10-10.

- Upstream: https://github.com/danprince/norman-the-necromancer at
  `5db8ca88c8038b3d0888addaa521a16db3c259a8`.
- License: the Unlicense (public-domain dedication), preserved at
  `environment/upstream/LICENSE.md`.
- Assets: `src/sprites.png`, `font.png` and the `.aseprite` sources were drawn by the author, and
  all sound is procedural Web Audio code; the author's retrospective
  (https://danthedev.com/norman-the-necromancer/) states that no third-party assets were used.
- No runtime libraries are bundled; Vite, Terser and TypeScript are build tools only.
- Modified upstream files: `index.html` (fixed seed), `src/index.ts` (read-only state accessor)
  and `yarn.lock` (registry URLs). Added: `package-lock.json` and the `runtime/` build. See
  `environment/PROVENANCE.md`.
