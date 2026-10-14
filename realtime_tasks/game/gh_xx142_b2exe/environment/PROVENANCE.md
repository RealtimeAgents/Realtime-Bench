# Source provenance

- **Game:** xx142-b2.exe
- **Canonical repository:** https://github.com/bencoder/js13k-2019
- **Pinned commit:** `39bd729b5f18f82bda6c7b1e094e70521c09fbfd`
- **Commit date:** 2019-10-14
- **License:** MIT (`upstream/LICENSE`)
- **Recognition:** first place overall and desktop in js13kGames 2019, as recorded by the
  upstream README and GitHub's 2019 competition highlights.
- **Source archive:** `upstream/source.tar.gz`, produced with `git archive` from the pinned
  canonical commit. Its SHA-256 is stored in `upstream/source.sha256`.

The game is self-contained and makes no runtime network requests.

## Source and asset audit

Runnable files are flattened from upstream `src/`; `PROVENANCE.json` records that
mapping, exact local hashes, and changed files. The original source archive and its
checksum are retained. See [local patches](PATCHES.md) for the three integration edits.
Geometry, levels, and procedural sounds are represented by those pinned source files.

`jsfxr.min.js` is the bundled jsfxr sound generator. Preserve its
[Apache-2.0 notice](licenses/JSFXR-LICENSE.txt) from
[mneubrand/jsfxr](https://github.com/mneubrand/jsfxr/blob/b2bfe50fddc7e281e1c1944870071b2ccd2c0440/LICENSE.txt)
alongside the game's MIT notice; its exact vendored bytes are identified by the game
snapshot rather than an inferred npm version.
