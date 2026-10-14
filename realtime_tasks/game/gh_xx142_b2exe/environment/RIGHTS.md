# xx142-b2.exe rights review

Selected for public-20, reviewed 2026-10-08.
Source: https://github.com/bencoder/js13k-2019/tree/39bd729b5f18f82bda6c7b1e094e70521c09fbfd

Original game code, procedural WebGL visuals and levels: MIT, copyright 2019 Ben
Clark and Salvatore Previti (`environment/upstream/LICENSE`). No external image,
font or recorded audio file is loaded by the game. The source archive contains its
original level authoring files and screenshots as well as the build configuration.

The game is **not wholly MIT**. `jsfxr.min.js` is the mneubrand/jsfxr synthesizer,
whose exact game submodule commit is `b2bfe50fddc7e281e1c1944870071b2ccd2c0440`.
The full Apache-2.0 text and original authoring source with Thomas Vian's 2010
copyright notices are now included. The source restores attribution lost in
compression. Matrix rotate/translate routines follow gl-matrix; the Brandon Jones /
Colin MacKenzie IV MIT notice accompanies those adapted routines. Version 2.7.1
documents that matching lineage, not a recovered original import commit.

Local upstream changes in `game.js`, `main.js` and `styles.css` are identified in
`environment/PATCHES.md` and `environment/source-audit.json`. They remain MIT.
Original benchmark contributions have `environment/ADAPTER-LICENSE.txt`.
Preserve the Apache terms, all copyright notices and the MIT game license together.
The existing gameplay was not changed during this review. See `environment/LICENSES.html`.
