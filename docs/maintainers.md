# Maintainers

## Private tasks

The held-out tasks, the `private` suite, and the maintainer `tasks` union (all 50 tasks) live in
the private repository `RealtimeAgents/Realtime-Bench-Private`. Its `install.sh` copies them into
a checkout of this repository and lists them in `.git/info/exclude`; task paths are unchanged, so
suite hashes match the locks. This repository keeps only `realtime_tasks/sets/public.yaml` and
its tasks, which the *Public tasks* workflow enforces.

## Versions

The suites (`public`, `private`, and `tasks`) are released together under one `MAJOR.MINOR`
version. Each manifest declares that `version` and locks the released `suite_hash`. A checkout
whose task content no longer matches the lock reports `<version>+dev`; such runs are labeled as
such and the official private report rejects them.

- Bump MAJOR when results stop being comparable: tasks are added or removed, or a target,
  time limit or scoring rule changes.
- Bump MINOR for fixes meant to keep difficulty unchanged, such as a game bug or a rule
  clarification.

The suite hash covers the manifest, the selected task packages (every file, including READMEs),
and the shared task execution sources. The harness is versioned the same way in `pyproject.toml`
`[tool.rtbench.harness]`. Its hash covers
`realtime_framework/{entity,harnesses,providers,evaluation,sandbox}` and the locked runtime
dependency closure from `uv.lock`, not `configs/` (each run's model and runtime settings are
recorded in `runtime_config_hash`) or dev tooling. Bump its MAJOR when framework changes make
results incomparable and its MINOR for compatible fixes.

To release:

1. Install the private tasks and set the same new `version` in every manifest of both
   repositories.
2. Copy each `suite_hash` and the harness `hash` printed by `uv run rtbench suites`.
3. Merge both repositories and tag each merge commit `rtbench-vMAJOR.MINOR`.

Earlier versions are reproduced by checking out their tag; `--suite public@1.0` verifies the
checkout.

## Sandbox images

`rtbench run` builds `.cache/apptainer/{sandbox,realtime}.sif` from
`realtime_framework/sandbox/images/*.def` and rebuilds them when their sources change. Builds are
pinned: base images by digest, Ubuntu/Debian packages through one dated archive snapshot
(`APT_SNAPSHOT`), Google Chrome by version and SHA-256 (`CHROME_*`), and Python packages by
hashed lock files (`guest_server/requirements.txt`, `images/realtime-requirements.txt`, compiled
from the neighbouring `.in` files with `uv pip compile --universal --generate-hashes`). arm64
hosts use Debian's Chromium from the same snapshot, so their browser differs from amd64.

The sandbox carries `org.rtbench.*` labels (browser, OS, base image, snapshot, package-list
hashes); see them with `apptainer inspect --labels .cache/apptainer/sandbox.sif`. Each run records
them with the image SHA-256s under `container_runtime` in `run_config.json`, inside the runtime
hash, so reports reject jobs that mix images. A prebuilt `SANDBOX_IMAGE` without these labels
is refused.

Google eventually drops old Chrome packages. Keep a copy of the pinned `.deb` and set
`RTBENCH_CHROME_DEB_URL` to it when building; the SHA-256 check still applies. To move to a new
Chrome, update `APT_SNAPSHOT` and all `CHROME_*` arguments in `sandbox.def` together (the
SHA-256 is in Google's `dists/stable/main/binary-amd64/Packages` index).

## Task seeding

Web tasks derive their world from the seed in the engine. Web restart retains the seed while
rotating the episode generation and security tokens.

For browser games the runtime registers `realtime_tasks/game/assets/benchmark_seed.js`, bound to
the seed, in Chrome before the first game document loads, so it runs ahead of every game script
in each document, frame and reload: `Math.random`, `crypto.getRandomValues`/`randomUUID` and
argument-less `seedrandom()` all draw from it, and it publishes the seed as
`window.__benchmarkSeed`. Games carry no PRNG of their own. The runner rejects an episode whose
applied seed differs from the requested one and records `applied_seed` next to `seed`.

`gomoku` has fixed content; `gh_synthblast` builds its arenas from layout seed `0x12345678` and
uses the task seed only for the rest of its randomness. The private repository lists its own
fixed-content games. The page preview
(`uv run python -m realtime_tasks.preview --seed 123`) inlines the same script into game pages
and resets Web tasks with the same seed.
