# Add a task

A task is a [Harbor task](https://github.com/harbor-framework/harbor) directory under
`realtime_tasks/web/` or `realtime_tasks/game/`. The model sees only `instruction.md` (plus a
game's `rules` and `keys`); everything else is for the runtime and for reviewers.

```text
realtime_tasks/<web|game>/<task-id>/
├── README.md            # for humans: what the task tests and how it was built
├── instruction.md       # the exact model goal
├── task.toml            # Harbor sections + [metadata]
├── environment/         # page, engine or vendored game
├── tests/evaluation.py  # success criterion
├── tests/test.sh        # Harbor verifier entry point
└── RIGHTS.md            # games only: rights review of vendored sources
```

## Steps

1. **Copy a similar task.** Start from an existing task of the same kind and backend; keep its
   `task.toml` Harbor sections (`[agent]`, `[verifier]`, `[environment]`) and `tests/test.sh`
   unchanged.
2. **Write `instruction.md`.** State the goal, deadline and any irreversible step. It is the only
   prompt source; do not put a `prompt` in `task.toml`.
3. **Fill `[metadata]`** (see [fields](#metadata-fields)).
4. **Build the environment.**
   - Web: `backend = "product"` puts business transitions in `environment/engine.py`;
     `configured` puts parameters in `environment/scenario.json`, an embedded world in
     `environment/world.py`, and record, payload or policy overrides in `environment/hooks.py`;
     `bespoke` puts a whole engine class (`Engine`) in `environment/engine.py`. Phone layout
     rules go in `environment/mobile.css`, linked after the shared `mobile.css`.
   - Game: vendor the upstream source unmodified under `environment/upstream/` and put benchmark
     changes in sibling files (launch page, `game_api.js`, `game_config.js`). Touch controls for
     phones are listed in `environment/mobile-profile.json`. Record the upstream
     URL, immutable revision, local changes, license text and asset origins in
     `environment/PROVENANCE.md` and `RIGHTS.md`.
5. **Draw randomness from the seed.** Web engines receive it in `reset(seed=...)`; use a local
   `random.Random` keyed by it, never the global `random` module. Games need no seed code: the
   runtime seeds `Math.random` and `crypto` in every document. A game that wants several streams
   restarts `Math.random` with `window.__restartRandomStream(state)`, deriving `state` from
   `window.__benchmarkSeed`. IDs and security tokens may stay unseeded.
6. **Write `tests/evaluation.py`.** The runner calls it on the live state frozen at the
   deadline; Harbor copies `tests/` into the trial but only runs `test.sh`, which checks the
   run's report and converts its result to Harbor's reward.
7. **Add the task to a suite** in `realtime_tasks/sets/<suite>.yaml`, adjusting the declared
   totals and category counts. That file alone defines membership and order. Add a row to
   [tasks](tasks.md) for a public task.
8. **Write `README.md`**: what the task tests, what changes during the episode, and what makes it
   hard. Do not repeat `description` or `instruction.md`; link them.
9. **Validate** (see [checklist](#checklist)).

## Metadata fields

All tasks:

| Field | Meaning |
| --- | --- |
| `task_id` | Must equal the directory name. |
| `category` | `realtime` (Web) or `browser_game` (Game). |
| `description` | One paragraph for reviewers and reports; not shown to the model. |
| `instruction` | `instruction.md`. |
| `evaluation` | `tests/evaluation.py`. |
| `entry` | `environment/index.html`. |
| `interactables` | Optional; bound read-only to every observation, with no inspection call. |

Web tasks add `backend` (`product`, `configured` or `bespoke`), `scenario`, `engine`
(`product`, `bespoke`), and `scenario_file` plus optional `world` and `hooks` (`configured`).
Task-specific code never goes into the shared `realtime_tasks/web/engines/` or `static/`.

Game tasks add:

| Field | Meaning |
| --- | --- |
| `game_key`, `tier` | `NN_name` and `NN_01`. |
| `idx`, `http_port`, `dbg_port` | Unique index; ports must be `8940 + idx` and `9240 + idx`. Ask a maintainer for a free `idx`, since private tasks also use them. |
| `score_field`, `target` | Read-only state field the evaluation compares, and its success value. |
| `keys`, `rules` | Allowed keys and the agent-visible rules appended to the instruction. |
| `source`, `player_mode` | Upstream `owner/repo @ revision`; `single`. |
| `deferred_start` | Optional; hold the game until the scored horizon starts. |
| `think_budget_s`, `per_move_cap_s` | Optional task-specific limits. |

## Checklist

Copy into the pull request:

- [ ] `uv run python -m realtime_tasks.preview` shows the task; reset, clock transitions and
      terminal conditions behave as described.
- [ ] `instruction.md` and `tests/evaluation.py` agree: everything scored is stated, and
      nothing stated is left unscored.
- [ ] Doing nothing (only `wait`) fails, and a run with a different seed changes the content.
- [ ] Success cannot be reached through browser chrome, URL changes or reading hidden state.
- [ ] One strong model was run on the task (`rtbench run -i <task-id>`); the failure or success
      is explained in the pull request.
- [ ] Games: upstream source unmodified, `PROVENANCE.md` and `RIGHTS.md` complete.
- [ ] `uv run rtbench suites` shows the expected suite change; the suite version is bumped as
      described in [maintainers](maintainers.md#versions).
