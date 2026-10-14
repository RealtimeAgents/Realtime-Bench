# Contributing to Realtime-Bench

Thanks for your interest in contributing. This guide walks you through creating, testing, and
submitting a task. Model API and framework contributions are covered at the
[end](#other-contributions).

**How to contribute a task:**

1. **Calibrate:** read the [task guidelines](#task-guidelines) and the [public tasks](docs/public_tasks.md).
2. **Propose:** share the idea with us before building it ([propose your task](#propose-your-task)).
3. **Create:** [implement](#create-your-task) the task and [test](#test-your-task) it locally.
4. **Submit:** [open a pull request](#submit-your-task).

## Table of contents

- [Task guidelines](#task-guidelines)
- [Propose your task](#propose-your-task)
- [Development environment setup](#development-environment-setup)
- [Create your task](#create-your-task)
- [Test your task](#test-your-task)
- [Submit your task](#submit-your-task)
- [What to expect after submitting](#what-to-expect-after-submitting)
- [Other contributions](#other-contributions)
- [Contact](#contact)

## Task guidelines

A Realtime-Bench task is a Web app or browser game whose world keeps changing while the agent
reasons. Success depends on both the action and when it arrives.

### Difficulty

- The challenge should come from deciding what to do and when in a changing world: live updates,
  competing events, deadlines, irreversible steps.
- Do not make a task hard through reaction speed alone. At one observation per second, a task
  that a simple scripted policy loses only on timing measures latency, not reasoning.
- Do not make a task hard through hidden rules, ambiguity or unnecessary UI friction.

### Quality

A good task has:

- an `instruction.md` that states everything that is scored and nothing that is not;
- a success criterion that passes if and only if the instruction was followed;
- content that depends on the task seed, so it cannot be memorized, and is reproducible for a
  given seed;
- no shortcut through browser chrome, URL changes or reading hidden state.

### Games

Games must come from sources whose license allows redistribution. Vendor the upstream source
unmodified and record its origin and license (see [environment](#4-environment)).

## Propose your task

Before investing time in building a task, [open an issue](#contact) describing the scenario,
what changes during the episode, what makes it hard, and how success is judged.

## Development environment setup

Requires Linux, [uv](https://docs.astral.sh/uv/) and Apptainer with fakeroot.

1. Fork [Realtime-Bench](https://github.com/RealtimeAgents/Realtime-Bench) and clone your fork:

   ```bash
   git clone https://github.com/<YOUR_GITHUB_USERNAME>/Realtime-Bench.git
   cd Realtime-Bench
   uv sync
   cp .env.example .env   # add a provider key for test runs
   ```

2. Check that the suites load: `uv run rtbench suites`.

## Create your task

### 1. Create a branch and copy a similar task

```bash
git checkout -b <task-id>
cp -r realtime_tasks/web/<similar-task> realtime_tasks/web/<task-id>   # or realtime_tasks/game/
```

A task is a [Harbor task](https://github.com/harbor-framework/harbor) directory:

```text
realtime_tasks/<web|game>/<task-id>/
├── instruction.md       # the model's goal
├── task.toml            # Harbor sections + [metadata]
├── environment/         # Web app or vendored game
└── tests/
    ├── evaluation.py    # success criterion
    └── test.sh          # Harbor verifier entry point (keep as copied)
```

### 2. task.toml

Keep the copied `[agent]`, `[verifier]` and `[environment]` sections. In `[metadata]`, set
`task_id` to the directory name and write a one-paragraph `description` for reviewers. Games
also declare their `rules`, `keys`, `score_field` and `target`, which the model sees; ask us for a
free `idx`. The loader reports any missing or invalid field.

### 3. instruction.md

The model's only prompt: the goal, the deadline and any irreversible step. Write it yourself and
include all the information the model needs and nothing more.

### 4. Environment

- **Web**, by `backend`:
  - `product`: business transitions in `environment/engine.py`.
  - `configured`: parameters in `environment/scenario.json`, optionally `world.py` and `hooks.py`.
  - `bespoke`: an `Engine` class in `environment/engine.py`.

  Keep task-specific code in the task; do not edit the shared `realtime_tasks/web/` sources for one
  task.
- **Game**: put the upstream source unmodified under `environment/upstream/` and benchmark changes
  in sibling files. Record the upstream URL, immutable revision, local changes, license text and
  asset origins in `environment/PROVENANCE.md` and `RIGHTS.md`.

### 5. Randomness

Draw every content-affecting random value from the task seed. Web engines receive it in
`reset(seed=...)`; use a local `random.Random`, never the global `random` module. Games are
seeded by the runtime (`Math.random` and `crypto`); see `realtime_tasks/game/assets/benchmark_seed.js`.

### 6. tests/evaluation.py

The success criterion, evaluated on the task state frozen at the deadline.

### 7. List the task

Add it to `realtime_tasks/sets/public.yaml`, updating the counts, and to [docs/public_tasks.md](docs/public_tasks.md).

## Test your task

1. **Preview** it in a browser and play it yourself: check reset, live updates, deadlines,
   terminal conditions, and that doing nothing fails.

   ```bash
   uv run python -m realtime_tasks.preview   # open http://127.0.0.1:18080/
   ```

2. **Try another seed** (`uv run python -m realtime_tasks.preview --seed 123`) and check that
   the content changes.
3. **Run a strong model** and read its trajectory in the viewer ([running](docs/evaluation.md)):

   ```bash
   uv run rtbench run -m "provider/model-id" -i <task-id>
   uv run rtbench view ./temp/harbor-jobs --port 18081
   ```

## Submit your task

Push to your fork and open a pull request. In the description:

1. Fill out the checklist in the [pull request template](.github/pull_request_template.md).
2. Summarize earlier feedback from the proposal issue.
3. Explain the model run: why it failed or succeeded, and why the failure reflects the task's
   difficulty rather than a specification problem.

## What to expect after submitting

1. CI checks run on every push: the task must be listed in `public.yaml` with the required
   metadata, and paths must be unique without regard to case.
2. A maintainer reviews the task, plays it, and approves or requests changes.
3. A maintainer merges it and handles the suite version and hash locks for the next release.

## Other contributions

- **Model APIs:** see [adding a model API](docs/model_configuration.md#add-a-model-api).
- **Framework code** lives in `realtime_framework/`. Keep pull requests focused, describe how you
  validated them, and say so explicitly if they change task objectives, timing or scoring.

## Contact

Open an issue in [RealtimeAgents/Realtime-Bench](https://github.com/RealtimeAgents/Realtime-Bench/issues)
or email **jingyu@utexas.edu**.
