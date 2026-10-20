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

### Requirements

Every task must meet all three:

1. **RT-Perceive:** Relevant visible state changes after the initial observation; stale
   positions, board state, threats, resources, or opportunities can cause a mistake.
2. **RT-Deliberate:** the task must test the model's reasoning, not reaction speed alone, and
   whether it can control how long it reasons to fit the time available: thinking too long lets
   the world move on, thinking too little leads to poor decisions.
3. **RT-Act:** action usefulness changes with its execution time. Waiting, moving,
   defending, attacking, switching, or committing has different consequences as play evolves.

### Quality

A good task has:

- a scenario that reflects a real need, the way people actually use such an app or play such a
  game, rather than a contrived setup built only to be hard;
- an `instruction.md` that states everything that is scored and nothing that is not;
- a success criterion that passes if and only if the instruction was followed;
- content that depends on the task seed, so it cannot be memorized, and is reproducible for a
  given seed;
- no shortcut through browser chrome, URL changes or reading hidden state.

### Games

Games must come from sources whose license allows redistribution. Record the upstream URL,
revision, local changes and license in the task package.

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

### 1. Create a branch and a task directory

```bash
git checkout -b <task-id>
cp -r realtime_tasks/web/<similar-task> realtime_tasks/web/<task-id>   # or realtime_tasks/game/
```

Starting from a similar task is the quickest way to get a working skeleton. A task is a
[Harbor task](https://github.com/harbor-framework/harbor) directory:

```text
realtime_tasks/<web|game>/<task-id>/
├── instruction.md       # the model's goal
├── task.toml            # Harbor sections + [metadata]
├── environment/         # your Web app or game
└── tests/
    ├── evaluation.py    # success criterion
    └── test.sh          # Harbor verifier entry point (keep as copied)
```

### 2. task.toml

Keep the copied `[agent]`, `[verifier]` and `[environment]` sections. In `[metadata]`, set
`task_id` to the directory name and write a one-paragraph `description` for reviewers. The loader
reports any missing or invalid field.

### 3. instruction.md

The model's only prompt: the goal, the deadline and any irreversible step. Write it yourself and
include all the information the model needs and nothing more.

### 4. Environment

Build `environment/` however suits your task. Keep task-specific code inside the task directory
rather than in shared sources, and draw every content-affecting random value from the task seed.

### 5. tests/evaluation.py

The success criterion, evaluated on the task state frozen at the deadline.

### 6. List the task

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
or email **realtime-agents@utlists.utexas.edu**.
