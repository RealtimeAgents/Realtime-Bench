# Running evaluations

Run commands from the repository root. `uv run` uses the project environment; with pip, activate
your environment and drop the `uv run` prefix.

## Run

```bash
uv run rtbench run -m "provider/model-id"
```

`-m` takes a LiteLLM `provider/model-id` or a [model preset](models.md) name such as
`gpt-realtime-2`; repeat it to evaluate several models in one job. Put the provider's key in
`.env` (for example `OPENROUTER_API_KEY`).

`rtbench` is Harbor's CLI plus this benchmark's `setup`, `report`, `suites` (versions and suite
hashes), and viewer. `rtbench run` builds or reuses the task images, selects tasks from a suite in
`realtime_tasks/sets/`, and writes results to `temp/harbor-jobs`. Each task directory is a
Harbor task (`task.toml`, `instruction.md`, `environment/`, `tests/`).

`rtbench` turns off Harbor's usage telemetry, which would send each job's agent and model names
to Harbor's PostHog project; set `HARBOR_TELEMETRY=1` in the shell (not `.env`) to send it.

| Option | Use |
| --- | --- |
| `-i TASK_ID` | Run selected tasks of the suite (repeat `-i`; glob patterns allowed). |
| `--suite NAME[@VERSION]` | Suite from `realtime_tasks/sets/` (default: `public`). `public@1.0` refuses to run unless this checkout holds exactly that release. |
| `-c JOB.yaml` | Harbor job configuration; see [job files](#job-files). |
| `--ak duration_minutes=5` | Scored time limit per task in minutes (default: 5). |
| `--ak fps=1` / `--ak seed=42` | Observation rate and [task seed](#task-seed) (defaults shown). |
| `--ak clock=wall` | `wall` (default) or `token`; see [comparison clocks](#comparison-clocks). |
| `--ak reasoning_effort=high` | Reasoning effort supported by the model (default: the preset's `reasoning_effort`, else its API's default, `low` for most APIs; provider default for `-m provider/model-id` and `chat_completions` presets). |
| `--ak api_base=https://HOST/v1` | Custom endpoint, e.g. with `-m openai/MODEL` for an OpenAI-compatible service. |
| `--ak api_key_env=MY_KEY` | Read the credential from another variable (`provider/model-id` only; set a preset's in its YAML). |
| `--ak model_revision=REV` | Record the deployment revision (`provider/model-id` only; presets use `endpoint.revision`). |
| `--ek lane=2` | Use a separate [lane](#parallel-runs) for a concurrent job. |
| `--job-name NAME` | Name the job's results directory (default: Harbor's timestamp). |
| `--print-config` | Show the resolved Harbor configuration without running. |

Other options are passed to Harbor unchanged. Harbor records the resolved configuration in the
job directory as `config.json`.

## Try two tasks first

```bash
uv run rtbench run -m "provider/model-id" -i gh_javascript_pong -i rt_accessible_trip_dispatch
```

Both trials should finish without setup, provider, or verifier errors. A reward of 0 is a
valid unsuccessful run.

`duration_minutes` is the scored time limit, excluding setup and finalization, and is recorded
and checked in the report. Harbor's separate agent timeout is 900 seconds per task. For longer
runs, increase it with `--agent-timeout-multiplier` to cover the scored duration plus setup
and finalization. On the token and pause clocks the scored duration is game time, so `rtbench`
sets the agent timeout to 900 seconds plus 36 real seconds per scored second unless the job
file sets `override_timeout_sec`.

## Job files

A job file is an ordinary Harbor job configuration that points at the task folders:

```yaml
# configs/jobs/smoke.yaml
datasets:
  - path: realtime_tasks/game
    task_names: [gh_javascript_pong]
  - path: realtime_tasks/web
    task_names: [rt_accessible_trip_dispatch]
agents:
  - kwargs:
      suite: public
      duration_minutes: 5
```

```bash
uv run rtbench run -c configs/jobs/smoke.yaml -m "provider/model-id"
```

`rtbench` adds whatever the file omits (agent, environment, results directory). `-m` applies to
every agent in the file; `--suite` or `-i` replace its datasets. A trial records and verifies its
task against the agent's `suite`, and the report scores the job against that suite's full task
list.

## View results

```bash
uv run rtbench view ./temp/harbor-jobs --port 18081
```

Open a job, then a trial. **Trajectory** shows screenshots, actions, and feedback;
**Recording** plays the scored interval with action overlays. Reward is 1 for success and 0
for a valid failure; failed evaluations appear under **Errors**. If the port is taken, pass a
range such as `--port 18081-18089`.

## Parallel runs

Each job reserves a lane. Give concurrent jobs distinct lanes (`--ek lane=1`, `--ek lane=2`);
lanes have separate browsers, displays, and ports, but share provider quotas and host
resources. Lanes isolate state, not untrusted code.

## Comparison clocks

The pause and token clocks ([protocol](protocol.md#clocks)) are agent options:

```bash
uv run rtbench run -m claude-opus-5 -i gh_javascript_pong --ak clock=token --ak tokens_per_frame=64
```

Set `--ak tokens_per_frame=unlimited` for the pause clock. The plans in `configs/runs/` also run
directly; `-m` replaces their models:

```bash
uv run rtbench run -c configs/runs/token-clock.yaml   # paused-clock.yaml for the pause clock
```

Both record `clock_mode: token`; the pause clock is the one with `tokens_per_frame: null` in
`run_summary.json`.

## Task seed

`--ak seed=123` selects the task's random seed, an unsigned 32-bit integer (default 42). It
controls task initialization, not provider sampling, and fixes the random stream, not timing:
under the wall clock, frame timing and model latency can still make two runs diverge. The seed
is part of the runtime hash, so the report rejects jobs that mix seeds. How tasks consume it is
described in [maintainers](maintainers.md#task-seeding).
