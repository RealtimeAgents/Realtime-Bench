# Running evaluations

Run commands from the repository root. `uv run` uses the project environment; with pip, activate
your environment and drop the `uv run` prefix.

## Run

```bash
uv run rtbench run -m "provider/model-id"
```

`-m` takes a LiteLLM `provider/model-id` or a [model preset](model_configuration.md) name such as
`gpt-realtime-2`; repeat it to evaluate several models in one job. Put the provider's key in
`.env` (for example `OPENROUTER_API_KEY`).

`rtbench` is Harbor's CLI plus this benchmark's `setup`, `report`, `suites`, `upload`
([submit](submit.md)) and viewer; results go to `temp/harbor-jobs`. It turns off Harbor's usage telemetry (set `HARBOR_TELEMETRY=1` in the
shell to send it).

| Option | Use |
| --- | --- |
| `-i TASK_ID` | Run selected tasks of the suite (repeat `-i`; glob patterns allowed). |
| `--suite NAME[@VERSION]` | Suite from `realtime_tasks/sets/` (default: `public`). `public@1.0` refuses to run unless this checkout holds exactly that release. |
| `-c JOB.yaml` | Harbor job configuration; see [job files](#job-files). |
| `--ak duration_minutes=5` | Scored time limit per task in minutes (default: 5). |
| `--ak fps=1` / `--ak seed=42` | Environment frame rate and [task seed](#task-seed) (defaults shown). |
| `--ak model_input_fps=1` | Frames per second fed to the model, at most `fps`; `tokens_per_frame` counts these frames (default: `fps`). |
| `--ak interaction_mode=turn_based` | `turn_based` (default): one frame per model generation. `streaming` pushes frames into a live session (OpenAI Realtime, Gemini Live) on the wall clock and is reported separately; no model supports it yet. |
| `--ak clock=wall` | `wall` (default) or `token`; see [clocks](#clocks). |
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

Harbor's agent timeout is 900 seconds per task; for `duration_minutes` above 5, raise it with
`--agent-timeout-multiplier`.

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

`rtbench` adds whatever the file omits (agent, environment, results directory); `-m`, `--suite`
and `-i` override the file.

## View results

```bash
uv run rtbench view ./temp/harbor-jobs --port 18081
```

Open a job, then a trial. **Trajectory** shows screenshots, actions, and feedback;
**Recording** plays the scored interval with action overlays. Each trial ends as `pass` (reward
1), `fail` (the task ended unsolved), `timeout` (the scored time ran out) or `error` (provider or
infrastructure failure, listed under **Errors**). All but `pass` score 0, and the success rate is
passes over all tasks. If the port is taken, pass a
range such as `--port 18081-18089`.

## Parallel runs

Each job reserves a lane. Give concurrent jobs distinct lanes (`--ek lane=1`, `--ek lane=2`);
lanes have separate browsers, displays, and ports, but share provider quotas and host
resources. Lanes isolate state, not untrusted code.

## Clocks

| Clock | Time advances by | Select with |
| --- | --- | --- |
| Wall (default) | Real time: observation, network, inference and action latency all count | `--ak clock=wall` |
| Token | Provider-reported output tokens ÷ `tokens_per_frame` | `--ak clock=token --ak tokens_per_frame=64` |
| Pause | One frame per action; the action applies at the observed tick | `--ak clock=token --ak tokens_per_frame=unlimited` |

The wall clock is the benchmark setting; token and pause are comparison settings and need
complete provider usage (`litellm` and `chat_completions` presets support the wall clock only).
Results under different clocks, time limits or frame rates are not comparable; the leaderboard
runs both frame rates at 1 and accepts only turn-based runs. The plans in
`configs/runs/` run them directly; `-m` replaces their models:

```bash
uv run rtbench run -c configs/runs/token-clock.yaml   # paused-clock.yaml for the pause clock
```

## Task seed

`--ak seed=123` selects the task's random seed (unsigned 32-bit, default 42). It fixes task
content, not timing or provider sampling, and is part of the runtime hash.
