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
| `--ak interaction_mode=turn_based` | `turn_based` (default): one frame per model generation. `streaming` pushes frames into a live session (OpenAI Realtime, Gemini Live) on the wall clock and is reported separately ([streaming runs](#streaming-runs)). |
| `--ak action_space=gui` | `gui` (default): GUI actions only. `gui+cli` adds `run_script`, a script in a network-less workspace that acts through the action gateway; turn-based wall clock only, never on the leaderboard ([scripts](#scripts)). |
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

## Usage and cost

Only usage the provider reported counts, and a count it did not report stays unknown, never 0. A
Decision that takes its only option without asking, as on a frame that offers only a wait, makes
no call: its usage and cost are a known 0, and it still counts as one response. No model has a
rate card: a cost in USD is the one the provider reported (OpenRouter's, which the Decisions APIs
of Luna and JEV report with each question), else unknown. A Decision costs what its questions
did, and one that failed after some of them counts their usage and cost, which leaves the totals
incomplete. A turn-based trial's `run_summary.json` keeps two totals:

- `token_usage`, `token_accounting` and `cost_usd` (`cost_accounting`) count the responses the
  episode settled: each call whose result came back before it was given up on, and each call
  given up on before that at the per-move cap, the deadline or the provider watchdog, without
  usage, which leaves them incomplete. A call given up on before its result came back because
  the task ended or the run was invalidated does not count. The provider may still bill such a
  call, so the totals can be slightly low even when marked complete.
- `spend_usd`, `spend_known_usd` and `spend_accounting` count what every model call spent, the
  calls given up on included: a call given up on before its result was settled counts the usage
  its result reported if it came back while the episode's abandoned calls were drained (a
  Decision's, that of the questions it answered), and otherwise counts as above.
  `spend_accounting` has `responses` (of them `given_up_responses`, calls given up on whose
  usage came back), `status` and `complete`, `total_usd` (only when every response has a cost),
  `known_usd` (the sum of the costs known), and `tokens`: the provider's input, output and total
  tokens, each with `total`, `known_total` and `complete`. A call's own retries are not counted.

In the job's report, each task has `known_spend_usd`, `spend_complete` and, when scored,
`spend_usd`; each group has `mean_spend_usd`, the mean known spend of its scored tasks with
one, over `spend_tasks` tasks, of which `spend_incomplete_tasks` are incomplete (the mean is
then a lower bound), and `invalid_known_spend_usd`, the known spend of its invalid tasks.
`mean_cost_usd` stays the mean over the tasks whose cost is complete. Archived retry attempts
list theirs under `retry_attempts`.

## Streaming runs

A streaming trial's `run_summary.json`, and each streaming group of the job's report, carry a
`streaming` section: the frames captured, pushed into the live session, dropped (and why) and
confirmed by the provider; the actions that arrived, the age of the frame each one named and what
became of each; and the replies started and given up. The episode trace records each frame and
action on its own: a frame's capture and push under its observation id, and the number it went
into the session under, with when it was captured, written and confirmed (`FrameSent`).

A streaming episode has no turns, so the fields of `run_summary.json` that count them stay 0 or
empty: `decisions`, `avg_delta_ticks`, `dropped_intermediate_frames`,
`avg_frame_age_at_action_s` and `max_frame_age_at_action_s`, `token_usage` and the metrics drawn
from the decision lifecycle, which it does not write. Its replies are no model calls to time
either: `model_timing` is `not_applicable` and `wall_time_s` empty. Read the `streaming` section instead; the
accepted replies' tokens are in `adapter_token_usage`. Its cost and spend are not accounted
either (`cost_accounting.reason` and `spend_accounting.reason` are `not_accounted_streaming`):
sum the usage the provider reported for every reply, those given up included, from the episode
trace's session messages; the accepted replies' tokens are a lower bound. The report scores a streaming run only where its
streaming section is the one its episode trace gives.

The two live APIs keep a session's context differently, and `context_control` says which way a
run went. OpenAI Realtime's is exact: the harness keeps the newest `retained_frames` screenshots
in the session and replaces older ones with a placeholder itself. Gemini Live's is approximate:
once the session reaches `compression_trigger_tokens`, the server compresses it down to
`compression_target_tokens` without saying what it removed, and Gemini confirms no frame it
received. Both are [streaming settings](model_configuration.md#streaming-settings) of the model.

## Scripts

`--ak action_space=gui+cli` (or `action_space: gui+cli` in a run plan's `runtime`, as in
`configs/runs/gui-cli.yaml`) lets a text JSON or function-call model answer a turn with
`run_script`: a Python or bash script that runs in the episode's workspace while the world keeps
running, and observes and acts on the task only through `rtact` (`realtime_framework/workspace/
rtact.py`). Each `rtact` call goes over a unix socket to the action gateway, which accepts a
validated GUI action or a screenshot request and nothing else; the runner executes the action
exactly as it executes the model's own, with the end check right after it. The script's exit
code, its gateway actions and the tail of its output come back as that turn's EXECUTION RESULT;
the episode trace keeps its full record (`ScriptFinished`), and the workspace stays in the run
directory.

Scripts run in the workspace image (`realtime_framework/sandbox/images/workspace.def`: Python,
Pillow and bash, nothing of the task's), which `rtbench setup` builds beside the task images. Its
instance starts before the episode's clock, with `--containall --cleanenv --net --network none`:
no route to the task's services, its state API, Chrome's devtools port or the X display, which
the task sandbox serves on the host network, and no host files, `/sys` or `/dev/pts`. Each script
is an `apptainer exec` into that instance, so it pays only for joining it, and every process it
leaves is killed when it ends. A script is given `/workspace` and, read-only, `/rtact`: the
gateway's socket and the `rtact` client. A run's scripts always run there; the unisolated local
backend is the tests' alone, and a run refuses `RTBENCH_WORKSPACE_BACKEND=local`.
`tests/workspace/test_isolation.py` checks the isolation on a host with Apptainer.

A gui+cli run's configuration hash differs from the GUI-only run's, so its results are never
mixed with them, and `rtbench submit` refuses them. The script's timeout (`timeout_s`, default
30 s, at most 300 s), the task's end and the deadline each kill it, with every process it started.

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
