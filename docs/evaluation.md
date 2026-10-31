# Running evaluations

This page takes a model from connection to results. The [reference](reference.md) lists every option
and field.

Run every command from the repository root.

## Connect a model

A preset is a file in `configs/models/` that fixes a model's route and settings, so runs stay
comparable. Pick the first case that fits:

| Your model | What to do |
| --- | --- |
| Already in `configs/models/` | Run it by name, as `-m gemini-3.8-flash` |
| Supported by [LiteLLM](https://docs.litellm.ai/docs/providers) | [Write a preset](#write-a-preset) with `api: litellm` |
| Behind your own OpenAI-compatible endpoint | [Write a preset](#write-a-preset) with `api: chat_completions` |
| Uses a new API protocol, such as a realtime session API | [Add a model API](custom_drivers.md#add-a-model-api) |
| Uses a new API protocol, and you keep the driver in your own package | [Use an external driver](custom_drivers.md#use-an-external-driver) (results are not official) |

The harness sends screenshots, so the model needs image input. Keys stay in `.env`, never in a
preset.

### Write a preset

Save the model as `configs/models/<name>.yaml` and run it with `-m <name>`. Start from
`configs/models/my-model.yaml`:

```yaml
endpoint:
  api: chat_completions            # how to reach the model (below)
  model: YOUR_MODEL                # the model id the endpoint expects
  base_url: https://YOUR_HOST/v1   # required for chat_completions
  api_key_env: OPENAI_API_KEY      # variable in .env holding the key
  revision: "2026-10-01"           # optional: deployment revision, recorded with every run
  native_output_limit: 32768       # optional: the model's maximum output tokens, reasoning included
reasoning_effort: high             # optional; `fixed` sends no reasoning setting
```

A LiteLLM model needs only its `provider/model-id` and its key:

```yaml
endpoint:
  api: litellm
  model: provider/model-id
  api_key_env: PROVIDER_API_KEY
reasoning_effort: low
```

`endpoint.api` selects how the request is sent:

| `api` | Use for | Extra fields |
| --- | --- | --- |
| `litellm` | Any LiteLLM `provider/model-id` | |
| `chat_completions` | Your OpenAI-compatible endpoint | `base_url` |
| `openrouter` | A model through OpenRouter, pinned to one upstream | `upstream_provider` |
| `anthropic_messages` | The Anthropic Messages API | `native_output_limit` |
| `package.module:Class` | Your own driver class ([external driver](custom_drivers.md#use-an-external-driver)) | |

- `chat_completions` streams its replies and asks for the usage at the end of the stream
  (`stream_options.include_usage`). Set `include_usage: false` for an endpoint that rejects
  `stream_options`, or `stream: false` for complete responses.
- `openrouter`: a model whose provider caches prompts only at explicit breakpoints, as Anthropic's
  and Gemini's do, sets `cache_breakpoint: true` in its [provider options](reference.md#provider-options), and
  each request then marks its newest message as a cache breakpoint. OpenAI's models cache on their
  own and leave it out.
- `anthropic_messages` marks every request as a cache breakpoint.

## Run a model

```bash
uv run rtbench run -m gemini-3.8-flash
```

`-m` takes a [preset](#connect-a-model) name such as `gpt-realtime-2`, or a LiteLLM
`provider/model-id` ([run without a preset](#run-without-a-preset)). Repeat it to evaluate several
models in one job. Put the provider's key in `.env` (for example `OPENROUTER_API_KEY`).

`rtbench` is Harbor's CLI plus `setup`, `report`, `suites`, `upload` ([submit](submit.md)) and a
viewer. Results go to `temp/harbor-jobs`. It turns off Harbor's usage telemetry; set
`HARBOR_TELEMETRY=1` in the shell to send it.

Common variations:

```bash
# Selected tasks; repeat -i, glob patterns are allowed
uv run rtbench run -m gemini-3.8-flash -i gomoku -i 'rt_*'

# Several models in one job
uv run rtbench run -m gemini-3.8-flash -m gpt-6-astra

# Name the results directory
uv run rtbench run -m gemini-3.8-flash --job-name first-run

# Show the resolved configuration without running
uv run rtbench run -m gemini-3.8-flash --print-config
```

Every option is in [run options](reference.md#run-options).

### Run without a preset

`-m provider/model-id` also runs a LiteLLM model, on LiteLLM's defaults. It cannot set a preset's
route, output limit or provider options, so its results can differ from a preset's.

- Put the provider's key in `.env` under the name LiteLLM expects (for example `OPENAI_API_KEY`).
- The output limit comes from LiteLLM's model map ([output limit](reference.md#output-limit)).
- `--ak reasoning_effort` is passed through unchanged, and the provider rejects a value it does not
  take. LiteLLM maps it per provider; OpenRouter, for example, receives `max` as `xhigh`. Since
  LiteLLM 1.106.1 it no longer refuses `reasoning_effort` for an `openrouter/` model that its map
  does not mark as a reasoning model: the request is sent and OpenRouter decides.

### Try two tasks first

```bash
uv run rtbench run -m gemini-3.8-flash -i gh_javascript_pong -i rt_accessible_trip_dispatch
```

Both trials should finish without setup, provider or verifier errors. A reward of 0 is a valid
unsuccessful run.

Harbor's agent timeout is 900 seconds per task. For `duration_minutes` above 5, raise it with
`--agent-timeout-multiplier`.

### Job files

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
uv run rtbench run -c configs/jobs/smoke.yaml -m gemini-3.8-flash
```

`rtbench` adds whatever the file omits (agent, environment, results directory). `-m`, `--suite` and
`-i` override the file.

## Choose a clock

| Clock | Time advances by | Select with |
| --- | --- | --- |
| Wall (default) | Real time: observation, network, inference and action latency all count | `--ak clock=wall` |
| Token | Provider-reported output tokens ÷ `tokens_per_frame` | `--ak clock=token --ak tokens_per_frame=64` |
| Pause | One frame per action; the action applies at the observed tick | `--ak clock=token --ak tokens_per_frame=unlimited` |

The wall clock is the benchmark setting. Token and pause are comparison settings and need complete
provider usage (`litellm` and `chat_completions` presets support the wall clock only).

Results under different clocks, time limits or frame rates are not comparable. The leaderboard runs
both frame rates at 1 and accepts only turn-based runs.

The plans in `configs/runs/` run them directly; `-m` replaces their models:

```bash
uv run rtbench run -c configs/runs/token-clock.yaml   # paused-clock.yaml for the pause clock
```

## Run jobs in parallel

Each job reserves a lane. Give concurrent jobs distinct lanes (`--ek lane=1`, `--ek lane=2`). Lanes
have separate browsers, displays and ports, but share provider quotas and host resources. Lanes
isolate state, not untrusted code.

## View results

```bash
uv run rtbench view ./temp/harbor-jobs --port 18081
```

If the port is taken, pass a range such as `--port 18081-18089`. Open a job, then a trial:
**Trajectory** shows screenshots, actions and feedback, and **Recording** plays the scored interval
with action overlays.

| Outcome | Meaning | Score |
| --- | --- | --- |
| `pass` | The task was solved | 1 |
| `fail` | The task ended unsolved | 0 |
| `timeout` | The scored time ran out | 0 |
| `error` | Provider or infrastructure failure, listed under **Errors** | 0 |

The success rate is passes over all tasks.

## Check usage and cost

- Only usage the provider reported counts. A count it did not report stays unknown, never 0.
- No model has a rate card. A cost in USD is the one the provider reported (OpenRouter's, which the
  Decisions APIs of Luna and JEV report with each question), else unknown.
- A Decision that takes its only option without asking, as on a frame that offers only a wait,
  makes no call: its usage and cost are a known 0, and it still counts as one response.
- A Decision costs what its questions did. One that failed after some of them counts their usage
  and cost, which leaves the totals incomplete.

A turn-based trial's `run_summary.json` keeps two totals: `cost_usd` for the responses the episode
settled, and `spend_usd` for every model call, the calls given up on included. Use `spend_usd` for
what a run cost. The fields are in [usage fields](reference.md#usage-fields).

## Streaming runs

```bash
uv run rtbench run -m gpt-realtime-2 --ak interaction_mode=streaming
```

A streaming run pushes frames into a live session (OpenAI Realtime, Gemini Live) on the wall clock.
It is reported separately from turn-based runs, and the leaderboard does not accept it.

A streaming trial's `run_summary.json`, and each streaming group of its `report.json`, carry a
`streaming` section:

- the frames captured, pushed into the live session, dropped (and why) and confirmed by the provider
- the actions that arrived, the age of the frame each one named and what became of each
- the replies started and given up

The episode trace records each frame and action on its own: a frame's capture and push under its
observation id, and the number it went into the session under, with when it was captured, written
and confirmed (`FrameSent`).

The two live APIs keep a session's context differently, and `context_control` says which way a run
went:

- OpenAI Realtime is exact: the harness keeps the newest `retained_frames` screenshots in the
  session and replaces older ones with a placeholder itself.
- Gemini Live is approximate: once the session reaches `compression_trigger_tokens`, the server
  compresses it down to `compression_target_tokens` without saying what it removed, and Gemini
  confirms no frame it received.

Both are [streaming settings](reference.md#streaming-settings) of the model.

The fields that differ are in [streaming fields](reference.md#streaming-fields).
