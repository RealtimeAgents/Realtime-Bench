# Reference

Every option and field of a run. For the steps, see [running evaluations](evaluation.md).

## Run options

| Option | Use |
| --- | --- |
| `-i TASK_ID` | Run selected tasks of the suite. Repeat it; glob patterns are allowed. |
| `--suite NAME[@VERSION]` | Suite from `realtime_tasks/sets/` (default: `public`). `public@1.0` refuses to run unless this checkout holds exactly that release. |
| `-c JOB.yaml` | Harbor job configuration; see [job files](evaluation.md#job-files). |
| `--ak duration_minutes=5` | Scored time limit per task in minutes (default: 5). |
| `--ak fps=1` / `--ak seed=42` | Environment frame rate and [task seed](#task-seed) (defaults shown). |
| `--ak model_input_fps=1` | Frames per second fed to the model, at most `fps`; `tokens_per_frame` counts these frames (default: `fps`). |
| `--ak interaction_mode=turn_based` | `turn_based` (default): one frame per model generation. `streaming`: pushes frames into a live session (OpenAI Realtime, Gemini Live) on the wall clock, reported separately ([streaming runs](evaluation.md#streaming-runs)). |
| `--ak clock=wall` | `wall` (default) or `token`; see [clocks](evaluation.md#choose-a-clock). |
| `--ak reasoning_effort=high` | Reasoning effort supported by the model (default: the preset's `reasoning_effort`, else the provider default). |
| `--ak api_base=https://HOST/v1` | Custom endpoint, e.g. with `-m openai/MODEL` for an OpenAI-compatible service. |
| `--ak api_key_env=MY_KEY` | Read the credential from another variable (`provider/model-id` only; set a preset's in its YAML). |
| `--ak model_revision=REV` | Record the deployment revision (`provider/model-id` only; presets use `endpoint.revision`). |
| `--ak context_file=PATH` | An experiment file: a YAML mapping of [context settings](#context-settings) to values, for every model of the job. |
| `--ak context='{"reasoning_replay": true}'` | Context settings of the run, over the experiment file's and the model YAML's; values keep their JSON types. A run off the standard settings cannot be uploaded. |
| `--ek lane=2` | Use a separate [lane](evaluation.md#run-jobs-in-parallel) for a concurrent job. |
| `--job-name NAME` | Name the job's results directory (default: Harbor's timestamp). |
| `--print-config` | Show the resolved Harbor configuration without running. |

Other options go to Harbor unchanged. Harbor records the resolved configuration in the job
directory as `config.json`.

## Task seed

`--ak seed=123` selects the task's random seed (unsigned 32-bit, default 42). It fixes task content,
not timing or provider sampling, and is part of the runtime hash.

## Output limit

`native_output_limit` caps the output tokens of every request, reasoning included. Unless a token
clock budget is smaller, it is the limit each request sends. An external driver is the exception:
it gets no limit on the wall and paused clocks.

The shipped presets declare it, except Luna's and JEV's. A preset without it has no limit and gets
the provider default, except on `api: litellm`.

Direct runs, and presets on `api: litellm` without a limit, take the model's `max_output_tokens`
from the model map bundled with the locked LiteLLM version, looked up as `provider/model-id`. These
limits follow LiteLLM, so upgrading it changes them: from 1.104.1 to 1.106.1, 151 models gained,
lost or changed a limit and 54 limits went down, for example `openrouter/qwen/qwen3-235b-a22b-2507`
from 235929 to 16384 tokens. To keep a limit across upgrades, declare it in a preset.

The run records the resolved preset, limit included, so results with different settings are never
mixed.

## Context settings

Ten context settings decide what each model is shown at every step, and where.

- Standard values: `realtime_framework/context_defaults.yaml`, the only source of them.
- Definitions: `SETTINGS` in `realtime_framework/entity/context_policy.py`.
- A preset's top-level `context` (optional) sets the few a model needs instead: `reasoning_replay`,
  `tool_call_history`, `observation`, `execution_feedback` and `decision_history`, each only where
  it applies to the model's API.
- The defaults file lists the deviations some models are approved to take (`approved_deviations`).

The settings, and the deviations the defaults file approves:

<!-- Generated from SETTINGS and the defaults file by settings_table(). -->

| Setting | Standard | Values | Applies to | Kind | Model YAML | What it is |
| --- | --- | --- | --- | --- | --- | --- |
| `screenshots_kept` | `20` | a positive count | every API | evaluation condition | no | Screenshots kept in the history; one more folds the oldest screenshot_fold_block |
| `screenshot_fold_block` | `10` | a positive count | every API | evaluation condition | no | Screenshots folded together into a fixed text |
| `turns_kept` | `50` | a positive count | every API | evaluation condition | no | Turns kept verbatim; one more archives the oldest turn_archive_block into the ledger |
| `turn_archive_block` | `10` | a positive count | every API | evaluation condition | no | Turns archived together, also under context pressure, and archived actions left out together from a request that still does not fit |
| `execution_feedback` | `separate_message` | `separate_message`, `digest` | every API | what an API's channel can render | yes | Where each action's execution feedback goes: its own message, or the action digest |
| `observation` | `som_screenshot` | `som_screenshot`, `observation_json_screenshot`, `observation_json_text` | every API | what an API's channel can render | yes | What one observation carries: the SoM screenshot, or the observation JSON with or without the screenshot |
| `reasoning_replay` | `false` | `true`, `false` | every API | what a model's protocol may need | yes | Send the model's past reasoning back in later requests |
| `archive_over_window` | `true` | `true`, `false` | every API | evaluation condition | no | Archive the oldest turns while a request exceeds the declared context window |
| `tool_call_history` | `native_tools` | `native_tools`, `tool_transcript` | `gemini_live` | what a model's protocol may need | yes | How a rebuilt live session replays past tool calls: as native calls or as a transcript |
| `decision_history` | `turns` | `turns`, `action_digest` | `openai_decisions`, `jev_decisions` | what an API's channel can render | yes | How a Decision request replays history: as turns or as an action digest |

| API | Model | Settings | Required | Reason |
| --- | --- | --- | --- | --- |
| `moonshot` | `kimi-k3` | `reasoning_replay: true` | yes | Kimi's thinking-model guide requires K3's past reasoning_content back in multi-turn conversations and tool-call loops. |
| `litellm` | `moonshot/kimi-k3` | `reasoning_replay: true` | yes | Kimi's thinking-model guide requires K3's past reasoning_content back in multi-turn conversations and tool-call loops. |
| `gemini_live` | `gemini-3.1-flash-live-preview` | `tool_call_history: tool_transcript` | no | Past calls replayed as native tool calls can stall the next interaction of a rebuilt session (providers/gemini/session.py SessionCore._tool_contents). |
| `gemini_live` | `gemini-3.8-live` | `tool_call_history: tool_transcript` | no | Past calls replayed as native tool calls can stall the next interaction of a rebuilt session (providers/gemini/session.py SessionCore._tool_contents). |
| `gemini_live` | `gemini-3.8-live-extended-thinking` | `tool_call_history: tool_transcript` | no | Past calls replayed as native tool calls can stall the next interaction of a rebuilt session (providers/gemini/session.py SessionCore._tool_contents). |
| `openai_decisions` | `openai/gpt-6-luna-decisions` | `observation: observation_json_screenshot` | no | Luna chooses among the candidates the harness lists in the observation JSON, which it keeps beside the screenshot; it is not compared with the chat models. |
| `jev_decisions` | `typesafe/jev-1.13` | `observation: observation_json_text`, `execution_feedback: digest`, `decision_history: action_digest` | no | JEV takes text only, within a 32k prompt: an action digest with the feedback in it, and the observation JSON without a screenshot; it is not compared with the chat models. |

<!-- End of the generated tables. -->

A run may set any setting for every model. Each source below overrides the one before it:

| Order | Source | Trial worker | Harbor |
| --- | --- | --- | --- |
| 1 | Experiment file: a YAML mapping of settings to values | `--context-file PATH` | `--ak context_file=PATH` |
| 2 | The model's YAML | | |
| 3 | The run's command line, the value read as JSON | `--context KEY=VALUE`, repeatable | `--ak context='{"KEY": VALUE}'` |

- A setting that does not apply to a model's API is kept and recorded as not applying.
- A setting the API cannot render is refused before the run.
- The `STREAM_*` retention variables and `BENCHMARK_CONTEXT_PRESET` are refused: give the settings
  that replaced them.
- The run records each setting's value, where it came from, what it replaced and whether it is
  standard.

**Kimi K3.** `configs/models/kimi-k3.yaml` sets `reasoning_replay: true`, which sends K3's past
reasoning back, as Kimi requires for K3. Other moonshot models leave it out and send none. The
defaults file marks it required for K3, on `moonshot` and as LiteLLM's `moonshot/kimi-k3`: K3
without it is refused before the run. K3 behind your own endpoint (`api: chat_completions`, served
by vLLM or SGLang) sets it too. An endpoint given on the trial worker's command line has no YAML
(`python -m realtime_framework.evaluation._trial --model DEPLOYMENT_ID --base-url URL`) and sets it
with `--context reasoning_replay=true`.

The run records each context-policy value as the policy sets it, and two of them say what the
policy allows rather than what the run does:

- **Pressure archiving** is on for every API, but runs only where the driver estimates its requests
  (it overrides `request_context_status`) and the model has a context window
  (`context_window_tokens`, from the driver's model defaults or `BENCHMARK_CONTEXT_WINDOW_TOKENS`).
  There:
  1. A request over the window minus its reserve archives the oldest turns, a block at a time.
  2. With no turn left, the earliest archived actions give way to a note in the ledger, a block at
     a time, and stay out for the rest of the episode, as archived turns do.
  3. A request that does not fit with every archived action left out is not sent, and the run is
     invalid (`context_exceeded_before_request`).

  On `chat_completions` the reserve (`BENCHMARK_CONTEXT_RESERVE_TOKENS`, 4096 by default) must be
  at least the endpoint's `native_output_limit`, which each request sends, or the run is refused.
- **Ledger role.** Luna's driver, which has no roles, places the archived-action ledger by position
  and header, whatever ledger role the policy gives. JEV's, which has none either, labels the
  ledger's entry with that role.

Whether either takes effect is not recorded on its own: it follows from the driver code, which the
harness fingerprint covers (an external driver's identity, for an external driver), and from the
model configuration, which the run hash covers.

## Reasoning efforts

`reasoning_efforts` (top level, optional) lists the `reasoning_effort` values a model takes when it
rejects some that its API would send. `configs/models/gemini-3.8-live.yaml` takes only
`provider_default` and `fixed`, because 3.8 Live rejects a thinking setting. The preset's
`reasoning_effort` must be one of them, and a run asking for another is refused before it starts,
from the command line, a run plan or Harbor.

## Provider options

`provider_options` (top level) sets wire options of the API that its driver declares
(`compat.provider_options`):

| `api` | Options |
| --- | --- |
| `openrouter` | `cache_breakpoint`: `true` or `false` (default) |
| `gemini_live` | `tool_behavior`, `completion_signal`, `input_transport`, `tool_response_scheduling` |

A Gemini Live model must set all four: a model without them, a raw model id among them, is refused
before the run. The allowed values are in `LIVE_OPTIONS` (`realtime_framework/providers/gemini/live.py`),
and the Gemini presets in `configs/models` set the combinations their models need. How past function
calls are replayed into a rebuilt session is the context setting `tool_call_history` (`native_tools`
or `tool_transcript`), which the Gemini presets set to `tool_transcript` in their `context`.

## Streaming settings

`streaming` (top level, optional) sets what a live API keeps of a native-streaming session
(`--ak interaction_mode=streaming`). Each value is a positive integer, and a key left out keeps the
API's default (`streaming_settings` of its driver):

| `api` | Setting | Meaning |
| --- | --- | --- |
| `openai_realtime` | `retained_frames` | The newest frames that keep their screenshots |
| `gemini_live` | `compression_trigger_tokens` | The context size at which the server compresses the session |
| `gemini_live` | `compression_target_tokens` | The size it compresses to, which must be smaller |

```yaml
endpoint:
  api: openai_realtime
  model: gpt-realtime-2
  api_key_env: OPENAI_API_KEY
reasoning_effort: low
streaming:
  retained_frames: 10
```

Only a streaming run uses and records them: its configuration lists every setting with the value it
ran with, and a turn-based run's configuration has none. To compare values, save each as its own
preset, so the runs keep apart by model name too.

## Usage fields

A turn-based trial's `run_summary.json` keeps two totals:

| Total | Fields | Counts |
| --- | --- | --- |
| Settled responses | `token_usage`, `token_accounting`, `cost_usd`, `cost_accounting` | Each call whose result came back before it was given up on, and each call given up on before that at the per-move cap, the deadline or the provider watchdog. The second kind has no usage, which leaves the totals incomplete. |
| Every call | `spend_usd`, `spend_known_usd`, `spend_accounting` | The same calls, plus the calls given up on. A call given up on before its result was settled counts the usage its result reported if it came back while the episode's abandoned calls were drained (a Decision's, that of the questions it answered), and otherwise counts as in the settled total. |

A call given up on before its result came back because the task ended or the run was invalidated
is not in the settled total. The provider may still bill it, so that total can be slightly low even
when marked complete.

`spend_accounting` holds:

| Field | Meaning |
| --- | --- |
| `responses` | Responses counted; `given_up_responses` of them are calls given up on whose usage came back |
| `status`, `complete` | Whether the total is complete |
| `total_usd` | Total cost, only when every response has a cost |
| `known_usd` | The sum of the costs known |
| `tokens` | The provider's input, output and total tokens, each with `total`, `known_total` and `complete` |

A call's own retries are not counted.

A trial's `artifacts/logs/artifacts/report.json` holds:

| Level | Field | Meaning |
| --- | --- | --- |
| Task | `known_spend_usd`, `spend_complete` | Known spend and whether it is complete |
| Task | `spend_usd` | Spend, when scored |
| Group | `mean_spend_usd` | Mean known spend of its scored tasks with one, over `spend_tasks` tasks; a lower bound when `spend_incomplete_tasks` is above 0 |
| Group | `invalid_known_spend_usd` | Known spend of its invalid tasks |
| Group | `mean_cost_usd` | Mean over the tasks whose cost is complete |

Archived retry attempts list theirs under `retry_attempts`.

A turn-based trial's `run_summary.json` also counts the past reasoning its requests sent back (the
[context setting](#context-settings) `reasoning_replay`), in `reasoning_replay`:

| Field | Meaning |
| --- | --- |
| `requests` | Generation requests recorded as sent, each attempt and resend its own, including one that failed to connect, since a request is recorded before it is sent |
| `requests_with_reasoning` | Those that carried at least one past reasoning |
| `replayed_reasoning` | The past reasonings over all of them; each request's own count is its `ModelRequestSent` record's `replayed_reasoning` in the episode trace |
| `exact_wire` | False where the counts are of LiteLLM's final request dict rather than the bytes sent (the generic `litellm` API, as `-m provider/model-id` runs); those counts miss a resend LiteLLM's own HTTP handler makes after a connection error |

Only LiteLLM's APIs record these counts; for the others the section is null.

## Streaming fields

A streaming episode has no turns, so some fields mean nothing:

| Field | Value in a streaming run |
| --- | --- |
| `decisions`, `avg_delta_ticks`, `dropped_intermediate_frames`, `avg_frame_age_at_action_s`, `max_frame_age_at_action_s`, `token_usage`, and the metrics drawn from the decision lifecycle, which it does not write | 0 or empty |
| `model_timing` | `not_applicable` |
| `wall_time_s` | Empty |
| `cost_accounting.reason`, `spend_accounting.reason` | `not_accounted_streaming` |

Read the `streaming` section instead. The accepted replies' tokens are in `adapter_token_usage`, a
lower bound. For the cost, sum the usage the provider reported for every reply, those given up
included, from the episode trace's session messages. The report scores a streaming run only where
its streaming section is the one its episode trace gives.
