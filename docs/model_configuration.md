# Model configuration

How to configure a new model. Pick the first case that fits:

| Your model | What to do |
| --- | --- |
| Supported by [LiteLLM](https://docs.litellm.ai/docs/providers) | [Run it directly](#run-it-directly) with `-m provider/model-id` |
| Behind your own OpenAI-compatible endpoint | [Write a preset](#write-a-preset) with `api: chat_completions` |
| Needs fixed settings (routing, revision, effort, output limit) | [Write a preset](#write-a-preset) |
| Uses a new API protocol, such as a realtime session API | [Add a model API](#add-a-model-api) |
| Uses a new API protocol, and you keep the driver in your own package | [Use an external driver](#use-an-external-driver) (results are not official) |

The harness sends screenshots, so the model needs image input. Keys stay in `.env`, never in a
preset.

## Run it directly

```bash
uv run rtbench run -m "provider/model-id" --ak reasoning_effort=high
```

Put the provider's key in `.env` under the name LiteLLM expects (for example `OPENAI_API_KEY`).
Other agent options:

| Option | Use |
| --- | --- |
| `--ak api_base=https://HOST/v1` | Custom endpoint, e.g. with `-m openai/MODEL` |
| `--ak api_key_env=MY_KEY` | Read the key from another variable |
| `--ak model_revision=REV` | Record the deployment revision |
| `--ak reasoning_effort=EFFORT` | Passed through unchanged; the provider rejects unsupported values (default: provider default) |

The output limit of a direct run comes from LiteLLM's model map ([output limit](#output-limit)).
LiteLLM maps `reasoning_effort` per provider; OpenRouter, for example, receives `max` as `xhigh`.
Since LiteLLM 1.106.1 it no longer refuses `reasoning_effort` for an `openrouter/` model that its
map does not mark as a reasoning model: the request is sent and OpenRouter decides.

## Write a preset

A preset saves a model's connection as `configs/models/<name>.yaml`; run it with `-m <name>`.
Start from `configs/models/my-model.yaml`:

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

`endpoint.api` selects how the request is sent:

| `api` | Use for | Extra fields |
| --- | --- | --- |
| `litellm` | Any LiteLLM `provider/model-id` | |
| `chat_completions` | Your OpenAI-compatible endpoint | `base_url` |
| `openrouter` | A model through OpenRouter, pinned to one upstream | `upstream_provider` |
| `anthropic_messages` | The Anthropic Messages API | `native_output_limit` |
| `package.module:Class` | Your own driver class ([external driver](#use-an-external-driver)) | |

A `chat_completions` endpoint streams its replies and asks for the usage at the end of the stream
(`stream_options.include_usage`). Set `include_usage: false` for an endpoint that rejects
`stream_options`, or `stream: false` for complete responses.

An `openrouter` model whose provider caches prompts only at explicit breakpoints, as Anthropic's
and Gemini's do, sets `cache_breakpoint: true` in its [provider options](#provider-options): each
request then marks its newest message as a cache breakpoint. OpenAI's models cache on their own
and leave it out. `anthropic_messages` marks every request.

### Output limit

`native_output_limit` caps the output tokens of every request, reasoning included; unless a token
clock budget is smaller, it is the output limit each request sends. Every preset in
`configs/models/` declares it. A preset without it has no limit and gets the provider default,
except on `api: litellm`.

Direct runs, and presets on `api: litellm` without a limit, take the model's `max_output_tokens`
from the model map bundled with the locked LiteLLM version, looked up as `provider/model-id`.
These limits follow LiteLLM, so upgrading it changes them. From LiteLLM 1.104.1 to 1.106.1, 151
models gained, lost or changed a limit and 54 limits went down, for example
`openrouter/qwen/qwen3-235b-a22b-2507` from 235929 to 16384 tokens. To keep a limit across
upgrades, declare it in a preset. The run records the resolved preset, limit included, so results
with different settings are never mixed.

### Context preset

`context_preset` (top level, optional) names a set of context-policy values the model takes in
place of the standard ones. A model can name only `kimi_k3` (`MODEL_PRESETS` in
`realtime_framework/entity/context_policy.py`), which sends K3's past reasoning back, as Kimi
requires for K3: `configs/models/kimi-k3.yaml` sets it, and other moonshot models leave it out and
send none. K3 behind your own endpoint (`api: chat_completions`, served by vLLM or SGLang) names it
too; no API sends past reasoning without a preset. An endpoint given on the trial worker's command
line has no YAML (`python -m realtime_framework.evaluation._trial --model DEPLOYMENT_ID
--base-url URL`) and names it with `--context-preset kimi_k3`. A preset the API cannot render is
refused before the run, and the run records the preset with its configuration.

The run records each context-policy value as the policy sets it, and two of them say what the
policy allows rather than what the run does. Pressure archiving is on for every API, but it runs
only where the driver estimates its requests (`request_size_estimation`) and the model has a
context window (`context_window_tokens`, from the driver's model defaults or
`BENCHMARK_CONTEXT_WINDOW_TOKENS`). There, a request over the window minus its reserve archives
the oldest turns, a block at a time. On `chat_completions` the reserve
(`BENCHMARK_CONTEXT_RESERVE_TOKENS`, 4096 by default) must be at least the endpoint's
`native_output_limit`, which each request sends, or the run is refused. With no turn left, the earliest archived actions give way to a note in the
ledger, a block at a time, and stay out for the rest of the episode, as archived turns do. A
request that does not fit with every archived action left out is not sent, and the run is
invalid (`context_exceeded_before_request`). Luna's driver, which has no roles, places the
archived-action ledger by position and header, whatever ledger role the policy gives; JEV's, which
has none either, labels the ledger's entry with that role. Whether either takes effect is not recorded on its own: it follows from the driver code, which the
harness fingerprint covers (an external driver's identity, for an external driver), and from the
model configuration, which the run hash covers.

### Reasoning efforts

`reasoning_efforts` (top level, optional) lists the `reasoning_effort` values a model takes when
it rejects some that its API would send: `configs/models/gemini-3.8-live.yaml` takes only
`provider_default` and `fixed`, because 3.8 Live rejects a thinking setting. The preset's
`reasoning_effort` must be one of them, and a run asking for another is refused before it starts,
from the command line, a run plan or Harbor.

### Provider options

`provider_options` (top level) sets wire options of the API that its driver declares
(`compat.provider_options`). Of the registered APIs, `openrouter` takes `cache_breakpoint` (`true`
or `false`, the default), and `gemini_live` takes the options below; a Gemini Live model sets each
of `tool_behavior`, `completion_signal`, `input_transport`, `tool_response_scheduling` and
`history_format` (`native_tools` or `tool_transcript`: how past function calls are replayed into
a rebuilt session): a model without them, a raw model id among them, is refused before the run.
The allowed values are in `LIVE_PROTOCOL_OPTIONS` and `LIVE_HISTORY_OPTIONS`
(`realtime_framework/providers/gemini/live.py`), and the Gemini presets in `configs/models` set
the combinations their models need.

### Streaming settings

`streaming` (top level, optional) sets what a live API keeps of a native-streaming session
(`--ak interaction_mode=streaming`): `retained_frames` for `openai_realtime`, the newest frames
that keep their screenshots, and `compression_trigger_tokens` and `compression_target_tokens` for
`gemini_live`, the context size at which the server compresses the session and the size it
compresses it to, which must be smaller. Each is a positive integer, and a key left out keeps the
API's default (`streaming_settings` of its driver):

```yaml
endpoint:
  api: openai_realtime
  model: gpt-realtime-2
  api_key_env: OPENAI_API_KEY
reasoning_effort: low
streaming:
  retained_frames: 10
```

Only a streaming run uses and records them: its configuration lists every setting with the value
it ran with, and a turn-based run's configuration has none. To compare values, save each as its
own preset, so the runs keep apart by model name too.

## Add a model API

A model that needs a new wire protocol gets a driver:

1. Write a driver in `realtime_framework/providers/` implementing `RealtimeDriver`
   (`realtime_framework/entity/model.py`): `connect`, `send_observation`, `start_generation`,
   `events`, `interrupt`, `sync_history`, `close`, plus `bind_trace(trace)`, which receives the
   episode trace. The harness builds it as `Driver(config, clock=clock)`, with the episode's
   operational clock (`realtime_framework/clock.py`); an external driver whose constructor cannot
   take `clock` is refused when its model is loaded, and the tests hold every registered driver
   to the same check. Keep it as `self.clock`: the capture hooks below time each response on it,
   and the driver's own waits and time reads (a pause between requests, a grace period) go
   through it. Transport timeouts, such as an HTTP client's or a WebSocket receive's, stay in
   real time. Stateless APIs make `connect` and `sync_history`
   no-ops and send the canonical request (`GenerationRequest.canonical`,
   `realtime_framework/entity/request.py`) in their own format; session APIs send each
   observation and keep native history. A session API (`persistent_session`) also manages its
   session: the runtime calls its `ensure_ready` before each generation, `synchronize` after the
   retained history changed, `renew`, `repair` after each generation, `abort` and `mark_dirty`,
   and reads its `state`. Each of them but `abort` and `mark_dirty` takes the runtime's
   `record(sync, result=None, error=None)`: called once for each sync the operation runs, with
   the sync request and its result or error, and with `sync=None` once it reopened the session,
   it writes the history change to the trace. A session API inherits `SessionLifecycle`
   (`realtime_framework/providers/session_lifecycle.py`), which implements them, with
   `connect`, `sync_history` and `close`, over your `_connect`, `_sync_history`,
   `_close_current` and `interrupt`; it reads `self.clock`, writes its records to the trace
   `bind_trace` gave you, kept as `self._trace`, and `_wait` records a wait and takes it. Record
   what became of each content you write (sent, confirmed or refused by the server, or unknown
   after a failed write) with `_record_delivery`. Build HTTP clients with the hooks in
   `realtime_framework/providers/capture.py`, so every request and response is recorded.
2. Declare `capabilities` (`RealtimeCapabilities`): session persistence, interruption, usage
   timing, `action_channel` (`"text_json"` or `"function_call"`), the clocks the API supports,
   which optional methods it implements, and what it can render of the context policy: `roles`,
   `feedback_placements`, `reasoning_replay`, `server_context` and, for function-call sessions,
   `tool_call_replay`. `server_context` is `"unknown"` when the provider keeps items of the
   session the client never sees (reasoning, thoughts, audio), `"none"` when it keeps nothing
   beside what the client sent; the run identity records it, and no run is refused for it. The
   pre-run checks (`providers/preflight.py`) read the rest; an undeclared value counts as
   unsupported, and a context policy the driver cannot render is refused.
   Declare `deployment` (`Deployment`: catalog family and provider, default key variable, and
   whether each model names the upstream provider its requests pin or its responses are checked
   against, `requires_upstream_provider`; a driver family's subclass adds where its requests go,
   as `ChatDeployment`,
   `DecisionsDeployment` and `GeminiDeployment` do),
   `compat` (`Compat`: the API's restrictions the pre-run checks read: whether every request
   sends an output limit, the [provider options](#provider-options) a model may or must set with
   their allowed values, and the reasoning efforts the driver maps; `Compat()` for none) and
   `model_defaults` (per-model ModelConfig defaults, usually `{}`). A driver that lists
   `"streaming"` in `interaction_modes` also declares `streaming_settings`, the defaults of its
   [streaming settings](#streaming-settings) (a dict), takes `streaming`: the
   native-streaming agent builds it as `Driver(config, clock=clock, streaming=True)`, and has
   `push_frame`, `request_reply` and `classify_transport_error`, which the agent pushes frames,
   asks for replies and classifies their failures with. A
   driver whose `action_channel` is `"choice"` has `decide` and `classify_question_error`, which
   the selection flow (`realtime_framework/providers/decisions/`) asks each question and
   classifies its failures with. An external driver without any of these is refused when its
   model is loaded, and the tests hold every registered driver to the same checks.
   Function-call drivers send the function declarations the harness puts in
   `SessionSeed.answer`.
3. Emit `TextDelta`, `FunctionCall`, `UsageSnapshot`, and `GenerationBoundary` events. Report only
   provider-measured usage.
4. Implement `classify_transport_error(exc) -> TransportFailure` and declare
   `classifies_transport_errors=True`. Do not write a retry loop in the driver or turn on SDK
   retries: the shared executor (`RequestRetry`, `realtime_framework/entity/model.py`) retries
   transient failures of each turn itself, stops once the runner abandons the decision, and
   records every attempt.
5. Register the driver class as one `"package.module:Class"` entry in `APIS`
   (`realtime_framework/providers/registry.py`) and reference it from a preset with
   `endpoint.api`.
6. Add the API's own row to `current_context_policy()`
   (`realtime_framework/entity/context_policy.py`); a registered API without one is refused when
   its context policy is resolved.

## Use an external driver

A driver can also stay in your own package, outside this repository. Its results are not
official: `rtbench upload` refuses them. To put the model on the leaderboard, contribute the
driver as a [model API](#add-a-model-api).

1. Write the driver class as in steps 1-4 above, in your own package. Declare `capabilities`,
   `deployment`, `compat` and `model_defaults` as class attributes. Set `finite_frame_budget`,
   `wall_clock_only` and `accepts_reasoning_effort` explicitly: their defaults claim that the
   driver reports complete usage on a finite token clock, runs on any clock and accepts every
   reasoning effort. Declare `roles` (the request roles you map, e.g.
   `("instruction", "input", "model")`), `feedback_placements=("separate_message",)` and
   `server_context="none"` for a stateless API. Each optional-method flag you set
   (`classifies_transport_errors`, `connection_renewal`, `request_size_estimation`,
   `maps_output_cap`) needs its method, and `persistent_session` needs `ensure_ready`,
   `synchronize`, `renew`, `repair`, `abort`, `mark_dirty` and `state`. The driver runs under the
   `chat_completions` context policy, with the model's [context preset](#context-preset) if it
   names one. Its prompts ask for one JSON action in the visible text, so a driver whose
   `action_channel` is not `"text_json"` is refused before the run. With
   `request_size_estimation`, the runtime asks `request_context_status(request)`
   before each request and archives the oldest turns while the request exceeds the context
   window. With no turn left, the earliest archived actions give way to a note in the ledger,
   and stay out for the rest of the episode, as archived turns do; a request that still does
   not fit is not sent, and the run is invalid (`context_exceeded_before_request`).
   Without a declared window the method returns
   `{"pressure": False, "counter": "unconfigured"}`. The flag, not the method, decides this: a
   driver that implements `request_context_status` without declaring `request_size_estimation`
   is refused when its YAML is loaded, and so is a persistent session that declares it: the
   runtime archives the history a stateless request carries, not a session's.
2. Install the package into the environment that runs the benchmark, for example
   `uv pip install -e /path/to/your-driver`. `PYTHONPATH` is not enough: each Harbor trial runs
   with `PYTHONPATH` set to this repository.
3. Write a preset in `configs/models/<name>.yaml` whose `endpoint.api` is the class:

   ```yaml
   endpoint:
     api: your_package.chat:YourDriver
     model: YOUR_MODEL
     base_url: https://YOUR_HOST/v1   # optional; passed to the driver as api_base_url
     api_key_env: YOUR_API_KEY        # optional; default: deployment.key_env
     native_output_limit: 32768       # optional
   reasoning_effort: low
   ```

   `stream` and `include_usage` belong to registered APIs, and `upstream_provider` to a driver
   whose `deployment` sets `requires_upstream_provider`, which every one of its presets must then
   name, unless its `model_defaults` does; other per-model settings go in the class's
   `model_defaults`, or in the preset's `provider_options` where `compat` declares them. A framework class is selected by its registered name, not by its path.

Run it with `-m <name>`. Every lookup of the catalog imports the class and checks these
declarations, so a broken driver stops every run until its preset is fixed or removed. The run
configuration records the module, the class, a digest of every file in the package and the
context policy row (`chat_completions`), so results from different driver code are never mixed.

`tests/replay/external_template/rtb_ext/chat.py` is a complete example of a stateless chat
driver.

If your model cannot be reached through any of these, [contact us](submit.md#contact).
