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

### Output limit

`native_output_limit` caps the output tokens of every request, reasoning included; unless a token
clock budget is smaller, it is the output limit each request sends. Without it,
presets on `openrouter`, `moonshot` and `anthropic_messages`, and direct runs, take the model's
`max_output_tokens` from the model map bundled with the locked LiteLLM version, looked up as
`provider/model-id`. A `chat_completions` endpoint, or a model the map does not list that way, has
no limit and gets the provider default.

These limits follow LiteLLM, so upgrading it changes them. From LiteLLM 1.104.1 to 1.106.1, 151
models gained, lost or changed a limit and 54 limits went down, for example
`openrouter/qwen/qwen3-235b-a22b-2507` from 235929 to 16384 tokens. To keep a limit across
upgrades, declare it in a preset, as `configs/models/deepseek-v4.1-flash.yaml` does. The run
records the resolved preset, limit included, so results with different settings are never mixed.

### Context preset

`context_preset` (top level, optional) names a set of context-policy values the model takes in
place of the standard ones. A model can name only `kimi_k3` (`MODEL_PRESETS` in
`realtime_framework/entity/context_policy.py`), which sends K3's past reasoning back, as Kimi
requires for K3: `configs/models/kimi-k3.yaml` sets it, and other moonshot models leave it out and
send none. A preset the API cannot render is refused before the run, and the run records the
preset with its configuration.

## Add a model API

A model that needs a new wire protocol gets a driver:

1. Write a driver in `realtime_framework/providers/` implementing `RealtimeDriver`
   (`realtime_framework/entity/model.py`): `connect`, `send_observation`, `start_generation`,
   `events`, `interrupt`, `sync_history`, `close`, plus `bind_trace(trace)`, which receives the
   episode trace. Its constructor takes the model config and the keyword `clock`
   (`realtime_framework/clock.py`), kept as `self.clock`; every wait and time read goes through
   it. Stateless APIs make `connect` and `sync_history` no-ops and send the canonical
   request (`GenerationRequest.canonical`, `realtime_framework/entity/request.py`) in their own
   format; session APIs send each observation and keep native history. A session API
   (`persistent_session`) also implements `ensure_ready`, `abort` and `mark_dirty`, which ready,
   abort and mark its session, and records each of its waits before it starts. Inherit
   `SessionLifecycle` (`realtime_framework/providers/session_lifecycle.py`) for these: it
   implements `connect`, `sync_history`, `close` and those three over your `_connect`,
   `_sync_history`, `_close_current` and `interrupt`, and `_wait` records a wait and takes it.
   Build HTTP clients
   with the hooks in `realtime_framework/providers/capture.py`, so every request and response is
   recorded.
2. Declare `capabilities` (`RealtimeCapabilities`): session persistence, interruption, usage
   timing, `action_channel` (`"text_json"` or `"function_call"`), the clocks the API supports,
   which optional methods it implements, and what it can render of the context policy: `roles`,
   `feedback_placements`, `reasoning_replay`, `server_context` and, for function-call sessions,
   `tool_call_replay`. The pre-run checks (`providers/preflight.py`) read it; an undeclared value
   counts as unsupported, and a context policy the driver cannot render is refused.
   Declare `deployment` (`Deployment`: catalog family and provider, default key variable) and
   `model_defaults` (per-model ModelConfig defaults, usually `{}`). Function-call drivers send the
   function declarations the harness puts in `SessionSeed.answer`.
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
   `deployment` and `model_defaults` as class attributes. Set `finite_frame_budget`,
   `wall_clock_only` and `accepts_reasoning_effort` explicitly: their defaults claim that the
   driver reports complete usage on a finite token clock, runs on any clock and accepts every
   reasoning effort. Declare `roles` (the request roles you map, e.g.
   `("instruction", "input", "model")`), `feedback_placements=("separate_message",)` and
   `server_context="none"` for a stateless API. Each optional-method flag you set
   (`classifies_transport_errors`, `connection_renewal`, `request_size_estimation`,
   `maps_output_cap`) needs its method, and `persistent_session` needs `ensure_ready`, `abort`
   and `mark_dirty`. The driver runs under the `chat_completions` context
   policy, with the model's [context preset](#context-preset) if it names one. With
   `request_size_estimation`, the runtime asks `request_context_status(request)`
   before each request and archives the oldest turns while the request exceeds the context
   window; without a declared window the method returns
   `{"pressure": False, "counter": "unconfigured"}`.
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

   `stream`, `include_usage` and `upstream_provider` belong to registered APIs; per-model
   settings go in the class's `model_defaults`. A framework class is selected by its registered
   name, not by its path.

Run it with `-m <name>`. Every lookup of the catalog imports the class and checks these
declarations, so a broken driver stops every run until its preset is fixed or removed. The run
configuration records the module, the class, a digest of every file in the package and the
context policy row (`chat_completions`), so results from different driver code are never mixed.

`tests/replay/external_template/rtb_ext/chat.py` is a complete example of a stateless chat
driver.

If your model cannot be reached through any of these, [contact us](submit.md#contact).
