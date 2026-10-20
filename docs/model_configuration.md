# Model configuration

How to configure a new model. Pick the first case that fits:

| Your model | What to do |
| --- | --- |
| Supported by [LiteLLM](https://docs.litellm.ai/docs/providers) | [Run it directly](#run-it-directly) with `-m provider/model-id` |
| Behind your own OpenAI-compatible endpoint | [Write a preset](#write-a-preset) with `api: chat_completions` |
| Needs fixed settings (routing, revision, effort, output limit) | [Write a preset](#write-a-preset) |
| Uses a new API protocol, such as a realtime session API | [Add a model API](#add-a-model-api) |

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

Without `native_output_limit`, the limit comes from LiteLLM's bundled model map, or the provider
default if the model is unknown there. The run records the resolved preset, so results with
different settings are never mixed.

## Add a model API

A model that needs a new wire protocol gets a driver:

1. Write a driver in `realtime_framework/providers/` implementing `RealtimeDriver`
   (`realtime_framework/entity/model.py`): `connect`, `send_observation`, `start_generation`,
   `events`, `interrupt`, `sync_history`, `close`. Stateless APIs make `connect` and
   `sync_history` no-ops and send `GenerationRequest.messages`; session APIs send each observation
   and keep native history.
2. Declare `capabilities` (`RealtimeCapabilities`): session persistence, interruption, usage
   timing, `action_channel` (`"text_json"` or `"function_call"`), the clocks the API supports and
   which optional methods it implements. Declare `deployment` (`Deployment`: catalog family and
   provider, default key variable) and `model_defaults` (per-model ModelConfig defaults, usually
   `{}`). Function-call drivers declare `entity.actions.gui_function_declarations()`.
3. Emit `TextDelta`, `FunctionCall`, `UsageSnapshot`, and `GenerationBoundary` events. Report only
   provider-measured usage.
4. Implement `classify_transport_error(exc) -> TransportFailure`. Do not retry in the driver or
   SDK; the harness retries transient failures itself and records every attempt.
5. Register the driver class as one `"package.module:Class"` entry in `APIS`
   (`realtime_framework/providers/registry.py`) and reference it from a preset with
   `endpoint.api`.

If your model cannot be reached through any of these, [contact us](submit.md#contact).
