# Models

There are two ways to run a model. Pick the first that fits:

| Your model | What to do |
| --- | --- |
| Reachable through LiteLLM or an OpenAI-compatible API | Nothing: pass `-m provider/model-id` ([running](running.md)). Optionally save a preset YAML. |
| A new API protocol (e.g. a new realtime/session API) | Write one provider driver and register it ([below](#adding-a-model-api)). |

## Presets

A preset is `configs/models/<name>.yaml`; select it with `-m <name>`. It has one `endpoint`
block and an optional `reasoning_effort`; without it the API's default applies (`low` for most
APIs, the provider default for `litellm` and `chat_completions`). Keys stay in `.env`.

```yaml
endpoint:
  api: openrouter
  model: anthropic/claude-opus-5
  api_key_env: OPENROUTER_API_KEY
  upstream_provider: anthropic
reasoning_effort: low
```

`endpoint.native_output_limit` (optional) declares the model's physical maximum output tokens,
including reasoning. It is not a benchmark budget: requests send the token-clock cap when one
applies, otherwise this limit. Without it, the limit comes from `max_output_tokens` in LiteLLM's
bundled model map (the pinned package's copy, never fetched at runtime), as Harbor's Terminus
agents do. If neither knows the model the parameter is omitted and the provider default applies.
`chat_completions` endpoints never use the map, because their host is user-defined.
`anthropic_messages` requires a limit because the API does. The run config records the value and
its source (`endpoint` or `litellm`).

`reasoning_effort` is likewise not checked against a per-model list: the YAML value is passed
through and the driver or provider rejects what the model does not accept. APIs whose driver sends
no reasoning setting (the Decisions APIs) accept only `fixed`; `fixed` on other APIs omits the
setting for models whose reasoning is not configurable.

`endpoint.api` selects a registered API (`APIS` in `realtime_framework/providers/registry.py`):

| API | Driver | Notes |
| --- | --- | --- |
| `litellm` | `providers/litellm.py` | Any `provider/model-id`; wall clock only |
| `chat_completions` | `providers/litellm.py` | Your OpenAI-compatible endpoint; needs `base_url`; wall clock only |
| `openrouter` | `providers/litellm.py` | Requires a pinned `upstream_provider` |
| `anthropic_messages` | `providers/litellm.py` | Anthropic Messages API |
| `moonshot` | `providers/litellm.py` | Kimi-compatible endpoints |
| `openai_realtime` | `providers/openai_realtime.py` | GPT Realtime session (`gpt-realtime-2`) |
| `gemini_live` | `providers/gemini/live.py` | Gemini Live session with native GUI function calls (`gemini-live`, `gemini-3.8-live*`) |
| `jev_decisions` | `providers/jev/decisions.py` | [JEV](experimental-models.md#jev) through OpenRouter's Decisions API; text-only choice model (`jev-1.13`); Web tasks, wall clock only |
| `openai_decisions` | `providers/openai_decisions.py` | [GPT-6 Luna Decisions](experimental-models.md#gpt-6-luna-decisions) through OpenRouter's Decisions API; screenshot + text choice model; pinned upstream, Web and game tasks, wall clock only |

Generative HTTP APIs share one LiteLLM driver configured by a per-API wire policy. Only
provider-reported usage is accepted; missing usage stays unknown. Per-model limits, reasoning
efforts, and driver options live in `MODEL_PROFILES`.

Generative APIs run in `DefaultHarness`; choice APIs reuse the candidate
harness in `harnesses/jev/`. Neither checks provider names: driver
capabilities decide whether history is resent or kept in a session, and whether the model
answers with a JSON action or a native GUI function call.

## Adding a model API

1. Write a driver in `realtime_framework/providers/` implementing `RealtimeDriver` (`realtime_framework/entity/model.py`):
   `connect`, `send_observation`, `start_generation`, `events`, `interrupt`, `sync_history`,
   `close`. Stateless APIs make `connect` and `sync_history` no-ops and send
   `GenerationRequest.messages`; session APIs send each observation and keep native history.
2. Declare `capabilities`: session persistence, interruption, usage timing, and
   `action_channel` (`"text_json"`, `"function_call"` or `"choice"`). Function-call drivers
   declare `entity.actions.gui_function_declarations()`. Drivers without image input set
   `observation_modality="text"`; the environment then captures the accessibility tree and
   rendered control state for them instead of relying on the screenshot.
3. Emit `TextDelta`, `FunctionCall`, `UsageSnapshot`, and `GenerationBoundary` events. Report
   only measured usage.
4. Implement `classify_transport_error(exc) -> TransportFailure`. Do not retry in the driver
   or SDK; the harness applies the shared [retry policy](protocol.md#transport-retries).
5. Add one `ApiSpec` row to `APIS` (plus a `MODEL_PROFILES` entry if needed) and reference it
   from a preset with `endpoint.api`.

Choice APIs implement `ChoiceClient` instead of `RealtimeDriver`. The choice harness owns
candidate enumeration, grouping and text composition; the provider only transports questions.
