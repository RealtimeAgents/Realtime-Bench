# Custom model drivers

A driver connects a model that speaks a new wire protocol. Most models do not need one: check
[connect a model](evaluation.md#connect-a-model) first.

Start from `examples/external_driver/rtb_ext/chat.py`, a complete stateless chat driver. Then choose
where the driver lives:

| Where | Results | Next |
| --- | --- | --- |
| In this repository, in `realtime_framework/providers/` | Official | [Write a driver](#write-a-driver), then [add a model API](#add-a-model-api) |
| In your own package | Not official: `rtbench upload` refuses them | [Write a driver](#write-a-driver), then [use an external driver](#use-an-external-driver) |

## Write a driver

### 1. Choose a base class

Both are in `realtime_framework/entity/model.py`, unless noted.

| API | Base class | What it does |
| --- | --- | --- |
| Stateless | `StatelessEncoder` | Sends the canonical request (`GenerationRequest.canonical`, `realtime_framework/entity/request.py`) in its own format with each generation. The runtime asks it nothing else: an abort interrupts the generation under way and closes the driver, which reopens what the next generation needs. |
| Session | `SessionEncoder` | Sends each observation, keeps native history and manages its session. Inherit `SessionLifecycle` (`realtime_framework/providers/session_lifecycle.py`), the `SessionEncoder` that implements the session methods. |

### 2. Implement the methods

Abstract in both base classes:

| Method | Use |
| --- | --- |
| `start_generation` | Start a generation |
| `events` | The generation's events |
| `interrupt` | Stop the generation |
| `close` | Close the driver |
| `bind_trace(trace)` | Receive the episode trace |

A session API also has these:

| Method | When |
| --- | --- |
| `connect` | Open the session |
| `send_observation(observation, history)` | Send each observation |
| `ensure_ready` | Before each generation |
| `synchronize` | After the retained history changed |
| `renew` | Before a generation; it renews the connection only when the provider asked for it |
| `repair` | After each generation |
| `abort`, `mark_dirty` | On an abort, and to mark the session dirty |
| `state` | Read by the runtime |

`ensure_ready`, `synchronize`, `renew` and `repair` take the runtime's
`record(sync, result=None, error=None)`. It is called once for each sync the operation
runs, with the sync request and its result or error, and with `sync=None` once the operation
reopened the session. It writes the history change to the trace.

`SessionLifecycle` implements them, with `connect` and `close`, over your `_connect`, `_sync_history`,
`_close_current` and `interrupt`.

- It reads `self.clock`.
- Write records to the trace `bind_trace` gave you, kept as `self._trace`.
- `_wait` records a wait and takes it.
- Record what became of each content you write (sent, confirmed or refused by the server, or
  unknown after a failed write) with `_record_delivery`.

Build HTTP clients with the hooks in `realtime_framework/providers/capture.py`, so every request and
response is recorded.

### 3. Use the clock

The harness builds the driver as `Driver(config, clock=clock)`, with the episode's operational clock
(`realtime_framework/clock.py`). Keep it as `self.clock`:

- The capture hooks time each response on it.
- The driver's own waits and time reads (a pause between requests, a grace period) go through it.
- Transport timeouts, such as an HTTP client's or a WebSocket receive's, stay in real time.

### 4. Declare what the driver supports

`capabilities` (`RealtimeCapabilities`) also names the clocks the API supports:

| Field | Meaning |
| --- | --- |
| `action_channel` | `"text_json"`, `"function_call"` or `"choice"` |
| `roles`, `feedback_placements`, `reasoning_replay` | What it can render of the context policy |
| `tool_call_replay` | The same, for function-call sessions |
| `server_context` | `"unknown"` when the provider keeps items of the session the client never sees (reasoning, thoughts, audio), `"none"` when it keeps nothing beside what the client sent. The run identity records it, and no run is refused for it. |

The pre-run checks (`providers/preflight.py`) read the rest. An undeclared value counts as
unsupported, and a context policy the driver cannot render is refused.

Other declarations:

| Declaration | Holds |
| --- | --- |
| `deployment` (`Deployment`) | Catalog family and provider, default key variable, and whether each model names the upstream provider its requests pin or its responses are checked against (`requires_upstream_provider`). A driver family's subclass adds where its requests go, as `ChatDeployment`, `DecisionsDeployment` and `GeminiDeployment` do. |
| `compat` (`Compat`) | The API's restrictions the pre-run checks read: whether every request sends an output limit, the [provider options](reference.md#provider-options) a model may or must set with their allowed values, and the reasoning efforts the driver maps. `Compat()` for none. |
| `model_defaults` | Per-model `ModelConfig` defaults, usually `{}` |

Depending on the driver:

| Driver | Must have |
| --- | --- |
| Lists `"streaming"` in `interaction_modes` | `streaming_settings`, the defaults of its [streaming settings](reference.md#streaming-settings) (a dict); a `streaming` constructor argument, as the native-streaming agent builds it with `Driver(config, clock=clock, streaming=True)`; `push_frame` and `request_reply`, which the agent pushes frames and asks for replies with; and `classify_transport_error` for their failures |
| `action_channel` is `"choice"` | `decide` and `classify_question_error`, which the selection flow (`realtime_framework/providers/decisions/`) asks each question and classifies its failures with |
| Uses function calls | Send the function declarations the harness puts in `SessionSeed.answer` |

An external driver without any of these is refused when its model is loaded, as is one that leaves
out an abstract method or whose constructor cannot take `clock`. The tests hold every registered
driver to the same checks.

### 5. Emit events and classify failures

- Emit `TextDelta`, `FunctionCall`, `UsageSnapshot` and `GenerationBoundary` events. Report only
  provider-measured usage.
- Override `classify_transport_error(exc) -> TransportFailure`. The default takes every failure as
  fatal.
- Do not write a retry loop in the driver or turn on SDK retries. The shared executor (`RequestRetry`,
  `realtime_framework/entity/model.py`) retries transient failures of each turn itself, stops once
  the runner abandons the decision, and records every attempt.

## Add a model API

Put the driver in `realtime_framework/providers/`, then register it:

1. Add the class as one `"package.module:Class"` entry in `APIS`
   (`realtime_framework/providers/registry.py`), and reference it from a preset with `endpoint.api`.
2. Add the API's own row to `ROW_PROMPTS` (`realtime_framework/entity/context_policy.py`). A
   registered API without one is refused when its context policy is resolved.

## Use an external driver

A driver can stay in your own package, outside this repository. To put the model on the leaderboard,
contribute the driver as a [model API](#add-a-model-api) instead.

### 1. Write the driver class

Follow [write a driver](#write-a-driver) in your own package, with these differences.

Declare `capabilities`, `deployment`, `compat` and `model_defaults` as class attributes. In
`capabilities`:

| Declare | Value |
| --- | --- |
| `finite_frame_budget`, `wall_clock_only`, `accepts_reasoning_effort` | Set each explicitly: their defaults claim that the driver reports complete usage on a finite token clock, runs on any clock and accepts every reasoning effort |
| `roles` | The request roles you map, e.g. `("instruction", "input", "model")` |
| `feedback_placements` | `("separate_message",)` |
| `server_context` | `"none"` for a stateless API |

The optional methods `classify_transport_error`, `transport_output_token_cap`,
`request_context_status` (stateless) and `connection_renewal_reason` (session) each have a default;
overriding one takes it up.

Refused when its YAML is loaded:

- A class that subclasses neither `StatelessEncoder` nor `SessionEncoder`, or both, or leaves out an
  abstract method of the one it subclasses. A session API subclasses `SessionLifecycle`, which is a
  `SessionEncoder`.
- A `SessionEncoder` that has `request_context_status`: the runtime archives the history a stateless
  request carries, not a session's.

Refused before the run: a driver whose `action_channel` is not `"text_json"`. The driver runs under
the `chat_completions` context policy, with the model's
[context settings](reference.md#context-settings) if it sets any, and its prompts ask for one JSON
action in the visible text.

Where the driver overrides `request_context_status(request)`, the runtime asks it before each
request and archives the oldest turns while the request exceeds the context window. With no turn
left, the earliest archived actions give way to a note in the ledger, and stay out for the rest of
the episode, as archived turns do. A request that still does not fit is not sent, and the run is
invalid (`context_exceeded_before_request`). Without a declared window the method returns
`{"pressure": False, "counter": "unconfigured"}`.

### 2. Install the package

Install it into the environment that runs the benchmark:

```bash
uv pip install -e /path/to/your-driver
```

`PYTHONPATH` is not enough: each Harbor trial runs with `PYTHONPATH` set to this repository.

### 3. Write a preset

In `configs/models/<name>.yaml`, set `endpoint.api` to the class:

```yaml
endpoint:
  api: your_package.chat:YourDriver
  model: YOUR_MODEL
  base_url: https://YOUR_HOST/v1   # optional; passed to the driver as api_base_url
  api_key_env: YOUR_API_KEY        # optional; default: deployment.key_env
  native_output_limit: 32768       # optional
reasoning_effort: low
```

- `stream` and `include_usage` belong to registered APIs.
- `upstream_provider` belongs to a driver whose `deployment` sets `requires_upstream_provider`.
  Every one of its presets must then name it, unless its `model_defaults` does.
- Other per-model settings go in the class's `model_defaults`, or in the preset's `provider_options`
  where `compat` declares them.
- A framework class is selected by its registered name, not by its path.

### 4. Run it

Run it with `-m <name>`.

- Every lookup of the catalog imports the class and checks these declarations, so a broken driver
  stops every run until its preset is fixed or removed.
- The run configuration records the module, the class, a digest of every file in the package and
  the context policy row (`chat_completions`), so results from different driver code are never mixed.

`examples/external_driver/rtb_ext/chat.py` is a complete example of a stateless chat driver, a
`StatelessEncoder`.

If your model cannot be reached through any of these, [contact us](submit.md#contact).
