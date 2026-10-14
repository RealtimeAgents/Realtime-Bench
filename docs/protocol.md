# Evaluation protocol

The default evaluation uses **wall clock, 1 FPS, and five minutes per task** for both public
and private tasks. The environment keeps changing while the model reasons; it alone decides
success.

The harness sends one observation per model decision; every model runs in this single mode.

## Clocks

| Clock | Time advancement |
| --- | --- |
| Wall (default) | Real time: observation, network, inference, and action latency all count. |
| Pause | The action applies at the observed tick, then time advances one frame. |
| Token | Time advances by provider-reported output tokens ÷ `tokens_per_frame`. |

Pause and token clocks are comparison settings ([how to run](running.md#comparison-clocks)).
They need complete provider usage; missing counts invalidate a token-clock run. The
`litellm`/`chat_completions` APIs support the wall clock only.

## Timing

Setup happens before the scored horizon. The horizon starts at the task reset and ends at the
deadline; later actions are not applied. Initial setup, draining, video encoding, and scoring
are excluded. `active-loop.json` records the scored interval; total trial time is diagnostic
and must not be used to reject a score. All models use the provider's default temperature.

## Actions

Each decision returns one canonical action:

```json
{"action": "click", "target": "P1"}
```

Verbs: `click`, `right_click`, `double_click`, `triple_click`, `mouse_down`, `mouse_up`,
`move`, `drag`, `type`, `key`, `hold_key`, `scroll`, `wait`. Task instructions list the
parameters. `done` and `fail` are not actions.

Each screenshot comes with a read-only Set-of-Marks legend: `P1` is a point or control, and
`R1@500,500` is a point inside a region in local 0–1000 coordinates. Raw coordinates are
pixels of that screenshot. Marks always resolve against the source screenshot, not a later
frame. Durations are at most 10 seconds.

The screenshot shows the whole browser, but the model acts on the page only. Marks cover page
content, never the browser's tabs, address bar or menus. Pointer actions outside the page and
browser shortcuts (address bar, tabs, windows, history, find, zoom, developer tools) are rejected
with feedback and count as failed actions. Before desktop key input, the harness returns
keyboard focus to the page if Chrome has moved it to its toolbar.

## Memory

The default harness keeps history in blocks: at the 21st active screenshot the oldest 10 become
placeholders, and above 50 detailed turns the oldest 10 move into an executed-action ledger.
Session APIs keep this history in their native session.

## Output and failures

The parser accepts alternative formats only when they identify exactly one action; otherwise
the harness executes a recorded WAIT. Function-like text is parsed, never executed. Native
function-call APIs (Gemini Live) take actions only from function calls, never from audio.

## Transport retries

Every model API uses the same retry policy (`TRANSPORT_RETRY` in
`realtime_framework/entity/model.py`). Drivers only classify a failed request; the harness
decides whether and when to retry, and SDK-level retries are disabled so every attempt is
recorded in the trajectory.

| Failure | Examples | Handling |
| --- | --- | --- |
| Transient | connection loss, timeout, HTTP 408 or 5xx, overloaded | retried |
| Rate limited | HTTP 429 without a quota signal | retried |
| Quota exhausted | HTTP 402, 429 naming quota, credits or billing | error, no retry |
| Refused | the model declined to answer (Decisions: `refused to answer`) | error, no retry |
| Request failed | any other rejection or protocol error | error, no retry |

A request is retried at most 3 times, after about 1, 3 and 8 seconds (±25% jitter). A
`Retry-After` header lengthens the wait; one above 30 seconds ends retrying. Only a request
that has received no output, usage, or call is retried, from canonical history, so a retry can
never duplicate an action; a failure after partial output is an error. Choice models
(JEV, Decisions) apply the policy to each question.

Retries are part of the decision. On the wall clock the world keeps running while the harness
waits; on the token clock a request without output charges no tokens.

## Outcomes

Each episode ends with exactly one outcome, recorded as `outcome` in `run_summary.json`, the
episode report and the Harbor verifier's `evaluation.json`:

| Outcome | Condition | Scored |
| --- | --- | --- |
| `pass` | The task is solved. | 1 |
| `fail` | Unsolved, and the episode ended before the scored horizon for a task-specific reason (game lost, a task deadline or rule violation, the model gave up) or an exhausted deliberation budget. | 0 |
| `timeout` | Unsolved, and the episode ran the whole scored horizon (`duration_minutes`, 5 by default), even if the task also reported an end at that moment. | 0 |
| `error` | Provider, environment, harness, evaluator, or infrastructure failure. | 0 |

An `error` also records `error_category` and `error_reason`, for example
`provider_rate_limited`, `provider_unavailable`, `provider_quota_exhausted`,
`provider_refused`, `provider_request_failed`, `provider_stream_interrupted`, `generation_watchdog_timeout`, or
`setup_error`. An error is scored as a failure: the success rate is passes divided by all
tasks, and an evaluation is complete once every task has a trial. `timeout` always means the
model used its time; a Harbor agent timeout is an `error`.

## Reporting

Report the model and driver versions, reasoning effort, memory policy, and
serving conditions. Different clocks, horizons, or FPS are different settings. Usage and cost
come only from provider reports.
