# Experimental models

Choice models answer typed questions over executable candidates instead of generating actions.
Their candidate strategies are experimental systems (`benchmark_eligible: false`): the presets
are excluded from maintained leaderboard sweeps, their scores are not pooled with the screenshot
leaderboard, and they require `ALLOW_EXPERIMENTAL_MODEL=1`. Both run on the wall clock only.

Both use the shared [retry policy](protocol.md#transport-retries) per question, and every
grouping, composition and retry call consumes wall time.

## GPT-6 Luna Decisions

`-m gpt-6-luna-decisions` uses
[`openai/gpt-6-luna-decisions`](https://openrouter.ai/openai/gpt-6-luna-decisions) through
OpenRouter's `POST /api/alpha/decisions` (set `OPENROUTER_API_KEY`). The preset pins the upstream
to OpenAI and disables provider fallbacks. This is the OpenRouter protocol, whose `state` and
named `questions` differ from the
[direct OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions).

```bash
ALLOW_EXPERIMENTAL_MODEL=1 \
  uv run rtbench run -m gpt-6-luna-decisions -i rt_accessible_trip_dispatch
```

- **Tasks:** Web and game tasks. On Web it reuses the [JEV](#jev) candidate strategy, including
  grouped comparisons and text composition. On games the harness offers observation-declared key
  taps, short holds and simultaneous key pairs, visible grid positions, and pointer actions;
  continuous pointer locations are chosen through bounded Choice questions over integer
  coordinates 0–1000 in a region or the screenshot. These are candidate policies, not
  task-specific game solvers.
- **History:** the same retention as every other stateless model. Each call adds the current
  observation's accessibility tree and interactables; the candidates are the Choice criteria.
  The `state` keeps the canonical message order, with each image as a top-level part at its
  position, as the
  [multimodal Decisions protocol](https://openrouter.ai/docs/guides/community/multimodal-decisions)
  requires, so older turns stay a stable prefix for prompt caching. Before each decision the
  runtime estimates the request against the 1.05M-token window (512 tokens per screenshot, a 32k
  reserve for one question's criteria) and archives history blocks under pressure.
- **Settings:** `reasoning_effort: fixed`; no reasoning, sampling or generation-limit parameters
  are sent. Provider-reported usage (including zero output tokens) is retained; serialized GUI
  actions are not counted as generated tokens.
- **Failures:** refusals, invalid choices, missing native usage and unexpected responses fail the
  decision. OpenAI sometimes refuses a question over long history, most often a many-option
  text-composition question; a refusal is not retried and ends the run as an `error` with
  `provider_refused`.

## JEV

[JEV](https://openrouter.ai/docs/guides/community/jev-tutorial) (`typesafe/jev-1.13`) answers
typed questions over a text `state` and never sees images or generates text, so it runs on Web
tasks only.

```bash
ALLOW_EXPERIMENTAL_MODEL=1 \
  uv run rtbench run -m jev-1.13 -i rt_accessible_trip_dispatch
```

- **Transport** (`providers/jev/decisions.py`): a stateless client that asks one Choice question,
  with failure classification, context-limit detection and response validation. It knows
  nothing about candidates, prompts or history.
- **Observation** (`harnesses/jev/candidates.py`): each frame becomes a text state with the
  accessibility tree, interactables, mark legend and focused field, plus the frame's candidate
  actions: `wait`, clicks, drag/drop, scrolls, native select options, focused-field keys, observed
  literals, and `compose_text`.
- **Decision** (`harnesses/jev/selection.py`): a Choice accepts at most 255 options, so larger
  candidate sets are compared in groups and the winners compared again. A context overflow splits
  the group; if a single option still overflows, the oldest history is dropped.
- **History** (`harnesses/jev/context.py`): TypeSafe caps prompts at 32k tokens, so JEV declares
  `history_replay="action_digest"`: earlier turns are sent as their actions and execution
  feedback, and only the current observation carries interactables. `compose_text` builds input
  through repeated character/word Choices (`composition.py`) and ends in one `type` action.
