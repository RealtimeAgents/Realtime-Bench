# Native-streaming smoke tests

A person runs these against the real OpenAI Realtime and Gemini Live APIs before native streaming
gets its defaults. They cost money and are never run by the test suite. Each check below says why
it is run, the command, which record to read, and what passes. What the checks settle:

- the newest frames OpenAI Realtime keeps (`retained_frames`, now 20) and Gemini Live's
  compression thresholds (now 65536 and 32768 tokens);
- the stuck thresholds of a streaming episode (a reply 30 s, a frame push 10 s, 3 failed pushes in
  a row);
- how Gemini Live starts a reply while frames keep coming (frames as realtime_input video, a reply
  started by one client_content turn carrying the newest frame's text and screenshot);
- whether models read the frame number drawn on each screenshot;
- the highest frame rate OpenAI Realtime takes, which its driver does not declare yet;
- that a streaming capture probing the page on a CDP connection of its own works in a real Chrome.

Write down the numbers each check asks for ([what to report](#what-to-report)).

## Before you start

- Both live drivers take `interaction_mode=streaming`, and the runner then runs the
  native-streaming agent (`realtime_framework/harnesses/streaming/agent.py`) on the streaming
  shell. A streaming run on the token clock, of any other API, or faster than its API streams is
  refused before its episode starts: a Harbor trial when it is resolved, before any sandbox; the
  runner started directly once it has built the images and started the sandbox, so such a run
  still takes the setup time. OpenAI Realtime streams at most 1 fps until check 8 finds its rate.
- A Linux host with Apptainer (fakeroot) and uv, as for any run ([README](../README.md)). Run
  `uv sync`, then `uv run rtbench setup` once to build the images.
- Put `OPENAI_API_KEY` and `GEMINI_API_KEY` in `.env` at the repository root. A Harbor trial reads
  it; for the runner started directly, also export it: `set -a; . ./.env; set +a`.
- The live presets in `configs/models/`:

  | Preset | API | Model |
  | --- | --- | --- |
  | `gpt-realtime-2` | `openai_realtime` | `gpt-realtime-2` |
  | `gemini-live` | `gemini_live` | `gemini-3.1-flash-live-preview` |
  | `gemini-3.8-live` | `gemini_live` | `gemini-3.8-live` |
  | `gemini-3.8-live-extended-thinking` | `gemini_live` | `gemini-3.8-live-extended-thinking` |

- **Cost.** A streaming session generates the whole episode, and `run_summary.json` does not
  price it: `cost_usd` is empty and `cost_accounting.reason` is `not_accounted_streaming`.
  `adapter_token_usage` counts the replies the episode accepted; a reply given up
  (`streaming.replies.abandoned`) is billed too, so it is a lower bound. Price a run yourself from
  the usage the provider reported for every reply ([reading message bodies](#reading-the-records))
  and the provider's price list: OpenAI Realtime's in each `response.done` (`response.usage`),
  cancelled replies included; Gemini Live's in the `usageMetadata` of its server messages
  (whether a message counts its turn or the session so far is not checked). Run one 1-minute
  episode per preset first and scale from it before running 3 or 5 minutes.

## Run one episode

Through Harbor, as a leaderboard run goes (results under `temp/harbor-jobs`):

```bash
uv run rtbench run --model gpt-realtime-2 --include-task-name gh_javascript_pong \
  --agent-kwarg interaction_mode=streaming --agent-kwarg duration_minutes=3
```

The episode's own directory, which the checks read, is
`temp/<UTC time>/<preset>/<task>/`. The job's copy under
`temp/harbor-jobs/<job>/<trial>/agent/benchmark/` lacks `episode_blobs.pack`, which holds the
message bodies.

Or the trial worker Harbor starts for each trial, alone (the same episode, without Harbor's
verifier); it checks the run, writes its launch file and runs the runner on it:

```bash
mkdir -p results/smoke
uv run python -m realtime_framework.evaluation._trial --catalog-model gpt-realtime-2 \
  --task gh_javascript_pong --interaction-mode streaming --duration-minutes 3 \
  2>&1 | tee "results/smoke/oai-pong-$(date -u +%Y%m%dT%H%M%SZ).out"
```

It runs on the wall clock at 1 fps unless told otherwise. The episode's directory is
`temp/<UTC time>/<preset>/<task>/`, as above, and each run gets a directory of its own. The runner
prints to its terminal only, so keep its output with `tee`.

**Other values of K or the thresholds.** Save the preset under a new name with a `streaming`
block ([streaming settings](model_configuration.md#streaming-settings)) and run that name, for
example `configs/models/gpt-realtime-2-k10.yaml`:

```yaml
endpoint:
  api: openai_realtime
  model: gpt-realtime-2
  api_key_env: OPENAI_API_KEY
reasoning_effort: low
streaming:
  retained_frames: 10
```

The model YAMLs are outside the harness hash, so this does not mark the harness `+dev`; the run's
configuration records the values (`run_config.json` → `streaming`), and its own preset name keeps
it apart. The stuck thresholds are code (`StreamingConfig` in
`realtime_framework/evaluation/kernel/streaming.py`); a run records the values it ran with in its
trace (`StreamingPolicyStarted`, below).

## Reading the records

In the episode's directory:

- `run_config.json`: `streaming` holds the streaming settings the run used.
- `run_summary.json`: `streaming` sums the episode up. `context_control` is `exact` for OpenAI
  Realtime and `approximate` for Gemini Live. `frames` has `captured`, `pushed`, `push_failed`,
  `dropped` with `dropped_by_reason` (`late`, `capture_busy`, `superseded`, `push_failed`,
  `ended`), `confirmed` and `refused` by the provider, and the distributions `capture_to_push_s`
  (from a frame's capture to when its push was written) and `send_to_confirm_s` (from then to
  the server's confirmation, OpenAI Realtime's only). `actions` has `arrived`, `outcomes` (`applied`, `wait`, `superseded`,
  `cancelled_generation`, `cancelled_call`, `invalid_grounding_frame`, `rejected`,
  `not_applied_time_limit`, ...), and the distributions `frame_age_s`, `frame_age_frames`,
  `arrival_to_start_s` and `execution_s`. `replies` has `started` and `abandoned` by reason.
- `episode_trace.jsonl`, one JSON object per line:
  - `event: "KernelRecord"` with `payload: {"kind", "data"}`: the streaming kernel's records.
    `FrameCaptured` (`observation_id`, `timestamp`), `FramePushed` (`observation_id`, `sent_at`,
    `error`), `FrameDropped` (`reason`), `InferenceStarted` and `GenerationFinished`
    (`generation`), `GenerationAbandoned` (`generation`, `reason`), `ActionArrived` (`call_id`,
    `frame`, `frame_age_s`, `frame_age_frames`, `arrived_at`, `rejection`, `detail`),
    `ActionResolved` (the same, with `status`, `started_at`, `finished_at`),
    `CallCancelled`, `StreamingPolicyStarted` (the thresholds). Times are seconds from the
    episode's start; each line also has `semantic` and `ops`. A run made invalid has
    `event: "RunInvalidated"`.
  - `event: "SessionMessage"`: each WebSocket message, with `channel` (`openai_realtime.ws` or
    `gemini_live.ws`), `connection_id`, `direction` (`client`, `server`, `lifecycle`),
    `message_type`, `captured_wall_time_ns`, and `body`, which points into
    `episode_blobs.pack`.
  - `event: "SessionDelivery"`: what became of what the session wrote: `state` (`sent`,
    `confirmed`, `refused`, `unknown`) and `contents`, whose `block` is `f<n>.frame` for frame `n`.
  - `event: "FrameSent"`: the number `n` (`frame`) a frame went into the session under, by its
    `observation_id`, with when it was captured (`captured_unix_s`), written (`sent_unix_s`) and
    confirmed by the server (`acked_unix_s`, null when no confirmation came, as always on Gemini
    Live, which confirms no input), in Unix seconds. It ties a frame's `FrameCaptured` and
    `FramePushed` to its block's `SessionDelivery`.
  - `event: "SessionWait"`: a wait the driver took, with `wait` and `seconds`.

Counting records:

```bash
jq -r 'select(.event=="KernelRecord") | .payload.kind' episode_trace.jsonl | sort | uniq -c
jq -c '.streaming' run_summary.json
```

Reading message bodies: save this as `session_bodies.py` and run
`uv run python session_bodies.py <episode dir> [message_type ...]`. Each line is
`[seconds since the episode started, connection, direction, message type, body]`, with long
strings (screenshots, audio) replaced by their length.

```python
import json
import sys
from pathlib import Path

from realtime_framework.evaluation.agent_trace import load_body


def shrink(value):
    if isinstance(value, dict):
        return {k: shrink(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shrink(v) for v in value]
    return f"<{len(value)} chars>" if isinstance(value, str) and len(value) > 300 else value


run, wanted = Path(sys.argv[1]), set(sys.argv[2:])
events = [json.loads(line) for line in (run / "episode_trace.jsonl").open()]
start = [e for e in events if e["event"] == "ActiveLoopStarted"][-1]["payload"]["start_unix_s"]
for e in events:
    p = e["payload"]
    if e["event"] != "SessionMessage" or p["direction"] == "lifecycle":
        continue
    if wanted and p["message_type"] not in wanted:
        continue
    body = json.loads(load_body(run / "episode_blobs.pack", p["body"]))
    at = round(p["captured_wall_time_ns"] / 1e9 - start, 3)
    print(json.dumps([at, p["connection_id"], p["direction"], p["message_type"], shrink(body)]))
```

Keeping a run as a test cassette:
`uv run python -m tests.streaming.ws_cassette <episode dir> <cassette.jsonl>`. It needs
`episode_trace.jsonl` and `episode_blobs.pack`; a lifecycle record in the session (a failed
handshake or send) makes it stop with an error.

## Checks

### 1. OpenAI Realtime: tokens per frame, the context limit and K

phase2.md, 落地前要验证 1. Run `gpt-realtime-2` for 3 minutes, so that more than 20 frames go in.

- Per reply: `session_bodies.py <dir> response.done`, then for each line
  `response.usage.input_tokens`, `response.usage.input_token_details.image_tokens` and
  `.cached_tokens`, and `response.status`.
- The usage details are always there (decision N42):
  `uv run python session_bodies.py <dir> response.done | jq -r '.[4].response.usage.input_token_details | type' | sort | uniq -c`
  prints only `object`. A `null` one is left unhandled: the first accepted reply that has it ends
  the run invalid (`invalid_run` `runner` / `run_exception`), and the runner prints
  `[streaming] shell raised:` with a traceback ending in `AttributeError: 'NoneType' object has
  no attribute 'get'` (read from the code, not run).
- Folds: older frames are replaced in place, a `conversation.item.delete` and a
  `conversation.item.create` each:
  `jq -r 'select(.event=="SessionMessage" and .payload.message_type=="conversation.item.delete") | .payload.capture_seq' episode_trace.jsonl | wc -l`.

Pass: `image_tokens` grows with the frames pushed for the first 20 (the slope is the tokens per
frame) and then stays level while the deletes grow with the frames pushed; `input_tokens` peaks
well under the model's context limit, and no `error` message appears. Report the tokens per frame,
the peak `input_tokens` and the limit (LiteLLM's table says 128000; the provider's figure is not
checked), and whether any reply's `input_token_details` was null. Try `retained_frames` 10 and 40
the same way.

### 2. Gemini Live: compression past 2 minutes, and the task still known

phase2.md, 落地前要验证 2, and the thresholds. Run `gemini-3.8-live` (and `gemini-live`) for 5
minutes.

- The setup: `session_bodies.py <dir> setup`; `setup.contextWindowCompression` is
  `{"trigger_tokens": 65536, "sliding_window": {"target_tokens": 32768}}` (or your preset's) and
  `setup.realtimeInputConfig` is `{"automatic_activity_detection": {"disabled": true},
  "activity_handling": "NO_INTERRUPTION", "turn_coverage": "TURN_INCLUDES_ALL_INPUT"}`.
- The prompt size over time: the `usageMetadata.promptTokenCount` of each server message that has
  one. A drop from above the trigger to near the target is one compression. The trigger is when the
  server compresses, not a cap on a reply's prompt. Each reply's trigger adds its screenshot as an
  image (about 1100 tokens in the 10-29 probe, against about 66 for a video frame): report the
  `IMAGE` tokens of `promptTokensDetails` per reply as well, and whether a reply's count is the
  whole session or that turn's alone.
- The session lasted: `run_summary.json` `invalid_run` is null and the last `gemini_live.ws`
  message is past 120 s.
- The task: whether the actions after the first compression still serve the task (the
  `ActionResolved` records and `solved`). The task text is in the system instruction, which
  google-genai's documentation of the sliding window says compression leaves alone; only you can
  judge this one.

Pass: at least one compression happened, the session ran to the end, and the late actions still
make sense for the task. Report the prompt tokens per frame (the slope before the first
compression), the time between compressions and how close the peak came to the model's 131072-token
window.

### 3. Frame numbers read back

phase2.md, 落地前要验证 3; stream-frames, stream-agent. Each screenshot carries `FRAME n` in its
top left corner, and each action must name the frame it is based on: a JSON field for OpenAI
Realtime, a function argument for Gemini Live. Run each live preset once.

- `run_summary.json` `streaming.actions.outcomes.invalid_grounding_frame` against
  `streaming.actions.arrived`.
- Why each was refused:
  `jq -r 'select(.event=="KernelRecord" and .payload.kind=="ActionArrived" and .payload.data.rejection!=null) | .payload.data.detail' episode_trace.jsonl | sort | uniq -c`.
  `missing_frame`: no frame named. `invalid_frame_syntax`: not an integer, such as Gemini's
  `12.0` or OpenAI's `"12"`, which the frame table refuses as written. `frame_not_in_table`: a
  frame never pushed.
- `streaming.actions.frame_age_frames`: a model that names the frame it looks at keeps this small.
  A wrong but existing number shows up only here; compare a few `SessionDelivery` sends of
  `f<n>.frame` with the screenshot by hand if in doubt.
- Look at a few of the screenshots the model got: the label is legible and covers no SoM badge
  near the corner. `frames/` holds the raw captures, which have no label; the labelled images are
  in the message bodies. Save this as `pushed_frames.py` and run
  `uv run python pushed_frames.py <episode dir>`; it writes each pushed image to
  `<episode dir>/pushed/<capture_seq>.<png|jpeg>`.

  ```python
  import base64
  import json
  import sys
  from pathlib import Path

  from realtime_framework.evaluation.agent_trace import load_body

  run = Path(sys.argv[1])
  out = run / "pushed"
  out.mkdir(exist_ok=True)
  for line in (run / "episode_trace.jsonl").open():
      e = json.loads(line)
      p = e["payload"]
      if e["event"] != "SessionMessage" or p["direction"] != "client":
          continue
      body = json.loads(load_body(run / "episode_blobs.pack", p["body"]))
      if p["message_type"] == "conversation.item.create":
          urls = [c["image_url"] for c in body["item"]["content"] if c["type"] == "input_image"]
          images = [(url[5:].split(";")[0], url.split(",", 1)[1]) for url in urls]
      elif p["message_type"] == "realtime_input" and "video" in body["realtime_input"]:
          video = body["realtime_input"]["video"]
          images = [(video["mime_type"], video["data"])]
      else:
          continue
      for mime, data in images:
          # OpenAI sends standard base64, google-genai URL-safe base64.
          image = base64.urlsafe_b64decode(data.replace("+", "-").replace("/", "_"))
          path = out / f"{p['capture_seq']:06d}.{mime.split('/')[1]}"
          path.write_bytes(image)
          print(path)
  ```

- All three Gemini presets accept the integer `frame` argument (`minimum: 1`): no setup error.

Pass: refusals stay rare (you decide the bar; under 5% of the actions that arrived is a
reasonable one), and no detail dominates. If `invalid_frame_syntax` is common, report which form
the model sends: the parser could normalize it as it does x and y. If the numbers are unreliable,
phase2.md's fallback is element ids that stay fixed across frames.

### 4. Gemini Live: starting a reply while frames keep coming

phase2.md, 落地前要验证 4; decision 22, whose trigger is now one client_content turn instead of
an activity (10-29); stream-gemini. Run each Gemini preset, `gemini-live`, `gemini-3.8-live` and
`gemini-3.8-live-extended-thinking`, for 3 minutes.

- Frames go in as `realtime_input` video, outside any activity, and keep going during replies:
  every client `realtime_input` message carries `video`, about once a second throughout,
  `uv run python session_bodies.py <dir> realtime_input | jq -r 'select(.[4].realtime_input.video) | .[0]'`
  printing their times. Report the largest gap between two during a reply, and
  `streaming.frames.dropped_by_reason`.
- Each reply starts with one `client_content` whose `turnComplete` is true: the feedback since the
  last trigger, each its own user turn, then the newest frame pushed, its text naming the number
  on its screenshot (`... labelled FRAME 12. ...`, with its element list) and its screenshot as
  `inlineData`, the same image as that frame's video. A trigger with no frame pushed since the
  last carries the same frame again. After a rebuild the seed's own `client_content` messages
  come first, starting no reply, and the trigger after them.
- Each trigger is followed by its reply's whole tool round, well within the reply watchdog: the
  `toolCall`, its `tool_response`, and the turn's end with its `usageMetadata` (`turnComplete`;
  on `gemini-3.8-live-extended-thinking` the `IDLE` interaction status after any intermediate
  `turnComplete`). Then nothing comes until the next trigger: no `serverContent`, `toolCall` or
  `toolCallCancellation` between a turn's end and the next `client_content`, also when the
  `tool_response` went out after the turn's end (the driver only logs such a message and runs none
  of its calls). A client_content trigger interrupts whatever the model is still generating, so
  this must hold before the defaults are set; `gemini-live` declares non-blocking functions that
  Google documents no asynchronous calling for.
- No reply is interrupted:
  `uv run python session_bodies.py <dir> serverContent | jq -r 'select(.[4].serverContent.interrupted) | .[0]'`
  prints nothing.
- No transcription holds `¿Qué es el número de cuenta?` (RealtimeAgents/Realtime-Bench#4), what
  the 100 ms of silent audio the activity trigger sent before was heard as.
- The screenshot, its number and its element list agree: for a few triggers, the `FRAME n` drawn
  on the inline image, the n its text names and the marks of its list on that image. Count the
  actions that name a frame pushed between two triggers, which the model got as video alone, with
  how their marks resolved (`reference_resolution`), and their `frame_age_frames` against the
  actions naming a trigger's frame.
- No fixed wait after the video in streaming:
  `jq -c 'select(.event=="SessionWait" and .payload.wait=="video_ingestion")' episode_trace.jsonl`
  prints nothing.

Pass: every trigger got its whole tool round, nothing came between a turn's end and the next
trigger, no reply was interrupted, and frames kept their rate during replies. If anything arrives
between a turn's end and the next trigger, report it: a client_content trigger is then not safe as
it stands.

### 5. Gemini Live: resumption and cancelled calls

stream-gemini. From the 5-minute run of check 2.

- `session_bodies.py <dir> sessionResumptionUpdate`: how often a handle comes and whether
  `resumable` is true while frames keep coming. Streaming resumes with the newest resumable
  handle; a run of `resumable: false` means a resume loses what came since.
- `goAway` messages and the reconnects after them (a new `connection_id`).
- `toolCallCancellation`: whether it comes before or after its reply's `turnComplete`, and the
  records it leaves: an action still waiting to execute is resolved as cancelled (`ActionResolved`
  with `status` `cancelled_call`); one executing, executed or resolved otherwise stays as it was,
  with a `CallCancelled` record.

Report the handle interval, the share of resumable handles, and any `GoAwayDeadlineExceeded`
failure (`invalid_run`).

### 6. The stuck thresholds

stream-plan 4.7. From every run above.

- The values the run used:
  `jq -c 'select(.event=="KernelRecord" and .payload.kind=="StreamingPolicyStarted") | .payload.data | {reply_watchdog_s, push_watchdog_s, push_failure_limit}' episode_trace.jsonl`.
- Reply durations, from each `InferenceStarted` to its `GenerationFinished` (`ops` seconds):

  ```bash
  jq -s '[.[] | select(.event=="KernelRecord")] as $k
    | ($k | map(select(.payload.kind=="InferenceStarted"))
          | map({key: (.payload.data.generation|tostring), value: .ops}) | from_entries) as $s
    | [$k[] | select(.payload.kind=="GenerationFinished") | .ops - $s[.payload.data.generation|tostring]]
    | {replies: length, max_s: max}' episode_trace.jsonl
  ```

- Push durations: `run_summary.json` `streaming.frames.capture_to_push_s`, from a frame's capture
  to when its push was written, and `send_to_confirm_s`, from then to the server's confirmation.
  OpenAI Realtime's push returns once the server confirmed the frame, so its push watchdog times
  both together. Gemini Live confirms no input: its `send_to_confirm_s` is always empty, and its
  push ends once written.
- Whether any watchdog fired: `GenerationAbandoned` records with reason `reply_watchdog`,
  `push_watchdog` or `push_failed`, and `RunInvalidated` with `frame_push_stuck` or
  `frame_push_failed`.

Pass: no watchdog fired in a normal run. Report the largest reply and push durations; each
threshold should sit well above them.

### 7. The capture's own CDP connection (decision N33)

stream-wsrec. In streaming, a capture's DOM and AX probes use a CDP connection of their own to the
page, so that they do not wait behind an action's. No record names the connection a probe used,
so check it indirectly, on one streaming and one turn-based run of a web task. The page's debug
port is the task's `dbg_port`, offset by the lane: `pgrep -af remote-debugging-port` on the host
shows it in the page's Chrome command line as `--remote-debugging-port=<debug port>`.

1. During the run, `ss -tnp | grep ':<debug port>'` on the host shows two established
   connections to the page's debug port in streaming and one in turn-based (the sandbox runs on
   the host's network, `realtime_framework/sandbox/apptainer.py`). After the episode they are
   gone.
2. Actions' focus checks and barriers succeed: no `took longer than 8 s` or `disappeared; refusing
   to switch pages` in the runner's output (an action that fails so ends the run invalid,
   `[invalid-run] environment:action_execution_failed — ...`), and `invalid_run` is not
   `action_execution_failed`.
3. The captures' probes succeed. Their failures are written into each frame's inspection, never
   printed: in `tool_calls.jsonl` (one row per frame inspected; the last few may be
   `incomplete: true`, written as the episode closes, with the capture's part complete),
   `jq -r '.canonical_result | (.scene.reason // empty), (.warnings[]? | split(":")[0])' tool_calls.jsonl | sort | uniq -c`
   shows no `capture_probe_failed:...` reason and no `window_probe_failed` warning, and with text
   observations no more `ax_tree_unavailable` than in the turn-based run.
4. Both connections stay on the task's page. On a run kept for this check, list the page targets
   with `curl -s http://127.0.0.1:<debug port>/json` and close the task's with
   `curl -s http://127.0.0.1:<debug port>/json/close/<id>`. Neither connection moves to another
   page: the next capture's `scene.reason` is `capture_probe_failed:RuntimeError` with a
   `CDP target ... disappeared; refusing to switch pages` warning, and the next action that probes
   the page ends the run invalid with the same message, as in turn-based (read from the code, not
   run).

### 8. OpenAI Realtime's frame-rate cap

review-stream-identity, low 8. OpenAI Realtime declares `max_streaming_fps=1.0` in
`realtime_framework/providers/openai_realtime.py`, a placeholder that refuses a faster streaming
run (`openai_realtime streams at most 1 fps; requested 2`). For this check raise it in a local
copy of the code (the harness is then `+dev`, which a smoke test does not mind). Run
`gpt-realtime-2` at `--agent-kwarg fps=1 --agent-kwarg model_input_fps=1`, then 2, then higher, on
a game task. At each rate:

- `streaming.frames.send_to_confirm_s` (p95) against the frame period `1/fps`;
- `streaming.frames.push_failed` and `dropped_by_reason` (`late`, `superseded`, `capture_busy`);
- any `error` or `rate_limits.updated` message (`session_bodies.py <dir> error rate_limits.updated`);
- `run_summary.json` `avg_capture_latency_s`: a slow capture limits the rate as well.

Pass at a rate: confirmations stay under the frame period, no push fails, no error, and drops do
not grow with the rate. Report the highest rate that passes, for the driver to declare as its
`max_streaming_fps`.

### 9. OpenAI Realtime's wire, as the session core assumes it

stream-oai. From the run of check 1.

- A refused push: the server's `error.event_id` echoes the push's `event_id` (`rtb_push_...`), and
  the refused item is not in the session afterwards.
- `previous_item_id` in the server's acknowledgement of an item: an item the client created names
  the item before it; note what a reply's own items name when a frame went in during the reply.
  The session core takes only `conversation.item.created` or `.done` as the acknowledgement
  (`_CREATE_ACKS` in `openai_realtime_session.py`), and keeps the `previous_item_id` of whichever
  comes first; the trace also links `conversation.item.added` to the create (`_ITEM_ACKS` in
  `openai_realtime.py`), which confirms nothing. The repository's fake server sends `added` and
  `done`, and which of the three the real API sends for a client create is not checked: note it,
  and whether the `previous_item_id` on `.created` or `.done` names the item before.
- When `response.output_text.done` arrives within its reply: its arrival is the action's
  `arrived_at`.

### 10. Empty replies

stream-policy. Replies follow one another without waiting for a new frame, so a fast model asks
again on the same frame. Count the replies that came to no action, the `GenerationFinished`
records without an `ActionArrived` of their generation, against `streaming.replies.started`, and
price them with check 1's or check 2's tokens:

```bash
jq -s '[.[] | select(.event=="KernelRecord") | .payload] as $k
  | ([$k[] | select(.kind=="ActionArrived") | .data.generation] | unique) as $acted
  | [$k[] | select(.kind=="GenerationFinished") | .data.generation]
  | {finished: length, without_action: map(select(IN($acted[]) | not)) | length}' episode_trace.jsonl
```

## What to report

| Check | Numbers |
| --- | --- |
| 1 | tokens per frame, peak `input_tokens`, context limit, result for K 10/20/40, null usage details |
| 2 | prompt tokens per frame, time between compressions, peak against 131072, task still served |
| 3 | per preset: actions arrived, `invalid_grounding_frame` by detail, `frame_age_frames` p90 |
| 4 | per preset: triggers without a whole tool round, messages between rounds, largest video gap during a reply, actions naming a frame between triggers |
| 5 | handle interval, resumable share, goAway reconnects, cancellation timing |
| 6 | largest reply and push durations, watchdogs fired |
| 7 | connections seen, errors found |
| 8 | highest frame rate that passes |
| 9 | the three behaviours as seen |
| 10 | share of empty replies and their tokens |
