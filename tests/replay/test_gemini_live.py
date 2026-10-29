"""Replay baselines for the gemini_live API on the finite token clock.

Record a baseline with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it the
run is compared with tests/replay/baselines/.

The harness builds a canonical request for this API on every turn, but the driver sends only part
of it (harnesses/default/harness.py DefaultHarness._generate; providers/gemini/live.py
GeminiLiveDriver.send_observation): the model sees the session's setup, the history replayed on each
new connection and one observation per decision. The baseline is therefore what reaches the
WebSocket, the client frames of every connection in arrival order, plus each decision's parsed
action, usage and commit.

Scripts:
- the long scenario for two maintained models (their YAMLs' provider_options, configs/models),
  which differ in the completion signal and the tool acknowledgement;
- a model that sets the driver's default options (live.py LIVE_OPTIONS) and the standard
  tool_call_history: realtime_input observations and native_tools history;
- a resume that the server rejects, which falls back to replaying the history
  (live.py GeminiLiveDriver._sync_history);
- 3.1 Flash Live (configs/models/gemini-live.yaml) on the driver-default script: non-blocking
  calls, acknowledged silently, and a goAway that replays the history.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import realtime_framework.entity.model as model_module
import realtime_framework.providers.registry as catalog
from realtime_framework.clock import SystemClock
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.model import (
    ARCHIVED_ACTION_LEDGER_HEADER,
    QUOTA_EXHAUSTED,
    TransportFailureError,
)
from realtime_framework.evaluation.agent_trace import close_agent_trace
from realtime_framework.providers.gemini.live import LIVE_OPTIONS

from . import baseline
from .build import Episode, build_episode
from .frames import ReplayTask
from .scenario import CLICK_AGREE, CLICK_NAME, LONG_SCENARIO, TASK_INSTRUCTION
from .steps import Step, commit, decide
from .ws_gemini_live import (
    AWAIT_ACK,
    TRANSCRIPT_HEADER,
    LiveStub,
    WireNormalizer,
    interrupted_before_output,
    realtime_input_order,
)

BASELINES = Path(__file__).parent / "baselines"
API_KEY = "replay-dummy-key"

# A token-clock run at 1 fps (Gemini Live accepts at most 1: live.py
# GeminiLiveDriver.capabilities, providers/preflight.py check_fps) with B = 256 tokens per frame
# and a 5-minute episode, 5 * 60 * fps ticks (runner.py main, kernel/turns.py
# TurnConfig.max_ticks). The catalog's native limit for these models is 65536, as their endpoints
# declare (configs/models/), so the runtime cap min(65536, (N - t_k) * B)
# (kernel/turns.py _captured, episode_records.py decision_budget) falls below it from t_k = 45 on
# and is 61696 at the last decision.
FPS = 1.0
TOKENS_PER_FRAME = 256
MAX_TICKS = 300
FIRST_CAPPED_STEP = 45

# Decision k (0-based) holds k earlier turns plus its own frame. A 21st active screenshot folds the
# oldest 10 (entity/context_policy.py effective_visual_fold_policy; harnesses/default/history.py
# apply_active_visual_budget) and a 51st turn archives the oldest 10 (context_policy.py
# effective_detailed_history_policy; history.py History._retain). Either change rebuilds the
# session, because a persistent Gemini session cannot be edited in place (context.py
# RealtimeRuntime._run_turn; live.py GeminiLiveDriver.capabilities). Counted from 1, these are
# decisions 21, 31, 41 and 51.
REBUILD_STEPS = (20, 30, 40, 50)
FOLDED = 10
ARCHIVE_STEP, ARCHIVED = 50, 10

# Replies in LONG_SCENARIO written as a call, and the declared function each one names
# (entity/actions.py gui_function_declarations).
CALL_SYNTAX = {
    'click(target="P3")': ("click", {"target": "P3"}),
    'press_key("tab")': ("key", {"key": "tab"}),
}
AUDIO_MIME = "audio/pcm;rate=24000"


def native_reply(reply: str) -> tuple[list[tuple[str, dict]], str]:
    """A scenario reply as Gemini gives it on the function_call channel, where only native calls
    can carry an action (harness.py DefaultHarness._generate; parsing.py
    FunctionCallTurnCollector.finish): the calls it makes, or what it says when the reply holds no
    action."""
    text = reply
    if text.startswith("```"):  # a fenced action
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if text in CALL_SYNTAX:
        return [CALL_SYNTAX[text]], ""
    if not text.startswith("{"):
        return [], text  # an empty reply, or speech
    decoder = json.JSONDecoder()
    calls, index = [], 0
    while index < len(text):  # one or more action objects
        action, index = decoder.raw_decode(text, index)
        calls.append((action.pop("action"), action))
        while index < len(text) and text[index].isspace():
            index += 1
    return calls, ""


def usage_metadata(k: int, step: Step, speech: str) -> dict:
    """Usage in the scale of scenario.replies. Speech is the forced AUDIO output, which the token
    clock does not charge (live.py GeminiLiveDriver._usage_dict). Zero counts are omitted, as in the
    protobuf JSON."""
    prompt = 1500 + 260 * k
    response = 4 + len(step.reply) // 4
    thoughts = len(step.reasoning) // 4
    counts = {
        "promptTokenCount": prompt,
        "cachedContentTokenCount": 0 if k == 0 else 1280,
        "responseTokenCount": response,
        "thoughtsTokenCount": thoughts,
        "totalTokenCount": prompt + response + thoughts,
    }
    usage = {name: value for name, value in counts.items() if value}
    usage["responseTokensDetails"] = [
        {"modality": "AUDIO" if speech else "TEXT", "tokenCount": response}
    ]
    return usage


def turn_complete(k: int, usage: dict) -> list[dict]:
    """completion_signal=turn_complete ends the turn at turnComplete (gemini/session.py
    SessionCore.received)."""
    return [
        {"serverContent": {"generationComplete": True}},
        {"serverContent": {"turnComplete": True}, "usageMetadata": usage},
    ]


def interaction_idle(k: int, usage: dict) -> list[dict]:
    """completion_signal=interaction_idle spans intermediate turnComplete messages until
    interactionStatus is IDLE (live.py _Receiver). The SDK ends its receive iterator at a bare
    turnComplete, so the driver reopens it, and continues it past an IN_PROGRESS status
    (google/genai/live.py:87-99, 471-475); alternate decisions take each shape."""
    intermediate = {"turnComplete": True}
    if k % 2:
        intermediate["interactionStatus"] = "IN_PROGRESS"
    return [
        {"serverContent": intermediate},
        {"serverContent": {"interactionStatus": "IDLE"}, "usageMetadata": usage},
    ]


@dataclass(frozen=True)
class Preset:
    name: str
    model: str  # catalog key, or a model id no YAML names
    history_format: str
    # what each queued acknowledgement adds (gemini/session.py SessionCore.tool_response_options)
    ack_options: dict
    completion: Callable[[int, dict], list[dict]]


# Two maintained models (configs/models) and the driver defaults (live.py LIVE_OPTIONS) with the
# standard tool_call_history.
TOOL_TRANSCRIPT = Preset(
    "tool-transcript",
    "gemini-3.8-live",
    "tool_transcript",
    {"will_continue": False, "scheduling": "SILENT"},
    turn_complete,
)
EXTENDED_THINKING = Preset(
    "extended-thinking",
    "gemini-3.8-live-extended-thinking",
    "tool_transcript",
    {},
    interaction_idle,
)
DRIVER_DEFAULT = Preset("driver-default", "replay-gemini-live", "native_tools", {}, turn_complete)
# A model sets each wire option (providers/preflight.py check_provider_options); these are the
# driver's defaults.
DRIVER_DEFAULT_OPTIONS = {name: default for name, (default, _) in LIVE_OPTIONS.items()}
# 3.1 Flash Live takes 3.8 Live's options (configs/models/gemini-live.yaml).
FLASH_LIVE = Preset(
    "flash-live",
    "gemini-live",
    "tool_transcript",
    {"will_continue": False, "scheduling": "SILENT"},
    turn_complete,
)


def turn_script(
    k: int, step: Step, preset: Preset, *, resumable: bool = True, go_away: bool = False
) -> list:
    """The server's reply to decision k: speech or calls, a resumption update, then completion.

    ``resumable=False`` withholds the resumption handle (gemini/session.py SessionCore._control);
    ``go_away`` asks the client to reconnect after the turn (gemini/session.py SessionCore._control;
    context.py RealtimeRuntime._run_turn).
    """
    calls, speech = native_reply(step.reply)
    script = []
    if speech:
        script.append(
            {
                "serverContent": {
                    "modelTurn": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": AUDIO_MIME,
                                    "data": base64.b64encode(bytes(960)).decode(),
                                }
                            }
                        ]
                    }
                }
            }
        )
        script.append({"serverContent": {"outputTranscription": {"text": speech}}})
    if calls:
        function_calls = [
            {"id": f"call-{k}-{index}", "name": name, "args": args}
            for index, (name, args) in enumerate(calls)
        ]
        script += [{"toolCall": {"functionCalls": function_calls}}, AWAIT_ACK]
    update = {"newHandle": f"handle-{k}", "resumable": True} if resumable else {"resumable": False}
    script.append({"sessionResumptionUpdate": update})
    if go_away:
        script.append({"goAway": {"timeLeft": "30s"}})
    return script + preset.completion(k, usage_metadata(k, step, speech))


@pytest.fixture
def live_env(monkeypatch):
    """Client settings for the stub, set explicitly so that a repository .env cannot fill them
    (create_agent loads it with setdefault: harnesses/registry.py create_agent,
    providers/registry.py load_project_secrets)."""
    monkeypatch.setenv("GEMINI_API_KEY", API_KEY)  # key slot 1 (live.py GeminiLiveDriver.__init__)
    monkeypatch.setenv("GEMINI_API_KEY_START_SLOT", "1")  # live.py GeminiLiveDriver.__init__
    # Either would switch the client to Vertex AI (google/genai/_api_client.py:655-681).
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "")
    # Selects the SDK's record/replay client (google/genai/client.py:278, 460-475).
    monkeypatch.setenv("GOOGLE_GENAI_CLIENT_MODE", "")
    return monkeypatch


def catalog_config(key: str) -> ModelConfig:
    """What the runner builds for a catalog key (runner.py main). Its
    api_base_url stays None; the stub is reached through GOOGLE_GEMINI_BASE_URL."""
    effort = catalog.get(key).defaults["reasoning_effort"]
    return catalog.to_model_config(key, fps=FPS, reasoning_effort=effort)


def step_frames(stub: LiveStub, first: int) -> list[dict]:
    """The client's frames and the connection events since ``first``; the server's frames are the
    script."""
    return [event for event in stub.events[first:] if event["dir"] != "s2c"]


def run_live_steps(
    episode: Episode, steps: list[Step], stub: LiveStub
) -> tuple[list[dict], list[dict]]:
    """steps.run_steps for a WebSocket session: per decision, the frames the stub received while
    the decision ran. Returns the records and the raw results."""
    agent = episode.agent
    records, results = [], []
    for k, step in enumerate(steps):
        observation = episode.frames.capture(k)
        budget = episode.budget(observation.tick_number)  # t_k: shell.py _start_generation
        first = len(stub.events)
        result = decide(agent, k, observation, budget)
        results.append(result)
        record = {
            "record": "step",
            "step": k,
            "observation": {
                "observation_id": observation.metadata["observation_id"],
                "tick": observation.tick_number,
                "timestamp": observation.timestamp,
                "screenshot_sha256": observation.metadata["screenshot_sha256"],
            },
            "budget": {
                "runtime_output_tokens": budget.runtime_output_tokens,
                "provider_output_tokens": budget.provider_output_tokens,
                "transport_output_tokens": budget.transport_output_tokens,
            },
            "action": result["action"],
            # A failed decision has no action protocol (harnesses/default/harness.py
            # DefaultHarness._generate).
            "action_protocol": result.get("action_protocol"),
            "usage": result["usage"],
            # What the runner charges the token clock with (episode_records.py clock_output_tokens).
            "clock_usage": {
                name: result.get(name)
                for name in (
                    "provider_usage_missing_reason",
                    "provider_output_tokens",
                    "native_received_output_tokens",
                    "native_received_output_tokens_source",
                )
            },
        }
        if result.get("invalid_reason"):
            # run_stream invalidates the episode and stops without committing
            # (kernel/turns.py _unusable).
            stub.wait_until_idle()
            invalid = {name: result.get(name) for name in ("invalid_reason", "invalid_detail")}
            records.append({**record, "frames": step_frames(stub, first), **invalid})
            break
        commit(agent, k, step.status, step.detail)
        stub.wait_until_idle()
        records.append(
            {
                **record,
                "frames": step_frames(stub, first),
                "commit": {"status": step.status, "detail": step.detail},
                "action_history_entry": dict(agent.history.actions[-1]),
            }
        )
    return records, results


def run_episode(
    name: str,
    config: ModelConfig,
    model_key: str,
    steps: list[Step],
    turns: list[list],
    preset: Preset,
    env,
    tmp_path: Path,
    *,
    setup_closes: dict[str, tuple[int, str]] | None = None,
    connection_closes: dict[int, tuple[int, str]] | None = None,
    before_initialize=None,
) -> tuple[list[dict], list[dict], LiveStub]:
    """Run one script against the stub; returns the baseline records, the raw results and the
    stopped stub. ``before_initialize`` goes to build_episode."""
    with LiveStub(
        turns,
        cert_dir=tmp_path / "tls",
        api_key=API_KEY,
        history_format=preset.history_format,
        ack_options=preset.ack_options,
        setup_closes=setup_closes,
        connection_closes=connection_closes,
    ) as stub:
        # Both are read when the driver builds its client, inside build_episode.
        env.setenv("GOOGLE_GEMINI_BASE_URL", stub.base_url)
        env.setenv("SSL_CERT_FILE", str(stub.cert_file))
        episode = build_episode(
            config,
            tmp_path / "run",
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=model_key,
            clock_mode="token",
            tokens_per_frame=TOKENS_PER_FRAME,
            max_ticks=MAX_TICKS,
            before_initialize=before_initialize,
        )
        try:
            stub.wait_until_idle()
            initialize = step_frames(stub, 0)
            records, results = run_live_steps(episode, steps, stub)
            last = len(stub.events)
        finally:
            episode.finish()
    header = {
        "record": "header",
        "api": "gemini_live",
        "scenario": name,
        "clock": {
            "mode": "token",
            "fps": FPS,
            "tokens_per_frame": TOKENS_PER_FRAME,
            "max_ticks": MAX_TICKS,
        },
        "config": config.model_dump(),
    }
    closing = {"record": "close", "frames": step_frames(stub, last)}
    records = [header, {"record": "initialize", "frames": initialize}, *records, closing]
    return records, results, stub


def session_waits(run_dir: Path) -> list[str]:
    """What the driver waited for, in order: the SessionWait records of the episode trace,
    written before each wait (providers/session_lifecycle.py _record_wait)."""
    lines = (run_dir / "episode_trace.jsonl").read_text(encoding="utf-8").splitlines()
    return [e["payload"]["wait"] for e in map(json.loads, lines) if e["event"] == "SessionWait"]


# Each rebuild closes the connection and waits out the session release before it reconnects;
# agent.close() closes the last connection.
REBUILD_WAITS = ["connection_close", "session_release"] * len(REBUILD_STEPS) + ["connection_close"]


@contextlib.contextmanager
def matches_baseline(name: str, records: list[dict], run_dir: Path):
    """Compare with the baseline before the checks in the body, so that any change first shows as
    field-level differences; with REPLAY_RECORD=1, write it only after the checks pass."""
    actual = WireNormalizer()([*records, *baseline.trajectory(run_dir)])
    path = BASELINES / f"gemini_live-{name}.jsonl"
    recording = os.environ.get("REPLAY_RECORD") == "1"
    if not recording:
        problems = baseline.compare(baseline.read(path), actual)
        assert not problems, f"{len(problems)} difference(s) from {path.name}:\n" + "\n".join(
            problems[:60]
        )
    yield
    if recording:
        baseline.write(path, actual)


def check_session(stub: LiveStub, results: list[dict]) -> None:
    assert stub.violations == []
    assert stub.unscripted == 0, "the agent sent observations beyond the script"
    assert stub.unused_turns == 0, "scripted replies were left unused"
    assert all(result["inspection"]["valid"] for result in results)
    assert not any(result.get("invalid_reason") for result in results)


def client_messages(record: dict, key: str) -> list[dict]:
    return [frame["msg"][key] for frame in record["frames"] if key in frame["msg"]]


def seed_and_observations(record: dict) -> tuple[list[dict], list[dict]]:
    """A decision's client_content split into the history seeded after a setup with
    historyConfig, up to the message completing it (live.py GeminiLiveDriver._open), and the
    observations (live.py GeminiLiveDriver.send_observation). A rebuild seeds before the
    observation, a replay after the turn seeds after it (harnesses/default/context.py
    RealtimeRuntime._run_turn)."""
    seed, observations, seeding = [], [], False
    for frame in record["frames"]:
        message = frame["msg"]
        if "setup" in message:
            seeding = "historyConfig" in message["setup"]
        elif "client_content" in message:
            content = message["client_content"]
            if seeding:
                seed += content["turns"]
                seeding = not content["turnComplete"]
            else:
                observations.append(content)
    return seed, observations


def seed_contents(record: dict) -> list[dict]:
    return seed_and_observations(record)[0]


def observation_texts(record: dict) -> list[str]:
    """The texts of the decision's observation, for either transport (live.py send_observation):
    the feedback the session did not hold yet, each its own Content or text, then the frame's
    prompt."""
    realtime_input = client_messages(record, "realtime_input")
    texts = [message["text"] for message in realtime_input if "text" in message]
    if texts:
        return texts
    (observation,) = seed_and_observations(record)[1]
    return [turn["parts"][0]["text"] for turn in observation["turns"]]


def check_rebuilds(steps: list[dict], results: list[dict]) -> None:
    opened = [
        k for k, record in enumerate(steps) if any(f["dir"] == "open" for f in record["frames"])
    ]
    assert opened == list(REBUILD_STEPS)
    for k, result in enumerate(results):
        runtime = result["realtime_runtime"]
        sync = runtime["history_sync"]
        if k in REBUILD_STEPS:
            assert (sync["disposition"], sync["detail"]) == ("rebuilt", "history_block_compaction")
            assert sync["session_connect_mode"] == "canonical_history_replay"
            assert runtime["history_folded_visuals"] == FOLDED
            (setup,) = client_messages(steps[k], "setup")
            assert setup["historyConfig"] == {"initial_history_in_client_content": True}
        else:
            assert (sync["disposition"], sync["detail"]) == ("exact", "already synchronized")
    firsts = {k: seed_contents(steps[k])[0] for k in REBUILD_STEPS}
    texts = {k: first["parts"][0].get("text", "") for k, first in firsts.items()}
    ledgers = [k for k, text in texts.items() if text.startswith(ARCHIVED_ACTION_LEDGER_HEADER)]
    assert ledgers == [ARCHIVE_STEP]
    assert firsts[ARCHIVE_STEP]["role"] == "user"
    assert len(texts[ARCHIVE_STEP].splitlines()) == 1 + ARCHIVED
    runtime = results[ARCHIVE_STEP]["realtime_runtime"]
    assert runtime["history_archived_turns"] == ARCHIVED
    assert runtime["history_archived_actions_total"] == ARCHIVED


def check_rejection_feedback(steps: list[dict]) -> None:
    """A rejected action's detail reaches the model as feedback of its own ahead of the next
    frame, whose prompt it no longer opens (none of these frames rebuilds the session)."""
    rejected = [k for k, record in enumerate(steps) if record["commit"]["detail"]]
    assert rejected and not set(rejected) & {k - 1 for k in REBUILD_STEPS}
    for k in rejected:
        feedback, prompt = observation_texts(steps[k + 1])
        assert feedback.startswith("EXECUTION RESULT: ") and "status=rejected." in feedback
        assert steps[k]["commit"]["detail"] in feedback
        assert not prompt.startswith("EXECUTION RESULT: ")


def check_token_clock(steps: list[dict], config: ModelConfig) -> None:
    caps = [record["budget"]["runtime_output_tokens"] for record in steps]
    assert min(caps) > 0
    below = [k for k, cap in enumerate(caps) if cap < config.native_output_limit]
    assert below == list(range(FIRST_CAPPED_STEP, len(steps)))


@pytest.mark.parametrize("preset", [TOOL_TRANSCRIPT, EXTENDED_THINKING], ids=lambda p: p.name)
def test_long_scenario(preset, live_env, tmp_path):
    config = catalog_config(preset.model)
    turns = [turn_script(k, step, preset) for k, step in enumerate(LONG_SCENARIO)]
    name = f"{preset.name}-long"
    records, results, stub = run_episode(
        name, config, preset.model, LONG_SCENARIO, turns, preset, live_env, tmp_path
    )
    steps = records[2:-1]
    with matches_baseline(name, records, tmp_path / "run"):
        check_session(stub, results)
        check_rebuilds(steps, results)
        check_rejection_feedback(steps)
        check_token_clock(steps, config)
    assert session_waits(tmp_path / "run") == REBUILD_WAITS


DEFAULT_SCENARIO = [
    Step(CLICK_NAME),
    # two calls: WAIT (parsing.py FunctionCallTurnCollector.finish)
    Step(f"{CLICK_NAME} {CLICK_AGREE}", status="wait"),
    Step(CLICK_AGREE),
]


def test_driver_default_short(live_env, tmp_path):
    """A model that sets the driver's default options (live.py LIVE_OPTIONS), with the standard
    native_tools history. The goAway of decision 1
    comes without a resumption handle, so the session is replayed at once (live.py
    GeminiLiveDriver._sync_history), in the native_tools format with the thought-signature sentinel
    (gemini/session.py REPLAY_THOUGHT_SIGNATURE, SessionCore._tool_contents)."""
    preset = DRIVER_DEFAULT
    config = ModelConfig(
        adapter_type="gemini_live",
        model_name=preset.model,
        fps=FPS,
        reasoning_effort="low",
        provider_options=DRIVER_DEFAULT_OPTIONS,
    )
    turns = [
        turn_script(k, step, preset, resumable=k != 1, go_away=k == 1)
        for k, step in enumerate(DEFAULT_SCENARIO)
    ]
    name = "driver-default-short"
    records, results, stub = run_episode(
        name, config, preset.model, DEFAULT_SCENARIO, turns, preset, live_env, tmp_path
    )
    steps = records[2:-1]
    with matches_baseline(name, records, tmp_path / "run"):
        check_session(stub, results)
        # From decision 1 on, the feedback on the decision before goes as a text of its own.
        for k, record in enumerate(steps):
            sent = [next(iter(message)) for message in client_messages(record, "realtime_input")]
            assert sent == list(realtime_input_order(feedback=int(k > 0)))
            assert seed_and_observations(record)[1] == []
        sync = results[1]["realtime_runtime"]["history_sync"]
        assert (sync["disposition"], sync["detail"]) == ("rebuilt", "go_away")
        seed = seed_contents(steps[1])
        calls = [content["parts"] for content in seed if content["role"] == "model"]
        assert [len(parts) for parts in calls] == [1, 2]
        for parts in calls:
            assert parts[0]["thoughtSignature"] == "skip_thought_signature_validator"
            assert all("thoughtSignature" not in part for part in parts[1:])
    # realtime_input waits after each video. At 1 fps the 1 s ingestion wait already covers the
    # frame interval, so no video waits for it. The reads after the goAway wait for its deadline.
    assert session_waits(tmp_path / "run") == [
        "video_ingestion",
        "video_ingestion",
        "go_away_deadline",
        "connection_close",
        "session_release",
        "video_ingestion",
        "connection_close",
    ]


def test_flash_live_short(live_env, tmp_path):
    """3.1 Flash Live (configs/models/gemini-live.yaml) on the driver-default script. Every setup
    declares the functions NON_BLOCKING (live.py
    GeminiLiveDriver._function_declarations), and each acknowledgement is the queued response,
    SILENT and not continued (gemini/session.py SessionCore.tool_response_options;
    live.py GeminiLiveDriver._events). The goAway of decision 1 comes without a resumption handle,
    so the session is replayed at once (gemini/session.py SessionCore._tool_contents; live.py
    GeminiLiveDriver._sync_history), and its tool transcript holds each receipt as it was sent
    (live.py GeminiLiveDriver._events; harnesses/default/request.py _turn_items)."""
    preset = FLASH_LIVE
    config = catalog_config(preset.model)
    turns = [
        turn_script(k, step, preset, resumable=k != 1, go_away=k == 1)
        for k, step in enumerate(DEFAULT_SCENARIO)
    ]
    name = "flash-live-short"
    records, results, stub = run_episode(
        name, config, preset.model, DEFAULT_SCENARIO, turns, preset, live_env, tmp_path
    )
    steps = records[2:-1]
    with matches_baseline(name, records, tmp_path / "run"):
        check_session(stub, results)
        setups = [setup for record in records[1:-1] for setup in client_messages(record, "setup")]
        assert ["historyConfig" in setup for setup in setups] == [False, True]
        for setup in setups:
            (tool,) = setup["tools"]
            assert {item["behavior"] for item in tool["functionDeclarations"]} == {"NON_BLOCKING"}
        acknowledged = [
            message["functionResponses"]
            for record in steps
            for message in client_messages(record, "tool_response")
        ]
        assert [[response["id"] for response in responses] for responses in acknowledged] == [
            ["call-0-0"],
            ["call-1-0", "call-1-1"],
            ["call-2-0"],
        ]
        for responses in acknowledged:
            assert all(
                (response["scheduling"], response["will_continue"]) == ("SILENT", False)
                for response in responses
            )
        sync = results[1]["realtime_runtime"]["history_sync"]
        assert (sync["disposition"], sync["detail"]) == ("rebuilt", "go_away")
        assert sync["session_connect_mode"] == "canonical_history_replay"
        # The seed after the turn replays decisions 0 and 1, each call as model text.
        texts = [
            content["parts"][0]["text"]
            for content in seed_contents(steps[1])
            if content["role"] == "model"
        ]
        assert all(text.startswith(TRANSCRIPT_HEADER) for text in texts)
        receipts = [
            exchange["function_responses"]
            for text in texts
            for exchange in json.loads(text[len(TRANSCRIPT_HEADER) :])
        ]
        assert receipts == acknowledged[:2]
    # The reads after the goAway wait for its deadline; the replay closes the connection and
    # waits out the session release.
    assert session_waits(tmp_path / "run") == [
        "go_away_deadline",
        "connection_close",
        "session_release",
        "connection_close",
    ]


class _SleepRecordingClock(SystemClock):
    """The driver's clock, whose waits (live.py GeminiLiveDriver._connect,
    GeminiLiveDriver.send_observation) only advance its monotonic time and are recorded; everything
    else is real time."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def wait(self, waitable, timeout):
        self.sleeps.append(timeout)
        self.now += timeout
        return False


def test_realtime_input_video_follows_the_model_input_rate(live_env, tmp_path):
    """On the realtime_input transport (the driver defaults) one video frame follows the last by
    at least one model-input frame interval (live.py GeminiLiveDriver.send_observation): 4 s at 0.25
    fps. Each video is followed by the 1 s ingestion wait (live.py
    GeminiLiveDriver.send_observation), so the wait before the next video is the other 3 s."""
    clock = _SleepRecordingClock()
    preset = DRIVER_DEFAULT
    config = ModelConfig(
        adapter_type="gemini_live",
        model_name=preset.model,
        fps=0.25,
        reasoning_effort="low",
        provider_options=DRIVER_DEFAULT_OPTIONS,
    )
    scenario = [Step(CLICK_NAME), Step(CLICK_AGREE), Step(CLICK_NAME)]
    turns = [turn_script(k, step, preset) for k, step in enumerate(scenario)]
    records, results, stub = run_episode(
        "realtime-input-spacing",
        config,
        preset.model,
        scenario,
        turns,
        preset,
        live_env,
        tmp_path,
        before_initialize=lambda agent: setattr(agent.driver, "clock", clock),
    )
    check_session(stub, results)
    for k, record in enumerate(records[2:-1]):
        sent = [next(iter(message)) for message in client_messages(record, "realtime_input")]
        assert sent == list(realtime_input_order(feedback=int(k > 0)))
    assert clock.sleeps == [1.0, 3.0, 1.0, 3.0, 1.0]
    assert session_waits(tmp_path / "run") == [
        "video_ingestion",
        "video_interval",
        "video_ingestion",
        "video_interval",
        "video_ingestion",
        "connection_close",
    ]


def test_resume_failure_short(live_env, tmp_path):
    """After a goAway the driver resumes with the promoted handle; the server rejects it, and the
    driver replays the full history on a new session instead (live.py
    GeminiLiveDriver._sync_history)."""
    preset = TOOL_TRANSCRIPT
    config = catalog_config(preset.model)
    scenario = LONG_SCENARIO[:3]
    turns = [turn_script(k, step, preset, go_away=k == 1) for k, step in enumerate(scenario)]
    name = "resume-failure-short"
    records, results, stub = run_episode(
        name,
        config,
        preset.model,
        scenario,
        turns,
        preset,
        live_env,
        tmp_path,
        setup_closes={"handle-1": (1008, "Session resumption handle is invalid.")},
    )
    steps = records[2:-1]
    with matches_baseline(name, records, tmp_path / "run"):
        check_session(stub, results)
        resume, replay = client_messages(steps[1], "setup")
        assert resume["sessionResumption"] == {"handle": "handle-1"}
        assert "historyConfig" not in resume
        assert replay["sessionResumption"] == {} and "historyConfig" in replay
        sync = results[1]["realtime_runtime"]["history_sync"]
        assert (sync["disposition"], sync["detail"]) == ("rebuilt", "go_away")
        assert sync["session_connect_mode"] == "canonical_history_replay"
        assert "Session resumption handle is invalid." in sync["resume_error"]
    # The resume closes the old connection and waits the release; its refused setup is closed by
    # google-genai, and the replay that follows waits the release again.
    assert session_waits(tmp_path / "run") == [
        "go_away_deadline",
        "connection_close",
        "session_release",
        "session_release",
        "connection_close",
    ]


def test_a_repair_whose_rebuild_is_closed_transiently_keeps_the_generation(live_env, tmp_path):
    """Bug #12, as decision 35 for OpenAI Realtime: the goAway of decision 0 comes without a
    resumption handle, so the repair after the generation replays the history on a new session,
    whose setup the server closes with a transient 1011. The repair comes back broken instead of
    raising the close (live.py GeminiLiveDriver._sync_history): the decision keeps its output and
    fails as history_sync_failed, not as a stream interrupted after its output, which drops it."""
    preset = TOOL_TRANSCRIPT
    config = catalog_config(preset.model)
    scenario = LONG_SCENARIO[:1]
    turns = [turn_script(0, scenario[0], preset, resumable=False, go_away=True)]
    _, (result,), stub = run_episode(
        "repair-closed",
        config,
        preset.model,
        scenario,
        turns,
        preset,
        live_env,
        tmp_path,
        connection_closes={2: (1011, "Internal error.")},
    )
    assert stub.violations == [] and stub.unused_turns == 0
    assert result["invalid_reason"] == "history_sync_failed"
    assert "Internal error." in result["invalid_detail"]
    assert result["raw"] == scenario[0].reply
    assert result["action"].action_type.value == "click"
    assert result["realtime_runtime"]["history_sync"]["disposition"] == "broken"


def test_a_first_connect_refused_for_the_quota_fails_as_an_exhausted_quota(live_env, tmp_path):
    """The server closes the episode's first connection at its setup with 1011 for an exhausted
    quota, as on the workstation (trace-review pbglive 4.1). The connect makes its one attempt,
    and initialize() raises its failure as the driver classifies a turn's (live.py
    GeminiLiveDriver.classify_transport_error), which the runner invalidates the run for, not as
    the bare APIError it took for its own run_exception."""
    preset = TOOL_TRANSCRIPT
    quota = "You exceeded your current quota, please check your plan and billing details."
    agents = []
    with LiveStub(
        [],
        cert_dir=tmp_path / "tls",
        api_key=API_KEY,
        history_format=preset.history_format,
        ack_options=preset.ack_options,
        connection_closes={1: (1011, quota)},
    ) as stub:
        live_env.setenv("GOOGLE_GEMINI_BASE_URL", stub.base_url)
        live_env.setenv("SSL_CERT_FILE", str(stub.cert_file))
        try:
            with pytest.raises(TransportFailureError) as raised:
                build_episode(
                    catalog_config(preset.model),
                    tmp_path / "run",
                    task=ReplayTask(),
                    instruction=TASK_INSTRUCTION,
                    model_key=preset.model,
                    clock_mode="token",
                    tokens_per_frame=TOKENS_PER_FRAME,
                    max_ticks=MAX_TICKS,
                    before_initialize=agents.append,
                )
        finally:
            # What Episode.finish closes; build_episode raised before it made the episode.
            for agent in agents:
                agent.close()
                close_agent_trace(agent._trace)
                agent._trace.trace.close()
    failure = raised.value.failure
    assert (failure.kind, failure.invalid_reason) == (QUOTA_EXHAUSTED, "provider_quota_exhausted")
    assert quota in str(raised.value.__cause__)
    assert {event["conn"] for event in stub.events} == {1}
    assert stub.violations == []


def test_a_fold_whose_rebuild_is_refused_for_the_quota_fails_as_an_exhausted_quota(
    live_env, tmp_path
):
    """Decision 20 folds the oldest screenshots, and the session, which Gemini Live cannot edit
    in place, is rebuilt from the history before the decision's input (REBUILD_STEPS;
    harnesses/default/context.py RealtimeRuntime._run_turn). The server closes the rebuild's
    connection at its setup with 1011 for an exhausted quota, which the sync takes as broken
    (live.py GeminiLiveDriver._sync_history). The decision fails for that, not retried, as an
    exhausted quota, not as the bare provider_request_failed of the sync's own failure."""
    preset = TOOL_TRANSCRIPT
    quota = "You exceeded your current quota, please check your plan and billing details."
    scenario = LONG_SCENARIO[: REBUILD_STEPS[0] + 1]
    turns = [turn_script(k, step, preset) for k, step in enumerate(scenario)]
    _, (*_, failed), stub = run_episode(
        "fold-quota",
        catalog_config(preset.model),
        preset.model,
        scenario,
        turns,
        preset,
        live_env,
        tmp_path,
        connection_closes={2: (1011, quota)},
    )
    assert failed["invalid_reason"] == "provider_quota_exhausted"
    assert failed["invalid_detail"] == (
        f"RuntimeError(\"history synchronization failed: APIError('1011 None. {quota}')\")"
    )
    # Not retried: no connection follows the rebuild's.
    assert {event["conn"] for event in stub.events} == {1, 2}
    assert stub.violations == []


def test_a_call_its_function_schema_rejects_executes_wait(live_env, tmp_path):
    """Decision 0 calls click with its coordinates as strings, which the declared schema
    rejects, though the text parser would read them. The call is no action
    (FunctionCallTurnCollector.finish): the harness executes WAIT and tells the model its response
    was not exactly one valid action. The output stays the call as the model made it."""
    preset = TOOL_TRANSCRIPT
    config = catalog_config(preset.model)
    call = '{"action": "click", "x": "384", "y": "130"}'
    scenario = [Step(call, status="wait"), Step(CLICK_NAME)]
    turns = [turn_script(k, step, preset) for k, step in enumerate(scenario)]
    records, results, stub = run_episode(
        "schema-rejected-call", config, preset.model, scenario, turns, preset, live_env, tmp_path
    )
    check_session(stub, results)
    rejected = results[0]
    assert rejected["action"].action_type.value == "wait"
    assert rejected["raw"] == call
    assert "must both be numeric" in rejected["function_validation_error"]
    protocol = rejected["action_protocol"]
    assert (protocol["protocol_failure"], protocol["parse_mode"]) == ("invalid_action_output", None)
    steps = records[2:-1]
    assert steps[0]["action_history_entry"]["requested"] == "invalid_action"
    feedback, _ = observation_texts(steps[1])
    assert feedback.startswith("EXECUTION RESULT: actual=wait(); requested=invalid_action;")
    assert feedback.endswith("(invalid_action_output)")


def test_interrupted_empty_turn_retry_keeps_its_usage(live_env, tmp_path):
    """The server interrupts decision 1 before any output and reports only its prompt. The
    collector raises EmptyInterruptedToolTurnError (harnesses/default/parsing.py
    FunctionCallTurnCollector.finish), a transient failure (providers/gemini/live.py
    GeminiLiveDriver._is_transient) that the harness replays although the attempt had usage
    (harnesses/default/harness.py DefaultHarness._generate). The retry rebuilds the session from canonical
    history and succeeds; the decision keeps the interrupted attempt's usage."""
    live_env.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    preset = TOOL_TRANSCRIPT
    config = catalog_config(preset.model)
    scenario = LONG_SCENARIO[:2]
    turns = [turn_script(k, step, preset) for k, step in enumerate(scenario)]
    prompt = 1760
    turns.insert(1, interrupted_before_output(prompt))
    records, results, stub = run_episode(
        "interrupted-empty-turn",
        config,
        preset.model,
        scenario,
        turns,
        preset,
        live_env,
        tmp_path,
    )
    check_session(stub, results)
    runtime = results[1]["realtime_runtime"]
    assert runtime["transport_reconnect_attempts"] == 1
    assert runtime["transport_reconnect_reason"].startswith(
        "transient: EmptyInterruptedToolTurnError("
    )
    (usage,) = runtime["transport_retry_usage"]
    assert (usage["input_tokens"], usage["output_tokens"]) == (prompt, 0)
    assert usage["raw_native_usage"] == {"prompt_token_count": prompt, "total_token_count": prompt}
    assert usage["native_response_usage_exact"] is True
    assert runtime["transport_retry_usage_status"] == "complete"
    # Only the retry opens a connection: its recovery closes the interrupted one and waits the
    # release before it replays the history (providers/session_lifecycle.py
    # SessionLifecycle.ensure_ready; live.py GeminiLiveDriver._connect).
    steps = records[2:-1]
    opened = [
        k for k, record in enumerate(steps) if any(f["dir"] == "open" for f in record["frames"])
    ]
    assert opened == [1]
    assert session_waits(tmp_path / "run") == [
        "connection_close",
        "session_release",
        "connection_close",
    ]
