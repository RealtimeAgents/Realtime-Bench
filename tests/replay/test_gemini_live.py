"""Replay baselines for the gemini_live API on the finite token clock.

Record a baseline with ``REPLAY_RECORD=1 .venv/bin/python -m pytest tests/replay``; without it the
run is compared with tests/replay/baselines/.

The harness builds OpenAI-format messages for this API on every turn, but the driver never reads
them (harnesses/default/harness.py:1209-1217; providers/gemini/live.py:771-794): the model sees the
session's setup, the history replayed on each new connection and one observation per decision.
The baseline is therefore what reaches the WebSocket, the client frames of every connection in
arrival order, plus each decision's parsed action, usage and commit.

Scripts:
- the long scenario for the two maintained presets (providers/registry.py:52-75), which differ in
  the completion signal and the tool acknowledgement;
- the driver defaults, reached by a model id without a profile (evaluation/runner.py:1546-1553):
  realtime_input observations and native_tools history (live.py:62-69);
- a resume that the server rejects, which falls back to replaying the history
  (live.py:1290-1328).
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

import realtime_framework.providers.registry as catalog
from realtime_framework.entity.config import DEFAULT_TOKENS_PER_FRAME, ModelConfig
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER

from . import baseline
from .build import Episode, build_episode
from .frames import ReplayTask
from .scenario import CLICK_AGREE, CLICK_NAME, LONG_SCENARIO, TASK_INSTRUCTION
from .steps import Step
from .ws_gemini_live import AWAIT_ACK, REALTIME_INPUT_ORDER, LiveStub, WireNormalizer

BASELINES = Path(__file__).parent / "baselines"
API_KEY = "replay-dummy-key"

# The runner's defaults at 1 fps (Gemini Live accepts at most 1: live.py:92-98,
# harnesses/default/harness.py:972-980): B is --tokens-per-frame (entity/config.py:33,
# evaluation/runner.py:1346-1351) and the episode has --max-minutes 5 * 60 * fps ticks
# (runner.py:1359, 1773). The catalog's native limit for these models is LiteLLM's 65536
# (providers/registry.py:207-233), so the runtime cap min(65536, (N - t_k) * B) (runner.py:861-887)
# falls below it from t_k = 45 on and is 61696 at the last decision.
FPS = 1.0
TOKENS_PER_FRAME = DEFAULT_TOKENS_PER_FRAME
MAX_TICKS = 300
FIRST_CAPPED_STEP = 45

# Decision k (0-based) holds k earlier turns plus its own frame. A 21st active screenshot folds the
# oldest 10 (entity/config.py:35-36; harnesses/default/context.py:27-36) and a 51st turn archives the
# oldest 10 (config.py:37-38; context.py:262-272). Either change rebuilds the session, because a
# persistent Gemini session cannot be edited in place (context.py:403-412; live.py:92-98). Counted
# from 1, these are decisions 21, 31, 41 and 51.
REBUILD_STEPS = (20, 30, 40, 50)
FOLDED = 10
ARCHIVE_STEP, ARCHIVED = 50, 10

# Replies in LONG_SCENARIO written as a call, and the declared function each one names
# (entity/actions.py:712-847).
CALL_SYNTAX = {
    'click(target="P3")': ("click", {"target": "P3"}),
    'press_key("tab")': ("key", {"key": "tab"}),
}
AUDIO_MIME = "audio/pcm;rate=24000"


def native_reply(reply: str) -> tuple[list[tuple[str, dict]], str]:
    """A scenario reply as Gemini gives it on the function_call channel, where only native calls
    can carry an action (harness.py:1219-1223; parsing.py:837-955): the calls it makes, or what
    it says when the reply holds no action."""
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
    clock does not charge (live.py:796-956). Zero counts are omitted, as in the protobuf JSON."""
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
    """completion_signal=turn_complete ends the turn at turnComplete (live.py:977-981)."""
    return [
        {"serverContent": {"generationComplete": True}},
        {"serverContent": {"turnComplete": True}, "usageMetadata": usage},
    ]


def interaction_idle(k: int, usage: dict) -> list[dict]:
    """completion_signal=interaction_idle spans intermediate turnComplete messages until
    interactionStatus is IDLE (live.py:983-1007). The SDK ends its receive iterator at a bare
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
    model: str  # catalog key, or the model id of a model without a profile
    history_format: str
    ack_options: dict  # what each queued acknowledgement adds (live.py:323-329)
    completion: Callable[[int, dict], list[dict]]


# The two maintained presets (providers/registry.py:52-75) and the driver defaults (live.py:62-69).
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


def turn_script(
    k: int, step: Step, preset: Preset, *, resumable: bool = True, go_away: bool = False
) -> list:
    """The server's reply to decision k: speech or calls, a resumption update, then completion.

    ``resumable=False`` withholds the resumption handle (live.py:1011-1016); ``go_away`` asks the
    client to reconnect after the turn (live.py:1022-1045; context.py:549-558).
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
    (create_agent loads it with setdefault: harnesses/registry.py:11, providers/registry.py:15-39)."""
    monkeypatch.setenv("GEMINI_API_KEY", API_KEY)  # key slot 1 (live.py:108-119)
    monkeypatch.setenv("GEMINI_API_KEY_START_SLOT", "1")  # live.py:126-135
    # Either would switch the client to Vertex AI (google/genai/_api_client.py:655-681).
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "")
    # Selects the SDK's record/replay client (google/genai/client.py:278, 460-475).
    monkeypatch.setenv("GOOGLE_GENAI_CLIENT_MODE", "")
    return monkeypatch


def catalog_config(key: str) -> ModelConfig:
    """What the runner builds for a catalog key (runner.py:1424-1431, 1539-1544). Its
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
        budget = episode.budget(observation.tick_number)  # t_k: runner.py:851
        first = len(stub.events)
        agent.configure_decision(budget)
        agent.stream_prefill(observation)
        result = agent.stream_generate(tick=k)
        agent.finalize_decision_audit(result)  # consumes the trajectory hand-off entry
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
            # A failed decision has no action protocol (harnesses/default/harness.py:1317-1329).
            "action_protocol": result.get("action_protocol"),
            "usage": result["usage"],
            # What the runner charges the token clock with (runner.py:207-229).
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
            # (runner.py:453-459, 1166-1167).
            stub.wait_until_idle()
            invalid = {name: result.get(name) for name in ("invalid_reason", "invalid_detail")}
            records.append({**record, "frames": step_frames(stub, first), **invalid})
            break
        agent.commit_action(step.status, step.detail)
        stub.wait_until_idle()
        records.append(
            {
                **record,
                "frames": step_frames(stub, first),
                "commit": {"status": step.status, "detail": step.detail},
                "action_history_entry": dict(agent.action_history[-1]),
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
) -> tuple[list[dict], list[dict], LiveStub]:
    """Run one script against the stub; returns the baseline records, the raw results and the
    stopped stub."""
    with LiveStub(
        turns,
        cert_dir=tmp_path / "tls",
        api_key=API_KEY,
        history_format=preset.history_format,
        ack_options=preset.ack_options,
        setup_closes=setup_closes,
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
        )
        try:
            stub.wait_until_idle()
            initialize = step_frames(stub, 0)
            records, results = run_live_steps(episode, steps, stub)
            last = len(stub.events)
        finally:
            episode.agent.close()
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
    historyConfig, up to the message completing it (live.py:644-653), and the observations
    (live.py:721-741). A rebuild seeds before the observation, a replay after the turn seeds
    after it (harnesses/default/context.py:403-414, 549-558)."""
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


def observation_prompt(record: dict) -> str:
    """The text of the decision's observation, for either transport (live.py:721-769)."""
    realtime_input = client_messages(record, "realtime_input")
    texts = [message["text"] for message in realtime_input if "text" in message]
    if texts:
        (text,) = texts
        return text
    (observation,) = seed_and_observations(record)[1]
    return observation["turns"][0]["parts"][0]["text"]


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
    """A rejected action's detail reaches the model at the start of the next frame's prompt on a
    persistent session (harness.py:1005-1013, 1526-1546)."""
    rejected = [k for k, record in enumerate(steps) if record["commit"]["detail"]]
    assert rejected
    for k in rejected:
        feedback = observation_prompt(steps[k + 1]).split("\n", 1)[0]
        assert feedback.startswith("EXECUTION RESULT: ") and "status=rejected." in feedback
        assert steps[k]["commit"]["detail"] in feedback


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


DEFAULT_SCENARIO = [
    Step(CLICK_NAME),
    Step(f"{CLICK_NAME} {CLICK_AGREE}", status="wait"),  # two calls: WAIT (parsing.py:883-891)
    Step(CLICK_AGREE),
]


def test_driver_default_short(live_env, tmp_path):
    """A model id without a profile runs on the driver defaults (live.py:62-84). The goAway of
    decision 1 comes without a resumption handle, so the session is replayed at once
    (live.py:1280-1328), in the native_tools format with the thought-signature sentinel
    (live.py:43-47, 459-463)."""
    preset = DRIVER_DEFAULT
    config = ModelConfig(  # runner.py:1434, 1546-1553 for --adapter gemini_live --model <id>
        adapter_type="gemini_live", model_name=preset.model, fps=FPS, reasoning_effort="low"
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
        for record in steps:
            sent = [next(iter(message)) for message in client_messages(record, "realtime_input")]
            assert sent == list(REALTIME_INPUT_ORDER)
            assert seed_and_observations(record)[1] == []
        sync = results[1]["realtime_runtime"]["history_sync"]
        assert (sync["disposition"], sync["detail"]) == ("rebuilt", "go_away")
        seed = seed_contents(steps[1])
        calls = [content["parts"] for content in seed if content["role"] == "model"]
        assert [len(parts) for parts in calls] == [1, 2]
        for parts in calls:
            assert parts[0]["thoughtSignature"] == "skip_thought_signature_validator"
            assert all("thoughtSignature" not in part for part in parts[1:])


def test_resume_failure_short(live_env, tmp_path):
    """After a goAway the driver resumes with the promoted handle; the server rejects it, and the
    driver replays the full history on a new session instead (live.py:1290-1328)."""
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
