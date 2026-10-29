"""LiteLLM 1.106.1 behaviors the five LiteLLM-backed APIs depend on, pinned on the wire.

The drivers in providers/litellm.py hand LiteLLM a request. LiteLLM rewrites parts of it before the
provider sees it, drops or rejects some parameters, and rewrites some provider answers before the
driver reads them; none of that is visible in our code. Each test builds a driver from an explicit
ModelConfig, runs it against a local stub (tests/agent_trace/stubs.py), and asserts on what the stub
received and what the driver emitted, so a LiteLLM upgrade that changes one of these behaviors fails
the test that names it. Docstrings cite the 1.106.1 source, relative to site-packages/litellm/.

No test imports LiteLLM or patches its attributes. LITELLM_USER_AGENT reaches it through
monkeypatch.setenv because LiteLLM reads it each time its AsyncHTTPHandler builds a client
(llms/custom_httpx/http_handler.py:152, read at :700), which the driver does at its first request.
LITELLM_DROP_PARAMS (__init__.py:249) and LITELLM_ROUTE_ALL_CHAT_OPENAI_TO_RESPONSES (:268-270)
are read once, at ``import litellm``, so their tests also run the driver in a child process. The
silent resend after a connection error on httpx's transport is pinned in
tests/agent_trace/test_http.py.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path

import pytest

import realtime_framework.entity.model as model_module
from realtime_framework.clock import SystemClock
from realtime_framework.entity.actions import Observation
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import path_policy, resolve_context
from realtime_framework.entity.events import GenerationBoundary, TextDelta, UsageSnapshot
from realtime_framework.entity.model import (
    FATAL,
    TRANSPORT_RETRY,
    CanonicalTurn,
    GenerationRequest,
    ProviderObservation,
)
from realtime_framework.entity.records import trace_scope
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.evaluation.agent_trace import AgentTraceSink, close_agent_trace, load_body
from realtime_framework.evaluation.reporting import EpisodeTrace
from realtime_framework.harnesses.default.request import build_request, system_block
from realtime_framework.providers.litellm import LITELLM_VERSION, MoonshotDriver, chat_messages
from realtime_framework.providers.registry import adapter_class

from ..agent_trace.stubs import Answer, HttpStub
from ..agent_trace.test_http import _anthropic_answer, _events, _openai_answer, _payloads
from .build import build_episode
from .frames import ReplayTask
from .scenario import TASK_INSTRUCTION
from .steps import decide
from .stub import CREATED

KEY_ENV = "REPLAY_BEHAVIOR_KEY"
KEY = "sk-replay-behavior-0123456789abcdef"
SYSTEM = "Answer with one action."
IMAGE = b"png"
ADAPTIVE = {"type": "adaptive", "display": "summarized"}
LITELLM_USER_AGENT = f"litellm/{LITELLM_VERSION}"
SDK_USER_AGENT = f"AsyncOpenAI/Python {version('openai')}"
EMPTY_PLACEHOLDER = "[System: Empty message content sanitised to satisfy protocol]"
OUTPUT_LIMIT_MESSAGE = (
    "Could not finish the message because max_tokens or model output limit was reached. "
    "Please try again with higher max_tokens."
)
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.setenv(KEY_ENV, KEY)
    # hermetic.py clears every LITELLM_* variable; these others change requests too.
    for name in ("DISABLE_AIOHTTP_TRANSPORT", "OR_SITE_URL", "OR_APP_NAME"):
        monkeypatch.delenv(name, raising=False)


def _model(origin: str, adapter_type: str, model_name: str, base: str, **fields) -> ModelConfig:
    return ModelConfig(
        adapter_type=adapter_type,
        model_name=model_name,
        api_base_url=origin + base,
        api_key_env=KEY_ENV,
        native_output_limit=4096,
        native_output_limit_source="endpoint",
        fps=1.0,
        **fields,
    )


def _turn(k: int, output: str = "", reasoning: str = "", feedback: str = "") -> CanonicalTurn:
    observation = ProviderObservation(
        observation=Observation(b"raw", 100.0 + k, k, {"observation_id": f"obs-{k}"}),
        prompt=f"frame {k}",
        image_bytes=IMAGE,
        mime_type="image/png",
        image_detail="high",
        geometry={},
    )
    return CanonicalTurn(observation, output, feedback, decision=k, output_reasoning=reasoning)


def _canonical(config: ModelConfig, turns=(), reasoning_replay: bool = False):
    """The canonical request of ``turns`` and a frame under ``config``'s policy, with its reasoning
    replay on when ``reasoning_replay``: past the pre-run checks, which refuse it where the API
    cannot send it."""
    policy = path_policy(config)
    if reasoning_replay:
        policy = replace(policy, reasoning_replay=True)
    return build_request(
        policy,
        system_block(SYSTEM, None),
        (*turns, _turn(len(turns))),
        None,
        AnswerRequirements("text_json", "standard_vlm"),
        CallParameters(config.model_name, config.reasoning_effort, None),
    )


@dataclass
class Generated:
    events: list
    error: Exception | None
    failure: str | None  # the driver's classification of ``error``
    trace: list[dict]


def _generate(config, run_dir: Path, canonical=None, *, cap=4096, driver_class=None):
    """One generation through the driver, outside the harness; failures are returned."""
    run_dir.mkdir(parents=True, exist_ok=True)
    clock = SystemClock()
    trace = EpisodeTrace(run_dir / "episode_trace.jsonl", run_id="behavior", clock=clock)
    sink = AgentTraceSink(trace, run_dir / "episode_blobs.pack", clock=clock)
    driver = (driver_class or adapter_class(config.adapter_type))(config)
    driver.bind_trace(sink)
    request = GenerationRequest(
        generation_id="behavior-0",
        transport_output_token_cap=cap,
        canonical=canonical or _canonical(config),
    )

    async def generate():
        try:
            with trace_scope(phase="decision", decision=0, generation_id="behavior-0", attempt=1):
                handle = await driver.start_generation(request)
                return [event async for event in driver.events(handle)]
        finally:
            await driver.close()

    events, error = [], None
    try:
        events = asyncio.run(generate())
    except Exception as exc:  # noqa: BLE001 - the failure is what several tests assert on
        error = exc
    finally:
        close_agent_trace(sink)
        trace.close()
    lines = (run_dir / "episode_trace.jsonl").read_text().splitlines()
    failure = None if error is None else driver.classify_transport_error(error).kind
    return Generated(events, error, failure, [json.loads(line) for line in lines])


def _chat_answer(index, received):
    return _openai_answer(0, received)


def _messages_answer(index, received):
    return _anthropic_answer(0, received)


def _sent(stub: HttpStub) -> dict:
    (wire,) = stub.requests
    return json.loads(wire.body)


def _headers(stub: HttpStub) -> dict:
    (wire,) = stub.requests
    return {name.lower(): value for name, value in wire.headers}


def _visible(events) -> list[str]:
    return [e.text for e in events if isinstance(e, TextDelta) and e.channel == "visible"]


def _cache_marks(body: dict) -> list[tuple[int, int]]:
    return [
        (m, p)
        for m, message in enumerate(body["messages"])
        if isinstance(message["content"], list)
        for p, part in enumerate(message["content"])
        if "cache_control" in part
    ]


# ---- OpenRouter ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter_type, model_name, upstream",
    [
        ("openrouter", "anthropic/claude-opus-5", "anthropic"),
        ("litellm", "openrouter/anthropic/claude-opus-5", None),
    ],
)
def test_openrouter_requests_ask_for_usage(adapter_type, model_name, upstream, tmp_path):
    """OpenRouter's transform_request adds ``usage: {"include": true}`` to every body without a
    ``usage`` key (llms/openrouter/chat/transformation.py:161-162), on the OpenRouter API and on the
    generic API with an ``openrouter/`` model."""
    with HttpStub(_chat_answer) as stub:
        config = _model(
            stub.origin,
            adapter_type,
            model_name,
            "/api/v1",
            upstream_provider=upstream,
            chat_stream=adapter_type != "litellm",
        )
        assert _generate(config, tmp_path).error is None
    assert _sent(stub)["usage"] == {"include": True}


@pytest.mark.parametrize(
    "adapter_type, model_name, upstream, reasoning",
    [
        (
            "openrouter",
            "anthropic/claude-opus-5",
            "anthropic",
            {"reasoning": {"effort": "max", "exclude": False}},
        ),
        ("openrouter", "anthropic/claude-opus-5", None, {"reasoning_effort": "xhigh"}),
        ("litellm", "openrouter/anthropic/claude-opus-5", None, {"reasoning_effort": "xhigh"}),
    ],
)
def test_openrouter_effort_max_is_sent_as_xhigh_unless_nested(
    adapter_type, model_name, upstream, reasoning, tmp_path
):
    """OpenRouter's map_openai_params rewrites a top-level ``reasoning_effort="max"`` to
    ``"xhigh"`` (llms/openrouter/chat/transformation.py:55-57). With a pinned upstream the
    OpenRouter driver nests the effort in ``extra_body.reasoning``, which transform_request merges
    into the body unread (:155-157), so ``max`` reaches OpenRouter; without one, and on the generic
    API, it does not."""
    with HttpStub(_chat_answer) as stub:
        config = _model(
            stub.origin,
            adapter_type,
            model_name,
            "/api/v1",
            upstream_provider=upstream,
            reasoning_effort="max",
            chat_stream=adapter_type != "litellm",
        )
        assert _generate(config, tmp_path).error is None
    sent = _sent(stub)
    assert {key: sent[key] for key in ("reasoning", "reasoning_effort") if key in sent} == reasoning


@pytest.mark.parametrize(
    "model_name, options, kept",
    [
        ("anthropic/claude-opus-5", {"cache_breakpoint": True}, True),
        ("google/gemini-3.8-flash", {"cache_breakpoint": True}, True),
        ("google/gemma-4-31b", {"cache_breakpoint": True}, False),
        ("deepseek/deepseek-v4.1-flash", {"cache_breakpoint": True}, False),
        ("anthropic/claude-opus-5", {}, False),
        ("z-ai/glm-5", {"cache_breakpoint": True}, True),
    ],
)
def test_openrouter_cache_control_survives_only_for_listed_model_names(
    model_name, options, kept, tmp_path
):
    """Outside CacheControlSupportedModels (claude, gemini, minimax, glm, z-ai: llms/openrouter/
    chat/transformation.py:30-37, matched as substrings of the model name at :75-83), LiteLLM
    removes every ``cache_control`` key from the messages (:85-94, llms/openai/chat/
    gpt_transformation.py:401-423). ``google/gemma-4-31b`` is marked by the driver exactly as
    ``google/gemini-3.8-flash`` is, so the difference is LiteLLM's name list. The driver marks a
    model whose YAML sets the cache_breakpoint option, whatever its name: an ``anthropic/`` model
    without it is not marked, and ``z-ai/glm-5`` with it is."""
    with HttpStub(_chat_answer) as stub:
        config = _model(
            stub.origin,
            "openrouter",
            model_name,
            "/api/v1",
            upstream_provider=model_name.split("/")[0],
            provider_options=options,
        )
        assert _generate(config, tmp_path).error is None
    sent = _sent(stub)
    last = len(sent["messages"]) - 1
    assert _cache_marks(sent) == (
        [(last, len(sent["messages"][last]["content"]) - 1)] if kept else []
    )


def _done_answer(index, received):
    head = {
        "id": "chatcmpl-done",
        "object": "chat.completion.chunk",
        "created": CREATED,
        "model": json.loads(received.body)["model"],
    }
    pieces = ["before ", "say [DONE] now", " after"]
    events = [
        {**head, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
        for piece in pieces
    ]
    events.append({**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    usage = {"prompt_tokens": 1500, "completion_tokens": 3, "total_tokens": 1503}
    events.append({**head, "choices": [], "usage": usage})
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return Answer(200, body.encode(), "text/event-stream")


@pytest.mark.parametrize(
    "adapter_type, model_name, base, visible",
    [
        ("openrouter", "anthropic/claude-opus-5", "/api/v1", ["before ", " after"]),
        ("moonshot", "kimi-k3", "/v1", ["before ", "say [DONE] now", " after"]),
        ("chat_completions", "replay-vlm", "/v1", ["before ", "say [DONE] now", " after"]),
    ],
)
def test_done_inside_a_streamed_chunk_drops_that_chunk_on_openrouter(
    adapter_type, model_name, base, visible, tmp_path
):
    """LiteLLM's own SSE iterator, which the OpenRouter API streams through, treats any line that
    contains ``[DONE]`` as an end-of-stream marker and returns an empty finished chunk without
    parsing it (llms/base_llm/base_model_iterator.py:121), so that chunk's text is lost and the
    stream goes on. The APIs on our OpenAI SDK client parse the SSE themselves and keep it."""
    with HttpStub(_done_answer) as stub:
        config = _model(stub.origin, adapter_type, model_name, base, chat_include_usage=True)
        generated = _generate(config, tmp_path)
    assert generated.error is None
    assert _visible(generated.events) == visible


# ---- Anthropic Messages ----------------------------------------------------------------------

THINKING = [
    # model, reasoning_effort, output cap, sent thinking, sent output_config, anthropic-beta
    ("claude-opus-5", None, 4096, ADAPTIVE, None, None),
    ("claude-opus-5", "none", 4096, {"type": "disabled"}, None, None),
    ("claude-opus-5", "low", 4096, ADAPTIVE, {"effort": "low"}, None),
    ("claude-opus-5", "xhigh", 4096, ADAPTIVE, {"effort": "xhigh"}, None),
    ("claude-opus-5", "max", 4096, ADAPTIVE, {"effort": "max"}, None),
    ("claude-replay-9", "max", 4096, ADAPTIVE, {"effort": "max"}, None),
    ("claude-fable-5", "none", 4096, None, None, None),
    ("claude-fable-5", "low", 4096, ADAPTIVE, {"effort": "low"}, None),
    ("claude-sonnet-4-6", "max", 4096, ADAPTIVE, {"effort": "max"}, None),
    ("claude-opus-4-5", "none", 4096, {"type": "disabled"}, None, None),
    ("claude-opus-4-5", None, 4096, {"type": "enabled", "budget_tokens": 2048}, None, None),
    (
        "claude-opus-4-5",
        "low",
        4096,
        {"type": "enabled", "budget_tokens": 2048},
        {"effort": "low"},
        "effort-2025-11-24",
    ),
    (
        "claude-opus-4-5",
        "low",
        1500,
        {"type": "enabled", "budget_tokens": 1499},
        {"effort": "low"},
        "effort-2025-11-24",
    ),
    ("claude-opus-4-5", "low", 1024, None, {"effort": "low"}, "effort-2025-11-24"),
]


@pytest.mark.parametrize("model_name, effort, cap, thinking, output_config, beta", THINKING)
def test_anthropic_thinking_is_rewritten_from_the_model_map(
    model_name, effort, cap, thinking, output_config, beta, tmp_path
):
    """The driver always sends adaptive thinking, or ``disabled`` for ``none``
    (providers/litellm.py anthropic_thinking); LiteLLM rewrites it from the bundled model map.

    - ``thinking_always_on`` models (claude-fable-5): ``disabled`` is removed
      (llms/anthropic/common_utils.py:898-915, called at llms/anthropic/chat/transformation.py:
      2018-2022).
    - Models without ``supports_adaptive_thinking`` (claude-opus-4-5): adaptive becomes the
      legacy ``enabled`` budget of reasoning_effort ``medium``, whatever the effort
      (transformation.py:1629-1656, constants.py DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET
      2048), capped below max_tokens and dropped when max_tokens is at most 1024 (:1384-1398);
      ``output_config.effort`` stays and adds the effort beta header (common_utils.py:967-1001,
      :1086).
    - claude-replay-9 is in no map entry; the ``claude-adaptive-thinking`` fallback rule of the
      bundled map gives it adaptive thinking, as claude-opus-5 has.
    """
    with HttpStub(_messages_answer) as stub:
        config = _model(stub.origin, "anthropic_messages", model_name, "", reasoning_effort=effort)
        assert _generate(config, tmp_path, cap=cap).error is None
    sent = _sent(stub)
    assert (sent.get("thinking"), sent.get("output_config"), sent["max_tokens"]) == (
        thinking,
        output_config,
        cap,
    )
    assert _headers(stub).get("anthropic-beta") == beta


@pytest.mark.parametrize(
    "model_name, effort, error",
    [
        ("claude-opus-4-5", "xhigh", "BadRequestError"),
        ("claude-opus-4-5", "max", "BadRequestError"),
        ("claude-sonnet-4-6", "xhigh", "BadRequestError"),
        ("claude-replay-9", "xhigh", "BadRequestError"),
        ("replay-model", None, "UnsupportedParamsError"),
        ("replay-model", "none", "UnsupportedParamsError"),
        ("replay-model", "low", "UnsupportedParamsError"),
    ],
)
def test_anthropic_rejects_an_unmapped_effort_before_any_request(
    model_name, effort, error, tmp_path
):
    """``xhigh`` needs ``supports_xhigh_reasoning_effort`` and ``max`` adaptive thinking or
    ``supports_max_reasoning_effort`` in the model map (llms/anthropic/chat/transformation.py:
    412-421, raised as BadRequestError from _apply_output_config at :2157-2203). A model id the map
    does not know as a reasoning model gets no ``thinking`` parameter at all (:538-569), and
    get_optional_params rejects it (utils.py:4583-4591; UnsupportedParamsError is a 400,
    exceptions.py:1005). Nothing reaches the provider and the driver classifies it as fatal."""
    with HttpStub(_messages_answer) as stub:
        config = _model(stub.origin, "anthropic_messages", model_name, "", reasoning_effort=effort)
        generated = _generate(config, tmp_path)
    assert stub.requests == []
    assert type(generated.error).__name__ == error
    assert generated.failure == FATAL


def test_anthropic_message_conversion(tmp_path):
    """LiteLLM's Messages conversion (litellm_core_utils/prompt_templates/factory.py
    anthropic_messages_pt, :2435):

    - the leading system message becomes the top-level ``system`` (llms/anthropic/chat/
      transformation.py:2052-2059);
    - consecutive user messages merge into one, so feedback joins the next frame (factory.py:
      2493-2595);
    - an image keeps only its url and format; ``detail`` is dropped (:2504-2513);
    - an empty assistant message gets a placeholder text (:2076, :2464);
    - ``reasoning_content`` on an assistant message is not sent;
    - ``anthropic-version: 2023-06-01`` is sent (llms/anthropic/common_utils.py:1216).

    With reasoning replay on, the driver hands LiteLLM assistant messages with
    ``reasoning_content``.
    """
    turns = (_turn(0, reasoning="r0", feedback="fb0"), _turn(1, "click", "r1", "fb1"))
    with HttpStub(_messages_answer) as stub:
        config = _model(stub.origin, "anthropic_messages", "claude-opus-5", "")
        canonical = _canonical(config, turns, reasoning_replay=True)
        assert _generate(config, tmp_path, canonical).error is None
    ours = chat_messages(canonical)
    assert [m["role"] for m in ours] == [
        "system",
        "user",
        "assistant",
        "user",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert [m.get("reasoning_content") for m in ours if m["role"] == "assistant"] == ["r0", "r1"]
    assert {part["image_url"]["detail"] for m in ours[1::3] for part in m["content"][1:]} == {
        "high"
    }
    sent = _sent(stub)
    assert sent["system"] == [{"type": "text", "text": ours[0]["content"]}]
    assert [m["role"] for m in sent["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    image = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(IMAGE).decode(),
        },
    }

    def text(message):
        return {"type": "text", "text": message["content"]}

    def frame(message):
        return [{"type": "text", "text": message["content"][0]["text"]}, image]

    _, frame0, _, feedback0, frame1, _, feedback1, frame2 = ours
    assert sent["messages"][0]["content"] == frame(frame0)
    assert sent["messages"][1]["content"] == [{"type": "text", "text": EMPTY_PLACEHOLDER}]
    assert sent["messages"][2]["content"] == [text(feedback0), *frame(frame1)]
    assert sent["messages"][3]["content"] == [{"type": "text", "text": "click"}]
    assert sent["messages"][4]["content"] == [
        text(feedback1),
        *frame(frame2)[:1],
        {**image, "cache_control": {"type": "ephemeral"}},
    ]
    assert _headers(stub)["anthropic-version"] == "2023-06-01"


# ---- Moonshot --------------------------------------------------------------------------------


class _CompletionTokensMoonshot(MoonshotDriver):
    compat = replace(MoonshotDriver.compat, output_limit_param="max_completion_tokens")


@pytest.mark.parametrize("driver_class", [MoonshotDriver, _CompletionTokensMoonshot])
def test_moonshot_sends_the_output_limit_as_max_tokens(driver_class, tmp_path):
    """Moonshot's map_openai_params renames ``max_completion_tokens`` to ``max_tokens``
    (llms/moonshot/chat/transformation.py:141-142), so either request field the driver uses
    reaches Kimi as ``max_tokens``."""
    with HttpStub(_chat_answer) as stub:
        config = _model(stub.origin, "moonshot", "kimi-k3", "/v1")
        assert _generate(config, tmp_path, cap=1000, driver_class=driver_class).error is None
    sent = _sent(stub)
    assert sent["max_tokens"] == 1000 and "max_completion_tokens" not in sent


@pytest.mark.parametrize("effort", ["low", "medium", "max"])
def test_moonshot_sends_any_reasoning_effort_for_a_mapped_reasoning_model(effort, tmp_path):
    """kimi-k3 has ``supports_reasoning`` in the bundled map, which makes ``reasoning_effort`` a
    supported parameter (llms/moonshot/chat/transformation.py:121-122); the map's
    ``reasoning_effort_levels`` (low, high, max) are not checked, so ``medium`` goes through."""
    with HttpStub(_chat_answer) as stub:
        config = _model(stub.origin, "moonshot", "kimi-k3", "/v1", reasoning_effort=effort)
        assert _generate(config, tmp_path).error is None
    assert _sent(stub)["reasoning_effort"] == effort


@pytest.mark.parametrize("model_name", ["kimi-replay", "moonshot-v1-8k"])
def test_moonshot_rejects_reasoning_effort_for_a_model_the_map_does_not_mark(model_name, tmp_path):
    """Without ``supports_reasoning`` in the bundled map (an unlisted id, or a listed model without
    the flag) ``reasoning_effort`` is not a supported parameter (llms/moonshot/chat/
    transformation.py:117-123) and, with the driver's ``drop_params=False``, get_optional_params
    raises UnsupportedParamsError (utils.py:4583-4591), a 400 (exceptions.py:1005): no request is
    sent and the driver classifies it as fatal. With no effort the same model is sent."""
    with HttpStub(_chat_answer) as stub:
        config = _model(stub.origin, "moonshot", model_name, "/v1", reasoning_effort="low")
        generated = _generate(config, tmp_path / "low")
        assert stub.requests == []
        assert type(generated.error).__name__ == "UnsupportedParamsError"
        assert "reasoning_effort" in str(generated.error) and generated.failure == FATAL
        unset = config.model_copy(update={"reasoning_effort": None})
        assert _generate(unset, tmp_path / "unset").error is None
    assert "reasoning_effort" not in _sent(stub)


def test_moonshot_keeps_list_content_when_a_message_has_an_image(tmp_path):
    """Moonshot flattens list content to strings only when no message has a non-text part
    (llms/moonshot/chat/transformation.py:42-68, via litellm_core_utils/prompt_templates/
    common_utils.py:82-92). Every request the driver builds carries the current frame's image,
    so text-only content stays as the driver wrote it and the frames stay lists."""
    with HttpStub(_chat_answer) as stub:
        config = _model(stub.origin, "moonshot", "kimi-k3", "/v1")
        canonical = _canonical(config, (_turn(0, "click", feedback="clicked"),))
        assert _generate(config, tmp_path, canonical).error is None
    ours = chat_messages(canonical)
    sent = _sent(stub)["messages"]
    assert [type(m["content"]) for m in sent] == [type(m["content"]) for m in ours]
    assert [type(m["content"]).__name__ for m in sent] == ["str", "list", "str", "str", "list"]


# ---- Settings LiteLLM reads at import ----------------------------------------------------------

# A fresh process, so LiteLLM's import sees the settings the test gives it. They are set after
# hermetic.py, which clears every LITELLM_* variable, and before anything imports LiteLLM.
CHILD = """
import os
import sys
import tests.replay.hermetic
os.environ.update(setting.split("=", 1) for setting in sys.argv[4:])
from tests.replay.test_litellm_behavior import _child
_child(*sys.argv[1:4])
"""


def test_generic_litellm_hands_a_kept_provider_the_past_reasoning(tmp_path):
    """With reasoning replay on, the generic API hands LiteLLM the turns' reasoning_content, and
    a provider whose conversion keeps it (hosted_vllm) sends it: in LiteLLM's final request dict,
    which the trace records, and on the wire."""
    turns = (_turn(0, "wait", "r0", "fb0"), _turn(1, "click", "r1", "fb1"))
    with HttpStub(_chat_answer) as stub:
        config = _model(
            stub.origin, "litellm", "hosted_vllm/replay-vlm", "/v1", chat_stream=False,
            context=resolve_context(
                "litellm", "hosted_vllm/replay-vlm", run={"reasoning_replay": True}
            ),
        )  # fmt: skip
        generated = _generate(config, tmp_path, _canonical(config, turns))
        assert generated.error is None
    replayed = [
        message.get("reasoning_content")
        for message in _sent(stub)["messages"]
        if message["role"] == "assistant"
    ]
    assert replayed == ["r0", "r1"]
    (sent,) = [row["payload"] for row in generated.trace if row["event"] == "ModelRequestSent"]
    body = json.loads(load_body(tmp_path / "episode_blobs.pack", sent["body"]))
    assert [m.get("reasoning_content") for m in body["messages"] if m["role"] == "assistant"] == (
        replayed
    )
    # The count is of LiteLLM's final request dict, the record's body, which is not the wire's.
    assert (sent["replayed_reasoning"], sent["exact_wire"]) == (2, False)


def _child_config(case: str, origin: str) -> ModelConfig:
    if case == "drop_params":
        return _model(origin, "moonshot", "kimi-replay", "/v1", reasoning_effort="low")
    return _model(origin, "litellm", "openai/replay-vlm", "/v1", chat_stream=False)


def _child(case: str, origin: str, run_dir: str) -> None:
    generated = _generate(_child_config(case, origin), Path(run_dir))
    error = None if generated.error is None else type(generated.error).__name__
    outcome = {"error": error, "failure": generated.failure}
    (Path(run_dir) / "child.json").write_text(json.dumps(outcome))


def _in_child(case: str, origin: str, run_dir: Path, **settings: str) -> dict:
    subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD,
            case,
            origin,
            str(run_dir),
            *(f"{name}={value}" for name, value in settings.items()),
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        timeout=300,
    )
    return json.loads((run_dir / "child.json").read_text())


def test_litellm_drop_params_is_read_at_import_and_overrides_drop_params_false(
    tmp_path, monkeypatch
):
    """The driver passes ``drop_params=False``, but get_optional_params drops an unsupported
    parameter when ``litellm.drop_params is True or drop_params is True`` (utils.py:4583-4585),
    and LITELLM_DROP_PARAMS sets ``litellm.drop_params`` once, at import (__init__.py:249,
    litellm_core_utils/core_helpers.py:96-106). Set after the import it changes nothing; set
    before it, the unmarked Moonshot model rejected above is sent without its
    ``reasoning_effort``."""
    with HttpStub(_chat_answer) as stub:
        config = _child_config("drop_params", stub.origin)
        adapter_class(config.adapter_type)(config)
        monkeypatch.setenv("LITELLM_DROP_PARAMS", "true")
        here = _generate(config, tmp_path / "here")
        assert type(here.error).__name__ == "UnsupportedParamsError" and stub.requests == []
        child = _in_child(
            "drop_params", stub.origin, tmp_path / "child", LITELLM_DROP_PARAMS="true"
        )
    assert child == {"error": None, "failure": None}
    sent = _sent(stub)
    assert sent["model"] == "kimi-replay" and "reasoning_effort" not in sent


def test_route_all_chat_openai_to_responses_is_read_at_import(tmp_path, monkeypatch):
    """LITELLM_ROUTE_ALL_CHAT_OPENAI_TO_RESPONSES is read once, at import (__init__.py:268-270).
    Set after the import it changes nothing. Set before it, the generic API hands an ``openai/``
    model's chat request to the Responses API bridge (main.py:1102, :5815-5834), which imports
    the optional ``mcp`` package (responses/main.py:1050, proxy/_experimental/mcp_server/
    catalog.py:20); the locked environment lacks it, so every request fails before it is sent,
    as an APIConnectionError the driver classifies as transient."""
    with HttpStub(_chat_answer) as stub:
        config = _child_config("responses", stub.origin)
        adapter_class(config.adapter_type)(config)
        monkeypatch.setenv("LITELLM_ROUTE_ALL_CHAT_OPENAI_TO_RESPONSES", "true")
        assert _generate(config, tmp_path / "here").error is None
        child = _in_child(
            "responses",
            stub.origin,
            tmp_path / "child",
            LITELLM_ROUTE_ALL_CHAT_OPENAI_TO_RESPONSES="true",
        )
    assert child == {"error": "APIConnectionError", "failure": "transient"}
    assert [received.path for received in stub.requests] == ["/v1/chat/completions"]


# ---- chat_completions ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name, field",
    [
        ("replay-vlm", "max_tokens"),
        ("gpt-5-replay", "max_completion_tokens"),
        ("gpt-6-replay", "max_completion_tokens"),
        ("openai/gpt-5.6-sol", "max_completion_tokens"),
    ],
)
def test_chat_completions_renames_max_tokens_for_gpt5_names(model_name, field, tmp_path):
    """Any model name containing ``gpt-5`` or ``gpt-6`` takes the GPT-5 parameter mapping
    (llms/openai/chat/gpt_5_transformation.py:69), which moves ``max_tokens`` to
    ``max_completion_tokens`` (:329-330), on an OpenAI-compatible endpoint too."""
    with HttpStub(_chat_answer) as stub:
        config = _model(stub.origin, "chat_completions", model_name, "/v1")
        assert _generate(config, tmp_path, cap=1000).error is None
    sent = _sent(stub)
    assert sent[field] == 1000
    assert {"max_tokens", "max_completion_tokens"} & sent.keys() == {field}


def test_chat_completions_rejects_xhigh_for_an_unmapped_gpt5_name(tmp_path):
    """``xhigh`` is opt-in per model on the GPT-5 mapping: a name whose map entry does not list it
    raises UnsupportedParamsError(400) before any request (llms/openai/chat/
    gpt_5_transformation.py:299-309)."""
    with HttpStub(_chat_answer) as stub:
        config = _model(
            stub.origin, "chat_completions", "gpt-5-replay", "/v1", reasoning_effort="xhigh"
        )
        generated = _generate(config, tmp_path)
    assert stub.requests == []
    assert type(generated.error).__name__ == "UnsupportedParamsError"
    assert generated.failure == FATAL


def _output_limit(index, received):
    body = {"error": {"message": OUTPUT_LIMIT_MESSAGE, "type": "invalid_request_error"}}
    return Answer(400, json.dumps(body).encode())


@pytest.mark.parametrize(
    "adapter_type, model_name, status_code",
    [("chat_completions", "replay-vlm", 400), ("litellm", "openai/replay-vlm", None)],
)
def test_an_output_limit_400_becomes_an_empty_reply_with_estimated_usage(
    tmp_path, adapter_type, model_name, status_code
):
    """A non-streamed 400 whose message says the output limit was reached (llms/openai/
    common_utils.py:133-135) is returned as a successful empty completion, finish_reason
    ``length``, with no completion tokens and prompt tokens LiteLLM counts itself (:148-162,
    :189-201; llms/openai/openai.py:534-537). chat_completions' hooked client sees the 400, and
    the driver yields no usage for it (bug #16). The generic API has no hooked client, its
    logged reply looks like a provider's, and the driver still yields the estimate as
    reported."""
    with HttpStub(_output_limit) as stub:
        config = _model(stub.origin, adapter_type, model_name, "/v1", chat_stream=False)
        generated = _generate(config, tmp_path)
    assert generated.error is None and len(stub.requests) == 1
    assert _visible(generated.events) == []
    usages = [e.usage for e in generated.events if isinstance(e, UsageSnapshot)]
    if adapter_type == "chat_completions":
        assert usages == []
    else:
        (usage,) = usages
        assert usage["completion_tokens"] == 0
        assert usage["prompt_tokens"] == usage["total_tokens"] > 0
    assert isinstance(generated.events[-1], GenerationBoundary)
    (response,) = [e["payload"] for e in generated.trace if e["event"] == "ModelResponseReceived"]
    assert response["status_code"] == status_code


def test_a_streamed_output_limit_400_fails_closed(tmp_path):
    """Streamed, LiteLLM answers the same 400 with a one-chunk iterator of its own (llms/openai/
    common_utils.py:200-203), which has no provider SSE behind it; the driver refuses it
    (providers/litellm.py start_generation) and the failure is fatal."""
    with HttpStub(_output_limit) as stub:
        config = _model(
            stub.origin, "chat_completions", "replay-vlm", "/v1", chat_include_usage=True
        )
        generated = _generate(config, tmp_path)
    assert len(stub.requests) == 1
    assert isinstance(generated.error, RuntimeError)
    assert "no auditable provider SSE parser" in str(generated.error)
    assert generated.failure == FATAL


# ---- Headers ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter_type, model_name, base, answer, user_agent",
    [
        ("openrouter", "anthropic/claude-opus-5", "/api/v1", _chat_answer, LITELLM_USER_AGENT),
        ("anthropic_messages", "claude-opus-5", "", _messages_answer, LITELLM_USER_AGENT),
        (
            "litellm",
            "openrouter/anthropic/claude-opus-5",
            "/api/v1",
            _chat_answer,
            LITELLM_USER_AGENT,
        ),
        ("moonshot", "kimi-k3", "/v1", _chat_answer, SDK_USER_AGENT),
        ("chat_completions", "replay-vlm", "/v1", _chat_answer, SDK_USER_AGENT),
        ("litellm", "openai/replay-vlm", "/v1", _chat_answer, SDK_USER_AGENT),
    ],
)
def test_user_agent_names_the_http_client(
    adapter_type, model_name, base, answer, user_agent, tmp_path
):
    """LiteLLM's AsyncHTTPHandler sends ``User-Agent: litellm/<version>`` (llms/custom_httpx/
    http_handler.py:145-160, set per client at :700-701); the OpenAI SDK client, ours or the one
    LiteLLM builds for an ``openai/`` model, sends its own."""
    with HttpStub(answer) as stub:
        config = _model(
            stub.origin,
            adapter_type,
            model_name,
            base,
            upstream_provider="anthropic" if adapter_type == "openrouter" else None,
            chat_stream=adapter_type != "litellm",
        )
        assert _generate(config, tmp_path).error is None
    assert _headers(stub)["user-agent"] == user_agent


@pytest.mark.parametrize(
    "adapter_type, model_name, base, answer, user_agent",
    [
        ("openrouter", "anthropic/claude-opus-5", "/api/v1", _chat_answer, "replay-agent/1"),
        ("anthropic_messages", "claude-opus-5", "", _messages_answer, "replay-agent/1"),
        ("moonshot", "kimi-k3", "/v1", _chat_answer, SDK_USER_AGENT),
    ],
)
def test_litellm_user_agent_replaces_the_handlers_user_agent(
    adapter_type, model_name, base, answer, user_agent, tmp_path, monkeypatch
):
    """LITELLM_USER_AGENT replaces the whole header, read when the handler builds its client
    (llms/custom_httpx/http_handler.py:150-153); the OpenAI SDK client ignores it."""
    monkeypatch.setenv("LITELLM_USER_AGENT", "replay-agent/1")
    with HttpStub(answer) as stub:
        config = _model(
            stub.origin,
            adapter_type,
            model_name,
            base,
            upstream_provider="anthropic" if adapter_type == "openrouter" else None,
        )
        assert _generate(config, tmp_path).error is None
    assert _headers(stub)["user-agent"] == user_agent


@pytest.mark.parametrize(
    "adapter_type, model_name, upstream, referer, title",
    [
        (
            "openrouter",
            "anthropic/claude-opus-5",
            "anthropic",
            "https://github.com/RealtimeAgents/Realtime-Bench",
            "Realtime-Bench",
        ),
        ("litellm", "openrouter/anthropic/claude-opus-5", None, "https://litellm.ai", "liteLLM"),
    ],
)
def test_openrouter_attribution_headers(
    adapter_type, model_name, upstream, referer, title, tmp_path
):
    """On the generic API LiteLLM attributes an ``openrouter/`` request to itself unless
    OR_SITE_URL and OR_APP_NAME are set (main.py:3482-3483); the OpenRouter API sends ours."""
    with HttpStub(_chat_answer) as stub:
        config = _model(
            stub.origin,
            adapter_type,
            model_name,
            "/api/v1",
            upstream_provider=upstream,
            chat_stream=adapter_type != "litellm",
        )
        assert _generate(config, tmp_path).error is None
    headers = _headers(stub)
    assert (headers["http-referer"], headers["x-title"]) == (referer, title)


# ---- Resend after a connection error ---------------------------------------------------------


@pytest.mark.parametrize(
    "adapter_type, model_name, base, per_attempt",
    [
        ("openrouter", "anthropic/claude-opus-5", "/api/v1", 2),
        ("anthropic_messages", "claude-opus-5", "", 2),
        ("moonshot", "kimi-k3", "/v1", 1),
        ("chat_completions", "replay-vlm", "/v1", 1),
        ("litellm", "openai/replay-vlm", "/v1", 1),
    ],
)
def test_a_refused_connection_is_sent_twice_per_attempt_on_litellms_handler(
    adapter_type, model_name, base, per_attempt, tmp_path, monkeypatch
):
    """On the default aiohttp transport a refused connection is a ConnectError
    (llms/custom_httpx/aiohttp_transport.py:41-69), which LiteLLM's AsyncHTTPHandler resends once
    (llms/custom_httpx/http_handler.py:851-866): each harness attempt on the OpenRouter and
    Anthropic APIs records two identical requests. Our OpenAI SDK client, with ``max_retries=0``,
    and the generic API's logger record one."""
    monkeypatch.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    with HttpStub(_chat_answer) as stub:
        origin = stub.origin
    # K3 runs only with its past reasoning back (its YAML's context).
    k3 = {"reasoning_replay": True} if model_name == "kimi-k3" else {}
    config = _model(
        origin,
        adapter_type,
        model_name,
        base,
        upstream_provider="anthropic" if adapter_type == "openrouter" else None,
        chat_stream=adapter_type != "litellm",
        context=resolve_context(adapter_type, model_name, model_layer=k3),
    )
    episode = build_episode(
        config, tmp_path, task=ReplayTask(), instruction=TASK_INSTRUCTION, model_key="replay-model"
    )
    try:
        observation = episode.frames.capture(0)
        result = decide(episode.agent, 0, observation, episode.budget(observation.tick_number))
    finally:
        assert episode.finish() is None
    assert result["invalid_reason"] == "provider_unavailable"
    events = _events(tmp_path)
    attempts = range(1, TRANSPORT_RETRY.max_retries + 2)
    sent = _payloads(events, "ModelRequestSent")
    assert [r["scope"]["attempt"] for r in sent] == [
        a for a in attempts for _ in range(per_attempt)
    ]
    assert len({json.dumps(r["body"]) for r in sent}) == 1
    assert _payloads(events, "ModelResponseReceived") == []
