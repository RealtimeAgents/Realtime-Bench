"""An OpenAI-compatible chat driver: one complete response per request.

It uses only the framework's public contracts (entity/ and providers/capture.py) and is registered
nowhere; a model YAML selects it with ``endpoint.api: rtb_ext.chat:ExternalChatDriver``.
"""

from __future__ import annotations

import base64
import os
from typing import ClassVar

import httpx

from realtime_framework.entity.events import (
    GenerationBoundary,
    ProviderHistoryState,
    TextDelta,
    UsageSnapshot,
)
from realtime_framework.entity.model import (
    FATAL,
    TRANSIENT,
    Deployment,
    GenerationHandle,
    HistorySyncResult,
    InterruptResult,
    RealtimeCapabilities,
    SyncDisposition,
    TransportFailure,
    classify_http_failure,
    retry_after_seconds,
)
from realtime_framework.providers.capture import HttpCapture

CHANNEL = "rtb_ext.chat"
# The canonical request's neutral roles (entity/request.py) as chat roles.
ROLES = {"instruction": "system", "input": "user", "model": "assistant"}


def chat_messages(request) -> list[dict]:
    """The canonical request as chat messages, in the order and roles the context policy set."""
    messages = []
    for item in request.items:
        block = item.block
        text = block.text
        if block.kind == "action_ledger":
            text = request.policy.texts.ledger_header + "\n" + text
        content = text
        if block.image is not None:
            data = base64.b64encode(block.image.data).decode("ascii")
            content = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.image.mime_type};base64,{data}",
                        "detail": block.image.detail,
                    },
                },
            ]
        messages.append({"role": ROLES[item.intent.role], "content": content})
    return messages


class ExternalChatDriver:
    capabilities = RealtimeCapabilities(
        protocol_family="external_chat",
        persistent_session=False,
        # Stated explicitly: the defaults claim complete usage on a finite token clock.
        finite_frame_budget=False,
        wall_clock_only=True,
        classifies_transport_errors=True,
        request_size_estimation=True,
    )
    deployment = Deployment(
        family="vlm", provider="replay-external", key_env="REPLAY_EXT_API_KEY", vendor="replay"
    )
    model_defaults: ClassVar[dict[str, dict]] = {}

    def __init__(self, config):
        self.config = config
        self._key = os.environ.get(config.api_key_env) if config.api_key_env else None
        self._trace = None
        self._client = None

    def bind_trace(self, trace) -> None:
        self._trace = trace

    async def connect(self, request) -> None:
        return None

    async def send_observation(self, observation, history) -> dict:
        # Stateless: start_generation sends the whole canonical request.
        return {"observation_id": observation.observation_id}

    async def sync_history(self, request) -> HistorySyncResult:
        return HistorySyncResult(
            SyncDisposition.EXACT, ProviderHistoryState.EXACT, detail="stateless"
        )

    async def request_context_status(self, request) -> dict:
        # The context policy archives turns when a request exceeds the context window; this
        # endpoint declares none, so no request does.
        return {"pressure": False, "counter": "unconfigured"}

    async def start_generation(self, request) -> GenerationHandle:
        if self._client is None:
            # The capture hooks record each request and response for the episode trace.
            self._client = httpx.AsyncClient(
                trust_env=False,
                timeout=30.0,
                event_hooks=HttpCapture(self, CHANNEL, "generation").hooks,
            )
        body = {
            "model": self.config.model_name,
            "messages": chat_messages(request.canonical),
        }
        if request.transport_output_token_cap is not None:
            body["max_tokens"] = int(request.transport_output_token_cap)
        if self.config.reasoning_effort is not None:
            body["reasoning_effort"] = self.config.reasoning_effort
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        response = await self._client.post(
            self.config.api_base_url + "/chat/completions", json=body, headers=headers
        )
        response.raise_for_status()
        return GenerationHandle(request.generation_id, metadata={"reply": response.json()})

    async def events(self, generation):
        generation_id = generation.generation_id
        reply = generation.metadata["reply"]
        message = reply["choices"][0]["message"]
        sequence = 0
        if reply.get("usage"):
            sequence += 1
            yield UsageSnapshot(generation_id, sequence, CHANNEL, reply["usage"])
        for channel, text in (
            ("reasoning", message.get("reasoning_content") or ""),
            ("visible", message.get("content") or ""),
        ):
            if text:
                sequence += 1
                yield TextDelta(generation_id, sequence, CHANNEL, text, channel)
        yield GenerationBoundary(
            generation_id, sequence + 1, CHANNEL, "turn_complete", ProviderHistoryState.EXACT
        )

    async def interrupt(self, generation) -> InterruptResult:
        return InterruptResult(ProviderHistoryState.EXACT)

    async def close(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await client.aclose()

    def classify_transport_error(self, exc: BaseException) -> TransportFailure:
        if isinstance(exc, httpx.HTTPStatusError):
            response = exc.response
            return classify_http_failure(
                response.status_code, response.text, retry_after_seconds(response.headers)
            )
        return TransportFailure(TRANSIENT if isinstance(exc, httpx.TransportError) else FATAL)
