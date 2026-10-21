"""The Gemini Live driver in native streaming (providers/gemini/live.py, ``streaming=True``), on
google-genai's AsyncSession over a socket the test feeds.

A streaming connection is read as messages arrive, so the scripted socket of
tests/session/test_gemini_live_receive.py, which hands out its whole script at once, would turn a
reply into orphans before its generation started. FedSocket hands out only what the test fed it,
when it fed it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace

from google.genai import live as genai_live
from google.genai import types

from realtime_framework.entity.actions import gui_function_declarations
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.model import ConnectRequest, SessionSeed
from realtime_framework.entity.records import UNTRACED
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.request import system_block
from realtime_framework.harnesses.streaming.history import history_view
from realtime_framework.providers.gemini import live
from tests.session.test_gemini_live_receive import RecordingClock

MODEL = (
    "gemini-3.8-live"  # the client_content preset: streaming goes on realtime_input all the same
)


class FedSocket:
    """What google-genai's AsyncSession reads and writes: the server messages the test feeds, in
    order, each once fed; an exception fed is raised by the read that takes it."""

    def __init__(self, number: int):
        self.trace_connection_id = f"gemini_live.ws-{number}"
        self.sent: list[dict] = []
        self._frames: asyncio.Queue = asyncio.Queue()

    def feed(self, *frames) -> None:
        for frame in frames:
            self._frames.put_nowait(
                frame if isinstance(frame, BaseException) else json.dumps(frame)
            )

    async def recv(self, decode=None):
        frame = await self._frames.get()
        if isinstance(frame, BaseException):
            raise frame
        return frame

    async def send(self, message):
        self.sent.append(json.loads(message))


class FedLive:
    """client.aio.live: each connect opens a session on a new FedSocket; ``configs`` holds each
    connect's setup."""

    def __init__(self):
        self.sockets: list[FedSocket] = []
        self.configs: list = []

    def connect(self, *, model, config):
        self.configs.append(config)
        socket = FedSocket(len(self.sockets) + 1)
        self.sockets.append(socket)
        session = genai_live.AsyncSession(
            api_client=SimpleNamespace(vertexai=False),
            websocket=socket,
            setup_complete=types.LiveServerSetupComplete(),
        )
        return contextlib.asynccontextmanager(lambda: _yield(session))()


async def _yield(value):
    yield value


class Rig:
    """A GeminiLiveDriver for ``model`` on fed connections, in native streaming unless
    ``streaming`` is False; ``log`` holds its waits."""

    def __init__(self, model: str = MODEL, *, streaming: bool = True):
        self.log: list = []
        config = ModelConfig(
            adapter_type="gemini_live",
            model_name=model,
            api_key="replay-dummy-key",
            **live.GeminiLiveDriver.model_defaults.get(model, {}),
        )
        self.driver = live.GeminiLiveDriver(
            config, clock=RecordingClock(self.log), streaming=streaming
        )
        self.driver.bind_trace(UNTRACED)
        self.live = FedLive()
        self.driver.client = SimpleNamespace(aio=SimpleNamespace(live=self.live))
        self.answer = AnswerRequirements(
            "function_call", "standard_vlm", tuple(gui_function_declarations())
        )
        self.seed = SessionSeed("system instruction", model, None, 1.0, self.answer)
        self.policy = current_context_policy().for_path(
            config.adapter_type, config.provider_options, config.context_preset
        )
        self.call = CallParameters(model, config.reasoning_effort, None)

    def view(self, blocks, *, frames: int = 2):
        """The streaming history view of the accepted ``blocks``."""
        return history_view(
            blocks,
            policy=self.policy,
            system=system_block(self.seed.system_instruction, None),
            retained_frames=frames,
            answer=self.answer,
            call=self.call,
        )

    @property
    def socket(self) -> FedSocket:
        return self.live.sockets[-1]

    async def connect(self, history=None) -> None:
        await self.driver.connect(ConnectRequest(seed=self.seed, history=history))


def test_a_streaming_setup_compresses_the_context_and_lets_a_trigger_not_interrupt():
    """A streaming session's setup turns on the server's sliding-window compression at the
    driver's thresholds, takes realtime input whatever transport the model's preset names, and
    lets a new activity leave a reply in progress alone; the connection records the thresholds,
    and that what the server keeps of the context is unknown. A turn-based setup has neither."""
    rig, turn_based = Rig(), Rig(streaming=False)

    async def run():
        await rig.connect()
        await turn_based.connect()
        await rig.driver.close()
        await turn_based.driver.close()

    asyncio.run(run())
    (config,) = rig.live.configs
    settings = live.GeminiLiveDriver.streaming_settings
    compression = config.context_window_compression
    assert compression.trigger_tokens == settings["compression_trigger_tokens"]
    assert compression.sliding_window.target_tokens == settings["compression_target_tokens"]
    realtime = config.realtime_input_config
    assert realtime.automatic_activity_detection.disabled
    assert realtime.activity_handling == types.ActivityHandling.NO_INTERRUPTION
    assert realtime.turn_coverage == types.TurnCoverage.TURN_INCLUDES_ALL_INPUT
    assert config.session_resumption.transparent
    metadata = rig.driver._connect_metadata
    assert metadata["gemini_live_context_window_compression"] == settings
    assert metadata["gemini_live_server_context"] == "unknown"

    (turn_based_config,) = turn_based.live.configs
    assert turn_based_config.context_window_compression is None
    assert turn_based_config.realtime_input_config is None  # the preset's client_content
    assert not turn_based_config.session_resumption.transparent
    assert "gemini_live_context_window_compression" not in turn_based.driver._connect_metadata
