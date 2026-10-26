"""What the live drivers connect with: Gemini Live's key pool (its GeminiDeployment), and the
bound on each connection's handshake."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from types import SimpleNamespace

import pytest

import realtime_framework.entity.model as model_module
from realtime_framework.entity.actions import gui_function_declarations
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.model import (
    TRANSIENT,
    ConnectRequest,
    SessionSeed,
    TransportFailureError,
    exception_chain,
)
from realtime_framework.entity.records import UNTRACED
from realtime_framework.entity.request import AnswerRequirements
from realtime_framework.providers import openai_realtime
from realtime_framework.providers.gemini import live

POOL = ("GEMINI_API_KEY", "GOOGLE_API_KEY", *(f"GEMINI_API_KEY_{index}" for index in range(2, 11)))
SEED = SessionSeed(
    "system instruction",
    "m",
    None,
    1.0,
    AnswerRequirements("function_call", "standard_vlm", tuple(gui_function_declarations())),
)


def gemini_driver(monkeypatch, env: dict, **fields) -> live.GeminiLiveDriver:
    """A driver built in an environment that holds only ``env`` of the key variables."""
    for name in (*POOL, "GEMINI_API_KEY_START_SLOT"):
        monkeypatch.delenv(name, raising=False)
    # Either would switch the client to Vertex AI (google/genai/_api_client.py:655-681).
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return live.GeminiLiveDriver(ModelConfig(adapter_type="gemini_live", model_name="m", **fields))


@pytest.mark.parametrize(
    "env, fields, keys, slot",
    [
        # The pool: the default variable, then its _2 to _9.
        (
            {"GEMINI_API_KEY": "k1", "GEMINI_API_KEY_3": "k3", "GEMINI_API_KEY_9": "k9",
             "GEMINI_API_KEY_10": "k10"},
            {"api_key_env": "GEMINI_API_KEY"},
            ("k1", "k3", "k9"),
            1,
        ),
        # GOOGLE_API_KEY stands in for an unset GEMINI_API_KEY; the start slot wraps the pool.
        (
            {"GOOGLE_API_KEY": "g", "GEMINI_API_KEY_2": "k2", "GEMINI_API_KEY_START_SLOT": "3"},
            {},
            ("g", "k2"),
            1,
        ),
        ({"GEMINI_API_KEY": "k1", "OTHER": "o"}, {"api_key_env": "OTHER"}, ("o",), 1),
        ({"GEMINI_API_KEY": "k1"}, {"api_key": "given"}, ("given",), 1),
    ],
    ids=["pool", "fallback-and-slot", "other-variable", "given-key"],
)
def test_gemini_key_pool(monkeypatch, env, fields, keys, slot):
    driver = gemini_driver(monkeypatch, env, **fields)
    assert (driver._api_keys, driver._api_key_index + 1) == (keys, slot)


def test_gemini_needs_a_key(monkeypatch):
    with pytest.raises(RuntimeError) as caught:
        gemini_driver(monkeypatch, {})
    assert str(caught.value) == (
        "Gemini Live needs GEMINI_API_KEY/GOOGLE_API_KEY or ModelConfig.api_key"
    )


@pytest.mark.parametrize("slot", ["x", "0"])
def test_gemini_start_slot_is_a_positive_integer(monkeypatch, slot):
    with pytest.raises(ValueError) as caught:
        gemini_driver(monkeypatch, {"GEMINI_API_KEY": "k1", "GEMINI_API_KEY_START_SLOT": slot})
    assert str(caught.value) == "GEMINI_API_KEY_START_SLOT must be a positive integer"


class SetupNeverAcknowledged:
    """client.aio.live whose server takes the setup and never answers it, as google-genai's
    connect would wait for setupComplete; the connection closes on the way out."""

    def __init__(self):
        self.closed = False

    def connect(self, *, model, config):
        return contextlib.asynccontextmanager(self._session)()

    async def _session(self):
        try:
            await asyncio.Event().wait()
            yield
        finally:
            self.closed = True


def test_gemini_handshake_is_bounded(monkeypatch):
    """A connect whose setup is never acknowledged fails at the handshake bound (live.py
    GeminiLiveDriver._open) with a transient failure, and its connection is closed."""
    monkeypatch.setattr(live, "_HANDSHAKE_TIMEOUT_S", 0.05)
    driver = gemini_driver(monkeypatch, {"GEMINI_API_KEY": "k1"})
    driver.bind_trace(UNTRACED)
    stalled = SetupNeverAcknowledged()
    driver.client = SimpleNamespace(aio=SimpleNamespace(live=stalled))
    started = time.monotonic()
    with pytest.raises(TimeoutError) as caught:
        asyncio.run(driver.connect(ConnectRequest(seed=SEED)))
    assert time.monotonic() - started < 1.0
    assert driver.classify_transport_error(caught.value).kind == TRANSIENT
    assert stalled.closed


def test_openai_handshake_is_bounded(monkeypatch):
    """A server that takes the TCP connection and never answers the WebSocket upgrade fails each
    connect attempt at the handshake bound (openai_realtime.py OpenAIRealtimeDriver._connect), a
    transient failure that the connect retries until the shared retry budget is spent."""
    monkeypatch.setattr(openai_realtime, "_HANDSHAKE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(model_module, "_backoff", lambda clock, cancel, seconds: None)
    with socket.create_server(("127.0.0.1", 0)) as server:  # never accepts
        driver = openai_realtime.OpenAIRealtimeDriver(
            ModelConfig(
                adapter_type="openai_realtime",
                model_name="m",
                api_base_url=f"http://127.0.0.1:{server.getsockname()[1]}/v1",
                api_key="sk-handshake",
            )
        )
        driver.bind_trace(UNTRACED)
        started = time.monotonic()
        with pytest.raises(TransportFailureError) as caught:
            asyncio.run(driver.connect(ConnectRequest(seed=SEED)))
    assert time.monotonic() - started < 2.0
    assert caught.value.failure.kind == TRANSIENT
    assert any(isinstance(error, TimeoutError) for error in exception_chain(caught.value))
