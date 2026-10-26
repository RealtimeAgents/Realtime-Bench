"""What the live drivers connect with: Gemini Live's key pool (its GeminiDeployment)."""

from __future__ import annotations

import pytest

from realtime_framework.entity.config import ModelConfig
from realtime_framework.providers.gemini import live

POOL = ("GEMINI_API_KEY", "GOOGLE_API_KEY", *(f"GEMINI_API_KEY_{index}" for index in range(2, 11)))


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
