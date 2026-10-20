"""Physical request-context checks shared by stateless drivers."""

from __future__ import annotations

import json


def estimated_request_tokens(messages, image_tokens: int) -> int:
    """Estimate a pending request as UTF-8 text bytes plus a calibrated per-image allowance.

    Do not count base64 bytes as text. The image allowance is deployment-specific and must be
    calibrated; this estimate is never used for billing or the token clock.
    """
    image_count = 0
    text_messages = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "image_url":
                    image_count += 1
                    part = {**part, "image_url": "<image>"}
                parts.append(part)
            message = {**message, "content": parts}
        text_messages.append(message)
    return (
        len(json.dumps(text_messages, ensure_ascii=False).encode("utf-8"))
        + image_count * image_tokens
    )


def context_status(config, count: int, counter: str) -> dict:
    """Compare a counted request with the configured window minus its generation reserve."""
    limit = config.context_window_tokens
    reserve = config.context_reserve_tokens
    return {
        "input_tokens": count,
        "native_limit": limit,
        "reserve_tokens": reserve,
        "soft_limit": limit - reserve,
        "pressure": count > limit - reserve,
        "counter": counter,
    }


def validate_context_reserve(config) -> None:
    if (
        config.context_window_tokens is not None
        and config.context_reserve_tokens >= config.context_window_tokens
    ):
        raise ValueError("context reserve must be smaller than deployed context capacity")
