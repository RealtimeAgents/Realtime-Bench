from __future__ import annotations

# Token usage
"""Dependency-free contracts for normalizing provider token usage."""

TOKEN_NORMALIZATION_SCHEMA = "realtime_bench.provider_token_normalization.v1"


def _dict(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return value if isinstance(value, dict) else None


def _number(data, *names):
    data = _dict(data)
    if data is None:
        return None, None
    for name in names:
        value = data.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return int(value), name
    return None, None


def _output_details(usage):
    usage = _dict(usage) or {}
    for name in (
        "output_token_details",
        "output_tokens_details",
        "completion_tokens_details",
    ):
        details = _dict(usage.get(name))
        if details is not None:
            return details, name
    return {}, None


def _existing_number(existing, name):
    value = (existing or {}).get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return None


def normalize_provider_token_usage(
    usage,
    *,
    protocol_family: str | None = None,
    upstream_provider: str | None = None,
    existing: dict | None = None,
) -> dict:
    """Return reliable provider token splits plus their exact derivation.

    ``text_tokens`` is intentionally not treated as visible output. In OpenAI Realtime usage it
    is a modality total that includes reasoning tokens. Known inclusive-total contracts instead
    derive visible output from ``output_tokens - reasoning_tokens``. Unknown contracts retain an
    unknown visible split.
    """
    raw = _dict(usage)
    values = raw or {}
    existing = existing or {}
    details, details_name = _output_details(values)

    provider_output, output_name = _number(values, "output_tokens", "completion_tokens")
    provider_reasoning, reasoning_name = _number(details, "reasoning_tokens", "thinking_tokens")
    if provider_reasoning is None:
        provider_reasoning, reasoning_name = _number(
            values,
            "reasoning_tokens",
            "reasoning_output_tokens",
            "thoughts_token_count",
        )
        reasoning_path = reasoning_name
    else:
        reasoning_path = f"{details_name}.{reasoning_name}"
    provider_audio, audio_name = _number(details, "audio_tokens")
    audio_path = f"{details_name}.{audio_name}" if audio_name else None

    # Adapter-normalized values are authoritative for transports such as Gemini Live whose native
    # response/audio accounting is intentionally transformed before it reaches this boundary.
    for field in (
        "provider_output_tokens",
        "provider_reasoning_output_tokens",
        "provider_audio_output_tokens",
    ):
        explicit = _existing_number(existing, field)
        if explicit is not None:
            if field == "provider_output_tokens":
                provider_output = explicit
            elif field == "provider_reasoning_output_tokens":
                provider_reasoning = explicit
            else:
                provider_audio = explicit

    accounting_contract = values.get("output_accounting")
    if accounting_contract:
        semantics = str(accounting_contract)
    elif protocol_family == "openai_realtime":
        semantics = "openai_realtime_output_tokens_include_reasoning_v1"
    elif protocol_family == "openai_chat":
        semantics = "openrouter_chat_completion_tokens_include_reasoning_v1"
    elif protocol_family == "moonshot_chat":
        semantics = "moonshot_chat_completion_tokens_include_reasoning_v1"
    elif protocol_family == "anthropic_messages":
        semantics = "anthropic_messages_output_tokens_include_thinking_v2"
    elif protocol_family:
        semantics = f"{protocol_family}_provider_usage"
    else:
        semantics = "unidentified_provider_usage"

    visible = _existing_number(existing, "provider_visible_output_tokens")
    visible_source = existing.get("provider_visible_output_tokens_source")
    visible_missing_reason = existing.get("provider_visible_output_tokens_missing_reason")
    if visible is None:
        visible, visible_name = _number(values, "visible_output_tokens")
        if visible is not None:
            visible_source = f"provider_usage.{visible_name}"
            visible_missing_reason = None
        else:
            inclusive_reasoning_contract = protocol_family in {
                "openai_realtime",
                "openai_chat",
                "moonshot_chat",
                "anthropic_messages",
            } or accounting_contract in {
                "anthropic_messages_output_tokens_include_thinking_v2",
                "gemini_native_tool_turn_usage_excludes_audio_v2",
            }
            if inclusive_reasoning_contract and provider_output is not None:
                if provider_reasoning is None:
                    visible_missing_reason = "provider_reasoning_breakdown_unavailable"
                elif provider_reasoning > provider_output:
                    visible_missing_reason = "provider_reasoning_exceeds_output_total"
                else:
                    visible = provider_output - provider_reasoning
                    visible_source = (
                        "derived:provider_output_tokens_minus_provider_reasoning_output_tokens"
                    )
                    visible_missing_reason = None
            elif raw is None:
                visible_missing_reason = "provider_usage_unavailable"
            else:
                visible_missing_reason = "provider_schema_does_not_define_visible_split"

    def source(field, fallback):
        return existing.get(field) or fallback

    return {
        "provider_token_normalization_schema": TOKEN_NORMALIZATION_SCHEMA,
        "provider_token_semantics": semantics,
        "provider_output_tokens": provider_output,
        "provider_output_tokens_source": source(
            "provider_output_tokens_source",
            f"provider_usage.{output_name}" if output_name else None,
        ),
        "provider_reasoning_output_tokens": provider_reasoning,
        "provider_reasoning_output_tokens_source": source(
            "provider_reasoning_output_tokens_source",
            f"provider_usage.{reasoning_path}" if reasoning_path else None,
        ),
        "provider_visible_output_tokens": visible,
        "provider_visible_output_tokens_source": visible_source,
        "provider_visible_output_tokens_missing_reason": visible_missing_reason,
        "provider_audio_output_tokens": provider_audio,
        "provider_audio_output_tokens_source": source(
            "provider_audio_output_tokens_source",
            f"provider_usage.{audio_path}" if audio_path else None,
        ),
        "provider_audio_output_tokens_missing_reason": (
            None if provider_audio is not None else "provider_audio_breakdown_unavailable"
        ),
        "provider_protocol_family": protocol_family,
        "provider_upstream": upstream_provider,
    }


