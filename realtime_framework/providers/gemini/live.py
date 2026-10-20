from __future__ import annotations

# Gemini live
"""Gemini Live protocol driver for the shared realtime runtime."""

import asyncio as asyncio
import json as json
import math as math
import os as os
import time as time

from realtime_framework.entity.model import ConnectRequest
from realtime_framework.entity.events import FunctionCall
from realtime_framework.entity.events import GenerationBoundary
from realtime_framework.entity.model import GenerationHandle
from realtime_framework.entity.model import GenerationRequest
from realtime_framework.entity.model import HistorySyncRequest
from realtime_framework.entity.model import HistorySyncResult
from realtime_framework.entity.model import InterruptResult
from realtime_framework.entity.events import ProviderHistoryState
from realtime_framework.entity.model import ProviderObservation
from realtime_framework.entity.model import RealtimeCapabilities
from realtime_framework.entity.model import SyncDisposition
from realtime_framework.entity.events import TextDelta
from realtime_framework.entity.events import UsageSnapshot
from realtime_framework.entity.actions import gui_function_declarations
from realtime_framework.entity.events import EmptyInterruptedToolTurnError
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER
from realtime_framework.entity.model import FOLDED_VISUAL_PLACEHOLDER
from realtime_framework.entity.model import FATAL
from realtime_framework.entity.model import QUOTA_EXHAUSTED
from realtime_framework.entity.model import RATE_LIMITED
from realtime_framework.entity.model import TRANSIENT
from realtime_framework.entity.model import TransportFailure
from realtime_framework.entity.context_policy import GEMINI_QUEUED_TOOL_RESPONSE
from realtime_framework.entity.context_policy import GEMINI_TOOL_HISTORY_HEADER
from realtime_framework.providers.capture import async_ws_connection
from realtime_framework.providers.legacy import session_frame_text

# Gemini 3 requires a thought signature when a manually reconstructed history contains a model
# function-call part. Live tool-call messages do not expose the original signature because bidi
# function calls use the dedicated ToolCall message rather than model Content. Google documents
# this sentinel for deterministic client-injected function-call history.
_REPLAY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"

# Keep initial-history WebSocket messages small. HistoryConfig deliberately allows several
# client-content messages terminated by one turn_complete=True boundary.
_HISTORY_CONTENTS_PER_MESSAGE = 12
_SESSION_RELEASE_GRACE_S = 2.0
_GO_AWAY_CLOSE_MARGIN_S = 2.0
_CONNECTION_CLOSE_TIMEOUT_S = 1.0
_EXTENDED_THINKING_MODEL = "gemini-3.8-live-extended-thinking"


# Wire options carried in ModelConfig.provider_options. Gemini 3.8 Extended Thinking requires
# asynchronous tool declarations and remains active across intermediate utterance-level
# turnComplete messages until interaction_status becomes IDLE. Ordered screenshot/text turns
# (client_content) avoid synthesizing an audio activity for GUI input.
LIVE_OPTIONS = {
    "tool_behavior": ("blocking", ("blocking", "non_blocking")),
    "completion_signal": ("turn_complete", ("turn_complete", "interaction_idle")),
    "input_transport": ("realtime_input", ("realtime_input", "client_content")),
    # when_idle and omit both send no scheduling field; omit names that choice for Extended Thinking.
    "tool_response_scheduling": ("when_idle", ("when_idle", "silent", "omit")),
    "history_format": ("native_tools", ("native_tools", "tool_transcript")),
}


_RESPONSE_MESSAGES = {"serverContent", "toolCall", "toolCallCancellation", "usageMetadata"}


def classify_live_message(direction: str, message: dict):
    """Name a Gemini Live message for the agent trace; only setup is acknowledged.

    client_content, realtime_input and tool_response have no acknowledgement in the protocol, so
    they stay sent. goAway, sessionResumptionUpdate and voice activity come unsolicited.
    """
    kind = next(iter(message), "")
    if direction == "client":
        return kind, [("setup",)] if kind == "setup" else [], None
    if "setupComplete" in message:
        return kind, [("setup",)], "unsolicited"
    return kind, [], "response" if _RESPONSE_MESSAGES & set(message) else "unsolicited"


def live_options(config) -> dict:
    """Resolve and validate Gemini Live options with their documented defaults."""
    options = dict(getattr(config, "provider_options", None) or {})
    unknown = set(options) - set(LIVE_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown Gemini Live provider_options: {sorted(unknown)}")
    resolved = {}
    for name, (default, allowed) in LIVE_OPTIONS.items():
        value = options.get(name, default)
        if value not in allowed:
            raise ValueError(f"Gemini Live {name} must be one of {list(allowed)}, got {value!r}")
        resolved[name] = value
    return resolved


class GoAwayDeadlineExceeded(RuntimeError):
    """An incomplete native turn must remain invalid, even without visible output."""


class GeminiLiveDriver:
    capabilities = RealtimeCapabilities(
        protocol_family="gemini_live",
        persistent_session=True,
        max_video_fps=1.0,
        terminal_output_usage=True,
        action_channel="function_call",
    )

    def __init__(self, config):
        from google import genai as genai
        from google.genai import errors as errors
        from google.genai import types as types
        from websockets.exceptions import ConnectionClosed as ConnectionClosed

        self.options = live_options(config)
        self._validate_model_contract(config)
        if config.api_key:
            keys = [config.api_key]
        elif config.api_key_env and config.api_key_env != "GEMINI_API_KEY":
            keys = [os.environ.get(config.api_key_env)]
        else:
            keys = [os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")]
            keys.extend(os.environ.get(f"GEMINI_API_KEY_{index}") for index in range(2, 10))
        self._api_keys = tuple(dict.fromkeys(key for key in keys if key))
        if not self._api_keys:
            raise RuntimeError(
                "Gemini Live needs GEMINI_API_KEY/GOOGLE_API_KEY or ModelConfig.api_key"
            )
        self.config = config
        self.types = types
        self._api_error_type = errors.APIError
        self._connection_closed_type = ConnectionClosed
        self._client_type = genai.Client
        self._clients = {}
        requested_slot_raw = os.environ.get("GEMINI_API_KEY_START_SLOT", "1")
        try:
            requested_slot = int(requested_slot_raw)
        except ValueError as exc:
            raise ValueError("GEMINI_API_KEY_START_SLOT must be a positive integer") from exc
        if requested_slot < 1:
            raise ValueError("GEMINI_API_KEY_START_SLOT must be a positive integer")
        self._api_key_requested_slot = requested_slot
        self._api_key_index = (requested_slot - 1) % len(self._api_keys)
        self._ws_connection = async_ws_connection(self, "gemini_live.ws", classify_live_message)
        self.client = self._client_for_slot(self._api_key_index)
        self._cm = None
        self._session = None
        self._activity_open = False
        self._last_video_sent_at: float | None = None
        self._resume_handle = None
        self._candidate_resume_handle = None
        self._resume_after_turn = False
        self._connect_metadata: dict = {}
        self._go_away_deadline: float | None = None
        self._last_turn_clean = False

    @staticmethod
    def _validate_model_contract(config) -> None:
        """Reject setup combinations that violate the selected Live model's API contract."""
        options = live_options(config)
        if (
            options["tool_response_scheduling"] == "silent"
            and options["tool_behavior"] != "non_blocking"
        ):
            raise ValueError("Silent Gemini tool responses require non-blocking tools")
        model_name = str(config.model_name).rsplit("/", 1)[-1]
        if model_name == "gemini-3.8-live" and config.reasoning_effort not in {None, "fixed"}:
            raise ValueError(
                "gemini-3.8-live requires omitting thinking_config; use reasoning_effort=fixed"
            )
        if model_name != _EXTENDED_THINKING_MODEL:
            return
        invalid = []
        if config.reasoning_effort not in {"low", "medium", "high"}:
            invalid.append("reasoning_effort must be low, medium, or high")
        if options["tool_behavior"] != "non_blocking":
            invalid.append("tool_behavior must be non_blocking")
        if options["completion_signal"] != "interaction_idle":
            invalid.append("completion_signal must be interaction_idle")
        if options["tool_response_scheduling"] == "silent":
            invalid.append("function scheduling is unsupported; use tool_response_scheduling=omit")
        if invalid:
            raise ValueError(f"{_EXTENDED_THINKING_MODEL} requires " + "; ".join(invalid))

    def _client_for_slot(self, index: int):
        client = self._clients.get(index)
        if client is None:
            # google-genai passes create_connection only to the WebSocket connect.
            http_options = {"async_client_args": {"create_connection": self._ws_connection}}
            if self.config.api_base_url:
                http_options["base_url"] = self.config.api_base_url
            client = self._client_type(api_key=self._api_keys[index], http_options=http_options)
            self._clients[index] = client
        return client

    def bind_trace(self, trace) -> None:
        self._trace = trace

    @staticmethod
    def _exception_chain(exc: BaseException):
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            yield current
            seen.add(id(current))
            current = current.__cause__ or current.__context__

    def _rate_limit_kind(self, exc: BaseException) -> str | None:
        for current in self._exception_chain(exc):
            code = int(getattr(current, "code", 0) or 0)
            status = str(getattr(current, "status", "") or "").strip().lower()
            detail = (
                str(getattr(current, "message", None) or getattr(current, "details", "") or "")
                .strip()
                .lower()
            )
            if isinstance(current, self._connection_closed_type):
                close = getattr(current, "rcvd", None) or getattr(current, "sent", None)
                code = int(getattr(close, "code", code) or 0)
                detail = str(getattr(close, "reason", detail) or "").strip().lower()
            if "exceeded your current quota" in detail or "quota exceeded" in detail:
                return QUOTA_EXHAUSTED
            if code == 429 or status == "resource_exhausted":
                return RATE_LIMITED
        return None

    def classify_transport_error(self, exc: BaseException) -> TransportFailure:
        """Classify a failed turn for the harness's shared retry policy."""
        if any(
            isinstance(current, GoAwayDeadlineExceeded) for current in self._exception_chain(exc)
        ):
            return TransportFailure(FATAL)
        limited = self._rate_limit_kind(exc)
        if limited is not None:
            return TransportFailure(limited)
        return TransportFailure(TRANSIENT if self._is_transient(exc) else FATAL)

    def _is_transient(self, exc: BaseException) -> bool:
        """Classify only connection failures that are safe for canonical replay.

        WebSocket close code 1007 normally covers invalid payloads and must not be retried
        indiscriminately. Gemini Live also uses it for the transient session-level
        ``Precondition check failed.`` close observed after an accepted setup/history write, so
        restrict that code to the exact provider reason.
        The unified runtime additionally requires that no text, native call, or audio was received.
        Usage must either be absent or a terminal interrupted turn proving exactly zero generated
        tokens.
        Generic ABORTED and service-unavailable closes are also transient candidates; other 1008
        policy violations and explicit GoAway deadline failures remain terminal.
        """
        for current in self._exception_chain(exc):
            if isinstance(current, EmptyInterruptedToolTurnError):
                return True
            if isinstance(current, self._connection_closed_type):
                close = getattr(current, "rcvd", None) or getattr(current, "sent", None)
                code = int(getattr(close, "code", 0) or 0)
                reason = str(getattr(close, "reason", "") or "").strip().lower()
                if code in {0, 1001, 1006, 1012, 1013}:
                    return True
                if code == 1007 and reason == "precondition check failed.":
                    return True
                if code == 1008 and reason == "the operation was aborted.":
                    return True
                if code == 1011 and (
                    "internal error" in reason or reason == "the service is currently unavailable."
                ):
                    return True
            if isinstance(current, (ConnectionError, TimeoutError)):
                return True
            if isinstance(current, self._api_error_type):
                code = int(getattr(current, "code", 0) or 0)
                detail = (
                    str(getattr(current, "message", None) or getattr(current, "details", "") or "")
                    .strip()
                    .lower()
                )
                if code == 1006:
                    return True
                if code == 1007 and detail == "precondition check failed.":
                    return True
                if code == 1008 and detail == "the operation was aborted.":
                    return True
                if code == 1011 and (
                    "internal error" in detail or detail == "the service is currently unavailable."
                ):
                    return True
        return False

    def _realtime_input_config(self):
        return self.types.RealtimeInputConfig(
            automatic_activity_detection=self.types.AutomaticActivityDetection(disabled=True),
            activity_handling=self.types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            turn_coverage=self.types.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
        )

    def _thinking_config(self):
        effort = self.config.reasoning_effort
        if effort is None:
            return None
        # Gemini 3.8 Live has a fixed interleaved-reasoning profile and rejects thinking_config.
        # The catalog's explicit `fixed` value keeps that provider behavior identity-bearing while
        # ensuring the field is omitted from session setup.
        if effort == "fixed":
            return None
        if effort == "none":
            return self.types.ThinkingConfig(thinking_budget=0)
        levels = {
            "minimal": self.types.ThinkingLevel.MINIMAL,
            "low": self.types.ThinkingLevel.LOW,
            "medium": self.types.ThinkingLevel.MEDIUM,
            "high": self.types.ThinkingLevel.HIGH,
        }
        try:
            level = levels[effort]
        except KeyError as exc:
            raise ValueError(
                "Gemini Live reasoning_effort must be one of "
                "none, fixed, minimal, low, medium, or high"
            ) from exc
        return self.types.ThinkingConfig(thinking_level=level)

    def _function_declarations(self):
        behavior = self.options["tool_behavior"]
        declarations = []
        for item in gui_function_declarations():
            values = dict(item)
            if behavior == "non_blocking":
                values["behavior"] = self.types.Behavior.NON_BLOCKING
            else:
                # 3.8 defaults to NON_BLOCKING on the server; select blocking explicitly.
                values["behavior"] = self.types.Behavior.BLOCKING
            declarations.append(self.types.FunctionDeclaration(**values))
        return declarations

    def _tool_response_options(self) -> dict:
        if self.options["tool_response_scheduling"] == "silent":
            return {
                "will_continue": False,
                "scheduling": self.types.FunctionResponseScheduling.SILENT,
            }
        return {}

    def _live_config(self, request: ConnectRequest, *, resume_handle=None, has_history=False):
        instruction = request.seed.system_instruction
        return self.types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=self.types.AudioTranscriptionConfig(),
            tools=[self.types.Tool(function_declarations=self._function_declarations())],
            thinking_config=self._thinking_config(),
            system_instruction=self.types.Content(parts=[self.types.Part(text=instruction)]),
            session_resumption=self.types.SessionResumptionConfig(handle=resume_handle),
            history_config=(
                self.types.HistoryConfig(initial_history_in_client_content=True)
                if has_history
                else None
            ),
            realtime_input_config=(
                self._realtime_input_config()
                if self.options["input_transport"] == "realtime_input"
                else None
            ),
        )

    def _history(self, request: ConnectRequest):
        contents = []
        if request.executed_action_ledger:
            contents.append(
                self.types.Content(
                    role="user",
                    parts=[
                        self.types.Part(
                            text=ARCHIVED_ACTION_LEDGER_HEADER
                            + "\n"
                            + request.executed_action_ledger
                        )
                    ],
                )
            )
        for index, turn in enumerate(request.turns):
            observation = turn.observation
            parts = [
                self.types.Part(
                    text=(
                        FOLDED_VISUAL_PLACEHOLDER
                        if turn.visual_folded
                        else session_frame_text(observation)
                    )
                )
            ]
            if not turn.visual_folded:
                parts.append(
                    self.types.Part(
                        inline_data=self.types.Blob(
                            mime_type=observation.mime_type,
                            data=observation.image_bytes,
                        )
                    )
                )
            contents.append(
                self.types.Content(
                    role="user",
                    parts=parts,
                )
            )
            native_exchanges = turn.provider_refs.get("native_tool_exchanges")
            initial_text = turn.provider_refs.get("native_initial_text")
            if initial_text:
                contents.append(
                    self.types.Content(
                        role="model",
                        parts=[self.types.Part(text=initial_text)],
                    )
                )
            if not isinstance(native_exchanges, list):
                native_call = turn.provider_refs.get("native_function_call")
                native_exchanges = (
                    [{"function_calls": [native_call]}] if isinstance(native_call, dict) else []
                )
            if native_exchanges and self.options["history_format"] == "tool_transcript":
                # Live tool messages do not expose their original thought signatures. Replaying
                # synthetic native calls can stall the next interaction even after setup succeeds.
                # Preserve the complete past exchange as data, with current actions still required
                # to use the declared native functions. Never replay it as a new tool request.
                contents.append(
                    self.types.Content(
                        role="model",
                        parts=[
                            self.types.Part(
                                text=GEMINI_TOOL_HISTORY_HEADER
                                + "\n"
                                + json.dumps(native_exchanges, ensure_ascii=False),
                            )
                        ],
                    )
                )
                continue
            replayed_native_exchange = False
            for exchange_index, exchange in enumerate(native_exchanges):
                if not isinstance(exchange, dict):
                    continue
                calls = exchange.get("function_calls")
                if not isinstance(calls, list):
                    continue
                valid_calls = [
                    call for call in calls if isinstance(call, dict) and call.get("name")
                ]
                if not valid_calls:
                    continue
                replayed_native_exchange = True
                model_parts = []
                response_parts = []
                stored_responses = exchange.get("function_responses")
                if not isinstance(stored_responses, list):
                    stored_responses = []
                responses_by_id = {
                    str(response.get("id") or ""): response
                    for response in stored_responses
                    if isinstance(response, dict)
                }
                for call_index, call in enumerate(valid_calls):
                    call_id = str(
                        call.get("id") or f"history-{index}-{exchange_index}-{call_index}"
                    )
                    name = str(call["name"])
                    arguments = call.get("arguments")
                    if not isinstance(arguments, dict):
                        arguments = {}
                    part_values = {
                        "function_call": self.types.FunctionCall(
                            id=call_id,
                            name=name,
                            args=arguments,
                        )
                    }
                    # Gemini 3 parallel calls carry the thought signature only on the first
                    # function-call part in one model Content.
                    if call_index == 0:
                        part_values["thought_signature"] = _REPLAY_THOUGHT_SIGNATURE
                    model_parts.append(self.types.Part(**part_values))
                    stored_response = responses_by_id.get(call_id, {})
                    response = stored_response.get("response")
                    if not isinstance(response, dict):
                        response = dict(GEMINI_QUEUED_TOOL_RESPONSE)
                    response_options = self._tool_response_options()
                    response_options.update(
                        {
                            key: stored_response[key]
                            for key in ("will_continue", "scheduling")
                            if key in stored_response
                        }
                    )
                    response_parts.append(
                        self.types.Part(
                            function_response=self.types.FunctionResponse(
                                id=call_id,
                                name=str(stored_response.get("name") or name),
                                response=response,
                                **response_options,
                            )
                        )
                    )
                # Rebuild every provider-native tool exchange, including a response that had
                # zero or multiple GUI calls and therefore became a benchmark WAIT. Replaying
                # only a selected call would change the model-visible trajectory after reconnect.
                contents.append(self.types.Content(role="model", parts=model_parts))
                contents.append(self.types.Content(role="user", parts=response_parts))
                if exchange.get("completion_text"):
                    contents.append(
                        self.types.Content(
                            role="model",
                            parts=[
                                self.types.Part(text=exchange["completion_text"]),
                            ],
                        )
                    )
            if replayed_native_exchange:
                continue
            if turn.output:
                native_call = turn.provider_refs.get("native_function_call")
                if not isinstance(native_call, dict):
                    try:
                        action = json.loads(turn.output)
                    except (TypeError, ValueError):
                        action = None
                    if isinstance(action, dict) and isinstance(action.get("action"), str):
                        native_call = {
                            "id": f"history-{index}",
                            "name": action["action"],
                            "arguments": {
                                key: value for key, value in action.items() if key != "action"
                            },
                        }
                if isinstance(native_call, dict) and native_call.get("name"):
                    call_id = str(native_call.get("id") or f"history-{index}")
                    name = str(native_call["name"])
                    arguments = native_call.get("arguments")
                    if not isinstance(arguments, dict):
                        arguments = {}
                    # Rebuild the provider-native tool exchange. Replaying the canonical action
                    # as ordinary model text makes Gemini switch to spoken JSON after the first
                    # visual-history fold, which also loses authoritative response-token usage.
                    contents.append(
                        self.types.Content(
                            role="model",
                            parts=[
                                self.types.Part(
                                    function_call=self.types.FunctionCall(
                                        id=call_id,
                                        name=name,
                                        args=arguments,
                                    ),
                                    # Pydantic intentionally preserves Google's documented literal
                                    # sentinel on the wire even though genuine signatures are bytes.
                                    thought_signature=_REPLAY_THOUGHT_SIGNATURE,
                                )
                            ],
                        )
                    )
                    contents.append(
                        self.types.Content(
                            role="user",
                            parts=[
                                self.types.Part(
                                    function_response=self.types.FunctionResponse(
                                        id=call_id,
                                        name=name,
                                        response=dict(GEMINI_QUEUED_TOOL_RESPONSE),
                                        **self._tool_response_options(),
                                    )
                                )
                            ],
                        )
                    )
                    continue
                contents.append(
                    self.types.Content(
                        role="model",
                        parts=[self.types.Part(text=turn.output)],
                    )
                )
        return contents

    @staticmethod
    def _history_batches(history: list, limit: int = _HISTORY_CONTENTS_PER_MESSAGE) -> list[list]:
        """Pack history without splitting one observation/model/tool-response exchange."""
        groups = []
        group = []
        for content in history:
            parts = getattr(content, "parts", None) or []
            is_function_response = any(
                getattr(part, "function_response", None) is not None for part in parts
            )
            has_function_call = any(
                getattr(part, "function_call", None) is not None for part in parts
            )
            if getattr(content, "role", None) == "user" and group and not is_function_response:
                groups.append(group)
                group = []
            group.append(content)
            if is_function_response or (
                getattr(content, "role", None) == "model" and not has_function_call
            ):
                groups.append(group)
                group = []
        if group:
            groups.append(group)

        batches = []
        batch = []
        for atomic_group in groups:
            if batch and len(batch) + len(atomic_group) > limit:
                batches.append(batch)
                batch = []
            batch.extend(atomic_group)
        if batch:
            batches.append(batch)
        return batches

    async def _close_current(self) -> None:
        cm, self._cm = self._cm, None
        self._session = None
        self._activity_open = False
        self._last_video_sent_at = None
        self._candidate_resume_handle = None
        self._go_away_deadline = None
        if cm is not None:
            try:
                await asyncio.wait_for(
                    cm.__aexit__(None, None, None),
                    _CONNECTION_CLOSE_TIMEOUT_S,
                )
            except Exception:  # noqa: BLE001, S110 - best-effort SDK cleanup
                pass

    async def connect(self, request: ConnectRequest) -> None:
        await self._connect_once(request)

    async def _connect_once(self, request: ConnectRequest) -> None:
        replacing_session = self._cm is not None
        await self._close_current()
        self._resume_handle = request.resume_handle
        history = [] if request.resume_handle is not None else self._history(request)
        history_batches = self._history_batches(history)
        if replacing_session or history:
            # Closing the local WebSocket and releasing the server-side Live session are not one
            # atomic operation. Apply the grace period to every reconnect/history replay, including
            # a retry whose failed predecessor no longer has a local context manager. This is
            # wall-clock-only and avoids briefly exceeding a one-session provider quota.
            await asyncio.sleep(_SESSION_RELEASE_GRACE_S)
        config = self._live_config(
            request,
            resume_handle=request.resume_handle,
            has_history=bool(history),
        )
        cm = self.client.aio.live.connect(model=request.seed.model_name, config=config)
        try:
            session = await cm.__aenter__()
            if getattr(session, "setup_complete", None) is None:
                raise RuntimeError("Gemini Live did not acknowledge session setup")
            if history:
                # Seed history explicitly before either input transport. Merely leaving
                # turn_complete=False can cause historical model content to be echoed as new
                # output. HistoryConfig makes this a non-generating initialization boundary.
                # Bound the messages so screenshots do not form one oversized WebSocket frame.
                for index, batch in enumerate(history_batches):
                    await session.send_client_content(
                        turns=batch,
                        turn_complete=index == len(history_batches) - 1,
                    )
        except BaseException as exc:
            try:
                await cm.__aexit__(type(exc), exc, exc.__traceback__)
            except Exception:  # noqa: BLE001, S110 - preserve the handshake failure
                pass
            raise
        self._cm = cm
        self._session = session
        self._activity_open = False
        self._last_video_sent_at = None
        self._resume_after_turn = False
        self._last_turn_clean = request.resume_handle is not None or bool(history)
        self._connect_metadata = {
            "session_connect_mode": (
                "session_resumption"
                if request.resume_handle is not None
                else "canonical_history_replay"
                if history
                else "new_session"
            ),
            "session_resume_handle_used": request.resume_handle is not None,
            "history_seed_contents": len(history),
            "history_seed_messages": len(history_batches),
            "session_replacement_grace_s": (
                _SESSION_RELEASE_GRACE_S if replacing_session or history else 0.0
            ),
            "api_key_slot": self._api_key_index + 1,
            "api_key_requested_slot": self._api_key_requested_slot,
            "api_key_pool_size": len(self._api_keys),
        }

    def connection_renewal_reason(self) -> str | None:
        # Let the provider-defined connection lifetime be observable. Reconnect only when Gemini
        # sends GoAway and supplies a resumption handle; there is no client-side age threshold.
        if self._resume_after_turn:
            return "go_away"
        return None

    async def _before_go_away_deadline(self, operation):
        """Bound receive and tool acknowledgements by the server's connection-close deadline."""
        if self._go_away_deadline is None:
            return await operation
        remaining = max(0.0, self._go_away_deadline - time.monotonic())
        try:
            return await asyncio.wait_for(operation, remaining)
        except asyncio.TimeoutError:
            await self._close_current()
            # A partial native turn cannot safely be replayed or charged as zero. This distinct
            # failure is deliberately non-retryable; the task must be rerun with complete usage.
            raise GoAwayDeadlineExceeded(
                "gemini_live_go_away_deadline_before_turn_complete"
            ) from None

    async def send_observation(self, observation: ProviderObservation) -> dict:
        """Send one decision's screenshot as one input activity."""
        if self._resume_after_turn:
            raise RuntimeError(
                "Gemini Live requires canonical history synchronization before input"
            )
        session = self._session
        if session is None:
            raise RuntimeError("Gemini Live session is not connected")
        if self._activity_open:
            raise RuntimeError("Gemini Live input activity is already open")
        self._last_turn_clean = False
        self._resume_handle = None
        self._candidate_resume_handle = None
        if self.options["input_transport"] == "client_content":
            # Send the screenshot with its grounding text in one Content that triggers generation
            # after any seeded history has been ingested. A realtime text trigger mixed with
            # synthetic silent audio can create a second user activity (including hallucinated
            # speech) and leave synchronous tools unfinished.
            await session.send_client_content(
                turns=self.types.Content(
                    role="user",
                    parts=[
                        self.types.Part(text=session_frame_text(observation)),
                        self.types.Part(
                            inline_data=self.types.Blob(
                                data=observation.image_bytes,
                                mime_type=observation.mime_type,
                            )
                        ),
                    ],
                ),
                turn_complete=True,
            )
            return {"observation_id": observation.observation_id}
        await session.send_realtime_input(activity_start=self.types.ActivityStart())
        self._activity_open = True
        try:
            loop = asyncio.get_running_loop()
            if self._last_video_sent_at is not None:
                await asyncio.sleep(max(0.0, 1.0 - (loop.time() - self._last_video_sent_at)))
            await session.send_realtime_input(
                video=self.types.Blob(
                    data=observation.image_bytes,
                    mime_type=observation.mime_type,
                )
            )
            self._last_video_sent_at = loop.time()
            # Let media ingestion catch up before the corresponding SoM text arrives.
            await asyncio.sleep(1.0)
            await session.send_realtime_input(text=session_frame_text(observation))
            # Native audio activity must contain audio. This fixed 100-ms silent carrier
            # has no task information; its input usage is audited, never output-charged.
            await session.send_realtime_input(
                audio=self.types.Blob(data=bytes(3200), mime_type="audio/pcm;rate=16000")
            )
            await session.send_realtime_input(activity_end=self.types.ActivityEnd())
            self._activity_open = False
        except BaseException:
            # An incomplete activity cannot be resumed as canonical context.
            self._activity_open = True
            raise
        return {"observation_id": observation.observation_id}

    async def start_generation(self, request: GenerationRequest) -> GenerationHandle:
        if self._session is None:
            raise RuntimeError("Gemini Live session is not connected")
        return GenerationHandle(
            request.generation_id,
            metadata={
                **self._connect_metadata,
                "input_audio_carrier_ms": (
                    100 if self.options["input_transport"] == "realtime_input" else 0
                ),
                "video_ingestion_wait_ms": (
                    1000 if self.options["input_transport"] == "realtime_input" else 0
                ),
                "gemini_live_input_transport": self.options["input_transport"],
                "gemini_live_tool_response_scheduling": self.options["tool_response_scheduling"],
                "gemini_live_history_format": self.options["history_format"],
                "gemini_live_tool_behavior": self.options["tool_behavior"],
                "gemini_live_completion_signal": self.options["completion_signal"],
                "completed_turn_refs": {
                    "native_function_calls": [],
                    "native_tool_exchanges": [],
                },
            },
        )

    @staticmethod
    def _usage_dict(usage) -> dict | None:
        """Normalize Live usage and exclude forced audio output from the token clock.

        Gemini 3.1 Live requires AUDIO response modality even when the semantic output is a
        function call. ``response_token_count`` includes any generated audio, while the benchmark
        charges only model deliberation and the non-audio function-call response. Keep the native
        totals alongside the normalized values so provider usage and billing remain auditable.
        """
        if usage is None:
            return None
        raw_usage = (
            usage if isinstance(usage, dict) else usage.model_dump(mode="json", exclude_none=True)
        )

        def member(value, snake_name: str, camel_name: str | None = None):
            if isinstance(value, dict):
                if snake_name in value:
                    return value.get(snake_name)
                return value.get(camel_name) if camel_name else None
            return getattr(value, snake_name, None)

        def value(name: str, camel_name: str | None = None) -> int:
            return int(member(usage, name, camel_name) or 0)

        details_value = member(usage, "response_tokens_details", "responseTokensDetails")
        details = details_value or []
        response_details = [
            detail if isinstance(detail, dict) else detail.model_dump(mode="json")
            for detail in details
        ]
        prompt_details_value = member(
            usage,
            "prompt_tokens_details",
            "promptTokensDetails",
        )
        prompt_details = prompt_details_value or []
        normalized_prompt_details = [
            detail if isinstance(detail, dict) else detail.model_dump(mode="json")
            for detail in prompt_details
        ]

        def modality(detail) -> str:
            raw = member(detail, "modality")
            raw = getattr(raw, "value", raw)
            return str(raw or "").rsplit(".", 1)[-1].upper()

        def detail_tokens(detail) -> int:
            return int(member(detail, "token_count", "tokenCount") or 0)

        def first_reported(*names: str):
            for name in names:
                if name in raw_usage and raw_usage.get(name) is not None:
                    return raw_usage.get(name), name
            return None, None

        response_value, response_source = first_reported(
            "response_token_count",
            "responseTokenCount",
            "candidates_token_count",
            "candidatesTokenCount",
        )
        if response_value is not None:
            native_response = int(response_value)
            native_response_exact = True
        elif details_value is not None:
            native_response = sum(detail_tokens(detail) for detail in details)
            native_response_exact = True
            response_source = "response_tokens_details"
        else:
            native_response = 0
            native_response_exact = False
        audio_response = sum(
            detail_tokens(detail) for detail in details if modality(detail) == "AUDIO"
        )
        native_total_value = member(usage, "total_token_count", "totalTokenCount")
        native_total = int(native_total_value or 0)
        prompt_value = member(usage, "prompt_token_count", "promptTokenCount")
        tool_prompt_value = member(
            usage,
            "tool_use_prompt_token_count",
            "toolUsePromptTokenCount",
        )
        reasoning_value = member(usage, "thoughts_token_count", "thoughtsTokenCount")
        reasoning = int(reasoning_value or 0)
        tool_prompt = int(tool_prompt_value or 0)
        if (
            not native_response_exact
            and native_total_value is not None
            and prompt_value is not None
        ):
            residual = native_total - int(prompt_value)
            # With non-negative usage components, a zero residual proves that response,
            # tool-use prompt, and thought counts are all zero even when protobuf omits those
            # optional zero-valued fields. For a positive residual, derive only when both other
            # components were explicitly reported so the response count remains exact.
            if residual == 0 or (
                residual >= 0 and tool_prompt_value is not None and reasoning_value is not None
            ):
                native_response = residual - tool_prompt - reasoning
                native_response_exact = native_response >= 0
                if native_response_exact:
                    response_source = "total_minus_prompt_tool_use_and_thoughts"
        if native_response < 0:
            raise RuntimeError("Gemini response token count is negative")
        if audio_response > native_response:
            raise RuntimeError("Gemini response AUDIO token detail exceeds response token count")
        chargeable_response = native_response - audio_response
        input_tokens = (
            int(prompt_value)
            if prompt_value is not None
            else max(0, native_total - native_response - reasoning - tool_prompt)
        )
        clock_output = chargeable_response + reasoning
        input_details = {
            "cached_tokens": value(
                "cached_content_token_count",
                "cachedContentTokenCount",
            ),
        }
        if prompt_details_value is not None:
            for name in ("TEXT", "IMAGE", "VIDEO", "AUDIO"):
                input_details[f"{name.lower()}_tokens"] = sum(
                    detail_tokens(detail) for detail in prompt_details if modality(detail) == name
                )
        return {
            "input_tokens": input_tokens,
            "output_tokens": clock_output,
            "visible_output_tokens": chargeable_response,
            "total_tokens": input_tokens + clock_output,
            "input_token_details": input_details,
            "output_token_details": {
                "reasoning_tokens": reasoning,
                "audio_tokens": audio_response,
            },
            "native_response_tokens": native_response,
            "native_response_tokens_source": response_source,
            "native_audio_response_tokens": audio_response,
            "native_audio_response_tokens_reported": any(
                modality(detail) == "AUDIO" for detail in details
            ),
            "native_non_audio_response_tokens": chargeable_response,
            "native_total_tokens": (native_total if native_total_value is not None else None),
            "native_total_tokens_reported": native_total_value is not None,
            "native_tool_use_prompt_tokens": tool_prompt,
            "output_accounting": "gemini_native_tool_turn_usage_excludes_audio_v2",
            "native_output_tokens_source": "gemini_turn_usage_metadata",
            "native_audio_tokens_source": "gemini_response_tokens_details",
            "raw_native_usage": raw_usage,
            "native_response_usage_reported": (
                raw_usage.get("response_token_count") is not None
                or raw_usage.get("responseTokenCount") is not None
                or raw_usage.get("candidates_token_count") is not None
                or raw_usage.get("candidatesTokenCount") is not None
                or bool(raw_usage.get("response_tokens_details"))
                or bool(raw_usage.get("responseTokensDetails"))
            ),
            "native_response_usage_exact": native_response_exact,
            "response_tokens_details": response_details,
            "prompt_tokens_details": normalized_prompt_details,
        }

    @staticmethod
    def _text_parts(message) -> list[str]:
        server = getattr(message, "server_content", None)
        transcript = getattr(server, "output_transcription", None)
        text = getattr(transcript, "text", None)
        return [text] if text else []

    @staticmethod
    def _interaction_status(message) -> str | None:
        # The current SDK exposes status on server_content. Accept the top-level spelling too so
        # the driver remains compatible with the wire shape documented for tool-call messages.
        status = getattr(message, "interaction_status", None)
        if status is None:
            status = getattr(getattr(message, "server_content", None), "interaction_status", None)
        status = getattr(status, "value", status)
        if status is None:
            return None
        return str(status).rsplit(".", 1)[-1].upper()

    def _interaction_complete(self, message) -> bool:
        if self.options["completion_signal"] == "interaction_idle":
            return self._interaction_status(message) == "IDLE"
        server = getattr(message, "server_content", None)
        return bool(server is not None and getattr(server, "turn_complete", False))

    async def _interaction_messages(self, session):
        """Span all utterances that belong to one model interaction.

        google-genai's ``session.receive()`` iterator ends at every ``turn_complete``. Extended
        Thinking uses that marker for intermediate spoken fillers, so open another receive iterator
        until the provider reports the interaction itself as IDLE.
        """
        while True:
            received = False
            messages = session.receive().__aiter__()
            while True:
                try:
                    message = await self._before_go_away_deadline(anext(messages))
                except StopAsyncIteration:
                    break
                received = True
                yield message
                if self._interaction_complete(message):
                    return
            if self.options["completion_signal"] != "interaction_idle":
                return
            if not received:
                raise RuntimeError(
                    "Gemini Live receive stream ended before interaction_status became IDLE"
                )

    def _capture_control(self, message, *, interaction_complete: bool) -> dict:
        metadata = {}
        update = getattr(message, "session_resumption_update", None)
        if update is not None:
            if not getattr(update, "resumable", False):
                self._candidate_resume_handle = None
            elif getattr(update, "new_handle", None):
                self._candidate_resume_handle = update.new_handle
        # Resumption updates often arrive before the final server-content message. Promote the
        # latest candidate only at a clean turn boundary; a handle captured mid-generation may not
        # represent the complete provider turn.
        if interaction_complete and self._candidate_resume_handle:
            self._resume_handle = self._candidate_resume_handle
        go_away = getattr(message, "go_away", None)
        if go_away is not None:
            self._resume_after_turn = True
            raw_time_left = getattr(go_away, "time_left", None)
            try:
                seconds = float(str(raw_time_left).removesuffix("s"))
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("invalid GoAway duration")
            except (TypeError, ValueError):
                # Missing or malformed provider deadlines cannot justify waiting indefinitely.
                seconds = 0.0
            deadline = time.monotonic() + max(0.0, seconds - _GO_AWAY_CLOSE_MARGIN_S)
            self._go_away_deadline = (
                deadline
                if self._go_away_deadline is None
                else min(deadline, self._go_away_deadline)
            )
            metadata.update(
                {
                    "reconnect_required": True,
                    "reconnect_reason": "go_away",
                    "go_away_time_left": getattr(go_away, "time_left", None),
                }
            )
        return metadata

    async def events(self, generation: GenerationHandle):
        session = self._session
        if session is None:
            raise RuntimeError("Gemini Live session is not connected")
        sequence = 0
        interrupted = False
        latest_usage = None
        tool_calls_seen = 0
        async for message in self._interaction_messages(session):
            raw_type = type(message).__name__
            server = getattr(message, "server_content", None)
            turn = getattr(server, "model_turn", None)
            parts = getattr(turn, "parts", None) or []
            if parts and not generation.metadata.get("native_output_received"):
                # Native output may precede its transcription and terminal usage. The
                # zero-output transport retry must not discard audio or untranscribed text.
                generation.metadata["native_output_received"] = True
                sequence += 1
                yield GenerationBoundary(
                    generation.generation_id,
                    sequence,
                    raw_type,
                    "started",
                    ProviderHistoryState.UNKNOWN,
                    {"native_output_received": True},
                )
            for part in parts:
                audio = getattr(part, "inline_data", None)
                if audio is not None and getattr(audio, "data", None):
                    generation.metadata["audio_bytes_received"] = generation.metadata.get(
                        "audio_bytes_received", 0
                    ) + len(audio.data)
            usage = self._usage_dict(getattr(message, "usage_metadata", None))
            if usage is not None:
                latest_usage = usage
                sequence += 1
                yield UsageSnapshot(
                    generation.generation_id,
                    sequence,
                    raw_type,
                    usage,
                )
            server = getattr(message, "server_content", None)
            interaction_status = self._interaction_status(message)
            interaction_complete = self._interaction_complete(message)
            if interaction_status is not None:
                generation.metadata["gemini_live_interaction_status"] = interaction_status
            message_interrupted = bool(server is not None and getattr(server, "interrupted", False))
            interrupted = interrupted or message_interrupted
            metadata = self._capture_control(
                message,
                interaction_complete=interaction_complete and not interrupted,
            )
            if metadata:
                generation.metadata.update(metadata)
                sequence += 1
                yield GenerationBoundary(
                    generation.generation_id,
                    sequence,
                    raw_type,
                    "go_away",
                    (
                        ProviderHistoryState.EXACT
                        if interaction_complete
                        else ProviderHistoryState.UNKNOWN
                    ),
                    metadata,
                )
            cancellation = getattr(message, "tool_call_cancellation", None)
            if cancellation is not None and getattr(cancellation, "ids", None):
                raise RuntimeError("Gemini cancelled a queued GUI function call")
            # A transcript delivered with a new call precedes that call's acknowledgement.
            # Keep it with the previous exchange (or initial model text), in provider order.
            for text in self._text_parts(message):
                refs = generation.metadata["completed_turn_refs"]
                exchanges = refs["native_tool_exchanges"]
                if exchanges:
                    exchange = exchanges[-1]
                    exchange["completion_text"] = exchange.get("completion_text", "") + text
                else:
                    refs["native_initial_text"] = refs.get("native_initial_text", "") + text
                sequence += 1
                yield TextDelta(
                    generation.generation_id,
                    sequence,
                    raw_type,
                    text,
                    "visible",
                )
            calls = getattr(getattr(message, "tool_call", None), "function_calls", None) or []
            if calls:
                tool_calls_seen += len(calls)
                call_records = []
                for call in calls:
                    arguments = dict(call.args or {})
                    call_record = {
                        "id": call.id or "",
                        "name": call.name or "",
                        "arguments": arguments,
                    }
                    call_records.append(call_record)
                    sequence += 1
                    yield FunctionCall(
                        generation.generation_id,
                        sequence,
                        raw_type,
                        call.id or "",
                        call.name or "",
                        arguments,
                    )
                function_responses = [
                    self.types.FunctionResponse(
                        id=call.id,
                        name=call.name,
                        response=dict(GEMINI_QUEUED_TOOL_RESPONSE),
                        **self._tool_response_options(),
                    )
                    for call in calls
                ]
                # Finalize the queue request, not GUI execution. SILENT on a non-blocking tool
                # prevents its acknowledgement from triggering another model generation while
                # the current decision drains through authoritative terminal usage.
                await self._before_go_away_deadline(
                    session.send_tool_response(function_responses=function_responses),
                )
                completed_refs = generation.metadata["completed_turn_refs"]
                completed_refs["native_function_calls"].extend(call_records)
                completed_refs["native_tool_exchanges"].append(
                    {
                        "function_calls": call_records,
                        "function_responses": [
                            response.model_dump(mode="json", exclude_none=True)
                            for response in function_responses
                        ],
                    }
                )
                if tool_calls_seen == 1:
                    completed_refs["native_function_call"] = call_records[0]
                else:
                    completed_refs.pop("native_function_call", None)
                generation.metadata["function_calls_seen"] = tool_calls_seen
                generation.metadata["tool_response_policy"] = (
                    "queue_acknowledgement_before_execution"
                )
            if message_interrupted:
                sequence += 1
                yield GenerationBoundary(
                    generation.generation_id,
                    sequence,
                    raw_type,
                    "interrupted",
                    ProviderHistoryState.CONTAMINATED,
                )
            if (
                server is not None
                and getattr(server, "generation_complete", False)
                and self.options["completion_signal"] == "turn_complete"
            ):
                sequence += 1
                yield GenerationBoundary(
                    generation.generation_id,
                    sequence,
                    raw_type,
                    "generation_complete",
                    (
                        ProviderHistoryState.CONTAMINATED
                        if interrupted
                        else ProviderHistoryState.UNKNOWN
                    ),
                )
            if interaction_complete:
                generation.metadata["clean_turn_complete"] = not interrupted
                self._last_turn_clean = not interrupted
                generation.metadata["function_calls_seen"] = tool_calls_seen
                if (
                    generation.metadata.get("audio_bytes_received", 0)
                    and latest_usage is not None
                    and not latest_usage["native_audio_response_tokens_reported"]
                ):
                    raise RuntimeError(
                        "Gemini returned audio without an AUDIO response token breakdown"
                    )
                sequence += 1
                yield GenerationBoundary(
                    generation.generation_id,
                    sequence,
                    raw_type,
                    "turn_complete",
                    (
                        ProviderHistoryState.CONTAMINATED
                        if interrupted
                        else ProviderHistoryState.EXACT
                    ),
                    {
                        **generation.metadata,
                        "output_modality": "AUDIO",
                        "terminal_usage_received": usage is not None,
                        "action_text_source": "native_function_call",
                        "submission_boundary": (
                            "native_interaction_idle"
                            if self.options["completion_signal"] == "interaction_idle"
                            else "native_turn_complete"
                        ),
                    },
                )
                return

    async def interrupt(self, generation: GenerationHandle) -> InterruptResult:
        if self._session is None or self._activity_open:
            return InterruptResult(ProviderHistoryState.UNKNOWN)
        if self.options["input_transport"] == "client_content":
            # This transport does not opt into manual audio activity detection. Closing the
            # connection cancels the incomplete turn without injecting an invalid ActivityStart.
            await self._close_current()
            return InterruptResult(
                ProviderHistoryState.CONTAMINATED,
                {"interrupt_method": "connection_close"},
            )
        try:
            await self._session.send_realtime_input(activity_start=self.types.ActivityStart())
            self._activity_open = True
            self._resume_handle = None
            return InterruptResult(ProviderHistoryState.CONTAMINATED)
        except Exception as exc:  # noqa: BLE001 - provider control boundary
            return InterruptResult(
                ProviderHistoryState.UNKNOWN,
                {"interrupt_error": repr(exc)[:300]},
            )

    async def sync_history(self, request: HistorySyncRequest) -> HistorySyncResult:
        resume_error = None
        try:
            can_resume = bool(
                request.reason == "go_away"
                and not request.force_rebuild
                and self._resume_handle
                and self._last_turn_clean
                and (
                    request.current_generation is None
                    or request.current_generation.metadata.get("clean_turn_complete")
                )
            )
            if can_resume:
                handle = self._resume_handle
                try:
                    await self._connect_once(
                        ConnectRequest(
                            seed=request.seed,
                            resume_handle=handle,
                        )
                    )
                    return HistorySyncResult(
                        SyncDisposition.EXACT,
                        ProviderHistoryState.EXACT,
                        detail=f"resumed clean Gemini Live session: {request.reason}",
                        provider_metadata={
                            **self._connect_metadata,
                            "session_resumed": True,
                            "history_session_reconnected": True,
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - fall through to canonical replay
                    resume_error = repr(exc)[:300]
            # Renew and replay with the same task credential, including resume failure fallback.
            await self._connect_once(
                ConnectRequest(
                    seed=request.seed,
                    turns=request.turns,
                    executed_action_ledger=request.executed_action_ledger,
                )
            )
            return HistorySyncResult(
                SyncDisposition.REBUILT,
                ProviderHistoryState.EXACT,
                detail=request.reason,
                provider_metadata={
                    **self._connect_metadata,
                    "history_session_reconnected": True,
                    **({"resume_error": resume_error} if resume_error else {}),
                },
            )
        except Exception as exc:
            # Preserve typed transport failures so the outer zero-output retry gate can safely
            # rebuild the session. Converting them to a string-only BROKEN result would discard the
            # WebSocket close code and make a later retry impossible to classify.
            if self.classify_transport_error(exc).retryable:
                raise
            return HistorySyncResult(
                SyncDisposition.BROKEN,
                ProviderHistoryState.UNKNOWN,
                detail=repr(exc)[:500],
            )

    async def close(self) -> None:
        await self._close_current()
