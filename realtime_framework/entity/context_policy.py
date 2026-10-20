from __future__ import annotations

# Context policy
"""What each model API is shown at every step, and where: twelve named items and a version.

The phase-1 values copy what each code path did before the policy existed; changing any value
changes the model-visible content, so it bumps the version. The values count in the effective run
config and therefore in its hash. Retention, feedback placement, reasoning and verbatim replay,
pressure archiving and the Decision history form are read from here, and the encoders take their
fixed texts from here. The ledger role, Decision overflow, tool-call replay format, prompts and
observation form are recorded and pinned by tests until the encoders and the Decision agent read
them (phase 1 steps 4 and 6).
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from realtime_framework.entity.config import (
    effective_detailed_history_policy,
    effective_visual_fold_policy,
)
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER, FOLDED_VISUAL_PLACEHOLDER
from realtime_framework.entity.records import _jsonable

CONTEXT_POLICY_VERSION = 1

# Fixed model-visible texts the encoders write (item 5).
JEV_LEDGER_HEADER = (
    "ARCHIVED EXECUTED ACTIONS (JEV lossless consecutive runs; repeat_count and "
    "occurrences preserve every action; start/step/count expands exact indices):"
)
JEV_HISTORY_PREFIX = "Recent action and feedback history: "
JEV_CURRENT_PREFIX = "Current observation: "
LUNA_ACTION_LABEL = "Chosen action: "
LUNA_OBSERVATION_LABEL = "Observation: "
GEMINI_TOOL_HISTORY_HEADER = (
    "Historical GUI tool exchanges, already acknowledged "
    "(current actions must use native functions):"
)
GEMINI_QUEUED_TOOL_RESPONSE = {
    "status": "queued_not_executed",
    "instruction": (
        "End this turn silently. Await the next observation for the actual execution result."
    ),
}


@dataclass(frozen=True)
class Retention:
    keep: int  # screenshots: active images; turns: verbatim turns
    block: int  # screenshots: fold block; turns: archive block, also under context pressure


@dataclass(frozen=True)
class DecisionHistory:
    form: Literal["turns", "action_digest"]
    # What a Decision question does when the server rejects even one option for context.
    overflow: Literal["raise", "drop_oldest_half"]


@dataclass(frozen=True)
class FixedTexts:
    folded_visual: str
    ledger_header: str
    jev_ledger_header: str
    jev_history_prefix: str
    jev_current_prefix: str
    luna_action_label: str
    luna_observation_label: str
    gemini_tool_history_header: str
    gemini_queued_tool_response: Mapping[str, str]


PHASE1_TEXTS = FixedTexts(
    folded_visual=FOLDED_VISUAL_PLACEHOLDER,
    ledger_header=ARCHIVED_ACTION_LEDGER_HEADER,
    jev_ledger_header=JEV_LEDGER_HEADER,
    jev_history_prefix=JEV_HISTORY_PREFIX,
    jev_current_prefix=JEV_CURRENT_PREFIX,
    luna_action_label=LUNA_ACTION_LABEL,
    luna_observation_label=LUNA_OBSERVATION_LABEL,
    gemini_tool_history_header=GEMINI_TOOL_HISTORY_HEADER,
    gemini_queued_tool_response=GEMINI_QUEUED_TOOL_RESPONSE,
)


@dataclass(frozen=True)
class PathPolicy:
    """The policy as one model API applies it."""

    version: int
    api: str
    visuals: Retention  # 1 screenshot retention
    turns: Retention  # 2 turn retention; archived actions form the ledger
    feedback: Literal["separate_message", "frame_prefix", "digest"]  # 3 execution feedback
    ledger_role: Literal["instruction", "input", "none"]  # 4 role of the archived-action ledger
    texts: FixedTexts  # 5 placeholder and fixed texts
    reasoning_replay: bool  # 6 send past reasoning back
    verbatim_replay: bool  # 7 send the provider's own assistant message back
    pressure_archive: bool  # 8 archive more turns when the request exceeds the context
    decision_history: DecisionHistory | None  # 9 Decision history form and overflow
    tool_call_replay: Literal["native_tools", "tool_transcript"] | None  # 10 past live tool calls
    # 11 task category -> InteractionProfile.name; None: the category is rejected
    prompts: Mapping[str, str | None]
    # 12 what one observation carries
    observation: Literal["som_screenshot", "observation_json_screenshot", "observation_json_text"]

    def to_record(self) -> dict:
        return _jsonable(self)


@dataclass(frozen=True)
class ContextPolicy:
    version: int
    paths: Mapping[str, PathPolicy]  # one per registered API

    def for_path(self, adapter_type: str, provider_options: Mapping) -> PathPolicy:
        """Resolve one API's policy for a model config, before any driver exists."""
        path = self.paths[adapter_type]
        if path.tool_call_replay is None:
            return path
        # Gemini Live takes the format from its maintained model profile (providers/registry.py
        # MODEL_PROFILES); other models keep the driver default (gemini/live.py LIVE_OPTIONS).
        return replace(
            path,
            tool_call_replay=provider_options.get("history_format", path.tool_call_replay),
        )


_TEXT_JSON_PROMPTS = {"realtime": "standard_vlm", "browser_game": "standard_vlm"}


def phase1_context_policy() -> ContextPolicy:
    """The phase-1 values; history windows keep their STREAM_* environment overrides."""
    visuals = Retention(*effective_visual_fold_policy())
    turns = Retention(*effective_detailed_history_policy())

    def path(api, **values) -> PathPolicy:
        defaults = {
            "feedback": "separate_message",
            "ledger_role": "input",
            "reasoning_replay": False,
            "verbatim_replay": False,
            "pressure_archive": False,
            "decision_history": None,
            "tool_call_replay": None,
            "prompts": _TEXT_JSON_PROMPTS,
            "observation": "som_screenshot",
        }
        return PathPolicy(
            version=CONTEXT_POLICY_VERSION,
            api=api,
            visuals=visuals,
            turns=turns,
            texts=PHASE1_TEXTS,
            **{**defaults, **values},
        )

    paths = (
        path("litellm", reasoning_replay=True, verbatim_replay=True),
        path("chat_completions", pressure_archive=True),
        path("openrouter"),
        path("anthropic_messages"),
        path("moonshot", reasoning_replay=True),
        path("openai_realtime", feedback="frame_prefix", ledger_role="instruction"),
        path(
            "gemini_live",
            feedback="frame_prefix",
            tool_call_replay="native_tools",
            prompts={"realtime": "function_call", "browser_game": "function_call"},
        ),
        path(
            "openai_decisions",
            ledger_role="none",
            pressure_archive=True,
            decision_history=DecisionHistory("turns", "raise"),
            prompts={
                "realtime": "multimodal_decisions",
                "browser_game": "multimodal_game_decisions",
            },
            observation="observation_json_screenshot",
        ),
        path(
            "jev_decisions",
            feedback="digest",
            decision_history=DecisionHistory("action_digest", "drop_oldest_half"),
            prompts={"realtime": "jev_decisions", "browser_game": None},
            observation="observation_json_text",
        ),
    )
    return ContextPolicy(CONTEXT_POLICY_VERSION, {value.api: value for value in paths})
