from __future__ import annotations

# Context policy
"""What each model API is shown at every step, and where: twelve named items and a version.

Version 1 copied what each code path did before the policy existed; every later version makes
one change (see CONTEXT_POLICY_VERSION), to one item, except version 2, which changed generic
LiteLLM's reasoning and verbatim replay (items 6 and 7) together. A change that alters what some
model is shown bumps the version (in phase 2 once, at its release: see CONTEXT_POLICY_VERSION);
moving a preset from an API row to the model that needs it alters nothing shown and keeps it.
The values count in the effective run config and therefore in its hash. Every item but verbatim
replay (item 7), which is off everywhere and only recorded, is read from here: retention,
feedback placement, the ledger role, reasoning replay, pressure archiving, the Decision history
form, the tool-call replay format, the prompts (item 11) and the observation form (item 12); the
encoders take their fixed texts (item 5) from here, and ContextPolicy.for_path takes Gemini's
tool-call replay format from the model's provider_options.

An API row takes the standard values, then the items of its preset, if it has one (PRESETS),
then its own. A preset comes from one of two places. A row's applies to every model of its API:
Luna's (openai_decisions) and JEV's (jev_decisions). A model config's (ModelConfig.context_preset,
from its YAML, one of MODEL_PRESETS) applies to that model alone and takes the place of its row's:
Kimi K3's, so other moonshot models keep the standard values. The pre-run checks hold each
resolved policy against what the API declares (providers/preflight.py check_context_policy).

Two values say what the policy allows, not what a run does. Pressure archiving (item 8) is on for
every API, and it acts only where the encoder estimates its requests and the model config has a
context window (ModelConfig.context_window_tokens). Luna's encoder, which has no roles, is given
the ledger role (item 4) but places the ledger by position and header; JEV's, which has none
either, labels the ledger's entry with it (providers/decisions/history.py JEV_ROLES). Whether
either takes effect is not recorded on its own: the driver code decides it, which the harness
fingerprint (or an external driver's identity) covers, and so does the model config, which the
run hash covers.
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

# 2: generic LiteLLM rebuilds its replies from the model output and replays no reasoning; Kimi
#    K3 keeps its reasoning through its preset.
# 3: the action ledger is an input wherever there are roles; OpenAI Realtime sent it as an
#    instruction (a system item), and Decisions state places it by position and header.
# 4: the fold placeholder says only what it removed; it claimed that the step's reasoning and
#    action remain.
# 5: one pressure-archive rule for every API that declares a request estimate; JEV estimates
#    its digest against its declared window instead of dropping history for a rejected
#    one-option question.
# 6: text-JSON prompts: "All generated output tokens count." only under a finite B, and frame
#    texts that stay true on older frames.
# Since 6, phase 2 changes what models are shown without bumping the version; its release bumps
# it once for all of them (decision 28). They include: a request still over the window with
# every turn archived leaves out the earliest archived actions, a note in their place, and JEV's
# ledger header then says every listed action is kept; Gemini Live and Luna frames name their own
# screen, as the text-JSON frames do, Luna's in its frame text as well; the clock note names the
# environment ticks when the two frame rates differ; the notes on a done run as a wait and on a
# decision with no accepted answer;
# OpenAI Realtime and Gemini Live send each action's execution feedback as its own item, not in
# front of the next frame; the streaming harness's feedback says that a superseded action was
# replaced while it waited, why a cancelled call did not run, and that a reply's protocol failure
# leaves each action it handed over its own feedback (harnesses/streaming/feedback.py); the SoM
# mark badges of a streaming screenshot keep clear of its frame number; an action naming a mark
# its frame does not have is a grounding failure, whose feedback names the mark and the action
# asked for, not output without a valid action; the function-call output contract charges token
# usage to simulated time only under a finite B, as the text-JSON one does.
CONTEXT_POLICY_VERSION = 6

# Fixed model-visible texts the encoders write (item 5).
LEDGER_OMISSION = "[{count} earlier archived actions were omitted to fit the context window.]"
JEV_LEDGER_HEADER = (
    "ARCHIVED EXECUTED ACTIONS (JEV lossless consecutive runs; repeat_count and "
    "occurrences preserve every action; start/step/count expands exact indices):"
)
# Once the earliest actions are left out (LEDGER_OMISSION follows it).
JEV_LEDGER_HEADER_OMITTED = (
    "ARCHIVED EXECUTED ACTIONS (JEV lossless consecutive runs; repeat_count and "
    "occurrences preserve every listed action; start/step/count expands exact indices):"
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
    # screenshots: fold block; turns: archive block, also under context pressure, and the block
    # of archived actions a request that still does not fit leaves out
    block: int


@dataclass(frozen=True)
class DecisionHistory:
    form: Literal["turns", "action_digest"]


@dataclass(frozen=True)
class FixedTexts:
    folded_visual: str
    ledger_header: str
    ledger_omission: str  # the ledger's first line once actions were left out; {count} of them
    jev_ledger_header: str
    jev_ledger_header_omitted: str
    jev_history_prefix: str
    jev_current_prefix: str
    luna_action_label: str
    luna_observation_label: str
    gemini_tool_history_header: str
    gemini_queued_tool_response: Mapping[str, str]


FIXED_TEXTS = FixedTexts(
    folded_visual=FOLDED_VISUAL_PLACEHOLDER,
    ledger_header=ARCHIVED_ACTION_LEDGER_HEADER,
    ledger_omission=LEDGER_OMISSION,
    jev_ledger_header=JEV_LEDGER_HEADER,
    jev_ledger_header_omitted=JEV_LEDGER_HEADER_OMITTED,
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
    preset: str | None  # the PRESETS entry in effect: the model config's, else the row's
    visuals: Retention  # 1 screenshot retention
    turns: Retention  # 2 turn retention; archived actions form the ledger
    # 3 execution feedback; under the action digest (item 9) every turn's feedback is in the
    # digest, whatever the value
    feedback: Literal["separate_message", "digest"]
    # 4 role of the archived-action ledger; an encoder without roles places it by position and
    # header (RealtimeCapabilities.labels_instead_of_roles), and the value is at most a label
    # there (providers/decisions/history.py JEV_ROLES)
    ledger_role: Literal["instruction", "input"]
    texts: FixedTexts  # 5 placeholder and fixed texts
    reasoning_replay: bool  # 6 send past reasoning back
    # 7 send the provider's own assistant message back; off: the reply is rebuilt from the model
    # output, and no encoder can do otherwise, so nothing reads the value
    verbatim_replay: bool
    # 8 before each request, archive the oldest turns while it exceeds the declared context
    # window, then, with no turn left, leave out the earliest archived actions, which stay out
    # for the rest of the episode as archived turns do. On allows it; it acts only where the API
    # declares an estimate of its requests (RealtimeCapabilities.request_size_estimation) and
    # the model config has a window
    pressure_archive: bool
    # 9 Decision history form; the pre-run checks take only the form the API replays
    # (RealtimeCapabilities.history_replay)
    decision_history: DecisionHistory | None
    tool_call_replay: Literal["native_tools", "tool_transcript"] | None  # 10 past live tool calls
    # 11 task category -> InteractionProfile.name; None: no profile for the category. The pre-run
    # checks reject the one such pairing, a text-only choice model on a browser game, from the
    # capabilities (providers/preflight.py check_choice_task), and take only profiles that ask for
    # the API's action channel (PROMPT_CHANNELS)
    prompts: Mapping[str, str | None]
    # 12 what one observation carries; the pre-run checks take only the one the API's observation
    # modality and action channel allow
    observation: Literal["som_screenshot", "observation_json_screenshot", "observation_json_text"]

    def to_record(self) -> dict:
        return _jsonable(self)


# The row an external driver ("package.module:Class") uses: the plain stateless chat values.
EXTERNAL_DRIVER_ROW = "chat_completions"


@dataclass(frozen=True)
class ContextPolicy:
    version: int
    paths: Mapping[str, PathPolicy]  # one per registered API
    own_values: Mapping[str, Mapping]  # each row's own values, which win over any preset

    def for_path(
        self, adapter_type: str, provider_options: Mapping, context_preset: str | None
    ) -> PathPolicy:
        """Resolve one API's policy for a model config, before any driver exists.

        A registered API has its own row; an external driver gets EXTERNAL_DRIVER_ROW. Registered
        names contain no colon and external ones are "package.module:Class"
        (providers/registry.py), so a name with no row and no colon is a registered API whose row
        is missing. The model config's preset (ModelConfig.context_preset) takes the place of the
        row's: the standard values, then the preset's items, then the row's own.
        """
        if adapter_type in self.paths:
            path = self.paths[adapter_type]
        elif ":" in adapter_type:
            path = self.paths[EXTERNAL_DRIVER_ROW]
        else:
            raise ValueError(
                f"API {adapter_type!r} has no context-policy row; add one in "
                "current_context_policy() (realtime_framework/entity/context_policy.py)"
            )
        if context_preset is not None:
            path = replace(
                path,
                preset=context_preset,
                **{**STANDARD, **PRESETS[context_preset], **self.own_values[path.api]},
            )
        if path.tool_call_replay is None:
            return path
        # Gemini Live takes the format from the model YAML's provider_options; a model that sets
        # none keeps the row's.
        return replace(
            path,
            tool_call_replay=provider_options.get("history_format", path.tool_call_replay),
        )


# The action channel (RealtimeCapabilities.action_channel) each profile item 11 names asks the
# model to answer in (harnesses/default/harness.py INTERACTION_PROFILES).
PROMPT_CHANNELS = {
    "standard_vlm": "text_json",
    "function_call": "function_call",
    "jev_decisions": "choice",
    "multimodal_decisions": "choice",
    "multimodal_game_decisions": "choice",
}
_TEXT_JSON_PROMPTS = {"realtime": "standard_vlm", "browser_game": "standard_vlm"}
# The standard values of the items that come neither from the environment (items 1 and 2) nor
# from FIXED_TEXTS (item 5).
STANDARD = {
    "feedback": "separate_message",
    "ledger_role": "input",
    "reasoning_replay": False,
    "verbatim_replay": False,
    "pressure_archive": True,
    "decision_history": None,
    "tool_call_replay": None,
    "prompts": _TEXT_JSON_PROMPTS,
    "observation": "som_screenshot",
}
# What a model takes instead of the standard values, by preset name: through its API row (Luna,
# JEV) or through its model config (Kimi K3).
PRESETS = {
    # Luna's candidates are listed by the harness, so it is not compared with the chat models:
    # it keeps the observation JSON beside the screenshot.
    "luna": {"observation": "observation_json_screenshot"},
    # JEV takes text only, within a 32k prompt: an action digest with the feedback in it, and
    # the observation JSON without a screenshot.
    "jev": {
        "feedback": "digest",
        "decision_history": DecisionHistory("action_digest"),
        "observation": "observation_json_text",
    },
    # Kimi's thinking-model guide requires K3's past reasoning_content in multi-turn
    # conversations. The requirement is K3's, so K3's YAML names this preset
    # (configs/models/kimi-k3.yaml), on moonshot or behind a chat_completions endpoint, and
    # other models of either API send no reasoning back.
    "kimi_k3": {"reasoning_replay": True},
}
# The presets a model config may name; Luna's and JEV's belong to their API rows.
MODEL_PRESETS = ("kimi_k3",)


def current_context_policy() -> ContextPolicy:
    """The values at CONTEXT_POLICY_VERSION; history windows keep their STREAM_* environment
    overrides."""
    visuals = Retention(*effective_visual_fold_policy())
    turns = Retention(*effective_detailed_history_policy())
    own_values = {}

    def path(api, preset=None, **values) -> PathPolicy:
        own_values[api] = values
        return PathPolicy(
            version=CONTEXT_POLICY_VERSION,
            api=api,
            preset=preset,
            visuals=visuals,
            turns=turns,
            texts=FIXED_TEXTS,
            **{**STANDARD, **(PRESETS[preset] if preset else {}), **values},
        )

    paths = (
        path("litellm"),
        path("chat_completions"),
        path("openrouter"),
        path("anthropic_messages"),
        path("moonshot"),
        path("openai_realtime"),
        path(
            "gemini_live",
            tool_call_replay="native_tools",
            prompts={"realtime": "function_call", "browser_game": "function_call"},
        ),
        path(
            "openai_decisions",
            "luna",
            decision_history=DecisionHistory("turns"),
            prompts={
                "realtime": "multimodal_decisions",
                "browser_game": "multimodal_game_decisions",
            },
        ),
        path(
            "jev_decisions",
            "jev",
            prompts={"realtime": "jev_decisions", "browser_game": None},
        ),
    )
    return ContextPolicy(CONTEXT_POLICY_VERSION, {value.api: value for value in paths}, own_values)
