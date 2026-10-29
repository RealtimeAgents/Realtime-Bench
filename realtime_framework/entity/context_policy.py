from __future__ import annotations

# Context policy
"""What each model API is shown at every step, and where: eleven named items and a version.

Version 1 copied what each code path did before the policy existed; every later version makes
one change (see CONTEXT_POLICY_VERSION), to one item, except version 2, which changed generic
LiteLLM's reasoning and verbatim replay (items 6 and 7) together. Item 7, which no encoder could
apply, has since been dropped. A change that alters what some model is shown bumps the version
(in phase 2 once, at its release: see CONTEXT_POLICY_VERSION); moving a preset from an API row
to the model that needs it alters nothing shown and keeps it. The values count in the effective
run config and therefore in its hash. Every item is read from here: retention, feedback
placement, the ledger role, reasoning replay, pressure archiving, the Decision history form, the
tool-call replay format, the prompts (item 11) and the observation form (item 12); the encoders
take their fixed texts (item 5) from here, and ContextPolicy.for_path takes Gemini's tool-call
replay format from the model's provider_options.

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

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import yaml

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
# usage to simulated time only under a finite B, as the text-JSON one does; a native-streaming
# run's instructions, function declarations, frame texts and SoM legends read an action's marks
# and coordinates from the frame it names, its frames name no decision, its clock note counts
# replies, and nothing is executed, not WAIT, for a reply without a valid action; a turn's system
# instruction asks for the newest screenshot's marks (P6), and the frame texts and SoM legends of
# every path say only what holds for their own screenshot, not that it is this decision's or that
# its marks are the current ones, the only ones to use; a run's instructions and function
# declarations describe only the clock it is on: on the wall clock a hold or wait lasts physical
# elapsed time and the function-call output contract names no token clock, and on the token
# clock that contract names no wall clock.
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


# Context settings
# The configurable items, their kinds of value, the API rows they apply to (None: every row),
# what kind of setting each is and whether a model's YAML may set it. Their values come from the
# defaults file alone (DEFAULTS_FILE): the standard of this harness version.
@dataclass(frozen=True)
class Setting:
    # int: a positive count; bool; or the strings allowed
    values: type | tuple[str, ...]
    applies: tuple[str, ...] | None
    # condition: an evaluation condition, the same for every model; protocol: what a model's
    # protocol may need (an approved deviation); capability: what an API's channel can render
    kind: Literal["condition", "protocol", "capability"]
    model: bool  # a model's YAML may set it
    about: str


SETTINGS = {
    "screenshots_kept": Setting(
        int, None, "condition", False,
        "Screenshots kept in the history; one more folds the oldest screenshot_fold_block",
    ),
    "screenshot_fold_block": Setting(
        int, None, "condition", False, "Screenshots folded together into a fixed text",
    ),
    "turns_kept": Setting(
        int, None, "condition", False,
        "Turns kept verbatim; one more archives the oldest turn_archive_block into the ledger",
    ),
    "turn_archive_block": Setting(
        int, None, "condition", False,
        "Turns archived together, also under context pressure, and archived actions left out "
        "together from a request that still does not fit",
    ),
    "execution_feedback": Setting(
        ("separate_message", "digest"), None, "capability", True,
        "Where each action's execution feedback goes: its own message, or the action digest",
    ),
    "observation": Setting(
        ("som_screenshot", "observation_json_screenshot", "observation_json_text"),
        None, "capability", True,
        "What one observation carries: the SoM screenshot, or the observation JSON with or "
        "without the screenshot",
    ),
    "reasoning_replay": Setting(
        bool, None, "protocol", True, "Send the model's past reasoning back in later requests",
    ),
    "archive_over_window": Setting(
        bool, None, "condition", False,
        "Archive the oldest turns while a request exceeds the declared context window",
    ),
    "tool_call_history": Setting(
        ("native_tools", "tool_transcript"), ("gemini_live",), "protocol", True,
        "How a rebuilt live session replays past tool calls: as native calls or as a transcript",
    ),
    "decision_history": Setting(
        ("turns", "action_digest"), ("openai_decisions", "jev_decisions"), "capability", True,
        "How a Decision request replays history: as turns or as an action digest",
    ),
}  # fmt: skip
DEFAULTS_FILE = Path(__file__).resolve().parents[1] / "context_defaults.yaml"
# Each block against the count it folds or archives from.
_BLOCKS = (("screenshot_fold_block", "screenshots_kept"), ("turn_archive_block", "turns_kept"))


def setting_error(key: str, value) -> str | None:
    """What is wrong with ``value`` for setting ``key``, if anything."""
    allowed = SETTINGS[key].values
    if allowed is int:
        if type(value) is not int or value <= 0:
            return f"{key} must be a positive integer, got {value!r}"
    elif allowed is bool:
        if type(value) is not bool:
            return f"{key} must be true or false, got {value!r}"
    elif type(value) is not str or value not in allowed:
        return f"{key} must be one of {list(allowed)}, got {value!r}"
    return None


def blocks_error(values: Mapping) -> str | None:
    """A block larger than the count it folds or archives from, if any."""
    for block, kept in _BLOCKS:
        if values[block] > values[kept]:
            return f"{block} ({values[block]}) must be at most {kept} ({values[kept]})"
    return None


def load_context_defaults(path) -> dict:
    """The defaults file at ``path``, checked: every setting's standard value, and nothing else."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"The context defaults file is missing: {path}")
    data = yaml.safe_load(path.read_text())
    name = path.name
    if not isinstance(data, dict) or not isinstance(data.get("standard"), dict):
        raise ValueError(  # noqa: TRY004 - a malformed file, as the catalog's YAMLs
            f"{name}: the file needs a standard section, a mapping of the settings"
        )
    others = sorted(set(data) - {"standard"})
    if others:
        raise ValueError(f"{name}: the file holds standard alone; got {others}")
    standard = data["standard"]
    for key in SETTINGS:
        if key not in standard:
            raise ValueError(f"{name}: standard lacks {key}")
    unknown = sorted(set(standard) - set(SETTINGS))
    if unknown:
        raise ValueError(f"{name}: standard has unknown settings {unknown}")
    for key in SETTINGS:
        error = setting_error(key, standard[key])
        if error:
            raise ValueError(f"{name}: standard {error}")
    error = blocks_error(standard)
    if error:
        raise ValueError(f"{name}: standard {error}")
    return {"standard": MappingProxyType({key: standard[key] for key in SETTINGS})}


@lru_cache(maxsize=1)
def context_defaults() -> dict:
    """The package's defaults file (DEFAULTS_FILE), read once per process."""
    return load_context_defaults(DEFAULTS_FILE)


def effective_detailed_history_policy() -> tuple[int, int]:
    """Return the verbatim text-history window and block-archive size.

    At ``window + 1`` turns, the oldest ``drop_block_size`` complete detailed turns are archived
    together. With the maintained defaults, verbatim text/action context therefore cycles between
    41 and 50 turns while older executed actions remain available through the compact ledger,
    unless a request that does not fit leaves the earliest of them out.
    """
    standard = context_defaults()["standard"]
    window = int(os.environ.get("STREAM_DETAILED_HISTORY_TURNS", str(standard["turns_kept"])))
    drop_block_size = int(
        os.environ.get(
            "STREAM_DETAILED_HISTORY_DROP_BLOCK_SIZE",
            str(standard["turn_archive_block"]),
        )
    )
    if window <= 0:
        raise ValueError("detailed history turns must be positive")
    if drop_block_size <= 0 or drop_block_size > window:
        raise ValueError("detailed history drop block size must be in [1, detailed history turns]")
    return window, drop_block_size


def effective_visual_fold_policy() -> tuple[int, int]:
    """Return the independent active-image budget and visual folding block size.

    At ``budget + 1`` active visual observations, the oldest ``block_size`` screenshots and their
    bound interactables are replaced together by fixed text. The matching text/action turns remain
    until the separate detailed-history policy archives them.
    """
    standard = context_defaults()["standard"]
    budget = int(os.environ.get("STREAM_ACTIVE_VISUAL_BUDGET", str(standard["screenshots_kept"])))
    block_size = int(
        os.environ.get(
            "STREAM_VISUAL_FOLD_BLOCK_SIZE",
            str(standard["screenshot_fold_block"]),
        )
    )
    if budget <= 0:
        raise ValueError("active visual budget must be positive")
    if block_size <= 0 or block_size > budget:
        raise ValueError("visual fold block size must be in [1, active visual budget]")
    return budget, block_size


@dataclass(frozen=True)
class PathPolicy:
    """The policy as one model API applies it."""

    version: int
    api: str
    preset: str | None  # the PRESETS entry in effect: the model config's, else the row's
    # 1 screenshot retention: at screenshots_kept + 1 active screenshots, the oldest
    # screenshot_fold_block are folded
    screenshots_kept: int
    screenshot_fold_block: int
    # 2 turn retention, the archived actions forming the ledger: at turns_kept + 1 verbatim turns,
    # the oldest turn_archive_block are archived; the block is also the one archived under context
    # pressure, and the block of archived actions a request that still does not fit leaves out
    turns_kept: int
    turn_archive_block: int
    # 3 execution feedback; under the action digest (item 9) every turn's feedback is in the
    # digest, whatever the value
    execution_feedback: Literal["separate_message", "digest"]
    # 4 role of the archived-action ledger; an encoder without roles places it by position and
    # header (RealtimeCapabilities.labels_instead_of_roles), and the value is at most a label
    # there (providers/decisions/history.py JEV_ROLES)
    ledger_role: Literal["instruction", "input"]
    texts: FixedTexts  # 5 placeholder and fixed texts
    reasoning_replay: bool  # 6 send past reasoning back; a reply is rebuilt from the model output
    # 8 before each request, archive the oldest turns while it exceeds the declared context
    # window, then, with no turn left, leave out the earliest archived actions, which stay out
    # for the rest of the episode as archived turns do. On allows it; it acts only where the
    # API's driver estimates its requests (it overrides StatelessEncoder.request_context_status)
    # and the model config has a window
    archive_over_window: bool
    # 9 Decision history form; the pre-run checks take only the form the API replays
    # (RealtimeCapabilities.history_replay)
    decision_history: DecisionHistory | None
    tool_call_history: Literal["native_tools", "tool_transcript"] | None  # 10 past live tool calls
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
        if path.tool_call_history is None:
            return path
        # Gemini Live takes the format from the model YAML's provider_options; a model that sets
        # none keeps the row's.
        return replace(
            path,
            tool_call_history=provider_options.get("history_format", path.tool_call_history),
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
# from FIXED_TEXTS (item 5): the defaults file's, but for items 9 and 10, which apply only to the
# rows that name their own (the Decision rows, gemini_live), and items 4 and 11, which the code
# decides.
_DEFAULT = context_defaults()["standard"]
STANDARD = {
    "execution_feedback": _DEFAULT["execution_feedback"],
    "ledger_role": "input",
    "reasoning_replay": _DEFAULT["reasoning_replay"],
    "archive_over_window": _DEFAULT["archive_over_window"],
    "decision_history": None,
    "tool_call_history": None,
    "prompts": _TEXT_JSON_PROMPTS,
    "observation": _DEFAULT["observation"],
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
        "execution_feedback": "digest",
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
    screenshots_kept, screenshot_fold_block = effective_visual_fold_policy()
    turns_kept, turn_archive_block = effective_detailed_history_policy()
    own_values = {}

    def path(api, preset=None, **values) -> PathPolicy:
        own_values[api] = values
        return PathPolicy(
            version=CONTEXT_POLICY_VERSION,
            api=api,
            preset=preset,
            screenshots_kept=screenshots_kept,
            screenshot_fold_block=screenshot_fold_block,
            turns_kept=turns_kept,
            turn_archive_block=turn_archive_block,
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
            tool_call_history=_DEFAULT["tool_call_history"],
            prompts={"realtime": "function_call", "browser_game": "function_call"},
        ),
        path(
            "openai_decisions",
            "luna",
            decision_history=DecisionHistory(_DEFAULT["decision_history"]),
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
