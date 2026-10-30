from __future__ import annotations

# Context policy
"""What each model API is shown at every step, and where: eleven named items and a version.

Version 1 copied what each code path did before the policy existed; every later version makes
one change (see CONTEXT_POLICY_VERSION), to one item, except version 2, which changed generic
LiteLLM's reasoning and verbatim replay (items 6 and 7) together. Item 7, which no encoder could
apply, has since been dropped. A change that alters what some model is shown bumps the version
(in phase 2 once, at its release: see CONTEXT_POLICY_VERSION); moving a setting from an API row
to the model that needs it alters nothing shown and keeps it. The values count in the effective
run config and therefore in its hash. Every item is read from here: retention, feedback
placement, the ledger role, reasoning replay, pressure archiving, the Decision history form, the
tool-call replay format, the prompts (item 11) and the observation form (item 12); the encoders
take their fixed texts (item 5) from here.

A model configuration's policy (path_policy) takes the items that are context settings (SETTINGS)
from its resolved context settings (ModelConfig.context, resolve_context): the defaults file's
standard values (realtime_framework/context_defaults.yaml), then those of the model's YAML, then
the run's. Its API row adds what the code decides: the prompts, the ledger role and the fixed
texts. The pre-run checks hold each policy against what the API declares
(providers/preflight.py check_context_policy).

Two values say what the policy allows, not what a run does. Pressure archiving (item 8) is on for
every API, and it acts only where the encoder estimates its requests and the model config has a
context window (ModelConfig.context_window_tokens). Luna's encoder, which has no roles, is given
the ledger role (item 4) but places the ledger by position and header; JEV's, which has none
either, labels the ledger's entry with it (providers/decisions/history.py JEV_ROLES). Whether
either takes effect is not recorded on its own: the driver code decides it, which the harness
fingerprint (or an external driver's identity) covers, and so does the model config, which the
run hash covers.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER, FOLDED_VISUAL_PLACEHOLDER
from realtime_framework.entity.records import _jsonable

# 2: generic LiteLLM rebuilds its replies from the model output and replays no reasoning; Kimi
#    K3 keeps its reasoning through its own settings.
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
# defaults file alone (DEFAULTS_FILE): the standard of this harness version, and the deviations
# from it that some models are approved to take.
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


@dataclass(frozen=True)
class ApprovedDeviation:
    """Settings a model of an API is approved to take instead of the standard values, and why."""

    api: str
    model: str  # ModelConfig.model_name
    settings: Mapping[str, bool | int | str]
    reason: str
    # The model's documentation requires the settings: it runs only with them
    # (providers/preflight.py check_required_context).
    required: bool = False


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


def settings_error(settings) -> str | None:
    """What is wrong with a mapping of some settings to values, if anything."""
    if not isinstance(settings, Mapping):
        return "must be a mapping of settings to values"
    unknown = sorted(set(settings) - set(SETTINGS))
    if unknown:
        return f"has unknown settings {unknown}"
    for key, value in settings.items():
        error = setting_error(key, value)
        if error:
            return error
    return None


_DEVIATION_FIELDS = {"api", "model", "settings", "reason"}


def _approved_deviation(name: str, entry) -> ApprovedDeviation:
    fields = set(entry) if isinstance(entry, dict) else set()
    if not _DEVIATION_FIELDS <= fields <= {*_DEVIATION_FIELDS, "required"}:
        raise ValueError(
            f"{name}: each approved deviation has api, model, settings and reason, and may say "
            f"it is required; got {entry!r}"
        )
    where = f"{name}: the approved deviation of {entry['api']} {entry['model']}"
    error = settings_error(entry["settings"])
    if error or not entry["settings"]:
        raise ValueError(f"{where}: settings {error or 'name no setting'}")
    if not isinstance(entry["reason"], str) or not entry["reason"].strip():
        raise ValueError(f"{where}: give its reason")
    required = entry.get("required", False)
    if type(required) is not bool:
        raise ValueError(f"{where}: required must be true or false, got {required!r}")
    return ApprovedDeviation(
        entry["api"],
        entry["model"],
        MappingProxyType(dict(entry["settings"])),
        entry["reason"],
        required,
    )


def load_context_defaults(path) -> dict:
    """The defaults file at ``path``, checked: every setting's standard value, and the deviations
    approved for some models."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"The context defaults file is missing: {path}")
    data = yaml.safe_load(path.read_text())
    name = path.name
    if not isinstance(data, dict) or not isinstance(data.get("standard"), dict):
        raise ValueError(  # noqa: TRY004 - a malformed file, as the catalog's YAMLs
            f"{name}: the file needs a standard section, a mapping of the settings"
        )
    others = sorted(set(data) - {"standard", "approved_deviations"})
    if others:
        raise ValueError(f"{name}: the file holds standard and approved_deviations; got {others}")
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
    approved = {}
    for entry in data.get("approved_deviations") or ():
        deviation = _approved_deviation(name, entry)
        if (deviation.api, deviation.model) in approved:
            raise ValueError(f"{name}: {deviation.api} {deviation.model} is approved twice")
        approved[deviation.api, deviation.model] = deviation
    return {
        "standard": MappingProxyType({key: standard[key] for key in SETTINGS}),
        "approved_deviations": MappingProxyType(approved),
    }


@lru_cache(maxsize=1)
def context_defaults() -> dict:
    """The package's defaults file (DEFAULTS_FILE), read once per process."""
    return load_context_defaults(DEFAULTS_FILE)


def _shown(value) -> str:
    """A setting's value as a YAML file writes it."""
    return ("true" if value else "false") if isinstance(value, bool) else str(value)


_KINDS = {
    "condition": "evaluation condition",
    "protocol": "what a model's protocol may need",
    "capability": "what an API's channel can render",
}


def settings_table() -> str:
    """The guide's tables (docs/model_configuration.md, Context settings): every setting, from
    SETTINGS and the defaults file's standard values, and the approved deviations."""
    standard = context_defaults()["standard"]
    lines = [
        "| Setting | Standard | Values | Applies to | Kind | Model YAML | What it is |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, setting in SETTINGS.items():
        values = (
            "a positive count"
            if setting.values is int
            else "`true`, `false`"
            if setting.values is bool
            else ", ".join(f"`{value}`" for value in setting.values)
        )
        applies = (
            ", ".join(f"`{api}`" for api in setting.applies) if setting.applies else "every API"
        )
        lines.append(
            f"| `{key}` | `{_shown(standard[key])}` | {values} | {applies} | "
            f"{_KINDS[setting.kind]} | {'yes' if setting.model else 'no'} | {setting.about} |"
        )
    lines += [
        "",
        "| API | Model | Settings | Required | Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for deviation in context_defaults()["approved_deviations"].values():
        settings = ", ".join(
            f"`{key}: {_shown(value)}`" for key, value in deviation.settings.items()
        )
        required = "yes" if deviation.required else "no"
        lines.append(
            f"| `{deviation.api}` | `{deviation.model}` | {settings} | {required} | "
            f"{deviation.reason} |"
        )
    return "\n".join(lines) + "\n"


# The resolved context settings
Source = Literal["default", "experiment", "model", "run"]
# A setting's value as its definition takes it (setting_error), never coerced from another type.
Value = StrictBool | StrictInt | StrictStr
_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Overridden(BaseModel):
    """A value a higher layer replaced, and the layer that gave it."""

    model_config = _FROZEN

    source: Source
    value: Value


class ContextItem(BaseModel):
    """One setting as resolved: its value and where it came from, the values it replaced, lowest
    layer first, whether it applies to the API, and how it stands against the standard: standard,
    approved (an approved deviation of the model) or nonstandard; None where it does not apply."""

    model_config = _FROZEN

    key: str
    value: Value
    source: Source
    overridden: tuple[Overridden, ...]
    applies: bool
    category: Literal["standard", "approved", "nonstandard"] | None


class Deviation(BaseModel):
    """An applicable setting away from its standard value."""

    model_config = _FROZEN

    key: str
    standard: Value
    value: Value
    source: Source
    category: Literal["approved", "nonstandard"]


class ExperimentFile(BaseModel):
    """The experiment file a run's settings came from (read_experiment_file)."""

    model_config = _FROZEN

    path: str  # absolute
    sha256: str


class ContextSettings(BaseModel):
    """A model configuration's context settings, resolved once (resolve_context): every setting
    in SETTINGS order, the experiment file some came from, and whether the run is on the standard
    context, which takes every applicable setting standard or approved for the model. Immutable
    all the way down, and the same after a dump and a validation, which read it back rather than
    resolve it again."""

    model_config = _FROZEN

    api: str  # ModelConfig.adapter_type
    model: str  # ModelConfig.model_name, which approved deviations name
    items: tuple[ContextItem, ...]
    # The record names one only where a run took one.
    experiment_file: ExperimentFile | None = Field(
        default=None, exclude_if=lambda file: file is None
    )
    standard: bool
    deviations: tuple[Deviation, ...]

    def values(self) -> dict:
        """Each setting's value, the part of the record the configuration hashes take."""
        return {item.key: item.value for item in self.items}

    def item(self, key: str) -> ContextItem:
        return next(item for item in self.items if item.key == key)


def context_values(record: Mapping) -> dict:
    """Each setting's value in a dumped ContextSettings ``record`` (ModelConfig.model_dump), the
    part of it the configuration hashes take (evaluation/config.py configuration_sha256,
    effective_run_config): not where a value came from or how it stands."""
    return {item["key"]: item["value"] for item in record["items"]}


def _row_name(api: str) -> str:
    """The row an API takes for applicability: an external driver ("package.module:Class") the
    plain chat row, any other its own name."""
    return EXTERNAL_DRIVER_ROW if ":" in api else api


def layer_error(source: Source, api: str, layer: Mapping) -> str | None:
    """What a layer may not set, if anything: a model's YAML only the settings a model may set
    (Setting.model), and only those that apply to its API; an experiment file and the command line
    any setting."""
    error = settings_error(layer)
    if error:
        return error
    if source == "model":
        for key in layer:
            if not SETTINGS[key].model:
                allowed = [key for key, setting in SETTINGS.items() if setting.model]
                return f"a model may set only {allowed}, not {key}"
            applies = SETTINGS[key].applies
            if applies is not None and _row_name(api) not in applies:
                return f"{key} does not apply to the {api} API"
    return None


# The variables and options context settings replaced, refused with where each went.
LEGACY_CONTEXT_VARIABLES = {
    "STREAM_ACTIVE_VISUAL_BUDGET": "screenshots_kept",
    "STREAM_VISUAL_FOLD_BLOCK_SIZE": "screenshot_fold_block",
    "STREAM_DETAILED_HISTORY_TURNS": "turns_kept",
    "STREAM_DETAILED_HISTORY_DROP_BLOCK_SIZE": "turn_archive_block",
    "BENCHMARK_CONTEXT_PRESET": "reasoning_replay",
}
CONTEXT_PRESET_OPTION_REMOVED = (
    "--context-preset was removed; give the context setting instead "
    "(kimi_k3 was --context reasoning_replay=true)"
)


def legacy_context_error(environ: Mapping) -> str | None:
    """The removed context variables set in ``environ``, with the settings that replaced them."""
    found = [variable for variable in LEGACY_CONTEXT_VARIABLES if variable in environ]
    if not found:
        return None
    replaced = ", ".join(
        f"{variable} by {LEGACY_CONTEXT_VARIABLES[variable]}" for variable in found
    )
    return (
        f"Unset {', '.join(found)}: context settings replaced them ({replaced}); give them with "
        "--context KEY=VALUE or in an experiment file (--context-file)"
    )


def read_experiment_file(path) -> tuple[ExperimentFile, dict]:
    """The experiment file at ``path``, a YAML mapping of context settings to values, and its
    settings, which resolve_context checks as an experiment layer."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"The context file is missing: {path}")
    data = path.read_bytes()
    settings = yaml.safe_load(data)
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: a context file is a mapping of context settings to values")  # noqa: TRY004
    return ExperimentFile(path=str(path), sha256=hashlib.sha256(data).hexdigest()), settings


@dataclass(frozen=True)
class ContextLayers:
    """The context settings a run gives beside its model's YAML: an experiment file's and its
    command line's (the trial worker's --context-file and --context)."""

    experiment: Mapping = MappingProxyType({})
    experiment_file: ExperimentFile | None = None
    run: Mapping = MappingProxyType({})


def resolve_context(
    api: str,
    model: str,
    *,
    experiment: Mapping = MappingProxyType({}),
    experiment_file: ExperimentFile | None = None,
    model_layer: Mapping = MappingProxyType({}),
    run: Mapping = MappingProxyType({}),
) -> ContextSettings:
    """The context settings of model ``model`` of API ``api``: the defaults file's standard
    values, then an experiment file's (``experiment``, read from ``experiment_file``), then the
    model YAML's (``model_layer``), then the run's command line (``run``), each layer checked for
    what it may set (layer_error). Each setting applies to the rows SETTINGS names; an external
    driver takes the plain chat row's. Values that do not apply are kept, and take no part in
    whether the run is on the standard context."""
    layers = (("experiment", experiment), ("model", model_layer), ("run", run))
    for source, layer in layers:
        error = layer_error(source, api, layer)
        if error:
            raise ValueError(f"The {source} context settings of {model}: {error}")
    defaults = context_defaults()
    standard = defaults["standard"]
    deviation = defaults["approved_deviations"].get((api, model))
    approved = deviation.settings if deviation is not None else {}
    items, deviations = [], []
    for key, setting in SETTINGS.items():
        value, source, overridden = standard[key], "default", []
        for name, layer in layers:
            if key in layer:
                overridden.append(Overridden(source=source, value=value))
                value, source = layer[key], name
        applies = setting.applies is None or _row_name(api) in setting.applies
        category = None
        if applies:
            category = (
                "standard"
                if value == standard[key]
                else "approved"
                if key in approved and approved[key] == value
                else "nonstandard"
            )
            if category != "standard":
                deviations.append(
                    Deviation(
                        key=key,
                        standard=standard[key],
                        value=value,
                        source=source,
                        category=category,
                    )
                )
        items.append(
            ContextItem(
                key=key,
                value=value,
                source=source,
                overridden=tuple(overridden),
                applies=applies,
                category=category,
            )
        )
    values = {item.key: item.value for item in items}
    error = blocks_error(values)
    if error:
        raise ValueError(f"The context settings of {model}: {error}")
    return ContextSettings(
        api=api,
        model=model,
        items=tuple(items),
        experiment_file=experiment_file,
        standard=all(item.category != "nonstandard" for item in items),
        deviations=tuple(deviations),
    )


@dataclass(frozen=True)
class PathPolicy:
    """The policy as one model API applies it."""

    version: int
    api: str
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
    # 9 Decision history form, None where it does not apply; the pre-run checks take only the
    # form the API replays (RealtimeCapabilities.history_replay)
    decision_history: DecisionHistory | None
    # 10 past live tool calls, None where it does not apply
    tool_call_history: Literal["native_tools", "tool_transcript"] | None
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
# What the code decides for each registered API's row, beside the fixed texts and the ledger role
# (input everywhere): the profile each task category is prompted with (item 11).
ROW_PROMPTS = {
    "litellm": _TEXT_JSON_PROMPTS,
    "chat_completions": _TEXT_JSON_PROMPTS,
    "openrouter": _TEXT_JSON_PROMPTS,
    "anthropic_messages": _TEXT_JSON_PROMPTS,
    "moonshot": _TEXT_JSON_PROMPTS,
    "openai_realtime": _TEXT_JSON_PROMPTS,
    "gemini_live": {"realtime": "function_call", "browser_game": "function_call"},
    "openai_decisions": {
        "realtime": "multimodal_decisions",
        "browser_game": "multimodal_game_decisions",
    },
    "jev_decisions": {"realtime": "jev_decisions", "browser_game": None},
}


def context_row(api: str) -> str:
    """The row API ``api`` takes: its own, or EXTERNAL_DRIVER_ROW for an external driver.

    Registered names contain no colon and external ones are "package.module:Class"
    (providers/registry.py), so a name with no row and no colon is a registered API whose row is
    missing.
    """
    if api in ROW_PROMPTS:
        return api
    if ":" in api:
        return EXTERNAL_DRIVER_ROW
    raise ValueError(
        f"API {api!r} has no context-policy row; add one in ROW_PROMPTS "
        "(realtime_framework/entity/context_policy.py)"
    )


def path_policy(config) -> PathPolicy:
    """The policy a model configuration runs under, before any driver exists: its resolved
    context settings (ModelConfig.context) on its API's row, with None for the items that do not
    apply to it."""
    row = context_row(config.adapter_type)
    context = config.context
    values = context.values()
    history = context.item("decision_history")
    tools = context.item("tool_call_history")
    return PathPolicy(
        version=CONTEXT_POLICY_VERSION,
        api=row,
        screenshots_kept=values["screenshots_kept"],
        screenshot_fold_block=values["screenshot_fold_block"],
        turns_kept=values["turns_kept"],
        turn_archive_block=values["turn_archive_block"],
        execution_feedback=values["execution_feedback"],
        ledger_role="input",
        texts=FIXED_TEXTS,
        reasoning_replay=values["reasoning_replay"],
        archive_over_window=values["archive_over_window"],
        decision_history=DecisionHistory(history.value) if history.applies else None,
        tool_call_history=tools.value if tools.applies else None,
        prompts=ROW_PROMPTS[row],
        observation=values["observation"],
    )
