from __future__ import annotations

# Prompts
"""Scenario roles, model-specific instructions, and stream contracts for desktop agents.

Task-specific objectives come from each task's instruction. All adapters compose the shared stream
contract with the model/interface-specific prompts centralized here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptOptions:
    """The run settings an episode's prompts are composed for, set before initialize()
    (harnesses/registry.py configure_agent). Besides the task and the frame rates, the prompts
    read nothing else of the run."""

    interaction_mode: str  # "turn_based" or "streaming"
    clock_mode: str  # "wall" or "token"
    # Only a finite B charges output tokens to simulated time; the wall clock charges none, and the
    # pause clock (the token clock with B unlimited) lands each action at the tick it observed.
    finite_budget: bool


# Keep action examples identical across text and native-tool adapters.
GUI_ACTION_EXAMPLES = """- {"action": "click", "target": "P1"|"R1@u,v", "button": "left"|"middle"|"right"}
- {"action": "right_click"|"double_click"|"triple_click", "target": "P1"|"R1@u,v"}
- {"action": "mouse_down"|"mouse_up"|"move", "target": "P1"|"R1@u,v", "duration": <seconds>}
- {"action": "type", "text": "<string>"}
- {"action": "key", "key": "<key_or_combo>", "repeat": <int>}  (short tap; examples: "enter", "ctrl+s")
- {"action": "hold_key", "key": "<key_or_combo>", "duration": <seconds>}  (continuous control)
- {"action": "scroll", "target": "P1"|"R1", "dx": <int>, "dy": <int>}  (positive x=right, positive y=up)
- {"action": "drag", "start": "P1"|"R1@u,v", "end": "P2"|"R1@u,v", "duration": <seconds>}
- {"action": "wait", "duration": <seconds>}
"""

_POINTER_RULES = """- P/R marks belong only to the screenshot that supplies them. Pn is an exact target point.
  Rn@u,v addresses a continuous region using local integers from 0..1000: left/top is 0 and
  right/bottom is 1000. {marks}
- Prefer a supplied mark. If no suitable mark exists, replace target with x/y, start with x1/y1,
  or end with x2/y2 using the screenshot coordinate space. Reference and coordinate forms may be
  mixed for drag endpoints, but never specify both forms for the same point.
- Use the coordinate contract supplied with {screenshot}. The harness converts image
  coordinates to the executor's 1920x1080 desktop space.
- Pointer coordinates may be omitted for click variants, mouse_down/up, and scroll only to act at
  the current cursor position. When present, x and y must be supplied together.
"""
TURN_POINTER_RULES = _POINTER_RULES.format(
    marks="Never reuse a mark from an earlier screenshot.", screenshot="the current screenshot"
)
# A streaming action's marks and coordinates are read from the frame it names
# (FRAME_REFERENCE_RULES).
STREAMING_POINTER_RULES = _POINTER_RULES.format(
    marks="An action's marks are read from the screenshot its frame names.",
    screenshot="the screenshot the action's frame names",
)
_TIMING_AND_MODIFIER_RULES = """- duration must be between 0 and 10. For hold_key and wait it is simulated world time under the
  token clock; hold_key defaults to 1 and wait without duration waits for the next observation.
  For move, mouse_down/up, and drag it controls executor-side pointer interpolation only; GUI
  execution time is excluded from simulated time.
- hold_key keeps every key in the '+'-joined chord down for its full declared duration. A later
  action does not cancel the hold; another hold on the same key can only extend it. Keys are released
  automatically at expiry and all held input is released when the episode ends. A held key
  auto-repeats like a physical keyboard (first repeat after 0.5 s, then every 0.05 s), so a game
  that moves one step per key press keeps stepping for as long as the key is held.
- key.repeat is optional, defaults to 1, and must be an integer from 1 to 100.
- Click variants, scroll, and drag may include "modifier": a '+'-joined combination of
  "shift", "ctrl", "alt", and "super". Modifiers apply only for that action."""
_GESTURE_RULES = """- mouse_down remains active until the matching mouse_up; use drag for an ordinary complete drag.
  For a gesture that must react to changing frames, use mouse_down, then move actions on later
  observations, then mouse_up; each step may use that observation's current marks.
"""


def _gui_json_action_space(pointer_rules: str) -> str:
    return (
        "AVAILABLE GUI ACTIONS (canonical JSON):\n"
        + GUI_ACTION_EXAMPLES
        + "\n- Prefer canonical JSON with double quotes. The adapter can recover one unambiguous "
        "action from\n  common wrappers or shorthand, but canonical JSON is shortest and safest.\n"
        + pointer_rules
        + _TIMING_AND_MODIFIER_RULES
        + "\n"
        + _GESTURE_RULES
    )


GUI_JSON_ACTION_SPACE = _gui_json_action_space(TURN_POINTER_RULES)
STREAMING_GUI_JSON_ACTION_SPACE = _gui_json_action_space(STREAMING_POINTER_RULES)

# Native streaming runs nothing for a reply that holds no valid action
# (harnesses/streaming/feedback.py).
_STREAM_OUTPUT_CONTRACT = """ACTION OUTPUT

Use your configured or native reasoning internally. Your entire visible response must be exactly
one valid action from the active action space. Output the action and its arguments only: no
reasoning, commentary, labels, wrappers, tags, Markdown, ellipses, or placeholders.
The action is final and immediately executable.{output_tokens}
The adapter accepts common formatting wrappers when it can recover exactly one unambiguous valid
action. If it finds no valid action or multiple different actions, {no_action}.
"""
STREAM_OUTPUT_CONTRACT = _STREAM_OUTPUT_CONTRACT.format(
    output_tokens="", no_action="the harness executes WAIT"
)
# PromptOptions.finite_budget.
FINITE_BUDGET_OUTPUT_CONTRACT = _STREAM_OUTPUT_CONTRACT.format(
    output_tokens=" All generated output tokens count.", no_action="the harness executes WAIT"
)
# A native-streaming run's (STREAM_OUTPUT_CONTRACT is a turn's), on the wall clock only.
STREAMING_OUTPUT_CONTRACT = _STREAM_OUTPUT_CONTRACT.format(
    output_tokens="", no_action="nothing is executed for it"
)


# Model/interface-specific native-tool instructions.
def _function_call_action_space(pointer_rules: str) -> str:
    """The shared rules plus argument-completeness rules for move/drag/scroll and optional
    fields."""
    return (
        "AVAILABLE GUI ACTIONS (native function calls):\n"
        "The examples below describe the same actions and parameters as the shared GUI protocol. "
        "They are reference notation, NOT text to output: use the example's action value as the "
        "declared function name and its remaining fields as function arguments. Do not include "
        "an action argument. Call exactly one declared function; do not output JSON text or "
        "speech.\n"
        + GUI_ACTION_EXAMPLES
        + "\n"
        + pointer_rules
        + """- move requires a destination. drag requires end or x2/y2; start or x1/y1 may be omitted to
  start at the current cursor. Never mix a reference and coordinates for the same endpoint.
- scroll requires both dx and dy, and they must not both be zero.
"""
        + _TIMING_AND_MODIFIER_RULES
        + """ If no modifier
  is needed, omit the field entirely; do not send "none", "", or null.
- Optional fields may be omitted; do not fill unused fields with placeholder values. For example,
  {"action":"click","target":"P1"} has no x/y or modifier fields.
"""
        + _GESTURE_RULES
    )


FUNCTION_CALL_ACTION_SPACE = _function_call_action_space(TURN_POINTER_RULES)
STREAMING_FUNCTION_CALL_ACTION_SPACE = _function_call_action_space(STREAMING_POINTER_RULES)
_FUNCTION_CALL_OUTPUT_CONTRACT = (
    "NATIVE GUI ACTION PROTOCOL: Do not produce a spoken response or text action. "
    "Reason internally, then call exactly one GUI action function. That call fixes your "
    "decision. Its response only acknowledges queuing, not execution. After the queue "
    "acknowledgement, end the turn silently without further calls. {token_clock} Under the "
    "wall clock, environment time keeps advancing during generation. It executes "
    "the action only if its timestamp remains inside the task horizon. Otherwise it executes "
    "WAIT. The next observation includes the actual execution result. Do not assume success "
    "from the queue acknowledgement. Use wait when no UI action is needed."
)
FUNCTION_CALL_OUTPUT_CONTRACT = _FUNCTION_CALL_OUTPUT_CONTRACT.format(
    token_clock="Under the token clock, the benchmark does not advance simulated time for native "
    "response or thought tokens."
)
# PromptOptions.finite_budget, as FINITE_BUDGET_OUTPUT_CONTRACT.
FINITE_BUDGET_FUNCTION_CALL_OUTPUT_CONTRACT = _FUNCTION_CALL_OUTPUT_CONTRACT.format(
    token_clock="The benchmark waits for native response plus thought token usage under the "
    "token clock, then advances simulated time."
)
# Shared screenshot marker for the frame prompts. Each frame keeps its text once newer frames
# follow, so it names the screen of its own decision.
FRAME_PROMPT = "Screen observed for this decision."

FUNCTION_CALL_FRAME_PROMPT = (
    FRAME_PROMPT + " Use current P/R marks when supplied. Submit exactly one GUI action by "
    "calling its function. Do not speak."
)
# A streamed frame is no decision's: an action names whichever frame it is based on.
STREAMING_FRAME_PROMPT = "Streamed screenshot; its number is in its FRAME label."
STREAMING_FUNCTION_CALL_FRAME_PROMPT = (
    STREAMING_FRAME_PROMPT + " Submit exactly one GUI action by calling its function, with the "
    "frame it is based on. Do not speak."
)

# Streaming only: each action names the frame it is based on.
FRAME_REFERENCE_RULES = """FRAME REFERENCE
Screenshots arrive continuously; each shows its number in a FRAME label in its top-left corner.
Every action must name the screenshot it is based on in "frame", for example
{"action": "click", "target": "P1", "frame": 12}. Its P/R marks and coordinates are read from that
screenshot, even after newer ones arrive. An action without the number of a screenshot you
received is not executed."""
FUNCTION_CALL_FRAME_REFERENCE_RULES = """FRAME REFERENCE
Screenshots arrive continuously; each shows its number in a FRAME label in its top-left corner.
Every GUI function call must pass the number of the screenshot it is based on as its frame
argument. Its P/R marks and coordinates are read from that screenshot, even after newer ones
arrive. A call without the number of a screenshot you received is not executed."""

GAME_ROLE = """You are playing a live browser game. Use only the visible game controls and keep acting
toward the TASK goal. Do not open a terminal, editor, source code, developer tools, or test harness,
do not reload the page, and do not modify the game. Prefer hold_key with a short duration
for continuous movement instead of many discrete taps. Keep playing until the environment ends the episode."""

REALTIME_WEB_ROLE = """You are operating a live web task. The page and server state may change while
you reason, so use the latest visible state and complete the TASK through the visible UI. Do not inspect
or edit source code, call hidden APIs, or use developer tools or terminal shortcuts. The environment
determines when the episode ends; use wait when no further UI interaction is needed."""

SCENARIO_ROLES = {"browser_game": GAME_ROLE, "realtime": REALTIME_WEB_ROLE}


def _scenario_prompt(role: str, action_space: str, contract: str) -> str:
    return "\n\n".join((role, action_space, contract))


REALTIME_WEB_SYSTEM_PROMPT = _scenario_prompt(
    REALTIME_WEB_ROLE, GUI_JSON_ACTION_SPACE, STREAM_OUTPUT_CONTRACT
)


def gui_system_prompt(
    task_instruction: str,
    *,
    scenario_prompt: str | None = None,
    clock_prompt: str | None = None,
    frame_reference_rules: str | None = None,
) -> str:
    """Compose identical episode instructions for screenshot VLM and streaming models."""
    return "\n\n".join(
        filter(
            None,
            (
                scenario_prompt or REALTIME_WEB_SYSTEM_PROMPT,
                frame_reference_rules,
                clock_prompt,
                f"TASK:\n{task_instruction}",
            ),
        )
    )


def system_prompt_for_category(category: str | None, options: PromptOptions) -> str:
    if category not in SCENARIO_ROLES:
        raise ValueError(f"unsupported task category for scenario prompt: {category!r}")
    role = SCENARIO_ROLES[category]
    if options.interaction_mode == "streaming":
        return _scenario_prompt(role, STREAMING_GUI_JSON_ACTION_SPACE, STREAMING_OUTPUT_CONTRACT)
    contract = FINITE_BUDGET_OUTPUT_CONTRACT if options.finite_budget else STREAM_OUTPUT_CONTRACT
    return _scenario_prompt(role, GUI_JSON_ACTION_SPACE, contract)


# Stateful text-output VLM sessions can under-weight an old system message, so each frame repeats only
# the wire format. Scenario role and task objective remain exclusively in the session seed.
TEXT_ACTION_FRAME_PROMPT = (
    FRAME_PROMPT
    + " Use current P/R marks when supplied. Output exactly one action and nothing else."
)
STREAMING_TEXT_ACTION_FRAME_PROMPT = (
    STREAMING_FRAME_PROMPT
    + " Output exactly one action, with the frame it is based on, and nothing else."
)


def budget_clock_prompt(
    fps: float,
    tokens_per_frame: int | None,
    options: PromptOptions,
    env_fps: float | None = None,
) -> str:
    """Describe the effective evaluation clock without assuming a provider session mechanism.
    ``fps`` is the model-input frame rate and ``tokens_per_frame`` B, set where the budget is
    finite. ``env_fps``, the environment's, is described only where it differs: frames then fall
    between environment ticks."""
    second = "second" if options.clock_mode == "wall" else "simulated second"
    ticks = (
        ""
        if env_fps is None or env_fps == fps
        else f"Each frame is captured at the first environment tick ({env_fps:g} per {second}) "
        "at or after its scheduled time; observed_at_tick in the action ledger counts these ticks. "
    )
    if options.clock_mode == "wall":
        # A native-streaming run has replies, not decisions.
        unit = "reply" if options.interaction_mode == "streaming" else "decision"
        return (
            f"WALL CLOCK: observations are scheduled at {fps:g} fps. "
            + ticks
            + "Environment time follows physical elapsed time. Produce at most one action "
            f"per {unit} and use execution feedback to track which actions were applied."
        )
    timing = (
        f"TOKEN CLOCK: observations are scheduled at f={fps:g} frames per simulated second. "
        + ticks
        + "Physical inference and GUI execution latency do not advance simulated time. "
        "Produce at most one action per observation. "
    )
    if not options.finite_budget:
        advance = (
            "to the next scheduled frame" if ticks else "exactly one frame interval (1/f seconds)"
        )
        return timing + (
            "B is unlimited: output length does not affect simulated time, so each action lands "
            f"at the tick it observed. The environment then advances {advance}, and the next "
            "observation includes the action's effect."
        )
    timing += (
        f"B={tokens_per_frame} charged tokens/frame; kappa=B*f={tokens_per_frame * fps:g} "
        "charged tokens per simulated second. After n provider/native output tokens, an action "
        "lands n/kappa simulated seconds after the observation. "
    )
    timing += (
        "B is a conversion rate, not a maximum response length. A decision may use more than B "
        "tokens, cross scheduled frame boundaries, and miss the corresponding observations. "
    )
    return timing + (
        "Finishing early does not provide an observation before the next scheduled frame boundary. "
        "The task ends at its fixed simulated-time limit T, with at most kappa*T charged tokens "
        "over the episode. Executed actions and their outcomes are retained in your context."
    )


INVALID_ACTION_FEEDBACK = (
    "NOTE: your previous response was not exactly one valid action, so WAIT was executed. "
    "Output one actual action with its arguments and nothing else."
)
GROUNDING_FAILED_FEEDBACK = (
    "NOTE: your previous action names {mark}, which is not a mark of the screenshot it was based "
    "on, so WAIT was executed."
)
UNSUPPORTED_TERMINAL_ACTION_FEEDBACK = (
    "NOTE: your previous action was done or fail, which are not available actions, so WAIT was "
    "executed. The environment determines when the episode ends."
)

# Notes on a decision of which no answer was accepted: no turn of the history holds its frame.
UNANSWERED_STEP_NOTES = {
    "skipped_timeout": "NOTE: a step timed out at the per-move cap before your answer arrived; no action was executed.",
}
UNANSWERED_STEP_NOTE = "NOTE: no answer was accepted for a step; no action was executed."

ACTION_RESULT_NOTES = {
    "none": INVALID_ACTION_FEEDBACK,
    "skipped_timeout": "NOTE: your previous action was SKIPPED because it exceeded the per-move cap; nothing was applied.",
    "skipped_budget": "NOTE: your previous action was NOT applied because the deliberation budget was exhausted.",
    "not_applied_time_limit": "NOTE: your previous action was NOT applied because the time limit was reached; the screen is unchanged.",
}
