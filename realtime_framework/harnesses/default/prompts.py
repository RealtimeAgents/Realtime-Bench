from __future__ import annotations

# Prompts
"""Scenario roles, model-specific instructions, and stream contracts for desktop agents.

Task-specific objectives come from each task's instruction. All adapters compose the shared stream
contract with the model/interface-specific prompts centralized here.
"""

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
  right/bottom is 1000. Never reuse a mark from an earlier screenshot.
- Prefer a supplied mark. If no suitable mark exists, replace target with x/y, start with x1/y1,
  or end with x2/y2 using the screenshot coordinate space. Reference and coordinate forms may be
  mixed for drag endpoints, but never specify both forms for the same point.
- Use the coordinate contract supplied with the current screenshot. The harness converts image
  coordinates to the executor's 1920x1080 desktop space.
- Pointer coordinates may be omitted for click variants, mouse_down/up, and scroll only to act at
  the current cursor position. When present, x and y must be supplied together.
"""
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
GUI_ACTION_PARAMETER_RULES = _POINTER_RULES + _TIMING_AND_MODIFIER_RULES + "\n" + _GESTURE_RULES

# Function-call wording: the shared rules plus argument-completeness rules for move/drag/scroll
# and optional fields.
FUNCTION_CALL_PARAMETER_RULES = (
    _POINTER_RULES
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

GUI_JSON_ACTION_SPACE = (
    "AVAILABLE GUI ACTIONS (canonical JSON):\n"
    + GUI_ACTION_EXAMPLES
    + "\n- Prefer canonical JSON with double quotes. The adapter can recover one unambiguous action "
    "from\n  common wrappers or shorthand, but canonical JSON is shortest and safest.\n"
    + GUI_ACTION_PARAMETER_RULES
)

_STREAM_OUTPUT_CONTRACT = """ACTION OUTPUT

Use your configured or native reasoning internally. Your entire visible response must be exactly
one valid action from the active action space. Output the action and its arguments only: no
reasoning, commentary, labels, wrappers, tags, Markdown, ellipses, or placeholders.
The action is final and immediately executable.{output_tokens}
The adapter accepts common formatting wrappers when it can recover exactly one unambiguous valid
action. If it finds no valid action or multiple different actions, the harness executes WAIT.
"""
STREAM_OUTPUT_CONTRACT = _STREAM_OUTPUT_CONTRACT.format(output_tokens="")
# Only a finite B charges output tokens to simulated time (budget_clock_prompt); the wall clock
# charges none, and the pause clock lands each action at the tick it observed.
FINITE_BUDGET_OUTPUT_CONTRACT = _STREAM_OUTPUT_CONTRACT.format(
    output_tokens=" All generated output tokens count."
)


# Model/interface-specific native-tool instructions.
FUNCTION_CALL_ACTION_SPACE = (
    "AVAILABLE GUI ACTIONS (native function calls):\n"
    "The examples below describe the same actions and parameters as the shared GUI protocol. "
    "They are reference notation, NOT text to output: use the example's action value as the "
    "declared function name and its remaining fields as function arguments. Do not include an "
    "action argument. Call exactly one declared function; do not output JSON text or speech.\n"
    + GUI_ACTION_EXAMPLES
    + "\n"
    + FUNCTION_CALL_PARAMETER_RULES
)
FUNCTION_CALL_OUTPUT_CONTRACT = (
    "NATIVE GUI ACTION PROTOCOL: Do not produce a spoken response or text action. "
    "Reason internally, then call exactly one GUI action function. That call fixes your "
    "decision. Its response only acknowledges queuing, not execution. After the queue "
    "acknowledgement, end the turn silently without further calls. The benchmark waits "
    "for native response plus thought token usage under the token clock, then advances simulated "
    "time. Under the wall clock, environment time keeps advancing during generation. It executes "
    "the action only if its timestamp remains inside the task horizon. Otherwise it executes "
    "WAIT. The next observation includes the actual execution result. Do not assume success "
    "from the queue acknowledgement. Use wait when no UI action is needed."
)
FUNCTION_CALL_FRAME_PROMPT = (
    "Latest screen. Use current P/R marks when supplied. Submit exactly one GUI action by "
    "calling its function. Do not speak."
)

GAME_ROLE = """You are playing a live browser game. Use only the visible game controls and keep acting
toward the TASK goal. Do not open a terminal, editor, source code, developer tools, or test harness,
do not reload the page, and do not modify the game. Prefer hold_key with a short duration
for continuous movement instead of many discrete taps. Keep playing until the environment ends the episode."""

REALTIME_WEB_ROLE = """You are operating a live web task. The page and server state may change while
you reason, so use the latest visible state and complete the TASK through the visible UI. Do not inspect
or edit source code, call hidden APIs, or use developer tools or terminal shortcuts. The environment
determines when the episode ends; use wait when no further UI interaction is needed."""

SCENARIO_ROLES = {"browser_game": GAME_ROLE, "realtime": REALTIME_WEB_ROLE}


def _scenario_prompt(role: str, contract: str) -> str:
    return "\n\n".join((role, GUI_JSON_ACTION_SPACE, contract))


REALTIME_WEB_SYSTEM_PROMPT = _scenario_prompt(REALTIME_WEB_ROLE, STREAM_OUTPUT_CONTRACT)


def gui_system_prompt(
    task_instruction: str, *, scenario_prompt: str | None = None, clock_prompt: str | None = None
) -> str:
    """Compose identical episode instructions for screenshot VLM and streaming models."""
    return "\n\n".join(
        filter(
            None,
            (
                scenario_prompt or REALTIME_WEB_SYSTEM_PROMPT,
                clock_prompt,
                f"TASK:\n{task_instruction}",
            ),
        )
    )


def system_prompt_for_category(category: str | None, *, finite_budget: bool) -> str:
    if category not in SCENARIO_ROLES:
        raise ValueError(f"unsupported task category for scenario prompt: {category!r}")
    contract = FINITE_BUDGET_OUTPUT_CONTRACT if finite_budget else STREAM_OUTPUT_CONTRACT
    return _scenario_prompt(SCENARIO_ROLES[category], contract)


# Shared screenshot marker for text-action adapters. Each frame keeps its text once newer frames
# follow, so it names the screen of its own decision.
FRAME_PROMPT = "Screen observed for this decision."

# Stateful text-output VLM sessions can under-weight an old system message, so each frame repeats only
# the wire format. Scenario role and task objective remain exclusively in the session seed.
TEXT_ACTION_FRAME_PROMPT = (
    FRAME_PROMPT
    + " Use current P/R marks when supplied. Output exactly one action and nothing else."
)


def budget_clock_prompt(fps: float, tokens_per_frame: int | None, clock_mode: str = "token") -> str:
    """Describe the effective evaluation clock without assuming a provider session mechanism."""
    if clock_mode == "wall":
        return (
            f"WALL CLOCK: observations are scheduled at {fps:g} fps. "
            "Environment time follows physical elapsed time. Produce at most one action "
            "per decision and use execution feedback to track which actions were applied."
        )
    timing = (
        f"TOKEN CLOCK: observations are scheduled at f={fps:g} frames per simulated second. "
        "Physical inference and GUI execution latency do not advance simulated time. "
        "Produce at most one action per observation. "
    )
    if tokens_per_frame is None:
        return timing + (
            "B is unlimited: output length does not affect simulated time, so each action lands "
            "at the tick it observed. The environment then advances exactly one frame interval "
            "(1/f seconds), and the next observation includes the action's effect."
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

ACTION_RESULT_NOTES = {
    "none": INVALID_ACTION_FEEDBACK,
    "skipped_timeout": "NOTE: your previous action was SKIPPED because it exceeded the per-move cap; nothing was applied.",
    "skipped_budget": "NOTE: your previous action was NOT applied because the deliberation budget was exhausted.",
    "not_applied_time_limit": "NOTE: your previous action was NOT applied because the time limit was reached; the screen is unchanged.",
}
