"""JEV interaction profile and the instructions of its Choice questions."""

from __future__ import annotations

from realtime_framework.harnesses.default.harness import InteractionProfile

JEV_PROFILE = InteractionProfile(
    "jev_decisions",
    action_space="Select one of the current observation's candidate actions.",
    output_contract=(
        "Return a typed Choice. You receive the current full interactables SoM and its "
        "action-bound mark legend, without screenshots. Choose only executable candidates. "
        "Typing choices contain observed literals, field suggestions, and calendar dates "
        "in or adjacent to observed months, filtered by explicit field constraints. "
        "Compose text lets you build other input through successive choices of exact words, "
        "characters, or Unicode code points, then finish to type it. Spaces and casing are explicit. "
        "You receive the current browser accessibility tree, visible control content, "
        "and JEV-specific appearance descriptors. Hidden application state is unavailable."
    ),
    frame_prompt="Choose the next action from the current observation.",
)

MULTIMODAL_CHOICE_PROFILE = InteractionProfile(
    "multimodal_decisions",
    action_space=JEV_PROFILE.action_space,
    output_contract=(
        "Return a typed Choice from the current observation's executable candidates. "
        "You receive its screenshot, full interactables SoM, action-bound mark legend, "
        "accessibility tree and visible control state. Hidden application state is unavailable. "
        "Typing choices contain observed literals and field suggestions. Compose text builds "
        "other input through successive choices of words, characters or Unicode code points, "
        "then finishes to type it. Spaces and casing are explicit."
    ),
    frame_prompt=JEV_PROFILE.frame_prompt,
)

MULTIMODAL_GAME_CHOICE_PROFILE = InteractionProfile(
    "multimodal_game_decisions",
    action_space=MULTIMODAL_CHOICE_PROFILE.action_space,
    output_contract=(
        MULTIMODAL_CHOICE_PROFILE.output_contract
        + " In games, declared keys offer taps and short holds, including simultaneous pairs. "
        "Holds persist for their stated duration; later actions do not cancel them. "
        "Grid choices name visible board positions without judging move legality. "
        "Pointer composition chooses coordinates before executing one action. "
        "mouse_down stays pressed until mouse_up; use move between them for a drag."
    ),
    frame_prompt=MULTIMODAL_CHOICE_PROFILE.frame_prompt,
)

GUI_INSTRUCTIONS = (
    "Choose exactly one next GUI action to accomplish the task. Use the newest "
    "AX tree, SoM observation, task constraints, and execution feedback. "
    "For compact typing choices, the current state's focused_input identifies the field; "
    "the exact value remains in the Choice criterion. "
    "Click a text field to focus it, then select a suitable literal typing action or "
    "compose_text to build new input from exact words/characters. Composition inserts "
    "at the caret or replaces the current selection, like a normal type action. "
    "To replace a nonempty value, select all, then type to replace the selection. "
    "Clicking the field repeatedly does not enter text. Use drag/drop choices for "
    "Web drag targets. Wait only when timing matters "
    "or no offered action advances the task. Earlier marks are stale. Candidate "
    "descriptions identify actions, not recommendations."
)
GAME_INSTRUCTIONS = (
    "Choose exactly one next GUI action toward the game objective using the newest screenshot, "
    "current marks, declared controls and execution feedback. Choose a tap for one press or "
    "a short hold for continuous control; paired keys are held simultaneously. "
    "Use visible point/grid choices or compose a pointer location. For a drag, press the mouse, "
    "move on later observations, then release it. Do not reload or use browser shortcuts. "
    "Candidates describe available inputs, not recommended moves or hidden game state."
)
COMPOSE_INSTRUCTIONS = (
    "Compose the exact text to insert into focused_input to accomplish the user's task. "
    "Choose one edit of text_composition.draft, or finish when it is ready to type. "
    "Append choices add their exact text, with NO automatic spaces, capitalization, "
    "punctuation or correction. Select space/newline/tab explicitly when needed. "
    "The draft is not yet on the page. Finish inserts it at the existing caret or "
    "replaces the current selection; it does not clear the field or submit the form. "
    "You may delete, clear, or cancel the draft. Field constraints are context, not "
    "a restriction on intermediate drafts. Every choice remains available after repeats."
)
UNICODE_INSTRUCTIONS = (
    "Choose the next hexadecimal digit of the Unicode scalar to append to "
    "the draft. Use exactly six digits including leading zeros. Back returns "
    "to editing without adding a character."
)
GROUP_NOTE = (
    " This is one group of a larger candidate set. Select the best action "
    "in this group; it will be compared with the other group winners."
)
