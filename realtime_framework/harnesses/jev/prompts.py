"""Interaction profiles of the choice models."""

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
