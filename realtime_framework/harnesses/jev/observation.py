"""The observation JSON of the choice models (context policy item 12) and its answer options.

Text-only clients receive structured control state; multimodal clients additionally receive
its marked screenshot. Candidate and text-composition policies are shared by both.
"""

from __future__ import annotations

import json

from realtime_framework.entity.request import Candidate, ChoiceRequirements, SelectionMaterial
from realtime_framework.harnesses.jev.candidates import CANDIDATE_POLICY, observation_candidates
from realtime_framework.harnesses.jev.game_candidates import (
    GAME_CANDIDATE_POLICY,
    add_game_candidates,
)
from realtime_framework.providers.decisions.questions import (
    GAME_INSTRUCTIONS,
    GUI_INSTRUCTIONS,
    TEXT_COMPOSITION_POLICY,
)
from realtime_framework.sandbox.observation.interactables import format_interactables
from realtime_framework.sandbox.observation.som import OBSERVATION_INTERACTABLES_HEADER


def choice_observation(task_instruction, observation, execution, *, frame_prompt, game):
    """Write the observation JSON into the bound ``observation``'s prompt and return the
    decision's answer options and selection material.

    Only the current observation's candidates are executable: they are the answer options, and
    the rest of its state is material for the selection flow only.
    """
    delivered = (execution or {}).get("delivered") or {}
    marks = (execution or {}).get("marks") or {}
    active_window = next(
        (window for window in delivered.get("windows", []) if window.get("active")),
        {},
    )
    candidates, focused_input, text_composition = observation_candidates(
        task_instruction, marks, active_window
    )
    if game:
        add_game_candidates(candidates, marks, delivered, observation.geometry)
    state = {
        "observation_id": observation.observation_id,
        "timestamp": observation.observation.timestamp,
        "mark_legend": (execution or {}).get("text", ""),
        "full_som": format_interactables(delivered) if delivered else "",
        "som_truncated": False,
        "ax_tree": active_window.get("ax_tree") or "",
        "ax_capture": active_window.get("ax_capture") or {},
        "focused_input": focused_input,
        "candidate_count": len(candidates),
        "candidate_policy": GAME_CANDIDATE_POLICY if game else CANDIDATE_POLICY,
        "game_choices": game,
        "text_composition_policy": TEXT_COMPOSITION_POLICY,
        "text_composition": text_composition,
        "candidates": candidates,
    }
    # Candidates and the composition inventory are current-only Choice criteria and stay
    # in the decision audit; retained history keeps the observation state.
    visible = {
        key: value for key, value in state.items()
        if key not in {"candidates", "text_composition", "game_choices"}
    }
    observation.prompt = (
        frame_prompt + OBSERVATION_INTERACTABLES_HEADER + json.dumps(visible, ensure_ascii=False)
    )
    choice = ChoiceRequirements(
        instructions=GAME_INSTRUCTIONS if state["game_choices"] else GUI_INSTRUCTIONS,
        candidates=tuple(
            (key, Candidate(value["description"], value["action"], value.get("audit")))
            for key, value in state["candidates"].items()
        ),
        candidate_policy=state["candidate_policy"],
    )
    selection = SelectionMaterial(
        observation_id=state["observation_id"],
        game_choices=state["game_choices"],
        focused_input=state["focused_input"],
        text_composition=state["text_composition"],
    )
    return choice, selection
