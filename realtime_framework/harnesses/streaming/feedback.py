"""What the model is told of each streaming outcome: the text of its execution feedback block
(phase2.md, native streaming). The streaming policy commits an outcome for every action, the three
refusals included, and one for a reply that came to no action (evaluation/kernel/streaming.py).

The text is the turn-based execution result (harnesses/default/harness.py commit_action): what
ran, what was asked for, and the status. An action that did not run ran nothing; each refusal says
why, so that the model can tell the action it sent was never applied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from realtime_framework.entity.actions import Action
from realtime_framework.harnesses.default.context import action_result_note
from realtime_framework.harnesses.default.frames import (
    FRAME_NOT_IN_TABLE,
    INVALID_FRAME_SYNTAX,
    INVALID_GROUNDING_FRAME,
    MISSING_FRAME,
)

if TYPE_CHECKING:  # the agent does not import the evaluation package at run time
    from realtime_framework.evaluation.kernel.events import Outcome

# The statuses of an action that ran: one the environment applied, and a wait, which ends where it
# arrives (evaluation/kernel/streaming.py).
_RAN = {"applied", "wait"}
# The policy's statuses of an action a newer one took the place of (SUPERSEDED) and of one whose
# reply was given up (CANCELLED_GENERATION), and of a reply that came to no action.
_NOTES = {
    "superseded": (
        "NOTE: this action was NOT executed: a newer action arrived while another one was "
        "executing and took its place."
    ),
    "cancelled_generation": (
        "NOTE: this action was NOT executed because the reply it came in was cancelled."
    ),
    "none": "NOTE: your reply had no valid action, so nothing was executed.",
}
_FRAME_NOTE = (
    "NOTE: this action was NOT executed because its frame field {why}. Set frame to the FRAME "
    "number drawn on the screenshot the action is based on."
)
_FRAME_ERRORS = {
    MISSING_FRAME: "is missing",
    INVALID_FRAME_SYNTAX: "is not an integer",
    FRAME_NOT_IN_TABLE: "names no frame that was sent",
}


def action_text(action: Action) -> str:
    """An action as the action record writes it (harnesses/default/harness.py stream_generate)."""
    return action.action_type.value + (f" {action.parameters}" if action.parameters else "")


def execution_feedback(outcome: Outcome, requested: str) -> str:
    """The feedback on one outcome. ``requested`` is what the model asked for, as action_text
    writes it, or "none": the outcome holds only what ran, and no action at all when the frame
    the action names cannot be found."""
    status = outcome.status
    actual = action_text(outcome.action) if status in _RAN else "none"
    text = f"EXECUTION RESULT: actual={actual}; requested={requested}; status={status}."
    if status == INVALID_GROUNDING_FRAME:
        return f"{text} {_FRAME_NOTE.format(why=_FRAME_ERRORS[outcome.detail])}"
    if outcome.detail:
        text += f" Detail: {outcome.detail}"
    note = _NOTES.get(status) or action_result_note(status)
    return f"{text} {note}" if note else text
