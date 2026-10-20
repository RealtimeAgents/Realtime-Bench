"""Temporary bridges from the canonical request and turn records to what each encoder reads today.

Phase 1 step 4 moves the encoders onto the canonical request and deletes this module.
"""

from __future__ import annotations


def frame_text(prefix: str, prompt: str) -> str:
    """A session frame's text: the pending execution feedback, then the frame prompt."""
    return f"{prefix}\n{prompt}" if prefix else prompt


def session_frame_text(observation) -> str:
    """The prompt the live drivers send for one ProviderObservation."""
    return frame_text(observation.feedback_prefix, observation.prompt)
