"""Temporary bridges from the canonical request and turn records to what each encoder reads today.

Phase 1 step 4 moves the encoders onto the canonical request and deletes this module.
"""

from __future__ import annotations

import base64
from copy import deepcopy

from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER


def frame_text(prefix: str, prompt: str) -> str:
    """A session frame's text: the pending execution feedback, then the frame prompt."""
    return f"{prefix}\n{prompt}" if prefix else prompt


def session_frame_text(observation) -> str:
    """The prompt the live drivers send for one ProviderObservation."""
    return frame_text(observation.feedback_prefix, observation.prompt)


def chat_messages(request) -> tuple[dict, ...]:
    """The OpenAI-format message list the chat and Decision encoders read.

    Every ledger is a user message at index 1, whatever its intended role; Luna's transcript drops
    the roles itself. Live tool exchanges have no message form here.
    """
    messages = []
    prefix = ""
    for item in request.items:
        block = item.block
        if block.kind == "system_instruction":
            messages.append({"role": "system", "content": block.text})
        elif block.kind == "action_ledger":
            messages.append(
                {"role": "user", "content": ARCHIVED_ACTION_LEDGER_HEADER + "\n" + block.text}
            )
        elif item.intent.position == "frame_prefix":
            prefix = block.text
        elif block.kind == "frame":
            if block.folded:
                messages.append({"role": "user", "content": block.text})
            else:
                encoded = base64.b64encode(block.image.data).decode("ascii")
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": frame_text(prefix, block.text)},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block.image.mime_type};base64,{encoded}",
                                    "detail": block.image.detail,
                                },
                            },
                        ],
                    }
                )
            prefix = ""
        elif block.kind == "model_output":
            if block.provider_message is not None:
                messages.append(deepcopy(block.provider_message))
            else:
                message = {"role": "assistant", "content": block.text}
                if block.reasoning:
                    message["reasoning_content"] = block.reasoning
                messages.append(message)
        elif block.kind == "execution_feedback":
            messages.append({"role": "user", "content": block.text})
    return tuple(messages)
