"""Decisions state from the canonical request: Luna's transcript of the retained turns and JEV's
lossless action digest. Labels and headers come from the policy on the request; nothing here
controls candidate eligibility.
"""
from __future__ import annotations

import base64
import json

from realtime_framework.sandbox.observation.som import FRAME_INTERACTABLES_HEADER


def compact_repeated_records(records: list[dict], varying: tuple[str, ...]) -> list[dict]:
    """Preserve every action and occurrence while encoding consecutive repetitions."""
    output = []
    index = 0
    while index < len(records):
        common = {k: v for k, v in records[index].items() if k not in varying}
        end = index + 1
        while end < len(records) and common == {
            k: v for k, v in records[end].items() if k not in varying
        }:
            end += 1
        if end == index + 1:
            output.append(records[index])
        else:
            occurrences = {}
            for key in varying:
                values = [r.get(key) for r in records[index:end]]
                if all(type(v) is int for v in values):
                    step = values[1] - values[0]
                    if all(v == values[0] + i * step for i, v in enumerate(values)):
                        occurrences[key] = {"start": values[0], "step": step, "count": len(values)}
                        continue
                occurrences[key] = values
            output.append({**common, "repeat_count": end - index, "occurrences": occurrences})
        index = end
    return output


# JEV message roles of the neutral item roles.
JEV_ROLES = {"instruction": "system", "input": "user"}


def render_transcript(request) -> tuple[list, list[str]]:
    """Luna's state: every block in order, each image a top-level entry at its causal position.

    Decisions state has no roles, and image parts must be top-level entries of the state array.
    The canonical order keeps older turns a stable prefix for prompt caching; only the current
    question's state follows the newest observation. Observations and chosen actions carry
    plain labels: role-tagged turns make OpenAI refuse many-option questions more often.
    Returns the entries up to the last image and the text after it.
    """
    texts = request.policy.texts
    entries, text = [], []
    for item in request.items:
        block = item.block
        if block.kind == "frame":
            text.append(texts.luna_observation_label + block.text)
            if block.image is not None:
                encoded = base64.b64encode(block.image.data).decode("ascii")
                entries.append("\n".join(text))
                entries.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.image.mime_type};base64,{encoded}",
                            "detail": block.image.detail,
                        },
                    }
                )
                text = []
        elif block.kind == "model_output":
            text.append(texts.luna_action_label + block.text)
        elif block.kind == "action_ledger":
            text.append(texts.ledger_header + "\n" + block.text)
        else:  # system instruction, execution feedback
            text.append(block.text)
    return entries, text


def transcript_state(transcript: tuple[list, list[str]], extra: dict) -> list:
    """One question's state: the transcript, then the question's own state as JSON text."""
    entries, text = transcript
    if extra:
        text = [*text, json.dumps(extra, ensure_ascii=False)]
    return [*entries, "\n".join(text)] if text else list(entries)


def render_digest(request, *, compact_ledger: bool = True) -> list[dict]:
    """JEV's history: the system text, the archived ledger, a digest of the older turns' actions
    and feedback, and the current observation after the feedback its frame opens with. Older
    screenshots, SoM and choices are not resent.

    The ledger is sent compacted; the trajectory summary shows it as the runtime keeps it.
    """
    texts = request.policy.texts
    messages, history, prefix = [], [], ""
    for item in request.items:
        block = item.block
        if block.kind == "system_instruction":
            messages.append({"role": JEV_ROLES[item.intent.role], "content": block.text})
        elif block.kind == "action_ledger":
            if compact_ledger:
                lines = [line for line in block.text.splitlines() if line]
                # Each archived action is a JSON object; the note for omitted ones is not
                # (harnesses/default/context.py RealtimeRuntime.ledger_block).
                notes = [line for line in lines if not line.startswith("{")]
                records = [json.loads(line) for line in lines if line.startswith("{")]
                records = compact_repeated_records(records, ("decision", "observed_at_tick"))
                record_lines = [
                    json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    for r in records
                ]
                content = "\n".join([texts.jev_ledger_header, *notes, *record_lines])
            else:
                content = texts.ledger_header + "\n" + block.text
            messages.append({"role": JEV_ROLES[item.intent.role], "content": content})
        elif block.kind == "model_output":
            history.append({"observation_id": block.based_on, "action": block.text})
        elif item.intent.position == "frame_prefix":
            prefix = block.text + "\n"
        elif block.kind == "execution_feedback" and block.observation_id is not None:
            # What an older frame opened with: the feedback on a decision no turn records.
            history.append(
                {"observation_id": block.observation_id, "execution_feedback": block.text}
            )
        elif block.kind == "execution_feedback":
            history[-1]["execution_feedback"] = block.text
        elif block.kind == "frame":
            if history:
                compacted = compact_repeated_records(history, ("observation_id",))
                messages.append(
                    {
                        "role": "user",
                        "content": texts.jev_history_prefix
                        + json.dumps(compacted, ensure_ascii=False),
                    }
                )
            # The harness omits live candidates and the composition vocabulary from the prompt;
            # they are sent as native Choice criteria.
            current = json.loads(block.text.partition(FRAME_INTERACTABLES_HEADER)[2])
            messages.append(
                {
                    "role": JEV_ROLES[item.intent.role],
                    "content": prefix
                    + texts.jev_current_prefix
                    + json.dumps(current, ensure_ascii=False),
                }
            )
    return messages
