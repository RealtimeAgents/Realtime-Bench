"""JEV-only lossless history serialization; never controls candidate eligibility."""
from __future__ import annotations

import json

from realtime_framework.entity.context_policy import (
    JEV_CURRENT_PREFIX,
    JEV_HISTORY_PREFIX,
    JEV_LEDGER_HEADER,
)
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER
from realtime_framework.providers.legacy import chat_messages
from realtime_framework.sandbox.observation.som import OBSERVATION_INTERACTABLES_HEADER


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


def compact_ledger_messages(messages) -> list[dict]:
    """Compact the shared ledger only in JEV request states, without mutating it."""
    result = []
    prefix = ARCHIVED_ACTION_LEDGER_HEADER + "\n"
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and content.startswith(prefix):
            records = [json.loads(line) for line in content[len(prefix):].splitlines() if line]
            records = compact_repeated_records(records, ("decision", "observed_at_tick"))
            content = (
                JEV_LEDGER_HEADER + "\n"
                + "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")) for r in records)
            )
            message = {**message, "content": content}
        result.append(message)
    return result


def observation_messages(system_instruction: str, turns) -> tuple[dict, ...]:
    """Action-digest state: the executed action/feedback ledger plus the current observation.

    Older screenshots, SoM and choices are not resent.
    """
    messages = [{"role": "system", "content": system_instruction}]
    if not turns:
        return tuple(messages)
    history = []
    for turn in turns[:-1]:
        if turn.output or turn.execution_feedback:
            history.append({
                "observation_id": turn.observation.observation_id,
                "action": turn.output,
                "execution_feedback": turn.execution_feedback or "",
            })
    if history:
        messages.append({"role": "user", "content": JEV_HISTORY_PREFIX
                         + json.dumps(compact_repeated_records(history, ("observation_id",)),
                                      ensure_ascii=False)})
    latest = turns[-1]
    # The harness omits live candidates and the composition vocabulary from the prompt; they
    # are sent as native Choice criteria.
    current = json.loads(latest.observation.prompt.partition(OBSERVATION_INTERACTABLES_HEADER)[2])
    messages.append({"role": "user", "content": JEV_CURRENT_PREFIX
                     + json.dumps(current, ensure_ascii=False)})
    return tuple(messages)


def digest_messages(request) -> tuple[dict, ...]:
    """Action-digest state: the executed action/feedback ledger plus the current observation.

    Older screenshots, SoM and choices are not resent.
    """
    messages = []
    history = []
    for item in request.items:
        block = item.block
        if block.kind == "system_instruction":
            messages.append({"role": "system", "content": block.text})
        elif block.kind == "action_ledger":
            messages.append(
                {"role": "user", "content": ARCHIVED_ACTION_LEDGER_HEADER + "\n" + block.text}
            )
        elif block.kind == "model_output":
            history.append({"observation_id": block.based_on, "action": block.text})
        elif block.kind == "execution_feedback":
            history[-1]["execution_feedback"] = block.text
        elif block.kind == "frame":
            if history:
                compacted = compact_repeated_records(history, ("observation_id",))
                messages.append({"role": "user", "content": JEV_HISTORY_PREFIX
                                 + json.dumps(compacted, ensure_ascii=False)})
            # The harness omits live candidates and the composition vocabulary from the prompt;
            # they are sent as native Choice criteria.
            current = json.loads(block.text.partition(OBSERVATION_INTERACTABLES_HEADER)[2])
            messages.append({"role": "user", "content": JEV_CURRENT_PREFIX
                             + json.dumps(current, ensure_ascii=False)})
    return tuple(messages)


def decision_messages(request) -> tuple[dict, ...]:
    """The message list ChoiceDecision reads: canonical turns or the action digest."""
    if request.history_form == "action_digest":
        return digest_messages(request)
    return chat_messages(request)
