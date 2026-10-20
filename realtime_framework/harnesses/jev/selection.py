"""One JEV wall-clock decision as a sequence of Choice questions.

JEV never generates text. Each question's criteria are candidate actions (or text edits) of the
current observation. Candidate sets beyond the client's max_choices are compared in groups and
their winners compared in a further Choice. All calls belong to the same wall-clock decision.
"""

from __future__ import annotations

import asyncio
import json
import time

from realtime_framework.entity.model import (
    TRANSPORT_RETRY,
    ChoiceContextLimitError,
    TransportFailureError,
)
from realtime_framework.harnesses.jev.composition import (
    TEXT_COMPOSITION_POLICY,
    composition_candidates,
    unicode_candidates,
)
from realtime_framework.harnesses.jev.context import compact_ledger_messages
from realtime_framework.harnesses.jev.game_candidates import (
    POINTER_COMPOSITION_POLICY,
    coordinate_choices,
)
from realtime_framework.harnesses.jev.prompts import (
    COMPOSE_INSTRUCTIONS,
    GAME_INSTRUCTIONS,
    GROUP_NOTE,
    GUI_INSTRUCTIONS,
    UNICODE_INSTRUCTIONS,
)

SELECTION_POLICY = "exhaustive_groups_255_provider_split_then_winners_pointer_v6"


class ChoiceDecision:
    """Resolve one observation's candidates into one executable action, with a full trace."""

    def __init__(self, client, observation_state, messages):
        self.client = client
        self.observation = observation_state
        # Canonical replay is already bounded by the runtime's retention and context checks.
        self.canonical = client.capabilities.history_replay == "canonical"
        self.history = list(messages) if self.canonical else compact_ledger_messages(messages)
        self.calls = []
        self.retries = []
        self.context_splits = []
        self.history_reductions = []
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    async def run(self) -> dict:
        candidates = self.observation["candidates"]
        if not candidates:
            raise ValueError("JEV requires at least one candidate action")
        instructions = GAME_INSTRUCTIONS if self.observation.get("game_choices") else GUI_INSTRUCTIONS
        choice = await self._choose(candidates, instructions, phase="gui_action")
        # The deciding action-choice call; composition calls that follow stay in jev_calls.
        decisive = self.calls[-1]
        action = candidates[choice]["action"]
        composition = None
        pointer_composition = None
        key_composition = None
        if action["action"] == "compose_text":
            action, composition = await self._compose_text()
        elif action["action"] == "compose_pointer":
            action, pointer_composition = await self._compose_pointer(action)
        elif action["action"] == "compose_hold":
            action, key_composition = await self._compose_hold(action)
        return {
            "action": action,
            "usage": self.usage,
            "jev_response": decisive["response"],
            "jev_question": decisive["question"],
            "jev_calls": self.calls,
            "candidate_count": len(candidates),
            "candidate_criteria_bytes": sum(
                len(json.dumps({key: value["description"]}, ensure_ascii=False).encode())
                for key, value in candidates.items()
            ),
            "selected_candidate": choice,
            "selected_candidate_audit": candidates[choice].get("audit"),
            "context_splits": self.context_splits,
            "history_reductions": self.history_reductions,
            "observation_id": self.observation["observation_id"],
            "retries": self.retries,
            "text_composition": composition,
            "pointer_composition": pointer_composition,
            "key_composition": key_composition,
            "candidate_policy": self.observation.get("candidate_policy"),
        }

    async def _compose_hold(self, spec):
        keys = []
        for index in range(2):
            options = {str(i): {"description": f"Hold key {key}", "key": key}
                       for i, key in enumerate(spec["keys"]) if key not in keys}
            choice = await self._choose(
                options, f"Choose key {index + 1} of a simultaneous two-key game hold. "
                f"Already chosen: {keys}. Use the visible game controls and task instructions.",
                phase=f"hold_key_{index + 1}",
            )
            keys.append(options[choice]["key"])
        durations = {str(value): {"description": f"Hold for {value:g} seconds", "value": value}
                     for value in (0.1, 0.5, 1.0)}
        choice = await self._choose(
            durations, f"Choose how long to hold {'+'.join(keys)} together before release.",
            phase="hold_duration",
        )
        duration = durations[choice]["value"]
        return {"action": "hold_key", "key": "+".join(keys), "duration": duration}, {
            "policy": "declared_key_pair_then_duration_v1", "keys": keys, "duration": duration,
        }

    async def _compose_pointer(self, spec):
        """Resolve coordinates within this observation; all calls share the decision deadline."""
        coordinates = []
        target = spec.get("region") or "the full current screenshot"
        for axis in ("x", "y"):
            instruction = (
                f"Choose the {axis} coordinate for {spec['operation']} within {target}. "
                "Coordinates are normalized integers 0..1000: x runs left to right and "
                f"y runs top to bottom. Already selected coordinates: {coordinates}. "
                "Use the newest screenshot. This chooses a parameter, not another GUI action."
            )
            coarse = coordinate_choices()
            chosen = await self._choose(coarse, instruction, phase=f"pointer_{axis}_range")
            base = coarse[chosen]["value"]
            if base == 1000:
                value = base
            else:
                fine = coordinate_choices(base)
                chosen = await self._choose(fine, instruction, phase=f"pointer_{axis}_value")
                value = fine[chosen]["value"]
            coordinates.append(value)
        u, v = coordinates
        action = {"action": spec["operation"]}
        if spec.get("region"):
            action["target"] = f"{spec['region']}@{u},{v}"
        else:
            width, height = spec["image_size"]
            action.update(x=min(width - 1, round(width * u / 1000)),
                          y=min(height - 1, round(height * v / 1000)))
        return action, {"policy": POINTER_COMPOSITION_POLICY, "spec": spec,
                        "coordinates": coordinates}

    async def _choose(self, candidates, instructions, *, phase, text_state=None):
        """Compare every remaining candidate; winners of separate groups meet in a new Choice.

        Groups use the client's option boundary. A context rejection bisects that group
        losslessly. Probabilities are never compared across different questions.
        """
        remaining = list(candidates)
        round_index = 0
        while True:
            winners = []
            size = self.client.max_choices
            groups = [remaining[start : start + size] for start in range(0, len(remaining), size)]
            group_index = 0
            while groups:
                group = groups.pop(0)
                question = {
                    "type": "choice",
                    "instructions": instructions + (GROUP_NOTE if len(group) < len(remaining) else ""),
                    "criteria": {key: candidates[key]["description"] for key in group},
                }
                started = time.monotonic()
                try:
                    reply = await self._ask(question, phase=phase, text_state=text_state)
                except ChoiceContextLimitError:
                    if len(group) == 1:
                        raise
                    middle = len(group) // 2
                    self.context_splits.append(
                        {
                            "phase": phase,
                            "round": round_index,
                            "candidates": list(group),
                            "latency_s": time.monotonic() - started,
                        }
                    )
                    groups[0:0] = [group[:middle], group[middle:]]
                    continue
                choice = reply.answer.get("choice")
                if reply.answer.get("type") != "choice" or choice not in group:
                    raise ValueError("JEV returned an action outside the current candidate group")
                for key in self.usage:
                    self.usage[key] += reply.usage[key]
                self.calls.append(
                    {
                        "phase": phase,
                        "round": round_index,
                        "group": group_index,
                        "question": question,
                        "response": reply.response,
                        "latency_s": time.monotonic() - started,
                        **({"text_state": dict(text_state)} if text_state is not None else {}),
                    }
                )
                winners.append(choice)
                group_index += 1
            if len(winners) == 1:
                return winners[0]
            if len(winners) == len(remaining):
                raise ChoiceContextLimitError(
                    "JEV cannot compare even two candidates within the context limit"
                )
            remaining = winners
            round_index += 1

    async def _ask(self, question, *, phase, text_state=None):
        """Ask one question; if even one option overflows a digest, drop its oldest history.

        The digest state is [system, *older history, current observation]. The current observation
        and instructions are never removed; every reduction is recorded. Canonical history is
        never reduced here: the runtime archives it before the request.
        """
        history = list(self.history)
        while True:
            state = {"history": history}
            if self.observation.get("images"):
                state["images"] = self.observation["images"]
            if text_state is not None:
                state["text_composition"] = text_state
            try:
                return await self._decide_with_retries(state, question, phase=phase)
            except ChoiceContextLimitError:
                droppable = len(history) - 2
                if self.canonical or len(question["criteria"]) > 1 or droppable <= 0:
                    raise
                removed = max(1, droppable // 2)
                history = [history[0], *history[1 + removed :]]
                self.history_reductions.append(
                    {"phase": phase, "removed_messages": removed, "remaining": len(history)}
                )

    async def _decide_with_retries(self, state, question, *, phase):
        """One question under the shared transport retry policy; each attempt is recorded."""
        key = f"{self.observation['observation_id']}:{len(self.calls)}"
        retry = 0
        while True:
            try:
                return await self.client.decide(state, question)
            except ChoiceContextLimitError:
                raise
            except Exception as exc:
                retry += 1
                failure = self.client.classify_transport_error(exc)
                delay = TRANSPORT_RETRY.delay_s(retry, key, failure)
                if delay is None:
                    # Name the question so an error report shows which step failed.
                    cause = RuntimeError(f"{phase} question {len(self.calls)}: {exc!r}")
                    raise TransportFailureError(failure, cause, retry - 1) from exc
                self.retries.append(
                    {
                        "call": len(self.calls),
                        "phase": phase,
                        "kind": failure.kind,
                        "delay_s": delay,
                        "detail": repr(exc)[:300],
                    }
                )
                await asyncio.sleep(delay)

    async def _compose_text(self):
        inventory = self.observation.get("text_composition")
        if not inventory or not self.observation.get("focused_input"):
            raise ValueError("JEV text composition requires a focused observation-bound input")
        candidates = composition_candidates(inventory)
        draft, edits = "", []
        started = time.monotonic()
        # No forced completion, repetition guard or answer repair. The runtime/harness
        # deadline bounds this generation, exactly as it bounds a generative model's output.
        while True:
            state = {
                "draft": draft,
                "step": len(edits),
                "focused_input": self.observation["focused_input"],
            }
            choice = await self._choose(
                candidates, COMPOSE_INSTRUCTIONS, phase="text_edit", text_state=state
            )
            selected = candidates[choice]
            operation = selected["operation"]
            edit = {"choice": choice, "operation": operation, "length_before": len(draft)}
            if operation == "append":
                draft += selected["text"]
                edit.update(text=selected["text"], sources=selected.get("sources", []))
            elif operation == "delete":
                draft = draft[:-1]
            elif operation == "clear":
                draft = ""
            elif operation == "unicode":
                prefix = ""
                while len(prefix) < 6:
                    hex_choices = unicode_candidates(prefix)
                    digit = await self._choose(
                        hex_choices, UNICODE_INSTRUCTIONS, phase="text_unicode",
                        text_state={**state, "unicode_hex_prefix": prefix},
                    )
                    if hex_choices[digit]["operation"] == "back":
                        break
                    prefix += hex_choices[digit]["text"]
                edit["unicode_hex"] = prefix
                if len(prefix) == 6:
                    edit["text"] = chr(int(prefix, 16))
                    draft += edit["text"]
            edit["length_after"] = len(draft)
            edits.append(edit)
            if operation in {"finish", "cancel"}:
                action = (
                    {"action": "type", "text": draft}
                    if operation == "finish"
                    else {"action": "wait"}
                )
                return action, {
                    "policy": TEXT_COMPOSITION_POLICY,
                    "status": operation,
                    "draft": draft,
                    "edits": edits,
                    "candidate_count": len(candidates),
                    "latency_s": time.monotonic() - started,
                }
