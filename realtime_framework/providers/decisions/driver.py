"""Present a stateless ChoiceClient to the realtime runtime as one generation per decision.

The harness binds each observation's structured state (candidates, focused input, composition
inventory) by observation_id; the client receives Choice questions over text and optional images.
"""

from __future__ import annotations

import json

from realtime_framework.entity.events import (
    GenerationBoundary,
    ProviderHistoryState,
    TextDelta,
    UsageSnapshot,
)
from realtime_framework.entity.model import (
    FATAL,
    GenerationHandle,
    HistorySyncResult,
    InterruptResult,
    SyncDisposition,
    TransportFailure,
    TransportFailureError,
    exception_chain,
)
from realtime_framework.providers.decisions.history import decision_messages
from realtime_framework.providers.decisions.selection import (
    SELECTION_POLICY,
    SHADOW,
    ChoiceDecision,
    shadow_check,
)


class ChoiceDecisionDriver:
    def __init__(self, client, observation_states: dict[str, dict]):
        self.client = client
        self.capabilities = client.capabilities
        self.observation_states = observation_states
        self.observation = None
        if self.capabilities.history_replay != "canonical":
            # A digest is not the canonical request, so the runtime cannot measure or archive it.
            self.request_context_status = None

    async def request_context_status(self, request):
        return await self.client.request_context_status(request)

    def bind_trace(self, trace) -> None:
        self._trace = trace
        self.client.bind_trace(trace)

    async def connect(self, request):
        return None

    @staticmethod
    def classify_transport_error(exc):
        """Each Choice question was already retried; keep its reason, never retry the decision."""
        for current in exception_chain(exc):
            if isinstance(current, TransportFailureError):
                return current.failure
        return TransportFailure(FATAL)

    async def send_observation(self, observation):
        state = self.observation_states.get(observation.observation_id)
        if state is None or state["observation_id"] != observation.observation_id:
            raise ValueError("Action choices belong to a different observation")
        self.observation = state
        return {"observation_id": observation.observation_id}

    async def start_generation(self, request):
        if request.clock_mode != "wall":
            raise ValueError("Choice Decisions supports wall clock only")
        if self.observation is None:
            raise RuntimeError("Choice Decisions requires a current observation")
        messages = decision_messages(request.canonical)
        labels = {"turns": "canonical", "action_digest": "action_digest"}
        shadow_check(
            "label", labels[request.canonical.history_form] == self.capabilities.history_replay
        )
        SHADOW["labels"] += 1
        metadata = await ChoiceDecision(
            self.client, self.observation, messages, request.canonical, trace=self._trace
        ).run()
        return GenerationHandle(request.generation_id, metadata=metadata)

    async def events(self, generation):
        data = generation.metadata
        serialization = "deterministic_choice_mapping"
        if data["text_composition"]:
            serialization = "deterministic_choice_composition"
        elif data["pointer_composition"]:
            serialization = "deterministic_pointer_composition"
        elif data["key_composition"]:
            serialization = "deterministic_key_composition"
        yield UsageSnapshot(generation.generation_id, 1, "jev.usage", data["usage"])
        # A deterministic serialization of model choices, including a composed draft.
        yield TextDelta(
            generation.generation_id, 2, "jev.choice", json.dumps(data["action"]), "visible"
        )
        yield GenerationBoundary(
            generation.generation_id,
            3,
            "jev.done",
            "turn_complete",
            ProviderHistoryState.EXACT,
            {
                "jev_response": data["jev_response"],
                "jev_question": data["jev_question"],
                "observation_id": data["observation_id"],
                "action_serialization": serialization,
                "observation_modality": (
                    "screenshot_plus_ax_tree_plus_interactables_text"
                    if self.capabilities.observation_modality == "screenshot"
                    else "ax_tree_plus_interactables_text"
                ),
                "choice_protocol_family": self.capabilities.protocol_family,
                "choice_history_replay": self.capabilities.history_replay,
                "choice_max_choices": self.client.max_choices,
                "jev_transport_retries": data["retries"],
                "jev_calls": data["jev_calls"],
                "jev_candidate_count": data["candidate_count"],
                "jev_candidate_policy": data["candidate_policy"],
                "jev_candidate_criteria_bytes": data["candidate_criteria_bytes"],
                "jev_selected_candidate": data["selected_candidate"],
                "jev_selected_candidate_audit": data["selected_candidate_audit"],
                "jev_selection_policy": SELECTION_POLICY,
                "jev_context_splits": data["context_splits"],
                "jev_history_reductions": data["history_reductions"],
                "jev_text_composition": data["text_composition"],
                "jev_pointer_composition": data["pointer_composition"],
                "jev_key_composition": data["key_composition"],
            },
        )

    async def interrupt(self, generation):
        return InterruptResult(ProviderHistoryState.EXACT)

    async def sync_history(self, request):
        return HistorySyncResult(
            SyncDisposition.EXACT,
            ProviderHistoryState.EXACT,
            detail="stateless request uses canonical retained history",
        )

    async def close(self):
        await self.client.close()
