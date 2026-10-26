"""Present a stateless Decisions protocol to the realtime runtime as one generation per decision.

The canonical request of each decision carries the current observation's candidates, the material
of its follow-up questions and the history; the protocol receives Choice questions over the
history rendered as Decisions state.
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
    RetryCancelled,
    SyncDisposition,
    TransportFailure,
    TransportFailureError,
    exception_chain,
)
from realtime_framework.providers.decisions.selection import SELECTION_POLICY, ChoiceDecision

# The logged history labels predate the context policy's names for the two forms.
HISTORY_REPLAY_LABELS = {"turns": "canonical", "action_digest": "action_digest"}


class DecisionEncoder:
    """The RealtimeDriver half of a Decisions encoder.

    A registered Decision API subclasses it with its protocol (``decide``,
    ``classify_question_error``, ``bind_trace``, ``close``), capabilities (with ``max_choices``),
    deployment and the ``clock`` its constructor takes, so the registered class is the complete
    driver.
    """

    async def connect(self, request):
        return None

    @staticmethod
    def classify_transport_error(exc):
        """Each Choice question was already retried; keep its reason, never retry the decision."""
        for current in exception_chain(exc):
            if isinstance(current, TransportFailureError):
                return current.failure
        return TransportFailure(FATAL)

    async def send_observation(self, observation, history):
        return {"observation_id": observation.observation_id}

    async def start_generation(self, request):
        decision = ChoiceDecision(self, request.canonical, trace=self._trace, clock=self.clock)
        try:
            metadata = await decision.run()
        except (Exception, RetryCancelled) as exc:
            # The questions answered before the decision failed or was given up on; the harness
            # counts them as its usage (harnesses/default/harness.py, decisions 19 and 37).
            if decision.calls:
                exc.partial_usage = dict(decision.usage)
            raise
        metadata["history_form"] = request.canonical.history_form
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
                "choice_history_replay": HISTORY_REPLAY_LABELS[data["history_form"]],
                "choice_max_choices": self.capabilities.max_choices,
                "jev_transport_retries": data["retries"],
                "jev_calls": data["jev_calls"],
                "jev_candidate_count": data["candidate_count"],
                "jev_candidate_policy": data["candidate_policy"],
                "jev_candidate_criteria_bytes": data["candidate_criteria_bytes"],
                "jev_selected_candidate": data["selected_candidate"],
                "jev_selected_candidate_audit": data["selected_candidate_audit"],
                "jev_selection_policy": SELECTION_POLICY,
                "jev_context_splits": data["context_splits"],
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
