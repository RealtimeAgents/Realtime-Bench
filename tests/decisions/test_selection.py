"""The selection flow of one Choice decision (providers/decisions/selection.py ChoiceDecision),
with a fake client in place of the HTTP ones: which questions it asks, in which order, and when it
gives up. What a client sends and how it reads the reply is tested in test_client.py.
"""

from __future__ import annotations

import asyncio

import pytest

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.model import (
    FATAL,
    CanonicalTurn,
    ChoiceContextLimitError,
    ChoiceReply,
    ProviderObservation,
    TransportFailure,
)
from realtime_framework.entity.records import TraceEvent
from realtime_framework.entity.request import (
    AnswerRequirements,
    CallParameters,
    Candidate,
    ChoiceRequirements,
    SelectionMaterial,
)
from realtime_framework.harnesses.default.request import build_request, system_block
from realtime_framework.providers.decisions.questions import GUI_INSTRUCTIONS
from realtime_framework.providers.decisions.selection import ChoiceDecision
from realtime_framework.providers.openai_decisions import OpenAIDecisionsClient

MODEL = "openai/gpt-6-luna-decisions"
NO_PROGRESS = "JEV cannot compare even two candidates within the context limit"
# Far more questions than any decision below asks; past it the client fails the question.
QUESTION_CAP = 100


class _Trace:
    def __init__(self):
        self.events = []

    def capture(self, event, payload, body=None):
        self.events.append((event, payload))


class _OneOptionClient:
    """A client whose context limit leaves room for one option: a question offering two or more is
    rejected with ChoiceContextLimitError, as the HTTP clients reject one on a 413, and a question
    offering one is answered with it."""

    capabilities = OpenAIDecisionsClient.capabilities

    def __init__(self):
        self.sizes = []

    async def decide(self, state, question):
        criteria = question["criteria"]
        self.sizes.append(len(criteria))
        if len(self.sizes) > QUESTION_CAP:
            raise RuntimeError(f"more than {QUESTION_CAP} questions")
        if len(criteria) > 1:
            raise ChoiceContextLimitError("Decisions request exceeds its context limit: Too large")
        (choice,) = criteria
        answer = {
            "type": "choice",
            "choice": choice,
            "probabilities": {choice: 1.0},
            "confidence": 1.0,
        }
        return ChoiceReply(answer, {"input_tokens": 1000, "output_tokens": 1}, {})

    @staticmethod
    def classify_question_error(exc):
        return TransportFailure(FATAL)


def _request(count: int):
    """Luna's canonical request for one frame offering ``count`` candidate clicks."""
    observation = ProviderObservation(
        observation=Observation(b"raw", 100.0, 0.0, {"observation_id": "obs-0"}),
        prompt="Choose.",
        image_bytes=b"png",
        mime_type="image/png",
        image_detail="auto",
        geometry={},
    )
    choice = ChoiceRequirements(
        GUI_INSTRUCTIONS,
        tuple(
            (f"c{i}", Candidate(f"click P{i}", {"action": "click", "target": f"P{i}"}))
            for i in range(count)
        ),
        "replay_candidates",
    )
    return build_request(
        current_context_policy().for_path("openai_decisions", {}, None),
        system_block("system", None),
        (CanonicalTurn(observation, decision=0),),
        None,
        AnswerRequirements("choice", "replay_choice", choice=choice),
        CallParameters(MODEL, "fixed", None),
        SelectionMaterial("obs-0", False, None, None),
    )


@pytest.mark.parametrize(
    "count, sizes",
    [
        (2, [2, 1, 1]),
        (3, [3, 1, 2, 1, 1]),
        (5, [5, 2, 1, 1, 3, 1, 2, 1, 1]),
    ],
)
def test_a_round_without_progress_ends_the_decision(count, sizes):
    """When even two options exceed the context limit, every group is bisected down to single
    options, each of which wins its own question, so the round ends with as many winners as it
    began with candidates (ChoiceDecision._choose). The next round would ask the same questions
    again, forever; the decision ends instead with a context-limit error, after the one round."""
    client, trace = _OneOptionClient(), _Trace()
    decision = ChoiceDecision(client, _request(count), trace=trace)
    with pytest.raises(ChoiceContextLimitError) as caught:
        asyncio.run(decision.run())
    assert type(caught.value) is ChoiceContextLimitError
    assert str(caught.value) == NO_PROGRESS
    assert client.sizes == sizes
    assert {call["round"] for call in decision.calls} == {0}
    assert [len(call["question"]["criteria"]) for call in decision.calls] == [1] * count
    assert decision.usage == {"input_tokens": 1000 * count, "output_tokens": count}
    outcomes = [
        payload["outcome"]
        for event, payload in trace.events
        if event == TraceEvent.MODEL_ATTEMPT_FINISHED
    ]
    assert outcomes == ["context_limit" if size > 1 else "ok" for size in sizes]
