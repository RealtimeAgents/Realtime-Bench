"""The selection flow of one Choice decision (providers/decisions/selection.py ChoiceDecision),
with a fake client in place of the HTTP ones: which questions it asks, in which order, and when it
gives up. What a client sends and how it reads the reply is tested in test_client.py.
"""

from __future__ import annotations

import asyncio

import pytest

from realtime_framework.entity.actions import Observation
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

from ..context import api_policy

MODEL = "openai/gpt-6-luna-decisions"
NO_PROGRESS = "JEV cannot compare even two candidates within the context limit"
# Far more questions than any decision below asks; past it the client fails the question.
QUESTION_CAP = 100


class _Trace:
    attempt_id = "selection"

    def __init__(self):
        self.events = []

    def capture(self, event, payload, body=None):
        self.events.append((event, payload))


class _OneOptionClient:
    """A client whose context limit leaves room for one option: a question offering two or more is
    rejected with ChoiceContextLimitError, as the HTTP clients reject one on a 413. A question
    offering one is refused, as OpenAI's upstream refuses it with HTTP 400."""

    capabilities = OpenAIDecisionsClient.capabilities

    def __init__(self):
        self.sizes = []

    async def decide(self, state, question):
        criteria = question["criteria"]
        self.sizes.append(len(criteria))
        if len(self.sizes) > QUESTION_CAP:
            raise RuntimeError(f"more than {QUESTION_CAP} questions")
        if len(criteria) < 2:
            raise RuntimeError("a question needs at least two options")
        raise ChoiceContextLimitError("Decisions request exceeds its context limit: Too large")

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
        api_policy("openai_decisions"),
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
        (2, [2]),
        (3, [3, 2]),
        (5, [5, 2, 3, 2]),
    ],
)
def test_a_round_without_progress_ends_the_decision(count, sizes):
    """When even two options exceed the context limit, every group is bisected down to single
    options, each of which wins without a question, so the round ends with as many winners as it
    began with candidates (ChoiceDecision._choose). The next round would ask the same questions
    again, forever; the decision ends instead with a context-limit error, after the one round."""
    client, trace = _OneOptionClient(), _Trace()
    decision = ChoiceDecision(client, _request(count), trace=trace)
    with pytest.raises(ChoiceContextLimitError) as caught:
        asyncio.run(decision.run())
    assert type(caught.value) is ChoiceContextLimitError
    assert str(caught.value) == NO_PROGRESS
    assert client.sizes == sizes
    assert decision.calls == []
    assert decision.usage == {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
    outcomes = [
        payload["outcome"]
        for event, payload in trace.events
        if event == TraceEvent.MODEL_ATTEMPT_FINISHED
    ]
    assert outcomes == ["context_limit"] * len(sizes)


class _AnsweringClient(_OneOptionClient):
    """A client that answers every question it is asked with its first option."""

    async def decide(self, state, question):
        criteria = question["criteria"]
        self.sizes.append(len(criteria))
        if len(criteria) < 2:
            raise RuntimeError("a question needs at least two options")
        choice = next(iter(criteria))
        answer = {
            "type": "choice",
            "choice": choice,
            "probabilities": {key: float(key == choice) for key in criteria},
            "confidence": 1.0,
        }
        return ChoiceReply(answer, {"input_tokens": 1000, "output_tokens": 1}, {})


def test_a_lone_candidate_is_chosen_without_a_question():
    """A frame offering one candidate has nothing to choose from: it is the action, and the
    decision asks no question and reports no call, and no usage and no cost: a known 0
    (ChoiceDecision.run)."""
    client = _AnsweringClient()
    result = asyncio.run(ChoiceDecision(client, _request(1), trace=_Trace()).run())
    assert client.sizes == []
    assert result["action"] == {"action": "click", "target": "P0"}
    assert (result["selected_candidate"], result["candidate_count"]) == ("c0", 1)
    assert result["jev_calls"] == [] and result["retries"] == []
    assert result["jev_response"] is None and result["jev_question"] is None
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}


class _CostingClient(_AnsweringClient):
    """An _AnsweringClient whose replies report the next of ``costs``, in USD, with their usage,
    as OpenRouter's do, and no cost once they run out."""

    def __init__(self, costs):
        super().__init__()
        self.costs = list(costs)

    async def decide(self, state, question):
        reply = await super().decide(state, question)
        if self.costs:
            reply.usage["cost"] = self.costs.pop(0)
        return reply


# Three replies' costs in a 10-27 JEV run; summed as floats they make 0.0006368039999999999.
COSTS = (0.000182868, 0.000229824, 0.000224112)


def test_a_decision_costs_what_its_questions_cost():
    """A decision over 510 candidates asks three questions, one per group of 255 and one between
    the two winners, and costs what the three replies report. One whose last reply reports no
    cost has an unknown cost, not that of the others."""
    client = _CostingClient(COSTS)
    result = asyncio.run(ChoiceDecision(client, _request(510), trace=_Trace()).run())
    assert client.sizes == [255, 255, 2]
    assert result["usage"] == {"input_tokens": 3000, "output_tokens": 3, "cost": 0.000636804}
    client = _CostingClient(COSTS[:2])
    result = asyncio.run(ChoiceDecision(client, _request(510), trace=_Trace()).run())
    assert result["usage"] == {"input_tokens": 3000, "output_tokens": 3}


def test_a_lone_group_wins_without_a_question():
    """A group of one, here the last of 256 candidates past the 255-option boundary, wins its
    group without a question and meets the other group's winner (ChoiceDecision._choose)."""
    client = _AnsweringClient()
    result = asyncio.run(ChoiceDecision(client, _request(256), trace=_Trace()).run())
    assert client.sizes == [255, 2]
    assert [(call["round"], call["group"]) for call in result["jev_calls"]] == [(0, 0), (1, 0)]
    assert result["selected_candidate"] == "c0"
