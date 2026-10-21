"""The execution feedback the model is told of each streaming outcome
(harnesses/streaming/feedback.py), on the outcomes the streaming policy commits in its
discrete-event simulation (tests/kernel/streaming_simulation.py)."""

from __future__ import annotations

import pytest

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.evaluation.kernel.streaming import StreamingConfig
from realtime_framework.harnesses.default.frames import (
    FRAME_NOT_IN_TABLE,
    INVALID_FRAME_SYNTAX,
    MISSING_FRAME,
)
from realtime_framework.harnesses.streaming.feedback import action_text, execution_feedback
from tests.kernel.streaming_simulation import Call, Reply, Simulation, World

CLICK = Action.click(10, 20)
TYPE = Action(ActionType.TYPE, {"text": "hi"})
KEY = Action(ActionType.KEY, {"key": "enter"})


def _outcomes(world: World, duration_s: float = 5.0, **fields) -> dict:
    """The outcomes committed, by call id."""
    sim = Simulation(StreamingConfig(duration_s, **fields), world).run()
    return {outcome.call_id: outcome for outcome in sim.outcomes}


def test_an_applied_action_ran_what_was_asked():
    outcome = _outcomes(World(replies=[Reply(1.0, calls=(Call(0.25, CLICK, "a"),))]))["a"]
    assert execution_feedback(outcome, action_text(CLICK)) == (
        "EXECUTION RESULT: actual=click {'x': 10, 'y': 20, 'button': 'left'}; "
        "requested=click {'x': 10, 'y': 20, 'button': 'left'}; status=applied."
    )


def test_a_superseded_action_ran_nothing_and_says_a_newer_one_took_its_place():
    second = Reply(3.0, calls=(Call(0.5, TYPE, "b"), Call(1.0, KEY, "c")))
    world = World(replies=[Reply(1.0, calls=(Call(0.25, CLICK, "a", execute_s=4.0),)), second])
    outcome = _outcomes(world, 6.0)["b"]
    assert outcome.status == "superseded"
    assert execution_feedback(outcome, action_text(TYPE)) == (
        "EXECUTION RESULT: actual=none; requested=type {'text': 'hi'}; status=superseded. "
        "NOTE: this action was NOT executed: a newer action arrived while another one was "
        "executing and took its place."
    )


def test_an_action_of_a_cancelled_reply_ran_nothing_and_says_so():
    first = Reply(4.0, calls=(Call(1.0, CLICK, "a"), Call(3.0, TYPE, "b")), interruptible=False)
    world = World(replies=[first, Reply(1.0)], execute_s=1.5)
    outcome = _outcomes(world, 4.5, reply_watchdog_s=2.0)["b"]
    assert outcome.status == "cancelled_generation"
    assert execution_feedback(outcome, action_text(TYPE)) == (
        "EXECUTION RESULT: actual=none; requested=type {'text': 'hi'}; "
        "status=cancelled_generation. "
        "NOTE: this action was NOT executed because the reply it came in was cancelled."
    )


@pytest.mark.parametrize(
    ("grounding", "why"),
    [
        (MISSING_FRAME, "is missing"),
        (INVALID_FRAME_SYNTAX, "is not an integer"),
        (FRAME_NOT_IN_TABLE, "names no frame that was sent"),
    ],
)
def test_an_action_whose_frame_cannot_be_found_keeps_what_was_asked_and_says_why(grounding, why):
    world = World(replies=[Reply(1.0, calls=(Call(0.25, CLICK, "a", grounding=grounding),))])
    outcome = _outcomes(world)["a"]
    assert (outcome.status, outcome.detail, outcome.action) == (
        "invalid_grounding_frame",
        grounding,
        None,
    )
    assert execution_feedback(outcome, action_text(CLICK)) == (
        "EXECUTION RESULT: actual=none; requested=click {'x': 10, 'y': 20, 'button': 'left'}; "
        f"status=invalid_grounding_frame. NOTE: this action was NOT executed because its frame "
        f"field {why}. Set frame to the FRAME number drawn on the screenshot the action is based "
        "on."
    )


def test_a_reply_with_no_action_ran_nothing():
    world = World(replies=[Reply(1.0, protocol_failure="invalid_action_output"), Reply(1.0)])
    outcome = _outcomes(world, 1.5)[None]
    assert outcome.status == "none"
    text = execution_feedback(outcome, "none")
    assert text == (
        "EXECUTION RESULT: actual=none; requested=none; status=none. "
        "NOTE: your reply had no valid action, so nothing was executed."
    )
    assert "WAIT was executed" not in text


def test_a_done_ran_as_the_wait_it_became():
    world = World(replies=[Reply(1.0, calls=(Call(0.25, Action(ActionType.DONE, {}), "a"),))])
    outcome = _outcomes(world)["a"]
    assert (outcome.status, outcome.protocol_failure) == ("wait", "unsupported_terminal_action")
    assert execution_feedback(outcome, "done").startswith(
        "EXECUTION RESULT: actual=wait; requested=done; status=wait."
    )


def test_an_action_the_deadline_kept_from_running_says_the_time_limit_was_reached():
    calls = (Call(1.0, CLICK, "a", execute_s=3.0), Call(1.5, TYPE, "b"))
    outcome = _outcomes(World(replies=[Reply(3.0, calls=calls)]), 2.0)["b"]
    assert outcome.status == "not_applied_time_limit"
    assert execution_feedback(outcome, action_text(TYPE)).endswith(
        "status=not_applied_time_limit. NOTE: your previous action was NOT applied because the "
        "time limit was reached; the screen is unchanged."
    )


def test_an_action_the_environment_refused_ran_nothing_and_gives_the_refusal():
    world = World(replies=[Reply(1.0, calls=(Call(0.25, CLICK, "a", rejection="outside"),))])
    outcome = _outcomes(world)["a"]
    assert execution_feedback(outcome, action_text(CLICK)) == (
        "EXECUTION RESULT: actual=none; requested=click {'x': 10, 'y': 20, 'button': 'left'}; "
        "status=rejected. Detail: outside"
    )
