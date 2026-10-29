"""The text action grammar through its resolvers (harnesses/default/parsing.py): each resolver reads
the bodies of a model's output itself and asks its caller only to ground the action each body reads
as. The groundings here stand in for the harness's (harness.py parse_grounded_action): one keeps
the action as written, one lacks some marks, one places a mark differently in each frame.

What is asserted is what the action records keep: the action, where it came from (its parse mode,
the text it was read from, the candidates it was chosen among), and the frame field each body
named. The forms tested are those recorded runs show, and the cases the grammar must decide:
ambiguity, a repeated field, JSON in the text to type, the frame field. The other forms it accepts
wait on the decision about its aliases (decisions-needed item 10)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.harnesses.default.parsing import (
    resolve_action_output,
    resolve_structured_output,
)

CLICK_P1 = Action(ActionType.CLICK, {"target": "P1"})
NO_ACTION = {
    "source": "invalid_action_fallback_wait",
    "valid_action": False,
    "action_text": None,
    "parse_mode": None,
}


@dataclass(frozen=True)
class Read:
    """What a grounding made of an action: the action, None where its frame lacks a mark it
    names, and the frame field it was handed."""

    action: Action | None
    reference: object = None


def as_written(asked, reference):
    return Read(asked, reference)


def lacking(*marks):
    """A frame without ``marks``: an action targeting one of them resolves to none."""

    def ground(asked, reference):
        return Read(None if asked.parameters.get("target") in marks else asked, reference)

    return ground


def per_frame(asked, reference):
    """P1 is at (10, 20) in frame 1 and at (30, 40) in frame 2."""
    x, y = {1: (10, 20), 2: (30, 40)}[reference]
    return Read(Action(asked.action_type, {"x": x, "y": y}), reference)


def test_canonical_json_is_the_models_action_as_written():
    text = '{"action": "click", "target": "P1"}'
    output = resolve_action_output(text, as_written)
    assert output.decision.metadata() == {
        "source": "model_action",
        "valid_action": True,
        "action_text": text,
        "protocol_failure": None,
        "parse_mode": "canonical_json",
        "candidate_count": 1,
    }
    assert output.decision.action == CLICK_P1
    assert output.chosen == Read(CLICK_P1)
    assert output.parses == (output.chosen,)


@pytest.mark.parametrize(
    ("text", "parse_mode", "action_text"),
    [
        (
            '```json\n{"action": "click", "target": "P1"}\n```',
            "structured_envelope",
            '{"action":"click","target":"P1"}',
        ),
        (
            'I will click the name field. {"action": "click", "target": "P1"}',
            "embedded_json",
            '{"action":"click","target":"P1"}',
        ),
        (
            "{'action': 'click', 'target': 'P1'}",
            "python_literal",
            '{"action":"click","target":"P1"}',
        ),
        # A whole-line shorthand is read from its JSON as json.dumps writes it, with spaces.
        ("click P1", "text_shorthand", '{"action": "click", "target": "P1"}'),
    ],
)
def test_a_recovered_action_records_the_form_and_the_text_it_was_read_from(
    text, parse_mode, action_text
):
    output = resolve_action_output(text, as_written)
    assert output.decision.metadata() == {
        "source": "recovered_model_action",
        "valid_action": True,
        "action_text": action_text,
        "protocol_failure": None,
        "parse_mode": parse_mode,
        "candidate_count": 1,
    }
    assert output.decision.action == CLICK_P1


def test_two_different_actions_are_ambiguous_and_wait_stands_in():
    output = resolve_action_output("click P1. click P2", as_written)
    assert output.decision.metadata() == {
        **NO_ACTION,
        "protocol_failure": "ambiguous_action_output",
        "candidate_count": 2,
    }
    assert output.decision.action == Action.wait()
    assert output.chosen is None
    assert output.parses == (Read(CLICK_P1), Read(Action(ActionType.CLICK, {"target": "P2"})))


def test_one_action_written_twice_is_one_action_read_from_its_first_text():
    """Equal once a click's default button is filled in; the first is the one chosen."""
    text = (
        '{"action": "click", "target": "P1"} {"action": "click", "target": "P1", "button": "left"}'
    )
    output = resolve_action_output(text, as_written)
    assert output.decision.metadata()["candidate_count"] == 1
    assert output.decision.action_text == '{"action":"click","target":"P1"}'
    assert output.decision.action == CLICK_P1
    assert output.parses == (
        Read(CLICK_P1),
        Read(Action(ActionType.CLICK, {"target": "P1", "button": "left"})),
    )


def test_an_action_beside_one_its_frame_cannot_ground_is_ambiguous():
    """A candidate whose mark its frame lacks is a rejected candidate; beside a grounded one the
    output is ambiguous. Its grounding stays among the parses, in order, for the harness to name
    the mark (parse_grounded_action)."""
    text = '{"action": "click", "target": "P9"} {"action": "click", "target": "P1"}'
    output = resolve_action_output(text, lacking("P9"))
    assert output.decision.metadata() == {
        **NO_ACTION,
        "protocol_failure": "ambiguous_action_output",
        "candidate_count": 1,
    }
    assert output.parses == (Read(None), Read(CLICK_P1))


def test_candidates_none_of_which_ground_are_no_action_not_ambiguous():
    text = '{"action": "click", "target": "P9"} {"action": "click", "target": "P8"}'
    output = resolve_action_output(text, lacking("P9", "P8"))
    assert output.decision.metadata() == {
        **NO_ACTION,
        "protocol_failure": "invalid_action_output",
        "candidate_count": 0,
    }
    assert output.parses == (Read(None), Read(None))


def test_an_envelope_of_several_actions_is_no_action_not_its_first():
    text = '{"actions": [{"action": "click", "target": "P1"}, {"action": "click", "target": "P2"}]}'
    output = resolve_action_output(text, as_written)
    assert output.decision.metadata() == {
        **NO_ACTION,
        "protocol_failure": "invalid_action_output",
        "candidate_count": 0,
    }
    assert output.parses == ()


@pytest.mark.parametrize(
    "text",
    [
        '{"action": "click", "target": "P1", "target": "P2"}',
        # Field names are compared as the grammar folds them.
        '{"action": "click", "Target": "P1", "target": "P2"}',
        "{'action': 'click', 'target': 'P1', 'target': 'P2'}",
        # Two fields naming the action, naming different actions.
        '{"action": "click", "name": "type"}',
    ],
)
def test_a_field_given_twice_in_text_is_no_action(text):
    output = resolve_action_output(text, as_written)
    assert output.decision.metadata() == {
        **NO_ACTION,
        "protocol_failure": "invalid_action_output",
        "candidate_count": 0,
    }
    assert output.decision.action == Action.wait()
    assert output.parses == ()


def test_json_in_the_text_to_type_is_typed_not_read_as_an_action():
    typed = '{"action": "click", "target": "P1"}'
    output = resolve_action_output(f"type {typed}", as_written)
    assert output.decision.action == Action(ActionType.TYPE, {"text": typed})
    assert output.decision.metadata() == {
        "source": "recovered_model_action",
        "valid_action": True,
        "action_text": '{"action": "type", "text": "{\\"action\\": \\"click\\", \\"target\\": \\"P1\\"}"}',
        "protocol_failure": None,
        "parse_mode": "text_shorthand",
        "candidate_count": 1,
    }

    text = '{"action": "type", "text": "{\\"action\\": \\"click\\"}"}'
    output = resolve_action_output(text, as_written)
    assert output.decision.action == Action(ActionType.TYPE, {"text": '{"action": "click"}'})
    assert (output.decision.parse_mode, output.decision.action_text) == ("canonical_json", text)


@pytest.mark.parametrize(
    ("frame", "reference"),
    [(', "frame": 2', 2), (', "frame": "2"', "2"), ("", None)],
    ids=["integer", "string", "missing"],
)
def test_a_streaming_action_hands_its_frame_field_to_grounding_as_written(frame, reference):
    """Unconverted: the harness refuses a frame field that is no integer, or none."""
    text = '{"action": "click", "target": "P1"' + frame + "}"
    output = resolve_action_output(text, as_written, framed=True)
    assert (output.decision.parse_mode, output.decision.action_text) == ("canonical_json", text)
    assert output.decision.action == CLICK_P1
    assert output.chosen == Read(CLICK_P1, reference)


def test_a_recovered_streaming_action_keeps_its_frame_field():
    text = 'I will click it. {"action": "click", "target": "P1", "frame": 1}'
    output = resolve_action_output(text, as_written, framed=True)
    assert output.decision.parse_mode == "embedded_json"
    assert output.decision.action_text == '{"action":"click","target":"P1","frame":1}'
    assert output.chosen == Read(CLICK_P1, 1)


def test_a_turn_based_action_names_no_frame():
    output = resolve_action_output('{"action": "click", "target": "P1", "frame": 2}', as_written)
    assert output.decision.metadata() == {
        **NO_ACTION,
        "protocol_failure": "invalid_action_output",
        "candidate_count": 0,
    }


def test_streaming_candidates_are_the_same_action_only_as_their_frames_ground_them():
    """One mark named against two frames is two actions; against one frame twice, one."""
    clicks = '{"action": "click", "target": "P1", "frame": %d}'
    output = resolve_action_output(f"{clicks % 1} {clicks % 2}", per_frame, framed=True)
    assert output.decision.metadata() == {
        **NO_ACTION,
        "protocol_failure": "ambiguous_action_output",
        "candidate_count": 2,
    }
    assert [read.reference for read in output.parses] == [1, 2]

    output = resolve_action_output(f"{clicks % 2} {clicks % 2}", per_frame, framed=True)
    assert output.decision.candidate_count == 1
    assert output.decision.action == Action(ActionType.CLICK, {"x": 30, "y": 40})
    assert output.chosen.reference == 2


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_output_is_no_action(text):
    output = resolve_action_output(text, as_written)
    assert output.decision.metadata() == {
        **NO_ACTION,
        "protocol_failure": "empty_action_output",
        "candidate_count": 0,
    }
    assert output.decision.action == Action.wait()


def test_a_structured_action_is_the_object_its_channel_gave():
    text = '{"action": "click", "target": "P1", "frame": 2}'
    output = resolve_structured_output(text, as_written)
    assert output.decision.metadata() == {
        "source": "model_action",
        "valid_action": True,
        "action_text": text,
        "protocol_failure": None,
        "parse_mode": "canonical_json",
        "candidate_count": 1,
    }
    assert output.chosen == Read(CLICK_P1, 2)
    assert output.parses == (output.chosen,)


@pytest.mark.parametrize(
    "text",
    [
        "click P1",
        '```json\n{"action": "click", "target": "P1"}\n```',
        'I will click. {"action": "click", "target": "P1"}',
    ],
)
def test_structured_output_is_not_recovered_from_text(text):
    output = resolve_structured_output(text, as_written)
    assert output.decision.metadata() == {
        **NO_ACTION,
        "protocol_failure": "invalid_action_output",
        "candidate_count": 0,
    }
    assert output.parses == ()
