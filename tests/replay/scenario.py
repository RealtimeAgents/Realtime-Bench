"""The long replay scenario: 60 decisions on the synthetic form in frames.py.

Decision k sees frame k + 1 (frame 0 is the runner's initial capture, discarded after initialize).
Marks per frame: P1 Name, P2 the checkbox, P3 Submit, R1 the route map; Next is U1 up to frame 30
and P4 from frame 31 on (frames.next_enabled). With the default policies the first screenshot
fold happens at decision 20 and the first archive at decision 50 (see test_chat_completions.py).

Replies are text actions for the text_json action channel. Statuses are the ones run_stream
reports for them under the wall clock: "applied" for executed GUI actions and "wait" for WAIT
(sandbox/actions.py:830-936), "rejected" with the environment's message when it refuses an
action (runner.py:533-537, 674), and "not_applied_time_limit" past the deadline (runner.py:526).
A reply that yields no valid action becomes WAIT (harnesses/default/parsing.py:640-744) and is
reported as "wait".
"""

from __future__ import annotations

from .steps import Step
from .stub import Reply

TASK_INSTRUCTION = (
    "Enter the name Ada Lovelace, accept the terms, trace the route on the map, "
    "then press Submit and continue with Next."
)

# The guest's HTTP 400 for a key it cannot press (sandbox/guest.py:72-74 wrapping
# sandbox/guest_server/server.py:63).
UNSUPPORTED_KEY = (
    "400 guest rejected the action: unsupported key for pyautogui: 'hyper' (normalized: 'hyper')"
)
# The environment's refusal of pointer input on the browser toolbar (sandbox/actions.py:485-488).
OUTSIDE_PAGE = (
    "click lands outside the page: only the page content can be used, not the browser's "
    "address bar, tabs or the desktop around it"
)

CLICK_NAME = '{"action": "click", "target": "P1"}'
CLICK_AGREE = '{"action": "click", "target": "P2"}'
CLICK_SUBMIT = '{"action": "click", "target": "P3"}'
CLICK_NEXT = '{"action": "click", "target": "P4"}'

LONG_SCENARIO = [
    # 0-9
    Step(CLICK_NAME, reasoning="The name field comes first."),
    Step('{"action": "type", "text": "Ada Lovelace"}'),
    Step('{"action": "key", "key": "tab"}'),
    Step(CLICK_AGREE, reasoning="Accept the terms before submitting."),
    Step('```json\n{"action": "click", "target": "R1@250,750"}\n```'),
    Step('{"action": "wait", "duration": 1}', status="wait"),
    Step("Let me look at the form again.", status="wait"),
    Step('{"action": "key", "key": "hyper"}', status="rejected", detail=UNSUPPORTED_KEY),
    Step('{"action": "key", "key": "ctrl+a"}'),
    Step('{"action": "drag", "start": "R1@100,100", "end": "R1@900,900", "duration": 0.5}'),
    # 10-19
    Step('{"action": "scroll", "target": "R1", "dx": 0, "dy": -3}', reasoning="Pan the map."),
    Step('click(target="P3")'),
    Step('{"action": "click", "x": 384, "y": 216}'),
    Step('{"action": "hold_key", "key": "shift", "duration": 0.5}'),
    Step("", status="wait"),
    Step('{"action": "move", "target": "P3", "duration": 0.2}'),
    Step('{"action": "double_click", "target": "P1"}', reasoning="Select the typed name."),
    Step(f"{CLICK_NAME} {CLICK_AGREE}", status="wait"),
    Step(CLICK_NEXT, status="wait"),  # Next is still disabled: frame 19 has no P4
    Step('{"action": "wait"}', status="wait"),
    # 20-29: decision 20 is the first request with folded screenshots
    Step(CLICK_SUBMIT, reasoning="Submit the form."),
    Step('{"action": "click", "x": 100, "y": 20}', status="rejected", detail=OUTSIDE_PAGE),
    Step('{"action": "type", "text": "Ada"}'),
    Step('{"action": "click", "target": "R1@500,500"}'),
    Step('{"action": "key", "key": "enter"}'),
    Step('{"action": "right_click", "target": "R1@600,400"}'),
    Step('{"action": "triple_click", "target": "P1"}'),
    Step('{"action": "key", "key": "backspace", "repeat": 3}'),
    Step('{"action": "mouse_down", "target": "R1@200,200"}'),
    Step('{"action": "move", "target": "R1@800,600", "duration": 0.3}'),
    # 30-39: frame 31 on, Next is P4
    Step('{"action": "mouse_up", "target": "R1@800,600"}'),
    Step(CLICK_NEXT, reasoning="Next is enabled now."),
    Step('{"action": "scroll", "target": "R1", "dx": 2, "dy": 0}'),
    Step('{"action": "wait", "duration": 0.5}', status="wait"),
    Step('```\n{"action": "key", "key": "escape"}\n```'),
    Step('press_key("tab")'),
    Step('{"action": "type", "text": "Lovelace"}'),
    Step(CLICK_AGREE),
    Step('{"action": "click", "target": "P9"}', status="wait"),  # no such mark
    Step('{"action": "hold_key", "key": "shift+tab", "duration": 0.2}'),
    # 40-49
    Step(CLICK_SUBMIT, reasoning="Submit again after the edits."),
    Step('{"action": "drag", "start": "P3", "end": "R1@500,500"}'),
    Step('{"action": "click", "target": "R1@50,950", "button": "right"}'),
    Step("Thinking about which control to use next.", status="wait"),
    Step('{"action": "key", "key": "ctrl+z"}'),
    Step(CLICK_NEXT),
    Step('{"action": "scroll", "target": "P1", "dx": 0, "dy": 2}'),
    Step('{"action": "wait"}', status="wait"),
    Step('{"action": "click", "target": "P3", "modifier": "shift"}'),
    Step(CLICK_NAME, reasoning="Back to the name field."),
    # 50-59: decision 50 is the first request with an archived-action ledger
    Step('{"action": "type", "text": "Ada Lovelace"}'),
    Step(CLICK_AGREE),
    Step(CLICK_SUBMIT, reasoning="Final submit."),
    Step('{"action": "key", "key": "hyper"}', status="rejected", detail=UNSUPPORTED_KEY),
    Step(CLICK_NEXT),
    Step('{"action": "click", "target": "R1@750,250"}'),
    Step('{"action": "wait", "duration": 2}', status="wait"),
    Step("", status="wait"),
    Step(CLICK_NEXT),
    Step(CLICK_SUBMIT, status="not_applied_time_limit"),
]


def replies(steps: list[Step]) -> list[Reply]:
    """The stub's reply for each step, with fixed OpenAI-shaped usage."""
    out = []
    for k, step in enumerate(steps):
        reasoning_tokens = len(step.reasoning) // 4
        completion_tokens = 4 + len(step.reply) // 4 + reasoning_tokens
        prompt_tokens = 1500 + 260 * k
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0 if k == 0 else 1280},
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        }
        out.append(Reply(step.reply, usage, step.reasoning))
    return out
