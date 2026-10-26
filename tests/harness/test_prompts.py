"""The prompts of every interaction profile, under each interaction mode and clock, composed for
the run's prompt options alone (harnesses/registry.py configure_agent): the system instruction,
the frame text, the SoM legend and, on the function channel, the GUI function declarations. None
says what is false for its run.

Of the profiles, interaction modes and clocks, the runs the pre-run checks allow are composed
(providers/preflight.py): a streaming run is on the wall clock (check_interaction_mode), on a
live driver, whose profiles are standard_vlm (OpenAI Realtime) and function_call (Gemini Live);
the choice models' runs are turn-based, on the wall clock (check_wall_clock)."""

from __future__ import annotations

import json
import types

import pytest

from realtime_framework.entity.actions import gui_function_declarations
from realtime_framework.harnesses.default.harness import INTERACTION_PROFILES
from realtime_framework.harnesses.registry import configure_agent
from realtime_framework.sandbox.observation.som import build_hybrid_som

LIVE_PROFILES = ("standard_vlm", "function_call")
CLOCKS = {"wall": ("wall", None), "finite_b": ("token", 256), "pause": ("token", None)}

# (whether it applies to a run's prompt options, a sentence no prompt of such a run holds, why)
FORBIDDEN = [
    (lambda o: not o.finite_budget, "All generated output tokens count.", "only B charges tokens"),
    (
        lambda o: not o.finite_budget,
        "thought token usage under the token clock, then advances simulated time",
        "decision 8: only B charges tokens, in the function-call contract too",
    ),
    (lambda o: o.interaction_mode == "turn_based", "FRAME REFERENCE", "a turn names no frame"),
    (lambda o: True, "Latest screen", "decision 42: a frame stays in the context once older"),
    (lambda o: True, "CURRENT-SCREEN", "decision 42"),
]


def _runs():
    for profile in sorted(INTERACTION_PROFILES):
        for mode in ("turn_based", "streaming"):
            for clock in CLOCKS:
                live = profile in LIVE_PROFILES
                if mode == "streaming" and not live:
                    continue
                if clock != "wall" and (mode == "streaming" or not live):
                    continue
                yield profile, mode, clock


def _compose(name: str, mode: str, clock: str):
    profile = INTERACTION_PROFILES[name]
    clock_mode, tokens_per_frame = CLOCKS[clock]
    agent = types.SimpleNamespace(task_instruction="Do the task.")
    configure_agent(
        agent,
        fps=1.0,
        tokens_per_frame=tokens_per_frame,
        clock_mode=clock_mode,
        category="browser_game" if name == "multimodal_game_decisions" else "realtime",
        interaction_mode=mode,
    )
    streaming = mode == "streaming"
    legend = build_hybrid_som(
        {"actionable_now": []},
        None,
        action_width=768,
        action_height=432,
        frame_number=1 if streaming else None,
    )
    texts = {
        "system instruction": profile.system_instruction(agent),
        "frame": profile.frame_prompt,
        "SoM legend": legend["text"],
    }
    if name == "function_call":
        texts["functions"] = json.dumps(gui_function_declarations(frame_reference=streaming))
    return agent.prompt_options, texts


@pytest.mark.parametrize(("profile", "mode", "clock"), list(_runs()))
def test_no_prompt_says_what_is_false_for_its_run(profile, mode, clock):
    options, texts = _compose(profile, mode, clock)
    found = [
        (sentence, where, why)
        for applies, sentence, why in FORBIDDEN
        if applies(options)
        for where, text in texts.items()
        if sentence in text
    ]
    assert found == []


def test_every_profile_has_a_run():
    assert {profile for profile, _, _ in _runs()} == set(INTERACTION_PROFILES)
    assert {(mode, clock) for _, mode, clock in _runs()} == {
        ("turn_based", "wall"),
        ("turn_based", "finite_b"),
        ("turn_based", "pause"),
        ("streaming", "wall"),
    }
