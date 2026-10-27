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
from realtime_framework.harnesses.default.harness import INTERACTION_PROFILES, interaction_profile
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
    (lambda o: True, "Use current P/R marks", "decision 11: so is P6"),
    # A frame's text and legend say only what holds for their own screenshot (user, 10-27).
    (lambda o: True, "Screen observed for this decision", "it names the newest frame's decision"),
    (lambda o: True, "current-only", "a legend's marks are its own screenshot's"),
    (lambda o: True, "Use only these marks", "an older frame's are not the ones to use"),
    # Decision 6: a streaming action's marks and coordinates are read from the frame it names,
    # however old; a streaming frame is no decision's; no WAIT runs for what is not executed.
    *(
        (lambda o: o.interaction_mode == "streaming", sentence, "decision 6")
        for sentence in (
            "reuse a mark from an earlier screenshot",
            "Never reuse an older screenshot's mark",
            "newest screenshot's P/R marks",
            "Current-screenshot",
            "current screenshot",
            "current P/R marks",
            "this decision",
            "per decision",
            "the harness executes WAIT",
        )
    ),
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
    profile = interaction_profile(name, agent.prompt_options)
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


# Words that hold only for the newest frame. A frame's text and SoM legend stay in the context once
# newer frames follow, so they say none of them (user, 10-27); JEV's digest keeps no older
# observation (default/request.py _digest_items), and its frame text is left as it is (decision 7).
NEWEST_ONLY = ("current", "newest", "latest", "this decision", "only these", "Use only")
# Which screenshot's marks an action uses, and the coordinates without a suitable one, are said
# once, in the system instruction of every profile whose model writes marks.
MARK_RULES = {
    "turn_based": (
        "P/R marks belong only to the screenshot that supplies them.",
        "Use the newest screenshot's P/R marks when\n  supplied;",
        "If no suitable mark exists, replace target with x/y",
    ),
    "streaming": (
        "P/R marks belong only to the screenshot that supplies them.",
        "An action's marks are read from the screenshot its frame names.",
        "If no suitable mark exists, replace target with x/y",
    ),
}


@pytest.mark.parametrize(("profile", "mode", "clock"), list(_runs()))
def test_frames_say_only_what_holds_for_their_screenshot(profile, mode, clock):
    _, texts = _compose(profile, mode, clock)
    places = ("SoM legend",) if profile == "jev_decisions" else ("frame", "SoM legend")
    found = [
        (word, where)
        for where in places
        for word in NEWEST_ONLY
        if word.lower() in texts[where].lower()
    ]
    assert found == []
    if profile in LIVE_PROFILES:
        system = texts["system instruction"]
        assert [rule for rule in MARK_RULES[mode] if system.count(rule) != 1] == []


def test_every_profile_has_a_run():
    assert {profile for profile, _, _ in _runs()} == set(INTERACTION_PROFILES)
    assert {(mode, clock) for _, mode, clock in _runs()} == {
        ("turn_based", "wall"),
        ("turn_based", "finite_b"),
        ("turn_based", "pause"),
        ("streaming", "wall"),
    }
