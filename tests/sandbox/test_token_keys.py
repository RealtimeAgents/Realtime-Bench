"""Token-clock key input: what GuestEnvironment sends a game for each key action, and how the
game's CDP calls carry it to the page.

The environment refuses the same shortcuts as on the wall clock; a held key auto-repeats until
another non-modifier key is pressed, and keeps its lease when it is held again; every delay, lease
and offset is in simulated seconds, whatever the frame rate. A task clock without a browser
(tests/replay/clock_task.py) records what the environment pushes into it. The real cdp_key_event
(sandbox/browser/clock.py) sends its key events to FakeCDP (tests/sandbox/fakes.py), and so does a
real catalog game (realtime_tasks/game/runtime.py), whose frame advances are recorded where it
asks CDP for them, as FakeCDP has no page clock to run them.
"""

from __future__ import annotations

import pytest

import realtime_tasks.registry as tasks
from realtime_framework.entity.actions import Action, ActionType
from realtime_framework.sandbox.actions import TAP_HOLD_S, GuestEnvironment
from realtime_framework.sandbox.browser.cdp import close_port
from realtime_framework.sandbox.browser.clock import cdp_key_event
from realtime_tasks.game import runtime

from ..replay.clock_task import ClockTask
from .fakes import FakeCDP


def _token(fps=1.0):
    task = ClockTask(fps)
    return GuestEnvironment(None, task, clock_mode="token"), task


def _of(task, *kinds):
    return [event for event in task.events if event["event"] in kinds]


def _repeats(advance, fps):
    """(world second, key, held modifiers) of each auto-repeat press an advance carries."""
    return [
        (
            round(advance["from_s"] + event["at_frames"] / fps, 6),
            event["key"],
            tuple(event["held_modifiers"]),
        )
        for event in advance["key_events"]
    ]


def _trace(task):
    """Key events as (type, chord, held modifiers), advances as (ticks, their repeats)."""
    fps = task.clock_snapshot()["fps"]
    return [
        (round(event["frames"], 6), _repeats(event, fps))
        if event["event"] == "advance"
        else (event["event"], event["key"], tuple(event["held_modifiers"]))
        for event in task.events
    ]


def _every_50_ms(start, stop, held=()):
    """Left's auto-repeat presses from ``start`` to ``stop`` seconds, both included."""
    count = round((stop - start) / 0.05) + 1
    return [(round(start + 0.05 * k, 6), "left", held) for k in range(count)]


@pytest.fixture
def cdp():
    with FakeCDP() as fake:
        yield fake
        close_port(fake.port)


def _key_events(fake):
    """(type, key, modifiers, autoRepeat) of each Input.dispatchKeyEvent FakeCDP received."""
    return [
        (params["type"], params["key"], params["modifiers"], params.get("autoRepeat", False))
        for params in (
            command["params"]
            for command in fake.commands
            if command["method"] == "Input.dispatchKeyEvent"
        )
    ]


# ---- refusals -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "refusal"),
    [
        pytest.param(
            Action(ActionType.KEY, {"key": "ctrl+r"}),
            "page reload shortcut 'ctrl+r' is disabled",
            id="a-tap-of-ctrl+r",
        ),
        pytest.param(
            Action.hold_key("f5", 1.0), "page reload shortcut 'f5' is disabled", id="a-hold-of-f5"
        ),
        pytest.param(
            Action(ActionType.KEY, {"key": "ctrl+l"}),
            "browser shortcut 'ctrl+l' is disabled",
            id="a-tap-of-ctrl+l",
        ),
    ],
)
def test_a_token_game_refuses_a_reload_or_browser_shortcut_and_sends_no_key(action, refusal):
    env, task = _token()
    assert env.execute(action) == "failed"
    assert env._last_rejected_by_guest is True
    assert env._last_exec_error.startswith(refusal)
    assert task.events == []
    assert getattr(env, "_invalid_run", None) is None


# ---- CDP key events -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "sent"),
    [
        pytest.param(
            ("keyDown", "ctrl+shift+ArrowLeft", False, ()),
            [
                ("keyDown", "Control", 2, False),
                ("keyDown", "Shift", 10, False),
                ("keyDown", "ArrowLeft", 10, False),
            ],
            id="a-press-sets-each-modifier-as-it-goes-down",
        ),
        pytest.param(
            ("keyUp", "ctrl+shift+ArrowLeft", False, ()),
            [
                ("keyUp", "ArrowLeft", 10, False),
                ("keyUp", "Shift", 2, False),
                ("keyUp", "Control", 0, False),
            ],
            id="a-release-goes-up-in-reverse-clearing-each-modifier",
        ),
        pytest.param(
            ("keyDown", "a", False, ("ctrl",)),
            [("keyDown", "a", 2, False)],
            id="a-press-carries-the-held-modifiers",
        ),
        pytest.param(
            ("keyUp", "shift+ArrowLeft", False, ("ctrl",)),
            [("keyUp", "ArrowLeft", 10, False), ("keyUp", "Shift", 2, False)],
            id="a-release-keeps-the-held-modifiers",
        ),
        pytest.param(
            ("keyDown", "ArrowLeft", True, ("ctrl", "shift")),
            [("keyDown", "ArrowLeft", 10, True)],
            id="an-auto-repeat",
        ),
        pytest.param(
            ("keyUp", "ArrowLeft", True, ()),
            [("keyUp", "ArrowLeft", 0, False)],
            id="a-release-is-no-auto-repeat",
        ),
    ],
)
def test_each_key_of_a_chord_reaches_the_page_as_a_cdp_key_event_with_the_modifiers_down(
    cdp, call, sent
):
    """``call`` is (type, chord, auto_repeat, held modifiers)."""
    event_type, chord, auto_repeat, held = call
    cdp_key_event(cdp.port, event_type, chord, auto_repeat=auto_repeat, held_modifiers=held)
    assert _key_events(cdp) == sent


# ---- auto-repeat ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("actions", "trace"),
    [
        pytest.param(
            [Action(ActionType.KEY, {"key": "shift"})],
            [
                ("keyDown", "shift", ()),
                (round(TAP_HOLD_S, 6), _every_50_ms(1.0, 1.0)),
                ("keyUp", "shift", ()),
                (1.0, _every_50_ms(1.05, 2.0)),
            ],
            id="a-tapped-modifier",
        ),
        pytest.param(
            [Action.hold_key("ctrl", 10.0)],
            [("keyDown", "ctrl", ()), (1.0, _every_50_ms(1.0, 1.95, ("ctrl",)))],
            id="a-held-modifier",
        ),
        pytest.param(
            [Action.hold_key("ctrl", 10.0), Action(ActionType.KEY, {"key": "ctrl"})],
            [
                ("keyDown", "ctrl", ()),
                ("keyDown", "ctrl", ()),
                (round(TAP_HOLD_S, 6), _every_50_ms(1.0, 1.0, ("ctrl",))),
                ("keyUp", "ctrl", ()),
                ("keyDown", "ctrl", ()),
                (1.0, _every_50_ms(1.05, 2.0, ("ctrl",))),
            ],
            id="a-tapped-held-modifier-pressed-again",
        ),
        pytest.param(
            [Action.hold_key("a", 0.0)],
            [("keyDown", "a", ()), ("keyUp", "a", ()), (1.0, [])],
            id="a-0-s-hold-of-another-key-takes-the-repeat-over",
        ),
        pytest.param(
            [Action.hold_key("left", 0.5)],
            [(1.0, _every_50_ms(1.0, 1.95))],
            id="a-shorter-hold-of-the-same-key-keeps-its-lease",
        ),
    ],
)
def test_a_held_key_keeps_repeating_until_another_non_modifier_key_is_pressed(actions, trace):
    env, task = _token()
    env.execute(Action.hold_key("left", 10.0))
    env.step_clock(1.0)
    for action in actions:
        assert env.execute(action) == "applied"
    env.step_clock(1.0)
    assert _trace(task) == [("keyDown", "left", ()), (1.0, _every_50_ms(0.5, 0.95)), *trace]


@pytest.mark.parametrize(
    ("fps", "duration", "repeats"),
    [
        (2.0, 0.72, 5),
        (30.0, 0.72, 5),
        (60.0, 0.72, 5),
        pytest.param(30.0, 4.15, 73, id="30.0-4.15-none-due-as-the-key-goes-up"),
    ],
)
def test_a_held_key_repeats_every_50_ms_from_500_ms_until_it_goes_up_at_any_frame_rate(
    fps, duration, repeats
):
    """At 30 fps the 74th repeat of a 4.15 s hold is due 1.4e-14 ticks before the lease ends,
    where its advance ends too: it is not sent."""
    env, task = _token(fps)
    env.execute(Action.hold_key("left", duration))
    env.wait_until_tick(5 * fps)
    keys = [(event["event"], event["at_s"]) for event in _of(task, "keyDown", "keyUp")]
    assert keys == [("keyDown", 0.0), ("keyUp", duration)]
    seconds = [
        second for advance in _of(task, "advance") for second, _, _ in _repeats(advance, fps)
    ]
    assert seconds == [round(0.5 + 0.05 * k, 6) for k in range(repeats)]


@pytest.mark.parametrize("fps", [2.0, 30.0, 60.0])
def test_the_holds_a_token_execution_reports_count_down_in_simulated_seconds(fps):
    env, _ = _token(fps)
    env.execute(Action.hold_key("left", 2.0))
    env.step_clock(0.5 * fps)
    env.execute(Action.hold_key("shift", 1.0))
    assert env._last_execution_result["active_holds"] == [
        {"key": "left", "remaining_simulated_seconds": 1.5},
        {"key": "shift", "remaining_simulated_seconds": 1.0},
    ]


def test_a_game_at_2_fps_asks_cdp_for_simulated_seconds_and_key_offsets_in_seconds(
    cdp, monkeypatch
):
    """A 0.72 s hold ends at tick 1.44, which cuts the advance from tick 1; its repeats, due from
    0.5 s (tick 1) on, all fall in that advance."""
    advances = []

    def cdp_advance_rendered_frame(dbg_port, seconds, raf_hz=60.0, key_events=()):
        advances.append(
            (round(seconds, 9), [round(event["at_seconds"], 9) for event in key_events])
        )
        return {"compositor_committed": True}

    monkeypatch.setattr(runtime, "cdp_advance_rendered_frame", cdp_advance_rendered_frame)
    task = tasks.get("gh_javascript_pong")
    monkeypatch.setattr(task, "dbg_port", cdp.port, raising=False)
    monkeypatch.setattr(task, "_clock_config", None, raising=False)
    monkeypatch.setattr(task, "_last_clock_advance", None, raising=False)
    task.configure_clock("token", 2.0)
    task._clock_config["active"] = True
    env = GuestEnvironment(None, task, clock_mode="token")
    assert env.execute(Action.hold_key("left", 0.72)) == "applied"
    env.wait_until_tick(2)
    assert advances == [(0.5, []), (0.22, [0.0, 0.05, 0.1, 0.15, 0.2]), (0.28, [])]
    assert _key_events(cdp) == [
        ("keyDown", "ArrowLeft", 0, False),
        ("keyUp", "ArrowLeft", 0, False),
    ]
