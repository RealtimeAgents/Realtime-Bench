"""A point of a pointer action lies on the screenshot the model was sent, or the action is refused
(user, 10-28): the conversion to desktop pixels clamps a point past the screenshot onto the screen's
edge (FrameGeometry.image_to_screen_point), where it used to run. The refusal names the point and
the screenshot's size, and the run stays valid. A point on the screenshot's edge runs, as does an
action that leaves its point out for the current cursor, and a P/R mark, which the mark map keeps
on the screenshot.

GuestEnvironment (sandbox/actions.py) sends its actions through the real GuestClient to FakeGuest
(tests/sandbox/fakes.py), which records the desktop pixels it is sent.
"""

from __future__ import annotations

import time

import pytest

from realtime_framework.entity.actions import Action, ActionType, FrameGeometry, bind_action
from realtime_framework.sandbox.actions import GuestEnvironment
from realtime_framework.sandbox.guest import GuestClient
from realtime_framework.sandbox.observation.capture import prepare_for_model
from realtime_framework.sandbox.observation.som import ToolRuntime, resolve_action_references

from ..replay import frames as replay_frames
from .fakes import FakeGuest

# The 1920x1080 desktop as the model sees it, at 768x432: scale 2.5.
GEOMETRY = FrameGeometry((768, 432), (1920, 1080))


class _WallTask:
    def clock_snapshot(self):
        return {"fps": 1.0}


@pytest.fixture
def guest():
    with FakeGuest() as fake:
        yield fake


def _environment(guest: FakeGuest) -> GuestEnvironment:
    return GuestEnvironment(
        GuestClient(host="127.0.0.1", port=guest.port),
        _WallTask(),
        clock_mode="wall",
        wall_start_time=time.time(),
    )


def _sent(name: str, **parameters) -> Action:
    """An action as the harness hands it to the environment, in the screenshot's pixels
    (harnesses/default/harness.py InteractionProfile.bind_action)."""
    return bind_action(Action(ActionType(name), parameters), GEOMETRY)


def _outside(name: str, x, y) -> str:
    return f"{name} point ({x}, {y}) lands outside the screenshot, which is 768x432 pixels"


@pytest.mark.parametrize(
    ("action", "refusal"),
    [
        pytest.param(_sent("click", x=405, y=732), _outside("click", 405, 732), id="click-below"),
        pytest.param(
            _sent("right_click", x=768, y=100), _outside("right_click", 768, 100), id="right_click"
        ),
        pytest.param(
            _sent("double_click", x=-1, y=100), _outside("double_click", -1, 100), id="double_click"
        ),
        pytest.param(
            _sent("triple_click", x=100, y=-0.5),
            _outside("triple_click", 100, -0.5),
            id="triple_click",
        ),
        pytest.param(
            _sent("mouse_down", x=100, y=432), _outside("mouse_down", 100, 432), id="mouse_down"
        ),
        pytest.param(_sent("mouse_up", x=5000, y=5), _outside("mouse_up", 5000, 5), id="mouse_up"),
        pytest.param(
            _sent("move", x=768.0, y=0, duration=0.2), _outside("move", 768.0, 0), id="move"
        ),
        pytest.param(
            _sent("scroll", x=800, y=200, dx=0, dy=-3), _outside("scroll", 800, 200), id="scroll"
        ),
        pytest.param(
            _sent("drag", x1=-3, y1=10, x2=100, y2=100), _outside("drag", -3, 10), id="drag-start"
        ),
        pytest.param(
            _sent("drag", x1=10, y1=10, x2=300, y2=432), _outside("drag", 300, 432), id="drag-end"
        ),
    ],
)
def test_a_point_outside_the_screenshot_is_refused_and_sends_nothing(guest, action, refusal):
    env = _environment(guest)
    assert env.execute(action) == "failed"
    assert env._last_rejected_by_guest is True
    assert env._last_exec_error == refusal
    assert getattr(env, "_invalid_run", None) is None
    assert guest.actions == []


@pytest.mark.parametrize(
    ("action", "desktop"),
    [
        pytest.param(_sent("click", x=0, y=0), {"x": 0, "y": 0}, id="the-first-pixel"),
        pytest.param(_sent("click", x=767, y=431), {"x": 1918, "y": 1078}, id="the-last-pixel"),
        # 1918.75 and 1078.75 round past the screen and are kept on its last pixel.
        pytest.param(
            _sent("click", x=767.5, y=431.5), {"x": 1919, "y": 1079}, id="inside-the-last-pixel"
        ),
        pytest.param(
            _sent("drag", x1=0, y1=0, x2=767, y2=431),
            {"x1": 0, "y1": 0, "x2": 1918, "y2": 1078},
            id="a-drag-corner-to-corner",
        ),
        pytest.param(_sent("scroll", dx=0, dy=-3), {"dx": 0, "dy": -3}, id="at-the-cursor"),
        pytest.param(_sent("mouse_up"), {}, id="a-release-at-the-cursor"),
    ],
)
def test_a_point_on_the_screenshot_runs(guest, action, desktop):
    env = _environment(guest)
    assert env.execute(action) == "applied"
    assert guest.actions == [{"action_type": action.action_type.value, "parameters": desktop}]


def _edge_probe(frame: int, *, text_observation: bool = False) -> dict:
    """The replay form with Submit (mark P3) in the screen's bottom-right corner, its click point
    on the last pixel, and the route map (R1) reaching the right and bottom edges. A capture clips
    each box to the viewport (sandbox/browser/cdp_interactables.py visibleRect)."""
    value = replay_frames.probe(frame, text_observation=text_observation)
    (submit,) = [item for item in value["actionable_now"] if item["id"] == "dom-0003"]
    submit["bbox"] = [1800, 1020, 1920, 1080]
    submit["click_point"] = [1919, 1079]
    (route_map,) = value["canvas_surfaces"]
    route_map["bbox"] = [1420, 580, 1920, 1080]
    return value


@pytest.mark.parametrize(
    ("name", "parameters", "desktop"),
    [
        pytest.param("click", {"target": "P3"}, {"x": 1918, "y": 1078}, id="a-point-mark"),
        pytest.param(
            "click", {"target": "R1@1000,1000"}, {"x": 1918, "y": 1078}, id="a-region-s-corner"
        ),
        pytest.param("move", {"target": "R1@0,0"}, {"x": 1420, "y": 580}, id="a-region-s-origin"),
        pytest.param(
            "scroll",
            {"target": "R1", "dx": 0, "dy": -3},
            {"x": 1670, "y": 830, "dx": 0, "dy": -3},
            id="a-region-s-center",
        ),
        pytest.param(
            "drag",
            {"start": "R1@0,0", "end": "R1@1000,1000"},
            {"x1": 1420, "y1": 580, "x2": 1918, "y2": 1078},
            id="a-drag-across-a-region",
        ),
    ],
)
def test_a_mark_at_the_screen_s_edge_resolves_on_the_screenshot_and_runs(
    guest, name, parameters, desktop
):
    """The marks are those the frame's inspection makes (harnesses/default/harness.py
    bind_observation), resolved and bound as the harness does (parse_grounded_action)."""
    source = replay_frames.FrameSource(
        replay_frames.ReplayTask(), text_observation=False, fps=1.0, probe=_edge_probe
    )
    observation = source.capture(0)
    geometry = prepare_for_model(observation.screenshot).geometry
    assert geometry.image_size == (768, 432)
    marks = ToolRuntime().inspect(
        observation,
        requested_observation_id=observation.metadata["observation_id"],
        tick=0,
        model_width=768,
        model_height=432,
    )["marks"]
    action, resolution = resolve_action_references(Action(ActionType(name), parameters), marks)
    assert resolution["valid"] is True
    env = _environment(guest)
    assert env.execute(bind_action(action, geometry)) == "applied"
    assert guest.actions == [{"action_type": name, "parameters": desktop}]
