"""Scripted Choice Decisions episodes: the 60-decision web scenario, a short game page, failures.

Decision k sees frame k + 1. A decision's script holds one entry per question attempt: the
choice key the model answers, or a Failure the endpoint returns. The criteria keys come from the
current frame's candidates (harnesses/jev/candidates.py:159-311, game_candidates.py:8-71,
providers/decisions/questions.py:57-113). Both APIs get the same candidates on the same frame,
so the web scripts serve both.

Statuses are scripted as in scenario.py: "applied" for executed GUI actions, "wait" for WAIT
(sandbox/actions.py:830-936), "rejected" with the environment's message (shell.py _execute,
kernel/turns.py _executed) and "not_applied_time_limit" past the deadline (kernel/turns.py
_skipped). Every web candidate is a page mark or a key the guest accepts
(sandbox/guest_server/server.py:22-63), so the web scenario has no rejection; the game page has
one, a composed screen pointer on the browser toolbar.
"""

from __future__ import annotations

import hashlib
import io
import json

from PIL import Image, ImageDraw

from . import frames
from .scenario import OUTSIDE_PAGE
from .steps_decisions import Decision
from .stub_decisions import Answer, Failure


def decision(*entries, status: str = "applied", detail: str = "") -> Decision:
    """A choice key becomes an Answer whose usage is fixed by its position in the decision:
    1000 + 10 * index input tokens and one output token."""
    script = tuple(
        entry
        if isinstance(entry, Failure)
        else Answer(entry, {"input_tokens": 1000 + 10 * index, "output_tokens": 1})
        for index, entry in enumerate(entries)
    )
    return Decision(script, status, detail)


def wait() -> Decision:
    return decision("wait", status="wait")


# Composition keys on frames 2 and 16 (providers/decisions/questions.py:71-83). Fragments are
# numbered in inventory order, which starts with the task instruction's words
# (composition.py:14, 47-54).
ADA, LOVELACE, SPACE = "fragment_3", "fragment_4", "char_000020"
# Frame 40: the datalist's "Ada Lovelace" is the eleventh literal (candidates.py:101-102, 263-271).
SUGGESTED_NAME = "type_11"

WEB_LONG = [
    # 0-9: compose the name, accept the terms, use the map
    decision("click_P1"),
    decision("compose_text", ADA, SPACE, LOVELACE, "finish"),  # types "Ada Lovelace"
    decision("press_tab"),
    decision("click_P2"),
    decision("click_R1"),
    wait(),
    # two identical turns: the JEV digest and ledger compact them
    # (providers/decisions/history.py:13-38)
    wait(),
    decision("scroll_down_R1"),
    decision("pan_right_R1"),
    decision("click_P3"),
    # 10-19
    decision("pan_down_R1"),
    decision("scroll_up_R1"),
    wait(),
    decision("click_P1"),
    decision("select_all"),
    # Every edit operation once: a character, a code point by six hex digits, delete, a Unicode
    # round left with Back, clear, then cancel, which yields WAIT (selection.py:356-414).
    decision(
        "compose_text",
        "char_000041",
        "unicode",
        "hex_0",
        "hex_0",
        "hex_0",
        "hex_0",
        "hex_E",
        "hex_9",
        "delete_last",
        "unicode",
        "back",
        "clear",
        "cancel",
        status="wait",
    ),
    decision("delete_text"),
    decision("press_tab"),
    wait(),
    decision("click_P2"),
    # 20-29: decision 20 is the first with folded screenshots
    decision("click_P3"),
    decision("pan_left_R1"),
    wait(),
    decision("scroll_down_R1"),
    decision("click_R1"),
    decision("pan_up_R1"),
    wait(),
    decision("click_P2"),
    decision("scroll_up_R1"),
    wait(),
    # 30-39: from frame 31 Next is P4
    decision("click_P4"),
    decision("pan_down_R1"),
    wait(),
    decision("click_P3"),
    decision("scroll_down_R1"),
    decision("click_P4"),
    wait(),
    decision("pan_left_R1"),
    decision("click_P1"),
    # 261 candidates: groups of 255 and 6, then their winners (selection.py:159-229)
    decision(SUGGESTED_NAME, "pan_up_R1", SUGGESTED_NAME),
    # 40-49
    decision("press_enter"),
    decision("click_P2"),
    wait(),
    decision("click_R1"),
    decision("pan_right_R1"),
    decision("click_P3"),
    wait(),
    decision("scroll_up_R1"),
    decision("click_P4"),
    wait(),
    # 50-59: decision 50 is the first with an archived-action ledger
    decision("click_P3"),
    decision("pan_up_R1"),
    decision("click_P4"),
    wait(),
    decision("click_R1"),
    decision("scroll_down_R1"),
    wait(),
    decision("click_P2"),
    decision("click_P3"),
    decision("click_P4", status="not_applied_time_limit"),
]
GROUPED_DECISION = 39


# A browser-game page: one canvas with declared keys and no DOM controls -----------------------
GAME_INSTRUCTION = (
    "Steer the ship with the arrow keys, fire with Space, and drag on the board to aim."
)
GAME_URL = "http://localhost:18081/"
GAME_TITLE = "Replay game"
BOARD_BOX = (360, 140, 1560, 1040)


class ReplayGameTask(frames.ReplayTask):
    """The attributes the runner and BrowserSceneCapture read for a browser game."""

    task_id = "replay-game"
    category = "browser_game"  # GAME_ROLE (harnesses/default/prompts.py SCENARIO_ROLES)
    keys = ("ArrowLeft", "ArrowRight", "Space")  # read by load_tool_spec (interactables.py:868)

    def interactable_providers(self):
        """As catalog games declare their keys (realtime_tasks/game/runtime.py:297-298)."""
        return ({"type": "keyboard", "keys": tuple(self.keys)},)


def game_screenshot(frame: int) -> bytes:
    image = Image.new("RGB", frames.DESKTOP_SIZE, (20, 20, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1919, 79), fill=(60, 60, 60))  # browser toolbar, outside the page
    draw.rectangle(BOARD_BOX, fill=(10, 30, 70))
    x = 420 + (61 * frame) % 1080
    draw.polygon([(x, 980), (x - 30, 1030), (x + 30, 1030)], fill=(240, 240, 240))  # the ship
    for index in range(4):  # asteroids drifting down the board
        ax, ay = 450 + 260 * index, 160 + (47 * (frame + 3 * index)) % 760
        draw.ellipse((ax - 25, ay - 25, ax + 25, ay + 25), fill=(150, 110, 70))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def game_probe(frame: int, *, text_observation: bool = False) -> dict:
    """What the extractor returns for the game page (cdp_interactables.py:470-484, 569-583)."""
    board = {
        "id": "canvas-1",
        "dom_order": 0,
        "source": "dom",
        "frame_path": "",
        "role": "canvas",
        "label": "Game board",
        "context": "",
        "semantic_label": "Game board",
        "tag": "canvas",
        "bbox": list(BOARD_BOX),
        "click_point": [960, 590],
        "css_size": [1200, 900],
        "backing_size": [1200, 900],
        "device_pixel_ratio": 1,
        "actions": ["click"],
        "enabled": True,
        "occluded": False,
        "unavailable_reasons": [],
        # The spec default for a task with declared keys (interactables.py:868-869).
        "internal_target_coverage": "not_applicable",
        "interaction": "",
        "internal_target_count": 0,
    }
    readiness = {"document_ready_state": "complete", "selector_present": False, "selector_text": ""}
    payload = json.dumps(
        {
            "actionable_now": [],
            "visible_but_unavailable": [],
            "canvas_surfaces": [board],
            "readiness": readiness,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    window = {
        "scene": {
            "status": "ok",
            "kind": "browser_dom",
            "application": "google-chrome",
            "url": GAME_URL,
            "title": GAME_TITLE,
            "active": True,
            "visible": True,
            "target_id": "replay-game-target",
        },
        "binding": {
            "stable": True,
            "revision": "",
            "fingerprint": hashlib.sha256(payload.encode()).hexdigest(),
        },
        "actionable_now": [],
        "visible_but_unavailable": [],
        "canvas_surfaces": [board],
        "limitations": ["browser_chrome_not_inspected"],
        "readiness": readiness,
        "metrics": {
            "screenX": 0,
            "screenY": 0,
            "outerWidth": 1920,
            "outerHeight": 1080,
            "innerWidth": 1920,
            "innerHeight": 1000,
            "devicePixelRatio": 1,
            "screenWidth": 1920,
            "screenHeight": 1080,
            "contentLeft": 0,
            "contentTop": 80,
            "inspectedFrameDocuments": 1,
            "inaccessibleFrames": 0,
        },
        "window_id": "replay-game-window",
        "title": GAME_TITLE,
        "active": True,
    }
    if text_observation:
        window.update(
            frames.ax_fields(
                [
                    frames.ax_node(1, "RootWebArea", GAME_TITLE, children=(2,)),
                    frames.ax_node(2, "Canvas", "Game board"),
                ]
            )
        )
    return {**window, "binding": dict(window["binding"]), "windows": [window]}


GAME_SCENE = {"screenshot": game_screenshot, "probe": game_probe}

# Pointer composition: x then y, each a coarse hundred and then the value within it, unless the
# coarse choice is 1000 (selection.py:127-157). The model image is 768x432 (capture.py:315-325).
GAME_DECISIONS = [
    decision("game_pointer_screen_click", "400", "437", "200", "250"),  # click (336, 108)
    # (393, 13) in the model image is (982, 32) on the desktop (entity/actions.py:273-280), on the
    # browser toolbar above the page, which the environment refuses (sandbox/actions.py:473-488)
    decision(
        "game_pointer_screen_click",
        "500",
        "512",
        "0",
        "30",
        status="rejected",
        detail=OUTSIDE_PAGE,
    ),
    decision("game_pointer_R1_mouse_down", "1000", "500", "512"),  # R1@1000,512
    decision("game_pointer_R1_move", "700", "750", "500", "500"),  # R1@750,500
    decision("game_mouse_release"),
    # The second key's options keep the first pick's gap (selection.py:105-125)
    decision("game_hold_pair", "0", "2", "0.5"),  # ArrowLeft+Space for 0.5 s
    decision("game_key_1"),  # tap ArrowRight
    decision("game_hold_2_1.0"),  # hold Space for 1 s
    wait(),
    decision("click_R1"),
]


# Failures -------------------------------------------------------------------------------------
# One retry: a transient 503 on the second question of a composition (selection.py:313-354).
RETRY = [
    decision("click_P1"),
    decision(
        "compose_text", Failure(503, "Provider returned error"), ADA, SPACE, LOVELACE, "finish"
    ),
]

TOO_LARGE = Failure(413, "Request exceeds the model's context window")
# Too many candidates. Frame 1 offers 11 (wait, click_P1-P3, click_R1, scroll x2, pan x4). A
# rejected group is split in halves, both asked next (selection.py _choose); winners meet in
# round 1.
CANDIDATE_OVERFLOW = [
    decision(TOO_LARGE, "click_P1", "pan_up_R1", "click_P1"),
    # Frame 2 offers 20: halves of 10, 5, 2, then one option. History is never reduced at
    # question time; the error ends the decision.
    decision(TOO_LARGE, TOO_LARGE, TOO_LARGE, TOO_LARGE, TOO_LARGE),
]
# History too long: the first decisions of the long script under a small declared window
# (test_decisions.py PRESSURE_WINDOWS). Before decision 12 the runtime archives the oldest ten
# turns into the ledger, and the questions go out with the shorter history.
HISTORY_OVERFLOW = WEB_LONG[:14]
HISTORY_OVERFLOW_ARCHIVE = 12

# A text-heavy frame: the terms of service, in Chinese, below the form. JEV reads them in the
# accessibility tree (at most 16000 characters, sandbox/browser/ax.py MAX_AX_CHARS), three UTF-8
# bytes a character, so the frame's observation is 25-32 KB. The Name field must not be focused
# on it, so that the terms add no typing candidates (harnesses/jev/candidates.py
# observation_candidates).
TERMS_CLAUSES = 172


def terms_clause(number: int) -> str:
    return (
        f"第{number}条：用户在提交表单之前应当完整阅读本服务条款，"
        "平台仅为完成本次登记而处理所填写的姓名信息。"
    )


def terms_scene(terms_frame: int) -> dict:
    """The form, with the terms on frame ``terms_frame``."""

    def probe(frame: int, *, text_observation: bool = False) -> dict:
        window = frames.probe(frame, text_observation=text_observation)
        if text_observation and frame == terms_frame:
            root, *form = frames.form_ax_nodes(frame)
            region = len(form) + 2  # the form's nodes are 1, the root, to len(form) + 1
            clauses = range(region + 1, region + 1 + TERMS_CLAUSES)
            terms = [
                frames.ax_node(region, "region", "服务条款", children=tuple(clauses)),
                *(frames.ax_node(n, "StaticText", terms_clause(n - region)) for n in clauses),
            ]
            ax = frames.ax_fields(
                [{**root, "childIds": [*root["childIds"], str(region)]}, *form, *terms]
            )
            window.update(ax)
            window["windows"][0].update(ax)
        return window

    return {"probe": probe}


# Decision 4 sees the terms; the decisions around it are the long script's.
LARGE_OBSERVATION = WEB_LONG[:6]
LARGE_OBSERVATION_DECISION = 4
# Decision 24 sees the terms with 24 turns retained (test_decisions.py ARCHIVE_WINDOWS): the
# runtime archives two blocks before its request goes out, or, under a smaller window, archives
# every turn and still cannot send it, which ends the episode, or archives every turn and leaves
# ten actions out, which the decisions after it do too.
TERMS_ARCHIVE_DECISION = 24
TERMS_ARCHIVE = WEB_LONG[:26]
TERMS_EXHAUSTED = [*WEB_LONG[:TERMS_ARCHIVE_DECISION], decision()]
TERMS_OMISSION = WEB_LONG[:28]

# The long script under a window its ledger outgrows (test_decisions.py LEDGER_RUNS), up to
# decision 38: decision 39's frame lists 240 names, which no window this small holds.
LEDGER_OVERFLOW = WEB_LONG[:39]
