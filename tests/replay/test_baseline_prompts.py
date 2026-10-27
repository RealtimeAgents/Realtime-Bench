"""What the model is shown in every replay baseline, read from the recorded requests themselves.

A frame's text and SoM legend stay in the context once newer frames follow, so no frame but the
newest of a request says what holds only for the newest one; which screenshot's marks an action
uses is said once, in the system instruction (user, 10-27).

A run's prompts and function declarations describe only the clock it is on: a wall-clock run's
say nothing of the token clock, and a token-clock or pause run's (the token clock with B
unlimited) nothing of the wall clock (user, 10-27).

A request's model-visible texts are a stateless request's body, a canonical request's blocks and
answer, and the client messages of each OpenAI Realtime or Gemini Live connection, whose session
keeps them. The baselines that record only digests of their requests show no text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from realtime_framework.harnesses.default.prompts import SCENARIO_ROLES
from realtime_framework.sandbox.observation.som import FRAME_INTERACTABLES_HEADER

from . import baseline, baseline_decisions

BASELINES = sorted((Path(__file__).parent / "baselines").glob("*.jsonl"))

# Words that hold only for the newest frame (tests/harness/test_prompts.py NEWEST_ONLY).
NEWEST_ONLY = ("current", "newest", "latest", "this decision", "only these", "use only")
# The mark rules of a system instruction whose model writes marks (prompts.py TURN_POINTER_RULES);
# a choice model's picks candidates bound to the newest observation's marks.
MARK_RULES = (
    "P/R marks belong only to the screenshot that supplies them.",
    "Use the newest screenshot's P/R marks when\n  supplied;",
    "If no suitable mark exists, replace target with x/y",
)
WRITES_MARKS = "AVAILABLE GUI ACTIONS"
# A legend line naming one mark, or the page's keys (som.py _mark_map): page content, not the
# harness's.
PAGE_LINE = re.compile(r"[PRU]\d+ |KEYS ")
# What only one clock's prompts say, case aside (tests/harness/test_prompts.py).
TOKEN_CLOCK_WORDS = ("token clock", "simulated", "kappa", "charged tokens")
WALL_CLOCK_WORDS = ("wall clock", "physical elapsed time", "environment time keeps")
# The clock note of each clock's system instruction (prompts.py budget_clock_prompt).
CLOCK_NOTES = {
    "wall": ("WALL CLOCK: ",),
    "finite_b": ("TOKEN CLOCK: ", " charged tokens/frame;"),
    "pause": ("TOKEN CLOCK: ", "B is unlimited"),
}


def _read(path: Path) -> list[dict]:
    blobs = any(json.loads(line)["record"] == "blob" for line in path.open(encoding="utf-8"))
    return (baseline_decisions.read if blobs else baseline.read)(path)


def _strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def conversations(records: list[dict]) -> list[list[str]]:
    """The model-visible texts of each request, or each live connection, in the order sent."""
    blocks = {record["key"]: record["block"] for record in records if record["record"] == "block"}
    requests, sessions = [], {}
    for record in records:
        for request in record.get("requests", ()):
            requests.append(list(_strings(request["body"])))
        if record["record"] == "request":
            items = [blocks[item["block"]] for item in record["items"]]
            requests.append([*_strings(items), *_strings(record["answer"])])
        for sent in record.get("client_events", ()):  # OpenAI Realtime
            sessions.setdefault(("realtime", sent["connection"]), []).extend(
                _strings(sent["events"])
            )
        for frame in record.get("frames", ()):  # Gemini Live
            if frame["dir"] == "c2s":
                sessions.setdefault(("live", frame["conn"]), []).extend(_strings(frame["msg"]))
    return requests + list(sessions.values())


def harness_text(frame: str) -> str:
    """What the harness wrote on a frame: its frame prompt, the line before its interactables
    (what precedes it in the same text, Luna's system instruction or the feedback on the action
    before, is not the frame's), and the lines of its SoM legend that name no mark; a choice
    model's legend is its observation JSON's."""
    before, _, rest = frame.partition(FRAME_INTERACTABLES_HEADER)
    if rest.startswith("{"):
        rest = json.loads(rest)["mark_legend"]
    legend = (line for line in rest.split("\n") if not PAGE_LINE.match(line))
    return "\n".join([before.rpartition("\n")[2], *legend])


def _system_texts(texts: list[str]) -> list[str]:
    return [text for text in texts if text.startswith(tuple(SCENARIO_ROLES.values()))]


def clock(records: list[dict]) -> str:
    """The baseline's clock, from its header: "wall", "finite_b" or "pause". A header that names
    none, and a canonical snapshot, which has none, is of build_episode's default wall clock
    (tests/replay/build.py)."""
    header = next((record for record in records if record["record"] == "header"), {})
    named = header.get("clock", header.get("clock_mode", "wall"))
    if isinstance(named, dict):
        mode, tokens = named.get("mode", named.get("clock_mode")), named.get("tokens_per_frame")
    else:
        mode, tokens = named, header.get("tokens_per_frame")
    if mode == "wall":
        return "wall"
    return "pause" if tokens is None else "finite_b"


def test_the_baselines_show_their_requests():
    shown = {path.name for path in BASELINES if any(conversations(_read(path)))}
    digests_only = {
        path.name
        for path in BASELINES
        if path.name.startswith(("chat_completions-runner-", "frame_rates-"))
    }
    assert shown == {path.name for path in BASELINES} - digests_only


@pytest.mark.parametrize("path", BASELINES, ids=lambda path: path.stem)
def test_older_frames_say_nothing_only_the_newest_holds(path):
    found = []
    for texts in conversations(_read(path)):
        frames = [text for text in texts if FRAME_INTERACTABLES_HEADER in text]
        for frame in frames[:-1]:
            written = harness_text(frame).lower()
            found += [(word, frame[:120]) for word in NEWEST_ONLY if word in written]
        for system in _system_texts(texts):
            if WRITES_MARKS in system:
                found += [(rule, "system") for rule in MARK_RULES if system.count(rule) != 1]
    assert found == []


@pytest.mark.parametrize("path", BASELINES, ids=lambda path: path.stem)
def test_the_prompts_describe_only_the_run_s_clock(path):
    records = _read(path)
    run_clock = clock(records)
    other = WALL_CLOCK_WORDS if run_clock != "wall" else TOKEN_CLOCK_WORDS
    found = []
    for texts in conversations(records):
        systems = _system_texts(texts)
        assert systems, "a request without its system instruction"
        found += [
            (note, "system") for system in systems for note in CLOCK_NOTES[run_clock]
            if note not in system
        ]
        found += [
            (word, text[:120]) for text in texts for word in other if word in text.lower()
        ]
    assert found == []


def test_older_frames_are_checked_on_every_path():
    """Every API whose requests keep older frames has some; JEV's digest keeps none
    (default/request.py _digest_items)."""
    paths = set()
    for path in BASELINES:
        for texts in conversations(_read(path)):
            if sum(FRAME_INTERACTABLES_HEADER in text for text in texts) > 1:
                paths.add(path.name.split("-")[0])
    assert paths == {
        "anthropic_messages",
        "canonical",
        "chat_completions",
        "external_driver",
        "gemini_live",
        "litellm",
        "moonshot",
        "openai_decisions",
        "openai_realtime",
        "openrouter",
    }
