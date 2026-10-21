"""Random differential scripts from a seed, and greedy shrinking of one that fails
(wave2-plan.md, P4 随机脚本).

Scripts take the acceptance frame rates (1/1, 60/1, 30/7, 2/2) and put times and token counts on
the points where the rules turn: whole input frames, half and quarter frames, the deadline, B and
its multiples. On the token clock they fail runs of end checks and reads inside taps, which X12
counts. They leave out what the rules do not compare one by one: retries, wall reads that fail
(where the kernel's periodic reads and today's fall at one instant, the order is not today's to
show; named scripts cover them), operations that take time on the wall clock (each named script
that needs one says so), and wall replies near the deadline plus today's 2 s grace or the per-move
cap, where today's bounded wait and the reply would fall due at one instant (wave2-plan.md, R13);
the cap and the think budget appear on the token clock only. Random scripts may show any expected
difference; only what no rule explains fails them.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import replace

from .rules import ORDER
from .script import Attempt, Script, Turn, parse
from .simulated import ties

RATES = ((1.0, 1.0), (60.0, 1.0), (30.0, 7.0), (2.0, 2.0))
BUDGETS = (64, 100, 160, 256)


def generate(seed: int) -> Script:
    """The script of a seed: the first the seed's draws give that no instant ties on the wall
    clock's threads (simulated.ties)."""
    rng = random.Random(seed)
    while True:
        script = _draw(rng, seed)
        if not ties(script):
            return script


def _draw(rng: random.Random, seed: int) -> Script:
    clock = rng.choice(("wall", "token", "pause"))
    env_fps, input_fps = rng.choice(RATES)
    frame = 1.0 / input_fps
    duration_s = (rng.randint(2, 6) + rng.choice((0.0, 0.25, 0.5, 0.75))) * frame
    tokens_per_frame = rng.choice(BUDGETS) if clock == "token" else None
    count = math.ceil(duration_s * input_fps) + 2  # a decision takes at least one input frame
    turns = tuple(_turn(rng, clock, frame, tokens_per_frame) for _ in range(count))
    fields: dict = {}
    if rng.random() < 0.3:
        fields["ends_at_s"] = rng.randint(1, max(1, int(duration_s * 4 * env_fps))) / (4 * env_fps)
    if rng.random() < 0.15:
        fields["ends_after"] = rng.randint(1, 3)
        fields["end_reason"] = rng.choice((None, "task_success"))
    if rng.random() < 0.05:
        fields["frozen_from"] = rng.randint(1, 3)
    if clock == "wall":
        if rng.random() < 0.3:
            fields["snapshot_s"] = rng.choice((0.25, 0.5, 1.5))
    else:
        if rng.random() < 0.1:
            start = rng.randint(1, 12)
            fields["check_errors"] = tuple(range(start, start + rng.randint(1, 4)))
        # The environment cuts a tap where a lease ends and the simulation does not
        # (tests/kernel/simulation.py:14-16), so reads inside taps are numbered alike only
        # without holds.
        if rng.random() < 0.06 and all(parse(turn.action).kind != "hold" for turn in turns):
            start = rng.randint(1, 3)
            fields["tap_check_errors"] = tuple(range(start, start + rng.randint(1, 2)))
        if rng.random() < 0.04:
            fields["advance_errors"] = (rng.randint(1, 8),)
    if clock == "token":  # in simulated seconds, so no reply can race them
        if rng.random() < 0.08:
            fields["per_move_cap_s"] = rng.choice((0.5, 1.0, 1.5)) * frame
        if rng.random() < 0.08:
            fields["think_budget_s"] = rng.choice((1.0, 2.0, 3.0)) * frame
    if rng.random() < 0.03:
        fields["snapshot_error"] = "snapshot read failed"
    return Script(
        clock,
        duration_s,
        turns,
        env_fps=env_fps,
        input_fps=input_fps,
        tokens_per_frame=tokens_per_frame,
        expect=frozenset(ORDER),
        name=f"random-{seed}",
        **fields,
    )


def _turn(rng: random.Random, clock: str, frame: float, tokens_per_frame: int | None) -> Turn:
    seconds = rng.choice((0.25, 0.5, 1.0, 1.5)) * frame
    word = rng.choices(
        (
            "click",
            "type:Ada",
            "key:tab",
            "key:tab:2",
            f"hold:shift:{seconds}",
            f"hold:a:{seconds}",
            f"wait:{seconds}",
            "wait",
            "done",
            "garbage",
        ),
        weights=(30, 8, 10, 4, 7, 7, 14, 4, 6, 6),
    )[0]
    fields: dict = {}
    refusable = parse(word).kind in (
        ("click", "type") if clock != "wall" else ("click", "type", "key", "hold")
    )
    if refusable and rng.random() < 0.05:
        fields["rejected"] = "the guest refused the action"
    elif refusable and rng.random() < 0.03:
        fields["fault"] = "guest connection reset"
    if clock == "wall":
        if rng.random() < 0.04:
            return Turn(word, attempts=(Attempt(hang=True),), **fields)
        # Shorter than the grace: a reply to a generation started before the deadline always
        # comes back before today's bounded wait gives up on it.
        latency = rng.choice((0.125, 0.25, 0.5, 0.75, 1.0, 1.25)) * rng.choice((1.0, frame))
        return Turn(word, attempts=(Attempt(latency),), **fields)
    if tokens_per_frame is None:
        return Turn(word, rng.randint(1, 400), **fields)
    if rng.random() < 0.03:
        return Turn(word, None, **fields)
    tokens = rng.choice(
        (0, tokens_per_frame // 4, tokens_per_frame // 2, tokens_per_frame, 2 * tokens_per_frame)
    ) + rng.choice((0, 0, 1, 7))
    return Turn(word, tokens, **fields)


# ---- shrinking --------------------------------------------------------------------------------


def shrink(script: Script, fails: Callable[[Script], bool], budget: int = 120) -> Script:
    """The smallest script found that still ``fails``: each pass tries one simplification after
    another and keeps the first that still fails, until none does or ``budget`` tries are spent."""
    tries = 0
    while tries < budget:
        for candidate in _smaller(script):
            tries += 1
            if _still_fails(fails, candidate):
                script = candidate
                break
            if tries >= budget:
                return script
        else:
            return script
    return script


def _still_fails(fails: Callable[[Script], bool], script: Script) -> bool:
    try:
        return fails(script)
    except (StopIteration, ValueError):
        return False  # the simplification ran a side out of replies, or made the script invalid


def _smaller(script: Script):
    if len(script.turns) > 1:
        yield script.but(turns=script.turns[:-1])
    for name, plain in (
        ("frozen_from", None),
        ("ends_after", None),
        ("ends_at_s", None),
        ("check_errors", ()),
        ("tap_check_errors", ()),
        ("advance_errors", ()),
        ("snapshot_error", None),
        ("snapshot_s", 0.0),
    ):
        if getattr(script, name) != plain:
            yield script.but(**{name: plain})
    frame = 1.0 / script.input_fps
    if script.duration_s > 2 * frame:
        yield script.but(duration_s=script.duration_s - frame)
    for k, turn in enumerate(script.turns):
        for simpler in _simpler(turn, script):
            yield script.but(turns=script.turns[:k] + (simpler,) + script.turns[k + 1 :])


def _simpler(turn: Turn, script: Script):
    if turn.rejected or turn.fault:
        yield replace(turn, rejected=None, fault=None)
    if turn.action != "click":
        yield replace(turn, action="click")
    if script.clock == "wall" and turn.attempts != (Attempt(),):
        yield replace(turn, attempts=(Attempt(),))
    if script.tokens_per_frame is not None and turn.output_tokens not in (None, 0):
        whole = turn.output_tokens - turn.output_tokens % script.tokens_per_frame
        if whole != turn.output_tokens:
            yield replace(turn, output_tokens=whole)
