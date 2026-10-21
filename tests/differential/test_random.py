"""Random scripts (generate.py) three ways, as the named ones: seeds 0 to 24, or as DIFF_SEEDS
says: a count ("500") or a range ("500:1000"). A failing script is shrunk to the smallest that
fails the same way, printed so that it can become a named script."""

from __future__ import annotations

import os
import re

import pytest

from .compare import compare, describe, record
from .generate import generate, shrink
from .script import Attempt, Script, Turn
from .simulated import ties

_first, _, _last = os.environ.get("DIFF_SEEDS", "25").rpartition(":")
SEEDS = range(int(_first or 0), int(_last))


def _signature(verdict) -> tuple:
    """How a verdict fails: the rules it did not expect, and where the first unexplained
    difference is, decision numbers aside."""
    first = (
        re.sub(r"\.\d+\.", ".<k>.", verdict.problems[0].split(":")[0]) if verdict.problems else None
    )
    return tuple(verdict.unexpected), first


@pytest.mark.parametrize("seed", SEEDS)
def test_random_script(seed, tmp_path_factory):
    script = generate(seed)
    verdict = compare(script, tmp_path_factory.mktemp(script.name))
    record(verdict, named=False)
    if verdict.ok:
        return
    failure = _signature(verdict)

    def fails(candidate) -> bool:
        if ties(candidate):
            return False
        return _signature(compare(candidate, tmp_path_factory.mktemp("shrink"))) == failure

    smallest = shrink(script, fails)
    pytest.fail(f"{describe(verdict)}\nthe smallest script that fails the same way:\n{smallest!r}")


def test_shrinking_keeps_the_failure():
    """A script that fails while it holds a refused click loses every turn after it, its end,
    its frozen stream and a frame, and the turns before it become plain clicks."""
    script = Script(
        "wall",
        3.0,
        (
            Turn("wait:1", attempts=(Attempt(0.75),)),
            Turn("click", rejected="no"),
            Turn("type:Ada"),
        ),
        ends_at_s=1.0,
        frozen_from=2,
    )

    def fails(candidate) -> bool:
        return any(turn.rejected for turn in candidate.turns)

    assert shrink(script, fails) == Script(
        "wall", 2.0, (Turn("click"), Turn("click", rejected="no"))
    )
