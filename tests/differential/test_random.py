"""Random scripts (generate.py) three ways, as the named ones: seeds 0 to 24, or as DIFF_SEEDS
says: a count ("500") or a range ("500:1000"); anything else, or no seed, fails
test_the_seeds_are_given_right. A failing script is shrunk to the smallest that fails the same
way, printed so that it can become a named script."""

from __future__ import annotations

import os
import re

import pytest

from .compare import compare, describe, record
from .generate import generate, shrink
from .rules import Verdict
from .script import Attempt, Script, Turn, parse
from .simulated import ties


def _seeds(text: str) -> tuple[range, str | None]:
    first, _, last = text.rpartition(":")
    try:
        seeds = range(int(first or 0), int(last))
    except ValueError:
        return range(0), f"DIFF_SEEDS={text!r} is neither a count nor a range"
    return seeds, None if seeds else f"DIFF_SEEDS={text!r} gives no seed"


SEEDS, _SEEDS_ERROR = _seeds(os.environ.get("DIFF_SEEDS", "25"))


def test_the_seeds_are_given_right():
    assert _SEEDS_ERROR is None, _SEEDS_ERROR


def _signature(verdict) -> tuple:
    """How a verdict fails: the rules it did not expect, and where the first unexplained
    difference and the first one between the shell and the simulation are, with decision and
    item numbers left out."""

    def where(differences: list[str]) -> str | None:
        if not differences:
            return None
        path = differences[0].split(":")[0]
        return re.sub(r"\[\d+\]", "[<n>]", re.sub(r"\.\d+(?=\.|\[|$)", ".<k>", path))

    return tuple(verdict.unexpected), where(verdict.problems), where(verdict.apart)


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
        shrunk = compare(candidate, tmp_path_factory.mktemp("shrink"))
        return not shrunk.ok and _signature(shrunk) == failure

    try:
        smallest = repr(shrink(script, fails))
    except Exception as error:  # noqa: BLE001 - the failure found is what the test reports
        smallest = f"none: shrinking raised {error!r}"
    pytest.fail(f"{describe(verdict)}\nthe smallest script that fails the same way:\n{smallest}")


def test_shrinking_keeps_the_failure():
    """A script that fails while it holds a refused click loses every other turn, its end, its
    frozen stream and a frame."""
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

    assert shrink(script, fails) == Script("wall", 2.0, (Turn("click", rejected="no"),))


def test_shrinking_snaps_times_and_zeroes_operations():
    """Times go onto the frames and operations take none, as long as the script still fails."""
    script = Script(
        "wall",
        3.0,
        (Turn("wait:0.3", attempts=(Attempt(0.7),)), Turn("hold:a:0.8")),
        ends_at_s=1.3,
        check_s=0.05,
    )

    def fails(candidate) -> bool:
        kinds = [parse(turn.action).kind for turn in candidate.turns]
        return kinds == ["wait", "hold"] and candidate.ends_at_s is not None

    assert shrink(script, fails) == Script(
        "wall", 2.0, (Turn("wait:0.5"), Turn("hold:a:1")), ends_at_s=1.0
    )


def test_a_failure_names_where_the_shell_and_the_simulation_part():
    """Two failures are the same when they part at the same place, whatever the decision or the
    item; a verdict that passes has no difference anywhere."""
    script = Script("wall", 2.0, (Turn("click"),))

    def verdict(apart=(), problems=()) -> Verdict:
        return Verdict(script, problems=list(problems), apart=list(apart))

    third = verdict(apart=["decisions.3.resolved.status: 'applied' -> 'wait'"])
    fifth = verdict(apart=["decisions.5.resolved.status: 'rejected' -> 'wait'"])
    trace = verdict(problems=["trace.InferenceCompleted[3][0].latency_s: 1 -> 2"])
    other = verdict(problems=["trace.InferenceCompleted[4][1].latency_s: 1 -> 3"])
    assert _signature(third) == _signature(fifth) != _signature(verdict())
    assert _signature(trace) == _signature(other)
    assert not third.ok
