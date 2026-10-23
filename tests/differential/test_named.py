"""The named scripts (scenarios.py) three ways: the shell agrees with the kernel's simulation,
and today's loop with the shell once the expected differences a script names are applied, each
of which it shows. No named script has an instant where the fake clock's threads tie
(simulated.ties)."""

from __future__ import annotations

import pytest

from .compare import as_expected, compare, describe, record
from .scenarios import NAMED, PENDING
from .simulated import ties


@pytest.mark.parametrize(
    "script",
    [
        pytest.param(script, marks=pytest.mark.xfail(strict=True, reason=PENDING[script.name]))
        if script.name in PENDING
        else script
        for script in NAMED
    ],
    ids=[script.name for script in NAMED],
)
def test_named_script(script, tmp_path):
    assert not ties(script), ties(script)
    verdict = compare(script, tmp_path)
    record(verdict, named=True)
    assert as_expected(verdict), describe(verdict, named=True)
