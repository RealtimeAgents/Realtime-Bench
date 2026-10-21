"""The named scripts (scenarios.py) three ways: the shell agrees with the kernel's simulation,
and today's loop with the shell once the expected differences a script names are applied. No
named script has an instant where the fake clock's threads tie (simulated.ties)."""

from __future__ import annotations

import pytest

from .compare import compare, describe, record
from .scenarios import NAMED
from .simulated import ties


@pytest.mark.parametrize("script", NAMED, ids=[script.name for script in NAMED])
def test_named_script(script, tmp_path):
    assert not ties(script), ties(script)
    verdict = compare(script, tmp_path)
    record(verdict, named=True)
    assert verdict.ok, describe(verdict)
