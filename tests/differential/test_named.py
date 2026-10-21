"""Today's loop against the kernel's simulation on the named scripts (scenarios.py): once the
expected differences a script names are applied, the two agree."""

from __future__ import annotations

import pytest

from .compare import compare, describe, record
from .scenarios import NAMED


@pytest.mark.parametrize("script", NAMED, ids=[script.name for script in NAMED])
def test_named_script(script, tmp_path):
    verdict = compare(script, tmp_path)
    record(verdict, named=True)
    assert verdict.ok, describe(verdict)
