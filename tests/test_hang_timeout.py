"""The per-test timeout of tests/conftest.py."""

from __future__ import annotations

import threading

import pytest

from . import conftest


def test_a_test_stuck_in_a_real_wait_fails(monkeypatch):
    """A wait nothing will end, such as a blocking receive whose reply needs the fake clock to
    move, fails the test once the timeout passes."""
    monkeypatch.setattr(conftest, "TEST_TIMEOUT_S", 0.2)
    hook = conftest.pytest_runtest_call(None)
    next(hook)
    try:
        with pytest.raises(pytest.fail.Exception, match=r"^the test ran longer than 0.2 s$"):
            threading.Event().wait()
    finally:
        hook.close()
