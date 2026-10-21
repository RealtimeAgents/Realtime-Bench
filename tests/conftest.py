"""pytest loads this before any test module, whichever folder or file is run, so the hermetic and
offline settings (tests/replay/hermetic.py) precede every framework import.

A test that runs longer than TEST_TIMEOUT_S of real time fails instead of hanging the run: SIGALRM
interrupts the test's thread wherever it waits, and the stacks of every thread are printed to
stderr first. A participant stuck on the fake clock is caught sooner (tests/fake_clock.py); this
catches the test's own thread stuck in a wait the clock cannot see.
"""

import faulthandler
import signal
import sys

import pytest

from tests.replay import hermetic  # noqa: F401

TEST_TIMEOUT_S = 600


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    def timed_out(signum, frame):
        faulthandler.dump_traceback(sys.__stderr__, all_threads=True)
        pytest.fail(f"the test ran longer than {TEST_TIMEOUT_S:g} s")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, TEST_TIMEOUT_S)
    try:
        return (yield)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
