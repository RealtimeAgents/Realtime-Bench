"""pytest loads this before any test module, whichever folder or file is run, so the hermetic and
offline settings (tests/replay/hermetic.py) precede every framework import."""

from tests.replay import hermetic  # noqa: F401
