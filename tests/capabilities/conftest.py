"""pytest loads this before the test modules, so the hermetic settings precede framework imports."""

from tests.replay import hermetic  # noqa: F401
