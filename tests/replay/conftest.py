"""pytest loads this before the test modules, so the hermetic settings precede framework imports."""

from . import hermetic  # noqa: F401
