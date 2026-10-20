"""Choice harness: enumerate actions and select through the registered Decisions encoder.

The default harness builds the observation JSON and the answer options when context policy
item 12 asks for them (harnesses/jev/observation.py).
"""

from __future__ import annotations

from realtime_framework.harnesses.default.harness import DefaultHarness


class JevHarness(DefaultHarness):
    pass
