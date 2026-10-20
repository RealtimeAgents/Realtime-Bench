"""Process settings that keep the replay tests hermetic and offline.

conftest.py imports this module before any framework module is imported.
"""

from __future__ import annotations

import os
import sys

# Importing LiteLLM fetches its model cost map over the network unless this is set
# (litellm/litellm_core_utils/get_model_cost_map.py:643). The replay model id is in neither the
# bundled nor the remote map, so the choice changes no request.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"

# Environment settings that change what the model receives. Baselines use the code defaults.
# Frame encoding is read once, at import (FRAME_ENCODING in sandbox/observation/capture.py).
FRAME_ENV = ("FRAME_MAX_DIM", "FRAME_JPEG_QUALITY", "FRAME_IMAGE_DETAIL")
# Screenshot folding and history archiving are read on every request (entity/config.py:41-90).
HISTORY_ENV = (
    "STREAM_ACTIVE_VISUAL_BUDGET",
    "STREAM_VISUAL_FOLD_BLOCK_SIZE",
    "STREAM_DETAILED_HISTORY_TURNS",
    "STREAM_DETAILED_HISTORY_DROP_BLOCK_SIZE",
)
# The runner's grace, drain, freeze and watchdog bounds are read once, at import
# (RUNNER_TIMING in evaluation/runner.py).
RUNNER_ENV = (
    "STREAM_DEADLINE_GRACE_S",
    "STREAM_ABANDONED_DRAIN_S",
    "STREAM_OBSERVATION_FREEZE_N",
    "STREAM_PROVIDER_WATCHDOG_S",
)
for _name in FRAME_ENV + HISTORY_ENV + RUNNER_ENV:
    os.environ.pop(_name, None)


def _refuse_remote_connections(event, args):
    """Every request must reach the local stub; refuse any other IP connection."""
    if event == "socket.connect":
        address = args[1]
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1"):
            raise ConnectionRefusedError(f"replay tests are offline; refused {address!r}")


sys.addaudithook(_refuse_remote_connections)
