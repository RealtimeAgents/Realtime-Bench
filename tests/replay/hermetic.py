"""Process settings that keep the tests hermetic and offline.

tests/conftest.py imports this module before any framework module is imported, whichever test
folder or file is run.
"""

from __future__ import annotations

import os
import socket
import sys

# LiteLLM reads its settings once, at its import, and some change requests: LITELLM_DROP_PARAMS
# and LITELLM_ROUTE_ALL_CHAT_OPENAI_TO_RESPONSES among them (litellm/__init__.py:249-270). Tests
# run with its defaults.
for _name in [name for name in os.environ if name.startswith("LITELLM_")]:
    os.environ.pop(_name)
# Importing LiteLLM fetches its model cost map over the network unless this is set
# (litellm/litellm_core_utils/get_model_cost_map.py:646). The replay model id is in neither the
# bundled nor the remote map, so the choice changes no request.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"

# Environment settings that change what the model receives. Baselines use the code defaults.
# Frame encoding is read once, at import (FRAME_ENCODING in sandbox/observation/capture.py).
FRAME_ENV = ("FRAME_MAX_DIM", "FRAME_JPEG_QUALITY", "FRAME_IMAGE_DETAIL")
# Screenshot folding and history archiving are read when the context policy is built
# (entity/config.py:41-90).
HISTORY_ENV = (
    "STREAM_ACTIVE_VISUAL_BUDGET",
    "STREAM_VISUAL_FOLD_BLOCK_SIZE",
    "STREAM_DETAILED_HISTORY_TURNS",
    "STREAM_DETAILED_HISTORY_DROP_BLOCK_SIZE",
)
# The runner's drain, freeze and watchdog bounds are read once, at import (RUNNER_TIMING in
# evaluation/runner.py), and so is the legacy loop's deadline grace (tests/differential).
RUNNER_ENV = (
    "STREAM_DEADLINE_GRACE_S",
    "STREAM_ABANDONED_DRAIN_S",
    "STREAM_OBSERVATION_FREEZE_N",
    "STREAM_PROVIDER_WATCHDOG_S",
)
for _name in FRAME_ENV + HISTORY_ENV + RUNNER_ENV:
    os.environ.pop(_name, None)


LOOPBACK = ("127.0.0.1", "::1", "localhost")


def _refuse_remote_connections(event, args):
    """Every request must reach the local stub; refuse any other IP connection, and the name
    lookup that would precede it (a lookup alone already queries the resolver)."""
    if event == "socket.connect":
        address = args[1]
        if isinstance(address, tuple) and address[0] not in LOOPBACK:
            raise ConnectionRefusedError(f"tests are offline; refused {address!r}")
    elif event == "socket.getaddrinfo":
        host = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if host is not None and host not in LOOPBACK:
            raise socket.gaierror(f"tests are offline; refused a lookup of {host!r}")


sys.addaudithook(_refuse_remote_connections)
