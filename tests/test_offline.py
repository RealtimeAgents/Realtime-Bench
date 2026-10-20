"""A test folder run on its own stays offline.

pytest loads tests/conftest.py, and with it tests/replay/hermetic.py, whichever folder it is
given. tests/config once ran without it, and importing LiteLLM there looked up
raw.githubusercontent.com to fetch LiteLLM's model cost map.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = "tests/config"
# A child pytest whose first audit hook, installed before pytest imports anything, records and
# refuses every non-loopback connection and name lookup the run attempts.
CHILD = r"""
import json, os, socket, sys

LOOPBACK = (None, "127.0.0.1", "::1", "localhost")
attempts = []


def refuse(event, args):
    if event == "socket.connect" and isinstance(args[1], tuple) and args[1][0] not in LOOPBACK:
        attempts.append(f"connect {args[1][0]}")
        raise ConnectionRefusedError("offline")
    if event == "socket.getaddrinfo":
        host = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if host not in LOOPBACK:
            attempts.append(f"lookup {host}")
            raise socket.gaierror("offline")


sys.addaudithook(refuse)
import pytest

record, arguments = sys.argv[1], sys.argv[2:]  # before any test can rewrite sys.argv
code = int(pytest.main(arguments))
with open(record, "w") as out:
    json.dump({"exit": code, "attempts": attempts}, out)
sys.stdout.flush()
os._exit(code)
"""


def test_a_folder_run_alone_makes_no_remote_connection(tmp_path):
    record = tmp_path / "record.json"
    # Without the cost-map setting this process inherited from hermetic.py, as from a shell.
    environ = {k: v for k, v in os.environ.items() if k != "LITELLM_LOCAL_MODEL_COST_MAP"}
    child = subprocess.run(
        [sys.executable, "-c", CHILD, str(record), FOLDER, "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        env=environ,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert record.is_file(), child.stdout[-4000:] + child.stderr[-4000:]
    assert json.loads(record.read_text()) == {"exit": 0, "attempts": []}, child.stdout[-4000:]
