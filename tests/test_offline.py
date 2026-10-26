"""A test folder run on its own stays offline, and so does a run's import of LiteLLM.

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
# An audit hook that records and refuses every non-loopback connection and name lookup.
REFUSE = r"""
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
"""
# A child pytest whose first audit hook, installed before pytest imports anything, records and
# refuses every non-loopback connection and name lookup the run attempts.
CHILD = REFUSE + r"""
import pytest

record, arguments = sys.argv[1], sys.argv[2:]  # before any test can rewrite sys.argv
code = int(pytest.main(arguments))
with open(record, "w") as out:
    json.dump({"exit": code, "attempts": attempts}, out)
sys.stdout.flush()
os._exit(code)
"""


# A process that loads the catalog, which reads LiteLLM's model map (providers/registry.py
# _litellm_model_map), and builds a LiteLLM driver (providers/litellm.py LiteLLMDriver._load_sdk),
# as a run does. It records each .env python-dotenv is asked to load: LiteLLM's import loads one by
# default (litellm/__init__.py:29-30).
RUN = REFUSE + r"""
import dotenv

loads = []
dotenv.load_dotenv = lambda *args, **kwargs: loads.append(repr((args, kwargs))) or False

from realtime_framework.entity.config import ModelConfig
from realtime_framework.providers import registry
from realtime_framework.providers.litellm import ChatCompletionsDriver

registry.get("litellm-vlm")
ChatCompletionsDriver(
    ModelConfig(adapter_type="chat_completions", model_name="m", api_base_url="http://127.0.0.1:9/v1")
)
with open(sys.argv[1], "w") as out:
    json.dump({"attempts": attempts, "dotenv_loads": loads}, out)
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


def test_a_run_imports_litellm_without_a_dotenv_or_the_network(tmp_path):
    """Neither setting LiteLLM's import reads (providers/litellm.py import_litellm) comes from the
    shell, yet LiteLLM's import neither looks up a host to fetch its model cost map nor loads a
    .env."""
    record = tmp_path / "record.json"
    unset = {"LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_MODE"}
    environ = {k: v for k, v in os.environ.items() if k not in unset}
    child = subprocess.run(
        [sys.executable, "-c", RUN, str(record)],
        cwd=ROOT,
        env=environ,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert record.is_file(), child.stdout[-4000:] + child.stderr[-4000:]
    assert json.loads(record.read_text()) == {"attempts": [], "dotenv_loads": []}
