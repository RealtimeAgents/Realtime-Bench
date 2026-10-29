"""The effective-config export reaches the runner's decision loop from each entry point."""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import realtime_framework.providers.registry as catalog
import realtime_tasks.registry as tasks
from realtime_framework.entity.context_policy import path_policy
from realtime_framework.evaluation.effective_config import export
from realtime_framework.harnesses.default.harness import INTERACTION_PROFILES
from realtime_framework.harnesses.registry import configure_agent

MODEL, TASK = "claude-opus-5", "gh_timed_chess"
KEY = "sk-export-test-not-a-key"


def test_the_trial_reaches_the_decision_loop():
    """The trial worker's dry run and the runner on its launch file, each in a fresh interpreter,
    as the trial worker execs it."""
    environ = {"HOME": os.environ.get("HOME", "/"), "PATH": "/usr/bin:/bin"}
    environ["OPENROUTER_API_KEY"] = KEY
    exported = export(
        environ=environ,
        env_file="/nonexistent/.env",
        entries=["trial"],
        models=[MODEL],
        tasks=[TASK],
    )
    trial = exported["entries"]["trial"]["models"][MODEL]
    assert trial["error"] is None
    assert trial["run"]["clock_mode"] == "wall"  # the trial worker's default (run_spec.py)
    # The hash _trial's runner records before the episode (reporting.py prepare_episode_report)
    # and after setup (runner.py main) agree.
    assert trial["expected_runtime_config_hash"] == trial["runtime_config_hash"]
    assert KEY not in json.dumps(exported), "the export must not contain secrets"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_the_system_prompt_hash_is_of_the_system_block_sent():
    """system_prompt_sha256 hashes the system block each agent's initialize() builds from its
    context policy's profile (item 11): the scenario prompt opens only the text-JSON one, and the
    function-call (Gemini Live) and choice (Luna, JEV) ones leave it out."""
    web = "rt_waitlist_fit"
    profiles = {
        "claude-opus-5": "standard_vlm",
        "gemini-live": "function_call",
        "gpt-6-luna-decisions": "multimodal_decisions",
        "jev-1.13": "jev_decisions",
    }
    environ = {"HOME": os.environ.get("HOME", "/"), "PATH": "/usr/bin:/bin"}
    environ.update(OPENROUTER_API_KEY=KEY, GEMINI_API_KEY=KEY, ALLOW_EXPERIMENTAL_MODEL="1")
    exported = export(
        environ=environ,
        env_file="/nonexistent/.env",
        entries=["trial"],
        models=list(profiles),
        tasks=[web],
    )
    task = tasks.get(web)
    for model, profile in profiles.items():
        record = exported["entries"]["trial"]["models"][model]
        assert (record["first_task"], record["error"]) == (web, None), model
        assert record["task"]["instruction_sha256"] == _sha256(task.description())
        config = catalog.to_model_config(model, fps=1.0)
        prompts = path_policy(config).prompts
        assert prompts[task.category] == profile
        # The agent as the runner configures and initializes it, on the trial worker's wall clock.
        agent = SimpleNamespace(task_instruction=task.description())
        configure_agent(
            agent,
            fps=1.0,
            tokens_per_frame=None,
            clock_mode="wall",
            category=task.category,
            interaction_mode="turn_based",
        )
        assert agent.streaming_note == record["clock_prompt"]
        sent = INTERACTION_PROFILES[profile].system_instruction(agent)
        assert record["task"]["system_prompt_sha256"] == _sha256(sent), model
        assert sent.startswith(agent.system_prompt_override) == (profile == "standard_vlm")
