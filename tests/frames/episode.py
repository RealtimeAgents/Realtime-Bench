"""A chat_completions episode on the replay stub and synthetic frames, driven one decision at a
time in the order the shell calls the agent (tests/replay/steps.py)."""

from __future__ import annotations

from tests.replay import steps
from tests.replay.build import build_episode
from tests.replay.frames import ReplayTask
from tests.replay.scenario import TASK_INSTRUCTION
from tests.replay.test_chat_completions import MODEL_KEY, chat_completions_config


def open_episode(stub, run_dir, *, interaction_mode=None, scene=None):
    def before_initialize(agent):
        if interaction_mode is not None:
            agent.interaction_mode = interaction_mode

    return build_episode(
        chat_completions_config(stub.base_url, stream=False),
        run_dir,
        task=ReplayTask(),
        instruction=TASK_INSTRUCTION,
        model_key=MODEL_KEY,
        before_initialize=before_initialize,
        scene=scene,
    )


def decide(episode, k: int, status: str = "applied") -> tuple[object, dict]:
    """Capture frame k + 1, run decision k and report ``status``; the observation and result."""
    observation = episode.frames.capture(k)
    budget = episode.budget(observation.tick_number)
    result = steps.decide(episode.agent, k, observation, budget)
    steps.commit(episode.agent, k, status)
    return observation, result
