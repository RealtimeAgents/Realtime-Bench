"""A chat_completions episode on the replay stub and synthetic frames, driven one decision at a
time in the order run_stream calls the agent (tests/replay/steps.py)."""

from __future__ import annotations

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
    agent = episode.agent
    observation = episode.frames.capture(k)
    agent.configure_decision(episode.budget(observation.tick_number))
    agent.stream_prefill(observation)
    result = agent.stream_generate(tick=k)
    agent.finalize_decision_audit(result)
    agent.commit_action(status, "")
    return observation, result
