"""The age DefaultHarness gives a streaming action's frame counts every frame it sent before the
action arrived, also one sent at that same instant: the send and the arrival are read alike
(harnesses/default/frames.py FrameTable.record_sent and age). On the fake clock
(tests/fake_clock.py), at an instant whose nanosecond reading rounds to just after its seconds."""

from __future__ import annotations

from tests.fake_clock import FakeClock
from tests.frames.episode import decide
from tests.replay.build import build_episode
from tests.replay.frames import CLOCK_ORIGIN, ReplayTask
from tests.replay.scenario import TASK_INSTRUCTION, replies
from tests.replay.steps import Step
from tests.replay.stub import ChatCompletionsStub
from tests.replay.test_chat_completions import MODEL_KEY, chat_completions_config

# From the origin: Clock.time() reads 1760000313.7679653 here, Clock.time_ns() / 1e9 reads
# 1760000313.7679656.
ROUNDING_INSTANT_S = 313.76796543509624


def _streaming(agent) -> None:
    agent.interaction_mode = "streaming"


def test_the_frame_a_decision_sent_counts_in_the_age_of_an_older_frame_it_names(tmp_path):
    """Decision 1 names frame 1, and its own frame 2 went out at the instant its action
    arrived: frame 1 is one frame old."""
    clock = FakeClock(wall_origin=CLOCK_ORIGIN)
    steps = [
        Step('{"action": "wait", "frame": 1}', status="wait"),
        Step('{"action": "wait", "frame": 1}', status="wait"),
    ]
    with ChatCompletionsStub(replies(steps), clock=clock) as stub, clock.participant():
        clock.sleep(ROUNDING_INSTANT_S)
        assert clock.time_ns() / 1e9 > clock.time()
        episode = build_episode(
            chat_completions_config(stub.base_url, stream=False),
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=MODEL_KEY,
            before_initialize=_streaming,
            clock=clock,
        )
        try:
            results = [decide(episode, k, step.status)[1] for k, step in enumerate(steps)]
        finally:
            episode.finish()
    references = [result["action_protocol"]["frame_reference"] for result in results]
    assert [(r["frame"], r["age_frames"]) for r in references] == [(1, 0), (1, 1)]
