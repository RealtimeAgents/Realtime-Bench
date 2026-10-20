"""Turn-based decisions enter each frame in the frame table, record its send, and interpret the
action against the newest frame."""

from __future__ import annotations

from tests.replay.scenario import replies
from tests.replay.steps import Step
from tests.replay.stub import ChatCompletionsStub

from .episode import decide, open_episode

STEPS = [
    Step('{"action": "click", "target": "P1"}'),
    Step('{"action": "click", "target": "P3"}'),
    Step('{"action": "click", "x": 100, "y": 200}'),
]


def test_each_decision_sends_and_uses_the_newest_frame(tmp_path):
    with ChatCompletionsStub(replies(STEPS)) as stub:
        episode = open_episode(stub, tmp_path)
        try:
            seen = []
            for k in range(len(STEPS)):
                observation, result = decide(episode, k)
                frame = episode.agent.frames.latest()
                seen.append(frame)
                assert frame.frame == k + 1
                assert frame.observation_id == observation.metadata["observation_id"]
                assert frame.captured_unix_s == observation.timestamp
                assert frame.send_status == "sent"
                assert isinstance(frame.sent_unix_s, float)
                assert result["action"].coordinate_transform == frame.geometry
                resolutions = result["action_protocol"]["grounding"].get("resolutions", ())
                for resolution in resolutions:
                    assert resolution["resolved"] == frame.marks[resolution["mark"]]["point"]
        finally:
            episode.finish()
    # Older frames stay in the table with their own entries.
    assert [frame.frame for frame in seen] == [1, 2, 3]
    assert all(frame.send_status == "sent" for frame in seen)
