"""Today's loop's calls on DefaultHarness (legacy_turn.py), where they differ from the kernel's."""

from __future__ import annotations

from tests.harness.test_generations import (
    WAIT_S,
    _chat_episode,
    _hold_after_the_reply,
    _in_worker,
    _joined,
)
from tests.replay.build import wait_for_abort
from tests.replay.scenario import LONG_SCENARIO, replies
from tests.replay.stub import ChatCompletionsStub

from .legacy_turn import LegacyTurn


def test_the_legacy_loop_drops_a_reply_its_abort_cancelled(tmp_path):
    """The legacy loop's bounded wait expires once the reply is back, before the runtime has
    read its cancel event: stream_generate hands the result back stale and writes nothing."""
    with ChatCompletionsStub(replies(LONG_SCENARIO[:1])) as stub:
        episode = _chat_episode(stub, tmp_path)
        agent = LegacyTurn(episode.agent)
        try:
            entered, release = _hold_after_the_reply(agent.driver)
            observation = episode.frames.capture(0)
            agent.configure_decision(episode.budget(observation.tick_number))
            agent.stream_prefill(observation)
            box, done = _in_worker(lambda: agent.stream_generate(tick=0))
            assert entered.wait(WAIT_S), "decision 0 got no reply"
            agent.abort()
            wait_for_abort(agent, WAIT_S)
            release.set()
            result = _joined(box, done)
        finally:
            episode.finish()
    assert len(stub.requests) == 1
    assert result["stale"] is True and result["action"] is None
    assert agent.history.turns == () and agent.history.actions == ()
