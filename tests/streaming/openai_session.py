"""Drive the OpenAI Realtime driver through a few turns on one connection, on a given clock.

The calls are the ones RealtimeRuntime (harnesses/default/context.py) makes in start and in
run_turn for a turn of the legacy loop: connect, then per turn send_observation with the history
ending in the turn's frame, start_generation with the same history, and every event of the
generation; then close. The history is the request builder's
(harnesses/default/request.py:43), as tests/agent_trace/test_ws.py builds it.
"""

from __future__ import annotations

import io

from PIL import Image

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.events import TextDelta
from realtime_framework.entity.model import (
    CanonicalTurn,
    ConnectRequest,
    GenerationRequest,
    ProviderObservation,
    SessionSeed,
)
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.execution_thread import ExecutionThread
from realtime_framework.harnesses.default.request import build_request, system_block
from realtime_framework.providers import registry as catalog
from realtime_framework.providers.openai_realtime import OpenAIRealtimeDriver

from .clocked_ws import REAL_LIMIT_S, real_time_limit

MODEL_KEY = "gpt-realtime-2"  # configs/models/gpt-realtime-2.yaml
STUB_API_KEY = "sk-replay-stub"  # a placeholder, as tests/replay/test_openai_realtime.py uses
SYSTEM_INSTRUCTION = "system instruction"


def frame_png(k: int) -> bytes:
    """A 4x4 frame whose colour tells the turns apart; the cassette keeps its pixel digest."""
    out = io.BytesIO()
    Image.new("RGB", (4, 4), (40 * k, 80, 120)).save(out, format="PNG")
    return out.getvalue()


def _observation(k: int) -> ProviderObservation:
    return ProviderObservation(
        observation=Observation(b"", float(k), k, {"observation_id": f"obs-{k}"}),
        prompt=f"frame {k}",
        image_bytes=frame_png(k),
        mime_type="image/png",
        image_detail="auto",
        geometry={},
    )


def driver_for(port: int, clock, trace) -> OpenAIRealtimeDriver:
    config = catalog.to_model_config(
        MODEL_KEY, fps=1.0, reasoning_effort="low", api_key=STUB_API_KEY
    ).model_copy(update={"api_base_url": f"http://127.0.0.1:{port}/v1"})
    driver = OpenAIRealtimeDriver(config, clock=clock)
    driver.bind_trace(trace)
    return driver


async def run_turns(driver: OpenAIRealtimeDriver, turns: int) -> list[tuple[str, float]]:
    """Run ``turns`` turns; return each text delta with the clock's monotonic time it arrived."""
    config = driver.config
    policy = current_context_policy().for_path(
        config.adapter_type, config.provider_options, config.context_preset
    )
    answer = AnswerRequirements(driver.capabilities.action_channel, "standard_vlm", ())
    seed = SessionSeed(SYSTEM_INSTRUCTION, config.model_name, None, 1.0, answer)
    system = system_block(seed.system_instruction, None)
    call = CallParameters(config.model_name, config.reasoning_effort, None)
    deltas = []
    history: list[CanonicalTurn] = []
    await driver.connect(ConnectRequest(seed=seed))
    try:
        for k in range(turns):
            turn = CanonicalTurn(observation=_observation(k), decision=k)
            request = build_request(policy, system, (*history, turn), None, answer, call)
            await driver.send_observation(turn.observation, request)
            generation = await driver.start_generation(
                GenerationRequest(f"turn-{k}", None, "inf", "wall", canonical=request)
            )
            text = []
            async for event in driver.events(generation):
                if isinstance(event, TextDelta):
                    text.append(event.text)
                    deltas.append((event.text, driver.clock.monotonic()))
            turn.output = "".join(text)
            turn.execution_feedback = "wait executed"
            history.append(turn)
    finally:
        await driver.close()
    return deltas


def run_on_fake_clock(
    clock, coroutine, timeout: float = 60.0, *, real_limit_s: float = REAL_LIMIT_S
):
    """Run ``coroutine`` on an execution thread of the fake clock, as the harness runs driver
    operations (harnesses/default/execution_thread.py), while this thread takes part in the
    clock: at most ``timeout`` fake seconds, and ``real_limit_s`` real ones in which the fake
    clock does not move. A coroutine that did not finish is cancelled when the thread stops, and
    its cleanup, the driver's close, runs there."""
    execution = ExecutionThread(clock, "replay")
    execution.start()
    try:
        future = execution.submit(coroutine)
        with real_time_limit(real_limit_s, future.cancel, clock.monotonic), clock.participant():
            assert clock.wait(future, timeout), "the coroutine did not finish"
            return future.result()
    finally:
        execution.stop()
