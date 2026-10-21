"""The agent's clock reaches each wait and time read its driver makes (realtime_framework/clock.py).

runner.set_up_agent gives create_agent, the agent's trace sink and the tool runtime the clock
main() hands it. create_agent gives the harness its clock, and the harness builds the driver and
the runtime with it (harnesses/default/harness.py _setup_backend). A driver passes it on to its
connect retry, its request pacing, its Choice decisions and the hooks that time its HTTP
responses. Each of these falls back to real time when it is given no clock, so a clock lost on
the way changes nothing a test on real time can see. Here every agent is built by create_agent on
the fake clock (tests/fake_clock.py). A case that reads a time reads one only the agent's clock
can have produced; the runtime and what set_up_agent builds read no time a test can see, so they
are checked for holding the clock itself.
"""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import httpx
import pytest

import realtime_framework.providers.litellm as litellm_driver
from realtime_framework.entity.config import ModelConfig
from realtime_framework.entity.model import (
    TRANSIENT,
    TRANSPORT_RETRY,
    ChoiceReply,
    ConnectRequest,
    GenerationRequest,
    RetryCancelled,
    SessionSeed,
    TransportFailure,
    retry_cancellation,
)
from realtime_framework.entity.records import TraceEvent
from realtime_framework.entity.request import AnswerRequirements
from realtime_framework.harnesses.default.execution_thread import ExecutionThread
from realtime_framework.harnesses.registry import create_agent
from realtime_framework.providers.registry import adapter_class

from .agent_trace.test_http import CANDIDATES, _request
from .fake_clock import FakeClock
from .replay.build import build_episode
from .replay.frames import CLOCK_ORIGIN, ReplayTask
from .replay.scenario import TASK_INSTRUCTION
from .replay.test_chat_completions import MODEL_KEY, chat_completions_config

S = 1_000_000_000  # nanoseconds
DECISIONS = ModelConfig(
    adapter_type="openai_decisions",
    model_name="openai/gpt-6-luna-decisions",
    api_key="sk-clock-wiring",
    upstream_provider="openai",
)


def _run(clock: FakeClock, coroutine):
    """Run ``coroutine`` on an event loop ``clock`` made, as the runtime runs a driver operation,
    while the test thread waits for it on the clock."""
    thread = ExecutionThread(clock, "clock-wiring")
    with clock.participant():
        thread.start()
        try:
            return thread.run(coroutine, timeout=600.0)
        finally:
            thread.stop()


class _Trace:
    """An agent trace that keeps the event and payload of each record."""

    def __init__(self):
        self.records: list[tuple[TraceEvent, dict]] = []

    def capture(self, event, payload, *, body=None, classify=None):
        self.records.append((event, payload))
        return len(self.records)


def test_set_up_agent_builds_the_agent_its_sink_and_tool_runtime_on_its_clock(tmp_path):
    """runner.set_up_agent, which main() calls with the episode's operational clock, hands it to
    the agent it creates, the agent's trace sink and the tool runtime (evaluation/runner.py)."""
    clock = FakeClock(wall_origin=CLOCK_ORIGIN)
    with clock.participant():
        episode = build_episode(
            chat_completions_config("http://127.0.0.1:9/v1", stream=False),
            tmp_path,
            task=ReplayTask(),
            instruction=TASK_INSTRUCTION,
            model_key=MODEL_KEY,
            clock=clock,
        )
        try:
            assert episode.agent.clock is clock
            assert episode.agent_trace.clock is clock
            assert episode.agent.tool_runtime.clock is clock
        finally:
            episode.finish()


def test_the_agent_builds_its_driver_and_runtime_on_its_clock():
    """The runtime only waits on its clock for the driver operations it runs on its own thread
    (harnesses/default/context.py), which a test cannot time, so its clock is checked as such."""
    clock = FakeClock()
    agent = create_agent(DECISIONS, clock=clock)
    assert agent.driver.clock is clock
    assert agent._runtime.clock is clock


def test_an_openai_realtime_connect_backs_off_on_the_agent_clock():
    """A refused connect is tried again after the shared backoff (providers/openai_realtime.py
    _connect)."""
    clock = FakeClock()
    config = ModelConfig(
        adapter_type="openai_realtime",
        model_name="gpt-realtime-replay",
        api_base_url="http://127.0.0.1:9/v1",  # never connected to
        api_key="sk-clock-wiring",
    )
    driver = create_agent(config, clock=clock).driver
    attempts = []

    @contextlib.contextmanager
    def connection():
        number = len(attempts)
        yield SimpleNamespace(
            _connection=SimpleNamespace(
                trace_connection_number=number, trace_connection_id=f"openai_realtime.ws-{number}"
            ),
            session=SimpleNamespace(update=lambda **arguments: None),
        )

    def connect(*, model, websocket_connection_options):
        attempts.append(clock.monotonic())
        if len(attempts) == 1:
            raise ConnectionError("connection refused")
        return connection()

    driver.client = SimpleNamespace(realtime=SimpleNamespace(connect=connect))
    answer = AnswerRequirements("text_json", "standard_vlm")
    seed = SessionSeed("system", config.model_name, None, 1.0, answer)

    async def connect_and_close():
        await driver.connect(ConnectRequest(seed=seed))
        await driver.close()

    _run(clock, connect_and_close())
    delay = TRANSPORT_RETRY.delay_s(1, "empty:connect", TransportFailure(TRANSIENT))
    assert attempts == [0.0, delay]


class _Built(Exception):
    """Raised in place of sending a request once it is built."""


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "complete"])
def test_a_litellm_request_is_paced_on_the_agent_clock(stream, monkeypatch):
    """With STREAM_MIN_REQUEST_INTERVAL set, a request is built only once that interval has
    passed since the process's last one (providers/litellm.py pace_request)."""
    clock = FakeClock(wall_origin=1000.0)
    monkeypatch.setenv("STREAM_MIN_REQUEST_INTERVAL", "30")
    # The process-wide time of the last request: 10 s ago on the agent's clock.
    monkeypatch.setattr(litellm_driver, "_LAST_REQUEST_AT", [990.0])
    config = ModelConfig(
        adapter_type="chat_completions",
        model_name="replay-vlm",
        api_base_url="http://127.0.0.1:9/v1",
        api_key="sk-clock-wiring",
        chat_stream=stream,
    )
    driver = create_agent(config, clock=clock).driver
    built = []

    def request_kwargs(request):
        built.append(clock.time())
        raise _Built

    monkeypatch.setattr(driver, "_request_kwargs", request_kwargs)
    with pytest.raises(_Built):
        _run(clock, driver.start_generation(None))
    assert built == [1020.0]
    assert litellm_driver._LAST_REQUEST_AT == [1020.0]


def _paced_driver(clock, stream, monkeypatch):
    """A chat_completions driver whose next request must wait 20 s for the pacing interval, and
    the times at which it built a request."""
    monkeypatch.setenv("STREAM_MIN_REQUEST_INTERVAL", "30")
    monkeypatch.setattr(litellm_driver, "_LAST_REQUEST_AT", [990.0])
    config = ModelConfig(
        adapter_type="chat_completions",
        model_name="replay-vlm",
        api_base_url="http://127.0.0.1:9/v1",
        api_key="sk-clock-wiring",
        chat_stream=stream,
    )
    driver = create_agent(config, clock=clock).driver
    built = []

    def request_kwargs(request):
        built.append(clock.time())
        raise _Built

    monkeypatch.setattr(driver, "_request_kwargs", request_kwargs)
    return driver, built


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "complete"])
def test_a_litellm_request_without_a_pacing_interval_reads_no_trace(stream, monkeypatch):
    """Without an interval (the default) pacing does nothing: it reads the driver's trace only
    for a wait, so a driver not bound to one yet still builds its request."""
    clock = FakeClock(wall_origin=1000.0)
    config = ModelConfig(
        adapter_type="chat_completions",
        model_name="replay-vlm",
        api_base_url="http://127.0.0.1:9/v1",
        api_key="sk-clock-wiring",
        chat_stream=stream,
    )
    driver = adapter_class("chat_completions")(config, clock=clock)
    built = []

    def request_kwargs(request):
        built.append(clock.time())
        raise _Built

    monkeypatch.setattr(driver, "_request_kwargs", request_kwargs)
    with pytest.raises(_Built):
        _run(clock, driver.start_generation(None))
    assert built == [1000.0]


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "complete"])
def test_a_litellm_request_pacing_wait_is_recorded_before_it_starts(stream, monkeypatch):
    """The pacing wait is written to the trace before it starts, with its length and the time it
    ends on the operational clock, as the session encoders record theirs (decision 1)."""
    clock = FakeClock(wall_origin=1000.0)
    driver, built = _paced_driver(clock, stream, monkeypatch)
    trace = _Trace()
    driver.bind_trace(trace)
    with pytest.raises(_Built):
        _run(clock, driver.start_generation(None))
    wait = {"wait": "request_pacing", "seconds": 20.0, "until_monotonic_ns": 20 * S}
    assert trace.records == [(TraceEvent.SESSION_WAIT, wait)]
    assert built == [1020.0]


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "complete"])
def test_an_abandoned_generation_ends_its_request_pacing_at_once(stream, monkeypatch):
    """The kernel gives the generation up 5 s into its 20 s pacing wait: the wait ends then, as
    RequestRetry's backoff does, and RetryCancelled unwinds the request unsent. The interval
    still counts from the request before, as none was sent."""
    clock = FakeClock(wall_origin=1000.0)
    driver, built = _paced_driver(clock, stream, monkeypatch)
    cancel = clock.event()
    clock.timer(5.0, cancel.set)
    with retry_cancellation(cancel), pytest.raises(RetryCancelled):
        _run(clock, driver.start_generation(None))
    assert clock.monotonic() == 5.0
    assert built == []
    assert litellm_driver._LAST_REQUEST_AT == [990.0]


def test_a_choice_question_retries_and_is_timed_on_the_agent_clock(monkeypatch):
    """A question whose first attempt fails is asked again after the shared backoff, and the
    decision reads each call's latency on the clock (providers/decisions/selection.py)."""
    clock = FakeClock()
    driver = create_agent(DECISIONS, clock=clock).driver
    asked = []

    async def decide(state, question):
        asked.append(clock.monotonic())
        if len(asked) == 1:
            raise httpx.ConnectError("connection refused")
        await clock.sleep_async(2.0)
        choice = next(iter(question["criteria"]))
        usage = {"input_tokens": 1, "output_tokens": 1}
        return ChoiceReply({"type": "choice", "choice": choice}, usage, {})

    monkeypatch.setattr(driver, "decide", decide)
    request = GenerationRequest("turn-1", None, None, "wall", canonical=_request(DECISIONS))
    metadata = _run(clock, driver.start_generation(request)).metadata
    delay = TRANSPORT_RETRY.delay_s(1, "obs-1:0", TransportFailure(TRANSIENT))
    assert asked == [0.0, delay]
    (call,) = metadata["jev_calls"]
    assert call["latency_s"] == pytest.approx(delay + 2.0)
    assert metadata["action"] == CANDIDATES["a"]["action"]


def test_a_decisions_response_is_timed_on_the_agent_clock():
    """The driver's HTTP hooks time each response as it is read (providers/capture.py
    _RecordingStream); this one arrives 3 s after the question."""
    clock = FakeClock()
    driver = create_agent(DECISIONS, clock=clock).driver
    trace = _Trace()
    driver.bind_trace(trace)

    async def serve(request):
        await clock.sleep_async(3.0)
        (criterion,) = json.loads(request.content)["questions"]["next_action"]["criteria"]
        answer = {
            "type": "choice",
            "choice": criterion,
            "probabilities": {criterion: 1.0},
            "confidence": 1.0,
        }
        body = {
            "provider": "openai",
            "model": DECISIONS.model_name,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"next_action": answer},
        }
        # A body given as content= is read before the response hooks wrap the stream.
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps(body).encode()),
        )

    driver.client = httpx.AsyncClient(
        base_url="http://127.0.0.1:9/api/alpha/",
        transport=httpx.MockTransport(serve),
        event_hooks=driver.client.event_hooks,
    )
    question = {"type": "choice", "instructions": "Choose.", "criteria": {"a": "click Submit"}}

    async def ask():
        try:
            return await driver.decide({}, question)
        finally:
            await driver.close()

    assert _run(clock, ask()).answer["choice"] == "a"
    (received,) = [p for e, p in trace.records if e == TraceEvent.MODEL_RESPONSE_RECEIVED]
    assert received["headers_monotonic_ns"] == 3 * S
    assert received["first_byte_monotonic_ns"] == received["last_byte_monotonic_ns"] == 3 * S
