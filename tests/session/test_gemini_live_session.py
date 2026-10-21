"""The Gemini Live session core (providers/gemini/session.py), driven directly: no driver, no
WebSocket.

The history views come from the request builder, as the runtime builds them; the server messages
are google-genai's LiveServerMessage, built from the wire JSON the replay stub sends
(tests/replay/ws_gemini_live.py).
"""

from __future__ import annotations

from google.genai import types

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.model import CanonicalTurn, ProviderObservation
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.request import build_request, ledger_block, system_block
from realtime_framework.providers.gemini.session import (
    REPLAY_THOUGHT_SIGNATURE,
    Content,
    Delivery,
    Orphan,
    Part,
    Received,
    SessionCore,
)

GO_AWAY_RECONNECT = {
    "reconnect_required": True,
    "reconnect_reason": "go_away",
    "go_away_time_left": "30s",
}
QUEUED = {"status": "queued"}


def session_core(completion_signal="turn_complete", tool_response_scheduling="when_idle"):
    return SessionCore(
        {
            "completion_signal": completion_signal,
            "tool_response_scheduling": tool_response_scheduling,
        }
    )


def turn(k: int, *, calls=("click",), feedback: str = "", text_after: str = "") -> CanonicalTurn:
    """Decision k, answered with one tool round of ``calls`` and their queued receipts."""
    observation = ProviderObservation(
        observation=Observation(b"", float(k), k, {"observation_id": f"obs-{k}"}),
        prompt=f"frame {k}",
        image_bytes=b"png %d" % k,
        mime_type="image/png",
        image_detail="auto",
        geometry={},
        feedback_prefix=feedback,
    )
    exchange = {
        "function_calls": [
            {"id": f"call-{k}-{i}", "name": name, "arguments": {"target": "P1"}}
            for i, name in enumerate(calls)
        ],
        "function_responses": [
            {"id": f"call-{k}-{i}", "name": name, "response": QUEUED}
            for i, name in enumerate(calls)
        ],
        "completion_text": text_after,
    }
    refs = {"native_tool_exchanges": [exchange]} if calls else {}
    return CanonicalTurn(observation, output="{}", provider_refs=refs, decision=k)


def view(turns, *, ledger: str = "", history_format: str = "native_tools"):
    """The history view the runtime builds for gemini_live from these turns."""
    policy = current_context_policy().for_path(
        "gemini_live", {"history_format": history_format}, None
    )
    return build_request(
        policy,
        system_block("system instruction", None),
        tuple(turns),
        ledger_block(ledger, 1, 0),
        AnswerRequirements("function_call", "standard_vlm"),
        CallParameters("gemini-live", None, None),
    )


def message(wire: dict) -> types.LiveServerMessage:
    return types.LiveServerMessage.model_validate(wire)


def update(handle: str | None) -> types.LiveServerMessage:
    if handle is None:
        return message({"sessionResumptionUpdate": {"resumable": False}})
    return message({"sessionResumptionUpdate": {"newHandle": handle, "resumable": True}})


def go_away(time_left: str = "30s") -> types.LiveServerMessage:
    return message({"goAway": {"timeLeft": time_left}})


def server_content(**fields) -> types.LiveServerMessage:
    return message({"serverContent": fields})


def plan(core: SessionCore, history=None, resume_handle=None):
    """Close the open connection, as the driver does first, and plan the next one."""
    replacing = core.connection is not None
    core.closed()
    return core.connect(history, resume_handle, replacing=replacing)


def connected(core: SessionCore, history=None, resume_handle=None):
    connect = plan(core, history, resume_handle)
    core.connected(connect)
    return connect


def receive(core: SessionCore, message, now: float = 0.0):
    """A message on the open connection."""
    return core.received(core.connection, message, now)


def arrive(core: SessionCore, message, now: float = 0.0):
    """A message on the open connection, read as it arrives."""
    return core.arrived(core.connection, message, now)


def start_turn(core: SessionCore, generation_id: str) -> None:
    core.observe(view([turn(0)]))
    core.start_generation(generation_id)


def clean_turn(core: SessionCore, generation_id: str, handle: str, *messages) -> None:
    """One observation and its reply: a resumption update, ``messages``, then turnComplete."""
    start_turn(core, generation_id)
    for received in (update(handle), *messages, server_content(turnComplete=True)):
        receive(core, received)


def frame_content(k: int, text: str | None = None) -> Content:
    image = view([turn(k)]).items[1].block.image
    return Content("user", (Part(text=text or f"frame {k}"), Part(image=image)))


def test_new_session_then_rebuild():
    core = session_core()
    first = connected(core)
    assert (first.mode, first.release_grace, first.seed) == ("new_session", False, ())
    assert core.connection == 1
    assert core.delivery == {("setup",): Delivery.CONFIRMED}
    assert not core.last_turn_clean  # nothing on the session yet: no handle to resume

    rebuild = plan(core, view([turn(k) for k in range(10)]))
    assert core.connection is None  # the open connection is closed first
    assert (rebuild.mode, rebuild.release_grace) == ("canonical_history_replay", True)
    # Each turn is its frame, its calls and their receipts; twelve contents per message at most.
    assert [len(batch) for batch in rebuild.seed] == [12, 12, 6]
    assert rebuild.contents == 30
    seeded = tuple(f"d{k}.{kind}" for k in range(10) for kind in ("frame", "tools"))
    assert rebuild.seeded == seeded

    core.connected(rebuild)
    assert core.connection == 2
    assert core.delivery == {
        ("setup",): Delivery.CONFIRMED,
        **{("block", block_id): Delivery.SENT for block_id in seeded},
    }
    assert core.last_turn_clean and not core.renewal_required


def test_seed_batches_keep_each_exchange_whole():
    """A ledger ahead of the turns puts the exchanges off the 12-content grid; each message still
    ends with a turn's receipts, never between a call and its receipt."""
    connect = plan(session_core(), view([turn(k) for k in range(10)], ledger='{"decision": 0}'))
    assert [len(batch) for batch in connect.seed] == [10, 12, 9]
    for batch in connect.seed:
        assert batch[-1].role == "user" and batch[-1].parts[0].function_response is not None


def test_seed_replays_the_history_view():
    """The ledger first, then per turn its frame after its feedback, its native calls (the
    thought-signature sentinel on the first of a round only) and their stored receipts."""
    core = session_core(tool_response_scheduling="silent")
    turns = [
        turn(0),
        turn(1, calls=("click", "key"), feedback="EXECUTION RESULT: ok", text_after="done"),
    ]
    connect = plan(core, view(turns, ledger='{"decision": 0}'))
    (contents,) = connect.seed
    ledger, frame_0, calls_0, receipts_0, frame_1, calls_1, receipts_1, after_1 = contents
    texts = current_context_policy().for_path("gemini_live", {}, None).texts
    assert ledger == Content("user", (Part(text=texts.ledger_header + '\n{"decision": 0}'),))
    assert frame_0 == frame_content(0)
    assert frame_1 == frame_content(1, "EXECUTION RESULT: ok\nframe 1")
    assert calls_0 == Content(
        "model",
        (
            Part(
                function_call={"id": "call-0-0", "name": "click", "args": {"target": "P1"}},
                thought_signature=REPLAY_THOUGHT_SIGNATURE,
            ),
        ),
    )
    assert [part.thought_signature for part in calls_1.parts] == [REPLAY_THOUGHT_SIGNATURE, None]
    silent = {"will_continue": False, "scheduling": "SILENT"}
    assert receipts_0 == Content(
        "user",
        (
            Part(
                function_response={"id": "call-0-0", "name": "click", "response": QUEUED, **silent}
            ),
        ),
    )
    assert [part.function_response["id"] for part in receipts_1.parts] == ["call-1-0", "call-1-1"]
    assert after_1 == Content("model", (Part(text="done"),))
    assert connect.seeded == ("ledger", "d0.frame", "d0.tools", "d1.prefix", "d1.frame", "d1.tools")


def test_seed_replays_tool_rounds_as_a_transcript():
    connect = plan(session_core(), view([turn(0)], history_format="tool_transcript"))
    (contents,) = connect.seed
    frame, transcript = contents
    assert frame == frame_content(0)
    assert transcript.role == "model" and len(transcript.parts) == 1
    assert transcript.parts[0].text.endswith(
        '"function_calls": [{"id": "call-0-0", "name": "click", '
        '"arguments": {"target": "P1"}}], "function_responses": '
        '[{"id": "call-0-0", "name": "click", "response": '
        '{"status": "queued"}}], "completion_text": ""}]'
    )


def test_observation_and_acknowledgement_delivery():
    core = session_core()
    connected(core)
    frame = core.observe(view([turn(0), turn(1, feedback="EXECUTION RESULT: ok")]))
    assert (frame.role, frame.text, frame.observation_id) == (
        "user",
        "EXECUTION RESULT: ok\nframe 1",
        "obs-1",
    )
    assert frame.content == frame_content(1, "EXECUTION RESULT: ok\nframe 1")
    assert frame.keys == (("block", "d1.frame"), ("block", "d1.prefix"))
    # Sent once handed to the connection; unknown when the send fails.
    assert all(core.delivery[key] == Delivery.SENT for key in frame.keys)
    core.send_failed(frame.keys)
    assert all(core.delivery[key] == Delivery.UNKNOWN for key in frame.keys)

    core.start_generation("turn-0")
    core.calls_received([("call-0", "click")])
    core.calls_received([("call-1", "key")])
    # One queued acknowledgement answers every call still pending.
    assert core.acknowledging() == (("call-0", "click"), ("call-1", "key"))
    assert core.delivery[("tool_response", "call-1")] == Delivery.SENT
    core.acknowledgement_failed()  # still pending: the next acknowledgement answers them again
    assert core.delivery[("tool_response", "call-0")] == Delivery.UNKNOWN
    assert core.acknowledging() == (("call-0", "click"), ("call-1", "key"))
    core.acknowledgement_sent()
    assert core.delivery[("tool_response", "call-0")] == Delivery.SENT
    assert core.acknowledging() == ()


def test_go_away_resumes_after_a_clean_turn():
    core = session_core()
    connected(core)
    start_turn(core, "turn-0")

    received = receive(core, update("handle-0"), now=100.0)
    assert (received.connection, received.generation, received.generation_id) == (1, 1, "turn-0")
    assert not received.complete and received.control == {}
    # A handle captured mid-generation is promoted only at a clean boundary.
    assert core.resume_handle is None

    received = receive(core, go_away("30s"), now=100.0)
    assert received.control == GO_AWAY_RECONNECT
    assert core.renewal_required
    # The server closes the connection after timeLeft; the client keeps a 2-second margin.
    assert core.go_away_deadline == 128.0
    receive(core, go_away("5s"), now=110.0)
    assert core.go_away_deadline == 113.0  # the earliest deadline holds
    receive(core, go_away("soon"), now=111.0)
    assert core.go_away_deadline == 111.0  # a malformed timeLeft closes at once

    received = receive(core, server_content(turnComplete=True), now=112.0)
    assert received.complete and received.ends and not received.contaminated
    assert core.resume_handle == "handle-0" and core.last_turn_clean
    assert core.replies == {}

    assert core.resume_handle_for("go_away", force_rebuild=False, generation_clean=True) == (
        "handle-0"
    )
    assert core.resume_handle_for("go_away", force_rebuild=True, generation_clean=True) is None
    assert core.resume_handle_for("go_away", force_rebuild=False, generation_clean=False) is None
    assert (
        core.resume_handle_for(
            "history_block_compaction", force_rebuild=False, generation_clean=True
        )
        is None
    )

    resume = plan(core, resume_handle="handle-0")
    assert (resume.mode, resume.resume_handle, resume.seed, resume.release_grace) == (
        "session_resumption",
        "handle-0",
        (),
        True,
    )
    assert core.go_away_deadline is None  # it belonged to the closed connection
    core.connected(resume)
    assert core.connection == 2
    assert core.last_turn_clean and not core.renewal_required


def test_resume_failure_rebuilds():
    core = session_core()
    connected(core)
    clean_turn(core, "turn-0", "handle-0", go_away())

    plan(core, resume_handle="handle-0")
    # The resume was refused: connected() never comes. The renewal is still owed.
    assert core.connection is None and core.renewal_required
    assert core.resume_handle == "handle-0"

    rebuild = plan(core, view([turn(0)]))
    # No open connection is left to replace, but a replayed history still waits for the release.
    assert rebuild.release_grace and rebuild.mode == "canonical_history_replay"
    assert core.resume_handle is None
    core.connected(rebuild)
    assert core.connection == 2  # the refused resume never counted as a connection
    assert core.last_turn_clean and not core.renewal_required


def test_interrupted_or_unresumable_turn_does_not_resume():
    core = session_core()
    connected(core)
    clean_turn(core, "turn-0", "handle-0")
    assert core.resume_handle == "handle-0"

    start_turn(core, "turn-1")
    assert core.resume_handle is None and not core.last_turn_clean
    assert receive(core, update("handle-1")).generation == 2  # the second on this connection
    received = receive(core, server_content(interrupted=True))
    assert received.interrupted and received.contaminated and not received.ends
    received = receive(core, server_content(turnComplete=True))
    assert received.complete and received.contaminated and not received.interrupted
    assert core.resume_handle is None and not core.last_turn_clean
    assert core.resume_handle_for("go_away", force_rebuild=False, generation_clean=True) is None

    # The next observation drops the interrupted turn's candidate: a clean turn without an
    # update of its own has no handle to promote.
    start_turn(core, "turn-2")
    receive(core, server_content(turnComplete=True))
    assert core.resume_handle is None and core.last_turn_clean

    start_turn(core, "turn-3")
    receive(core, update("handle-3"))
    receive(core, update(None))  # not resumable any more
    receive(core, server_content(turnComplete=True))
    assert core.resume_handle is None and core.last_turn_clean


def test_extended_thinking_spans_intermediate_turn_complete():
    core = session_core("interaction_idle")
    connected(core)
    start_turn(core, "turn-0")
    receive(core, update("handle-0"))

    # Spoken fillers end with turnComplete, with or without an IN_PROGRESS status.
    for wire in (
        {"turnComplete": True},
        {"turnComplete": True, "interactionStatus": "IN_PROGRESS"},
    ):
        received = receive(core, server_content(**wire))
        assert not received.complete and not received.ends
        assert (received.generation, received.generation_id) == (1, "turn-0")
    assert received.status == "IN_PROGRESS"
    assert core.resume_handle is None and not core.last_turn_clean

    received = receive(core, server_content(interactionStatus="IDLE"))
    assert received.status == "IDLE" and received.complete and received.ends
    assert core.resume_handle == "handle-0" and core.last_turn_clean
    assert core.replies == {}


def test_idle_ends_a_turn_complete_interaction_uncleanly():
    core = session_core()
    connected(core)
    start_turn(core, "turn-0")
    receive(core, update("handle-0"))
    received = receive(core, server_content(interactionStatus="IDLE"))
    assert received.ends and not received.complete
    assert core.resume_handle is None and not core.last_turn_clean


def test_reply_read_after_close_keeps_its_generation():
    """An abort closes the connection while its reply is still being read. What its reader
    drains keeps that connection and generation, even once the next connection has started a
    generation of its own, and leaves the session alone."""
    core = session_core()
    connected(core)
    start_turn(core, "turn-0")
    core.calls_received([("call-0", "click")])
    receive(core, go_away())
    core.closed()
    assert core.connection is None and core.delivery == {}
    assert core.acknowledging() == () and core.go_away_deadline is None
    assert core.renewal_required  # the session state stays until the next connection

    connected(core, view([turn(0)]))
    assert core.connection == 2 and not core.renewal_required
    start_turn(core, "turn-1")
    receive(core, update("handle-2"))  # connection 2's turn is in progress

    for message in (update("handle-1"), go_away("5s")):
        stale = core.received(1, message, 0.0)
        assert (stale.connection, stale.generation, stale.generation_id) == (1, 1, "turn-0")
        assert stale.control == {}
    stale = core.received(1, server_content(turnComplete=True), 0.0)
    assert stale.complete and stale.ends
    # Connection 1's clean completion does not promote connection 2's candidate mid-turn.
    assert (core.resume_handle, core.candidate_resume_handle) == (None, "handle-2")
    assert not core.renewal_required and core.go_away_deadline is None
    assert not core.last_turn_clean
    current = receive(core, server_content(turnComplete=True))
    assert (current.connection, current.generation, current.generation_id) == (2, 1, "turn-1")
    assert current.complete and core.replies == {}


def test_an_orphan_message_changes_nothing():
    """After the reply has ended, and before the next generation starts, a message on the
    connection is an orphan: received() takes it as nothing, and no state changes."""
    core = session_core()
    connected(core)
    clean_turn(core, "turn-0", "handle-0")

    def state():
        return (
            core.resume_handle,
            core.candidate_resume_handle,
            core.renewal_required,
            core.go_away_deadline,
            core.last_turn_clean,
            dict(core.replies),
        )

    before = state()
    for orphan in (update("handle-1"), go_away(), server_content(turnComplete=True)):
        assert receive(core, orphan) is None
    assert state() == before


def test_a_reply_message_arrives_as_received():
    """While a generation's reply is being read, arrived() applies a message as received() does."""
    core = session_core()
    connected(core)
    start_turn(core, "turn-0")
    received = arrive(core, update("handle-0"))
    assert isinstance(received, Received)
    assert (received.connection, received.generation, received.generation_id) == (1, 1, "turn-0")
    assert arrive(core, server_content(turnComplete=True)).ends
    assert core.resume_handle == "handle-0" and core.replies == {}


def test_an_orphan_go_away_asks_for_a_new_connection():
    """Between replies, a goAway read as it arrives asks for a new connection, by the server's
    deadline, as it does in a reply."""
    core = session_core()
    connected(core)
    clean_turn(core, "turn-0", "handle-0")
    orphan = arrive(core, go_away("30s"), now=100.0)
    assert isinstance(orphan, Orphan)
    assert (orphan.connection, orphan.control) == (1, GO_AWAY_RECONNECT)
    assert core.renewal_required and core.go_away_deadline == 128.0
    # Nothing was sent since turn-0 completed cleanly: the renewal resumes with its handle.
    assert core.resume_handle_for("go_away", force_rebuild=False, generation_clean=True) == (
        "handle-0"
    )


def test_an_orphan_resumption_update_is_promoted_at_a_clean_boundary():
    """An orphan's handle is the session's while nothing has been sent since a clean completion
    or a seed. Once an observation is sent it is only the candidate."""
    core = session_core()
    connected(core)
    clean_turn(core, "turn-0", "handle-0")
    assert arrive(core, update("handle-1")).control == {}
    assert core.resume_handle == "handle-1"
    arrive(core, update(None))  # not resumable from here on: the boundary's handle stays
    assert (core.resume_handle, core.candidate_resume_handle) == ("handle-1", None)

    core.observe(view([turn(1)]))
    arrive(core, update("handle-2"))
    assert (core.resume_handle, core.candidate_resume_handle) == (None, "handle-2")

    seeded = session_core()
    connected(seeded, view([turn(0)]))
    arrive(seeded, update("handle-0"))
    assert seeded.resume_handle == "handle-0"
    fresh = session_core()
    connected(fresh)
    arrive(fresh, update("handle-0"))  # nothing on the session yet: no handle to resume
    assert (fresh.resume_handle, fresh.candidate_resume_handle) == (None, "handle-0")


def test_an_interrupt_ends_the_clean_boundary():
    """An interrupt is an input: a handle that arrives after it is not the clean boundary's."""
    core = session_core()
    connected(core)
    clean_turn(core, "turn-0", "handle-0")
    core.interrupt_sent()
    arrive(core, update("handle-1"))
    assert (core.resume_handle, core.candidate_resume_handle) == (None, "handle-1")
    assert not core.last_turn_clean


def test_an_orphan_from_a_closed_connection_changes_nothing():
    core = session_core()
    connected(core)
    clean_turn(core, "turn-0", "handle-0")
    connected(core, view([turn(0)]))
    for message in (update("handle-1"), go_away()):
        orphan = core.arrived(1, message, 0.0)
        assert (orphan.connection, orphan.control) == (1, {})
    assert (core.resume_handle, core.candidate_resume_handle) == (None, None)
    assert not core.renewal_required and core.go_away_deadline is None
