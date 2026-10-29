"""The Gemini Live session core (providers/gemini/session.py), driven directly: no driver, no
WebSocket.

The history views come from the request builder, as the runtime builds them, and for native
streaming from the streaming history view (harnesses/streaming/history.py); the server messages
are google-genai's LiveServerMessage, built from the wire JSON the replay stub sends
(tests/replay/ws_gemini_live.py).
"""

from __future__ import annotations

import json
from dataclasses import replace

from google.genai import types

from realtime_framework.entity.actions import Observation
from realtime_framework.entity.context_policy import current_context_policy
from realtime_framework.entity.model import CanonicalTurn, ProviderObservation
from realtime_framework.entity.request import AnswerRequirements, CallParameters
from realtime_framework.harnesses.default.frames import FrameRecord
from realtime_framework.harnesses.default.request import (
    build_request,
    ledger_block,
    system_block,
    tool_exchange_block,
)
from realtime_framework.harnesses.streaming.history import (
    feedback_block,
    frame_block,
    history_view,
    output_block,
)
from realtime_framework.providers.gemini.session import (
    REPLAY_THOUGHT_SIGNATURE,
    Content,
    Delivery,
    Orphan,
    Part,
    Received,
    SessionCore,
    Trigger,
)

GO_AWAY_RECONNECT = {
    "reconnect_required": True,
    "reconnect_reason": "go_away",
    "go_away_time_left": "30s",
}
QUEUED = {"status": "queued"}


def session_core(
    completion_signal="turn_complete", tool_response_scheduling="when_idle", streaming=False
):
    return SessionCore(
        {
            "completion_signal": completion_signal,
            "tool_response_scheduling": tool_response_scheduling,
        },
        streaming=streaming,
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
    return CanonicalTurn(
        observation, output="{}", unattached_feedback=feedback, provider_refs=refs, decision=k
    )


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


def pushed(k: int):
    """Frame k as the streaming agent accepts it, under frame-table number k."""
    observation = turn(k).observation
    return frame_block(FrameRecord(k, observation.observation_id, float(k), {}, None), observation)


def streaming_view(blocks, *, frames: int = 2, history_format: str = "native_tools"):
    """The streaming history view of the accepted ``blocks``: every reply kept, the newest
    ``frames`` frames with their screenshots."""
    return history_view(
        blocks,
        policy=current_context_policy().for_path(
            "gemini_live", {"history_format": history_format}, None
        ),
        system=system_block("system instruction", None),
        retained_frames=frames,
        answer=AnswerRequirements("function_call", "standard_vlm"),
        call=CallParameters("gemini-live", None, None),
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


def start_reply(core: SessionCore, generation_id: str, blocks=None) -> None:
    """A streaming reply: the newest of the accepted ``blocks`` pushed and written, the
    generation, then its trigger."""
    history = streaming_view(blocks or [pushed(0)])
    core.pushed(core.push(history))
    core.start_generation(generation_id)
    core.trigger(history)


def clean_reply(core: SessionCore, generation_id: str, handle: str) -> None:
    """A streaming reply read as it arrives: a resumption update, then turnComplete."""
    start_reply(core, generation_id)
    for received in (update(handle), server_content(turnComplete=True)):
        arrive(core, received)


def renewal_handle(core: SessionCore) -> str | None:
    """The handle a goAway renewal after a clean generation resumes with, None to rebuild."""
    return core.resume_handle_for("go_away", force_rebuild=False, generation_clean=True)


def reply_ended(core: SessionCore, connection: int) -> None:
    """The reply read on ``connection`` has ended: it takes no later message read there, and is
    in flight no more once its events() are done too."""
    generation = core.replies[connection]
    assert generation.ended and core.received(connection, server_content(), 0.0) is None
    core.reply_done(generation)
    assert connection not in core.replies


def frame_content(k: int, text: str | None = None) -> Content:
    image = view([turn(k)]).items[1].block.image
    return Content("user", (Part(text=text or f"frame {k}"), Part(image=image)))


def test_new_session_then_rebuild():
    core = session_core()
    first = connected(core)
    assert (first.mode, first.release_grace, first.seed) == ("new_session", False, ())
    assert core.connection == 1
    assert core.delivery == {("setup",): Delivery.CONFIRMED}
    assert renewal_handle(core) is None  # nothing on the session yet: no handle to resume

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
    assert renewal_handle(core) is None and not core.renewal_required


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


def test_a_streaming_rebuild_replays_the_streaming_view():
    """A streaming session is rebuilt from the streaming view, in its order: the frames, the
    older ones folded to text, each reply's output as model text and each feedback as user
    text. A frame is its own text, as a push sends it."""
    action = '{"action": "click", "target": "P1", "frame": 1}'
    history = streaming_view(
        [
            pushed(1),
            output_block(1, action, "obs-1", None),
            pushed(2),
            feedback_block(1, None, "EXECUTION RESULT: ok", None),
            pushed(3),
        ],
    )
    connect = plan(session_core(streaming=True), history)
    assert connect.mode == "canonical_history_replay"
    (contents,) = connect.seed
    assert contents == [
        Content("user", (Part(text=history.policy.texts.folded_visual),)),
        Content("model", (Part(text=action),)),
        frame_content(2),
        Content("user", (Part(text="EXECUTION RESULT: ok"),)),
        frame_content(3),
    ]
    assert connect.seeded == ("f1.frame", "d1.output", "f2.frame", "d1.feedback", "f3.frame")


def test_a_streaming_rebuild_replays_a_replys_tool_rounds_as_a_turns_transcript():
    """Decision 24: a reply whose tool rounds the view holds replays them as a turn's are, the
    policy's transcript under its header, and not its output, the action as plain model text,
    which a turn-based seed avoids too."""
    action = '{"action": "click", "target": "P1", "frame": 1}'
    call = {"id": "c1", "name": "click", "arguments": {"target": "P1", "frame": 1}}
    receipt = {"id": "c1", "name": "click", "response": {"status": "queued"}}
    exchange = {"function_calls": [call], "function_responses": [receipt]}
    blocks = [pushed(1), output_block(1, action, "obs-1", None), pushed(2)]
    history = streaming_view(blocks, history_format="tool_transcript")
    tools = tool_exchange_block(
        history.policy,
        {"native_tool_exchanges": [exchange]},
        decision=1,
        based_on="obs-1",
        wall_ns=None,
    )
    history = streaming_view([*blocks[:2], tools, blocks[2]], history_format="tool_transcript")
    connect = plan(session_core(streaming=True), history)
    (contents,) = connect.seed
    transcript = history.policy.texts.gemini_tool_history_header + "\n" + json.dumps([exchange])
    assert contents == [
        frame_content(1),
        Content("model", (Part(text=transcript),)),
        frame_content(2),
    ]
    assert connect.seeded == ("f1.frame", "d1.tools", "f2.frame")


def test_a_streaming_rebuild_skips_an_empty_reply_output():
    """A reply that said nothing replays as nothing, as a turn-based seed skips empty text."""
    connect = plan(
        session_core(streaming=True),
        streaming_view([pushed(1), output_block(1, "", "obs-1", None), pushed(2)]),
    )
    (contents,) = connect.seed
    assert [content.role for content in contents] == ["user", "user"]
    assert connect.seeded == ("f1.frame", "f2.frame")


def test_a_streaming_frame_is_pushed_as_its_video_alone():
    """A pushed frame is a video frame, with no activity around it, and counts as sent once
    written; no push carries its text. Written, it is the newest frame the session got, which
    the next trigger carries, its text with its screenshot."""
    core = session_core(streaming=True)
    connected(core)
    push = core.push(streaming_view([pushed(1), pushed(2)]))
    assert push.image == frame_content(2).parts[1].image
    assert (push.observation_id, push.keys) == ("obs-2", (("block", "f2.frame"),))
    assert core.delivery[("block", "f2.frame")] == Delivery.UNKNOWN
    core.pushed(push)
    assert core.delivery[("block", "f2.frame")] == Delivery.SENT
    assert core.trigger(streaming_view([pushed(1), pushed(2)])).contents == (frame_content(2),)


def test_a_streaming_trigger_is_one_turn_of_the_feedback_since_the_last_and_the_newest_frame():
    """A trigger is one client content turn: the feedback no seed or trigger has carried into
    the session yet, each its own Content, then the newest frame written to the session, its
    text with its screenshot, as a seed replays a frame. Every trigger carries that frame, one
    with no feedback to carry too. A rebuild's seed carries the feedback it replays; a resume
    keeps what the session held."""
    core = session_core(streaming=True)
    connected(core)
    blocks = [pushed(1)]
    core.pushed(core.push(streaming_view(blocks)))
    blocks += [
        output_block(1, "click P1", "obs-1", None),
        feedback_block(1, None, "EXECUTION RESULT: ok", None),
        pushed(2),
    ]
    core.pushed(core.push(streaming_view(blocks)))
    connected(core, streaming_view(blocks))
    newest = ("frame_text", "f2.frame")
    assert core.trigger(streaming_view(blocks)) == Trigger((frame_content(2),), (newest,))

    for decision, feedback in ((2, "EXECUTION RESULT: missed"), (3, "SUPERSEDED")):
        blocks += [
            output_block(decision, "click P1", "obs-2", None),
            feedback_block(decision, None, feedback, None),
        ]
    trigger = core.trigger(streaming_view(blocks))
    keys = (("block", "d2.feedback"), ("block", "d3.feedback"), newest)
    contents = (
        feedback_content("EXECUTION RESULT: missed"),
        feedback_content("SUPERSEDED"),
        frame_content(2),
    )
    assert trigger == Trigger(contents, keys)
    assert all(core.delivery[key] == Delivery.UNKNOWN for key in keys)  # until written
    assert core.trigger(streaming_view(blocks)) == Trigger((frame_content(2),), (newest,))

    connected(core, resume_handle="handle-0")
    blocks += [
        output_block(4, "click P1", "obs-2", None),
        feedback_block(4, None, "EXECUTION RESULT: ok", None),
    ]
    trigger = core.trigger(streaming_view(blocks))
    assert trigger == Trigger(
        (feedback_content("EXECUTION RESULT: ok"), frame_content(2)),
        (("block", "d4.feedback"), newest),
    )


def test_a_streaming_trigger_carries_the_newest_frame_written_and_no_earlier_frame_s_text():
    """The frames pushed between two triggers went in as video alone: a trigger carries the
    newest one written, and not the text of those before it. A frame whose push failed was not
    written: the newest is the one written before it, and the failed one stays unknown. The view
    a trigger is planned from may not hold the newest frame written yet, as the agent enters a
    frame in its view once its push returned: the trigger carries that frame all the same. Each
    block is heard once written."""
    heard = []
    core = SessionCore(
        {"completion_signal": "turn_complete", "tool_response_scheduling": "when_idle"},
        streaming=True,
        on_delivery=lambda state, contents, connection: heard.append((state, contents)),
    )
    connected(core)
    blocks = []
    for k in (1, 2, 3):
        blocks.append(pushed(k))
        core.pushed(core.push(streaming_view(blocks)))
    core.send_failed(core.push(streaming_view([*blocks, pushed(4)])).keys)
    view = [
        *blocks[:2],
        output_block(1, "click P1", "obs-1", None),
        feedback_block(1, None, "EXECUTION RESULT: ok", None),
    ]
    trigger = core.trigger(streaming_view(view))
    keys = (("block", "d1.feedback"), ("frame_text", "f3.frame"))
    assert trigger == Trigger((feedback_content("EXECUTION RESULT: ok"), frame_content(3)), keys)
    core.sent(trigger.keys)
    assert heard[-1] == ("sent", [(keys[0], "d1.feedback"), (keys[1], "f3.frame")])
    assert core.delivery[("block", "f4.frame")] == Delivery.UNKNOWN


def test_a_trigger_after_a_rebuild_or_a_resume_carries_the_newest_frame_again():
    """A rebuild's seed replays the newest frame with its text and screenshot, and a resumed
    session holds what its handle held, which may lack the newest frame; either way the next
    trigger carries the newest frame, as every trigger does."""
    core = session_core(streaming=True)
    connected(core)
    blocks = [pushed(1)]
    core.pushed(core.push(streaming_view(blocks)))
    rebuilt = connected(core, streaming_view(blocks))
    assert rebuilt.seed == ([frame_content(1)],)
    newest = Trigger((frame_content(1),), (("frame_text", "f1.frame"),))
    assert core.trigger(streaming_view(blocks)) == newest
    connected(core, resume_handle="handle-1")
    assert core.trigger(streaming_view(blocks)) == newest


def fed(k: int, feedback: str) -> CanonicalTurn:
    """Decision k with the execution feedback on its action."""
    decided = turn(k)
    decided.execution_feedback = feedback
    return decided


def separate_view(turns):
    """view() under the policy with each action's feedback as its own block (item 3)."""
    policy = replace(
        current_context_policy().for_path("gemini_live", {"history_format": "native_tools"}, None),
        feedback="separate_message",
    )
    return build_request(
        policy,
        system_block("system instruction", None),
        tuple(turns),
        None,
        AnswerRequirements("function_call", "standard_vlm"),
        CallParameters("gemini-live", None, None),
    )


def feedback_content(text: str) -> Content:
    return Content("user", (Part(text=text),))


def test_seed_replays_each_feedback_after_its_turn_s_tool_rounds():
    connect = plan(session_core(), separate_view([fed(0, "F0"), turn(1)]))
    contents = [content for batch in connect.seed for content in batch]
    assert contents[3] == feedback_content("F0")
    # Frame, call, receipt and feedback of turn 0; frame, call and receipt of turn 1.
    roles = ["user", "model", "user", "user", "user", "model", "user"]
    assert [content.role for content in contents] == roles
    assert contents[4] == frame_content(1)
    assert "d0.feedback" in connect.seeded


def test_separate_feedback_is_the_same_content_in_native_streaming_as_turn_based():
    """Under separate_message an action's feedback is a user text Content of its own after the
    reply it reports and before the next frame, under the same block id, whether a turn-based
    seed (after the turn's tool rounds) or a native-streaming one (after the reply's text)
    replays it."""
    turn_based = plan(session_core(), separate_view([fed(1, "F1"), turn(2)]))
    blocks = [
        pushed(1),
        output_block(1, "click P1", "obs-1", None),
        feedback_block(1, None, "F1", None),
        pushed(2),
    ]
    streaming = plan(
        session_core(streaming=True),
        streaming_view(blocks, frames=5),
    )
    replayed = [content for batch in turn_based.seed for content in batch]
    (streamed,) = streaming.seed
    # Turn-based: frame 1, its call and the receipt, then the feedback; streaming: frame 1 and
    # the reply's text, then the feedback.
    assert [content.role for content in replayed[:3]] == ["user", "model", "user"]
    assert streamed[1] == Content("model", (Part(text="click P1"),))
    assert replayed[0] == streamed[0] == frame_content(1)
    assert replayed[3] == streamed[2] == feedback_content("F1")
    assert replayed[4] == streamed[3] == frame_content(2)
    assert "d1.feedback" in turn_based.seeded and "d1.feedback" in streaming.seeded


def test_only_a_streaming_seed_counts_the_feedback_it_carries():
    """What a seed carries is for the triggers after it, which only streaming sends."""
    assert plan(session_core(), separate_view([fed(0, "F0")])).carried == ()
    streaming = plan(
        session_core(streaming=True),
        streaming_view(
            [
                pushed(0),
                output_block(0, "click P1", "obs-0", None),
                feedback_block(0, None, "F0", None),
            ]
        ),
    )
    assert streaming.carried == (("d0.feedback", "F0"),)


def test_an_observation_carries_the_feedback_the_session_does_not_hold_yet():
    core = session_core()
    connected(core)
    frame = core.observe(separate_view([fed(0, "F0"), turn(1)]))
    assert frame.contents == (feedback_content("F0"), frame_content(1))
    assert frame.keys == (("block", "d0.feedback"), ("block", "d1.frame"))
    assert all(core.delivery[key] == Delivery.UNKNOWN for key in frame.keys)  # until written

    # A connection seeded with it already holds it.
    connected(core, separate_view([fed(0, "F0")]))
    assert core.observe(separate_view([fed(0, "F0"), turn(1)])).feedback == ()

    # A resumed session holds the earlier turns' feedback: only the last turn's is new.
    connected(core, resume_handle="handle-1")
    frame = core.observe(separate_view([fed(0, "F0"), fed(1, "F1"), turn(2)]))
    assert frame.feedback == (feedback_content("F1"),)


def test_each_change_in_delivery_is_heard_with_its_block():
    """on_delivery hears what the event log records: the setup confirmed and the seed sent once
    the connection is up; the setup confirmed and the seed unknown when its write failed before;
    each observation and acknowledgement sent once written, nothing while it is written, and
    unknown once a write failed. A turn-based session, as this one, records so as well as a
    streaming one: a write is not known to have gone out before it returned."""
    heard = []
    core = SessionCore(
        {"completion_signal": "turn_complete", "tool_response_scheduling": "when_idle"},
        on_delivery=lambda state, contents, connection: heard.append((state, contents)),
    )
    core.seed_failed(plan(core, view([turn(0)])), 0)
    connected(core, view([turn(0)]))
    frame = core.observe(view([turn(0), turn(1)]))
    core.sent(frame.keys)
    frame = core.observe(view([turn(0), turn(1), turn(2)]))
    core.send_failed(frame.keys)
    core.calls_received([("call-1", "click")], core.connection)
    core.acknowledgement_sent(core.acknowledging(core.connection), core.connection)
    seed = [(("block", "d0.frame"), "d0.frame"), (("block", "d0.tools"), "d0.tools")]
    assert heard == [
        ("confirmed", [(("setup",), "system")]),
        ("unknown", seed),
        ("confirmed", [(("setup",), "system")]),
        ("sent", seed),
        ("sent", [(("block", "d1.frame"), "d1.frame")]),
        ("unknown", [(("block", "d2.frame"), "d2.frame")]),
        ("sent", [(("tool_response", "call-1"), None)]),
    ]


def test_a_block_whose_seed_batch_failed_is_unknown_also_if_an_earlier_batch_carried_it():
    """A tool block's rounds can fall in two batches: the failed second batch leaves it unknown,
    and only what the first batch carried alone is sent."""
    heard = []
    core = SessionCore(
        {"completion_signal": "turn_complete", "tool_response_scheduling": "when_idle"},
        on_delivery=lambda state, contents, connection: heard.append((state, contents)),
    )
    connect = replace(
        plan(core, view([turn(0), turn(1)])),
        batched=(("d0.frame", "d0.tools"), ("d0.tools", "d1.frame"), ("d1.tools",)),
    )
    core.seed_failed(connect, 1)
    assert heard == [
        ("confirmed", [(("setup",), "system")]),
        ("sent", [(("block", "d0.frame"), "d0.frame")]),
        ("unknown", [(("block", "d0.tools"), "d0.tools"), (("block", "d1.frame"), "d1.frame")]),
    ]


def test_an_acknowledgement_written_as_its_connection_closes_is_still_heard():
    """What became of an acknowledgement's write is heard for the calls it answered, on the
    connection it was written on, also when an abort closed that connection while it was
    written; and it is no write of the connection opened since, whose pending calls stay."""
    heard = []
    core = SessionCore(
        {"completion_signal": "turn_complete", "tool_response_scheduling": "when_idle"},
        on_delivery=lambda state, contents, connection: heard.append((state, contents, connection)),
    )
    for call_id, write in (("call-0", core.acknowledgement_sent), ("call-1", None)):
        connected(core)
        core.calls_received([(call_id, "click")], core.connection)
        acknowledged, written_on = core.acknowledging(core.connection), core.connection
        connected(core)
        core.calls_received([("call-later", "key")], core.connection)
        (write or core.acknowledgement_failed)(acknowledged, written_on)
        assert ("tool_response", call_id) not in core.delivery
        assert core.pending_calls == [("call-later", "key")]
        core.pending_calls.clear()
    acknowledgements = [entry for entry in heard if entry[1][0][0][0] == "tool_response"]
    assert acknowledgements == [
        ("sent", [(("tool_response", "call-0"), None)], 1),
        ("unknown", [(("tool_response", "call-1"), None)], 3),
    ]


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
    # Unknown while it is written, sent once written; unknown when the write fails.
    assert all(core.delivery[key] == Delivery.UNKNOWN for key in frame.keys)
    core.sent(frame.keys)
    assert all(core.delivery[key] == Delivery.SENT for key in frame.keys)
    core.send_failed(frame.keys)
    assert all(core.delivery[key] == Delivery.UNKNOWN for key in frame.keys)

    core.start_generation("turn-0")
    core.calls_received([("call-0", "click")], core.connection)
    core.calls_received([("call-1", "key")], core.connection)
    # One queued acknowledgement answers every call still pending.
    acknowledged = core.acknowledging(core.connection)
    assert acknowledged == (("call-0", "click"), ("call-1", "key"))
    assert core.delivery[("tool_response", "call-1")] == Delivery.UNKNOWN
    # Still pending: the next acknowledgement answers them again.
    core.acknowledgement_failed(acknowledged, core.connection)
    assert core.delivery[("tool_response", "call-0")] == Delivery.UNKNOWN
    acknowledged = core.acknowledging(core.connection)
    assert acknowledged == (("call-0", "click"), ("call-1", "key"))
    core.acknowledgement_sent(acknowledged, core.connection)
    assert core.delivery[("tool_response", "call-0")] == Delivery.SENT
    assert core.acknowledging(core.connection) == ()


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
    assert core.resume_handle == "handle-0"
    reply_ended(core, 1)

    assert renewal_handle(core) == "handle-0"
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
    assert renewal_handle(core) == "handle-0" and not core.renewal_required


def test_a_later_go_away_that_gives_more_time_leaves_the_earliest_deadline():
    """Of a connection's goAways the earliest deadline holds, also when a later one gives more
    time: the driver does not count on a later goAway to put off a close the server already
    announced, and bounds the reply by the first."""
    core = session_core()
    connected(core)
    start_turn(core, "turn-0")
    receive(core, go_away("5s"), now=110.0)
    receive(core, go_away("30s"), now=111.0)
    assert core.go_away_deadline == 113.0


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
    assert renewal_handle(core) is None and not core.renewal_required


def test_interrupted_or_unresumable_turn_does_not_resume():
    core = session_core()
    connected(core)
    clean_turn(core, "turn-0", "handle-0")
    assert core.resume_handle == "handle-0"

    start_turn(core, "turn-1")
    assert core.resume_handle is None
    assert receive(core, update("handle-1")).generation == 2  # the second on this connection
    received = receive(core, server_content(interrupted=True))
    assert received.interrupted and received.contaminated and not received.ends
    received = receive(core, server_content(turnComplete=True))
    assert received.complete and received.contaminated and not received.interrupted
    assert core.resume_handle is None
    assert renewal_handle(core) is None

    # The next observation drops the interrupted turn's candidate: a clean turn without an
    # update of its own has no handle to promote.
    start_turn(core, "turn-2")
    receive(core, server_content(turnComplete=True))
    assert renewal_handle(core) is None

    start_turn(core, "turn-3")
    receive(core, update("handle-3"))
    receive(core, update(None))  # not resumable any more
    receive(core, server_content(turnComplete=True))
    assert renewal_handle(core) is None


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
    assert renewal_handle(core) is None

    received = receive(core, server_content(interactionStatus="IDLE"))
    assert received.status == "IDLE" and received.complete and received.ends
    assert renewal_handle(core) == "handle-0"
    reply_ended(core, 1)


def test_idle_ends_a_turn_complete_interaction_uncleanly():
    core = session_core()
    connected(core)
    start_turn(core, "turn-0")
    receive(core, update("handle-0"))
    received = receive(core, server_content(interactionStatus="IDLE"))
    assert received.ends and not received.complete
    assert renewal_handle(core) is None


def test_reply_read_after_close_keeps_its_generation():
    """An abort closes the connection while its reply is still being read. What its reader
    drains keeps that connection and generation, even once the next connection has started a
    generation of its own, and leaves the session alone."""
    core = session_core()
    connected(core)
    start_turn(core, "turn-0")
    core.calls_received([("call-0", "click")], core.connection)
    receive(core, go_away())
    core.closed()
    assert core.connection is None and core.delivery == {}
    assert core.acknowledging(core.connection) == () and core.go_away_deadline is None
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
    current = receive(core, server_content(turnComplete=True))
    assert (current.connection, current.generation, current.generation_id) == (2, 1, "turn-1")
    assert current.complete
    reply_ended(core, 1)
    reply_ended(core, 2)


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
            dict(core.replies),
        )

    before = state()
    for orphan in (update("handle-1"), go_away(), server_content(turnComplete=True)):
        assert receive(core, orphan) is None
    assert state() == before


def test_a_reply_message_arrives_as_received():
    """While a generation's reply is being read, arrived() applies a message as received() does."""
    core = session_core(streaming=True)
    connected(core)
    start_reply(core, "turn-0")
    received = arrive(core, update("handle-0"))
    assert isinstance(received, Received)
    assert (received.connection, received.generation, received.generation_id) == (1, 1, "turn-0")
    assert arrive(core, server_content(turnComplete=True)).ends
    assert core.resume_handle == "handle-0"
    reply_ended(core, 1)


def test_a_reply_s_first_message_is_its_generation_s_once_it_starts_before_the_trigger():
    """Read as it arrives, a call right after the trigger is the reply's when its generation
    started first, and an orphan that no generation takes when it did not."""
    call = message({"toolCall": {"functionCalls": [{"id": "call-0", "name": "click", "args": {}}]}})
    core = session_core(streaming=True)
    connected(core)
    start_reply(core, "turn-0")
    assert arrive(core, call).generation_id == "turn-0"

    late = session_core(streaming=True)
    connected(late)
    late.pushed(late.push(streaming_view([pushed(0)])))
    late.trigger(streaming_view([pushed(0)]))
    assert isinstance(arrive(late, call), Orphan)


def test_a_message_carries_the_time_it_was_read():
    """Each message carries the driver's time it was read, in a reply or between replies: when
    it came off the wire for a reader that reads every message as it arrives, and so when the
    calls it carries arrived."""
    core = session_core(streaming=True)
    connected(core)
    start_reply(core, "turn-0")
    call = message({"toolCall": {"functionCalls": [{"id": "call-0", "name": "click", "args": {}}]}})
    assert arrive(core, call, now=12.5).at == 12.5
    assert receive(core, server_content(turnComplete=True), now=13.0).at == 13.0
    assert arrive(core, update("handle-0"), now=14.0).at == 14.0


def test_an_orphan_go_away_asks_for_a_new_connection():
    """Between replies, a goAway read as it arrives asks for a new connection, by the server's
    deadline, as it does in a reply."""
    core = session_core(streaming=True)
    connected(core)
    clean_reply(core, "turn-0", "handle-0")
    orphan = arrive(core, go_away("30s"), now=100.0)
    assert isinstance(orphan, Orphan)
    assert (orphan.connection, orphan.control) == (1, GO_AWAY_RECONNECT)
    assert core.renewal_required and core.go_away_deadline == 128.0
    assert core.resume_handle_for("go_away", force_rebuild=False, generation_clean=True) == (
        "handle-0"
    )


def test_an_orphan_resumption_update_is_the_latest_handle():
    """Between replies, a resumable handle read as it arrives is the one a streaming renewal
    resumes with; an unresumable update leaves the last resumable one."""
    core = session_core(streaming=True)
    connected(core)
    clean_reply(core, "turn-0", "handle-0")
    core.push(streaming_view([pushed(0), pushed(1)]))
    assert isinstance(arrive(core, update("handle-1")), Orphan)
    arrive(core, update(None))
    assert core.resume_handle == "handle-1"
    assert core.resume_handle_for("go_away", force_rebuild=False, generation_clean=True) == (
        "handle-1"
    )


CANCELLATION = message({"toolCallCancellation": {"ids": ["call-1", "call-0"]}})


def test_a_cancellation_names_the_calls_and_the_reply_goes_on():
    """A toolCallCancellation says which calls not to execute, in the server's order, whether
    it comes in their reply, after it, in a reply still read from a closed connection or after
    that reply; it does not end the reply, and changes neither what was written nor which calls
    the next acknowledgement answers."""
    core = session_core(streaming=True)
    connected(core)
    start_reply(core, "turn-0")
    core.calls_received([("call-0", "click")], core.connection)
    core.acknowledgement_sent(core.acknowledging(core.connection), core.connection)
    core.calls_received([("call-1", "key")], core.connection)
    written = dict(core.delivery)
    received = receive(core, CANCELLATION)
    assert received.cancelled == ("call-1", "call-0")
    assert (received.generation_id, received.complete, received.ends) == ("turn-0", False, False)
    assert core.delivery == written
    assert core.acknowledging(core.connection) == (("call-1", "key"),)
    received = receive(core, server_content(turnComplete=True))
    assert received.ends and received.cancelled == ()

    orphan = arrive(core, CANCELLATION)
    assert (orphan.cancelled, orphan.control) == (("call-1", "call-0"), {})

    start_reply(core, "turn-1")
    connected(core, streaming_view([pushed(0)]))
    assert core.received(1, CANCELLATION, 0.0).cancelled == ("call-1", "call-0")
    core.received(1, server_content(turnComplete=True), 0.0)
    assert core.arrived(1, CANCELLATION, 0.0).cancelled == ("call-1", "call-0")


def test_streaming_resumes_with_the_latest_resumable_handle():
    """A streaming goAway renewal resumes with the latest handle the server called resumable,
    through the frames pushed after it, unresumable updates and an interrupted reply; a forced
    rebuild, or another reason, rebuilds."""
    core = session_core(streaming=True)
    connected(core)
    start_reply(core, "turn-0", [pushed(1)])
    arrive(core, update("handle-1"))
    core.push(streaming_view([pushed(1), pushed(2)]))  # the next frame, while the reply runs
    for received in (
        update(None),
        go_away(),
        server_content(interrupted=True),
        server_content(turnComplete=True),
    ):
        arrive(core, received)
    renewal = {"force_rebuild": False, "generation_clean": True}
    assert core.resume_handle_for("go_away", **renewal) == "handle-1"
    assert core.resume_handle_for("go_away", force_rebuild=False, generation_clean=False) == (
        "handle-1"
    )
    assert core.resume_handle_for("go_away", force_rebuild=True, generation_clean=True) is None
    assert core.resume_handle_for("history_block_compaction", **renewal) is None

    connected(core, resume_handle="handle-resumed")
    assert core.resume_handle_for("go_away", **renewal) == "handle-resumed"
    arrive(core, update("handle-2"))  # between replies
    assert core.resume_handle_for("go_away", **renewal) == "handle-2"
    connected(core, streaming_view([pushed(1)]))  # a rebuild has no handle until one comes
    assert core.resume_handle_for("go_away", **renewal) is None


def test_the_latest_handle_is_the_open_connection_s():
    """The core keeps a closed connection's latest handle, which a renewal resumes with
    (GeminiLiveDriver._sync_history). A reply still read from the closed connection once the
    next one is up does not change the next one's."""
    core = session_core(streaming=True)
    connected(core)
    clean_reply(core, "turn-0", "handle-0")
    start_reply(core, "turn-1")
    core.closed()
    renewal = {"force_rebuild": False, "generation_clean": True}
    assert core.resume_handle_for("go_away", **renewal) == "handle-0"
    connected(core, resume_handle="handle-0")
    assert core.received(1, update("handle-stale"), 0.0).generation_id == "turn-1"
    assert core.resume_handle_for("go_away", **renewal) == "handle-0"


def test_an_orphan_from_a_closed_connection_changes_nothing():
    core = session_core(streaming=True)
    connected(core)
    clean_reply(core, "turn-0", "handle-0")
    connected(core, streaming_view([pushed(0)]))
    for message in (update("handle-1"), go_away()):
        orphan = core.arrived(1, message, 0.0)
        assert (orphan.connection, orphan.control) == (1, {})
    assert (core.resume_handle, core.candidate_resume_handle) == (None, None)
    assert not core.renewal_required and core.go_away_deadline is None
