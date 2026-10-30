"""Gemini Live session core: what the session holds and what the driver does next, without I/O.

From a history view the core plans what the session is sent: the contents that seed a new
connection, the input of each observation and the trigger of each native-streaming reply, as
neutral Contents and Parts, and each frame native streaming pushes, as its video. GeminiLiveDriver
(live.py) turns them into google-genai types, writes and reads the WebSocket and reads the clock,
and reports back what it wrote and received. The core keeps the session state and answers with
the operations to perform: how to open the next connection and what to seed on it, whether a sync
resumes or rebuilds, which tool calls to acknowledge and which the server cancels, which
connection and generation a server message belongs to and whether that generation's output ends
with it. Nothing in it awaits, so each call is one transition, in the order of the session.

The open connection keeps what the client wrote on it, by key: ``("setup",)``, ``("block",
<history block id>)``, ``("frame_text", <frame block id>)`` (a streamed frame's text with its
screenshot, which a trigger carries) and ``("tool_response", <call id>)``, each unknown while it is
written, sent once written, confirmed when the server acknowledges it (only the setup is:
setupComplete) and unknown when its write failed. A seed is recorded once the connection
counts, as sent. When the write of a seed batch failed, the connection never counts: the setup is
heard as confirmed, the blocks of the batches written before as sent and those of the failed
batch as unknown. A resumed session also holds what the server restored from the handle, which
this record does not list. Each change but the start of a write is handed to ``on_delivery``,
which the driver writes to the event log.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from realtime_framework.entity.request import CanonicalRequest, FrameImage


class Delivery(str, Enum):
    """What is known about one write, as for OpenAI Realtime (openai_realtime_session.py)."""

    SENT = "sent"
    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"  # being written, or its write failed: the server may or may not have it


# Keep initial-history WebSocket messages small. HistoryConfig deliberately allows several
# client-content messages terminated by one turn_complete=True boundary.
HISTORY_CONTENTS_PER_MESSAGE = 12
_GO_AWAY_CLOSE_MARGIN_S = 2.0

# Gemini 3 requires a thought signature when a manually reconstructed history contains a model
# function-call part. Live tool-call messages do not expose the original signature because bidi
# function calls use the dedicated ToolCall message rather than model Content. Google documents
# this sentinel for deterministic client-injected function-call history.
REPLAY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"

# Content roles for the request intents this session carries (ledger, frame, feedback; tool
# exchange).
_ROLES = {"input": "user", "model": "model"}


@dataclass(frozen=True)
class Part:
    """One part of a planned Content: text, a screenshot, a function call (with the thought
    signature it carries) or a function response, as the google-genai Part of the same name."""

    text: str | None = None
    image: FrameImage | None = None
    function_call: dict | None = None  # id, name, args
    thought_signature: str | None = None
    function_response: dict | None = None  # id, name, response, and its scheduling options


@dataclass(frozen=True)
class Content:
    role: str
    parts: tuple[Part, ...]


@dataclass(frozen=True)
class Observation:
    """The newest frame of a history view as one input: first the execution feedback on the
    action before it that the session does not hold yet, each its own Content (``feedback``);
    then the frame, the feedback prefixed to it, then its prompt, with its screenshot; and the
    delivery keys of the blocks it carries."""

    role: str
    text: str
    image: FrameImage | None
    observation_id: str
    keys: tuple[tuple, ...]
    feedback: tuple[Content, ...] = ()

    @property
    def content(self) -> Content:
        return _frame_content(self.role, self.text, self.image)

    @property
    def contents(self) -> tuple[Content, ...]:
        return (*self.feedback, self.content)


@dataclass(frozen=True)
class Push:
    """The newest frame of a streaming history view as native streaming pushes it: its
    screenshot on the realtime_input video stream, outside any activity, so that it starts no
    reply. No push carries its text: sent outside an activity once the model has begun its
    reply, a text is input that interrupts the reply (Gemini 3.1 Flash Live), which a video frame
    is not. Once written, the frame is the newest the session got, which each trigger carries
    (Trigger)."""

    image: FrameImage
    observation_id: str | None
    keys: tuple[tuple, ...]  # the delivery key of its frame block
    # The frame block's id, and its text with its screenshot as one Content, as a trigger
    # carries it.
    frame: tuple[str, Content]


@dataclass(frozen=True)
class Trigger:
    """A reply's start as native streaming sends it: one client content turn that completes the
    client's turn, not an activity. It holds the execution feedback no seed or trigger has handed
    to the session yet, each its own Content, then the newest frame written to the session, its
    text with its screenshot, as a seed replays a frame. Every trigger carries that frame, also
    one an earlier trigger or a seed carried; the text of a frame pushed before it goes with
    none."""

    contents: tuple[Content, ...]
    keys: tuple[tuple, ...]  # the delivery keys of the feedback blocks and the frame it carries


def _frame_prefixes(history: CanonicalRequest) -> dict:
    """The execution-feedback block sent ahead of each frame's prompt, by frame block id."""
    return {
        item.block.attached_to: item.block
        for item in history.items
        if item.intent.position == "frame_prefix"
    }


def _frame_text(frame, prefix) -> str:
    return f"{prefix.text}\n{frame.text}" if prefix is not None else frame.text


def _frame_content(role: str, text: str, image: FrameImage | None) -> Content:
    parts = (Part(text=text),) if image is None else (Part(text=text), Part(image=image))
    return Content(role, parts)


def _is_feedback(item) -> bool:
    """An action's execution feedback on its own, not prefixed to a frame."""
    return item.block.kind == "execution_feedback" and item.intent.position != "frame_prefix"


def _feedback_content(item) -> Content:
    return Content(_ROLES[item.intent.role], (Part(text=item.block.text),))


def _block_of(key: tuple) -> str | None:
    """The history block a delivery key carries: the setup carries the system block, a frame's
    text its frame block."""
    if key == ("setup",):
        return "system"
    return key[1] if key[0] in ("block", "frame_text") else None


def _unrecorded(state: str, contents: list[tuple], connection: int | None) -> None:
    pass


def history_batches(history: list[Content], limit: int = HISTORY_CONTENTS_PER_MESSAGE) -> list:
    """Pack history without splitting one observation/model/tool-response exchange."""
    groups = []
    group = []
    for content in history:
        is_function_response = any(part.function_response is not None for part in content.parts)
        has_function_call = any(part.function_call is not None for part in content.parts)
        if content.role == "user" and group and not is_function_response:
            groups.append(group)
            group = []
        group.append(content)
        if is_function_response or (content.role == "model" and not has_function_call):
            groups.append(group)
            group = []
    if group:
        groups.append(group)

    batches = []
    batch = []
    for atomic_group in groups:
        if batch and len(batch) + len(atomic_group) > limit:
            batches.append(batch)
            batch = []
        batch.extend(atomic_group)
    if batch:
        batches.append(batch)
    return batches


def interaction_status(message) -> str | None:
    status = getattr(getattr(message, "server_content", None), "interaction_status", None)
    if status is None:
        return None
    return str(getattr(status, "value", status)).rsplit(".", 1)[-1].upper()


def _cancelled(message) -> tuple[str, ...]:
    """The ids of the tool calls a toolCallCancellation cancels, in the server's order."""
    return tuple(getattr(getattr(message, "tool_call_cancellation", None), "ids", None) or ())


@dataclass(frozen=True)
class Connect:
    """Open the next connection, once the current one is closed.

    Wait the session-release grace first when ``release_grace`` is set. The setup resumes
    ``resume_handle``, or declares HistoryConfig when there is a seed; each seed batch then goes out
    as client content, the last one completing the turn.
    """

    resume_handle: str | None
    seed: tuple[list[Content], ...]  # batches of the history view's contents
    seeded: tuple[str, ...]  # ids of the history-view blocks the seed replays
    release_grace: bool
    carried: tuple[tuple[str, str], ...]  # (id, text) of the execution feedback the seed replays
    # The ids of the blocks each seed batch carries, in order; a tool block can span batches.
    batched: tuple[tuple[str, ...], ...] = ()

    @property
    def contents(self) -> int:
        return sum(len(batch) for batch in self.seed)

    @property
    def mode(self) -> str:
        if self.resume_handle is not None:
            return "session_resumption"
        return "canonical_history_replay" if self.seed else "new_session"


@dataclass(frozen=True)
class Received:
    """One server message, as the core applied it.

    ``generation`` is the local generation in flight on the connection when the message was
    read, not proof that the message belongs to its reply. The receiver reads only while a
    generation asks for a message, so a message that arrived after one reply ended and before
    the next generation started is read, and numbered, as part of the next. Attributing messages
    by arrival is left to streaming, which reads as messages arrive: one read between a reply's
    end and the next generation's start then belongs to no generation, an Orphan.
    """

    connection: int  # the connection it was read from
    generation: int  # the generation in flight on that connection when it was read
    generation_id: str  # that generation's request id
    message: Any
    # When it was read, on the driver's monotonic clock: for a reader that reads every message
    # as it arrives, when it came off the wire, and so when an action it carries arrived.
    at: float
    status: str | None  # interaction_status
    interrupted: bool  # this message reports an interruption
    contaminated: bool  # the generation was interrupted, by this message or an earlier one
    complete: bool  # the interaction completes here, by the configured completion signal
    ends: bool  # the generation's output ends with this message
    control: dict  # what a goAway asks of the turn: reconnect after it
    # The ids of the tool calls this message cancels: not to be executed. It does not end the
    # reply. They may be an earlier reply's: tell them by call id, not by this message's
    # generation. A cancelled call still pending is acknowledged all the same; the driver
    # acknowledges a message's calls before it takes the reply's next message, so none is
    # pending then.
    cancelled: tuple[str, ...]


@dataclass(frozen=True)
class Orphan:
    """One server message that no generation's reply takes, as the core applied it (arrived).

    Its goAway still asks for a new connection and its resumption update is still the latest
    handle; read from a closed connection, it changes nothing. Any calls in it are not pending:
    no acknowledgement answers a reply the server started of its own accord.
    """

    connection: int  # the connection it was read from
    message: Any
    at: float  # when it was read, as in Received
    control: dict  # what a goAway asks of the session: reconnect after it
    cancelled: tuple[str, ...]  # the ids of the tool calls this message cancels, as in Received


@dataclass
class _Generation:
    connection: int
    number: int  # on its connection, from 1
    id: str
    interrupted: bool = False
    # Its reply's messages as they were read, until its events() takes them.
    messages: deque[Received] = field(default_factory=deque)
    # Streaming: the cancellations read between replies before it started, which its events()
    # hand over first.
    held: list[Orphan] = field(default_factory=list)
    ended: bool = False  # its reply's end was read
    done: bool = False  # its events() are done


class SessionCore:
    """The Live session as this client wrote it, across the connections that carry it.

    Connections are numbered from 1 as they are set up, generations from 1 on each connection.
    The numbers a message gets say which generation was in flight when it was read; they do not
    establish that it belongs to that reply (see Received).

    Closing a connection clears what belonged to it; the session state stays until the next
    connection replaces it. A reply still being read from a closed connection keeps its
    generation and still ends where it ends, but no longer changes the session: no resumption
    handle, goAway or clean boundary is taken from it.
    """

    def __init__(self, options: dict, *, streaming: bool = False, on_delivery=_unrecorded):
        """``options`` are the driver's resolved Live options (live.py live_options);
        ``streaming``: native streaming drives the session, pushing frames, triggering replies
        and reading every message as it arrives; ``on_delivery(state, contents, connection)``
        hears each change in the delivery record, with each key and the history block it carries
        ("system" for the setup), and the number of the connection written on (None for a seed
        that failed before its connection counted)."""
        self._completion_signal = options["completion_signal"]
        self.streaming = streaming
        self._on_delivery = on_delivery
        # What a queued acknowledgement adds to each function response, and so each replayed
        # receipt that did not store its own.
        self.tool_response_options = (
            {"will_continue": False, "scheduling": "SILENT"}
            if options["tool_response_scheduling"] == "silent"
            else {}
        )
        self._connections = 0  # connections set up so far
        self._generations = 0  # generations started on the open connection
        self.connection: int | None = None  # the open connection's number, from 1
        self.delivery: dict[tuple, Delivery] = {}  # what the client wrote on it
        # The handle a goAway renewal resumes with, since the connection was planned: the one it
        # resumes, until a later one comes. Turn-based, a generation's latest resumption update is
        # its candidate, promoted at a clean interaction boundary; streaming, which has few clean
        # boundaries, takes each handle the server calls resumable as it comes.
        self.candidate_resume_handle: str | None = None  # latest resumption update of this turn
        self.resume_handle: str | None = None
        # (id, name) of the tool calls received and not yet acknowledged.
        self.pending_calls: list[tuple] = []
        self.go_away_deadline: float | None = None  # on the driver's monotonic clock
        self.renewal_required = False  # a goAway asked for a new connection
        # Streaming: the execution feedback a seed or a trigger has handed to a connection of the
        # session, over the episode, the text handed by block id; a trigger's counts from when
        # it is planned, and one whose write failed leaves the session to be rebuilt, whose
        # seed replays it. A resume keeps it: what a trigger handed after the handle the session
        # resumed from is lost with the server's state, and no trigger hands it again.
        self.carried: dict[str, str] = {}
        # Streaming: the newest frame a push wrote, over the episode, its block id and its text
        # with its screenshot (Push.frame), which every trigger carries. A trigger planned while a
        # push writes waits for it (the driver holds the session for both), so it carries that
        # frame, which the history view it is planned from may not hold yet: the streaming agent
        # enters a frame in its view once its push returned (harnesses/streaming/agent.py _push).
        self.newest: tuple[str, Content] | None = None
        # The generation started last on each connection, until its reply ended and its events()
        # are done: the one whose reply takes what is read there until it ended.
        self.replies: dict[int, _Generation] = {}
        # Streaming: the cancellations read between replies, also past their connection's close,
        # which the next reply to start holds.
        self.cancelled_between: list[Orphan] = []

    # ---- connections ---------------------------------------------------------------------

    def connect(
        self, history: CanonicalRequest | None, resume_handle, *, replacing: bool
    ) -> Connect:
        """Plan the next connection, once the open one is closed (``replacing``: there was one),
        from the history view (None for an empty session). A resume carries no history: the
        server restores the session from the handle."""
        self.resume_handle = resume_handle
        contents, carriers = self._seed(history)
        seeded = tuple(dict.fromkeys(block_id for ids in carriers for block_id in ids))
        batches = tuple(history_batches(contents))
        batched, start = [], 0
        for batch in batches:
            ids = (block_id for ids in carriers[start : start + len(batch)] for block_id in ids)
            batched.append(tuple(dict.fromkeys(ids)))
            start += len(batch)
        # Only streaming triggers read what the seed carries.
        items = history.items if history is not None and self.streaming else ()
        return Connect(
            resume_handle,
            batches,
            seeded,
            release_grace=replacing or bool(contents),
            carried=tuple(
                (item.block.id, item.block.text)
                for item in items
                if item.block.kind == "execution_feedback" and item.block.id in seeded
            ),
            batched=tuple(batched),
        )

    def _seed(self, history: CanonicalRequest | None) -> tuple[list[Content], list[tuple]]:
        """The Contents that replay the history view, and for each the ids of the blocks it
        carries: a frame's with the feedback prefixed to it, a tool block's in each of its rounds.

        The system block travels in the setup instead. In a turn-based view a model output is the
        canonical action of the native call that its turn's tool block carries, so it has no
        Content of its own; an execution feedback of its own is a text Content after the turn's
        tool rounds. In the streaming view (harnesses/streaming/history.py) a reply's tool block
        likewise stands for its output: each other output that said anything and each execution
        feedback is a text Content, in the view's order, and a frame is its own text with its
        screenshot. That is not what the live session got: there every frame went in as video,
        each feedback came with the trigger after it, and only the frame each trigger carried came
        with its text and screenshot, which the seed gives every frame the view keeps the
        screenshot of.
        """
        if history is None:
            return [], []
        texts = history.policy.texts
        prefixes = _frame_prefixes(history)
        tooled = {
            item.block.id.removesuffix(".tools") + ".output"
            for item in history.items
            if item.block.kind == "tool_exchange"
        }
        contents, carriers, turn_index = [], [], -1
        for item in history.items:
            block = item.block
            if block.kind == "action_ledger":
                text = texts.ledger_header + "\n" + block.text
                contents.append(Content(_ROLES[item.intent.role], (Part(text=text),)))
                carriers.append((block.id,))
            elif block.kind == "frame":
                turn_index += 1
                prefix = prefixes.get(block.id)
                text = _frame_text(block, prefix)
                contents.append(_frame_content(_ROLES[item.intent.role], text, block.image))
                carriers.append((block.id,) if prefix is None else (prefix.id, block.id))
            elif block.kind == "tool_exchange":
                tool_contents = self._tool_contents(
                    block, _ROLES[item.intent.role], turn_index, texts
                )
                contents += tool_contents
                carriers += [(block.id,)] * len(tool_contents)
            elif self.streaming and block.kind in ("model_output", "execution_feedback"):
                if block.text and block.id not in tooled:
                    contents.append(Content(_ROLES[item.intent.role], (Part(text=block.text),)))
                    carriers.append((block.id,))
            elif _is_feedback(item):
                contents.append(_feedback_content(item))
                carriers.append((block.id,))
        return contents, carriers

    def _tool_contents(self, block, role: str, turn_index: int, texts) -> list[Content]:
        """One turn's tool rounds: the text spoken before the first call, then the policy's
        transcript, or each round as native calls and their stored receipts."""
        contents = []
        if block.initial_text:
            contents.append(Content(role, (Part(text=block.initial_text),)))
        if block.text:
            # Live tool messages do not expose their original thought signatures. Replaying
            # synthetic native calls can stall the next interaction even after setup succeeds.
            # The transcript keeps the complete past exchange as data, never as a new tool
            # request; current actions still use the declared native functions. Nor is it the
            # canonical action alone as ordinary model text, the form the native rebuild below
            # avoids: with it Gemini switched to spoken JSON after the first visual-history fold,
            # which also loses authoritative response-token usage.
            contents.append(Content(role, (Part(text=block.text),)))
            return contents
        for exchange_index, exchange in enumerate(block.exchanges):
            calls = [call for call in exchange.calls if call.get("name")]
            if not calls:
                continue
            responses_by_id = {
                str(response.get("id") or ""): response for response in exchange.receipts
            }
            model_parts, response_parts = [], []
            for call_index, call in enumerate(calls):
                call_id = str(
                    call.get("id") or f"history-{turn_index}-{exchange_index}-{call_index}"
                )
                name = str(call["name"])
                # Gemini 3 parallel calls carry the thought signature only on the first
                # function-call part in one model Content. Pydantic keeps Google's documented
                # literal sentinel on the wire even though genuine signatures are bytes.
                model_parts.append(
                    Part(
                        function_call={"id": call_id, "name": name, "args": call["arguments"]},
                        thought_signature=REPLAY_THOUGHT_SIGNATURE if call_index == 0 else None,
                    )
                )
                stored_response = responses_by_id.get(call_id, {})
                response_options = dict(self.tool_response_options)
                response_options.update(
                    {
                        key: stored_response[key]
                        for key in ("will_continue", "scheduling")
                        if key in stored_response
                    }
                )
                response_parts.append(
                    Part(
                        function_response={
                            "id": call_id,
                            "name": str(stored_response.get("name") or name),
                            "response": stored_response.get(
                                "response", dict(texts.gemini_queued_tool_response)
                            ),
                            **response_options,
                        }
                    )
                )
            # Rebuild every provider-native tool exchange, including a response that had
            # zero or multiple GUI calls and therefore became a benchmark WAIT. Replaying
            # only a selected call would change the model-visible trajectory after reconnect.
            contents.append(Content(role, tuple(model_parts)))
            contents.append(Content("user", tuple(response_parts)))
            if exchange.text_after:
                contents.append(Content(role, (Part(text=exchange.text_after),)))
        return contents

    def connected(self, connect: Connect) -> None:
        """The setup was acknowledged and the whole seed sent: the connection counts from here."""
        self._connections += 1
        self._generations = 0
        self.connection = self._connections
        self.delivery = {}
        self._set([("setup",)], Delivery.CONFIRMED)
        self._set((("block", block_id) for block_id in connect.seeded), Delivery.SENT)
        self.carried.update(connect.carried)
        self.renewal_required = False

    def closed(self) -> None:
        self.connection = None
        self.delivery = {}
        self.candidate_resume_handle = None
        self.pending_calls.clear()
        self.go_away_deadline = None

    def resume_handle_for(self, reason: str, *, force_rebuild: bool, generation_clean: bool):
        """The handle a sync resumes with, or None to rebuild from the history. Only a goAway
        renewal resumes: turn-based only after a clean interaction, with the handle promoted at
        it; streaming, which has few clean boundaries, with the latest handle the server called
        resumable, whatever was sent or read after it, so the session it restores may lack input
        or output the history has, or hold output the history lacks. A session that holds what
        the history does not know of (an abort closed its connection, a reply failed, a trigger
        did not go out whole) is closed or dirty for its lifecycle, which rebuilds it before a
        renewal reads a handle (SessionLifecycle.renew)."""
        if reason != "go_away" or force_rebuild:
            return None
        return self.resume_handle if self.streaming or generation_clean else None

    # ---- writes ----------------------------------------------------------------------------

    def sending(self, keys) -> None:
        """Record writes that start: unknown until written (sent) or failed (send_failed); the
        event log hears neither until then."""
        for key in keys:
            self.delivery[key] = Delivery.UNKNOWN

    def sent(self, keys) -> None:
        """The writes returned: the connection took them."""
        self._set(keys, Delivery.SENT)

    def send_failed(self, keys) -> None:
        self._set(keys, Delivery.UNKNOWN)

    def seed_failed(self, connect: Connect, batch: int) -> None:
        """The write of seed batch ``batch`` failed, after the server acknowledged the setup,
        before the connection counted: the setup is confirmed, what the batches before carried
        alone sent, what this one carried unknown. The batches after it were not written."""
        failed = connect.batched[batch]
        written = dict.fromkeys(
            block_id
            for ids in connect.batched[:batch]
            for block_id in ids
            if block_id not in failed
        )
        self._on_delivery(Delivery.CONFIRMED.value, [(("setup",), "system")], None)
        for state, ids in ((Delivery.SENT, written), (Delivery.UNKNOWN, failed)):
            if ids:
                self._on_delivery(
                    state.value, [(("block", block_id), block_id) for block_id in ids], None
                )

    def _set(self, keys, delivery: Delivery, connection: int | None = None) -> None:
        """Record writes on ``connection``, the open one by default; one on a connection closed
        since is heard, and is no write of the open one."""
        keys = list(keys)
        connection = self.connection if connection is None else connection
        if connection == self.connection:
            for key in keys:
                self.delivery[key] = delivery
        if keys:
            self._on_delivery(delivery.value, [(key, _block_of(key)) for key in keys], connection)

    def observe(self, history: CanonicalRequest) -> Observation:
        """Plan the history view's newest frame, after its feedback, as one input, and record it
        as being written: no handle resumes the session until its turn completes cleanly.

        Of the separate feedback, only that of the turn before the frame can be new: the
        turn-based loop commits each action, and with it its feedback, before it observes the
        next frame, so the session holds every earlier turn's, sent before the frames after it,
        seeded or resumed. It is new unless this connection was seeded with it."""
        frames = [index for index, item in enumerate(history.items) if item.block.kind == "frame"]
        current = history.items[frames[-1]]
        frame = current.block
        prefix = _frame_prefixes(history).get(frame.id)
        previous_turn = history.items[frames[-2] + 1 : frames[-1]] if len(frames) > 1 else ()
        feedback = [
            item
            for item in previous_turn
            if _is_feedback(item) and ("block", item.block.id) not in self.delivery
        ]
        observation = Observation(
            _ROLES[current.intent.role],
            _frame_text(frame, prefix),
            frame.image,
            frame.observation_id,
            tuple(("block", item.block.id) for item in feedback)
            + tuple(("block", block.id) for block in (frame, prefix) if block is not None),
            tuple(_feedback_content(item) for item in feedback),
        )
        self.resume_handle = None
        self.candidate_resume_handle = None
        self.sending(observation.keys)
        return observation

    def push(self, history: CanonicalRequest) -> Push:
        """Plan the streaming history view's newest frame, one the session does not hold yet, as
        native streaming pushes it, and record it as being written."""
        item = next(item for item in reversed(history.items) if item.block.kind == "frame")
        frame = item.block
        push = Push(
            frame.image,
            frame.observation_id,
            (("block", frame.id),),
            (frame.id, _frame_content(_ROLES[item.intent.role], frame.text, frame.image)),
        )
        self.sending(push.keys)
        return push

    def pushed(self, push: Push) -> None:
        """The push was written: its frame is sent, and the newest the session got."""
        self.sent(push.keys)
        self.newest = push.frame

    def trigger(self, history: CanonicalRequest) -> Trigger:
        """Plan the client content that starts a streaming session's next reply, with what the
        session lacks of the history view's execution feedback and the newest frame written, and
        record it as being written. Start the reply's generation (start_generation) before it
        goes out: read as it arrives, the reply's first message would otherwise be an orphan,
        and its calls lost."""
        lacking = [
            item
            for item in history.items
            if item.block.kind == "execution_feedback"
            and self.carried.get(item.block.id) != item.block.text
        ]
        frame_id, frame = self.newest
        trigger = Trigger(
            (*(_feedback_content(item) for item in lacking), frame),
            tuple(("block", item.block.id) for item in lacking) + (("frame_text", frame_id),),
        )
        self.carried.update((item.block.id, item.block.text) for item in lacking)
        self.sending(trigger.keys)
        return trigger

    def calls_received(self, calls: list[tuple], connection: int) -> None:
        """Calls read on ``connection``, pending there until acknowledged; read on a connection
        closed since, they are pending nowhere."""
        if connection == self.connection:
            self.pending_calls.extend(calls)

    def unacknowledged(self, connection: int) -> tuple[tuple, ...]:
        """The (id, name) of the calls a queued acknowledgement on ``connection`` answers: every
        one pending there, none on a connection closed since."""
        return tuple(self.pending_calls) if connection == self.connection else ()

    def acknowledging(self, connection: int) -> tuple[tuple, ...]:
        """The calls a queued acknowledgement on ``connection`` answers (unacknowledged), now
        being written."""
        pending = self.unacknowledged(connection)
        self.sending(("tool_response", call_id) for call_id, _ in pending)
        return pending

    def acknowledgement_sent(self, calls: tuple[tuple, ...], connection: int) -> None:
        """The acknowledgement of ``calls`` (acknowledging) was written on ``connection``, also
        if an abort closed it meanwhile."""
        keys = (("tool_response", call_id) for call_id, _ in calls)
        self._set(keys, Delivery.SENT, connection)
        if connection == self.connection:
            self.pending_calls.clear()

    def acknowledgement_failed(self, calls: tuple[tuple, ...], connection: int) -> None:
        """Its write on ``connection`` failed: still pending there, the next acknowledgement
        answers them again."""
        self._set(
            (("tool_response", call_id) for call_id, _ in calls), Delivery.UNKNOWN, connection
        )

    # ---- reads -----------------------------------------------------------------------------

    def start_generation(self, generation_id: str) -> _Generation:
        """Start the open connection's next generation, whose reply takes what is read there
        from now; streaming, it holds the cancellations read between replies."""
        self._generations += 1
        generation = _Generation(self.connection, self._generations, generation_id)
        if self.streaming:
            generation.held, self.cancelled_between = self.cancelled_between, []
        self.replies[self.connection] = generation
        return generation

    @property
    def replying(self) -> bool:
        """A reply started on the open connection has not ended, or its events() are not done:
        a connection carries one reply at a time."""
        generation = self.replies.get(self.connection)
        return generation is not None and not generation.done

    def reply_done(self, generation: _Generation) -> None:
        """Its events() are done: once its reply ended too, it is in flight no more."""
        generation.done = True
        if generation.ended and self.replies.get(generation.connection) is generation:
            del self.replies[generation.connection]

    def reader_failed(self, connection: int) -> None:
        """The connection's reader failed: no reply is read on it any more."""
        self.replies.pop(connection, None)

    def reply_abandoned(self) -> None:
        """The reply started on the open connection will not come (its trigger did not go out):
        its generation takes none of the connection's messages, and the cancellations it held
        are the next reply's."""
        generation = self.replies.pop(self.connection, None)
        if generation is not None:
            self.cancelled_between[:0] = generation.held

    def received(self, connection: int, message, now: float) -> Received | None:
        """Apply one server message that arrived on ``connection`` at ``now`` (the driver's
        monotonic clock), and hand it to the reply that takes it. With no generation's reply
        being read on that connection the message is an orphan: nothing takes it, and it changes
        nothing (None; see arrived)."""
        generation = self.replies.get(connection)
        if generation is None or generation.ended:
            return None
        server = getattr(message, "server_content", None)
        status = interaction_status(message)
        if self._completion_signal == "interaction_idle":
            complete = status == "IDLE"
        else:
            complete = bool(server is not None and getattr(server, "turn_complete", False))
        interrupted = bool(server is not None and getattr(server, "interrupted", False))
        generation.interrupted = generation.interrupted or interrupted
        if connection == self.connection:
            control = self._control(message, clean=complete and not generation.interrupted, now=now)
        else:
            # Read from a closed connection: the session is no longer there to change.
            control = {}
        # IDLE ends the interaction whichever completion signal is configured; under turn_complete
        # it ends it without completing it.
        ends = complete or status == "IDLE"
        received = Received(
            connection,
            generation.number,
            generation.id,
            message,
            now,
            status,
            interrupted,
            generation.interrupted,
            complete,
            ends,
            control,
            _cancelled(message),
        )
        generation.messages.append(received)
        if ends:
            generation.ended = True
            if generation.done:
                del self.replies[connection]
        return received

    def arrived(self, connection: int, message, now: float) -> Received | Orphan:
        """Apply one server message as a reader that reads every message as it arrives does
        (streaming): as received() does while a generation's reply is being read on
        ``connection``, else as an orphan, whose goAway still asks for a new connection and whose
        resumption update is still the latest handle; one that cancels calls is held for the
        next reply to start (cancelled_between)."""
        received = self.received(connection, message, now)
        if received is not None:
            return received
        control = (
            self._control(message, clean=False, now=now) if connection == self.connection else {}
        )
        orphan = Orphan(connection, message, now, control, _cancelled(message))
        if orphan.cancelled:
            self.cancelled_between.append(orphan)
        return orphan

    def _control(self, message, *, clean: bool, now: float) -> dict:
        update = getattr(message, "session_resumption_update", None)
        if update is not None:
            if not getattr(update, "resumable", False):
                self.candidate_resume_handle = None
            elif getattr(update, "new_handle", None):
                self.candidate_resume_handle = update.new_handle
                if self.streaming:
                    self.resume_handle = update.new_handle
        # Resumption updates often arrive before the final server-content message. Promote the
        # latest candidate only at a clean turn boundary; a handle captured mid-generation may not
        # represent the complete provider turn.
        if clean and self.candidate_resume_handle:
            self.resume_handle = self.candidate_resume_handle
        go_away = getattr(message, "go_away", None)
        if go_away is None:
            return {}
        self.renewal_required = True
        raw_time_left = getattr(go_away, "time_left", None)
        try:
            seconds = float(str(raw_time_left).removesuffix("s"))
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("invalid GoAway duration")
        except (TypeError, ValueError):
            # Missing or malformed provider deadlines cannot justify waiting indefinitely.
            seconds = 0.0
        deadline = now + max(0.0, seconds - _GO_AWAY_CLOSE_MARGIN_S)
        self.go_away_deadline = (
            deadline if self.go_away_deadline is None else min(deadline, self.go_away_deadline)
        )
        return {
            "reconnect_required": True,
            "reconnect_reason": "go_away",
            "go_away_time_left": raw_time_left,
        }
