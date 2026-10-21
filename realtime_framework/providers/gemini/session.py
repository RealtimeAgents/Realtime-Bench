"""Gemini Live session core: what the session holds and what the driver does next, without I/O.

From a history view the core plans what the session is sent: the contents that seed a new
connection and the input of each observation, as neutral Contents and Parts. GeminiLiveDriver
(live.py) turns them into google-genai types, writes and reads the WebSocket and reads the clock,
and reports back what it wrote and received. The core keeps the session state and answers with
the operations to perform: how to open the next connection and what to seed on it, whether a sync
resumes or rebuilds, which tool calls to acknowledge and which the server cancels, which
connection and generation a server message belongs to and whether that generation's output ends
with it. Nothing in it awaits, so each call is one transition, in the order of the session.

The open connection keeps what the client wrote on it, by key: ``("setup",)``, ``("block",
<history block id>)`` and ``("tool_response", <call id>)``, each sent once handed to the
connection, confirmed when the server acknowledges it (only the setup is: setupComplete) and
unknown when its send failed. A resumed session also holds what the server restored from the
handle, which this record does not list. Nothing reads it yet; the phase-2 split of the EXACT
session state puts it in the event log.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from realtime_framework.entity.request import CanonicalRequest, FrameImage


class Delivery(str, Enum):
    """What is known about one write, as for OpenAI Realtime (openai_realtime_session.py)."""

    SENT = "sent"
    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"  # the send failed, so the server may or may not have it


# Keep initial-history WebSocket messages small. HistoryConfig deliberately allows several
# client-content messages terminated by one turn_complete=True boundary.
HISTORY_CONTENTS_PER_MESSAGE = 12
_GO_AWAY_CLOSE_MARGIN_S = 2.0

# Gemini 3 requires a thought signature when a manually reconstructed history contains a model
# function-call part. Live tool-call messages do not expose the original signature because bidi
# function calls use the dedicated ToolCall message rather than model Content. Google documents
# this sentinel for deterministic client-injected function-call history.
REPLAY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"

# Content roles for the request intents this session carries (ledger, frame; tool exchange).
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
    """The newest frame of a history view as one input: the frame's execution feedback, then its
    prompt, with its screenshot; and the delivery keys of the blocks it carries."""

    role: str
    text: str
    image: FrameImage | None
    observation_id: str
    keys: tuple[tuple, ...]

    @property
    def content(self) -> Content:
        return _frame_content(self.role, self.text, self.image)


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
    status: str | None  # interaction_status
    interrupted: bool  # this message reports an interruption
    contaminated: bool  # the generation was interrupted, by this message or an earlier one
    complete: bool  # the interaction completes here, by the configured completion signal
    ends: bool  # the generation's output ends with this message
    control: dict  # what a goAway asks of the turn: reconnect after it
    # The ids of the tool calls this message cancels: not to be executed. It does not end the
    # reply.
    cancelled: tuple[str, ...]


@dataclass(frozen=True)
class Orphan:
    """One server message that no generation's reply takes, as the core applied it (arrived).

    Its goAway still asks for a new connection and its resumption update is still the latest
    handle; read from a closed connection, it changes nothing.
    """

    connection: int  # the connection it was read from
    message: Any
    control: dict  # what a goAway asks of the session: reconnect after it
    cancelled: tuple[str, ...]  # the ids of the tool calls this message cancels, as in Received


@dataclass
class _Generation:
    number: int  # on its connection, from 1
    id: str
    interrupted: bool = False


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

    def __init__(self, options: dict, *, streaming: bool = False):
        """``options`` are the driver's resolved Live options (live.py live_options);
        ``streaming``: native streaming drives the session, pushing frames, triggering replies
        and reading every message as it arrives."""
        self._completion_signal = options["completion_signal"]
        self.streaming = streaming
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
        self.candidate_resume_handle: str | None = None  # latest resumption update of this turn
        # (id, name) of the tool calls received and not yet acknowledged.
        self.pending_calls: list[tuple] = []
        self.go_away_deadline: float | None = None  # on the driver's monotonic clock
        self.resume_handle: str | None = None  # promoted at a clean interaction boundary
        # Since the connection was planned: the latest handle the server called resumable, else
        # the one the connection resumes. Streaming resumes with it.
        self.latest_resume_handle: str | None = None
        self.renewal_required = False  # a goAway asked for a new connection
        self.last_turn_clean = False
        # The generation whose reply is being read, by connection, until the reply ends.
        self.replies: dict[int, _Generation] = {}

    # ---- connections ---------------------------------------------------------------------

    def connect(
        self, history: CanonicalRequest | None, resume_handle, *, replacing: bool
    ) -> Connect:
        """Plan the next connection, once the open one is closed (``replacing``: there was one),
        from the history view (None for an empty session). A resume carries no history: the
        server restores the session from the handle."""
        self.resume_handle = resume_handle
        self.latest_resume_handle = resume_handle
        contents, seeded = self._seed(history)
        return Connect(
            resume_handle,
            tuple(history_batches(contents)),
            seeded,
            release_grace=replacing or bool(contents),
        )

    def _seed(self, history: CanonicalRequest | None) -> tuple[list[Content], tuple[str, ...]]:
        """The Contents that replay the history view, and the ids of their blocks.

        The system block travels in the setup instead. In a turn-based view a model output is the
        canonical action of the native call that its turn's tool block carries, so it has no
        Content of its own. The streaming view (harnesses/streaming/history.py) has no tool blocks:
        each reply's output and each execution feedback is a text Content, in the view's order.
        """
        if history is None:
            return [], ()
        texts = history.policy.texts
        prefixes = _frame_prefixes(history)
        contents, seeded, turn_index = [], [], -1
        for item in history.items:
            block = item.block
            if block.kind == "action_ledger":
                text = texts.ledger_header + "\n" + block.text
                contents.append(Content(_ROLES[item.intent.role], (Part(text=text),)))
            elif block.kind == "frame":
                turn_index += 1
                prefix = prefixes.get(block.id)
                text = _frame_text(block, prefix)
                contents.append(_frame_content(_ROLES[item.intent.role], text, block.image))
                if prefix is not None:
                    seeded.append(prefix.id)
            elif block.kind == "tool_exchange":
                tool_contents = self._tool_contents(
                    block, _ROLES[item.intent.role], turn_index, texts
                )
                if not tool_contents:
                    continue
                contents += tool_contents
            elif self.streaming and block.kind in ("model_output", "execution_feedback"):
                contents.append(Content(_ROLES[item.intent.role], (Part(text=block.text),)))
            else:
                continue
            seeded.append(block.id)
        return contents, tuple(seeded)

    def _tool_contents(self, block, role: str, turn_index: int, texts) -> list[Content]:
        """One turn's tool rounds: the text spoken before the first call, then the policy's
        transcript, or each round as native calls and their stored receipts."""
        contents = []
        if block.initial_text:
            contents.append(Content(role, (Part(text=block.initial_text),)))
        if block.text:
            # Live tool messages do not expose their original thought signatures. Replaying
            # synthetic native calls can stall the next interaction even after setup succeeds.
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
        self.delivery = {("setup",): Delivery.CONFIRMED}
        self.sending(("block", block_id) for block_id in connect.seeded)
        self.renewal_required = False
        self.last_turn_clean = connect.resume_handle is not None or bool(connect.seeded)

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
        or output the history has."""
        if self.streaming:
            return self.latest_resume_handle if reason == "go_away" and not force_rebuild else None
        if reason == "go_away" and not force_rebuild and self.last_turn_clean and generation_clean:
            return self.resume_handle
        return None

    # ---- writes ----------------------------------------------------------------------------

    def sending(self, keys) -> None:
        """Record writes as handed to the connection, before any answer can arrive."""
        for key in keys:
            self.delivery[key] = Delivery.SENT

    def send_failed(self, keys) -> None:
        for key in keys:
            self.delivery[key] = Delivery.UNKNOWN

    def observe(self, history: CanonicalRequest) -> Observation:
        """Plan the history view's newest frame, after its feedback, as one input, and record it
        as being written: no handle resumes the session until its turn completes cleanly."""
        current = next(item for item in reversed(history.items) if item.block.kind == "frame")
        frame = current.block
        prefix = _frame_prefixes(history).get(frame.id)
        observation = Observation(
            _ROLES[current.intent.role],
            _frame_text(frame, prefix),
            frame.image,
            frame.observation_id,
            tuple(("block", block.id) for block in (frame, prefix) if block is not None),
        )
        self.last_turn_clean = False
        self.resume_handle = None
        self.candidate_resume_handle = None
        self.sending(observation.keys)
        return observation

    def interrupt_sent(self) -> None:
        self.resume_handle = None

    def calls_received(self, calls: list[tuple]) -> None:
        self.pending_calls.extend(calls)

    def acknowledging(self) -> tuple[tuple, ...]:
        """The (id, name) of the calls a queued acknowledgement answers: every pending one."""
        pending = tuple(self.pending_calls)
        self.sending(("tool_response", call_id) for call_id, _ in pending)
        return pending

    def acknowledgement_sent(self) -> None:
        self.pending_calls.clear()

    def acknowledgement_failed(self) -> None:
        self.send_failed(("tool_response", call_id) for call_id, _ in self.pending_calls)

    # ---- reads -----------------------------------------------------------------------------

    def start_generation(self, generation_id: str) -> None:
        self._generations += 1
        self.replies[self.connection] = _Generation(self._generations, generation_id)

    def generation_failed(self, connection: int) -> None:
        """The reply on ``connection`` failed after the core applied its messages: if that is the
        open connection, its session has no clean boundary to resume from."""
        if connection == self.connection:
            self.last_turn_clean = False
            self.resume_handle = None

    def reader_failed(self, connection: int) -> None:
        """The connection's reader failed: no reply is read on it any more."""
        self.replies.pop(connection, None)

    def received(self, connection: int, message, now: float) -> Received | None:
        """Apply one server message that arrived on ``connection`` at ``now`` (the driver's
        monotonic clock). With no generation's reply being read on that connection the message is
        an orphan: nothing takes it, and it changes nothing (None; see arrived)."""
        generation = self.replies.get(connection)
        if generation is None:
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
            if complete:
                self.last_turn_clean = not generation.interrupted
        else:
            # Read from a closed connection: the session is no longer there to change.
            control = {}
        # IDLE ends the interaction whichever completion signal is configured; under turn_complete
        # it ends it without completing it.
        ends = complete or status == "IDLE"
        if ends:
            del self.replies[connection]
        return Received(
            connection,
            generation.number,
            generation.id,
            message,
            status,
            interrupted,
            generation.interrupted,
            complete,
            ends,
            control,
            _cancelled(message),
        )

    def arrived(self, connection: int, message, now: float) -> Received | Orphan:
        """Apply one server message as a reader that reads every message as it arrives does
        (streaming): as received() does while a generation's reply is being read on
        ``connection``, else as an orphan, whose goAway still asks for a new connection and whose
        resumption update is still the latest handle."""
        received = self.received(connection, message, now)
        if received is not None:
            return received
        control = (
            self._control(message, clean=False, now=now) if connection == self.connection else {}
        )
        return Orphan(connection, message, control, _cancelled(message))

    def _control(self, message, *, clean: bool, now: float) -> dict:
        update = getattr(message, "session_resumption_update", None)
        if update is not None:
            if not getattr(update, "resumable", False):
                self.candidate_resume_handle = None
            elif getattr(update, "new_handle", None):
                self.candidate_resume_handle = update.new_handle
                self.latest_resume_handle = update.new_handle
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
