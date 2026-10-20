"""Gemini Live session core: what the session holds and what the driver does next, without I/O.

GeminiLiveDriver (live.py) renders the history, writes and reads the WebSocket and reads the clock.
It reports to this core what it rendered, wrote and received; the core keeps the session state and
answers with the operations to perform: how to open the next connection and what to seed on it,
whether a sync resumes or rebuilds, which tool calls to acknowledge, which connection and
generation a server message belongs to and whether that generation's output ends with it. Nothing
in it awaits, so each call is one transition, in the order of the session.

Each connection keeps what the client delivered on it, by key: ``("setup",)``, ``("block", <history
block id>)`` and ``("tool_response", <call id>)``. A write is unknown from its start until it
returns, then sent; the setup is the only message the server acknowledges (setupComplete).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

SENT = "sent"
ACKNOWLEDGED = "acknowledged"
UNKNOWN = "unknown"

# Keep initial-history WebSocket messages small. HistoryConfig deliberately allows several
# client-content messages terminated by one turn_complete=True boundary.
HISTORY_CONTENTS_PER_MESSAGE = 12
_GO_AWAY_CLOSE_MARGIN_S = 2.0


def history_batches(history: list, limit: int = HISTORY_CONTENTS_PER_MESSAGE) -> list[list]:
    """Pack history without splitting one observation/model/tool-response exchange."""
    groups = []
    group = []
    for content in history:
        parts = getattr(content, "parts", None) or []
        is_function_response = any(
            getattr(part, "function_response", None) is not None for part in parts
        )
        has_function_call = any(getattr(part, "function_call", None) is not None for part in parts)
        if getattr(content, "role", None) == "user" and group and not is_function_response:
            groups.append(group)
            group = []
        group.append(content)
        if is_function_response or (
            getattr(content, "role", None) == "model" and not has_function_call
        ):
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
    # The current SDK exposes status on server_content. Accept the top-level spelling too so
    # the driver remains compatible with the wire shape documented for tool-call messages.
    status = getattr(message, "interaction_status", None)
    if status is None:
        status = getattr(getattr(message, "server_content", None), "interaction_status", None)
    status = getattr(status, "value", status)
    if status is None:
        return None
    return str(status).rsplit(".", 1)[-1].upper()


@dataclass(frozen=True)
class Connect:
    """Open the next connection, once the current one is closed.

    Wait the session-release grace first when ``release_grace`` is set. The setup resumes
    ``resume_handle``, or declares HistoryConfig when there is a seed; each seed batch then goes out
    as client content, the last one completing the turn.
    """

    resume_handle: str | None
    seed: tuple[list, ...]  # batches of the rendered history
    seeded: tuple[str, ...]  # ids of the history-view blocks the seed replays
    release_grace: bool

    @property
    def mode(self) -> str:
        if self.resume_handle is not None:
            return "session_resumption"
        return "canonical_history_replay" if self.seed else "new_session"


@dataclass(frozen=True)
class Received:
    """One server message, as the core applied it."""

    connection: int  # the connection it arrived on
    generation: int  # the generation on that connection whose reply was being read
    generation_id: str  # that generation's request id
    message: Any
    status: str | None  # interaction_status
    interrupted: bool  # this message reports an interruption
    contaminated: bool  # the generation was interrupted, by this message or an earlier one
    complete: bool  # the interaction completes here, by the configured completion signal
    ends: bool  # the generation's output ends with this message
    control: dict  # what a goAway asks of the turn: reconnect after it


@dataclass
class _Generation:
    number: int  # on its connection, from 1
    id: str
    interrupted: bool = False


class SessionCore:
    """The Live session as this client wrote it, across the connections that carry it.

    Closing a connection clears what belonged to it; the session state stays until the next
    connection replaces it. A reply still being read from a closed connection keeps its
    generation, and is otherwise applied like any other message.
    """

    def __init__(self, completion_signal: str):
        self._completion_signal = completion_signal
        self._connections = 0  # connections set up so far
        self._generations = 0  # generations started on the open connection
        self.connection: int | None = None  # the open connection's number, from 1
        self.delivery: dict[tuple, str] = {}  # what the client delivered on it
        self.candidate_resume_handle: str | None = None  # latest resumption update of this turn
        # (id, name) of the tool calls received and not yet acknowledged.
        self.pending_calls: list[tuple] = []
        self.go_away_deadline: float | None = None  # on the driver's monotonic clock
        self.resume_handle: str | None = None  # promoted at a clean interaction boundary
        self.renewal_required = False  # a goAway asked for a new connection
        self.last_turn_clean = False
        # The generation whose reply is being read, by connection, until the reply ends.
        self.replies: dict[int, _Generation] = {}

    # ---- connections ---------------------------------------------------------------------

    def connect(self, history: list, seeded: tuple[str, ...], resume_handle) -> Connect:
        """Close the open connection and plan the next one. A resume carries no history: the
        server restores the session from the handle."""
        replacing = self.connection is not None
        self.closed()
        self.resume_handle = resume_handle
        return Connect(
            resume_handle,
            tuple(history_batches(history)),
            seeded,
            release_grace=replacing or bool(history),
        )

    def connected(self, connect: Connect) -> None:
        """The setup was acknowledged and the whole seed sent: the connection counts from here."""
        self._connections += 1
        self._generations = 0
        self.connection = self._connections
        self.delivery = {("setup",): ACKNOWLEDGED}
        self.delivery.update((("block", block_id), SENT) for block_id in connect.seeded)
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
        renewal after a clean interaction resumes."""
        if reason == "go_away" and not force_rebuild and self.last_turn_clean and generation_clean:
            return self.resume_handle
        return None

    # ---- writes ----------------------------------------------------------------------------

    def sending(self, keys) -> None:
        for key in keys:
            self.delivery[key] = UNKNOWN

    def sent(self, keys) -> None:
        for key in keys:
            self.delivery[key] = SENT

    def input_started(self, keys) -> None:
        """An observation is being written: no handle resumes the session until its turn
        completes cleanly."""
        self.last_turn_clean = False
        self.resume_handle = None
        self.candidate_resume_handle = None
        self.sending(keys)

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
        self.sent(("tool_response", call_id) for call_id, _ in self.pending_calls)
        self.pending_calls.clear()

    # ---- reads -----------------------------------------------------------------------------

    def generation_started(self, generation_id: str) -> None:
        self._generations += 1
        self.replies[self.connection] = _Generation(self._generations, generation_id)

    def received(self, connection: int, message, now: float) -> Received:
        """Apply one server message that arrived on ``connection`` at ``now`` (the driver's
        monotonic clock)."""
        server = getattr(message, "server_content", None)
        status = interaction_status(message)
        if self._completion_signal == "interaction_idle":
            complete = status == "IDLE"
        else:
            complete = bool(server is not None and getattr(server, "turn_complete", False))
        interrupted = bool(server is not None and getattr(server, "interrupted", False))
        generation = self.replies[connection]
        generation.interrupted = generation.interrupted or interrupted
        control = self._control(message, clean=complete and not generation.interrupted, now=now)
        if complete:
            self.last_turn_clean = not generation.interrupted
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
        )

    def _control(self, message, *, clean: bool, now: float) -> dict:
        update = getattr(message, "session_resumption_update", None)
        if update is not None:
            if not getattr(update, "resumable", False):
                self.candidate_resume_handle = None
            elif getattr(update, "new_handle", None):
                self.candidate_resume_handle = update.new_handle
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
