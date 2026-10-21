"""The conversation of one OpenAI Realtime connection as the driver keeps it, without I/O.

The core mirrors what the server holds as far as the driver wrote it: each frame's user item, the
assistant items after it, and the archived-action ledger item; in native streaming, an item for
each block of the streaming history view, a frame, a reply's output or an execution feedback, in
the order the agent accepted them. From a history view it plans the item creates and deletes that
bring the server there, and it records what became of each one: unknown while it is written, sent
once the write returned, acknowledged once the server confirmed it. Each server event the driver
reads is handed to ``receive``, in the order the connection delivered it, which passes it to the
state it belongs to: an acknowledgement or error settles the in-place edit waiting for it, an
error about a block native streaming pushed refuses that block alone, a response event joins its
generation. Sending and receiving are the driver's; it reads only while something waits for an
event, so what arrives after the last wait reaches the core with the next read.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum

from realtime_framework.entity.context_policy import FixedTexts
from realtime_framework.entity.request import CanonicalRequest, FrameImage, RequestItem

# Neutral request roles as Realtime message roles.
ROLES = {"instruction": "system", "input": "user", "model": "assistant"}
# The server events that confirm a create, as the in-place edit has always waited for them. The
# trace's classification (openai_realtime.py _ITEM_ACKS) links a create to its first answer,
# conversation.item.added, as well.
_CREATE_ACKS = {"conversation.item.created", "conversation.item.done"}


@dataclass(frozen=True)
class Create:
    """One planned conversation.item.create of a message item. In-place edits anchor it and name
    its client event, so the server's acknowledgement or error can be matched to it."""

    item_id: str
    role: str
    text: str
    image: FrameImage | None = None
    previous_item_id: str | None = None
    event_id: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return ("create", self.item_id)


@dataclass(frozen=True)
class Delete:
    """One planned conversation.item.delete."""

    item_id: str
    event_id: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return ("delete", self.item_id)


Operation = Create | Delete


class Delivery(str, Enum):
    """What is known about one planned operation, in the Gemini Live core's terms
    (providers/gemini/session.py): unknown from the start of its write until the write returns,
    sent after that, acknowledged once the server confirmed it. A write that fails leaves it
    unknown: the server may or may not have it."""

    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    UNKNOWN = "unknown"


@dataclass
class SentFrame:
    """A frame as the driver last wrote it into the conversation."""

    user_item_id: str
    text: str
    folded: bool
    assistant_item_ids: list[str] = field(default_factory=list)


@dataclass
class Push:
    """A block native streaming pushed, a frame or an execution feedback: its block id, its user
    item id, and the server error that refused the item's create, if one did."""

    block_id: str
    item_id: str
    error: object | None = None


@dataclass(frozen=True)
class EditBatch:
    """One planned in-place edit and the session it leaves: the ledger item, each retained
    frame's user item id, text and folded state (a native-streaming edit lists the frames it
    replaces), and the frames it removes. It leaves assistant items as they are."""

    operations: tuple[Operation, ...]
    ledger_item_id: str | None
    ledger_text: str
    frames: dict[str, tuple[str, str, bool]]
    removed: tuple[str, ...]
    visuals_folded: int
    prompts_edited: int

    @property
    def created(self) -> set[str]:
        return {op.item_id for op in self.operations if isinstance(op, Create)}

    @property
    def deleted(self) -> set[str]:
        return {op.item_id for op in self.operations if isinstance(op, Delete)}

    @property
    def client_events(self) -> set[str]:
        return {op.event_id for op in self.operations}


@dataclass(frozen=True)
class ViewFrame:
    """One frame of a history view, with the frame prefix before it and the output after it."""

    item: RequestItem
    prefix: str = ""
    output: RequestItem | None = None

    @property
    def id(self) -> str:
        return self.item.block.id

    @property
    def text(self) -> str:
        """The frame's user item text: the frame prefix, then the frame."""
        text = self.item.block.text
        return f"{self.prefix}\n{text}" if self.prefix else text


def history_frames(items) -> tuple[RequestItem | None, list[ViewFrame]]:
    """Read request items as this session holds them: the ledger, then the frames in order."""
    ledger, frames, prefix = None, [], ""
    for item in items:
        block = item.block
        if block.kind == "action_ledger":
            ledger = item
        elif item.intent.position == "frame_prefix":
            prefix = block.text
        elif block.kind == "frame":
            frames.append(ViewFrame(item, prefix))
            prefix = ""
        elif block.kind == "model_output":
            frames[-1] = replace(frames[-1], output=item)
    return ledger, frames


def streaming_blocks(items) -> list[RequestItem]:
    """Read a native-streaming history view (harnesses/streaming/history.py): its blocks in the
    order the agent accepted them, frames, replies' outputs and execution feedback, each one item
    of the session."""
    return [item for item in items if item.intent.position == "in_order"]


@dataclass
class Generation:
    """One response on this connection: its number on the connection, the frame it answers (in
    native streaming, the output block it becomes instead), its response id once the server
    created it, the output items the server made for it, the client event that cancels it, and
    its server events not yet taken, in arrival order."""

    number: int
    answered: SentFrame | None
    response_id: str | None = None
    output_item_ids: list[str] = field(default_factory=list)
    cancel_event_id: str | None = None
    events: deque = field(default_factory=deque)
    output_id: str | None = None
    # When native streaming's reader received each completed output text, by server event id.
    received_at: dict[str, float] = field(default_factory=dict)


class RealtimeSessionCore:
    """One connection's conversation: what it holds, the operations that change it, their fate.

    ``new_id(kind)`` names each planned client item and client event. The mirror takes in a plan
    as it is made, before the driver sends it (an in-place edit only once it is acknowledged), so
    after a failed send it is ahead of the server; the session is not used again then, as the
    runtime reconnects or rebuilds (harnesses/default/context.py). A pushed block the server
    refuses leaves the mirror, and the session goes on without it; the history views it is edited
    to must leave the block out too, or they cannot be reached in place.
    """

    def __init__(self, new_id: Callable[[str], str]):
        self._new_id = new_id
        # The conversation as the driver last wrote it, which each history view is diffed
        # against: every frame block id with its items (in native streaming, every frame and
        # feedback block id), and the archived-action ledger item.
        self.frames: dict[str, SentFrame] = {}
        self.ledger_item_id: str | None = None
        self.ledger_text = ""
        # Every operation handed to the connection, by Operation.key.
        self.delivery: dict[tuple[str, str], Delivery] = {}
        # When native streaming's reader received the first event that confirmed each operation,
        # by Operation.key, on the driver's clock.
        self.acknowledged_at: dict[tuple[str, str], float] = {}
        # Every block native streaming pushed, by the client event of its create.
        self.pushes: dict[str, Push] = {}
        # The items of each reply native streaming requested, by the output block it becomes.
        self.outputs: dict[str, list[str]] = {}
        # The in-place edit waiting for the server: its operations not yet confirmed, and the
        # error that failed it.
        self._edit: EditBatch | None = None
        self._unconfirmed: set[tuple[str, str]] = set()
        self._edit_error = None
        # The history view a native-streaming edit keeps until the edit before it is finished.
        self._queued: CanonicalRequest | None = None
        # The responses in progress. The server answers each response.create with a
        # response.created in the order it received them, so a requested response waits here for
        # its id; every later event of a response names that id.
        self._requested: deque[Generation] = deque()
        self._responses: dict[str, Generation] = {}
        self._generations = 0

    # ---- planning ------------------------------------------------------------------------

    def _frame_create(
        self,
        frame: ViewFrame,
        *,
        previous_item_id=None,
        event_id: str | None = None,
    ) -> Create:
        """The user item of one frame: its text, and its screenshot unless folded."""
        return Create(
            self._new_id("user"),
            ROLES[frame.item.intent.role],
            frame.text,
            frame.item.block.image,
            previous_item_id,
            event_id,
        )

    def _output_create(self, output: RequestItem | None) -> Create | None:
        """The assistant item of a model output, if there is one."""
        if output is None:
            return None
        return Create(self._new_id("assistant"), ROLES[output.intent.role], output.block.text)

    def _ledger_create(
        self,
        ledger: RequestItem,
        texts: FixedTexts,
        *,
        previous_item_id=None,
        event_id: str | None = None,
    ) -> Create:
        """The archived-action ledger item, in the role the policy gives the ledger."""
        return Create(
            self._new_id("ledger"),
            ROLES[ledger.intent.role],
            texts.ledger_header + "\n" + ledger.block.text,
            None,
            previous_item_id,
            event_id,
        )

    def replay(self, history: CanonicalRequest | None) -> list[Create]:
        """Plan a new session's items from the history view (None for an empty session): the
        ledger, then each frame's user item and its output's assistant item. The mirror becomes
        the session they make."""
        ledger, frames = (None, []) if history is None else history_frames(history.items)
        operations: list[Create] = []
        if ledger is not None:
            create = self._ledger_create(ledger, history.policy.texts)
            operations.append(create)
            self.ledger_item_id, self.ledger_text = create.item_id, ledger.block.text
        for frame in frames:
            user = self._frame_create(frame)
            assistant = self._output_create(frame.output)
            operations += [user] if assistant is None else [user, assistant]
            self.frames[frame.id] = SentFrame(
                user.item_id,
                frame.text,
                frame.item.block.folded,
                [] if assistant is None else [assistant.item_id],
            )
        return operations

    def replay_streaming(self, history: CanonicalRequest | None) -> list[Create]:
        """Plan a new session's items from a native-streaming history view (None for an empty
        session), one per block in order: a frame's user item, a folded one the placeholder
        without its screenshot; a reply's assistant item, none for an empty output as a turn's
        has none; an execution feedback's user item. The mirror becomes the session they make."""
        operations: list[Create] = []
        for item in [] if history is None else streaming_blocks(history.items):
            block = item.block
            if block.kind == "model_output":
                assistant = self._output_create(item if block.text else None)
                self.outputs[block.id] = [] if assistant is None else [assistant.item_id]
                operations += [] if assistant is None else [assistant]
            else:
                user = self._frame_create(ViewFrame(item))
                self.frames[block.id] = SentFrame(user.item_id, block.text, block.folded)
                operations.append(user)
        return operations

    def observe(self, history: CanonicalRequest) -> Create:
        """Plan the user item of the newest frame; the history ends with it."""
        frame = history_frames(history.items)[1][-1]
        user = self._frame_create(frame)
        self.frames[frame.id] = SentFrame(user.item_id, frame.text, frame.item.block.folded)
        return user

    def push(self, history: CanonicalRequest) -> Create:
        """Plan the user item of a native-streaming history view's newest block, a frame or an
        execution feedback, as native streaming pushes it while responses and edits are in
        progress: its create names a client event, so a server error about it refuses this block
        alone."""
        item = streaming_blocks(history.items)[-1]
        user = self._frame_create(ViewFrame(item), event_id=self._new_id("push"))
        self.frames[item.block.id] = SentFrame(user.item_id, item.block.text, item.block.folded)
        self.pushes[user.event_id] = Push(item.block.id, user.item_id)
        return user

    def in_place_problem(self, history: CanonicalRequest) -> str | None:
        """Why the history view cannot be reached by editing this conversation, if it cannot.
        Checked before the first operation is sent, so a refused edit sends nothing."""
        for frame in history_frames(history.items)[1]:
            sent = self.frames.get(frame.id)
            if sent is None:
                return "retained turn is missing its provider user item ID"
            if sent.folded and not frame.item.block.folded:
                return "provider visual state cannot be unfolded in place"
        return None

    def begin_edit(self, history: CanonicalRequest) -> EditBatch:
        """Plan one in-place edit to the history view, which then waits for the server to
        confirm each of its operations."""
        ledger, frames = history_frames(history.items)
        return self._begin(self._edit_batch(ledger, frames, history.policy.texts))

    def _begin(self, batch: EditBatch) -> EditBatch:
        self._edit, self._edit_error = batch, None
        self._unconfirmed = {op.key for op in batch.operations}
        return batch

    def _replacement(
        self, frame: ViewFrame, sent: SentFrame, previous_item_id: str
    ) -> list[Operation]:
        """Replace a sent user item whose text changed or whose screenshot folded with the
        view's, where it was: delete it, then create the view's after ``previous_item_id``.
        Nothing when it is unchanged."""
        if frame.text == sent.text and not (frame.item.block.folded and not sent.folded):
            return []
        delete = Delete(sent.user_item_id, self._new_id("delete"))
        create = self._frame_create(
            frame, previous_item_id=previous_item_id, event_id=self._new_id("create")
        )
        return [delete, create]

    def _edit_batch(
        self, ledger: RequestItem | None, frames: list[ViewFrame], texts: FixedTexts
    ) -> EditBatch:
        """Plan one in-place edit from the history view, in send order: delete the items of sent
        frames the view no longer holds, replace the ledger if its text changed, then replace each
        user item whose text changed or whose screenshot folded, after the item before it. Every
        operation names its client event."""
        operations: list[Operation] = []
        retained = {frame.id for frame in frames}
        for frame_id, sent in self.frames.items():
            if frame_id not in retained:
                for item_id in (sent.user_item_id, *sent.assistant_item_ids):
                    operations.append(Delete(item_id, self._new_id("delete")))

        ledger_text = "" if ledger is None else ledger.block.text
        ledger_item_id = self.ledger_item_id
        if ledger_text != self.ledger_text:
            if self.ledger_item_id:
                operations.append(Delete(self.ledger_item_id, self._new_id("delete")))
            ledger_item_id = None
            if ledger is not None:
                create = self._ledger_create(
                    ledger, texts, previous_item_id="root", event_id=self._new_id("create")
                )
                operations.append(create)
                ledger_item_id = create.item_id

        previous_item_id = ledger_item_id or "root"
        sent_frames: dict[str, tuple[str, str, bool]] = {}
        folded_count = 0
        prompt_edit_count = 0
        for frame in frames:
            sent = self.frames[frame.id]
            folded = frame.item.block.folded
            replacement = self._replacement(frame, sent, previous_item_id)
            operations += replacement
            folded_count += int(folded and not sent.folded)
            prompt_edit_count += int(frame.text != sent.text)
            user_item_id = replacement[-1].item_id if replacement else sent.user_item_id
            sent_frames[frame.id] = (user_item_id, frame.text, folded)
            assistant_item_ids = sent.assistant_item_ids
            previous_item_id = assistant_item_ids[-1] if assistant_item_ids else user_item_id
        return EditBatch(
            tuple(operations),
            ledger_item_id,
            ledger_text,
            sent_frames,
            tuple(frame_id for frame_id in self.frames if frame_id not in retained),
            folded_count,
            prompt_edit_count,
        )

    def repair(self, history: CanonicalRequest, output_item_ids: list[str]) -> list[Operation]:
        """Plan the replacement of the output items a response left with the complete collected
        output: delete them, then append its assistant item. The response answered the history
        view's newest frame."""
        _, frames = history_frames(history.items)
        replacement = self._output_create(frames[-1].output)
        operations: list[Operation] = [Delete(item_id) for item_id in output_item_ids]
        if replacement is not None:
            operations.append(replacement)
        self.frames[frames[-1].id].assistant_item_ids = (
            [] if replacement is None else [replacement.item_id]
        )
        return operations

    # ---- what became of each operation ----------------------------------------------------

    def sending(self, operation: Operation) -> None:
        """Record an operation whose write starts, before any answer to it can arrive."""
        self.delivery[operation.key] = Delivery.UNKNOWN

    def sent(self, operation: Operation) -> None:
        # While another waiter reads, the acknowledgement can arrive before the write returns.
        if self.delivery[operation.key] is Delivery.UNKNOWN:
            self.delivery[operation.key] = Delivery.SENT

    @property
    def edit_answered(self) -> bool:
        """Whether the server confirmed every operation of the pending edit, or failed it."""
        return not self._unconfirmed or self._edit_error is not None

    def finish_edit(self) -> EditBatch:
        """End the answered edit. Confirmed whole, it is applied to the mirror in place, so a
        response that ended while it waited keeps its output items; a server error fails it, and
        the conversation is then whatever the server made of it."""
        batch, error = self._edit, self._edit_error
        self._edit, self._edit_error, self._unconfirmed = None, None, set()
        if error is not None:
            raise RuntimeError(f"Realtime history edit failed: {error!r}")
        self.ledger_item_id = batch.ledger_item_id
        self.ledger_text = batch.ledger_text
        for frame_id in batch.removed:
            del self.frames[frame_id]
        for frame_id, planned in batch.frames.items():
            sent = self.frames[frame_id]
            sent.user_item_id, sent.text, sent.folded = planned
        return batch

    def queue_edit(self, history: CanonicalRequest) -> EditBatch | None:
        """Plan a native-streaming edit to the streaming history view, or keep the view while an
        edit is not finished: one edit is sent at a time. Returns the batch to send now, if
        any."""
        self._queued = history
        return self.next_edit()

    def next_edit(self) -> EditBatch | None:
        """Plan the view a native-streaming edit kept, once the edit before it is finished. Only
        the newest view kept is planned, as each is the whole session wanted. A view the session
        cannot reach in place raises before anything is planned, and is dropped."""
        if self._edit is not None or self._queued is None:
            return None
        history, self._queued = self._queued, None
        blocks = self._held_blocks(history)
        if not all(self._holds(item.block) for item in blocks):
            # The view still holds a block whose push the server refused.
            raise RuntimeError(
                "Realtime history edit refused: retained block is missing its provider item ID"
            )
        return self._begin(self._fold_batch(blocks))

    def _holds(self, block) -> bool:
        return block.id in (self.outputs if block.kind == "model_output" else self.frames)

    def _held_blocks(self, history: CanonicalRequest) -> list[RequestItem]:
        """A native-streaming view's blocks up to the newest one the session holds, which an edit
        covers. The view's blocks after it are not pushed yet, or were refused; the session's
        blocks the view lacks were pushed after it was made, as a view keeps every block it once
        had (decision 4.6). The edit leaves both alone."""
        blocks = streaming_blocks(history.items)
        held = [index for index, item in enumerate(blocks) if self._holds(item.block)]
        return blocks[: held[-1] + 1] if held else []

    def _fold_batch(self, blocks: list[RequestItem]) -> EditBatch:
        """Plan one native-streaming edit from the blocks it covers, in send order: replace each
        user item whose text changed or whose screenshot folded, after the item of the block
        before it in the view, as a turn-based edit does (decision 23: the view folds each frame
        older than the newest K to the turn-based placeholder). A frame pushed while a reply ran
        is before the reply in the view but may be after its items on the server; its
        replacement goes where the view has it, as a rebuild would put it. Every block stays, so
        nothing else is deleted. Every operation names its client event."""
        operations: list[Operation] = []
        replaced: dict[str, tuple[str, str, bool]] = {}
        folded_count = 0
        prompt_edit_count = 0
        previous_item_id = "root"
        for item in blocks:
            block = item.block
            if block.kind == "model_output":
                if self.outputs[block.id]:
                    previous_item_id = self.outputs[block.id][-1]
                continue
            sent = self.frames[block.id]
            replacement = self._replacement(ViewFrame(item), sent, previous_item_id)
            operations += replacement
            folded_count += int(block.folded and not sent.folded)
            prompt_edit_count += int(block.text != sent.text)
            previous_item_id = replacement[-1].item_id if replacement else sent.user_item_id
            if replacement:
                replaced[block.id] = (previous_item_id, block.text, block.folded)
        return EditBatch(
            tuple(operations),
            self.ledger_item_id,
            self.ledger_text,
            replaced,
            (),
            folded_count,
            prompt_edit_count,
        )

    def start_generation(self, frame_id: str) -> Generation:
        """Register the response about to be requested for a sent frame. The first
        response.created not yet taken is its; its events go to it until its response.done."""
        self._generations += 1
        generation = Generation(self._generations, self.frames[frame_id])
        self._requested.append(generation)
        return generation

    def start_reply(self, output_id: str) -> Generation:
        """Register a response native streaming is about to request, whose output becomes the
        history view's block ``output_id``; it takes its events as start_generation's does."""
        self._generations += 1
        generation = Generation(self._generations, None, output_id=output_id)
        self._requested.append(generation)
        return generation

    def receive(self, event) -> bool:
        """Apply one server event of this connection, in arrival order. Returns whether a state
        took it: an operation it confirms, or the push, edit or response it is about."""
        kind = getattr(event, "type", "")
        key = self._acknowledged(kind, event)
        if key is not None:
            return self._confirm(key)
        if kind == "error":
            error = getattr(event, "error", None)
            event_id = getattr(error, "event_id", None)
            push = self.pushes.get(event_id)
            if push is not None:
                # The server did not make the block's item.
                push.error = error
                del self.frames[push.block_id]
                return True
            edit = self._edit
            about_edit = edit is not None and event_id in edit.client_events
            taken = False
            # An error that names no client event may be about any of them.
            if self._unconfirmed and self._edit_error is None and (event_id is None or about_edit):
                self._edit_error = error
                taken = True
            if not about_edit:
                live = [*self._requested, *self._responses.values()]
                cancelled = [g for g in live if event_id and g.cancel_event_id == event_id]
                for generation in cancelled or live:
                    generation.events.append(event)
                    taken = True
            return taken
        if kind.startswith("response."):
            generation = self._response_of(kind, event)
            if generation is None:
                return False
            generation.events.append(event)
            self._response_event(generation, kind, event)
            return True
        return False

    def arrived(self, event, now: float) -> bool:
        """Apply one server event as native streaming's reader does, which reads every event as it
        arrives, at ``now`` on the driver's clock: as receive() does, and record when the server
        acknowledged an operation and when a reply's completed output text came in. An action
        parsed from that text arrived then (decisions 19 and 4.17)."""
        if not self.receive(event):
            return False
        kind = getattr(event, "type", "")
        key = self._acknowledged(kind, event)
        if key is not None:
            self.acknowledged_at.setdefault(key, now)
        elif kind == "response.output_text.done":
            generation = self._responses[getattr(event, "response_id", None)]
            generation.received_at[getattr(event, "event_id", None)] = now
        return True

    @staticmethod
    def _acknowledged(kind: str, event) -> tuple[str, str] | None:
        """The operation a server event acknowledges, by Operation.key, if it is an
        acknowledgement."""
        if kind in _CREATE_ACKS:
            return ("create", str(getattr(getattr(event, "item", None), "id", "")))
        if kind == "conversation.item.deleted":
            return ("delete", str(getattr(event, "item_id", "")))
        return None

    def _confirm(self, key: tuple[str, str]) -> bool:
        self._unconfirmed.discard(key)
        if key not in self.delivery:
            return False
        self.delivery[key] = Delivery.ACKNOWLEDGED
        return True

    def _response_of(self, kind: str, event) -> Generation | None:
        """The generation a response event belongs to: response.created binds the earliest one
        still waiting for its id, every later event goes to the one its response id names."""
        if kind == "response.created":
            if not self._requested:
                return None
            generation = self._requested.popleft()
            generation.response_id = getattr(getattr(event, "response", None), "id", None)
            self._responses[generation.response_id] = generation
            return generation
        if kind == "response.done":
            return self._responses.get(getattr(getattr(event, "response", None), "id", None))
        return self._responses.get(getattr(event, "response_id", None))

    def _response_event(self, generation: Generation, kind: str, event) -> None:
        """Record what a response event says about the conversation: the output items the server
        made, and at response.done the answered frame's assistant items, or the items of the
        output block a native-streaming reply becomes."""
        if kind in {"response.output_item.added", "response.output_item.done"}:
            self._output_item(generation, getattr(getattr(event, "item", None), "id", None))
        elif kind == "response.done":
            response = getattr(event, "response", None)
            for item in getattr(response, "output", None) or []:
                self._output_item(generation, getattr(item, "id", None))
            if getattr(response, "status", None) != "failed":
                # A superseded response's session is closed and replayed before its next use.
                if generation.answered is None:
                    self.outputs[generation.output_id] = list(generation.output_item_ids)
                else:
                    generation.answered.assistant_item_ids = list(generation.output_item_ids)
            del self._responses[generation.response_id]

    @staticmethod
    def _output_item(generation: Generation, item_id) -> None:
        if item_id and item_id not in generation.output_item_ids:
            generation.output_item_ids.append(item_id)
