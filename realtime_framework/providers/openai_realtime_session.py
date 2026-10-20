"""The conversation of one OpenAI Realtime connection as the driver keeps it, without I/O.

The core mirrors what the server holds as far as the driver wrote it: each frame's user item, the
assistant items after it, and the archived-action ledger item. From a history view it plans the
item creates and deletes that bring the server there, and it records what became of each one:
sent, confirmed by the server, or unknown when the send failed. Server events are handed to
``receive`` in the order the connection delivered them; an in-place edit is settled or failed
there. Sending and receiving are the driver's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum

from realtime_framework.entity.context_policy import FixedTexts
from realtime_framework.entity.request import CanonicalRequest, FrameImage, RequestItem

# Neutral request roles as Realtime message roles.
ROLES = {"instruction": "system", "input": "user", "model": "assistant"}
# The server events that confirm a create; conversation.item.added does not.
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
    """What is known about one planned operation."""

    SENT = "sent"
    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"  # the send failed, so the server may or may not have it


@dataclass
class SentFrame:
    """A frame as the driver last wrote it into the conversation."""

    user_item_id: str
    text: str
    folded: bool
    assistant_item_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EditBatch:
    """One planned in-place edit and the session it leaves: the ledger item and the frames."""

    operations: tuple[Operation, ...]
    ledger_item_id: str | None
    ledger_text: str
    frames: dict[str, SentFrame]
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


class RealtimeSessionCore:
    """One connection's conversation: what it holds, the operations that change it, their fate.

    ``new_id(kind)`` names each planned client item and client event.
    """

    def __init__(self, new_id: Callable[[str], str]):
        self._new_id = new_id
        # The conversation as the driver last wrote it, which each history view is diffed
        # against: every frame block id with its items, and the archived-action ledger item.
        self.frames: dict[str, SentFrame] = {}
        self.ledger_item_id: str | None = None
        self.ledger_text = ""
        # Every operation handed to the connection, by Operation.key.
        self.delivery: dict[tuple[str, str], Delivery] = {}
        # The in-place edit waiting for the server: its operations not yet confirmed, and the
        # error that failed it.
        self._edit: EditBatch | None = None
        self._unconfirmed: set[tuple[str, str]] = set()
        self._edit_error = None

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

    def _output_create(self, frame: ViewFrame) -> Create | None:
        """The assistant item of the model output that answered a frame, if it has one."""
        if frame.output is None:
            return None
        return Create(
            self._new_id("assistant"),
            ROLES[frame.output.intent.role],
            frame.output.block.text,
        )

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
            assistant = self._output_create(frame)
            operations += [user] if assistant is None else [user, assistant]
            self.frames[frame.id] = SentFrame(
                user.item_id,
                frame.text,
                frame.item.block.folded,
                [] if assistant is None else [assistant.item_id],
            )
        return operations

    def observe(self, history: CanonicalRequest) -> Create:
        """Plan the user item of the newest frame; the history ends with it."""
        frame = history_frames(history.items)[1][-1]
        user = self._frame_create(frame)
        self.frames[frame.id] = SentFrame(user.item_id, frame.text, frame.item.block.folded)
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
        batch = self._edit_batch(ledger, frames, history.policy.texts)
        self._edit, self._edit_error = batch, None
        self._unconfirmed = {op.key for op in batch.operations}
        return batch

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
        sent_frames: dict[str, SentFrame] = {}
        folded_count = 0
        prompt_edit_count = 0
        for frame in frames:
            sent = self.frames[frame.id]
            folded = frame.item.block.folded
            user_item_id = sent.user_item_id
            prompt_changed = frame.text != sent.text
            if prompt_changed or (folded and not sent.folded):
                operations.append(Delete(user_item_id, self._new_id("delete")))
                replacement = self._frame_create(
                    frame,
                    previous_item_id=previous_item_id,
                    event_id=self._new_id("create"),
                )
                operations.append(replacement)
                folded_count += int(folded and not sent.folded)
                prompt_edit_count += int(prompt_changed)
                user_item_id = replacement.item_id
            assistant_item_ids = list(sent.assistant_item_ids)
            sent_frames[frame.id] = SentFrame(user_item_id, frame.text, folded, assistant_item_ids)
            previous_item_id = assistant_item_ids[-1] if assistant_item_ids else user_item_id
        return EditBatch(
            tuple(operations),
            ledger_item_id,
            ledger_text,
            sent_frames,
            folded_count,
            prompt_edit_count,
        )

    def repair(self, history: CanonicalRequest, output_item_ids: list[str]) -> list[Operation]:
        """Plan the replacement of the output items a response left with the complete collected
        output: delete them, then append its assistant item. The response answered the history
        view's newest frame."""
        _, frames = history_frames(history.items)
        replacement = self._output_create(frames[-1])
        operations: list[Operation] = [Delete(item_id) for item_id in output_item_ids]
        if replacement is not None:
            operations.append(replacement)
        self.frames[frames[-1].id].assistant_item_ids = (
            [] if replacement is None else [replacement.item_id]
        )
        return operations

    # ---- what became of each operation ----------------------------------------------------

    def sending(self, operation: Operation) -> None:
        """Record an operation as handed to the connection, before any answer can arrive."""
        self.delivery[operation.key] = Delivery.SENT

    def send_failed(self, operation: Operation) -> None:
        self.delivery[operation.key] = Delivery.UNKNOWN

    @property
    def edit_answered(self) -> bool:
        """Whether the server confirmed every operation of the pending edit, or failed it."""
        return not self._unconfirmed or self._edit_error is not None

    def finish_edit(self) -> EditBatch:
        """End the answered edit. Confirmed whole, the session it leaves becomes the mirror; a
        server error fails it, and the conversation is then whatever the server made of it."""
        batch, error = self._edit, self._edit_error
        self._edit, self._edit_error, self._unconfirmed = None, None, set()
        if error is not None:
            raise RuntimeError(f"Realtime history edit failed: {error!r}")
        self.ledger_item_id = batch.ledger_item_id
        self.ledger_text = batch.ledger_text
        self.frames = batch.frames
        return batch

    def receive(self, event) -> None:
        """Apply one server event of this connection, in arrival order."""
        kind = getattr(event, "type", "")
        if kind in _CREATE_ACKS:
            self._confirm(("create", str(getattr(getattr(event, "item", None), "id", ""))))
        elif kind == "conversation.item.deleted":
            self._confirm(("delete", str(getattr(event, "item_id", ""))))
        elif kind == "error":
            error = getattr(event, "error", None)
            event_id = getattr(error, "event_id", None)
            edit = self._edit
            # An error that names no client event may be about any of them.
            if (
                self._unconfirmed
                and self._edit_error is None
                and (event_id is None or event_id in edit.client_events)
            ):
                self._edit_error = error

    def _confirm(self, key: tuple[str, str]) -> None:
        if key in self.delivery:
            self.delivery[key] = Delivery.CONFIRMED
        self._unconfirmed.discard(key)
