"""The local frame table: every frame the agent prepares for the model, kept for the whole episode.

An action is interpreted against one frame's entry: its SoM marks resolve the action's element
references and its geometry maps the action's image coordinates to the desktop. The entry stays
when the provider context folds or drops the frame. A streaming action names the frame it is based
on; a turn-based action is interpreted against the frame of its own generation. A captured frame
dropped before it is prepared takes no number, so the numbers the model sees have no gaps; the
streaming policy counts the drops (evaluation/kernel/streaming.py).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

# A streaming action whose frame cannot be found is rejected, not executed (phase2.md, 动作的定位依据).
INVALID_GROUNDING_FRAME = "invalid_grounding_frame"
# The frame cannot be found because the action names no frame, names it with something other
# than an integer, or names a frame that was never sent.
MISSING_FRAME = "missing_frame"
INVALID_FRAME_SYNTAX = "invalid_frame_syntax"
FRAME_NOT_IN_TABLE = "frame_not_in_table"


@dataclass
class FrameRecord:
    frame: int  # 1-based, in the order frames are prepared for the model
    observation_id: str
    # Observation.timestamp: wall time, or simulated seconds on the token clock (sandbox/actions.py).
    captured_at: float
    geometry: dict  # FrameGeometry of the image the model is sent
    marks: dict | None  # the SoM mark map; None when the frame had no inspection
    # "prepared" until its request or session push went out, then "sent".
    send_status: str = "prepared"
    sent_unix_s: float | None = None
    # Streaming: whether the provider confirmed it holds the pushed frame. "acked", when the
    # confirmation arrived (OpenAI Realtime's conversation.item.created or .done); "unknown"
    # where the protocol confirms no frame (Gemini Live). None until the driver says which.
    ack_status: str | None = None
    acked_unix_s: float | None = None


class FrameTable:
    """Streaming may enter, send and look up frames on different threads, so each method holds
    the table's lock."""

    def __init__(self) -> None:
        self._frames: dict[int, FrameRecord] = {}
        self._lock = threading.Lock()

    def register(
        self, observation_id: str, captured_at: float, geometry: dict, marks: dict | None
    ) -> FrameRecord:
        with self._lock:
            frame = len(self._frames) + 1
            record = FrameRecord(frame, observation_id, captured_at, geometry, marks)
            self._frames[record.frame] = record
            return record

    def record_sent(self, frame: int, sent_unix_s: float) -> None:
        with self._lock:
            record = self._frames[frame]
            record.send_status = "sent"
            record.sent_unix_s = sent_unix_s

    def record_acked(self, frame: int, acked_unix_s: float) -> None:
        with self._lock:
            record = self._frames[frame]
            record.ack_status = "acked"
            record.acked_unix_s = acked_unix_s

    def record_ack_unknown(self, frame: int) -> None:
        with self._lock:
            self._frames[frame].ack_status = "unknown"

    def latest(self) -> FrameRecord | None:
        with self._lock:
            return self._frames.get(len(self._frames))

    def locate(self, reference: object) -> tuple[FrameRecord | None, str | None]:
        """The sent frame an action's frame field names, or None and why it cannot be found."""
        if reference is None:
            return None, MISSING_FRAME
        if type(reference) is not int:
            return None, INVALID_FRAME_SYNTAX
        with self._lock:
            record = self._frames.get(reference)
            if record is None or record.send_status != "sent":
                return None, FRAME_NOT_IN_TABLE
            return record, None

    def age(self, record: FrameRecord, at_unix_s: float) -> dict:
        """How old a frame is when an action based on it arrives: the frames sent after it by
        then (not those still unsent), and the seconds since its capture. Streaming runs on the
        wall clock only, so its frames' capture times are wall time as well.

        ``at_unix_s`` must be read as the send times were, with Clock.time(): a frame sent at the
        arrival instant counts, but the same instant read another way, as Clock.time_ns() / 1e9,
        can round to just after it."""
        with self._lock:
            newer = (self._frames[n] for n in range(record.frame + 1, len(self._frames) + 1))
            return {
                "age_frames": sum(
                    other.send_status == "sent" and other.sent_unix_s <= at_unix_s
                    for other in newer
                ),
                "age_s": at_unix_s - record.captured_at,
            }
