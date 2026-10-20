"""The local frame table: every frame the agent prepares for the model, kept for the whole episode.

An action is interpreted against one frame's entry: its SoM marks resolve the action's element
references and its geometry maps the action's image coordinates to the desktop. The entry stays
when the provider context folds or drops the frame. A turn-based action is interpreted against the
newest frame.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameRecord:
    frame: int  # 1-based, in the order frames are prepared for the model
    observation_id: str
    captured_unix_s: float  # Observation.timestamp
    geometry: dict  # FrameGeometry of the image the model is sent
    marks: dict | None  # the SoM mark map; None when the frame had no inspection
    # "prepared" until its request or session push went out, then "sent".
    send_status: str = "prepared"
    sent_unix_s: float | None = None


class FrameTable:
    def __init__(self) -> None:
        self._frames: dict[int, FrameRecord] = {}

    def register(
        self, observation_id: str, captured_unix_s: float, geometry: dict, marks: dict | None
    ) -> FrameRecord:
        record = FrameRecord(
            len(self._frames) + 1, observation_id, captured_unix_s, geometry, marks
        )
        self._frames[record.frame] = record
        return record

    def record_sent(self, frame: int, sent_unix_s: float) -> None:
        record = self._frames[frame]
        record.send_status = "sent"
        record.sent_unix_s = sent_unix_s

    def latest(self) -> FrameRecord | None:
        return self._frames.get(len(self._frames))
