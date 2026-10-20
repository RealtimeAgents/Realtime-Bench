from __future__ import annotations

import io
from PIL import Image
from realtime_framework.entity.actions import (
    DESKTOP_SIZE,
    FrameGeometry,
    PreparedFrame,
    Observation,
    _positive_pair,
)


def image_size(data: bytes, fallback: tuple[int, int] = DESKTOP_SIZE) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return _positive_pair(image.size, fallback)
    except (OSError, ValueError):
        return fallback


def prepare_frame(
    data: bytes,
    *,
    screen_size: tuple[int, int] = DESKTOP_SIZE,
    max_dimension: int = 0,
    jpeg_quality: int = 85,
    detail: str = "original",
) -> PreparedFrame:
    """Prepare model bytes and return the exact geometry of the transmitted image.

    The default is lossless and size-preserving.  An explicit ``max_dimension`` enables
    downscaling, but its affine transform travels with the frame instead of becoming an
    undocumented adapter convention.
    """
    original_size = image_size(data, screen_size)
    if max_dimension <= 0 or max(original_size) <= max_dimension:
        return PreparedFrame(
            data, "png", detail, FrameGeometry(original_size, screen_size=screen_size)
        )
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert("RGB")
            scale = max_dimension / max(image.size)
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.BILINEAR)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=max(1, min(100, jpeg_quality)))
        return PreparedFrame(
            output.getvalue(), "jpeg", detail, FrameGeometry(size, screen_size=screen_size)
        )
    except (OSError, ValueError):
        return PreparedFrame(
            data, "png", detail, FrameGeometry(original_size, screen_size=screen_size)
        )


def ensure_observation_geometry(
    observation: Observation, screen_size: tuple[int, int] = DESKTOP_SIZE
) -> FrameGeometry:
    metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
    observation.metadata = metadata
    existing = metadata.get("frame_geometry")
    if isinstance(existing, dict):
        return FrameGeometry.from_dict(existing)
    geometry = FrameGeometry(image_size(observation.screenshot, screen_size), screen_size)
    metadata["frame_geometry"] = geometry.to_dict()
    return geometry


# Browser scene
"""Bind a desktop screenshot to every visible Chrome window's interactable DOM snapshot."""

import hashlib as hashlib
import threading as threading
import time as time
from typing import Callable as Callable

from realtime_framework.entity.config import SANDBOX_CONTAINER
from realtime_framework.entity.actions import DESKTOP_SIZE
from realtime_framework.sandbox.browser.cdp import close_port
from realtime_framework.sandbox.browser.cdp import ensure_page_focus
from realtime_framework.sandbox.browser.cdp import probe_interactables
from realtime_framework.sandbox.observation.interactables import canonical_result
from realtime_framework.sandbox.observation.interactables import load_tool_spec

_CAPTURE_BINDING_ATTEMPTS = 3


class BrowserSceneCapture:
    """Capture screenshots plus immutable, observation-bound browser interactables."""

    def __init__(
        self,
        guest,
        task=None,
        *,
        debug_port: int | None = None,
        container: str = SANDBOX_CONTAINER,
        probe: Callable[[dict], dict] | None = None,
        desktop_size: tuple[int, int] = DESKTOP_SIZE,
        text_observation: bool = False,
    ):
        self.guest = guest
        self.task = task
        self.task_id = getattr(task, "task_id", None)
        self.debug_port = debug_port
        self.container = container
        self.spec = load_tool_spec(task)
        # Text-observation models also receive the accessibility tree and rendered control state.
        self.text_observation = text_observation
        if text_observation:
            self.spec = {**self.spec, "__text_observation": True}
        self.desktop_size = desktop_size
        self._probe_override = probe
        self._lock = threading.Lock()
        self._sequence = 0
        self._target_hint = self._infer_target_hint(task)
        # The page viewport in desktop pixels from the latest probe; the browser toolbar, tabs and
        # desktop panel lie outside it and are not part of the task.
        self.page_rect: tuple[int, int, int, int] | None = None

    @staticmethod
    def _infer_target_hint(task) -> str:
        if task is None:
            return ""
        port = getattr(task, "http_port", None)
        if port:
            return f"localhost:{port}"
        scenario = getattr(task, "scenario", None)
        return str(scenario or "")

    def _next_id(self) -> str:
        with self._lock:
            self._sequence += 1
            return f"{self.task_id or 'desktop'}:{self._sequence:06d}"

    def _probe(self, *, include_ax_tree: bool = False) -> dict:
        payload = {
            "target_hint": self._target_hint,
            "spec": self.spec,
            "desktop_size": list(self.desktop_size),
        }
        if self._probe_override is not None:
            return self._probe_override(payload)
        if self.debug_port is None:
            return {
                "scene": {
                    "status": "unsupported",
                    "kind": "desktop",
                    "reason": "task_has_no_browser_debug_port",
                },
                "binding": {"stable": True, "reason_codes": ["no_debug_port"]},
                "actionable_now": [],
                "visible_but_unavailable": [],
                "canvas_surfaces": [],
                "limitations": ["browser_dom_inspection_unavailable"],
            }
        try:
            value = probe_interactables(
                self.debug_port,
                target_hint=payload["target_hint"],
                spec=payload["spec"],
                desktop_size=tuple(payload["desktop_size"]),
                include_ax_tree=include_ax_tree,
            )
            if not isinstance(value, dict):
                raise RuntimeError("CDP helper returned a non-object")
            return value
        except Exception as exc:
            return {
                "scene": {
                    "status": "error",
                    "kind": "browser_dom",
                    "reason": f"capture_probe_failed:{type(exc).__name__}",
                },
                "binding": {
                    "stable": False,
                    "unstable": True,
                    "reason_codes": ["capture_probe_error"],
                },
                "actionable_now": [],
                "visible_but_unavailable": [],
                "canvas_surfaces": [],
                "warnings": [str(exc)[:240]],
            }

    def close(self) -> None:
        if self.debug_port is not None:
            close_port(self.debug_port)

    def ensure_page_focus(self) -> bool:
        """Return keyboard focus to the page; True when it had moved to the browser toolbar."""
        if self.debug_port is None or self._probe_override is not None:
            return False
        return ensure_page_focus(self.debug_port, target_hint=self._target_hint)

    def _page_rect(self, raw: dict) -> tuple[int, int, int, int] | None:
        metrics = raw.get("metrics") if raw.get("scene", {}).get("status") == "ok" else None
        if not isinstance(metrics, dict):
            return None
        try:
            scale_x = self.desktop_size[0] / (float(metrics.get("screenWidth")) or 1.0)
            scale_y = self.desktop_size[1] / (float(metrics.get("screenHeight")) or 1.0)
            left = float(metrics["contentLeft"])
            top = float(metrics["contentTop"])
            right = left + float(metrics["innerWidth"])
            bottom = top + float(metrics["innerHeight"])
        except (KeyError, TypeError, ValueError):
            return None
        if right <= left or bottom <= top:
            return None
        return (
            round(left * scale_x),
            round(top * scale_y),
            round(right * scale_x),
            round(bottom * scale_y),
        )

    def _enrich(self, raw: dict) -> dict:
        raw = dict(raw)
        keys = []
        provider = getattr(self.task, "interactable_providers", None)
        if callable(provider):
            try:
                descriptors = provider() or ()
                for descriptor in descriptors:
                    if not isinstance(descriptor, dict) or descriptor.get("type") != "keyboard":
                        continue
                    for value in descriptor.get("keys", []):
                        if isinstance(value, str) and value:
                            keys.append(
                                {
                                    "id": f"key:{value}",
                                    "key": value,
                                    "label": value,
                                    "source": "scenario_declared",
                                }
                            )
            except Exception as exc:
                raise RuntimeError(
                    f"scenario interactables provider failed: {type(exc).__name__}: {exc}"
                ) from exc
        raw["keyboard_actions"] = keys
        return raw

    def capture(self, tick_number: float) -> Observation:
        observation_id = self._next_id()
        render_sync_attempts = 0
        binding_capture_attempts = 0
        for binding_capture_attempts in range(1, _CAPTURE_BINDING_ATTEMPTS + 1):
            # The AX tree is read once, with the observation's binding probe.
            pre = self._probe(include_ax_tree=self.text_observation)
            screenshot = self.guest.screenshot()
            render_sync_attempts += 1
            frame_is_current = getattr(self.task, "captured_frame_is_current", None)
            if callable(frame_is_current):
                deadline = time.monotonic() + 5.0
                while not frame_is_current(screenshot):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"desktop capture never reached the render fence for {self.task_id}"
                        )
                    time.sleep(0.01)
                    screenshot = self.guest.screenshot()
                    render_sync_attempts += 1
            captured_at = time.time()
            post = self._probe() if pre.get("scene", {}).get("status") == "ok" else pre
            pre_binding = pre.setdefault("binding", {})
            if pre.get("scene", {}).get("status") != "ok":
                break
            same = pre.get("scene", {}).get("target_id") == post.get("scene", {}).get(
                "target_id"
            ) and pre_binding.get("fingerprint") == post.get("binding", {}).get("fingerprint")
            pre_binding["stable"] = bool(same)
            pre_binding["unstable"] = not same
            if same:
                break
            pre_binding.setdefault("reason_codes", []).append(
                "interactables_changed_during_capture"
            )
        page_rect = self._page_rect(pre)
        if page_rect is not None:
            self.page_rect = page_rect
        digest = hashlib.sha256(screenshot).hexdigest()
        geometry = FrameGeometry(image_size(screenshot, self.desktop_size), self.desktop_size)
        canonical = canonical_result(
            self._enrich(pre),
            observation_id=observation_id,
            screenshot_sha256=digest,
            captured_at=captured_at,
            task_id=self.task_id,
            frame_geometry=geometry,
        )
        return Observation(
            screenshot=screenshot,
            timestamp=captured_at,
            tick_number=tick_number,
            metadata={
                "observation_id": observation_id,
                "screenshot_sha256": digest,
                "frame_geometry": geometry.to_dict(),
                "interactables": canonical,
                "render_sync_attempts": render_sync_attempts,
                "binding_capture_attempts": binding_capture_attempts,
            },
        )


# Frames
"""Shared screenshot encoding policy for VLM and native multimodal adapters."""
from realtime_framework.entity.config import FrameEncodingConfig


FRAME_ENCODING = FrameEncodingConfig.from_environ()  # read once, at import


def prepare_for_model(png_bytes: bytes):
    """Prepare a chat-VLM frame and retain its exact coordinate transform."""
    return prepare_frame(
        png_bytes,
        max_dimension=FRAME_ENCODING.max_dimension,
        jpeg_quality=FRAME_ENCODING.jpeg_quality,
        detail=FRAME_ENCODING.image_detail,
    )
