"""Screenshot appearance of visible Web grid cells and drop targets for text-observation models."""
from __future__ import annotations

import hashlib
import io
from collections import Counter

from PIL import Image, ImageStat
from PIL import __version__ as pillow_version

POLICY = "jev_web_control_pixel_patches_v1"
COLORS = (
    ("black", (0, 0, 0)), ("white", (255, 255, 255)), ("gray", (128, 128, 128)),
    ("red", (220, 40, 40)), ("orange", (240, 140, 40)), ("yellow", (240, 225, 60)),
    ("green", (40, 160, 60)), ("cyan", (60, 200, 210)), ("blue", (40, 80, 220)),
    ("purple", (150, 60, 180)), ("brown", (140, 95, 50)), ("pink", (240, 160, 185)),
)


def perception_identity() -> dict:
    return {"policy": POLICY, "pillow_version": pillow_version}


def _palette() -> Image.Image:
    palette = Image.new("P", (1, 1))
    rgb = [channel for _, color in COLORS for channel in color]
    palette.putpalette(rgb + list(COLORS[0][1]) * (256 - len(COLORS)))
    return palette


def patch_content(image: Image.Image, box: list[int]) -> dict:
    left, top, right, bottom = box
    box = (max(0, left), max(0, top), min(image.width, right), min(image.height, bottom))
    if box[2] <= box[0] or box[3] <= box[1]:
        return {"source": "screenshot", "status": "outside_capture"}
    patch = image.crop(box).convert("RGB")
    small = patch.resize((12, 12), Image.Resampling.BOX)
    counts = Counter(small.quantize(palette=_palette(), dither=Image.Dither.NONE).getdata())
    colors = [{"color": COLORS[index][0], "fraction": round(count / 144, 2)}
              for index, count in counts.most_common(3)]
    return {
        "source": "screenshot", "colors": colors,
        "contrast": round(ImageStat.Stat(small.convert("L")).stddev[0], 1),
        "patch_sha256": hashlib.sha256(patch.tobytes()).hexdigest(),
    }


def attach_web_control_appearance(result: dict, image_bytes: bytes, *, width: int, height: int) -> None:
    """Annotate visible DOM grid cells and targets from the screenshot."""
    if not image_bytes:
        return
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scale_x, scale_y = image.width / width, image.height / height
    groups = [result.get("actionable_now", [])]
    groups.extend(window.get("interactables", [])
                  for window in result.get("windows", []) if window.get("active"))
    cache = {}
    for items in groups:
        for item in items:
            if (item.get("availability") == "canvas_surfaces"
                    or item.get("canvas_id") or item.get("tag") == "canvas-target"
                    or item.get("role") not in {"gridcell", "target"}):
                continue
            box = item.get("bbox")
            if not isinstance(box, list) or len(box) != 4:
                continue
            key = (item["id"], tuple(box))
            if key not in cache:
                source_box = [round(box[0] * scale_x), round(box[1] * scale_y),
                              round(box[2] * scale_x), round(box[3] * scale_y)]
                cache[key] = patch_content(image, source_box)
            item["appearance"] = cache[key]
