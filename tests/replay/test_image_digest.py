"""Replay baselines compare images by their pixels (baseline.image_digest): another PNG encoding
of the same pixels compares equal, a single changed pixel does not, and a field that holds only
an image's SHA-256 needs the image registered.

The images here are drawn for this module alone, so no other test in the process registers them.
"""

from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image

from . import baseline


def _png(image: Image.Image, **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **options)
    return buffer.getvalue()


def test_the_encoding_does_not_count_and_every_pixel_does():
    image = Image.new("RGB", (64, 48), (245, 245, 245))
    image.paste((30, 110, 220), (8, 8, 40, 24))
    fast, small = _png(image, compress_level=1), _png(image, optimize=True)
    assert fast != small
    assert baseline.image_digest(fast) == baseline.image_digest(small)
    assert baseline.image_digest(fast).startswith("rgba:64x48:")
    image.putpixel((63, 47), (245, 245, 244))
    assert baseline.image_digest(_png(image)) != baseline.image_digest(fast)


def test_an_image_sha256_field_needs_the_image_registered():
    data = _png(Image.new("RGB", (40, 30), (200, 60, 40)))
    sha256 = hashlib.sha256(data).hexdigest()
    with pytest.raises(LookupError, match=sha256):
        baseline.Normalizer()({"screenshot_sha256": sha256})
    digest = baseline.image_digest(data)
    normalize = baseline.Normalizer()
    assert normalize({"screenshot_sha256": sha256}) == {"screenshot_sha256": digest}
    assert normalize({"image": {"data": {"byte_length": len(data), "sha256": sha256}}}) == {
        "image": {"data": digest}
    }
