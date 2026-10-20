"""Settings moved from module-level environment reads into config fields keep their values."""

from __future__ import annotations

import pytest

from realtime_framework.entity.config import FrameEncodingConfig, RunnerTimingConfig


def replaced_reads(environ):
    """The reads the config fields replaced (sandbox/observation/capture.py:317-319 and
    evaluation/runner.py:49-60 at 8319a10)."""
    return {
        "max_dimension": int(environ.get("FRAME_MAX_DIM", "768")),
        "jpeg_quality": int(environ.get("FRAME_JPEG_QUALITY", "85")),
        "image_detail": environ.get("FRAME_IMAGE_DETAIL", "auto"),
        "deadline_grace_s": float(environ.get("STREAM_DEADLINE_GRACE_S", "2.0")),
        "abandoned_drain_s": float(environ.get("STREAM_ABANDONED_DRAIN_S", "30.0")),
        "observation_freeze_n": max(2, int(environ.get("STREAM_OBSERVATION_FREEZE_N", "3"))),
        "provider_watchdog_s": float(environ.get("STREAM_PROVIDER_WATCHDOG_S", "300")),
    }


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {
            "FRAME_MAX_DIM": "512",
            "FRAME_JPEG_QUALITY": "70",
            "FRAME_IMAGE_DETAIL": "high",
            "STREAM_DEADLINE_GRACE_S": "1.5",
            "STREAM_ABANDONED_DRAIN_S": "10",
            "STREAM_OBSERVATION_FREEZE_N": "1",
            "STREAM_PROVIDER_WATCHDOG_S": "0",
        },
    ],
    ids=["unset", "set"],
)
def test_fields_match_the_replaced_reads(environ):
    frame = FrameEncodingConfig.from_environ(environ)
    timing = RunnerTimingConfig.from_environ(environ)
    values = {**vars(frame), **vars(timing)}
    expected = replaced_reads(environ)
    assert values == expected
    assert [type(value) for value in values.values()] == [
        type(value) for value in expected.values()
    ]
