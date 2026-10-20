"""The episode clocks (realtime_framework/clock.py)."""

from __future__ import annotations

import asyncio
import threading

from realtime_framework.clock import FrameClock, SystemClock, episode_clocks


def test_episode_clocks_share_one_clock_on_the_wall_clock():
    clocks = episode_clocks("wall", 1.0)
    assert isinstance(clocks.semantic, SystemClock)
    assert clocks.semantic is clocks.operational


def test_episode_clocks_read_frames_under_the_token_clock():
    clocks = episode_clocks("token", 2.0)
    assert isinstance(clocks.semantic, FrameClock)
    assert isinstance(clocks.operational, SystemClock)
    clocks.semantic.frame = 3
    assert clocks.semantic.monotonic() == 1.5


def test_system_clock_waits_for_an_event_or_a_future():
    clock = SystemClock()
    event = clock.event()
    assert clock.wait(event, 0.01) is False
    clock.timer(0.01, event.set)
    assert clock.wait(event, 5.0) is True
    loop = _running_loop(clock)
    future = asyncio.run_coroutine_threadsafe(clock.sleep_async(0.01), loop)
    assert clock.wait(future, 5.0) is True
    loop.call_soon_threadsafe(loop.stop)


def _running_loop(clock):
    loop = clock.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop
