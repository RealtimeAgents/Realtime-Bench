"""A CDP call has one deadline for the lock, the connect and every reply (sandbox/browser/cdp.py).

Each test drives the real PersistentCDPClient against FakeCDP. Before, the lock wait was not timed,
the target lookup and the handshake had their own 3 s and 5 s caps outside the call's timeout, and
every message that was not the reply restarted the receive timeout, so the flood case below never
returned.
"""

from __future__ import annotations

import threading
import time

import pytest

from realtime_framework.sandbox.browser.cdp import PersistentCDPClient

from .fakes import FakeCDP

TIMEOUT = 0.5
# A failing call must end by its timeout plus this much scheduling slack, still before the 3 s and
# 5 s caps the old connect had.
SLACK = 1.5
# The timeout of calls that must succeed, far above how long they take.
ROOMY = 10.0


def _client(fake: FakeCDP) -> PersistentCDPClient:
    return PersistentCDPClient(fake.port, target_id=fake.target_id)


def _times_out(call) -> float:
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        call()
    elapsed = time.monotonic() - started
    assert TIMEOUT - 0.05 <= elapsed < TIMEOUT + SLACK, elapsed
    return elapsed


def _evaluate(client):
    return lambda: client.call("Runtime.evaluate", {"expression": "1"}, timeout=TIMEOUT)


def test_a_reply_after_unrelated_events_returns_and_keeps_the_events():
    with FakeCDP(values={"1+1": 2}) as fake:
        client = _client(fake)
        assert client.evaluate("1+1", timeout=ROOMY) == 2
        assert client.wait_event("Page.frameNavigated", timeout=ROOMY)["method"] == (
            "Page.frameNavigated"
        )
        client.close()


def test_unrelated_events_do_not_restart_the_timeout():
    with FakeCDP(mode="flood") as fake:
        client = _client(fake)
        _times_out(_evaluate(client))
        assert [command["method"] for command in fake.commands] == ["Runtime.evaluate"]


def test_waiting_for_an_event_ignores_other_events_for_one_timeout():
    with FakeCDP(mode="flood") as fake:
        client = _client(fake)
        _times_out(lambda: client.wait_event("Page.loadEventFired", timeout=TIMEOUT))


def test_waiting_for_the_lock_spends_the_timeout():
    with FakeCDP(values={"1": 1}) as fake:
        client = _client(fake)
        held, release = threading.Event(), threading.Event()

        def hold():
            with client.transaction():
                held.set()
                release.wait(10)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        held.wait(5)
        try:
            _times_out(_evaluate(client))
            assert fake.commands == []
        finally:
            release.set()
            holder.join(5)
        assert client.evaluate("1", timeout=ROOMY) == 1
        client.close()


def test_a_slow_target_lookup_spends_the_timeout():
    with FakeCDP(json_delay=5) as fake:
        _times_out(_evaluate(_client(fake)))
        assert fake.connections == 0


def test_a_slow_handshake_spends_the_timeout():
    with FakeCDP(handshake_delay=5) as fake:
        _times_out(_evaluate(_client(fake)))
        assert fake.json_requests == 1
        assert fake.commands == []


def test_the_lookup_and_the_handshake_share_one_timeout():
    # Each wait alone is below its own cap (3 s, 5 s) and below the call's timeout.
    with FakeCDP(json_delay=0.3, handshake_delay=0.3) as fake:
        _times_out(_evaluate(_client(fake)))
        assert fake.commands == []
