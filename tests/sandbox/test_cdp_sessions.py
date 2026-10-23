"""Sessions on one page target (sandbox/browser/cdp.py client_for).

Under native streaming a capture runs alongside an action, and a CDP call's timeout spends the wait
for its session's lock. So the capture's DOM and AX probes use a session of their own on the same
target (decision N33): a probe never holds the lock the action's page barriers wait for. The
turn-based shell runs both on its one thread and keeps the one session.
"""

from __future__ import annotations

import threading

import pytest
from websockets.exceptions import ConnectionClosed

from realtime_framework.sandbox.browser import cdp
from realtime_framework.sandbox.observation.capture import BrowserSceneCapture

from .fakes import FakeCDP

ROOMY = 10.0


def test_a_named_session_has_a_connection_and_a_lock_of_its_own():
    """A probe holding the capture session's lock leaves the shared session's calls to answer in
    their own time, well within a timeout shorter than the probe."""
    with FakeCDP(values={"1": 1}) as fake:
        shared = cdp.client_for(fake.port, target_id=fake.target_id)
        capture = cdp.client_for(fake.port, target_id=fake.target_id, session="capture")
        assert capture is not shared
        assert cdp.client_for(fake.port, target_id=fake.target_id, session="capture") is capture
        held, release = threading.Event(), threading.Event()

        def probe():
            with capture.transaction():
                assert capture.evaluate("1", timeout=ROOMY) == 1
                held.set()
                release.wait(ROOMY)

        prober = threading.Thread(target=probe, daemon=True)
        prober.start()
        try:
            assert held.wait(ROOMY)
            assert shared.evaluate("1", timeout=0.5) == 1
        finally:
            release.set()
            prober.join(ROOMY)
            cdp.close_port(fake.port)
        assert fake.connections == 2


@pytest.mark.parametrize(
    "meet",
    [
        lambda client: client.evaluate("1", timeout=ROOMY),
        lambda client: client.wait_event("Page.loadEventFired", timeout=ROOMY),
    ],
    ids=["call", "event-wait"],
)
def test_a_connection_the_page_dropped_fails_one_call_and_the_next_call_reconnects(meet):
    """The session lasts the episode: were the closed connection kept, every later call on it
    would fail too."""
    with FakeCDP(values={"1": 1}) as fake:
        client = cdp.client_for(fake.port)
        try:
            assert client.evaluate("1", timeout=ROOMY) == 1
            fake.drop()
            with pytest.raises(ConnectionClosed):
                meet(client)
            assert client.evaluate("1", timeout=ROOMY) == 1
        finally:
            cdp.close_port(fake.port)
        assert fake.connections == 2


def test_a_client_asked_for_by_target_leaves_the_default_client_on_its_page():
    """The capture probes each page target by id (cdp.py _probe_one); the clock and the task's
    state reads ask for the default client, which must stay on the page it found."""
    with FakeCDP(values={"1": 1}) as fake:
        default = cdp.client_for(fake.port)
        try:
            cdp.client_for(fake.port, target_id="popup-2")
            assert cdp.client_for(fake.port) is default
        finally:
            cdp.close_port(fake.port)


def test_the_probes_of_a_capture_ask_for_its_session(monkeypatch):
    """The DOM probe and the AX tree read share the client of the capture's session; by default
    it is the shared one."""
    target = {"id": "page-1", "url": "http://localhost:8930/", "webSocketDebuggerUrl": "ws://x"}
    sessions = []

    class Client:
        def evaluate(self, expression, *, await_promise=True, timeout=120):
            return {"scene": {"status": "ok", "active": True}}

        def call(self, method, params=None, *, timeout=120, session_id=None):
            return {"result": {"windowId": 1}}

    def client_for(port, *, target_id=None, target_hint="", session=""):
        sessions.append(session)
        return Client()

    monkeypatch.setattr(cdp, "page_targets", lambda port: [target])
    monkeypatch.setattr(cdp, "client_for", client_for)
    scene = BrowserSceneCapture(guest=None, debug_port=9222)
    assert scene._probe()["scene"]["status"] == "ok"
    scene.cdp_session = "capture"
    assert scene._probe()["scene"]["status"] == "ok"
    assert sessions == ["", "capture"]
