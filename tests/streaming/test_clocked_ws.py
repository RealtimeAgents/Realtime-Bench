"""CassetteServer and clocked_connection (tests/streaming/clocked_ws.py) on the fake clock.

A minimal client, websockets' synchronous client with the clocked connection class, replays a
hand-written cassette over a real socket on 127.0.0.1. Timed server messages arrive when the
fake clock reads their time, and the strict cursor's problems reach the client as a 1011 close.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.sync.client import ClientConnection, connect

from tests import fake_clock
from tests.fake_clock import FakeClock

from . import clocked_ws
from .clocked_ws import CassetteServer, clocked_connection


def _record(direction, msg=None, at=None, conn=1):
    return {"conn": conn, "dir": direction, "at": at, "msg": msg}


def _replay(records, client) -> list[str]:
    """Run ``client(connect, clock)`` against the cassette; return the server's problems."""
    clock = FakeClock()
    with CassetteServer(records, clock) as server:
        connection = clocked_connection(ClientConnection, clock)

        def open_connection():
            return connect(f"ws://127.0.0.1:{server.port}/session", create_connection=connection)

        with clock.participant():
            client(open_connection, clock)
    return server.finish()


def _send(ws, message: dict) -> None:
    ws.send(json.dumps(message))


def _recv(ws) -> dict:
    return json.loads(ws.recv())


def test_timed_messages_arrive_at_their_time_and_reactive_ones_at_once():
    records = [
        _record("open", at=0.0),
        _record("s2c", {"type": "session.created"}),
        _record("c2s", {"type": "hello"}, 1.0),
        _record("s2c", {"type": "reply", "n": 1}, 2.5),
        _record("close", {"by": "client", "code": 1000}, 3.0),
    ]
    arrivals = []

    def client(open_connection, clock):
        ws = open_connection()
        arrivals.append((_recv(ws)["type"], clock.monotonic()))
        clock.sleep(1.0)
        _send(ws, {"type": "hello"})
        arrivals.append((_recv(ws)["type"], clock.monotonic()))
        clock.sleep(0.5)
        ws.close()

    assert _replay(records, client) == []
    assert arrivals == [("session.created", 0.0), ("reply", 2.5)]


def test_a_server_close_is_sent_at_its_time_and_the_next_connection_follows():
    records = [
        _record("open", at=0.0),
        _record("close", {"by": "server", "code": 1011, "reason": "scripted"}, 2.0),
        _record("open", conn=2, at=2.5),
        _record("s2c", {"type": "session.created"}, conn=2),
    ]
    seen = []

    def client(open_connection, clock):
        ws = open_connection()
        with pytest.raises(ConnectionClosedError) as closed:
            ws.recv()
        seen.append((closed.value.rcvd.code, closed.value.rcvd.reason, clock.monotonic()))
        clock.sleep(0.5)
        ws = open_connection()
        seen.append((_recv(ws)["type"], clock.monotonic()))
        ws.close()

    assert _replay(records, client) == []
    assert seen == [(1011, "scripted", 2.0), ("session.created", 2.5)]


def _closed_over_a_problem(ws) -> str:
    with pytest.raises(ConnectionClosedError) as closed:
        ws.recv()
    assert closed.value.rcvd.code == 1011
    return closed.value.rcvd.reason


def test_a_message_sent_early_closes_the_connection():
    records = [
        _record("open", at=0.0),
        _record("c2s", {"type": "first"}, 0.0),
        _record("s2c", {"type": "answer"}, 5.0),
        _record("c2s", {"type": "second"}, 5.0),
    ]
    reasons = []

    def client(open_connection, clock):
        ws = open_connection()
        _send(ws, {"type": "first"})
        clock.sleep(3.0)
        _send(ws, {"type": "second"})
        reasons.append(_closed_over_a_problem(ws))

    problem = "conn 1 record 2: c2s second sent at t=3, before the recorded s2c answer at t=5"
    assert _replay(records, client) == [problem]
    assert reasons == [problem]


def test_a_message_that_differs_closes_the_connection():
    records = [_record("open", at=0.0), _record("c2s", {"type": "hello", "n": 1}, 0.0)]

    def client(open_connection, clock):
        ws = open_connection()
        _send(ws, {"type": "hello", "n": 2})
        _closed_over_a_problem(ws)

    assert _replay(records, client) == ["conn 1 record 1: c2s hello differs:\n  msg.n: 1 -> 2"]


def test_records_left_unused_are_reported_at_the_end():
    records = [
        _record("open", at=0.0),
        _record("c2s", {"type": "hello"}, 0.0),
        _record("open", at=1.0, conn=2),
    ]

    def client(open_connection, clock):
        ws = open_connection()
        _send(ws, {"type": "hello"})
        ws.close()

    assert _replay(records, client) == ["conn 2: 1 record(s) left unused, from record 2 (open)"]


def test_a_message_sent_at_another_time_than_the_scripts_is_a_problem():
    records = [_record("open", at=0.0), _record("c2s", {"type": "hello"}, 1.0)]

    def client_after(delay):
        def client(open_connection, clock):
            ws = open_connection()
            clock.sleep(delay)
            _send(ws, {"type": "hello"})
            ws.close()

        return client

    assert _replay(records, client_after(1.0)) == []
    assert _replay(records, client_after(1.5)) == [
        "conn 1 record 1: c2s hello at t=1.5, recorded at t=1"
    ]


def test_waiting_for_delivery_never_blocks_on_a_read(monkeypatch):
    """H1: while the client sends, websockets holds its protocol lock (sync/connection.py:1024) and
    the clocked reader may take the bytes off the socket between the look that finds them and the
    peek. A blocking peek then waits for bytes that never come, holding that lock."""
    sock, peer = socket.socketpair()
    unacknowledged = iter([1, 1])
    monkeypatch.setattr(fake_clock, "_unacknowledged", lambda fd: next(unacknowledged, 0))
    monkeypatch.setattr(fake_clock, "_fds_ready", lambda fds: True)  # the reader took the bytes
    done = threading.Event()

    def wait():
        clocked_ws.wait_delivered(sock)
        done.set()

    threading.Thread(target=wait, daemon=True).start()
    returned = done.wait(2.0)
    peer.close()  # ends a peek that blocked
    sock.close()
    assert returned, "the wait blocked on a read"
