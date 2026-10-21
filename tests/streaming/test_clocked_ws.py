"""CassetteServer and clocked_connection (tests/streaming/clocked_ws.py) on the fake clock.

A minimal client, websockets' synchronous client with the clocked connection class, replays a
hand-written cassette over a real socket on 127.0.0.1. Timed server messages arrive when the
fake clock reads their time, and the strict cursor's problems reach the client as a 1011 close.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.sync.client import ClientConnection, connect

from tests import fake_clock
from tests.fake_clock import FakeClock

from . import clocked_ws
from .clocked_ws import REAL_LIMIT_S, CassetteServer, clocked_connection, real_time_limit
from .ws_cassette import LEFT_OPEN


def _record(direction, msg=None, at=None, conn=1):
    return {"conn": conn, "dir": direction, "at": at, "msg": msg}


def _replay(records, client, *, connection=None, real_limit_s=REAL_LIMIT_S) -> list[str]:
    """Run ``client(connect, clock)`` against the cassette, on the clocked connection class unless
    ``connection`` is given; return the server's problems."""
    clock = FakeClock()
    with CassetteServer(records, clock) as server:
        connection = connection or clocked_connection(ClientConnection, clock)

        def open_connection():
            return connect(f"ws://127.0.0.1:{server.port}/session", create_connection=connection)

        def progress():
            return clock.monotonic(), server.cursor.left

        with real_time_limit(real_limit_s, server.drop, progress), clock.participant():
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


def test_the_server_knows_a_pushed_frames_and_a_cancels_event_ids():
    """The server's default normalizer is ws_cassette.RealtimeNormalizer: these event ids are this
    session's random ones, which the recording has as placeholders."""
    records = [
        _record("open", at=0.0),
        _record("c2s", {"type": "conversation.item.create", "event_id": "<rtb_push-1>"}, 0.0),
        _record("c2s", {"type": "response.cancel", "event_id": "<rtb_cancel-1>"}, 0.0),
    ]

    def client(open_connection, clock):
        ws = open_connection()
        _send(ws, {"type": "conversation.item.create", "event_id": "rtb_push_9f8e"})
        _send(ws, {"type": "response.cancel", "event_id": "rtb_cancel_0a1b"})
        ws.close()

    assert _replay(records, client) == []


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


def test_a_client_stuck_in_real_time_fails_instead_of_hanging():
    """websockets' own connection class waits for the timed message in real time, so the fake
    clock never reaches it; the real-time limit breaks the connections up and fails the replay,
    though the client takes the end of its connection in its stride."""
    records = [_record("open", at=0.0), _record("s2c", {"type": "later"}, 1.0)]

    def client(open_connection, clock):
        with pytest.raises(ConnectionClosedError):
            open_connection().recv()

    with pytest.raises(AssertionError, match="made no progress for 1 real second"):
        _replay(records, client, connection=ClientConnection, real_limit_s=1.0)


def test_a_replay_that_goes_on_making_progress_is_not_stuck():
    """L6: the limit counts the real time since the last progress, the clock's or the cursor's,
    so a long replay, slow in real time as each round on Linux's loopback is, goes on."""
    rounds = 6
    records = [_record("open", at=0.0)]
    for n in range(rounds):
        records += [_record("c2s", {"type": "ping", "n": n}, 0.0), _record("s2c", {"n": n})]

    def client(open_connection, clock):
        ws = open_connection()
        for n in range(rounds):
            time.sleep(0.2)
            _send(ws, {"type": "ping", "n": n})
            assert _recv(ws) == {"n": n}
        ws.close()

    assert _replay(records, client, real_limit_s=0.6) == []


def test_a_connection_the_server_ends_twice_raises_nothing():
    """L10: the server's exit ends a connection the client left open (shutdown), and the closing
    handshake that follows ends it again, which macOS refuses with ENOTCONN."""
    records = [_record("open", at=0.0), _record("c2s", {"type": "hello"}, 0.0)]

    def client(open_connection, clock):
        _send(open_connection(), {"type": "hello"})  # and never closed

    assert not [problem for problem in _replay(records, client) if "OSError" in problem]


def test_a_connection_the_client_leaves_open_is_a_problem():
    records = [_record("open", at=0.0), _record("c2s", {"type": "hello"}, 0.0)]

    def client(open_connection, clock):
        _send(open_connection(), {"type": "hello"})  # and never closed

    assert _replay(records, client) == [LEFT_OPEN.format(conn=1)]


def test_a_message_due_when_the_client_closes_goes_first():
    """L1: the server releases what is due before it reads what the client did, a close too, so a
    close at the time a message comes due always follows that message, as the script has it."""
    records = [
        _record("open", at=0.0),
        _record("s2c", {"type": "last"}, 2.0),
        _record("close", {"by": "client", "code": 1000}, 2.0),
    ]

    def client(open_connection, clock):
        ws = open_connection()
        clock.sleep(2.0)
        ws.close()

    assert [_replay(records, client) for _ in range(10)] == [[]] * 10


@pytest.mark.parametrize(
    "conn_1_next", [_record("c2s", {"type": "idle"}, 9.0), _record("s2c", {"type": "later"}, 5.0)]
)
def test_a_problem_on_one_connection_ends_the_others_at_once(conn_1_next):
    """L3: the connection with the problem is closed with 1011 and the problem as its reason; every
    other connection is dropped then too, whether it waits for the client or for a time."""
    records = [
        _record("open", at=0.0),
        _record("open", at=0.0, conn=2),
        conn_1_next,
        _record("c2s", {"type": "hello", "n": 1}, 1.0, conn=2),
    ]
    ended = []

    def client(open_connection, clock):
        ws_1, ws_2 = open_connection(), open_connection()
        clock.sleep(1.0)
        _send(ws_2, {"type": "hello", "n": 2})
        for ws in (ws_2, ws_1):
            with pytest.raises(ConnectionClosedError) as closed:
                ws.recv()
            ended.append((closed.value.rcvd and closed.value.rcvd.code, clock.monotonic()))

    assert _replay(records, client) == ["conn 2 record 3: c2s hello differs:\n  msg.n: 1 -> 2"]
    assert ended == [(1011, 1.0), (None, 1.0)]


def test_the_connection_with_the_problem_closes_it_though_another_finds_it_first(monkeypatch):
    """L10(d): connection 1 finds the problem connection 2 had while connection 2 is still on its
    way to drop the others; connection 2's client still gets the 1011. The drops wait for each
    other, so two connections that both set out to drop the others would drop each other."""
    drop, both = CassetteServer.drop, threading.Barrier(2, timeout=0.5)

    def drop_together(self, but=None):
        try:
            both.wait()
        except threading.BrokenBarrierError:
            pass  # only one set out
        drop(self, but)

    monkeypatch.setattr(CassetteServer, "drop", drop_together)
    records = [
        _record("open", at=0.0),
        _record("open", at=0.0, conn=2),
        _record("c2s", {"type": "hello", "n": 1}, 1.0, conn=2),
        _record("c2s", {"type": "idle"}, 1.0),
    ]
    ended = []

    def client(open_connection, clock):
        ws_1, ws_2 = open_connection(), open_connection()
        clock.sleep(1.0)
        _send(ws_2, {"type": "hello", "n": 2})
        _send(ws_1, {"type": "idle"})
        for ws in (ws_2, ws_1):
            with pytest.raises(ConnectionClosedError) as closed:
                ws.recv()
            ended.append(closed.value.rcvd and closed.value.rcvd.code)

    assert _replay(records, client) == ["conn 2 record 2: c2s hello differs:\n  msg.n: 1 -> 2"]
    assert ended == [1011, None]


def test_a_client_that_resets_its_connection_closes_it_without_a_close_frame():
    """L10(c): a connection the client resets ends its stream like one it shuts down."""
    records = [
        _record("open", at=0.0),
        _record("c2s", {"type": "hello"}, 0.0),
        _record("s2c", {"type": "ack"}),
        _record("close", {"by": "client", "code": None}, 0.0),
    ]

    def client(open_connection, clock):
        ws = open_connection()
        _send(ws, {"type": "hello"})
        assert _recv(ws) == {"type": "ack"}
        ws.socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        ws.close_socket()

    assert [_replay(records, client) for _ in range(10)] == [[]] * 10


def test_a_reset_while_the_server_awaits_the_clients_end_is_that_end():
    """L10(c): websockets' close_socket shuts the client's side down before it closes the socket,
    so with SO_LINGER(1, 0) the reset can come after the end of the stream, while the server waits
    for the client's end."""
    listener = socket.create_server(("127.0.0.1", 0))
    peer = socket.create_connection(listener.getsockname())
    sock, _ = listener.accept()
    listener.close()
    peer.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    peer.close()
    try:
        clocked_ws._Connection(None, sock)._await_client_end()
    finally:
        sock.close()


def test_a_binary_message_is_a_problem():
    """L10(f): a recording holds text messages only."""
    records = [_record("open", at=0.0)]

    def client(open_connection, clock):
        ws = open_connection()
        ws.send(b"\x00")
        with pytest.raises(ConnectionClosedError):
            ws.recv()

    assert _replay(records, client) == ["conn 1: the client sent a binary message"]


def test_messages_the_client_leaves_unread_never_stop_its_reader():
    """A reader that paused at websockets' default queue of 16 messages would block on a lock,
    which on the fake clock is busy, and time would never reach the client's next send."""
    replies = [_record("s2c", {"type": "reply", "n": n}) for n in range(20)]
    records = [_record("open", at=0.0), *replies, _record("c2s", {"type": "done"}, 1.0)]
    read = []

    def client(open_connection, clock):
        ws = open_connection()
        clock.sleep(1.0)  # none read meanwhile
        _send(ws, {"type": "done"})
        read.extend(_recv(ws)["n"] for _ in replies)
        ws.close()

    assert _replay(records, client, real_limit_s=10.0) == []
    assert read == list(range(20))


def test_big_messages_both_ways_at_once_never_deadlock():
    """The client sends a big frame while the server's big replies to the message before it are
    still under way, as screenshots go up while audio comes down. Each side waits until the
    other's kernel holds what it wrote; were neither to read meanwhile, both would wait forever:
    the client's reader needs the protocol lock its sender holds. The sizes are past what macOS
    buffers on the loopback; Linux deadlocks already at a tenth of them."""
    replies = [_record("s2c", {"type": "audio", "n": n, "data": "a" * 1_000_000}) for n in range(5)]
    big = {"type": "frame", "data": "f" * 5_000_000}
    records = [
        _record("open", at=0.0),
        _record("c2s", {"type": "hello"}, 0.0),
        *replies,
        _record("c2s", big, 0.0),
    ]
    read = []

    def client(open_connection, clock):
        ws = open_connection()
        _send(ws, {"type": "hello"})
        _send(ws, big)
        read.extend(_recv(ws)["n"] for _ in replies)
        ws.close()

    assert _replay(records, client, real_limit_s=10.0) == []
    assert read == list(range(5))


def test_messages_that_arrive_together_each_follow_what_the_one_before_released():
    """Two messages in one read: the reply the first one released goes before the second one is
    compared, or the second would find that reply still next."""
    records = [
        _record("open", at=0.0),
        _record("c2s", {"type": "a"}, 0.0),
        _record("s2c", {"type": "ack"}),
        _record("c2s", {"type": "b"}, 0.0),
    ]

    def client(open_connection, clock):
        ws = open_connection()
        with ws.protocol_mutex:  # both frames go out in one write
            for kind in ("a", "b"):
                ws.protocol.send_text(json.dumps({"type": kind}).encode())
            ws.send_data()
        assert _recv(ws) == {"type": "ack"}
        ws.close()

    assert _replay(records, client) == []
