"""WebSocket server and client connections that run on the fake clock (tests/fake_clock.py).

CassetteServer replays a cassette (tests/streaming/ws_cassette.py) on 127.0.0.1 with an
OS-assigned port. It runs websockets' sans-I/O ServerProtocol on threads the clock starts: one
accepts, one serves each connection. Every wait is a wait on the clock, as in
tests/replay/stub.py's ``_serve_on_clock``: a connection waits for its socket to turn readable, and
for no longer than the next timed record's time, so a timed message leaves when the clock reads its
``at``. Semantic time is the clock's monotonic time minus ``origin``. The server offers no
extension (the client then sends uncompressed frames) and no message size limit: client frames
carry base64 screenshots. Every message is sent as a text frame: the episode trace a cassette may
come from keeps a message's bytes, not its frame type. Once the cursor finds a problem, the
connection it found it on is closed with 1011 and the problem as its reason.

clocked_connection makes a websockets.sync client connection class wait on the clock instead. The
connection reads its socket on a thread websockets starts in its constructor
(websockets/sync/connection.py:130-136, 799-814), which takes no part in the clock, and recv()
blocks on the message queue that thread fills (websockets/sync/messages.py:72-92). On the fake
clock a thread blocked like that counts as busy, so time never moves while the client waits for a
message. The subclass does the reading on a thread the clock starts, which waits until its socket
turns readable, and recv() waits on the clock until a message is queued or the connection ends.
Reading never pauses: a paused reader would block on a lock until the client reads
(sync/connection.py:89-96, 811), which on the fake clock is busy too. Like the server, the client
waits until what it sent is delivered before it goes on.
"""

from __future__ import annotations

import json
import socket
import threading
import traceback

from websockets.frames import Opcode
from websockets.protocol import CLOSED, OPEN
from websockets.server import ServerProtocol

from tests.fake_clock import wait_delivered
from tests.replay import baseline

from .ws_cassette import Cursor

RECV_BYTES = 65536
CLIENT_END_WAIT_S = 10.0  # real seconds, like websockets' close timeout


class CassetteServer:
    def __init__(
        self, records: list[dict], clock, *, normalizer=None, tolerance_s=None, origin=0.0
    ):
        self._clock = clock
        self._origin = origin
        self.cursor = Cursor(records, normalizer or baseline.Normalizer(), tolerance_s=tolerance_s)
        # Every connection's thread reads and moves the one cursor.
        self._lock = threading.Lock()
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._closing = False
        self._threads: list[threading.Thread] = []
        self._sockets: list[socket.socket] = []
        # The request path of each upgrade, in order, and any error a connection thread raised.
        self.paths: list[str] = []
        self.errors: list[str] = []

    @property
    def port(self) -> int:
        return self._listener.getsockname()[1]

    def __enter__(self):
        self._threads.append(self._clock.start_thread(self._accept, name="cassette-accept"))
        return self

    def __exit__(self, *exc_info):
        self._closing = True
        socket.create_connection(("127.0.0.1", self.port)).close()  # wakes the accept thread
        for sock in self._sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)  # ends a connection the client left open
            except OSError:
                pass
        for thread in self._threads:
            thread.join(5.0)
        self._listener.close()

    def finish(self) -> list[str]:
        """The cursor's problems, with the records left unused, and any error a connection thread
        raised. Call it once the client is done."""
        with self._lock:
            return [*self.cursor.finish(), *self.errors]

    def _now(self) -> float:
        return self._clock.monotonic() - self._origin

    def _accept(self) -> None:
        while self._clock.wait_readable(self._listener) and not self._closing:
            sock, _ = self._listener.accept()
            if self._closing:
                sock.close()
                return
            self._sockets.append(sock)
            connection = _Connection(self, sock)
            self._threads.append(
                self._clock.start_thread(connection.serve, name="cassette-connection")
            )


class _Connection:
    """One accepted connection on its own thread: the handshake, then the cursor's records."""

    def __init__(self, server: CassetteServer, sock: socket.socket):
        self.server = server
        self.sock = sock
        self.protocol = ServerProtocol(max_size=None)
        self.number: int | None = None
        self.closing = False  # the server sent its close

    def serve(self) -> None:
        server = self.server
        try:
            if not self._handshake():
                return
            with server._lock:
                self.number = server.cursor.open(server._now())
            if self.number is None:
                self._fail()
            else:
                self._converse()
        except Exception:  # noqa: BLE001 - finish() reports it
            with server._lock:
                server.errors.append(traceback.format_exc())
        finally:
            self.sock.close()

    def _handshake(self) -> bool:
        protocol = self.protocol
        events = []
        while not events:
            if not self._read():
                return False
            events = protocol.events_received()
        (request,) = events
        protocol.send_response(protocol.accept(request))
        self._send()
        with self.server._lock:
            self.server.paths.append(request.path)
        return True

    def _converse(self) -> None:
        server, cursor = self.server, self.server.cursor
        while self.protocol.state is not CLOSED:
            with server._lock:
                self._release(server._now())
                failed, due = cursor.failed, cursor.next_due(self.number)
            if failed and not self.closing:
                return self._fail()
            self._send()
            timeout = None if due is None else max(0.0, due - server._now())
            if not server._clock.wait_readable(self.sock, timeout):
                continue  # the next timed record is due
            if not self._read() and self.protocol.close_rcvd is None and not self.closing:
                with server._lock:
                    cursor.client_closed(self.number, None, server._now())  # dropped, no close
            for frame in self.protocol.events_received():
                with server._lock:
                    self._received(frame, server._now())
            self._send()

    def _received(self, frame, now: float) -> None:
        cursor = self.server.cursor
        if frame.opcode is Opcode.TEXT:
            assert frame.fin, "the client fragmented a message"
            # What is due goes first: only a record not due yet can be ahead of this message.
            self._release(now)
            if not self.closing:
                cursor.receive(self.number, json.loads(frame.data), now)
        elif frame.opcode is Opcode.CLOSE and self.protocol.close_rcvd_then_sent:
            # A close the client started; one answering the server's close is not recorded.
            cursor.client_closed(self.number, self.protocol.close_rcvd.code, now)

    def _release(self, now: float) -> None:
        if self.closing or self.protocol.state is not OPEN:
            return
        for record in self.server.cursor.release(self.number, now):
            if record["dir"] == "s2c":
                self.protocol.send_text(json.dumps(record["msg"]).encode())
            else:
                self.protocol.send_close(record["msg"]["code"], record["msg"].get("reason", ""))
                self.closing = True
                return

    def _fail(self) -> None:
        """Close the connection over the first problem, with the problem as its reason."""
        with self.server._lock:
            problems = self.server.cursor.problems
            problem = problems[0] if problems else "unexpected connection"
        if self.protocol.state is OPEN:
            # A close reason holds at most 123 bytes.
            self.protocol.send_close(1011, problem.encode()[:120].decode(errors="ignore"))
            self.closing = True
        self._send()
        while self.protocol.state is not CLOSED and self._read():
            self.protocol.events_received()
            self._send()

    def _read(self) -> bool:
        """Feed what the socket holds next to the protocol; False at the end of the stream."""
        self.server._clock.wait_readable(self.sock)
        data = self.sock.recv(RECV_BYTES)
        if data:
            self.protocol.receive_data(data)
        else:
            self.protocol.receive_eof()
        return bool(data)

    def _send(self) -> None:
        ended = False
        for data in self.protocol.data_to_send():
            if data:
                self.sock.sendall(data)
            else:
                self.sock.shutdown(socket.SHUT_WR)
                ended = True
        # Nothing the server sent may be in flight while it waits on the clock again.
        wait_delivered(self.sock)
        if ended:
            self._await_client_end()

    def _await_client_end(self) -> None:
        """The server ended its side after the closing handshake; the client ends its own as soon
        as it reads that, with no time passing. Wait for it in real time, busy on the clock: the end
        of a stream in flight, unlike bytes, does not keep the fake clock still, so while it
        travels every participant could be waiting and the clock would stall."""
        self.sock.settimeout(CLIENT_END_WAIT_S)
        while self.sock.recv(RECV_BYTES):
            pass
        self.protocol.receive_eof()


def clocked_connection(base, clock):
    """``base``, a websockets.sync ClientConnection class, waiting on ``clock``. Pass it as
    ``create_connection`` to websockets.sync.client.connect."""

    class ClockedConnection(base):
        def __init__(self, sock, protocol, **kwargs):
            self._arrived = clock.event()
            super().__init__(_ClockedSocket(sock, clock), protocol, **{**kwargs, "max_queue": None})

        def recv_events(self) -> None:
            clock.start_thread(super().recv_events, name="clocked-ws-reader").join()

        def process_event(self, event) -> None:
            super().process_event(event)
            self._arrived.set()

        def close_socket(self) -> None:
            super().close_socket()
            self._arrived.set()

        def send_data(self) -> None:
            super().send_data()
            # The fake clock looks at its participants one at a time: bytes that reach the
            # server between its look at the server's socket and its look at this one's would
            # be seen in neither place. Delivered before anyone waits, they are in the first.
            wait_delivered(self.socket._sock)

        def recv(self, timeout=None, decode=None):
            assert timeout is None, "a recv timeout would run on real time"
            queued = self.recv_messages
            while queued.frames.empty() and not queued.closed:
                self._arrived.clear()
                # Cleared before the check: a frame queued after it sets the event again.
                if queued.frames.empty() and not queued.closed:
                    clock.wait(self._arrived, None)
            return super().recv(timeout, decode)

    return ClockedConnection


class _ClockedSocket:
    """A connection's socket whose ``recv`` first waits on the clock until it turns readable."""

    def __init__(self, sock: socket.socket, clock):
        self._sock = sock
        self._clock = clock

    def recv(self, size: int) -> bytes:
        if self._sock.fileno() >= 0:
            self._clock.wait_readable(self._sock)
        return self._sock.recv(size)

    def __getattr__(self, name):
        return getattr(self._sock, name)
