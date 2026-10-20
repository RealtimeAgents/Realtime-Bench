"""Local stub servers on 127.0.0.1 that keep every raw message and answer from a script.

LiteLLM sends OpenRouter and Anthropic requests through its aiohttp transport, which an httpx
mock cannot see, so every HTTP path is tested against a real socket.
"""

from __future__ import annotations

import datetime
import ipaddress
import json
import ssl
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from websockets.sync.server import serve


@dataclass(frozen=True)
class Received:
    method: str
    path: str
    headers: list[tuple[str, str]]
    body: bytes


@dataclass(frozen=True)
class Answer:
    status: int
    body: bytes
    content_type: str = "application/json"


DROP = "drop"  # close the connection without answering


class HttpStub:
    """``answer(index, received)`` returns an Answer or DROP for each request in turn."""

    def __init__(self, answer: Callable[[int, Received], Answer | str]):
        self._answer = answer
        self.requests: list[Received] = []
        self.served: list[bytes | None] = []
        self._server = HTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def _handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received = Received("POST", self.path, list(self.headers.items()), body)
                stub.requests.append(received)
                answer = stub._answer(len(stub.requests) - 1, received)
                if answer == DROP:
                    stub.served.append(None)
                    self.close_connection = True
                    return
                stub.served.append(answer.body)
                self.send_response(answer.status)
                self.send_header("Content-Type", answer.content_type)
                self.send_header("Content-Length", str(len(answer.body)))
                self.end_headers()
                self.wfile.write(answer.body)

            def log_message(self, *args):
                pass

        return Handler


@dataclass
class WsConnection:
    """One accepted WebSocket connection: frames in arrival order, both directions."""

    received: list[str | bytes] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)


class WsStub:
    """``handle(stub, connection, message)`` returns the JSON messages to send back.

    ``reject`` refuses the opening handshake of the first N connections with HTTP 503.
    """

    def __init__(self, handle, *, reject: int = 0, tls: ssl.SSLContext | None = None):
        self._handle = handle
        self._reject = reject
        self.rejected = 0
        self.connections: list[WsConnection] = []
        self._server = serve(
            self._serve, "127.0.0.1", 0, process_request=self._process_request, ssl=tls
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._server.socket.getsockname()[1]

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._thread.join()

    def _process_request(self, connection, request):
        if self.rejected < self._reject:
            self.rejected += 1
            return connection.respond(503, "busy\n")
        return None

    def _serve(self, websocket):
        connection = WsConnection()
        self.connections.append(connection)
        self._send(websocket, connection, self._handle(self, connection, None))
        for message in websocket:
            connection.received.append(message)
            reply = self._handle(self, connection, json.loads(message))
            if reply is None:
                websocket.close(1011, "stub closed the session")
                return
            self._send(websocket, connection, reply)

    @staticmethod
    def _send(websocket, connection, messages):
        for message in messages or ():
            text = json.dumps(message)
            connection.sent.append(text)
            websocket.send(text)


def self_signed_tls(directory: Path) -> tuple[ssl.SSLContext, Path]:
    """A server context for 127.0.0.1 and the certificate file a client trusts."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path = directory / "stub-cert.pem"
    key_path = directory / "stub-key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    return context, cert_path
