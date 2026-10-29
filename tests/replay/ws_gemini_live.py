"""Scripted Gemini Live endpoint: a TLS WebSocket server on 127.0.0.1 with an OS-assigned port.

The gemini_live driver opens every session with google-genai's live client
(providers/gemini/live.py GeminiLiveDriver._client_for_slot, GeminiLiveDriver._connect). With an
API key the client connects to
``wss://<base>/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent``
(google/genai/live.py:963-996, 1117-1119), always over TLS: the base URL's scheme is replaced by
wss (google/genai/_api_client.py:1297-1304). The catalog configs carry no api_base_url, so the
tests reach this server through GOOGLE_GEMINI_BASE_URL (google/genai/_base_url.py:44-50) and
trust its certificate through SSL_CERT_FILE, which the client reads when the driver builds it
(google/genai/_api_client.py:897-900, 1254-1258).

Each connection runs BidiGenerateContent the way the driver uses it: one ``setup`` answered by
``setupComplete`` (google/genai/live.py:1120-1161), a seeded history when the setup asks for one
(live.py GeminiLiveDriver._open), then one scripted reply per observation, consumed in order across
connections. Every frame is kept in arrival order with its connection number. The server also
checks what the protocol makes it acknowledge: the setup of every reconnect, the seeded history
against the tool exchanges it acknowledged before, and each queued tool acknowledgement.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import hashlib
import ipaddress
import json
import ssl
import threading
from pathlib import Path

# cryptography comes with google-genai, which requires google-auth (uv.lock:864-870, 846-851).
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from realtime_framework.entity.config import DEFAULT_ACTIVE_VISUAL_BUDGET
from realtime_framework.entity.model import ARCHIVED_ACTION_LEDGER_HEADER

from . import baseline

BIDI_PATH = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
# The driver acknowledges every call as queued, before the runner executes it
# (entity/context_policy.py GEMINI_QUEUED_TOOL_RESPONSE; providers/gemini/live.py
# GeminiLiveDriver._events).
QUEUED_RESPONSE = {
    "status": "queued_not_executed",
    "instruction": (
        "End this turn silently. Await the next observation for the actual execution result."
    ),
}
# The tool_transcript history replays past exchanges as one model text each (gemini/session.py
# SessionCore._tool_contents; harnesses/default/request.py _turn_items).
TRANSCRIPT_HEADER = (
    "Historical GUI tool exchanges, already acknowledged "
    "(current actions must use native functions):\n"
)
SEED_CONTENTS_PER_MESSAGE = 12  # gemini/session.py HISTORY_CONTENTS_PER_MESSAGE, history_batches


def realtime_input_order(feedback: int = 0) -> tuple[str, ...]:
    """The messages realtime_input sends one observation as (live.py send_observation): a text
    for each feedback the session does not hold yet, before the frame's."""
    return ("activityStart", "video", *("text",) * feedback, "text", "audio", "activityEnd")


# Answers an observation that has no scripted reply. The driver classifies 1008 with this reason
# as FATAL (live.py GeminiLiveDriver.classify_transport_error), which the shared policy never
# retries (entity/model.py TransportFailure.retryable, TransportRetryPolicy.delay_s), so the
# decision fails at once instead of waiting for a reply.
UNSCRIPTED_CLOSE = (1008, "observation beyond the script")

AWAIT_ACK = object()  # script step: read the acknowledgement of every call sent so far


def interrupted_before_output(prompt_tokens: int) -> list[dict]:
    """A reply the server interrupts before it generates anything: ``interrupted``, then the
    turnComplete that ends the turn, without generationComplete (google/genai/types.py
    LiveServerContent.generation_complete). The turnComplete carries the usage of the prompt
    alone, the terminal usage the driver reads from that message (providers/gemini/live.py
    GeminiLiveDriver._events); protobuf JSON omits the zero response count, which the driver
    reads as zero (live.py GeminiLiveDriver._usage_dict)."""
    usage = {"promptTokenCount": prompt_tokens, "totalTokenCount": prompt_tokens}
    return [
        {"serverContent": {"interrupted": True}},
        {"serverContent": {"turnComplete": True}, "usageMetadata": usage},
    ]


def self_signed_certificate(directory: Path) -> tuple[Path, Path]:
    """Write a certificate for IP 127.0.0.1 and its key, valid for one day from now."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "gemini-live-replay-stub")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        # The client checks the server's IP address against the certificate.
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    directory.mkdir(parents=True, exist_ok=True)
    cert_file = directory / "stub-cert.pem"
    key_file = directory / "stub-key.pem"
    cert_file.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_file, key_file


class LiveStub:
    """BidiGenerateContent server with scripted replies.

    ``turns`` holds one reply per observation. A reply is a list of server frames, with
    AWAIT_ACK where the client must acknowledge the calls sent before it. ``history_format`` and
    ``ack_options`` are the client's Live options the checks depend on (live.py
    LIVE_PROTOCOL_OPTIONS, LIVE_HISTORY_OPTIONS; gemini/session.py
    SessionCore.tool_response_options). ``setup_closes`` maps a resumption handle to the close code
    and reason that answer a setup resuming with it, ``connection_closes`` a connection number to
    those that answer its setup.
    """

    def __init__(
        self,
        turns: list[list],
        *,
        cert_dir: Path,
        api_key: str,
        history_format: str,
        ack_options: dict,
        setup_closes: dict[str, tuple[int, str]] | None = None,
        connection_closes: dict[int, tuple[int, str]] | None = None,
    ):
        self._turns = [list(turn) for turn in turns]
        self._api_key = api_key
        self._history_format = history_format
        self._ack_options = dict(ack_options)
        self._setup_closes = dict(setup_closes or {})
        self._connection_closes = dict(connection_closes or {})
        self.cert_file, self._key_file = self_signed_certificate(cert_dir)
        self.events: list[dict] = []  # {"conn", "dir", "msg"} in arrival order
        self.violations: list[str] = []
        self.unscripted = 0
        self._exchanges: list[dict] = []  # every acknowledged tool exchange, in order
        self._resume_handle: str | None = None  # the handle a resume may use after a turn
        self._first_setup: dict | None = None
        self._connections = 0
        self._states: dict[int, str] = {}
        self._condition = threading.Condition()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._server = None

    @property
    def base_url(self) -> str:
        return f"https://127.0.0.1:{self._server.sockets[0].getsockname()[1]}"

    @property
    def unused_turns(self) -> int:
        return len(self._turns)

    def __enter__(self):
        self._thread.start()
        self._server = self._run(self._start())
        return self

    def __exit__(self, *exc_info):
        self._run(self._stop())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    def _run(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout=30)

    async def _start(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.cert_file, self._key_file)
        # A seeded history carries up to 12 contents with screenshots in one message, above the
        # default 1 MiB frame limit.
        return await serve(
            self._serve, "127.0.0.1", 0, ssl=context, max_size=None, ping_interval=None
        )

    async def _stop(self):
        # The client has closed every session by now; wait for the handlers to record it.
        self._server.close(close_connections=False)
        await self._server.wait_closed()

    def wait_until_idle(self, timeout: float = 30.0) -> None:
        """Block until the newest connection waits for the next observation, or has ended.

        The driver returns from a decision once it has sent its last frame, which can still be in
        flight, for example the seed of a history replayed after the turn (live.py
        GeminiLiveDriver._open; harnesses/default/context.py RealtimeRuntime._run_turn). Waiting
        here assigns every frame to the decision that sent it. A decision ends on a closed
        connection when the stub closed it for an observation beyond the script.
        """
        with self._condition:
            idle = self._condition.wait_for(
                lambda: self._states.get(self._connections) in ("idle", "closed"), timeout
            )
        if not idle:
            raise TimeoutError(f"the stub did not become idle: {self._states}")

    def _set_state(self, conn: int, state: str) -> None:
        with self._condition:
            self._states[conn] = state
            self._condition.notify_all()

    def _record(self, conn: int, direction: str, message: dict) -> None:
        self.events.append({"conn": conn, "dir": direction, "msg": message})

    def _check(self, ok: bool, problem: str) -> None:
        if not ok:
            self.violations.append(problem)

    async def _recv(self, ws, conn: int) -> dict:
        message = json.loads(await ws.recv())
        self._record(conn, "c2s", message)
        return message

    async def _send(self, ws, conn: int, frame: dict) -> None:
        self._record(conn, "s2c", frame)
        await ws.send(json.dumps(frame))

    async def _close(self, ws, conn: int, code: int, reason: str) -> None:
        self._record(conn, "s2c_close", {"code": code, "reason": reason})
        await ws.close(code, reason)

    async def _serve(self, ws) -> None:
        with self._condition:
            self._connections += 1
            conn = self._connections
        request = ws.request
        key = request.headers.get("x-goog-api-key")  # google/genai/_api_client.py:842-846
        self._record(conn, "open", {"path": request.path, "x-goog-api-key": key})
        self._check(request.path == BIDI_PATH, f"conn {conn}: path {request.path!r}")
        self._check(key == self._api_key, f"conn {conn}: API key {key!r}")
        try:
            await self._session(ws, conn)
        except ConnectionClosed as closed:
            # The driver closes every session it replaces or ends with a close frame
            # (live.py GeminiLiveDriver._close_current).
            self._record(
                conn, "c2s_close", {"code": closed.rcvd.code, "reason": closed.rcvd.reason}
            )
        finally:
            self._set_state(conn, "closed")

    async def _session(self, ws, conn: int) -> None:
        self._set_state(conn, "setup")
        setup = (await self._recv(ws, conn))["setup"]
        if not await self._accept_setup(ws, conn, setup):
            return
        await self._send(ws, conn, {"setupComplete": {}})
        images = await self._seed(ws, conn) if "historyConfig" in setup else 0
        while True:
            self._set_state(conn, "idle")
            await self._observation(ws, conn)
            # The session keeps every screenshot it was sent: the seed's and one per observation.
            images += 1
            self._check(
                images <= DEFAULT_ACTIVE_VISUAL_BUDGET,
                f"conn {conn}: {images} screenshots in one session",
            )
            if not self._turns:
                self.unscripted += 1
                await self._close(ws, conn, *UNSCRIPTED_CLOSE)
                return
            await self._play(ws, conn, self._turns.pop(0))

    async def _accept_setup(self, ws, conn: int, setup: dict) -> bool:
        """Check a setup before acknowledging it; False when the script closes it instead."""
        handle = setup["sessionResumption"].get("handle")
        if handle is not None:
            # A resume carries the handle promoted by the last clean turn and no history
            # (live.py GeminiLiveDriver._connect, GeminiLiveDriver._sync_history).
            self._check(handle == self._resume_handle, f"conn {conn}: resumes with {handle!r}")
            self._check("historyConfig" not in setup, f"conn {conn}: a resume seeds history")
        else:
            # Every later session replays the history it replaces (live.py
            # GeminiLiveDriver._connect, GeminiLiveDriver._sync_history).
            self._check(
                ("historyConfig" in setup) == (conn > 1),
                f"conn {conn}: historyConfig {setup.get('historyConfig')!r}",
            )
        # Everything else is the same LiveConnectConfig on every connect (live.py
        # GeminiLiveDriver._live_config).
        session = {
            name: value
            for name, value in setup.items()
            if name not in ("historyConfig", "sessionResumption")
        }
        if self._first_setup is None:
            self._first_setup = session
        self._check(session == self._first_setup, f"conn {conn}: setup differs from conn 1")
        if handle in self._setup_closes:
            await self._close(ws, conn, *self._setup_closes[handle])
            return False
        if conn in self._connection_closes:
            await self._close(ws, conn, *self._connection_closes[conn])
            return False
        return True

    async def _seed(self, ws, conn: int) -> int:
        """Read the seeded history: client_content messages, the last one completing the turn
        (live.py GeminiLiveDriver._open). Returns the number of screenshots in it."""
        self._set_state(conn, "seeding")
        contents = []
        while True:
            content = (await self._recv(ws, conn))["client_content"]
            self._check(
                len(content["turns"]) <= SEED_CONTENTS_PER_MESSAGE,
                f"conn {conn}: {len(content['turns'])} contents in one seed message",
            )
            contents += content["turns"]
            if content["turnComplete"]:
                break
        texts = [content["parts"][0].get("text", "") for content in contents]
        ledgers = [
            i for i, text in enumerate(texts) if text.startswith(ARCHIVED_ACTION_LEDGER_HEADER)
        ]
        # The archived-action ledger is the first content, from the user (gemini/session.py
        # SessionCore._seed).
        self._check(
            ledgers in ([], [0]) and all(contents[i]["role"] == "user" for i in ledgers),
            f"conn {conn}: ledger contents at {ledgers}",
        )
        # The retained turns are the newest ones, so the seed must hold the newest acknowledged
        # exchanges, exactly as acknowledged.
        exchanges = self._seeded_exchanges(conn, contents)
        latest = self._exchanges[len(self._exchanges) - len(exchanges) :]
        self._check(
            bool(exchanges) == bool(self._exchanges) and exchanges == latest,
            f"conn {conn}: the seed does not replay the latest acknowledged tool exchanges",
        )
        return sum("inlineData" in part for content in contents for part in content["parts"])

    def _seeded_exchanges(self, conn: int, contents: list[dict]) -> list[dict]:
        """The tool exchanges a seed replays, in the form _play records them."""
        parts = [part for content in contents for part in content["parts"]]
        if self._history_format == "tool_transcript":
            # Past calls are model text, never function parts (gemini/session.py
            # SessionCore._tool_contents).
            self._check(
                not any("functionCall" in part or "functionResponse" in part for part in parts),
                f"conn {conn}: function parts in a tool_transcript seed",
            )
            exchanges = []
            for content in contents:
                text = content["parts"][0].get("text", "")
                if content["role"] == "model" and text.startswith(TRANSCRIPT_HEADER):
                    exchanges += json.loads(text[len(TRANSCRIPT_HEADER) :])
            return exchanges
        # native_tools: a model content of function calls, then a user content of their
        # responses (gemini/session.py SessionCore._tool_contents).
        exchanges = []
        for content in contents:
            calls = [part["functionCall"] for part in content["parts"] if "functionCall" in part]
            responses = [
                part["functionResponse"] for part in content["parts"] if "functionResponse" in part
            ]
            if calls:
                exchanges.append(
                    {
                        "function_calls": [
                            {"id": call["id"], "name": call["name"], "arguments": call["args"]}
                            for call in calls
                        ],
                        "function_responses": [],
                    }
                )
            if responses:
                exchanges[-1]["function_responses"] = responses
        return exchanges

    async def _observation(self, ws, conn: int) -> None:
        """Read one observation: a completed client_content turn, its feedback Contents (one
        text part each) before the frame, or the realtime_input sequence (live.py
        send_observation)."""
        message = await self._recv(ws, conn)
        self._set_state(conn, "generating")
        if "client_content" in message:
            content = message["client_content"]
            self._check(
                content["turnComplete"] is True
                and all(len(turn["parts"]) == 1 for turn in content["turns"][:-1]),
                f"conn {conn}: observation {list(content)}",
            )
            return
        sequence = [next(iter(message["realtime_input"]))]
        while sequence[-1] != "activityEnd":
            sequence.append(next(iter((await self._recv(ws, conn))["realtime_input"])))
        self._check(
            tuple(sequence) == realtime_input_order(sequence.count("text") - 1),
            f"conn {conn}: realtime_input {sequence}",
        )

    async def _play(self, ws, conn: int, turn: list) -> None:
        # Sending an observation clears the client's resumption handles (gemini/session.py
        # SessionCore.observe); only an update within this turn can supply the next one
        # (gemini/session.py SessionCore._control).
        self._resume_handle = None
        calls = []
        for step in turn:
            if step is AWAIT_ACK:
                await self._acknowledgement(ws, conn, calls)
                calls = []
                continue
            calls += step.get("toolCall", {}).get("functionCalls", [])
            update = step.get("sessionResumptionUpdate")
            if update is not None:
                self._resume_handle = update.get("newHandle") if update["resumable"] else None
            await self._send(ws, conn, step)
        self._check(
            not calls, f"conn {conn}: calls {[call['id'] for call in calls]} unacknowledged"
        )

    async def _acknowledgement(self, ws, conn: int, calls: list[dict]) -> None:
        """All calls of one toolCall are acknowledged together, once each, as queued
        (live.py GeminiLiveDriver._events)."""
        responses = (await self._recv(ws, conn))["tool_response"]["functionResponses"]
        expected = [
            {
                "id": call["id"],
                "name": call["name"],
                "response": QUEUED_RESPONSE,
                **self._ack_options,
            }
            for call in calls
        ]
        self._check(responses == expected, f"conn {conn}: acknowledgement {responses}")
        self._exchanges.append(
            {
                "function_calls": [
                    {"id": call["id"], "name": call["name"], "arguments": call["args"]}
                    for call in calls
                ],
                "function_responses": responses,
            }
        )


class WireNormalizer(baseline.Normalizer):
    """baseline.Normalizer for Gemini Live frames.

    google-genai sends bytes as URL-safe base64 (google/genai/_common.py:579, 699-729). The driver
    sends bytes in three blobs: a content part's inlineData (providers/gemini/live.py
    GeminiLiveDriver._part) and realtime_input's video and audio (live.py
    GeminiLiveDriver.send_observation). An image blob's data becomes its pixel digest, an audio
    blob's the SHA-256 of its decoded bytes; their other keys, and everything outside the blobs, are
    normalized as for the chat payloads.
    """

    BLOBS = frozenset({"inlineData", "video", "audio"})

    def __call__(self, value, key: str | None = None):
        if key in self.BLOBS:
            data = base64.b64decode(value["data"], altchars=b"-_", validate=True)
            if value["mime_type"].startswith("image/"):
                digest = baseline.image_digest(data)
            else:
                digest = "sha256:" + hashlib.sha256(data).hexdigest()
            value = {**value, "data": digest}
        return super().__call__(value, key)
