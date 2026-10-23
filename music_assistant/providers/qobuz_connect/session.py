"""Connection of one Qobuz Connect device to the Qobuz cloud session of its account."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

import aiohttp

from . import codec
from .codec import DeviceInfo, Message, MessageType, QueueVersion, RendererState, TrackRef

# Returns a fresh (websocket endpoint, jwt_qws) pair; every connection needs its own token.
TokenProvider = Callable[[], Awaitable[tuple[str, str]]]
MessageHandler = Callable[[Message], Awaitable[None]]

HEARTBEAT_S = 30
BACKOFF_MAX_S = 60
# A connection that lived this long resets the reconnect backoff.
STABLE_S = 60


class QobuzConnectSession:
    """
    Keeps one device joined to the Qobuz Connect session of an account.

    The connection reconnects on its own with a fresh token. Received messages are
    handed to ``on_message`` one at a time, off the socket loop, so a slow handler
    never stalls the socket (the cloud drops a socket that stops answering).
    Messages sent while disconnected are dropped: the device re-reports its
    state once the session is joined again.
    """

    def __init__(
        self,
        http_session: aiohttp.ClientSession,
        get_token: TokenProvider,
        device: DeviceInfo,
        on_message: MessageHandler,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the session without connecting.

        :param http_session: The HTTP session to open the websocket with.
        :param get_token: Returns a fresh (endpoint, jwt_qws) pair for each connection.
        :param device: How the device presents itself to the session.
        :param on_message: Called for every received message, in order.
        :param logger: Logger to use.
        """
        self.device = device
        self.logger = logger
        self.renderer_id: int | None = None
        # every renderer id this device was registered as; the cloud may still name an old
        # one active right after a reconnect
        self.own_renderer_ids: set[int] = set()
        self.active_renderer_id: int = -1
        self.session_uuid = b""
        self.queue_version = QueueVersion()
        # Qobuz queue item id -> track id, for every item of the session queue
        self.queue_tracks: dict[int, int] = {}
        # the last state the session's active renderer reported (possibly this device)
        self.playback: RendererState | None = None
        # the renderer that was active before the current one (-1: none)
        self.previous_active_renderer_id = -1
        # renderers whose last state was asked for, and a signal for the answer
        self._asked: set[int] = set()
        self._playback_answer = asyncio.Event()
        self._http = http_session
        self._get_token = get_token
        self._on_message = on_message
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._inbox: asyncio.Queue[Message] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self._msg_id = 0
        self._batch_id = 0

    @property
    def connected(self) -> bool:
        """Whether the device is registered in the session right now."""
        return self.renderer_id is not None and self._ws is not None and not self._ws.closed

    @property
    def active(self) -> bool:
        """Whether this device is the session's active renderer."""
        return self.renderer_id is not None and self.active_renderer_id == self.renderer_id

    async def start(self) -> None:
        """Start connecting (returns immediately)."""
        self._tasks = [
            asyncio.create_task(self._connection_loop()),
            asyncio.create_task(self._dispatch_loop()),
        ]

    async def stop(self) -> None:
        """Disconnect for good."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        if self._ws is not None:
            await self._ws.close()
        self._ws = None
        self.renderer_id = None

    def send(self, *messages: bytes) -> None:
        """
        Send serialized QConnectMessages in one batch (fire and forget).

        :param messages: Messages built with the ``codec.encode_*`` helpers.
        """
        ws = self._ws
        if ws is None or ws.closed:
            return
        self._batch_id += 1
        frame = codec.encode_payload(self._next_msg_id(), self._batch_id, list(messages))
        task = asyncio.create_task(ws.send_bytes(frame))
        task.add_done_callback(self._log_send_error)

    def current_track(self) -> TrackRef | None:
        """Return the session's current track, as last reported by its active renderer."""
        if self.playback is None or self.playback.queue_item_id < 0:
            return None
        track_id = self.queue_tracks.get(self.playback.queue_item_id)
        if track_id is None:
            return None
        return TrackRef(self.playback.queue_item_id, track_id)

    async def fetch_playback(self, timeout: float = 2.0) -> RendererState | None:
        """
        Ask the server where the session is, when no renderer reported it yet.

        After a (re)join the device has not seen any renderer report. The server keeps
        the last state of the session (renderer -1 once no renderer is active) and
        answers for the renderer that was active before this one.

        :param timeout: Seconds to wait for the answer.
        """
        if self.current_track() is not None:
            return self.playback
        self._playback_answer.clear()
        asked = self.previous_active_renderer_id
        self._ask_renderer_state(asked)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._playback_answer.wait(), timeout)
        # unanswered, a later report of that renderer must not speak for the session
        self._asked.discard(asked)
        return self.playback

    def activate(self) -> None:
        """Ask the session to make this device its active renderer."""
        if self.renderer_id is not None:
            self.send(codec.encode_set_active_renderer(self.renderer_id))

    async def _connection_loop(self) -> None:
        backoff = 1.0
        while True:
            started = time.monotonic()
            try:
                await self._connect_and_read()
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - any failure reconnects
                self.logger.warning("Qobuz Connect connection lost: %s", err)
            self._ws = None
            self.renderer_id = None
            self.active_renderer_id = -1
            if time.monotonic() - started > STABLE_S:
                backoff = 1.0
            self.logger.debug("Reconnecting to Qobuz Connect in %.0f seconds", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX_S)

    async def _connect_and_read(self) -> None:
        endpoint, jwt = await self._get_token()
        async with self._http.ws_connect(endpoint, heartbeat=HEARTBEAT_S) as ws:
            self._ws = ws
            await ws.send_bytes(codec.encode_authenticate(self._next_msg_id(), jwt))
            await ws.send_bytes(codec.encode_subscribe(self._next_msg_id()))
            self.send(codec.encode_join_session(self.device))
            self.logger.debug("Connected to Qobuz Connect as %s", self.device.friendly_name)
            async for ws_msg in ws:
                if ws_msg.type != aiohttp.WSMsgType.BINARY:
                    continue
                for frame in codec.decode_frames(ws_msg.data):
                    if frame.type == codec.FRAME_DISCONNECT:
                        self.logger.debug("Qobuz Connect asked us to reconnect")
                        return
                    if frame.type == codec.FRAME_ERROR:
                        self.logger.warning("Qobuz Connect error frame: %s", frame.raw[:200])
                    for message in frame.messages:
                        self._track(message)
                        self._inbox.put_nowait(message)

    async def _dispatch_loop(self) -> None:
        while True:
            message = await self._inbox.get()
            try:
                await self._on_message(message)
            except Exception:
                self.logger.exception("Error handling Qobuz Connect message %s", message.type)

    def _track(self, message: Message) -> None:
        """Keep the session bookkeeping current before the message is handled."""
        match message.type:
            case MessageType.ERROR if message.error:
                self.logger.warning(
                    "Qobuz Connect error %s: %s", message.error.code, message.error.message
                )
            case MessageType.SRVR_CTRL_ADD_RENDERER if (
                message.renderer_uuid == self.device.device_uuid
            ):
                self.renderer_id = message.renderer_id
                if message.renderer_id is not None:
                    self.own_renderer_ids.add(message.renderer_id)
            case MessageType.SRVR_CTRL_SESSION_STATE if message.session:
                self.session_uuid = message.session.session_uuid
                self.active_renderer_id = message.session.active_renderer_id
                self.queue_version = message.session.queue_version
                self.send(codec.encode_ask_for_queue_state(self.queue_version, uuid.uuid4().bytes))
                # renderer reports made before this join were never seen here
                self._ask_renderer_state(self.active_renderer_id)
            case MessageType.SRVR_CTRL_ACTIVE_RENDERER_CHANGED:
                self._set_active_renderer(
                    message.renderer_id if message.renderer_id is not None else -1
                )
            case MessageType.SRVR_RNDR_SET_ACTIVE:
                if message.active:
                    self._set_active_renderer(self.renderer_id or -1)
                elif self.active:
                    self._set_active_renderer(-1)
            case MessageType.SRVR_CTRL_RENDERER_STATE_UPDATED if message.renderer_state:
                # only the active renderer speaks for the session (or the one asked
                # for); a late report of the renderer that just handed over must not win
                renderer_id = message.renderer_state.renderer_id
                if renderer_id == self.active_renderer_id or renderer_id in self._asked:
                    self._asked.discard(renderer_id)
                    self.playback = message.renderer_state
                    self._playback_answer.set()
            case MessageType.SRVR_RNDR_SET_STATE if (
                message.set_state and message.set_state.queue_version
            ):
                self.queue_version = message.set_state.queue_version
            case _ if message.queue_version is not None:
                self.queue_version = message.queue_version
                self._track_queue(message)

    def _set_active_renderer(self, renderer_id: int) -> None:
        if renderer_id != self.active_renderer_id:
            self.previous_active_renderer_id = self.active_renderer_id
            self.active_renderer_id = renderer_id

    def _ask_renderer_state(self, renderer_id: int) -> None:
        self._asked.add(renderer_id)
        self.send(codec.encode_ask_for_renderer_state(renderer_id))

    def _track_queue(self, message: Message) -> None:
        if message.queue_tracks is not None:
            if message.queue_replaced:
                self.queue_tracks = {}
            for track in message.queue_tracks:
                self.queue_tracks[track.queue_item_id] = track.track_id
        for queue_item_id in message.queue_removed or []:
            self.queue_tracks.pop(queue_item_id, None)

    def _next_msg_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _log_send_error(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (err := task.exception()):
            self.logger.debug("Sending to Qobuz Connect failed: %s", err)
