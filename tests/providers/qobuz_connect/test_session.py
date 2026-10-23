"""Tests for the Qobuz Connect cloud session against a local fake of the Qobuz socket."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web

from music_assistant.providers.qobuz_connect import codec
from music_assistant.providers.qobuz_connect.codec import MessageType, QueueVersion
from music_assistant.providers.qobuz_connect.session import QobuzConnectSession

if TYPE_CHECKING:
    from pytest_aiohttp import AiohttpServer

DEVICE = codec.DeviceInfo(
    device_uuid=b"d" * 16,
    friendly_name="Kitchen",
    brand="Music Assistant",
    model="Qobuz Connect",
    software_version="test",
)


def server_payload(*messages: bytes) -> bytes:
    """Wrap messages the way the cloud sends them."""
    return codec.encode_payload(1, 1, list(messages))


def add_renderer(renderer_id: int, device_uuid: bytes) -> bytes:
    """Build SRVR_CTRL_ADD_RENDERER."""
    info = codec._bytes(1, device_uuid) + codec._string(2, "x")
    return codec._message(
        MessageType.SRVR_CTRL_ADD_RENDERER, codec._int(1, renderer_id) + codec._bytes(2, info)
    )


def session_state(active_renderer_id: int) -> bytes:
    """Build SRVR_CTRL_SESSION_STATE."""
    return codec._message(
        MessageType.SRVR_CTRL_SESSION_STATE,
        codec._bytes(1, b"s" * 16)
        + codec._int(2, active_renderer_id)
        + codec._bytes(3, codec._queue_version(QueueVersion(3, 1))),
    )


def renderer_state(renderer_id: int, queue_item_id: int, position_ms: int) -> bytes:
    """Build SRVR_CTRL_RENDERER_STATE_UPDATED."""
    position = codec._fixed64(1, codec.now_ms()) + codec._uint(2, position_ms)
    state = (
        codec._uint(1, codec.PlayingState.PAUSED)
        + codec._bytes(3, position)
        + codec._uint(4, 240_000)
        + codec._int(5, queue_item_id)
    )
    return codec._message(
        MessageType.SRVR_CTRL_RENDERER_STATE_UPDATED,
        codec._int(1, renderer_id) + codec._int(2, 1) + codec._bytes(3, state),
    )


class FakeCloud:
    """A fake Qobuz Connect socket that records what each connection sent."""

    def __init__(self) -> None:
        """Start without connections."""
        self.connections: list[list[codec.Frame]] = []
        self.joined = asyncio.Event()
        self.drop_first = False

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        """Answer a join with a renderer registration and an active session state."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        frames: list[codec.Frame] = []
        self.connections.append(frames)
        async for msg in ws:
            for frame in codec.decode_frames(msg.data):
                frames.append(frame)
                if frame.type != codec.FRAME_PAYLOAD:
                    continue
                for m in frame.messages:
                    if m.type == MessageType.CTRL_SRVR_ASK_FOR_RENDERER_STATE:
                        asked = codec._sub(codec._fields(m.raw), m.type)
                        # the server answers for the renderer asked about
                        await ws.send_bytes(
                            server_payload(renderer_state(codec._first_int(asked, 1), 1, 47_000))
                        )
                if any(m.type == MessageType.CTRL_SRVR_JOIN_SESSION for m in frame.messages):
                    await ws.send_bytes(
                        server_payload(add_renderer(7, DEVICE.device_uuid), session_state(7))
                    )
                    self.joined.set()
                    if self.drop_first and len(self.connections) == 1:
                        await ws.close()
        return ws


@pytest.fixture
async def cloud(aiohttp_server: AiohttpServer) -> tuple[FakeCloud, str]:
    """Run the fake cloud and return it with its websocket url."""
    fake = FakeCloud()
    app = web.Application()
    app.router.add_get("/ws", fake.handler)
    server = await aiohttp_server(app)
    return fake, str(server.make_url("/ws"))


async def wait_for(condition: object, timeout: float = 5) -> None:
    """Wait until the callable condition is true."""
    async with asyncio.timeout(timeout):
        while not condition():  # type: ignore[operator]
            await asyncio.sleep(0.01)


async def test_join_registers_and_activates(cloud: tuple[FakeCloud, str]) -> None:
    """The session authenticates, subscribes, joins and learns its renderer id and state."""
    fake, url = cloud
    tokens = iter(["jwt-1", "jwt-2"])
    received: list[codec.Message] = []

    async def get_token() -> tuple[str, str]:
        return url, next(tokens)

    async def on_message(message: codec.Message) -> None:
        received.append(message)

    async with aiohttp.ClientSession() as http:
        sess = QobuzConnectSession(http, get_token, DEVICE, on_message, logging.getLogger())
        await sess.start()
        await wait_for(lambda: sess.active)
        await wait_for(lambda: len(received) >= 2)
        assert sess.renderer_id == 7
        assert sess.queue_version == QueueVersion(3, 1)
        await sess.stop()
    frames = fake.connections[0]
    assert [f.type for f in frames[:2]] == [codec.FRAME_AUTHENTICATE, codec.FRAME_SUBSCRIBE]
    assert codec._first_str(codec._fields(frames[0].raw), 3) == "jwt-1"
    assert [m.type for m in received[:2]] == [
        MessageType.SRVR_CTRL_ADD_RENDERER,
        MessageType.SRVR_CTRL_SESSION_STATE,
    ]


async def test_reconnect_uses_fresh_token_and_rejoins_without_session_uuid(
    cloud: tuple[FakeCloud, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped socket reconnects with a new token and a plain join."""
    fake, url = cloud
    fake.drop_first = True
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    tokens: list[str] = []

    async def get_token() -> tuple[str, str]:
        tokens.append(f"jwt-{len(tokens) + 1}")
        return url, tokens[-1]

    async def on_message(_message: codec.Message) -> None:
        return

    async with aiohttp.ClientSession() as http:
        sess = QobuzConnectSession(http, get_token, DEVICE, on_message, logging.getLogger())
        await sess.start()
        await wait_for(lambda: len(fake.connections) == 2 and sess.connected)
        await sess.stop()
    assert tokens[:2] == ["jwt-1", "jwt-2"]
    rejoin = next(
        m
        for f in fake.connections[1]
        for m in f.messages
        if m.type == MessageType.CTRL_SRVR_JOIN_SESSION
    )
    join = codec._sub(codec._fields(rejoin.raw), MessageType.CTRL_SRVR_JOIN_SESSION)
    assert 1 not in join  # no session uuid: the cloud refuses to join a known session by uuid


async def test_slow_handler_does_not_stall_the_socket(cloud: tuple[FakeCloud, str]) -> None:
    """Bookkeeping follows the socket even while a message handler is still busy."""
    _, url = cloud
    release = asyncio.Event()

    async def get_token() -> tuple[str, str]:
        return url, "jwt"

    async def on_message(_message: codec.Message) -> None:
        await release.wait()

    async with aiohttp.ClientSession() as http:
        sess = QobuzConnectSession(http, get_token, DEVICE, on_message, logging.getLogger())
        await sess.start()
        # the handler blocks on the first message, yet the session state still arrives
        await wait_for(lambda: sess.active)
        release.set()
        await sess.stop()


_real_sleep = asyncio.sleep


async def _fast_sleep(delay: float) -> None:
    await _real_sleep(min(delay, 0.01))


async def test_join_asks_where_the_session_is(cloud: tuple[FakeCloud, str]) -> None:
    """Right after joining, the session learns the last state the server kept."""
    _, url = cloud

    async def get_token() -> tuple[str, str]:
        return url, "jwt"

    async def on_message(_message: codec.Message) -> None:
        return

    async with aiohttp.ClientSession() as http:
        sess = QobuzConnectSession(http, get_token, DEVICE, on_message, logging.getLogger())
        await sess.start()
        await wait_for(lambda: sess.playback is not None)
        assert sess.playback is not None
        assert (sess.playback.renderer_id, sess.playback.queue_item_id) == (7, 1)
        assert sess.playback.position_ms == 47_000
        await sess.stop()


def bare_session() -> QobuzConnectSession:
    """Return a session that never connects, for its bookkeeping alone."""
    return QobuzConnectSession(
        MagicMock(), AsyncMock(), DEVICE, AsyncMock(), logging.getLogger("test")
    )


async def test_unanswered_ask_does_not_let_that_renderer_speak_later() -> None:
    """A late report from a renderer asked about in vain is not the session's playback."""
    session = bare_session()
    session.active_renderer_id = 5
    session.previous_active_renderer_id = 3
    assert await session.fetch_playback(timeout=0.01) is None
    for frame in codec.decode_frames(server_payload(renderer_state(3, 1, 10_000))):
        for message in frame.messages:
            session._track(message)
    assert session.playback is None


def test_every_registration_is_remembered_as_own() -> None:
    """Renderer ids from earlier connections stay known as this device's."""
    session = bare_session()
    for renderer_id in (4, 9):
        session._track(
            codec.Message(
                type=MessageType.SRVR_CTRL_ADD_RENDERER,
                renderer_id=renderer_id,
                renderer_uuid=DEVICE.device_uuid,
            )
        )
    assert session.renderer_id == 9
    assert session.own_renderer_ids == {4, 9}
