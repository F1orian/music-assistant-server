"""Tests for the Qobuz Connect wire codec."""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant.providers.qobuz_connect import codec
from music_assistant.providers.qobuz_connect.codec import (
    FRAME_PAYLOAD,
    BufferState,
    MessageType,
    PlayingState,
    QueueVersion,
)

if TYPE_CHECKING:
    import pytest

# Real frames captured between the Qobuz desktop app (renderer) and the Qobuz web player
# (controller), published in the qonductor project's session debug notes.
CAPTURED_SET_STATE_PLAY = bytes.fromhex(
    "062608cf1510e8ae8eaab33318012201023a1509689743359b0100001089151a070829ca02020802"
)
CAPTURED_SET_STATE_PAUSE = bytes.fromhex(
    "062608d51510a2fa8faab33318012201023a150922fd43359b010000108f151a070829ca02020803"
)
CAPTURED_STATE_UPDATED = bytes.fromhex(
    "064408101091af8eaab33318012a01023a3409919743359b010000100e1a270817ba01220a2008021002"
    "1a0c09839743359b01000010f82f20bdc4112a040802100130013802"
)


def test_decodes_captured_play_and_pause() -> None:
    """Captured SET_STATE frames decode to play and pause deltas without a track jump."""
    (play,) = codec.decode_frames(CAPTURED_SET_STATE_PLAY)
    (pause,) = codec.decode_frames(CAPTURED_SET_STATE_PAUSE)
    assert play.type == FRAME_PAYLOAD
    assert [m.type for m in play.messages] == [MessageType.SRVR_RNDR_SET_STATE]
    assert play.messages[0].set_state == codec.SetState(playing=PlayingState.PLAYING)
    assert pause.messages[0].set_state == codec.SetState(playing=PlayingState.PAUSED)


def test_state_report_matches_captured_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Our state report serializes byte-for-byte like the official desktop app's."""
    (frame,) = codec.decode_frames(CAPTURED_STATE_UPDATED)
    captured = frame.messages[0].raw
    # the position timestamp in the capture
    monkeypatch.setattr(codec, "now_ms", lambda: 0x19B35439783)
    ours = codec.encode_state_updated(
        playing=PlayingState.PLAYING,
        buffer=BufferState.OK,
        position_ms=6136,
        duration_ms=287293,
        queue_version=QueueVersion(2, 1),
        current_queue_item_id=1,
        next_queue_item_id=2,
    )
    assert ours == captured


def test_set_state_with_tracks_and_negative_ids() -> None:
    """A jump carries current and next refs; negative queue item ids survive a round trip."""
    ref = codec._int(1, -1) + codec._key(2, 5) + (9140024).to_bytes(4, "little")
    nxt = codec._int(1, 7) + codec._key(2, 5) + (9140025).to_bytes(4, "little")
    body = (
        codec._uint(1, PlayingState.PLAYING)
        + codec._uint(2, 60000)
        + codec._bytes(3, codec._queue_version(QueueVersion(3, 1)))
        + codec._bytes(4, ref)
        + codec._bytes(5, nxt)
    )
    message = codec.decode_message(codec._message(MessageType.SRVR_RNDR_SET_STATE, body))
    assert message.set_state == codec.SetState(
        playing=PlayingState.PLAYING,
        position_ms=60000,
        queue_version=QueueVersion(3, 1),
        current=codec.TrackRef(-1, 9140024),
        next=codec.TrackRef(7, 9140025),
    )


def test_payload_frame_round_trip() -> None:
    """Messages we send come back out of a payload frame unchanged."""
    messages = [codec.encode_volume_changed(42), codec.encode_set_active_renderer(5)]
    data = codec.encode_payload(1, 1, messages) + codec.encode_payload(2, 2, messages[:1])
    frames = codec.decode_frames(data)
    assert [len(f.messages) for f in frames] == [2, 1]
    assert [m.raw for m in frames[0].messages] == messages


def test_decodes_session_renderer_and_error_messages() -> None:
    """Session bookkeeping messages decode into their fields."""
    info = codec._bytes(1, b"u" * 16) + codec._string(2, "Kitchen")
    added = codec.decode_message(
        codec._message(MessageType.SRVR_CTRL_ADD_RENDERER, codec._int(1, 6) + codec._bytes(2, info))
    )
    assert (added.renderer_id, added.renderer_name, added.renderer_uuid) == (
        6,
        "Kitchen",
        b"u" * 16,
    )

    session = codec.decode_message(
        codec._message(
            MessageType.SRVR_CTRL_SESSION_STATE,
            codec._bytes(1, b"s" * 16) + codec._bytes(3, codec._queue_version(QueueVersion(4, 2))),
        )
    )
    assert session.session == codec.SessionState(b"s" * 16, -1, QueueVersion(4, 2))

    error = codec.decode_message(
        codec._uint(1, MessageType.ERROR)
        + codec._bytes(2, codec._string(1, "10002") + codec._string(2, "nope"))
    )
    assert error.error == codec.ErrorMessage("10002", "nope")


def test_join_session_carries_no_session_uuid() -> None:
    """The cloud refuses joins by session uuid, so the join message never carries one."""
    device = codec.DeviceInfo(
        device_uuid=b"d" * 16,
        friendly_name="Kitchen",
        brand="Music Assistant",
        model="Qobuz Connect",
        software_version="1",
    )
    fields = codec._fields(codec.encode_join_session(device))
    join = codec._sub(fields, MessageType.CTRL_SRVR_JOIN_SESSION)
    assert 1 not in join
    assert codec._first_str(codec._sub(join, 2), 2) == "Kitchen"


CAPTURED_RENDERER_STATE = bytes.fromhex(
    "064208d01510d0af8eaab33318012201023a3109d09743359b010000108a151a23085292051e080910011a18080210021a0c09839743359b01000010f82f20bdc4112801"
)


def test_decodes_captured_renderer_state_broadcast() -> None:
    """The server's broadcast of a renderer's report carries its item and timed position."""
    (frame,) = codec.decode_frames(CAPTURED_RENDERER_STATE)
    state = frame.messages[0].renderer_state
    assert state == codec.RendererState(
        renderer_id=9,
        playing=PlayingState.PLAYING,
        position_ms=6136,
        position_timestamp_ms=1766125180803,
        duration_ms=287293,
        queue_item_id=1,
        buffer=BufferState.OK,
    )
    assert state.position_at(1766125180803 + 4000) == 10136
    assert state.position_at(1766125180803 + 10**9) == 287293  # clamped to the duration


def test_decodes_queue_track_events() -> None:
    """Queue events yield their tracks; removals decode from a packed id list."""
    track = codec._int(1, 4) + codec._key(2, 5) + (9140024).to_bytes(4, "little")
    version = codec._bytes(1, codec._queue_version(QueueVersion(5, 1)))
    loaded = codec.decode_message(
        codec._message(MessageType.SRVR_CTRL_QUEUE_TRACKS_LOADED, version + codec._bytes(3, track))
    )
    assert loaded.queue_tracks == [codec.TrackRef(4, 9140024)]
    assert loaded.queue_replaced
    packed = codec._bytes(3, codec._varint(4) + codec._varint(7))
    removed = codec.decode_message(
        codec._message(MessageType.SRVR_CTRL_QUEUE_TRACKS_REMOVED, version + packed)
    )
    assert removed.queue_removed == [4, 7]
    assert removed.queue_version == QueueVersion(5, 1)
