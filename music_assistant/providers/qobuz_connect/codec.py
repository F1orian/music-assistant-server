"""
Wire codec for the Qobuz Connect cloud socket.

Every websocket frame is one type byte, a varint length and a protobuf body. A PAYLOAD frame
carries a QConnectBatch of QConnectMessages. The field numbers follow the schema the Qobuz
web player uses (as published in the MIT-licensed ciaens/qobuz-connect project). Only the
messages a renderer device sends or reacts to are modelled; everything else is skipped.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum

# Frame types of the cloud socket.
FRAME_AUTHENTICATE = 1
FRAME_SUBSCRIBE = 2
FRAME_PAYLOAD = 6
FRAME_ERROR = 9
FRAME_DISCONNECT = 10

PROTO_QCONNECT = 1
# The backend channel every device addresses its payloads to.
BACKEND_CHANNEL = b"\x02"

_VARINT = 0
_FIXED64 = 1
_LEN = 2
_FIXED32 = 5


class MessageType(IntEnum):
    """QConnectMessage types handled by a renderer device."""

    ERROR = 1
    RNDR_SRVR_STATE_UPDATED = 23
    RNDR_SRVR_RENDERER_ACTION = 24
    RNDR_SRVR_VOLUME_CHANGED = 25
    RNDR_SRVR_FILE_AUDIO_QUALITY_CHANGED = 26
    RNDR_SRVR_VOLUME_MUTED = 29
    SRVR_RNDR_SET_STATE = 41
    SRVR_RNDR_SET_VOLUME = 42
    SRVR_RNDR_SET_ACTIVE = 43
    SRVR_RNDR_SET_MAX_AUDIO_QUALITY = 44
    SRVR_RNDR_SET_LOOP_MODE = 45
    SRVR_RNDR_SET_SHUFFLE_MODE = 46
    SRVR_RNDR_MUTE_VOLUME = 47
    CTRL_SRVR_JOIN_SESSION = 61
    CTRL_SRVR_SET_ACTIVE_RENDERER = 63
    CTRL_SRVR_ASK_FOR_QUEUE_STATE = 76
    CTRL_SRVR_ASK_FOR_RENDERER_STATE = 77
    SRVR_CTRL_SESSION_STATE = 81
    SRVR_CTRL_RENDERER_STATE_UPDATED = 82
    SRVR_CTRL_ADD_RENDERER = 83
    SRVR_CTRL_UPDATE_RENDERER = 84
    SRVR_CTRL_REMOVE_RENDERER = 85
    SRVR_CTRL_ACTIVE_RENDERER_CHANGED = 86
    SRVR_CTRL_QUEUE_CLEARED = 89
    SRVR_CTRL_QUEUE_STATE = 90
    SRVR_CTRL_QUEUE_TRACKS_LOADED = 91
    SRVR_CTRL_QUEUE_TRACKS_INSERTED = 92
    SRVR_CTRL_QUEUE_TRACKS_ADDED = 93
    SRVR_CTRL_QUEUE_TRACKS_REMOVED = 94
    SRVR_CTRL_QUEUE_TRACKS_REORDERED = 95
    SRVR_CTRL_SHUFFLE_MODE_SET = 96
    SRVR_CTRL_AUTOPLAY_TRACKS_LOADED = 103
    SRVR_CTRL_AUTOPLAY_TRACKS_REMOVED = 104


# Queue events all carry the new queue version in field 1.
QUEUE_VERSION_EVENTS = frozenset(
    {
        MessageType.SRVR_CTRL_QUEUE_CLEARED,
        MessageType.SRVR_CTRL_QUEUE_STATE,
        MessageType.SRVR_CTRL_QUEUE_TRACKS_LOADED,
        MessageType.SRVR_CTRL_QUEUE_TRACKS_INSERTED,
        MessageType.SRVR_CTRL_QUEUE_TRACKS_ADDED,
        MessageType.SRVR_CTRL_QUEUE_TRACKS_REMOVED,
        MessageType.SRVR_CTRL_QUEUE_TRACKS_REORDERED,
        MessageType.SRVR_CTRL_SHUFFLE_MODE_SET,
        MessageType.SRVR_CTRL_AUTOPLAY_TRACKS_LOADED,
        MessageType.SRVR_CTRL_AUTOPLAY_TRACKS_REMOVED,
    }
)

# Queue events that carry tracks (queue item id + track id) in field 3.
_QUEUE_TRACK_EVENTS = frozenset(
    {
        MessageType.SRVR_CTRL_QUEUE_STATE,
        MessageType.SRVR_CTRL_QUEUE_TRACKS_LOADED,
        MessageType.SRVR_CTRL_QUEUE_TRACKS_INSERTED,
        MessageType.SRVR_CTRL_QUEUE_TRACKS_ADDED,
        MessageType.SRVR_CTRL_AUTOPLAY_TRACKS_LOADED,
    }
)
# Queue events that carry removed queue item ids in field 3.
_QUEUE_REMOVE_EVENTS = frozenset(
    {MessageType.SRVR_CTRL_QUEUE_TRACKS_REMOVED, MessageType.SRVR_CTRL_AUTOPLAY_TRACKS_REMOVED}
)


class PlayingState(IntEnum):
    """Playing state of a renderer."""

    UNKNOWN = 0
    STOPPED = 1
    PLAYING = 2
    PAUSED = 3


class BufferState(IntEnum):
    """Buffer state of a renderer."""

    UNKNOWN = 0
    BUFFERING = 1
    OK = 2
    ERROR = 3
    UNDERRUN = 4


class ActionType(IntEnum):
    """Actions a renderer can ask the session for."""

    PREVIOUS = 1
    NEXT = 2


class DeviceType(IntEnum):
    """Device type, which picks the icon in the Qobuz apps."""

    SPEAKER = 1
    STREAMER = 2


class AudioQuality(IntEnum):
    """Audio quality tiers of Qobuz Connect."""

    MP3 = 1
    CD = 2
    HIRES_LEVEL1 = 3
    HIRES_LEVEL2 = 4
    HIRES_LEVEL3 = 5


def now_ms() -> int:
    """Return the current time in epoch milliseconds."""
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------- data


@dataclass
class QueueVersion:
    """Version of the session queue; echoed in renderer state reports."""

    major: int = 0
    minor: int = 0


@dataclass
class TrackRef:
    """A queue entry as the server names it to the renderer."""

    queue_item_id: int
    track_id: int


@dataclass
class DeviceInfo:
    """How this device presents itself to the session."""

    device_uuid: bytes
    friendly_name: str
    brand: str
    model: str
    software_version: str
    device_type: DeviceType = DeviceType.SPEAKER
    max_audio_quality: AudioQuality = AudioQuality.HIRES_LEVEL3
    volume_control: bool = True


@dataclass
class SetState:
    """
    SRVR_RNDR_SET_STATE: a delta of what the renderer should do.

    A ``current`` track means jump to it; a bare ``position`` means seek; a bare
    ``playing`` means play or pause; a negative ``current.queue_item_id`` means stop.
    """

    playing: PlayingState | None = None
    position_ms: int | None = None
    queue_version: QueueVersion | None = None
    current: TrackRef | None = None
    next: TrackRef | None = None


@dataclass
class SessionState:
    """SRVR_CTRL_SESSION_STATE: the session this device joined."""

    session_uuid: bytes
    active_renderer_id: int
    queue_version: QueueVersion


@dataclass
class RendererState:
    """SRVR_CTRL_RENDERER_STATE_UPDATED: the last state a renderer reported."""

    renderer_id: int
    playing: PlayingState
    position_ms: int
    # epoch ms at which position_ms was measured
    position_timestamp_ms: int
    duration_ms: int
    queue_item_id: int
    buffer: BufferState = BufferState.UNKNOWN

    def position_at(self, now: int) -> int:
        """
        Return the position extrapolated to a point in time.

        :param now: Epoch milliseconds.
        """
        if self.playing != PlayingState.PLAYING or not self.position_timestamp_ms:
            return self.position_ms
        position = self.position_ms + max(0, now - self.position_timestamp_ms)
        return min(position, self.duration_ms) if self.duration_ms else position


@dataclass
class ErrorMessage:
    """An error the server reports for a message we sent."""

    code: str
    message: str


@dataclass
class Message:
    """A decoded QConnectMessage: its type plus the fields of the types we model."""

    type: int
    set_state: SetState | None = None
    volume: int | None = None
    volume_delta: int | None = None
    active: bool | None = None
    muted: bool | None = None
    renderer_id: int | None = None
    renderer_name: str | None = None
    renderer_uuid: bytes | None = None
    session: SessionState | None = None
    queue_version: QueueVersion | None = None
    # queue events: the tracks they carry, and whether they replace the whole queue
    queue_tracks: list[TrackRef] | None = None
    queue_replaced: bool = False
    queue_removed: list[int] | None = None
    renderer_state: RendererState | None = None
    error: ErrorMessage | None = None
    raw: bytes = field(default=b"", repr=False)


@dataclass
class Frame:
    """A decoded cloud frame."""

    type: int
    messages: list[Message] = field(default_factory=list)
    raw: bytes = field(default=b"", repr=False)


# ---------------------------------------------------------------------------- frames


def encode_frame(frame_type: int, body: bytes) -> bytes:
    """
    Wrap a protobuf body into a cloud frame.

    :param frame_type: One of the FRAME_* constants.
    :param body: The serialized frame body.
    """
    return bytes([frame_type]) + _varint(len(body)) + body


def encode_authenticate(msg_id: int, jwt: str) -> bytes:
    """
    Build the AUTHENTICATE frame that opens a connection.

    :param msg_id: Frame sequence number.
    :param jwt: The jwt_qws socket token.
    """
    return encode_frame(
        FRAME_AUTHENTICATE,
        _uint(1, msg_id) + _uint(2, now_ms()) + _string(3, jwt),
    )


def encode_subscribe(msg_id: int) -> bytes:
    """
    Build the SUBSCRIBE frame for the Qobuz Connect protocol.

    :param msg_id: Frame sequence number.
    """
    return encode_frame(
        FRAME_SUBSCRIBE, _uint(1, msg_id) + _uint(2, now_ms()) + _uint(3, PROTO_QCONNECT)
    )


def encode_payload(msg_id: int, batch_id: int, messages: list[bytes]) -> bytes:
    """
    Build a PAYLOAD frame carrying QConnectMessages to the backend.

    :param msg_id: Frame sequence number.
    :param batch_id: Batch sequence number.
    :param messages: Serialized QConnectMessages (see the encode_* message builders).
    """
    batch = _fixed64(1, now_ms()) + _uint(2, batch_id) + b"".join(_bytes(3, m) for m in messages)
    body = (
        _uint(1, msg_id)
        + _uint(2, now_ms())
        + _uint(3, PROTO_QCONNECT)
        + _bytes(5, BACKEND_CHANNEL)
        + _bytes(7, batch)
    )
    return encode_frame(FRAME_PAYLOAD, body)


def decode_frames(data: bytes) -> list[Frame]:
    """
    Decode all frames in one websocket message.

    :param data: The binary websocket message.
    """
    frames: list[Frame] = []
    pos = 0
    while pos < len(data):
        frame_type = data[pos]
        length, pos = _read_varint(data, pos + 1)
        body = data[pos : pos + length]
        pos += length
        frame = Frame(type=frame_type, raw=body)
        if frame_type == FRAME_PAYLOAD:
            batch = _fields(body).get(7, [b""])[0]
            assert isinstance(batch, bytes)
            for raw_message in _fields(batch).get(3, []):
                assert isinstance(raw_message, bytes)
                frame.messages.append(decode_message(raw_message))
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------- messages


def encode_join_session(device: DeviceInfo) -> bytes:
    """
    Build CTRL_SRVR_JOIN_SESSION, which registers this device in the account's session.

    No session uuid is sent: the cloud refuses to join a known session by uuid, so a
    reconnect joins anew and the server hands back the same renderer id.

    :param device: How the device presents itself.
    """
    # volume remote control: 1 not allowed, 2 allowed
    capabilities = (
        _uint(1, AudioQuality.MP3)
        + _uint(2, device.max_audio_quality)
        + _uint(3, 2 if device.volume_control else 1)
    )
    info = (
        _bytes(1, device.device_uuid)
        + _string(2, device.friendly_name)
        + _string(3, device.brand)
        + _string(4, device.model)
        + _uint(6, device.device_type)
        + _bytes(7, capabilities)
        + _string(8, device.software_version)
    )
    return _message(MessageType.CTRL_SRVR_JOIN_SESSION, _bytes(2, info))


def encode_state_updated(
    playing: PlayingState,
    buffer: BufferState,
    position_ms: int,
    duration_ms: int,
    queue_version: QueueVersion,
    current_queue_item_id: int,
    next_queue_item_id: int | None = None,
) -> bytes:
    """
    Build RNDR_SRVR_STATE_UPDATED, the renderer's playback state report.

    :param playing: The playing state.
    :param buffer: The buffer state.
    :param position_ms: Playback position in the current item, in milliseconds.
    :param duration_ms: Duration of the current item, in milliseconds.
    :param queue_version: The latest queue version the device knows.
    :param current_queue_item_id: The queue item being played.
    :param next_queue_item_id: The queue item that plays next, if known.
    """
    position = _fixed64(1, now_ms()) + _uint(2, max(0, position_ms))
    state = (
        _uint(1, playing)
        + _uint(2, buffer)
        + _bytes(3, position)
        + _uint(4, max(0, duration_ms))
        + _bytes(5, _queue_version(queue_version))
        + _int(6, current_queue_item_id)
    )
    if next_queue_item_id is not None:
        state += _int(7, next_queue_item_id)
    return _message(MessageType.RNDR_SRVR_STATE_UPDATED, _bytes(1, state))


def encode_volume_changed(volume: int) -> bytes:
    """
    Build RNDR_SRVR_VOLUME_CHANGED.

    :param volume: The volume level, 0-100.
    """
    return _message(MessageType.RNDR_SRVR_VOLUME_CHANGED, _uint(1, volume))


def encode_volume_muted(muted: bool) -> bytes:
    """
    Build RNDR_SRVR_VOLUME_MUTED.

    :param muted: Whether the device is muted.
    """
    return _message(MessageType.RNDR_SRVR_VOLUME_MUTED, _uint(1, int(muted)))


def encode_renderer_action(action: ActionType) -> bytes:
    """
    Build RNDR_SRVR_RENDERER_ACTION, asking the session to move through its queue.

    :param action: The action to perform.
    """
    return _message(MessageType.RNDR_SRVR_RENDERER_ACTION, _uint(2, action))


def encode_file_audio_quality(
    sample_rate: int, bit_depth: int, channels: int, lossy: bool = False
) -> bytes:
    """
    Build RNDR_SRVR_FILE_AUDIO_QUALITY_CHANGED, the format of the file being played.

    :param sample_rate: Sample rate in Hz.
    :param bit_depth: Bits per sample.
    :param channels: Number of channels.
    :param lossy: Whether the file is lossy (MP3).
    """
    quality = AudioQuality.CD
    if lossy:
        quality = AudioQuality.MP3
    elif bit_depth > 16:
        quality = (
            AudioQuality.HIRES_LEVEL3
            if sample_rate > 96000
            else AudioQuality.HIRES_LEVEL2
            if sample_rate > 48000
            else AudioQuality.HIRES_LEVEL1
        )
    body = _uint(1, sample_rate) + _uint(2, bit_depth) + _uint(3, channels) + _uint(4, quality)
    return _message(MessageType.RNDR_SRVR_FILE_AUDIO_QUALITY_CHANGED, body)


def encode_set_active_renderer(renderer_id: int) -> bytes:
    """
    Build CTRL_SRVR_SET_ACTIVE_RENDERER.

    :param renderer_id: The renderer to make active.
    """
    return _message(MessageType.CTRL_SRVR_SET_ACTIVE_RENDERER, _int(1, renderer_id))


def encode_ask_for_queue_state(queue_version: QueueVersion, action_uuid: bytes) -> bytes:
    """
    Build CTRL_SRVR_ASK_FOR_QUEUE_STATE.

    :param queue_version: The queue version the device knows.
    :param action_uuid: A 16-byte request id.
    """
    body = _bytes(1, _queue_version(queue_version)) + _bytes(2, action_uuid)
    return _message(MessageType.CTRL_SRVR_ASK_FOR_QUEUE_STATE, body)


def encode_ask_for_renderer_state(renderer_id: int) -> bytes:
    """
    Build CTRL_SRVR_ASK_FOR_RENDERER_STATE; the answer is a renderer state broadcast.

    :param renderer_id: The renderer whose last state to ask for.
    """
    return _message(MessageType.CTRL_SRVR_ASK_FOR_RENDERER_STATE, _int(1, renderer_id))


def decode_message(data: bytes) -> Message:
    """
    Decode one QConnectMessage into the fields a renderer device needs.

    :param data: The serialized QConnectMessage.
    """
    fields = _fields(data)
    msg_type = _first_int(fields, 1)
    message = Message(type=msg_type, raw=data)
    if msg_type == MessageType.ERROR:
        err = _sub(fields, 2)
        message.error = ErrorMessage(code=_first_str(err, 1), message=_first_str(err, 2))
        return message
    # From type 21 on, the payload of a message lives in the field matching its type.
    body = _sub(fields, msg_type)
    match msg_type:
        case MessageType.SRVR_RNDR_SET_STATE:
            message.set_state = SetState(
                playing=PlayingState(_first_int(body, 1)) if 1 in body else None,
                position_ms=_first_int(body, 2) if 2 in body else None,
                queue_version=_decode_queue_version(_sub(body, 3)) if 3 in body else None,
                current=_decode_track_ref(_sub(body, 4)) if 4 in body else None,
                next=_decode_track_ref(_sub(body, 5)) if 5 in body else None,
            )
        case MessageType.SRVR_RNDR_SET_VOLUME:
            message.volume = _first_int(body, 1) if 1 in body else None
            message.volume_delta = _first_int(body, 2) if 2 in body else None
        case MessageType.SRVR_RNDR_SET_ACTIVE:
            message.active = bool(_first_int(body, 1))
        case MessageType.SRVR_RNDR_MUTE_VOLUME:
            message.muted = bool(_first_int(body, 1))
        case MessageType.SRVR_CTRL_SESSION_STATE:
            message.session = SessionState(
                session_uuid=_first_bytes(body, 1),
                active_renderer_id=_first_int(body, 2) if 2 in body else -1,
                queue_version=_decode_queue_version(_sub(body, 3)),
            )
        case MessageType.SRVR_CTRL_ADD_RENDERER | MessageType.SRVR_CTRL_UPDATE_RENDERER:
            message.renderer_id = _first_int(body, 1)
            info = _sub(body, 2)
            message.renderer_uuid = _first_bytes(info, 1)
            message.renderer_name = _first_str(info, 2)
        case MessageType.SRVR_CTRL_REMOVE_RENDERER:
            message.renderer_id = _first_int(body, 1)
        case MessageType.SRVR_CTRL_ACTIVE_RENDERER_CHANGED:
            message.renderer_id = _first_int(body, 1) if 1 in body else -1
        case MessageType.SRVR_CTRL_RENDERER_STATE_UPDATED:
            state = _sub(body, 3)
            position = _sub(state, 3)
            message.renderer_state = RendererState(
                renderer_id=_first_int(body, 1),
                playing=_playing_state(_first_int(state, 1)),
                position_ms=_first_int(position, 2),
                position_timestamp_ms=_first_int(position, 1),
                duration_ms=_first_int(state, 4),
                queue_item_id=_first_int(state, 5) if 5 in state else -1,
                buffer=_buffer_state(_first_int(state, 2)),
            )
        case _ if msg_type in QUEUE_VERSION_EVENTS:
            message.queue_version = _decode_queue_version(_sub(body, 1))
            if msg_type in _QUEUE_TRACK_EVENTS:
                tracks = [_decode_track_ref(_fields(raw)) for raw in _all_bytes(body, 3)]
                if msg_type == MessageType.SRVR_CTRL_QUEUE_STATE:
                    tracks += [_decode_track_ref(_fields(raw)) for raw in _all_bytes(body, 8)]
                message.queue_tracks = tracks
                message.queue_replaced = msg_type in (
                    MessageType.SRVR_CTRL_QUEUE_STATE,
                    MessageType.SRVR_CTRL_QUEUE_TRACKS_LOADED,
                )
            elif msg_type in _QUEUE_REMOVE_EVENTS:
                message.queue_removed = _repeated_ints(body, 3)
            elif msg_type == MessageType.SRVR_CTRL_QUEUE_CLEARED:
                message.queue_tracks = []
                message.queue_replaced = True
    return message


# ---------------------------------------------------------------------------- wire helpers


def _varint(value: int) -> bytes:
    # negative int32/int64 values travel as their 64-bit two's complement
    value &= (1 << 64) - 1
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(data):
            msg = "truncated varint"
            raise ValueError(msg)
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _key(number: int, wire_type: int) -> bytes:
    return _varint(number << 3 | wire_type)


def _uint(number: int, value: int) -> bytes:
    return _key(number, _VARINT) + _varint(int(value))


def _int(number: int, value: int) -> bytes:
    return _key(number, _VARINT) + _varint(int(value))


def _fixed64(number: int, value: int) -> bytes:
    return _key(number, _FIXED64) + struct.pack("<Q", value)


def _bytes(number: int, value: bytes) -> bytes:
    return _key(number, _LEN) + _varint(len(value)) + value


def _string(number: int, value: str) -> bytes:
    return _bytes(number, value.encode())


def _message(msg_type: int, body: bytes) -> bytes:
    return _uint(1, msg_type) + _bytes(msg_type, body)


def _queue_version(version: QueueVersion) -> bytes:
    return _uint(1, version.major) + _uint(2, version.minor)


def _fields(data: bytes) -> dict[int, list[int | bytes]]:
    """Parse a protobuf message into field number -> raw values (ints or bytes)."""
    fields: dict[int, list[int | bytes]] = {}
    pos = 0
    while pos < len(data):
        key, pos = _read_varint(data, pos)
        number, wire_type = key >> 3, key & 7
        value: int | bytes
        if wire_type == _VARINT:
            value, pos = _read_varint(data, pos)
        elif wire_type == _FIXED64:
            value = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
        elif wire_type == _LEN:
            length, pos = _read_varint(data, pos)
            value = data[pos : pos + length]
            pos += length
        elif wire_type == _FIXED32:
            value = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        else:
            msg = f"unsupported wire type {wire_type}"
            raise ValueError(msg)
        fields.setdefault(number, []).append(value)
    return fields


def _sub(fields: dict[int, list[int | bytes]], number: int) -> dict[int, list[int | bytes]]:
    value = fields.get(number, [b""])[-1]
    return _fields(value) if isinstance(value, bytes) else {}


def _first_int(fields: dict[int, list[int | bytes]], number: int) -> int:
    value = fields.get(number, [0])[-1]
    if not isinstance(value, int):
        return 0
    # int32 fields carry negatives as 64-bit two's complement
    return value - (1 << 64) if value >= 1 << 63 else value


def _all_bytes(fields: dict[int, list[int | bytes]], number: int) -> list[bytes]:
    return [v for v in fields.get(number, []) if isinstance(v, bytes)]


def _repeated_ints(fields: dict[int, list[int | bytes]], number: int) -> list[int]:
    """Return a repeated int32 field, packed or not."""
    values: list[int] = []
    for value in fields.get(number, []):
        if isinstance(value, int):
            values.append(value)
            continue
        pos = 0
        while pos < len(value):
            item, pos = _read_varint(value, pos)
            values.append(item)
    return [v - (1 << 64) if v >= 1 << 63 else v for v in values]


def _buffer_state(value: int) -> BufferState:
    try:
        return BufferState(value)
    except ValueError:
        return BufferState.UNKNOWN


def _playing_state(value: int) -> PlayingState:
    try:
        return PlayingState(value)
    except ValueError:
        return PlayingState.UNKNOWN


def _first_bytes(fields: dict[int, list[int | bytes]], number: int) -> bytes:
    value = fields.get(number, [b""])[-1]
    return value if isinstance(value, bytes) else b""


def _first_str(fields: dict[int, list[int | bytes]], number: int) -> str:
    return _first_bytes(fields, number).decode(errors="replace")


def _decode_queue_version(fields: dict[int, list[int | bytes]]) -> QueueVersion:
    return QueueVersion(major=_first_int(fields, 1), minor=_first_int(fields, 2))


def _decode_track_ref(fields: dict[int, list[int | bytes]]) -> TrackRef:
    return TrackRef(queue_item_id=_first_int(fields, 1), track_id=_first_int(fields, 2))
