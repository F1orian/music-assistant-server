"""Tests for the Qobuz Connect device: mirroring the session into an MA queue and reporting."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import (
    ContentType,
    EventType,
    PlaybackState,
    PlayerType,
    QueueOption,
    RepeatMode,
)
from music_assistant_models.errors import UnsupportedFeaturedException

from music_assistant.providers.qobuz_connect import codec
from music_assistant.providers.qobuz_connect import device as device_mod
from music_assistant.providers.qobuz_connect.codec import (
    BufferState,
    Message,
    MessageType,
    PlayingState,
    QueueVersion,
    SetState,
    TrackRef,
)
from music_assistant.providers.qobuz_connect.device import ConnectDevice

PLAYER = "player_1"
QOBUZ = "qobuz--test"


class FakeQueue:
    """Just enough of MA's player queue controller and one PlayerQueue."""

    def __init__(self) -> None:
        """Start empty and idle."""
        self.items_list: list[SimpleNamespace] = []
        self.current_index: int | None = None
        self.state = PlaybackState.IDLE
        self.elapsed_time = 0.0
        self.elapsed_time_last_updated = time.time()
        self.calls: list[tuple[Any, ...]] = []
        self.fail_play = False
        # like an AirPlay group: the queue names its current item only once the stream runs
        self.lazy_current = False
        # the queue turns PLAYING only once the stream runs
        self.lazy_state = False
        self.repeat_mode = RepeatMode.OFF

    # PlayerQueue attributes
    @property
    def current_item(self) -> SimpleNamespace | None:
        """Return the item being played."""
        if self.current_index is None or self.current_index >= len(self.items_list):
            return None
        return self.items_list[self.current_index]

    @property
    def corrected_elapsed_time(self) -> float:
        """Return the elapsed time of the current item."""
        return self.elapsed_time

    # controller API
    def get(self, queue_id: str) -> FakeQueue:
        """Return the queue."""
        return self

    def get_item(self, queue_id: str, item: int | str | None) -> SimpleNamespace | None:
        """Return an item by index or id."""
        if isinstance(item, int):
            return self.items_list[item] if item < len(self.items_list) else None
        return next((i for i in self.items_list if i.queue_item_id == item), None)

    def items(self, queue_id: str) -> list[SimpleNamespace]:
        """Return all items."""
        return list(self.items_list)

    def clear(self, queue_id: str, skip_stop: bool = False) -> None:
        """Drop all items."""
        self.calls.append(("clear",))
        self.items_list.clear()
        self.current_index = None

    async def play_media(self, queue_id: str, uri: str, option: QueueOption) -> None:
        """Insert an item after the current one, dropping what followed for REPLACE_NEXT."""
        self.calls.append(("play_media", uri, option))
        item = SimpleNamespace(
            queue_item_id=uuid.uuid4().hex, uri=uri, duration=300, streamdetails=None
        )
        insert_at = 0 if self.current_index is None else self.current_index + 1
        if option == QueueOption.REPLACE_NEXT:
            del self.items_list[insert_at:]
        self.items_list.insert(insert_at, item)

    async def play_index(self, queue_id: str, item_id: str, seek_position: int = 0) -> None:
        """Play an item from a position."""
        self.calls.append(("play_index", item_id, seek_position))
        if self.fail_play:
            msg = "stream failed"
            raise RuntimeError(msg)
        index = next(i for i, item in enumerate(self.items_list) if item.queue_item_id == item_id)
        self.current_index = None if self.lazy_current else index
        self.state = PlaybackState.IDLE if self.lazy_state else PlaybackState.PLAYING
        self.elapsed_time = seek_position
        self.elapsed_time_last_updated = time.time()

    async def pause(self, queue_id: str) -> None:
        """Pause playback."""
        self.calls.append(("pause",))
        # a paused Sendspin player shows up as an idle queue
        self.state = PlaybackState.IDLE

    async def play(self, queue_id: str) -> None:
        """Resume playback."""
        self.calls.append(("play",))
        self.state = PlaybackState.PLAYING

    async def seek(self, queue_id: str, position: int) -> None:
        """Seek within the current item; like MA's, it restarts the stream, so it plays."""
        self.calls.append(("seek", position))
        self.elapsed_time = position
        self.elapsed_time_last_updated = time.time()
        self.state = PlaybackState.PLAYING

    async def set_repeat(self, queue_id: str, repeat_mode: RepeatMode) -> None:
        """Set the queue's repeat mode."""
        self.calls.append(("set_repeat", repeat_mode))
        self.repeat_mode = repeat_mode

    async def stop(self, queue_id: str) -> None:
        """Stop playback."""
        self.calls.append(("stop",))
        self.state = PlaybackState.IDLE


class FakeSession:
    """Stands in for QobuzConnectSession and records what the device sends."""

    def __init__(self) -> None:
        """Start as the registered, active renderer."""
        self.sent: list[bytes] = []
        self.renderer_id: int | None = 5
        self.own_renderer_ids = {5, 8}
        self.active_renderer_id = 5
        self.queue_version = QueueVersion(1, 1)
        self.queue_tracks: dict[int, int] = {}
        self.playback: codec.RendererState | None = None

        # what the server answers when asked where the session is
        self.answer: codec.RendererState | None = None

    async def fetch_playback(self, timeout: float = 2.0) -> codec.RendererState | None:
        """Return the known playback, or the server's answer when none is known."""
        if self.current_track() is None and self.answer is not None:
            self.playback = self.answer
        return self.playback

    def current_track(self) -> TrackRef | None:
        """Return the session's current track."""
        if self.playback is None or self.playback.queue_item_id not in self.queue_tracks:
            return None
        return TrackRef(self.playback.queue_item_id, self.queue_tracks[self.playback.queue_item_id])

    @property
    def active(self) -> bool:
        """Return whether the device is the active renderer."""
        return self.renderer_id is not None and self.active_renderer_id == self.renderer_id

    def send(self, *messages: bytes) -> None:
        """Record sent messages."""
        self.sent.extend(messages)

    def activate(self) -> None:
        """Record an activation request."""
        self.sent.append(codec.encode_set_active_renderer(self.renderer_id or 0))

    def reports(self) -> list[tuple[int, int, int, int]]:
        """Return (playing, buffer, current item, position ms) of every state report."""
        out = []
        for raw in self.sent:
            fields = codec._fields(raw)
            if codec._first_int(fields, 1) != MessageType.RNDR_SRVR_STATE_UPDATED:
                continue
            state = codec._sub(codec._sub(fields, MessageType.RNDR_SRVR_STATE_UPDATED), 1)
            position = codec._first_int(codec._sub(state, 3), 2)
            out.append(
                (
                    codec._first_int(state, 1),
                    codec._first_int(state, 2),
                    codec._first_int(state, 6),
                    position,
                )
            )
        return out

    def volumes(self) -> list[int]:
        """Return every reported volume."""
        return [
            codec._first_int(codec._sub(codec._fields(raw), 25), 1)
            for raw in self.sent
            if codec._first_int(codec._fields(raw), 1) == MessageType.RNDR_SRVR_VOLUME_CHANGED
        ]


@pytest.fixture
def queue() -> FakeQueue:
    """Return a fresh fake queue."""
    return FakeQueue()


@pytest.fixture
def device(queue: FakeQueue) -> ConnectDevice:
    """Return a device wired to the fake queue and session."""
    player = SimpleNamespace(volume_level=40, type=PlayerType.PLAYER, group_volume=None)
    mass = SimpleNamespace(
        http_session=MagicMock(),
        version="test",
        player_queues=queue,
        players=SimpleNamespace(
            get_player=lambda _pid: player,
            cmd_volume_set=AsyncMock(),
            cmd_volume_mute=AsyncMock(),
        ),
        subscribe=MagicMock(return_value=lambda: None),
    )
    provider = SimpleNamespace(
        mass=mass,
        logger=logging.getLogger("test"),
        instance_id="qobuz_connect",
        qobuz_instance_id=QOBUZ,
        mint_socket_token=AsyncMock(),
        shares_speakers=lambda *_args: False,
    )
    dev = ConnectDevice(provider, PLAYER, "Kitchen", volume_control=True)  # type: ignore[arg-type]
    dev.session = FakeSession()  # type: ignore[assignment]
    return dev


def set_state(**kwargs: Any) -> Message:
    """Build a SET_STATE message."""
    return Message(type=MessageType.SRVR_RNDR_SET_STATE, set_state=SetState(**kwargs))


def uri(track_id: int) -> str:
    """Return the MA uri of a Qobuz track on the linked provider."""
    return f"{QOBUZ}://track/{track_id}"


def queue_event(event: EventType = EventType.QUEUE_UPDATED) -> Any:
    """Return an MA queue event for the player."""
    return SimpleNamespace(event=event, object_id=PLAYER, data=None)


async def start_playing(device: ConnectDevice, position_ms: int = 0) -> None:
    """Make the session play item 0 (track 100) with item 1 (track 101) next."""
    await device._on_message(
        set_state(
            playing=PlayingState.PLAYING,
            position_ms=position_ms,
            current=TrackRef(0, 100),
            next=TrackRef(1, 101),
        )
    )


async def test_jump_takes_over_and_mirrors_current_and_next(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """A first jump replaces the MA queue with the current track and the next one."""
    queue.items_list.append(
        SimpleNamespace(queue_item_id="old", uri="radio", duration=None, streamdetails=None)
    )
    await start_playing(device, position_ms=30_000)
    assert [i.uri for i in queue.items_list] == [uri(100), uri(101)]
    assert queue.calls[0] == ("clear",)
    assert ("play_index", queue.items_list[0].queue_item_id, 30) in queue.calls
    assert device.session.reports()[-1][:3] == (PlayingState.PLAYING, BufferState.OK, 0)  # type: ignore[attr-defined]


async def test_echoed_current_track_does_not_restart(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """The server repeating the current track only updates next; playback is not restarted."""
    await start_playing(device)
    calls = len([c for c in queue.calls if c[0] == "play_index"])
    await device._on_message(set_state(current=TrackRef(0, 100), next=TrackRef(1, 101)))
    assert len([c for c in queue.calls if c[0] == "play_index"]) == calls
    assert [i.uri for i in queue.items_list] == [uri(100), uri(101)]


async def test_natural_track_end_is_reported(device: ConnectDevice, queue: FakeQueue) -> None:
    """When MA moves on to the mirrored next item, the session hears about the new current."""
    await start_playing(device)
    queue.current_index = 1
    device._on_mass_event(queue_event())
    assert device.session.reports()[-1][2] == 1  # type: ignore[attr-defined]
    # the server then names the item after it, which gets mirrored
    await device._on_message(set_state(current=TrackRef(1, 101), next=TrackRef(2, 102)))
    assert [i.uri for i in queue.items_list] == [uri(100), uri(101), uri(102)]


async def test_pause_reports_paused_even_when_queue_goes_idle(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """A session pause is reported as paused, not stopped, although the MA queue turns idle."""
    await start_playing(device)
    await device._on_message(set_state(playing=PlayingState.PAUSED))
    assert ("pause",) in queue.calls
    assert queue.state == PlaybackState.IDLE
    device._on_mass_event(queue_event())
    assert device.session.reports()[-1][0] == PlayingState.PAUSED  # type: ignore[attr-defined]


async def test_resume_at_position_restarts_once(device: ConnectDevice, queue: FakeQueue) -> None:
    """Resume with a position is one play_index with a seek, not a seek followed by a play."""
    await start_playing(device)
    await device._on_message(set_state(playing=PlayingState.PAUSED))
    queue.calls.clear()
    await device._on_message(set_state(playing=PlayingState.PLAYING, position_ms=60_000))
    assert [c[0] for c in queue.calls] == ["play_index"]
    assert queue.calls[0][2] == 60


async def test_seek_reports_requested_position_until_ma_catches_up(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """While MA's elapsed time still counts the previous stream, the requested position wins."""
    await start_playing(device)
    queue.seek = AsyncMock()  # type: ignore[method-assign]
    queue.elapsed_time = 94.4  # what MA briefly shows: 90 plus the old stream's elapsed time
    await device._on_message(set_state(position_ms=90_000))
    assert abs(device.session.reports()[-1][3] - 90_000) < 500  # type: ignore[attr-defined]


async def test_mass_takeover_reports_stopped_and_resume_retakes(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """Something else playing in MA stops the device; play from the session takes it back."""
    await start_playing(device)
    queue.elapsed_time = 42
    device._on_mass_event(queue_event(EventType.QUEUE_TIME_UPDATED))
    queue.items_list.insert(
        0, SimpleNamespace(queue_item_id="radio", uri="radio", duration=None, streamdetails=None)
    )
    queue.current_index = 0
    device._on_mass_event(queue_event())
    assert device.session.reports()[-1][0] == PlayingState.STOPPED  # type: ignore[attr-defined]
    queue.calls.clear()
    await device._on_message(set_state(playing=PlayingState.PLAYING))
    play_index = next(c for c in queue.calls if c[0] == "play_index")
    assert play_index[2] == 42


async def test_negative_item_stops(device: ConnectDevice, queue: FakeQueue) -> None:
    """A negative current queue item id means stop."""
    await start_playing(device)
    await device._on_message(set_state(current=TrackRef(-1, 0)))
    assert ("stop",) in queue.calls
    assert device.session.reports()[-1][0] == PlayingState.STOPPED  # type: ignore[attr-defined]


async def test_failed_playback_is_reported_as_error(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """When MA cannot start the stream, the session gets a stopped state with a buffer error."""
    queue.fail_play = True
    await start_playing(device)
    assert device.session.reports()[-1][:2] == (PlayingState.STOPPED, BufferState.ERROR)  # type: ignore[attr-defined]


async def test_volume_from_session(device: ConnectDevice) -> None:
    """Session volume (absolute or delta) sets the player volume and is reported back."""
    await device._on_message(Message(type=MessageType.SRVR_RNDR_SET_VOLUME, volume=30))
    await device._on_message(Message(type=MessageType.SRVR_RNDR_SET_VOLUME, volume_delta=-5))
    device.mass.players.cmd_volume_set.assert_any_await(PLAYER, 30)  # type: ignore[attr-defined]
    device.mass.players.cmd_volume_set.assert_any_await(PLAYER, 35)  # type: ignore[attr-defined]
    assert device.session.volumes() == [30, 35]  # type: ignore[attr-defined]


async def test_volume_unsupported_by_player(device: ConnectDevice) -> None:
    """A player without volume control rejects the change and nothing is reported."""
    device.mass.players.cmd_volume_set.side_effect = UnsupportedFeaturedException("no")  # type: ignore[attr-defined]
    await device._on_message(Message(type=MessageType.SRVR_RNDR_SET_VOLUME, volume=30))
    assert device.session.volumes() == []  # type: ignore[attr-defined]


async def test_queue_updates_while_jumping_do_not_revert_the_jump(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """MA's queue still showing the old item mid-jump is not mistaken for MA moving back."""
    await start_playing(device)
    original_play_index = queue.play_index

    async def play_index_with_stale_event(
        queue_id: str, item_id: str, seek_position: int = 0
    ) -> None:
        device._on_mass_event(queue_event())  # still on item 0 at this point
        await original_play_index(queue_id, item_id, seek_position)

    queue.play_index = play_index_with_stale_event  # type: ignore[method-assign]
    await device._on_message(
        set_state(playing=PlayingState.PLAYING, current=TrackRef(1, 101), next=TrackRef(2, 102))
    )
    assert {r[2] for r in device.session.reports()[-1:]} == {1}  # type: ignore[attr-defined]
    assert queue.current_item.uri == uri(101)  # type: ignore[union-attr]


def session_playing(
    device: ConnectDevice,
    queue_item_id: int,
    track_id: int,
    position_ms: int,
    *,
    playing: bool,
    age_ms: int = 0,
) -> None:
    """Make another renderer the session's reporter of where playback is."""
    device.session.queue_tracks[queue_item_id] = track_id
    device.session.playback = codec.RendererState(
        renderer_id=99,
        playing=PlayingState.PLAYING if playing else PlayingState.PAUSED,
        position_ms=position_ms,
        position_timestamp_ms=codec.now_ms() - age_ms,
        duration_ms=300_000,
        queue_item_id=queue_item_id,
    )


async def test_handover_without_track_resumes_the_session(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """Playback handed to a fresh device starts the session's track where it is now."""
    session_playing(device, 1, 101, 60_000, playing=True, age_ms=5_000)
    await device._on_message(set_state(playing=PlayingState.PLAYING))
    assert [i.uri for i in queue.items_list] == [uri(101)]
    play_index = next(c for c in queue.calls if c[0] == "play_index")
    assert play_index[2] in (64, 65)
    assert device.session.reports()[-1][2] == 1  # type: ignore[attr-defined]


async def test_switching_back_resumes_where_the_session_is_not_where_this_device_was(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """A device that played earlier picks up the session's progress made elsewhere."""
    await start_playing(device, position_ms=10_000)
    await device._on_message(Message(type=MessageType.SRVR_RNDR_SET_ACTIVE, active=False))
    session_playing(device, 1, 101, 30_000, playing=False)
    queue.calls.clear()
    await device._on_message(set_state(playing=PlayingState.PLAYING))
    play_index = next(c for c in queue.calls if c[0] == "play_index")
    assert queue.get_item(PLAYER, play_index[1]).uri == uri(101)  # type: ignore[union-attr]
    assert play_index[2] == 30


async def test_commands_while_not_active_are_ignored(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """A command arriving after another renderer took over does not start playback here."""
    device.session.active_renderer_id = 6
    await start_playing(device)
    assert queue.calls == []


async def test_group_volume(device: ConnectDevice, monkeypatch: pytest.MonkeyPatch) -> None:
    """A group reports and nudges its group volume."""
    group = SimpleNamespace(type=PlayerType.GROUP, volume_level=None, group_volume=30)
    monkeypatch.setattr(device.mass.players, "get_player", lambda _pid: group)
    device._report_volume(force=True)
    await device._on_message(Message(type=MessageType.SRVR_RNDR_SET_VOLUME, volume_delta=5))
    assert device.session.volumes() == [30, 35]  # type: ignore[attr-defined]
    device.mass.players.cmd_volume_set.assert_awaited_with(PLAYER, 35)  # type: ignore[attr-defined]


async def test_handover_right_after_a_restart_asks_the_server(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """With no renderer report seen yet, the device asks where the session is and resumes there."""
    device.session.queue_tracks[1] = 101
    device.session.answer = codec.RendererState(  # type: ignore[attr-defined]
        renderer_id=-1,
        playing=PlayingState.PAUSED,
        position_ms=47_000,
        position_timestamp_ms=codec.now_ms(),
        duration_ms=240_000,
        queue_item_id=1,
    )
    await device._on_message(set_state(playing=PlayingState.PLAYING))
    play_index = next(c for c in queue.calls if c[0] == "play_index")
    assert queue.get_item(PLAYER, play_index[1]).uri == uri(101)  # type: ignore[union-attr]
    assert play_index[2] == 47


def play_restarts(queue: FakeQueue) -> int:
    """Return how often the MA stream was (re)started."""
    return len([c for c in queue.calls if c[0] == "play_index"])


async def test_echo_while_the_stream_is_still_starting_does_not_restart(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """The server repeating the track before MA names its current item is not a new track."""
    queue.lazy_current = True
    await start_playing(device, position_ms=30_000)
    await device._on_message(set_state(current=TrackRef(0, 100), next=TrackRef(1, 101)))
    await device._on_message(
        set_state(playing=PlayingState.PLAYING, current=TrackRef(0, 100), next=TrackRef(1, 101))
    )
    assert play_restarts(queue) == 1


async def test_position_close_to_the_current_one_does_not_restart(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """The app re-sending (nearly) the position it shows is no seek."""
    await start_playing(device, position_ms=30_000)
    await device._on_message(set_state(playing=PlayingState.PLAYING, position_ms=31_000))
    await device._on_message(set_state(position_ms=29_500))
    assert play_restarts(queue) == 1
    assert not [c for c in queue.calls if c[0] == "seek"]
    # a real seek still goes through
    await device._on_message(set_state(position_ms=90_000))
    assert ("seek", 90) in queue.calls


def report_count(device: ConnectDevice) -> int:
    """Return how many state reports the device sent."""
    return len(device.session.reports())  # type: ignore[attr-defined]


async def test_set_state_without_intent_sends_no_report(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """The server's reply to a report (version / next only) must not trigger another report."""
    await start_playing(device)
    before = report_count(device)
    no_next = TrackRef(-1, 4294967295)
    for _ in range(5):
        await device._on_message(set_state(queue_version=QueueVersion(4, 1), next=no_next))
    assert report_count(device) == before
    assert play_restarts(queue) == 1


async def test_activation_continues_a_playing_session(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """Picking this device while the session plays carries on without a play command."""
    session_playing(device, 1, 101, 60_000, playing=True, age_ms=2_000)
    await device._on_message(Message(type=MessageType.SRVR_RNDR_SET_ACTIVE, active=True))
    play_index = next(c for c in queue.calls if c[0] == "play_index")
    assert queue.get_item(PLAYER, play_index[1]).uri == uri(101)  # type: ignore[union-attr]
    assert play_index[2] in (61, 62)
    # the app's play right after the switch does not restart the stream again
    await device._on_message(set_state(playing=PlayingState.PLAYING))
    assert play_restarts(queue) == 1
    assert ("play",) not in queue.calls


async def test_activation_of_a_paused_session_stays_quiet(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """Picking this device while the session is paused starts nothing."""
    session_playing(device, 1, 101, 60_000, playing=False)
    await device._on_message(Message(type=MessageType.SRVR_RNDR_SET_ACTIVE, active=True))
    assert play_restarts(queue) == 0


@pytest.fixture
def fast_handover(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the handover timings."""
    monkeypatch.setattr(device_mod, "HANDOVER_OVERLAP_S", 0.05)
    monkeypatch.setattr(device_mod, "HANDOVER_NOBODY_S", 0.1)
    monkeypatch.setattr(device_mod, "HANDOVER_TIMEOUT_S", 1.0)


def new_renderer_reports(device: ConnectDevice, playing: PlayingState, buffer: BufferState) -> None:
    """Let renderer 6 be active and report its state."""
    device.session.active_renderer_id = 6
    device.session.playback = codec.RendererState(
        renderer_id=6,
        playing=playing,
        position_ms=1000,
        position_timestamp_ms=codec.now_ms(),
        duration_ms=300_000,
        queue_item_id=0,
        buffer=buffer,
    )


async def deactivate(device: ConnectDevice) -> None:
    """Deactivate the device the way the server does: first SET_ACTIVE false, no successor yet."""
    device.session.active_renderer_id = -1
    await device._on_message(Message(type=MessageType.SRVR_RNDR_SET_ACTIVE, active=False))


@pytest.mark.usefixtures("fast_handover")
async def test_handover_keeps_playing_until_the_new_renderer_plays(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """The old device stops only after the new renderer reports it is playing."""
    await start_playing(device)
    await deactivate(device)
    new_renderer_reports(device, PlayingState.PLAYING, BufferState.BUFFERING)
    await asyncio.sleep(0.3)
    assert ("stop",) not in queue.calls  # still buffering over there
    new_renderer_reports(device, PlayingState.PLAYING, BufferState.OK)
    await asyncio.sleep(0.3)
    assert ("stop",) in queue.calls


@pytest.mark.usefixtures("fast_handover")
async def test_deactivation_without_successor_stops(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """When nobody takes the session over, the device stops after a short wait."""
    await start_playing(device)
    await deactivate(device)
    await asyncio.sleep(0.4)
    assert ("stop",) in queue.calls


@pytest.mark.usefixtures("fast_handover")
async def test_switching_back_during_the_overlap_carries_on(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """Re-selecting the device while it still plays neither stops nor restarts it."""
    await start_playing(device)
    await deactivate(device)
    new_renderer_reports(device, PlayingState.PLAYING, BufferState.BUFFERING)
    device.session.active_renderer_id = 5
    await device._on_message(Message(type=MessageType.SRVR_RNDR_SET_ACTIVE, active=True))
    await asyncio.sleep(1.2)
    assert ("stop",) not in queue.calls
    assert play_restarts(queue) == 1


async def test_reports_buffering_until_the_stream_runs(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """While MA starts the stream the device reports buffering; once it runs, OK."""
    queue.lazy_state = True
    await start_playing(device)
    assert device.session.reports()[-1][:2] == (PlayingState.PLAYING, BufferState.BUFFERING)  # type: ignore[attr-defined]
    queue.state = PlaybackState.PLAYING
    device._on_mass_event(queue_event())
    assert device.session.reports()[-1][:2] == (PlayingState.PLAYING, BufferState.OK)  # type: ignore[attr-defined]


@pytest.mark.usefixtures("fast_handover")
async def test_handover_to_a_device_on_the_same_speakers_stops_at_once(
    device: ConnectDevice, queue: FakeQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A group handing over to one of its members (or back) frees the speakers right away."""
    monkeypatch.setattr(device.provider, "shares_speakers", lambda *_args: True)
    await start_playing(device)
    await deactivate(device)
    new_renderer_reports(device, PlayingState.PLAYING, BufferState.BUFFERING)
    await asyncio.sleep(0.2)
    assert ("stop",) in queue.calls


def file_qualities(device: ConnectDevice) -> list[tuple[int, int, int, int]]:
    """Return (sample rate, bit depth, channels, quality tier) of every file quality report."""
    out = []
    for raw in device.session.sent:  # type: ignore[attr-defined]
        fields = codec._fields(raw)
        if codec._first_int(fields, 1) != MessageType.RNDR_SRVR_FILE_AUDIO_QUALITY_CHANGED:
            continue
        body = codec._sub(fields, MessageType.RNDR_SRVR_FILE_AUDIO_QUALITY_CHANGED)
        out.append(tuple(codec._first_int(body, n) for n in (1, 2, 3, 4)))
    return out  # type: ignore[return-value]


def stream_format(item: SimpleNamespace, rate: int, depth: int, content: ContentType) -> None:
    """Give a queue item the stream details MA fills in once it streams."""
    item.streamdetails = SimpleNamespace(
        audio_format=SimpleNamespace(
            sample_rate=rate, bit_depth=depth, channels=2, content_type=content
        )
    )


async def test_reports_the_file_quality_once_the_track_streams(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """The Qobuz apps learn the format of the file being played, once per track."""
    await start_playing(device)
    assert file_qualities(device) == []  # MA has no stream details yet
    stream_format(queue.items_list[0], 88200, 24, ContentType.FLAC)
    device._on_mass_event(queue_event())
    device._on_mass_event(queue_event())
    assert file_qualities(device) == [(88200, 24, 2, codec.AudioQuality.HIRES_LEVEL2)]
    # the next track is CD quality
    stream_format(queue.items_list[1], 44100, 16, ContentType.FLAC)
    queue.current_index = 1
    device._on_mass_event(queue_event())
    assert file_qualities(device)[-1] == (44100, 16, 2, codec.AudioQuality.CD)


async def test_mp3_is_reported_as_mp3(device: ConnectDevice, queue: FakeQueue) -> None:
    """A lossy stream is not reported as CD quality."""
    await start_playing(device)
    stream_format(queue.items_list[0], 44100, 16, ContentType.MPEG)
    device._on_mass_event(queue_event())
    assert file_qualities(device) == [(44100, 16, 2, codec.AudioQuality.MP3)]


async def test_new_queue_reusing_item_ids_plays_the_new_tracks(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """A newly loaded playlist numbers its items from 0 again; its tracks must still play."""
    await start_playing(device)  # item 0 = track 100, next item 1 = track 101
    await device._on_message(
        set_state(
            playing=PlayingState.PLAYING,
            position_ms=0,
            queue_version=QueueVersion(13, 1),
            current=TrackRef(1, 200),
            next=TrackRef(2, 201),
        )
    )
    assert queue.current_item.uri == uri(200)  # type: ignore[union-attr]
    upcoming = queue.get_item(PLAYER, queue.current_index + 1)  # type: ignore[operator]
    assert upcoming.uri == uri(201)  # type: ignore[union-attr]


async def test_new_queue_with_the_same_current_item_id_plays_the_new_track(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """Same queue item id, different track: that is a new track, not an echo."""
    await start_playing(device)
    await device._on_message(
        set_state(playing=PlayingState.PLAYING, position_ms=0, current=TrackRef(0, 300))
    )
    assert queue.current_item.uri == uri(300)  # type: ignore[union-attr]


async def test_natural_end_after_a_reload_follows_the_new_track(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """After a reload MA advances to the new next track; the report names its item."""
    await start_playing(device)
    await device._on_message(
        set_state(playing=PlayingState.PLAYING, current=TrackRef(1, 200), next=TrackRef(2, 201))
    )
    queue.current_index += 1  # type: ignore[operator]
    device._on_mass_event(queue_event())
    assert device.session.reports()[-1][2] == 2  # type: ignore[attr-defined]
    assert device._current == TrackRef(2, 201)


async def test_activation_without_a_known_track_waits_quietly(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """When the session's track is unknown at activation, the device waits for the command."""
    device.session.answer = codec.RendererState(  # type: ignore[attr-defined]
        renderer_id=-1,
        playing=PlayingState.PLAYING,
        position_ms=0,
        position_timestamp_ms=codec.now_ms(),
        duration_ms=0,
        queue_item_id=7,  # not in the known queue
    )
    await device._on_message(Message(type=MessageType.SRVR_RNDR_SET_ACTIVE, active=True))
    assert queue.calls == []
    assert BufferState.ERROR not in {r[1] for r in device.session.reports()}  # type: ignore[attr-defined]


async def test_seek_while_paused_stays_paused(device: ConnectDevice, queue: FakeQueue) -> None:
    """MA's seek restarts the stream; while paused the position waits for the next play."""
    await start_playing(device)
    await device._on_message(set_state(playing=PlayingState.PAUSED))
    queue.calls.clear()
    await device._on_message(set_state(position_ms=90_000))
    assert queue.state != PlaybackState.PLAYING
    assert device.session.reports()[-1][0] == PlayingState.PAUSED  # type: ignore[attr-defined]
    assert device.session.reports()[-1][3] == 90_000  # type: ignore[attr-defined]
    await device._on_message(set_state(playing=PlayingState.PLAYING))
    assert [c[0] for c in queue.calls] == ["play_index"]
    assert queue.calls[0][2] == 90


async def test_next_track_that_cannot_be_enqueued_does_not_fail_playback(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """An unavailable next track is skipped over; the current one plays on, without an error."""
    await start_playing(device)
    original = queue.play_media

    async def refuse_102(queue_id: str, uri: str, option: QueueOption) -> None:
        if not uri.endswith("/102"):
            await original(queue_id, uri, option)

    queue.play_media = refuse_102  # type: ignore[method-assign]
    await device._on_message(set_state(queue_version=QueueVersion(1, 2), next=TrackRef(2, 102)))
    await device._on_message(set_state(current=TrackRef(0, 100), next=TrackRef(2, 102)))
    assert BufferState.ERROR not in {r[1] for r in device.session.reports()}  # type: ignore[attr-defined]
    assert ("stop",) not in queue.calls


@pytest.mark.usefixtures("fast_handover")
async def test_rejoin_after_another_device_took_over_stops_here(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """The session moved on while the socket was down: this device must not keep playing."""
    await start_playing(device)
    device.session.renderer_id = 8  # registered anew after the reconnect
    new_renderer_reports(device, PlayingState.PLAYING, BufferState.OK)
    await device._on_message(Message(type=MessageType.SRVR_CTRL_SESSION_STATE))
    await asyncio.sleep(0.3)
    assert ("stop",) in queue.calls


async def test_rejoin_while_playing_reclaims_the_session(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """Nobody took over during the reconnect: the device makes itself active again."""
    await start_playing(device)
    device.session.renderer_id = 8
    device.session.active_renderer_id = -1
    before = len(device.session.sent)  # type: ignore[attr-defined]
    await device._on_message(Message(type=MessageType.SRVR_CTRL_SESSION_STATE))
    assert codec.encode_set_active_renderer(8) in device.session.sent[before:]  # type: ignore[attr-defined]
    assert ("stop",) not in queue.calls


async def test_new_queue_replaces_the_mirrored_items(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """A newly loaded Qobuz queue drops the old one's items, so MA's previous cannot reach them."""
    await start_playing(device)
    device.session.queue_version = QueueVersion(2, 1)
    await device._on_message(
        set_state(
            playing=PlayingState.PLAYING,
            position_ms=0,
            queue_version=QueueVersion(2, 1),
            current=TrackRef(1, 200),
            next=TrackRef(2, 201),
        )
    )
    assert [i.uri for i in queue.items_list] == [uri(200), uri(201)]


async def test_takeover_turns_off_ma_repeat(device: ConnectDevice, queue: FakeQueue) -> None:
    """Qobuz owns repeat; MA repeating its queue would replay mirrored items out of order."""
    queue.repeat_mode = RepeatMode.ALL
    await start_playing(device)
    assert queue.repeat_mode == RepeatMode.OFF


async def test_repeating_one_track_reports_each_play(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """With repeat one the next item is the current one; MA moving on to it is reported."""
    await device._on_message(
        set_state(
            playing=PlayingState.PLAYING,
            position_ms=0,
            current=TrackRef(0, 100),
            next=TrackRef(0, 100),
        )
    )
    assert [i.uri for i in queue.items_list] == [uri(100), uri(100)]
    before = report_count(device)
    queue.current_index = 1
    queue.elapsed_time = 0
    device._on_mass_event(queue_event())
    assert report_count(device) == before + 1
    await device._on_message(set_state(queue_version=QueueVersion(1, 2), next=TrackRef(0, 100)))
    assert [i.uri for i in queue.items_list] == [uri(100), uri(100), uri(100)]


async def test_stop_naming_a_track_does_not_start_it(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """A stop that also names a track (the queue's start, say) stops without a stream."""
    await start_playing(device)
    queue.calls.clear()
    await device._on_message(set_state(playing=PlayingState.STOPPED, current=TrackRef(0, 100)))
    await device._on_message(set_state(playing=PlayingState.STOPPED, current=TrackRef(3, 103)))
    assert play_restarts(queue) == 0
    assert ("stop",) in queue.calls
    assert device.session.reports()[-1][:3] == (PlayingState.STOPPED, BufferState.OK, 3)  # type: ignore[attr-defined]
    # play then starts the track the stop moved to
    await device._on_message(set_state(playing=PlayingState.PLAYING))
    assert queue.current_item.uri == uri(103)  # type: ignore[union-attr]
    assert play_restarts(queue) == 1


async def test_resume_in_ma_after_a_paused_seek_reports_mas_position(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """A seek kept for the next play is dropped once MA itself resumes from where it was."""
    await start_playing(device)
    await device._on_message(set_state(playing=PlayingState.PAUSED))
    await device._on_message(set_state(position_ms=90_000))
    device._requested = (PlayingState.PAUSED, time.monotonic() - 10)
    queue.state = PlaybackState.PLAYING
    queue.elapsed_time = 20
    device._anchor = None
    device._on_mass_event(queue_event())
    device._report(force=True)
    assert device.session.reports()[-1][:1] == (PlayingState.PLAYING,)  # type: ignore[attr-defined]
    assert device.session.reports()[-1][3] == 20_000  # type: ignore[attr-defined]


async def test_mute_unsupported_by_player(device: ConnectDevice) -> None:
    """A player without mute rejects it quietly."""
    device.mass.players.cmd_volume_mute.side_effect = UnsupportedFeaturedException("no")  # type: ignore[attr-defined]
    await device._on_message(Message(type=MessageType.SRVR_RNDR_MUTE_VOLUME, muted=True))


@pytest.mark.usefixtures("fast_handover")
async def test_activation_after_a_handover_does_not_enqueue_the_old_next(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """The next track known from before the handover is stale; the server names the new one."""
    await start_playing(device)
    await deactivate(device)
    await asyncio.sleep(0.4)
    session_playing(device, 5, 105, 0, playing=True)
    device.session.active_renderer_id = 5
    await device._on_message(Message(type=MessageType.SRVR_RNDR_SET_ACTIVE, active=True))
    assert [i.uri for i in queue.items_list] == [uri(105)]


@pytest.mark.usefixtures("fast_handover")
async def test_activation_without_a_known_track_does_not_replay_the_old_one(
    device: ConnectDevice, queue: FakeQueue
) -> None:
    """A session track the device cannot name is left to the play command that follows."""
    await start_playing(device)
    await deactivate(device)
    await asyncio.sleep(0.4)
    queue.calls.clear()
    device.session.active_renderer_id = 5
    device.session.answer = codec.RendererState(  # type: ignore[attr-defined]
        renderer_id=-1,
        playing=PlayingState.PLAYING,
        position_ms=0,
        position_timestamp_ms=codec.now_ms(),
        duration_ms=0,
        queue_item_id=7,
    )
    await device._on_message(Message(type=MessageType.SRVR_RNDR_SET_ACTIVE, active=True))
    assert play_restarts(queue) == 0
