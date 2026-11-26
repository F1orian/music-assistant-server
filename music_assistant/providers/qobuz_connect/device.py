"""One Qobuz Connect device, bound to one Music Assistant player."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import TYPE_CHECKING

from music_assistant_models.enums import (
    EventType,
    PlaybackState,
    PlayerType,
    QueueOption,
    RepeatMode,
)
from music_assistant_models.errors import InvalidCommand, UnsupportedFeaturedException

from . import codec
from .codec import (
    BufferState,
    DeviceInfo,
    Message,
    MessageType,
    PlayingState,
    SetState,
    TrackRef,
)
from .session import QobuzConnectSession

if TYPE_CHECKING:
    from collections.abc import Callable

    from music_assistant_models.event import MassEvent
    from music_assistant_models.player_queue import PlayerQueue
    from music_assistant_models.queue_item import QueueItem

    from music_assistant.models.player import Player

    from .provider import QobuzConnectProvider

# While playing, the position is re-reported this often; controllers extrapolate in between.
REPORT_INTERVAL_S = 5.0
# An idle queue this soon after the device asked MA to play is still starting up.
STARTUP_GRACE_S = 10.0
# On a handover this device keeps playing until the new renderer plays, plus this long
# for the new renderer's output buffer; it gives up waiting after HANDOVER_TIMEOUT_S.
HANDOVER_OVERLAP_S = 2.0
HANDOVER_TIMEOUT_S = 12.0
# A session still without an active renderer this long after a deactivation was stopped.
HANDOVER_NOBODY_S = 2.0
# A requested position this close to the current one is not worth a restart.
SEEK_TOLERANCE_MS = 2500
# For this long after a session command, MA's state may still lag behind it.
COMMAND_SETTLE_S = 3.0

_PLAYING_STATE = {
    PlaybackState.PLAYING: PlayingState.PLAYING,
    PlaybackState.PAUSED: PlayingState.PAUSED,
    PlaybackState.IDLE: PlayingState.STOPPED,
}


class ConnectDevice:
    """
    A Qobuz Connect renderer that plays through a Music Assistant player's queue.

    The Qobuz session owns the play order. The device mirrors what a renderer is told,
    the current and the next queue item, into the player's MA queue. MA then plays them
    through the linked Qobuz music provider and the device reports MA's playback state back.
    Whatever MA plays beyond those two items (another source, autoplay) means MA took the
    player over: the device then reports itself stopped until Qobuz plays on it again.
    """

    def __init__(
        self, provider: QobuzConnectProvider, player_id: str, name: str, volume_control: bool
    ) -> None:
        """
        Initialize the device without connecting it.

        :param provider: The plugin provider this device belongs to.
        :param player_id: The Music Assistant player (and queue) this device plays on.
        :param name: The name shown in the Qobuz apps.
        :param volume_control: Whether the player's volume can be set, which shows the
            volume slider in the Qobuz apps.
        """
        self.provider = provider
        self.mass = provider.mass
        self.logger = provider.logger
        self.player_id = player_id
        self.name = name
        device_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"{provider.instance_id}:{player_id}")
        self.session = QobuzConnectSession(
            self.mass.http_session,
            provider.mint_socket_token,
            DeviceInfo(
                device_uuid=device_uuid.bytes,
                friendly_name=name,
                brand="Music Assistant",
                model="Qobuz Connect",
                software_version=self.mass.version,
                volume_control=volume_control,
            ),
            self._on_message,
            self.logger,
        )
        # what the session told us to play
        self._current: TrackRef | None = None
        self._next: TrackRef | None = None
        # whether the MA queue currently plays what the session told us
        self._owned = False
        # when the device last asked MA to start playing; an idle queue shortly after
        # is still starting up, not stopped
        self._play_requested_at = 0.0
        # the session paused the device; MA may show that as an idle queue
        self._paused_by_session = False
        # the play/pause state the session asked for last, and when
        self._requested: tuple[PlayingState, float] | None = None
        # (wall clock, position ms) the device asked MA for; reported until MA confirms it
        self._anchor: tuple[float, int] | None = None
        # a position the session sought to while paused; MA's seek would start playback
        self._pending_seek_ms: int | None = None
        # a session command is being applied; MA's intermediate queue states are not ours to follow
        self._applying = False
        # (queue item, sample rate, bit depth, channels, lossy) last reported as file quality
        self._last_file_quality: tuple[str, int, int, int, bool] | None = None
        # stops this device once the renderer it handed over to plays
        self._handover_task: asyncio.Task[None] | None = None
        # position of the session's track when MA took the player over
        self._interrupted_at_ms = 0
        # (Qobuz queue item id, track id) <-> MA queue item id of the mirrored items. Keyed on
        # both: a newly loaded Qobuz queue numbers its items from 0 again.
        self._ma_item_by_ref: dict[tuple[int, int], str] = {}
        self._ref_by_ma_item: dict[str, tuple[int, int]] = {}
        # the major version of the session queue the mirrored items belong to
        self._mirror_major: int | None = None
        # the MA queue item the device last saw playing (repeat one plays a track twice)
        self._playing_ma_item: str | None = None
        self._last_report: tuple[PlayingState, BufferState, int] | None = None
        self._last_report_time = 0.0
        self._last_volume: int | None = None
        self._unsubscribe: Callable[[], None] | None = None

    async def start(self) -> None:
        """Join the Qobuz session and follow the player's queue."""
        self._unsubscribe = self.mass.subscribe(
            self._on_mass_event,
            (EventType.QUEUE_UPDATED, EventType.QUEUE_TIME_UPDATED, EventType.PLAYER_UPDATED),
            self.player_id,
        )
        await self.session.start()

    async def stop(self) -> None:
        """Leave the Qobuz session."""
        self._cancel_handover()
        if self._unsubscribe:
            # MA may already have dropped the listener while shutting down
            with contextlib.suppress(KeyError, ValueError):
                self._unsubscribe()
            self._unsubscribe = None
        await self.session.stop()

    # ------------------------------------------------------------------ Qobuz -> MA

    async def _on_message(self, message: Message) -> None:
        match message.type:
            case MessageType.SRVR_RNDR_SET_STATE if message.set_state:
                self._applying = True
                try:
                    await self._apply_set_state(message.set_state)
                except Exception as err:  # noqa: BLE001 - any failure is reported to the session
                    # tell the session playback failed rather than leave it waiting
                    self.logger.warning("Qobuz Connect [%s] could not play: %s", self.name, err)
                    self._play_requested_at = 0.0
                    self._report(force=True, playing=PlayingState.STOPPED, buffer=BufferState.ERROR)
                finally:
                    self._applying = False
            case MessageType.SRVR_RNDR_SET_VOLUME:
                await self._apply_volume(message.volume, message.volume_delta)
            case MessageType.SRVR_RNDR_MUTE_VOLUME if message.muted is not None:
                try:
                    await self.mass.players.cmd_volume_mute(self.player_id, message.muted)
                except UnsupportedFeaturedException:
                    self.logger.debug("Qobuz Connect [%s] player cannot mute", self.name)
            case MessageType.SRVR_RNDR_SET_ACTIVE:
                if message.active:
                    self._report_volume(force=True)
                    if self._cancel_handover():
                        # switched back while this device was still playing: carry on
                        self._owned = True
                    if self._owned:
                        self._report(force=True)
                        self._last_file_quality = None
                        if queue := self.mass.player_queues.get(self.player_id):
                            self._report_file_quality(queue)
                    else:
                        await self._continue_session()
                else:
                    await self._release()
            case MessageType.SRVR_CTRL_ACTIVE_RENDERER_CHANGED if not self.session.active:
                await self._release()
            case MessageType.SRVR_CTRL_SESSION_STATE:
                await self._on_session_joined()

    async def _apply_set_state(self, state: SetState) -> None:
        self.logger.debug("Qobuz Connect [%s] set state: %s", self.name, state)
        if not self.session.active:
            # commands are for the active renderer; a late one must not start playback here
            self.logger.debug("Qobuz Connect [%s] ignores a command, not active", self.name)
            return
        if state.next is not None:
            self._next = state.next if state.next.queue_item_id >= 0 else None
        if state.playing is None and state.position_ms is None and state.current is None:
            # The server's reply to a state report: only queue version and next. Reporting
            # back would make it reply again, in a loop.
            await self._sync_next()
            return
        current = state.current
        position_ms = state.position_ms
        if state.playing == PlayingState.STOPPED or (current and current.queue_item_id < 0):
            self._pending_seek_ms = None
            if current is not None and current.queue_item_id >= 0 and current != self._current:
                # a stop that also moves to another track: the next play starts that one
                self._current = current
                self._pending_seek_ms = position_ms or 0
            await self._stop()
            return
        if (
            current is None
            and self._owned
            and state.playing == PlayingState.PLAYING
            and not self._started_current()
        ):
            # stopped on another track than MA's: play starts that one
            current = self._current
            if position_ms is None:
                position_ms = self._pending_seek_ms
        if (
            current is None
            and not self._owned
            and (state.playing == PlayingState.PLAYING or position_ms is not None)
        ):
            # Playback was handed to this device (or MA had taken the player over):
            # the command names no track, so resume where the session is.
            current, resume_ms = await self._resume_point()
            if position_ms is None:
                position_ms = resume_ms
        jumped = False
        if current is not None:
            # compared with what the device last started, not MA's current item: right
            # after a start MA may not name one yet, and the server repeats the track
            needs_play = not self._owned or self._current != current or not self._started_current()
            self._current = current
            if needs_play:
                await self._play(current, position_ms or 0)
                jumped = True
        if not jumped:
            jumped = await self._apply_position(state.playing, position_ms)
        if state.playing is not None or current is not None:
            self._paused_by_session = state.playing == PlayingState.PAUSED
            self._requested = (
                PlayingState.PAUSED if self._paused_by_session else PlayingState.PLAYING,
                time.monotonic(),
            )
        if state.playing == PlayingState.PAUSED:
            self._play_requested_at = 0.0
            await self.mass.player_queues.pause(self.player_id)
        elif (
            state.playing == PlayingState.PLAYING and not jumped and not self._playing_or_starting()
        ):
            self._play_requested_at = time.monotonic()
            await self.mass.player_queues.play(self.player_id)
        await self._sync_next()
        # report what the session asked for; MA's own state catches up a moment later
        requested = state.playing
        if requested not in (PlayingState.PLAYING, PlayingState.PAUSED):
            requested = None
        self._report(force=True, playing=requested)

    async def _apply_position(self, playing: PlayingState | None, position_ms: int | None) -> bool:
        """
        Move playback of the current track to a position the session asked for.

        :param playing: The playing state the same command asked for.
        :param position_ms: The requested position, if any.
        :return: Whether the stream was restarted.
        """
        if position_ms is not None and self._is_redundant_seek(position_ms):
            # the app re-sends the position it shows; restarting for it only stutters
            position_ms = None
        if position_ms is None and playing == PlayingState.PLAYING:
            position_ms = self._pending_seek_ms
        if position_ms is None:
            return False
        if playing == PlayingState.PLAYING and (item_id := self._current_ma_item()):
            # resume at a position: one restart instead of a seek followed by a play
            await self._restart_at(item_id, position_ms)
            return True
        if self._paused_by_session or playing == PlayingState.PAUSED:
            # MA's seek restarts the stream; the position waits for the next play
            self._pending_seek_ms = position_ms
            return False
        self._anchor = (time.time(), position_ms)
        await self.mass.player_queues.seek(self.player_id, position_ms // 1000)
        return False

    async def _continue_session(self) -> None:
        """
        Carry on playing after playback was handed to this device.

        On a switch the server only activates the new device; the apps expect it to
        continue where the session is when it was playing, without a play command.
        """
        playback = await self.session.fetch_playback()
        if playback is None or playback.playing != PlayingState.PLAYING:
            return
        if self.session.current_track() is None:
            # a new queue is being loaded, or the device lost track of it: the device's own
            # last track may be from another queue, so the play command names the track
            self.logger.debug("Qobuz Connect [%s] waits for the session's track", self.name)
            return
        try:
            track, position_ms = await self._resume_point()
            self._current = track
            self._paused_by_session = False
            self._requested = (PlayingState.PLAYING, time.monotonic())
            await self._play(track, position_ms)
        except Exception as err:  # noqa: BLE001 - any failure is reported to the session
            self.logger.warning("Qobuz Connect [%s] could not take over: %s", self.name, err)
            self._report(force=True, playing=PlayingState.STOPPED, buffer=BufferState.ERROR)
            return
        await self._sync_next()
        self._report(force=True, playing=PlayingState.PLAYING)

    def _playing_or_starting(self) -> bool:
        """
        Return whether MA plays, or still starts, the stream the device asked for.

        A play command then has nothing to do: MA's queue play on a queue that is not
        paused resumes it, which restarts the stream.
        """
        queue = self.mass.player_queues.get(self.player_id)
        if not self._owned or queue is None:
            return False
        if queue.state == PlaybackState.PLAYING:
            return True
        return time.monotonic() - self._play_requested_at < STARTUP_GRACE_S

    def _is_redundant_seek(self, position_ms: int) -> bool:
        """Return whether a requested position is (nearly) where playback already is."""
        queue = self.mass.player_queues.get(self.player_id)
        if not self._owned or queue is None:
            return False
        return abs(position_ms - self._position_ms(queue)) < SEEK_TOLERANCE_MS

    async def _resume_point(self) -> tuple[TrackRef, int]:
        """
        Return the session's current track and position, for resuming on this device.

        The session's active renderer reports both, wherever it played. The device's own
        last track is only the fallback, for a session that has no report yet.

        :raises RuntimeError: When neither the session nor the device knows a track.
        """
        await self.session.fetch_playback()
        if (track := self.session.current_track()) and self.session.playback:
            return track, self.session.playback.position_at(codec.now_ms())
        if self._current is not None:
            return self._current, self._interrupted_at_ms
        msg = "the Qobuz session names no track to resume"
        raise RuntimeError(msg)

    async def _play(self, track: TrackRef, position_ms: int) -> None:
        """Make the MA queue play a session track, from the given position."""
        major = self.session.queue_version.major
        item_id = self._ma_item_by_ref.get(_ref(track))
        if item_id is None or self.mass.player_queues.get_item(self.player_id, item_id) is None:
            if not self._owned or major != self._mirror_major:
                # taking the player over, or a new Qobuz queue: the session's play order
                # replaces the MA queue, so MA's previous cannot reach another queue's items
                self.mass.player_queues.clear(self.player_id, skip_stop=True)
                self._forget_mirror()
            item_id = await self._enqueue(track, QueueOption.NEXT)
        if not self._owned:
            await self._turn_off_ma_repeat()
        self._owned = True
        self._mirror_major = major
        await self._restart_at(item_id, position_ms)

    async def _restart_at(self, item_id: str, position_ms: int) -> None:
        """Start the MA stream of a mirrored item at a position."""
        self._pending_seek_ms = None
        self._playing_ma_item = item_id
        self._play_requested_at = time.monotonic()
        self._anchor = (time.time(), position_ms)
        await self.mass.player_queues.play_index(
            self.player_id, item_id, seek_position=position_ms // 1000
        )

    async def _turn_off_ma_repeat(self) -> None:
        """Leave repeating to Qobuz: MA repeating its queue would replay mirrored items."""
        queue = self.mass.player_queues.get(self.player_id)
        if queue is not None and queue.repeat_mode != RepeatMode.OFF:
            self.logger.debug("Qobuz Connect [%s] turns off MA's repeat", self.name)
            # a dynamic queue refuses; it has no repeat to replay mirrored items with
            with contextlib.suppress(InvalidCommand):
                await self.mass.player_queues.set_repeat(self.player_id, RepeatMode.OFF)

    async def _sync_next(self) -> None:
        """Make the MA queue item after the current one the session's next track."""
        if not self._owned or self._next is None:
            return
        queue = self.mass.player_queues.get(self.player_id)
        if queue is None or queue.current_index is None:
            return
        upcoming = self.mass.player_queues.get_item(self.player_id, queue.current_index + 1)
        if upcoming and self._ref_by_ma_item.get(upcoming.queue_item_id) == _ref(self._next):
            return
        try:
            await self._enqueue(self._next, QueueOption.REPLACE_NEXT)
        except Exception as err:  # noqa: BLE001 - the current track must play on
            # the current track plays on; MA's queue just ends after it
            self.logger.warning(
                "Qobuz Connect [%s] could not queue the next track: %s", self.name, err
            )

    async def _enqueue(self, track: TrackRef, option: QueueOption) -> str:
        """
        Put a session track in the MA queue right after the current item.

        :param track: The session track to enqueue.
        :param option: NEXT to insert, REPLACE_NEXT to also drop what followed.
        :return: The MA queue item id of the enqueued track.
        """
        uri = f"{self.provider.qobuz_instance_id}://track/{track.track_id}"
        before = {item.queue_item_id for item in self.mass.player_queues.items(self.player_id)}
        await self.mass.player_queues.play_media(self.player_id, uri, option=option)
        added = [
            item
            for item in self.mass.player_queues.items(self.player_id)
            if item.queue_item_id not in before
        ]
        if not added:
            msg = f"Qobuz track {track.track_id} could not be enqueued"
            raise RuntimeError(msg)
        item_id = added[0].queue_item_id
        self._ma_item_by_ref[_ref(track)] = item_id
        self._ref_by_ma_item[item_id] = _ref(track)
        return item_id

    async def _apply_volume(self, volume: int | None, delta: int | None) -> None:
        player = self.mass.players.get_player(self.player_id)
        if player is None:
            return
        if volume is None and delta is not None:
            volume = (_volume_of(player) or 0) + delta
        if volume is None:
            return
        volume = max(0, min(100, volume))
        try:
            await self.mass.players.cmd_volume_set(self.player_id, volume)
        except UnsupportedFeaturedException:
            self.logger.debug("Qobuz Connect [%s] player has no volume control", self.name)
            return
        self._last_volume = volume
        self.session.send(codec.encode_volume_changed(volume))

    async def _stop(self) -> None:
        self._play_requested_at = 0.0
        if self._owned:
            await self.mass.player_queues.stop(self.player_id)
        self._report(force=True, playing=PlayingState.STOPPED)

    async def _release(self) -> None:
        """Another renderer took the session over: stop playing its tracks here, overlapping."""
        # the next track and a pending seek belong to the session as it was; once active
        # again the server names the next track anew
        self._next = None
        self._pending_seek_ms = None
        if not self._owned:
            return
        self.logger.debug("Qobuz Connect [%s] is no longer the active device", self.name)
        self._owned = False
        queue = self.mass.player_queues.get(self.player_id)
        if queue is None or queue.state != PlaybackState.PLAYING:
            await self.mass.player_queues.stop(self.player_id)
            return
        if self._handover_task is None or self._handover_task.done():
            self._handover_task = asyncio.create_task(self._stop_after_handover())

    async def _stop_after_handover(self) -> None:
        """
        Keep playing until the renderer that took over plays, then stop.

        The new renderer needs a few seconds to start; stopping here at once leaves that
        gap silent. The server deactivates this device before it names the new one, so
        a session that stays without an active renderer was a plain stop.
        """
        released = time.monotonic()
        overlap = False
        while (waited := time.monotonic() - released) < HANDOVER_TIMEOUT_S:
            active = self.session.active_renderer_id
            if active == -1 and waited > HANDOVER_NOBODY_S:
                break
            if active != -1 and self.provider.shares_speakers(self.player_id, active):
                # the new device plays on (some of) these speakers: no overlap possible
                break
            playback = self.session.playback
            if active != -1 and playback is not None and playback.renderer_id == active:
                if playback.playing == PlayingState.PLAYING and playback.buffer == BufferState.OK:
                    overlap = True
                    break
                if playback.playing in (PlayingState.PAUSED, PlayingState.STOPPED):
                    break
            await asyncio.sleep(0.1)
        if overlap:
            # the new renderer's own output buffer still delays its first sound
            await asyncio.sleep(HANDOVER_OVERLAP_S)
        self.logger.debug(
            "Qobuz Connect [%s] hands over after %.1f s", self.name, time.monotonic() - released
        )
        await self.mass.player_queues.stop(self.player_id)

    def _cancel_handover(self) -> bool:
        """Cancel a pending handover stop; returns whether one was pending."""
        if self._handover_task is None or self._handover_task.done():
            return False
        self._handover_task.cancel()
        self._handover_task = None
        return True

    async def _on_session_joined(self) -> None:
        """
        Re-claim the session after a reconnect while MA still plays its tracks.

        The cloud forgets the active renderer when its socket drops; when nobody
        else took over in the meantime, the device makes itself active again. When
        another renderer did, the device hands over as if it had been told.
        """
        if not self._owned:
            return
        active = self.session.active_renderer_id
        if active != -1 and active not in self.session.own_renderer_ids:
            await self._release()
            return
        queue = self.mass.player_queues.get(self.player_id)
        if queue and queue.state == PlaybackState.PLAYING:
            self.session.activate()

    # ------------------------------------------------------------------ MA -> Qobuz

    def _on_mass_event(self, event: MassEvent) -> None:
        if event.event == EventType.PLAYER_UPDATED:
            self._report_volume()
            return
        if not self._owned or self._applying:
            return
        queue = self.mass.player_queues.get(self.player_id)
        if queue is None:
            return
        if (
            self._pending_seek_ms is not None
            and queue.state == PlaybackState.PLAYING
            and not self._settling()
        ):
            # resumed in MA, which plays on from where it paused
            self._pending_seek_ms = None
        if event.event == EventType.QUEUE_UPDATED:
            self._follow_queue(queue)
            self._report()
            self._report_file_quality(queue)
            return
        if self._current_qid() is not None:
            self._interrupted_at_ms = int(queue.corrected_elapsed_time * 1000)
        self._report(periodic=True)

    def _follow_queue(self, queue: PlayerQueue) -> None:
        """Track where MA's queue went on its own (track end, MA-side next/previous)."""
        item: QueueItem | None = queue.current_item
        if item is None:
            return
        ref = self._ref_by_ma_item.get(item.queue_item_id)
        if ref is not None:
            self._interrupted_at_ms = int(queue.corrected_elapsed_time * 1000)
        if ref is None:
            # MA plays something the session did not ask for
            self.logger.debug("Qobuz Connect [%s] player was taken over in MA", self.name)
            self._report(force=True, playing=PlayingState.STOPPED)
            self._owned = False
            return
        if item.queue_item_id == self._playing_ma_item:
            return
        # compared by MA item, not session track: with repeat one, the next item is the same track
        self._playing_ma_item = item.queue_item_id
        playing = TrackRef(*ref)
        self._current = playing
        if self._next == playing:
            self._next = None
        self._report(force=True)

    def _report(
        self,
        force: bool = False,
        periodic: bool = False,
        playing: PlayingState | None = None,
        buffer: BufferState = BufferState.OK,
    ) -> None:
        """
        Report the renderer state to the session.

        :param force: Report even when nothing changed.
        :param periodic: A position tick; reported at most every REPORT_INTERVAL_S.
        :param playing: Report this playing state instead of the queue's.
        :param buffer: Report this buffer state (ERROR after a failed command).
        """
        if not self.session.active or self._current is None:
            return
        queue = self.mass.player_queues.get(self.player_id)
        if queue is None:
            return
        now = time.monotonic()
        if not self._owned and playing is None:
            # MA plays something else on this player (or nothing)
            playing = PlayingState.STOPPED
        state = playing or _PLAYING_STATE.get(queue.state, PlayingState.PAUSED)
        if playing is None and self._requested is not None and self._settling():
            state = self._requested[0]
        if playing is None and self._paused_by_session and state == PlayingState.STOPPED:
            state = PlayingState.PAUSED
        starting = (
            self._owned
            and queue.state != PlaybackState.PLAYING
            and now - self._play_requested_at < STARTUP_GRACE_S
        )
        if (
            starting
            and buffer == BufferState.OK
            and (
                state == PlayingState.PLAYING
                or (playing is None and queue.state == PlaybackState.IDLE)
            )
        ):
            # MA is still (re)starting the stream the session asked for. Reported as
            # buffering, so the device that handed over keeps playing until it runs.
            state, buffer = PlayingState.PLAYING, BufferState.BUFFERING
        position_ms = self._interrupted_at_ms if not self._owned else self._position_ms(queue)
        key = (state, buffer, self._current.queue_item_id)
        if periodic:
            if state != PlayingState.PLAYING or now - self._last_report_time < REPORT_INTERVAL_S:
                return
        elif not force and key == self._last_report:
            return
        item = queue.current_item
        self._last_report = key
        self._last_report_time = now
        self.session.send(
            codec.encode_state_updated(
                playing=state,
                buffer=buffer,
                position_ms=position_ms,
                duration_ms=int((item.duration or 0) * 1000) if item else 0,
                queue_version=self.session.queue_version,
                current_queue_item_id=self._current.queue_item_id,
                next_queue_item_id=self._next.queue_item_id if self._next else None,
            )
        )

    def _report_file_quality(self, queue: PlayerQueue) -> None:
        """Tell the session the format of the file being played, once MA streams it."""
        if not self.session.active or not self._owned or self._current_qid() is None:
            return
        item = queue.current_item
        details = item.streamdetails if item else None
        if item is None or details is None or not details.audio_format.sample_rate:
            return
        fmt = details.audio_format
        quality = (
            item.queue_item_id,
            fmt.sample_rate,
            fmt.bit_depth,
            fmt.channels,
            not fmt.content_type.is_lossless(),
        )
        if quality == self._last_file_quality:
            return
        self._last_file_quality = quality
        self.session.send(
            codec.encode_file_audio_quality(
                fmt.sample_rate, fmt.bit_depth, fmt.channels or 2, lossy=quality[4]
            )
        )

    def _report_volume(self, force: bool = False) -> None:
        if not self.session.active:
            return
        player = self.mass.players.get_player(self.player_id)
        volume = _volume_of(player) if player else None
        if volume is None:
            return
        if force or volume != self._last_volume:
            self._last_volume = volume
            self.session.send(codec.encode_volume_changed(volume))

    # ------------------------------------------------------------------ helpers

    def _position_ms(self, queue: PlayerQueue) -> int:
        """
        Return the position to report: MA's, once it agrees with the last request.

        Right after a seek or jump MA's elapsed time can still include the previous
        stream for a moment, so for COMMAND_SETTLE_S the device reports the requested
        position, advanced by the time since. A seek while paused is reported as is.
        """
        if self._pending_seek_ms is not None:
            return self._pending_seek_ms
        if self._anchor is not None:
            requested_at, position_ms = self._anchor
            since = time.time() - requested_at
            if since < COMMAND_SETTLE_S:
                return position_ms + int(since * 1000)
            self._anchor = None
        return int(queue.corrected_elapsed_time * 1000)

    def _settling(self) -> bool:
        """Return whether MA's state may still lag behind the last session command."""
        if self._requested is None:
            return False
        return time.monotonic() - self._requested[1] < COMMAND_SETTLE_S

    def _started_current(self) -> bool:
        """Return whether the MA item the device last started or followed is the session's track."""
        if self._current is None or self._playing_ma_item is None:
            return False
        return self._ref_by_ma_item.get(self._playing_ma_item) == _ref(self._current)

    def _current_ma_item(self) -> str | None:
        queue = self.mass.player_queues.get(self.player_id)
        if queue is None or queue.current_item is None:
            return None
        return queue.current_item.queue_item_id

    def _current_qid(self) -> int | None:
        queue = self.mass.player_queues.get(self.player_id)
        if queue is None or queue.current_item is None:
            return None
        ref = self._ref_by_ma_item.get(queue.current_item.queue_item_id)
        return ref[0] if ref else None

    def _forget_mirror(self) -> None:
        self._ma_item_by_ref.clear()
        self._ref_by_ma_item.clear()
        self._playing_ma_item = None


def _ref(track: TrackRef) -> tuple[int, int]:
    """Return the key a mirrored session track is known by."""
    return (track.queue_item_id, track.track_id)


def _volume_of(player: Player) -> int | None:
    """Return a player's volume; for a group, the group volume MA sets through it."""
    if player.type == PlayerType.GROUP:
        return player.group_volume
    return player.volume_level
