# Qobuz Connect Provider

The Qobuz Connect plugin makes Music Assistant players show up as devices in the official
Qobuz apps. When you pick one in the app, the app controls that player: play, pause, seek,
skip and volume. Music Assistant plays the tracks through the linked **Qobuz music
provider**, at native quality and with its normal queue items, artwork and history.

## Module layout

```
qobuz_connect/
├── provider.py   QobuzConnectProvider: config, linked qobuz instance, one device per player
├── device.py     ConnectDevice: mirrors the session into the player's MA queue, reports back
├── session.py    QobuzConnectSession: cloud socket, join, reconnect, ordered dispatch
└── codec.py      Protobuf wire codec for the messages a renderer needs (no dependency)
```

## How it works

- **Joining.** Each device joins the Qobuz Connect session of the linked provider's account
  through the cloud. The Qobuz API mints a socket token (`qws/createToken`) from that provider's
  user auth token, so a device needs no LAN discovery. It appears in every Qobuz app of that
  account on any network. The device uuid is derived from the plugin instance and the player id,
  so the apps see the same device across restarts.
- **Qobuz owns the play order.** A renderer is only told the current and the next queue item,
  in `SRVR_RNDR_SET_STATE`. The device mirrors exactly those two into the player's MA queue
  (current via `play_index`, next via `REPLACE_NEXT`), so shuffle, repeat and autoplay stay with
  Qobuz. Large Qobuz queues cost nothing up front.
- **Reporting.** MA queue events are turned into `RNDR_SRVR_STATE_UPDATED`. A report goes out
  on every state or item change and every 5 s while playing. When MA moves on to the mirrored
  next item by itself (track end), the device reports the new current item and the session answers
  with the item after it.
- **Handover.** Switching devices in the app only activates the new device. A device that is
  activated while the session plays carries on by itself, at the session's position. It learns that
  position from the renderer state broadcasts, or by asking the server (`ASK_FOR_RENDERER_STATE`)
  right after it joins. The device that hands over keeps playing until the new renderer reports
  playing with an OK buffer, then for 2 s more for that renderer's output buffer. It stops at once
  when nobody takes over or when the two devices share speakers (a group and its member). Switching
  back during that overlap carries on without a restart. While its own stream starts, a device
  reports buffering.
- **Quality.** Once MA streams a track, the device reports the file's sample rate and bit depth to
  the session (`RNDR_SRVR_FILE_AUDIO_QUALITY_CHANGED`), which the apps show as the quality badge.
- **Taking the player away.** When MA plays something the session did not ask for on that
  player, the device reports itself stopped. A play from the Qobuz app takes the player back
  and resumes the track where it was interrupted. When another Qobuz device becomes active,
  MA stops.

## Protocol notes

- Messages are protobuf inside cloud frames (`type byte + varint length + body`). The field numbers
  follow the Qobuz web player's schema as published by
  [ciaens/qobuz-connect](https://github.com/ciaens/qobuz-connect) (MIT).
- The cloud refuses to join a known session by uuid (error 10002). A reconnect sends a plain join
  and gets the same renderer id back.
- `SET_STATE` is a delta. A current track means jump, a bare position means seek, a bare
  playing state means play or pause, and a negative queue item id means stop.
- Right after a seek or jump, MA's elapsed time can still include the previous stream for a
  moment. The device reports the requested position for a short settle window.
- Message handling runs off the socket loop. A handler that blocks the socket long enough
  makes the cloud drop the connection.

## Not supported (yet)

- LAN handover from other accounts (`_qobuz-connect._tcp` + `/streamcore`). A device can only
  play for the linked account, because the tracks play through that provider.
- Edits to the queue made in the MA UI are not pushed back to the Qobuz session.
- MA-side next/previous/shuffle/repeat are not forwarded to the session.

## Related documentation

- **PluginProvider contract:** `music_assistant/models/plugin.py`
- **Qobuz music provider:** `music_assistant/providers/qobuz/`

---

*Update this document when the provider's design changes materially.*
