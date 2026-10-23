"""The Qobuz Connect plugin provider."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Final, cast

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import (
    ConfigEntryType,
    EventType,
    PlayerFeature,
    PlayerType,
    ProviderFeature,
)
from music_assistant_models.errors import LoginFailed, SetupFailedError

from music_assistant.helpers.app_vars import app_var
from music_assistant.helpers.config_entries import (
    CONF_CONNECTED_PLAYERS,
    CONF_PUBLISH_NAME_TEMPLATE,
    create_connected_players_entry,
    create_publish_name_template_entry,
    resolve_publish_name,
)
from music_assistant.models.plugin import PluginProvider

from .device import ConnectDevice

if TYPE_CHECKING:
    from music_assistant_models.event import MassEvent

    from music_assistant.providers.qobuz import QobuzProvider

CONF_QOBUZ_PROVIDER: Final = "qobuz_provider"
TOKEN_URL: Final = "https://www.qobuz.com/api.json/0.2/qws/createToken"  # noqa: S105 - an endpoint

SUPPORTED_FEATURES: Final[set[ProviderFeature]] = set()


class QobuzConnectProvider(PluginProvider):
    """
    Makes Music Assistant players appear as devices in the Qobuz apps.

    Every connected player gets its own ConnectDevice, joined to the Qobuz Connect
    session of the account of the linked Qobuz music provider. The devices play
    through that provider, so the session always belongs to its account.
    """

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return the options of this plugin."""
        qobuz_options = [
            ConfigValueOption(value=prov.instance_id, title=prov.name)
            for prov in self.mass.get_providers()
            if prov.domain == "qobuz"
        ]
        return (
            ConfigEntry(
                key=CONF_QOBUZ_PROVIDER,
                type=ConfigEntryType.STRING,
                required=True,
                options=qobuz_options,
                default_value=qobuz_options[0].value if qobuz_options else None,
            ),
            create_connected_players_entry(
                self.mass, cast("list[str]", self.get_config_value(CONF_CONNECTED_PLAYERS) or [])
            ),
            create_publish_name_template_entry(self.get_config_value(CONF_PUBLISH_NAME_TEMPLATE)),
        )

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self.qobuz_instance_id = cast("str", self.get_config_value(CONF_QOBUZ_PROVIDER))
        self._player_ids: tuple[str, ...] = tuple(
            cast("list[str]", self.get_config_value(CONF_CONNECTED_PLAYERS) or [])
        )
        self._devices: dict[str, ConnectDevice] = {}
        self._reconcile_lock = asyncio.Lock()
        self._unsubscribe = self.mass.subscribe(
            self._on_player_event,
            (EventType.PLAYER_ADDED, EventType.PLAYER_UPDATED, EventType.PLAYER_REMOVED),
        )
        if not self.qobuz_instance_id:
            msg = "No Qobuz music provider selected"
            raise SetupFailedError(msg)

    async def loaded_in_mass(self) -> None:
        """Start a device for every connected player that is registered."""
        await super().loaded_in_mass()
        await self._reconcile()

    async def unload(self, is_removed: bool = False) -> None:
        """Stop all devices."""
        # MA may already have dropped the listener while shutting down
        with contextlib.suppress(KeyError, ValueError):
            self._unsubscribe()
        async with self._reconcile_lock:
            devices = list(self._devices.values())
            self._devices.clear()
        await asyncio.gather(*(device.stop() for device in devices), return_exceptions=True)

    async def mint_socket_token(self) -> tuple[str, str]:
        """
        Mint a Qobuz Connect socket token for the linked account.

        :return: The websocket endpoint and its jwt_qws.
        """
        qobuz = cast("QobuzProvider | None", self.mass.get_provider(self.qobuz_instance_id))
        if qobuz is None:
            msg = "The linked Qobuz music provider is not available"
            raise LoginFailed(msg)
        user_auth_token = await qobuz.get_user_auth_token()
        async with self.mass.http_session.post(
            TOKEN_URL,
            headers={"X-App-Id": app_var("qobuz_app_id"), "X-User-Auth-Token": user_auth_token},
            data={"jwt": "jwt_qws"},
        ) as resp:
            if resp.status != 200:
                msg = f"Qobuz refused a Connect token (HTTP {resp.status})"
                raise LoginFailed(msg)
            token = (await resp.json())["jwt_qws"]
        return str(token["endpoint"]), str(token["jwt"])

    def shares_speakers(self, player_id: str, renderer_id: int) -> bool:
        """
        Return whether a player and the device of a Qobuz renderer play on the same speakers.

        A group and one of its members, for example; a handover between them cannot overlap.

        :param player_id: The player of the device that hands over.
        :param renderer_id: The Qobuz renderer id of the device taking over.
        """
        other = next(
            (d for d in self._devices.values() if d.session.renderer_id == renderer_id), None
        )
        if other is None or other.player_id == player_id:
            return False
        return bool(self._speakers(player_id) & self._speakers(other.player_id))

    def _speakers(self, player_id: str) -> set[str]:
        player = self.mass.players.get_player(player_id)
        return {player_id, *(player.group_members if player else [])}

    async def _on_player_event(self, event: MassEvent) -> None:
        if event.object_id in self._player_ids:
            await self._reconcile()

    async def _reconcile(self) -> None:
        """Align the running devices with the connected players and their names."""
        async with self._reconcile_lock:
            template = self.get_config_value(CONF_PUBLISH_NAME_TEMPLATE)
            for player_id in self._player_ids:
                player = self.mass.players.get_player(player_id)
                device = self._devices.get(player_id)
                if player is None:
                    if device is not None:
                        del self._devices[player_id]
                        await device.stop()
                    continue
                name = resolve_publish_name(template, player.name)
                if device is not None and device.name == name:
                    continue
                if device is not None:
                    # the name shown in the Qobuz apps follows the player name
                    await device.stop()
                # MA sets a group's volume through its members, whatever the group
                # reports as its own features while it is still forming
                volume_control = (
                    player.type == PlayerType.GROUP
                    or PlayerFeature.VOLUME_SET in player.supported_features
                )
                device = ConnectDevice(self, player_id, name, volume_control)
                self._devices[player_id] = device
                await device.start()
