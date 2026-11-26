"""Tests for the Qobuz Connect provider's configuration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from music_assistant.providers.qobuz_connect.provider import (
    CONF_QOBUZ_PROVIDER,
    QobuzConnectProvider,
)


async def test_qobuz_account_option_value_is_the_instance_id() -> None:
    """The account dropdown stores the qobuz instance id, which the devices look up."""
    qobuz = SimpleNamespace(domain="qobuz", instance_id="qobuz--abc", name="Qobuz")
    other = SimpleNamespace(domain="tidal", instance_id="tidal--x", name="Tidal")
    provider = MagicMock(spec=QobuzConnectProvider)
    provider.mass = SimpleNamespace(
        get_providers=lambda: [qobuz, other],
        players=SimpleNamespace(all_players=lambda *_args, **_kwargs: []),
    )
    provider.get_config_value = lambda _key: None
    entries = await QobuzConnectProvider.get_config_entries(provider)
    account = next(e for e in entries if e.key == CONF_QOBUZ_PROVIDER)
    assert [(o.value, o.title) for o in account.options] == [("qobuz--abc", "Qobuz")]
    assert account.default_value == "qobuz--abc"


def test_shares_speakers_between_a_group_and_its_member() -> None:
    """A group and its member share speakers; two separate speakers do not."""
    players = {
        "group": SimpleNamespace(group_members=["kitchen", "living"]),
        "kitchen": SimpleNamespace(group_members=[]),
        "bedroom": SimpleNamespace(group_members=[]),
    }
    provider = MagicMock(spec=QobuzConnectProvider)
    provider.mass = SimpleNamespace(players=SimpleNamespace(get_player=players.get))
    provider._devices = {
        pid: SimpleNamespace(player_id=pid, session=SimpleNamespace(renderer_id=rid))
        for pid, rid in (("group", 1), ("kitchen", 2), ("bedroom", 3))
    }
    provider._speakers = lambda pid: QobuzConnectProvider._speakers(provider, pid)
    assert QobuzConnectProvider.shares_speakers(provider, "group", 2)
    assert QobuzConnectProvider.shares_speakers(provider, "kitchen", 1)
    assert not QobuzConnectProvider.shares_speakers(provider, "kitchen", 3)
    assert not QobuzConnectProvider.shares_speakers(provider, "kitchen", 99)  # not ours
