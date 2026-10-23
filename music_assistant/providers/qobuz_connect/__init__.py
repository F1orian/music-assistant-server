"""
Qobuz Connect plugin for Music Assistant.

Makes Music Assistant players appear as devices in the official Qobuz apps. A device
mirrors what the Qobuz Connect session tells it to play into its player's queue and
plays it through the linked Qobuz music provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .provider import SUPPORTED_FEATURES, QobuzConnectProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

__all__ = ["QobuzConnectProvider", "setup"]


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return QobuzConnectProvider(mass, manifest, config, SUPPORTED_FEATURES)
