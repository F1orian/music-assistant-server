"""Tests for handing the Qobuz login to other providers."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from music_assistant_models.errors import LoginFailed

from music_assistant.providers.qobuz import SUPPORTED_FEATURES, QobuzProvider


@pytest.fixture
def provider() -> QobuzProvider:
    """Create a QobuzProvider that is not logged in yet."""
    manifest = Mock()
    manifest.domain = "qobuz"
    config = Mock()
    config.instance_id = "qobuz_test"
    config.name = "Qobuz Test"
    config.enabled = True
    config.get_value.side_effect = lambda key, default=None: {"log_level": "GLOBAL"}.get(
        key, default
    )
    return QobuzProvider(Mock(), manifest, config, SUPPORTED_FEATURES)


async def test_returns_the_token_of_the_logged_in_account(provider: QobuzProvider) -> None:
    """A logged-in provider hands out its user auth token."""
    provider._user_auth_info = {"user_auth_token": "token-123"}
    assert await provider.get_user_auth_token() == "token-123"


async def test_raises_when_login_fails(
    provider: QobuzProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider that cannot log in raises instead of returning an empty token."""

    async def no_token() -> None:
        return None

    monkeypatch.setattr(provider, "_auth_token", no_token)
    monkeypatch.setattr(provider, "get_setup_value", lambda _key: "user")
    with pytest.raises(LoginFailed):
        await provider.get_user_auth_token()
