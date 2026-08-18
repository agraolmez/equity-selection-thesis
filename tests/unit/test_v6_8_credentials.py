from __future__ import annotations

from types import SimpleNamespace

import pytest

from portfolio_bot.research_v6_8 import credentials


def test_environment_credentials_take_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_KEY", "environment-key")
    monkeypatch.setenv("PAPER_SECRET", "environment-secret")
    monkeypatch.setattr(credentials, "_read_macos_keychain", lambda service: "keychain")
    resolved = credentials.load_paper_credentials(
        key_environment="PAPER_KEY", secret_environment="PAPER_SECRET"
    )
    assert resolved.source == "environment"
    assert resolved.key_id == "environment-key"
    assert resolved.secret_key == "environment-secret"


def test_keychain_fallback_is_secret_free_in_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PAPER_KEY", raising=False)
    monkeypatch.delenv("PAPER_SECRET", raising=False)
    monkeypatch.setattr(
        credentials,
        "_read_macos_keychain",
        lambda service: "stored-value",
    )
    presence = credentials.paper_credential_presence(
        key_environment="PAPER_KEY", secret_environment="PAPER_SECRET"
    )
    assert presence == {"key": True, "secret": True, "source": "macos_keychain"}


def test_keychain_reader_returns_empty_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credentials.sys, "platform", "darwin")
    monkeypatch.setattr(
        credentials.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=44, stdout=""),
    )
    assert credentials._read_macos_keychain("missing") == ""
