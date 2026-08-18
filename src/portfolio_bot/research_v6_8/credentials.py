"""Secret-free credential resolution for prospective V6.8 paper trading."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field


KEY_ENVIRONMENT = "APCA_API_KEY_ID"
SECRET_ENVIRONMENT = "APCA_API_SECRET_KEY"
KEYCHAIN_ACCOUNT = "portfolio-bot"
KEYCHAIN_KEY_SERVICE = "portfolio-bot-alpaca-paper-key-id"
KEYCHAIN_SECRET_SERVICE = "portfolio-bot-alpaca-paper-secret-key"


class PaperCredentialError(RuntimeError):
    """The paper credential pair could not be resolved safely."""


@dataclass(frozen=True, slots=True)
class PaperCredentialPair:
    key_id: str = field(repr=False)
    secret_key: str = field(repr=False)
    source: str


def _read_macos_keychain(service: str) -> str:
    if sys.platform != "darwin":
        return ""
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-a",
                KEYCHAIN_ACCOUNT,
                "-s",
                service,
                "-w",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def paper_credential_presence(
    *,
    key_environment: str = KEY_ENVIRONMENT,
    secret_environment: str = SECRET_ENVIRONMENT,
) -> dict[str, object]:
    """Return credential availability and provenance without secret values."""

    key_from_environment = os.environ.get(key_environment, "").strip()
    secret_from_environment = os.environ.get(secret_environment, "").strip()
    key = key_from_environment or _read_macos_keychain(KEYCHAIN_KEY_SERVICE)
    secret = secret_from_environment or _read_macos_keychain(KEYCHAIN_SECRET_SERVICE)
    if key_from_environment and secret_from_environment:
        source = "environment"
    elif key and secret and not key_from_environment and not secret_from_environment:
        source = "macos_keychain"
    elif key or secret:
        source = "mixed_or_incomplete"
    else:
        source = "unavailable"
    return {"key": bool(key), "secret": bool(secret), "source": source}


def load_paper_credentials(
    *,
    key_environment: str = KEY_ENVIRONMENT,
    secret_environment: str = SECRET_ENVIRONMENT,
) -> PaperCredentialPair:
    """Resolve Alpaca paper credentials, preferring an explicit environment."""

    presence = paper_credential_presence(
        key_environment=key_environment,
        secret_environment=secret_environment,
    )
    key = os.environ.get(key_environment, "").strip() or _read_macos_keychain(
        KEYCHAIN_KEY_SERVICE
    )
    secret = os.environ.get(secret_environment, "").strip() or _read_macos_keychain(
        KEYCHAIN_SECRET_SERVICE
    )
    if not key or not secret:
        raise PaperCredentialError(
            "Alpaca paper credentials are missing from both the configured "
            "environment and macOS Keychain"
        )
    return PaperCredentialPair(
        key_id=key,
        secret_key=secret,
        source=str(presence["source"]),
    )


__all__ = [
    "PaperCredentialError",
    "PaperCredentialPair",
    "load_paper_credentials",
    "paper_credential_presence",
]
