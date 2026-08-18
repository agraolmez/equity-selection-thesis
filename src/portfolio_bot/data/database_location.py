"""Resolve database paths recorded before the 2026-07-15 root rename."""

from __future__ import annotations

import os
from pathlib import Path


HISTORICAL_DATABASE_ROOT = Path("/Users/agraemirolmez/Desktop/database")
CANONICAL_DATABASE_ROOT = Path("/Users/agraemirolmez/Desktop/Thesis")


def lexical_absolute(path: str | Path) -> Path:
    """Return an absolute lexical path without requiring the target to exist."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def resolve_recorded_database_path(
    recorded_path: str | Path,
    *,
    current_database_root: str | Path,
) -> Path:
    """Map a sealed old-root path onto the current root, preserving its suffix.

    Paths outside the historical database root are returned unchanged. This
    permits immutable manifests to retain their original provenance while new
    readers use the relocation-bound active root.
    """

    recorded = lexical_absolute(recorded_path)
    current = lexical_absolute(current_database_root)
    try:
        relative = recorded.relative_to(HISTORICAL_DATABASE_ROOT)
    except ValueError:
        return recorded
    if relative == Path("."):
        return current
    return lexical_absolute(current / relative)
