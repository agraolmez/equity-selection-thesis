"""Authenticated missing-value fallbacks for non-equivalent live providers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


class LiveMissingFallbackError(RuntimeError):
    """A requested provider-mismatch fallback is outside the frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinals(ranges: list[list[int]]) -> list[int]:
    return [value for start, end in ranges for value in range(int(start), int(end) + 1)]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def materialize_missing_fallback(
    *,
    config_path: Path,
    group: str,
    universe_path: Path,
    registry_path: Path,
    output: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_live_missing_fallbacks_v1":
        raise LiveMissingFallbackError("missing-fallback config schema differs")
    specification = config["allowed_groups"].get(group)
    if specification is None:
        raise LiveMissingFallbackError(f"missing fallback is not authorized for {group}")
    universe = pd.read_parquet(universe_path)
    registry = pd.read_parquet(registry_path)
    if universe.empty or universe["live_security_id"].duplicated().any():
        raise LiveMissingFallbackError("fallback universe grain is invalid")
    ordinals = _ordinals(specification["ordinal_ranges"])
    selected = registry.loc[registry["model_ordinal"].isin(ordinals)].sort_values(
        "model_ordinal", kind="mergesort"
    )
    if len(selected) != len(ordinals):
        raise LiveMissingFallbackError("fallback feature ordinals differ from the registry")
    result = universe[["live_security_id"]].copy()
    constants = dict(specification.get("categorical_constants", {}))
    for row in selected.itertuples(index=False):
        name = str(row.feature_name)
        if name in constants:
            result[name] = str(constants[name])
        elif str(row.kind).startswith("categorical"):
            result[name] = None
        else:
            result[name] = np.nan
    data_path = output / f"{group}_features.parquet"
    _atomic_parquet(data_path, result)
    receipt = {
        "schema": "research_v6_8_live_missing_fallback_receipt_v1",
        "status": "PASS",
        "group": group,
        "rows": int(len(result)),
        "feature_count": int(len(selected)),
        "disposition": "AUTHENTIC_FROZEN_MISSING_FALLBACK",
        "reason": str(specification["reason"]),
        "proxy_values_used": False,
        "output_sha256": _sha256(data_path),
        "config_sha256": _sha256(config_path),
        "universe_sha256": _sha256(universe_path),
        "registry_sha256": _sha256(registry_path),
    }
    _atomic_json(output / "receipt.json", receipt)
    return receipt


__all__ = ["LiveMissingFallbackError", "materialize_missing_fallback"]
