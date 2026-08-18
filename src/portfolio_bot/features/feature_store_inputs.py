"""Resolve and authenticate the verified Stage 3–5 inputs for Stage 6."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from portfolio_bot.features.feature_store_contract import (
    DEFAULT_POLICY,
    FeatureStoreContractError,
    PROJECT_ROOT,
    sha256_file,
    validate_feature_store_contract,
)


@dataclass(frozen=True)
class InputTable:
    gate: str
    name: str
    path: Path
    sha256: str
    size_bytes: int
    rows: int
    row_groups: int


@dataclass(frozen=True)
class FeatureStoreInputs:
    database_root: Path
    tables: Mapping[str, InputTable]

    def path(self, name: str) -> Path:
        try:
            return self.tables[name].path
        except KeyError as exc:
            raise FeatureStoreContractError(f"Unregistered Stage 6 input table: {name}") from exc


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise FeatureStoreContractError(f"Expected JSON object: {path}")
    return value


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _output_receipts(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    outputs = manifest.get("outputs")
    if isinstance(outputs, list):
        candidates.extend(item for item in outputs if isinstance(item, dict))
    elif isinstance(outputs, dict):
        for key, item in outputs.items():
            if isinstance(item, dict):
                candidates.append({"table_key": key, **item})
    output = manifest.get("output")
    if isinstance(output, dict):
        candidates.append(output)
    receipts: dict[str, dict[str, Any]] = {}
    for item in candidates:
        path = item.get("path")
        if not isinstance(path, str) or not path.endswith(".parquet"):
            continue
        name = Path(path).name
        if name in receipts:
            raise FeatureStoreContractError(f"Duplicate upstream Parquet basename: {name}")
        receipts[name] = dict(item)
    return receipts


def _relocated_path(
    value: str,
    *,
    database_root: Path,
    historical_root: Path,
    interim_path: str | None,
) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        try:
            relative = raw.relative_to(historical_root)
        except ValueError:
            try:
                relative = raw.relative_to(database_root)
            except ValueError as exc:
                raise FeatureStoreContractError(
                    f"Upstream output is outside the authenticated database roots: {value}"
                ) from exc
        result = database_root / relative
    elif len(raw.parts) > 1:
        result = database_root / raw
    elif interim_path is not None:
        result = database_root / interim_path / raw
    else:
        raise FeatureStoreContractError(f"Cannot resolve upstream output path: {value}")
    resolved = result.resolve()
    try:
        resolved.relative_to(database_root.resolve())
    except ValueError as exc:
        raise FeatureStoreContractError(f"Resolved input escaped active database root: {value}") from exc
    return resolved


def resolve_input_catalog(
    policy_path: Path = DEFAULT_POLICY,
    *,
    authenticate_hashes: bool = True,
) -> FeatureStoreInputs:
    """Return the exact verified Stage 6 allowlist and fail closed on drift."""

    validate_feature_store_contract(policy_path)
    policy = _toml(policy_path)
    semantic = _toml(_project_path(policy["semantic_contract"]["policy_path"]))
    database_root = Path(policy["database"]["root"]).resolve()
    historical_root = Path(semantic["database"]["historical_root"]).resolve()
    relocation = _json(_project_path(policy["database"]["relocation_receipt"]))
    if relocation.get("new_database_root") != database_root.as_posix():
        raise FeatureStoreContractError("Relocation receipt does not identify the active database root")
    if relocation.get("old_database_root") != historical_root.as_posix():
        raise FeatureStoreContractError("Relocation receipt does not identify the historical database root")

    tables: dict[str, InputTable] = {}
    for gate, specification in semantic["upstream"].items():
        manifest_path = _project_path(specification["primary_manifest"])
        manifest = _json(manifest_path)
        receipts = _output_receipts(manifest)
        interim_path = manifest.get("interim_path")
        if interim_path is not None and not isinstance(interim_path, str):
            raise FeatureStoreContractError(f"Invalid interim_path for upstream gate {gate}")
        for name in specification["required_tables"]:
            if name in tables:
                raise FeatureStoreContractError(f"Stage 6 input table is registered twice: {name}")
            try:
                receipt = receipts[name]
            except KeyError as exc:
                raise FeatureStoreContractError(
                    f"Required Stage 6 input is absent from verified manifest: {name}"
                ) from exc
            path = _relocated_path(
                str(receipt["path"]),
                database_root=database_root,
                historical_root=historical_root,
                interim_path=interim_path,
            )
            if not path.is_file() or path.is_symlink():
                raise FeatureStoreContractError(f"Missing or symlinked Stage 6 input: {path}")
            observed_size = path.stat().st_size
            expected_size = receipt.get("size_bytes")
            if not isinstance(expected_size, int) or observed_size != expected_size:
                raise FeatureStoreContractError(f"Stage 6 input size differs: {name}")
            expected_hash = receipt.get("sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise FeatureStoreContractError(f"Stage 6 input hash receipt is missing: {name}")
            if authenticate_hashes and sha256_file(path) != expected_hash:
                raise FeatureStoreContractError(f"Stage 6 input hash differs: {name}")
            parquet = pq.ParquetFile(path)
            rows = parquet.metadata.num_rows
            expected_rows = receipt.get("rows")
            if expected_rows is not None and rows != int(expected_rows):
                raise FeatureStoreContractError(f"Stage 6 input row count differs: {name}")
            tables[name] = InputTable(
                gate=str(gate),
                name=name,
                path=path,
                sha256=expected_hash,
                size_bytes=observed_size,
                rows=rows,
                row_groups=parquet.metadata.num_row_groups,
            )

    expected_names = {
        name
        for specification in semantic["upstream"].values()
        for name in specification["required_tables"]
    }
    if set(tables) != expected_names or len(tables) != 26:
        raise FeatureStoreContractError("Stage 6 input allowlist is not the exact 26-table contract")
    prohibited = {
        name
        for specification in semantic["upstream"].values()
        for name in specification.get("prohibited_tables", [])
    }
    if set(tables).intersection(prohibited):
        raise FeatureStoreContractError("A prohibited upstream staging table entered Stage 6")
    return FeatureStoreInputs(database_root=database_root, tables=dict(sorted(tables.items())))
