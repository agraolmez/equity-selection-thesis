"""Audited, router-only cache migration for a Stage 6 identity-only change.

This module exists for the narrow case in which the complete CRSP signal
router was produced with identical inputs, configuration, and CRSP-producing
code, but the enclosing Stage 6 build identity changed because publication
infrastructure changed.  It never copies family outputs and never writes raw
inputs.

The caller must stop the Stage 6 build before invoking ``migrate_router_cache``.
The migration fails closed unless the stopped build left a nonempty,
authenticated partial router whose complete committed prefix is exactly equal
to the old complete router.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import tomllib
import uuid
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import pyarrow as pa

from portfolio_bot.features.crsp_features import CRSP_FEATURE_NAMES
from portfolio_bot.features.feature_lineage import VALUE_STATUS_SET
from portfolio_bot.features.feature_store_contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
)
from portfolio_bot.features.feature_store_materialization import (
    FeatureStoreBuildIdentity,
)
from portfolio_bot.features.materialize_crsp_routing import (
    _spool_schema,
    spool_crsp_signal_router,
)
from portfolio_bot.features.phase_03_contract import canonical_json_bytes


_DATABASE_NAME = "crsp_signal_router.duckdb"
_RECEIPT_NAME = "migration_receipt.json"
_SAFE_COMPONENT = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONFIRMATION = "I CONFIRM THE STAGE 6 BUILD IS STOPPED"
_TABLES = ("routing_metadata", "unfinalized_crsp")
_METADATA_SCHEMA_NAMES = (
    "singleton",
    "schema_version",
    "source_id",
    "expected_rows",
    "row_count",
    "last_permno",
    "last_signal_date",
    "status",
)
_METADATA_SCHEMA_TYPES = (
    pa.bool_(),
    pa.int32(),
    pa.string(),
    pa.int64(),
    pa.int64(),
    pa.int64(),
    pa.date32(),
    pa.string(),
)
_METADATA_KEYS = (
    "schema_version",
    "source_id",
    "expected_rows",
    "row_count",
    "last_permno",
    "last_signal_date",
    "status",
)
_ALLOWED_CHANGED_CODE = frozenset(
    {
        "src/portfolio_bot/data/secure_publication.py",
        "src/portfolio_bot/features/feature_store_materialization.py",
        "src/portfolio_bot/features/feature_store_family_orchestration.py",
        "src/portfolio_bot/features/feature_store_publication.py",
    }
)
_CRSP_PRODUCING_CODE = frozenset(
    {
        "src/portfolio_bot/features/phase_03_contract.py",
        "src/portfolio_bot/features/feature_lineage.py",
        "src/portfolio_bot/features/feature_math.py",
        "src/portfolio_bot/features/feature_store_contract.py",
        "src/portfolio_bot/features/feature_store_inputs.py",
        "src/portfolio_bot/features/crsp_features.py",
        "src/portfolio_bot/features/fast_crsp_features.py",
        "src/portfolio_bot/features/crsp_cross_section.py",
        "src/portfolio_bot/features/materialize_crsp.py",
        "src/portfolio_bot/features/materialize_crsp_streaming.py",
        "src/portfolio_bot/features/materialize_crsp_routing.py",
    }
)


class RouterCacheMigrationError(RuntimeError):
    """Raised when any router-migration gate differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RouterCacheMigrationError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RouterCacheMigrationError(
            f"Migration identity receipt is unreadable: {path}"
        ) from exc
    _require(isinstance(value, dict), "Migration identity receipt is not an object")
    return value


def _safe_component(value: str, label: str) -> str:
    _require(
        isinstance(value, str)
        and _SAFE_COMPONENT.fullmatch(value) is not None,
        f"Unsafe {label}",
    )
    return value


def _real_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    _require(absolute.is_dir() and not absolute.is_symlink(), f"Unsafe {label}")
    return absolute


def _real_file(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        observed = os.lstat(absolute)
    except OSError as exc:
        raise RouterCacheMigrationError(f"Missing {label}: {absolute}") from exc
    _require(
        stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        f"Unsafe {label}: {absolute}",
    )
    return absolute


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, dt.date) else value


def _metadata_payload(row: Sequence[Any]) -> dict[str, Any]:
    return {key: _iso(value) for key, value in zip(_METADATA_KEYS, row)}


def _identity_content(
    identity: FeatureStoreBuildIdentity,
    *,
    code_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": identity.contract_id,
        "policy_id": identity.policy_id,
        "configuration_hashes": identity.configuration_hash_map,
        "code_hashes": (
            identity.code_hash_map if code_hashes is None else dict(code_hashes)
        ),
        "input_receipts": [
            item.as_dict() for item in identity.input_receipts
        ],
    }


def _fingerprint(content: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def _hash_map(value: Any, label: str) -> dict[str, str]:
    _require(isinstance(value, dict) and value, f"{label} is not a nonempty object")
    result: dict[str, str] = {}
    for key, digest in value.items():
        _require(
            isinstance(key, str)
            and isinstance(digest, str)
            and _SHA256.fullmatch(digest) is not None,
            f"{label} contains an invalid hash",
        )
        result[key] = digest
    return result


def validate_identity_equivalence(
    *,
    old_receipt: Mapping[str, Any],
    old_build_id: str,
    new_identity: FeatureStoreBuildIdentity,
    expected_new_build_id: str,
) -> dict[str, Any]:
    """Prove that only the reviewed four infrastructure hashes changed."""

    _safe_component(old_build_id, "old build ID")
    _safe_component(expected_new_build_id, "new build ID")
    _require(
        new_identity.build_id == expected_new_build_id,
        "Authenticated new build identity differs from the expected build",
    )
    _require(
        old_receipt.get("build_id") == old_build_id,
        "Old identity receipt build ID differs",
    )
    old_fingerprint = old_receipt.get("content_fingerprint")
    _require(
        isinstance(old_fingerprint, str)
        and _SHA256.fullmatch(old_fingerprint) is not None
        and old_build_id == f"phase03-features-{old_fingerprint[:24]}",
        "Old identity receipt fingerprint differs",
    )
    _require(
        old_receipt.get("contract_id") == new_identity.contract_id
        and old_receipt.get("policy_id") == new_identity.policy_id,
        "Old and new contract or policy bindings differ",
    )
    _require(
        len(new_identity.input_receipts) == 26
        and len({item.name for item in new_identity.input_receipts}) == 26,
        "New identity does not contain the frozen 26-input allowlist",
    )
    old_input_digest = old_receipt.get("input_receipts_sha256")
    _require(
        old_input_digest == new_identity.input_receipts_sha256,
        "Old and new input-receipt digests differ",
    )
    old_configuration = _hash_map(
        old_receipt.get("configuration_hashes"),
        "Old configuration hashes",
    )
    _require(
        old_configuration == new_identity.configuration_hash_map,
        "Old and new configuration hashes differ",
    )
    _require(
        len(old_configuration) == 8,
        "Frozen configuration allowlist does not contain eight files",
    )
    old_code = _hash_map(old_receipt.get("code_hashes"), "Old code hashes")
    new_code = new_identity.code_hash_map
    _require(
        len(old_code) == len(new_code) == 26
        and set(old_code) == set(new_code),
        "Old and new frozen code allowlists differ",
    )
    changed = frozenset(
        label for label in new_code if old_code[label] != new_code[label]
    )
    _require(
        changed == _ALLOWED_CHANGED_CODE,
        "Code changes are not the exact reviewed infrastructure-only set",
    )
    _require(
        _CRSP_PRODUCING_CODE.issubset(new_code)
        and all(old_code[label] == new_code[label] for label in _CRSP_PRODUCING_CODE),
        "CRSP-producing code differs",
    )
    reconstructed_old = _fingerprint(
        _identity_content(new_identity, code_hashes=old_code)
    )
    reconstructed_new = _fingerprint(_identity_content(new_identity))
    _require(
        reconstructed_old == old_fingerprint,
        "Old content fingerprint cannot be reconstructed",
    )
    _require(
        reconstructed_new == new_identity.content_fingerprint
        and new_identity.build_id
        == f"phase03-features-{reconstructed_new[:24]}",
        "New content fingerprint cannot be reconstructed",
    )
    return {
        "old_build_id": old_build_id,
        "old_content_fingerprint": reconstructed_old,
        "new_build_id": new_identity.build_id,
        "new_content_fingerprint": reconstructed_new,
        "input_receipts_sha256": new_identity.input_receipts_sha256,
        "shared_ordered_input_receipts": [
            item.as_dict() for item in new_identity.input_receipts
        ],
        "old_configuration_hashes": old_configuration,
        "new_configuration_hashes": new_identity.configuration_hash_map,
        "old_code_hashes": old_code,
        "new_code_hashes": new_code,
        "changed_code_paths": sorted(changed),
        "unchanged_crsp_producing_code_hashes": {
            label: new_code[label] for label in sorted(_CRSP_PRODUCING_CODE)
        },
    }


@dataclass(frozen=True)
class RouterContract:
    expected_rows: int
    expected_signal_dates: int
    first_signal_date: dt.date
    last_signal_date: dt.date


@dataclass(frozen=True)
class RouterSnapshot:
    path: Path
    file_sha256: str
    size_bytes: int
    metadata: Mapping[str, Any]
    table_names: tuple[str, ...]
    unfinalized_schema_sha256: str
    metadata_schema_sha256: str
    schema_fields: int
    rows: int
    distinct_keys: int
    signal_dates: int
    first_signal_date: dt.date
    last_signal_date: dt.date
    first_permno: int
    last_permno: int
    canonical_last_key: tuple[int, dt.date]
    null_audit_rows: int
    feature_validation_checks: int
    feature_validation_violations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "file_sha256": self.file_sha256,
            "size_bytes": self.size_bytes,
            "metadata": dict(self.metadata),
            "table_names": list(self.table_names),
            "unfinalized_schema_sha256": self.unfinalized_schema_sha256,
            "metadata_schema_sha256": self.metadata_schema_sha256,
            "schema_fields": self.schema_fields,
            "rows": self.rows,
            "distinct_keys": self.distinct_keys,
            "signal_dates": self.signal_dates,
            "first_signal_date": self.first_signal_date.isoformat(),
            "last_signal_date": self.last_signal_date.isoformat(),
            "first_permno": self.first_permno,
            "last_permno": self.last_permno,
            "canonical_last_key": {
                "permno": self.canonical_last_key[0],
                "signal_date": self.canonical_last_key[1].isoformat(),
            },
            "null_audit_rows": self.null_audit_rows,
            "feature_validation_checks": self.feature_validation_checks,
            "feature_validation_violations": self.feature_validation_violations,
        }


def _feature_validation_sql() -> tuple[str, int]:
    allowed = ",".join(f"'{value}'" for value in sorted(VALUE_STATUS_SET))
    expressions: list[str] = []
    for name in CRSP_FEATURE_NAMES:
        value = f'"{name}"'
        status = f'"{name}__value_status"'
        observation = f'"{name}__observation"'
        availability = f'"{name}__availability"'
        stale = f'"{name}__stale_after"'
        expressions.extend(
            (
                f"sum(CASE WHEN {status} NOT IN ({allowed}) "
                f"OR {status} IS NULL THEN 1 ELSE 0 END)",
                f"sum(CASE WHEN ({value} IS NULL) = "
                f"({status} = 'PRESENT') THEN 1 ELSE 0 END)",
                f"sum(CASE WHEN {status} = 'PRESENT' AND "
                f"({observation} IS NULL OR {availability} IS NULL) "
                f"THEN 1 ELSE 0 END)",
                f"sum(CASE WHEN {status} <> 'PRESENT' AND "
                f"({observation} IS NOT NULL OR {availability} IS NOT NULL "
                f"OR {stale} IS NOT NULL) THEN 1 ELSE 0 END)",
                f"sum(CASE WHEN {status} = 'PRESENT' AND "
                f"({observation} > signal_date OR "
                f"{availability} > signal_date OR "
                f"({stale} IS NOT NULL AND {stale} < signal_date)) "
                f"THEN 1 ELSE 0 END)",
                f"sum(CASE WHEN {value} IS NOT NULL "
                f"AND NOT isfinite({value}) THEN 1 ELSE 0 END)",
            )
        )
    return "SELECT " + ",".join(expressions) + " FROM unfinalized_crsp", len(
        expressions
    )


def inspect_router(
    path: Path,
    *,
    contract: RouterContract,
    expected_source_id: str,
    expected_status: str,
    require_complete_scope: bool,
) -> RouterSnapshot:
    """Authenticate one stopped router without changing it."""

    database = _real_file(path, "router database")
    try:
        connection = duckdb.connect(database.as_posix(), read_only=True)
    except duckdb.Error as exc:
        raise RouterCacheMigrationError(
            "Router is locked or unreadable; the Stage 6 build may still be running"
        ) from exc
    try:
        tables = tuple(
            row[0]
            for row in connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                ORDER BY table_name
                """
            ).fetchall()
        )
        _require(tables == tuple(sorted(_TABLES)), "Router table set differs")
        metadata_rows = connection.execute(
            """
            SELECT
                schema_version, source_id, expected_rows, row_count,
                last_permno, last_signal_date, status
            FROM routing_metadata
            WHERE singleton = TRUE
            """
        ).fetchall()
        metadata_count = int(
            connection.execute(
                "SELECT count(*) FROM routing_metadata"
            ).fetchone()[0]
        )
        _require(
            len(metadata_rows) == metadata_count == 1,
            "Router metadata cardinality differs",
        )
        metadata = _metadata_payload(metadata_rows[0])
        _require(
            metadata["schema_version"] == 1
            and metadata["source_id"] == expected_source_id
            and metadata["expected_rows"] == contract.expected_rows
            and metadata["status"] == expected_status,
            "Router metadata identity or state differs",
        )
        unfinalized_schema = connection.execute(
            "SELECT * FROM unfinalized_crsp LIMIT 0"
        ).fetch_arrow_table().schema
        metadata_schema = connection.execute(
            "SELECT * FROM routing_metadata LIMIT 0"
        ).fetch_arrow_table().schema
        _require(
            tuple(metadata_schema.names) == _METADATA_SCHEMA_NAMES
            and tuple(field.type for field in metadata_schema)
            == _METADATA_SCHEMA_TYPES,
            "Router metadata schema differs",
        )
        expected_schema = _spool_schema()
        _require(
            unfinalized_schema.names == expected_schema.names
            and tuple(field.type for field in unfinalized_schema)
            == tuple(field.type for field in expected_schema),
            "Router unfinalized schema differs",
        )
        aggregate = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT (permno, signal_date)),
                count(DISTINCT signal_date),
                min(signal_date),
                max(signal_date),
                min(permno),
                max(permno),
                count(*) FILTER (
                    WHERE signal_date IS NULL
                       OR permno IS NULL
                       OR permco IS NULL
                       OR long_eligible IS NULL
                       OR short_eligible IS NULL
                       OR primary_exchange IS NULL
                       OR share_type IS NULL
                )
            FROM unfinalized_crsp
            """
        ).fetchone()
        _require(aggregate is not None and aggregate[0] > 0, "Router is empty")
        last_key = connection.execute(
            """
            SELECT permno, signal_date
            FROM unfinalized_crsp
            ORDER BY permno DESC, signal_date DESC
            LIMIT 1
            """
        ).fetchone()
        _require(last_key is not None, "Router has no canonical last key")
        validation_sql, validation_checks = _feature_validation_sql()
        violations = connection.execute(validation_sql).fetchone()
        _require(violations is not None, "Router feature validation failed")
        total_violations = sum(int(value) for value in violations)
    except duckdb.Error as exc:
        raise RouterCacheMigrationError("Router SQL validation failed") from exc
    finally:
        connection.close()
    rows = int(aggregate[0])
    distinct_keys = int(aggregate[1])
    _require(rows == distinct_keys, "Router signal key is not unique")
    _require(
        metadata["row_count"] == rows
        and metadata["last_permno"] == int(last_key[0])
        and metadata["last_signal_date"] == last_key[1].isoformat(),
        "Router metadata checkpoint differs from its table",
    )
    _require(
        int(aggregate[7]) == 0 and total_violations == 0,
        "Router contains invalid audit, feature, or lineage values",
    )
    if require_complete_scope:
        _require(
            rows == contract.expected_rows
            and int(aggregate[2]) == contract.expected_signal_dates
            and aggregate[3] == contract.first_signal_date
            and aggregate[4] == contract.last_signal_date,
            "Complete router scope differs from the frozen contract",
        )
    else:
        _require(
            0 < rows < contract.expected_rows,
            "Stopped target is not a nonempty partial router",
        )
    return RouterSnapshot(
        path=database,
        file_sha256=_sha256_file(database),
        size_bytes=database.stat().st_size,
        metadata=metadata,
        table_names=tables,
        unfinalized_schema_sha256=_schema_sha256(unfinalized_schema),
        metadata_schema_sha256=_schema_sha256(metadata_schema),
        schema_fields=len(unfinalized_schema),
        rows=rows,
        distinct_keys=distinct_keys,
        signal_dates=int(aggregate[2]),
        first_signal_date=aggregate[3],
        last_signal_date=aggregate[4],
        first_permno=int(aggregate[5]),
        last_permno=int(aggregate[6]),
        canonical_last_key=(int(last_key[0]), last_key[1]),
        null_audit_rows=int(aggregate[7]),
        feature_validation_checks=validation_checks,
        feature_validation_violations=total_violations,
    )


def _batch_bytes(batch: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def _reader_batches(
    connection: duckdb.DuckDBPyConnection,
    sql: str,
    parameters: Sequence[Any],
    batch_rows: int,
) -> Iterable[pa.RecordBatch]:
    return connection.execute(sql, list(parameters)).fetch_record_batch(batch_rows)


def compare_router_rows(
    left: Path,
    right: Path,
    *,
    prefix_last_key: tuple[int, dt.date] | None = None,
    batch_rows: int = 1_024,
) -> dict[str, Any]:
    """Compare every value and null in canonical key order, with bounded memory."""

    _require(
        isinstance(batch_rows, int)
        and not isinstance(batch_rows, bool)
        and batch_rows > 0,
        "Comparison batch size must be positive",
    )
    left_path = _real_file(left, "left comparison router")
    right_path = _real_file(right, "right comparison router")
    left_connection = duckdb.connect(left_path.as_posix(), read_only=True)
    right_connection = duckdb.connect(right_path.as_posix(), read_only=True)
    where = ""
    parameters: tuple[Any, ...] = ()
    if prefix_last_key is not None:
        where = (
            "WHERE permno < ? OR "
            "(permno = ? AND signal_date <= ?)"
        )
        parameters = (
            prefix_last_key[0],
            prefix_last_key[0],
            prefix_last_key[1],
        )
    left_sql = (
        f"SELECT * FROM unfinalized_crsp {where} "
        "ORDER BY permno, signal_date"
    )
    right_sql = (
        "SELECT * FROM unfinalized_crsp ORDER BY permno, signal_date"
    )
    left_digest = hashlib.sha256()
    right_digest = hashlib.sha256()
    rows = 0
    batches = 0
    try:
        left_reader = _reader_batches(
            left_connection, left_sql, parameters, batch_rows
        )
        right_reader = _reader_batches(
            right_connection, right_sql, (), batch_rows
        )
        for ordinal, pair in enumerate(
            zip_longest(left_reader, right_reader),
            start=1,
        ):
            left_batch, right_batch = pair
            _require(
                left_batch is not None and right_batch is not None,
                "Router comparison cardinality differs",
            )
            _require(
                left_batch.equals(right_batch),
                f"Router rows differ in canonical comparison batch {ordinal}",
            )
            left_bytes = _batch_bytes(left_batch)
            right_bytes = _batch_bytes(right_batch)
            left_digest.update(len(left_bytes).to_bytes(8, "big"))
            left_digest.update(left_bytes)
            right_digest.update(len(right_bytes).to_bytes(8, "big"))
            right_digest.update(right_bytes)
            rows += left_batch.num_rows
            batches += 1
    except duckdb.Error as exc:
        raise RouterCacheMigrationError("Router row comparison failed") from exc
    finally:
        left_connection.close()
        right_connection.close()
    _require(
        left_digest.digest() == right_digest.digest(),
        "Router canonical stream hashes differ",
    )
    return {
        "rows_compared": rows,
        "batches_compared": batches,
        "batch_rows": batch_rows,
        "canonical_arrow_stream_sha256": left_digest.hexdigest(),
        "differences": 0,
    }


def _recover_stopped_partial(database: Path) -> None:
    """Acquire DuckDB's writer lock, recover any WAL, and release it."""

    try:
        connection = duckdb.connect(database.as_posix(), read_only=False)
    except duckdb.Error as exc:
        raise RouterCacheMigrationError(
            "Target router is locked; stop the Stage 6 build before migration"
        ) from exc
    try:
        connection.execute("CHECKPOINT")
    except duckdb.Error as exc:
        raise RouterCacheMigrationError(
            "Stopped partial router could not be recovered"
        ) from exc
    finally:
        connection.close()
    _require(
        not database.with_name(database.name + ".wal").exists(),
        "Stopped partial router still has a WAL after checkpoint",
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_file_receipts(path: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        if child.is_symlink():
            raise RouterCacheMigrationError(
                "Stopped partial directory contains a symlink"
            )
        if child.is_file():
            receipts.append(
                {
                    "name": child.name,
                    "sha256": _sha256_file(child),
                    "size_bytes": child.stat().st_size,
                }
            )
    _require(receipts, "Stopped partial directory contains no files")
    _require(
        {item["name"] for item in receipts}.issubset(
            {_DATABASE_NAME, f"{_DATABASE_NAME}.wal"}
        )
        and any(item["name"] == _DATABASE_NAME for item in receipts),
        "Stopped partial directory file set differs",
    )
    return receipts


def _receipt_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require("receipt_hash" not in payload, "Receipt payload already has a hash")
    digest = hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()
    return {
        **dict(payload),
        "receipt_hash": {
            "algorithm": "sha256",
            "canonicalization": "phase03-canonical-json-v1",
            "scope": "all-top-level-fields-except-receipt_hash",
            "sha256": digest,
        },
    }


def authenticate_migration_receipt(receipt: Mapping[str, Any]) -> None:
    value = receipt.get("receipt_hash")
    _require(
        isinstance(value, dict)
        and value.get("algorithm") == "sha256"
        and value.get("canonicalization") == "phase03-canonical-json-v1"
        and value.get("scope") == "all-top-level-fields-except-receipt_hash"
        and isinstance(value.get("sha256"), str),
        "Migration receipt hash contract differs",
    )
    payload = {key: item for key, item in receipt.items() if key != "receipt_hash"}
    observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    _require(
        observed == value["sha256"],
        "Migration receipt self-hash differs",
    )


def _publish_receipt(path: Path, payload: Mapping[str, Any]) -> str:
    receipt = _receipt_with_hash(payload)
    authenticate_migration_receipt(receipt)
    destination = Path(os.path.abspath(path.expanduser()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(
        destination.parent.is_dir()
        and not destination.parent.is_symlink()
        and not destination.exists(),
        "Migration receipt destination is unsafe or already exists",
    )
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        encoded = canonical_json_bytes(receipt) + b"\n"
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)
    return _sha256_file(destination)


@dataclass(frozen=True)
class RouterMigrationSettings:
    router_root: Path
    artifact_root: Path
    old_build_id: str
    expected_new_build_id: str
    old_identity_receipt: Path
    build_stopped_confirmation: str
    comparison_batch_rows: int = 1_024


@dataclass(frozen=True)
class RouterMigrationResult:
    old_build_id: str
    new_build_id: str
    target_directory: Path
    quarantined_partial_directory: Path
    receipt_path: Path
    receipt_file_sha256: str


def migrate_router_cache(
    settings: RouterMigrationSettings,
    *,
    new_identity: FeatureStoreBuildIdentity,
    contract: RouterContract,
) -> RouterMigrationResult:
    """Migrate one stopped partial target after all equivalence gates pass."""

    _require(
        settings.build_stopped_confirmation == _CONFIRMATION,
        "Explicit stopped-build confirmation is required",
    )
    old_build_id = _safe_component(settings.old_build_id, "old build ID")
    new_build_id = _safe_component(
        settings.expected_new_build_id, "new build ID"
    )
    router_root = _real_directory(settings.router_root, "router cache root")
    artifact_root = Path(os.path.abspath(settings.artifact_root.expanduser()))
    _require(
        artifact_root == PROJECT_ROOT / "artifacts" / "phase_03_features"
        or PROJECT_ROOT not in artifact_root.parents,
        "Artifact root is ambiguous",
    )
    source_directory = router_root / old_build_id
    target_directory = router_root / new_build_id
    source_database = _real_file(
        source_directory / _DATABASE_NAME, "old complete router"
    )
    target_database = _real_file(
        target_directory / _DATABASE_NAME, "new partial router"
    )
    _require(
        source_directory != target_directory
        and source_directory.parent == target_directory.parent == router_root,
        "Router paths do not share the exact cache namespace",
    )
    receipt_path = (
        artifact_root
        / "router_cache_migrations"
        / new_build_id
        / _RECEIPT_NAME
    )
    _require(not receipt_path.exists(), "Migration receipt already exists")
    old_receipt_path = _real_file(
        settings.old_identity_receipt, "old identity receipt"
    )
    identity_evidence = validate_identity_equivalence(
        old_receipt=_json(old_receipt_path),
        old_build_id=old_build_id,
        new_identity=new_identity,
        expected_new_build_id=new_build_id,
    )
    identity_evidence["old_identity_receipt"] = {
        "path": old_receipt_path.as_posix(),
        "sha256": _sha256_file(old_receipt_path),
        "size_bytes": old_receipt_path.stat().st_size,
    }
    _require(
        os.stat(router_root).st_dev
        == os.stat(source_database).st_dev
        == os.stat(target_database).st_dev,
        "Source, target, and staging root are not on one filesystem",
    )

    # This writer-lock attempt is also the final active-build guard.  It may
    # recover a stopped WAL, but cannot succeed while the build owns DuckDB.
    partial_pre_recovery_files = _directory_file_receipts(target_directory)
    _recover_stopped_partial(target_database)
    old_snapshot = inspect_router(
        source_database,
        contract=contract,
        expected_source_id=identity_evidence["old_content_fingerprint"],
        expected_status="COMPLETE",
        require_complete_scope=True,
    )
    partial_snapshot = inspect_router(
        target_database,
        contract=contract,
        expected_source_id=identity_evidence["new_content_fingerprint"],
        expected_status="SPOOLING",
        require_complete_scope=False,
    )
    prefix_comparison = compare_router_rows(
        source_database,
        target_database,
        prefix_last_key=partial_snapshot.canonical_last_key,
        batch_rows=settings.comparison_batch_rows,
    )
    _require(
        prefix_comparison["rows_compared"] == partial_snapshot.rows,
        "Compared old prefix does not equal partial checkpoint cardinality",
    )

    migration_id = "router-migration-" + hashlib.sha256(
        canonical_json_bytes(
            {
                "old_build_id": old_build_id,
                "new_build_id": new_build_id,
                "old_database_sha256": old_snapshot.file_sha256,
                "partial_database_sha256": partial_snapshot.file_sha256,
            }
        )
    ).hexdigest()[:20]
    quarantine = router_root / (
        f"{new_build_id}.partial-quarantine-{migration_id}"
    )
    _require(not quarantine.exists(), "Partial quarantine path already exists")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{new_build_id}.migration-staging-",
            dir=router_root,
        )
    )
    _require(
        os.stat(staging).st_dev == os.stat(router_root).st_dev,
        "Migration staging is not on the router filesystem",
    )
    candidate_database = staging / _DATABASE_NAME
    target_published = False
    try:
        # Preserve the stopped partial before creating the candidate target.
        _require(
            target_database.stat().st_size == partial_snapshot.size_bytes
            and _sha256_file(target_database) == partial_snapshot.file_sha256,
            "Stopped partial changed after authentication",
        )
        os.rename(target_directory, quarantine)
        _fsync_directory(router_root)
        quarantined_database = quarantine / _DATABASE_NAME
        _require(
            quarantined_database.is_file()
            and _sha256_file(quarantined_database)
            == partial_snapshot.file_sha256,
            "Quarantining changed the stopped partial",
        )
        shutil.copyfile(source_database, candidate_database)
        _fsync_file(candidate_database)
        _fsync_directory(staging)
        copied_sha256 = _sha256_file(candidate_database)
        _require(
            copied_sha256 == old_snapshot.file_sha256
            and candidate_database.stat().st_size == old_snapshot.size_bytes,
            "Staged router is not a byte-exact copy of the old router",
        )
        copied_snapshot = inspect_router(
            candidate_database,
            contract=contract,
            expected_source_id=identity_evidence["old_content_fingerprint"],
            expected_status="COMPLETE",
            require_complete_scope=True,
        )
        _require(
            copied_snapshot.as_dict()
            == {
                **old_snapshot.as_dict(),
                "path": candidate_database.as_posix(),
            },
            "Staged router differs before metadata migration",
        )
        connection = duckdb.connect(candidate_database.as_posix())
        transaction_open = False
        try:
            connection.execute("BEGIN TRANSACTION")
            transaction_open = True
            changed = connection.execute(
                """
                UPDATE routing_metadata
                SET source_id = ?
                WHERE singleton = TRUE
                  AND schema_version = 1
                  AND source_id = ?
                  AND expected_rows = ?
                  AND row_count = ?
                  AND last_permno = ?
                  AND last_signal_date = ?
                  AND status = 'COMPLETE'
                RETURNING source_id
                """,
                [
                    identity_evidence["new_content_fingerprint"],
                    identity_evidence["old_content_fingerprint"],
                    contract.expected_rows,
                    contract.expected_rows,
                    old_snapshot.canonical_last_key[0],
                    old_snapshot.canonical_last_key[1],
                ],
            ).fetchall()
            _require(
                changed
                == [(identity_evidence["new_content_fingerprint"],)],
                "Router metadata update did not change exactly one bound row",
            )
            connection.execute("COMMIT")
            transaction_open = False
            connection.execute("CHECKPOINT")
        except Exception:
            if transaction_open:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        _require(
            not candidate_database.with_name(
                candidate_database.name + ".wal"
            ).exists(),
            "Migrated candidate retains a WAL",
        )
        migrated_snapshot = inspect_router(
            candidate_database,
            contract=contract,
            expected_source_id=identity_evidence["new_content_fingerprint"],
            expected_status="COMPLETE",
            require_complete_scope=True,
        )
        expected_metadata = {
            **dict(old_snapshot.metadata),
            "source_id": identity_evidence["new_content_fingerprint"],
        }
        _require(
            dict(migrated_snapshot.metadata) == expected_metadata,
            "Migrated metadata differs beyond source_id",
        )
        full_comparison = compare_router_rows(
            source_database,
            candidate_database,
            batch_rows=settings.comparison_batch_rows,
        )
        _require(
            full_comparison["rows_compared"] == contract.expected_rows,
            "Full router comparison cardinality differs",
        )
        _require(
            _sha256_file(source_database) == old_snapshot.file_sha256,
            "Old router changed during migration",
        )

        os.rename(staging, target_directory)
        target_published = True
        _fsync_directory(router_root)
        published_database = target_directory / _DATABASE_NAME
        published_snapshot = inspect_router(
            published_database,
            contract=contract,
            expected_source_id=identity_evidence["new_content_fingerprint"],
            expected_status="COMPLETE",
            require_complete_scope=True,
        )
        _require(
            published_snapshot.file_sha256 == migrated_snapshot.file_sha256,
            "Atomic publication changed the migrated router",
        )
        reuse_probe = spool_crsp_signal_router(
            (),
            spool_path=target_directory,
            source_id=identity_evidence["new_content_fingerprint"],
            expected_rows=contract.expected_rows,
        )
        _require(
            reuse_probe.reused_existing
            and reuse_probe.row_count == contract.expected_rows
            and len(reuse_probe.signal_dates) == contract.expected_signal_dates
            and reuse_probe.signal_dates[0] == contract.first_signal_date
            and reuse_probe.signal_dates[-1] == contract.last_signal_date,
            "Production router API did not reuse the migrated target",
        )
        _require(
            _sha256_file(source_database) == old_snapshot.file_sha256,
            "Old router changed after atomic publication",
        )
        wrapper = PROJECT_ROOT / "scripts" / "migrate_stage_06_crsp_router_cache.py"
        receipt_payload = {
            "schema_version": 1,
            "operation": "stage06_crsp_router_cache_identity_migration",
            "status": "MIGRATION_VERIFIED",
            "created_at_utc": (
                dt.datetime.now(dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            ),
            "migration_id": migration_id,
            "contract_id": new_identity.contract_id,
            "policy_id": new_identity.policy_id,
            "identity_equivalence": identity_evidence,
            "scope": {
                "router_only": True,
                "raw_files_written": 0,
                "family_feature_payloads_read_or_copied": 0,
                "old_identity_receipts_read_for_hash_binding": 1,
                "changed_logical_cells": [
                    "routing_metadata[singleton=true].source_id"
                ],
            },
            "old_complete_router": old_snapshot.as_dict(),
            "stopped_partial_router": partial_snapshot.as_dict(),
            "stopped_partial_pre_recovery_files": (
                partial_pre_recovery_files
            ),
            "partial_prefix_comparison": prefix_comparison,
            "quarantined_partial_directory": quarantine.as_posix(),
            "quarantined_partial_database_sha256": (
                _sha256_file(quarantined_database)
            ),
            "staged_copy": {
                "pre_update_database_sha256": copied_sha256,
                "pre_update_size_bytes": copied_snapshot.size_bytes,
                "post_update_database_sha256": migrated_snapshot.file_sha256,
                "post_update_size_bytes": migrated_snapshot.size_bytes,
                "metadata_before": dict(old_snapshot.metadata),
                "metadata_after": dict(migrated_snapshot.metadata),
            },
            "full_row_comparison": full_comparison,
            "published_router": published_snapshot.as_dict(),
            "production_router_reuse_probe": {
                "reused_existing": reuse_probe.reused_existing,
                "row_count": reuse_probe.row_count,
                "signal_dates": len(reuse_probe.signal_dates),
                "first_signal_date": reuse_probe.signal_dates[0].isoformat(),
                "last_signal_date": reuse_probe.signal_dates[-1].isoformat(),
            },
            "publication": {
                "same_filesystem_staging": True,
                "atomic_directory_rename": True,
                "target_directory": target_directory.as_posix(),
            },
            "implementation": {
                "module": "portfolio_bot.features.migrate_crsp_router_cache",
                "module_sha256": _sha256_file(Path(__file__)),
                "wrapper_script": wrapper.relative_to(PROJECT_ROOT).as_posix(),
                "wrapper_script_sha256": _sha256_file(wrapper),
                "python_version": sys.version,
                "duckdb_version": duckdb.__version__,
                "platform": platform.platform(),
                "receipt_hash_definition": (
                    "sha256 of phase03 canonical JSON over every top-level "
                    "field except receipt_hash"
                ),
            },
        }
        receipt_file_sha256 = _publish_receipt(
            receipt_path,
            receipt_payload,
        )
    except Exception:
        # Never leave an unreceipted migrated target in the live namespace.
        if target_published and target_directory.exists():
            failed = router_root / (
                f".{new_build_id}.unreceipted-{migration_id}"
            )
            if not failed.exists():
                os.rename(target_directory, failed)
        if not target_directory.exists() and quarantine.exists():
            os.rename(quarantine, target_directory)
        _fsync_directory(router_root)
        raise
    return RouterMigrationResult(
        old_build_id=old_build_id,
        new_build_id=new_build_id,
        target_directory=target_directory,
        quarantined_partial_directory=quarantine,
        receipt_path=receipt_path,
        receipt_file_sha256=receipt_file_sha256,
    )


def contract_from_policy(policy_path: Path = DEFAULT_POLICY) -> RouterContract:
    """Load the frozen router cardinality and date scope."""

    with policy_path.open("rb") as handle:
        policy = tomllib.load(handle)
    scope = policy["scope"]
    try:
        result = RouterContract(
            expected_rows=int(scope["required_spine_rows"]),
            expected_signal_dates=int(scope["required_signal_dates"]),
            first_signal_date=dt.date.fromisoformat(scope["first_signal_date"]),
            last_signal_date=dt.date.fromisoformat(scope["last_signal_date"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RouterCacheMigrationError(
            "Frozen router scope is invalid"
        ) from exc
    _require(
        result.expected_rows > 0
        and result.expected_signal_dates > 0
        and result.first_signal_date <= result.last_signal_date,
        "Frozen router scope is invalid",
    )
    return result


def production_router_root(
    database_root: Path,
    policy_path: Path = DEFAULT_POLICY,
) -> Path:
    """Resolve the policy-bound router cache root without writing it."""

    with policy_path.open("rb") as handle:
        policy = tomllib.load(handle)
    relative = Path(str(policy["database"]["cache_root"]))
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Frozen router cache root is unsafe",
    )
    return (
        Path(os.path.abspath(database_root.expanduser()))
        / relative
        / "crsp_signal_router"
    )


STOPPED_BUILD_CONFIRMATION = _CONFIRMATION
