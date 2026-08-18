"""Independent verification of the Stage-2 auxiliary primary bundle.

The verifier is intentionally a consumer of a *finished* primary bundle.  It
does not import any auxiliary builder, does not open raw source files, and does
not publish or mutate pointers.  Raw identity is authenticated against the
frozen Stage-1 recheck; all normalized Parquet outputs are scanned in full.

The returned ``independent_check_payload`` and ``differences_payload`` are
canonical, mutually bound bytes that a separate immutable publisher can
validate and freeze.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import itertools
import json
import os
import re
import tempfile
import tomllib
import weakref
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import duckdb

from portfolio_bot.data import secure_publication as secure
from portfolio_bot.data.verify_phase_02_auxiliary_linkage import (
    reconcile_linkage_payloads,
)


PRIMARY_STATUS = "auxiliary_primary_passed_pending_independent_verification"
PASS_STATUS = "verified_passed"
FAIL_STATUS = "verification_failed"
REQUIRED_TABLE_COUNT = 18
CONTRACT_ID = "phase02-auxiliary-availability-linkage-v1"
BUILD_PREFIX = "auxiliary-"
REQUIRED_TABLE_NAMES = (
    "ccm_link_interval",
    "compustat_annual_observation",
    "compustat_quarterly_observation",
    "compustat_segment_observation",
    "sec_filing_observation",
    "sec_8k_item_observation",
    "sec_13f_filing_observation",
    "sec_13f_holding_observation",
    "restatement_event_observation",
    "restatement_crsp_link",
    "optionmetrics_name_interval",
    "optionmetrics_crsp_link_interval",
    "optionmetrics_option_volume_observation",
    "optionmetrics_historical_volatility_observation",
    "ff_factor_daily_observation",
    "ff_factor_monthly_observation",
    "ps_liquidity_observation",
    "auxiliary_signal_linkage",
)
VERIFICATION_SUBDIRECTORY = "auxiliary_availability_linkage_verifications"
VERIFIED_POINTER = "LATEST_AUXILIARY_AVAILABILITY_LINKAGE_VERIFIED.json"
VERIFICATION_FILES = (
    "differences.csv",
    "independent_check.json",
    "output_hashes.sha256",
)

_COLUMN_RE = re.compile(r"([^:]+):([^!?]+(?:\([^)]*\))?)([!?])\Z")
_DECIMAL_RE = re.compile(r"decimal128\((\d+),(\d+)\)\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CUSIP8_RE = re.compile(r"[A-Z0-9]{8}\Z")

_AGGREGATE_QUALITY_GATES = (
    "before_after_link_counts_by_family_required",
    "missingness_by_family_year_exchange_domicile_adr_and_long_short_required",
    "sec_form_coverage_by_year_domicile_adr_and_form_required",
    "optionmetrics_match_genuine_zero_optionability_by_year_required",
    "factor_date_alignment_and_unit_receipt_required",
    "ps_release_lag_receipt_required",
)

_NON_TABLE_CHECK_IDS = {
    "AGGREGATE_QA_RECONSTRUCTED",
    "CONTRACT_POLICY_SCHEMA_FACTOR_AND_UPSTREAM_HASHES",
    "EXACT_REQUIRED_TABLE_SET",
    "FACTOR_LOCK_SOURCE_HASHES_MATCH_STAGE_01",
    "FACTOR_UNIT_LOCK_AGGREGATES",
    "FACTOR_UNIT_SELECTION_IS_FROZEN_DECIMAL_RATIO",
    "FROZEN_POLICY_INTERNAL_BINDINGS",
    "FROZEN_STAGE_01_EVIDENCE",
    "LINKAGE_COVERAGE_CSV_RECONSTRUCTED",
    "MANIFEST_AND_PARQUET_FILE_RECEIPTS",
    "NAMED_POLICY_GATE_RECEIPTS_RECONSTRUCTED",
    "OPTIONMETRICS_COVERAGE_CSV_RECONSTRUCTED",
    "OUTPUT_HASH_RECEIPT_RECONCILED",
    "PRIMARY_AGGREGATE_QA_RECEIPT_RECONSTRUCTED",
    "PRIMARY_CHECKS_CSV_RECONSTRUCTED",
    "PRIMARY_CONTENT_ADDRESS_AND_CONTRACT",
    "PRIMARY_MANIFEST_SIGNAL_UNIVERSE_RECEIPT",
    "PRIMARY_NAMESPACE_AND_AUTHORIZATION",
    "PRIMARY_POINTER_AND_MANIFEST",
    "PRIMARY_SUCCESS_RECEIPT",
    "PRIMARY_SUMMARY_RECEIPT_RECONSTRUCTED",
    "QA_ONLY_SOURCE_RECEIPTS_BIND_STAGE_01",
    "QUALITY_DIRECTORY_CLOSED_SET",
    "QUALITY_RECEIPT_FILE_HASHES",
    "RAW_CONTRACT_HASHES_MATCH_FROZEN_STAGE_01",
    "RAW_SOURCE_HASHES_MATCH_FROZEN_STAGE_01",
    "RAW_SOURCE_HASH_RECEIPT_RECONCILED",
    "REQUIRED_AGGREGATE_QUALITY_RECEIPTS",
    "SEC_FORM_COVERAGE_CSV_RECONSTRUCTED",
    "SIGNAL_UNIVERSE_ROW_AND_PRIMARY_KEY_PRESERVATION",
    "TABLE_COUNTS_CSV_RECONSTRUCTED",
    "TABLE_DIRECTORY_CLOSED_SET",
    "POINT_IN_TIME_FORMULAS_RECONSTRUCTED",
    "LINKAGE_PAYLOAD_RECONSTRUCTED",
    "VERIFIED_SIGNAL_UNIVERSE_EVIDENCE",
    "VERIFIED_SIGNAL_UNIVERSE_FILE_RECEIPT",
    "VERIFIED_SIGNAL_UNIVERSE_POINTER",
    "VERIFIED_TRADING_CALENDAR_EVIDENCE",
}
_REQUIRED_CHECK_IDS = frozenset(
    _NON_TABLE_CHECK_IDS
    | {f"SCHEMA_{name}" for name in REQUIRED_TABLE_NAMES}
    | {f"FULL_SCAN_{name}" for name in REQUIRED_TABLE_NAMES}
)
_QUALITY_RECONSTRUCTION_CHECK_IDS = (
    "AGGREGATE_QA_RECONSTRUCTED",
    "LINKAGE_COVERAGE_CSV_RECONSTRUCTED",
    "NAMED_POLICY_GATE_RECEIPTS_RECONSTRUCTED",
    "OPTIONMETRICS_COVERAGE_CSV_RECONSTRUCTED",
    "PRIMARY_AGGREGATE_QA_RECEIPT_RECONSTRUCTED",
    "PRIMARY_CHECKS_CSV_RECONSTRUCTED",
    "PRIMARY_SUMMARY_RECEIPT_RECONSTRUCTED",
    "QA_ONLY_SOURCE_RECEIPTS_BIND_STAGE_01",
    "QUALITY_DIRECTORY_CLOSED_SET",
    "QUALITY_RECEIPT_FILE_HASHES",
    "RAW_SOURCE_HASH_RECEIPT_RECONCILED",
    "REQUIRED_AGGREGATE_QUALITY_RECEIPTS",
    "SEC_FORM_COVERAGE_CSV_RECONSTRUCTED",
    "TABLE_COUNTS_CSV_RECONSTRUCTED",
)

_BINDING_ALIASES: dict[str, tuple[str, ...]] = {
    "policy_sha256": ("policy_sha256",),
    "schema_sha256": ("schema_sha256",),
    "source_config_sha256": ("source_config_sha256", "config_sha256"),
    "phase_02_config_sha256": ("phase_02_config_sha256",),
    "factor_unit_lock_sha256": (
        "factor_unit_lock_sha256",
        "factor_lock_sha256",
    ),
    "phase_01_freeze_manifest_sha256": (
        "phase_01_freeze_manifest_sha256",
        "stage_01_freeze_manifest_sha256",
    ),
    "signal_universe_verified_pointer_sha256": (
        "signal_universe_verified_pointer_sha256",
        "signal_verified_pointer_sha256",
    ),
}


class AuxiliaryBundleVerificationError(RuntimeError):
    """Raised when independent verification cannot be executed safely."""


class AuxiliaryVerificationResult(dict[str, Any]):
    """In-process verifier result carrying an unforgeable execution identity."""

    __hash__ = object.__hash__


@dataclass(frozen=True)
class _ExecutionBinding:
    settings: AuxiliaryVerificationSettings
    independent_check_sha256: str
    differences_sha256: str
    authority_snapshot: tuple[tuple[str, tuple[Any, ...] | None], ...]


_EXECUTION_BINDINGS: weakref.WeakKeyDictionary[
    AuxiliaryVerificationResult, _ExecutionBinding
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class AuxiliaryVerificationSettings:
    """All paths needed to verify a closed bundle, with no raw-data path."""

    project_root: Path
    primary_pointer_path: Path
    primary_manifest_path: Path
    tables_dir: Path
    quality_dir: Path
    policy_path: Path
    schema_path: Path
    source_config_path: Path
    phase_02_config_path: Path
    factor_unit_lock_path: Path
    stage_01_freeze_manifest_path: Path
    stage_01_raw_hashes_path: Path
    signal_verified_pointer_path: Path
    signal_primary_manifest_path: Path
    signal_universe_path: Path
    calendar_verified_pointer_path: Path
    calendar_primary_manifest_path: Path
    trading_calendar_path: Path
    batch_size: int = 65_536

    def normalized(self) -> "AuxiliaryVerificationSettings":
        values = {
            name: Path(getattr(self, name)).resolve()
            for name in (
                "project_root",
                "primary_pointer_path",
                "primary_manifest_path",
                "tables_dir",
                "quality_dir",
                "policy_path",
                "schema_path",
                "source_config_path",
                "phase_02_config_path",
                "factor_unit_lock_path",
                "stage_01_freeze_manifest_path",
                "stage_01_raw_hashes_path",
                "signal_verified_pointer_path",
                "signal_primary_manifest_path",
                "signal_universe_path",
                "calendar_verified_pointer_path",
                "calendar_primary_manifest_path",
                "trading_calendar_path",
            )
        }
        return AuxiliaryVerificationSettings(**values, batch_size=self.batch_size)


def _stat_identity(path: Path) -> tuple[Any, ...] | None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _authority_snapshot(
    settings: AuxiliaryVerificationSettings, manifest: Mapping[str, Any]
) -> tuple[tuple[str, tuple[Any, ...] | None], ...]:
    paths = {
        settings.primary_pointer_path,
        settings.primary_manifest_path,
        settings.primary_manifest_path.parent / "output_hashes.sha256",
        settings.tables_dir,
        settings.tables_dir / "_SUCCESS.json",
        settings.quality_dir,
        settings.policy_path,
        settings.schema_path,
        settings.source_config_path,
        settings.phase_02_config_path,
        settings.factor_unit_lock_path,
        settings.stage_01_freeze_manifest_path,
        settings.stage_01_raw_hashes_path,
        settings.signal_verified_pointer_path,
        settings.signal_primary_manifest_path,
        settings.signal_universe_path,
        settings.calendar_verified_pointer_path,
        settings.calendar_primary_manifest_path,
        settings.trading_calendar_path,
    }
    for name in REQUIRED_TABLE_NAMES:
        paths.add(settings.tables_dir / f"{name}.parquet")
    for encoded, receipt in _quality_receipts(manifest).items():
        try:
            paths.add(
                _safe_receipt_path(
                    settings.quality_dir, receipt.get("path", encoded)
                )
            )
        except AuxiliaryBundleVerificationError:
            continue
    try:
        signal_pointer = _load_json(settings.signal_verified_pointer_path)
        verification_path = signal_pointer.get("verification_path")
        if isinstance(verification_path, str):
            root = _safe_receipt_path(settings.project_root, verification_path)
            paths.update(
                {root, root / "independent_check.json", root / "differences.csv"}
            )
    except (OSError, AuxiliaryBundleVerificationError):
        pass
    return tuple(
        (str(path), _stat_identity(path))
        for path in sorted(paths, key=lambda value: str(value))
    )


def _assert_authority_snapshot(
    snapshot: Sequence[tuple[str, tuple[Any, ...] | None]],
) -> None:
    changed = [
        path
        for path, expected in snapshot
        if _stat_identity(Path(path)) != expected
    ]
    if changed:
        raise AuxiliaryBundleVerificationError(
            "Primary authority changed after independent verification: "
            + ", ".join(changed[:5])
        )


@dataclass
class _Checks:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        check_id: str,
        *,
        category: str,
        expected: Any,
        actual: Any,
        evidence: str,
    ) -> None:
        expected_value = _jsonable(expected)
        actual_value = _jsonable(actual)
        self.rows.append(
            {
                "check_id": check_id,
                "category": category,
                "status": "PASS" if expected_value == actual_value else "FAIL",
                "expected": expected_value,
                "actual": actual_value,
                "evidence": evidence,
            }
        )


@dataclass
class TableScan:
    table: str
    rows: int = 0
    violations: Counter[str] = field(default_factory=Counter)
    status_counts: Counter[str] = field(default_factory=Counter)
    metrics: Counter[str] = field(default_factory=Counter)
    availability_min: dt.date | None = None
    availability_max: dt.date | None = None
    factor_stats: dict[str, Any] | None = None

    def nonzero_violations(self) -> dict[str, int]:
        return {
            key: int(value)
            for key, value in sorted(self.violations.items())
            if value
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuxiliaryBundleVerificationError(f"Cannot read JSON object {path}") from exc
    if not isinstance(value, dict):
        raise AuxiliaryBundleVerificationError(f"Expected JSON object: {path}")
    return value


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise AuxiliaryBundleVerificationError(f"Cannot read TOML object {path}") from exc
    if not isinstance(value, dict):
        raise AuxiliaryBundleVerificationError(f"Expected TOML object: {path}")
    return value


def _arrow_type(encoded: str) -> pa.DataType:
    fixed = {
        "utf8": pa.string(),
        "int64": pa.int64(),
        "int32": pa.int32(),
        "int16": pa.int16(),
        "bool": pa.bool_(),
        "date32": pa.date32(),
        "timestamp_us_utc": pa.timestamp("us", tz="UTC"),
    }
    if encoded in fixed:
        return fixed[encoded]
    match = _DECIMAL_RE.fullmatch(encoded)
    if match:
        return pa.decimal128(int(match.group(1)), int(match.group(2)))
    raise AuxiliaryBundleVerificationError(f"Unsupported Arrow type {encoded!r}")


def _table_contract(
    schema_document: Mapping[str, Any], table_name: str
) -> Mapping[str, Any]:
    matches = [
        value
        for value in schema_document.get("tables", [])
        if value.get("table") == table_name
    ]
    if len(matches) != 1:
        raise AuxiliaryBundleVerificationError(
            f"Schema contract is not unique for {table_name!r}"
        )
    return matches[0]


def expected_table_schema(
    schema_document: Mapping[str, Any], table_name: str
) -> pa.Schema:
    """Independently reconstruct exact fields, types, order, and nullability."""

    contract = _table_contract(schema_document, table_name)
    fields: list[pa.Field] = []
    for encoded in contract.get("columns", []):
        match = _COLUMN_RE.fullmatch(str(encoded))
        if match is None:
            raise AuxiliaryBundleVerificationError(
                f"Malformed compact column contract {encoded!r}"
            )
        name, logical_type, suffix = match.groups()
        fields.append(
            pa.field(name, _arrow_type(logical_type), nullable=suffix == "?")
        )
    for payload in contract.get("source_payload_columns", []):
        if table_name == "compustat_annual_observation" and payload == "fyear":
            logical_type = pa.int16()
        elif table_name == "compustat_quarterly_observation" and payload in {
            "fyearq",
            "fqtr",
            "fyr",
        }:
            logical_type = pa.int16()
        elif table_name == "compustat_quarterly_observation" and payload == "datafqtr":
            logical_type = pa.string()
        else:
            logical_type = pa.decimal128(38, 12)
        fields.append(pa.field(str(payload), logical_type, nullable=True))
    return pa.schema(fields)


def _schema_without_metadata(schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [
            pa.field(field.name, field.type, nullable=field.nullable)
            for field in schema
        ]
    )


def _parse_shasum(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text("utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or _SHA256_RE.fullmatch(parts[0]) is None:
            raise AuxiliaryBundleVerificationError(
                f"Malformed Stage-1 hash line {line_number}: {path}"
            )
        name = parts[1].lstrip("*").strip()
        if not name or Path(name).name != name or name in result:
            raise AuxiliaryBundleVerificationError(
                f"Unsafe or duplicate Stage-1 hash name {name!r}"
            )
        result[name] = parts[0]
    return result


def _manifest_outputs(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("outputs")
    result: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        for name, receipt in raw.items():
            if not isinstance(receipt, Mapping):
                continue
            value = dict(receipt)
            value.setdefault("table", str(name))
            result[str(name)] = value
    elif isinstance(raw, list):
        for receipt in raw:
            if isinstance(receipt, Mapping) and isinstance(receipt.get("table"), str):
                result[str(receipt["table"])] = dict(receipt)
    return result


def _manifest_raw_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    raw = manifest.get("raw_source_hashes", manifest.get("input_hashes", {}))
    result: dict[str, str] = {}
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            digest = value.get("sha256") if isinstance(value, Mapping) else value
            if isinstance(digest, str):
                result[Path(str(name)).name] = digest
    elif isinstance(raw, list):
        for value in raw:
            if not isinstance(value, Mapping):
                continue
            name = value.get("source", value.get("filename", value.get("path")))
            digest = value.get("sha256")
            if isinstance(name, str) and isinstance(digest, str):
                result[Path(name).name] = digest
    return result


def _derive_primary_build_id(manifest: Mapping[str, Any]) -> str | None:
    required = (
        "contract_id",
        "database_contract_id",
        "bindings",
        "raw_source_hashes",
        "signal_universe",
        "outputs",
        "quality_receipts",
        "aggregate_qa",
    )
    if any(name not in manifest for name in required):
        return None
    outputs = _manifest_outputs(manifest)
    semantic = {
        "contract_id": manifest["contract_id"],
        "database_contract_id": manifest["database_contract_id"],
        "bindings": manifest["bindings"],
        "raw_source_hashes": manifest["raw_source_hashes"],
        "signal_universe": manifest["signal_universe"],
        "outputs": {
            name: {key: value for key, value in receipt.items() if key != "path"}
            for name, receipt in sorted(outputs.items())
        },
        "quality_receipts": manifest["quality_receipts"],
        "aggregate_qa": manifest["aggregate_qa"],
    }
    return BUILD_PREFIX + hashlib.sha256(canonical_json(semantic)).hexdigest()[:24]


def _binding_value(bindings: Mapping[str, Any], canonical_name: str) -> Any:
    for name in _BINDING_ALIASES[canonical_name]:
        if name in bindings:
            return bindings[name]
    return None


def _safe_receipt_path(root: Path, encoded: object) -> Path:
    if not isinstance(encoded, str) or not encoded:
        raise AuxiliaryBundleVerificationError("Receipt path is absent")
    relative = Path(encoded)
    if relative.is_absolute() or ".." in relative.parts:
        raise AuxiliaryBundleVerificationError(f"Unsafe receipt path {encoded!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AuxiliaryBundleVerificationError(f"Receipt escapes root: {encoded!r}") from exc
    return candidate


def _path_has_relative_suffix(path: Path, encoded: object) -> bool:
    if not isinstance(encoded, str) or not encoded:
        return False
    relative = Path(encoded)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return False
    parts = path.resolve().parts
    return len(parts) >= len(relative.parts) and parts[-len(relative.parts) :] == relative.parts


def _hash_receipt(path: Path) -> dict[str, Any]:
    return {
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _flat_directory_inventory(root: Path) -> dict[str, list[str]]:
    files: list[str] = []
    non_regular: list[str] = []
    if not root.is_dir():
        return {"files": files, "non_regular": ["<missing-directory>"]}
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        if path.is_symlink():
            non_regular.append(f"symlink:{path.name}")
        elif path.is_file():
            files.append(path.name)
        elif path.is_dir():
            non_regular.append(f"directory:{path.name}")
        else:
            non_regular.append(f"other:{path.name}")
    return {"files": files, "non_regular": non_regular}


def _receipt_expected(receipt: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "sha256": receipt.get("sha256"),
        "size_bytes": receipt.get("size_bytes"),
    }
    if "rows" in receipt:
        result["rows"] = receipt.get("rows")
    return result


def _receipt_actual(path: Path, *, include_rows: bool) -> dict[str, Any]:
    result = _hash_receipt(path)
    if include_rows:
        result["rows"] = pq.ParquetFile(path).metadata.num_rows
    return result


def _actual_contract_bindings(
    settings: AuxiliaryVerificationSettings,
) -> dict[str, str]:
    return {
        "policy_sha256": sha256_path(settings.policy_path),
        "schema_sha256": sha256_path(settings.schema_path),
        "source_config_sha256": sha256_path(settings.source_config_path),
        "phase_02_config_sha256": sha256_path(settings.phase_02_config_path),
        "factor_unit_lock_sha256": sha256_path(settings.factor_unit_lock_path),
        "phase_01_freeze_manifest_sha256": sha256_path(
            settings.stage_01_freeze_manifest_path
        ),
        "signal_universe_verified_pointer_sha256": sha256_path(
            settings.signal_verified_pointer_path
        ),
    }


def _authenticate_contracts(
    settings: AuxiliaryVerificationSettings,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    source_config: Mapping[str, Any],
    freeze: Mapping[str, Any],
    stage_01_hashes: Mapping[str, str],
    factor_lock: Mapping[str, Any],
    checks: _Checks,
) -> None:
    actual_bindings = _actual_contract_bindings(settings)
    manifest_bindings = manifest.get("bindings", {})
    expected_manifest_bindings = {
        key: _binding_value(manifest_bindings, key)
        if isinstance(manifest_bindings, Mapping)
        else None
        for key in actual_bindings
    }
    checks.add(
        "CONTRACT_POLICY_SCHEMA_FACTOR_AND_UPSTREAM_HASHES",
        category="authentication",
        expected=expected_manifest_bindings,
        actual=actual_bindings,
        evidence="independently hashed closed configuration and upstream pointer bytes",
    )

    policy_bindings = policy.get("bindings", {})
    expected_internal = {
        "schema_sha256": policy_bindings.get("schema_sha256"),
        "source_config_sha256": policy_bindings.get("source_config_sha256"),
        "phase_02_config_sha256": policy_bindings.get("phase_02_config_sha256"),
        "phase_01_freeze_manifest_sha256": policy_bindings.get(
            "phase_01_freeze_manifest_sha256"
        ),
        "factor_unit_lock_sha256": policy.get("factor_unit_lock", {}).get("sha256"),
    }
    actual_internal = {
        key: actual_bindings[key]
        for key in expected_internal
    }
    checks.add(
        "FROZEN_POLICY_INTERNAL_BINDINGS",
        category="authentication",
        expected=expected_internal,
        actual=actual_internal,
        evidence="policy-declared hashes versus the exact supplied frozen files",
    )

    evidence = freeze.get("identity", {}).get("evidence_groups", {})
    final_recheck = evidence.get("final_raw_recheck", [])
    declared_recheck = None
    if isinstance(final_recheck, list):
        for item in final_recheck:
            if isinstance(item, Mapping) and Path(str(item.get("path", ""))).name == settings.stage_01_raw_hashes_path.name:
                declared_recheck = item.get("sha256")
                break
    source_stage_binding = source_config.get("stage_01_binding", {})
    freeze_identity = freeze.get("identity", {})
    if not isinstance(freeze_identity, Mapping):
        freeze_identity = {}
    checks.add(
        "FROZEN_STAGE_01_EVIDENCE",
        category="authentication",
        expected={
            "freeze_status": "phase_01_frozen",
            "raw_hash_receipt_sha256": declared_recheck,
            "raw_snapshot_id": source_stage_binding.get("raw_snapshot_id"),
            "phase_01_build_id": source_stage_binding.get("phase_01_build_id"),
        },
        actual={
            "freeze_status": freeze.get("status"),
            "raw_hash_receipt_sha256": sha256_path(settings.stage_01_raw_hashes_path),
            "raw_snapshot_id": freeze.get(
                "raw_snapshot_id", freeze_identity.get("raw_snapshot_id")
            ),
            "phase_01_build_id": freeze.get(
                "phase_01_build_id", freeze_identity.get("phase_01_build_id")
            ),
        },
        evidence="frozen Stage-1 manifest and its authenticated final raw recheck",
    )

    required_sources = {
        str(item["filename"])
        for item in source_config.get("inputs", [])
        if isinstance(item, Mapping) and isinstance(item.get("filename"), str)
    }
    primary_hashes = _manifest_raw_hashes(manifest)
    expected_hashes = {
        name: stage_01_hashes.get(name) for name in sorted(required_sources)
    }
    actual_hashes = {
        name: primary_hashes.get(name) for name in sorted(required_sources)
    }
    checks.add(
        "RAW_SOURCE_HASHES_MATCH_FROZEN_STAGE_01",
        category="authentication",
        expected={"sources": expected_hashes, "source_set": sorted(required_sources)},
        actual={"sources": actual_hashes, "source_set": sorted(primary_hashes)},
        evidence="primary raw claims compared only with frozen Stage-1 hash evidence; raw files were not opened",
    )

    factor_sources = {
        "daily": factor_lock.get("daily", {}),
        "monthly": factor_lock.get("monthly", {}),
    }
    checks.add(
        "FACTOR_LOCK_SOURCE_HASHES_MATCH_STAGE_01",
        category="authentication",
        expected={
            name: stage_01_hashes.get(str(value.get("source")))
            for name, value in factor_sources.items()
        },
        actual={
            name: value.get("source_sha256") for name, value in factor_sources.items()
        },
        evidence="factor-unit lock source identities versus frozen Stage-1 hashes",
    )
    checks.add(
        "FACTOR_UNIT_SELECTION_IS_FROZEN_DECIMAL_RATIO",
        category="authentication",
        expected={
            "policy_multiplier": "1",
            "lock_multiplier": "1",
            "selection_precedes_primary_build": True,
        },
        actual={
            "policy_multiplier": str(
                policy.get("factor_unit_lock", {}).get(
                    "selected_source_to_decimal_ratio_multiplier", ""
                )
            ),
            "lock_multiplier": str(
                factor_lock.get("selected_source_to_decimal_ratio_multiplier", "")
            ),
            "selection_precedes_primary_build": bool(
                policy.get("factor_unit_lock", {}).get(
                    "selection_precedes_primary_build", False
                )
            ),
        },
        evidence="prebuild factor-unit choice in both policy and empirical lock",
    )

    implementation_evidence = freeze.get("identity", {}).get(
        "evidence_groups", {}
    ).get("implementation_and_configuration", [])
    frozen_raw_contracts: dict[str, str] = {}
    if isinstance(implementation_evidence, list):
        for item in implementation_evidence:
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("path", ""))
            if "configs/raw_contracts/" in path and path.endswith(".toml"):
                frozen_raw_contracts[Path(path).stem] = str(item.get("sha256"))
    source_raw_contracts = source_config.get("raw_contract_bindings", {})
    policy_raw_contracts = policy.get("bindings", {}).get("raw_contracts", {})
    checks.add(
        "RAW_CONTRACT_HASHES_MATCH_FROZEN_STAGE_01",
        category="authentication",
        expected={
            "source_config": dict(sorted(frozen_raw_contracts.items())),
            "policy": dict(sorted(frozen_raw_contracts.items())),
        },
        actual={
            "source_config": (
                dict(sorted(source_raw_contracts.items()))
                if isinstance(source_raw_contracts, Mapping)
                else None
            ),
            "policy": (
                dict(sorted(policy_raw_contracts.items()))
                if isinstance(policy_raw_contracts, Mapping)
                else None
            ),
        },
        evidence="all five raw-contract hashes reconciled to the frozen Stage-1 implementation evidence",
    )


def _signal_output_receipt(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    output = manifest.get("output")
    if isinstance(output, Mapping):
        return output
    outputs = manifest.get("outputs", [])
    candidates: list[Mapping[str, Any]] = []
    if isinstance(outputs, Mapping):
        candidates = [value for value in outputs.values() if isinstance(value, Mapping)]
    elif isinstance(outputs, list):
        candidates = [value for value in outputs if isinstance(value, Mapping)]
    parquet = [
        item for item in candidates if str(item.get("path", "")).endswith(".parquet")
    ]
    if len(parquet) != 1:
        raise AuxiliaryBundleVerificationError("Signal-universe Parquet receipt is not unique")
    return parquet[0]


def _authenticate_signal_universe(
    settings: AuxiliaryVerificationSettings,
    policy: Mapping[str, Any],
    checks: _Checks,
) -> int:
    pointer = _load_json(settings.signal_verified_pointer_path)
    manifest = _load_json(settings.signal_primary_manifest_path)
    upstream = policy.get("upstream", {}).get("signal_universe", {})
    required_authorization = str(
        upstream.get("verified_authorization", "signal_date_universe_authorized")
    )
    pointer_authorized = bool(
        pointer.get("authorizations", {}).get(
            required_authorization, pointer.get(required_authorization, False)
        )
    )
    checks.add(
        "VERIFIED_SIGNAL_UNIVERSE_POINTER",
        category="authentication",
        expected={
            "pointer_sha256": upstream.get("verified_pointer_sha256"),
            "status": upstream.get("verified_status", "verified_passed"),
            "build_id": upstream.get("required_build_id"),
            "manifest_build_id": pointer.get("build_id"),
            "authorized": True,
            "primary_manifest_sha256": pointer.get("primary_manifest_sha256"),
        },
        actual={
            "pointer_sha256": sha256_path(settings.signal_verified_pointer_path),
            "status": pointer.get("status"),
            "build_id": pointer.get("build_id"),
            "manifest_build_id": manifest.get("build_id"),
            "authorized": pointer_authorized,
            "primary_manifest_sha256": sha256_path(settings.signal_primary_manifest_path),
        },
        evidence="independently verified signal-universe pointer and exact primary manifest bytes",
    )

    evidence_actual: dict[str, Any] = {}
    verification_path = pointer.get("verification_path")
    if isinstance(verification_path, str):
        evidence_root = _safe_receipt_path(settings.project_root, verification_path)
        report_path = evidence_root / "independent_check.json"
        differences_path = evidence_root / "differences.csv"
        if report_path.is_file() and differences_path.is_file():
            report = _load_json(report_path)
            evidence_actual = {
                "independent_check_sha256": sha256_path(report_path),
                "differences_sha256": sha256_path(differences_path),
                "report_status": report.get("status"),
                "report_build_id": report.get("build_id"),
                "report_fail_count": report.get("summary", {}).get("fail_count", 0),
            }
    checks.add(
        "VERIFIED_SIGNAL_UNIVERSE_EVIDENCE",
        category="authentication",
        expected={
            "independent_check_sha256": pointer.get("independent_check_sha256"),
            "differences_sha256": pointer.get("differences_sha256"),
            "report_status": "verified_passed",
            "report_build_id": pointer.get("build_id"),
            "report_fail_count": 0,
        },
        actual=evidence_actual,
        evidence="hash-bound upstream independent-check and differences files",
    )

    receipt = _signal_output_receipt(manifest)
    expected = _receipt_expected(receipt)
    actual = _receipt_actual(settings.signal_universe_path, include_rows=True)
    checks.add(
        "VERIFIED_SIGNAL_UNIVERSE_FILE_RECEIPT",
        category="authentication",
        expected=expected,
        actual=actual,
        evidence="full signal-universe Parquet hash, size, and metadata row count",
    )
    return int(actual["rows"])


def _authenticate_trading_calendar(
    settings: AuxiliaryVerificationSettings,
    primary_manifest: Mapping[str, Any],
    checks: _Checks,
) -> tuple[dt.date, ...]:
    """Authenticate and load the small verified CRSP calendar, never raw data."""

    pointer = _load_json(settings.calendar_verified_pointer_path)
    manifest = _load_json(settings.calendar_primary_manifest_path)
    candidates = [
        item
        for item in manifest.get("outputs", [])
        if isinstance(item, Mapping)
        and Path(str(item.get("path", ""))).name == "trading_calendar.parquet"
    ]
    if len(candidates) != 1:
        raise AuxiliaryBundleVerificationError(
            "Verified trading-calendar receipt is not unique"
        )
    receipt = candidates[0]
    parquet = pq.ParquetFile(settings.trading_calendar_path)
    schema = parquet.schema_arrow
    table = parquet.read(columns=["calendar_date"])
    values = table["calendar_date"].combine_chunks().to_pylist()
    dates = tuple(value for value in values if isinstance(value, dt.date))
    actual = {
        "pointer_status": pointer.get("status"),
        "pointer_build_id": pointer.get("build_id"),
        "manifest_build_id": manifest.get("build_id"),
        "pointer_contract_id": pointer.get("contract_id"),
        "manifest_contract_id": manifest.get("contract_id"),
        "auxiliary_database_contract_id": primary_manifest.get(
            "database_contract_id"
        ),
        "primary_manifest_sha256": sha256_path(
            settings.calendar_primary_manifest_path
        ),
        "calendar_sha256": sha256_path(settings.trading_calendar_path),
        "calendar_size_bytes": settings.trading_calendar_path.stat().st_size,
        "calendar_rows": parquet.metadata.num_rows,
        "calendar_date_field": (
            {
                "name": schema.field("calendar_date").name,
                "type": str(schema.field("calendar_date").type),
                "nullable": schema.field("calendar_date").nullable,
            }
            if "calendar_date" in schema.names
            else None
        ),
        "strictly_sorted_unique_nonnull": (
            len(dates) == len(values)
            and all(left < right for left, right in itertools.pairwise(dates))
        ),
    }
    expected = {
        "pointer_status": "verified_passed",
        "pointer_build_id": manifest.get("build_id"),
        "manifest_build_id": manifest.get("build_id"),
        "pointer_contract_id": manifest.get("contract_id"),
        "manifest_contract_id": manifest.get("contract_id"),
        "auxiliary_database_contract_id": manifest.get("contract_id"),
        "primary_manifest_sha256": pointer.get("primary_manifest_sha256"),
        "calendar_sha256": receipt.get("sha256"),
        "calendar_size_bytes": receipt.get("size_bytes"),
        "calendar_rows": parquet.metadata.num_rows,
        "calendar_date_field": {
            "name": "calendar_date",
            "type": "date32[day]",
            "nullable": True,
        },
        "strictly_sorted_unique_nonnull": True,
    }
    checks.add(
        "VERIFIED_TRADING_CALENDAR_EVIDENCE",
        category="authentication",
        expected=expected,
        actual=actual,
        evidence="verified calendar pointer, primary manifest, exact Parquet receipt and full date scan",
    )
    return dates


def _sortable(value: Any) -> tuple[int, Any]:
    if value is None:
        return (0, "")
    if isinstance(value, dt.datetime):
        return (1, value.isoformat())
    if isinstance(value, dt.date):
        return (1, value.toordinal())
    if isinstance(value, Decimal):
        return (1, value)
    return (1, value)


def _sort_tuple(values: Sequence[Any]) -> tuple[tuple[int, Any], ...]:
    return tuple(_sortable(value) for value in values)


def _pk_fingerprint(values: Sequence[Any]) -> bytes:
    # Store the complete canonical key rather than a truncated digest: PK
    # uniqueness is an exact gate, not a probabilistic collision check.
    return canonical_json(list(values))


def _values(table: pa.Table, names: Iterable[str]) -> dict[str, list[Any]]:
    return {name: table[name].combine_chunks().to_pylist() for name in names}


def _update_date_bounds(scan: TableScan, values: Iterable[dt.date | None]) -> None:
    for value in values:
        if value is None:
            continue
        scan.availability_min = (
            value if scan.availability_min is None else min(scan.availability_min, value)
        )
        scan.availability_max = (
            value if scan.availability_max is None else max(scan.availability_max, value)
        )


def _semantic_batch(
    table_name: str,
    table: pa.Table,
    scan: TableScan,
    *,
    contract: Mapping[str, Any],
    last_signal: dt.date,
    last_exit: dt.date,
) -> None:
    names = set(table.column_names)
    size = table.num_rows
    columns = _values(table, names)

    for name in names:
        if "cusip8" in name:
            scan.violations["invalid_normalized_cusip8"] += sum(
                value is not None and _CUSIP8_RE.fullmatch(str(value)) is None
                for value in columns[name]
            )
    if "source_record_id" in names:
        scan.violations["invalid_source_record_id"] += sum(
            value is None or _SHA256_RE.fullmatch(str(value)) is None
            for value in columns["source_record_id"]
        )

    if {"observation_date", "public_availability_date"} <= names:
        observations = columns["observation_date"]
        availability = columns["public_availability_date"]
        maximum = (
            last_exit
            if table_name
            in {"ff_factor_daily_observation", "ff_factor_monthly_observation"}
            else last_signal
        )
        scan.violations["availability_after_scope"] += sum(
            value is not None and value > maximum for value in availability
        )
        _update_date_bounds(scan, availability)
        if "availability_basis" in names:
            scan.violations["blank_availability_basis"] += sum(
                value is None or not str(value).strip()
                for value in columns["availability_basis"]
            )
    if {"public_availability_date", "stale_after_date"} <= names:
        scan.violations["stale_before_availability"] += sum(
            left is not None and right is not None and right < left
            for left, right in zip(
                columns["public_availability_date"],
                columns["stale_after_date"],
                strict=True,
            )
        )

    interval_pairs = (
        ("link_start_date", "link_end_date"),
        ("effective_start_date", "effective_end_date"),
        ("affected_period_start", "affected_period_end"),
        ("matched_identity_start_date", "matched_identity_end_date"),
    )
    for start_name, end_name in interval_pairs:
        if {start_name, end_name} <= names:
            scan.violations[f"invalid_interval_{start_name}"] += sum(
                start is not None and end is not None and start > end
                for start, end in zip(
                    columns[start_name], columns[end_name], strict=True
                )
            )

    if table_name == "ccm_link_interval":
        for index in range(size):
            linktype = columns["linktype"][index]
            linkprim = columns["linkprim"][index]
            allowed = linktype in {"LC", "LU"} and linkprim in {"P", "C"}
            scan.violations["ccm_allowed_flag"] += columns["link_allowed"][index] != allowed
            expected_precedence = 1 if linkprim == "P" else 2 if linkprim == "C" else None
            scan.violations["ccm_precedence"] += (
                columns["link_precedence"][index] != expected_precedence
            )

    if table_name in {
        "compustat_annual_observation",
        "compustat_quarterly_observation",
    }:
        for index in range(size):
            eligible = all(
                columns[name][index] == value
                for name, value in {
                    "indfmt": "INDL",
                    "consol": "C",
                    "popsrc": "D",
                    "datafmt": "STD",
                }.items()
            )
            scan.violations["standard_format_flag"] += (
                columns["standard_format_eligible"][index] != eligible
            )

    if table_name == "sec_filing_observation":
        allowed_forms = {"10-K", "10-Q", "20-F", "40-F", "8-K", "6-K"}
        scan.violations["unsupported_sec_form"] += sum(
            value not in allowed_forms for value in columns["form_family"]
        )

    if table_name == "sec_8k_item_observation":
        scan.violations["non_8k_item_taxonomy"] += sum(
            form != "8-K" for form in columns["form_family"]
        )

    if table_name == "sec_13f_filing_observation":
        scan.violations["report_date_after_filing"] += sum(
            report is not None and filing is not None and report > filing
            for report, filing in zip(
                columns["report_date"], columns["filing_date"], strict=True
            )
        )

    if table_name == "sec_13f_holding_observation":
        scan.violations["report_date_after_filing"] += sum(
            report is not None and filing is not None and report > filing
            for report, filing in zip(
                columns["report_date"], columns["filing_date"], strict=True
            )
        )

    if table_name == "restatement_event_observation":
        scan.violations["affected_period_reversed"] += sum(
            left is not None and right is not None and left > right
            for left, right in zip(
                columns["affected_period_start"],
                columns["affected_period_end"],
                strict=True,
            )
        )

    if table_name == "restatement_crsp_link":
        allowed_basis = {
            "normalized_cusip8_date_valid",
            "company_key_bridge_date_valid",
            "UNMATCHED",
        }
        for index in range(size):
            status = str(columns["link_status"][index])
            basis = columns["link_basis"][index]
            count = columns["candidate_count"][index]
            permno = columns["permno"][index]
            scan.status_counts[status] += 1
            scan.violations["invalid_link_status"] += status not in {
                "MATCHED",
                "AMBIGUOUS",
                "UNMATCHED",
            }
            scan.violations["invalid_link_basis"] += basis not in allowed_basis
            if status == "MATCHED":
                scan.violations["matched_link_shape"] += not (
                    count == 1 and permno is not None and basis != "UNMATCHED"
                )
            elif status == "AMBIGUOUS":
                scan.violations["ambiguous_link_shape"] += not (
                    isinstance(count, int) and count > 1 and permno is None
                )
            elif status == "UNMATCHED":
                scan.violations["unmatched_link_shape"] += not (
                    count == 0 and permno is None and basis == "UNMATCHED"
                )

    if table_name == "optionmetrics_name_interval":
        scan.violations["name_interval_after_scope"] += sum(
            value is not None and value > last_signal
            for value in columns["effective_end_date"]
        )

    if table_name == "optionmetrics_crsp_link_interval":
        for index in range(size):
            status = str(columns["link_status"][index])
            count = columns["candidate_count"][index]
            scan.status_counts[status] += 1
            scan.violations["invalid_link_status"] += status not in {
                "MATCHED",
                "AMBIGUOUS",
            }
            scan.violations["candidate_count_status"] += not (
                (status == "MATCHED" and count == 1)
                or (status == "AMBIGUOUS" and isinstance(count, int) and count > 1)
            )

    if table_name == "optionmetrics_option_volume_observation":
        for index in range(size):
            role = str(columns["row_role"][index])
            authoritative = columns["authoritative_flow_component"][index]
            scan.status_counts[role] += 1
            scan.violations["invalid_option_role"] += role not in {
                "CALL",
                "PUT",
                "TOTAL",
            }
            scan.violations["authoritative_role_flag"] += (
                authoritative != (role in {"CALL", "PUT"})
            )
            for value_name, flag_name in (
                ("volume", "genuine_zero_volume"),
                ("open_interest", "genuine_zero_open_interest"),
            ):
                value = columns[value_name][index]
                flag = columns[flag_name][index]
                scan.violations[f"negative_{value_name}"] += (
                    value is not None and value < 0
                )
                scan.violations[f"incorrect_{flag_name}"] += flag != (
                    None if value is None else value == 0
                )

    if table_name == "optionmetrics_historical_volatility_observation":
        supported = {
            int(value)
            for value in contract.get("rules", {}).get(
                "supported_horizons", [30, 60, 91, 182, 365]
            )
        }
        for index in range(size):
            days = columns["days"][index]
            volatility = columns["volatility"][index]
            scan.violations["supported_horizon_flag"] += (
                columns["supported_horizon"][index] != (days in supported)
            )
            scan.violations["nonpositive_horizon"] += days is None or days <= 0
            scan.violations["negative_historical_volatility"] += (
                volatility is not None and volatility < 0
            )

    if table_name in {"ff_factor_daily_observation", "ff_factor_monthly_observation"}:
        observation = columns["observation_date"]
        availability = columns["public_availability_date"]
        scan.violations["factor_availability_not_dated_close"] += sum(
            left != right
            for left, right in zip(observation, availability, strict=True)
        )
        if table_name == "ff_factor_monthly_observation":
            scan.violations["factor_month_not_observation"] += sum(
                left != right
                for left, right in zip(
                    columns["factor_month"], observation, strict=True
                )
            )
        current = scan.factor_stats or {
            "rows": 0,
            "minimum_date": None,
            "maximum_date": None,
            "maximum_absolute_mktrf": Decimal(0),
            "maximum_absolute_rf": Decimal(0),
        }
        current["rows"] += size
        for value in observation:
            if value is None:
                continue
            current["minimum_date"] = (
                value
                if current["minimum_date"] is None
                else min(current["minimum_date"], value)
            )
            current["maximum_date"] = (
                value
                if current["maximum_date"] is None
                else max(current["maximum_date"], value)
            )
        for name in ("mktrf", "rf"):
            key = f"maximum_absolute_{name}"
            for value in columns[name]:
                if value is not None:
                    current[key] = max(current[key], abs(value))
        scan.factor_stats = current

    if table_name == "ps_liquidity_observation":
        for index in range(size):
            month = columns["observation_month"][index]
            observation = columns["observation_date"][index]
            release = columns["recorded_release_date"][index]
            availability = columns["public_availability_date"][index]
            basis = str(columns["availability_basis"][index])
            policy_violation = not (
                month == observation
                and observation is not None
                and availability is not None
                and availability > observation
                and basis == "one_full_calendar_month_lag_fallback"
            )
            scan.metrics["ps_policy_violations"] += policy_violation
            scan.violations["ps_availability_not_lagged"] += policy_violation
            if release is not None:
                scan.violations["ps_recorded_release_not_lagged"] += not (
                    availability is not None and availability > release
                )
            elif "fallback" not in basis.lower():
                scan.violations["ps_missing_release_without_fallback"] += 1
            if observation is not None and availability is not None:
                lag = (availability - observation).days
                if scan.metrics.get("ps_lag_rows", 0) == 0:
                    scan.metrics["ps_min_lag_days"] = lag
                    scan.metrics["ps_max_lag_days"] = lag
                else:
                    scan.metrics["ps_min_lag_days"] = min(
                        scan.metrics["ps_min_lag_days"], lag
                    )
                    scan.metrics["ps_max_lag_days"] = max(
                        scan.metrics["ps_max_lag_days"], lag
                    )
                scan.metrics["ps_lag_rows"] += 1

    if table_name == "auxiliary_signal_linkage":
        allowed_statuses = {"MATCHED", "UNMATCHED", "AMBIGUOUS", "NOT_COVERED"}
        families = {
            "ccm": ("ccm_match_status", ("ccm_gvkey", "ccm_linktype", "ccm_linkprim")),
            "sec": ("sec_match_status", ("sec_cik",)),
            "optionmetrics": (
                "optionmetrics_match_status",
                ("optionmetrics_secid",),
            ),
        }
        for index in range(size):
            statuses: list[str] = []
            for family, (status_name, value_names) in families.items():
                status = str(columns[status_name][index])
                statuses.append(status)
                scan.status_counts[f"{family}:{status}"] += 1
                scan.violations[f"invalid_{family}_status"] += status not in allowed_statuses
                values = [columns[name][index] for name in value_names]
                if status == "MATCHED":
                    scan.violations[f"matched_{family}_fields"] += any(
                        value is None for value in values
                    )
                else:
                    scan.violations[f"nonmatched_{family}_fields"] += any(
                        value is not None for value in values
                    )
                if status == "AMBIGUOUS":
                    scan.metrics["ambiguous_field_violations"] += any(
                        value is not None for value in values
                    )
            ambiguity = "AMBIGUOUS" in statuses
            scan.metrics["ambiguity_rows"] += ambiguity
            scan.metrics["reported_ambiguity_rows"] += bool(
                columns["any_auxiliary_ambiguity"][index]
            )
            scan.violations["ambiguity_flag"] += (
                columns["any_auxiliary_ambiguity"][index] != ambiguity
            )
            scan.violations["universe_row_not_preserved"] += not bool(
                columns["universe_row_preserved"][index]
            )
            signal_date = columns["signal_date"][index]
            scan.violations["signal_after_scope"] += (
                signal_date is not None and signal_date > last_signal
            )


def _array(table: pa.Table, name: str) -> pa.Array:
    return table[name].combine_chunks()


def _count_true(mask: pa.Array | pa.ChunkedArray) -> int:
    if len(mask) == 0:
        return 0
    value = pc.sum(
        pc.cast(pc.fill_null(mask, False), pa.int64())
    ).as_py()
    return int(value or 0)


def _count_false(mask: pa.Array | pa.ChunkedArray) -> int:
    return _count_true(pc.invert(pc.fill_null(mask, False)))


def _nullable_difference_count(
    actual: pa.Array, expected: pa.Array
) -> int:
    actual_valid = pc.is_valid(actual)
    expected_valid = pc.is_valid(expected)
    validity_diff = pc.not_equal(actual_valid, expected_valid)
    both_valid = pc.and_(actual_valid, expected_valid)
    value_diff = pc.and_(
        both_valid,
        pc.invert(
            pc.fill_null(
                pc.equal(actual, expected),
                False,
            )
        ),
    )
    return _count_true(pc.or_(validity_diff, value_diff))


def _update_bounds_from_array(scan: TableScan, values: pa.Array) -> None:
    if len(values) == 0 or values.null_count == len(values):
        return
    minimum = pc.min(values).as_py()
    maximum = pc.max(values).as_py()
    if minimum is not None:
        scan.availability_min = (
            minimum
            if scan.availability_min is None
            else min(scan.availability_min, minimum)
        )
    if maximum is not None:
        scan.availability_max = (
            maximum
            if scan.availability_max is None
            else max(scan.availability_max, maximum)
        )


def _add_value_counts(scan: TableScan, values: pa.Array, *, prefix: str = "") -> None:
    for row in pc.value_counts(values).to_pylist():
        value = row["values"]
        if value is not None:
            scan.status_counts[prefix + str(value)] += int(row["counts"])


def _semantic_batch_vectorized(
    table_name: str,
    table: pa.Table,
    scan: TableScan,
    *,
    contract: Mapping[str, Any],
    last_signal: dt.date,
    last_exit: dt.date,
) -> None:
    """Arrow-kernel semantic checks; no per-row Python conversion on large tables."""

    names = set(table.column_names)
    for name in names:
        if "cusip8" not in name:
            continue
        values = _array(table, name)
        matches = pc.match_substring_regex(values, r"^[A-Z0-9]{8}$")
        scan.violations["invalid_normalized_cusip8"] += _count_true(
            pc.and_(pc.is_valid(values), pc.invert(pc.fill_null(matches, False)))
        )
    if "source_record_id" in names:
        values = _array(table, "source_record_id")
        matches = pc.match_substring_regex(values, r"^[0-9a-f]{64}$")
        scan.violations["invalid_source_record_id"] += _count_true(
            pc.invert(pc.fill_null(matches, False))
        )

    if {"observation_date", "public_availability_date"} <= names:
        observation = _array(table, "observation_date")
        availability = _array(table, "public_availability_date")
        maximum = (
            last_exit
            if table_name
            in {"ff_factor_daily_observation", "ff_factor_monthly_observation"}
            else last_signal
        )
        scan.violations["availability_after_scope"] += _count_true(
            pc.greater(availability, pa.scalar(maximum, type=pa.date32()))
        )
        _update_bounds_from_array(scan, availability)
        if "availability_basis" in names:
            basis = _array(table, "availability_basis")
            blank = pc.equal(pc.utf8_trim_whitespace(basis), "")
            scan.violations["blank_availability_basis"] += _count_true(
                pc.or_(pc.is_null(basis), pc.fill_null(blank, True))
            )
    if {"public_availability_date", "stale_after_date"} <= names:
        scan.violations["stale_before_availability"] += _count_true(
            pc.less(
                _array(table, "stale_after_date"),
                _array(table, "public_availability_date"),
            )
        )

    for start_name, end_name in (
        ("link_start_date", "link_end_date"),
        ("effective_start_date", "effective_end_date"),
        ("affected_period_start", "affected_period_end"),
        ("matched_identity_start_date", "matched_identity_end_date"),
    ):
        if {start_name, end_name} <= names:
            start = _array(table, start_name)
            end = _array(table, end_name)
            scan.violations[f"invalid_interval_{start_name}"] += _count_true(
                pc.and_(pc.is_valid(end), pc.greater(start, end))
            )

    if table_name == "ccm_link_interval":
        linktype = _array(table, "linktype")
        linkprim = _array(table, "linkprim")
        expected_allowed = pc.and_(
            pc.is_in(linktype, value_set=pa.array(["LC", "LU"])),
            pc.is_in(linkprim, value_set=pa.array(["P", "C"])),
        )
        scan.violations["ccm_allowed_flag"] += _count_false(
            pc.equal(_array(table, "link_allowed"), expected_allowed)
        )
        expected_precedence = pc.if_else(
            pc.equal(linkprim, "P"),
            pa.scalar(1, type=pa.int16()),
            pc.if_else(
                pc.equal(linkprim, "C"),
                pa.scalar(2, type=pa.int16()),
                pa.scalar(None, type=pa.int16()),
            ),
        )
        scan.violations["ccm_precedence"] += _nullable_difference_count(
            _array(table, "link_precedence"), expected_precedence
        )

    if table_name in {
        "compustat_annual_observation",
        "compustat_quarterly_observation",
    }:
        expected = pc.and_(
            pc.and_(pc.equal(_array(table, "indfmt"), "INDL"), pc.equal(_array(table, "consol"), "C")),
            pc.and_(pc.equal(_array(table, "popsrc"), "D"), pc.equal(_array(table, "datafmt"), "STD")),
        )
        scan.violations["standard_format_flag"] += _count_false(
            pc.equal(_array(table, "standard_format_eligible"), expected)
        )

    if table_name == "sec_filing_observation":
        forms = _array(table, "form_family")
        scan.violations["unsupported_sec_form"] += _count_true(
            pc.invert(
                pc.is_in(
                    forms,
                    value_set=pa.array(["10-K", "10-Q", "20-F", "40-F", "8-K", "6-K"]),
                )
            )
        )
    if table_name == "sec_8k_item_observation":
        scan.violations["non_8k_item_taxonomy"] += _count_true(
            pc.not_equal(_array(table, "form_family"), "8-K")
        )
    if table_name in {"sec_13f_filing_observation", "sec_13f_holding_observation"}:
        scan.violations["report_date_after_filing"] += _count_true(
            pc.greater(_array(table, "report_date"), _array(table, "filing_date"))
        )

    if table_name == "restatement_crsp_link":
        status = _array(table, "link_status")
        basis = _array(table, "link_basis")
        count = _array(table, "candidate_count")
        permno = _array(table, "permno")
        _add_value_counts(scan, status)
        scan.violations["invalid_link_status"] += _count_true(
            pc.invert(pc.is_in(status, value_set=pa.array(["MATCHED", "AMBIGUOUS", "UNMATCHED"])))
        )
        scan.violations["invalid_link_basis"] += _count_true(
            pc.invert(pc.is_in(basis, value_set=pa.array(["normalized_cusip8_date_valid", "company_key_bridge_date_valid", "UNMATCHED"])))
        )
        matched = pc.equal(status, "MATCHED")
        matched_shape = pc.and_(
            pc.and_(pc.equal(count, 1), pc.is_valid(permno)),
            pc.not_equal(basis, "UNMATCHED"),
        )
        scan.violations["matched_link_shape"] += _count_true(
            pc.and_(matched, pc.invert(pc.fill_null(matched_shape, False)))
        )
        ambiguous = pc.equal(status, "AMBIGUOUS")
        ambiguous_shape = pc.and_(pc.greater(count, 1), pc.is_null(permno))
        scan.violations["ambiguous_link_shape"] += _count_true(
            pc.and_(ambiguous, pc.invert(pc.fill_null(ambiguous_shape, False)))
        )
        unmatched = pc.equal(status, "UNMATCHED")
        unmatched_shape = pc.and_(
            pc.and_(pc.equal(count, 0), pc.is_null(permno)),
            pc.equal(basis, "UNMATCHED"),
        )
        scan.violations["unmatched_link_shape"] += _count_true(
            pc.and_(unmatched, pc.invert(pc.fill_null(unmatched_shape, False)))
        )

    if table_name == "optionmetrics_name_interval":
        scan.violations["name_interval_after_scope"] += _count_true(
            pc.greater(
                _array(table, "effective_end_date"),
                pa.scalar(last_signal, type=pa.date32()),
            )
        )
    if table_name == "optionmetrics_crsp_link_interval":
        status = _array(table, "link_status")
        count = _array(table, "candidate_count")
        _add_value_counts(scan, status)
        scan.violations["invalid_link_status"] += _count_true(
            pc.invert(pc.is_in(status, value_set=pa.array(["MATCHED", "AMBIGUOUS"])))
        )
        valid_shape = pc.or_(
            pc.and_(pc.equal(status, "MATCHED"), pc.equal(count, 1)),
            pc.and_(pc.equal(status, "AMBIGUOUS"), pc.greater(count, 1)),
        )
        scan.violations["candidate_count_status"] += _count_false(valid_shape)

    if table_name == "optionmetrics_option_volume_observation":
        role = _array(table, "row_role")
        authoritative = _array(table, "authoritative_flow_component")
        _add_value_counts(scan, role)
        allowed_role = pc.is_in(role, value_set=pa.array(["CALL", "PUT", "TOTAL"]))
        expected_authoritative = pc.is_in(role, value_set=pa.array(["CALL", "PUT"]))
        scan.violations["invalid_option_role"] += _count_false(allowed_role)
        scan.violations["authoritative_role_flag"] += _count_false(
            pc.equal(authoritative, expected_authoritative)
        )
        for value_name, flag_name in (
            ("volume", "genuine_zero_volume"),
            ("open_interest", "genuine_zero_open_interest"),
        ):
            value = _array(table, value_name)
            scan.violations[f"negative_{value_name}"] += _count_true(pc.less(value, 0))
            expected_zero = pc.if_else(
                pc.is_null(value),
                pa.scalar(None, type=pa.bool_()),
                pc.equal(value, 0),
            )
            scan.violations[f"incorrect_{flag_name}"] += _nullable_difference_count(
                _array(table, flag_name), expected_zero
            )

    if table_name == "optionmetrics_historical_volatility_observation":
        days = _array(table, "days")
        volatility = _array(table, "volatility")
        supported_values = pa.array(
            [
                int(value)
                for value in contract.get("rules", {}).get(
                    "supported_horizons", [30, 60, 91, 182, 365]
                )
            ],
            type=pa.int16(),
        )
        expected_supported = pc.is_in(days, value_set=supported_values)
        scan.violations["supported_horizon_flag"] += _count_false(
            pc.equal(_array(table, "supported_horizon"), expected_supported)
        )
        scan.violations["nonpositive_horizon"] += _count_true(pc.less_equal(days, 0))
        scan.violations["negative_historical_volatility"] += _count_true(
            pc.less(volatility, 0)
        )

    if table_name in {"ff_factor_daily_observation", "ff_factor_monthly_observation"}:
        observation = _array(table, "observation_date")
        scan.violations["factor_availability_not_dated_close"] += _count_false(
            pc.equal(observation, _array(table, "public_availability_date"))
        )
        if table_name == "ff_factor_monthly_observation":
            scan.violations["factor_month_not_observation"] += _count_false(
                pc.equal(_array(table, "factor_month"), observation)
            )
        current = scan.factor_stats or {
            "rows": 0,
            "minimum_date": None,
            "maximum_date": None,
            "maximum_absolute_mktrf": Decimal(0),
            "maximum_absolute_rf": Decimal(0),
        }
        current["rows"] += table.num_rows
        if table.num_rows:
            minimum = pc.min(observation).as_py()
            maximum = pc.max(observation).as_py()
            current["minimum_date"] = minimum if current["minimum_date"] is None else min(current["minimum_date"], minimum)
            current["maximum_date"] = maximum if current["maximum_date"] is None else max(current["maximum_date"], maximum)
        for name in ("mktrf", "rf"):
            extrema = pc.min_max(_array(table, name)).as_py()
            candidates = [abs(value) for value in extrema.values() if value is not None]
            if candidates:
                key = f"maximum_absolute_{name}"
                current[key] = max(current[key], *candidates)
        scan.factor_stats = current

    if table_name == "ps_liquidity_observation":
        values = _values(
            table,
            (
                "observation_month",
                "observation_date",
                "recorded_release_date",
                "public_availability_date",
                "availability_basis",
            ),
        )
        for index in range(table.num_rows):
            month = values["observation_month"][index]
            observation = values["observation_date"][index]
            release = values["recorded_release_date"][index]
            availability = values["public_availability_date"][index]
            basis = values["availability_basis"][index]
            one_full_month_later = None
            if observation is not None:
                following_month = (
                    observation.replace(day=1) + dt.timedelta(days=63)
                ).replace(day=1)
                one_full_month_later = following_month - dt.timedelta(days=1)
            policy_violation = not (
                month == observation
                and observation is not None
                and availability is not None
                and one_full_month_later is not None
                and availability >= one_full_month_later
                and release is None
                and basis == "one_full_calendar_month_lag_fallback"
            )
            scan.metrics["ps_policy_violations"] += policy_violation
            scan.violations["ps_availability_not_lagged"] += policy_violation
            if release is not None:
                scan.violations["ps_recorded_release_not_lagged"] += not (
                    availability is not None and availability > release
                )
            if (
                one_full_month_later is not None
                and availability is not None
                and availability < one_full_month_later
            ):
                scan.violations["ps_fallback_before_full_month"] += 1
            if observation is not None and availability is not None:
                lag = (availability - observation).days
                if scan.metrics.get("ps_lag_rows", 0) == 0:
                    scan.metrics["ps_min_lag_days"] = lag
                    scan.metrics["ps_max_lag_days"] = lag
                else:
                    scan.metrics["ps_min_lag_days"] = min(scan.metrics["ps_min_lag_days"], lag)
                    scan.metrics["ps_max_lag_days"] = max(scan.metrics["ps_max_lag_days"], lag)
                scan.metrics["ps_lag_rows"] += 1

    if table_name == "auxiliary_signal_linkage":
        allowed = pa.array(["MATCHED", "UNMATCHED", "AMBIGUOUS", "NOT_COVERED"])
        ambiguity_masks: list[pa.Array] = []
        for family, status_name, value_names in (
            ("ccm", "ccm_match_status", ("ccm_gvkey", "ccm_linktype", "ccm_linkprim")),
            ("sec", "sec_match_status", ("sec_cik",)),
            ("optionmetrics", "optionmetrics_match_status", ("optionmetrics_secid",)),
        ):
            status = _array(table, status_name)
            _add_value_counts(scan, status, prefix=f"{family}:")
            scan.violations[f"invalid_{family}_status"] += _count_true(
                pc.invert(pc.is_in(status, value_set=allowed))
            )
            matched = pc.equal(status, "MATCHED")
            ambiguous = pc.equal(status, "AMBIGUOUS")
            ambiguity_masks.append(ambiguous)
            any_missing = pa.array([False] * table.num_rows)
            any_present = pa.array([False] * table.num_rows)
            for name in value_names:
                any_missing = pc.or_(any_missing, pc.is_null(_array(table, name)))
                any_present = pc.or_(any_present, pc.is_valid(_array(table, name)))
            scan.violations[f"matched_{family}_fields"] += _count_true(
                pc.and_(matched, any_missing)
            )
            scan.violations[f"nonmatched_{family}_fields"] += _count_true(
                pc.and_(pc.invert(pc.fill_null(matched, False)), any_present)
            )
            scan.metrics["ambiguous_field_violations"] += _count_true(
                pc.and_(ambiguous, any_present)
            )
        implied_ambiguity = pc.or_(ambiguity_masks[0], pc.or_(ambiguity_masks[1], ambiguity_masks[2]))
        reported = _array(table, "any_auxiliary_ambiguity")
        scan.metrics["ambiguity_rows"] += _count_true(implied_ambiguity)
        scan.metrics["reported_ambiguity_rows"] += _count_true(reported)
        scan.violations["ambiguity_flag"] += _count_false(pc.equal(reported, implied_ambiguity))
        scan.violations["universe_row_not_preserved"] += _count_false(
            _array(table, "universe_row_preserved")
        )
        scan.violations["signal_after_scope"] += _count_true(
            pc.greater(_array(table, "signal_date"), pa.scalar(last_signal, type=pa.date32()))
        )


def _adjacent_lexicographic_less_count(
    table: pa.Table, names: Sequence[str]
) -> int:
    size = table.num_rows - 1
    if size <= 0:
        return 0
    prefix_equal: pa.Array = pa.array([True] * size, type=pa.bool_())
    less: pa.Array = pa.array([False] * size, type=pa.bool_())
    for name in names:
        values = _array(table, name)
        prior = values.slice(0, size)
        current = values.slice(1, size)
        prior_valid = pc.is_valid(prior)
        current_valid = pc.is_valid(current)
        both_valid = pc.and_(prior_valid, current_valid)
        current_less = pc.or_(
            pc.and_(pc.is_null(current), prior_valid),
            pc.and_(both_valid, pc.fill_null(pc.less(current, prior), False)),
        )
        current_equal = pc.or_(
            pc.and_(pc.is_null(prior), pc.is_null(current)),
            pc.and_(both_valid, pc.fill_null(pc.equal(current, prior), False)),
        )
        less = pc.or_(less, pc.and_(prefix_equal, current_less))
        prefix_equal = pc.and_(prefix_equal, current_equal)
    return _count_true(less)


def _adjacent_equal_count(table: pa.Table, names: Sequence[str]) -> int:
    size = table.num_rows - 1
    if size <= 0:
        return 0
    equal: pa.Array = pa.array([True] * size, type=pa.bool_())
    for name in names:
        values = _array(table, name)
        equal = pc.and_(
            equal,
            pc.fill_null(
                pc.equal(values.slice(1, size), values.slice(0, size)), False
            ),
        )
    return _count_true(equal)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _duckdb_duplicate_count(path: Path, primary_key: Sequence[str]) -> int:
    columns = ",".join(_quote_identifier(name) for name in primary_key)
    query = f"""
        SELECT coalesce(sum(n - 1), 0)::BIGINT
        FROM (
          SELECT count(*)::BIGINT AS n
          FROM read_parquet(?)
          GROUP BY {columns}
          HAVING count(*) > 1
        ) duplicates
    """
    with tempfile.TemporaryDirectory(prefix="auxiliary-verifier-duckdb-") as scratch:
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET preserve_insertion_order = false")
            connection.execute("SET temp_directory = ?", [scratch])
            value = connection.execute(query, [str(path)]).fetchone()[0]
        finally:
            connection.close()
    return int(value or 0)


def _duckdb_option_candidate_count_violations(path: Path) -> int:
    query = """
        SELECT coalesce(sum(group_rows), 0)::BIGINT
        FROM (
          SELECT
            count(*)::BIGINT AS group_rows,
            count(DISTINCT permno)::BIGINT
              * count(DISTINCT secid)::BIGINT AS expected_candidates,
            min(candidate_count)::BIGINT AS minimum_claim,
            max(candidate_count)::BIGINT AS maximum_claim
          FROM read_parquet(?)
          GROUP BY normalized_cusip8, link_start_date, link_end_date
        ) grouped
        WHERE group_rows != expected_candidates
           OR minimum_claim != expected_candidates
           OR maximum_claim != expected_candidates
    """
    with tempfile.TemporaryDirectory(prefix="auxiliary-verifier-duckdb-") as scratch:
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET preserve_insertion_order = false")
            connection.execute("SET temp_directory = ?", [scratch])
            value = connection.execute(query, [str(path)]).fetchone()[0]
        finally:
            connection.close()
    return int(value or 0)


def _sql_string(value: Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _point_in_time_formula_violations(
    output_paths: Mapping[str, Path],
    calendar_path: Path,
) -> dict[str, int]:
    """Recompute every frozen availability formula from normalized inputs."""

    names = (
        "compustat_annual_observation",
        "compustat_quarterly_observation",
        "compustat_segment_observation",
        "sec_filing_observation",
        "sec_8k_item_observation",
        "sec_13f_filing_observation",
        "sec_13f_holding_observation",
        "restatement_event_observation",
        "optionmetrics_option_volume_observation",
        "optionmetrics_historical_volatility_observation",
        "ff_factor_daily_observation",
        "ff_factor_monthly_observation",
        "ps_liquidity_observation",
    )
    missing = sorted(set(names) - set(output_paths))
    if missing:
        raise AuxiliaryBundleVerificationError(
            "PIT reconciliation inputs are missing: " + ", ".join(missing)
        )
    with tempfile.TemporaryDirectory(prefix="auxiliary-pit-duckdb-") as scratch:
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET preserve_insertion_order = false")
            connection.execute("SET threads = 1")
            connection.execute("SET TimeZone = 'UTC'")
            connection.execute("SET temp_directory = ?", [scratch])
            connection.execute(
                "CREATE TEMP VIEW verified_calendar AS "
                f"SELECT calendar_date FROM read_parquet({_sql_string(calendar_path)})"
            )
            for name in names:
                connection.execute(
                    f"CREATE TEMP VIEW {_quote_identifier(name)} AS "
                    f"SELECT * FROM read_parquet({_sql_string(output_paths[name])})"
                )

            violations: dict[str, int] = {}

            def aggregate(
                table: str,
                predicates: Mapping[str, str],
                *,
                from_sql: str | None = None,
            ) -> None:
                expressions = ", ".join(
                    "coalesce(sum(CASE WHEN "
                    + predicate
                    + f" THEN 1 ELSE 0 END), 0)::BIGINT AS {_quote_identifier(label)}"
                    for label, predicate in predicates.items()
                )
                row = connection.execute(
                    f"SELECT {expressions} FROM {from_sql or _quote_identifier(table)}"
                ).fetchone()
                for label, value in zip(predicates, row, strict=True):
                    violations[label] = int(value or 0)

            annual_sql = """
                (
                  WITH candidate_dates AS (
                    SELECT a.source_record_id, min(s.filing_date) AS first_filing_date
                    FROM compustat_annual_observation a
                    LEFT JOIN sec_filing_observation s
                      ON a.cik = s.cik
                     AND s.form_family IN ('10-K', '20-F', '40-F')
                     AND s.filing_date > a.datadate
                     AND s.filing_date <= a.datadate + INTERVAL 180 DAY
                    GROUP BY a.source_record_id
                  ), candidate_availability AS (
                    SELECT a.source_record_id,
                           d.first_filing_date,
                           min(s.public_availability_date) AS filing_availability
                    FROM compustat_annual_observation a
                    JOIN candidate_dates d USING (source_record_id)
                    LEFT JOIN sec_filing_observation s
                      ON a.cik = s.cik
                     AND s.form_family IN ('10-K', '20-F', '40-F')
                     AND s.filing_date = d.first_filing_date
                    GROUP BY a.source_record_id, d.first_filing_date
                  )
                  SELECT a.*,
                         CASE WHEN c.first_filing_date IS NOT NULL
                              THEN c.filing_availability
                              ELSE (SELECT min(calendar_date)
                                    FROM verified_calendar
                                    WHERE calendar_date >= a.datadate + INTERVAL 180 DAY)
                          END AS expected_availability,
                         CASE WHEN c.first_filing_date IS NOT NULL
                              THEN 'matched_10k_20f_40f_filing_plus_one_trading_day'
                              ELSE 'datadate_plus_180_calendar_days_fallback'
                          END AS expected_basis
                  FROM compustat_annual_observation a
                  JOIN candidate_availability c USING (source_record_id)
                ) annual_expected
            """
            aggregate(
                "compustat_annual_observation",
                {
                    "annual_observation_not_datadate": "observation_date IS DISTINCT FROM datadate",
                    "annual_availability_formula": "public_availability_date IS DISTINCT FROM expected_availability",
                    "annual_basis_formula": "availability_basis IS DISTINCT FROM expected_basis",
                    "annual_stale_formula": "stale_after_date IS DISTINCT FROM expected_availability + INTERVAL 540 DAY",
                },
                from_sql=annual_sql,
            )

            quarterly_sql = """
                (
                  SELECT q.*,
                         CASE WHEN rdq IS NOT NULL
                              THEN (SELECT min(calendar_date) FROM verified_calendar
                                    WHERE calendar_date > q.rdq)
                              ELSE (SELECT min(calendar_date) FROM verified_calendar
                                    WHERE calendar_date >= q.datadate + INTERVAL 120 DAY)
                          END AS expected_availability,
                         CASE WHEN rdq IS NOT NULL
                              THEN 'rdq_plus_one_verified_trading_day'
                              ELSE 'datadate_plus_120_calendar_days_fallback'
                          END AS expected_basis
                  FROM compustat_quarterly_observation q
                ) quarterly_expected
            """
            aggregate(
                "compustat_quarterly_observation",
                {
                    "quarterly_observation_not_datadate": "observation_date IS DISTINCT FROM datadate",
                    "quarterly_availability_formula": "public_availability_date IS DISTINCT FROM expected_availability",
                    "quarterly_basis_formula": "availability_basis IS DISTINCT FROM expected_basis",
                    "quarterly_stale_formula": "stale_after_date IS DISTINCT FROM expected_availability + INTERVAL 180 DAY",
                },
                from_sql=quarterly_sql,
            )

            segment_sql = """
                (
                  SELECT s.*,
                         CASE WHEN srcdate IS NOT NULL
                              THEN (SELECT min(calendar_date) FROM verified_calendar
                                    WHERE calendar_date > s.srcdate)
                              ELSE (SELECT min(calendar_date) FROM verified_calendar
                                    WHERE calendar_date >= s.datadate + INTERVAL 180 DAY)
                          END AS expected_availability,
                         CASE WHEN srcdate IS NOT NULL
                              THEN 'srcdate_plus_one_verified_trading_day'
                              ELSE 'datadate_plus_180_calendar_days_fallback'
                          END AS expected_basis
                  FROM compustat_segment_observation s
                ) segment_expected
            """
            aggregate(
                "compustat_segment_observation",
                {
                    "segment_observation_not_datadate": "observation_date IS DISTINCT FROM datadate",
                    "segment_availability_formula": "public_availability_date IS DISTINCT FROM expected_availability",
                    "segment_basis_formula": "availability_basis IS DISTINCT FROM expected_basis",
                    "segment_stale_formula": "stale_after_date IS DISTINCT FROM expected_availability + INTERVAL 540 DAY",
                },
                from_sql=segment_sql,
            )

            for table, source_column, basis, prefix in (
                (
                    "sec_filing_observation",
                    "filing_date",
                    "filing_date_plus_one_trading_day",
                    "sec_filing",
                ),
                (
                    "sec_13f_filing_observation",
                    "filing_date",
                    "13f_filing_or_amendment_date_plus_one_trading_day",
                    "thirteen_f_filing",
                ),
                (
                    "sec_13f_holding_observation",
                    "filing_date",
                    "13f_filing_or_amendment_date_plus_one_trading_day",
                    "thirteen_f_holding",
                ),
            ):
                expected = (
                    f"(SELECT min(calendar_date) FROM verified_calendar "
                    f"WHERE calendar_date > t.{source_column})"
                )
                predicates = {
                    f"{prefix}_observation_formula": (
                        f"observation_date IS DISTINCT FROM {source_column}"
                    ),
                    f"{prefix}_availability_formula": (
                        f"public_availability_date IS DISTINCT FROM {expected}"
                    ),
                    f"{prefix}_basis_formula": (
                        f"availability_basis IS DISTINCT FROM '{basis}'"
                    ),
                }
                if table == "sec_13f_filing_observation":
                    predicates[f"{prefix}_stale_formula"] = (
                        f"stale_after_date IS DISTINCT FROM {expected} + INTERVAL 180 DAY"
                    )
                aggregate(table, predicates, from_sql=f"{_quote_identifier(table)} t")

            eight_k_sql = """
                (
                  SELECT e.*,
                         CASE WHEN acceptance_date IS NOT NULL AND acceptance_time IS NOT NULL
                              THEN acceptance_date ELSE filing_date END AS expected_observation,
                         CASE WHEN acceptance_date IS NOT NULL AND acceptance_time IS NOT NULL
                              THEN 'validated_sec_acceptance_date_time_plus_one_trading_day'
                              ELSE 'filing_date_plus_one_trading_day_fallback' END AS expected_basis
                  FROM sec_8k_item_observation e
                ) eight_k_expected
            """
            aggregate(
                "sec_8k_item_observation",
                {
                    "eight_k_acceptance_pair_or_time": "(acceptance_date IS NULL) <> (acceptance_time IS NULL) OR (acceptance_time IS NOT NULL AND NOT regexp_full_match(acceptance_time, '^(?:[01][0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9](?:\\.[0-9]{1,6})?)?$'))",
                    "eight_k_observation_formula": "observation_date IS DISTINCT FROM expected_observation",
                    "eight_k_availability_formula": "public_availability_date IS DISTINCT FROM (SELECT min(calendar_date) FROM verified_calendar WHERE calendar_date > expected_observation)",
                    "eight_k_basis_formula": "availability_basis IS DISTINCT FROM expected_basis",
                },
                from_sql=eight_k_sql,
            )

            restatement_sql = """
                (
                  SELECT r.*,
                         CASE WHEN file_accepted IS NOT NULL
                              THEN CAST(file_accepted AS DATE) ELSE file_date END AS expected_observation,
                         CASE WHEN file_accepted IS NOT NULL
                              THEN 'file_accepted_plus_one_trading_day'
                              ELSE 'file_date_plus_one_trading_day_fallback' END AS expected_basis
                  FROM restatement_event_observation r
                ) restatement_expected
            """
            aggregate(
                "restatement_event_observation",
                {
                    "restatement_observation_formula": "observation_date IS DISTINCT FROM expected_observation",
                    "restatement_availability_formula": "public_availability_date IS DISTINCT FROM (SELECT min(calendar_date) FROM verified_calendar WHERE calendar_date > expected_observation)",
                    "restatement_basis_formula": "availability_basis IS DISTINCT FROM expected_basis",
                },
                from_sql=restatement_sql,
            )

            for table, prefix in (
                ("optionmetrics_option_volume_observation", "option_volume"),
                (
                    "optionmetrics_historical_volatility_observation",
                    "historical_volatility",
                ),
            ):
                aggregate(
                    table,
                    {
                        f"{prefix}_availability_formula": "public_availability_date IS DISTINCT FROM observation_date",
                        f"{prefix}_basis_formula": "availability_basis IS DISTINCT FROM 'DATED_CLOSE'",
                    },
                )
            for table, prefix in (
                ("ff_factor_daily_observation", "factor_daily"),
                ("ff_factor_monthly_observation", "factor_monthly"),
            ):
                predicates = {
                    f"{prefix}_availability_formula": "public_availability_date IS DISTINCT FROM observation_date",
                    f"{prefix}_basis_formula": "availability_basis IS DISTINCT FROM 'dated_factor_close'",
                }
                if table == "ff_factor_monthly_observation":
                    predicates[f"{prefix}_month_formula"] = (
                        "factor_month IS DISTINCT FROM observation_date"
                    )
                aggregate(table, predicates)

            ps_sql = """
                (
                  SELECT p.*,
                         (SELECT min(calendar_date) FROM verified_calendar
                          WHERE calendar_date >= last_day(p.observation_month + INTERVAL 1 MONTH))
                          AS expected_availability
                  FROM ps_liquidity_observation p
                ) ps_expected
            """
            aggregate(
                "ps_liquidity_observation",
                {
                    "ps_observation_formula": "observation_date IS DISTINCT FROM observation_month",
                    "ps_recorded_release_formula": "recorded_release_date IS NOT NULL",
                    "ps_availability_formula": "public_availability_date IS DISTINCT FROM expected_availability",
                    "ps_basis_formula": "availability_basis IS DISTINCT FROM 'one_full_calendar_month_lag_fallback'",
                },
                from_sql=ps_sql,
            )
            return dict(sorted(violations.items()))
        finally:
            connection.close()


def scan_table(
    path: Path,
    contract: Mapping[str, Any],
    *,
    batch_size: int,
    last_signal: dt.date,
    last_exit: dt.date,
) -> TableScan:
    """Fully scan one required table without importing primary computation."""

    table_name = str(contract["table"])
    primary_key = tuple(str(value) for value in contract["primary_key"])
    sort_key = tuple(str(value) for value in contract["sort_key"])
    parquet = pq.ParquetFile(path)
    scan = TableScan(table=table_name)
    if parquet.metadata.num_rows == 0:
        return scan
    previous_sort: tuple[tuple[int, Any], ...] | None = None
    previous_pk: tuple[Any, ...] | None = None
    pending_name_interval: tuple[int, dt.date] | None = None
    adjacent_pk_safe = tuple(sort_key[: len(primary_key)]) == primary_key

    required_fields = [field.name for field in parquet.schema_arrow if not field.nullable]
    for batch in parquet.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        scan.rows += table.num_rows
        for name in required_fields:
            scan.violations["required_value_null"] += table[name].null_count

        if table.num_rows:
            first_sort_values = tuple(_array(table, name)[0].as_py() for name in sort_key)
            rendered_first_sort = _sort_tuple(first_sort_values)
            if previous_sort is not None and rendered_first_sort < previous_sort:
                scan.violations["sort_order"] += 1
            scan.violations["sort_order"] += _adjacent_lexicographic_less_count(
                table, sort_key
            )
            previous_sort = _sort_tuple(
                tuple(_array(table, name)[table.num_rows - 1].as_py() for name in sort_key)
            )

            if adjacent_pk_safe:
                first_pk = tuple(_array(table, name)[0].as_py() for name in primary_key)
                if previous_pk is not None and first_pk == previous_pk:
                    scan.violations["duplicate_primary_key"] += 1
                scan.violations["duplicate_primary_key"] += _adjacent_equal_count(
                    table, primary_key
                )
                previous_pk = tuple(
                    _array(table, name)[table.num_rows - 1].as_py()
                    for name in primary_key
                )

            if table_name == "optionmetrics_name_interval":
                secids = _array(table, "secid")
                starts = _array(table, "effective_start_date")
                ends = _array(table, "effective_end_date")
                first_secid = int(secids[0].as_py())
                first_start = starts[0].as_py()
                if pending_name_interval is not None:
                    prior_secid, prior_end = pending_name_interval
                    expected_end = (
                        first_start - dt.timedelta(days=1)
                        if prior_secid == first_secid
                        else last_signal
                    )
                    scan.violations["name_interval_end_rule"] += (
                        prior_end != expected_end
                    )
                if table.num_rows > 1:
                    size = table.num_rows - 1
                    same_secid = pc.equal(
                        secids.slice(0, size), secids.slice(1, size)
                    )
                    prior_day = pc.cast(
                        pc.cast(
                            pc.subtract(
                                pc.cast(starts.slice(1, size), pa.int32()),
                                pa.scalar(1, type=pa.int32()),
                            ),
                            pa.int32(),
                        ),
                        pa.date32(),
                    )
                    expected_ends = pc.if_else(
                        same_secid,
                        prior_day,
                        pa.scalar(last_signal, type=pa.date32()),
                    )
                    scan.violations["name_interval_end_rule"] += _count_false(
                        pc.equal(ends.slice(0, size), expected_ends)
                    )
                pending_name_interval = (
                    int(secids[table.num_rows - 1].as_py()),
                    ends[table.num_rows - 1].as_py(),
                )

        _semantic_batch_vectorized(
            table_name,
            table,
            scan,
            contract=contract,
            last_signal=last_signal,
            last_exit=last_exit,
        )
    if not adjacent_pk_safe or scan.violations.get("sort_order", 0):
        # A non-prefix primary key needs a global uniqueness check.  A normally
        # prefix-safe key also needs one when the physical sort contract failed,
        # because equal keys are no longer guaranteed to be adjacent.
        scan.violations["duplicate_primary_key"] = _duckdb_duplicate_count(
            path, primary_key
        )
    if pending_name_interval is not None:
        scan.violations["name_interval_end_rule"] += (
            pending_name_interval[1] != last_signal
        )
    if table_name == "optionmetrics_crsp_link_interval":
        scan.violations[
            "candidate_count_group_consistency"
        ] += _duckdb_option_candidate_count_violations(path)
    if scan.rows != parquet.metadata.num_rows:
        scan.violations["metadata_row_count"] += abs(
            scan.rows - parquet.metadata.num_rows
        ) or 1
    return scan


def _iter_key_rows(path: Path, names: Sequence[str], batch_size: int) -> Iterator[tuple[Any, ...]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=list(names), batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        values = _values(table, names)
        for index in range(table.num_rows):
            yield tuple(values[name][index] for name in names)


def _compare_signal_spine(
    signal_path: Path,
    linkage_path: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    sentinel = object()
    signal_rows = linkage_rows = differences = signal_sort_violations = 0
    prior_signal: tuple[Any, ...] | None = None
    for signal, linkage in itertools.zip_longest(
        _iter_key_rows(signal_path, ("signal_date", "permno"), batch_size),
        _iter_key_rows(linkage_path, ("signal_date", "permno"), batch_size),
        fillvalue=sentinel,
    ):
        if signal is not sentinel:
            signal_rows += 1
            assert isinstance(signal, tuple)
            if prior_signal is not None and _sort_tuple(signal) <= _sort_tuple(prior_signal):
                signal_sort_violations += 1
            prior_signal = signal
        if linkage is not sentinel:
            linkage_rows += 1
        if signal is sentinel or linkage is sentinel or signal != linkage:
            differences += 1
    return {
        "signal_rows": signal_rows,
        "linkage_rows": linkage_rows,
        "primary_key_differences": differences,
        "signal_sort_or_duplicate_violations": signal_sort_violations,
    }


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def _factor_lock_aggregates(
    factor_lock: Mapping[str, Any], scans: Mapping[str, TableScan]
) -> tuple[dict[str, Any], dict[str, Any]]:
    mapping = {
        "daily": "ff_factor_daily_observation",
        "monthly": "ff_factor_monthly_observation",
    }
    expected: dict[str, Any] = {}
    actual: dict[str, Any] = {}
    for family, table_name in mapping.items():
        locked = factor_lock.get(family, {})
        expected[family] = {
            "rows": locked.get("rows_through_2014_12_01"),
            "minimum_date": locked.get("minimum_date"),
            "maximum_date": locked.get("maximum_date"),
            "maximum_absolute_mktrf": _decimal_text(
                Decimal(str(locked.get("maximum_absolute_mktrf", "NaN")))
            ),
            "maximum_absolute_rf": _decimal_text(
                Decimal(str(locked.get("maximum_absolute_rf", "NaN")))
            ),
        }
        scan = scans.get(table_name)
        if scan is None:
            actual[family] = None
            continue
        stats = scan.factor_stats or {}
        actual[family] = {
            "rows": stats.get("rows"),
            "minimum_date": _jsonable(stats.get("minimum_date")),
            "maximum_date": _jsonable(stats.get("maximum_date")),
            "maximum_absolute_mktrf": _decimal_text(
                Decimal(str(stats.get("maximum_absolute_mktrf", "NaN")))
            ),
            "maximum_absolute_rf": _decimal_text(
                Decimal(str(stats.get("maximum_absolute_rf", "NaN")))
            ),
        }
    return expected, actual


def _quality_receipts(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("quality_receipts", {})
    result: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            if isinstance(value, Mapping):
                receipt = dict(value)
                receipt.setdefault("path", str(name))
                result[str(name)] = receipt
    elif isinstance(raw, list):
        for value in raw:
            if isinstance(value, Mapping) and isinstance(value.get("path"), str):
                result[str(value["path"])] = dict(value)
    return result


def _sensitive_examples(value: Any, *, path: tuple[str, ...] = ()) -> list[str]:
    result: list[str] = []
    sensitive = {
        "permno",
        "permco",
        "gvkey",
        "cik",
        "cusip",
        "cusip8",
        "secid",
        "ticker",
        "company_name",
        "issuer_name",
        "source_record_id",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            child = (*path, str(key))
            if key_text in sensitive and item not in (None, [], {}, ""):
                result.append(".".join(child))
            result.extend(_sensitive_examples(item, path=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_sensitive_examples(item, path=(*path, str(index))))
    return result


def _receipt_identifier(path: str, value: Mapping[str, Any]) -> str:
    identifier = value.get("receipt_id", value.get("check_id", value.get("name")))
    return str(identifier) if isinstance(identifier, str) else Path(path).stem


def _primary_quality_check_rows(
    schema_document: Mapping[str, Any],
    scans: Mapping[str, TableScan],
    linkage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    interval_columns = {
        "ccm_link_interval": "link_start_date",
        "optionmetrics_name_interval": "effective_start_date",
        "optionmetrics_crsp_link_interval": "link_start_date",
    }
    for contract in schema_document.get("tables", []):
        if not isinstance(contract, Mapping):
            continue
        name = str(contract["table"])
        scan = scans.get(name)
        if scan is None:
            continue
        expected_schema = expected_table_schema(schema_document, name)
        if "public_availability_date" in expected_schema.names:
            violations = int(scan.violations.get("availability_after_scope", 0))
            rows.append(
                {
                    "check": f"{name}.availability_within_development_scope",
                    "passed": violations == 0,
                    "actual": violations,
                    "expected": 0,
                }
            )
        start_name = interval_columns.get(name)
        if start_name is not None:
            violations = int(
                scan.violations.get(f"invalid_interval_{start_name}", 0)
            )
            rows.append(
                {
                    "check": f"{name}.interval_start_not_after_end",
                    "passed": violations == 0,
                    "actual": violations,
                    "expected": 0,
                }
            )
    signal_rows = linkage.get("signal_rows")
    linkage_rows = linkage.get("linkage_rows")
    keys_preserved = (
        signal_rows == linkage_rows
        and linkage.get("primary_key_differences") == 0
        and linkage.get("signal_sort_or_duplicate_violations") == 0
    )
    linkage_scan = scans.get("auxiliary_signal_linkage")
    all_preserved = bool(
        linkage_scan is not None
        and linkage_scan.violations.get("universe_row_not_preserved", 0) == 0
    )
    ambiguous_field_violations = (
        int(linkage_scan.metrics.get("ambiguous_field_violations", 0))
        if linkage_scan is not None
        else 1
    )
    rows.extend(
        [
            {
                "check": "auxiliary_signal_linkage.row_count_preserved",
                "passed": signal_rows == linkage_rows,
                "actual": linkage_rows,
                "expected": signal_rows,
            },
            {
                "check": "auxiliary_signal_linkage.primary_key_and_order_preserved",
                "passed": keys_preserved,
                "actual": int(keys_preserved),
                "expected": 1,
            },
            {
                "check": "auxiliary_signal_linkage.all_rows_marked_preserved",
                "passed": all_preserved,
                "actual": int(all_preserved),
                "expected": 1,
            },
            {
                "check": "auxiliary_signal_linkage.ambiguous_fields_null",
                "passed": ambiguous_field_violations == 0,
                "actual": ambiguous_field_violations,
                "expected": 0,
            },
        ]
    )
    return rows


def _primary_aggregate_qa(
    settings: AuxiliaryVerificationSettings,
    schema_document: Mapping[str, Any],
    scans: Mapping[str, TableScan],
    linkage: Mapping[str, Any],
    factor_lock: Mapping[str, Any],
    output_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    availability: dict[str, Any] = {}
    for contract in schema_document.get("tables", []):
        if not isinstance(contract, Mapping):
            continue
        name = str(contract["table"])
        scan = scans.get(name)
        if scan is None:
            continue
        if "public_availability_date" in expected_table_schema(
            schema_document, name
        ).names:
            availability[name] = {
                "max_public_availability_date": _jsonable(scan.availability_max)
            }
    quality_checks = _primary_quality_check_rows(schema_document, scans, linkage)
    linkage_scan = scans.get("auxiliary_signal_linkage")
    keys_preserved = (
        linkage.get("signal_rows") == linkage.get("linkage_rows")
        and linkage.get("primary_key_differences") == 0
        and linkage.get("signal_sort_or_duplicate_violations") == 0
    )
    all_preserved = bool(
        linkage_scan is not None
        and linkage_scan.violations.get("universe_row_not_preserved", 0) == 0
    )
    linkage_path = output_paths["auxiliary_signal_linkage"]
    sec_summary, sec_detail = _sec_form_coverage_qa(
        settings.signal_universe_path,
        linkage_path,
        output_paths["sec_filing_observation"],
        batch_size=settings.batch_size,
    )
    option_summary, option_detail = _optionmetrics_coverage_qa(
        settings.signal_universe_path,
        linkage_path,
        output_paths["optionmetrics_option_volume_observation"],
        batch_size=settings.batch_size,
    )
    factor_summary = _factor_alignment_qa(settings, factor_lock, scans)
    ps_summary = _ps_release_lag_qa(scans)
    summary = {
        "schema_version": 1,
        "status": PRIMARY_STATUS,
        "checks_passed": sum(bool(item["passed"]) for item in quality_checks),
        "checks_failed": sum(not bool(item["passed"]) for item in quality_checks),
        "availability": availability,
        "linkage": {
            "source_spine_rows": linkage.get("signal_rows"),
            "linkage_rows": linkage.get("linkage_rows"),
            "primary_key_preserved": keys_preserved,
            "all_rows_preserved": all_preserved,
            "status_counts": (
                dict(sorted(linkage_scan.status_counts.items()))
                if linkage_scan is not None
                else {}
            ),
            "ambiguity_rows": (
                int(linkage_scan.metrics.get("reported_ambiguity_rows", 0))
                if linkage_scan is not None
                else None
            ),
        },
        "sec_form_coverage": sec_summary,
        "optionmetrics_coverage": option_summary,
        "factor_alignment": factor_summary,
        "ps_release_lag": ps_summary,
        "authorizations": {
            "auxiliary_panel_authorized": False,
            "features_labels_models_portfolios_pnl_performance_or_backtests_authorized": False,
            "validation_or_final_test_access_authorized": False,
        },
    }
    return summary, {
        "sec_form_coverage.csv": sec_detail,
        "optionmetrics_coverage.csv": option_detail,
    }


def _csv_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise AuxiliaryBundleVerificationError(f"Invalid CSV header: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise AuxiliaryBundleVerificationError(f"Invalid CSV width: {path}")
    return rows


def _quality_check_csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "check": str(row["check"]),
            "passed": str(bool(row["passed"])),
            "actual": str(row["actual"]),
            "expected": str(row["expected"]),
        }
        for row in rows
    ]


def _linkage_coverage_rows(
    signal_path: Path,
    linkage_path: Path,
    *,
    batch_size: int,
) -> list[dict[str, str]]:
    signal_columns = (
        "signal_date",
        "primary_exchange",
        "us_incorporation_flag",
        "share_type",
        "long_eligible",
        "short_eligible",
    )
    link_columns = (
        "ccm_match_status",
        "sec_match_status",
        "optionmetrics_match_status",
    )
    signal_schema = pq.ParquetFile(signal_path).schema_arrow
    if not set(signal_columns) <= set(signal_schema.names):
        raise AuxiliaryBundleVerificationError(
            "Signal universe lacks dimensions required to verify linkage_coverage.csv"
        )
    counts: Counter[tuple[Any, ...]] = Counter()
    sentinel = object()
    signal_rows = _iter_key_rows(signal_path, signal_columns, batch_size)
    linkage_rows = _iter_key_rows(linkage_path, link_columns, batch_size)
    for signal, statuses in itertools.zip_longest(
        signal_rows, linkage_rows, fillvalue=sentinel
    ):
        if signal is sentinel or statuses is sentinel:
            raise AuxiliaryBundleVerificationError(
                "Cannot reconstruct linkage coverage from unequal row counts"
            )
        assert isinstance(signal, tuple) and isinstance(statuses, tuple)
        day = signal[0]
        if not isinstance(day, dt.date):
            raise AuxiliaryBundleVerificationError("Signal date is not date32")
        counts[(day.year, *signal[1:], *statuses)] += 1
    fieldnames = (
        "signal_year",
        "primary_exchange",
        "us_incorporation_flag",
        "share_type",
        "long_eligible",
        "short_eligible",
        "ccm_match_status",
        "sec_match_status",
        "optionmetrics_match_status",
        "rows",
    )
    return [
        {
            name: str(value)
            for name, value in zip(fieldnames, (*key, count), strict=True)
        }
        for key, count in sorted(
            counts.items(), key=lambda item: tuple(str(value) for value in item[0])
        )
    ]


def _sec_form_coverage_qa(
    signal_path: Path,
    linkage_path: Path,
    filing_path: Path,
    *,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    forms = ("10-K", "10-Q", "20-F", "40-F", "8-K", "6-K")
    filing_dates: dict[tuple[str, str], list[int]] = defaultdict(list)
    epoch = dt.date(1970, 1, 1)
    for values in _iter_key_rows(
        filing_path,
        ("cik", "form_family", "public_availability_date"),
        batch_size,
    ):
        cik, form, available = values
        if form in forms and isinstance(available, dt.date):
            filing_dates[(str(cik), str(form))].append((available - epoch).days)
    for dates in filing_dates.values():
        dates.sort()

    signal_columns = ("signal_date", "us_incorporation_flag", "share_type")
    link_columns = ("sec_match_status", "sec_cik")
    totals: Counter[tuple[int, str, bool, str]] = Counter()
    covered: Counter[tuple[int, str, bool, str]] = Counter()
    sentinel = object()
    signal_rows = 0
    for signal, link in itertools.zip_longest(
        _iter_key_rows(signal_path, signal_columns, batch_size),
        _iter_key_rows(linkage_path, link_columns, batch_size),
        fillvalue=sentinel,
    ):
        if signal is sentinel or link is sentinel:
            raise AuxiliaryBundleVerificationError(
                "Cannot reconstruct SEC form coverage from unequal spines"
            )
        assert isinstance(signal, tuple) and isinstance(link, tuple)
        day, domicile, share_type = signal
        status, cik = link
        if not isinstance(day, dt.date):
            raise AuxiliaryBundleVerificationError("Signal date is not date32")
        signal_rows += 1
        ordinal = (day - epoch).days
        adr = share_type == "AD"
        for form in forms:
            key = (day.year, str(domicile), adr, form)
            totals[key] += 1
            dates = (
                filing_dates.get((str(cik), form), ())
                if status == "MATCHED" and cik
                else ()
            )
            if dates and bisect_right(dates, ordinal):
                covered[key] += 1
    fieldnames = (
        "signal_year",
        "us_incorporation_flag",
        "adr_share_type",
        "form_family",
        "rows",
        "covered_rows",
        "missing_rows",
    )
    detail = [
        {
            name: str(value)
            for name, value in zip(
                fieldnames,
                (*key, totals[key], covered[key], totals[key] - covered[key]),
                strict=True,
            )
        }
        for key in sorted(totals, key=lambda value: tuple(str(item) for item in value))
    ]
    summary = {
        "forms": list(forms),
        "aggregate_rows": len(totals),
        "signal_rows_per_form": signal_rows,
        "current_company_name_join_used": False,
        "foreign_form_absence_removed_spine_row": False,
    }
    return summary, detail


def _optionmetrics_coverage_qa(
    signal_path: Path,
    linkage_path: Path,
    volume_path: Path,
    *,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    fieldnames = (
        "signal_year",
        "primary_exchange",
        "us_incorporation_flag",
        "share_type",
        "long_eligible",
        "short_eligible",
        "optionmetrics_match_status",
        "rows",
        "rows_with_dated_vendor_flow_coverage",
        "genuine_zero_volume_rows",
        "genuine_zero_open_interest_rows",
    )
    # The authoritative volume table can contain hundreds of millions of rows.
    # Keep all high-cardinality state inside DuckDB, restrict aggregation to the
    # signal-date/secid pairs that can affect this receipt, and permit /tmp spill.
    query = """
        WITH spine AS (
          SELECT
            u.signal_date,
            u.primary_exchange,
            u.us_incorporation_flag,
            u.share_type,
            u.long_eligible,
            u.short_eligible,
            l.optionmetrics_match_status,
            l.optionmetrics_secid
          FROM read_parquet(?) u
          JOIN read_parquet(?) l USING (signal_date, permno)
        ), relevant AS (
          SELECT DISTINCT
            optionmetrics_secid AS secid,
            signal_date AS observation_date
          FROM spine
          WHERE optionmetrics_secid IS NOT NULL
        ), flow AS (
          SELECT
            v.secid,
            v.observation_date,
            count(*) AS component_rows,
            count(v.volume) AS nonnull_volume_rows,
            count(v.open_interest) AS nonnull_open_interest_rows,
            bool_and(coalesce(v.genuine_zero_volume, false))
              AS all_observed_volume_components_zero,
            bool_and(coalesce(v.genuine_zero_open_interest, false))
              AS all_observed_open_interest_components_zero
          FROM read_parquet(?) v
          JOIN relevant r USING (secid, observation_date)
          WHERE v.authoritative_flow_component
          GROUP BY v.secid, v.observation_date
        )
        SELECT
          year(s.signal_date) AS signal_year,
          s.primary_exchange,
          s.us_incorporation_flag,
          s.share_type,
          s.long_eligible,
          s.short_eligible,
          s.optionmetrics_match_status,
          count(*) AS rows,
          count(f.secid) AS rows_with_dated_vendor_flow_coverage,
          sum(CASE WHEN f.component_rows > 0
                        AND f.nonnull_volume_rows = f.component_rows
                        AND f.all_observed_volume_components_zero
                   THEN 1 ELSE 0 END) AS genuine_zero_volume_rows,
          sum(CASE WHEN f.component_rows > 0
                        AND f.nonnull_open_interest_rows = f.component_rows
                        AND f.all_observed_open_interest_components_zero
                   THEN 1 ELSE 0 END) AS genuine_zero_open_interest_rows
        FROM spine s
        LEFT JOIN flow f
          ON s.optionmetrics_secid = f.secid
         AND s.signal_date = f.observation_date
        GROUP BY ALL
        ORDER BY 1,2,3,4,5,6,7
    """
    with tempfile.TemporaryDirectory(
        prefix="auxiliary-verifier-option-qa-"
    ) as scratch:
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET memory_limit = '768MB'")
            connection.execute("SET preserve_insertion_order = false")
            connection.execute("SET temp_directory = ?", [scratch])
            values = connection.execute(
                query,
                [str(signal_path), str(linkage_path), str(volume_path)],
            ).fetchall()
        finally:
            connection.close()
    detail = [
        {
            name: (str(value).lower() if isinstance(value, bool) else str(value))
            for name, value in zip(fieldnames, row, strict=True)
        }
        for row in values
    ]
    summary = {
        "aggregate_rows": len(detail),
        "authoritative_flow_roles": ["CALL", "PUT"],
        "total_or_blank_added_to_call_plus_put": False,
        "genuine_zero_requires_dated_vendor_flow_coverage": True,
        "option_coverage_used_as_universe_filter": False,
    }
    return summary, detail


def _factor_alignment_qa(
    settings: AuxiliaryVerificationSettings,
    factor_lock: Mapping[str, Any],
    scans: Mapping[str, TableScan],
) -> dict[str, Any]:
    daily = scans["ff_factor_daily_observation"].factor_stats or {}
    monthly = scans["ff_factor_monthly_observation"].factor_stats or {}
    return {
        "factor_unit_lock_sha256": sha256_path(settings.factor_unit_lock_path),
        "selected_source_to_decimal_ratio_multiplier": str(
            factor_lock.get("selected_source_to_decimal_ratio_multiplier")
        ),
        "source_values_are_decimal_ratios": bool(
            factor_lock.get("source_values_are_decimal_ratios")
        ),
        "daily_rows": daily.get("rows"),
        "monthly_rows": monthly.get("rows"),
        "daily_min_date": _jsonable(daily.get("minimum_date")),
        "daily_max_date": _jsonable(daily.get("maximum_date")),
        "monthly_min_date": _jsonable(monthly.get("minimum_date")),
        "monthly_max_date": _jsonable(monthly.get("maximum_date")),
        "daily_monthly_crosscheck": factor_lock.get("daily_monthly_crosscheck"),
        "models_portfolios_or_backtests_used_for_unit_selection": False,
    }


def _ps_release_lag_qa(scans: Mapping[str, TableScan]) -> dict[str, Any]:
    scan = scans["ps_liquidity_observation"]
    return {
        "rows": scan.rows,
        "violations": int(scan.metrics.get("ps_policy_violations", 0)),
        "minimum_calendar_lag_days": (
            int(scan.metrics["ps_min_lag_days"])
            if scan.metrics.get("ps_lag_rows", 0)
            else None
        ),
        "maximum_calendar_lag_days": (
            int(scan.metrics["ps_max_lag_days"])
            if scan.metrics.get("ps_lag_rows", 0)
            else None
        ),
        "raw_PS_INNOV_present": False,
        "recursive_innovation_constructed": False,
        "historically_revised_series_disclosed": True,
    }


def _validate_quality_receipts(
    settings: AuxiliaryVerificationSettings,
    manifest: Mapping[str, Any],
    source_config: Mapping[str, Any],
    policy: Mapping[str, Any],
    schema_document: Mapping[str, Any],
    stage_01_hashes: Mapping[str, str],
    aggregate_qa: Mapping[str, Any],
    detail_expectations: Mapping[str, list[dict[str, str]]],
    scans: Mapping[str, TableScan],
    linkage: Mapping[str, Any],
    linkage_path: Path,
    checks: _Checks,
) -> None:
    receipts = _quality_receipts(manifest)
    expected_quality_files = sorted(
        str(receipt.get("path", encoded_path))
        for encoded_path, receipt in receipts.items()
    )
    checks.add(
        "QUALITY_DIRECTORY_CLOSED_SET",
        category="authentication",
        expected={"files": expected_quality_files, "non_regular": []},
        actual=_flat_directory_inventory(settings.quality_dir),
        evidence="quality directory contains exactly the manifest-hashed aggregate receipts",
    )
    configured_source_ids = {
        str(receipt_id)
        for item in source_config.get("inputs", [])
        if isinstance(item, Mapping)
        for receipt_id in item.get("qa_receipts", [])
    }
    hard_gates = policy.get("hard_quality_gates", {})
    configured_gate_ids = {
        gate.removesuffix("_required")
        for gate in _AGGREGATE_QUALITY_GATES
        if hard_gates.get(gate) is True
    }
    named_required_ids = configured_source_ids | configured_gate_ids
    hash_expected: dict[str, Any] = {}
    hash_actual: dict[str, Any] = {}
    identifiers: set[str] = set()
    structural_failures: list[dict[str, Any]] = []
    source_receipts: dict[str, list[Mapping[str, Any]]] = {}
    parsed_json: dict[str, Mapping[str, Any]] = {}
    parsed_csv: dict[str, list[dict[str, str]]] = {}

    for encoded_path, receipt in sorted(receipts.items()):
        path = _safe_receipt_path(settings.quality_dir, receipt.get("path", encoded_path))
        hash_expected[encoded_path] = _receipt_expected(receipt)
        if not path.is_file():
            hash_actual[encoded_path] = None
            continue
        hash_actual[encoded_path] = _hash_receipt(path)
        parsed: Mapping[str, Any] | None = None
        if path.suffix.lower() == ".json":
            try:
                value = json.loads(path.read_text("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                structural_failures.append(
                    {"path": encoded_path, "problem": f"invalid_json:{type(exc).__name__}"}
                )
            else:
                if isinstance(value, Mapping):
                    parsed = value
                    parsed_json[path.name] = value
                else:
                    structural_failures.append(
                        {"path": encoded_path, "problem": "json_not_object"}
                    )
        elif path.suffix.lower() == ".csv":
            try:
                parsed_csv[path.name] = _csv_dict_rows(path)
            except (OSError, UnicodeError, csv.Error, StopIteration, ValueError) as exc:
                structural_failures.append(
                    {"path": encoded_path, "problem": f"invalid_csv:{type(exc).__name__}"}
                )
        else:
            structural_failures.append(
                {"path": encoded_path, "problem": "unsupported_receipt_type"}
            )
        if parsed is None:
            identifiers.add(path.stem)
            continue
        identifier = _receipt_identifier(encoded_path, parsed)
        identifiers.add(identifier)
        if path.name == "summary.json":
            passing_statuses = {PRIMARY_STATUS}
        elif identifier in named_required_ids:
            passing_statuses = {"passed"}
        else:
            passing_statuses = set()
        if passing_statuses and parsed.get("status") not in passing_statuses:
            structural_failures.append(
                {"path": encoded_path, "problem": "nonpassing_or_missing_status"}
            )
        sensitive = _sensitive_examples(parsed)
        if sensitive:
            structural_failures.append(
                {
                    "path": encoded_path,
                    "problem": "raw_identifier_examples",
                    "locations": sensitive,
                }
            )
        source = parsed.get("source")
        if isinstance(source, str):
            source_receipts.setdefault(source, []).append(parsed)

    checks.add(
        "QUALITY_RECEIPT_FILE_HASHES",
        category="authentication",
        expected=hash_expected,
        actual=hash_actual,
        evidence="every aggregate quality receipt hashed and sized independently",
    )

    required_ids = named_required_ids | {
        "aggregate_qa",
        "checks",
        "linkage_coverage",
        "optionmetrics_coverage",
        "raw_source_hashes",
        "sec_form_coverage",
        "summary",
        "table_counts",
    }
    checks.add(
        "REQUIRED_AGGREGATE_QUALITY_RECEIPTS",
        category="quality",
        expected={"required_ids": sorted(required_ids), "structural_failures": []},
        actual={
            "required_ids": sorted(identifiers & required_ids),
            "structural_failures": structural_failures,
        },
        evidence="frozen source-config receipts plus policy hard-gate aggregate receipts",
    )

    qa_source_actual: dict[str, Any] = {}
    qa_source_expected: dict[str, Any] = {}
    for item in source_config.get("inputs", []):
        if not isinstance(item, Mapping) or item.get("disposition") != "qa_scan_only_no_row_level_materialization":
            continue
        source = str(item["filename"])
        required_for_source = set(map(str, item.get("qa_receipts", [])))
        values = source_receipts.get(source, [])
        qa_source_expected[source] = {
            "receipt_ids": sorted(required_for_source),
            "source_sha256": stage_01_hashes.get(source),
            "row_level_materialized": False,
        }
        qa_source_actual[source] = {
            "receipt_ids": sorted(_receipt_identifier("", value) for value in values),
            "source_sha256": (
                values[0].get("source_sha256") if values and all(
                    value.get("source_sha256") == values[0].get("source_sha256")
                    for value in values
                ) else None
            ),
            "row_level_materialized": (
                any(bool(value.get("row_level_materialized", value.get("row_level_table_materialized", False))) for value in values)
            ),
        }
    checks.add(
        "QA_ONLY_SOURCE_RECEIPTS_BIND_STAGE_01",
        category="quality",
        expected=qa_source_expected,
        actual=qa_source_actual,
        evidence="aggregate-only raw QA identities; no raw source was reopened",
    )

    checks.add(
        "AGGREGATE_QA_RECONSTRUCTED",
        category="aggregate",
        expected=_jsonable(manifest.get("aggregate_qa")),
        actual=_jsonable(aggregate_qa),
        evidence="independent full-table aggregates and exact signal-spine comparison",
    )

    checks.add(
        "PRIMARY_SUMMARY_RECEIPT_RECONSTRUCTED",
        category="aggregate",
        expected=_jsonable(aggregate_qa),
        actual=_jsonable(parsed_json.get("summary.json")),
        evidence="summary.json independently reconstructed from complete Parquet scans",
    )
    checks.add(
        "PRIMARY_AGGREGATE_QA_RECEIPT_RECONSTRUCTED",
        category="aggregate",
        expected={"status": "passed", "aggregate_qa": _jsonable(aggregate_qa)},
        actual=_jsonable(parsed_json.get("aggregate_qa.json")),
        evidence="aggregate_qa.json independently reconstructed from complete Parquet scans",
    )
    checks.add(
        "RAW_SOURCE_HASH_RECEIPT_RECONCILED",
        category="aggregate",
        expected=_jsonable(manifest.get("raw_source_hashes")),
        actual=_jsonable(parsed_json.get("raw_source_hashes.json")),
        evidence="aggregate raw-identity receipt reconciled to manifest and Stage-1 evidence",
    )

    expected_check_rows = _quality_check_csv_rows(
        _primary_quality_check_rows(schema_document, scans, linkage)
    )
    checks.add(
        "PRIMARY_CHECKS_CSV_RECONSTRUCTED",
        category="aggregate",
        expected=expected_check_rows,
        actual=parsed_csv.get("checks.csv"),
        evidence="primary availability, interval, and linkage checks independently reconstructed",
    )
    expected_table_counts = [
        {
            "table": str(contract["table"]),
            "rows": str(scans[str(contract["table"])].rows),
        }
        for contract in schema_document.get("tables", [])
        if isinstance(contract, Mapping) and str(contract["table"]) in scans
    ]
    checks.add(
        "TABLE_COUNTS_CSV_RECONSTRUCTED",
        category="aggregate",
        expected=expected_table_counts,
        actual=parsed_csv.get("table_counts.csv"),
        evidence="table_counts.csv rebuilt from all eighteen Parquet scans",
    )
    expected_coverage = _linkage_coverage_rows(
        settings.signal_universe_path,
        linkage_path,
        batch_size=settings.batch_size,
    )
    checks.add(
        "LINKAGE_COVERAGE_CSV_RECONSTRUCTED",
        category="aggregate",
        expected=expected_coverage,
        actual=parsed_csv.get("linkage_coverage.csv"),
        evidence="coverage groups rebuilt from all aligned signal and linkage rows",
    )
    checks.add(
        "SEC_FORM_COVERAGE_CSV_RECONSTRUCTED",
        category="aggregate",
        expected=detail_expectations.get("sec_form_coverage.csv"),
        actual=parsed_csv.get("sec_form_coverage.csv"),
        evidence="point-in-time SEC form coverage independently rebuilt without identifiers",
    )
    checks.add(
        "OPTIONMETRICS_COVERAGE_CSV_RECONSTRUCTED",
        category="aggregate",
        expected=detail_expectations.get("optionmetrics_coverage.csv"),
        actual=parsed_csv.get("optionmetrics_coverage.csv"),
        evidence="dated vendor-flow and genuine-zero coverage independently rebuilt",
    )

    def without_control(value: Mapping[str, Any] | None) -> Any:
        if value is None:
            return None
        return {
            key: item
            for key, item in value.items()
            if key not in {"receipt_id", "check_id", "status"}
        }

    sec_summary = dict(aggregate_qa.get("sec_form_coverage", {}))
    sec_summary.update(
        {
            "detail_file": "sec_form_coverage.csv",
            "detail_sha256": (
                sha256_path(settings.quality_dir / "sec_form_coverage.csv")
                if (settings.quality_dir / "sec_form_coverage.csv").is_file()
                else None
            ),
        }
    )
    option_summary = dict(aggregate_qa.get("optionmetrics_coverage", {}))
    option_summary.update(
        {
            "detail_file": "optionmetrics_coverage.csv",
            "detail_sha256": (
                sha256_path(settings.quality_dir / "optionmetrics_coverage.csv")
                if (settings.quality_dir / "optionmetrics_coverage.csv").is_file()
                else None
            ),
        }
    )
    linkage_summary = aggregate_qa.get("linkage", {})
    expected_gate_claims = {
        "before_after_link_counts_by_family.json": {
            "before_rows": linkage_summary.get("source_spine_rows"),
            "after_rows": linkage_summary.get("linkage_rows"),
            "primary_key_preserved": linkage_summary.get("primary_key_preserved"),
            "all_rows_preserved": linkage_summary.get("all_rows_preserved"),
            "status_counts": linkage_summary.get("status_counts"),
        },
        "missingness_by_family_year_exchange_domicile_adr_and_long_short.json": {
            "detail_file": "linkage_coverage.csv",
            "detail_sha256": (
                sha256_path(settings.quality_dir / "linkage_coverage.csv")
                if (settings.quality_dir / "linkage_coverage.csv").is_file()
                else None
            ),
            "aggregate_rows": len(expected_coverage),
            "dimensions": [
                "signal_year",
                "primary_exchange",
                "us_incorporation_flag",
                "share_type",
                "long_eligible",
                "short_eligible",
                "family_match_status",
            ],
            "raw_identifiers_present": False,
        },
        "sec_form_coverage_by_year_domicile_adr_and_form.json": sec_summary,
        "optionmetrics_match_genuine_zero_optionability_by_year.json": option_summary,
        "factor_date_alignment_and_unit_receipt.json": aggregate_qa.get(
            "factor_alignment"
        ),
        "ps_release_lag_receipt.json": aggregate_qa.get("ps_release_lag"),
    }
    checks.add(
        "NAMED_POLICY_GATE_RECEIPTS_RECONSTRUCTED",
        category="aggregate",
        expected=expected_gate_claims,
        actual={
            name: without_control(parsed_json.get(name))
            for name in expected_gate_claims
        },
        evidence="all six explicit policy gate summaries independently reconstructed",
    )


def render_differences_csv(report: Mapping[str, Any]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["check_id", "category", "expected_json", "actual_json"])
    for row in report.get("checks", []):
        if row.get("status") != "FAIL":
            continue
        writer.writerow(
            [
                row.get("check_id"),
                row.get("category"),
                json.dumps(_jsonable(row.get("expected")), sort_keys=True, separators=(",", ":")),
                json.dumps(_jsonable(row.get("actual")), sort_keys=True, separators=(",", ":")),
            ]
        )
    return stream.getvalue().encode("utf-8")


def derive_verification_id(
    report_without_id: Mapping[str, Any], differences_payload: bytes
) -> str:
    value = dict(report_without_id)
    value.pop("verification_id", None)
    digest = hashlib.sha256()
    digest.update(canonical_json(value))
    digest.update(differences_payload)
    return "verify-" + digest.hexdigest()[:16]


def validate_verification_payloads(
    independent_check_payload: bytes,
    differences_payload: bytes,
    *,
    require_passing: bool = False,
) -> dict[str, Any]:
    """Validate canonical, hash-bound verifier bytes before immutable publication."""

    try:
        report = json.loads(independent_check_payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuxiliaryBundleVerificationError("Independent check is not valid JSON") from exc
    if not isinstance(report, dict) or canonical_json(report) != independent_check_payload:
        raise AuxiliaryBundleVerificationError("Independent check is not canonical JSON")
    required_report_keys = {
        "schema_version",
        "status",
        "build_id",
        "contract_id",
        "primary_manifest_sha256",
        "verifier_implementation_sha256",
        "checks",
        "summary",
        "independence",
        "authorizations",
        "verification_id",
    }
    if set(report) != required_report_keys or report.get("schema_version") != 1:
        raise AuxiliaryBundleVerificationError("Independent check schema differs")
    if report.get("contract_id") != CONTRACT_ID or re.fullmatch(
        r"auxiliary-[0-9a-f]{24}", str(report.get("build_id", ""))
    ) is None:
        raise AuxiliaryBundleVerificationError(
            "Independent check contract or build identity differs"
        )
    for digest_name in (
        "primary_manifest_sha256",
        "verifier_implementation_sha256",
    ):
        if _SHA256_RE.fullmatch(str(report.get(digest_name, ""))) is None:
            raise AuxiliaryBundleVerificationError(
                f"Independent check {digest_name} is malformed"
            )
    check_rows = report.get("checks")
    if not isinstance(check_rows, list):
        raise AuxiliaryBundleVerificationError("Independent checks are not a list")
    check_keys = {
        "check_id",
        "category",
        "status",
        "expected",
        "actual",
        "evidence",
    }
    check_ids: list[str] = []
    for row in check_rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != check_keys
            or row.get("status") not in {"PASS", "FAIL"}
            or not isinstance(row.get("check_id"), str)
            or not isinstance(row.get("category"), str)
            or not isinstance(row.get("evidence"), str)
            or not row.get("evidence")
            or (row.get("expected") == row.get("actual"))
            != (row.get("status") == "PASS")
        ):
            raise AuxiliaryBundleVerificationError(
                "Independent check row schema or comparison differs"
            )
        check_ids.append(str(row["check_id"]))
    if (
        check_ids != sorted(check_ids)
        or len(check_ids) != len(set(check_ids))
        or set(check_ids) != _REQUIRED_CHECK_IDS
    ):
        raise AuxiliaryBundleVerificationError(
            "Independent required check inventory differs"
        )
    if render_differences_csv(report) != differences_payload:
        raise AuxiliaryBundleVerificationError("Differences bytes do not match report failures")
    expected_id = derive_verification_id(report, differences_payload)
    if report.get("verification_id") != expected_id:
        raise AuxiliaryBundleVerificationError("Verification ID is not payload-bound")
    failures = sum(row.get("status") == "FAIL" for row in check_rows)
    summary = report.get("summary", {})
    summary_keys = {
        "check_count",
        "pass_count",
        "fail_count",
        "difference_count",
        "required_tables",
        "tables_fully_scanned",
        "parquet_rows_scanned",
        "signal_universe_rows_compared",
    }
    if (
        not isinstance(summary, Mapping)
        or set(summary) != summary_keys
        or any(type(summary[name]) is not int or summary[name] < 0 for name in summary)
        or summary["check_count"] != len(check_rows)
        or summary["pass_count"] != len(check_rows) - failures
        or summary["fail_count"] != failures
        or summary["difference_count"] != failures
        or summary["required_tables"] != REQUIRED_TABLE_COUNT
        or summary["tables_fully_scanned"] > REQUIRED_TABLE_COUNT
    ):
        raise AuxiliaryBundleVerificationError("Verification failure summary is inconsistent")
    expected_status = PASS_STATUS if failures == 0 else FAIL_STATUS
    if report.get("status") != expected_status:
        raise AuxiliaryBundleVerificationError("Verification status is inconsistent")
    independence = report.get("independence")
    expected_independence = {
        "primary_builder_imported": False,
        "primary_builder_called": False,
        "primary_publisher_imported": False,
        "raw_source_files_opened": 0,
        "raw_source_rows_scanned": 0,
        "desktop_database_written": False,
        "full_parquet_tables_scanned": summary["tables_fully_scanned"],
        "full_parquet_rows_scanned": summary["parquet_rows_scanned"],
    }
    if independence != expected_independence or (
        failures == 0
        and summary["tables_fully_scanned"] != REQUIRED_TABLE_COUNT
    ):
        raise AuxiliaryBundleVerificationError(
            "Verification independence boundary is inconsistent"
        )
    expected_authorizations = {
        "auxiliary_panel_authorized": failures == 0,
        "phase_03_inputs_authorized": failures == 0,
        "features_authorized": False,
        "models_authorized": False,
        "backtests_authorized": False,
        "features_labels_models_portfolios_pnl_performance_or_backtests_authorized": False,
        "validation_or_final_test_access_authorized": False,
    }
    if report.get("authorizations") != expected_authorizations:
        raise AuxiliaryBundleVerificationError(
            "Verification authorization boundary is inconsistent"
        )
    if require_passing and failures:
        raise AuxiliaryBundleVerificationError("Only zero-difference evidence may publish")
    return report


def independently_verify_auxiliary_bundle(
    settings: AuxiliaryVerificationSettings,
) -> dict[str, Any]:
    """Verify a closed auxiliary primary bundle and return publication-ready bytes."""

    settings = settings.normalized()
    if settings.batch_size <= 0:
        raise AuxiliaryBundleVerificationError("batch_size must be positive")

    # Capture the fixed authority set before the first read.  The second
    # snapshot adds manifest-declared quality receipts before any of them are
    # consumed.  Revalidating both at the end closes hash-to-scan replacement
    # races without retaining descriptors for every large Parquet member.
    preliminary_authority_snapshot = _authority_snapshot(settings, {})
    pointer = _load_json(settings.primary_pointer_path)
    manifest = _load_json(settings.primary_manifest_path)
    authority_snapshot = _authority_snapshot(settings, manifest)
    _assert_authority_snapshot(preliminary_authority_snapshot)
    policy = _load_toml(settings.policy_path)
    source_config = _load_toml(settings.source_config_path)
    schema_document = _load_json(settings.schema_path)
    factor_lock = _load_json(settings.factor_unit_lock_path)
    freeze = _load_json(settings.stage_01_freeze_manifest_path)
    stage_01_hashes = _parse_shasum(settings.stage_01_raw_hashes_path)
    checks = _Checks()

    checks.add(
        "PRIMARY_POINTER_AND_MANIFEST",
        category="authentication",
        expected={
            "status": PRIMARY_STATUS,
            "manifest_status": PRIMARY_STATUS,
            "authorized": False,
            "build_id": manifest.get("build_id"),
            "contract_id": manifest.get("contract_id"),
            "manifest_sha256": pointer.get(
                "manifest_sha256", pointer.get("primary_manifest_sha256")
            ),
        },
        actual={
            "status": pointer.get("status"),
            "manifest_status": manifest.get("status"),
            "authorized": bool(
                pointer.get(
                    "auxiliary_panel_authorized",
                    pointer.get("authorizations", {}).get(
                        "auxiliary_panel_authorized", False
                    ),
                )
            ),
            "build_id": pointer.get("build_id"),
            "contract_id": pointer.get("contract_id"),
            "manifest_sha256": sha256_path(settings.primary_manifest_path),
        },
        evidence="closed pending primary pointer and exact manifest bytes",
    )
    signal_verified_pointer = _load_json(settings.signal_verified_pointer_path)
    derived_build_id = _derive_primary_build_id(manifest)
    expected_database_contract = signal_verified_pointer.get("contract_id")
    checks.add(
        "PRIMARY_CONTENT_ADDRESS_AND_CONTRACT",
        category="authentication",
        expected={
            "policy_contract_id": CONTRACT_ID,
            "manifest_contract_id": CONTRACT_ID,
            "pointer_contract_id": CONTRACT_ID,
            "database_contract_id": expected_database_contract,
            "manifest_build_id": derived_build_id,
            "pointer_build_id": derived_build_id,
        },
        actual={
            "policy_contract_id": policy.get("contract_id"),
            "manifest_contract_id": manifest.get("contract_id"),
            "pointer_contract_id": pointer.get("contract_id"),
            "database_contract_id": manifest.get("database_contract_id"),
            "manifest_build_id": manifest.get("build_id"),
            "pointer_build_id": pointer.get("build_id"),
        },
        evidence="frozen contract identity and independently reconstructed primary content address",
    )
    pending_authorizations = {
        "auxiliary_panel_authorized": False,
        "independent_verification_required": True,
        "features_labels_models_portfolios_pnl_performance_or_backtests_authorized": False,
        "validation_or_final_test_access_authorized": False,
    }
    try:
        artifact_path_matches = _safe_receipt_path(
            settings.project_root, pointer.get("artifact_path")
        ) == settings.primary_manifest_path.parent
    except AuxiliaryBundleVerificationError:
        artifact_path_matches = False
    checks.add(
        "PRIMARY_NAMESPACE_AND_AUTHORIZATION",
        category="authentication",
        expected={
            "manifest_authorizations": pending_authorizations,
            "pointer_auxiliary_panel_authorized": False,
            "manifest_and_pointer_interim_path_equal": True,
            "manifest_and_pointer_quality_path_equal": True,
            "artifact_path_matches_manifest_parent": True,
            "interim_path_matches_supplied_tables": True,
            "quality_path_matches_supplied_receipts": True,
        },
        actual={
            "manifest_authorizations": manifest.get("authorizations"),
            "pointer_auxiliary_panel_authorized": pointer.get(
                "auxiliary_panel_authorized"
            ),
            "manifest_and_pointer_interim_path_equal": manifest.get(
                "interim_path"
            )
            == pointer.get("interim_path"),
            "manifest_and_pointer_quality_path_equal": manifest.get(
                "quality_path"
            )
            == pointer.get("quality_path"),
            "artifact_path_matches_manifest_parent": artifact_path_matches,
            "interim_path_matches_supplied_tables": _path_has_relative_suffix(
                settings.tables_dir, pointer.get("interim_path")
            ),
            "quality_path_matches_supplied_receipts": _path_has_relative_suffix(
                settings.quality_dir, pointer.get("quality_path")
            ),
        },
        evidence="pending authorization boundary and primary artifact/database namespaces",
    )

    _authenticate_contracts(
        settings,
        manifest,
        policy,
        source_config,
        freeze,
        stage_01_hashes,
        factor_lock,
        checks,
    )
    signal_rows = _authenticate_signal_universe(settings, policy, checks)
    _authenticate_trading_calendar(settings, manifest, checks)
    signal_primary_manifest = _load_json(settings.signal_primary_manifest_path)
    signal_primary_output = _signal_output_receipt(signal_primary_manifest)
    claimed_signal_receipt = manifest.get("signal_universe", {})
    claimed_signal_path = (
        claimed_signal_receipt.get("path")
        if isinstance(claimed_signal_receipt, Mapping)
        else None
    )
    actual_signal_receipt = {
        "path": (
            claimed_signal_path
            if isinstance(claimed_signal_path, str)
            and Path(claimed_signal_path).name
            == Path(str(signal_primary_output.get("path"))).name
            else None
        ),
        "sha256": sha256_path(settings.signal_universe_path),
        "size_bytes": settings.signal_universe_path.stat().st_size,
        "rows": pq.ParquetFile(settings.signal_universe_path).metadata.num_rows,
        "primary_manifest_sha256": sha256_path(settings.signal_primary_manifest_path),
    }
    checks.add(
        "PRIMARY_MANIFEST_SIGNAL_UNIVERSE_RECEIPT",
        category="authentication",
        expected=claimed_signal_receipt,
        actual=actual_signal_receipt,
        evidence="auxiliary manifest signal-spine receipt versus verified upstream bytes",
    )

    contracts = schema_document.get("tables", [])
    contract_names = [
        str(value.get("table")) for value in contracts if isinstance(value, Mapping)
    ]
    contracts_by_name = {
        str(value.get("table")): value
        for value in contracts
        if isinstance(value, Mapping)
    }
    policy_tables = list(policy.get("materialization", {}).get("required_tables", []))
    source_tables = {
        str(name)
        for item in source_config.get("inputs", [])
        if isinstance(item, Mapping)
        for name in item.get("outputs", [])
    }
    linkage_table = source_config.get("required_linkage_output", {}).get("table")
    if isinstance(linkage_table, str):
        source_tables.add(linkage_table)
    checks.add(
        "EXACT_REQUIRED_TABLE_SET",
        category="contract",
        expected={
            "count": REQUIRED_TABLE_COUNT,
            "schema": sorted(contract_names),
            "policy": sorted(contract_names),
            "source_config": sorted(contract_names),
        },
        actual={
            "count": len(contract_names),
            "schema": sorted(set(contract_names)),
            "policy": sorted(policy_tables),
            "source_config": sorted(source_tables),
        },
        evidence="three frozen contracts independently reconciled",
    )

    outputs = _manifest_outputs(manifest)
    output_expected: dict[str, Any] = {}
    output_actual: dict[str, Any] = {}
    output_paths: dict[str, Path] = {}
    for table_name in contract_names:
        receipt = outputs.get(table_name)
        if receipt is None:
            output_expected[table_name] = None
            output_actual[table_name] = None
            continue
        output_expected[table_name] = {
            "table": receipt.get("table"),
            "path": receipt.get("path"),
            "rows": receipt.get("rows"),
            "row_groups": receipt.get("row_groups"),
            "size_bytes": receipt.get("size_bytes"),
            "sha256": receipt.get("sha256"),
            "primary_key": receipt.get("primary_key"),
            "sort_key": receipt.get("sort_key"),
        }
        try:
            path = _safe_receipt_path(
                settings.tables_dir,
                receipt.get("path", f"{table_name}.parquet"),
            )
            output_paths[table_name] = path
            if not path.is_file():
                output_actual[table_name] = None
                continue
            parquet = pq.ParquetFile(path)
            output_actual[table_name] = {
                "table": table_name,
                "path": path.relative_to(settings.tables_dir).as_posix(),
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.metadata.num_row_groups,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_path(path),
                "primary_key": list(contracts_by_name[table_name]["primary_key"]),
                "sort_key": list(contracts_by_name[table_name]["sort_key"]),
            }
        except Exception as exc:
            output_actual[table_name] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
    checks.add(
        "MANIFEST_AND_PARQUET_FILE_RECEIPTS",
        category="authentication",
        expected={
            "table_set": sorted(contract_names),
            "receipts": output_expected,
        },
        actual={
            "table_set": sorted(outputs),
            "receipts": output_actual,
        },
        evidence="all required Parquet files independently hashed, sized, and row-counted",
    )
    checks.add(
        "TABLE_DIRECTORY_CLOSED_SET",
        category="authentication",
        expected={
            "files": sorted(
                [
                    str(receipt.get("path", f"{name}.parquet"))
                    for name, receipt in outputs.items()
                ]
                + ["_SUCCESS.json"]
            ),
            "non_regular": [],
        },
        actual=_flat_directory_inventory(settings.tables_dir),
        evidence="interim directory contains only the eighteen manifest outputs and success receipt",
    )
    output_hashes_path = settings.primary_manifest_path.parent / "output_hashes.sha256"
    output_hashes_actual = (
        _parse_shasum(output_hashes_path) if output_hashes_path.is_file() else None
    )
    output_hashes_expected = {
        str(receipt.get("path", f"{name}.parquet")): receipt.get("sha256")
        for name, receipt in outputs.items()
    }
    checks.add(
        "OUTPUT_HASH_RECEIPT_RECONCILED",
        category="authentication",
        expected=dict(sorted(output_hashes_expected.items())),
        actual=(
            dict(sorted(output_hashes_actual.items()))
            if output_hashes_actual is not None
            else None
        ),
        evidence="artifact-level output_hashes.sha256 versus all manifest Parquet receipts",
    )
    success_path = settings.tables_dir / "_SUCCESS.json"
    success_actual = _load_json(success_path) if success_path.is_file() else None
    checks.add(
        "PRIMARY_SUCCESS_RECEIPT",
        category="authentication",
        expected={
            "schema_version": 1,
            "status": PRIMARY_STATUS,
            "build_id": manifest.get("build_id"),
            "manifest_sha256": sha256_path(settings.primary_manifest_path),
            "auxiliary_panel_authorized": False,
        },
        actual=success_actual,
        evidence="closed interim success receipt bound to this exact pending manifest",
    )

    last_signal = dt.date.fromisoformat(
        str(policy.get("scope", {}).get("last_predictor_signal_date", "2014-10-31"))
    )
    last_exit = dt.date.fromisoformat(
        str(policy.get("scope", {}).get("last_development_exit_date", "2014-12-01"))
    )
    scans: dict[str, TableScan] = {}
    for contract in contracts:
        table_name = str(contract["table"])
        path = output_paths.get(table_name)
        if path is None or not path.is_file():
            checks.add(
                f"SCHEMA_{table_name}",
                category="schema",
                expected=True,
                actual=False,
                evidence="required Parquet output is missing",
            )
            checks.add(
                f"FULL_SCAN_{table_name}",
                category="full_scan",
                expected={"rows": outputs.get(table_name, {}).get("rows"), "violations": {}},
                actual={"rows": None, "violations": {"missing_output": 1}},
                evidence="required Parquet output is missing",
            )
            continue
        try:
            parquet = pq.ParquetFile(path)
            expected_schema = expected_table_schema(schema_document, table_name)
            schema_matches = _schema_without_metadata(parquet.schema_arrow).equals(
                expected_schema, check_metadata=False
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            checks.add(
                f"SCHEMA_{table_name}",
                category="schema",
                expected=True,
                actual=failure,
                evidence="required Parquet schema could not be authenticated",
            )
            checks.add(
                f"FULL_SCAN_{table_name}",
                category="full_scan",
                expected={"rows": outputs.get(table_name, {}).get("rows"), "violations": {}},
                actual={"rows": None, "violations": {"unreadable_output": 1}},
                evidence="semantic scan could not safely open required Parquet",
            )
            continue
        checks.add(
            f"SCHEMA_{table_name}",
            category="schema",
            expected=True,
            actual=schema_matches,
            evidence="complete field order, logical types, and field nullability",
        )
        if not schema_matches:
            checks.add(
                f"FULL_SCAN_{table_name}",
                category="full_scan",
                expected={"rows": outputs[table_name].get("rows"), "violations": {}},
                actual={"rows": parquet.metadata.num_rows, "violations": {"schema_prevents_safe_scan": 1}},
                evidence="semantic scan withheld because exact schema differs",
            )
            continue
        try:
            scan = scan_table(
                path,
                contract,
                batch_size=settings.batch_size,
                last_signal=last_signal,
                last_exit=last_exit,
            )
        except Exception as exc:
            checks.add(
                f"FULL_SCAN_{table_name}",
                category="full_scan",
                expected={"rows": outputs[table_name].get("rows"), "violations": {}},
                actual={
                    "rows": None,
                    "violations": {
                        "scan_error": f"{type(exc).__name__}: {exc}"
                    },
                },
                evidence="complete semantic scan failed closed",
            )
            continue
        scans[table_name] = scan
        checks.add(
            f"FULL_SCAN_{table_name}",
            category="full_scan",
            expected={"rows": outputs[table_name].get("rows"), "violations": {}},
            actual={"rows": scan.rows, "violations": scan.nonzero_violations()},
            evidence="every Parquet row scanned for nulls, PK uniqueness, order, date, interval, and status rules",
        )

    pit_expected: Any
    pit_actual: Any
    try:
        pit_actual = _point_in_time_formula_violations(
            output_paths, settings.trading_calendar_path
        )
        pit_expected = {name: 0 for name in pit_actual}
    except Exception as exc:
        pit_expected = "all frozen per-family formulas independently reconstructed"
        pit_actual = f"{type(exc).__name__}: {exc}"
    checks.add(
        "POINT_IN_TIME_FORMULAS_RECONSTRUCTED",
        category="full_scan",
        expected=pit_expected,
        actual=pit_actual,
        evidence="disk-backed exact availability, observation, basis, and staleness reconstruction on verified trading dates",
    )

    linkage_payload_expected: Any
    linkage_payload_actual: Any
    try:
        linkage_payload_actual = reconcile_linkage_payloads(output_paths)
        linkage_payload_expected = {name: 0 for name in linkage_payload_actual}
    except Exception as exc:
        linkage_payload_expected = "all linkage payloads independently reconstructed"
        linkage_payload_actual = f"{type(exc).__name__}: {exc}"
    checks.add(
        "LINKAGE_PAYLOAD_RECONSTRUCTED",
        category="linkage",
        expected=linkage_payload_expected,
        actual=linkage_payload_actual,
        evidence="disk-backed CCM/SEC/OptionMetrics as-of payload and restatement-link reconciliation",
    )

    linkage_name = "auxiliary_signal_linkage"
    linkage_path = output_paths.get(linkage_name)
    if linkage_name in scans and linkage_path is not None and linkage_path.is_file():
        try:
            linkage = _compare_signal_spine(
                settings.signal_universe_path,
                linkage_path,
                batch_size=settings.batch_size,
            )
        except Exception as exc:
            linkage = {
                "signal_rows": signal_rows,
                "linkage_rows": None,
                "primary_key_differences": None,
                "signal_sort_or_duplicate_violations": None,
                "comparison_error": f"{type(exc).__name__}: {exc}",
            }
    else:
        linkage = {
            "signal_rows": signal_rows,
            "linkage_rows": None,
            "primary_key_differences": None,
            "signal_sort_or_duplicate_violations": None,
        }
    checks.add(
        "SIGNAL_UNIVERSE_ROW_AND_PRIMARY_KEY_PRESERVATION",
        category="linkage",
        expected={
            "signal_rows": signal_rows,
            "linkage_rows": signal_rows,
            "primary_key_differences": 0,
            "signal_sort_or_duplicate_violations": 0,
        },
        actual=linkage,
        evidence="streaming row-for-row comparison of both complete (signal_date, permno) spines",
    )

    factor_expected, factor_actual = _factor_lock_aggregates(factor_lock, scans)
    checks.add(
        "FACTOR_UNIT_LOCK_AGGREGATES",
        category="aggregate",
        expected=factor_expected,
        actual=factor_actual,
        evidence="full factor-table row/date/extrema scan against the prebuild empirical lock",
    )

    quality_checks = _Checks()
    try:
        if set(scans) != set(REQUIRED_TABLE_NAMES):
            raise AuxiliaryBundleVerificationError(
                "one or more required tables could not be scanned"
            )
        if linkage_path is None or not linkage_path.is_file():
            raise AuxiliaryBundleVerificationError(
                "linkage Parquet is unavailable"
            )
        aggregate_qa, detail_expectations = _primary_aggregate_qa(
            settings,
            schema_document,
            scans,
            linkage,
            factor_lock,
            output_paths,
        )
        _validate_quality_receipts(
            settings,
            manifest,
            source_config,
            policy,
            schema_document,
            stage_01_hashes,
            aggregate_qa,
            detail_expectations,
            scans,
            linkage,
            linkage_path,
            quality_checks,
        )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        for check_id in _QUALITY_RECONSTRUCTION_CHECK_IDS:
            checks.add(
                check_id,
                category="aggregate",
                expected="independent reconstruction completed",
                actual=failure,
                evidence="aggregate receipt reconstruction was unavailable after a required-table failure",
            )
    else:
        checks.rows.extend(quality_checks.rows)

    ordered_checks = sorted(checks.rows, key=lambda row: row["check_id"])
    fail_count = sum(row["status"] == "FAIL" for row in ordered_checks)
    status = PASS_STATUS if fail_count == 0 else FAIL_STATUS
    rows_scanned = sum(scan.rows for scan in scans.values())
    report_without_id: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "build_id": manifest.get("build_id"),
        "contract_id": manifest.get("contract_id"),
        "primary_manifest_sha256": sha256_path(settings.primary_manifest_path),
        "verifier_implementation_sha256": sha256_path(Path(__file__)),
        "checks": ordered_checks,
        "summary": {
            "check_count": len(ordered_checks),
            "pass_count": len(ordered_checks) - fail_count,
            "fail_count": fail_count,
            "difference_count": fail_count,
            "required_tables": REQUIRED_TABLE_COUNT,
            "tables_fully_scanned": len(scans),
            "parquet_rows_scanned": rows_scanned,
            "signal_universe_rows_compared": linkage.get("signal_rows"),
        },
        "independence": {
            "primary_builder_imported": False,
            "primary_builder_called": False,
            "primary_publisher_imported": False,
            "raw_source_files_opened": 0,
            "raw_source_rows_scanned": 0,
            "desktop_database_written": False,
            "full_parquet_tables_scanned": len(scans),
            "full_parquet_rows_scanned": rows_scanned,
        },
        "authorizations": {
            "auxiliary_panel_authorized": status == PASS_STATUS,
            "phase_03_inputs_authorized": status == PASS_STATUS,
            "features_authorized": False,
            "models_authorized": False,
            "backtests_authorized": False,
            "features_labels_models_portfolios_pnl_performance_or_backtests_authorized": False,
            "validation_or_final_test_access_authorized": False,
        },
    }
    provisional_differences = render_differences_csv(report_without_id)
    verification_id = derive_verification_id(
        report_without_id, provisional_differences
    )
    report = dict(report_without_id)
    report["verification_id"] = verification_id
    differences_payload = render_differences_csv(report)
    if differences_payload != provisional_differences:
        raise AuxiliaryBundleVerificationError(
            "Verification ID unexpectedly changed differences bytes"
        )
    independent_check_payload = canonical_json(report)
    validate_verification_payloads(
        independent_check_payload, differences_payload, require_passing=False
    )
    _assert_authority_snapshot(preliminary_authority_snapshot)
    _assert_authority_snapshot(authority_snapshot)
    result = AuxiliaryVerificationResult({
        "status": status,
        "verification_id": verification_id,
        "report": report,
        "independent_check_payload": independent_check_payload,
        "differences_payload": differences_payload,
        "independent_check_sha256": _sha256_bytes(independent_check_payload),
        "differences_sha256": _sha256_bytes(differences_payload),
    })
    _EXECUTION_BINDINGS[result] = _ExecutionBinding(
        settings=settings,
        independent_check_sha256=result["independent_check_sha256"],
        differences_sha256=result["differences_sha256"],
        authority_snapshot=authority_snapshot,
    )
    return result


def _verification_output_hashes(
    independent_check_payload: bytes, differences_payload: bytes
) -> bytes:
    return (
        f"{_sha256_bytes(differences_payload)}  differences.csv\n"
        f"{_sha256_bytes(independent_check_payload)}  independent_check.json\n"
    ).encode("utf-8")


def _safe_component(value: Any, label: str) -> str:
    rendered = str(value)
    if (
        not rendered
        or rendered in {".", ".."}
        or Path(rendered).name != rendered
        or "/" in rendered
        or "\\" in rendered
        or "\x00" in rendered
    ):
        raise AuxiliaryBundleVerificationError(
            f"Unsafe {label} component: {rendered!r}"
        )
    return rendered


def _project_path(project_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise AuxiliaryBundleVerificationError(f"Missing {label} path")
    raw = Path(value)
    path = (raw if raw.is_absolute() else project_root / raw).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise AuxiliaryBundleVerificationError(
            f"{label} path escapes project root"
        ) from exc
    return path


def _rooted_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise AuxiliaryBundleVerificationError(f"Missing {label} path")
    raw = Path(value)
    path = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AuxiliaryBundleVerificationError(
            f"{label} path escapes its configured root"
        ) from exc
    return path


def publish_independent_verification(
    result: Mapping[str, Any],
    *,
    settings: AuxiliaryVerificationSettings,
    artifact_root: Path | None = None,
    authority_gate: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Publish one verifier execution while retaining its primary authorities."""

    if not isinstance(result, AuxiliaryVerificationResult):
        raise AuxiliaryBundleVerificationError(
            "Verification result did not originate from this verifier execution"
        )
    binding = _EXECUTION_BINDINGS.get(result)
    if binding is None:
        raise AuxiliaryBundleVerificationError(
            "Verification result has no live verifier execution binding"
        )
    settings = settings.normalized()
    if settings != binding.settings:
        raise AuxiliaryBundleVerificationError(
            "Publication settings differ from the verifier execution binding"
        )
    project_root = settings.project_root
    artifact_root = (
        Path(artifact_root).resolve()
        if artifact_root is not None
        else project_root / "artifacts/phase_02_canonical_panel"
    )
    try:
        artifact_root.relative_to(project_root)
    except ValueError as exc:
        raise AuxiliaryBundleVerificationError(
            "Verification artifact root must be beneath project root"
        ) from exc

    independent_check_payload = result.get("independent_check_payload")
    differences_payload = result.get("differences_payload")
    if not isinstance(independent_check_payload, bytes) or not isinstance(
        differences_payload, bytes
    ):
        raise AuxiliaryBundleVerificationError(
            "Verification publication requires exact report and differences bytes"
        )
    if (
        _sha256_bytes(independent_check_payload)
        != binding.independent_check_sha256
        or _sha256_bytes(differences_payload) != binding.differences_sha256
    ):
        raise AuxiliaryBundleVerificationError(
            "Verification payloads differ from their verifier execution binding"
        )
    report = validate_verification_payloads(
        independent_check_payload,
        differences_payload,
        require_passing=False,
    )
    if (
        result.get("status") != report["status"]
        or result.get("verification_id") != report["verification_id"]
        or result.get("independent_check_sha256")
        != _sha256_bytes(independent_check_payload)
        or result.get("differences_sha256") != _sha256_bytes(differences_payload)
    ):
        raise AuxiliaryBundleVerificationError(
            "Verification result wrapper differs from its authenticated payloads"
        )

    _assert_authority_snapshot(binding.authority_snapshot)
    if report.get("verifier_implementation_sha256") != sha256_path(
        Path(__file__).resolve()
    ):
        raise AuxiliaryBundleVerificationError(
            "Verifier implementation changed after report construction"
        )
    current_manifest_sha256 = sha256_path(settings.primary_manifest_path)
    if report.get("primary_manifest_sha256") != current_manifest_sha256:
        raise AuxiliaryBundleVerificationError(
            "Primary manifest changed after independent verification"
        )
    current_pointer_payload = settings.primary_pointer_path.read_bytes()
    current_pointer = json.loads(current_pointer_payload)
    current_manifest = _load_json(settings.primary_manifest_path)
    expected_primary_identity = {
        "status": PRIMARY_STATUS,
        "build_id": report.get("build_id"),
        "contract_id": report.get("contract_id"),
        "manifest_sha256": current_manifest_sha256,
    }
    actual_pointer_identity = {
        "status": current_pointer.get("status"),
        "build_id": current_pointer.get("build_id"),
        "contract_id": current_pointer.get("contract_id"),
        "manifest_sha256": current_pointer.get(
            "manifest_sha256", current_pointer.get("primary_manifest_sha256")
        ),
    }
    actual_manifest_identity = {
        "status": current_manifest.get("status"),
        "build_id": current_manifest.get("build_id"),
        "contract_id": current_manifest.get("contract_id"),
        "manifest_sha256": current_manifest_sha256,
    }
    if (
        actual_pointer_identity != expected_primary_identity
        or actual_manifest_identity != expected_primary_identity
    ):
        raise AuxiliaryBundleVerificationError(
            "Current primary authority differs from the verified primary identity"
        )

    build_id = _safe_component(report.get("build_id"), "build id")
    verification_id = _safe_component(
        report.get("verification_id"), "verification id"
    )
    if re.fullmatch(r"verify-[0-9a-f]{16}", verification_id) is None:
        raise AuxiliaryBundleVerificationError("Verification id format differs")
    output_hashes_payload = _verification_output_hashes(
        independent_check_payload, differences_payload
    )
    payloads = {
        "differences.csv": differences_payload,
        "independent_check.json": independent_check_payload,
        "output_hashes.sha256": output_hashes_payload,
    }
    expected_hashes = {
        name: _sha256_bytes(payload) for name, payload in payloads.items()
    }
    destination = (
        artifact_root
        / VERIFICATION_SUBDIRECTORY
        / build_id
        / verification_id
    )
    with secure.AuthenticatedPointerGuard.open(
        trusted_root=project_root,
        path=settings.primary_pointer_path,
        expected_payload=current_pointer_payload,
        label="auxiliary pending-primary pointer",
    ) as primary_pointer_guard, secure.AuthenticatedPointerGuard.open(
        trusted_root=project_root,
        path=settings.primary_manifest_path,
        expected_sha256=current_manifest_sha256,
        label="auxiliary pending-primary manifest",
    ) as primary_manifest_guard:

        def stable_primary_gate() -> None:
            primary_pointer_guard.gate()
            primary_manifest_guard.gate()
            _assert_authority_snapshot(binding.authority_snapshot)
            if report.get("verifier_implementation_sha256") != sha256_path(
                Path(__file__).resolve()
            ):
                raise AuxiliaryBundleVerificationError(
                    "Verifier implementation changed before publication commit"
                )

        stable_primary_gate()
        with secure.SecureStagingDirectory.create(
            trusted_root=project_root,
            destination=destination,
            expected_files=VERIFICATION_FILES,
            label="auxiliary independent verification bundle",
        ) as stage:
            for name in sorted(payloads):
                with stage.open_exclusive_binary(name) as stream:
                    stream.write(payloads[name])
            snapshot = stage.snapshot()
            if dict(snapshot.hashes) != expected_hashes:
                raise AuxiliaryBundleVerificationError(
                    "Staged verification bytes differ before publication"
                )
            stable_primary_gate()
            bundle = stage.publish(
                expected_hashes=expected_hashes,
                pre_publish_gate=stable_primary_gate,
                post_publish_gate=stable_primary_gate,
            )

        publication: dict[str, Any] = {
            "verification_path": destination,
            "pointer_path": None,
            "reused_existing": bundle.reused_existing,
            "pointer_changed": False,
        }
        if report["status"] != PASS_STATUS:
            stable_primary_gate()
            return publication

        validate_verification_payloads(
            independent_check_payload, differences_payload, require_passing=True
        )
        authorizations = dict(report["authorizations"])
        pointer_payload = canonical_json(
            {
                "schema_version": 1,
                "status": PASS_STATUS,
                "build_id": build_id,
                "contract_id": report.get("contract_id"),
                "primary_manifest_sha256": report.get("primary_manifest_sha256"),
                "verification_id": verification_id,
                "verification_path": destination.relative_to(project_root).as_posix(),
                "independent_check_sha256": expected_hashes[
                    "independent_check.json"
                ],
                "differences_sha256": expected_hashes["differences.csv"],
                "output_hashes_sha256": expected_hashes["output_hashes.sha256"],
                "auxiliary_panel_authorized": True,
                "phase_03_inputs_authorized": True,
                "authorizations": authorizations,
            }
        )
        pointer_path = artifact_root / VERIFIED_POINTER
        with secure.AuthenticatedBundleGuard.open(
            trusted_root=project_root,
            path=destination,
            expected_files=VERIFICATION_FILES,
            expected_hashes=expected_hashes,
            label="published auxiliary independent verification bundle",
        ) as bundle_guard:

            def pointer_gate() -> None:
                stable_primary_gate()
                bundle_guard.gate()
                if authority_gate is not None:
                    authority_gate()

            pointer_changed = secure.replace_pointer_last(
                pointer_path,
                pointer_payload,
                trusted_root=project_root,
                pre_replace_gate=pointer_gate,
                post_replace_gate=pointer_gate,
                label="auxiliary verified pointer",
            )
        publication.update(
            {
                "pointer_path": pointer_path,
                "pointer_changed": pointer_changed,
            }
        )
        return publication


def load_auxiliary_verification_settings(
    project_root: Path,
    *,
    database_root: Path | None = None,
    primary_pointer_path: Path | None = None,
    policy_path: Path | None = None,
    batch_size: int = 65_536,
) -> AuxiliaryVerificationSettings:
    """Resolve the closed primary and frozen authorities used by the CLI."""

    project_root = Path(project_root).resolve()
    if database_root is None:
        paths_config = _load_toml(project_root / "configs/paths.toml")
        configured_database_root = paths_config.get("database", {}).get("root")
        selected_database_root = os.environ.get(
            "PORTFOLIO_DATABASE_ROOT", str(configured_database_root or "")
        )
        if not selected_database_root:
            raise AuxiliaryBundleVerificationError("Database root is absent")
        database_root = Path(selected_database_root)
    database_root = Path(database_root).expanduser().resolve()
    policy_path = (
        Path(policy_path).resolve()
        if policy_path is not None
        else project_root
        / "configs/phase_02_canonicalization/auxiliary_source_policy_v1.toml"
    )
    policy = _load_toml(policy_path)
    bindings = policy.get("bindings", {})
    if not isinstance(bindings, Mapping):
        raise AuxiliaryBundleVerificationError("Auxiliary policy bindings are absent")
    primary_pointer_path = (
        Path(primary_pointer_path).resolve()
        if primary_pointer_path is not None
        else project_root
        / "artifacts/phase_02_canonical_panel/"
        "LATEST_AUXILIARY_AVAILABILITY_LINKAGE_PRIMARY.json"
    )
    primary_pointer = _load_json(primary_pointer_path)
    primary_artifact = _project_path(
        project_root, primary_pointer.get("artifact_path"), "primary artifact"
    )
    tables_dir = _rooted_path(
        database_root, primary_pointer.get("interim_path"), "primary interim"
    )
    quality_dir = _rooted_path(
        database_root, primary_pointer.get("quality_path"), "primary quality"
    )

    freeze_path = _project_path(
        project_root,
        bindings.get("phase_01_freeze_manifest"),
        "Stage-1 freeze manifest",
    )
    freeze = _load_json(freeze_path)
    final_rechecks = (
        freeze.get("identity", {})
        .get("evidence_groups", {})
        .get("final_raw_recheck", [])
    )
    recheck_paths = [
        item.get("path")
        for item in final_rechecks
        if isinstance(item, Mapping)
        and str(item.get("path", "")).endswith(".sha256")
    ]
    if len(recheck_paths) != 1:
        raise AuxiliaryBundleVerificationError(
            "Stage-1 final raw-recheck receipt is not unique"
        )

    upstream = policy.get("upstream", {}).get("signal_universe", {})
    if not isinstance(upstream, Mapping):
        raise AuxiliaryBundleVerificationError(
            "Signal-universe upstream policy is absent"
        )
    signal_primary_pointer_path = _project_path(
        project_root, upstream.get("primary_pointer"), "signal primary pointer"
    )
    signal_primary_pointer = _load_json(signal_primary_pointer_path)
    signal_primary_artifact = _project_path(
        project_root,
        signal_primary_pointer.get("artifact_path"),
        "signal primary artifact",
    )
    signal_primary_manifest_path = signal_primary_artifact / "manifest.json"
    signal_manifest = _load_json(signal_primary_manifest_path)
    signal_output = _signal_output_receipt(signal_manifest)
    signal_interim = _rooted_path(
        database_root,
        signal_primary_pointer.get("interim_path"),
        "signal primary interim",
    )
    signal_universe_path = _safe_receipt_path(
        signal_interim, signal_output.get("path")
    )

    calendar_verified_pointer_path = (
        project_root
        / "artifacts/phase_02_canonical_panel/LATEST_CALENDAR_VERIFIED.json"
    ).resolve()
    calendar_verified_pointer = _load_json(calendar_verified_pointer_path)
    calendar_primary_artifact = _project_path(
        project_root,
        calendar_verified_pointer.get("primary_artifact_path"),
        "calendar primary artifact",
    )
    calendar_primary_manifest_path = calendar_primary_artifact / "manifest.json"
    calendar_manifest = _load_json(calendar_primary_manifest_path)
    calendar_outputs = calendar_manifest.get("outputs", [])
    calendar_receipts = [
        item
        for item in calendar_outputs
        if isinstance(item, Mapping)
        and Path(str(item.get("path", ""))).name == "trading_calendar.parquet"
    ]
    if len(calendar_receipts) != 1:
        raise AuxiliaryBundleVerificationError(
            "Verified trading-calendar Parquet receipt is not unique"
        )
    calendar_contract_id = _safe_component(
        calendar_verified_pointer.get("contract_id"), "calendar contract id"
    )
    calendar_build_id = _safe_component(
        calendar_verified_pointer.get("build_id"), "calendar build id"
    )
    trading_calendar_path = (
        database_root
        / "interim/phase_02_canonical_panel"
        / calendar_contract_id
        / "calendar"
        / calendar_build_id
        / "trading_calendar.parquet"
    ).resolve()

    return AuxiliaryVerificationSettings(
        project_root=project_root,
        primary_pointer_path=primary_pointer_path,
        primary_manifest_path=primary_artifact / "manifest.json",
        tables_dir=tables_dir,
        quality_dir=quality_dir,
        policy_path=policy_path,
        schema_path=_project_path(
            project_root, bindings.get("schema"), "auxiliary schema"
        ),
        source_config_path=_project_path(
            project_root, bindings.get("source_config"), "auxiliary source config"
        ),
        phase_02_config_path=_project_path(
            project_root, bindings.get("phase_02_config"), "Phase-2 config"
        ),
        factor_unit_lock_path=_project_path(
            project_root,
            policy.get("factor_unit_lock", {}).get("path"),
            "factor-unit lock",
        ),
        stage_01_freeze_manifest_path=freeze_path,
        stage_01_raw_hashes_path=_project_path(
            project_root, recheck_paths[0], "Stage-1 raw-recheck"
        ),
        signal_verified_pointer_path=_project_path(
            project_root,
            upstream.get("verified_pointer"),
            "signal verified pointer",
        ),
        signal_primary_manifest_path=signal_primary_manifest_path,
        signal_universe_path=signal_universe_path,
        calendar_verified_pointer_path=calendar_verified_pointer_path,
        calendar_primary_manifest_path=calendar_primary_manifest_path,
        trading_calendar_path=trading_calendar_path,
        batch_size=batch_size,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--primary-pointer", type=Path)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    settings = load_auxiliary_verification_settings(
        args.project_root,
        database_root=args.database_root,
        primary_pointer_path=args.primary_pointer,
        policy_path=args.policy,
        batch_size=args.batch_size,
    )
    result = independently_verify_auxiliary_bundle(settings)
    publication: dict[str, Any] | None = None
    if not args.check_only:
        publication = publish_independent_verification(
            result,
            settings=settings,
            artifact_root=args.artifact_root,
        )
    print(
        json.dumps(
            {
                "build_id": result["report"]["build_id"],
                "verification_id": result["verification_id"],
                "status": result["status"],
                "checks": result["report"]["summary"]["check_count"],
                "failures": result["report"]["summary"]["fail_count"],
                "raw_source_rows_scanned": 0,
                "verification_path": (
                    str(publication["verification_path"])
                    if publication is not None
                    else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == PASS_STATUS else 1


__all__ = [
    "AuxiliaryBundleVerificationError",
    "AuxiliaryVerificationSettings",
    "FAIL_STATUS",
    "PASS_STATUS",
    "PRIMARY_STATUS",
    "VERIFICATION_FILES",
    "VERIFICATION_SUBDIRECTORY",
    "VERIFIED_POINTER",
    "canonical_json",
    "derive_verification_id",
    "expected_table_schema",
    "independently_verify_auxiliary_bundle",
    "load_auxiliary_verification_settings",
    "main",
    "publish_independent_verification",
    "render_differences_csv",
    "scan_table",
    "sha256_path",
    "validate_verification_payloads",
]


if __name__ == "__main__":
    raise SystemExit(main())
