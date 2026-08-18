"""Publication-safe cache materialization primitives for Stage 6 feature families.

This module deliberately stops at immutable cache batches.  It does not scan
upstream payload columns, assemble the complete feature store, publish a
primary build, construct labels, or authorize any downstream model work.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    AuthenticatedBundleGuard,
    AuthenticatedPointerGuard,
    FileRecord,
    SecurePublicationError,
    SecureStagingDirectory,
)
from portfolio_bot.features.feature_lineage import (
    FeatureFamilyResult,
    FeatureLineage,
    VALUE_STATUS_SET,
)
from portfolio_bot.features.feature_store_contract import (
    DEFAULT_POLICY,
    LINEAGE_SCHEMA,
    PROJECT_ROOT,
    FeatureStoreContractError,
    expanded_features,
    sha256_file,
    validate_feature_store_contract,
)
from portfolio_bot.features.feature_store_inputs import (
    FeatureStoreInputs,
    InputTable,
)
from portfolio_bot.features.phase_03_contract import canonical_json_bytes


_SAFE_COMPONENT = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CACHE_STATUS = "CACHE_FAMILY_BATCH_COMPLETE"
_SUCCESS_NAME = "_SUCCESS.json"

_CODE_PATHS = (
    "src/portfolio_bot/data/secure_publication.py",
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
    "src/portfolio_bot/features/fundamental_features.py",
    "src/portfolio_bot/features/materialize_fundamentals.py",
    "src/portfolio_bot/features/ownership_event_features.py",
    "src/portfolio_bot/features/materialize_ownership_events.py",
    "src/portfolio_bot/features/optionmetrics_features.py",
    "src/portfolio_bot/features/materialize_optionmetrics.py",
    "src/portfolio_bot/features/aggregate_state_features.py",
    "src/portfolio_bot/features/materialize_aggregate_state.py",
    "src/portfolio_bot/features/feature_store_registries.py",
    "src/portfolio_bot/features/feature_store_quality.py",
    "src/portfolio_bot/features/feature_store_materialization.py",
    "src/portfolio_bot/features/feature_store_family_orchestration.py",
    "src/portfolio_bot/features/feature_store_publication.py",
    "scripts/build_stage_06_feature_store.py",
)

_SUCCESS_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "content_fingerprint",
        "location_kind",
        "status",
        "family",
        "batch_id",
        "configuration_hashes",
        "code_hashes",
        "input_receipts_sha256",
        "row_count",
        "lineage_row_count",
        "table_receipts",
    }
)
_TABLE_RECEIPT_KEYS = frozenset(
    {"path", "sha256", "size_bytes", "rows", "row_groups", "schema_sha256"}
)


class FeatureStoreMaterializationError(RuntimeError):
    """Raised when a cache batch cannot be authenticated or materialized."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeatureStoreMaterializationError(message)


@dataclass(frozen=True)
class AuthenticatedInputReceipt:
    gate: str
    name: str
    sha256: str
    size_bytes: int
    rows: int
    row_groups: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "row_groups": self.row_groups,
        }


@dataclass(frozen=True)
class FeatureStoreBuildIdentity:
    contract_id: str
    policy_id: str
    build_id: str
    content_fingerprint: str
    configuration_hashes: tuple[tuple[str, str], ...]
    code_hashes: tuple[tuple[str, str], ...]
    input_receipts: tuple[AuthenticatedInputReceipt, ...]

    @property
    def configuration_hash_map(self) -> dict[str, str]:
        return dict(self.configuration_hashes)

    @property
    def code_hash_map(self) -> dict[str, str]:
        return dict(self.code_hashes)

    @property
    def input_receipts_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes([item.as_dict() for item in self.input_receipts])
        ).hexdigest()


@dataclass(frozen=True)
class FeatureMaterializationRow:
    signal_date: dt.date
    permno: int
    permco: int
    long_eligible: bool
    short_eligible: bool
    primary_exchange: str | None
    share_type: str | None
    result: FeatureFamilyResult


@dataclass(frozen=True)
class TableReceipt:
    path: str
    sha256: str
    size_bytes: int
    rows: int
    row_groups: int
    schema_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "row_groups": self.row_groups,
            "schema_sha256": self.schema_sha256,
        }


@dataclass(frozen=True)
class FamilyBatchReceipt:
    path: Path
    build_id: str
    family: str
    batch_id: str
    row_count: int
    lineage_row_count: int
    table_receipts: tuple[TableReceipt, ...]
    success_sha256: str
    success_size_bytes: int
    reused_existing: bool


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    _require(isinstance(value, dict), f"Expected TOML object: {path}")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _frozen_policy(
    policy_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    policy_path = policy_path.resolve()
    contract = validate_feature_store_contract(policy_path)
    policy = _toml(policy_path)
    semantic = _toml(_project_path(policy["semantic_contract"]["policy_path"]))
    return policy, semantic, contract


def _expected_input_gates(semantic: Mapping[str, Any]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for gate, specification in semantic["upstream"].items():
        for name in specification["required_tables"]:
            _require(name not in expected, f"Duplicate frozen input name: {name}")
            expected[str(name)] = str(gate)
    _require(len(expected) == 26, "Frozen Stage 6 input allowlist is not 26 tables")
    return expected


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _regular_descendant(database_root: Path, candidate: Path, label: str) -> Path:
    root = Path(os.path.abspath(database_root.expanduser()))
    target = Path(os.path.abspath(candidate.expanduser()))
    _require(root.is_absolute(), "Feature-store database root must be absolute")
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise FeatureStoreMaterializationError(
            f"Cannot inspect feature-store database root: {root}"
        ) from exc
    _require(
        stat.S_ISDIR(root_stat.st_mode) and not stat.S_ISLNK(root_stat.st_mode),
        "Feature-store database root must be a real directory",
    )
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise FeatureStoreMaterializationError(
            f"{label} escaped feature-store database root"
        ) from exc
    cursor = root
    for part in relative.parts:
        cursor /= part
        try:
            observed = os.lstat(cursor)
        except OSError as exc:
            raise FeatureStoreMaterializationError(
                f"Cannot inspect {label}: {cursor}"
            ) from exc
        _require(not stat.S_ISLNK(observed.st_mode), f"Symlink prohibited for {label}")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_target = target.resolve(strict=True)
        resolved_target.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise FeatureStoreMaterializationError(
            f"{label} escaped its authenticated root"
        ) from exc
    _require(target.is_file(), f"{label} is not a regular file")
    return target


def _authenticate_input_table(
    database_root: Path,
    key: str,
    expected_gate: str,
    table: InputTable,
) -> AuthenticatedInputReceipt:
    _require(key == table.name, f"Input catalog key/name mismatch: {key}")
    _require(table.gate == expected_gate, f"Input gate mismatch: {key}")
    _require(_SHA256.fullmatch(table.sha256) is not None, f"Invalid input hash: {key}")
    path = _regular_descendant(database_root, table.path, f"input table {key}")
    before = os.stat(path, follow_symlinks=False)
    _require(
        stat.S_ISREG(before.st_mode) and before.st_nlink >= 1,
        f"Input is not a regular file: {key}",
    )
    _require(before.st_size == table.size_bytes, f"Input size differs: {key}")
    digest = sha256_file(path)
    _require(digest == table.sha256, f"Input hash differs: {key}")
    try:
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        row_groups = parquet.metadata.num_row_groups
    except Exception as exc:
        raise FeatureStoreMaterializationError(
            f"Input is not authenticated Parquet: {key}"
        ) from exc
    after = os.stat(path, follow_symlinks=False)
    _require(_same_file_identity(before, after), f"Input changed while authenticated: {key}")
    _require(rows == table.rows, f"Input row count differs: {key}")
    _require(row_groups == table.row_groups, f"Input row-group count differs: {key}")
    return AuthenticatedInputReceipt(
        gate=table.gate,
        name=table.name,
        sha256=table.sha256,
        size_bytes=table.size_bytes,
        rows=table.rows,
        row_groups=table.row_groups,
    )


def authenticate_feature_store_inputs(
    inputs: FeatureStoreInputs,
    policy_path: Path = DEFAULT_POLICY,
) -> tuple[AuthenticatedInputReceipt, ...]:
    """Rehash and metadata-authenticate the exact frozen 26-table allowlist."""

    _policy, semantic, _contract = _frozen_policy(policy_path)
    expected = _expected_input_gates(semantic)
    _require(set(inputs.tables) == set(expected), "FeatureStoreInputs allowlist differs")
    receipts = [
        _authenticate_input_table(
            inputs.database_root,
            name,
            expected[name],
            inputs.tables[name],
        )
        for name in sorted(expected)
    ]
    paths = [
        Path(os.path.abspath(inputs.tables[item.name].path)).as_posix()
        for item in receipts
    ]
    _require(len(paths) == len(set(paths)), "FeatureStoreInputs reuse one physical path")
    return tuple(receipts)


def _configuration_hashes(
    policy_path: Path, policy: Mapping[str, Any]
) -> tuple[tuple[str, str], ...]:
    paths = (
        policy_path.resolve(),
        _project_path(policy["semantic_contract"]["policy_path"]),
        _project_path(policy["semantic_contract"]["registry_path"]),
        _project_path(policy["semantic_contract"]["verified_pointer"]),
        _project_path(policy["physical_rules"]["path"]),
        _project_path(policy["ff12_mapping"]["path"]),
        _project_path(policy["database"]["paths_config"]),
        _project_path(policy["database"]["relocation_receipt"]),
    )
    result: list[tuple[str, str]] = []
    for path in paths:
        _require(path.is_file() and not path.is_symlink(), f"Unsafe frozen binding: {path}")
        try:
            label = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            label = path.as_posix()
        result.append((label, sha256_file(path)))
    _require(len(result) == len({label for label, _ in result}), "Duplicate config binding")
    return tuple(sorted(result))


def _code_hashes() -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for label in _CODE_PATHS:
        path = (PROJECT_ROOT / label).resolve()
        _require(path.is_file() and not path.is_symlink(), f"Unsafe Stage 6 code path: {label}")
        result.append((label, sha256_file(path)))
    return tuple(result)


def compute_feature_store_build_identity(
    inputs: FeatureStoreInputs,
    policy_path: Path = DEFAULT_POLICY,
) -> FeatureStoreBuildIdentity:
    """Compute the frozen content fingerprint after authenticating all inputs."""

    policy, _semantic, _contract = _frozen_policy(policy_path)
    receipts = authenticate_feature_store_inputs(inputs, policy_path)
    configuration = _configuration_hashes(policy_path, policy)
    code = _code_hashes()
    content = {
        "schema_version": 1,
        "contract_id": policy["contract_id"],
        "policy_id": policy["policy_id"],
        "configuration_hashes": dict(configuration),
        "code_hashes": dict(code),
        "input_receipts": [item.as_dict() for item in receipts],
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    return FeatureStoreBuildIdentity(
        contract_id=str(policy["contract_id"]),
        policy_id=str(policy["policy_id"]),
        build_id=f"phase03-features-{fingerprint[:24]}",
        content_fingerprint=fingerprint,
        configuration_hashes=configuration,
        code_hashes=code,
        input_receipts=receipts,
    )


def _validated_precomputed_identity(
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    policy_path: Path,
) -> FeatureStoreBuildIdentity:
    """Validate an already authenticated build identity without rehashing inputs.

    Stage 6 orchestrators authenticate the full 2.69 GB input allowlist once,
    reuse that immutable identity for bounded cache batches, and perform one
    full reauthentication before primary publication.  This helper validates
    the in-memory receipt and current catalog metadata; it is not a substitute
    for the final full reauthentication.
    """

    _require(
        isinstance(identity, FeatureStoreBuildIdentity),
        "Precomputed Stage 6 identity has the wrong type",
    )
    policy, semantic, _contract = _frozen_policy(policy_path)
    _require(identity.contract_id == policy["contract_id"], "Build contract ID differs")
    _require(identity.policy_id == policy["policy_id"], "Build policy ID differs")
    expected_gates = _expected_input_gates(semantic)
    _require(
        len(identity.input_receipts) == 26
        and {item.name for item in identity.input_receipts} == set(expected_gates),
        "Precomputed identity input allowlist differs",
    )
    receipt_by_name = {item.name: item for item in identity.input_receipts}
    for name, table in inputs.tables.items():
        receipt = receipt_by_name.get(name)
        _require(receipt is not None, f"Precomputed identity omits input: {name}")
        _require(receipt.gate == expected_gates[name] == table.gate, f"Input gate differs: {name}")
        _require(receipt.sha256 == table.sha256, f"Input hash receipt differs: {name}")
        _require(receipt.size_bytes == table.size_bytes, f"Input size receipt differs: {name}")
        _require(receipt.rows == table.rows, f"Input row receipt differs: {name}")
        _require(receipt.row_groups == table.row_groups, f"Input row-group receipt differs: {name}")
    content = {
        "schema_version": 1,
        "contract_id": identity.contract_id,
        "policy_id": identity.policy_id,
        "configuration_hashes": identity.configuration_hash_map,
        "code_hashes": identity.code_hash_map,
        "input_receipts": [item.as_dict() for item in identity.input_receipts],
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    _require(
        identity.content_fingerprint == fingerprint,
        "Precomputed identity fingerprint differs",
    )
    _require(
        identity.build_id == f"phase03-features-{fingerprint[:24]}",
        "Precomputed build ID differs",
    )
    return identity


def validate_precomputed_feature_store_build_identity(
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    policy_path: Path = DEFAULT_POLICY,
) -> FeatureStoreBuildIdentity:
    """Validate a preauthenticated identity without rehashing payload bytes."""

    return _validated_precomputed_identity(
        inputs,
        identity,
        policy_path,
    )


def reauthenticate_feature_store_build_identity(
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    policy_path: Path = DEFAULT_POLICY,
) -> FeatureStoreBuildIdentity:
    """Fully rehash inputs/configuration/code and require the same build ID."""

    observed = compute_feature_store_build_identity(inputs, policy_path)
    _require(observed == identity, "Stage 6 build identity changed")
    return observed


def _registry_features(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    return expanded_features(_json(_project_path(policy["semantic_contract"]["registry_path"])))


def family_feature_names(
    family: str, policy_path: Path = DEFAULT_POLICY
) -> tuple[str, ...]:
    policy, _semantic, _contract = _frozen_policy(policy_path)
    names = tuple(
        str(feature["name"])
        for feature in _registry_features(policy)
        if feature["family"] == family
    )
    _require(names, f"Unknown frozen feature family: {family}")
    return names


def family_value_schema(
    family: str, policy_path: Path = DEFAULT_POLICY
) -> pa.Schema:
    """Return audit columns plus one family's features in registry order."""

    policy, _semantic, contract = _frozen_policy(policy_path)
    family_names = {
        str(feature["name"])
        for feature in _registry_features(policy)
        if feature["family"] == family
    }
    _require(family_names, f"Unknown frozen feature family: {family}")
    raw_schema: pa.Schema = contract["raw_schema"]
    raw_fields = list(raw_schema)
    fields = raw_fields[:7]
    for field in raw_fields[7:]:
        if field.name not in family_names:
            continue
        if pa.types.is_list(field.type):
            # Parquet canonicalizes Arrow list child names to ``element``.
            # Use that physical spelling before writing so schema receipts
            # remain identical after a Parquet round trip.
            field = pa.field(
                field.name,
                pa.list_(
                    pa.field(
                        "element",
                        field.type.value_type,
                        nullable=True,
                    )
                ),
                nullable=field.nullable,
                metadata=field.metadata,
            )
        fields.append(field)
    schema = pa.schema(fields)
    _require(
        schema.names[7:] == list(family_feature_names(family, policy_path)),
        f"Family schema order drifted: {family}",
    )
    return schema


def lineage_schema() -> pa.Schema:
    """Return the deterministic long-lineage schema."""

    return LINEAGE_SCHEMA


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _safe_component(value: str, label: str) -> str:
    _require(_SAFE_COMPONENT.fullmatch(value) is not None, f"Unsafe {label}: {value!r}")
    return value


def _cache_root(inputs: FeatureStoreInputs, policy: Mapping[str, Any]) -> Path:
    relative = Path(str(policy["database"]["cache_root"]))
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Frozen cache namespace is unsafe",
    )
    return Path(os.path.abspath(inputs.database_root.expanduser())) / relative


def _batch_path(
    inputs: FeatureStoreInputs,
    policy: Mapping[str, Any],
    identity: FeatureStoreBuildIdentity,
    family: str,
    batch_id: str,
) -> Path:
    return (
        _cache_root(inputs, policy)
        / "family_materialization"
        / identity.build_id
        / _safe_component(family, "family")
        / _safe_component(batch_id, "batch ID")
    )


def _date(value: Any, label: str) -> dt.date:
    _require(
        isinstance(value, dt.date) and not isinstance(value, dt.datetime),
        f"{label} must be a date",
    )
    return value


def _validate_feature_value(name: str, value: Any, field: pa.Field) -> None:
    if value is None:
        _require(field.nullable, f"Nonnullable feature is null: {name}")
        return
    if pa.types.is_floating(field.type):
        _require(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value)),
            f"Invalid numeric feature value: {name}",
        )
    elif pa.types.is_int8(field.type):
        _require(
            value in (0, 1, False, True),
            f"Binary feature escaped 0/1: {name}",
        )
    elif pa.types.is_string(field.type):
        _require(isinstance(value, str) and bool(value), f"Invalid categorical value: {name}")
    elif pa.types.is_list(field.type):
        _require(
            isinstance(value, (list, tuple))
            and bool(value)
            and all(isinstance(item, str) and item for item in value),
            f"Invalid multihot value: {name}",
        )
    else:
        raise FeatureStoreMaterializationError(f"Unsupported family field type: {field}")


def _materialize_tables(
    rows: Iterable[FeatureMaterializationRow],
    family: str,
    policy_path: Path,
) -> tuple[pa.Table, pa.Table]:
    value_schema = family_value_schema(family, policy_path)
    names = tuple(value_schema.names[7:])
    items = list(rows)
    _require(items, "Feature-family cache batch cannot be empty")
    items.sort(key=lambda item: (item.signal_date, item.permno))
    keys = [(item.signal_date, item.permno) for item in items]
    _require(len(keys) == len(set(keys)), "Duplicate signal_date/permno family row")

    policy, _semantic, _contract = _frozen_policy(policy_path)
    first = dt.date.fromisoformat(str(policy["scope"]["first_signal_date"]))
    last = dt.date.fromisoformat(str(policy["scope"]["last_signal_date"]))
    value_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for item in items:
        signal = _date(item.signal_date, "signal_date")
        _require(first <= signal <= last, f"Signal date escaped development cutoff: {signal}")
        _require(
            isinstance(item.permno, int)
            and not isinstance(item.permno, bool)
            and isinstance(item.permco, int)
            and not isinstance(item.permco, bool),
            "permno/permco must be integers",
        )
        _require(
            isinstance(item.long_eligible, bool)
            and isinstance(item.short_eligible, bool),
            "Eligibility flags must be Boolean",
        )
        _require(
            item.primary_exchange is None or isinstance(item.primary_exchange, str),
            "primary_exchange must be string or null",
        )
        _require(
            item.share_type is None or isinstance(item.share_type, str),
            "share_type must be string or null",
        )
        _require(isinstance(item.result, FeatureFamilyResult), "Invalid feature-family result")
        _require(tuple(item.result.values) == names, f"Family value keys differ: {family}")
        _require(tuple(item.result.lineage) == names, f"Family lineage keys differ: {family}")

        value_row: dict[str, Any] = {
            "signal_date": signal,
            "permno": item.permno,
            "permco": item.permco,
            "long_eligible": item.long_eligible,
            "short_eligible": item.short_eligible,
            "primary_exchange": item.primary_exchange,
            "share_type": item.share_type,
        }
        for name in names:
            value = item.result.values[name]
            _validate_feature_value(name, value, value_schema.field(name))
            lineage = item.result.lineage[name]
            _require(isinstance(lineage, FeatureLineage), f"Invalid lineage: {name}")
            _require(lineage.value_status in VALUE_STATUS_SET, f"Unknown lineage status: {name}")
            if value is None:
                _require(lineage.value_status != "PRESENT", f"Null feature marked PRESENT: {name}")
            if lineage.value_status == "PRESENT":
                observation = _date(lineage.max_observation_date, f"{name} observation")
                availability = _date(
                    lineage.max_public_availability_date, f"{name} availability"
                )
                _require(observation <= signal, f"Observation after signal: {name}")
                _require(availability <= signal, f"Availability after signal: {name}")
                if lineage.stale_after_date is not None:
                    stale = _date(lineage.stale_after_date, f"{name} stale_after")
                    _require(signal <= stale, f"Stale feature used: {name}")
            value_row[name] = list(value) if isinstance(value, tuple) else value
            lineage_rows.append(
                {
                    "signal_date": signal,
                    "permno": item.permno,
                    "feature_name": name,
                    "max_observation_date": lineage.max_observation_date,
                    "max_public_availability_date": lineage.max_public_availability_date,
                    "stale_after_date": lineage.stale_after_date,
                    "value_status": lineage.value_status,
                }
            )
        value_rows.append(value_row)
    try:
        values = pa.Table.from_pylist(value_rows, schema=value_schema)
        lineage = pa.Table.from_pylist(lineage_rows, schema=LINEAGE_SCHEMA)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise FeatureStoreMaterializationError("Feature batch violates Arrow schema") from exc
    return values, lineage


class _HashingSink:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self._size = 0
        self.closed = False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._size

    def write(self, payload: bytes | bytearray | memoryview) -> int:
        _require(not self.closed, "Parquet sink is closed")
        view = memoryview(payload)
        written = self._stream.write(view)
        _require(written == len(view), "Short Parquet write")
        self._digest.update(view)
        self._size += written
        return written

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        if not self.closed:
            self.flush()
            self.closed = True

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    @property
    def size_bytes(self) -> int:
        return self._size


def _write_parquet(
    stage: SecureStagingDirectory,
    name: str,
    table: pa.Table,
    *,
    row_group_size: int,
    compression: str,
    dictionary: bool,
) -> TableReceipt:
    with stage.open_exclusive_binary(name) as stream:
        sink = _HashingSink(stream)
        pq.write_table(
            table,
            sink,
            row_group_size=row_group_size,
            compression=compression,
            use_dictionary=dictionary,
            version="2.6",
            write_statistics=True,
        )
        sink.flush()
    return TableReceipt(
        path=name,
        sha256=sink.sha256,
        size_bytes=sink.size_bytes,
        rows=table.num_rows,
        row_groups=max(1, math.ceil(table.num_rows / row_group_size)),
        schema_sha256=_schema_sha256(table.schema),
    )


def _parquet_validator(
    schemas: Mapping[str, pa.Schema],
    receipts: Mapping[str, TableReceipt],
):
    def validate(name: str, descriptor: int, record: FileRecord) -> None:
        if name == _SUCCESS_NAME:
            return
        receipt = receipts[name]
        _require(record.sha256 == receipt.sha256, f"Parquet hash differs: {name}")
        _require(record.size_bytes == receipt.size_bytes, f"Parquet size differs: {name}")
        duplicate = os.dup(descriptor)
        try:
            with os.fdopen(duplicate, "rb") as handle:
                handle.seek(0)
                parquet = pq.ParquetFile(handle)
                observed_schema = parquet.schema_arrow
                observed_rows = parquet.metadata.num_rows
                observed_groups = parquet.metadata.num_row_groups
        except Exception as exc:
            raise FeatureStoreMaterializationError(
                f"Cannot validate cache Parquet: {name}"
            ) from exc
        _require(observed_schema.equals(schemas[name], check_metadata=True), f"Schema differs: {name}")
        _require(_schema_sha256(observed_schema) == receipt.schema_sha256, f"Schema hash differs: {name}")
        _require(observed_rows == receipt.rows, f"Row count differs: {name}")
        _require(observed_groups == receipt.row_groups, f"Row-group count differs: {name}")

    return validate


def _success_payload(
    identity: FeatureStoreBuildIdentity,
    policy: Mapping[str, Any],
    family: str,
    batch_id: str,
    receipts: Sequence[TableReceipt],
) -> dict[str, Any]:
    row_count = next(item.rows for item in receipts if item.path.startswith("feature_values_"))
    lineage_count = next(item.rows for item in receipts if item.path.startswith("feature_lineage_"))
    return {
        "schema_version": 1,
        "contract_id": identity.contract_id,
        "policy_id": identity.policy_id,
        "build_id": identity.build_id,
        "content_fingerprint": identity.content_fingerprint,
        "location_kind": "cache_only_family_batch",
        "status": _CACHE_STATUS,
        "family": family,
        "batch_id": batch_id,
        "configuration_hashes": identity.configuration_hash_map,
        "code_hashes": identity.code_hash_map,
        "input_receipts_sha256": identity.input_receipts_sha256,
        "row_count": row_count,
        "lineage_row_count": lineage_count,
        "table_receipts": {
            item.path: item.as_dict() for item in sorted(receipts, key=lambda value: value.path)
        },
    }


def _table_receipt(value: Any, expected_name: str) -> TableReceipt:
    _require(isinstance(value, dict), f"Invalid table receipt: {expected_name}")
    _require(set(value) == _TABLE_RECEIPT_KEYS, f"Table receipt keys differ: {expected_name}")
    _require(value["path"] == expected_name, f"Table receipt path differs: {expected_name}")
    _require(_SHA256.fullmatch(str(value["sha256"])) is not None, f"Bad table hash: {expected_name}")
    _require(_SHA256.fullmatch(str(value["schema_sha256"])) is not None, f"Bad schema hash: {expected_name}")
    for key in ("size_bytes", "rows", "row_groups"):
        _require(
            isinstance(value[key], int)
            and not isinstance(value[key], bool)
            and value[key] >= 0,
            f"Invalid {key}: {expected_name}",
        )
    _require(value["row_groups"] >= 1, f"Invalid row groups: {expected_name}")
    return TableReceipt(**value)


def _parse_success(
    payload: bytes,
    identity: FeatureStoreBuildIdentity,
    family: str,
    batch_id: str,
    value_name: str,
    lineage_name: str,
    feature_count: int,
) -> tuple[dict[str, Any], tuple[TableReceipt, ...]]:
    try:
        success = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureStoreMaterializationError("Invalid family cache receipt JSON") from exc
    _require(isinstance(success, dict), "Family cache receipt is not an object")
    _require(set(success) == _SUCCESS_KEYS, "Family cache receipt keys differ")
    expected_scalars = {
        "schema_version": 1,
        "contract_id": identity.contract_id,
        "policy_id": identity.policy_id,
        "build_id": identity.build_id,
        "content_fingerprint": identity.content_fingerprint,
        "location_kind": "cache_only_family_batch",
        "status": _CACHE_STATUS,
        "family": family,
        "batch_id": batch_id,
        "configuration_hashes": identity.configuration_hash_map,
        "code_hashes": identity.code_hash_map,
        "input_receipts_sha256": identity.input_receipts_sha256,
    }
    for key, expected in expected_scalars.items():
        _require(success.get(key) == expected, f"Family cache binding differs: {key}")
    tables = success.get("table_receipts")
    _require(
        isinstance(tables, dict) and set(tables) == {value_name, lineage_name},
        "Family cache table set differs",
    )
    receipts = (
        _table_receipt(tables[value_name], value_name),
        _table_receipt(tables[lineage_name], lineage_name),
    )
    _require(success["row_count"] == receipts[0].rows, "Family row receipt differs")
    _require(
        success["lineage_row_count"] == receipts[1].rows,
        "Family lineage receipt differs",
    )
    _require(
        receipts[1].rows == receipts[0].rows * feature_count,
        "Family lineage cardinality differs",
    )
    return success, receipts


def authenticate_materialized_family_batch(
    *,
    inputs: FeatureStoreInputs,
    family: str,
    batch_id: str,
    policy_path: Path = DEFAULT_POLICY,
    build_identity: FeatureStoreBuildIdentity | None = None,
) -> FamilyBatchReceipt:
    """Authenticate an exact cache bundle for safe deterministic resume."""

    identity = (
        compute_feature_store_build_identity(inputs, policy_path)
        if build_identity is None
        else _validated_precomputed_identity(inputs, build_identity, policy_path)
    )
    policy, _semantic, _contract = _frozen_policy(policy_path)
    names = family_feature_names(family, policy_path)
    value_name = f"feature_values_{family}.parquet"
    lineage_name = f"feature_lineage_{family}.parquet"
    destination = _batch_path(inputs, policy, identity, family, batch_id)
    cache_root = _cache_root(inputs, policy)
    success_path = destination / _SUCCESS_NAME
    try:
        with AuthenticatedPointerGuard.open(
            trusted_root=cache_root,
            path=success_path,
            label="family cache success receipt",
        ) as success_guard:
            _success, receipts = _parse_success(
                success_guard.payload,
                identity,
                family,
                batch_id,
                value_name,
                lineage_name,
                len(names),
            )
            receipt_map = {item.path: item for item in receipts}
            expected_hashes = {
                _SUCCESS_NAME: success_guard.payload_sha256,
                **{item.path: item.sha256 for item in receipts},
            }
            schemas = {
                value_name: family_value_schema(family, policy_path),
                lineage_name: LINEAGE_SCHEMA,
            }
            with AuthenticatedBundleGuard.open(
                trusted_root=cache_root,
                path=destination,
                expected_files=tuple(expected_hashes),
                expected_hashes=expected_hashes,
                file_validator=_parquet_validator(schemas, receipt_map),
                label="family cache bundle",
                _permit_authenticated_directory_ctime_refresh=True,
            ) as bundle:
                success_guard.gate()
                bundle.gate()
    except (SecurePublicationError, FeatureStoreMaterializationError):
        raise
    except Exception as exc:
        raise FeatureStoreMaterializationError(
            f"Cannot authenticate family cache batch: {family}/{batch_id}"
        ) from exc
    return FamilyBatchReceipt(
        path=destination,
        build_id=identity.build_id,
        family=family,
        batch_id=batch_id,
        row_count=receipts[0].rows,
        lineage_row_count=receipts[1].rows,
        table_receipts=receipts,
        success_sha256=success_guard.payload_sha256,
        success_size_bytes=success_guard.record.size_bytes,
        reused_existing=True,
    )


def materialize_feature_family_batch(
    *,
    inputs: FeatureStoreInputs,
    family: str,
    batch_id: str,
    rows: Iterable[FeatureMaterializationRow],
    policy_path: Path = DEFAULT_POLICY,
    build_identity: FeatureStoreBuildIdentity | None = None,
) -> FamilyBatchReceipt:
    """Serialize one deterministic family batch to an immutable cache bundle."""

    identity = (
        compute_feature_store_build_identity(inputs, policy_path)
        if build_identity is None
        else _validated_precomputed_identity(inputs, build_identity, policy_path)
    )
    policy, _semantic, _contract = _frozen_policy(policy_path)
    values, lineage = _materialize_tables(rows, family, policy_path)
    row_group_size = int(policy["build"]["row_group_size"])
    compression = str(policy["build"]["compression"])
    dictionary = bool(policy["build"]["dictionary_encode_strings"])
    value_name = f"feature_values_{family}.parquet"
    lineage_name = f"feature_lineage_{family}.parquet"
    expected_files = (_SUCCESS_NAME, value_name, lineage_name)
    cache_root = _cache_root(inputs, policy)
    destination = _batch_path(inputs, policy, identity, family, batch_id)

    try:
        with SecureStagingDirectory.create(
            trusted_root=inputs.database_root,
            destination=destination,
            expected_files=expected_files,
            label="Stage 6 cache family batch",
        ) as stage:
            receipts = (
                _write_parquet(
                    stage,
                    value_name,
                    values,
                    row_group_size=row_group_size,
                    compression=compression,
                    dictionary=dictionary,
                ),
                _write_parquet(
                    stage,
                    lineage_name,
                    lineage,
                    row_group_size=row_group_size,
                    compression=compression,
                    dictionary=dictionary,
                ),
            )
            success_bytes = canonical_json_bytes(
                _success_payload(identity, policy, family, batch_id, receipts)
            )
            with stage.open_exclusive_binary(_SUCCESS_NAME) as stream:
                stream.write(success_bytes)
            receipt_map = {item.path: item for item in receipts}
            schemas = {value_name: values.schema, lineage_name: lineage.schema}
            validator = _parquet_validator(schemas, receipt_map)
            written_hashes = {
                _SUCCESS_NAME: hashlib.sha256(success_bytes).hexdigest(),
                **{item.path: item.sha256 for item in receipts},
            }
            snapshot = stage.snapshot(
                expected_hashes=written_hashes,
                file_validator=validator,
            )
            _require(
                snapshot.hashes[value_name] == receipts[0].sha256
                and snapshot.hashes[lineage_name] == receipts[1].sha256
                and snapshot.hashes[_SUCCESS_NAME]
                == written_hashes[_SUCCESS_NAME],
                "Staged family receipts differ from written bytes",
            )

            def reauthenticate_build() -> None:
                if build_identity is None:
                    reauthenticate_feature_store_build_identity(
                        inputs,
                        identity,
                        policy_path,
                    )
                else:
                    _validated_precomputed_identity(
                        inputs,
                        identity,
                        policy_path,
                    )

            published = stage.publish(
                expected_hashes=snapshot.hashes,
                file_validator=validator,
                pre_publish_gate=reauthenticate_build,
                post_publish_gate=reauthenticate_build,
            )
    except (SecurePublicationError, FeatureStoreMaterializationError):
        raise
    except (FeatureStoreContractError, pa.ArrowException, OSError) as exc:
        raise FeatureStoreMaterializationError(
            f"Cannot materialize family cache batch: {family}/{batch_id}"
        ) from exc

    authenticated = authenticate_materialized_family_batch(
        inputs=inputs,
        family=family,
        batch_id=batch_id,
        policy_path=policy_path,
        build_identity=identity,
    )
    return FamilyBatchReceipt(
        path=authenticated.path,
        build_id=authenticated.build_id,
        family=authenticated.family,
        batch_id=authenticated.batch_id,
        row_count=authenticated.row_count,
        lineage_row_count=authenticated.lineage_row_count,
        table_receipts=authenticated.table_receipts,
        success_sha256=authenticated.success_sha256,
        success_size_bytes=authenticated.success_size_bytes,
        reused_existing=published.reused_existing,
    )
