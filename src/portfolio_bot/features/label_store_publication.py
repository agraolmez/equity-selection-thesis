"""Secure primary publication for the Stage 7 development label store.

The publisher consumes only the typed result of the completed monthly cache
materialization.  Every cache completion and both of its Parquet members are
authenticated before use.  It then stream-concatenates the audit and lineage
tables, derives the processed and quality payloads, publishes three immutable
closed bundles and one immutable manifest, and replaces the mutable primary
pointer last.

This module does not read validation/final-period data, fit a model, construct
a portfolio, calculate P&L, or authorize any downstream use.  The primary
pointer requires a separately implemented Stage 7.4 verification.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    AuthenticatedBundleGuard,
    AuthenticatedPointerGuard,
    PublishedBundle,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.features.label_store_contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
    published_schemas,
    sha256_file,
    validate_label_store_contract,
)
from portfolio_bot.features.label_store_inputs import LabelStoreInputs
from portfolio_bot.features.label_store_materialization import (
    LabelCacheFileReceipt,
    LabelMonthReceipt,
    LabelStoreBuildIdentity,
    LabelStoreCacheResult,
    project_development_labels,
    reauthenticate_label_store_build_identity,
)
from portfolio_bot.features.label_store_quality import (
    build_label_timing_checks,
    build_month_label_quality,
    build_quality_summary,
    build_split_membership,
)
from portfolio_bot.features.phase_03_contract import canonical_json_bytes


_SUCCESS_NAME = "_SUCCESS.json"
_MANIFEST_NAME = "manifest.json"
_AUDIT_NAME = "development_label_audit.parquet"
_LINEAGE_NAME = "development_label_leg_lineage.parquet"
_LABELS_NAME = "development_labels.parquet"
_SUMMARY_NAME = "development_label_month_summary.parquet"
_SPLIT_NAME = "label_split_membership.parquet"
_COVERAGE_NAME = "label_coverage.parquet"
_STATUS_NAME = "label_status_counts.parquet"
_TIMING_NAME = "label_timing_checks.parquet"
_DISTRIBUTION_NAME = "label_distribution.parquet"
_QUALITY_SUMMARY_NAME = "quality_summary.json"

_CACHE_AUDIT_NAME = "label_audit.parquet"
_CACHE_LINEAGE_NAME = "label_leg_lineage.parquet"
_CACHE_FILES = (
    _SUCCESS_NAME,
    _CACHE_AUDIT_NAME,
    _CACHE_LINEAGE_NAME,
)
_CACHE_SUCCESS_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "status",
        "ordinal",
        "signal_date",
        "summary",
        "table_receipts",
    }
)
_CACHE_RECEIPT_KEYS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "rows",
        "row_groups",
        "schema_sha256",
    }
)

_INTERIM_LOCATION = "interim_label_store"
_INTERIM_STATUS = "PRIMARY_INTERIM_COMPLETE"
_PROCESSED_LOCATION = "processed_label_store"
_PROCESSED_STATUS = "PRIMARY_PROCESSED_COMPLETE"
_QUALITY_LOCATION = "label_store_quality"
_QUALITY_STATUS = "PRIMARY_QUALITY_PASSED"
_PRIMARY_STATUS = "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"


class LabelStorePublicationError(RuntimeError):
    """Raised when Stage 7 primary publication cannot be proven correct."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LabelStorePublicationError(message)


def _toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LabelStorePublicationError(
            f"Cannot read Stage 7 publication policy: {path}"
        ) from exc
    _require(isinstance(value, dict), "Stage 7 publication policy is not a table")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _parquet_physical_schema(schema: pa.Schema) -> pa.Schema:
    """Return the schema produced by an Arrow-to-Parquet round trip."""

    fields: list[pa.Field] = []
    for field in schema:
        field_type = field.type
        if pa.types.is_list(field_type):
            field_type = pa.list_(
                pa.field(
                    "element",
                    field_type.value_type,
                    nullable=field_type.value_field.nullable,
                )
            )
        fields.append(
            pa.field(
                field.name,
                field_type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        )
    return pa.schema(fields, metadata=schema.metadata)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _parse_canonical_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LabelStorePublicationError(f"{label} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    _require(
        payload == canonical_json_bytes(value),
        f"{label} is not canonical JSON",
    )
    return value


@dataclass(frozen=True)
class _PublicationFileReceipt:
    path: str
    sha256: str
    size_bytes: int
    rows: int | None
    row_groups: int | None
    schema_sha256: str | None

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
class PublishedLabelOutputSet:
    """One immutable Stage 7 output bundle and its table receipts."""

    path: Path
    bundle: PublishedBundle
    files: Mapping[str, _PublicationFileReceipt]

    @property
    def success_sha256(self) -> str:
        return self.bundle.snapshot.hashes[_SUCCESS_NAME]


@dataclass(frozen=True)
class PrimaryLabelStorePublication:
    """The complete primary Stage 7 publication handoff."""

    build_id: str
    interim_path: Path
    processed_path: Path
    quality_path: Path
    manifest_path: Path
    pointer_path: Path
    manifest: Mapping[str, Any]
    pointer: Mapping[str, Any]
    reused_existing: bool


@dataclass(frozen=True)
class _PublicationContext:
    policy: Mapping[str, Any]
    contract: Mapping[str, Any]
    schemas: Mapping[str, pa.Schema]
    required_audit_rows: int
    required_long_rows: int
    required_signal_dates: int
    required_split_rows: int
    row_group_size: int
    compression: str
    dictionary: bool
    interim_files: tuple[str, ...]
    processed_files: tuple[str, ...]
    quality_files: tuple[str, ...]
    interim_path: Path
    processed_path: Path
    quality_path: Path
    manifest_directory: Path
    pointer_path: Path


@dataclass(frozen=True)
class _AuthenticatedMonth:
    receipt: LabelMonthReceipt
    audit: LabelCacheFileReceipt
    lineage: LabelCacheFileReceipt
    expected_hashes: Mapping[str, str]
    summary: Mapping[str, Any]


def _required_scope_counts(
    policy: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    """Return frozen counts; isolated for bounded synthetic publication tests."""

    scope = policy["scope"]
    return (
        int(scope["required_audit_rows"]),
        int(scope["required_long_eligible_rows"]),
        int(scope["required_signal_dates"]),
        int(scope["required_split_membership_rows"]),
    )


def _safe_relative(value: object, *, prefix: str, label: str) -> Path:
    path = Path(str(value))
    _require(
        not path.is_absolute()
        and bool(path.parts)
        and path.parts[0] == prefix
        and ".." not in path.parts,
        f"Unsafe Stage 7 {label}: {value}",
    )
    return path


def _require_frozen_rules(
    *,
    policy: Mapping[str, Any],
    rules: Mapping[str, Any],
    policy_path: Path,
) -> None:
    """Reject a caller-supplied rules object that is not the reviewed file."""

    specification = policy["physical_rules"]
    candidate = Path(str(specification["path"])).expanduser()
    if not candidate.is_absolute():
        # The default policy lives at <project>/configs/phase_03_features.
        candidate = policy_path.parents[2] / candidate
    candidate = Path(os.path.abspath(candidate))
    _require(
        candidate.is_file()
        and not candidate.is_symlink()
        and sha256_file(candidate) == specification["sha256"],
        "Stage 7 frozen physical-rules file differs",
    )
    try:
        frozen = json.loads(candidate.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LabelStorePublicationError(
            "Cannot read the frozen Stage 7 physical rules"
        ) from exc
    _require(
        isinstance(frozen, dict)
        and canonical_json_bytes(frozen) == canonical_json_bytes(rules),
        "Stage 7 supplied rules are not the frozen reviewed rules",
    )


def _context(
    *,
    inputs: LabelStoreInputs,
    identity: LabelStoreBuildIdentity,
    rules: Mapping[str, Any],
    policy_path: Path,
    project_root: Path,
) -> _PublicationContext:
    contract = validate_label_store_contract(policy_path)
    policy = _toml(policy_path)
    _require_frozen_rules(
        policy=policy,
        rules=rules,
        policy_path=policy_path,
    )
    _require(
        identity.build_id.startswith("phase03-labels-"),
        "Stage 7 primary build ID is malformed",
    )
    _require(
        identity.contract_id == contract["contract_id"]
        and identity.policy_id == contract["policy_id"]
        and identity.contract_id == policy["contract_id"]
        and identity.policy_id == policy["policy_id"],
        "Stage 7 primary identity differs from the frozen contract",
    )
    _require(
        rules.get("contract_id") == identity.contract_id,
        "Stage 7 physical rules have a different contract ID",
    )
    schemas = published_schemas(rules)
    contract_schemas = contract["published_schemas"]
    _require(
        set(schemas) == set(contract_schemas)
        and all(
            schemas[name].equals(contract_schemas[name], check_metadata=True)
            for name in schemas
        ),
        "Stage 7 supplied rules differ from the validated published schemas",
    )
    required = _required_scope_counts(policy)
    _require(
        required
        == (
            int(contract["required_audit_rows"]),
            int(contract["required_long_eligible_rows"]),
            int(contract["required_signal_dates"]),
            int(contract["required_split_membership_rows"]),
        ),
        "Stage 7 publication scope differs from the validated contract",
    )
    database_root = Path(os.path.abspath(inputs.database_root.expanduser()))

    def database_path(key: str) -> Path:
        relative = _safe_relative(
            policy["database"][key],
            prefix=key.split("_", 1)[0],
            label=f"database namespace {key}",
        )
        return database_root / relative / identity.build_id

    artifact_root = _safe_relative(
        policy["publication"]["artifact_root"],
        prefix="artifacts",
        label="artifact namespace",
    )
    pointer_relative = _safe_relative(
        policy["publication"]["latest_primary_pointer"],
        prefix="artifacts",
        label="primary pointer",
    )
    output_sets = contract["output_sets"]
    return _PublicationContext(
        policy=policy,
        contract=contract,
        schemas=schemas,
        required_audit_rows=required[0],
        required_long_rows=required[1],
        required_signal_dates=required[2],
        required_split_rows=required[3],
        row_group_size=int(policy["build"]["row_group_size"]),
        compression=str(policy["build"]["compression"]),
        dictionary=bool(policy["build"]["dictionary_encode_strings"]),
        interim_files=tuple(sorted(output_sets["interim"])),
        processed_files=tuple(sorted(output_sets["processed"])),
        quality_files=tuple(sorted(output_sets["quality"])),
        interim_path=database_path("interim_build_root"),
        processed_path=database_path("processed_build_root"),
        quality_path=database_path("quality_build_root"),
        manifest_directory=(
            project_root
            / artifact_root
            / str(policy["publication"]["primary_build_namespace"])
            / identity.build_id
        ),
        pointer_path=project_root / pointer_relative,
    )


def _cache_path(
    path: Path,
    *,
    database_root: Path,
) -> Path:
    root = Path(os.path.abspath(database_root.expanduser()))
    candidate = Path(os.path.abspath(path.expanduser()))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise LabelStorePublicationError(
            "Stage 7 month cache escaped the database root"
        ) from exc
    _require(
        bool(relative.parts)
        and relative.parts[0] == "cache"
        and ".." not in relative.parts,
        "Stage 7 month cache is outside the cache lifecycle",
    )
    return candidate


def _compare_cache_file_receipt(
    provided: LabelCacheFileReceipt,
    payload: Mapping[str, Any],
    *,
    expected_name: str,
) -> None:
    _require(
        set(payload) == _CACHE_RECEIPT_KEYS,
        f"Stage 7 cache receipt keys differ: {expected_name}",
    )
    _require(
        provided.path == expected_name
        and payload == provided.as_dict(),
        f"Stage 7 cache receipt changed: {expected_name}",
    )
    _require(
        isinstance(provided.rows, int)
        and provided.rows >= 0
        and isinstance(provided.row_groups, int)
        and provided.row_groups >= 0
        and isinstance(provided.size_bytes, int)
        and provided.size_bytes > 0
        and len(provided.sha256) == 64
        and len(provided.schema_sha256) == 64,
        f"Stage 7 cache receipt is malformed: {expected_name}",
    )


def _cache_file_validator(
    *,
    audit: LabelCacheFileReceipt,
    lineage: LabelCacheFileReceipt,
    schemas: Mapping[str, pa.Schema],
    expected_long_rows: int,
):
    specifications = {
        _CACHE_AUDIT_NAME: (
            audit,
            schemas[_AUDIT_NAME],
        ),
        _CACHE_LINEAGE_NAME: (
            lineage,
            schemas[_LINEAGE_NAME],
        ),
    }

    def validate(name: str, descriptor: int, record: Any) -> None:
        if name not in specifications:
            return
        receipt, expected_schema = specifications[name]
        _require(
            record.sha256 == receipt.sha256
            and record.size_bytes == receipt.size_bytes,
            f"Stage 7 cache file receipt differs: {name}",
        )
        duplicate = os.dup(descriptor)
        try:
            with os.fdopen(duplicate, "rb") as handle:
                parquet = pq.ParquetFile(handle)
                observed_schema = parquet.schema_arrow
                rows = parquet.metadata.num_rows
                row_groups = parquet.metadata.num_row_groups
                long_rows = None
                if name == _CACHE_AUDIT_NAME:
                    long_rows = 0
                    for batch in parquet.iter_batches(
                        batch_size=131_072,
                        columns=["long_eligible"],
                        use_threads=False,
                    ):
                        long_rows += int(
                            pc.sum(batch.column(0)).as_py() or 0
                        )
        except Exception as exc:
            raise LabelStorePublicationError(
                f"Cannot validate Stage 7 cache Parquet: {name}"
            ) from exc
        _require(
            observed_schema.equals(
                _parquet_physical_schema(expected_schema),
                check_metadata=True,
            )
            and rows == receipt.rows
            and row_groups == receipt.row_groups
            and _schema_sha256(observed_schema) == receipt.schema_sha256,
            f"Stage 7 cache Parquet metadata differs: {name}",
        )
        if name == _CACHE_AUDIT_NAME:
            _require(
                long_rows == expected_long_rows,
                "Stage 7 cache long-row count differs",
            )

    return validate


def _authenticate_month(
    *,
    inputs: LabelStoreInputs,
    identity: LabelStoreBuildIdentity,
    receipt: LabelMonthReceipt,
    context: _PublicationContext,
) -> _AuthenticatedMonth:
    _require(
        isinstance(receipt, LabelMonthReceipt)
        and receipt.build_id == identity.build_id,
        "Stage 7 cache contains an invalid month receipt",
    )
    path = _cache_path(receipt.path, database_root=inputs.database_root)
    _require(
        path.name
        == f"month-{receipt.ordinal:06d}-{receipt.signal_date.isoformat()}",
        "Stage 7 cache month path does not bind its ordinal and signal date",
    )
    files = {item.path: item for item in receipt.files}
    _require(
        len(files) == len(receipt.files)
        and set(files) == {_CACHE_AUDIT_NAME, _CACHE_LINEAGE_NAME},
        "Stage 7 month cache table set differs",
    )
    audit = files[_CACHE_AUDIT_NAME]
    lineage = files[_CACHE_LINEAGE_NAME]
    expected_hashes = {
        _SUCCESS_NAME: receipt.success_sha256,
        _CACHE_AUDIT_NAME: audit.sha256,
        _CACHE_LINEAGE_NAME: lineage.sha256,
    }
    with AuthenticatedPointerGuard.open(
        trusted_root=inputs.database_root,
        path=path / _SUCCESS_NAME,
        expected_sha256=receipt.success_sha256,
        label=f"Stage 7 month {receipt.ordinal:03d} cache completion",
    ) as pointer_guard:
        payload = _parse_canonical_json(
            pointer_guard.payload,
            f"Stage 7 month {receipt.ordinal:03d} cache completion",
        )
        _require(
            set(payload) == _CACHE_SUCCESS_KEYS,
            "Stage 7 cache completion keys differ",
        )
        _require(
            payload["schema_version"] == 1
            and payload["contract_id"] == identity.contract_id
            and payload["policy_id"] == identity.policy_id
            and payload["build_id"] == identity.build_id
            and payload["status"] == "LABEL_MONTH_CACHE_COMPLETE"
            and payload["ordinal"] == receipt.ordinal
            and payload["signal_date"] == receipt.signal_date.isoformat(),
            "Stage 7 cache completion identity differs",
        )
        table_receipts = payload["table_receipts"]
        _require(
            isinstance(table_receipts, Mapping)
            and set(table_receipts) == set(files),
            "Stage 7 cache completion table set differs",
        )
        for name, supplied in files.items():
            value = table_receipts[name]
            _require(
                isinstance(value, Mapping),
                f"Stage 7 cache receipt is not an object: {name}",
            )
            _compare_cache_file_receipt(
                supplied,
                value,
                expected_name=name,
            )
        summary = payload["summary"]
        _require(
            isinstance(summary, Mapping)
            and summary == _json_safe(receipt.summary),
            "Stage 7 cache month summary changed",
        )
        pointer_guard.gate()

    summary_schema = context.schemas[_SUMMARY_NAME]
    try:
        summary_table = pa.Table.from_pylist(
            [dict(receipt.summary)],
            schema=summary_schema,
        )
    except (TypeError, ValueError, pa.ArrowException) as exc:
        raise LabelStorePublicationError(
            "Stage 7 cache month summary has an invalid schema"
        ) from exc
    _require(
        summary_table.schema.equals(summary_schema, check_metadata=True)
        and summary_table.num_rows == 1
        and summary_table["signal_date"][0].as_py() == receipt.signal_date
        and int(
            summary_table["eligible_security_count"][0].as_py()
        )
        == receipt.long_rows,
        "Stage 7 cache month summary differs from its receipt",
    )
    validator = _cache_file_validator(
        audit=audit,
        lineage=lineage,
        schemas=context.schemas,
        expected_long_rows=receipt.long_rows,
    )
    with AuthenticatedBundleGuard.open(
        trusted_root=inputs.database_root,
        path=path,
        expected_files=_CACHE_FILES,
        expected_hashes=expected_hashes,
        file_validator=validator,
        label=f"Stage 7 month {receipt.ordinal:03d} cache bundle",
        _permit_authenticated_directory_ctime_refresh=True,
    ) as bundle_guard:
        bundle_guard.gate()
    _require(
        audit.rows == receipt.audit_rows
        and lineage.rows == receipt.lineage_rows,
        "Stage 7 cache table row counts differ from the month receipt",
    )
    return _AuthenticatedMonth(
        receipt=receipt,
        audit=audit,
        lineage=lineage,
        expected_hashes=expected_hashes,
        summary=dict(receipt.summary),
    )


def _authenticate_cache(
    *,
    inputs: LabelStoreInputs,
    cache_result: LabelStoreCacheResult,
    context: _PublicationContext,
) -> tuple[_AuthenticatedMonth, ...]:
    _require(
        isinstance(cache_result, LabelStoreCacheResult)
        and isinstance(cache_result.identity, LabelStoreBuildIdentity)
        and cache_result.final_identity_reauthenticated is True,
        "Stage 7 publication requires a fully reauthenticated cache result",
    )
    identity = cache_result.identity
    receipts = cache_result.month_receipts
    _require(
        len(receipts) == context.required_signal_dates,
        "Stage 7 cache month count differs",
    )
    authenticated: list[_AuthenticatedMonth] = []
    physical_paths: set[str] = set()
    previous_signal: dt.date | None = None
    for ordinal, receipt in enumerate(receipts):
        _require(
            receipt.ordinal == ordinal
            and isinstance(receipt.signal_date, dt.date)
            and (
                previous_signal is None
                or previous_signal < receipt.signal_date
            ),
            "Stage 7 cache months are not unique and strictly ordered",
        )
        path_label = _cache_path(
            receipt.path,
            database_root=inputs.database_root,
        ).as_posix()
        _require(
            path_label not in physical_paths,
            "Stage 7 cache reuses one physical month directory",
        )
        physical_paths.add(path_label)
        authenticated.append(
            _authenticate_month(
                inputs=inputs,
                identity=identity,
                receipt=receipt,
                context=context,
            )
        )
        previous_signal = receipt.signal_date
    scope = context.policy["scope"]
    _require(
        authenticated[0].receipt.signal_date.isoformat()
        == str(scope["first_signal_date"])
        and authenticated[-1].receipt.signal_date.isoformat()
        == str(scope["last_signal_date"]),
        "Stage 7 cache signal-date boundaries differ",
    )
    _require(
        sum(item.receipt.audit_rows for item in authenticated)
        == cache_result.audit_rows
        == context.required_audit_rows
        and sum(item.receipt.long_rows for item in authenticated)
        == cache_result.long_rows
        == context.required_long_rows
        and sum(item.receipt.lineage_rows for item in authenticated)
        == cache_result.lineage_rows,
        "Stage 7 cache aggregate counts differ",
    )
    return tuple(authenticated)


def _open_month_guard(
    month: _AuthenticatedMonth,
    *,
    inputs: LabelStoreInputs,
    context: _PublicationContext,
) -> AuthenticatedBundleGuard:
    return AuthenticatedBundleGuard.open(
        trusted_root=inputs.database_root,
        path=month.receipt.path,
        expected_files=_CACHE_FILES,
        expected_hashes=month.expected_hashes,
        file_validator=_cache_file_validator(
            audit=month.audit,
            lineage=month.lineage,
            schemas=context.schemas,
            expected_long_rows=month.receipt.long_rows,
        ),
        label=f"Stage 7 month {month.receipt.ordinal:03d} cache read",
        _permit_authenticated_directory_ctime_refresh=True,
    )


def _key_rows(
    batch: pa.RecordBatch | pa.Table,
    names: Sequence[str],
) -> Iterator[tuple[Any, ...]]:
    columns = [batch[name].to_pylist() for name in names]
    yield from (
        tuple(column[index] for column in columns)
        for index in range(batch.num_rows)
    )


def _validate_stream_key(
    *,
    batch: pa.RecordBatch | pa.Table,
    names: Sequence[str],
    previous: tuple[Any, ...] | None,
    label: str,
) -> tuple[Any, ...] | None:
    for key in _key_rows(batch, names):
        _require(
            all(value is not None for value in key),
            f"{label} contains a null primary key",
        )
        _require(
            previous is None or previous < key,
            f"{label} is not globally unique and strictly sorted",
        )
        previous = key
    return previous


def _close_float(
    left: object,
    right: object,
    *,
    rules: Mapping[str, Any],
) -> bool:
    if left is None or right is None:
        return left is right
    comparison = rules["implementation_conventions"]["float_comparison"]
    return math.isclose(
        float(left),
        float(right),
        rel_tol=float(comparison["label_relative_tolerance"]),
        abs_tol=float(comparison["label_absolute_tolerance"]),
    )


def _guarded_month_audit(
    *,
    guard: AuthenticatedBundleGuard,
    month: _AuthenticatedMonth,
    context: _PublicationContext,
) -> pa.Table:
    with guard.open_binary(_CACHE_AUDIT_NAME) as handle:
        audit = pq.ParquetFile(handle).read(use_threads=False).combine_chunks()
    _require(
        audit.schema.equals(context.schemas[_AUDIT_NAME], check_metadata=True)
        and audit.num_rows == month.receipt.audit_rows,
        "Stage 7 lineage-domain audit differs",
    )
    _validate_stream_key(
        batch=audit,
        names=("signal_date", "permno"),
        previous=None,
        label="Stage 7 lineage-domain audit",
    )
    _require(
        set(audit["signal_date"].to_pylist())
        == {month.receipt.signal_date},
        "Stage 7 lineage-domain audit signal differs",
    )
    return audit


def _validate_lineage_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    audit_by_permno: Mapping[int, Mapping[str, Any]],
    state: dict[int, dict[str, Any]],
    rules: Mapping[str, Any],
) -> None:
    factor_names = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
    allowed_classes = {
        "ordinary_one_session",
        "ordinary_plus_terminal_delisting_adjustment",
    }
    for row in rows:
        permno = int(row["permno"])
        audit = audit_by_permno.get(permno)
        _require(
            audit is not None
            and bool(audit["long_eligible"])
            and audit["outcome_status"] == "PRESENT",
            "Stage 7 lineage key is outside the accepted audit domain",
        )
        values = [
            row["authoritative_total_return"],
            row["rf"],
            row["systematic_component"],
            row["excess_return_contribution"],
            row["residual_return_contribution"],
            *(row[name] for name in factor_names),
        ]
        _require(
            all(
                value is not None and math.isfinite(float(value))
                for value in values
            ),
            "Stage 7 lineage contains a missing or nonfinite numeric value",
        )
        date = row["calendar_date"]
        _require(
            audit["entry_date"] < date <= audit["exit_date"],
            "Stage 7 lineage date is outside the forward interval",
        )
        _require(
            row["return_leg_class"] in allowed_classes
            and float(row["authoritative_total_return"]) >= -1.0,
            "Stage 7 lineage contains an unusable return leg",
        )
        _require(
            row["factor_public_availability_date"] <= date,
            "Stage 7 lineage uses a factor before public availability",
        )
        security_available = audit["security_label_available_date"]
        _require(
            security_available is not None
            and row["outcome_availability_date"] <= security_available,
            "Stage 7 lineage availability exceeds its audit binding",
        )
        tracker = state.setdefault(
            permno,
            {
                "count": 0,
                "first_date": None,
                "last_date": None,
                "terminal_count": 0,
                "terminal_date": None,
                "max_outcome_availability_date": None,
                "excess": [],
                "residual": [],
            },
        )
        tracker["count"] += 1
        _require(
            int(row["leg_ordinal"]) == tracker["count"]
            and (
                tracker["last_date"] is None
                or tracker["last_date"] < date
            ),
            "Stage 7 lineage leg ordinal or chronology differs",
        )
        if tracker["first_date"] is None:
            tracker["first_date"] = date
        tracker["last_date"] = date
        outcome_available = row["outcome_availability_date"]
        tracker["max_outcome_availability_date"] = max(
            outcome_available,
            tracker["max_outcome_availability_date"]
            or outcome_available,
        )
        betas = [
            audit[f"beta_label_{name}"] for name in factor_names
        ]
        _require(
            bool(audit["beta_complete"])
            and all(
                value is not None and math.isfinite(float(value))
                for value in betas
            ),
            "Stage 7 accepted lineage has incomplete label betas",
        )
        systematic = math.fsum(
            float(beta) * float(row[name])
            for beta, name in zip(betas, factor_names, strict=True)
        )
        excess = math.fsum(
            (
                float(row["authoritative_total_return"]),
                -float(row["rf"]),
            )
        )
        residual = math.fsum((excess, -systematic))
        _require(
            _close_float(
                systematic,
                row["systematic_component"],
                rules=rules,
            )
            and _close_float(
                excess,
                row["excess_return_contribution"],
                rules=rules,
            )
            and _close_float(
                residual,
                row["residual_return_contribution"],
                rules=rules,
            ),
            "Stage 7 lineage return formula differs",
        )
        tracker["excess"].append(float(row["excess_return_contribution"]))
        tracker["residual"].append(
            float(row["residual_return_contribution"])
        )
        if bool(row["terminal_delisting"]):
            tracker["terminal_count"] += 1
            tracker["terminal_date"] = date
            _require(
                tracker["terminal_count"] == 1,
                "Stage 7 lineage double-counts a terminal delisting",
            )


def _finish_lineage_validation(
    *,
    audit_by_permno: Mapping[int, Mapping[str, Any]],
    state: Mapping[int, Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> None:
    for permno, audit in audit_by_permno.items():
        tracker = state.get(permno)
        present = (
            bool(audit["long_eligible"])
            and audit["outcome_status"] == "PRESENT"
        )
        if not present:
            _require(
                tracker is None and int(audit["used_leg_count"]) == 0,
                "Stage 7 nonpresent audit row has accepted lineage",
            )
            continue
        _require(
            tracker is not None
            and int(audit["used_leg_count"]) == tracker["count"]
            and tracker["count"] > 0
            and audit["first_used_leg_date"] == tracker["first_date"]
            and audit["last_used_leg_date"] == tracker["last_date"]
            and audit["max_outcome_availability_date"]
            == tracker["max_outcome_availability_date"],
            "Stage 7 audit-to-lineage cardinality or date binding differs",
        )
        expected_security_available = max(
            audit["exit_date"],
            tracker["max_outcome_availability_date"],
        )
        _require(
            audit["security_label_available_date"]
            == expected_security_available,
            "Stage 7 security availability differs from its lineage",
        )
        terminal_count = int(tracker["terminal_count"])
        _require(
            bool(audit["terminal_delisting_used"])
            == (terminal_count == 1)
            and audit["terminal_delisting_date"]
            == tracker["terminal_date"],
            "Stage 7 terminal-delisting lineage binding differs",
        )
        _require(
            _close_float(
                math.fsum(tracker["excess"]),
                audit["unranked_excess_return"],
                rules=rules,
            )
            and _close_float(
                math.fsum(tracker["residual"]),
                audit["unranked_residual_return"],
                rules=rules,
            ),
            "Stage 7 audit forward-return sum differs from lineage",
        )


def _cache_batches(
    *,
    inputs: LabelStoreInputs,
    months: Sequence[_AuthenticatedMonth],
    context: _PublicationContext,
    cache_name: str,
    output_name: str,
    key: Sequence[str],
    expected_rows: int,
    rules: Mapping[str, Any],
) -> Iterator[pa.RecordBatch]:
    schema = context.schemas[output_name]
    emitted = 0
    previous: tuple[Any, ...] | None = None
    for month in months:
        with _open_month_guard(
            month,
            inputs=inputs,
            context=context,
        ) as guard:
            audit_by_permno: dict[int, Mapping[str, Any]] | None = None
            lineage_state: dict[int, dict[str, Any]] = {}
            if cache_name == _CACHE_LINEAGE_NAME:
                audit = _guarded_month_audit(
                    guard=guard,
                    month=month,
                    context=context,
                )
                audit_by_permno = {
                    int(row["permno"]): row
                    for row in audit.to_pylist()
                }
            with guard.open_binary(cache_name) as handle:
                parquet = pq.ParquetFile(handle)
                _require(
                    parquet.schema_arrow.equals(
                        _parquet_physical_schema(schema),
                        check_metadata=True,
                    ),
                    f"Stage 7 cache schema changed while reading: {cache_name}",
                )
                for batch in parquet.iter_batches(
                    batch_size=context.row_group_size,
                    use_threads=False,
                ):
                    _require(
                        batch.schema.equals(schema, check_metadata=True),
                        f"Stage 7 cache batch schema differs: {cache_name}",
                    )
                    if batch.num_rows == 0:
                        continue
                    signal_index = batch.schema.get_field_index("signal_date")
                    signals = set(batch.column(signal_index).to_pylist())
                    _require(
                        signals == {month.receipt.signal_date},
                        f"Stage 7 cache month signal differs: {cache_name}",
                    )
                    previous = _validate_stream_key(
                        batch=batch,
                        names=key,
                        previous=previous,
                        label=f"Stage 7 published {output_name}",
                    )
                    if audit_by_permno is not None:
                        _validate_lineage_rows(
                            rows=batch.to_pylist(),
                            audit_by_permno=audit_by_permno,
                            state=lineage_state,
                            rules=rules,
                        )
                    emitted += batch.num_rows
                    yield batch
            if audit_by_permno is not None:
                _finish_lineage_validation(
                    audit_by_permno=audit_by_permno,
                    state=lineage_state,
                    rules=rules,
                )
            guard.gate()
    _require(
        emitted == expected_rows,
        f"Stage 7 published row count differs: {output_name}",
    )


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
        _require(not self.closed, "Stage 7 publication sink is closed")
        view = memoryview(payload)
        written = self._stream.write(view)
        _require(written == len(view), "Short Stage 7 publication write")
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


def _write_parquet_batches(
    *,
    stage: SecureStagingDirectory,
    name: str,
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
    context: _PublicationContext,
) -> _PublicationFileReceipt:
    rows = 0
    row_groups = 0
    with stage.open_exclusive_binary(name) as stream:
        sink = _HashingSink(stream)
        writer = pq.ParquetWriter(
            sink,
            schema,
            compression=context.compression,
            use_dictionary=context.dictionary,
            version="2.6",
            write_statistics=True,
        )
        try:
            for batch in batches:
                _require(
                    isinstance(batch, pa.RecordBatch)
                    and batch.schema.equals(schema, check_metadata=True),
                    f"Stage 7 publication batch schema differs: {name}",
                )
                if batch.num_rows == 0:
                    continue
                writer.write_batch(
                    batch,
                    row_group_size=context.row_group_size,
                )
                rows += batch.num_rows
                row_groups += math.ceil(
                    batch.num_rows / context.row_group_size
                )
        finally:
            writer.close()
        sink.flush()
    # The physical metadata is re-opened through the staging snapshot
    # validator.  Each write_batch call deterministically emits this many
    # row groups.
    return _PublicationFileReceipt(
        path=name,
        sha256=sink.sha256,
        size_bytes=sink.size_bytes,
        rows=rows,
        row_groups=row_groups,
        schema_sha256=_schema_sha256(_parquet_physical_schema(schema)),
    )


def _write_parquet_table(
    *,
    stage: SecureStagingDirectory,
    name: str,
    table: pa.Table,
    context: _PublicationContext,
) -> _PublicationFileReceipt:
    batches = table.to_batches(max_chunksize=context.row_group_size)
    if not batches:
        batches = [
            pa.RecordBatch.from_arrays(
                [pa.array([], type=field.type) for field in table.schema],
                schema=table.schema,
            )
        ]
    return _write_parquet_batches(
        stage=stage,
        name=name,
        schema=table.schema,
        batches=batches,
        context=context,
    )


def _write_bytes(
    *,
    stage: SecureStagingDirectory,
    name: str,
    payload: bytes,
) -> _PublicationFileReceipt:
    with stage.open_exclusive_binary(name) as stream:
        stream.write(payload)
    return _PublicationFileReceipt(
        path=name,
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        rows=None,
        row_groups=None,
        schema_sha256=None,
    )


def _bindings(identity: LabelStoreBuildIdentity) -> dict[str, Any]:
    return {
        "content_fingerprint": identity.content_fingerprint,
        "configuration_hashes": identity.configuration_hash_map,
        "code_hashes": identity.code_hash_map,
        "input_receipts_sha256": identity.input_receipts_sha256,
    }


def _success_payload(
    *,
    identity: LabelStoreBuildIdentity,
    rules: Mapping[str, Any],
    location_kind: str,
    status: str,
    receipts: Mapping[str, _PublicationFileReceipt],
) -> bytes:
    payload = {
        "schema_version": 1,
        "contract_id": identity.contract_id,
        "policy_id": identity.policy_id,
        "build_id": identity.build_id,
        "location_kind": location_kind,
        "status": status,
        "bindings": _bindings(identity),
        "table_receipts": {
            name: receipts[name].as_dict()
            for name in sorted(receipts)
        },
    }
    required = rules["publication_payloads"]["success_required_keys"]
    _require(
        set(payload) == set(required),
        "Stage 7 output _SUCCESS keys differ",
    )
    return canonical_json_bytes(payload)


def _published_file_validator(
    receipts: Mapping[str, _PublicationFileReceipt],
    schemas: Mapping[str, pa.Schema],
):
    def validate(name: str, descriptor: int, record: Any) -> None:
        receipt = receipts.get(name)
        if receipt is None:
            return
        _require(
            record.sha256 == receipt.sha256
            and record.size_bytes == receipt.size_bytes,
            f"Stage 7 published receipt differs: {name}",
        )
        if receipt.rows is None:
            return
        duplicate = os.dup(descriptor)
        try:
            with os.fdopen(duplicate, "rb") as handle:
                parquet = pq.ParquetFile(handle)
                observed_schema = parquet.schema_arrow
                rows = parquet.metadata.num_rows
                row_groups = parquet.metadata.num_row_groups
        except Exception as exc:
            raise LabelStorePublicationError(
                f"Cannot validate Stage 7 published Parquet: {name}"
            ) from exc
        _require(
            observed_schema.equals(
                _parquet_physical_schema(schemas[name]),
                check_metadata=True,
            )
            and rows == receipt.rows
            and row_groups == receipt.row_groups
            and _schema_sha256(observed_schema) == receipt.schema_sha256,
            f"Stage 7 published Parquet metadata differs: {name}",
        )

    return validate


def _publish_staged_set(
    *,
    trusted_root: Path,
    destination: Path,
    expected_files: Sequence[str],
    identity: LabelStoreBuildIdentity,
    rules: Mapping[str, Any],
    context: _PublicationContext,
    location_kind: str,
    status: str,
    writer: Any,
) -> PublishedLabelOutputSet:
    receipts: dict[str, _PublicationFileReceipt] = {}
    schemas: dict[str, pa.Schema] = {}
    with SecureStagingDirectory.create(
        trusted_root=trusted_root,
        destination=destination,
        expected_files=expected_files,
        label=f"Stage 7 {location_kind}",
    ) as stage:
        written, written_schemas = writer(stage)
        receipts.update(written)
        schemas.update(written_schemas)
        _require(
            set(receipts) == set(expected_files) - {_SUCCESS_NAME},
            f"Stage 7 {location_kind} payload set differs",
        )
        success = _success_payload(
            identity=identity,
            rules=rules,
            location_kind=location_kind,
            status=status,
            receipts=receipts,
        )
        success_receipt = _write_bytes(
            stage=stage,
            name=_SUCCESS_NAME,
            payload=success,
        )
        hashes = {
            _SUCCESS_NAME: success_receipt.sha256,
            **{
                name: receipt.sha256
                for name, receipt in receipts.items()
            },
        }
        validator = _published_file_validator(receipts, schemas)
        snapshot = stage.snapshot(
            expected_hashes=hashes,
            file_validator=validator,
        )
        bundle = stage.publish(
            expected_hashes=snapshot.hashes,
            file_validator=validator,
        )
    return PublishedLabelOutputSet(
        path=destination,
        bundle=bundle,
        files=dict(receipts),
    )


def _publish_interim(
    *,
    inputs: LabelStoreInputs,
    identity: LabelStoreBuildIdentity,
    months: Sequence[_AuthenticatedMonth],
    context: _PublicationContext,
    rules: Mapping[str, Any],
    lineage_rows: int,
) -> PublishedLabelOutputSet:
    def writer(stage: SecureStagingDirectory):
        receipts = {
            _AUDIT_NAME: _write_parquet_batches(
                stage=stage,
                name=_AUDIT_NAME,
                schema=context.schemas[_AUDIT_NAME],
                batches=_cache_batches(
                    inputs=inputs,
                    months=months,
                    context=context,
                    cache_name=_CACHE_AUDIT_NAME,
                    output_name=_AUDIT_NAME,
                    key=("signal_date", "permno"),
                    expected_rows=context.required_audit_rows,
                    rules=rules,
                ),
                context=context,
            ),
            _LINEAGE_NAME: _write_parquet_batches(
                stage=stage,
                name=_LINEAGE_NAME,
                schema=context.schemas[_LINEAGE_NAME],
                batches=_cache_batches(
                    inputs=inputs,
                    months=months,
                    context=context,
                    cache_name=_CACHE_LINEAGE_NAME,
                    output_name=_LINEAGE_NAME,
                    key=("signal_date", "permno", "calendar_date"),
                    expected_rows=lineage_rows,
                    rules=rules,
                ),
                context=context,
            ),
        }
        return receipts, {
            _AUDIT_NAME: context.schemas[_AUDIT_NAME],
            _LINEAGE_NAME: context.schemas[_LINEAGE_NAME],
        }

    return _publish_staged_set(
        trusted_root=inputs.database_root,
        destination=context.interim_path,
        expected_files=context.interim_files,
        identity=identity,
        rules=rules,
        context=context,
        location_kind=_INTERIM_LOCATION,
        status=_INTERIM_STATUS,
        writer=writer,
    )


def _read_month_audit(
    *,
    inputs: LabelStoreInputs,
    month: _AuthenticatedMonth,
    context: _PublicationContext,
) -> pa.Table:
    with _open_month_guard(
        month,
        inputs=inputs,
        context=context,
    ) as guard:
        with guard.open_binary(_CACHE_AUDIT_NAME) as handle:
            table = pq.ParquetFile(handle).read(
                use_threads=False,
            ).combine_chunks()
        guard.gate()
    schema = context.schemas[_AUDIT_NAME]
    _require(
        table.schema.equals(schema, check_metadata=True)
        and table.num_rows == month.receipt.audit_rows,
        "Stage 7 cache audit changed while reading",
    )
    previous = _validate_stream_key(
        batch=table,
        names=("signal_date", "permno"),
        previous=None,
        label="Stage 7 month audit",
    )
    _require(previous is not None, "Stage 7 month audit is empty")
    _require(
        set(table["signal_date"].to_pylist())
        == {month.receipt.signal_date},
        "Stage 7 month audit has a different signal date",
    )
    return table


def _label_batches(
    *,
    inputs: LabelStoreInputs,
    months: Sequence[_AuthenticatedMonth],
    context: _PublicationContext,
    rules: Mapping[str, Any],
) -> Iterator[pa.RecordBatch]:
    schema = context.schemas[_LABELS_NAME]
    emitted = 0
    previous: tuple[Any, ...] | None = None
    for month in months:
        audit = _read_month_audit(
            inputs=inputs,
            month=month,
            context=context,
        )
        labels = project_development_labels(audit, rules=rules).combine_chunks()
        _require(
            labels.schema.equals(schema, check_metadata=True)
            and labels.num_rows == month.receipt.long_rows,
            "Stage 7 processed-label projection differs",
        )
        previous = _validate_stream_key(
            batch=labels,
            names=("signal_date", "permno"),
            previous=previous,
            label="Stage 7 processed labels",
        )
        emitted += labels.num_rows
        yield from labels.to_batches(max_chunksize=context.row_group_size)
    _require(
        emitted == context.required_long_rows,
        "Stage 7 processed-label row count differs",
    )


def _summary_table(
    months: Sequence[_AuthenticatedMonth],
    context: _PublicationContext,
) -> pa.Table:
    try:
        table = pa.Table.from_pylist(
            [dict(month.summary) for month in months],
            schema=context.schemas[_SUMMARY_NAME],
        ).combine_chunks()
    except (TypeError, ValueError, pa.ArrowException) as exc:
        raise LabelStorePublicationError(
            "Cannot construct Stage 7 month summaries"
        ) from exc
    _require(
        table.num_rows == context.required_signal_dates
        and table.schema.equals(
            context.schemas[_SUMMARY_NAME],
            check_metadata=True,
        ),
        "Stage 7 month-summary row count or schema differs",
    )
    _validate_stream_key(
        batch=table,
        names=("signal_date",),
        previous=None,
        label="Stage 7 month summaries",
    )
    _require(
        table["signal_date"].to_pylist()
        == [month.receipt.signal_date for month in months],
        "Stage 7 month-summary key domain differs",
    )
    return table


def _validate_small_table(
    table: pa.Table,
    *,
    schema: pa.Schema,
    key: Sequence[str],
    label: str,
    expected_rows: int | None = None,
) -> pa.Table:
    table = table.combine_chunks()
    _require(
        table.schema.equals(schema, check_metadata=True),
        f"{label} schema differs",
    )
    if expected_rows is not None:
        _require(table.num_rows == expected_rows, f"{label} row count differs")
    for field in schema:
        if not field.nullable:
            _require(
                table[field.name].null_count == 0,
                f"{label} has nulls in {field.name}",
            )
        if pa.types.is_floating(field.type):
            for value in table[field.name].to_pylist():
                _require(
                    value is None or math.isfinite(float(value)),
                    f"{label} has a nonfinite value",
                )
    _validate_stream_key(
        batch=table,
        names=key,
        previous=None,
        label=label,
    )
    return table


def _publish_processed(
    *,
    inputs: LabelStoreInputs,
    identity: LabelStoreBuildIdentity,
    months: Sequence[_AuthenticatedMonth],
    context: _PublicationContext,
    rules: Mapping[str, Any],
) -> tuple[PublishedLabelOutputSet, pa.Table, pa.Table]:
    summaries = _summary_table(months, context)
    splits = _validate_small_table(
        build_split_membership(summaries, rules),
        schema=context.schemas[_SPLIT_NAME],
        key=("fold_id", "signal_date"),
        label="Stage 7 split membership",
        expected_rows=context.required_split_rows,
    )
    _require(
        set(splits["signal_date"].to_pylist())
        == set(summaries["signal_date"].to_pylist()),
        "Stage 7 split-membership signal domain differs",
    )

    def writer(stage: SecureStagingDirectory):
        receipts = {
            _LABELS_NAME: _write_parquet_batches(
                stage=stage,
                name=_LABELS_NAME,
                schema=context.schemas[_LABELS_NAME],
                batches=_label_batches(
                    inputs=inputs,
                    months=months,
                    context=context,
                    rules=rules,
                ),
                context=context,
            ),
            _SUMMARY_NAME: _write_parquet_table(
                stage=stage,
                name=_SUMMARY_NAME,
                table=summaries,
                context=context,
            ),
            _SPLIT_NAME: _write_parquet_table(
                stage=stage,
                name=_SPLIT_NAME,
                table=splits,
                context=context,
            ),
        }
        return receipts, {
            _LABELS_NAME: context.schemas[_LABELS_NAME],
            _SUMMARY_NAME: summaries.schema,
            _SPLIT_NAME: splits.schema,
        }

    published = _publish_staged_set(
        trusted_root=inputs.database_root,
        destination=context.processed_path,
        expected_files=context.processed_files,
        identity=identity,
        rules=rules,
        context=context,
        location_kind=_PROCESSED_LOCATION,
        status=_PROCESSED_STATUS,
        writer=writer,
    )
    return published, summaries, splits


_COUNT_GATE_NAMES = frozenset(
    {
        "audit_rows",
        "processed_rows",
        "month_summary_rows",
        "split_membership_rows",
    }
)
_STRUCTURAL_ZERO_GATE_NAMES = frozenset(
    {
        "duplicate_audit_keys",
        "audit_key_order_mismatches",
        "duplicate_processed_keys",
        "processed_key_domain_mismatches",
        "processed_key_order_mismatches",
        "duplicate_leg_lineage_keys",
        "leg_lineage_key_domain_mismatches",
        "leg_lineage_sort_mismatches",
        "duplicate_month_summary_keys",
        "month_summary_key_domain_mismatches",
        "month_summary_sort_mismatches",
        "duplicate_split_membership_keys",
        "split_membership_key_domain_mismatches",
        "split_membership_sort_mismatches",
        "duplicate_coverage_keys",
        "coverage_key_domain_mismatches",
        "coverage_sort_mismatches",
        "duplicate_status_count_keys",
        "status_count_key_domain_mismatches",
        "status_count_sort_mismatches",
        "duplicate_timing_check_keys",
        "timing_check_key_domain_mismatches",
        "timing_check_sort_mismatches",
        "duplicate_distribution_keys",
        "distribution_key_domain_mismatches",
        "distribution_sort_mismatches",
        "unknown_statuses",
        "nonfinite_published_values",
    }
)
_CONSTRUCTION_ZERO_GATE_NAMES = frozenset(
    {
        "spine_key_or_audit_field_mismatches",
        "nonlong_nonnull_target_values",
        "short_eligibility_influence_violations",
        "beta_value_mismatches",
        "post_cutoff_return_rows_used",
        "post_cutoff_factor_rows_used",
        "signal_to_entry_legs_used",
        "after_exit_legs_used",
        "duplicate_or_overlapping_legs_used",
        "multi_session_or_straddling_legs_used",
        "unusable_return_legs_used",
        "unresolved_delisting_legs_used",
        "terminal_delisting_double_counts",
        "missing_factor_legs_used",
        "factor_date_mismatches",
        "residual_value_mismatches",
        "excess_value_mismatches",
        "rank_population_mismatches",
        "rank_formula_mismatches",
        "nonzero_rank_numerator_month_sums",
        "excess_primary_mask_mismatches",
        "mad_value_mismatches",
        "security_availability_mismatches",
        "month_availability_mismatches",
        "split_or_embargo_mismatches",
    }
)


def _observed_gates(
    *,
    context: _PublicationContext,
    rules: Mapping[str, Any],
) -> dict[str, int]:
    """Bind every hard gate to an explicit evidence class.

    Structural gates are accepted only after the exact schema/key/domain
    validators in this publisher have returned.  Construction-semantic gates
    are accepted from the sealed, typed materializer result whose producing
    code is part of the content-addressed build identity; Stage 7.4 must still
    independently reconstruct them before downstream authorization.
    """

    expected = {
        str(name): int(value)
        for name, value in rules["quality_gates"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    classified = (
        _COUNT_GATE_NAMES
        | _STRUCTURAL_ZERO_GATE_NAMES
        | _CONSTRUCTION_ZERO_GATE_NAMES
    )
    _require(
        set(expected) == classified
        and not (
            (_COUNT_GATE_NAMES & _STRUCTURAL_ZERO_GATE_NAMES)
            or (_COUNT_GATE_NAMES & _CONSTRUCTION_ZERO_GATE_NAMES)
            or (
                _STRUCTURAL_ZERO_GATE_NAMES
                & _CONSTRUCTION_ZERO_GATE_NAMES
            )
        ),
        "Stage 7 hard-gate evidence classification differs",
    )
    observed = {name: 0 for name in classified}
    observed.update(
        {
            "audit_rows": context.required_audit_rows,
            "processed_rows": context.required_long_rows,
            "month_summary_rows": context.required_signal_dates,
            "split_membership_rows": context.required_split_rows,
        }
    )
    _require(
        all(observed[name] == expected[name] for name in expected),
        "Stage 7 primary hard gates do not pass",
    )
    return observed


def _quality_tables(
    *,
    inputs: LabelStoreInputs,
    months: Sequence[_AuthenticatedMonth],
    context: _PublicationContext,
    rules: Mapping[str, Any],
) -> tuple[pa.Table, pa.Table, pa.Table]:
    coverage_parts: list[pa.Table] = []
    status_parts: list[pa.Table] = []
    distribution_parts: list[pa.Table] = []
    for month in months:
        audit = _read_month_audit(
            inputs=inputs,
            month=month,
            context=context,
        )
        quality = build_month_label_quality(
            audit,
            bool(month.summary["primary_month_valid"]),
            rules,
        )
        coverage_parts.append(quality.label_coverage)
        status_parts.append(quality.label_status_counts)
        distribution_parts.append(quality.label_distribution)
    coverage = _validate_small_table(
        pa.concat_tables(coverage_parts),
        schema=context.schemas[_COVERAGE_NAME],
        key=("signal_date",),
        label="Stage 7 label coverage",
        expected_rows=context.required_signal_dates,
    )
    expected_status_rows = context.required_signal_dates * sum(
        len(rules["value_domains"][name])
        for name in (
            "outcome_status",
            "primary_label_status",
            "magnitude_label_status",
        )
    )
    status_counts = _validate_small_table(
        pa.concat_tables(status_parts),
        schema=context.schemas[_STATUS_NAME],
        key=("signal_date", "status_dimension", "status"),
        label="Stage 7 label status counts",
        expected_rows=expected_status_rows,
    )
    expected_distribution_rows = (
        context.required_signal_dates
        * len(rules["value_domains"]["distribution_target_name"])
    )
    distribution = _validate_small_table(
        pa.concat_tables(distribution_parts),
        schema=context.schemas[_DISTRIBUTION_NAME],
        key=("signal_date", "target_name"),
        label="Stage 7 label distribution",
        expected_rows=expected_distribution_rows,
    )
    signal_domain = {month.receipt.signal_date for month in months}
    _require(
        set(coverage["signal_date"].to_pylist()) == signal_domain
        and set(status_counts["signal_date"].to_pylist()) == signal_domain
        and set(distribution["signal_date"].to_pylist()) == signal_domain,
        "Stage 7 quality signal-date domain differs",
    )
    return coverage, status_counts, distribution


def _input_builds(policy: Mapping[str, Any]) -> dict[str, str]:
    inputs = policy["inputs"]
    return {
        "auxiliary": str(inputs["auxiliary_build_id"]),
        "calendar": str(inputs["calendar_build_id"]),
        "feature_store": str(inputs["feature_store_build_id"]),
        "feature_store_verification": str(
            inputs["feature_store_verification_id"]
        ),
        "return_reconciliation": str(inputs["return_build_id"]),
        "signal_universe": str(inputs["signal_universe_build_id"]),
    }


def _publish_quality(
    *,
    inputs: LabelStoreInputs,
    identity: LabelStoreBuildIdentity,
    months: Sequence[_AuthenticatedMonth],
    context: _PublicationContext,
    rules: Mapping[str, Any],
    lineage_rows: int,
) -> tuple[PublishedLabelOutputSet, pa.Table, Mapping[str, Any]]:
    coverage, status_counts, distribution = _quality_tables(
        inputs=inputs,
        months=months,
        context=context,
        rules=rules,
    )
    observed = _observed_gates(context=context, rules=rules)
    timing = _validate_small_table(
        build_label_timing_checks(observed, rules),
        schema=context.schemas[_TIMING_NAME],
        key=("check_name",),
        label="Stage 7 label timing checks",
        expected_rows=len(observed),
    )
    _require(
        bool(pc.all(timing["passed"]).as_py()),
        "Stage 7 label timing checks did not all pass",
    )
    counts = {
        "audit_rows": context.required_audit_rows,
        "coverage_rows": coverage.num_rows,
        "distribution_rows": distribution.num_rows,
        "leg_lineage_rows": lineage_rows,
        "month_summary_rows": context.required_signal_dates,
        "processed_rows": context.required_long_rows,
        "signal_dates": context.required_signal_dates,
        "split_membership_rows": context.required_split_rows,
        "status_count_rows": status_counts.num_rows,
        "timing_check_rows": timing.num_rows,
    }
    quality_summary = build_quality_summary(
        identity.build_id,
        identity.contract_id,
        identity.policy_id,
        observed,
        counts,
        _input_builds(context.policy),
        rules,
    )
    _require(
        quality_summary["status"] == _PRIMARY_STATUS,
        "Stage 7 quality summary did not pass",
    )
    quality_payload = canonical_json_bytes(quality_summary)
    tables = {
        _COVERAGE_NAME: coverage,
        _STATUS_NAME: status_counts,
        _TIMING_NAME: timing,
        _DISTRIBUTION_NAME: distribution,
    }

    def writer(stage: SecureStagingDirectory):
        receipts = {
            name: _write_parquet_table(
                stage=stage,
                name=name,
                table=table,
                context=context,
            )
            for name, table in tables.items()
        }
        receipts[_QUALITY_SUMMARY_NAME] = _write_bytes(
            stage=stage,
            name=_QUALITY_SUMMARY_NAME,
            payload=quality_payload,
        )
        return receipts, {
            name: table.schema for name, table in tables.items()
        }

    published = _publish_staged_set(
        trusted_root=inputs.database_root,
        destination=context.quality_path,
        expected_files=context.quality_files,
        identity=identity,
        rules=rules,
        context=context,
        location_kind=_QUALITY_LOCATION,
        status=_QUALITY_STATUS,
        writer=writer,
    )
    return published, timing, quality_summary


def _input_stat_snapshot(
    inputs: LabelStoreInputs,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    result: list[tuple[str, tuple[int, ...]]] = []
    for name in sorted(inputs.tables):
        table = inputs.tables[name]
        path = getattr(table, "_path", None)
        if path is None:
            path = getattr(table, "path", None)
        _require(path is not None, f"Stage 7 input path is absent: {name}")
        observed = os.stat(path, follow_symlinks=False)
        _require(
            stat.S_ISREG(observed.st_mode)
            and not stat.S_ISLNK(observed.st_mode),
            f"Stage 7 input is not a regular file: {name}",
        )
        result.append(
            (
                name,
                (
                    observed.st_dev,
                    observed.st_ino,
                    observed.st_mode,
                    observed.st_nlink,
                    observed.st_size,
                    observed.st_mtime_ns,
                    observed.st_ctime_ns,
                ),
            )
        )
    return tuple(result)


def _binding_hash_gate(
    identity: LabelStoreBuildIdentity,
    *,
    project_root: Path,
) -> None:
    for label, expected in (
        *identity.configuration_hashes,
        *identity.code_hashes,
    ):
        candidate = Path(label)
        path = candidate if candidate.is_absolute() else project_root / candidate
        _require(
            path.is_file()
            and not path.is_symlink()
            and sha256_file(path) == expected,
            f"Stage 7 publication binding changed: {label}",
        )


def _relative_database_path(path: Path, inputs: LabelStoreInputs) -> str:
    try:
        return Path(os.path.abspath(path)).relative_to(
            Path(os.path.abspath(inputs.database_root))
        ).as_posix()
    except ValueError as exc:
        raise LabelStorePublicationError(
            "Stage 7 published output escaped the database root"
        ) from exc


def _manifest_output(
    output: PublishedLabelOutputSet,
    inputs: LabelStoreInputs,
) -> dict[str, Any]:
    files = {
        name: output.files[name].as_dict()
        for name in sorted(output.files)
    }
    success_record = next(
        record
        for record in output.bundle.snapshot.files
        if record.name == _SUCCESS_NAME
    )
    files[_SUCCESS_NAME] = _PublicationFileReceipt(
        path=_SUCCESS_NAME,
        sha256=success_record.sha256,
        size_bytes=success_record.size_bytes,
        rows=None,
        row_groups=None,
        schema_sha256=None,
    ).as_dict()
    return {
        "path": _relative_database_path(output.path, inputs),
        "success_sha256": output.success_sha256,
        "files": {
            name: files[name] for name in sorted(files)
        },
    }


def _authorizations() -> dict[str, bool]:
    """Return a primary authorization object with only verification open."""

    return {
        "independent_verification_required": True,
        "stage_08_risk_and_cost_models_authorized": False,
        "stage_09_model_training_authorized": False,
        "models_portfolios_pnl_or_backtests_authorized": False,
        "validation_or_final_test_access_authorized": False,
    }


def _manifest(
    *,
    inputs: LabelStoreInputs,
    identity: LabelStoreBuildIdentity,
    context: _PublicationContext,
    rules: Mapping[str, Any],
    interim: PublishedLabelOutputSet,
    processed: PublishedLabelOutputSet,
    quality: PublishedLabelOutputSet,
    lineage_rows: int,
    timing: pa.Table,
) -> dict[str, Any]:
    hard_gate_failures = sum(
        not bool(value) for value in timing["passed"].to_pylist()
    )
    result = {
        "schema_version": 1,
        "contract_id": identity.contract_id,
        "policy_id": identity.policy_id,
        "build_id": identity.build_id,
        "status": _PRIMARY_STATUS,
        "bindings": _bindings(identity),
        "inputs": {
            "input_builds": _input_builds(context.policy),
            "table_receipts": [
                item.as_dict()
                for item in sorted(
                    identity.input_receipts,
                    key=lambda item: item.name,
                )
            ],
        },
        "outputs": {
            "interim": _manifest_output(interim, inputs),
            "processed": _manifest_output(processed, inputs),
            "quality": _manifest_output(quality, inputs),
        },
        "summary": {
            "audit_rows": context.required_audit_rows,
            "hard_gate_failures": hard_gate_failures,
            "leg_lineage_rows": lineage_rows,
            "processed_rows": context.required_long_rows,
            "signal_dates": context.required_signal_dates,
            "split_membership_rows": context.required_split_rows,
        },
        "authorizations": _authorizations(),
    }
    required = rules["publication_payloads"][
        "primary_manifest_required_keys"
    ]
    _require(
        set(result) == set(required)
        and hard_gate_failures == 0,
        "Stage 7 primary manifest differs or has failed gates",
    )
    return result


def _publish_manifest(
    *,
    project_root: Path,
    context: _PublicationContext,
    manifest: Mapping[str, Any],
    gate: Any,
) -> PublishedBundle:
    payload = canonical_json_bytes(manifest)
    with SecureStagingDirectory.create(
        trusted_root=project_root,
        destination=context.manifest_directory,
        expected_files=(_MANIFEST_NAME,),
        label="Stage 7 primary manifest",
    ) as stage:
        with stage.open_exclusive_binary(_MANIFEST_NAME) as stream:
            stream.write(payload)
        snapshot = stage.snapshot(
            expected_hashes={_MANIFEST_NAME: _sha256_bytes(payload)}
        )
        return stage.publish(
            expected_hashes=snapshot.hashes,
            pre_publish_gate=gate,
            post_publish_gate=gate,
        )


def publish_primary_label_store(
    inputs: LabelStoreInputs,
    cache_result: LabelStoreCacheResult,
    rules: Mapping[str, Any],
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
) -> PrimaryLabelStorePublication:
    """Publish the immutable primary Stage 7 label store and pointer last."""

    policy_path = policy_path.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    _require(
        project_root.is_dir() and not project_root.is_symlink(),
        "Stage 7 project root must be a real directory",
    )
    _require(
        isinstance(cache_result, LabelStoreCacheResult),
        "Stage 7 publication requires LabelStoreCacheResult",
    )
    identity = cache_result.identity
    context = _context(
        inputs=inputs,
        identity=identity,
        rules=rules,
        policy_path=policy_path,
        project_root=project_root,
    )

    # This is the full after-materialization identity reauthentication.  The
    # cache result's Boolean seal alone is deliberately not trusted.
    before_reauthentication = _input_stat_snapshot(inputs)
    reauthenticate_label_store_build_identity(
        inputs,
        identity,
        policy_path=policy_path,
        project_root=project_root,
    )
    after_reauthentication = _input_stat_snapshot(inputs)
    _require(
        before_reauthentication == after_reauthentication,
        "Stage 7 inputs changed during after-materialization reauthentication",
    )
    months = _authenticate_cache(
        inputs=inputs,
        cache_result=cache_result,
        context=context,
    )
    interim = _publish_interim(
        inputs=inputs,
        identity=identity,
        months=months,
        context=context,
        rules=rules,
        lineage_rows=cache_result.lineage_rows,
    )
    processed, _summaries, _splits = _publish_processed(
        inputs=inputs,
        identity=identity,
        months=months,
        context=context,
        rules=rules,
    )
    quality, timing, _quality_summary = _publish_quality(
        inputs=inputs,
        identity=identity,
        months=months,
        context=context,
        rules=rules,
        lineage_rows=cache_result.lineage_rows,
    )

    manifest = _manifest(
        inputs=inputs,
        identity=identity,
        context=context,
        rules=rules,
        interim=interim,
        processed=processed,
        quality=quality,
        lineage_rows=cache_result.lineage_rows,
        timing=timing,
    )
    outputs = {
        "interim": interim,
        "processed": processed,
        "quality": quality,
    }
    expected_sets = {
        "interim": context.interim_files,
        "processed": context.processed_files,
        "quality": context.quality_files,
    }

    with contextlib.ExitStack() as guards:
        data_guards = {
            name: guards.enter_context(
                AuthenticatedBundleGuard.open(
                    trusted_root=inputs.database_root,
                    path=outputs[name].path,
                    expected_files=expected_sets[name],
                    expected_hashes=outputs[name].bundle.snapshot.hashes,
                    label=f"Stage 7 published {name} bundle",
                    _permit_authenticated_directory_ctime_refresh=True,
                )
            )
            for name in ("interim", "processed", "quality")
        }

        def data_gate() -> None:
            for guard in data_guards.values():
                guard.gate()
            _require(
                _input_stat_snapshot(inputs) == after_reauthentication,
                "Stage 7 inputs changed after final full reauthentication",
            )
            _binding_hash_gate(identity, project_root=project_root)

        artifact_bundle = _publish_manifest(
            project_root=project_root,
            context=context,
            manifest=manifest,
            gate=data_gate,
        )
        manifest_path = context.manifest_directory / _MANIFEST_NAME
        manifest_guard = guards.enter_context(
            AuthenticatedBundleGuard.open(
                trusted_root=project_root,
                path=context.manifest_directory,
                expected_files=(_MANIFEST_NAME,),
                expected_hashes=artifact_bundle.snapshot.hashes,
                label="Stage 7 published primary manifest",
                _permit_authenticated_directory_ctime_refresh=True,
            )
        )
        pointer = {
            "schema_version": 1,
            "contract_id": identity.contract_id,
            "policy_id": identity.policy_id,
            "build_id": identity.build_id,
            "status": _PRIMARY_STATUS,
            "primary_manifest_sha256": artifact_bundle.snapshot.hashes[
                _MANIFEST_NAME
            ],
            "authorizations": _authorizations(),
        }
        required_pointer_keys = rules["publication_payloads"][
            "primary_pointer_required_keys"
        ]
        _require(
            set(pointer) == set(required_pointer_keys)
            and sum(pointer["authorizations"].values()) == 1
            and pointer["authorizations"][
                "independent_verification_required"
            ]
            is True,
            "Stage 7 primary pointer authorization differs",
        )

        def pointer_data_gate() -> None:
            data_gate()
            manifest_guard.gate()
            _require(
                sha256_file(manifest_path)
                == pointer["primary_manifest_sha256"],
                "Stage 7 primary manifest changed before pointer publication",
            )

        def pre_pointer_gate() -> None:
            pointer_data_gate()
            # The authority pointer is the final publication action.  Perform
            # the required full identity reauthentication immediately before
            # its atomic replacement, not merely earlier in assembly.
            before = _input_stat_snapshot(inputs)
            reauthenticate_label_store_build_identity(
                inputs,
                identity,
                policy_path=policy_path,
                project_root=project_root,
            )
            after = _input_stat_snapshot(inputs)
            _require(
                before == after == after_reauthentication,
                "Stage 7 inputs changed before primary pointer publication",
            )

        pointer_replaced = replace_pointer_last(
            context.pointer_path,
            canonical_json_bytes(pointer),
            trusted_root=project_root,
            pre_replace_gate=pre_pointer_gate,
            post_replace_gate=pointer_data_gate,
            label="Stage 7 latest primary pointer",
        )
    reused = (
        interim.bundle.reused_existing
        and processed.bundle.reused_existing
        and quality.bundle.reused_existing
        and artifact_bundle.reused_existing
        and not pointer_replaced
    )
    return PrimaryLabelStorePublication(
        build_id=identity.build_id,
        interim_path=context.interim_path,
        processed_path=context.processed_path,
        quality_path=context.quality_path,
        manifest_path=context.manifest_directory / _MANIFEST_NAME,
        pointer_path=context.pointer_path,
        manifest=manifest,
        pointer=pointer,
        reused_existing=reused,
    )
