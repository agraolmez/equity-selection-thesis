"""Validation and manifest scaffold for prebuilt Stage 11 predictor data.

The module never scans the raw vendor files and never constructs a label,
outcome, portfolio, trade, or P&L row.  A separate scope-aware historical
market materializer must first create the twelve candidate Parquet payloads.
This module then enforces their point-in-time boundary and prepares the
content-addressed manifest that an immutable publisher can consume.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.features.feature_lineage import VALUE_STATUS_SET
from portfolio_bot.risk_cost.contract import FACTOR_NAMES
from portfolio_bot.validation.validation_build_code_identity import (
    MATERIALIZATION_CODE_PATHS,
    materialization_code_hashes,
)
from portfolio_bot.validation.validation_data_contract import (
    DEFAULT_POLICY,
    FIRST_PROHIBITED_SIGNAL,
    INTERNAL_CONTEXT_SIGNAL,
    PROJECT_ROOT,
    REQUIRED_FACTOR_COVARIANCE_ROWS,
    REQUIRED_SIGNAL_COUNT,
    ValidationDataContract,
    canonical_json_bytes,
    require,
    schema_sha256,
    sha256_file,
    unexpected_forbidden_columns,
    validate_validation_data_contract,
)
from portfolio_bot.validation.validation_development_parity import (
    DevelopmentParityReceipt,
    PARITY_METHOD,
    PARITY_STATUS,
    ValidatedDevelopmentParity,
)
from portfolio_bot.validation.validation_feature_semantics import (
    audit_raw_lineage_semantics,
)
import portfolio_bot.validation.validation_empirical_feature_stream as empirical_stream
from portfolio_bot.validation.validation_empirical_feature_stream import (
    EMPIRICAL_BUILD_STATUS,
    ValidatedEmpiricalCandidateBuild,
)


class ValidationDataMaterializationError(RuntimeError):
    """Raised when a candidate validation-data bundle fails closed."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationDataMaterializationError(message)


@dataclass(frozen=True)
class FileReceipt:
    role: str
    path: str
    sha256: str
    size_bytes: int
    rows: int
    row_groups: int
    schema_sha256: str
    minimum_signal_date: str
    maximum_signal_date: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "row_groups": self.row_groups,
            "schema_sha256": self.schema_sha256,
            "minimum_signal_date": self.minimum_signal_date,
            "maximum_signal_date": self.maximum_signal_date,
        }


@dataclass(frozen=True)
class FeatureAudit:
    rows: int
    long_eligible_rows: int
    inactive_segment_nonmissing_values: int
    signal_dates: tuple[dt.date, ...]
    key_sha256: str
    long_eligible_key_sha256: str


@dataclass(frozen=True)
class LineageAudit:
    family: str
    rows: int
    feature_count: int
    key_sha256: str
    timing_violations: int


@dataclass(frozen=True)
class CandidateValidationResult:
    contract: ValidationDataContract
    root: Path
    receipts: Mapping[str, FileReceipt]
    feature_audit: FeatureAudit
    lineage_audits: Mapping[str, LineageAudit]
    risk_audit_rows: int
    risk_input_rows: int
    factor_covariance_rows: int
    month_summary_rows: int
    manifest: Mapping[str, Any]


def materialization_plan(
    contract: ValidationDataContract,
) -> Mapping[str, Any]:
    """Return the deterministic plan; this function opens no market payload."""

    return MappingProxyType(
        {
            "status": "VALIDATION_DATA_MATERIALIZATION_REQUIRED",
            "contract_id": str(contract.policy["contract_id"]),
            "policy_sha256": contract.policy_sha256,
            "internal_context_signal_date": INTERNAL_CONTEXT_SIGNAL.isoformat(),
            "context_is_publishable": False,
            "scored_signal_count": len(contract.validation_signals),
            "first_scored_signal": contract.scored_signal_dates[0].isoformat(),
            "last_scored_signal": contract.scored_signal_dates[-1].isoformat(),
            "economic_feature_count": len(contract.feature_names),
            "physical_feature_column_count": len(contract.raw_feature_schema),
            "payload_roles": [
                {
                    "role": value.role,
                    "filename": value.filename,
                    "schema_sha256": value.schema_sha256,
                    "cardinality_rule": value.cardinality_rule,
                }
                for value in contract.payload_specs.values()
            ],
            "development_parity_required": True,
            "validation_outcome_rows_read": 0,
            "final_period_rows_read": 0,
            "raw_materialization_executed_by_this_call": False,
        }
    )


def _safe_candidate_root(root: Path) -> Path:
    lexical = Path(os.path.abspath(root.expanduser()))
    _check(lexical.is_dir() and not lexical.is_symlink(), "Unsafe candidate root")
    _check(lexical.resolve(strict=True) == lexical, "Candidate root uses a symlinked path")
    return lexical


def _safe_payload(root: Path, filename: str) -> Path:
    _check(Path(filename).name == filename, f"Unsafe candidate filename: {filename}")
    path = root / filename
    _check(path.is_file() and not path.is_symlink(), f"Missing or unsafe candidate payload: {filename}")
    _check(path.resolve(strict=True).parent == root, f"Candidate payload escaped root: {filename}")
    return path


def _stable_identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _file_receipt(
    role: str,
    path: Path,
    expected_schema: pa.Schema,
    *,
    minimum_signal_date: dt.date,
    maximum_signal_date: dt.date,
) -> FileReceipt:
    before = _stable_identity(path)
    parquet = pq.ParquetFile(path)
    actual_schema = parquet.schema_arrow
    _check(
        actual_schema.equals(expected_schema, check_metadata=True),
        f"Candidate schema differs for role {role}",
    )
    digest = sha256_file(path)
    after = _stable_identity(path)
    _check(before == after, f"Candidate payload changed while authenticated: {role}")
    return FileReceipt(
        role=role,
        path=path.name,
        sha256=digest,
        size_bytes=int(before[2]),
        rows=int(parquet.metadata.num_rows),
        row_groups=int(parquet.metadata.num_row_groups),
        schema_sha256=schema_sha256(actual_schema),
        minimum_signal_date=minimum_signal_date.isoformat(),
        maximum_signal_date=maximum_signal_date.isoformat(),
    )


def _date(value: object, label: str) -> dt.date:
    _check(
        isinstance(value, dt.date) and not isinstance(value, dt.datetime),
        f"Invalid {label}",
    )
    return value


def _integer(value: object, label: str) -> int:
    _check(isinstance(value, int) and not isinstance(value, bool), f"Invalid {label}")
    return int(value)


def _update_key(digest: Any, signal_date: dt.date, permno: int) -> None:
    digest.update(struct.pack(">Iq", signal_date.toordinal(), permno))


def _rows(path: Path, columns: Sequence[str], *, batch_size: int = 131_072) -> Iterator[Mapping[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=list(columns),
        use_threads=False,
    ):
        for row in batch.to_pylist():
            yield row


def _audit_feature_payload(
    path: Path,
    contract: ValidationDataContract,
) -> FeatureAudit:
    parquet = pq.ParquetFile(path)
    actual_columns = tuple(parquet.schema_arrow.names)
    expected_columns = tuple(contract.raw_feature_schema.names)
    forbidden = unexpected_forbidden_columns(actual_columns, allowed_columns=expected_columns)
    _check(not forbidden, f"Forbidden predictor columns: {list(forbidden)}")
    _check(actual_columns == expected_columns, "Validation predictor column order differs")

    expected_dates = frozenset(contract.scored_signal_dates)
    observed_dates: set[dt.date] = set()
    key_digest = hashlib.sha256()
    long_digest = hashlib.sha256()
    previous: tuple[dt.date, int] | None = None
    rows = 0
    long_rows = 0
    segment_names = (
        "business_segment_count",
        "geographic_segment_count",
        "business_segment_revenue_hhi",
        "business_segment_revenue_hhi_change_1y",
    )
    segment_nonmissing = 0
    for row in _rows(
        path,
        ("signal_date", "permno", "long_eligible", *segment_names),
    ):
        signal = _date(row["signal_date"], "feature signal date")
        permno = _integer(row["permno"], "feature permno")
        _check(signal in expected_dates, f"Feature row escaped exact validation signals: {signal}")
        _check(signal != INTERNAL_CONTEXT_SIGNAL, "Internal context signal was published")
        _check(signal < FIRST_PROHIBITED_SIGNAL, "Embargo or final feature row was published")
        key = (signal, permno)
        _check(previous is None or key > previous, "Feature keys are duplicate or unsorted")
        previous = key
        observed_dates.add(signal)
        _update_key(key_digest, signal, permno)
        rows += 1
        eligible = row["long_eligible"]
        _check(isinstance(eligible, bool), "Null or invalid long eligibility")
        if eligible:
            _update_key(long_digest, signal, permno)
            long_rows += 1
        segment_nonmissing += sum(row[name] is not None for name in segment_names)
    _check(rows > 0, "Validation predictor payload is empty")
    _check(observed_dates == expected_dates, "Validation predictor signal set is not exactly 59 months")
    _check(
        segment_nonmissing == 0,
        "Frozen-inactive segment features were populated in validation",
    )
    return FeatureAudit(
        rows=rows,
        long_eligible_rows=long_rows,
        inactive_segment_nonmissing_values=segment_nonmissing,
        signal_dates=tuple(sorted(observed_dates)),
        key_sha256=key_digest.hexdigest(),
        long_eligible_key_sha256=long_digest.hexdigest(),
    )


def _audit_lineage_payload(
    path: Path,
    *,
    feature_path: Path,
    family: str,
    feature_names: tuple[str, ...],
    feature_audit: FeatureAudit,
    contract: ValidationDataContract,
) -> LineageAudit:
    expected_rows = feature_audit.rows * len(feature_names)
    _check(
        pq.ParquetFile(path).metadata.num_rows == expected_rows,
        f"Lineage row count differs for {family}",
    )
    expected_dates = frozenset(contract.scored_signal_dates)
    digest = hashlib.sha256()
    group_key: tuple[dt.date, int] | None = None
    ordinal = 0
    groups = 0
    rows = 0
    timing_violations = 0
    for row in _rows(path, contract.lineage_schema.names):
        signal = _date(row["signal_date"], "lineage signal date")
        permno = _integer(row["permno"], "lineage permno")
        _check(signal in expected_dates, f"Lineage row escaped validation scope: {family}")
        key = (signal, permno)
        if key != group_key:
            if group_key is not None:
                _check(ordinal == len(feature_names), f"Incomplete lineage feature group: {family}")
            _check(group_key is None or key > group_key, f"Lineage keys are duplicate or unsorted: {family}")
            group_key = key
            ordinal = 0
            groups += 1
            _update_key(digest, signal, permno)
        _check(ordinal < len(feature_names), f"Too many lineage rows per key: {family}")
        _check(row["feature_name"] == feature_names[ordinal], f"Lineage feature order differs: {family}")
        status = row["value_status"]
        _check(status in VALUE_STATUS_SET, f"Unknown lineage status: {status}")
        observation = row["max_observation_date"]
        availability = row["max_public_availability_date"]
        stale_after = row["stale_after_date"]
        if status == "PRESENT":
            if (
                not isinstance(observation, dt.date)
                or not isinstance(availability, dt.date)
                or observation > signal
                or availability > signal
                or (stale_after is not None and signal > stale_after)
            ):
                timing_violations += 1
        elif any(value is not None for value in (observation, availability, stale_after)):
            timing_violations += 1
        ordinal += 1
        rows += 1
    if group_key is not None:
        _check(ordinal == len(feature_names), f"Incomplete final lineage group: {family}")
    _check(groups == feature_audit.rows, f"Lineage key coverage differs: {family}")
    _check(digest.hexdigest() == feature_audit.key_sha256, f"Lineage keys differ from feature keys: {family}")
    _check(timing_violations == 0, f"Point-in-time lineage violations: {family}")
    semantic = audit_raw_lineage_semantics(
        raw_path=feature_path,
        lineage_path=path,
        feature_names=feature_names,
        allowed_signal_dates=contract.scored_signal_dates,
    )
    _check(
        semantic.raw_rows == feature_audit.rows
        and semantic.lineage_rows == expected_rows
        and semantic.difference_count == 0,
        f"Raw-value/lineage semantic consistency differs: {family}",
    )
    return LineageAudit(family, rows, len(feature_names), digest.hexdigest(), timing_violations)


def _audit_risk_table(
    path: Path,
    *,
    feature_audit: FeatureAudit,
    input_only: bool,
    contract: ValidationDataContract,
) -> int:
    expected_dates = frozenset(contract.scored_signal_dates)
    digest = hashlib.sha256()
    previous: tuple[dt.date, int] | None = None
    rows = 0
    columns = (
        "signal_date",
        "permno",
        "information_cutoff_date",
        "post_cutoff_return_rows_used",
        "post_cutoff_factor_rows_used",
    )
    for row in _rows(path, columns):
        signal = _date(row["signal_date"], "risk/cost signal date")
        permno = _integer(row["permno"], "risk/cost permno")
        cutoff = _date(row["information_cutoff_date"], "risk/cost cutoff date")
        _check(signal in expected_dates, "Risk/cost row escaped exact validation signals")
        _check(cutoff <= signal, "Risk/cost information cutoff exceeds signal date")
        _check(row["post_cutoff_return_rows_used"] == 0, "Post-cutoff return row was used")
        _check(row["post_cutoff_factor_rows_used"] == 0, "Post-cutoff factor row was used")
        key = (signal, permno)
        _check(previous is None or key > previous, "Risk/cost keys are duplicate or unsorted")
        previous = key
        _update_key(digest, signal, permno)
        rows += 1
    expected_rows = feature_audit.long_eligible_rows if input_only else feature_audit.rows
    expected_digest = (
        feature_audit.long_eligible_key_sha256 if input_only else feature_audit.key_sha256
    )
    _check(rows == expected_rows, "Risk/cost row count differs from its feature population")
    _check(digest.hexdigest() == expected_digest, "Risk/cost keys differ from their feature population")
    return rows


def _audit_factor_covariance(
    path: Path,
    contract: ValidationDataContract,
) -> int:
    expected_dates = contract.scored_signal_dates
    expected_pairs = tuple(
        (row_name, column_name)
        for row_name in FACTOR_NAMES
        for column_name in FACTOR_NAMES
    )
    rows = 0
    cursor = 0
    current_signal: dt.date | None = None
    observed_signals: list[dt.date] = []
    for row in _rows(
        path,
        (
            "signal_date",
            "factor_row",
            "factor_col",
            "factor_row_ordinal",
            "factor_col_ordinal",
            "factor_window_last_date",
        ),
    ):
        signal = _date(row["signal_date"], "factor covariance signal date")
        if signal != current_signal:
            _check(cursor in (0, len(expected_pairs)), "Incomplete covariance matrix")
            _check(not observed_signals or signal > observed_signals[-1], "Covariance signals are unsorted")
            observed_signals.append(signal)
            current_signal = signal
            cursor = 0
        _check(cursor < len(expected_pairs), "Too many covariance cells for a signal")
        expected_row, expected_column = expected_pairs[cursor]
        _check(
            row["factor_row"] == expected_row and row["factor_col"] == expected_column,
            "Factor covariance order differs",
        )
        _check(row["factor_row_ordinal"] == FACTOR_NAMES.index(expected_row), "Factor row ordinal differs")
        _check(row["factor_col_ordinal"] == FACTOR_NAMES.index(expected_column), "Factor column ordinal differs")
        last_window = row["factor_window_last_date"]
        _check(last_window is None or last_window <= signal, "Factor covariance uses a post-signal row")
        cursor += 1
        rows += 1
    _check(cursor == len(expected_pairs), "Incomplete final covariance matrix")
    _check(tuple(observed_signals) == expected_dates, "Covariance signals are not exact")
    _check(rows == REQUIRED_FACTOR_COVARIANCE_ROWS, "Factor covariance is not 59 x 36 rows")
    return rows


def _audit_month_summary(
    path: Path,
    contract: ValidationDataContract,
) -> int:
    observed: list[dt.date] = []
    for row in _rows(
        path,
        (
            "signal_date",
            "post_cutoff_return_rows_used",
            "post_cutoff_factor_rows_used",
        ),
    ):
        signal = _date(row["signal_date"], "risk/cost summary signal date")
        _check(row["post_cutoff_return_rows_used"] == 0, "Summary reports post-cutoff returns")
        _check(row["post_cutoff_factor_rows_used"] == 0, "Summary reports post-cutoff factors")
        observed.append(signal)
    _check(tuple(observed) == contract.scored_signal_dates, "Risk/cost month summary signals differ")
    _check(len(observed) == REQUIRED_SIGNAL_COUNT, "Risk/cost summary is not 59 rows")
    return len(observed)


def verify_development_parity(
    *,
    candidate_feature_table: pa.Table,
    reference_feature_table: pa.Table,
    candidate_lineage_tables: Mapping[str, pa.Table],
    reference_lineage_tables: Mapping[str, pa.Table],
    contract: ValidationDataContract,
) -> DevelopmentParityReceipt:
    """Compare a generalized builder's development replay with frozen Stage 6.

    This lower-level diagnostic remains useful for exact tables or slices.  It
    is not a publication authority; publication accepts only the separately
    reauthenticated, opaque ``ValidatedDevelopmentParity`` authority.
    """

    _check(
        candidate_feature_table.schema.equals(
            reference_feature_table.schema, check_metadata=True
        ),
        "Development feature parity schema differs",
    )
    _check(
        candidate_feature_table.schema.equals(
            contract.raw_feature_schema, check_metadata=True
        ),
        "Development parity did not use the frozen feature schema",
    )
    differences = 0 if candidate_feature_table.equals(reference_feature_table) else 1
    _check(
        set(candidate_lineage_tables) == set(contract.family_feature_names)
        == set(reference_lineage_tables),
        "Development lineage parity family set differs",
    )
    lineage_rows = 0
    for family in contract.family_feature_names:
        candidate = candidate_lineage_tables[family]
        reference = reference_lineage_tables[family]
        _check(
            candidate.schema.equals(contract.lineage_schema, check_metadata=True)
            and reference.schema.equals(contract.lineage_schema, check_metadata=True),
            f"Development lineage parity schema differs: {family}",
        )
        if not candidate.equals(reference):
            differences += 1
        lineage_rows += candidate.num_rows
    _check(differences == 0, "Development parity differs from frozen Stage 6")
    evidence = {
        "comparison": "exact_in_memory_arrow_table_equality",
        "feature_tables_compared": 2,
        "lineage_families_compared": len(contract.family_feature_names),
        "complete_reference_coverage": (
            candidate_feature_table.num_rows
            == int(contract.rules["development_parity"]["reference_feature_rows"])
            and lineage_rows
            == int(contract.rules["development_parity"]["reference_lineage_rows"])
        ),
    }
    return DevelopmentParityReceipt(
        schema_version=1,
        status="PASS",
        method="exact_empirical_arrow_replay_comparison",
        feature_rows=candidate_feature_table.num_rows,
        lineage_rows=lineage_rows,
        empirical_replay_feature_rows=candidate_feature_table.num_rows,
        empirical_replay_lineage_rows=lineage_rows,
        identity_mismatch_count=0,
        empirical_difference_count=0,
        reference_build_id=str(
            contract.rules["development_parity"]["reference_build_id"]
        ),
        reference_verification_id=str(
            contract.rules["development_parity"]["reference_verification_id"]
        ),
        evidence_sha256=hashlib.sha256(canonical_json_bytes(evidence)).hexdigest(),
        evidence=MappingProxyType(evidence),
    )


def _manifest(
    contract: ValidationDataContract,
    candidate_root: Path,
    receipts: Mapping[str, FileReceipt],
    feature_audit: FeatureAudit,
    lineage_audits: Mapping[str, LineageAudit],
    *,
    development_parity: ValidatedDevelopmentParity,
    empirical_build: ValidatedEmpiricalCandidateBuild,
) -> Mapping[str, Any]:
    _check(
        type(development_parity) is ValidatedDevelopmentParity,
        "An authenticated development-parity authority is required",
    )
    parity_receipt = development_parity.receipt
    _check(
        parity_receipt.status == PARITY_STATUS
        and parity_receipt.method == PARITY_METHOD
        and parity_receipt.identity_mismatch_count == 0
        and parity_receipt.empirical_difference_count == 0
        and parity_receipt.empirical_replay_feature_rows
        == parity_receipt.feature_rows
        and parity_receipt.empirical_replay_lineage_rows
        == parity_receipt.lineage_rows
        and parity_receipt.evidence.get("full_development_logical_replay", {}).get(
            "difference_count"
        )
        == 0
        and parity_receipt.evidence.get("actual_stage11_orchestration_canary", {}).get(
            "difference_count"
        )
        == 0
        and parity_receipt.evidence.get("development_scope", {}).get("exact_noop")
        is True
        and parity_receipt.evidence.get("stage6_identity", {}).get(
            "all_input_bytes_reauthenticated"
        )
        is True
        and parity_receipt.evidence.get("reference_outputs", {}).get(
            "all_declared_bytes_reauthenticated"
        )
        is True,
        "Authenticated full development replay parity is required",
    )
    parity_rules = contract.rules["development_parity"]
    _check(
        parity_receipt.feature_rows
        == int(parity_rules["reference_feature_rows"]),
        "Development parity did not cover the complete frozen feature store",
    )
    _check(
        parity_receipt.lineage_rows
        == int(parity_rules["reference_lineage_rows"])
        == parity_receipt.feature_rows * len(contract.feature_names),
        "Development parity did not cover complete frozen lineage",
    )
    _check(
        parity_receipt.reference_build_id == parity_rules["reference_build_id"]
        and parity_receipt.reference_verification_id
        == parity_rules["reference_verification_id"],
        "Development parity authority differs",
    )
    _check(
        type(empirical_build) is ValidatedEmpiricalCandidateBuild
        and empirical_build._seal is empirical_stream._EMPIRICAL_BUILD_RECEIPT_SEAL,
        "An authenticated empirical-build authority is required",
    )
    document = empirical_build.receipt.as_dict()
    _check(
        document.get("status") == EMPIRICAL_BUILD_STATUS
        and empirical_build.candidate_root == candidate_root,
        "Empirical-build status/candidate root differs",
    )
    receipt_bytes = empirical_build.receipt_path.read_bytes()
    _check(
        receipt_bytes == empirical_build.receipt.canonical_bytes
        and hashlib.sha256(receipt_bytes).hexdigest()
        == empirical_build.receipt_sha256,
        "Empirical-build receipt bytes differ",
    )
    expected_outputs = {
        role: receipts[role].as_dict() for role in sorted(receipts)
    }
    _check(
        document.get("outputs") == expected_outputs,
        "Empirical-build output receipts differ from validated candidate",
    )
    code_identity = document.get("code_identity", {})
    current_code_hashes = dict(materialization_code_hashes())
    _check(
        code_identity.get("start_hashes")
        == code_identity.get("end_hashes")
        == current_code_hashes,
        "Empirical build-time code differs from candidate-finalization code",
    )
    try:
        empirical_receipt_relative = empirical_build.receipt_path.relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix()
    except ValueError as exc:
        raise ValidationDataMaterializationError(
            "Empirical-build receipt escaped the project root"
        ) from exc
    _check(
        tuple(Path(empirical_receipt_relative).parts[:2])
        == ("artifacts", "phase_05_validation"),
        "Empirical-build receipt escaped the Stage 11 artifact namespace",
    )
    feature_input = document["feature_input"]
    identities = document["feature_identities"]
    router = document["crsp_router"]
    empirical_binding = {
        "build_id": document["build_id"],
        "content_fingerprint": document["content_fingerprint"],
        "receipt_path": empirical_receipt_relative,
        "receipt_sha256": empirical_build.receipt_sha256,
        "feature_input_receipt_sha256": feature_input["receipt_sha256"],
        "feature_input_bundle_sha256": feature_input["input_bundle_sha256"],
        "feature_input_adapter_code_sha256": feature_input["adapter_code_sha256"],
        "scored_build_id": identities["scored_build_id"],
        "scored_content_fingerprint": identities["scored_content_fingerprint"],
        "optionmetrics_working_build_id": identities[
            "optionmetrics_working_build_id"
        ],
        "optionmetrics_working_content_fingerprint": identities[
            "optionmetrics_working_content_fingerprint"
        ],
        "crsp_source_id": router["source_id"],
        "crsp_database_sha256": router["database_sha256"],
        "output_receipts_sha256": document["output_receipts_sha256"],
        "materialization_code_hashes_sha256": code_identity[
            "end_hashes_sha256"
        ],
    }
    receipt_payload = [receipts[role].as_dict() for role in sorted(receipts)]
    code_hashes = dict(materialization_code_hashes())
    fingerprint_payload = {
        "schema_version": 1,
        "contract_id": str(contract.policy["contract_id"]),
        "policy_sha256": contract.policy_sha256,
        "frozen_bindings": dict(contract.frozen_bindings),
        "code_hashes": code_hashes,
        "payloads": receipt_payload,
        "development_parity": parity_receipt.as_dict(),
        "empirical_build": empirical_binding,
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(fingerprint_payload)).hexdigest()
    build_id = f"stage11-validation-data-{fingerprint[:24]}"
    manifest = {
        "schema_version": 1,
        "contract_id": str(contract.policy["contract_id"]),
        "policy_id": str(contract.policy["policy_id"]),
        "build_id": build_id,
        "status": "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
        "bindings": {
            "policy_sha256": contract.policy_sha256,
            "frozen_bindings": dict(contract.frozen_bindings),
            "code_hashes": code_hashes,
            "content_fingerprint": fingerprint,
        },
        "scope": {
            "internal_context_signal_date": INTERNAL_CONTEXT_SIGNAL.isoformat(),
            "internal_context_rows_published": 0,
            "first_signal_date": contract.scored_signal_dates[0].isoformat(),
            "last_signal_date": contract.scored_signal_dates[-1].isoformat(),
            "signal_dates": len(contract.scored_signal_dates),
            "feature_rows": feature_audit.rows,
            "economic_features": len(contract.feature_names),
            "frozen_category_vocabulary_reused": True,
            "validation_vocabulary_expansions": 0,
            "frozen_model_support_changes": 0,
            "inactive_segment_nonmissing_values": feature_audit.inactive_segment_nonmissing_values,
            "logical_lineage_rows": sum(value.rows for value in lineage_audits.values()),
        },
        "payloads": {receipt.role: receipt.as_dict() for receipt in receipts.values()},
        "development_parity": parity_receipt.as_dict(),
        "empirical_build": empirical_binding,
        "access_counters": {
            "validation_label_rows_read": 0,
            "validation_outcome_rows_read": 0,
            "validation_forward_return_rows_read": 0,
            "final_period_rows_read": 0,
            "real_validation_weights_created": 0,
            "real_validation_trades_created": 0,
            "real_validation_pnl_or_performance_rows_created": 0,
        },
        "authorizations": {
            "independent_verification_completed": False,
            "validation_predictor_risk_and_cost_use_authorized": False,
            "validation_outcome_access_authorized": False,
            "real_validation_weights_trades_or_pnl_authorized": False,
            "final_period_access_authorized": False,
        },
    }
    return MappingProxyType(manifest)


def validate_candidate_bundle(
    candidate_root: Path,
    *,
    policy_path: Path = DEFAULT_POLICY,
    development_parity: ValidatedDevelopmentParity,
    empirical_build: ValidatedEmpiricalCandidateBuild,
) -> CandidateValidationResult:
    """Fully scan a prebuilt no-outcome candidate and prepare its manifest."""

    contract = validate_validation_data_contract(policy_path)
    root = _safe_candidate_root(candidate_root)
    expected_names = {value.filename for value in contract.payload_specs.values()}
    observed_names = {value.name for value in root.iterdir()}
    _check(observed_names == expected_names, "Candidate payload set is not closed")
    paths = {
        role: _safe_payload(root, spec.filename)
        for role, spec in contract.payload_specs.items()
    }

    feature_path = paths["validation_predictor_payload"]
    feature_audit = _audit_feature_payload(feature_path, contract)
    lineage_audits: dict[str, LineageAudit] = {}
    for family, feature_names in contract.family_feature_names.items():
        lineage_audits[family] = _audit_lineage_payload(
            paths[f"validation_feature_lineage_{family}"],
            feature_path=feature_path,
            family=family,
            feature_names=feature_names,
            feature_audit=feature_audit,
            contract=contract,
        )

    risk_audit_rows = _audit_risk_table(
        paths["validation_risk_cost_audit"],
        feature_audit=feature_audit,
        input_only=False,
        contract=contract,
    )
    risk_input_rows = _audit_risk_table(
        paths["validation_risk_cost_inputs"],
        feature_audit=feature_audit,
        input_only=True,
        contract=contract,
    )
    covariance_rows = _audit_factor_covariance(
        paths["validation_factor_covariance"], contract
    )
    summary_rows = _audit_month_summary(
        paths["validation_risk_cost_month_summary"], contract
    )

    receipts = MappingProxyType(
        {
            role: _file_receipt(
                role,
                paths[role],
                specification.schema,
                minimum_signal_date=contract.scored_signal_dates[0],
                maximum_signal_date=contract.scored_signal_dates[-1],
            )
            for role, specification in contract.payload_specs.items()
        }
    )
    manifest = _manifest(
        contract,
        root,
        receipts,
        feature_audit,
        lineage_audits,
        development_parity=development_parity,
        empirical_build=empirical_build,
    )
    return CandidateValidationResult(
        contract=contract,
        root=root,
        receipts=receipts,
        feature_audit=feature_audit,
        lineage_audits=MappingProxyType(lineage_audits),
        risk_audit_rows=risk_audit_rows,
        risk_input_rows=risk_input_rows,
        factor_covariance_rows=covariance_rows,
        month_summary_rows=summary_rows,
        manifest=manifest,
    )


def write_candidate_manifest(
    result: CandidateValidationResult,
    destination: Path,
) -> Path:
    """Write one canonical manifest with exclusive-create semantics."""

    path = Path(os.path.abspath(destination.expanduser()))
    _check(path.parent.is_dir() and not path.parent.is_symlink(), "Unsafe manifest directory")
    payload = canonical_json_bytes(dict(result.manifest))
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValidationDataMaterializationError("Manifest destination already exists") from exc
    return path


__all__ = [
    "CandidateValidationResult",
    "DevelopmentParityReceipt",
    "ValidatedDevelopmentParity",
    "FeatureAudit",
    "FileReceipt",
    "LineageAudit",
    "MATERIALIZATION_CODE_PATHS",
    "ValidationDataMaterializationError",
    "materialization_plan",
    "materialization_code_hashes",
    "validate_candidate_bundle",
    "verify_development_parity",
    "write_candidate_manifest",
]
