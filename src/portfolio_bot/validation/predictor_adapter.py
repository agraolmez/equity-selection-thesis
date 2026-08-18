"""Authenticated Stage 11 validation-predictor registration.

This module turns one independently verified, closed validation-data bundle
into the exact
``RegisteredPayload`` accepted by the no-outcome access broker.  The caller
cannot supply or widen a column allowlist: the 170 physical columns are
reconstructed from the hash-frozen Stage 6 registry and compared with both the
verified development Parquet footer and the validation-predictor footer.

Only Parquet metadata and predictor-file bytes are inspected here.  No label,
forward-return, outcome, portfolio, trade, or P&L payload is opened.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.features.feature_store_contract import (
    expanded_features,
    raw_feature_schema,
)
from portfolio_bot.validation.access import (
    PayloadRole,
    PreflightAccessBroker,
    RegisteredPayload,
)
from portfolio_bot.validation.validation_development_parity import (
    PARITY_METHOD,
    PARITY_STATUS,
    prove_development_code_identity_parity,
)
from portfolio_bot.validation.validation_data_independent_verification import (
    independently_verify_candidate_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATABASE_ROOT = Path("/Users/agraemirolmez/Desktop/Thesis")

REGISTRATION_NAME = "authenticated_validation_predictors"
MANIFEST_ID = "stage11-validation-predictor-manifest-v2"
MANIFEST_STATUS = "SEALED_VALIDATION_PREDICTORS_NO_OUTCOMES"

CORRECTED_STAGE6_BUILD_ID = "phase03-features-fda21f0e51b8276c10246f9f"
STAGE6_VERIFICATION_ID = "verify-350915782d3ef12e"
STAGE6_VERIFIED_POINTER_RELATIVE = Path(
    "artifacts/phase_03_features/LATEST_FEATURE_STORE_VERIFIED.json"
)
STAGE6_PRIMARY_MANIFEST_RELATIVE = Path(
    "artifacts/phase_03_features/feature_store_builds/"
    f"{CORRECTED_STAGE6_BUILD_ID}/manifest.json"
)
FEATURE_REGISTRY_RELATIVE = Path(
    "configs/phase_03_features/feature_registry_v1.json"
)
DEVELOPMENT_RAW_RELATIVE = Path(
    "interim/phase_03_features/phase03-feature-label-split-v1/"
    f"feature_store_builds/{CORRECTED_STAGE6_BUILD_ID}/feature_store_raw.parquet"
)

EXPECTED_STAGE6_VERIFIED_POINTER_SHA256 = (
    "29421df4f8d5becd5f68af566dacad1b6683c57f98083c03cfa624649b591308"
)
EXPECTED_STAGE6_PRIMARY_MANIFEST_SHA256 = (
    "96ab135c0628054699fa3f27a76abd1cb30f3e3e46279a84052a29c895b51375"
)
EXPECTED_FEATURE_REGISTRY_SHA256 = (
    "7bac81cd8083c9a5c44ab2482223a342ed6ce2ce265d73cd8cc1bd843e33805a"
)
EXPECTED_RAW_FEATURE_SCHEMA_SHA256 = (
    "e67436adbe6e2adb2f445a799f07002a272a5cdf780662aca639f0eb3c0969c3"
)
EXPECTED_DEVELOPMENT_RAW_SHA256 = (
    "07c588ddddb1b9b48db8a4cc8b1d5527aefab0d771b001d651cc33f7a735a854"
)
EXPECTED_DEVELOPMENT_RAW_SIZE_BYTES = 749_178_306
EXPECTED_DEVELOPMENT_RAW_ROWS = 1_263_048
EXPECTED_DEVELOPMENT_LINEAGE_ROWS = 205_876_824
EXPECTED_DEVELOPMENT_RAW_ROW_GROUPS = 125
EXPECTED_RAW_COLUMN_COUNT = 170
EXPECTED_ECONOMIC_FEATURE_COUNT = 163

MINIMUM_VALIDATION_SIGNAL_DATE = dt.date(2014, 12, 31)
MAXIMUM_VALIDATION_SIGNAL_DATE = dt.date(2019, 10, 31)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "status",
        "payload",
        "stage06_binding",
        "validation_data_verification",
    }
)
_PAYLOAD_KEYS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "rows",
        "row_groups",
        "schema_sha256",
        "column_count",
        "columns_sha256",
        "minimum_signal_date",
        "maximum_signal_date",
    }
)
_STAGE6_BINDING_KEYS = frozenset(
    {
        "corrected_build_id",
        "verification_id",
        "verified_pointer_path",
        "verified_pointer_sha256",
        "primary_manifest_path",
        "primary_manifest_sha256",
        "feature_registry_path",
        "feature_registry_sha256",
    }
)
_VALIDATION_DATA_BINDING_KEYS = frozenset(
    {
        "build_id",
        "verification_id",
        "candidate_manifest_path",
        "candidate_manifest_sha256",
        "verification_report_path",
        "verification_report_sha256",
        "materialization_code_hashes_sha256",
        "predictor_payload_role",
        "empirical_build_id",
        "empirical_build_receipt_path",
        "empirical_build_receipt_sha256",
        "empirical_build_content_fingerprint",
    }
)
_VALIDATION_DATA_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "verification_id",
        "status",
        "build_id",
        "manifest_sha256",
        "materialization_code_hashes_sha256",
        "payload_count",
        "payload_rows",
        "signal_dates_checked_per_payload",
        "difference_count",
        "raw_lineage_semantic_difference_count",
        "empirical_build_id",
        "empirical_build_receipt_path",
        "empirical_build_receipt_sha256",
        "empirical_build_content_fingerprint",
        "validation_outcome_rows_read",
        "final_period_rows_read",
        "validation_predictor_risk_and_cost_use_authorized",
        "validation_outcome_access_authorized",
        "real_validation_weights_trades_or_pnl_authorized",
        "final_period_access_authorized",
    }
)
_VALIDATION_DATA_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "status",
        "bindings",
        "scope",
        "payloads",
        "development_parity",
        "empirical_build",
        "access_counters",
        "authorizations",
    }
)
_VALIDATION_DATA_PAYLOAD_RECEIPT_KEYS = frozenset(
    {
        "role",
        "path",
        "sha256",
        "size_bytes",
        "rows",
        "row_groups",
        "schema_sha256",
        "minimum_signal_date",
        "maximum_signal_date",
    }
)
_EMPIRICAL_BUILD_BINDING_KEYS = frozenset(
    {
        "build_id",
        "content_fingerprint",
        "receipt_path",
        "receipt_sha256",
        "feature_input_receipt_sha256",
        "feature_input_bundle_sha256",
        "feature_input_adapter_code_sha256",
        "scored_build_id",
        "scored_content_fingerprint",
        "optionmetrics_working_build_id",
        "optionmetrics_working_content_fingerprint",
        "crsp_source_id",
        "crsp_database_sha256",
        "output_receipts_sha256",
        "materialization_code_hashes_sha256",
    }
)
VALIDATION_PREDICTOR_ROLE = "validation_predictor_payload"
_EXACT_VALIDATION_DATA_ROLES = frozenset(
    {
        "validation_predictor_payload",
        "validation_feature_lineage_price_trends_path",
        "validation_feature_lineage_risk_return_distribution",
        "validation_feature_lineage_size_liquidity_trading_frictions",
        "validation_feature_lineage_valuation_fundamentals",
        "validation_feature_lineage_ownership_filings_events_segments",
        "validation_feature_lineage_optionmetrics",
        "validation_feature_lineage_aggregate_state",
        "validation_risk_cost_audit",
        "validation_risk_cost_inputs",
        "validation_factor_covariance",
        "validation_risk_cost_month_summary",
    }
)
_POINTER_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "verification_id",
        "status",
        "primary_manifest_sha256",
        "independent_check_sha256",
        "differences_sha256",
        "authorizations",
    }
)
_POINTER_AUTHORIZATION_KEYS = frozenset(
    {
        "independent_verification_completed",
        "models_portfolios_pnl_or_backtests_authorized",
        "stage_07_labels_authorized",
        "validation_or_final_test_access_authorized",
    }
)
_FORBIDDEN_EXACT_NAMES = frozenset(
    {
        "forward_return",
        "future_return",
        "residual_return",
        "realized_return",
        "outcome",
        "target",
        "label",
    }
)


class PredictorAdapterError(RuntimeError):
    """Raised when predictor authority, identity, or schema fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PredictorAdapterError(message)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return _sha256_bytes(schema.serialize().to_pybytes())


def _column_order_sha256(columns: tuple[str, ...]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(columns)))


def _exact_keys(value: object, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    observed = frozenset(value)
    _require(
        observed == expected,
        f"{label} keys differ; missing={sorted(expected - observed)}; "
        f"extra={sorted(observed - expected)}",
    )
    return value


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _ordinary_file_under(path: Path, root: Path, *, label: str) -> tuple[Path, os.stat_result]:
    root_absolute = Path(os.path.abspath(root.expanduser()))
    lexical = Path(os.path.abspath(path.expanduser()))
    try:
        lexical.relative_to(root_absolute)
    except ValueError as exc:
        raise PredictorAdapterError(f"{label} escaped its authenticated root") from exc
    _require(
        lexical.resolve(strict=False) == lexical,
        f"{label} uses a symlinked path",
    )
    try:
        metadata = lexical.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise PredictorAdapterError(f"Missing {label}") from exc
    _require(
        not stat.S_ISLNK(metadata.st_mode) and stat.S_ISREG(metadata.st_mode),
        f"{label} is not an ordinary non-symlink file",
    )
    _require(metadata.st_nlink == 1, f"{label} has unsafe hard links")
    return lexical, metadata


def _read_small_authority(
    relative: Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Path, bytes]:
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"{label} path is unsafe",
    )
    path, before = _ordinary_file_under(PROJECT_ROOT / relative, PROJECT_ROOT, label=label)
    payload = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    _require(_stable_identity(before) == _stable_identity(after), f"{label} changed while read")
    _require(_sha256_bytes(payload) == expected_sha256, f"{label} SHA-256 differs")
    return path, payload


def _read_validation_authority(
    relative_text: object,
    expected_sha256: object,
    *,
    label: str,
) -> tuple[Path, bytes]:
    _require(isinstance(relative_text, str) and bool(relative_text), f"{label} path is invalid")
    relative = PurePosixPath(relative_text)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts[:2] == ("artifacts", "phase_05_validation"),
        f"{label} escaped the Stage 11 artifact namespace",
    )
    digest = str(expected_sha256)
    _require(_SHA256.fullmatch(digest) is not None, f"{label} SHA-256 is malformed")
    path, before = _ordinary_file_under(
        PROJECT_ROOT.joinpath(*relative.parts),
        PROJECT_ROOT,
        label=label,
    )
    payload = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    _require(_stable_identity(before) == _stable_identity(after), f"{label} changed while read")
    _require(_sha256_bytes(payload) == digest, f"{label} SHA-256 differs")
    return path, payload


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictorAdapterError(f"{label} is not valid UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _physical_raw_schema(registry_payload: bytes) -> pa.Schema:
    registry = _json_object(registry_payload, label="frozen feature registry")
    features = expanded_features(registry)
    _require(
        len(features) == EXPECTED_ECONOMIC_FEATURE_COUNT,
        "frozen feature registry is not exactly 163 features",
    )
    logical = raw_feature_schema(features)
    fields: list[pa.Field] = []
    for field in logical:
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
    schema = pa.schema(fields)
    _require(len(schema) == EXPECTED_RAW_COLUMN_COUNT, "raw feature schema is not 170 columns")
    _require(
        _schema_sha256(schema) == EXPECTED_RAW_FEATURE_SCHEMA_SHA256,
        "frozen raw feature schema SHA-256 differs",
    )
    return schema


def _expected_stage6_binding() -> dict[str, object]:
    return {
        "corrected_build_id": CORRECTED_STAGE6_BUILD_ID,
        "verification_id": STAGE6_VERIFICATION_ID,
        "verified_pointer_path": STAGE6_VERIFIED_POINTER_RELATIVE.as_posix(),
        "verified_pointer_sha256": EXPECTED_STAGE6_VERIFIED_POINTER_SHA256,
        "primary_manifest_path": STAGE6_PRIMARY_MANIFEST_RELATIVE.as_posix(),
        "primary_manifest_sha256": EXPECTED_STAGE6_PRIMARY_MANIFEST_SHA256,
        "feature_registry_path": FEATURE_REGISTRY_RELATIVE.as_posix(),
        "feature_registry_sha256": EXPECTED_FEATURE_REGISTRY_SHA256,
    }


def _authenticate_stage6_authority() -> tuple[pa.Schema, Path]:
    _pointer_path, pointer_payload = _read_small_authority(
        STAGE6_VERIFIED_POINTER_RELATIVE,
        EXPECTED_STAGE6_VERIFIED_POINTER_SHA256,
        label="Stage 6 verified pointer",
    )
    pointer = _exact_keys(
        _json_object(pointer_payload, label="Stage 6 verified pointer"),
        _POINTER_KEYS,
        "Stage 6 verified pointer",
    )
    authorizations = _exact_keys(
        pointer["authorizations"],
        _POINTER_AUTHORIZATION_KEYS,
        "Stage 6 verified pointer authorizations",
    )
    _require(pointer.get("schema_version") == 1, "Stage 6 pointer schema changed")
    _require(pointer.get("status") == "VERIFIED_PASSED", "Stage 6 is not verified")
    _require(pointer.get("build_id") == CORRECTED_STAGE6_BUILD_ID, "wrong Stage 6 build")
    _require(pointer.get("verification_id") == STAGE6_VERIFICATION_ID, "wrong Stage 6 verification")
    _require(
        pointer.get("primary_manifest_sha256")
        == EXPECTED_STAGE6_PRIMARY_MANIFEST_SHA256,
        "Stage 6 pointer does not bind the corrected manifest",
    )
    _require(
        authorizations.get("independent_verification_completed") is True,
        "Stage 6 independent verification is incomplete",
    )

    _manifest_path, primary_payload = _read_small_authority(
        STAGE6_PRIMARY_MANIFEST_RELATIVE,
        EXPECTED_STAGE6_PRIMARY_MANIFEST_SHA256,
        label="Stage 6 primary manifest",
    )
    primary = _json_object(primary_payload, label="Stage 6 primary manifest")
    _require(primary.get("build_id") == CORRECTED_STAGE6_BUILD_ID, "primary build ID differs")
    try:
        registry_binding = primary["bindings"]["configuration_hashes"][
            FEATURE_REGISTRY_RELATIVE.as_posix()
        ]
        interim = primary["outputs"]["interim"]
        raw_receipt = interim["files"]["feature_store_raw.parquet"]
    except (KeyError, TypeError) as exc:
        raise PredictorAdapterError("Stage 6 primary manifest is missing raw-feature bindings") from exc
    _require(
        registry_binding == EXPECTED_FEATURE_REGISTRY_SHA256,
        "Stage 6 manifest registry binding differs",
    )
    _require(
        interim.get("path") == DEVELOPMENT_RAW_RELATIVE.parent.as_posix(),
        "Stage 6 development raw path differs",
    )
    expected_raw_receipt = {
        "path": "feature_store_raw.parquet",
        "sha256": EXPECTED_DEVELOPMENT_RAW_SHA256,
        "size_bytes": EXPECTED_DEVELOPMENT_RAW_SIZE_BYTES,
        "rows": EXPECTED_DEVELOPMENT_RAW_ROWS,
        "row_groups": EXPECTED_DEVELOPMENT_RAW_ROW_GROUPS,
        "schema_sha256": EXPECTED_RAW_FEATURE_SCHEMA_SHA256,
    }
    _require(raw_receipt == expected_raw_receipt, "Stage 6 raw-feature receipt differs")

    _registry_path, registry_payload = _read_small_authority(
        FEATURE_REGISTRY_RELATIVE,
        EXPECTED_FEATURE_REGISTRY_SHA256,
        label="frozen feature registry",
    )
    return _physical_raw_schema(registry_payload), DATABASE_ROOT / DEVELOPMENT_RAW_RELATIVE


def _forbidden_predictor_name(name: str, frozen_columns: frozenset[str]) -> bool:
    if name in frozen_columns:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    tokens = tuple(token for token in normalized.split("_") if token)
    if normalized in _FORBIDDEN_EXACT_NAMES:
        return True
    if any(token in {"outcome", "target", "label", "forward", "future"} for token in tokens):
        return True
    if "return" in tokens and any(token in {"realized", "residual"} for token in tokens):
        return True
    return False


def _parquet_footer(
    path: Path,
    *,
    expected_size: int,
    label: str,
) -> tuple[pa.Schema, int, int, dt.date | None, dt.date | None]:
    ordinary, before = _ordinary_file_under(path, DATABASE_ROOT, label=label)
    _require(before.st_size == expected_size, f"{label} size differs")
    try:
        parquet = pq.ParquetFile(ordinary)
    except (pa.ArrowException, OSError) as exc:
        raise PredictorAdapterError(f"{label} is not readable Parquet") from exc
    metadata = parquet.metadata
    schema = parquet.schema_arrow
    signal_index = schema.get_field_index("signal_date")
    _require(signal_index >= 0, f"{label} has no signal_date column")
    minima: list[dt.date] = []
    maxima: list[dt.date] = []
    for index in range(metadata.num_row_groups):
        statistics = metadata.row_group(index).column(signal_index).statistics
        _require(
            statistics is not None and statistics.has_min_max,
            f"{label} signal_date row-group statistics are incomplete",
        )
        minimum = statistics.min
        maximum = statistics.max
        _require(
            type(minimum) is dt.date and type(maximum) is dt.date,
            f"{label} signal_date statistics have the wrong type",
        )
        minima.append(minimum)
        maxima.append(maximum)
    after = ordinary.stat(follow_symlinks=False)
    _require(_stable_identity(before) == _stable_identity(after), f"{label} changed while inspected")
    return (
        schema,
        int(metadata.num_rows),
        int(metadata.num_row_groups),
        min(minima) if minima else None,
        max(maxima) if maxima else None,
    )


def _sha256_regular_file(path: Path, *, expected_size: int, label: str) -> str:
    ordinary, before = _ordinary_file_under(path, DATABASE_ROOT, label=label)
    _require(before.st_size == expected_size, f"{label} size differs")
    digest = hashlib.sha256()
    with ordinary.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    after = ordinary.stat(follow_symlinks=False)
    _require(_stable_identity(before) == _stable_identity(after), f"{label} changed while hashed")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DevelopmentSchemaReconciliationReceipt:
    corrected_stage6_build_id: str
    stage6_verified_pointer_sha256: str
    stage6_primary_manifest_sha256: str
    feature_registry_sha256: str
    development_raw_sha256: str
    development_raw_rows: int
    development_raw_schema_sha256: str
    validation_raw_schema_sha256: str
    raw_column_count: int
    economic_feature_count: int
    development_column_order_sha256: str
    validation_column_order_sha256: str
    exact_column_order_match: bool
    exact_arrow_schema_match: bool
    parquet_footer_only: bool
    outcome_or_label_payloads_opened: int
    outcome_or_label_rows_read: int


@dataclass(frozen=True, slots=True)
class ValidationDataVerificationReceipt:
    build_id: str
    verification_id: str
    candidate_manifest_path: Path
    candidate_manifest_sha256: str
    verification_report_path: Path
    verification_report_sha256: str
    materialization_code_hashes_sha256: str
    empirical_build_id: str
    empirical_build_receipt_path: Path
    empirical_build_receipt_sha256: str
    empirical_build_content_fingerprint: str
    candidate_root: Path
    payload_receipts: Mapping[str, Mapping[str, Any]]
    difference_count: int
    validation_outcome_rows_read: int
    final_period_rows_read: int

    @property
    def predictor_payload_receipt(self) -> Mapping[str, Any]:
        """Compatibility view of the exact verified predictor receipt."""

        return self.payload_receipts[VALIDATION_PREDICTOR_ROLE]

    def payload_receipt(self, role: str) -> Mapping[str, Any]:
        """Return one receipt only when it belongs to the closed 12-role set."""

        if role not in _EXACT_VALIDATION_DATA_ROLES:
            raise PredictorAdapterError(
                f"unregistered verified validation-data role: {role}"
            )
        return self.payload_receipts[role]


@dataclass(frozen=True, slots=True)
class AuthenticatedValidationPredictor:
    manifest_path: Path
    manifest_sha256: str
    registration_name: str
    registration: RegisteredPayload
    payload_rows: int
    payload_row_groups: int
    reconciliation: DevelopmentSchemaReconciliationReceipt
    validation_data_verification: ValidationDataVerificationReceipt

    def registration_mapping(self) -> Mapping[str, RegisteredPayload]:
        """Return the immutable, exact mapping accepted by the access broker."""

        return MappingProxyType({self.registration_name: self.registration})

    def build_broker(self) -> PreflightAccessBroker:
        """Construct a broker whose predictor allowlist cannot be widened."""

        return PreflightAccessBroker(self.registration_mapping())


def _authenticate_validation_data_verification(
    value: object,
) -> ValidationDataVerificationReceipt:
    binding = _exact_keys(
        value,
        _VALIDATION_DATA_BINDING_KEYS,
        "validation-data verification binding",
    )
    _require(
        binding["predictor_payload_role"] == VALIDATION_PREDICTOR_ROLE,
        "validation-data predictor role differs",
    )
    manifest_path, manifest_payload = _read_validation_authority(
        binding["candidate_manifest_path"],
        binding["candidate_manifest_sha256"],
        label="validation-data candidate manifest",
    )
    report_path, report_payload = _read_validation_authority(
        binding["verification_report_path"],
        binding["verification_report_sha256"],
        label="validation-data independent verification report",
    )
    candidate = _exact_keys(
        _json_object(manifest_payload, label="validation-data candidate manifest"),
        _VALIDATION_DATA_MANIFEST_KEYS,
        "validation-data candidate manifest",
    )
    report = _exact_keys(
        _json_object(report_payload, label="validation-data independent verification report"),
        _VALIDATION_DATA_REPORT_KEYS,
        "validation-data independent verification report",
    )
    _require(
        manifest_payload == _canonical_json_bytes(candidate) + b"\n",
        "validation-data candidate manifest is not canonical JSON",
    )
    _require(
        report_payload == _canonical_json_bytes(report) + b"\n",
        "validation-data verification report is not canonical JSON",
    )
    empirical = _exact_keys(
        candidate["empirical_build"],
        _EMPIRICAL_BUILD_BINDING_KEYS,
        "validation-data empirical-build binding",
    )
    _require(
        empirical["build_id"] == binding["empirical_build_id"]
        == report["empirical_build_id"]
        and empirical["receipt_path"]
        == binding["empirical_build_receipt_path"]
        == report["empirical_build_receipt_path"]
        and empirical["receipt_sha256"]
        == binding["empirical_build_receipt_sha256"]
        == report["empirical_build_receipt_sha256"]
        and empirical["content_fingerprint"]
        == binding["empirical_build_content_fingerprint"]
        == report["empirical_build_content_fingerprint"],
        "validation-data empirical-build identity differs",
    )
    _empirical_path, empirical_payload = _read_validation_authority(
        empirical["receipt_path"],
        empirical["receipt_sha256"],
        label="empirical-build receipt",
    )
    empirical_document = _json_object(
        empirical_payload,
        label="empirical-build receipt",
    )
    _require(
        empirical_payload == _canonical_json_bytes(empirical_document) + b"\n",
        "empirical-build receipt is not canonical JSON",
    )
    candidate_root_text = empirical_document.get("candidate_root")
    _require(
        isinstance(candidate_root_text, str) and bool(candidate_root_text),
        "empirical-build candidate root is absent",
    )
    candidate_root = Path(candidate_root_text)
    _require(candidate_root.is_absolute(), "empirical-build candidate root is not absolute")
    try:
        candidate_relative = candidate_root.relative_to(DATABASE_ROOT.resolve())
    except ValueError as exc:
        raise PredictorAdapterError(
            "empirical-build candidate root escaped the database root"
        ) from exc
    _require(
        candidate_relative.parts[:2] == ("interim", "phase_05_validation"),
        "empirical-build candidate root escaped the Stage 11 interim namespace",
    )
    reproduced = independently_verify_candidate_manifest(
        manifest_path,
        candidate_root,
    )
    _require(
        report_payload == _canonical_json_bytes(dict(reproduced.report)) + b"\n",
        "validation-data independent verification did not byte-exactly reproduce",
    )
    _require(candidate["schema_version"] == 1, "validation-data manifest schema differs")
    _require(
        candidate["status"] == "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
        "validation-data candidate status differs",
    )
    _require(report["schema_version"] == 1, "validation-data report schema differs")
    _require(report["status"] == "VERIFIED_PASSED", "validation-data bundle is not independently verified")
    _require(
        candidate["build_id"] == report["build_id"] == binding["build_id"],
        "validation-data build identity differs",
    )
    _require(
        report["verification_id"] == binding["verification_id"],
        "validation-data verification identity differs",
    )
    _require(
        report["manifest_sha256"] == binding["candidate_manifest_sha256"],
        "validation-data report does not bind the candidate manifest",
    )
    code_hashes = candidate.get("bindings", {}).get("code_hashes")
    _require(isinstance(code_hashes, dict) and bool(code_hashes), "validation-data code hashes are missing")
    code_digest = _sha256_bytes(_canonical_json_bytes(code_hashes) + b"\n")
    _require(
        code_digest
        == report["materialization_code_hashes_sha256"]
        == binding["materialization_code_hashes_sha256"],
        "validation-data code-hash identity differs",
    )
    _require(
        report["difference_count"] == 0
        and report["raw_lineage_semantic_difference_count"] == 0
        and report["payload_count"] == 12
        and report["signal_dates_checked_per_payload"] == 59,
        "validation-data independent verification is incomplete",
    )
    _require(
        report["validation_predictor_risk_and_cost_use_authorized"] is True
        and report["validation_outcome_access_authorized"] is False
        and report["real_validation_weights_trades_or_pnl_authorized"] is False
        and report["final_period_access_authorized"] is False,
        "validation-data verification authorizations differ",
    )
    _require(
        report["validation_outcome_rows_read"] == 0
        and report["final_period_rows_read"] == 0,
        "validation-data verification reports a prohibited read",
    )
    _require(
        all(value == 0 for value in candidate.get("access_counters", {}).values()),
        "validation-data primary manifest reports a prohibited read/output",
    )
    parity = candidate.get("development_parity")
    _require(
        isinstance(parity, dict)
        and parity.get("status") == PARITY_STATUS
        and parity.get("method") == PARITY_METHOD
        and parity.get("identity_mismatch_count") == 0
        and parity.get("empirical_difference_count") == 0
        and parity.get("empirical_replay_feature_rows")
        == EXPECTED_DEVELOPMENT_RAW_ROWS
        and parity.get("empirical_replay_lineage_rows")
        == EXPECTED_DEVELOPMENT_LINEAGE_ROWS
        and parity.get("evidence", {})
        .get("full_development_logical_replay", {})
        .get("difference_count")
        == 0
        and parity.get("evidence", {})
        .get("actual_stage11_orchestration_canary", {})
        .get("difference_count")
        == 0,
        "validation-data development full replay parity differs",
    )
    _require(
        parity == prove_development_code_identity_parity().receipt.as_dict(),
        "validation-data development parity did not reauthenticate",
    )
    primary_authorizations = candidate.get("authorizations", {})
    _require(
        primary_authorizations.get("independent_verification_completed") is False
        and primary_authorizations.get("validation_predictor_risk_and_cost_use_authorized") is False,
        "unverified validation-data primary self-authorized use",
    )
    payloads = candidate.get("payloads")
    _require(
        isinstance(payloads, dict)
        and frozenset(payloads) == _EXACT_VALIDATION_DATA_ROLES,
        "validation-data payload role set differs",
    )
    for role in sorted(_EXACT_VALIDATION_DATA_ROLES):
        receipt = _exact_keys(
            payloads[role],
            _VALIDATION_DATA_PAYLOAD_RECEIPT_KEYS,
            f"verified validation-data receipt {role}",
        )
        _require(
            receipt["role"] == role
            and isinstance(receipt["path"], str)
            and PurePosixPath(str(receipt["path"])).name == receipt["path"]
            and isinstance(receipt["sha256"], str)
            and _SHA256.fullmatch(str(receipt["sha256"])) is not None
            and type(receipt["size_bytes"]) is int
            and receipt["size_bytes"] > 0
            and type(receipt["rows"]) is int
            and receipt["rows"] > 0
            and type(receipt["row_groups"]) is int
            and receipt["row_groups"] > 0,
            f"validation-data payload receipt differs: {role}",
        )
    predictor_receipt = _exact_keys(
        payloads.get(VALIDATION_PREDICTOR_ROLE),
        _VALIDATION_DATA_PAYLOAD_RECEIPT_KEYS,
        "verified validation predictor receipt",
    )
    _require(
        predictor_receipt["role"] == VALIDATION_PREDICTOR_ROLE,
        "verified validation predictor receipt role differs",
    )
    scope = candidate.get("scope", {})
    _require(
        scope.get("internal_context_rows_published") == 0
        and scope.get("signal_dates") == 59
        and scope.get("first_signal_date") == MINIMUM_VALIDATION_SIGNAL_DATE.isoformat()
        and scope.get("last_signal_date") == MAXIMUM_VALIDATION_SIGNAL_DATE.isoformat()
        and scope.get("validation_vocabulary_expansions") == 0
        and scope.get("frozen_model_support_changes") == 0
        and scope.get("inactive_segment_nonmissing_values") == 0,
        "validation-data predictor scope/encoding lock differs",
    )
    immutable_payloads = MappingProxyType(
        {
            role: MappingProxyType(dict(payloads[role]))
            for role in sorted(_EXACT_VALIDATION_DATA_ROLES)
        }
    )
    return ValidationDataVerificationReceipt(
        build_id=str(binding["build_id"]),
        verification_id=str(binding["verification_id"]),
        candidate_manifest_path=manifest_path,
        candidate_manifest_sha256=str(binding["candidate_manifest_sha256"]),
        verification_report_path=report_path,
        verification_report_sha256=str(binding["verification_report_sha256"]),
        materialization_code_hashes_sha256=str(
            binding["materialization_code_hashes_sha256"]
        ),
        empirical_build_id=str(binding["empirical_build_id"]),
        empirical_build_receipt_path=_empirical_path,
        empirical_build_receipt_sha256=str(
            binding["empirical_build_receipt_sha256"]
        ),
        empirical_build_content_fingerprint=str(
            binding["empirical_build_content_fingerprint"]
        ),
        candidate_root=candidate_root,
        payload_receipts=immutable_payloads,
        difference_count=0,
        validation_outcome_rows_read=0,
        final_period_rows_read=0,
    )


def authenticate_validation_predictor_manifest(
    manifest_path: Path,
) -> AuthenticatedValidationPredictor:
    """Authenticate one closed manifest and return its exact broker record."""

    path, before = _ordinary_file_under(
        manifest_path,
        DATABASE_ROOT,
        label="validation-predictor manifest",
    )
    manifest_payload = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    _require(
        _stable_identity(before) == _stable_identity(after),
        "validation-predictor manifest changed while read",
    )
    manifest = _exact_keys(
        _json_object(manifest_payload, label="validation-predictor manifest"),
        _MANIFEST_KEYS,
        "validation-predictor manifest",
    )
    _require(
        manifest_payload == _canonical_json_bytes(manifest),
        "validation-predictor manifest is not canonical JSON",
    )
    _require(manifest["schema_version"] == 2, "predictor manifest schema differs")
    _require(manifest["manifest_id"] == MANIFEST_ID, "predictor manifest ID differs")
    _require(manifest["status"] == MANIFEST_STATUS, "predictor manifest is not sealed")
    stage6_binding = _exact_keys(
        manifest["stage06_binding"],
        _STAGE6_BINDING_KEYS,
        "predictor manifest Stage 6 binding",
    )
    _require(
        dict(stage6_binding) == _expected_stage6_binding(),
        "predictor manifest does not bind the corrected verified Stage 6 build",
    )
    validation_data_verification = _authenticate_validation_data_verification(
        manifest["validation_data_verification"]
    )

    frozen_schema, development_path = _authenticate_stage6_authority()
    frozen_columns = tuple(frozen_schema.names)
    frozen_column_set = frozenset(frozen_columns)
    columns_sha256 = _column_order_sha256(frozen_columns)

    payload = _exact_keys(
        manifest["payload"],
        _PAYLOAD_KEYS,
        "validation-predictor payload receipt",
    )
    relative_text = payload["path"]
    _require(isinstance(relative_text, str) and bool(relative_text), "predictor path is invalid")
    relative = PurePosixPath(relative_text)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts[:2] == ("interim", "phase_05_validation")
        and relative.suffix == ".parquet",
        "predictor path escaped the Stage 11 interim namespace",
    )
    payload_path = DATABASE_ROOT.joinpath(*relative.parts)
    digest = payload["sha256"]
    size_bytes = payload["size_bytes"]
    rows = payload["rows"]
    row_groups = payload["row_groups"]
    _require(isinstance(digest, str) and _SHA256.fullmatch(digest) is not None, "predictor SHA-256 is malformed")
    _require(type(size_bytes) is int and size_bytes > 0, "predictor size must be a positive integer")
    _require(type(rows) is int and rows > 0, "predictor rows must be a positive integer")
    _require(type(row_groups) is int and row_groups > 0, "predictor row groups must be positive")
    _require(payload["schema_sha256"] == EXPECTED_RAW_FEATURE_SCHEMA_SHA256, "predictor schema hash differs")
    _require(payload["column_count"] == EXPECTED_RAW_COLUMN_COUNT, "predictor column count differs")
    _require(payload["columns_sha256"] == columns_sha256, "predictor column ordering hash differs")
    _require(
        payload["minimum_signal_date"] == MINIMUM_VALIDATION_SIGNAL_DATE.isoformat()
        and payload["maximum_signal_date"] == MAXIMUM_VALIDATION_SIGNAL_DATE.isoformat(),
        "predictor signal-date scope differs",
    )
    verified_payload = validation_data_verification.predictor_payload_receipt
    _require(
        payload["sha256"] == verified_payload["sha256"]
        and payload["size_bytes"] == verified_payload["size_bytes"]
        and payload["rows"] == verified_payload["rows"]
        and payload["row_groups"] == verified_payload["row_groups"]
        and payload["schema_sha256"] == verified_payload["schema_sha256"]
        and PurePosixPath(str(payload["path"])).name == verified_payload["path"]
        and payload["minimum_signal_date"]
        == verified_payload["minimum_signal_date"]
        and payload["maximum_signal_date"]
        == verified_payload["maximum_signal_date"],
        "predictor payload is not the independently verified candidate payload",
    )

    observed_digest = _sha256_regular_file(
        payload_path,
        expected_size=size_bytes,
        label="validation-predictor payload",
    )
    _require(observed_digest == digest, "validation-predictor payload SHA-256 differs")
    validation_schema, observed_rows, observed_groups, minimum, maximum = _parquet_footer(
        payload_path,
        expected_size=size_bytes,
        label="validation-predictor payload",
    )
    prohibited = tuple(
        name
        for name in validation_schema.names
        if _forbidden_predictor_name(name, frozen_column_set)
    )
    _require(
        not prohibited,
        f"validation predictor exposes prohibited outcome/target/label/return columns: {prohibited}",
    )
    _require(observed_rows == rows, "validation-predictor row count differs")
    _require(observed_groups == row_groups, "validation-predictor row-group count differs")
    _require(
        minimum == MINIMUM_VALIDATION_SIGNAL_DATE
        and maximum == MAXIMUM_VALIDATION_SIGNAL_DATE,
        "validation-predictor Parquet statistics do not prove the exact signal scope",
    )
    validation_schema_hash = _schema_sha256(validation_schema)
    _require(
        validation_schema_hash == EXPECTED_RAW_FEATURE_SCHEMA_SHA256,
        "validation-predictor physical schema SHA-256 differs",
    )
    _require(validation_schema.equals(frozen_schema), "validation-predictor schema/order differs")

    development_schema, dev_rows, dev_groups, _dev_min, _dev_max = _parquet_footer(
        development_path,
        expected_size=EXPECTED_DEVELOPMENT_RAW_SIZE_BYTES,
        label="verified Stage 6 development predictors",
    )
    _require(dev_rows == EXPECTED_DEVELOPMENT_RAW_ROWS, "development predictor rows differ")
    _require(
        dev_groups == EXPECTED_DEVELOPMENT_RAW_ROW_GROUPS,
        "development predictor row groups differ",
    )
    development_schema_hash = _schema_sha256(development_schema)
    _require(
        development_schema_hash == EXPECTED_RAW_FEATURE_SCHEMA_SHA256,
        "development predictor physical schema SHA-256 differs",
    )
    _require(development_schema.equals(frozen_schema), "development schema differs from registry")
    development_columns_sha256 = _column_order_sha256(tuple(development_schema.names))
    validation_columns_sha256 = _column_order_sha256(tuple(validation_schema.names))

    registration = RegisteredPayload(
        role=PayloadRole.VALIDATION_PREDICTOR,
        path=payload_path,
        sha256=digest,
        size_bytes=size_bytes,
        allowed_columns=frozen_columns,
        minimum_signal_date=MINIMUM_VALIDATION_SIGNAL_DATE,
        maximum_signal_date=MAXIMUM_VALIDATION_SIGNAL_DATE,
        contains_outcome_or_forward_return=False,
        trusted_schema_sha256=EXPECTED_RAW_FEATURE_SCHEMA_SHA256,
    )
    reconciliation = DevelopmentSchemaReconciliationReceipt(
        corrected_stage6_build_id=CORRECTED_STAGE6_BUILD_ID,
        stage6_verified_pointer_sha256=EXPECTED_STAGE6_VERIFIED_POINTER_SHA256,
        stage6_primary_manifest_sha256=EXPECTED_STAGE6_PRIMARY_MANIFEST_SHA256,
        feature_registry_sha256=EXPECTED_FEATURE_REGISTRY_SHA256,
        development_raw_sha256=EXPECTED_DEVELOPMENT_RAW_SHA256,
        development_raw_rows=EXPECTED_DEVELOPMENT_RAW_ROWS,
        development_raw_schema_sha256=development_schema_hash,
        validation_raw_schema_sha256=validation_schema_hash,
        raw_column_count=len(frozen_columns),
        economic_feature_count=len(frozen_columns) - 7,
        development_column_order_sha256=development_columns_sha256,
        validation_column_order_sha256=validation_columns_sha256,
        exact_column_order_match=development_schema.names == validation_schema.names,
        exact_arrow_schema_match=development_schema.equals(validation_schema),
        parquet_footer_only=True,
        outcome_or_label_payloads_opened=0,
        outcome_or_label_rows_read=0,
    )
    return AuthenticatedValidationPredictor(
        manifest_path=path,
        manifest_sha256=_sha256_bytes(manifest_payload),
        registration_name=REGISTRATION_NAME,
        registration=registration,
        payload_rows=rows,
        payload_row_groups=row_groups,
        reconciliation=reconciliation,
        validation_data_verification=validation_data_verification,
    )


def write_verified_validation_predictor_manifest(
    *,
    candidate_manifest_path: Path,
    verification_report_path: Path,
    candidate_root: Path,
    destination: Path,
) -> AuthenticatedValidationPredictor:
    """Bridge one independently verified bundle into the scoring manifest.

    The function derives the predictor receipt from the verified primary
    manifest; callers cannot provide a replacement hash, row count, schema, or
    date range.
    """

    project = PROJECT_ROOT.resolve()

    def artifact_relative(path: Path, label: str) -> str:
        lexical = Path(os.path.abspath(path.expanduser()))
        try:
            relative = lexical.relative_to(project)
        except ValueError as exc:
            raise PredictorAdapterError(f"{label} escaped the project root") from exc
        _require(
            relative.parts[:2] == ("artifacts", "phase_05_validation"),
            f"{label} escaped the Stage 11 artifact namespace",
        )
        return relative.as_posix()

    manifest_relative = artifact_relative(
        candidate_manifest_path, "validation-data candidate manifest"
    )
    report_relative = artifact_relative(
        verification_report_path, "validation-data verification report"
    )
    candidate_bytes = Path(candidate_manifest_path).read_bytes()
    report_bytes = Path(verification_report_path).read_bytes()
    candidate = _json_object(candidate_bytes, label="validation-data candidate manifest")
    report = _json_object(report_bytes, label="validation-data verification report")
    binding = {
        "build_id": report.get("build_id"),
        "verification_id": report.get("verification_id"),
        "candidate_manifest_path": manifest_relative,
        "candidate_manifest_sha256": _sha256_bytes(candidate_bytes),
        "verification_report_path": report_relative,
        "verification_report_sha256": _sha256_bytes(report_bytes),
        "materialization_code_hashes_sha256": report.get(
            "materialization_code_hashes_sha256"
        ),
        "predictor_payload_role": VALIDATION_PREDICTOR_ROLE,
        "empirical_build_id": report.get("empirical_build_id"),
        "empirical_build_receipt_path": report.get(
            "empirical_build_receipt_path"
        ),
        "empirical_build_receipt_sha256": report.get(
            "empirical_build_receipt_sha256"
        ),
        "empirical_build_content_fingerprint": report.get(
            "empirical_build_content_fingerprint"
        ),
    }
    verification = _authenticate_validation_data_verification(binding)
    receipt = verification.predictor_payload_receipt
    root = Path(os.path.abspath(candidate_root.expanduser()))
    _require(root.is_dir() and not root.is_symlink(), "Unsafe validation-data candidate root")
    _require(
        root.resolve(strict=True) == root == verification.candidate_root,
        "Candidate root differs from the authenticated empirical build",
    )
    payload_path = root / str(receipt["path"])
    payload_path, _metadata = _ordinary_file_under(
        payload_path,
        DATABASE_ROOT,
        label="verified validation predictor payload",
    )
    _require(
        _sha256_regular_file(
            payload_path,
            expected_size=int(receipt["size_bytes"]),
            label="verified validation predictor payload",
        )
        == receipt["sha256"],
        "verified validation predictor payload SHA-256 differs",
    )
    frozen_schema, _development_path = _authenticate_stage6_authority()
    relative_payload = payload_path.relative_to(DATABASE_ROOT.resolve()).as_posix()
    manifest = {
        "schema_version": 2,
        "manifest_id": MANIFEST_ID,
        "status": MANIFEST_STATUS,
        "payload": {
            "path": relative_payload,
            "sha256": receipt["sha256"],
            "size_bytes": receipt["size_bytes"],
            "rows": receipt["rows"],
            "row_groups": receipt["row_groups"],
            "schema_sha256": receipt["schema_sha256"],
            "column_count": len(frozen_schema),
            "columns_sha256": _column_order_sha256(tuple(frozen_schema.names)),
            "minimum_signal_date": receipt["minimum_signal_date"],
            "maximum_signal_date": receipt["maximum_signal_date"],
        },
        "stage06_binding": _expected_stage6_binding(),
        "validation_data_verification": binding,
    }
    output = Path(os.path.abspath(destination.expanduser()))
    database = DATABASE_ROOT.resolve()
    try:
        output.relative_to(database)
    except ValueError as exc:
        raise PredictorAdapterError("Predictor manifest escaped the database root") from exc
    _require(output.parent.is_dir() and not output.parent.is_symlink(), "Unsafe predictor manifest directory")
    try:
        with output.open("xb") as handle:
            handle.write(_canonical_json_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PredictorAdapterError("Predictor manifest destination already exists") from exc
    return authenticate_validation_predictor_manifest(output)


__all__ = [
    "AuthenticatedValidationPredictor",
    "CORRECTED_STAGE6_BUILD_ID",
    "DATABASE_ROOT",
    "DevelopmentSchemaReconciliationReceipt",
    "ValidationDataVerificationReceipt",
    "EXPECTED_FEATURE_REGISTRY_SHA256",
    "EXPECTED_RAW_FEATURE_SCHEMA_SHA256",
    "EXPECTED_RAW_COLUMN_COUNT",
    "EXPECTED_STAGE6_PRIMARY_MANIFEST_SHA256",
    "EXPECTED_STAGE6_VERIFIED_POINTER_SHA256",
    "MANIFEST_ID",
    "MANIFEST_STATUS",
    "MAXIMUM_VALIDATION_SIGNAL_DATE",
    "MINIMUM_VALIDATION_SIGNAL_DATE",
    "PredictorAdapterError",
    "REGISTRATION_NAME",
    "VALIDATION_PREDICTOR_ROLE",
    "authenticate_validation_predictor_manifest",
    "write_verified_validation_predictor_manifest",
]
