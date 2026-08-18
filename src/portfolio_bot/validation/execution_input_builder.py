"""Deterministic, no-outcome publication of the Stage 11 execution inputs.

This is the last input-only step before Gate B.  It authenticates the frozen
calendar, validation predictor, independently verified Gate-A source bundle,
development selector/OOF payload, risk/cost payloads, and corporate-action
mapping.  It then publishes one physically bounded 59-row calendar shard and
the two fixed manifests consumed by the empirical runner.  No validation
outcome value, model forecast, portfolio weight, trade, P&L value, December
2019 holding outcome, or 2020+ row is read or created here.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import portfolio_bot.validation.corporate_action_adapter as corporate_actions
import portfolio_bot.validation.execution_input_manifest as execution_inputs
import portfolio_bot.validation.predictor_adapter as predictor_adapter
from portfolio_bot.validation.access import FROZEN_STAGE6_RAW_SCHEMA_SHA256, PayloadRole
from portfolio_bot.validation.calendar import (
    FIRST_VALIDATION_ENTRY_DATE,
    FIRST_VALIDATION_SIGNAL_DATE,
    LAST_VALIDATION_EXIT_DATE,
    LAST_VALIDATION_SIGNAL_DATE,
    VALIDATION_HOLDING_MONTHS,
    ValidationCalendarRow,
    validate_validation_calendar,
)
from portfolio_bot.validation.execution_access import (
    ExecutionDateAxis,
    RegisteredExecutionPayload,
)
from portfolio_bot.validation.execution_risk_inputs import RISK_COST_REGISTRATION_NAMES
from portfolio_bot.validation.outcome_materialization import (
    CONTRACT_ID as OUTCOME_CONTRACT_ID,
)
from portfolio_bot.validation.outcome_materialization import DEFAULT_INPUT_NAMES
from portfolio_bot.validation.outcome_materialization import (
    FROZEN_INPUT_MANIFEST_PATH as OUTCOME_INPUT_MANIFEST_PATH,
)
from portfolio_bot.validation.outcome_materialization import (
    _validate_registration_contract,
)
from portfolio_bot.validation.predictor_adapter import DATABASE_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXECUTION_INPUT_MANIFEST_PATH = execution_inputs.FROZEN_EXECUTION_INPUT_MANIFEST_PATH
VERIFIED_POINTER_PATH = (
    PROJECT_ROOT
    / "artifacts/phase_05_validation/STAGE_11_EXECUTION_INPUTS_VERIFIED.json"
)
CALENDAR_SHARD_PATH = (
    DATABASE_ROOT
    / "interim/phase_05_validation/stage_11_execution_v1"
    / "outcome_inputs/validation_calendar.parquet"
)
PREDICTOR_MANIFEST_PATH = (
    DATABASE_ROOT
    / "interim/phase_05_validation/stage11-validation-data-v1"
    / "validation_predictor_manifest_f22cf90d995a1fdfd916f24f.json"
)
PREDICTOR_MANIFEST_SHA256 = (
    "e151e595e947c62551ce9dbfcea54e64e629a750d9dc1c344407075cf87ba65d"
)
SOURCE_CALENDAR_PATH = (
    DATABASE_ROOT
    / "interim/phase_02_canonical_panel/phase02-crsp-b03263a970e52fbd"
    / "calendar/calendar-ae2e4ba4a280d66e/rebalance_calendar.parquet"
)
SOURCE_CALENDAR_SHA256 = (
    "cc03002bb9c349c07f14d699bf3667fd4ccbcd012f20579b1ce1c22dd19f2591"
)
SOURCE_CALENDAR_SIZE_BYTES = 6_648
CALENDAR_VERIFIED_POINTER_PATH = (
    PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CALENDAR_VERIFIED.json"
)
CALENDAR_VERIFIED_POINTER_SHA256 = (
    "6e4377fa1cc898d676d8e44832ce6497e936707944d87ca212889500c97536a9"
)
CALENDAR_BUILD_ID = "calendar-ae2e4ba4a280d66e"
CALENDAR_VERIFICATION_ID = "verify-7dc718e9be47873c"
VALIDATION_DATA_RULES_PATH = (
    PROJECT_ROOT / "configs/phase_05_validation/stage_11_validation_data_rules_v1.json"
)
VALIDATION_DATA_RULES_SHA256 = (
    "43f52f578f5ca810e537962380272d93398d1d2734ce688262724f81b080ce1a"
)
CALENDAR_COLUMNS = ("signal_date", "holding_month", "entry_date", "exit_date")
CALENDAR_SCHEMA = pa.schema(
    [
        pa.field("signal_date", pa.date32()),
        pa.field("holding_month", pa.string()),
        pa.field("entry_date", pa.date32()),
        pa.field("exit_date", pa.date32()),
    ]
)
VERIFICATION_IDENTITY = "phase05-stage11-execution-inputs-independent-verifier-v1"
VERIFICATION_STATUS = "STAGE11_EXECUTION_INPUTS_VERIFIED"


class Stage11ExecutionInputBuildError(RuntimeError):
    """Raised when any no-outcome input authority or exact byte differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11ExecutionInputBuildError(message)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _ordinary_file(
    path: Path, *, root: Path, label: str
) -> tuple[Path, os.stat_result]:
    root = Path(os.path.abspath(root.expanduser()))
    selected = Path(os.path.abspath(path.expanduser()))
    try:
        selected.relative_to(root)
    except ValueError as exc:
        raise Stage11ExecutionInputBuildError(
            f"{label} escaped its trusted root"
        ) from exc
    _require(
        selected.resolve(strict=False) == selected, f"{label} uses a symlinked path"
    )
    try:
        metadata = selected.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise Stage11ExecutionInputBuildError(f"{label} is absent") from exc
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_nlink == 1,
        f"{label} is not an ordinary single-link file",
    )
    return selected, metadata


def _read_bound_file(
    path: Path, *, root: Path, expected_sha256: str, label: str
) -> bytes:
    selected, before = _ordinary_file(path, root=root, label=label)
    payload = selected.read_bytes()
    after = selected.stat(follow_symlinks=False)
    _require(
        _stable_identity(before) == _stable_identity(after),
        f"{label} changed while read",
    )
    _require(_sha256_bytes(payload) == expected_sha256, f"{label} SHA-256 differs")
    return payload


def _hash_bound_file(
    path: Path, *, root: Path, expected_sha256: str, expected_size: int, label: str
) -> None:
    selected, before = _ordinary_file(path, root=root, label=label)
    _require(before.st_size == expected_size, f"{label} size differs")
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    after = selected.stat(follow_symlinks=False)
    _require(
        _stable_identity(before) == _stable_identity(after),
        f"{label} changed while hashed",
    )
    _require(digest.hexdigest() == expected_sha256, f"{label} SHA-256 differs")


def _json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11ExecutionInputBuildError(f"{label} is invalid JSON") from exc
    _require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


def _safe_parent(path: Path, *, root: Path, label: str) -> Path:
    root = Path(os.path.abspath(root.expanduser()))
    selected = Path(os.path.abspath(path.expanduser()))
    try:
        selected.relative_to(root)
    except ValueError as exc:
        raise Stage11ExecutionInputBuildError(
            f"{label} escaped its fixed root"
        ) from exc
    selected.parent.mkdir(parents=True, exist_ok=True)
    _require(
        selected.parent.is_dir()
        and not selected.parent.is_symlink()
        and selected.parent.resolve(strict=False) == selected.parent,
        f"{label} parent is unsafe",
    )
    return selected


def _atomic_write(path: Path, payload: bytes, *, root: Path, label: str) -> None:
    selected = _safe_parent(path, root=root, label=label)
    temporary = selected.with_name(f".{selected.name}.tmp-{os.getpid()}")
    try:
        temporary.unlink(missing_ok=True)
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, selected)
    finally:
        temporary.unlink(missing_ok=True)
    _require(
        selected.read_bytes() == payload, f"{label} bytes differ after publication"
    )


def _authenticate_calendar_authority() -> tuple[ValidationCalendarRow, ...]:
    pointer_payload = _read_bound_file(
        CALENDAR_VERIFIED_POINTER_PATH,
        root=PROJECT_ROOT,
        expected_sha256=CALENDAR_VERIFIED_POINTER_SHA256,
        label="verified calendar pointer",
    )
    pointer = _json(pointer_payload, label="verified calendar pointer")
    _require(
        pointer_payload == canonical_json_bytes(pointer),
        "verified calendar pointer is not canonical",
    )
    _require(
        pointer.get("status") == "verified_passed"
        and pointer.get("build_id") == CALENDAR_BUILD_ID
        and pointer.get("verification_id") == CALENDAR_VERIFICATION_ID,
        "verified calendar authority differs",
    )
    rules_payload = _read_bound_file(
        VALIDATION_DATA_RULES_PATH,
        root=PROJECT_ROOT,
        expected_sha256=VALIDATION_DATA_RULES_SHA256,
        label="validation-data rules",
    )
    rules = _json(rules_payload, label="validation-data rules")
    calendar = rules.get("calendar")
    _require(
        isinstance(calendar, Mapping)
        and calendar.get("calendar_build_id") == CALENDAR_BUILD_ID
        and calendar.get("calendar_verification_id") == CALENDAR_VERIFICATION_ID
        and calendar.get("rebalance_calendar_sha256") == SOURCE_CALENDAR_SHA256
        and calendar.get("validation_stage_value") == "validation",
        "validation calendar rules differ",
    )
    _hash_bound_file(
        SOURCE_CALENDAR_PATH,
        root=DATABASE_ROOT,
        expected_sha256=SOURCE_CALENDAR_SHA256,
        expected_size=SOURCE_CALENDAR_SIZE_BYTES,
        label="verified source calendar",
    )
    # Both predicates are pushed into the scanner.  Only the 59 validation
    # metadata rows are materialized; 2020+ rows never enter Python memory.
    predicate = (pc.field("stage") == pa.scalar("validation")) & (
        pc.field("signal_date") <= pa.scalar(LAST_VALIDATION_SIGNAL_DATE)
    )
    selected = (
        ds.dataset(SOURCE_CALENDAR_PATH, format="parquet")
        .scanner(
            columns=[
                "holding_month",
                "signal_date",
                "entry_date",
                "exit_date",
                "is_embargo",
                "is_partial",
                "full_feature_time_eligible",
            ],
            filter=predicate,
            use_threads=False,
        )
        .to_table()
    )
    _require(selected.num_rows == 59, "verified calendar projection is not 59 rows")
    rows: list[ValidationCalendarRow] = []
    for value in selected.to_pylist():
        _require(
            value["is_embargo"] is False
            and value["is_partial"] is False
            and value["full_feature_time_eligible"] is True,
            "validation calendar contains an ineligible row",
        )
        rows.append(
            ValidationCalendarRow(
                holding_month=str(value["holding_month"]),
                signal_date=value["signal_date"],
                entry_date=value["entry_date"],
                exit_date=value["exit_date"],
            )
        )
    return validate_validation_calendar(rows)


def _calendar_parquet_bytes(rows: tuple[ValidationCalendarRow, ...]) -> bytes:
    table = pa.Table.from_pylist(
        [
            {
                "signal_date": row.signal_date,
                "holding_month": row.holding_month,
                "entry_date": row.entry_date,
                "exit_date": row.exit_date,
            }
            for row in rows
        ],
        schema=CALENDAR_SCHEMA,
    )
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=59,
        version="2.6",
        data_page_version="1.0",
        use_compliant_nested_type=True,
        store_schema=True,
    )
    return sink.getvalue().to_pybytes()


def _calendar_registration(payload: bytes) -> RegisteredExecutionPayload:
    record = RegisteredExecutionPayload(
        role=PayloadRole.VALIDATION_CALENDAR_METADATA,
        path=CALENDAR_SHARD_PATH,
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        allowed_columns=CALENDAR_COLUMNS,
        date_column="signal_date",
        date_axis=ExecutionDateAxis.SIGNAL,
        minimum_date=FIRST_VALIDATION_SIGNAL_DATE,
        maximum_date=LAST_VALIDATION_SIGNAL_DATE,
        holding_month_column="holding_month",
        trusted_schema_sha256=None,
    )
    parquet = pq.ParquetFile(pa.BufferReader(payload))
    _require(
        parquet.metadata.num_rows == 59
        and parquet.metadata.num_row_groups == 1
        and parquet.schema_arrow.equals(CALENDAR_SCHEMA),
        "isolated validation calendar footer differs",
    )
    date_index = parquet.schema_arrow.get_field_index("signal_date")
    stats = parquet.metadata.row_group(0).column(date_index).statistics
    _require(
        stats is not None
        and stats.has_min_max
        and stats.min == FIRST_VALIDATION_SIGNAL_DATE
        and stats.max == LAST_VALIDATION_SIGNAL_DATE,
        "isolated validation calendar is not physically bounded",
    )
    return record


def _record_mapping(record: RegisteredExecutionPayload) -> dict[str, object]:
    return {
        "role": record.role.value,
        "path": str(record.path),
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "allowed_columns": list(record.allowed_columns),
        "date_column": record.date_column,
        "date_axis": record.date_axis.value,
        "minimum_date": record.minimum_date.isoformat(),
        "maximum_date": record.maximum_date.isoformat(),
        "holding_month_column": record.holding_month_column,
        "trusted_schema_sha256": record.trusted_schema_sha256,
    }


@dataclass(frozen=True, slots=True)
class _PredictorAuthority:
    registration: RegisteredExecutionPayload
    binding: Mapping[str, object]
    payload_receipts: Mapping[str, Mapping[str, Any]]
    candidate_root: Path


def _project_authority(
    relative_text: object, expected_sha256: object, *, label: str
) -> tuple[Path, Mapping[str, Any]]:
    relative = Path(str(relative_text))
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts[:2] == ("artifacts", "phase_05_validation"),
        f"{label} path is unsafe",
    )
    payload = _read_bound_file(
        PROJECT_ROOT / relative,
        root=PROJECT_ROOT,
        expected_sha256=str(expected_sha256),
        label=label,
    )
    value = _json(payload, label=label)
    _require(payload == canonical_json_bytes(value), f"{label} is not canonical")
    return PROJECT_ROOT / relative, value


def _predictor_authority() -> _PredictorAuthority:
    manifest_payload = _read_bound_file(
        PREDICTOR_MANIFEST_PATH,
        root=DATABASE_ROOT,
        expected_sha256=PREDICTOR_MANIFEST_SHA256,
        label="validation predictor manifest",
    )
    manifest = _json(manifest_payload, label="validation predictor manifest")
    _require(
        manifest_payload == canonical_json_bytes(manifest)[:-1]
        and manifest.get("schema_version") == 2
        and manifest.get("manifest_id") == predictor_adapter.MANIFEST_ID
        and manifest.get("status") == predictor_adapter.MANIFEST_STATUS,
        "validation predictor manifest identity differs",
    )
    stage6 = manifest.get("stage06_binding")
    expected_stage6 = {
        "corrected_build_id": predictor_adapter.CORRECTED_STAGE6_BUILD_ID,
        "verification_id": predictor_adapter.STAGE6_VERIFICATION_ID,
        "verified_pointer_path": predictor_adapter.STAGE6_VERIFIED_POINTER_RELATIVE.as_posix(),
        "verified_pointer_sha256": predictor_adapter.EXPECTED_STAGE6_VERIFIED_POINTER_SHA256,
        "primary_manifest_path": predictor_adapter.STAGE6_PRIMARY_MANIFEST_RELATIVE.as_posix(),
        "primary_manifest_sha256": predictor_adapter.EXPECTED_STAGE6_PRIMARY_MANIFEST_SHA256,
        "feature_registry_path": predictor_adapter.FEATURE_REGISTRY_RELATIVE.as_posix(),
        "feature_registry_sha256": predictor_adapter.EXPECTED_FEATURE_REGISTRY_SHA256,
    }
    _require(stage6 == expected_stage6, "predictor Stage-6 binding differs")
    for path_key, sha_key in (
        ("verified_pointer_path", "verified_pointer_sha256"),
        ("primary_manifest_path", "primary_manifest_sha256"),
        ("feature_registry_path", "feature_registry_sha256"),
    ):
        _read_bound_file(
            PROJECT_ROOT / str(stage6[path_key]),
            root=PROJECT_ROOT,
            expected_sha256=str(stage6[sha_key]),
            label=f"predictor Stage-6 {path_key}",
        )

    verification = manifest.get("validation_data_verification")
    _require(
        isinstance(verification, Mapping), "predictor verification binding is absent"
    )
    _candidate_path, candidate = _project_authority(
        verification.get("candidate_manifest_path"),
        verification.get("candidate_manifest_sha256"),
        label="validation-data candidate manifest",
    )
    _report_path, report = _project_authority(
        verification.get("verification_report_path"),
        verification.get("verification_report_sha256"),
        label="validation-data independent report",
    )
    _empirical_path, empirical = _project_authority(
        verification.get("empirical_build_receipt_path"),
        verification.get("empirical_build_receipt_sha256"),
        label="validation-data empirical build receipt",
    )
    _require(
        candidate.get("status") == "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
        and candidate.get("build_id") == verification.get("build_id")
        and report.get("status") == "VERIFIED_PASSED"
        and report.get("build_id") == verification.get("build_id")
        and report.get("verification_id") == verification.get("verification_id")
        and report.get("manifest_sha256")
        == verification.get("candidate_manifest_sha256")
        and report.get("materialization_code_hashes_sha256")
        == verification.get("materialization_code_hashes_sha256")
        and report.get("difference_count") == 0
        and report.get("raw_lineage_semantic_difference_count") == 0
        and report.get("payload_count") == 12
        and report.get("signal_dates_checked_per_payload") == 59
        and report.get("validation_predictor_risk_and_cost_use_authorized") is True
        and report.get("validation_outcome_access_authorized") is False
        and report.get("real_validation_weights_trades_or_pnl_authorized") is False
        and report.get("final_period_access_authorized") is False
        and report.get("validation_outcome_rows_read") == 0
        and report.get("final_period_rows_read") == 0,
        "validation-data independent verification authority differs",
    )
    _require(
        all(value == 0 for value in candidate.get("access_counters", {}).values())
        and all(value == 0 for value in empirical.get("access_counters", {}).values()),
        "predictor authority reports a prohibited read or output",
    )
    candidate_root = Path(str(empirical.get("candidate_root", "")))
    _require(
        candidate_root.is_absolute()
        and candidate_root.resolve(strict=False) == candidate_root
        and candidate_root.is_dir()
        and not candidate_root.is_symlink(),
        "predictor candidate root is unsafe",
    )
    payload_receipts = candidate.get("payloads")
    _require(
        isinstance(payload_receipts, Mapping)
        and len(payload_receipts) == 12
        and predictor_adapter.VALIDATION_PREDICTOR_ROLE in payload_receipts,
        "predictor verified payload set differs",
    )
    _require(
        {path.name for path in candidate_root.iterdir()}
        == {str(receipt["path"]) for receipt in payload_receipts.values()},
        "predictor candidate payload directory is not closed",
    )
    predictor_receipt = payload_receipts[predictor_adapter.VALIDATION_PREDICTOR_ROLE]
    raw = manifest.get("payload")
    _require(
        isinstance(raw, Mapping)
        and raw.get("sha256") == predictor_receipt.get("sha256")
        and raw.get("size_bytes") == predictor_receipt.get("size_bytes")
        and raw.get("rows") == predictor_receipt.get("rows")
        and raw.get("row_groups") == predictor_receipt.get("row_groups")
        and raw.get("schema_sha256") == FROZEN_STAGE6_RAW_SCHEMA_SHA256
        and raw.get("minimum_signal_date") == "2014-12-31"
        and raw.get("maximum_signal_date") == "2019-10-31"
        and raw.get("column_count") == 170,
        "predictor payload receipt differs",
    )
    predictor_path = DATABASE_ROOT / str(raw["path"])
    _require(
        predictor_path == candidate_root / str(predictor_receipt["path"]),
        "predictor path differs from its verified candidate",
    )
    _hash_bound_file(
        predictor_path,
        root=DATABASE_ROOT,
        expected_sha256=str(raw["sha256"]),
        expected_size=int(raw["size_bytes"]),
        label="verified validation predictor",
    )
    parquet = pq.ParquetFile(predictor_path)
    schema = parquet.schema_arrow
    _require(
        parquet.metadata.num_rows == raw["rows"]
        and parquet.metadata.num_row_groups == raw["row_groups"]
        and len(schema) == 170
        and _sha256_bytes(schema.serialize().to_pybytes())
        == FROZEN_STAGE6_RAW_SCHEMA_SHA256,
        "verified predictor footer differs",
    )
    columns = tuple(schema.names)
    columns_payload = json.dumps(
        list(columns), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    _require(
        _sha256_bytes(columns_payload) == raw.get("columns_sha256"),
        "verified predictor column order differs",
    )
    date_index = schema.get_field_index("signal_date")
    minima: list[dt.date] = []
    maxima: list[dt.date] = []
    for index in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(index).column(date_index).statistics
        _require(
            stats is not None and stats.has_min_max,
            "predictor date statistics are absent",
        )
        minima.append(stats.min)
        maxima.append(stats.max)
    _require(
        min(minima) == FIRST_VALIDATION_SIGNAL_DATE
        and max(maxima) == LAST_VALIDATION_SIGNAL_DATE,
        "verified predictor date scope differs",
    )
    registration = RegisteredExecutionPayload(
        role=PayloadRole.VALIDATION_PREDICTOR,
        path=predictor_path,
        sha256=str(raw["sha256"]),
        size_bytes=int(raw["size_bytes"]),
        allowed_columns=columns,
        date_column="signal_date",
        date_axis=ExecutionDateAxis.SIGNAL,
        minimum_date=FIRST_VALIDATION_SIGNAL_DATE,
        maximum_date=LAST_VALIDATION_SIGNAL_DATE,
        holding_month_column=None,
        trusted_schema_sha256=FROZEN_STAGE6_RAW_SCHEMA_SHA256,
    )
    binding = MappingProxyType(
        {
            "manifest_path": str(PREDICTOR_MANIFEST_PATH),
            "manifest_sha256": PREDICTOR_MANIFEST_SHA256,
            "validation_data_build_id": verification["build_id"],
            "validation_data_verification_id": verification["verification_id"],
            "candidate_manifest_sha256": verification["candidate_manifest_sha256"],
            "verification_report_sha256": verification["verification_report_sha256"],
            "materialization_code_hashes_sha256": verification[
                "materialization_code_hashes_sha256"
            ],
            "payload_sha256": raw["sha256"],
            "payload_schema_sha256": FROZEN_STAGE6_RAW_SCHEMA_SHA256,
            "payload_rows": raw["rows"],
        }
    )
    return _PredictorAuthority(
        registration=registration,
        binding=binding,
        payload_receipts=MappingProxyType(
            {
                name: MappingProxyType(dict(value))
                for name, value in payload_receipts.items()
            }
        ),
        candidate_root=candidate_root,
    )


def _normalized_source_authority() -> (
    tuple[dict[str, object], Mapping[str, Mapping[str, Any]]]
):
    primary_path = PROJECT_ROOT / execution_inputs.SOURCE_PRIMARY_POINTER_RELATIVE_PATH
    verified_path = (
        PROJECT_ROOT / execution_inputs.SOURCE_VERIFIED_POINTER_RELATIVE_PATH
    )
    primary_payload = primary_path.read_bytes()
    verified_payload = verified_path.read_bytes()
    primary = _json(primary_payload, label="Gate-A primary pointer")
    verified = _json(verified_payload, label="Gate-A verified pointer")
    plan_payload = _read_bound_file(
        PROJECT_ROOT / execution_inputs.SOURCE_PLAN_RELATIVE_PATH,
        root=PROJECT_ROOT,
        expected_sha256=execution_inputs.SOURCE_PLAN_SHA256,
        label="Gate-A source plan",
    )
    plan = _json(plan_payload, label="Gate-A source plan")
    publication = plan.get("publication")
    build_id = primary.get("build_id")
    _require(
        isinstance(publication, Mapping)
        and isinstance(build_id, str)
        and execution_inputs._SOURCE_BUILD_ID.fullmatch(build_id) is not None,
        "Gate-A bundle authority is malformed",
    )
    primary_root = Path(str(publication.get("primary_bundle_root", "")))
    expected_bundle = primary_root / build_id
    declared_bundle = Path(str(primary.get("bundle_path", "")))
    _require(
        primary_root.is_absolute()
        and primary_root.resolve(strict=False) == primary_root
        and declared_bundle == expected_bundle
        and declared_bundle.resolve(strict=False) == declared_bundle
        and declared_bundle.is_dir()
        and not declared_bundle.is_symlink(),
        "Gate-A primary bundle escaped its fixed plan root",
    )
    primary_manifest_path = declared_bundle / "manifest.json"
    primary_manifest_payload = primary_manifest_path.read_bytes()
    primary_manifest = _json(primary_manifest_payload, label="Gate-A primary manifest")
    raw_records = primary_manifest.get("execution_input_records")
    _require(isinstance(raw_records, Mapping), "Gate-A execution records are absent")
    binding: dict[str, object] = {
        "plan_path": execution_inputs.SOURCE_PLAN_RELATIVE_PATH,
        "plan_sha256": execution_inputs.SOURCE_PLAN_SHA256,
        "plan_payload_sha256": execution_inputs.SOURCE_PLAN_PAYLOAD_SHA256,
        "primary_pointer_path": execution_inputs.SOURCE_PRIMARY_POINTER_RELATIVE_PATH,
        "primary_pointer_sha256": _sha256_bytes(primary_payload),
        "verified_pointer_path": execution_inputs.SOURCE_VERIFIED_POINTER_RELATIVE_PATH,
        "verified_pointer_sha256": _sha256_bytes(verified_payload),
        "source_authority_sha256": primary.get("authority_sha256"),
        "primary_build_id": primary.get("build_id"),
        "primary_manifest_sha256": _sha256_bytes(primary_manifest_payload),
        "verification_id": verified.get("verification_id"),
        "verification_report_sha256": verified.get("verification_report_sha256"),
        "execution_input_records_sha256": _sha256_bytes(
            canonical_json_bytes(raw_records)
        ),
    }
    authority = execution_inputs._authenticate_normalized_sources(binding)
    for name in execution_inputs.NORMALIZED_SOURCE_NAMES:
        _require(
            _record_mapping(authority.records[name]) == raw_records[name],
            f"authenticated Gate-A record differs: {name}",
        )
    return binding, MappingProxyType(dict(raw_records))


def _authenticate_risk_cost(predictor: _PredictorAuthority) -> None:
    for name in RISK_COST_REGISTRATION_NAMES:
        receipt = predictor.payload_receipts[name]
        path = predictor.candidate_root / str(receipt["path"])
        _require(
            path.parent == predictor.candidate_root
            and receipt.get("role") == name
            and receipt.get("minimum_signal_date") == "2014-12-31"
            and receipt.get("maximum_signal_date") == "2019-10-31",
            f"verified risk/cost receipt scope differs: {name}",
        )
        _hash_bound_file(
            path,
            root=DATABASE_ROOT,
            expected_sha256=str(receipt["sha256"]),
            expected_size=int(receipt["size_bytes"]),
            label=f"verified risk/cost payload {name}",
        )
        parquet = pq.ParquetFile(path)
        _require(
            parquet.metadata.num_rows == receipt["rows"]
            and _sha256_bytes(parquet.schema_arrow.serialize().to_pybytes())
            == receipt["schema_sha256"],
            f"verified risk/cost footer differs: {name}",
        )


def _selector_binding() -> dict[str, object]:
    binding: dict[str, object] = {
        "verified_pointer_path": execution_inputs.SELECTOR_VERIFIED_POINTER_RELATIVE_PATH,
        "verified_pointer_sha256": execution_inputs.SELECTOR_VERIFIED_POINTER_SHA256,
        "primary_manifest_path": execution_inputs.SELECTOR_PRIMARY_MANIFEST_RELATIVE_PATH,
        "primary_manifest_sha256": execution_inputs.SELECTOR_PRIMARY_MANIFEST_SHA256,
        "build_id": execution_inputs.SELECTOR_BUILD_ID,
        "verification_id": execution_inputs.SELECTOR_VERIFICATION_ID,
        "input_receipts_sha256": execution_inputs.SELECTOR_INPUT_RECEIPTS_SHA256,
        "oof_payload_sha256": execution_inputs.DEVELOPMENT_OOF_SHA256,
        "oof_schema_sha256": execution_inputs.DEVELOPMENT_OOF_SCHEMA_SHA256,
        "oof_size_bytes": execution_inputs.DEVELOPMENT_OOF_SIZE_BYTES,
        "oof_rows": execution_inputs.DEVELOPMENT_OOF_ROWS,
    }
    execution_inputs._authenticate_development_oof(binding)
    return binding


def _authenticate_corporate_action_mapping() -> dict[str, object]:
    expected = corporate_actions.FROZEN_MAPPING_MANIFEST_SHA256
    _require(isinstance(expected, str), "corporate-action mapping remains unfrozen")
    payload = _read_bound_file(
        corporate_actions.FROZEN_MAPPING_MANIFEST_PATH,
        root=PROJECT_ROOT,
        expected_sha256=expected,
        label="corporate-action mapping",
    )
    manifest = _json(payload, label="corporate-action mapping")
    _require(
        payload == canonical_json_bytes(manifest),
        "corporate-action mapping is not canonical",
    )
    _require(
        manifest.get("manifest_id") == corporate_actions.MAPPING_MANIFEST_ID
        and manifest.get("status") == corporate_actions.MAPPING_MANIFEST_STATUS
        and manifest.get("match_fields") == list(corporate_actions.MATCH_FIELDS),
        "corporate-action mapping identity differs",
    )
    without_digest = dict(manifest)
    supplied_payload = without_digest.pop("payload_sha256", None)
    _require(
        supplied_payload == _sha256_bytes(canonical_json_bytes(without_digest)),
        "corporate-action mapping payload digest differs",
    )
    _require(
        manifest.get("rules_sha256")
        == _sha256_bytes(canonical_json_bytes(manifest.get("rules"))),
        "corporate-action mapping rule digest differs",
    )
    return {
        "mapping_manifest_path": corporate_actions.MAPPING_MANIFEST_RELATIVE_PATH,
        "mapping_manifest_sha256": expected,
        "mapping_manifest_id": corporate_actions.MAPPING_MANIFEST_ID,
    }


def _outcome_manifest(
    calendar_record: RegisteredExecutionPayload,
    predictor_record: RegisteredExecutionPayload,
    source_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, object], bytes]:
    inputs: dict[str, object] = {
        DEFAULT_INPUT_NAMES.calendar: _record_mapping(calendar_record),
        DEFAULT_INPUT_NAMES.predictor: _record_mapping(predictor_record),
    }
    inputs.update(
        {
            name: dict(source_records[name])
            for name in execution_inputs.NORMALIZED_SOURCE_NAMES
        }
    )
    output_root = (
        DATABASE_ROOT
        / "interim/phase_05_validation/stage_11_execution_v1/outcome_tables"
    )
    manifest: dict[str, object] = {
        "contract_id": OUTCOME_CONTRACT_ID,
        "inputs": inputs,
        "output_root": str(output_root),
    }
    payload = canonical_json_bytes(manifest)
    parsed, declared_root = execution_inputs._parse_authenticated_outcome_manifest(
        payload
    )
    _validate_registration_contract(parsed, DEFAULT_INPUT_NAMES)
    _require(declared_root == output_root, "outcome output root differs")
    return manifest, payload


def _execution_manifest(
    *,
    predictor_binding: Mapping[str, object],
    source_binding: Mapping[str, object],
    outcome_sha256: str,
    selector_binding: Mapping[str, object],
    corporate_binding: Mapping[str, object],
) -> tuple[dict[str, object], bytes]:
    base = DATABASE_ROOT / "interim/phase_05_validation/stage_11_execution_v1"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": execution_inputs.EXECUTION_INPUT_MANIFEST_ID,
        "status": execution_inputs.EXECUTION_INPUT_MANIFEST_STATUS,
        "scope": {
            "first_signal_date": "2014-12-31",
            "last_signal_date": "2019-10-31",
            "last_exit_date": "2019-12-02",
            "first_holding_month": VALIDATION_HOLDING_MONTHS[0],
            "last_holding_month": VALIDATION_HOLDING_MONTHS[-1],
            "holding_month_count": len(VALIDATION_HOLDING_MONTHS),
            "embargoed_holding_month": "2019-12",
            "first_prohibited_final_date": "2020-01-02",
        },
        "predictor_authority": dict(predictor_binding),
        "normalized_sources": dict(source_binding),
        "outcome_materialization": {
            "input_manifest_path": OUTCOME_INPUT_MANIFEST_PATH.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "input_manifest_sha256": outcome_sha256,
            "contract_id": OUTCOME_CONTRACT_ID,
        },
        "development_selector": dict(selector_binding),
        "corporate_actions": dict(corporate_binding),
        "output_roots": {
            "outcome_tables": str(base / "outcome_tables"),
            "selector_checkpoints": str(base / "selector_checkpoints"),
            "raw_forecasts": str(base / "raw_forecasts"),
            "execution_results": str(base / "execution_results"),
        },
    }
    manifest["payload_sha256"] = _sha256_bytes(canonical_json_bytes(manifest))
    execution_inputs._validate_manifest(manifest)
    return manifest, canonical_json_bytes(manifest)


@dataclass(frozen=True, slots=True)
class Stage11ExecutionInputPublication:
    calendar_path: Path
    calendar_sha256: str
    calendar_size_bytes: int
    calendar_rows: int
    outcome_manifest_path: Path
    outcome_manifest_sha256: str
    execution_manifest_path: Path
    execution_manifest_sha256: str
    execution_manifest_payload_sha256: str
    source_build_id: str
    source_verification_id: str
    predictor_build_id: str
    predictor_verification_id: str


def build_stage11_execution_inputs() -> Stage11ExecutionInputPublication:
    """Authenticate and publish the exact Gate-B input manifests, without Gate B."""

    rows = _authenticate_calendar_authority()
    calendar_payload = _calendar_parquet_bytes(rows)
    calendar_record = _calendar_registration(calendar_payload)
    predictor = _predictor_authority()
    predictor_binding = predictor.binding
    predictor_record = predictor.registration
    source_binding, source_records = _normalized_source_authority()
    _authenticate_risk_cost(predictor)
    selector_binding = _selector_binding()
    corporate_binding = _authenticate_corporate_action_mapping()
    outcome_manifest, outcome_payload = _outcome_manifest(
        calendar_record, predictor_record, source_records
    )
    outcome_sha = _sha256_bytes(outcome_payload)
    execution_manifest, execution_payload = _execution_manifest(
        predictor_binding=predictor_binding,
        source_binding=source_binding,
        outcome_sha256=outcome_sha,
        selector_binding=selector_binding,
        corporate_binding=corporate_binding,
    )
    # Publish only after every upstream authority and both complete byte sets
    # have passed.  Calendar first, execution manifest last.
    _atomic_write(
        CALENDAR_SHARD_PATH,
        calendar_payload,
        root=DATABASE_ROOT,
        label="isolated validation calendar",
    )
    _atomic_write(
        OUTCOME_INPUT_MANIFEST_PATH,
        outcome_payload,
        root=PROJECT_ROOT,
        label="outcome input manifest",
    )
    _atomic_write(
        EXECUTION_INPUT_MANIFEST_PATH,
        execution_payload,
        root=PROJECT_ROOT,
        label="execution input manifest",
    )
    return Stage11ExecutionInputPublication(
        calendar_path=CALENDAR_SHARD_PATH,
        calendar_sha256=calendar_record.sha256,
        calendar_size_bytes=calendar_record.size_bytes,
        calendar_rows=len(rows),
        outcome_manifest_path=OUTCOME_INPUT_MANIFEST_PATH,
        outcome_manifest_sha256=outcome_sha,
        execution_manifest_path=EXECUTION_INPUT_MANIFEST_PATH,
        execution_manifest_sha256=_sha256_bytes(execution_payload),
        execution_manifest_payload_sha256=str(execution_manifest["payload_sha256"]),
        source_build_id=str(source_binding["primary_build_id"]),
        source_verification_id=str(source_binding["verification_id"]),
        predictor_build_id=str(predictor_binding["validation_data_build_id"]),
        predictor_verification_id=str(
            predictor_binding["validation_data_verification_id"]
        ),
    )


def _independent_calendar_parquet_bytes(
    rows: tuple[ValidationCalendarRow, ...],
) -> bytes:
    """Verifier-owned byte reconstruction; deliberately separate from the builder."""

    arrays = [
        pa.array([value.signal_date for value in rows], type=pa.date32()),
        pa.array([value.holding_month for value in rows], type=pa.string()),
        pa.array([value.entry_date for value in rows], type=pa.date32()),
        pa.array([value.exit_date for value in rows], type=pa.date32()),
    ]
    table = pa.Table.from_arrays(arrays, schema=CALENDAR_SCHEMA)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=59,
        version="2.6",
        data_page_version="1.0",
        use_compliant_nested_type=True,
        store_schema=True,
    )
    return sink.getvalue().to_pybytes()


def _independent_outcome_manifest_bytes(
    calendar_record: RegisteredExecutionPayload,
    predictor_record: RegisteredExecutionPayload,
    source_records: Mapping[str, Mapping[str, Any]],
) -> bytes:
    """Verifier-owned reconstruction of the exact three-field outcome manifest."""

    records: dict[str, object] = {}
    records[DEFAULT_INPUT_NAMES.calendar] = _record_mapping(calendar_record)
    records[DEFAULT_INPUT_NAMES.predictor] = _record_mapping(predictor_record)
    for name in execution_inputs.NORMALIZED_SOURCE_NAMES:
        records[name] = dict(source_records[name])
    value = {
        "contract_id": OUTCOME_CONTRACT_ID,
        "inputs": records,
        "output_root": str(
            DATABASE_ROOT
            / "interim/phase_05_validation/stage_11_execution_v1/outcome_tables"
        ),
    }
    payload = canonical_json_bytes(value)
    parsed, _root = execution_inputs._parse_authenticated_outcome_manifest(payload)
    _validate_registration_contract(parsed, DEFAULT_INPUT_NAMES)
    return payload


def _independent_execution_manifest_bytes(
    *,
    predictor_binding: Mapping[str, object],
    source_binding: Mapping[str, object],
    outcome_sha256: str,
    selector_binding: Mapping[str, object],
    corporate_binding: Mapping[str, object],
) -> tuple[Mapping[str, object], bytes]:
    """Verifier-owned reconstruction of the closed execution manifest."""

    root = DATABASE_ROOT / "interim/phase_05_validation/stage_11_execution_v1"
    value: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": execution_inputs.EXECUTION_INPUT_MANIFEST_ID,
        "status": execution_inputs.EXECUTION_INPUT_MANIFEST_STATUS,
        "scope": {
            "first_signal_date": "2014-12-31",
            "last_signal_date": "2019-10-31",
            "last_exit_date": "2019-12-02",
            "first_holding_month": "2015-01",
            "last_holding_month": "2019-11",
            "holding_month_count": 59,
            "embargoed_holding_month": "2019-12",
            "first_prohibited_final_date": "2020-01-02",
        },
        "predictor_authority": dict(predictor_binding),
        "normalized_sources": dict(source_binding),
        "outcome_materialization": {
            "input_manifest_path": (
                "artifacts/phase_05_validation/STAGE_11_OUTCOME_INPUTS.json"
            ),
            "input_manifest_sha256": outcome_sha256,
            "contract_id": OUTCOME_CONTRACT_ID,
        },
        "development_selector": dict(selector_binding),
        "corporate_actions": dict(corporate_binding),
        "output_roots": {
            "outcome_tables": str(root / "outcome_tables"),
            "selector_checkpoints": str(root / "selector_checkpoints"),
            "raw_forecasts": str(root / "raw_forecasts"),
            "execution_results": str(root / "execution_results"),
        },
    }
    value["payload_sha256"] = _sha256_bytes(canonical_json_bytes(value))
    execution_inputs._validate_manifest(value)
    return MappingProxyType(value), canonical_json_bytes(value)


def _verification_receipt(
    publication: Stage11ExecutionInputPublication,
) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": 1,
        "verification_identity": VERIFICATION_IDENTITY,
        "status": VERIFICATION_STATUS,
        "stage": 11,
        "calendar_path": str(publication.calendar_path),
        "calendar_sha256": publication.calendar_sha256,
        "calendar_size_bytes": publication.calendar_size_bytes,
        "calendar_rows": publication.calendar_rows,
        "outcome_manifest_path": OUTCOME_INPUT_MANIFEST_PATH.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "outcome_manifest_sha256": publication.outcome_manifest_sha256,
        "execution_manifest_path": EXECUTION_INPUT_MANIFEST_PATH.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "execution_manifest_sha256": publication.execution_manifest_sha256,
        "execution_manifest_payload_sha256": publication.execution_manifest_payload_sha256,
        "source_build_id": publication.source_build_id,
        "source_verification_id": publication.source_verification_id,
        "predictor_build_id": publication.predictor_build_id,
        "predictor_verification_id": publication.predictor_verification_id,
        "checks": {
            "calendar_source_authority_reauthenticated": True,
            "calendar_exact_59_rows_reconstructed": True,
            "calendar_physically_bounded_before_2020": True,
            "predictor_independent_verification_reauthenticated": True,
            "gate_a_ten_source_chain_reauthenticated": True,
            "development_selector_oof_reauthenticated": True,
            "risk_cost_authority_reauthenticated": True,
            "corporate_action_mapping_reauthenticated": True,
            "outcome_manifest_bytes_independently_reproduced": True,
            "execution_manifest_bytes_independently_reproduced": True,
        },
        "access_counters": {
            "validation_outcome_rows_read": 0,
            "final_2020_2025_rows_read": 0,
            "forecasts_weights_trades_pnl_rows_created": 0,
        },
        "gate_b_execution_authorized": False,
    }
    base["verification_id"] = (
        "verify-stage11-execution-inputs-"
        + _sha256_bytes(canonical_json_bytes(base))[:20]
    )
    return base


def verify_stage11_execution_inputs() -> Mapping[str, object]:
    """Independently reconstruct and byte-compare all published input artifacts."""

    # Reauthenticate every authority in a fresh call path and independently
    # reconstruct all expected artifact bytes.
    rows = _authenticate_calendar_authority()
    expected_calendar = _independent_calendar_parquet_bytes(rows)
    calendar_record = _calendar_registration(expected_calendar)
    predictor = _predictor_authority()
    predictor_binding = predictor.binding
    predictor_record = predictor.registration
    source_binding, source_records = _normalized_source_authority()
    _authenticate_risk_cost(predictor)
    selector_binding = _selector_binding()
    corporate_binding = _authenticate_corporate_action_mapping()

    # The verifier constructs these dicts independently from the published
    # JSON.  Published bytes are not used as inputs to expected-byte creation.
    expected_outcome = _independent_outcome_manifest_bytes(
        calendar_record, predictor_record, source_records
    )
    expected_outcome_sha = _sha256_bytes(expected_outcome)
    expected_execution_manifest, expected_execution = (
        _independent_execution_manifest_bytes(
            predictor_binding=predictor_binding,
            source_binding=source_binding,
            outcome_sha256=expected_outcome_sha,
            selector_binding=selector_binding,
            corporate_binding=corporate_binding,
        )
    )
    actual_calendar = _read_bound_file(
        CALENDAR_SHARD_PATH,
        root=DATABASE_ROOT,
        expected_sha256=_sha256_bytes(expected_calendar),
        label="published isolated validation calendar",
    )
    actual_outcome = _read_bound_file(
        OUTCOME_INPUT_MANIFEST_PATH,
        root=PROJECT_ROOT,
        expected_sha256=expected_outcome_sha,
        label="published outcome input manifest",
    )
    expected_execution_sha = _sha256_bytes(expected_execution)
    actual_execution = _read_bound_file(
        EXECUTION_INPUT_MANIFEST_PATH,
        root=PROJECT_ROOT,
        expected_sha256=expected_execution_sha,
        label="published execution input manifest",
    )
    _require(actual_calendar == expected_calendar, "calendar bytes differ")
    _require(actual_outcome == expected_outcome, "outcome manifest bytes differ")
    _require(actual_execution == expected_execution, "execution manifest bytes differ")
    publication = Stage11ExecutionInputPublication(
        calendar_path=CALENDAR_SHARD_PATH,
        calendar_sha256=_sha256_bytes(expected_calendar),
        calendar_size_bytes=len(expected_calendar),
        calendar_rows=len(rows),
        outcome_manifest_path=OUTCOME_INPUT_MANIFEST_PATH,
        outcome_manifest_sha256=expected_outcome_sha,
        execution_manifest_path=EXECUTION_INPUT_MANIFEST_PATH,
        execution_manifest_sha256=expected_execution_sha,
        execution_manifest_payload_sha256=str(
            expected_execution_manifest["payload_sha256"]
        ),
        source_build_id=str(source_binding["primary_build_id"]),
        source_verification_id=str(source_binding["verification_id"]),
        predictor_build_id=str(predictor_binding["validation_data_build_id"]),
        predictor_verification_id=str(
            predictor_binding["validation_data_verification_id"]
        ),
    )
    receipt = _verification_receipt(publication)
    receipt_payload = canonical_json_bytes(receipt)
    _atomic_write(
        VERIFIED_POINTER_PATH,
        receipt_payload,
        root=PROJECT_ROOT,
        label="execution-input verified pointer",
    )
    return MappingProxyType(
        {
            **receipt,
            "verified_pointer_sha256": _sha256_bytes(receipt_payload),
        }
    )


__all__ = [
    "CALENDAR_SHARD_PATH",
    "EXECUTION_INPUT_MANIFEST_PATH",
    "OUTCOME_INPUT_MANIFEST_PATH",
    "PREDICTOR_MANIFEST_PATH",
    "Stage11ExecutionInputBuildError",
    "Stage11ExecutionInputPublication",
    "VERIFIED_POINTER_PATH",
    "build_stage11_execution_inputs",
    "canonical_json_bytes",
    "verify_stage11_execution_inputs",
]
