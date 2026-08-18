"""Atomic construction of the 32 frozen Stage 13 input registrations.

This module deliberately separates two contracts which must not be confused:

* the *normalized source bundle* is the output of the raw-to-normalized final
  period formulas; and
* the *registration bundle* is the exact 32-file projection consumed by the
  Stage 13 access broker.

The projection implemented here is complete, deterministic, content addressed,
and independently re-opened before publication.  The public entry point has no
path, schema, reader, callback, or table arguments.  It authenticates the
verified Stage 12 binding and the repository-fixed Stage 13 approval before it
can open either a normalized source payload or an already memoized final input.

The companion raw-to-normalized publisher is separately authority gated and
publishes its verified pointer last.  This constructor reuses that pointer
when present and invokes the sealed publisher only when it is absent.  A
private seam exercises the projection/publication state machine with synthetic
Arrow tables without granting the public entry point a caller-supplied table
surface.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import numpy as np

from portfolio_bot.final_test.authorization import (
    Stage12VerifiedBinding,
    Stage13ApprovalReceipt,
    Stage13AuthorizationError,
    load_authenticated_stage12_binding,
    require_stage13_unblind_authorization,
)
from portfolio_bot.final_test.calendar import (
    CORE_COMPLETE_HOLDING_MONTHS,
    FINAL_COMPLETE_HOLDING_MONTHS,
    FINAL_HOLDING_MONTHS,
    PARTIAL_DECEMBER_HOLDING_MONTH,
    PRIMARY_FULL_HOLDING_MONTHS,
    final_holding_month_descriptor,
    shift_month,
)
from portfolio_bot.final_test.input_materialization import (
    FINAL_INPUT_NAMESPACE,
    VERIFIED_REGISTRATION_POINTER,
    FinalInputMaterializationError,
    _configured_database_root,
    _independently_derived_schemas,
    _inspect_parquet,
    publish_verified_final_input_registration,
)
from portfolio_bot.risk_cost.contract import FACTOR_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGE13_APPROVAL_PATH = Path("artifacts/phase_06_final_test/STAGE_13_APPROVAL.json")
NORMALIZED_SOURCE_POINTER = Path(
    "artifacts/phase_06_final_test/LATEST_STAGE_13_NORMALIZED_SOURCE_VERIFIED.json"
)
NORMALIZED_SOURCE_MANIFEST_NAMESPACE = Path(
    "artifacts/phase_06_final_test/stage_13_normalized_source_manifests"
)
NORMALIZED_SOURCE_NAMESPACE = Path(
    "interim/phase_06_final_test/stage13-normalized-sources-v2"
)
CONSTRUCTION_NAMESPACE = Path(
    "artifacts/phase_06_final_test/stage_13_input_constructions"
)
CONSTRUCTION_POINTER = Path(
    "artifacts/phase_06_final_test/LATEST_STAGE_13_INPUT_CONSTRUCTION.json"
)
CONSTRUCTION_POINTER_HISTORY_NAMESPACE = Path(
    "artifacts/phase_06_final_test/stage_13_input_construction_pointer_history"
)
REGISTRATION_POINTER_HISTORY_NAMESPACE = Path(
    "artifacts/phase_06_final_test/stage_13_registration_pointer_history"
)

CONSTRUCTOR_ID = "stage13-final-input-constructor-v1"
RAW_TO_NORMALIZED_IMPLEMENTATION_STATUS = (
    "IMPLEMENTED_AUTHENTICATED_FINAL_RAW_TO_NORMALIZED_SOURCE_PUBLISHER"
)
CONSTRUCTION_STATUS = "STAGE13_FINAL_INPUTS_CONSTRUCTED_AND_VERIFIED"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_BUNDLE_ID = re.compile(r"^stage13-normalized-sources-[0-9a-f]{24}$")
_CONSTRUCTION_ID = re.compile(r"^stage13-input-construction-[0-9a-f]{24}$")
_REGISTRATION_MANIFEST_ID = re.compile(r"^stage13-inputs-[0-9a-f]{24}$")
_TEST_SOURCE_TOKEN = object()
_SOURCE_SEAL = object()

_SOURCE_NAMES = (
    "final_calendar",
    "final_predictors",
    "final_risk_cost",
    "final_factor_covariance",
    "final_risk_cost_month_summary",
    "final_market_inputs",
    "final_market_accounting",
    "final_factors",
    "final_benchmarks",
    "final_completed_labels",
    "final_outcomes",
    "partial_descriptive_outcome",
)

_SOURCE_MONTHS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "final_calendar": FINAL_HOLDING_MONTHS,
        "final_predictors": FINAL_HOLDING_MONTHS,
        "final_risk_cost": FINAL_HOLDING_MONTHS,
        "final_factor_covariance": FINAL_HOLDING_MONTHS,
        "final_risk_cost_month_summary": FINAL_HOLDING_MONTHS,
        "final_market_inputs": FINAL_HOLDING_MONTHS,
        "final_market_accounting": FINAL_HOLDING_MONTHS,
        "final_factors": FINAL_COMPLETE_HOLDING_MONTHS,
        "final_benchmarks": FINAL_HOLDING_MONTHS,
        "final_completed_labels": FINAL_COMPLETE_HOLDING_MONTHS,
        "final_outcomes": FINAL_COMPLETE_HOLDING_MONTHS,
        "partial_descriptive_outcome": (PARTIAL_DECEMBER_HOLDING_MONTH,),
    }
)

_OUTPUT_MONTHS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "final_calendar": FINAL_HOLDING_MONTHS,
        **{
            f"primary_{suffix}": PRIMARY_FULL_HOLDING_MONTHS
            for suffix in (
                "predictors",
                "risk_cost",
                "cost_inputs",
                "factor_covariance",
                "risk_cost_month_summary",
                "market_inputs",
                "market_accounting",
                "factors",
                "benchmarks",
                "completed_labels",
                "outcomes",
            )
        },
        **{
            f"core_{suffix}": CORE_COMPLETE_HOLDING_MONTHS
            for suffix in (
                "predictors",
                "risk_cost",
                "cost_inputs",
                "factor_covariance",
                "risk_cost_month_summary",
                "market_inputs",
                "market_accounting",
                "factors",
                "benchmarks",
                "completed_labels",
                "outcomes",
            )
        },
        **{
            f"partial_{suffix}": (PARTIAL_DECEMBER_HOLDING_MONTH,)
            for suffix in (
                "predictors",
                "risk_cost",
                "cost_inputs",
                "factor_covariance",
                "risk_cost_month_summary",
                "market_inputs",
                "market_accounting",
                "benchmarks",
            )
        },
        "partial_descriptive_outcome": (PARTIAL_DECEMBER_HOLDING_MONTH,),
    }
)


class FinalInputConstructionError(RuntimeError):
    """Raised when a final-input build is incomplete, unsafe, or unbound."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalInputConstructionError(message)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FinalInputConstructionError(
            "construction metadata is not finite canonical JSON"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _schema_sha256(value: pa.Schema) -> str:
    return hashlib.sha256(value.serialize().to_pybytes()).hexdigest()


def _physical_schema(schema: pa.Schema) -> pa.Schema:
    fields: list[pa.Field] = []
    for field in schema:
        kind = field.type
        if pa.types.is_list(kind):
            kind = pa.list_(
                pa.field(
                    "element",
                    kind.value_type,
                    nullable=kind.value_field.nullable,
                )
            )
        fields.append(pa.field(field.name, kind, nullable=field.nullable))
    return pa.schema(fields)


def _source_schemas(project_root: Path) -> Mapping[str, pa.Schema]:
    derived = _independently_derived_schemas(project_root)
    result = {
        "final_calendar": derived["final_calendar"],
        "final_predictors": derived["primary_predictors"],
        "final_risk_cost": derived["primary_risk_cost"],
        "final_factor_covariance": derived["primary_factor_covariance"],
        "final_risk_cost_month_summary": derived["primary_risk_cost_month_summary"],
        "final_market_inputs": derived["primary_market_inputs"],
        "final_market_accounting": derived["primary_market_accounting"],
        "final_factors": derived["primary_factors"],
        "final_benchmarks": derived["primary_benchmarks"],
        "final_completed_labels": derived["primary_completed_labels"],
        "final_outcomes": derived["primary_outcomes"],
        "partial_descriptive_outcome": derived["partial_descriptive_outcome"],
    }
    _require(tuple(result) == _SOURCE_NAMES, "normalized source set differs")
    return MappingProxyType(result)


def _binding_mapping(binding: Stage12VerifiedBinding) -> dict[str, object]:
    return {**binding.to_mapping(), "stage12_binding_sha256": binding.binding_sha256}


def _stable_file_bytes(
    path: Path,
    *,
    trusted_root: Path,
    label: str,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    root = Path(os.path.abspath(trusted_root.expanduser()))
    selected = Path(os.path.abspath(path.expanduser()))
    try:
        relative = selected.relative_to(root)
    except ValueError as exc:
        raise FinalInputConstructionError(f"{label} escaped its root") from exc
    _require(bool(relative.parts), f"{label} cannot equal its root")
    current = root
    for index, component in enumerate(relative.parts):
        current /= component
        try:
            metadata = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise FinalInputConstructionError(f"{label} is absent") from exc
        final = index == len(relative.parts) - 1
        _require(not stat.S_ISLNK(metadata.st_mode), f"{label} is symlinked")
        _require(
            stat.S_ISREG(metadata.st_mode) if final else stat.S_ISDIR(metadata.st_mode),
            f"{label} path type differs",
        )
    descriptor = os.open(selected, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and before.st_size <= maximum_bytes,
            f"{label} identity is unsafe",
        )
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    _require(identity(before) == identity(after), f"{label} changed while read")
    return b"".join(chunks)


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalInputConstructionError(f"{label} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _load_fixed_approval(
    project_root: Path,
    *,
    binding: Stage12VerifiedBinding,
) -> Stage13ApprovalReceipt:
    payload = _stable_file_bytes(
        project_root / STAGE13_APPROVAL_PATH,
        trusted_root=project_root,
        label="fixed Stage 13 approval receipt",
    )
    value = _json_object(payload, label="fixed Stage 13 approval receipt")
    _require(
        payload == _canonical_json_bytes(value),
        "fixed Stage 13 approval receipt is not canonical JSON",
    )
    try:
        receipt = Stage13ApprovalReceipt.from_mapping(value)
        return require_stage13_unblind_authorization(
            receipt,
            expected_stage12_binding=binding,
        )
    except Stage13AuthorizationError as exc:
        raise FinalInputConstructionError(str(exc)) from exc


def _month_values(table: pa.Table, *, label: str) -> tuple[str, ...]:
    _require("holding_month" in table.column_names, f"{label} lacks holding_month")
    values = table["holding_month"].to_pylist()
    _require(
        values and all(isinstance(value, str) for value in values),
        f"{label} holding_month values are invalid",
    )
    return tuple(sorted(set(values)))


def _validate_table(
    table: pa.Table,
    *,
    schema: pa.Schema,
    months: tuple[str, ...],
    label: str,
) -> pa.Table:
    _require(isinstance(table, pa.Table), f"{label} is not an Arrow table")
    source = table.combine_chunks()
    _require(
        source.schema.equals(schema, check_metadata=True),
        f"{label} Arrow schema differs",
    )
    _require(source.num_rows > 0, f"{label} is empty")
    _require(_month_values(source, label=label) == months, f"{label} months differ")
    for field in schema:
        if not field.nullable:
            _require(
                source[field.name].null_count == 0,
                f"{label} has nulls in nonnullable {field.name}",
            )
    return source


def _validate_key(
    table: pa.Table,
    columns: tuple[str, ...],
    *,
    label: str,
) -> None:
    _require(
        all(name in table.column_names for name in columns),
        f"{label} key columns differ",
    )
    key_table = table.select(list(columns))
    _require(
        all(key_table[name].null_count == 0 for name in columns),
        f"{label} key contains nulls",
    )
    unique = key_table.group_by(list(columns)).aggregate([])
    _require(unique.num_rows == table.num_rows, f"{label} keys duplicate")


@dataclass(frozen=True, slots=True)
class AuthenticatedNormalizedFinalSources:
    """Sealed, content-identified output of the final formula publisher."""

    tables: Mapping[str, pa.Table]
    source_identity: Mapping[str, object]
    _seal: object

    def __post_init__(self) -> None:
        _require(self._seal is _SOURCE_SEAL, "normalized final sources are unissued")
        _require(
            tuple(self.tables) == _SOURCE_NAMES,
            "normalized final source table order differs",
        )


@dataclass(frozen=True, slots=True)
class FinalInputConstructionResult:
    construction_id: str
    construction_sha256: str
    construction_receipt_path: str
    output_namespace: str
    reused: bool
    registration_manifest_id: str
    registration_verification_id: str
    registration_pointer_sha256: str


def _issue_test_normalized_sources(
    tables: Mapping[str, pa.Table],
    *,
    source_identity: Mapping[str, object],
    _token: object = _TEST_SOURCE_TOKEN,
) -> AuthenticatedNormalizedFinalSources:
    """Private deterministic seam; never called by the production entry."""

    _require(_token is _TEST_SOURCE_TOKEN, "test normalized-source token differs")
    return AuthenticatedNormalizedFinalSources(
        tables=MappingProxyType(dict(tables)),
        source_identity=MappingProxyType(dict(source_identity)),
        _seal=_SOURCE_SEAL,
    )


def _validated_sources(
    sources: AuthenticatedNormalizedFinalSources,
    *,
    project_root: Path,
) -> Mapping[str, pa.Table]:
    _require(
        isinstance(sources, AuthenticatedNormalizedFinalSources),
        "normalized source authority type differs",
    )
    return _validate_normalized_source_tables(
        sources.tables,
        project_root=project_root,
    )


def _validate_normalized_source_tables(
    tables: Mapping[str, pa.Table],
    *,
    project_root: Path,
) -> Mapping[str, pa.Table]:
    """Independently validate a prospective 12-table normalized bundle.

    The raw-to-normalized publisher calls this before it writes any payload.
    Keeping the validator here makes publication and later authenticated reads
    use one exact schema, key, month, and chronology contract without granting
    the publisher access to the sealed source-authority constructor.
    """

    _require(
        isinstance(tables, Mapping) and tuple(tables) == _SOURCE_NAMES,
        "normalized source table set or order differs",
    )
    schemas = _source_schemas(project_root)
    result: dict[str, pa.Table] = {}
    for name in _SOURCE_NAMES:
        result[name] = _validate_table(
            tables[name],
            schema=schemas[name],
            months=_SOURCE_MONTHS[name],
            label=f"normalized source {name}",
        )
    _validate_key(result["final_calendar"], ("holding_month",), label="calendar")
    for name in ("final_predictors", "final_risk_cost"):
        _validate_key(
            result[name],
            ("holding_month", "signal_date", "permno"),
            label=name,
        )
    _validate_key(
        result["final_factor_covariance"],
        ("holding_month", "factor_row", "factor_col"),
        label="final_factor_covariance",
    )
    _validate_key(
        result["final_risk_cost_month_summary"],
        ("holding_month",),
        label="final_risk_cost_month_summary",
    )
    for name in ("final_market_inputs", "final_market_accounting"):
        _validate_key(
            result[name],
            ("holding_month", "calendar_date", "permno"),
            label=name,
        )
    _validate_key(result["final_factors"], ("holding_month",), label="factors")
    _validate_key(
        result["final_benchmarks"],
        ("holding_month", "benchmark_id"),
        label="benchmarks",
    )
    for name in (
        "final_completed_labels",
        "final_outcomes",
        "partial_descriptive_outcome",
    ):
        _validate_key(result[name], ("holding_month", "permno"), label=name)
    _validate_source_semantics(result)
    return MappingProxyType(result)


def _validate_source_semantics(tables: Mapping[str, pa.Table]) -> None:
    calendar_rows = tables["final_calendar"].to_pylist()
    calendar_by_month = {str(row["holding_month"]): row for row in calendar_rows}
    _require(
        tuple(calendar_by_month) == FINAL_HOLDING_MONTHS,
        "calendar row order differs",
    )
    for month, row in calendar_by_month.items():
        descriptor = final_holding_month_descriptor(month)
        signal = row["signal_date"]
        entry = row["entry_date"]
        exit_date = row["exit_date"]
        availability = row["outcome_availability_date"]
        _require(
            signal < entry < exit_date <= availability
            and signal.strftime("%Y-%m") == shift_month(month, -1)
            and entry.strftime("%Y-%m") == month
            and row["segment"] == descriptor.segment.value
            and row["complete_holding_interval"] is descriptor.complete_holding_interval
            and row["confirmatory_inference_eligible"]
            is descriptor.confirmatory_inference_eligible,
            f"calendar semantics differ: {month}",
        )
        expected_exit_month = (
            shift_month(month, 1) if descriptor.complete_holding_interval else month
        )
        _require(
            exit_date.strftime("%Y-%m") == expected_exit_month,
            f"calendar exit month differs: {month}",
        )

    for name in (
        "final_predictors",
        "final_risk_cost",
        "final_factor_covariance",
        "final_risk_cost_month_summary",
        "final_market_inputs",
        "final_market_accounting",
        "final_completed_labels",
    ):
        for row in tables[name].select(["holding_month", "signal_date"]).to_pylist():
            month = str(row["holding_month"])
            _require(
                row["signal_date"] == calendar_by_month[month]["signal_date"],
                f"{name} signal date differs: {month}",
            )

    covariance = tables["final_factor_covariance"].to_pylist()
    by_month: dict[str, list[dict[str, object]]] = {}
    for row in covariance:
        by_month.setdefault(str(row["holding_month"]), []).append(row)
    factor_names = tuple(FACTOR_NAMES)
    expected_pairs = {
        (row_name, col_name) for row_name in factor_names for col_name in factor_names
    }
    for month in FINAL_HOLDING_MONTHS:
        rows = by_month.get(month, [])
        _require(len(rows) == 36, f"factor covariance is not 6x6: {month}")
        observed_pairs = {
            (str(row["factor_row"]), str(row["factor_col"])) for row in rows
        }
        _require(observed_pairs == expected_pairs, f"factor names differ: {month}")
        matrix = np.empty((6, 6), dtype=np.float64)
        for row in rows:
            row_name = str(row["factor_row"])
            col_name = str(row["factor_col"])
            row_index = factor_names.index(row_name)
            col_index = factor_names.index(col_name)
            value = row["covariance_monthly"]
            _require(
                row["factor_row_ordinal"] == row_index
                and row["factor_col_ordinal"] == col_index
                and isinstance(value, (int, float))
                and math.isfinite(float(value)),
                f"factor covariance value differs: {month}/{row_name}/{col_name}",
            )
            matrix[row_index, col_index] = float(value)
        _require(
            np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)
            and float(np.min(np.linalg.eigvalsh(matrix))) >= -1e-10,
            f"factor covariance is not symmetric PSD: {month}",
        )

    summaries = tables["final_risk_cost_month_summary"].to_pylist()
    for row in summaries:
        month = str(row["holding_month"])
        exit_volatility = row["exit_volatility_p95_daily"]
        _require(
            row["factor_covariance_present"] is True
            and isinstance(exit_volatility, (int, float))
            and math.isfinite(float(exit_volatility))
            and float(exit_volatility) >= 0.0,
            f"monthly risk input is incomplete: {month}",
        )

    for name in ("final_completed_labels",):
        for row in tables[name].to_pylist():
            month = str(row["holding_month"])
            _require(
                row["entry_date"] == calendar_by_month[month]["entry_date"]
                and row["exit_date"] == calendar_by_month[month]["exit_date"]
                and row["outcome_availability_date"]
                == calendar_by_month[month]["outcome_availability_date"]
                and row["primary_label_status"] == "PRESENT"
                and math.isfinite(float(row["unranked_residual_return"]))
                and math.isfinite(float(row["residual_return_rank"]))
                and -0.5 <= float(row["residual_return_rank"]) <= 0.5,
                f"completed label semantics differ: {month}",
            )

    for name in ("final_market_inputs", "final_market_accounting"):
        for row in (
            tables[name]
            .select(
                [
                    "holding_month",
                    "entry_date",
                    "exit_date",
                    "calendar_date",
                ]
            )
            .to_pylist()
        ):
            month = str(row["holding_month"])
            calendar = calendar_by_month[month]
            _require(
                row["entry_date"] == calendar["entry_date"]
                and row["exit_date"] == calendar["exit_date"]
                and calendar["entry_date"]
                <= row["calendar_date"]
                <= calendar["exit_date"],
                f"{name} holding interval differs: {month}",
            )

    for row in tables["final_outcomes"].to_pylist():
        month = str(row["holding_month"])
        calendar = calendar_by_month[month]
        _require(
            row["entry_date"] == calendar["entry_date"]
            and row["exit_date"] == calendar["exit_date"]
            and math.isfinite(float(row["total_return"])),
            f"completed outcome semantics differ: {month}",
        )

    for row in tables["partial_descriptive_outcome"].to_pylist():
        month = str(row["holding_month"])
        calendar = calendar_by_month[month]
        _require(
            row["entry_date"] == calendar["entry_date"]
            and row["administrative_close_date"] == calendar["exit_date"]
            and math.isfinite(float(row["partial_total_return"])),
            f"partial descriptive outcome semantics differ: {month}",
        )

    for row in tables["final_factors"].to_pylist():
        _require(
            all(
                math.isfinite(float(row[name]))
                for name in ("mktrf", "smb", "hml", "rmw", "cma", "umd", "rf")
            ),
            f"completed factor return is nonfinite: {row['holding_month']}",
        )
    for row in tables["final_benchmarks"].to_pylist():
        _require(
            isinstance(row["benchmark_id"], str)
            and bool(row["benchmark_id"])
            and math.isfinite(float(row["total_return"])),
            f"benchmark return is invalid: {row['holding_month']}",
        )


def _filter_months(table: pa.Table, months: tuple[str, ...]) -> pa.Table:
    mask = pc.is_in(table["holding_month"], value_set=pa.array(months, pa.string()))
    return table.filter(mask)


def _project(
    table: pa.Table,
    *,
    schema: pa.Schema,
    months: tuple[str, ...],
    label: str,
) -> pa.Table:
    selected = _filter_months(table, months).select(schema.names)
    if not selected.schema.equals(schema, check_metadata=True):
        try:
            selected = selected.cast(schema, safe=True)
        except pa.ArrowException as exc:
            raise FinalInputConstructionError(
                f"{label} could not be cast to the frozen schema"
            ) from exc
    return _validate_table(selected, schema=schema, months=months, label=label)


def compile_final_input_tables(
    sources: AuthenticatedNormalizedFinalSources,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Mapping[str, pa.Table]:
    """Pure, exact twelve-source to 32-registration compiler.

    This function performs no I/O.  Its only accepted source object is sealed;
    production code can obtain that seal only from the fixed source publisher.
    """

    root = project_root.absolute()
    source = _validated_sources(sources, project_root=root)
    schemas = _independently_derived_schemas(root)
    result: dict[str, pa.Table] = {
        "final_calendar": source["final_calendar"],
    }
    segment_specs = (
        ("primary", PRIMARY_FULL_HOLDING_MONTHS),
        ("core", CORE_COMPLETE_HOLDING_MONTHS),
        ("partial", (PARTIAL_DECEMBER_HOLDING_MONTH,)),
    )
    source_by_suffix = {
        "predictors": "final_predictors",
        "risk_cost": "final_risk_cost",
        "cost_inputs": "final_risk_cost",
        "factor_covariance": "final_factor_covariance",
        "risk_cost_month_summary": "final_risk_cost_month_summary",
        "market_inputs": "final_market_inputs",
        "market_accounting": "final_market_accounting",
        "benchmarks": "final_benchmarks",
    }
    for prefix, months in segment_specs:
        for suffix, source_name in source_by_suffix.items():
            name = f"{prefix}_{suffix}"
            result[name] = _project(
                source[source_name],
                schema=schemas[name],
                months=months,
                label=f"compiled {name}",
            )
    for prefix, months in (
        ("primary", PRIMARY_FULL_HOLDING_MONTHS),
        ("core", CORE_COMPLETE_HOLDING_MONTHS),
    ):
        for suffix, source_name in (
            ("factors", "final_factors"),
            ("completed_labels", "final_completed_labels"),
            ("outcomes", "final_outcomes"),
        ):
            name = f"{prefix}_{suffix}"
            result[name] = _project(
                source[source_name],
                schema=schemas[name],
                months=months,
                label=f"compiled {name}",
            )
    result["partial_descriptive_outcome"] = source["partial_descriptive_outcome"]
    _require(set(result) == set(schemas) == set(_OUTPUT_MONTHS), "output set differs")
    ordered = {
        name: _validate_table(
            result[name],
            schema=schemas[name],
            months=_OUTPUT_MONTHS[name],
            label=f"compiled {name}",
        )
        for name in schemas
    }
    # Risk and cost are two immutable semantic registrations over identical
    # frozen inputs; byte equality makes that fact auditable.
    for prefix in ("primary", "core", "partial"):
        _require(
            ordered[f"{prefix}_risk_cost"].equals(ordered[f"{prefix}_cost_inputs"]),
            f"{prefix} risk/cost projections differ",
        )
    return MappingProxyType(ordered)


def _table_fingerprint(table: pa.Table) -> str:
    batches = table.combine_chunks().to_batches(max_chunksize=131_072)
    digest = hashlib.sha256()
    digest.update(table.schema.serialize().to_pybytes())
    for batch in batches:
        digest.update(batch.serialize().to_pybytes())
    return digest.hexdigest()


def _construction_body(
    *,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
    sources: AuthenticatedNormalizedFinalSources,
    tables: Mapping[str, pa.Table],
) -> dict[str, object]:
    table_identities = {
        name: {
            "rows": table.num_rows,
            "arrow_schema_sha256": _schema_sha256(table.schema),
            "logical_content_sha256": _table_fingerprint(table),
            "holding_months": list(_OUTPUT_MONTHS[name]),
        }
        for name, table in tables.items()
    }
    return {
        "schema_version": 1,
        "status": CONSTRUCTION_STATUS,
        "constructor_id": CONSTRUCTOR_ID,
        "stage12_binding": _binding_mapping(binding),
        "approval_payload_sha256": approval.payload_sha256,
        "normalized_source_identity": dict(sources.source_identity),
        "output_namespace": FINAL_INPUT_NAMESPACE.as_posix(),
        "table_identities": table_identities,
    }


def _safe_directory(path: Path, *, trusted_root: Path, label: str) -> None:
    root = Path(os.path.abspath(trusted_root))
    selected = Path(os.path.abspath(path))
    try:
        relative = selected.relative_to(root)
    except ValueError as exc:
        raise FinalInputConstructionError(f"{label} escaped root") from exc
    _require(bool(relative.parts), f"{label} cannot equal root")
    current = root
    for component in relative.parts:
        current /= component
        if not current.exists():
            break
        metadata = current.stat(follow_symlinks=False)
        _require(
            stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
            f"{label} path is unsafe",
        )


def _write_tables_to_directory(
    destination: Path,
    *,
    tables: Mapping[str, pa.Table],
) -> None:
    _require(
        destination.is_dir() and not any(destination.iterdir()), "stage is not empty"
    )
    for name, table in tables.items():
        target = destination / f"{name}.parquet"
        pq.write_table(
            table,
            target,
            compression="zstd",
            row_group_size=131_072,
            use_dictionary=True,
            write_statistics=True,
        )
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_output_directory(
    destination: Path,
    *,
    database_root: Path,
    project_root: Path,
) -> Mapping[str, Mapping[str, object]]:
    schemas = _independently_derived_schemas(project_root)
    _require(
        destination.is_dir()
        and not destination.is_symlink()
        and {item.name for item in destination.iterdir()}
        == {f"{name}.parquet" for name in schemas},
        "constructed final-input member set differs",
    )
    receipts: dict[str, Mapping[str, object]] = {}
    for name, schema in schemas.items():
        receipts[name] = MappingProxyType(
            _inspect_parquet(
                destination / f"{name}.parquet",
                database_root=database_root,
                expected_schema=_physical_schema(schema),
                expected_months=_OUTPUT_MONTHS[name],
                label=f"constructed final input {name}",
            )
        )
    return MappingProxyType(receipts)


def _atomic_json(path: Path, value: object, *, root: Path) -> bytes:
    payload = _canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_directory(path.parent, trusted_root=root, label="publication directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return payload


def _publish_immutable_pointer_history(
    *,
    publication_root: Path,
    namespace: Path,
    identity: str,
    document: Mapping[str, object],
    label: str,
) -> None:
    """Preserve a superseded fixed pointer before its atomic replacement."""

    target = publication_root / namespace / f"{identity}.json"
    expected = _canonical_json_bytes(dict(document))
    if target.exists():
        observed = _stable_file_bytes(
            target,
            trusted_root=publication_root,
            label=label,
        )
        _require(observed == expected, f"immutable {label} differs")
        return
    _atomic_json(target, dict(document), root=publication_root)


def _archive_stale_construction_pointer(
    *,
    publication_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
    expected_construction_id: str,
) -> None:
    pointer_path = publication_root / CONSTRUCTION_POINTER
    if not pointer_path.exists():
        return
    pointer = _json_object(
        _stable_file_bytes(
            pointer_path,
            trusted_root=publication_root,
            label="final-input construction pointer",
        ),
        label="final-input construction pointer",
    )
    keys = {
        "schema_version",
        "status",
        "construction_id",
        "construction_sha256",
        "construction_receipt_path",
        "construction_receipt_sha256",
        "stage12_binding_sha256",
        "approval_payload_sha256",
    }
    construction_id = pointer.get("construction_id")
    _require(
        set(pointer) == keys
        and pointer.get("schema_version") == 1
        and pointer.get("status") == CONSTRUCTION_STATUS
        and isinstance(construction_id, str)
        and _CONSTRUCTION_ID.fullmatch(construction_id) is not None
        and all(
            isinstance(pointer.get(field), str)
            and _HEX64.fullmatch(str(pointer[field])) is not None
            for field in (
                "construction_sha256",
                "construction_receipt_sha256",
                "stage12_binding_sha256",
                "approval_payload_sha256",
            )
        ),
        "existing final-input construction pointer is invalid",
    )
    if (
        construction_id == expected_construction_id
        and pointer["stage12_binding_sha256"] == binding.binding_sha256
        and pointer["approval_payload_sha256"] == approval.payload_sha256
    ):
        return
    _publish_immutable_pointer_history(
        publication_root=publication_root,
        namespace=CONSTRUCTION_POINTER_HISTORY_NAMESPACE,
        identity=construction_id,
        document=pointer,
        label="final-input construction pointer history",
    )


def _archive_stale_registration_pointer(
    *,
    publication_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
) -> None:
    pointer_path = publication_root / VERIFIED_REGISTRATION_POINTER
    if not pointer_path.exists():
        return
    pointer = _json_object(
        _stable_file_bytes(
            pointer_path,
            trusted_root=publication_root,
            label="verified final-input registration pointer",
        ),
        label="verified final-input registration pointer",
    )
    manifest_id = pointer.get("manifest_id")
    _require(
        pointer.get("schema_version") == 1
        and pointer.get("status") == "VERIFIED_PASSED"
        and isinstance(manifest_id, str)
        and _REGISTRATION_MANIFEST_ID.fullmatch(manifest_id) is not None
        and isinstance(pointer.get("stage12_binding_sha256"), str)
        and _HEX64.fullmatch(str(pointer["stage12_binding_sha256"])) is not None
        and isinstance(pointer.get("approval_payload_sha256"), str)
        and _HEX64.fullmatch(str(pointer["approval_payload_sha256"])) is not None,
        "existing verified final-input registration pointer is invalid",
    )
    if (
        pointer["stage12_binding_sha256"] == binding.binding_sha256
        and pointer["approval_payload_sha256"] == approval.payload_sha256
    ):
        return
    _publish_immutable_pointer_history(
        publication_root=publication_root,
        namespace=REGISTRATION_POINTER_HISTORY_NAMESPACE,
        identity=manifest_id,
        document=pointer,
        label="verified final-input registration pointer history",
    )


def _construct_with_authenticated_sources(
    *,
    project_root: Path,
    database_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
    sources: AuthenticatedNormalizedFinalSources,
    publish_registration: bool,
    publication_root: Path | None = None,
) -> FinalInputConstructionResult:
    root = project_root.absolute()
    database = database_root.absolute()
    publication = (publication_root or root).absolute()
    authorization = require_stage13_unblind_authorization(
        approval,
        expected_stage12_binding=binding,
    )
    tables = compile_final_input_tables(sources, project_root=root)
    body = _construction_body(
        binding=binding,
        approval=authorization,
        sources=sources,
        tables=tables,
    )
    construction_sha = _sha256_bytes(_canonical_json_bytes(body))
    construction_id = f"stage13-input-construction-{construction_sha[:24]}"
    _archive_stale_construction_pointer(
        publication_root=publication,
        binding=binding,
        approval=authorization,
        expected_construction_id=construction_id,
    )
    if publish_registration:
        _archive_stale_registration_pointer(
            publication_root=publication,
            binding=binding,
            approval=authorization,
        )
    receipt_relative = CONSTRUCTION_NAMESPACE / f"{construction_id}.json"
    receipt_path = publication / receipt_relative
    final_root = database / FINAL_INPUT_NAMESPACE
    _safe_directory(final_root, trusted_root=database, label="final input namespace")
    reused = False
    if final_root.exists():
        _require(
            receipt_path.exists(),
            "existing final inputs lack their content-addressed construction receipt",
        )
        prior_payload = _stable_file_bytes(
            receipt_path,
            trusted_root=publication,
            label="construction receipt",
        )
        prior_document = _json_object(
            prior_payload,
            label="construction receipt",
        )
        _require(
            all(prior_document.get(key) == value for key, value in body.items())
            and prior_document.get("construction_id") == construction_id
            and prior_document.get("construction_sha256") == construction_sha,
            "existing construction receipt belongs to another build",
        )
        receipts = _verify_output_directory(
            final_root,
            database_root=database,
            project_root=root,
        )
        expected = prior_document.get("output_receipts")
        _require(isinstance(expected, Mapping), "memoized output receipts are absent")
        _require(
            all(
                isinstance(expected.get(name), Mapping)
                and dict(receipts[name]) == dict(expected[name])
                for name in tables
            ),
            "memoized final inputs do not match this construction",
        )
        reused = True
    else:
        final_root.parent.mkdir(parents=True, exist_ok=True)
        _safe_directory(
            final_root.parent,
            trusted_root=database,
            label="final input parent",
        )
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{construction_id}.",
                suffix=".partial",
                dir=final_root.parent,
            )
        )
        try:
            _write_tables_to_directory(stage, tables=tables)
            _verify_output_directory(
                stage,
                database_root=database,
                project_root=root,
            )
            os.replace(stage, final_root)
            parent = os.open(final_root.parent, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    output_receipts = _verify_output_directory(
        final_root,
        database_root=database,
        project_root=root,
    )
    document = {
        **body,
        "construction_id": construction_id,
        "construction_sha256": construction_sha,
        "output_receipts": {
            name: dict(receipt) for name, receipt in output_receipts.items()
        },
    }
    if receipt_path.exists():
        observed = _stable_file_bytes(
            receipt_path,
            trusted_root=publication,
            label="construction receipt",
        )
        _require(
            observed == _canonical_json_bytes(document),
            "memoized construction receipt differs",
        )
    else:
        _atomic_json(receipt_path, document, root=publication)
    pointer = {
        "schema_version": 1,
        "status": CONSTRUCTION_STATUS,
        "construction_id": construction_id,
        "construction_sha256": construction_sha,
        "construction_receipt_path": receipt_relative.as_posix(),
        "construction_receipt_sha256": _sha256_bytes(_canonical_json_bytes(document)),
        "stage12_binding_sha256": binding.binding_sha256,
        "approval_payload_sha256": authorization.payload_sha256,
    }
    _atomic_json(
        publication / CONSTRUCTION_POINTER,
        pointer,
        root=publication,
    )
    if publish_registration:
        registration = publish_verified_final_input_registration(authorization)
    else:
        registration = {
            "manifest_id": "stage13-inputs-" + "0" * 24,
            "verification_id": "verify-stage13-inputs-" + "0" * 20,
        }
    registration_pointer_payload = (
        _stable_file_bytes(
            root
            / "artifacts/phase_06_final_test/LATEST_STAGE_13_REGISTRATION_VERIFIED.json",
            trusted_root=root,
            label="verified registration pointer",
        )
        if publish_registration
        else _canonical_json_bytes(registration)
    )
    return FinalInputConstructionResult(
        construction_id=construction_id,
        construction_sha256=construction_sha,
        construction_receipt_path=receipt_relative.as_posix(),
        output_namespace=FINAL_INPUT_NAMESPACE.as_posix(),
        reused=reused,
        registration_manifest_id=str(registration["manifest_id"]),
        registration_verification_id=str(registration["verification_id"]),
        registration_pointer_sha256=_sha256_bytes(registration_pointer_payload),
    )


def _construct_with_test_sources(
    *,
    project_root: Path,
    database_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
    sources: AuthenticatedNormalizedFinalSources,
) -> FinalInputConstructionResult:
    """Private synthetic state-machine runner without fixed metadata writes."""

    return _construct_with_authenticated_sources(
        project_root=project_root,
        database_root=database_root,
        binding=binding,
        approval=approval,
        sources=sources,
        publish_registration=False,
        publication_root=database_root,
    )


def _load_authenticated_normalized_sources(
    *,
    project_root: Path,
    database_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
    publication_root: Path | None = None,
) -> AuthenticatedNormalizedFinalSources:
    """Re-authenticate the fixed normalized-source pointer and every payload.

    The pointer cannot be supplied by a caller.  Its manifest binds the exact
    executable publisher code/config members, raw/builder provenance, and the
    12 fixed source paths.
    """

    require_stage13_unblind_authorization(
        approval,
        expected_stage12_binding=binding,
    )
    metadata_root = (publication_root or project_root).absolute()
    pointer_path = metadata_root / NORMALIZED_SOURCE_POINTER
    if not pointer_path.exists():
        raise FinalInputConstructionError(
            "fixed verified normalized-source pointer is absent"
        )
    pointer_payload = _stable_file_bytes(
        pointer_path,
        trusted_root=metadata_root,
        label="normalized source verified pointer",
    )
    pointer = _json_object(pointer_payload, label="normalized source verified pointer")
    pointer_keys = {
        "schema_version",
        "status",
        "source_bundle_id",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_manifest_size_bytes",
        "stage12_binding_sha256",
        "approval_payload_sha256",
        "difference_count",
    }
    bundle_id = pointer.get("source_bundle_id")
    manifest_relative = pointer.get("source_manifest_path")
    _require(
        set(pointer) == pointer_keys
        and pointer.get("schema_version") == 1
        and pointer.get("status") == "FINAL_NORMALIZED_SOURCE_VERIFIED"
        and pointer.get("difference_count") == 0
        and isinstance(bundle_id, str)
        and _SOURCE_BUNDLE_ID.fullmatch(bundle_id) is not None
        and pointer.get("stage12_binding_sha256") == binding.binding_sha256
        and pointer.get("approval_payload_sha256") == approval.payload_sha256
        and isinstance(manifest_relative, str),
        "normalized source pointer identity differs",
    )
    manifest_path = metadata_root / manifest_relative
    try:
        relative = manifest_path.relative_to(
            metadata_root / NORMALIZED_SOURCE_MANIFEST_NAMESPACE
        )
    except ValueError as exc:
        raise FinalInputConstructionError(
            "normalized source manifest escaped its namespace"
        ) from exc
    _require(
        len(relative.parts) == 1
        and relative.suffix == ".json"
        and relative.name == f"{bundle_id}.json",
        "normalized source manifest path differs",
    )
    manifest_payload = _stable_file_bytes(
        manifest_path,
        trusted_root=metadata_root,
        label="normalized source manifest",
    )
    _require(
        isinstance(pointer.get("source_manifest_sha256"), str)
        and _HEX64.fullmatch(str(pointer["source_manifest_sha256"])) is not None
        and pointer["source_manifest_sha256"] == _sha256_bytes(manifest_payload)
        and pointer.get("source_manifest_size_bytes") == len(manifest_payload),
        "normalized source manifest content identity differs",
    )
    manifest = _json_object(manifest_payload, label="normalized source manifest")
    manifest_keys = {
        "schema_version",
        "status",
        "source_bundle_id",
        "publisher_id",
        "publisher_code_hashes",
        "publisher_code_hashes_sha256",
        "publisher_config_hashes",
        "publisher_config_hashes_sha256",
        "stage12_binding_sha256",
        "approval_payload_sha256",
        "database_root",
        "fixed_source_namespace",
        "raw_source_identity",
        "raw_source_identity_sha256",
        "builder_identity",
        "builder_identity_sha256",
        "entries",
    }
    _require(
        set(manifest) == manifest_keys
        and manifest.get("schema_version") == 1
        and manifest.get("status") == "FINAL_NORMALIZED_SOURCE_PUBLISHED"
        and manifest.get("source_bundle_id") == bundle_id
        and manifest.get("stage12_binding_sha256") == binding.binding_sha256
        and manifest.get("approval_payload_sha256") == approval.payload_sha256
        and manifest.get("database_root") == str(database_root)
        and manifest.get("fixed_source_namespace")
        == NORMALIZED_SOURCE_NAMESPACE.as_posix()
        and isinstance(manifest.get("publisher_id"), str)
        and bool(str(manifest["publisher_id"])),
        "normalized source manifest identity differs",
    )
    for identity_field in ("raw_source_identity", "builder_identity"):
        identity_value = manifest.get(identity_field)
        identity_sha_field = f"{identity_field}_sha256"
        _require(
            isinstance(identity_value, Mapping)
            and bool(identity_value)
            and isinstance(manifest.get(identity_sha_field), str)
            and _HEX64.fullmatch(str(manifest[identity_sha_field])) is not None
            and manifest[identity_sha_field]
            == _sha256_bytes(_canonical_json_bytes(dict(identity_value))),
            f"normalized source {identity_field} differs",
        )
    builder_identity = manifest["builder_identity"]
    _require(
        builder_identity.get("stage12_binding_sha256") == binding.binding_sha256
        and builder_identity.get("approval_payload_sha256") == approval.payload_sha256,
        "normalized source builder authorization identity differs",
    )
    if builder_identity.get("builder_id") == "stage13-final-raw-formula-builder-v1":
        formula_identity = builder_identity.get("formula_identity")
        _require(
            isinstance(formula_identity, Mapping),
            "normalized source formula identity is absent",
        )
        for hashes_field in ("formula_code_hashes", "formula_config_hashes"):
            formula_hashes = formula_identity.get(hashes_field)
            _require(
                isinstance(formula_hashes, Mapping) and bool(formula_hashes),
                f"normalized source {hashes_field} is absent",
            )
            normalized_formula_hashes: dict[str, str] = {}
            for raw_path, raw_hash in formula_hashes.items():
                _require(
                    isinstance(raw_path, str)
                    and raw_path
                    and isinstance(raw_hash, str)
                    and _HEX64.fullmatch(raw_hash) is not None,
                    f"normalized source {hashes_field} member is invalid",
                )
                relative_path = Path(raw_path)
                _require(
                    not relative_path.is_absolute()
                    and ".." not in relative_path.parts
                    and relative_path.parts[0] in {"src", "configs"},
                    f"normalized source {hashes_field} member path is unsafe",
                )
                member_payload = _stable_file_bytes(
                    project_root / relative_path,
                    trusted_root=project_root,
                    label=f"normalized source formula member {raw_path}",
                    maximum_bytes=32 * 1024 * 1024,
                )
                _require(
                    _sha256_bytes(member_payload) == raw_hash,
                    f"normalized source formula implementation changed: {raw_path}",
                )
                normalized_formula_hashes[raw_path] = raw_hash
            aggregate_field = f"{hashes_field}_sha256"
            _require(
                formula_identity.get(aggregate_field)
                == _sha256_bytes(_canonical_json_bytes(normalized_formula_hashes)),
                f"normalized source {hashes_field} aggregate differs",
            )

        raw_identity = manifest["raw_source_identity"]
        _require(
            raw_identity.get("raw_authority_id")
            == "phase01-frozen-raw-authority-for-stage13-v1",
            "normalized source raw authority identity differs",
        )
        for path_field in (
            "frozen_pointer_path",
            "verified_pointer_path",
            "transitive_manifest_path",
            "primary_manifest_path",
        ):
            relative_value = raw_identity.get(path_field)
            sha_value = raw_identity.get(f"{path_field.removesuffix('_path')}_sha256")
            _require(
                isinstance(relative_value, str)
                and isinstance(sha_value, str)
                and _HEX64.fullmatch(sha_value) is not None,
                f"normalized source raw {path_field} identity is invalid",
            )
            relative_path = Path(relative_value)
            _require(
                not relative_path.is_absolute()
                and ".." not in relative_path.parts
                and relative_path.parts[:2] == ("artifacts", "phase_01_raw_manifest"),
                f"normalized source raw {path_field} path is unsafe",
            )
            raw_metadata_payload = _stable_file_bytes(
                project_root / relative_path,
                trusted_root=project_root,
                label=f"normalized source raw authority {path_field}",
                maximum_bytes=64 * 1024 * 1024,
            )
            _require(
                _sha256_bytes(raw_metadata_payload) == sha_value,
                f"normalized source raw authority changed: {path_field}",
            )
    implementation_identity: dict[str, object] = {
        "publisher_id": manifest["publisher_id"],
        "raw_source_identity": dict(manifest["raw_source_identity"]),
        "raw_source_identity_sha256": manifest["raw_source_identity_sha256"],
        "builder_identity": dict(manifest["builder_identity"]),
        "builder_identity_sha256": manifest["builder_identity_sha256"],
    }
    for field in ("publisher_code_hashes", "publisher_config_hashes"):
        hashes = manifest.get(field)
        _require(
            isinstance(hashes, Mapping) and bool(hashes),
            f"normalized source {field} is absent",
        )
        normalized: dict[str, str] = {}
        for raw_path, raw_hash in hashes.items():
            _require(
                isinstance(raw_path, str)
                and raw_path
                and isinstance(raw_hash, str)
                and _HEX64.fullmatch(raw_hash) is not None,
                f"normalized source {field} member is invalid",
            )
            relative_path = Path(raw_path)
            _require(
                not relative_path.is_absolute()
                and ".." not in relative_path.parts
                and relative_path.parts[0] in {"src", "configs"},
                f"normalized source {field} member path is unsafe",
            )
            payload = _stable_file_bytes(
                project_root / relative_path,
                trusted_root=project_root,
                label=f"normalized source implementation {raw_path}",
                maximum_bytes=32 * 1024 * 1024,
            )
            _require(
                _sha256_bytes(payload) == raw_hash,
                f"normalized source implementation changed: {raw_path}",
            )
            normalized[raw_path] = raw_hash
        digest_field = f"{field}_sha256"
        _require(
            manifest.get(digest_field)
            == _sha256_bytes(_canonical_json_bytes(normalized)),
            f"normalized source {field} aggregate differs",
        )
        implementation_identity[field] = normalized
        implementation_identity[digest_field] = manifest[digest_field]

    entries = manifest.get("entries")
    _require(
        isinstance(entries, list) and len(entries) == len(_SOURCE_NAMES),
        "normalized source entry count differs",
    )
    schemas = _source_schemas(project_root)
    tables: dict[str, pa.Table] = {}
    source_receipts: dict[str, object] = {}
    for expected_name, raw_entry in zip(_SOURCE_NAMES, entries, strict=True):
        _require(isinstance(raw_entry, Mapping), "normalized source entry is invalid")
        entry_keys = {
            "source_name",
            "relative_path",
            "absolute_path",
            "sha256",
            "size_bytes",
            "rows",
            "row_groups",
            "arrow_schema_sha256",
            "holding_months",
        }
        relative_path = NORMALIZED_SOURCE_NAMESPACE / f"{expected_name}.parquet"
        absolute_path = database_root / relative_path
        _require(
            set(raw_entry) == entry_keys
            and raw_entry.get("source_name") == expected_name
            and raw_entry.get("relative_path") == relative_path.as_posix()
            and raw_entry.get("absolute_path") == str(absolute_path)
            and raw_entry.get("holding_months") == list(_SOURCE_MONTHS[expected_name]),
            f"normalized source entry contract differs: {expected_name}",
        )
        observed = _inspect_parquet(
            absolute_path,
            database_root=database_root,
            expected_schema=_physical_schema(schemas[expected_name]),
            expected_months=_SOURCE_MONTHS[expected_name],
            label=f"normalized source {expected_name}",
        )
        _require(
            all(raw_entry.get(key) == value for key, value in observed.items()),
            f"normalized source receipt differs: {expected_name}",
        )
        try:
            table = pq.read_table(absolute_path, use_threads=False).combine_chunks()
        except (OSError, pa.ArrowException) as exc:
            raise FinalInputConstructionError(
                f"normalized source could not be opened: {expected_name}"
            ) from exc
        tables[expected_name] = _validate_table(
            table,
            schema=schemas[expected_name],
            months=_SOURCE_MONTHS[expected_name],
            label=f"normalized source {expected_name}",
        )
        source_receipts[expected_name] = dict(raw_entry)
    identity = {
        "source_pointer_path": NORMALIZED_SOURCE_POINTER.as_posix(),
        "source_pointer_sha256": _sha256_bytes(pointer_payload),
        "source_manifest_path": manifest_relative,
        "source_manifest_sha256": _sha256_bytes(manifest_payload),
        "source_bundle_id": bundle_id,
        "implementation": implementation_identity,
        "entries": source_receipts,
    }
    return AuthenticatedNormalizedFinalSources(
        tables=MappingProxyType(tables),
        source_identity=MappingProxyType(identity),
        _seal=_SOURCE_SEAL,
    )


def construct_and_publish_final_inputs() -> FinalInputConstructionResult:
    """Zero-argument production construction and registration entry point.

    Authentication order is observable and fixed: verified Stage 12, separate
    repository-fixed Stage 13 approval, configured database root, normalized
    source authority, atomic construction, independent verification, and only
    then verified registration publication.
    """

    root = PROJECT_ROOT.absolute()
    try:
        binding = load_authenticated_stage12_binding(root)
        approval = _load_fixed_approval(root, binding=binding)
        database = _configured_database_root(root)
        # Delayed import keeps raw adapters unreachable until both gates above
        # have authenticated.  The publisher itself classifies the pointer as
        # current or as the exact frozen predecessor; mere existence is never
        # treated as reuse authority.
        from portfolio_bot.final_test.final_raw_source_publication import (
            _publish_final_normalized_sources_authenticated,
        )

        _publish_final_normalized_sources_authenticated(
            project_root=root,
            database_root=database,
            binding=binding,
            approval=approval,
        )
        sources = _load_authenticated_normalized_sources(
            project_root=root,
            database_root=database,
            binding=binding,
            approval=approval,
        )
        return _construct_with_authenticated_sources(
            project_root=root,
            database_root=database,
            binding=binding,
            approval=approval,
            sources=sources,
            publish_registration=True,
            publication_root=root,
        )
    except (Stage13AuthorizationError, FinalInputMaterializationError) as exc:
        raise FinalInputConstructionError(str(exc)) from exc


__all__ = [
    "CONSTRUCTION_STATUS",
    "FinalInputConstructionError",
    "FinalInputConstructionResult",
    "RAW_TO_NORMALIZED_IMPLEMENTATION_STATUS",
    "compile_final_input_tables",
    "construct_and_publish_final_inputs",
]
