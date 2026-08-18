"""Independent verification of the Stage 11 predictor market extensions.

This verifier never imports the Stage 11 builders.  It reauthenticates the
frozen schemas and source bindings, scans every produced Parquet footer and
primary key, proves paired-key equality, and rejects any observation or
observable delisting outcome after the 2019-10-31 predictor cutoff.  It does
not read model labels, forecasts, portfolio weights, P&L, or 2020+ payloads.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data import phase_02_crsp_return_reconciliation as returns
from portfolio_bot.data import phase_02_crsp_typed_backbone as typed
from portfolio_bot.data import phase_02_crsp_typed_backbone_discovery as discovery
from portfolio_bot.data import phase_02_crsp_typed_backbone_transforms as transforms
from portfolio_bot.data.phase_02_crsp_typed_backbone_core import (
    SchemaRegistry,
    TableSpec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATABASE_ROOT = Path("/Users/agraemirolmez/Desktop/Thesis")
CONTRACT_ID = "phase05-stage11-validation-market-extension-v1"
EXTENSION_START = dt.date(2015, 1, 1)
LAST_SIGNAL = dt.date(2019, 10, 31)
RESERVED_OUTCOME_START = dt.date(2019, 11, 1)
RESERVED_OUTCOME_END = dt.date(2019, 12, 2)
ROW_GROUP_SIZE = 262_144

INTERVAL_TABLES = {
    "crsp_identity_interval": "start_date",
    "crsp_shares_interval": "start_date",
}
FACT_TABLES = {
    "crsp_security_day": "calendar_date",
    "crsp_security_day_return_staging": "calendar_date",
    "crsp_security_month_crosscheck": "calendar_month",
    "crsp_security_month_return_staging": "calendar_month",
}
ALL_TYPED_TABLES = {**INTERVAL_TABLES, **FACT_TABLES}
RETURN_DATE_COLUMNS = {
    "crsp_distribution_event": "ex_date",
    "crsp_delisting_event": "delisting_date",
    "crsp_security_day_return": "calendar_date",
    "crsp_security_month_return_reconciliation": "calendar_month",
}
EXPECTED_NS_AUDIT = {
    "schema_version": 1,
    "rule": "NS_new_security_partial_start_is_diagnostic_not_complete_month",
    "monthly_ns_rows": 2_561,
    "within_frozen_tolerance_rows": 2_560,
    "out_of_tolerance_diagnostic_rows": 1,
    "diagnostics": [
        {
            "permno": 16_493,
            "calendar_month": "2016-11-30",
            "monthly_flag": "NS",
            "vendor_monthly_return": "-0.125273000000",
            "compounded_available_daily_return": "0.033844855499",
            "absolute_difference": "0.159117855499",
            "frozen_absolute_tolerance": "0.000060000000",
            "first_scoped_daily_date": "2016-11-07",
            "first_numeric_daily_return_date": "2016-11-09",
            "classification": "new_security_partial_start_diagnostic_difference",
        }
    ],
}


class ValidationMarketVerificationError(RuntimeError):
    """Raised when independent predictor-market verification fails closed."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationMarketVerificationError(message)


def _canonical_json(value: object) -> bytes:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _safe_root(path: Path, label: str) -> Path:
    candidate = Path(os.path.abspath(path.expanduser()))
    _check(candidate.is_dir() and not candidate.is_symlink(), f"Unsafe {label}")
    _check(candidate.resolve(strict=True) == candidate, f"Symlinked {label}")
    database = DATABASE_ROOT.resolve(strict=True)
    try:
        candidate.relative_to(database)
    except ValueError as exc:
        raise ValidationMarketVerificationError(
            f"{label} escaped the database root"
        ) from exc
    return candidate


def _safe_file(root: Path, filename: str) -> Path:
    _check(Path(filename).name == filename, "Unsafe payload filename")
    path = root / filename
    _check(path.is_file() and not path.is_symlink(), f"Missing payload {filename}")
    _check(path.resolve(strict=True).parent == root, f"Payload escaped root: {filename}")
    return path


def _canonical_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _check(path.is_file() and not path.is_symlink(), f"Missing {label}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationMarketVerificationError(f"Invalid {label} JSON") from exc
    _check(isinstance(value, dict), f"{label} is not an object")
    _check(payload == _canonical_json(value), f"{label} is not canonical JSON")
    return value, payload


def _expected_typed_schema(
    spec: TableSpec,
    *,
    minimum: dt.date,
    maximum: dt.date,
    disclose_scope_semantics: bool,
) -> pa.Schema:
    frozen = spec.arrow_schema()
    metadata = dict(frozen.metadata or {})
    metadata.pop(b"expected_rows", None)
    metadata.update(
        {
            b"contract_id": CONTRACT_ID.encode(),
            b"grain": spec.grain.encode(),
            b"primary_key": _canonical_json(list(spec.primary_key))[:-1],
            b"primary_source": spec.primary_source.encode(),
            b"role": ("stage11_validation_extension_of_" + spec.role).encode(),
            b"scope_maximum_date": maximum.isoformat().encode(),
            b"scope_minimum_date": minimum.isoformat().encode(),
            b"sort_key": _canonical_json(list(spec.sort_key))[:-1],
            b"table": spec.name.encode(),
        }
    )
    if disclose_scope_semantics:
        metadata[b"scope_semantics"] = (
            b"retain_intervals_overlapping_closed_scope; endpoint values may extend outside"
            if spec.name in INTERVAL_TABLES
            else b"retain_observation_dates_inside_closed_scope"
        )
    return pa.schema(list(frozen), metadata=metadata)


def _key_array(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    _check(values.null_count == 0, "Primary key contains nulls")
    if pa.types.is_date32(values.type):
        return (
            values.to_numpy(zero_copy_only=False)
            .astype("datetime64[D]")
            .astype(np.int64)
        )
    if pa.types.is_integer(values.type):
        return values.to_numpy(zero_copy_only=False).astype(np.int64)
    raise ValidationMarketVerificationError(f"Unsupported primary-key type {values.type}")


def _scan_parquet(
    path: Path,
    *,
    expected_schema: pa.Schema,
    primary_key: tuple[str, ...],
    date_column: str,
    maximum_date: dt.date,
    minimum_date: dt.date | None,
    availability_column: str | None = None,
    authorization_column: str | None = None,
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    _check(
        parquet.schema_arrow.equals(expected_schema, check_metadata=True),
        f"Exact schema differs: {path.name}",
    )
    group_rows = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.num_row_groups)
    ]
    _check(group_rows and all(value > 0 for value in group_rows), f"Empty row group: {path.name}")
    _check(
        all(value == ROW_GROUP_SIZE for value in group_rows[:-1])
        and group_rows[-1] <= ROW_GROUP_SIZE,
        f"Row-group geometry differs: {path.name}",
    )
    columns = list(
        dict.fromkeys(
            [
                *primary_key,
                date_column,
                *(() if availability_column is None else (availability_column,)),
                *(() if authorization_column is None else (authorization_column,)),
            ]
        )
    )
    digest = hashlib.sha256()
    digest.update(("\0".join(primary_key) + "\0").encode())
    previous: tuple[int, ...] | None = None
    rows = 0
    minimum_ordinal: int | None = None
    maximum_ordinal: int | None = None
    availability_maximum: int | None = None
    authorization_true_rows = 0
    for batch in parquet.iter_batches(
        batch_size=ROW_GROUP_SIZE,
        columns=columns,
        use_threads=False,
    ):
        table = pa.Table.from_batches([batch])
        arrays = [_key_array(table[name]) for name in primary_key]
        if table.num_rows:
            undecided = np.ones(table.num_rows - 1, dtype=bool)
            violation = np.zeros(table.num_rows - 1, dtype=bool)
            duplicate = np.ones(table.num_rows - 1, dtype=bool)
            for array in arrays:
                violation |= undecided & (array[1:] < array[:-1])
                undecided &= array[1:] == array[:-1]
                duplicate &= array[1:] == array[:-1]
            digest.update(
                np.column_stack(arrays).astype("<i8", copy=False).tobytes()
            )
            _check(not bool(violation.any()), f"Primary key is unsorted: {path.name}")
            _check(not bool(duplicate.any()), f"Duplicate primary key: {path.name}")
            first = tuple(int(array[0]) for array in arrays)
            last = tuple(int(array[-1]) for array in arrays)
            _check(previous is None or first > previous, f"Cross-batch key failure: {path.name}")
            previous = last
        dates = _key_array(table[date_column])
        if len(dates):
            observed_min = int(dates.min())
            observed_max = int(dates.max())
            minimum_ordinal = observed_min if minimum_ordinal is None else min(minimum_ordinal, observed_min)
            maximum_ordinal = observed_max if maximum_ordinal is None else max(maximum_ordinal, observed_max)
        if availability_column is not None:
            availability = table[availability_column].combine_chunks()
            _check(availability.null_count == 0, f"Null outcome availability: {path.name}")
            available_values = _key_array(availability)
            if len(available_values):
                value = int(available_values.max())
                availability_maximum = value if availability_maximum is None else max(availability_maximum, value)
        if authorization_column is not None:
            authorization = table[authorization_column].combine_chunks()
            _check(authorization.null_count == 0, f"Null authorization guard: {path.name}")
            authorization_true_rows += int(pc.sum(pc.cast(authorization, pa.int64())).as_py())
        rows += table.num_rows
    _check(rows == parquet.metadata.num_rows and rows > 0, f"Row count differs: {path.name}")
    epoch = dt.date(1970, 1, 1)
    observed_minimum = None if minimum_ordinal is None else epoch + dt.timedelta(days=minimum_ordinal)
    observed_maximum = None if maximum_ordinal is None else epoch + dt.timedelta(days=maximum_ordinal)
    _check(observed_maximum is not None and observed_maximum <= maximum_date, f"Reserved outcome row reached: {path.name}")
    if minimum_date is not None:
        _check(observed_minimum is not None and observed_minimum >= minimum_date, f"Pre-extension fact row: {path.name}")
    if availability_maximum is not None:
        _check(
            epoch + dt.timedelta(days=availability_maximum) <= maximum_date,
            f"Post-signal outcome became observable: {path.name}",
        )
    _check(authorization_true_rows == 0, f"Payload self-authorized return use: {path.name}")
    return {
        "name": path.stem,
        "path": path.name,
        "rows": rows,
        "row_groups": parquet.num_row_groups,
        "size_bytes": path.stat(follow_symlinks=False).st_size,
        "sha256": _sha256(path),
        "schema_sha256": _schema_sha256(parquet.schema_arrow),
        "minimum_date": None if observed_minimum is None else observed_minimum.isoformat(),
        "maximum_date": None if observed_maximum is None else observed_maximum.isoformat(),
        "primary_key_sha256": digest.hexdigest(),
        "maximum_outcome_availability_date": (
            None
            if availability_maximum is None
            else (epoch + dt.timedelta(days=availability_maximum)).isoformat()
        ),
        "authorization_true_rows": authorization_true_rows,
    }


def _interval_structure(path: Path) -> dict[str, int]:
    table = pq.read_table(path, columns=["permno", "start_date", "end_date"])
    permno = _key_array(table["permno"])
    starts = _key_array(table["start_date"])
    ends = _key_array(table["end_date"])
    same = permno[1:] == permno[:-1]
    duplicate = same & (starts[1:] == starts[:-1]) & (ends[1:] == ends[:-1])
    overlap = same & (starts[1:] <= ends[:-1])
    contiguous = same & (starts[1:] == ends[:-1] + 1)
    gap = same & (starts[1:] > ends[:-1] + 1)
    result = {
        "rows": len(permno),
        "unique_permnos": len(np.unique(permno)),
        "duplicate_key_groups": int(np.count_nonzero(duplicate)),
        "contiguous_edges": int(np.count_nonzero(contiguous)),
        "gap_edges": int(np.count_nonzero(gap)),
        "overlap_edges": int(np.count_nonzero(overlap)),
    }
    _check(result["duplicate_key_groups"] == 0, f"Duplicate intervals: {path.name}")
    _check(result["overlap_edges"] == 0, f"Overlapping intervals: {path.name}")
    _check(result["gap_edges"] == 0, f"Interval gaps: {path.name}")
    return result


def _registry_with_rows(registry: SchemaRegistry, rows: Mapping[str, int]) -> SchemaRegistry:
    _check(set(rows) == set(registry.tables), "Return row-count role set differs")
    return dataclasses.replace(
        registry,
        tables=MappingProxyType(
            {
                name: dataclasses.replace(spec, expected_rows=int(rows[name]))
                for name, spec in registry.tables.items()
            }
        ),
    )


def _verify_typed(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], bytes]:
    receipt, receipt_bytes = _canonical_object(root / "typed_extension_receipt.json", "typed receipt")
    current_keys = {
        "schema_version",
        "contract_id",
        "status",
        "materialization_role",
        "scope",
        "source_bindings",
        "tables",
        "interval_structure",
        "authorizations",
    }
    legacy_keys = current_keys - {"interval_structure"}
    _check(
        frozenset(receipt) in {frozenset(current_keys), frozenset(legacy_keys)},
        "Typed receipt keys differ",
    )
    current_receipt = "interval_structure" in receipt
    _check(receipt["schema_version"] == 1 and receipt["contract_id"] == CONTRACT_ID, "Typed receipt identity differs")
    _check(receipt["status"] == "TYPED_VALIDATION_PREDICTOR_MARKET_EXTENSION_BUILT_PENDING_VERIFICATION", "Typed receipt status differs")
    _check(receipt["materialization_role"] == "point_in_time_predictor_context_only", "Typed materialization role differs")
    scope = receipt["scope"]
    _check(scope["maximum_payload_date"] == LAST_SIGNAL.isoformat(), "Typed maximum date differs")
    reserved = scope["reserved_outcome_interval"]
    _check(
        reserved
        == {
            "start": RESERVED_OUTCOME_START.isoformat(),
            "end": RESERVED_OUTCOME_END.isoformat(),
            "materialized": False,
            "separate_unblind_authority_required": True,
        },
        "Typed outcome reserve differs",
    )
    reserve_counter_key = (
        "post_2019_10_31_market_observation_or_interval_start_rows_retained"
        if current_receipt
        else "post_2019_10_31_payload_rows_retained"
    )
    _check(scope[reserve_counter_key] == 0, "Typed receipt reports reserved rows")
    locked = typed.load_production_settings()
    expected_sources = {
        name: {
            "filename": spec.filename,
            "sha256": spec.sha256,
            "size_bytes": spec.size_bytes,
            "full_file_rows": spec.full_file_rows,
        }
        for name, spec in sorted(locked.sources.items())
    }
    _check(receipt["source_bindings"] == expected_sources, "Typed frozen source bindings differ")
    _check(set(receipt["tables"]) == set(ALL_TYPED_TABLES), "Typed table role set differs")
    _check(
        {item.name for item in root.iterdir()}
        == {"typed_extension_receipt.json", *(f"{name}.parquet" for name in ALL_TYPED_TABLES)},
        "Typed directory is not closed",
    )
    actual: dict[str, dict[str, Any]] = {}
    for name, date_column in ALL_TYPED_TABLES.items():
        spec = locked.registry.tables[name]
        minimum = dt.date(2000, 1, 1) if name in INTERVAL_TABLES else EXTENSION_START
        schema = _expected_typed_schema(
            spec,
            minimum=minimum,
            maximum=LAST_SIGNAL,
            disclose_scope_semantics=current_receipt,
        )
        scan = _scan_parquet(
            _safe_file(root, f"{name}.parquet"),
            expected_schema=schema,
            primary_key=spec.primary_key,
            date_column=date_column,
            maximum_date=LAST_SIGNAL,
            minimum_date=None if name in INTERVAL_TABLES else EXTENSION_START,
        )
        declared = receipt["tables"][name]
        comparison = {key: scan[key] for key in declared}
        _check(declared == comparison, f"Typed table receipt differs: {name}")
        actual[name] = scan
    _check(
        actual["crsp_security_day"]["primary_key_sha256"]
        == actual["crsp_security_day_return_staging"]["primary_key_sha256"],
        "Daily consumer/staging keys differ",
    )
    _check(
        actual["crsp_security_month_crosscheck"]["primary_key_sha256"]
        == actual["crsp_security_month_return_staging"]["primary_key_sha256"],
        "Monthly consumer/staging keys differ",
    )
    structures = {
        "identity": _interval_structure(root / "crsp_identity_interval.parquet"),
        "shares": _interval_structure(root / "crsp_shares_interval.parquet"),
    }
    if current_receipt:
        _check(receipt["interval_structure"] == structures, "Interval structure receipt differs")
    authorizations = receipt["authorizations"]
    _check(
        authorizations["predictor_context_materialization_only"] is True
        and all(
            authorizations[key] is False
            for key in (
                "validation_labels_or_selection_authorized",
                "portfolio_weights_trades_pnl_or_performance_authorized",
                "final_2020_2025_access_authorized",
            )
        ),
        "Typed receipt authorization boundary differs",
    )
    return receipt, actual, receipt_bytes


def _verify_returns(
    root: Path,
    *,
    typed_root: Path,
    typed_actual: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], bytes]:
    receipt, receipt_bytes = _canonical_object(root / "return_extension_receipt.json", "return receipt")
    _check(
        set(receipt)
        == {
            "schema_version",
            "contract_id",
            "status",
            "build_id",
            "materialization_role",
            "scope",
            "bindings",
            "pointer_publication_attempted",
            "production_return_pointer_sha256_after",
            "rehearsal_pointer",
            "monthly_transport_audit",
            "authoritative_daily_return",
            "authorizations",
        },
        "Return receipt keys differ",
    )
    _check(receipt["schema_version"] == 1 and receipt["contract_id"] == CONTRACT_ID, "Return receipt identity differs")
    _check(receipt["status"] == "VALIDATION_PREDICTOR_RETURN_EXTENSION_BUILT_PENDING_VERIFICATION", "Return receipt status differs")
    _check(receipt["materialization_role"] == "point_in_time_predictor_context_only", "Return materialization role differs")
    _check(receipt["pointer_publication_attempted"] is False, "Return pointer publication was attempted")
    _check(receipt["monthly_transport_audit"] == EXPECTED_NS_AUDIT, "NS monthly transport audit differs")
    evidence_filters = [
        ("permno", "=", 16_493),
        ("calendar_date", ">=", dt.date(2016, 11, 1)),
        ("calendar_date", "<=", dt.date(2016, 11, 30)),
    ]
    evidence_daily = pq.read_table(
        typed_root / "crsp_security_day_return_staging.parquet",
        filters=evidence_filters,
        columns=[
            "calendar_date",
            "raw_total_return",
            "raw_return_missing_flag",
        ],
    ).to_pylist()
    evidence_market = pq.read_table(
        typed_root / "crsp_security_day.parquet",
        filters=evidence_filters,
        columns=["calendar_date", "diagnostic_price_usd_per_share"],
    ).to_pylist()
    _check(
        evidence_daily[0]["calendar_date"] == dt.date(2016, 11, 7)
        and evidence_daily[0]["raw_return_missing_flag"] == "MP"
        and evidence_daily[1]["calendar_date"] == dt.date(2016, 11, 8)
        and evidence_daily[1]["raw_return_missing_flag"] == "NS"
        and evidence_daily[1]["raw_total_return"] is None
        and evidence_market[1]["diagnostic_price_usd_per_share"]
        == Decimal("40.625000"),
        "NS partial-start source evidence differs",
    )
    evidence_gross = Decimal(1)
    evidence_numeric_dates: list[dt.date] = []
    for row in evidence_daily:
        value = row["raw_total_return"]
        if value is not None:
            evidence_numeric_dates.append(row["calendar_date"])
            evidence_gross *= Decimal(1) + value
    evidence_compound = (evidence_gross - Decimal(1)).quantize(
        Decimal("0.000000000001")
    )
    _check(
        evidence_numeric_dates[0] == dt.date(2016, 11, 9)
        and evidence_compound == Decimal("0.033844855499"),
        "NS partial-start daily compound evidence differs",
    )
    _check(receipt["production_return_pointer_sha256_after"] == receipt["bindings"]["production_return_pointer_sha256_before"], "Production return pointer changed")
    scope = receipt["scope"]
    _check(scope["start"] == EXTENSION_START.isoformat() and scope["end"] == LAST_SIGNAL.isoformat(), "Return scope differs")
    _check(
        scope["reserved_outcome_interval"]
        == {
            "start": RESERVED_OUTCOME_START.isoformat(),
            "end": RESERVED_OUTCOME_END.isoformat(),
            "materialized": False,
            "separate_unblind_authority_required": True,
        },
        "Return outcome reserve differs",
    )
    bindings = receipt["bindings"]
    _check(bindings["contract_id"] == CONTRACT_ID, "Return binding contract differs")
    _check(bindings["scope_start"] == EXTENSION_START.isoformat() and bindings["scope_end"] == LAST_SIGNAL.isoformat(), "Return binding scope differs")
    _check(bindings["reserved_outcome_start"] == RESERVED_OUTCOME_START.isoformat() and bindings["reserved_outcome_end"] == RESERVED_OUTCOME_END.isoformat(), "Return reserve binding differs")
    _check(
        bindings["typed_daily_staging_sha256"]
        == typed_actual["crsp_security_day_return_staging"]["sha256"]
        and bindings["typed_monthly_staging_sha256"]
        == typed_actual["crsp_security_month_return_staging"]["sha256"],
        "Return typed-input bindings differ",
    )
    from portfolio_bot.validation import validation_return_adapter

    _check(bindings["reused_reconciliation_source_sha256"] == _sha256(Path(returns.__file__).resolve()), "Frozen return source binding differs")
    _check(bindings["stage11_return_adapter_source_sha256"] == _sha256(Path(validation_return_adapter.__file__).resolve()), "Return adapter binding differs")
    market_source = PROJECT_ROOT / "src/portfolio_bot/validation/validation_market_extension.py"
    _check(bindings["stage11_market_extension_source_sha256"] == _sha256(market_source), "Market adapter binding differs")
    build_id = receipt["build_id"]
    interim = _safe_root(root / "interim" / build_id, "return interim root")
    quality = _safe_root(root / "quality" / build_id, "return quality root")
    artifact = _safe_root(root / "artifact" / build_id, "return artifact root")
    _check(
        {item.name for item in root.iterdir()}
        == {"interim", "quality", "artifact", "return_extension_receipt.json"},
        "Return root is not closed",
    )
    _check({item.name for item in (root / "interim").iterdir()} == {build_id}, "Return interim build set is not closed")
    _check({item.name for item in (root / "quality").iterdir()} == {build_id}, "Return quality build set is not closed")
    _check({item.name for item in (root / "artifact").iterdir()} == {build_id}, "Return artifact build set is not closed")
    expected_rows = {name: int(value) for name, value in bindings["expected_rows"].items()}
    registry = _registry_with_rows(returns.load_schema_registry(returns.SCHEMA_PATH), expected_rows)
    actual: dict[str, dict[str, Any]] = {}
    for name, date_column in RETURN_DATE_COLUMNS.items():
        availability = "outcome_availability_date" if name in {"crsp_delisting_event", "crsp_security_day_return"} else None
        authorization = "return_history_authorized" if name == "crsp_security_day_return" else None
        scan = _scan_parquet(
            _safe_file(interim, f"{name}.parquet"),
            expected_schema=registry.tables[name].arrow_schema(),
            primary_key=registry.tables[name].primary_key,
            date_column=date_column,
            maximum_date=LAST_SIGNAL,
            minimum_date=EXTENSION_START,
            availability_column=availability,
            authorization_column=authorization,
        )
        _check(scan["rows"] == expected_rows[name], f"Return row count differs: {name}")
        actual[name] = scan
    monthly_table = pq.read_table(
        interim / "crsp_security_month_return_reconciliation.parquet",
        columns=[
            "permno",
            "calendar_month",
            "raw_monthly_total_return",
            "monthly_flag_class",
            "compounded_raw_numeric_daily_total_return",
            "tolerance_used",
            "numeric_transport_status",
            "reconciliation_status",
        ],
    )
    diagnostic_mask = pc.equal(
        monthly_table["numeric_transport_status"],
        "new_security_partial_start_diagnostic_difference",
    )
    diagnostics = monthly_table.filter(diagnostic_mask).to_pylist()
    _check(
        diagnostics
        == [
            {
                "permno": 16_493,
                "calendar_month": dt.date(2016, 11, 30),
                "raw_monthly_total_return": Decimal("-0.125273"),
                "monthly_flag_class": "new_security_partial_start",
                "compounded_raw_numeric_daily_total_return": Decimal("0.033844855499"),
                "tolerance_used": Decimal("0.000060000000"),
                "numeric_transport_status": "new_security_partial_start_diagnostic_difference",
                "reconciliation_status": "accounted",
            }
        ],
        "Monthly NS diagnostic payload differs",
    )
    _check(
        pc.sum(
            pc.cast(
                pc.equal(monthly_table["reconciliation_status"], "hard_failure"),
                pa.int64(),
            )
        ).as_py()
        == 0,
        "Monthly reconciliation retains a hard failure",
    )
    _check(
        actual["crsp_security_day_return"]["primary_key_sha256"]
        == typed_actual["crsp_security_day_return_staging"]["primary_key_sha256"],
        "Reconciled daily/staging keys differ",
    )
    _check(
        actual["crsp_security_month_return_reconciliation"]["primary_key_sha256"]
        == typed_actual["crsp_security_month_return_staging"]["primary_key_sha256"],
        "Reconciled monthly/staging keys differ",
    )
    declared_daily = receipt["authoritative_daily_return"]
    _check(declared_daily == {key: actual["crsp_security_day_return"][key] for key in declared_daily}, "Authoritative daily receipt differs")
    manifest, manifest_bytes = _canonical_object(artifact / "manifest.json", "return manifest")
    _check({item.name for item in artifact.iterdir()} == {"manifest.json"}, "Return artifact directory is not closed")
    pointer = receipt["rehearsal_pointer"]
    _check(pointer["build_id"] == build_id and pointer["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest(), "Rehearsal pointer differs")
    _check(manifest["build_id"] == build_id and manifest["scope"] == {"start": EXTENSION_START.isoformat(), "end": LAST_SIGNAL.isoformat()}, "Return manifest scope differs")
    manifest_outputs = {item["path"]: item for item in manifest["outputs"]}
    _check(set(manifest_outputs) == {f"{name}.parquet" for name in RETURN_DATE_COLUMNS}, "Return manifest output set differs")
    for name, scan in actual.items():
        declared = manifest_outputs[f"{name}.parquet"]
        _check(
            declared
            == {
                "path": f"{name}.parquet",
                "rows": scan["rows"],
                "sha256": scan["sha256"],
                "size_bytes": scan["size_bytes"],
            },
            f"Return manifest payload differs: {name}",
        )
    success, success_bytes = _canonical_object(interim / "_SUCCESS.json", "return success")
    _check(
        {item.name for item in interim.iterdir()}
        == {"_SUCCESS.json", *(f"{name}.parquet" for name in RETURN_DATE_COLUMNS)},
        "Return interim directory is not closed",
    )
    summary, summary_bytes = _canonical_object(quality / "summary.json", "return summary")
    _check({item.name for item in quality.iterdir()} == {"summary.json", "checks.csv"}, "Return quality directory is not closed")
    _check(success_bytes == summary_bytes and success == summary, "Return success/quality summaries differ")
    _check(
        success["monthly_reconciliation_status"].get(
            "new_security_partial_start_diagnostic"
        )
        == 1,
        "Return summary omitted the NS diagnostic",
    )
    _check(success["authorizations"]["validation_or_final_test_access_authorized"] is False, "Inner return build self-authorized validation")
    _check(
        all(
            receipt["authorizations"][key] is False
            for key in (
                "validation_label_or_model_selection_access_authorized",
                "portfolio_or_performance_authorized",
                "final_2020_2025_access_authorized",
            )
        ),
        "Return receipt authorization boundary differs",
    )
    prohibited_pointer = PROJECT_ROOT / "artifacts/phase_05_validation/PROHIBITED_UNUSED_RETURN_POINTER.json"
    _check(not prohibited_pointer.exists(), "Prohibited return pointer exists")
    production_pointer = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_RETURN_RECONCILIATION_PRIMARY.json"
    current_pointer_hash = _sha256(production_pointer) if production_pointer.is_file() else None
    _check(current_pointer_hash == receipt["production_return_pointer_sha256_after"], "Production return pointer changed after build")
    return receipt, actual, receipt_bytes


def independently_verify_validation_market_extensions(
    typed_directory: Path,
    return_base_directory: Path,
) -> Mapping[str, Any]:
    """Return a deterministic zero-difference independent verification report."""

    typed_root = _safe_root(typed_directory, "typed predictor root")
    return_root = _safe_root(return_base_directory, "return predictor root")
    typed_receipt, typed_actual, typed_bytes = _verify_typed(typed_root)
    return_receipt, return_actual, return_bytes = _verify_returns(
        return_root,
        typed_root=typed_root,
        typed_actual=typed_actual,
    )
    preimage = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "verification_implementation_sha256": _sha256(Path(__file__).resolve()),
        "typed_receipt_sha256": hashlib.sha256(typed_bytes).hexdigest(),
        "return_receipt_sha256": hashlib.sha256(return_bytes).hexdigest(),
        "typed_payload_sha256": {
            name: value["sha256"] for name, value in sorted(typed_actual.items())
        },
        "return_payload_sha256": {
            name: value["sha256"] for name, value in sorted(return_actual.items())
        },
        "difference_count": 0,
    }
    verification_id = "verify-stage11-market-" + hashlib.sha256(
        _canonical_json(preimage)
    ).hexdigest()[:20]
    return MappingProxyType(
        {
            **preimage,
            "verification_id": verification_id,
            "status": "VERIFIED_PREDICTOR_MARKET_AND_RETURN_EXTENSIONS_PASSED",
            "scope": {
                "predictor_start": EXTENSION_START.isoformat(),
                "predictor_end": LAST_SIGNAL.isoformat(),
                "reserved_outcome_start": RESERVED_OUTCOME_START.isoformat(),
                "reserved_outcome_end": RESERVED_OUTCOME_END.isoformat(),
                "reserved_outcome_rows_or_observable_delistings": 0,
            },
            "checks": {
                "hash_size_footer_schema": "PASS",
                "sorted_unique_primary_keys": "PASS",
                "paired_key_identity": "PASS",
                "frozen_source_bindings": "PASS",
                "reserved_outcome_exclusion": "PASS",
                "production_pointer_unchanged": "PASS",
            },
            "authorizations": {
                "technical_predictor_extension_integrity_verified": True,
                "validation_outcome_access_authorized": False,
                "model_selection_authorized": False,
                "portfolio_weights_trades_pnl_or_performance_authorized": False,
                "final_2020_2025_access_authorized": False,
            },
            "typed_build_status": typed_receipt["status"],
            "return_build_status": return_receipt["status"],
        }
    )


def write_verification_report(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically write a canonical report, refusing a differing overwrite."""

    target = Path(os.path.abspath(path.expanduser()))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(dict(report))
    if target.exists():
        _check(target.is_file() and not target.is_symlink(), "Unsafe verification report")
        _check(target.read_bytes() == payload, "Refusing to overwrite a differing report")
        return
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, target)


__all__ = [
    "ValidationMarketVerificationError",
    "independently_verify_validation_market_extensions",
    "write_verification_report",
]
