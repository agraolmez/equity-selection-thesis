"""Deterministic raw-to-normalized extractor for Stage 11 Gate A.

This module is the only empirical slicer used by Gate A.  Its public entrypoint
requires the separate live source-materialization authority before it examines
the plan, a source path, or a work directory.  It produces only the ten
normalized main/reserve tables consumed by the outcome materializer; it never
constructs a target, forecast, portfolio, trade, P&L, or performance metric.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import duckdb

from portfolio_bot.data import phase_02_crsp_return_reconciliation as reconciliation
from portfolio_bot.data import phase_02_crsp_typed_backbone as typed
from portfolio_bot.data import phase_02_crsp_typed_backbone_discovery as discovery
from portfolio_bot.data import phase_02_crsp_typed_backbone_transforms as transforms
from portfolio_bot.data.phase_02_auxiliary_core import optional_decimal, required_date
from portfolio_bot.validation import validation_market_extension as market_extension
from portfolio_bot.validation import validation_return_adapter
from portfolio_bot.validation.validation_source_materialization import (
    MAIN_END,
    MAIN_START,
    RESERVE_END,
    RESERVE_START,
    Stage11SourceMaterializationAuthority,
    canonical_json_bytes,
    require_live_stage11_source_materialization_authority,
    sha256_bytes,
    sha256_file,
)


EXTRACTION_ID = "phase05-stage11-source-extraction-v1"
NORMALIZED_STATUS = "STAGE11_SOURCE_NORMALIZATION_COMPLETE"
ROW_GROUP_SIZE = 100_000

SESSION_SCHEMA = pa.schema([pa.field("calendar_date", pa.date32(), nullable=False)])
MARKET_SCHEMA = pa.schema(
    [
        pa.field("permno", pa.int64(), nullable=False),
        pa.field("calendar_date", pa.date32(), nullable=False),
        pa.field("close_usd_per_share", pa.decimal128(38, 6), nullable=True),
    ]
)
RETURN_SCHEMA = pa.schema(
    [
        pa.field("permno", pa.int64(), nullable=False),
        pa.field("calendar_date", pa.date32(), nullable=False),
        pa.field("return_leg_class", pa.string(), nullable=False),
        pa.field("return_span_start_date", pa.date32(), nullable=True),
        pa.field("return_span_end_date", pa.date32(), nullable=True),
        pa.field("covered_trading_sessions", pa.int16(), nullable=True),
        pa.field("span_alignment_status", pa.string(), nullable=False),
        pa.field("delisting_event_count", pa.int16(), nullable=False),
        pa.field("delisting_treatment_status", pa.string(), nullable=False),
        pa.field("delisting_adjustment_return", pa.decimal128(38, 12), nullable=True),
        pa.field("authoritative_total_return", pa.decimal128(38, 12), nullable=True),
        pa.field("outcome_availability_date", pa.date32(), nullable=False),
        pa.field("authoritative_return_state", pa.string(), nullable=False),
        pa.field("reconciliation_status", pa.string(), nullable=False),
        pa.field("return_history_usable", pa.bool_(), nullable=False),
        pa.field("return_history_authorized", pa.bool_(), nullable=False),
    ]
)
FACTOR_SCHEMA = pa.schema(
    [
        pa.field("observation_date", pa.date32(), nullable=False),
        pa.field("public_availability_date", pa.date32(), nullable=False),
        *(
            pa.field(name, pa.decimal128(38, 12), nullable=False)
            for name in ("mktrf", "smb", "hml", "rmw", "cma", "rf", "umd")
        ),
    ]
)
DELISTING_SCHEMA = pa.schema(
    [
        pa.field("permno", pa.int64(), nullable=False),
        pa.field("delisting_date", pa.date32(), nullable=False),
        pa.field("posting_date", pa.date32(), nullable=False),
        pa.field("action_type", pa.string(), nullable=False),
        pa.field("status_type", pa.string(), nullable=False),
        pa.field("reason_type", pa.string(), nullable=False),
        pa.field("payment_type", pa.string(), nullable=False),
        pa.field("successor_permno", pa.int64(), nullable=True),
        pa.field("successor_permco", pa.int64(), nullable=True),
        pa.field("observed_delisting_return", pa.decimal128(38, 6), nullable=True),
        pa.field("return_missing_type", pa.string(), nullable=False),
        pa.field("next_date", pa.date32(), nullable=True),
        pa.field("next_price", pa.decimal128(38, 6), nullable=True),
        pa.field("amount_date", pa.date32(), nullable=True),
        pa.field("delisting_dividend_amount", pa.decimal128(38, 6), nullable=True),
        pa.field("distribution_type", pa.string(), nullable=False),
        pa.field("primary_exchange", pa.string(), nullable=False),
        pa.field("performance_related", pa.bool_(), nullable=True),
        pa.field("embedding_status", pa.string(), nullable=False),
        pa.field("treatment_status", pa.string(), nullable=False),
        pa.field("effective_delisting_return", pa.decimal128(38, 12), nullable=True),
        pa.field("imputation_applied", pa.bool_(), nullable=False),
        pa.field("assigned_return_date", pa.date32(), nullable=True),
        pa.field("outcome_availability_date", pa.date32(), nullable=False),
        pa.field("assigned_exactly_once", pa.bool_(), nullable=False),
    ]
)

OUTPUT_SCHEMAS = MappingProxyType(
    {
        "sessions": SESSION_SCHEMA,
        "market": MARKET_SCHEMA,
        "returns": RETURN_SCHEMA,
        "factors": FACTOR_SCHEMA,
        "delistings": DELISTING_SCHEMA,
    }
)


class Stage11SourceExtractionError(RuntimeError):
    """Raised when a source, slice, normalization, or checkpoint differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11SourceExtractionError(message)


def schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def expected_output_schema(name: str) -> pa.Schema:
    for kind, schema in OUTPUT_SCHEMAS.items():
        if name.startswith(f"validation_{kind}_"):
            return schema
    raise Stage11SourceExtractionError(f"unknown normalized output role: {name}")


def _part_bounds(name: str) -> tuple[dt.date, dt.date]:
    if name.endswith("_main"):
        return MAIN_START, MAIN_END
    if name.endswith("_reserve"):
        return RESERVE_START, RESERVE_END
    raise Stage11SourceExtractionError(f"normalized output part is unknown: {name}")


def _source(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    sources = plan.get("sources")
    _require(
        isinstance(sources, dict) and name in sources, f"fixed source is absent: {name}"
    )
    value = sources[name]
    _require(isinstance(value, dict), f"fixed source record is invalid: {name}")
    return value


def _ordinary_path(path: Path, *, label: str, must_exist: bool) -> Path:
    _require(path.is_absolute(), f"{label} must be absolute")
    lexical = Path(os.path.abspath(path.expanduser()))
    cursor = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        cursor /= component
        if cursor.exists() or cursor.is_symlink():
            observed = os.lstat(cursor)
            _require(
                not stat.S_ISLNK(observed.st_mode), f"symlinked {label} is prohibited"
            )
    if must_exist:
        _require(lexical.exists(), f"{label} is absent")
    return lexical


def _duckdb_literal(value: str | Path) -> str:
    """Return a quoted DuckDB string literal for a previously validated path."""

    return "'" + str(value).replace("'", "''") + "'"


def _merge_ordered_daily_staging(
    main_path: Path,
    reserve_path: Path,
    output_path: Path,
) -> None:
    """Merge two schema-identical staging files in deterministic key order."""

    connection = duckdb.connect(":memory:")
    try:
        connection.execute("PRAGMA threads=1")
        connection.execute("PRAGMA preserve_insertion_order=false")
        connection.execute(
            "COPY (SELECT * FROM read_parquet(["
            f"{_duckdb_literal(main_path)}, {_duckdb_literal(reserve_path)}"
            "]) ORDER BY permno, calendar_date) "
            f"TO {_duckdb_literal(output_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
    finally:
        connection.close()


def _stable_small_source(record: Mapping[str, Any], *, label: str) -> bytes:
    path = _ordinary_path(Path(str(record["path"])), label=label, must_exist=True)
    before = os.lstat(path)
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, f"unsafe {label}")
    payload = path.read_bytes()
    after = os.lstat(path)
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
    _require(len(payload) == int(record["size_bytes"]), f"{label} size differs")
    _require(sha256_bytes(payload) == record["sha256"], f"{label} SHA-256 differs")
    return payload


def _authenticate_source_file(
    plan: Mapping[str, Any],
    source_name: str,
    *,
    label: str | None = None,
) -> Path:
    """Authenticate one fixed regular source without loading it into memory."""

    record = _source(plan, source_name)
    selected_label = source_name if label is None else label
    path = _ordinary_path(
        Path(str(record["path"])), label=selected_label, must_exist=True
    )
    before = os.lstat(path)
    _require(
        stat.S_ISREG(before.st_mode)
        and before.st_nlink == 1
        and before.st_size == int(record["size_bytes"]),
        f"unsafe or changed {selected_label}",
    )
    digest = sha256_file(path)
    after = os.lstat(path)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    _require(
        identity(before) == identity(after), f"{selected_label} changed while hashed"
    )
    _require(digest == record["sha256"], f"{selected_label} SHA-256 differs")
    return path


class _Sink:
    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.path = path
        self.schema = schema
        self.rows = 0
        self._writer = pq.ParquetWriter(
            path,
            schema,
            version="2.6",
            compression="zstd",
            compression_level=9,
            use_dictionary=True,
            write_statistics=True,
            data_page_version="2.0",
        )

    def add(self, table: pa.Table) -> None:
        if not table.num_rows:
            return
        _require(
            table.schema.equals(self.schema, check_metadata=True),
            f"normalized schema differs while writing {self.path.name}",
        )
        self._writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
        self.rows += table.num_rows

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "_Sink":
        return self

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False


def _clean(table: pa.Table, schema: pa.Schema) -> pa.Table:
    selected = table.select(schema.names)
    arrays = [
        pc.cast(selected[name], field.type, safe=True)
        for name, field in zip(schema.names, schema, strict=True)
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _slice(
    table: pa.Table, column: str, minimum: dt.date, maximum: dt.date
) -> pa.Table:
    mask = pc.and_(
        pc.greater_equal(table[column], minimum), pc.less_equal(table[column], maximum)
    )
    return table.filter(mask)


def _table_receipt(path: Path, *, date_column: str) -> dict[str, Any]:
    before = os.lstat(path)
    _require(
        stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
        f"unsafe normalized table: {path.name}",
    )
    parquet = pq.ParquetFile(path)
    minimum: dt.date | None = None
    maximum: dt.date | None = None
    for index in range(parquet.num_row_groups):
        values = parquet.read_row_group(index, columns=[date_column])[date_column]
        _require(values.null_count == 0, f"normalized date contains nulls: {path.name}")
        if len(values):
            extrema = pc.min_max(values).as_py()
            minimum = (
                extrema["min"] if minimum is None else min(minimum, extrema["min"])
            )
            maximum = (
                extrema["max"] if maximum is None else max(maximum, extrema["max"])
            )
    digest = sha256_file(path)
    after = os.lstat(path)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    _require(
        identity(before) == identity(after),
        f"normalized table changed while receipted: {path.name}",
    )
    return {
        "path": path.name,
        "sha256": digest,
        "size_bytes": before.st_size,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "schema_sha256": schema_sha256(parquet.schema_arrow),
        "minimum_date": None if minimum is None else minimum.isoformat(),
        "maximum_date": None if maximum is None else maximum.isoformat(),
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _require(not temporary.exists(), f"stale temporary JSON exists: {path.name}")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet_writer(path: Path) -> tuple[Path, Path]:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _require(
        not path.exists() and not temporary.exists(),
        f"unreceipted or stale output exists: {path.name}",
    )
    return path, temporary


def _commit_temp(temporary: Path, destination: Path) -> None:
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def _output_contract(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = plan["outputs"][name]
    schema = expected_output_schema(name)
    minimum, maximum = _part_bounds(name)
    _require(
        value["filename"] == f"{name}.parquet", f"fixed output filename differs: {name}"
    )
    _require(
        tuple(value["allowed_columns"]) == tuple(schema.names),
        f"fixed output columns differ: {name}",
    )
    _require(
        value["schema_sha256"] == schema_sha256(schema),
        f"fixed output schema differs: {name}",
    )
    _require(
        value["minimum_date"] == minimum.isoformat()
        and value["maximum_date"] == maximum.isoformat(),
        f"fixed output dates differ: {name}",
    )
    _require(
        value["date_column"] == _date_column(name),
        f"fixed output date column differs: {name}",
    )
    _require(
        value["predictor_use_permitted"] is False, f"predictor use was widened: {name}"
    )
    return value


def _write_sessions(plan: Mapping[str, Any], normalized: Path) -> None:
    record = _source(plan, "trading_calendar")
    payload = _stable_small_source(record, label="verified trading calendar")
    source = pq.read_table(pa.BufferReader(payload), columns=["calendar_date"])
    _require(source["calendar_date"].null_count == 0, "trading calendar contains nulls")
    source = _clean(source, SESSION_SCHEMA)
    for part, minimum, maximum in (
        ("main", MAIN_START, MAIN_END),
        ("reserve", RESERVE_START, RESERVE_END),
    ):
        name = f"validation_sessions_{part}"
        _output_contract(plan, name)
        path, temporary = _atomic_parquet_writer(normalized / f"{name}.parquet")
        try:
            table = _slice(source, "calendar_date", minimum, maximum)
            _require(
                table.num_rows > 0
                and table["calendar_date"][0].as_py() == minimum
                and table["calendar_date"][-1].as_py() == maximum,
                f"trading session slice is incomplete: {part}",
            )
            pq.write_table(
                table,
                temporary,
                compression="zstd",
                compression_level=9,
                use_dictionary=True,
                row_group_size=ROW_GROUP_SIZE,
                version="2.6",
            )
            _commit_temp(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _write_factors(plan: Mapping[str, Any], normalized: Path) -> None:
    record = _source(plan, "ff_factor_daily")
    payload = _stable_small_source(record, label="frozen daily factor source")
    text = payload.decode("utf-8-sig")
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(text.splitlines())
    expected_raw = tuple(record["allowed_columns"])
    _require(
        tuple(reader.fieldnames or ()) == expected_raw, "daily factor header differs"
    )
    for raw in reader:
        observation = required_date(raw["date"], "factor date")
        if observation < MAIN_START or observation > RESERVE_END:
            continue
        row: dict[str, Any] = {
            "observation_date": observation,
            "public_availability_date": observation,
        }
        for name in ("mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"):
            value = optional_decimal(raw[name], scale=12, label=name)
            _require(isinstance(value, Decimal), f"daily factor is blank: {name}")
            row[name] = value
        rows.append(row)
    full = pa.Table.from_pylist(rows, schema=FACTOR_SCHEMA)
    _require(full.num_rows > 0, "daily factor scope is empty")
    for part, minimum, maximum in (
        ("main", MAIN_START, MAIN_END),
        ("reserve", RESERVE_START, RESERVE_END),
    ):
        name = f"validation_factors_{part}"
        _output_contract(plan, name)
        if part == "main":
            _project_verified_source(
                plan,
                source_name="verified_factor_main",
                output_name=name,
                normalized=normalized,
                schema=FACTOR_SCHEMA,
                date_column="observation_date",
            )
            projected = pq.read_table(normalized / f"{name}.parquet")
            raw_expected = _slice(full, "observation_date", minimum, maximum)
            _require(
                projected.equals(raw_expected),
                "verified main factor projection differs from its frozen raw lineage",
            )
            continue
        path, temporary = _atomic_parquet_writer(normalized / f"{name}.parquet")
        try:
            table = _slice(full, "observation_date", minimum, maximum)
            _require(table.num_rows > 0, f"daily factor slice is empty: {part}")
            pq.write_table(
                table,
                temporary,
                compression="zstd",
                compression_level=9,
                use_dictionary=True,
                row_group_size=ROW_GROUP_SIZE,
                version="2.6",
            )
            _commit_temp(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _compare_raw_source(
    locked: Any, plan: Mapping[str, Any], locked_name: str, plan_name: str
) -> None:
    specification = locked.sources[locked_name]
    record = _source(plan, plan_name)
    _require(
        record["path"] == str(locked.raw_root / specification.filename)
        and record["sha256"] == specification.sha256
        and int(record["size_bytes"]) == specification.size_bytes
        and int(record["full_file_rows"]) == specification.full_file_rows,
        f"frozen raw identity differs: {plan_name}",
    )


def _project_verified_source(
    plan: Mapping[str, Any],
    *,
    source_name: str,
    output_name: str,
    normalized: Path,
    schema: pa.Schema,
    date_column: str,
) -> None:
    """Reauthenticate and project one already independently verified main table."""

    record = _source(plan, source_name)
    source = _ordinary_path(
        Path(str(record["path"])), label=source_name, must_exist=True
    )
    before = os.lstat(source)
    _require(
        stat.S_ISREG(before.st_mode)
        and before.st_nlink == 1
        and before.st_size == int(record["size_bytes"]),
        f"unsafe or changed verified source: {source_name}",
    )
    _require(
        sha256_file(source) == record["sha256"],
        f"verified source SHA-256 differs: {source_name}",
    )
    minimum, maximum = _part_bounds(output_name)
    destination, temporary = _atomic_parquet_writer(
        normalized / f"{output_name}.parquet"
    )
    writer = _Sink(temporary, schema)
    try:
        parquet = pq.ParquetFile(source)
        _require(
            set(schema.names).issubset(parquet.schema_arrow.names),
            f"verified source projection differs: {source_name}",
        )
        for batch in parquet.iter_batches(
            batch_size=131_072,
            columns=schema.names,
            use_threads=False,
        ):
            table = _clean(pa.Table.from_batches([batch]), schema)
            writer.add(_slice(table, date_column, minimum, maximum))
    finally:
        writer.close()
    try:
        _commit_temp(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    after = os.lstat(source)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    _require(
        identity(before) == identity(after),
        f"verified source changed while projected: {source_name}",
    )


def _write_daily_and_staging(
    plan: Mapping[str, Any], normalized: Path, typed_root: Path
) -> None:
    locked = typed.load_production_settings(block_size_bytes=64 << 20)
    _compare_raw_source(locked, plan, "daily", "crsp_daily")
    _output_contract(plan, "validation_market_main")
    _project_verified_source(
        plan,
        source_name="verified_market_main",
        output_name="validation_market_main",
        normalized=normalized,
        schema=MARKET_SCHEMA,
        date_column="calendar_date",
    )
    settings = dataclasses.replace(
        locked, start_date=RESERVE_START, end_date=RESERVE_END
    )
    market_path, market_temp = _atomic_parquet_writer(
        normalized / "validation_market_reserve.parquet"
    )
    _output_contract(plan, "validation_market_reserve")
    reserve_staging_path, reserve_staging_temp = _atomic_parquet_writer(
        typed_root / "reserve_crsp_security_day_return_staging.parquet"
    )
    market_writer = _Sink(market_temp, MARKET_SCHEMA)
    staging_schema = market_extension.validation_table_schema(
        locked.registry.tables["crsp_security_day_return_staging"],
        minimum_date=RESERVE_START,
        maximum_date=RESERVE_END,
    )
    staging_writer = market_extension._ValidationParquetSink(
        reserve_staging_temp, staging_schema
    )
    metrics = typed.CanonicalizationMetrics()
    try:
        with discovery._AuthenticatedSource.open(
            locked.raw_root, locked.sources["daily"]
        ) as source:
            scoped = typed._scoped_market_batches(
                source, date_column="DlyCalDt", settings=settings, metrics=metrics
            )
            for raw in typed.canonicalize_daily_batches(scoped, metrics):
                consumer, staging = transforms.transform_daily(
                    raw,
                    locked.registry.tables["crsp_security_day"],
                    locked.registry.tables["crsp_security_day_return_staging"],
                )
                staging_writer.add(staging)
                market = _clean(consumer, MARKET_SCHEMA)
                market_writer.add(
                    _slice(market, "calendar_date", RESERVE_START, RESERVE_END)
                )
        _require(
            metrics.conflicting_groups == 0,
            "daily raw source has conflicting duplicate groups",
        )
    finally:
        staging_writer.close()
        market_writer.close()
    try:
        _commit_temp(reserve_staging_temp, reserve_staging_path)
        _commit_temp(market_temp, market_path)
    finally:
        for temporary in (reserve_staging_temp, market_temp):
            if temporary.exists():
                temporary.unlink()

    # Reuse the authenticated main staging payload and merge only the new
    # reserve rows.  The ordered merge is deterministic and avoids parsing or
    # transforming five years of CRSP market rows a second time.
    main_path = _authenticate_source_file(
        plan,
        "verified_daily_return_staging_main",
        label="verified main daily return staging",
    )
    full_path = typed_root / "crsp_security_day_return_staging.parquet"
    temporary = full_path.with_name(f".{full_path.name}.{os.getpid()}.tmp")
    _require(
        not full_path.exists() and not temporary.exists(),
        "daily staging merge output exists",
    )
    _merge_ordered_daily_staging(main_path, reserve_staging_path, temporary)
    _commit_temp(temporary, full_path)


@contextlib.contextmanager
def _full_validation_return_scope() -> Iterator[None]:
    acquired = market_extension._RECONCILIATION_SCOPE_LOCK.acquire(blocking=False)
    _require(acquired, "another return-reconciliation scope is active")
    previous = (
        reconciliation.SCOPE_START,
        reconciliation.SCOPE_END,
        reconciliation._read_events,
        reconciliation._distribution_mapping,
        reconciliation._delisting_treatments,
        reconciliation._build_monthly,
    )
    reconciliation.SCOPE_START = dt.date(2015, 1, 1)
    reconciliation.SCOPE_END = RESERVE_END
    reconciliation._read_events = validation_return_adapter.validation_read_events
    reconciliation._distribution_mapping = (
        validation_return_adapter.validation_distribution_mapping
    )
    reconciliation._delisting_treatments = (
        validation_return_adapter.validation_delisting_treatments
    )
    reconciliation._build_monthly = validation_return_adapter.validation_build_monthly
    validation_return_adapter.reset_monthly_transport_audit()
    try:
        yield
    finally:
        (
            reconciliation.SCOPE_START,
            reconciliation.SCOPE_END,
            reconciliation._read_events,
            reconciliation._distribution_mapping,
            reconciliation._delisting_treatments,
            reconciliation._build_monthly,
        ) = previous
        market_extension._RECONCILIATION_SCOPE_LOCK.release()


def _copy_verified_monthly(plan: Mapping[str, Any], typed_root: Path) -> None:
    record = _source(plan, "verified_monthly_return_staging")
    source = _authenticate_source_file(
        plan,
        "verified_monthly_return_staging",
        label="verified monthly return staging",
    )
    destination = typed_root / "crsp_security_month_return_staging.parquet"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    _require(
        not destination.exists() and not temporary.exists(),
        "monthly staging work output already exists",
    )
    try:
        shutil.copyfile(source, temporary)
        _require(
            sha256_file(temporary) == record["sha256"], "monthly staging copy differs"
        )
        _commit_temp(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_reconciliation(plan: Mapping[str, Any], typed_root: Path, work: Path) -> Path:
    # Historical main outputs are authenticated as non-governing parity
    # references.  The full-scope recomputation remains authoritative because
    # late-observable terminal events can legitimately update main-date rows.
    _authenticate_source_file(plan, "verified_return_main_parity")
    _authenticate_source_file(plan, "verified_delisting_main_parity")
    base = reconciliation.load_settings(output_database_root=work / "settings_probe")
    daily_rows = pq.ParquetFile(
        typed_root / "crsp_security_day_return_staging.parquet"
    ).metadata.num_rows
    monthly_rows = pq.ParquetFile(
        typed_root / "crsp_security_month_return_staging.parquet"
    ).metadata.num_rows
    # load_settings has already rehashed the exact distribution/delisting
    # files against the frozen empirical lock.  Bind those same identities to
    # the Gate-A plan before the date-first event reader is permitted to run.
    for source_name, binding_name in (
        ("crsp_distribution", "raw_distribution_sha256"),
        ("crsp_delisting", "raw_delisting_sha256"),
    ):
        record = _source(plan, source_name)
        expected_path = base.raw_root / Path(str(record["path"])).name
        _require(
            Path(str(record["path"])) == expected_path
            and base.empirical_lock["bindings"].get(binding_name) == record["sha256"]
            and expected_path.stat().st_size == int(record["size_bytes"]),
            f"Gate-A event-source identity differs: {source_name}",
        )
    probe = dataclasses.replace(base, typed_root=typed_root)
    with _full_validation_return_scope():
        distributions, delistings, _context = reconciliation._read_events(probe)
    expected_rows = {
        "crsp_distribution_event": len(distributions),
        "crsp_delisting_event": len(delistings),
        "crsp_security_day_return": int(daily_rows),
        "crsp_security_month_return_reconciliation": int(monthly_rows),
    }
    registry = market_extension._registry_with_expected_rows(base.schema, expected_rows)
    identity = {
        "extraction_id": EXTRACTION_ID,
        "plan_payload_sha256": plan["payload_sha256"],
        "typed_daily_sha256": sha256_file(
            typed_root / "crsp_security_day_return_staging.parquet"
        ),
        "monthly_sha256": sha256_file(
            typed_root / "crsp_security_month_return_staging.parquet"
        ),
        "expected_rows": expected_rows,
    }
    build_id = (
        "stage11-source-return-" + sha256_bytes(canonical_json_bytes(identity))[:24]
    )
    output = work / "return_reconciliation"
    settings = reconciliation.Settings(
        database_root=base.database_root,
        output_database_root=work,
        is_production=False,
        raw_root=base.raw_root,
        typed_root=typed_root,
        calendar_root=base.calendar_root,
        schema=registry,
        empirical_lock=base.empirical_lock,
        build_id=build_id,
        interim_dir=output / "interim" / build_id,
        quality_dir=output / "quality" / build_id,
        artifact_dir=output / "artifact" / build_id,
        pointer_path=output / "PROHIBITED_UNUSED_POINTER.json",
    )
    _require(
        not settings.interim_dir.exists(),
        "unreceipted return-reconciliation work exists",
    )
    with _full_validation_return_scope():
        reconciliation.build(settings, publish_pointer=False)
    return settings.interim_dir


def _split_parquet(
    source: Path,
    normalized: Path,
    *,
    kind: str,
    date_column: str,
    plan: Mapping[str, Any],
) -> None:
    schema = OUTPUT_SCHEMAS[kind]
    targets: dict[str, tuple[Path, Path]] = {}
    writers: dict[str, _Sink] = {}
    for part in ("main", "reserve"):
        name = f"validation_{kind}_{part}"
        _output_contract(plan, name)
        targets[part] = _atomic_parquet_writer(normalized / f"{name}.parquet")
        writers[part] = _Sink(targets[part][1], schema)
    try:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(
            batch_size=131_072, columns=schema.names, use_threads=False
        ):
            table = _clean(pa.Table.from_batches([batch]), schema)
            writers["main"].add(_slice(table, date_column, MAIN_START, MAIN_END))
            writers["reserve"].add(
                _slice(table, date_column, RESERVE_START, RESERVE_END)
            )
    finally:
        for writer in writers.values():
            writer.close()
    try:
        for destination, temporary in targets.values():
            _commit_temp(temporary, destination)
    finally:
        for _destination, temporary in targets.values():
            if temporary.exists():
                temporary.unlink()


def _date_column(name: str) -> str:
    if "sessions" in name or "market" in name or "returns" in name:
        return "calendar_date"
    if "factors" in name:
        return "observation_date"
    if "delistings" in name:
        return "assigned_return_date"
    raise Stage11SourceExtractionError(f"date column is unknown for {name}")


def _authenticate_completed_work(
    plan: Mapping[str, Any], normalized: Path
) -> Mapping[str, Any]:
    receipts: dict[str, Any] = {}
    for name in sorted(plan["outputs"]):
        contract = _output_contract(plan, name)
        path = normalized / f"{name}.parquet"
        _require(
            path.is_file() and not path.is_symlink(),
            f"normalized output is absent: {name}",
        )
        receipt = _table_receipt(path, date_column=_date_column(name))
        _require(
            receipt["schema_sha256"] == contract["schema_sha256"],
            f"normalized output schema differs: {name}",
        )
        _require(
            receipt["minimum_date"] is not None and receipt["maximum_date"] is not None,
            f"normalized output is empty: {name}",
        )
        _require(
            receipt["minimum_date"] >= contract["minimum_date"]
            and receipt["maximum_date"] <= contract["maximum_date"],
            f"normalized output escaped its date boundary: {name}",
        )
        receipts[name] = receipt
    return MappingProxyType(receipts)


@dataclass(frozen=True, slots=True)
class NormalizedSourceBuild:
    work_root: Path
    normalized_root: Path
    build_id: str
    input_fingerprint: str
    receipts: Mapping[str, Mapping[str, Any]]
    reused: bool


def _materialize_normalized_stage11_sources(
    authority: Stage11SourceMaterializationAuthority,
    *,
    purpose: str,
    work_root_field: str,
) -> NormalizedSourceBuild:
    live = require_live_stage11_source_materialization_authority(authority)
    plan = live.plan
    publication = plan["publication"]
    _require(
        purpose in {"primary", "independent_verification_replay"},
        "unknown Gate-A normalization purpose",
    )
    fingerprint = sha256_bytes(
        canonical_json_bytes(
            {
                "authority_sha256": live.authority_sha256,
                "extraction_id": EXTRACTION_ID,
                "implementation_code_hashes": plan["implementation_code_hashes"],
                "plan_payload_sha256": plan["payload_sha256"],
                "purpose": purpose,
                "sources": plan["sources"],
            }
        )
    )
    prefix = (
        "stage11-sources-" if purpose == "primary" else "stage11-sources-verify-replay-"
    )
    build_id = f"{prefix}{fingerprint[:24]}"
    work_parent = _ordinary_path(
        Path(str(publication[work_root_field])),
        label="Gate-A source work root",
        must_exist=False,
    )
    work_parent.mkdir(parents=True, exist_ok=True)
    work = work_parent / build_id
    normalized = work / "normalized"
    complete = work / "source_normalization_receipt.json"
    if complete.exists():
        value = json.loads(complete.read_text("utf-8"))
        _require(
            value.get("input_fingerprint") == fingerprint
            and value.get("status") == NORMALIZED_STATUS,
            "completed source work receipt differs",
        )
        receipts = _authenticate_completed_work(plan, normalized)
        _require(
            dict(receipts) == value.get("outputs"),
            "completed normalized outputs changed",
        )
        return NormalizedSourceBuild(
            work, normalized, build_id, fingerprint, receipts, True
        )

    _require(
        not work.exists(),
        "incomplete unreceipted source work exists; preserve it for audit before retry",
    )
    work.mkdir(parents=True)
    normalized.mkdir()
    typed_root = work / "typed"
    typed_root.mkdir()
    try:
        _write_sessions(plan, normalized)
        _write_factors(plan, normalized)
        _write_daily_and_staging(plan, normalized, typed_root)
        _copy_verified_monthly(plan, typed_root)
        reconciliation_root = _run_reconciliation(plan, typed_root, work)
        _split_parquet(
            reconciliation_root / "crsp_security_day_return.parquet",
            normalized,
            kind="returns",
            date_column="calendar_date",
            plan=plan,
        )
        _split_parquet(
            reconciliation_root / "crsp_delisting_event.parquet",
            normalized,
            kind="delistings",
            date_column="assigned_return_date",
            plan=plan,
        )
        receipts = _authenticate_completed_work(plan, normalized)
        receipt = {
            "schema_version": 1,
            "status": NORMALIZED_STATUS,
            "extraction_id": EXTRACTION_ID,
            "build_id": build_id,
            "input_fingerprint": fingerprint,
            "authority_sha256": live.authority_sha256,
            "plan_sha256": live.plan_sha256,
            "plan_payload_sha256": live.plan_payload_sha256,
            "outputs": dict(receipts),
            "authorizations": {
                "source_normalization_only": True,
                "labels_forecasts_weights_trades_pnl_performance_created": False,
                "final_2020_plus_rows_read": 0,
            },
        }
        _atomic_json(complete, receipt)
        require_live_stage11_source_materialization_authority(live)
        return NormalizedSourceBuild(
            work, normalized, build_id, fingerprint, receipts, False
        )
    except BaseException:
        # Preserve incomplete work as evidence.  Nothing is published and the
        # next run fails closed until the incomplete directory is reviewed.
        raise


def materialize_normalized_stage11_sources(
    authority: Stage11SourceMaterializationAuthority,
) -> NormalizedSourceBuild:
    """Build or resume the fixed ten normalized Gate-A source shards."""

    return _materialize_normalized_stage11_sources(
        authority,
        purpose="primary",
        work_root_field="work_root",
    )


def recompute_normalized_stage11_sources_for_verification(
    authority: Stage11SourceMaterializationAuthority,
) -> NormalizedSourceBuild:
    """Independently replay main projections and reserve raw normalization."""

    return _materialize_normalized_stage11_sources(
        authority,
        purpose="independent_verification_replay",
        work_root_field="verification_work_root",
    )


__all__ = [
    "DELISTING_SCHEMA",
    "EXTRACTION_ID",
    "FACTOR_SCHEMA",
    "MARKET_SCHEMA",
    "NORMALIZED_STATUS",
    "NormalizedSourceBuild",
    "OUTPUT_SCHEMAS",
    "RETURN_SCHEMA",
    "SESSION_SCHEMA",
    "Stage11SourceExtractionError",
    "expected_output_schema",
    "materialize_normalized_stage11_sources",
    "recompute_normalized_stage11_sources_for_verification",
    "schema_sha256",
]
