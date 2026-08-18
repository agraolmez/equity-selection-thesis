"""Secure, single-scan return routing for Stage 7 label construction.

The verified return table is ordered by security while Stage 7 materializes
one complete signal-date cross-section at a time.  This module expands the 178
global holding intervals into a tiny date-to-signal map, performs exact hash
joins in DuckDB, writes a signal-sorted immutable Parquet spool, and exposes
bounded signal-date readers.  The expansion avoids a large security-by-range
intermediate while preserving the inclusive entry/exit routing boundary.  It
does not calculate labels or publish any primary Stage 7 artifact.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    AuthenticatedBundleGuard,
    AuthenticatedPointerGuard,
    FileRecord,
    SecurePublicationError,
    SecureStagingDirectory,
)
from portfolio_bot.features.phase_03_contract import canonical_json_bytes


_RETURN_TABLE = "crsp_security_day_return.parquet"
_PARQUET_NAME = "routed_returns.parquet"
_SUCCESS_NAME = "_SUCCESS.json"
_STATUS = "LABEL_RETURN_ROUTER_COMPLETE"
_CUTOFF = dt.date(2014, 12, 1)
_EXIT_STRADDLE_PROBE_DAYS = 45
_SAFE_COMPONENT = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_FILES = (_SUCCESS_NAME, _PARQUET_NAME)
_SUCCESS_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "build_id",
        "source_fingerprint",
        "intervals_sha256",
        "cutoff_date",
        "projected_return_columns",
        "routed_row_count",
        "rows_by_signal",
        "table_receipt",
    }
)
_TABLE_RECEIPT_KEYS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "rows",
        "row_groups",
        "schema_sha256",
    }
)


class LabelReturnRoutingError(RuntimeError):
    """Raised when the bounded Stage 7 return router fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LabelReturnRoutingError(message)


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    # Keep the exact ordering used by LabelInputTable.stat_identity.
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_gid),
    )


def _same_stat_except_ctime(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> bool:
    return all(
        left[index] == right[index]
        for index in range(len(left))
        if index != 4
    )


def _safe_text(value: object, label: str) -> str:
    result = str(value)
    _require(
        _SAFE_COMPONENT.fullmatch(result) is not None,
        f"Unsafe Stage 7 return-router {label}",
    )
    return result


def _safe_fingerprint(value: object) -> str:
    result = str(value)
    _require(
        _SHA256.fullmatch(result) is not None,
        "Unsafe Stage 7 return-router source fingerprint",
    )
    return result


def _safe_destination(database_root: Path, destination: Path) -> Path:
    root = Path(os.path.abspath(Path(database_root).expanduser()))
    target = Path(os.path.abspath(Path(destination).expanduser()))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise LabelReturnRoutingError(
            "Stage 7 return-router destination escaped database root"
        ) from exc
    _require(
        bool(relative.parts) and ".." not in relative.parts,
        "Unsafe Stage 7 return-router destination",
    )
    _safe_text(target.name, "destination")
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise LabelReturnRoutingError(
            "Cannot inspect Stage 7 return-router database root"
        ) from exc
    _require(
        stat.S_ISDIR(root_stat.st_mode) and not stat.S_ISLNK(root_stat.st_mode),
        "Stage 7 return-router database root must be a real directory",
    )
    cursor = root
    for part in relative.parts[:-1]:
        cursor /= part
        if not cursor.exists():
            continue
        observed = os.lstat(cursor)
        _require(
            stat.S_ISDIR(observed.st_mode)
            and not stat.S_ISLNK(observed.st_mode),
            "Stage 7 return-router parent must be a real directory",
        )
    return target


def _source_snapshot(inputs: Any) -> tuple[Any, ...]:
    """Return a cheap, stable identity token for the authenticated source."""

    custom = getattr(inputs, "router_source_snapshot", None)
    if callable(custom):
        return ("custom", custom())
    root = Path(os.path.abspath(Path(inputs.database_root).expanduser()))
    root_stat = os.lstat(root)
    _require(
        stat.S_ISDIR(root_stat.st_mode) and not stat.S_ISLNK(root_stat.st_mode),
        "Stage 7 input root changed",
    )
    tables = getattr(inputs, "tables", None)
    _require(
        isinstance(tables, Mapping) and _RETURN_TABLE in tables,
        "Stage 7 return input receipt is unavailable",
    )
    table = tables[_RETURN_TABLE]
    raw_path = getattr(table, "_path", None)
    _require(raw_path is not None, "Stage 7 return input path is unavailable")
    path = Path(os.path.abspath(Path(raw_path).expanduser()))
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise LabelReturnRoutingError(
            "Stage 7 return input escaped database root"
        ) from exc
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        "Stage 7 return input is not a regular file",
    )
    declared_identity = getattr(table, "stat_identity", None)
    if declared_identity is not None:
        _require(
            tuple(declared_identity) == _stat_identity(observed),
            "Stage 7 return input identity changed",
        )
    return (
        "catalog",
        (
            int(root_stat.st_dev),
            int(root_stat.st_ino),
            int(root_stat.st_mode),
            int(root_stat.st_uid),
            int(root_stat.st_gid),
        ),
        path.as_posix(),
        _stat_identity(observed),
        getattr(table, "sha256", None),
        getattr(table, "schema_sha256", None),
        getattr(table, "size_bytes", None),
        getattr(table, "rows", None),
        getattr(table, "row_groups", None),
        tuple(getattr(table, "projected_columns", ())),
    )


def _validate_intervals(intervals: pa.Table) -> pa.Table:
    _require(isinstance(intervals, pa.Table), "Stage 7 intervals must be Arrow")
    names = ("signal_date", "permno", "entry_date", "exit_date")
    _require(
        set(names).issubset(intervals.schema.names),
        "Stage 7 interval columns differ",
    )
    selected = intervals.select(names).combine_chunks()
    expected_types = (pa.date32(), pa.int64(), pa.date32(), pa.date32())
    _require(
        tuple(field.type for field in selected.schema) == expected_types,
        "Stage 7 interval types differ",
    )
    _require(selected.num_rows > 0, "Stage 7 intervals cannot be empty")
    _require(
        all(selected[name].null_count == 0 for name in names),
        "Stage 7 intervals contain null keys or dates",
    )
    rows = selected.to_pylist()
    keys: set[tuple[dt.date, int]] = set()
    for row in rows:
        key = (row["signal_date"], int(row["permno"]))
        _require(key not in keys, "Duplicate Stage 7 interval key")
        keys.add(key)
        _require(
            row["signal_date"] < row["entry_date"] < row["exit_date"]
            and row["exit_date"] <= _CUTOFF,
            "Stage 7 interval escaped the development boundary",
        )
    return selected


def _intervals_sha256(intervals: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, intervals.schema) as writer:
        writer.write_table(intervals)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def compute_label_intervals_sha256(intervals: pa.Table) -> str:
    """Hash the canonical validated interval payload bound to a router."""

    return _intervals_sha256(_validate_intervals(intervals))


def _routing_relations(
    intervals: pa.Table,
) -> tuple[pa.Table, pa.Table]:
    """Return exact long keys and a small date-to-signal routing map.

    Normal rows cover signal through exit.  Ordinary signal-to-entry returns
    are ignored by label math, while a terminal payoff on/before entry makes
    the prospective holding non-executable.  A bounded post-exit probe also
    routes a multi-session row whose span crosses the exit, so the label
    receives ``MISSING_INTERVAL_STRADDLE`` rather than a false gap.
    The authenticated source has at most ten covered sessions; forty-five
    calendar days is a conservative detection envelope.
    """

    rows = intervals.to_pylist()
    calendar: dict[dt.date, tuple[dt.date, dt.date]] = {}
    for row in rows:
        signal = row["signal_date"]
        boundary = (row["entry_date"], row["exit_date"])
        previous = calendar.setdefault(signal, boundary)
        _require(
            previous == boundary,
            "Stage 7 signal has conflicting holding boundaries",
        )
    date_rows: list[dict[str, object]] = []
    for signal in sorted(calendar):
        _entry, exit_date = calendar[signal]
        current = signal
        while current <= exit_date:
            date_rows.append(
                {
                    "signal_date": signal,
                    "calendar_date": current,
                    "route_kind": "NORMAL",
                    "exit_boundary": None,
                }
            )
            current += dt.timedelta(days=1)
        probe_end = min(
            _CUTOFF,
            exit_date + dt.timedelta(days=_EXIT_STRADDLE_PROBE_DAYS),
        )
        current = exit_date + dt.timedelta(days=1)
        while current <= probe_end:
            date_rows.append(
                {
                    "signal_date": signal,
                    "calendar_date": current,
                    "route_kind": "EXIT_STRADDLE_PROBE",
                    "exit_boundary": exit_date,
                }
            )
            current += dt.timedelta(days=1)
    routing_dates = pa.Table.from_pylist(
        date_rows,
        schema=pa.schema(
            [
                pa.field("signal_date", pa.date32(), nullable=False),
                pa.field("calendar_date", pa.date32(), nullable=False),
                pa.field("route_kind", pa.string(), nullable=False),
                pa.field("exit_boundary", pa.date32(), nullable=True),
            ]
        ),
    )
    long_keys = intervals.select(("signal_date", "permno"))
    return long_keys, routing_dates


def _validate_source_batch(
    batch: pa.RecordBatch,
    expected_schema: pa.Schema,
) -> pa.RecordBatch:
    _require(
        isinstance(batch, pa.RecordBatch)
        and batch.schema.equals(expected_schema, check_metadata=True),
        "Stage 7 return source batch schema changed",
    )
    for required in ("permno", "calendar_date"):
        _require(
            batch.schema.get_field_index(required) >= 0,
            f"Stage 7 return source omits {required}",
        )
    dates = batch.column(batch.schema.get_field_index("calendar_date"))
    violation = pc.greater(dates, pa.scalar(_CUTOFF, type=pa.date32()))
    _require(
        not pc.any(pc.fill_null(violation, False)).as_py(),
        "Post-cutoff return row escaped Stage 7 routing",
    )
    for name in ("return_span_start_date", "return_span_end_date"):
        index = batch.schema.get_field_index(name)
        if index < 0:
            continue
        span = batch.column(index)
        violation = pc.greater(
            span, pa.scalar(_CUTOFF, type=pa.date32())
        )
        _require(
            not pc.any(pc.fill_null(violation, False)).as_py(),
            "Post-cutoff return span escaped Stage 7 routing",
        )
    return batch


def _validated_batches(
    first: pa.RecordBatch,
    remaining: Iterable[pa.RecordBatch],
) -> Iterator[pa.RecordBatch]:
    schema = first.schema
    yield _validate_source_batch(first, schema)
    for batch in remaining:
        yield _validate_source_batch(batch, schema)


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _output_schema(source_schema: pa.Schema) -> pa.Schema:
    _require(
        "signal_date" not in source_schema.names,
        "Return source already contains signal_date",
    )
    return pa.schema(
        [
            pa.field("signal_date", pa.date32(), nullable=False),
            *list(source_schema),
        ]
    )


def _coerce_output_batch(
    batch: pa.RecordBatch,
    schema: pa.Schema,
) -> pa.RecordBatch:
    _require(
        batch.schema.names == schema.names,
        "DuckDB routed-return column order differs",
    )
    arrays: list[pa.Array] = []
    for field in schema:
        array = batch.column(batch.schema.get_field_index(field.name))
        if not array.type.equals(field.type):
            try:
                array = pc.cast(array, field.type, safe=True)
            except pa.ArrowException as exc:
                raise LabelReturnRoutingError(
                    f"DuckDB routed-return type differs: {field.name}"
                ) from exc
        arrays.append(array)
    try:
        return pa.RecordBatch.from_arrays(arrays, schema=schema)
    except (pa.ArrowException, ValueError, TypeError) as exc:
        raise LabelReturnRoutingError(
            "DuckDB routed-return batch violates output schema"
        ) from exc


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
        _require(not self.closed, "Stage 7 hashing sink is closed")
        view = memoryview(payload)
        written = self._stream.write(view)
        _require(written == len(view), "Short Stage 7 routed-return write")
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


@dataclass(frozen=True)
class _TableReceipt:
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


def _write_query_result(
    *,
    stage: SecureStagingDirectory,
    reader: pa.RecordBatchReader,
    schema: pa.Schema,
    row_group_size: int,
    compression: str,
) -> tuple[_TableReceipt, Mapping[dt.date, int]]:
    rows = 0
    expected_row_groups = 0
    counts: Counter[dt.date] = Counter()
    previous_key: tuple[dt.date, int, dt.date] | None = None
    with stage.open_exclusive_binary(_PARQUET_NAME) as stream:
        sink = _HashingSink(stream)
        writer = pq.ParquetWriter(
            sink,
            schema,
            compression=compression,
            use_dictionary=True,
            version="2.6",
            write_statistics=True,
        )
        try:
            for raw_batch in reader:
                batch = _coerce_output_batch(raw_batch, schema)
                if batch.num_rows == 0:
                    continue
                signals = batch["signal_date"].to_pylist()
                permnos = batch["permno"].to_pylist()
                dates = batch["calendar_date"].to_pylist()
                for signal, permno, calendar_date in zip(
                    signals, permnos, dates, strict=True
                ):
                    _require(
                        isinstance(signal, dt.date)
                        and isinstance(permno, int)
                        and isinstance(calendar_date, dt.date)
                        and calendar_date <= _CUTOFF,
                        "Invalid routed-return key",
                    )
                    key = (signal, permno, calendar_date)
                    _require(
                        previous_key is None or previous_key < key,
                        "Routed returns are not uniquely sorted",
                    )
                    previous_key = key
                    counts[signal] += 1
                writer.write_batch(
                    batch, row_group_size=row_group_size
                )
                rows += batch.num_rows
                expected_row_groups += math.ceil(
                    batch.num_rows / row_group_size
                )
        finally:
            writer.close()
        sink.flush()
    return (
        _TableReceipt(
            path=_PARQUET_NAME,
            sha256=sink.sha256,
            size_bytes=sink.size_bytes,
            rows=rows,
            row_groups=expected_row_groups,
            schema_sha256=_schema_sha256(schema),
        ),
        MappingProxyType(dict(sorted(counts.items()))),
    )


def _success_payload(
    *,
    build_id: str,
    source_fingerprint: str,
    intervals_sha256: str,
    source_schema: pa.Schema,
    receipt: _TableReceipt,
    rows_by_signal: Mapping[dt.date, int],
) -> bytes:
    value = {
        "schema_version": 1,
        "status": _STATUS,
        "build_id": build_id,
        "source_fingerprint": source_fingerprint,
        "intervals_sha256": intervals_sha256,
        "cutoff_date": _CUTOFF.isoformat(),
        "projected_return_columns": source_schema.names,
        "routed_row_count": receipt.rows,
        "rows_by_signal": [
            {"signal_date": signal.isoformat(), "rows": rows}
            for signal, rows in sorted(rows_by_signal.items())
        ],
        "table_receipt": receipt.as_dict(),
    }
    _require(set(value) == _SUCCESS_KEYS, "Router success keys differ")
    return canonical_json_bytes(value)


def _parse_success(
    payload: bytes,
    *,
    build_id: str,
    source_fingerprint: str,
    intervals_sha256: str,
) -> tuple[dict[str, Any], _TableReceipt, Mapping[dt.date, int]]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabelReturnRoutingError(
            "Stage 7 return-router receipt is invalid JSON"
        ) from exc
    _require(
        isinstance(value, dict) and set(value) == _SUCCESS_KEYS,
        "Stage 7 return-router receipt keys differ",
    )
    _require(
        value["schema_version"] == 1
        and value["status"] == _STATUS
        and value["build_id"] == build_id
        and value["source_fingerprint"] == source_fingerprint
        and value["intervals_sha256"] == intervals_sha256
        and value["cutoff_date"] == _CUTOFF.isoformat(),
        "Stage 7 return-router receipt binding differs",
    )
    columns = value["projected_return_columns"]
    _require(
        isinstance(columns, list)
        and len(columns) == len(set(columns))
        and "permno" in columns
        and "calendar_date" in columns
        and "signal_date" not in columns,
        "Stage 7 return-router projected columns differ",
    )
    raw_receipt = value["table_receipt"]
    _require(
        isinstance(raw_receipt, dict)
        and set(raw_receipt) == _TABLE_RECEIPT_KEYS
        and raw_receipt["path"] == _PARQUET_NAME,
        "Stage 7 return-router table receipt differs",
    )
    for name in ("size_bytes", "rows", "row_groups"):
        _require(
            isinstance(raw_receipt[name], int)
            and not isinstance(raw_receipt[name], bool)
            and raw_receipt[name] >= 0,
            f"Invalid routed-return receipt {name}",
        )
    _require(
        _SHA256.fullmatch(str(raw_receipt["sha256"])) is not None
        and _SHA256.fullmatch(str(raw_receipt["schema_sha256"]))
        is not None,
        "Invalid routed-return receipt hash",
    )
    receipt = _TableReceipt(**raw_receipt)
    _require(
        value["routed_row_count"] == receipt.rows,
        "Stage 7 routed-return row total differs",
    )
    raw_counts = value["rows_by_signal"]
    _require(
        isinstance(raw_counts, list),
        "Stage 7 routed-return signal counts differ",
    )
    counts: dict[dt.date, int] = {}
    for row in raw_counts:
        _require(
            isinstance(row, dict)
            and set(row) == {"signal_date", "rows"}
            and isinstance(row["rows"], int)
            and not isinstance(row["rows"], bool)
            and row["rows"] > 0,
            "Invalid routed-return signal count",
        )
        try:
            signal = dt.date.fromisoformat(row["signal_date"])
        except (TypeError, ValueError) as exc:
            raise LabelReturnRoutingError(
                "Invalid routed-return signal date"
            ) from exc
        _require(signal not in counts, "Duplicate routed-return signal count")
        counts[signal] = row["rows"]
    _require(
        list(counts) == sorted(counts)
        and sum(counts.values()) == receipt.rows,
        "Routed-return signal counts do not reconcile",
    )
    return value, receipt, MappingProxyType(counts)


def _parquet_validator(
    receipt: _TableReceipt,
    *,
    expected_schema: pa.Schema | None = None,
):
    def validate(name: str, descriptor: int, record: FileRecord) -> None:
        if name == _SUCCESS_NAME:
            return
        _require(name == _PARQUET_NAME, "Unexpected router bundle member")
        _require(
            record.sha256 == receipt.sha256
            and record.size_bytes == receipt.size_bytes,
            "Routed-return file receipt differs",
        )
        duplicate = os.dup(descriptor)
        try:
            with os.fdopen(duplicate, "rb") as handle:
                parquet = pq.ParquetFile(handle)
                observed_schema = parquet.schema_arrow
                observed_rows = parquet.metadata.num_rows
                observed_groups = parquet.metadata.num_row_groups
        except Exception as exc:
            raise LabelReturnRoutingError(
                "Cannot inspect routed-return Parquet"
            ) from exc
        _require(
            observed_rows == receipt.rows
            and observed_groups == receipt.row_groups
            and _schema_sha256(observed_schema)
            == receipt.schema_sha256,
            "Routed-return Parquet metadata differs",
        )
        if expected_schema is not None:
            _require(
                observed_schema.equals(
                    expected_schema, check_metadata=True
                ),
                "Routed-return Parquet schema differs",
            )

    return validate


@dataclass(frozen=True)
class LabelReturnRouter:
    path: Path
    build_id: str
    source_fingerprint: str
    intervals_sha256: str
    routed_row_count: int
    rows_by_signal: Mapping[dt.date, int]
    reused_existing: bool
    success_sha256: str
    parquet_sha256: str
    success_stat_identity: tuple[int, ...]
    parquet_stat_identity: tuple[int, ...]

    def gate(self) -> None:
        """Recheck the authenticated immutable router identity in place."""

        self._gate()

    def _gate(self) -> tuple[Path, Path]:
        success_path = self.path / _SUCCESS_NAME
        parquet_path = self.path / _PARQUET_NAME
        _require(
            self.path.is_dir()
            and not self.path.is_symlink()
            and success_path.is_file()
            and not success_path.is_symlink()
            and parquet_path.is_file()
            and not parquet_path.is_symlink(),
            "Routed-return bundle is unavailable",
        )
        success_current = _stat_identity(
            os.stat(success_path, follow_symlinks=False)
        )
        parquet_current = _stat_identity(
            os.stat(parquet_path, follow_symlinks=False)
        )
        _require(
            _sha256_file(success_path) == self.success_sha256,
            "Routed-return authenticated success bytes changed",
        )
        if success_current != self.success_stat_identity:
            _require(
                _same_stat_except_ctime(
                    success_current,
                    self.success_stat_identity,
                ),
                "Routed-return authenticated identity changed",
            )
            success_after = _stat_identity(
                os.stat(success_path, follow_symlinks=False)
            )
            _require(
                _same_stat_except_ctime(success_current, success_after)
                and _sha256_file(success_path) == self.success_sha256,
                "Routed-return success changed during ctime refresh",
            )
            object.__setattr__(
                self,
                "success_stat_identity",
                success_after,
            )
        if parquet_current != self.parquet_stat_identity:
            _require(
                _same_stat_except_ctime(
                    parquet_current,
                    self.parquet_stat_identity,
                )
                and _sha256_file(parquet_path) == self.parquet_sha256,
                "Routed-return authenticated identity changed",
            )
            parquet_after = _stat_identity(
                os.stat(parquet_path, follow_symlinks=False)
            )
            _require(
                _same_stat_except_ctime(parquet_current, parquet_after)
                and _sha256_file(parquet_path) == self.parquet_sha256,
                "Routed-return Parquet changed during ctime refresh",
            )
            object.__setattr__(
                self,
                "parquet_stat_identity",
                parquet_after,
            )
        return success_path, parquet_path

    @property
    def schema(self) -> pa.Schema:
        _success, path = self._gate()
        return pq.ParquetFile(path).schema_arrow

    def iter_signal_batches(
        self,
        signal_date: dt.date,
        batch_rows: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        """Yield only one signal's rows from the immutable sorted router."""

        _require(
            isinstance(signal_date, dt.date)
            and not isinstance(signal_date, dt.datetime),
            "Invalid routed-return signal date",
        )
        _require(
            isinstance(batch_rows, int)
            and not isinstance(batch_rows, bool)
            and batch_rows > 0,
            "Routed-return batch_rows must be positive",
        )
        success_path, parquet_path = self._gate()
        before = os.stat(parquet_path, follow_symlinks=False)
        payload = success_path.read_bytes()
        _value, receipt, counts = _parse_success(
            payload,
            build_id=self.build_id,
            source_fingerprint=self.source_fingerprint,
            intervals_sha256=self.intervals_sha256,
        )
        _require(
            receipt.rows == self.routed_row_count
            and dict(counts) == dict(self.rows_by_signal)
            and before.st_size == receipt.size_bytes,
            "Routed-return handle differs from receipt",
        )
        _require(
            receipt.sha256 == self.parquet_sha256,
            "Routed-return receipt hash differs from authenticated handle",
        )
        parquet = pq.ParquetFile(parquet_path)
        _require(
            parquet.metadata.num_rows == receipt.rows
            and parquet.metadata.num_row_groups == receipt.row_groups
            and _schema_sha256(parquet.schema_arrow)
            == receipt.schema_sha256,
            "Routed-return metadata changed",
        )
        expected = int(self.rows_by_signal.get(signal_date, 0))
        emitted = 0
        previous_key: tuple[int, dt.date] | None = None
        scanner = ds.dataset(parquet_path, format="parquet").scanner(
            filter=ds.field("signal_date")
            == pa.scalar(signal_date, type=pa.date32()),
            batch_size=batch_rows,
            use_threads=False,
        )
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            signals = batch["signal_date"].to_pylist()
            permnos = batch["permno"].to_pylist()
            dates = batch["calendar_date"].to_pylist()
            for observed_signal, permno, calendar_date in zip(
                signals, permnos, dates, strict=True
            ):
                _require(
                    observed_signal == signal_date,
                    "Signal filter leaked another routed month",
                )
                key = (int(permno), calendar_date)
                _require(
                    previous_key is None or previous_key < key,
                    "Signal-routed returns are not uniquely sorted",
                )
                previous_key = key
            emitted += batch.num_rows
            yield batch
        after = os.stat(parquet_path, follow_symlinks=False)
        self._gate()
        _require(
            emitted == expected
            and _stat_identity(before) == _stat_identity(after),
            "Routed-return signal delivery changed",
        )


def _authenticate_router(
    *,
    trusted_root: Path,
    destination: Path,
    build_id: str,
    source_fingerprint: str,
    intervals_sha256: str,
    reused_existing: bool,
) -> LabelReturnRouter:
    success_path = destination / _SUCCESS_NAME
    authenticated_success_hash = ""
    authenticated_success_stat: tuple[int, ...] | None = None
    authenticated_parquet_stat: tuple[int, ...] | None = None
    try:
        with AuthenticatedPointerGuard.open(
            trusted_root=trusted_root,
            path=success_path,
            label="Stage 7 return-router receipt",
        ) as pointer:
            _value, receipt, counts = _parse_success(
                pointer.payload,
                build_id=build_id,
                source_fingerprint=source_fingerprint,
                intervals_sha256=intervals_sha256,
            )
            expected_hashes = {
                _SUCCESS_NAME: pointer.payload_sha256,
                _PARQUET_NAME: receipt.sha256,
            }
            with AuthenticatedBundleGuard.open(
                trusted_root=trusted_root,
                path=destination,
                expected_files=_EXPECTED_FILES,
                expected_hashes=expected_hashes,
                file_validator=_parquet_validator(receipt),
                label="Stage 7 return-router bundle",
                _permit_authenticated_directory_ctime_refresh=True,
            ) as bundle:
                pointer.gate()
                bundle.gate()
                authenticated_success_hash = pointer.payload_sha256
    except (SecurePublicationError, LabelReturnRoutingError):
        raise
    except Exception as exc:
        raise LabelReturnRoutingError(
            "Cannot authenticate Stage 7 return router"
        ) from exc
    parquet_path = destination / _PARQUET_NAME
    success_before = _stat_identity(
        os.stat(success_path, follow_symlinks=False)
    )
    parquet_before = _stat_identity(
        os.stat(parquet_path, follow_symlinks=False)
    )
    _require(
        bool(authenticated_success_hash)
        and _sha256_file(success_path) == authenticated_success_hash
        and _sha256_file(parquet_path) == receipt.sha256,
        "Stage 7 return-router bytes changed after authentication",
    )
    authenticated_success_stat = _stat_identity(
        os.stat(success_path, follow_symlinks=False)
    )
    authenticated_parquet_stat = _stat_identity(
        os.stat(parquet_path, follow_symlinks=False)
    )
    _require(
        authenticated_success_stat == success_before
        and authenticated_parquet_stat == parquet_before,
        "Stage 7 return-router identity changed after authentication",
    )
    return LabelReturnRouter(
        path=destination,
        build_id=build_id,
        source_fingerprint=source_fingerprint,
        intervals_sha256=intervals_sha256,
        routed_row_count=receipt.rows,
        rows_by_signal=counts,
        reused_existing=reused_existing,
        success_sha256=authenticated_success_hash,
        parquet_sha256=receipt.sha256,
        success_stat_identity=authenticated_success_stat,
        parquet_stat_identity=authenticated_parquet_stat,
    )


def spool_label_return_router(
    inputs: Any,
    intervals: pa.Table,
    destination: Path,
    build_id: str,
    source_fingerprint: str,
    row_group_size: int = 131_072,
    compression: str = "zstd",
) -> LabelReturnRouter:
    """Range-route the bounded return source once and publish an immutable spool."""

    build = _safe_text(build_id, "build ID")
    fingerprint = _safe_fingerprint(source_fingerprint)
    _require(
        isinstance(row_group_size, int)
        and not isinstance(row_group_size, bool)
        and row_group_size > 0,
        "Stage 7 router row_group_size must be positive",
    )
    _require(
        isinstance(compression, str)
        and compression in {"zstd", "snappy", "gzip", "brotli", "lz4", "none"},
        "Unsafe Stage 7 router compression",
    )
    root = Path(os.path.abspath(Path(inputs.database_root).expanduser()))
    target = _safe_destination(root, destination)
    selected_intervals = _validate_intervals(intervals)
    intervals_hash = _intervals_sha256(selected_intervals)
    long_keys, routing_dates = _routing_relations(selected_intervals)
    source_snapshot = _source_snapshot(inputs)
    if os.path.lexists(target):
        _require(
            _source_snapshot(inputs) == source_snapshot,
            "Stage 7 return source changed before router reuse",
        )
        return _authenticate_router(
            trusted_root=root,
            destination=target,
            build_id=build,
            source_fingerprint=fingerprint,
            intervals_sha256=intervals_hash,
            reused_existing=True,
        )

    source_iterator = iter(
        inputs.iter_development_return_batches(batch_rows=65_536)
    )
    try:
        first = next(source_iterator)
    except StopIteration as exc:
        raise LabelReturnRoutingError(
            "Stage 7 bounded return source is empty"
        ) from exc
    _require(
        isinstance(first, pa.RecordBatch),
        "Stage 7 return source did not yield RecordBatch values",
    )
    source_schema = first.schema
    validated = _validated_batches(first, source_iterator)
    reader = pa.RecordBatchReader.from_batches(
        source_schema, validated
    )
    output_schema = _output_schema(source_schema)

    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET threads = 1")
        connection.execute("SET preserve_insertion_order = false")
        connection.register("_stage7_long_keys", long_keys)
        connection.register("_stage7_routing_dates", routing_dates)
        connection.register("_stage7_returns", reader)
        projected = ", ".join(
            f"r.{_quoted_identifier(name)}"
            for name in source_schema.names
        )
        query = f"""
            SELECT
                d.signal_date AS signal_date,
                {projected}
            FROM _stage7_returns AS r
            INNER JOIN _stage7_routing_dates AS d
                ON r.calendar_date = d.calendar_date
            INNER JOIN _stage7_long_keys AS k
                ON d.signal_date = k.signal_date
               AND r.permno = k.permno
            WHERE
                d.route_kind = 'NORMAL'
                OR (
                    d.route_kind = 'EXIT_STRADDLE_PROBE'
                    AND r.return_span_start_date < d.exit_boundary
                    AND r.return_span_end_date > d.exit_boundary
                )
            ORDER BY
                d.signal_date,
                r.permno,
                r.calendar_date
        """
        with SecureStagingDirectory.create(
            trusted_root=root,
            destination=target,
            expected_files=_EXPECTED_FILES,
            label="Stage 7 return router",
        ) as stage:
            result_reader = connection.execute(query).fetch_record_batch(
                rows_per_batch=min(row_group_size, 65_536)
            )
            receipt, rows_by_signal = _write_query_result(
                stage=stage,
                reader=result_reader,
                schema=output_schema,
                row_group_size=row_group_size,
                compression=compression,
            )
            success = _success_payload(
                build_id=build,
                source_fingerprint=fingerprint,
                intervals_sha256=intervals_hash,
                source_schema=source_schema,
                receipt=receipt,
                rows_by_signal=rows_by_signal,
            )
            with stage.open_exclusive_binary(_SUCCESS_NAME) as stream:
                stream.write(success)
            written_hashes = {
                _SUCCESS_NAME: hashlib.sha256(success).hexdigest(),
                _PARQUET_NAME: receipt.sha256,
            }
            validator = _parquet_validator(
                receipt, expected_schema=output_schema
            )
            snapshot = stage.snapshot(
                expected_hashes=written_hashes,
                file_validator=validator,
            )

            def source_gate() -> None:
                _require(
                    _source_snapshot(inputs) == source_snapshot,
                    "Stage 7 return source changed during routing",
                )

            source_gate()
            published = stage.publish(
                expected_hashes=snapshot.hashes,
                file_validator=validator,
                pre_publish_gate=source_gate,
                post_publish_gate=source_gate,
            )
    except duckdb.Error as exc:
        message = str(exc)
        for source_message in (
            "Post-cutoff return row escaped Stage 7 routing",
            "Post-cutoff return span escaped Stage 7 routing",
            "Stage 7 return source batch schema changed",
        ):
            if source_message in message:
                raise LabelReturnRoutingError(source_message) from exc
        raise LabelReturnRoutingError(
            "DuckDB could not build the Stage 7 return router"
        ) from exc
    except pa.ArrowException as exc:
        raise LabelReturnRoutingError(
            "DuckDB could not build the Stage 7 return router"
        ) from exc
    finally:
        connection.close()

    return _authenticate_router(
        trusted_root=root,
        destination=target,
        build_id=build,
        source_fingerprint=fingerprint,
        intervals_sha256=intervals_hash,
        reused_existing=published.reused_existing,
    )
