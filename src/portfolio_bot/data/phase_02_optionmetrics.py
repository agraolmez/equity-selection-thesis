"""Development-only OptionMetrics normalization and date-valid CRSP linkage.

This module performs no feature engineering.  It streams caller-supplied raw
files into a caller-supplied staging directory and never publishes authority
pointers.  Large CSV batches are gated on the source date while every field is
still text; payload columns are converted only for retained development rows.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


SCOPE_START = dt.date(2000, 1, 1)
SCOPE_END = dt.date(2014, 10, 31)
ROW_GROUP_SIZE = 250_000
DEFAULT_BLOCK_SIZE = 64 << 20

NAME_FILE = "optionmetrics_security_name_history.csv"
VOLUME_FILE = "optionmetrics_option_volume.csv"
VOLATILITY_FILE = "optionmetrics_historical_volatility.csv"
PRICE_FILE = "optionmetrics_security_prices.csv"

OUTPUT_FILES = {
    "optionmetrics_name_interval": "optionmetrics_name_interval.parquet",
    "optionmetrics_crsp_link_interval": "optionmetrics_crsp_link_interval.parquet",
    "optionmetrics_option_volume_observation": "optionmetrics_option_volume_observation.parquet",
    "optionmetrics_historical_volatility_observation": "optionmetrics_historical_volatility_observation.parquet",
}
PRICE_QA_FILE = "optionmetrics_security_prices_qa.json"


class OptionMetricsBuildError(RuntimeError):
    """Raised when an OptionMetrics staging contract is violated."""


@dataclass(frozen=True)
class OptionMetricsSettings:
    raw_root: Path
    crsp_identity_path: Path
    staging_dir: Path
    scope_start: dt.date = SCOPE_START
    scope_end: dt.date = SCOPE_END
    block_size_bytes: int = DEFAULT_BLOCK_SIZE
    row_group_size: int = ROW_GROUP_SIZE


NAME_SCHEMA = pa.schema(
    [
        pa.field("secid", pa.int64(), nullable=False),
        pa.field("effective_start_date", pa.date32(), nullable=False),
        pa.field("effective_end_date", pa.date32(), nullable=False),
        pa.field("normalized_cusip8", pa.string()),
        pa.field("ticker_audit_only", pa.string()),
        pa.field("class", pa.string()),
        pa.field("issue", pa.string()),
        pa.field("issuer_name_audit_only", pa.string()),
    ]
)

LINK_SCHEMA = pa.schema(
    [
        pa.field("permno", pa.int64(), nullable=False),
        pa.field("secid", pa.int64(), nullable=False),
        pa.field("link_start_date", pa.date32(), nullable=False),
        pa.field("link_end_date", pa.date32(), nullable=False),
        pa.field("normalized_cusip8", pa.string(), nullable=False),
        pa.field("link_status", pa.string(), nullable=False),
        pa.field("candidate_count", pa.int32(), nullable=False),
    ]
)

VOLUME_SCHEMA = pa.schema(
    [
        pa.field("secid", pa.int64(), nullable=False),
        pa.field("observation_date", pa.date32(), nullable=False),
        pa.field("public_availability_date", pa.date32(), nullable=False),
        pa.field("availability_basis", pa.string(), nullable=False),
        pa.field("row_role", pa.string(), nullable=False),
        pa.field("authoritative_flow_component", pa.bool_(), nullable=False),
        pa.field("volume", pa.int64()),
        pa.field("open_interest", pa.int64()),
        pa.field("genuine_zero_volume", pa.bool_()),
        pa.field("genuine_zero_open_interest", pa.bool_()),
        pa.field("normalized_cusip8_audit_only", pa.string()),
    ]
)

VOLATILITY_SCHEMA = pa.schema(
    [
        pa.field("secid", pa.int64(), nullable=False),
        pa.field("observation_date", pa.date32(), nullable=False),
        pa.field("public_availability_date", pa.date32(), nullable=False),
        pa.field("availability_basis", pa.string(), nullable=False),
        pa.field("days", pa.int16(), nullable=False),
        pa.field("volatility", pa.decimal128(38, 12)),
        pa.field("supported_horizon", pa.bool_(), nullable=False),
        pa.field("normalized_cusip8_audit_only", pa.string()),
    ]
)


def _header(path: Path) -> list[str]:
    with path.open("rb") as handle:
        line = handle.readline(1 << 20)
    if not line or (len(line) == 1 << 20 and not line.endswith((b"\n", b"\r"))):
        raise OptionMetricsBuildError(f"Missing or overlong CSV header: {path.name}")
    try:
        return next(csv.reader([line.decode("utf-8-sig").rstrip("\r\n")]))
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise OptionMetricsBuildError(f"Invalid CSV header: {path.name}") from exc


def _open_text_batches(
    path: Path,
    columns: Sequence[str],
    *,
    block_size_bytes: int,
    use_threads: bool = True,
) -> pacsv.CSVStreamingReader:
    header = _header(path)
    missing = sorted(set(columns) - set(header))
    if missing:
        raise OptionMetricsBuildError(f"{path.name} lacks columns {missing}")
    try:
        return pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(
                block_size=max(4096, block_size_bytes),
                use_threads=use_threads,
                encoding="utf8",
            ),
            parse_options=pacsv.ParseOptions(
                delimiter=",",
                quote_char='"',
                double_quote=True,
                escape_char=False,
                newlines_in_values=True,
                ignore_empty_lines=False,
            ),
            convert_options=pacsv.ConvertOptions(
                include_columns=list(columns),
                column_types={name: pa.string() for name in columns},
                strings_can_be_null=False,
                quoted_strings_can_be_null=False,
                null_values=[],
                check_utf8=False,
            ),
        )
    except (OSError, pa.ArrowException) as exc:
        raise OptionMetricsBuildError(f"Cannot open {path.name}") from exc


def _column(batch: pa.RecordBatch | pa.Table, name: str) -> pa.Array | pa.ChunkedArray:
    return batch.column(batch.schema.get_field_index(name))


def _strict_date_text(values: pa.Array, *, label: str) -> pa.Array:
    lexical = pc.match_substring_regex(values, r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    parsed = pc.strptime(values, format="%Y-%m-%d", unit="s", error_is_null=True)
    round_trip = pc.strftime(parsed, format="%Y-%m-%d")
    valid = pc.and_(
        lexical,
        pc.and_(pc.is_valid(parsed), pc.fill_null(pc.equal(round_trip, values), False)),
    )
    if not bool(pc.all(valid).as_py()):
        raise OptionMetricsBuildError(f"{label} contains a blank or invalid date")
    return pc.cast(parsed, pa.date32())


def _date_first_filter(
    batch: pa.RecordBatch,
    date_column: str,
    *,
    minimum: dt.date | None,
    maximum: dt.date,
) -> tuple[pa.RecordBatch, pa.Array, int]:
    """Validate only the date, filter scope, then permit payload conversion."""

    raw_date = _column(batch, date_column)
    assert isinstance(raw_date, pa.Array)
    dates = _strict_date_text(raw_date, label=date_column)
    mask = pc.less_equal(dates, pa.scalar(maximum, type=pa.date32()))
    if minimum is not None:
        mask = pc.and_(
            mask, pc.greater_equal(dates, pa.scalar(minimum, type=pa.date32()))
        )
    retained = batch.filter(mask)
    retained_dates = pc.filter(dates, mask)
    return retained, retained_dates, batch.num_rows - retained.num_rows


def _required_positive_int(values: pa.Array, *, label: str, out_type: pa.DataType = pa.int64()) -> pa.Array:
    valid = pc.match_substring_regex(values, r"^[0-9]+$")
    if not bool(pc.all(valid).as_py()):
        raise OptionMetricsBuildError(f"{label} is not an unsigned integer")
    try:
        output = pc.cast(values, out_type, safe=True)
    except pa.ArrowException as exc:
        raise OptionMetricsBuildError(f"{label} exceeds its exact integer type") from exc
    if bool(pc.any(pc.less_equal(output, pa.scalar(0, type=out_type))).as_py()):
        raise OptionMetricsBuildError(f"{label} is not strictly positive")
    return output


def _nullable_nonnegative_int(values: pa.Array, *, label: str) -> pa.Array:
    blank = pc.equal(values, "")
    nonblank_valid = pc.match_substring_regex(values, r"^[0-9]+$")
    valid = pc.or_(blank, nonblank_valid)
    if not bool(pc.all(valid).as_py()):
        raise OptionMetricsBuildError(f"{label} is not a nullable nonnegative integer")
    nullable = pc.if_else(blank, pa.scalar(None, type=pa.string()), values)
    try:
        return pc.cast(nullable, pa.int64(), safe=True)
    except pa.ArrowException as exc:
        raise OptionMetricsBuildError(f"{label} exceeds int64") from exc


def _nullable_exact_decimal(values: pa.Array, *, label: str) -> pa.Array:
    blank = pc.equal(values, "")
    grammar = pc.match_substring_regex(
        values, r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?$"
    )
    if not bool(pc.all(pc.or_(blank, grammar)).as_py()):
        raise OptionMetricsBuildError(f"{label} is not an exact decimal")
    nullable = pc.if_else(blank, pa.scalar(None, type=pa.string()), values)
    try:
        return pc.cast(nullable, pa.decimal128(38, 12), safe=True)
    except pa.ArrowException as exc:
        raise OptionMetricsBuildError(
            f"{label} exceeds decimal128(38,12) without lossless representation"
        ) from exc


def normalize_cusip8(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).replace(" ", "").replace("-", "").upper()
    if len(cleaned) < 8 or not cleaned.isascii() or not cleaned.isalnum():
        return None
    return cleaned[:8]


def _normalized_cusip_array(values: pa.Array) -> pa.Array:
    cleaned = pc.utf8_upper(pc.replace_substring_regex(values, r"[ -]", ""))
    sliced = pc.utf8_slice_codeunits(cleaned, 0, 8)
    valid = pc.match_substring_regex(cleaned, r"^[A-Z0-9]{8,}$")
    return pc.if_else(valid, sliced, pa.scalar(None, type=pa.string()))


def _string_or_null(values: pa.Array) -> pa.Array:
    return pc.if_else(
        pc.equal(values, ""), pa.scalar(None, type=pa.string()), values
    )


def _output_path(settings: OptionMetricsSettings, table_name: str) -> Path:
    return settings.staging_dir / OUTPUT_FILES[table_name]


def _write_table_atomic(
    table: pa.Table,
    path: Path,
    *,
    row_group_size: int,
) -> None:
    if path.exists():
        raise OptionMetricsBuildError(f"Refusing to overwrite {path.name}")
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    if temporary.exists():
        raise OptionMetricsBuildError(f"Prior partial output exists: {temporary.name}")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            row_group_size=row_group_size,
        )
        os.replace(temporary, path)
    except Exception:
        raise


def build_name_intervals(settings: OptionMetricsSettings) -> pa.Table:
    path = settings.raw_root / NAME_FILE
    columns = ["secid", "effect_date", "ticker", "cusip", "class", "issue", "issuer"]
    records: list[tuple[int, dt.date, str | None, str | None, str | None, str | None, str | None]] = []
    for batch in _open_text_batches(
        path, columns, block_size_bytes=settings.block_size_bytes
    ):
        retained, dates, _ = _date_first_filter(
            batch, "effect_date", minimum=None, maximum=settings.scope_end
        )
        if retained.num_rows == 0:
            continue
        secids = _required_positive_int(
            _column(retained, "secid"), label="name history secid"
        ).to_pylist()
        payload = {
            name: _column(retained, name).to_pylist()
            for name in ("ticker", "cusip", "class", "issue", "issuer")
        }
        for index, (secid, day) in enumerate(zip(secids, dates.to_pylist(), strict=True)):
            records.append(
                (
                    int(secid),
                    day,
                    normalize_cusip8(payload["cusip"][index]),
                    payload["ticker"][index] or None,
                    payload["class"][index] or None,
                    payload["issue"][index] or None,
                    payload["issuer"][index] or None,
                )
            )
    records.sort(key=lambda row: (row[0], row[1]))
    deduplicated: list[tuple[int, dt.date, str | None, str | None, str | None, str | None, str | None]] = []
    for record in records:
        if deduplicated and record[:2] == deduplicated[-1][:2]:
            if record != deduplicated[-1]:
                raise OptionMetricsBuildError("Conflicting secid/effect_date name rows")
            continue
        deduplicated.append(record)

    output: dict[str, list[Any]] = {name: [] for name in NAME_SCHEMA.names}
    for index, record in enumerate(deduplicated):
        secid, start, cusip8, ticker, security_class, issue, issuer = record
        next_start = (
            deduplicated[index + 1][1]
            if index + 1 < len(deduplicated)
            and deduplicated[index + 1][0] == secid
            else None
        )
        end = settings.scope_end if next_start is None else min(
            settings.scope_end, next_start - dt.timedelta(days=1)
        )
        if end < settings.scope_start or start > settings.scope_end:
            continue
        output["secid"].append(secid)
        output["effective_start_date"].append(start)
        output["effective_end_date"].append(end)
        output["normalized_cusip8"].append(cusip8)
        output["ticker_audit_only"].append(ticker)
        output["class"].append(security_class)
        output["issue"].append(issue)
        output["issuer_name_audit_only"].append(issuer)
    table = pa.Table.from_pydict(output, schema=NAME_SCHEMA)
    _write_table_atomic(
        table,
        _output_path(settings, "optionmetrics_name_interval"),
        row_group_size=settings.row_group_size,
    )
    return table


def _interval_rows(
    table: pa.Table,
    *,
    id_column: str,
    start_column: str,
    end_column: str,
    cusip_column: str,
) -> dict[str, list[tuple[int, dt.date, dt.date]]]:
    output: dict[str, list[tuple[int, dt.date, dt.date]]] = defaultdict(list)
    for identifier, start, end, cusip in zip(
        table[id_column].to_pylist(),
        table[start_column].to_pylist(),
        table[end_column].to_pylist(),
        table[cusip_column].to_pylist(),
        strict=True,
    ):
        if cusip:
            output[str(cusip)].append((int(identifier), start, end))
    return output


def build_crsp_links(settings: OptionMetricsSettings, names: pa.Table) -> pa.Table:
    crsp = pq.read_table(
        settings.crsp_identity_path,
        columns=["permno", "start_date", "end_date", "cusip"],
    )
    crsp_values = {
        "permno": crsp["permno"].to_pylist(),
        "start": crsp["start_date"].to_pylist(),
        "end": crsp["end_date"].to_pylist(),
        "cusip": crsp["cusip"].to_pylist(),
    }
    crsp_by_cusip: dict[str, list[tuple[int, dt.date, dt.date]]] = defaultdict(list)
    for permno, start, end, cusip in zip(
        crsp_values["permno"], crsp_values["start"], crsp_values["end"], crsp_values["cusip"], strict=True
    ):
        normalized = normalize_cusip8(cusip)
        clipped_start = max(start, settings.scope_start)
        clipped_end = min(end, settings.scope_end)
        if normalized and clipped_start <= clipped_end:
            crsp_by_cusip[normalized].append((int(permno), clipped_start, clipped_end))
    option_by_cusip = _interval_rows(
        names,
        id_column="secid",
        start_column="effective_start_date",
        end_column="effective_end_date",
        cusip_column="normalized_cusip8",
    )

    rows: list[tuple[int, int, dt.date, dt.date, str, str, int]] = []
    for cusip in sorted(set(crsp_by_cusip) & set(option_by_cusip)):
        crsp_rows = crsp_by_cusip[cusip]
        option_rows = option_by_cusip[cusip]
        boundaries: set[dt.date] = set()
        for _, start, end in crsp_rows + option_rows:
            boundaries.add(start)
            if end < settings.scope_end:
                boundaries.add(end + dt.timedelta(days=1))
        ordered = sorted(boundaries)
        for index, start in enumerate(ordered):
            end = (
                ordered[index + 1] - dt.timedelta(days=1)
                if index + 1 < len(ordered)
                else settings.scope_end
            )
            permnos = sorted(
                {identifier for identifier, left, right in crsp_rows if left <= start <= right}
            )
            secids = sorted(
                {identifier for identifier, left, right in option_rows if left <= start <= right}
            )
            if not permnos or not secids:
                continue
            candidate_count = len(permnos) * len(secids)
            status = "MATCHED" if candidate_count == 1 else "AMBIGUOUS"
            for permno in permnos:
                for secid in secids:
                    rows.append(
                        (permno, secid, start, end, cusip, status, candidate_count)
                    )
    rows.sort(key=lambda row: (row[0], row[2], row[1], row[3]))
    table = pa.Table.from_pylist(
        [dict(zip(LINK_SCHEMA.names, row, strict=True)) for row in rows],
        schema=LINK_SCHEMA,
    )
    _write_table_atomic(
        table,
        _output_path(settings, "optionmetrics_crsp_link_interval"),
        row_group_size=settings.row_group_size,
    )
    return table


class _GroupedSortedParquetSink:
    """Sort each source key group while carrying the final split group."""

    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        *,
        base_keys: Sequence[str],
        sort_keys: Sequence[str],
        row_group_size: int,
    ) -> None:
        if path.exists():
            raise OptionMetricsBuildError(f"Refusing to overwrite {path.name}")
        self.path = path
        self.temporary = path.with_name(path.name + f".partial-{os.getpid()}")
        if self.temporary.exists():
            raise OptionMetricsBuildError(f"Prior partial output exists: {self.temporary.name}")
        self.schema = schema
        self.base_keys = tuple(base_keys)
        self.sort_keys = tuple(sort_keys)
        self.writer = pq.ParquetWriter(
            self.temporary,
            schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        self.pending: pa.Table | None = None
        self.previous_input_base: tuple[Any, ...] | None = None
        self.previous_output_key: tuple[Any, ...] | None = None
        self.rows = 0
        self.row_group_size = row_group_size

    @staticmethod
    def _row_key(table: pa.Table, columns: Sequence[str], index: int) -> tuple[Any, ...]:
        return tuple(table[name][index].as_py() for name in columns)

    def _validate_input_base_order(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if len(self.base_keys) != 1:
            raise OptionMetricsBuildError("Streaming group validation requires one source group key")
        group = table[self.base_keys[0]].combine_chunks().to_numpy(zero_copy_only=False)
        if np.any(group[1:] < group[:-1]):
            raise OptionMetricsBuildError("Retained source group key is not nondecreasing")
        first = self._row_key(table, self.base_keys, 0)
        last = self._row_key(table, self.base_keys, table.num_rows - 1)
        if self.previous_input_base is not None and first < self.previous_input_base:
            raise OptionMetricsBuildError("Retained source base key decreases across batches")
        self.previous_input_base = last

    def _write(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        first = self._row_key(table, self.sort_keys, 0)
        last = self._row_key(table, self.sort_keys, table.num_rows - 1)
        if self.previous_output_key is not None and first <= self.previous_output_key:
            raise OptionMetricsBuildError("Duplicate or unsorted normalized output key")
        if table.num_rows > 1:
            equal = np.ones(table.num_rows - 1, dtype=bool)
            for name in self.sort_keys:
                values = table[name].combine_chunks()
                adjacent_equal = pc.fill_null(
                    pc.equal(
                        values.slice(0, table.num_rows - 1),
                        values.slice(1, table.num_rows - 1),
                    ),
                    False,
                ).to_numpy(zero_copy_only=False)
                equal &= adjacent_equal
            if bool(equal.any()):
                raise OptionMetricsBuildError("Duplicate normalized output key")
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.previous_output_key = last
        self.rows += table.num_rows

    def consume(self, table: pa.Table) -> None:
        self._validate_input_base_order(table)
        combined = table if self.pending is None else pa.concat_tables([self.pending, table])
        order = pc.sort_indices(
            combined,
            sort_keys=[(name, "ascending") for name in self.sort_keys],
        )
        sorted_table = pc.take(combined, order)
        last_base = self._row_key(sorted_table, self.base_keys, sorted_table.num_rows - 1)
        split = sorted_table.num_rows
        while split > 0 and self._row_key(sorted_table, self.base_keys, split - 1) == last_base:
            split -= 1
        self._write(sorted_table.slice(0, split))
        self.pending = sorted_table.slice(split)

    def finish(self) -> int:
        try:
            if self.pending is not None:
                self._write(self.pending)
            self.writer.close()
            os.replace(self.temporary, self.path)
            return self.rows
        except Exception:
            try:
                self.writer.close()
            except Exception:
                pass
            raise


def _availability_array(dates: pa.Array) -> pa.Array:
    return dates


def stream_option_volume(settings: OptionMetricsSettings) -> dict[str, Any]:
    path = settings.raw_root / VOLUME_FILE
    columns = ["secid", "date", "cp_flag", "cusip", "volume", "open_interest"]
    sink = _GroupedSortedParquetSink(
        _output_path(settings, "optionmetrics_option_volume_observation"),
        VOLUME_SCHEMA,
        # The vendor file is ordered by secid, but its dates/roles are not
        # reliably ordered within a secid.  Carry one complete secid group and
        # sort it before writing, keeping memory bounded by one security.
        base_keys=("secid",),
        sort_keys=("secid", "observation_date", "row_role"),
        row_group_size=settings.row_group_size,
    )
    input_rows = retained_rows = excluded_rows = 0
    role_counts: Counter[str] = Counter()
    for batch in _open_text_batches(path, columns, block_size_bytes=settings.block_size_bytes):
        input_rows += batch.num_rows
        retained, dates, excluded = _date_first_filter(
            batch, "date", minimum=settings.scope_start, maximum=settings.scope_end
        )
        excluded_rows += excluded
        if retained.num_rows == 0:
            continue
        retained_rows += retained.num_rows
        secid = _required_positive_int(_column(retained, "secid"), label="option volume secid")
        flags = _column(retained, "cp_flag")
        flag_values = flags.to_pylist()
        unknown = sorted({value for value in flag_values if value not in {"", "C", "P"}})
        if unknown:
            raise OptionMetricsBuildError(f"Unknown cp_flag values: {unknown}")
        roles = pa.array(
            ["CALL" if value == "C" else "PUT" if value == "P" else "TOTAL" for value in flag_values],
            type=pa.string(),
        )
        authoritative = pc.is_in(roles, value_set=pa.array(["CALL", "PUT"]))
        volume = _nullable_nonnegative_int(_column(retained, "volume"), label="option volume")
        open_interest = _nullable_nonnegative_int(
            _column(retained, "open_interest"), label="option open interest"
        )
        table = pa.Table.from_arrays(
            [
                secid,
                dates,
                _availability_array(dates),
                pa.array(["DATED_CLOSE"] * retained.num_rows),
                roles,
                authoritative,
                volume,
                open_interest,
                pc.equal(volume, pa.scalar(0, type=pa.int64())),
                pc.equal(open_interest, pa.scalar(0, type=pa.int64())),
                _normalized_cusip_array(_column(retained, "cusip")),
            ],
            schema=VOLUME_SCHEMA,
        )
        role_counts.update(roles.to_pylist())
        sink.consume(table)
    output_rows = sink.finish()
    return {
        "source_rows": input_rows,
        "retained_rows": retained_rows,
        "excluded_rows": excluded_rows,
        "output_rows": output_rows,
        "role_counts": dict(sorted(role_counts.items())),
        "authoritative_roles": ["CALL", "PUT"],
        "total_rows_double_counted": 0,
    }


def stream_historical_volatility(settings: OptionMetricsSettings) -> dict[str, Any]:
    path = settings.raw_root / VOLATILITY_FILE
    columns = ["secid", "date", "days", "volatility", "cusip"]
    sink = _GroupedSortedParquetSink(
        _output_path(settings, "optionmetrics_historical_volatility_observation"),
        VOLATILITY_SCHEMA,
        base_keys=("secid",),
        sort_keys=("secid", "observation_date", "days"),
        row_group_size=settings.row_group_size,
    )
    input_rows = retained_rows = excluded_rows = 0
    supported_rows = 0
    for batch in _open_text_batches(path, columns, block_size_bytes=settings.block_size_bytes):
        input_rows += batch.num_rows
        retained, dates, excluded = _date_first_filter(
            batch, "date", minimum=settings.scope_start, maximum=settings.scope_end
        )
        excluded_rows += excluded
        if retained.num_rows == 0:
            continue
        retained_rows += retained.num_rows
        secid = _required_positive_int(_column(retained, "secid"), label="volatility secid")
        days = _required_positive_int(
            _column(retained, "days"), label="volatility horizon", out_type=pa.int16()
        )
        supported = pc.is_in(
            days, value_set=pa.array([30, 60, 91, 182, 365], type=pa.int16())
        )
        supported_rows += int(pc.sum(pc.cast(supported, pa.int64())).as_py() or 0)
        table = pa.Table.from_arrays(
            [
                secid,
                dates,
                _availability_array(dates),
                pa.array(["DATED_CLOSE"] * retained.num_rows),
                days,
                _nullable_exact_decimal(
                    _column(retained, "volatility"), label="historical volatility"
                ),
                supported,
                _normalized_cusip_array(_column(retained, "cusip")),
            ],
            schema=VOLATILITY_SCHEMA,
        )
        sink.consume(table)
    output_rows = sink.finish()
    return {
        "source_rows": input_rows,
        "retained_rows": retained_rows,
        "excluded_rows": excluded_rows,
        "output_rows": output_rows,
        "supported_horizon_rows": supported_rows,
    }


def scan_security_prices_qa(settings: OptionMetricsSettings) -> dict[str, Any]:
    """Create aggregate coverage/validity QA without a row-level price table.

    The returned receipt is persisted by the family/orchestrator receipt and
    later copied into the quality namespace.  It must not be written beside
    the four canonical Parquet tables in the interim table directory.
    """

    path = settings.raw_root / PRICE_FILE
    columns = ["secid", "date", "cusip", "close", "volume"]
    by_year: dict[int, Counter[str]] = defaultdict(Counter)
    distinct_secids: set[int] = set()
    input_rows = retained_rows = excluded_rows = 0
    for batch in _open_text_batches(path, columns, block_size_bytes=settings.block_size_bytes):
        input_rows += batch.num_rows
        retained, dates, excluded = _date_first_filter(
            batch, "date", minimum=settings.scope_start, maximum=settings.scope_end
        )
        excluded_rows += excluded
        if retained.num_rows == 0:
            continue
        retained_rows += retained.num_rows
        secids = _required_positive_int(_column(retained, "secid"), label="security price secid")
        close = _nullable_exact_decimal(_column(retained, "close"), label="security close")
        volume = _nullable_nonnegative_int(_column(retained, "volume"), label="security volume")
        cusip = _normalized_cusip_array(_column(retained, "cusip"))
        years = pc.year(dates).to_numpy(zero_copy_only=False)
        close_valid = pc.is_valid(close).to_numpy(zero_copy_only=False)
        volume_valid = pc.is_valid(volume).to_numpy(zero_copy_only=False)
        cusip_valid = pc.is_valid(cusip).to_numpy(zero_copy_only=False)
        secid_values = secids.to_numpy(zero_copy_only=False)
        distinct_secids.update(int(value) for value in np.unique(secid_values))
        for year in np.unique(years):
            mask = years == year
            counter = by_year[int(year)]
            counter["rows"] += int(mask.sum())
            counter["valid_close_rows"] += int(close_valid[mask].sum())
            counter["valid_volume_rows"] += int(volume_valid[mask].sum())
            counter["valid_cusip8_rows"] += int(cusip_valid[mask].sum())
    receipt = {
        "schema_version": 1,
        "source": PRICE_FILE,
        "scope": {"start": str(settings.scope_start), "end": str(settings.scope_end)},
        "source_rows": input_rows,
        "retained_rows": retained_rows,
        "excluded_rows": excluded_rows,
        "distinct_secids": len(distinct_secids),
        "row_level_table_materialized": False,
        "crsp_is_authoritative_market_source": True,
        "by_year": [
            {"year": year, **dict(sorted(counts.items()))}
            for year, counts in sorted(by_year.items())
        ],
    }
    return receipt


def _table_receipt(table: pa.Table, path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "rows": table.num_rows,
        "columns": table.column_names,
        "size_bytes": path.stat().st_size,
    }


def build_optionmetrics_family(
    settings: OptionMetricsSettings,
    *,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build all four normalized tables and aggregate-only price QA."""

    progress = progress or (lambda _stage, _details: None)
    settings.staging_dir.mkdir(parents=True, exist_ok=True)
    targets = [_output_path(settings, name) for name in OUTPUT_FILES]
    existing = [path.name for path in targets if path.exists()]
    if existing:
        raise OptionMetricsBuildError(f"Refusing overwrite of existing outputs: {existing}")

    progress("optionmetrics_names_started", {})
    names = build_name_intervals(settings)
    progress("optionmetrics_links_started", {"name_rows": names.num_rows})
    links = build_crsp_links(settings, names)
    progress("optionmetrics_volume_started", {})
    volume = stream_option_volume(settings)
    progress("optionmetrics_volatility_started", {})
    volatility = stream_historical_volatility(settings)
    progress("optionmetrics_price_qa_started", {})
    prices = scan_security_prices_qa(settings)
    result = {
        "schema_version": 1,
        "scope": {"start": str(settings.scope_start), "end": str(settings.scope_end)},
        "tables": {
            "optionmetrics_name_interval": _table_receipt(
                names, _output_path(settings, "optionmetrics_name_interval")
            ),
            "optionmetrics_crsp_link_interval": _table_receipt(
                links, _output_path(settings, "optionmetrics_crsp_link_interval")
            ),
            "optionmetrics_option_volume_observation": volume,
            "optionmetrics_historical_volatility_observation": volatility,
        },
        "security_prices_qa": prices,
        "authorizations": {
            "auxiliary_panel_authorized": False,
            "features_labels_models_portfolios_pnl_performance_or_backtests_authorized": False,
        },
    }
    progress("optionmetrics_complete", {"tables": 4})
    return result


def profile_first_batch(
    path: Path,
    *,
    columns: Sequence[str],
    date_column: str,
    minimum: dt.date | None = SCOPE_START,
    maximum: dt.date = SCOPE_END,
    block_size_bytes: int = 8 << 20,
) -> dict[str, Any]:
    """Read only the header and first small Arrow block for throughput planning."""

    started = time.perf_counter()
    reader = _open_text_batches(
        path,
        columns,
        block_size_bytes=block_size_bytes,
        use_threads=False,
    )
    try:
        batch = reader.read_next_batch()
    except StopIteration as exc:
        raise OptionMetricsBuildError(f"No data rows in {path.name}") from exc
    retained, _, excluded = _date_first_filter(
        batch, date_column, minimum=minimum, maximum=maximum
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "source": path.name,
        "header_columns": len(_header(path)),
        "first_batch_rows": batch.num_rows,
        "first_batch_retained_rows": retained.num_rows,
        "first_batch_excluded_rows": excluded,
        "decoded_batch_bytes": batch.nbytes,
        "elapsed_seconds": round(elapsed, 6),
        "decoded_mib_per_second": round(batch.nbytes / (1 << 20) / elapsed, 3),
        "configured_source_block_mib": block_size_bytes / (1 << 20),
        "approx_source_mib_per_second": round(
            block_size_bytes / (1 << 20) / elapsed, 3
        ),
        "full_scan_performed": False,
    }
