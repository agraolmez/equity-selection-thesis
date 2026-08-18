"""Build the development-only SEC, 13F, and restatement auxiliary family.

Large CSVs are read in native Arrow record batches.  Required date/form text is
validated first and the semantic payload is converted to Python only for rows
inside the development knowledge boundary.  This module writes only into a
caller-supplied staging directory and does not publish pointers or database
authority.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


SCOPE_END = dt.date(2014, 10, 31)
FIRST_SIGNAL = dt.date(2000, 1, 31)
FORM_FAMILIES = frozenset({"10-K", "10-Q", "20-F", "40-F", "8-K", "6-K"})
CSV_BLOCK_SIZE_BYTES = 8 << 20
_CUSIP_PATTERN = re.compile(r"^[A-Z0-9]{8,}$")
_TIME_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d(?:\.\d{1,6})?)?$")


class SecRestatementBuildError(RuntimeError):
    """Raised when a source or output violates the frozen contract."""


def _schema(fields: Sequence[tuple[str, pa.DataType, bool]]) -> pa.Schema:
    return pa.schema([pa.field(name, kind, nullable=nullable) for name, kind, nullable in fields])


D6 = pa.decimal128(38, 6)
SEC_FILING_SCHEMA = _schema([
    ("source_record_id", pa.string(), False), ("cik", pa.string(), False),
    ("filing_date", pa.date32(), False), ("form_family", pa.string(), False),
    ("source_filename", pa.string(), False), ("source", pa.string(), False),
    ("observation_date", pa.date32(), False), ("public_availability_date", pa.date32(), False),
    ("availability_basis", pa.string(), False),
])
SEC_8K_SCHEMA = _schema([
    ("source_record_id", pa.string(), False), ("accession", pa.string(), True),
    ("cik", pa.string(), False), ("filing_date", pa.date32(), False),
    ("form_family", pa.string(), False), ("item_number", pa.string(), True),
    ("item_text", pa.string(), True), ("acceptance_date", pa.date32(), True),
    ("acceptance_time", pa.string(), True), ("observation_date", pa.date32(), False),
    ("public_availability_date", pa.date32(), False), ("availability_basis", pa.string(), False),
])
SEC_13F_FILING_SCHEMA = _schema([
    ("filing_id", pa.string(), False), ("manager_cik", pa.string(), False),
    ("filing_date", pa.date32(), False), ("report_date", pa.date32(), False),
    ("is_amendment", pa.bool_(), False), ("amendment_type", pa.string(), True),
    ("reported_entry_total", pa.int64(), True), ("reported_value_total", D6, True),
    ("other_manager_count", pa.int32(), False), ("observation_date", pa.date32(), False),
    ("public_availability_date", pa.date32(), False), ("availability_basis", pa.string(), False),
    ("stale_after_date", pa.date32(), False),
])
SEC_13F_HOLDING_SCHEMA = _schema([
    ("filing_id", pa.string(), False), ("source_row_ordinal", pa.int64(), False),
    ("manager_cik", pa.string(), False), ("filing_date", pa.date32(), False),
    ("report_date", pa.date32(), False), ("normalized_cusip8", pa.string(), True),
    ("issuer_name_audit_only", pa.string(), True), ("title_of_class", pa.string(), True),
    ("reported_value", D6, True), ("reported_shares", D6, True),
    ("share_type", pa.string(), True), ("put_call", pa.string(), True),
    ("discretion", pa.string(), True), ("sole", D6, True),
    ("shared", D6, True), ("none", D6, True),
    ("observation_date", pa.date32(), False), ("public_availability_date", pa.date32(), False),
    ("availability_basis", pa.string(), False),
])
RESTATEMENT_SCHEMA = _schema([
    ("restatement_notification_key", pa.string(), False), ("company_fkey", pa.string(), False),
    ("file_date", pa.date32(), False), ("file_accepted", pa.timestamp("us", tz="UTC"), True),
    ("event_date", pa.date32(), True), ("normalized_cusip8", pa.string(), True),
    ("affected_period_start", pa.date32(), True), ("affected_period_end", pa.date32(), True),
    ("accounting_flag", pa.bool_(), True), ("fraud_flag", pa.bool_(), True),
    ("adverse_flag", pa.bool_(), True), ("improves_flag", pa.bool_(), True),
    ("general_type", pa.string(), True), ("currency_code", pa.string(), True),
    ("observation_date", pa.date32(), False), ("public_availability_date", pa.date32(), False),
    ("availability_basis", pa.string(), False),
])
RESTATEMENT_LINK_SCHEMA = _schema([
    ("restatement_notification_key", pa.string(), False), ("permno", pa.int64(), True),
    ("link_status", pa.string(), False), ("link_basis", pa.string(), False),
    ("candidate_count", pa.int32(), False), ("matched_identity_start_date", pa.date32(), True),
    ("matched_identity_end_date", pa.date32(), True),
])

TABLE_SPECS: dict[str, tuple[pa.Schema, tuple[str, ...], tuple[str, ...]]] = {
    "sec_filing_observation": (SEC_FILING_SCHEMA, ("source_record_id",), ("cik", "filing_date", "source_record_id")),
    "sec_8k_item_observation": (SEC_8K_SCHEMA, ("source_record_id",), ("cik", "filing_date", "accession", "source_record_id")),
    "sec_13f_filing_observation": (SEC_13F_FILING_SCHEMA, ("filing_id",), ("filing_date", "filing_id")),
    "sec_13f_holding_observation": (SEC_13F_HOLDING_SCHEMA, ("filing_id", "source_row_ordinal"), ("filing_id", "source_row_ordinal")),
    "restatement_event_observation": (RESTATEMENT_SCHEMA, ("restatement_notification_key",), ("public_availability_date", "restatement_notification_key")),
    "restatement_crsp_link": (RESTATEMENT_LINK_SCHEMA, ("restatement_notification_key",), ("restatement_notification_key",)),
}


@dataclass(frozen=True)
class GateDecision:
    include: bool
    observation_date: dt.date | None
    availability_date: dt.date | None
    availability_basis: str | None
    disposition: str


@dataclass
class SourceReceipt:
    source_records_seen: int = 0
    arrow_record_batches: int = 0
    payload_rows_decoded: int = 0
    output_rows: int = 0
    pre_first_signal_context_rows: int = 0
    post_scope_rows_not_decoded: int = 0
    boundary_rows_not_decoded: int = 0
    invalid_gate_rows: int = 0
    unsupported_form_rows_not_decoded: int = 0
    blank_amendment_flags_inferred_from_form: int = 0
    missing_required_report_date_rows_excluded: int = 0
    stopped_by_profile_limit: bool = False


def normalize_cusip8(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace(" ", "").replace("-", "").upper()
    if not _CUSIP_PATTERN.fullmatch(normalized):
        return None
    return normalized[:8]


def _form_family(value: str) -> str:
    return value.strip().upper().removesuffix("/A")


def _clean(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _required(value: str, name: str) -> str:
    result = _clean(value)
    if result is None:
        raise SecRestatementBuildError(f"Required source value {name!r} is blank")
    return result


def _date(value: str, *, required: bool = False) -> dt.date | None:
    stripped = value.strip()
    if not stripped:
        if required:
            raise SecRestatementBuildError("Required date is blank")
        return None
    try:
        return dt.date.fromisoformat(stripped[:10])
    except ValueError as exc:
        if required:
            raise SecRestatementBuildError(f"Invalid date {stripped!r}") from exc
        return None


def _timestamp_utc(value: str) -> dt.datetime | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = dt.datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _bool_value(value: str, *, required: bool = False) -> bool | None:
    stripped = value.strip().upper()
    if not stripped:
        if required:
            raise SecRestatementBuildError("Required Boolean is blank")
        return None
    if stripped in {"1", "Y", "YES", "TRUE", "T"}:
        return True
    if stripped in {"0", "N", "NO", "FALSE", "F"}:
        return False
    raise SecRestatementBuildError(f"Invalid Boolean token {value!r}")


def _integer(value: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    if not re.fullmatch(r"[+-]?\d+", stripped):
        raise SecRestatementBuildError(f"Non-integral value {value!r}")
    return int(stripped)


def _decimal6(value: str) -> Decimal | None:
    stripped = value.strip().replace(",", "")
    if not stripped:
        return None
    try:
        source = Decimal(stripped)
    except Exception as exc:
        raise SecRestatementBuildError(f"Invalid exact decimal {value!r}") from exc
    quantized = source.quantize(Decimal("0.000001"))
    if quantized != source:
        raise SecRestatementBuildError(f"Decimal exceeds frozen scale: {value!r}")
    return quantized


def _source_id(source: str, ordinal: int, *parts: str) -> str:
    digest = hashlib.sha256()
    for value in (source, str(ordinal), *parts):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


class TradingCalendar:
    def __init__(self, dates: Sequence[dt.date]):
        ordered = np.asarray(sorted(set(dates)), dtype="datetime64[D]").astype(np.int64)
        if not len(ordered):
            raise SecRestatementBuildError("Trading calendar is empty")
        self.ordinals = ordered
        self._strict_after_cache: dict[dt.date, dt.date] = {}

    @classmethod
    def from_parquet(cls, path: Path) -> "TradingCalendar":
        table = pq.read_table(path)
        name = "calendar_date" if "calendar_date" in table.column_names else next(
            item for item in table.column_names if "date" in item.lower()
        )
        values = table[name].combine_chunks().to_pylist()
        return cls(values)

    def strictly_after(self, value: dt.date) -> dt.date:
        cached = self._strict_after_cache.get(value)
        if cached is not None:
            return cached
        ordinal = np.datetime64(value, "D").astype(np.int64)
        position = int(np.searchsorted(self.ordinals, ordinal, side="right"))
        if position >= len(self.ordinals):
            raise SecRestatementBuildError(f"Calendar has no session after {value}")
        result = dt.date(1970, 1, 1) + dt.timedelta(days=int(self.ordinals[position]))
        self._strict_after_cache[value] = result
        return result


def _csv_header(path: Path) -> list[str]:
    with path.open("rb") as handle:
        line = handle.readline(1 << 20)
    if not line or (len(line) == 1 << 20 and not line.endswith((b"\n", b"\r"))):
        raise SecRestatementBuildError(f"Missing or overlong CSV header: {path.name}")
    try:
        return next(csv.reader([line.decode("utf-8-sig").rstrip("\r\n")]))
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise SecRestatementBuildError(f"Invalid CSV header: {path.name}") from exc


def _open_text_batches(path: Path, columns: Sequence[str]) -> pacsv.CSVStreamingReader:
    """Open selected columns as text; typed conversion follows the date gate."""

    header = _csv_header(path)
    missing = sorted(set(columns) - set(header))
    if missing:
        raise SecRestatementBuildError(f"Missing columns in {path.name}: {missing}")
    try:
        return pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(
                block_size=max(4_096, CSV_BLOCK_SIZE_BYTES),
                use_threads=True,
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
        raise SecRestatementBuildError(f"Cannot open {path.name}") from exc


class _Gate(Protocol):
    def decide(
        self,
        columns: Mapping[str, list[str]],
        index: int,
    ) -> GateDecision: ...


def _date_first_rows(
    path: Path,
    gate_columns: Sequence[str],
    payload_columns: Sequence[str],
    gate: _Gate,
    receipt: SourceReceipt,
    *,
    max_source_records: int | None,
) -> Iterator[tuple[int, dict[str, str], GateDecision]]:
    selected_columns = tuple(dict.fromkeys((*gate_columns, *payload_columns)))
    reader = _open_text_batches(path, selected_columns)
    try:
        for source_batch in reader:
            if max_source_records is not None:
                remaining = max_source_records - receipt.source_records_seen
                if remaining <= 0:
                    receipt.stopped_by_profile_limit = True
                    break
                if source_batch.num_rows > remaining:
                    source_batch = source_batch.slice(0, remaining)
                    receipt.stopped_by_profile_limit = True
            first_ordinal = receipt.source_records_seen + 1
            receipt.source_records_seen += source_batch.num_rows
            receipt.arrow_record_batches += 1
            gate_values = {
                name: source_batch.column(source_batch.schema.get_field_index(name)).to_pylist()
                for name in gate_columns
            }
            retained_indices: list[int] = []
            retained_decisions: list[GateDecision] = []
            retained_ordinals: list[int] = []
            for index in range(source_batch.num_rows):
                ordinal = first_ordinal + index
                try:
                    decision = gate.decide(gate_values, index)
                except Exception:
                    receipt.invalid_gate_rows += 1
                    continue
                if decision.include:
                    retained_indices.append(index)
                    retained_decisions.append(decision)
                    retained_ordinals.append(ordinal)
                    continue
                if decision.disposition == "post_scope":
                    receipt.post_scope_rows_not_decoded += 1
                elif decision.disposition == "boundary":
                    receipt.boundary_rows_not_decoded += 1
                elif decision.disposition == "unsupported_form":
                    receipt.unsupported_form_rows_not_decoded += 1
                else:
                    receipt.invalid_gate_rows += 1
            if retained_indices:
                retained = source_batch.take(pa.array(retained_indices, type=pa.int64()))
                payload_rows = retained.to_pylist()
                receipt.payload_rows_decoded += retained.num_rows
                for ordinal, row, decision in zip(
                    retained_ordinals, payload_rows, retained_decisions, strict=True
                ):
                    if (
                        decision.availability_date is not None
                        and decision.availability_date < FIRST_SIGNAL
                    ):
                        receipt.pre_first_signal_context_rows += 1
                    yield ordinal, row, decision
            if (
                max_source_records is not None
                and receipt.source_records_seen >= max_source_records
            ):
                receipt.stopped_by_profile_limit = True
                break
    except pa.ArrowException as exc:
        raise SecRestatementBuildError(f"Arrow CSV parse failed for {path.name}") from exc
    finally:
        reader.close()


@dataclass(frozen=True)
class _FilingGate:
    calendar: TradingCalendar
    allowed_forms: frozenset[str]

    def decide(self, columns: Mapping[str, list[str]], index: int) -> GateDecision:
        family = _form_family(columns["form"][index])
        if family not in self.allowed_forms:
            return GateDecision(False, None, None, None, "unsupported_form")
        filing = _date(columns["fdate"][index], required=True)
        assert filing is not None
        if filing > SCOPE_END:
            return GateDecision(False, filing, None, None, "post_scope")
        available = self.calendar.strictly_after(filing)
        if available > SCOPE_END:
            return GateDecision(False, filing, available, None, "boundary")
        return GateDecision(True, filing, available, "filing_date_plus_one_trading_day", "include")


@dataclass(frozen=True)
class _EightKGate:
    calendar: TradingCalendar

    def decide(self, columns: Mapping[str, list[str]], index: int) -> GateDecision:
        if _form_family(columns["form"][index]) != "8-K":
            return GateDecision(False, None, None, None, "unsupported_form")
        filing = _date(columns["fdate"][index], required=True)
        assert filing is not None
        acceptance = _date(columns["secadate"][index])
        valid_time = bool(_TIME_PATTERN.fullmatch(columns["secatime"][index].strip()))
        observation = acceptance if acceptance is not None and valid_time else filing
        basis = (
            "validated_sec_acceptance_date_time_plus_one_trading_day"
            if acceptance is not None and valid_time
            else "filing_date_plus_one_trading_day_fallback"
        )
        if observation > SCOPE_END:
            return GateDecision(False, observation, None, None, "post_scope")
        available = self.calendar.strictly_after(observation)
        if available > SCOPE_END:
            return GateDecision(False, observation, available, None, "boundary")
        return GateDecision(True, observation, available, basis, "include")


@dataclass(frozen=True)
class _RestatementGate:
    calendar: TradingCalendar

    def decide(self, columns: Mapping[str, list[str]], index: int) -> GateDecision:
        file_date = _date(columns["file_date"][index], required=True)
        assert file_date is not None
        accepted = _timestamp_utc(columns["file_accepted"][index])
        observation = accepted.date() if accepted is not None else file_date
        basis = "file_accepted_plus_one_trading_day" if accepted is not None else "file_date_plus_one_trading_day_fallback"
        if observation > SCOPE_END:
            return GateDecision(False, observation, None, None, "post_scope")
        available = self.calendar.strictly_after(observation)
        if available > SCOPE_END:
            return GateDecision(False, observation, available, None, "boundary")
        return GateDecision(True, observation, available, basis, "include")


def _filing_gate(calendar: TradingCalendar, *, allowed_forms: frozenset[str]) -> _FilingGate:
    return _FilingGate(calendar, allowed_forms)


def _eight_k_gate(calendar: TradingCalendar) -> _EightKGate:
    return _EightKGate(calendar)


def _restatement_gate(calendar: TradingCalendar) -> _RestatementGate:
    return _RestatementGate(calendar)


def _duckdb_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _duckdb_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _rewrite_with_exact_schema(
    source: Path,
    target: Path,
    schema: pa.Schema,
    *,
    batch_rows: int,
) -> None:
    """Stream a DuckDB-sorted Parquet into a file carrying the frozen Arrow schema."""

    writer = pq.ParquetWriter(target, schema, compression="zstd", use_dictionary=True)
    try:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=batch_rows):
            table = pa.Table.from_batches([batch]).select(schema.names).cast(schema, safe=True)
            writer.write_table(table, row_group_size=batch_rows)
    finally:
        writer.close()


def _stream_verify_sort(
    path: Path,
    sort_key: tuple[str, ...],
    *,
    batch_rows: int,
    adjacent_primary_key: tuple[str, ...] | None = None,
) -> None:
    """Verify global order and, when aligned, exact adjacent PK uniqueness."""

    prior: tuple[tuple[int, Any], ...] | None = None
    prior_primary: tuple[Any, ...] | None = None
    parquet = pq.ParquetFile(path)
    columns_needed = list(dict.fromkeys((*sort_key, *(adjacent_primary_key or ()))))
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns_needed):
        columns = batch.to_pydict()
        for index in range(batch.num_rows):
            key = tuple(
                (0, None) if columns[name][index] is None else (1, columns[name][index])
                for name in sort_key
            )
            if prior is not None and key < prior:
                raise SecRestatementBuildError(f"Global sort verification failed for {path.name}")
            prior = key
            if adjacent_primary_key is not None:
                primary = tuple(columns[name][index] for name in adjacent_primary_key)
                if prior_primary is not None and primary == prior_primary:
                    raise SecRestatementBuildError(f"Duplicate primary key in {path.name}")
                prior_primary = primary


class _Sink:
    def __init__(
        self,
        path: Path,
        unsorted_path: Path,
        schema: pa.Schema,
        primary_key: tuple[str, ...],
        sort_key: tuple[str, ...],
        batch_rows: int,
    ):
        self.path = path
        self.unsorted_path = unsorted_path
        self.schema = schema
        self.primary_key = primary_key
        self.sort_key = sort_key
        self.batch_rows = batch_rows
        self.pending: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def append(self, row: dict[str, Any]) -> None:
        self.pending.append(row)
        if len(self.pending) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        table = pa.Table.from_pylist(self.pending, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.unsorted_path, self.schema, compression="zstd", use_dictionary=True
            )
        self.writer.write_table(table)
        self.rows += len(self.pending)
        self.pending.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            pq.write_table(
                pa.Table.from_pylist([], schema=self.schema),
                self.unsorted_path,
                compression="zstd",
            )
        else:
            self.writer.close()

    def external_sort_and_verify(
        self,
        connection: duckdb.DuckDBPyConnection,
        work_dir: Path,
    ) -> Path:
        """Disk-backed global sort, exact-schema rewrite, and bounded validation."""

        primary_is_sort_prefix = self.sort_key[: len(self.primary_key)] == self.primary_key
        if not primary_is_sort_prefix:
            primary = ", ".join(_duckdb_identifier(name) for name in self.primary_key)
            duplicate = connection.execute(
                f"SELECT {primary}, count(*) AS n "
                f"FROM read_parquet({_duckdb_literal(self.unsorted_path)}) "
                f"GROUP BY {primary} HAVING count(*) > 1 LIMIT 1"
            ).fetchone()
            if duplicate is not None:
                raise SecRestatementBuildError(f"Duplicate primary key in {self.path.name}")

        order = ", ".join(
            f"{_duckdb_identifier(name)} ASC NULLS FIRST" for name in self.sort_key
        )
        sorted_path = work_dir / f"{self.path.stem}.sorted.parquet"
        exact_path = work_dir / f"{self.path.stem}.exact.parquet"
        row_group_size = max(2_048, self.batch_rows)
        connection.execute(
            f"COPY (SELECT * FROM read_parquet({_duckdb_literal(self.unsorted_path)}) "
            f"ORDER BY {order}) TO {_duckdb_literal(sorted_path)} "
            f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {row_group_size})"
        )
        _rewrite_with_exact_schema(
            sorted_path, exact_path, self.schema, batch_rows=self.batch_rows
        )
        parquet = pq.ParquetFile(exact_path)
        if not parquet.schema_arrow.equals(self.schema, check_metadata=True):
            raise SecRestatementBuildError(f"Exact output schema differs for {self.path.stem}")
        if parquet.metadata.num_rows != self.rows:
            raise SecRestatementBuildError(f"Output row count differs for {self.path.stem}")
        _stream_verify_sort(
            exact_path,
            self.sort_key,
            batch_rows=self.batch_rows,
            adjacent_primary_key=self.primary_key if primary_is_sort_prefix else None,
        )
        return exact_path


def _identity_index(path: Path) -> dict[str, list[tuple[int, dt.date, dt.date]]]:
    table = pq.read_table(path, columns=["permno", "start_date", "end_date", "cusip"])
    result: dict[str, list[tuple[int, dt.date, dt.date]]] = defaultdict(list)
    for permno, start, end, cusip in zip(
        table["permno"].to_pylist(), table["start_date"].to_pylist(),
        table["end_date"].to_pylist(), table["cusip"].to_pylist(), strict=True,
    ):
        normalized = normalize_cusip8(cusip)
        if normalized is not None:
            result[normalized].append((int(permno), start, end))
    return dict(result)


def _company_bridge(path: Path | None) -> dict[str, list[tuple[int, dt.date, dt.date]]]:
    if path is None:
        return {}
    table = pq.read_table(path)
    required = {"company_fkey", "permno", "start_date", "end_date"}
    if not required.issubset(table.column_names):
        raise SecRestatementBuildError("Company-key bridge schema differs")
    result: dict[str, list[tuple[int, dt.date, dt.date]]] = defaultdict(list)
    for company, permno, start, end in zip(
        table["company_fkey"].to_pylist(), table["permno"].to_pylist(),
        table["start_date"].to_pylist(), table["end_date"].to_pylist(), strict=True,
    ):
        result[str(company)].append((int(permno), start, end))
    return dict(result)


def _link_restatement(
    row: Mapping[str, Any],
    identity: Mapping[str, list[tuple[int, dt.date, dt.date]]],
    bridge: Mapping[str, list[tuple[int, dt.date, dt.date]]],
) -> dict[str, Any]:
    date = row["public_availability_date"]
    bridge_candidates = [item for item in bridge.get(row["company_fkey"], []) if item[1] <= date <= item[2]]
    if bridge_candidates:
        candidates = bridge_candidates
        basis = "company_key_bridge_date_valid"
    else:
        candidates = [
            item for item in identity.get(row["normalized_cusip8"], [])
            if item[1] <= date <= item[2]
        ] if row["normalized_cusip8"] is not None else []
        basis = "normalized_cusip8_date_valid" if row["normalized_cusip8"] is not None else "UNMATCHED"
    by_permno: dict[int, list[tuple[int, dt.date, dt.date]]] = defaultdict(list)
    for candidate in candidates:
        by_permno[candidate[0]].append(candidate)
    if not by_permno:
        return {
            "restatement_notification_key": row["restatement_notification_key"],
            "permno": None, "link_status": "UNMATCHED", "link_basis": "UNMATCHED",
            "candidate_count": 0, "matched_identity_start_date": None,
            "matched_identity_end_date": None,
        }
    if len(by_permno) > 1:
        return {
            "restatement_notification_key": row["restatement_notification_key"],
            "permno": None, "link_status": "AMBIGUOUS", "link_basis": basis,
            "candidate_count": len(by_permno), "matched_identity_start_date": None,
            "matched_identity_end_date": None,
        }
    permno = next(iter(by_permno))
    chosen = sorted(by_permno[permno], key=lambda item: (item[1], item[2]))[-1]
    return {
        "restatement_notification_key": row["restatement_notification_key"],
        "permno": permno, "link_status": "MATCHED", "link_basis": basis,
        "candidate_count": 1, "matched_identity_start_date": chosen[1],
        "matched_identity_end_date": chosen[2],
    }


def build_sec_restatement_family(
    *,
    raw_root: Path,
    trading_calendar_path: Path,
    crsp_identity_path: Path,
    staging_dir: Path,
    company_key_bridge_path: Path | None = None,
    batch_rows: int = 100_000,
    max_source_records_per_file: int | None = None,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build six frozen tables into ``staging_dir`` and return aggregate receipts."""

    progress = progress or (lambda _stage, _details: None)
    if batch_rows <= 0:
        raise SecRestatementBuildError("batch_rows must be positive")
    staging_dir.mkdir(parents=True, exist_ok=True)
    if any((staging_dir / f"{name}.parquet").exists() for name in TABLE_SPECS):
        raise SecRestatementBuildError("A target SEC/restatement staging output already exists")
    calendar = TradingCalendar.from_parquet(trading_calendar_path)
    identity = _identity_index(crsp_identity_path)
    bridge = _company_bridge(company_key_bridge_path)
    work_context = tempfile.TemporaryDirectory(prefix=".sec_restatement_work_", dir=staging_dir)
    work_dir = Path(work_context.name)
    receipts = {name: SourceReceipt() for name in (
        "sec_filings_index.csv", "sec_8k_items.csv", "sec_13f_summary.csv",
        "sec_13f_holdings.csv", "audit_financial_restatements.csv",
    )}
    sinks = {
        name: _Sink(
            staging_dir / f"{name}.parquet",
            work_dir / f"{name}.unsorted.parquet",
            schema,
            primary,
            sort,
            batch_rows,
        )
        for name, (schema, primary, sort) in TABLE_SPECS.items()
    }

    filing_receipt = receipts["sec_filings_index.csv"]
    progress("sec_filings_index_started", {})
    for ordinal, row, decision in _date_first_rows(
        raw_root / "sec_filings_index.csv", ("fdate", "form"),
        ("cik", "fdate", "form", "fname", "source"),
        _filing_gate(calendar, allowed_forms=FORM_FAMILIES), filing_receipt,
        max_source_records=max_source_records_per_file,
    ):
        family = _form_family(row["form"])
        output = {
            "source_record_id": _source_id("sec_filings_index.csv", ordinal, row["cik"], row["fdate"], row["fname"]),
            "cik": _required(row["cik"], "cik"), "filing_date": _date(row["fdate"], required=True),
            "form_family": family, "source_filename": _required(row["fname"], "fname"),
            "source": _required(row["source"], "source"), "observation_date": decision.observation_date,
            "public_availability_date": decision.availability_date, "availability_basis": decision.availability_basis,
        }
        sinks["sec_filing_observation"].append(output)
        filing_receipt.output_rows += 1
    progress("sec_filings_index_complete", vars(filing_receipt).copy())

    item_receipt = receipts["sec_8k_items.csv"]
    progress("sec_8k_items_started", {})
    for ordinal, row, decision in _date_first_rows(
        raw_root / "sec_8k_items.csv", ("fdate", "form", "secadate", "secatime"),
        ("cik", "fdate", "form", "secadate", "secatime", "accession", "nitem", "item"),
        _eight_k_gate(calendar), item_receipt,
        max_source_records=max_source_records_per_file,
    ):
        acceptance_date = _date(row["secadate"])
        acceptance_time = _clean(row["secatime"])
        if acceptance_date is None or acceptance_time is None or not _TIME_PATTERN.fullmatch(acceptance_time):
            acceptance_date = None
            acceptance_time = None
        output = {
            "source_record_id": _source_id("sec_8k_items.csv", ordinal, row["accession"], row["cik"], row["nitem"]),
            "accession": _clean(row["accession"]), "cik": _required(row["cik"], "cik"),
            "filing_date": _date(row["fdate"], required=True), "form_family": "8-K",
            "item_number": _clean(row["nitem"]), "item_text": _clean(row["item"]),
            "acceptance_date": acceptance_date, "acceptance_time": acceptance_time,
            "observation_date": decision.observation_date, "public_availability_date": decision.availability_date,
            "availability_basis": decision.availability_basis,
        }
        sinks["sec_8k_item_observation"].append(output)
        item_receipt.output_rows += 1
    progress("sec_8k_items_complete", vars(item_receipt).copy())

    summary_receipt = receipts["sec_13f_summary.csv"]
    progress("sec_13f_summary_started", {})
    summaries: dict[str, dict[str, Any]] = {}
    manager_tokens: dict[str, set[str]] = defaultdict(set)
    reported_manager_counts: dict[str, int] = defaultdict(int)
    summary_gate = _filing_gate(calendar, allowed_forms=frozenset({"13F-HR"}))
    for _, row, decision in _date_first_rows(
        raw_root / "sec_13f_summary.csv", ("fdate", "form"),
        (
            "cik", "fdate", "form", "rdate", "fname", "isamendment",
            "amendmenttype", "tableentrytotal", "tablevaluetotal",
            "otherincludedmanagerscount", "om", "omcik",
        ),
        summary_gate,
        summary_receipt, max_source_records=max_source_records_per_file,
    ):
        filing_id = _required(row["fname"], "fname")
        report_date = _date(row["rdate"])
        if report_date is None:
            summary_receipt.missing_required_report_date_rows_excluded += 1
            continue
        amendment_flag = _bool_value(row["isamendment"])
        if amendment_flag is None:
            amendment_flag = row["form"].strip().upper().endswith("/A")
            summary_receipt.blank_amendment_flags_inferred_from_form += 1
        core = {
            "filing_id": filing_id, "manager_cik": _required(row["cik"], "cik"),
            "filing_date": _date(row["fdate"], required=True), "report_date": report_date,
            "is_amendment": amendment_flag,
            "amendment_type": _clean(row["amendmenttype"]),
            "reported_entry_total": _integer(row["tableentrytotal"]),
            "reported_value_total": _decimal6(row["tablevaluetotal"]),
            "observation_date": decision.observation_date, "public_availability_date": decision.availability_date,
            "availability_basis": "13f_filing_or_amendment_date_plus_one_trading_day",
            "stale_after_date": decision.availability_date + dt.timedelta(days=180),
        }
        prior = summaries.get(filing_id)
        if prior is not None and prior != core:
            raise SecRestatementBuildError(f"Inconsistent 13F summary rows for filing {filing_id!r}")
        summaries[filing_id] = core
        token = _clean(row.get("om", "")) or _clean(row.get("omcik", ""))
        if token is not None:
            manager_tokens[filing_id].add(token)
        reported_manager_counts[filing_id] = max(
            reported_manager_counts[filing_id],
            _integer(row.get("otherincludedmanagerscount", "")) or 0,
        )
    for filing_id, core in sorted(summaries.items(), key=lambda item: (item[1]["filing_date"], item[0])):
        core = {
            **core,
            "other_manager_count": max(
                len(manager_tokens[filing_id]), reported_manager_counts[filing_id]
            ),
        }
        sinks["sec_13f_filing_observation"].append(core)
        summary_receipt.output_rows += 1
    progress("sec_13f_summary_complete", vars(summary_receipt).copy())

    holding_receipt = receipts["sec_13f_holdings.csv"]
    progress("sec_13f_holdings_started", {})
    holding_ordinals: Counter[str] = Counter()
    for _, row, decision in _date_first_rows(
        raw_root / "sec_13f_holdings.csv", ("fdate", "form"),
        (
            "cik", "fdate", "form", "rdate", "fname", "nameofissuer",
            "titleofclass", "cusip", "value", "sshprnamt", "sshprnamttype",
            "putcall", "investmentdiscretion", "sole", "shared", "none",
        ),
        summary_gate,
        holding_receipt, max_source_records=max_source_records_per_file,
    ):
        filing_id = _required(row["fname"], "fname")
        report_date = _date(row["rdate"])
        if report_date is None:
            holding_receipt.missing_required_report_date_rows_excluded += 1
            continue
        holding_ordinals[filing_id] += 1
        output = {
            "filing_id": filing_id, "source_row_ordinal": holding_ordinals[filing_id],
            "manager_cik": _required(row["cik"], "cik"), "filing_date": _date(row["fdate"], required=True),
            "report_date": report_date, "normalized_cusip8": normalize_cusip8(row["cusip"]),
            "issuer_name_audit_only": _clean(row["nameofissuer"]), "title_of_class": _clean(row["titleofclass"]),
            "reported_value": _decimal6(row["value"]), "reported_shares": _decimal6(row["sshprnamt"]),
            "share_type": _clean(row["sshprnamttype"]), "put_call": _clean(row["putcall"]),
            "discretion": _clean(row["investmentdiscretion"]), "sole": _decimal6(row["sole"]),
            "shared": _decimal6(row["shared"]), "none": _decimal6(row["none"]),
            "observation_date": decision.observation_date, "public_availability_date": decision.availability_date,
            "availability_basis": "13f_filing_or_amendment_date_plus_one_trading_day",
        }
        sinks["sec_13f_holding_observation"].append(output)
        holding_receipt.output_rows += 1
    progress("sec_13f_holdings_complete", vars(holding_receipt).copy())

    restatement_receipt = receipts["audit_financial_restatements.csv"]
    progress("restatements_started", {})
    link_status_counts: Counter[str] = Counter()
    reversed_affected_period_intervals_set_null = 0
    for _, row, decision in _date_first_rows(
        raw_root / "audit_financial_restatements.csv", ("file_date", "file_accepted"),
        (
            "company_fkey", "file_date", "file_accepted", "event_date",
            "restatement_notification_key", "cusip_number", "res_begin_date",
            "res_end_date", "res_accounting", "res_fraud", "res_adverse",
            "res_improves", "general_type", "currency_code_fkey",
        ),
        _restatement_gate(calendar), restatement_receipt,
        max_source_records=max_source_records_per_file,
    ):
        key = _required(row["restatement_notification_key"], "restatement_notification_key")
        affected_period_start = _date(row["res_begin_date"])
        affected_period_end = _date(row["res_end_date"])
        if (
            affected_period_start is not None
            and affected_period_end is not None
            and affected_period_start > affected_period_end
        ):
            # A reversed source interval cannot safely support an affected-
            # period feature.  Retain the publicly observed restatement event,
            # but do not swap, infer, or expose either invalid endpoint.
            affected_period_start = None
            affected_period_end = None
            reversed_affected_period_intervals_set_null += 1
        output = {
            "restatement_notification_key": key, "company_fkey": _required(row["company_fkey"], "company_fkey"),
            "file_date": _date(row["file_date"], required=True), "file_accepted": _timestamp_utc(row["file_accepted"]),
            "event_date": _date(row["event_date"]), "normalized_cusip8": normalize_cusip8(row["cusip_number"]),
            "affected_period_start": affected_period_start,
            "affected_period_end": affected_period_end,
            "accounting_flag": _bool_value(row["res_accounting"]), "fraud_flag": _bool_value(row["res_fraud"]),
            "adverse_flag": _bool_value(row["res_adverse"]), "improves_flag": _bool_value(row["res_improves"]),
            "general_type": _clean(row["general_type"]), "currency_code": _clean(row["currency_code_fkey"]),
            "observation_date": decision.observation_date, "public_availability_date": decision.availability_date,
            "availability_basis": decision.availability_basis,
        }
        link = _link_restatement(output, identity, bridge)
        sinks["restatement_event_observation"].append(output)
        sinks["restatement_crsp_link"].append(link)
        link_status_counts[link["link_status"]] += 1
        restatement_receipt.output_rows += 1
    progress("restatements_complete", vars(restatement_receipt).copy())

    for sink in sinks.values():
        sink.close()

    spill_dir = work_dir / "duckdb_spill"
    spill_dir.mkdir()
    connection = duckdb.connect(":memory:")
    exact_candidates: dict[str, Path] = {}
    try:
        connection.execute("SET threads = 1")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET memory_limit = '256MB'")
        connection.execute("SET temp_directory = ?", [str(spill_dir)])
        for name, sink in sinks.items():
            progress("sec_external_sort_started", {"table": name, "rows": sink.rows})
            exact_candidates[name] = sink.external_sort_and_verify(connection, work_dir)
            progress("sec_external_sort_complete", {"table": name, "rows": sink.rows})
    finally:
        connection.close()
    for name, candidate in exact_candidates.items():
        candidate.replace(sinks[name].path)

    table_receipts: dict[str, Any] = {}
    for name, sink in sinks.items():
        path = sink.path
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(TABLE_SPECS[name][0], check_metadata=True):
            raise SecRestatementBuildError(f"Exact output schema differs for {name}")
        table_receipts[name] = {
            "rows": sink.rows,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "sort_verified": True,
            "primary_key_unique": True,
            "sort_engine": "duckdb_disk_backed_external",
        }
    source_receipts = {name: vars(receipt).copy() for name, receipt in receipts.items()}
    result = {
        "schema_version": 1,
        "status": "sec_restatement_family_built_in_supplied_staging",
        "scope": {"last_public_availability_date": SCOPE_END.isoformat()},
        "date_first_gate": {
            "excluded_payload_rows_decoded": 0,
            "minimal_boundary_context_is_separate_receipt_only": True,
            "reader": "pyarrow_csv_streaming_selected_text_columns",
            "semantic_payload_conversion": "retained_rows_only_after_date_and_form_gate",
            "csv_block_size_bytes": CSV_BLOCK_SIZE_BYTES,
        },
        "tables": table_receipts,
        "sources": source_receipts,
        "linkage": {
            "ticker_join_used": False, "company_name_join_used": False,
            "normalized_cusip8_only_or_date_valid_company_bridge": True,
            "matched": link_status_counts["MATCHED"],
            "ambiguous": link_status_counts["AMBIGUOUS"],
            "unmatched": link_status_counts["UNMATCHED"],
        },
        "amendments": {
            "retroactive_overwrite_used": False,
            "filing_versions_retained": len(summaries),
        },
        "restatement_affected_period_intervals": {
            "reversed_pairs_set_null": reversed_affected_period_intervals_set_null,
            "reversed_pairs_swapped": 0,
            "affected_period_dates_used_as_availability": False,
        },
        "profile_limited": max_source_records_per_file is not None,
        "authorizations": {
            "features_labels_models_portfolios_pnl_performance_or_backtests_authorized": False,
            "validation_or_final_test_access_authorized": False,
        },
    }
    work_context.cleanup()
    progress("sec_restatement_family_complete", {"tables": len(table_receipts)})
    return result
