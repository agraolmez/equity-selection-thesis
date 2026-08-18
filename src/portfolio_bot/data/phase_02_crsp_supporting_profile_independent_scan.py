"""Independent aggregate scanner for the five CRSP supporting sources.

This module intentionally imports neither the primary compiler nor its
publisher.  Arrow is used only as a CSV decoder and date-first row gate;
global duplicate classification is performed separately with SQLite.  Raw
security keys exist only in memory or private scratch storage and never appear
in the returned evidence.
"""

from __future__ import annotations

import bisect
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import sqlite3
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.csv as pacsv


SOURCE_ORDER = (
    "crsp_names_ciz.csv",
    "crsp_shares_outstanding_ciz.csv",
    "crsp_distribution_ciz.csv",
    "crsp_delisting_ciz.csv",
    "crsp_stock_header_ciz.csv",
)


class IndependentSupportingScanError(RuntimeError):
    """Raised when independent raw evidence cannot be scanned exactly."""


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _assert_no_symlink_components(path: Path, label: str) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise IndependentSupportingScanError(
                f"Symlink is prohibited in {label} path: {current}"
            )


def _operational_stat(path: Path) -> dict[str, int]:
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise IndependentSupportingScanError(f"Raw source is not regular: {path}")
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(metadata.st_mode),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
    }


def _stable_hash(path: Path) -> tuple[str, dict[str, int], dict[str, int]]:
    _assert_no_symlink_components(path, "raw source")
    before = _operational_stat(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        descriptor_stat = os.fstat(descriptor)
        if (
            descriptor_stat.st_dev != before["device"]
            or descriptor_stat.st_ino != before["inode"]
            or not stat.S_ISREG(descriptor_stat.st_mode)
        ):
            raise IndependentSupportingScanError(
                f"Raw source identity changed before hashing: {path}"
            )
        while chunk := os.read(descriptor, 8 << 20):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    after = _operational_stat(path)
    if before != after:
        raise IndependentSupportingScanError(
            f"Raw source changed while hashing: {path}"
        )
    return digest.hexdigest(), before, after


def _strict_date(value: Any, label: str) -> dt.date:
    if (
        not isinstance(value, str)
        or len(value) != 10
        or value[4:5] != "-"
        or value[7:8] != "-"
    ):
        raise IndependentSupportingScanError(f"{label} is not strict YYYY-MM-DD")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise IndependentSupportingScanError(
            f"{label} is not a Gregorian date"
        ) from exc
    if parsed.isoformat() != value:
        raise IndependentSupportingScanError(f"{label} is not canonical")
    return parsed


def _positive_integer(value: Any, label: str) -> int:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise IndependentSupportingScanError(f"{label} is not ASCII decimal digits")
    parsed = int(value)
    if parsed <= 0 or parsed > 9_223_372_036_854_775_807:
        raise IndependentSupportingScanError(
            f"{label} is nonpositive or exceeds signed int64"
        )
    return parsed


def _header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return next(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise IndependentSupportingScanError(
            f"Cannot decode raw header {path.name}: {exc}"
        ) from exc


def _open_reader(
    path: Path,
    header: Sequence[str],
    include: Sequence[str],
    block_size_bytes: int,
) -> pa.RecordBatchReader:
    try:
        return pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(
                use_threads=False,
                block_size=block_size_bytes,
                skip_rows=1,
                column_names=list(header),
            ),
            parse_options=pacsv.ParseOptions(
                delimiter=",",
                quote_char='"',
                double_quote=True,
                newlines_in_values=True,
            ),
            convert_options=pacsv.ConvertOptions(
                column_types={column: pa.string() for column in header},
                include_columns=list(include),
                strings_can_be_null=False,
                quoted_strings_can_be_null=False,
                check_utf8=True,
            ),
        )
    except (pa.ArrowException, OSError) as exc:
        raise IndependentSupportingScanError(
            f"Cannot open locked CSV {path.name}: {exc}"
        ) from exc


def _safe_scratch_name(filename: str) -> str:
    if Path(filename).name != filename or filename not in SOURCE_ORDER:
        raise IndependentSupportingScanError(f"Unsafe source name: {filename!r}")
    return filename.replace(".", "_") + ".sqlite"


class _DuplicateLedger:
    """Order-independent key and full-row multiplicity ledger."""

    def __init__(self, path: Path, *, compare_full_rows: bool) -> None:
        self.compare_full_rows = compare_full_rows
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TABLE ledger (key_blob BLOB NOT NULL,row_blob BLOB NOT NULL,"
            "n INTEGER NOT NULL,PRIMARY KEY(key_blob,row_blob)) WITHOUT ROWID"
        )

    def add(self, rows: Sequence[tuple[tuple[Any, ...], tuple[str, ...]]]) -> None:
        values = []
        for key, row in rows:
            key_blob = json.dumps(
                key, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            row_blob = (
                json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                if self.compare_full_rows
                else b""
            )
            values.append((key_blob, row_blob, 1))
        self.connection.executemany(
            "INSERT INTO ledger(key_blob,row_blob,n) VALUES(?,?,?) "
            "ON CONFLICT(key_blob,row_blob) DO UPDATE SET n=n+1",
            values,
        )

    def finish(self) -> dict[str, Any]:
        self.connection.commit()
        histogram: Counter[int] = Counter()
        groups = duplicates = surplus = exact = conflicts = 0
        cursor = self.connection.execute(
            "SELECT SUM(n),COUNT(*) FROM ledger GROUP BY key_blob"
        )
        for total_value, variants_value in cursor:
            total = int(total_value)
            variants = int(variants_value)
            groups += 1
            histogram[total] += 1
            duplicates += int(total > 1)
            surplus += total - 1
            exact += int(self.compare_full_rows and total > 1 and variants == 1)
            conflicts += int(self.compare_full_rows and variants > 1)
        self.connection.close()
        return {
            "distinct_keys": groups,
            "duplicate_key_groups": duplicates,
            "duplicate_surplus_rows": surplus,
            "exact_duplicate_groups": exact,
            "conflicting_duplicate_groups": conflicts,
            "multiplicity": dict(sorted(histogram.items())),
        }

    def close(self) -> None:
        self.connection.close()


@dataclasses.dataclass
class _Context:
    calendar_sorted: tuple[dt.date, ...]
    names_intervals: dict[int, list[tuple[dt.date, dt.date]]] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )
    names_lifetimes: dict[int, set[tuple[dt.date | None, dt.date | None]]] = (
        dataclasses.field(default_factory=lambda: defaultdict(set))
    )
    header_permnos: set[int] = dataclasses.field(default_factory=set)
    referential: Counter[str] = dataclasses.field(default_factory=Counter)


def _scope_rows(
    batch: pa.RecordBatch,
    source: Mapping[str, Any],
    start: dt.date,
    end: dt.date,
) -> tuple[list[int], dict[str, list[dt.date]]]:
    kind = source["scope_kind"]
    if kind == "qa_key_only":
        return list(range(batch.num_rows)), {}
    if kind == "interval_overlap":
        gate_columns = [source["interval_start"], source["interval_end"]]
    else:
        gate_columns = list(source["scope_date_columns"])
    parsed: dict[str, list[dt.date]] = {}
    for column in gate_columns:
        values = batch.column(batch.schema.get_field_index(column)).to_pylist()
        parsed[column] = [
            _strict_date(value, f"{source['filename']} {column}") for value in values
        ]
    if kind == "interval_overlap":
        lefts, rights = parsed[gate_columns[0]], parsed[gate_columns[1]]
        if any(left > right for left, right in zip(lefts, rights, strict=True)):
            raise IndependentSupportingScanError(
                f"{source['filename']} contains an inverted scope interval"
            )
        keep = [
            index
            for index, (left, right) in enumerate(zip(lefts, rights, strict=True))
            if right >= start and left <= end
        ]
    elif kind == "event_date":
        keep = [
            index
            for index, value in enumerate(parsed[gate_columns[0]])
            if start <= value <= end
        ]
    elif kind == "event_or_posting_date":
        keep = [
            index
            for index, pair in enumerate(
                zip(parsed[gate_columns[0]], parsed[gate_columns[1]], strict=True)
            )
            if any(start <= value <= end for value in pair)
        ]
    else:
        raise IndependentSupportingScanError(f"Unknown scope kind: {kind!r}")
    return keep, parsed


def _optional_date(
    raw: str, filename: str, column: str, metrics: Counter[str]
) -> dt.date | None:
    if raw == "":
        metrics[f"blank_{column}"] += 1
        return None
    return _strict_date(raw, f"{filename} retained {column}")


def _overlap_count(
    intervals: Sequence[tuple[dt.date, dt.date]], left: dt.date, right: dt.date
) -> int:
    return sum(start <= right and finish >= left for start, finish in intervals)


def _point_count(intervals: Sequence[tuple[dt.date, dt.date]], value: dt.date) -> int:
    return sum(start <= value <= finish for start, finish in intervals)


def _consume_retained(
    source: Mapping[str, Any],
    values: Mapping[str, str],
    permno: int,
    scope_dates: Mapping[str, dt.date],
    start: dt.date,
    end: dt.date,
    categorical: Mapping[str, Counter[str]],
    date_metrics: Counter[str],
    numeric_metrics: Counter[str],
    intervals: dict[int, list[tuple[dt.date, dt.date]]],
    context: _Context,
    calendar_dates: set[dt.date],
) -> None:
    filename = source["filename"]
    for column in source.get("categorical_columns", []):
        categorical[column][values[column]] += 1
    optional = {
        column: _optional_date(values[column], filename, column, date_metrics)
        for column in source.get("optional_date_columns", [])
    }
    for column in source.get("required_date_columns", []):
        if values[column] == "":
            date_metrics[f"blank_{column}"] += 1
        _strict_date(values[column], f"{filename} retained {column}")
    for column in source.get("positive_integer_columns", []):
        raw = values[column]
        if raw == "":
            numeric_metrics[f"blank_{column}"] += 1
        if not raw or not raw.isascii() or not raw.isdecimal():
            numeric_metrics[f"invalid_{column}"] += 1
            raise IndependentSupportingScanError(
                f"{filename} retained {column} is invalid"
            )
        if int(raw) <= 0:
            numeric_metrics[f"nonpositive_{column}"] += 1
            raise IndependentSupportingScanError(
                f"{filename} retained {column} is nonpositive"
            )

    kind = source["scope_kind"]
    if kind == "interval_overlap":
        left = scope_dates[source["interval_start"]]
        right = scope_dates[source["interval_end"]]
        intervals[permno].append((left, right))
        if filename == "crsp_names_ciz.csv":
            context.names_intervals[permno].append((left, right))
            try:
                _positive_integer(values["PERMCO"], "names retained PERMCO")
            except IndependentSupportingScanError:
                context.referential["names_invalid_or_nonpositive_permco_rows"] += 1
            life = (optional.get("SecurityBegDt"), optional.get("SecurityEndDt"))
            context.names_lifetimes[permno].add(life)
            life_start, life_end = life
            if life_start is not None and life_end is not None:
                date_metrics["security_life_inversions"] += int(life_start > life_end)
                date_metrics["information_interval_outside_security_life"] += int(
                    left < life_start or right > life_end
                )
        elif filename == "crsp_shares_outstanding_ciz.csv":
            count = _overlap_count(context.names_intervals.get(permno, ()), left, right)
            suffix = (
                "exactly_one_overlapping_names_interval"
                if count == 1
                else (
                    "zero_overlapping_names_intervals"
                    if count == 0
                    else "multiple_overlapping_names_intervals"
                )
            )
            context.referential[f"shares_rows_with_{suffix}"] += 1
    elif filename == "crsp_distribution_ciz.csv":
        event = scope_dates["DisExDt"]
        context.referential[
            (
                "distribution_rows_on_verified_trading_date"
                if event in calendar_dates
                else "distribution_rows_off_verified_trading_date"
            )
        ] += 1
        count = _point_count(context.names_intervals.get(permno, ()), event)
        qualifier = (
            "exactly_one_names_match"
            if count == 1
            else "zero_names_matches" if count == 0 else "multiple_names_matches"
        )
        context.referential[f"distribution_rows_with_{qualifier}_on_ex_date"] += 1
        declare, record, pay = (
            optional["DisDeclareDt"],
            optional["DisRecordDt"],
            optional["DisPayDt"],
        )
        date_metrics["declare_after_ex"] += int(declare is not None and declare > event)
        date_metrics["record_before_ex"] += int(record is not None and record < event)
        date_metrics["pay_before_ex"] += int(pay is not None and pay < event)
        date_metrics["declare_after_record"] += int(
            declare is not None and record is not None and declare > record
        )
        date_metrics["record_after_pay"] += int(
            record is not None and pay is not None and record > pay
        )
    elif filename == "crsp_delisting_ciz.csv":
        event, posting = scope_dates["DelistingDt"], scope_dates["DelDlyDt"]
        date_metrics["delisting_date_in_scope"] += int(start <= event <= end)
        date_metrics["posting_date_in_scope"] += int(start <= posting <= end)
        date_metrics["both_dates_in_scope"] += int(
            start <= event <= end and start <= posting <= end
        )
        date_metrics["posting_before_delisting"] += int(posting < event)
        date_metrics["next_before_delisting"] += int(
            optional["DelNextDt"] is not None and optional["DelNextDt"] < event
        )
        date_metrics["amount_before_delisting"] += int(
            optional["DelAmtDt"] is not None and optional["DelAmtDt"] < event
        )
        context.referential["delisting_rows_on_verified_trading_date"] += int(
            event in calendar_dates
        )
        context.referential["delisting_posting_rows_on_verified_trading_date"] += int(
            posting in calendar_dates
        )
        position = bisect.bisect_right(context.calendar_sorted, event)
        context.referential[
            "delisting_posting_equals_next_verified_trading_date"
        ] += int(
            position < len(context.calendar_sorted)
            and context.calendar_sorted[position] == posting
        )
        for when, suffix in ((event, "delisting_date"), (posting, "posting_date")):
            count = _point_count(context.names_intervals.get(permno, ()), when)
            qualifier = (
                "exactly_one_names_match"
                if count == 1
                else "zero_names_matches" if count == 0 else "multiple_names_matches"
            )
            context.referential[f"delisting_rows_with_{qualifier}_on_{suffix}"] += 1


def _interval_metrics(
    values: Mapping[int, Sequence[tuple[dt.date, dt.date]]],
    start: dt.date,
    end: dt.date,
) -> dict[str, int]:
    result: Counter[str] = Counter()
    for intervals in values.values():
        maximum: dt.date | None = None
        for left, right in sorted(intervals):
            result["starts_before_scope"] += int(left < start)
            result["ends_after_scope"] += int(right > end)
            result["spans_entire_scope"] += int(left < start and right > end)
            result["inverted_intervals"] += int(left > right)
            if maximum is not None:
                if left <= maximum:
                    result["overlap_edges"] += 1
                elif left == maximum + dt.timedelta(days=1):
                    result["contiguous_edges"] += 1
                else:
                    result["gap_edges"] += 1
            maximum = right if maximum is None else max(maximum, right)
    return dict(result)


def _scan_source(
    path: Path,
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    scratch_directory: Path,
    context: _Context,
    calendar_dates: set[dt.date],
    block_size_bytes: int,
) -> dict[str, Any]:
    expected_header = list(source["header"])
    if _header(path) != expected_header:
        raise IndependentSupportingScanError(
            f"{source['filename']} header differs from the locked config"
        )
    qa_only = source["scope_kind"] == "qa_key_only"
    include = ["PERMNO"] if qa_only else expected_header
    reader = _open_reader(path, expected_header, include, block_size_bytes)
    ledger = _DuplicateLedger(
        scratch_directory / _safe_scratch_name(source["filename"]),
        compare_full_rows=not qa_only,
    )
    start = dt.date.fromisoformat(config["scope"]["start_date"])
    end = dt.date.fromisoformat(config["scope"]["end_date"])
    total = retained_count = 0
    permnos: set[int] = set()
    categoricals = {
        column: Counter() for column in source.get("categorical_columns", [])
    }
    date_metrics: Counter[str] = Counter()
    numeric_metrics: Counter[str] = Counter()
    intervals: dict[int, list[tuple[dt.date, dt.date]]] = defaultdict(list)
    try:
        for batch in reader:
            total += batch.num_rows
            keep, scope_values = _scope_rows(batch, source, start, end)
            if not keep:
                continue
            retained = batch.take(pa.array(keep, type=pa.int64()))
            columns = retained.to_pydict()
            observed_rows = list(
                zip(*(columns[column] for column in include), strict=True)
            )
            ledger_rows: list[tuple[tuple[Any, ...], tuple[str, ...]]] = []
            for local_index, raw_row in enumerate(observed_rows):
                values = dict(zip(include, raw_row, strict=True))
                permno = _positive_integer(
                    values["PERMNO"], f"{source['filename']} retained PERMNO"
                )
                permnos.add(permno)
                if qa_only:
                    context.header_permnos.add(permno)
                    key = (permno,)
                else:
                    global_index = keep[local_index]
                    dates = {
                        column: parsed[global_index]
                        for column, parsed in scope_values.items()
                    }
                    key_values: list[Any] = []
                    for column in source["key"]:
                        if column == "PERMNO":
                            key_values.append(permno)
                        elif column in dates:
                            key_values.append(dates[column].isoformat())
                        elif column in source.get("positive_integer_columns", []):
                            key_values.append(
                                _positive_integer(
                                    values[column],
                                    f"{source['filename']} retained {column}",
                                )
                            )
                        else:
                            key_values.append(values[column])
                    key = tuple(key_values)
                    _consume_retained(
                        source,
                        values,
                        permno,
                        dates,
                        start,
                        end,
                        categoricals,
                        date_metrics,
                        numeric_metrics,
                        intervals,
                        context,
                        calendar_dates,
                    )
                ledger_rows.append((key, tuple(raw_row)))
            ledger.add(ledger_rows)
            retained_count += len(observed_rows)
    except BaseException:
        ledger.close()
        raise
    duplicates = ledger.finish()
    return {
        "source": source["filename"],
        "scope_kind": source["scope_kind"],
        "header_columns": len(expected_header),
        "key_fields": list(source["key"]),
        "full_file_rows": total,
        "retained_rows": retained_count,
        "excluded_rows": total - retained_count,
        "unique_permnos": len(permnos),
        "excluded_non_date_values_compared_or_aggregated": 0,
        "duplicates": duplicates,
        "categorical_counts": {
            column: dict(
                sorted(counter.items(), key=lambda item: item[0].encode("utf-8"))
            )
            for column, counter in categoricals.items()
        },
        "date_metrics": dict(date_metrics),
        "numeric_metrics": dict(numeric_metrics),
        "interval_metrics": _interval_metrics(intervals, start, end),
    }


def scan_supporting_sources_independently(
    config: Mapping[str, Any],
    raw_paths: Mapping[str, str | Path],
    *,
    calendar_dates: Sequence[dt.date],
    scratch_directory: Path,
    block_size_bytes: int = 16 << 20,
) -> dict[str, Any]:
    """Reread and independently aggregate every locked supporting source."""
    filenames = tuple(source["filename"] for source in config.get("sources", []))
    if filenames != SOURCE_ORDER or set(raw_paths) != set(SOURCE_ORDER):
        raise IndependentSupportingScanError(
            "Independent scanner requires the exact five-source closed set"
        )
    scratch_directory = _lexical_absolute(scratch_directory)
    _assert_no_symlink_components(scratch_directory, "scratch")
    scratch_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    _assert_no_symlink_components(scratch_directory, "scratch")
    calendar_set = set(calendar_dates)
    if len(calendar_set) != len(calendar_dates):
        raise IndependentSupportingScanError("Calendar dates contain duplicates")
    context = _Context(calendar_sorted=tuple(sorted(calendar_set)))
    results: list[dict[str, Any]] = []
    receipts: dict[str, dict[str, Any]] = {}
    raw_operational: dict[str, dict[str, dict[str, int]]] = {}
    source_by_name = {source["filename"]: source for source in config["sources"]}
    for filename in SOURCE_ORDER:
        path = _lexical_absolute(raw_paths[filename])
        digest, before, hash_after = _stable_hash(path)
        result = _scan_source(
            path,
            source_by_name[filename],
            config,
            scratch_directory,
            context,
            calendar_set,
            block_size_bytes,
        )
        after = _operational_stat(path)
        if before != hash_after or before != after:
            raise IndependentSupportingScanError(
                f"Raw source changed during independent scan: {filename}"
            )
        receipts[filename] = {
            "filename": filename,
            "sha256": digest,
            "size_bytes": before["size_bytes"],
            "full_file_rows_expected": int(source_by_name[filename]["full_file_rows"]),
            "full_file_rows_scanned": result["full_file_rows"],
        }
        raw_operational[filename] = {"before": before, "after": after}
        results.append(result)
        if filename == "crsp_names_ciz.csv":
            for intervals in context.names_intervals.values():
                intervals.sort()

    context.referential["names_nonconstant_security_lifetime_permnos"] = sum(
        len(values) > 1 for values in context.names_lifetimes.values()
    )
    context.referential["names_permnos_missing_current_header_key"] = sum(
        permno not in context.header_permnos for permno in context.names_intervals
    )
    context.referential["verified_calendar_dates"] = len(calendar_set)
    return {
        "independence": {
            "aggregate_only": True,
            "full_raw_hashes_recomputed": True,
            "grouping_depends_on_input_sort": False,
            "primary_compiler_imported": False,
            "primary_publisher_imported": False,
            "raw_files_reread": len(SOURCE_ORDER),
        },
        "receipts": receipts,
        "raw_operational": raw_operational,
        "referential": dict(context.referential),
        "sources": results,
        "summary": {
            "source_count": len(results),
            "full_file_rows_read": sum(row["full_file_rows"] for row in results),
            "retained_rows": sum(row["retained_rows"] for row in results),
        },
    }


__all__ = [
    "IndependentSupportingScanError",
    "SOURCE_ORDER",
    "scan_supporting_sources_independently",
]
