"""Independent raw scanner for the Phase 2 CRSP development profile.

This module is intentionally standalone.  In particular, it does not import
the primary profile builder, compiler, or either publication module.  Its
grouping algorithm also does not trust source ordering:

1. each raw file is read byte-for-byte for an independent SHA-256 and stable
   operational-stat receipt;
2. a date-first, selected-column CSV pass writes only retained typed keys to
   private fixed-width ``PERMNO >> 8`` partitions and counts retained domains;
3. every private partition is sorted unconditionally, after which only the
   discovered duplicate keys are used for a sparse all-36-field replay.

Only aggregate counts and final multiplicity digests leave the scanner.  Raw
keys, duplicate-key lookups, and decoded duplicate rows remain inside a
temporary directory/lifetime and are discarded on success and failure.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import stat
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv


FINGERPRINT_DOMAIN = b"portfolio-bot/crsp-key-multiplicity/v1\0"
BYTE_CHUNK_SIZE = 8 << 20
_MIN_ARROW_BLOCK_SIZE = 4096

_DAILY_DTYPE = np.dtype(
    [("permno", "<i8"), ("calendar", "<i4")], align=False
)
_MONTHLY_DTYPE = np.dtype(
    [
        ("permno", "<i8"),
        ("calendar", "<i4"),
        ("interval_start", "<i4"),
        ("interval_end", "<i4"),
    ],
    align=False,
)
_CANONICAL_DTYPE = _DAILY_DTYPE
_UNIX_EPOCH = dt.date(1970, 1, 1)


class IndependentScanError(RuntimeError):
    """Raised when the independent raw scan cannot produce trustworthy evidence."""


def _lexical_absolute(path: Path | str) -> Path:
    """Return an absolute lexical path without dereferencing symlinks."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _assert_no_symlink_components(path: Path, label: str) -> None:
    """Reject an existing symlink at any component without revealing its path."""

    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError:
            raise IndependentScanError(f"Cannot authenticate {label} path components") from None
        if stat.S_ISLNK(mode):
            raise IndependentScanError(f"Symlink is prohibited in {label} path")


def _operational_stat(path: Path) -> dict[str, int]:
    try:
        value = path.lstat()
    except OSError:
        raise IndependentScanError(f"Cannot stat authenticated raw source {path.name}") from None
    if not stat.S_ISREG(value.st_mode):
        raise IndependentScanError(f"Authenticated raw source is not a regular file: {path.name}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "mtime_ns": int(value.st_mtime_ns),
        "size_bytes": int(value.st_size),
    }


def _hash_with_stable_stat(path: Path) -> tuple[str, dict[str, int], dict[str, int]]:
    _assert_no_symlink_components(path, "raw source")
    before = _operational_stat(path)
    digest = hashlib.sha256()
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        opened_stat = {
            "device": int(opened.st_dev),
            "inode": int(opened.st_ino),
            "mode": int(opened.st_mode),
            "mtime_ns": int(opened.st_mtime_ns),
            "size_bytes": int(opened.st_size),
        }
        if opened_stat != before or not stat.S_ISREG(opened.st_mode):
            os.close(descriptor)
            raise IndependentScanError(
                f"Authenticated raw source identity changed before byte hash: {path.name}"
            )
        with os.fdopen(descriptor, "rb") as handle:
            while chunk := handle.read(BYTE_CHUNK_SIZE):
                digest.update(chunk)
            closed = os.fstat(handle.fileno())
            closed_stat = {
                "device": int(closed.st_dev),
                "inode": int(closed.st_ino),
                "mode": int(closed.st_mode),
                "mtime_ns": int(closed.st_mtime_ns),
                "size_bytes": int(closed.st_size),
            }
            if closed_stat != before:
                raise IndependentScanError(
                    f"Authenticated raw source changed during byte hash: {path.name}"
                )
    except IndependentScanError:
        raise
    except OSError:
        raise IndependentScanError(f"Cannot hash authenticated raw source {path.name}") from None
    _assert_no_symlink_components(path, "raw source")
    after = _operational_stat(path)
    if before != after:
        raise IndependentScanError(
            f"Authenticated raw source changed during byte hash: {path.name}"
        )
    return digest.hexdigest(), before, after


def _require_unchanged(path: Path, authenticated: Mapping[str, int], label: str) -> dict[str, int]:
    _assert_no_symlink_components(path, "raw source")
    current = _operational_stat(path)
    if current != dict(authenticated):
        raise IndependentScanError(f"Authenticated raw source changed {label}: {path.name}")
    return current


def _strict_date32(values: pa.Array, *, label: str) -> pa.Array:
    """Parse strict Gregorian YYYY-MM-DD without exposing invalid raw values."""

    lexical = pc.match_substring_regex(values, r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    parsed = pc.strptime(values, format="%Y-%m-%d", unit="s", error_is_null=True)
    round_trip = pc.strftime(parsed, format="%Y-%m-%d")
    valid = pc.and_(
        pc.and_(lexical, pc.is_valid(parsed)),
        pc.and_(
            pc.fill_null(pc.equal(round_trip, values), False),
            pc.greater_equal(values, "0001-01-01"),
        ),
    )
    valid_numpy = np.asarray(valid.to_numpy(zero_copy_only=False), dtype=bool)
    if not bool(valid_numpy.all()):
        blank = int(pc.sum(pc.cast(pc.equal(values, ""), pa.int64())).as_py() or 0)
        invalid = len(values) - int(valid_numpy.sum()) - blank
        raise IndependentScanError(
            f"{label} contains blank or invalid dates (blank={blank}, invalid={invalid})"
        )
    return pc.cast(pc.if_else(valid, parsed, None), pa.date32())


def _optional_strict_date(values: pa.Array, *, label: str) -> int:
    blank_mask = pc.equal(values, "")
    blank = int(pc.sum(pc.cast(blank_mask, pa.int64())).as_py() or 0)
    if blank != len(values):
        _strict_date32(pc.filter(values, pc.invert(blank_mask)), label=label)
    return blank


def _positive_permno(values: pa.Array, *, label: str) -> pa.Array:
    lexical = np.asarray(
        pc.match_substring_regex(values, r"^[0-9]+$").to_numpy(zero_copy_only=False),
        dtype=bool,
    )
    if not bool(lexical.all()):
        blank = int(pc.sum(pc.cast(pc.equal(values, ""), pa.int64())).as_py() or 0)
        invalid = len(values) - int(lexical.sum()) - blank
        raise IndependentScanError(
            f"{label} contains blank or invalid PERMNO values "
            f"(blank={blank}, invalid={invalid})"
        )
    try:
        parsed = pc.cast(values, pa.int64(), safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        raise IndependentScanError(f"{label} contains PERMNO outside positive int64") from None
    nonpositive = int(pc.sum(pc.cast(pc.less_equal(parsed, 0), pa.int64())).as_py() or 0)
    if nonpositive:
        raise IndependentScanError(f"{label} contains {nonpositive} nonpositive PERMNO values")
    return parsed


def _date_days(values: pa.Array) -> np.ndarray:
    return (
        np.asarray(values.to_numpy(zero_copy_only=False))
        .astype("datetime64[D]")
        .astype(np.int32)
    )


def _int64(values: pa.Array) -> np.ndarray:
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.int64)


def _read_header(path: Path) -> list[str]:
    """Read just the lexical header; full data reads remain exactly three passes."""

    import csv

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return next(csv.reader(handle))
    except (OSError, StopIteration, UnicodeError, csv.Error):
        raise IndependentScanError(f"Cannot decode CSV header for {path.name}") from None


def _open_csv(
    path: Path,
    expected_header: Sequence[str],
    included_columns: Sequence[str],
    *,
    block_size_bytes: int,
) -> pacsv.CSVStreamingReader:
    if _read_header(path) != list(expected_header):
        raise IndependentScanError(f"CSV header differs from locked contract: {path.name}")
    included = list(dict.fromkeys(included_columns))
    if any(column not in expected_header for column in included):
        raise IndependentScanError(f"Requested scan column is absent from {path.name}")
    try:
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(
                block_size=max(int(block_size_bytes), _MIN_ARROW_BLOCK_SIZE),
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
                include_columns=included,
                column_types={column: pa.string() for column in included},
                strings_can_be_null=False,
                quoted_strings_can_be_null=False,
                null_values=[],
                check_utf8=True,
            ),
        )
    except (OSError, pa.ArrowException):
        raise IndependentScanError(f"Cannot open CSV stream for {path.name}") from None
    if reader.schema.names != included:
        raise IndependentScanError(f"Selected CSV schema differs for {path.name}")
    if any(not pa.types.is_string(field.type) for field in reader.schema):
        raise IndependentScanError(f"CSV fields were not decoded as strings: {path.name}")
    return reader


def _aggregate_only_batches(
    reader: pacsv.CSVStreamingReader, source_label: str
) -> Iterable[pa.RecordBatch]:
    """Yield batches while suppressing parser messages that may quote raw rows."""

    while True:
        try:
            yield reader.read_next_batch()
        except StopIteration:
            return
        except (OSError, UnicodeError, pa.ArrowException):
            raise IndependentScanError(
                f"CSV stream failed during aggregate-only {source_label} scan"
            ) from None


def _scope_retained(
    batch: pa.RecordBatch,
    date_column: str,
    *,
    start_date: str,
    end_date: str,
    source_label: str,
) -> tuple[pa.RecordBatch, pa.Array]:
    # The scope date is deliberately the first and only field evaluated before
    # filtering.  No other raw column is converted, counted, logged, or returned
    # until this gate has been applied.
    date_index = batch.schema.get_field_index(date_column)
    raw_dates = batch.column(date_index)
    parsed_dates = _strict_date32(raw_dates, label=f"{source_label} scope date")
    mask = pc.and_(
        pc.greater_equal(raw_dates, start_date), pc.less_equal(raw_dates, end_date)
    )
    return batch.filter(mask), pc.filter(parsed_dates, mask)


def _compact_keys(records: np.ndarray, fields: Sequence[str]) -> np.ndarray:
    dtype = np.dtype(
        [(field, records.dtype.fields[field][0]) for field in fields], align=False
    )
    compact = np.empty(len(records), dtype=dtype)
    for field in fields:
        compact[field] = records[field]
    return compact


def _sort_violation_count(
    records: np.ndarray,
    fields: Sequence[str],
    previous: tuple[int, ...] | None,
) -> tuple[int, tuple[int, ...] | None]:
    if len(records) == 0:
        return 0, previous
    violations = 0
    if previous is not None:
        first = tuple(int(records[field][0]) for field in fields)
        violations += int(first < previous)
    if len(records) > 1:
        prefix_equal = np.ones(len(records) - 1, dtype=bool)
        less = np.zeros(len(records) - 1, dtype=bool)
        for field in fields:
            left = records[field][:-1]
            right = records[field][1:]
            less |= prefix_equal & (right < left)
            prefix_equal &= right == left
        violations += int(less.sum())
    last = tuple(int(records[field][-1]) for field in fields)
    return violations, last


class _PackedPartitions:
    """Private fixed-width partitions whose names and contents never escape."""

    def __init__(self, root: Path, dtype: np.dtype) -> None:
        self.root = root
        self.dtype = dtype
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.paths: dict[int, Path] = {}

    def append(self, records: np.ndarray) -> None:
        if records.dtype != self.dtype:
            raise IndependentScanError("Private packed-record dtype differs")
        if len(records) == 0:
            return
        partitions = np.right_shift(records["permno"], 8)
        order = np.argsort(partitions, kind="stable")
        ordered_partitions = partitions[order]
        starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(ordered_partitions[1:] != ordered_partitions[:-1]) + 1)
        )
        ends = np.concatenate((starts[1:], np.array([len(order)], dtype=np.int64)))
        for start, end in zip(starts.tolist(), ends.tolist()):
            partition = int(ordered_partitions[start])
            path = self.paths.get(partition)
            if path is None:
                path = self.root / f"p-{partition:020d}.bin"
                self.paths[partition] = path
            existed = path.exists()
            try:
                with path.open("ab") as handle:
                    handle.write(np.ascontiguousarray(records[order[start:end]]).tobytes())
                if not existed:
                    path.chmod(0o600)
            except OSError:
                raise IndependentScanError("Cannot write private packed partition") from None

    def sorted_records(self) -> Iterable[np.ndarray]:
        fields = list(self.dtype.names or ())
        for partition in sorted(self.paths):
            path = self.paths[partition]
            try:
                size = path.stat().st_size
                if size % self.dtype.itemsize:
                    raise IndependentScanError("Private packed partition is truncated")
                records = np.fromfile(path, dtype=self.dtype).copy()
            except OSError:
                raise IndependentScanError("Cannot read private packed partition") from None
            # Sorting is unconditional, even if the input happened to be ordered
            # and even for a singleton partition.  Grouping therefore cannot
            # inherit an ordering assumption from CRSP.
            records.sort(order=fields, kind="stable")
            yield records


@dataclass
class _MultiplicityAggregate:
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(FINGERPRINT_DOMAIN)
        self.histogram: Counter[int] = Counter()
        self.distinct_groups = 0
        self.duplicate_groups = 0
        self.affected_rows = 0
        self.surplus_rows = 0
        self.maximum_group_size = 0
        self._duplicate_chunks: list[np.ndarray] = []
        self._duplicate_size_chunks: list[np.ndarray] = []

    def _update_fingerprint(self, keys: np.ndarray, sizes: np.ndarray) -> None:
        chunk_size = 100_000
        for offset in range(0, len(keys), chunk_size):
            selected = keys[offset : offset + chunk_size]
            selected_sizes = sizes[offset : offset + chunk_size]
            arguments: list[pa.Array | pa.Scalar] = [pa.scalar("[")]
            for index, field in enumerate(self.fields):
                if index:
                    arguments.append(pa.scalar(","))
                if field == "permno":
                    rendered = pc.cast(pa.array(selected[field], type=pa.int64()), pa.string())
                    arguments.append(rendered)
                else:
                    dates = pa.array(selected[field], type=pa.date32())
                    rendered = pc.cast(dates, pa.string())
                    arguments.extend((pa.scalar('"'), rendered, pa.scalar('"')))
            arguments.extend(
                (
                    pa.scalar(","),
                    pc.cast(pa.array(selected_sizes, type=pa.int64()), pa.string()),
                    pa.scalar("]\n"),
                    pa.scalar(""),
                )
            )
            payloads = pc.binary_join_element_wise(*arguments)
            self._digest.update("".join(payloads.to_pylist()).encode("utf-8"))

    def consume(self, sorted_records: np.ndarray) -> None:
        if len(sorted_records) == 0:
            return
        same = np.ones(max(0, len(sorted_records) - 1), dtype=bool)
        for field in self.fields:
            same &= sorted_records[field][1:] == sorted_records[field][:-1]
        starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(~same) + 1)
        )
        ends = np.concatenate(
            (starts[1:], np.array([len(sorted_records)], dtype=np.int64))
        )
        sizes = ends - starts
        keys = _compact_keys(sorted_records[starts], self.fields)
        self._update_fingerprint(keys, sizes)
        unique_sizes, counts = np.unique(sizes, return_counts=True)
        for size, count in zip(unique_sizes.tolist(), counts.tolist()):
            self.histogram[int(size)] += int(count)
        self.distinct_groups += len(sizes)
        self.maximum_group_size = max(self.maximum_group_size, int(sizes.max()))
        duplicate = sizes > 1
        duplicate_sizes = sizes[duplicate]
        self.duplicate_groups += int(duplicate.sum())
        self.affected_rows += int(duplicate_sizes.sum())
        self.surplus_rows += int((duplicate_sizes - 1).sum())
        if bool(duplicate.any()):
            self._duplicate_chunks.append(keys[duplicate].copy())
            self._duplicate_size_chunks.append(duplicate_sizes.astype(np.int64, copy=True))

    def finish(self) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        dtype = np.dtype(
            [(field, _MONTHLY_DTYPE.fields[field][0] if field in _MONTHLY_DTYPE.fields else _DAILY_DTYPE.fields[field][0]) for field in self.fields],
            align=False,
        )
        duplicates = (
            np.concatenate(self._duplicate_chunks)
            if self._duplicate_chunks
            else np.empty(0, dtype=dtype)
        )
        duplicate_sizes = (
            np.concatenate(self._duplicate_size_chunks)
            if self._duplicate_size_chunks
            else np.empty(0, dtype=np.int64)
        )
        if len(duplicates) != len(duplicate_sizes):
            raise IndependentScanError("Private duplicate multiplicity vectors differ")
        metrics = {
            "distinct_groups": int(self.distinct_groups),
            "duplicate_groups": int(self.duplicate_groups),
            "affected_rows": int(self.affected_rows),
            "surplus_rows": int(self.surplus_rows),
            "maximum_group_size": int(self.maximum_group_size),
            "multiplicity_fingerprint_sha256": self._digest.hexdigest(),
            "histogram": dict(sorted(self.histogram.items())),
        }
        return metrics, duplicates, duplicate_sizes


def _count_categoricals(
    batch: pa.RecordBatch,
    columns: Sequence[str],
    counters: Mapping[str, Counter[str]],
) -> None:
    for column in columns:
        counts = pc.value_counts(batch.column(batch.schema.get_field_index(column)))
        values = counts.field("values").to_pylist()
        frequencies = counts.field("counts").to_pylist()
        counters[column].update(dict(zip(values, frequencies)))


def _daily_key_pass(
    path: Path,
    config: Mapping[str, Any],
    partitions: _PackedPartitions,
    *,
    block_size_bytes: int,
) -> dict[str, Any]:
    source = config["sources"]["daily"]
    scope = config["scope"]
    categoricals = list(config["profiled_categorical_columns"]["daily"])
    required = [
        source["date_column"],
        "PERMNO",
        "SecInfoStartDt",
        "SecInfoEndDt",
        *categoricals,
    ]
    reader = _open_csv(path, source["header"], required, block_size_bytes=block_size_bytes)
    counters = {column: Counter() for column in categoricals}
    full_rows = retained_rows = inverted_rows = sort_violations = 0
    previous: tuple[int, ...] | None = None
    for batch in _aggregate_only_batches(reader, "daily retained-key"):
        full_rows += len(batch)
        retained, calendar = _scope_retained(
            batch,
            source["date_column"],
            start_date=scope["start_date"],
            end_date=scope["end_date"],
            source_label="daily",
        )
        if len(retained) == 0:
            continue
        permno = _positive_permno(
            retained.column(retained.schema.get_field_index("PERMNO")),
            label="daily retained rows",
        )
        interval_start = _strict_date32(
            retained.column(retained.schema.get_field_index("SecInfoStartDt")),
            label="daily interval start",
        )
        interval_end = _strict_date32(
            retained.column(retained.schema.get_field_index("SecInfoEndDt")),
            label="daily interval end",
        )
        inverted = int(
            pc.sum(pc.cast(pc.greater(interval_start, interval_end), pa.int64())).as_py()
            or 0
        )
        inverted_rows += inverted
        if inverted:
            raise IndependentScanError(
                f"Daily retained rows contain {inverted} inverted intervals"
            )
        records = np.empty(len(retained), dtype=_DAILY_DTYPE)
        records["permno"] = _int64(permno)
        records["calendar"] = _date_days(calendar)
        violations, previous = _sort_violation_count(
            records, ("permno", "calendar"), previous
        )
        sort_violations += violations
        partitions.append(records)
        _count_categoricals(retained, categoricals, counters)
        retained_rows += len(retained)
    return {
        "header": list(source["header"]),
        "full_file_rows_scanned": int(full_rows),
        "retained_rows": int(retained_rows),
        "sort_violations": int(sort_violations),
        "categorical_counts": {
            column: dict(
                sorted(counter.items(), key=lambda item: item[0].encode("utf-8"))
            )
            for column, counter in counters.items()
        },
        "additional_diagnostics": {"inverted_interval_rows": int(inverted_rows)},
    }


def _monthly_key_pass(
    path: Path,
    config: Mapping[str, Any],
    partitions: _PackedPartitions,
    *,
    block_size_bytes: int,
) -> dict[str, Any]:
    source = config["sources"]["monthly"]
    scope = config["scope"]
    categoricals = list(config["profiled_categorical_columns"]["monthly"])
    required = [
        source["date_column"],
        "PERMNO",
        source["interval_start_column"],
        source["interval_end_column"],
        source["price_date_column"],
        *categoricals,
    ]
    reader = _open_csv(path, source["header"], required, block_size_bytes=block_size_bytes)
    counters = {column: Counter() for column in categoricals}
    full_rows = retained_rows = inverted_rows = missing_price = 0
    raw_sort_violations = canonical_sort_violations = 0
    previous_raw: tuple[int, ...] | None = None
    previous_canonical: tuple[int, ...] | None = None
    for batch in _aggregate_only_batches(reader, "monthly retained-key"):
        full_rows += len(batch)
        retained, calendar = _scope_retained(
            batch,
            source["date_column"],
            start_date=scope["start_date"],
            end_date=scope["end_date"],
            source_label="monthly",
        )
        if len(retained) == 0:
            continue
        permno = _positive_permno(
            retained.column(retained.schema.get_field_index("PERMNO")),
            label="monthly retained rows",
        )
        interval_start = _strict_date32(
            retained.column(retained.schema.get_field_index(source["interval_start_column"])),
            label="monthly interval start",
        )
        interval_end = _strict_date32(
            retained.column(retained.schema.get_field_index(source["interval_end_column"])),
            label="monthly interval end",
        )
        inverted = int(
            pc.sum(pc.cast(pc.greater(interval_start, interval_end), pa.int64())).as_py()
            or 0
        )
        inverted_rows += inverted
        if inverted:
            raise IndependentScanError(
                f"Monthly retained rows contain {inverted} inverted intervals"
            )
        missing_price += _optional_strict_date(
            retained.column(retained.schema.get_field_index(source["price_date_column"])),
            label="monthly price date",
        )
        records = np.empty(len(retained), dtype=_MONTHLY_DTYPE)
        records["permno"] = _int64(permno)
        records["calendar"] = _date_days(calendar)
        records["interval_start"] = _date_days(interval_start)
        records["interval_end"] = _date_days(interval_end)
        violations, previous_raw = _sort_violation_count(
            records,
            ("permno", "calendar", "interval_start", "interval_end"),
            previous_raw,
        )
        raw_sort_violations += violations
        violations, previous_canonical = _sort_violation_count(
            records, ("permno", "calendar"), previous_canonical
        )
        canonical_sort_violations += violations
        partitions.append(records)
        _count_categoricals(retained, categoricals, counters)
        retained_rows += len(retained)
    return {
        "header": list(source["header"]),
        "full_file_rows_scanned": int(full_rows),
        "retained_rows": int(retained_rows),
        "raw_sort_violations": int(raw_sort_violations),
        "canonical_sort_violations": int(canonical_sort_violations),
        "categorical_counts": {
            column: dict(
                sorted(counter.items(), key=lambda item: item[0].encode("utf-8"))
            )
            for column, counter in counters.items()
        },
        "additional_diagnostics": {
            "inverted_interval_rows": int(inverted_rows),
            "missing_price_date_rows_overall": int(missing_price),
        },
    }


def _summarize_daily_partitions(
    partitions: _PackedPartitions,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    aggregate = _MultiplicityAggregate(("permno", "calendar"))
    for records in partitions.sorted_records():
        aggregate.consume(records)
    return aggregate.finish()


def _summarize_monthly_partitions(
    partitions: _PackedPartitions,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    raw = _MultiplicityAggregate(
        ("permno", "calendar", "interval_start", "interval_end")
    )
    pre = _MultiplicityAggregate(("permno", "calendar"))
    post = _MultiplicityAggregate(("permno", "calendar"))
    for records in partitions.sorted_records():
        raw.consume(records)
        pre.consume(records)
        if len(records):
            raw_same = np.ones(len(records) - 1, dtype=bool)
            for field in ("permno", "calendar", "interval_start", "interval_end"):
                raw_same &= records[field][1:] == records[field][:-1]
            raw_starts = np.concatenate(
                (np.array([0], dtype=np.int64), np.flatnonzero(~raw_same) + 1)
            )
            post.consume(records[raw_starts])
    raw_metrics, raw_duplicates, raw_duplicate_sizes = raw.finish()
    pre_metrics, pre_duplicates, pre_duplicate_sizes = pre.finish()
    post_metrics, post_duplicates, post_duplicate_sizes = post.finish()
    return (
        raw_metrics,
        raw_duplicates,
        raw_duplicate_sizes,
        pre_metrics,
        pre_duplicates,
        pre_duplicate_sizes,
        post_metrics,
        post_duplicates,
        post_duplicate_sizes,
    )


def _membership_mask(records: np.ndarray, duplicates: np.ndarray) -> np.ndarray:
    if len(duplicates) == 0:
        return np.zeros(len(records), dtype=bool)
    return np.isin(records, duplicates, assume_unique=False)


def _private_multiplicity_lookup(
    keys: np.ndarray, sizes: np.ndarray, fields: Sequence[str]
) -> dict[tuple[int, ...], int]:
    if len(keys) != len(sizes):
        raise IndependentScanError("Private duplicate-key and multiplicity counts differ")
    result = {
        tuple(int(key[field]) for field in fields): int(size)
        for key, size in zip(keys, sizes)
    }
    if len(result) != len(keys) or any(size <= 1 for size in result.values()):
        raise IndependentScanError("Private duplicate multiplicity lookup is invalid")
    return result


def _rows_from_selection(
    batch: pa.RecordBatch, selected_indexes: np.ndarray
) -> list[tuple[str, ...]]:
    if not len(selected_indexes):
        return []
    selected = batch.take(pa.array(selected_indexes, type=pa.int64()))
    columns = [column.to_pylist() for column in selected.columns]
    return [
        tuple(column[position] for column in columns)
        for position in range(len(selected_indexes))
    ]


def _replay_daily_duplicates(
    path: Path,
    config: Mapping[str, Any],
    duplicate_keys: np.ndarray,
    duplicate_sizes: np.ndarray,
    *,
    block_size_bytes: int,
) -> dict[str, Any]:
    source = config["sources"]["daily"]
    scope = config["scope"]
    header = list(source["header"])
    reader = _open_csv(path, header, header, block_size_bytes=block_size_bytes)
    groups: dict[tuple[int, int], list[tuple[str, ...]]] = defaultdict(list)
    full_rows = retained_rows = 0
    for batch in _aggregate_only_batches(reader, "daily sparse replay"):
        full_rows += len(batch)
        retained, calendar = _scope_retained(
            batch,
            source["date_column"],
            start_date=scope["start_date"],
            end_date=scope["end_date"],
            source_label="daily sparse replay",
        )
        if len(retained) == 0:
            continue
        permno = _positive_permno(
            retained.column(retained.schema.get_field_index("PERMNO")),
            label="daily sparse replay retained rows",
        )
        interval_start = _strict_date32(
            retained.column(retained.schema.get_field_index("SecInfoStartDt")),
            label="daily sparse replay interval start",
        )
        interval_end = _strict_date32(
            retained.column(retained.schema.get_field_index("SecInfoEndDt")),
            label="daily sparse replay interval end",
        )
        if bool(np.asarray(pc.greater(interval_start, interval_end)).any()):
            raise IndependentScanError("Daily sparse replay contains inverted intervals")
        records = np.empty(len(retained), dtype=_DAILY_DTYPE)
        records["permno"] = _int64(permno)
        records["calendar"] = _date_days(calendar)
        selected_indexes = np.flatnonzero(_membership_mask(records, duplicate_keys))
        rows = _rows_from_selection(retained, selected_indexes)
        for position, row in zip(selected_indexes.tolist(), rows):
            key = (int(records["permno"][position]), int(records["calendar"][position]))
            groups[key].append(row)
        retained_rows += len(retained)
    expected_multiplicities = _private_multiplicity_lookup(
        duplicate_keys, duplicate_sizes, ("permno", "calendar")
    )
    observed_multiplicities = {key: len(rows) for key, rows in groups.items()}
    if observed_multiplicities != expected_multiplicities:
        raise IndependentScanError(
            "Daily sparse duplicate replay multiplicities differ from retained-key pass"
        )
    exact = sum(len(set(rows)) == 1 for rows in groups.values())
    return {
        "full_file_rows_read": int(full_rows),
        "retained_rows": int(retained_rows),
        "all_field_exact_groups": int(exact),
        "conflicting_groups": int(len(groups) - exact),
    }


def _utf8_row_order(row: tuple[str, ...]) -> tuple[bytes, ...]:
    return tuple(value.encode("utf-8") for value in row)


def _one_or_values(values: set[str]) -> Any:
    ordered = sorted(values, key=lambda value: value.encode("utf-8"))
    return ordered[0] if len(ordered) == 1 else ordered


def _analyze_monthly_groups(
    groups: Mapping[tuple[int, int], Sequence[tuple[str, ...]]],
    header: Sequence[str],
    expected_pattern: Mapping[str, Any],
) -> dict[str, Any]:
    indexes = {column: index for index, column in enumerate(header)}
    raw_exact = raw_conflicting = pre_exact = embedded_exact = candidates = 0
    difference_counts = {column: 0 for column in header}
    candidate_sizes: Counter[int] = Counter()
    match_counts = {
        "groups_with_exactly_one_price_date_match": 0,
        "groups_with_zero_price_date_matches": 0,
        "groups_with_multiple_price_date_matches": 0,
        "groups_with_exactly_one_calendar_date_match": 0,
        "groups_with_zero_calendar_date_matches": 0,
        "groups_with_multiple_calendar_date_matches": 0,
    }
    missing_inside = different_selected = observed_pattern = 0
    selected_conditional: set[str] = set()
    selected_status: set[str] = set()
    rejected_conditional: set[str] = set()
    rejected_status: set[str] = set()

    for rows in groups.values():
        if len(set(rows)) == 1:
            pre_exact += 1
        raw_groups: dict[tuple[int, int, int, int], list[tuple[str, ...]]] = defaultdict(list)
        for row in rows:
            raw_key = (
                int(row[indexes["PERMNO"]]),
                dt.date.fromisoformat(row[indexes["MthCalDt"]]).toordinal(),
                dt.date.fromisoformat(row[indexes["SecInfoStartDt"]]).toordinal(),
                dt.date.fromisoformat(row[indexes["SecInfoEndDt"]]).toordinal(),
            )
            raw_groups[raw_key].append(row)
        for raw_rows in raw_groups.values():
            if len(raw_rows) > 1:
                if len(set(raw_rows)) == 1:
                    raw_exact += 1
                else:
                    raw_conflicting += 1
        collapsed = [min(raw_rows, key=_utf8_row_order) for raw_rows in raw_groups.values()]
        collapsed.sort(
            key=lambda row: (
                int(row[indexes["PERMNO"]]),
                row[indexes["MthCalDt"]],
                row[indexes["SecInfoStartDt"]],
                row[indexes["SecInfoEndDt"]],
                _utf8_row_order(row),
            )
        )
        if len(collapsed) <= 1:
            continue
        candidates += 1
        embedded_exact += sum(
            len(raw_rows) > 1 and len(set(raw_rows)) == 1
            for raw_rows in raw_groups.values()
        )
        candidate_sizes[len(collapsed)] += 1
        for column, index in indexes.items():
            if len({row[index] for row in collapsed}) > 1:
                difference_counts[column] += 1

        price_matches: list[int] = []
        calendar_matches: list[int] = []
        for position, row in enumerate(collapsed):
            start = dt.date.fromisoformat(row[indexes["SecInfoStartDt"]])
            end = dt.date.fromisoformat(row[indexes["SecInfoEndDt"]])
            calendar = dt.date.fromisoformat(row[indexes["MthCalDt"]])
            price_text = row[indexes["MthPrcDt"]]
            if price_text == "":
                missing_inside += 1
            elif start <= dt.date.fromisoformat(price_text) <= end:
                price_matches.append(position)
            if start <= calendar <= end:
                calendar_matches.append(position)
        price_key = (
            "groups_with_exactly_one_price_date_match"
            if len(price_matches) == 1
            else "groups_with_zero_price_date_matches"
            if not price_matches
            else "groups_with_multiple_price_date_matches"
        )
        calendar_key = (
            "groups_with_exactly_one_calendar_date_match"
            if len(calendar_matches) == 1
            else "groups_with_zero_calendar_date_matches"
            if not calendar_matches
            else "groups_with_multiple_calendar_date_matches"
        )
        match_counts[price_key] += 1
        match_counts[calendar_key] += 1
        if len(price_matches) != 1 or len(calendar_matches) != 1:
            continue
        if price_matches[0] != calendar_matches[0]:
            different_selected += 1
            continue
        selected_index = price_matches[0]
        selected = collapsed[selected_index]
        rejected = [row for index, row in enumerate(collapsed) if index != selected_index]
        selected_conditional.add(selected[indexes["ConditionalType"]])
        selected_status.add(selected[indexes["TradingStatusFlg"]])
        rejected_conditional.update(row[indexes["ConditionalType"]] for row in rejected)
        rejected_status.update(row[indexes["TradingStatusFlg"]] for row in rejected)
        if (
            selected[indexes["ConditionalType"]]
            == expected_pattern["selected_conditional_type"]
            and selected[indexes["TradingStatusFlg"]]
            == expected_pattern["selected_trading_status"]
            and all(
                row[indexes["ConditionalType"]]
                == expected_pattern["rejected_conditional_type"]
                and row[indexes["TradingStatusFlg"]]
                == expected_pattern["rejected_trading_status"]
                for row in rejected
            )
        ):
            observed_pattern += 1

    return {
        "raw_all_field_exact_groups": int(raw_exact),
        "raw_conflicting_groups": int(raw_conflicting),
        "pre_all_field_exact_groups": int(pre_exact),
        "pre_effective_interval_candidate_groups": int(candidates),
        "raw_exact_duplicate_groups_embedded_in_effective_interval_candidates": int(embedded_exact),
        "post_candidate_groups": int(candidates),
        "candidate_size_histogram": dict(sorted(candidate_sizes.items())),
        "candidate_group_size": (
            next(iter(candidate_sizes))
            if len(candidate_sizes) == 1
            else sorted(candidate_sizes)
        ),
        **match_counts,
        "missing_price_date_rows_inside_candidates": int(missing_inside),
        "price_and_calendar_selected_different_rows": int(different_selected),
        "selected_conditional_type": _one_or_values(selected_conditional),
        "selected_trading_status": _one_or_values(selected_status),
        "rejected_conditional_type": _one_or_values(rejected_conditional),
        "rejected_trading_status": _one_or_values(rejected_status),
        "groups_with_observed_pattern": int(observed_pattern),
        "column_difference_groups": difference_counts,
    }


def _replay_monthly_duplicates(
    path: Path,
    config: Mapping[str, Any],
    duplicate_canonical_keys: np.ndarray,
    duplicate_canonical_sizes: np.ndarray,
    duplicate_raw_keys: np.ndarray,
    duplicate_raw_sizes: np.ndarray,
    duplicate_post_keys: np.ndarray,
    duplicate_post_sizes: np.ndarray,
    *,
    block_size_bytes: int,
) -> dict[str, Any]:
    source = config["sources"]["monthly"]
    scope = config["scope"]
    header = list(source["header"])
    reader = _open_csv(path, header, header, block_size_bytes=block_size_bytes)
    groups: dict[tuple[int, int], list[tuple[str, ...]]] = defaultdict(list)
    full_rows = retained_rows = 0
    for batch in _aggregate_only_batches(reader, "monthly sparse replay"):
        full_rows += len(batch)
        retained, calendar = _scope_retained(
            batch,
            source["date_column"],
            start_date=scope["start_date"],
            end_date=scope["end_date"],
            source_label="monthly sparse replay",
        )
        if len(retained) == 0:
            continue
        permno = _positive_permno(
            retained.column(retained.schema.get_field_index("PERMNO")),
            label="monthly sparse replay retained rows",
        )
        interval_start = _strict_date32(
            retained.column(retained.schema.get_field_index(source["interval_start_column"])),
            label="monthly sparse replay interval start",
        )
        interval_end = _strict_date32(
            retained.column(retained.schema.get_field_index(source["interval_end_column"])),
            label="monthly sparse replay interval end",
        )
        if bool(np.asarray(pc.greater(interval_start, interval_end)).any()):
            raise IndependentScanError("Monthly sparse replay contains inverted intervals")
        _optional_strict_date(
            retained.column(retained.schema.get_field_index(source["price_date_column"])),
            label="monthly sparse replay price date",
        )
        records = np.empty(len(retained), dtype=_CANONICAL_DTYPE)
        records["permno"] = _int64(permno)
        records["calendar"] = _date_days(calendar)
        selected_indexes = np.flatnonzero(
            _membership_mask(records, duplicate_canonical_keys)
        )
        rows = _rows_from_selection(retained, selected_indexes)
        for position, row in zip(selected_indexes.tolist(), rows):
            key = (int(records["permno"][position]), int(records["calendar"][position]))
            groups[key].append(row)
        retained_rows += len(retained)
    expected_canonical = _private_multiplicity_lookup(
        duplicate_canonical_keys,
        duplicate_canonical_sizes,
        ("permno", "calendar"),
    )
    observed_canonical = {key: len(rows) for key, rows in groups.items()}
    if observed_canonical != expected_canonical:
        raise IndependentScanError(
            "Monthly sparse canonical replay multiplicities differ from retained-key pass"
        )

    expected_raw = _private_multiplicity_lookup(
        duplicate_raw_keys,
        duplicate_raw_sizes,
        ("permno", "calendar", "interval_start", "interval_end"),
    )
    observed_raw_all: Counter[tuple[int, int, int, int]] = Counter()
    indexes = {column: index for index, column in enumerate(header)}
    for rows in groups.values():
        for row in rows:
            observed_raw_all[
                (
                    int(row[indexes["PERMNO"]]),
                    (dt.date.fromisoformat(row[indexes["MthCalDt"]]) - _UNIX_EPOCH).days,
                    (
                        dt.date.fromisoformat(row[indexes["SecInfoStartDt"]])
                        - _UNIX_EPOCH
                    ).days,
                    (
                        dt.date.fromisoformat(row[indexes["SecInfoEndDt"]])
                        - _UNIX_EPOCH
                    ).days,
                )
            ] += 1
    observed_raw = {
        key: int(observed_raw_all.get(key, 0)) for key in expected_raw
    }
    if observed_raw != expected_raw:
        raise IndependentScanError(
            "Monthly sparse raw-key replay multiplicities differ from retained-key pass"
        )

    expected_post = _private_multiplicity_lookup(
        duplicate_post_keys,
        duplicate_post_sizes,
        ("permno", "calendar"),
    )
    observed_post: dict[tuple[int, int], int] = {}
    for key in expected_post:
        raw_keys = {
            (
                int(row[indexes["PERMNO"]]),
                row[indexes["MthCalDt"]],
                row[indexes["SecInfoStartDt"]],
                row[indexes["SecInfoEndDt"]],
            )
            for row in groups.get(key, ())
        }
        observed_post[key] = len(raw_keys)
    if observed_post != expected_post:
        raise IndependentScanError(
            "Monthly sparse post-collapse replay multiplicities differ from retained-key pass"
        )
    diagnostics = _analyze_monthly_groups(
        groups, header, config["expected"]["monthly_candidate_diagnostics"]
    )
    return {
        "full_file_rows_read": int(full_rows),
        "retained_rows": int(retained_rows),
        "diagnostics": diagnostics,
    }


def _flag_rows(
    config: Mapping[str, Any], daily: Mapping[str, Any], monthly: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source_name, scan, stage in (
        ("daily", daily, "daily_retained_raw_rows"),
        ("monthly", monthly, "monthly_retained_raw_rows"),
    ):
        hard_domains = config["expected"]["flag_domains"][source_name]
        for column in config["profiled_categorical_columns"][source_name]:
            policy = "hard_gate" if column in hard_domains else "discovery"
            expected = set(hard_domains.get(column, []))
            values = scan["categorical_counts"][column]
            for value in sorted(
                values, key=lambda item: (item != "", item.encode("utf-8"))
            ):
                classification = (
                    "blank"
                    if value == ""
                    else "discovered_nonblank"
                    if policy == "discovery"
                    else "expected_nonblank"
                    if value in expected
                    else "unexpected_nonblank"
                )
                result.append(
                    {
                        "source": config["sources"][source_name]["filename"],
                        "stage": stage,
                        "column": column,
                        "raw_value": value,
                        "row_count": int(values[value]),
                        "is_blank": value == "",
                        "domain_policy": policy,
                        "classification": classification,
                    }
                )
    return result


def _public_stage(
    metrics: Mapping[str, Any],
    *,
    source: str,
    stage: str,
    key_fields: Sequence[str],
    input_rows: int,
    sort_violations: int,
    all_field_exact_groups: int,
    conflicting_groups: int,
    effective_interval_candidate_groups: int,
) -> dict[str, Any]:
    return {
        "source": source,
        "stage": stage,
        "key_fields": list(key_fields),
        "input_rows": int(input_rows),
        "distinct_groups": int(metrics["distinct_groups"]),
        "duplicate_groups": int(metrics["duplicate_groups"]),
        "affected_rows": int(metrics["affected_rows"]),
        "surplus_rows": int(metrics["surplus_rows"]),
        "all_field_exact_groups": int(all_field_exact_groups),
        "conflicting_groups": int(conflicting_groups),
        "effective_interval_candidate_groups": int(effective_interval_candidate_groups),
        "output_rows": int(metrics["distinct_groups"]),
        "sort_violations": int(sort_violations),
        "maximum_group_size": int(metrics["maximum_group_size"]),
        "multiplicity_fingerprint_sha256": metrics["multiplicity_fingerprint_sha256"],
        "histogram": dict(metrics["histogram"]),
    }


def _normalize_raw_paths(raw_paths: Mapping[str, Path | str]) -> dict[str, Path]:
    if set(raw_paths) != {"daily", "monthly"}:
        raise IndependentScanError("raw_paths must contain exactly daily and monthly")
    result = {
        name: _lexical_absolute(raw_paths[name]) for name in ("daily", "monthly")
    }
    for path in result.values():
        _assert_no_symlink_components(path, "raw source")
    return result


def scan_crsp_profile_independently(
    config: Mapping[str, Any],
    raw_paths: Mapping[str, Path | str],
    *,
    scratch_root: Path | str | None = None,
    block_size_bytes: int = 16 << 20,
) -> dict[str, Any]:
    """Perform the independent three-read scan and return aggregate-only evidence."""

    paths = _normalize_raw_paths(raw_paths)
    if block_size_bytes <= 0:
        raise IndependentScanError("block_size_bytes must be positive")
    hashes: dict[str, str] = {}
    stats: dict[str, dict[str, dict[str, int]]] = {}
    for source_name, path in paths.items():
        digest, before, after = _hash_with_stable_stat(path)
        hashes[source_name] = digest
        stats[source_name] = {"before_hash": before, "after_hash": after}

    scratch_parent = _lexical_absolute(scratch_root) if scratch_root else None
    if scratch_parent is not None:
        _assert_no_symlink_components(scratch_parent, "scratch root")
        try:
            scratch_parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise IndependentScanError("Cannot prepare private scratch root") from None
        _assert_no_symlink_components(scratch_parent, "scratch root")
        try:
            scratch_mode = scratch_parent.lstat().st_mode
        except OSError:
            raise IndependentScanError("Cannot authenticate private scratch root") from None
        if not stat.S_ISDIR(scratch_mode):
            raise IndependentScanError("Private scratch root is not a directory")
    with tempfile.TemporaryDirectory(
        prefix=".crsp-profile-independent-", dir=scratch_parent
    ) as temporary_name:
        private_root = Path(temporary_name)
        private_root.chmod(0o700)
        daily_partitions = _PackedPartitions(private_root / "d", _DAILY_DTYPE)
        monthly_partitions = _PackedPartitions(private_root / "m", _MONTHLY_DTYPE)

        daily = _daily_key_pass(
            paths["daily"], config, daily_partitions, block_size_bytes=block_size_bytes
        )
        stats["daily"]["after_key_pass"] = _require_unchanged(
            paths["daily"], stats["daily"]["after_hash"], "during retained-key pass"
        )
        monthly = _monthly_key_pass(
            paths["monthly"], config, monthly_partitions, block_size_bytes=block_size_bytes
        )
        stats["monthly"]["after_key_pass"] = _require_unchanged(
            paths["monthly"], stats["monthly"]["after_hash"], "during retained-key pass"
        )

        (
            daily_metrics,
            daily_duplicate_keys,
            daily_duplicate_sizes,
        ) = _summarize_daily_partitions(
            daily_partitions
        )
        (
            monthly_raw_metrics,
            monthly_raw_duplicate_keys,
            monthly_raw_duplicate_sizes,
            monthly_pre_metrics,
            monthly_pre_duplicate_keys,
            monthly_pre_duplicate_sizes,
            monthly_post_metrics,
            monthly_post_duplicate_keys,
            monthly_post_duplicate_sizes,
        ) = _summarize_monthly_partitions(monthly_partitions)

        daily_replay = _replay_daily_duplicates(
            paths["daily"],
            config,
            daily_duplicate_keys,
            daily_duplicate_sizes,
            block_size_bytes=block_size_bytes,
        )
        stats["daily"]["after_sparse_replay"] = _require_unchanged(
            paths["daily"], stats["daily"]["after_hash"], "during sparse replay"
        )
        monthly_replay = _replay_monthly_duplicates(
            paths["monthly"],
            config,
            monthly_pre_duplicate_keys,
            monthly_pre_duplicate_sizes,
            monthly_raw_duplicate_keys,
            monthly_raw_duplicate_sizes,
            monthly_post_duplicate_keys,
            monthly_post_duplicate_sizes,
            block_size_bytes=block_size_bytes,
        )
        stats["monthly"]["after_sparse_replay"] = _require_unchanged(
            paths["monthly"], stats["monthly"]["after_hash"], "during sparse replay"
        )

        # Authenticate both inputs once more after *both* replay streams have
        # closed.  This final barrier is stronger than checking each source
        # only when its own replay ends: neither file can change while the
        # other source is still being processed without being detected.
        for source_name in ("daily", "monthly"):
            stats[source_name]["final"] = _require_unchanged(
                paths[source_name],
                stats[source_name]["after_hash"],
                "before final evidence release",
            )

        if (
            daily_replay["full_file_rows_read"] != daily["full_file_rows_scanned"]
            or daily_replay["retained_rows"] != daily["retained_rows"]
            or monthly_replay["full_file_rows_read"] != monthly["full_file_rows_scanned"]
            or monthly_replay["retained_rows"] != monthly["retained_rows"]
        ):
            raise IndependentScanError("Sparse replay row counts differ from retained-key pass")
        if daily_replay["all_field_exact_groups"] + daily_replay["conflicting_groups"] != daily_metrics["duplicate_groups"]:
            raise IndependentScanError("Daily exact/conflicting duplicate conservation failed")

        candidate = monthly_replay["diagnostics"]
        if candidate["raw_all_field_exact_groups"] + candidate["raw_conflicting_groups"] != monthly_raw_metrics["duplicate_groups"]:
            raise IndependentScanError("Monthly exact/conflicting raw duplicate conservation failed")
        if candidate["post_candidate_groups"] != monthly_post_metrics["duplicate_groups"]:
            raise IndependentScanError("Monthly post-collapse candidate reconstruction differs")

        daily["filename"] = config["sources"]["daily"]["filename"]
        daily["source"] = "daily"
        daily["stage"] = _public_stage(
            daily_metrics,
            source=daily["filename"],
            stage="daily_raw_key",
            key_fields=config["sources"]["daily"]["raw_key"],
            input_rows=daily["retained_rows"],
            sort_violations=daily["sort_violations"],
            all_field_exact_groups=daily_replay["all_field_exact_groups"],
            conflicting_groups=daily_replay["conflicting_groups"],
            effective_interval_candidate_groups=0,
        )
        daily["scope_gate"] = {
            "date_column": config["sources"]["daily"]["date_column"],
            "start_date": config["scope"]["start_date"],
            "end_date": config["scope"]["end_date"],
            "excluded_rows": daily["full_file_rows_scanned"] - daily["retained_rows"],
            "excluded_non_date_values_compared_or_aggregated": 0,
        }
        daily.pop("sort_violations")

        monthly["filename"] = config["sources"]["monthly"]["filename"]
        monthly["source"] = "monthly"
        monthly["collapsed_rows"] = int(monthly_raw_metrics["distinct_groups"])
        monthly["stages"] = {
            "monthly_raw_key": _public_stage(
                monthly_raw_metrics,
                source=monthly["filename"],
                stage="monthly_raw_key",
                key_fields=config["sources"]["monthly"]["raw_key"],
                input_rows=monthly["retained_rows"],
                sort_violations=monthly["raw_sort_violations"],
                all_field_exact_groups=candidate["raw_all_field_exact_groups"],
                conflicting_groups=candidate["raw_conflicting_groups"],
                effective_interval_candidate_groups=0,
            ),
            "monthly_canonical_key_before_raw_exact_collapse": _public_stage(
                monthly_pre_metrics,
                source=monthly["filename"],
                stage="monthly_canonical_key_before_raw_exact_collapse",
                key_fields=config["sources"]["monthly"]["canonical_key"],
                input_rows=monthly["retained_rows"],
                sort_violations=monthly["canonical_sort_violations"],
                all_field_exact_groups=candidate["pre_all_field_exact_groups"],
                conflicting_groups=0,
                effective_interval_candidate_groups=candidate["pre_effective_interval_candidate_groups"],
            ),
            "monthly_canonical_key_after_raw_exact_collapse": _public_stage(
                monthly_post_metrics,
                source=monthly["filename"],
                stage="monthly_canonical_key_after_raw_exact_collapse",
                key_fields=config["sources"]["monthly"]["canonical_key"],
                input_rows=monthly_raw_metrics["distinct_groups"],
                sort_violations=0,
                all_field_exact_groups=0,
                conflicting_groups=0,
                effective_interval_candidate_groups=candidate["post_candidate_groups"],
            ),
        }
        monthly["candidate_diagnostics"] = {
            key: value
            for key, value in candidate.items()
            if key
            not in {
                "raw_all_field_exact_groups",
                "raw_conflicting_groups",
                "pre_all_field_exact_groups",
            }
        }
        monthly["scope_gate"] = {
            "date_column": config["sources"]["monthly"]["date_column"],
            "start_date": config["scope"]["start_date"],
            "end_date": config["scope"]["end_date"],
            "excluded_rows": monthly["full_file_rows_scanned"] - monthly["retained_rows"],
            "excluded_non_date_values_compared_or_aggregated": 0,
        }
        monthly.pop("raw_sort_violations")
        monthly.pop("canonical_sort_violations")

        flags = _flag_rows(config, daily, monthly)
        stage_map = {"daily_raw_key": daily["stage"], **monthly["stages"]}
        stage_order = config["outputs"]["population_registry"]["duplicate_stage_order"]
        summary_fields = config["outputs"]["schemas"]["duplicate_summary"]["columns"]
        duplicate_summary = [
            {field: stage_map[stage][field] for field in summary_fields}
            for stage in stage_order
        ]
        duplicate_multiplicity = [
            {
                "source": stage_map[stage]["source"],
                "stage": stage,
                "multiplicity": int(multiplicity),
                "group_count": int(group_count),
                "affected_rows": int(multiplicity) * int(group_count),
            }
            for stage in stage_order
            for multiplicity, group_count in sorted(stage_map[stage]["histogram"].items())
            if int(group_count) > 0
        ]
        expected_differences = config["expected"]["monthly_candidate_column_difference_groups"]
        duplicate_column_differences = [
            {
                "source": config["sources"]["monthly"]["filename"],
                "stage": "monthly_canonical_key_after_raw_exact_collapse",
                "column": column,
                "groups_differing": int(candidate["column_difference_groups"][column]),
                "expected_groups_differing": int(expected_differences.get(column, 0)),
                "status": (
                    "PASS"
                    if int(candidate["column_difference_groups"][column])
                    == int(expected_differences.get(column, 0))
                    else "FAIL"
                ),
            }
            for column in config["sources"]["monthly"]["header"]
        ]

        # Delete or lose every last reference capable of revealing a row-level
        # key before constructing the public return object.
        del (
            daily_duplicate_keys,
            daily_duplicate_sizes,
            monthly_raw_duplicate_keys,
            monthly_raw_duplicate_sizes,
            monthly_pre_duplicate_keys,
            monthly_pre_duplicate_sizes,
            monthly_post_duplicate_keys,
            monthly_post_duplicate_sizes,
        )

    # TemporaryDirectory has completed here, including exceptional cleanup.
    return {
        "raw_hashes": hashes,
        "operational_stats": stats,
        "daily": daily,
        "monthly": monthly,
        "duplicate_summary": duplicate_summary,
        "duplicate_multiplicity": duplicate_multiplicity,
        "duplicate_column_differences": duplicate_column_differences,
        "flag_rows": flags,
        "independence": {
            "full_raw_hashes_recomputed": True,
            "grouping_depends_on_input_sort": False,
            "primary_builder_imported": False,
            "raw_files_reread": 2,
        },
        "temporary_files_remaining": 0,
    }


__all__ = [
    "IndependentScanError",
    "scan_crsp_profile_independently",
]
