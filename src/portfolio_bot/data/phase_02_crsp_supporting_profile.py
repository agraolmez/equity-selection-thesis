"""Compile aggregate-only QA for the CRSP supporting development sources.

This module deliberately stops before canonical-table materialisation.  It
authenticates the frozen inputs, applies the locked 2000--2014 scope rule before
Python observes any excluded non-date field, and publishes only aggregate
quality evidence.  Security keys and row payloads are confined to an ephemeral
SQLite database used to classify duplicates globally (input ordering is never
treated as a grouping guarantee).
"""

from __future__ import annotations

import bisect
import csv
import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_supporting_sources_development_v1.toml"
)
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.toml"
DEFAULT_PHASE_CONFIG = PROJECT_ROOT / "configs" / "phase_02_canonical_panel.toml"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "phase_02_canonical_panel"
BYTE_CHUNK_SIZE = 8 << 20

SOURCE_ORDER = (
    "crsp_names_ciz.csv",
    "crsp_shares_outstanding_ciz.csv",
    "crsp_distribution_ciz.csv",
    "crsp_delisting_ciz.csv",
    "crsp_stock_header_ciz.csv",
)

CSV_SCHEMAS: dict[str, tuple[str, ...]] = {
    "categorical_domains.csv": (
        "source",
        "column",
        "raw_value",
        "row_count",
        "is_blank",
        "classification",
    ),
    "checks.csv": (
        "check_id",
        "source",
        "category",
        "metric",
        "expected",
        "actual",
        "status",
        "fatal",
    ),
    "date_diagnostics.csv": ("source", "metric", "expected", "actual", "status"),
    "duplicate_multiplicity.csv": ("source", "multiplicity", "key_group_count"),
    "duplicate_summary.csv": (
        "source",
        "key_fields",
        "retained_rows",
        "distinct_keys",
        "unique_permnos",
        "duplicate_key_groups",
        "duplicate_surplus_rows",
        "exact_duplicate_groups",
        "conflicting_duplicate_groups",
        "full_decoded_row_comparison",
    ),
    "interval_diagnostics.csv": (
        "source",
        "metric",
        "expected",
        "actual",
        "status",
    ),
    "numeric_validity.csv": ("source", "metric", "expected", "actual", "status"),
    "referential_integrity.csv": (
        "metric",
        "expected",
        "actual",
        "status",
        "fatal",
    ),
    "source_structure.csv": (
        "source",
        "scope_kind",
        "full_file_rows",
        "retained_rows",
        "excluded_rows",
        "header_columns",
        "key_fields",
        "size_bytes",
        "sha256",
        "excluded_non_date_values_compared_or_aggregated",
    ),
}


class SupportingProfileError(RuntimeError):
    """Raised when an authenticated input, profile gate, or renderer fails."""


@dataclasses.dataclass(frozen=True)
class SupportingProfileSettings:
    project_root: Path
    profile_config_path: Path
    paths_config_path: Path
    phase_config_path: Path
    config: dict[str, Any]
    database_root: Path
    raw_root: Path
    quality_root: Path
    artifact_root: Path
    run_root: Path
    block_size_bytes: int = 16 << 20


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _canonical_json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path, chunk_size: int = BYTE_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _project_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupportingProfileError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SupportingProfileError(f"Expected JSON object: {path}")
    return value


def load_settings(
    profile_config_path: Path = DEFAULT_PROFILE_CONFIG,
    paths_config_path: Path = DEFAULT_PATHS_CONFIG,
    phase_config_path: Path = DEFAULT_PHASE_CONFIG,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    run_root: Path = DEFAULT_RUN_ROOT,
    block_size_bytes: int = 16 << 20,
) -> SupportingProfileSettings:
    """Load paths and the locked supporting-profile contract without scanning."""
    profile_config_path = _lexical_absolute(profile_config_path)
    paths_config_path = _lexical_absolute(paths_config_path)
    phase_config_path = _lexical_absolute(phase_config_path)
    try:
        with profile_config_path.open("rb") as handle:
            config = tomllib.load(handle)
        with paths_config_path.open("rb") as handle:
            paths = tomllib.load(handle)
        with phase_config_path.open("rb") as handle:
            phase = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError, KeyError) as exc:
        raise SupportingProfileError(f"Cannot load supporting-profile settings: {exc}") from exc
    filenames = tuple(source.get("filename") for source in config.get("sources", []))
    if filenames != SOURCE_ORDER:
        raise SupportingProfileError(
            f"Supporting source order differs: expected {SOURCE_ORDER}, actual {filenames}"
        )
    configured_root = Path(paths["database"]["root"]).expanduser()
    database_root = _lexical_absolute(
        os.environ.get("PORTFOLIO_DATABASE_ROOT", configured_root)
    )
    namespace = phase["storage"]["quality_namespace"]
    return SupportingProfileSettings(
        project_root=PROJECT_ROOT,
        profile_config_path=profile_config_path,
        paths_config_path=paths_config_path,
        phase_config_path=phase_config_path,
        config=config,
        database_root=database_root,
        raw_root=_lexical_absolute(database_root / paths["database"]["raw"]),
        quality_root=_lexical_absolute(
            database_root / paths["database"]["quality"] / namespace
        ),
        artifact_root=_lexical_absolute(artifact_root),
        run_root=_lexical_absolute(run_root),
        block_size_bytes=int(block_size_bytes),
    )


def _operational_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "mode": int(stat.st_mode),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _strict_date(value: str, *, label: str, blank_allowed: bool = False) -> dt.date | None:
    if value == "" and blank_allowed:
        return None
    if len(value) != 10 or value[4:5] != "-" or value[7:8] != "-":
        raise SupportingProfileError(f"{label} contains a blank or invalid strict date")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise SupportingProfileError(f"{label} contains a blank or invalid strict date") from exc
    if parsed.isoformat() != value:
        raise SupportingProfileError(f"{label} contains a non-canonical strict date")
    return parsed


def _positive_int(value: str, *, label: str) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise SupportingProfileError(f"{label} is blank or is not ASCII decimal digits")
    parsed = int(value)
    if parsed <= 0 or parsed > 9_223_372_036_854_775_807:
        raise SupportingProfileError(f"{label} is nonpositive or exceeds signed int64")
    return parsed


def _wire(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return _canonical_json_text(value)
    return str(value)


def _csv_bytes(fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=list(fields),
        delimiter=",",
        quotechar='"',
        doublequote=True,
        escapechar=None,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _wire(row.get(field)) for field in fields})
    return handle.getvalue().encode("utf-8")


def _exact_header(path: Path, expected: Sequence[str]) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            actual = next(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise SupportingProfileError(f"Cannot decode header for {path.name}: {exc}") from exc
    if actual != list(expected):
        raise SupportingProfileError(
            f"{path.name} header differs: expected {list(expected)}, actual {actual}"
        )


class _GlobalDuplicateAccumulator:
    """Disk-backed exact row comparison with no input-order assumption."""

    def __init__(self, path: Path, *, full_rows: bool = True) -> None:
        self._full_rows = full_rows
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute("PRAGMA temp_store=FILE")
        self._connection.execute(
            "CREATE TABLE seen (key_blob BLOB NOT NULL, row_blob BLOB NOT NULL, "
            "multiplicity INTEGER NOT NULL, PRIMARY KEY(key_blob,row_blob)) WITHOUT ROWID"
        )

    def consume(self, values: Iterable[tuple[tuple[Any, ...], tuple[str, ...]]]) -> None:
        rows = []
        for key, row in values:
            key_blob = _canonical_json_text(key).encode("utf-8")
            row_blob = (
                _canonical_json_text(row).encode("utf-8") if self._full_rows else b""
            )
            rows.append((key_blob, row_blob, 1))
        self._connection.executemany(
            "INSERT INTO seen(key_blob,row_blob,multiplicity) VALUES(?,?,?) "
            "ON CONFLICT(key_blob,row_blob) DO UPDATE SET multiplicity=multiplicity+1",
            rows,
        )

    def finish(self) -> dict[str, Any]:
        try:
            self._connection.commit()
            grouped = list(
                self._connection.execute(
                    "SELECT SUM(multiplicity),COUNT(*) FROM seen GROUP BY key_blob"
                )
            )
            histogram = Counter(int(total) for total, _ in grouped)
            duplicate = sum(total > 1 for total, _ in grouped)
            surplus = sum(int(total) - 1 for total, _ in grouped)
            exact = sum(total > 1 and variants == 1 for total, variants in grouped)
            conflict = sum(variants > 1 for _, variants in grouped)
        finally:
            self._connection.close()
        return {
            "distinct_keys": len(grouped),
            "duplicate_key_groups": duplicate,
            "duplicate_surplus_rows": surplus,
            "exact_duplicate_groups": exact if self._full_rows else 0,
            "conflicting_duplicate_groups": conflict if self._full_rows else 0,
            "multiplicity": dict(sorted(histogram.items())),
        }

    def close(self) -> None:
        self._connection.close()


def _arrow_reader(
    path: Path,
    header: Sequence[str],
    *,
    block_size_bytes: int,
    include_columns: Sequence[str] | None = None,
) -> pa.RecordBatchReader:
    columns = list(include_columns or header)
    try:
        return pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(
                use_threads=True,
                # Arrow must fit the complete header in its first block.  Keep
                # tiny test fixtures useful without weakening production blocks.
                block_size=max(4096, block_size_bytes),
                skip_rows=1,
                column_names=list(header),
            ),
            parse_options=pacsv.ParseOptions(
                delimiter=",", quote_char='"', double_quote=True, newlines_in_values=True
            ),
            convert_options=pacsv.ConvertOptions(
                column_types={column: pa.string() for column in header},
                include_columns=columns,
                strings_can_be_null=False,
                quoted_strings_can_be_null=False,
                check_utf8=True,
            ),
        )
    except (pa.ArrowException, OSError) as exc:
        raise SupportingProfileError(f"Cannot open {path.name} as locked CSV: {exc}") from exc


def _scope_indices(
    batch: pa.RecordBatch,
    source: Mapping[str, Any],
    start: dt.date,
    end: dt.date,
) -> tuple[list[int], dict[str, list[dt.date]]]:
    """Parse only gate dates, globally, before returning retained row indices."""
    kind = source["scope_kind"]
    parsed: dict[str, list[dt.date]] = {}
    if kind == "interval_overlap":
        columns = (source["interval_start"], source["interval_end"])
    elif kind in {"event_date", "event_or_posting_date"}:
        columns = tuple(source["scope_date_columns"])
    elif kind == "qa_key_only":
        return list(range(len(batch))), parsed
    else:
        raise SupportingProfileError(f"Unknown scope kind {kind!r}")
    for column in columns:
        parsed[column] = [
            _strict_date(value, label=f"{source['filename']} {column}")  # type: ignore[arg-type]
            for value in batch.column(batch.schema.get_field_index(column)).to_pylist()
        ]
    if kind == "interval_overlap":
        starts, ends = parsed[columns[0]], parsed[columns[1]]
        inverted = sum(left > right for left, right in zip(starts, ends, strict=True))
        if inverted:
            raise SupportingProfileError(
                f"{source['filename']} contains {inverted} inverted interval(s)"
            )
        keep = [index for index, (left, right) in enumerate(zip(starts, ends, strict=True)) if right >= start and left <= end]
    elif kind == "event_date":
        dates = parsed[columns[0]]
        keep = [index for index, value in enumerate(dates) if start <= value <= end]
    else:
        first, second = parsed[columns[0]], parsed[columns[1]]
        keep = [
            index
            for index, (left, right) in enumerate(zip(first, second, strict=True))
            if start <= left <= end or start <= right <= end
        ]
    return keep, parsed


def _interval_match_count(
    intervals: Sequence[tuple[dt.date, dt.date]], start: dt.date, end: dt.date
) -> int:
    return sum(left <= end and right >= start for left, right in intervals)


def _point_match_count(
    intervals: Sequence[tuple[dt.date, dt.date]], value: dt.date
) -> int:
    return sum(left <= value <= right for left, right in intervals)


def _expected_for(config: Mapping[str, Any], filename: str) -> Mapping[str, Any]:
    return config["expected"][filename.replace(".", "_")]


def _source_by_name(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {source["filename"]: source for source in config["sources"]}


@dataclasses.dataclass
class _ScanContext:
    names_intervals: dict[int, list[tuple[dt.date, dt.date]]] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )
    names_permco: dict[int, set[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(set)
    )
    names_lifetime: dict[int, set[tuple[dt.date | None, dt.date | None]]] = dataclasses.field(
        default_factory=lambda: defaultdict(set)
    )
    header_permnos: set[int] = dataclasses.field(default_factory=set)
    referential: Counter[str] = dataclasses.field(default_factory=Counter)
    calendar_sorted: tuple[dt.date, ...] = ()


def _scan_one_source(
    path: Path,
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    scratch_directory: Path,
    context: _ScanContext,
    calendar_dates: set[dt.date],
    block_size_bytes: int,
) -> dict[str, Any]:
    """Scan one authenticated source and return aggregates only."""
    header = list(source["header"])
    _exact_header(path, header)
    qa_only = source["scope_kind"] == "qa_key_only"
    include = ["PERMNO"] if qa_only else header
    reader = _arrow_reader(
        path, header, block_size_bytes=block_size_bytes, include_columns=include
    )
    duplicate = _GlobalDuplicateAccumulator(
        scratch_directory / f"{source['filename']}.sqlite", full_rows=not qa_only
    )
    start = dt.date.fromisoformat(config["scope"]["start_date"])
    end = dt.date.fromisoformat(config["scope"]["end_date"])
    full_rows = retained_rows = 0
    permnos: set[int] = set()
    categorical = {column: Counter() for column in source.get("categorical_columns", [])}
    date_metrics: Counter[str] = Counter()
    numeric_metrics: Counter[str] = Counter()
    retained_intervals: dict[int, list[tuple[dt.date, dt.date]]] = defaultdict(list)

    try:
        for batch in reader:
            full_rows += len(batch)
            keep, parsed_scope = _scope_indices(batch, source, start, end)
            if not keep:
                continue
            retained = batch.take(pa.array(keep, type=pa.int64()))
            columns = {
                column: retained.column(retained.schema.get_field_index(column)).to_pylist()
                for column in include
            }
            rows = list(zip(*(columns[column] for column in include), strict=True))
            keyed: list[tuple[tuple[Any, ...], tuple[str, ...]]] = []
            for local, row in enumerate(rows):
                values = dict(zip(include, row, strict=True))
                permno = _positive_int(
                    values["PERMNO"], label=f"{source['filename']} retained PERMNO"
                )
                permnos.add(permno)
                if qa_only:
                    key = (permno,)
                    context.header_permnos.add(permno)
                else:
                    # Scope dates were parsed globally.  Reuse the retained subset.
                    global_index = keep[local]
                    date_values = {
                        column: parsed[global_index]
                        for column, parsed in parsed_scope.items()
                    }
                    key_parts: list[Any] = []
                    for column in source["key"]:
                        if column == "PERMNO":
                            key_parts.append(permno)
                        elif column in date_values:
                            key_parts.append(date_values[column].isoformat())
                        elif column in source.get("positive_integer_columns", []):
                            key_parts.append(
                                _positive_int(
                                    values[column],
                                    label=f"{source['filename']} retained {column}",
                                )
                            )
                        else:
                            key_parts.append(values[column])
                    key = tuple(key_parts)
                    _consume_retained_row(
                        source,
                        values,
                        permno,
                        date_values,
                        start,
                        end,
                        categorical,
                        date_metrics,
                        numeric_metrics,
                        retained_intervals,
                        context,
                        calendar_dates,
                    )
                keyed.append((key, tuple(row)))
            duplicate.consume(keyed)
            retained_rows += len(rows)
    except (pa.ArrowException, UnicodeDecodeError, csv.Error) as exc:
        duplicate.close()
        raise SupportingProfileError(f"Cannot scan {source['filename']}: {exc}") from exc
    except BaseException:
        duplicate.close()
        raise

    duplicate_metrics = duplicate.finish()
    interval_metrics = _finish_intervals(retained_intervals, start, end)
    result = {
        "source": source["filename"],
        "scope_kind": source["scope_kind"],
        "header_columns": len(header),
        "key_fields": list(source["key"]),
        "full_file_rows": full_rows,
        "retained_rows": retained_rows,
        "excluded_rows": full_rows - retained_rows,
        "unique_permnos": len(permnos),
        "excluded_non_date_values_compared_or_aggregated": 0,
        "duplicates": duplicate_metrics,
        "categorical_counts": {
            column: dict(sorted(counter.items(), key=lambda item: item[0].encode("utf-8")))
            for column, counter in categorical.items()
        },
        "date_metrics": dict(date_metrics),
        "numeric_metrics": dict(numeric_metrics),
        "interval_metrics": interval_metrics,
    }
    _gate_source_result(config, source, result)
    return result


def _consume_retained_row(
    source: Mapping[str, Any],
    values: Mapping[str, str],
    permno: int,
    scope_dates: Mapping[str, dt.date],
    scope_start: dt.date,
    scope_end: dt.date,
    categorical: Mapping[str, Counter[str]],
    date_metrics: Counter[str],
    numeric_metrics: Counter[str],
    retained_intervals: dict[int, list[tuple[dt.date, dt.date]]],
    context: _ScanContext,
    calendar_dates: set[dt.date],
) -> None:
    filename = source["filename"]
    for column in source.get("categorical_columns", []):
        categorical[column][values[column]] += 1
    parsed_optional: dict[str, dt.date | None] = {}
    for column in source.get("optional_date_columns", []):
        raw = values[column]
        if raw == "":
            date_metrics[f"blank_{column}"] += 1
            parsed_optional[column] = None
        else:
            parsed_optional[column] = _strict_date(
                raw, label=f"{filename} retained {column}", blank_allowed=True
            )
    for column in source.get("required_date_columns", []):
        if values[column] == "":
            date_metrics[f"blank_{column}"] += 1
        _strict_date(values[column], label=f"{filename} retained {column}")
    for column in source.get("positive_integer_columns", []):
        raw = values[column]
        if raw == "":
            numeric_metrics[f"blank_{column}"] += 1
        if not raw or not raw.isascii() or not raw.isdecimal():
            numeric_metrics[f"invalid_{column}"] += 1
            raise SupportingProfileError(f"{filename} retained {column} is invalid")
        if int(raw) <= 0:
            numeric_metrics[f"nonpositive_{column}"] += 1
            raise SupportingProfileError(f"{filename} retained {column} is nonpositive")

    if source["scope_kind"] == "interval_overlap":
        left = scope_dates[source["interval_start"]]
        right = scope_dates[source["interval_end"]]
        retained_intervals[permno].append((left, right))
        if filename == "crsp_names_ciz.csv":
            context.names_intervals[permno].append((left, right))
            try:
                permco = _positive_int(values["PERMCO"], label="names retained PERMCO")
            except SupportingProfileError:
                context.referential["names_invalid_or_nonpositive_permco_rows"] += 1
            else:
                context.names_permco[permno].add(permco)
            life = (parsed_optional.get("SecurityBegDt"), parsed_optional.get("SecurityEndDt"))
            context.names_lifetime[permno].add(life)
            beg, life_end = life
            if beg is not None and life_end is not None:
                if beg > life_end:
                    date_metrics["security_life_inversions"] += 1
                if left < beg or right > life_end:
                    date_metrics["information_interval_outside_security_life"] += 1
        elif filename == "crsp_shares_outstanding_ciz.csv":
            count = _interval_match_count(context.names_intervals.get(permno, ()), left, right)
            context.referential[
                "shares_rows_with_exactly_one_overlapping_names_interval"
                if count == 1
                else "shares_rows_with_zero_overlapping_names_intervals"
                if count == 0
                else "shares_rows_with_multiple_overlapping_names_intervals"
            ] += 1
    elif filename == "crsp_distribution_ciz.csv":
        ex = scope_dates["DisExDt"]
        date_metrics["declare_after_ex"] += int(
            parsed_optional["DisDeclareDt"] is not None
            and parsed_optional["DisDeclareDt"] > ex
        )
        date_metrics["record_before_ex"] += int(
            parsed_optional["DisRecordDt"] is not None
            and parsed_optional["DisRecordDt"] < ex
        )
        date_metrics["pay_before_ex"] += int(
            parsed_optional["DisPayDt"] is not None and parsed_optional["DisPayDt"] < ex
        )
        date_metrics["declare_after_record"] += int(
            parsed_optional["DisDeclareDt"] is not None
            and parsed_optional["DisRecordDt"] is not None
            and parsed_optional["DisDeclareDt"] > parsed_optional["DisRecordDt"]
        )
        date_metrics["record_after_pay"] += int(
            parsed_optional["DisRecordDt"] is not None
            and parsed_optional["DisPayDt"] is not None
            and parsed_optional["DisRecordDt"] > parsed_optional["DisPayDt"]
        )
        context.referential[
            "distribution_rows_on_verified_trading_date"
            if ex in calendar_dates
            else "distribution_rows_off_verified_trading_date"
        ] += 1
        count = _point_match_count(context.names_intervals.get(permno, ()), ex)
        context.referential[
            "distribution_rows_with_exactly_one_names_match_on_ex_date"
            if count == 1
            else "distribution_rows_with_zero_names_matches_on_ex_date"
            if count == 0
            else "distribution_rows_with_multiple_names_matches_on_ex_date"
        ] += 1
    elif filename == "crsp_delisting_ciz.csv":
        event = scope_dates["DelistingDt"]
        posting = scope_dates["DelDlyDt"]
        date_metrics["delisting_date_in_scope"] += int(scope_start <= event <= scope_end)
        date_metrics["posting_date_in_scope"] += int(scope_start <= posting <= scope_end)
        date_metrics["both_dates_in_scope"] += int(
            scope_start <= event <= scope_end and scope_start <= posting <= scope_end
        )
        date_metrics["posting_before_delisting"] += int(posting < event)
        date_metrics["next_before_delisting"] += int(
            parsed_optional["DelNextDt"] is not None
            and parsed_optional["DelNextDt"] < event
        )
        date_metrics["amount_before_delisting"] += int(
            parsed_optional["DelAmtDt"] is not None
            and parsed_optional["DelAmtDt"] < event
        )
        context.referential["delisting_rows_on_verified_trading_date"] += int(
            event in calendar_dates
        )
        context.referential["delisting_posting_rows_on_verified_trading_date"] += int(
            posting in calendar_dates
        )
        calendar = context.calendar_sorted
        position = bisect.bisect_right(calendar, event)
        context.referential[
            "delisting_posting_equals_next_verified_trading_date"
        ] += int(position < len(calendar) and calendar[position] == posting)
        for when, suffix in ((event, "delisting_date"), (posting, "posting_date")):
            count = _point_match_count(context.names_intervals.get(permno, ()), when)
            context.referential[
                f"delisting_rows_with_exactly_one_names_match_on_{suffix}"
                if count == 1
                else f"delisting_rows_with_zero_names_matches_on_{suffix}"
                if count == 0
                else f"delisting_rows_with_multiple_names_matches_on_{suffix}"
            ] += 1


def _finish_intervals(
    intervals: Mapping[int, Sequence[tuple[dt.date, dt.date]]],
    start: dt.date,
    end: dt.date,
) -> dict[str, int]:
    metrics: Counter[str] = Counter()
    for rows in intervals.values():
        ordered = sorted(rows)
        maximum_prior: dt.date | None = None
        for left, right in ordered:
            metrics["starts_before_scope"] += int(left < start)
            metrics["ends_after_scope"] += int(right > end)
            metrics["spans_entire_scope"] += int(left < start and right > end)
            if left > right:
                metrics["inverted_intervals"] += 1
            if maximum_prior is not None:
                if left <= maximum_prior:
                    metrics["overlap_edges"] += 1
                elif left == maximum_prior + dt.timedelta(days=1):
                    metrics["contiguous_edges"] += 1
                else:
                    metrics["gap_edges"] += 1
            maximum_prior = right if maximum_prior is None else max(maximum_prior, right)
    return dict(metrics)


def _actual_metric(result: Mapping[str, Any], metric: str) -> int:
    if metric in result:
        return int(result[metric])
    mapping = {
        "distinct_keys": result["duplicates"]["distinct_keys"],
        "duplicate_key_groups": result["duplicates"]["duplicate_key_groups"],
        "duplicate_surplus_rows": result["duplicates"]["duplicate_surplus_rows"],
        "conflicting_duplicate_groups": result["duplicates"]["conflicting_duplicate_groups"],
    }
    if metric in mapping:
        return int(mapping[metric])
    for group in ("date_metrics", "numeric_metrics", "interval_metrics"):
        if metric in result[group]:
            return int(result[group][metric])
    # A missing counter is a deterministic zero, not missing evidence.
    return 0


def _gate_source_result(
    config: Mapping[str, Any], source: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    expected = _expected_for(config, source["filename"])
    failures: dict[str, Any] = {}
    for metric, wanted in expected.items():
        if isinstance(wanted, bool) or not isinstance(wanted, int):
            continue
        actual = _actual_metric(result, metric)
        if actual != wanted:
            failures[metric] = {"expected": wanted, "actual": actual}
    if result["full_file_rows"] != int(source["full_file_rows"]):
        failures["full_file_rows"] = {
            "expected": int(source["full_file_rows"]),
            "actual": result["full_file_rows"],
        }
    if result["duplicates"]["exact_duplicate_groups"]:
        failures["exact_duplicate_groups"] = {
            "expected": 0,
            "actual": result["duplicates"]["exact_duplicate_groups"],
        }
    if result["duplicates"]["conflicting_duplicate_groups"]:
        failures["conflicting_duplicate_groups_policy"] = {
            "expected": 0,
            "actual": result["duplicates"]["conflicting_duplicate_groups"],
        }
    if (
        source["scope_kind"] == "interval_overlap"
        and int(result["interval_metrics"].get("overlap_edges", 0)) != 0
    ):
        failures["overlap_edges_policy"] = {
            "expected": 0,
            "actual": int(result["interval_metrics"].get("overlap_edges", 0)),
        }
    for column, values in config.get("expected_domains", {}).get(
        source["filename"].replace(".", "_"), {}
    ).items():
        observed = sorted(
            (value for value in result["categorical_counts"][column] if value != ""),
            key=lambda value: value.encode("utf-8"),
        )
        if observed != list(values):
            failures[f"domain:{column}"] = {"expected": list(values), "actual": observed}
        if sum(result["categorical_counts"][column].values()) != result["retained_rows"]:
            failures[f"frequency:{column}"] = {
                "expected": result["retained_rows"],
                "actual": sum(result["categorical_counts"][column].values()),
            }
    if failures:
        raise SupportingProfileError(
            f"{source['filename']} configured regression gates failed: {failures}"
        )


def _authenticate_pointer(
    locked: Mapping[str, Any], *, kind: str
) -> tuple[dict[str, Any], Path]:
    pointer_path = _project_path(locked["verified_pointer"])
    expected_hash = locked["verified_pointer_sha256"]
    actual_hash = sha256_file(pointer_path)
    if actual_hash != expected_hash:
        raise SupportingProfileError(
            f"Verified {kind} pointer SHA-256 differs: expected {expected_hash}, actual {actual_hash}"
        )
    pointer = _read_json(pointer_path)
    if pointer.get("status") != locked["required_status"]:
        raise SupportingProfileError(f"Verified {kind} pointer has a non-passing status")
    if kind == "contract":
        identifiers = {
            "contract_id": locked["contract_id"],
            "verification_id": locked["verification_id"],
        }
    else:
        identifiers = {
            "build_id": locked["build_id"],
            "verification_id": locked["verification_id"],
        }
    for key, expected in identifiers.items():
        if pointer.get(key) != expected:
            raise SupportingProfileError(
                f"Verified {kind} pointer {key} differs: expected {expected!r}, "
                f"actual {pointer.get(key)!r}"
            )
    return pointer, pointer_path


def _read_hash_inventory(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SupportingProfileError(f"Cannot read SHA-256 inventory {path}: {exc}") from exc
    for number, line in enumerate(lines, 1):
        try:
            digest, name = line.split("  ", 1)
        except ValueError as exc:
            raise SupportingProfileError(
                f"Malformed SHA-256 inventory line {path}:{number}"
            ) from exc
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not name
            or name in result
        ):
            raise SupportingProfileError(
                f"Malformed or duplicate SHA-256 inventory line {path}:{number}"
            )
        result[name] = digest
    return result


def _authenticate_upstreams(settings: SupportingProfileSettings) -> dict[str, Any]:
    config = settings.config
    upstream = config["upstream"]
    methodology_path = _project_path(upstream["methodology"]["freeze_manifest"])
    if sha256_file(methodology_path) != upstream["methodology"]["freeze_manifest_sha256"]:
        raise SupportingProfileError("Methodology freeze manifest SHA-256 differs")
    methodology = _read_json(methodology_path)
    if methodology.get("status") != "frozen_preimplementation":
        raise SupportingProfileError("Methodology is not frozen_preimplementation")

    phase_01_path = _project_path(upstream["phase_01"]["freeze_manifest"])
    if sha256_file(phase_01_path) != upstream["phase_01"]["freeze_manifest_sha256"]:
        raise SupportingProfileError("Phase 1 freeze manifest SHA-256 differs")
    phase_01 = _read_json(phase_01_path)
    if (
        phase_01.get("status") != "phase_01_frozen"
        or phase_01.get("freeze_id") != upstream["phase_01"]["freeze_id"]
    ):
        raise SupportingProfileError("Phase 1 freeze identity/status differs")

    phase_path = _project_path(upstream["phase_02"]["phase_config"])
    if phase_path != settings.phase_config_path.resolve():
        raise SupportingProfileError("Loaded Phase 2 config path differs from locked path")
    if sha256_file(phase_path) != upstream["phase_02"]["phase_config_sha256"]:
        raise SupportingProfileError("Phase 2 config SHA-256 differs")

    contract, contract_path = _authenticate_pointer(
        upstream["contract"], kind="contract"
    )
    calendar, calendar_path = _authenticate_pointer(
        upstream["calendar"], kind="calendar"
    )
    daily_monthly, profile_path = _authenticate_pointer(
        upstream["daily_monthly_profile"], kind="daily_monthly_profile"
    )
    if daily_monthly.get("primary_manifest_sha256") != upstream[
        "daily_monthly_profile"
    ]["primary_manifest_sha256"]:
        raise SupportingProfileError("Daily/monthly primary manifest SHA-256 differs")
    return {
        "methodology": methodology,
        "methodology_path": methodology_path,
        "phase_01": phase_01,
        "phase_01_path": phase_01_path,
        "contract": contract,
        "contract_pointer_path": contract_path,
        "calendar": calendar,
        "calendar_pointer_path": calendar_path,
        "daily_monthly_profile": daily_monthly,
        "daily_monthly_pointer_path": profile_path,
    }


def _frozen_hashes(settings: SupportingProfileSettings) -> dict[str, str]:
    freeze_path = _project_path(settings.config["upstream"]["phase_01"]["freeze_manifest"])
    inventory_path = freeze_path.parent / "raw_freeze_recheck.sha256"
    inventory = _read_hash_inventory(inventory_path)
    missing = sorted(set(SOURCE_ORDER) - set(inventory))
    if missing:
        raise SupportingProfileError(
            f"Phase 1 final recheck inventory lacks supporting sources: {missing}"
        )
    return {filename: inventory[filename] for filename in SOURCE_ORDER}


def _calendar_dates(
    settings: SupportingProfileSettings, upstreams: Mapping[str, Any]
) -> tuple[set[dt.date], dict[str, Any]]:
    pointer = upstreams["calendar"]
    manifest_path = (
        settings.project_root / pointer["primary_artifact_path"] / "manifest.json"
    ).resolve()
    if sha256_file(manifest_path) != pointer["primary_manifest_sha256"]:
        raise SupportingProfileError("Verified calendar manifest SHA-256 differs")
    manifest = _read_json(manifest_path)
    matches = [
        item
        for item in manifest.get("outputs", [])
        if Path(item.get("path", "")).name == "trading_calendar.csv"
    ]
    if len(matches) != 1:
        raise SupportingProfileError("Verified calendar manifest lacks one trading_calendar.csv")
    receipt = matches[0]
    path = Path(receipt["path"]).resolve()
    if not path.is_file() or sha256_file(path) != receipt["sha256"]:
        raise SupportingProfileError("Verified trading calendar bytes differ from manifest")
    dates: set[dt.date] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "trading_date" not in reader.fieldnames:
                # Existing calendar uses observed_date in some versions; accept exactly one
                # unambiguous date column after authenticating the immutable bytes.
                candidates = [
                    column
                    for column in (reader.fieldnames or [])
                    if column
                    in {"calendar_date", "date", "observed_date", "trading_date"}
                ]
                if len(candidates) != 1:
                    raise SupportingProfileError(
                        "Verified trading calendar lacks one recognized date column"
                    )
                date_column = candidates[0]
            else:
                date_column = "trading_date"
            for row in reader:
                value = _strict_date(row[date_column], label="verified trading calendar")
                assert value is not None
                if value in dates:
                    raise SupportingProfileError("Verified trading calendar contains duplicates")
                dates.add(value)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        raise SupportingProfileError(f"Cannot read verified trading calendar: {exc}") from exc
    expected = int(settings.config["referential_integrity"]["verified_calendar_dates"])
    if len(dates) != expected:
        raise SupportingProfileError(
            f"Verified calendar date count differs: expected {expected}, actual {len(dates)}"
        )
    try:
        portable_path = path.relative_to(settings.database_root.resolve()).as_posix()
    except ValueError as exc:
        raise SupportingProfileError(
            "Verified trading calendar is outside configured database root"
        ) from exc
    return dates, {
        "path": portable_path,
        "sha256": receipt["sha256"],
        "size_bytes": int(receipt["size_bytes"]),
        "date_count": len(dates),
    }


def _check_referential(
    config: Mapping[str, Any], context: _ScanContext
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = config["referential_integrity"]
    context.referential["names_nonconstant_security_lifetime_permnos"] = sum(
        len(values) > 1 for values in context.names_lifetime.values()
    )
    context.referential["names_permnos_missing_current_header_key"] = sum(
        permno not in context.header_permnos for permno in context.names_intervals
    )
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    failures: dict[str, Any] = {}
    for metric, wanted in expected.items():
        if isinstance(wanted, bool) or not isinstance(wanted, int):
            continue
        actual = len(context.names_intervals) if metric == "verified_calendar_dates" else int(
            context.referential[metric]
        )
        # Calendar count is supplied separately by the authenticated calendar and
        # overwritten by the compiler; it is not the names-PERMNO count.
        if metric == "verified_calendar_dates":
            continue
        fatal = metric in {
            "names_invalid_or_nonpositive_permco_rows",
            "names_nonconstant_security_lifetime_permnos",
            "names_permnos_missing_current_header_key",
        } or metric.startswith("shares_rows_with_") or (
            metric.startswith("distribution_rows_with_")
            and metric.endswith("_on_ex_date")
        ) or (
            metric.startswith("delisting_rows_with_")
            and metric.endswith("_on_delisting_date")
        )
        status = "PASS" if actual == wanted else "FAIL"
        rows.append(
            {
                "metric": metric,
                "expected": wanted,
                "actual": actual,
                "status": status,
                "fatal": fatal,
            }
        )
        checks.append(
            _check_row(
                f"REFERENTIAL::{metric}",
                "ALL",
                "referential_integrity",
                metric,
                wanted,
                actual,
                fatal=fatal,
            )
        )
        if actual != wanted:
            failures[metric] = {"expected": wanted, "actual": actual}
    if failures:
        raise SupportingProfileError(
            f"Configured referential-integrity gates failed: {failures}"
        )
    return rows, checks


def _check_row(
    check_id: str,
    source: str,
    category: str,
    metric: str,
    expected: Any,
    actual: Any,
    *,
    fatal: bool = True,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "source": source,
        "category": category,
        "metric": metric,
        "expected": expected,
        "actual": actual,
        "status": "PASS" if actual == expected else "FAIL",
        "fatal": fatal,
    }


def _metric_category(metric: str, source: Mapping[str, Any] | None = None) -> str:
    if metric.startswith("blank_") and source is not None:
        column = metric.removeprefix("blank_")
        if column in source.get("positive_integer_columns", []):
            return "numeric"
    if metric.startswith(("duplicate_", "distinct_")):
        return "duplicates"
    if metric.startswith(("blank_", "declare_", "record_", "pay_", "posting_", "next_", "amount_", "delisting_", "both_", "security_", "information_")):
        return "dates"
    if metric.startswith(("invalid_", "nonpositive_")):
        return "numeric"
    if metric in {
        "inverted_intervals",
        "overlap_edges",
        "contiguous_edges",
        "gap_edges",
        "starts_before_scope",
        "ends_after_scope",
        "spans_entire_scope",
    }:
        return "intervals"
    return "structure"


def _compile_rows(
    config: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    receipts: Mapping[str, Mapping[str, Any]],
    referential_rows: Sequence[Mapping[str, Any]],
    referential_checks: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {
        "source_structure": [],
        "duplicate_summary": [],
        "duplicate_multiplicity": [],
        "categorical_domains": [],
        "date_diagnostics": [],
        "interval_diagnostics": [],
        "numeric_validity": [],
        "referential_integrity": list(referential_rows),
        "checks": list(referential_checks),
    }
    source_map = _source_by_name(config)
    for result in results:
        filename = result["source"]
        source = source_map[filename]
        receipt = receipts[filename]
        output["source_structure"].append(
            {
                "source": filename,
                "scope_kind": result["scope_kind"],
                "full_file_rows": result["full_file_rows"],
                "retained_rows": result["retained_rows"],
                "excluded_rows": result["excluded_rows"],
                "header_columns": result["header_columns"],
                "key_fields": result["key_fields"],
                "size_bytes": receipt["size_bytes"],
                "sha256": receipt["sha256"],
                "excluded_non_date_values_compared_or_aggregated": result[
                    "excluded_non_date_values_compared_or_aggregated"
                ],
            }
        )
        output["duplicate_summary"].append(
            {
                "source": filename,
                "key_fields": result["key_fields"],
                "retained_rows": result["retained_rows"],
                "distinct_keys": result["duplicates"]["distinct_keys"],
                "unique_permnos": result["unique_permnos"],
                "duplicate_key_groups": result["duplicates"]["duplicate_key_groups"],
                "duplicate_surplus_rows": result["duplicates"]["duplicate_surplus_rows"],
                "exact_duplicate_groups": result["duplicates"]["exact_duplicate_groups"],
                "conflicting_duplicate_groups": result["duplicates"][
                    "conflicting_duplicate_groups"
                ],
                "full_decoded_row_comparison": source["scope_kind"] != "qa_key_only",
            }
        )
        for multiplicity, groups in result["duplicates"]["multiplicity"].items():
            output["duplicate_multiplicity"].append(
                {
                    "source": filename,
                    "multiplicity": multiplicity,
                    "key_group_count": groups,
                }
            )
        expected = _expected_for(config, filename)
        for metric, wanted in expected.items():
            if isinstance(wanted, bool) or not isinstance(wanted, int):
                continue
            actual = _actual_metric(result, metric)
            category = _metric_category(metric, source)
            output["checks"].append(
                _check_row(
                    f"SOURCE::{filename}::{metric}",
                    filename,
                    category,
                    metric,
                    wanted,
                    actual,
                )
            )
            diagnostic_row = {
                "source": filename,
                "metric": metric,
                "expected": wanted,
                "actual": actual,
                "status": "PASS" if actual == wanted else "FAIL",
            }
            if category == "dates":
                output["date_diagnostics"].append(diagnostic_row)
            elif category == "intervals":
                output["interval_diagnostics"].append(diagnostic_row)
            elif category == "numeric":
                output["numeric_validity"].append(diagnostic_row)
        domains = config.get("expected_domains", {}).get(
            filename.replace(".", "_"), {}
        )
        for column in source.get("categorical_columns", []):
            counts = result["categorical_counts"][column]
            expected_values = list(domains[column])
            observed_nonblank = sorted(
                (value for value in counts if value != ""),
                key=lambda value: value.encode("utf-8"),
            )
            output["checks"].append(
                _check_row(
                    f"DOMAIN::{filename}::{column}",
                    filename,
                    "categorical_domain",
                    column,
                    expected_values,
                    observed_nonblank,
                )
            )
            output["checks"].append(
                _check_row(
                    f"FREQUENCY::{filename}::{column}",
                    filename,
                    "categorical_frequency",
                    column,
                    result["retained_rows"],
                    sum(counts.values()),
                )
            )
            for raw_value, count in counts.items():
                output["categorical_domains"].append(
                    {
                        "source": filename,
                        "column": column,
                        "raw_value": raw_value,
                        "row_count": count,
                        "is_blank": raw_value == "",
                        "classification": (
                            "blank"
                            if raw_value == ""
                            else "expected_nonblank"
                            if raw_value in expected_values
                            else "unexpected_nonblank"
                        ),
                    }
                )
    order = {name: index for index, name in enumerate(SOURCE_ORDER)}
    for name, rows in output.items():
        if name == "checks":
            rows.sort(key=lambda row: row["check_id"].encode("utf-8"))
        elif name == "referential_integrity":
            rows.sort(key=lambda row: row["metric"].encode("utf-8"))
        elif name == "categorical_domains":
            rows.sort(
                key=lambda row: (
                    order[row["source"]],
                    row["column"].encode("utf-8"),
                    row["raw_value"] != "",
                    row["raw_value"].encode("utf-8"),
                )
            )
        else:
            rows.sort(
                key=lambda row: (
                    order.get(row.get("source", SOURCE_ORDER[0]), -1),
                    str(row.get("metric", row.get("multiplicity", ""))).encode("utf-8"),
                )
            )
    return output


def _compute_build_id(
    binding: Mapping[str, Any],
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    receipts: Mapping[str, Mapping[str, Any]],
) -> str:
    receipt_fields = (
        "filename",
        "sha256",
        "size_bytes",
        "full_file_rows_expected",
        "full_file_rows_scanned",
    )
    portable_receipts = {
        name: {field: receipt[field] for field in receipt_fields if field in receipt}
        for name, receipt in sorted(receipts.items())
    }
    identity = {
        "binding": binding,
        "quality_rows": rows,
        "source_receipts": portable_receipts,
    }
    digest = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    return f"crsp-support-{digest[:24]}"


def _validate_compiled(
    config: Mapping[str, Any], compiled: Mapping[str, Any]
) -> None:
    required = {
        "binding",
        "build_id",
        "categorical_domains",
        "checks",
        "date_diagnostics",
        "duplicate_multiplicity",
        "duplicate_summary",
        "input_receipt",
        "interval_diagnostics",
        "numeric_validity",
        "profile_contract",
        "referential_integrity",
        "source_structure",
        "summary",
    }
    if set(compiled) != required:
        raise SupportingProfileError(
            f"Compiled profile keys differ: expected {sorted(required)}, "
            f"actual {sorted(compiled)}"
        )
    build_id = compiled["build_id"]
    if (
        not isinstance(build_id, str)
        or not build_id.startswith("crsp-support-")
        or len(build_id.removeprefix("crsp-support-")) != 24
        or any(
            char not in "0123456789abcdef"
            for char in build_id.removeprefix("crsp-support-")
        )
    ):
        raise SupportingProfileError("Supporting build ID format differs")
    checks = compiled["checks"]
    if not checks or any(row["status"] != "PASS" for row in checks):
        raise SupportingProfileError("Compiled profile contains a missing or failed check")
    summary = compiled["summary"]
    if (
        summary["check_count"] != len(checks)
        or summary["pass_count"] != len(checks)
        or summary["fail_count"] != 0
        or summary["build_id"] != build_id
        or summary["status"]
        != "supporting_profile_primary_passed_pending_independent_verification"
    ):
        raise SupportingProfileError("Compiled summary does not reconcile to checks")
    if set(CSV_SCHEMAS) != {
        name for name in config["outputs"]["closed_quality_file_set"] if name.endswith(".csv")
    }:
        raise SupportingProfileError("Locked CSV file set differs from compiler schemas")
    calendar_path = compiled["input_receipt"].get("calendar", {}).get("path")
    if calendar_path is not None and Path(calendar_path).is_absolute():
        raise SupportingProfileError("Immutable calendar receipt path is not portable")
    # No row-level security key may escape the scanner.
    forbidden = {"PERMNO", "PERMCO", "CUSIP", "Ticker", "IssuerNm"}
    for section in (
        "categorical_domains",
        "checks",
        "date_diagnostics",
        "duplicate_multiplicity",
        "duplicate_summary",
        "interval_diagnostics",
        "numeric_validity",
        "referential_integrity",
        "source_structure",
    ):
        for row in compiled[section]:
            if forbidden & set(row):
                raise SupportingProfileError(
                    f"Raw security key field escaped in aggregate section {section}"
                )


def compile_supporting_profile(
    settings: SupportingProfileSettings | None = None,
) -> dict[str, Any]:
    """Authenticate, scan, gate, and compile aggregate supporting-source QA.

    The function has no publication side effects.  Its only writes are an
    automatically removed SQLite scratch directory below ``settings.run_root``.
    """
    settings = settings or load_settings()
    sealed_config_sha256 = sha256_file(settings.profile_config_path)
    sealed_paths_sha256 = sha256_file(settings.paths_config_path)
    sealed_phase_sha256 = sha256_file(settings.phase_config_path)
    sealed_implementation_sha256 = sha256_file(Path(__file__).resolve())
    with settings.profile_config_path.open("rb") as handle:
        sealed_config = tomllib.load(handle)
    if sealed_config != settings.config:
        raise SupportingProfileError("Loaded settings differ from sealed profile TOML")

    upstreams = _authenticate_upstreams(settings)
    frozen_hashes = _frozen_hashes(settings)
    calendar_dates, calendar_receipt = _calendar_dates(settings, upstreams)
    source_map = _source_by_name(settings.config)
    before: dict[str, dict[str, int]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for filename in SOURCE_ORDER:
        source = source_map[filename]
        path = settings.raw_root / filename
        if not path.is_file():
            raise SupportingProfileError(f"Missing raw supporting source: {path}")
        before[filename] = _operational_stat(path)
        if before[filename]["size_bytes"] != int(source["size_bytes"]):
            raise SupportingProfileError(f"{filename} size differs from locked config")
        digest = sha256_file(path)
        if digest != source["sha256"] or digest != frozen_hashes[filename]:
            raise SupportingProfileError(
                f"{filename} SHA-256 differs from locked config/Phase 1 freeze"
            )
        _exact_header(path, source["header"])
        receipts[filename] = {
            "filename": filename,
            "sha256": digest,
            "size_bytes": before[filename]["size_bytes"],
            "full_file_rows_expected": int(source["full_file_rows"]),
        }

    settings.run_root.mkdir(parents=True, exist_ok=True)
    context = _ScanContext(calendar_sorted=tuple(sorted(calendar_dates)))
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="crsp-supporting-profile-", dir=settings.run_root
    ) as temporary:
        scratch = Path(temporary)
        for filename in SOURCE_ORDER:
            result = _scan_one_source(
                settings.raw_root / filename,
                source_map[filename],
                settings.config,
                scratch_directory=scratch,
                context=context,
                calendar_dates=calendar_dates,
                block_size_bytes=settings.block_size_bytes,
            )
            results.append(result)
            if filename == "crsp_names_ciz.csv":
                for intervals in context.names_intervals.values():
                    intervals.sort()

    after = {
        filename: _operational_stat(settings.raw_root / filename)
        for filename in SOURCE_ORDER
    }
    for filename in SOURCE_ORDER:
        if before[filename] != after[filename]:
            raise SupportingProfileError(f"{filename} changed during the profile scan")
        receipts[filename]["full_file_rows_scanned"] = next(
            item["full_file_rows"] for item in results if item["source"] == filename
        )

    if sha256_file(settings.profile_config_path) != sealed_config_sha256:
        raise SupportingProfileError("Supporting profile config changed during scan")
    if sha256_file(settings.paths_config_path) != sealed_paths_sha256:
        raise SupportingProfileError("Paths config changed during scan")
    if sha256_file(settings.phase_config_path) != sealed_phase_sha256:
        raise SupportingProfileError("Phase config changed during scan")
    if sha256_file(Path(__file__).resolve()) != sealed_implementation_sha256:
        raise SupportingProfileError("Primary implementation changed during scan")

    referential_rows, referential_checks = _check_referential(
        settings.config, context
    )
    calendar_expected = int(
        settings.config["referential_integrity"]["verified_calendar_dates"]
    )
    calendar_row = {
        "metric": "verified_calendar_dates",
        "expected": calendar_expected,
        "actual": len(calendar_dates),
        "status": "PASS" if len(calendar_dates) == calendar_expected else "FAIL",
        "fatal": True,
    }
    referential_rows.append(calendar_row)
    referential_checks.append(
        _check_row(
            "REFERENTIAL::verified_calendar_dates",
            "ALL",
            "referential_integrity",
            "verified_calendar_dates",
            calendar_expected,
            len(calendar_dates),
        )
    )
    rows = _compile_rows(
        settings.config, results, receipts, referential_rows, referential_checks
    )
    binding = {
        "schema_version": 1,
        "contract_id": upstreams["contract"]["contract_id"],
        "contract_verification_id": upstreams["contract"]["verification_id"],
        "calendar_build_id": upstreams["calendar"]["build_id"],
        "calendar_verification_id": upstreams["calendar"]["verification_id"],
        "daily_monthly_profile_build_id": upstreams["daily_monthly_profile"][
            "build_id"
        ],
        "daily_monthly_profile_verification_id": upstreams[
            "daily_monthly_profile"
        ]["verification_id"],
        "methodology_freeze_sha256": settings.config["upstream"]["methodology"][
            "freeze_manifest_sha256"
        ],
        "phase_01_freeze_manifest_sha256": settings.config["upstream"]["phase_01"][
            "freeze_manifest_sha256"
        ],
        "phase_config_sha256": sealed_phase_sha256,
        "paths_config_sha256": sealed_paths_sha256,
        "profile_config_sha256": sealed_config_sha256,
        "implementation_sha256": sealed_implementation_sha256,
    }
    build_id = _compute_build_id(binding, rows, receipts)
    summary = {
        "schema_version": 1,
        "profile_name": settings.config["profile_name"],
        "build_id": build_id,
        "status": "supporting_profile_primary_passed_pending_independent_verification",
        "source_count": len(results),
        "full_file_rows_scanned": sum(item["full_file_rows"] for item in results),
        "retained_rows": sum(item["retained_rows"] for item in results),
        "check_count": len(rows["checks"]),
        "pass_count": sum(row["status"] == "PASS" for row in rows["checks"]),
        "fail_count": sum(row["status"] != "PASS" for row in rows["checks"]),
        "canonical_tables_materialized": False,
        "numeric_units_validated": False,
        "returns_reconciled": False,
        "features_built": False,
        "targets_built": False,
        "portfolios_built": False,
        "pnl_built": False,
        "performance_analyzed": False,
    }
    profile_contract = {
        "schema_version": 1,
        "profile_name": settings.config["profile_name"],
        "profile_version": settings.config["profile_version"],
        "scope": settings.config["scope"],
        "key_semantics": settings.config["key_semantics"],
        "duplicate_policy": settings.config["duplicate_policy"],
        "interval_policy": settings.config["interval_policy"],
        "categorical_policy": settings.config["categorical_policy"],
        "diagnostic_status": settings.config["diagnostic_status"],
        "csv_schemas": {name: list(columns) for name, columns in CSV_SCHEMAS.items()},
        "closed_quality_file_set": settings.config["outputs"][
            "closed_quality_file_set"
        ],
    }
    input_receipt = {
        "schema_version": 1,
        "build_id": build_id,
        "binding": binding,
        "calendar": calendar_receipt,
        "sources": [receipts[filename] for filename in SOURCE_ORDER],
    }
    compiled: dict[str, Any] = {
        "binding": binding,
        "build_id": build_id,
        "categorical_domains": rows["categorical_domains"],
        "checks": rows["checks"],
        "date_diagnostics": rows["date_diagnostics"],
        "duplicate_multiplicity": rows["duplicate_multiplicity"],
        "duplicate_summary": rows["duplicate_summary"],
        "input_receipt": input_receipt,
        "interval_diagnostics": rows["interval_diagnostics"],
        "numeric_validity": rows["numeric_validity"],
        "profile_contract": profile_contract,
        "referential_integrity": rows["referential_integrity"],
        "source_structure": rows["source_structure"],
        "summary": summary,
    }
    _validate_compiled(settings.config, compiled)
    return compiled


def render_quality_payloads(
    config: Mapping[str, Any],
    compiled: Mapping[str, Any],
    *,
    settings: SupportingProfileSettings | None = None,
) -> dict[str, bytes]:
    """Render the exact closed aggregate quality file set without writing it."""
    _validate_compiled(config, compiled)
    payloads: dict[str, bytes] = {
        "_SUCCESS.json": canonical_json_bytes(compiled["summary"]),
        "input_receipt.json": canonical_json_bytes(compiled["input_receipt"]),
        "profile_contract.json": canonical_json_bytes(compiled["profile_contract"]),
        "summary.json": canonical_json_bytes(compiled["summary"]),
    }
    section_by_filename = {
        "categorical_domains.csv": "categorical_domains",
        "checks.csv": "checks",
        "date_diagnostics.csv": "date_diagnostics",
        "duplicate_multiplicity.csv": "duplicate_multiplicity",
        "duplicate_summary.csv": "duplicate_summary",
        "interval_diagnostics.csv": "interval_diagnostics",
        "numeric_validity.csv": "numeric_validity",
        "referential_integrity.csv": "referential_integrity",
        "source_structure.csv": "source_structure",
    }
    for filename, section in section_by_filename.items():
        payloads[filename] = _csv_bytes(CSV_SCHEMAS[filename], compiled[section])
    expected = set(config["outputs"]["closed_quality_file_set"])
    if set(payloads) != expected:
        raise SupportingProfileError(
            f"Rendered quality file set differs: expected {sorted(expected)}, "
            f"actual {sorted(payloads)}"
        )
    return dict(sorted(payloads.items()))


def render_artifact_documents(
    settings: SupportingProfileSettings,
    compiled: Mapping[str, Any],
    quality_directory: Path,
    quality_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    """Render manifest, SHA inventory, and primary pointer without writes."""
    _validate_compiled(settings.config, compiled)
    expected = set(settings.config["outputs"]["closed_quality_file_set"])
    if set(quality_payloads) != expected:
        raise SupportingProfileError("Artifact renderer received a non-closed quality set")
    expected_directory = (
        settings.quality_root
        / compiled["binding"]["contract_id"]
        / settings.config["outputs"]["quality_subdirectory"]
        / compiled["build_id"]
    ).resolve()
    if quality_directory.resolve() != expected_directory:
        raise SupportingProfileError("Quality directory differs from locked destination")
    try:
        quality_relative = quality_directory.resolve().relative_to(
            settings.database_root.resolve()
        )
    except ValueError as exc:
        raise SupportingProfileError("Quality directory is outside database root") from exc
    outputs = [
        {
            "path": (quality_relative / filename).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for filename, payload in sorted(quality_payloads.items())
    ]
    manifest = {
        "schema_version": 1,
        "profile_name": settings.config["profile_name"],
        "build_id": compiled["build_id"],
        "contract_id": compiled["binding"]["contract_id"],
        "status": compiled["summary"]["status"],
        "binding": compiled["binding"],
        "summary": compiled["summary"],
        "outputs": outputs,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    inventory_bytes = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in outputs
    ).encode("utf-8")
    artifact_directory = (
        settings.artifact_root
        / settings.config["outputs"]["artifact_subdirectory"]
        / compiled["build_id"]
    )
    try:
        artifact_path = artifact_directory.resolve().relative_to(
            settings.project_root.resolve()
        ).as_posix()
    except ValueError as exc:
        raise SupportingProfileError("Artifact root is outside project root") from exc
    pointer = {
        "schema_version": 1,
        "build_id": compiled["build_id"],
        "contract_id": compiled["binding"]["contract_id"],
        "status": compiled["summary"]["status"],
        "artifact_path": artifact_path,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    return {
        "artifact_directory": artifact_directory,
        "artifact_payloads": {
            "manifest.json": manifest_bytes,
            "output_hashes.sha256": inventory_bytes,
        },
        "manifest": manifest,
        "pointer": pointer,
        "pointer_bytes": canonical_json_bytes(pointer),
    }


__all__ = [
    "SupportingProfileError",
    "SupportingProfileSettings",
    "canonical_json_bytes",
    "compile_supporting_profile",
    "load_settings",
    "render_artifact_documents",
    "render_quality_payloads",
    "sha256_file",
]
