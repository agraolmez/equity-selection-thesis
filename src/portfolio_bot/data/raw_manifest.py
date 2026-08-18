"""Build a deterministic, streaming inventory of the frozen raw CSV snapshot.

The module deliberately separates authoritative snapshot facts from operational
metadata. CSV values are never rewritten and raw records are never copied into
project artifacts.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import re
import sys
import time
import tomllib
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASE_CONFIG = PROJECT_ROOT / "configs" / "phase_01_raw_snapshot.toml"
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.toml"
BYTE_CHUNK_SIZE = 16 << 20


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _sha256_file(path: Path, chunk_size: int = BYTE_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _utc_iso_from_ns(value_ns: int | None) -> str | None:
    if value_ns is None:
        return None
    return dt.datetime.fromtimestamp(value_ns / 1_000_000_000, tz=dt.timezone.utc).isoformat()


def _stat_fingerprint(path: Path) -> dict[str, int | None]:
    stat = path.stat()
    birth_ns = getattr(stat, "st_birthtime_ns", None)
    if birth_ns is None and hasattr(stat, "st_birthtime"):
        birth_ns = int(stat.st_birthtime * 1_000_000_000)
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "birthtime_ns": birth_ns,
    }


def _same_scan_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("device", "inode", "size_bytes", "mtime_ns"))


@dataclasses.dataclass(frozen=True)
class FileContract:
    filename: str
    family: str
    role: str
    coverage_date: str
    date_columns: tuple[str, ...]
    key_columns: tuple[str, ...]
    key_semantics: str
    grain: str
    required_columns: tuple[str, ...]
    availability_note: str


@dataclasses.dataclass(frozen=True)
class PhaseSettings:
    phase_config_path: Path
    paths_config_path: Path
    raw_root: Path
    database_root: Path
    artifact_root: Path
    staging_directory: Path
    final_directory: Path
    verification_directory: Path
    freeze_directory: Path
    expected_file_count: int
    methodology_freeze: Path
    parser: dict[str, Any]
    quality: dict[str, Any]
    contracts: tuple[FileContract, ...]
    configuration_hash: str
    path_configuration_hash: str
    contract_hashes: dict[str, str]


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(
    phase_config: Path = DEFAULT_PHASE_CONFIG,
    paths_config: Path = DEFAULT_PATHS_CONFIG,
) -> PhaseSettings:
    phase_config = phase_config.resolve()
    paths_config = paths_config.resolve()
    with phase_config.open("rb") as handle:
        phase = tomllib.load(handle)
    with paths_config.open("rb") as handle:
        paths = tomllib.load(handle)

    configured_root = Path(paths["database"]["root"]).expanduser()
    database_root = Path(os.environ.get("PORTFOLIO_DATABASE_ROOT", configured_root)).expanduser().resolve()
    raw_root = (database_root / paths["database"]["raw"]).resolve()

    contract_directory = _project_path(phase["snapshot"]["contract_directory"])
    contracts: list[FileContract] = []
    contract_hashes: dict[str, str] = {}
    for contract_path in sorted(contract_directory.glob("*.toml")):
        with contract_path.open("rb") as handle:
            payload = tomllib.load(handle)
        try:
            contract_label = str(contract_path.relative_to(PROJECT_ROOT))
        except ValueError:
            contract_label = contract_path.name
        contract_hashes[contract_label] = _sha256_file(contract_path)
        for item in payload.get("files", []):
            contracts.append(
                FileContract(
                    filename=item["filename"],
                    family=item["family"],
                    role=item["role"],
                    coverage_date=item["coverage_date"],
                    date_columns=tuple(item["date_columns"]),
                    key_columns=tuple(item["key_columns"]),
                    key_semantics=item.get("key_semantics", "candidate_grain_diagnostic"),
                    grain=item["grain"],
                    required_columns=tuple(item["required_columns"]),
                    availability_note=item.get(
                        "availability_note",
                        "This Phase 1 field is a coverage axis only. Phase 2 constructs public availability exactly as specified in Methodology v1.",
                    ),
                )
            )

    filenames = [contract.filename for contract in contracts]
    if len(filenames) != len(set(filenames)):
        duplicates = sorted(name for name, count in Counter(filenames).items() if count > 1)
        raise ValueError(f"Duplicate filenames in raw contracts: {duplicates}")

    configuration_identity = {
        "phase_config_sha256": _sha256_file(phase_config),
        "contract_hashes": contract_hashes,
    }
    return PhaseSettings(
        phase_config_path=phase_config,
        paths_config_path=paths_config,
        raw_root=raw_root,
        database_root=database_root,
        artifact_root=_project_path(phase["snapshot"]["artifact_root"]),
        staging_directory=_project_path(phase["snapshot"]["staging_directory"]),
        final_directory=_project_path(phase["snapshot"]["final_directory"]),
        verification_directory=_project_path(phase["snapshot"]["verification_directory"]),
        freeze_directory=_project_path(phase["snapshot"]["freeze_directory"]),
        expected_file_count=int(phase["snapshot"]["expected_file_count"]),
        methodology_freeze=_project_path(phase["snapshot"]["methodology_freeze"]),
        parser=dict(phase["parser"]),
        quality=dict(phase["quality"]),
        contracts=tuple(sorted(contracts, key=lambda item: item.filename)),
        configuration_hash=_sha256_json(configuration_identity),
        path_configuration_hash=_sha256_file(paths_config),
        contract_hashes=contract_hashes,
    )


def verify_methodology_freeze(path: Path) -> dict[str, Any]:
    """Verify the declared pre-implementation freeze and every frozen artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("status") != "frozen_preimplementation":
        failures.append(f"unexpected status {payload.get('status')!r}")
    if payload.get("empirical_results_seen") is not False:
        failures.append("empirical_results_seen is not false")
    if payload.get("implementation_status") != "not_started":
        failures.append(f"unexpected implementation_status {payload.get('implementation_status')!r}")
    verified_files: list[dict[str, Any]] = []
    for item in payload.get("files", []):
        artifact_path = _project_path(item["path"])
        if not artifact_path.is_file():
            failures.append(f"missing frozen artifact {item['path']}")
            continue
        actual = _sha256_file(artifact_path)
        verified_files.append(
            {"path": item["path"], "declared_sha256": item["sha256"], "actual_sha256": actual}
        )
        if actual != item["sha256"]:
            failures.append(f"hash mismatch for {item['path']}")
    if len(verified_files) != len(payload.get("files", [])):
        failures.append("not every frozen artifact could be verified")
    if failures:
        raise RuntimeError("Methodology freeze verification failed: " + "; ".join(failures))
    return {
        "freeze_id": payload["freeze_id"],
        "methodology_version": payload["methodology_version"],
        "freeze_manifest_sha256": _sha256_file(path),
        "verified_files": verified_files,
    }


def _configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_header(path: Path) -> list[str]:
    _configure_csv_limit()
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        return next(csv.reader(handle), [])


_INTEGER = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[ T].+")


def _classify_value(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return "empty"
    if _DATE.fullmatch(stripped):
        return "date"
    if _DATETIME.fullmatch(stripped):
        return "datetime"
    if _INTEGER.fullmatch(stripped):
        return "integer"
    if _FLOAT.fullmatch(stripped):
        return "numeric"
    return "text"


def sample_schema(path: Path, header: list[str], sample_rows: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _configure_csv_limit()
    observed: list[Counter[str]] = [Counter() for _ in header]
    sampled = 0
    malformed = 0
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if sampled >= sample_rows:
                break
            if len(row) != len(header):
                malformed += 1
                continue
            sampled += 1
            for position, value in enumerate(row):
                observed[position][_classify_value(value)] += 1

    inferred: list[dict[str, Any]] = []
    for position, (name, counts) in enumerate(zip(header, observed)):
        nonempty_types = set(counts) - {"empty"}
        if not nonempty_types:
            inferred_type = "unknown_in_sample"
        elif nonempty_types <= {"integer"}:
            inferred_type = "integer_like"
        elif nonempty_types <= {"integer", "numeric"}:
            inferred_type = "numeric_like"
        elif nonempty_types <= {"date"}:
            inferred_type = "iso_date_like"
        elif nonempty_types <= {"date", "datetime"}:
            inferred_type = "iso_datetime_like"
        else:
            inferred_type = "text_or_mixed"
        inferred.append(
            {
                "position": position,
                "name": name,
                "physical_csv_type": "text",
                "sample_inferred_type": inferred_type,
                "sample_nonempty": sum(count for kind, count in counts.items() if kind != "empty"),
                "sample_empty": counts.get("empty", 0),
            }
        )
    return inferred, {"sampled_data_rows": sampled, "malformed_sample_rows": malformed}


def scan_bytes(path: Path, progress_gib: int = 2) -> dict[str, Any]:
    before = _stat_fingerprint(path)
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    utf8_valid = True
    bytes_read = 0
    lf_count = 0
    cr_count = 0
    crlf_count = 0
    nul_count = 0
    first_bytes = b""
    last_byte = b""
    previous_ended_cr = False
    next_progress = progress_gib * (1 << 30)

    with path.open("rb") as handle:
        while chunk := handle.read(BYTE_CHUNK_SIZE):
            if not first_bytes:
                first_bytes = chunk[:4]
            digest.update(chunk)
            bytes_read += len(chunk)
            lf_count += chunk.count(b"\n")
            cr_count += chunk.count(b"\r")
            crlf_count += chunk.count(b"\r\n")
            if previous_ended_cr and chunk.startswith(b"\n"):
                crlf_count += 1
            previous_ended_cr = chunk.endswith(b"\r")
            nul_count += chunk.count(b"\x00")
            last_byte = chunk[-1:]
            if utf8_valid:
                try:
                    decoder.decode(chunk, final=False)
                except UnicodeDecodeError:
                    utf8_valid = False
            if progress_gib > 0 and bytes_read >= next_progress:
                print(
                    json.dumps(
                        {
                            "event": "hash_progress",
                            "file": path.name,
                            "gib_read": round(bytes_read / (1 << 30), 2),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                while next_progress <= bytes_read:
                    next_progress += progress_gib * (1 << 30)
    if utf8_valid:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            utf8_valid = False

    after = _stat_fingerprint(path)
    if crlf_count and crlf_count == lf_count == cr_count:
        newline_style = "CRLF"
    elif lf_count and cr_count == 0:
        newline_style = "LF"
    elif not lf_count and cr_count:
        newline_style = "CR"
    elif lf_count or cr_count:
        newline_style = "MIXED"
    else:
        newline_style = "NONE"

    return {
        "sha256": digest.hexdigest(),
        "bytes_read": bytes_read,
        "size_bytes": before["size_bytes"],
        "stat_before": before,
        "stat_after": after,
        "stable_during_byte_scan": _same_scan_identity(before, after),
        "filesystem_mtime_utc": _utc_iso_from_ns(before["mtime_ns"]),
        "filesystem_birthtime_utc": _utc_iso_from_ns(before["birthtime_ns"]),
        "has_utf8_bom": first_bytes.startswith(codecs.BOM_UTF8),
        "utf8_valid": utf8_valid,
        "nul_bytes": nul_count,
        "lf_count": lf_count,
        "crlf_count": crlf_count,
        "lone_cr_count": cr_count - crlf_count,
        "newline_style": newline_style,
        "ends_with_line_break": last_byte in {b"\n", b"\r"},
    }


def _trimmed(array: pa.Array) -> pa.Array:
    return pc.utf8_trim_whitespace(array)


def _nonblank(array: pa.Array) -> pa.Array:
    trimmed = _trimmed(array)
    return pc.fill_null(pc.greater(pc.utf8_length(trimmed), 0), False)


def _true_count(array: pa.Array) -> int:
    result = pc.sum(pc.cast(pc.fill_null(array, False), pa.int64())).as_py()
    return int(result or 0)


class DateAccumulator:
    def __init__(self, name: str, sentinels: Iterable[str]):
        self.name = name
        self.sentinels = tuple(sentinels)
        self.total = 0
        self.nonblank = 0
        self.sentinel = 0
        self.valid = 0
        self.invalid = 0
        self.minimum: dt.datetime | None = None
        self.maximum: dt.datetime | None = None
        self.by_year: Counter[int] = Counter()
        self.by_format: Counter[str] = Counter()

    def update(self, array: pa.Array) -> None:
        self.total += len(array)
        trimmed = _trimmed(array)
        nonblank = _nonblank(array)
        nonblank_count = _true_count(nonblank)
        self.nonblank += nonblank_count
        if self.sentinels:
            sentinel_mask = pc.and_(
                nonblank,
                pc.fill_null(
                    pc.is_in(trimmed, value_set=pa.array(self.sentinels, type=pa.string())),
                    False,
                ),
            )
        else:
            sentinel_mask = pc.fill_null(pc.equal(trimmed, "<NO_CONFIGURED_SENTINEL>"), False)
        self.sentinel += _true_count(sentinel_mask)
        eligible = pc.and_(nonblank, pc.invert(sentinel_mask))
        prefix = pc.utf8_slice_codeunits(trimmed, 0, 10)
        iso_date_shape = pc.fill_null(
            pc.match_substring_regex(trimmed, r"^\d{4}-\d{2}-\d{2}$"), False
        )
        iso_datetime_shape = pc.fill_null(
            pc.match_substring_regex(
                trimmed,
                r"^\d{4}-\d{2}-\d{2}[ T](?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):?[0-5]\d)?$",
            ),
            False,
        )
        iso_shape = pc.or_(iso_date_shape, iso_datetime_shape)
        parsed_iso = pc.strptime(prefix, format="%Y-%m-%d", unit="s", error_is_null=True)
        iso_calendar_valid = pc.fill_null(
            pc.equal(pc.strftime(parsed_iso, format="%Y-%m-%d"), prefix), False
        )
        iso_mask = pc.and_(pc.and_(eligible, iso_shape), iso_calendar_valid)
        mdy_shape = pc.fill_null(
            pc.match_substring_regex(trimmed, r"^\d{2}-\d{2}-\d{4}$"), False
        )
        parsed_mdy = pc.strptime(prefix, format="%m-%d-%Y", unit="s", error_is_null=True)
        mdy_calendar_valid = pc.fill_null(
            pc.equal(pc.strftime(parsed_mdy, format="%m-%d-%Y"), prefix), False
        )
        mdy_mask = pc.and_(pc.and_(eligible, mdy_shape), mdy_calendar_valid)
        self.by_format["YYYY-MM-DD"] += _true_count(iso_mask)
        self.by_format["MM-DD-YYYY"] += _true_count(mdy_mask)
        valid_mask = pc.or_(iso_mask, mdy_mask)
        parsed = pc.if_else(iso_mask, parsed_iso, parsed_mdy)
        valid_count = _true_count(valid_mask)
        self.valid += valid_count
        self.invalid += _true_count(pc.and_(eligible, pc.invert(valid_mask)))
        if not valid_count:
            return
        valid_dates = pc.filter(parsed, valid_mask)
        batch_min = pc.min(valid_dates).as_py()
        batch_max = pc.max(valid_dates).as_py()
        self.minimum = batch_min if self.minimum is None or batch_min < self.minimum else self.minimum
        self.maximum = batch_max if self.maximum is None or batch_max > self.maximum else self.maximum
        years = pc.year(valid_dates)
        for item in pc.value_counts(years).to_pylist():
            self.by_year[int(item["values"])] += int(item["counts"])

    def result(self) -> dict[str, Any]:
        blank = self.total - self.nonblank
        accounted = blank + self.sentinel + self.valid + self.invalid
        return {
            "column": self.name,
            "rows": self.total,
            "null_or_blank": blank,
            "open_ended_sentinel": self.sentinel,
            "valid_date_or_datetime": self.valid,
            "invalid_nonblank": self.invalid,
            "category_count_reconciles": accounted == self.total,
            "minimum": self.minimum.date().isoformat() if self.minimum else None,
            "maximum": self.maximum.date().isoformat() if self.maximum else None,
            "valid_counts_by_format": {
                name: self.by_format[name] for name in ("YYYY-MM-DD", "MM-DD-YYYY")
            },
            "counts_by_year": {str(year): self.by_year[year] for year in sorted(self.by_year)},
        }


class KeyAccumulator:
    def __init__(self, columns: tuple[str, ...], semantics: str):
        self.columns = columns
        self.semantics = semantics
        self.rows = 0
        self.blank_by_column: Counter[str] = Counter()
        self.trim_changes_by_column: Counter[str] = Counter()
        self.digit_only_by_column: Counter[str] = Counter()
        self.alphanumeric_by_column: Counter[str] = Counter()
        self.lengths_by_column: dict[str, Counter[int]] = {name: Counter() for name in columns}
        self.incomplete_rows = 0
        self.adjacent_duplicate_occurrences_all = 0
        self.adjacent_duplicate_occurrences_complete = 0
        self.previous_key: tuple[str | None, ...] | None = None

    def update(self, batch: pa.RecordBatch) -> None:
        arrays = [_trimmed(batch.column(name)) for name in self.columns]
        original = [batch.column(name) for name in self.columns]
        nrows = batch.num_rows
        self.rows += nrows
        complete = None
        for name, raw, trimmed in zip(self.columns, original, arrays):
            nonblank = _nonblank(raw)
            self.blank_by_column[name] += nrows - _true_count(nonblank)
            complete = nonblank if complete is None else pc.and_(complete, nonblank)
            changed = pc.and_(pc.is_valid(raw), pc.not_equal(raw, trimmed))
            self.trim_changes_by_column[name] += _true_count(changed)
            self.digit_only_by_column[name] += _true_count(
                pc.fill_null(pc.match_substring_regex(trimmed, r"^[0-9]+$"), False)
            )
            self.alphanumeric_by_column[name] += _true_count(
                pc.fill_null(pc.match_substring_regex(trimmed, r"^[A-Za-z0-9]+$"), False)
            )
            lengths = pc.filter(pc.utf8_length(trimmed), nonblank)
            for item in pc.value_counts(lengths).to_pylist():
                self.lengths_by_column[name][int(item["values"])] += int(item["counts"])
        assert complete is not None
        self.incomplete_rows += nrows - _true_count(complete)

        if nrows:
            first_key = tuple(array[0].as_py() for array in arrays)
            first_complete = bool(complete[0].as_py())
            if self.previous_key is not None and first_key == self.previous_key:
                self.adjacent_duplicate_occurrences_all += 1
                if first_complete and all(value not in (None, "") for value in self.previous_key):
                    self.adjacent_duplicate_occurrences_complete += 1

        if nrows > 1:
            same = None
            for array in arrays:
                left = pc.fill_null(array.slice(1), "<NULL>")
                right = pc.fill_null(array.slice(0, nrows - 1), "<NULL>")
                equal = pc.equal(left, right)
                same = equal if same is None else pc.and_(same, equal)
            assert same is not None
            self.adjacent_duplicate_occurrences_all += _true_count(same)
            complete_pairs = pc.and_(complete.slice(1), complete.slice(0, nrows - 1))
            self.adjacent_duplicate_occurrences_complete += _true_count(pc.and_(same, complete_pairs))

        if nrows:
            self.previous_key = tuple(array[nrows - 1].as_py() for array in arrays)

    def result(self) -> dict[str, Any]:
        return {
            "diagnostic_key_columns": list(self.columns),
            "key_semantics": self.semantics,
            "rows": self.rows,
            "grain_note": (
                "Grouping diagnostic only; these columns are explicitly nonunique and no natural raw line key exists."
                if self.semantics == "nonunique_filing_group_only"
                else "Candidate-grain diagnostic only; global uniqueness is not asserted in Phase 1."
            ),
            "rows_with_blank_diagnostic_key_fields": self.incomplete_rows,
            "blank_by_column": dict(self.blank_by_column),
            "trim_changes_by_column": dict(self.trim_changes_by_column),
            "digit_only_by_column": dict(self.digit_only_by_column),
            "alphanumeric_by_column": dict(self.alphanumeric_by_column),
            "length_histograms": {
                name: {str(length): counts[length] for length in sorted(counts)}
                for name, counts in self.lengths_by_column.items()
            },
            "adjacent_duplicate_occurrences_all": self.adjacent_duplicate_occurrences_all,
            "adjacent_duplicate_occurrences_complete": self.adjacent_duplicate_occurrences_complete,
        }


class InvalidRowCollector:
    def __init__(self, limit: int = 20):
        self.count = 0
        self.limit = limit
        self.examples: list[dict[str, Any]] = []

    def __call__(self, row: Any) -> str:
        self.count += 1
        if len(self.examples) < self.limit:
            raw_text = str(getattr(row, "text", ""))
            self.examples.append(
                {
                    "row_number": getattr(row, "number", None),
                    "expected_columns": getattr(row, "expected_columns", None),
                    "actual_columns": getattr(row, "actual_columns", None),
                    "row_text_sha256": hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest(),
                    "row_text_bytes": len(raw_text.encode("utf-8", errors="replace")),
                }
            )
        return "skip"


def scan_csv_semantics(path: Path, contract: FileContract, settings: PhaseSettings) -> dict[str, Any]:
    header = read_header(path)
    selected_columns = list(dict.fromkeys((*contract.date_columns, *contract.key_columns)))
    missing_selected = [name for name in selected_columns if name not in header]
    if missing_selected:
        current_stat = _stat_fingerprint(path)
        return {
            "parse_status": "failed_missing_profile_columns",
            "parsed_data_rows": 0,
            "malformed_rows": None,
            "error": f"Missing configured date/key columns: {missing_selected}",
            "date_coverage": [],
            "key_diagnostics": {},
            "stable_during_semantic_scan": True,
            "stat_before": current_stat,
            "stat_after": current_stat,
        }

    invalid_rows = InvalidRowCollector()
    read_options = pacsv.ReadOptions(
        block_size=int(settings.parser["block_size_mib"]) << 20,
        use_threads=True,
        encoding=str(settings.parser["encoding"]),
    )
    parse_options = pacsv.ParseOptions(
        delimiter=str(settings.parser["delimiter"]),
        quote_char=str(settings.parser["quote_char"]),
        double_quote=True,
        newlines_in_values=bool(settings.parser["allow_quoted_newlines"]),
        ignore_empty_lines=bool(settings.parser["ignore_empty_lines"]),
        invalid_row_handler=invalid_rows,
    )
    convert_options = pacsv.ConvertOptions(
        include_columns=selected_columns,
        column_types={name: pa.string() for name in selected_columns},
        strings_can_be_null=True,
        null_values=[""],
    )
    sentinel_registry = settings.quality.get("open_ended_date_sentinels", {})
    date_accumulators = {
        name: DateAccumulator(name, sentinel_registry.get(name, [])) for name in contract.date_columns
    }
    key_accumulator = KeyAccumulator(contract.key_columns, contract.key_semantics)
    parsed_rows = 0
    stat_before = _stat_fingerprint(path)
    next_row_report = 10_000_000
    try:
        reader = pacsv.open_csv(
            path,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
        for batch in reader:
            parsed_rows += batch.num_rows
            for name, accumulator in date_accumulators.items():
                accumulator.update(batch.column(name))
            key_accumulator.update(batch)
            if parsed_rows >= next_row_report:
                print(
                    json.dumps(
                        {"event": "parse_progress", "file": path.name, "rows": parsed_rows},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                while next_row_report <= parsed_rows:
                    next_row_report += 10_000_000
        parse_status = "passed" if invalid_rows.count == 0 else "failed_malformed_rows"
        error = None
    except Exception as exc:  # stored as evidence; caller converts this to a failed gate
        parse_status = "failed_exception"
        error = f"{type(exc).__name__}: {exc}"
    stat_after = _stat_fingerprint(path)

    return {
        "parse_status": parse_status,
        "parsed_data_rows": parsed_rows,
        "malformed_rows": invalid_rows.count,
        "malformed_examples": invalid_rows.examples,
        "error": error,
        "stable_during_semantic_scan": _same_scan_identity(stat_before, stat_after),
        "stat_before": stat_before,
        "stat_after": stat_after,
        "date_coverage": [date_accumulators[name].result() for name in contract.date_columns],
        "key_diagnostics": key_accumulator.result(),
    }


def _normalized_header_key(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold()


def _check(
    check_id: str,
    scope: str,
    status: str,
    severity: str,
    observed: Any,
    expected: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "scope": scope,
        "status": status,
        "severity": severity,
        "observed": observed,
        "expected": expected,
        "message": message,
    }


def profile_file(
    path: Path, contract: FileContract, settings: PhaseSettings
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    print(json.dumps({"event": "file_start", "file": contract.filename}, sort_keys=True), flush=True)
    byte_profile = scan_bytes(path, int(settings.parser["progress_gib"]))
    header_stat_before = _stat_fingerprint(path)
    header = read_header(path)
    sample_columns, sample_stats = sample_schema(path, header, int(settings.parser["sample_rows"]))
    header_stat_after = _stat_fingerprint(path)
    semantic = scan_csv_semantics(path, contract, settings)
    header_sample_stable = _same_scan_identity(header_stat_before, header_stat_after)
    same_file_across_passes = (
        _same_scan_identity(byte_profile["stat_after"], header_stat_before)
        and _same_scan_identity(header_stat_after, semantic["stat_before"])
    )

    duplicate_headers = sorted(name for name, count in Counter(header).items() if count > 1)
    normalized_counts = Counter(_normalized_header_key(name) for name in header)
    normalization_collisions = sorted(name for name, count in normalized_counts.items() if count > 1)
    blank_headers = [position for position, name in enumerate(header) if not name.strip()]
    missing_required = [name for name in contract.required_columns if name not in header]
    header_hash = _sha256_json(header)
    selected_roles = {
        column["name"]: {
            "is_required": column["name"] in contract.required_columns,
            "is_date": column["name"] in contract.date_columns,
            "is_diagnostic_key": column["name"] in contract.key_columns,
            "diagnostic_key_role": (
                "nonunique_grouping_field"
                if column["name"] in contract.key_columns
                and contract.key_semantics == "nonunique_filing_group_only"
                else "candidate_grain_field"
                if column["name"] in contract.key_columns
                else "not_applicable"
            ),
        }
        for column in sample_columns
    }
    for column in sample_columns:
        column.update(selected_roles[column["name"]])

    operational = {
        "physical_path": str(path.resolve()),
        "is_regular_file": path.is_file(),
        "is_symlink": path.is_symlink(),
        "byte_scan": {
            "stat_before": byte_profile["stat_before"],
            "stat_after": byte_profile["stat_after"],
            "filesystem_mtime_utc": byte_profile["filesystem_mtime_utc"],
            "filesystem_birthtime_utc": byte_profile["filesystem_birthtime_utc"],
            "stable": byte_profile["stable_during_byte_scan"],
        },
        "header_sample_scan": {
            "stat_before": header_stat_before,
            "stat_after": header_stat_after,
            "stable": header_sample_stable,
        },
        "semantic_scan": {
            "stat_before": semantic["stat_before"],
            "stat_after": semantic["stat_after"],
            "stable": semantic["stable_during_semantic_scan"],
        },
        "same_file_across_all_primary_passes": same_file_across_passes,
    }
    deterministic_byte_identity = {
        key: value
        for key, value in byte_profile.items()
        if key
        not in {
            "stat_before",
            "stat_after",
            "filesystem_mtime_utc",
            "filesystem_birthtime_utc",
            "stable_during_byte_scan",
        }
    }
    deterministic_semantic = {
        key: value
        for key, value in semantic.items()
        if key not in {"stat_before", "stat_after", "stable_during_semantic_scan"}
    }

    profile = {
        "filename": contract.filename,
        "family": contract.family,
        "role": contract.role,
        "grain": contract.grain,
        "byte_identity": deterministic_byte_identity,
        "csv": {
            "delimiter": settings.parser["delimiter"],
            "quote_char": settings.parser["quote_char"],
            "encoding": settings.parser["encoding"],
            "ordered_header": header,
            "column_count": len(header),
            "header_sha256": header_hash,
            "duplicate_headers": duplicate_headers,
            "normalization_collisions": normalization_collisions,
            "blank_header_positions": blank_headers,
            "missing_required_columns": missing_required,
            "sample_schema": sample_columns,
            "sample_stats": sample_stats,
        },
        "semantic_profile": deterministic_semantic,
        "coverage_date": contract.coverage_date,
        "coverage_date_semantics": "Coverage/profiling axis only; never automatically used as public_availability_date.",
        "availability_note": contract.availability_note,
        "date_columns": list(contract.date_columns),
        "diagnostic_key_columns": list(contract.key_columns),
        "key_semantics": contract.key_semantics,
        "source_creation_or_as_of": None,
        "source_creation_or_as_of_note": "No separate extraction provenance file was supplied; filesystem times and maximum data dates are not treated as source as-of dates.",
    }

    checks: list[dict[str, Any]] = []
    scope = contract.filename
    checks.append(_check("FILE_NONEMPTY", scope, "PASS" if byte_profile["size_bytes"] else "FAIL", "critical", byte_profile["size_bytes"], "> 0", "Raw file must be non-empty."))
    checks.append(_check("BYTE_SCAN_COMPLETE", scope, "PASS" if byte_profile["bytes_read"] == byte_profile["size_bytes"] else "FAIL", "critical", byte_profile["bytes_read"], byte_profile["size_bytes"], "Every byte must enter the SHA-256 calculation."))
    checks.append(_check("FILE_STABLE_BYTE_SCAN", scope, "PASS" if byte_profile["stable_during_byte_scan"] else "FAIL", "critical", byte_profile["stable_during_byte_scan"], True, "File identity, size, and mtime must not change during hashing."))
    checks.append(_check("FILE_STABLE_HEADER_SAMPLE_SCAN", scope, "PASS" if header_sample_stable else "FAIL", "critical", header_sample_stable, True, "File identity, size, and mtime must not change during header/sample inspection."))
    checks.append(_check("FILE_STABLE_SEMANTIC_SCAN", scope, "PASS" if semantic.get("stable_during_semantic_scan") else "FAIL", "critical", semantic.get("stable_during_semantic_scan"), True, "File identity, size, and mtime must not change during parsing."))
    checks.append(_check("SAME_FILE_ACROSS_PRIMARY_PASSES", scope, "PASS" if same_file_across_passes else "FAIL", "critical", same_file_across_passes, True, "Hash, header/sample, and semantic scans must describe the same inode, size, and mtime."))
    checks.append(_check("UTF8_VALID", scope, "PASS" if byte_profile["utf8_valid"] else "FAIL", "critical", byte_profile["utf8_valid"], True, "Configured UTF-8 encoding must decode without error."))
    checks.append(_check("NO_NUL_BYTES", scope, "PASS" if byte_profile["nul_bytes"] == 0 else "FAIL", "critical", byte_profile["nul_bytes"], 0, "CSV files must not contain NUL bytes."))
    checks.append(_check("HEADER_PRESENT", scope, "PASS" if header else "FAIL", "critical", len(header), "> 0", "A header row is required."))
    checks.append(_check("HEADER_NAMES_UNIQUE", scope, "PASS" if not duplicate_headers else "FAIL", "critical", duplicate_headers, [], "Exact duplicate column names are prohibited."))
    checks.append(_check("HEADER_NORMALIZATION_UNIQUE", scope, "PASS" if not normalization_collisions else "FAIL", "critical", normalization_collisions, [], "Case/Unicode-normalized column-name collisions are prohibited."))
    checks.append(_check("HEADER_NAMES_NONBLANK", scope, "PASS" if not blank_headers else "FAIL", "critical", blank_headers, [], "Blank column names are prohibited."))
    checks.append(_check("REQUIRED_COLUMNS_PRESENT", scope, "PASS" if not missing_required else "FAIL", "critical", missing_required, [], "All methodology-critical raw columns must be present."))
    checks.append(_check("CSV_PARSE", scope, "PASS" if semantic["parse_status"] == "passed" else "FAIL", "critical", semantic["parse_status"], "passed", semantic.get("error") or "CSV parser accepted all records."))
    checks.append(_check("MALFORMED_ROWS", scope, "PASS" if semantic.get("malformed_rows") == 0 else "FAIL", "critical", semantic.get("malformed_rows"), 0, "No malformed record may be silently used."))

    coverage = next((item for item in semantic.get("date_coverage", []) if item["column"] == contract.coverage_date), None)
    coverage_valid = bool(coverage and coverage["valid_date_or_datetime"] > 0)
    coverage_invalid = None if coverage is None else coverage["invalid_nonblank"]
    coverage_blank = None if coverage is None else coverage["null_or_blank"]
    coverage_sentinel = None if coverage is None else coverage["open_ended_sentinel"]
    date_accounting_failures = [
        item["column"]
        for item in semantic.get("date_coverage", [])
        if not item.get("category_count_reconciles", False)
    ]
    checks.append(_check("DATE_CATEGORY_ACCOUNTING", scope, "PASS" if not date_accounting_failures else "FAIL", "critical", date_accounting_failures, [], "Every configured date value must be classified exactly once as blank, sentinel, valid, or invalid."))
    checks.append(_check("COVERAGE_DATE_PRESENT", scope, "PASS" if coverage_valid else "FAIL", "critical", 0 if coverage is None else coverage["valid_date_or_datetime"], "> 0", "The configured coverage axis must have valid observations; it is not automatically public availability."))
    checks.append(_check("COVERAGE_DATE_PARSE", scope, "PASS" if coverage_invalid == 0 else "FAIL", "critical", coverage_invalid, 0, "Every nonblank coverage-axis value must match a complete configured date/datetime format."))
    checks.append(_check("COVERAGE_DATE_COMPLETE", scope, "PASS" if coverage_blank == 0 else "FAIL", "critical", coverage_blank, 0, "The coverage axis may not be blank."))
    checks.append(_check("COVERAGE_DATE_NO_SENTINEL", scope, "PASS" if coverage_sentinel == 0 else "FAIL", "critical", coverage_sentinel, 0, "Open-ended interval sentinels are prohibited in the coverage axis."))
    secondary_invalid = {item["column"]: item["invalid_nonblank"] for item in semantic.get("date_coverage", []) if item["column"] != contract.coverage_date and item["invalid_nonblank"]}
    checks.append(_check("SECONDARY_DATE_PARSE", scope, "PASS" if not secondary_invalid else "WARN", "warning", secondary_invalid, {}, "Invalid configured secondary dates require investigation before their fields are used."))

    physical_records = (
        byte_profile["lf_count"]
        + byte_profile["lone_cr_count"]
        + (0 if byte_profile["ends_with_line_break"] else 1)
    )
    logical_records = semantic.get("parsed_data_rows", 0) + semantic.get("malformed_rows", 0) + 1
    relation_status = "PASS" if physical_records == logical_records else "WARN"
    checks.append(_check("PHYSICAL_LOGICAL_RECORD_RELATION", scope, relation_status, "warning", {"physical_line_records": physical_records, "logical_csv_records": logical_records}, "equal unless quoted embedded newlines or blank physical records exist", "A mismatch is diagnostic, not automatically corruption, because quoted fields may contain newlines."))

    profile["primary_build_status"] = "failed" if any(item["status"] == "FAIL" for item in checks) else "passed"
    print(
        json.dumps(
            {
                "event": "file_complete",
                "file": contract.filename,
                "status": profile["primary_build_status"],
                "rows": semantic.get("parsed_data_rows"),
                "sha256": byte_profile["sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return profile, checks, operational


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(payload))
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    os.replace(temporary, path)


def _flatten_outputs(profiles: list[dict[str, Any]], checks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    columns: list[dict[str, Any]] = []
    dates: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    for profile in profiles:
        byte = profile["byte_identity"]
        semantic = profile["semantic_profile"]
        files.append(
            {
                "filename": profile["filename"],
                "family": profile["family"],
                "role": profile["role"],
                "grain": profile["grain"],
                "size_bytes": byte["size_bytes"],
                "sha256": byte["sha256"],
                "column_count": profile["csv"]["column_count"],
                "parsed_data_rows": semantic["parsed_data_rows"],
                "malformed_rows": semantic["malformed_rows"],
                "coverage_date": profile["coverage_date"],
                "coverage_date_semantics": profile["coverage_date_semantics"],
                "newline_style": byte["newline_style"],
                "utf8_valid": byte["utf8_valid"],
                "source_as_of": profile["source_creation_or_as_of"],
                "primary_build_status": profile["primary_build_status"],
            }
        )
        for column in profile["csv"]["sample_schema"]:
            columns.append({"filename": profile["filename"], **column})
        for date_item in semantic["date_coverage"]:
            dates.append({"filename": profile["filename"], "is_coverage_axis": date_item["column"] == profile["coverage_date"], **date_item})
        keys.append({"filename": profile["filename"], "grain": profile["grain"], **semantic["key_diagnostics"]})
    return {"files": files, "columns": columns, "dates": dates, "keys": keys, "checks": checks}


def _raw_snapshot_id(profiles: list[dict[str, Any]]) -> str:
    identity = [
        {
            "filename": item["filename"],
            "size_bytes": item["byte_identity"]["size_bytes"],
            "sha256": item["byte_identity"]["sha256"],
        }
        for item in sorted(profiles, key=lambda row: row["filename"])
    ]
    return "raw-v1-" + _sha256_json(identity)[:16]


def _build_id(
    raw_snapshot_id: str,
    configuration_hash: str,
    implementation_hash: str,
    methodology_freeze_sha256: str,
    engine_identity: dict[str, str],
) -> str:
    identity = {
        "raw_snapshot_id": raw_snapshot_id,
        "configuration_hash": configuration_hash,
        "implementation_sha256": implementation_hash,
        "methodology_freeze_sha256": methodology_freeze_sha256,
        "engine_identity": engine_identity,
    }
    return "phase01-" + _sha256_json(identity)[:16]


def build_snapshot(
    settings: PhaseSettings,
    only: set[str] | None = None,
    publish: bool = True,
) -> Path:
    started = dt.datetime.now(dt.timezone.utc)
    source_path = Path(__file__).resolve()
    implementation_hash = _sha256_file(source_path)
    methodology_verification = verify_methodology_freeze(settings.methodology_freeze)
    engine_identity = {
        "python": platform.python_version(),
        "pyarrow": pa.__version__,
        "csv_parser": "pyarrow.csv",
    }
    settings.staging_directory.mkdir(parents=True, exist_ok=True)
    attempt_id = started.strftime("attempt-%Y%m%dT%H%M%S%fZ") + f"-p{os.getpid()}"
    attempt_directory = settings.staging_directory / attempt_id
    checkpoint_directory = attempt_directory / "file_profiles"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    expected = {contract.filename for contract in settings.contracts}
    physical_csv = {path.name for path in settings.raw_root.glob("*.csv") if path.is_file()}
    missing = sorted(expected - physical_csv)
    unexpected = sorted(physical_csv - expected)
    selected_contracts = [contract for contract in settings.contracts if only is None or contract.filename in only]
    unknown_requested = sorted((only or set()) - expected)
    if unknown_requested:
        raise ValueError(f"Requested files are not in the frozen contract: {unknown_requested}")

    global_checks = [
        _check("METHODOLOGY_FREEZE_VERIFIED", "snapshot", "PASS", "critical", methodology_verification["freeze_id"], "frozen pre-implementation methodology with matching artifact hashes", "The configured methodology freeze and all six frozen artifacts were independently hash-verified before the data scan."),
        _check("EXPECTED_CONTRACT_COUNT", "snapshot", "PASS" if len(expected) == settings.expected_file_count else "FAIL", "critical", len(expected), settings.expected_file_count, "Contract registry must contain the methodology-declared number of files."),
        _check("PHYSICAL_CSV_COUNT", "snapshot", "PASS" if len(physical_csv) == settings.expected_file_count else "FAIL", "critical", len(physical_csv), settings.expected_file_count, "Raw directory must contain exactly the registered CSV set."),
        _check("MISSING_EXPECTED_FILES", "snapshot", "PASS" if not missing else "FAIL", "critical", missing, [], "Every required raw CSV must be present."),
        _check("UNEXPECTED_CSV_FILES", "snapshot", "PASS" if not unexpected else "FAIL", "critical", unexpected, [], "Unexpected CSVs are not silently ingested into the frozen snapshot."),
    ]
    if missing and only is None:
        raise FileNotFoundError(f"Missing raw files: {missing}")

    profiles: list[dict[str, Any]] = []
    checks = list(global_checks)
    durations: dict[str, float] = {}
    operational_by_file: dict[str, Any] = {}
    for contract in selected_contracts:
        path = settings.raw_root / contract.filename
        if not path.exists():
            continue
        file_started = time.monotonic()
        profile, file_checks, operational = profile_file(path, contract, settings)
        durations[contract.filename] = round(time.monotonic() - file_started, 3)
        profiles.append(profile)
        checks.extend(file_checks)
        operational_by_file[contract.filename] = operational
        _write_json(
            checkpoint_directory / f"{contract.filename}.json",
            {
                "configuration_hash": settings.configuration_hash,
                "profile": profile,
                "checks": file_checks,
                "operational": operational,
            },
        )

    profiles.sort(key=lambda item: item["filename"])
    physical_csv_at_end = {
        path.name for path in settings.raw_root.glob("*.csv") if path.is_file()
    }
    checks.append(
        _check(
            "PHYSICAL_CSV_REGISTRY_STABLE_DURING_RUN",
            "snapshot",
            "PASS" if physical_csv_at_end == physical_csv else "FAIL",
            "critical",
            sorted(physical_csv_at_end),
            sorted(physical_csv),
            "No raw CSV may be added, removed, or renamed while the primary scan is running.",
        )
    )
    for profile in profiles:
        filename = profile["filename"]
        expected_stat = operational_by_file[filename]["semantic_scan"]["stat_after"]
        try:
            final_stat = _stat_fingerprint(settings.raw_root / filename)
            stable_through_run = _same_scan_identity(expected_stat, final_stat)
        except OSError as exc:
            final_stat = {"error": f"{type(exc).__name__}: {exc}"}
            stable_through_run = False
        operational_by_file[filename]["end_of_primary_run_stat"] = final_stat
        operational_by_file[filename]["stable_through_entire_primary_run"] = stable_through_run
        checks.append(
            _check(
                "FILE_STABLE_THROUGH_ENTIRE_PRIMARY_RUN",
                filename,
                "PASS" if stable_through_run else "FAIL",
                "critical",
                stable_through_run,
                True,
                "The file fingerprint at publication time must match the file that completed semantic parsing.",
            )
        )
        if not stable_through_run:
            profile["primary_build_status"] = "failed"
    implementation_hash_after = _sha256_file(source_path)
    settings_after = load_settings(
        settings.phase_config_path, settings.paths_config_path
    )
    methodology_verification_after = verify_methodology_freeze(settings.methodology_freeze)
    checks.extend(
        [
            _check("IMPLEMENTATION_STABLE_DURING_RUN", "snapshot", "PASS" if implementation_hash_after == implementation_hash else "FAIL", "critical", implementation_hash_after, implementation_hash, "Primary implementation source must not change during a scan."),
            _check("CONFIGURATION_STABLE_DURING_RUN", "snapshot", "PASS" if settings_after.configuration_hash == settings.configuration_hash else "FAIL", "critical", settings_after.configuration_hash, settings.configuration_hash, "Phase configuration and raw contracts must not change during a scan."),
            _check("PATH_CONFIGURATION_STABLE_DURING_RUN", "snapshot", "PASS" if settings_after.path_configuration_hash == settings.path_configuration_hash else "FAIL", "critical", settings_after.path_configuration_hash, settings.path_configuration_hash, "The local database pointer configuration must not change during a scan."),
            _check("METHODOLOGY_FREEZE_STABLE_DURING_RUN", "snapshot", "PASS" if methodology_verification_after == methodology_verification else "FAIL", "critical", methodology_verification_after, methodology_verification, "The verified methodology freeze must not change during a scan."),
        ]
    )
    checks.sort(key=lambda item: (item["scope"], item["check_id"]))
    raw_snapshot_id = _raw_snapshot_id(profiles)
    build_id = _build_id(
        raw_snapshot_id,
        settings.configuration_hash,
        implementation_hash,
        methodology_verification["freeze_manifest_sha256"],
        engine_identity,
    )
    has_failure = any(item["status"] == "FAIL" for item in checks)
    complete_scope = only is None and len(profiles) == settings.expected_file_count
    primary_status = (
        "primary_passed_pending_independent_verification"
        if complete_scope and not has_failure
        else "primary_failed_or_partial"
    )
    manifest = {
        "schema_version": 2,
        "raw_snapshot_id": raw_snapshot_id,
        "phase_01_build_id": build_id,
        "methodology_freeze": methodology_verification,
        "implementation_sha256": implementation_hash,
        "engine_identity": engine_identity,
        "configuration_hash": settings.configuration_hash,
        "contract_hashes": settings.contract_hashes,
        "raw_identity": {
            "expected_files": settings.expected_file_count,
            "profiled_files": len(profiles),
            "missing_files": missing,
            "unexpected_csv_files": unexpected,
        },
        "primary_status": primary_status,
        "phase_01_status": "primary_only_incomplete_until_separate_independent_verification_and_freeze",
        "files": profiles,
        "scope_boundary": {
            "verified_here": ["raw byte identity", "CSV structure", "parseability", "configured date coverage", "diagnostic-key formatting"],
            "not_yet_verified": ["point-in-time joins", "availability lags", "canonical panel uniqueness", "return reconciliation", "feature missingness", "leakage safety"],
        },
    }
    flattened = _flatten_outputs(profiles, checks)
    bundle_directory = attempt_directory / build_id
    bundle_directory.mkdir(parents=True, exist_ok=False)

    _write_json(bundle_directory / "manifest.json", manifest)
    _write_csv(bundle_directory / "files.csv", flattened["files"], list(flattened["files"][0]) if flattened["files"] else ["filename"])
    _write_csv(bundle_directory / "columns.csv", flattened["columns"], list(flattened["columns"][0]) if flattened["columns"] else ["filename"])
    _write_csv(bundle_directory / "date_coverage.csv", flattened["dates"], list(flattened["dates"][0]) if flattened["dates"] else ["filename"])
    _write_csv(bundle_directory / "key_diagnostics.csv", flattened["keys"], list(flattened["keys"][0]) if flattened["keys"] else ["filename"])
    _write_csv(bundle_directory / "quality_checks.csv", checks, ["check_id", "scope", "status", "severity", "observed", "expected", "message"])
    with (bundle_directory / "hashes.sha256").open("w", encoding="utf-8") as handle:
        for profile in profiles:
            handle.write(f"{profile['byte_identity']['sha256']}  {profile['filename']}\n")

    completed = dt.datetime.now(dt.timezone.utc)
    run_metadata = {
        "raw_snapshot_id": raw_snapshot_id,
        "phase_01_build_id": build_id,
        "attempt_id": attempt_id,
        "run_started_utc": started.isoformat(),
        "run_completed_utc": completed.isoformat(),
        "elapsed_seconds": round((completed - started).total_seconds(), 3),
        "per_file_elapsed_seconds": durations,
        "per_file_operational_metadata": operational_by_file,
        "database_root": str(settings.database_root),
        "raw_root": str(settings.raw_root),
        "python": sys.version,
        "platform": platform.platform(),
        "pyarrow": pa.__version__,
        "implementation_sha256": implementation_hash,
        "configuration_hash": settings.configuration_hash,
        "path_configuration_sha256": settings.path_configuration_hash,
        "methodology_freeze_sha256": methodology_verification["freeze_manifest_sha256"],
        "command_scope": "all" if only is None else sorted(only),
    }
    _write_json(bundle_directory / "run_metadata.json", run_metadata)

    should_publish = publish and complete_scope and not has_failure
    if should_publish:
        settings.final_directory.mkdir(parents=True, exist_ok=True)
        final_directory = settings.final_directory / build_id
        if final_directory.exists():
            raise FileExistsError(
                f"Immutable Phase 1 build already exists: {final_directory}. Refusing to overwrite it."
            )
        os.replace(bundle_directory, final_directory)
        output_directory = final_directory
        _write_json(
            settings.artifact_root / "LATEST_PRIMARY.json",
            {
                "raw_snapshot_id": raw_snapshot_id,
                "phase_01_build_id": build_id,
                "path": str(output_directory.relative_to(settings.artifact_root)),
                "primary_status": primary_status,
                "verification_status": "not_run",
            },
        )
    else:
        output_directory = bundle_directory

    _write_json(
        settings.artifact_root / "LATEST_ATTEMPT.json",
        {
            "raw_snapshot_id": raw_snapshot_id,
            "phase_01_build_id": build_id,
            "path": str(output_directory.relative_to(settings.artifact_root)),
            "primary_status": primary_status,
            "published_as_primary": should_publish,
        },
    )
    print(
        json.dumps(
            {
                "event": "snapshot_primary_complete",
                "raw_snapshot_id": raw_snapshot_id,
                "phase_01_build_id": build_id,
                "status": primary_status,
                "published": should_publish,
                "output_directory": str(output_directory),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return output_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-config", type=Path, default=DEFAULT_PHASE_CONFIG)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--only", nargs="*", help="Smoke-test selected registered filenames")
    parser.add_argument("--no-publish", action="store_true", help="Keep output under staging")
    args = parser.parse_args(argv)
    settings = load_settings(args.phase_config, args.paths_config)
    output_directory = build_snapshot(
        settings,
        only=set(args.only) if args.only else None,
        publish=not args.no_publish,
    )
    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    published = (
        output_directory.parent.resolve() == settings.final_directory.resolve()
        and manifest.get("primary_status")
        == "primary_passed_pending_independent_verification"
    )
    return 0 if published else 2


if __name__ == "__main__":
    raise SystemExit(main())
