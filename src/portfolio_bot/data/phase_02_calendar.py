"""Build the observed CRSP trading-date and rebalance calendars.

This is the first raw-reading Phase 2 slice.  It streams only ``DlyCalDt``;
it cannot construct returns, features, labels, portfolios, or performance.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import stat as stat_module
import sys
import tempfile
import time
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASE_CONFIG = PROJECT_ROOT / "configs" / "phase_02_canonical_panel.toml"
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.toml"
DEFAULT_VERIFIED_POINTER = (
    PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel" / "LATEST_CONTRACT_VERIFIED.json"
)
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "phase_02_canonical_panel"
BYTE_CHUNK_SIZE = 8 << 20


class CalendarBuildError(RuntimeError):
    """Raised when a calendar gate fails or a sealed output is inconsistent."""


@dataclasses.dataclass(frozen=True)
class CalendarSettings:
    phase_config_path: Path
    paths_config_path: Path
    verified_pointer_path: Path
    phase_config: dict[str, Any]
    database_root: Path
    raw_path: Path
    interim_root: Path
    quality_root: Path
    artifact_root: Path
    run_root: Path
    block_size_bytes: int = 64 << 20


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = BYTE_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def project_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalendarBuildError(f"Expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _wire(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return str(value)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _wire(row.get(field)) for field in fields})


def load_settings(
    phase_config_path: Path = DEFAULT_PHASE_CONFIG,
    paths_config_path: Path = DEFAULT_PATHS_CONFIG,
    *,
    verified_pointer_path: Path = DEFAULT_VERIFIED_POINTER,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    run_root: Path = DEFAULT_RUN_ROOT,
) -> CalendarSettings:
    phase_config_path = phase_config_path.resolve()
    paths_config_path = paths_config_path.resolve()
    with phase_config_path.open("rb") as handle:
        phase = tomllib.load(handle)
    with paths_config_path.open("rb") as handle:
        paths = tomllib.load(handle)
    configured = Path(paths["database"]["root"]).expanduser()
    database_root = Path(os.environ.get("PORTFOLIO_DATABASE_ROOT", configured)).expanduser().resolve()
    raw_root = database_root / paths["database"]["raw"]
    namespace = phase["storage"]["interim_namespace"]
    return CalendarSettings(
        phase_config_path=phase_config_path,
        paths_config_path=paths_config_path,
        verified_pointer_path=verified_pointer_path.resolve(),
        phase_config=phase,
        database_root=database_root,
        raw_path=(raw_root / phase["calendar"]["calendar_owner_file"]).resolve(),
        interim_root=(database_root / paths["database"]["interim"] / namespace).resolve(),
        quality_root=(database_root / paths["database"]["quality"] / namespace).resolve(),
        artifact_root=artifact_root.resolve(),
        run_root=run_root.resolve(),
    )


def operational_stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "size_bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
    }


def authenticate_verified_contract(settings: CalendarSettings) -> dict[str, Any]:
    pointer = _read_json(settings.verified_pointer_path)
    required = {
        "contract_id",
        "contract_path",
        "verification_id",
        "verification_path",
        "contract_json_sha256",
        "artifact_hashes_sha256",
        "independent_check_sha256",
        "differences_sha256",
        "status",
    }
    missing = sorted(required - set(pointer))
    if missing:
        raise CalendarBuildError(f"Verified contract pointer is missing fields: {missing}")
    if pointer["status"] != "verified_passed":
        raise CalendarBuildError(
            f"Verified contract pointer status is not verified_passed: {pointer['status']!r}"
        )
    contract_dir = project_path(pointer["contract_path"])
    verification_dir = project_path(pointer["verification_path"])
    expected_files = {
        contract_dir / "contract.json": pointer["contract_json_sha256"],
        contract_dir / "artifact_hashes.sha256": pointer["artifact_hashes_sha256"],
        verification_dir / "independent_check.json": pointer["independent_check_sha256"],
        verification_dir / "differences.csv": pointer["differences_sha256"],
    }
    mismatches = {
        project_label(path): None if not path.is_file() else sha256_file(path)
        for path, expected in expected_files.items()
        if not path.is_file() or sha256_file(path) != expected
    }
    if mismatches:
        raise CalendarBuildError(f"Verified contract pointer hash mismatch: {mismatches}")
    contract = _read_json(contract_dir / "contract.json")
    independent = _read_json(verification_dir / "independent_check.json")
    if contract.get("contract_id") != pointer["contract_id"]:
        raise CalendarBuildError("Contract ID does not match verified pointer")
    if independent.get("contract_id") != pointer["contract_id"]:
        raise CalendarBuildError("Independent verification contract ID mismatch")
    if independent.get("verification_id") != pointer["verification_id"]:
        raise CalendarBuildError("Independent verification ID mismatch")
    if independent.get("status") != "verified_passed":
        raise CalendarBuildError("Independent verification is not a pass")
    if independent.get("summary", {}).get("fail_count") != 0:
        raise CalendarBuildError("Independent verification recorded failures")
    if independent.get("summary", {}).get("difference_count") != 0:
        raise CalendarBuildError("Independent verification recorded differences")
    if independent.get("independence", {}).get("raw_csv_rows_read") != 0:
        raise CalendarBuildError("Contract verification unexpectedly read raw rows")
    source_files = _read_csv(contract_dir / "source_files.csv")
    boundaries = _read_csv(contract_dir / "calendar_boundary_expectations.csv")
    return {
        "pointer": pointer,
        "contract": contract,
        "independent_check": independent,
        "contract_dir": contract_dir,
        "verification_dir": verification_dir,
        "source_files": source_files,
        "boundaries": boundaries,
    }


def frozen_raw_identity(settings: CalendarSettings) -> dict[str, Any]:
    freeze_manifest = project_path(settings.phase_config["phase_01_freeze_manifest"])
    recheck_path = freeze_manifest.parent / "raw_freeze_recheck_metadata.json"
    recheck = _read_json(recheck_path)
    filename = settings.raw_path.name
    try:
        raw = recheck["per_file_operational_metadata"][filename]["end_of_complete_recheck_stat"]
    except KeyError as exc:
        raise CalendarBuildError(f"Frozen raw operational identity missing for {filename}") from exc
    fields = ("device", "inode", "mode", "size_bytes", "mtime_ns")
    expected = {field: int(raw[field]) for field in fields}
    return {
        "path": recheck_path,
        "sha256": sha256_file(recheck_path),
        "expected_stat": expected,
        "freeze_recheck_status": recheck.get("status"),
        "all_hashes_match_primary": recheck.get("all_hashes_match_primary"),
        "all_files_stable": recheck.get("all_files_stable_through_entire_recheck"),
    }


def scan_observed_dates(
    path: Path,
    *,
    date_column: str = "DlyCalDt",
    block_size_bytes: int = 64 << 20,
) -> dict[str, Any]:
    """Stream one CSV column and return strict distinct-date diagnostics."""
    started = time.monotonic()
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(
            block_size=block_size_bytes,
            use_threads=True,
            encoding="utf8",
        ),
        parse_options=pacsv.ParseOptions(
            delimiter=",",
            quote_char='"',
            double_quote=True,
            newlines_in_values=True,
            ignore_empty_lines=False,
        ),
        convert_options=pacsv.ConvertOptions(
            include_columns=[date_column],
            column_types={date_column: pa.string()},
            strings_can_be_null=False,
        ),
    )
    row_count = 0
    null_count = 0
    blank_count = 0
    distinct_text: set[str] = set()
    for batch in reader:
        array = batch.column(0)
        row_count += len(array)
        null_count += array.null_count
        trimmed = pc.utf8_trim_whitespace(array)
        blank_mask = pc.equal(trimmed, "")
        blank_count += int(pc.sum(pc.cast(blank_mask, pa.int64())).as_py() or 0)
        for value in pc.unique(trimmed).to_pylist():
            if value is not None and value:
                distinct_text.add(value)

    valid_dates: list[dt.date] = []
    invalid_values: list[str] = []
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for value in sorted(distinct_text):
        if not date_pattern.fullmatch(value):
            invalid_values.append(value)
            continue
        try:
            valid_dates.append(dt.date.fromisoformat(value))
        except ValueError:
            invalid_values.append(value)
    valid_dates.sort()
    return {
        "row_count": row_count,
        "null_count": null_count,
        "blank_count": blank_count,
        "distinct_nonblank_count": len(distinct_text),
        "valid_distinct_date_count": len(valid_dates),
        "invalid_distinct_values": invalid_values,
        "dates": valid_dates,
        "minimum": valid_dates[0].isoformat() if valid_dates else None,
        "maximum": valid_dates[-1].isoformat() if valid_dates else None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "selected_columns": [date_column],
    }


def _next_month(month: tuple[int, int]) -> tuple[int, int]:
    year, number = month
    return (year + 1, 1) if number == 12 else (year, number + 1)


def _holding_stage(holding_month: str) -> str:
    if holding_month <= "2014-11":
        return "development"
    if holding_month == "2014-12":
        return "embargo"
    if holding_month <= "2019-11":
        return "validation"
    if holding_month == "2019-12":
        return "embargo"
    if holding_month <= "2025-08":
        return "primary_final"
    if holding_month <= "2025-11":
        return "core_extension"
    if holding_month == "2025-12":
        return "extension_descriptive"
    return "outside_locked_scope"


def derive_calendars(dates: Sequence[dt.date]) -> dict[str, Any]:
    ordered = sorted(set(dates))
    by_month: dict[tuple[int, int], list[dt.date]] = defaultdict(list)
    for date in ordered:
        by_month[(date.year, date.month)].append(date)
    months = sorted(by_month)
    month_continuity = all(right == _next_month(left) for left, right in zip(months, months[1:]))
    month_end_set = {values[-1] for values in by_month.values()}
    trading_rows = [
        {
            "calendar_date": date,
            "year": date.year,
            "month": date.month,
            "weekday_iso": date.isoweekday(),
            "is_observed_month_end": date in month_end_set,
        }
        for date in ordered
    ]

    rebalance_rows: list[dict[str, Any]] = []
    for index in range(1, len(months)):
        holding_key = months[index]
        holding_month = f"{holding_key[0]:04d}-{holding_key[1]:02d}"
        signal_date = by_month[months[index - 1]][-1]
        entry_date = by_month[holding_key][0]
        is_last = index == len(months) - 1
        if is_last:
            exit_date = by_month[holding_key][-1]
            is_partial = True
        else:
            exit_date = by_month[months[index + 1]][0]
            is_partial = False
        stage = _holding_stage(holding_month)
        rebalance_rows.append(
            {
                "holding_month": holding_month,
                "stage": stage,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "is_embargo": stage == "embargo",
                "is_partial": is_partial,
                "full_feature_time_eligible": holding_month <= "2025-08" and stage != "embargo",
                "formal_inference_expected": stage == "primary_final" and not is_partial,
            }
        )
    return {
        "trading_rows": trading_rows,
        "rebalance_rows": rebalance_rows,
        "month_count": len(months),
        "month_continuity": month_continuity,
        "duplicate_input_dates_removed": len(dates) - len(ordered),
    }


def _check(
    rows: list[dict[str, Any]],
    check_id: str,
    *,
    passed: bool,
    expected: Any,
    actual: Any,
    evidence: str,
    message: str,
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "status": "PASS" if passed else "FAIL",
            "expected": expected,
            "actual": actual,
            "evidence": evidence,
            "message": message,
        }
    )


def validate_calendar(
    settings: CalendarSettings,
    authenticated: Mapping[str, Any],
    frozen: Mapping[str, Any],
    stat_before: Mapping[str, int],
    stat_after: Mapping[str, int],
    scan: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    phase = settings.phase_config
    calendar = phase["calendar"]
    source_rows = {
        row["filename"]: row for row in authenticated["source_files"]
    }
    daily = source_rows[settings.raw_path.name]

    _check(
        checks,
        "VERIFIED_CONTRACT_GATE",
        passed=authenticated["pointer"]["status"] == "verified_passed",
        expected="verified_passed",
        actual=authenticated["pointer"]["status"],
        evidence=project_label(settings.verified_pointer_path),
        message="Calendar build consumes only the independently verified contract.",
    )
    _check(
        checks,
        "RAW_FROZEN_IDENTITY_BEFORE_SCAN",
        passed=dict(stat_before) == frozen["expected_stat"],
        expected=frozen["expected_stat"],
        actual=dict(stat_before),
        evidence=project_label(frozen["path"]),
        message="Device, inode, mode, size, and mtime match the final Phase 1 raw recheck.",
    )
    _check(
        checks,
        "RAW_IDENTITY_STABLE_DURING_SCAN",
        passed=dict(stat_after) == dict(stat_before),
        expected=dict(stat_before),
        actual=dict(stat_after),
        evidence=settings.raw_path.as_posix(),
        message="The raw file did not change while DlyCalDt was streamed.",
    )
    expected_rows = int(daily["parsed_data_rows"])
    _check(
        checks,
        "SCANNED_ROW_COUNT_MATCHES_PHASE_1",
        passed=scan["row_count"] == expected_rows,
        expected=expected_rows,
        actual=scan["row_count"],
        evidence="verified contract source_files.csv",
        message="Every raw security-day row contributed to the date-column scan.",
    )
    _check(
        checks,
        "DATE_COLUMN_ONLY",
        passed=scan["selected_columns"] == [calendar["calendar_date_column"]],
        expected=[calendar["calendar_date_column"]],
        actual=scan["selected_columns"],
        evidence="calendar scanner receipt",
        message="No return, price, feature, target, or performance column was read.",
    )
    _check(
        checks,
        "DATE_VALUES_COMPLETE_AND_VALID",
        passed=(
            scan["null_count"] == 0
            and scan["blank_count"] == 0
            and scan["invalid_distinct_values"] == []
        ),
        expected={"null_count": 0, "blank_count": 0, "invalid_distinct_values": []},
        actual={
            "null_count": scan["null_count"],
            "blank_count": scan["blank_count"],
            "invalid_distinct_values": scan["invalid_distinct_values"],
        },
        evidence=settings.raw_path.name,
        message="Observed calendar dates are strictly parseable ISO dates.",
    )
    expected_range = {
        "minimum": calendar["expected_observed_minimum"],
        "maximum": calendar["expected_observed_maximum"],
    }
    actual_range = {"minimum": scan["minimum"], "maximum": scan["maximum"]}
    _check(
        checks,
        "OBSERVED_DATE_COVERAGE_EXACT",
        passed=actual_range == expected_range,
        expected=expected_range,
        actual=actual_range,
        evidence="Phase 2 calendar contract and complete DlyCalDt scan",
        message="The observed CRSP calendar covers the locked raw date range.",
    )
    ordered_dates = scan["dates"]
    strict_unique = all(left < right for left, right in zip(ordered_dates, ordered_dates[1:]))
    _check(
        checks,
        "TRADING_CALENDAR_STRICTLY_INCREASING_UNIQUE",
        passed=strict_unique and derived["duplicate_input_dates_removed"] == 0,
        expected={"strictly_increasing": True, "duplicate_input_dates_removed": 0},
        actual={
            "strictly_increasing": strict_unique,
            "duplicate_input_dates_removed": derived["duplicate_input_dates_removed"],
        },
        evidence="derived trading_calendar",
        message="The canonical date calendar has one ordered row per observed trading date.",
    )
    _check(
        checks,
        "CALENDAR_MONTHS_CONTIGUOUS",
        passed=derived["month_continuity"],
        expected=True,
        actual=derived["month_continuity"],
        evidence="derived trading_calendar",
        message="Every calendar month in the frozen coverage has at least one observed trading date.",
    )

    by_holding = {row["holding_month"]: row for row in derived["rebalance_rows"]}
    for boundary in authenticated["boundaries"]:
        holding = boundary["holding_month"]
        actual = by_holding.get(holding)
        actual_boundary = None
        if actual:
            actual_boundary = {
                "stage": actual["stage"],
                "signal_date": actual["signal_date"].isoformat(),
                "entry_date": actual["entry_date"].isoformat(),
                "exit_date": actual["exit_date"].isoformat(),
                "is_embargo": actual["is_embargo"],
                "is_partial": actual["is_partial"],
                "formal_inference_expected": actual["formal_inference_expected"],
            }
        expected_boundary = {
            "stage": boundary["stage"],
            "signal_date": boundary["signal_date"],
            "entry_date": boundary["entry_date"],
            "exit_date": boundary["exit_date"],
            "is_embargo": boundary["is_embargo"].lower() == "true",
            "is_partial": boundary["is_partial"].lower() == "true",
            "formal_inference_expected": boundary["formal_inference_expected"].lower()
            == "true",
        }
        _check(
            checks,
            "BOUNDARY_" + boundary["boundary_id"].upper(),
            passed=actual_boundary == expected_boundary,
            expected=expected_boundary,
            actual=actual_boundary,
            evidence="verified calendar_boundary_expectations.csv and observed DlyCalDt calendar",
            message=f"Locked {holding} stage, dates, and inference flags reproduce exactly.",
        )

    partial = by_holding.get("2025-12", {})
    _check(
        checks,
        "PARTIAL_DECEMBER_EXCLUDED_FROM_FORMAL_INFERENCE",
        passed=(
            partial.get("is_partial") is True
            and partial.get("formal_inference_expected") is False
            and partial.get("exit_date") == dt.date(2025, 12, 31)
        ),
        expected={"is_partial": True, "formal_inference_expected": False, "exit_date": "2025-12-31"},
        actual={
            "is_partial": partial.get("is_partial"),
            "formal_inference_expected": partial.get("formal_inference_expected"),
            "exit_date": _wire(partial.get("exit_date")),
        },
        evidence="derived rebalance_calendar",
        message="The administratively closed December interval remains descriptive only.",
    )
    return sorted(checks, key=lambda row: row["check_id"])


def compile_calendar_build(settings: CalendarSettings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    authenticated = authenticate_verified_contract(settings)
    frozen = frozen_raw_identity(settings)
    if not settings.raw_path.is_file():
        raise CalendarBuildError(f"Missing raw calendar owner file: {settings.raw_path}")
    stat_before = operational_stat(settings.raw_path)
    if stat_before != frozen["expected_stat"]:
        raise CalendarBuildError(
            "Raw identity differs from the Phase 1 final recheck before scan: "
            f"expected {frozen['expected_stat']}, actual {stat_before}"
        )
    scan = scan_observed_dates(
        settings.raw_path,
        date_column=settings.phase_config["calendar"]["calendar_date_column"],
        block_size_bytes=settings.block_size_bytes,
    )
    stat_after = operational_stat(settings.raw_path)
    derived = derive_calendars(scan["dates"])
    checks = validate_calendar(
        settings,
        authenticated,
        frozen,
        stat_before,
        stat_after,
        scan,
        derived,
    )
    failures = [row for row in checks if row["status"] == "FAIL"]
    if failures:
        raise CalendarBuildError(
            "Calendar validation failed: "
            + "; ".join(f"{row['check_id']}: {row['message']}" for row in failures)
        )

    daily = next(
        row for row in authenticated["source_files"] if row["filename"] == settings.raw_path.name
    )
    binding = {
        "schema_version": 1,
        "contract_id": authenticated["pointer"]["contract_id"],
        "verification_id": authenticated["pointer"]["verification_id"],
        "contract_json_sha256": authenticated["pointer"]["contract_json_sha256"],
        "independent_check_sha256": authenticated["pointer"]["independent_check_sha256"],
        "phase_config_sha256": sha256_file(settings.phase_config_path),
        "paths_config_sha256": sha256_file(settings.paths_config_path),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "raw_file": settings.raw_path.name,
        "raw_sha256_from_frozen_phase_1": daily["sha256"],
        "raw_size_bytes": int(daily["size_bytes"]),
        "date_column": settings.phase_config["calendar"]["calendar_date_column"],
        "scanner": {"engine": "pyarrow.csv.open_csv", "pyarrow": pa.__version__},
    }
    build_digest = sha256_json(binding)
    build_id = f"calendar-{build_digest[:16]}"
    scan_public = {key: value for key, value in scan.items() if key != "dates"}
    summary = {
        "schema_version": 1,
        "build_id": build_id,
        "contract_id": binding["contract_id"],
        "verification_id": binding["verification_id"],
        "status": "calendar_primary_passed_pending_independent_verification",
        "raw_rows_scanned": scan["row_count"],
        "raw_columns_read": scan["selected_columns"],
        "observed_trading_dates": len(derived["trading_rows"]),
        "rebalance_intervals": len(derived["rebalance_rows"]),
        "complete_rebalance_intervals": sum(
            not row["is_partial"] for row in derived["rebalance_rows"]
        ),
        "partial_rebalance_intervals": sum(
            row["is_partial"] for row in derived["rebalance_rows"]
        ),
        "minimum_date": scan["minimum"],
        "maximum_date": scan["maximum"],
        "check_count": len(checks),
        "pass_count": len(checks),
        "fail_count": 0,
        "targets_built": False,
        "features_built": False,
        "portfolios_built": False,
        "performance_results_seen": False,
    }
    input_receipt = {
        "schema_version": 1,
        "binding": binding,
        "verified_pointer": authenticated["pointer"],
        "raw_operational_identity_before": stat_before,
        "raw_operational_identity_after": stat_after,
        "frozen_raw_identity_source": {
            "path": project_label(frozen["path"]),
            "sha256": frozen["sha256"],
        },
        "scan": scan_public,
    }
    return {
        "build_id": build_id,
        "binding": binding,
        "summary": summary,
        "input_receipt": input_receipt,
        "checks": checks,
        "trading_rows": derived["trading_rows"],
        "rebalance_rows": derived["rebalance_rows"],
        "authenticated": authenticated,
    }


TRADING_FIELDS = (
    "calendar_date",
    "year",
    "month",
    "weekday_iso",
    "is_observed_month_end",
)
REBALANCE_FIELDS = (
    "holding_month",
    "stage",
    "signal_date",
    "entry_date",
    "exit_date",
    "is_embargo",
    "is_partial",
    "full_feature_time_eligible",
    "formal_inference_expected",
)
CHECK_FIELDS = ("check_id", "status", "expected", "actual", "evidence", "message")


def _write_parquet_tables(data_dir: Path, compiled: Mapping[str, Any]) -> None:
    trading_schema = pa.schema(
        [
            ("calendar_date", pa.date32()),
            ("year", pa.int16()),
            ("month", pa.int8()),
            ("weekday_iso", pa.int8()),
            ("is_observed_month_end", pa.bool_()),
        ]
    )
    rebalance_schema = pa.schema(
        [
            ("holding_month", pa.string()),
            ("stage", pa.string()),
            ("signal_date", pa.date32()),
            ("entry_date", pa.date32()),
            ("exit_date", pa.date32()),
            ("is_embargo", pa.bool_()),
            ("is_partial", pa.bool_()),
            ("full_feature_time_eligible", pa.bool_()),
            ("formal_inference_expected", pa.bool_()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(compiled["trading_rows"], schema=trading_schema),
        data_dir / "trading_calendar.parquet",
        compression="zstd",
        version="2.6",
    )
    pq.write_table(
        pa.Table.from_pylist(compiled["rebalance_rows"], schema=rebalance_schema),
        data_dir / "rebalance_calendar.parquet",
        compression="zstd",
        version="2.6",
    )


def _directory_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _publish_staged_directory(staged: Path, final: Path) -> None:
    expected = _directory_hashes(staged)
    if final.exists():
        actual = _directory_hashes(final)
        if actual != expected:
            raise CalendarBuildError(
                f"Existing calendar directory differs from deterministic build: {final}"
            )
        shutil.rmtree(staged)
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staged, final)


def publish_calendar_build(
    compiled: Mapping[str, Any], settings: CalendarSettings | None = None
) -> dict[str, Path]:
    settings = settings or load_settings()
    contract_id = compiled["binding"]["contract_id"]
    build_id = compiled["build_id"]
    interim_final = settings.interim_root / contract_id / "calendar" / build_id
    quality_final = settings.quality_root / contract_id / "calendar" / build_id
    settings.database_root.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f"{build_id}-", dir=settings.database_root)
    )
    interim_stage = staging_parent / "interim"
    quality_stage = staging_parent / "quality"
    interim_stage.mkdir()
    quality_stage.mkdir()
    try:
        _write_csv(interim_stage / "trading_calendar.csv", TRADING_FIELDS, compiled["trading_rows"])
        _write_csv(
            interim_stage / "rebalance_calendar.csv", REBALANCE_FIELDS, compiled["rebalance_rows"]
        )
        _write_parquet_tables(interim_stage, compiled)
        _write_json(interim_stage / "_SUCCESS.json", compiled["summary"])
        _write_csv(quality_stage / "calendar_checks.csv", CHECK_FIELDS, compiled["checks"])
        _write_json(quality_stage / "calendar_summary.json", compiled["summary"])
        _write_json(quality_stage / "input_receipt.json", compiled["input_receipt"])
        _publish_staged_directory(interim_stage, interim_final)
        _publish_staged_directory(quality_stage, quality_final)
    finally:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)

    output_files = [
        *(path for path in sorted(interim_final.iterdir()) if path.is_file()),
        *(path for path in sorted(quality_final.iterdir()) if path.is_file()),
    ]
    outputs = [
        {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in output_files
    ]
    manifest = {
        "schema_version": 1,
        "build_id": build_id,
        "contract_id": contract_id,
        "verification_id": compiled["binding"]["verification_id"],
        "status": compiled["summary"]["status"],
        "binding": compiled["binding"],
        "summary": compiled["summary"],
        "outputs": outputs,
    }
    artifact_final = settings.artifact_root / "calendar_builds" / build_id
    if artifact_final.exists():
        existing = _read_json(artifact_final / "manifest.json")
        if existing != manifest:
            raise CalendarBuildError("Existing calendar artifact manifest differs")
    else:
        (settings.artifact_root / "_staging").mkdir(parents=True, exist_ok=True)
        artifact_stage_parent = Path(
            tempfile.mkdtemp(prefix=f"{build_id}-", dir=settings.artifact_root / "_staging")
        )
        artifact_stage = artifact_stage_parent / build_id
        artifact_stage.mkdir()
        try:
            _write_json(artifact_stage / "manifest.json", manifest)
            with (artifact_stage / "output_hashes.sha256").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                for item in outputs:
                    handle.write(f"{item['sha256']}  {item['path']}\n")
            artifact_final.parent.mkdir(parents=True, exist_ok=True)
            os.rename(artifact_stage, artifact_final)
        finally:
            if artifact_stage_parent.exists():
                shutil.rmtree(artifact_stage_parent)

    pointer = {
        "schema_version": 1,
        "build_id": build_id,
        "contract_id": contract_id,
        "artifact_path": project_label(artifact_final),
        "manifest_sha256": sha256_file(artifact_final / "manifest.json"),
        "status": compiled["summary"]["status"],
    }
    pointer_tmp = settings.artifact_root / f".LATEST_CALENDAR_PRIMARY.{os.getpid()}.tmp"
    _write_json(pointer_tmp, pointer)
    os.replace(pointer_tmp, settings.artifact_root / "LATEST_CALENDAR_PRIMARY.json")

    run_dir = settings.run_root / contract_id / "calendar" / build_id
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / "build_receipt.json"
    if not receipt_path.exists():
        _write_json(
            receipt_path,
            {
                "schema_version": 1,
                "build_id": build_id,
                "contract_id": contract_id,
                "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": compiled["summary"]["status"],
                "manifest_sha256": pointer["manifest_sha256"],
                "raw_columns_read": ["DlyCalDt"],
            },
        )
    return {
        "interim": interim_final,
        "quality": quality_final,
        "artifact": artifact_final,
        "receipt": receipt_path,
    }


def build_calendar(settings: CalendarSettings | None = None) -> tuple[dict[str, Any], dict[str, Path]]:
    settings = settings or load_settings()
    compiled = compile_calendar_build(settings)
    paths = publish_calendar_build(compiled, settings)
    return compiled, paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-config", type=Path, default=DEFAULT_PHASE_CONFIG)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    args = parser.parse_args(argv)
    settings = load_settings(args.phase_config, args.paths_config)
    compiled, paths = build_calendar(settings)
    print(
        json.dumps(
            {
                "build_id": compiled["build_id"],
                "status": compiled["summary"]["status"],
                "observed_trading_dates": compiled["summary"]["observed_trading_dates"],
                "rebalance_intervals": compiled["summary"]["rebalance_intervals"],
                "interim_path": paths["interim"].as_posix(),
                "quality_path": paths["quality"].as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
