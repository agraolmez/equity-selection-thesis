"""Independent verification of a Phase 1 raw-snapshot manifest.

This verifier intentionally uses macOS ``shasum`` for byte identity and the
pandas C CSV engine for record/date checks. It does not reuse PyArrow batches or
the primary scanner's date accumulator.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import portfolio_bot.data.raw_manifest as raw_manifest_module

from portfolio_bot.data.raw_manifest import (
    DEFAULT_PATHS_CONFIG,
    DEFAULT_PHASE_CONFIG,
    PhaseSettings,
    load_settings,
    verify_methodology_freeze,
)


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(payload))
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "metric", "primary_value", "verification_value", "message"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_header_independently(path: Path) -> list[str]:
    return list(
        pd.read_csv(
            path,
            nrows=0,
            engine="c",
            encoding="utf-8-sig",
            keep_default_na=False,
        ).columns
    )


def _shasum(path: Path) -> str:
    completed = subprocess.run(
        ["/usr/bin/shasum", "-a", "256", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.split()[0]


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T](?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):?[0-5]\d)?$"
)
_MDY_DATE = re.compile(r"^\d{2}-\d{2}-\d{4}$")


def _parse_date_counter(values: Counter[str], sentinels: set[str]) -> dict[str, Any]:
    rows = sum(values.values())
    blank = 0
    sentinel = 0
    valid = 0
    invalid = 0
    minimum: dt.datetime | None = None
    maximum: dt.datetime | None = None
    by_year: Counter[int] = Counter()
    by_format: Counter[str] = Counter()
    for raw_value, count in values.items():
        value = str(raw_value).strip()
        if not value:
            blank += count
            continue
        if value in sentinels:
            sentinel += count
            continue
        parsed = None
        selected_format = None
        candidates: list[tuple[str, str, str]] = []
        if _ISO_DATE.fullmatch(value) or _ISO_DATETIME.fullmatch(value):
            candidates.append(("YYYY-MM-DD", value[:10], "%Y-%m-%d"))
        elif _MDY_DATE.fullmatch(value):
            candidates.append(("MM-DD-YYYY", value, "%m-%d-%Y"))
        for label, parse_value, date_format in candidates:
            try:
                parsed = dt.datetime.strptime(parse_value, date_format)
                selected_format = label
                break
            except ValueError:
                pass
        if parsed is None:
            invalid += count
            continue
        valid += count
        by_format[selected_format] += count
        by_year[parsed.year] += count
        minimum = parsed if minimum is None or parsed < minimum else minimum
        maximum = parsed if maximum is None or parsed > maximum else maximum
    return {
        "rows": rows,
        "null_or_blank": blank,
        "open_ended_sentinel": sentinel,
        "valid_date_or_datetime": valid,
        "invalid_nonblank": invalid,
        "category_count_reconciles": blank + sentinel + valid + invalid == rows,
        "minimum": minimum.date().isoformat() if minimum else None,
        "maximum": maximum.date().isoformat() if maximum else None,
        "valid_counts_by_format": {
            name: by_format[name] for name in ("YYYY-MM-DD", "MM-DD-YYYY")
        },
        "counts_by_year": {str(year): by_year[year] for year in sorted(by_year)},
    }


def _pandas_row_and_date_profile(
    path: Path,
    date_columns: tuple[str, ...],
    sentinel_registry: dict[str, list[str]],
    chunksize: int = 2_000_000,
) -> tuple[int, dict[str, dict[str, Any]]]:
    counters = {name: Counter() for name in date_columns}
    rows = 0
    next_report = 10_000_000
    reader = pd.read_csv(
        path,
        usecols=list(date_columns),
        dtype="string",
        chunksize=chunksize,
        engine="c",
        encoding="utf-8-sig",
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="error",
    )
    for chunk in reader:
        rows += len(chunk)
        for name in date_columns:
            counts = chunk[name].astype(str).value_counts(dropna=False, sort=False)
            counters[name].update({str(value): int(count) for value, count in counts.items()})
        if rows >= next_report:
            print(
                json.dumps({"event": "verification_parse_progress", "file": path.name, "rows": rows}),
                flush=True,
            )
            while next_report <= rows:
                next_report += 10_000_000
    return rows, {
        name: _parse_date_counter(counter, set(sentinel_registry.get(name, [])))
        for name, counter in counters.items()
    }


def _difference(
    rows: list[dict[str, Any]],
    filename: str,
    metric: str,
    primary: Any,
    verification: Any,
    message: str,
) -> None:
    if primary != verification:
        rows.append(
            {
                "filename": filename,
                "metric": metric,
                "primary_value": json.dumps(primary, sort_keys=True, ensure_ascii=False),
                "verification_value": json.dumps(verification, sort_keys=True, ensure_ascii=False),
                "message": message,
            }
        )


def resolve_snapshot_directory(settings: PhaseSettings, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    pointer = settings.artifact_root / "LATEST_PRIMARY.json"
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    return (settings.artifact_root / payload["path"]).resolve()


def verify_snapshot(settings: PhaseSettings, snapshot_directory: Path) -> Path:
    started = dt.datetime.now(dt.timezone.utc)
    implementation_path = Path(__file__).resolve()
    verifier_hash = _sha256_file(implementation_path)
    imported_primary_path = Path(raw_manifest_module.__file__).resolve()
    imported_primary_hash = _sha256_file(imported_primary_path)
    manifest_path = snapshot_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = _sha256_file(manifest_path)
    build_id = manifest["phase_01_build_id"]
    raw_snapshot_id = manifest["raw_snapshot_id"]
    verification_identity = {
        "phase_01_build_id": build_id,
        "manifest_sha256": manifest_hash,
        "verifier_implementation_sha256": verifier_hash,
        "imported_primary_implementation_sha256": imported_primary_hash,
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
    }
    verification_id = "verify-" + hashlib.sha256(
        _canonical_json_bytes(verification_identity)
    ).hexdigest()[:16]
    attempt_id = started.strftime("verification-%Y%m%dT%H%M%S%fZ") + f"-p{os.getpid()}"
    attempt_directory = settings.staging_directory / attempt_id
    bundle_directory = attempt_directory / verification_id
    bundle_directory.mkdir(parents=True, exist_ok=False)

    profiles = {item["filename"]: item for item in manifest["files"]}
    contracts = {item.filename: item for item in settings.contracts}
    differences: list[dict[str, Any]] = []
    verified_files: list[dict[str, Any]] = []
    per_file_elapsed: dict[str, float] = {}
    operational_by_file: dict[str, Any] = {}
    shasum_lines: list[str] = []
    sentinel_registry = settings.quality.get("open_ended_date_sentinels", {})
    physical_files_before = sorted(
        path.name for path in settings.raw_root.glob("*.csv") if path.is_file()
    )

    current_methodology = verify_methodology_freeze(settings.methodology_freeze)
    _difference(differences, "snapshot", "primary_status", manifest.get("primary_status"), "primary_passed_pending_independent_verification", "Only a passing complete primary build may be independently verified.")
    _difference(differences, "snapshot", "configuration_hash", manifest.get("configuration_hash"), settings.configuration_hash, "Verifier configuration differs from the primary build.")
    _difference(differences, "snapshot", "methodology_freeze", manifest.get("methodology_freeze"), current_methodology, "Current methodology freeze differs from the one linked by the primary build.")
    _difference(differences, "snapshot", "build_directory", snapshot_directory.name, build_id, "Primary directory name differs from its embedded immutable build ID.")

    for filename in sorted(contracts):
        file_started = time.monotonic()
        print(json.dumps({"event": "verification_file_start", "file": filename}), flush=True)
        contract = contracts[filename]
        primary = profiles.get(filename)
        if primary is None:
            differences.append(
                {
                    "filename": filename,
                    "metric": "manifest_presence",
                    "primary_value": "null",
                    "verification_value": "true",
                    "message": "Registered file is absent from the primary manifest.",
                }
            )
            continue
        path = settings.raw_root / filename
        before = path.stat()
        independent_hash = _shasum(path)
        shasum_lines.append(f"{independent_hash}  {filename}\n")
        header = _read_header_independently(path)
        row_count, date_profiles = _pandas_row_and_date_profile(
            path, contract.date_columns, sentinel_registry
        )
        after = path.stat()
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        operational_by_file[filename] = {
            "path": str(path.resolve()),
            "stat_before": {
                "device": before.st_dev,
                "inode": before.st_ino,
                "size_bytes": before.st_size,
                "mtime_ns": before.st_mtime_ns,
            },
            "stat_after": {
                "device": after.st_dev,
                "inode": after.st_ino,
                "size_bytes": after.st_size,
                "mtime_ns": after.st_mtime_ns,
            },
        }

        _difference(differences, filename, "sha256", primary["byte_identity"]["sha256"], independent_hash, "macOS shasum differs from the primary Python hash.")
        _difference(differences, filename, "size_bytes", primary["byte_identity"]["size_bytes"], before.st_size, "Filesystem size differs from the primary manifest.")
        _difference(differences, filename, "ordered_header", primary["csv"]["ordered_header"], header, "Independent pandas C-engine CSV header differs.")
        _difference(differences, filename, "parsed_data_rows", primary["semantic_profile"]["parsed_data_rows"], row_count, "Pandas C-engine row count differs from PyArrow.")
        _difference(differences, filename, "stable_during_verification", True, stable, "Raw file changed during the independent pass.")

        primary_dates = {
            item["column"]: {key: value for key, value in item.items() if key != "column"}
            for item in primary["semantic_profile"]["date_coverage"]
        }
        for name in contract.date_columns:
            _difference(
                differences,
                filename,
                f"date_profile:{name}",
                primary_dates.get(name),
                date_profiles.get(name),
                "Independent full-column date profile differs from the primary profile.",
            )

        verified_files.append(
            {
                "filename": filename,
                "sha256": independent_hash,
                "size_bytes": before.st_size,
                "column_count": len(header),
                "parsed_data_rows": row_count,
                "date_coverage": date_profiles,
                "stable_during_verification": stable,
                "verification_status": "passed"
                if not any(item["filename"] == filename for item in differences)
                else "failed",
            }
        )
        per_file_elapsed[filename] = round(time.monotonic() - file_started, 3)
        print(
            json.dumps(
                {
                    "event": "verification_file_complete",
                    "file": filename,
                    "rows": row_count,
                    "status": verified_files[-1]["verification_status"],
                }
            ),
            flush=True,
        )

    expected_files = sorted(contracts)
    physical_files_after = sorted(
        path.name for path in settings.raw_root.glob("*.csv") if path.is_file()
    )
    _difference(differences, "snapshot", "physical_file_registry", expected_files, physical_files_after, "Physical CSV registry differs from the frozen contract.")
    _difference(differences, "snapshot", "physical_registry_stable_during_verification", physical_files_before, physical_files_after, "No raw CSV may be added, removed, or renamed during independent verification.")
    _difference(differences, "snapshot", "manifest_file_registry", expected_files, sorted(profiles), "Primary manifest registry differs from the frozen contract.")
    verified_by_name = {item["filename"]: item for item in verified_files}
    for filename, operational in operational_by_file.items():
        expected_stat = operational["stat_after"]
        try:
            stat = (settings.raw_root / filename).stat()
            final_stat = {
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            stable_through_run = all(
                expected_stat.get(key) == final_stat.get(key)
                for key in ("device", "inode", "size_bytes", "mtime_ns")
            )
        except OSError as exc:
            final_stat = {"error": f"{type(exc).__name__}: {exc}"}
            stable_through_run = False
        operational["end_of_verification_run_stat"] = final_stat
        operational["stable_through_entire_verification_run"] = stable_through_run
        _difference(
            differences,
            filename,
            "stable_through_entire_verification_run",
            True,
            stable_through_run,
            "The file fingerprint at verification completion must match the file independently parsed and hashed.",
        )
        if filename in verified_by_name:
            verified_by_name[filename]["stable_through_entire_verification_run"] = stable_through_run
            if not stable_through_run:
                verified_by_name[filename]["verification_status"] = "failed"
    verifier_hash_after = _sha256_file(implementation_path)
    imported_primary_hash_after = _sha256_file(imported_primary_path)
    manifest_hash_after = _sha256_file(manifest_path)
    settings_after = load_settings(
        settings.phase_config_path, settings.paths_config_path
    )
    methodology_after = verify_methodology_freeze(settings.methodology_freeze)
    _difference(differences, "snapshot", "verifier_stable_during_run", verifier_hash, verifier_hash_after, "Verifier implementation changed during the independent pass.")
    _difference(differences, "snapshot", "imported_primary_implementation_stable_during_run", imported_primary_hash, imported_primary_hash_after, "Imported settings/freeze implementation changed during the independent pass.")
    _difference(differences, "snapshot", "primary_manifest_stable_during_verification", manifest_hash, manifest_hash_after, "The immutable primary manifest changed during independent verification.")
    _difference(differences, "snapshot", "configuration_stable_during_run", settings.configuration_hash, settings_after.configuration_hash, "Configuration changed during the independent pass.")
    _difference(differences, "snapshot", "path_configuration_stable_during_run", settings.path_configuration_hash, settings_after.path_configuration_hash, "Local database pointer configuration changed during the independent pass.")
    _difference(differences, "snapshot", "methodology_freeze_stable_during_run", current_methodology, methodology_after, "Methodology freeze changed during the independent pass.")

    verification = {
        "schema_version": 2,
        "raw_snapshot_id": raw_snapshot_id,
        "phase_01_build_id": build_id,
        "verification_id": verification_id,
        "manifest_sha256": manifest_hash,
        "verifier_implementation_sha256": verifier_hash,
        "imported_primary_implementation_sha256": imported_primary_hash,
        "configuration_hash": settings.configuration_hash,
        "verification_status": "passed" if not differences else "failed",
        "difference_count": len(differences),
        "verification_engines": {
            "byte_hash": "/usr/bin/shasum -a 256",
            "csv_rows_and_dates": f"pandas {pd.__version__} C engine",
            "header": f"pandas {pd.__version__} C engine",
            "python": platform.python_version(),
        },
        "files": verified_files,
    }
    _write_json(bundle_directory / "independent_check.json", verification)
    _write_csv(bundle_directory / "differences.csv", differences)
    (bundle_directory / "shasum.sha256").write_text("".join(shasum_lines), encoding="utf-8")

    completed = dt.datetime.now(dt.timezone.utc)
    metadata = {
        "raw_snapshot_id": raw_snapshot_id,
        "phase_01_build_id": build_id,
        "verification_id": verification_id,
        "attempt_id": attempt_id,
        "run_started_utc": started.isoformat(),
        "run_completed_utc": completed.isoformat(),
        "elapsed_seconds": round((completed - started).total_seconds(), 3),
        "per_file_elapsed_seconds": per_file_elapsed,
        "per_file_operational_metadata": operational_by_file,
        "python": sys.version,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "implementation_sha256": verifier_hash,
        "imported_primary_implementation_sha256": imported_primary_hash,
        "raw_root": str(settings.raw_root),
    }
    _write_json(bundle_directory / "verification_run_metadata.json", metadata)

    passed = verification["verification_status"] == "passed"
    if passed:
        final_parent = settings.verification_directory / build_id
        final_parent.mkdir(parents=True, exist_ok=True)
        final_directory = final_parent / verification_id
        if final_directory.exists():
            raise FileExistsError(
                f"Immutable independent verification already exists: {final_directory}"
            )
        os.replace(bundle_directory, final_directory)
        output_directory = final_directory
        _write_json(
            settings.artifact_root / "LATEST_VERIFIED.json",
            {
                "raw_snapshot_id": raw_snapshot_id,
                "phase_01_build_id": build_id,
                "verification_id": verification_id,
                "primary_path": str(snapshot_directory.relative_to(settings.artifact_root)),
                "verification_path": str(output_directory.relative_to(settings.artifact_root)),
                "verification_status": "passed",
            },
        )
    else:
        output_directory = bundle_directory
    _write_json(
        settings.artifact_root / "LATEST_VERIFICATION_ATTEMPT.json",
        {
            "raw_snapshot_id": raw_snapshot_id,
            "phase_01_build_id": build_id,
            "verification_id": verification_id,
            "path": str(output_directory.relative_to(settings.artifact_root)),
            "verification_status": verification["verification_status"],
            "published_as_verified": passed,
        },
    )
    print(
        json.dumps(
            {
                "event": "independent_verification_complete",
                "raw_snapshot_id": raw_snapshot_id,
                "phase_01_build_id": build_id,
                "verification_id": verification_id,
                "status": verification["verification_status"],
                "differences": len(differences),
            }
        ),
        flush=True,
    )
    return output_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-config", type=Path, default=DEFAULT_PHASE_CONFIG)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--snapshot-dir", type=Path)
    args = parser.parse_args(argv)
    settings = load_settings(args.phase_config, args.paths_config)
    snapshot_directory = resolve_snapshot_directory(settings, args.snapshot_dir)
    output_directory = verify_snapshot(settings, snapshot_directory)
    result = json.loads(
        (output_directory / "independent_check.json").read_text(encoding="utf-8")
    )
    published = (
        output_directory.parent.parent.resolve()
        == settings.verification_directory.resolve()
        and result.get("verification_status") == "passed"
    )
    return 0 if published else 3


if __name__ == "__main__":
    raise SystemExit(main())
