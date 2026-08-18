"""Build reviewed Phase 1 summaries from immutable primary and verification bundles."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = Path("artifacts/phase_01_raw_manifest")
REPORT_ROOT = Path("reports/phase_01_raw_snapshot")
HEADLINE_SQL = """SELECT file_count, parsed_rows, size_gb, primary_pass_count,
       independent_difference_count
FROM phase_01_headline;"""
FAMILY_SQL = """SELECT family, file_count, parsed_rows, size_bytes, size_gb,
       coverage_start, coverage_end
FROM phase_01_family_summary
ORDER BY parsed_rows DESC, family ASC;"""
FILE_SQL = """SELECT filename, family, parsed_rows, size_bytes, size_gb,
       coverage_axis, coverage_start, coverage_end, primary_status,
       independent_status, stable_through_verification
FROM phase_01_file_summary
ORDER BY filename ASC;"""
SCOPE_SQL = """SELECT claim, status, evidence, next_phase
FROM phase_01_scope
ORDER BY claim ASC;"""
PROVENANCE_SQL = """SELECT raw_snapshot_id, phase_01_build_id, verification_id,
       primary_manifest_sha256, independent_check_sha256,
       primary_implementation_sha256, verifier_implementation_sha256
FROM phase_01_provenance;"""

EXPECTED_PHASE_01_FILE_COUNT = 25
EXPECTED_PHASE_01_DATE_PROFILE_COUNT = 83
EXPECTED_PHASE_01_QUALITY_CHECK_COUNT = 585
EXPECTED_SNAPSHOT_CHECK_IDS = {
    "CONFIGURATION_STABLE_DURING_RUN",
    "EXPECTED_CONTRACT_COUNT",
    "IMPLEMENTATION_STABLE_DURING_RUN",
    "METHODOLOGY_FREEZE_STABLE_DURING_RUN",
    "METHODOLOGY_FREEZE_VERIFIED",
    "MISSING_EXPECTED_FILES",
    "PATH_CONFIGURATION_STABLE_DURING_RUN",
    "PHYSICAL_CSV_COUNT",
    "PHYSICAL_CSV_REGISTRY_STABLE_DURING_RUN",
    "UNEXPECTED_CSV_FILES",
}
EXPECTED_FILE_CHECK_IDS = {
    "BYTE_SCAN_COMPLETE",
    "COVERAGE_DATE_COMPLETE",
    "COVERAGE_DATE_NO_SENTINEL",
    "COVERAGE_DATE_PARSE",
    "COVERAGE_DATE_PRESENT",
    "CSV_PARSE",
    "DATE_CATEGORY_ACCOUNTING",
    "FILE_NONEMPTY",
    "FILE_STABLE_BYTE_SCAN",
    "FILE_STABLE_HEADER_SAMPLE_SCAN",
    "FILE_STABLE_SEMANTIC_SCAN",
    "FILE_STABLE_THROUGH_ENTIRE_PRIMARY_RUN",
    "HEADER_NAMES_NONBLANK",
    "HEADER_NAMES_UNIQUE",
    "HEADER_NORMALIZATION_UNIQUE",
    "HEADER_PRESENT",
    "MALFORMED_ROWS",
    "NO_NUL_BYTES",
    "PHYSICAL_LOGICAL_RECORD_RELATION",
    "REQUIRED_COLUMNS_PRESENT",
    "SAME_FILE_ACROSS_PRIMARY_PASSES",
    "SECONDARY_DATE_PARSE",
    "UTF8_VALID",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_hash_registry(path: Path) -> dict[str, str]:
    registry: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, filename = line.strip().split("  ", 1)
            if filename in registry:
                raise ValueError(f"Duplicate filename in hash registry {path}: {filename}")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"Invalid SHA-256 digest in hash registry {path}")
            registry[filename] = digest
    return registry


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return "" if value is None else str(value)


def _csv_row(row: dict[str, Any]) -> dict[str, str]:
    return {key: _csv_value(value) for key, value in row.items()}


def _manifest_file_rows(manifest: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for profile in manifest.get("files", []):
        byte_identity = profile["byte_identity"]
        semantic_profile = profile["semantic_profile"]
        rows.append(
            _csv_row(
                {
                    "filename": profile["filename"],
                    "family": profile["family"],
                    "role": profile["role"],
                    "grain": profile["grain"],
                    "size_bytes": byte_identity["size_bytes"],
                    "sha256": byte_identity["sha256"],
                    "column_count": profile["csv"]["column_count"],
                    "parsed_data_rows": semantic_profile["parsed_data_rows"],
                    "malformed_rows": semantic_profile["malformed_rows"],
                    "coverage_date": profile["coverage_date"],
                    "coverage_date_semantics": profile["coverage_date_semantics"],
                    "newline_style": byte_identity["newline_style"],
                    "utf8_valid": byte_identity["utf8_valid"],
                    "source_as_of": profile["source_creation_or_as_of"],
                    "primary_build_status": profile["primary_build_status"],
                }
            )
        )
    return rows


def _manifest_date_rows(manifest: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for profile in manifest.get("files", []):
        for date_profile in profile["semantic_profile"]["date_coverage"]:
            rows.append(
                _csv_row(
                    {
                        "filename": profile["filename"],
                        "is_coverage_axis": date_profile["column"]
                        == profile["coverage_date"],
                        **date_profile,
                    }
                )
            )
    return rows


def _manifest_hash_registry(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        profile["filename"]: profile["byte_identity"]["sha256"]
        for profile in manifest.get("files", [])
    }


def _expected_independent_files(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for profile in manifest.get("files", []):
        expected.append(
            {
                "column_count": profile["csv"]["column_count"],
                "date_coverage": {
                    item["column"]: {
                        key: value for key, value in item.items() if key != "column"
                    }
                    for item in profile["semantic_profile"]["date_coverage"]
                },
                "filename": profile["filename"],
                "parsed_data_rows": profile["semantic_profile"]["parsed_data_rows"],
                "sha256": profile["byte_identity"]["sha256"],
                "size_bytes": profile["byte_identity"]["size_bytes"],
                "stable_during_verification": True,
                "stable_through_entire_verification_run": True,
                "verification_status": "passed",
            }
        )
    return expected


def _inside(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Evidence path escapes its artifact root: {candidate}")
    return resolved


def _as_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _minimum(values: list[str]) -> str | None:
    observed = [value for value in values if value]
    return min(observed) if observed else None


def _maximum(values: list[str]) -> str | None:
    observed = [value for value in values if value]
    return max(observed) if observed else None


def load_review_context(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load, reconcile, and summarize the latest independently verified Phase 1 build."""
    project_root = project_root.resolve()
    artifact_root = project_root / ARTIFACT_ROOT
    verified_pointer = _read_json(artifact_root / "LATEST_VERIFIED.json")
    primary_directory = _inside(
        artifact_root, artifact_root / verified_pointer["primary_path"]
    )
    verification_directory = _inside(
        artifact_root, artifact_root / verified_pointer["verification_path"]
    )

    manifest_path = primary_directory / "manifest.json"
    independent_path = verification_directory / "independent_check.json"
    manifest = _read_json(manifest_path)
    independent = _read_json(independent_path)
    primary_metadata = _read_json(primary_directory / "run_metadata.json")
    verification_metadata = _read_json(
        verification_directory / "verification_run_metadata.json"
    )
    files = _read_csv(primary_directory / "files.csv")
    dates = _read_csv(primary_directory / "date_coverage.csv")
    checks = _read_csv(primary_directory / "quality_checks.csv")
    differences = _read_csv(verification_directory / "differences.csv")
    primary_hashes = _read_hash_registry(primary_directory / "hashes.sha256")
    verified_hashes = _read_hash_registry(verification_directory / "shasum.sha256")

    primary_manifest_sha256 = _sha256_file(manifest_path)
    independent_check_sha256 = _sha256_file(independent_path)
    primary_source_sha256 = _sha256_file(
        project_root / "src" / "portfolio_bot" / "data" / "raw_manifest.py"
    )
    verifier_source_sha256 = _sha256_file(
        project_root / "src" / "portfolio_bot" / "data" / "verify_raw_manifest.py"
    )
    manifest_files = manifest.get("files", [])
    manifest_filenames = [item["filename"] for item in manifest_files]
    manifest_filename_set = set(manifest_filenames)
    expected_file_rows = _manifest_file_rows(manifest)
    expected_date_rows = _manifest_date_rows(manifest)
    expected_hashes = _manifest_hash_registry(manifest)
    expected_independent_files = _expected_independent_files(manifest)
    quality_check_fields = {
        "check_id",
        "scope",
        "status",
        "severity",
        "observed",
        "expected",
        "message",
    }
    check_keys = [(row.get("scope"), row.get("check_id")) for row in checks]
    snapshot_checks = {
        row.get("check_id") for row in checks if row.get("scope") == "snapshot"
    }
    expected_check_keys = {
        (filename, check_id)
        for filename in manifest_filename_set
        for check_id in EXPECTED_FILE_CHECK_IDS
    } | {("snapshot", check_id) for check_id in EXPECTED_SNAPSHOT_CHECK_IDS}
    file_stability_checks = [
        row
        for row in checks
        if row.get("check_id") == "FILE_STABLE_THROUGH_ENTIRE_PRIMARY_RUN"
    ]
    reconciliation = {
        "manifest_file_count_exact": len(manifest_files)
        == EXPECTED_PHASE_01_FILE_COUNT,
        "manifest_filenames_unique": len(manifest_filenames)
        == len(manifest_filename_set),
        "primary_files_match_manifest": files == expected_file_rows,
        "primary_dates_match_manifest": dates == expected_date_rows,
        "manifest_date_profile_count_exact": len(expected_date_rows)
        == EXPECTED_PHASE_01_DATE_PROFILE_COUNT,
        "primary_hashes_match_manifest": primary_hashes == expected_hashes,
        "verified_hashes_match_manifest": verified_hashes == expected_hashes,
        "primary_manifest_hash_matches_independent": independent.get(
            "manifest_sha256"
        )
        == primary_manifest_sha256,
        "independent_files_match_manifest": independent.get("files", [])
        == expected_independent_files,
        "independent_difference_count_matches": independent.get("difference_count")
        == len(differences)
        == 0,
        "configuration_hashes_match": independent.get("configuration_hash")
        == manifest.get("configuration_hash"),
        "primary_implementation_hashes_match": independent.get(
            "imported_primary_implementation_sha256"
        )
        == manifest.get("implementation_sha256")
        == primary_source_sha256,
        "verifier_implementation_hash_matches": independent.get(
            "verifier_implementation_sha256"
        )
        == verifier_source_sha256,
        "quality_check_count_exact": len(checks)
        == EXPECTED_PHASE_01_QUALITY_CHECK_COUNT,
        "quality_check_fields_exact": all(
            set(row) == quality_check_fields for row in checks
        ),
        "quality_check_keys_unique": len(check_keys) == len(set(check_keys)),
        "quality_check_keys_exact": set(check_keys) == expected_check_keys,
        "quality_check_scopes_exact": {row.get("scope") for row in checks}
        == manifest_filename_set | {"snapshot"},
        "quality_check_statuses_exact": Counter(
            row.get("status") for row in checks
        )
        == Counter({"PASS": EXPECTED_PHASE_01_QUALITY_CHECK_COUNT}),
        "snapshot_check_ids_exact": snapshot_checks
        == EXPECTED_SNAPSHOT_CHECK_IDS,
        "file_stability_checks_exact": len(file_stability_checks)
        == EXPECTED_PHASE_01_FILE_COUNT
        and {row.get("scope") for row in file_stability_checks}
        == manifest_filename_set
        and all(
            row.get("status") == "PASS"
            and row.get("observed") == "True"
            and row.get("expected") == "True"
            for row in file_stability_checks
        ),
    }

    independent_by_file = {
        item["filename"]: item for item in independent.get("files", [])
    }
    coverage_by_file = {
        item["filename"]: item
        for item in dates
        if _as_bool(item.get("is_coverage_axis"))
    }

    file_summary: list[dict[str, Any]] = []
    for row in files:
        filename = row["filename"]
        coverage = coverage_by_file.get(filename, {})
        verified_file = independent_by_file.get(filename, {})
        file_summary.append(
            {
                "filename": filename,
                "family": row["family"],
                "parsed_rows": int(row["parsed_data_rows"]),
                "size_bytes": int(row["size_bytes"]),
                "size_gb": round(int(row["size_bytes"]) / 1_000_000_000, 3),
                "coverage_axis": row["coverage_date"],
                "coverage_start": coverage.get("minimum") or None,
                "coverage_end": coverage.get("maximum") or None,
                "primary_status": row["primary_build_status"],
                "independent_status": verified_file.get("verification_status", "missing"),
                "stable_through_verification": bool(
                    verified_file.get("stable_through_entire_verification_run", False)
                ),
            }
        )
    file_summary.sort(key=lambda item: item["filename"])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in file_summary:
        grouped[item["family"]].append(item)
    family_summary: list[dict[str, Any]] = []
    for family, rows in grouped.items():
        family_summary.append(
            {
                "family": family,
                "file_count": len(rows),
                "parsed_rows": sum(item["parsed_rows"] for item in rows),
                "size_bytes": sum(item["size_bytes"] for item in rows),
                "size_gb": round(
                    sum(item["size_bytes"] for item in rows) / 1_000_000_000, 3
                ),
                "coverage_start": _minimum(
                    [str(item["coverage_start"] or "") for item in rows]
                ),
                "coverage_end": _maximum(
                    [str(item["coverage_end"] or "") for item in rows]
                ),
            }
        )
    family_summary.sort(key=lambda item: (-item["parsed_rows"], item["family"]))

    check_counts = Counter(row["status"] for row in checks)
    summary = {
        "phase": "01_raw_snapshot",
        "phase_status": "verified_pending_explicit_phase_freeze",
        "raw_snapshot_id": manifest["raw_snapshot_id"],
        "phase_01_build_id": manifest["phase_01_build_id"],
        "verification_id": independent["verification_id"],
        "file_count": len(files),
        "parsed_rows": sum(int(row["parsed_data_rows"]) for row in files),
        "size_bytes": sum(int(row["size_bytes"]) for row in files),
        "size_gb": round(sum(int(row["size_bytes"]) for row in files) / 1_000_000_000, 3),
        "primary_pass_count": check_counts.get("PASS", 0),
        "primary_warning_count": check_counts.get("WARN", 0),
        "primary_failure_count": check_counts.get("FAIL", 0),
        "independent_difference_count": len(differences),
        "verified_file_count": len(independent.get("files", [])),
        "date_profile_count": len(dates),
        "date_accounting_all_reconciles": all(
            _as_bool(row.get("category_count_reconciles")) for row in dates
        ),
        "hash_registries_equal": primary_hashes == verified_hashes,
        "all_primary_files_passed": all(
            row["primary_build_status"] == "passed" for row in files
        ),
        "all_independent_files_passed": all(
            item.get("verification_status") == "passed"
            for item in independent.get("files", [])
        ),
        "all_files_stable_through_verification": all(
            item.get("stable_through_entire_verification_run") is True
            for item in independent.get("files", [])
        ),
        "primary_completed_utc": primary_metadata["run_completed_utc"],
        "verification_completed_utc": verification_metadata["run_completed_utc"],
        "primary_elapsed_seconds": primary_metadata["elapsed_seconds"],
        "verification_elapsed_seconds": verification_metadata["elapsed_seconds"],
    }

    scope_rows = [
        {
            "claim": "Raw byte identity",
            "status": "Verified; ready to freeze",
            "evidence": "Python SHA-256 and macOS shasum agree for all 25 files",
            "next_phase": "Explicit Phase 1 freeze",
        },
        {
            "claim": "CSV structure and parseability",
            "status": "Verified; ready to freeze",
            "evidence": "Primary Python/PyArrow implementation and pandas verifier agree on headers and 683,495,563 rows",
            "next_phase": "Explicit Phase 1 freeze",
        },
        {
            "claim": "Configured date coverage",
            "status": "Verified; ready to freeze",
            "evidence": "Every date profile reconciles; independent profiles have zero differences",
            "next_phase": "Explicit Phase 1 freeze",
        },
        {
            "claim": "Public availability and point-in-time joins",
            "status": "Deferred by design",
            "evidence": "Coverage axes are explicitly not treated as availability dates",
            "next_phase": "Phase 2 after explicit Phase 1 freeze",
        },
        {
            "claim": "Canonical panel uniqueness and link resolution",
            "status": "Deferred by design",
            "evidence": "Phase 1 key fields are diagnostics; global uniqueness is not asserted",
            "next_phase": "Phase 2 after explicit Phase 1 freeze",
        },
        {
            "claim": "Feature leakage, labels, models, and backtests",
            "status": "Not started",
            "evidence": "No empirical portfolio result is produced by this phase",
            "next_phase": "Phases 3–7",
        },
    ]

    return {
        "project_root": project_root,
        "artifact_root": artifact_root,
        "primary_directory": primary_directory,
        "verification_directory": verification_directory,
        "verified_pointer": verified_pointer,
        "manifest": manifest,
        "independent": independent,
        "primary_metadata": primary_metadata,
        "verification_metadata": verification_metadata,
        "files": files,
        "dates": dates,
        "checks": checks,
        "differences": differences,
        "primary_hashes": primary_hashes,
        "verified_hashes": verified_hashes,
        "primary_manifest_sha256": primary_manifest_sha256,
        "independent_check_sha256": independent_check_sha256,
        "primary_source_sha256": primary_source_sha256,
        "verifier_source_sha256": verifier_source_sha256,
        "reconciliation": reconciliation,
        "summary": summary,
        "file_summary": file_summary,
        "family_summary": family_summary,
        "scope_rows": scope_rows,
    }


def validation_failures(context: dict[str, Any]) -> list[str]:
    """Return evidence failures that prohibit Phase 1 sign-off."""
    summary = context["summary"]
    pointer = context["verified_pointer"]
    manifest = context["manifest"]
    independent = context["independent"]
    failures: list[str] = []
    if manifest.get("schema_version") != 2:
        failures.append("primary manifest schema is not 2")
    if manifest.get("primary_status") != "primary_passed_pending_independent_verification":
        failures.append("primary manifest does not have a passing status")
    if independent.get("verification_status") != "passed":
        failures.append("independent verification did not pass")
    if independent.get("schema_version") != 2:
        failures.append("independent verification schema is not 2")
    if summary["file_count"] != 25 or summary["verified_file_count"] != 25:
        failures.append("the 25-file registry is incomplete")
    if summary["primary_failure_count"] != 0:
        failures.append("primary quality checks contain failures")
    if summary["primary_warning_count"] != 0:
        failures.append("primary quality checks contain unresolved warnings")
    if summary["independent_difference_count"] != 0:
        failures.append("independent verification contains differences")
    for field in (
        "date_accounting_all_reconciles",
        "hash_registries_equal",
        "all_primary_files_passed",
        "all_independent_files_passed",
        "all_files_stable_through_verification",
    ):
        if summary[field] is not True:
            failures.append(f"required invariant failed: {field}")
    if not (
        manifest["raw_snapshot_id"]
        == independent["raw_snapshot_id"]
        == pointer["raw_snapshot_id"]
    ):
        failures.append("raw snapshot identifiers do not agree")
    if not (
        manifest["phase_01_build_id"]
        == independent["phase_01_build_id"]
        == pointer["phase_01_build_id"]
    ):
        failures.append("Phase 1 build identifiers do not agree")
    if independent.get("verification_id") != pointer.get("verification_id"):
        failures.append("verification identifiers do not agree")
    for invariant, passed in context["reconciliation"].items():
        if passed is not True:
            failures.append(f"authenticated evidence reconciliation failed: {invariant}")
    return failures


def provenance_record(context: dict[str, Any]) -> dict[str, str]:
    """Return the flat identity shared by SQLite and standalone CSV exports."""
    summary = context["summary"]
    return {
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "verification_id": summary["verification_id"],
        "primary_manifest_sha256": context["primary_manifest_sha256"],
        "independent_check_sha256": context["independent_check_sha256"],
        "primary_implementation_sha256": context["manifest"][
            "implementation_sha256"
        ],
        "verifier_implementation_sha256": context["independent"][
            "verifier_implementation_sha256"
        ],
    }


def report_provenance(context: dict[str, Any]) -> dict[str, Any]:
    """Return self-identifying provenance embedded directly in artifact.json."""
    project_root = context["project_root"]
    return {
        "schema_version": 1,
        **provenance_record(context),
        "primary_manifest_path": (
            context["primary_directory"] / "manifest.json"
        ).relative_to(project_root).as_posix(),
        "independent_check_path": (
            context["verification_directory"] / "independent_check.json"
        ).relative_to(project_root).as_posix(),
    }


def with_provenance_columns(
    context: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach the verified evidence identity to every standalone CSV row."""
    identity = provenance_record(context)
    return [{**identity, **row} for row in rows]


def _source_specs(context: dict[str, Any]) -> list[dict[str, Any]]:
    summary = context["summary"]
    build_id = summary["phase_01_build_id"]
    verification_id = summary["verification_id"]
    return [
        {
            "id": "phase_01_summary",
            "label": "Reconciled Phase 1 summary and scope boundary",
            "path": "reports/phase_01_raw_snapshot/phase_01_summary.json",
        },
        {
            "id": "headline_query",
            "label": "Verified Phase 1 headline metrics",
            "path": "reports/phase_01_raw_snapshot/phase_01_review.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": HEADLINE_SQL,
                "description": "Reads the one-row gate summary built after reconciling the immutable primary and independent bundles.",
                "tables_used": ["phase_01_headline"],
                "filters": ["complete 25-file primary build", "zero-difference independent verification"],
                "metric_definitions": [
                    "Parsed rows are logical CSV data rows excluding headers.",
                    "Raw size in GB is registered bytes divided by 1,000,000,000.",
                    "Independent differences are rows in the independently produced differences table.",
                ],
            },
        },
        {
            "id": "family_query",
            "label": "Verified source-family summary",
            "path": "reports/phase_01_raw_snapshot/phase_01_review.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": FAMILY_SQL,
                "description": "Aggregates registered files, rows, bytes, and coverage axes by source family.",
                "tables_used": ["phase_01_family_summary"],
                "filters": ["all 25 registered files"],
                "metric_definitions": [
                    "Parsed rows are summed within source family.",
                    "Coverage start and end are extrema of each file's configured coverage axis, not public-availability dates.",
                ],
            },
        },
        {
            "id": "file_query",
            "label": "Verified 25-file registry",
            "path": "reports/phase_01_raw_snapshot/phase_01_review.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": FILE_SQL,
                "description": "Reads the exact file-level scale, coverage, and verification status used by the report.",
                "tables_used": ["phase_01_file_summary"],
                "filters": ["all 25 registered files"],
            },
        },
        {
            "id": "scope_query",
            "label": "Phase 1 scope boundary",
            "path": "reports/phase_01_raw_snapshot/phase_01_review.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": SCOPE_SQL,
                "description": "Separates verified Phase 1 claims from the gates deliberately deferred to later phases.",
                "tables_used": ["phase_01_scope"],
            },
        },
        {
            "id": "primary_manifest",
            "label": "Immutable Phase 1 primary manifest",
            "path": f"artifacts/phase_01_raw_manifest/builds/{build_id}/manifest.json",
        },
        {
            "id": "primary_quality_checks",
            "label": "Phase 1 primary quality-check ledger",
            "path": f"artifacts/phase_01_raw_manifest/builds/{build_id}/quality_checks.csv",
            "query": {
                "engine": "evidence ledger",
                "language": "csv",
                "description": "The exact 585-row primary ledger records every check identifier, scope, outcome, observed value, expected value, and gate message.",
            },
        },
        {
            "id": "independent_verification",
            "label": "Independent Phase 1 verification",
            "path": f"artifacts/phase_01_raw_manifest/verifications/{build_id}/{verification_id}/independent_check.json",
            "related_paths": [
                f"artifacts/phase_01_raw_manifest/builds/{build_id}/manifest.json",
                "src/portfolio_bot/data/raw_manifest.py",
            ],
            "query": {
                "engine": "authenticated evidence bundle",
                "language": "json and python",
                "description": "The independent record binds the exact primary-manifest and primary-implementation hashes, records separate verification engines, and reports every file-level hash, header, row-count, date-profile, and stability comparison.",
            },
        },
        {
            "id": "methodology_freeze",
            "label": "Frozen pre-implementation Methodology v1",
            "path": "outputs/methodology_v1_freeze.json",
        },
    ]


def build_report_artifact(
    context: dict[str, Any], generated_at: str | None = None
) -> dict[str, Any]:
    failures = validation_failures(context)
    if failures:
        raise RuntimeError("Phase 1 review failed: " + "; ".join(failures))
    generated_at = generated_at or dt.datetime.now(dt.timezone.utc).isoformat()
    summary = context["summary"]
    headline = {
        "file_count": summary["file_count"],
        "parsed_rows": summary["parsed_rows"],
        "size_gb": summary["size_gb"],
        "primary_pass_count": summary["primary_pass_count"],
        "independent_difference_count": summary["independent_difference_count"],
    }
    sources = _source_specs(context)
    cards = [
        {
            "id": "registered_files",
            "description": "Every contracted raw CSV is present and independently verified.",
            "dataset": "headline",
            "sourceId": "headline_query",
            "metrics": [{"label": "Registered files", "field": "file_count", "format": "number"}],
        },
        {
            "id": "parsed_rows",
            "description": "Logical CSV data rows; header rows are excluded.",
            "dataset": "headline",
            "sourceId": "headline_query",
            "metrics": [{"label": "Parsed rows", "field": "parsed_rows", "format": "compact"}],
        },
        {
            "id": "raw_size",
            "description": "Decimal gigabytes across the registered raw snapshot.",
            "dataset": "headline",
            "sourceId": "headline_query",
            "metrics": [{"label": "Raw size (GB)", "field": "size_gb", "format": "number"}],
        },
        {
            "id": "primary_checks",
            "description": "All mandatory and diagnostic primary checks completed without warning or failure.",
            "dataset": "headline",
            "sourceId": "headline_query",
            "metrics": [{"label": "Primary checks passed", "field": "primary_pass_count", "format": "number"}],
        },
        {
            "id": "independent_differences",
            "description": "Differences between the primary and independent implementations.",
            "dataset": "headline",
            "sourceId": "headline_query",
            "metrics": [{"label": "Independent differences", "field": "independent_difference_count", "format": "number"}],
        },
    ]
    tables = [
        {
            "id": "family_summary_table",
            "title": "Registered data by source family",
            "subtitle": "All 25 source files; size uses decimal GB and dates are coverage axes, not public-availability dates",
            "showDescription": True,
            "dataset": "family_summary",
            "defaultSort": {"field": "parsed_rows", "direction": "desc"},
            "density": "spacious",
            "sourceId": "family_query",
            "layout": "full",
            "columns": [
                {"field": "family", "label": "Source family", "type": "text"},
                {"field": "file_count", "label": "Files", "format": "number"},
                {"field": "parsed_rows", "label": "Parsed rows", "format": "number"},
                {"field": "size_gb", "label": "Size (GB)", "format": "number"},
                {"field": "coverage_start", "label": "Coverage start", "type": "date"},
                {"field": "coverage_end", "label": "Coverage end", "type": "date"},
            ],
        },
        {
            "id": "file_registry_table",
            "title": "File-level identity and coverage review",
            "subtitle": "Exact registry used by the immutable build and zero-difference verifier",
            "showDescription": True,
            "dataset": "file_summary",
            "defaultSort": {"field": "filename", "direction": "asc"},
            "density": "dense",
            "sourceId": "file_query",
            "layout": "full",
            "columns": [
                {"field": "filename", "label": "File", "type": "text"},
                {"field": "family", "label": "Family", "type": "text"},
                {"field": "parsed_rows", "label": "Rows", "format": "number"},
                {"field": "size_gb", "label": "GB", "format": "number"},
                {"field": "coverage_axis", "label": "Coverage axis", "type": "text"},
                {"field": "coverage_start", "label": "Start", "type": "date"},
                {"field": "coverage_end", "label": "End", "type": "date"},
                {"field": "independent_status", "label": "Independent check", "type": "text"},
            ],
        },
        {
            "id": "scope_table",
            "title": "What this phase establishes—and what it does not",
            "subtitle": "Raw verification is ready to freeze; later point-in-time and backtest gates remain blocked",
            "showDescription": True,
            "dataset": "scope_rows",
            "defaultSort": {"field": "claim", "direction": "asc"},
            "density": "spacious",
            "sourceId": "scope_query",
            "layout": "full",
            "columns": [
                {"field": "claim", "label": "Claim", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "evidence", "label": "Evidence or boundary", "type": "text"},
                {"field": "next_phase", "label": "Owning gate", "type": "text"},
            ],
        },
    ]
    charts = [
        {
            "id": "family_rows_chart",
            "title": "Parsed rows by source family",
            "subtitle": "All 683.5M logical data rows; volume indicates processing load, not data quality",
            "showDescription": True,
            "intent": "comparison",
            "question": "Which source families would dominate Phase 2 processing work after the explicit Phase 1 freeze?",
            "rationale": "A sorted horizontal bar chart is the simplest comparison for eight families with long labels and one common magnitude measure.",
            "type": "horizontalBar",
            "dataset": "family_summary",
            "sourceId": "family_query",
            "encodings": {
                "x": {
                    "field": "family",
                    "type": "nominal",
                    "label": "Source family",
                },
                "y": {
                    "field": "parsed_rows",
                    "type": "quantitative",
                    "format": "compact",
                    "label": "Parsed rows",
                },
                "tooltip": [
                    {"field": "parsed_rows", "type": "quantitative", "format": "number", "label": "Parsed rows"},
                    {"field": "file_count", "type": "quantitative", "format": "number", "label": "Files"},
                    {"field": "size_gb", "type": "quantitative", "format": "number", "label": "Size (GB)"},
                ],
            },
            "valueFormat": "compact",
            "layout": "full",
            "maxRows": 8,
            "labels": {"values": "all"},
            "palette": {"kind": "sequential", "name": "blue"},
            "settings": {"orientation": "horizontal", "showValues": True, "sort": "descending"},
            "surface": {"surface": "export", "interactiveLegend": False, "showControls": False, "viewMode": "visualization"},
        }
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Phase 1 Raw Data Snapshot: Verification Report", "layout": "full"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## Verification passed; Phase 1 is ready for explicit freeze\n\n"
                "The raw snapshot passed both the primary and independent implementations. "
                "All **25 files**, **683,495,563 logical data rows**, and **76.651 GB** were reconciled; "
                "the primary run recorded **585 passed checks, zero warnings, and zero failures**, while the "
                "independent run found **zero differences**.\n\n"
                "This result establishes raw-byte identity, CSV parseability, registered fields, and configured "
                "date coverage. It does **not** yet establish point-in-time investability, leakage-safe features, "
                "model performance, or portfolio performance. **Phase 2 remains blocked until an explicit Phase 1 "
                "freeze is published and verified.**"
            ),
            "layout": "full",
            "sourceId": "phase_01_summary",
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards], "layout": "full"},
        {
            "id": "independent_agreement",
            "type": "markdown",
            "body": (
                "## Two independent implementations agree on every registered source\n\n"
                "The primary pass read ordered headers with Python `csv.reader`, streamed rows with PyArrow, and "
                "computed SHA-256 with Python `hashlib`. The verification pass used macOS `shasum` plus the pandas "
                "C CSV engine. Across these independent implementations, every byte hash, ordered header, logical "
                "row count, and configured date profile agreed. The verification run also rechecked "
                "the full registry and every earlier file at completion, preventing a mixed snapshot from being published."
            ),
            "layout": "full",
            "sourceId": "independent_verification",
        },
        {
            "id": "workload_interpretation",
            "type": "markdown",
            "body": (
                "## Processing load is concentrated in OptionMetrics and SEC/13F\n\n"
                "OptionMetrics contains **449,806,277 rows**, about two-thirds of the registered observations, "
                "while SEC/13F contributes another **104,450,393 rows**. If the explicit Phase 1 freeze succeeds, "
                "this concentration will matter for Phase 2 storage, partitioning, and incremental validation. It is "
                "a workload signal only: a larger source is not more reliable or more predictive."
            ),
            "layout": "full",
            "sourceId": "family_query",
        },
        {"id": "family_rows_visual", "type": "chart", "chartId": "family_rows_chart", "layout": "full"},
        {"id": "family_summary", "type": "table", "tableId": "family_summary_table", "layout": "full"},
        {
            "id": "coverage_interpretation",
            "type": "markdown",
            "body": (
                "## Broad coverage still requires point-in-time alignment\n\n"
                "The registered sources span the intended research era, but their end dates differ: some extend into "
                "2026, OptionMetrics ends during 2025, and the liquidity series ends in 2024. These are observation or "
                "coverage dates only. After an explicit Phase 1 freeze, Phase 2 must construct public-availability dates, "
                "enforce lags, resolve identifiers, and filter the methodology calendar before any row can enter a "
                "feature or label."
            ),
            "layout": "full",
            "sourceId": "file_query",
        },
        {"id": "file_registry", "type": "table", "tableId": "file_registry_table", "layout": "full"},
        {
            "id": "scope_interpretation",
            "type": "markdown",
            "body": (
                "## The gate proves integrity, not point-in-time eligibility\n\n"
                "Phase 1 intentionally makes no global uniqueness claim for diagnostic keys and no inference from a "
                "coverage date to an investor's information set. Keeping that boundary explicit prevents raw-data "
                "success from being mistaken for a valid backtest."
            ),
            "layout": "full",
            "sourceId": "primary_manifest",
        },
        {"id": "scope", "type": "table", "tableId": "scope_table", "layout": "full"},
        {
            "id": "robustness",
            "type": "markdown",
            "body": (
                "## Robustness is strong within the Phase 1 claim\n\n"
                "- Date values are partitioned exactly once as blank, sentinel, valid, or invalid, and every profile reconciles.\n"
                "- Impossible calendar dates, impossible times, trailing garbage, malformed records, registry changes, and file replacement are hard-gated.\n"
                "- The primary ledger records 585 passed checks with zero warnings and zero failures, including end-of-run stability checks for every file."
            ),
            "layout": "full",
            "sourceId": "primary_quality_checks",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limitations remain explicit\n\n"
                "- No separate extraction-provenance file was supplied, so filesystem times and maximum observation dates are not treated as source as-of dates.\n"
                "- Point-in-time joins, accounting-report lags, filing acceptance times, identifier resolution, and feature eligibility remain untested.\n"
                "- Current-vintage metadata is documented as prohibited; after the explicit freeze, Phase 2 must enforce that rule with a machine-readable feature allowlist or denylist.\n"
                "- No model has been fit and no development, validation, or final-test return has been observed."
            ),
            "layout": "full",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Next permitted step: publish and verify the explicit Phase 1 freeze\n\n"
                "The raw evidence is verified and ready to freeze, but **Phase 2 is not yet permitted**.\n\n"
                "1. Re-authenticate the methodology, primary build, independent verification, report, notebook, tests, and configuration.\n"
                "2. Complete the final full-file raw hash and stability recheck.\n"
                "3. Publish the immutable Phase 1 freeze and verify its authoritative frozen pointer.\n"
                "4. Only after that freeze validates, begin the availability-aligned canonical panel in Phase 2."
            ),
            "layout": "full",
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Questions queued for Phase 2 after the explicit Phase 1 freeze\n\n"
                "- Which source-specific timestamp and lag produces the earliest defensible public-availability date?\n"
                "- How many CRSP securities are lost or multiplied at each identifier-linking step?\n"
                "- Which sources begin late enough to require missingness flags or optional feature blocks?\n"
                "- Can the canonical monthly universe reproduce CRSP total returns, including delistings, before any ML target is formed?"
            ),
            "layout": "full",
        },
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Phase 1 Raw Data Snapshot: Verification Report",
        "description": "Independent verification and pre-freeze scope boundary for the thesis raw-data snapshot.",
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "headline": [headline],
            "family_summary": context["family_summary"],
            "file_summary": context["file_summary"],
            "scope_rows": context["scope_rows"],
        },
        "accessIssues": [],
    }
    return {
        "surface": "report",
        "provenance": report_provenance(context),
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty review table: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_review_database(path: Path, context: dict[str, Any]) -> None:
    """Materialize the reviewed report rows so report provenance is runnable SQL."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            CREATE TABLE phase_01_headline (
                file_count INTEGER NOT NULL,
                parsed_rows INTEGER NOT NULL,
                size_gb REAL NOT NULL,
                primary_pass_count INTEGER NOT NULL,
                independent_difference_count INTEGER NOT NULL
            );
            CREATE TABLE phase_01_family_summary (
                family TEXT PRIMARY KEY,
                file_count INTEGER NOT NULL,
                parsed_rows INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                size_gb REAL NOT NULL,
                coverage_start TEXT,
                coverage_end TEXT
            );
            CREATE TABLE phase_01_file_summary (
                filename TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                parsed_rows INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                size_gb REAL NOT NULL,
                coverage_axis TEXT NOT NULL,
                coverage_start TEXT,
                coverage_end TEXT,
                primary_status TEXT NOT NULL,
                independent_status TEXT NOT NULL,
                stable_through_verification INTEGER NOT NULL CHECK (stable_through_verification IN (0, 1))
            );
            CREATE TABLE phase_01_scope (
                claim TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                evidence TEXT NOT NULL,
                next_phase TEXT NOT NULL
            );
            CREATE TABLE phase_01_provenance (
                raw_snapshot_id TEXT NOT NULL,
                phase_01_build_id TEXT NOT NULL,
                verification_id TEXT NOT NULL,
                primary_manifest_sha256 TEXT NOT NULL,
                independent_check_sha256 TEXT NOT NULL,
                primary_implementation_sha256 TEXT NOT NULL,
                verifier_implementation_sha256 TEXT NOT NULL
            );
            """
        )
        summary = context["summary"]
        connection.execute(
            "INSERT INTO phase_01_headline VALUES (?, ?, ?, ?, ?)",
            (
                summary["file_count"],
                summary["parsed_rows"],
                summary["size_gb"],
                summary["primary_pass_count"],
                summary["independent_difference_count"],
            ),
        )
        connection.executemany(
            "INSERT INTO phase_01_family_summary VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["family"],
                    row["file_count"],
                    row["parsed_rows"],
                    row["size_bytes"],
                    row["size_gb"],
                    row["coverage_start"],
                    row["coverage_end"],
                )
                for row in context["family_summary"]
            ],
        )
        connection.executemany(
            "INSERT INTO phase_01_file_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["filename"],
                    row["family"],
                    row["parsed_rows"],
                    row["size_bytes"],
                    row["size_gb"],
                    row["coverage_axis"],
                    row["coverage_start"],
                    row["coverage_end"],
                    row["primary_status"],
                    row["independent_status"],
                    int(row["stable_through_verification"]),
                )
                for row in context["file_summary"]
            ],
        )
        connection.executemany(
            "INSERT INTO phase_01_scope VALUES (?, ?, ?, ?)",
            [
                (row["claim"], row["status"], row["evidence"], row["next_phase"])
                for row in context["scope_rows"]
            ],
        )
        provenance = provenance_record(context)
        connection.execute(
            """INSERT INTO phase_01_provenance (
                   raw_snapshot_id, phase_01_build_id, verification_id,
                   primary_manifest_sha256, independent_check_sha256,
                   primary_implementation_sha256, verifier_implementation_sha256
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            tuple(
                provenance[field]
                for field in (
                    "raw_snapshot_id",
                    "phase_01_build_id",
                    "verification_id",
                    "primary_manifest_sha256",
                    "independent_check_sha256",
                    "primary_implementation_sha256",
                    "verifier_implementation_sha256",
                )
            ),
        )
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "phase_01_headline",
                "phase_01_family_summary",
                "phase_01_file_summary",
                "phase_01_scope",
                "phase_01_provenance",
            )
        }
        expected = {
            "phase_01_headline": 1,
            "phase_01_family_summary": len(context["family_summary"]),
            "phase_01_file_summary": 25,
            "phase_01_scope": len(context["scope_rows"]),
            "phase_01_provenance": 1,
        }
        if counts != expected:
            raise RuntimeError(f"Review database row counts do not reconcile: {counts} != {expected}")
    finally:
        connection.close()
    os.replace(temporary, path)


def _query_review_database(path: Path, query: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(query).fetchall()]
    finally:
        connection.close()


def build_source_notes(context: dict[str, Any]) -> str:
    """Return the canonical source notes frozen with the Phase 1 report."""
    return f"""# Phase 1 report source notes

- Audience: technical thesis reviewer.
- Delivery mode: portable HTML fallback because no MCP artifact validator/renderer is callable in this runtime.
- Decision: whether the raw snapshot is verified and ready for the explicit Phase 1 freeze. Phase 2 remains blocked until that freeze is published and verified.
- Scope: raw byte identity, registry, CSV structure, parseability, configured date coverage, and diagnostic key formatting.
- Evidence as of: {context['summary']['verification_completed_utc']}.
- Primary comparison: Python `csv.reader` headers, PyArrow row streaming, and Python `hashlib` SHA-256 versus the pandas C-engine/macOS `shasum` independent pass.
- Report queries: runnable SQLite statements over `reports/phase_01_raw_snapshot/phase_01_review.sqlite` materialize every card, chart, and table row.
- Chart map: `Processing load is concentrated in OptionMetrics and SEC/13F` asks which families would dominate Phase 2 work after the explicit Phase 1 freeze; it uses a sorted horizontal `bar` over `family` and `parsed_rows`, a single blue-root palette with direct value labels, the `family_summary` dataset, and the portable HTML report as its QA surface.
- Chart caveat: row volume is used only as a processing-load measure; exact quality and coverage judgments remain in the audit tables.
- Required technical-report roles: title; technical summary; key findings and exact tables; scope and definitions; methodology; limitations and robustness; next steps; further questions.
- Uncertainty boundary: Phase 1 does not establish point-in-time availability, canonical uniqueness, leakage safety, model skill, or portfolio performance.
- Primary build: {context['summary']['phase_01_build_id']}.
- Independent verification: {context['summary']['verification_id']}.
- Raw snapshot: {context['summary']['raw_snapshot_id']}.
- Primary manifest SHA-256: {context['primary_manifest_sha256']}.
- Independent check SHA-256: {context['independent_check_sha256']}.
- Verifier implementation SHA-256: {context['independent']['verifier_implementation_sha256']}.
- Standalone CSV exports repeat the evidence IDs and hashes on every row; the review database stores the same identity once in `phase_01_provenance`.
"""


def write_review_artifacts(
    project_root: Path = PROJECT_ROOT, generated_at: str | None = None
) -> dict[str, Path]:
    context = load_review_context(project_root)
    failures = validation_failures(context)
    if failures:
        raise RuntimeError("Phase 1 review failed: " + "; ".join(failures))
    report_directory = project_root.resolve() / REPORT_ROOT
    report_directory.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "summary": context["summary"],
        "provenance": report_provenance(context),
        "evidence_reconciliation": context["reconciliation"],
        "validation_failures": failures,
        "verified_pointer": context["verified_pointer"],
        "scope_boundary": context["manifest"]["scope_boundary"],
    }
    _write_json(report_directory / "phase_01_summary.json", summary_payload)
    _write_csv(
        report_directory / "source_family_summary.csv",
        with_provenance_columns(context, context["family_summary"]),
    )
    _write_csv(
        report_directory / "source_file_summary.csv",
        with_provenance_columns(context, context["file_summary"]),
    )
    database_path = report_directory / "phase_01_review.sqlite"
    _write_review_database(database_path, context)
    artifact = build_report_artifact(context, generated_at=generated_at)
    artifact["snapshot"]["datasets"] = {
        "headline": _query_review_database(database_path, HEADLINE_SQL),
        "family_summary": _query_review_database(database_path, FAMILY_SQL),
        "file_summary": _query_review_database(database_path, FILE_SQL),
        "scope_rows": _query_review_database(database_path, SCOPE_SQL),
    }
    _write_json(report_directory / "artifact.json", artifact)
    source_notes = build_source_notes(context)
    source_notes_path = report_directory / "source_notes.md"
    temporary_notes = source_notes_path.with_suffix(".md.tmp")
    temporary_notes.write_text(source_notes, encoding="utf-8")
    os.replace(temporary_notes, source_notes_path)
    return {
        "report_directory": report_directory,
        "summary": report_directory / "phase_01_summary.json",
        "families": report_directory / "source_family_summary.csv",
        "files": report_directory / "source_file_summary.csv",
        "artifact": report_directory / "artifact.json",
        "database": database_path,
        "source_notes": report_directory / "source_notes.md",
    }
