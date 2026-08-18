"""Independent verifier for the Phase 2 observed-date calendar build.

The verifier deliberately does not import :mod:`phase_02_calendar`.  It
authenticates the verified Phase 2 contract, the primary calendar receipt,
and the Phase 1 frozen raw identity; projects only ``DlyCalDt`` from the raw
CRSP daily CSV; and independently reconstructs both published calendars.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from portfolio_bot.data.database_location import resolve_recorded_database_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASE_CONFIG = PROJECT_ROOT / "configs" / "phase_02_canonical_panel.toml"
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.toml"
DEFAULT_CONTRACT_POINTER = (
    PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel" / "LATEST_CONTRACT_VERIFIED.json"
)
DEFAULT_PRIMARY_POINTER = (
    PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel" / "LATEST_CALENDAR_PRIMARY.json"
)
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel"
EXPECTED_CORRECTED_BUILD_ID = "calendar-ae2e4ba4a280d66e"
REJECTED_BUILD_IDS = frozenset({"calendar-dc8cf148817d4ea0"})
BYTE_CHUNK_SIZE = 8 << 20

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
PRIMARY_CHECK_FIELDS = ("check_id", "status", "expected", "actual", "evidence", "message")
DIFFERENCE_FIELDS = (
    "check_id",
    "category",
    "expected",
    "actual",
    "evidence",
    "message",
)
EXPECTED_OUTPUT_BASENAMES = frozenset(
    {
        "_SUCCESS.json",
        "rebalance_calendar.csv",
        "rebalance_calendar.parquet",
        "trading_calendar.csv",
        "trading_calendar.parquet",
        "calendar_checks.csv",
        "calendar_summary.json",
        "input_receipt.json",
    }
)

EXPECTED_BOUNDARY_IDS = (
    "last_development_eligible",
    "development_validation_embargo",
    "first_validation",
    "last_validation_eligible",
    "validation_final_embargo",
    "first_primary_final",
    "last_full_feature_primary",
    "partial_december",
)


class CalendarVerificationError(RuntimeError):
    """Raised when a verification result cannot be published safely."""


@dataclasses.dataclass(frozen=True)
class VerificationSettings:
    project_root: Path
    phase_config_path: Path
    paths_config_path: Path
    contract_pointer_path: Path
    primary_pointer_path: Path
    artifact_root: Path
    database_root: Path
    raw_path: Path
    phase_config: dict[str, Any]
    paths_config: dict[str, Any]
    expected_build_id: str | None = EXPECTED_CORRECTED_BUILD_ID
    block_size_bytes: int = 64 << 20


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = BYTE_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalendarVerificationError(f"Expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(_jsonable(row.get(field)), sort_keys=True, separators=(",", ":"))
                    if isinstance(row.get(field), (dict, list, tuple, bool))
                    else "" if row.get(field) is None else str(row.get(field))
                    for field in fields
                }
            )


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _project_label(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_settings(
    phase_config_path: Path = DEFAULT_PHASE_CONFIG,
    paths_config_path: Path = DEFAULT_PATHS_CONFIG,
    *,
    contract_pointer_path: Path = DEFAULT_CONTRACT_POINTER,
    primary_pointer_path: Path = DEFAULT_PRIMARY_POINTER,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    raw_path: Path | None = None,
    project_root: Path = PROJECT_ROOT,
    expected_build_id: str | None = EXPECTED_CORRECTED_BUILD_ID,
    block_size_bytes: int = 64 << 20,
) -> VerificationSettings:
    project_root = project_root.resolve()
    phase_config_path = phase_config_path.resolve()
    paths_config_path = paths_config_path.resolve()
    with phase_config_path.open("rb") as handle:
        phase = tomllib.load(handle)
    with paths_config_path.open("rb") as handle:
        paths = tomllib.load(handle)
    configured_database = Path(paths["database"]["root"]).expanduser()
    database_root = Path(os.environ.get("PORTFOLIO_DATABASE_ROOT", configured_database)).expanduser().resolve()
    calendar_owner = phase["calendar"]["calendar_owner_file"]
    resolved_raw = raw_path or database_root / paths["database"]["raw"] / calendar_owner
    return VerificationSettings(
        project_root=project_root,
        phase_config_path=phase_config_path,
        paths_config_path=paths_config_path,
        contract_pointer_path=contract_pointer_path.resolve(),
        primary_pointer_path=primary_pointer_path.resolve(),
        artifact_root=artifact_root.resolve(),
        database_root=database_root,
        raw_path=resolved_raw.resolve(),
        phase_config=phase,
        paths_config=paths,
        expected_build_id=expected_build_id,
        block_size_bytes=block_size_bytes,
    )


def operational_stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "size_bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


class _Checks:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        check_id: str,
        category: str,
        *,
        passed: bool,
        expected: Any,
        actual: Any,
        evidence: str | Path,
        message: str,
    ) -> None:
        evidence_text = (
            _project_label(evidence, self.project_root) if isinstance(evidence, Path) else evidence
        )
        self.rows.append(
            {
                "check_id": check_id,
                "category": category,
                "status": "PASS" if passed else "FAIL",
                "expected": _jsonable(expected),
                "actual": _jsonable(actual),
                "evidence": evidence_text,
                "message": message,
            }
        )


def _parse_hash_file(path: Path, *, basename_only: bool = False) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    problems: list[str] = []
    if not path.is_file():
        return values, ["missing hash file"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        digest, separator, label = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest) or not label:
            problems.append(f"malformed line {line_number}")
            continue
        if label in values:
            problems.append(f"duplicate label {label!r}")
            continue
        if basename_only and Path(label).name != label:
            problems.append(f"non-basename label {label!r}")
            continue
        values[label] = digest
    return values, problems


def _authenticate_contract(settings: VerificationSettings, checks: _Checks) -> dict[str, Any]:
    pointer = _read_json(settings.contract_pointer_path)
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
    checks.add(
        "CONTRACT_POINTER_COMPLETE_AND_PASSED",
        "contract_authentication",
        passed=required.issubset(pointer) and pointer.get("status") == "verified_passed",
        expected={"required_fields": sorted(required), "status": "verified_passed"},
        actual={"fields": sorted(pointer), "status": pointer.get("status")},
        evidence=settings.contract_pointer_path,
        message="The calendar consumes a complete independently-passed contract pointer.",
    )
    contract_dir = _project_path(pointer["contract_path"], settings.project_root)
    verification_dir = _project_path(pointer["verification_path"], settings.project_root)
    contract_json_path = contract_dir / "contract.json"
    hash_list_path = contract_dir / "artifact_hashes.sha256"
    independent_path = verification_dir / "independent_check.json"
    differences_path = verification_dir / "differences.csv"
    pointer_hashes = {
        contract_json_path: pointer.get("contract_json_sha256"),
        hash_list_path: pointer.get("artifact_hashes_sha256"),
        independent_path: pointer.get("independent_check_sha256"),
        differences_path: pointer.get("differences_sha256"),
    }
    pointer_actual = {
        _project_label(path, settings.project_root): None if not path.is_file() else sha256_file(path)
        for path in pointer_hashes
    }
    pointer_expected = {
        _project_label(path, settings.project_root): digest for path, digest in pointer_hashes.items()
    }
    checks.add(
        "CONTRACT_POINTER_HASHES_EXACT",
        "contract_authentication",
        passed=pointer_actual == pointer_expected,
        expected=pointer_expected,
        actual=pointer_actual,
        evidence=settings.contract_pointer_path,
        message="All contract artifacts named by the verified pointer retain their exact bytes.",
    )

    sealed_hashes, hash_problems = _parse_hash_file(hash_list_path, basename_only=True)
    actual_artifact_files = sorted(path.name for path in contract_dir.iterdir() if path.is_file())
    expected_artifact_files = sorted([*sealed_hashes, "artifact_hashes.sha256"])
    sealed_actual = {
        label: None if not (contract_dir / label).is_file() else sha256_file(contract_dir / label)
        for label in sealed_hashes
    }
    checks.add(
        "VERIFIED_CONTRACT_ARTIFACT_SET_AND_HASHES_EXACT",
        "contract_authentication",
        passed=(
            not hash_problems
            and actual_artifact_files == expected_artifact_files
            and sealed_actual == sealed_hashes
        ),
        expected={"files": expected_artifact_files, "hashes": sealed_hashes, "problems": []},
        actual={"files": actual_artifact_files, "hashes": sealed_actual, "problems": hash_problems},
        evidence=hash_list_path,
        message="The complete sealed contract directory is authenticated, not only contract.json.",
    )
    contract = _read_json(contract_json_path)
    independent = _read_json(independent_path)
    differences_fields, differences = _read_csv(differences_path)
    contract_semantics = contract.get("binding", {}).get("semantic_contract", {})
    phase_semantics = settings.phase_config
    checks.add(
        "CONTRACT_IDENTIFIERS_AND_INDEPENDENT_PASS_EXACT",
        "contract_authentication",
        passed=(
            contract.get("contract_id") == pointer.get("contract_id")
            and independent.get("contract_id") == pointer.get("contract_id")
            and independent.get("verification_id") == pointer.get("verification_id")
            and independent.get("status") == "verified_passed"
            and independent.get("summary", {}).get("fail_count") == 0
            and independent.get("summary", {}).get("difference_count") == 0
            and not differences
        ),
        expected={
            "contract_id": pointer.get("contract_id"),
            "verification_id": pointer.get("verification_id"),
            "status": "verified_passed",
            "fail_count": 0,
            "difference_count": 0,
            "differences": 0,
        },
        actual={
            "contract_id": contract.get("contract_id"),
            "independent_contract_id": independent.get("contract_id"),
            "verification_id": independent.get("verification_id"),
            "status": independent.get("status"),
            "fail_count": independent.get("summary", {}).get("fail_count"),
            "difference_count": independent.get("summary", {}).get("difference_count"),
            "differences": len(differences),
            "difference_fields": differences_fields,
        },
        evidence=independent_path,
        message="The upstream contract's own independent verification is an exact zero-difference pass.",
    )
    phase_hash_label = _project_label(settings.phase_config_path, settings.project_root)
    bound_phase_hash = contract.get("binding", {}).get("input_hashes", {}).get(phase_hash_label)
    checks.add(
        "PHASE_CONFIG_AUTHENTIC_AND_MATCHES_CONTRACT_SEMANTICS",
        "contract_authentication",
        passed=(
            bound_phase_hash == sha256_file(settings.phase_config_path)
            and contract_semantics == phase_semantics
        ),
        expected={
            "phase_config_sha256": bound_phase_hash,
            "semantic_contract_sha256": sha256_json(contract_semantics),
        },
        actual={
            "phase_config_sha256": sha256_file(settings.phase_config_path),
            "semantic_contract_sha256": sha256_json(phase_semantics),
        },
        evidence=contract_json_path,
        message="The locally interpreted Phase 2 calendar contract is the exact verified semantic contract.",
    )
    source_fields, source_rows = _read_csv(contract_dir / "source_files.csv")
    boundary_fields, boundary_rows = _read_csv(contract_dir / "calendar_boundary_expectations.csv")
    return {
        "pointer": pointer,
        "contract_dir": contract_dir,
        "verification_dir": verification_dir,
        "contract": contract,
        "independent": independent,
        "source_fields": source_fields,
        "source_rows": source_rows,
        "boundary_fields": boundary_fields,
        "boundary_rows": boundary_rows,
    }


def _authenticate_frozen_raw(
    settings: VerificationSettings,
    contract: Mapping[str, Any],
    checks: _Checks,
) -> dict[str, Any]:
    freeze_path = _project_path(settings.phase_config["phase_01_freeze_manifest"], settings.project_root)
    freeze = _read_json(freeze_path)
    expected_freeze_hash = settings.phase_config["phase_01_freeze_manifest_sha256"]
    actual_freeze_hash = sha256_file(freeze_path)
    checks.add(
        "PHASE_01_FREEZE_MANIFEST_AUTHENTIC",
        "raw_authentication",
        passed=(
            actual_freeze_hash == expected_freeze_hash
            and freeze.get("identity", {}).get("status") == "phase_01_frozen"
            and freeze.get("freeze_id") == settings.phase_config["phase_01_freeze_id"]
        ),
        expected={
            "sha256": expected_freeze_hash,
            "status": "phase_01_frozen",
            "freeze_id": settings.phase_config["phase_01_freeze_id"],
        },
        actual={
            "sha256": actual_freeze_hash,
            "status": freeze.get("identity", {}).get("status"),
            "freeze_id": freeze.get("freeze_id"),
        },
        evidence=freeze_path,
        message="The operational raw identity is anchored in the exact Phase 1 freeze.",
    )
    operational_entries = freeze.get("identity", {}).get("evidence_groups", {}).get(
        "operational_evidence", []
    )
    entry = next(
        (
            item
            for item in operational_entries
            if Path(str(item.get("path", ""))).name == "raw_freeze_recheck_metadata.json"
        ),
        None,
    )
    if not entry:
        raise CalendarVerificationError("Phase 1 freeze lacks raw recheck operational evidence")
    recheck_path = _project_path(entry["path"], settings.project_root)
    recheck = _read_json(recheck_path)
    recheck_hash = sha256_file(recheck_path)
    filename = settings.raw_path.name
    raw_meta = recheck.get("per_file_operational_metadata", {}).get(filename, {})
    end_stat = raw_meta.get("end_of_complete_recheck_stat", {})
    stat_fields = ("device", "inode", "mode", "size_bytes", "mtime_ns", "ctime_ns")
    expected_stat = {field: int(end_stat[field]) for field in stat_fields}
    checks.add(
        "RAW_FREEZE_RECHECK_AUTHENTIC_AND_PASSED",
        "raw_authentication",
        passed=(
            recheck_hash == entry.get("sha256")
            and recheck.get("status") == "passed"
            and recheck.get("all_hashes_match_primary") is True
            and recheck.get("all_files_stable_through_entire_recheck") is True
            and raw_meta.get("stable_through_entire_recheck") is True
        ),
        expected={
            "sha256": entry.get("sha256"),
            "status": "passed",
            "all_hashes_match_primary": True,
            "all_files_stable_through_entire_recheck": True,
            "raw_file_stable": True,
        },
        actual={
            "sha256": recheck_hash,
            "status": recheck.get("status"),
            "all_hashes_match_primary": recheck.get("all_hashes_match_primary"),
            "all_files_stable_through_entire_recheck": recheck.get(
                "all_files_stable_through_entire_recheck"
            ),
            "raw_file_stable": raw_meta.get("stable_through_entire_recheck"),
        },
        evidence=recheck_path,
        message="The Phase 1 full-byte raw recheck and per-file stability receipt remain authentic.",
    )
    daily_rows = [
        row for row in contract["source_rows"] if row.get("filename") == filename
    ]
    checks.add(
        "CONTRACT_HAS_ONE_CALENDAR_OWNER_RAW_ROW",
        "raw_authentication",
        passed=len(daily_rows) == 1 and daily_rows[0].get("calendar_owner") == "true",
        expected={"matching_rows": 1, "calendar_owner": "true"},
        actual={
            "matching_rows": len(daily_rows),
            "calendar_owner": daily_rows[0].get("calendar_owner") if daily_rows else None,
        },
        evidence=contract["contract_dir"] / "source_files.csv",
        message="Exactly one verified source row owns the observed CRSP calendar.",
    )
    return {
        "freeze_path": freeze_path,
        "freeze": freeze,
        "recheck_path": recheck_path,
        "recheck_sha256": recheck_hash,
        "expected_stat": expected_stat,
        "source_row": daily_rows[0],
    }


def _authenticate_primary(
    settings: VerificationSettings,
    contract: Mapping[str, Any],
    checks: _Checks,
) -> dict[str, Any]:
    pointer = _read_json(settings.primary_pointer_path)
    required = {
        "build_id",
        "contract_id",
        "artifact_path",
        "manifest_sha256",
        "status",
    }
    build_id = pointer.get("build_id")
    corrected = build_id not in REJECTED_BUILD_IDS and (
        settings.expected_build_id is None or build_id == settings.expected_build_id
    )
    checks.add(
        "CORRECTED_PRIMARY_BUILD_SELECTED",
        "primary_authentication",
        passed=required.issubset(pointer) and corrected,
        expected={
            "required_fields": sorted(required),
            "expected_build_id": settings.expected_build_id,
            "rejected_build_ids": sorted(REJECTED_BUILD_IDS),
        },
        actual={"fields": sorted(pointer), "build_id": build_id},
        evidence=settings.primary_pointer_path,
        message="The superseded calendar is never eligible for a passing verification.",
    )
    artifact_dir = _project_path(pointer["artifact_path"], settings.project_root)
    manifest_path = artifact_dir / "manifest.json"
    output_hash_path = artifact_dir / "output_hashes.sha256"
    artifact_files = sorted(path.name for path in artifact_dir.iterdir() if path.is_file())
    manifest_hash = sha256_file(manifest_path)
    checks.add(
        "PRIMARY_ARTIFACT_SET_AND_POINTER_HASH_EXACT",
        "primary_authentication",
        passed=(
            artifact_files == ["manifest.json", "output_hashes.sha256"]
            and manifest_hash == pointer.get("manifest_sha256")
        ),
        expected={
            "files": ["manifest.json", "output_hashes.sha256"],
            "manifest_sha256": pointer.get("manifest_sha256"),
        },
        actual={"files": artifact_files, "manifest_sha256": manifest_hash},
        evidence=artifact_dir,
        message="The primary artifact directory and manifest bytes exactly match its pointer.",
    )
    manifest = _read_json(manifest_path)
    recomputed_build_id = "calendar-" + sha256_json(manifest.get("binding", {}))[:16]
    identity_ok = (
        manifest.get("build_id") == build_id
        and manifest.get("contract_id") == contract["pointer"].get("contract_id")
        and manifest.get("verification_id") == contract["pointer"].get("verification_id")
        and manifest.get("status") == "calendar_primary_passed_pending_independent_verification"
        and recomputed_build_id == build_id
    )
    checks.add(
        "PRIMARY_BINDING_AND_BUILD_ID_RECOMPUTED",
        "primary_authentication",
        passed=identity_ok,
        expected={
            "build_id": build_id,
            "contract_id": contract["pointer"].get("contract_id"),
            "verification_id": contract["pointer"].get("verification_id"),
            "status": "calendar_primary_passed_pending_independent_verification",
        },
        actual={
            "build_id": manifest.get("build_id"),
            "recomputed_build_id": recomputed_build_id,
            "contract_id": manifest.get("contract_id"),
            "verification_id": manifest.get("verification_id"),
            "status": manifest.get("status"),
        },
        evidence=manifest_path,
        message="The primary build ID is independently reproduced from its complete binding.",
    )
    hash_rows, hash_problems = _parse_hash_file(output_hash_path)
    manifest_outputs = manifest.get("outputs", [])
    manifest_map = {
        str(item.get("path")): str(item.get("sha256"))
        for item in manifest_outputs
        if isinstance(item, dict)
    }
    manifest_sizes = {
        str(item.get("path")): item.get("size_bytes")
        for item in manifest_outputs
        if isinstance(item, dict)
    }
    resolved_output_paths = {
        label: resolve_recorded_database_path(
            label, current_database_root=settings.database_root
        )
        for label in manifest_map
    }
    actual_output_hashes = {
        label: None if not path.is_file() else sha256_file(path)
        for label, path in resolved_output_paths.items()
    }
    actual_output_sizes = {
        label: None if not path.is_file() else path.stat().st_size
        for label, path in resolved_output_paths.items()
    }
    manifest_labels = set(manifest_map)
    basename_set = {Path(label).name for label in manifest_labels}
    output_exact = (
        not hash_problems
        and len(manifest_outputs) == len(manifest_map) == 8
        and hash_rows == manifest_map
        and basename_set == EXPECTED_OUTPUT_BASENAMES
        and actual_output_hashes == manifest_map
        and actual_output_sizes == manifest_sizes
    )
    checks.add(
        "PRIMARY_EIGHT_OUTPUTS_HASH_SIZE_AND_RECEIPT_EXACT",
        "primary_authentication",
        passed=output_exact,
        expected={
            "output_count": 8,
            "basenames": sorted(EXPECTED_OUTPUT_BASENAMES),
            "hashes": manifest_map,
            "sizes": manifest_sizes,
            "hash_file_problems": [],
        },
        actual={
            "output_count": len(manifest_outputs),
            "unique_paths": len(manifest_map),
            "basenames": sorted(basename_set),
            "hashes": actual_output_hashes,
            "sizes": actual_output_sizes,
            "hash_file_matches_manifest": hash_rows == manifest_map,
            "hash_file_problems": hash_problems,
        },
        evidence=output_hash_path,
        message="All eight primary data and quality outputs match both hash receipts and manifest sizes.",
    )
    output_by_name = {
        Path(label).name: resolved_output_paths[label] for label in manifest_labels
    }
    if set(output_by_name) == EXPECTED_OUTPUT_BASENAMES:
        namespace = settings.phase_config["storage"]["interim_namespace"]
        contract_id = contract["pointer"]["contract_id"]
        expected_interim_dir = (
            settings.database_root
            / settings.paths_config["database"]["interim"]
            / namespace
            / contract_id
            / "calendar"
            / build_id
        ).resolve()
        expected_quality_dir = (
            settings.database_root
            / settings.paths_config["database"]["quality"]
            / settings.phase_config["storage"]["quality_namespace"]
            / contract_id
            / "calendar"
            / build_id
        ).resolve()
        interim_dir = output_by_name["trading_calendar.csv"].parent
        quality_dir = output_by_name["calendar_checks.csv"].parent
        expected_paths = {
            "_SUCCESS.json": expected_interim_dir / "_SUCCESS.json",
            "rebalance_calendar.csv": expected_interim_dir / "rebalance_calendar.csv",
            "rebalance_calendar.parquet": expected_interim_dir / "rebalance_calendar.parquet",
            "trading_calendar.csv": expected_interim_dir / "trading_calendar.csv",
            "trading_calendar.parquet": expected_interim_dir / "trading_calendar.parquet",
            "calendar_checks.csv": expected_quality_dir / "calendar_checks.csv",
            "calendar_summary.json": expected_quality_dir / "calendar_summary.json",
            "input_receipt.json": expected_quality_dir / "input_receipt.json",
        }
        checks.add(
            "PRIMARY_OUTPUT_PATHS_MATCH_STORAGE_CONTRACT",
            "primary_authentication",
            passed=(
                artifact_dir == settings.artifact_root / "calendar_builds" / build_id
                and output_by_name == expected_paths
            ),
            expected={
                "artifact_dir": settings.artifact_root / "calendar_builds" / build_id,
                "outputs": expected_paths,
            },
            actual={"artifact_dir": artifact_dir, "outputs": output_by_name},
            evidence=settings.primary_pointer_path,
            message="All primary files are in the contract's dedicated interim, quality, and artifact namespaces.",
        )
        actual_interim = sorted(path.name for path in interim_dir.iterdir() if path.is_file())
        actual_quality = sorted(path.name for path in quality_dir.iterdir() if path.is_file())
        expected_interim = sorted(
            {
                "_SUCCESS.json",
                "rebalance_calendar.csv",
                "rebalance_calendar.parquet",
                "trading_calendar.csv",
                "trading_calendar.parquet",
            }
        )
        expected_quality = sorted({"calendar_checks.csv", "calendar_summary.json", "input_receipt.json"})
        checks.add(
            "PRIMARY_OUTPUT_DIRECTORIES_HAVE_NO_UNSEALED_FILES",
            "primary_authentication",
            passed=actual_interim == expected_interim and actual_quality == expected_quality,
            expected={"interim": expected_interim, "quality": expected_quality},
            actual={"interim": actual_interim, "quality": actual_quality},
            evidence="primary interim and quality calendar directories",
            message="The published calendar directories contain exactly the eight sealed outputs.",
        )
    return {
        "pointer": pointer,
        "artifact_dir": artifact_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_hash,
        "output_hash_path": output_hash_path,
        "manifest": manifest,
        "output_by_name": output_by_name,
    }


def scan_raw_date_projection(
    path: Path,
    *,
    date_column: str = "DlyCalDt",
    block_size_bytes: int = 64 << 20,
) -> dict[str, Any]:
    """Project one CSV column and retain only its exact date-domain diagnostics."""
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
    schema_names_seen: set[tuple[str, ...]] = set()
    distinct_text: set[str] = set()
    for batch in reader:
        schema_names_seen.add(tuple(batch.schema.names))
        if batch.schema.names != [date_column]:
            raise CalendarVerificationError(
                f"Raw projection materialized unexpected columns: {batch.schema.names}"
            )
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
    invalid: list[str] = []
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for value in sorted(distinct_text):
        if not pattern.fullmatch(value):
            invalid.append(value)
            continue
        try:
            valid_dates.append(dt.date.fromisoformat(value))
        except ValueError:
            invalid.append(value)
    valid_dates.sort()
    return {
        "row_count": row_count,
        "null_count": null_count,
        "blank_count": blank_count,
        "distinct_nonblank_count": len(distinct_text),
        "valid_distinct_date_count": len(valid_dates),
        "invalid_distinct_values": invalid,
        "dates": valid_dates,
        "minimum": valid_dates[0].isoformat() if valid_dates else None,
        "maximum": valid_dates[-1].isoformat() if valid_dates else None,
        "requested_columns": [date_column],
        "materialized_schema_names": [list(names) for names in sorted(schema_names_seen)],
        "non_date_columns_materialized": 0 if schema_names_seen == {(date_column,)} else None,
        "engine": "pyarrow.csv.open_csv",
    }


def _next_month(value: tuple[int, int]) -> tuple[int, int]:
    year, month = value
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _independent_stage(holding_month: str) -> str:
    # These exact stage spans are the locked Methodology-v1 contract, expressed
    # independently of the primary builder implementation.
    if "2000-01" <= holding_month <= "2014-11":
        return "development"
    if holding_month in {"2014-12", "2019-12"}:
        return "embargo"
    if "2015-01" <= holding_month <= "2019-11":
        return "validation"
    if "2020-01" <= holding_month <= "2025-08":
        return "primary_final"
    if "2025-09" <= holding_month <= "2025-11":
        return "core_extension"
    if holding_month == "2025-12":
        return "extension_descriptive"
    return "outside_locked_scope"


def independently_reconstruct_calendars(dates: Sequence[dt.date]) -> dict[str, Any]:
    ordered = sorted(set(dates))
    by_month: dict[tuple[int, int], list[dt.date]] = defaultdict(list)
    for date in ordered:
        by_month[(date.year, date.month)].append(date)
    months = sorted(by_month)
    month_end_dates = {values[-1] for values in by_month.values()}
    trading_rows = [
        {
            "calendar_date": date,
            "year": date.year,
            "month": date.month,
            "weekday_iso": date.isoweekday(),
            "is_observed_month_end": date in month_end_dates,
        }
        for date in ordered
    ]
    rebalance_rows: list[dict[str, Any]] = []
    for index, current_month in enumerate(months[1:], start=1):
        holding_month = f"{current_month[0]:04d}-{current_month[1]:02d}"
        is_partial = index == len(months) - 1
        stage = _independent_stage(holding_month)
        rebalance_rows.append(
            {
                "holding_month": holding_month,
                "stage": stage,
                "signal_date": by_month[months[index - 1]][-1],
                "entry_date": by_month[current_month][0],
                "exit_date": (
                    by_month[current_month][-1]
                    if is_partial
                    else by_month[months[index + 1]][0]
                ),
                "is_embargo": stage == "embargo",
                "is_partial": is_partial,
                "full_feature_time_eligible": (
                    holding_month <= "2025-08" and stage != "embargo"
                ),
                "formal_inference_expected": stage == "primary_final" and not is_partial,
            }
        )
    return {
        "trading_rows": trading_rows,
        "rebalance_rows": rebalance_rows,
        "months": months,
        "month_continuity": all(
            right == _next_month(left) for left, right in zip(months, months[1:])
        ),
        "duplicate_input_dates_removed": len(dates) - len(ordered),
    }


def _parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"not a strict lowercase boolean: {value!r}")


def _load_trading_csv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    fields, rows = _read_csv(path)
    typed = [
        {
            "calendar_date": dt.date.fromisoformat(row["calendar_date"]),
            "year": int(row["year"]),
            "month": int(row["month"]),
            "weekday_iso": int(row["weekday_iso"]),
            "is_observed_month_end": _parse_bool(row["is_observed_month_end"]),
        }
        for row in rows
    ]
    return fields, typed


def _load_rebalance_csv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    fields, rows = _read_csv(path)
    typed = [
        {
            "holding_month": row["holding_month"],
            "stage": row["stage"],
            "signal_date": dt.date.fromisoformat(row["signal_date"]),
            "entry_date": dt.date.fromisoformat(row["entry_date"]),
            "exit_date": dt.date.fromisoformat(row["exit_date"]),
            "is_embargo": _parse_bool(row["is_embargo"]),
            "is_partial": _parse_bool(row["is_partial"]),
            "full_feature_time_eligible": _parse_bool(row["full_feature_time_eligible"]),
            "formal_inference_expected": _parse_bool(row["formal_inference_expected"]),
        }
        for row in rows
    ]
    return fields, typed


def _row_signature(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"row_count": len(rows), "sha256": sha256_json(list(rows))}


def _first_row_mismatch(
    expected: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return {"index": index, "expected": left, "actual": right}
    if len(expected) != len(actual):
        return {"index": min(len(expected), len(actual)), "expected": None, "actual": None}
    return None


def _compare_rows(
    checks: _Checks,
    check_id: str,
    category: str,
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    evidence: str | Path,
    message: str,
) -> None:
    checks.add(
        check_id,
        category,
        passed=list(expected) == list(actual),
        expected=_row_signature(expected),
        actual={**_row_signature(actual), "first_mismatch": _first_row_mismatch(expected, actual)},
        evidence=evidence,
        message=message,
    )


def _expected_primary_actuals(
    contract: Mapping[str, Any],
    frozen: Mapping[str, Any],
    stat_before: Mapping[str, int],
    stat_after: Mapping[str, int],
    scan: Mapping[str, Any],
    reconstructed: Mapping[str, Any],
) -> dict[str, Any]:
    builder_stat_fields = ("device", "inode", "mode", "size_bytes", "mtime_ns")
    before = {field: stat_before[field] for field in builder_stat_fields}
    after = {field: stat_after[field] for field in builder_stat_fields}
    by_holding = {
        row["holding_month"]: row for row in reconstructed["rebalance_rows"]
    }
    result: dict[str, Any] = {
        "VERIFIED_CONTRACT_GATE": "verified_passed",
        "RAW_FROZEN_IDENTITY_BEFORE_SCAN": before,
        "RAW_IDENTITY_STABLE_DURING_SCAN": after,
        "SCANNED_ROW_COUNT_MATCHES_PHASE_1": scan["row_count"],
        "DATE_COLUMN_ONLY": scan["requested_columns"],
        "DATE_VALUES_COMPLETE_AND_VALID": {
            "null_count": scan["null_count"],
            "blank_count": scan["blank_count"],
            "invalid_distinct_values": scan["invalid_distinct_values"],
        },
        "OBSERVED_DATE_COVERAGE_EXACT": {
            "minimum": scan["minimum"],
            "maximum": scan["maximum"],
        },
        "TRADING_CALENDAR_STRICTLY_INCREASING_UNIQUE": {
            "strictly_increasing": all(
                left < right for left, right in zip(scan["dates"], scan["dates"][1:])
            ),
            "duplicate_input_dates_removed": reconstructed["duplicate_input_dates_removed"],
        },
        "CALENDAR_MONTHS_CONTIGUOUS": reconstructed["month_continuity"],
    }
    for boundary in contract["boundary_rows"]:
        actual = by_holding.get(boundary["holding_month"])
        result["BOUNDARY_" + boundary["boundary_id"].upper()] = (
            None
            if actual is None
            else {
                "stage": actual["stage"],
                "signal_date": actual["signal_date"].isoformat(),
                "entry_date": actual["entry_date"].isoformat(),
                "exit_date": actual["exit_date"].isoformat(),
                "is_embargo": actual["is_embargo"],
                "is_partial": actual["is_partial"],
                "formal_inference_expected": actual["formal_inference_expected"],
            }
        )
    partial = by_holding.get("2025-12", {})
    result["PARTIAL_DECEMBER_EXCLUDED_FROM_FORMAL_INFERENCE"] = {
        "is_partial": partial.get("is_partial"),
        "formal_inference_expected": partial.get("formal_inference_expected"),
        "exit_date": _jsonable(partial.get("exit_date")),
    }
    return result


def _verify_data_outputs(
    settings: VerificationSettings,
    contract: Mapping[str, Any],
    frozen: Mapping[str, Any],
    primary: Mapping[str, Any],
    stat_before: Mapping[str, int],
    stat_after: Mapping[str, int],
    raw_sha256: str,
    scan: Mapping[str, Any],
    checks: _Checks,
) -> dict[str, Any]:
    outputs = primary["output_by_name"]
    reconstructed = independently_reconstruct_calendars(scan["dates"])
    trading_fields, trading_csv = _load_trading_csv(outputs["trading_calendar.csv"])
    rebalance_fields, rebalance_csv = _load_rebalance_csv(outputs["rebalance_calendar.csv"])
    trading_parquet_table = pq.read_table(outputs["trading_calendar.parquet"])
    rebalance_parquet_table = pq.read_table(outputs["rebalance_calendar.parquet"])
    trading_parquet = trading_parquet_table.to_pylist()
    rebalance_parquet = rebalance_parquet_table.to_pylist()
    expected_trading_schema = pa.schema(
        [
            ("calendar_date", pa.date32()),
            ("year", pa.int16()),
            ("month", pa.int8()),
            ("weekday_iso", pa.int8()),
            ("is_observed_month_end", pa.bool_()),
        ]
    )
    expected_rebalance_schema = pa.schema(
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
    checks.add(
        "CALENDAR_CSV_AND_PARQUET_SCHEMAS_EXACT",
        "output_schema",
        passed=(
            trading_fields == list(TRADING_FIELDS)
            and rebalance_fields == list(REBALANCE_FIELDS)
            and trading_parquet_table.schema == expected_trading_schema
            and rebalance_parquet_table.schema == expected_rebalance_schema
        ),
        expected={
            "trading_csv": list(TRADING_FIELDS),
            "rebalance_csv": list(REBALANCE_FIELDS),
            "trading_parquet": str(expected_trading_schema),
            "rebalance_parquet": str(expected_rebalance_schema),
        },
        actual={
            "trading_csv": trading_fields,
            "rebalance_csv": rebalance_fields,
            "trading_parquet": str(trading_parquet_table.schema),
            "rebalance_parquet": str(rebalance_parquet_table.schema),
        },
        evidence="four primary calendar data files",
        message="CSV column order and Arrow physical types are exact and contain no extra fields.",
    )
    _compare_rows(
        checks,
        "TRADING_CSV_EQUALS_INDEPENDENT_RAW_RECONSTRUCTION",
        "calendar_reconstruction",
        reconstructed["trading_rows"],
        trading_csv,
        outputs["trading_calendar.csv"],
        "Every trading-calendar field equals the independent raw-date reconstruction.",
    )
    _compare_rows(
        checks,
        "TRADING_PARQUET_EQUALS_INDEPENDENT_RAW_RECONSTRUCTION",
        "calendar_reconstruction",
        reconstructed["trading_rows"],
        trading_parquet,
        outputs["trading_calendar.parquet"],
        "Every Parquet trading-calendar field equals the independent raw-date reconstruction.",
    )
    _compare_rows(
        checks,
        "REBALANCE_CSV_311_ROWS_EQUAL_INDEPENDENT_RECONSTRUCTION",
        "calendar_reconstruction",
        reconstructed["rebalance_rows"],
        rebalance_csv,
        outputs["rebalance_calendar.csv"],
        "All 311 signal, entry, exit, stage, embargo, feature, partial, and inference fields agree.",
    )
    _compare_rows(
        checks,
        "REBALANCE_PARQUET_311_ROWS_EQUAL_INDEPENDENT_RECONSTRUCTION",
        "calendar_reconstruction",
        reconstructed["rebalance_rows"],
        rebalance_parquet,
        outputs["rebalance_calendar.parquet"],
        "All 311 Parquet rebalance rows agree with the independent reconstruction.",
    )
    _compare_rows(
        checks,
        "TRADING_CSV_PARQUET_EQUAL",
        "cross_format",
        trading_csv,
        trading_parquet,
        "trading_calendar.csv and trading_calendar.parquet",
        "CSV and Parquet trading-calendar values are exactly equal.",
    )
    _compare_rows(
        checks,
        "REBALANCE_CSV_PARQUET_EQUAL",
        "cross_format",
        rebalance_csv,
        rebalance_parquet,
        "rebalance_calendar.csv and rebalance_calendar.parquet",
        "CSV and Parquet rebalance-calendar values are exactly equal.",
    )

    by_holding = {row["holding_month"]: row for row in reconstructed["rebalance_rows"]}
    boundary_actual: list[dict[str, Any]] = []
    boundary_expected: list[dict[str, Any]] = []
    for boundary in contract["boundary_rows"]:
        expected = {
            "boundary_id": boundary["boundary_id"],
            "holding_month": boundary["holding_month"],
            "stage": boundary["stage"],
            "signal_date": boundary["signal_date"],
            "entry_date": boundary["entry_date"],
            "exit_date": boundary["exit_date"],
            "is_embargo": _parse_bool(boundary["is_embargo"]),
            "is_partial": _parse_bool(boundary["is_partial"]),
            "formal_inference_expected": _parse_bool(boundary["formal_inference_expected"]),
        }
        row = by_holding.get(boundary["holding_month"])
        actual = {
            "boundary_id": boundary["boundary_id"],
            "holding_month": boundary["holding_month"],
            "stage": None if row is None else row["stage"],
            "signal_date": None if row is None else row["signal_date"].isoformat(),
            "entry_date": None if row is None else row["entry_date"].isoformat(),
            "exit_date": None if row is None else row["exit_date"].isoformat(),
            "is_embargo": None if row is None else row["is_embargo"],
            "is_partial": None if row is None else row["is_partial"],
            "formal_inference_expected": (
                None if row is None else row["formal_inference_expected"]
            ),
        }
        boundary_expected.append(expected)
        boundary_actual.append(actual)
    checks.add(
        "EIGHT_LOCKED_BOUNDARIES_REPRODUCED_EXACTLY",
        "calendar_boundaries",
        passed=(
            [row["boundary_id"] for row in contract["boundary_rows"]]
            == list(EXPECTED_BOUNDARY_IDS)
            and boundary_actual == boundary_expected
        ),
        expected={"boundary_ids": list(EXPECTED_BOUNDARY_IDS), "rows": boundary_expected},
        actual={
            "boundary_ids": [row["boundary_id"] for row in contract["boundary_rows"]],
            "rows": boundary_actual,
        },
        evidence=contract["contract_dir"] / "calendar_boundary_expectations.csv",
        message="All eight locked calendar boundaries match the independently derived date calendar.",
    )

    check_fields, primary_checks = _read_csv(outputs["calendar_checks.csv"])
    expected_actuals = _expected_primary_actuals(
        contract, frozen, stat_before, stat_after, scan, reconstructed
    )
    primary_by_id = {row["check_id"]: row for row in primary_checks}
    expected_values: dict[str, Any] = {}
    actual_values: dict[str, Any] = {}
    parse_problems: dict[str, str] = {}
    for check_id, row in primary_by_id.items():
        for cell_name, destination in (("expected", expected_values), ("actual", actual_values)):
            cell = row[cell_name]
            if check_id == "VERIFIED_CONTRACT_GATE" and cell == "verified_passed":
                destination[check_id] = cell
                continue
            try:
                destination[check_id] = json.loads(cell)
            except Exception as exc:
                parse_problems[f"{check_id}:{cell_name}"] = f"{type(exc).__name__}: {exc}"
    expected_ids = sorted(expected_actuals)
    actual_ids = [row["check_id"] for row in primary_checks]
    all_primary_pass = (
        check_fields == list(PRIMARY_CHECK_FIELDS)
        and len(primary_checks) == len(primary_by_id) == 18
        and actual_ids == expected_ids
        and all(row["status"] == "PASS" for row in primary_checks)
        and not parse_problems
    )
    checks.add(
        "PRIMARY_18_CHECK_ROWS_EXACT_AND_ALL_PASS",
        "primary_quality_receipt",
        passed=all_primary_pass,
        expected={
            "fields": list(PRIMARY_CHECK_FIELDS),
            "count": 18,
            "ordered_ids": expected_ids,
            "statuses": ["PASS"],
            "parse_problems": {},
        },
        actual={
            "fields": check_fields,
            "count": len(primary_checks),
            "unique_count": len(primary_by_id),
            "ordered_ids": actual_ids,
            "statuses": sorted({row["status"] for row in primary_checks}),
            "parse_problems": parse_problems,
        },
        evidence=outputs["calendar_checks.csv"],
        message="The primary receipt contains exactly the expected 18 ordered, machine-readable PASS checks.",
    )
    checks.add(
        "PRIMARY_18_CHECK_ACTUALS_MATCH_INDEPENDENT_EVIDENCE",
        "primary_quality_receipt",
        passed=expected_values == actual_values == expected_actuals,
        expected={
            "check_count": len(expected_actuals),
            "claims_sha256": sha256_json(expected_actuals),
        },
        actual={
            "expected_cell_count": len(expected_values),
            "actual_cell_count": len(actual_values),
            "expected_cells_sha256": sha256_json(expected_values),
            "actual_cells_sha256": sha256_json(actual_values),
            "mismatching_ids": sorted(
                check_id
                for check_id in set(expected_actuals) | set(expected_values) | set(actual_values)
                if not (
                    expected_actuals.get(check_id)
                    == expected_values.get(check_id)
                    == actual_values.get(check_id)
                )
            ),
        },
        evidence=outputs["calendar_checks.csv"],
        message="Every primary check's expected and actual claims are reproduced from independent evidence.",
    )

    success = _read_json(outputs["_SUCCESS.json"])
    summary_file = _read_json(outputs["calendar_summary.json"])
    manifest_summary = primary["manifest"].get("summary", {})
    expected_summary = {
        "schema_version": 1,
        "build_id": primary["pointer"]["build_id"],
        "contract_id": contract["pointer"]["contract_id"],
        "verification_id": contract["pointer"]["verification_id"],
        "status": "calendar_primary_passed_pending_independent_verification",
        "raw_rows_scanned": scan["row_count"],
        "raw_columns_read": [settings.phase_config["calendar"]["calendar_date_column"]],
        "observed_trading_dates": len(reconstructed["trading_rows"]),
        "rebalance_intervals": len(reconstructed["rebalance_rows"]),
        "complete_rebalance_intervals": sum(
            not row["is_partial"] for row in reconstructed["rebalance_rows"]
        ),
        "partial_rebalance_intervals": sum(
            row["is_partial"] for row in reconstructed["rebalance_rows"]
        ),
        "minimum_date": scan["minimum"],
        "maximum_date": scan["maximum"],
        "check_count": 18,
        "pass_count": 18,
        "fail_count": 0,
        "targets_built": False,
        "features_built": False,
        "portfolios_built": False,
        "performance_results_seen": False,
    }
    checks.add(
        "SUCCESS_SUMMARY_AND_MANIFEST_SUMMARY_EXACT",
        "primary_quality_receipt",
        passed=success == summary_file == manifest_summary == expected_summary,
        expected=expected_summary,
        actual={
            "success_sha256": sha256_json(success),
            "quality_summary_sha256": sha256_json(summary_file),
            "manifest_summary_sha256": sha256_json(manifest_summary),
            "expected_sha256": sha256_json(expected_summary),
        },
        evidence="_SUCCESS.json, calendar_summary.json, and manifest.json",
        message="All three primary summaries are identical and independently reproduced.",
    )

    receipt = _read_json(outputs["input_receipt.json"])
    source_row = frozen["source_row"]
    builder_stat_fields = ("device", "inode", "mode", "size_bytes", "mtime_ns")
    before_builder = {field: stat_before[field] for field in builder_stat_fields}
    after_builder = {field: stat_after[field] for field in builder_stat_fields}
    receipt_scan = receipt.get("scan", {})
    receipt_scan_expected = {
        "row_count": scan["row_count"],
        "null_count": scan["null_count"],
        "blank_count": scan["blank_count"],
        "distinct_nonblank_count": scan["distinct_nonblank_count"],
        "valid_distinct_date_count": scan["valid_distinct_date_count"],
        "invalid_distinct_values": scan["invalid_distinct_values"],
        "minimum": scan["minimum"],
        "maximum": scan["maximum"],
        "selected_columns": scan["requested_columns"],
    }
    receipt_scan_actual = {key: receipt_scan.get(key) for key in receipt_scan_expected}
    receipt_ok = (
        receipt.get("binding") == primary["manifest"].get("binding")
        and receipt.get("verified_pointer") == contract["pointer"]
        and receipt.get("raw_operational_identity_before") == before_builder
        and receipt.get("raw_operational_identity_after") == after_builder
        and receipt_scan_actual == receipt_scan_expected
        and isinstance(receipt_scan.get("elapsed_seconds"), (int, float))
        and receipt_scan.get("elapsed_seconds", -1) >= 0
        and receipt.get("frozen_raw_identity_source", {}).get("sha256")
        == frozen["recheck_sha256"]
    )
    checks.add(
        "PRIMARY_INPUT_RECEIPT_MATCHES_INDEPENDENT_SCAN_AND_IDENTITIES",
        "primary_quality_receipt",
        passed=receipt_ok,
        expected={
            "binding_sha256": sha256_json(primary["manifest"].get("binding", {})),
            "verified_pointer_sha256": sha256_json(contract["pointer"]),
            "raw_before": before_builder,
            "raw_after": after_builder,
            "scan": receipt_scan_expected,
            "frozen_raw_identity_sha256": frozen["recheck_sha256"],
        },
        actual={
            "binding_sha256": sha256_json(receipt.get("binding", {})),
            "verified_pointer_sha256": sha256_json(receipt.get("verified_pointer", {})),
            "raw_before": receipt.get("raw_operational_identity_before"),
            "raw_after": receipt.get("raw_operational_identity_after"),
            "scan": receipt_scan_actual,
            "elapsed_seconds_valid": isinstance(receipt_scan.get("elapsed_seconds"), (int, float))
            and receipt_scan.get("elapsed_seconds", -1) >= 0,
            "frozen_raw_identity_sha256": receipt.get("frozen_raw_identity_source", {}).get(
                "sha256"
            ),
        },
        evidence=outputs["input_receipt.json"],
        message="The primary input receipt agrees with the independent raw scan and frozen identities.",
    )

    binding = primary["manifest"].get("binding", {})
    binding_expected = {
        "contract_id": contract["pointer"]["contract_id"],
        "verification_id": contract["pointer"]["verification_id"],
        "contract_json_sha256": contract["pointer"]["contract_json_sha256"],
        "independent_check_sha256": contract["pointer"]["independent_check_sha256"],
        "phase_config_sha256": sha256_file(settings.phase_config_path),
        "paths_config_sha256": sha256_file(settings.paths_config_path),
        "raw_file": settings.raw_path.name,
        "raw_sha256_from_frozen_phase_1": source_row["sha256"],
        "raw_size_bytes": int(source_row["size_bytes"]),
        "date_column": settings.phase_config["calendar"]["calendar_date_column"],
    }
    binding_actual = {key: binding.get(key) for key in binding_expected}
    checks.add(
        "PRIMARY_BINDING_AUTHENTICATES_CONTRACT_CONFIG_AND_RAW",
        "primary_authentication",
        passed=binding_actual == binding_expected and raw_sha256 == source_row["sha256"],
        expected={**binding_expected, "current_raw_sha256": source_row["sha256"]},
        actual={**binding_actual, "current_raw_sha256": raw_sha256},
        evidence=primary["manifest_path"],
        message="The primary binding and independent full-byte hash identify the same frozen raw file.",
    )
    return {
        "reconstructed": reconstructed,
        "trading_csv": trading_csv,
        "rebalance_csv": rebalance_csv,
        "trading_parquet": trading_parquet,
        "rebalance_parquet": rebalance_parquet,
    }


def independently_verify_calendar(
    settings: VerificationSettings | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    checks = _Checks(settings.project_root)
    contract = _authenticate_contract(settings, checks)
    frozen = _authenticate_frozen_raw(settings, contract, checks)
    primary = _authenticate_primary(settings, contract, checks)
    if not settings.raw_path.is_file():
        raise CalendarVerificationError(f"Missing raw calendar owner file: {settings.raw_path}")
    stat_before = operational_stat(settings.raw_path)
    checks.add(
        "RAW_OPERATIONAL_IDENTITY_BEFORE_READ_EXACT",
        "raw_authentication",
        passed=stat_before == frozen["expected_stat"],
        expected=frozen["expected_stat"],
        actual=stat_before,
        evidence=frozen["recheck_path"],
        message="Current device, inode, mode, size, mtime, and ctime match the Phase 1 freeze.",
    )
    raw_sha256 = sha256_file(settings.raw_path)
    scan = scan_raw_date_projection(
        settings.raw_path,
        date_column=settings.phase_config["calendar"]["calendar_date_column"],
        block_size_bytes=settings.block_size_bytes,
    )
    stat_after = operational_stat(settings.raw_path)
    checks.add(
        "RAW_OPERATIONAL_IDENTITY_STABLE_THROUGH_HASH_AND_PROJECTION",
        "raw_authentication",
        passed=stat_after == stat_before == frozen["expected_stat"],
        expected=frozen["expected_stat"],
        actual={"before": stat_before, "after": stat_after},
        evidence=settings.raw_path,
        message="The raw file remained exactly stable across the full-byte hash and date projection.",
    )
    source_row = frozen["source_row"]
    raw_domain_ok = (
        raw_sha256 == source_row["sha256"]
        and scan["row_count"] == int(source_row["parsed_data_rows"])
        and scan["null_count"] == 0
        and scan["blank_count"] == 0
        and scan["invalid_distinct_values"] == []
        and scan["minimum"] == settings.phase_config["calendar"]["expected_observed_minimum"]
        and scan["maximum"] == settings.phase_config["calendar"]["expected_observed_maximum"]
    )
    checks.add(
        "FULL_RAW_HASH_ROW_COUNT_AND_DATE_DOMAIN_EXACT",
        "raw_projection",
        passed=raw_domain_ok,
        expected={
            "sha256": source_row["sha256"],
            "row_count": int(source_row["parsed_data_rows"]),
            "null_count": 0,
            "blank_count": 0,
            "invalid_distinct_values": [],
            "minimum": settings.phase_config["calendar"]["expected_observed_minimum"],
            "maximum": settings.phase_config["calendar"]["expected_observed_maximum"],
        },
        actual={
            "sha256": raw_sha256,
            "row_count": scan["row_count"],
            "null_count": scan["null_count"],
            "blank_count": scan["blank_count"],
            "invalid_distinct_values": scan["invalid_distinct_values"],
            "minimum": scan["minimum"],
            "maximum": scan["maximum"],
        },
        evidence=settings.raw_path,
        message="The independently-read raw bytes, row count, and strict date domain match the frozen source.",
    )
    checks.add(
        "ONLY_DLYCALDT_PROJECTED_FROM_RAW",
        "raw_projection",
        passed=(
            scan["requested_columns"] == ["DlyCalDt"]
            and scan["materialized_schema_names"] == [["DlyCalDt"]]
            and scan["non_date_columns_materialized"] == 0
        ),
        expected={
            "requested_columns": ["DlyCalDt"],
            "materialized_schema_names": [["DlyCalDt"]],
            "non_date_columns_materialized": 0,
        },
        actual={
            "requested_columns": scan["requested_columns"],
            "materialized_schema_names": scan["materialized_schema_names"],
            "non_date_columns_materialized": scan["non_date_columns_materialized"],
        },
        evidence="independent pyarrow include_columns projection receipt",
        message="No raw price, return, identifier, feature, target, or outcome column was materialized.",
    )
    data = _verify_data_outputs(
        settings,
        contract,
        frozen,
        primary,
        stat_before,
        stat_after,
        raw_sha256,
        scan,
        checks,
    )

    sorted_checks = sorted(checks.rows, key=lambda row: row["check_id"])
    failures = [row for row in sorted_checks if row["status"] == "FAIL"]
    core = {
        "schema_version": 1,
        "build_id": primary["pointer"]["build_id"],
        "contract_id": contract["pointer"]["contract_id"],
        "primary_manifest_sha256": primary["manifest_sha256"],
        "verifier_implementation_sha256": sha256_file(Path(__file__).resolve()),
        "status": "verified_passed" if not failures else "verification_failed",
        "summary": {
            "check_count": len(sorted_checks),
            "pass_count": len(sorted_checks) - len(failures),
            "fail_count": len(failures),
            "difference_count": len(failures),
            "raw_rows_read": scan["row_count"],
            "raw_distinct_dates": len(scan["dates"]),
            "trading_rows_verified": len(data["trading_csv"]),
            "rebalance_rows_verified": len(data["rebalance_csv"]),
            "locked_boundaries_verified": len(contract["boundary_rows"]),
            "primary_checks_verified": 18,
        },
        "independence": {
            "primary_builder_module_imported": False,
            "raw_projection_engine": scan["engine"],
            "raw_projection_passes": 1,
            "raw_byte_hash_passes": 1,
            "raw_csv_rows_read": scan["row_count"],
            "raw_columns_requested": scan["requested_columns"],
            "raw_materialized_schema_names": scan["materialized_schema_names"],
            "non_date_raw_columns_materialized": scan["non_date_columns_materialized"],
            "calendar_reconstruction": "standalone verifier implementation",
        },
        "raw_identity": {
            "path": settings.raw_path.as_posix(),
            "sha256": raw_sha256,
            "stat_before": stat_before,
            "stat_after": stat_after,
        },
        "calendar_fingerprints": {
            "trading_rows": _row_signature(data["trading_csv"]),
            "rebalance_rows": _row_signature(data["rebalance_csv"]),
        },
        "checks": sorted_checks,
    }
    verification_id = "verify-" + sha256_json(core)[:16]
    return {**core, "verification_id": verification_id}


def publish_independent_verification(
    report: Mapping[str, Any],
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    artifact_root = artifact_root.resolve()
    project_root = project_root.resolve()
    build_id = str(report["build_id"])
    verification_id = str(report["verification_id"])
    final = artifact_root / "calendar_verifications" / build_id / verification_id
    differences = [row for row in report["checks"] if row["status"] == "FAIL"]
    stage_parent = artifact_root / "_staging"
    stage_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{verification_id}-", dir=stage_parent))
    try:
        _write_json(staging / "independent_check.json", report)
        _write_csv(staging / "differences.csv", DIFFERENCE_FIELDS, differences)
        expected_hashes = {
            path.name: sha256_file(path) for path in sorted(staging.iterdir()) if path.is_file()
        }
        if final.exists():
            actual_hashes = {
                path.name: sha256_file(path) for path in sorted(final.iterdir()) if path.is_file()
            }
            if actual_hashes != expected_hashes:
                raise CalendarVerificationError(
                    f"Existing content-addressed verification differs: {final}"
                )
        else:
            final.parent.mkdir(parents=True, exist_ok=True)
            os.rename(staging, final)
        if report["status"] == "verified_passed":
            pointer = {
                "schema_version": 1,
                "build_id": build_id,
                "contract_id": report["contract_id"],
                "primary_artifact_path": f"artifacts/phase_02_canonical_panel/calendar_builds/{build_id}",
                "primary_manifest_sha256": report["primary_manifest_sha256"],
                "verification_id": verification_id,
                "verification_path": _project_label(final, project_root),
                "independent_check_sha256": sha256_file(final / "independent_check.json"),
                "differences_sha256": sha256_file(final / "differences.csv"),
                "status": "verified_passed",
                "raw_csv_rows_read": report["summary"]["raw_rows_read"],
                "raw_columns_read": report["independence"]["raw_columns_requested"],
            }
            pointer_tmp = artifact_root / f".LATEST_CALENDAR_VERIFIED.{os.getpid()}.tmp"
            _write_json(pointer_tmp, pointer)
            os.replace(pointer_tmp, artifact_root / "LATEST_CALENDAR_VERIFIED.json")
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-config", type=Path, default=DEFAULT_PHASE_CONFIG)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--contract-pointer", type=Path, default=DEFAULT_CONTRACT_POINTER)
    parser.add_argument("--primary-pointer", type=Path, default=DEFAULT_PRIMARY_POINTER)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--expected-build-id", default=EXPECTED_CORRECTED_BUILD_ID)
    args = parser.parse_args(argv)
    settings = load_settings(
        args.phase_config,
        args.paths_config,
        contract_pointer_path=args.contract_pointer,
        primary_pointer_path=args.primary_pointer,
        artifact_root=args.artifact_root,
        expected_build_id=args.expected_build_id,
    )
    report = independently_verify_calendar(settings)
    destination = publish_independent_verification(
        report,
        artifact_root=settings.artifact_root,
        project_root=settings.project_root,
    )
    print(
        json.dumps(
            {
                "build_id": report["build_id"],
                "verification_id": report["verification_id"],
                "status": report["status"],
                "checks": report["summary"]["check_count"],
                "failures": report["summary"]["fail_count"],
                "raw_rows_read": report["summary"]["raw_rows_read"],
                "raw_columns_read": report["independence"]["raw_columns_requested"],
                "verification_path": destination.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "verified_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
