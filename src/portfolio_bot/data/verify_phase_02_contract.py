"""Independent verifier for the published Phase 2 CRSP contract.

This module intentionally does not import the Phase 2 contract compiler.  It
reconstructs the published contract from the two TOML contracts and the
authenticated Phase 1 evidence.  It never opens a raw CSV file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASE_CONFIG = PROJECT_ROOT / "configs" / "phase_02_canonical_panel.toml"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel"
COMPILER_PATH = PROJECT_ROOT / "src" / "portfolio_bot" / "data" / "phase_02_contract.py"
VERIFIER_PATH = Path(__file__).resolve()
BYTE_CHUNK_SIZE = 8 << 20

SEALED_ARTIFACT_NAMES = {
    "assertion_registry.csv",
    "calendar_boundary_expectations.csv",
    "contract.json",
    "input_hashes.sha256",
    "known_gaps.csv",
    "output_table_contracts.json",
    "run_metadata.json",
    "source_field_map.csv",
    "source_files.csv",
}

EXACT_CRSP_FILES = {
    "crsp_daily_ciz.csv",
    "crsp_monthly_ciz.csv",
    "crsp_names_ciz.csv",
    "crsp_stock_header_ciz.csv",
    "crsp_shares_outstanding_ciz.csv",
    "crsp_distribution_ciz.csv",
    "crsp_delisting_ciz.csv",
}

DIFFERENCE_FIELDS = (
    "check_id",
    "category",
    "expected",
    "actual",
    "evidence",
    "message",
)

CATEGORY_NAMES = (
    "identity_columns",
    "date_columns",
    "eligibility_columns",
    "classification_columns",
    "return_columns",
    "execution_columns",
    "reconciliation_columns",
    "prohibited_historical_feature_columns",
)


class IndependentVerificationError(RuntimeError):
    """Raised when independent verification cannot be safely published."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = BYTE_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def project_label(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _wire_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, set):
        return sorted(value)
    return value


class _Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        check_id: str,
        category: str,
        *,
        expected: Any,
        actual: Any,
        passed: bool,
        evidence: str,
        message: str,
    ) -> None:
        self.rows.append(
            {
                "check_id": check_id,
                "category": category,
                "status": "PASS" if passed else "FAIL",
                "expected": _jsonable(expected),
                "actual": _jsonable(actual),
                "evidence": evidence,
                "message": message,
            }
        )


def _parse_hash_lines(path: Path) -> tuple[dict[str, str], list[str]]:
    result: dict[str, str] = {}
    problems: list[str] = []
    if not path.is_file():
        return result, [f"missing file: {path.name}"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        digest, separator, label = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest) or not label:
            problems.append(f"malformed line {line_number}")
            continue
        if label in result:
            problems.append(f"duplicate label {label!r}")
            continue
        if path.name == "artifact_hashes.sha256" and Path(label).name != label:
            problems.append(f"non-basename artifact label {label!r}")
            continue
        result[label] = digest
    return result, problems


def _safe_json(path: Path, checks: _Checks, check_id: str) -> dict[str, Any]:
    try:
        value = _read_json(path)
    except Exception as exc:
        checks.add(
            check_id,
            "artifact_structure",
            expected="readable JSON object",
            actual=f"{type(exc).__name__}: {exc}",
            passed=False,
            evidence=project_label(path),
            message="The sealed JSON artifact must be independently parseable.",
        )
        return {}
    checks.add(
        check_id,
        "artifact_structure",
        expected="readable JSON object",
        actual="readable JSON object",
        passed=True,
        evidence=project_label(path),
        message="The sealed JSON artifact is independently parseable.",
    )
    return value


def _safe_csv(path: Path, checks: _Checks, check_id: str) -> list[dict[str, str]]:
    try:
        value = _read_csv(path)
    except Exception as exc:
        checks.add(
            check_id,
            "artifact_structure",
            expected="readable CSV",
            actual=f"{type(exc).__name__}: {exc}",
            passed=False,
            evidence=project_label(path),
            message="The sealed CSV artifact must be independently parseable.",
        )
        return []
    checks.add(
        check_id,
        "artifact_structure",
        expected="readable CSV",
        actual="readable CSV",
        passed=True,
        evidence=project_label(path),
        message="The sealed CSV artifact is independently parseable.",
    )
    return value


def _freeze_entries(freeze: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    groups = freeze.get("identity", {}).get("evidence_groups", {})
    if not isinstance(groups, dict):
        return entries
    for group in groups.values():
        if not isinstance(group, list):
            continue
        for item in group:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                entries[item["path"]] = dict(item)
    return entries


def _primary_evidence_labels(entries: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    names = {"files.csv", "columns.csv", "date_coverage.csv", "key_diagnostics.csv"}
    result: dict[str, str] = {}
    for label in entries:
        name = Path(label).name
        if "/builds/" in label and name in names:
            result[name] = label
    return result


def _expected_source_rows(
    source_specs: Sequence[Mapping[str, Any]],
    files_rows: Sequence[Mapping[str, str]],
    columns_rows: Sequence[Mapping[str, str]],
    date_rows: Sequence[Mapping[str, str]],
    key_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    files_by = {row["filename"]: row for row in files_rows}
    columns_by: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in columns_rows:
        columns_by[row["filename"]].append(row)
    for rows in columns_by.values():
        rows.sort(key=lambda row: int(row["position"]))
    date_by = {(row["filename"], row["column"]): row for row in date_rows}
    key_by = {row["filename"]: row for row in key_rows}
    result: list[dict[str, str]] = []
    for source in sorted(source_specs, key=lambda row: row["filename"]):
        filename = str(source["filename"])
        file_row = files_by.get(filename, {})
        observed = columns_by.get(filename, [])
        coverage = date_by.get((filename, str(source["coverage_axis"])), {})
        keys = key_by.get(filename, {})
        native = {
            "filename": filename,
            "source_family": file_row.get("family", ""),
            "role": source["role"],
            "grain": source["grain"],
            "candidate_key": source["candidate_key"],
            "coverage_axis": source["coverage_axis"],
            "availability_rule": source["availability_rule"],
            "calendar_owner": source["calendar_owner"],
            "size_bytes": int(file_row.get("size_bytes", 0)),
            "sha256": file_row.get("sha256", ""),
            "parsed_data_rows": int(file_row.get("parsed_data_rows", 0)),
            "observed_column_count": len(observed),
            "coverage_minimum": coverage.get("minimum", ""),
            "coverage_maximum": coverage.get("maximum", ""),
            "blank_candidate_key_rows": int(
                keys.get("rows_with_blank_diagnostic_key_fields", 0)
            ),
            "phase_01_adjacent_duplicate_occurrences": int(
                keys.get("adjacent_duplicate_occurrences_complete", 0)
            ),
            "global_uniqueness_status": "NOT_PROVED_PENDING_PHASE_2_SCAN",
        }
        result.append({key: _wire_value(value) for key, value in native.items()})
    return result


def _expected_field_rows(
    source_specs: Sequence[Mapping[str, Any]],
    vendor_fields: Sequence[Mapping[str, Any]],
    columns_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    columns_by: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in columns_rows:
        columns_by[row["filename"]].append(row)
    for rows in columns_by.values():
        rows.sort(key=lambda row: int(row["position"]))
    vendor_by = {row["field"]: row for row in vendor_fields}
    result: list[dict[str, Any]] = []
    for source in sorted(source_specs, key=lambda row: row["filename"]):
        filename = str(source["filename"])
        categories = {name: set(source.get(name, [])) for name in CATEGORY_NAMES}
        prohibited = categories["prohibited_historical_feature_columns"]
        stage_columns = set(source["stage_columns"])
        for observed in columns_by.get(filename, []):
            field = observed["name"]
            labels = sorted(
                name.removesuffix("_columns")
                for name, values in categories.items()
                if name != "prohibited_historical_feature_columns" and field in values
            )
            vendor = vendor_by.get(field, {})
            mapping_status = (
                "SELECTED_FOR_FIRST_SLICE" if field in stage_columns else "SOURCE_CONTEXT_ONLY"
            )
            blocker = ""
            if field in prohibited:
                mapping_status = "PROHIBITED_AS_HISTORICAL_PRIMARY"
                blocker = "SOURCE_AUTHORITY_UNIQUE"
            result.append(
                {
                    "source_file": filename,
                    "source_column": field,
                    "source_position": int(observed["position"]),
                    "physical_csv_type": observed["physical_csv_type"],
                    "sample_inferred_type": observed["sample_inferred_type"],
                    "field_categories": labels,
                    "canonical_unit": vendor.get("unit", "UNSPECIFIED_NOT_YET_TRANSFORMED"),
                    "vendor_definition": vendor.get("definition", ""),
                    "source_role": source["role"],
                    "observation_date_rule": source["coverage_axis"],
                    "availability_rule": source["availability_rule"],
                    "mapping_status": mapping_status,
                    "blocker_id": blocker,
                }
            )
    result.sort(key=lambda row: (row["source_file"], row["source_position"]))
    return [{key: _wire_value(value) for key, value in row.items()} for row in result]


def _expected_calendar_rows(phase: Mapping[str, Any]) -> list[dict[str, str]]:
    calendar = phase["calendar"]
    b = calendar["boundaries"]
    rows: list[dict[str, Any]] = [
        {
            "boundary_id": "last_development_eligible",
            "holding_month": "2014-11",
            "stage": "development",
            "signal_date": b["last_development_signal"],
            "entry_date": b["last_development_entry"],
            "exit_date": b["last_development_exit"],
            "is_embargo": False,
            "is_partial": False,
            "formal_inference_expected": False,
        },
        {
            "boundary_id": "development_validation_embargo",
            "holding_month": "2014-12",
            "stage": "embargo",
            "signal_date": b["development_validation_embargo_signal"],
            "entry_date": b["development_validation_embargo_entry"],
            "exit_date": b["development_validation_embargo_exit"],
            "is_embargo": True,
            "is_partial": False,
            "formal_inference_expected": False,
        },
        {
            "boundary_id": "first_validation",
            "holding_month": "2015-01",
            "stage": "validation",
            "signal_date": b["first_validation_signal"],
            "entry_date": b["first_validation_entry"],
            "exit_date": b["first_validation_exit"],
            "is_embargo": False,
            "is_partial": False,
            "formal_inference_expected": False,
        },
        {
            "boundary_id": "last_validation_eligible",
            "holding_month": "2019-11",
            "stage": "validation",
            "signal_date": b["last_validation_signal"],
            "entry_date": b["last_validation_entry"],
            "exit_date": b["last_validation_exit"],
            "is_embargo": False,
            "is_partial": False,
            "formal_inference_expected": False,
        },
        {
            "boundary_id": "validation_final_embargo",
            "holding_month": "2019-12",
            "stage": "embargo",
            "signal_date": b["embargo_signal"],
            "entry_date": b["embargo_entry"],
            "exit_date": b["embargo_exit"],
            "is_embargo": True,
            "is_partial": False,
            "formal_inference_expected": False,
        },
        {
            "boundary_id": "first_primary_final",
            "holding_month": "2020-01",
            "stage": "primary_final",
            "signal_date": b["first_final_signal"],
            "entry_date": b["first_final_entry"],
            "exit_date": b["first_final_exit"],
            "is_embargo": False,
            "is_partial": False,
            "formal_inference_expected": True,
        },
        {
            "boundary_id": "last_full_feature_primary",
            "holding_month": "2025-08",
            "stage": "primary_final",
            "signal_date": calendar["full_feature_last_signal_date"],
            "entry_date": calendar["full_feature_last_entry_date"],
            "exit_date": calendar["full_feature_last_exit_date"],
            "is_embargo": False,
            "is_partial": False,
            "formal_inference_expected": True,
        },
        {
            "boundary_id": "partial_december",
            "holding_month": "2025-12",
            "stage": "extension_descriptive",
            "signal_date": "2025-11-28",
            "entry_date": "2025-12-01",
            "exit_date": calendar["partial_december_close_date"],
            "is_embargo": False,
            "is_partial": True,
            "formal_inference_expected": False,
        },
    ]
    for row in rows:
        row["methodology_anchor"] = "Chronology and information boundaries"
    return [{key: _wire_value(value) for key, value in row.items()} for row in rows]


def _csv_bytes(rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _wire_value(row.get(name)) for name in fieldnames})
    return buffer.getvalue().encode("utf-8")


def independently_verify_contract(
    contract_directory: Path,
    *,
    phase_config_path: Path = DEFAULT_PHASE_CONFIG,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Reconstruct and verify a published contract without raw-row access."""

    contract_directory = contract_directory.resolve()
    project_root = project_root.resolve()
    phase_config_path = phase_config_path.resolve()
    checks = _Checks()

    with phase_config_path.open("rb") as handle:
        phase = tomllib.load(handle)
    source_paths = [project_path(label, project_root) for label in phase["source_contracts"]]
    source_contracts: list[dict[str, Any]] = []
    for path in source_paths:
        with path.open("rb") as handle:
            source_contracts.append(tomllib.load(handle))

    artifact_hashes, hash_line_problems = _parse_hash_lines(
        contract_directory / "artifact_hashes.sha256"
    )
    checks.add(
        "ARTIFACT_HASH_MANIFEST_FORMAT",
        "artifact_integrity",
        expected=[],
        actual=hash_line_problems,
        passed=not hash_line_problems,
        evidence=project_label(contract_directory / "artifact_hashes.sha256", project_root),
        message="The artifact hash manifest is strict, unique, and basename-only.",
    )
    checks.add(
        "ARTIFACT_HASH_REGISTRY_EXACT",
        "artifact_integrity",
        expected=sorted(SEALED_ARTIFACT_NAMES),
        actual=sorted(artifact_hashes),
        passed=set(artifact_hashes) == SEALED_ARTIFACT_NAMES,
        evidence=project_label(contract_directory / "artifact_hashes.sha256", project_root),
        message="Exactly the nine contract artifacts are sealed.",
    )
    actual_files = sorted(path.name for path in contract_directory.iterdir() if path.is_file())
    expected_files = sorted(SEALED_ARTIFACT_NAMES | {"artifact_hashes.sha256"})
    checks.add(
        "ARTIFACT_FILE_SET_EXACT",
        "artifact_integrity",
        expected=expected_files,
        actual=actual_files,
        passed=actual_files == expected_files,
        evidence=project_label(contract_directory, project_root),
        message="No sealed artifact is missing and no unsealed file is present.",
    )
    observed_artifact_hashes = {
        name: sha256_file(contract_directory / name)
        for name in artifact_hashes
        if (contract_directory / name).is_file()
    }
    checks.add(
        "ARTIFACT_CONTENT_HASHES_EXACT",
        "artifact_integrity",
        expected=artifact_hashes,
        actual=observed_artifact_hashes,
        passed=observed_artifact_hashes == artifact_hashes,
        evidence=project_label(contract_directory, project_root),
        message="Every published artifact matches its SHA-256 seal.",
    )

    contract = _safe_json(contract_directory / "contract.json", checks, "CONTRACT_JSON_READABLE")
    source_rows = _safe_csv(
        contract_directory / "source_files.csv", checks, "SOURCE_FILES_CSV_READABLE"
    )
    field_rows = _safe_csv(
        contract_directory / "source_field_map.csv", checks, "SOURCE_FIELD_MAP_CSV_READABLE"
    )
    assertion_rows = _safe_csv(
        contract_directory / "assertion_registry.csv", checks, "ASSERTION_REGISTRY_CSV_READABLE"
    )
    gap_rows = _safe_csv(
        contract_directory / "known_gaps.csv", checks, "KNOWN_GAPS_CSV_READABLE"
    )
    calendar_rows = _safe_csv(
        contract_directory / "calendar_boundary_expectations.csv",
        checks,
        "CALENDAR_BOUNDARIES_CSV_READABLE",
    )
    # The compiler publishes this file as a JSON array.
    try:
        output_value = json.loads(
            (contract_directory / "output_table_contracts.json").read_text(encoding="utf-8")
        )
        if not isinstance(output_value, list):
            raise ValueError("expected a JSON array")
    except Exception as exc:
        output_rows: list[dict[str, Any]] = []
        checks.add(
            "OUTPUT_CONTRACTS_ARRAY",
            "artifact_structure",
            expected="JSON array",
            actual=f"{type(exc).__name__}: {exc}",
            passed=False,
            evidence=project_label(contract_directory / "output_table_contracts.json", project_root),
            message="Output table contracts must be represented as an array.",
        )
    else:
        output_rows = output_value
        checks.add(
            "OUTPUT_CONTRACTS_ARRAY",
            "artifact_structure",
            expected="JSON array",
            actual="JSON array",
            passed=True,
            evidence=project_label(contract_directory / "output_table_contracts.json", project_root),
            message="Output table contracts are represented as an array.",
        )
    configured_paths = {
        "methodology": project_path(phase["methodology_freeze_manifest"], project_root),
        "freeze": project_path(phase["phase_01_freeze_manifest"], project_root),
        "attestation": project_path(phase["phase_01_recovery_attestation"], project_root),
    }
    configured_hashes = {
        "methodology": phase["methodology_freeze_manifest_sha256"],
        "freeze": phase["phase_01_freeze_manifest_sha256"],
        "attestation": phase["phase_01_recovery_attestation_sha256"],
    }
    upstream: dict[str, dict[str, Any]] = {}
    for name, path in configured_paths.items():
        actual_sha = sha256_file(path) if path.is_file() else None
        checks.add(
            f"UPSTREAM_{name.upper()}_HASH",
            "phase_01_provenance",
            expected=configured_hashes[name],
            actual=actual_sha,
            passed=actual_sha == configured_hashes[name],
            evidence=project_label(path, project_root),
            message="The configured upstream content binding matches the current sealed file.",
        )
        try:
            upstream[name] = _read_json(path)
        except Exception:
            upstream[name] = {}

    methodology = upstream["methodology"]
    freeze = upstream["freeze"]
    attestation = upstream["attestation"]
    freeze_identity_digest = sha256_bytes(canonical_json_bytes(freeze.get("identity", {})))
    checks.add(
        "PHASE_01_FREEZE_IDENTITY_DIGEST",
        "phase_01_provenance",
        expected=freeze.get("identity_sha256"),
        actual=freeze_identity_digest,
        passed=freeze_identity_digest == freeze.get("identity_sha256"),
        evidence=project_label(configured_paths["freeze"], project_root),
        message="The internal Phase 1 freeze identity is content-addressed.",
    )
    identifier_sets = {
        "methodology_freeze_id": [
            phase["methodology_freeze_id"],
            methodology.get("freeze_id"),
            freeze.get("identity", {}).get("methodology_freeze_id"),
        ],
        "raw_snapshot_id": [
            phase["raw_snapshot_id"],
            freeze.get("identity", {}).get("raw_snapshot_id"),
            attestation.get("raw_snapshot_id"),
        ],
        "phase_01_build_id": [
            phase["phase_01_build_id"],
            freeze.get("identity", {}).get("phase_01_build_id"),
            attestation.get("phase_01_build_id"),
        ],
        "verification_id": [
            phase["verification_id"],
            freeze.get("identity", {}).get("verification_id"),
            attestation.get("verification_id"),
        ],
        "phase_01_freeze_id": [
            phase["phase_01_freeze_id"],
            freeze.get("freeze_id"),
            attestation.get("freeze_id"),
        ],
    }
    for name, values in identifier_sets.items():
        checks.add(
            f"UPSTREAM_ID_{name.upper()}",
            "phase_01_provenance",
            expected=values[0],
            actual=values,
            passed=len(set(values)) == 1,
            evidence="Phase 2 config, Methodology v1 freeze, Phase 1 freeze, recovery attestation",
            message="All available upstream identifiers reconcile.",
        )
    handoff_state = {
        "freeze_status": freeze.get("status") == "phase_01_frozen",
        "attestation_status": attestation.get("status") == "passed",
        "content_verifier": attestation.get("content_aware_full_verifier", {}).get("status")
        == "passed",
        "phase_2_permitted": str(attestation.get("next_permitted_action", "")).startswith(
            "Begin Phase 2"
        ),
        "manifest_sealed": attestation.get("manifest", {}).get("sha256")
        == configured_hashes["freeze"],
    }
    checks.add(
        "PHASE_01_HANDOFF_STATUS",
        "phase_01_provenance",
        expected={key: True for key in handoff_state},
        actual=handoff_state,
        passed=all(handoff_state.values()),
        evidence=project_label(configured_paths["attestation"], project_root),
        message="Phase 1 explicitly permits contract-only Phase 2 work.",
    )

    frozen_entries = _freeze_entries(freeze)
    evidence_labels = _primary_evidence_labels(frozen_entries)
    expected_evidence_names = {"files.csv", "columns.csv", "date_coverage.csv", "key_diagnostics.csv"}
    checks.add(
        "PHASE_01_PRIMARY_EVIDENCE_SET",
        "phase_01_provenance",
        expected=sorted(expected_evidence_names),
        actual=sorted(evidence_labels),
        passed=set(evidence_labels) == expected_evidence_names,
        evidence=project_label(configured_paths["freeze"], project_root),
        message="The four required Phase 1 evidence tables are uniquely discoverable.",
    )
    evidence_paths = {
        name: project_path(label, project_root) for name, label in evidence_labels.items()
    }
    evidence_identity_actual: dict[str, Any] = {}
    evidence_identity_expected: dict[str, Any] = {}
    for name, label in sorted(evidence_labels.items()):
        path = evidence_paths[name]
        entry = frozen_entries[label]
        expected_identity = {
            "path": label,
            "size_bytes": entry.get("size_bytes"),
            "sha256": entry.get("sha256"),
        }
        actual_identity = {
            "path": label,
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        evidence_identity_expected[name] = expected_identity
        evidence_identity_actual[name] = actual_identity
    checks.add(
        "PHASE_01_PRIMARY_EVIDENCE_IDENTITIES",
        "phase_01_provenance",
        expected=evidence_identity_expected,
        actual=evidence_identity_actual,
        passed=evidence_identity_actual == evidence_identity_expected,
        evidence=project_label(configured_paths["freeze"], project_root),
        message="Every transferred Phase 1 evidence table matches its frozen path, size, and SHA-256.",
    )
    raw_contract_labels = [contract["phase_01_raw_contract"] for contract in source_contracts]
    raw_contract_state: dict[str, Any] = {}
    for label in raw_contract_labels:
        entry = frozen_entries.get(label)
        path = project_path(label, project_root)
        raw_contract_state[label] = {
            "frozen": entry is not None,
            "size_matches": entry is not None
            and path.is_file()
            and path.stat().st_size == entry.get("size_bytes"),
            "sha256_matches": entry is not None
            and path.is_file()
            and sha256_file(path) == entry.get("sha256"),
        }
    checks.add(
        "PHASE_01_RAW_CONTRACT_AUTHENTIC",
        "phase_01_provenance",
        expected={label: {"frozen": True, "size_matches": True, "sha256_matches": True} for label in raw_contract_labels},
        actual=raw_contract_state,
        passed=all(all(state.values()) for state in raw_contract_state.values()),
        evidence=project_label(configured_paths["freeze"], project_root),
        message="The source contract used for schema transfer remains the Phase 1 frozen content.",
    )

    input_hash_rows, input_hash_problems = _parse_hash_lines(
        contract_directory / "input_hashes.sha256"
    )
    checks.add(
        "INPUT_HASH_MANIFEST_FORMAT",
        "contract_binding",
        expected=[],
        actual=input_hash_problems,
        passed=not input_hash_problems,
        evidence=project_label(contract_directory / "input_hashes.sha256", project_root),
        message="Input content bindings are strict and unique.",
    )
    expected_input_hashes: dict[str, str] = {
        project_label(phase_config_path, project_root): sha256_file(phase_config_path),
        phase["methodology_freeze_manifest"]: phase["methodology_freeze_manifest_sha256"],
        phase["phase_01_freeze_manifest"]: phase["phase_01_freeze_manifest_sha256"],
        phase["phase_01_recovery_attestation"]: phase[
            "phase_01_recovery_attestation_sha256"
        ],
        project_label(COMPILER_PATH, project_root): sha256_file(COMPILER_PATH),
    }
    for path in source_paths:
        expected_input_hashes[project_label(path, project_root)] = sha256_file(path)
    for path in evidence_paths.values():
        expected_input_hashes[project_label(path, project_root)] = sha256_file(path)
    expected_input_hashes = dict(sorted(expected_input_hashes.items()))
    checks.add(
        "INPUT_HASH_BINDINGS_EXACT",
        "contract_binding",
        expected=expected_input_hashes,
        actual=input_hash_rows,
        passed=input_hash_rows == expected_input_hashes,
        evidence=project_label(contract_directory / "input_hashes.sha256", project_root),
        message="The published input manifest contains exactly the independently reconstructed bindings.",
    )
    bound_input_hashes = contract.get("binding", {}).get("input_hashes", {})
    checks.add(
        "CONTRACT_BOUND_INPUT_HASHES_EXACT",
        "contract_binding",
        expected=expected_input_hashes,
        actual=bound_input_hashes,
        passed=bound_input_hashes == expected_input_hashes,
        evidence=project_label(contract_directory / "contract.json", project_root),
        message="The JSON binding and standalone input hash manifest agree exactly.",
    )

    expected_binding = {
        "schema_version": 1,
        "phase": phase["phase"],
        "contract_version": phase["contract_version"],
        "methodology_freeze_id": phase["methodology_freeze_id"],
        "raw_snapshot_id": phase["raw_snapshot_id"],
        "phase_01_build_id": phase["phase_01_build_id"],
        "verification_id": phase["verification_id"],
        "phase_01_freeze_id": phase["phase_01_freeze_id"],
        "input_hashes": expected_input_hashes,
        "semantic_contract": phase,
        "source_contracts": source_contracts,
        "runtime_identity": {
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "tomllib": "stdlib",
            "serialization": "canonical_json_sorted_keys_compact_lf_and_csv_lf",
        },
    }
    actual_binding = contract.get("binding")
    checks.add(
        "CONTRACT_BINDING_RECONSTRUCTED_EXACTLY",
        "contract_binding",
        expected=sha256_bytes(canonical_json_bytes(expected_binding)),
        actual=sha256_bytes(canonical_json_bytes(actual_binding)) if isinstance(actual_binding, dict) else None,
        passed=actual_binding == expected_binding,
        evidence="Phase 2 TOMLs and authenticated Phase 1 evidence",
        message="An independent implementation reconstructs the complete deterministic binding.",
    )
    expected_contract_digest = sha256_bytes(canonical_json_bytes(expected_binding))
    expected_contract_id = f"phase02-crsp-{expected_contract_digest[:16]}"
    digest_state = {
        "contract_sha256": contract.get("contract_sha256"),
        "contract_id": contract.get("contract_id"),
        "directory_name": contract_directory.name,
    }
    expected_digest_state = {
        "contract_sha256": expected_contract_digest,
        "contract_id": expected_contract_id,
        "directory_name": expected_contract_id,
    }
    checks.add(
        "CONTRACT_ID_AND_DIGEST_RECOMPUTED",
        "contract_binding",
        expected=expected_digest_state,
        actual=digest_state,
        passed=digest_state == expected_digest_state,
        evidence=project_label(contract_directory / "contract.json", project_root),
        message="The contract identifier is the prefix of the independently recomputed binding digest.",
    )

    safe_scope = {
        "status": contract.get("status") == "contract_compiled_data_gates_pending",
        "raw_rows_read": contract.get("scope", {}).get("raw_rows_read") == 0,
        "targets_built": contract.get("scope", {}).get("targets_built") is False,
        "features_built": contract.get("scope", {}).get("features_built") is False,
        "portfolios_built": contract.get("scope", {}).get("portfolios_built") is False,
        "performance_results_seen": contract.get("scope", {}).get("performance_results_seen")
        is False,
    }
    checks.add(
        "CONTRACT_SCOPE_REMAINS_PRE_DATA",
        "scope",
        expected={key: True for key in safe_scope},
        actual=safe_scope,
        passed=all(safe_scope.values()),
        evidence=project_label(contract_directory / "contract.json", project_root),
        message="Contract compilation does not claim any raw scan, target, feature, portfolio, or performance result.",
    )

    source_specs = [source for doc in source_contracts for source in doc["sources"]]
    vendor_fields = [field for doc in source_contracts for field in doc["vendor_fields"]]
    source_filenames = [source["filename"] for source in source_specs]
    checks.add(
        "EXACT_SEVEN_CRSP_SOURCES",
        "source_mapping",
        expected=sorted(EXACT_CRSP_FILES),
        actual=sorted(source_filenames),
        passed=set(source_filenames) == EXACT_CRSP_FILES
        and len(source_filenames) == len(set(source_filenames)) == 7,
        evidence="configs/phase_02_sources/crsp_backbone.toml",
        message="The source contract has exactly the seven locked CRSP files.",
    )
    evidence_csv: dict[str, list[dict[str, str]]] = {}
    for name, path in evidence_paths.items():
        evidence_csv[name] = _read_csv(path) if path.is_file() else []
    expected_sources = _expected_source_rows(
        source_specs,
        evidence_csv.get("files.csv", []),
        evidence_csv.get("columns.csv", []),
        evidence_csv.get("date_coverage.csv", []),
        evidence_csv.get("key_diagnostics.csv", []),
    )
    checks.add(
        "SEVEN_SOURCE_ROWS_RECONSTRUCTED_EXACTLY",
        "source_mapping",
        expected={"count": 7, "sha256": sha256_bytes(canonical_json_bytes(expected_sources))},
        actual={"count": len(source_rows), "sha256": sha256_bytes(canonical_json_bytes(source_rows))},
        passed=source_rows == expected_sources and len(source_rows) == 7,
        evidence="Phase 2 source TOML plus frozen Phase 1 files/date/key evidence",
        message="All seven source inventory rows are independently reconstructed cell-for-cell.",
    )
    expected_fields = _expected_field_rows(
        source_specs, vendor_fields, evidence_csv.get("columns.csv", [])
    )
    configured_field_count = sum(len(source["expected_columns"]) for source in source_specs)
    checks.add(
        "ONE_HUNDRED_EIGHTY_SIX_FIELDS_RECONSTRUCTED_EXACTLY",
        "source_mapping",
        expected={
            "configured_count": 186,
            "row_count": 186,
            "sha256": sha256_bytes(canonical_json_bytes(expected_fields)),
        },
        actual={
            "configured_count": configured_field_count,
            "row_count": len(field_rows),
            "sha256": sha256_bytes(canonical_json_bytes(field_rows)),
        },
        passed=configured_field_count == 186 and len(field_rows) == 186 and field_rows == expected_fields,
        evidence="Phase 2 source TOML plus frozen Phase 1 columns.csv",
        message="The complete 186-field map is independently reconstructed cell-for-cell.",
    )
    header_state: dict[str, Any] = {}
    columns_by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in evidence_csv.get("columns.csv", []):
        columns_by[row["filename"]].append(row)
    for rows in columns_by.values():
        rows.sort(key=lambda row: int(row["position"]))
    for source in source_specs:
        observed = [row["name"] for row in columns_by.get(source["filename"], [])]
        header_state[source["filename"]] = observed == source["expected_columns"]
    checks.add(
        "SOURCE_HEADERS_EQUAL_FROZEN_PHASE_01",
        "source_mapping",
        expected={filename: True for filename in sorted(EXACT_CRSP_FILES)},
        actual=dict(sorted(header_state.items())),
        passed=set(header_state) == EXACT_CRSP_FILES and all(header_state.values()),
        evidence="configs/phase_02_sources/crsp_backbone.toml; frozen Phase 1 columns.csv",
        message="Every case-sensitive ordered header equals both the source contract and Phase 1 evidence.",
    )

    expected_outputs = sorted(
        [row for doc in source_contracts for row in doc["outputs"]], key=lambda row: row["table"]
    )
    checks.add(
        "OUTPUT_TABLE_CONTRACTS_EXACT",
        "source_mapping",
        expected=sha256_bytes(canonical_json_bytes(expected_outputs)),
        actual=sha256_bytes(canonical_json_bytes(output_rows)),
        passed=output_rows == expected_outputs,
        evidence="configs/phase_02_sources/crsp_backbone.toml",
        message="Published output table contracts exactly equal the configured source authorities.",
    )

    configured_gates = sorted(
        [gate for doc in source_contracts for gate in doc["gates"]], key=lambda row: row["gate_id"]
    )
    future_by = {
        row.get("check_id", "").removeprefix("FUTURE_GATE_"): row
        for row in assertion_rows
        if row.get("check_id", "").startswith("FUTURE_GATE_")
    }
    gate_state: dict[str, Any] = {}
    compile_gate_ids = {
        "CRSP_SCHEMA_EXACT",
        "PHASE_01_BOUNDARY_AUTHENTIC",
        "SOURCE_AUTHORITY_UNIQUE",
    }
    for gate in configured_gates:
        gate_id = gate["gate_id"]
        row = future_by.get(gate_id, {})
        expected_status = "PASS" if gate["stage"] == "contract_compile" else "NOT_RUN"
        expected_blocks = "none" if expected_status == "PASS" else gate["stage"]
        gate_state[gate_id] = {
            "present": bool(row),
            "status": row.get("status"),
            "scope": row.get("scope"),
            "blocks_action": row.get("blocks_action"),
            "matches": row.get("status") == expected_status
            and row.get("scope") == gate["stage"]
            and row.get("blocks_action") == expected_blocks,
        }
        if gate["stage"] == "contract_compile" and gate_id not in compile_gate_ids:
            gate_state[gate_id]["matches"] = False
    gates_pass = (
        set(future_by) == {gate["gate_id"] for gate in configured_gates}
        and all(state["matches"] for state in gate_state.values())
    )
    checks.add(
        "DATA_DEPENDENT_GATES_NOT_OVERCLAIMED",
        "pending_gates",
        expected={
            gate["gate_id"]: "PASS" if gate["stage"] == "contract_compile" else "NOT_RUN"
            for gate in configured_gates
        },
        actual={gate_id: state["status"] for gate_id, state in gate_state.items()},
        passed=gates_pass,
        evidence=project_label(contract_directory / "assertion_registry.csv", project_root),
        message="Only compile-time gates pass; every data-dependent gate remains NOT_RUN and blocking.",
    )
    deterministic_assertions = [
        row for row in assertion_rows if row.get("check_kind") == "deterministic_contract_assertion"
    ]
    assertion_state = {
        "deterministic_all_pass": all(row.get("status") == "PASS" for row in deterministic_assertions),
        "no_fail_rows": all(row.get("status") != "FAIL" for row in assertion_rows),
    }
    checks.add(
        "COMPILE_ASSERTION_STATUS_COHERENT",
        "pending_gates",
        expected={"deterministic_all_pass": True, "no_fail_rows": True},
        actual=assertion_state,
        passed=all(assertion_state.values()),
        evidence=project_label(contract_directory / "assertion_registry.csv", project_root),
        message="Compile assertions passed while data-dependent gates were left pending.",
    )

    expected_absent = {
        (source["filename"], field)
        for source in source_specs
        for field in source.get("expected_vendor_fields_absent_from_extract", [])
    }
    observed_absent = {
        (row.get("source_file", ""), row.get("missing_or_ambiguous_item", ""))
        for row in gap_rows
        if row.get("gap_id", "").startswith("VENDOR_FIELD_ABSENT_")
        and row.get("status") == "DISCLOSED_REQUIRES_RECONCILIATION"
    }
    actually_absent = {
        (filename, field)
        for filename, field in expected_absent
        if field not in {row["name"] for row in columns_by.get(filename, [])}
    }
    checks.add(
        "MISSING_VENDOR_FIELDS_DISCLOSED_EXACTLY",
        "known_gaps",
        expected=sorted(expected_absent),
        actual=sorted(observed_absent),
        passed=observed_absent == expected_absent == actually_absent,
        evidence="source contract, known_gaps.csv, frozen Phase 1 columns.csv",
        message="All eight documented-but-absent vendor flags are absent and explicitly disclosed.",
    )
    return_blocked = {
        row.get("missing_or_ambiguous_item")
        for row in gap_rows
        if row.get("blocked_action") == "canonical returns and labels"
        and row.get("status") == "DISCLOSED_REQUIRES_RECONCILIATION"
    }
    expected_return_flags = {"DlyDelFlg", "DlyDistRetFlg", "MthDelFlg"}
    checks.add(
        "RETURN_AND_DELISTING_GAPS_BLOCK_LABELS",
        "known_gaps",
        expected=sorted(expected_return_flags),
        actual=sorted(return_blocked),
        passed=return_blocked == expected_return_flags
        and phase["returns"]["exactly_once_status_before_reconciliation"] == "unproved",
        evidence="known_gaps.csv; configs/phase_02_canonical_panel.toml",
        message="Absent return/delisting flags block labels until exactly-once reconciliation passes.",
    )

    expected_calendar = _expected_calendar_rows(phase)
    checks.add(
        "CALENDAR_BOUNDARIES_RECONSTRUCTED_EXACTLY",
        "calendar",
        expected=sha256_bytes(canonical_json_bytes(expected_calendar)),
        actual=sha256_bytes(canonical_json_bytes(calendar_rows)),
        passed=calendar_rows == expected_calendar,
        evidence="configs/phase_02_canonical_panel.toml; calendar_boundary_expectations.csv",
        message="All eight boundary rows, including both embargoes, are independently reconstructed.",
    )
    embargo_rows = [row for row in calendar_rows if row.get("is_embargo") == "true"]
    embargo_state = {
        row.get("boundary_id", ""): (
            row.get("signal_date"),
            row.get("entry_date"),
            row.get("exit_date"),
        )
        for row in embargo_rows
    }
    expected_embargo_state = {
        "development_validation_embargo": ("2014-11-28", "2014-12-01", "2015-01-02"),
        "validation_final_embargo": ("2019-11-29", "2019-12-02", "2020-01-02"),
    }
    checks.add(
        "BOTH_EMBARGOES_EXACT",
        "calendar",
        expected=expected_embargo_state,
        actual=embargo_state,
        passed=embargo_state == expected_embargo_state,
        evidence=project_label(contract_directory / "calendar_boundary_expectations.csv", project_root),
        message="Both stage-boundary embargo holding intervals are explicit and exact.",
    )

    vendor_by = {row["field"]: row for row in vendor_fields}
    field_by = {(row.get("source_file"), row.get("source_column")): row for row in field_rows}
    units = phase["units"]
    semantic_state = {
        "screening_price": units.get("screening_price_column") == "DlyClose",
        "execution_price": units.get("execution_price_column") == "DlyClose",
        "dollar_volume_formula": units.get("dollar_volume_formula")
        == "DlyClose * DlyVol on valid liquidity sessions",
        "DlyClose_vendor_definition": vendor_by.get("DlyClose", {}).get("definition")
        == "last regular-session trade only",
        "DlyClose_unit": field_by.get(("crsp_daily_ciz.csv", "DlyClose"), {}).get(
            "canonical_unit"
        )
        == "usd_per_share",
        "DlyVol_vendor_definition": vendor_by.get("DlyVol", {}).get("definition")
        == "raw shares traded during the day, not split-adjusted within month",
        "DlyVol_unit": field_by.get(("crsp_daily_ciz.csv", "DlyVol"), {}).get(
            "canonical_unit"
        )
        == "shares",
        "DlyPrc_is_diagnostic": units.get("diagnostic_price_column") == "DlyPrc",
        "missing_close_rule_pending": units.get("missing_scheduled_execution_close_rule")
        == "unresolved_hard_gate_before_backtest_execution",
    }
    checks.add(
        "DLYCLOSE_DLYVOL_SEMANTICS_EXACT",
        "field_semantics",
        expected={key: True for key in semantic_state},
        actual=semantic_state,
        passed=all(semantic_state.values()),
        evidence="Phase 2 TOMLs and source_field_map.csv",
        message="Screening/execution and liquidity use the locked trade-close and raw-share semantics.",
    )

    summary = contract.get("compile_summary", {})
    expected_summary_state = {
        "source_file_count": len(source_rows),
        "source_field_count": len(field_rows),
        "output_table_contract_count": len(output_rows),
        "compile_assertion_pass_count": sum(row.get("status") == "PASS" for row in assertion_rows),
        "compile_assertion_fail_count": sum(row.get("status") == "FAIL" for row in assertion_rows),
        "future_gate_not_run_count": sum(row.get("status") == "NOT_RUN" for row in assertion_rows),
        "known_gap_count": len(gap_rows),
    }
    checks.add(
        "COMPILE_SUMMARY_RECONCILES",
        "artifact_structure",
        expected=expected_summary_state,
        actual=summary,
        passed=summary == expected_summary_state,
        evidence="contract.json and all sealed tabular artifacts",
        message="Published counts reconcile to the sealed artifacts.",
    )

    checks.rows.sort(key=lambda row: row["check_id"])
    fail_count = sum(row["status"] == "FAIL" for row in checks.rows)
    checks_digest = sha256_bytes(canonical_json_bytes(checks.rows))
    artifact_hash_manifest_sha = (
        sha256_file(contract_directory / "artifact_hashes.sha256")
        if (contract_directory / "artifact_hashes.sha256").is_file()
        else None
    )
    contract_json_sha = (
        sha256_file(contract_directory / "contract.json")
        if (contract_directory / "contract.json").is_file()
        else None
    )
    verification_binding = {
        "schema_version": 1,
        "verifier": "independent_phase_02_crsp_contract_verifier",
        "verifier_sha256": sha256_file(VERIFIER_PATH),
        "contract_id": contract.get("contract_id", contract_directory.name),
        "contract_json_sha256": contract_json_sha,
        "artifact_hashes_sha256": artifact_hash_manifest_sha,
        "phase_config_sha256": sha256_file(phase_config_path),
        "source_contract_sha256": {
            project_label(path, project_root): sha256_file(path) for path in source_paths
        },
        "phase_01_freeze_manifest_sha256": configured_hashes["freeze"],
        "phase_01_recovery_attestation_sha256": configured_hashes["attestation"],
        "checks_sha256": checks_digest,
        "raw_rows_read": 0,
    }
    verification_digest = sha256_bytes(canonical_json_bytes(verification_binding))
    verification_id = f"verify-{verification_digest[:16]}"
    status = "verified_passed" if fail_count == 0 else "verification_failed"
    report = {
        "schema_version": 1,
        "verification_id": verification_id,
        "verification_sha256": verification_digest,
        "contract_id": contract.get("contract_id", contract_directory.name),
        "status": status,
        "independence": {
            "compiler_module_imported": False,
            "raw_csv_rows_read": 0,
            "reconstruction_sources": [
                "Phase 2 semantic TOML",
                "Phase 2 CRSP source TOML",
                "authenticated Phase 1 evidence",
            ],
        },
        "summary": {
            "check_count": len(checks.rows),
            "pass_count": len(checks.rows) - fail_count,
            "fail_count": fail_count,
            "source_file_count": len(source_rows),
            "source_field_count": len(field_rows),
            "difference_count": fail_count,
        },
        "binding": verification_binding,
        "checks": checks.rows,
    }
    return report


def _verification_files(report: Mapping[str, Any]) -> dict[str, bytes]:
    differences = [row for row in report["checks"] if row["status"] == "FAIL"]
    return {
        "differences.csv": _csv_bytes(differences, DIFFERENCE_FIELDS),
        "independent_check.json": canonical_json_bytes(report),
    }


def publish_independent_verification(
    report: Mapping[str, Any],
    contract_directory: Path,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Publish a content-addressed verification and update the pass pointer."""

    artifact_root = artifact_root.resolve()
    project_root = project_root.resolve()
    contract_directory = contract_directory.resolve()
    contract_id = str(report["contract_id"])
    verification_id = str(report["verification_id"])
    if contract_id != contract_directory.name:
        raise IndependentVerificationError("contract id does not match the contract directory")
    files = _verification_files(report)
    root = artifact_root / "verifications" / contract_id
    final_directory = root / verification_id
    staging = artifact_root / "_verification_staging"
    root.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    if final_directory.exists():
        observed = {path.name: path.read_bytes() for path in final_directory.iterdir() if path.is_file()}
        if observed != files:
            raise IndependentVerificationError(
                "existing content-addressed verification does not match reconstructed bytes"
            )
    else:
        stage_parent = Path(tempfile.mkdtemp(prefix="verify-", dir=staging))
        stage_directory = stage_parent / verification_id
        try:
            stage_directory.mkdir()
            for name, value in files.items():
                (stage_directory / name).write_bytes(value)
            os.rename(stage_directory, final_directory)
        finally:
            if stage_parent.exists():
                shutil.rmtree(stage_parent)

    if report["status"] == "verified_passed":
        pointer = {
            "schema_version": 1,
            "contract_id": contract_id,
            "contract_path": project_label(contract_directory, project_root),
            "verification_id": verification_id,
            "verification_path": project_label(final_directory, project_root),
            "contract_json_sha256": sha256_file(contract_directory / "contract.json"),
            "artifact_hashes_sha256": sha256_file(
                contract_directory / "artifact_hashes.sha256"
            ),
            "independent_check_sha256": sha256_file(final_directory / "independent_check.json"),
            "differences_sha256": sha256_file(final_directory / "differences.csv"),
            "status": "verified_passed",
        }
        artifact_root.mkdir(parents=True, exist_ok=True)
        temporary = artifact_root / f".LATEST_CONTRACT_VERIFIED.{os.getpid()}.tmp"
        temporary.write_bytes(canonical_json_bytes(pointer))
        os.replace(temporary, artifact_root / "LATEST_CONTRACT_VERIFIED.json")
    return final_directory


def verify_and_publish(
    contract_directory: Path,
    *,
    phase_config_path: Path = DEFAULT_PHASE_CONFIG,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], Path]:
    report = independently_verify_contract(
        contract_directory, phase_config_path=phase_config_path, project_root=project_root
    )
    directory = publish_independent_verification(
        report,
        contract_directory,
        artifact_root=artifact_root,
        project_root=project_root,
    )
    return report, directory


def _resolve_primary_contract(artifact_root: Path, project_root: Path) -> Path:
    pointer_path = artifact_root / "LATEST_CONTRACT_PRIMARY.json"
    pointer = _read_json(pointer_path)
    return project_path(pointer["contract_path"], project_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PHASE_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--contract-directory", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    artifact_root = args.artifact_root.resolve()
    contract_directory = (
        args.contract_directory.resolve()
        if args.contract_directory
        else _resolve_primary_contract(artifact_root, PROJECT_ROOT)
    )
    report = independently_verify_contract(contract_directory, phase_config_path=args.config)
    output_path: str | None = None
    if not args.check_only:
        directory = publish_independent_verification(
            report, contract_directory, artifact_root=artifact_root
        )
        output_path = project_label(directory)
    print(
        json.dumps(
            {
                "contract_id": report["contract_id"],
                "verification_id": report["verification_id"],
                "status": report["status"],
                "check_count": report["summary"]["check_count"],
                "fail_count": report["summary"]["fail_count"],
                "raw_rows_read": 0,
                "path": output_path,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "verified_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
