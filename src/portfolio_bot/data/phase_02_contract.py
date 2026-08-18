"""Compile the Phase 2 CRSP backbone contract without reading raw data rows.

The compiler transfers only authenticated Phase 1 evidence into a deterministic,
content-addressed contract.  It deliberately leaves every data-dependent claim
pending until a later scanner has produced auditable receipts.
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
import sys
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASE_CONFIG = PROJECT_ROOT / "configs" / "phase_02_canonical_panel.toml"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel"
RUN_ROOT = PROJECT_ROOT / "runs" / "phase_02_canonical_panel"
BYTE_CHUNK_SIZE = 8 << 20


class ContractValidationError(RuntimeError):
    """Raised when a compile-time Phase 2 contract assertion fails."""


@dataclasses.dataclass(frozen=True)
class Phase2ContractSettings:
    phase_config_path: Path
    phase_config: dict[str, Any]
    source_contract_paths: tuple[Path, ...]
    source_contracts: tuple[dict[str, Any], ...]
    artifact_root: Path
    run_root: Path


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
        raise ContractValidationError(f"Expected a JSON object: {project_label(path)}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def load_settings(
    phase_config_path: Path = DEFAULT_PHASE_CONFIG,
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    run_root: Path = RUN_ROOT,
) -> Phase2ContractSettings:
    phase_config_path = phase_config_path.resolve()
    with phase_config_path.open("rb") as handle:
        phase = tomllib.load(handle)
    source_paths = tuple(project_path(item) for item in phase.get("source_contracts", []))
    source_contracts: list[dict[str, Any]] = []
    for path in source_paths:
        with path.open("rb") as handle:
            source_contracts.append(tomllib.load(handle))
    return Phase2ContractSettings(
        phase_config_path=phase_config_path,
        phase_config=phase,
        source_contract_paths=source_paths,
        source_contracts=tuple(source_contracts),
        artifact_root=artifact_root.resolve(),
        run_root=run_root.resolve(),
    )


def _assertion(
    rows: list[dict[str, Any]],
    check_id: str,
    *,
    gate: str,
    expression: str,
    expected: Any,
    actual: Any,
    passed: bool,
    evidence_source: str,
    message: str,
    blocks_action: str = "contract_publication",
    severity: str = "hard",
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "gate": gate,
            "scope": "contract_compile",
            "severity": severity,
            "check_kind": "deterministic_contract_assertion",
            "expression": expression,
            "expected": expected,
            "actual": actual,
            "evidence_source": evidence_source,
            "status": "PASS" if passed else "FAIL",
            "blocks_action": blocks_action,
            "message": message,
        }
    )


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    label: str,
    assertions: list[dict[str, Any]],
) -> None:
    observed = set(value)
    unknown = sorted(observed - allowed)
    missing = sorted(required - observed)
    _assertion(
        assertions,
        f"SCHEMA_KEYS_{re.sub(r'[^A-Z0-9]+', '_', label.upper()).strip('_')}",
        gate="input_schema",
        expression=f"keys({label}) are recognized and required keys exist",
        expected={"unknown": [], "missing": []},
        actual={"unknown": unknown, "missing": missing},
        passed=not unknown and not missing,
        evidence_source=label,
        message="Strict configuration schema validation.",
    )


def _validate_configuration_schema(
    phase: dict[str, Any], source_contracts: tuple[dict[str, Any], ...]
) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = []
    phase_allowed = {
        "schema_version",
        "phase",
        "contract_version",
        "methodology_freeze_id",
        "methodology_freeze_manifest",
        "methodology_freeze_manifest_sha256",
        "raw_snapshot_id",
        "phase_01_build_id",
        "verification_id",
        "phase_01_freeze_id",
        "phase_01_freeze_manifest",
        "phase_01_freeze_manifest_sha256",
        "phase_01_recovery_attestation",
        "phase_01_recovery_attestation_sha256",
        "source_contracts",
        "first_slice",
        "storage",
        "calendar",
        "universe",
        "availability",
        "returns",
        "units",
        "quality",
        "deferred_contracts",
        "sealed_periods",
    }
    phase_required = phase_allowed
    _require_exact_keys(
        phase,
        allowed=phase_allowed,
        required=phase_required,
        label="configs/phase_02_canonical_panel.toml",
        assertions=assertions,
    )
    source_allowed = {
        "schema_version",
        "contract_name",
        "contract_version",
        "source_family",
        "authoritative_for",
        "official_documentation",
        "official_documentation_title",
        "official_documentation_last_modified",
        "phase_01_raw_contract",
        "scope",
        "outputs",
        "sources",
        "vendor_fields",
        "gates",
    }
    source_required = source_allowed
    for source in source_contracts:
        _require_exact_keys(
            source,
            allowed=source_allowed,
            required=source_required,
            label=f"source contract {source.get('contract_name', '<unnamed>')}",
            assertions=assertions,
        )
    return assertions


def _frozen_entry_map(freeze: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    groups = freeze.get("identity", {}).get("evidence_groups", {})
    for entries in groups.values():
        for entry in entries:
            result[entry["path"]] = dict(entry)
    return result


def _verify_frozen_file(
    label: str,
    frozen_entries: Mapping[str, Mapping[str, Any]],
    assertions: list[dict[str, Any]],
) -> Path:
    path = project_path(label)
    entry = frozen_entries.get(label)
    actual_sha = sha256_file(path) if path.is_file() else None
    actual_size = path.stat().st_size if path.is_file() else None
    expected = None if entry is None else {"sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}
    actual = {"sha256": actual_sha, "size_bytes": actual_size}
    passed = entry is not None and actual == expected
    _assertion(
        assertions,
        "FROZEN_EVIDENCE_" + re.sub(r"[^A-Z0-9]+", "_", label.upper()).strip("_"),
        gate="input_provenance",
        expression="current path size and SHA-256 equal Phase 1 frozen evidence entry",
        expected=expected,
        actual=actual,
        passed=passed,
        evidence_source=label,
        message="Authenticated transfer of Phase 1 evidence; no raw row scan.",
    )
    return path


def _load_authenticated_phase1(
    settings: Phase2ContractSettings,
    assertions: list[dict[str, Any]],
) -> dict[str, Any]:
    phase = settings.phase_config
    methodology_path = project_path(phase["methodology_freeze_manifest"])
    freeze_path = project_path(phase["phase_01_freeze_manifest"])
    attestation_path = project_path(phase["phase_01_recovery_attestation"])

    for check_id, path, expected_sha in (
        ("METHODOLOGY_FREEZE_HASH", methodology_path, phase["methodology_freeze_manifest_sha256"]),
        ("PHASE_01_FREEZE_HASH", freeze_path, phase["phase_01_freeze_manifest_sha256"]),
        (
            "PHASE_01_RECOVERY_ATTESTATION_HASH",
            attestation_path,
            phase["phase_01_recovery_attestation_sha256"],
        ),
    ):
        actual_sha = sha256_file(path) if path.is_file() else None
        _assertion(
            assertions,
            check_id,
            gate="input_provenance",
            expression="SHA-256 equals configured content binding",
            expected=expected_sha,
            actual=actual_sha,
            passed=actual_sha == expected_sha,
            evidence_source=project_label(path),
            message="Content-addressed upstream boundary.",
        )

    methodology = _read_json(methodology_path)
    freeze = _read_json(freeze_path)
    attestation = _read_json(attestation_path)

    identity_expectations = {
        "methodology_freeze_id": (
            phase["methodology_freeze_id"],
            methodology.get("freeze_id"),
            freeze.get("identity", {}).get("methodology_freeze_id"),
        ),
        "raw_snapshot_id": (
            phase["raw_snapshot_id"],
            freeze.get("identity", {}).get("raw_snapshot_id"),
            attestation.get("raw_snapshot_id"),
        ),
        "phase_01_build_id": (
            phase["phase_01_build_id"],
            freeze.get("identity", {}).get("phase_01_build_id"),
            attestation.get("phase_01_build_id"),
        ),
        "verification_id": (
            phase["verification_id"],
            freeze.get("identity", {}).get("verification_id"),
            attestation.get("verification_id"),
        ),
        "phase_01_freeze_id": (
            phase["phase_01_freeze_id"],
            freeze.get("freeze_id"),
            attestation.get("freeze_id"),
        ),
    }
    for name, observed in identity_expectations.items():
        _assertion(
            assertions,
            f"UPSTREAM_ID_{name.upper()}",
            gate="input_provenance",
            expression=f"all {name} values are identical",
            expected=observed[0],
            actual=list(observed),
            passed=len(set(observed)) == 1,
            evidence_source="Phase 2 config; methodology freeze; Phase 1 freeze; recovery attestation",
            message="Upstream identifiers reconcile.",
        )

    status_checks = {
        "PHASE_01_STATUS": freeze.get("status") == "phase_01_frozen",
        "ATTESTATION_STATUS": attestation.get("status") == "passed",
        "ATTESTATION_CONTENT_VERIFIER": attestation.get("content_aware_full_verifier", {}).get("status")
        == "passed",
        "ATTESTATION_RAW_RECHECK": all(
            attestation.get("final_raw_recheck", {}).get(name) is True
            for name in (
                "all_files_stable_during_own_hash",
                "all_files_stable_through_entire_recheck",
                "all_hashes_match_primary",
                "current_raw_registry_and_stat_identities_match",
            )
        ),
        "PHASE_02_PERMITTED": str(attestation.get("next_permitted_action", "")).startswith(
            "Begin Phase 2"
        ),
    }
    for check_id, passed in status_checks.items():
        _assertion(
            assertions,
            check_id,
            gate="input_provenance",
            expression=check_id.lower(),
            expected=True,
            actual=passed,
            passed=passed,
            evidence_source=project_label(attestation_path),
            message="Phase 1 handoff permits contract-only Phase 2 work.",
        )

    freeze_sha_in_attestation = attestation.get("manifest", {}).get("sha256")
    _assertion(
        assertions,
        "ATTESTATION_SEALS_PHASE_01_MANIFEST",
        gate="input_provenance",
        expression="attestation manifest SHA-256 equals Phase 2 configured Phase 1 freeze SHA-256",
        expected=phase["phase_01_freeze_manifest_sha256"],
        actual=freeze_sha_in_attestation,
        passed=freeze_sha_in_attestation == phase["phase_01_freeze_manifest_sha256"],
        evidence_source=project_label(attestation_path),
        message="Recovery attestation binds the exact freeze consumed by Phase 2.",
    )

    frozen_entries = _frozen_entry_map(freeze)
    required_primary_names = ("files.csv", "columns.csv", "date_coverage.csv", "key_diagnostics.csv")
    primary_entries = {
        Path(label).name: label
        for label in frozen_entries
        if "/builds/" in label and Path(label).name in required_primary_names
    }
    evidence_paths: dict[str, Path] = {}
    for name in required_primary_names:
        label = primary_entries.get(name, "")
        if not label:
            _assertion(
                assertions,
                f"PRIMARY_EVIDENCE_PRESENT_{name.upper().replace('.', '_')}",
                gate="input_provenance",
                expression="required Phase 1 primary evidence entry exists",
                expected=True,
                actual=False,
                passed=False,
                evidence_source=project_label(freeze_path),
                message=f"Missing frozen {name} entry.",
            )
            continue
        evidence_paths[name] = _verify_frozen_file(label, frozen_entries, assertions)

    for source_contract in settings.source_contracts:
        label = source_contract["phase_01_raw_contract"]
        _verify_frozen_file(label, frozen_entries, assertions)

    return {
        "methodology": methodology,
        "freeze": freeze,
        "attestation": attestation,
        "frozen_entries": frozen_entries,
        "evidence_paths": evidence_paths,
    }


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _validate_methodology_invariants(
    phase: dict[str, Any], assertions: list[dict[str, Any]]
) -> None:
    long = phase["universe"]["long"]
    short = phase["universe"]["short"]
    monotonic = {
        "minimum_price_usd": short["minimum_price_usd"] >= long["minimum_price_usd"],
        "minimum_market_cap_usd": short["minimum_market_cap_usd"]
        >= long["minimum_market_cap_usd"],
        "minimum_median_daily_dollar_volume_usd": short[
            "minimum_median_daily_dollar_volume_usd"
        ]
        >= long["minimum_median_daily_dollar_volume_usd"],
        "minimum_positive_volume_fraction": short["minimum_positive_volume_fraction"]
        >= long["minimum_positive_volume_fraction"],
    }
    _assertion(
        assertions,
        "SHORT_THRESHOLDS_NO_WEAKER_THAN_LONG",
        gate="methodology_invariants",
        expression="each short threshold is at least its long threshold",
        expected={key: True for key in monotonic},
        actual=monotonic,
        passed=all(monotonic.values()),
        evidence_source="configs/phase_02_canonical_panel.toml",
        message="Short candidates inherit the long-investable screen and add stricter feasibility rules.",
    )

    fractions = [long["minimum_positive_volume_fraction"], short["minimum_positive_volume_fraction"]]
    windows_valid = (
        0 <= long["minimum_valid_liquidity_sessions"] <= long["liquidity_window_sessions"]
        and 0 <= short["minimum_valid_liquidity_sessions"] <= short["liquidity_window_sessions"]
        and 0 <= long["minimum_valid_daily_returns"] <= long["return_history_window_sessions"]
    )
    _assertion(
        assertions,
        "THRESHOLD_DOMAINS_VALID",
        gate="methodology_invariants",
        expression="fractions are in [0,1], counts do not exceed windows, and numeric thresholds are nonnegative",
        expected=True,
        actual={"fractions": fractions, "windows_valid": windows_valid},
        passed=all(0 <= value <= 1 for value in fractions)
        and windows_valid
        and all(
            value >= 0
            for value in (
                long["minimum_price_usd"],
                long["minimum_market_cap_usd"],
                long["minimum_median_daily_dollar_volume_usd"],
                short["minimum_price_usd"],
                short["minimum_market_cap_usd"],
                short["minimum_median_daily_dollar_volume_usd"],
            )
        ),
        evidence_source="configs/phase_02_canonical_panel.toml",
        message="Universe thresholds are numerically coherent.",
    )

    base = phase["universe"]["base"]
    cross_border = base["us_incorporation_is_filter"] is False and set(base["share_types"]) == {
        "NS",
        "AD",
    }
    _assertion(
        assertions,
        "US_TRADABLE_NOT_US_DOMICILE_UNIVERSE",
        gate="methodology_invariants",
        expression="USIncFlg is not a filter and NS/AD share types are eligible",
        expected=True,
        actual=cross_border,
        passed=cross_border,
        evidence_source="configs/phase_02_canonical_panel.toml",
        message="Foreign issuers remain eligible when their security is U.S.-tradable.",
    )

    first_slice = phase["first_slice"]
    scope_safe = all(
        (
            first_slice["may_read_final_test_outcomes"] is False,
            first_slice["build_targets"] is False,
            first_slice["build_features"] is False,
            first_slice["build_portfolios"] is False,
            _parse_date(first_slice["end_date"]) < _parse_date("2015-01-01"),
        )
    )
    _assertion(
        assertions,
        "FIRST_SLICE_SCOPE_SAFE",
        gate="sealed_periods",
        expression="first slice is development-only and cannot build outcomes, features, or portfolios",
        expected=True,
        actual=scope_safe,
        passed=scope_safe,
        evidence_source="configs/phase_02_canonical_panel.toml",
        message="The compiler and first data slice preserve the untouched final test.",
    )

    calendar = phase["calendar"]
    boundaries = calendar["boundaries"]
    triples = {
        "last_development": (
            boundaries["last_development_signal"],
            boundaries["last_development_entry"],
            boundaries["last_development_exit"],
        ),
        "development_validation_embargo": (
            boundaries["development_validation_embargo_signal"],
            boundaries["development_validation_embargo_entry"],
            boundaries["development_validation_embargo_exit"],
        ),
        "first_validation": (
            boundaries["first_validation_signal"],
            boundaries["first_validation_entry"],
            boundaries["first_validation_exit"],
        ),
        "last_validation": (
            boundaries["last_validation_signal"],
            boundaries["last_validation_entry"],
            boundaries["last_validation_exit"],
        ),
        "final_embargo": (
            boundaries["embargo_signal"],
            boundaries["embargo_entry"],
            boundaries["embargo_exit"],
        ),
        "first_final": (
            boundaries["first_final_signal"],
            boundaries["first_final_entry"],
            boundaries["first_final_exit"],
        ),
        "last_full_feature": (
            calendar["full_feature_last_signal_date"],
            calendar["full_feature_last_entry_date"],
            calendar["full_feature_last_exit_date"],
        ),
    }
    triple_checks = {
        name: _parse_date(signal) < _parse_date(entry) < _parse_date(exit_)
        for name, (signal, entry, exit_) in triples.items()
    }
    _assertion(
        assertions,
        "CALENDAR_BOUNDARY_CHRONOLOGY",
        gate="calendar_contract",
        expression="signal < entry < exit for every locked boundary interval",
        expected={name: True for name in triple_checks},
        actual=triple_checks,
        passed=all(triple_checks.values()),
        evidence_source="configs/phase_02_canonical_panel.toml",
        message="Both embargo intervals and the full-feature cutoff are explicit.",
    )

    exact_dates = {
        "development_validation_embargo": ("2014-11-28", "2014-12-01", "2015-01-02"),
        "final_embargo": ("2019-11-29", "2019-12-02", "2020-01-02"),
        "first_final": ("2019-12-31", "2020-01-02", "2020-02-03"),
        "last_full_feature": ("2025-07-31", "2025-08-01", "2025-09-02"),
    }
    exact_pass = all(triples[name] == dates for name, dates in exact_dates.items())
    _assertion(
        assertions,
        "LOCKED_BOUNDARY_DATES_EXACT",
        gate="calendar_contract",
        expression="critical boundary triples equal Methodology v1",
        expected=exact_dates,
        actual={name: triples[name] for name in exact_dates},
        passed=exact_pass,
        evidence_source="outputs/methodology_v1_source.md; configs/phase_02_canonical_panel.toml",
        message="Calendar derivation will later prove these dates against observed DlyCalDt values.",
    )

    units = phase["units"]
    field_formula = re.findall(r"\b(?:Dly|Mth|Shr)[A-Za-z0-9]+\b", units["dollar_volume_formula"])
    allowed_formula_fields = {"DlyClose", "DlyVol"}
    formula_pass = set(field_formula) == allowed_formula_fields
    _assertion(
        assertions,
        "DOLLAR_VOLUME_FORMULA_WHITELIST",
        gate="universe_contract",
        expression="dollar-volume formula references only DlyClose and DlyVol",
        expected=sorted(allowed_formula_fields),
        actual=sorted(set(field_formula)),
        passed=formula_pass,
        evidence_source="configs/phase_02_canonical_panel.toml",
        message="Quote-fallback DlyPrc remains diagnostic rather than becoming a traded-close observation.",
    )

    returns = phase["returns"]
    return_bounds = (
        -1 <= returns["missing_performance_delisting_nyse_american"] <= 0
        and -1 <= returns["missing_performance_delisting_nasdaq"] <= 0
        and returns["combined_return_floor"] == -1
        and returns["universal_missing_performance_delisting_stress"] == -1
        and returns["exactly_once_status_before_reconciliation"] == "unproved"
    )
    _assertion(
        assertions,
        "RETURN_IMPUTATION_BOUNDS_AND_PENDING_STATUS",
        gate="returns_contract",
        expression="locked imputation values are bounded and exactly-once status remains unproved",
        expected=True,
        actual=return_bounds,
        passed=return_bounds,
        evidence_source="configs/phase_02_canonical_panel.toml",
        message="No canonical return or label can be published before the reconciliation gate passes.",
    )


def _load_raw_contract(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return {item["filename"]: item for item in payload.get("files", [])}


def _source_categories(source: Mapping[str, Any]) -> dict[str, set[str]]:
    category_names = (
        "identity_columns",
        "date_columns",
        "eligibility_columns",
        "classification_columns",
        "return_columns",
        "execution_columns",
        "reconciliation_columns",
        "prohibited_historical_feature_columns",
    )
    return {name: set(source.get(name, [])) for name in category_names}


def _validate_and_transfer_sources(
    settings: Phase2ContractSettings,
    phase1: dict[str, Any],
    assertions: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_paths = phase1["evidence_paths"]
    required = {"files.csv", "columns.csv", "date_coverage.csv", "key_diagnostics.csv"}
    if set(evidence_paths) != required:
        return {
            "source_inventory": [],
            "source_field_map": [],
            "date_profiles": [],
            "gaps": [],
            "outputs": [],
            "configured_gates": [],
        }

    files_rows = {row["filename"]: row for row in _read_csv(evidence_paths["files.csv"])}
    columns_rows = _read_csv(evidence_paths["columns.csv"])
    columns_by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in columns_rows:
        columns_by_file[row["filename"]].append(row)
    for rows in columns_by_file.values():
        rows.sort(key=lambda item: int(item["position"]))
    key_rows = {row["filename"]: row for row in _read_csv(evidence_paths["key_diagnostics.csv"])}
    date_rows = _read_csv(evidence_paths["date_coverage.csv"])
    date_by_key = {(row["filename"], row["column"]): row for row in date_rows}

    all_source_specs: list[dict[str, Any]] = []
    all_outputs: list[dict[str, Any]] = []
    all_gates: list[dict[str, Any]] = []
    raw_contracts: dict[str, dict[str, Any]] = {}
    vendor_fields: dict[str, dict[str, str]] = {}
    for source_contract in settings.source_contracts:
        all_source_specs.extend(source_contract["sources"])
        all_outputs.extend(source_contract["outputs"])
        all_gates.extend(source_contract["gates"])
        raw_contracts.update(_load_raw_contract(project_path(source_contract["phase_01_raw_contract"])))
        vendor_fields.update({item["field"]: item for item in source_contract["vendor_fields"]})

    filenames = [item["filename"] for item in all_source_specs]
    expected_seven = {
        "crsp_daily_ciz.csv",
        "crsp_monthly_ciz.csv",
        "crsp_names_ciz.csv",
        "crsp_stock_header_ciz.csv",
        "crsp_shares_outstanding_ciz.csv",
        "crsp_distribution_ciz.csv",
        "crsp_delisting_ciz.csv",
    }
    _assertion(
        assertions,
        "EXACT_CRSP_SEVEN_FILE_REGISTRY",
        gate="crsp_schema_mapping",
        expression="source contract contains exactly the seven methodology-owned CRSP files",
        expected=sorted(expected_seven),
        actual=sorted(filenames),
        passed=set(filenames) == expected_seven and len(filenames) == len(set(filenames)),
        evidence_source="configs/phase_02_sources/crsp_backbone.toml; outputs/methodology_v1_source.md",
        message="No raw source is silently added or omitted.",
    )

    output_names = [item["table"] for item in all_outputs]
    output_primary = [item.get("primary_source") for item in all_outputs]
    output_authority_pass = (
        len(output_names) == len(set(output_names))
        and all(isinstance(value, str) and value for value in output_primary)
        and next(
            item for item in all_source_specs if item["filename"] == "crsp_stock_header_ciz.csv"
        ).get("availability_rule", "").startswith("reference-only")
    )
    _assertion(
        assertions,
        "SOURCE_AUTHORITY_UNIQUE",
        gate="crsp_schema_mapping",
        expression="each output has one primary source and current header is reference-only",
        expected=True,
        actual=output_authority_pass,
        passed=output_authority_pass,
        evidence_source="configs/phase_02_sources/crsp_backbone.toml",
        message="Cross-check sources cannot silently become primary field owners.",
    )

    source_inventory: list[dict[str, Any]] = []
    field_map: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for source in sorted(all_source_specs, key=lambda item: item["filename"]):
        filename = source["filename"]
        observed_rows = columns_by_file.get(filename, [])
        observed_columns = [row["name"] for row in observed_rows]
        expected_columns = source["expected_columns"]
        stage_columns = source["stage_columns"]
        raw_contract = raw_contracts.get(filename)
        raw_required = [] if raw_contract is None else raw_contract["required_columns"]
        raw_key = [] if raw_contract is None else raw_contract["key_columns"]

        checks = {
            "present_in_files_csv": filename in files_rows,
            "exact_header": observed_columns == expected_columns,
            "stage_columns_subset": set(stage_columns).issubset(observed_columns),
            "candidate_key_present": set(source["candidate_key"]).issubset(observed_columns),
            "raw_required_subset": set(raw_required).issubset(observed_columns),
            "raw_candidate_key_match": raw_key == source["candidate_key"],
        }
        _assertion(
            assertions,
            "SOURCE_SCHEMA_" + filename.upper().replace(".", "_").replace("-", "_"),
            gate="crsp_schema_mapping",
            expression="file registry, exact header, stage columns, and frozen raw contract agree",
            expected={key: True for key in checks},
            actual=checks,
            passed=all(checks.values()),
            evidence_source=f"Phase 1 files.csv/columns.csv; {filename}; frozen CRSP contract",
            message="Case-sensitive schema mapping is exact.",
        )

        absent_fields = source.get("expected_vendor_fields_absent_from_extract", [])
        absent_checks = {field: field not in observed_columns for field in absent_fields}
        _assertion(
            assertions,
            "EXPECTED_VENDOR_ABSENCE_" + filename.upper().replace(".", "_"),
            gate="crsp_schema_mapping",
            expression="documented vendor fields known absent from this extract remain absent",
            expected={field: True for field in absent_fields},
            actual=absent_checks,
            passed=all(absent_checks.values()),
            evidence_source=f"official CRSP CIZ guide; Phase 1 columns.csv; {filename}",
            message="A newly present field is treated as source drift requiring contract review.",
        )

        file_row = files_rows.get(filename, {})
        key_row = key_rows.get(filename, {})
        date_row = date_by_key.get((filename, source["coverage_axis"]), {})
        source_inventory.append(
            {
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
                "observed_column_count": len(observed_columns),
                "coverage_minimum": date_row.get("minimum", ""),
                "coverage_maximum": date_row.get("maximum", ""),
                "blank_candidate_key_rows": int(
                    key_row.get("rows_with_blank_diagnostic_key_fields", 0)
                ),
                "phase_01_adjacent_duplicate_occurrences": int(
                    key_row.get("adjacent_duplicate_occurrences_complete", 0)
                ),
                "global_uniqueness_status": "NOT_PROVED_PENDING_PHASE_2_SCAN",
            }
        )
        adjacent = int(key_row.get("adjacent_duplicate_occurrences_complete", 0))
        if adjacent:
            gaps.append(
                {
                    "gap_id": f"PHASE1_ADJACENT_DUPLICATES_{filename}",
                    "severity": "hard",
                    "status": "OPEN_REQUIRED",
                    "source_file": filename,
                    "missing_or_ambiguous_item": f"{adjacent} adjacent duplicate-key occurrences; global duplicate count unknown",
                    "methodology_impact": "canonical key uniqueness is not established",
                    "allowed_action": "profile and classify exact versus conflicting duplicates",
                    "blocked_action": "publish canonical security rows",
                    "resolution_required": "full global key profile and deterministic collision audit",
                    "evidence": "Phase 1 key_diagnostics.csv",
                }
            )

        for field in absent_fields:
            gap_id = f"VENDOR_FIELD_ABSENT_{filename}_{field}"
            blocked = "canonical returns and labels" if "Del" in field or "Dist" in field else "unit-sensitive market fields"
            gaps.append(
                {
                    "gap_id": gap_id,
                    "severity": "hard",
                    "status": "DISCLOSED_REQUIRES_RECONCILIATION",
                    "source_file": filename,
                    "missing_or_ambiguous_item": field,
                    "methodology_impact": "vendor guide documents the field but frozen extract omits it",
                    "allowed_action": "retain explicit gap and use configured cross-file diagnostics",
                    "blocked_action": blocked,
                    "resolution_required": "pass the applicable unit or return-reconciliation gate",
                    "evidence": "official CRSP CIZ guide and Phase 1 columns.csv",
                }
            )

        categories = _source_categories(source)
        prohibited = categories["prohibited_historical_feature_columns"]
        for row in observed_rows:
            field = row["name"]
            field_categories = sorted(
                name.removesuffix("_columns")
                for name, values in categories.items()
                if field in values and name != "prohibited_historical_feature_columns"
            )
            vendor = vendor_fields.get(field, {})
            mapping_status = "SELECTED_FOR_FIRST_SLICE" if field in stage_columns else "SOURCE_CONTEXT_ONLY"
            blocker = ""
            if field in prohibited:
                mapping_status = "PROHIBITED_AS_HISTORICAL_PRIMARY"
                blocker = "SOURCE_AUTHORITY_UNIQUE"
            field_map.append(
                {
                    "source_file": filename,
                    "source_column": field,
                    "source_position": int(row["position"]),
                    "physical_csv_type": row["physical_csv_type"],
                    "sample_inferred_type": row["sample_inferred_type"],
                    "field_categories": field_categories,
                    "canonical_unit": vendor.get("unit", "UNSPECIFIED_NOT_YET_TRANSFORMED"),
                    "vendor_definition": vendor.get("definition", ""),
                    "source_role": source["role"],
                    "observation_date_rule": source["coverage_axis"],
                    "availability_rule": source["availability_rule"],
                    "mapping_status": mapping_status,
                    "blocker_id": blocker,
                }
            )

    mth_prc = date_by_key.get(("crsp_monthly_ciz.csv", "MthPrcDt"), {})
    mth_blank = int(mth_prc.get("null_or_blank", 0))
    if mth_blank:
        gaps.append(
            {
                "gap_id": "MONTHLY_PRICE_DATE_MISSING",
                "severity": "hard",
                "status": "OPEN_REQUIRED",
                "source_file": "crsp_monthly_ciz.csv",
                "missing_or_ambiguous_item": f"MthPrcDt blank on {mth_blank} rows",
                "methodology_impact": "MthCalDt cannot substitute as the point-in-time market observation date",
                "allowed_action": "use affected rows only for explicitly labeled ex-post reconciliation where valid",
                "blocked_action": "use affected rows as point-in-time market observations",
                "resolution_required": "preserve missing availability and report coverage",
                "evidence": "Phase 1 date_coverage.csv",
            }
        )

    return {
        "source_inventory": source_inventory,
        "source_field_map": sorted(
            field_map, key=lambda row: (row["source_file"], row["source_position"])
        ),
        "date_profiles": [row for row in date_rows if row["filename"] in expected_seven],
        "gaps": sorted(gaps, key=lambda row: row["gap_id"]),
        "outputs": sorted(all_outputs, key=lambda row: row["table"]),
        "configured_gates": sorted(all_gates, key=lambda row: row["gate_id"]),
    }


def _calendar_boundary_rows(phase: Mapping[str, Any]) -> list[dict[str, Any]]:
    calendar = phase["calendar"]
    b = calendar["boundaries"]
    return [
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
            "methodology_anchor": "Chronology and information boundaries",
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
            "methodology_anchor": "Chronology and information boundaries",
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
            "methodology_anchor": "Chronology and information boundaries",
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
            "methodology_anchor": "Chronology and information boundaries",
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
            "methodology_anchor": "Chronology and information boundaries",
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
            "methodology_anchor": "Chronology and information boundaries",
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
            "methodology_anchor": "Chronology and information boundaries",
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
            "methodology_anchor": "Chronology and information boundaries",
        },
    ]


def _future_assertions(
    configured_gates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    compile_pass = {"CRSP_SCHEMA_EXACT", "PHASE_01_BOUNDARY_AUTHENTIC", "SOURCE_AUTHORITY_UNIQUE"}
    for gate in configured_gates:
        gate_id = gate["gate_id"]
        is_compile = gate["stage"] == "contract_compile"
        status = "PASS" if is_compile and gate_id in compile_pass else "NOT_RUN"
        result.append(
            {
                "check_id": f"FUTURE_GATE_{gate_id}",
                "gate": gate_id,
                "scope": gate["stage"],
                "severity": gate["severity"],
                "check_kind": "configured_stage_gate",
                "expression": gate["requirement"],
                "expected": "PASS before blocked action",
                "actual": "contract-only" if status == "NOT_RUN" else "compile evidence passed",
                "evidence_source": "configs/phase_02_sources/crsp_backbone.toml",
                "status": status,
                "blocks_action": "none" if status == "PASS" else gate["stage"],
                "message": "Data-dependent checks remain pending; contract compilation is not evidence of a pass.",
            }
        )
    return result


def compile_contract(settings: Phase2ContractSettings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    assertions = _validate_configuration_schema(settings.phase_config, settings.source_contracts)
    phase1 = _load_authenticated_phase1(settings, assertions)
    _validate_methodology_invariants(settings.phase_config, assertions)
    transfer = _validate_and_transfer_sources(settings, phase1, assertions)

    compile_failures = [row for row in assertions if row["status"] == "FAIL"]
    if compile_failures:
        summary = "; ".join(f"{row['check_id']}: {row['message']}" for row in compile_failures)
        raise ContractValidationError(f"Phase 2 contract compilation failed: {summary}")

    future = _future_assertions(transfer["configured_gates"])
    assertion_registry = sorted(assertions + future, key=lambda row: row["check_id"])
    phase = settings.phase_config
    input_hashes = {
        project_label(settings.phase_config_path): sha256_file(settings.phase_config_path),
        **{
            project_label(path): sha256_file(path) for path in sorted(settings.source_contract_paths)
        },
        phase["methodology_freeze_manifest"]: phase["methodology_freeze_manifest_sha256"],
        phase["phase_01_freeze_manifest"]: phase["phase_01_freeze_manifest_sha256"],
        phase["phase_01_recovery_attestation"]: phase[
            "phase_01_recovery_attestation_sha256"
        ],
        project_label(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
    }
    for path in phase1["evidence_paths"].values():
        input_hashes[project_label(path)] = sha256_file(path)

    binding = {
        "schema_version": 1,
        "phase": phase["phase"],
        "contract_version": phase["contract_version"],
        "methodology_freeze_id": phase["methodology_freeze_id"],
        "raw_snapshot_id": phase["raw_snapshot_id"],
        "phase_01_build_id": phase["phase_01_build_id"],
        "verification_id": phase["verification_id"],
        "phase_01_freeze_id": phase["phase_01_freeze_id"],
        "input_hashes": dict(sorted(input_hashes.items())),
        "semantic_contract": phase,
        "source_contracts": list(settings.source_contracts),
        "runtime_identity": {
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "tomllib": "stdlib",
            "serialization": "canonical_json_sorted_keys_compact_lf_and_csv_lf",
        },
    }
    contract_digest = sha256_json(binding)
    contract_id = f"phase02-crsp-{contract_digest[:16]}"
    contract = {
        "schema_version": 1,
        "contract_id": contract_id,
        "contract_sha256": contract_digest,
        "phase": phase["phase"],
        "status": "contract_compiled_data_gates_pending",
        "scope": {
            "raw_rows_read": 0,
            "targets_built": False,
            "features_built": False,
            "portfolios_built": False,
            "performance_results_seen": False,
            "first_permitted_data_action": "build and validate observed CRSP trading-date calendar",
        },
        "upstream": {
            "methodology_freeze_id": phase["methodology_freeze_id"],
            "raw_snapshot_id": phase["raw_snapshot_id"],
            "phase_01_build_id": phase["phase_01_build_id"],
            "verification_id": phase["verification_id"],
            "phase_01_freeze_id": phase["phase_01_freeze_id"],
            "phase_01_freeze_manifest_sha256": phase["phase_01_freeze_manifest_sha256"],
            "phase_01_recovery_attestation_sha256": phase[
                "phase_01_recovery_attestation_sha256"
            ],
        },
        "compile_summary": {
            "source_file_count": len(transfer["source_inventory"]),
            "source_field_count": len(transfer["source_field_map"]),
            "output_table_contract_count": len(transfer["outputs"]),
            "compile_assertion_pass_count": sum(
                row["status"] == "PASS" for row in assertion_registry
            ),
            "compile_assertion_fail_count": 0,
            "future_gate_not_run_count": sum(
                row["status"] == "NOT_RUN" for row in assertion_registry
            ),
            "known_gap_count": len(transfer["gaps"]),
        },
        "binding": binding,
    }
    return {
        "contract_id": contract_id,
        "contract_digest": contract_digest,
        "contract": contract,
        "source_inventory": transfer["source_inventory"],
        "source_field_map": transfer["source_field_map"],
        "output_table_contracts": transfer["outputs"],
        "calendar_boundary_expectations": _calendar_boundary_rows(phase),
        "assertion_registry": assertion_registry,
        "known_gaps": transfer["gaps"],
        "input_hashes": dict(sorted(input_hashes.items())),
        "run_metadata": {
            "schema_version": 1,
            "contract_id": contract_id,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "compiler": {
                "path": project_label(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "operational_note": "Timestamp-free deterministic artifact metadata; invocation timing is stored under runs/.",
        },
    }


SOURCE_INVENTORY_FIELDS = (
    "filename",
    "source_family",
    "role",
    "grain",
    "candidate_key",
    "coverage_axis",
    "availability_rule",
    "calendar_owner",
    "size_bytes",
    "sha256",
    "parsed_data_rows",
    "observed_column_count",
    "coverage_minimum",
    "coverage_maximum",
    "blank_candidate_key_rows",
    "phase_01_adjacent_duplicate_occurrences",
    "global_uniqueness_status",
)

FIELD_MAP_FIELDS = (
    "source_file",
    "source_column",
    "source_position",
    "physical_csv_type",
    "sample_inferred_type",
    "field_categories",
    "canonical_unit",
    "vendor_definition",
    "source_role",
    "observation_date_rule",
    "availability_rule",
    "mapping_status",
    "blocker_id",
)

ASSERTION_FIELDS = (
    "check_id",
    "gate",
    "scope",
    "severity",
    "check_kind",
    "expression",
    "expected",
    "actual",
    "evidence_source",
    "status",
    "blocks_action",
    "message",
)

GAP_FIELDS = (
    "gap_id",
    "severity",
    "status",
    "source_file",
    "missing_or_ambiguous_item",
    "methodology_impact",
    "allowed_action",
    "blocked_action",
    "resolution_required",
    "evidence",
)

CALENDAR_FIELDS = (
    "boundary_id",
    "holding_month",
    "stage",
    "signal_date",
    "entry_date",
    "exit_date",
    "is_embargo",
    "is_partial",
    "formal_inference_expected",
    "methodology_anchor",
)


def _write_contract_directory(directory: Path, compiled: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "contract.json", compiled["contract"])
    _write_csv(
        directory / "source_files.csv", SOURCE_INVENTORY_FIELDS, compiled["source_inventory"]
    )
    _write_csv(
        directory / "source_field_map.csv", FIELD_MAP_FIELDS, compiled["source_field_map"]
    )
    _write_json(directory / "output_table_contracts.json", compiled["output_table_contracts"])
    _write_csv(
        directory / "calendar_boundary_expectations.csv",
        CALENDAR_FIELDS,
        compiled["calendar_boundary_expectations"],
    )
    _write_csv(
        directory / "assertion_registry.csv", ASSERTION_FIELDS, compiled["assertion_registry"]
    )
    _write_csv(directory / "known_gaps.csv", GAP_FIELDS, compiled["known_gaps"])
    _write_json(directory / "run_metadata.json", compiled["run_metadata"])
    with (directory / "input_hashes.sha256").open("w", encoding="utf-8", newline="") as handle:
        for label, digest in compiled["input_hashes"].items():
            handle.write(f"{digest}  {label}\n")

    artifact_files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "artifact_hashes.sha256"
    )
    with (directory / "artifact_hashes.sha256").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        for path in artifact_files:
            handle.write(f"{sha256_file(path)}  {path.name}\n")


def verify_published_contract(directory: Path) -> list[str]:
    failures: list[str] = []
    hash_path = directory / "artifact_hashes.sha256"
    if not hash_path.is_file():
        return ["missing artifact_hashes.sha256"]
    expected_files: set[str] = set()
    for line in hash_path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            failures.append(f"malformed hash line: {line!r}")
            continue
        expected_files.add(name)
        path = directory / name
        if not path.is_file():
            failures.append(f"missing sealed artifact: {name}")
        elif sha256_file(path) != digest:
            failures.append(f"artifact hash mismatch: {name}")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    expected_all = expected_files | {"artifact_hashes.sha256"}
    if actual != expected_all:
        failures.append(
            f"artifact file-set mismatch: expected {sorted(expected_all)}, observed {sorted(actual)}"
        )
    try:
        contract = _read_json(directory / "contract.json")
        if contract.get("contract_id") != directory.name:
            failures.append("contract_id does not match directory name")
    except Exception as exc:  # pragma: no cover - defensive verifier path
        failures.append(f"contract JSON unreadable: {exc}")
    return failures


def publish_contract(
    compiled: Mapping[str, Any], settings: Phase2ContractSettings | None = None
) -> Path:
    settings = settings or load_settings()
    contract_id = compiled["contract_id"]
    contracts_root = settings.artifact_root / "contracts"
    staging_root = settings.artifact_root / "_staging"
    contracts_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    final_directory = contracts_root / contract_id

    if final_directory.exists():
        failures = verify_published_contract(final_directory)
        if failures:
            raise ContractValidationError(
                "Existing content-addressed contract is invalid: " + "; ".join(failures)
            )
    else:
        stage_parent = Path(tempfile.mkdtemp(prefix="contract-", dir=staging_root))
        staged = stage_parent / contract_id
        try:
            _write_contract_directory(staged, compiled)
            failures = verify_published_contract(staged)
            if failures:
                raise ContractValidationError(
                    "Staged Phase 2 contract verification failed: " + "; ".join(failures)
                )
            os.rename(staged, final_directory)
        finally:
            if stage_parent.exists():
                shutil.rmtree(stage_parent)

    pointer = {
        "schema_version": 1,
        "contract_id": contract_id,
        "contract_path": project_label(final_directory),
        "contract_json_sha256": sha256_file(final_directory / "contract.json"),
        "artifact_hashes_sha256": sha256_file(final_directory / "artifact_hashes.sha256"),
        "status": compiled["contract"]["status"],
    }
    settings.artifact_root.mkdir(parents=True, exist_ok=True)
    pointer_tmp = settings.artifact_root / f".LATEST_CONTRACT_PRIMARY.{os.getpid()}.tmp"
    _write_json(pointer_tmp, pointer)
    os.replace(pointer_tmp, settings.artifact_root / "LATEST_CONTRACT_PRIMARY.json")

    run_directory = settings.run_root / contract_id
    run_directory.mkdir(parents=True, exist_ok=True)
    receipt_path = run_directory / "contract_build_receipt.json"
    if not receipt_path.exists():
        receipt = {
            "schema_version": 1,
            "contract_id": contract_id,
            "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "published_primary_pending_independent_verification",
            "artifact_path": project_label(final_directory),
            "contract_json_sha256": pointer["contract_json_sha256"],
            "artifact_hashes_sha256": pointer["artifact_hashes_sha256"],
            "raw_rows_read": 0,
        }
        _write_json(receipt_path, receipt)
    return final_directory


def build_contract(settings: Phase2ContractSettings | None = None) -> tuple[dict[str, Any], Path]:
    settings = settings or load_settings()
    compiled = compile_contract(settings)
    final_directory = publish_contract(compiled, settings)
    return compiled, final_directory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PHASE_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    compiled = compile_contract(settings)
    if args.check_only:
        print(
            json.dumps(
                {
                    "contract_id": compiled["contract_id"],
                    "status": compiled["contract"]["status"],
                    "compile_assertion_fail_count": compiled["contract"]["compile_summary"][
                        "compile_assertion_fail_count"
                    ],
                    "raw_rows_read": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    _, path = build_contract(settings)
    print(
        json.dumps(
            {
                "contract_id": compiled["contract_id"],
                "path": project_label(path),
                "status": compiled["contract"]["status"],
                "raw_rows_read": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
