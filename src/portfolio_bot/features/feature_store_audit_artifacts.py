"""Verification-bound Stage 6 notebook and portable report payload.

This module is deliberately downstream of the independent verifier.  It does
not calculate feature values, change a primary build, or grant authority.  It
authenticates one verified Stage 6 build and turns the already-published
quality evidence into:

* ``notebooks/06_stage_06_feature_store_quality.ipynb``; and
* ``reports/stage_06_feature_store/artifact.json``.

HTML delivery remains a separate, explicit step.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import nbformat as nbf
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.features.feature_store_contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
)
from portfolio_bot.features.feature_store_quality import (
    FEATURE_COVERAGE_SCHEMA,
    FEATURE_TIMING_CHECKS_SCHEMA,
)
from portfolio_bot.features.phase_03_contract import canonical_json_bytes


DEFAULT_NOTEBOOK = (
    PROJECT_ROOT / "notebooks/06_stage_06_feature_store_quality.ipynb"
)
DEFAULT_REPORT_DIRECTORY = PROJECT_ROOT / "reports/stage_06_feature_store"
_VERIFIED_STATUS = "VERIFIED_PASSED"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BUILD_ID = re.compile(r"phase03-features-[0-9a-f]{24}\Z")
_VERIFICATION_ID = re.compile(r"verify-[0-9a-f]{16}\Z")
_VERIFIED_AUTHORIZATIONS = {
    "independent_verification_completed": True,
    "stage_07_labels_authorized": True,
    "models_portfolios_pnl_or_backtests_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
_PRIMARY_AUTHORIZATIONS = {
    "independent_verification_required": True,
    "stage_07_labels_authorized": False,
    "models_portfolios_pnl_or_backtests_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
_VERIFIED_POINTER_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "verification_id",
        "status",
        "primary_manifest_sha256",
        "independent_check_sha256",
        "differences_sha256",
        "authorizations",
    }
)
_INDEPENDENT_CHECK_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "verification_id",
        "status",
        "primary_manifest_sha256",
        "checks",
        "differences_sha256",
        "output_hashes_sha256",
        "authorizations",
    }
)
_CHECK_KEYS = frozenset(
    {"check_count", "pass_count", "fail_count", "results_sha256"}
)
_PRIMARY_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "status",
        "bindings",
        "inputs",
        "outputs",
        "summary",
        "authorizations",
    }
)
_FILE_RECEIPT_KEYS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "rows",
        "row_groups",
        "schema_sha256",
    }
)
_PRIMARY_STATUS = "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
_FAMILY_DISPLAY_CODES = {
    "aggregate_state": "AGG",
    "optionmetrics": "OPT",
    "ownership_filings_events_segments": "OWN",
    "price_trends_path": "PRC",
    "risk_return_distribution": "RSK",
    "size_liquidity_trading_frictions": "LIQ",
    "valuation_fundamentals": "VAL",
}


class Stage6AuditArtifactError(RuntimeError):
    """Raised when report inputs are not one authenticated verified build."""


@dataclass(frozen=True)
class Stage6AuditArtifactReceipt:
    """Small machine-readable handoff for the generated audit artifacts."""

    build_id: str
    verification_id: str
    notebook_path: Path
    artifact_path: Path
    notebook_sha256: str
    artifact_sha256: str
    coverage_rows: int
    gate_rows: int
    notebook_executed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "build_id": self.build_id,
            "verification_id": self.verification_id,
            "notebook_path": self.notebook_path.as_posix(),
            "artifact_path": self.artifact_path.as_posix(),
            "notebook_sha256": self.notebook_sha256,
            "artifact_sha256": self.artifact_sha256,
            "coverage_rows": self.coverage_rows,
            "gate_rows": self.gate_rows,
            "notebook_executed": self.notebook_executed,
        }


@dataclass(frozen=True)
class _AuthenticatedEvidence:
    policy: Mapping[str, Any]
    pointer: Mapping[str, Any]
    independent: Mapping[str, Any]
    manifest: Mapping[str, Any]
    summary: Mapping[str, Any]
    coverage: pa.Table
    timing: pa.Table
    project_root: Path
    database_root: Path
    quality_relative_path: PurePosixPath
    verified_pointer_relative_path: PurePosixPath
    independent_check_relative_path: PurePosixPath
    primary_manifest_relative_path: PurePosixPath


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage6AuditArtifactError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage6AuditArtifactError(
            f"Cannot read canonical JSON evidence: {path}"
        ) from exc
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise Stage6AuditArtifactError(
            f"Cannot read Stage 6 policy: {path}"
        ) from exc
    _require(isinstance(value, dict), "Stage 6 policy must be a TOML table")
    return value


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    _require(not candidate.is_symlink(), f"Symlink prohibited for {label}")
    resolved = candidate.resolve()
    _require(resolved.is_file(), f"Missing {label}: {resolved}")
    return resolved


def _real_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    _require(not candidate.is_symlink(), f"Symlink prohibited for {label}")
    resolved = candidate.resolve()
    _require(resolved.is_dir(), f"Missing {label}: {resolved}")
    return resolved


def _safe_relative(path: Path, root: Path, label: str) -> PurePosixPath:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise Stage6AuditArtifactError(
            f"{label} must remain below the project root"
        ) from exc
    result = PurePosixPath(relative.as_posix())
    _require(
        not result.is_absolute() and ".." not in result.parts,
        f"Unsafe relative provenance for {label}",
    )
    return result


def _safe_database_relative(value: Any) -> PurePosixPath:
    _require(isinstance(value, str) and value, "Missing quality output path")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts,
        "Quality output path is not portable",
    )
    return path


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _parquet_receipt(path: Path) -> tuple[int, int, str]:
    parquet = pq.ParquetFile(path)
    return (
        parquet.metadata.num_rows,
        parquet.metadata.num_row_groups,
        _schema_sha256(parquet.schema_arrow),
    )


def _authenticate_file_receipt(
    *, quality_dir: Path, name: str, receipt: Mapping[str, Any]
) -> None:
    _require(
        set(receipt) == _FILE_RECEIPT_KEYS,
        f"Quality file receipt keys differ: {name}",
    )
    _require(receipt["path"] == name, f"Quality receipt path differs: {name}")
    path = _regular_file(quality_dir / name, f"quality file {name}")
    _require(
        _SHA256.fullmatch(str(receipt["sha256"])) is not None,
        f"Invalid quality SHA-256: {name}",
    )
    _require(
        path.stat().st_size == receipt["size_bytes"],
        f"Quality file size differs: {name}",
    )
    _require(
        _sha256_file(path) == receipt["sha256"],
        f"Quality file hash differs: {name}",
    )
    if path.suffix == ".parquet":
        rows, row_groups, schema_hash = _parquet_receipt(path)
        _require(rows == receipt["rows"], f"Quality row count differs: {name}")
        _require(
            row_groups == receipt["row_groups"],
            f"Quality row-group count differs: {name}",
        )
        _require(
            schema_hash == receipt["schema_sha256"],
            f"Quality schema hash differs: {name}",
        )
    else:
        _require(
            receipt["rows"] is None
            and receipt["row_groups"] is None
            and receipt["schema_sha256"] is None,
            f"Non-Parquet quality receipt contains table metadata: {name}",
        )


def _authenticate_evidence(
    *,
    verified_pointer_path: str | Path,
    independent_check_path: str | Path,
    primary_manifest_path: str | Path,
    quality_directory: str | Path,
    project_root: str | Path,
    database_root: str | Path | None,
    policy_path: str | Path,
) -> _AuthenticatedEvidence:
    project = _real_directory(project_root, "project root")
    policy_file = _regular_file(policy_path, "Stage 6 policy")
    policy = _toml(policy_file)
    database = (
        _real_directory(database_root, "database root")
        if database_root is not None
        else _real_directory(policy["database"]["root"], "database root")
    )
    pointer_path = _regular_file(
        verified_pointer_path, "Stage 6 verified pointer"
    )
    independent_path = _regular_file(
        independent_check_path, "Stage 6 independent check"
    )
    manifest_path = _regular_file(
        primary_manifest_path, "Stage 6 primary manifest"
    )
    quality_dir = _real_directory(quality_directory, "Stage 6 quality directory")

    expected_pointer_path = (
        project / str(policy["publication"]["latest_verified_pointer"])
    ).resolve()
    _require(
        pointer_path == expected_pointer_path,
        "Verified pointer path differs from the frozen policy",
    )

    pointer = _json(pointer_path)
    independent = _json(independent_path)
    manifest = _json(manifest_path)
    _require(
        set(pointer) == _VERIFIED_POINTER_KEYS,
        "Verified pointer keys differ from the final Stage 6 schema",
    )
    _require(
        set(independent) == _INDEPENDENT_CHECK_KEYS,
        "Independent-check keys differ from the final Stage 6 schema",
    )
    _require(
        set(manifest) == _PRIMARY_MANIFEST_KEYS,
        "Primary-manifest keys differ",
    )

    build_id = pointer.get("build_id")
    verification_id = pointer.get("verification_id")
    _require(
        isinstance(build_id, str) and _BUILD_ID.fullmatch(build_id) is not None,
        "Invalid Stage 6 build ID",
    )
    _require(
        isinstance(verification_id, str)
        and _VERIFICATION_ID.fullmatch(verification_id) is not None,
        "Invalid Stage 6 verification ID",
    )
    expected_independent_path = (
        project
        / str(policy["publication"]["artifact_root"])
        / str(policy["publication"]["verification_namespace"])
        / build_id
        / verification_id
        / "independent_check.json"
    ).resolve()
    _require(
        independent_path == expected_independent_path,
        "Independent-check path differs from the frozen publication layout",
    )
    expected_manifest_path = (
        project
        / str(policy["publication"]["artifact_root"])
        / str(policy["publication"]["primary_build_namespace"])
        / build_id
        / "manifest.json"
    ).resolve()
    _require(
        manifest_path == expected_manifest_path,
        "Primary-manifest path differs from the frozen publication layout",
    )

    common = ("schema_version", "contract_id", "policy_id", "build_id")
    for key in common:
        _require(
            pointer[key] == independent[key],
            f"Pointer and independent check disagree: {key}",
        )
    for key in ("schema_version", "contract_id", "policy_id", "build_id"):
        _require(
            pointer[key] == manifest[key],
            f"Verified and primary evidence disagree: {key}",
        )
    _require(pointer["schema_version"] == 1, "Wrong verified schema version")
    _require(
        pointer["contract_id"] == policy["contract_id"]
        and pointer["policy_id"] == policy["policy_id"],
        "Verified identity differs from the frozen Stage 6 policy",
    )
    _require(
        pointer["status"] == independent["status"] == _VERIFIED_STATUS,
        "Stage 6 evidence is not VERIFIED_PASSED",
    )
    _require(
        pointer["authorizations"]
        == independent["authorizations"]
        == _VERIFIED_AUTHORIZATIONS,
        "Verified Stage 6 authorization boundary differs",
    )
    _require(
        independent["verification_id"] == verification_id,
        "Verification IDs differ",
    )
    _require(
        manifest["status"] == _PRIMARY_STATUS,
        "Primary manifest did not pass its pre-verification gate",
    )
    _require(
        manifest["authorizations"] == _PRIMARY_AUTHORIZATIONS,
        "Primary manifest authorization boundary differs",
    )

    primary_hash = _sha256_file(manifest_path)
    _require(
        _SHA256.fullmatch(str(pointer["primary_manifest_sha256"])) is not None
        and pointer["primary_manifest_sha256"]
        == independent["primary_manifest_sha256"]
        == primary_hash,
        "Primary-manifest hash binding differs",
    )
    _require(
        _SHA256.fullmatch(str(pointer["independent_check_sha256"])) is not None
        and pointer["independent_check_sha256"]
        == _sha256_file(independent_path),
        "Independent-check hash binding differs",
    )
    _require(
        _SHA256.fullmatch(str(independent["output_hashes_sha256"])) is not None,
        "Invalid independent output-hashes digest",
    )
    checks = independent.get("checks")
    _require(
        isinstance(checks, Mapping) and set(checks) == _CHECK_KEYS,
        "Independent check summary differs",
    )
    _require(
        isinstance(checks["check_count"], int)
        and checks["check_count"] > 0
        and checks["pass_count"] == checks["check_count"]
        and checks["fail_count"] == 0
        and _SHA256.fullmatch(str(checks["results_sha256"])) is not None,
        "Independent verification did not pass every check",
    )

    differences_path = _regular_file(
        independent_path.parent / "differences.json",
        "Stage 6 verification differences",
    )
    differences_bytes = differences_path.read_bytes()
    try:
        differences = json.loads(differences_bytes)
    except json.JSONDecodeError as exc:
        raise Stage6AuditArtifactError(
            "Stage 6 differences payload is not JSON"
        ) from exc
    _require(
        differences == [] and differences_bytes == canonical_json_bytes([]),
        "Verified Stage 6 differences payload is not the canonical empty list",
    )
    differences_hash = _sha256_file(differences_path)
    _require(
        pointer["differences_sha256"]
        == independent["differences_sha256"]
        == differences_hash,
        "Verification differences hash binding differs",
    )

    outputs = manifest.get("outputs")
    _require(
        isinstance(outputs, Mapping)
        and set(outputs) == {"interim", "processed", "quality"},
        "Primary output sets differ",
    )
    quality_output = outputs["quality"]
    _require(
        isinstance(quality_output, Mapping)
        and set(quality_output) == {"path", "success_sha256", "files"},
        "Primary quality output receipt differs",
    )
    quality_relative = _safe_database_relative(quality_output["path"])
    _require(
        quality_dir == (database / Path(quality_relative.as_posix())).resolve(),
        "Quality directory differs from the authenticated primary manifest",
    )
    expected_names = set(policy["outputs"]["quality"]["required_files"])
    receipts = quality_output.get("files")
    _require(
        isinstance(receipts, Mapping) and set(receipts) == expected_names,
        "Quality output file set is not closed",
    )
    observed_names = {
        child.name
        for child in quality_dir.iterdir()
        if child.is_file() and not child.is_symlink()
    }
    _require(
        observed_names == expected_names
        and all(not child.is_symlink() for child in quality_dir.iterdir()),
        "Quality directory file set differs from the manifest",
    )
    for name in sorted(expected_names):
        receipt = receipts[name]
        _require(
            isinstance(receipt, Mapping),
            f"Invalid quality file receipt: {name}",
        )
        _authenticate_file_receipt(
            quality_dir=quality_dir, name=name, receipt=receipt
        )
    _require(
        quality_output["success_sha256"]
        == receipts["_SUCCESS.json"]["sha256"],
        "Quality success receipt differs",
    )

    summary_path = quality_dir / "quality_summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = _json(summary_path)
    _require(
        summary_bytes == canonical_json_bytes(summary),
        "Quality summary is not canonical JSON",
    )
    _require(
        summary.get("build_id") == build_id
        and summary.get("contract_id") == pointer["contract_id"]
        and summary.get("policy_id") == pointer["policy_id"]
        and summary.get("status") == _PRIMARY_STATUS,
        "Quality summary identity or status differs",
    )
    counts = summary.get("counts")
    manifest_summary = manifest.get("summary")
    _require(
        isinstance(counts, Mapping) and isinstance(manifest_summary, Mapping),
        "Missing Stage 6 summary counts",
    )
    for quality_key, manifest_key in (
        ("raw_rows", "raw_rows"),
        ("raw_feature_columns", "economic_features"),
        ("feature_families", "feature_families"),
        ("lineage_rows", "lineage_rows"),
    ):
        _require(
            counts.get(quality_key) == manifest_summary.get(manifest_key),
            f"Primary and quality counts disagree: {quality_key}",
        )
    _require(
        manifest_summary.get("hard_gate_failures") == 0,
        "Primary manifest records hard-gate failures",
    )

    coverage = pq.read_table(quality_dir / "feature_coverage.parquet")
    timing = pq.read_table(quality_dir / "feature_timing_checks.parquet")
    _require(
        coverage.schema.equals(FEATURE_COVERAGE_SCHEMA, check_metadata=True),
        "Feature-coverage schema differs",
    )
    _require(
        timing.schema.equals(FEATURE_TIMING_CHECKS_SCHEMA, check_metadata=True),
        "Feature-timing-check schema differs",
    )
    return _AuthenticatedEvidence(
        policy=policy,
        pointer=pointer,
        independent=independent,
        manifest=manifest,
        summary=summary,
        coverage=coverage,
        timing=timing,
        project_root=project,
        database_root=database,
        quality_relative_path=quality_relative,
        verified_pointer_relative_path=_safe_relative(
            pointer_path, project, "verified pointer"
        ),
        independent_check_relative_path=_safe_relative(
            independent_path, project, "independent check"
        ),
        primary_manifest_relative_path=_safe_relative(
            manifest_path, project, "primary manifest"
        ),
    )


def _coverage_rows(evidence: _AuthenticatedEvidence) -> list[dict[str, Any]]:
    aggregate: dict[tuple[str, int], dict[str, Any]] = {}
    observed_keys: set[tuple[str, int]] = set()
    for row in evidence.coverage.to_pylist():
        feature = row["feature_name"]
        family = row["feature_family"]
        year = int(row["signal_year"])
        total = int(row["total_rows"])
        present = int(row["present_rows"])
        missing = int(row["missing_rows"])
        rate = float(row["coverage_rate"])
        key = (feature, year)
        _require(key not in observed_keys, "Duplicate feature-coverage key")
        observed_keys.add(key)
        _require(
            isinstance(feature, str)
            and feature
            and isinstance(family, str)
            and family,
            "Blank feature-coverage identity",
        )
        _require(
            total > 0
            and present >= 0
            and missing >= 0
            and present + missing == total
            and math.isfinite(rate)
            and math.isclose(
                rate, present / total, rel_tol=0.0, abs_tol=1e-15
            ),
            f"Feature-coverage arithmetic differs: {feature}/{year}",
        )
        bucket = aggregate.setdefault(
            (family, year),
            {
                "feature_family": family,
                "feature_family_code": _FAMILY_DISPLAY_CODES.get(
                    family, family
                ),
                "signal_year": year,
                "total_rows": 0,
                "present_rows": 0,
                "missing_rows": 0,
                "feature_count": 0,
            },
        )
        bucket["total_rows"] += total
        bucket["present_rows"] += present
        bucket["missing_rows"] += missing
        bucket["feature_count"] += 1
    _require(bool(aggregate), "Feature-coverage table is empty")
    rows: list[dict[str, Any]] = []
    for key in sorted(aggregate, key=lambda item: (item[1], item[0].encode())):
        row = aggregate[key]
        row["coverage_rate"] = row["present_rows"] / row["total_rows"]
        rows.append(row)
    _require(len(rows) <= 2_000, "Coverage snapshot exceeds portable limit")
    return rows


def _gate_rows(evidence: _AuthenticatedEvidence) -> list[dict[str, Any]]:
    hard_gates = evidence.summary.get("hard_gates")
    _require(isinstance(hard_gates, Mapping), "Missing quality hard gates")
    observed: dict[str, Mapping[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for raw in evidence.timing.to_pylist():
        name = raw["check_name"]
        _require(name not in observed, "Duplicate timing-check name")
        observed[name] = raw
        _require(
            raw["passed"] is True
            and raw["observed_count"] == raw["expected_count"],
            f"Stage 6 timing or integrity gate failed: {name}",
        )
        rows.append(
            {
                "check_name": name,
                "gate_class": (
                    "frozen_policy"
                    if name in hard_gates
                    else "structural_integrity"
                ),
                "observed_count": int(raw["observed_count"]),
                "expected_count": int(raw["expected_count"]),
                "passed": bool(raw["passed"]),
                "result": "PASS",
                "severity": raw["severity"],
                "details": raw["details"],
            }
        )
    _require(bool(rows), "Feature-timing-check table is empty")
    _require(
        set(hard_gates).issubset(observed),
        "Quality hard gates are missing from the exact timing table",
    )
    for name, value in hard_gates.items():
        _require(
            observed[name]["observed_count"] == value,
            f"Quality-summary gate differs from timing table: {name}",
        )
    rows.sort(key=lambda row: row["check_name"].encode("utf-8"))
    _require(len(rows) <= 2_000, "Gate snapshot exceeds portable limit")
    return rows


def _source_query(
    *,
    source_id: str,
    label: str,
    sql: str,
    tables: Sequence[str],
    description: str,
    filters: Sequence[str],
    metric_definitions: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "description": description,
            "tables_used": list(tables),
            "filters": list(filters),
            "metric_definitions": list(metric_definitions),
        },
    }


def _sources(evidence: _AuthenticatedEvidence) -> list[dict[str, Any]]:
    build_id = str(evidence.pointer["build_id"])
    verification_id = str(evidence.pointer["verification_id"])
    pointer_path = evidence.verified_pointer_relative_path.as_posix()
    independent_path = evidence.independent_check_relative_path.as_posix()
    manifest_path = evidence.primary_manifest_relative_path.as_posix()
    quality = evidence.quality_relative_path.as_posix()
    verification_sql = f"""
WITH pointer AS (
    SELECT * FROM read_json_auto({_sql_literal(pointer_path)})
),
independent AS (
    SELECT * FROM read_json_auto({_sql_literal(independent_path)})
),
primary_manifest AS (
    SELECT * FROM read_json_auto({_sql_literal(manifest_path)})
)
SELECT
    pointer.build_id,
    pointer.verification_id,
    pointer.status,
    independent.checks.check_count AS check_count,
    independent.checks.pass_count AS pass_count,
    independent.checks.fail_count AS fail_count,
    primary_manifest.summary.raw_rows AS raw_rows,
    primary_manifest.summary.economic_features AS economic_features,
    primary_manifest.summary.lineage_rows AS lineage_rows
FROM pointer
CROSS JOIN independent
CROSS JOIN primary_manifest
""".strip()
    summary_path = f"{quality}/quality_summary.json"
    coverage_path = f"{quality}/feature_coverage.parquet"
    timing_path = f"{quality}/feature_timing_checks.parquet"
    filters = [
        f"build_id = {build_id}",
        f"verification_id = {verification_id}",
        f"status = {_VERIFIED_STATUS}",
        "development-only Stage 6 scope",
    ]
    return [
        _source_query(
            source_id="verification_evidence",
            label="Verified Stage 6 pointer, independent check, and primary manifest",
            sql=verification_sql,
            tables=(pointer_path, independent_path, manifest_path),
            description=(
                "Binds the final report to one verified pointer, its exact "
                "independent check, and its authenticated primary manifest."
            ),
            filters=filters,
        ),
        _source_query(
            source_id="quality_summary",
            label="Authenticated Stage 6 quality summary",
            sql=(
                "SELECT *\n"
                f"FROM read_json_auto({_sql_literal(summary_path)})\n"
                f"WHERE build_id = {_sql_literal(build_id)}"
            ),
            tables=(summary_path,),
            description=(
                "Reads the canonical primary quality summary whose receipt "
                "was authenticated before report generation."
            ),
            filters=filters,
        ),
        _source_query(
            source_id="coverage_by_family_year",
            label="Authenticated feature coverage by family and year",
            sql=f"""
SELECT
    feature_family,
    signal_year,
    COUNT(DISTINCT feature_name)::BIGINT AS feature_count,
    SUM(total_rows)::BIGINT AS total_rows,
    SUM(present_rows)::BIGINT AS present_rows,
    SUM(missing_rows)::BIGINT AS missing_rows,
    SUM(present_rows)::DOUBLE / SUM(total_rows) AS coverage_rate
FROM read_parquet({_sql_literal(coverage_path)})
GROUP BY feature_family, signal_year
ORDER BY signal_year, feature_family
""".strip(),
            tables=(coverage_path,),
            description=(
                "Aggregates authenticated feature-level coverage to the "
                "family-year grain using the full denominators."
            ),
            filters=filters,
            metric_definitions=(
                "coverage_rate = SUM(present_rows) / SUM(total_rows) within each feature family and signal year.",
                "The denominator is feature-row opportunities; it is not a distinct-stock count.",
                "Missing rows remain in the denominator and are not converted to zero.",
            ),
        ),
        _source_query(
            source_id="exact_quality_gates",
            label="Authenticated exact timing and integrity gates",
            sql=(
                "SELECT check_name, observed_count, expected_count, passed, "
                "severity, details\n"
                f"FROM read_parquet({_sql_literal(timing_path)})\n"
                "ORDER BY check_name"
            ),
            tables=(timing_path,),
            description=(
                "Reads every exact Stage 6 timing and structural-integrity "
                "gate from the authenticated primary quality bundle."
            ),
            filters=filters,
            metric_definitions=(
                "A gate passes only when passed is true and observed_count equals expected_count.",
            ),
        ),
    ]


def _artifact_payload(
    evidence: _AuthenticatedEvidence,
    coverage_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    build_id = str(evidence.pointer["build_id"])
    verification_id = str(evidence.pointer["verification_id"])
    checks = evidence.independent["checks"]
    counts = evidence.summary["counts"]
    families = len({str(row["feature_family"]) for row in coverage_rows})
    years = len({int(row["signal_year"]) for row in coverage_rows})
    hard_count = sum(row["gate_class"] == "frozen_policy" for row in gate_rows)
    sources = _sources(evidence)
    title = "Stage 6 Feature Store: Verified Quality Audit"
    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "layout": "full",
            "body": f"# {title}",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "layout": "full",
            "sourceId": "verification_evidence",
            "body": (
                "## Independent verification passed\n\n"
                f"Build `{build_id}` is bound to verification "
                f"`{verification_id}`. All **{checks['check_count']:,} of "
                f"{checks['check_count']:,}** independent checks passed, with "
                "zero reported failures and a canonical empty differences "
                "payload. This authorizes Stage 7 label construction only; "
                "models, portfolios, P&L, backtests, and later-period access "
                "remain unauthorized."
            ),
        },
        {
            "id": "coverage_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "coverage_by_family_year",
            "body": (
                "## Coverage is descriptive and visibly heterogeneous\n\n"
                f"The authenticated coverage evidence spans **{families:,} "
                f"feature families** and **{years:,} signal years**. Each "
                "point below is weighted by feature-row opportunities: "
                "`sum(present_rows) / sum(total_rows)`. It is not an average "
                "of feature percentages, a distinct-stock coverage rate, or "
                "evidence that a feature is economically useful.\n\n"
                "**Chart codes:** AGG aggregate state; OPT options; OWN "
                "ownership/filings/events/segments; PRC price trends; RSK "
                "risk/return; LIQ size/liquidity/frictions; VAL "
                "valuation/fundamentals."
            ),
        },
        {
            "id": "coverage_visual",
            "type": "chart",
            "layout": "full",
            "chartId": "coverage_chart",
        },
        {
            "id": "gate_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "exact_quality_gates",
            "body": (
                "## Every published timing and integrity gate passed\n\n"
                f"The exact table preserves **{len(gate_rows):,} checks**, "
                f"including **{hard_count:,} frozen policy gates**, without "
                "collapsing observed and expected values into a summary flag."
            ),
        },
        {
            "id": "gate_table_block",
            "type": "table",
            "layout": "full",
            "tableId": "gate_table",
        },
        {
            "id": "scope_and_definitions",
            "type": "markdown",
            "layout": "full",
            "sourceId": "quality_summary",
            "body": (
                "## Scope, data, and metric definitions\n\n"
                f"The development feature store contains **{counts['raw_rows']:,} "
                f"security-signal rows**, **{counts['raw_feature_columns']:,} "
                f"economic feature columns**, and **{counts['lineage_rows']:,} "
                "logical lineage rows. `PRESENT` contributes to the coverage "
                "numerator; every explicit missingness status remains in the "
                "denominator. Coverage and outlier summaries are descriptive "
                "QA, while timing, leakage, nonfinite-value, schema, key, and "
                "lineage conditions are gates."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "layout": "full",
            "sourceId": "coverage_by_family_year",
            "body": (
                "## Methodology preserves the audited denominators\n\n"
                "The generator first authenticates the verified pointer, "
                "independent check, primary manifest, empty differences "
                "payload, closed quality file set, and every file receipt. It "
                "then aggregates the authenticated feature-level coverage "
                "rows by family and signal year using summed numerators and "
                "denominators. No imputation, winsorization, model fitting, "
                "ranking, or performance calculation occurs in this report."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Limitations and robustness boundary\n\n"
                "Verification establishes conformance to the frozen Stage 6 "
                "data and timing contract. It does not establish predictor "
                "skill, causal validity, investability after costs, or "
                "portfolio performance. Family-level coverage can conceal "
                "feature-level variation, and structurally unavailable "
                "sources remain missing by design. Those limitations should "
                "be handled explicitly in later preprocessing and robustness "
                "tests, not reinterpreted as zero observations."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Recommended next step\n\n"
                "Proceed only to roadmap Stage 7: construct the frozen "
                "FF5-plus-momentum residual-return labels and independently "
                "verify holding-interval alignment, residualization timing, "
                "purging, embargoes, and leakage controls. Do not fit a model "
                "or run a portfolio backtest from this report."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Further questions\n\n"
                "- Do Stage 7 labels begin strictly after each executable entry date?\n"
                "- Are factor estimates used for residualization available at the required time?\n"
                "- Which structural missingness indicators must remain explicit in Stage 9 preprocessing?\n"
                "- Can the label verifier reproduce every target without importing the primary label builder?"
            ),
        },
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Independent-verification-bound Stage 6 feature-store quality "
            "evidence and authorization boundary."
        ),
        "filters": [],
        "cards": [],
        "charts": [
            {
                "id": "coverage_chart",
                "dataset": "coverage_by_family_year",
                "type": "line",
                "intent": "trend",
                "title": "Feature coverage by family and signal year",
                "subtitle": (
                    "Present feature-row opportunities divided by all "
                    "feature-row opportunities; missing rows stay in scope"
                ),
                "question": (
                    "How does observed feature availability vary across "
                    "families and development years?"
                ),
                "rationale": (
                    "A grouped line chart preserves the time path for every "
                    "family while exact numerators and denominators remain in "
                    "the source dataset."
                ),
                "sourceId": "coverage_by_family_year",
                "valueFormat": "percent",
                "showDescription": True,
                "layout": "full",
                "maxRows": len(coverage_rows),
                "encodings": {
                    "x": {
                        "field": "signal_year",
                        "type": "ordinal",
                        "label": "Signal year",
                    },
                    "y": {
                        "field": "coverage_rate",
                        "type": "quantitative",
                        "label": "Coverage rate",
                        "format": "percent",
                    },
                    "color": {
                        "field": "feature_family_code",
                        "type": "nominal",
                        "label": "Feature-family code",
                    },
                    "tooltip": [
                        {
                            "field": "present_rows",
                            "type": "quantitative",
                            "label": "Present feature-rows",
                            "format": "number",
                        },
                        {
                            "field": "missing_rows",
                            "type": "quantitative",
                            "label": "Missing feature-rows",
                            "format": "number",
                        },
                        {
                            "field": "total_rows",
                            "type": "quantitative",
                            "label": "Feature-row opportunities",
                            "format": "number",
                        },
                        {
                            "field": "feature_count",
                            "type": "quantitative",
                            "label": "Features",
                            "format": "number",
                        },
                    ],
                },
            }
        ],
        "tables": [
            {
                "id": "gate_table",
                "dataset": "exact_quality_gates",
                "title": "Exact Stage 6 timing and integrity gates",
                "subtitle": (
                    "Authenticated observed and expected counts; all rows "
                    "must pass before independent verification"
                ),
                "sourceId": "exact_quality_gates",
                "layout": "full",
                "density": "comfortable",
                "showDescription": True,
                "defaultSort": {"field": "check_name", "direction": "asc"},
                "columns": [
                    {
                        "field": "check_name",
                        "label": "Check",
                        "type": "text",
                    },
                    {
                        "field": "gate_class",
                        "label": "Class",
                        "type": "text",
                    },
                    {
                        "field": "observed_count",
                        "label": "Observed",
                        "type": "number",
                        "format": "number",
                    },
                    {
                        "field": "expected_count",
                        "label": "Expected",
                        "type": "number",
                        "format": "number",
                    },
                    {
                        "field": "result",
                        "label": "Result",
                        "type": "text",
                    },
                    {
                        "field": "severity",
                        "label": "Severity",
                        "type": "text",
                    },
                    {
                        "field": "details",
                        "label": "Evidence note",
                        "type": "text",
                    },
                ],
            }
        ],
        "sources": sources,
        "blocks": blocks,
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "datasets": {
                "coverage_by_family_year": list(coverage_rows),
                "exact_quality_gates": list(gate_rows),
            },
        },
        "sources": sources,
    }


def _notebook(
    *,
    evidence: _AuthenticatedEvidence,
    artifact_relative_path: PurePosixPath,
    coverage_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
) -> nbf.NotebookNode:
    build_id = str(evidence.pointer["build_id"])
    verification_id = str(evidence.pointer["verification_id"])
    checks = evidence.independent["checks"]
    counts = evidence.summary["counts"]
    cells: list[nbf.NotebookNode] = []

    def markdown(cell_id: str, source: str) -> None:
        cell = nbf.v4.new_markdown_cell(source)
        cell["id"] = cell_id
        cells.append(cell)

    def code(cell_id: str, source: str) -> None:
        cell = nbf.v4.new_code_cell(source)
        cell["id"] = cell_id
        cells.append(cell)

    markdown(
        "title",
        "# Stage 6 Feature Store Quality Audit\n\n"
        "This notebook reviews the bounded, verification-bound snapshot in "
        f"`{artifact_relative_path.as_posix()}`. It contains no feature-build "
        "logic and no model or backtest logic.",
    )
    markdown(
        "tldr",
        "## tl;dr\n\n"
        f"Build `{build_id}` is bound to verification `{verification_id}`. "
        f"All **{checks['check_count']:,}** independent checks and all "
        f"**{len(gate_rows):,}** published timing/integrity gates passed. "
        "This opens Stage 7 labels only; it is not evidence of ML or portfolio "
        "performance.",
    )
    markdown(
        "context_methods",
        "## Context & Methods\n\n"
        "The report generator authenticated the verified pointer, independent "
        "check, canonical empty differences payload, primary manifest, closed "
        "quality file set, and every quality receipt before creating the "
        "snapshot. Coverage is aggregated as "
        "`sum(present_rows) / sum(total_rows)` within feature family and signal "
        "year. Missing observations stay in the denominator.",
    )
    code(
        "load_artifact",
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n"
        "from IPython.display import display\n\n"
        f"ARTIFACT_RELATIVE_PATH = Path({artifact_relative_path.as_posix()!r})\n"
        "search = Path.cwd().resolve()\n"
        "while search != search.parent and not (search / ARTIFACT_RELATIVE_PATH).is_file():\n"
        "    search = search.parent\n"
        "ARTIFACT_PATH = search / ARTIFACT_RELATIVE_PATH\n"
        "if not ARTIFACT_PATH.is_file():\n"
        "    raise FileNotFoundError(f'Cannot locate {ARTIFACT_RELATIVE_PATH} from {Path.cwd()}')\n"
        "artifact = json.loads(ARTIFACT_PATH.read_text(encoding='utf-8'))\n"
        "assert artifact['surface'] == 'report'\n"
        "assert artifact['snapshot']['status'] == 'ready'\n"
        "coverage = pd.DataFrame(artifact['snapshot']['datasets']['coverage_by_family_year'])\n"
        "gates = pd.DataFrame(artifact['snapshot']['datasets']['exact_quality_gates'])\n"
        f"BUILD_ID = {build_id!r}\n"
        f"VERIFICATION_ID = {verification_id!r}\n"
        "print({'build_id': BUILD_ID, 'verification_id': VERIFICATION_ID, "
        "'artifact': ARTIFACT_RELATIVE_PATH.as_posix()})",
    )
    markdown(
        "data",
        "## Data\n\n"
        f"The authenticated store has **{counts['raw_rows']:,} "
        f"security-signal rows**, **{counts['raw_feature_columns']:,} economic "
        f"features**, and **{counts['lineage_rows']:,} logical lineage rows. "
        "The notebook uses only the bounded reviewed rows embedded in the "
        "portable artifact; the source queries and exact safe relative "
        "provenance are preserved in `artifact['sources']`.",
    )
    code(
        "inspect_data",
        "assert not coverage.empty and not gates.empty\n"
        "assert coverage['coverage_rate'].between(0.0, 1.0).all()\n"
        "assert (coverage['present_rows'] + coverage['missing_rows'] == coverage['total_rows']).all()\n"
        "binding = pd.DataFrame([{\n"
        "    'build_id': BUILD_ID,\n"
        "    'verification_id': VERIFICATION_ID,\n"
        "    'coverage_rows': len(coverage),\n"
        "    'gate_rows': len(gates),\n"
        "}])\n"
        "display(binding)",
    )
    markdown(
        "results",
        "## Results\n\n"
        "The first table is the grouped coverage view used by the report's "
        "native line chart. The second is the exact gate table. Coverage is "
        "descriptive; a low or high rate is not a performance conclusion.",
    )
    code(
        "coverage_result",
        "coverage_view = coverage.sort_values(['signal_year', 'feature_family']).copy()\n"
        "coverage_view['coverage_pct'] = coverage_view['coverage_rate'].map(lambda x: f'{x:.2%}')\n"
        "display(coverage_view[[\n"
        "    'signal_year', 'feature_family', 'feature_count', 'present_rows',\n"
        "    'missing_rows', 'total_rows', 'coverage_pct'\n"
        "]])",
    )
    code(
        "gate_result",
        "assert gates['passed'].all()\n"
        "assert (gates['observed_count'] == gates['expected_count']).all()\n"
        "display(gates[[\n"
        "    'check_name', 'gate_class', 'observed_count', 'expected_count',\n"
        "    'result', 'severity', 'details'\n"
        "]].sort_values('check_name'))",
    )
    markdown(
        "takeaways",
        "## Takeaways\n\n"
        "1. Stage 6 quality evidence is internally consistent with its final "
        "independent verification binding.\n"
        "2. Missingness and coverage remain explicit and descriptive; they "
        "were not silently converted to zero or used as performance evidence.\n"
        "3. The next authorized action is Stage 7 label construction and "
        "independent label verification—not model training or backtesting.",
    )
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "stage_06_binding": {
                "build_id": build_id,
                "verification_id": verification_id,
                "artifact_path": artifact_relative_path.as_posix(),
            },
        },
    )
    return notebook


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.parent.is_symlink(), f"Output parent is a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_notebook(path: Path, notebook: nbf.NotebookNode) -> None:
    payload = nbf.writes(notebook, version=4).encode("utf-8")
    _atomic_write(path, payload)


def _execute_notebook(
    notebook: nbf.NotebookNode, *, project_root: Path
) -> nbf.NotebookNode:
    try:
        from nbclient import NotebookClient
    except ImportError as exc:
        raise Stage6AuditArtifactError(
            "Notebook execution requires the research dependency nbclient"
        ) from exc
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": project_root.as_posix()}},
    )
    return client.execute()


def generate_stage6_audit_artifacts(
    *,
    verified_pointer_path: str | Path,
    independent_check_path: str | Path,
    primary_manifest_path: str | Path,
    quality_directory: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    database_root: str | Path | None = None,
    policy_path: str | Path = DEFAULT_POLICY,
    notebook_path: str | Path | None = None,
    report_directory: str | Path | None = None,
    execute_notebook: bool = True,
) -> Stage6AuditArtifactReceipt:
    """Generate the Stage 6 audit notebook and canonical report artifact.

    All evidence is authenticated before either output is written.
    """

    project = _real_directory(project_root, "project root")
    notebook_output = Path(
        notebook_path
        if notebook_path is not None
        else project / "notebooks/06_stage_06_feature_store_quality.ipynb"
    ).expanduser().resolve()
    report_dir = Path(
        report_directory
        if report_directory is not None
        else project / "reports/stage_06_feature_store"
    ).expanduser().resolve()
    _safe_relative(notebook_output, project, "notebook output")
    _safe_relative(report_dir, project, "report output")
    artifact_path = report_dir / "artifact.json"
    artifact_relative = _safe_relative(artifact_path, project, "report artifact")

    evidence = _authenticate_evidence(
        verified_pointer_path=verified_pointer_path,
        independent_check_path=independent_check_path,
        primary_manifest_path=primary_manifest_path,
        quality_directory=quality_directory,
        project_root=project,
        database_root=database_root,
        policy_path=policy_path,
    )
    coverage_rows = _coverage_rows(evidence)
    gate_rows = _gate_rows(evidence)
    artifact = _artifact_payload(evidence, coverage_rows, gate_rows)
    notebook = _notebook(
        evidence=evidence,
        artifact_relative_path=artifact_relative,
        coverage_rows=coverage_rows,
        gate_rows=gate_rows,
    )

    _atomic_write(artifact_path, canonical_json_bytes(artifact))
    if execute_notebook:
        notebook = _execute_notebook(notebook, project_root=project)
    _write_notebook(notebook_output, notebook)
    return Stage6AuditArtifactReceipt(
        build_id=str(evidence.pointer["build_id"]),
        verification_id=str(evidence.pointer["verification_id"]),
        notebook_path=notebook_output,
        artifact_path=artifact_path,
        notebook_sha256=_sha256_file(notebook_output),
        artifact_sha256=_sha256_file(artifact_path),
        coverage_rows=len(coverage_rows),
        gate_rows=len(gate_rows),
        notebook_executed=execute_notebook,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the verification-bound Stage 6 quality notebook and "
            "portable report artifact. No HTML is generated."
        )
    )
    parser.add_argument("--verified-pointer", required=True, type=Path)
    parser.add_argument("--independent-check", required=True, type=Path)
    parser.add_argument("--primary-manifest", required=True, type=Path)
    parser.add_argument("--quality-directory", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--report-directory", type=Path)
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Write a runnable notebook without executing it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = generate_stage6_audit_artifacts(
        verified_pointer_path=args.verified_pointer,
        independent_check_path=args.independent_check,
        primary_manifest_path=args.primary_manifest,
        quality_directory=args.quality_directory,
        project_root=args.project_root,
        database_root=args.database_root,
        policy_path=args.policy,
        notebook_path=args.notebook,
        report_directory=args.report_directory,
        execute_notebook=not args.no_execute,
    )
    print(json.dumps(receipt.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
