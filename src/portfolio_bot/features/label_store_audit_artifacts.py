"""Verification-bound Stage 7 label-store audit artifacts.

This module is deliberately downstream of the independent Stage 7 verifier.
It does not build labels, recompute targets, fit a model, construct a portfolio,
or calculate performance.  It authenticates one final verified label-store
graph and produces only:

* ``notebooks/07_stage_07_label_store_quality.ipynb``; and
* ``reports/stage_07_label_store/artifact.json``.

HTML rendering is intentionally a separate, explicit delivery step.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import nbformat as nbf
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.features.label_store_contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
)
from portfolio_bot.features.phase_03_contract import canonical_json_bytes


DEFAULT_NOTEBOOK = PROJECT_ROOT / "notebooks/07_stage_07_label_store_quality.ipynb"
DEFAULT_REPORT_DIRECTORY = PROJECT_ROOT / "reports/stage_07_label_store"
REVIEWED_POLICY_SHA256 = (
    "1f6f5649cfd1959a707c6e04465b70a334b1fbc4b32e9b81158baf521c2abcad"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BUILD_ID = re.compile(r"phase03-labels-[0-9a-f]{24}\Z")
_VERIFICATION_ID = re.compile(r"verify-[0-9a-f]{16}\Z")
_HOLDING_MONTH = re.compile(r"(20[0-9]{2})-(0[1-9]|1[0-2])\Z")
_VERIFIED_STATUS = "VERIFIED_PASSED"
_PRIMARY_STATUS = "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"

_VERIFIED_AUTHORIZATIONS = {
    "independent_verification_completed": True,
    "models_portfolios_pnl_or_backtests_authorized": False,
    "stage_08_risk_and_cost_models_authorized": True,
    "stage_09_model_training_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
_PRIMARY_AUTHORIZATIONS = {
    "independent_verification_required": True,
    "models_portfolios_pnl_or_backtests_authorized": False,
    "stage_08_risk_and_cost_models_authorized": False,
    "stage_09_model_training_authorized": False,
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
    {
        "check_count",
        "pass_count",
        "fail_count",
        "results_sha256",
        "verification_code_hashes",
        "results",
    }
)
_VERIFICATION_RESULT_KEYS = frozenset(
    {"artifact_authentication", "recomputation", "verification_code_hashes"}
)
_ARTIFACT_AUTHENTICATION_KEYS = frozenset(
    {
        "artifact_graph_authenticated",
        "build_identity_matches",
        "closed_output_sets_authenticated",
        "input_tables_authenticated",
        "output_files_authenticated",
        "primary_manifest_sha256",
    }
)
_RECOMPUTATION_KEYS = frozenset({"hard_gates", "metrics"})
_RECOMPUTATION_METRIC_KEYS = frozenset(
    {
        "beta_complete_rows",
        "lineage_rows_recomputed",
        "long_rows_recomputed",
        "return_rows_delivered_beta_pass",
        "return_rows_delivered_forward_pass",
        "signal_dates_recomputed",
    }
)
_VERIFICATION_CODE_PATHS = frozenset(
    {
        "scripts/verify_stage_07_label_store.py",
        "src/portfolio_bot/features/stage7_independent_math.py",
        "src/portfolio_bot/features/stage7_independent_recompute.py",
        "src/portfolio_bot/features/verify_label_store.py",
    }
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
_OUTPUT_KEYS = frozenset({"path", "success_sha256", "files"})
_FILE_RECEIPT_KEYS = frozenset(
    {"path", "sha256", "size_bytes", "rows", "row_groups", "schema_sha256"}
)
_SUCCESS_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "location_kind",
        "status",
        "bindings",
        "table_receipts",
    }
)
_QUALITY_COUNT_TO_FILE = {
    "coverage_rows": "label_coverage.parquet",
    "status_count_rows": "label_status_counts.parquet",
    "timing_check_rows": "label_timing_checks.parquet",
    "distribution_rows": "label_distribution.parquet",
}
_DIMENSION_DOMAIN = {
    "outcome": "outcome_status",
    "primary": "primary_label_status",
    "magnitude": "magnitude_label_status",
}
_COVERAGE_FIELDS = {
    "outcome": "outcome_present_count",
    "primary": "primary_present_count",
    "magnitude": "magnitude_present_count",
}
_COVERAGE_LABELS = {
    "outcome": "Outcome present",
    "primary": "Primary label present",
    "magnitude": "Magnitude label present",
}


class Stage7AuditArtifactError(RuntimeError):
    """Raised when Stage 7 audit evidence is unsafe or inconsistent."""


@dataclass(frozen=True)
class Stage7AuditArtifactReceipt:
    """Machine-readable receipt for the generated Stage 7 audit artifacts."""

    build_id: str
    verification_id: str
    notebook_path: Path
    artifact_path: Path
    notebook_sha256: str
    artifact_sha256: str
    coverage_rows: int
    status_rows: int
    outcome_missing_rows: int
    delayed_rows: int
    distribution_rows: int
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
            "status_rows": self.status_rows,
            "outcome_missing_rows": self.outcome_missing_rows,
            "delayed_rows": self.delayed_rows,
            "distribution_rows": self.distribution_rows,
            "gate_rows": self.gate_rows,
            "notebook_executed": self.notebook_executed,
        }


@dataclass(frozen=True)
class _AuthenticatedEvidence:
    policy: Mapping[str, Any]
    rules: Mapping[str, Any]
    pointer: Mapping[str, Any]
    independent: Mapping[str, Any]
    manifest: Mapping[str, Any]
    summary: Mapping[str, Any]
    coverage: pa.Table
    statuses: pa.Table
    distributions: pa.Table
    timing: pa.Table
    month_summary: pa.Table
    project_root: Path
    database_root: Path
    quality_relative_path: PurePosixPath
    processed_relative_path: PurePosixPath
    verified_pointer_relative_path: PurePosixPath
    independent_check_relative_path: PurePosixPath
    differences_relative_path: PurePosixPath
    primary_manifest_relative_path: PurePosixPath


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage7AuditArtifactError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _real_directory(path: str | Path, label: str) -> Path:
    lexical = Path(path).expanduser().absolute()
    _require(not lexical.is_symlink(), f"Symlink prohibited for {label}")
    resolved = lexical.resolve()
    _require(resolved.is_dir(), f"Missing {label}: {resolved}")
    return resolved


def _anchored_path(
    path: str | Path,
    *,
    root: Path,
    label: str,
    directory: bool,
) -> Path:
    lexical = Path(path).expanduser().absolute()
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise Stage7AuditArtifactError(f"{label} escaped its trusted root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        _require(not current.is_symlink(), f"Symlink prohibited for {label}")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Stage7AuditArtifactError(f"{label} resolved outside its root") from exc
    _require(
        resolved.is_dir() if directory else resolved.is_file(),
        f"Missing {label}: {resolved}",
    )
    return resolved


def _safe_relative(path: Path, root: Path, label: str) -> PurePosixPath:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise Stage7AuditArtifactError(
            f"{label} must remain below the project root"
        ) from exc
    result = PurePosixPath(relative.as_posix())
    _require(
        not result.is_absolute() and ".." not in result.parts,
        f"Unsafe relative provenance for {label}",
    )
    return result


def _safe_database_relative(value: Any, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and bool(value), f"Missing {label} path")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute() and ".." not in path.parts and "." not in path.parts,
        f"Unsafe {label} path",
    )
    return path


def _stable_bytes(path: Path, label: str) -> bytes:
    before = path.stat()
    _require(not path.is_symlink(), f"Symlink prohibited for {label}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise Stage7AuditArtifactError(f"Cannot read {label}: {path}") from exc
    after = path.stat()
    _require(
        _stat_identity(before) == _stat_identity(after),
        f"{label} changed while being read",
    )
    return payload


def _json_object(
    path: Path,
    label: str,
    *,
    canonical: bool = True,
) -> dict[str, Any]:
    payload = _stable_bytes(path, label)
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7AuditArtifactError(f"Invalid JSON for {label}") from exc
    _require(isinstance(value, dict), f"Expected JSON object for {label}")
    if canonical:
        _require(
            payload == canonical_json_bytes(value),
            f"{label} is not canonical JSON",
        )
    return value


def _toml_object(path: Path, label: str) -> dict[str, Any]:
    payload = _stable_bytes(path, label)
    try:
        value = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise Stage7AuditArtifactError(f"Invalid TOML for {label}") from exc
    _require(isinstance(value, dict), f"Expected TOML table for {label}")
    return value


def _validate_declared_receipt(name: str, receipt: Mapping[str, Any]) -> None:
    _require(
        set(receipt) == _FILE_RECEIPT_KEYS,
        f"File receipt keys differ: {name}",
    )
    _require(receipt["path"] == name, f"File receipt path differs: {name}")
    _require(
        _SHA256.fullmatch(str(receipt["sha256"])) is not None,
        f"Invalid file receipt SHA-256: {name}",
    )
    _require(
        isinstance(receipt["size_bytes"], int)
        and not isinstance(receipt["size_bytes"], bool)
        and receipt["size_bytes"] >= 0,
        f"Invalid file receipt size: {name}",
    )
    if name.endswith(".parquet"):
        _require(
            isinstance(receipt["rows"], int)
            and not isinstance(receipt["rows"], bool)
            and receipt["rows"] >= 0
            and isinstance(receipt["row_groups"], int)
            and not isinstance(receipt["row_groups"], bool)
            and receipt["row_groups"] >= 0
            and _SHA256.fullmatch(str(receipt["schema_sha256"])) is not None,
            f"Invalid Parquet receipt metadata: {name}",
        )
    else:
        _require(
            receipt["rows"] is None
            and receipt["row_groups"] is None
            and receipt["schema_sha256"] is None,
            f"Non-Parquet receipt contains table metadata: {name}",
        )


def _authenticate_file_receipt(
    path: Path,
    *,
    name: str,
    receipt: Mapping[str, Any],
) -> None:
    _validate_declared_receipt(name, receipt)
    payload = _stable_bytes(path, name)
    _require(len(payload) == receipt["size_bytes"], f"File size differs: {name}")
    _require(_sha256_bytes(payload) == receipt["sha256"], f"File hash differs: {name}")
    if path.suffix == ".parquet":
        before = _stat_identity(path.stat())
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        groups = int(parquet.metadata.num_row_groups)
        schema_hash = _schema_sha256(parquet.schema_arrow)
        after = _stat_identity(path.stat())
        _require(before == after, f"Parquet metadata changed while read: {name}")
        _require(rows == receipt["rows"], f"Parquet row count differs: {name}")
        _require(groups == receipt["row_groups"], f"Parquet row groups differ: {name}")
        _require(
            schema_hash == receipt["schema_sha256"],
            f"Parquet schema hash differs: {name}",
        )


def _authenticated_table(
    path: Path,
    *,
    name: str,
    receipt: Mapping[str, Any],
) -> pa.Table:
    _authenticate_file_receipt(path, name=name, receipt=receipt)
    before = _stat_identity(path.stat())
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as exc:
        raise Stage7AuditArtifactError(f"Cannot read Parquet table: {name}") from exc
    after = _stat_identity(path.stat())
    _require(before == after, f"Parquet payload changed while read: {name}")
    _require(_sha256_file(path) == receipt["sha256"], f"Parquet rehash differs: {name}")
    return table


def _output_hashes(manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    outputs = manifest.get("outputs")
    _require(
        isinstance(outputs, Mapping)
        and set(outputs) == {"interim", "processed", "quality"},
        "Primary output sets differ",
    )
    result: dict[str, dict[str, str]] = {}
    for kind in sorted(outputs):
        specification = outputs[kind]
        _require(
            isinstance(specification, Mapping) and set(specification) == _OUTPUT_KEYS,
            f"Primary {kind} output receipt differs",
        )
        receipts = specification.get("files")
        _require(isinstance(receipts, Mapping), f"Missing {kind} file receipts")
        hashes: dict[str, str] = {}
        for name in sorted(receipts):
            receipt = receipts[name]
            _require(isinstance(receipt, Mapping), f"Invalid {kind}/{name} receipt")
            _validate_declared_receipt(name, receipt)
            hashes[name] = str(receipt["sha256"])
        _require(
            specification["success_sha256"] == hashes.get("_SUCCESS.json"),
            f"{kind} success binding differs",
        )
        result[kind] = hashes
    return result


def _validate_verification_checks(
    *,
    checks: Mapping[str, Any],
    project_root: Path,
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    primary_manifest_sha256: str,
) -> None:
    _require(set(checks) == _CHECK_KEYS, "Independent check summary differs")
    results = checks.get("results")
    code_hashes = checks.get("verification_code_hashes")
    _require(
        isinstance(results, Mapping)
        and set(results) == _VERIFICATION_RESULT_KEYS
        and isinstance(code_hashes, Mapping)
        and set(code_hashes) == _VERIFICATION_CODE_PATHS,
        "Independent verification results or code-hash domain differs",
    )
    _require(
        isinstance(checks["check_count"], int)
        and not isinstance(checks["check_count"], bool)
        and checks["check_count"] == len(results) == 3
        and checks["pass_count"] == checks["check_count"]
        and checks["fail_count"] == 0,
        "Independent verification did not pass every reported check",
    )
    _require(
        _SHA256.fullmatch(str(checks["results_sha256"])) is not None
        and checks["results_sha256"]
        == _sha256_bytes(canonical_json_bytes(dict(results))),
        "Independent verification results digest differs",
    )
    embedded_code_hashes = results.get("verification_code_hashes")
    _require(
        isinstance(embedded_code_hashes, Mapping)
        and dict(embedded_code_hashes) == dict(code_hashes),
        "Embedded verification code hashes differ",
    )
    for relative_path in sorted(_VERIFICATION_CODE_PATHS):
        expected_hash = code_hashes[relative_path]
        _require(
            isinstance(expected_hash, str)
            and _SHA256.fullmatch(expected_hash) is not None,
            f"Invalid verification code hash: {relative_path}",
        )
        code_path = _anchored_path(
            project_root / relative_path,
            root=project_root,
            label=f"verification code {relative_path}",
            directory=False,
        )
        _require(
            _sha256_file(code_path) == expected_hash,
            f"Verification code hash differs: {relative_path}",
        )

    artifact = results.get("artifact_authentication")
    _require(
        isinstance(artifact, Mapping)
        and set(artifact) == _ARTIFACT_AUTHENTICATION_KEYS,
        "Artifact-authentication result differs",
    )
    expected_input_count = len(manifest.get("inputs", {}).get("table_receipts", ()))
    expected_output_count = sum(
        len(specification["files"]) for specification in manifest["outputs"].values()
    )
    _require(
        artifact["artifact_graph_authenticated"] is True
        and artifact["build_identity_matches"] is True
        and artifact["closed_output_sets_authenticated"] == len(manifest["outputs"])
        and artifact["input_tables_authenticated"] == expected_input_count
        and artifact["output_files_authenticated"] == expected_output_count
        and artifact["primary_manifest_sha256"] == primary_manifest_sha256,
        "Artifact-authentication result is not bound to the primary manifest",
    )

    recomputation = results.get("recomputation")
    _require(
        isinstance(recomputation, Mapping)
        and set(recomputation) == _RECOMPUTATION_KEYS,
        "Independent recomputation result differs",
    )
    expected_hard_gates = {
        key: int(value)
        for key, value in policy["quality_gates"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    observed_hard_gates = recomputation.get("hard_gates")
    _require(
        isinstance(observed_hard_gates, Mapping)
        and len(observed_hard_gates) == 57
        and dict(observed_hard_gates) == expected_hard_gates,
        "Independent recomputation hard gates differ",
    )
    metrics = recomputation.get("metrics")
    _require(
        isinstance(metrics, Mapping)
        and set(metrics) == _RECOMPUTATION_METRIC_KEYS
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in metrics.values()
        ),
        "Independent recomputation metrics differ",
    )
    summary = manifest["summary"]
    _require(
        metrics["beta_complete_rows"] == summary["processed_rows"]
        and metrics["long_rows_recomputed"] == summary["processed_rows"]
        and metrics["lineage_rows_recomputed"] == summary["leg_lineage_rows"]
        and metrics["signal_dates_recomputed"] == summary["signal_dates"]
        and metrics["return_rows_delivered_beta_pass"]
        == metrics["return_rows_delivered_forward_pass"]
        and metrics["return_rows_delivered_beta_pass"] > 0,
        "Independent recomputation metrics disagree with the primary summary",
    )


def _authenticate_output_directory(
    *,
    evidence_kind: str,
    output: Mapping[str, Any],
    policy: Mapping[str, Any],
    database_root: Path,
    build_id: str,
    supplied_directory: str | Path | None,
    authenticate_names: Sequence[str],
) -> tuple[Path, PurePosixPath, Mapping[str, Any]]:
    relative = _safe_database_relative(output["path"], f"{evidence_kind} output")
    root_key = f"{evidence_kind}_build_root"
    expected_relative = PurePosixPath(
        str(policy["database"][root_key]) + "/" + build_id
    )
    _require(relative == expected_relative, f"{evidence_kind} output path differs")
    expected_path = database_root / Path(relative.as_posix())
    directory = _anchored_path(
        supplied_directory if supplied_directory is not None else expected_path,
        root=database_root,
        label=f"Stage 7 {evidence_kind} directory",
        directory=True,
    )
    _require(directory == expected_path, f"Supplied {evidence_kind} directory differs")
    expected_names = set(policy["outputs"][evidence_kind]["required_files"])
    receipts = output["files"]
    _require(
        isinstance(receipts, Mapping) and set(receipts) == expected_names,
        f"{evidence_kind} output file set is not closed",
    )
    observed = {child.name for child in directory.iterdir()}
    _require(
        observed == expected_names
        and all(not child.is_symlink() for child in directory.iterdir()),
        f"{evidence_kind} directory file set differs",
    )
    for name in authenticate_names:
        _require(
            name in expected_names, f"Unknown {evidence_kind} evidence file: {name}"
        )
        member = _anchored_path(
            directory / name,
            root=database_root,
            label=f"{evidence_kind}/{name}",
            directory=False,
        )
        receipt = receipts[name]
        _require(
            isinstance(receipt, Mapping), f"Invalid {evidence_kind}/{name} receipt"
        )
        _authenticate_file_receipt(member, name=name, receipt=receipt)
    return directory, relative, receipts


def _authenticate_success(
    *,
    path: Path,
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    kind: str,
    location_kind: str,
    status: str,
) -> None:
    _authenticate_file_receipt(path, name="_SUCCESS.json", receipt=receipt)
    payload = _json_object(path, f"{kind} success receipt")
    _require(set(payload) == _SUCCESS_KEYS, f"{kind} success keys differ")
    expected_tables = {
        name: value
        for name, value in manifest["outputs"][kind]["files"].items()
        if name != "_SUCCESS.json"
    }
    _require(
        payload["schema_version"] == 1
        and payload["contract_id"] == manifest["contract_id"]
        and payload["policy_id"] == manifest["policy_id"]
        and payload["build_id"] == manifest["build_id"]
        and payload["location_kind"] == location_kind
        and payload["status"] == status
        and payload["bindings"] == manifest["bindings"]
        and payload["table_receipts"] == expected_tables,
        f"{kind} success receipt binding differs",
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
    reviewed_policy_sha256: str,
    processed_directory: str | Path | None,
) -> _AuthenticatedEvidence:
    project = _real_directory(project_root, "project root")
    database = (
        _real_directory(
            database_root if database_root is not None else ".",
            "database root",
        )
        if database_root is not None
        else None
    )
    policy_file = Path(policy_path).expanduser().resolve()
    _require(
        policy_file.is_file() and not policy_file.is_symlink(), "Missing Stage 7 policy"
    )
    _require(
        _SHA256.fullmatch(reviewed_policy_sha256) is not None
        and _sha256_file(policy_file) == reviewed_policy_sha256,
        "Stage 7 policy hash differs from the reviewed policy",
    )
    policy = _toml_object(policy_file, "Stage 7 policy")
    if database is None:
        database = _real_directory(policy["database"]["root"], "database root")

    rules_relative = _safe_database_relative(
        policy["physical_rules"]["path"], "physical rules"
    )
    rules_path = _anchored_path(
        project / Path(rules_relative.as_posix()),
        root=project,
        label="Stage 7 physical rules",
        directory=False,
    )
    rules_bytes = _stable_bytes(rules_path, "Stage 7 physical rules")
    _require(
        _sha256_bytes(rules_bytes) == policy["physical_rules"]["sha256"],
        "Stage 7 physical-rules hash differs",
    )
    try:
        rules = json.loads(rules_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7AuditArtifactError("Invalid Stage 7 physical rules") from exc
    _require(
        isinstance(rules, Mapping)
        and rules.get("rules_id") == policy["physical_rules"]["required_rules_id"]
        and rules.get("contract_id") == policy["contract_id"],
        "Stage 7 physical-rules identity differs",
    )

    pointer_path = _anchored_path(
        verified_pointer_path,
        root=project,
        label="Stage 7 verified pointer",
        directory=False,
    )
    independent_path = _anchored_path(
        independent_check_path,
        root=project,
        label="Stage 7 independent check",
        directory=False,
    )
    manifest_path = _anchored_path(
        primary_manifest_path,
        root=project,
        label="Stage 7 primary manifest",
        directory=False,
    )
    expected_pointer = (
        project / policy["publication"]["latest_verified_pointer"]
    ).resolve()
    _require(
        pointer_path == expected_pointer, "Verified pointer path differs from policy"
    )

    pointer = _json_object(pointer_path, "Stage 7 verified pointer")
    independent = _json_object(independent_path, "Stage 7 independent check")
    manifest = _json_object(manifest_path, "Stage 7 primary manifest")
    _require(set(pointer) == _VERIFIED_POINTER_KEYS, "Verified pointer keys differ")
    _require(
        set(independent) == _INDEPENDENT_CHECK_KEYS,
        "Independent-check keys differ",
    )
    _require(set(manifest) == _PRIMARY_MANIFEST_KEYS, "Primary-manifest keys differ")

    build_id = pointer.get("build_id")
    verification_id = pointer.get("verification_id")
    _require(
        isinstance(build_id, str) and _BUILD_ID.fullmatch(build_id) is not None,
        "Invalid Stage 7 build ID",
    )
    _require(
        isinstance(verification_id, str)
        and _VERIFICATION_ID.fullmatch(verification_id) is not None,
        "Invalid Stage 7 verification ID",
    )
    expected_independent = (
        project
        / policy["publication"]["artifact_root"]
        / policy["publication"]["verification_namespace"]
        / build_id
        / verification_id
        / "independent_check.json"
    ).resolve()
    expected_manifest = (
        project
        / policy["publication"]["artifact_root"]
        / policy["publication"]["primary_build_namespace"]
        / build_id
        / "manifest.json"
    ).resolve()
    _require(
        independent_path == expected_independent,
        "Independent-check path differs from publication layout",
    )
    _require(
        manifest_path == expected_manifest,
        "Primary-manifest path differs from publication layout",
    )

    for key in ("schema_version", "contract_id", "policy_id", "build_id"):
        _require(
            pointer[key] == independent[key], f"Verified evidence disagrees: {key}"
        )
        _require(pointer[key] == manifest[key], f"Primary evidence disagrees: {key}")
    _require(
        pointer["schema_version"] == 1
        and pointer["contract_id"] == policy["contract_id"]
        and pointer["policy_id"] == policy["policy_id"],
        "Verified identity differs from Stage 7 policy",
    )
    _require(
        pointer["status"] == independent["status"] == _VERIFIED_STATUS,
        "Stage 7 evidence is not VERIFIED_PASSED",
    )
    _require(
        pointer["authorizations"]
        == independent["authorizations"]
        == _VERIFIED_AUTHORIZATIONS,
        "Verified Stage 7 authorization boundary differs",
    )
    _require(
        independent["verification_id"] == verification_id,
        "Verification IDs differ",
    )
    _require(
        manifest["status"] == _PRIMARY_STATUS
        and manifest["authorizations"] == _PRIMARY_AUTHORIZATIONS,
        "Primary Stage 7 authority boundary differs",
    )

    primary_hash = _sha256_file(manifest_path)
    _require(
        pointer["primary_manifest_sha256"]
        == independent["primary_manifest_sha256"]
        == primary_hash,
        "Primary-manifest hash binding differs",
    )
    _require(
        pointer["independent_check_sha256"] == _sha256_file(independent_path),
        "Independent-check hash binding differs",
    )
    checks = independent.get("checks")
    _require(isinstance(checks, Mapping), "Independent checks are not a mapping")
    _validate_verification_checks(
        checks=checks,
        project_root=project,
        policy=policy,
        manifest=manifest,
        primary_manifest_sha256=primary_hash,
    )

    differences_path = _anchored_path(
        independent_path.parent / "differences.json",
        root=project,
        label="Stage 7 differences payload",
        directory=False,
    )
    differences_bytes = _stable_bytes(differences_path, "Stage 7 differences payload")
    try:
        differences = json.loads(differences_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7AuditArtifactError("Invalid Stage 7 differences payload") from exc
    _require(
        differences == [] and differences_bytes == canonical_json_bytes([]),
        "Verified Stage 7 differences are not the canonical empty list",
    )
    differences_hash = _sha256_bytes(differences_bytes)
    _require(
        pointer["differences_sha256"]
        == independent["differences_sha256"]
        == differences_hash,
        "Differences hash binding differs",
    )

    output_hashes = _output_hashes(manifest)
    _require(
        independent["output_hashes_sha256"]
        == _sha256_bytes(canonical_json_bytes(output_hashes)),
        "Independent output-hashes binding differs",
    )

    outputs = manifest["outputs"]
    quality_dir, quality_relative, quality_receipts = _authenticate_output_directory(
        evidence_kind="quality",
        output=outputs["quality"],
        policy=policy,
        database_root=database,
        build_id=build_id,
        supplied_directory=quality_directory,
        authenticate_names=tuple(policy["outputs"]["quality"]["required_files"]),
    )
    processed_dir, processed_relative, processed_receipts = (
        _authenticate_output_directory(
            evidence_kind="processed",
            output=outputs["processed"],
            policy=policy,
            database_root=database,
            build_id=build_id,
            supplied_directory=processed_directory,
            authenticate_names=(
                "_SUCCESS.json",
                "development_label_month_summary.parquet",
            ),
        )
    )
    _authenticate_success(
        path=quality_dir / "_SUCCESS.json",
        receipt=quality_receipts["_SUCCESS.json"],
        manifest=manifest,
        kind="quality",
        location_kind="label_store_quality",
        status="PRIMARY_QUALITY_PASSED",
    )
    _authenticate_success(
        path=processed_dir / "_SUCCESS.json",
        receipt=processed_receipts["_SUCCESS.json"],
        manifest=manifest,
        kind="processed",
        location_kind="processed_label_store",
        status="PRIMARY_PROCESSED_COMPLETE",
    )

    summary_path = quality_dir / "quality_summary.json"
    summary = _json_object(summary_path, "Stage 7 quality summary")
    hard_gates = {
        key: int(value)
        for key, value in policy["quality_gates"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    quality_flags = {
        key: value
        for key, value in policy["quality_gates"].items()
        if isinstance(value, bool)
    }
    _require(
        summary.get("schema_version") == 1
        and summary.get("contract_id") == policy["contract_id"]
        and summary.get("policy_id") == policy["policy_id"]
        and summary.get("build_id") == build_id
        and summary.get("status") == _PRIMARY_STATUS
        and summary.get("hard_gates") == hard_gates
        and summary.get("quality_policy_flags") == quality_flags,
        "Quality-summary identity, gates, or policy flags differ",
    )
    manifest_summary = manifest.get("summary")
    counts = summary.get("counts")
    _require(
        isinstance(manifest_summary, Mapping)
        and manifest_summary.get("hard_gate_failures") == 0
        and isinstance(counts, Mapping),
        "Stage 7 summary counts are missing or failed",
    )
    for quality_key, manifest_key in (
        ("audit_rows", "audit_rows"),
        ("leg_lineage_rows", "leg_lineage_rows"),
        ("processed_rows", "processed_rows"),
        ("signal_dates", "signal_dates"),
        ("split_membership_rows", "split_membership_rows"),
    ):
        _require(
            counts.get(quality_key) == manifest_summary.get(manifest_key),
            f"Primary and quality counts disagree: {quality_key}",
        )
    for count_name, file_name in _QUALITY_COUNT_TO_FILE.items():
        _require(
            counts.get(count_name) == quality_receipts[file_name]["rows"],
            f"Quality count differs from receipt: {count_name}",
        )
    _require(
        counts.get("month_summary_rows")
        == processed_receipts["development_label_month_summary.parquet"]["rows"],
        "Month-summary count differs from processed receipt",
    )

    coverage = _authenticated_table(
        quality_dir / "label_coverage.parquet",
        name="label_coverage.parquet",
        receipt=quality_receipts["label_coverage.parquet"],
    )
    statuses = _authenticated_table(
        quality_dir / "label_status_counts.parquet",
        name="label_status_counts.parquet",
        receipt=quality_receipts["label_status_counts.parquet"],
    )
    distributions = _authenticated_table(
        quality_dir / "label_distribution.parquet",
        name="label_distribution.parquet",
        receipt=quality_receipts["label_distribution.parquet"],
    )
    timing = _authenticated_table(
        quality_dir / "label_timing_checks.parquet",
        name="label_timing_checks.parquet",
        receipt=quality_receipts["label_timing_checks.parquet"],
    )
    month_summary = _authenticated_table(
        processed_dir / "development_label_month_summary.parquet",
        name="development_label_month_summary.parquet",
        receipt=processed_receipts["development_label_month_summary.parquet"],
    )

    return _AuthenticatedEvidence(
        policy=policy,
        rules=rules,
        pointer=pointer,
        independent=independent,
        manifest=manifest,
        summary=summary,
        coverage=coverage,
        statuses=statuses,
        distributions=distributions,
        timing=timing,
        month_summary=month_summary,
        project_root=project,
        database_root=database,
        quality_relative_path=quality_relative,
        processed_relative_path=processed_relative,
        verified_pointer_relative_path=_safe_relative(
            pointer_path, project, "verified pointer"
        ),
        independent_check_relative_path=_safe_relative(
            independent_path, project, "independent check"
        ),
        differences_relative_path=_safe_relative(
            differences_path, project, "verification differences"
        ),
        primary_manifest_relative_path=_safe_relative(
            manifest_path, project, "primary manifest"
        ),
    )


def _holding_year(value: Any) -> int:
    _require(
        isinstance(value, str) and _HOLDING_MONTH.fullmatch(value) is not None,
        "Invalid holding-month value",
    )
    return int(value[:4])


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"Nonfinite or nonnumeric {label}",
    )
    return float(value)


def _coverage_rows(evidence: _AuthenticatedEvidence) -> list[dict[str, Any]]:
    required = {
        "signal_date",
        "holding_month",
        "eligible_count",
        "beta_complete_count",
        "outcome_present_count",
        "primary_present_count",
        "magnitude_present_count",
        "eligible_denominator",
        "outcome_coverage_rate",
        "primary_coverage_rate",
        "magnitude_coverage_rate",
        "primary_month_valid",
    }
    _require(
        set(evidence.coverage.column_names) == required,
        "Label-coverage columns differ",
    )
    seen: set[tuple[dt.date, str]] = set()
    prior: tuple[int, bytes] | None = None
    aggregate: dict[int, dict[str, Any]] = {}
    for raw in evidence.coverage.to_pylist():
        signal_date = raw["signal_date"]
        holding_month = raw["holding_month"]
        _require(isinstance(signal_date, dt.date), "Invalid coverage signal date")
        key = (signal_date, holding_month)
        _require(key not in seen, "Duplicate label-coverage key")
        seen.add(key)
        normalized = (signal_date.toordinal(), holding_month.encode("utf-8"))
        _require(prior is None or normalized >= prior, "Label coverage is not sorted")
        prior = normalized
        year = _holding_year(holding_month)
        eligible = int(raw["eligible_count"])
        denominator = int(raw["eligible_denominator"])
        beta_complete = int(raw["beta_complete_count"])
        _require(
            eligible >= 0
            and denominator == eligible
            and 0 <= beta_complete <= eligible,
            "Label-coverage eligibility arithmetic differs",
        )
        for dimension, present_field in _COVERAGE_FIELDS.items():
            present = int(raw[present_field])
            rate = _finite(raw[f"{dimension}_coverage_rate"], "coverage rate")
            expected_rate = 0.0 if denominator == 0 else present / denominator
            _require(
                0 <= present <= denominator
                and math.isclose(rate, expected_rate, rel_tol=0.0, abs_tol=1e-15),
                f"{dimension} coverage arithmetic differs",
            )
        bucket = aggregate.setdefault(
            year,
            {
                "signal_months": 0,
                "valid_primary_months": 0,
                "eligible_denominator": 0,
                "outcome": 0,
                "primary": 0,
                "magnitude": 0,
            },
        )
        bucket["signal_months"] += 1
        bucket["valid_primary_months"] += int(bool(raw["primary_month_valid"]))
        bucket["eligible_denominator"] += denominator
        for dimension, field in _COVERAGE_FIELDS.items():
            bucket[dimension] += int(raw[field])
    _require(
        len(seen) == int(evidence.summary["counts"]["coverage_rows"])
        and len(seen) == int(evidence.summary["counts"]["signal_dates"]),
        "Coverage key count differs from quality summary",
    )
    rows: list[dict[str, Any]] = []
    for year in sorted(aggregate):
        bucket = aggregate[year]
        denominator = int(bucket["eligible_denominator"])
        for dimension in ("outcome", "primary", "magnitude"):
            present = int(bucket[dimension])
            rows.append(
                {
                    "holding_year": year,
                    "coverage_type": _COVERAGE_LABELS[dimension],
                    "signal_months": int(bucket["signal_months"]),
                    "valid_primary_months": int(bucket["valid_primary_months"]),
                    "present_count": present,
                    "missing_count": denominator - present,
                    "eligible_denominator": denominator,
                    "coverage_rate": 0.0 if denominator == 0 else present / denominator,
                }
            )
    _require(bool(rows) and len(rows) <= 150, "Coverage snapshot is empty or unbounded")
    return rows


def _coverage_by_key(
    evidence: _AuthenticatedEvidence,
) -> dict[tuple[dt.date, str], dict[str, Any]]:
    return {
        (row["signal_date"], row["holding_month"]): row
        for row in evidence.coverage.to_pylist()
    }


def _status_rows(
    evidence: _AuthenticatedEvidence,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = {
        "signal_date",
        "holding_month",
        "status_dimension",
        "status",
        "row_count",
        "dimension_total",
        "row_fraction",
    }
    _require(
        set(evidence.statuses.column_names) == required,
        "Label-status columns differ",
    )
    domains = evidence.rules.get("value_domains")
    _require(isinstance(domains, Mapping), "Missing Stage 7 status domains")
    allowed: dict[str, tuple[str, ...]] = {}
    for dimension, domain_name in _DIMENSION_DOMAIN.items():
        values = domains.get(domain_name)
        _require(
            isinstance(values, list)
            and values
            and all(isinstance(item, str) and item for item in values),
            f"Invalid {dimension} status domain",
        )
        allowed[dimension] = tuple(values)

    coverage = _coverage_by_key(evidence)
    grouped: dict[tuple[dt.date, str, str], dict[str, Any]] = {}
    seen: set[tuple[dt.date, str, str, str]] = set()
    aggregate: dict[tuple[str, str], dict[str, int]] = {}
    prior: tuple[int, bytes, bytes] | None = None
    for raw in evidence.statuses.to_pylist():
        signal_date = raw["signal_date"]
        holding_month = raw["holding_month"]
        dimension = raw["status_dimension"]
        status = raw["status"]
        _require(
            isinstance(signal_date, dt.date)
            and dimension in allowed
            and status in allowed[dimension],
            "Unknown Stage 7 status row",
        )
        _holding_year(holding_month)
        key = (signal_date, holding_month, dimension, status)
        _require(key not in seen, "Duplicate label-status key")
        seen.add(key)
        normalized = (
            signal_date.toordinal(),
            dimension.encode("utf-8"),
            status.encode("utf-8"),
        )
        _require(prior is None or normalized >= prior, "Label statuses are not sorted")
        prior = normalized
        count = int(raw["row_count"])
        total = int(raw["dimension_total"])
        fraction = _finite(raw["row_fraction"], "status fraction")
        _require(
            count >= 0
            and total > 0
            and count <= total
            and math.isclose(fraction, count / total, rel_tol=0.0, abs_tol=1e-15),
            "Label-status arithmetic differs",
        )
        group = grouped.setdefault(
            (signal_date, holding_month, dimension),
            {"dimension_total": total, "counts": {}},
        )
        _require(
            group["dimension_total"] == total, "Status denominator differs within month"
        )
        group["counts"][status] = count
        bucket = aggregate.setdefault(
            (dimension, status),
            {"row_count": 0, "dimension_total": 0},
        )
        bucket["row_count"] += count
        bucket["dimension_total"] += total

    expected_row_count = len(coverage) * sum(len(values) for values in allowed.values())
    _require(
        len(seen) == expected_row_count
        and len(seen) == int(evidence.summary["counts"]["status_count_rows"]),
        "Dense status key domain differs",
    )
    for (signal_date, holding_month), coverage_row in coverage.items():
        for dimension in allowed:
            group = grouped.get((signal_date, holding_month, dimension))
            _require(group is not None, "Missing dense status month/dimension")
            counts = group["counts"]
            _require(
                set(counts) == set(allowed[dimension])
                and sum(counts.values()) == group["dimension_total"],
                "Dense status domain or total differs",
            )
            present_field = _COVERAGE_FIELDS[dimension]
            eligible = int(coverage_row["eligible_count"])
            _require(
                counts["PRESENT"] == int(coverage_row[present_field])
                and counts["NOT_LONG_ELIGIBLE"] == group["dimension_total"] - eligible
                and sum(
                    count
                    for status, count in counts.items()
                    if status.startswith("MISSING_")
                )
                == eligible - counts["PRESENT"],
                "Status counts disagree with coverage",
            )

    totals_by_dimension: dict[str, dict[str, int]] = {}
    for (dimension, status), bucket in aggregate.items():
        dimension_bucket = totals_by_dimension.setdefault(
            dimension,
            {"audit": int(bucket["dimension_total"]), "not_long": 0},
        )
        _require(
            dimension_bucket["audit"] == int(bucket["dimension_total"]),
            "Aggregated status denominators differ",
        )
        if status == "NOT_LONG_ELIGIBLE":
            dimension_bucket["not_long"] = int(bucket["row_count"])

    summary_rows: list[dict[str, Any]] = []
    for dimension in ("outcome", "primary", "magnitude"):
        totals = totals_by_dimension[dimension]
        eligible = totals["audit"] - totals["not_long"]
        for status in allowed[dimension]:
            bucket = aggregate[(dimension, status)]
            count = int(bucket["row_count"])
            summary_rows.append(
                {
                    "status_dimension": dimension,
                    "status": status,
                    "row_count": count,
                    "audit_denominator": totals["audit"],
                    "share_of_audit": count / totals["audit"],
                    "long_eligible_denominator": eligible,
                    "share_of_long_eligible": (
                        None
                        if status == "NOT_LONG_ELIGIBLE"
                        else (0.0 if eligible == 0 else count / eligible)
                    ),
                }
            )
    _require(len(summary_rows) <= 100, "Status summary exceeds portable limit")

    outcome_totals = totals_by_dimension["outcome"]
    outcome_eligible = outcome_totals["audit"] - outcome_totals["not_long"]
    missing_items = [
        (status, int(aggregate[("outcome", status)]["row_count"]))
        for status in allowed["outcome"]
        if status.startswith("MISSING_")
    ]
    missing_total = sum(count for _, count in missing_items)
    outcome_rows = [
        {
            "missing_classification": status,
            "classification_label": status.removeprefix("MISSING_")
            .replace("_", " ")
            .title(),
            "missing_count": count,
            "share_of_missing_outcomes": (
                0.0 if missing_total == 0 else count / missing_total
            ),
            "share_of_long_eligible": (
                0.0 if outcome_eligible == 0 else count / outcome_eligible
            ),
            "missing_outcome_denominator": missing_total,
            "long_eligible_denominator": outcome_eligible,
        }
        for status, count in sorted(
            missing_items, key=lambda item: (-item[1], item[0].encode("utf-8"))
        )
    ]
    _require(
        bool(outcome_rows) and len(outcome_rows) <= 50,
        "Outcome-missing snapshot differs",
    )
    return summary_rows, outcome_rows


def _delayed_rows(evidence: _AuthenticatedEvidence) -> list[dict[str, Any]]:
    required = {
        "signal_date",
        "holding_month",
        "entry_date",
        "exit_date",
        "eligible_security_count",
        "beta_complete_count",
        "valid_forward_count",
        "valid_primary_count",
        "minimum_required_count",
        "primary_month_valid",
        "terminal_delisting_label_count",
        "delayed_availability_count",
        "month_label_available_date",
        "residual_median",
        "residual_mad",
        "residual_rank_numerator_sum",
        "excess_rank_numerator_sum",
        "residual_rank_mean",
        "excess_rank_mean",
        "month_status",
    }
    _require(
        set(evidence.month_summary.column_names) == required,
        "Month-summary columns differ",
    )
    coverage = _coverage_by_key(evidence)
    aggregate: dict[int, dict[str, Any]] = {}
    seen: set[tuple[dt.date, str]] = set()
    prior: int | None = None
    for raw in evidence.month_summary.to_pylist():
        signal_date = raw["signal_date"]
        holding_month = raw["holding_month"]
        _require(isinstance(signal_date, dt.date), "Invalid month-summary signal date")
        key = (signal_date, holding_month)
        _require(
            key not in seen and key in coverage, "Month-summary key domain differs"
        )
        seen.add(key)
        _require(
            prior is None or signal_date.toordinal() >= prior,
            "Month summary is not sorted",
        )
        prior = signal_date.toordinal()
        year = _holding_year(holding_month)
        eligible = int(raw["eligible_security_count"])
        beta_complete = int(raw["beta_complete_count"])
        valid_forward = int(raw["valid_forward_count"])
        valid_primary = int(raw["valid_primary_count"])
        terminal = int(raw["terminal_delisting_label_count"])
        delayed = int(raw["delayed_availability_count"])
        minimum = int(raw["minimum_required_count"])
        _require(
            eligible == int(coverage[key]["eligible_count"])
            and 0 <= beta_complete <= eligible
            and 0 <= valid_forward <= beta_complete
            and 0 <= valid_primary <= valid_forward
            and 0 <= terminal <= valid_primary
            and 0 <= delayed <= valid_primary
            and minimum > 0,
            "Month-summary count ordering differs",
        )
        month_available = raw["month_label_available_date"]
        exit_date = raw["exit_date"]
        _require(
            isinstance(exit_date, dt.date)
            and (month_available is None or isinstance(month_available, dt.date)),
            "Invalid month availability dates",
        )
        if bool(raw["primary_month_valid"]):
            _require(
                valid_primary >= minimum
                and month_available is not None
                and month_available >= exit_date,
                "Valid primary month availability differs",
            )
        lag = None if month_available is None else (month_available - exit_date).days
        _require(lag is None or lag >= 0, "Month label is available before exit")
        bucket = aggregate.setdefault(
            year,
            {
                "signal_months": 0,
                "valid_primary_months": 0,
                "months_with_delayed_availability": 0,
                "eligible_security_count": 0,
                "valid_primary_count": 0,
                "terminal_delisting_label_count": 0,
                "delayed_availability_count": 0,
                "maximum_month_availability_lag_days": 0,
            },
        )
        bucket["signal_months"] += 1
        bucket["valid_primary_months"] += int(bool(raw["primary_month_valid"]))
        bucket["months_with_delayed_availability"] += int(delayed > 0)
        bucket["eligible_security_count"] += eligible
        bucket["valid_primary_count"] += valid_primary
        bucket["terminal_delisting_label_count"] += terminal
        bucket["delayed_availability_count"] += delayed
        if lag is not None:
            bucket["maximum_month_availability_lag_days"] = max(
                bucket["maximum_month_availability_lag_days"], lag
            )
    _require(
        set(seen) == set(coverage)
        and len(seen) == int(evidence.summary["counts"]["month_summary_rows"]),
        "Month-summary population differs from coverage",
    )
    rows: list[dict[str, Any]] = []
    for year in sorted(aggregate):
        bucket = aggregate[year]
        valid_primary = int(bucket["valid_primary_count"])
        rows.append(
            {
                "holding_year": year,
                **{key: int(value) for key, value in bucket.items()},
                "delayed_share_of_valid_primary": (
                    0.0
                    if valid_primary == 0
                    else int(bucket["delayed_availability_count"]) / valid_primary
                ),
            }
        )
    _require(
        bool(rows) and len(rows) <= 50, "Delayed-availability snapshot is unbounded"
    )
    return rows


def _distribution_rows(evidence: _AuthenticatedEvidence) -> list[dict[str, Any]]:
    required = {
        "signal_date",
        "holding_month",
        "target_name",
        "finite_count",
        "missing_count",
        "minimum",
        "p01",
        "median",
        "mean",
        "p99",
        "maximum",
    }
    _require(
        set(evidence.distributions.column_names) == required,
        "Label-distribution columns differ",
    )
    target_domain = evidence.rules.get("value_domains", {}).get(
        "distribution_target_name"
    )
    _require(
        isinstance(target_domain, list)
        and target_domain
        and all(isinstance(value, str) and value for value in target_domain),
        "Distribution target domain differs",
    )
    target_names = tuple(target_domain)
    coverage = _coverage_by_key(evidence)
    seen: set[tuple[dt.date, str]] = set()
    by_month: dict[tuple[dt.date, str], set[str]] = {}
    aggregate: dict[str, dict[str, Any]] = {
        target: {
            "signal_months": 0,
            "months_with_finite_values": 0,
            "finite_count": 0,
            "missing_count": 0,
            "weighted_mean_numerator": [],
            "minimums": [],
            "p01s": [],
            "medians": [],
            "p99s": [],
            "maximums": [],
        }
        for target in target_names
    }
    prior: tuple[int, bytes] | None = None
    for raw in evidence.distributions.to_pylist():
        signal_date = raw["signal_date"]
        holding_month = raw["holding_month"]
        target = raw["target_name"]
        _require(
            isinstance(signal_date, dt.date)
            and target in aggregate
            and (signal_date, holding_month) in coverage,
            "Unknown label-distribution key",
        )
        key = (signal_date, target)
        _require(key not in seen, "Duplicate label-distribution key")
        seen.add(key)
        normalized = (signal_date.toordinal(), target.encode("utf-8"))
        _require(
            prior is None or normalized >= prior, "Label distributions are not sorted"
        )
        prior = normalized
        by_month.setdefault((signal_date, holding_month), set()).add(target)
        finite_count = int(raw["finite_count"])
        missing_count = int(raw["missing_count"])
        eligible = int(coverage[(signal_date, holding_month)]["eligible_count"])
        _require(
            finite_count >= 0
            and missing_count >= 0
            and finite_count + missing_count == eligible,
            "Label-distribution population differs",
        )
        statistics_fields = ("minimum", "p01", "median", "mean", "p99", "maximum")
        if finite_count == 0:
            _require(
                all(raw[field] is None for field in statistics_fields),
                "Empty label distribution has non-null statistics",
            )
        else:
            values = [
                _finite(raw[field], f"distribution {field}")
                for field in statistics_fields
            ]
            minimum, p01, median, mean, p99, maximum = values
            _require(
                minimum <= p01 <= median <= p99 <= maximum
                and minimum <= mean <= maximum,
                "Label-distribution order differs",
            )
            bucket = aggregate[target]
            bucket["months_with_finite_values"] += 1
            bucket["weighted_mean_numerator"].append(mean * finite_count)
            bucket["minimums"].append(minimum)
            bucket["p01s"].append(p01)
            bucket["medians"].append(median)
            bucket["p99s"].append(p99)
            bucket["maximums"].append(maximum)
        bucket = aggregate[target]
        bucket["signal_months"] += 1
        bucket["finite_count"] += finite_count
        bucket["missing_count"] += missing_count
    _require(
        set(by_month) == set(coverage)
        and all(values == set(target_names) for values in by_month.values())
        and len(seen) == int(evidence.summary["counts"]["distribution_rows"]),
        "Distribution key domain differs",
    )
    rows: list[dict[str, Any]] = []
    for target in target_names:
        bucket = aggregate[target]
        finite_count = int(bucket["finite_count"])
        total = finite_count + int(bucket["missing_count"])
        rows.append(
            {
                "target_name": target,
                "signal_months": int(bucket["signal_months"]),
                "months_with_finite_values": int(bucket["months_with_finite_values"]),
                "finite_count": finite_count,
                "missing_count": int(bucket["missing_count"]),
                "finite_share": 0.0 if total == 0 else finite_count / total,
                "observed_minimum": (
                    None if not bucket["minimums"] else min(bucket["minimums"])
                ),
                "lowest_month_p01": None if not bucket["p01s"] else min(bucket["p01s"]),
                "median_of_monthly_medians": (
                    None
                    if not bucket["medians"]
                    else float(statistics.median(bucket["medians"]))
                ),
                "finite_weighted_mean": (
                    None
                    if finite_count == 0
                    else math.fsum(bucket["weighted_mean_numerator"]) / finite_count
                ),
                "highest_month_p99": (
                    None if not bucket["p99s"] else max(bucket["p99s"])
                ),
                "observed_maximum": (
                    None if not bucket["maximums"] else max(bucket["maximums"])
                ),
            }
        )
    _require(0 < len(rows) <= 20, "Distribution snapshot is empty or unbounded")
    return rows


def _gate_rows(evidence: _AuthenticatedEvidence) -> list[dict[str, Any]]:
    hard_gates = {
        key: int(value)
        for key, value in evidence.policy["quality_gates"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    _require(len(hard_gates) == 57, "Frozen Stage 7 gate count is not exactly 57")
    rows: list[dict[str, Any]] = []
    observed_names: set[str] = set()
    prior: bytes | None = None
    for raw in evidence.timing.to_pylist():
        name = raw["check_name"]
        _require(
            isinstance(name, str) and name in hard_gates and name not in observed_names,
            "Timing-check key domain differs",
        )
        encoded = name.encode("utf-8")
        _require(prior is None or encoded >= prior, "Timing checks are not sorted")
        prior = encoded
        observed_names.add(name)
        observed = raw["observed_count"]
        expected = raw["expected_count"]
        _require(
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and isinstance(expected, int)
            and not isinstance(expected, bool)
            and observed == expected == hard_gates[name]
            and raw["passed"] is True
            and raw["severity"] == "CRITICAL",
            f"Stage 7 gate failed or differs: {name}",
        )
        rows.append(
            {
                "check_name": name,
                "observed_count": observed,
                "expected_count": expected,
                "passed": True,
                "result": "PASS",
                "severity": raw["severity"],
                "details": raw["details"],
            }
        )
    _require(
        observed_names == set(hard_gates)
        and len(rows) == 57
        and evidence.summary["hard_gates"] == hard_gates,
        "Exact Stage 7 gate set differs",
    )
    return rows


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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
    pointer = evidence.verified_pointer_relative_path.as_posix()
    independent = evidence.independent_check_relative_path.as_posix()
    differences = evidence.differences_relative_path.as_posix()
    manifest = evidence.primary_manifest_relative_path.as_posix()
    quality = evidence.quality_relative_path.as_posix()
    processed = evidence.processed_relative_path.as_posix()
    common_filters = [
        f"build_id = {build_id}",
        f"verification_id = {verification_id}",
        "status = VERIFIED_PASSED",
        "development labels only: 2000-01-31 through 2014-10-31 signals",
        "no validation or final-period payload",
    ]
    return [
        _source_query(
            source_id="verification_evidence",
            label="Verified Stage 7 pointer, check, differences, and primary manifest",
            sql=f"""
WITH pointer AS (
    SELECT * FROM read_json_auto({_sql_literal(pointer)})
), independent AS (
    SELECT * FROM read_json_auto({_sql_literal(independent)})
), differences AS (
    SELECT COUNT(*) AS difference_count
    FROM read_json_auto({_sql_literal(differences)})
), primary_manifest AS (
    SELECT * FROM read_json_auto({_sql_literal(manifest)})
)
SELECT
    pointer.build_id,
    pointer.verification_id,
    pointer.status,
    independent.checks.check_count,
    independent.checks.pass_count,
    independent.checks.fail_count,
    differences.difference_count,
    primary_manifest.summary.audit_rows,
    primary_manifest.summary.processed_rows,
    primary_manifest.summary.signal_dates
FROM pointer
CROSS JOIN independent
CROSS JOIN differences
CROSS JOIN primary_manifest
""".strip(),
            tables=(pointer, independent, differences, manifest),
            description=(
                "Binds the report to one verified pointer, its exact independent "
                "check and canonical empty differences payload, and the hash-bound "
                "primary manifest."
            ),
            filters=common_filters,
        ),
        _source_query(
            source_id="quality_summary",
            label="Authenticated Stage 7 quality summary",
            sql=(
                "SELECT *\n"
                f"FROM read_json_auto({_sql_literal(quality + '/quality_summary.json')})\n"
                f"WHERE build_id = {_sql_literal(build_id)}"
            ),
            tables=(quality + "/quality_summary.json",),
            description="Reads the canonical quality summary authenticated from its primary receipt.",
            filters=common_filters,
        ),
        _source_query(
            source_id="coverage_by_year",
            label="Authenticated label coverage by holding year",
            sql=f"""
SELECT
    CAST(SUBSTR(holding_month, 1, 4) AS INTEGER) AS holding_year,
    SUM(eligible_denominator)::BIGINT AS eligible_denominator,
    SUM(outcome_present_count)::BIGINT AS outcome_present_count,
    SUM(primary_present_count)::BIGINT AS primary_present_count,
    SUM(magnitude_present_count)::BIGINT AS magnitude_present_count
FROM read_parquet({_sql_literal(quality + '/label_coverage.parquet')})
GROUP BY holding_year
ORDER BY holding_year
""".strip(),
            tables=(quality + "/label_coverage.parquet",),
            description="Aggregates authenticated monthly coverage using summed counts and denominators.",
            filters=common_filters,
            metric_definitions=(
                "coverage_rate = SUM(present_count) / SUM(eligible_denominator) within holding year.",
                "Coverage is descriptive and missing outcomes are never filled with zero.",
            ),
        ),
        _source_query(
            source_id="status_summary",
            label="Authenticated Stage 7 status summary",
            sql=f"""
SELECT
    status_dimension,
    status,
    SUM(row_count)::BIGINT AS row_count,
    SUM(dimension_total)::BIGINT AS audit_denominator
FROM read_parquet({_sql_literal(quality + '/label_status_counts.parquet')})
GROUP BY status_dimension, status
ORDER BY status_dimension, status
""".strip(),
            tables=(quality + "/label_status_counts.parquet",),
            description="Aggregates the dense authenticated status cube without dropping zero-count categories.",
            filters=common_filters,
            metric_definitions=(
                "share_of_audit uses all audit-spine rows in the selected status dimension.",
                "share_of_long_eligible excludes NOT_LONG_ELIGIBLE rows from its denominator.",
            ),
        ),
        _source_query(
            source_id="outcome_missing_classification",
            label="Authenticated outcome-missing classifications",
            sql=f"""
SELECT status, SUM(row_count)::BIGINT AS missing_count
FROM read_parquet({_sql_literal(quality + '/label_status_counts.parquet')})
WHERE status_dimension = 'outcome' AND status LIKE 'MISSING_%'
GROUP BY status
ORDER BY missing_count DESC, status
""".strip(),
            tables=(quality + "/label_status_counts.parquet",),
            description="Preserves every allowed missing-outcome reason, including zero-count reasons.",
            filters=common_filters,
            metric_definitions=(
                "share_of_missing_outcomes divides by all outcome statuses beginning MISSING_ only.",
            ),
        ),
        _source_query(
            source_id="delayed_availability_by_year",
            label="Authenticated delayed label availability by holding year",
            sql=f"""
SELECT
    CAST(SUBSTR(holding_month, 1, 4) AS INTEGER) AS holding_year,
    COUNT(*)::BIGINT AS signal_months,
    SUM(valid_primary_count)::BIGINT AS valid_primary_count,
    SUM(delayed_availability_count)::BIGINT AS delayed_availability_count,
    MAX(DATE_DIFF('day', exit_date, month_label_available_date))::BIGINT
        AS maximum_month_availability_lag_days
FROM read_parquet({_sql_literal(processed + '/development_label_month_summary.parquet')})
GROUP BY holding_year
ORDER BY holding_year
""".strip(),
            tables=(processed + "/development_label_month_summary.parquet",),
            description="Uses the authenticated monthly processed summary because delay counts are not quality-table fields.",
            filters=common_filters,
            metric_definitions=(
                "delayed_availability_count counts valid primary outcomes available after the scheduled exit date.",
                "delayed_share_of_valid_primary divides that count by valid_primary_count.",
            ),
        ),
        _source_query(
            source_id="distribution_summary",
            label="Authenticated Stage 7 target-distribution summary",
            sql=f"""
SELECT
    target_name,
    SUM(finite_count)::BIGINT AS finite_count,
    SUM(missing_count)::BIGINT AS missing_count,
    MIN(minimum) AS observed_minimum,
    MEDIAN(median) AS median_of_monthly_medians,
    SUM(mean * finite_count) / NULLIF(SUM(finite_count), 0) AS finite_weighted_mean,
    MAX(maximum) AS observed_maximum
FROM read_parquet({_sql_literal(quality + '/label_distribution.parquet')})
GROUP BY target_name
ORDER BY target_name
""".strip(),
            tables=(quality + "/label_distribution.parquet",),
            description="Summarizes authenticated monthly distribution rows without claiming pooled quantiles.",
            filters=common_filters,
            metric_definitions=(
                "finite_weighted_mean weights each monthly mean by that month's finite_count.",
                "median_of_monthly_medians is descriptive and is not the pooled security-level median.",
            ),
        ),
        _source_query(
            source_id="exact_quality_gates",
            label="Authenticated exact 57 Stage 7 gates",
            sql=f"""
SELECT check_name, observed_count, expected_count, passed, severity, details
FROM read_parquet({_sql_literal(quality + '/label_timing_checks.parquet')})
ORDER BY check_name
""".strip(),
            tables=(quality + "/label_timing_checks.parquet",),
            description="Reads all 57 exact frozen Stage 7 timing and integrity gates.",
            filters=common_filters,
            metric_definitions=(
                "A gate passes only when passed is true and observed_count equals the frozen expected_count.",
            ),
        ),
    ]


def _table_column(
    field: str,
    label: str,
    column_type: str,
    *,
    format_name: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"field": field, "label": label, "type": column_type}
    if format_name is not None:
        value["format"] = format_name
    return value


def _artifact_payload(
    evidence: _AuthenticatedEvidence,
    *,
    coverage_rows: Sequence[Mapping[str, Any]],
    status_rows: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
    delayed_rows: Sequence[Mapping[str, Any]],
    distribution_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    build_id = str(evidence.pointer["build_id"])
    verification_id = str(evidence.pointer["verification_id"])
    checks = evidence.independent["checks"]
    counts = evidence.summary["counts"]
    sources = _sources(evidence)
    primary_coverage = [
        row for row in coverage_rows if row["coverage_type"] == "Primary label present"
    ]
    _require(bool(primary_coverage), "Primary coverage summary is empty")
    primary_present = sum(int(row["present_count"]) for row in primary_coverage)
    primary_denominator = sum(
        int(row["eligible_denominator"]) for row in primary_coverage
    )
    primary_rate = (
        0.0 if primary_denominator == 0 else primary_present / primary_denominator
    )
    outcome_missing_total = sum(int(row["missing_count"]) for row in outcome_rows)
    outcome_eligible = int(outcome_rows[0]["long_eligible_denominator"])
    delayed_total = sum(int(row["delayed_availability_count"]) for row in delayed_rows)
    delayed_denominator = sum(int(row["valid_primary_count"]) for row in delayed_rows)
    delayed_rate = (
        0.0 if delayed_denominator == 0 else delayed_total / delayed_denominator
    )
    valid_months = sum(int(row["valid_primary_months"]) for row in delayed_rows)
    title = "Stage 7 Label Store: Verified Quality Audit"

    blocks: list[dict[str, Any]] = [
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
                "## Verification sealed the development label store\n\n"
                f"Build {build_id} is bound to verification {verification_id}. "
                f"The independent receipt records **{checks['pass_count']:,} of "
                f"{checks['check_count']:,}** reported check groups passed, zero "
                "failures, and a canonical empty differences payload. The "
                f"authenticated quality bundle separately preserves **{len(gate_rows)} "
                "of 57** exact frozen gates as passing. This authority opens only "
                "Stage 8 risk and cost work; it does not evaluate predictive or "
                "portfolio results."
            ),
        },
        {
            "id": "coverage_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "coverage_by_year",
            "body": (
                "## Primary-label coverage remains explicit across holding years\n\n"
                f"Across **{len(primary_coverage)} holding years**, the primary target "
                f"is present for **{primary_present:,} of {primary_denominator:,}** "
                f"long-eligible label opportunities ({primary_rate:.2%}). The chart "
                "intentionally shows only the primary target; the complete outcome, "
                "primary, and magnitude annual rows remain in the bounded audit "
                "snapshot. Rates use summed numerators and denominators within year. "
                "Missing outcomes remain missing; no row is converted to a zero return."
            ),
        },
        {
            "id": "coverage_chart_block",
            "type": "chart",
            "layout": "full",
            "chartId": "coverage_chart",
        },
        {
            "id": "status_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "status_summary",
            "body": (
                "## Dense statuses reconcile every audit row\n\n"
                f"The bounded status summary retains **{len(status_rows)}** allowed "
                "dimension-status combinations, including categories with zero rows. "
                "Shares are shown against both the full audit spine and, where "
                "meaningful, the long-eligible population."
            ),
        },
        {
            "id": "status_table_block",
            "type": "table",
            "layout": "full",
            "tableId": "status_table",
        },
        {
            "id": "outcome_missing_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "outcome_missing_classification",
            "body": (
                "## Missing outcomes retain their operational cause\n\n"
                f"There are **{outcome_missing_total:,}** missing outcomes among "
                f"**{outcome_eligible:,}** long-eligible rows. The exact allowed "
                "classifications below remain separate so return gaps, interval "
                "straddles, unusable legs, unresolved delistings, and other causes "
                "cannot be silently pooled or zero-filled."
            ),
        },
        {
            "id": "outcome_missing_table_block",
            "type": "table",
            "layout": "full",
            "tableId": "outcome_missing_table",
        },
        {
            "id": "availability_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "delayed_availability_by_year",
            "body": (
                "## Delayed availability is measured after the scheduled exit\n\n"
                f"The authenticated monthly summary identifies **{delayed_total:,}** "
                f"delayed labels among **{delayed_denominator:,}** valid primary "
                f"labels ({delayed_rate:.3%}), across **{valid_months:,}** valid "
                "signal months. Availability dates—not exit dates alone—remain the "
                "training-time boundary."
            ),
        },
        {
            "id": "availability_table_block",
            "type": "table",
            "layout": "full",
            "tableId": "availability_table",
        },
        {
            "id": "distribution_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "distribution_summary",
            "body": (
                "## Target distributions are descriptive, not evidence of skill\n\n"
                f"The report preserves bounded summaries for **{len(distribution_rows)}** "
                "published targets. The weighted mean uses monthly finite counts, "
                "while the displayed median is explicitly the median of monthly "
                "medians—not a pooled security-level quantile."
            ),
        },
        {
            "id": "distribution_table_block",
            "type": "table",
            "layout": "full",
            "tableId": "distribution_table",
        },
        {
            "id": "gate_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "exact_quality_gates",
            "body": (
                "## All 57 frozen timing and integrity gates are visible\n\n"
                "The table keeps every observed and expected integer value. A row "
                "appears as passing only when its boolean pass flag is true and its "
                "observed count exactly equals the frozen expected count."
            ),
        },
        {
            "id": "gate_table_block",
            "type": "table",
            "layout": "full",
            "tableId": "gate_table",
        },
        {
            "id": "scope_definitions",
            "type": "markdown",
            "layout": "full",
            "sourceId": "quality_summary",
            "body": (
                "## Scope, data, and metric definitions\n\n"
                f"This development-only store covers **{counts['audit_rows']:,} "
                f"audit-spine rows**, **{counts['processed_rows']:,} long-eligible "
                f"processed rows**, **{counts['signal_dates']:,} signal dates**, and "
                f"**{counts['leg_lineage_rows']:,} return-leg lineage rows**. Coverage "
                "denominators are long-eligible rows. Status denominators are all "
                "audit-spine rows within each signal date and dimension."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "layout": "full",
            "sourceId": "verification_evidence",
            "body": (
                "## Methodology binds every summary to authenticated receipts\n\n"
                "Before deriving any snapshot row, the generator authenticates the "
                "reviewed policy and physical rules, verified pointer, independent "
                "check, canonical empty differences payload, primary manifest, "
                "independent output-hash digest, closed quality file set, every "
                "quality receipt, and the processed monthly-summary receipt. Files "
                "are rechecked around payload reads. The report then performs only "
                "bounded aggregations of already-published quality evidence."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Limitations and robustness boundary\n\n"
                "Verification establishes conformity to the frozen development label "
                "contract and a zero-difference independent seal. It does not establish "
                "forecasting usefulness, investability, economic significance, or "
                "behavior outside the development period. Aggregated annual coverage "
                "can conceal month-level variation, and the distribution table does not "
                "reconstruct pooled quantiles from security-level rows."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Recommended next step\n\n"
                "Proceed only to Stage 8 risk and cost model preparation under the "
                "verified authorization boundary. Keep Stage 9 training, portfolio "
                "construction, P&L, backtesting, and later-period payloads closed until "
                "their roadmap gates explicitly authorize them."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Further questions\n\n"
                "- Which Stage 8 covariance and transaction-cost inputs need their own availability dates?\n"
                "- Which missing-outcome causes should be monitored separately during later rolling fits?\n"
                "- Should availability-delay summaries become a recurring training-data guardrail?\n"
                "- Which robustness variants require their own immutable data and configuration receipts?"
            ),
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Verification-bound Stage 7 development label-store quality evidence, "
            "availability controls, and authorization boundary."
        ),
        "filters": [],
        "cards": [],
        "charts": [
            {
                "id": "coverage_chart",
                "dataset": "primary_coverage_by_year",
                "type": "line",
                "intent": "trend",
                "title": "Primary-label coverage by holding year",
                "subtitle": (
                    "Present primary labels divided by long-eligible label opportunities; "
                    "development holding years only"
                ),
                "question": (
                    "How does primary-label availability vary across development "
                    "holding years?"
                ),
                "rationale": (
                    "A single-series line chart preserves the primary target's annual "
                    "path without drawing coincident coverage series on top of one "
                    "another. Exact counts for all three measures remain in the "
                    "bounded audit dataset."
                ),
                "sourceId": "coverage_by_year",
                "valueFormat": "percent",
                "showDescription": True,
                "layout": "full",
                "maxRows": len(primary_coverage),
                "encodings": {
                    "x": {
                        "field": "holding_year",
                        "type": "ordinal",
                        "label": "Holding year",
                    },
                    "y": {
                        "field": "coverage_rate",
                        "type": "quantitative",
                        "label": "Coverage rate",
                        "format": "percent",
                    },
                    "tooltip": [
                        _table_column(
                            "present_count",
                            "Present labels",
                            "quantitative",
                            format_name="number",
                        ),
                        _table_column(
                            "missing_count",
                            "Missing labels",
                            "quantitative",
                            format_name="number",
                        ),
                        _table_column(
                            "eligible_denominator",
                            "Long-eligible denominator",
                            "quantitative",
                            format_name="number",
                        ),
                        _table_column(
                            "signal_months",
                            "Signal months",
                            "quantitative",
                            format_name="number",
                        ),
                    ],
                },
            }
        ],
        "tables": [
            {
                "id": "status_table",
                "dataset": "status_summary",
                "title": "Published status counts by target dimension",
                "subtitle": "All allowed statuses across the full development audit spine",
                "sourceId": "status_summary",
                "layout": "full",
                "density": "comfortable",
                "showDescription": True,
                "defaultSort": {"field": "status_dimension", "direction": "asc"},
                "columns": [
                    _table_column("status_dimension", "Dimension", "text"),
                    _table_column("status", "Status", "text"),
                    _table_column("row_count", "Rows", "number", format_name="number"),
                    _table_column(
                        "share_of_audit",
                        "Share of audit",
                        "number",
                        format_name="percent",
                    ),
                    _table_column(
                        "share_of_long_eligible",
                        "Share of long-eligible",
                        "number",
                        format_name="percent",
                    ),
                ],
            },
            {
                "id": "outcome_missing_table",
                "dataset": "outcome_missing_classification",
                "title": "Missing-outcome classifications",
                "subtitle": "Exact allowed causes across long-eligible development rows",
                "sourceId": "outcome_missing_classification",
                "layout": "full",
                "density": "comfortable",
                "showDescription": True,
                "defaultSort": {"field": "missing_count", "direction": "desc"},
                "columns": [
                    _table_column("classification_label", "Classification", "text"),
                    _table_column("missing_classification", "Frozen status", "text"),
                    _table_column(
                        "missing_count", "Missing rows", "number", format_name="number"
                    ),
                    _table_column(
                        "share_of_missing_outcomes",
                        "Share of missing",
                        "number",
                        format_name="percent",
                    ),
                    _table_column(
                        "share_of_long_eligible",
                        "Share of long-eligible",
                        "number",
                        format_name="percent",
                    ),
                ],
            },
            {
                "id": "availability_table",
                "dataset": "delayed_availability_by_year",
                "title": "Delayed label availability by holding year",
                "subtitle": "Valid primary labels available after the scheduled exit date",
                "sourceId": "delayed_availability_by_year",
                "layout": "full",
                "density": "comfortable",
                "showDescription": True,
                "defaultSort": {"field": "holding_year", "direction": "asc"},
                "columns": [
                    _table_column(
                        "holding_year", "Holding year", "number", format_name="number"
                    ),
                    _table_column(
                        "signal_months", "Signal months", "number", format_name="number"
                    ),
                    _table_column(
                        "valid_primary_count",
                        "Valid primary labels",
                        "number",
                        format_name="number",
                    ),
                    _table_column(
                        "delayed_availability_count",
                        "Delayed labels",
                        "number",
                        format_name="number",
                    ),
                    _table_column(
                        "delayed_share_of_valid_primary",
                        "Delayed share",
                        "number",
                        format_name="percent",
                    ),
                    _table_column(
                        "maximum_month_availability_lag_days",
                        "Maximum month lag (days)",
                        "number",
                        format_name="number",
                    ),
                ],
            },
            {
                "id": "distribution_table",
                "dataset": "distribution_summary",
                "title": "Published target-distribution summaries",
                "subtitle": "Finite and missing counts across all development signal months",
                "sourceId": "distribution_summary",
                "layout": "full",
                "density": "comfortable",
                "showDescription": True,
                "defaultSort": {"field": "target_name", "direction": "asc"},
                "columns": [
                    _table_column("target_name", "Target", "text"),
                    _table_column(
                        "finite_count", "Finite", "number", format_name="number"
                    ),
                    _table_column(
                        "missing_count", "Missing", "number", format_name="number"
                    ),
                    _table_column(
                        "finite_share", "Finite share", "number", format_name="percent"
                    ),
                    _table_column(
                        "observed_minimum",
                        "Observed minimum",
                        "number",
                        format_name="number",
                    ),
                    _table_column(
                        "median_of_monthly_medians",
                        "Median of monthly medians",
                        "number",
                        format_name="number",
                    ),
                    _table_column(
                        "finite_weighted_mean",
                        "Finite-weighted mean",
                        "number",
                        format_name="number",
                    ),
                    _table_column(
                        "observed_maximum",
                        "Observed maximum",
                        "number",
                        format_name="number",
                    ),
                ],
            },
            {
                "id": "gate_table",
                "dataset": "exact_quality_gates",
                "title": "Exact 57 Stage 7 timing and integrity gates",
                "subtitle": "Authenticated observed and frozen expected integer values",
                "sourceId": "exact_quality_gates",
                "layout": "full",
                "density": "dense",
                "showDescription": True,
                "defaultSort": {"field": "check_name", "direction": "asc"},
                "columns": [
                    _table_column("check_name", "Check", "text"),
                    _table_column(
                        "observed_count", "Observed", "number", format_name="number"
                    ),
                    _table_column(
                        "expected_count", "Expected", "number", format_name="number"
                    ),
                    _table_column("result", "Result", "text"),
                    _table_column("severity", "Severity", "text"),
                    _table_column("details", "Evidence note", "text"),
                ],
            },
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
                "coverage_by_year": list(coverage_rows),
                "primary_coverage_by_year": list(primary_coverage),
                "status_summary": list(status_rows),
                "outcome_missing_classification": list(outcome_rows),
                "delayed_availability_by_year": list(delayed_rows),
                "distribution_summary": list(distribution_rows),
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
    status_rows: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
    delayed_rows: Sequence[Mapping[str, Any]],
    distribution_rows: Sequence[Mapping[str, Any]],
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
        "# Stage 7 Label Store Quality Audit\n\n"
        "This notebook reads only the bounded, verification-bound snapshot in "
        f"`{artifact_relative_path.as_posix()}`. It contains no label-building, "
        "model-training, portfolio, P&L, or backtest logic.",
    )
    markdown(
        "tldr",
        "## tl;dr\n\n"
        f"Build `{build_id}` is bound to verification `{verification_id}`. "
        f"The independent receipt reports **{checks['pass_count']:,} of "
        f"{checks['check_count']:,}** check groups passed with zero failures and "
        f"the authenticated quality bundle preserves **{len(gate_rows)} of 57** "
        "exact frozen gates as passing. This notebook is a data-quality audit, "
        "not evidence about a strategy result.",
    )
    markdown(
        "context_methods",
        "## Context & Methods\n\n"
        "The generator authenticated the reviewed policy and rules, verified "
        "pointer, independent check, canonical empty differences payload, primary "
        "manifest, independent output-hash digest, closed quality file set, all "
        "quality receipts, and the processed monthly-summary receipt. The embedded "
        "datasets are bounded aggregations of those authenticated sources.\n\n"
        "### Key Assumptions\n\n"
        "- Coverage uses summed present counts divided by summed long-eligible denominators.\n"
        "- Missing outcomes remain classified and are never zero-filled.\n"
        "- Availability is measured against the scheduled exit date.\n"
        "- Distribution medians are medians of monthly medians, not pooled quantiles.",
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
        "datasets = artifact['snapshot']['datasets']\n"
        "coverage = pd.DataFrame(datasets['coverage_by_year'])\n"
        "primary_coverage = pd.DataFrame(datasets['primary_coverage_by_year'])\n"
        "statuses = pd.DataFrame(datasets['status_summary'])\n"
        "outcome_missing = pd.DataFrame(datasets['outcome_missing_classification'])\n"
        "delayed = pd.DataFrame(datasets['delayed_availability_by_year'])\n"
        "distributions = pd.DataFrame(datasets['distribution_summary'])\n"
        "gates = pd.DataFrame(datasets['exact_quality_gates'])\n"
        f"BUILD_ID = {build_id!r}\n"
        f"VERIFICATION_ID = {verification_id!r}\n"
        "print({'build_id': BUILD_ID, 'verification_id': VERIFICATION_ID, "
        "'artifact': ARTIFACT_RELATIVE_PATH.as_posix()})",
    )
    markdown(
        "data",
        "## Data\n\n"
        f"The authenticated development store contains **{counts['audit_rows']:,} "
        f"audit rows**, **{counts['processed_rows']:,} processed long-eligible "
        f"rows**, **{counts['signal_dates']:,} signal dates**, and "
        f"**{counts['leg_lineage_rows']:,} return-leg lineage rows**. The notebook "
        "uses only the small reviewed datasets embedded in the report artifact.",
    )
    code(
        "inspect_data",
        "expected_sizes = {\n"
        f"    'coverage_by_year': {len(coverage_rows)},\n"
        f"    'primary_coverage_by_year': {len([row for row in coverage_rows if row['coverage_type'] == 'Primary label present'])},\n"
        f"    'status_summary': {len(status_rows)},\n"
        f"    'outcome_missing_classification': {len(outcome_rows)},\n"
        f"    'delayed_availability_by_year': {len(delayed_rows)},\n"
        f"    'distribution_summary': {len(distribution_rows)},\n"
        f"    'exact_quality_gates': {len(gate_rows)},\n"
        "}\n"
        "observed_sizes = {name: len(rows) for name, rows in datasets.items()}\n"
        "assert observed_sizes == expected_sizes\n"
        "assert coverage['coverage_rate'].between(0.0, 1.0).all()\n"
        "assert len(primary_coverage) == coverage['holding_year'].nunique()\n"
        "assert set(primary_coverage['coverage_type']) == {'Primary label present'}\n"
        "assert outcome_missing['share_of_long_eligible'].between(0.0, 1.0).all()\n"
        "assert delayed['delayed_share_of_valid_primary'].between(0.0, 1.0).all()\n"
        "display(pd.DataFrame([observed_sizes]))",
    )
    markdown(
        "results",
        "## Results\n\n"
        "The following bounded views reproduce the report evidence: annual "
        "coverage, dense statuses, explicit outcome-missing causes, delayed "
        "availability, target distributions, and every exact frozen gate.",
    )
    code(
        "coverage_results",
        "coverage_view = coverage.copy()\n"
        "coverage_view['coverage_pct'] = coverage_view['coverage_rate'].map(lambda x: f'{x:.2%}')\n"
        "display(coverage_view[[\n"
        "    'holding_year', 'coverage_type', 'present_count', 'missing_count',\n"
        "    'eligible_denominator', 'coverage_pct'\n"
        "]].sort_values(['holding_year', 'coverage_type']))",
    )
    code(
        "status_results",
        "assert set(statuses['status_dimension']) == {'outcome', 'primary', 'magnitude'}\n"
        "display(statuses[[\n"
        "    'status_dimension', 'status', 'row_count', 'share_of_audit',\n"
        "    'share_of_long_eligible'\n"
        "]].sort_values(['status_dimension', 'status']))\n"
        "display(outcome_missing[[\n"
        "    'classification_label', 'missing_count', 'share_of_missing_outcomes',\n"
        "    'share_of_long_eligible'\n"
        "]])",
    )
    code(
        "availability_results",
        "display(delayed[[\n"
        "    'holding_year', 'valid_primary_count', 'delayed_availability_count',\n"
        "    'delayed_share_of_valid_primary', 'maximum_month_availability_lag_days'\n"
        "]].sort_values('holding_year'))",
    )
    code(
        "distribution_results",
        "assert (distributions['finite_count'] + distributions['missing_count'] > 0).all()\n"
        "display(distributions.sort_values('target_name'))",
    )
    code(
        "gate_results",
        "assert len(gates) == 57\n"
        "assert gates['passed'].all()\n"
        "assert (gates['observed_count'] == gates['expected_count']).all()\n"
        "display(gates[[\n"
        "    'check_name', 'observed_count', 'expected_count', 'result',\n"
        "    'severity', 'details'\n"
        "]].sort_values('check_name'))",
    )
    markdown(
        "takeaways",
        "## Takeaways\n\n"
        "1. The report and notebook are bound to one dynamic Stage 7 build and verification ID.\n"
        "2. Coverage, missing causes, and availability delays remain explicit rather than being treated as observed zero outcomes.\n"
        "3. All 57 published frozen gates match their expected integer values, and the independent verification bundle reports zero differences.\n"
        "4. The verified boundary permits Stage 8 risk and cost work only; later analytical stages remain separately gated.",
    )
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "stage_07_binding": {
                "build_id": build_id,
                "verification_id": verification_id,
                "artifact_path": artifact_relative_path.as_posix(),
            },
        },
    )


def _validate_output_path(path: Path, *, project_root: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    _safe_relative(lexical, project_root, label)
    relative = lexical.relative_to(project_root)
    current = project_root
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            _require(not current.is_symlink(), f"Symlink prohibited for {label}")
    return lexical


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
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _execute_notebook(
    notebook: nbf.NotebookNode,
    *,
    project_root: Path,
    artifact_relative_path: PurePosixPath,
    artifact_payload: bytes,
) -> nbf.NotebookNode:
    try:
        from nbclient import NotebookClient
    except ImportError as exc:
        raise Stage7AuditArtifactError(
            "Notebook execution requires the research dependency nbclient"
        ) from exc
    with tempfile.TemporaryDirectory(
        prefix=".stage7-audit-exec-", dir=project_root
    ) as name:
        execution_root = Path(name)
        staged_artifact = execution_root / Path(artifact_relative_path.as_posix())
        staged_artifact.parent.mkdir(parents=True, exist_ok=True)
        staged_artifact.write_bytes(artifact_payload)
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": execution_root.as_posix()}},
        )
        return client.execute()


def generate_stage7_audit_artifacts(
    *,
    verified_pointer_path: str | Path,
    independent_check_path: str | Path,
    primary_manifest_path: str | Path,
    quality_directory: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    database_root: str | Path | None = None,
    policy_path: str | Path = DEFAULT_POLICY,
    reviewed_policy_sha256: str = REVIEWED_POLICY_SHA256,
    processed_directory: str | Path | None = None,
    notebook_path: str | Path | None = None,
    report_directory: str | Path | None = None,
    execute_notebook: bool = True,
) -> Stage7AuditArtifactReceipt:
    """Generate verification-bound Stage 7 notebook and report payload.

    Every source artifact is authenticated before either requested output is
    published.  The function never creates HTML.
    """

    project = _real_directory(project_root, "project root")
    notebook_output = _validate_output_path(
        Path(
            notebook_path
            if notebook_path is not None
            else project / "notebooks/07_stage_07_label_store_quality.ipynb"
        ),
        project_root=project,
        label="notebook output",
    )
    report_dir = _validate_output_path(
        Path(
            report_directory
            if report_directory is not None
            else project / "reports/stage_07_label_store"
        ),
        project_root=project,
        label="report output",
    )
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
        reviewed_policy_sha256=reviewed_policy_sha256,
        processed_directory=processed_directory,
    )
    coverage_rows = _coverage_rows(evidence)
    status_rows, outcome_rows = _status_rows(evidence)
    delayed_rows = _delayed_rows(evidence)
    distribution_rows = _distribution_rows(evidence)
    gate_rows = _gate_rows(evidence)
    artifact = _artifact_payload(
        evidence,
        coverage_rows=coverage_rows,
        status_rows=status_rows,
        outcome_rows=outcome_rows,
        delayed_rows=delayed_rows,
        distribution_rows=distribution_rows,
        gate_rows=gate_rows,
    )
    artifact_bytes = canonical_json_bytes(artifact)
    notebook = _notebook(
        evidence=evidence,
        artifact_relative_path=artifact_relative,
        coverage_rows=coverage_rows,
        status_rows=status_rows,
        outcome_rows=outcome_rows,
        delayed_rows=delayed_rows,
        distribution_rows=distribution_rows,
        gate_rows=gate_rows,
    )
    if execute_notebook:
        notebook = _execute_notebook(
            notebook,
            project_root=project,
            artifact_relative_path=artifact_relative,
            artifact_payload=artifact_bytes,
        )
    notebook_bytes = nbf.writes(notebook, version=4).encode("utf-8")

    _atomic_write(artifact_path, artifact_bytes)
    _atomic_write(notebook_output, notebook_bytes)
    return Stage7AuditArtifactReceipt(
        build_id=str(evidence.pointer["build_id"]),
        verification_id=str(evidence.pointer["verification_id"]),
        notebook_path=notebook_output,
        artifact_path=artifact_path,
        notebook_sha256=_sha256_file(notebook_output),
        artifact_sha256=_sha256_file(artifact_path),
        coverage_rows=len(coverage_rows),
        status_rows=len(status_rows),
        outcome_missing_rows=len(outcome_rows),
        delayed_rows=len(delayed_rows),
        distribution_rows=len(distribution_rows),
        gate_rows=len(gate_rows),
        notebook_executed=execute_notebook,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the verification-bound Stage 7 label-store quality notebook "
            "and canonical report artifact. No HTML is generated."
        )
    )
    parser.add_argument("--verified-pointer", required=True, type=Path)
    parser.add_argument("--independent-check", required=True, type=Path)
    parser.add_argument("--primary-manifest", required=True, type=Path)
    parser.add_argument("--quality-directory", required=True, type=Path)
    parser.add_argument("--processed-directory", type=Path)
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
    receipt = generate_stage7_audit_artifacts(
        verified_pointer_path=args.verified_pointer,
        independent_check_path=args.independent_check,
        primary_manifest_path=args.primary_manifest,
        quality_directory=args.quality_directory,
        processed_directory=args.processed_directory,
        project_root=args.project_root,
        database_root=args.database_root,
        policy_path=args.policy,
        notebook_path=args.notebook,
        report_directory=args.report_directory,
        execute_notebook=not args.no_execute,
    )
    print(json.dumps(receipt.as_dict(), sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_NOTEBOOK",
    "DEFAULT_REPORT_DIRECTORY",
    "REVIEWED_POLICY_SHA256",
    "Stage7AuditArtifactError",
    "Stage7AuditArtifactReceipt",
    "generate_stage7_audit_artifacts",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
