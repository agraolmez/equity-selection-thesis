"""Immutable publication for the independent typed-backbone verification."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from portfolio_bot.data import secure_publication as secure


SCHEMA_VERSION = 1
PASS_STATUS = "verified_passed"
FAIL_STATUS = "verification_failed"
VERIFICATION_SUBDIRECTORY = "crsp_typed_backbone_verifications"
VERIFIED_POINTER = "LATEST_CRSP_TYPED_BACKBONE_VERIFIED.json"
VERIFICATION_FILES = ("differences.csv", "independent_check.json")
DIFFERENCE_COLUMNS = (
    "check_id", "category", "expected", "actual", "evidence", "message",
)
VERIFIER_IMPLEMENTATION_LABELS = (
    "scripts/verify_phase_02_crsp_typed_backbone.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_independent_scan.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_verification_publication.py",
    "src/portfolio_bot/data/verify_phase_02_crsp_typed_backbone.py",
    "src/portfolio_bot/data/secure_publication.py",
)
_VERIFICATION_DOMAIN = b"phase02-crsp-typed-independent-verification-v1\x00"
_IMPLEMENTATION_DOMAIN = b"phase02-crsp-typed-verifier-implementation-v1\x00"


class TypedVerificationPublicationError(RuntimeError):
    """Raised when independent evidence cannot be published safely."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypedVerificationPublicationError("Value is not canonical JSON") from exc
    return (rendered + "\n").encode("utf-8")


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, bool)) or value is None:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return value


def render_differences_csv(report: Mapping[str, Any]) -> bytes:
    rows = [check for check in report["checks"] if check["status"] != "PASS"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=DIFFERENCE_COLUMNS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for check in rows:
        writer.writerow({field: _csv_cell(check[field]) for field in DIFFERENCE_COLUMNS})
    return buffer.getvalue().encode("utf-8")


def implementation_bundle_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_IMPLEMENTATION_DOMAIN)
    for label in VERIFIER_IMPLEMENTATION_LABELS:
        path = project_root / label
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise TypedVerificationPublicationError(f"Cannot read verifier component {label}") from exc
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def bind_verification_id(report_without_id: Mapping[str, Any], differences: bytes) -> dict[str, Any]:
    if "verification_id" in report_without_id:
        raise TypedVerificationPublicationError("Unbound report already contains verification_id")
    digest = hashlib.sha256()
    digest.update(_VERIFICATION_DOMAIN)
    digest.update(canonical_json_bytes(report_without_id))
    digest.update(differences)
    report = dict(report_without_id)
    report["verification_id"] = f"verify-{digest.hexdigest()[:16]}"
    return report


def _validate_report(report: Mapping[str, Any], differences: bytes, project_root: Path) -> None:
    required = {
        "authorities", "authorizations", "build_id", "checks", "contract_id", "independence",
        "primary_manifest_sha256", "schema_version", "status", "summary",
        "verification_id", "verifier_implementation_sha256",
    }
    if set(report) != required:
        raise TypedVerificationPublicationError("Independent report keys differ")
    checks = report["checks"]
    if not isinstance(checks, list) or not checks:
        raise TypedVerificationPublicationError("Independent report checks are empty")
    ids = [check.get("check_id") for check in checks]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise TypedVerificationPublicationError("Independent check IDs must be sorted and unique")
    check_keys = {
        "actual", "category", "check_id", "evidence", "expected", "message", "status",
    }
    if any(set(check) != check_keys or check["status"] not in {"PASS", "FAIL"} for check in checks):
        raise TypedVerificationPublicationError("Independent check schema differs")
    fail_count = sum(check["status"] == "FAIL" for check in checks)
    expected_status = PASS_STATUS if fail_count == 0 else FAIL_STATUS
    if report["status"] != expected_status:
        raise TypedVerificationPublicationError("Independent report status differs from checks")
    summary = report["summary"]
    if set(summary) != {
        "check_count", "difference_count", "fail_count", "pass_count",
        "raw_rows_reread", "typed_rows_compared", "typed_values_compared",
    } or any(type(summary[name]) is not int or summary[name] < 0 for name in summary):
        raise TypedVerificationPublicationError("Independent summary schema differs")
    if (
        summary["check_count"] != len(checks)
        or summary["difference_count"] != fail_count
        or summary["fail_count"] != fail_count
        or summary["pass_count"] != len(checks) - fail_count
    ):
        raise TypedVerificationPublicationError("Independent summary arithmetic differs")
    if render_differences_csv(report) != differences:
        raise TypedVerificationPublicationError("Differences bytes are not canonical for report")
    authorizations = report["authorizations"]
    before_keys = {
        "typed_market_backbone_authorized", "universe_unit_inputs_authorized",
        "eligibility_authorized", "return_history_authorized",
        "beta_estimation_authorized", "labels_authorized", "targets_authorized",
        "features_authorized", "models_authorized", "portfolios_authorized",
        "pnl_authorized", "performance_authorized", "backtests_authorized",
        "validation_or_final_test_access_authorized",
    }
    if expected_status == FAIL_STATUS:
        if set(authorizations) != before_keys or any(
            value is not False for value in authorizations.values()
        ):
            raise TypedVerificationPublicationError("Failed verification opened an authorization")
    else:
        if set(authorizations) != before_keys | {"remaining_blockers"}:
            raise TypedVerificationPublicationError("Passed authorization schema differs")
        if (
            authorizations["typed_market_backbone_authorized"] is not True
            or authorizations["universe_unit_inputs_authorized"] is not True
            or authorizations["eligibility_authorized"] is not False
            or not isinstance(authorizations["remaining_blockers"], list)
            or not authorizations["remaining_blockers"]
        ):
            raise TypedVerificationPublicationError("Passed authorization boundary differs")
    forbidden = (
        "return_history_authorized", "beta_estimation_authorized", "labels_authorized",
        "targets_authorized", "features_authorized", "models_authorized",
        "portfolios_authorized", "pnl_authorized", "performance_authorized",
        "backtests_authorized", "validation_or_final_test_access_authorized",
    )
    if any(authorizations.get(name) is not False for name in forbidden):
        raise TypedVerificationPublicationError("A prohibited downstream gate is open")
    independence = report["independence"]
    if set(independence) != {
        "full_raw_hashes_recomputed",
        "primary_builder_core_transforms_or_publisher_imported",
        "raw_files_reread", "six_parquet_tables_fully_scanned",
    }:
        raise TypedVerificationPublicationError("Independence evidence schema differs")
    if (
        independence["primary_builder_core_transforms_or_publisher_imported"] is not False
        or type(independence["raw_files_reread"]) is not int
        or not 0 <= independence["raw_files_reread"] <= 4
        or type(independence["full_raw_hashes_recomputed"]) is not bool
        or type(independence["six_parquet_tables_fully_scanned"]) is not bool
    ):
        raise TypedVerificationPublicationError("Independence evidence differs")
    if expected_status == PASS_STATUS and independence != {
        "full_raw_hashes_recomputed": True,
        "primary_builder_core_transforms_or_publisher_imported": False,
        "raw_files_reread": 4,
        "six_parquet_tables_fully_scanned": True,
    }:
        raise TypedVerificationPublicationError("Passing independence evidence differs")
    digest = report["primary_manifest_sha256"]
    if (
        not isinstance(digest, str) or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise TypedVerificationPublicationError("Primary manifest SHA-256 differs")
    if report["schema_version"] != SCHEMA_VERSION:
        raise TypedVerificationPublicationError("Independent schema version differs")
    without_id = dict(report)
    supplied_id = without_id.pop("verification_id")
    expected = bind_verification_id(without_id, differences)["verification_id"]
    if supplied_id != expected:
        raise TypedVerificationPublicationError("verification_id differs from content binding")
    if report["verifier_implementation_sha256"] != implementation_bundle_sha256(project_root):
        raise TypedVerificationPublicationError("Verifier implementation binding differs")
    authorities = report["authorities"]
    if (
        set(authorities)
        != {"identity_empirical_lock_sha256", "policy_sha256", "schema_sha256"}
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in authorities.values()
        )
    ):
        raise TypedVerificationPublicationError("Frozen authority binding differs")


def publish_verification(
    report: Mapping[str, Any],
    differences: bytes,
    *,
    project_root: Path,
    artifact_root: Path,
    authority_gate: Callable[[], None] | None = None,
) -> dict[str, Path]:
    """Publish immutable evidence and move the verified pointer last on pass."""

    _validate_report(report, differences, project_root)
    report_bytes = canonical_json_bytes(report)
    destination = artifact_root / VERIFICATION_SUBDIRECTORY / report["verification_id"]
    payloads = {
        "differences.csv": differences,
        "independent_check.json": report_bytes,
    }
    with secure.SecureStagingDirectory.create(
        trusted_root=project_root,
        destination=destination,
        expected_files=VERIFICATION_FILES,
        label="typed-backbone independent verification bundle",
    ) as stage:
        for name in sorted(payloads):
            with stage.open_exclusive_binary(name) as stream:
                stream.write(payloads[name])
        snapshot = stage.snapshot()
        bundle = stage.publish(expected_hashes=snapshot.hashes)
    result = {"bundle": destination}
    if report["status"] != PASS_STATUS:
        return result

    pointer_payload = {
        "authorities": report["authorities"],
        "authorizations": report["authorizations"],
        "build_id": report["build_id"],
        "contract_id": report["contract_id"],
        "differences_sha256": bundle.snapshot.hashes["differences.csv"],
        "independent_check_sha256": bundle.snapshot.hashes["independent_check.json"],
        "primary_manifest_sha256": report["primary_manifest_sha256"],
        "schema_version": SCHEMA_VERSION,
        "status": PASS_STATUS,
        "verification_id": report["verification_id"],
        "verification_path": destination.relative_to(project_root).as_posix(),
    }
    pointer = artifact_root / VERIFIED_POINTER
    with secure.AuthenticatedBundleGuard.open(
        trusted_root=project_root,
        path=destination,
        expected_files=VERIFICATION_FILES,
        expected_hashes=bundle.snapshot.hashes,
        label="published typed-backbone verification bundle",
    ) as verifier_guard:
        def gate() -> None:
            verifier_guard.gate()
            if authority_gate is not None:
                authority_gate()

        secure.replace_pointer_last(
            pointer,
            canonical_json_bytes(pointer_payload),
            trusted_root=project_root,
            pre_replace_gate=gate,
            post_replace_gate=gate,
            label="typed-backbone verified pointer",
        )
    result["pointer"] = pointer
    return result


__all__ = [
    "FAIL_STATUS", "PASS_STATUS", "VERIFICATION_FILES", "VERIFIED_POINTER",
    "bind_verification_id", "canonical_json_bytes", "implementation_bundle_sha256",
    "publish_verification", "render_differences_csv",
]
