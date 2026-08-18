"""Explicit Phase 1 implementation and test-evidence contract.

The lists in this module are intentionally phase-specific.  Later phases may add
code and tests without silently changing the evidence inventory frozen for
Phase 1.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any


PHASE_01_IMPLEMENTATION_RELATIVE_PATHS = (
    "pyproject.toml",
    "configs/paths.toml",
    "configs/phase_01_raw_snapshot.toml",
    "configs/raw_contracts/compustat_ccm.toml",
    "configs/raw_contracts/crsp.toml",
    "configs/raw_contracts/factors_liquidity.toml",
    "configs/raw_contracts/optionmetrics.toml",
    "configs/raw_contracts/sec_restatements.toml",
    "src/portfolio_bot/__init__.py",
    "src/portfolio_bot/data/__init__.py",
    "src/portfolio_bot/data/phase_01_evidence.py",
    "src/portfolio_bot/data/phase_01_notebook.py",
    "src/portfolio_bot/data/raw_manifest.py",
    "src/portfolio_bot/data/verify_raw_manifest.py",
    "src/portfolio_bot/data/phase_01_review.py",
    "src/portfolio_bot/data/phase_01_handoff.py",
    "src/portfolio_bot/data/phase_01_freeze.py",
    "scripts/run_phase_01_manifest.py",
    "scripts/verify_phase_01_manifest.py",
    "scripts/build_phase_01_review.py",
    "scripts/build_phase_01_notebook.py",
    "scripts/execute_phase_01_notebook.py",
    "scripts/deliver_phase_01_report.py",
    "scripts/validate_phase_01_handoff.py",
    "scripts/run_phase_01_tests.py",
    "scripts/freeze_phase_01.py",
    "tests/integration/test_phase_01_review.py",
    "tests/unit/test_raw_manifest.py",
    "tests/unit/test_verify_raw_manifest.py",
    "tests/unit/test_phase_01_freeze.py",
)


# Update this exact registry only when a Phase 1 test is deliberately added or
# removed, then regenerate pytest.xml and validation_report.json.
PHASE_01_TEST_IDS = (
    "tests.integration.test_phase_01_review::test_phase_01_published_evidence_reconciles",
    "tests.integration.test_phase_01_review::test_phase_01_report_artifact_has_complete_technical_reading_path",
    "tests.integration.test_phase_01_review::test_raw_snapshot_pointer_records_verified_pre_freeze_state",
    "tests.unit.test_phase_01_freeze::test_final_raw_recheck_detects_early_file_changed_after_its_hash",
    "tests.unit.test_phase_01_freeze::test_final_raw_recheck_records_whole_run_stability",
    "tests.unit.test_phase_01_freeze::test_global_registry_is_idempotent_and_conflicts_fail",
    "tests.unit.test_phase_01_freeze::test_phase_01_freeze_identity_binds_semantics_and_all_groups",
    "tests.unit.test_phase_01_freeze::test_navigation_repair_is_idempotent_after_atomic_bundle_commit",
    "tests.unit.test_phase_01_freeze::test_publish_recovers_after_post_commit_navigation_failure",
    "tests.unit.test_phase_01_freeze::test_published_verifier_accepts_complete_bundle_and_rejects_semantic_tampering",
    "tests.unit.test_phase_01_freeze::test_published_verifier_rejects_missing_required_groups",
    "tests.unit.test_phase_01_freeze::test_staged_bundle_mutation_is_rejected_before_commit",
    "tests.unit.test_raw_manifest::test_byte_scan_is_exact",
    "tests.unit.test_raw_manifest::test_cross_pass_file_replacement_is_a_hard_failure",
    "tests.unit.test_raw_manifest::test_date_accumulator_partitions_every_row_and_rejects_invalid_times",
    "tests.unit.test_raw_manifest::test_frozen_registry_contains_exactly_25_unique_files",
    "tests.unit.test_raw_manifest::test_malformed_row_evidence_is_hashed_not_copied",
    "tests.unit.test_raw_manifest::test_only_passing_complete_build_publishes_and_builds_are_immutable",
    "tests.unit.test_raw_manifest::test_streaming_profile_handles_quoted_newline_and_open_end_sentinel",
    "tests.unit.test_verify_raw_manifest::test_independent_date_parser_requires_complete_supported_format",
    "tests.unit.test_verify_raw_manifest::test_pandas_verifier_counts_quoted_newline_as_one_record",
    "tests.unit.test_verify_raw_manifest::test_verification_publication_and_failure_preserve_verified_pointer",
)


PHASE_01_HANDOFF_CHECK_IDS = (
    "PUBLISHED_EVIDENCE_RECONCILES",
    "REPORT_READING_PATH_COMPLETE",
    "REPORT_MATCHES_CURRENT_VERIFIED_CONTEXT",
    "REPORT_SQL_REPRODUCES_SNAPSHOT",
    "REPORT_BROWSER_QA_PASSED",
    "REPORT_SELF_CONTAINED_MARKERS_PRESENT",
    "NOTEBOOK_EXECUTED_TOP_TO_BOTTOM",
    "NOTEBOOK_EXECUTION_BINDS_IMPLEMENTATION",
    "AUTOMATED_TEST_SUITE_PASSED",
    "TEST_EXECUTION_BINDS_IMPLEMENTATION",
)


def phase_01_implementation_paths(project_root: Path) -> list[Path]:
    """Return the exact, future-proof Phase 1 implementation inventory."""
    root = project_root.resolve()
    return [root / relative for relative in PHASE_01_IMPLEMENTATION_RELATIVE_PATHS]


def _stat_identity(observed: os.stat_result) -> dict[str, int]:
    """Return the metadata needed to prove one path and open file agree."""
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": observed.st_mode,
        "size_bytes": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
    }


def _observe_stable_file(
    path: Path,
    *,
    capture_bytes: bool,
    chunk_size: int = 4 << 20,
) -> tuple[str, dict[str, int], bytes | None]:
    """Read one descriptor while proving its pathname identity is unchanged.

    The pathname is checked with ``lstat`` before and after reading, while the
    opened descriptor is checked with ``fstat`` before and after reading.  All
    four identities must match.  ``O_NOFOLLOW`` closes the pathname-symlink
    race at open time on platforms that provide it; the lstat/fstat equality is
    retained as an independent guard.
    """
    candidate = Path(path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    path_before = os.lstat(candidate)
    if stat.S_ISLNK(path_before.st_mode):
        raise RuntimeError(f"Evidence path must not be a symlink: {candidate}")
    if not stat.S_ISREG(path_before.st_mode):
        raise RuntimeError(f"Evidence path must be a regular file: {candidate}")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        descriptor_before = os.fstat(descriptor)
        before_identity = _stat_identity(path_before)
        if _stat_identity(descriptor_before) != before_identity:
            raise RuntimeError(
                f"Evidence path changed between lstat and open: {candidate}"
            )

        digest = hashlib.sha256()
        captured = bytearray() if capture_bytes else None
        bytes_read = 0
        while chunk := os.read(descriptor, chunk_size):
            digest.update(chunk)
            bytes_read += len(chunk)
            if captured is not None:
                captured.extend(chunk)

        descriptor_after = os.fstat(descriptor)
        path_after = os.lstat(candidate)
        observed_identities = (
            before_identity,
            _stat_identity(descriptor_before),
            _stat_identity(descriptor_after),
            _stat_identity(path_after),
        )
        if any(item != before_identity for item in observed_identities[1:]):
            raise RuntimeError(f"Evidence changed while hashing: {candidate}")
        if bytes_read != before_identity["size_bytes"]:
            raise RuntimeError(
                f"Evidence byte count changed while hashing: {candidate}"
            )
    finally:
        os.close(descriptor)

    return (
        digest.hexdigest(),
        before_identity,
        bytes(captured) if captured is not None else None,
    )


def stable_file_receipt(
    path: Path,
    *,
    display_path: str | None = None,
    include_identity: bool = False,
    chunk_size: int = 4 << 20,
) -> dict[str, Any]:
    """Hash one stable non-symlink regular file from one checked descriptor."""
    digest, identity, _ = _observe_stable_file(
        path,
        capture_bytes=False,
        chunk_size=chunk_size,
    )

    receipt: dict[str, Any] = {
        "path": display_path if display_path is not None else str(Path(path)),
        "size_bytes": identity["size_bytes"],
        "sha256": digest,
    }
    if include_identity:
        receipt["identity"] = identity
    return receipt


def stable_file_bytes_receipt(
    path: Path,
    *,
    display_path: str | None = None,
    include_identity: bool = False,
    chunk_size: int = 4 << 20,
) -> tuple[bytes, dict[str, Any]]:
    """Return bytes and their receipt from the same checked descriptor read."""
    digest, identity, payload = _observe_stable_file(
        path,
        capture_bytes=True,
        chunk_size=chunk_size,
    )
    if payload is None:  # Defensive: capture_bytes=True must always return bytes.
        raise RuntimeError(f"Stable evidence capture failed: {path}")
    receipt: dict[str, Any] = {
        "path": display_path if display_path is not None else str(Path(path)),
        "size_bytes": identity["size_bytes"],
        "sha256": digest,
    }
    if include_identity:
        receipt["identity"] = identity
    return payload, receipt


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    """Return a stable SHA-256 digest for a non-symlink regular file."""
    return str(stable_file_receipt(path, chunk_size=chunk_size)["sha256"])


def implementation_receipt(
    project_root: Path, *, include_identity: bool = False
) -> list[dict[str, Any]]:
    """Hash every required Phase 1 implementation file for a test receipt."""
    root = project_root.resolve()
    paths = phase_01_implementation_paths(root)
    receipts: list[dict[str, Any]] = []
    for path in paths:
        try:
            relative = path.relative_to(root).as_posix()
            receipts.append(
                stable_file_receipt(
                    path,
                    display_path=relative,
                    include_identity=include_identity,
                )
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing Phase 1 implementation evidence: {path}"
            ) from exc
    return sorted(receipts, key=lambda item: item["path"])
