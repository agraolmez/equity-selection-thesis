"""Crash-safe publication and verification of the immutable Phase 1 bundle."""

from __future__ import annotations

import csv
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
import uuid
import xml.etree.ElementTree as et
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from portfolio_bot.data.phase_01_evidence import (
    PHASE_01_HANDOFF_CHECK_IDS,
    PHASE_01_TEST_IDS,
    phase_01_implementation_paths,
)
from portfolio_bot.data.phase_01_handoff import validate_handoff
from portfolio_bot.data.phase_01_review import (
    PROJECT_ROOT,
    load_review_context,
    validation_failures,
)
from portfolio_bot.data.raw_manifest import load_settings, verify_methodology_freeze


PRIMARY_FILES = (
    "columns.csv",
    "date_coverage.csv",
    "files.csv",
    "hashes.sha256",
    "key_diagnostics.csv",
    "manifest.json",
    "quality_checks.csv",
    "run_metadata.json",
)
VERIFICATION_FILES = (
    "differences.csv",
    "independent_check.json",
    "shasum.sha256",
    "verification_run_metadata.json",
)
REQUIRED_EVIDENCE_GROUPS = (
    "methodology",
    "primary_build",
    "independent_verification",
    "review_and_validation",
    "implementation_and_configuration",
    "final_raw_recheck",
    "publication_records",
    "operational_evidence",
)
BUNDLE_FILES = (
    "manifest.json",
    "raw_freeze_recheck.sha256",
    "raw_freeze_recheck_metadata.json",
    "registry_entry.csv",
    "run_record.json",
)
PROHIBITIONS = (
    "Do not treat coverage dates as public-availability dates.",
    "Do not use current-vintage metadata in features.",
    "Do not claim canonical uniqueness, leakage safety, model skill, or portfolio performance from Phase 1.",
    "Do not access or tune on 2020-2025 final-test outcomes before the pre-final freeze.",
)
SEALED_PERIODS = {
    "development_2000_2014": "not_started",
    "validation_2015_2019": "not_started",
    "final_test_2020_2025": "untouched_and_sealed",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_hash(path: Path, capture: bool = False) -> tuple[str, int, tuple[int, ...], bytes | None]:
    """Hash a regular non-symlink file while proving pathname/FD stability."""
    before_path = path.lstat()
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise RuntimeError(f"Freeze evidence is not a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    chunks: list[bytes] | None = [] if capture else None
    digest = hashlib.sha256()
    try:
        before_fd = os.fstat(descriptor)
        if _fingerprint(before_fd) != _fingerprint(before_path):
            raise RuntimeError(f"Evidence pathname changed before hashing: {path}")
        while True:
            chunk = os.read(descriptor, 4 << 20)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    identities = {
        _fingerprint(before_path),
        _fingerprint(before_fd),
        _fingerprint(after_fd),
        _fingerprint(after_path),
    }
    if len(identities) != 1:
        raise RuntimeError(f"Evidence changed while it was hashed: {path}")
    return (
        digest.hexdigest(),
        after_fd.st_size,
        _fingerprint(after_fd),
        b"".join(chunks) if chunks is not None else None,
    )


def _relative_path(project_root: Path, path: Path) -> str:
    root = project_root.resolve()
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Freeze evidence escapes the project root: {path}") from exc
    return relative.as_posix()


def _bound_relative_path(project_root: Path, path: Path) -> str:
    """Return a project-relative logical path that need not exist yet."""
    root = project_root.resolve()
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Freeze evidence escapes the project root: {path}") from exc
    return relative.as_posix()


def _stable_entry(project_root: Path, path: Path) -> tuple[dict[str, Any], tuple[int, ...]]:
    relative = _relative_path(project_root, path)
    digest, size, fingerprint, _ = _stable_hash(path)
    return {"path": relative, "size_bytes": size, "sha256": digest}, fingerprint


def _capture_snapshot(
    project_root: Path, paths: Iterable[Path]
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, ...]]]:
    materialized = sorted(list(paths), key=lambda item: item.as_posix())
    if len({str(path.resolve(strict=False)) for path in materialized}) != len(materialized):
        raise RuntimeError("Freeze evidence inventory contains duplicate paths")
    missing = [path for path in materialized if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required freeze evidence is missing: "
            + ", ".join(str(path) for path in missing)
        )
    entries: list[dict[str, Any]] = []
    fingerprints: dict[str, tuple[int, ...]] = {}
    for path in materialized:
        entry, fingerprint = _stable_entry(project_root, path)
        entries.append(entry)
        fingerprints[entry["path"]] = fingerprint
    _assert_fingerprints(project_root, fingerprints)
    return entries, fingerprints


def _assert_fingerprints(
    project_root: Path, fingerprints: dict[str, tuple[int, ...]]
) -> None:
    for relative, expected in sorted(fingerprints.items()):
        path = project_root / relative
        try:
            observed = _fingerprint(path.lstat())
        except OSError as exc:
            raise RuntimeError(f"Evidence disappeared after hashing: {relative}") from exc
        if observed != expected:
            raise RuntimeError(f"Evidence changed after hashing: {relative}")


def _stable_json(path: Path) -> tuple[dict[str, Any], str]:
    digest, _, _, content = _stable_hash(path, capture=True)
    assert content is not None
    return json.loads(content.decode("utf-8")), digest


def _stable_json_entry(
    project_root: Path,
    path: Path,
) -> tuple[Any, dict[str, Any], tuple[int, ...]]:
    """Parse JSON and construct its receipt from one stable captured read."""
    relative = _relative_path(project_root, path)
    digest, size, fingerprint, content = _stable_hash(path, capture=True)
    assert content is not None
    return (
        json.loads(content.decode("utf-8")),
        {"path": relative, "size_bytes": size, "sha256": digest},
        fingerprint,
    )


def _write_bytes_fsync(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_fsync(path: Path, payload: Any) -> None:
    _write_bytes_fsync(path, _canonical_json_bytes(payload))


def _write_mutable_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    temporary.write_bytes(_canonical_json_bytes(payload))
    os.replace(temporary, path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _publication_lock(freeze_directory: Path):
    freeze_directory.mkdir(parents=True, exist_ok=True)
    lock_path = freeze_directory / ".phase_01_freeze.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _shasum(path: Path) -> str:
    completed = subprocess.run(
        ["/usr/bin/shasum", "-a", "256", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.split()[0]


def _raw_registry(settings: Any) -> list[str]:
    return sorted(path.name for path in settings.raw_root.glob("*.csv") if path.is_file())


def _stat_payload(fingerprint: tuple[int, ...]) -> dict[str, int]:
    return {
        "device": fingerprint[0],
        "inode": fingerprint[1],
        "mode": fingerprint[2],
        "size_bytes": fingerprint[3],
        "mtime_ns": fingerprint[4],
        "ctime_ns": fingerprint[5],
    }


def _identity_receipt_valid(receipt: Any) -> bool:
    identity_keys = {
        "device",
        "inode",
        "mode",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
    }
    return (
        isinstance(receipt, dict)
        and set(receipt) == {"path", "size_bytes", "sha256", "identity"}
        and isinstance(receipt.get("path"), str)
        and type(receipt.get("size_bytes")) is int
        and receipt["size_bytes"] >= 0
        and isinstance(receipt.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"]) is not None
        and isinstance(receipt.get("identity"), dict)
        and set(receipt["identity"]) == identity_keys
        and all(type(receipt["identity"][key]) is int for key in identity_keys)
        and receipt["identity"]["size_bytes"] == receipt["size_bytes"]
    )


def final_raw_recheck(
    project_root: Path,
    output_directory: Path,
    expected_hashes: dict[str, str],
) -> tuple[Path, Path, dict[str, tuple[int, ...]]]:
    """Rehash all raw files and prove every file remains stable through the run."""
    del project_root  # Raw storage is deliberately outside the project workspace.
    settings = load_settings()
    expected_files = sorted(expected_hashes)
    before_registry = _raw_registry(settings)
    if before_registry != expected_files:
        raise RuntimeError(
            f"Raw registry changed before freeze: {before_registry} != {expected_files}"
        )

    started = dt.datetime.now(dt.timezone.utc)
    observed_hashes: dict[str, str] = {}
    after_fingerprints: dict[str, tuple[int, ...]] = {}
    file_metadata: dict[str, Any] = {}
    for index, filename in enumerate(expected_files, start=1):
        path = settings.raw_root / filename
        raw_stat = path.lstat()
        if stat.S_ISLNK(raw_stat.st_mode) or not stat.S_ISREG(raw_stat.st_mode):
            raise RuntimeError(f"Raw file is not a regular non-symlink file: {filename}")
        before = _fingerprint(raw_stat)
        print(
            json.dumps(
                {
                    "event": "phase_01_freeze_raw_recheck_start",
                    "file": filename,
                    "position": index,
                    "total": len(expected_files),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        observed = _shasum(path)
        after = _fingerprint(path.lstat())
        observed_hashes[filename] = observed
        after_fingerprints[filename] = after
        file_metadata[filename] = {
            "stat_before": _stat_payload(before),
            "stat_after_hash": _stat_payload(after),
        }
        status = (
            "passed"
            if before == after and observed == expected_hashes[filename]
            else "failed"
        )
        print(
            json.dumps(
                {
                    "event": "phase_01_freeze_raw_recheck_complete",
                    "file": filename,
                    "position": index,
                    "total": len(expected_files),
                    "status": status,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    after_registry = _raw_registry(settings)
    changed_hashes = {
        filename: {
            "expected": expected_hashes[filename],
            "observed": observed_hashes.get(filename),
        }
        for filename in expected_files
        if observed_hashes.get(filename) != expected_hashes[filename]
    }
    unstable_during_hash: list[str] = []
    unstable_through_run: list[str] = []
    final_fingerprints: dict[str, tuple[int, ...]] = {}
    for filename in expected_files:
        path = settings.raw_root / filename
        final = _fingerprint(path.lstat())
        final_fingerprints[filename] = final
        metadata = file_metadata[filename]
        before_tuple = (
            metadata["stat_before"]["device"],
            metadata["stat_before"]["inode"],
            metadata["stat_before"]["mode"],
            metadata["stat_before"]["size_bytes"],
            metadata["stat_before"]["mtime_ns"],
            metadata["stat_before"]["ctime_ns"],
        )
        if before_tuple != after_fingerprints[filename]:
            unstable_during_hash.append(filename)
        if final != after_fingerprints[filename]:
            unstable_through_run.append(filename)
        metadata["end_of_complete_recheck_stat"] = _stat_payload(final)
        metadata["stable_during_own_hash"] = before_tuple == after_fingerprints[filename]
        metadata["stable_through_entire_recheck"] = final == after_fingerprints[filename]

    if (
        after_registry != expected_files
        or changed_hashes
        or unstable_during_hash
        or unstable_through_run
    ):
        raise RuntimeError(
            "Raw files changed before freeze; "
            f"registry_after={after_registry}, hash_differences={changed_hashes}, "
            f"unstable_during_hash={unstable_during_hash}, "
            f"unstable_through_run={unstable_through_run}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    registry_path = output_directory / "raw_freeze_recheck.sha256"
    registry_bytes = "".join(
        f"{observed_hashes[name]}  {name}\n" for name in expected_files
    ).encode("utf-8")
    _write_bytes_fsync(registry_path, registry_bytes)
    completed = dt.datetime.now(dt.timezone.utc)
    metadata_path = output_directory / "raw_freeze_recheck_metadata.json"
    _write_json_fsync(
        metadata_path,
        {
            "schema_version": 2,
            "status": "passed",
            "hash_engine": "/usr/bin/shasum -a 256",
            "file_count": len(expected_files),
            "run_started_utc": started.isoformat(),
            "run_completed_utc": completed.isoformat(),
            "elapsed_seconds": round((completed - started).total_seconds(), 3),
            "raw_root": str(settings.raw_root),
            "raw_registry_before": before_registry,
            "raw_registry_after": after_registry,
            "all_hashes_match_primary": True,
            "all_files_stable_during_own_hash": True,
            "all_files_stable_through_entire_recheck": True,
            "per_file_operational_metadata": file_metadata,
        },
    )
    return registry_path, metadata_path, final_fingerprints


def _assert_raw_snapshot(
    settings: Any,
    expected_files: list[str],
    fingerprints: dict[str, tuple[int, ...]],
) -> None:
    if _raw_registry(settings) != expected_files:
        raise RuntimeError("Raw registry changed after the complete freeze recheck")
    for filename in expected_files:
        if _fingerprint((settings.raw_root / filename).lstat()) != fingerprints[filename]:
            raise RuntimeError(f"Raw file changed after the complete freeze recheck: {filename}")


def _exact_directory_paths(directory: Path, names: Iterable[str]) -> list[Path]:
    expected = sorted(names)
    observed = sorted(path.name for path in directory.iterdir() if path.is_file())
    nonfiles = sorted(path.name for path in directory.iterdir() if not path.is_file())
    if observed != expected or nonfiles:
        raise RuntimeError(
            f"Immutable directory inventory changed for {directory}: "
            f"files={observed}, expected={expected}, nonfiles={nonfiles}"
        )
    return [directory / name for name in expected]


def _methodology_paths(project_root: Path) -> list[Path]:
    freeze_path = project_root / "outputs" / "methodology_v1_freeze.json"
    payload, _ = _stable_json(freeze_path)
    declared = [str(item["path"]) for item in payload.get("files", [])]
    if not declared or len(set(declared)) != len(declared):
        raise RuntimeError("Methodology freeze has an empty or duplicate artifact registry")
    return [freeze_path, *(project_root / relative for relative in declared)]


def _review_paths(project_root: Path, build_id: str) -> list[Path]:
    report_directory = project_root / "reports" / "phase_01_raw_snapshot"
    run_directory = project_root / "runs" / "phase_01_raw_snapshot" / build_id
    return [
        report_directory / "artifact.json",
        report_directory / "phase_01_verification_report.html",
        report_directory / "phase_01_summary.json",
        report_directory / "source_family_summary.csv",
        report_directory / "source_file_summary.csv",
        report_directory / "phase_01_review.sqlite",
        report_directory / "source_notes.md",
        project_root / "notebooks" / "01_phase_01_raw_snapshot_review.ipynb",
        run_directory / "notebook_execution_receipt.json",
        run_directory / "pytest.xml",
        run_directory / "test_execution_receipt.json",
        run_directory / "report_delivery_receipt.json",
        run_directory / "validation_report.json",
    ]


def _external_group_paths(
    project_root: Path, context: dict[str, Any]
) -> dict[str, list[Path]]:
    build_id = context["summary"]["phase_01_build_id"]
    return {
        "methodology": _methodology_paths(project_root),
        "primary_build": _exact_directory_paths(
            context["primary_directory"], PRIMARY_FILES
        ),
        "independent_verification": _exact_directory_paths(
            context["verification_directory"], VERIFICATION_FILES
        ),
        "review_and_validation": _review_paths(project_root, build_id),
        "implementation_and_configuration": phase_01_implementation_paths(
            project_root
        ),
    }


def _flatten_groups(groups: dict[str, list[Path]]) -> list[Path]:
    return [path for group in groups.values() for path in group]


def _entries_for_paths(
    entries: list[dict[str, Any]], project_root: Path, paths: Iterable[Path]
) -> list[dict[str, Any]]:
    by_path = {entry["path"]: entry for entry in entries}
    selected = []
    for path in paths:
        relative = _relative_path(project_root, path)
        if relative not in by_path:
            raise RuntimeError(f"Captured evidence is missing required path: {relative}")
        selected.append(by_path[relative])
    return sorted(selected, key=lambda item: item["path"])


def _authenticated_state(project_root: Path) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    settings = load_settings()
    context = load_review_context(project_root)
    failures = validation_failures(context)
    if failures:
        raise RuntimeError("Phase 1 verified context failed: " + "; ".join(failures))

    artifact_root = settings.artifact_root
    latest_verified, latest_verified_sha = _stable_json(
        artifact_root / "LATEST_VERIFIED.json"
    )
    latest_primary, latest_primary_sha = _stable_json(
        artifact_root / "LATEST_PRIMARY.json"
    )
    summary = context["summary"]
    expected_verified = {
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "verification_id": summary["verification_id"],
        "primary_path": str(context["primary_directory"].relative_to(artifact_root)),
        "verification_path": str(
            context["verification_directory"].relative_to(artifact_root)
        ),
        "verification_status": "passed",
    }
    if latest_verified != expected_verified:
        raise RuntimeError("LATEST_VERIFIED does not exactly select the loaded context")
    if (
        latest_primary.get("phase_01_build_id") != summary["phase_01_build_id"]
        or latest_primary.get("raw_snapshot_id") != summary["raw_snapshot_id"]
        or latest_primary.get("path") != expected_verified["primary_path"]
    ):
        raise RuntimeError("LATEST_PRIMARY does not select the verified primary build")

    _exact_directory_paths(context["primary_directory"], PRIMARY_FILES)
    _exact_directory_paths(context["verification_directory"], VERIFICATION_FILES)
    manifest = context["manifest"]
    independent = context["independent"]
    methodology = verify_methodology_freeze(settings.methodology_freeze)
    if methodology != manifest.get("methodology_freeze"):
        raise RuntimeError("Current methodology freeze differs from the primary manifest")
    raw_manifest_hash, _, _, _ = _stable_hash(
        project_root / "src" / "portfolio_bot" / "data" / "raw_manifest.py"
    )
    verifier_hash, _, _, _ = _stable_hash(
        project_root / "src" / "portfolio_bot" / "data" / "verify_raw_manifest.py"
    )
    primary_manifest_hash, _, _, _ = _stable_hash(
        context["primary_directory"] / "manifest.json"
    )
    if raw_manifest_hash != manifest.get("implementation_sha256"):
        raise RuntimeError("Current primary implementation differs from the primary build")
    if raw_manifest_hash != independent.get("imported_primary_implementation_sha256"):
        raise RuntimeError("Current primary implementation differs from independent verification")
    if verifier_hash != independent.get("verifier_implementation_sha256"):
        raise RuntimeError("Current verifier implementation differs from independent verification")
    if primary_manifest_hash != independent.get("manifest_sha256"):
        raise RuntimeError("Primary manifest differs from the independently verified manifest")
    if settings.configuration_hash != manifest.get("configuration_hash"):
        raise RuntimeError("Current Phase 1 configuration differs from the primary build")
    if settings.configuration_hash != independent.get("configuration_hash"):
        raise RuntimeError("Current Phase 1 configuration differs from verification")
    if (
        settings.path_configuration_hash
        != context["primary_metadata"].get("path_configuration_sha256")
    ):
        raise RuntimeError("Current database-path configuration differs from the primary run")
    if independent.get("verification_status") != "passed" or independent.get(
        "difference_count"
    ) != 0:
        raise RuntimeError("Independent verification is not a zero-difference pass")
    if context["primary_hashes"] != context["verified_hashes"]:
        raise RuntimeError("Primary and independent raw hash registries differ")

    pin = {
        "latest_verified": latest_verified,
        "latest_verified_sha256": latest_verified_sha,
        "latest_primary": latest_primary,
        "latest_primary_sha256": latest_primary_sha,
        "primary_directory": str(context["primary_directory"].relative_to(project_root)),
        "verification_directory": str(
            context["verification_directory"].relative_to(project_root)
        ),
        "primary_manifest_sha256": primary_manifest_hash,
        "primary_implementation_sha256": raw_manifest_hash,
        "verifier_implementation_sha256": verifier_hash,
        "configuration_hash": settings.configuration_hash,
        "path_configuration_hash": settings.path_configuration_hash,
        "methodology_freeze": methodology,
    }
    return context, settings, pin


def _without_validation_time(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "validated_at_utc"}


def _authenticated_handoff(project_root: Path, build_id: str) -> dict[str, Any]:
    current = validate_handoff(project_root)
    if current.get("assessment") != "Ready to freeze" or current.get(
        "failed_check_count"
    ):
        raise RuntimeError("Phase 1 handoff is not ready to freeze")
    saved_path = (
        project_root
        / "runs"
        / "phase_01_raw_snapshot"
        / build_id
        / "validation_report.json"
    )
    saved, _ = _stable_json(saved_path)
    if _without_validation_time(saved) != _without_validation_time(current):
        raise RuntimeError(
            "Saved handoff validation is stale relative to current report, tests, or code"
        )
    return current


def _stage_entry(
    project_root: Path, staged_path: Path, final_path: Path
) -> dict[str, Any]:
    digest, size, _, _ = _stable_hash(staged_path)
    final_relative = final_path.resolve(strict=False).relative_to(
        project_root.resolve()
    )
    return {
        "path": final_relative.as_posix(),
        "size_bytes": size,
        "sha256": digest,
    }


def _verify_staged_bundle(
    project_root: Path,
    attempt: Path,
    final_bundle: Path,
    expected_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, ...]]]:
    """Authenticate every staged byte against its bound final-path entry."""
    observed_names = sorted(path.name for path in attempt.iterdir())
    if observed_names != sorted(BUNDLE_FILES):
        raise RuntimeError(
            f"Staged freeze bundle inventory mismatch: {observed_names}"
        )
    manifest, _ = _stable_json(attempt / "manifest.json")
    if manifest != expected_manifest:
        raise RuntimeError("Staged freeze manifest differs from the assembled manifest")
    groups = expected_manifest["identity"]["evidence_groups"]
    bound_by_name: dict[str, dict[str, Any]] = {}
    for group_name in (
        "final_raw_recheck",
        "publication_records",
        "operational_evidence",
    ):
        for entry in groups[group_name]:
            bound_by_name[Path(entry["path"]).name] = entry
    for name in BUNDLE_FILES:
        if name == "manifest.json":
            continue
        staged = attempt / name
        observed = _stage_entry(project_root, staged, final_bundle / name)
        if observed != bound_by_name.get(name):
            raise RuntimeError(f"Staged file differs from its bound identity entry: {name}")
    return _capture_snapshot(
        project_root, [attempt / name for name in sorted(BUNDLE_FILES)]
    )


def _registry_csv_bytes(row: dict[str, str]) -> bytes:
    fieldnames = list(row)
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def freeze_identity(
    raw_snapshot_id: str,
    build_id: str,
    verification_id: str,
    methodology_freeze_id: str,
    source_selection: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
    validation_assessment: dict[str, Any],
    scope_boundary: dict[str, Any],
    frozen_at_utc: str,
    runtime: dict[str, str],
) -> dict[str, Any]:
    """Return the complete semantic and immutable-evidence identity."""
    return {
        "schema_version": 1,
        "phase": "01_raw_snapshot",
        "status": "phase_01_frozen",
        "raw_snapshot_id": raw_snapshot_id,
        "phase_01_build_id": build_id,
        "verification_id": verification_id,
        "methodology_freeze_id": methodology_freeze_id,
        "source_selection": source_selection,
        "validation_assessment": validation_assessment,
        "scope_boundary": scope_boundary,
        "prohibitions_until_later_gates": list(PROHIBITIONS),
        "sealed_periods": SEALED_PERIODS,
        "next_permitted_phase": "02_canonical_point_in_time_panel",
        "frozen_at_utc": frozen_at_utc,
        "runtime": runtime,
        "evidence_groups": groups,
    }


def _run_record(
    project_root: Path,
    context: dict[str, Any],
    final_bundle: Path,
    created: str,
) -> dict[str, Any]:
    summary = context["summary"]
    return {
        "schema_version": 2,
        "phase": "01_raw_snapshot",
        "run_id": summary["phase_01_build_id"],
        "status": "phase_01_frozen",
        "created_utc": created,
        "methodology_freeze_id": context["manifest"]["methodology_freeze"]["freeze_id"],
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "verification_id": summary["verification_id"],
        "primary_build": str(context["primary_directory"].relative_to(project_root)),
        "independent_verification": str(
            context["verification_directory"].relative_to(project_root)
        ),
        "report": "reports/phase_01_raw_snapshot/phase_01_verification_report.html",
        "notebook": "notebooks/01_phase_01_raw_snapshot_review.ipynb",
        "validation": str(
            (
                project_root
                / "runs"
                / "phase_01_raw_snapshot"
                / summary["phase_01_build_id"]
                / "validation_report.json"
            ).relative_to(project_root)
        ),
        "freeze_manifest": str((final_bundle / "manifest.json").relative_to(project_root)),
        "sealed_periods": SEALED_PERIODS,
        "next_permitted_action": (
            "Begin Phase 2 canonical point-in-time panel contracts and implementation."
        ),
        "freeze_id_location": "manifest.json",
    }


def _immutable_registry_row(run_record: dict[str, Any]) -> dict[str, str]:
    return {
        "phase": run_record["phase"],
        "run_id": run_record["run_id"],
        "status": run_record["status"],
        "methodology_freeze_id": run_record["methodology_freeze_id"],
        "raw_snapshot_id": run_record["raw_snapshot_id"],
        "phase_01_build_id": run_record["phase_01_build_id"],
        "verification_id": run_record["verification_id"],
        "freeze_manifest": run_record["freeze_manifest"],
        "report": run_record["report"],
        "notebook": run_record["notebook"],
        "validation": run_record["validation"],
        "created_utc": run_record["created_utc"],
        "next_permitted_action": run_record["next_permitted_action"],
    }


def _navigation_row(identity: dict[str, Any], freeze_id: str, run_record: dict[str, Any]) -> dict[str, str]:
    row = _immutable_registry_row(run_record)
    row["freeze_id"] = freeze_id
    return row


def _write_global_registry(project_root: Path, row: dict[str, str]) -> Path:
    path = project_root / "runs" / "registry.csv"
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    matches = [item for item in rows if item.get("phase") == "01_raw_snapshot"]
    if matches:
        if len(matches) != 1 or matches[0] != row:
            raise RuntimeError("Global run registry contains a conflicting Phase 1 row")
        return path
    rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def _repair_navigation(
    project_root: Path,
    settings: Any,
    manifest_path: Path,
    payload: dict[str, Any],
) -> None:
    identity = payload["identity"]
    manifest_sha = _stable_hash(manifest_path)[0]
    latest_payload = {
        "raw_snapshot_id": identity["raw_snapshot_id"],
        "phase_01_build_id": identity["phase_01_build_id"],
        "verification_id": identity["verification_id"],
        "freeze_id": payload["freeze_id"],
        "path": str(manifest_path.relative_to(settings.artifact_root)),
        "status": "phase_01_frozen",
        "sha256": manifest_sha,
    }
    _write_mutable_json(settings.artifact_root / "LATEST_FROZEN.json", latest_payload)

    raw_pointer_path = project_root / "data" / "raw_snapshot.pointer.json"
    raw_pointer = {
        "database_root": str(settings.database_root),
        "raw_subdirectory": str(settings.raw_root.relative_to(settings.database_root)),
        "path_override_environment_variable": "PORTFOLIO_DATABASE_ROOT",
        "expected_csv_files": settings.expected_file_count,
        "snapshot_status": "phase_01_frozen",
        "methodology_freeze_id": identity["methodology_freeze_id"],
        "raw_snapshot_id": identity["raw_snapshot_id"],
        "phase_01_build_id": identity["phase_01_build_id"],
        "verification_id": identity["verification_id"],
        "freeze_id": payload["freeze_id"],
        "freeze_manifest": str(manifest_path.relative_to(project_root)),
        "freeze_manifest_sha256": manifest_sha,
    }
    _write_mutable_json(raw_pointer_path, raw_pointer)

    run_record = json.loads(
        (manifest_path.parent / "run_record.json").read_text(encoding="utf-8")
    )
    _write_global_registry(
        project_root,
        _navigation_row(identity, payload["freeze_id"], run_record),
    )


def _preflight_unfrozen_navigation(
    project_root: Path,
    settings: Any,
    methodology_freeze_id: str,
) -> dict[str, Any]:
    """Fail before the long hash if mutable navigation is stale or conflicting."""
    latest_frozen = settings.artifact_root / "LATEST_FROZEN.json"
    if latest_frozen.exists():
        raise RuntimeError("LATEST_FROZEN exists without its expected immutable bundle")
    registry_path = project_root / "runs" / "registry.csv"
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if any(row.get("phase") == "01_raw_snapshot" for row in rows):
            raise RuntimeError("Global registry already contains a Phase 1 row")
    pointer_path = project_root / "data" / "raw_snapshot.pointer.json"
    pointer, pointer_sha = _stable_json(pointer_path)
    expected_pointer = {
        "database_root": str(settings.database_root),
        "raw_subdirectory": str(settings.raw_root.relative_to(settings.database_root)),
        "path_override_environment_variable": "PORTFOLIO_DATABASE_ROOT",
        "expected_csv_files": settings.expected_file_count,
        "snapshot_status": "verified_pending_explicit_phase_freeze",
        "methodology_freeze_id": methodology_freeze_id,
    }
    if pointer != expected_pointer:
        raise RuntimeError("Pre-freeze raw snapshot pointer is not canonical")
    return {"payload": pointer, "sha256": pointer_sha}


def _read_hash_registry(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        filename = filename.lstrip("* ")
        if filename in values:
            raise RuntimeError(f"Duplicate hash-registry filename: {filename}")
        values[filename] = digest
    return values


def _canonical_hash_registry_bytes(values: dict[str, str]) -> bytes:
    return "".join(
        f"{values[filename]}  {filename}\n" for filename in sorted(values)
    ).encode("utf-8")


def _parse_utc_timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def publish_phase_01_freeze(project_root: Path = PROJECT_ROOT) -> Path:
    """Publish one complete immutable bundle, then repair mutable navigation."""
    project_root = project_root.resolve()
    initial_context, initial_settings, initial_pin = _authenticated_state(project_root)
    summary = initial_context["summary"]
    final_bundle = initial_settings.freeze_directory / summary["raw_snapshot_id"]
    manifest_path = final_bundle / "manifest.json"

    with _publication_lock(initial_settings.freeze_directory):
        if final_bundle.exists():
            failures = verify_published_phase_01_freeze(
                manifest_path, project_root, require_navigation=False
            )
            if failures:
                raise RuntimeError(
                    "Existing Phase 1 freeze bundle is invalid: " + "; ".join(failures)
                )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            _repair_navigation(project_root, initial_settings, manifest_path, payload)
            failures = verify_published_phase_01_freeze(manifest_path, project_root)
            if failures:
                raise RuntimeError(
                    "Existing Phase 1 freeze navigation could not be repaired: "
                    + "; ".join(failures)
                )
            return manifest_path

        build_id = summary["phase_01_build_id"]
        navigation_before = _preflight_unfrozen_navigation(
            project_root,
            initial_settings,
            initial_context["manifest"]["methodology_freeze"]["freeze_id"],
        )
        handoff_before = _authenticated_handoff(project_root, build_id)
        external_groups = _external_group_paths(project_root, initial_context)
        external_before, _ = _capture_snapshot(
            project_root, _flatten_groups(external_groups)
        )

        staging_root = initial_settings.freeze_directory / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        attempt = staging_root / (
            dt.datetime.now(dt.timezone.utc).strftime("freeze-%Y%m%dT%H%M%S%fZ")
            + f"-p{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        attempt.mkdir(parents=False, exist_ok=False)
        raw_registry_path, raw_metadata_path, raw_fingerprints = final_raw_recheck(
            project_root, attempt, initial_context["primary_hashes"]
        )

        current_context, current_settings, current_pin = _authenticated_state(project_root)
        if current_pin != initial_pin or current_context["summary"] != summary:
            raise RuntimeError("Verified source selection changed during the raw recheck")
        if current_settings.freeze_directory != initial_settings.freeze_directory:
            raise RuntimeError("Freeze directory configuration changed during the raw recheck")
        handoff_after = _authenticated_handoff(project_root, build_id)
        if _without_validation_time(handoff_after) != _without_validation_time(
            handoff_before
        ):
            raise RuntimeError("Handoff assessment changed during the raw recheck")
        external_after, external_fingerprints = _capture_snapshot(
            project_root, _flatten_groups(external_groups)
        )
        if external_after != external_before:
            raise RuntimeError("Phase 1 evidence changed during the raw recheck")

        created = dt.datetime.now(dt.timezone.utc).isoformat()
        run_record = _run_record(project_root, current_context, final_bundle, created)
        run_record_stage = attempt / "run_record.json"
        _write_json_fsync(run_record_stage, run_record)
        registry_row = _immutable_registry_row(run_record)
        registry_stage = attempt / "registry_entry.csv"
        _write_bytes_fsync(registry_stage, _registry_csv_bytes(registry_row))

        external_entry_map = {entry["path"]: entry for entry in external_after}
        groups: dict[str, list[dict[str, Any]]] = {}
        for group_name, group_paths in external_groups.items():
            groups[group_name] = _entries_for_paths(
                external_after, project_root, group_paths
            )
        groups["final_raw_recheck"] = [
            _stage_entry(
                project_root,
                raw_registry_path,
                final_bundle / raw_registry_path.name,
            )
        ]
        groups["publication_records"] = sorted(
            [
                _stage_entry(
                    project_root, run_record_stage, final_bundle / run_record_stage.name
                ),
                _stage_entry(
                    project_root, registry_stage, final_bundle / registry_stage.name
                ),
            ],
            key=lambda item: item["path"],
        )
        groups["operational_evidence"] = [
            _stage_entry(
                project_root,
                raw_metadata_path,
                final_bundle / raw_metadata_path.name,
            )
        ]
        if tuple(groups) != REQUIRED_EVIDENCE_GROUPS:
            raise RuntimeError("Freeze evidence groups were assembled out of contract order")
        if any(not entries for entries in groups.values()):
            raise RuntimeError("Freeze evidence groups may not be empty")

        validation_assessment = {
            "assessment": handoff_after["assessment"],
            "failed_check_count": handoff_after["failed_check_count"],
            "checks": [
                {"check_id": item["check_id"], "status": item["status"]}
                for item in handoff_after["checks"]
            ],
        }
        runtime = {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        identity = freeze_identity(
            summary["raw_snapshot_id"],
            summary["phase_01_build_id"],
            summary["verification_id"],
            current_context["manifest"]["methodology_freeze"]["freeze_id"],
            current_pin,
            groups,
            validation_assessment,
            current_context["manifest"]["scope_boundary"],
            created,
            runtime,
        )
        identity_sha = _sha256_bytes(_canonical_json_bytes(identity))
        freeze_id = "phase01-freeze-" + identity_sha[:16]
        freeze_manifest = {
            "schema_version": 2,
            "phase": "01_raw_snapshot",
            "status": "phase_01_frozen",
            "freeze_id": freeze_id,
            "identity_sha256": identity_sha,
            "identity": identity,
            "frozen_at_utc": created,
            "runtime": runtime,
        }
        _write_json_fsync(attempt / "manifest.json", freeze_manifest)
        staged_snapshot, _ = _verify_staged_bundle(
            project_root, attempt, final_bundle, freeze_manifest
        )
        _fsync_directory(attempt)

        # Final pre-commit gates: rehash all external evidence, repin source
        # pointers/methodology/configuration, and re-stat every raw file.
        external_precommit, _ = _capture_snapshot(
            project_root, _flatten_groups(external_groups)
        )
        if external_precommit != external_after:
            raise RuntimeError("Phase 1 evidence changed before atomic publication")
        final_context, final_settings, final_pin = _authenticated_state(project_root)
        if final_pin != initial_pin or final_context["summary"] != summary:
            raise RuntimeError("Verified source selection changed before publication")
        _authenticated_handoff(project_root, build_id)
        navigation_precommit = _preflight_unfrozen_navigation(
            project_root,
            final_settings,
            final_context["manifest"]["methodology_freeze"]["freeze_id"],
        )
        if navigation_precommit != navigation_before:
            raise RuntimeError("Pre-freeze navigation changed during publication")
        _assert_fingerprints(project_root, external_fingerprints)
        _assert_raw_snapshot(
            final_settings, sorted(initial_context["primary_hashes"]), raw_fingerprints
        )
        staged_precommit, staged_precommit_fingerprints = _verify_staged_bundle(
            project_root, attempt, final_bundle, freeze_manifest
        )
        if staged_precommit != staged_snapshot:
            raise RuntimeError("Staged freeze bundle changed before atomic publication")
        # Newly created files on macOS can receive the system-managed
        # ``com.apple.provenance`` xattr asynchronously.  That changes ctime
        # without changing any bound byte identity.  Refresh the operational
        # fingerprints only after the complete byte snapshot has independently
        # re-authenticated against the original snapshot.
        _assert_fingerprints(project_root, staged_precommit_fingerprints)

        # Exercise the complete published-bundle verifier against the staged
        # bytes while interpreting their bound paths as the final atomic
        # destination.  No semantically invalid bundle may cross the rename.
        staged_semantic_failures = verify_published_phase_01_freeze(
            attempt / "manifest.json",
            project_root,
            require_navigation=False,
            logical_bundle=final_bundle,
        )
        if staged_semantic_failures:
            raise RuntimeError(
                "Staged Phase 1 bundle failed full semantic verification: "
                + "; ".join(staged_semantic_failures)
            )
        _assert_raw_snapshot(
            final_settings, sorted(initial_context["primary_hashes"]), raw_fingerprints
        )
        staged_final, staged_final_fingerprints = _verify_staged_bundle(
            project_root, attempt, final_bundle, freeze_manifest
        )
        if staged_final != staged_snapshot:
            raise RuntimeError(
                "Staged freeze bundle changed during semantic verification"
            )
        _assert_fingerprints(project_root, staged_final_fingerprints)

        os.rename(attempt, final_bundle)
        _fsync_directory(initial_settings.freeze_directory)
        failures = verify_published_phase_01_freeze(
            manifest_path, project_root, require_navigation=False
        )
        if failures:
            raise RuntimeError(
                "Published immutable Phase 1 bundle failed verification: "
                + "; ".join(failures)
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        _repair_navigation(project_root, final_settings, manifest_path, payload)
        failures = verify_published_phase_01_freeze(manifest_path, project_root)
        if failures:
            raise RuntimeError(
                "Phase 1 navigation verification failed after publication: "
                + "; ".join(failures)
            )
        print(
            json.dumps(
                {
                    "event": "phase_01_freeze_published",
                    "freeze_id": freeze_id,
                    "raw_snapshot_id": summary["raw_snapshot_id"],
                    "phase_01_build_id": summary["phase_01_build_id"],
                    "verification_id": summary["verification_id"],
                    "path": str(manifest_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return manifest_path


def _expected_group_paths(
    project_root: Path, identity: dict[str, Any], bundle: Path
) -> dict[str, list[str]]:
    build_id = identity["phase_01_build_id"]
    verification_id = identity["verification_id"]
    artifact_root = project_root / "artifacts" / "phase_01_raw_manifest"
    primary = artifact_root / "builds" / build_id
    verification = artifact_root / "verifications" / build_id / verification_id
    context_like = {
        "summary": {"phase_01_build_id": build_id},
        "primary_directory": primary,
        "verification_directory": verification,
    }
    groups = _external_group_paths(project_root, context_like)
    result = {
        name: sorted(_relative_path(project_root, path) for path in paths)
        for name, paths in groups.items()
    }
    result["final_raw_recheck"] = [
        _bound_relative_path(project_root, bundle / "raw_freeze_recheck.sha256")
    ]
    result["publication_records"] = sorted(
        [
            _bound_relative_path(project_root, bundle / "registry_entry.csv"),
            _bound_relative_path(project_root, bundle / "run_record.json"),
        ]
    )
    result["operational_evidence"] = [
        _bound_relative_path(project_root, bundle / "raw_freeze_recheck_metadata.json")
    ]
    return result


def verify_published_phase_01_freeze(
    freeze_path: Path,
    project_root: Path = PROJECT_ROOT,
    require_navigation: bool = True,
    logical_bundle: Path | None = None,
) -> list[str]:
    """Fail closed on inventory, identity, cross-file, hash, and pointer drift."""
    project_root = project_root.resolve()
    failures: list[str] = []
    try:
        freeze_path = freeze_path.resolve(strict=True)
        bundle = freeze_path.parent
        logical_bundle = (
            bundle
            if logical_bundle is None
            else logical_bundle.resolve(strict=False)
        )
        _bound_relative_path(project_root, logical_bundle)
        if freeze_path.name != "manifest.json":
            failures.append("freeze path is not a bundle manifest.json")
        observed_bundle = sorted(path.name for path in bundle.iterdir())
        if observed_bundle != sorted(BUNDLE_FILES):
            failures.append(
                f"bundle inventory mismatch: {observed_bundle} != {sorted(BUNDLE_FILES)}"
            )
        payload, manifest_entry, manifest_fingerprint = _stable_json_entry(
            project_root, freeze_path
        )
        manifest_sha = manifest_entry["sha256"]
    except Exception as exc:
        return [f"freeze manifest could not be read stably: {type(exc).__name__}: {exc}"]
    if not isinstance(payload, dict):
        return failures + ["freeze manifest is not a JSON object"]
    verified_fingerprints: dict[str, tuple[int, ...]] = {
        manifest_entry["path"]: manifest_fingerprint
    }

    expected_top_keys = {
        "schema_version",
        "phase",
        "status",
        "freeze_id",
        "identity_sha256",
        "identity",
        "frozen_at_utc",
        "runtime",
    }
    if set(payload) != expected_top_keys:
        failures.append("freeze manifest top-level key inventory mismatch")
    if payload.get("schema_version") != 2:
        failures.append("freeze manifest schema is not 2")
    if payload.get("phase") != "01_raw_snapshot" or payload.get("status") != "phase_01_frozen":
        failures.append("freeze phase/status mismatch")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        return failures + ["freeze identity is not an object"]
    expected_identity_keys = {
        "schema_version",
        "phase",
        "status",
        "raw_snapshot_id",
        "phase_01_build_id",
        "verification_id",
        "methodology_freeze_id",
        "source_selection",
        "validation_assessment",
        "scope_boundary",
        "prohibitions_until_later_gates",
        "sealed_periods",
        "next_permitted_phase",
        "frozen_at_utc",
        "runtime",
        "evidence_groups",
    }
    if set(identity) != expected_identity_keys:
        failures.append("freeze identity key inventory mismatch")
    if identity.get("schema_version") != 1:
        failures.append("freeze identity schema is not 1")
    identity_sha = _sha256_bytes(_canonical_json_bytes(identity))
    if identity_sha != payload.get("identity_sha256"):
        failures.append("freeze identity hash mismatch")
    if "phase01-freeze-" + identity_sha[:16] != payload.get("freeze_id"):
        failures.append("freeze ID mismatch")
    if identity.get("phase") != payload.get("phase") or identity.get("status") != payload.get("status"):
        failures.append("manifest phase/status differs from bound identity")
    if identity.get("prohibitions_until_later_gates") != list(PROHIBITIONS):
        failures.append("bound prohibitions mismatch")
    if identity.get("sealed_periods") != SEALED_PERIODS:
        failures.append("bound sealed-period status mismatch")
    if identity.get("next_permitted_phase") != "02_canonical_point_in_time_panel":
        failures.append("bound next phase mismatch")
    if identity.get("frozen_at_utc") != payload.get("frozen_at_utc"):
        failures.append("manifest freeze timestamp differs from bound identity")
    if identity.get("runtime") != payload.get("runtime"):
        failures.append("manifest runtime differs from bound identity")

    identifier_fields = (
        "raw_snapshot_id",
        "phase_01_build_id",
        "verification_id",
    )
    safe_identifier = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    identifiers_are_safe = all(
        isinstance(identity.get(field), str)
        and identity.get(field) not in {".", ".."}
        and safe_identifier.fullmatch(identity[field]) is not None
        for field in identifier_fields
    )
    if not identifiers_are_safe:
        failures.append("freeze identity contains an unsafe identifier")

    groups = identity.get("evidence_groups")
    if not isinstance(groups, dict):
        return failures + ["evidence_groups is not an object"]
    if set(groups) != set(REQUIRED_EVIDENCE_GROUPS):
        failures.append("required evidence group inventory mismatch")
    try:
        if not identifiers_are_safe:
            raise ValueError("unsafe identifiers prohibit evidence-path construction")
        expected_paths = _expected_group_paths(project_root, identity, logical_bundle)
    except Exception as exc:
        failures.append(f"could not derive exact evidence inventory: {type(exc).__name__}: {exc}")
        expected_paths = {}

    seen_paths: set[str] = set()
    for group_name in REQUIRED_EVIDENCE_GROUPS:
        entries = groups.get(group_name)
        if not isinstance(entries, list) or not entries:
            failures.append(f"missing or empty evidence group: {group_name}")
            continue
        observed_paths = [
            entry.get("path")
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        ]
        if len(observed_paths) != len(entries):
            failures.append(f"invalid or missing evidence path in {group_name}")
        if observed_paths != sorted(observed_paths):
            failures.append(f"evidence group is not path-sorted: {group_name}")
        if observed_paths != expected_paths.get(group_name):
            failures.append(f"exact path inventory mismatch: {group_name}")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
                failures.append(f"invalid evidence entry schema in {group_name}")
                continue
            relative = entry["path"]
            if relative in seen_paths:
                failures.append(f"duplicate evidence path: {relative}")
            seen_paths.add(relative)
            logical_path = (project_root / relative).resolve(strict=False)
            try:
                logical_path.relative_to(project_root)
            except ValueError:
                failures.append(f"evidence path escapes project root: {relative}")
                continue
            path = (
                bundle / logical_path.name
                if logical_path.parent == logical_bundle
                and logical_path.name in BUNDLE_FILES
                else logical_path
            )
            try:
                physical_observed, fingerprint = _stable_entry(project_root, path)
                observed = {
                    "path": relative,
                    "size_bytes": physical_observed["size_bytes"],
                    "sha256": physical_observed["sha256"],
                }
            except Exception as exc:
                failures.append(
                    f"invalid {group_name} evidence {relative}: {type(exc).__name__}: {exc}"
                )
                continue
            verified_fingerprints[physical_observed["path"]] = fingerprint
            if observed != entry:
                failures.append(f"evidence size/hash mismatch: {relative}")

    try:
        methodology = verify_methodology_freeze(
            project_root / "outputs" / "methodology_v1_freeze.json"
        )
        if methodology.get("freeze_id") != identity.get("methodology_freeze_id"):
            failures.append("methodology freeze ID mismatch")
        source_selection = identity["source_selection"]
        expected_source_keys = {
            "latest_verified",
            "latest_verified_sha256",
            "latest_primary",
            "latest_primary_sha256",
            "primary_directory",
            "verification_directory",
            "primary_manifest_sha256",
            "primary_implementation_sha256",
            "verifier_implementation_sha256",
            "configuration_hash",
            "path_configuration_hash",
            "methodology_freeze",
        }
        if set(source_selection) != expected_source_keys:
            failures.append("bound source-selection key inventory mismatch")
        if methodology != source_selection.get("methodology_freeze"):
            failures.append("methodology authentication differs from bound source selection")

        if not identifiers_are_safe:
            raise ValueError("unsafe identifiers prohibit cross-file verification")
        build_id = identity["phase_01_build_id"]
        verification_id = identity["verification_id"]
        artifact_root = project_root / "artifacts" / "phase_01_raw_manifest"
        primary_directory = artifact_root / "builds" / build_id
        verification_directory = artifact_root / "verifications" / build_id / verification_id
        manifest = json.loads((primary_directory / "manifest.json").read_text(encoding="utf-8"))
        independent = json.loads(
            (verification_directory / "independent_check.json").read_text(encoding="utf-8")
        )
        primary_metadata = json.loads(
            (primary_directory / "run_metadata.json").read_text(encoding="utf-8")
        )
        latest_verified = source_selection.get("latest_verified", {})
        latest_primary = source_selection.get("latest_primary", {})
        if source_selection.get("latest_verified_sha256") != _sha256_bytes(
            _canonical_json_bytes(latest_verified)
        ):
            failures.append("bound LATEST_VERIFIED payload/hash mismatch")
        if source_selection.get("latest_primary_sha256") != _sha256_bytes(
            _canonical_json_bytes(latest_primary)
        ):
            failures.append("bound LATEST_PRIMARY payload/hash mismatch")
        expected_primary_relative = str(primary_directory.relative_to(project_root))
        expected_verification_relative = str(verification_directory.relative_to(project_root))
        if (
            source_selection.get("primary_directory") != expected_primary_relative
            or source_selection.get("verification_directory")
            != expected_verification_relative
            or latest_verified.get("phase_01_build_id") != build_id
            or latest_verified.get("raw_snapshot_id") != identity.get("raw_snapshot_id")
            or latest_verified.get("verification_id") != verification_id
            or latest_verified.get("verification_status") != "passed"
            or latest_verified.get("primary_path")
            != str(primary_directory.relative_to(artifact_root))
            or latest_verified.get("verification_path")
            != str(verification_directory.relative_to(artifact_root))
            or latest_primary.get("phase_01_build_id") != build_id
            or latest_primary.get("raw_snapshot_id") != identity.get("raw_snapshot_id")
            or latest_primary.get("path")
            != str(primary_directory.relative_to(artifact_root))
        ):
            failures.append("bound source-pointer selection does not resolve to the frozen build")
        current_primary_hash = _stable_hash(primary_directory / "manifest.json")[0]
        current_primary_code_hash = _stable_hash(
            project_root / "src" / "portfolio_bot" / "data" / "raw_manifest.py"
        )[0]
        current_verifier_code_hash = _stable_hash(
            project_root / "src" / "portfolio_bot" / "data" / "verify_raw_manifest.py"
        )[0]
        if (
            source_selection.get("primary_manifest_sha256") != current_primary_hash
            or source_selection.get("primary_implementation_sha256")
            != current_primary_code_hash
            or source_selection.get("verifier_implementation_sha256")
            != current_verifier_code_hash
        ):
            failures.append("bound source implementation or manifest hash mismatch")
        if (
            manifest.get("phase_01_build_id") != build_id
            or manifest.get("raw_snapshot_id") != identity.get("raw_snapshot_id")
            or manifest.get("primary_status")
            != "primary_passed_pending_independent_verification"
        ):
            failures.append("primary manifest IDs mismatch")
        if identity.get("scope_boundary") != manifest.get("scope_boundary"):
            failures.append("bound scope boundary differs from the primary manifest")
        if (
            independent.get("phase_01_build_id") != build_id
            or independent.get("verification_id") != verification_id
            or independent.get("verification_status") != "passed"
            or independent.get("difference_count") != 0
        ):
            failures.append("independent verification identity/status mismatch")
        if (
            independent.get("manifest_sha256") != current_primary_hash
            or independent.get("imported_primary_implementation_sha256")
            != current_primary_code_hash
            or independent.get("verifier_implementation_sha256")
            != current_verifier_code_hash
            or source_selection.get("configuration_hash")
            != manifest.get("configuration_hash")
            or source_selection.get("configuration_hash")
            != independent.get("configuration_hash")
            or source_selection.get("path_configuration_hash")
            != primary_metadata.get("path_configuration_sha256")
        ):
            failures.append("bound build/configuration hashes do not reconcile")
        primary_hashes = _read_hash_registry(primary_directory / "hashes.sha256")
        verified_hashes = _read_hash_registry(verification_directory / "shasum.sha256")
        final_hashes = _read_hash_registry(bundle / "raw_freeze_recheck.sha256")
        settings_for_contract = load_settings()
        if (
            settings_for_contract.configuration_hash
            != source_selection.get("configuration_hash")
            or settings_for_contract.path_configuration_hash
            != source_selection.get("path_configuration_hash")
            or settings_for_contract.contract_hashes != manifest.get("contract_hashes")
            or settings_for_contract.artifact_root.resolve() != artifact_root.resolve()
            or settings_for_contract.freeze_directory.resolve()
            != (artifact_root / "freezes").resolve()
        ):
            failures.append("current configuration or raw contracts differ from the frozen pin")

        current_verified_path = artifact_root / "LATEST_VERIFIED.json"
        current_primary_path = artifact_root / "LATEST_PRIMARY.json"
        (
            current_verified,
            current_verified_entry,
            current_verified_fingerprint,
        ) = _stable_json_entry(project_root, current_verified_path)
        (
            current_primary,
            current_primary_entry,
            current_primary_fingerprint,
        ) = _stable_json_entry(project_root, current_primary_path)
        verified_fingerprints[current_verified_entry["path"]] = (
            current_verified_fingerprint
        )
        verified_fingerprints[current_primary_entry["path"]] = (
            current_primary_fingerprint
        )
        expected_current_primary = {
            "path": str(primary_directory.relative_to(artifact_root)),
            "phase_01_build_id": build_id,
            "primary_status": "primary_passed_pending_independent_verification",
            "raw_snapshot_id": identity.get("raw_snapshot_id"),
            "verification_status": "not_run",
        }
        if (
            current_verified != latest_verified
            or current_verified_entry["sha256"]
            != source_selection.get("latest_verified_sha256")
            or current_primary != latest_primary
            or current_primary != expected_current_primary
            or current_primary_entry["sha256"]
            != source_selection.get("latest_primary_sha256")
        ):
            failures.append("current source-selection pointers differ from the frozen pin")

        expected_raw_files = sorted(contract.filename for contract in settings_for_contract.contracts)
        primary_manifest_files = sorted(
            item.get("filename") for item in manifest.get("files", [])
        )
        independent_files = sorted(
            item.get("filename") for item in independent.get("files", [])
        )
        digest_registry_valid = all(
            re.fullmatch(r"[0-9a-f]{64}", digest) is not None
            for registry in (primary_hashes, verified_hashes, final_hashes)
            for digest in registry.values()
        )
        if (
            len(expected_raw_files) != settings_for_contract.expected_file_count
            or settings_for_contract.expected_file_count != 25
            or sorted(primary_hashes) != expected_raw_files
            or primary_manifest_files != expected_raw_files
            or independent_files != expected_raw_files
            or not digest_registry_valid
        ):
            failures.append("raw hash registries do not match the exact 25-file contract")
        if not (primary_hashes == verified_hashes == final_hashes):
            failures.append("primary, independent, and final raw hash registries differ")

        registry_paths_and_values = (
            (primary_directory / "hashes.sha256", primary_hashes),
            (verification_directory / "shasum.sha256", verified_hashes),
            (bundle / "raw_freeze_recheck.sha256", final_hashes),
        )
        if any(
            path.read_bytes() != _canonical_hash_registry_bytes(values)
            for path, values in registry_paths_and_values
        ):
            failures.append("a raw hash registry is not in canonical sorted form")

        manifest_by_file = {
            item.get("filename"): item for item in manifest.get("files", [])
        }
        independent_by_file = {
            item.get("filename"): item for item in independent.get("files", [])
        }
        per_file_identity_matches = all(
            manifest_by_file[filename].get("byte_identity", {}).get("sha256")
            == primary_hashes.get(filename)
            == independent_by_file[filename].get("sha256")
            and manifest_by_file[filename].get("byte_identity", {}).get("size_bytes")
            == independent_by_file[filename].get("size_bytes")
            and manifest_by_file[filename].get("semantic_profile", {}).get(
                "parsed_data_rows"
            )
            == independent_by_file[filename].get("parsed_data_rows")
            for filename in expected_raw_files
            if filename in manifest_by_file and filename in independent_by_file
        )
        if (
            independent.get("schema_version") != 2
            or independent.get("raw_snapshot_id") != identity.get("raw_snapshot_id")
            or len(manifest_by_file) != len(expected_raw_files)
            or len(independent_by_file) != len(expected_raw_files)
            or not per_file_identity_matches
        ):
            failures.append("per-file primary and independent identities do not reconcile")

        metadata = json.loads(
            (bundle / "raw_freeze_recheck_metadata.json").read_text(encoding="utf-8")
        )
        per_file_metadata = metadata.get("per_file_operational_metadata", {})
        expected_metadata_keys = {
            "schema_version",
            "status",
            "hash_engine",
            "file_count",
            "run_started_utc",
            "run_completed_utc",
            "elapsed_seconds",
            "raw_root",
            "raw_registry_before",
            "raw_registry_after",
            "all_hashes_match_primary",
            "all_files_stable_during_own_hash",
            "all_files_stable_through_entire_recheck",
            "per_file_operational_metadata",
        }
        expected_per_file_keys = {
            "stat_before",
            "stat_after_hash",
            "end_of_complete_recheck_stat",
            "stable_during_own_hash",
            "stable_through_entire_recheck",
        }
        expected_stat_keys = {
            "device",
            "inode",
            "mode",
            "size_bytes",
            "mtime_ns",
            "ctime_ns",
        }
        per_file_stability = True
        for filename in expected_raw_files:
            item = per_file_metadata.get(filename)
            expected_size = manifest_by_file.get(filename, {}).get(
                "byte_identity", {}
            ).get("size_bytes")
            if not isinstance(item, dict) or set(item) != expected_per_file_keys:
                per_file_stability = False
                continue
            stat_payloads = [
                item.get("stat_before"),
                item.get("stat_after_hash"),
                item.get("end_of_complete_recheck_stat"),
            ]
            stats_valid = all(
                isinstance(stat_payload, dict)
                and set(stat_payload) == expected_stat_keys
                and all(
                    type(stat_payload[key]) is int for key in expected_stat_keys
                )
                and stat_payload["size_bytes"] == expected_size
                and stat.S_ISREG(stat_payload["mode"])
                and not stat.S_ISLNK(stat_payload["mode"])
                for stat_payload in stat_payloads
            )
            per_file_stability = per_file_stability and (
                stats_valid
                and item.get("stable_during_own_hash") is True
                and item.get("stable_through_entire_recheck") is True
                and stat_payloads[0] == stat_payloads[1] == stat_payloads[2]
            )
            raw_path = settings_for_contract.raw_root / filename
            try:
                current_raw_stat = raw_path.lstat()
                current_raw_matches = (
                    stat.S_ISREG(current_raw_stat.st_mode)
                    and not stat.S_ISLNK(current_raw_stat.st_mode)
                    and _stat_payload(_fingerprint(current_raw_stat))
                    == item.get("end_of_complete_recheck_stat")
                )
            except OSError:
                current_raw_matches = False
            per_file_stability = per_file_stability and current_raw_matches
        started_at = _parse_utc_timestamp(metadata.get("run_started_utc"))
        completed_at = _parse_utc_timestamp(metadata.get("run_completed_utc"))
        elapsed = metadata.get("elapsed_seconds")
        elapsed_matches_timestamps = (
            started_at is not None
            and completed_at is not None
            and type(elapsed) in {int, float}
            and abs((completed_at - started_at).total_seconds() - elapsed) <= 1.0
        )
        if (
            set(metadata) != expected_metadata_keys
            or metadata.get("schema_version") != 2
            or metadata.get("status") != "passed"
            or metadata.get("hash_engine") != "/usr/bin/shasum -a 256"
            or metadata.get("file_count") != len(primary_hashes)
            or metadata.get("raw_root") != str(settings_for_contract.raw_root)
            or started_at is None
            or completed_at is None
            or started_at > completed_at
            or type(elapsed) not in {int, float}
            or elapsed < 0
            or not elapsed_matches_timestamps
            or _raw_registry(settings_for_contract) != expected_raw_files
            or metadata.get("raw_registry_before") != sorted(primary_hashes)
            or metadata.get("raw_registry_after") != sorted(primary_hashes)
            or metadata.get("all_hashes_match_primary") is not True
            or metadata.get("all_files_stable_during_own_hash") is not True
            or metadata.get("all_files_stable_through_entire_recheck") is not True
            or sorted(metadata.get("per_file_operational_metadata", {})) != sorted(primary_hashes)
            or not per_file_stability
        ):
            failures.append("final raw recheck metadata does not prove the complete invariant")

        run_record = json.loads((bundle / "run_record.json").read_text(encoding="utf-8"))
        expected_run_record_keys = {
            "schema_version",
            "phase",
            "run_id",
            "status",
            "created_utc",
            "methodology_freeze_id",
            "raw_snapshot_id",
            "phase_01_build_id",
            "verification_id",
            "primary_build",
            "independent_verification",
            "report",
            "notebook",
            "validation",
            "freeze_manifest",
            "sealed_periods",
            "next_permitted_action",
            "freeze_id_location",
        }
        expected_validation_relative = (
            Path("runs")
            / "phase_01_raw_snapshot"
            / build_id
            / "validation_report.json"
        ).as_posix()
        if (
            set(run_record) != expected_run_record_keys
            or run_record.get("schema_version") != 2
            or run_record.get("phase") != "01_raw_snapshot"
            or run_record.get("run_id") != build_id
            or run_record.get("status") != "phase_01_frozen"
            or run_record.get("raw_snapshot_id") != identity.get("raw_snapshot_id")
            or run_record.get("phase_01_build_id") != build_id
            or run_record.get("verification_id") != verification_id
            or run_record.get("methodology_freeze_id") != identity.get("methodology_freeze_id")
            or run_record.get("freeze_manifest")
            != _bound_relative_path(project_root, logical_bundle / "manifest.json")
            or run_record.get("sealed_periods") != SEALED_PERIODS
            or run_record.get("created_utc") != identity.get("frozen_at_utc")
            or run_record.get("freeze_id_location") != "manifest.json"
            or run_record.get("next_permitted_action")
            != "Begin Phase 2 canonical point-in-time panel contracts and implementation."
            or run_record.get("report")
            != "reports/phase_01_raw_snapshot/phase_01_verification_report.html"
            or run_record.get("notebook")
            != "notebooks/01_phase_01_raw_snapshot_review.ipynb"
            or run_record.get("primary_build") != expected_primary_relative
            or run_record.get("independent_verification")
            != expected_verification_relative
            or run_record.get("validation") != expected_validation_relative
        ):
            failures.append("immutable run record mismatch")
        with (bundle / "registry_entry.csv").open("r", encoding="utf-8", newline="") as handle:
            immutable_reader = csv.DictReader(handle)
            immutable_rows = list(immutable_reader)
        expected_immutable_row = _immutable_registry_row(run_record)
        if (
            immutable_rows != [expected_immutable_row]
            or immutable_reader.fieldnames != list(expected_immutable_row)
            or (bundle / "registry_entry.csv").read_bytes()
            != _registry_csv_bytes(expected_immutable_row)
        ):
            failures.append("immutable registry entry mismatch")

        review_entries = {
            item["path"]: item for item in groups.get("review_and_validation", [])
        }
        validation_path = project_root / expected_validation_relative
        if validation_path.resolve() != (
            project_root / expected_validation_relative
        ).resolve() or expected_validation_relative not in review_entries:
            failures.append("canonical validation path is not bound in review evidence")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        current_validation = validate_handoff(project_root)
        if _without_validation_time(current_validation) != _without_validation_time(
            validation
        ):
            failures.append("saved handoff validation is stale relative to current evidence")
        validation_check_ids = [
            item.get("check_id") for item in validation.get("checks", [])
        ]
        if (
            validation.get("schema_version") != 1
            or validation.get("phase") != "01_raw_snapshot"
            or validation.get("assessment") != "Ready to freeze"
            or validation.get("failed_check_count") != 0
            or validation.get("raw_snapshot_id") != identity.get("raw_snapshot_id")
            or validation.get("phase_01_build_id") != build_id
            or validation.get("verification_id") != verification_id
            or validation_check_ids != list(PHASE_01_HANDOFF_CHECK_IDS)
            or any(item.get("status") != "PASS" for item in validation.get("checks", []))
        ):
            failures.append("saved handoff validation mismatch")
        expected_validation_assessment = {
            "assessment": validation.get("assessment"),
            "failed_check_count": validation.get("failed_check_count"),
            "checks": [
                {"check_id": item.get("check_id"), "status": item.get("status")}
                for item in validation.get("checks", [])
            ],
        }
        if identity.get("validation_assessment") != expected_validation_assessment:
            failures.append("bound validation assessment differs from saved validation")
        implementation_entries = groups.get("implementation_and_configuration", [])
        if validation.get("tested_implementation") != implementation_entries:
            failures.append("test receipt does not bind the frozen implementation inventory")
        pytest_receipt = validation.get("pytest_receipt", {})
        pytest_entry = review_entries.get(pytest_receipt.get("path"))
        if not pytest_entry or any(
            pytest_receipt.get(key) != pytest_entry.get(key)
            for key in ("path", "size_bytes", "sha256")
        ):
            failures.append("saved validation does not bind the frozen pytest receipt")
        junit_path = (
            project_root
            / "runs"
            / "phase_01_raw_snapshot"
            / build_id
            / "pytest.xml"
        )
        junit_root = et.parse(junit_path).getroot()
        suites = list(junit_root.findall("testsuite"))
        testcases = [
            case for suite in suites for case in suite.findall("testcase")
        ]
        observed_test_ids = sorted(
            f"{case.attrib.get('classname')}::{case.attrib.get('name')}"
            for case in testcases
        )
        junit_summary = {
            "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
            "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
            "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
            "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        }
        junit_child_results = {
            "failures": sum(len(case.findall("failure")) for case in testcases),
            "errors": sum(len(case.findall("error")) for case in testcases),
            "skipped": sum(len(case.findall("skipped")) for case in testcases),
        }
        if (
            observed_test_ids != sorted(PHASE_01_TEST_IDS)
            or len(testcases) != len(PHASE_01_TEST_IDS)
            or junit_summary
            != {
                "tests": len(PHASE_01_TEST_IDS),
                "failures": 0,
                "errors": 0,
                "skipped": 0,
            }
            or junit_child_results
            != {"failures": 0, "errors": 0, "skipped": 0}
            or pytest_receipt.get("test_ids") != observed_test_ids
        ):
            failures.append("frozen JUnit does not match the exact passing Phase 1 suite")
        test_receipt_relative = (
            Path("runs")
            / "phase_01_raw_snapshot"
            / build_id
            / "test_execution_receipt.json"
        ).as_posix()
        test_receipt_path = project_root / test_receipt_relative
        test_receipt = json.loads(test_receipt_path.read_text(encoding="utf-8"))
        test_receipt_entry = review_entries.get(test_receipt_relative)
        observed_test_receipt_entry, _ = _stable_entry(project_root, test_receipt_path)
        junit_relative = junit_path.relative_to(project_root).as_posix()
        junit_entry = review_entries.get(junit_relative)
        expected_junit_structure = {
            "root_element": junit_root.tag,
            "suite_count": len(suites),
            "testcase_count": len(testcases),
            "child_result_counts": {
                "failure": junit_child_results["failures"],
                "error": junit_child_results["errors"],
                "skipped": junit_child_results["skipped"],
            },
        }
        recorded_junit = test_receipt.get("junit", {})
        recorded_junit_identity = (
            {
                key: recorded_junit.get(key)
                for key in ("path", "size_bytes", "sha256", "identity")
            }
            if isinstance(recorded_junit, dict)
            else {}
        )
        junit_receipt_valid = (
            isinstance(recorded_junit, dict)
            and set(recorded_junit)
            == {
                "path",
                "size_bytes",
                "sha256",
                "identity",
                *expected_junit_structure.keys(),
            }
            and _identity_receipt_valid(recorded_junit_identity)
            and {
                key: recorded_junit.get(key)
                for key in ("path", "size_bytes", "sha256")
            }
            == junit_entry
            and all(
                recorded_junit.get(key) == value
                for key, value in expected_junit_structure.items()
            )
        )
        expected_implementation_integrity = [
            {
                **entry,
                "identity": _stat_payload(verified_fingerprints[entry["path"]]),
            }
            for entry in implementation_entries
        ]
        expected_test_source_selection = {
            "artifact_root": artifact_root.relative_to(project_root).as_posix(),
            "raw_snapshot_id": identity.get("raw_snapshot_id"),
            "phase_01_build_id": build_id,
            "verification_id": verification_id,
            "selected_primary_directory": expected_primary_relative,
            "selected_verification_directory": expected_verification_relative,
            "latest_verified": {
                "receipt": {
                    **current_verified_entry,
                    "identity": _stat_payload(current_verified_fingerprint),
                },
                "payload": current_verified,
            },
            "latest_primary": {
                "receipt": {
                    **current_primary_entry,
                    "identity": _stat_payload(current_primary_fingerprint),
                },
                "payload": current_primary,
            },
        }
        python_executable = Path(sys.executable).resolve(strict=True)
        executable_digest, executable_size, executable_fingerprint, _ = _stable_hash(
            python_executable
        )
        executable_receipt = {
            "path": str(python_executable),
            "size_bytes": executable_size,
            "sha256": executable_digest,
            "identity": _stat_payload(executable_fingerprint),
        }
        expected_test_runtime = {
            "python_invocation": sys.executable,
            "python_executable": str(python_executable),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "pytest_version": importlib.metadata.version("pytest"),
            "executable_before": executable_receipt,
            "executable_after": executable_receipt,
        }
        expected_test_command = [
            str(python_executable),
            "-m",
            "pytest",
            "-q",
            "tests",
            f"--junitxml={junit_path}",
        ]
        expected_test_receipt_keys = {
            "schema_version",
            "receipt_type",
            "status",
            "run_started_utc",
            "run_completed_utc",
            "raw_snapshot_id",
            "phase_01_build_id",
            "verification_id",
            "cwd",
            "runtime",
            "command",
            "return_code",
            "source_selection_before",
            "source_selection_after",
            "test_summary",
            "test_ids",
            "junit",
            "implementation_before",
            "implementation_after",
            "implementation_integrity_before",
            "implementation_integrity_after",
        }
        test_started = _parse_utc_timestamp(test_receipt.get("run_started_utc"))
        test_completed = _parse_utc_timestamp(test_receipt.get("run_completed_utc"))
        if (
            test_receipt_entry != observed_test_receipt_entry
            or validation.get("test_execution_receipt") != test_receipt
            or set(test_receipt) != expected_test_receipt_keys
            or type(test_receipt.get("schema_version")) is not int
            or test_receipt.get("schema_version") != 2
            or test_receipt.get("receipt_type")
            != "phase_01_exact_test_execution"
            or test_receipt.get("status") != "passed"
            or test_started is None
            or test_completed is None
            or test_started > test_completed
            or type(test_receipt.get("return_code")) is not int
            or test_receipt.get("return_code") != 0
            or test_receipt.get("raw_snapshot_id") != identity.get("raw_snapshot_id")
            or test_receipt.get("phase_01_build_id") != build_id
            or test_receipt.get("verification_id") != verification_id
            or test_receipt.get("cwd") != str(project_root)
            or test_receipt.get("runtime") != expected_test_runtime
            or test_receipt.get("command") != expected_test_command
            or test_receipt.get("source_selection_before")
            != expected_test_source_selection
            or test_receipt.get("source_selection_after")
            != expected_test_source_selection
            or test_receipt.get("test_summary") != junit_summary
            or test_receipt.get("test_ids") != observed_test_ids
            or not junit_receipt_valid
            or test_receipt.get("implementation_before") != implementation_entries
            or test_receipt.get("implementation_after") != implementation_entries
            or test_receipt.get("implementation_integrity_before")
            != expected_implementation_integrity
            or test_receipt.get("implementation_integrity_after")
            != expected_implementation_integrity
        ):
            failures.append("causal test-execution receipt mismatch")
    except Exception as exc:
        failures.append(f"cross-file verification failed: {type(exc).__name__}: {exc}")

    expected_bundle = (
        project_root
        / "artifacts"
        / "phase_01_raw_manifest"
        / "freezes"
        / str(identity.get("raw_snapshot_id"))
    )
    if logical_bundle != expected_bundle:
        failures.append("freeze bundle path does not match its raw snapshot ID")

    if require_navigation:
        try:
            settings = load_settings()
            latest_path = settings.artifact_root / "LATEST_FROZEN.json"
            latest, latest_entry, latest_fingerprint = _stable_json_entry(
                project_root, latest_path
            )
            verified_fingerprints[latest_entry["path"]] = latest_fingerprint
            current_verified_path = settings.artifact_root / "LATEST_VERIFIED.json"
            current_primary_path = settings.artifact_root / "LATEST_PRIMARY.json"
            (
                current_verified,
                current_verified_entry,
                current_verified_fingerprint,
            ) = _stable_json_entry(project_root, current_verified_path)
            (
                current_primary,
                current_primary_entry,
                current_primary_fingerprint,
            ) = _stable_json_entry(project_root, current_primary_path)
            current_verified_sha = current_verified_entry["sha256"]
            current_primary_sha = current_primary_entry["sha256"]
            verified_fingerprints[current_verified_entry["path"]] = (
                current_verified_fingerprint
            )
            verified_fingerprints[current_primary_entry["path"]] = (
                current_primary_fingerprint
            )
            if (
                current_verified != identity["source_selection"]["latest_verified"]
                or current_verified_sha
                != identity["source_selection"]["latest_verified_sha256"]
                or current_primary != identity["source_selection"]["latest_primary"]
                or current_primary_sha
                != identity["source_selection"]["latest_primary_sha256"]
            ):
                failures.append("current source-selection pointers differ from the frozen pin")
            expected_latest = {
                "raw_snapshot_id": identity["raw_snapshot_id"],
                "phase_01_build_id": identity["phase_01_build_id"],
                "verification_id": identity["verification_id"],
                "freeze_id": payload["freeze_id"],
                "path": str(freeze_path.relative_to(settings.artifact_root)),
                "status": "phase_01_frozen",
                "sha256": manifest_sha,
            }
            if latest != expected_latest:
                failures.append("LATEST_FROZEN pointer or full-manifest hash mismatch")
            raw_pointer_path = project_root / "data" / "raw_snapshot.pointer.json"
            (
                raw_pointer,
                raw_pointer_entry,
                raw_pointer_fingerprint,
            ) = _stable_json_entry(
                project_root, raw_pointer_path
            )
            verified_fingerprints[raw_pointer_entry["path"]] = raw_pointer_fingerprint
            expected_raw_pointer = {
                "database_root": str(settings.database_root),
                "raw_subdirectory": str(
                    settings.raw_root.relative_to(settings.database_root)
                ),
                "path_override_environment_variable": "PORTFOLIO_DATABASE_ROOT",
                "expected_csv_files": settings.expected_file_count,
                "snapshot_status": "phase_01_frozen",
                "methodology_freeze_id": identity["methodology_freeze_id"],
                "raw_snapshot_id": identity["raw_snapshot_id"],
                "phase_01_build_id": identity["phase_01_build_id"],
                "verification_id": identity["verification_id"],
                "freeze_id": payload["freeze_id"],
                "freeze_manifest": str(freeze_path.relative_to(project_root)),
                "freeze_manifest_sha256": manifest_sha,
            }
            if raw_pointer != expected_raw_pointer:
                failures.append("raw snapshot navigation pointer mismatch")
            run_record = json.loads((bundle / "run_record.json").read_text(encoding="utf-8"))
            expected_row = _navigation_row(identity, payload["freeze_id"], run_record)
            global_registry_path = project_root / "runs" / "registry.csv"
            global_registry_entry, global_registry_fingerprint = _stable_entry(
                project_root, global_registry_path
            )
            verified_fingerprints[
                global_registry_entry["path"]
            ] = global_registry_fingerprint
            with global_registry_path.open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                global_reader = csv.DictReader(handle)
                rows = list(global_reader)
            matching = [row for row in rows if row.get("phase") == "01_raw_snapshot"]
            if (
                matching != [expected_row]
                or global_reader.fieldnames != list(expected_row)
                or len(global_reader.fieldnames or [])
                != len(set(global_reader.fieldnames or []))
            ):
                failures.append("global run registry does not contain one exact Phase 1 row")
        except Exception as exc:
            failures.append(f"navigation verification failed: {type(exc).__name__}: {exc}")
    try:
        if sorted(path.name for path in bundle.iterdir()) != sorted(BUNDLE_FILES):
            failures.append("bundle inventory changed during published verification")
        _exact_directory_paths(
            project_root
            / "artifacts"
            / "phase_01_raw_manifest"
            / "builds"
            / str(identity.get("phase_01_build_id")),
            PRIMARY_FILES,
        )
        _exact_directory_paths(
            project_root
            / "artifacts"
            / "phase_01_raw_manifest"
            / "verifications"
            / str(identity.get("phase_01_build_id"))
            / str(identity.get("verification_id")),
            VERIFICATION_FILES,
        )
        _assert_fingerprints(project_root, verified_fingerprints)
    except Exception as exc:
        failures.append(
            f"evidence changed during published verification: {type(exc).__name__}: {exc}"
        )
    return failures
