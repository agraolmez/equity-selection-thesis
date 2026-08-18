"""Operational recovery for interrupted Stage 6 family-cache resumes.

This module is intentionally not part of the Stage 6 scientific build hash.
It changes no feature values.  It only quarantines byte-identical forensic
staging candidates left by the secure publisher and reuses authenticated
batch bundles after comparing them with the freshly generated batch tables.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import pyarrow.parquet as pq

from portfolio_bot.features import feature_store_family_orchestration as orchestration
from portfolio_bot.features import feature_store_materialization as materialization
from portfolio_bot.features.feature_store_contract import DEFAULT_POLICY
from portfolio_bot.features.feature_store_inputs import FeatureStoreInputs
from portfolio_bot.features.phase_03_contract import canonical_json_bytes


_RETIRED_BATCH = re.compile(
    r"^\.\.(batch-[0-9]{6})\.staging-[0-9a-f]+\.[0-9a-f]+\.retired$"
)
_RECOVERY_NAMESPACE = "stage06_resume_recovery_quarantine"


class Stage6ResumeRecoveryError(RuntimeError):
    """Raised when an interrupted cache cannot be recovered without ambiguity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage6ResumeRecoveryError(message)


def _same_file(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _hash_regular_file(path: Path) -> tuple[str, int]:
    try:
        before = os.lstat(path)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage6ResumeRecoveryError(f"Cannot open recovery member: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and _same_file(before, opened),
            f"Recovery member is not a stable singly linked regular file: {path}",
        )
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(path)
    except OSError as exc:
        raise Stage6ResumeRecoveryError(f"Recovery member disappeared: {path}") from exc
    _require(
        _same_file(before, after_fd) and _same_file(before, after_path),
        f"Recovery member changed while hashing: {path}",
    )
    return digest.hexdigest(), int(before.st_size)


def _real_directory(path: Path, label: str) -> None:
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise Stage6ResumeRecoveryError(f"Cannot inspect {label}: {path}") from exc
    _require(
        stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        f"{label} is not a real directory: {path}",
    )


def _mkdir_real_descendant(root: Path, relative: Path) -> Path:
    _require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe recovery path")
    _real_directory(root, "trusted recovery root")
    current = root
    for component in relative.parts:
        _require(component not in ("", "."), "Unsafe empty recovery component")
        current = current / component
        try:
            os.mkdir(current, 0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise Stage6ResumeRecoveryError(
                f"Cannot create recovery directory: {current}"
            ) from exc
        _real_directory(current, "recovery directory")
    return current


def _closed_hashes(path: Path, expected: set[str]) -> dict[str, dict[str, Any]]:
    _real_directory(path, "recovery bundle")
    try:
        names = set(os.listdir(path))
    except OSError as exc:
        raise Stage6ResumeRecoveryError(f"Cannot list recovery bundle: {path}") from exc
    _require(names == expected, f"Recovery bundle has an open file set: {path}")
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        digest, size = _hash_regular_file(path / name)
        result[name] = {"sha256": digest, "size_bytes": size}
    return result


@dataclass(frozen=True)
class RecoveryReceipt:
    recovery_id: str
    quarantine_path: Path | None
    receipt_path: Path | None
    moved_candidates: int


def quarantine_identical_retired_candidates(
    *,
    inputs: FeatureStoreInputs,
    identity: materialization.FeatureStoreBuildIdentity,
    artifact_root: Path,
    policy_path: Path = DEFAULT_POLICY,
) -> RecoveryReceipt:
    """Move only byte-identical retired batch candidates out of family roots."""

    cache_root = orchestration._cache_root(inputs, policy_path)
    _real_directory(cache_root, "Stage 6 cache root")
    candidates: list[dict[str, Any]] = []
    for family in orchestration._ALL_FAMILIES:
        family_root = orchestration._family_root(inputs, identity, family, policy_path)
        if not os.path.lexists(family_root):
            continue
        _real_directory(family_root, "Stage 6 family root")
        for candidate in sorted(family_root.iterdir(), key=lambda value: value.name):
            match = _RETIRED_BATCH.fullmatch(candidate.name)
            if match is None:
                continue
            batch_id = match.group(1)
            destination = family_root / batch_id
            _require(os.path.lexists(destination), f"Retired candidate has no batch: {candidate}")
            receipt = materialization.authenticate_materialized_family_batch(
                inputs=inputs,
                family=family,
                batch_id=batch_id,
                policy_path=policy_path,
                build_identity=identity,
            )
            expected_names = {
                "_SUCCESS.json",
                f"feature_values_{family}.parquet",
                f"feature_lineage_{family}.parquet",
            }
            final_hashes = _closed_hashes(receipt.path, expected_names)
            retired_hashes = _closed_hashes(candidate, expected_names)
            _require(
                retired_hashes == final_hashes,
                f"Retired candidate is not byte-identical to its batch: {candidate}",
            )
            candidates.append(
                {
                    "family": family,
                    "batch_id": batch_id,
                    "source_name": candidate.name,
                    "files": retired_hashes,
                }
            )

    if not candidates:
        return RecoveryReceipt("none", None, None, 0)

    intent = {
        "schema_version": 1,
        "operation": "quarantine_byte_identical_stage6_retired_reuse_candidates",
        "build_id": identity.build_id,
        "content_fingerprint": identity.content_fingerprint,
        "candidates": candidates,
    }
    recovery_id = "recovery-" + hashlib.sha256(canonical_json_bytes(intent)).hexdigest()[:16]
    quarantine = _mkdir_real_descendant(
        cache_root,
        Path(_RECOVERY_NAMESPACE) / identity.build_id / recovery_id,
    )
    for item in candidates:
        family = str(item["family"])
        family_destination = _mkdir_real_descendant(
            quarantine,
            Path(family),
        )
        source = orchestration._family_root(inputs, identity, family, policy_path) / str(
            item["source_name"]
        )
        destination = family_destination / source.name
        _require(not os.path.lexists(destination), f"Recovery destination already exists: {destination}")
        try:
            os.rename(source, destination)
        except OSError as exc:
            raise Stage6ResumeRecoveryError(f"Cannot quarantine recovery candidate: {source}") from exc
        expected_names = set(str(name) for name in item["files"])
        _require(
            _closed_hashes(destination, expected_names) == item["files"],
            f"Quarantined recovery candidate changed: {destination}",
        )

    receipt_payload = {
        **intent,
        "recovery_id": recovery_id,
        "status": "QUARANTINED_IDENTICAL_REUSE_CANDIDATES",
        "quarantine_path": quarantine.as_posix(),
        "moved_candidates": len(candidates),
    }
    receipt_directory = artifact_root / "resume_recoveries" / identity.build_id
    receipt_directory.mkdir(parents=True, exist_ok=True)
    _real_directory(receipt_directory, "recovery receipt directory")
    receipt_path = receipt_directory / f"{recovery_id}.json"
    payload = canonical_json_bytes(receipt_payload)
    try:
        with receipt_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        _require(receipt_path.read_bytes() == payload, "Existing recovery receipt differs")
    return RecoveryReceipt(recovery_id, quarantine, receipt_path, len(candidates))


@contextmanager
def reuse_authenticated_existing_batches() -> Iterator[None]:
    """Temporarily make the orchestrator resume existing batches without restaging."""

    original = orchestration.materialize_feature_family_batch

    def resume_or_materialize(
        *,
        inputs: FeatureStoreInputs,
        family: str,
        batch_id: str,
        rows: Iterable[materialization.FeatureMaterializationRow],
        policy_path: Path = DEFAULT_POLICY,
        build_identity: materialization.FeatureStoreBuildIdentity | None = None,
    ) -> materialization.FamilyBatchReceipt:
        materialized_rows = tuple(rows)
        if build_identity is None:
            return original(
                inputs=inputs,
                family=family,
                batch_id=batch_id,
                rows=materialized_rows,
                policy_path=policy_path,
                build_identity=build_identity,
            )
        destination = (
            orchestration._family_root(inputs, build_identity, family, policy_path)
            / batch_id
        )
        if not os.path.lexists(destination):
            return original(
                inputs=inputs,
                family=family,
                batch_id=batch_id,
                rows=materialized_rows,
                policy_path=policy_path,
                build_identity=build_identity,
            )

        receipt = materialization.authenticate_materialized_family_batch(
            inputs=inputs,
            family=family,
            batch_id=batch_id,
            policy_path=policy_path,
            build_identity=build_identity,
        )
        expected_values, expected_lineage = materialization._materialize_tables(
            materialized_rows,
            family,
            policy_path,
        )
        value_name = f"feature_values_{family}.parquet"
        lineage_name = f"feature_lineage_{family}.parquet"
        try:
            observed_values = pq.read_table(receipt.path / value_name, use_threads=False)
            observed_lineage = pq.read_table(receipt.path / lineage_name, use_threads=False)
        except Exception as exc:
            raise Stage6ResumeRecoveryError(
                f"Cannot compare existing resumed batch: {family}/{batch_id}"
            ) from exc
        _require(
            observed_values.schema.equals(expected_values.schema, check_metadata=True)
            and observed_lineage.schema.equals(expected_lineage.schema, check_metadata=True)
            and observed_values.equals(expected_values)
            and observed_lineage.equals(expected_lineage),
            f"Existing batch differs from freshly generated rows: {family}/{batch_id}",
        )
        reauthenticated = materialization.authenticate_materialized_family_batch(
            inputs=inputs,
            family=family,
            batch_id=batch_id,
            policy_path=policy_path,
            build_identity=build_identity,
        )
        _require(
            reauthenticated.success_sha256 == receipt.success_sha256,
            f"Existing batch changed during resume: {family}/{batch_id}",
        )
        return reauthenticated

    orchestration.materialize_feature_family_batch = resume_or_materialize
    try:
        yield
    finally:
        orchestration.materialize_feature_family_batch = original


__all__ = [
    "RecoveryReceipt",
    "Stage6ResumeRecoveryError",
    "quarantine_identical_retired_candidates",
    "reuse_authenticated_existing_batches",
]
