"""Authenticated, atomic publication of the 12 final normalized sources.

This is the only bridge from the repository-native final-period formula
builder to the fixed Stage 13 source namespace.  The production entry point is
sealed: callers cannot supply paths, dates, schemas, tables, or readers.  It
authenticates the verified Stage 12 binding and the separate fixed Stage 13
approval before either memoized final payloads or raw-backed builder code can
be reached.

Publication is all-or-nothing.  The twelve Parquet members are first written
to and independently inspected in a sibling staging directory, the directory
is renamed into the fixed namespace, an immutable content-addressed manifest
is written, and the single verified pointer is replaced last.  An interrupted
run can therefore resume from the last verified boundary without rebuilding
already-published content.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pyarrow as pa

from portfolio_bot.final_test.authorization import (
    Stage12VerifiedBinding,
    Stage13ApprovalReceipt,
    load_authenticated_stage12_binding,
    require_stage13_unblind_authorization,
)
from portfolio_bot.final_test.final_input_construction import (
    NORMALIZED_SOURCE_MANIFEST_NAMESPACE,
    NORMALIZED_SOURCE_NAMESPACE,
    NORMALIZED_SOURCE_POINTER,
    PROJECT_ROOT,
    _SOURCE_MONTHS,
    _SOURCE_NAMES,
    _atomic_json,
    _canonical_json_bytes,
    _load_authenticated_normalized_sources,
    _load_fixed_approval,
    _physical_schema,
    _source_schemas,
    _stable_file_bytes,
    _validate_normalized_source_tables,
    _write_tables_to_directory,
)
from portfolio_bot.final_test.input_materialization import (
    _configured_database_root,
    _inspect_parquet,
)


PUBLISHER_ID = "stage13-final-normalized-source-publisher-v1"
PUBLISHED_STATUS = "FINAL_NORMALIZED_SOURCE_PUBLISHED"
VERIFIED_STATUS = "FINAL_NORMALIZED_SOURCE_VERIFIED"
PREDECESSOR_SOURCE_POINTER_SHA256 = (
    "36200c2528529ab139f8d3f858a1f860b34dc85b32776d048926b0cf76f11386"
)
PREDECESSOR_SOURCE_MANIFEST_SHA256 = (
    "1c93a0efd96f9c6eaeba7481504bd964ba8ec8b73536e15ec63b19ef91ee46ae"
)
PREDECESSOR_SOURCE_MANIFEST_PATH = (
    "artifacts/phase_06_final_test/stage_13_normalized_source_manifests/"
    "stage13-normalized-sources-e1be73b7b2f65335306fc684.json"
)
NORMALIZED_SOURCE_POINTER_HISTORY_NAMESPACE = Path(
    "artifacts/phase_06_final_test/stage_13_normalized_source_pointer_history"
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CODE_MEMBERS = (
    Path("src/portfolio_bot/final_test/final_raw_source_publication.py"),
    Path("src/portfolio_bot/final_test/final_source_builder.py"),
    Path("src/portfolio_bot/final_test/final_input_construction.py"),
)
_CONFIG_MEMBERS = (
    Path("configs/paths.toml"),
    Path("configs/phase_06_final_test/stage_12_final_system_v1.json"),
    Path("configs/phase_06_final_test/stage_12_freeze_policy_v1.json"),
    Path("configs/phase_06_final_test/stage_13_input_registry_v1.json"),
)


class FinalNormalizedSourcePublicationError(RuntimeError):
    """Raised when normalized final sources are unauthenticated or incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalNormalizedSourcePublicationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_safe_mapping(value: object, *, label: str) -> dict[str, object]:
    _require(isinstance(value, Mapping) and bool(value), f"{label} is absent")
    result = dict(value)
    # Canonical serialization rejects NaN/infinity and non-JSON provenance.
    try:
        round_trip = json.loads(_canonical_json_bytes(result))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FinalNormalizedSourcePublicationError(
            f"{label} is not finite canonical JSON"
        ) from exc
    _require(isinstance(round_trip, dict), f"{label} must be an object")
    return round_trip


def _member_hashes(project_root: Path, members: tuple[Path, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in members:
        payload = _stable_file_bytes(
            project_root / relative,
            trusted_root=project_root,
            label=f"normalized-source publisher member {relative.as_posix()}",
            maximum_bytes=32 * 1024 * 1024,
        )
        hashes[relative.as_posix()] = _sha256_bytes(payload)
    return hashes


def _implementation_identity(project_root: Path) -> dict[str, object]:
    code = _member_hashes(project_root, _CODE_MEMBERS)
    config = _member_hashes(project_root, _CONFIG_MEMBERS)
    return {
        "publisher_id": PUBLISHER_ID,
        "publisher_code_hashes": code,
        "publisher_code_hashes_sha256": _sha256_bytes(
            _canonical_json_bytes(code)
        ),
        "publisher_config_hashes": config,
        "publisher_config_hashes_sha256": _sha256_bytes(
            _canonical_json_bytes(config)
        ),
    }


def _safe_directory(path: Path, *, trusted_root: Path, label: str) -> None:
    root = Path(os.path.abspath(trusted_root.expanduser()))
    selected = Path(os.path.abspath(path.expanduser()))
    try:
        relative = selected.relative_to(root)
    except ValueError as exc:
        raise FinalNormalizedSourcePublicationError(
            f"{label} escaped its root"
        ) from exc
    _require(bool(relative.parts), f"{label} cannot equal its root")
    current = root
    for component in relative.parts:
        current /= component
        if not current.exists():
            break
        metadata = current.stat(follow_symlinks=False)
        _require(
            stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
            f"{label} path is unsafe",
        )


def _source_receipts(
    destination: Path,
    *,
    project_root: Path,
    database_root: Path,
) -> tuple[dict[str, object], ...]:
    schemas = _source_schemas(project_root)
    _require(
        destination.is_dir()
        and not destination.is_symlink()
        and {member.name for member in destination.iterdir()}
        == {f"{name}.parquet" for name in _SOURCE_NAMES},
        "normalized source member set differs",
    )
    records: list[dict[str, object]] = []
    for name in _SOURCE_NAMES:
        relative = NORMALIZED_SOURCE_NAMESPACE / f"{name}.parquet"
        absolute = database_root / relative
        selected = destination / f"{name}.parquet"
        observed = _inspect_parquet(
            selected,
            database_root=database_root,
            expected_schema=_physical_schema(schemas[name]),
            expected_months=_SOURCE_MONTHS[name],
            label=f"normalized source {name}",
        )
        records.append(
            {
                "source_name": name,
                "relative_path": relative.as_posix(),
                "absolute_path": str(absolute),
                **observed,
                "holding_months": list(_SOURCE_MONTHS[name]),
            }
        )
    return tuple(records)


def _same_published_directory(
    destination: Path,
    stage: Path,
    *,
    project_root: Path,
    database_root: Path,
) -> bool:
    actual = _source_receipts(
        destination,
        project_root=project_root,
        database_root=database_root,
    )
    staged = _source_receipts(
        stage,
        project_root=project_root,
        database_root=database_root,
    )
    return all(
        left["sha256"] == right["sha256"]
        and left["size_bytes"] == right["size_bytes"]
        and left["rows"] == right["rows"]
        and left["row_groups"] == right["row_groups"]
        and left["arrow_schema_sha256"] == right["arrow_schema_sha256"]
        for left, right in zip(actual, staged, strict=True)
    )


@dataclass(frozen=True, slots=True)
class FinalNormalizedSourcePublication:
    source_bundle_id: str
    source_manifest_path: str
    source_manifest_sha256: str
    source_pointer_sha256: str
    output_namespace: str
    reused: bool


def _result_from_authenticated_identity(
    identity: Mapping[str, object],
    *,
    reused: bool,
) -> FinalNormalizedSourcePublication:
    bundle_id = identity.get("source_bundle_id")
    manifest_path = identity.get("source_manifest_path")
    manifest_sha = identity.get("source_manifest_sha256")
    pointer_sha = identity.get("source_pointer_sha256")
    _require(
        all(isinstance(value, str) and value for value in (
            bundle_id,
            manifest_path,
            manifest_sha,
            pointer_sha,
        ))
        and _HEX64.fullmatch(str(manifest_sha)) is not None
        and _HEX64.fullmatch(str(pointer_sha)) is not None,
        "authenticated normalized-source identity differs",
    )
    return FinalNormalizedSourcePublication(
        source_bundle_id=str(bundle_id),
        source_manifest_path=str(manifest_path),
        source_manifest_sha256=str(manifest_sha),
        source_pointer_sha256=str(pointer_sha),
        output_namespace=NORMALIZED_SOURCE_NAMESPACE.as_posix(),
        reused=reused,
    )


def _publish_authenticated_tables(
    tables: Mapping[str, pa.Table],
    *,
    raw_source_identity: Mapping[str, object],
    builder_identity: Mapping[str, object],
    project_root: Path,
    database_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
    publication_root: Path | None = None,
) -> FinalNormalizedSourcePublication:
    """Private synthetic-safe publication seam after explicit authorization."""

    root = project_root.absolute()
    database = database_root.absolute()
    metadata_root = (publication_root or root).absolute()
    authorized = require_stage13_unblind_authorization(
        approval,
        expected_stage12_binding=binding,
    )
    normalized = _validate_normalized_source_tables(tables, project_root=root)
    raw_identity = _json_safe_mapping(
        raw_source_identity,
        label="raw source identity",
    )
    native_builder = _json_safe_mapping(
        builder_identity,
        label="native builder identity",
    )
    _require(
        native_builder.get("stage12_binding_sha256") == binding.binding_sha256
        and native_builder.get("approval_payload_sha256")
        == authorized.payload_sha256,
        "native builder authorization identity differs",
    )
    implementation = _implementation_identity(root)

    destination = database / NORMALIZED_SOURCE_NAMESPACE
    try:
        database_metadata = database.stat(follow_symlinks=False)
    except OSError as exc:
        raise FinalNormalizedSourcePublicationError(
            "configured database root is absent"
        ) from exc
    _require(
        stat.S_ISDIR(database_metadata.st_mode)
        and not stat.S_ISLNK(database_metadata.st_mode),
        "configured database root is unsafe",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _safe_directory(
        destination.parent,
        trusted_root=database,
        label="normalized source parent",
    )
    stage = Path(
        tempfile.mkdtemp(
            prefix=".stage13-normalized-sources-",
            suffix=".partial",
            dir=destination.parent,
        )
    )
    reused = False
    try:
        _write_tables_to_directory(stage, tables=normalized)
        _source_receipts(
            stage,
            project_root=root,
            database_root=database,
        )
        if destination.exists():
            _require(
                _same_published_directory(
                    destination,
                    stage,
                    project_root=root,
                    database_root=database,
                ),
                "fixed normalized source namespace belongs to another build",
            )
            reused = True
        else:
            os.replace(stage, destination)
            parent_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    entries = _source_receipts(
        destination,
        project_root=root,
        database_root=database,
    )
    descriptor = {
        "publisher_id": PUBLISHER_ID,
        "publisher_code_hashes_sha256": implementation[
            "publisher_code_hashes_sha256"
        ],
        "publisher_config_hashes_sha256": implementation[
            "publisher_config_hashes_sha256"
        ],
        "stage12_binding_sha256": binding.binding_sha256,
        "approval_payload_sha256": authorized.payload_sha256,
        "database_root": str(database),
        "fixed_source_namespace": NORMALIZED_SOURCE_NAMESPACE.as_posix(),
        "raw_source_identity_sha256": _sha256_bytes(
            _canonical_json_bytes(raw_identity)
        ),
        "builder_identity_sha256": _sha256_bytes(
            _canonical_json_bytes(native_builder)
        ),
        "entries": list(entries),
    }
    descriptor_sha = _sha256_bytes(_canonical_json_bytes(descriptor))
    bundle_id = f"stage13-normalized-sources-{descriptor_sha[:24]}"
    manifest = {
        "schema_version": 1,
        "status": PUBLISHED_STATUS,
        "source_bundle_id": bundle_id,
        **implementation,
        "stage12_binding_sha256": binding.binding_sha256,
        "approval_payload_sha256": authorized.payload_sha256,
        "database_root": str(database),
        "fixed_source_namespace": NORMALIZED_SOURCE_NAMESPACE.as_posix(),
        "raw_source_identity": raw_identity,
        "raw_source_identity_sha256": descriptor[
            "raw_source_identity_sha256"
        ],
        "builder_identity": native_builder,
        "builder_identity_sha256": descriptor["builder_identity_sha256"],
        "entries": list(entries),
    }
    manifest_payload = _canonical_json_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_payload)
    manifest_relative = (
        NORMALIZED_SOURCE_MANIFEST_NAMESPACE / f"{bundle_id}.json"
    )
    manifest_path = metadata_root / manifest_relative
    if manifest_path.exists():
        observed = _stable_file_bytes(
            manifest_path,
            trusted_root=metadata_root,
            label="normalized source manifest",
        )
        _require(
            observed == manifest_payload,
            "content-addressed normalized source manifest differs",
        )
    else:
        _atomic_json(manifest_path, manifest, root=metadata_root)

    # Re-open the destination after the immutable manifest exists and before
    # the visible pointer is replaced.
    final_entries = _source_receipts(
        destination,
        project_root=root,
        database_root=database,
    )
    _require(
        list(final_entries) == list(entries),
        "normalized sources changed before pointer publication",
    )
    pointer = {
        "schema_version": 1,
        "status": VERIFIED_STATUS,
        "source_bundle_id": bundle_id,
        "source_manifest_path": manifest_relative.as_posix(),
        "source_manifest_sha256": manifest_sha,
        "source_manifest_size_bytes": len(manifest_payload),
        "stage12_binding_sha256": binding.binding_sha256,
        "approval_payload_sha256": authorized.payload_sha256,
        "difference_count": 0,
    }
    pointer_payload = _atomic_json(
        metadata_root / NORMALIZED_SOURCE_POINTER,
        pointer,
        root=metadata_root,
    )
    return FinalNormalizedSourcePublication(
        source_bundle_id=bundle_id,
        source_manifest_path=manifest_relative.as_posix(),
        source_manifest_sha256=manifest_sha,
        source_pointer_sha256=_sha256_bytes(pointer_payload),
        output_namespace=NORMALIZED_SOURCE_NAMESPACE.as_posix(),
        reused=reused,
    )


def _classify_existing_pointer(
    *,
    metadata_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
) -> str:
    """Return ``current`` or authenticate and archive the exact predecessor."""

    pointer_path = metadata_root / NORMALIZED_SOURCE_POINTER
    pointer_payload = _stable_file_bytes(
        pointer_path,
        trusted_root=metadata_root,
        label="normalized source verified pointer",
    )
    try:
        pointer_document = json.loads(pointer_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalNormalizedSourcePublicationError(
            "normalized source verified pointer is invalid"
        ) from exc
    _require(
        isinstance(pointer_document, Mapping),
        "normalized source verified pointer is invalid",
    )
    if (
        pointer_document.get("stage12_binding_sha256") == binding.binding_sha256
        and pointer_document.get("approval_payload_sha256") == approval.payload_sha256
    ):
        return "current"

    _require(
        _sha256_bytes(pointer_payload) == PREDECESSOR_SOURCE_POINTER_SHA256
        and pointer_document.get("source_manifest_sha256")
        == PREDECESSOR_SOURCE_MANIFEST_SHA256,
        "normalized source pointer is neither current nor the frozen predecessor",
    )
    manifest_relative = pointer_document.get("source_manifest_path")
    _require(
        isinstance(manifest_relative, str)
        and manifest_relative == PREDECESSOR_SOURCE_MANIFEST_PATH,
        "predecessor normalized source manifest path differs",
    )
    manifest_payload = _stable_file_bytes(
        metadata_root / manifest_relative,
        trusted_root=metadata_root,
        label="predecessor normalized source manifest",
    )
    _require(
        _sha256_bytes(manifest_payload) == PREDECESSOR_SOURCE_MANIFEST_SHA256,
        "predecessor normalized source manifest differs",
    )
    bundle_id = pointer_document.get("source_bundle_id")
    _require(
        isinstance(bundle_id, str)
        and re.fullmatch(r"stage13-normalized-sources-[0-9a-f]{24}", bundle_id)
        is not None,
        "predecessor normalized source bundle identity differs",
    )
    history_path = (
        metadata_root
        / NORMALIZED_SOURCE_POINTER_HISTORY_NAMESPACE
        / f"{bundle_id}.json"
    )
    if history_path.exists():
        _require(
            _stable_file_bytes(
                history_path,
                trusted_root=metadata_root,
                label="normalized source predecessor pointer history",
            )
            == pointer_payload,
            "immutable normalized source predecessor pointer history differs",
        )
    else:
        archived_payload = _atomic_json(
            history_path,
            dict(pointer_document),
            root=metadata_root,
        )
        _require(
            archived_payload == pointer_payload,
            "normalized source predecessor pointer history differs",
        )
    return "frozen_predecessor"


def _publish_final_normalized_sources_authenticated(
    *,
    project_root: Path,
    database_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
) -> FinalNormalizedSourcePublication:
    """Reuse current authority or transition only the exact frozen predecessor."""

    authorized = require_stage13_unblind_authorization(
        approval,
        expected_stage12_binding=binding,
    )
    pointer = project_root / NORMALIZED_SOURCE_POINTER
    if pointer.exists():
        if _classify_existing_pointer(
            metadata_root=project_root,
            binding=binding,
            approval=authorized,
        ) == "current":
            # A current pointer never falls back to rebuilding: manifest or
            # payload corruption must remain a fatal authentication failure.
            existing = _load_authenticated_normalized_sources(
                project_root=project_root,
                database_root=database_root,
                binding=binding,
                approval=authorized,
            )
            return _result_from_authenticated_identity(
                existing.source_identity,
                reused=True,
            )

    # Delayed import keeps the construction/loader layer independent of raw
    # adapters and makes the authorization-before-raw ordering observable.
    from portfolio_bot.final_test.final_source_builder import (
        FinalNormalizedSourceBuild,
        build_final_normalized_source_tables,
    )

    build = build_final_normalized_source_tables()
    _require(
        type(build) is FinalNormalizedSourceBuild,
        "native final source build authority type differs",
    )
    for field in ("tables", "raw_source_identity", "builder_identity"):
        _require(hasattr(build, field), f"native final source build lacks {field}")
    builder_stage12 = build.builder_identity.get("stage12_binding_sha256")
    builder_approval = build.builder_identity.get("approval_payload_sha256")
    _require(
        builder_stage12 == binding.binding_sha256,
        "native builder Stage 12 identity differs",
    )
    _require(
        builder_approval == authorized.payload_sha256,
        "native builder approval identity differs",
    )
    return _publish_authenticated_tables(
        build.tables,
        raw_source_identity=build.raw_source_identity,
        builder_identity=build.builder_identity,
        project_root=project_root,
        database_root=database_root,
        binding=binding,
        approval=authorized,
    )


def publish_final_normalized_sources() -> FinalNormalizedSourcePublication:
    """Sealed zero-argument raw-to-normalized build/publication entry point."""

    root = PROJECT_ROOT.absolute()
    binding = load_authenticated_stage12_binding(root)
    approval = _load_fixed_approval(root, binding=binding)
    database = _configured_database_root(root)
    return _publish_final_normalized_sources_authenticated(
        project_root=root,
        database_root=database,
        binding=binding,
        approval=approval,
    )


__all__ = [
    "FinalNormalizedSourcePublication",
    "FinalNormalizedSourcePublicationError",
    "PUBLISHER_ID",
    "publish_final_normalized_sources",
]
