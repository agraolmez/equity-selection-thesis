"""Immutable publication of the ten normalized Stage 11 Gate-A sources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from portfolio_bot.data.secure_publication import (
    SecurePublicationError,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.validation.validation_source_extraction import NormalizedSourceBuild
from portfolio_bot.validation.validation_source_materialization import (
    PROJECT_ROOT,
    Stage11SourceMaterializationAuthority,
    canonical_json_bytes,
    require_live_stage11_source_materialization_authority,
    sha256_bytes,
    sha256_file,
)


PUBLICATION_ID = "phase05-stage11-source-materialization-publication-v1"
PRIMARY_STATUS = "STAGE11_SOURCE_MATERIALIZATION_COMPLETE"
MANIFEST_NAME = "manifest.json"
SUCCESS_NAME = "_SUCCESS.json"
PRIMARY_POINTER_PATH = (
    PROJECT_ROOT
    / "artifacts/phase_05_validation/STAGE_11_SOURCE_MATERIALIZATION_PRIMARY.json"
)


class Stage11SourcePublicationError(RuntimeError):
    """Raised when a Gate-A bundle is incomplete, mutable, or out of scope."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11SourcePublicationError(message)


def _execution_record(
    name: str,
    receipt: Mapping[str, Any],
    destination: Path,
    output_contract: Mapping[str, Any],
) -> dict[str, Any]:
    kind = name.removeprefix("validation_").rsplit("_", 1)[0]
    role = {
        "sessions": "validation_trading_session_metadata",
        "market": "validation_market_input",
        "returns": "validation_forward_return",
        "factors": "validation_accounting_input",
        "delistings": "validation_accounting_input",
    }[kind]
    date_column = {
        "sessions": "calendar_date",
        "market": "calendar_date",
        "returns": "calendar_date",
        "factors": "observation_date",
        "delistings": "assigned_return_date",
    }[kind]
    return {
        "role": role,
        "path": str(destination / str(receipt["path"])),
        "sha256": receipt["sha256"],
        "size_bytes": receipt["size_bytes"],
        "allowed_columns": list(output_contract["allowed_columns"]),
        "date_column": date_column,
        "date_axis": "observation_or_exit_date",
        "minimum_date": output_contract["minimum_date"],
        "maximum_date": output_contract["maximum_date"],
        "holding_month_column": None,
        "trusted_schema_sha256": None,
    }


@dataclass(frozen=True, slots=True)
class Stage11SourcePublication:
    build_id: str
    bundle_path: Path
    manifest_sha256: str
    pointer_sha256: str
    reused: bool


def publish_stage11_normalized_sources(
    authority: Stage11SourceMaterializationAuthority,
    build: NormalizedSourceBuild,
) -> Stage11SourcePublication:
    """Publish a closed source-only bundle and replace its pointer last."""

    live = require_live_stage11_source_materialization_authority(authority)
    _require(
        type(build) is NormalizedSourceBuild,
        "Gate-A publication requires an exact normalized build",
    )
    plan = live.plan
    publication = plan["publication"]
    bundle_root = Path(str(publication["primary_bundle_root"]))
    destination = bundle_root / build.build_id
    names = tuple(sorted(plan["outputs"]))
    table_files = tuple(f"{name}.parquet" for name in names)
    expected_files = (*table_files, MANIFEST_NAME, SUCCESS_NAME)

    source_records: dict[str, Any] = {}
    for name in names:
        receipt = dict(build.receipts[name])
        _require(
            receipt["path"] == f"{name}.parquet", f"normalized filename differs: {name}"
        )
        source = build.normalized_root / receipt["path"]
        _require(
            source.is_file()
            and not source.is_symlink()
            and sha256_file(source) == receipt["sha256"],
            f"normalized source changed before publication: {name}",
        )
        source_records[name] = receipt

    manifest = {
        "schema_version": 1,
        "publication_id": PUBLICATION_ID,
        "status": PRIMARY_STATUS,
        "stage": 11,
        "gate": "A",
        "build_id": build.build_id,
        "input_fingerprint": build.input_fingerprint,
        "authority": {
            "authority_sha256": live.authority_sha256,
            "approval_file_sha256": live.approval_file_sha256,
            "approval_payload_sha256": live.approval_payload_sha256,
            "approval_text_sha256": live.approval_text_sha256,
            "plan_sha256": live.plan_sha256,
            "plan_payload_sha256": live.plan_payload_sha256,
        },
        "implementation_code_hashes": dict(plan["implementation_code_hashes"]),
        "upstream_authorities": dict(plan["upstream_authorities"]),
        "source_identities": dict(plan["sources"]),
        "tables": source_records,
        "execution_input_records": {
            name: _execution_record(
                name,
                source_records[name],
                destination,
                plan["outputs"][name],
            )
            for name in names
        },
        "scope": dict(plan["scope"]),
        "authorizations": {
            "source_materialization_only": True,
            "gate_b_validation_execution_authorized": False,
            "labels_or_targets_created": False,
            "forecasts_or_model_selection_created": False,
            "portfolio_weights_or_trades_created": False,
            "pnl_or_performance_created": False,
            "december_2019_holding_month_included": False,
            "final_2020_2025_included": False,
            "stage12_or_later_authorized": False,
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha = sha256_bytes(manifest_bytes)
    success = {
        "schema_version": 1,
        "status": PRIMARY_STATUS,
        "build_id": build.build_id,
        "manifest_sha256": manifest_sha,
        "closed_members": list(expected_files),
    }
    success_bytes = canonical_json_bytes(success)

    bundle_root.mkdir(parents=True, exist_ok=True)
    try:
        with SecureStagingDirectory.create(
            trusted_root=bundle_root,
            destination=destination,
            expected_files=expected_files,
            label="Stage 11 Gate-A source bundle",
        ) as stage:
            for name in names:
                filename = f"{name}.parquet"
                with stage.open_exclusive_binary(filename) as stream:
                    with (build.normalized_root / filename).open("rb") as source:
                        while block := source.read(8 << 20):
                            stream.write(block)
            with stage.open_exclusive_binary(MANIFEST_NAME) as stream:
                stream.write(manifest_bytes)
            with stage.open_exclusive_binary(SUCCESS_NAME) as stream:
                stream.write(success_bytes)
            expected_hashes = {
                **{f"{name}.parquet": source_records[name]["sha256"] for name in names},
                MANIFEST_NAME: manifest_sha,
                SUCCESS_NAME: sha256_bytes(success_bytes),
            }
            snapshot = stage.snapshot(expected_hashes=expected_hashes)
            published = stage.publish(
                expected_hashes=snapshot.hashes,
                pre_publish_gate=lambda: require_live_stage11_source_materialization_authority(
                    live
                ),
                post_publish_gate=lambda: require_live_stage11_source_materialization_authority(
                    live
                ),
            )
    except SecurePublicationError as exc:
        raise Stage11SourcePublicationError(
            "secure Gate-A source publication failed"
        ) from exc

    pointer = {
        "schema_version": 1,
        "publication_id": PUBLICATION_ID,
        "status": PRIMARY_STATUS,
        "stage": 11,
        "gate": "A",
        "build_id": build.build_id,
        "bundle_path": str(destination),
        "bundle_files": dict(published.snapshot.hashes),
        "manifest_sha256": manifest_sha,
        "authority_sha256": live.authority_sha256,
        "plan_sha256": live.plan_sha256,
        "gate_b_validation_execution_authorized": False,
        "final_2020_2025_authorized": False,
    }
    pointer_bytes = canonical_json_bytes(pointer)
    try:
        replace_pointer_last(
            PRIMARY_POINTER_PATH,
            pointer_bytes,
            trusted_root=PROJECT_ROOT,
            pre_replace_gate=lambda: require_live_stage11_source_materialization_authority(
                live
            ),
            post_replace_gate=lambda: require_live_stage11_source_materialization_authority(
                live
            ),
            label="Stage 11 Gate-A primary pointer",
        )
    except SecurePublicationError as exc:
        raise Stage11SourcePublicationError(
            "Gate-A primary pointer publication failed"
        ) from exc
    return Stage11SourcePublication(
        build_id=build.build_id,
        bundle_path=destination,
        manifest_sha256=manifest_sha,
        pointer_sha256=sha256_bytes(pointer_bytes),
        reused=published.reused_existing,
    )


def load_primary_manifest(
    authority: Stage11SourceMaterializationAuthority,
    publication: Stage11SourcePublication,
) -> Mapping[str, Any]:
    """Reopen and authenticate the exact primary manifest inside a live scope."""

    require_live_stage11_source_materialization_authority(authority)
    path = publication.bundle_path / MANIFEST_NAME
    _require(
        path.is_file() and not path.is_symlink(), "Gate-A primary manifest is absent"
    )
    payload = path.read_bytes()
    _require(
        sha256_bytes(payload) == publication.manifest_sha256,
        "Gate-A primary manifest changed",
    )
    value = json.loads(payload)
    _require(
        isinstance(value, dict) and value.get("build_id") == publication.build_id,
        "Gate-A primary manifest identity differs",
    )
    return MappingProxyType(value)


__all__ = [
    "MANIFEST_NAME",
    "PRIMARY_POINTER_PATH",
    "PRIMARY_STATUS",
    "PUBLICATION_ID",
    "SUCCESS_NAME",
    "Stage11SourcePublication",
    "Stage11SourcePublicationError",
    "load_primary_manifest",
    "publish_stage11_normalized_sources",
]
