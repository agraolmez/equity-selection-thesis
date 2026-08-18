"""Immutable primary publication for the roadmap Stage 8 risk/cost store."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    PublishedBundle,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.risk_cost.contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
    canonical_json_bytes,
    project_path,
    published_schemas,
    require,
    schema_sha256,
    sha256_file,
    validate_contract,
)
from portfolio_bot.risk_cost.inputs import Stage8Inputs, resolve_stage8_inputs


SUCCESS_NAME = "_SUCCESS.json"
MANIFEST_NAME = "manifest.json"
PRIMARY_CODE_PATHS = (
    "scripts/build_stage_08_risk_cost.py",
    "src/portfolio_bot/data/secure_publication.py",
    "src/portfolio_bot/features/phase_03_contract.py",
    "src/portfolio_bot/features/feature_store_contract.py",
    "src/portfolio_bot/features/label_store_contract.py",
    "src/portfolio_bot/features/label_store_inputs.py",
    "src/portfolio_bot/risk_cost/contract.py",
    "src/portfolio_bot/risk_cost/inputs.py",
    "src/portfolio_bot/risk_cost/risk_math.py",
    "src/portfolio_bot/risk_cost/cost_math.py",
    "src/portfolio_bot/risk_cost/materialization.py",
    "src/portfolio_bot/risk_cost/publication.py",
)


@dataclass(frozen=True)
class Stage8BuildIdentity:
    build_id: str
    content_fingerprint: str
    code_hashes: Mapping[str, str]
    configuration_hashes: Mapping[str, str]
    input_receipts_sha256: str


@dataclass(frozen=True)
class FileReceipt:
    path: str
    sha256: str
    size_bytes: int
    rows: int | None
    row_groups: int | None
    schema_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "row_groups": self.row_groups,
            "schema_sha256": self.schema_sha256,
        }


@dataclass(frozen=True)
class OutputPublication:
    path: Path
    bundle: PublishedBundle
    files: Mapping[str, FileReceipt]


@dataclass(frozen=True)
class PrimaryRiskCostPublication:
    build_id: str
    interim_path: Path
    processed_path: Path
    quality_path: Path
    manifest_path: Path
    pointer_path: Path
    manifest: Mapping[str, Any]
    pointer: Mapping[str, Any]
    reused_existing: bool


class _HashingSink:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self._size = 0
        self.closed = False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._size

    def write(self, payload: bytes | bytearray | memoryview) -> int:
        view = memoryview(payload)
        written = self._stream.write(view)
        require(written == len(view), "Short Stage 8 publication write")
        self._digest.update(view)
        self._size += written
        return written

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        if not self.closed:
            self.flush()
            self.closed = True

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    @property
    def size_bytes(self) -> int:
        return self._size


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_paths(paths: Sequence[str]) -> dict[str, str]:
    return {value: sha256_file(project_path(value)) for value in paths}


def compute_build_identity(
    inputs: Stage8Inputs,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> Stage8BuildIdentity:
    policy, _ = validate_contract(policy_path)
    configuration_paths = (
        str(policy_path.resolve().relative_to(PROJECT_ROOT)),
        str(policy["physical_rules"]["path"]),
        str(policy["methodology"]["source"]),
        str(policy["methodology"]["freeze_receipt"]),
        str(policy["authority"]["verified_stage07_pointer"]),
        str(policy["verified_feature_store"]["verified_pointer"]),
        str(policy["verified_feature_store"]["primary_manifest"]),
        str(policy["database"]["paths_config"]),
        str(policy["database"]["relocation_receipt"]),
    )
    code_hashes = _hash_paths(PRIMARY_CODE_PATHS)
    configuration_hashes = _hash_paths(configuration_paths)
    input_payload = canonical_json_bytes(list(inputs.input_receipts))
    input_receipts_sha256 = _sha256_bytes(input_payload)
    fingerprint_payload = {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "contract_id": policy["contract_id"],
        "authority": {
            "build_id": inputs.authority["build_id"],
            "verification_id": inputs.authority["verification_id"],
            "pointer_sha256": configuration_hashes[
                str(policy["authority"]["verified_stage07_pointer"])
            ],
        },
        "code_hashes": code_hashes,
        "configuration_hashes": configuration_hashes,
        "input_receipts": list(inputs.input_receipts),
    }
    fingerprint = _sha256_bytes(canonical_json_bytes(fingerprint_payload))
    return Stage8BuildIdentity(
        build_id=f"phase04-risk-cost-{fingerprint[:24]}",
        content_fingerprint=fingerprint,
        code_hashes=code_hashes,
        configuration_hashes=configuration_hashes,
        input_receipts_sha256=input_receipts_sha256,
    )


def _write_bytes(stage: SecureStagingDirectory, name: str, payload: bytes) -> FileReceipt:
    with stage.open_exclusive_binary(name) as stream:
        written = stream.write(payload)
        require(written == len(payload), f"Short Stage 8 JSON write: {name}")
    return FileReceipt(name, _sha256_bytes(payload), len(payload), None, None, None)


def _write_table(
    stage: SecureStagingDirectory,
    name: str,
    table: pa.Table,
    *,
    row_group_size: int,
    compression: str,
    dictionary: bool,
) -> FileReceipt:
    expected = published_schemas()[name]
    require(
        table.schema.equals(expected, check_metadata=True),
        f"Stage 8 table schema differs: {name}",
    )
    with stage.open_exclusive_binary(name) as stream:
        sink = _HashingSink(stream)
        pq.write_table(
            table,
            sink,
            row_group_size=row_group_size,
            compression=compression,
            use_dictionary=dictionary,
            version="2.6",
            write_statistics=True,
            data_page_version="1.0",
        )
        sink.flush()
    return FileReceipt(
        path=name,
        sha256=sink.sha256,
        size_bytes=sink.size_bytes,
        rows=table.num_rows,
        row_groups=max(1, math.ceil(table.num_rows / row_group_size)),
        schema_sha256=schema_sha256(expected),
    )


def _table_map(result: Any) -> dict[str, pa.Table]:
    mapping = {
        "development_risk_cost_audit.parquet": result.audit,
        "development_risk_cost_inputs.parquet": result.processed,
        "development_factor_covariance.parquet": result.factor_covariance,
        "development_risk_cost_month_summary.parquet": result.month_summary,
        "risk_cost_timing_checks.parquet": result.timing,
        "risk_cost_coverage.parquet": result.coverage,
    }
    for name, table in mapping.items():
        require(isinstance(table, pa.Table), f"Stage 8 result is not a table: {name}")
    return mapping


def _publish_output_set(
    *,
    trusted_root: Path,
    destination: Path,
    expected_files: Sequence[str],
    tables: Mapping[str, pa.Table],
    json_payloads: Mapping[str, Mapping[str, Any]],
    identity: Stage8BuildIdentity,
    location: str,
    policy: Mapping[str, Any],
) -> OutputPublication:
    receipts: dict[str, FileReceipt] = {}
    with SecureStagingDirectory.create(
        trusted_root=trusted_root,
        destination=destination,
        expected_files=expected_files,
        label=f"Stage 8 {location}",
    ) as stage:
        for name in expected_files:
            if name == SUCCESS_NAME:
                continue
            if name in tables:
                receipts[name] = _write_table(
                    stage,
                    name,
                    tables[name],
                    row_group_size=int(policy["build"]["row_group_size"]),
                    compression=str(policy["build"]["compression"]),
                    dictionary=bool(policy["build"]["dictionary_encode_strings"]),
                )
            elif name in json_payloads:
                receipts[name] = _write_bytes(
                    stage, name, canonical_json_bytes(json_payloads[name])
                )
            else:
                raise RuntimeError(f"No Stage 8 payload supplied for {name}")
        success_payload = {
            "schema_version": 1,
            "contract_id": policy["contract_id"],
            "policy_id": policy["policy_id"],
            "build_id": identity.build_id,
            "content_fingerprint": identity.content_fingerprint,
            "location": location,
            "status": f"PRIMARY_{location.upper()}_COMPLETE",
            "files": {name: receipt.as_dict() for name, receipt in sorted(receipts.items())},
        }
        success = _write_bytes(stage, SUCCESS_NAME, canonical_json_bytes(success_payload))
        hashes = {SUCCESS_NAME: success.sha256, **{k: v.sha256 for k, v in receipts.items()}}
        snapshot = stage.snapshot(expected_hashes=hashes)
        bundle = stage.publish(expected_hashes=snapshot.hashes)
    return OutputPublication(destination, bundle, receipts)


def _output_manifest(value: OutputPublication, database_root: Path) -> dict[str, Any]:
    return {
        "path": value.path.relative_to(database_root).as_posix(),
        "success_sha256": value.bundle.snapshot.hashes[SUCCESS_NAME],
        "files": {name: receipt.as_dict() for name, receipt in sorted(value.files.items())},
    }


def _publish_manifest(
    destination: Path,
    manifest: Mapping[str, Any],
    *,
    gate: Any,
) -> PublishedBundle:
    payload = canonical_json_bytes(manifest)
    with SecureStagingDirectory.create(
        trusted_root=PROJECT_ROOT,
        destination=destination,
        expected_files=(MANIFEST_NAME,),
        label="Stage 8 primary manifest",
    ) as stage:
        receipt = _write_bytes(stage, MANIFEST_NAME, payload)
        snapshot = stage.snapshot(expected_hashes={MANIFEST_NAME: receipt.sha256})
        return stage.publish(
            expected_hashes=snapshot.hashes,
            pre_publish_gate=gate,
            post_publish_gate=gate,
        )


def publish_primary_risk_cost(
    inputs: Stage8Inputs,
    result: Any,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> PrimaryRiskCostPublication:
    """Publish three immutable output sets, manifest, then primary pointer."""

    policy, rules = validate_contract(policy_path)
    identity = compute_build_identity(inputs, policy_path=policy_path)
    tables = _table_map(result)
    gates = dict(result.metrics.get("hard_gates", {}))
    expected_gates = policy["quality_gates"]
    for name, expected in expected_gates.items():
        require(name in gates, f"Stage 8 materialization omitted gate {name}")
        if isinstance(expected, bool):
            require(gates[name] is expected, f"Stage 8 gate failed: {name}")
        else:
            require(gates[name] == expected, f"Stage 8 gate failed: {name}")

    database_root = inputs.database_root
    interim_path = database_root / policy["database"]["interim_build_root"] / identity.build_id
    processed_path = database_root / policy["database"]["processed_build_root"] / identity.build_id
    quality_path = database_root / policy["database"]["quality_build_root"] / identity.build_id

    interim = _publish_output_set(
        trusted_root=database_root,
        destination=interim_path,
        expected_files=tuple(policy["outputs"]["interim"]["required_files"]),
        tables={"development_risk_cost_audit.parquet": tables["development_risk_cost_audit.parquet"]},
        json_payloads={},
        identity=identity,
        location="interim",
        policy=policy,
    )
    processed = _publish_output_set(
        trusted_root=database_root,
        destination=processed_path,
        expected_files=tuple(policy["outputs"]["processed"]["required_files"]),
        tables={name: tables[name] for name in (
            "development_risk_cost_inputs.parquet",
            "development_factor_covariance.parquet",
            "development_risk_cost_month_summary.parquet",
        )},
        json_payloads={"model_parameters.json": dict(result.model_parameters)},
        identity=identity,
        location="processed",
        policy=policy,
    )
    quality = _publish_output_set(
        trusted_root=database_root,
        destination=quality_path,
        expected_files=tuple(policy["outputs"]["quality"]["required_files"]),
        tables={name: tables[name] for name in (
            "risk_cost_timing_checks.parquet",
            "risk_cost_coverage.parquet",
        )},
        json_payloads={"quality_summary.json": dict(result.metrics)},
        identity=identity,
        location="quality",
        policy=policy,
    )

    manifest = {
        "schema_version": 1,
        "contract_id": policy["contract_id"],
        "policy_id": policy["policy_id"],
        "build_id": identity.build_id,
        "status": "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
        "authorizations": {
            "independent_verification_required": True,
            "stage_09_model_training_authorized": False,
            "stage_10_or_later_authorized": False,
            "models_portfolios_pnl_or_backtests_authorized": False,
            "validation_or_final_test_access_authorized": False,
        },
        "bindings": {
            "content_fingerprint": identity.content_fingerprint,
            "code_hashes": dict(identity.code_hashes),
            "configuration_hashes": dict(identity.configuration_hashes),
            "input_receipts_sha256": identity.input_receipts_sha256,
        },
        "inputs": {
            "authority_build_id": inputs.authority["build_id"],
            "authority_verification_id": inputs.authority["verification_id"],
            "table_receipts": list(inputs.input_receipts),
            "label_payload_rows_read": 0,
        },
        "outputs": {
            "interim": _output_manifest(interim, database_root),
            "processed": _output_manifest(processed, database_root),
            "quality": _output_manifest(quality, database_root),
        },
        "summary": dict(result.metrics.get("summary", {})),
        "hard_gates": gates,
    }

    artifact_root = project_path(policy["publication"]["artifact_root"])
    manifest_directory = artifact_root / policy["publication"]["primary_build_namespace"] / identity.build_id
    pointer_path = project_path(policy["publication"]["latest_primary_pointer"])

    def reauthenticate() -> None:
        current = resolve_stage8_inputs(policy_path)
        require(current.input_receipts == inputs.input_receipts, "Stage 8 inputs changed")
        require(current.authority == inputs.authority, "Stage 8 authority changed")

    artifact_bundle = _publish_manifest(manifest_directory, manifest, gate=reauthenticate)
    manifest_path = manifest_directory / MANIFEST_NAME
    manifest_sha = artifact_bundle.snapshot.hashes[MANIFEST_NAME]
    pointer = {
        "schema_version": 1,
        "contract_id": policy["contract_id"],
        "policy_id": policy["policy_id"],
        "build_id": identity.build_id,
        "status": "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
        "primary_manifest_sha256": manifest_sha,
        "authorizations": {
            "independent_verification_completed": False,
            "stage_09_model_training_authorized": False,
            "stage_10_or_later_authorized": False,
            "models_portfolios_pnl_or_backtests_authorized": False,
            "validation_or_final_test_access_authorized": False,
        },
    }
    replaced = replace_pointer_last(
        pointer_path,
        canonical_json_bytes(pointer),
        trusted_root=PROJECT_ROOT,
        pre_replace_gate=reauthenticate,
        post_replace_gate=reauthenticate,
        label="Stage 8 latest primary pointer",
    )
    reused = (
        interim.bundle.reused_existing
        and processed.bundle.reused_existing
        and quality.bundle.reused_existing
        and artifact_bundle.reused_existing
        and not replaced
    )
    return PrimaryRiskCostPublication(
        build_id=identity.build_id,
        interim_path=interim_path,
        processed_path=processed_path,
        quality_path=quality_path,
        manifest_path=manifest_path,
        pointer_path=pointer_path,
        manifest=manifest,
        pointer=pointer,
        reused_existing=reused,
    )
