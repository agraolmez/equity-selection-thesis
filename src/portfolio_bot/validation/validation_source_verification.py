"""Independent Gate-A verification of main projections and reserve extraction."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    SecurePublicationError,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.validation.validation_source_extraction import (
    recompute_normalized_stage11_sources_for_verification,
    schema_sha256,
)
from portfolio_bot.validation.validation_source_materialization import (
    PROJECT_ROOT,
    Stage11SourceMaterializationAuthority,
    canonical_json_bytes,
    require_live_stage11_source_materialization_authority,
    sha256_bytes,
    sha256_file,
)
from portfolio_bot.validation.validation_source_publication import (
    MANIFEST_NAME,
    PRIMARY_POINTER_PATH,
    PRIMARY_STATUS,
    PUBLICATION_ID,
    SUCCESS_NAME,
)


VERIFICATION_IDENTITY = "phase05-stage11-source-materialization-independent-verifier-v1"
VERIFIED_STATUS = "STAGE11_SOURCE_MATERIALIZATION_VERIFIED"
FAILED_STATUS = "STAGE11_SOURCE_MATERIALIZATION_VERIFICATION_FAILED"
VERIFIED_POINTER_PATH = (
    PROJECT_ROOT
    / "artifacts/phase_05_validation/STAGE_11_SOURCE_MATERIALIZATION_VERIFIED.json"
)
REPORT_NAME = "verification_report.json"
DIFFERENCES_NAME = "differences.json"
VERIFICATION_MEMBERS = (REPORT_NAME, DIFFERENCES_NAME, SUCCESS_NAME)


class Stage11SourceVerificationError(RuntimeError):
    """Raised when the primary source bundle or independent replay is unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11SourceVerificationError(message)


def _stable_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    _require(path.is_absolute(), f"{label} path is not absolute")
    before = os.lstat(path)
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, f"unsafe {label}")
    payload = path.read_bytes()
    after = os.lstat(path)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    _require(identity(before) == identity(after), f"{label} changed while read")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11SourceVerificationError(f"invalid {label} JSON") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value, sha256_bytes(payload)


def _semantic_parquet_equal(left: Path, right: Path) -> bool:
    try:
        a = pq.ParquetFile(left)
        b = pq.ParquetFile(right)
        if a.metadata.num_rows != b.metadata.num_rows or not a.schema_arrow.equals(
            b.schema_arrow, check_metadata=True
        ):
            return False
        left_batches = a.iter_batches(batch_size=65_536, use_threads=False)
        right_batches = b.iter_batches(batch_size=65_536, use_threads=False)
        while True:
            try:
                left_batch = next(left_batches)
            except StopIteration:
                left_batch = None
            try:
                right_batch = next(right_batches)
            except StopIteration:
                right_batch = None
            if left_batch is None or right_batch is None:
                return left_batch is None and right_batch is None
            if not left_batch.equals(right_batch):
                return False
    except (OSError, pa.ArrowException):
        return False


@dataclass(frozen=True, slots=True)
class Stage11SourceVerification:
    verification_id: str
    bundle_path: Path
    report_sha256: str
    difference_count: int
    verified_pointer_sha256: str | None
    reused: bool


def verify_stage11_normalized_sources(
    authority: Stage11SourceMaterializationAuthority,
) -> Stage11SourceVerification:
    """Reopen the primary bundle and independently replay every normalization."""

    live = require_live_stage11_source_materialization_authority(authority)
    plan = live.plan
    pointer, primary_pointer_sha = _stable_json(
        PRIMARY_POINTER_PATH,
        label="Gate-A primary pointer",
    )
    _require(
        pointer.get("publication_id") == PUBLICATION_ID
        and pointer.get("status") == PRIMARY_STATUS
        and pointer.get("stage") == 11
        and pointer.get("gate") == "A"
        and pointer.get("authority_sha256") == live.authority_sha256
        and pointer.get("plan_sha256") == live.plan_sha256
        and pointer.get("gate_b_validation_execution_authorized") is False
        and pointer.get("final_2020_2025_authorized") is False,
        "Gate-A primary pointer identity or authority differs",
    )
    bundle = Path(str(pointer["bundle_path"]))
    expected_bundle = Path(str(plan["publication"]["primary_bundle_root"])) / str(
        pointer["build_id"]
    )
    _require(
        bundle == expected_bundle
        and bundle.is_absolute()
        and bundle.is_dir()
        and not bundle.is_symlink(),
        "Gate-A primary bundle is unsafe or outside the fixed root",
    )
    names = tuple(sorted(plan["outputs"]))
    expected_member_order = (
        *(f"{name}.parquet" for name in names),
        MANIFEST_NAME,
        SUCCESS_NAME,
    )
    expected_members = set(expected_member_order)
    _require(
        {path.name for path in bundle.iterdir()} == expected_members,
        "Gate-A primary bundle is not a closed set",
    )
    pointer_files = pointer.get("bundle_files")
    _require(
        isinstance(pointer_files, dict) and set(pointer_files) == expected_members,
        "Gate-A primary pointer closed-member set differs",
    )
    observed_hashes = {name: sha256_file(bundle / name) for name in expected_members}
    _require(
        observed_hashes == pointer_files,
        "Gate-A primary bundle differs from its pointer hashes",
    )
    manifest, manifest_sha = _stable_json(
        bundle / MANIFEST_NAME, label="Gate-A primary manifest"
    )
    _require(
        manifest_sha == pointer["manifest_sha256"]
        and manifest.get("build_id") == pointer["build_id"]
        and manifest.get("authority", {}).get("authority_sha256")
        == live.authority_sha256
        and manifest.get("authority", {}).get("plan_sha256") == live.plan_sha256,
        "Gate-A primary manifest binding differs",
    )
    _require(
        manifest.get("implementation_code_hashes")
        == dict(plan["implementation_code_hashes"])
        and manifest.get("upstream_authorities") == dict(plan["upstream_authorities"])
        and manifest.get("source_identities") == dict(plan["sources"])
        and manifest.get("scope") == dict(plan["scope"])
        and manifest.get("authorizations")
        == {
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
        "Gate-A primary manifest plan bindings differ",
    )
    success, success_sha = _stable_json(
        bundle / SUCCESS_NAME,
        label="Gate-A primary success receipt",
    )
    _require(
        success_sha == pointer_files[SUCCESS_NAME]
        and success.get("status") == PRIMARY_STATUS
        and success.get("build_id") == pointer["build_id"]
        and success.get("manifest_sha256") == manifest_sha
        and isinstance(success.get("closed_members"), list)
        and tuple(success["closed_members"]) == expected_member_order,
        "Gate-A primary success receipt differs",
    )
    differences: list[str] = []
    primary_receipts = manifest.get("tables")
    if not isinstance(primary_receipts, dict) or set(primary_receipts) != set(names):
        raise Stage11SourceVerificationError("Gate-A primary receipt set differs")
    execution_records = manifest.get("execution_input_records")
    _require(
        isinstance(execution_records, dict) and set(execution_records) == set(names),
        "Gate-A execution-record set differs",
    )
    for name in names:
        record = primary_receipts[name]
        _require(
            isinstance(record, dict), f"Gate-A primary table receipt is invalid: {name}"
        )
        path = bundle / f"{name}.parquet"
        contract = plan["outputs"][name]
        kind = name.removeprefix("validation_").rsplit("_", 1)[0]
        expected_role = {
            "sessions": "validation_trading_session_metadata",
            "market": "validation_market_input",
            "returns": "validation_forward_return",
            "factors": "validation_accounting_input",
            "delistings": "validation_accounting_input",
        }[kind]
        expected_execution_record = {
            "role": expected_role,
            "path": str(path),
            "sha256": record.get("sha256"),
            "size_bytes": record.get("size_bytes"),
            "allowed_columns": list(contract["allowed_columns"]),
            "date_column": contract["date_column"],
            "date_axis": "observation_or_exit_date",
            "minimum_date": contract["minimum_date"],
            "maximum_date": contract["maximum_date"],
            "holding_month_column": None,
            "trusted_schema_sha256": None,
        }
        if execution_records[name] != expected_execution_record:
            differences.append(f"execution_record:{name}")
        if (
            not path.is_file()
            or path.is_symlink()
            or observed_hashes[path.name] != record.get("sha256")
            or path.stat().st_size != record.get("size_bytes")
        ):
            differences.append(f"primary_physical:{name}")
            continue
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != record.get("rows") or schema_sha256(
            parquet.schema_arrow
        ) != record.get("schema_sha256"):
            differences.append(f"primary_shape:{name}")

    # This has a distinct content-addressed work root and purpose domain.  It
    # independently reauthenticates the main upstream receipts, projects them
    # again, and reruns the raw reserve slicer/reconciler.  It cannot reuse the
    # primary normalized files.
    replay = recompute_normalized_stage11_sources_for_verification(live)
    for name in names:
        primary = bundle / f"{name}.parquet"
        independent = replay.normalized_root / f"{name}.parquet"
        if sha256_file(primary) != sha256_file(independent):
            differences.append(f"byte_replay:{name}")
            if not _semantic_parquet_equal(primary, independent):
                differences.append(f"semantic_replay:{name}")
    differences = sorted(set(differences))
    status = VERIFIED_STATUS if not differences else FAILED_STATUS
    report_identity = {
        "schema_version": 1,
        "verification_identity": VERIFICATION_IDENTITY,
        "status": status,
        "stage": 11,
        "gate": "A",
        "primary_build_id": pointer["build_id"],
        "primary_pointer_sha256": primary_pointer_sha,
        "primary_manifest_sha256": manifest_sha,
        "authority_sha256": live.authority_sha256,
        "plan_sha256": live.plan_sha256,
        "independent_replay_build_id": replay.build_id,
        "independent_replay_input_fingerprint": replay.input_fingerprint,
        "independent_replay_receipts": dict(replay.receipts),
        "difference_count": len(differences),
        "differences": differences,
        "checks": {
            "primary_closed_set_authenticated": True,
            "main_upstream_receipts_reauthenticated_and_reprojected": True,
            "reserve_raw_extraction_independently_replayed": True,
            "all_output_byte_hashes_compared": True,
            "semantic_fallback_compared_on_byte_difference": True,
            "labels_forecasts_weights_trades_pnl_performance_created": False,
            "final_2020_plus_access_authorized": False,
            "gate_b_execution_authorized": False,
        },
    }
    verification_id = (
        "verify-stage11-sources-"
        + sha256_bytes(canonical_json_bytes(report_identity))[:20]
    )
    report = {**report_identity, "verification_id": verification_id}
    report_bytes = canonical_json_bytes(report)
    difference_bytes = canonical_json_bytes(
        {
            "schema_version": 1,
            "difference_count": len(differences),
            "differences": differences,
        }
    )
    success_bytes = canonical_json_bytes(
        {
            "schema_version": 1,
            "status": status,
            "verification_id": verification_id,
            "report_sha256": sha256_bytes(report_bytes),
            "differences_sha256": sha256_bytes(difference_bytes),
            "primary_manifest_sha256": manifest_sha,
            "closed_members": list(VERIFICATION_MEMBERS),
        }
    )
    verification_root = Path(str(plan["publication"]["verification_bundle_root"]))
    destination = verification_root / verification_id
    verification_root.mkdir(parents=True, exist_ok=True)
    payloads = {
        REPORT_NAME: report_bytes,
        DIFFERENCES_NAME: difference_bytes,
        SUCCESS_NAME: success_bytes,
    }
    try:
        with SecureStagingDirectory.create(
            trusted_root=verification_root,
            destination=destination,
            expected_files=VERIFICATION_MEMBERS,
            label="Stage 11 Gate-A source verification bundle",
        ) as stage:
            for name in VERIFICATION_MEMBERS:
                with stage.open_exclusive_binary(name) as stream:
                    stream.write(payloads[name])
            hashes = {name: sha256_bytes(payload) for name, payload in payloads.items()}
            snapshot = stage.snapshot(expected_hashes=hashes)
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
        raise Stage11SourceVerificationError(
            "secure Gate-A verification publication failed"
        ) from exc

    verified_pointer_sha: str | None = None
    if not differences:
        verified_pointer = {
            "schema_version": 1,
            "verification_identity": VERIFICATION_IDENTITY,
            "status": VERIFIED_STATUS,
            "stage": 11,
            "gate": "A",
            "primary_build_id": pointer["build_id"],
            "primary_pointer_sha256": primary_pointer_sha,
            "primary_manifest_sha256": manifest_sha,
            "verification_id": verification_id,
            "verification_bundle_path": str(destination),
            "verification_bundle_files": dict(published.snapshot.hashes),
            "verification_report_sha256": sha256_bytes(report_bytes),
            "difference_count": 0,
            "authority_sha256": live.authority_sha256,
            "plan_sha256": live.plan_sha256,
            "gate_b_validation_execution_authorized": False,
            "final_2020_2025_authorized": False,
        }
        pointer_bytes = canonical_json_bytes(verified_pointer)
        try:
            replace_pointer_last(
                VERIFIED_POINTER_PATH,
                pointer_bytes,
                trusted_root=PROJECT_ROOT,
                pre_replace_gate=lambda: require_live_stage11_source_materialization_authority(
                    live
                ),
                post_replace_gate=lambda: require_live_stage11_source_materialization_authority(
                    live
                ),
                label="Stage 11 Gate-A verified pointer",
            )
        except SecurePublicationError as exc:
            raise Stage11SourceVerificationError(
                "Gate-A verified pointer publication failed"
            ) from exc
        verified_pointer_sha = sha256_bytes(pointer_bytes)
    return Stage11SourceVerification(
        verification_id=verification_id,
        bundle_path=destination,
        report_sha256=sha256_bytes(report_bytes),
        difference_count=len(differences),
        verified_pointer_sha256=verified_pointer_sha,
        reused=published.reused_existing,
    )


__all__ = [
    "DIFFERENCES_NAME",
    "FAILED_STATUS",
    "REPORT_NAME",
    "VERIFICATION_IDENTITY",
    "VERIFICATION_MEMBERS",
    "VERIFIED_POINTER_PATH",
    "VERIFIED_STATUS",
    "Stage11SourceVerification",
    "Stage11SourceVerificationError",
    "verify_stage11_normalized_sources",
]
