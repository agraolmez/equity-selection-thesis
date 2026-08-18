"""Independent byte replay and publication for the Stage 12 pre-final freeze."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from portfolio_bot.data.secure_publication import (
    PublishedBundle,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.final_freeze.inventory import (
    Stage12InventoryError,
    build_environment_lock,
    build_project_file_inventory,
    canonical_json_bytes,
    confined_existing_path,
    confined_relative_path,
    load_json_object,
    sha256_bytes,
    stable_file,
)
from portfolio_bot.final_freeze.publication import (
    DEFAULT_POLICY,
    PRIMARY_FILES,
    PROJECT_ROOT,
    PreparedFreeze,
    Stage12FreezeError,
    _STAGE13_BLOCKERS,
    _load_policy,
    prepare_stage12_freeze,
)
from portfolio_bot.final_freeze.regression import SANITIZED_ENVIRONMENT


VERIFICATION_FILES = (
    "_SUCCESS.json",
    "byte_differences.json",
    "verification_report.json",
)


@dataclass(frozen=True, slots=True)
class PrimaryReplay:
    prepared: PreparedFreeze
    pointer: Mapping[str, Any]
    pointer_sha256: str
    bundle_path: Path
    difference_count: int
    differences: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class Stage12VerificationPublication:
    verification_id: str
    bundle: PublishedBundle
    pointer_path: Path
    pointer_sha256: str
    report_sha256: str


def _semantic_require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage12FreezeError(message)


def _embedded_amendment_semantics_match(
    amendment: Mapping[str, Any], *, expected_semantic_sha256: str
) -> bool:
    """Recompute embedded amendment semantics independently of source formatting."""

    return sha256_bytes(canonical_json_bytes(amendment)) == expected_semantic_sha256


def _read_canonical_primary_documents(
    bundle_path: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    documents: dict[str, Mapping[str, Any]] = {}
    receipts: dict[str, Mapping[str, Any]] = {}
    for name in PRIMARY_FILES:
        try:
            member = confined_relative_path(
                bundle_path,
                name,
                file_required=True,
                label="primary Stage 12 semantic member",
            )
            record = stable_file(
                member,
                keep_payload=True,
                trusted_root=bundle_path,
            )
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(
                f"cannot safely read primary semantic member: {name}"
            ) from exc
        assert record.payload is not None
        try:
            value = json.loads(record.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage12FreezeError(f"invalid primary JSON member: {name}") from exc
        if not isinstance(value, Mapping):
            raise Stage12FreezeError(f"primary member is not an object: {name}")
        if canonical_json_bytes(value) != record.payload:
            raise Stage12FreezeError(f"primary member is not canonical JSON: {name}")
        documents[name] = value
        receipts[name] = {
            "sha256": record.sha256,
            "size_bytes": record.size_bytes,
        }
    return documents, receipts


def _independently_validate_legacy_v11_primary_semantics(
    *,
    bundle_path: Path,
    policy: Mapping[str, Any],
    policy_sha256: str,
    pointer: Mapping[str, Any],
) -> None:
    """Retain the unpublished v11 semantic audit for historical forensics only."""

    documents, receipts = _read_canonical_primary_documents(bundle_path)
    success = documents["_SUCCESS.json"]
    manifest = documents["pre_final_manifest.json"]
    content_receipts = {
        name: receipt
        for name, receipt in receipts.items()
        if name not in {"_SUCCESS.json", "pre_final_manifest.json"}
    }
    success_receipts = {
        name: receipt for name, receipt in receipts.items() if name != "_SUCCESS.json"
    }
    _semantic_require(
        success.get("members") == success_receipts,
        "primary success member receipt graph differs",
    )
    _semantic_require(
        manifest.get("members") == content_receipts,
        "primary manifest member receipt graph differs",
    )
    bindings = manifest.get("bindings")
    _semantic_require(isinstance(bindings, Mapping), "primary bindings are missing")
    assert isinstance(bindings, Mapping)
    methodology = bindings.get("methodology")
    _semantic_require(
        isinstance(methodology, Mapping), "primary methodology binding is missing"
    )
    assert isinstance(methodology, Mapping)
    fingerprint_basis = {
        "schema_version": 1,
        "policy_sha256": policy_sha256,
        "methodology_source_sha256": (
            methodology.get("source", {}).get("sha256")
            if isinstance(methodology.get("source"), Mapping)
            else None
        ),
        "project_tree_sha256": documents["file_inventory.json"].get("tree_sha256"),
        "environment_sha256": sha256_bytes(
            canonical_json_bytes(documents["environment_lock.json"])
        ),
        "members": content_receipts,
    }
    content_fingerprint = sha256_bytes(canonical_json_bytes(fingerprint_basis))
    freeze_id = f"stage12-freeze-{content_fingerprint[:24]}"
    _semantic_require(
        manifest.get("schema_version") == 1
        and manifest.get("stage") == 12
        and manifest.get("policy_id") == policy.get("policy_id")
        and manifest.get("status") == policy["outputs"]["primary_status"]
        and manifest.get("scope")
        == "RULE_RESTORING_ECONOMIC_POST_UNBLIND_REPLACEMENT_FREEZE"
        and manifest.get("content_fingerprint") == content_fingerprint
        and manifest.get("freeze_id") == freeze_id,
        "primary manifest independent identity differs",
    )
    _semantic_require(
        success.get("schema_version") == 1
        and success.get("stage") == 12
        and success.get("status") == policy["outputs"]["primary_status"]
        and success.get("content_fingerprint") == content_fingerprint
        and success.get("freeze_id") == freeze_id,
        "primary success independent identity differs",
    )
    _semantic_require(
        pointer.get("freeze_id") == freeze_id
        and pointer.get("content_fingerprint") == content_fingerprint
        and pointer.get("manifest_sha256")
        == receipts["pre_final_manifest.json"]["sha256"]
        and pointer.get("success_sha256") == receipts["_SUCCESS.json"]["sha256"],
        "primary pointer independent semantic binding differs",
    )

    file_inventory = documents["file_inventory.json"]
    files = file_inventory.get("files")
    _semantic_require(isinstance(files, list), "file inventory members are missing")
    assert isinstance(files, list)
    _semantic_require(
        file_inventory.get("file_count") == len(files)
        and file_inventory.get("tree_sha256")
        == sha256_bytes(canonical_json_bytes(files))
        and bindings.get("project_tree_sha256") == file_inventory.get("tree_sha256"),
        "file inventory independent digest differs",
    )
    _semantic_require(
        bindings.get("policy_sha256") == policy_sha256
        and bindings.get("environment_sha256")
        == fingerprint_basis["environment_sha256"],
        "environment independent digest differs",
    )
    replacement_delta = documents["replacement_inventory_delta.json"]
    changed_members = replacement_delta.get("changed_members")
    added_members = replacement_delta.get("added_members")
    removed_members = replacement_delta.get("removed_members")
    delta_checks = replacement_delta.get("semantic_checks")
    prior_delta_inventory = replacement_delta.get("prior_inventory")
    replacement_delta_inventory = replacement_delta.get("replacement_inventory")
    expected_changed_paths = {
        "src/portfolio_bot/final_freeze/inventory.py",
        "src/portfolio_bot/final_freeze/publication.py",
        "src/portfolio_bot/final_freeze/verification.py",
        "src/portfolio_bot/final_test/economic_kernel.py",
        "src/portfolio_bot/final_test/publication.py",
        "src/portfolio_bot/final_test/production_runtime.py",
        "tests/unit/test_stage12_final_freeze.py",
        "tests/unit/test_final_test_economic_kernel.py",
        "tests/unit/test_final_test_publication.py",
    }
    expected_added_paths = {
        ("configs/phase_06_final_test/" "stage_12_freeze_policy_v11.json"),
        (
            "configs/phase_06_final_test/"
            "stage_12_post_unblind_equal_weight_missing_cost_gate_amendment_v1.json"
        ),
    }
    _semantic_require(
        replacement_delta.get("status")
        == "EXACT_POST_UNBLIND_REPLACEMENT_INVENTORY_DELTA_AUTHENTICATED"
        and isinstance(changed_members, list)
        and replacement_delta.get("changed_member_count") == len(changed_members) == 9
        and {
            item.get("relative_path")
            for item in changed_members
            if isinstance(item, Mapping)
        }
        == expected_changed_paths
        and isinstance(added_members, list)
        and replacement_delta.get("added_member_count") == len(added_members) == 2
        and {
            item.get("relative_path")
            for item in added_members
            if isinstance(item, Mapping)
        }
        == expected_added_paths
        and removed_members == []
        and replacement_delta.get("removed_member_count") == 0
        and isinstance(delta_checks, Mapping)
        and all(value is True for value in delta_checks.values())
        and isinstance(prior_delta_inventory, Mapping)
        and prior_delta_inventory.get("sha256")
        == "ad81241b4b6449c9133b5799578c91abd9cd786ae7175345b1cd52d7209dd3ba"
        and prior_delta_inventory.get("file_count") == 610
        and prior_delta_inventory.get("tree_sha256")
        == "77a08b63b34e93a9750e0e820b6329b3817cbf50f5970999d0087a593cd6c032"
        and isinstance(replacement_delta_inventory, Mapping)
        and replacement_delta_inventory.get("file_count")
        == file_inventory.get("file_count")
        == 612
        and replacement_delta_inventory.get("tree_sha256")
        == file_inventory.get("tree_sha256")
        and bindings.get("replacement_inventory_delta_sha256")
        == receipts["replacement_inventory_delta.json"]["sha256"],
        "replacement inventory delta independent reconciliation differs",
    )

    authority = documents["authority_inventory.json"]
    authority_records = authority.get("records")
    _semantic_require(
        isinstance(authority_records, list), "authority records are missing"
    )
    assert isinstance(authority_records, list)
    authority_sha = sha256_bytes(canonical_json_bytes(authority_records))
    calibration_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "pre_unblind_calibration_scope_amendment"
    ]
    post_unblind_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "post_unblind_technical_amendment"
    ]
    scratch_topology_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "post_unblind_scratch_topology_amendment"
    ]
    scratch_topology_blocker_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_parent_scratch_topology_blocker"
    ]
    hash_audit_incident_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "post_unblind_hash_audit_incident"
    ]
    migration_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_source_checkpoint_context_migration"
    ]
    zero_row_amendment_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "post_unblind_zero_row_merge_provenance_amendment"
    ]
    zero_row_blocker_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_zero_row_copy_guard_blocker"
    ]
    december_amendment_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role")
        == "post_unblind_december_market_input_scope_partition_amendment"
    ]
    december_blocker_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_december_market_input_blocker"
    ]
    decision_rebind_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_decision_checkpoint_rebind"
    ]
    metadata_exception_authorizations = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "checkpoint_rebind_metadata_correction_authorization"
    ]
    metadata_exception_blockers = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_checkpoint_rebind_metadata_blocker"
    ]
    metadata_exception_preflights = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_checkpoint_rebind_metadata_preflight"
    ]
    entry_market_amendment_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role")
        == "post_unblind_entry_market_canonicalization_scope_alignment_amendment"
    ]
    entry_market_blocker_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_entry_market_projection_contract_blocker"
    ]
    entry_market_preflight_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_entry_market_share_coverage_preflight"
    ]
    materiality_policy_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "thesis_error_materiality_policy"
    ]
    accounting_amendment_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role")
        == "post_unblind_accounting_missing_execution_cost_transport_amendment"
    ]
    accounting_authorization_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role")
        == "accounting_missing_execution_cost_transport_authorization"
    ]
    accounting_blocker_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_adv20_nullability_blocker"
    ]
    accounting_preflight_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_accounting_missing_execution_cost_preflight"
    ]
    logical_fingerprint_amendment_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "post_unblind_logical_fingerprint_transport_amendment"
    ]
    logical_fingerprint_authorization_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "logical_fingerprint_transport_authorization"
    ]
    logical_fingerprint_blocker_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_final_market_input_logical_fingerprint_blocker"
    ]
    logical_fingerprint_preflight_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role")
        == "stage13_final_market_input_logical_fingerprint_preflight"
    ]
    factor_covariance_amendment_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role")
        == "post_unblind_factor_covariance_ordinal_basis_alignment_amendment"
    ]
    factor_covariance_authorization_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role")
        == "factor_covariance_ordinal_basis_alignment_authorization"
    ]
    factor_covariance_blocker_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_factor_covariance_ordinal_basis_blocker"
    ]
    factor_covariance_preflight_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_factor_covariance_ordinal_basis_preflight"
    ]
    equal_weight_failure_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping) and item.get("role") == "stage13_resume3_failure"
    ]
    equal_weight_preflight_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "stage13_equal_weight_missing_cost_preflight"
    ]
    equal_weight_authorization_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "equal_weight_missing_cost_gate_authorization"
    ]
    equal_weight_amendment_authorities = [
        item
        for item in authority_records
        if isinstance(item, Mapping)
        and item.get("role") == "post_unblind_equal_weight_missing_cost_gate_amendment"
    ]
    _semantic_require(
        authority.get("authority_count") == len(authority_records) == 49
        and authority.get("authority_inventory_sha256") == authority_sha
        and bindings.get("authority_inventory_sha256") == authority_sha
        and len(calibration_authorities) == 1
        and calibration_authorities[0].get("status")
        == "STAGE12_PREFINAL_METHODOLOGY_AMENDMENT_FROZEN"
        and bindings.get("calibration_scope_amendment_sha256")
        == calibration_authorities[0].get("sha256")
        and len(post_unblind_authorities) == 1
        and post_unblind_authorities[0].get("status")
        == "STAGE12_POST_UNBLIND_TECHNICAL_AMENDMENT_AUTHORIZED"
        and bindings.get("post_unblind_technical_amendment_sha256")
        == post_unblind_authorities[0].get("sha256")
        and len(scratch_topology_authorities) == 1
        and scratch_topology_authorities[0].get("status")
        == "STAGE12_POST_UNBLIND_SCRATCH_TOPOLOGY_AMENDMENT_AUTHORIZED"
        and scratch_topology_authorities[0].get("sha256")
        == "8e394a97d61e7f18b16baef65a29402960a73f83871151047012a3ee48ea97fb"
        and bindings.get("post_unblind_scratch_topology_amendment_sha256")
        == scratch_topology_authorities[0].get("sha256")
        and len(scratch_topology_blocker_authorities) == 1
        and scratch_topology_blocker_authorities[0].get("status")
        == (
            "STAGE13_FAILED_CLOSED_ON_PARENT_SCRATCH_TOPOLOGY_"
            "CONTRACT_PREMISE_MISMATCH"
        )
        and scratch_topology_blocker_authorities[0].get("sha256")
        == "927fd04990a93ac01faf67bee9e600cf345d830e323c708420fd0d1ab1c6069e"
        and bindings.get("stage13_parent_scratch_topology_blocker_sha256")
        == scratch_topology_blocker_authorities[0].get("sha256")
        and len(hash_audit_incident_authorities) == 1
        and hash_audit_incident_authorities[0].get("status")
        == "STAGE12_POST_UNBLIND_HASH_AUDIT_INCIDENT_DISCLOSED"
        and bindings.get("post_unblind_hash_audit_incident_sha256")
        == hash_audit_incident_authorities[0].get("sha256")
        and len(migration_authorities) == 1
        and migration_authorities[0].get("status")
        == "STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_V2_AUTHORIZED"
        and migration_authorities[0].get("sha256")
        == "d504e2dc75c58645b458d255db4f140bd0c9d8a9fa20022b021b4098f586613e"
        and bindings.get("stage13_source_checkpoint_context_migration_sha256")
        == migration_authorities[0].get("sha256")
        and len(zero_row_amendment_authorities) == 1
        and zero_row_amendment_authorities[0].get("status")
        == "STAGE12_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_AUTHORIZED"
        and zero_row_amendment_authorities[0].get("sha256")
        == "ed224a251c99c6ec5c47a5856f872e052e903ac8959f50204636efe9e1abc7aa"
        and bindings.get("post_unblind_zero_row_merge_provenance_amendment_sha256")
        == zero_row_amendment_authorities[0].get("sha256")
        and len(zero_row_blocker_authorities) == 1
        and zero_row_blocker_authorities[0].get("status")
        == "BLOCKED_PENDING_NARROW_TECHNICAL_AMENDMENT_AUTHORIZATION"
        and zero_row_blocker_authorities[0].get("sha256")
        == "7fc5fecfefd4c291bc6196e16d7fd2f14f0763cf0cefcc1475ef869c44465bd7"
        and bindings.get("stage13_zero_row_copy_guard_blocker_sha256")
        == zero_row_blocker_authorities[0].get("sha256")
        and len(december_amendment_authorities) == 1
        and december_amendment_authorities[0].get("status")
        == "STAGE12_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT_AUTHORIZED"
        and december_amendment_authorities[0].get("sha256")
        == "302d4994195f77d2fe554e0411915a676e41a1ba1f2800634f805e344a9827a7"
        and bindings.get(
            "post_unblind_december_market_input_scope_partition_amendment_sha256"
        )
        == december_amendment_authorities[0].get("sha256")
        and len(december_blocker_authorities) == 1
        and december_blocker_authorities[0].get("status")
        == "FAILED_CLOSED_BEFORE_DECISION_BARRIER_MISSING_FINAL_MARKET_INPUT_2025_12"
        and december_blocker_authorities[0].get("sha256")
        == "82a08d940c3b7f6f54bdbd4bc3e6690754f8298c70e4ce57ad366ea80528329b"
        and bindings.get("stage13_december_market_input_blocker_sha256")
        == december_blocker_authorities[0].get("sha256")
        and len(decision_rebind_authorities) == 1
        and decision_rebind_authorities[0].get("status")
        == "STAGE13_DECISION_CHECKPOINT_V4_TO_V5_REBIND_AUTHORIZED"
        and decision_rebind_authorities[0].get("sha256")
        == "1453e467c622913660a9097fc260c5632bbd963474f5b99c167478212ee70908"
        and bindings.get("stage13_decision_checkpoint_rebind_sha256")
        == decision_rebind_authorities[0].get("sha256")
        and bindings.get("stage13_decision_checkpoint_rebind_v1_sha256")
        == "178db1903c087cb53b5f6deb34b6df85da92ae66e55af2124ecf63a69f9fec27"
        and len(metadata_exception_authorizations) == 1
        and metadata_exception_authorizations[0].get("status")
        == "STAGE12_CHECKPOINT_REBIND_METADATA_CORRECTION_AMENDMENT_AUTHORIZED"
        and bindings.get("checkpoint_rebind_metadata_correction_authorization_sha256")
        == metadata_exception_authorizations[0].get("sha256")
        and len(metadata_exception_blockers) == 1
        and metadata_exception_blockers[0].get("status")
        == "STAGE13_FAILED_CLOSED_ON_CHECKPOINT_REBIND_METADATA_AUTHORITY_MISMATCH"
        and bindings.get("stage13_checkpoint_rebind_metadata_blocker_sha256")
        == metadata_exception_blockers[0].get("sha256")
        and len(metadata_exception_preflights) == 1
        and metadata_exception_preflights[0].get("status")
        == "STAGE13_CHECKPOINT_REBIND_METADATA_PREFLIGHT_COMPLETE"
        and bindings.get("stage13_checkpoint_rebind_metadata_preflight_sha256")
        == metadata_exception_preflights[0].get("sha256")
        and len(entry_market_amendment_authorities) == 1
        and entry_market_amendment_authorities[0].get("status")
        == "STAGE12_ENTRY_MARKET_CANONICALIZATION_SCOPE_ALIGNMENT_AMENDMENT_AUTHORIZED"
        and entry_market_amendment_authorities[0].get("sha256")
        == "fb01f060431c4b568053f1588b5c4d21e35371751762c87f76cc566dfa007626"
        and bindings.get(
            "post_unblind_entry_market_canonicalization_scope_alignment_amendment_sha256"
        )
        == entry_market_amendment_authorities[0].get("sha256")
        and len(entry_market_blocker_authorities) == 1
        and entry_market_blocker_authorities[0].get("status")
        == (
            "BLOCKED_PENDING_NARROW_ENTRY_MARKET_CANONICALIZATION_AND_"
            "TOMBSTONE_COVERAGE_AMENDMENT_AUTHORIZATION"
        )
        and entry_market_blocker_authorities[0].get("sha256")
        == "80eec8044486f63b92b12f63de3a39d3ba2840561b0f318874963e1e2ec121ac"
        and bindings.get("stage13_entry_market_projection_contract_blocker_sha256")
        == entry_market_blocker_authorities[0].get("sha256")
        and len(entry_market_preflight_authorities) == 1
        and entry_market_preflight_authorities[0].get("status")
        == "ENTRY_MARKET_SHARE_COVERAGE_PREFLIGHT_COMPLETE_NO_THIRD_PROJECTION_BLOCKER"
        and entry_market_preflight_authorities[0].get("sha256")
        == "f53e09331597f8197eb38648ed280032ee89601b821e050cab930f0c3e3205a0"
        and bindings.get("stage13_entry_market_share_coverage_preflight_sha256")
        == entry_market_preflight_authorities[0].get("sha256")
        and len(materiality_policy_authorities) == 1
        and materiality_policy_authorities[0].get("status")
        == "THESIS_ERROR_MATERIALITY_POLICY_ACTIVE"
        and materiality_policy_authorities[0].get("sha256")
        == "622d1335c540dfa84a4a396bade529e2e3632d1090a41ce66e1990dc582df03d"
        and bindings.get("thesis_error_materiality_policy_sha256")
        == materiality_policy_authorities[0].get("sha256")
        and len(accounting_amendment_authorities) == 1
        and accounting_amendment_authorities[0].get("status")
        == "STAGE12_ACCOUNTING_MISSING_EXECUTION_COST_TRANSPORT_AMENDMENT_AUTHORIZED"
        and accounting_amendment_authorities[0].get("sha256")
        == "2539bdc9b402be3df192690571e01b50ba83f90a381a9860066c62ee874d3b02"
        and bindings.get(
            "post_unblind_accounting_missing_execution_cost_transport_amendment_sha256"
        )
        == accounting_amendment_authorities[0].get("sha256")
        and len(accounting_authorization_authorities) == 1
        and accounting_authorization_authorities[0].get("status")
        == "STAGE12_ACCOUNTING_MISSING_EXECUTION_COST_TRANSPORT_AMENDMENT_AUTHORIZED"
        and accounting_authorization_authorities[0].get("sha256")
        == "593c0da4c8a6997a06626b97fc51c3111ed470863394731bf4dac560c89bd80c"
        and bindings.get(
            "accounting_missing_execution_cost_transport_authorization_sha256"
        )
        == accounting_authorization_authorities[0].get("sha256")
        and len(accounting_blocker_authorities) == 1
        and accounting_blocker_authorities[0].get("status")
        == (
            "STAGE13_FAILED_CLOSED_BEFORE_FIRST_OUTCOME_MONTH_ON_ADV20_"
            "NULLABILITY_CONTRADICTION"
        )
        and accounting_blocker_authorities[0].get("sha256")
        == "9468526f52dae7af4b60589a1efda362c562f2e0a4e739c0fb2a22997f6dc5e3"
        and bindings.get("stage13_adv20_nullability_blocker_sha256")
        == accounting_blocker_authorities[0].get("sha256")
        and len(accounting_preflight_authorities) == 1
        and accounting_preflight_authorities[0].get("status")
        == "STAGE13_ACCOUNTING_MISSING_EXECUTION_COST_PREFLIGHT_COMPLETE"
        and accounting_preflight_authorities[0].get("sha256")
        == "feb761454b5ca2ff2cd3b58faad4a902b67e3af57afcc25eed78eef081022deb"
        and bindings.get("stage13_accounting_missing_execution_cost_preflight_sha256")
        == accounting_preflight_authorities[0].get("sha256")
        and len(logical_fingerprint_amendment_authorities) == 1
        and logical_fingerprint_amendment_authorities[0].get("status")
        == "STAGE12_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_AUTHORIZED"
        and logical_fingerprint_amendment_authorities[0].get("sha256")
        == "b1fe0a1c9064bd26a025b21ac0a6d88d50a2050677a5779b25e822b68ab6a6c7"
        and bindings.get("post_unblind_logical_fingerprint_transport_amendment_sha256")
        == logical_fingerprint_amendment_authorities[0].get("sha256")
        and bindings.get(
            "post_unblind_logical_fingerprint_transport_amendment_semantic_sha256"
        )
        == "16a8ac6837d434849ca127fad4af9f1272d6483e6cf58aa2bfb8f8b26a4c5e6a"
        and len(logical_fingerprint_authorization_authorities) == 1
        and logical_fingerprint_authorization_authorities[0].get("status")
        == "STAGE12_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_AUTHORIZED"
        and logical_fingerprint_authorization_authorities[0].get("sha256")
        == "cbfa2ee2afd71294d71ea201c83c79a845ae4f87ace205a2f19e5c7d9dbac5e5"
        and bindings.get("logical_fingerprint_transport_authorization_sha256")
        == logical_fingerprint_authorization_authorities[0].get("sha256")
        and len(logical_fingerprint_blocker_authorities) == 1
        and logical_fingerprint_blocker_authorities[0].get("status")
        == (
            "STAGE13_FAILED_CLOSED_BEFORE_DECISION_BARRIER_ON_PHYSICAL_"
            "FINGERPRINT_FALSE_DIFFERENCE"
        )
        and logical_fingerprint_blocker_authorities[0].get("sha256")
        == "5b14591d9c559cb01f27262ceb7e7eaa12af2702e0575ccbf57b39a227c5d197"
        and bindings.get(
            "stage13_final_market_input_logical_fingerprint_blocker_sha256"
        )
        == logical_fingerprint_blocker_authorities[0].get("sha256")
        and len(logical_fingerprint_preflight_authorities) == 1
        and logical_fingerprint_preflight_authorities[0].get("status")
        == "STAGE13_FINAL_MARKET_INPUT_LOGICAL_FINGERPRINT_PREFLIGHT_COMPLETE"
        and logical_fingerprint_preflight_authorities[0].get("sha256")
        == "8daa9c7e078072bc9fcc73249086eeaf9c608192dd33331942b6896ea3013cf4"
        and bindings.get(
            "stage13_final_market_input_logical_fingerprint_preflight_sha256"
        )
        == logical_fingerprint_preflight_authorities[0].get("sha256")
        and len(factor_covariance_amendment_authorities) == 1
        and factor_covariance_amendment_authorities[0].get("status")
        == "STAGE12_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT_AUTHORIZED"
        and factor_covariance_amendment_authorities[0].get("sha256")
        == "9d1f8e93ef49b1621df442fd1760fe24fd6f405bcf06e7f1bd6d67b574f01f76"
        and bindings.get(
            "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_sha256"
        )
        == factor_covariance_amendment_authorities[0].get("sha256")
        and bindings.get(
            "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_semantic_sha256"
        )
        == "b1365b6bc7206ceedf62c70ba28986207fcb5bfaa3ff05b71458d130316e168c"
        and len(factor_covariance_authorization_authorities) == 1
        and factor_covariance_authorization_authorities[0].get("status")
        == "STAGE12_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT_AUTHORIZED"
        and factor_covariance_authorization_authorities[0].get("sha256")
        == "9c762291ce317c92837380ab02ccdf820d82c254f0c10252f5f3f715b40eb001"
        and bindings.get(
            "factor_covariance_ordinal_basis_alignment_authorization_sha256"
        )
        == factor_covariance_authorization_authorities[0].get("sha256")
        and len(factor_covariance_blocker_authorities) == 1
        and factor_covariance_blocker_authorities[0].get("status")
        == (
            "STAGE13_FAILED_CLOSED_BEFORE_NORMALIZED_SOURCE_PUBLICATION_ON_"
            "FACTOR_COVARIANCE_ORDINAL_BASIS_MISMATCH"
        )
        and factor_covariance_blocker_authorities[0].get("sha256")
        == "3d8e1a80741aadae7773368d17bc04541e01b4aee21a48e8ebe2f3b278630a18"
        and bindings.get("stage13_factor_covariance_ordinal_basis_blocker_sha256")
        == factor_covariance_blocker_authorities[0].get("sha256")
        and len(factor_covariance_preflight_authorities) == 1
        and factor_covariance_preflight_authorities[0].get("status")
        == "STAGE13_FACTOR_COVARIANCE_ORDINAL_BASIS_PREFLIGHT_COMPLETE"
        and factor_covariance_preflight_authorities[0].get("sha256")
        == "49e043391a4ea27f8392a55685cfd72151d4ebd9c1119f7b2f05b698c6926530"
        and bindings.get("stage13_factor_covariance_ordinal_basis_preflight_sha256")
        == factor_covariance_preflight_authorities[0].get("sha256")
        and len(equal_weight_failure_authorities) == 1
        and equal_weight_failure_authorities[0].get("status")
        == "STAGE13_RESUME3_FAILED_CLOSED_BEFORE_FIRST_ECONOMIC_CHECKPOINT"
        and bindings.get("stage13_resume3_failure_sha256")
        == equal_weight_failure_authorities[0].get("sha256")
        and len(equal_weight_preflight_authorities) == 1
        and equal_weight_preflight_authorities[0].get("status")
        == "STAGE13_EQUAL_WEIGHT_MISSING_COST_GATE_PREFLIGHT_COMPLETE"
        and bindings.get("stage13_equal_weight_missing_cost_preflight_sha256")
        == equal_weight_preflight_authorities[0].get("sha256")
        and len(equal_weight_authorization_authorities) == 1
        and equal_weight_authorization_authorities[0].get("status")
        == "STAGE12_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_AUTHORIZED"
        and bindings.get("equal_weight_missing_cost_gate_authorization_sha256")
        == equal_weight_authorization_authorities[0].get("sha256")
        and len(equal_weight_amendment_authorities) == 1
        and equal_weight_amendment_authorities[0].get("status")
        == "STAGE12_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_AUTHORIZED"
        and bindings.get("post_unblind_equal_weight_missing_cost_gate_amendment_sha256")
        == equal_weight_amendment_authorities[0].get("sha256")
        and bindings.get(
            "post_unblind_equal_weight_missing_cost_gate_amendment_semantic_sha256"
        )
        == "4968272bb6117cb3c808e1f0679b1053aaa3b3a3459bf7c9e8105f7b03d00b7f"
        and bindings.get("stage13_source_checkpoint_context_migration_v1_sha256")
        == "956fe984d350cff58142affb2f3832f0bd7b061749f61f998b2d817c59012b13"
        and bindings.get("prior_scratch_topology_stage12_primary_pointer_sha256")
        == "04d1105329b09ea078f1abb4c7842ce6441b7c9f70d7869a11bf4954ced274fe"
        and bindings.get("prior_scratch_topology_stage12_verified_pointer_sha256")
        == "850ba5805753406f9e7002e1cb71f8dfdea51c5d22606c80851baef8929384a6"
        and bindings.get("prior_scratch_topology_stage13_approval_sha256")
        == "3eece897f35e123659bc2fbf5721b42122d8be0394363d2fe31137c8d53548f2"
        and bindings.get("stage13_parent_feature_sources_tree_sha256")
        == "06483456c92d0302c124aba9f577fb17cc0d1330b6ab3dc5e08f7093ad89bac7"
        and bindings.get("base_stage12_policy_sha256")
        == "68b47869472a51453856822283bc68e065640fb0c447d1a87f409ad9c422bf31"
        and bindings.get("prior_replacement_policy_sha256")
        == "fb2d0f27020c1ec842fddbbb73bc6b1836088f5fef417a40974509d74827138f"
        and bindings.get("prior_equal_weight_stage12_file_inventory_sha256")
        == "ad81241b4b6449c9133b5799578c91abd9cd786ae7175345b1cd52d7209dd3ba"
        and bindings.get("prior_equal_weight_stage12_primary_pointer_sha256")
        == "115e2f65e16b1b30bcdaab13bfe9c9c55e0af8d6d98a0b7a3564bab7d8012011"
        and bindings.get("prior_equal_weight_stage12_verified_pointer_sha256")
        == "f71ef82e9db06df23acf33562dd40f40d426930478706f7c36885d2de7bc4c2b"
        and bindings.get("prior_equal_weight_stage13_approval_sha256")
        == "23423442f51da1df76bee27e1c93054b710091a38e40bd120b415d6f4a98dfd6"
        and bindings.get("equal_weight_kernel_pre_amendment_sha256")
        == "e8698fce50883217f22c5a6404eacaa101eb750fb7092589a47e4cef8ba26ddf"
        and bindings.get("equal_weight_kernel_post_amendment_sha256")
        == "c0b57ad65e4c11b3836f195129f250d9e4c44be49e10d1bbf15b693f703937e0"
        and bindings.get("equal_weight_publication_pre_amendment_sha256")
        == "7b20de47a4d04e67e42f9140d7c59e522f36470994f88c3bd5340f6a3f880755"
        and bindings.get("equal_weight_publication_post_amendment_sha256")
        == "14982491210ef1412d0a5c04b051a829e5f547ca0a6f91cfe74ed64a4ce715d7"
        and bindings.get("equal_weight_runtime_pre_amendment_sha256")
        == "f7ed29de71bd2b3731bc9e3bc6385414ce264bebb499f9faa3a4ddef3657e929"
        and bindings.get("equal_weight_runtime_post_amendment_sha256")
        == "5390d06e5220579dc2045ab67031521a3c94e7603ac3e945eaeb80d78375d75a"
        and bindings.get("failed_stage13_v9_launch_memo_sha256")
        == "9c0c3d1b675e86b8dad97bfcb5361c1f7fae212e73710f8afe84448ac9d3db3e"
        and bindings.get("prior_factor_covariance_stage12_file_inventory_sha256")
        == "ef709f1920723fe8deba82c55622f231fdfa9f9cd7780047abebadb2a6c151b5"
        and bindings.get("prior_factor_covariance_stage12_primary_pointer_sha256")
        == "a9799dbf0e033c9ffb0acf771037bb31fd7bd5b50cb4434272d9cd9857755117"
        and bindings.get("prior_factor_covariance_stage12_verified_pointer_sha256")
        == "69bc5ad1514fefd5a21a1ab9b7318a26ecd89862f3f07e86f8d8ad166bea4ec0"
        and bindings.get("prior_factor_covariance_stage13_approval_sha256")
        == "ac4432190a5b0a33b8c892f14e13cde0f06bb97b5d4b49878b2e613061202408"
        and bindings.get("factor_covariance_compiler_pre_amendment_sha256")
        == "70dc0492e0538051a027c4f9b75bacebd307554cd16a383ba7143b00259ae082"
        and bindings.get("factor_covariance_compiler_post_amendment_sha256")
        == "21cc65d99c872c7fb6988da63412790d091c666b2caadd9b0859d21f718e1a8f"
        and bindings.get("factor_covariance_runtime_pre_amendment_sha256")
        == "f36f07b00808eb4abcca5692c2b5732b67feaac15bb9369f36892cfdc4ef319b"
        and bindings.get("factor_covariance_runtime_post_amendment_sha256")
        == "f7ed29de71bd2b3731bc9e3bc6385414ce264bebb499f9faa3a4ddef3657e929"
        and bindings.get("prior_logical_fingerprint_stage12_primary_pointer_sha256")
        == "1a2314c978d92222e80d9a625fd2fecb692243dc7a964f5ead530c22bb6e4048"
        and bindings.get("prior_logical_fingerprint_stage12_verified_pointer_sha256")
        == "979d51cff41b05e3e6e32880ac91ba54f35c196eade63f2fce2483da70c7eb9c"
        and bindings.get("prior_logical_fingerprint_stage13_approval_sha256")
        == "5bafb96b36c7fc0e02ba44e4c607270a24c5d836c520a037b021506084dd1d4a"
        and bindings.get("logical_fingerprint_replay_pre_amendment_sha256")
        == "f88c9aa6532659c3d4b0cbddaa912a29d2d3d4594b01a5b184fa5edcfc99c664"
        and bindings.get("logical_fingerprint_replay_post_amendment_sha256")
        == "25cea785f5eb3c390041221ee07d0333ca9f0374ebab5d4702ebd8e2302c1cfa"
        and bindings.get("accounting_transport_replay_pre_amendment_sha256")
        == "48f84559528749d8090858ad21de59f8a3e31a92aa6250a62ebfcb12a90b7253"
        and bindings.get("accounting_transport_replay_post_amendment_sha256")
        == "f88c9aa6532659c3d4b0cbddaa912a29d2d3d4594b01a5b184fa5edcfc99c664"
        and bindings.get("accounting_transport_materializer_pre_amendment_sha256")
        == "1e2f8d7ce33782daccb5d7697f64c52e081279dffb9610bc93f37485e5e47e18"
        and bindings.get("accounting_transport_materializer_post_amendment_sha256")
        == "23fcb0e5a0752ae376e6bc9584773051fe3e53d1992f430094b5aa1ab62dc925"
        and bindings.get(
            "prior_checkpoint_rebind_metadata_stage12_primary_pointer_sha256"
        )
        == "81eb2bc85d960e9a005f6b62b64a8d8444df75068685ad90cd8c58f49c7d5bdf"
        and bindings.get("prior_checkpoint_rebind_metadata_stage12_manifest_sha256")
        == "5599c0efdd7d1a334071314ba8e68022a54701039419350c9b19d6b150984a97"
        and bindings.get(
            "prior_checkpoint_rebind_metadata_stage12_verified_pointer_sha256"
        )
        == "3a19790a88a64542f9b42429487e4b892d905e88fc1142ce100974f344720fca"
        and bindings.get("prior_checkpoint_rebind_metadata_verification_report_sha256")
        == "c60922692c2fd12bcea80975fbe1ee72496b5a82c37608f697002e8cba0853e3"
        and bindings.get("prior_checkpoint_rebind_metadata_stage13_approval_sha256")
        == "4286d671963429bfe5c113663ca37decdaffa1f34b95cad54b5f481874dbcd49"
        and bindings.get(
            "prior_checkpoint_rebind_metadata_stage13_approval_payload_sha256"
        )
        == "fc92568e651f79a74b03f7fae4ed79919561b19521ac913d1058cbd4f84daab5"
        and bindings.get("prior_checkpoint_rebind_metadata_stage12_binding_sha256")
        == "0a95ac10ff1103a788bb4f60f5a4321aa483dceec0c8bcebb8670da7adcc668c"
        and bindings.get("checkpoint_rebind_final_source_replay_sha256")
        == "48f84559528749d8090858ad21de59f8a3e31a92aa6250a62ebfcb12a90b7253"
        and bindings.get("prior_december_repair_file_inventory_sha256")
        == "87200b90aba352f761fddb61036f2888814067544d175b9855f64a6c727a6928"
        and bindings.get("pre_amendment_adapter_sha256")
        == "f02c0429a126a6a576197d2175030bc8a0018969ba01732b254b4a8318cfff31"
        and bindings.get("post_amendment_adapter_sha256")
        == "e2b3a85d32c87f0de42b67f75972a6a6203073fb603769cd7e11cbcfb69b7ec0"
        and authority.get("raw_payload_files_opened") == 0
        and authority.get("final_period_payload_rows_read") == 0,
        "authority inventory independent reconciliation differs",
    )

    final_system_document = documents["final_system.json"]
    final_system = final_system_document.get("system")
    _semantic_require(
        isinstance(final_system, Mapping),
        "final-system semantic document is missing",
    )
    assert isinstance(final_system, Mapping)
    learning = final_system.get("features_and_learning")
    readiness = final_system.get("implementation_readiness")
    _semantic_require(
        final_system.get("methodology_version") == "v1"
        and final_system.get("pre_unblind_calibration_scope_amendment")
        == "stage12-prefinal-shared-calibration-scope-v1"
        and isinstance(learning, Mapping)
        and learning.get("calibration_scope_policy_id")
        == "full-feature-prefinal-and-prior-live-shared-by-strategy-v1"
        and learning.get("calibration_source_scope") == "full_feature_primary"
        and learning.get("same_strategy_calibration_slope_shared_across_feature_scopes")
        is True
        and learning.get("reduced_core_scores_relabelled_as_prefinal_oof") is False
        and isinstance(readiness, Mapping)
        and readiness.get("pre_unblind_calibration_scope_amendment_frozen") is True,
        "final-system calibration-scope amendment semantics differ",
    )

    ledger = documents["trial_ledger.json"]
    rows = ledger.get("rows")
    _semantic_require(isinstance(rows, list), "trial ledger rows are missing")
    assert isinstance(rows, list)
    raw_count = sum(
        1
        for row in rows
        if isinstance(row, Mapping) and row.get("counts_in_raw_dsr") is True
    )
    ledger_sha = sha256_bytes(canonical_json_bytes(rows))
    _semantic_require(
        ledger.get("ledger_id") == "stage12-complete-frozen-trial-ledger-v1"
        and ledger.get("total_ledger_rows") == len(rows)
        and ledger.get("raw_trial_count") == raw_count == 386
        and ledger.get("excluded_row_count") == len(rows) - raw_count
        and ledger.get("ledger_sha256") == ledger_sha
        and bindings.get("trial_ledger_sha256") == ledger_sha,
        "trial ledger independent reconciliation differs",
    )

    source_drift = documents["source_drift.json"]
    _semantic_require(
        source_drift.get("status")
        == (
            "AUTHENTICATED_RULE_RESTORING_EQUAL_WEIGHT_MISSING_COST_GATE_"
            "SOURCE_TRANSITION"
        )
        and source_drift.get("record_id")
        == "stage12-equal-weight-missing-cost-gate-policy-v11"
        and source_drift.get("stage11_recomputed") is False
        and source_drift.get("validation_performance_used") is False
        and source_drift.get("final_period_rows_read") == 0,
        "source-transition semantic boundary differs",
    )
    amendment_receipt = source_drift.get("amendment_receipt")
    post_unblind_source_receipt = source_drift.get(
        "post_unblind_technical_amendment_receipt"
    )
    post_unblind_changes = source_drift.get("authorized_post_unblind_changed_members")
    zero_row_source_receipt = source_drift.get(
        "post_unblind_zero_row_merge_provenance_amendment_receipt"
    )
    zero_row_changes = source_drift.get(
        "authorized_zero_row_merge_provenance_changed_members"
    )
    december_source_receipt = source_drift.get(
        "post_unblind_december_market_input_scope_partition_amendment_receipt"
    )
    december_changes = source_drift.get(
        "authorized_december_market_input_scope_partition_members"
    )
    entry_market_source_amendment = source_drift.get(
        "post_unblind_entry_market_canonicalization_scope_alignment_amendment"
    )
    entry_market_source_receipt = source_drift.get(
        "post_unblind_entry_market_canonicalization_scope_alignment_amendment_receipt"
    )
    entry_market_changes = source_drift.get(
        "authorized_entry_market_canonicalization_scope_alignment_changed_members"
    )
    metadata_exception_changes = source_drift.get(
        "authorized_checkpoint_rebind_metadata_exception_changed_members"
    )
    accounting_source_amendment = source_drift.get(
        "post_unblind_accounting_missing_execution_cost_transport_amendment"
    )
    accounting_source_receipt = source_drift.get(
        "post_unblind_accounting_missing_execution_cost_transport_amendment_receipt"
    )
    accounting_transport_changes = source_drift.get(
        "authorized_accounting_missing_execution_cost_transport_changed_members"
    )
    logical_fingerprint_source_amendment = source_drift.get(
        "post_unblind_logical_fingerprint_transport_amendment"
    )
    logical_fingerprint_source_receipt = source_drift.get(
        "post_unblind_logical_fingerprint_transport_amendment_receipt"
    )
    logical_fingerprint_changes = source_drift.get(
        "authorized_logical_fingerprint_transport_changed_members"
    )
    factor_covariance_source_amendment = source_drift.get(
        "post_unblind_factor_covariance_ordinal_basis_alignment_amendment"
    )
    factor_covariance_source_receipt = source_drift.get(
        "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_receipt"
    )
    factor_covariance_changes = source_drift.get(
        "authorized_factor_covariance_ordinal_basis_alignment_changed_members"
    )
    equal_weight_source_amendment = source_drift.get(
        "post_unblind_equal_weight_missing_cost_gate_amendment"
    )
    equal_weight_source_receipt = source_drift.get(
        "post_unblind_equal_weight_missing_cost_gate_amendment_receipt"
    )
    equal_weight_changes = source_drift.get(
        "authorized_equal_weight_missing_cost_gate_changed_members"
    )
    _semantic_require(
        isinstance(amendment_receipt, Mapping)
        and bindings.get("source_amendment_sha256") == amendment_receipt.get("sha256")
        and isinstance(post_unblind_source_receipt, Mapping)
        and bindings.get("post_unblind_technical_amendment_sha256")
        == post_unblind_source_receipt.get("sha256")
        and isinstance(post_unblind_changes, list)
        and post_unblind_changes
        == [
            {
                "path": (
                    "src/portfolio_bot/validation/"
                    "validation_feature_input_adapter.py"
                ),
                "pre_amendment_sha256": (
                    "f02c0429a126a6a576197d2175030bc8a0018969ba01732b254b4a8318cfff31"
                ),
                "post_amendment_sha256": (
                    "e2b3a85d32c87f0de42b67f75972a6a6203073fb603769cd7e11cbcfb69b7ec0"
                ),
            }
        ]
        and isinstance(zero_row_source_receipt, Mapping)
        and zero_row_source_receipt.get("sha256")
        == bindings.get("post_unblind_zero_row_merge_provenance_amendment_sha256")
        and zero_row_changes
        == [
            {
                "path": (
                    "src/portfolio_bot/validation/"
                    "validation_feature_input_adapter.py"
                ),
                "pre_amendment_sha256": (
                    "e2b3a85d32c87f0de42b67f75972a6a6203073fb603769cd7e11cbcfb69b7ec0"
                ),
                "post_amendment_sha256": (
                    "a2d8ac04a84f957c61fca922b353116aa7c48512d188a2fb3f0547be5f1f6d45"
                ),
            },
            {
                "path": "src/portfolio_bot/final_test/final_source_replay.py",
                "pre_amendment_sha256": (
                    "4cd88a6fbb702207004f66006364efc5cf0a7d66633ebb6361bbb6f2cc2c53de"
                ),
                "post_amendment_sha256": (
                    "ebecc9d341331cb7e6dc0b15cd45bc522ed7dda5796608542b53412c8ccc28c6"
                ),
            },
        ]
        and isinstance(december_source_receipt, Mapping)
        and december_source_receipt.get("sha256")
        == bindings.get(
            "post_unblind_december_market_input_scope_partition_amendment_sha256"
        )
        and december_changes
        == [
            {
                "path": "src/portfolio_bot/final_test/final_source_replay.py",
                "pre_amendment_sha256": (
                    "ebecc9d341331cb7e6dc0b15cd45bc522ed7dda5796608542b53412c8ccc28c6"
                ),
                "post_amendment_sha256": (
                    "73539ebe6b569cfacc94bce78c246766551bfbe174a2f8a8aabc4b126ba96a4f"
                ),
            },
            {
                "path": ("src/portfolio_bot/validation/validation_market_extension.py"),
                "pre_amendment_sha256": (
                    "1bd9a4487351f1ac3337f2d3216f946ba553ee43a6006353ed1a019d7ad09446"
                ),
                "post_amendment_sha256": (
                    "b54e382705935b90c25372c4fc24cce7013da2beda741ae378e20113977e50f6"
                ),
            },
            {
                "path": ("src/portfolio_bot/final_test/source_checkpoint_rebind.py"),
                "pre_amendment_sha256": None,
                "post_amendment_sha256": (
                    "7216eb7b08b4cd0886e9ffff6497ed9a5b49c9013310c63e36a8f4e2d2d688bb"
                ),
            },
        ]
        and isinstance(entry_market_source_amendment, Mapping)
        and _embedded_amendment_semantics_match(
            entry_market_source_amendment,
            expected_semantic_sha256=(
                "519667052e7ed2e6bd3baf13a048301c158e2d7ff4f101793b188fbbe763fa75"
            ),
        )
        and entry_market_source_receipt
        == {
            "relative_path": (
                "configs/phase_06_final_test/"
                "stage_12_post_unblind_entry_market_canonicalization_scope_alignment_"
                "amendment_v1.json"
            ),
            "sha256": (
                "fb01f060431c4b568053f1588b5c4d21e35371751762c87f76cc566dfa007626"
            ),
            "size_bytes": 12404,
        }
        and entry_market_changes
        == [
            {
                "path": "src/portfolio_bot/final_test/final_source_replay.py",
                "pre_amendment_sha256": (
                    "73539ebe6b569cfacc94bce78c246766551bfbe174a2f8a8aabc4b126ba96a4f"
                ),
                "post_amendment_sha256": (
                    "c6e439f3411aa95692e890be8e8364547b92368e19e8a4af9f7b08ba4d218d58"
                ),
            },
            {
                "path": "src/portfolio_bot/validation/validation_market_extension.py",
                "pre_amendment_sha256": (
                    "b54e382705935b90c25372c4fc24cce7013da2beda741ae378e20113977e50f6"
                ),
                "post_amendment_sha256": (
                    "a5118f43874191e7d4ed3a6c0c2c5730dacb3052e72adbc1b97aee77952a8556"
                ),
            },
        ]
        and metadata_exception_changes
        == [
            {
                "path": "src/portfolio_bot/final_test/final_source_replay.py",
                "pre_exception_sha256": (
                    "c6e439f3411aa95692e890be8e8364547b92368e19e8a4af9f7b08ba4d218d58"
                ),
                "post_exception_sha256": (
                    "48f84559528749d8090858ad21de59f8a3e31a92aa6250a62ebfcb12a90b7253"
                ),
                "classification": (
                    "non_economic_checkpoint_rebind_metadata_authority_"
                    "compatibility_repair"
                ),
            }
        ]
        and isinstance(accounting_source_amendment, Mapping)
        and _embedded_amendment_semantics_match(
            accounting_source_amendment,
            expected_semantic_sha256=(
                "96f5999d7780ad907e73c77f82da96615b347d5fc4cf666a6839d9ad02501b76"
            ),
        )
        and accounting_source_receipt
        == {
            "relative_path": (
                "configs/phase_06_final_test/"
                "stage_12_post_unblind_accounting_missing_execution_cost_"
                "transport_amendment_v1.json"
            ),
            "sha256": (
                "2539bdc9b402be3df192690571e01b50ba83f90a381a9860066c62ee874d3b02"
            ),
            "size_bytes": 8660,
        }
        and accounting_transport_changes
        == [
            {
                "path": "src/portfolio_bot/final_test/final_source_replay.py",
                "pre_amendment_sha256": (
                    "48f84559528749d8090858ad21de59f8a3e31a92aa6250a62ebfcb12a90b7253"
                ),
                "post_amendment_sha256": (
                    "f88c9aa6532659c3d4b0cbddaa912a29d2d3d4594b01a5b184fa5edcfc99c664"
                ),
            },
            {
                "path": "src/portfolio_bot/final_test/input_materialization.py",
                "pre_amendment_sha256": (
                    "1e2f8d7ce33782daccb5d7697f64c52e081279dffb9610bc93f37485e5e47e18"
                ),
                "post_amendment_sha256": (
                    "23fcb0e5a0752ae376e6bc9584773051fe3e53d1992f430094b5aa1ab62dc925"
                ),
            },
        ]
        and isinstance(logical_fingerprint_source_amendment, Mapping)
        and _embedded_amendment_semantics_match(
            logical_fingerprint_source_amendment,
            expected_semantic_sha256=(
                "16a8ac6837d434849ca127fad4af9f1272d6483e6cf58aa2bfb8f8b26a4c5e6a"
            ),
        )
        and logical_fingerprint_source_receipt
        == {
            "relative_path": (
                "configs/phase_06_final_test/"
                "stage_12_post_unblind_logical_fingerprint_transport_amendment_v1.json"
            ),
            "sha256": (
                "b1fe0a1c9064bd26a025b21ac0a6d88d50a2050677a5779b25e822b68ab6a6c7"
            ),
            "size_bytes": 8059,
        }
        and logical_fingerprint_changes
        == [
            {
                "path": "src/portfolio_bot/final_test/final_source_replay.py",
                "pre_amendment_sha256": (
                    "f88c9aa6532659c3d4b0cbddaa912a29d2d3d4594b01a5b184fa5edcfc99c664"
                ),
                "post_amendment_sha256": (
                    "25cea785f5eb3c390041221ee07d0333ca9f0374ebab5d4702ebd8e2302c1cfa"
                ),
            }
        ]
        and isinstance(factor_covariance_source_amendment, Mapping)
        and _embedded_amendment_semantics_match(
            factor_covariance_source_amendment,
            expected_semantic_sha256=(
                "b1365b6bc7206ceedf62c70ba28986207fcb5bfaa3ff05b71458d130316e168c"
            ),
        )
        and factor_covariance_source_receipt
        == {
            "relative_path": (
                "configs/phase_06_final_test/"
                "stage_12_post_unblind_factor_covariance_ordinal_basis_alignment_"
                "amendment_v1.json"
            ),
            "sha256": (
                "9d1f8e93ef49b1621df442fd1760fe24fd6f405bcf06e7f1bd6d67b574f01f76"
            ),
            "semantic_sha256": (
                "b1365b6bc7206ceedf62c70ba28986207fcb5bfaa3ff05b71458d130316e168c"
            ),
            "size_bytes": 8089,
        }
        and factor_covariance_changes
        == [
            {
                "path": "src/portfolio_bot/final_test/final_input_construction.py",
                "pre_amendment_sha256": (
                    "70dc0492e0538051a027c4f9b75bacebd307554cd16a383ba7143b00259ae082"
                ),
                "post_amendment_sha256": (
                    "21cc65d99c872c7fb6988da63412790d091c666b2caadd9b0859d21f718e1a8f"
                ),
            },
            {
                "path": "src/portfolio_bot/final_test/production_runtime.py",
                "pre_amendment_sha256": (
                    "f36f07b00808eb4abcca5692c2b5732b67feaac15bb9369f36892cfdc4ef319b"
                ),
                "post_amendment_sha256": (
                    "f7ed29de71bd2b3731bc9e3bc6385414ce264bebb499f9faa3a4ddef3657e929"
                ),
            },
        ]
        and isinstance(equal_weight_source_amendment, Mapping)
        and _embedded_amendment_semantics_match(
            equal_weight_source_amendment,
            expected_semantic_sha256=(
                "4968272bb6117cb3c808e1f0679b1053aaa3b3a3459bf7c9e8105f7b03d00b7f"
            ),
        )
        and equal_weight_source_receipt
        == {
            "relative_path": (
                "configs/phase_06_final_test/"
                "stage_12_post_unblind_equal_weight_missing_cost_gate_"
                "amendment_v1.json"
            ),
            "sha256": (
                "72a94490bf1c835871104c966991ce42ef5a5ee5ee33ab7aabb20570d3de8ffa"
            ),
            "semantic_sha256": (
                "4968272bb6117cb3c808e1f0679b1053aaa3b3a3459bf7c9e8105f7b03d00b7f"
            ),
            "size_bytes": 8943,
        }
        and equal_weight_changes
        == [
            {
                "path": "src/portfolio_bot/final_test/economic_kernel.py",
                "pre_amendment_sha256": (
                    "e8698fce50883217f22c5a6404eacaa101eb750fb7092589a47e4cef8ba26ddf"
                ),
                "post_amendment_sha256": (
                    "c0b57ad65e4c11b3836f195129f250d9e4c44be49e10d1bbf15b693f703937e0"
                ),
            },
            {
                "path": "src/portfolio_bot/final_test/publication.py",
                "pre_amendment_sha256": (
                    "7b20de47a4d04e67e42f9140d7c59e522f36470994f88c3bd5340f6a3f880755"
                ),
                "post_amendment_sha256": (
                    "14982491210ef1412d0a5c04b051a829e5f547ca0a6f91cfe74ed64a4ce715d7"
                ),
            },
            {
                "path": "src/portfolio_bot/final_test/production_runtime.py",
                "pre_amendment_sha256": (
                    "f7ed29de71bd2b3731bc9e3bc6385414ce264bebb499f9faa3a4ddef3657e929"
                ),
                "post_amendment_sha256": (
                    "5390d06e5220579dc2045ab67031521a3c94e7603ac3e945eaeb80d78375d75a"
                ),
            },
        ],
        "source-transition binding differs",
    )

    tests = documents["test_receipts.json"]
    nodes = tests.get("nodes")
    _semantic_require(
        tests.get("status") == "ALL_REQUIRED_TEST_NODES_PASSED"
        and tests.get("node_count") == 2
        and isinstance(nodes, list)
        and len(nodes) == 2
        and {node.get("status") for node in nodes if isinstance(node, Mapping)}
        == {"PASSED"},
        "test receipt semantic graph differs",
    )
    checks = documents["freeze_checks.json"]
    check_values = checks.get("checks")
    counters = checks.get("counters")
    authorizations = checks.get("authorizations")
    calibration_receipt = checks.get("calibration_scope_amendment_receipt")
    calibration_access = (
        calibration_receipt.get("timing_and_access_boundary")
        if isinstance(calibration_receipt, Mapping)
        else None
    )
    post_unblind_receipt = checks.get("post_unblind_technical_amendment_receipt")
    post_unblind_boundary = (
        post_unblind_receipt.get("non_economic_boundary")
        if isinstance(post_unblind_receipt, Mapping)
        else None
    )
    scratch_topology_receipt = checks.get(
        "post_unblind_scratch_topology_amendment_receipt"
    )
    scratch_topology_authorization = (
        scratch_topology_receipt.get("authorization")
        if isinstance(scratch_topology_receipt, Mapping)
        else None
    )
    scratch_topology_authorization_receipt = (
        scratch_topology_receipt.get("authorization_receipt")
        if isinstance(scratch_topology_receipt, Mapping)
        else None
    )
    scratch_topology_contract = (
        scratch_topology_receipt.get("topology_contract")
        if isinstance(scratch_topology_receipt, Mapping)
        else None
    )
    scratch_topology_boundary = (
        scratch_topology_receipt.get("non_economic_boundary")
        if isinstance(scratch_topology_receipt, Mapping)
        else None
    )
    scratch_topology_replacement = (
        scratch_topology_receipt.get("replacement_authority")
        if isinstance(scratch_topology_receipt, Mapping)
        else None
    )
    scratch_topology_blocker = (
        scratch_topology_receipt.get("blocker")
        if isinstance(scratch_topology_receipt, Mapping)
        else None
    )
    zero_row_receipt = checks.get(
        "post_unblind_zero_row_merge_provenance_amendment_receipt"
    )
    zero_row_authorization_receipt = (
        zero_row_receipt.get("authorization_receipt")
        if isinstance(zero_row_receipt, Mapping)
        else None
    )
    zero_row_blocker_receipt = (
        zero_row_receipt.get("blocker_receipt")
        if isinstance(zero_row_receipt, Mapping)
        else None
    )
    zero_row_scientific_scope = (
        zero_row_receipt.get("frozen_scientific_scope")
        if isinstance(zero_row_receipt, Mapping)
        else None
    )
    zero_row_provenance_contract = (
        zero_row_receipt.get("provenance_contract")
        if isinstance(zero_row_receipt, Mapping)
        else None
    )
    zero_row_source_transition = (
        zero_row_receipt.get("source_transition")
        if isinstance(zero_row_receipt, Mapping)
        else None
    )
    zero_row_boundary = (
        zero_row_receipt.get("non_economic_boundary")
        if isinstance(zero_row_receipt, Mapping)
        else None
    )
    zero_row_replacement = (
        zero_row_receipt.get("replacement_authority")
        if isinstance(zero_row_receipt, Mapping)
        else None
    )
    december_receipt = checks.get(
        "post_unblind_december_market_input_scope_partition_amendment_receipt"
    )
    december_authorization_receipt = (
        december_receipt.get("authorization_receipt")
        if isinstance(december_receipt, Mapping)
        else None
    )
    december_blocker_receipt = (
        december_receipt.get("blocker_receipt")
        if isinstance(december_receipt, Mapping)
        else None
    )
    december_scope = (
        december_receipt.get("decision_outcome_scope_partition")
        if isinstance(december_receipt, Mapping)
        else None
    )
    december_preflight = (
        december_receipt.get("preflight_and_completion_contract")
        if isinstance(december_receipt, Mapping)
        else None
    )
    december_rebind = (
        december_receipt.get("decision_checkpoint_rebind")
        if isinstance(december_receipt, Mapping)
        else None
    )
    metadata_exception_receipt = checks.get(
        "checkpoint_rebind_metadata_exception_receipt"
    )
    december_rebind_aggregates = (
        december_rebind.get("aggregate_pins")
        if isinstance(december_rebind, Mapping)
        else None
    )
    december_rebind_cache = (
        december_rebind.get("parent_decision_market_cache")
        if isinstance(december_rebind, Mapping)
        else None
    )
    december_rebind_cache_scope = (
        december_rebind_cache.get("scope")
        if isinstance(december_rebind_cache, Mapping)
        else None
    )
    december_rebind_cache_safety = (
        december_rebind_cache.get("safety")
        if isinstance(december_rebind_cache, Mapping)
        else None
    )
    december_boundary = (
        december_receipt.get("non_economic_boundary")
        if isinstance(december_receipt, Mapping)
        else None
    )
    december_replacement = (
        december_receipt.get("replacement_authority")
        if isinstance(december_receipt, Mapping)
        else None
    )
    entry_market_receipt = checks.get(
        "post_unblind_entry_market_canonicalization_scope_alignment_amendment_receipt"
    )
    entry_market_authorization = (
        entry_market_receipt.get("authorization_receipt")
        if isinstance(entry_market_receipt, Mapping)
        else None
    )
    entry_market_blocker_receipt = (
        entry_market_receipt.get("combined_projection_blocker_receipt")
        if isinstance(entry_market_receipt, Mapping)
        else None
    )
    entry_market_preflight_receipt = (
        entry_market_receipt.get("share_coverage_preflight_receipt")
        if isinstance(entry_market_receipt, Mapping)
        else None
    )
    entry_market_prior_receipts = (
        entry_market_receipt.get("prior_authority_receipts")
        if isinstance(entry_market_receipt, Mapping)
        else None
    )
    entry_market_projection = (
        entry_market_receipt.get("entry_market_projection_contract")
        if isinstance(entry_market_receipt, Mapping)
        else None
    )
    entry_market_canonicalization = (
        entry_market_receipt.get("exact_duplicate_canonicalization_contract")
        if isinstance(entry_market_receipt, Mapping)
        else None
    )
    entry_market_share_coverage = (
        entry_market_receipt.get("closed_tombstone_share_coverage_contract")
        if isinstance(entry_market_receipt, Mapping)
        else None
    )
    entry_market_receipt_contract = (
        entry_market_receipt.get("receipt_contract")
        if isinstance(entry_market_receipt, Mapping)
        else None
    )
    entry_market_source_transition = (
        entry_market_receipt.get("source_transition")
        if isinstance(entry_market_receipt, Mapping)
        else None
    )
    entry_market_boundary = (
        entry_market_receipt.get("non_economic_boundary")
        if isinstance(entry_market_receipt, Mapping)
        else None
    )
    entry_market_replacement = (
        entry_market_receipt.get("replacement_authority")
        if isinstance(entry_market_receipt, Mapping)
        else None
    )
    accounting_transport_receipt = checks.get(
        "post_unblind_accounting_missing_execution_cost_transport_amendment_receipt"
    )
    logical_fingerprint_transport_receipt = checks.get(
        "post_unblind_logical_fingerprint_transport_amendment_receipt"
    )
    factor_covariance_ordinal_receipt = checks.get(
        "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_receipt"
    )
    equal_weight_missing_cost_gate_receipt = checks.get(
        "post_unblind_equal_weight_missing_cost_gate_amendment_receipt"
    )
    hash_audit_incident_receipt = checks.get("post_unblind_hash_audit_incident_receipt")
    hash_audit_facts = (
        hash_audit_incident_receipt.get("access_facts")
        if isinstance(hash_audit_incident_receipt, Mapping)
        else None
    )
    disclosed_hash_audit = checks.get("disclosed_amendment_hash_audit")
    replacement_authority = (
        post_unblind_receipt.get("replacement_authority")
        if isinstance(post_unblind_receipt, Mapping)
        else None
    )
    prior_replaced_authorities = (
        post_unblind_receipt.get("prior_replaced_authorities")
        if isinstance(post_unblind_receipt, Mapping)
        else None
    )
    prior_replaced_records = (
        prior_replaced_authorities.get("records")
        if isinstance(prior_replaced_authorities, Mapping)
        else None
    )
    migration_receipt = checks.get(
        "stage13_source_checkpoint_context_migration_receipt"
    )
    migration_parent = (
        migration_receipt.get("parent_context")
        if isinstance(migration_receipt, Mapping)
        else None
    )
    migration_sources = (
        migration_receipt.get("parent_feature_sources")
        if isinstance(migration_receipt, Mapping)
        else None
    )
    migration_safety = (
        migration_receipt.get("safety")
        if isinstance(migration_receipt, Mapping)
        else None
    )
    migration_excluded_parent_state = (
        migration_receipt.get("excluded_parent_state")
        if isinstance(migration_receipt, Mapping)
        else None
    )
    migration_supersedes = (
        migration_receipt.get("supersedes")
        if isinstance(migration_receipt, Mapping)
        else None
    )
    migration_superseded_authority_receipt = (
        migration_receipt.get("superseded_authority_receipt")
        if isinstance(migration_receipt, Mapping)
        else None
    )
    migration_scratch_topology_amendment = (
        migration_receipt.get("scratch_topology_amendment")
        if isinstance(migration_receipt, Mapping)
        else None
    )
    expected_scratch_descendants = [
        "left",
        "left/__stage13_exact_partition=27",
        "left/__stage13_exact_partition=79",
        "right",
        "right/__stage13_exact_partition=20",
        "right/__stage13_exact_partition=94",
    ]
    _semantic_require(
        checks.get("status") == "ALL_POST_UNBLIND_REPLACEMENT_FREEZE_GATES_PASSED"
        and isinstance(check_values, Mapping)
        and all(value is True for value in check_values.values())
        and check_values.get("pre_unblind_calibration_scope_amendment_authenticated")
        is True
        and isinstance(calibration_receipt, Mapping)
        and calibration_receipt.get("amendment_id")
        == "stage12-prefinal-shared-calibration-scope-v1"
        and calibration_receipt.get("status")
        == "STAGE12_PREFINAL_METHODOLOGY_AMENDMENT_FROZEN"
        and calibration_receipt.get("sha256")
        == bindings.get("calibration_scope_amendment_sha256")
        and check_values.get("post_unblind_technical_amendment_authenticated") is True
        and check_values.get("post_unblind_hash_audit_incident_authenticated") is True
        and check_values.get("post_unblind_adapter_transition_exact") is True
        and check_values.get(
            "post_unblind_amendment_zero_outcome_boundary_authenticated"
        )
        is True
        and isinstance(post_unblind_receipt, Mapping)
        and post_unblind_receipt.get("amendment_id")
        == "stage12-post-unblind-auxiliary-nullability-transport-v1"
        and post_unblind_receipt.get("status")
        == "STAGE12_POST_UNBLIND_TECHNICAL_AMENDMENT_AUTHORIZED"
        and post_unblind_receipt.get("sha256")
        == bindings.get("post_unblind_technical_amendment_sha256")
        and isinstance(post_unblind_boundary, Mapping)
        and post_unblind_boundary.get(
            "final_period_predictor_prerequisite_unique_payload_files_opened_by_amendment_work"
        )
        == 64
        and post_unblind_boundary.get(
            "final_period_predictor_prerequisite_payload_file_open_events_by_amendment_work"
        )
        == 320
        and post_unblind_boundary.get(
            "final_period_predictor_prerequisite_bytes_hashed_by_amendment_work"
        )
        == 35953170670
        and all(
            int(post_unblind_boundary.get(name, -1)) == 0
            for name in (
                "outcome_or_economic_payload_files_opened_by_amendment_work",
                "final_period_rows_read_by_amendment_work",
                "economic_months_executed_by_amendment_work",
                "outcome_months_executed_by_amendment_work",
                "final_forecasts_weights_trades_or_pnl_created_by_amendment_work",
            )
        )
        and check_values.get("post_unblind_scratch_topology_amendment_authenticated")
        is True
        and check_values.get("post_unblind_scratch_topology_exact_and_non_mutating")
        is True
        and check_values.get(
            "post_unblind_scratch_topology_zero_payload_boundary_authenticated"
        )
        is True
        and isinstance(scratch_topology_receipt, Mapping)
        and scratch_topology_receipt.get("amendment_id")
        == "stage12-post-unblind-parent-scratch-topology-v1"
        and scratch_topology_receipt.get("status")
        == "STAGE12_POST_UNBLIND_SCRATCH_TOPOLOGY_AMENDMENT_AUTHORIZED"
        and scratch_topology_receipt.get("sha256")
        == bindings.get("post_unblind_scratch_topology_amendment_sha256")
        and isinstance(scratch_topology_authorization, Mapping)
        and scratch_topology_authorization.get("authorization_form")
        == "contextual_affirmative_reply_after_clarification"
        and scratch_topology_authorization.get("source_message_text")
        == "Okay lets start"
        and scratch_topology_authorization.get("authorized") is True
        and scratch_topology_authorization.get("parent_mutation_authorized") is False
        and scratch_topology_authorization.get("destructive_cleanup_authorized")
        is False
        and scratch_topology_authorization.get("stage13_execution_authorized_now")
        is False
        and isinstance(scratch_topology_authorization_receipt, Mapping)
        and scratch_topology_authorization_receipt.get("relative_path")
        == (
            "artifacts/phase_06_final_test/"
            "STAGE_12_SCRATCH_TOPOLOGY_AMENDMENT_AUTHORIZATION.json"
        )
        and scratch_topology_authorization_receipt.get("sha256")
        == "3f0a285671144a91eccccf84cc345c62a73b21e379d9debf0d00c40a44fab3d2"
        and scratch_topology_authorization_receipt.get("status")
        == "STAGE12_NON_MUTATING_SCRATCH_TOPOLOGY_AMENDMENT_AUTHORIZED"
        and scratch_topology_authorization_receipt.get("authorization_id")
        == "stage12-scratch-topology-amendment-authorization-20260726-v1"
        and int(scratch_topology_authorization_receipt.get("size_bytes", 0)) > 0
        and isinstance(scratch_topology_contract, Mapping)
        and scratch_topology_contract.get("root_relative_path")
        == ("full_predictor_context/.duckdb_spill/" "exact-table-compare-1ejjjtbr")
        and scratch_topology_contract.get("exact_descendant_relative_paths")
        == expected_scratch_descendants
        and scratch_topology_contract.get("exact_descendant_directory_count") == 6
        and scratch_topology_contract.get(
            "root_and_descendant_directory_identity_count"
        )
        == 7
        and all(
            scratch_topology_contract.get(name) == 0
            for name in (
                "regular_file_count",
                "payload_bytes",
                "symlink_count",
                "special_node_count",
            )
        )
        and scratch_topology_contract.get(
            "missing_extra_renamed_or_type_drift_permitted"
        )
        is False
        and scratch_topology_contract.get(
            "stable_root_and_descendant_identity_before_after_required"
        )
        is True
        and scratch_topology_contract.get("directory_identity_fields")
        == [
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        ]
        and scratch_topology_contract.get("parent_access_read_only") is True
        and scratch_topology_contract.get(
            "parent_entry_deletion_move_rename_or_write_permitted"
        )
        is False
        and scratch_topology_boundary
        == {
            "parent_entries_deleted_moved_renamed_or_mutated": False,
            "parent_payload_files_opened_by_stage12_amendment_work": 0,
            "outcome_or_economic_payload_files_opened_by_stage12_amendment_work": 0,
            "final_period_rows_read_by_stage12_amendment_work": 0,
            "economic_or_outcome_months_executed_by_stage12_amendment_work": 0,
            (
                "final_forecasts_weights_trades_or_pnl_created_by_"
                "stage12_amendment_work"
            ): 0,
            "scientific_or_economic_result_informed_choice": False,
        }
        and scratch_topology_replacement
        == {
            "replacement_stage12_freeze_required": True,
            "independent_stage12_verification_required": True,
            "prior_stage13_approval_valid_for_replacement_identity": False,
            (
                "renewed_explicit_stage13_approval_bound_to_exact_"
                "replacement_required"
            ): True,
            "renewed_stage13_approval_granted_by_this_amendment": False,
            "stage13_parent_context_open_authorized_now": False,
            "stage13_execution_authorized_now": False,
        }
        and scratch_topology_blocker
        == {
            "relative_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_PARENT_SCRATCH_TOPOLOGY_BLOCKER.json"
            ),
            "sha256": (
                "927fd04990a93ac01faf67bee9e600cf345d830e323c708420fd0d1ab1c6069e"
            ),
            "status": (
                "STAGE13_FAILED_CLOSED_ON_PARENT_SCRATCH_TOPOLOGY_"
                "CONTRACT_PREMISE_MISMATCH"
            ),
        }
        and check_values.get(
            "post_unblind_zero_row_merge_provenance_amendment_authenticated"
        )
        is True
        and check_values.get(
            "post_unblind_zero_row_merge_provenance_exact_scope_authenticated"
        )
        is True
        and check_values.get(
            "post_unblind_zero_row_merge_provenance_non_economic_boundary_authenticated"
        )
        is True
        and check_values.get(
            "post_unblind_entry_market_canonicalization_scope_alignment_amendment_authenticated"
        )
        is True
        and check_values.get(
            "entry_market_exact_duplicate_canonicalization_contract_authenticated"
        )
        is True
        and check_values.get(
            "entry_market_closed_tombstone_share_coverage_contract_authenticated"
        )
        is True
        and check_values.get(
            "entry_market_projection_truthful_receipt_algebra_authenticated"
        )
        is True
        and check_values.get(
            "entry_market_production_preflight_disclosure_authenticated"
        )
        is True
        and check_values.get("entry_market_no_third_projection_blocker_authenticated")
        is True
        and check_values.get("entry_market_standing_stage13_approval_authenticated")
        is True
        and check_values.get("entry_market_runtime_source_transition_exact") is True
        and check_values.get(
            "post_unblind_accounting_missing_execution_cost_transport_amendment_authenticated"
        )
        is True
        and check_values.get(
            "accounting_missing_numeric_transport_preserves_nontradable_semantics"
        )
        is True
        and check_values.get("accounting_frozen_primary_borrow_mapping_authenticated")
        is True
        and check_values.get("accounting_prewrite_nonnullable_audit_authenticated")
        is True
        and check_values.get("accounting_transport_non_economic_boundary_authenticated")
        is True
        and check_values.get("thesis_error_materiality_policy_authenticated") is True
        and accounting_transport_receipt
        == {
            "relative_path": (
                "configs/phase_06_final_test/"
                "stage_12_post_unblind_accounting_missing_execution_cost_"
                "transport_amendment_v1.json"
            ),
            "sha256": (
                "2539bdc9b402be3df192690571e01b50ba83f90a381a9860066c62ee874d3b02"
            ),
            "size_bytes": 8660,
            "authorization": {
                "sha256": (
                    "593c0da4c8a6997a06626b97fc51c3111ed470863394731bf4dac560c89bd80c"
                ),
                "size_bytes": 7256,
            },
            "materiality_policy": {
                "sha256": (
                    "622d1335c540dfa84a4a396bade529e2e3632d1090a41ce66e1990dc582df03d"
                ),
                "size_bytes": 2802,
            },
            "blocker": {
                "sha256": (
                    "9468526f52dae7af4b60589a1efda362c562f2e0a4e739c0fb2a22997f6dc5e3"
                ),
                "size_bytes": 4653,
            },
            "preflight": {
                "sha256": (
                    "feb761454b5ca2ff2cd3b58faad4a902b67e3af57afcc25eed78eef081022deb"
                ),
                "size_bytes": 5762,
            },
        }
        and check_values.get(
            "post_unblind_logical_fingerprint_transport_amendment_authenticated"
        )
        is True
        and check_values.get("logical_fingerprint_schema_and_shape_exact") is True
        and check_values.get("logical_fingerprint_null_masks_and_finite_bits_exact")
        is True
        and check_values.get("logical_fingerprint_nan_payload_canonicalization_only")
        is True
        and check_values.get("logical_fingerprint_persisted_authentication_unchanged")
        is True
        and check_values.get("logical_fingerprint_non_economic_boundary_authenticated")
        is True
        and logical_fingerprint_transport_receipt
        == {
            "relative_path": (
                "configs/phase_06_final_test/"
                "stage_12_post_unblind_logical_fingerprint_transport_amendment_v1.json"
            ),
            "sha256": (
                "b1fe0a1c9064bd26a025b21ac0a6d88d50a2050677a5779b25e822b68ab6a6c7"
            ),
            "size_bytes": 8059,
            "authorization": {
                "sha256": (
                    "cbfa2ee2afd71294d71ea201c83c79a845ae4f87ace205a2f19e5c7d9dbac5e5"
                ),
                "size_bytes": 1695,
            },
            "materiality_policy": {
                "sha256": (
                    "622d1335c540dfa84a4a396bade529e2e3632d1090a41ce66e1990dc582df03d"
                ),
                "size_bytes": 2802,
            },
            "blocker": {
                "sha256": (
                    "5b14591d9c559cb01f27262ceb7e7eaa12af2702e0575ccbf57b39a227c5d197"
                ),
                "size_bytes": 1384,
            },
            "preflight": {
                "sha256": (
                    "8daa9c7e078072bc9fcc73249086eeaf9c608192dd33331942b6896ea3013cf4"
                ),
                "size_bytes": 1767,
            },
        }
        and check_values.get(
            "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_authenticated"
        )
        is True
        and check_values.get("factor_covariance_zero_based_contract_authenticated")
        is True
        and check_values.get("factor_covariance_numeric_bits_unchanged") is True
        and check_values.get("factor_covariance_matrix_semantics_unchanged") is True
        and check_values.get("factor_covariance_non_economic_boundary_authenticated")
        is True
        and factor_covariance_ordinal_receipt
        == {
            "relative_path": (
                "configs/phase_06_final_test/"
                "stage_12_post_unblind_factor_covariance_ordinal_basis_alignment_"
                "amendment_v1.json"
            ),
            "sha256": (
                "9d1f8e93ef49b1621df442fd1760fe24fd6f405bcf06e7f1bd6d67b574f01f76"
            ),
            "size_bytes": 8089,
            "authorization": {
                "sha256": (
                    "9c762291ce317c92837380ab02ccdf820d82c254f0c10252f5f3f715b40eb001"
                ),
                "size_bytes": 2046,
            },
            "materiality_policy": {
                "sha256": (
                    "622d1335c540dfa84a4a396bade529e2e3632d1090a41ce66e1990dc582df03d"
                ),
                "size_bytes": 2802,
            },
            "blocker": {
                "sha256": (
                    "3d8e1a80741aadae7773368d17bc04541e01b4aee21a48e8ebe2f3b278630a18"
                ),
                "size_bytes": 1498,
            },
            "preflight": {
                "sha256": (
                    "49e043391a4ea27f8392a55685cfd72151d4ebd9c1119f7b2f05b698c6926530"
                ),
                "size_bytes": 1535,
            },
            "failed_launch_memo": {
                "sha256": (
                    "9c0c3d1b675e86b8dad97bfcb5361c1f7fae212e73710f8afe84448ac9d3db3e"
                ),
                "size_bytes": 4361,
            },
        }
        and isinstance(factor_covariance_source_amendment, Mapping)
        and factor_covariance_source_amendment.get("amendment_id")
        == "stage12-post-unblind-factor-covariance-ordinal-basis-alignment-v1"
        and factor_covariance_source_amendment.get("status")
        == "STAGE12_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT_AUTHORIZED"
        and factor_covariance_source_amendment.get("classification")
        == (
            "non_economic_rule_restoring_zero_based_factor_covariance_ordinal_"
            "contract_alignment"
        )
        and factor_covariance_source_amendment.get("diagnosis")
        == {
            "asymmetric_months": 0,
            "covariance_daily_bit_differences": 0,
            "covariance_monthly_bit_differences": 0,
            "covariance_rows": 2592,
            "factor_name_or_order_mismatches": 0,
            "incorrect_stage13_guard_domain": [1, 2, 3, 4, 5, 6],
            "minimum_monthly_eigenvalue": 7.802845357290469e-05,
            "monthly_scale_bit_differences_from_21_times_daily": 0,
            "months": 72,
            "non_psd_months": 0,
            "nonfinite_cells": 0,
            "source_ordinal_domain": [0, 1, 2, 3, 4, 5],
        }
        and factor_covariance_source_amendment.get("ordinal_contract")
        == {
            "accept_both_bases_permitted": False,
            "authoritative_basis": "zero_based_0_through_5",
            "renumber_stored_source_permitted": False,
            "stage11_materialization_verifier_requires_zero_based": True,
            "stage11_portfolio_backend_requires_zero_based": True,
            "stage13_compiler_must_require_zero_based": True,
            "stage13_runtime_must_require_zero_based": True,
            "stage8_generator_uses_zero_based_enumerate": True,
            "stage8_independent_verifier_requires_zero_based": True,
        }
        and factor_covariance_source_amendment.get("non_economic_boundary")
        == {
            "covariance_values_changed": False,
            "economic_month_checkpoints_before_failure": 0,
            "factor_names_or_order_changed": False,
            "matrix_sort_or_reshape_changed": False,
            "risk_formulas_models_weights_trades_returns_pnl_or_inference_changed": False,
            "schemas_changed": False,
        }
        and factor_covariance_source_amendment.get("source_transition")
        == {
            "exact_changed_runtime_member_count": 2,
            "members": factor_covariance_changes,
        }
        and factor_covariance_source_amendment.get("focused_test_evidence")
        == {
            "test_files": [
                {
                    "path": "tests/unit/test_final_input_construction.py",
                    "post_amendment_sha256": (
                        "a3fc7548a75c27a76a3b68bc68cfa8ab338ffb14ae47bfeaad58554bb7f5b4cf"
                    ),
                    "pre_amendment_sha256": (
                        "d34cacd45cf2b8bd520d1a656dbd2a6ff4ff510113beb298cadb756fc0f151ea"
                    ),
                },
                {
                    "path": "tests/unit/test_final_source_builder.py",
                    "post_amendment_sha256": (
                        "49084d2ce45ca83ac7782140f1219ce33a3f161b3048c9a478c2372e6fecd9e1"
                    ),
                    "pre_amendment_sha256": (
                        "828d86021a4b511c1a892817ef56dd56f91b0b41f0cf558ab2cd5ca757755824"
                    ),
                },
                {
                    "path": "tests/unit/test_final_test_production_runtime.py",
                    "post_amendment_sha256": (
                        "3458d95cf8fae3bbb9401657dbcbcdc30894f348113c629a9db9163d2880b016"
                    ),
                    "pre_amendment_sha256": (
                        "559ca9cd58cb45cb1f5f033defad59b97577e035abad68909fe1b9b5a2303904"
                    ),
                },
            ],
            "tests_passed": 60,
        }
        and check_values.get(
            "post_unblind_equal_weight_missing_cost_gate_amendment_authenticated"
        )
        is True
        and check_values.get("equal_weight_new_position_execution_gate_authenticated")
        is True
        and check_values.get("equal_weight_current_risk_input_gate_authenticated")
        is True
        and check_values.get("equal_weight_required_exit_exclusion_authenticated")
        is True
        and check_values.get(
            "equal_weight_missing_cost_inherited_exact_hold_authenticated"
        )
        is True
        and check_values.get(
            "equal_weight_frozen_hold_beta_fallback_boundary_authenticated"
        )
        is True
        and check_values.get(
            "equal_weight_exact_zero_trade_and_frozen_gross_failure_authenticated"
        )
        is True
        and check_values.get("equal_weight_residual_gross_allocation_authenticated")
        is True
        and check_values.get(
            "equal_weight_invalid_input_zero_capacity_and_nonzero_trade_fail_closed_authenticated"
        )
        is True
        and check_values.get("equal_weight_descriptive_candidate_audit_authenticated")
        is True
        and check_values.get(
            "equal_weight_rule_restoring_economic_boundary_authenticated"
        )
        is True
        and check_values.get("v10_model_state_direct_reuse_prohibited") is True
        and check_values.get("fresh_v11_six_model_fits_required") is True
        and equal_weight_missing_cost_gate_receipt
        == {
            "relative_path": (
                "configs/phase_06_final_test/"
                "stage_12_post_unblind_equal_weight_missing_cost_gate_"
                "amendment_v1.json"
            ),
            "sha256": (
                "72a94490bf1c835871104c966991ce42ef5a5ee5ee33ab7aabb20570d3de8ffa"
            ),
            "semantic_sha256": (
                "4968272bb6117cb3c808e1f0679b1053aaa3b3a3459bf7c9e8105f7b03d00b7f"
            ),
            "size_bytes": 8943,
            "authorization": {
                "sha256": (
                    "6f8063509a88792cb22d4d446a3633dc040a8a53030f6b79ae7e5076a8850a95"
                ),
                "size_bytes": 5675,
            },
            "materiality_policy": {
                "sha256": (
                    "622d1335c540dfa84a4a396bade529e2e3632d1090a41ce66e1990dc582df03d"
                ),
                "size_bytes": 2802,
            },
            "resume3_failure": {
                "sha256": (
                    "d0e4b76bf16bc2caa56e218b8af7159abb9dfaa0992e9b0a14104a6239112056"
                ),
                "size_bytes": 4687,
            },
            "preflight": {
                "sha256": (
                    "08c43ef17fad63878f3c996a8f863f75039ba707a849d6adc5235e30d9e1e5cf"
                ),
                "size_bytes": 7179,
            },
        }
        and isinstance(equal_weight_source_amendment, Mapping)
        and equal_weight_source_amendment.get("amendment_id")
        == "stage12-post-unblind-equal-weight-missing-cost-gate-v1"
        and equal_weight_source_amendment.get("status")
        == "STAGE12_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_AUTHORIZED"
        and equal_weight_source_amendment.get("classification")
        == "rule_restoring_economic_benchmark_execution_gate_correction"
        and equal_weight_source_amendment.get("source_transition")
        == {
            "exact_changed_runtime_member_count": 3,
            "members": equal_weight_changes,
        }
        and equal_weight_source_amendment.get("execution_gate_contract", {}).get(
            "new_or_increased_long_requires_complete_finite_current_factor_betas_and_positive_finite_specific_variance"
        )
        is True
        and equal_weight_source_amendment.get("execution_gate_contract", {}).get(
            "frozen_inherited_exact_hold_with_missing_current_beta_uses_last_authenticated_inherited_beta"
        )
        is True
        and equal_weight_source_amendment.get("execution_gate_contract", {}).get(
            "inherited_beta_fallback_may_not_authorize_new_or_increased_risk"
        )
        is True
        and equal_weight_source_amendment.get("checkpoint_reuse")
        == {
            "economic_checkpoint_reuse_available": False,
            "fresh_v11_full_and_reduced_model_fits_required": 6,
            "restart_boundary": "primary_full_2020-01_ordinal_1_predecessor_null",
            "resume3_economic_checkpoint_count": 0,
            "v10_model_state_direct_reuse_permitted": False,
        }
        and equal_weight_source_amendment.get("economic_boundary")
        == {
            "change_required_to_restore_prespecified_common_execution_gate": True,
            "common_transaction_cost_model_changed": False,
            "cost_input_transport_changed": False,
            "equal_weight_benchmark_outputs_may_change": True,
            "models_scores_ranks_and_other_five_strategy_rules_changed": False,
            "outcome_months_accepted_before_failure": 0,
            "stage1_through_stage11_retraining_required": False,
        }
        and isinstance(zero_row_receipt, Mapping)
        and zero_row_receipt.get("amendment_id")
        == "stage12-post-unblind-zero-row-merge-provenance-v1"
        and zero_row_receipt.get("status")
        == "STAGE12_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_AUTHORIZED"
        and zero_row_receipt.get("sha256")
        == bindings.get("post_unblind_zero_row_merge_provenance_amendment_sha256")
        and zero_row_authorization_receipt
        == {
            "relative_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_12_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_AUTHORIZATION.json"
            ),
            "sha256": (
                "aa669c218dac6ab82df278f6a1013261b2f5d3f32335ac295f7cca225af40c4f"
            ),
            "size_bytes": 4225,
            "status": "STAGE12_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_AUTHORIZED",
            "authorization_id": (
                "stage12-zero-row-merge-provenance-amendment-"
                "authorization-20260726-v1"
            ),
        }
        and zero_row_blocker_receipt
        == {
            "relative_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_ZERO_ROW_COPY_GUARD_BLOCKER.json"
            ),
            "sha256": (
                "7fc5fecfefd4c291bc6196e16d7fd2f14f0763cf0cefcc1475ef869c44465bd7"
            ),
            "size_bytes": 5279,
            "status": "BLOCKED_PENDING_NARROW_TECHNICAL_AMENDMENT_AUTHORIZATION",
        }
        and zero_row_scientific_scope
        == {
            "registered_feature_count": 163,
            "inactive_segment_feature_count": 4,
            "inactive_segment_features": [
                "business_segment_count",
                "geographic_segment_count",
                "business_segment_revenue_hhi",
                "business_segment_revenue_hhi_change_1y",
            ],
            "feature_registry_unchanged": True,
            "inactive_segment_support_unchanged": True,
            "rows_values_dates_types_keys_linkage_formulas_models_and_economics_unchanged": True,
        }
        and zero_row_provenance_contract
        == {
            "scope": "only_zero_plus_zero_auxiliary_merges",
            "observed_collision_table": "compustat_segment_observation.parquet",
            "development_rows_required": 0,
            "extension_rows_required": 0,
            "result_rows_required": 0,
            "result_row_groups_required": 0,
            "merge_operator": "UNION ALL",
            "physical_provenance_metadata_key": (
                "portfolio_bot.zero_row_auxiliary_merge_provenance"
            ),
            "independent_output_inode_required": True,
            "output_sha256_must_differ_from_each_parent": True,
            "exact_field_schema_and_zero_rows_must_be_preserved": True,
            "copy_hardlink_symlink_inode_alias_and_nonordinary_rejection_retained": True,
            "nonzero_or_one_sided_merge_behavior_changed": False,
        }
        and zero_row_source_transition
        == {
            "exact_changed_runtime_member_count": 2,
            "members": [
                {
                    "path": (
                        "src/portfolio_bot/validation/"
                        "validation_feature_input_adapter.py"
                    ),
                    "pre_amendment_sha256": (
                        "e2b3a85d32c87f0de42b67f75972a6a6203073fb603769cd7e11cbcfb69b7ec0"
                    ),
                    "post_amendment_sha256": (
                        "a2d8ac04a84f957c61fca922b353116aa7c48512d188a2fb3f0547be5f1f6d45"
                    ),
                },
                {
                    "path": "src/portfolio_bot/final_test/final_source_replay.py",
                    "pre_amendment_sha256": (
                        "4cd88a6fbb702207004f66006364efc5cf0a7d66633ebb6361bbb6f2cc2c53de"
                    ),
                    "post_amendment_sha256": (
                        "ebecc9d341331cb7e6dc0b15cd45bc522ed7dda5796608542b53412c8ccc28c6"
                    ),
                },
            ],
        }
        and zero_row_boundary
        == {
            "failed_context_or_sealed_parent_mutated": False,
            "parent_payload_files_opened_by_stage12_amendment_work": 0,
            "outcome_or_economic_payload_files_opened_by_stage12_amendment_work": 0,
            "final_period_rows_read_by_stage12_amendment_work": 0,
            "economic_or_outcome_months_executed_by_stage12_amendment_work": 0,
            "final_forecasts_weights_trades_or_pnl_created_by_stage12_amendment_work": 0,
            "scientific_or_economic_result_informed_choice": False,
        }
        and zero_row_replacement
        == {
            "replacement_stage12_freeze_required": True,
            "independent_stage12_verification_required": True,
            "prior_stage13_approval_valid_for_replacement_identity": False,
            "renewed_explicit_stage13_approval_bound_to_exact_replacement_required": True,
            "renewed_stage13_approval_granted_by_this_amendment": False,
            "stage13_parent_context_open_authorized_now": False,
            "stage13_execution_authorized_now": False,
        }
        and bindings.get("zero_row_merge_provenance_authorization_sha256")
        == "aa669c218dac6ab82df278f6a1013261b2f5d3f32335ac295f7cca225af40c4f"
        and bindings.get("stage13_zero_row_copy_guard_blocker_sha256")
        == "7fc5fecfefd4c291bc6196e16d7fd2f14f0763cf0cefcc1475ef869c44465bd7"
        and bindings.get("prior_zero_row_provenance_stage12_primary_pointer_sha256")
        == "4c8bbec0ffc8c35319f661dacde9ddd292628a6c3a62598c6b863980801eb0c2"
        and bindings.get("prior_zero_row_provenance_stage12_verified_pointer_sha256")
        == "5cda903338b0877222991084891afc79bfe277069e81408f1123dc85b93e05b5"
        and bindings.get("prior_zero_row_provenance_stage13_approval_sha256")
        == "b366335728dda2559b07acbcc4bb7ceb52bdcff64bb6d9993a934425cd1dec4e"
        and bindings.get("zero_row_adapter_pre_amendment_sha256")
        == "e2b3a85d32c87f0de42b67f75972a6a6203073fb603769cd7e11cbcfb69b7ec0"
        and bindings.get("zero_row_adapter_post_amendment_sha256")
        == "a2d8ac04a84f957c61fca922b353116aa7c48512d188a2fb3f0547be5f1f6d45"
        and bindings.get("zero_row_replay_pre_amendment_sha256")
        == "4cd88a6fbb702207004f66006364efc5cf0a7d66633ebb6361bbb6f2cc2c53de"
        and bindings.get("zero_row_replay_post_amendment_sha256")
        == "ebecc9d341331cb7e6dc0b15cd45bc522ed7dda5796608542b53412c8ccc28c6"
        and isinstance(hash_audit_incident_receipt, Mapping)
        and hash_audit_incident_receipt.get("status")
        == "STAGE12_POST_UNBLIND_HASH_AUDIT_INCIDENT_DISCLOSED"
        and hash_audit_incident_receipt.get("sha256")
        == bindings.get("post_unblind_hash_audit_incident_sha256")
        and isinstance(hash_audit_facts, Mapping)
        and hash_audit_facts.get("complete_hash_scans") == 5
        and hash_audit_facts.get("unique_predictor_prerequisite_payload_files") == 64
        and hash_audit_facts.get("predictor_prerequisite_payload_file_open_events")
        == 320
        and hash_audit_facts.get("total_predictor_prerequisite_bytes_hashed")
        == 35953170670
        and hash_audit_facts.get("outcome_or_economic_payload_files_opened") == 0
        and hash_audit_facts.get("rows_parsed") == 0
        and hash_audit_facts.get("values_parsed") == 0
        and disclosed_hash_audit == hash_audit_facts
        and isinstance(replacement_authority, Mapping)
        and replacement_authority.get("replacement_stage12_freeze_required") is True
        and replacement_authority.get("independent_stage12_verification_required")
        is True
        and replacement_authority.get(
            "prior_stage13_approval_valid_for_replacement_identity"
        )
        is False
        and replacement_authority.get(
            "renewed_explicit_stage13_approval_bound_to_exact_replacement_required"
        )
        is True
        and replacement_authority.get("stage13_execution_authorized_now") is False
        and isinstance(prior_replaced_authorities, Mapping)
        and prior_replaced_authorities.get("status")
        == "PRIOR_REPLACED_AUTHORITIES_AUTHENTICATED"
        and isinstance(prior_replaced_records, Mapping)
        and set(prior_replaced_records)
        == {
            "primary_pointer",
            "verified_pointer",
            "stage13_approval",
            "stage13_resume_blocker",
            "primary_manifest",
            "verification_report",
        }
        and all(isinstance(value, Mapping) for value in prior_replaced_records.values())
        and prior_replaced_records["primary_pointer"].get("sha256")
        == bindings.get("prior_stage12_primary_pointer_sha256")
        and prior_replaced_records["verified_pointer"].get("sha256")
        == bindings.get("prior_stage12_verified_pointer_sha256")
        and prior_replaced_records["stage13_approval"].get("sha256")
        == bindings.get("prior_stage13_approval_sha256")
        and prior_replaced_records["stage13_resume_blocker"].get("sha256")
        == bindings.get("stage13_resume_blocker_sha256")
        and check_values.get(
            "stage13_source_checkpoint_context_migration_authenticated"
        )
        is True
        and check_values.get("stage13_parent_historical_closed_tree_identity_bound")
        is True
        and check_values.get(
            "stage13_parent_context_open_still_requires_replacement_approval"
        )
        is True
        and isinstance(migration_receipt, Mapping)
        and migration_receipt.get("migration_id")
        == "stage13-source-checkpoint-context-migration-v2"
        and migration_receipt.get("status")
        == "STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_V2_AUTHORIZED"
        and migration_receipt.get("sha256")
        == bindings.get("stage13_source_checkpoint_context_migration_sha256")
        and migration_supersedes
        == {
            "path": (
                "configs/phase_06_final_test/"
                "stage_13_source_checkpoint_context_migration_v1.json"
            ),
            "sha256": (
                "956fe984d350cff58142affb2f3832f0bd7b061749f61f998b2d817c59012b13"
            ),
            "migration_id": "stage13-source-checkpoint-context-migration-v1",
        }
        and migration_superseded_authority_receipt
        == {
            "relative_path": (
                "configs/phase_06_final_test/"
                "stage_13_source_checkpoint_context_migration_v1.json"
            ),
            "sha256": (
                "956fe984d350cff58142affb2f3832f0bd7b061749f61f998b2d817c59012b13"
            ),
            "size_bytes": 5661,
            "migration_id": "stage13-source-checkpoint-context-migration-v1",
            "status": "STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_AUTHORIZED",
        }
        and migration_scratch_topology_amendment
        == {
            "path": (
                "configs/phase_06_final_test/"
                "stage_12_post_unblind_scratch_topology_amendment_v1.json"
            ),
            "sha256": (
                "8e394a97d61e7f18b16baef65a29402960a73f83871151047012a3ee48ea97fb"
            ),
            "amendment_id": "stage12-post-unblind-parent-scratch-topology-v1",
            "blocker_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_PARENT_SCRATCH_TOPOLOGY_BLOCKER.json"
            ),
            "blocker_sha256": (
                "927fd04990a93ac01faf67bee9e600cf345d830e323c708420fd0d1ab1c6069e"
            ),
        }
        and isinstance(migration_parent, Mapping)
        and migration_parent.get("context_sha256")
        == "dbccbf9cd220d890d8c025bb6e92854bf61098f1eb42517da934c62b98e7adf5"
        and isinstance(migration_sources, Mapping)
        and migration_sources.get("file_count") == 64
        and migration_sources.get("total_size_bytes") == 7190634134
        and migration_sources.get("canonical_tree_sha256")
        == bindings.get("stage13_parent_feature_sources_tree_sha256")
        and isinstance(migration_safety, Mapping)
        and migration_safety.get("parent_access_read_only") is True
        and migration_safety.get("parent_tree_must_be_identical_before_after") is True
        and migration_safety.get(
            "migration_created_hardlinks_copies_symlinks_forbidden"
        )
        is True
        and migration_safety.get("feature_bundle_reuse_forbidden") is True
        and migration_safety.get("decision_outcome_economic_reuse_forbidden") is True
        and migration_safety.get("ignored_recursively_payload_empty_scratch_topologies")
        == [
            {
                "relative_path": (
                    "full_predictor_context/.duckdb_spill/"
                    "exact-table-compare-1ejjjtbr"
                ),
                "exact_descendant_relative_paths": expected_scratch_descendants,
            }
        ]
        and migration_safety.get("ignored_scratch_topology_must_match_exactly") is True
        and migration_safety.get("ignored_scratch_regular_file_count_required") == 0
        and migration_safety.get("ignored_scratch_payload_bytes_required") == 0
        and migration_safety.get("ignored_scratch_symlink_count_required") == 0
        and migration_safety.get("ignored_scratch_special_node_count_required") == 0
        and migration_safety.get(
            "ignored_scratch_directory_identities_before_after_must_match"
        )
        is True
        and migration_safety.get(
            "ignored_scratch_entries_must_not_be_removed_or_mutated"
        )
        is True
        and migration_safety.get("new_context_allowed_writes")
        == [
            "feature_input_bundle",
            "stage13_source_checkpoint_context_migration_receipt.json",
        ]
        and migration_safety.get("stage12_tests_synthetic_only") is True
        and isinstance(migration_excluded_parent_state, Mapping)
        and migration_excluded_parent_state.get("only_parent_month_receipt_family")
        == "final_calendar"
        and migration_excluded_parent_state.get("final_calendar_month_receipt_count")
        == 72
        and migration_excluded_parent_state.get("parent_final_calendar_receipts_reused")
        is False
        and migration_excluded_parent_state.get("new_context_final_calendar_recomputed")
        is True
        and migration_excluded_parent_state.get("final_predictor_month_receipt_count")
        == 0
        and migration_excluded_parent_state.get(
            "economic_or_outcome_month_checkpoint_count"
        )
        == 0
        and migration_excluded_parent_state.get("feature_input_bundle_present") is False
        and migration_excluded_parent_state.get("decision_source_barrier_present")
        is False
        and check_values.get(
            "post_unblind_december_market_input_scope_partition_amendment_authenticated"
        )
        is True
        and check_values.get("prior_unguarded_empirical_regression_access_disclosed")
        is True
        and check_values.get(
            "december_exact_entry_market_projection_scope_authenticated"
        )
        is True
        and check_values.get(
            "decision_source_preflight_and_completeness_gate_authenticated"
        )
        is True
        and check_values.get("stage13_decision_checkpoint_rebind_authenticated") is True
        and check_values.get(
            "stage13_decision_checkpoint_rebind_431_checkpoint_pins_authenticated"
        )
        is True
        and check_values.get(
            "stage13_preserved_decision_checkpoint_cache_reauthentication_contract_authenticated"
        )
        is True
        and check_values.get(
            "stage13_decision_checkpoint_rebind_stage12_synthetic_only"
        )
        is True
        and check_values.get("checkpoint_rebind_metadata_exception_authenticated")
        is True
        and check_values.get(
            "checkpoint_rebind_metadata_source_transition_authenticated"
        )
        is True
        and isinstance(december_receipt, Mapping)
        and december_receipt.get("status")
        == "STAGE12_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT_AUTHORIZED"
        and december_receipt.get("amendment_id")
        == "stage12-post-unblind-december-market-input-scope-partition-v1"
        and december_receipt.get("sha256")
        == bindings.get(
            "post_unblind_december_market_input_scope_partition_amendment_sha256"
        )
        and isinstance(december_authorization_receipt, Mapping)
        and december_authorization_receipt.get("sha256")
        == bindings.get("december_market_input_scope_partition_authorization_sha256")
        == "b626fba569d16a00ee0235006f1cd66cd9c8f6021c94686e854643411a03313a"
        and isinstance(december_blocker_receipt, Mapping)
        and december_blocker_receipt.get("sha256")
        == bindings.get("stage13_december_market_input_blocker_sha256")
        == "82a08d940c3b7f6f54bdbd4bc3e6690754f8298c70e4ce57ad366ea80528329b"
        and december_scope
        == {
            "holding_month_count": 72,
            "first_holding_month": "2020-01",
            "last_holding_month": "2025-12",
            "last_signal_date": "2025-11-28",
            "exact_partial_entry_date": "2025-12-01",
            "last_exit_date": "2025-12-31",
            "decision_predictor_and_return_maximum_date": "2025-11-28",
            "entry_market_projection_exact_dates": ["2025-12-01"],
            "entry_market_projection_consumer_columns_only": True,
            "entry_market_projection_return_columns_requested": [],
            "entry_market_projection_labels_outcomes_accounting_or_performance_opened": False,
            "outcome_market_earliest_permitted_date": "2025-11-29",
            "outcome_sources_open_before_complete_decision_barrier": False,
            "calendar_dates_schemas_keys_features_models_and_economic_rules_changed": False,
        }
        and isinstance(december_preflight, Mapping)
        and december_preflight.get(
            "provider_preflight_runs_before_checkpoint_store_construction"
        )
        is True
        and december_preflight.get(
            "provider_preflight_requires_exact_72_month_calendar"
        )
        is True
        and december_preflight.get(
            "provider_preflight_authenticates_all_six_frozen_calendar_anchors"
        )
        is True
        and december_preflight.get("provider_preflight_opens_outcome_sources") is False
        and december_preflight.get(
            "family_commit_requires_built_month_set_equal_pending_month_set"
        )
        is True
        and december_preflight.get(
            "partial_or_missing_family_build_must_fail_before_first_fragment_commit"
        )
        is True
        and isinstance(december_rebind, Mapping)
        and december_rebind.get("sha256")
        == bindings.get("stage13_decision_checkpoint_rebind_sha256")
        == "1453e467c622913660a9097fc260c5632bbd963474f5b99c167478212ee70908"
        and december_rebind.get("authority_revision") == 2
        and december_rebind.get("supersedes")
        == {
            "path": (
                "configs/phase_06_final_test/"
                "stage_13_decision_checkpoint_rebind_v1.json"
            ),
            "sha256": (
                "178db1903c087cb53b5f6deb34b6df85da92ae66e55af2124ecf63a69f9fec27"
            ),
            "authority_revision": 1,
        }
        and december_rebind.get("status")
        == "STAGE13_DECISION_CHECKPOINT_V4_TO_V5_REBIND_AUTHORIZED"
        and december_rebind.get("rebind_id")
        == "stage13-decision-checkpoint-v4-to-v5-rebind-v1"
        and isinstance(december_rebind_aggregates, Mapping)
        and december_rebind_aggregates.get("checkpoint_count")
        == bindings.get("stage13_decision_checkpoint_rebind_receipt_count")
        == 431
        and december_rebind_aggregates.get("receipt_count") == 431
        and december_rebind_aggregates.get("total_rows") == 1522126
        and december_rebind_aggregates.get("total_payload_bytes") == 505136766
        and december_rebind_aggregates.get("logical_content_set_sha256")
        == "566500196d4c1744ec5ab1f83b029f3a1fc22015d5b62220b38cec1bd5fe4d26"
        and isinstance(december_rebind.get("safety"), Mapping)
        and december_rebind["safety"].get("parent_access_read_only") is True
        and december_rebind["safety"].get("payload_copy_bytes_exact") is True
        and december_rebind["safety"].get("hardlinks_symlinks_aliases_forbidden")
        is True
        and december_rebind["safety"].get("post_completion_state_verify_only") is True
        and december_rebind["safety"].get("completion_receipt_published_last") is True
        and december_rebind["safety"].get("stage12_tests_synthetic_only") is True
        and isinstance(december_rebind_cache, Mapping)
        and december_rebind_cache.get("reuse_mode")
        == "authenticated_read_only_direct_parent_paths"
        and december_rebind_cache.get("relative_path") == "formula_work/decision_market"
        and december_rebind_cache.get("typed_relative_path") == "typed"
        and december_rebind_cache.get("typed_receipt_sha256")
        == "1b287e50a9e7d2eb2ed7bfa2c4e84bf8dc52185899d7fda6bed6949589fc248e"
        and december_rebind_cache.get("returns_relative_path") == "returns"
        and december_rebind_cache.get("return_receipt_sha256")
        == "c01951d75930f6e46387781a78a452afb6a1c267081a3d06cacf204c99e3e7e1"
        and isinstance(metadata_exception_receipt, Mapping)
        and set(metadata_exception_receipt)
        == {
            "checkpoint_rebind_metadata_correction_authorization",
            "stage13_checkpoint_rebind_metadata_blocker",
            "stage13_checkpoint_rebind_metadata_preflight",
        }
        and metadata_exception_receipt[
            "checkpoint_rebind_metadata_correction_authorization"
        ].get("sha256")
        == bindings.get("checkpoint_rebind_metadata_correction_authorization_sha256")
        and metadata_exception_receipt[
            "stage13_checkpoint_rebind_metadata_blocker"
        ].get("sha256")
        == bindings.get("stage13_checkpoint_rebind_metadata_blocker_sha256")
        and metadata_exception_receipt[
            "stage13_checkpoint_rebind_metadata_preflight"
        ].get("sha256")
        == bindings.get("stage13_checkpoint_rebind_metadata_preflight_sha256")
        and december_rebind_cache.get("legacy_market_extension_source_sha256")
        == "1bd9a4487351f1ac3337f2d3216f946ba553ee43a6006353ed1a019d7ad09446"
        and december_rebind_cache.get("reconciliation_source_sha256")
        == "18e8209005ecb4b2672bb4e62567e61a3920abe4a346104c2abe2163690a1872"
        and december_rebind_cache.get("return_adapter_source_sha256")
        == "d5cc2046f607782625d3509a3f8acb3d8d63a78fdbfe8d43f3f60b25b63f1603"
        and december_rebind_cache_scope
        == {
            "extension_start": "2019-11-01",
            "last_signal": "2025-11-28",
            "last_exit": "2025-12-31",
            "first_context_signal": "2019-10-31",
        }
        and december_rebind_cache_safety
        == {
            "selected_subtrees_read_only": True,
            "selected_subtrees_reauthenticated_before_reuse": True,
            "selected_subtrees_identical_before_after": True,
            "unselected_siblings_not_opened": True,
            "hardlinks_symlinks_special_nodes_forbidden": True,
            "typed_and_return_builder_semantics_unchanged": True,
            "entry_projection_change_semantically_disjoint": True,
            "replacement_cache_copy_authorized": False,
            "fallback_rebuild_on_authentication_failure": False,
            "interrupted_siblings_may_not_be_opened_or_quarantined": True,
        }
        and december_boundary
        == {
            "measurement_scope": (
                "guarded_stage12_publication_and_verification_subprocesses_only"
            ),
            "failed_context_or_sealed_parent_mutated": False,
            "guarded_parent_checkpoint_payload_files_opened": 0,
            "guarded_raw_payload_files_opened": 0,
            "guarded_final_period_payload_files_opened": 0,
            "guarded_final_period_rows_read": 0,
            "guarded_stage13_outcome_or_economic_payload_files_opened": 0,
            "guarded_economic_or_outcome_months_executed": 0,
            "guarded_final_forecasts_weights_trades_or_pnl_created": 0,
            "scientific_or_economic_result_informed_choice": False,
            "prior_unguarded_regression_access_disclosure": {
                "historical_access_detected": True,
                "operator_turn_history_minimum_known_whole_file_test_executions": 4,
                "persisted_execution_receipt_count": 0,
                "complete_low_level_open_event_count_recorded": False,
                "distinct_raw_payload_paths_opened": 1,
                "raw_payload_paths": [
                    "/Users/agraemirolmez/Desktop/Thesis/raw/crsp_delisting_ciz.csv"
                ],
                "raw_physical_lines_read_per_execution": 2,
                "raw_record_period": "not_reauthenticated_unknown",
                "distinct_stage11_interim_payload_paths_opened": 3,
                "stage11_interim_payload_paths": [
                    "/Users/agraemirolmez/Desktop/Thesis/interim/phase_05_"
                    "validation/stage11-validation-data-v1/market_extension/"
                    "typed/crsp_security_day_return_staging.parquet",
                    "/Users/agraemirolmez/Desktop/Thesis/interim/phase_05_"
                    "validation/stage11-validation-data-v1/market_extension/"
                    "typed/crsp_security_day.parquet",
                    "/Users/agraemirolmez/Desktop/Thesis/interim/phase_05_"
                    "validation/stage11-validation-data-v1/market_extension/"
                    "typed/crsp_security_month_return_staging.parquet",
                ],
                "stage11_logical_filter": {
                    "permno": 16493,
                    "start_date": "2016-11-01",
                    "end_date": "2016-11-30",
                },
                "stage11_monthly_logical_rows_per_execution": 1,
                "other_logical_row_physical_row_group_and_byte_counts": (
                    "not_recorded"
                ),
                "stage13_parent_checkpoint_payload_paths_opened": 0,
                "stage13_final_context_opened": False,
                "stage13_outcome_sources_opened": False,
                "final_artifacts_created": 0,
                "payloads_mutated": False,
                "used_to_choose_december_rule_feature_model_or_economic_parameter": (
                    False
                ),
                "corrective_action": (
                    "whole_file_selector_replaced_by_exact_synthetic_nodes_under_"
                    "the_stage12_access_guard"
                ),
            },
        }
        and isinstance(december_replacement, Mapping)
        and december_replacement.get(
            "prior_stage13_approval_valid_for_replacement_identity"
        )
        is False
        and december_replacement.get(
            "renewed_explicit_stage13_approval_bound_to_exact_replacement_required"
        )
        is True
        and december_replacement.get(
            "renewed_stage13_approval_granted_by_this_amendment"
        )
        is False
        and december_replacement.get("stage13_parent_context_open_authorized_now")
        is False
        and december_replacement.get("stage13_execution_authorized_now") is False
        and december_replacement.get("stop_and_ask_before_restarting_stage13") is True
        and bindings.get("prior_december_repair_stage12_primary_pointer_sha256")
        == "72f1f01a00ea2789db5131172f1a8609da4295a00b27ee919b09528fca60debf"
        and bindings.get("prior_december_repair_stage12_verified_pointer_sha256")
        == "a333c7026c014a4a030a9a8f3548fb0fc2516f5786f4ec1023a9755a93db2c74"
        and bindings.get("prior_december_repair_stage13_approval_sha256")
        == "2bc69b11d224f7915d0c63687daf1fa4f6212e97d725ed3602c6f8ef940277ea"
        and isinstance(entry_market_receipt, Mapping)
        and entry_market_receipt.get("relative_path")
        == (
            "configs/phase_06_final_test/"
            "stage_12_post_unblind_entry_market_canonicalization_scope_alignment_"
            "amendment_v1.json"
        )
        and entry_market_receipt.get("sha256")
        == "fb01f060431c4b568053f1588b5c4d21e35371751762c87f76cc566dfa007626"
        and entry_market_receipt.get("size_bytes") == 12404
        and entry_market_receipt.get("amendment_id")
        == "stage12-post-unblind-entry-market-canonicalization-scope-alignment-v1"
        and entry_market_receipt.get("status")
        == "STAGE12_ENTRY_MARKET_CANONICALIZATION_SCOPE_ALIGNMENT_AMENDMENT_AUTHORIZED"
        and entry_market_authorization
        == {
            "relative_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_12_ENTRY_MARKET_CANONICALIZATION_SCOPE_ALIGNMENT_"
                "AMENDMENT_AUTHORIZATION.json"
            ),
            "sha256": (
                "56399168bfcf3b6476018cd0935f7e02d930294330d0e5c0f25a44bd7a75e820"
            ),
            "size_bytes": 7513,
            "status": (
                "STAGE12_ENTRY_MARKET_CANONICALIZATION_SCOPE_ALIGNMENT_"
                "AMENDMENT_AUTHORIZED"
            ),
            "authorization_id": (
                "stage12-entry-market-canonicalization-scope-alignment-"
                "authorization-20260727-v1"
            ),
            "standing_stage13_approval": {
                "additional_user_prompt_after_v6_verification_required": False,
                "approved_at_utc_for_future_receipt": "2026-07-27T07:59:13Z",
                "approved_by": "thesis-owner",
                "current_v5_stage13_approval_valid_for_v6": False,
                "fixed_path_receipt_may_be_replaced_only_after_v6_binding_authentication": True,
                "granted": True,
                "production_issue_stage13_unblind_approval_required": True,
                "production_require_stage13_unblind_authorization_required": True,
                "source_message_id_for_future_receipt": (
                    "thread-user-stage12-v6-standing-approval-20260727-01"
                ),
                "source_message_may_be_bound_to_exact_v6_identity_only_after_independent_verification": True,
                "source_message_sha256_for_future_receipt": (
                    "c1ead3d6041ffbfad689976c6609a70754fbff4c04c05bad701ec57591bde91e"
                ),
                "stage13_approval_receipt_issued_now": False,
                "stage13_execution_authorized_now": False,
                "stage13_launch_before_verified_v6_binding_and_fresh_receipt_forbidden": True,
            },
        }
        and entry_market_blocker_receipt
        == {
            "relative_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_ENTRY_MARKET_PROJECTION_CONTRACT_BLOCKER_20260727.json"
            ),
            "sha256": (
                "80eec8044486f63b92b12f63de3a39d3ba2840561b0f318874963e1e2ec121ac"
            ),
            "size_bytes": 12107,
            "status": (
                "BLOCKED_PENDING_NARROW_ENTRY_MARKET_CANONICALIZATION_AND_"
                "TOMBSTONE_COVERAGE_AMENDMENT_AUTHORIZATION"
            ),
            "classification": (
                "NON_ECONOMIC_ENTRY_MARKET_PROJECTION_GUARD_CONTRADICTIONS"
            ),
        }
        and entry_market_preflight_receipt
        == {
            "relative_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_ENTRY_MARKET_SHARE_COVERAGE_PREFLIGHT_20260727.json"
            ),
            "sha256": (
                "f53e09331597f8197eb38648ed280032ee89601b821e050cab930f0c3e3205a0"
            ),
            "size_bytes": 5894,
            "status": (
                "ENTRY_MARKET_SHARE_COVERAGE_PREFLIGHT_COMPLETE_NO_THIRD_"
                "PROJECTION_BLOCKER"
            ),
        }
        and entry_market_prior_receipts
        == {
            "policy": {
                "relative_path": (
                    "configs/phase_06_final_test/stage_12_freeze_policy_v5.json"
                ),
                "sha256": (
                    "be8486b694926758013a71bde959d24001d0d9b2d9513e94384ee69ccd74adfa"
                ),
                "size_bytes": 15319,
            },
            "file_inventory": {
                "relative_path": (
                    "artifacts/phase_06_final_test/stage_12_freezes/"
                    "stage12-freeze-fce138650a3857d90f0b65a2/file_inventory.json"
                ),
                "sha256": (
                    "182d0ae198340156a7e3c63310cade60de0a0ddb69e8e5488177f8aaa7c636dc"
                ),
                "size_bytes": 200527,
            },
            "primary_pointer_archive": {
                "relative_path": (
                    "artifacts/phase_06_final_test/"
                    "STAGE_12_PRIMARY_PRIOR_FREEZE_FCE13865.json"
                ),
                "sha256": (
                    "5d8820eba667dee1f311811ad7e19224fa5b2d67078224df258765012d5c00fe"
                ),
                "size_bytes": 1394,
            },
            "verified_pointer_archive": {
                "relative_path": (
                    "artifacts/phase_06_final_test/"
                    "STAGE_12_VERIFIED_PRIOR_FREEZE_FCE13865.json"
                ),
                "sha256": (
                    "aed589c5d3740337edc8f4298abc72ec0f9971fc0c0cb3aac289b61abd3b4028"
                ),
                "size_bytes": 1597,
            },
            "stage13_approval_archive": {
                "relative_path": (
                    "artifacts/phase_06_final_test/"
                    "STAGE_13_APPROVAL_PRIOR_STAGE12_FREEZE_FCE13865.json"
                ),
                "sha256": (
                    "dc856b1954e7d7779a43e7c0b7ecf0102f9a7fd8121f2bd3594d6567543283fa"
                ),
                "size_bytes": 973,
            },
        }
        and entry_market_projection
        == {
            "daily_output_return_columns": [],
            "daily_requested_consumer_column_count": 29,
            "daily_requested_return_column_count": 0,
            "daily_source_rows": 49886907,
            "daily_source_sha256": (
                "2e274021d6bf2a4485d2a6f4f3485fdd26c654380ca247ac975456c73e5646d9"
            ),
            "daily_source_size_bytes": 12740735288,
            "entry_date": "2025-12-01",
            "entry_projection_labels_outcomes_accounting_or_performance_opened": False,
            "final_market_inputs_inner_join_semantics_changed": False,
            "shares_source_rows": 2241129,
            "shares_source_sha256": (
                "b66cb7f4b4d2e51b9cd79cad9306911267da35d6c521caa7f15426dc02e2e966"
            ),
            "shares_source_size_bytes": 120698952,
        }
        and entry_market_canonicalization
        == {
            "all_consumer_field_conflicts_remain_fatal": True,
            "canonical_output_keys_unique": True,
            "canonicalizer": (
                "frozen_stage2_all_consumer_field_exact_adjacent_key_duplicate_"
                "collapse"
            ),
            "daily_canonical_rows": 10596,
            "daily_consumer_conflicting_duplicate_groups": 0,
            "daily_consumer_exact_duplicate_affected_rows": 18,
            "daily_consumer_exact_duplicate_key_groups": 9,
            "daily_consumer_exact_duplicate_surplus_rows": 9,
            "daily_output_duplicate_keys": 0,
            "daily_projection_retained_rows": 10605,
            "duplicate_groups_must_not_exceed_surplus_rows": True,
            "duplicate_groups_zero_if_and_only_if_surplus_zero": True,
            "exact_duplicate_multiplicity_only_removed": True,
            "retained_equals_canonical_plus_duplicate_surplus": True,
        }
        and entry_market_share_coverage
        == {
            "active_share_rows": 10593,
            "active_share_rows_without_daily": 0,
            "any_fourth_or_differently_shaped_zero_match_row_fatal": True,
            "final_market_inputs_inner_join_unchanged": True,
            "join_rule": (
                "equal permno and inclusive start_date <= 2025-12-01 <= end_date"
            ),
            "multiple_match_rows": 0,
            "multiple_share_matches_remain_fatal": True,
            "one_match_rows": 10593,
            "population_rows": 10596,
            "predicate_selected_key_set_must_equal_zero_match_key_set": True,
            "shares_duplicate_key_groups": 0,
            "shares_gap_edges": 0,
            "shares_non_entry_interval_rows": 0,
            "shares_overlap_edges": 0,
            "tombstone_key_set_sha256": (
                "5af68213b30cf617bf7b96eac18f1d62a12ca0cda2f3e19a27c132996d7a4b66"
            ),
            "tombstone_key_set_sha256_domain": (
                "stage13-entry-market-zero-share-tombstone-keys-v1\0"
            ),
            "tombstone_predicate_id": (
                "stage13-entry-market-closed-delisting-tombstone-v1"
            ),
            "tombstone_rows": 3,
            "unexpected_zero_match_rows": 0,
            "zero_match_policy": (
                "retain only the exact closed tombstone rows upstream; unchanged "
                "final_market_inputs inner join excludes them"
            ),
            "zero_match_rows": 3,
        }
        and isinstance(entry_market_receipt_contract, Mapping)
        and entry_market_receipt_contract.get("canonicalization_fields")
        == [
            "daily_projection_retained_rows",
            "daily_canonical_rows",
            "daily_consumer_exact_duplicate_key_groups",
            "daily_consumer_exact_duplicate_surplus_rows",
            "daily_consumer_conflicting_duplicate_groups",
            "daily_output_duplicate_keys",
            "shares_duplicate_key_groups",
            "shares_overlap_edges",
            "shares_gap_edges",
            "shares_non_entry_interval_rows",
        ]
        and entry_market_receipt_contract.get("share_coverage_fields")
        == [
            "join_rule",
            "population_rows",
            "one_match_rows",
            "zero_match_rows",
            "multiple_match_rows",
            "active_share_rows",
            "active_share_rows_without_daily",
            "zero_match_policy",
            "tombstone_predicate_id",
            "tombstone_rows",
            "tombstone_key_set_sha256",
            "final_market_inputs_join_semantics",
        ]
        and all(
            entry_market_receipt_contract.get(name) is True
            for name in (
                "all_counts_exact_nonnegative_integers",
                "daily_canonical_equals_daily_parquet_rows",
                "daily_canonical_equals_sum_of_entry_rows_by_date",
                "daily_with_active_shares_plus_active_shares_without_daily_equals_active_shares_rows",
                "daily_with_plus_without_active_shares_equals_daily_canonical",
                "receipt_tamper_must_fail_closed",
                "stable_reauthentication_recomputes_coverage_from_both_parquets",
            )
        )
        and entry_market_source_transition
        == {
            "exact_changed_runtime_member_count": 2,
            "members": [
                {
                    "path": "src/portfolio_bot/final_test/final_source_replay.py",
                    "post_amendment_sha256": (
                        "c6e439f3411aa95692e890be8e8364547b92368e19e8a4af9f7b08ba4d218d58"
                    ),
                    "pre_amendment_sha256": (
                        "73539ebe6b569cfacc94bce78c246766551bfbe174a2f8a8aabc4b126ba96a4f"
                    ),
                },
                {
                    "path": (
                        "src/portfolio_bot/validation/validation_market_extension.py"
                    ),
                    "post_amendment_sha256": (
                        "a5118f43874191e7d4ed3a6c0c2c5730dacb3052e72adbc1b97aee77952a8556"
                    ),
                    "pre_amendment_sha256": (
                        "b54e382705935b90c25372c4fc24cce7013da2beda741ae378e20113977e50f6"
                    ),
                },
            ],
        }
        and isinstance(entry_market_boundary, Mapping)
        and entry_market_boundary.get(
            "calendar_dates_schemas_keys_features_models_or_economic_rules_changed"
        )
        is False
        and entry_market_boundary.get("failed_context_or_preserved_parent_mutated")
        is False
        and all(
            entry_market_boundary.get(name) == 0
            for name in (
                "guarded_stage12_economic_or_outcome_months_executed",
                "guarded_stage12_final_forecasts_weights_trades_or_pnl_created",
                "guarded_stage12_final_period_payload_files_opened",
                "guarded_stage12_final_period_rows_read",
                "guarded_stage12_outcome_or_economic_payload_files_opened",
                "guarded_stage12_parent_checkpoint_payload_files_opened",
                "guarded_stage12_publication_and_verification_raw_payload_files_opened",
            )
        )
        and entry_market_boundary.get("scientific_or_economic_result_informed_choice")
        is False
        and entry_market_replacement
        == {
            "additional_user_prompt_after_v6_verification_required": False,
            "current_stage13_v5_approval_valid_for_v6": False,
            "fresh_binding_specific_stage13_receipt_issued_by_this_amendment": False,
            "fresh_binding_specific_stage13_receipt_required_after_v6": True,
            "independent_stage12_v6_verification_required": True,
            "stage12_v6_replacement_freeze_required": True,
            "stage13_execution_authorized_now": False,
            "stage13_parent_context_open_authorized_now": False,
            "stage13_retry_under_v5_forbidden": True,
            "standing_stage13_approval_granted": True,
        }
        and entry_market_projection
        == entry_market_source_amendment.get("entry_market_projection_contract")
        and entry_market_canonicalization
        == entry_market_source_amendment.get(
            "exact_duplicate_canonicalization_contract"
        )
        and entry_market_share_coverage
        == entry_market_source_amendment.get("closed_tombstone_share_coverage_contract")
        and entry_market_receipt_contract
        == entry_market_source_amendment.get("receipt_contract")
        and entry_market_source_transition
        == entry_market_source_amendment.get("source_transition")
        and entry_market_boundary
        == entry_market_source_amendment.get("non_economic_boundary")
        and entry_market_replacement
        == entry_market_source_amendment.get("replacement_authority")
        and bindings.get(
            "post_unblind_entry_market_canonicalization_scope_alignment_amendment_sha256"
        )
        == "fb01f060431c4b568053f1588b5c4d21e35371751762c87f76cc566dfa007626"
        and bindings.get(
            "entry_market_canonicalization_scope_alignment_authorization_sha256"
        )
        == "56399168bfcf3b6476018cd0935f7e02d930294330d0e5c0f25a44bd7a75e820"
        and bindings.get("stage13_entry_market_projection_contract_blocker_sha256")
        == "80eec8044486f63b92b12f63de3a39d3ba2840561b0f318874963e1e2ec121ac"
        and bindings.get("stage13_entry_market_share_coverage_preflight_sha256")
        == "f53e09331597f8197eb38648ed280032ee89601b821e050cab930f0c3e3205a0"
        and bindings.get("prior_entry_market_alignment_file_inventory_sha256")
        == "182d0ae198340156a7e3c63310cade60de0a0ddb69e8e5488177f8aaa7c636dc"
        and bindings.get("prior_entry_market_alignment_policy_sha256")
        == "be8486b694926758013a71bde959d24001d0d9b2d9513e94384ee69ccd74adfa"
        and bindings.get("prior_entry_market_alignment_stage12_primary_pointer_sha256")
        == "5d8820eba667dee1f311811ad7e19224fa5b2d67078224df258765012d5c00fe"
        and bindings.get("prior_entry_market_alignment_stage12_verified_pointer_sha256")
        == "aed589c5d3740337edc8f4298abc72ec0f9971fc0c0cb3aac289b61abd3b4028"
        and bindings.get("prior_entry_market_alignment_stage13_approval_sha256")
        == "dc856b1954e7d7779a43e7c0b7ecf0102f9a7fd8121f2bd3594d6567543283fa"
        and bindings.get("entry_market_final_source_replay_pre_amendment_sha256")
        == "73539ebe6b569cfacc94bce78c246766551bfbe174a2f8a8aabc4b126ba96a4f"
        and bindings.get("entry_market_final_source_replay_post_amendment_sha256")
        == "c6e439f3411aa95692e890be8e8364547b92368e19e8a4af9f7b08ba4d218d58"
        and bindings.get(
            "entry_market_validation_market_extension_pre_amendment_sha256"
        )
        == "b54e382705935b90c25372c4fc24cce7013da2beda741ae378e20113977e50f6"
        and bindings.get(
            "entry_market_validation_market_extension_post_amendment_sha256"
        )
        == "a5118f43874191e7d4ed3a6c0c2c5730dacb3052e72adbc1b97aee77952a8556"
        and isinstance(calibration_access, Mapping)
        and calibration_access.get("frozen_before_final_period_access") is True
        and all(
            int(calibration_access.get(name, -1)) == 0
            for name in (
                "final_period_payload_files_opened",
                "final_period_rows_read",
                "final_forecasts_created",
                "final_weights_created",
                "final_trades_created",
                "final_pnl_or_performance_rows_created",
            )
        )
        and isinstance(counters, Mapping)
        and counters.get("counter_scope")
        == "guarded_stage12_build_and_test_execution_only"
        and counters.get("raw_payload_files_opened") == 0
        and counters.get("parent_checkpoint_payload_files_opened") == 0
        and counters.get("final_period_payload_files_opened") == 0
        and counters.get("final_period_rows_read") == 0
        and isinstance(authorizations, Mapping)
        and authorizations.get("stage13_technical_gate_passed") is True
        and authorizations.get("stage13_requires_separate_user_approval") is True
        and authorizations.get("stage13_execution_authorized_now") is False
        and authorizations.get("final_period_access_authorized_now") is False
        and authorizations.get("post_unblind_technical_amendment_authorized") is True
        and authorizations.get("post_unblind_scratch_topology_amendment_authorized")
        is True
        and authorizations.get(
            "post_unblind_zero_row_merge_provenance_amendment_authorized"
        )
        is True
        and authorizations.get("prior_stage13_approval_valid_for_replacement_identity")
        is False
        and authorizations.get("renewed_stage13_approval_required") is True
        and authorizations.get("renewed_stage13_approval_granted_now") is False
        and authorizations.get("stage13_source_checkpoint_context_migration_authorized")
        is True
        and authorizations.get(
            "post_unblind_december_market_input_scope_partition_amendment_authorized"
        )
        is True
        and authorizations.get(
            "post_unblind_entry_market_canonicalization_scope_alignment_amendment_authorized"
        )
        is True
        and authorizations.get("standing_stage13_approval_recorded") is True
        and authorizations.get(
            "binding_specific_stage13_receipt_may_be_issued_after_v11_verification"
        )
        is True
        and authorizations.get("additional_user_prompt_after_v11_verification_required")
        is False
        and authorizations.get(
            "post_unblind_accounting_missing_execution_cost_transport_amendment_authorized"
        )
        is True
        and authorizations.get(
            "post_unblind_logical_fingerprint_transport_amendment_authorized"
        )
        is True
        and authorizations.get(
            "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_authorized"
        )
        is True
        and authorizations.get(
            "post_unblind_equal_weight_missing_cost_gate_amendment_authorized"
        )
        is True
        and authorizations.get("stage13_decision_checkpoint_rebind_authorized") is True
        and authorizations.get("checkpoint_rebind_metadata_correction_authorized")
        is True
        and authorizations.get("stage13_parent_context_open_authorized_now") is False
        and manifest.get("counters") == counters
        and manifest.get("authorizations") == authorizations,
        "freeze gate semantic boundary differs",
    )


def _independently_validate_v12_primary_semantics(
    *,
    bundle_path: Path,
    policy: Mapping[str, Any],
    policy_sha256: str,
    pointer: Mapping[str, Any],
) -> None:
    """Independently authenticate the consolidated v12 graph and treatment."""

    documents, receipts = _read_canonical_primary_documents(bundle_path)
    success = documents["_SUCCESS.json"]
    manifest = documents["pre_final_manifest.json"]
    content_receipts = {
        name: receipt
        for name, receipt in receipts.items()
        if name not in {"_SUCCESS.json", "pre_final_manifest.json"}
    }
    success_receipts = {
        name: receipt for name, receipt in receipts.items() if name != "_SUCCESS.json"
    }
    _semantic_require(
        success.get("members") == success_receipts,
        "v12 primary success member graph differs",
    )
    _semantic_require(
        manifest.get("members") == content_receipts,
        "v12 primary manifest member graph differs",
    )
    bindings = manifest.get("bindings")
    _semantic_require(isinstance(bindings, Mapping), "v12 primary bindings are missing")
    assert isinstance(bindings, Mapping)
    methodology = bindings.get("methodology")
    _semantic_require(
        isinstance(methodology, Mapping), "v12 methodology binding is missing"
    )
    assert isinstance(methodology, Mapping)
    source = methodology.get("source")
    _semantic_require(isinstance(source, Mapping), "v12 methodology source is missing")
    assert isinstance(source, Mapping)
    file_inventory = documents["file_inventory.json"]
    files = file_inventory.get("files")
    _semantic_require(isinstance(files, list), "v12 file inventory members are missing")
    assert isinstance(files, list)
    file_by_path = {
        str(item.get("relative_path")): item
        for item in files
        if isinstance(item, Mapping)
    }
    _semantic_require(
        len(file_by_path) == len(files)
        and file_inventory.get("file_count") == len(files)
        and file_inventory.get("tree_sha256")
        == sha256_bytes(canonical_json_bytes(files)),
        "v12 file inventory algebra differs",
    )
    environment_sha256 = sha256_bytes(
        canonical_json_bytes(documents["environment_lock.json"])
    )
    fingerprint_basis = {
        "schema_version": 1,
        "policy_sha256": policy_sha256,
        "methodology_source_sha256": source.get("sha256"),
        "project_tree_sha256": file_inventory.get("tree_sha256"),
        "environment_sha256": environment_sha256,
        "members": content_receipts,
    }
    content_fingerprint = sha256_bytes(canonical_json_bytes(fingerprint_basis))
    freeze_id = f"stage12-freeze-{content_fingerprint[:24]}"
    _semantic_require(
        manifest.get("schema_version") == 1
        and manifest.get("stage") == 12
        and manifest.get("policy_id")
        == "stage12-consolidated-stage13-completion-replacement-freeze-policy-v12"
        and manifest.get("status") == policy["outputs"]["primary_status"]
        and manifest.get("scope")
        == "CONSOLIDATED_SUBSTANTIVE_ECONOMIC_POST_UNBLIND_REPLACEMENT_FREEZE"
        and manifest.get("content_fingerprint") == content_fingerprint
        and manifest.get("freeze_id") == freeze_id,
        "v12 primary manifest identity differs",
    )
    _semantic_require(
        success.get("schema_version") == 1
        and success.get("stage") == 12
        and success.get("status") == policy["outputs"]["primary_status"]
        and success.get("content_fingerprint") == content_fingerprint
        and success.get("freeze_id") == freeze_id,
        "v12 primary success identity differs",
    )
    _semantic_require(
        pointer.get("freeze_id") == freeze_id
        and pointer.get("content_fingerprint") == content_fingerprint
        and pointer.get("manifest_sha256")
        == receipts["pre_final_manifest.json"]["sha256"]
        and pointer.get("success_sha256") == receipts["_SUCCESS.json"]["sha256"],
        "v12 primary pointer binding differs",
    )
    _semantic_require(
        bindings.get("policy_sha256") == policy_sha256
        and bindings.get("project_tree_sha256") == file_inventory.get("tree_sha256")
        and bindings.get("environment_sha256") == environment_sha256,
        "v12 primary digest bindings differ",
    )

    consolidation = policy.get("consolidated_stage13_completion_governance")
    _semantic_require(
        isinstance(consolidation, Mapping), "v12 consolidation policy is missing"
    )
    assert isinstance(consolidation, Mapping)
    inventory_authorization = consolidation.get("inventory_delta_authorization")
    terminal = consolidation.get("terminal_event_treatment")
    registry_contract = consolidation.get("final_input_registry_contract")
    report_template_contract = consolidation.get("report_template_contract")
    result_assembly_contract = consolidation.get("result_assembly_contract")
    implementation_members = consolidation.get("implementation_members")
    focused_test_members = consolidation.get("focused_test_members")
    _semantic_require(
        isinstance(inventory_authorization, Mapping)
        and isinstance(terminal, Mapping)
        and isinstance(registry_contract, Mapping)
        and isinstance(report_template_contract, Mapping)
        and isinstance(result_assembly_contract, Mapping)
        and isinstance(implementation_members, list)
        and isinstance(focused_test_members, list),
        "v12 consolidation subcontracts are missing",
    )
    assert isinstance(inventory_authorization, Mapping)
    assert isinstance(terminal, Mapping)
    assert isinstance(registry_contract, Mapping)
    assert isinstance(report_template_contract, Mapping)
    assert isinstance(result_assembly_contract, Mapping)
    assert isinstance(implementation_members, list)
    assert isinstance(focused_test_members, list)
    terminal_binding = policy.get(
        "stage13_public_interest_last_tradable_close_liquidation"
    )
    _semantic_require(
        isinstance(terminal_binding, Mapping)
        and terminal_binding.get("path")
        == "configs/phase_06_final_test/stage_13_public_interest_last_tradable_close_liquidation_v1.json"
        and isinstance(terminal_binding.get("sha256"), str)
        and len(str(terminal_binding.get("sha256"))) == 64,
        "v12 terminal policy binding differs",
    )
    assert isinstance(terminal_binding, Mapping)
    terminal_sha256 = str(terminal_binding["sha256"])
    standing_instruction_sha256 = (
        "13f24431ba741ce24158877302c803cb18a7f8a4ae435e67376594edeef05681"
    )
    _semantic_require(
        terminal.get("classification")
        == "substantive_economic_post_unblind_data_gap_treatment"
        and terminal.get("policy_sha256") == terminal_sha256
        and terminal.get("exact_expected_event_count") == 2
        and terminal.get("exact_permnos") == [85421, 88888]
        and terminal.get("primary_results_may_change") is True
        and terminal.get("vendor_missing_delisting_return_relabelled_as_zero") is False
        and terminal.get("required_counterfactual_value_multipliers")
        == [0.7, 0.45, 0.0]
        and terminal.get("required_reporting_only_scenario_count") == 4
        and terminal.get("execution_cost_stress_is_separate_not_cartesian") is True
        and terminal.get("within_holding_month_cash_carry_required") is True
        and terminal.get("primary_estimates_modified_by_sensitivity") is False
        and terminal.get("authorization_basis", {}).get(
            "standing_user_instruction_sha256"
        )
        == standing_instruction_sha256
        and terminal.get("authorization_basis", {}).get(
            "fresh_v12_binding_specific_stage13_approval_must_carry_this_basis_before_execution"
        )
        is True,
        "v12 substantive terminal treatment semantics differ",
    )
    _semantic_require(
        dict(registry_contract)
        == {
            "path": "configs/phase_06_final_test/stage_13_input_registry_v1.json",
            "registry_id": "stage13-final-input-registry-v1",
            "schema_version": 1,
            "template_count": 32,
            "total_arrow_column_count": 1210,
            "classification": (
                "authorized_terminal_sensitivity_evidence_transport_schema"
            ),
        }
        and dict(report_template_contract)
        == {
            "historical_stage13_paths": [
                "configs/phase_06_final_test/stage_13_report_template_v1.md"
            ],
            "active_stage13_path": (
                "configs/phase_06_final_test/stage_13_report_template_v2.md"
            ),
            "active_stage14_path": (
                "configs/phase_06_final_test/stage_14_report_template_v1.md"
            ),
            "active_stage13_template_version": 2,
            "historical_templates_may_be_rewritten": False,
        }
        and dict(result_assembly_contract)
        == {
            "assembler_id": "stage13-empirical-result-assembler-v3",
            "implementation_path": (
                "src/portfolio_bot/final_test/production_results.py"
            ),
            "economic_document_count": 13,
            "terminal_sensitivity_scenario_count": 4,
            "primary_estimates_modified": False,
            "model_fitting_scoring_selection_or_allocation_rerun": False,
        },
        "v12 registry, report-template, or assembler contract differs",
    )

    replacement_delta = documents["replacement_inventory_delta.json"]
    changed_members = replacement_delta.get("changed_members")
    added_members = replacement_delta.get("added_members")
    removed_members = replacement_delta.get("removed_members")
    prior_inventory = replacement_delta.get("prior_inventory")
    replacement_inventory = replacement_delta.get("replacement_inventory")
    _semantic_require(
        isinstance(changed_members, list)
        and isinstance(added_members, list)
        and isinstance(removed_members, list)
        and isinstance(prior_inventory, Mapping)
        and isinstance(replacement_inventory, Mapping),
        "v12 replacement inventory delta is incomplete",
    )
    assert isinstance(changed_members, list)
    assert isinstance(added_members, list)
    assert isinstance(removed_members, list)
    assert isinstance(prior_inventory, Mapping)
    assert isinstance(replacement_inventory, Mapping)
    changed_paths = {
        str(item.get("relative_path"))
        for item in changed_members
        if isinstance(item, Mapping)
    }
    added_paths = {
        str(item.get("relative_path"))
        for item in added_members
        if isinstance(item, Mapping)
    }
    _semantic_require(
        replacement_delta.get("status")
        == "EXACT_POST_UNBLIND_REPLACEMENT_INVENTORY_DELTA_AUTHENTICATED"
        and replacement_delta.get("changed_member_count") == len(changed_members)
        and replacement_delta.get("added_member_count") == len(added_members)
        and replacement_delta.get("removed_member_count") == len(removed_members)
        and len(changed_paths) == len(changed_members)
        and len(added_paths) == len(added_members)
        and changed_paths == set(inventory_authorization.get("changed_paths", []))
        and added_paths == set(inventory_authorization.get("added_paths", []))
        and removed_members == inventory_authorization.get("removed_paths") == []
        and prior_inventory.get("sha256")
        == "ad81241b4b6449c9133b5799578c91abd9cd786ae7175345b1cd52d7209dd3ba"
        and prior_inventory.get("file_count") == 610
        and prior_inventory.get("tree_sha256")
        == "77a08b63b34e93a9750e0e820b6329b3817cbf50f5970999d0087a593cd6c032"
        and replacement_inventory.get("file_count") == file_inventory.get("file_count")
        and replacement_inventory.get("tree_sha256")
        == file_inventory.get("tree_sha256"),
        "v12 replacement inventory delta reconciliation differs",
    )
    for item in changed_members + added_members:
        assert isinstance(item, Mapping)
        current = file_by_path.get(str(item.get("relative_path")))
        _semantic_require(
            isinstance(current, Mapping)
            and item.get("replacement_sha256") == current.get("sha256"),
            "v12 replacement member does not bind current inventory",
        )

    delta_semantics = replacement_delta.get("semantic_checks")
    frozen_stage11_adapter_path = "scripts/stage_11_allocation_numerical_adapter.py"
    frozen_stage11_adapter = file_by_path.get(frozen_stage11_adapter_path)
    allocation_delta_paths = {
        path
        for path in changed_paths | added_paths | set(removed_members)
        if path.startswith("src/portfolio_bot/allocation/")
    }
    _semantic_require(
        isinstance(delta_semantics, Mapping)
        and delta_semantics.get(
            "final_input_registry_1210_column_sensitivity_transport_authorized"
        )
        is True
        and delta_semantics.get("historical_stage13_v1_report_template_preserved")
        is True
        and delta_semantics.get("active_stage13_v2_report_template_added") is True
        and delta_semantics.get("stage13_empirical_result_assembler_v3_bound") is True
        and delta_semantics.get("model_code_unchanged") is True
        and delta_semantics.get(
            "allocation_change_limited_to_rule_preserving_solver_api_compatibility"
        )
        is True
        and delta_semantics.get("stage11_allocation_adapter_preserved_byte_exact")
        is True
        and "configs/phase_06_final_test/stage_13_input_registry_v1.json"
        in changed_paths
        and "src/portfolio_bot/allocation/solver.py" in changed_paths
        and "src/portfolio_bot/final_test/production_results.py" in changed_paths
        and "configs/phase_06_final_test/stage_13_report_template_v2.md" in added_paths
        and "src/portfolio_bot/final_test/terminal_event_sensitivity.py" in added_paths
        and allocation_delta_paths == {"src/portfolio_bot/allocation/solver.py"}
        and isinstance(frozen_stage11_adapter, Mapping)
        and frozen_stage11_adapter.get("sha256")
        == "fb5a2afd09a2e331b718d8d7f47842952e3d4aef41bcff7affab797fb49fa838"
        and frozen_stage11_adapter_path
        not in changed_paths | added_paths | set(removed_members),
        "v12 sensitivity/reporting inventory semantics differ",
    )

    registry_document = documents["final_input_registry.json"]
    registry_source = registry_document.get("source_receipt")
    registry = registry_document.get("registry")
    _semantic_require(
        registry_document.get("schema_version") == 1
        and isinstance(registry_source, Mapping)
        and isinstance(registry, Mapping),
        "v12 final-input registry document is incomplete",
    )
    assert isinstance(registry_source, Mapping)
    assert isinstance(registry, Mapping)
    registry_templates = registry.get("templates")
    _semantic_require(
        isinstance(registry_templates, list),
        "v12 final-input registry templates are missing",
    )
    assert isinstance(registry_templates, list)
    total_arrow_column_count = sum(
        len(item.get("arrow_schema", []))
        for item in registry_templates
        if isinstance(item, Mapping)
    )
    registry_inventory_member = file_by_path.get(str(registry_contract.get("path")))
    _semantic_require(
        registry.get("schema_version") == registry_contract.get("schema_version") == 1
        and registry.get("registry_id")
        == registry_contract.get("registry_id")
        == "stage13-final-input-registry-v1"
        and len(registry_templates) == registry_contract.get("template_count") == 32
        and total_arrow_column_count
        == registry_contract.get("total_arrow_column_count")
        == 1210
        and registry_source.get("relative_path") == registry_contract.get("path")
        and registry_source.get("template_count") == 32
        and registry_source.get("total_arrow_column_count") == 1210
        and isinstance(registry_inventory_member, Mapping)
        and registry_source.get("sha256") == registry_inventory_member.get("sha256")
        and bindings.get("final_input_registry_source") == registry_source,
        "v12 final-input registry contract does not independently reconcile",
    )

    source_drift = documents["source_drift.json"]
    drift_members = source_drift.get("authorized_consolidated_changed_members")
    _semantic_require(
        source_drift.get("record_id")
        == "stage12-consolidated-stage13-completion-policy-v12"
        and source_drift.get("status")
        == "AUTHENTICATED_CONSOLIDATED_STAGE13_COMPLETION_SOURCE_TRANSITION"
        and source_drift.get("classification") == consolidation.get("classification")
        and source_drift.get("amendment_receipt", {}).get("sha256") == terminal_sha256
        and source_drift.get("amendment_receipt", {}).get("substantive_economic")
        is True
        and source_drift.get("stage11_recomputed") is False
        and source_drift.get("final_period_rows_read") == 0
        and source_drift.get("stage13_execution_authorized_now") is False
        and source_drift.get("fresh_binding_specific_stage13_approval_required") is True
        and isinstance(drift_members, list)
        and source_drift.get("authorized_consolidated_changed_member_count")
        == len(drift_members)
        == len(implementation_members),
        "v12 consolidated source drift differs",
    )
    assert isinstance(drift_members, list)
    expected_implementation = {
        str(item.get("path")): item
        for item in implementation_members
        if isinstance(item, Mapping)
    }
    required_implementation_paths = {
        "configs/phase_06_final_test/stage_13_input_registry_v1.json",
        "configs/phase_06_final_test/stage_13_public_interest_last_tradable_close_liquidation_v1.json",
        "configs/phase_06_final_test/stage_13_report_template_v2.md",
        "src/portfolio_bot/allocation/solver.py",
        "src/portfolio_bot/final_test/economic_kernel.py",
        "src/portfolio_bot/final_test/final_input_construction.py",
        "src/portfolio_bot/final_test/final_raw_source_publication.py",
        "src/portfolio_bot/final_test/final_source_builder.py",
        "src/portfolio_bot/final_test/final_source_replay.py",
        "src/portfolio_bot/final_test/input_materialization.py",
        "src/portfolio_bot/final_test/production_results.py",
        "src/portfolio_bot/final_test/production_runtime.py",
        "src/portfolio_bot/final_test/publication.py",
        "src/portfolio_bot/final_test/results.py",
        "src/portfolio_bot/final_test/terminal_event_overlay.py",
        "src/portfolio_bot/final_test/terminal_event_sensitivity.py",
        "src/portfolio_bot/selectors/preprocessing.py",
        "src/portfolio_bot/validation/corporate_actions.py",
        "src/portfolio_bot/validation/empirical_selector_backend.py",
        "src/portfolio_bot/validation/session_authority.py",
    }
    observed_implementation = {
        str(item.get("path")): item
        for item in drift_members
        if isinstance(item, Mapping)
    }
    _semantic_require(
        len(expected_implementation) == len(implementation_members)
        and len(observed_implementation) == len(drift_members)
        and set(observed_implementation)
        == set(expected_implementation)
        == required_implementation_paths,
        "v12 consolidated source paths differ",
    )
    for path, expected in expected_implementation.items():
        observed = observed_implementation[path]
        current = file_by_path.get(path)
        _semantic_require(
            isinstance(current, Mapping)
            and observed.get("stage12_v12_sha256")
            == expected.get("sha256")
            == current.get("sha256")
            and observed.get("classification") == expected.get("classification"),
            f"v12 consolidated source identity differs: {path}",
        )
    _semantic_require(
        expected_implementation["src/portfolio_bot/allocation/solver.py"].get(
            "classification"
        )
        == (
            "rule_preserving_legacy_weight_api_compatibility_and_unchanged_"
            "residue_evidence"
        )
        and expected_implementation[
            "configs/phase_06_final_test/stage_13_report_template_v2.md"
        ].get("classification")
        == "active_stage13_terminal_sensitivity_reporting_contract"
        and expected_implementation[
            "src/portfolio_bot/final_test/production_results.py"
        ].get("classification")
        == "stage13_empirical_result_assembler_v3_terminal_sensitivity_assembly"
        and expected_implementation[
            "src/portfolio_bot/final_test/terminal_event_sensitivity.py"
        ].get("classification")
        == "reporting_only_terminal_event_counterfactual_engine",
        "v12 critical sensitivity member classifications differ",
    )
    expected_focused_tests = {
        str(item.get("path")): item
        for item in focused_test_members
        if isinstance(item, Mapping)
    }
    required_focused_test_paths = {
        "tests/unit/conftest.py",
        "tests/unit/test_final_input_construction.py",
        "tests/unit/test_final_input_materialization.py",
        "tests/unit/test_final_raw_source_publication.py",
        "tests/unit/test_final_session_authority.py",
        "tests/unit/test_final_source_builder.py",
        "tests/unit/test_final_terminal_event_overlay.py",
        "tests/unit/test_final_terminal_event_sensitivity.py",
        "tests/unit/test_final_test_economic_kernel.py",
        "tests/unit/test_final_test_production_results.py",
        "tests/unit/test_final_test_production_runtime.py",
        "tests/unit/test_final_test_publication.py",
        "tests/unit/test_final_test_results.py",
        "tests/unit/test_numerical_materiality.py",
        "tests/unit/test_stage11_empirical_execution.py",
        "tests/unit/test_stage12_numerical_materiality_amendment.py",
        "tests/unit/test_stage9_selector_preprocessing.py",
    }
    _semantic_require(
        len(expected_focused_tests) == len(focused_test_members)
        and set(expected_focused_tests) == required_focused_test_paths,
        "v12 focused sensitivity test paths differ",
    )
    for path, expected in expected_focused_tests.items():
        current = file_by_path.get(path)
        _semantic_require(
            isinstance(current, Mapping)
            and current.get("sha256") == expected.get("sha256"),
            f"v12 focused sensitivity test identity differs: {path}",
        )

    authority = documents["authority_inventory.json"]
    authority_records = authority.get("records")
    _semantic_require(
        isinstance(authority_records, list), "v12 authority records are missing"
    )
    assert isinstance(authority_records, list)
    authority_roles = {
        str(item.get("role")): item
        for item in authority_records
        if isinstance(item, Mapping)
    }
    terminal_authority = authority_roles.get(
        "stage13_public_interest_last_tradable_close_liquidation_policy"
    )
    authority_sha256 = sha256_bytes(canonical_json_bytes(authority_records))
    _semantic_require(
        len(authority_roles) == len(authority_records) == 50
        and authority.get("authority_count") == 50
        and authority.get("authority_inventory_sha256") == authority_sha256
        and bindings.get("authority_inventory_sha256") == authority_sha256
        and isinstance(terminal_authority, Mapping)
        and terminal_authority.get("sha256") == terminal_sha256
        and terminal_authority.get("status")
        == "post_unblind_economic_data_gap_treatment"
        and authority.get("raw_payload_files_opened") == 0
        and authority.get("final_period_payload_rows_read") == 0,
        "v12 authority inventory reconciliation differs",
    )

    checks = documents["freeze_checks.json"]
    check_values = checks.get("checks")
    consolidation_receipt = checks.get("stage12_v12_completion_consolidation_receipt")
    report_template_receipts = checks.get("report_template_receipts")
    authorizations = checks.get("authorizations")
    _semantic_require(
        isinstance(check_values, Mapping)
        and isinstance(consolidation_receipt, Mapping)
        and isinstance(report_template_receipts, list)
        and isinstance(authorizations, Mapping),
        "v12 freeze checks are incomplete",
    )
    assert isinstance(check_values, Mapping)
    assert isinstance(consolidation_receipt, Mapping)
    assert isinstance(report_template_receipts, list)
    assert isinstance(authorizations, Mapping)
    required_v12_checks = {
        "stage12_v12_completion_consolidation_authenticated",
        "unpublished_v11_recorded_as_non_authoritative_and_superseded",
        "substantive_2021_terminal_event_treatment_authenticated",
        "terminal_vendor_missing_delisting_return_preserved",
        "terminal_primary_cash_retirement_multiplier_one_authenticated",
        "terminal_counterfactual_sensitivity_contract_authenticated",
        "terminal_sensitivity_implementation_and_focused_tests_bound",
        "final_input_registry_1210_column_contract_authenticated",
        "historical_stage13_v1_report_template_preserved",
        "active_stage13_v2_report_template_authenticated",
        "stage13_empirical_result_assembler_v3_authenticated",
        "standing_stage13_completion_instruction_hash_authenticated",
        "fresh_v12_binding_specific_stage13_approval_required",
    }
    _semantic_require(
        all(check_values.get(name) is True for name in required_v12_checks)
        and consolidation_receipt.get("classification")
        == consolidation.get("classification")
        and consolidation_receipt.get("terminal_treatment_policy", {}).get("sha256")
        == terminal_sha256
        and consolidation_receipt.get("terminal_treatment_policy", {}).get(
            "substantive_economic"
        )
        is True
        and consolidation_receipt.get("standing_user_instruction_sha256")
        == standing_instruction_sha256
        and consolidation_receipt.get("final_input_registry_contract")
        == registry_contract
        and consolidation_receipt.get("report_template_contract")
        == report_template_contract
        and consolidation_receipt.get("result_assembly_contract")
        == result_assembly_contract
        and consolidation_receipt.get(
            "fresh_binding_specific_stage13_approval_required"
        )
        is True
        and consolidation_receipt.get("stage13_execution_authorized_now") is False
        and authorizations.get("substantive_terminal_event_treatment_frozen") is True
        and authorizations.get("stage13_execution_authorized_now") is False
        and authorizations.get(
            "binding_specific_stage13_receipt_may_be_issued_after_v12_verification"
        )
        is True
        and authorizations.get("additional_user_prompt_after_v12_verification_required")
        is False,
        "v12 substantive treatment freeze checks differ",
    )
    report_receipts_by_role = {
        str(item.get("role")): item
        for item in report_template_receipts
        if isinstance(item, Mapping)
    }
    historical_stage13_paths = report_template_contract.get("historical_stage13_paths")
    _semantic_require(
        isinstance(historical_stage13_paths, list)
        and len(historical_stage13_paths) == 1
        and len(report_receipts_by_role) == len(report_template_receipts) == 3
        and set(report_receipts_by_role)
        == {
            "historical_superseded_stage13",
            "active_stage13",
            "active_stage14",
        }
        and report_receipts_by_role["historical_superseded_stage13"].get(
            "relative_path"
        )
        == historical_stage13_paths[0]
        and report_receipts_by_role["historical_superseded_stage13"].get("active")
        is False
        and report_receipts_by_role["active_stage13"].get("relative_path")
        == report_template_contract.get("active_stage13_path")
        and report_receipts_by_role["active_stage13"].get("active") is True
        and report_receipts_by_role["active_stage14"].get("relative_path")
        == report_template_contract.get("active_stage14_path")
        and report_receipts_by_role["active_stage14"].get("active") is True,
        "v12 report-template role receipts differ",
    )
    for report_receipt in report_template_receipts:
        assert isinstance(report_receipt, Mapping)
        inventory_member = file_by_path.get(str(report_receipt.get("relative_path")))
        _semantic_require(
            isinstance(inventory_member, Mapping)
            and report_receipt.get("sha256") == inventory_member.get("sha256")
            and report_receipt.get("size_bytes") == inventory_member.get("size_bytes"),
            "v12 report-template receipt does not bind file inventory",
        )
    _semantic_require(
        bindings.get(
            "stage13_public_interest_last_tradable_close_liquidation_policy_sha256"
        )
        == terminal_sha256
        and bindings.get("unpublished_superseded_v11_policy_sha256")
        == "baa8ac787797a368dd84711603627e8aa3b5baf6cfd4b6c3ba022e27ab70dad4"
        and bindings.get("stage13_completion_standing_instruction_sha256")
        == standing_instruction_sha256
        and bindings.get("stage13_input_registry_total_arrow_column_count") == 1210
        and bindings.get("stage13_report_template_v1_historical_sha256")
        == report_receipts_by_role["historical_superseded_stage13"].get("sha256")
        and bindings.get("stage13_report_template_v2_active_sha256")
        == report_receipts_by_role["active_stage13"].get("sha256")
        and bindings.get("stage14_report_template_v1_active_sha256")
        == report_receipts_by_role["active_stage14"].get("sha256")
        and bindings.get("stage13_result_assembler_id")
        == result_assembly_contract.get("assembler_id")
        == "stage13-empirical-result-assembler-v3"
        and bindings.get("replacement_inventory_delta_sha256")
        == receipts["replacement_inventory_delta.json"]["sha256"],
        "v12 manifest consolidation bindings differ",
    )

    tests = documents["test_receipts.json"]
    test_nodes = tests.get("nodes")
    _semantic_require(
        isinstance(test_nodes, list)
        and len(test_nodes) == 2
        and all(item.get("status") == "PASSED" for item in test_nodes)
        and all(
            item.get("access_audit", {}).get("status") == "NO_FORBIDDEN_PAYLOAD_ACCESS"
            for item in test_nodes
        ),
        "v12 test receipt graph differs",
    )
    ledger = documents["trial_ledger.json"]
    rows = ledger.get("rows")
    _semantic_require(isinstance(rows, list), "v12 trial ledger rows are missing")
    assert isinstance(rows, list)
    raw_count = sum(
        1
        for row in rows
        if isinstance(row, Mapping) and row.get("counts_in_raw_dsr") is True
    )
    _semantic_require(
        ledger.get("raw_trial_count") == raw_count == 386
        and ledger.get("ledger_sha256") == sha256_bytes(canonical_json_bytes(rows)),
        "v12 trial ledger reconciliation differs",
    )


def replay_primary_stage12(
    *,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
) -> PrimaryReplay:
    """Recompute every primary byte from current frozen inputs."""

    root = project_root.absolute()
    policy, policy_sha = _load_policy(policy_path, project_root=root)
    try:
        pointer_path = confined_relative_path(
            root,
            policy["outputs"]["latest_primary_pointer"],
            file_required=True,
            label="primary Stage 12 pointer",
        )
        pointer, pointer_file = load_json_object(
            pointer_path,
            trusted_root=root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("primary Stage 12 pointer path is unsafe") from exc
    expected_pointer_keys = {
        "schema_version",
        "stage",
        "freeze_id",
        "status",
        "bundle_path",
        "content_fingerprint",
        "manifest_sha256",
        "success_sha256",
        "closed_member_count",
        "authorizations",
    }
    expected_authorizations = {
        "independent_stage12_verification_authorized": True,
        "stage13_technical_gate_passed": True,
        "stage13_requires_separate_user_approval": True,
        "stage13_execution_authorized_now": False,
        "final_period_access_authorized_now": False,
        "post_unblind_technical_amendment_authorized": True,
        "post_unblind_scratch_topology_amendment_authorized": True,
        "post_unblind_zero_row_merge_provenance_amendment_authorized": True,
        "post_unblind_december_market_input_scope_partition_amendment_authorized": True,
        "post_unblind_entry_market_canonicalization_scope_alignment_amendment_authorized": True,
        "post_unblind_accounting_missing_execution_cost_transport_amendment_authorized": True,
        "post_unblind_logical_fingerprint_transport_amendment_authorized": True,
        "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_authorized": True,
        "post_unblind_equal_weight_missing_cost_gate_amendment_authorized": True,
        "substantive_terminal_event_treatment_frozen": True,
        "standing_stage13_approval_recorded": True,
        "binding_specific_stage13_receipt_may_be_issued_after_v12_verification": True,
        "additional_user_prompt_after_v12_verification_required": False,
        "prior_stage13_approval_valid_for_replacement_identity": False,
        "renewed_stage13_approval_required": True,
        "renewed_stage13_approval_granted_now": False,
        "stage13_source_checkpoint_context_migration_authorized": True,
        "stage13_decision_checkpoint_rebind_authorized": True,
        "checkpoint_rebind_metadata_correction_authorized": True,
        "stage13_parent_context_open_authorized_now": False,
        "stage13_blockers": list(_STAGE13_BLOCKERS),
    }
    if (
        set(pointer) != expected_pointer_keys
        or pointer.get("schema_version") != 1
        or pointer.get("stage") != 12
        or pointer.get("status") != policy["outputs"]["primary_status"]
        or pointer.get("closed_member_count") != len(PRIMARY_FILES)
        or pointer.get("authorizations") != expected_authorizations
    ):
        raise Stage12FreezeError("primary Stage 12 pointer boundary differs")
    expected_bundle_relative = (
        Path(str(policy["outputs"]["artifact_root"]))
        / str(policy["outputs"]["primary_namespace"])
        / str(pointer.get("freeze_id"))
    ).as_posix()
    if pointer.get("bundle_path") != expected_bundle_relative:
        raise Stage12FreezeError("primary Stage 12 bundle identity differs")
    try:
        bundle_path = confined_relative_path(
            root,
            expected_bundle_relative,
            file_required=False,
            label="primary Stage 12 bundle",
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("primary Stage 12 bundle path is unsafe") from exc
    names = {path.name for path in bundle_path.iterdir()}
    if names != set(PRIMARY_FILES):
        raise Stage12FreezeError("primary Stage 12 closed set differs")
    try:
        tests_path = confined_relative_path(
            bundle_path,
            "test_receipts.json",
            file_required=True,
            label="primary Stage 12 test-receipt member",
        )
        tests, _ = load_json_object(tests_path, trusted_root=bundle_path)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("primary Stage 12 member path is unsafe") from exc
    prepared = prepare_stage12_freeze(
        tests,
        policy_path=policy_path,
        project_root=root,
    )
    differences: list[dict[str, Any]] = []
    for name in sorted(PRIMARY_FILES):
        try:
            member = confined_relative_path(
                bundle_path,
                name,
                file_required=True,
                label="primary Stage 12 member",
            )
            observed = stable_file(
                member,
                keep_payload=True,
                trusted_root=bundle_path,
            )
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(
                f"primary Stage 12 member path is unsafe: {name}"
            ) from exc
        expected = prepared.document_bytes[name]
        expected_sha = sha256_bytes(expected)
        if observed.sha256 != expected_sha or observed.payload != expected:
            differences.append(
                {
                    "member": name,
                    "expected_sha256": expected_sha,
                    "observed_sha256": observed.sha256,
                    "expected_size_bytes": len(expected),
                    "observed_size_bytes": observed.size_bytes,
                }
            )
    if (
        pointer.get("freeze_id") != prepared.freeze_id
        or pointer.get("manifest_sha256")
        != prepared.document_receipts["pre_final_manifest.json"]["sha256"]
        or pointer.get("success_sha256")
        != prepared.document_receipts["_SUCCESS.json"]["sha256"]
    ):
        differences.append(
            {
                "member": "LATEST_STAGE_12_PRIMARY.json",
                "expected_sha256": "semantic_replay",
                "observed_sha256": pointer_file.sha256,
                "reason": "primary pointer semantic binding differs",
            }
        )
    _independently_validate_v12_primary_semantics(
        bundle_path=bundle_path,
        policy=policy,
        policy_sha256=policy_sha,
        pointer=pointer,
    )
    return PrimaryReplay(
        prepared=prepared,
        pointer=pointer,
        pointer_sha256=pointer_file.sha256,
        bundle_path=bundle_path,
        difference_count=len(differences),
        differences=tuple(differences),
    )


def _validate_independent_test_receipt(
    item: Mapping[str, Any],
    *,
    replay: PrimaryReplay,
    project_root: Path,
) -> dict[str, Any]:
    checkpoint_id = str(item.get("checkpoint_id"))
    relative = str(item.get("relative_path"))
    expected_relative = (
        Path(str(replay.prepared.policy["outputs"]["artifact_root"]))
        / str(replay.prepared.policy["outputs"]["checkpoint_namespace"])
        / checkpoint_id
    ).as_posix()
    if relative != expected_relative:
        raise Stage12FreezeError("independent test checkpoint namespace differs")
    try:
        path = confined_relative_path(
            project_root,
            relative,
            file_required=False,
            label="independent Stage 12 test checkpoint",
        )
        receipt_path = confined_relative_path(
            path,
            "receipt.json",
            file_required=True,
            label="independent Stage 12 test receipt",
        )
        success_path = confined_relative_path(
            path,
            "_SUCCESS.json",
            file_required=True,
            label="independent Stage 12 test success receipt",
        )
        receipt, receipt_file = load_json_object(
            receipt_path,
            trusted_root=project_root,
        )
        success, success_file = load_json_object(
            success_path,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("independent test checkpoint path is unsafe") from exc
    fingerprint = receipt.get("input_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise Stage12FreezeError("independent test fingerprint is missing")
    summary = receipt.get("summary")
    if not isinstance(summary, Mapping):
        raise Stage12FreezeError("independent test summary is missing")
    junit = receipt.get("junit_receipt")
    access_audit = receipt.get("access_audit")
    if not isinstance(junit, Mapping) or not isinstance(access_audit, Mapping):
        raise Stage12FreezeError("independent test evidence is missing")
    access_counters = access_audit.get("counters")
    if not isinstance(access_counters, Mapping):
        raise Stage12FreezeError("independent test access counters are missing")
    environment = build_environment_lock(project_root=project_root)
    environment_sha = sha256_bytes(canonical_json_bytes(environment))
    project_inventory = build_project_file_inventory(
        replay.prepared.policy, project_root=project_root
    )
    if (
        fingerprint.get("node_name") != "independent_stage12_focused_replay"
        or fingerprint.get("project_tree_sha256") != project_inventory["tree_sha256"]
        or fingerprint.get("environment_sha256") != environment_sha
        or fingerprint.get("environment_overrides") != {"PYTHONHASHSEED": "0"}
        or fingerprint.get("process_environment")
        != {**SANITIZED_ENVIRONMENT, "PYTHONHASHSEED": "0"}
        or fingerprint.get("command")
        != replay.prepared.policy["tests"]["focused_command"]
        or checkpoint_id != receipt.get("checkpoint_id")
        or not checkpoint_id.startswith(
            "stage12-checkpoint-independent_stage12_focused_replay-"
        )
        or item.get("receipt_sha256") != receipt_file.sha256
        or item.get("success_sha256") != success_file.sha256
        or item.get("summary") != summary
        or item.get("junit_receipt") != junit
        or item.get("access_audit") != access_audit
        or success.get("receipt_sha256") != receipt_file.sha256
        or success.get("checkpoint_id") != checkpoint_id
        or receipt.get("schema_version") != 2
        or receipt.get("status") != "PASSED"
        or success.get("status") != "PASSED"
        or receipt.get("input_fingerprint_sha256")
        != sha256_bytes(canonical_json_bytes(fingerprint))
        or int(summary.get("tests", 0)) <= 0
        or int(summary.get("failures", -1)) != 0
        or int(summary.get("errors", -1)) != 0
        or int(summary.get("skipped", -1)) != 0
        or junit.get("test_case_identity_sha256")
        != summary.get("test_case_identity_sha256")
        or access_audit.get("status") != "NO_FORBIDDEN_PAYLOAD_ACCESS"
        or any(
            int(access_counters.get(counter, -1)) != 0
            for counter in (
                "forbidden_access_attempts_denied",
                "process_spawn_attempts_denied",
                "raw_payload_files_opened",
                "interim_payload_files_opened",
                "processed_payload_files_opened",
                "desktop_payload_files_opened",
                "final_period_payload_files_opened",
                "final_period_rows_read",
            )
        )
    ):
        raise Stage12FreezeError("independent focused test receipt differs")
    dependencies = fingerprint.get("dependency_ids")
    primary_nodes = replay.prepared.documents["test_receipts.json"]["nodes"]
    expected_dependencies = sorted(str(node["checkpoint_id"]) for node in primary_nodes)
    if dependencies != expected_dependencies:
        raise Stage12FreezeError("independent test dependency chain differs")
    primary_summaries = [
        node.get("summary") for node in primary_nodes if isinstance(node, Mapping)
    ]
    if not primary_summaries or any(
        not isinstance(primary_summary, Mapping)
        or primary_summary.get("tests") != summary.get("tests")
        or primary_summary.get("passed") != summary.get("passed")
        or primary_summary.get("test_case_identity_sha256")
        != summary.get("test_case_identity_sha256")
        for primary_summary in primary_summaries
    ):
        raise Stage12FreezeError("independent test collection identity differs")
    return {
        "checkpoint_id": checkpoint_id,
        "relative_path": path.relative_to(project_root).as_posix(),
        "receipt_sha256": receipt_file.sha256,
        "success_sha256": success_file.sha256,
        "input_fingerprint_sha256": receipt["input_fingerprint_sha256"],
        "status": "PASSED",
        "summary": dict(summary),
        "junit_receipt": dict(junit),
        "access_audit": dict(access_audit),
    }


def _write_member(
    stage: SecureStagingDirectory,
    name: str,
    payload: bytes,
) -> None:
    with stage.open_exclusive_binary(name) as stream:
        written = stream.write(payload)
        if written != len(payload):
            raise Stage12FreezeError(f"short verification write: {name}")


def _authenticate_verification_bundle(
    *,
    bundle_path: Path,
    trusted_root: Path,
    expected_hashes: Mapping[str, str],
    replay: PrimaryReplay,
    verification_id: str,
) -> None:
    """Authenticate the closed verification bundle and its semantic graph."""

    try:
        bundle = confined_existing_path(
            bundle_path,
            trusted_root=trusted_root,
            file_required=False,
            label="Stage 12 verification bundle",
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("verification bundle path is unsafe") from exc
    if {item.name for item in bundle.iterdir()} != set(VERIFICATION_FILES):
        raise Stage12FreezeError("verification bundle closed set differs")
    documents: dict[str, Mapping[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for name in VERIFICATION_FILES:
        try:
            member = confined_relative_path(
                bundle,
                name,
                file_required=True,
                label="Stage 12 verification member",
            )
            record = stable_file(
                member,
                keep_payload=True,
                trusted_root=bundle,
            )
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(
                f"verification member path is unsafe: {name}"
            ) from exc
        if record.sha256 != expected_hashes.get(name):
            raise Stage12FreezeError(f"verification member differs: {name}")
        assert record.payload is not None
        try:
            value = json.loads(record.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage12FreezeError(f"verification member is invalid: {name}") from exc
        if (
            not isinstance(value, Mapping)
            or canonical_json_bytes(value) != record.payload
        ):
            raise Stage12FreezeError(
                f"verification member is not a canonical object: {name}"
            )
        documents[name] = value
        receipts[name] = {
            "sha256": record.sha256,
            "size_bytes": record.size_bytes,
        }

    differences = documents["byte_differences.json"]
    report = documents["verification_report.json"]
    success = documents["_SUCCESS.json"]
    member_receipts = {
        name: receipt for name, receipt in receipts.items() if name != "_SUCCESS.json"
    }
    if (
        differences
        != {
            "schema_version": 1,
            "verification_id": verification_id,
            "difference_count": 0,
            "differences": [],
        }
        or success.get("schema_version") != 1
        or success.get("stage") != 12
        or success.get("verification_id") != verification_id
        or success.get("primary_freeze_id") != replay.prepared.freeze_id
        or success.get("status") != replay.prepared.policy["outputs"]["verified_status"]
        or success.get("difference_count") != 0
        or success.get("members") != member_receipts
    ):
        raise Stage12FreezeError("verification success graph differs")
    report_basis = {
        "schema_version": report.get("schema_version"),
        "stage": report.get("stage"),
        "primary_freeze_id": report.get("primary_freeze_id"),
        "primary_pointer_sha256": report.get("primary_pointer_sha256"),
        "primary_manifest_sha256": report.get("primary_manifest_sha256"),
        "primary_success_sha256": report.get("primary_success_sha256"),
        "independent_test_receipt": report.get("independent_test_receipt"),
        "difference_count": report.get("difference_count"),
    }
    expected_authorizations = {
        "stage13_technical_gate_passed": True,
        "stage13_requires_separate_user_approval": True,
        "stage13_execution_authorized_now": False,
        "final_period_access_authorized_now": False,
        "post_unblind_technical_amendment_authorized": True,
        "post_unblind_scratch_topology_amendment_authorized": True,
        "post_unblind_zero_row_merge_provenance_amendment_authorized": True,
        "post_unblind_december_market_input_scope_partition_amendment_authorized": True,
        "post_unblind_entry_market_canonicalization_scope_alignment_amendment_authorized": True,
        "post_unblind_accounting_missing_execution_cost_transport_amendment_authorized": True,
        "post_unblind_logical_fingerprint_transport_amendment_authorized": True,
        "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_authorized": True,
        "post_unblind_equal_weight_missing_cost_gate_amendment_authorized": True,
        "substantive_terminal_event_treatment_frozen": True,
        "standing_stage13_approval_recorded": True,
        "binding_specific_stage13_receipt_may_be_issued_after_v12_verification": True,
        "additional_user_prompt_after_v12_verification_required": False,
        "prior_stage13_approval_valid_for_replacement_identity": False,
        "renewed_stage13_approval_required": True,
        "renewed_stage13_approval_granted_now": False,
        "stage13_source_checkpoint_context_migration_authorized": True,
        "stage13_decision_checkpoint_rebind_authorized": True,
        "checkpoint_rebind_metadata_correction_authorized": True,
        "stage13_parent_context_open_authorized_now": False,
        "stage13_blockers": list(_STAGE13_BLOCKERS),
    }
    expected_counters = {
        "counter_scope": (
            "guarded_stage12_independent_verification_and_primary_byte_replay_only"
        ),
        "raw_payload_files_opened": 0,
        "parent_checkpoint_payload_files_opened": 0,
        "final_period_payload_files_opened": 0,
        "final_period_rows_read": 0,
        "stage13_outputs_opened": 0,
    }
    checks = report.get("checks")
    counters = report.get("counters")
    if (
        report_basis["schema_version"] != 1
        or report_basis["stage"] != 12
        or report_basis["primary_freeze_id"] != replay.prepared.freeze_id
        or report_basis["primary_pointer_sha256"] != replay.pointer_sha256
        or report_basis["primary_manifest_sha256"]
        != replay.prepared.document_receipts["pre_final_manifest.json"]["sha256"]
        or report_basis["primary_success_sha256"]
        != replay.prepared.document_receipts["_SUCCESS.json"]["sha256"]
        or report_basis["difference_count"] != 0
        or report.get("verification_id") != verification_id
        or report.get("verification_fingerprint")
        != sha256_bytes(canonical_json_bytes(report_basis))
        or report.get("status") != replay.prepared.policy["outputs"]["verified_status"]
        or report.get("authorizations") != expected_authorizations
        or not isinstance(checks, Mapping)
        or any(
            checks.get(name) is not True
            for name in (
                "primary_closed_set_exact",
                "all_primary_members_recomputed_byte_identical",
                "all_primary_member_hashes_match",
                "primary_pointer_authenticated",
                "primary_semantic_graph_independently_reconciled",
                "authority_inventory_reauthenticated",
                "project_tree_rehashed",
                "replacement_inventory_delta_reauthenticated",
                "environment_reauthenticated",
                "trial_ledger_rebuilt",
                "source_transition_reauthenticated",
                "focused_protocol_independently_replayed",
                "independent_synthetic_regression_reauthenticated",
                "pre_unblind_calibration_scope_amendment_reauthenticated",
                "post_unblind_technical_amendment_reauthenticated",
                "post_unblind_hash_audit_incident_reauthenticated",
                "post_unblind_adapter_transition_reauthenticated",
                "post_unblind_zero_outcome_boundary_reauthenticated",
                "post_unblind_scratch_topology_amendment_reauthenticated",
                "post_unblind_scratch_topology_exact_and_non_mutating",
                "post_unblind_scratch_topology_zero_payload_boundary_reauthenticated",
                "post_unblind_zero_row_merge_provenance_amendment_reauthenticated",
                "post_unblind_zero_row_merge_provenance_exact_scope_reauthenticated",
                "post_unblind_zero_row_merge_provenance_non_economic_boundary_reauthenticated",
                "stage13_source_checkpoint_context_migration_reauthenticated",
                "stage13_parent_checkpoint_migration_contract_reauthenticated",
                "post_unblind_december_market_input_scope_partition_amendment_reauthenticated",
                "prior_unguarded_empirical_regression_access_reauthenticated",
                "december_exact_entry_market_projection_scope_reauthenticated",
                "decision_source_preflight_and_completeness_gate_reauthenticated",
                "stage13_decision_checkpoint_rebind_reauthenticated",
                "stage13_decision_checkpoint_rebind_431_checkpoint_pins_reauthenticated",
                "stage13_preserved_decision_checkpoint_cache_reauthentication_contract_reauthenticated",
                "stage13_decision_checkpoint_rebind_stage12_synthetic_only",
                "checkpoint_rebind_metadata_exception_reauthenticated",
                "checkpoint_rebind_metadata_source_transition_reauthenticated",
                "post_unblind_entry_market_canonicalization_scope_alignment_amendment_reauthenticated",
                "entry_market_exact_duplicate_canonicalization_contract_reauthenticated",
                "entry_market_closed_tombstone_share_coverage_contract_reauthenticated",
                "entry_market_projection_truthful_receipt_algebra_reauthenticated",
                "entry_market_production_preflight_disclosure_reauthenticated",
                "entry_market_no_third_projection_blocker_reauthenticated",
                "entry_market_standing_stage13_approval_reauthenticated",
                "entry_market_runtime_source_transition_reauthenticated",
                "post_unblind_accounting_missing_execution_cost_transport_amendment_reauthenticated",
                "accounting_missing_numeric_transport_preserves_nontradable_semantics_reauthenticated",
                "accounting_frozen_primary_borrow_mapping_reauthenticated",
                "accounting_prewrite_nonnullable_audit_reauthenticated",
                "accounting_transport_non_economic_boundary_reauthenticated",
                "thesis_error_materiality_policy_reauthenticated",
                "post_unblind_logical_fingerprint_transport_amendment_reauthenticated",
                "logical_fingerprint_schema_and_shape_reauthenticated",
                "logical_fingerprint_null_masks_and_finite_bits_reauthenticated",
                "logical_fingerprint_nan_payload_canonicalization_only_reauthenticated",
                "logical_fingerprint_persisted_authentication_unchanged_reauthenticated",
                "logical_fingerprint_non_economic_boundary_reauthenticated",
                "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_reauthenticated",
                "factor_covariance_zero_based_contract_reauthenticated",
                "factor_covariance_numeric_bits_unchanged_reauthenticated",
                "factor_covariance_matrix_semantics_unchanged_reauthenticated",
                "factor_covariance_non_economic_boundary_reauthenticated",
                "post_unblind_equal_weight_missing_cost_gate_amendment_reauthenticated",
                "equal_weight_new_position_execution_gate_reauthenticated",
                "equal_weight_current_risk_input_gate_reauthenticated",
                "equal_weight_required_exit_exclusion_reauthenticated",
                "equal_weight_missing_cost_inherited_exact_hold_reauthenticated",
                "equal_weight_frozen_hold_beta_fallback_boundary_reauthenticated",
                "equal_weight_exact_zero_trade_and_frozen_gross_failure_reauthenticated",
                "equal_weight_residual_gross_allocation_reauthenticated",
                "equal_weight_invalid_input_zero_capacity_and_nonzero_trade_fail_closed_reauthenticated",
                "equal_weight_descriptive_candidate_audit_reauthenticated",
                "equal_weight_rule_restoring_economic_boundary_reauthenticated",
                "v10_model_state_direct_reuse_prohibition_reauthenticated",
                "fresh_v12_six_model_fits_requirement_reauthenticated",
                "stage12_v12_completion_consolidation_reauthenticated",
                "unpublished_v11_non_authoritative_disposition_reauthenticated",
                "substantive_2021_terminal_event_treatment_reauthenticated",
                "terminal_sensitivity_contract_reauthenticated",
                "standing_stage13_completion_instruction_reauthenticated",
                "fresh_v12_binding_specific_stage13_approval_requirement_reauthenticated",
                "certified_final_economic_kernel_available",
                "schema_bound_registration_materializer_available",
                "complete_final_input_arrow_schemas_available",
                "stage13_technical_gate_passed",
            )
        )
        or checks.get("final_period_payload_opened") is not False
        or counters != expected_counters
    ):
        raise Stage12FreezeError("verification report semantic graph differs")


def verify_and_publish_stage12(
    independent_test_receipt: Mapping[str, Any],
    *,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
) -> Stage12VerificationPublication:
    """Verify zero primary differences and publish the narrow verified pointer."""

    root = project_root.absolute()
    replay = replay_primary_stage12(
        policy_path=policy_path,
        project_root=root,
    )
    if replay.difference_count != 0:
        raise Stage12FreezeError(
            f"Stage 12 primary byte replay found {replay.difference_count} differences"
        )
    independent = _validate_independent_test_receipt(
        independent_test_receipt,
        replay=replay,
        project_root=root,
    )
    policy = replay.prepared.policy
    report_basis = {
        "schema_version": 1,
        "stage": 12,
        "primary_freeze_id": replay.prepared.freeze_id,
        "primary_pointer_sha256": replay.pointer_sha256,
        "primary_manifest_sha256": replay.prepared.document_receipts[
            "pre_final_manifest.json"
        ]["sha256"],
        "primary_success_sha256": replay.prepared.document_receipts["_SUCCESS.json"][
            "sha256"
        ],
        "independent_test_receipt": independent,
        "difference_count": 0,
    }
    verification_fingerprint = sha256_bytes(canonical_json_bytes(report_basis))
    verification_id = f"stage12-verify-{verification_fingerprint[:24]}"
    differences = {
        "schema_version": 1,
        "verification_id": verification_id,
        "difference_count": 0,
        "differences": [],
    }
    report = {
        **report_basis,
        "verification_id": verification_id,
        "verification_fingerprint": verification_fingerprint,
        "status": policy["outputs"]["verified_status"],
        "checks": {
            "primary_closed_set_exact": True,
            "all_primary_members_recomputed_byte_identical": True,
            "all_primary_member_hashes_match": True,
            "primary_pointer_authenticated": True,
            "primary_semantic_graph_independently_reconciled": True,
            "authority_inventory_reauthenticated": True,
            "project_tree_rehashed": True,
            "replacement_inventory_delta_reauthenticated": True,
            "environment_reauthenticated": True,
            "trial_ledger_rebuilt": True,
            "source_transition_reauthenticated": True,
            "focused_protocol_independently_replayed": True,
            "independent_synthetic_regression_reauthenticated": True,
            "pre_unblind_calibration_scope_amendment_reauthenticated": True,
            "post_unblind_technical_amendment_reauthenticated": True,
            "post_unblind_hash_audit_incident_reauthenticated": True,
            "post_unblind_adapter_transition_reauthenticated": True,
            "post_unblind_zero_outcome_boundary_reauthenticated": True,
            "post_unblind_scratch_topology_amendment_reauthenticated": True,
            "post_unblind_scratch_topology_exact_and_non_mutating": True,
            "post_unblind_scratch_topology_zero_payload_boundary_reauthenticated": True,
            "post_unblind_zero_row_merge_provenance_amendment_reauthenticated": True,
            "post_unblind_zero_row_merge_provenance_exact_scope_reauthenticated": True,
            "post_unblind_zero_row_merge_provenance_non_economic_boundary_reauthenticated": True,
            "stage13_source_checkpoint_context_migration_reauthenticated": True,
            "stage13_parent_checkpoint_migration_contract_reauthenticated": True,
            "post_unblind_december_market_input_scope_partition_amendment_reauthenticated": True,
            "prior_unguarded_empirical_regression_access_reauthenticated": True,
            "december_exact_entry_market_projection_scope_reauthenticated": True,
            "decision_source_preflight_and_completeness_gate_reauthenticated": True,
            "stage13_decision_checkpoint_rebind_reauthenticated": True,
            "stage13_decision_checkpoint_rebind_431_checkpoint_pins_reauthenticated": True,
            "stage13_preserved_decision_checkpoint_cache_reauthentication_contract_reauthenticated": True,
            "stage13_decision_checkpoint_rebind_stage12_synthetic_only": True,
            "checkpoint_rebind_metadata_exception_reauthenticated": True,
            "checkpoint_rebind_metadata_source_transition_reauthenticated": True,
            "post_unblind_entry_market_canonicalization_scope_alignment_amendment_reauthenticated": True,
            "entry_market_exact_duplicate_canonicalization_contract_reauthenticated": True,
            "entry_market_closed_tombstone_share_coverage_contract_reauthenticated": True,
            "entry_market_projection_truthful_receipt_algebra_reauthenticated": True,
            "entry_market_production_preflight_disclosure_reauthenticated": True,
            "entry_market_no_third_projection_blocker_reauthenticated": True,
            "entry_market_standing_stage13_approval_reauthenticated": True,
            "entry_market_runtime_source_transition_reauthenticated": True,
            "post_unblind_accounting_missing_execution_cost_transport_amendment_reauthenticated": True,
            "accounting_missing_numeric_transport_preserves_nontradable_semantics_reauthenticated": True,
            "accounting_frozen_primary_borrow_mapping_reauthenticated": True,
            "accounting_prewrite_nonnullable_audit_reauthenticated": True,
            "accounting_transport_non_economic_boundary_reauthenticated": True,
            "thesis_error_materiality_policy_reauthenticated": True,
            "post_unblind_logical_fingerprint_transport_amendment_reauthenticated": True,
            "logical_fingerprint_schema_and_shape_reauthenticated": True,
            "logical_fingerprint_null_masks_and_finite_bits_reauthenticated": True,
            "logical_fingerprint_nan_payload_canonicalization_only_reauthenticated": True,
            "logical_fingerprint_persisted_authentication_unchanged_reauthenticated": True,
            "logical_fingerprint_non_economic_boundary_reauthenticated": True,
            "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_reauthenticated": True,
            "factor_covariance_zero_based_contract_reauthenticated": True,
            "factor_covariance_numeric_bits_unchanged_reauthenticated": True,
            "factor_covariance_matrix_semantics_unchanged_reauthenticated": True,
            "factor_covariance_non_economic_boundary_reauthenticated": True,
            "post_unblind_equal_weight_missing_cost_gate_amendment_reauthenticated": True,
            "equal_weight_new_position_execution_gate_reauthenticated": True,
            "equal_weight_current_risk_input_gate_reauthenticated": True,
            "equal_weight_required_exit_exclusion_reauthenticated": True,
            "equal_weight_missing_cost_inherited_exact_hold_reauthenticated": True,
            "equal_weight_frozen_hold_beta_fallback_boundary_reauthenticated": True,
            "equal_weight_exact_zero_trade_and_frozen_gross_failure_reauthenticated": True,
            "equal_weight_residual_gross_allocation_reauthenticated": True,
            "equal_weight_invalid_input_zero_capacity_and_nonzero_trade_fail_closed_reauthenticated": True,
            "equal_weight_descriptive_candidate_audit_reauthenticated": True,
            "equal_weight_rule_restoring_economic_boundary_reauthenticated": True,
            "v10_model_state_direct_reuse_prohibition_reauthenticated": True,
            "fresh_v12_six_model_fits_requirement_reauthenticated": True,
            "stage12_v12_completion_consolidation_reauthenticated": True,
            "unpublished_v11_non_authoritative_disposition_reauthenticated": True,
            "substantive_2021_terminal_event_treatment_reauthenticated": True,
            "terminal_sensitivity_contract_reauthenticated": True,
            "standing_stage13_completion_instruction_reauthenticated": True,
            "fresh_v12_binding_specific_stage13_approval_requirement_reauthenticated": True,
            "certified_final_economic_kernel_available": True,
            "schema_bound_registration_materializer_available": True,
            "complete_final_input_arrow_schemas_available": True,
            "stage13_technical_gate_passed": True,
            "final_period_payload_opened": False,
        },
        "authorizations": {
            "stage13_technical_gate_passed": True,
            "stage13_requires_separate_user_approval": True,
            "stage13_execution_authorized_now": False,
            "final_period_access_authorized_now": False,
            "post_unblind_technical_amendment_authorized": True,
            "post_unblind_scratch_topology_amendment_authorized": True,
            "post_unblind_zero_row_merge_provenance_amendment_authorized": True,
            "post_unblind_december_market_input_scope_partition_amendment_authorized": True,
            "post_unblind_entry_market_canonicalization_scope_alignment_amendment_authorized": True,
            "post_unblind_accounting_missing_execution_cost_transport_amendment_authorized": True,
            "post_unblind_logical_fingerprint_transport_amendment_authorized": True,
            "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_authorized": True,
            "post_unblind_equal_weight_missing_cost_gate_amendment_authorized": True,
            "substantive_terminal_event_treatment_frozen": True,
            "standing_stage13_approval_recorded": True,
            "binding_specific_stage13_receipt_may_be_issued_after_v12_verification": True,
            "additional_user_prompt_after_v12_verification_required": False,
            "prior_stage13_approval_valid_for_replacement_identity": False,
            "renewed_stage13_approval_required": True,
            "renewed_stage13_approval_granted_now": False,
            "stage13_source_checkpoint_context_migration_authorized": True,
            "stage13_decision_checkpoint_rebind_authorized": True,
            "checkpoint_rebind_metadata_correction_authorized": True,
            "stage13_parent_context_open_authorized_now": False,
            "stage13_blockers": list(_STAGE13_BLOCKERS),
        },
        "counters": {
            "counter_scope": (
                "guarded_stage12_independent_verification_and_primary_byte_replay_only"
            ),
            "raw_payload_files_opened": 0,
            "parent_checkpoint_payload_files_opened": 0,
            "final_period_payload_files_opened": 0,
            "final_period_rows_read": 0,
            "stage13_outputs_opened": 0,
        },
    }
    payloads = {
        "byte_differences.json": canonical_json_bytes(differences),
        "verification_report.json": canonical_json_bytes(report),
    }
    receipts = {
        name: {"sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        for name, payload in payloads.items()
    }
    success = {
        "schema_version": 1,
        "stage": 12,
        "verification_id": verification_id,
        "primary_freeze_id": replay.prepared.freeze_id,
        "status": policy["outputs"]["verified_status"],
        "difference_count": 0,
        "members": receipts,
    }
    payloads["_SUCCESS.json"] = canonical_json_bytes(success)
    if json.loads(payloads["_SUCCESS.json"].decode("utf-8")) != success:
        raise Stage12FreezeError("verification success byte identity differs")
    hashes = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    destination = (
        root
        / str(policy["outputs"]["artifact_root"])
        / str(policy["outputs"]["verification_namespace"])
        / verification_id
    )

    def primary_gate() -> None:
        current = replay_primary_stage12(
            policy_path=policy_path,
            project_root=root,
        )
        if (
            current.difference_count != 0
            or current.prepared.freeze_id != replay.prepared.freeze_id
            or current.pointer_sha256 != replay.pointer_sha256
        ):
            raise Stage12FreezeError("Stage 12 verification gate replay differs")

    def published_verification_gate() -> None:
        primary_gate()
        _authenticate_verification_bundle(
            bundle_path=destination,
            trusted_root=root,
            expected_hashes=hashes,
            replay=replay,
            verification_id=verification_id,
        )

    with SecureStagingDirectory.create(
        trusted_root=root,
        destination=destination,
        expected_files=VERIFICATION_FILES,
        label="Stage 12 independent verification",
    ) as stage:
        for name in sorted(VERIFICATION_FILES):
            _write_member(stage, name, payloads[name])
        snapshot = stage.snapshot(expected_hashes=hashes)
        published = stage.publish(
            expected_hashes=snapshot.hashes,
            pre_publish_gate=primary_gate,
            post_publish_gate=published_verification_gate,
        )
    _authenticate_verification_bundle(
        bundle_path=published.path,
        trusted_root=root,
        expected_hashes=hashes,
        replay=replay,
        verification_id=verification_id,
    )
    pointer = {
        "schema_version": 1,
        "stage": 12,
        "verification_id": verification_id,
        "status": policy["outputs"]["verified_status"],
        "primary_freeze_id": replay.prepared.freeze_id,
        "primary_pointer_path": Path(
            str(policy["outputs"]["latest_primary_pointer"])
        ).as_posix(),
        "primary_pointer_sha256": replay.pointer_sha256,
        "primary_manifest_sha256": replay.prepared.document_receipts[
            "pre_final_manifest.json"
        ]["sha256"],
        "verification_bundle_path": published.path.relative_to(root).as_posix(),
        "verification_report_sha256": hashes["verification_report.json"],
        "verification_success_sha256": hashes["_SUCCESS.json"],
        "difference_count": 0,
        "authorizations": report["authorizations"],
    }
    pointer_bytes = canonical_json_bytes(pointer)
    pointer_path = root / str(policy["outputs"]["latest_verified_pointer"])
    replace_pointer_last(
        pointer_path,
        pointer_bytes,
        trusted_root=root,
        pre_replace_gate=published_verification_gate,
        label="Stage 12 verified authority pointer",
    )
    pointer_file = stable_file(pointer_path, trusted_root=root)
    if pointer_file.sha256 != sha256_bytes(pointer_bytes):
        raise Stage12FreezeError("Stage 12 verified pointer differs")
    published_verification_gate()
    return Stage12VerificationPublication(
        verification_id=verification_id,
        bundle=published,
        pointer_path=pointer_path,
        pointer_sha256=pointer_file.sha256,
        report_sha256=hashes["verification_report.json"],
    )


__all__ = [
    "PrimaryReplay",
    "Stage12VerificationPublication",
    "VERIFICATION_FILES",
    "replay_primary_stage12",
    "verify_and_publish_stage12",
]
