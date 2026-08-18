"""Independent manifest and payload verification for Stage 11 validation data.

The verifier does not import or call the candidate materializer.  It
reauthenticates the frozen contract, every candidate byte payload, every
Parquet footer and the exact signal-date domain.  It never opens raw sources,
labels, outcomes, final-period data, weights, trades, or P&L.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.validation.validation_data_contract import (
    DEFAULT_POLICY,
    INTERNAL_CONTEXT_SIGNAL,
    PROJECT_ROOT,
    canonical_json_bytes,
    schema_sha256,
    sha256_file,
    validate_validation_data_contract,
)
from portfolio_bot.validation.validation_data_materialization import (
    materialization_code_hashes,
)
from portfolio_bot.validation.validation_development_parity import (
    PARITY_METHOD,
    PARITY_STATUS,
    prove_development_code_identity_parity,
)
from portfolio_bot.validation.validation_feature_semantics import (
    audit_raw_lineage_semantics,
)
from portfolio_bot.validation.validation_empirical_feature_stream import (
    load_and_reauthenticate_empirical_candidate_build,
)


class ValidationDataVerificationError(RuntimeError):
    """Raised when independent validation-data verification fails closed."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationDataVerificationError(message)


@dataclass(frozen=True)
class IndependentVerificationResult:
    build_id: str
    verification_id: str
    manifest_sha256: str
    materialization_code_hashes_sha256: str
    payload_count: int
    payload_rows: int
    difference_count: int
    status: str
    report: Mapping[str, Any]


def _canonical_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    _check(path.is_file() and not path.is_symlink(), "Unsafe validation-data manifest")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationDataVerificationError("Invalid validation-data manifest JSON") from exc
    _check(isinstance(value, dict), "Validation-data manifest is not an object")
    _check(payload == canonical_json_bytes(value), "Validation-data manifest is not canonical JSON")
    return value, payload


def _safe_payload(root: Path, filename: str) -> Path:
    _check(Path(filename).name == filename, "Unsafe manifest payload path")
    path = root / filename
    _check(path.is_file() and not path.is_symlink(), f"Missing payload: {filename}")
    _check(path.resolve(strict=True).parent == root, f"Payload escaped candidate root: {filename}")
    return path


def _signal_domain(path: Path) -> tuple[dt.date, ...]:
    parquet = pq.ParquetFile(path)
    _check("signal_date" in parquet.schema_arrow.names, "Payload has no signal_date")
    observed: set[dt.date] = set()
    for batch in parquet.iter_batches(
        batch_size=262_144,
        columns=["signal_date"],
        use_threads=False,
    ):
        column = batch.column(0)
        _check(column.null_count == 0, "Payload has a null signal date")
        observed.update(value.as_py() for value in pc.unique(column))
    _check(all(isinstance(value, dt.date) for value in observed), "Invalid signal-date value")
    return tuple(sorted(observed))


def independently_verify_candidate_manifest(
    manifest_path: Path,
    candidate_root: Path,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> IndependentVerificationResult:
    """Independently verify a fully scanned candidate's canonical manifest."""

    contract = validate_validation_data_contract(policy_path)
    root = Path(os.path.abspath(candidate_root.expanduser()))
    _check(root.is_dir() and not root.is_symlink(), "Unsafe candidate root")
    _check(root.resolve(strict=True) == root, "Candidate root uses a symlink")
    manifest, manifest_bytes = _canonical_manifest(manifest_path)

    expected_top = {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "status",
        "bindings",
        "scope",
        "payloads",
        "development_parity",
        "empirical_build",
        "access_counters",
        "authorizations",
    }
    _check(set(manifest) == expected_top, "Validation-data manifest keys differ")
    _check(manifest["schema_version"] == 1, "Manifest schema differs")
    _check(manifest["contract_id"] == contract.policy["contract_id"], "Manifest contract differs")
    _check(manifest["policy_id"] == contract.policy["policy_id"], "Manifest policy differs")
    _check(manifest["status"] == "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION", "Manifest status differs")
    bindings = manifest["bindings"]
    _check(
        isinstance(bindings, dict)
        and set(bindings)
        == {
            "policy_sha256",
            "frozen_bindings",
            "code_hashes",
            "content_fingerprint",
        },
        "Manifest binding keys differ",
    )
    _check(bindings["policy_sha256"] == contract.policy_sha256, "Manifest policy hash differs")
    _check(bindings["frozen_bindings"] == dict(contract.frozen_bindings), "Manifest frozen bindings differ")
    observed_code_hashes = dict(materialization_code_hashes())
    _check(
        bindings["code_hashes"] == observed_code_hashes,
        "Manifest materialization code hashes differ",
    )

    scope = manifest["scope"]
    _check(scope["internal_context_signal_date"] == INTERNAL_CONTEXT_SIGNAL.isoformat(), "Manifest context signal differs")
    _check(scope["internal_context_rows_published"] == 0, "Context signal rows were published")
    _check(scope["signal_dates"] == len(contract.scored_signal_dates), "Manifest signal count differs")
    _check(scope["first_signal_date"] == contract.scored_signal_dates[0].isoformat(), "Manifest first signal differs")
    _check(scope["last_signal_date"] == contract.scored_signal_dates[-1].isoformat(), "Manifest last signal differs")
    _check(scope["economic_features"] == len(contract.feature_names), "Manifest feature count differs")
    _check(scope["frozen_category_vocabulary_reused"] is True, "Frozen category vocabulary was not reused")
    _check(scope["validation_vocabulary_expansions"] == 0, "Validation expanded the frozen vocabulary")
    _check(scope["frozen_model_support_changes"] == 0, "Validation changed frozen model support")
    _check(scope["inactive_segment_nonmissing_values"] == 0, "Frozen-inactive segment features were populated")

    parity = manifest["development_parity"]
    _check(
        isinstance(parity, dict)
        and parity.get("status") == PARITY_STATUS
        and parity.get("method") == PARITY_METHOD
        and parity.get("identity_mismatch_count") == 0
        and parity.get("empirical_difference_count") == 0
        and parity.get("empirical_replay_feature_rows")
        == int(contract.rules["development_parity"]["reference_feature_rows"])
        and parity.get("empirical_replay_lineage_rows")
        == int(contract.rules["development_parity"]["reference_lineage_rows"])
        and parity.get("evidence", {})
        .get("full_development_logical_replay", {})
        .get("difference_count")
        == 0
        and parity.get("evidence", {})
        .get("actual_stage11_orchestration_canary", {})
        .get("difference_count")
        == 0,
        "Development full replay parity differs",
    )
    independently_observed_parity = prove_development_code_identity_parity(
        contract
    ).receipt.as_dict()
    _check(
        parity == independently_observed_parity,
        "Development parity evidence did not independently reproduce",
    )
    empirical_binding = manifest["empirical_build"]
    empirical_keys = {
        "build_id",
        "content_fingerprint",
        "receipt_path",
        "receipt_sha256",
        "feature_input_receipt_sha256",
        "feature_input_bundle_sha256",
        "feature_input_adapter_code_sha256",
        "scored_build_id",
        "scored_content_fingerprint",
        "optionmetrics_working_build_id",
        "optionmetrics_working_content_fingerprint",
        "crsp_source_id",
        "crsp_database_sha256",
        "output_receipts_sha256",
        "materialization_code_hashes_sha256",
    }
    _check(
        isinstance(empirical_binding, dict)
        and set(empirical_binding) == empirical_keys,
        "Manifest empirical-build binding keys differ",
    )
    relative_receipt = Path(str(empirical_binding["receipt_path"]))
    _check(
        not relative_receipt.is_absolute()
        and ".." not in relative_receipt.parts
        and relative_receipt.parts[:2]
        == ("artifacts", "phase_05_validation"),
        "Manifest empirical-build receipt path is unsafe",
    )
    empirical_authority = load_and_reauthenticate_empirical_candidate_build(
        PROJECT_ROOT / relative_receipt,
        contract,
    )
    empirical_document = empirical_authority.receipt.as_dict()
    expected_empirical_binding = {
        "build_id": empirical_document["build_id"],
        "content_fingerprint": empirical_document["content_fingerprint"],
        "receipt_path": relative_receipt.as_posix(),
        "receipt_sha256": empirical_authority.receipt_sha256,
        "feature_input_receipt_sha256": empirical_document["feature_input"][
            "receipt_sha256"
        ],
        "feature_input_bundle_sha256": empirical_document["feature_input"][
            "input_bundle_sha256"
        ],
        "feature_input_adapter_code_sha256": empirical_document["feature_input"][
            "adapter_code_sha256"
        ],
        "scored_build_id": empirical_document["feature_identities"][
            "scored_build_id"
        ],
        "scored_content_fingerprint": empirical_document["feature_identities"][
            "scored_content_fingerprint"
        ],
        "optionmetrics_working_build_id": empirical_document[
            "feature_identities"
        ]["optionmetrics_working_build_id"],
        "optionmetrics_working_content_fingerprint": empirical_document[
            "feature_identities"
        ]["optionmetrics_working_content_fingerprint"],
        "crsp_source_id": empirical_document["crsp_router"]["source_id"],
        "crsp_database_sha256": empirical_document["crsp_router"][
            "database_sha256"
        ],
        "output_receipts_sha256": empirical_document[
            "output_receipts_sha256"
        ],
        "materialization_code_hashes_sha256": empirical_document[
            "code_identity"
        ]["end_hashes_sha256"],
    }
    _check(
        empirical_authority.candidate_root == root
        and empirical_binding == expected_empirical_binding,
        "Manifest empirical-build binding did not independently reproduce",
    )
    _check(
        all(value == 0 for value in manifest["access_counters"].values()),
        "Manifest reports a prohibited read or empirical output",
    )
    authorizations = manifest["authorizations"]
    _check(authorizations.get("independent_verification_completed") is False, "Primary manifest self-authorized verification")
    _check(authorizations.get("validation_predictor_risk_and_cost_use_authorized") is False, "Primary manifest self-authorized use")
    _check(
        all(
            authorizations.get(key) is False
            for key in (
                "validation_outcome_access_authorized",
                "real_validation_weights_trades_or_pnl_authorized",
                "final_period_access_authorized",
            )
        ),
        "Primary manifest reopened a prohibited authorization",
    )

    declared = manifest["payloads"]
    _check(isinstance(declared, dict), "Manifest payloads are not an object")
    _check(set(declared) == set(contract.payload_specs), "Manifest payload role set differs")
    _check(
        {item.name for item in root.iterdir()} == {
            specification.filename for specification in contract.payload_specs.values()
        },
        "Candidate payload directory is not closed",
    )

    receipt_payload: list[dict[str, Any]] = []
    total_rows = 0
    expected_dates = contract.scored_signal_dates
    for role in sorted(contract.payload_specs):
        specification = contract.payload_specs[role]
        receipt = declared[role]
        _check(receipt.get("role") == role, f"Payload receipt role differs: {role}")
        _check(receipt.get("path") == specification.filename, f"Payload filename differs: {role}")
        path = _safe_payload(root, specification.filename)
        parquet = pq.ParquetFile(path)
        _check(
            parquet.schema_arrow.equals(specification.schema, check_metadata=True),
            f"Payload schema differs: {role}",
        )
        independently_observed = {
            "role": role,
            "path": specification.filename,
            "sha256": sha256_file(path),
            "size_bytes": path.stat(follow_symlinks=False).st_size,
            "rows": int(parquet.metadata.num_rows),
            "row_groups": int(parquet.metadata.num_row_groups),
            "schema_sha256": schema_sha256(parquet.schema_arrow),
            "minimum_signal_date": expected_dates[0].isoformat(),
            "maximum_signal_date": expected_dates[-1].isoformat(),
        }
        _check(receipt == independently_observed, f"Payload receipt differs: {role}")
        _check(_signal_domain(path) == expected_dates, f"Payload signal domain differs: {role}")
        receipt_payload.append(independently_observed)
        total_rows += independently_observed["rows"]

    semantic_difference_count = 0
    predictor_path = _safe_payload(
        root,
        contract.payload_specs["validation_predictor_payload"].filename,
    )
    for family, feature_names in contract.family_feature_names.items():
        semantic = audit_raw_lineage_semantics(
            raw_path=predictor_path,
            lineage_path=_safe_payload(
                root,
                contract.payload_specs[
                    f"validation_feature_lineage_{family}"
                ].filename,
            ),
            feature_names=feature_names,
            allowed_signal_dates=contract.scored_signal_dates,
        )
        semantic_difference_count += semantic.difference_count
        _check(
            semantic.raw_rows == declared["validation_predictor_payload"]["rows"]
            and semantic.lineage_rows
            == declared[f"validation_feature_lineage_{family}"]["rows"],
            f"Independent raw/lineage semantic coverage differs: {family}",
        )
    _check(
        semantic_difference_count == 0,
        "Independent raw-value/lineage semantic consistency differs",
    )

    _check(
        declared["validation_predictor_payload"]["rows"] == scope["feature_rows"],
        "Manifest feature row count differs",
    )
    expected_lineage = scope["feature_rows"] * len(contract.feature_names)
    observed_lineage = sum(
        declared[f"validation_feature_lineage_{family}"]["rows"]
        for family in contract.family_feature_names
    )
    _check(observed_lineage == expected_lineage == scope["logical_lineage_rows"], "Manifest lineage arithmetic differs")
    _check(
        declared["validation_factor_covariance"]["rows"] == 59 * 36,
        "Manifest covariance row count differs",
    )
    _check(
        declared["validation_risk_cost_month_summary"]["rows"] == 59,
        "Manifest risk/cost summary row count differs",
    )

    fingerprint_payload = {
        "schema_version": 1,
        "contract_id": str(contract.policy["contract_id"]),
        "policy_sha256": contract.policy_sha256,
        "frozen_bindings": dict(contract.frozen_bindings),
        "code_hashes": observed_code_hashes,
        "payloads": receipt_payload,
        "development_parity": manifest["development_parity"],
        "empirical_build": manifest["empirical_build"],
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(fingerprint_payload)).hexdigest()
    _check(manifest["bindings"]["content_fingerprint"] == fingerprint, "Manifest fingerprint differs")
    _check(manifest["build_id"] == f"stage11-validation-data-{fingerprint[:24]}", "Manifest build ID differs")

    code_hashes_sha256 = hashlib.sha256(
        canonical_json_bytes(observed_code_hashes)
    ).hexdigest()
    verification_preimage = {
        "schema_version": 1,
        "contract_id": str(contract.policy["contract_id"]),
        "policy_id": str(contract.policy["policy_id"]),
        "build_id": manifest["build_id"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "materialization_code_hashes_sha256": code_hashes_sha256,
        "payload_count": len(declared),
        "payload_rows": total_rows,
        "difference_count": 0,
        "raw_lineage_semantic_difference_count": semantic_difference_count,
        "empirical_build_id": empirical_document["build_id"],
        "empirical_build_receipt_path": relative_receipt.as_posix(),
        "empirical_build_receipt_sha256": empirical_authority.receipt_sha256,
        "empirical_build_content_fingerprint": empirical_document[
            "content_fingerprint"
        ],
    }
    verification_id = "verify-" + hashlib.sha256(
        canonical_json_bytes(verification_preimage)
    ).hexdigest()[:16]
    report = MappingProxyType(
        {
            "schema_version": 1,
            "contract_id": str(contract.policy["contract_id"]),
            "policy_id": str(contract.policy["policy_id"]),
            "verification_id": verification_id,
            "status": "VERIFIED_PASSED",
            "build_id": manifest["build_id"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "materialization_code_hashes_sha256": code_hashes_sha256,
            "payload_count": len(declared),
            "payload_rows": total_rows,
            "signal_dates_checked_per_payload": len(expected_dates),
            "difference_count": 0,
            "raw_lineage_semantic_difference_count": semantic_difference_count,
            "empirical_build_id": empirical_document["build_id"],
            "empirical_build_receipt_path": relative_receipt.as_posix(),
            "empirical_build_receipt_sha256": empirical_authority.receipt_sha256,
            "empirical_build_content_fingerprint": empirical_document[
                "content_fingerprint"
            ],
            "validation_outcome_rows_read": 0,
            "final_period_rows_read": 0,
            "validation_predictor_risk_and_cost_use_authorized": True,
            "validation_outcome_access_authorized": False,
            "real_validation_weights_trades_or_pnl_authorized": False,
            "final_period_access_authorized": False,
        }
    )
    return IndependentVerificationResult(
        build_id=str(manifest["build_id"]),
        verification_id=verification_id,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        materialization_code_hashes_sha256=code_hashes_sha256,
        payload_count=len(declared),
        payload_rows=total_rows,
        difference_count=0,
        status="VERIFIED_PASSED",
        report=report,
    )


def write_independent_verification_report(
    result: IndependentVerificationResult,
    destination: Path,
) -> Path:
    """Persist the canonical authority consumed by the predictor adapter."""

    path = Path(os.path.abspath(destination.expanduser()))
    _check(path.parent.is_dir() and not path.parent.is_symlink(), "Unsafe verification-report directory")
    payload = canonical_json_bytes(dict(result.report))
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValidationDataVerificationError(
            "Verification-report destination already exists"
        ) from exc
    return path


__all__ = [
    "IndependentVerificationResult",
    "ValidationDataVerificationError",
    "independently_verify_candidate_manifest",
    "write_independent_verification_report",
]
