"""Authenticated, authority-bound recovery of the completed Stage 11 caches.

This module is deliberately outside ``src/portfolio_bot``.  The cached model
identities bind that source tree, so changing package source would invalidate
the 660 completed fits.  Recovery instead reauthenticates every checkpoint and
raw-forecast byte, reconstructs all fit identities, recomputes predictions
through the normal pseudo-live chronology, and uses the normal
portfolio/publication/verifier functions.

The only data-domain correction is explicit and local: development labels are
restricted to the exact Stage 9 OOF split membership before the existing
calibration adapter is called.  Two narrowly scoped worker/call-site bindings
are restored in ``finally`` blocks; no package source file is modified.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import portfolio_bot.allocation.rank as allocation_rank_module
import portfolio_bot.validation.empirical_portfolio_backend as portfolio_backend_module
import portfolio_bot.validation.empirical_selector_backend as selector_backend_module
import portfolio_bot.validation.execution_orchestrator as execution_orchestrator_module
from portfolio_bot.allocation.solver import (
    EMERGENCY_WORSENING_TOLERANCE,
    INDEPENDENT_ABSOLUTE_TOLERANCE,
)
from portfolio_bot.selectors.contract import validate_runtime_versions
from portfolio_bot.validation.development_calibration import (
    _LABEL_COLUMNS,
    _date_series,
    _observations,
    _selected_score_columns,
)
from portfolio_bot.validation.empirical_execution import (
    Stage11EmpiricalExecution,
    Stage11EmpiricalSelectorStage,
    stage11_execution_implementation_code_hashes,
)
from portfolio_bot.validation.empirical_portfolio_backend import (
    run_empirical_portfolio_accounting,
)
from portfolio_bot.validation.empirical_selector_backend import (
    EmpiricalStage11SelectorBackend,
    _model_code_sha256,
)
from portfolio_bot.validation.execution_input_manifest import (
    AuthenticatedStage11ExecutionInputs,
    authenticate_stage11_execution_inputs,
)
from portfolio_bot.validation.execution_orchestrator import (
    ELASTIC_NET_CHALLENGER_ID,
    EXPECTED_FITS_PER_REFIT,
    EXPECTED_REFIT_COUNT,
    EXPECTED_TOTAL_FITS,
    MonthlyExecutionStep,
    PRIMARY_STRATEGY_ID,
    Stage11ExecutionPlan,
    build_stage11_execution_plan,
    build_strategy_calibration_schedules,
    derive_empirical_locked_validation_selection,
    run_empirical_locked_walk_forward,
)
from portfolio_bot.validation.execution_publication import (
    MANIFEST_NAME,
    Stage11ExecutionArtifactSet,
    publish_stage11_execution,
    verify_stage11_execution,
)
from portfolio_bot.validation.execution_reporting import (
    REPORT_SUMMARY_NAME,
    publish_verified_stage11_execution_report,
)
from portfolio_bot.validation.locked_selection import LOCKED_SHORTLIST
from portfolio_bot.validation.outcome_materialization import (
    Stage11OutcomePublication,
    publish_from_authenticated_execution_inputs,
)
from portfolio_bot.validation.selector_runtime import (
    ForecastBatchReceipt,
    LoadedSelectorCheckpoint,
    array_sha256,
    canonical_json_bytes,
    load_selector_checkpoint,
    unpack_numeric_arrays,
    validate_complete_forecast_batches,
)
from portfolio_bot.validation.validation_runner import (
    require_live_stage11_validation_unblind_authority,
)
from portfolio_bot.validation.walk_forward import (
    EXPECTED_REBALANCE_CALENDAR_SHA256,
    AuthenticatedValidationCalendar,
)
from stage_11_allocation_numerical_adapter import (
    BOUNDED_DUAL_GAP_BAND,
    BOUNDED_DUAL_GAP_FRACTION,
    BOUNDED_DUAL_GAP_REFERENCE_TOLERANCE,
    BOUNDED_DUAL_STATIONARITY_MULTIPLIER,
    BOUNDED_FEASIBLE_DUAL_CERTIFICATE_SOURCE,
    DUAL_CERTIFICATE_AUDITABLE_STATUSES,
    DUAL_CERTIFICATE_GAP_TOLERANCE,
    DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE,
    DUAL_CERTIFICATE_RESIDUAL_TOLERANCE,
    DUAL_CERTIFICATE_STATIONARITY_FLOOR,
    DUAL_CERTIFIED_SOLVER_STATUS,
    DUAL_FAILURE_SNAPSHOT_ROOT,
    DUAL_RECONCILIATION_RETRY_WEIGHT,
    DUAL_STATIONARITY_REFINEMENT_TOLERANCE,
    EXPLICIT_DUAL_CERTIFICATE_SOURCE,
    RECONCILED_DUAL_CERTIFICATE_SOURCE,
    RETRY_DUAL_CERTIFICATE_SOURCE,
    TRADE_NUMERICAL_SCALE,
    solve_hybrid_stage11,
    solve_projection_stage11,
)

_CANONICAL_PORTFOLIO_CALIBRATION = (
    portfolio_backend_module.build_authenticated_development_calibration_observations
)
_CANONICAL_HYBRID_SOLVER = execution_orchestrator_module.solve_hybrid
_CANONICAL_RANK_PROJECTION_SOLVER = allocation_rank_module.solve_projection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "configs/phase_05_validation/stage_11_cache_recovery_v1.json"
)
NUMERICAL_AMENDMENT_PATH = (
    PROJECT_ROOT
    / "configs/phase_05_validation/stage_11_allocation_numerical_amendment_v1.json"
)
DECIMAL_SERIALIZATION_AMENDMENT_PATH = (
    PROJECT_ROOT
    / "configs/phase_05_validation/"
    "stage_11_decimal_serialization_amendment_v1.json"
)
_KEY_COLUMNS = ("signal_date", "holding_month", "permno")
_DATE_FIELDS = (
    "refit_signal_date",
    "signal_date",
    "entry_date",
    "exit_date",
    "predictor_availability_date_max",
)
_PRESENT = "PRESENT"
_HEX64 = frozenset("0123456789abcdef")


class Stage11CacheRecoveryError(RuntimeError):
    """Raised before a cached byte can influence the recovered execution."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11CacheRecoveryError(message)


def _canonical_json(value: object) -> bytes:
    """Canonical encoding used by frozen tree inventories (no trailing LF)."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_file_bytes(
    path: Path,
    *,
    label: str,
    expected_mode: int | None = None,
) -> bytes:
    before = path.stat(follow_symlinks=False)
    mode = stat.S_IMODE(before.st_mode)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink == 1
        and before.st_uid == os.getuid()
        and (mode & 0o022) == 0
        and (expected_mode is None or mode == expected_mode),
        f"{label} is not an ordinary single-link file",
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while block := os.read(descriptor, 8 << 20):
            chunks.append(block)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    after = path.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    _require(
        identity(before)
        == identity(opened_before)
        == identity(opened_after)
        == identity(after)
        and len(payload) == before.st_size,
        f"{label} changed while read",
    )
    return payload


def _require_safe_directory(
    path: Path,
    *,
    allowed_modes: frozenset[int] = frozenset({0o700, 0o755}),
    label: str,
) -> None:
    metadata = path.stat(follow_symlinks=False)
    mode = stat.S_IMODE(metadata.st_mode)
    _require(
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and mode in allowed_modes
        and (mode & 0o022) == 0,
        f"{label} is not a safe current-user directory",
    )


def _exact_keys(value: object, expected: set[str], *, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    _require(set(value) == expected, f"{label} fields differ")
    return value


def _sha(value: object, *, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX64),
        f"{label} is not a lowercase SHA-256",
    )
    return value


def _load_recovery_contract(path: Path = CONTRACT_PATH) -> Mapping[str, Any]:
    payload = _stable_file_bytes(path, label="Stage 11 recovery contract")
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11CacheRecoveryError("Stage 11 recovery contract is invalid") from exc
    contract = _exact_keys(
        parsed,
        {
            "schema_version",
            "contract_id",
            "status",
            "authority",
            "scope",
            "inventory_algorithm",
            "trees",
            "semantic_expectations",
            "outcome_tables",
        },
        label="Stage 11 recovery contract",
    )
    _require(
        contract["schema_version"] == 1
        and contract["contract_id"] == "phase05-stage11-cache-recovery-v1"
        and contract["status"] == "FROZEN_AUTHENTICATED_CACHE_RECOMPUTATION",
        "Stage 11 recovery contract identity differs",
    )
    authority = _exact_keys(
        contract["authority"],
        {
            "approval_contract_file_sha256",
            "approval_contract_payload_sha256",
            "execution_input_manifest_sha256",
            "predictor_payload_sha256",
            "cached_outcome_authority_build_id",
            "cached_outcome_authority_verification_id",
            "failure_receipt_path",
            "failure_receipt_sha256",
            "fresh_preflight_required",
        },
        label="recovery authority",
    )
    for name in (
        "approval_contract_file_sha256",
        "approval_contract_payload_sha256",
        "execution_input_manifest_sha256",
        "predictor_payload_sha256",
        "failure_receipt_sha256",
    ):
        _sha(authority[name], label=f"recovery authority {name}")
    failure_relative = Path(str(authority["failure_receipt_path"]))
    _require(
        not failure_relative.is_absolute()
        and ".." not in failure_relative.parts
        and failure_relative.as_posix() == authority["failure_receipt_path"],
        "recovery failure-receipt path is unsafe",
    )
    failure_path = PROJECT_ROOT.joinpath(*failure_relative.parts)
    _require(
        _sha256_bytes(
            _stable_file_bytes(failure_path, label="Stage 11 failure receipt")
        )
        == authority["failure_receipt_sha256"],
        "Stage 11 failure receipt digest differs",
    )
    _require(
        authority["fresh_preflight_required"] is True,
        "recovery must require a fresh preflight authority",
    )
    _require(
        contract["inventory_algorithm"]
        == {
            "algorithm_id": "sha256-canonical-json-file-inventory-v1",
            "record_fields": ["path", "sha256", "size_bytes"],
            "sort": "relative_posix_path_ascending",
            "json": "utf8_sort_keys_compact_no_trailing_newline",
        },
        "recovery inventory algorithm differs",
    )
    scope = contract["scope"]
    _require(
        scope
        == {
            "first_signal_date": "2014-12-31",
            "last_signal_date": "2019-10-31",
            "last_exit_date": "2019-12-02",
            "first_holding_month": "2015-01",
            "last_holding_month": "2019-11",
            "holding_month_count": 59,
            "first_prohibited_final_date": "2020-01-02",
        },
        "recovery date scope differs",
    )
    _require(
        set(contract["trees"])
        == {"selector_checkpoints", "raw_forecasts", "cached_outcomes"},
        "recovery tree set differs",
    )
    for name in ("selector_checkpoints", "raw_forecasts", "cached_outcomes"):
        tree = _exact_keys(
            contract["trees"][name],
            {
                "output_root_key",
                "directory_mode_counts",
                "file_mode",
                "file_count",
                "total_size_bytes",
                "inventory_sha256",
            },
            label=f"{name} tree",
        )
        _sha(tree["inventory_sha256"], label=f"{name} inventory")
        _require(
            isinstance(tree["directory_mode_counts"], Mapping)
            and bool(tree["directory_mode_counts"])
            and set(tree["directory_mode_counts"]).issubset({"0700", "0755"})
            and all(
                type(count) is int and count > 0
                for count in tree["directory_mode_counts"].values()
            )
            and tree["file_mode"] in {"0600", "0644"},
            f"{name} ownership/mode contract differs",
        )
        _require(
            type(tree["file_count"]) is int
            and tree["file_count"] > 0
            and type(tree["total_size_bytes"]) is int
            and tree["total_size_bytes"] > 0,
            f"{name} inventory dimensions differ",
        )
    expected = _exact_keys(
        contract["semantic_expectations"],
        {
            "recovery_mode",
            "checkpoint_count",
            "refit_count",
            "fits_per_refit",
            "forecast_receipt_count",
            "forecast_payload_count",
            "forecast_rows",
            "walk_forward_months",
            "development_oof_rows",
            "development_oof_present_rows",
            "development_oof_months",
            "materialized_validation_label_rows",
            "cached_forecast_independent_score_matrix_digest_present",
            "preexisting_cache_authenticated",
            "preexisting_cache_read",
            "used_as_prediction_authority",
            "normal_predictions_recomputed",
            "collision_byte_equality_required",
            "cache_authority_basis",
            "forecast_recomputation_policy",
            "model_code_sha256_by_family",
        },
        label="recovery semantic expectations",
    )
    _require(
        expected["recovery_mode"]
        == "authenticated_checkpoint_recovery_with_full_forecast_recomputation"
        and expected[
            "cached_forecast_independent_score_matrix_digest_present"
        ]
        is False
        and expected["preexisting_cache_authenticated"] is True
        and expected["preexisting_cache_read"] is True
        and expected["used_as_prediction_authority"] is False
        and expected["normal_predictions_recomputed"] is True
        and expected["collision_byte_equality_required"] is True
        and expected["checkpoint_count"] == EXPECTED_TOTAL_FITS == 660
        and expected["refit_count"] == EXPECTED_REFIT_COUNT == 20
        and expected["fits_per_refit"] == EXPECTED_FITS_PER_REFIT == 33
        and expected["forecast_receipt_count"] == 1_947
        and expected["forecast_payload_count"] == 1_947
        and expected["forecast_rows"] == 4_013_460
        and expected["development_oof_rows"] == 172_969
        and expected["development_oof_present_rows"] == 172_799
        and expected["development_oof_months"] == 95,
        "recovery semantic counts differ",
    )
    _require(
        set(expected["model_code_sha256_by_family"]) == set(LOCKED_SHORTLIST),
        "recovery model-code family set differs",
    )
    for family, digest in expected["model_code_sha256_by_family"].items():
        _sha(digest, label=f"{family} model code")
    _require(len(contract["outcome_tables"]) == 14, "outcome table receipt set differs")
    for name, receipt in contract["outcome_tables"].items():
        _exact_keys(
            receipt,
            {
                "path",
                "row_groups",
                "rows",
                "schema_sha256",
                "sha256",
                "size_bytes",
            },
            label=f"outcome receipt {name}",
        )
        _require(receipt["path"] == name, f"outcome receipt path differs: {name}")
        _sha(receipt["schema_sha256"], label=f"{name} schema")
        _sha(receipt["sha256"], label=f"{name} payload")
    return MappingProxyType(dict(contract))


def _load_numerical_amendment(
    path: Path = NUMERICAL_AMENDMENT_PATH,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate the unique Stage 11-only numerical correction."""

    payload = _stable_file_bytes(path, label="Stage 11 numerical amendment")
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11CacheRecoveryError(
            "Stage 11 numerical amendment is invalid"
        ) from exc
    amendment = _exact_keys(
        parsed,
        {
            "schema_version",
            "contract_id",
            "stage",
            "methodology_version",
            "status",
            "trigger",
            "amendment",
            "frozen_invariants",
            "empirical_numerical_reproduction",
            "required_authority_actions",
        },
        label="Stage 11 numerical amendment",
    )
    _require(
        amendment["schema_version"] == 1
        and amendment["contract_id"]
        == "phase05-stage11-allocation-numerical-amendment-v1"
        and amendment["stage"] == 11
        and amendment["methodology_version"] == "v1"
        and amendment["status"]
        == (
            "FROZEN_USER_AUTHORIZED_CERTIFICATE_RESIDUAL_GAP_OBJECTIVE_"
            "AGREEMENT_AND_FEASIBILITY_STATUS_ROUTING_POLICY"
        ),
        "Stage 11 numerical amendment identity differs",
    )

    correction = _exact_keys(
        amendment["amendment"],
        {
            "adapter_path",
            "adapter_sha256",
            "application_scope",
            "cap_normalized_trade_coordinate",
            "classification",
            "dual_certificate",
            "economic_trial",
            "emergency_final_numerical_path",
            "emergency_lexicographic_numerical_path",
            "failure_regression_manifest_path",
            "failure_regression_manifest_sha256",
            "failure_regression_test_path",
            "failure_regression_test_sha256",
            "test_path",
            "test_sha256",
            "trade_coordinate",
            "trade_numerical_scale",
            "weight_coordinate",
        },
        label="Stage 11 numerical correction",
    )
    certificate_contract = _exact_keys(
        correction["dual_certificate"],
        {
            "accept_primal_optimal_inaccurate_without_certificate",
            "candidate_source",
            "componentwise_stationarity_backward_error_required",
            "dual_cone_audit_required",
            "dual_candidate_source_order",
            "dual_solver_statuses_eligible_for_direct_audit",
            "dual_status_alone_permits_certification",
            "ordinary_feasibility_solver_statuses_eligible_for_candidate_audit",
            "ordinary_feasibility_status_alone_permits_certification",
            "ordinary_inaccurate_feasibility_requires_primal_and_dual_audits",
            "ordinary_infeasible_status_required_to_open_emergency",
            "primary_dual_source_id",
            "fallback_dual_source_id",
            "fallback_entry_rule",
            "fallback_uses_global_uniform_residual_tolerance",
            "failed_certificate_metrics_logged",
            "failure_snapshot_can_permit_acceptance",
            "failure_snapshot_mode",
            "failure_snapshot_root",
            "failure_snapshot_scope",
            "maximum_objective_cost_scaled_difference",
            "maximum_scaled_candidate_dual_gap",
            "previous_maximum_scaled_candidate_dual_gap",
            "maximum_stationarity_or_cone_residual",
            "objective_agreement_tolerance_multiplier_of_retry_gap_absolute",
            "objective_agreement_tolerance_scope",
            "objective_agreement_tolerance_uniform_across_routes_and_months",
            "original_maximum_stationarity_or_cone_residual",
            "previous_maximum_objective_cost_scaled_difference",
            "previous_maximum_stationarity_or_cone_residual",
            "residual_tolerance_changed",
            "residual_tolerance_multiplier_of_retry_feasibility",
            "residual_tolerance_scope",
            "residual_tolerance_uniform_across_routes_and_months",
            "primal_feasibility_tolerance_changed",
            "candidate_dual_gap_tolerance_changed",
            "candidate_dual_gap_tolerance_multiplier_of_independent_absolute",
            "candidate_dual_gap_tolerance_scope",
            "candidate_dual_gap_tolerance_uniform_across_routes_and_months",
            "objective_agreement_tolerance_changed",
            "solver_tolerances_changed",
            "missing_dual_vector_or_value_permitted",
            "primal_constraint_audit_required",
            "reconciled_dual_source_id",
            "reconciliation_changes_primal_candidate",
            "reconciliation_direct_reaudit_required",
            "reconciliation_entry_rule",
            "reconciliation_retry_weight",
            "reconciliation_uses_global_uniform_residual_tolerance",
            "bounded_feasible_dual_source_id",
            "bounded_feasible_entry_rule",
            "bounded_feasible_coordinate_scaling",
            "bounded_feasible_stationarity_multiplier",
            "bounded_feasible_gap_band_fraction",
            "bounded_feasible_gap_reference_tolerance",
            "bounded_feasible_gap_band_absolute",
            "bounded_feasible_gap_band_changed",
            "bounded_feasible_solver_settings",
            "bounded_feasible_changes_primal_candidate",
            "bounded_feasible_uses_global_uniform_residual_tolerance",
            "bounded_feasible_direct_reaudit_required",
            "reported_dual_objective_agreement_required",
            "scope",
            "solver",
            "stationarity_denominator_floor",
            "stationarity_refinement",
            "stationarity_refinement_scope",
            "stationarity_refinement_atol",
            "stationarity_refinement_btol",
            "stationarity_refinement_conlim",
            "stationarity_refinement_maxiter",
        },
        label="Stage 11 hybrid dual-certificate contract",
    )
    emergency_contract = _exact_keys(
        correction["emergency_final_numerical_path"],
        {
            "candidate_audit_scope",
            "certificate_if_equivalent_retry_optimal_inaccurate",
            "dual_certificate_gap_tolerance_changed",
            "dual_certificate_residual_tolerance_changed",
            "dual_certificate_uses_global_uniform_gap_tolerance",
            "dual_certificate_uses_global_uniform_residual_tolerance",
            "economic_objective_changed",
            "emergency_entry_rule_changed",
            "equivalent_final_constraints",
            "equivalent_retry_solver",
            "equivalent_retry_uses_frozen_retry_settings",
            "final_feasible_set_changed",
            "inherited_slack_worsening_tolerance",
            "locked_slack_audit_tolerance",
            "maximum_variable_role_in_final_objective",
            "original_final_constraints",
            "strategy_labeled_failures_required",
            "trigger_status_history",
        },
        label="Stage 11 emergency-final numerical contract",
    )
    lexicographic_contract = _exact_keys(
        correction["emergency_lexicographic_numerical_path"],
        {
            "candidate_dual_gap_tolerance_changed",
            "candidate_scope",
            "certificate_if_optimal_inaccurate",
            "certificate_source_order",
            "componentwise_stationarity_backward_error_required",
            "dual_cone_audit_required",
            "dual_status_alone_permits_certification",
            "economic_objective_changed",
            "eligible_steps",
            "emergency_entry_rule_changed",
            "feasible_set_changed",
            "objective_agreement_tolerance_changed",
            "primal_constraint_audit_required",
            "primal_feasibility_tolerance_changed",
            "reported_dual_objective_agreement_required",
            "solver",
            "solver_tolerances_changed",
            "uses_global_uniform_certificate_residual_tolerance",
            "uses_global_uniform_candidate_dual_gap_tolerance",
            "uses_global_uniform_objective_agreement_tolerance",
        },
        label="Stage 11 emergency lexicographic numerical contract",
    )
    invariants = amendment["frozen_invariants"]
    trigger = amendment["trigger"]
    evidence = amendment["empirical_numerical_reproduction"]
    certificate_evidence = _exact_keys(
        evidence["february_elastic_net_dual_certificate"],
        {
            "adapter_sha256",
            "certificate_source",
            "candidate_dual_gap_scaled",
            "candidate_maximum_constraint_violation",
            "candidate_scaled_objective",
            "deterministic_repeat",
            "dual_nonnegative_minimum",
            "dual_soc_minimum_margin",
            "dual_status",
            "dual_upper_scaled",
            "holding_month",
            "n_assets",
            "objective_cost_scaled_maximum_difference",
            "portfolio_performance_metrics_used",
            "production_solver_attempt",
            "production_solver_mode",
            "production_solver_status",
            "repeat_weight_maximum_absolute_difference",
            "snapshot_archive_sha256",
            "stationarity_absolute_maximum",
            "stationarity_backward_error_maximum",
            "weight_bytes_sha256",
        },
        label="Stage 11 February Elastic Net dual-certificate evidence",
    )
    april_certificate_evidence = _exact_keys(
        evidence["april_ensemble_retry_dual_certificate"],
        {
            "adapter_sha256",
            "candidate_dual_gap_scaled",
            "candidate_maximum_constraint_violation",
            "candidate_scaled_objective",
            "certificate_source",
            "deterministic_repeat",
            "dual_nonnegative_minimum",
            "dual_soc_minimum_margin",
            "dual_status",
            "dual_upper_scaled",
            "holding_month",
            "n_assets",
            "objective_cost_scaled_maximum_difference",
            "portfolio_performance_metrics_used",
            "primary_certificate_failure",
            "primary_certificate_status",
            "primary_candidate_dual_gap_scaled",
            "production_solver_attempt",
            "production_solver_mode",
            "production_solver_status",
            "repeat_weight_maximum_absolute_difference",
            "snapshot_archive_sha256",
            "stationarity_absolute_maximum",
            "stationarity_backward_error_maximum",
            "stationarity_refinement_correction_maximum",
            "stationarity_refinement_iterations",
            "stationarity_refinement_stop_code",
            "strategy_id",
            "weight_bytes_sha256",
        },
        label="Stage 11 April ensemble retry dual-certificate evidence",
    )
    july_certificate_evidence = _exact_keys(
        evidence["july_elastic_net_reconciled_dual_certificate"],
        {
            "adapter_sha256_at_reproduction",
            "current_adapter_sha256",
            "candidate_dual_gap_scaled",
            "candidate_maximum_constraint_violation",
            "candidate_scaled_objective",
            "certificate_source",
            "current_certificate_source",
            "current_deterministic_solver_history_sha256",
            "current_global_candidate_dual_gap_tolerance",
            "deterministic_repeat",
            "dual_nonnegative_minimum",
            "dual_soc_minimum_margin",
            "dual_status",
            "dual_upper_scaled",
            "explicit_candidate_dual_gap_scaled",
            "explicit_certificate_failure",
            "failure_receipt_path",
            "failure_receipt_sha256",
            "failure_seal_path",
            "failure_seal_sha256",
            "failure_snapshot_content_sha256",
            "failure_snapshot_file_sha256",
            "holding_month",
            "n_assets",
            "objective_cost_scaled_maximum_difference",
            "portfolio_performance_metrics_used",
            "production_solver_attempt",
            "production_solver_mode",
            "production_solver_status",
            "reconciliation_retry_weight",
            "regression_fixture_mode",
            "regression_fixture_path",
            "regression_fixture_sha256",
            "regression_fixture_size_bytes",
            "regression_manifest_path",
            "regression_manifest_sha256",
            "regression_test_path",
            "regression_test_sha256",
            "repeat_weight_maximum_absolute_difference",
            "retry_certificate_failure",
            "retry_dual_soc_minimum_margin",
            "stationarity_absolute_maximum",
            "stationarity_backward_error_maximum",
            "stationarity_refinement_correction_maximum",
            "stationarity_refinement_iterations",
            "stationarity_refinement_scale",
            "stationarity_refinement_stop_code",
            "strategy_id",
            "certificate_residual_tolerance_at_reproduction",
            "previous_global_certificate_residual_tolerance",
            "current_global_certificate_residual_tolerance",
            "current_uniform_policy_changes_earliest_accepted_route",
            "thresholds_changed_after_reproduction",
            "weight_bytes_sha256",
        },
        label="Stage 11 July Elastic Net reconciled dual-certificate evidence",
    )
    november_bounded_evidence = _exact_keys(
        evidence["november_ensemble_bounded_feasible_dual_certificate"],
        {
            "adapter_sha256_at_reproduction",
            "current_adapter_sha256",
            "holding_month",
            "strategy_id",
            "n_assets",
            "certificate_source",
            "dual_status",
            "production_solver_mode",
            "production_solver_attempt",
            "production_solver_status",
            "stationarity_absolute_maximum",
            "stationarity_backward_error_maximum",
            "dual_nonnegative_minimum",
            "dual_soc_minimum_margin",
            "candidate_dual_gap_scaled",
            "stationarity_equation_multiplier",
            "candidate_gap_band_absolute",
            "candidate_gap_band_changed",
            "candidate_gap_band_fraction",
            "candidate_gap_band_reference_tolerance",
            "original_explicit_certificate_failure",
            "original_retry_certificate_failure",
            "original_reconciled_certificate_failure",
            "regression_fixture_mode",
            "regression_fixture_path",
            "regression_fixture_sha256",
            "regression_fixture_size_bytes",
            "regression_manifest_path",
            "regression_manifest_sha256",
            "regression_test_path",
            "regression_test_sha256",
            "failure_report_path",
            "failure_report_sha256",
            "portfolio_weights_or_methodology_changed",
            "certificate_residual_tolerance_at_reproduction",
            "previous_global_certificate_residual_tolerance",
            "current_global_certificate_residual_tolerance",
            "current_uniform_policy_changes_earliest_accepted_route",
            "thresholds_changed_after_reproduction",
            "portfolio_performance_metrics_used",
            "stage12_or_later_data_accessed",
            "weight_bytes_sha256",
        },
        label="Stage 11 November bounded-feasible certificate evidence",
    )
    november_sum_evidence = _exact_keys(
        evidence[
            "november_ensemble_emergency_sum_lexicographic_certificate"
        ],
        {
            "adapter_sha256",
            "certificate_gap_tolerance",
            "certificate_residual_tolerance",
            "certificate_route",
            "certificate_source",
            "compatibility_sha256",
            "deterministic_repeat",
            "deterministic_weight_bytes_sha256",
            "dual_status",
            "failure_details_path",
            "failure_details_sha256",
            "failure_id",
            "failure_manifest_path",
            "failure_manifest_sha256",
            "failure_report_path",
            "failure_report_sha256",
            "failure_success_path",
            "failure_success_sha256",
            "final_status_history_entry",
            "holding_month",
            "lexicographic_final_status_history_entry",
            "lexicographic_step",
            "lexicographic_step_status",
            "n_assets",
            "objective_agreement_tolerance",
            "original_global_certificate_residual_tolerance",
            "original_status_history",
            "portfolio_performance_metrics_used",
            "portfolio_weights_or_methodology_changed",
            "problem_bundle_sha256",
            "regression_fixture_mode",
            "regression_fixture_path",
            "regression_fixture_sha256",
            "regression_fixture_size_bytes",
            "regression_manifest_mode",
            "regression_manifest_path",
            "regression_manifest_sha256",
            "regression_test_path",
            "regression_test_sha256",
            "solver_attempt",
            "solver_mode",
            "solver_status",
            "stage12_or_later_data_accessed",
            "strategy_id",
            "previous_global_certificate_residual_tolerance",
            "current_global_certificate_residual_tolerance",
        },
        label="Stage 11 November emergency-sum certificate evidence",
    )
    april_2016_policy_evidence = _exact_keys(
        evidence["april_2016_uniform_certificate_residual_policy"],
        {
            "adapter_sha256",
            "holding_month",
            "strategy_id",
            "failure_id",
            "problem_bundle_sha256",
            "problem_file_sha256",
            "previous_certificate_residual_tolerance",
            "current_certificate_residual_tolerance",
            "certificate_source_after_uniform_policy",
            "dual_status",
            "previous_failed_gate",
            "certificate_audit_after_uniform_policy",
            "candidate_maximum_constraint_violation",
            "stationarity_absolute_maximum",
            "stationarity_backward_error_maximum",
            "dual_nonnegative_minimum",
            "dual_soc_minimum_margin",
            "candidate_dual_gap_scaled",
            "objective_cost_scaled_maximum_difference",
            "primal_feasibility_tolerance_changed",
            "candidate_dual_gap_tolerance_changed",
            "objective_agreement_tolerance_changed",
            "solver_tolerances_changed",
            "portfolio_weights_changed",
            "portfolio_performance_metrics_used",
            "stage12_or_later_data_accessed",
        },
        label="Stage 11 April 2016 uniform certificate residual policy evidence",
    )
    february_2017_policy_evidence = _exact_keys(
        evidence["february_2017_uniform_certificate_residual_policy"],
        {
            "adapter_sha256",
            "certificate_audit_after_current_uniform_policy",
            "certificate_gap_tolerance",
            "certificate_source_at_residual_policy_reproduction",
            "compatibility_sha256",
            "current_candidate_dual_gap_scaled",
            "current_certificate_source",
            "current_dual_status",
            "current_global_candidate_dual_gap_tolerance",
            "current_global_certificate_residual_tolerance",
            "current_uniform_gap_policy_changes_earliest_accepted_route",
            "deterministic_solver_history_sha256",
            "deterministic_weight_bytes_sha256",
            "failure_id",
            "holding_month",
            "immediate_previous_global_certificate_residual_tolerance",
            "opening_checkpoint_sha256",
            "original_global_certificate_residual_tolerance",
            "portfolio_performance_metrics_used",
            "portfolio_weights_or_methodology_changed",
            "problem_bundle_sha256",
            "regression_fixture_mode",
            "regression_fixture_path",
            "regression_fixture_sha256",
            "regression_fixture_size_bytes",
            "regression_manifest_mode",
            "regression_manifest_path",
            "regression_manifest_sha256",
            "regression_test_path",
            "regression_test_sha256",
            "stage12_or_later_data_accessed",
            "strategy_id",
            "thresholds_changed_after_reproduction",
            "previous_global_candidate_dual_gap_tolerance",
        },
        label="Stage 11 February 2017 uniform residual policy evidence",
    )
    october_2017_policy_evidence = _exact_keys(
        evidence["october_2017_uniform_objective_agreement_policy"],
        {
            "adapter_sha256",
            "certificate_audit_at_objective_agreement_policy_reproduction",
            "certificate_gap_tolerance",
            "certificate_residual_tolerance",
            "certificate_source_at_objective_agreement_policy_reproduction",
            "compatibility_sha256",
            "current_economic_certificate_source",
            "current_global_candidate_dual_gap_tolerance",
            "current_global_objective_agreement_tolerance",
            "current_sum_certificate_source",
            "current_sum_candidate_dual_gap",
            "current_sum_complementarity_absolute",
            "current_uniform_gap_policy_changes_earliest_sum_route",
            "deterministic_solver_history_sha256",
            "deterministic_weight_bytes_sha256",
            "historical_retry_dual_status",
            "failure_authority_mode",
            "failure_details_path",
            "failure_details_sha256",
            "failure_id",
            "failure_manifest_path",
            "failure_manifest_sha256",
            "failure_report_path",
            "failure_report_sha256",
            "failure_success_path",
            "failure_success_sha256",
            "holding_month",
            "immediate_previous_global_objective_agreement_tolerance",
            "objective_agreement_maximum_difference",
            "opening_checkpoint_sha256",
            "portfolio_performance_metrics_used",
            "portfolio_weights_or_methodology_changed",
            "problem_bundle_sha256",
            "regression_fixture_mode",
            "regression_fixture_path",
            "regression_fixture_sha256",
            "regression_fixture_size_bytes",
            "regression_manifest_mode",
            "regression_manifest_path",
            "regression_manifest_sha256",
            "regression_test_path",
            "regression_test_sha256",
            "stage12_or_later_data_accessed",
            "strategy_id",
            "thresholds_changed_after_reproduction",
            "previous_global_candidate_dual_gap_tolerance",
            "uniform_policy_multiplier_of_retry_gap_absolute",
        },
        label="Stage 11 October 2017 uniform objective-agreement policy evidence",
    )
    june_2018_routing_evidence = _exact_keys(
        evidence["june_2018_inaccurate_feasibility_status_routing"],
        {
            "adapter_sha256",
            "candidate_dual_gap_scaled",
            "candidate_maximum_constraint_violation",
            "certificate_audit_after_status_routing",
            "certificate_source_after_status_routing",
            "compatibility_sha256",
            "deterministic_solver_history_sha256",
            "deterministic_weight_bytes_sha256",
            "dual_nonnegative_minimum",
            "dual_soc_minimum_margin",
            "dual_status",
            "emergency_entry_rule_changed",
            "failure_authority_mode",
            "failure_details_path",
            "failure_details_sha256",
            "failure_id",
            "failure_manifest_path",
            "failure_manifest_sha256",
            "failure_report_path",
            "failure_report_sha256",
            "failure_success_path",
            "failure_success_sha256",
            "feasibility_solver_status",
            "feasibility_status_alone_permits_certification",
            "full_primal_and_dual_audits_required",
            "holding_month",
            "n_assets",
            "objective_cost_scaled_maximum_difference",
            "opening_checkpoint_sha256",
            "portfolio_performance_metrics_used",
            "portfolio_weights_or_methodology_changed",
            "problem_bundle_sha256",
            "regression_fixture_mode",
            "regression_fixture_path",
            "regression_fixture_sha256",
            "regression_fixture_size_bytes",
            "regression_manifest_mode",
            "regression_manifest_path",
            "regression_manifest_sha256",
            "regression_test_path",
            "regression_test_sha256",
            "solver_status_after_routing",
            "stage12_or_later_data_accessed",
            "stationarity_absolute_maximum",
            "stationarity_backward_error_maximum",
            "strategy_id",
        },
        label="Stage 11 June 2018 feasibility-status routing evidence",
    )
    june_2019_gap_evidence = _exact_keys(
        evidence["june_2019_uniform_candidate_dual_gap_policy"],
        {
            "adapter_sha256",
            "allocation_constraints_or_limits_changed",
            "bounded_dual_gap_band_absolute",
            "bounded_dual_gap_band_changed",
            "bounded_dual_gap_band_fraction",
            "bounded_dual_gap_reference_tolerance",
            "candidate_audit_result",
            "candidate_dual_gap_scaled",
            "candidate_dual_gap_tolerance_multiplier_of_independent_absolute",
            "candidate_dual_gap_tolerance_scope",
            "candidate_dual_gap_tolerance_uniform_across_routes_and_months",
            "candidate_locked_sum_slack_violation",
            "candidate_maximum_hard_or_noninherited_violation",
            "candidate_maximum_inherited_worsening",
            "certificate_residual_tolerance_changed",
            "certificate_source_after_uniform_policy",
            "compatibility_sha256",
            "current_global_candidate_dual_gap_tolerance",
            "deterministic_repeat",
            "deterministic_solver_history_sha256",
            "deterministic_weight_bytes_sha256",
            "dual_nonnegative_minimum",
            "dual_soc_minimum_margin",
            "dual_status",
            "failure_authority_mode",
            "failure_bundle_sha256",
            "failure_details_path",
            "failure_details_sha256",
            "failure_id",
            "failure_manifest_path",
            "failure_manifest_sha256",
            "failure_report_path",
            "failure_report_sha256",
            "failure_success_path",
            "failure_success_sha256",
            "holding_month",
            "maximum_result_violation",
            "objective_agreement_tolerance_changed",
            "objective_changed",
            "objective_cost_scaled_maximum_difference",
            "opening_checkpoint_sha256",
            "portfolio_performance_metrics_used",
            "portfolio_weights_or_methodology_changed",
            "previous_failed_gate",
            "previous_global_candidate_dual_gap_tolerance",
            "primal_feasibility_tolerance_changed",
            "problem_bundle_sha256",
            "problem_file_sha256",
            "regression_fixture_mode",
            "regression_fixture_path",
            "regression_fixture_sha256",
            "regression_fixture_size_bytes",
            "regression_manifest_mode",
            "regression_manifest_path",
            "regression_manifest_sha256",
            "regression_test_path",
            "regression_test_sha256",
            "solver_attempt_after_policy",
            "solver_mode_after_policy",
            "solver_status_after_policy",
            "solver_tolerances_changed",
            "stage12_or_later_data_accessed",
            "stationarity_absolute_maximum",
            "stationarity_backward_error_maximum",
            "strategy_id",
        },
        label="Stage 11 June 2019 uniform candidate-dual-gap evidence",
    )
    emergency_evidence = _exact_keys(
        evidence["synthetic_emergency_final_reproduction"],
        {
            "candidate_dual_gap_scaled",
            "candidate_maximum_constraint_violation",
            "dual_nonnegative_minimum",
            "dual_soc_minimum_margin",
            "equivalent_retry_objective_absolute_difference",
            "equivalent_retry_status",
            "equivalent_retry_weight_maximum_absolute_difference",
            "forced_certificate_status",
            "forced_original_final_statuses",
            "locked_maximum_slack_violation",
            "locked_sum_slack_violation",
            "objective_cost_scaled_maximum_difference",
            "portfolio_performance_metrics_used",
            "stationarity_absolute_maximum",
            "stationarity_backward_error_maximum",
        },
        label="Stage 11 synthetic emergency-final evidence",
    )
    authority_actions = amendment["required_authority_actions"]
    _require(
        correction["classification"]
        == (
            "rule_restoring_numerical_correction_with_user_authorized_"
            "uniform_certificate_residual_gap_objective_agreement_and_"
            "feasibility_status_routing_policy"
        )
        and correction["economic_trial"] is False
        and correction["application_scope"]
        == "all_stage11_allocation_architectures_and_all_59_validation_months_from_month_1"
        and correction["cap_normalized_trade_coordinate"] is True
        and correction["weight_coordinate"]
        == "previous_weight_plus_side_cap_times_normalized_trade"
        and correction["trade_coordinate"]
        == "affine_basis_points_of_nav_without_pretrade_subtraction"
        and correction["trade_numerical_scale"] == TRADE_NUMERICAL_SCALE
        == 10_000.0,
        "Stage 11 numerical coordinate contract differs",
    )
    _require(
        certificate_contract["scope"]
        == (
            "ordinary_hybrid_emergency_final_economic_and_emergency_"
            "minimax_sum_lexicographic_optimal_inaccurate_paths_only"
        )
        and certificate_contract["candidate_source"]
        == (
            "independently_primal_audited_ordinary_hybrid_retry_or_"
            "emergency_equivalent_retry_or_unchanged_emergency_minimax_"
            "sum_lexicographic_candidate_optimal_inaccurate_after_"
            "feasibility_status_routing"
        )
        and certificate_contract["solver"] == "CLARABEL"
        and certificate_contract["dual_candidate_source_order"]
        == [
            "separately_solved_explicit_dual",
            "same_retry_economic_clarabel_canonical_dual",
            "equal_convex_blend_of_explicit_and_retry_duals",
            "cone_preserving_scaled_bounded_dual_feasibility",
        ]
        and certificate_contract["primary_dual_source_id"]
        == EXPLICIT_DUAL_CERTIFICATE_SOURCE
        and certificate_contract["fallback_dual_source_id"]
        == RETRY_DUAL_CERTIFICATE_SOURCE
        and certificate_contract["fallback_entry_rule"]
        == (
            "primary_explicit_dual_failed_one_or_more_active_"
            "direct_audit_gates"
        )
        and certificate_contract[
            "fallback_uses_global_uniform_residual_tolerance"
        ]
        is True
        and certificate_contract["reconciled_dual_source_id"]
        == RECONCILED_DUAL_CERTIFICATE_SOURCE
        and certificate_contract["reconciliation_entry_rule"]
        == (
            "explicit_and_retry_duals_each_failed_one_or_more_active_"
            "direct_audit_gates"
        )
        and certificate_contract["reconciliation_retry_weight"]
        == DUAL_RECONCILIATION_RETRY_WEIGHT
        == 0.5
        and certificate_contract["reconciliation_changes_primal_candidate"]
        is False
        and certificate_contract["reconciliation_direct_reaudit_required"]
        is True
        and certificate_contract[
            "reconciliation_uses_global_uniform_residual_tolerance"
        ]
        is True
        and certificate_contract["bounded_feasible_dual_source_id"]
        == BOUNDED_FEASIBLE_DUAL_CERTIFICATE_SOURCE
        and certificate_contract["bounded_feasible_entry_rule"]
        == (
            "explicit_retry_and_equal_blend_certificates_each_failed_"
            "one_or_more_active_direct_audit_gates"
        )
        and certificate_contract["bounded_feasible_coordinate_scaling"]
        == (
            "positive_diagonal_row_infinity_scaling_with_one_common_"
            "scale_per_second_order_cone"
        )
        and certificate_contract[
            "bounded_feasible_stationarity_multiplier"
        ]
        == BOUNDED_DUAL_STATIONARITY_MULTIPLIER
        == 10_000.0
        and certificate_contract["bounded_feasible_gap_band_fraction"]
        == BOUNDED_DUAL_GAP_FRACTION
        == 0.25
        and certificate_contract["bounded_feasible_gap_reference_tolerance"]
        == BOUNDED_DUAL_GAP_REFERENCE_TOLERANCE
        == INDEPENDENT_ABSOLUTE_TOLERANCE
        and certificate_contract["bounded_feasible_gap_band_absolute"]
        == BOUNDED_DUAL_GAP_BAND
        == (
            BOUNDED_DUAL_GAP_FRACTION
            * BOUNDED_DUAL_GAP_REFERENCE_TOLERANCE
        )
        and certificate_contract["bounded_feasible_gap_band_changed"] is False
        and certificate_contract["bounded_feasible_solver_settings"]
        == "frozen_retry"
        and certificate_contract["bounded_feasible_changes_primal_candidate"]
        is False
        and certificate_contract[
            "bounded_feasible_uses_global_uniform_residual_tolerance"
        ]
        is True
        and certificate_contract["bounded_feasible_direct_reaudit_required"]
        is True
        and certificate_contract["failed_certificate_metrics_logged"] is True
        and certificate_contract["failure_snapshot_can_permit_acceptance"]
        is False
        and certificate_contract["failure_snapshot_mode"] == "0600"
        and certificate_contract["failure_snapshot_root"]
        == DUAL_FAILURE_SNAPSHOT_ROOT.as_posix()
        and certificate_contract["failure_snapshot_scope"]
        == "ordinary_hybrid_after_all_certificate_routes_fail"
        and certificate_contract[
            "dual_solver_statuses_eligible_for_direct_audit"
        ]
        == ["optimal", "optimal_inaccurate"]
        and set(
            certificate_contract[
                "dual_solver_statuses_eligible_for_direct_audit"
            ]
        )
        == set(DUAL_CERTIFICATE_AUDITABLE_STATUSES)
        and certificate_contract[
            "dual_status_alone_permits_certification"
        ]
        is False
        and certificate_contract[
            "ordinary_feasibility_solver_statuses_eligible_for_candidate_audit"
        ]
        == ["optimal", "optimal_inaccurate"]
        and certificate_contract[
            "ordinary_feasibility_status_alone_permits_certification"
        ]
        is False
        and certificate_contract[
            "ordinary_inaccurate_feasibility_requires_primal_and_dual_audits"
        ]
        is True
        and certificate_contract[
            "ordinary_infeasible_status_required_to_open_emergency"
        ]
        is True
        and certificate_contract[
            "missing_dual_vector_or_value_permitted"
        ]
        is False
        and certificate_contract[
            "accept_primal_optimal_inaccurate_without_certificate"
        ]
        is False
        and certificate_contract["primal_constraint_audit_required"] is True
        and certificate_contract[
            "reported_dual_objective_agreement_required"
        ]
        is True
        and certificate_contract[
            "componentwise_stationarity_backward_error_required"
        ]
        is True
        and certificate_contract["dual_cone_audit_required"] is True
        and certificate_contract["maximum_stationarity_or_cone_residual"]
        == DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and certificate_contract[
            "original_maximum_stationarity_or_cone_residual"
        ]
        == DUAL_CERTIFICATE_RESIDUAL_TOLERANCE / 20.0
        and certificate_contract[
            "previous_maximum_stationarity_or_cone_residual"
        ]
        == DUAL_CERTIFICATE_RESIDUAL_TOLERANCE / 2.0
        and certificate_contract[
            "original_maximum_stationarity_or_cone_residual"
        ]
        < certificate_contract[
            "previous_maximum_stationarity_or_cone_residual"
        ]
        < certificate_contract["maximum_stationarity_or_cone_residual"]
        and certificate_contract["residual_tolerance_changed"] is True
        and certificate_contract[
            "residual_tolerance_multiplier_of_retry_feasibility"
        ]
        == 200.0
        and certificate_contract["residual_tolerance_scope"]
        == "all_stage11_validation_months_and_all_dual_certificate_routes"
        and certificate_contract[
            "residual_tolerance_uniform_across_routes_and_months"
        ]
        is True
        and certificate_contract["primal_feasibility_tolerance_changed"]
        is False
        and certificate_contract["candidate_dual_gap_tolerance_changed"]
        is True
        and certificate_contract["previous_maximum_scaled_candidate_dual_gap"]
        == INDEPENDENT_ABSOLUTE_TOLERANCE
        and certificate_contract["maximum_scaled_candidate_dual_gap"]
        == DUAL_CERTIFICATE_GAP_TOLERANCE
        == 2.0 * INDEPENDENT_ABSOLUTE_TOLERANCE
        and certificate_contract[
            "candidate_dual_gap_tolerance_multiplier_of_independent_absolute"
        ]
        == 2.0
        and certificate_contract["candidate_dual_gap_tolerance_scope"]
        == "all_stage11_validation_months_and_all_dual_certificate_routes"
        and certificate_contract[
            "candidate_dual_gap_tolerance_uniform_across_routes_and_months"
        ]
        is True
        and certificate_contract["objective_agreement_tolerance_changed"]
        is True
        and certificate_contract[
            "previous_maximum_objective_cost_scaled_difference"
        ]
        == DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE / 5.0
        and certificate_contract[
            "previous_maximum_objective_cost_scaled_difference"
        ]
        < certificate_contract["maximum_objective_cost_scaled_difference"]
        and certificate_contract[
            "objective_agreement_tolerance_multiplier_of_retry_gap_absolute"
        ]
        == 50.0
        and certificate_contract["objective_agreement_tolerance_scope"]
        == "all_stage11_validation_months_and_all_dual_certificate_routes"
        and certificate_contract[
            "objective_agreement_tolerance_uniform_across_routes_and_months"
        ]
        is True
        and certificate_contract["solver_tolerances_changed"] is False
        and certificate_contract["maximum_objective_cost_scaled_difference"]
        == DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and certificate_contract["stationarity_denominator_floor"]
        == DUAL_CERTIFICATE_STATIONARITY_FLOOR
        and certificate_contract["stationarity_refinement"]
        == "deterministic_minimum_norm_lsmr"
        and certificate_contract["stationarity_refinement_scope"]
        == "fallback_retry_dual_only"
        and certificate_contract["stationarity_refinement_atol"]
        == DUAL_STATIONARITY_REFINEMENT_TOLERANCE
        and certificate_contract["stationarity_refinement_btol"]
        == DUAL_STATIONARITY_REFINEMENT_TOLERANCE
        and certificate_contract["stationarity_refinement_conlim"] == 0.0
        and certificate_contract["stationarity_refinement_maxiter"]
        == "minimum_of_canonical_rows_and_columns",
        "Stage 11 hybrid dual-certificate contract differs",
    )
    _require(
        lexicographic_contract["candidate_scope"]
        == "unchanged_emergency_minimax_or_sum_primal_candidate_only"
        and lexicographic_contract["certificate_if_optimal_inaccurate"]
        is True
        and lexicographic_contract["certificate_source_order"]
        == [
            EXPLICIT_DUAL_CERTIFICATE_SOURCE,
            RETRY_DUAL_CERTIFICATE_SOURCE,
        ]
        and lexicographic_contract["eligible_steps"]
        == ["emergency_minimax", "emergency_sum"]
        and lexicographic_contract["solver"] == "CLARABEL"
        and lexicographic_contract[
            "uses_global_uniform_certificate_residual_tolerance"
        ]
        is True
        and lexicographic_contract[
            "uses_global_uniform_candidate_dual_gap_tolerance"
        ]
        is True
        and lexicographic_contract[
            "uses_global_uniform_objective_agreement_tolerance"
        ]
        is True
        and lexicographic_contract["candidate_dual_gap_tolerance_changed"]
        is True
        and lexicographic_contract[
            "objective_agreement_tolerance_changed"
        ]
        is True
        and lexicographic_contract["dual_status_alone_permits_certification"]
        is False
        and lexicographic_contract["primal_constraint_audit_required"]
        is True
        and lexicographic_contract["dual_cone_audit_required"] is True
        and lexicographic_contract[
            "componentwise_stationarity_backward_error_required"
        ]
        is True
        and lexicographic_contract[
            "reported_dual_objective_agreement_required"
        ]
        is True
        and all(
            lexicographic_contract[name] is False
            for name in (
                "economic_objective_changed",
                "emergency_entry_rule_changed",
                "feasible_set_changed",
                "primal_feasibility_tolerance_changed",
                "solver_tolerances_changed",
            )
        ),
        "Stage 11 emergency lexicographic numerical contract differs",
    )
    _require(
        emergency_contract["candidate_audit_scope"]
        == "equivalent_retry_or_dual_certified_candidate_only"
        and emergency_contract[
            "certificate_if_equivalent_retry_optimal_inaccurate"
        ]
        is True
        and emergency_contract[
            "dual_certificate_gap_tolerance_changed"
        ]
        is True
        and emergency_contract[
            "dual_certificate_residual_tolerance_changed"
        ]
        is True
        and emergency_contract[
            "dual_certificate_uses_global_uniform_gap_tolerance"
        ]
        is True
        and emergency_contract[
            "dual_certificate_uses_global_uniform_residual_tolerance"
        ]
        is True
        and emergency_contract["economic_objective_changed"] is False
        and emergency_contract["emergency_entry_rule_changed"] is False
        and emergency_contract["equivalent_final_constraints"]
        == (
            "slack_vector_le_locked_maximum_and_"
            "slack_sum_le_locked_sum"
        )
        and emergency_contract["equivalent_retry_solver"] == "CLARABEL"
        and emergency_contract[
            "equivalent_retry_uses_frozen_retry_settings"
        ]
        is True
        and emergency_contract["final_feasible_set_changed"] is False
        and emergency_contract["inherited_slack_worsening_tolerance"]
        == EMERGENCY_WORSENING_TOLERANCE
        and emergency_contract["locked_slack_audit_tolerance"]
        == INDEPENDENT_ABSOLUTE_TOLERANCE
        and emergency_contract["maximum_variable_role_in_final_objective"]
        == "none"
        and emergency_contract["original_final_constraints"]
        == (
            "slack_vector_le_maximum_and_maximum_le_locked_maximum_"
            "and_slack_sum_le_locked_sum"
        )
        and emergency_contract["strategy_labeled_failures_required"] is True
        and emergency_contract["trigger_status_history"]
        == [
            "emergency_economic_primary:optimal_inaccurate",
            "emergency_economic_retry:optimal_inaccurate",
        ],
        "Stage 11 emergency-final numerical contract differs",
    )
    _require(
        invariants["solver"] == "CLARABEL"
        and invariants["solver_version"] == "0.11.1"
        and invariants["alternate_solver_permitted"] is False
        and all(
            invariants[name] is False
            for name in (
                "borrow_cost_changed",
                "constraints_or_limits_changed",
                "emergency_entry_rule_changed",
                "expected_return_changed",
                "financing_cost_changed",
                "independent_primal_feasibility_audit_tolerance_changed",
                "objective_changed",
                "objective_numerical_scale_changed",
                "primary_and_retry_tolerances_changed",
                "transaction_cost_changed",
            )
        )
        and invariants["dual_certificate_residual_tolerance_changed"] is True
        and invariants["candidate_dual_gap_tolerance_changed"] is True
        and invariants["objective_agreement_tolerance_changed"] is True
        and invariants["ordinary_feasibility_status_routing_changed"] is True,
        "Stage 11 numerical amendment changed a frozen economic rule",
    )
    _require(
        trigger["no_weights_trades_pnl_or_result_bundle_published"] is True
        and evidence[
            "correction_choice_used_only_solver_status_and_algebraic_equivalence_audits"
        ]
        is True
        and evidence[
            "portfolio_performance_metrics_computed_or_published_before_correction"
        ]
        is False
        and evidence["ensemble_status_after_correction"] == "optimal"
        and evidence["elastic_net_status_after_correction"] == "optimal"
        and evidence["ensemble_corrected_maximum_constraint_violation"]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and evidence["elastic_net_corrected_maximum_constraint_violation"]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and evidence["repeat_weight_maximum_absolute_difference"] <= 1e-9,
        "Stage 11 numerical amendment evidence differs",
    )
    _sha(
        certificate_evidence["snapshot_archive_sha256"],
        label="February Elastic Net allocation snapshot",
    )
    _sha(
        certificate_evidence["weight_bytes_sha256"],
        label="February Elastic Net certified weights",
    )
    _require(
        certificate_evidence["adapter_sha256"]
        == correction["adapter_sha256"]
        and certificate_evidence["certificate_source"]
        == EXPLICIT_DUAL_CERTIFICATE_SOURCE
        and certificate_evidence["holding_month"] == "2015-02"
        and certificate_evidence["n_assets"] == 958
        and certificate_evidence["snapshot_archive_sha256"]
        == "5de4d6428896ca61d4842fcd3819411172e5649155250e384e53b7064bf64799"
        and certificate_evidence["weight_bytes_sha256"]
        == "64d56705ff9ef967c5416dc235632419dd5572a6198e5733e07d593d3c075a9f"
        and certificate_evidence["deterministic_repeat"] is True
        and certificate_evidence["repeat_weight_maximum_absolute_difference"]
        == 0.0
        and certificate_evidence["portfolio_performance_metrics_used"] is False
        and certificate_evidence["production_solver_mode"] == "ordinary"
        and certificate_evidence["production_solver_attempt"] == "retry"
        and certificate_evidence["production_solver_status"]
        == DUAL_CERTIFIED_SOLVER_STATUS
        and certificate_evidence["dual_status"]
        in certificate_contract[
            "dual_solver_statuses_eligible_for_direct_audit"
        ]
        and certificate_evidence["candidate_maximum_constraint_violation"]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and certificate_evidence["stationarity_absolute_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and certificate_evidence["stationarity_backward_error_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and certificate_evidence["dual_nonnegative_minimum"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and certificate_evidence["dual_soc_minimum_margin"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and abs(certificate_evidence["candidate_dual_gap_scaled"])
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and certificate_evidence["objective_cost_scaled_maximum_difference"]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and abs(
            certificate_evidence["dual_upper_scaled"]
            - certificate_evidence["candidate_scaled_objective"]
            - certificate_evidence["candidate_dual_gap_scaled"]
        )
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE,
        "Stage 11 February Elastic Net dual-certificate evidence differs",
    )
    _sha(
        april_certificate_evidence["snapshot_archive_sha256"],
        label="April ensemble allocation snapshot",
    )
    _sha(
        april_certificate_evidence["weight_bytes_sha256"],
        label="April ensemble certified weights",
    )
    _require(
        april_certificate_evidence["adapter_sha256"]
        == correction["adapter_sha256"]
        and april_certificate_evidence["certificate_source"]
        == RETRY_DUAL_CERTIFICATE_SOURCE
        and april_certificate_evidence["holding_month"] == "2015-04"
        and april_certificate_evidence["strategy_id"] == PRIMARY_STRATEGY_ID
        and april_certificate_evidence["n_assets"] == 1_217
        and april_certificate_evidence["snapshot_archive_sha256"]
        == "3060fdb7956537c007a47867ad43e3a1dec0da281e30858666a61e3d76198fb0"
        and april_certificate_evidence["weight_bytes_sha256"]
        == "5ff53ec26976da19abd178f87382a0096d1eb7b8108e185a8a5b98b17655d573"
        and april_certificate_evidence["deterministic_repeat"] is True
        and april_certificate_evidence[
            "repeat_weight_maximum_absolute_difference"
        ]
        == 0.0
        and april_certificate_evidence["portfolio_performance_metrics_used"]
        is False
        and april_certificate_evidence["primary_certificate_failure"]
        == "candidate_dual_gap"
        and april_certificate_evidence["primary_certificate_status"]
        == "optimal_inaccurate"
        and abs(
            april_certificate_evidence[
                "primary_candidate_dual_gap_scaled"
            ]
        )
        > DUAL_CERTIFICATE_GAP_TOLERANCE
        and april_certificate_evidence["production_solver_mode"] == "ordinary"
        and april_certificate_evidence["production_solver_attempt"] == "retry"
        and april_certificate_evidence["production_solver_status"]
        == DUAL_CERTIFIED_SOLVER_STATUS
        and april_certificate_evidence["dual_status"]
        == "optimal_inaccurate"
        and april_certificate_evidence["stationarity_refinement_stop_code"]
        == 1
        and april_certificate_evidence["stationarity_refinement_iterations"]
        == 28
        and april_certificate_evidence[
            "stationarity_refinement_correction_maximum"
        ]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and april_certificate_evidence[
            "candidate_maximum_constraint_violation"
        ]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and april_certificate_evidence["stationarity_absolute_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and april_certificate_evidence[
            "stationarity_backward_error_maximum"
        ]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and april_certificate_evidence["dual_nonnegative_minimum"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and april_certificate_evidence["dual_soc_minimum_margin"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and abs(april_certificate_evidence["candidate_dual_gap_scaled"])
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and april_certificate_evidence[
            "objective_cost_scaled_maximum_difference"
        ]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and abs(
            april_certificate_evidence["dual_upper_scaled"]
            - april_certificate_evidence["candidate_scaled_objective"]
            - april_certificate_evidence["candidate_dual_gap_scaled"]
        )
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE,
        "Stage 11 April ensemble dual-certificate evidence differs",
    )
    for field, label in (
        ("failure_receipt_sha256", "July dual-certificate failure receipt"),
        ("failure_seal_sha256", "July dual-certificate failure seal"),
        ("failure_snapshot_content_sha256", "July failure snapshot content"),
        ("failure_snapshot_file_sha256", "July failure snapshot file"),
        (
            "current_deterministic_solver_history_sha256",
            "July current solver history",
        ),
        ("regression_fixture_sha256", "July regression fixture"),
        ("regression_manifest_sha256", "July regression manifest"),
        ("regression_test_sha256", "July regression test"),
        ("weight_bytes_sha256", "July certified weights"),
    ):
        _sha(july_certificate_evidence[field], label=label)
    _require(
        july_certificate_evidence["adapter_sha256_at_reproduction"]
        == "1a215fb0f87338ce415767993caad666719d429446e0bd33704d23e7655e82eb"
        and july_certificate_evidence["current_adapter_sha256"]
        == correction["adapter_sha256"]
        and july_certificate_evidence["current_certificate_source"]
        == EXPLICIT_DUAL_CERTIFICATE_SOURCE
        and july_certificate_evidence[
            "current_deterministic_solver_history_sha256"
        ]
        == "4d82fe2a804dca5647724cd9ed7fae09bf693ca562a919a6d26f19f08036924c"
        and july_certificate_evidence[
            "current_global_candidate_dual_gap_tolerance"
        ]
        == DUAL_CERTIFICATE_GAP_TOLERANCE
        and july_certificate_evidence["holding_month"] == "2015-07"
        and july_certificate_evidence["strategy_id"]
        == ELASTIC_NET_CHALLENGER_ID
        and july_certificate_evidence["n_assets"] == 1_511
        and july_certificate_evidence["certificate_source"]
        == RECONCILED_DUAL_CERTIFICATE_SOURCE
        and july_certificate_evidence["reconciliation_retry_weight"]
        == DUAL_RECONCILIATION_RETRY_WEIGHT
        and july_certificate_evidence["explicit_certificate_failure"]
        == "candidate_dual_gap"
        and abs(
            july_certificate_evidence[
                "explicit_candidate_dual_gap_scaled"
            ]
        )
        > certificate_contract["previous_maximum_scaled_candidate_dual_gap"]
        and abs(
            july_certificate_evidence[
                "explicit_candidate_dual_gap_scaled"
            ]
        )
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and july_certificate_evidence["retry_certificate_failure"]
        == "dual_soc_cone"
        and july_certificate_evidence["retry_dual_soc_minimum_margin"]
        < -certificate_contract[
            "original_maximum_stationarity_or_cone_residual"
        ]
        and july_certificate_evidence[
            "certificate_residual_tolerance_at_reproduction"
        ]
        == certificate_contract[
            "original_maximum_stationarity_or_cone_residual"
        ]
        and july_certificate_evidence[
            "previous_global_certificate_residual_tolerance"
        ]
        == certificate_contract[
            "previous_maximum_stationarity_or_cone_residual"
        ]
        and july_certificate_evidence[
            "current_global_certificate_residual_tolerance"
        ]
        == DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and july_certificate_evidence[
            "current_uniform_policy_changes_earliest_accepted_route"
        ]
        is True
        and july_certificate_evidence[
            "thresholds_changed_after_reproduction"
        ]
        is True
        and july_certificate_evidence["portfolio_performance_metrics_used"]
        is False
        and july_certificate_evidence["deterministic_repeat"] is True
        and july_certificate_evidence[
            "repeat_weight_maximum_absolute_difference"
        ]
        == 0.0
        and july_certificate_evidence["production_solver_mode"] == "ordinary"
        and july_certificate_evidence["production_solver_attempt"] == "retry"
        and july_certificate_evidence["production_solver_status"]
        == DUAL_CERTIFIED_SOLVER_STATUS
        and july_certificate_evidence["dual_status"]
        == "optimal_inaccurate"
        and july_certificate_evidence["candidate_maximum_constraint_violation"]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and july_certificate_evidence["stationarity_absolute_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and july_certificate_evidence[
            "stationarity_backward_error_maximum"
        ]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and july_certificate_evidence["dual_nonnegative_minimum"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and july_certificate_evidence["dual_soc_minimum_margin"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and abs(july_certificate_evidence["candidate_dual_gap_scaled"])
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and july_certificate_evidence[
            "objective_cost_scaled_maximum_difference"
        ]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and abs(
            july_certificate_evidence["dual_upper_scaled"]
            - july_certificate_evidence["candidate_scaled_objective"]
            - july_certificate_evidence["candidate_dual_gap_scaled"]
        )
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and july_certificate_evidence["stationarity_refinement_scale"]
        == DUAL_RECONCILIATION_RETRY_WEIGHT
        and july_certificate_evidence["stationarity_refinement_stop_code"] == 1
        and july_certificate_evidence["stationarity_refinement_iterations"]
        == 30
        and july_certificate_evidence["regression_fixture_mode"] == "0644"
        and july_certificate_evidence["regression_fixture_size_bytes"]
        == 21_328_249
        and july_certificate_evidence["failure_snapshot_file_sha256"]
        == july_certificate_evidence["regression_fixture_sha256"]
        and july_certificate_evidence["regression_manifest_path"]
        == correction["failure_regression_manifest_path"]
        and july_certificate_evidence["regression_manifest_sha256"]
        == correction["failure_regression_manifest_sha256"]
        and july_certificate_evidence["regression_test_path"]
        == correction["failure_regression_test_path"]
        and july_certificate_evidence["regression_test_sha256"]
        == correction["failure_regression_test_sha256"],
        "Stage 11 July Elastic Net reconciled certificate evidence differs",
    )
    for field, label in (
        ("regression_fixture_sha256", "November bounded regression fixture"),
        ("regression_manifest_sha256", "November bounded regression manifest"),
        ("regression_test_sha256", "November bounded regression test"),
        ("failure_report_sha256", "November bounded failure report"),
        ("weight_bytes_sha256", "November bounded certified weights"),
    ):
        _sha(november_bounded_evidence[field], label=label)
    _require(
        november_bounded_evidence["adapter_sha256_at_reproduction"]
        == "1a215fb0f87338ce415767993caad666719d429446e0bd33704d23e7655e82eb"
        and november_bounded_evidence["current_adapter_sha256"]
        == correction["adapter_sha256"]
        and november_bounded_evidence["holding_month"] == "2015-11"
        and november_bounded_evidence["strategy_id"] == PRIMARY_STRATEGY_ID
        and november_bounded_evidence["n_assets"] == 1_552
        and november_bounded_evidence["certificate_source"]
        == BOUNDED_FEASIBLE_DUAL_CERTIFICATE_SOURCE
        and november_bounded_evidence["dual_status"] == "optimal_inaccurate"
        and november_bounded_evidence["production_solver_mode"] == "emergency"
        and november_bounded_evidence["production_solver_attempt"]
        == "emergency"
        and november_bounded_evidence["production_solver_status"]
        == DUAL_CERTIFIED_SOLVER_STATUS
        and november_bounded_evidence[
            "certificate_residual_tolerance_at_reproduction"
        ]
        == certificate_contract[
            "original_maximum_stationarity_or_cone_residual"
        ]
        and november_bounded_evidence[
            "previous_global_certificate_residual_tolerance"
        ]
        == certificate_contract[
            "previous_maximum_stationarity_or_cone_residual"
        ]
        and november_bounded_evidence[
            "current_global_certificate_residual_tolerance"
        ]
        == DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and november_bounded_evidence[
            "stationarity_absolute_maximum"
        ]
        <= november_bounded_evidence[
            "certificate_residual_tolerance_at_reproduction"
        ]
        and november_bounded_evidence[
            "stationarity_backward_error_maximum"
        ]
        <= november_bounded_evidence[
            "certificate_residual_tolerance_at_reproduction"
        ]
        and november_bounded_evidence["dual_nonnegative_minimum"]
        >= -november_bounded_evidence[
            "certificate_residual_tolerance_at_reproduction"
        ]
        and november_bounded_evidence["dual_soc_minimum_margin"]
        >= -november_bounded_evidence[
            "certificate_residual_tolerance_at_reproduction"
        ]
        and abs(november_bounded_evidence["candidate_dual_gap_scaled"])
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and november_bounded_evidence["stationarity_equation_multiplier"]
        == BOUNDED_DUAL_STATIONARITY_MULTIPLIER
        and november_bounded_evidence["candidate_gap_band_fraction"]
        == BOUNDED_DUAL_GAP_FRACTION
        and november_bounded_evidence[
            "candidate_gap_band_reference_tolerance"
        ]
        == BOUNDED_DUAL_GAP_REFERENCE_TOLERANCE
        and november_bounded_evidence["candidate_gap_band_absolute"]
        == BOUNDED_DUAL_GAP_BAND
        and november_bounded_evidence["candidate_gap_band_changed"] is False
        and november_bounded_evidence["original_explicit_certificate_failure"]
        == "dual_stationarity_backward_error"
        and november_bounded_evidence["original_retry_certificate_failure"]
        == "dual_soc_cone"
        and november_bounded_evidence[
            "original_reconciled_certificate_failure"
        ]
        == "dual_stationarity_backward_error"
        and november_bounded_evidence["regression_fixture_mode"] == "0600"
        and november_bounded_evidence["regression_fixture_size_bytes"]
        == 20_049_199
        and november_bounded_evidence["regression_test_path"]
        == correction["failure_regression_test_path"]
        and november_bounded_evidence["regression_test_sha256"]
        == correction["failure_regression_test_sha256"]
        and november_bounded_evidence[
            "current_uniform_policy_changes_earliest_accepted_route"
        ]
        is True
        and november_bounded_evidence["thresholds_changed_after_reproduction"]
        is True
        and all(
            november_bounded_evidence[name] is False
            for name in (
                "portfolio_weights_or_methodology_changed",
                "portfolio_performance_metrics_used",
                "stage12_or_later_data_accessed",
            )
        ),
        "Stage 11 November bounded-feasible certificate evidence differs",
    )
    for field, label in (
        ("adapter_sha256", "November emergency-sum adapter"),
        ("compatibility_sha256", "November emergency-sum compatibility"),
        (
            "deterministic_weight_bytes_sha256",
            "November emergency-sum certified weights",
        ),
        ("failure_details_sha256", "November emergency-sum failure details"),
        ("failure_id", "November emergency-sum failure"),
        ("failure_manifest_sha256", "November emergency-sum failure manifest"),
        ("failure_report_sha256", "November emergency-sum failure report"),
        ("failure_success_sha256", "November emergency-sum success receipt"),
        ("problem_bundle_sha256", "November emergency-sum problem bundle"),
        ("regression_fixture_sha256", "November emergency-sum fixture"),
        ("regression_manifest_sha256", "November emergency-sum manifest"),
        ("regression_test_sha256", "November emergency-sum regression test"),
    ):
        _sha(november_sum_evidence[field], label=label)
    november_manifest_relative = Path(
        str(november_sum_evidence["regression_manifest_path"])
    )
    _require(
        not november_manifest_relative.is_absolute()
        and ".." not in november_manifest_relative.parts
        and november_manifest_relative.as_posix()
        == november_sum_evidence["regression_manifest_path"],
        "November emergency-sum regression manifest path is unsafe",
    )
    november_manifest_payload = _stable_file_bytes(
        PROJECT_ROOT.joinpath(*november_manifest_relative.parts),
        label="Stage 11 November emergency-sum regression manifest",
        expected_mode=0o600,
    )
    _require(
        _sha256_bytes(november_manifest_payload)
        == november_sum_evidence["regression_manifest_sha256"],
        "Stage 11 November emergency-sum regression manifest digest differs",
    )
    try:
        november_manifest_value = json.loads(november_manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11CacheRecoveryError(
            "Stage 11 November emergency-sum regression manifest is invalid"
        ) from exc
    november_manifest = _exact_keys(
        november_manifest_value,
        {
            "schema_version",
            "failure_id",
            "holding_month",
            "strategy_id",
            "problem_fixture",
            "failure_authority",
            "original_failure",
            "expected_correction",
            "portfolio_weights_or_methodology_changed",
            "portfolio_performance_metrics_used",
            "stage12_or_later_data_accessed",
        },
        label="Stage 11 November emergency-sum regression manifest",
    )
    november_problem = _exact_keys(
        november_manifest["problem_fixture"],
        {
            "path",
            "sha256",
            "size_bytes",
            "mode",
            "problem_bundle_sha256",
        },
        label="Stage 11 November emergency-sum problem fixture",
    )
    november_failure_authority = _exact_keys(
        november_manifest["failure_authority"],
        {
            "failure_report_path",
            "failure_report_file_sha256",
            "failure_details_path",
            "failure_details_sha256",
            "manifest_path",
            "manifest_sha256",
            "success_path",
            "success_sha256",
            "compatibility_sha256",
            "mode",
        },
        label="Stage 11 November emergency-sum failure authority",
    )
    november_original_failure = _exact_keys(
        november_manifest["original_failure"],
        {
            "exception_class",
            "failure_code",
            "terminal_status_history",
            "rejected_status",
            "independent_lexicographic_certificate_route_previously_present",
        },
        label="Stage 11 November emergency-sum original failure",
    )
    november_correction = _exact_keys(
        november_manifest["expected_correction"],
        {
            "route",
            "n_assets",
            "solver_status",
            "mode",
            "solver_attempt",
            "lexicographic_step_status",
            "certificate_source",
            "dual_status",
            "certificate_residual_tolerance",
            "certificate_gap_tolerance",
            "objective_agreement_tolerance",
            "candidate_maximum_constraint_violation",
            "canonical_primal_residual_maximum",
            "primal_zero_cone_residual_maximum",
            "primal_nonnegative_minimum",
            "primal_soc_minimum_margin",
            "stationarity_absolute_maximum",
            "stationarity_backward_error_maximum",
            "dual_nonnegative_minimum",
            "dual_soc_minimum_margin",
            "candidate_objective",
            "dual_lower_bound",
            "candidate_dual_gap",
            "objective_agreement_maximum_difference",
            "complementarity_absolute",
            "maximum_emergency_constraint_violation_upper_bound",
            "deterministic_weight_bytes_sha256",
            "lexicographic_final_status_history_entry",
            "final_status_history_entry",
            "deterministic_repeat",
        },
        label="Stage 11 November emergency-sum correction evidence",
    )
    _require(
        november_manifest["schema_version"] == 1
        and november_manifest["failure_id"]
        == november_sum_evidence["failure_id"]
        == "bdbff27995f7c21fcab3c48c58422a3a328968f0459d4320d7e18757c4a974da"
        and november_manifest["holding_month"]
        == november_sum_evidence["holding_month"]
        == "2015-11"
        and november_manifest["strategy_id"]
        == november_sum_evidence["strategy_id"]
        == PRIMARY_STRATEGY_ID
        and november_problem
        == {
            "path": november_sum_evidence["regression_fixture_path"],
            "sha256": november_sum_evidence["regression_fixture_sha256"],
            "size_bytes": november_sum_evidence[
                "regression_fixture_size_bytes"
            ],
            "mode": november_sum_evidence["regression_fixture_mode"],
            "problem_bundle_sha256": november_sum_evidence[
                "problem_bundle_sha256"
            ],
        }
        and november_failure_authority
        == {
            "failure_report_path": november_sum_evidence[
                "failure_report_path"
            ],
            "failure_report_file_sha256": november_sum_evidence[
                "failure_report_sha256"
            ],
            "failure_details_path": november_sum_evidence[
                "failure_details_path"
            ],
            "failure_details_sha256": november_sum_evidence[
                "failure_details_sha256"
            ],
            "manifest_path": november_sum_evidence["failure_manifest_path"],
            "manifest_sha256": november_sum_evidence[
                "failure_manifest_sha256"
            ],
            "success_path": november_sum_evidence["failure_success_path"],
            "success_sha256": november_sum_evidence[
                "failure_success_sha256"
            ],
            "compatibility_sha256": november_sum_evidence[
                "compatibility_sha256"
            ],
            "mode": november_sum_evidence["regression_manifest_mode"],
        }
        and november_original_failure["exception_class"]
        == "AllocationSolveError"
        and november_original_failure["failure_code"]
        == "allocation_solver.allocationsolveerror"
        and november_original_failure["terminal_status_history"]
        == november_sum_evidence["original_status_history"]
        and november_original_failure["rejected_status"]
        == november_sum_evidence["lexicographic_step_status"]
        == "optimal_inaccurate"
        and november_original_failure[
            "independent_lexicographic_certificate_route_previously_present"
        ]
        is False
        and november_sum_evidence["adapter_sha256"]
        == correction["adapter_sha256"]
        and november_sum_evidence["regression_test_path"]
        == correction["failure_regression_test_path"]
        and november_sum_evidence["regression_test_sha256"]
        == correction["failure_regression_test_sha256"]
        and november_sum_evidence["lexicographic_step"] == "emergency_sum"
        and november_sum_evidence["certificate_route"]
        == november_correction["route"]
        == "unchanged_emergency_sum_candidate_separately_solved_direct_dual_certificate"
        and november_sum_evidence["n_assets"]
        == november_correction["n_assets"]
        == 1_545
        and november_sum_evidence["solver_status"]
        == november_correction["solver_status"]
        == DUAL_CERTIFIED_SOLVER_STATUS
        and november_sum_evidence["solver_mode"]
        == november_correction["mode"]
        == "emergency"
        and november_sum_evidence["solver_attempt"]
        == november_correction["solver_attempt"]
        == "emergency"
        and november_sum_evidence["certificate_source"]
        == november_correction["certificate_source"]
        == EXPLICIT_DUAL_CERTIFICATE_SOURCE
        and november_sum_evidence["dual_status"]
        == november_correction["dual_status"]
        == "optimal_inaccurate"
        and november_sum_evidence[
            "original_global_certificate_residual_tolerance"
        ]
        == certificate_contract[
            "original_maximum_stationarity_or_cone_residual"
        ]
        and november_sum_evidence[
            "previous_global_certificate_residual_tolerance"
        ]
        == certificate_contract[
            "previous_maximum_stationarity_or_cone_residual"
        ]
        and november_sum_evidence[
            "current_global_certificate_residual_tolerance"
        ]
        == DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and november_sum_evidence["certificate_residual_tolerance"]
        == certificate_contract[
            "previous_maximum_stationarity_or_cone_residual"
        ]
        and abs(
            november_correction["certificate_residual_tolerance"]
            - certificate_contract[
                "previous_maximum_stationarity_or_cone_residual"
            ]
        )
        <= np.spacing(
            certificate_contract[
                "previous_maximum_stationarity_or_cone_residual"
            ]
        )
        and november_sum_evidence["certificate_gap_tolerance"]
        == november_correction["certificate_gap_tolerance"]
        == DUAL_CERTIFICATE_GAP_TOLERANCE
        and november_sum_evidence["objective_agreement_tolerance"]
        == certificate_contract[
            "previous_maximum_objective_cost_scaled_difference"
        ]
        and abs(
            november_correction["objective_agreement_tolerance"]
            - certificate_contract[
                "previous_maximum_objective_cost_scaled_difference"
            ]
        )
        <= np.spacing(
            certificate_contract[
                "previous_maximum_objective_cost_scaled_difference"
            ]
        )
        and november_correction["objective_agreement_tolerance"]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and november_sum_evidence["deterministic_weight_bytes_sha256"]
        == november_correction["deterministic_weight_bytes_sha256"]
        and november_sum_evidence["lexicographic_final_status_history_entry"]
        == november_correction["lexicographic_final_status_history_entry"]
        == "emergency_sum_dual_certificate_audit:passed"
        and november_sum_evidence["final_status_history_entry"]
        == november_correction["final_status_history_entry"]
        == "emergency_economic_dual_certificate_audit:passed"
        and november_sum_evidence["deterministic_repeat"]
        is november_correction["deterministic_repeat"]
        is True
        and november_manifest["portfolio_weights_or_methodology_changed"]
        is november_sum_evidence["portfolio_weights_or_methodology_changed"]
        is False
        and november_manifest["portfolio_performance_metrics_used"]
        is november_sum_evidence["portfolio_performance_metrics_used"]
        is False
        and november_manifest["stage12_or_later_data_accessed"]
        is november_sum_evidence["stage12_or_later_data_accessed"]
        is False,
        "Stage 11 November emergency-sum evidence differs",
    )
    november_correction_metrics = np.asarray(
        [
            november_correction["candidate_maximum_constraint_violation"],
            november_correction["canonical_primal_residual_maximum"],
            november_correction["primal_zero_cone_residual_maximum"],
            november_correction["primal_nonnegative_minimum"],
            november_correction["primal_soc_minimum_margin"],
            november_correction["stationarity_absolute_maximum"],
            november_correction["stationarity_backward_error_maximum"],
            november_correction["dual_nonnegative_minimum"],
            november_correction["dual_soc_minimum_margin"],
            november_correction["candidate_objective"],
            november_correction["dual_lower_bound"],
            november_correction["candidate_dual_gap"],
            november_correction["objective_agreement_maximum_difference"],
            november_correction["complementarity_absolute"],
            november_correction[
                "maximum_emergency_constraint_violation_upper_bound"
            ],
        ],
        dtype=np.float64,
    )
    _require(
        np.isfinite(november_correction_metrics).all()
        and november_correction["candidate_maximum_constraint_violation"]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and november_correction["canonical_primal_residual_maximum"]
        <= november_correction["certificate_residual_tolerance"]
        and november_correction["primal_zero_cone_residual_maximum"]
        <= november_correction["certificate_residual_tolerance"]
        and november_correction["primal_nonnegative_minimum"]
        >= -november_correction["certificate_residual_tolerance"]
        and november_correction["primal_soc_minimum_margin"]
        >= -november_correction["certificate_residual_tolerance"]
        and november_correction["stationarity_absolute_maximum"]
        <= november_correction["certificate_residual_tolerance"]
        and november_correction["stationarity_backward_error_maximum"]
        <= november_correction["certificate_residual_tolerance"]
        and november_correction["dual_nonnegative_minimum"]
        >= -november_correction["certificate_residual_tolerance"]
        and november_correction["dual_soc_minimum_margin"]
        >= -november_correction["certificate_residual_tolerance"]
        and abs(november_correction["candidate_dual_gap"])
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and november_correction["objective_agreement_maximum_difference"]
        <= november_correction["objective_agreement_tolerance"]
        and abs(
            november_correction["candidate_objective"]
            - november_correction["dual_lower_bound"]
            - november_correction["candidate_dual_gap"]
        )
        <= november_correction["objective_agreement_tolerance"]
        and november_correction["complementarity_absolute"]
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and november_correction[
            "maximum_emergency_constraint_violation_upper_bound"
        ]
        >= 0.0,
        "Stage 11 November emergency-sum certificate metrics differ",
    )
    for field, label in (
        ("failure_id", "April 2016 diagnostic failure"),
        ("problem_bundle_sha256", "April 2016 problem bundle"),
        ("problem_file_sha256", "April 2016 ensemble problem"),
    ):
        _sha(april_2016_policy_evidence[field], label=label)
    _require(
        april_2016_policy_evidence["adapter_sha256"]
        == correction["adapter_sha256"]
        and april_2016_policy_evidence["holding_month"] == "2016-04"
        and april_2016_policy_evidence["strategy_id"] == PRIMARY_STRATEGY_ID
        and april_2016_policy_evidence[
            "previous_certificate_residual_tolerance"
        ]
        == certificate_contract[
            "original_maximum_stationarity_or_cone_residual"
        ]
        and april_2016_policy_evidence[
            "current_certificate_residual_tolerance"
        ]
        == certificate_contract[
            "previous_maximum_stationarity_or_cone_residual"
        ]
        and april_2016_policy_evidence[
            "certificate_source_after_uniform_policy"
        ]
        == RETRY_DUAL_CERTIFICATE_SOURCE
        and april_2016_policy_evidence["dual_status"]
        == "optimal_inaccurate"
        and april_2016_policy_evidence["previous_failed_gate"]
        == "dual_soc_cone"
        and april_2016_policy_evidence[
            "certificate_audit_after_uniform_policy"
        ]
        == "passed"
        and april_2016_policy_evidence[
            "candidate_maximum_constraint_violation"
        ]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and april_2016_policy_evidence["stationarity_absolute_maximum"]
        <= april_2016_policy_evidence[
            "current_certificate_residual_tolerance"
        ]
        and april_2016_policy_evidence[
            "stationarity_backward_error_maximum"
        ]
        <= april_2016_policy_evidence[
            "current_certificate_residual_tolerance"
        ]
        and april_2016_policy_evidence["dual_nonnegative_minimum"]
        >= -april_2016_policy_evidence[
            "current_certificate_residual_tolerance"
        ]
        and april_2016_policy_evidence["dual_soc_minimum_margin"]
        < -certificate_contract[
            "original_maximum_stationarity_or_cone_residual"
        ]
        and april_2016_policy_evidence["dual_soc_minimum_margin"]
        >= -april_2016_policy_evidence[
            "current_certificate_residual_tolerance"
        ]
        and abs(april_2016_policy_evidence["candidate_dual_gap_scaled"])
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and april_2016_policy_evidence[
            "objective_cost_scaled_maximum_difference"
        ]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and all(
            april_2016_policy_evidence[name] is False
            for name in (
                "primal_feasibility_tolerance_changed",
                "candidate_dual_gap_tolerance_changed",
                "objective_agreement_tolerance_changed",
                "solver_tolerances_changed",
                "portfolio_weights_changed",
                "portfolio_performance_metrics_used",
                "stage12_or_later_data_accessed",
            )
        ),
        "Stage 11 April 2016 uniform certificate residual policy evidence differs",
    )
    for field, label in (
        ("adapter_sha256", "February 2017 numerical adapter"),
        ("compatibility_sha256", "February 2017 diagnostic compatibility"),
        (
            "deterministic_solver_history_sha256",
            "February 2017 current solver history",
        ),
        (
            "deterministic_weight_bytes_sha256",
            "February 2017 certified weights",
        ),
        ("failure_id", "February 2017 diagnostic failure"),
        ("opening_checkpoint_sha256", "February 2017 opening checkpoint"),
        ("problem_bundle_sha256", "February 2017 problem bundle"),
        ("regression_fixture_sha256", "February 2017 regression fixture"),
        ("regression_manifest_sha256", "February 2017 regression manifest"),
        ("regression_test_sha256", "February 2017 regression test"),
    ):
        _sha(february_2017_policy_evidence[field], label=label)
    february_manifest_relative = Path(
        str(february_2017_policy_evidence["regression_manifest_path"])
    )
    _require(
        not february_manifest_relative.is_absolute()
        and ".." not in february_manifest_relative.parts
        and february_manifest_relative.as_posix()
        == february_2017_policy_evidence["regression_manifest_path"],
        "February 2017 regression manifest path is unsafe",
    )
    february_manifest_payload = _stable_file_bytes(
        PROJECT_ROOT.joinpath(*february_manifest_relative.parts),
        label="Stage 11 February 2017 regression manifest",
        expected_mode=0o600,
    )
    _require(
        _sha256_bytes(february_manifest_payload)
        == february_2017_policy_evidence["regression_manifest_sha256"],
        "Stage 11 February 2017 regression manifest digest differs",
    )
    try:
        february_manifest_value = json.loads(february_manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11CacheRecoveryError(
            "Stage 11 February 2017 regression manifest is invalid"
        ) from exc
    february_manifest = _exact_keys(
        february_manifest_value,
        {
            "schema_version",
            "failure_id",
            "holding_month",
            "strategy_id",
            "problem_fixture",
            "failure_authority",
            "original_failure",
            "expected_correction",
            "portfolio_weights_or_methodology_changed",
            "portfolio_performance_metrics_used",
            "stage12_or_later_data_accessed",
        },
        label="Stage 11 February 2017 regression manifest",
    )
    february_problem = _exact_keys(
        february_manifest["problem_fixture"],
        {
            "path",
            "sha256",
            "size_bytes",
            "mode",
            "problem_bundle_sha256",
        },
        label="Stage 11 February 2017 problem fixture",
    )
    february_failure_authority = _exact_keys(
        february_manifest["failure_authority"],
        {
            "failure_report_path",
            "failure_report_file_sha256",
            "failure_details_path",
            "failure_details_sha256",
            "manifest_path",
            "manifest_sha256",
            "success_path",
            "success_sha256",
            "compatibility_sha256",
            "opening_checkpoint_sha256",
            "mode",
        },
        label="Stage 11 February 2017 failure authority",
    )
    february_original = _exact_keys(
        february_manifest["original_failure"],
        {
            "exception_class",
            "failure_code",
            "immediate_previous_certificate_residual_tolerance",
            "failed_certificate_gates",
            "bounded_certificate_metrics",
        },
        label="Stage 11 February 2017 original failure",
    )
    february_metrics = _exact_keys(
        february_original["bounded_certificate_metrics"],
        {
            "dual_status",
            "certificate_source",
            "stationarity_refinement_stop_code",
            "stationarity_refinement_iterations",
            "soc_minimum_block_ordinal",
            "candidate_maximum_constraint_violation",
            "candidate_maximum_constraint_violation_hex",
            "stationarity_absolute_maximum",
            "stationarity_absolute_maximum_hex",
            "stationarity_backward_error_maximum",
            "stationarity_backward_error_maximum_hex",
            "dual_nonnegative_minimum",
            "dual_nonnegative_minimum_hex",
            "dual_soc_minimum_margin",
            "dual_soc_minimum_margin_hex",
            "candidate_scaled_objective",
            "candidate_scaled_objective_hex",
            "dual_upper_scaled",
            "dual_upper_scaled_hex",
            "candidate_dual_gap_scaled",
            "candidate_dual_gap_scaled_hex",
            "objective_cost_scaled_maximum_difference",
            "objective_cost_scaled_maximum_difference_hex",
            "stationarity_refinement_scale",
            "stationarity_refinement_correction_maximum",
            "dual_reconciliation_retry_weight",
            "dual_coordinate_scale_minimum",
            "dual_coordinate_scale_minimum_hex",
            "dual_coordinate_scale_maximum",
            "dual_coordinate_scale_maximum_hex",
            "stationarity_equation_multiplier",
            "stationarity_equation_multiplier_hex",
            "candidate_gap_band_fraction",
            "candidate_gap_band_fraction_hex",
        },
        label="Stage 11 February 2017 bounded-certificate metrics",
    )
    february_correction = _exact_keys(
        february_manifest["expected_correction"],
        {
            "route",
            "n_assets",
            "solver_status",
            "mode",
            "solver_attempt",
            "certificate_source",
            "certificate_residual_tolerance",
            "immediate_previous_certificate_residual_tolerance",
            "certificate_gap_tolerance",
            "objective_agreement_tolerance",
            "maximum_constraint_violation_upper_bound",
            "deterministic_weight_bytes_sha256",
            "deterministic_solver_history_sha256",
            "current_certificate_metrics",
            "final_status_history_entry",
        },
        label="Stage 11 February 2017 correction evidence",
    )
    february_current_metrics = _exact_keys(
        february_correction["current_certificate_metrics"],
        {
            "dual_status",
            "stationarity_refinement_stop_code",
            "stationarity_refinement_iterations",
            "soc_minimum_block_ordinal",
            "candidate_maximum_constraint_violation",
            "candidate_maximum_constraint_violation_hex",
            "stationarity_absolute_maximum",
            "stationarity_absolute_maximum_hex",
            "stationarity_backward_error_maximum",
            "stationarity_backward_error_maximum_hex",
            "dual_nonnegative_minimum",
            "dual_nonnegative_minimum_hex",
            "dual_soc_minimum_margin",
            "dual_soc_minimum_margin_hex",
            "candidate_scaled_objective",
            "candidate_scaled_objective_hex",
            "dual_upper_scaled",
            "dual_upper_scaled_hex",
            "candidate_dual_gap_scaled",
            "candidate_dual_gap_scaled_hex",
            "objective_cost_scaled_maximum_difference",
            "objective_cost_scaled_maximum_difference_hex",
            "stationarity_refinement_scale",
            "stationarity_refinement_scale_hex",
            "stationarity_refinement_correction_maximum",
            "stationarity_refinement_correction_maximum_hex",
            "dual_reconciliation_retry_weight",
            "dual_reconciliation_retry_weight_hex",
            "dual_coordinate_scale_minimum",
            "dual_coordinate_scale_minimum_hex",
            "dual_coordinate_scale_maximum",
            "dual_coordinate_scale_maximum_hex",
            "stationarity_equation_multiplier",
            "stationarity_equation_multiplier_hex",
            "candidate_gap_band_fraction",
            "candidate_gap_band_fraction_hex",
        },
        label="Stage 11 February 2017 current retry-certificate metrics",
    )
    for field, label in (
        ("failure_report_file_sha256", "February 2017 failure report"),
        ("failure_details_sha256", "February 2017 failure details"),
        ("manifest_sha256", "February 2017 failure manifest"),
        ("success_sha256", "February 2017 success receipt"),
        ("compatibility_sha256", "February 2017 failure compatibility"),
        ("opening_checkpoint_sha256", "February 2017 failure checkpoint"),
    ):
        _sha(february_failure_authority[field], label=label)
    _sha(
        february_correction["deterministic_weight_bytes_sha256"],
        label="February 2017 certified weights",
    )
    _sha(
        february_correction["deterministic_solver_history_sha256"],
        label="February 2017 current solver history",
    )
    for name in (
        "candidate_maximum_constraint_violation",
        "stationarity_absolute_maximum",
        "stationarity_backward_error_maximum",
        "dual_nonnegative_minimum",
        "dual_soc_minimum_margin",
        "candidate_scaled_objective",
        "dual_upper_scaled",
        "candidate_dual_gap_scaled",
        "objective_cost_scaled_maximum_difference",
        "dual_coordinate_scale_minimum",
        "dual_coordinate_scale_maximum",
        "stationarity_equation_multiplier",
        "candidate_gap_band_fraction",
    ):
        _require(
            float.fromhex(str(february_metrics[f"{name}_hex"]))
            == february_metrics[name],
            f"Stage 11 February 2017 {name} encoding differs",
        )
    for name in (
        "candidate_maximum_constraint_violation",
        "stationarity_absolute_maximum",
        "stationarity_backward_error_maximum",
        "dual_nonnegative_minimum",
        "dual_soc_minimum_margin",
        "candidate_scaled_objective",
        "dual_upper_scaled",
        "candidate_dual_gap_scaled",
        "objective_cost_scaled_maximum_difference",
        "stationarity_refinement_scale",
        "stationarity_refinement_correction_maximum",
        "dual_reconciliation_retry_weight",
        "dual_coordinate_scale_minimum",
        "dual_coordinate_scale_maximum",
        "stationarity_equation_multiplier",
        "candidate_gap_band_fraction",
    ):
        _require(
            float.fromhex(str(february_current_metrics[f"{name}_hex"]))
            == february_current_metrics[name],
            f"Stage 11 February 2017 current {name} encoding differs",
        )
    _require(
        february_manifest["schema_version"] == 1
        and february_manifest["failure_id"]
        == february_2017_policy_evidence["failure_id"]
        == "3bf7e10fde4d2d681a72d043d19d8294dc002c8abae93ba6c4901ee9a9073f28"
        and february_manifest["holding_month"]
        == february_2017_policy_evidence["holding_month"]
        == "2017-02"
        and february_manifest["strategy_id"]
        == february_2017_policy_evidence["strategy_id"]
        == PRIMARY_STRATEGY_ID
        and february_2017_policy_evidence["adapter_sha256"]
        == correction["adapter_sha256"]
        and february_2017_policy_evidence["regression_test_path"]
        == correction["failure_regression_test_path"]
        and february_2017_policy_evidence["regression_test_sha256"]
        == correction["failure_regression_test_sha256"]
        and february_problem
        == {
            "path": february_2017_policy_evidence[
                "regression_fixture_path"
            ],
            "sha256": february_2017_policy_evidence[
                "regression_fixture_sha256"
            ],
            "size_bytes": february_2017_policy_evidence[
                "regression_fixture_size_bytes"
            ],
            "mode": february_2017_policy_evidence[
                "regression_fixture_mode"
            ],
            "problem_bundle_sha256": february_2017_policy_evidence[
                "problem_bundle_sha256"
            ],
        }
        and february_failure_authority["compatibility_sha256"]
        == february_2017_policy_evidence["compatibility_sha256"]
        == "32e6f242b123412ccf077ffecbb809b369b2d6942bd73a620106ceaf2a953c53"
        and february_failure_authority["opening_checkpoint_sha256"]
        == february_2017_policy_evidence["opening_checkpoint_sha256"]
        == "5e9fbeb63f29fce0804d14210d064d4df7dc163e811ca33d2f34ef08bb3aca4d"
        and february_failure_authority["mode"]
        == february_2017_policy_evidence["regression_manifest_mode"]
        == "0600"
        and february_original["exception_class"] == "AllocationSolveError"
        and february_original["failure_code"]
        == "allocation_solver.allocationsolveerror"
        and february_original["failed_certificate_gates"]
        == [
            "ordinary_explicit_dual_certificate_audit:failed:candidate_dual_gap",
            "ordinary_retry_dual_certificate_audit:failed:candidate_dual_gap",
            "ordinary_reconciled_dual_certificate_audit:failed:candidate_dual_gap",
            (
                "ordinary_bounded_feasible_dual_certificate_audit:"
                "failed:dual_stationarity_backward_error"
            ),
        ]
        and february_2017_policy_evidence[
            "original_global_certificate_residual_tolerance"
        ]
        == certificate_contract[
            "original_maximum_stationarity_or_cone_residual"
        ]
        and february_2017_policy_evidence[
            "immediate_previous_global_certificate_residual_tolerance"
        ]
        == certificate_contract[
            "previous_maximum_stationarity_or_cone_residual"
        ]
        and february_2017_policy_evidence[
            "current_global_certificate_residual_tolerance"
        ]
        == DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and abs(
            february_original[
                "immediate_previous_certificate_residual_tolerance"
            ]
            - certificate_contract[
                "previous_maximum_stationarity_or_cone_residual"
            ]
        )
        <= np.spacing(
            certificate_contract[
                "previous_maximum_stationarity_or_cone_residual"
            ]
        )
        and february_correction["route"]
        == "uniform_dual_certificate_residual_tolerance"
        and february_correction["n_assets"] == 1_431
        and february_correction["solver_status"]
        == DUAL_CERTIFIED_SOLVER_STATUS
        and february_correction["mode"] == "ordinary"
        and february_correction["solver_attempt"] == "retry"
        and february_correction["certificate_source"]
        == february_2017_policy_evidence["current_certificate_source"]
        == RETRY_DUAL_CERTIFICATE_SOURCE
        and february_metrics["certificate_source"]
        == february_2017_policy_evidence[
            "certificate_source_at_residual_policy_reproduction"
        ]
        == BOUNDED_FEASIBLE_DUAL_CERTIFICATE_SOURCE
        and february_metrics["dual_status"] == "optimal"
        and february_current_metrics["dual_status"]
        == february_2017_policy_evidence["current_dual_status"]
        == "optimal_inaccurate"
        and abs(
            february_correction[
                "immediate_previous_certificate_residual_tolerance"
            ]
            - certificate_contract[
                "previous_maximum_stationarity_or_cone_residual"
            ]
        )
        <= np.spacing(
            certificate_contract[
                "previous_maximum_stationarity_or_cone_residual"
            ]
        )
        and abs(
            february_correction["certificate_residual_tolerance"]
            - DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        )
        <= np.spacing(DUAL_CERTIFICATE_RESIDUAL_TOLERANCE)
        and february_correction["certificate_gap_tolerance"]
        == DUAL_CERTIFICATE_GAP_TOLERANCE
        and february_2017_policy_evidence["certificate_gap_tolerance"]
        == DUAL_CERTIFICATE_GAP_TOLERANCE
        and february_2017_policy_evidence[
            "previous_global_candidate_dual_gap_tolerance"
        ]
        == certificate_contract["previous_maximum_scaled_candidate_dual_gap"]
        == INDEPENDENT_ABSOLUTE_TOLERANCE
        and february_2017_policy_evidence[
            "current_global_candidate_dual_gap_tolerance"
        ]
        == DUAL_CERTIFICATE_GAP_TOLERANCE
        and february_2017_policy_evidence[
            "current_uniform_gap_policy_changes_earliest_accepted_route"
        ]
        is True
        and february_2017_policy_evidence[
            "thresholds_changed_after_reproduction"
        ]
        is True
        and february_2017_policy_evidence[
            "certificate_audit_after_current_uniform_policy"
        ]
        == "passed"
        and abs(
            february_correction["objective_agreement_tolerance"]
            - certificate_contract[
                "previous_maximum_objective_cost_scaled_difference"
            ]
        )
        <= np.spacing(
            certificate_contract[
                "previous_maximum_objective_cost_scaled_difference"
            ]
        )
        and february_correction["objective_agreement_tolerance"]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and february_correction["maximum_constraint_violation_upper_bound"]
        == INDEPENDENT_ABSOLUTE_TOLERANCE
        and february_correction["final_status_history_entry"]
        == "ordinary_dual_certificate_audit:passed"
        and february_metrics["stationarity_refinement_stop_code"] == -1
        and february_metrics["stationarity_refinement_iterations"] == -1
        and february_metrics["soc_minimum_block_ordinal"] == 2_441
        and february_metrics["stationarity_refinement_scale"] == 0.0
        and february_metrics[
            "stationarity_refinement_correction_maximum"
        ]
        == 0.0
        and february_metrics["dual_reconciliation_retry_weight"] == 0.0
        and february_metrics["stationarity_equation_multiplier"]
        == BOUNDED_DUAL_STATIONARITY_MULTIPLIER
        and february_metrics["candidate_gap_band_fraction"]
        == BOUNDED_DUAL_GAP_FRACTION
        and february_metrics["stationarity_backward_error_maximum"]
        == 1.2563970998589815e-09
        and february_original[
            "immediate_previous_certificate_residual_tolerance"
        ]
        < february_metrics["stationarity_backward_error_maximum"]
        <= february_correction["certificate_residual_tolerance"]
        and february_metrics["candidate_maximum_constraint_violation"]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and february_metrics["stationarity_absolute_maximum"]
        <= february_correction["certificate_residual_tolerance"]
        and february_metrics["dual_nonnegative_minimum"]
        >= -february_correction["certificate_residual_tolerance"]
        and february_metrics["dual_soc_minimum_margin"]
        >= -february_correction["certificate_residual_tolerance"]
        and abs(february_metrics["candidate_dual_gap_scaled"])
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and february_metrics[
            "objective_cost_scaled_maximum_difference"
        ]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and february_current_metrics[
            "stationarity_refinement_stop_code"
        ]
        == 1
        and february_current_metrics[
            "stationarity_refinement_iterations"
        ]
        == 32
        and february_current_metrics["soc_minimum_block_ordinal"] == 814
        and february_current_metrics["stationarity_refinement_scale"] == 1.0
        and 0.0
        < february_current_metrics[
            "stationarity_refinement_correction_maximum"
        ]
        <= february_correction["certificate_residual_tolerance"]
        and february_current_metrics["dual_reconciliation_retry_weight"]
        == 0.0
        and february_current_metrics["dual_coordinate_scale_minimum"] == 0.0
        and february_current_metrics["dual_coordinate_scale_maximum"] == 0.0
        and february_current_metrics["stationarity_equation_multiplier"]
        == 0.0
        and february_current_metrics["candidate_gap_band_fraction"] == 0.0
        and february_current_metrics[
            "candidate_maximum_constraint_violation"
        ]
        <= february_correction["maximum_constraint_violation_upper_bound"]
        == INDEPENDENT_ABSOLUTE_TOLERANCE
        and february_current_metrics["stationarity_absolute_maximum"]
        <= february_correction["certificate_residual_tolerance"]
        and february_current_metrics[
            "stationarity_backward_error_maximum"
        ]
        <= february_correction["certificate_residual_tolerance"]
        and february_current_metrics["dual_nonnegative_minimum"]
        >= -february_correction["certificate_residual_tolerance"]
        and february_current_metrics["dual_soc_minimum_margin"]
        >= -february_correction["certificate_residual_tolerance"]
        and abs(february_current_metrics["candidate_dual_gap_scaled"])
        > february_2017_policy_evidence[
            "previous_global_candidate_dual_gap_tolerance"
        ]
        and abs(february_current_metrics["candidate_dual_gap_scaled"])
        <= february_correction["certificate_gap_tolerance"]
        and february_current_metrics["candidate_dual_gap_scaled"]
        == february_2017_policy_evidence["current_candidate_dual_gap_scaled"]
        and february_current_metrics[
            "objective_cost_scaled_maximum_difference"
        ]
        <= february_correction["objective_agreement_tolerance"]
        and abs(
            february_current_metrics["dual_upper_scaled"]
            - february_current_metrics["candidate_scaled_objective"]
            - february_current_metrics["candidate_dual_gap_scaled"]
        )
        <= february_correction["objective_agreement_tolerance"]
        and february_correction["deterministic_weight_bytes_sha256"]
        == february_2017_policy_evidence[
            "deterministic_weight_bytes_sha256"
        ]
        == "ee6eecba46b23e714ab50cb16d63f6d0fb769ff66eb5cae55ad81b2fbf4b2439"
        and february_correction["deterministic_solver_history_sha256"]
        == february_2017_policy_evidence[
            "deterministic_solver_history_sha256"
        ]
        == "e8bfb9beba6cc2317f72ff38eb033947684b89a3428ed641ebad18e7b21ab7e5"
        and february_manifest["portfolio_weights_or_methodology_changed"]
        is february_2017_policy_evidence[
            "portfolio_weights_or_methodology_changed"
        ]
        is False
        and february_manifest["portfolio_performance_metrics_used"]
        is february_2017_policy_evidence[
            "portfolio_performance_metrics_used"
        ]
        is False
        and february_manifest["stage12_or_later_data_accessed"]
        is february_2017_policy_evidence["stage12_or_later_data_accessed"]
        is False,
        "Stage 11 February 2017 uniform residual evidence differs",
    )
    for field, label in (
        ("adapter_sha256", "October 2017 numerical adapter"),
        ("compatibility_sha256", "October 2017 diagnostic compatibility"),
        ("deterministic_solver_history_sha256", "October 2017 solver history"),
        ("deterministic_weight_bytes_sha256", "October 2017 certified weights"),
        ("failure_details_sha256", "October 2017 failure details"),
        ("failure_id", "October 2017 diagnostic failure"),
        ("failure_manifest_sha256", "October 2017 failure manifest"),
        ("failure_report_sha256", "October 2017 failure report"),
        ("failure_success_sha256", "October 2017 success receipt"),
        ("opening_checkpoint_sha256", "October 2017 opening checkpoint"),
        ("problem_bundle_sha256", "October 2017 problem bundle"),
        ("regression_fixture_sha256", "October 2017 regression fixture"),
        ("regression_manifest_sha256", "October 2017 regression manifest"),
        ("regression_test_sha256", "October 2017 regression test"),
    ):
        _sha(october_2017_policy_evidence[field], label=label)
    october_manifest_relative = Path(
        str(october_2017_policy_evidence["regression_manifest_path"])
    )
    _require(
        not october_manifest_relative.is_absolute()
        and ".." not in october_manifest_relative.parts
        and october_manifest_relative.as_posix()
        == october_2017_policy_evidence["regression_manifest_path"],
        "October 2017 regression manifest path is unsafe",
    )
    october_manifest_payload = _stable_file_bytes(
        PROJECT_ROOT.joinpath(*october_manifest_relative.parts),
        label="Stage 11 October 2017 regression manifest",
        expected_mode=0o600,
    )
    _require(
        _sha256_bytes(october_manifest_payload)
        == october_2017_policy_evidence["regression_manifest_sha256"],
        "Stage 11 October 2017 regression manifest digest differs",
    )
    try:
        october_manifest_value = json.loads(october_manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11CacheRecoveryError(
            "Stage 11 October 2017 regression manifest is invalid"
        ) from exc
    october_manifest = _exact_keys(
        october_manifest_value,
        {
            "schema_version",
            "failure_id",
            "holding_month",
            "strategy_id",
            "problem_fixture",
            "failure_authority",
            "original_failure",
            "expected_correction",
            "portfolio_weights_or_methodology_changed",
            "portfolio_performance_metrics_used",
            "stage12_or_later_data_accessed",
        },
        label="Stage 11 October 2017 regression manifest",
    )
    october_problem = _exact_keys(
        october_manifest["problem_fixture"],
        {
            "path",
            "sha256",
            "size_bytes",
            "mode",
            "problem_bundle_sha256",
        },
        label="Stage 11 October 2017 problem fixture",
    )
    october_failure_authority = _exact_keys(
        october_manifest["failure_authority"],
        {
            "failure_report_path",
            "failure_report_file_sha256",
            "failure_details_path",
            "failure_details_sha256",
            "manifest_path",
            "manifest_sha256",
            "success_path",
            "success_sha256",
            "compatibility_sha256",
            "opening_checkpoint_sha256",
            "mode",
        },
        label="Stage 11 October 2017 failure authority",
    )
    october_original = _exact_keys(
        october_manifest["original_failure"],
        {
            "exception_class",
            "failure_code",
            "terminal_status_history",
            "failed_certificate_gates",
            "former_objective_agreement_tolerance",
            "retry_certificate_metrics",
        },
        label="Stage 11 October 2017 original failure",
    )
    october_retry_metrics = _exact_keys(
        october_original["retry_certificate_metrics"],
        {
            "primal_soc_minimum_block_ordinal",
            "dual_soc_minimum_block_ordinal",
            "stationarity_refinement_stop_code",
            "stationarity_refinement_iterations",
            "candidate_maximum_constraint_violation",
            "canonical_primal_residual_maximum",
            "primal_zero_cone_residual_maximum",
            "primal_nonnegative_minimum",
            "primal_soc_minimum_margin",
            "stationarity_absolute_maximum",
            "stationarity_backward_error_maximum",
            "dual_nonnegative_minimum",
            "dual_soc_minimum_margin",
            "candidate_objective",
            "dual_lower_bound",
            "candidate_dual_gap",
            "objective_agreement_maximum_difference",
            "complementarity_absolute",
            "stationarity_refinement_correction_maximum",
        },
        label="Stage 11 October 2017 retry-certificate metrics",
    )
    october_correction = _exact_keys(
        october_manifest["expected_correction"],
        {
            "route",
            "n_assets",
            "solver_status",
            "mode",
            "solver_attempt",
            "lexicographic_step_status",
            "sum_certificate_source",
            "economic_certificate_source",
            "certificate_residual_tolerance",
            "certificate_gap_tolerance",
            "immediate_previous_objective_agreement_tolerance",
            "objective_agreement_tolerance",
            "sum_certificate_metrics",
            "economic_candidate_metrics",
            "economic_certificate_metrics",
            "maximum_emergency_constraint_violation_upper_bound",
            "deterministic_weight_bytes_sha256",
            "deterministic_solver_history_sha256",
            "lexicographic_final_status_history_entry",
            "final_status_history_entry",
            "deterministic_repeat",
        },
        label="Stage 11 October 2017 correction evidence",
    )
    october_sum_metrics = _exact_keys(
        october_correction["sum_certificate_metrics"],
        {
            "primal_soc_minimum_block_ordinal",
            "dual_soc_minimum_block_ordinal",
            "stationarity_refinement_stop_code",
            "stationarity_refinement_iterations",
            "candidate_maximum_constraint_violation",
            "canonical_primal_residual_maximum",
            "primal_zero_cone_residual_maximum",
            "primal_nonnegative_minimum",
            "primal_soc_minimum_margin",
            "stationarity_absolute_maximum",
            "stationarity_backward_error_maximum",
            "dual_nonnegative_minimum",
            "dual_soc_minimum_margin",
            "candidate_objective",
            "dual_lower_bound",
            "candidate_dual_gap",
            "objective_agreement_maximum_difference",
            "complementarity_absolute",
            "stationarity_refinement_correction_maximum",
        },
        label="Stage 11 October 2017 sum-certificate metrics",
    )
    october_economic_candidate = _exact_keys(
        october_correction["economic_candidate_metrics"],
        {
            "maximum_hard_or_noninherited_violation",
            "maximum_inherited_worsening",
            "maximum_invalid_cost_trade",
            "locked_maximum_slack_violation",
            "locked_sum_slack_violation",
            "required_maximum_slack",
            "required_sum_slack",
        },
        label="Stage 11 October 2017 economic candidate metrics",
    )
    october_economic_certificate = _exact_keys(
        october_correction["economic_certificate_metrics"],
        {
            "stationarity_refinement_stop_code",
            "stationarity_refinement_iterations",
            "soc_minimum_block_ordinal",
            "candidate_maximum_constraint_violation",
            "stationarity_absolute_maximum",
            "stationarity_backward_error_maximum",
            "dual_nonnegative_minimum",
            "dual_soc_minimum_margin",
            "candidate_scaled_objective",
            "dual_upper_scaled",
            "candidate_dual_gap_scaled",
            "objective_cost_scaled_maximum_difference",
            "stationarity_refinement_scale",
            "stationarity_refinement_correction_maximum",
            "dual_reconciliation_retry_weight",
            "dual_coordinate_scale_minimum",
            "dual_coordinate_scale_maximum",
            "stationarity_equation_multiplier",
            "candidate_gap_band_fraction",
        },
        label="Stage 11 October 2017 economic certificate metrics",
    )
    october_authority_values: dict[str, Any] = {}
    for key, label in (
        ("failure_report_path", "failure report"),
        ("failure_details_path", "failure details"),
        ("failure_manifest_path", "failure manifest"),
        ("failure_success_path", "success receipt"),
    ):
        relative = Path(str(october_2017_policy_evidence[key]))
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.as_posix() == october_2017_policy_evidence[key],
            f"October 2017 {label} path is unsafe",
        )
        authority_payload = _stable_file_bytes(
            PROJECT_ROOT.joinpath(*relative.parts),
            label=f"Stage 11 October 2017 {label}",
            expected_mode=0o600,
        )
        try:
            value = json.loads(authority_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage11CacheRecoveryError(
                f"Stage 11 October 2017 {label} is invalid"
            ) from exc
        _require(
            authority_payload == _canonical_json(value) + b"\n",
            f"Stage 11 October 2017 {label} is not canonical JSON",
        )
        october_authority_values[key] = value
    october_failure_report = _exact_keys(
        october_authority_values["failure_report_path"],
        {
            "all_active_strategies_attempted_and_inactive_receipted",
            "failure_count",
            "holding_month",
            "holding_month_ordinal",
            "performance_fields_emitted",
            "problem_bundle_sha256",
            "protocol_id",
            "schema_version",
            "strategy_results",
        },
        label="Stage 11 October 2017 failure report",
    )
    _require(
        isinstance(october_failure_report["strategy_results"], list)
        and len(october_failure_report["strategy_results"]) == 3,
        "Stage 11 October 2017 strategy failure results differ",
    )
    october_failed_strategy = _exact_keys(
        october_failure_report["strategy_results"][0],
        {
            "exception_class",
            "failure_code",
            "failure_id",
            "holding_month",
            "private_detail_sha256",
            "problem_bundle_sha256",
            "status",
            "strategy_id",
        },
        label="Stage 11 October 2017 failed strategy",
    )
    october_private_details = _exact_keys(
        october_authority_values["failure_details_path"],
        {"details", "schema_version", "visibility"},
        label="Stage 11 October 2017 private failure details",
    )
    _require(
        isinstance(october_private_details["details"], list)
        and len(october_private_details["details"]) == 1,
        "Stage 11 October 2017 private failure detail count differs",
    )
    october_private_detail = _exact_keys(
        october_private_details["details"][0],
        {"private_detail_hex", "private_detail_sha256", "strategy_id"},
        label="Stage 11 October 2017 private failure detail",
    )
    try:
        october_private_detail_bytes = bytes.fromhex(
            str(october_private_detail["private_detail_hex"])
        )
        october_private_detail_value = json.loads(october_private_detail_bytes)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11CacheRecoveryError(
            "Stage 11 October 2017 private failure detail is invalid"
        ) from exc
    october_decoded_detail = _exact_keys(
        october_private_detail_value,
        {
            "allocator_architecture",
            "error_message",
            "error_type",
            "strategy_id",
        },
        label="Stage 11 October 2017 decoded private failure detail",
    )
    october_authority_manifest = _exact_keys(
        october_authority_values["failure_manifest_path"],
        {
            "all_active_strategies_attempted_and_inactive_receipted",
            "boundary",
            "checkpoint_advanced",
            "compatibility",
            "compatibility_sha256",
            "failure_report",
            "holding_month",
            "holding_month_ordinal",
            "kind",
            "opening_checkpoint_sha256",
            "performance_fields_emitted",
            "private_failure_details",
            "problem_bundle_sha256",
            "producer_implementation_sha256",
            "protocol_id",
            "schema_version",
        },
        label="Stage 11 October 2017 authority manifest",
    )
    october_boundary = _exact_keys(
        october_authority_manifest["boundary"],
        {
            "final_period_rows_read",
            "first_holding_month",
            "last_holding_month",
            "latest_permitted_observation_date",
            "stage",
            "stage12_access_authorized",
            "stage12_rows_read",
        },
        label="Stage 11 October 2017 authority boundary",
    )
    october_compatibility = _exact_keys(
        october_authority_manifest["compatibility"],
        {
            "accounting_transition_code_sha256",
            "allocation_problem_schema_sha256",
            "checkpoint_state_schema_sha256",
            "diagnostic_workflow_sha256",
            "execution_input_authority_sha256",
            "gate_b_sha256",
            "methodology_sha256",
            "preflight_sha256",
            "validation_calendar_sha256",
        },
        label="Stage 11 October 2017 authority compatibility",
    )
    october_failure_report_reference = _exact_keys(
        october_authority_manifest["failure_report"],
        {"failure_count", "failure_ids", "file_name", "sha256"},
        label="Stage 11 October 2017 failure report reference",
    )
    october_private_details_reference = _exact_keys(
        october_authority_manifest["private_failure_details"],
        {"file_name", "sha256", "size_bytes", "visibility"},
        label="Stage 11 October 2017 private details reference",
    )
    october_success = _exact_keys(
        october_authority_values["failure_success_path"],
        {
            "checkpoint_advanced",
            "failure_count",
            "failure_report_sha256",
            "manifest_sha256",
            "performance_fields_emitted",
            "schema_version",
            "status",
        },
        label="Stage 11 October 2017 success receipt",
    )
    for name, value in october_compatibility.items():
        _sha(value, label=f"October 2017 {name}")
    _sha(
        october_authority_manifest["producer_implementation_sha256"],
        label="October 2017 failure producer",
    )
    _require(
        october_manifest["schema_version"] == 1
        and october_manifest["failure_id"]
        == october_2017_policy_evidence["failure_id"]
        == "823f546f49a5a0fc6a140ea372225f27240f61e71849bd091deff4ec8970a312"
        and october_manifest["holding_month"]
        == october_2017_policy_evidence["holding_month"]
        == "2017-10"
        and october_manifest["strategy_id"]
        == october_2017_policy_evidence["strategy_id"]
        == PRIMARY_STRATEGY_ID
        and october_2017_policy_evidence["adapter_sha256"]
        == correction["adapter_sha256"]
        and october_2017_policy_evidence["regression_test_path"]
        == correction["failure_regression_test_path"]
        and october_2017_policy_evidence["regression_test_sha256"]
        == correction["failure_regression_test_sha256"]
        and october_problem
        == {
            "path": october_2017_policy_evidence["regression_fixture_path"],
            "sha256": october_2017_policy_evidence[
                "regression_fixture_sha256"
            ],
            "size_bytes": october_2017_policy_evidence[
                "regression_fixture_size_bytes"
            ],
            "mode": october_2017_policy_evidence["regression_fixture_mode"],
            "problem_bundle_sha256": october_2017_policy_evidence[
                "problem_bundle_sha256"
            ],
        }
        and october_failure_authority
        == {
            "failure_report_path": october_2017_policy_evidence[
                "failure_report_path"
            ],
            "failure_report_file_sha256": october_2017_policy_evidence[
                "failure_report_sha256"
            ],
            "failure_details_path": october_2017_policy_evidence[
                "failure_details_path"
            ],
            "failure_details_sha256": october_2017_policy_evidence[
                "failure_details_sha256"
            ],
            "manifest_path": october_2017_policy_evidence[
                "failure_manifest_path"
            ],
            "manifest_sha256": october_2017_policy_evidence[
                "failure_manifest_sha256"
            ],
            "success_path": october_2017_policy_evidence[
                "failure_success_path"
            ],
            "success_sha256": october_2017_policy_evidence[
                "failure_success_sha256"
            ],
            "compatibility_sha256": october_2017_policy_evidence[
                "compatibility_sha256"
            ],
            "opening_checkpoint_sha256": october_2017_policy_evidence[
                "opening_checkpoint_sha256"
            ],
            "mode": october_2017_policy_evidence["failure_authority_mode"],
        }
        and october_failure_authority["compatibility_sha256"]
        == "32e6f242b123412ccf077ffecbb809b369b2d6942bd73a620106ceaf2a953c53"
        and october_failure_authority["opening_checkpoint_sha256"]
        == "0a0e3b2ef6434582753a7cefca87d9da7d2fab2f9544529ff4345a0cdc307d05"
        and october_failure_authority["mode"]
        == october_2017_policy_evidence["failure_authority_mode"]
        == "0600"
        and october_2017_policy_evidence["regression_manifest_mode"]
        == "0600"
        and october_original["exception_class"] == "AllocationSolveError"
        and october_original["failure_code"]
        == "allocation_solver.allocationsolveerror"
        and october_original["terminal_status_history"]
        == [
            "ordinary_primary:solver_error:SolverError",
            "ordinary_retry:solver_error:SolverError",
            "ordinary_feasibility_retry:infeasible",
            "emergency_minimax:optimal",
            "emergency_sum:optimal_inaccurate",
        ]
        and october_original["failed_certificate_gates"]
        == [
            "emergency_sum_explicit_dual_certificate_audit:failed:candidate_dual_gap",
            "emergency_sum_retry_dual_certificate_audit:failed:objective_agreement",
        ]
        and october_original["former_objective_agreement_tolerance"]
        == certificate_contract[
            "previous_maximum_objective_cost_scaled_difference"
        ]
        and october_retry_metrics["objective_agreement_maximum_difference"]
        == october_2017_policy_evidence[
            "objective_agreement_maximum_difference"
        ]
        and october_retry_metrics["objective_agreement_maximum_difference"]
        > certificate_contract[
            "previous_maximum_objective_cost_scaled_difference"
        ]
        and october_retry_metrics["objective_agreement_maximum_difference"]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and october_retry_metrics["candidate_maximum_constraint_violation"]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and october_retry_metrics["canonical_primal_residual_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_retry_metrics["stationarity_absolute_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_retry_metrics["stationarity_backward_error_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_retry_metrics["dual_nonnegative_minimum"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_retry_metrics["dual_soc_minimum_margin"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and abs(october_retry_metrics["candidate_dual_gap"])
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and october_2017_policy_evidence[
            "certificate_source_at_objective_agreement_policy_reproduction"
        ]
        == RETRY_DUAL_CERTIFICATE_SOURCE
        and october_2017_policy_evidence["historical_retry_dual_status"]
        == "optimal_inaccurate"
        and october_sum_metrics["candidate_maximum_constraint_violation"]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and october_sum_metrics["canonical_primal_residual_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_sum_metrics["primal_zero_cone_residual_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_sum_metrics["primal_nonnegative_minimum"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_sum_metrics["primal_soc_minimum_margin"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_sum_metrics["stationarity_absolute_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_sum_metrics["stationarity_backward_error_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_sum_metrics["dual_nonnegative_minimum"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_sum_metrics["dual_soc_minimum_margin"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and abs(october_sum_metrics["candidate_dual_gap"])
        > october_2017_policy_evidence[
            "previous_global_candidate_dual_gap_tolerance"
        ]
        and abs(october_sum_metrics["candidate_dual_gap"])
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and october_sum_metrics["candidate_dual_gap"]
        == october_2017_policy_evidence["current_sum_candidate_dual_gap"]
        and october_sum_metrics["objective_agreement_maximum_difference"]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and october_sum_metrics["complementarity_absolute"]
        == october_2017_policy_evidence[
            "current_sum_complementarity_absolute"
        ]
        and october_sum_metrics["complementarity_absolute"]
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and abs(
            october_sum_metrics["candidate_objective"]
            - october_sum_metrics["dual_lower_bound"]
            - october_sum_metrics["candidate_dual_gap"]
        )
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and october_correction["route"]
        == "uniform_dual_certificate_objective_agreement_tolerance"
        and october_correction["n_assets"] == 1_539
        and october_correction["solver_status"]
        == DUAL_CERTIFIED_SOLVER_STATUS
        and october_correction["mode"] == "emergency"
        and october_correction["solver_attempt"] == "emergency"
        and october_correction["lexicographic_step_status"]
        == "optimal_inaccurate"
        and october_correction["sum_certificate_source"]
        == october_2017_policy_evidence[
            "current_sum_certificate_source"
        ]
        == EXPLICIT_DUAL_CERTIFICATE_SOURCE
        and october_correction["economic_certificate_source"]
        == october_2017_policy_evidence[
            "current_economic_certificate_source"
        ]
        == RETRY_DUAL_CERTIFICATE_SOURCE
        and abs(
            october_correction["certificate_residual_tolerance"]
            - DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        )
        <= np.spacing(DUAL_CERTIFICATE_RESIDUAL_TOLERANCE)
        and october_correction["certificate_gap_tolerance"]
        == DUAL_CERTIFICATE_GAP_TOLERANCE
        and october_correction[
            "immediate_previous_objective_agreement_tolerance"
        ]
        == certificate_contract[
            "previous_maximum_objective_cost_scaled_difference"
        ]
        and october_correction["objective_agreement_tolerance"]
        == DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and october_2017_policy_evidence[
            "immediate_previous_global_objective_agreement_tolerance"
        ]
        == certificate_contract[
            "previous_maximum_objective_cost_scaled_difference"
        ]
        and october_2017_policy_evidence[
            "current_global_objective_agreement_tolerance"
        ]
        == DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and october_2017_policy_evidence[
            "uniform_policy_multiplier_of_retry_gap_absolute"
        ]
        == certificate_contract[
            "objective_agreement_tolerance_multiplier_of_retry_gap_absolute"
        ]
        == 50.0
        and october_2017_policy_evidence[
            "certificate_residual_tolerance"
        ]
        == DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_2017_policy_evidence["certificate_gap_tolerance"]
        == DUAL_CERTIFICATE_GAP_TOLERANCE
        and october_2017_policy_evidence[
            "previous_global_candidate_dual_gap_tolerance"
        ]
        == certificate_contract["previous_maximum_scaled_candidate_dual_gap"]
        == INDEPENDENT_ABSOLUTE_TOLERANCE
        and october_2017_policy_evidence[
            "current_global_candidate_dual_gap_tolerance"
        ]
        == DUAL_CERTIFICATE_GAP_TOLERANCE
        and october_2017_policy_evidence[
            "current_uniform_gap_policy_changes_earliest_sum_route"
        ]
        is True
        and october_2017_policy_evidence[
            "thresholds_changed_after_reproduction"
        ]
        is True
        and october_2017_policy_evidence[
            "certificate_audit_at_objective_agreement_policy_reproduction"
        ]
        == "passed"
        and october_economic_candidate[
            "maximum_hard_or_noninherited_violation"
        ]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and october_economic_candidate["maximum_inherited_worsening"] == 0.0
        and october_economic_candidate["maximum_invalid_cost_trade"] == 0.0
        and october_economic_certificate[
            "candidate_maximum_constraint_violation"
        ]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and october_economic_certificate["stationarity_absolute_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_economic_certificate[
            "stationarity_backward_error_maximum"
        ]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_economic_certificate["dual_nonnegative_minimum"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and october_economic_certificate["dual_soc_minimum_margin"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and abs(october_economic_certificate["candidate_dual_gap_scaled"])
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and october_economic_certificate[
            "objective_cost_scaled_maximum_difference"
        ]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and october_correction["deterministic_weight_bytes_sha256"]
        == october_2017_policy_evidence[
            "deterministic_weight_bytes_sha256"
        ]
        == "b61f2763e683cacad3abb2494e427032856599c93e1479830bb4ed0c77897f3d"
        and october_correction["deterministic_solver_history_sha256"]
        == october_2017_policy_evidence[
            "deterministic_solver_history_sha256"
        ]
        == "c9f34974f4f8dfb4bc24e0897cfec14c2fec4fdd15501eee28a7f3d5af9d3091"
        and october_correction["lexicographic_final_status_history_entry"]
        == "emergency_sum_dual_certificate_audit:passed"
        and october_correction["final_status_history_entry"]
        == "emergency_economic_dual_certificate_audit:passed"
        and october_correction["deterministic_repeat"] is True
        and october_failure_report["schema_version"] == 1
        and october_failure_report["protocol_id"]
        == "stage11-authenticated-diagnostic-persistence-v1"
        and october_failure_report["holding_month"] == "2017-10"
        and october_failure_report["holding_month_ordinal"] == 34
        and october_failure_report["failure_count"] == 1
        and october_failure_report["performance_fields_emitted"] == 0
        and october_failure_report[
            "all_active_strategies_attempted_and_inactive_receipted"
        ]
        is True
        and october_failure_report["problem_bundle_sha256"]
        == october_2017_policy_evidence["problem_bundle_sha256"]
        and october_failed_strategy["status"] == "failed"
        and october_failed_strategy["strategy_id"] == PRIMARY_STRATEGY_ID
        and october_failed_strategy["holding_month"] == "2017-10"
        and october_failed_strategy["exception_class"] == "AllocationSolveError"
        and october_failed_strategy["failure_code"]
        == "allocation_solver.allocationsolveerror"
        and october_failed_strategy["failure_id"]
        == october_2017_policy_evidence["failure_id"]
        and october_failed_strategy["problem_bundle_sha256"]
        == october_2017_policy_evidence["problem_bundle_sha256"]
        and october_failure_report["strategy_results"][1]
        == {
            "status": "passed",
            "strategy_id": "elastic_net_hybrid_challenger",
        }
        and october_failure_report["strategy_results"][2]
        == {
            "status": "passed",
            "strategy_id": "ensemble_rank_challenger",
        }
        and october_private_details["schema_version"] == 1
        and october_private_details["visibility"] == "private_mode_0600"
        and october_private_detail["strategy_id"] == PRIMARY_STRATEGY_ID
        and october_private_detail["private_detail_sha256"]
        == october_failed_strategy["private_detail_sha256"]
        == _sha256_bytes(october_private_detail_bytes)
        and october_private_detail_bytes
        == _canonical_json(october_private_detail_value) + b"\n"
        and october_decoded_detail["allocator_architecture"]
        == "hybrid_alpha_risk_cost_convex"
        and october_decoded_detail["error_type"] == "AllocationSolveError"
        and october_decoded_detail["strategy_id"] == PRIMARY_STRATEGY_ID
        and str(october_decoded_detail["error_message"]).startswith(
            "emergency sum-slack step was not independently certified:"
        )
        and (
            "emergency_sum_explicit_dual_certificate_audit:"
            "failed:candidate_dual_gap"
        )
        in october_decoded_detail["error_message"]
        and (
            "emergency_sum_retry_dual_certificate_audit:"
            "failed:objective_agreement"
        )
        in october_decoded_detail["error_message"]
        and october_authority_manifest["schema_version"] == 1
        and october_authority_manifest["protocol_id"]
        == "stage11-authenticated-diagnostic-persistence-v1"
        and october_authority_manifest["kind"]
        == "stage11_aggregate_solver_failure_report"
        and october_authority_manifest["holding_month"] == "2017-10"
        and october_authority_manifest["holding_month_ordinal"] == 34
        and october_authority_manifest["checkpoint_advanced"] is False
        and october_authority_manifest["performance_fields_emitted"] == 0
        and october_authority_manifest[
            "all_active_strategies_attempted_and_inactive_receipted"
        ]
        is True
        and october_authority_manifest["problem_bundle_sha256"]
        == october_2017_policy_evidence["problem_bundle_sha256"]
        and october_authority_manifest["compatibility_sha256"]
        == october_2017_policy_evidence["compatibility_sha256"]
        and october_authority_manifest["opening_checkpoint_sha256"]
        == october_2017_policy_evidence["opening_checkpoint_sha256"]
        and october_boundary
        == {
            "final_period_rows_read": 0,
            "first_holding_month": "2015-01",
            "last_holding_month": "2019-11",
            "latest_permitted_observation_date": "2019-12-02",
            "stage": 11,
            "stage12_access_authorized": False,
            "stage12_rows_read": 0,
        }
        and october_failure_report_reference
        == {
            "failure_count": 1,
            "failure_ids": [october_2017_policy_evidence["failure_id"]],
            "file_name": "failure_report.json",
            "sha256": october_2017_policy_evidence["failure_report_sha256"],
        }
        and october_private_details_reference
        == {
            "file_name": "failure_details.json",
            "sha256": october_2017_policy_evidence["failure_details_sha256"],
            "size_bytes": 8_583,
            "visibility": "private_mode_0600",
        }
        and october_success
        == {
            "checkpoint_advanced": False,
            "failure_count": 1,
            "failure_report_sha256": october_2017_policy_evidence[
                "failure_report_sha256"
            ],
            "manifest_sha256": october_2017_policy_evidence[
                "failure_manifest_sha256"
            ],
            "performance_fields_emitted": 0,
            "schema_version": 1,
            "status": "ALL_STRATEGY_FAILURES_REPORTED_TOGETHER",
        }
        and october_manifest["portfolio_weights_or_methodology_changed"]
        is october_2017_policy_evidence[
            "portfolio_weights_or_methodology_changed"
        ]
        is False
        and october_manifest["portfolio_performance_metrics_used"]
        is october_2017_policy_evidence["portfolio_performance_metrics_used"]
        is False
        and october_manifest["stage12_or_later_data_accessed"]
        is october_2017_policy_evidence["stage12_or_later_data_accessed"]
        is False,
        "Stage 11 October 2017 uniform objective-agreement evidence differs",
    )
    for field, label in (
        ("adapter_sha256", "June 2018 numerical adapter"),
        ("compatibility_sha256", "June 2018 diagnostic compatibility"),
        ("deterministic_solver_history_sha256", "June 2018 solver history"),
        ("deterministic_weight_bytes_sha256", "June 2018 certified weights"),
        ("failure_details_sha256", "June 2018 failure details"),
        ("failure_id", "June 2018 diagnostic failure"),
        ("failure_manifest_sha256", "June 2018 failure manifest"),
        ("failure_report_sha256", "June 2018 failure report"),
        ("failure_success_sha256", "June 2018 success receipt"),
        ("opening_checkpoint_sha256", "June 2018 opening checkpoint"),
        ("problem_bundle_sha256", "June 2018 problem bundle"),
        ("regression_fixture_sha256", "June 2018 regression fixture"),
        ("regression_manifest_sha256", "June 2018 regression manifest"),
        ("regression_test_sha256", "June 2018 regression test"),
    ):
        _sha(june_2018_routing_evidence[field], label=label)
    june_manifest_relative = Path(
        str(june_2018_routing_evidence["regression_manifest_path"])
    )
    _require(
        not june_manifest_relative.is_absolute()
        and ".." not in june_manifest_relative.parts
        and june_manifest_relative.as_posix()
        == june_2018_routing_evidence["regression_manifest_path"],
        "June 2018 regression manifest path is unsafe",
    )
    june_manifest_payload = _stable_file_bytes(
        PROJECT_ROOT.joinpath(*june_manifest_relative.parts),
        label="Stage 11 June 2018 regression manifest",
        expected_mode=0o600,
    )
    _require(
        _sha256_bytes(june_manifest_payload)
        == june_2018_routing_evidence["regression_manifest_sha256"],
        "Stage 11 June 2018 regression manifest digest differs",
    )
    try:
        june_manifest_value = json.loads(june_manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11CacheRecoveryError(
            "Stage 11 June 2018 regression manifest is invalid"
        ) from exc
    june_manifest = _exact_keys(
        june_manifest_value,
        {
            "schema_version",
            "failure_id",
            "holding_month",
            "strategy_id",
            "problem_fixture",
            "failure_authority",
            "original_failure",
            "expected_correction",
            "portfolio_weights_or_methodology_changed",
            "portfolio_performance_metrics_used",
            "stage12_or_later_data_accessed",
        },
        label="Stage 11 June 2018 regression manifest",
    )
    june_problem = _exact_keys(
        june_manifest["problem_fixture"],
        {"path", "sha256", "size_bytes", "mode", "problem_bundle_sha256"},
        label="Stage 11 June 2018 problem fixture",
    )
    june_failure_authority = _exact_keys(
        june_manifest["failure_authority"],
        {
            "failure_report_path",
            "failure_report_file_sha256",
            "failure_details_path",
            "failure_details_sha256",
            "manifest_path",
            "manifest_sha256",
            "success_path",
            "success_sha256",
            "compatibility_sha256",
            "opening_checkpoint_sha256",
            "mode",
        },
        label="Stage 11 June 2018 failure authority",
    )
    june_original = _exact_keys(
        june_manifest["original_failure"],
        {
            "exception_class",
            "failure_code",
            "terminal_status_history",
            "blocked_route",
        },
        label="Stage 11 June 2018 original failure",
    )
    june_correction = _exact_keys(
        june_manifest["expected_correction"],
        {
            "route",
            "adapter_sha256",
            "n_assets",
            "solver_status",
            "mode",
            "solver_attempt",
            "feasibility_status",
            "feasibility_status_alone_permits_certification",
            "certificate_source",
            "certificate_residual_tolerance",
            "certificate_gap_tolerance",
            "objective_agreement_tolerance",
            "certificate_metrics",
            "deterministic_weight_bytes_sha256",
            "deterministic_solver_history_sha256",
            "final_status_history_entry",
            "deterministic_repeat",
        },
        label="Stage 11 June 2018 correction evidence",
    )
    june_metrics = _exact_keys(
        june_correction["certificate_metrics"],
        {
            "dual_status",
            "stationarity_refinement_stop_code",
            "stationarity_refinement_iterations",
            "soc_minimum_block_ordinal",
            "candidate_maximum_constraint_violation",
            "candidate_maximum_constraint_violation_hex",
            "stationarity_absolute_maximum",
            "stationarity_absolute_maximum_hex",
            "stationarity_backward_error_maximum",
            "stationarity_backward_error_maximum_hex",
            "dual_nonnegative_minimum",
            "dual_nonnegative_minimum_hex",
            "dual_soc_minimum_margin",
            "dual_soc_minimum_margin_hex",
            "candidate_scaled_objective",
            "candidate_scaled_objective_hex",
            "dual_upper_scaled",
            "dual_upper_scaled_hex",
            "candidate_dual_gap_scaled",
            "candidate_dual_gap_scaled_hex",
            "objective_cost_scaled_maximum_difference",
            "objective_cost_scaled_maximum_difference_hex",
        },
        label="Stage 11 June 2018 certificate metrics",
    )
    june_authority_values: dict[str, Any] = {}
    for path_key, label in (
        ("failure_report_path", "failure report"),
        ("failure_details_path", "failure details"),
        ("failure_manifest_path", "failure manifest"),
        ("failure_success_path", "success receipt"),
    ):
        relative = Path(str(june_2018_routing_evidence[path_key]))
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.as_posix() == june_2018_routing_evidence[path_key],
            f"June 2018 {label} path is unsafe",
        )
        authority_payload = _stable_file_bytes(
            PROJECT_ROOT.joinpath(*relative.parts),
            label=f"Stage 11 June 2018 {label}",
            expected_mode=0o600,
        )
        try:
            authority_value = json.loads(authority_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage11CacheRecoveryError(
                f"Stage 11 June 2018 {label} is invalid"
            ) from exc
        _require(
            authority_payload == _canonical_json(authority_value) + b"\n",
            f"Stage 11 June 2018 {label} is not canonical JSON",
        )
        june_authority_values[path_key] = authority_value
    june_failure_report = june_authority_values["failure_report_path"]
    june_failure_details = june_authority_values["failure_details_path"]
    june_authority_manifest = june_authority_values["failure_manifest_path"]
    june_success = june_authority_values["failure_success_path"]
    _require(
        isinstance(june_failure_report, Mapping)
        and isinstance(june_failure_report.get("strategy_results"), list)
        and len(june_failure_report["strategy_results"]) == 3,
        "Stage 11 June 2018 failure report semantics differ",
    )
    june_failed_rows = [
        row
        for row in june_failure_report["strategy_results"]
        if isinstance(row, Mapping) and row.get("status") == "failed"
    ]
    _require(
        isinstance(june_failure_details, Mapping)
        and isinstance(june_failure_details.get("details"), list)
        and len(june_failure_details["details"]) == 1,
        "Stage 11 June 2018 private failure details differ",
    )
    june_private_row = june_failure_details["details"][0]
    try:
        june_private_bytes = bytes.fromhex(
            str(june_private_row["private_detail_hex"])
        )
        june_private_value = json.loads(june_private_bytes)
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11CacheRecoveryError(
            "Stage 11 June 2018 private failure detail is invalid"
        ) from exc
    for name in (
        "candidate_maximum_constraint_violation",
        "stationarity_absolute_maximum",
        "stationarity_backward_error_maximum",
        "dual_nonnegative_minimum",
        "dual_soc_minimum_margin",
        "candidate_scaled_objective",
        "dual_upper_scaled",
        "candidate_dual_gap_scaled",
        "objective_cost_scaled_maximum_difference",
    ):
        _require(
            float.fromhex(str(june_metrics[f"{name}_hex"]))
            == june_metrics[name],
            f"Stage 11 June 2018 {name} encoding differs",
        )
    _require(
        june_manifest["schema_version"] == 1
        and june_manifest["failure_id"]
        == june_2018_routing_evidence["failure_id"]
        == "3685058a0356c30573436850654a35b9956c05cba1918a67ae96f38cb5e34e9f"
        and june_manifest["holding_month"]
        == june_2018_routing_evidence["holding_month"]
        == "2018-06"
        and june_manifest["strategy_id"]
        == june_2018_routing_evidence["strategy_id"]
        == ELASTIC_NET_CHALLENGER_ID
        and june_2018_routing_evidence["adapter_sha256"]
        == june_correction["adapter_sha256"]
        == correction["adapter_sha256"]
        and june_2018_routing_evidence["regression_test_path"]
        == correction["failure_regression_test_path"]
        and june_2018_routing_evidence["regression_test_sha256"]
        == correction["failure_regression_test_sha256"]
        and june_problem
        == {
            "path": june_2018_routing_evidence["regression_fixture_path"],
            "sha256": june_2018_routing_evidence["regression_fixture_sha256"],
            "size_bytes": june_2018_routing_evidence[
                "regression_fixture_size_bytes"
            ],
            "mode": june_2018_routing_evidence["regression_fixture_mode"],
            "problem_bundle_sha256": june_2018_routing_evidence[
                "problem_bundle_sha256"
            ],
        }
        and june_failure_authority
        == {
            "failure_report_path": june_2018_routing_evidence[
                "failure_report_path"
            ],
            "failure_report_file_sha256": june_2018_routing_evidence[
                "failure_report_sha256"
            ],
            "failure_details_path": june_2018_routing_evidence[
                "failure_details_path"
            ],
            "failure_details_sha256": june_2018_routing_evidence[
                "failure_details_sha256"
            ],
            "manifest_path": june_2018_routing_evidence[
                "failure_manifest_path"
            ],
            "manifest_sha256": june_2018_routing_evidence[
                "failure_manifest_sha256"
            ],
            "success_path": june_2018_routing_evidence[
                "failure_success_path"
            ],
            "success_sha256": june_2018_routing_evidence[
                "failure_success_sha256"
            ],
            "compatibility_sha256": june_2018_routing_evidence[
                "compatibility_sha256"
            ],
            "opening_checkpoint_sha256": june_2018_routing_evidence[
                "opening_checkpoint_sha256"
            ],
            "mode": june_2018_routing_evidence["failure_authority_mode"],
        }
        and june_failure_authority["compatibility_sha256"]
        == "32e6f242b123412ccf077ffecbb809b369b2d6942bd73a620106ceaf2a953c53"
        and june_failure_authority["opening_checkpoint_sha256"]
        == "fa249c58f16c4f88eeb9443764501c557232c06db3e4ea6e4cb02d679349efc6"
        and june_failure_authority["mode"]
        == june_2018_routing_evidence["regression_manifest_mode"]
        == "0600"
        and june_original["exception_class"] == "AllocationSolveError"
        and june_original["failure_code"]
        == "allocation_solver.allocationsolveerror"
        and june_original["terminal_status_history"]
        == [
            "ordinary_primary:optimal_inaccurate",
            "ordinary_retry:optimal_inaccurate",
            "ordinary_feasibility_retry:optimal_inaccurate",
        ]
        and june_original["blocked_route"]
        == (
            "ordinary_feasibility_optimal_inaccurate_was_not_eligible_"
            "for_full_primal_and_dual_certificate_audits"
        )
        and june_correction["route"]
        == "ordinary_inaccurate_feasibility_status_to_full_certificate_audits"
        and june_correction["n_assets"]
        == june_2018_routing_evidence["n_assets"]
        == 1_536
        and june_correction["solver_status"]
        == june_2018_routing_evidence["solver_status_after_routing"]
        == DUAL_CERTIFIED_SOLVER_STATUS
        and june_correction["mode"] == "ordinary"
        and june_correction["solver_attempt"] == "retry"
        and june_correction["feasibility_status"]
        == june_2018_routing_evidence["feasibility_solver_status"]
        == "optimal_inaccurate"
        and june_correction["feasibility_status_alone_permits_certification"]
        is june_2018_routing_evidence[
            "feasibility_status_alone_permits_certification"
        ]
        is False
        and june_2018_routing_evidence[
            "full_primal_and_dual_audits_required"
        ]
        is True
        and june_2018_routing_evidence["emergency_entry_rule_changed"] is False
        and june_correction["certificate_source"]
        == june_2018_routing_evidence[
            "certificate_source_after_status_routing"
        ]
        == EXPLICIT_DUAL_CERTIFICATE_SOURCE
        and june_metrics["dual_status"]
        == june_2018_routing_evidence["dual_status"]
        == "optimal_inaccurate"
        and abs(
            june_correction["certificate_residual_tolerance"]
            - DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        )
        <= np.spacing(DUAL_CERTIFICATE_RESIDUAL_TOLERANCE)
        and june_correction["certificate_gap_tolerance"]
        == DUAL_CERTIFICATE_GAP_TOLERANCE
        and june_correction["objective_agreement_tolerance"]
        == DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and june_metrics["candidate_maximum_constraint_violation"]
        == june_2018_routing_evidence[
            "candidate_maximum_constraint_violation"
        ]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and june_metrics["stationarity_absolute_maximum"]
        == june_2018_routing_evidence["stationarity_absolute_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and june_metrics["stationarity_backward_error_maximum"]
        == june_2018_routing_evidence[
            "stationarity_backward_error_maximum"
        ]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and june_metrics["dual_nonnegative_minimum"]
        == june_2018_routing_evidence["dual_nonnegative_minimum"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and june_metrics["dual_soc_minimum_margin"]
        == june_2018_routing_evidence["dual_soc_minimum_margin"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and june_metrics["candidate_dual_gap_scaled"]
        == june_2018_routing_evidence["candidate_dual_gap_scaled"]
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and june_metrics["objective_cost_scaled_maximum_difference"]
        == june_2018_routing_evidence[
            "objective_cost_scaled_maximum_difference"
        ]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and june_correction["deterministic_weight_bytes_sha256"]
        == june_2018_routing_evidence["deterministic_weight_bytes_sha256"]
        == "7398adc4023ece93bd839e2ef367fd1c24ee8f6f18eff8ed5420b43d1d62677c"
        and june_correction["deterministic_solver_history_sha256"]
        == june_2018_routing_evidence["deterministic_solver_history_sha256"]
        == "0e86c7afb807b24d76e5dff8c80723f5e2dd42699c016bedd561def50ace6a9c"
        and june_correction["final_status_history_entry"]
        == "ordinary_dual_certificate_audit:passed"
        and june_correction["deterministic_repeat"] is True
        and june_2018_routing_evidence[
            "certificate_audit_after_status_routing"
        ]
        == "passed"
        and len(june_failed_rows) == 1
        and june_failure_report.get("failure_count") == 1
        and june_failure_report.get("holding_month") == "2018-06"
        and june_failure_report.get("holding_month_ordinal") == 42
        and june_failure_report.get("performance_fields_emitted") == 0
        and june_failure_report.get("problem_bundle_sha256")
        == june_2018_routing_evidence["problem_bundle_sha256"]
        and june_failed_rows[0].get("failure_id")
        == june_2018_routing_evidence["failure_id"]
        and june_failed_rows[0].get("strategy_id") == ELASTIC_NET_CHALLENGER_ID
        and june_failed_rows[0].get("exception_class")
        == "AllocationSolveError"
        and june_failure_details.get("visibility") == "private_mode_0600"
        and june_private_row.get("strategy_id") == ELASTIC_NET_CHALLENGER_ID
        and june_private_row.get("private_detail_sha256")
        == _sha256_bytes(june_private_bytes)
        == june_failed_rows[0].get("private_detail_sha256")
        and june_private_bytes == _canonical_json(june_private_value) + b"\n"
        and june_private_value
        == {
            "allocator_architecture": "hybrid_alpha_risk_cost_convex",
            "error_message": (
                "ordinary feasibility was not exactly certified infeasible: "
                "ordinary_primary:optimal_inaccurate, "
                "ordinary_retry:optimal_inaccurate, "
                "ordinary_feasibility_retry:optimal_inaccurate"
            ),
            "error_type": "AllocationSolveError",
            "strategy_id": ELASTIC_NET_CHALLENGER_ID,
        }
        and june_authority_manifest.get("compatibility_sha256")
        == june_2018_routing_evidence["compatibility_sha256"]
        and june_authority_manifest.get("opening_checkpoint_sha256")
        == june_2018_routing_evidence["opening_checkpoint_sha256"]
        and june_authority_manifest.get("checkpoint_advanced") is False
        and june_authority_manifest.get("performance_fields_emitted") == 0
        and june_authority_manifest.get("boundary", {}).get("stage12_rows_read")
        == 0
        and june_authority_manifest.get("boundary", {}).get(
            "stage12_access_authorized"
        )
        is False
        and june_success
        == {
            "checkpoint_advanced": False,
            "failure_count": 1,
            "failure_report_sha256": june_2018_routing_evidence[
                "failure_report_sha256"
            ],
            "manifest_sha256": june_2018_routing_evidence[
                "failure_manifest_sha256"
            ],
            "performance_fields_emitted": 0,
            "schema_version": 1,
            "status": "ALL_STRATEGY_FAILURES_REPORTED_TOGETHER",
        }
        and june_manifest["portfolio_weights_or_methodology_changed"]
        is june_2018_routing_evidence[
            "portfolio_weights_or_methodology_changed"
        ]
        is False
        and june_manifest["portfolio_performance_metrics_used"]
        is june_2018_routing_evidence["portfolio_performance_metrics_used"]
        is False
        and june_manifest["stage12_or_later_data_accessed"]
        is june_2018_routing_evidence["stage12_or_later_data_accessed"]
        is False,
        "Stage 11 June 2018 feasibility-status routing evidence differs",
    )
    for field, label in (
        ("adapter_sha256", "June 2019 numerical adapter"),
        ("compatibility_sha256", "June 2019 diagnostic compatibility"),
        ("deterministic_solver_history_sha256", "June 2019 solver history"),
        ("deterministic_weight_bytes_sha256", "June 2019 certified weights"),
        ("failure_bundle_sha256", "June 2019 failure bundle"),
        ("failure_details_sha256", "June 2019 failure details"),
        ("failure_id", "June 2019 diagnostic failure"),
        ("failure_manifest_sha256", "June 2019 failure manifest"),
        ("failure_report_sha256", "June 2019 failure report"),
        ("failure_success_sha256", "June 2019 success receipt"),
        ("opening_checkpoint_sha256", "June 2019 opening checkpoint"),
        ("problem_bundle_sha256", "June 2019 problem bundle"),
        ("problem_file_sha256", "June 2019 problem file"),
        ("regression_fixture_sha256", "June 2019 regression fixture"),
        ("regression_manifest_sha256", "June 2019 regression manifest"),
        ("regression_test_sha256", "June 2019 regression test"),
    ):
        _sha(june_2019_gap_evidence[field], label=label)
    june_2019_manifest_relative = Path(
        str(june_2019_gap_evidence["regression_manifest_path"])
    )
    _require(
        not june_2019_manifest_relative.is_absolute()
        and ".." not in june_2019_manifest_relative.parts
        and june_2019_manifest_relative.as_posix()
        == june_2019_gap_evidence["regression_manifest_path"],
        "June 2019 regression manifest path is unsafe",
    )
    june_2019_manifest_payload = _stable_file_bytes(
        PROJECT_ROOT.joinpath(*june_2019_manifest_relative.parts),
        label="Stage 11 June 2019 regression manifest",
        expected_mode=0o600,
    )
    _require(
        _sha256_bytes(june_2019_manifest_payload)
        == june_2019_gap_evidence["regression_manifest_sha256"],
        "Stage 11 June 2019 regression manifest digest differs",
    )
    try:
        june_2019_manifest_value = json.loads(june_2019_manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11CacheRecoveryError(
            "Stage 11 June 2019 regression manifest is invalid"
        ) from exc
    june_2019_manifest = _exact_keys(
        june_2019_manifest_value,
        {
            "schema_version",
            "failure_id",
            "holding_month",
            "strategy_id",
            "problem_fixture",
            "failure_authority",
            "original_failure",
            "expected_correction",
            "portfolio_weights_or_methodology_changed",
            "portfolio_performance_metrics_used",
            "stage12_or_later_data_accessed",
        },
        label="Stage 11 June 2019 regression manifest",
    )
    june_2019_problem = _exact_keys(
        june_2019_manifest["problem_fixture"],
        {"path", "sha256", "size_bytes", "mode", "problem_bundle_sha256"},
        label="Stage 11 June 2019 problem fixture",
    )
    june_2019_failure_authority = _exact_keys(
        june_2019_manifest["failure_authority"],
        {
            "failure_report_path",
            "failure_report_file_sha256",
            "failure_details_path",
            "failure_details_sha256",
            "manifest_path",
            "manifest_sha256",
            "success_path",
            "success_sha256",
            "mode",
            "failure_bundle_sha256",
            "problem_bundle_sha256",
            "opening_checkpoint_sha256",
            "compatibility_sha256",
        },
        label="Stage 11 June 2019 failure authority",
    )
    june_2019_original = _exact_keys(
        june_2019_manifest["original_failure"],
        {
            "exception_class",
            "failure_code",
            "terminal_status_history",
            "candidate_audit_result",
            "candidate_maximum_hard_or_noninherited_violation_hex",
            "candidate_maximum_inherited_worsening_hex",
            "candidate_maximum_invalid_cost_trade_hex",
            "candidate_locked_maximum_slack_violation_hex",
            "candidate_locked_sum_slack_violation_hex",
            "retry_certificate_source",
            "retry_certificate_status",
            "retry_certificate_failed_gate",
            "retry_stationarity_absolute_hex",
            "retry_stationarity_backward_error_hex",
            "retry_dual_nonnegative_minimum_hex",
            "retry_dual_soc_minimum_margin_hex",
            "retry_candidate_dual_gap_scaled_hex",
            "retry_objective_cost_scaled_maximum_difference_hex",
            "previous_uniform_gap_tolerance",
        },
        label="Stage 11 June 2019 original failure",
    )
    june_2019_correction = _exact_keys(
        june_2019_manifest["expected_correction"],
        {
            "adapter_sha256",
            "n_assets",
            "uniform_gap_tolerance",
            "gap_tolerance_multiplier_of_independent_absolute",
            "candidate_dual_gap_audit_tolerance_changed",
            "application_scope",
            "bounded_dual_gap_reference_tolerance",
            "bounded_dual_gap_band_fraction",
            "bounded_dual_gap_band_absolute",
            "bounded_dual_gap_band_changed",
            "certificate_source",
            "mode",
            "solver_attempt",
            "solver_status",
            "final_status_history_entry",
            "maximum_constraint_violation_hex",
            "deterministic_weight_bytes_sha256",
            "deterministic_solver_history_sha256",
            "deterministic_repeat",
            "allocation_constraint_limit_changed",
            "primal_feasibility_tolerance_changed",
            "certificate_residual_tolerance_changed",
            "objective_agreement_tolerance_changed",
            "solver_tolerance_changed",
            "objective_changed",
            "portfolio_methodology_changed",
        },
        label="Stage 11 June 2019 correction evidence",
    )
    june_2019_authority_values: dict[str, Any] = {}
    for path_key, sha_key, label in (
        ("failure_report_path", "failure_report_sha256", "failure report"),
        ("failure_details_path", "failure_details_sha256", "failure details"),
        ("failure_manifest_path", "failure_manifest_sha256", "failure manifest"),
        ("failure_success_path", "failure_success_sha256", "success receipt"),
    ):
        relative = Path(str(june_2019_gap_evidence[path_key]))
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.as_posix() == june_2019_gap_evidence[path_key],
            f"June 2019 {label} path is unsafe",
        )
        authority_payload = _stable_file_bytes(
            PROJECT_ROOT.joinpath(*relative.parts),
            label=f"Stage 11 June 2019 {label}",
            expected_mode=0o600,
        )
        _require(
            _sha256_bytes(authority_payload)
            == june_2019_gap_evidence[sha_key],
            f"Stage 11 June 2019 {label} digest differs",
        )
        try:
            authority_value = json.loads(authority_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage11CacheRecoveryError(
                f"Stage 11 June 2019 {label} is invalid"
            ) from exc
        _require(
            authority_payload == _canonical_json(authority_value) + b"\n",
            f"Stage 11 June 2019 {label} is not canonical JSON",
        )
        june_2019_authority_values[path_key] = authority_value
    june_2019_failure_report = june_2019_authority_values[
        "failure_report_path"
    ]
    june_2019_failure_details = june_2019_authority_values[
        "failure_details_path"
    ]
    june_2019_authority_manifest = june_2019_authority_values[
        "failure_manifest_path"
    ]
    june_2019_success = june_2019_authority_values["failure_success_path"]
    _require(
        isinstance(june_2019_failure_report, Mapping)
        and isinstance(
            june_2019_failure_report.get("strategy_results"), list
        )
        and len(june_2019_failure_report["strategy_results"]) == 3,
        "Stage 11 June 2019 failure report semantics differ",
    )
    june_2019_failed_rows = [
        row
        for row in june_2019_failure_report["strategy_results"]
        if isinstance(row, Mapping) and row.get("status") == "failed"
    ]
    _require(
        isinstance(june_2019_failure_details, Mapping)
        and isinstance(june_2019_failure_details.get("details"), list)
        and len(june_2019_failure_details["details"]) == 1,
        "Stage 11 June 2019 private failure details differ",
    )
    june_2019_private_row = june_2019_failure_details["details"][0]
    try:
        june_2019_private_bytes = bytes.fromhex(
            str(june_2019_private_row["private_detail_hex"])
        )
        june_2019_private_value = json.loads(june_2019_private_bytes)
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise Stage11CacheRecoveryError(
            "Stage 11 June 2019 private failure detail is invalid"
        ) from exc
    june_2019_fixture_relative = Path(
        str(june_2019_gap_evidence["regression_fixture_path"])
    )
    _require(
        not june_2019_fixture_relative.is_absolute()
        and ".." not in june_2019_fixture_relative.parts
        and june_2019_fixture_relative.as_posix()
        == june_2019_gap_evidence["regression_fixture_path"],
        "June 2019 regression fixture path is unsafe",
    )
    june_2019_fixture_path = PROJECT_ROOT.joinpath(
        *june_2019_fixture_relative.parts
    )
    june_2019_fixture_payload = _stable_file_bytes(
        june_2019_fixture_path,
        label="Stage 11 June 2019 regression fixture",
        expected_mode=0o600,
    )
    _require(
        len(june_2019_fixture_payload)
        == june_2019_gap_evidence["regression_fixture_size_bytes"]
        and _sha256_bytes(june_2019_fixture_payload)
        == june_2019_gap_evidence["regression_fixture_sha256"],
        "Stage 11 June 2019 regression fixture differs",
    )
    _require(
        june_2019_manifest["schema_version"] == 1
        and june_2019_manifest["failure_id"]
        == june_2019_gap_evidence["failure_id"]
        and june_2019_manifest["holding_month"]
        == june_2019_gap_evidence["holding_month"]
        == "2019-06"
        and june_2019_manifest["strategy_id"]
        == june_2019_gap_evidence["strategy_id"]
        == PRIMARY_STRATEGY_ID
        and june_2019_gap_evidence["adapter_sha256"]
        == june_2019_correction["adapter_sha256"]
        == correction["adapter_sha256"]
        and june_2019_gap_evidence["regression_test_path"]
        == correction["failure_regression_test_path"]
        and june_2019_gap_evidence["regression_test_sha256"]
        == correction["failure_regression_test_sha256"]
        and june_2019_problem
        == {
            "path": june_2019_gap_evidence["regression_fixture_path"],
            "sha256": june_2019_gap_evidence["regression_fixture_sha256"],
            "size_bytes": june_2019_gap_evidence[
                "regression_fixture_size_bytes"
            ],
            "mode": june_2019_gap_evidence["regression_fixture_mode"],
            "problem_bundle_sha256": june_2019_gap_evidence[
                "problem_bundle_sha256"
            ],
        }
        and june_2019_gap_evidence["problem_file_sha256"]
        == june_2019_gap_evidence["regression_fixture_sha256"]
        and june_2019_failure_authority
        == {
            "failure_report_path": june_2019_gap_evidence[
                "failure_report_path"
            ],
            "failure_report_file_sha256": june_2019_gap_evidence[
                "failure_report_sha256"
            ],
            "failure_details_path": june_2019_gap_evidence[
                "failure_details_path"
            ],
            "failure_details_sha256": june_2019_gap_evidence[
                "failure_details_sha256"
            ],
            "manifest_path": june_2019_gap_evidence[
                "failure_manifest_path"
            ],
            "manifest_sha256": june_2019_gap_evidence[
                "failure_manifest_sha256"
            ],
            "success_path": june_2019_gap_evidence["failure_success_path"],
            "success_sha256": june_2019_gap_evidence[
                "failure_success_sha256"
            ],
            "mode": june_2019_gap_evidence["failure_authority_mode"],
            "failure_bundle_sha256": june_2019_gap_evidence[
                "failure_bundle_sha256"
            ],
            "problem_bundle_sha256": june_2019_gap_evidence[
                "problem_bundle_sha256"
            ],
            "opening_checkpoint_sha256": june_2019_gap_evidence[
                "opening_checkpoint_sha256"
            ],
            "compatibility_sha256": june_2019_gap_evidence[
                "compatibility_sha256"
            ],
        }
        and june_2019_gap_evidence["failure_authority_mode"]
        == june_2019_gap_evidence["regression_manifest_mode"]
        == june_2019_gap_evidence["regression_fixture_mode"]
        == "0600"
        and june_2019_original["exception_class"] == "AllocationSolveError"
        and june_2019_original["failure_code"]
        == "allocation_solver.allocationsolveerror"
        and june_2019_original["terminal_status_history"]
        == [
            "ordinary_primary:solver_error:SolverError",
            "ordinary_retry:solver_error:SolverError",
            "ordinary_feasibility_retry:infeasible",
            "emergency_minimax:optimal",
            "emergency_sum:optimal",
            "emergency_economic_primary:optimal_inaccurate",
            "emergency_economic_retry:optimal_inaccurate",
            "emergency_economic_equivalent_retry:optimal_inaccurate",
        ]
        and june_2019_original["candidate_audit_result"]
        == june_2019_gap_evidence["candidate_audit_result"]
        == "passed"
        and float.fromhex(
            june_2019_original[
                "candidate_maximum_hard_or_noninherited_violation_hex"
            ]
        )
        == june_2019_gap_evidence[
            "candidate_maximum_hard_or_noninherited_violation"
        ]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and float.fromhex(
            june_2019_original[
                "candidate_maximum_inherited_worsening_hex"
            ]
        )
        == june_2019_gap_evidence["candidate_maximum_inherited_worsening"]
        == 0.0
        and float.fromhex(
            june_2019_original["candidate_locked_sum_slack_violation_hex"]
        )
        == june_2019_gap_evidence["candidate_locked_sum_slack_violation"]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and june_2019_original["retry_certificate_source"]
        == june_2019_gap_evidence["certificate_source_after_uniform_policy"]
        == RETRY_DUAL_CERTIFICATE_SOURCE
        and june_2019_original["retry_certificate_status"]
        == june_2019_gap_evidence["dual_status"]
        == "optimal_inaccurate"
        and june_2019_original["retry_certificate_failed_gate"]
        == june_2019_gap_evidence["previous_failed_gate"]
        == "candidate_dual_gap"
        and float.fromhex(
            june_2019_original["retry_stationarity_absolute_hex"]
        )
        == june_2019_gap_evidence["stationarity_absolute_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and float.fromhex(
            june_2019_original["retry_stationarity_backward_error_hex"]
        )
        == june_2019_gap_evidence["stationarity_backward_error_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and abs(
            float.fromhex(
                june_2019_original["retry_dual_nonnegative_minimum_hex"]
            )
            - june_2019_gap_evidence["dual_nonnegative_minimum"]
        )
        <= abs(
            np.spacing(june_2019_gap_evidence["dual_nonnegative_minimum"])
        )
        and june_2019_gap_evidence["dual_nonnegative_minimum"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and abs(
            float.fromhex(
                june_2019_original["retry_dual_soc_minimum_margin_hex"]
            )
            - june_2019_gap_evidence["dual_soc_minimum_margin"]
        )
        <= abs(np.spacing(june_2019_gap_evidence["dual_soc_minimum_margin"]))
        and june_2019_gap_evidence["dual_soc_minimum_margin"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and float.fromhex(
            june_2019_original["retry_candidate_dual_gap_scaled_hex"]
        )
        == june_2019_gap_evidence["candidate_dual_gap_scaled"]
        and abs(june_2019_gap_evidence["candidate_dual_gap_scaled"])
        > INDEPENDENT_ABSOLUTE_TOLERANCE
        and abs(june_2019_gap_evidence["candidate_dual_gap_scaled"])
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and abs(
            float.fromhex(
                june_2019_original[
                    "retry_objective_cost_scaled_maximum_difference_hex"
                ]
            )
            - june_2019_gap_evidence[
                "objective_cost_scaled_maximum_difference"
            ]
        )
        <= abs(
            np.spacing(
                june_2019_gap_evidence[
                    "objective_cost_scaled_maximum_difference"
                ]
            )
        )
        and june_2019_gap_evidence[
            "objective_cost_scaled_maximum_difference"
        ]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and june_2019_original["previous_uniform_gap_tolerance"]
        == june_2019_gap_evidence[
            "previous_global_candidate_dual_gap_tolerance"
        ]
        == certificate_contract["previous_maximum_scaled_candidate_dual_gap"]
        == INDEPENDENT_ABSOLUTE_TOLERANCE
        and june_2019_gap_evidence[
            "current_global_candidate_dual_gap_tolerance"
        ]
        == certificate_contract["maximum_scaled_candidate_dual_gap"]
        == DUAL_CERTIFICATE_GAP_TOLERANCE
        and june_2019_gap_evidence[
            "candidate_dual_gap_tolerance_multiplier_of_independent_absolute"
        ]
        == certificate_contract[
            "candidate_dual_gap_tolerance_multiplier_of_independent_absolute"
        ]
        == 2.0
        and june_2019_gap_evidence["candidate_dual_gap_tolerance_scope"]
        == certificate_contract["candidate_dual_gap_tolerance_scope"]
        and june_2019_gap_evidence[
            "candidate_dual_gap_tolerance_uniform_across_routes_and_months"
        ]
        is certificate_contract[
            "candidate_dual_gap_tolerance_uniform_across_routes_and_months"
        ]
        is True
        and june_2019_gap_evidence["bounded_dual_gap_reference_tolerance"]
        == june_2019_correction["bounded_dual_gap_reference_tolerance"]
        == certificate_contract["bounded_feasible_gap_reference_tolerance"]
        == BOUNDED_DUAL_GAP_REFERENCE_TOLERANCE
        and june_2019_gap_evidence["bounded_dual_gap_band_fraction"]
        == june_2019_correction["bounded_dual_gap_band_fraction"]
        == certificate_contract["bounded_feasible_gap_band_fraction"]
        == BOUNDED_DUAL_GAP_FRACTION
        and june_2019_gap_evidence["bounded_dual_gap_band_absolute"]
        == june_2019_correction["bounded_dual_gap_band_absolute"]
        == certificate_contract["bounded_feasible_gap_band_absolute"]
        == BOUNDED_DUAL_GAP_BAND
        and june_2019_gap_evidence["bounded_dual_gap_band_changed"]
        is june_2019_correction["bounded_dual_gap_band_changed"]
        is certificate_contract["bounded_feasible_gap_band_changed"]
        is False
        and june_2019_correction["uniform_gap_tolerance"]
        == DUAL_CERTIFICATE_GAP_TOLERANCE
        and june_2019_correction[
            "gap_tolerance_multiplier_of_independent_absolute"
        ]
        == 2.0
        and june_2019_correction[
            "candidate_dual_gap_audit_tolerance_changed"
        ]
        is True
        and june_2019_correction["application_scope"]
        == "all_stage11_months_and_all_dual_certificate_routes"
        and june_2019_correction["certificate_source"]
        == RETRY_DUAL_CERTIFICATE_SOURCE
        and june_2019_correction["mode"]
        == june_2019_gap_evidence["solver_mode_after_policy"]
        == "emergency"
        and june_2019_correction["solver_attempt"]
        == june_2019_gap_evidence["solver_attempt_after_policy"]
        == "emergency"
        and june_2019_correction["solver_status"]
        == june_2019_gap_evidence["solver_status_after_policy"]
        == DUAL_CERTIFIED_SOLVER_STATUS
        and june_2019_correction["final_status_history_entry"]
        == "emergency_economic_dual_certificate_audit:passed"
        and float.fromhex(
            june_2019_correction["maximum_constraint_violation_hex"]
        )
        == june_2019_gap_evidence["maximum_result_violation"]
        and june_2019_correction["deterministic_weight_bytes_sha256"]
        == june_2019_gap_evidence["deterministic_weight_bytes_sha256"]
        and june_2019_correction["deterministic_solver_history_sha256"]
        == june_2019_gap_evidence["deterministic_solver_history_sha256"]
        and june_2019_correction["deterministic_repeat"]
        is june_2019_gap_evidence["deterministic_repeat"]
        is True
        and all(
            june_2019_correction[name] is False
            for name in (
                "allocation_constraint_limit_changed",
                "primal_feasibility_tolerance_changed",
                "certificate_residual_tolerance_changed",
                "objective_agreement_tolerance_changed",
                "solver_tolerance_changed",
                "objective_changed",
                "portfolio_methodology_changed",
            )
        )
        and all(
            june_2019_gap_evidence[name] is False
            for name in (
                "allocation_constraints_or_limits_changed",
                "primal_feasibility_tolerance_changed",
                "certificate_residual_tolerance_changed",
                "objective_agreement_tolerance_changed",
                "solver_tolerances_changed",
                "objective_changed",
                "portfolio_weights_or_methodology_changed",
                "portfolio_performance_metrics_used",
                "stage12_or_later_data_accessed",
            )
        )
        and june_2019_manifest["portfolio_weights_or_methodology_changed"]
        is june_2019_gap_evidence[
            "portfolio_weights_or_methodology_changed"
        ]
        is False
        and june_2019_manifest["portfolio_performance_metrics_used"]
        is june_2019_gap_evidence["portfolio_performance_metrics_used"]
        is False
        and june_2019_manifest["stage12_or_later_data_accessed"]
        is june_2019_gap_evidence["stage12_or_later_data_accessed"]
        is False
        and len(june_2019_failed_rows) == 1
        and june_2019_failure_report.get("failure_count") == 1
        and june_2019_failure_report.get("holding_month") == "2019-06"
        and june_2019_failure_report.get("holding_month_ordinal") == 54
        and june_2019_failure_report.get("performance_fields_emitted") == 0
        and june_2019_failure_report.get("problem_bundle_sha256")
        == june_2019_gap_evidence["problem_bundle_sha256"]
        and june_2019_failed_rows[0].get("failure_id")
        == june_2019_gap_evidence["failure_id"]
        and june_2019_failed_rows[0].get("strategy_id")
        == PRIMARY_STRATEGY_ID
        and june_2019_failed_rows[0].get("exception_class")
        == "AllocationSolveError"
        and june_2019_failure_details.get("visibility")
        == "private_mode_0600"
        and june_2019_private_row.get("strategy_id") == PRIMARY_STRATEGY_ID
        and june_2019_private_row.get("private_detail_sha256")
        == _sha256_bytes(june_2019_private_bytes)
        == june_2019_failed_rows[0].get("private_detail_sha256")
        and june_2019_private_bytes
        == _canonical_json(june_2019_private_value) + b"\n"
        and june_2019_private_value.get("error_type")
        == "AllocationSolveError"
        and june_2019_private_value.get("strategy_id") == PRIMARY_STRATEGY_ID
        and (
            "emergency_economic_retry_dual_certificate_audit:"
            "failed:candidate_dual_gap"
        )
        in june_2019_private_value.get("error_message", "")
        and june_2019_authority_manifest.get("schema_version") == 1
        and june_2019_authority_manifest.get("protocol_id")
        == "stage11-authenticated-diagnostic-persistence-v1"
        and june_2019_authority_manifest.get("kind")
        == "stage11_aggregate_solver_failure_report"
        and june_2019_authority_manifest.get("holding_month") == "2019-06"
        and june_2019_authority_manifest.get("holding_month_ordinal") == 54
        and june_2019_authority_manifest.get("checkpoint_advanced") is False
        and june_2019_authority_manifest.get("performance_fields_emitted") == 0
        and june_2019_authority_manifest.get(
            "all_active_strategies_attempted_and_inactive_receipted"
        )
        is True
        and june_2019_authority_manifest.get("problem_bundle_sha256")
        == june_2019_gap_evidence["problem_bundle_sha256"]
        and june_2019_authority_manifest.get("compatibility_sha256")
        == june_2019_gap_evidence["compatibility_sha256"]
        and june_2019_authority_manifest.get("opening_checkpoint_sha256")
        == june_2019_gap_evidence["opening_checkpoint_sha256"]
        and june_2019_authority_manifest.get("boundary")
        == {
            "final_period_rows_read": 0,
            "first_holding_month": "2015-01",
            "last_holding_month": "2019-11",
            "latest_permitted_observation_date": "2019-12-02",
            "stage": 11,
            "stage12_access_authorized": False,
            "stage12_rows_read": 0,
        }
        and june_2019_authority_manifest.get("failure_report")
        == {
            "failure_count": 1,
            "failure_ids": [june_2019_gap_evidence["failure_id"]],
            "file_name": "failure_report.json",
            "sha256": june_2019_gap_evidence["failure_report_sha256"],
        }
        and june_2019_authority_manifest.get("private_failure_details")
        == {
            "file_name": "failure_details.json",
            "sha256": june_2019_gap_evidence["failure_details_sha256"],
            "size_bytes": 18_845,
            "visibility": "private_mode_0600",
        }
        and june_2019_success
        == {
            "checkpoint_advanced": False,
            "failure_count": 1,
            "failure_report_sha256": june_2019_gap_evidence[
                "failure_report_sha256"
            ],
            "manifest_sha256": june_2019_gap_evidence[
                "failure_manifest_sha256"
            ],
            "performance_fields_emitted": 0,
            "schema_version": 1,
            "status": "ALL_STRATEGY_FAILURES_REPORTED_TOGETHER",
        },
        "Stage 11 June 2019 uniform candidate-dual-gap evidence differs",
    )
    _require(
        emergency_evidence["forced_original_final_statuses"]
        == ["optimal_inaccurate", "optimal_inaccurate"]
        and emergency_evidence["equivalent_retry_status"] == "optimal"
        and emergency_evidence["forced_certificate_status"]
        == DUAL_CERTIFIED_SOLVER_STATUS
        and emergency_evidence["portfolio_performance_metrics_used"] is False
        and emergency_evidence[
            "equivalent_retry_weight_maximum_absolute_difference"
        ]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and emergency_evidence[
            "equivalent_retry_objective_absolute_difference"
        ]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
        and emergency_evidence["candidate_maximum_constraint_violation"]
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
        and emergency_evidence["locked_maximum_slack_violation"]
        <= emergency_contract["locked_slack_audit_tolerance"]
        and emergency_evidence["locked_sum_slack_violation"]
        <= emergency_contract["locked_slack_audit_tolerance"]
        and emergency_evidence["stationarity_absolute_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and emergency_evidence["stationarity_backward_error_maximum"]
        <= DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and emergency_evidence["dual_nonnegative_minimum"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and emergency_evidence["dual_soc_minimum_margin"]
        >= -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
        and abs(emergency_evidence["candidate_dual_gap_scaled"])
        <= DUAL_CERTIFICATE_GAP_TOLERANCE
        and emergency_evidence[
            "objective_cost_scaled_maximum_difference"
        ]
        <= DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE,
        "Stage 11 synthetic emergency-final evidence differs",
    )
    prior_outcomes = _exact_keys(
        trigger["prior_authority_outcome_root"],
        {
            "authority_build_id",
            "authority_verification_id",
            "directory_mode_counts",
            "file_count",
            "file_mode",
            "inventory_sha256",
            "outcome_materialization_receipt_sha256",
            "table_receipts_equal_frozen_contract",
            "total_size_bytes",
        },
        label="prior-authority outcome root",
    )
    _sha(
        prior_outcomes["inventory_sha256"],
        label="prior-authority outcome inventory",
    )
    _sha(
        prior_outcomes["outcome_materialization_receipt_sha256"],
        label="prior-authority outcome receipt",
    )
    _require(
        prior_outcomes["authority_build_id"]
        == "stage11-preflight-b8961e69b15d87422c802032"
        and prior_outcomes["authority_verification_id"]
        == "verify-71b9fc29e693402a"
        and prior_outcomes["directory_mode_counts"] == {"0755": 1}
        and prior_outcomes["file_count"] == 29
        and prior_outcomes["file_mode"] == "0644"
        and prior_outcomes["total_size_bytes"] == 156_420_966
        and prior_outcomes["table_receipts_equal_frozen_contract"] is True,
        "prior-authority outcome-root contract differs",
    )
    follow_on = _exact_keys(
        trigger["follow_on_numerical_failure"],
        {
            "failure_receipt_path",
            "failure_receipt_sha256",
            "holding_month",
            "no_weights_trades_pnl_or_result_bundle_published",
            "original_status_history",
            "outcome_root",
            "production_order_predecessors",
            "root_cause",
            "selected_without_portfolio_performance_metrics",
            "strategy_id",
        },
        label="Stage 11 follow-on numerical failure",
    )
    _require(
        follow_on["holding_month"] == "2015-02"
        and follow_on["no_weights_trades_pnl_or_result_bundle_published"] is True
        and follow_on["selected_without_portfolio_performance_metrics"] is True
        and follow_on["strategy_id"] == ELASTIC_NET_CHALLENGER_ID
        and follow_on["original_status_history"]
        == [
            "ordinary_primary:optimal_inaccurate",
            "ordinary_retry:optimal_inaccurate",
            "ordinary_feasibility_retry:optimal",
        ]
        and follow_on["root_cause"]
        == (
            "the feasible ordinary hybrid retry candidate had no independent "
            "exact-optimality certificate after both economic solves returned "
            "optimal_inaccurate"
        ),
        "Stage 11 follow-on numerical failure contract differs",
    )
    _require(
        follow_on["production_order_predecessors"]
        == {
            PRIMARY_STRATEGY_ID: "emergency_optimal",
            "ensemble_rank_challenger": "emergency_optimal",
        },
        "Stage 11 follow-on production-order evidence differs",
    )
    follow_on_outcomes = _exact_keys(
        follow_on["outcome_root"],
        {
            "authority_build_id",
            "authority_verification_id",
            "directory_mode_counts",
            "file_count",
            "file_mode",
            "inventory_sha256",
            "outcome_materialization_receipt_sha256",
            "table_receipts_equal_frozen_contract",
            "total_size_bytes",
        },
        label="follow-on numerical-failure outcome root",
    )
    _require(
        dict(follow_on_outcomes) == dict(prior_outcomes),
        "follow-on numerical-failure outcome root differs",
    )
    status_failure = _exact_keys(
        trigger["follow_on_certificate_status_failure"],
        {
            "failure_receipt_path",
            "failure_receipt_sha256",
            "holding_month",
            "no_weights_trades_pnl_or_result_bundle_published",
            "original_status_history",
            "root_cause",
            "selected_without_portfolio_performance_metrics",
            "strategy_id",
        },
        label="Stage 11 dual-certificate status-routing failure",
    )
    _require(
        status_failure["holding_month"] == "2015-04"
        and status_failure[
            "no_weights_trades_pnl_or_result_bundle_published"
        ]
        is True
        and status_failure["selected_without_portfolio_performance_metrics"]
        is True
        and status_failure["strategy_id"] == PRIMARY_STRATEGY_ID
        and status_failure["original_status_history"]
        == [
            "ordinary_primary:optimal_inaccurate",
            "ordinary_retry:optimal_inaccurate",
            "ordinary_feasibility_retry:optimal",
            "ordinary_dual_certificate:optimal_inaccurate",
            "ordinary_dual_certificate_audit:failed:dual_not_exact_optimal",
        ]
        and status_failure["root_cause"]
        == (
            "the explicit dual candidate and value were discarded before "
            "direct residual and gap audits solely because the solver status "
            "was optimal_inaccurate"
        ),
        "Stage 11 dual-certificate status-routing failure differs",
    )
    emergency_failure = _exact_keys(
        trigger["follow_on_emergency_economic_failure"],
        {
            "failure_receipt_path",
            "failure_receipt_sha256",
            "holding_month",
            "no_weights_trades_pnl_or_result_bundle_published",
            "original_status_history",
            "root_cause",
            "seal_path",
            "seal_sha256",
            "selected_without_portfolio_performance_metrics",
            "strategy_id",
        },
        label="Stage 11 emergency-economic numerical failure",
    )
    _require(
        emergency_failure["holding_month"] == "2015-07"
        and emergency_failure[
            "no_weights_trades_pnl_or_result_bundle_published"
        ]
        is True
        and emergency_failure[
            "selected_without_portfolio_performance_metrics"
        ]
        is True
        and emergency_failure["strategy_id"]
        == "not_persisted_by_unlabeled_allocator_boundary"
        and emergency_failure["original_status_history"]
        == [
            "ordinary_primary:solver_error:SolverError",
            "ordinary_retry:solver_error:SolverError",
            "ordinary_feasibility_retry:infeasible",
            "emergency_minimax:optimal",
            "emergency_sum:optimal",
            "emergency_economic_primary:optimal_inaccurate",
            "emergency_economic_retry:optimal_inaccurate",
        ]
        and emergency_failure["root_cause"]
        == (
            "the final emergency economic problem retained an "
            "objective-inactive maximum variable and had no "
            "emergency-specific certified inaccurate-status path"
        ),
        "Stage 11 emergency-economic numerical failure differs",
    )
    _require(
        authority_actions
        == {
            "existing_gate_b_approval_remains_bound": True,
            "fresh_stage11_no_outcome_preflight_required": True,
            "independent_preflight_verification_required": True,
            "new_gate_b_user_approval_required": False,
            "ordinary_feasibility_status_certificate_routing_user_authorized": True,
            "stage10_rebuild_required": False,
            "uniform_certificate_objective_agreement_policy_user_authorized": True,
            "uniform_candidate_dual_gap_policy_user_authorized": True,
            "uniform_certificate_residual_policy_user_authorized": True,
        },
        "Stage 11 numerical amendment authority path differs",
    )

    for relative, expected_sha, label in (
        (
            correction["adapter_path"],
            correction["adapter_sha256"],
            "Stage 11 numerical adapter",
        ),
        (
            correction["test_path"],
            correction["test_sha256"],
            "Stage 11 numerical adapter test",
        ),
        (
            correction["failure_regression_test_path"],
            correction["failure_regression_test_sha256"],
            "Stage 11 July failure regression test",
        ),
        (
            correction["failure_regression_manifest_path"],
            correction["failure_regression_manifest_sha256"],
            "Stage 11 July failure regression manifest",
        ),
        (
            july_certificate_evidence["regression_fixture_path"],
            july_certificate_evidence["regression_fixture_sha256"],
            "Stage 11 July failure regression fixture",
        ),
        (
            july_certificate_evidence["failure_receipt_path"],
            july_certificate_evidence["failure_receipt_sha256"],
            "Stage 11 July dual-certificate failure receipt",
        ),
        (
            july_certificate_evidence["failure_seal_path"],
            july_certificate_evidence["failure_seal_sha256"],
            "Stage 11 July dual-certificate failure seal",
        ),
        (
            november_sum_evidence["regression_manifest_path"],
            november_sum_evidence["regression_manifest_sha256"],
            "Stage 11 November emergency-sum regression manifest",
        ),
        (
            november_sum_evidence["regression_fixture_path"],
            november_sum_evidence["regression_fixture_sha256"],
            "Stage 11 November emergency-sum regression fixture",
        ),
        (
            november_sum_evidence["regression_test_path"],
            november_sum_evidence["regression_test_sha256"],
            "Stage 11 November emergency-sum regression test",
        ),
        (
            november_sum_evidence["failure_report_path"],
            november_sum_evidence["failure_report_sha256"],
            "Stage 11 November emergency-sum failure report",
        ),
        (
            november_sum_evidence["failure_details_path"],
            november_sum_evidence["failure_details_sha256"],
            "Stage 11 November emergency-sum failure details",
        ),
        (
            november_sum_evidence["failure_manifest_path"],
            november_sum_evidence["failure_manifest_sha256"],
            "Stage 11 November emergency-sum failure manifest",
        ),
        (
            november_sum_evidence["failure_success_path"],
            november_sum_evidence["failure_success_sha256"],
            "Stage 11 November emergency-sum success receipt",
        ),
        (
            november_bounded_evidence["regression_manifest_path"],
            november_bounded_evidence["regression_manifest_sha256"],
            "Stage 11 November bounded regression manifest",
        ),
        (
            november_bounded_evidence["regression_fixture_path"],
            november_bounded_evidence["regression_fixture_sha256"],
            "Stage 11 November bounded regression fixture",
        ),
        (
            november_bounded_evidence["regression_test_path"],
            november_bounded_evidence["regression_test_sha256"],
            "Stage 11 November bounded regression test",
        ),
        (
            november_bounded_evidence["failure_report_path"],
            november_bounded_evidence["failure_report_sha256"],
            "Stage 11 November bounded failure report",
        ),
        (
            february_2017_policy_evidence["regression_manifest_path"],
            february_2017_policy_evidence["regression_manifest_sha256"],
            "Stage 11 February 2017 regression manifest",
        ),
        (
            february_2017_policy_evidence["regression_fixture_path"],
            february_2017_policy_evidence["regression_fixture_sha256"],
            "Stage 11 February 2017 regression fixture",
        ),
        (
            february_2017_policy_evidence["regression_test_path"],
            february_2017_policy_evidence["regression_test_sha256"],
            "Stage 11 February 2017 regression test",
        ),
        (
            february_failure_authority["failure_report_path"],
            february_failure_authority["failure_report_file_sha256"],
            "Stage 11 February 2017 failure report",
        ),
        (
            february_failure_authority["failure_details_path"],
            february_failure_authority["failure_details_sha256"],
            "Stage 11 February 2017 failure details",
        ),
        (
            february_failure_authority["manifest_path"],
            february_failure_authority["manifest_sha256"],
            "Stage 11 February 2017 failure manifest",
        ),
        (
            february_failure_authority["success_path"],
            february_failure_authority["success_sha256"],
            "Stage 11 February 2017 success receipt",
        ),
        (
            october_2017_policy_evidence["regression_manifest_path"],
            october_2017_policy_evidence["regression_manifest_sha256"],
            "Stage 11 October 2017 regression manifest",
        ),
        (
            october_2017_policy_evidence["regression_fixture_path"],
            october_2017_policy_evidence["regression_fixture_sha256"],
            "Stage 11 October 2017 regression fixture",
        ),
        (
            october_2017_policy_evidence["regression_test_path"],
            october_2017_policy_evidence["regression_test_sha256"],
            "Stage 11 October 2017 regression test",
        ),
        (
            october_2017_policy_evidence["failure_report_path"],
            october_2017_policy_evidence["failure_report_sha256"],
            "Stage 11 October 2017 failure report",
        ),
        (
            october_2017_policy_evidence["failure_details_path"],
            october_2017_policy_evidence["failure_details_sha256"],
            "Stage 11 October 2017 failure details",
        ),
        (
            october_2017_policy_evidence["failure_manifest_path"],
            october_2017_policy_evidence["failure_manifest_sha256"],
            "Stage 11 October 2017 failure manifest",
        ),
        (
            october_2017_policy_evidence["failure_success_path"],
            october_2017_policy_evidence["failure_success_sha256"],
            "Stage 11 October 2017 success receipt",
        ),
        (
            june_2018_routing_evidence["regression_manifest_path"],
            june_2018_routing_evidence["regression_manifest_sha256"],
            "Stage 11 June 2018 regression manifest",
        ),
        (
            june_2018_routing_evidence["regression_fixture_path"],
            june_2018_routing_evidence["regression_fixture_sha256"],
            "Stage 11 June 2018 regression fixture",
        ),
        (
            june_2018_routing_evidence["regression_test_path"],
            june_2018_routing_evidence["regression_test_sha256"],
            "Stage 11 June 2018 regression test",
        ),
        (
            june_2018_routing_evidence["failure_report_path"],
            june_2018_routing_evidence["failure_report_sha256"],
            "Stage 11 June 2018 failure report",
        ),
        (
            june_2018_routing_evidence["failure_details_path"],
            june_2018_routing_evidence["failure_details_sha256"],
            "Stage 11 June 2018 failure details",
        ),
        (
            june_2018_routing_evidence["failure_manifest_path"],
            june_2018_routing_evidence["failure_manifest_sha256"],
            "Stage 11 June 2018 failure manifest",
        ),
        (
            june_2018_routing_evidence["failure_success_path"],
            june_2018_routing_evidence["failure_success_sha256"],
            "Stage 11 June 2018 success receipt",
        ),
        (
            trigger["failure_receipt_path"],
            trigger["failure_receipt_sha256"],
            "Stage 11 numerical failure receipt",
        ),
        (
            follow_on["failure_receipt_path"],
            follow_on["failure_receipt_sha256"],
            "Stage 11 follow-on numerical failure receipt",
        ),
        (
            status_failure["failure_receipt_path"],
            status_failure["failure_receipt_sha256"],
            "Stage 11 dual-certificate status-routing failure receipt",
        ),
        (
            emergency_failure["failure_receipt_path"],
            emergency_failure["failure_receipt_sha256"],
            "Stage 11 emergency-economic numerical failure receipt",
        ),
        (
            emergency_failure["seal_path"],
            emergency_failure["seal_sha256"],
            "Stage 11 emergency-economic numerical failure seal",
        ),
        (
            invariants["stage10_solver_path"],
            invariants["stage10_solver_sha256"],
            "frozen Stage 10 solver",
        ),
        (
            invariants["stage10_costs_path"],
            invariants["stage10_costs_sha256"],
            "frozen Stage 10 costs",
        ),
    ):
        relative_path = Path(str(relative))
        _require(
            not relative_path.is_absolute()
            and ".." not in relative_path.parts
            and relative_path.as_posix() == relative,
            f"{label} path is unsafe",
        )
        _sha(expected_sha, label=f"{label} digest")
        _require(
            _sha256_bytes(
                _stable_file_bytes(
                    PROJECT_ROOT.joinpath(*relative_path.parts),
                    label=label,
                )
            )
            == expected_sha,
            f"{label} digest differs",
        )
    july_fixture_path = PROJECT_ROOT.joinpath(
        *Path(
            str(july_certificate_evidence["regression_fixture_path"])
        ).parts
    )
    july_fixture_stat = july_fixture_path.stat(follow_symlinks=False)
    _require(
        stat.S_ISREG(july_fixture_stat.st_mode)
        and not stat.S_ISLNK(july_fixture_stat.st_mode)
        and july_fixture_stat.st_uid == os.getuid()
        and july_fixture_stat.st_nlink == 1
        and stat.S_IMODE(july_fixture_stat.st_mode) == 0o644
        and july_fixture_stat.st_size
        == july_certificate_evidence["regression_fixture_size_bytes"],
        "Stage 11 July failure regression fixture ownership or mode differs",
    )
    for relative_key, label in (
        ("regression_manifest_path", "regression manifest"),
        ("regression_fixture_path", "regression fixture"),
        ("failure_report_path", "failure report"),
        ("failure_details_path", "failure details"),
        ("failure_manifest_path", "failure manifest"),
        ("failure_success_path", "success receipt"),
    ):
        relative = Path(str(november_sum_evidence[relative_key]))
        path = PROJECT_ROOT.joinpath(*relative.parts)
        path_stat = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(path_stat.st_mode)
            and not stat.S_ISLNK(path_stat.st_mode)
            and path_stat.st_uid == os.getuid()
            and path_stat.st_nlink == 1
            and stat.S_IMODE(path_stat.st_mode) == 0o600,
            "Stage 11 November emergency-sum "
            f"{label} ownership or mode differs",
        )
    november_fixture_path = PROJECT_ROOT.joinpath(
        *Path(
            str(november_sum_evidence["regression_fixture_path"])
        ).parts
    )
    _require(
        november_fixture_path.stat(follow_symlinks=False).st_size
        == november_sum_evidence["regression_fixture_size_bytes"],
        "Stage 11 November emergency-sum fixture size differs",
    )
    for relative_key, expected_mode, label in (
        ("regression_manifest_path", 0o644, "regression manifest"),
        ("regression_fixture_path", 0o600, "regression fixture"),
        ("failure_report_path", 0o600, "failure report"),
    ):
        relative = Path(str(november_bounded_evidence[relative_key]))
        path = PROJECT_ROOT.joinpath(*relative.parts)
        path_stat = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(path_stat.st_mode)
            and not stat.S_ISLNK(path_stat.st_mode)
            and path_stat.st_uid == os.getuid()
            and path_stat.st_nlink == 1
            and stat.S_IMODE(path_stat.st_mode) == expected_mode,
            "Stage 11 November bounded "
            f"{label} ownership or mode differs",
        )
    november_bounded_fixture_path = PROJECT_ROOT.joinpath(
        *Path(
            str(november_bounded_evidence["regression_fixture_path"])
        ).parts
    )
    _require(
        november_bounded_fixture_path.stat(follow_symlinks=False).st_size
        == november_bounded_evidence["regression_fixture_size_bytes"],
        "Stage 11 November bounded fixture size differs",
    )
    for relative, label in (
        (
            february_2017_policy_evidence["regression_manifest_path"],
            "regression manifest",
        ),
        (
            february_2017_policy_evidence["regression_fixture_path"],
            "regression fixture",
        ),
        (february_failure_authority["failure_report_path"], "failure report"),
        (february_failure_authority["failure_details_path"], "failure details"),
        (february_failure_authority["manifest_path"], "failure manifest"),
        (february_failure_authority["success_path"], "success receipt"),
    ):
        relative_path = Path(str(relative))
        path = PROJECT_ROOT.joinpath(*relative_path.parts)
        path_stat = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(path_stat.st_mode)
            and not stat.S_ISLNK(path_stat.st_mode)
            and path_stat.st_uid == os.getuid()
            and path_stat.st_nlink == 1
            and stat.S_IMODE(path_stat.st_mode) == 0o600,
            "Stage 11 February 2017 "
            f"{label} ownership or mode differs",
        )
    february_fixture_path = PROJECT_ROOT.joinpath(
        *Path(
            str(february_2017_policy_evidence["regression_fixture_path"])
        ).parts
    )
    _require(
        february_fixture_path.stat(follow_symlinks=False).st_size
        == february_2017_policy_evidence["regression_fixture_size_bytes"],
        "Stage 11 February 2017 fixture size differs",
    )
    for relative, label in (
        (
            october_2017_policy_evidence["regression_manifest_path"],
            "regression manifest",
        ),
        (
            october_2017_policy_evidence["regression_fixture_path"],
            "regression fixture",
        ),
        (
            october_2017_policy_evidence["failure_report_path"],
            "failure report",
        ),
        (
            october_2017_policy_evidence["failure_details_path"],
            "failure details",
        ),
        (
            october_2017_policy_evidence["failure_manifest_path"],
            "failure manifest",
        ),
        (
            october_2017_policy_evidence["failure_success_path"],
            "success receipt",
        ),
    ):
        relative_path = Path(str(relative))
        path = PROJECT_ROOT.joinpath(*relative_path.parts)
        path_stat = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(path_stat.st_mode)
            and not stat.S_ISLNK(path_stat.st_mode)
            and path_stat.st_uid == os.getuid()
            and path_stat.st_nlink == 1
            and stat.S_IMODE(path_stat.st_mode) == 0o600,
            "Stage 11 October 2017 "
            f"{label} ownership or mode differs",
        )
    october_fixture_path = PROJECT_ROOT.joinpath(
        *Path(
            str(october_2017_policy_evidence["regression_fixture_path"])
        ).parts
    )
    _require(
        october_fixture_path.stat(follow_symlinks=False).st_size
        == october_2017_policy_evidence["regression_fixture_size_bytes"],
        "Stage 11 October 2017 fixture size differs",
    )
    for relative, label in (
        (
            june_2018_routing_evidence["regression_manifest_path"],
            "regression manifest",
        ),
        (
            june_2018_routing_evidence["regression_fixture_path"],
            "regression fixture",
        ),
        (june_2018_routing_evidence["failure_report_path"], "failure report"),
        (
            june_2018_routing_evidence["failure_details_path"],
            "failure details",
        ),
        (
            june_2018_routing_evidence["failure_manifest_path"],
            "failure manifest",
        ),
        (
            june_2018_routing_evidence["failure_success_path"],
            "success receipt",
        ),
    ):
        relative_path = Path(str(relative))
        path = PROJECT_ROOT.joinpath(*relative_path.parts)
        path_stat = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(path_stat.st_mode)
            and not stat.S_ISLNK(path_stat.st_mode)
            and path_stat.st_uid == os.getuid()
            and path_stat.st_nlink == 1
            and stat.S_IMODE(path_stat.st_mode) == 0o600,
            "Stage 11 June 2018 "
            f"{label} ownership or mode differs",
        )
    june_fixture_path = PROJECT_ROOT.joinpath(
        *Path(str(june_2018_routing_evidence["regression_fixture_path"])).parts
    )
    _require(
        june_fixture_path.stat(follow_symlinks=False).st_size
        == june_2018_routing_evidence["regression_fixture_size_bytes"],
        "Stage 11 June 2018 fixture size differs",
    )
    july_seal_path = Path(str(july_certificate_evidence["failure_seal_path"]))
    july_seal = json.loads(
        _stable_file_bytes(
            PROJECT_ROOT.joinpath(*july_seal_path.parts),
            label="Stage 11 July dual-certificate failure seal",
        )
    )
    _require(
        july_seal
        == {
            "attempt_id": (
                "attempt-20260724-elastic-net-dual-certificate-2015-07"
            ),
            "canonical_result_published": False,
            "failure_receipt_sha256": (
                "c41f39b82100778209877f2879b82f34d313f6829ab0937d0de19b555a19f105"
            ),
            "status": "SEALED_NO_RESULT_PUBLISHED",
        },
        "Stage 11 July dual-certificate failure seal differs",
    )
    emergency_seal_path = Path(str(emergency_failure["seal_path"]))
    emergency_seal = json.loads(
        _stable_file_bytes(
            PROJECT_ROOT.joinpath(*emergency_seal_path.parts),
            label="Stage 11 emergency-economic numerical failure seal",
        )
    )
    _require(
        emergency_seal
        == {
            "attempt_id": "attempt-20260724-emergency-economic-2015-07",
            "canonical_result_published": False,
            "failure_receipt_sha256": (
                "76ed9070cd0e51b3d96dac662f1e801deb072824c1b552bd0386941a5e0036fd"
            ),
            "status": "SEALED_NO_RESULT_PUBLISHED",
        },
        "Stage 11 emergency-economic numerical failure seal differs",
    )
    return MappingProxyType(dict(amendment)), _sha256_bytes(payload)


def _load_decimal_serialization_amendment(
    path: Path = DECIMAL_SERIALIZATION_AMENDMENT_PATH,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate the exact-decimal corporate-action identity correction."""

    payload = _stable_file_bytes(
        path, label="Stage 11 decimal serialization amendment"
    )
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11CacheRecoveryError(
            "Stage 11 decimal serialization amendment is invalid"
        ) from exc
    amendment = _exact_keys(
        parsed,
        {
            "schema_version",
            "contract_id",
            "stage",
            "methodology_version",
            "status",
            "trigger",
            "amendment",
            "checkpoint_identity_compatibility",
            "frozen_invariants",
            "empirical_reproduction",
            "required_authority_actions",
            "validation_label_store_stability",
        },
        label="Stage 11 decimal serialization amendment",
    )
    _require(
        amendment["schema_version"] == 1
        and amendment["contract_id"]
        == "phase05-stage11-decimal-serialization-amendment-v1"
        and amendment["stage"] == 11
        and amendment["methodology_version"] == "v1"
        and amendment["status"] == "FROZEN_RULE_RESTORING_TECHNICAL_CORRECTION",
        "Stage 11 decimal serialization amendment identity differs",
    )
    correction = _exact_keys(
        amendment["amendment"],
        {
            "classification",
            "corrected_path",
            "corrected_path_sha256",
            "decimal_wire_representation",
            "float_conversion_permitted",
            "json_default_string_fallback_permitted",
            "test_path",
            "test_path_sha256",
        },
        label="Stage 11 decimal serialization correction",
    )
    evidence = _exact_keys(
        amendment["empirical_reproduction"],
        {
            "all_rows_serialized",
            "duplicate_source_record_hashes",
            "main_and_reserve_delisting_rows_checked",
            "production_failure_message_reproduced_before_correction",
            "unique_source_record_hashes",
        },
        label="Stage 11 decimal serialization evidence",
    )
    invariants = _exact_keys(
        amendment["frozen_invariants"],
        {
            "allocation_constraints_changed",
            "allocation_costs_changed",
            "allocation_objective_changed",
            "corporate_action_economic_treatment_changed",
            "corporate_action_mapping_rules_changed",
            "outcome_values_changed",
            "stage10_costs_sha256",
            "stage10_solver_sha256",
            "stage11_numerical_adapter_sha256",
        },
        label="Stage 11 decimal serialization invariants",
    )
    authority_actions = amendment["required_authority_actions"]
    label_stability = _exact_keys(
        amendment["validation_label_store_stability"],
        {
            "cross_call_device_inode_mtime_ctime_equality_required",
            "exact_content_sha256_required_before_each_outcome_read",
            "failure_receipt_path",
            "failure_receipt_sha256",
            "identity_stability_required_during_each_authenticated_reopen",
            "parquet_row_groups",
            "parquet_rows",
            "schema_sha256",
            "source_content_changed",
            "source_sha256",
            "source_size_bytes",
        },
        label="Stage 11 validation label stability correction",
    )
    identity_compatibility = _exact_keys(
        amendment["checkpoint_identity_compatibility"],
        {
            "cached_model_code_sha256_by_family",
            "current_model_code_sha256_by_family",
            "diagnostic_checkpoint_protected_identity_fields_changed",
            "diagnostic_checkpoint_resume_permitted",
            "exact_changed_portfolio_bot_source_paths",
            "full_forecast_recomputation_and_exact_cache_match_still_required",
            "prior_preflight_manifest_path",
            "prior_preflight_manifest_sha256",
            "selector_or_preprocessing_source_path_changed",
            "solver_only_dual_certificate_residual_policy_changed",
        },
        label="Stage 11 checkpoint identity compatibility",
    )
    trigger = _exact_keys(
        amendment["trigger"],
        {
            "failure_receipt_path",
            "failure_receipt_sha256",
            "no_weights_trades_pnl_or_result_bundle_published",
            "prior_failed_attempt_outcome_root",
            "prior_identity_seal_outcome_root",
            "prior_label_stability_seal_outcome_root",
        },
        label="Stage 11 decimal serialization trigger",
    )
    _require(
        correction["classification"]
        == "rule_restoring_source_identity_serialization_correction"
        and correction["decimal_wire_representation"]
        == "exact_fixed_point_string_via_format_f"
        and correction["float_conversion_permitted"] is False
        and correction["json_default_string_fallback_permitted"] is False
        and evidence["all_rows_serialized"] is True
        and evidence["main_and_reserve_delisting_rows_checked"] == 2_295
        and evidence["unique_source_record_hashes"] == 2_295
        and evidence["duplicate_source_record_hashes"] == 0
        and evidence["production_failure_message_reproduced_before_correction"]
        == "Object of type Decimal is not JSON serializable"
        and trigger["no_weights_trades_pnl_or_result_bundle_published"] is True,
        "Stage 11 decimal serialization correction or evidence differs",
    )
    _sha(label_stability["source_sha256"], label="validation label payload")
    _sha(label_stability["schema_sha256"], label="validation label schema")
    _require(
        label_stability["source_sha256"]
        == "55b7f41c45059a836544e7e745cb96486f1a2ceb5d5aaf3d1d5366e218170f2a"
        and label_stability["source_size_bytes"] == 5_087_418
        and label_stability["parquet_rows"] == 121_620
        and label_stability["parquet_row_groups"] == 2
        and label_stability["source_content_changed"] is False
        and label_stability[
            "cross_call_device_inode_mtime_ctime_equality_required"
        ]
        is False
        and label_stability[
            "exact_content_sha256_required_before_each_outcome_read"
        ]
        is True
        and label_stability[
            "identity_stability_required_during_each_authenticated_reopen"
        ]
        is True,
        "Stage 11 validation label stability correction differs",
    )
    _require(
        all(
            invariants[name] is False
            for name in (
                "allocation_constraints_changed",
                "allocation_costs_changed",
                "allocation_objective_changed",
                "corporate_action_economic_treatment_changed",
                "corporate_action_mapping_rules_changed",
                "outcome_values_changed",
            )
        )
        and invariants["stage10_costs_sha256"]
        == "66880dcbfbd54589a4ae314947dc76a156d2549e661a4177d5ef2addf97b949f"
        and invariants["stage10_solver_sha256"]
        == "c11fdef33f69b491ef5cfc91a4057112fd13075a367cf3ff313e6423257549de"
        and invariants["stage11_numerical_adapter_sha256"]
        == "fb5a2afd09a2e331b718d8d7f47842952e3d4aef41bcff7affab797fb49fa838",
        "Stage 11 decimal serialization amendment changed a frozen rule",
    )
    _require(
        authority_actions
        == {
            "fresh_stage11_no_outcome_preflight_required": True,
            "independent_preflight_verification_required": True,
            "new_economic_design_approval_required": False,
        },
        "Stage 11 decimal serialization authority path differs",
    )
    cached_model_hashes = identity_compatibility[
        "cached_model_code_sha256_by_family"
    ]
    current_model_hashes = identity_compatibility[
        "current_model_code_sha256_by_family"
    ]
    _require(
        cached_model_hashes
        == {
            "neural_network": "2fa12ea51ac47b88d2d843c53f1a0138469aed49c700c1b3df2d1e551a2873a2",
            "xgboost": "e6aa28097e8f5a5bce289b44e0709361b55e0ae2fe2344eb19cf42bf079d6c34",
            "elastic_net": "7ec0808617c5143401cd765eecdc39d6a8a54ea6413e103ba5cccb264bb16b1d",
        }
        and current_model_hashes
        == {
            family: _model_code_sha256(family) for family in LOCKED_SHORTLIST
        }
        and identity_compatibility[
            "full_forecast_recomputation_and_exact_cache_match_still_required"
        ]
        is True
        and identity_compatibility[
            "selector_or_preprocessing_source_path_changed"
        ]
        is False
        and identity_compatibility[
            "diagnostic_checkpoint_protected_identity_fields_changed"
        ]
        is True
        and identity_compatibility["diagnostic_checkpoint_resume_permitted"]
        is False
        and identity_compatibility[
            "solver_only_dual_certificate_residual_policy_changed"
        ]
        is True,
        "Stage 11 checkpoint identity compatibility differs",
    )
    prior_manifest_relative = Path(
        str(identity_compatibility["prior_preflight_manifest_path"])
    )
    _require(
        not prior_manifest_relative.is_absolute()
        and ".." not in prior_manifest_relative.parts
        and prior_manifest_relative.as_posix()
        == identity_compatibility["prior_preflight_manifest_path"],
        "prior preflight manifest path is unsafe",
    )
    _sha(
        identity_compatibility["prior_preflight_manifest_sha256"],
        label="prior preflight manifest",
    )
    prior_manifest_payload = _stable_file_bytes(
        PROJECT_ROOT.joinpath(*prior_manifest_relative.parts),
        label="prior preflight manifest",
    )
    _require(
        _sha256_bytes(prior_manifest_payload)
        == identity_compatibility["prior_preflight_manifest_sha256"],
        "prior preflight manifest digest differs",
    )
    prior_manifest = json.loads(prior_manifest_payload)
    prior_code_hashes = prior_manifest["bindings"]["code_hashes"]
    current_portfolio_bot_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256_bytes(
            _stable_file_bytes(path, label="current portfolio-bot source")
        )
        for path in sorted(
            (PROJECT_ROOT / "src/portfolio_bot").rglob("*.py"),
            key=lambda value: value.as_posix(),
        )
    }
    changed = {
        name: {
            "before_sha256": prior_code_hashes.get(name),
            "after_sha256": digest,
        }
        for name, digest in current_portfolio_bot_hashes.items()
        if prior_code_hashes.get(name) != digest
    }
    _require(
        len(current_portfolio_bot_hashes) == 194
        and changed
        == identity_compatibility["exact_changed_portfolio_bot_source_paths"],
        "portfolio-bot source closure changed beyond the decimal correction",
    )
    outcome_fields = {
        "authority_build_id",
        "authority_verification_id",
        "directory_mode_counts",
        "file_count",
        "file_mode",
        "inventory_sha256",
        "outcome_materialization_receipt_sha256",
        "table_receipts_equal_frozen_contract",
        "total_size_bytes",
    }
    for trigger_key, build_id, verification_id, label in (
        (
            "prior_failed_attempt_outcome_root",
            "stage11-preflight-20e52c189b75503f00128c11",
            "verify-7b0314c146654b5b",
            "decimal-failure",
        ),
        (
            "prior_identity_seal_outcome_root",
            "stage11-preflight-49105f986e401643e9974add",
            "verify-c35fd1805f7b4fd9",
            "checkpoint-identity-seal",
        ),
        (
            "prior_label_stability_seal_outcome_root",
            "stage11-preflight-bcbe724343688bb9e2884312",
            "verify-268e350c2414ec37",
            "label-stability-seal",
        ),
    ):
        prior_outcomes = _exact_keys(
            trigger[trigger_key],
            outcome_fields,
            label=f"{label} outcome root",
        )
        _sha(prior_outcomes["inventory_sha256"], label=f"{label} tree")
        _sha(
            prior_outcomes["outcome_materialization_receipt_sha256"],
            label=f"{label} outcome receipt",
        )
        _require(
            prior_outcomes["authority_build_id"] == build_id
            and prior_outcomes["authority_verification_id"] == verification_id
            and prior_outcomes["directory_mode_counts"] == {"0755": 1}
            and prior_outcomes["file_count"] == 29
            and prior_outcomes["file_mode"] == "0644"
            and prior_outcomes["total_size_bytes"] == 156_420_966
            and prior_outcomes["table_receipts_equal_frozen_contract"] is True,
            f"{label} outcome-root contract differs",
        )
    for relative, expected_sha, label in (
        (
            correction["corrected_path"],
            correction["corrected_path_sha256"],
            "corrected corporate-action adapter",
        ),
        (
            correction["test_path"],
            correction["test_path_sha256"],
            "decimal serialization regression test",
        ),
        (
            trigger["failure_receipt_path"],
            trigger["failure_receipt_sha256"],
            "decimal serialization failure receipt",
        ),
        (
            label_stability["failure_receipt_path"],
            label_stability["failure_receipt_sha256"],
            "validation label stability failure receipt",
        ),
        (
            "src/portfolio_bot/allocation/costs.py",
            invariants["stage10_costs_sha256"],
            "frozen Stage 10 costs",
        ),
        (
            "src/portfolio_bot/allocation/solver.py",
            invariants["stage10_solver_sha256"],
            "frozen Stage 10 solver",
        ),
        (
            "scripts/stage_11_allocation_numerical_adapter.py",
            invariants["stage11_numerical_adapter_sha256"],
            "Stage 11 numerical adapter",
        ),
    ):
        relative_path = Path(str(relative))
        _require(
            not relative_path.is_absolute()
            and ".." not in relative_path.parts
            and relative_path.as_posix() == relative,
            f"{label} path is unsafe",
        )
        _sha(expected_sha, label=f"{label} digest")
        _require(
            _sha256_bytes(
                _stable_file_bytes(
                    PROJECT_ROOT.joinpath(*relative_path.parts),
                    label=label,
                )
            )
            == expected_sha,
            f"{label} digest differs",
        )
    return MappingProxyType(dict(amendment)), _sha256_bytes(payload)


def _tree_inventory(
    root: Path,
    *,
    expected_directory_modes: frozenset[int] | None = None,
    expected_file_mode: int | None = None,
) -> Mapping[str, object]:
    """Authenticate every directory and file and return one exact tree digest."""

    root = Path(os.path.abspath(root.expanduser()))
    root_stat = root.stat(follow_symlinks=False)
    _require(
        stat.S_ISDIR(root_stat.st_mode)
        and not stat.S_ISLNK(root_stat.st_mode)
        and root_stat.st_uid == os.getuid()
        and (stat.S_IMODE(root_stat.st_mode) & 0o022) == 0
        and (
            expected_directory_modes is None
            or stat.S_IMODE(root_stat.st_mode) in expected_directory_modes
        ),
        f"recovery tree is absent or unsafe: {root}",
    )
    records: list[dict[str, object]] = []
    directory_mode_counts: dict[str, int] = {}

    def directory_identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_mode,
            value.st_uid,
        )

    def visit(directory: Path) -> None:
        before = directory.stat(follow_symlinks=False)
        directory_mode = stat.S_IMODE(before.st_mode)
        _require(
            stat.S_ISDIR(before.st_mode)
            and not stat.S_ISLNK(before.st_mode)
            and before.st_uid == os.getuid()
            and (directory_mode & 0o022) == 0
            and (
                expected_directory_modes is None
                or directory_mode in expected_directory_modes
            ),
            f"recovery directory is unsafe: {directory}",
        )
        mode_key = f"{directory_mode:04o}"
        directory_mode_counts[mode_key] = directory_mode_counts.get(mode_key, 0) + 1
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            child = Path(entry.path)
            metadata = child.stat(follow_symlinks=False)
            _require(not stat.S_ISLNK(metadata.st_mode), f"cache symlink is prohibited: {child}")
            if stat.S_ISDIR(metadata.st_mode):
                visit(child)
                continue
            _require(
                stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                f"cache member is not an ordinary single-link file: {child}",
            )
            payload = _stable_file_bytes(
                child,
                label=f"cache member {child.name}",
                expected_mode=expected_file_mode,
            )
            records.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "sha256": _sha256_bytes(payload),
                    "size_bytes": len(payload),
                }
            )
        after = directory.stat(follow_symlinks=False)
        _require(
            directory_identity(before) == directory_identity(after),
            f"cache directory changed while inventoried: {directory}",
        )

    visit(root)
    root_after = root.stat(follow_symlinks=False)
    _require(
        directory_identity(root_stat) == directory_identity(root_after),
        f"cache root changed while inventoried: {root}",
    )
    ordered = sorted(records, key=lambda row: str(row["path"]))
    return MappingProxyType(
        {
            "file_count": len(ordered),
            "total_size_bytes": sum(int(row["size_bytes"]) for row in ordered),
            "inventory_sha256": _sha256_bytes(_canonical_json(ordered)),
            "directory_mode_counts": dict(sorted(directory_mode_counts.items())),
        }
    )


def _require_tree_inventory(
    root: Path,
    expected: Mapping[str, object],
    *,
    label: str,
) -> Mapping[str, object]:
    actual = _tree_inventory(
        root,
        expected_directory_modes=frozenset(
            int(str(mode), 8) for mode in expected["directory_mode_counts"]
        ),
        expected_file_mode=int(str(expected["file_mode"]), 8),
    )
    _require(
        dict(actual)
        == {
            "file_count": expected["file_count"],
            "total_size_bytes": expected["total_size_bytes"],
            "inventory_sha256": expected["inventory_sha256"],
            "directory_mode_counts": dict(expected["directory_mode_counts"]),
        },
        f"{label} exact tree inventory differs",
    )
    return actual


def _contract_roots(
    execution_inputs: AuthenticatedStage11ExecutionInputs,
    contract: Mapping[str, Any],
) -> Mapping[str, Path]:
    roots: dict[str, Path] = {}
    for name, record in contract["trees"].items():
        key = str(record["output_root_key"])
        _require(key in execution_inputs.output_roots, f"{name} output-root key differs")
        roots[name] = execution_inputs.output_roots[key]
    _require(
        roots["selector_checkpoints"].parent
        == roots["raw_forecasts"].parent
        == roots["cached_outcomes"].parent,
        "recovery cache roots do not share the frozen execution root",
    )
    return MappingProxyType(roots)


def _validate_authority(
    authority: object,
    execution_inputs: AuthenticatedStage11ExecutionInputs,
    contract: Mapping[str, Any],
) -> None:
    live = require_live_stage11_validation_unblind_authority(authority)
    expected = contract["authority"]
    _require(
        live.approval_contract_file_sha256
        == expected["approval_contract_file_sha256"]
        and live.approval_contract_payload_sha256
        == expected["approval_contract_payload_sha256"]
        and live.execution_input_manifest_sha256
        == expected["execution_input_manifest_sha256"]
        == execution_inputs.manifest_sha256,
        "live recovery authority differs from Gate B or execution inputs",
    )
    _require(
        execution_inputs.predictor.registration.sha256
        == expected["predictor_payload_sha256"],
        "authenticated validation predictor differs from recovery authority",
    )
    _require(
        live.build_id != expected["cached_outcome_authority_build_id"],
        "recovery requires a newly published preflight authority",
    )
    execution_inputs.gate()


def _receipt_from_file(path: Path, forecast_root: Path) -> ForecastBatchReceipt:
    payload = _stable_file_bytes(
        path, label="forecast receipt", expected_mode=0o600
    )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11CacheRecoveryError("forecast receipt JSON is invalid") from exc
    expected_fields = {item.name for item in fields(ForecastBatchReceipt)}
    _require(set(value) == expected_fields, "forecast receipt fields differ")
    converted = dict(value)
    for name in _DATE_FIELDS:
        try:
            converted[name] = dt.date.fromisoformat(str(converted[name]))
        except ValueError as exc:
            raise Stage11CacheRecoveryError(
                f"forecast receipt {name} is invalid"
            ) from exc
    receipt = ForecastBatchReceipt(**converted)
    _require(
        path.name == f"{receipt.forecast_batch_id}.json"
        and path.parent == forecast_root
        and Path(receipt.payload_path)
        == forecast_root / f"{receipt.forecast_batch_id}.arrays",
        "forecast receipt path binding differs",
    )
    _require(
        payload == canonical_json_bytes(receipt.payload()),
        "forecast receipt canonical bytes differ",
    )
    _require(
        receipt.batch_index == 0
        and receipt.batch_count == 1
        and receipt.outcome_rows_read == 0
        and receipt.forward_label_rows_read == 0
        and receipt.portfolio_rows_read == 0
        and receipt.final_period_rows_read == 0,
        "forecast receipt contains an incomplete batch or prohibited read",
    )
    validate_complete_forecast_batches((receipt,))
    return receipt


@contextmanager
def _authenticated_checkpoint_fit_recovery(
    checkpoint_root: Path,
    identity_records: list[dict[str, object]],
):
    """Replace only native model fitting; preprocessing and scoring stay normal."""

    original = selector_backend_module._run_spawn_workers
    fit_target = selector_backend_module._fit_family_worker

    def recover_fit_workers(
        *,
        target: Callable[..., None],
        args_by_worker: tuple[tuple[object, ...], ...],
        names: tuple[str, ...],
        liveness_gate: Callable[[], None],
    ) -> tuple[Mapping[str, Any], ...]:
        if target is not fit_target:
            return original(
                target=target,
                args_by_worker=args_by_worker,
                names=names,
                liveness_gate=liveness_gate,
            )
        _require(
            bool(args_by_worker)
            and len(args_by_worker) == len(names)
            and len(names) == len(set(names))
            and len(names) <= len(LOCKED_SHORTLIST),
            "cached fit-worker batch identity differs",
        )
        messages: list[Mapping[str, Any]] = []
        for worker_args in args_by_worker:
            liveness_gate()
            _require(
                len(worker_args) == 7,
                "cached fit-worker argument contract differs",
            )
            family = str(worker_args[0])
            identities = worker_args[1]
            supplied_root = Path(str(worker_args[5]))
            _require(
                supplied_root == checkpoint_root
                and isinstance(identities, tuple)
                and bool(identities),
                "cached fit-worker root or identity group differs",
            )
            records: list[dict[str, object]] = []
            for identity in identities:
                _require(
                    identity.family == family,
                    "cached fit-worker family/identity differs",
                )
                checkpoint = load_selector_checkpoint(
                    checkpoint_root / identity.checkpoint_id,
                    expected_identity=identity,
                )
                records.append(
                    {
                        "checkpoint_path": str(checkpoint.root),
                        "manifest_sha256": checkpoint.manifest_sha256,
                        "identity_sha256": identity.sha256,
                    }
                )
                identity_records.append(
                    {
                        "refit_signal_date": identity.refit_signal_date.isoformat(),
                        "model_family": identity.family,
                        "config_id": identity.config_id,
                        "seed": identity.seed,
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
                        "reconstructed_identity_sha256": identity.sha256,
                    }
                )
            messages.append(MappingProxyType({"ok": True, "records": records}))
            liveness_gate()
        return tuple(messages)

    _require(
        selector_backend_module._run_spawn_workers is original,
        "selector worker binding changed before recovery",
    )
    selector_backend_module._run_spawn_workers = recover_fit_workers
    try:
        yield
    finally:
        changed = selector_backend_module._run_spawn_workers is not recover_fit_workers
        selector_backend_module._run_spawn_workers = original
        _require(not changed, "selector worker binding changed during recovery")


@contextmanager
def _cached_checkpoint_model_code_identity_scope(
    cached_model_code_sha256_by_family: Mapping[str, str],
):
    """Reconstruct only the content identity under which caches were built.

    The decimal correction changed one non-selector source file.  The cached
    checkpoint identity conservatively hashed every ``portfolio_bot`` module,
    so the identity digest changed even though selector/preprocessing code did
    not.  This scope restores the exact cached identity strings while normal
    preprocessing and scoring still run and must exactly match all 1,947
    content-addressed forecast payloads.
    """

    expected = dict(cached_model_code_sha256_by_family)
    _require(
        set(expected) == set(LOCKED_SHORTLIST)
        and all(
            isinstance(value, str)
            and len(value) == 64
            and set(value).issubset(_HEX64)
            for value in expected.values()
        ),
        "cached model-code identity mapping is malformed",
    )
    original = selector_backend_module._model_code_sha256
    _require(
        original is _model_code_sha256,
        "selector model-code identity binding changed before recovery",
    )

    def cached_identity(family: str) -> str:
        _require(
            family in expected,
            "selector requested an unknown cached model-code identity",
        )
        return expected[family]

    selector_backend_module._model_code_sha256 = cached_identity
    try:
        yield
    finally:
        changed = (
            selector_backend_module._model_code_sha256 is not cached_identity
        )
        selector_backend_module._model_code_sha256 = original
        _require(
            not changed,
            "selector model-code identity binding changed during recovery",
        )


class AuthenticatedStage11CacheRecoveryBackend(EmpiricalStage11SelectorBackend):
    """Production backend with read-only semantic cache authentication.

    Fit and score methods are inherited unchanged.  The scoped fit-worker
    binding loads exact checkpoints; scoring remains the production path.
    """

    def __init__(
        self,
        authority: object,
        execution_inputs: AuthenticatedStage11ExecutionInputs,
        outcome_publication: Stage11OutcomePublication,
        plan: Stage11ExecutionPlan,
        *,
        recovery_contract: Mapping[str, Any],
        checkpoint_identity_compatibility: Mapping[str, Any],
        progress: Callable[[str], None] | None = None,
    ) -> None:
        # Deliberately run the production initializer: it reauthenticates all
        # development inputs, validation predictors, outcome labels and broker
        # counters under the live Gate-B authority.
        self._checkpoint_identity_compatibility = (
            checkpoint_identity_compatibility
        )
        label_receipt = outcome_publication.receipt.get("tables", {}).get(
            "validation_labels.parquet"
        )
        _require(
            isinstance(label_receipt, Mapping),
            "validation label receipt is absent before recovery backend initialization",
        )
        self._validation_label_receipt = dict(label_receipt)
        self.validation_label_content_authentication_count = 0
        super().__init__(
            authority,
            execution_inputs,
            outcome_publication,
            plan,
            progress=progress,
        )
        self._recovery_contract = recovery_contract
        self._cached_checkpoints = self._authenticate_checkpoints()
        self._authenticated_forecast_arrays: dict[
            tuple[dt.date, str, str, int | None],
            tuple[np.ndarray, np.ndarray],
        ] = {}
        self._cached_forecasts = self._authenticate_forecasts()
        self._authenticated_forecast_arrays = MappingProxyType(
            dict(self._authenticated_forecast_arrays)
        )
        self.recomputed_forecast_match_count = 0
        self.gate()

    def _refresh_validation_label_content_identity(self) -> None:
        """Reauthenticate exact bytes while tolerating benign cross-call stat drift."""

        receipt = self._validation_label_receipt
        payload = _stable_file_bytes(
            self._outcome_path,
            label="validation label content reauthentication",
            expected_mode=0o644,
        )
        _require(
            len(payload) == receipt["size_bytes"]
            and _sha256_bytes(payload) == receipt["sha256"],
            "validation label content differs from its frozen receipt",
        )
        before = selector_backend_module._stable_identity(self._outcome_path)
        parquet = pq.ParquetFile(self._outcome_path)
        _require(
            parquet.metadata.num_rows == receipt["rows"]
            and parquet.metadata.num_row_groups == receipt["row_groups"],
            "validation label footer differs from its frozen receipt",
        )
        after = selector_backend_module._stable_identity(self._outcome_path)
        _require(
            before == after,
            "validation label metadata changed during one authenticated reopen",
        )
        # The base backend's cross-call identity guard now starts from this
        # just-authenticated identity.  A later cross-call metadata refresh is
        # tolerated only after the exact bytes are reauthenticated again.
        self._outcome_identity = after
        self.validation_label_content_authentication_count += 1

    def read_outcome_month(
        self,
        row: object,
        knowledge_date: dt.date,
    ) -> pd.DataFrame | None:
        self.gate()
        self._refresh_validation_label_content_identity()
        result = super().read_outcome_month(row, knowledge_date)
        self._refresh_validation_label_content_identity()
        self.gate()
        return result

    def _authenticate_checkpoints(
        self,
    ) -> Mapping[
        tuple[dt.date, str, str, int | None], LoadedSelectorCheckpoint
    ]:
        expected = {
            (job.refit_signal_date, job.model_family, job.config_id, job.seed)
            for job in self._plan.fit_jobs
        }
        directories = tuple(
            child
            for child in self._checkpoint_root.iterdir()
            if child.name != "_runtime_work"
        )
        _require(
            len(directories) == EXPECTED_TOTAL_FITS
            and all(child.is_dir() and not child.is_symlink() for child in directories),
            "selector checkpoint directory grid differs",
        )
        _require(
            not any(self._work_root.iterdir()),
            "selector runtime-work directory must remain empty during recovery",
        )
        current_hashes = {
            family: _model_code_sha256(family) for family in LOCKED_SHORTLIST
        }
        cached_hashes = dict(
            self._checkpoint_identity_compatibility[
                "cached_model_code_sha256_by_family"
            ]
        )
        _require(
            current_hashes
            == dict(
                self._checkpoint_identity_compatibility[
                    "current_model_code_sha256_by_family"
                ]
            )
            and cached_hashes
            == dict(
                self._recovery_contract["semantic_expectations"][
                    "model_code_sha256_by_family"
                ]
            ),
            "current or cached model-code identity differs from the correction",
        )
        result: dict[
            tuple[dt.date, str, str, int | None], LoadedSelectorCheckpoint
        ] = {}
        for directory in sorted(directories, key=lambda value: value.name):
            self.gate()
            checkpoint = load_selector_checkpoint(directory)
            identity = checkpoint.identity
            try:
                key = (
                    dt.date.fromisoformat(str(identity["refit_signal_date"])),
                    str(identity["family"]),
                    str(identity["config_id"]),
                    identity["seed"],
                )
            except (KeyError, ValueError) as exc:
                raise Stage11CacheRecoveryError(
                    "checkpoint job identity is invalid"
                ) from exc
            _require(
                key in expected
                and key not in result
                and identity.get("model_code_sha256") == cached_hashes[key[1]]
                and identity.get("target_name") == "residual_return_rank"
                and identity.get("final_2020_plus_data_read") is False
                and identity.get("portfolio_outcomes_read") is False
                and identity.get("validation_selection_metrics_read") is False,
                "checkpoint identity escaped the locked recovery grid or scope",
            )
            training = identity.get("training_data")
            _require(isinstance(training, Mapping), "checkpoint training receipt is absent")
            _require(
                dt.date.fromisoformat(str(training["max_exit_date"])) <= key[0]
                and dt.date.fromisoformat(
                    str(training["max_outcome_availability_date"])
                )
                <= key[0]
                and str(training["last_holding_month"]) < "2020-01"
                and int(training["full_rows"]) > 0
                and int(training["inner_train_rows"])
                + int(training["inner_validation_rows"])
                == int(training["full_rows"]),
                "checkpoint training chronology or counters differ",
            )
            result[key] = checkpoint
        _require(set(result) == expected, "checkpoint grid is incomplete or duplicated")
        return MappingProxyType(result)

    def _expected_permnos(self, signal_date: dt.date) -> np.ndarray:
        values = pd.to_numeric(
            self._combined_features.loc[
                self._combined_features["signal_date"].eq(signal_date), "permno"
            ],
            errors="raise",
        ).to_numpy(np.int64)
        values = np.sort(values, kind="stable")
        _require(
            values.size > 0
            and np.all(values > 0)
            and (values.size == 1 or np.all(np.diff(values) > 0)),
            "validation predictor scoring domain differs",
        )
        return values

    def _authenticate_forecasts(
        self,
    ) -> Mapping[
        tuple[dt.date, str, str, int | None], ForecastBatchReceipt
    ]:
        json_paths = tuple(sorted(self._forecast_root.glob("*.json")))
        array_paths = tuple(sorted(self._forecast_root.glob("*.arrays")))
        _require(
            len(json_paths)
            == len(array_paths)
            == self._recovery_contract["semantic_expectations"][
                "forecast_receipt_count"
            ],
            "forecast receipt/payload count differs",
        )
        steps = {step.calendar_row.signal_date: step for step in self._plan.monthly_steps}
        expected_keys = {
            (step.calendar_row.signal_date, job.model_family, job.config_id, job.seed)
            for step in self._plan.monthly_steps
            for job in self._plan.fit_jobs
            if job.refit_signal_date == step.active_refit_signal_date
        }
        result: dict[
            tuple[dt.date, str, str, int | None], ForecastBatchReceipt
        ] = {}
        domains: dict[dt.date, np.ndarray] = {}
        total_rows = 0
        for path in json_paths:
            self.gate()
            receipt = _receipt_from_file(path, self._forecast_root)
            key = (
                receipt.signal_date,
                receipt.family,
                receipt.config_id,
                receipt.seed,
            )
            _require(
                key in expected_keys and key not in result,
                "forecast grid is duplicated or escaped",
            )
            step = steps[receipt.signal_date]
            checkpoint_key = (
                receipt.refit_signal_date,
                receipt.family,
                receipt.config_id,
                receipt.seed,
            )
            checkpoint = self._cached_checkpoints.get(checkpoint_key)
            _require(
                checkpoint is not None
                and receipt.refit_signal_date == step.active_refit_signal_date
                and receipt.entry_date == step.calendar_row.entry_date
                and receipt.exit_date == step.calendar_row.exit_date
                and receipt.checkpoint_id == checkpoint.checkpoint_id
                and receipt.checkpoint_manifest_sha256
                == checkpoint.manifest_sha256
                and receipt.signal_date < dt.date(2020, 1, 1)
                and receipt.exit_date <= dt.date(2019, 12, 2),
                "forecast checkpoint/date authority differs",
            )
            arrays = unpack_numeric_arrays(
                _stable_file_bytes(
                    Path(receipt.payload_path),
                    label="forecast numeric payload",
                    expected_mode=0o600,
                )
            )
            permnos = np.asarray(arrays["permnos"], dtype=np.int64)
            forecasts = np.asarray(arrays["raw_forecasts"], dtype=np.float32)
            expected_domain = domains.get(receipt.signal_date)
            if expected_domain is None:
                expected_domain = self._expected_permnos(receipt.signal_date)
                domains[receipt.signal_date] = expected_domain
            _require(
                np.array_equal(permnos, expected_domain)
                and forecasts.shape == expected_domain.shape
                and np.isfinite(forecasts).all()
                and receipt.rows == expected_domain.size,
                "forecast payload does not match authenticated predictor domain",
            )
            retained_permnos = np.array(permnos, dtype=np.int64, copy=True)
            retained_forecasts = np.array(
                forecasts,
                dtype=np.float32,
                copy=True,
            )
            retained_permnos.setflags(write=False)
            retained_forecasts.setflags(write=False)
            self._authenticated_forecast_arrays[key] = (
                retained_permnos,
                retained_forecasts,
            )
            total_rows += receipt.rows
            result[key] = receipt
        _require(
            set(result) == expected_keys
            and set(self._authenticated_forecast_arrays) == expected_keys
            and total_rows
            == self._recovery_contract["semantic_expectations"]["forecast_rows"],
            "forecast grid or aggregate row count differs",
        )
        return MappingProxyType(result)

    def score_month(
        self,
        step: MonthlyExecutionStep,
        checkpoints: Mapping[tuple[str, str, int | None], str],
    ) -> pd.DataFrame:
        """Run production scoring, then compare all 33 arrays to frozen cache."""

        result = super().score_month(step, checkpoints)
        matched = 0
        jobs = tuple(
            job
            for job in self._plan.fit_jobs
            if job.refit_signal_date == step.active_refit_signal_date
        )
        for job in jobs:
            receipt = self._cached_forecasts[
                (
                    step.calendar_row.signal_date,
                    job.model_family,
                    job.config_id,
                    job.seed,
                )
            ]
            seed_mask = (
                result["seed"].isna()
                if job.seed is None
                else result["seed"].eq(job.seed)
            )
            selected = result.loc[
                result["model_family"].eq(job.model_family)
                & result["config_id"].eq(job.config_id)
                & seed_mask
            ].sort_values("permno", kind="mergesort")
            permnos = selected["permno"].to_numpy(np.int64)
            forecasts = selected["raw_forecast"].to_numpy(np.float32)
            _require(
                len(selected) == receipt.rows
                and receipt.checkpoint_manifest_sha256
                == checkpoints[(job.model_family, job.config_id, job.seed)]
                and array_sha256(permnos) == receipt.permnos_sha256
                and array_sha256(forecasts) == receipt.raw_forecasts_sha256,
                "production-recomputed forecast differs from authenticated cache",
            )
            matched += 1
        _require(matched == EXPECTED_FITS_PER_REFIT, "monthly forecast match count differs")
        self.recomputed_forecast_match_count += matched
        return result


def _strict_permnos(values: pd.Series, *, label: str) -> np.ndarray:
    """Normalize identifiers without permitting float-to-integer truncation."""

    raw = pd.to_numeric(values, errors="coerce").to_numpy(float)
    _require(
        np.isfinite(raw).all()
        and np.all(raw > 0.0)
        and np.all(raw == np.floor(raw)),
        f"{label} PERMNO domain differs",
    )
    return raw.astype(np.int64)


def _join_stage9_oof_label_frames(
    labels: pd.DataFrame,
    split: pd.DataFrame,
    oof_keys: pd.DataFrame,
    *,
    expected_months: int = 95,
    expected_rows: int = 172_969,
    expected_present_rows: int = 172_799,
) -> pd.DataFrame:
    """Apply the independently verified Stage 9 OOF membership join."""

    validation = split.loc[
        split["included_in_validation"].astype(bool),
        ["fold_id", "signal_date", "holding_month"],
    ].copy()
    _require(
        len(validation) == expected_months and not validation.duplicated().any(),
        "Stage 9 OOF split membership differs",
    )
    filtered = validation.merge(
        labels,
        on=["signal_date", "holding_month"],
        how="left",
        validate="one_to_many",
        sort=False,
    )
    _require(
        len(filtered) == expected_rows
        and int(filtered["primary_label_status"].astype(str).eq(_PRESENT).sum())
        == expected_present_rows,
        "Stage 9 split-filtered OOF label counts differ",
    )
    filtered["signal_date"] = pd.to_datetime(
        filtered["signal_date"], errors="raise"
    ).dt.date
    filtered["holding_month"] = filtered["holding_month"].astype(str)
    filtered["permno"] = _strict_permnos(
        filtered["permno"], label="Stage 9 split-filtered labels"
    )
    _require(
        not filtered.duplicated(list(_KEY_COLUMNS)).any(),
        "Stage 9 split-filtered OOF label keys are duplicated",
    )
    oof_keys = oof_keys.copy()
    oof_keys["signal_date"] = pd.to_datetime(
        oof_keys["signal_date"], errors="raise"
    ).dt.date
    oof_keys["holding_month"] = oof_keys["holding_month"].astype(str)
    oof_keys["permno"] = _strict_permnos(
        oof_keys["permno"], label="Stage 9 OOF keys"
    )
    left = filtered.loc[:, list(_KEY_COLUMNS)].sort_values(
        list(_KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    right = oof_keys.sort_values(
        list(_KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    _require(left.equals(right), "Stage 9 OOF and split-filtered label domains differ")
    return filtered


def _build_split_filtered_development_calibration(
    execution_inputs: AuthenticatedStage11ExecutionInputs,
    selection: object,
) -> Mapping[str, Sequence[object]]:
    execution_inputs.gate()
    nn_column, xgb_column, enet_column, nn_weight = _selected_score_columns(
        execution_inputs, selection
    )
    oof = pq.read_table(
        execution_inputs.development_oof.path,
        columns=[*_KEY_COLUMNS, nn_column, xgb_column, enet_column],
        use_threads=False,
    ).to_pandas()
    _require(
        len(oof) == 172_969
        and not oof.duplicated(list(_KEY_COLUMNS)).any(),
        "development OOF row count or key uniqueness differs",
    )
    oof["signal_date"] = _date_series(oof["signal_date"], label="development OOF")
    oof["holding_month"] = oof["holding_month"].astype(str)
    oof["permno"] = _strict_permnos(oof["permno"], label="development OOF")
    for column in (nn_column, xgb_column, enet_column):
        values = pd.to_numeric(oof[column], errors="coerce").to_numpy(float)
        _require(
            np.isfinite(values).all() and np.all(np.abs(values) <= 0.5),
            f"development OOF score differs: {column}",
        )
        oof[column] = values
    oof["ensemble_score"] = (
        nn_weight * oof[nn_column] + (1.0 - nn_weight) * oof[xgb_column]
    )
    oof["elastic_net_score"] = oof[enet_column]

    stage9 = execution_inputs.development_oof.stage9_inputs
    labels = _join_stage9_oof_label_frames(
        stage9.read_development_labels().to_pandas(),
        stage9.read_split_membership().to_pandas(),
        oof.loc[:, list(_KEY_COLUMNS)],
    )
    for column in (
        "signal_date",
        "entry_date",
        "exit_date",
        "month_label_available_date",
    ):
        labels[column] = _date_series(labels[column], label=f"development {column}")
    labels["holding_month"] = labels["holding_month"].astype(str)
    labels["permno"] = _strict_permnos(
        labels["permno"], label="development OOF labels"
    )
    merged = oof.loc[
        :, [*_KEY_COLUMNS, "ensemble_score", "elastic_net_score"]
    ].merge(
        labels.loc[:, list(_LABEL_COLUMNS)],
        on=list(_KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    _require(len(merged) == len(oof), "OOF calibration label join differs")
    residual = pd.to_numeric(
        merged["unranked_residual_return"], errors="coerce"
    ).to_numpy(float)
    present = (
        merged["primary_label_status"].astype(str).eq(_PRESENT).to_numpy(bool)
    )
    _require(
        int(present.sum()) == 172_799 and np.isfinite(residual[present]).all(),
        "development OOF PRESENT residual domain differs",
    )
    merged = merged.loc[present].copy()
    merged["unranked_residual_return"] = residual[present]
    merged = merged.sort_values(
        ["holding_month", "permno"], kind="mergesort"
    ).reset_index(drop=True)
    _require(
        not merged.duplicated(["holding_month", "permno"]).any(),
        "development calibration security-month domain is duplicated",
    )
    result = {
        PRIMARY_STRATEGY_ID: _observations(
            merged,
            strategy_id=PRIMARY_STRATEGY_ID,
            score_column="ensemble_score",
        ),
        ELASTIC_NET_CHALLENGER_ID: _observations(
            merged,
            strategy_id=ELASTIC_NET_CHALLENGER_ID,
            score_column="elastic_net_score",
        ),
    }
    execution_inputs.gate()
    return MappingProxyType(result)


@contextmanager
def _portfolio_calibration_override():
    """Narrowly route the portfolio's redundant calibration recheck."""

    original = (
        portfolio_backend_module.build_authenticated_development_calibration_observations
    )
    _require(
        original is _CANONICAL_PORTFOLIO_CALIBRATION,
        "portfolio calibration binding changed before recovery",
    )

    def adapter(execution_inputs: object, selection: object):
        return _build_split_filtered_development_calibration(
            execution_inputs, selection
        )

    portfolio_backend_module.build_authenticated_development_calibration_observations = (
        adapter
    )
    try:
        yield
    finally:
        changed = (
            portfolio_backend_module.build_authenticated_development_calibration_observations
            is not adapter
        )
        portfolio_backend_module.build_authenticated_development_calibration_observations = (
            _CANONICAL_PORTFOLIO_CALIBRATION
        )
        _require(not changed, "portfolio calibration binding changed during recovery")


@contextmanager
def _portfolio_allocation_numerical_override():
    """Install the reviewed Stage 11-only equivalent solver coordinates."""

    _require(
        execution_orchestrator_module.solve_hybrid is _CANONICAL_HYBRID_SOLVER
        and allocation_rank_module.solve_projection
        is _CANONICAL_RANK_PROJECTION_SOLVER,
        "portfolio allocation binding changed before numerical amendment",
    )
    execution_orchestrator_module.solve_hybrid = solve_hybrid_stage11
    allocation_rank_module.solve_projection = solve_projection_stage11
    try:
        yield
    finally:
        changed = (
            execution_orchestrator_module.solve_hybrid is not solve_hybrid_stage11
            or allocation_rank_module.solve_projection
            is not solve_projection_stage11
        )
        execution_orchestrator_module.solve_hybrid = _CANONICAL_HYBRID_SOLVER
        allocation_rank_module.solve_projection = (
            _CANONICAL_RANK_PROJECTION_SOLVER
        )
        _require(
            not changed,
            "portfolio allocation binding changed during numerical amendment",
        )


def _validate_outcome_tables(
    publication: Stage11OutcomePublication,
    contract: Mapping[str, Any],
    authority: object,
) -> None:
    receipt = publication.receipt
    live = require_live_stage11_validation_unblind_authority(authority)
    _require(
        receipt.get("authority_build_id") == live.build_id
        and receipt.get("authority_verification_id") == live.verification_id
        and receipt.get("first_holding_month") == "2015-01"
        and receipt.get("last_holding_month") == "2019-11"
        and receipt.get("label_months") == 59
        and receipt.get("december_2019_holding_month_rows") == 0
        and receipt.get("final_period_rows_read") == 0
        and receipt.get("outcome_reserve", {}).get("maximum_date") == "2019-12-02",
        "rematerialized outcome authority or scope differs",
    )
    _require(
        receipt.get("tables") == contract["outcome_tables"],
        "rematerialized outcome table bytes/schema/rows differ from frozen cache",
    )


def _prepare_outcome_publication(
    authority: object,
    execution_inputs: AuthenticatedStage11ExecutionInputs,
    contract: Mapping[str, Any],
    numerical_amendment: Mapping[str, Any],
    decimal_serialization_amendment: Mapping[str, Any],
    *,
    progress: Callable[[str], None],
) -> Stage11OutcomePublication:
    root = execution_inputs.output_roots["outcome_tables"]
    expected_tree = contract["trees"]["cached_outcomes"]
    old_build = contract["authority"]["cached_outcome_authority_build_id"]
    old_verification = contract["authority"][
        "cached_outcome_authority_verification_id"
    ]
    archive = (
        root.parent
        / "outcome_archives"
        / f"{old_build}-{expected_tree['inventory_sha256'][:16]}"
    )
    prior_amendment_tree = numerical_amendment["trigger"][
        "prior_authority_outcome_root"
    ]
    prior_amendment_archive = (
        root.parent
        / "outcome_archives"
        / (
            f"{prior_amendment_tree['authority_build_id']}-"
            f"{prior_amendment_tree['inventory_sha256'][:16]}"
        )
    )
    decimal_failure_tree = decimal_serialization_amendment["trigger"][
        "prior_failed_attempt_outcome_root"
    ]
    decimal_failure_archive = (
        root.parent
        / "outcome_archives"
        / (
            f"{decimal_failure_tree['authority_build_id']}-"
            f"{decimal_failure_tree['inventory_sha256'][:16]}"
        )
    )
    identity_seal_tree = decimal_serialization_amendment["trigger"][
        "prior_identity_seal_outcome_root"
    ]
    identity_seal_archive = (
        root.parent
        / "outcome_archives"
        / (
            f"{identity_seal_tree['authority_build_id']}-"
            f"{identity_seal_tree['inventory_sha256'][:16]}"
        )
    )
    label_stability_tree = decimal_serialization_amendment["trigger"][
        "prior_label_stability_seal_outcome_root"
    ]
    label_stability_archive = (
        root.parent
        / "outcome_archives"
        / (
            f"{label_stability_tree['authority_build_id']}-"
            f"{label_stability_tree['inventory_sha256'][:16]}"
        )
    )
    if root.exists():
        receipt_path = root / "outcome_materialization_receipt.json"
        payload = _stable_file_bytes(receipt_path, label="cached outcome receipt")
        receipt = json.loads(payload)
        if receipt.get("authority_build_id") == old_build:
            _require(
                receipt.get("authority_verification_id") == old_verification,
                "cached outcome verification authority differs",
            )
            progress("Stage 11: archiving the authenticated prior-authority outcome tree")
            _archive_authenticated_tree(root, archive, expected_tree)
        elif (
            receipt.get("authority_build_id")
            == prior_amendment_tree["authority_build_id"]
        ):
            _require(
                receipt.get("authority_verification_id")
                == prior_amendment_tree["authority_verification_id"]
                and _sha256_bytes(payload)
                == prior_amendment_tree[
                    "outcome_materialization_receipt_sha256"
                ]
                and receipt.get("tables") == contract["outcome_tables"],
                "pre-amendment outcome authority or table receipts differ",
            )
            _require_tree_inventory(
                archive,
                expected_tree,
                label="archived original-authority outcome tree",
            )
            progress(
                "Stage 11: archiving the authenticated pre-amendment outcome tree"
            )
            _archive_authenticated_tree(
                root,
                prior_amendment_archive,
                prior_amendment_tree,
            )
        elif (
            receipt.get("authority_build_id")
            == decimal_failure_tree["authority_build_id"]
        ):
            _require(
                receipt.get("authority_verification_id")
                == decimal_failure_tree["authority_verification_id"]
                and _sha256_bytes(payload)
                == decimal_failure_tree[
                    "outcome_materialization_receipt_sha256"
                ]
                and receipt.get("tables") == contract["outcome_tables"],
                "decimal-failure outcome authority or table receipts differ",
            )
            _require_tree_inventory(
                archive,
                expected_tree,
                label="archived original-authority outcome tree",
            )
            _require_tree_inventory(
                prior_amendment_archive,
                prior_amendment_tree,
                label="archived pre-numerical-amendment outcome tree",
            )
            progress(
                "Stage 11: archiving the authenticated decimal-failure "
                "outcome tree"
            )
            _archive_authenticated_tree(
                root,
                decimal_failure_archive,
                decimal_failure_tree,
            )
        elif (
            receipt.get("authority_build_id")
            == identity_seal_tree["authority_build_id"]
        ):
            _require(
                receipt.get("authority_verification_id")
                == identity_seal_tree["authority_verification_id"]
                and _sha256_bytes(payload)
                == identity_seal_tree[
                    "outcome_materialization_receipt_sha256"
                ]
                and receipt.get("tables") == contract["outcome_tables"],
                "identity-seal outcome authority or table receipts differ",
            )
            for archived_root, archived_tree, label in (
                (archive, expected_tree, "original-authority"),
                (
                    prior_amendment_archive,
                    prior_amendment_tree,
                    "pre-numerical-amendment",
                ),
                (
                    decimal_failure_archive,
                    decimal_failure_tree,
                    "decimal-failure",
                ),
            ):
                _require_tree_inventory(
                    archived_root,
                    archived_tree,
                    label=f"archived {label} outcome tree",
                )
            progress(
                "Stage 11: archiving the authenticated checkpoint-identity "
                "seal outcome tree"
            )
            _archive_authenticated_tree(
                root,
                identity_seal_archive,
                identity_seal_tree,
            )
        elif (
            receipt.get("authority_build_id")
            == label_stability_tree["authority_build_id"]
        ):
            _require(
                receipt.get("authority_verification_id")
                == label_stability_tree["authority_verification_id"]
                and _sha256_bytes(payload)
                == label_stability_tree[
                    "outcome_materialization_receipt_sha256"
                ]
                and receipt.get("tables") == contract["outcome_tables"],
                "label-stability outcome authority or table receipts differ",
            )
            for archived_root, archived_tree, label in (
                (archive, expected_tree, "original-authority"),
                (
                    prior_amendment_archive,
                    prior_amendment_tree,
                    "pre-numerical-amendment",
                ),
                (
                    decimal_failure_archive,
                    decimal_failure_tree,
                    "decimal-failure",
                ),
                (
                    identity_seal_archive,
                    identity_seal_tree,
                    "identity-seal",
                ),
            ):
                _require_tree_inventory(
                    archived_root,
                    archived_tree,
                    label=f"archived {label} outcome tree",
                )
            progress(
                "Stage 11: archiving the authenticated label-stability "
                "seal outcome tree"
            )
            _archive_authenticated_tree(
                root,
                label_stability_archive,
                label_stability_tree,
            )
        else:
            live = require_live_stage11_validation_unblind_authority(authority)
            _require(
                receipt.get("authority_build_id") == live.build_id
                and receipt.get("authority_verification_id") == live.verification_id,
                "outcome root belongs to an unrecognized authority",
            )
            _require_tree_inventory(
                archive,
                expected_tree,
                label="archived prior-authority outcome tree",
            )
    else:
        _require_tree_inventory(
            archive, expected_tree, label="archived prior-authority outcome tree"
        )
        if prior_amendment_archive.exists():
            _require_tree_inventory(
                prior_amendment_archive,
                prior_amendment_tree,
                label="archived pre-amendment outcome tree",
            )
        if decimal_failure_archive.exists():
            _require_tree_inventory(
                decimal_failure_archive,
                decimal_failure_tree,
                label="archived decimal-failure outcome tree",
            )
        if identity_seal_archive.exists():
            _require_tree_inventory(
                identity_seal_archive,
                identity_seal_tree,
                label="archived identity-seal outcome tree",
            )
        if label_stability_archive.exists():
            _require_tree_inventory(
                label_stability_archive,
                label_stability_tree,
                label="archived label-stability outcome tree",
            )
    progress("Stage 11: rematerializing outcome tables under the current authority")
    publication = publish_from_authenticated_execution_inputs(
        authority, execution_inputs
    )
    _validate_outcome_tables(publication, contract, authority)
    execution_inputs.gate()
    return publication


def _archive_authenticated_tree(
    root: Path,
    archive: Path,
    expected_tree: Mapping[str, object],
) -> None:
    """Atomically move an exact authenticated tree; never replace an archive."""

    _require_tree_inventory(root, expected_tree, label="tree before archive")
    _require(
        not archive.exists() and not archive.is_symlink(),
        "frozen archive destination exists",
    )
    _require_safe_directory(root.parent, label="outcome execution root")
    archive.parent.mkdir(parents=True, exist_ok=True)
    _require_safe_directory(archive.parent, label="outcome archive root")
    os.replace(root, archive)
    _require_tree_inventory(archive, expected_tree, label="tree after archive")


def _require_pristine_execution_outputs(
    execution_inputs: AuthenticatedStage11ExecutionInputs,
) -> None:
    """Fail closed if a prior recovered portfolio/result publication exists."""

    execution_root = execution_inputs.output_roots["execution_results"]
    prohibited = (
        execution_root / "primary",
        execution_root / "verification",
        PROJECT_ROOT
        / "artifacts/phase_05_validation/LATEST_STAGE_11_EXECUTION_VERIFIED.json",
        PROJECT_ROOT / "artifacts/phase_05_validation/stage_11_execution_reports",
    )
    _require(
        not any(path.exists() or path.is_symlink() for path in prohibited),
        "Stage 11 recovered execution output already exists; automatic overwrite "
        "or ambiguous publication restart is prohibited",
    )


def _publish_recovery_attestation(
    *,
    authority: object,
    execution_inputs: AuthenticatedStage11ExecutionInputs,
    contract: Mapping[str, Any],
    runtime_versions: Mapping[str, str | None],
    walk_forward: object,
    reconstructed_identity_records: Sequence[Mapping[str, object]],
    outcome_publication: Stage11OutcomePublication,
    checkpoint_inventory_before: Mapping[str, object],
    checkpoint_inventory_after: Mapping[str, object],
    forecast_inventory_before: Mapping[str, object],
    forecast_inventory_after: Mapping[str, object],
    recomputed_forecast_match_count: int,
    portfolio: object,
    numerical_amendment_sha256: str,
    decimal_serialization_amendment_sha256: str,
    validation_label_content_authentication_count: int,
) -> tuple[str, str]:
    """Publish and independently reopen one content-addressed recovery record."""

    live = require_live_stage11_validation_unblind_authority(authority)
    mappings = sorted(
        (dict(value) for value in reconstructed_identity_records),
        key=lambda value: (
            value["refit_signal_date"],
            value["model_family"],
            value["config_id"],
            -1 if value["seed"] is None else value["seed"],
        ),
    )
    _require(
        len(mappings) == 660
        and len(
            {
                (
                    value["refit_signal_date"],
                    value["model_family"],
                    value["config_id"],
                    value["seed"],
                )
                for value in mappings
            }
        )
        == 660,
        "recovery checkpoint attestation count/domain differs",
    )
    _require(
        all(
            walk_forward.checkpoint_sha256_by_refit_and_model[
                (
                    dt.date.fromisoformat(str(value["refit_signal_date"])),
                    str(value["model_family"]),
                    str(value["config_id"]),
                    value["seed"],
                )
            ]
            == value["checkpoint_manifest_sha256"]
            for value in mappings
        ),
        "reconstructed identity mappings differ from walk-forward checkpoints",
    )
    broker = execution_inputs.broker.ledger().to_dict()
    _require(
        broker["final_period_rows_read"] == 0
        and broker["stage12_or_later_rows_read"] == 0
        and broker["maximum_observation_or_exit_date_read"] == "2019-12-02"
        and broker["maximum_signal_date_read"] == "2019-10-31",
        "recovery attestation scope escaped Stage 11",
    )
    old_build = contract["authority"]["cached_outcome_authority_build_id"]
    archive = (
        execution_inputs.output_roots["outcome_tables"].parent
        / "outcome_archives"
        / (
            f"{old_build}-"
            f"{contract['trees']['cached_outcomes']['inventory_sha256'][:16]}"
        )
    )
    old_receipt_path = archive / "outcome_materialization_receipt.json"
    new_receipt_path = (
        outcome_publication.output_root / "outcome_materialization_receipt.json"
    )
    old_receipt_sha = _sha256_bytes(
        _stable_file_bytes(
            old_receipt_path,
            label="archived outcome receipt for attestation",
            expected_mode=0o644,
        )
    )
    new_receipt_sha = _sha256_bytes(
        _stable_file_bytes(
            new_receipt_path,
            label="current outcome receipt for attestation",
            expected_mode=0o644,
        )
    )
    payload = {
        "schema_version": 1,
        "attestation_id": "phase05-stage11-authenticated-checkpoint-recovery-v1",
        "status": "AUTHENTICATED_CHECKPOINT_RECOVERY_READY_FOR_PRIMARY_PUBLICATION",
        "recovery_mode": (
            "authenticated_checkpoint_recovery_with_full_forecast_recomputation"
        ),
        "authority": {
            "fresh_preflight_build_id": live.build_id,
            "fresh_preflight_verification_id": live.verification_id,
            "approval_contract_file_sha256": live.approval_contract_file_sha256,
            "approval_contract_payload_sha256": (
                live.approval_contract_payload_sha256
            ),
            "execution_input_manifest_sha256": live.execution_input_manifest_sha256,
            "failure_receipt_sha256": contract["authority"][
                "failure_receipt_sha256"
            ],
            "predictor_payload_sha256": contract["authority"][
                "predictor_payload_sha256"
            ],
        },
        "runtime_versions": dict(sorted(runtime_versions.items())),
        "checkpoint_reconstruction": {
            "identity_count": len(mappings),
            "fit_worker_substitution_scope": (
                "only empirical_selector_backend._fit_family_worker"
            ),
            "normal_preprocessing_reconstructed": True,
            "exact_expected_identity_loads": True,
            "mappings": mappings,
            "tree_inventory_before": dict(checkpoint_inventory_before),
            "tree_inventory_after": dict(checkpoint_inventory_after),
        },
        "forecast_recomputation": {
            "preexisting_cache_authenticated": True,
            "preexisting_cache_read": True,
            "used_as_prediction_authority": False,
            "normal_predictions_recomputed": True,
            "collision_byte_equality_required": True,
            "normal_production_score_path": True,
            "monthly_score_count": walk_forward.monthly_score_count,
            "receipt_count": 1_947,
            "exact_recomputed_cache_matches": recomputed_forecast_match_count,
            "rows": sum(
                value.rows for value in walk_forward.ledger.forecast_commits
            ),
            "tree_inventory_before": dict(forecast_inventory_before),
            "tree_inventory_after": dict(forecast_inventory_after),
        },
        "development_oof_reconciliation": {
            "full_label_rows": 286_628,
            "oof_rows": 172_969,
            "present_rows": 172_799,
            "split_membership_months": 95,
            "excluded_pre_oof_rows": 113_659,
            "missing_oof_keys": 0,
        },
        "outcome_authority_transition": {
            "old_authority_build_id": old_build,
            "old_authority_verification_id": contract["authority"][
                "cached_outcome_authority_verification_id"
            ],
            "old_tree_inventory": {
                key: contract["trees"]["cached_outcomes"][key]
                for key in (
                    "file_count",
                    "total_size_bytes",
                    "inventory_sha256",
                    "directory_mode_counts",
                    "file_mode",
                )
            },
            "old_receipt_sha256": old_receipt_sha,
            "new_authority_build_id": live.build_id,
            "new_authority_verification_id": live.verification_id,
            "new_receipt_sha256": new_receipt_sha,
            "table_receipts_byte_identical": (
                outcome_publication.receipt.get("tables")
                == contract["outcome_tables"]
            ),
            "materialized_validation_label_rows": 121_620,
        },
        "scope": {
            "first_signal_date": "2014-12-31",
            "last_signal_date": "2019-10-31",
            "maximum_observation_or_exit_date": "2019-12-02",
            "final_2020_plus_rows_read": 0,
            "stage12_or_later_rows_read": 0,
        },
        "portfolio_calibration": {
            "adapter": "exact_stage9_split_filtered_oof_join",
            "call_site_substitution_restored": (
                portfolio_backend_module.build_authenticated_development_calibration_observations
                is _CANONICAL_PORTFOLIO_CALIBRATION
            ),
            "portfolio_weight_rows": len(portfolio.portfolio_weights),
            "portfolio_trade_rows": len(portfolio.portfolio_trades),
            "daily_accounting_rows": len(portfolio.daily_accounting),
            "monthly_metric_rows": len(portfolio.monthly_metrics),
        },
        "portfolio_numerical_formulation": {
            "adapter": (
                "cap_normalized_weights_with_affine_basis_point_trade_coordinates"
            ),
            "algebraically_equivalent_to_frozen_stage10": True,
            "amendment_contract_id": (
                "phase05-stage11-allocation-numerical-amendment-v1"
            ),
            "amendment_contract_path": (
                "configs/phase_05_validation/"
                "stage_11_allocation_numerical_amendment_v1.json"
            ),
            "amendment_contract_sha256": numerical_amendment_sha256,
            "call_site_substitution_restored": (
                execution_orchestrator_module.solve_hybrid
                is _CANONICAL_HYBRID_SOLVER
                and allocation_rank_module.solve_projection
                is _CANONICAL_RANK_PROJECTION_SOLVER
            ),
            "economic_objective_changed": False,
            "feasible_set_changed": False,
            "solver_changed": False,
            "trade_numerical_scale": TRADE_NUMERICAL_SCALE,
        },
        "corporate_action_source_identity_serialization": {
            "amendment_contract_id": (
                "phase05-stage11-decimal-serialization-amendment-v1"
            ),
            "amendment_contract_path": (
                "configs/phase_05_validation/"
                "stage_11_decimal_serialization_amendment_v1.json"
            ),
            "amendment_contract_sha256": (
                decimal_serialization_amendment_sha256
            ),
            "decimal_wire_representation": (
                "exact_fixed_point_string_via_format_f"
            ),
            "float_conversion_permitted": False,
            "json_default_string_fallback_permitted": False,
            "main_and_reserve_delisting_rows_checked": 2_295,
            "unique_source_record_hashes": 2_295,
            "economic_treatment_changed": False,
            "cached_checkpoint_model_identity_scope_restored": (
                selector_backend_module._model_code_sha256
                is _model_code_sha256
            ),
            "normal_forecasts_recomputed_and_exactly_matched": True,
        },
        "validation_label_store_stability": {
            "source_sha256": (
                "55b7f41c45059a836544e7e745cb96486f1a2ceb5d5aaf3d1d5366e218170f2a"
            ),
            "source_size_bytes": 5_087_418,
            "parquet_rows": 121_620,
            "cross_call_stat_identity_required": False,
            "exact_content_reauthenticated_before_and_after_each_read": True,
            "content_authentication_count": (
                validation_label_content_authentication_count
            ),
            "source_content_changed": False,
        },
    }
    _require(
        payload["forecast_recomputation"]["rows"] == 4_013_460
        and payload["forecast_recomputation"]["monthly_score_count"] == 59
        and payload["forecast_recomputation"]["exact_recomputed_cache_matches"]
        == 1_947
        and payload["outcome_authority_transition"][
            "table_receipts_byte_identical"
        ]
        is True
        and payload["portfolio_calibration"]["call_site_substitution_restored"]
        is True
        and payload["portfolio_numerical_formulation"][
            "call_site_substitution_restored"
        ]
        is True
        and payload["portfolio_numerical_formulation"]["trade_numerical_scale"]
        == 10_000.0
        and payload["corporate_action_source_identity_serialization"][
            "main_and_reserve_delisting_rows_checked"
        ]
        == 2_295
        and payload["corporate_action_source_identity_serialization"][
            "unique_source_record_hashes"
        ]
        == 2_295
        and payload["corporate_action_source_identity_serialization"][
            "economic_treatment_changed"
        ]
        is False
        and payload["corporate_action_source_identity_serialization"][
            "cached_checkpoint_model_identity_scope_restored"
        ]
        is True
        and payload["validation_label_store_stability"][
            "content_authentication_count"
        ]
        >= 118
        and payload["validation_label_store_stability"][
            "source_content_changed"
        ]
        is False,
        "recovery attestation evidence is incomplete",
    )
    encoded = _canonical_json(payload)
    digest = _sha256_bytes(encoded)
    root = (
        PROJECT_ROOT
        / "artifacts/phase_05_validation/stage_11_recovery_attestations"
    )
    root.mkdir(parents=True, exist_ok=True)
    _require_safe_directory(root, label="recovery attestation root")
    destination = root / f"recovery-attestation-{digest}.json"
    if not destination.exists():
        temporary = root / f".{destination.name}.tmp-{os.getpid()}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError:
                pass
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
    reopened = _stable_file_bytes(
        destination,
        label="recovery attestation independent reopen",
        expected_mode=0o644,
    )
    _require(
        reopened == encoded
        and _sha256_bytes(reopened) == digest
        and json.loads(reopened) == payload,
        "recovery attestation independent reopen differs",
    )
    return destination.relative_to(PROJECT_ROOT).as_posix(), digest


def run_stage11_authenticated_cache_recovery(
    authority: object,
    *,
    progress: Callable[[str], None] | None = None,
) -> Stage11EmpiricalExecution:
    """Recover, publish and independently verify Stage 11 under Gate B."""

    report = progress if progress is not None else (lambda _message: None)
    require_live_stage11_validation_unblind_authority(authority)
    contract = _load_recovery_contract()
    numerical_amendment, numerical_amendment_sha256 = (
        _load_numerical_amendment()
    )
    (
        decimal_serialization_amendment,
        decimal_serialization_amendment_sha256,
    ) = _load_decimal_serialization_amendment()
    report(
        "Stage 11: authenticating execution inputs, recovery contract, "
        "numerical amendment, and exact-decimal amendment"
    )
    execution_inputs = authenticate_stage11_execution_inputs(authority)
    _validate_authority(authority, execution_inputs, contract)
    _require_pristine_execution_outputs(execution_inputs)
    roots = _contract_roots(execution_inputs, contract)
    checkpoint_inventory_before = _require_tree_inventory(
        roots["selector_checkpoints"],
        contract["trees"]["selector_checkpoints"],
        label="selector checkpoint cache",
    )
    forecast_inventory_before = _require_tree_inventory(
        roots["raw_forecasts"],
        contract["trees"]["raw_forecasts"],
        label="raw forecast cache",
    )
    base_code_hashes = stage11_execution_implementation_code_hashes()
    runtime_versions = validate_runtime_versions()
    outcome_publication = _prepare_outcome_publication(
        authority,
        execution_inputs,
        contract,
        numerical_amendment,
        decimal_serialization_amendment,
        progress=report,
    )
    calendar = AuthenticatedValidationCalendar(
        outcome_publication.calendar_rows,
        EXPECTED_REBALANCE_CALENDAR_SHA256,
    )
    plan = build_stage11_execution_plan(calendar)
    report("Stage 11: authenticating caches and recomputing production forecasts")
    reconstructed_identity_records: list[dict[str, object]] = []
    identity_compatibility = decimal_serialization_amendment[
        "checkpoint_identity_compatibility"
    ]
    with _cached_checkpoint_model_code_identity_scope(
        identity_compatibility["cached_model_code_sha256_by_family"]
    ):
        backend = AuthenticatedStage11CacheRecoveryBackend(
            authority,
            execution_inputs,
            outcome_publication,
            plan,
            recovery_contract=contract,
            checkpoint_identity_compatibility=identity_compatibility,
            progress=report,
        )
        with _authenticated_checkpoint_fit_recovery(
            backend._checkpoint_root,
            reconstructed_identity_records,
        ):
            walk_forward = run_empirical_locked_walk_forward(
                authority, execution_inputs.broker, plan, backend
            )
    _require(
        selector_backend_module._model_code_sha256 is _model_code_sha256,
        "selector model-code identity binding was not restored",
    )
    _require(
        len(reconstructed_identity_records) == 660,
        "reconstructed checkpoint identity record count differs",
    )
    _require(
        backend.recomputed_forecast_match_count == 1_947,
        "production-recomputed forecast match count differs",
    )
    checkpoint_inventory_after = _require_tree_inventory(
        backend._checkpoint_root,
        contract["trees"]["selector_checkpoints"],
        label="selector checkpoint cache after reconstructed identities",
    )
    forecast_inventory_after = _require_tree_inventory(
        backend._forecast_root,
        contract["trees"]["raw_forecasts"],
        label="raw forecast cache after production recomputation",
    )
    _require(walk_forward.ledger.complete, "recovered walk-forward is incomplete")
    selection = derive_empirical_locked_validation_selection(
        authority, execution_inputs.broker, walk_forward.ledger
    )
    development_oof = _build_split_filtered_development_calibration(
        execution_inputs, selection
    )
    _require(
        set(development_oof)
        == {PRIMARY_STRATEGY_ID, ELASTIC_NET_CHALLENGER_ID},
        "recovered development calibration strategy set differs",
    )
    calibration = build_strategy_calibration_schedules(
        selection, walk_forward.ledger, development_oof=development_oof
    )
    selector_stage = Stage11EmpiricalSelectorStage(
        execution_inputs=execution_inputs,
        outcome_publication=outcome_publication,
        calendar_authority=calendar,
        plan=plan,
        walk_forward=walk_forward,
        selection=selection,
        development_oof=development_oof,
        calibration_schedules=calibration,
    )
    report("Stage 11: running normal portfolio construction and net accounting")
    with (
        _portfolio_calibration_override(),
        _portfolio_allocation_numerical_override(),
    ):
        portfolio = run_empirical_portfolio_accounting(
            execution_inputs=execution_inputs,
            outcome_publication=outcome_publication,
            calendar_authority=calendar,
            ledger=walk_forward.ledger,
            selection=selection,
            calibration_schedules=calibration,
            progress=report,
        )
    execution_inputs.gate()
    attestation_path, attestation_sha256 = _publish_recovery_attestation(
        authority=authority,
        execution_inputs=execution_inputs,
        contract=contract,
        runtime_versions=runtime_versions,
        walk_forward=walk_forward,
        reconstructed_identity_records=reconstructed_identity_records,
        outcome_publication=outcome_publication,
        checkpoint_inventory_before=checkpoint_inventory_before,
        checkpoint_inventory_after=checkpoint_inventory_after,
        forecast_inventory_before=forecast_inventory_before,
        forecast_inventory_after=forecast_inventory_after,
        recomputed_forecast_match_count=backend.recomputed_forecast_match_count,
        portfolio=portfolio,
        numerical_amendment_sha256=numerical_amendment_sha256,
        decimal_serialization_amendment_sha256=(
            decimal_serialization_amendment_sha256
        ),
        validation_label_content_authentication_count=(
            backend.validation_label_content_authentication_count
        ),
    )
    implementation_hashes = {
        **base_code_hashes,
        attestation_path: attestation_sha256,
    }
    artifacts = Stage11ExecutionArtifactSet(
        plan=plan,
        ledger=walk_forward.ledger,
        selection=selection,
        calibration_schedules=calibration,
        execution_sessions=portfolio.execution_sessions,
        execution_session_receipts=portfolio.execution_session_receipts,
        portfolio_weights=portfolio.portfolio_weights,
        portfolio_trades=portfolio.portfolio_trades,
        daily_accounting=portfolio.daily_accounting,
        monthly_metrics=portfolio.monthly_metrics,
        access_ledger=execution_inputs.broker.ledger().to_dict(),
    )
    _require(
        stage11_execution_implementation_code_hashes() == base_code_hashes,
        "Stage 11 implementation changed during recovery",
    )
    output_root = execution_inputs.output_roots["execution_results"]
    publication = publish_stage11_execution(
        authority,
        execution_inputs.broker,
        artifacts,
        output_root=output_root / "primary",
        implementation_code_hashes=implementation_hashes,
    )
    primary_manifest = json.loads(
        _stable_file_bytes(
            publication.bundle_path / MANIFEST_NAME,
            label="published Stage 11 primary manifest",
        )
    )
    _require(
        primary_manifest.get("implementation_code_hashes", {}).get(
            attestation_path
        )
        == attestation_sha256,
        "primary execution manifest omitted the recovery attestation",
    )
    verification = verify_stage11_execution(
        authority,
        publication,
        verification_root=output_root / "verification",
    )
    _require(
        verification.difference_count == 0,
        "recovered Stage 11 publication failed independent verification",
    )
    _require(
        stage11_execution_implementation_code_hashes() == base_code_hashes,
        "Stage 11 implementation changed during independent verification",
    )
    verified_report = publish_verified_stage11_execution_report(
        authority, publication, verification
    )
    report_summary = json.loads(
        _stable_file_bytes(
            verified_report.bundle_path / REPORT_SUMMARY_NAME,
            label="verified Stage 11 report summary",
        )
    )
    _require(
        report_summary.get("authority", {}).get("primary_manifest_sha256")
        == publication.manifest_sha256,
        "verified report does not bind the attested primary manifest",
    )
    _require(
        stage11_execution_implementation_code_hashes() == base_code_hashes,
        "Stage 11 implementation changed during report publication",
    )
    execution_inputs.gate()
    return Stage11EmpiricalExecution(
        selector_stage=selector_stage,
        portfolio=portfolio,
        publication=publication,
        verification=verification,
        verified_report=verified_report,
    )


__all__ = [
    "AuthenticatedStage11CacheRecoveryBackend",
    "Stage11CacheRecoveryError",
    "run_stage11_authenticated_cache_recovery",
]
