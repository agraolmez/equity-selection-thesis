"""Authenticated Stage 11 portfolio-phase resume after completed forecast replay.

The immediately preceding execution reused all 20 completed refits and all
1,947 authenticated production forecast payloads, completed six portfolio
months, and then failed closed on an over-strict July fallback audit. Repeating
the model work would add no scientific information. This module therefore:

1. authenticates the exact frozen checkpoint and forecast trees;
2. replays those authenticated forecast arrays through the unchanged
   pseudo-live ledger and selection logic;
3. applies the authenticated missing-mark projection and continuity adapter,
   which never turns a source null into an observed zero return; and
4. runs the normal portfolio, publication, independent verification and report
   functions.

No Stage 12 or 2020+ source is addressable from this module.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import os
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pandas as pd

import portfolio_bot.allocation.rank as allocation_rank_module
import portfolio_bot.validation.empirical_portfolio_backend as portfolio_backend_module
import portfolio_bot.validation.empirical_selector_backend as selector_backend_module
import portfolio_bot.validation.execution_orchestrator as execution_orchestrator_module
from portfolio_bot.selectors.contract import validate_runtime_versions
from portfolio_bot.validation.empirical_execution import (
    Stage11EmpiricalExecution,
    Stage11EmpiricalSelectorStage,
    stage11_execution_implementation_code_hashes,
)
from portfolio_bot.validation.execution_orchestrator import (
    ELASTIC_NET_CHALLENGER_ID,
    EXPECTED_FITS_PER_REFIT,
    EXPECTED_TOTAL_FITS,
    FORECAST_COLUMNS,
    OUTCOME_COLUMNS,
    PRIMARY_STRATEGY_ID,
    RANK_CHALLENGER_ID,
    MonthlyExecutionStep,
    SelectorFitJob,
    build_stage11_execution_plan,
    build_strategy_calibration_schedules,
    derive_empirical_locked_validation_selection,
    run_empirical_locked_walk_forward,
)
from portfolio_bot.validation.execution_publication import (
    MANIFEST_NAME,
    PRIMARY_MEMBERS,
    SUCCESS_NAME,
    VERIFICATION_MEMBERS,
    Stage11ExecutionArtifactSet,
    publish_stage11_execution,
    publish_stage11_execution_core,
    verify_stage11_execution,
)
from portfolio_bot.validation.execution_reporting import (
    DEFAULT_REPORT_DIRECTORY,
    DEFAULT_LATEST_VERIFIED_POINTER_NAME,
    REPORT_MARKDOWN_NAME,
    REPORT_MEMBERS,
    REPORT_SUMMARY_NAME,
    REPORT_SUCCESS_NAME,
    publish_verified_stage11_execution_report,
)
from portfolio_bot.validation.selector_runtime import (
    array_sha256,
)
from portfolio_bot.validation.stage11_diagnostics import (
    DiagnosticCompletionReceipt,
    STAGE11_DIAGNOSTIC_STRATEGY_IDS,
    Stage11DiagnosticCompatibility,
    Stage11DiagnosticSession,
    load_authenticated_diagnostic_completion_seal,
    load_authenticated_failure_reports,
)
from stage_11_clean_candidate import (
    AuthenticatedStage11CleanPortfolioCandidate,
    load_authenticated_stage11_clean_candidate,
    publish_authenticated_stage11_clean_candidate,
)
from portfolio_bot.validation.validation_runner import (
    require_live_stage11_validation_unblind_authority,
)
from portfolio_bot.validation.walk_forward import (
    EXPECTED_REBALANCE_CALENDAR_SHA256,
    AuthenticatedValidationCalendar,
)

import stage_11_cache_recovery as recovery
from stage_11_missing_return_projection_adapter import (
    _CANONICAL_ACTION_MAPPER,
    _CANONICAL_DAILY_INPUTS,
    MissingReturnProjectionAudit,
    Stage11MissingReturnProjectionError,
    stage11_missing_return_projection_scope,
)
from stage_11_pending_materiality_adapter import (
    MIGRATION_BOUNDARY_DATE,
    MIGRATION_EVIDENCE_SHA256,
    MIGRATION_LAST_AUTHORITATIVE_MARK_DATE,
    MIGRATION_NEXT_HOLDING_MONTH,
    MIGRATION_NEXT_MONTH_INDEX,
    MIGRATION_OPENING_NAV_DOLLARS_HEX,
    MIGRATION_PENDING_EVIDENCE_SHA256,
    MIGRATION_PENDING_NO_MARK_COUNT,
    MIGRATION_PERMNO,
    MIGRATION_POSITION_VALUE_DOLLARS_HEX,
    MIGRATION_SOURCE_CHECKPOINT_SHA256,
    MIGRATION_SOURCE_COMPLETED_MONTH,
    MIGRATION_STRATEGY_ID,
    PENDING_MATERIALITY_NORMALIZED_REQUIRED_EXIT_LIMIT,
    PENDING_MATERIALITY_POLICY_ID,
    PENDING_MATERIALITY_THRESHOLD_DOLLARS,
    stage11_pending_materiality_scope,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESUME_AMENDMENT_PATH = (
    PROJECT_ROOT
    / "configs/phase_05_validation/stage_11_portfolio_resume_amendment_v1.json"
)
DIAGNOSTIC_WORKFLOW_PATH = (
    PROJECT_ROOT
    / "configs/phase_05_validation/stage_11_diagnostic_workflow_v1.json"
)
EXECUTION_APPROVAL_PATH = (
    PROJECT_ROOT
    / "configs/phase_05_validation/"
    "stage_11_validation_execution_approval_v1.json"
)
DIAGNOSTIC_REGRESSION_REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs/phase_05_validation/stage_11_diagnostic_regressions_v1.json"
)
_DIAGNOSTIC_REGRESSION_TEST_PATHS = (
    "tests/unit/test_stage11_failure_regressions.py",
    "tests/unit/test_stage11_numerical_adapter.py",
    "tests/unit/test_stage11_cache_recovery.py",
    "tests/unit/test_stage11_execution_orchestrator.py",
    "tests/unit/test_stage11_empirical_portfolio_backend.py",
    "tests/unit/test_stage11_diagnostics.py",
)
_CANONICAL_PORTFOLIO_SELECTION = (
    portfolio_backend_module.derive_locked_validation_selection
)
_CANONICAL_PORTFOLIO_DEVELOPMENT = (
    portfolio_backend_module.build_authenticated_development_calibration_observations
)
_CANONICAL_PORTFOLIO_CALIBRATION = (
    portfolio_backend_module.build_strategy_calibration_schedules
)
_EXPECTED_RETIRED_EXECUTION_GATE_ROWS = 106
_EXPECTED_RETIRED_EXECUTION_GATE_SHA256 = (
    "9c000de80aef24207b8a33081d091cdc2039d85c10918c307f30dc258721c824"
)
_EXPECTED_PENDING_DUST_EVENT_KEY_SHA256 = (
    "cbf2616214703d97eee7b130a26c9144b20c0642ac04600aa221010c8b1a525d"
)
_EXPECTED_PENDING_DUST_EVIDENCE_SHA256 = (
    "b8e938a3393e148f1c4cabcd87d861a5ea09a6120d4f7c05c34bc4c61a7c754e"
)
_CLEAN_CANDIDATE_ECONOMIC_PATHS = (
    "configs/phase_04_models/allocation_build_rules_v1.json",
    "configs/phase_04_models/allocation_policy_v1.toml",
    "configs/phase_05_validation/stage_11_corporate_action_mapping_v1.json",
    "scripts/stage_11_cache_recovery.py",
    "scripts/stage_11_missing_return_projection_adapter.py",
    "scripts/stage_11_pending_materiality_adapter.py",
    "src/portfolio_bot/allocation/candidates.py",
    "src/portfolio_bot/allocation/constraints.py",
    "src/portfolio_bot/allocation/costs.py",
    "src/portfolio_bot/allocation/domain.py",
    "src/portfolio_bot/allocation/rank.py",
    "src/portfolio_bot/allocation/risk.py",
    "src/portfolio_bot/allocation/solver.py",
    "src/portfolio_bot/validation/accounting.py",
    "src/portfolio_bot/validation/calibration.py",
    "src/portfolio_bot/validation/candidate_adapter.py",
    "src/portfolio_bot/validation/corporate_action_adapter.py",
    "src/portfolio_bot/validation/corporate_actions.py",
    "src/portfolio_bot/validation/daily_engine.py",
    "src/portfolio_bot/validation/empirical_portfolio_backend.py",
    "src/portfolio_bot/validation/execution_orchestrator.py",
    "src/portfolio_bot/validation/execution_risk_inputs.py",
    "src/portfolio_bot/validation/problem_builder.py",
)
_ACCESS_LEDGER_AUTHORITY_FIELDS = frozenset(
    {
        "authority_build_id",
        "authority_verification_id",
        "approval_contract_file_sha256",
        "approval_contract_payload_sha256",
        "execution_input_manifest_sha256",
    }
)
_ACCESS_LEDGER_OUTPUT_FIELDS = frozenset(
    {
        "real_validation_forecasts_created",
        "real_validation_selections_created",
        "real_validation_weights_created",
        "real_validation_trades_created",
        "real_validation_pnl_or_performance_rows_created",
        "maximum_empirical_output_date",
    }
)
_ACCESS_LEDGER_PROHIBITED_FIELDS = frozenset(
    {
        "embargoed_holding_month_rows_read",
        "final_period_rows_read",
        "stage12_or_later_rows_read",
    }
)
_CLEAN_CANDIDATE_MISSING_POLICY_FIELDS = (
    "classification",
    "authoritative_null_to_zero_permitted",
    "invalid_nonnull_numeric_suppression_permitted",
    "open_nonzero_position_without_return",
    "observed_return_span_start_continuity_required",
    "pending_identity_rebalance_trade_permitted",
    "pending_identity_corporate_action_permitted",
    "disconnected_or_overlapping_return_span_permitted",
    "final_nonzero_pending_identity_permitted",
    "unresolved_terminal_action_projection",
    "unresolved_action_or_settlement_creation_permitted",
    "canonical_resolved_action_mapping_required",
    "projection_bindings_restored_before_publication",
    "corporate_action_economic_treatment_changed",
)
_CLEAN_CANDIDATE_RETIRED_POLICY_FIELDS = (
    "classification",
    "signal_score_or_ic_domain_changed",
    "signal_date_universe_membership_changed",
    "execution_gate_timing",
    "future_terminal_event_permitted",
    "suppressed_posting_promoted_to_return_or_no_mark",
    "terminal_action_reapplied",
)
_CLEAN_CANDIDATE_PENDING_POLICY_FIELDS = (
    "classification",
    "policy_id",
    "application_scope",
    "strict_absolute_position_value_dollars_upper_bound",
    "normalized_required_exit_violation_upper_bound",
    "pending_no_mark_count_must_be_positive",
    "nav_preserved_bit_for_bit",
    "position_value_set_to_exact_zero",
    "mark_continuity_identity_removed",
    "synthetic_trade_permitted",
    "transaction_cost_permitted",
    "execution_price_read_permitted",
    "return_imputation_permitted",
    "corporate_action_permitted",
    "terminal_identity_retirement_permitted",
    "position_at_or_above_five_cents_permitted",
    "return_span_continuity_limit_changed",
    "next_representable_weight_change_permitted",
)
_EXPECTED_PENDING_DUST_BY_PERMNO = MappingProxyType(
    {
        32_791: {
            "strategy_ids": STAGE11_DIAGNOSTIC_STRATEGY_IDS,
            "holding_month": "2015-05",
            "boundary_date": dt.date(2015, 6, 1),
            "last_authoritative_mark_date": dt.date(2015, 5, 12),
            "pending_no_mark_count": 13,
            "pending_evidence_sha256": (
                "2a69b7cd9b62365fdb47ca35598dcafc2cd94020debb66e65aa91a2cd6935bce"
            ),
        },
        91_366: {
            "strategy_ids": STAGE11_DIAGNOSTIC_STRATEGY_IDS,
            "holding_month": "2018-05",
            "boundary_date": dt.date(2018, 6, 1),
            "last_authoritative_mark_date": dt.date(2018, 5, 11),
            "pending_no_mark_count": 14,
            "pending_evidence_sha256": (
                "cf5f4b5c489861babe70ae936badb27ccc0ee38b1895bf9e137b288aff674e2f"
            ),
        },
        92_102: {
            "strategy_ids": STAGE11_DIAGNOSTIC_STRATEGY_IDS,
            "holding_month": "2017-02",
            "boundary_date": dt.date(2017, 3, 1),
            "last_authoritative_mark_date": dt.date(2017, 2, 7),
            "pending_no_mark_count": 15,
            "pending_evidence_sha256": (
                "cba6abd90adc993246952c573ecbf2a45eb9e09ffdba4d83226c783cb32f1455"
            ),
        },
        91_907: {
            "strategy_ids": (
                ELASTIC_NET_CHALLENGER_ID,
                RANK_CHALLENGER_ID,
            ),
            "holding_month": "2018-08",
            "boundary_date": dt.date(2018, 9, 4),
            "last_authoritative_mark_date": dt.date(2018, 8, 22),
            "pending_no_mark_count": 8,
            "pending_evidence_sha256": (
                "7774fbf69f51c41649d13fd57d7bec5e769df0e44f55433ac1cc944dff391671"
            ),
        },
        76_736: {
            "strategy_ids": (
                ELASTIC_NET_CHALLENGER_ID,
                RANK_CHALLENGER_ID,
            ),
            "holding_month": "2018-10",
            "boundary_date": dt.date(2018, 11, 1),
            "last_authoritative_mark_date": dt.date(2018, 10, 31),
            "pending_no_mark_count": 1,
            "pending_evidence_sha256": (
                "053f1ddde843b60fb2ebfae5777fb02f8285c816b0c0ec700dfd43d2db42db7f"
            ),
        },
    }
)
_EXPECTED_PENDING_DUST_EVIDENCE_ROWS = sum(
    len(expected["strategy_ids"])
    for expected in _EXPECTED_PENDING_DUST_BY_PERMNO.values()
)
_EXPECTED_PENDING_DUST_UNIQUE_PERMNOS = tuple(
    sorted(_EXPECTED_PENDING_DUST_BY_PERMNO)
)
_PARTIAL_DIAGNOSTIC_AGGREGATE_AUDIT_ERRORS = frozenset(
    {
        "unresolved terminal action/return audit pairing differs",
        "unresolved terminal audit key digest differs",
        "no-mark source count or source-record uniqueness differs",
        "no-mark source key digest differs",
    }
)


class Stage11PortfolioResumeError(RuntimeError):
    """Raised before a resumed byte can influence a Stage 11 result."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11PortfolioResumeError(message)


@dataclass(frozen=True, slots=True)
class _PreparedStage11PortfolioInputs:
    live: object
    contract: Mapping[str, Any]
    numerical_amendment: Mapping[str, Any]
    numerical_amendment_sha256: str
    decimal_amendment: Mapping[str, Any]
    decimal_amendment_sha256: str
    resume_amendment: Mapping[str, Any]
    resume_amendment_sha256: str
    execution_inputs: object
    roots: Mapping[str, Path]
    checkpoint_before: Mapping[str, object]
    checkpoint_after: Mapping[str, object]
    forecast_before: Mapping[str, object]
    forecast_after: Mapping[str, object]
    outcome_publication: object
    calendar: object
    plan: object
    base_code_hashes: Mapping[str, str]
    runtime_versions: Mapping[str, str | None]
    backend: object
    walk_forward: object
    selection: object
    development_oof: Mapping[str, object]
    calibration: Mapping[str, object]
    selector_stage: Stage11EmpiricalSelectorStage


@dataclass(frozen=True, slots=True)
class Stage11DiagnosticExecution:
    portfolio_status: object
    completion: DiagnosticCompletionReceipt | None
    regression_suite_sha256: str | None

    @property
    def complete(self) -> bool:
        return self.completion is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_sha256(value: object) -> str:
    return hashlib.sha256(recovery._canonical_json(value)).hexdigest()


def _line_terminated_canonical_json_sha256(value: object) -> str:
    """Hash the frozen JSON-record domain, including its one terminal LF."""

    return hashlib.sha256(
        recovery._canonical_json(value) + b"\n"
    ).hexdigest()


def _stage11_diagnostic_compatibility(
    prepared: _PreparedStage11PortfolioInputs,
) -> Stage11DiagnosticCompatibility:
    approval = json.loads(
        recovery._stable_file_bytes(
            EXECUTION_APPROVAL_PATH,
            label="Stage 11 execution approval for diagnostic compatibility",
            expected_mode=0o644,
        )
    )
    execution_inputs = prepared.execution_inputs
    checkpoint_inventory = prepared.checkpoint_after
    forecast_inventory = prepared.forecast_after
    _require(
        prepared.checkpoint_before == checkpoint_inventory
        and prepared.forecast_before == forecast_inventory,
        "diagnostic compatibility cache inventories changed during replay",
    )
    stable_preflight_scope = {
        "stage": 11,
        "stage11_policy_sha256": approval["stage11_policy_sha256"],
        "stage11_rules_sha256": approval["stage11_rules_sha256"],
        "execution_input_manifest_payload_sha256": (
            execution_inputs.manifest_payload_sha256
        ),
        "stage10_verified_pointer_sha256": (
            approval["stage10_verified_pointer_sha256"]
        ),
    }
    exact_execution_values = {
        "execution_input_manifest_sha256": execution_inputs.manifest_sha256,
        "outcome_tables": prepared.outcome_publication.receipt["tables"],
        "selection_sha256": prepared.selection.selection_sha256,
        "selected_scores_sha256": (
            execution_orchestrator_module.canonical_frame_sha256(
                prepared.selection.selected_scores
            )
        ),
        "calibration_schedule_sha256_by_strategy": {
            strategy_id: schedule.schedule_sha256
            for strategy_id, schedule in sorted(prepared.calibration.items())
        },
        "selector_checkpoint_inventory_sha256": checkpoint_inventory[
            "inventory_sha256"
        ],
        "forecast_inventory_sha256": forecast_inventory["inventory_sha256"],
    }
    allocation_schema_paths = (
        PROJECT_ROOT / "src/portfolio_bot/allocation/domain.py",
        PROJECT_ROOT / "src/portfolio_bot/validation/problem_builder.py",
    )
    accounting_transition_paths = (
        PROJECT_ROOT / "src/portfolio_bot/validation/accounting.py",
        PROJECT_ROOT / "src/portfolio_bot/validation/corporate_actions.py",
        PROJECT_ROOT / "src/portfolio_bot/validation/corporate_action_adapter.py",
        PROJECT_ROOT / "src/portfolio_bot/validation/daily_engine.py",
        PROJECT_ROOT / "src/portfolio_bot/validation/empirical_portfolio_backend.py",
        PROJECT_ROOT / "scripts/stage_11_missing_return_projection_adapter.py",
    )
    return Stage11DiagnosticCompatibility(
        methodology_sha256=approval["methodology_freeze_sha256"],
        preflight_sha256=_aggregate_sha256(stable_preflight_scope),
        gate_b_sha256=prepared.live.approval_contract_payload_sha256,
        execution_input_authority_sha256=_aggregate_sha256(
            exact_execution_values
        ),
        validation_calendar_sha256=EXPECTED_REBALANCE_CALENDAR_SHA256,
        checkpoint_state_schema_sha256=(
            portfolio_backend_module.diagnostic_state_schema_sha256()
        ),
        allocation_problem_schema_sha256=_aggregate_sha256(
            {
                path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path)
                for path in allocation_schema_paths
            }
        ),
        accounting_transition_code_sha256=_aggregate_sha256(
            {
                path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path)
                for path in accounting_transition_paths
            }
        ),
        diagnostic_workflow_sha256=_sha256_file(DIAGNOSTIC_WORKFLOW_PATH),
    )


def _stage11_diagnostic_producer_sha256(
    prepared: _PreparedStage11PortfolioInputs,
) -> str:
    return _aggregate_sha256(dict(sorted(prepared.base_code_hashes.items())))


def _stage11_diagnostic_root(
    prepared: _PreparedStage11PortfolioInputs,
) -> Path:
    return prepared.execution_inputs.output_roots["execution_results"] / "diagnostics"


def _stage11_clean_candidate_root(
    prepared: _PreparedStage11PortfolioInputs,
) -> Path:
    return (
        prepared.execution_inputs.output_roots["execution_results"]
        / "clean_portfolio_candidates"
    )


def _resume_economic_orchestration_sha256() -> str:
    """Hash only the resume functions that can influence portfolio values."""

    source = PROJECT_ROOT / "scripts/stage_11_portfolio_resume.py"
    tree = ast.parse(
        recovery._stable_file_bytes(
            source,
            label="Stage 11 clean portfolio orchestration source",
        ),
        filename=source.as_posix(),
    )
    names = {
        "_portfolio_recheck_reuse_scope",
        "_load_or_run_clean_portfolio_candidate",
    }
    nodes = {
        node.name: ast.dump(node, annotate_fields=True, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    }
    _require(
        set(nodes) == names,
        "clean portfolio orchestration function identity differs",
    )
    return _aggregate_sha256(dict(sorted(nodes.items())))


def _selected_policy_fields(
    value: Mapping[str, object],
    names: tuple[str, ...],
    *,
    label: str,
) -> Mapping[str, object]:
    _require(
        all(name in value for name in names),
        f"{label} clean-candidate policy fields differ",
    )
    return MappingProxyType({name: value[name] for name in names})


def _stage11_clean_candidate_compatibility(
    *,
    prepared: _PreparedStage11PortfolioInputs,
    diagnostic_compatibility: Stage11DiagnosticCompatibility,
    diagnostic_completion: DiagnosticCompletionReceipt,
) -> Mapping[str, object]:
    """Bind only inputs and code that can change portfolio economics.

    The resume-attestation implementation is intentionally excluded.  A
    metadata-only correction may therefore reuse the private clean candidate,
    but only after a fresh preflight, forecast replay, selection/calibration
    reconstruction, and authenticated 59-month diagnostic seal have all
    passed again.
    """

    economic_paths = tuple(
        sorted(
            {
                *_CLEAN_CANDIDATE_ECONOMIC_PATHS,
                *(
                    path
                    for path in prepared.base_code_hashes
                    if path.startswith("src/portfolio_bot/")
                ),
            }
        )
    )
    code_hashes = {
        path: prepared.base_code_hashes[path] for path in economic_paths
    }
    _require(
        len(code_hashes) == len(economic_paths)
        and all(len(value) == 64 for value in code_hashes.values()),
        "clean portfolio candidate economic code identity differs",
    )
    resume = prepared.resume_amendment
    checkpoint = prepared.checkpoint_after
    forecast = prepared.forecast_after
    return MappingProxyType(
        {
            "schema_version": 1,
            "compatibility_id": (
                "phase05-stage11-clean-portfolio-economic-compatibility-v1"
            ),
            "execution_input_manifest_payload_sha256": (
                prepared.execution_inputs.manifest_payload_sha256
            ),
            "outcome_table_receipts": prepared.outcome_publication.receipt[
                "tables"
            ],
            "plan_sha256": prepared.plan.plan_sha256,
            "pseudo_live_ledger_sha256": (
                prepared.walk_forward.ledger.ledger_sha256
            ),
            "selection_sha256": prepared.selection.selection_sha256,
            "selected_scores_sha256": (
                execution_orchestrator_module.canonical_frame_sha256(
                    prepared.selection.selected_scores
                )
            ),
            "calibration_schedule_sha256_by_strategy": {
                strategy_id: schedule.schedule_sha256
                for strategy_id, schedule in sorted(
                    prepared.calibration.items()
                )
            },
            "selector_checkpoint_inventory_sha256": checkpoint[
                "inventory_sha256"
            ],
            "forecast_inventory_sha256": forecast["inventory_sha256"],
            "diagnostic_compatibility": diagnostic_compatibility.as_dict(),
            "diagnostic_latest_checkpoint_sha256": (
                diagnostic_completion.latest_checkpoint_sha256
            ),
            "diagnostic_checkpoint_chain_sha256": (
                diagnostic_completion.checkpoint_chain_sha256
            ),
            "numerical_amendment_sha256": (
                prepared.numerical_amendment_sha256
            ),
            "decimal_serialization_amendment_sha256": (
                prepared.decimal_amendment_sha256
            ),
            "missing_return_projection_policy": _selected_policy_fields(
                resume["missing_return_projection"],
                _CLEAN_CANDIDATE_MISSING_POLICY_FIELDS,
                label="missing-return projection",
            ),
            "retired_identity_execution_policy": _selected_policy_fields(
                resume["retired_identity_execution_gate"],
                _CLEAN_CANDIDATE_RETIRED_POLICY_FIELDS,
                label="retired-identity execution",
            ),
            "pending_identity_materiality_policy": _selected_policy_fields(
                resume["pending_identity_materiality_policy"],
                _CLEAN_CANDIDATE_PENDING_POLICY_FIELDS,
                label="pending-identity materiality",
            ),
            "runtime_versions": dict(sorted(prepared.runtime_versions.items())),
            "economic_implementation_code_sha256": code_hashes,
            "resume_economic_orchestration_ast_sha256": (
                _resume_economic_orchestration_sha256()
            ),
        }
    )


def _stable_access_ledger(
    value: Mapping[str, object],
) -> dict[str, object]:
    return {
        str(key): item
        for key, item in value.items()
        if key not in _ACCESS_LEDGER_AUTHORITY_FIELDS
    }


def _require_clean_candidate_access_transition(
    *,
    current: Mapping[str, object],
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> str:
    """Authenticate the skipped broker transition.

    Returns ``"fresh"`` when the current broker already contains the
    portfolio reads and ``"reused"`` when it exactly matches the saved
    pre-portfolio ledger.
    """

    _require(
        set(current) == set(before) == set(after),
        "clean candidate access-ledger field set differs",
    )
    _require(
        all(before[name] == after[name] for name in _ACCESS_LEDGER_AUTHORITY_FIELDS)
        and all(
            isinstance(current[name], str) and bool(current[name])
            for name in _ACCESS_LEDGER_AUTHORITY_FIELDS
        ),
        "clean candidate saved access authority differs",
    )
    for name in _ACCESS_LEDGER_PROHIBITED_FIELDS:
        _require(
            before[name] == after[name] == current[name] == 0,
            "clean candidate access ledger crossed the Stage 11 boundary",
        )
    for name in (
        "real_validation_weights_created",
        "real_validation_trades_created",
        "real_validation_pnl_or_performance_rows_created",
    ):
        _require(
            before[name] == after[name] == 0,
            "clean candidate was persisted after empirical publication",
        )
    _require(
        all(
            before[name] == after[name]
            for name in _ACCESS_LEDGER_OUTPUT_FIELDS
        ),
        "clean candidate portfolio run fabricated a published output",
    )
    for name, prior in before.items():
        subsequent = after[name]
        if (
            name not in _ACCESS_LEDGER_AUTHORITY_FIELDS
            and name not in _ACCESS_LEDGER_OUTPUT_FIELDS
            and type(prior) is int
        ):
            _require(
                type(subsequent) is int and subsequent >= prior,
                "clean candidate input-access counter regressed",
            )
    stable_current = _stable_access_ledger(current)
    stable_before = _stable_access_ledger(before)
    stable_after = _stable_access_ledger(after)
    if stable_current == stable_after:
        return "fresh"
    _require(
        stable_current == stable_before,
        "current broker differs from both clean-candidate access boundaries",
    )
    return "reused"


def _ledger_with_portfolio_outputs(
    *,
    current_resume: Mapping[str, object],
    portfolio: object,
) -> Mapping[str, object]:
    """Reconstruct the ledger produced by the ordinary publication wrapper."""

    result = dict(current_resume)
    result["real_validation_weights_created"] = len(
        portfolio.portfolio_weights
    )
    result["real_validation_trades_created"] = len(
        portfolio.portfolio_trades
    )
    result["real_validation_pnl_or_performance_rows_created"] = (
        len(portfolio.daily_accounting) + len(portfolio.monthly_metrics)
    )
    output_dates = []
    for frame, column in (
        (portfolio.portfolio_weights, "entry_date"),
        (portfolio.portfolio_trades, "entry_date"),
        (portfolio.daily_accounting, "date"),
        (portfolio.monthly_metrics, "exit_date"),
    ):
        if not frame.empty:
            output_dates.append(
                max(pd.to_datetime(frame[column], errors="raise").dt.date)
            )
    _require(output_dates, "clean candidate portfolio outputs are empty")
    prior_maximum = result["maximum_empirical_output_date"]
    if prior_maximum is not None:
        output_dates.append(dt.date.fromisoformat(str(prior_maximum)))
    result["maximum_empirical_output_date"] = max(output_dates).isoformat()
    return MappingProxyType(result)


def _candidate_reuse_access_provenance(
    *,
    candidate_post_portfolio: Mapping[str, object],
    current_resume: Mapping[str, object],
    portfolio: object,
) -> Mapping[str, object]:
    """Keep historical reads and current output creation under their own authorities."""

    current_with_outputs = _ledger_with_portfolio_outputs(
        current_resume=current_resume,
        portfolio=portfolio,
    )
    _require(
        all(
            candidate_post_portfolio[name] == 0
            and current_with_outputs[name] == 0
            for name in _ACCESS_LEDGER_PROHIBITED_FIELDS
        ),
        "clean candidate reuse access provenance crossed Stage 11",
    )
    return MappingProxyType(
        {
            "schema_version": 2,
            "mode": (
                "authenticated_clean_candidate_cross_authority_reuse"
            ),
            "portfolio_candidate_input_access_ledger": dict(
                candidate_post_portfolio
            ),
            "current_resume_and_output_access_ledger": dict(
                current_with_outputs
            ),
            "historical_input_authority_relabelled": False,
            "portfolio_accounting_recomputed": False,
            "embargoed_holding_month_rows_read": 0,
            "final_period_rows_read": 0,
            "stage12_or_later_rows_read": 0,
        }
    )


def _load_or_run_clean_portfolio_candidate(
    *,
    prepared: _PreparedStage11PortfolioInputs,
    diagnostic_compatibility: Stage11DiagnosticCompatibility,
    diagnostic_completion: DiagnosticCompletionReceipt,
    report: Callable[[str], None],
) -> tuple[
    AuthenticatedStage11CleanPortfolioCandidate,
    str,
    Mapping[str, object],
]:
    """Reuse a closed candidate or run accounting once and persist it."""

    compatibility = _stage11_clean_candidate_compatibility(
        prepared=prepared,
        diagnostic_compatibility=diagnostic_compatibility,
        diagnostic_completion=diagnostic_completion,
    )
    execution_inputs = prepared.execution_inputs
    access_before = execution_inputs.broker.ledger().to_dict()
    candidate = load_authenticated_stage11_clean_candidate(
        root=_stage11_clean_candidate_root(prepared),
        compatibility=compatibility,
    )
    if candidate is None:
        report(
            "Stage 11 resume: running portfolio construction and net "
            "accounting"
        )
        with (
            _portfolio_recheck_reuse_scope(
                execution_inputs=execution_inputs,
                ledger=prepared.walk_forward.ledger,
                selection=prepared.selection,
                development_oof=prepared.development_oof,
                calibration_schedules=prepared.calibration,
            ) as live_recheck_audit,
            recovery._portfolio_allocation_numerical_override(),
            stage11_pending_materiality_scope(),
            stage11_missing_return_projection_scope(
                expected_no_mark_source_count=prepared.resume_amendment[
                    "missing_return_projection"
                ]["expected_exact_no_mark_source_count"],
                expected_no_mark_key_sha256=prepared.resume_amendment[
                    "missing_return_projection"
                ]["expected_exact_no_mark_source_key_sha256"],
            ) as live_missing_audit,
        ):
            live_portfolio = recovery.run_empirical_portfolio_accounting(
                execution_inputs=execution_inputs,
                outcome_publication=prepared.outcome_publication,
                calendar_authority=prepared.calendar,
                ledger=prepared.walk_forward.ledger,
                selection=prepared.selection,
                calibration_schedules=prepared.calibration,
                progress=report,
            )
        execution_inputs.gate()
        candidate = publish_authenticated_stage11_clean_candidate(
            root=_stage11_clean_candidate_root(prepared),
            compatibility=compatibility,
            portfolio=live_portfolio,
            missing_return_projection=live_missing_audit.to_dict(),
            portfolio_recheck_reuse=live_recheck_audit.to_dict(),
            access_ledger_before_portfolio=access_before,
            access_ledger_after_portfolio=(
                execution_inputs.broker.ledger().to_dict()
            ),
        )
        report(
            "Stage 11 resume: authenticated clean portfolio candidate "
            "persisted and independently reopened"
        )
    else:
        report(
            "Stage 11 resume: reusing the authenticated clean portfolio "
            "candidate; portfolio months are not recomputed"
        )
    execution_inputs.gate()
    current_access = execution_inputs.broker.ledger().to_dict()
    access_mode = _require_clean_candidate_access_transition(
        current=current_access,
        before=candidate.access_ledger_before_portfolio,
        after=candidate.access_ledger_after_portfolio,
    )
    return (
        candidate,
        access_mode,
        MappingProxyType(dict(candidate.access_ledger_after_portfolio)),
    )


def _diagnostic_regression_mapping(
    *,
    prepared: _PreparedStage11PortfolioInputs,
    compatibility: Stage11DiagnosticCompatibility,
) -> Mapping[str, str]:
    reports = load_authenticated_failure_reports(
        root=_stage11_diagnostic_root(prepared),
        compatibility=compatibility,
    )
    observed = tuple(
        sorted(
            {
                failure_id
                for report in reports
                for failure_id in report.failure_ids
            }
        )
    )
    registry = json.loads(
        recovery._stable_file_bytes(
            DIAGNOSTIC_REGRESSION_REGISTRY_PATH,
            label="Stage 11 diagnostic regression registry",
            expected_mode=0o644,
        )
    )
    _require(
        set(registry)
        == {
            "schema_version",
            "contract_id",
            "status",
            "failure_regressions",
        }
        and registry["schema_version"] == 1
        and registry["contract_id"]
        == "phase05-stage11-diagnostic-regressions-v1"
        and registry["status"] == "FROZEN_FAILURE_TO_REGRESSION_REGISTRY"
        and isinstance(registry["failure_regressions"], dict),
        "Stage 11 diagnostic regression registry differs",
    )
    entries = registry["failure_regressions"]
    _require(
        set(observed).issubset(entries),
        "every observed diagnostic failure requires a frozen regression",
    )
    result: dict[str, str] = {}
    for failure_id in sorted(entries):
        entry = entries[failure_id]
        _require(
            isinstance(entry, dict)
            and set(entry)
            == {
                "fixture_path",
                "fixture_sha256",
                "regression_test_path",
                "regression_test_sha256",
            },
            "diagnostic regression entry fields differ",
        )
        authenticated_entry: dict[str, str] = {}
        for path_field, hash_field in (
            ("fixture_path", "fixture_sha256"),
            ("regression_test_path", "regression_test_sha256"),
        ):
            relative = Path(entry[path_field])
            _require(
                not relative.is_absolute()
                and ".." not in relative.parts
                and relative.as_posix() == entry[path_field],
                "diagnostic regression path is unsafe",
            )
            path = PROJECT_ROOT.joinpath(*relative.parts)
            _require(
                path.is_file()
                and _sha256_file(path) == entry[hash_field],
                "diagnostic regression artifact differs",
            )
            authenticated_entry[path_field] = entry[path_field]
            authenticated_entry[hash_field] = entry[hash_field]
        if failure_id in observed:
            result[failure_id] = _aggregate_sha256(authenticated_entry)
    return MappingProxyType(result)


def _run_and_record_diagnostic_regression_suite(
    *,
    prepared: _PreparedStage11PortfolioInputs,
) -> str:
    test_hashes = {
        relative: _sha256_file(PROJECT_ROOT / relative)
        for relative in _DIAGNOSTIC_REGRESSION_TEST_PATHS
    }
    command = (
        (PROJECT_ROOT / ".venv/bin/python").as_posix(),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *_DIAGNOSTIC_REGRESSION_TEST_PATHS,
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = (PROJECT_ROOT / "src").as_posix()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    receipt = {
        "schema_version": 1,
        "contract_id": "phase05-stage11-diagnostic-regression-suite-v1",
        "command": list(command[1:]),
        "test_file_sha256": test_hashes,
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "passed": completed.returncode == 0,
        "performance_fields_emitted": 0,
    }
    _require(
        completed.returncode == 0,
        "Stage 11 diagnostic regression suite failed",
    )
    encoded = recovery._canonical_json(receipt)
    digest = hashlib.sha256(encoded).hexdigest()
    root = _stage11_diagnostic_root(prepared)
    evidence_root = root / "regression_evidence"
    if not evidence_root.exists():
        evidence_root.mkdir(mode=0o700)
    _require(
        evidence_root.is_dir()
        and stat.S_IMODE(evidence_root.stat(follow_symlinks=False).st_mode)
        == 0o700,
        "diagnostic regression evidence directory is unsafe",
    )
    destination = evidence_root / f"regression-suite-{digest}.json"
    if not destination.exists():
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    reopened = recovery._stable_file_bytes(
        destination,
        label="Stage 11 diagnostic regression evidence",
        expected_mode=0o600,
    )
    _require(
        reopened == encoded and hashlib.sha256(reopened).hexdigest() == digest,
        "diagnostic regression evidence independent reopen differs",
    )
    return digest


def _publish_diagnostic_completion_attestation(
    *,
    compatibility: Stage11DiagnosticCompatibility,
    completion: DiagnosticCompletionReceipt,
    regression_evidence_sha256: str,
) -> tuple[str, str]:
    _require(
        regression_evidence_sha256 == completion.regression_suite_sha256,
        "diagnostic completion and regression evidence differ",
    )
    payload = {
        "schema_version": 1,
        "contract_id": "phase05-stage11-diagnostic-completion-attestation-v1",
        "status": "DIAGNOSTIC_PATH_COMPLETE_READY_FOR_CLEAN_RUN",
        "diagnostic_compatibility_sha256": compatibility.sha256,
        "completion_sha256": completion.completion_sha256,
        "producer_implementation_sha256": (
            completion.producer_implementation_sha256
        ),
        "latest_checkpoint_sha256": completion.latest_checkpoint_sha256,
        "checkpoint_chain_sha256": completion.checkpoint_chain_sha256,
        "completed_month_count": completion.completed_month_count,
        "observed_failure_ids": list(completion.observed_failure_ids),
        "regression_case_sha256_by_failure_id": dict(
            completion.regression_case_sha256_by_failure_id
        ),
        "regression_suite_sha256": completion.regression_suite_sha256,
        "performance_fields_emitted": 0,
    }
    encoded = recovery._canonical_json(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    root = (
        PROJECT_ROOT
        / "artifacts/phase_05_validation/stage_11_diagnostic_completion"
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"completion-attestation-{digest}.json"
    if not destination.exists():
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    reopened = recovery._stable_file_bytes(
        destination,
        label="Stage 11 diagnostic completion attestation",
        expected_mode=0o644,
    )
    _require(
        reopened == encoded and hashlib.sha256(reopened).hexdigest() == digest,
        "diagnostic completion attestation independent reopen differs",
    )
    return destination.relative_to(PROJECT_ROOT).as_posix(), digest


def _require_diagnostic_completion_for_clean_run(
    *,
    completion: DiagnosticCompletionReceipt,
    current_producer_implementation_sha256: str,
    current_regression_mapping: Mapping[str, str],
    regression_evidence: bytes,
) -> None:
    _require(
        completion.completed_month_count == 59
        and completion.producer_implementation_sha256
        == current_producer_implementation_sha256
        and dict(completion.regression_case_sha256_by_failure_id)
        == dict(current_regression_mapping)
        and hashlib.sha256(regression_evidence).hexdigest()
        == completion.regression_suite_sha256,
        "clean Stage 11 run requires the authenticated 59-month diagnostic seal",
    )


@dataclass(slots=True)
class PortfolioRecheckReuseAudit:
    """Evidence that only the redundant in-process recheck was reused."""

    selection_calls: int
    development_calls: int
    calibration_calls: int
    selection_sha256: str
    selected_scores_sha256: str
    development_observation_count_by_strategy: Mapping[str, int]
    calibration_schedule_sha256_by_strategy: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_reuse_calls": self.selection_calls,
            "development_oof_reuse_calls": self.development_calls,
            "calibration_schedule_reuse_calls": self.calibration_calls,
            "selection_sha256": self.selection_sha256,
            "selected_scores_sha256": self.selected_scores_sha256,
            "development_observation_count_by_strategy": dict(
                sorted(self.development_observation_count_by_strategy.items())
            ),
            "calibration_schedule_sha256_by_strategy": dict(
                sorted(self.calibration_schedule_sha256_by_strategy.items())
            ),
            "forecast_values_changed": False,
            "selection_values_changed": False,
            "calibration_values_changed": False,
        }


@contextmanager
def _diagnostic_missing_return_projection_scope(
) -> Iterator[MissingReturnProjectionAudit]:
    """Retain row audits while deferring only full-path totals to clean run."""

    body_completed = False
    try:
        with stage11_missing_return_projection_scope() as audit:
            yield audit
            body_completed = True
    except Stage11MissingReturnProjectionError as exc:
        if (
            not body_completed
            or str(exc) not in _PARTIAL_DIAGNOSTIC_AGGREGATE_AUDIT_ERRORS
        ):
            raise


@contextmanager
def _portfolio_recheck_reuse_scope(
    *,
    execution_inputs: object,
    ledger: object,
    selection: object,
    development_oof: Mapping[str, object],
    calibration_schedules: Mapping[str, object],
) -> Iterator[PortfolioRecheckReuseAudit]:
    """Reuse only freshly computed canonical objects in the backend recheck."""

    _require(
        portfolio_backend_module.derive_locked_validation_selection
        is _CANONICAL_PORTFOLIO_SELECTION
        and portfolio_backend_module.build_authenticated_development_calibration_observations
        is _CANONICAL_PORTFOLIO_DEVELOPMENT
        and portfolio_backend_module.build_strategy_calibration_schedules
        is _CANONICAL_PORTFOLIO_CALIBRATION,
        "portfolio recheck binding changed before authenticated reuse",
    )
    selection_sha256 = str(selection.selection_sha256)
    selected_scores_sha256 = (
        execution_orchestrator_module.canonical_frame_sha256(
            selection.selected_scores
        )
    )
    development_counts = {
        str(strategy_id): len(observations)
        for strategy_id, observations in development_oof.items()
    }
    schedule_hashes = {
        str(strategy_id): str(schedule.schedule_sha256)
        for strategy_id, schedule in calibration_schedules.items()
    }
    _require(
        len(selection_sha256) == 64
        and len(selected_scores_sha256) == 64
        and development_counts
        == {
            PRIMARY_STRATEGY_ID: 172_799,
            ELASTIC_NET_CHALLENGER_ID: 172_799,
        }
        and set(schedule_hashes)
        == {PRIMARY_STRATEGY_ID, ELASTIC_NET_CHALLENGER_ID}
        and all(len(value) == 64 for value in schedule_hashes.values()),
        "fresh canonical portfolio recheck objects differ",
    )
    audit = PortfolioRecheckReuseAudit(
        selection_calls=0,
        development_calls=0,
        calibration_calls=0,
        selection_sha256=selection_sha256,
        selected_scores_sha256=selected_scores_sha256,
        development_observation_count_by_strategy=MappingProxyType(
            development_counts
        ),
        calibration_schedule_sha256_by_strategy=MappingProxyType(
            schedule_hashes
        ),
    )

    def selection_adapter(candidate_ledger: object) -> object:
        audit.selection_calls += 1
        _require(
            audit.selection_calls == 1
            and candidate_ledger is ledger
            and selection.selection_sha256 == selection_sha256
            and execution_orchestrator_module.canonical_frame_sha256(
                selection.selected_scores
            )
            == selected_scores_sha256,
            "portfolio selection recheck reuse authority differs",
        )
        return selection

    def development_adapter(
        candidate_inputs: object,
        candidate_selection: object,
    ) -> Mapping[str, object]:
        audit.development_calls += 1
        _require(
            audit.development_calls == 1
            and candidate_inputs is execution_inputs
            and candidate_selection is selection
            and {
                str(strategy_id): len(observations)
                for strategy_id, observations in development_oof.items()
            }
            == development_counts,
            "portfolio development-OOF recheck reuse authority differs",
        )
        return development_oof

    def calibration_adapter(
        candidate_selection: object,
        candidate_ledger: object,
        *,
        development_oof: Mapping[str, object],
    ) -> Mapping[str, object]:
        audit.calibration_calls += 1
        _require(
            audit.calibration_calls == 1
            and candidate_selection is selection
            and candidate_ledger is ledger
            and development_oof is development_oof_authority
            and {
                str(strategy_id): str(schedule.schedule_sha256)
                for strategy_id, schedule in calibration_schedules.items()
            }
            == schedule_hashes,
            "portfolio calibration recheck reuse authority differs",
        )
        return calibration_schedules

    development_oof_authority = development_oof
    portfolio_backend_module.derive_locked_validation_selection = (
        selection_adapter
    )
    portfolio_backend_module.build_authenticated_development_calibration_observations = (
        development_adapter
    )
    portfolio_backend_module.build_strategy_calibration_schedules = (
        calibration_adapter
    )
    try:
        yield audit
        _require(
            audit.selection_calls
            == audit.development_calls
            == audit.calibration_calls
            == 1
            and selection.selection_sha256 == selection_sha256
            and execution_orchestrator_module.canonical_frame_sha256(
                selection.selected_scores
            )
            == selected_scores_sha256
            and {
                str(strategy_id): str(schedule.schedule_sha256)
                for strategy_id, schedule in calibration_schedules.items()
            }
            == schedule_hashes,
            "portfolio recheck reuse was incomplete or changed content",
        )
    finally:
        changed = (
            portfolio_backend_module.derive_locked_validation_selection
            is not selection_adapter
            or portfolio_backend_module.build_authenticated_development_calibration_observations
            is not development_adapter
            or portfolio_backend_module.build_strategy_calibration_schedules
            is not calibration_adapter
        )
        portfolio_backend_module.derive_locked_validation_selection = (
            _CANONICAL_PORTFOLIO_SELECTION
        )
        portfolio_backend_module.build_authenticated_development_calibration_observations = (
            _CANONICAL_PORTFOLIO_DEVELOPMENT
        )
        portfolio_backend_module.build_strategy_calibration_schedules = (
            _CANONICAL_PORTFOLIO_CALIBRATION
        )
        _require(
            not changed,
            "portfolio recheck binding changed during authenticated reuse",
        )


def _load_resume_amendment(
    path: Path = RESUME_AMENDMENT_PATH,
) -> tuple[Mapping[str, Any], str]:
    payload = recovery._stable_file_bytes(
        path,
        label="Stage 11 portfolio-resume amendment",
        expected_mode=0o644,
    )
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11PortfolioResumeError(
            "Stage 11 portfolio-resume amendment is invalid"
        ) from exc
    required = {
        "schema_version",
        "contract_id",
        "stage",
        "methodology_version",
        "status",
        "prior_failed_execution",
        "resume_authority",
        "missing_return_projection",
        "retired_identity_execution_gate",
        "pending_identity_materiality_policy",
        "diagnostic_compatibility_transition",
        "frozen_invariants",
    }
    _require(
        isinstance(parsed, Mapping)
        and set(parsed) == required
        and parsed["schema_version"] == 1
        and parsed["contract_id"]
        == "phase05-stage11-portfolio-resume-amendment-v1"
        and parsed["stage"] == 11
        and parsed["methodology_version"] == "v1"
        and parsed["status"] == "FROZEN_RULE_RESTORING_PORTFOLIO_PHASE_RESUME",
        "Stage 11 portfolio-resume amendment identity differs",
    )
    prior = parsed["prior_failed_execution"]
    resume = parsed["resume_authority"]
    projection = parsed["missing_return_projection"]
    retirement = parsed["retired_identity_execution_gate"]
    pending_policy = parsed["pending_identity_materiality_policy"]
    transition = parsed["diagnostic_compatibility_transition"]
    invariants = parsed["frozen_invariants"]
    _require(
        isinstance(prior, Mapping)
        and set(prior)
        == {
            "failure_receipt_path",
            "failure_receipt_sha256",
            "failure_seal_path",
            "failure_seal_sha256",
            "preflight_primary_pointer_sha256",
            "preflight_verified_pointer_sha256",
            "preceding_held_march_failure_receipt_path",
            "preceding_held_march_failure_receipt_sha256",
            "preceding_march_failure_receipt_path",
            "preceding_march_failure_receipt_sha256",
            "originating_february_failure_receipt_path",
            "originating_february_failure_receipt_sha256",
            "completed_refit_count",
            "completed_selector_fit_count",
            "completed_forecast_equality_check_count",
            "completed_forecast_rows",
            "completed_monthly_score_count",
            "failure_holding_month",
            "portfolio_months_completed",
            "failure_message",
            "failure_scope",
            "result_bundle_published",
            "outcome_root",
        },
        "prior Stage 11 failed-execution fields differ",
    )
    _require(
        isinstance(resume, Mapping)
        and set(resume)
        == {
            "mode",
            "repeat_model_fitting_permitted",
            "exact_cache_tree_authentication_required",
            "normal_pseudo_live_ledger_replay_required",
            "normal_selection_calibration_portfolio_publication_and_verification_required",
            "fresh_preflight_required",
            "performance_preserving_replay_optimizations",
        },
        "Stage 11 portfolio-resume authority fields differ",
    )
    replay_optimizations = resume["performance_preserving_replay_optimizations"]
    _require(
        isinstance(replay_optimizations, Mapping)
        and set(replay_optimizations)
        == {
            "authenticated_forecast_arrays_reused_in_memory",
            "completed_outcome_hash_computed_once_per_refit",
            "forecast_domain_scanned_once_per_signal_date",
            "fresh_canonical_portfolio_recheck_objects_reused_once",
            "portfolio_recheck_requires_exact_object_identity_and_content_hashes",
            "portfolio_recheck_bindings_restored_before_publication",
            "forecast_values_changed",
        },
        "Stage 11 replay-optimization fields differ",
    )
    _require(
        prior["completed_refit_count"] == 20
        and prior["completed_selector_fit_count"] == 660
        and prior["completed_forecast_equality_check_count"] == 1_947
        and prior["completed_forecast_rows"] == 4_013_460
        and prior["completed_monthly_score_count"] == 59
        and prior["failure_holding_month"] == "2015-07"
        and prior["portfolio_months_completed"] == 6
        and prior["failure_scope"]
        == "elastic_net_dual_certificate_exact_metric_capture_required"
        and prior["result_bundle_published"] is False
        and prior["failure_message"]
        == (
            "elastic_net_hybrid_challenger allocation failed: ordinary "
            "hybrid retry candidate was not dual-certified: "
            "ordinary_primary:optimal_inaccurate, "
            "ordinary_retry:optimal_inaccurate, "
            "ordinary_feasibility_retry:optimal, "
            "ordinary_explicit_dual_certificate:optimal, "
            "ordinary_explicit_dual_certificate_source:"
            "separately_solved_explicit_dual, "
            "ordinary_explicit_dual_certificate_audit:"
            "failed:candidate_dual_gap, "
            "ordinary_retry_dual_certificate:optimal_inaccurate, "
            "ordinary_retry_dual_certificate_source:"
            "clarabel_retry_canonical_dual_lsmr_refined, "
            "ordinary_retry_dual_certificate_audit:failed:dual_soc_cone"
        ),
        "prior Stage 11 failed-execution evidence differs",
    )
    _require(
        resume["mode"]
        == "authenticated_checkpoint_and_raw_forecast_replay_after_completed_exact_recomputation"
        and resume["repeat_model_fitting_permitted"] is False
        and resume["exact_cache_tree_authentication_required"] is True
        and resume["normal_pseudo_live_ledger_replay_required"] is True
        and resume["normal_selection_calibration_portfolio_publication_and_verification_required"]
        is True
        and resume["fresh_preflight_required"] is True
        and replay_optimizations[
            "authenticated_forecast_arrays_reused_in_memory"
        ]
        is True
        and replay_optimizations[
            "completed_outcome_hash_computed_once_per_refit"
        ]
        is True
        and replay_optimizations["forecast_domain_scanned_once_per_signal_date"]
        is True
        and replay_optimizations[
            "fresh_canonical_portfolio_recheck_objects_reused_once"
        ]
        is True
        and replay_optimizations[
            "portfolio_recheck_requires_exact_object_identity_and_content_hashes"
        ]
        is True
        and replay_optimizations[
            "portfolio_recheck_bindings_restored_before_publication"
        ]
        is True
        and replay_optimizations["forecast_values_changed"] is False,
        "Stage 11 portfolio-resume authority differs",
    )
    outcome_root = prior["outcome_root"]
    _require(
        isinstance(outcome_root, Mapping)
        and set(outcome_root)
        == {
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
        and outcome_root["authority_build_id"]
        == "stage11-preflight-d02668012a550a9a1b5b17c4"
        and outcome_root["authority_verification_id"]
        == "verify-d2c8a62968a1e3ac"
        and outcome_root["directory_mode_counts"] == {"0755": 1}
        and outcome_root["file_count"] == 29
        and outcome_root["file_mode"] == "0644"
        and outcome_root["inventory_sha256"]
        == "abe776e6382ac9bff16fe524d81a6f4692f57b498b7b0427382a7103d6601a41"
        and outcome_root["outcome_materialization_receipt_sha256"]
        == "862ff81a33d23f9e5e9a06f29bd767cd93c5dd4d2f3326071fb1bd2f682aac03"
        and outcome_root["table_receipts_equal_frozen_contract"] is True
        and outcome_root["total_size_bytes"] == 156_420_966,
        "prior Stage 11 failed-execution outcome root differs",
    )
    _require(
        isinstance(projection, Mapping)
        and set(projection)
        == {
            "classification",
            "failure_receipt_path",
            "failure_receipt_sha256",
            "adapter_path",
            "adapter_sha256",
            "test_path",
            "test_sha256",
            "daily_engine_path",
            "daily_engine_sha256",
            "daily_engine_test_path",
            "daily_engine_test_sha256",
            "portfolio_backend_path",
            "portfolio_backend_sha256",
            "execution_orchestrator_path",
            "execution_orchestrator_sha256",
            "execution_orchestrator_test_path",
            "execution_orchestrator_test_sha256",
            "authoritative_null_to_zero_permitted",
            "invalid_nonnull_numeric_suppression_permitted",
            "open_nonzero_position_without_return",
            "expected_exact_no_mark_source_count",
            "expected_exact_no_mark_source_key_sha256",
            "observed_return_span_start_continuity_required",
            "pending_identity_rebalance_trade_permitted",
            "pending_identity_corporate_action_permitted",
            "disconnected_or_overlapping_return_span_permitted",
            "final_nonzero_pending_identity_permitted",
            "unresolved_terminal_action_projection",
            "expected_unresolved_terminal_action_count",
            "expected_unresolved_terminal_return_twin_count",
            "expected_unresolved_terminal_key_sha256",
            "expected_resolved_canonical_action_count",
            "unresolved_action_or_settlement_creation_permitted",
            "canonical_resolved_action_mapping_required",
            "projection_bindings_restored_before_publication",
            "corporate_action_economic_treatment_changed",
        }
        and projection["classification"]
        == "authenticated_no_mark_projection_with_exact_span_continuity"
        and projection["authoritative_null_to_zero_permitted"] is False
        and projection["invalid_nonnull_numeric_suppression_permitted"] is False
        and projection["open_nonzero_position_without_return"]
        == "exact_missing_return_is_explicit_no_mark_all_other_nulls_fail_closed"
        and projection["expected_exact_no_mark_source_count"] == 66_198
        and projection["expected_exact_no_mark_source_key_sha256"]
        == "1d2eea9164a8030341c68c35c596f8bf9cab1cb910c68b57d875c063986f6978"
        and projection["observed_return_span_start_continuity_required"]
        is True
        and projection["pending_identity_rebalance_trade_permitted"] is False
        and projection["pending_identity_corporate_action_permitted"] is False
        and projection["disconnected_or_overlapping_return_span_permitted"]
        is False
        and projection["final_nonzero_pending_identity_permitted"] is False
        and projection["unresolved_terminal_action_projection"]
        == "audit_only_with_exact_nullable_return_twin"
        and projection["expected_unresolved_terminal_action_count"] == 20
        and projection["expected_unresolved_terminal_return_twin_count"] == 20
        and projection["expected_unresolved_terminal_key_sha256"]
        == "3888ef6b7e50a3eed374aeb1d53af00df8172d0cf1b62bff86efcdec92473bec"
        and projection["expected_resolved_canonical_action_count"] == 2_275
        and projection["unresolved_action_or_settlement_creation_permitted"]
        is False
        and projection["canonical_resolved_action_mapping_required"] is True
        and projection["projection_bindings_restored_before_publication"]
        is True
        and projection["corporate_action_economic_treatment_changed"] is False,
        "Stage 11 missing-return projection rule differs",
    )
    _require(
        isinstance(retirement, Mapping)
        and set(retirement)
        == {
            "classification",
            "trigger_failure_message",
            "trigger_holding_month",
            "trigger_session_date",
            "trigger_permno",
            "result_bundle_published",
            "portfolio_backend_path",
            "portfolio_backend_sha256",
            "test_path",
            "test_sha256",
            "signal_score_or_ic_domain_changed",
            "signal_date_universe_membership_changed",
            "execution_gate_timing",
            "future_terminal_event_permitted",
            "suppressed_posting_promoted_to_return_or_no_mark",
            "terminal_action_reapplied",
            "expected_execution_gate_rows",
            "expected_unique_permnos",
            "expected_execution_gate_sha256",
            "expected_unresolved_terminal_gate_rows",
        }
        and retirement["classification"]
        == "authenticated_terminal_identity_new_trade_execution_gate"
        and retirement["trigger_failure_message"]
        == "open identities lack an authoritative daily return/rate: 89558"
        and retirement["trigger_holding_month"] == "2015-03"
        and retirement["trigger_session_date"] == "2015-03-03"
        and retirement["trigger_permno"] == 89_558
        and retirement["result_bundle_published"] is False
        and retirement["signal_score_or_ic_domain_changed"] is False
        and retirement["signal_date_universe_membership_changed"] is False
        and retirement["execution_gate_timing"]
        == "terminal_action_assigned_on_or_before_entry_close"
        and retirement["future_terminal_event_permitted"] is False
        and retirement["suppressed_posting_promoted_to_return_or_no_mark"]
        is False
        and retirement["terminal_action_reapplied"] is False
        and retirement["expected_execution_gate_rows"]
        == _EXPECTED_RETIRED_EXECUTION_GATE_ROWS
        and retirement["expected_unique_permnos"]
        == _EXPECTED_RETIRED_EXECUTION_GATE_ROWS
        and retirement["expected_execution_gate_sha256"]
        == _EXPECTED_RETIRED_EXECUTION_GATE_SHA256
        and retirement["expected_unresolved_terminal_gate_rows"] == 0,
        "Stage 11 retired-identity execution gate differs",
    )
    pending_fixtures = (
        pending_policy.get("failure_fixtures")
        if isinstance(pending_policy, Mapping)
        else None
    )
    pending_migration = (
        pending_policy.get("checkpoint_tip_migration")
        if isinstance(pending_policy, Mapping)
        else None
    )
    _require(
        isinstance(pending_policy, Mapping)
        and set(pending_policy)
        == {
            "classification",
            "policy_id",
            "application_scope",
            "strict_absolute_position_value_dollars_upper_bound",
            "normalized_required_exit_violation_upper_bound",
            "pending_no_mark_count_must_be_positive",
            "nav_preserved_bit_for_bit",
            "position_value_set_to_exact_zero",
            "mark_continuity_identity_removed",
            "synthetic_trade_permitted",
            "transaction_cost_permitted",
            "execution_price_read_permitted",
            "return_imputation_permitted",
            "corporate_action_permitted",
            "terminal_identity_retirement_permitted",
            "position_at_or_above_five_cents_permitted",
            "return_span_continuity_limit_changed",
            "expected_evidence_rows",
            "expected_unique_permnos",
            "expected_strategy_ids_by_permno",
            "expected_event_key_sha256",
            "expected_evidence_sha256",
            "one_ulp_pending_weight_comparison",
            "one_ulp_comparison_tolerance",
            "next_representable_weight_change_permitted",
            "failure_ids",
            "failure_fixtures",
            "checkpoint_tip_migration",
        }
        and pending_policy["classification"]
        == (
            "uniform_pending_five_cent_post_month_boundary_retirement_"
            "with_authenticated_tip_migration"
        )
        and pending_policy["policy_id"] == PENDING_MATERIALITY_POLICY_ID
        and pending_policy["application_scope"]
        == (
            "all_strategies_all_holding_months_and_authenticated_"
            "checkpoint_045_tip"
        )
        and pending_policy[
            "strict_absolute_position_value_dollars_upper_bound"
        ]
        == PENDING_MATERIALITY_THRESHOLD_DOLLARS
        and pending_policy[
            "normalized_required_exit_violation_upper_bound"
        ]
        == PENDING_MATERIALITY_NORMALIZED_REQUIRED_EXIT_LIMIT
        and pending_policy["pending_no_mark_count_must_be_positive"] is True
        and pending_policy["nav_preserved_bit_for_bit"] is True
        and pending_policy["position_value_set_to_exact_zero"] is True
        and pending_policy["mark_continuity_identity_removed"] is True
        and all(
            pending_policy[name] is False
            for name in (
                "synthetic_trade_permitted",
                "transaction_cost_permitted",
                "execution_price_read_permitted",
                "return_imputation_permitted",
                "corporate_action_permitted",
                "terminal_identity_retirement_permitted",
                "position_at_or_above_five_cents_permitted",
                "return_span_continuity_limit_changed",
                "next_representable_weight_change_permitted",
            )
        )
        and pending_policy["expected_evidence_rows"]
        == _EXPECTED_PENDING_DUST_EVIDENCE_ROWS
        and pending_policy["expected_unique_permnos"]
        == list(_EXPECTED_PENDING_DUST_UNIQUE_PERMNOS)
        and pending_policy["expected_strategy_ids_by_permno"]
        == {
            str(permno): list(expected["strategy_ids"])
            for permno, expected in sorted(
                _EXPECTED_PENDING_DUST_BY_PERMNO.items()
            )
        }
        and pending_policy["expected_event_key_sha256"]
        == _EXPECTED_PENDING_DUST_EVENT_KEY_SHA256
        and pending_policy["expected_evidence_sha256"]
        == _EXPECTED_PENDING_DUST_EVIDENCE_SHA256
        and pending_policy["one_ulp_pending_weight_comparison"]
        == "exact_opening_weight_equals_target_weight"
        and pending_policy["one_ulp_comparison_tolerance"] == 0.0
        and pending_policy["failure_ids"]
        == [
            "3ec463f0fe9ca11ee7e89063eccd45033724a62f38b35dcdde0d8a71e48b9ce7",
            "4bf2a4eccdef94ff1c6373fb32f3416b5c2c58689a438ad4bfaa8216c405b8b8",
            "31c79c48ef576f37159215ed64c1ae2f9f386027754de846f8d5847c5de4a54e",
        ],
        "Stage 11 pending-identity materiality policy differs",
    )
    _require(
        isinstance(pending_fixtures, list)
        and pending_fixtures
        == [
            {
                "path": (
                    "tests/fixtures/stage11_failures/"
                    "2017-04-pending-weight-roundtrip/manifest.json"
                ),
                "sha256": (
                    "9d46864663946c1287989c98b38b0296f879b8c9df16a69a9"
                    "bf656b3438c685b"
                ),
            },
            {
                "path": (
                    "tests/fixtures/stage11_failures/"
                    "2018-10-pending-dust-reentry/manifest.json"
                ),
                "sha256": (
                    "2408d05d8755de324af71ffb349a1713e5a884ae5c4f191"
                    "871128e55271d0c47"
                ),
            },
        ],
        "Stage 11 pending-identity failure fixtures differ",
    )
    _require(
        isinstance(pending_migration, Mapping)
        and set(pending_migration)
        == {
            "adapter_path",
            "adapter_sha256",
            "regression_test_path",
            "regression_test_sha256",
            "source_compatibility_sha256",
            "source_checkpoint_sha256",
            "source_completed_month",
            "source_completed_month_count",
            "next_holding_month",
            "boundary_date",
            "checkpoint_rewritten",
            "completed_month_count_changed",
            "prior_month_replay_permitted",
            "strategy_id",
            "permno",
            "position_value_dollars_hex",
            "opening_nav_dollars_hex",
            "weight_hex",
            "last_authoritative_mark_date",
            "pending_no_mark_count",
            "pending_evidence_sha256",
            "expected_migration_evidence_sha256",
        }
        and pending_migration["source_compatibility_sha256"]
        == "32e6f242b123412ccf077ffecbb809b369b2d6942bd73a620106ceaf2a953c53"
        and pending_migration["source_checkpoint_sha256"]
        == MIGRATION_SOURCE_CHECKPOINT_SHA256
        and pending_migration["source_completed_month"]
        == MIGRATION_SOURCE_COMPLETED_MONTH
        and pending_migration["source_completed_month_count"]
        == MIGRATION_NEXT_MONTH_INDEX
        and pending_migration["next_holding_month"]
        == MIGRATION_NEXT_HOLDING_MONTH
        and pending_migration["boundary_date"]
        == MIGRATION_BOUNDARY_DATE.isoformat()
        and pending_migration["checkpoint_rewritten"] is False
        and pending_migration["completed_month_count_changed"] is False
        and pending_migration["prior_month_replay_permitted"] is False
        and pending_migration["strategy_id"] == MIGRATION_STRATEGY_ID
        and pending_migration["permno"] == MIGRATION_PERMNO
        and pending_migration["position_value_dollars_hex"]
        == MIGRATION_POSITION_VALUE_DOLLARS_HEX
        and pending_migration["opening_nav_dollars_hex"]
        == MIGRATION_OPENING_NAV_DOLLARS_HEX
        and pending_migration["weight_hex"]
        == (
            float.fromhex(MIGRATION_POSITION_VALUE_DOLLARS_HEX)
            / float.fromhex(MIGRATION_OPENING_NAV_DOLLARS_HEX)
        ).hex()
        and pending_migration["last_authoritative_mark_date"]
        == MIGRATION_LAST_AUTHORITATIVE_MARK_DATE.isoformat()
        and pending_migration["pending_no_mark_count"]
        == MIGRATION_PENDING_NO_MARK_COUNT
        and pending_migration["pending_evidence_sha256"]
        == MIGRATION_PENDING_EVIDENCE_SHA256
        and pending_migration["expected_migration_evidence_sha256"]
        == MIGRATION_EVIDENCE_SHA256,
        "Stage 11 pending-materiality checkpoint migration differs",
    )
    _require(
        isinstance(transition, Mapping)
        and set(transition)
        == {
            "classification",
            "prior_compatibility_sha256",
            "prior_accounting_transition_code_sha256",
            "prior_completed_month_count",
            "prior_latest_holding_month",
            "prior_next_holding_month",
            "prior_latest_checkpoint_sha256",
            "archive_relative_to_execution_results",
            "archive_tree",
            "archive_reverified_after_atomic_move",
            "prior_tree_reuse_permitted",
            "replacement_first_holding_month",
            "cached_forecasts_and_model_fits_reused",
            "repeat_model_fitting_permitted",
        }
        and transition["classification"]
        == "protected_accounting_identity_restart_with_whole_tree_archive"
        and transition["prior_compatibility_sha256"]
        == "854aa14aa9cd50cf718f6cb44b9cb8d0653c8ad5eef46e0ee8fa0e4d0478011d"
        and transition["prior_accounting_transition_code_sha256"]
        == "89a0e3ee2bc3638a7bcc46bb69401204a3dfad68738ea5e2991684ddaf4afbcf"
        and transition["prior_completed_month_count"] == 27
        and transition["prior_latest_holding_month"] == "2017-03"
        and transition["prior_next_holding_month"] == "2017-04"
        and transition["prior_latest_checkpoint_sha256"]
        == "b07a59790621c6c451fd499203461c9fd8c8c9ac8798ec698e8f8030b9346a1d"
        and transition["archive_relative_to_execution_results"]
        == (
            "diagnostic_archives/compatibility-854aa14aa9cd50cf-"
            "inventory-d3894ca0e15c633a"
        )
        and transition["archive_reverified_after_atomic_move"] is True
        and transition["prior_tree_reuse_permitted"] is False
        and transition["replacement_first_holding_month"] == "2015-01"
        and transition["cached_forecasts_and_model_fits_reused"] is True
        and transition["repeat_model_fitting_permitted"] is False,
        "Stage 11 diagnostic compatibility transition differs",
    )
    archive_tree = transition["archive_tree"]
    _require(
        isinstance(archive_tree, Mapping)
        and archive_tree
        == {
            "file_count": 279,
            "total_size_bytes": 1_641_191_747,
            "inventory_sha256": (
                "d3894ca0e15c633aabfa6ba07f658fcaf613ed83bf5218aedeaaed651108c345"
            ),
            "directory_mode_counts": {"0700": 67},
            "file_mode": "0600",
        },
        "Stage 11 archived diagnostic tree identity differs",
    )
    _require(
        all(
            invariants[name] is False
            for name in (
                "allocation_constraints_changed",
                "allocation_costs_changed",
                "allocation_objective_changed",
                "borrow_or_financing_cost_changed",
                "forecast_value_changed",
                "portfolio_selection_rule_changed",
                "return_value_imputed",
                "solver_changed",
            )
        )
        and invariants["final_2020_plus_data_permitted"] is False
        and invariants["stage12_or_later_data_permitted"] is False,
        "Stage 11 portfolio resume changed a frozen economic rule",
    )
    for owner, relative_key, sha_key, label in (
        (
            prior,
            "failure_receipt_path",
            "failure_receipt_sha256",
            "latest failure receipt",
        ),
        (
            prior,
            "failure_seal_path",
            "failure_seal_sha256",
            "latest failure seal",
        ),
        (
            prior,
            "preceding_held_march_failure_receipt_path",
            "preceding_held_march_failure_receipt_sha256",
            "preceding held-position March failure receipt",
        ),
        (
            prior,
            "preceding_march_failure_receipt_path",
            "preceding_march_failure_receipt_sha256",
            "preceding March failure receipt",
        ),
        (
            prior,
            "originating_february_failure_receipt_path",
            "originating_february_failure_receipt_sha256",
            "originating February failure receipt",
        ),
        (
            projection,
            "failure_receipt_path",
            "failure_receipt_sha256",
            "missing-return failure receipt",
        ),
        (
            projection,
            "adapter_path",
            "adapter_sha256",
            "missing-return adapter",
        ),
        (
            projection,
            "test_path",
            "test_sha256",
            "missing-return adapter test",
        ),
        (
            projection,
            "daily_engine_path",
            "daily_engine_sha256",
            "daily engine",
        ),
        (
            projection,
            "daily_engine_test_path",
            "daily_engine_test_sha256",
            "daily engine test",
        ),
        (
            projection,
            "portfolio_backend_path",
            "portfolio_backend_sha256",
            "empirical portfolio backend",
        ),
        (
            projection,
            "execution_orchestrator_path",
            "execution_orchestrator_sha256",
            "execution orchestrator",
        ),
        (
            projection,
            "execution_orchestrator_test_path",
            "execution_orchestrator_test_sha256",
            "execution orchestrator test",
        ),
        (
            retirement,
            "portfolio_backend_path",
            "portfolio_backend_sha256",
            "retired-identity execution gate implementation",
        ),
        (
            retirement,
            "test_path",
            "test_sha256",
            "retired-identity execution gate test",
        ),
    ):
        relative = Path(str(owner[relative_key]))
        expected_sha = str(owner[sha_key])
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and len(expected_sha) == 64
            and set(expected_sha).issubset(recovery._HEX64)
            and hashlib.sha256(
                recovery._stable_file_bytes(
                    PROJECT_ROOT.joinpath(*relative.parts),
                    label=label,
                    expected_mode=0o644,
                )
            ).hexdigest()
            == expected_sha,
            f"Stage 11 {label} binding differs",
        )
    for fixture, expected_mode in zip(
        pending_fixtures,
        (0o644, 0o600),
        strict=True,
    ):
        relative = Path(str(fixture["path"]))
        expected_sha = str(fixture["sha256"])
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and len(expected_sha) == 64
            and set(expected_sha).issubset(recovery._HEX64)
            and hashlib.sha256(
                recovery._stable_file_bytes(
                    PROJECT_ROOT.joinpath(*relative.parts),
                    label="pending-identity materiality failure fixture",
                    expected_mode=expected_mode,
                )
            ).hexdigest()
            == expected_sha,
            "Stage 11 pending-identity materiality fixture binding differs",
        )
    for relative_key, sha_key, label in (
        (
            "adapter_path",
            "adapter_sha256",
            "pending-materiality adapter",
        ),
        (
            "regression_test_path",
            "regression_test_sha256",
            "pending-materiality regression test",
        ),
    ):
        relative = Path(str(pending_migration[relative_key]))
        expected_sha = str(pending_migration[sha_key])
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and len(expected_sha) == 64
            and set(expected_sha).issubset(recovery._HEX64)
            and hashlib.sha256(
                recovery._stable_file_bytes(
                    PROJECT_ROOT.joinpath(*relative.parts),
                    label=label,
                    expected_mode=0o644,
                )
            ).hexdigest()
            == expected_sha,
            f"Stage 11 {label} binding differs",
        )
    failure_receipt = json.loads(
        recovery._stable_file_bytes(
            PROJECT_ROOT.joinpath(*Path(prior["failure_receipt_path"]).parts),
            label="latest failure receipt cross-check",
            expected_mode=0o644,
        )
    )
    _require(
        failure_receipt.get("authority")
        == {
            "preflight_build_id": outcome_root["authority_build_id"],
            "preflight_primary_pointer_sha256": prior[
                "preflight_primary_pointer_sha256"
            ],
            "preflight_verification_id": outcome_root[
                "authority_verification_id"
            ],
            "preflight_verified_pointer_sha256": prior[
                "preflight_verified_pointer_sha256"
            ],
        }
        and failure_receipt.get("outcome_root") == outcome_root
        and failure_receipt.get("failure", {}).get("holding_month")
        == prior["failure_holding_month"]
        and failure_receipt.get("failure", {}).get("message")
        == prior["failure_message"]
        and failure_receipt.get("failure", {}).get(
            "portfolio_months_completed_before_failure"
        )
        == prior["portfolio_months_completed"],
        "latest Stage 11 failure receipt provenance differs",
    )
    failure_seal = json.loads(
        recovery._stable_file_bytes(
            PROJECT_ROOT.joinpath(*Path(prior["failure_seal_path"]).parts),
            label="latest failure seal cross-check",
            expected_mode=0o644,
        )
    )
    _require(
        failure_seal
        == {
            "attempt_id": "attempt-20260724-elastic-net-dual-certificate-2015-07",
            "canonical_result_published": False,
            "failure_receipt_sha256": prior["failure_receipt_sha256"],
            "status": "SEALED_NO_RESULT_PUBLISHED",
        },
        "latest Stage 11 failure seal differs",
    )
    return MappingProxyType(dict(parsed)), hashlib.sha256(payload).hexdigest()


class AuthenticatedStage11ForecastReplayBackend(
    recovery.AuthenticatedStage11CacheRecoveryBackend
):
    """Replay only previously authenticated forecasts through the normal ledger."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        progress = kwargs.get("progress")
        self._replay_progress = (
            progress if callable(progress) else (lambda _message: None)
        )
        super().__init__(*args, **kwargs)
        self.replayed_fit_reference_count = 0
        self.replayed_forecast_receipt_count = 0
        self.replayed_forecast_rows = 0
        self.replayed_month_count = 0
        self._completed_outcome_sha_by_refit: dict[dt.date, str] = {}
        self._completed_outcome_rows_by_refit: dict[dt.date, int] = {}

    def fit_selector(
        self,
        job: SelectorFitJob,
        completed_outcomes: pd.DataFrame,
    ) -> str:
        """Return an exact authenticated checkpoint reference; never refit."""

        self.gate()
        _require(
            isinstance(job, SelectorFitJob)
            and isinstance(completed_outcomes, pd.DataFrame)
            and set(OUTCOME_COLUMNS).issubset(completed_outcomes.columns),
            "replayed selector fit inputs differ",
        )
        if not completed_outcomes.empty:
            availability = pd.to_datetime(
                completed_outcomes["outcome_availability_date"], errors="raise"
            ).dt.date
            _require(
                availability.max() <= job.refit_signal_date,
                "replayed checkpoint would see a future validation outcome",
            )
        completed_sha = self._completed_outcome_sha_by_refit.get(
            job.refit_signal_date
        )
        if completed_sha is None:
            completed_sha = execution_orchestrator_module.canonical_frame_sha256(
                completed_outcomes.loc[:, list(OUTCOME_COLUMNS)]
            )
            self._completed_outcome_sha_by_refit[job.refit_signal_date] = (
                completed_sha
            )
            self._completed_outcome_rows_by_refit[job.refit_signal_date] = len(
                completed_outcomes
            )
        else:
            _require(
                self._completed_outcome_rows_by_refit[job.refit_signal_date]
                == len(completed_outcomes),
                "completed outcome rows changed within one replayed refit",
            )
        key = (
            job.refit_signal_date,
            job.model_family,
            job.config_id,
            job.seed,
        )
        checkpoint = self._cached_checkpoints.get(key)
        _require(checkpoint is not None, "replayed checkpoint reference is absent")
        identity = checkpoint.identity
        training = identity.get("training_data")
        _require(
            isinstance(training, Mapping)
            and dt.date.fromisoformat(str(training["max_exit_date"]))
            <= job.refit_signal_date
            and dt.date.fromisoformat(
                str(training["max_outcome_availability_date"])
            )
            <= job.refit_signal_date,
            "replayed checkpoint training chronology differs",
        )
        self.replayed_fit_reference_count += 1
        self.gate()
        return checkpoint.manifest_sha256

    def score_month(
        self,
        step: MonthlyExecutionStep,
        checkpoints: Mapping[tuple[str, str, int | None], str],
    ) -> pd.DataFrame:
        """Reopen the exact authenticated 33 forecast payloads for one month."""

        self.gate()
        _require(
            isinstance(step, MonthlyExecutionStep),
            "replayed monthly scoring step differs",
        )
        jobs = tuple(
            job
            for job in self._plan.fit_jobs
            if job.refit_signal_date == step.active_refit_signal_date
        )
        _require(
            len(jobs) == EXPECTED_FITS_PER_REFIT,
            "replayed monthly forecast job grid differs",
        )
        rows: list[pd.DataFrame] = []
        expected_permnos: np.ndarray | None = None
        for job in jobs:
            model_key = (job.model_family, job.config_id, job.seed)
            receipt = self._cached_forecasts.get(
                (
                    step.calendar_row.signal_date,
                    job.model_family,
                    job.config_id,
                    job.seed,
                )
            )
            _require(
                receipt is not None
                and checkpoints.get(model_key)
                == receipt.checkpoint_manifest_sha256,
                "replayed forecast/checkpoint binding differs",
            )
            retained = self._authenticated_forecast_arrays.get(
                (
                    step.calendar_row.signal_date,
                    job.model_family,
                    job.config_id,
                    job.seed,
                )
            )
            _require(
                retained is not None,
                "authenticated in-memory forecast payload is absent",
            )
            permnos = np.asarray(retained[0], dtype=np.int64)
            forecasts = np.asarray(retained[1], dtype=np.float32)
            if expected_permnos is None:
                expected_permnos = permnos.copy()
            _require(
                np.array_equal(permnos, expected_permnos)
                and permnos.size == receipt.rows
                and forecasts.shape == permnos.shape
                and np.isfinite(forecasts).all()
                and array_sha256(permnos) == receipt.permnos_sha256
                and array_sha256(forecasts) == receipt.raw_forecasts_sha256,
                "replayed forecast payload differs from its authenticated receipt",
            )
            rows.append(
                pd.DataFrame(
                    {
                        "model_family": job.model_family,
                        "config_id": job.config_id,
                        "seed": job.seed,
                        "signal_date": step.calendar_row.signal_date,
                        "holding_month": step.calendar_row.holding_month,
                        "permno": permnos,
                        "raw_forecast": forecasts,
                    }
                )
            )
            self.replayed_forecast_receipt_count += 1
            self.replayed_forecast_rows += receipt.rows
        result = pd.concat(rows, ignore_index=True).loc[:, list(FORECAST_COLUMNS)]
        self.replayed_month_count += 1
        if (
            self.replayed_month_count == 1
            or self.replayed_month_count % 5 == 0
            or self.replayed_month_count == len(self._plan.monthly_steps)
        ):
            self._replay_progress(
                "Stage 11 forecast replay month "
                f"{self.replayed_month_count}/{len(self._plan.monthly_steps)}: "
                f"{step.calendar_row.holding_month}"
            )
        self.gate()
        return result


def _archive_prior_failure_outcomes(
    execution_inputs: object,
    amendment: Mapping[str, Any],
) -> None:
    """Archive the bound failed-attempt tree once, then permit exact reuse."""

    prior = amendment["prior_failed_execution"]
    root = execution_inputs.output_roots["outcome_tables"]
    expected = prior["outcome_root"]
    archive = (
        root.parent
        / "outcome_archives"
        / (
            f"{expected['authority_build_id']}-"
            f"{expected['inventory_sha256'][:16]}"
        )
    )
    if not root.exists():
        recovery._require_tree_inventory(
            archive,
            expected,
            label="already archived prior failed-execution outcome tree",
        )
        return
    _require(not root.is_symlink(), "prior outcome root is unsafe")
    receipt_path = root / "outcome_materialization_receipt.json"
    payload = recovery._stable_file_bytes(
        receipt_path,
        label="prior failed-execution outcome receipt",
        expected_mode=0o644,
    )
    receipt = json.loads(payload)
    if receipt.get("authority_build_id") != expected["authority_build_id"]:
        live = require_live_stage11_validation_unblind_authority(
            execution_inputs._stage11_authority
        )
        _require(
            receipt.get("authority_build_id") == live.build_id
            and receipt.get("authority_verification_id")
            == live.verification_id,
            "outcome root is neither the prior failure nor current authority",
        )
        recovery._require_tree_inventory(
            archive,
            expected,
            label="archived prior failed-execution outcome tree",
        )
        return
    _require(
        receipt.get("authority_verification_id")
        == expected["authority_verification_id"]
        and hashlib.sha256(payload).hexdigest()
        == expected["outcome_materialization_receipt_sha256"],
        "prior failed-execution outcome authority differs",
    )
    recovery._require_tree_inventory(
        root,
        expected,
        label="prior failed-execution outcome tree",
    )
    recovery._archive_authenticated_tree(root, archive, expected)


def _authenticate_diagnostic_transition_archive(
    execution_inputs: object,
    amendment: Mapping[str, Any],
) -> Mapping[str, object]:
    """Reopen the whole incompatible diagnostic tree at its frozen archive."""

    transition = amendment["diagnostic_compatibility_transition"]
    relative = Path(transition["archive_relative_to_execution_results"])
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts
        and relative.parts[0] == "diagnostic_archives"
        and relative.as_posix()
        == transition["archive_relative_to_execution_results"],
        "diagnostic compatibility archive path is unsafe",
    )
    execution_root = execution_inputs.output_roots["execution_results"]
    archive = execution_root.joinpath(*relative.parts)
    active = execution_root / "diagnostics"
    _require(
        archive != active and not archive.is_symlink(),
        "diagnostic compatibility archive aliases the active root",
    )
    return recovery._require_tree_inventory(
        archive,
        transition["archive_tree"],
        label="archived incompatible Stage 11 diagnostic tree",
    )


def _closed_bundle_payloads(
    bundle: Path,
    *,
    members: tuple[str, ...],
    label: str,
) -> Mapping[str, bytes]:
    _require(
        bundle.is_dir()
        and not bundle.is_symlink()
        and set(path.name for path in bundle.iterdir()) == set(members),
        f"{label} is not a closed bundle",
    )
    return MappingProxyType(
        {
            name: recovery._stable_file_bytes(
                bundle / name,
                label=f"{label} member {name}",
            )
            for name in members
        }
    )


def _require_closed_primary_restart_root(root: Path) -> None:
    if not root.exists() and not root.is_symlink():
        return
    _require(
        root.is_dir() and not root.is_symlink(),
        "Stage 11 restart primary root is unsafe",
    )
    for bundle in sorted(root.iterdir(), key=lambda value: value.name):
        payloads = _closed_bundle_payloads(
            bundle,
            members=PRIMARY_MEMBERS,
            label="prior Stage 11 primary restart bundle",
        )
        try:
            manifest = json.loads(payloads[MANIFEST_NAME])
            success = json.loads(payloads[SUCCESS_NAME])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage11PortfolioResumeError(
                "prior Stage 11 primary restart JSON is invalid"
            ) from exc
        members = manifest.get("members")
        expected_payload_names = set(PRIMARY_MEMBERS) - {
            MANIFEST_NAME,
            SUCCESS_NAME,
        }
        _require(
            isinstance(members, Mapping)
            and set(members) == expected_payload_names
            and manifest.get("build_id") == bundle.name
            and success.get("build_id") == bundle.name
            and success.get("manifest_sha256")
            == hashlib.sha256(payloads[MANIFEST_NAME]).hexdigest()
            and success.get("closed_members") == list(PRIMARY_MEMBERS)
            and success.get("status") == manifest.get("status")
            and all(
                isinstance(members[name], Mapping)
                and members[name].get("sha256")
                == hashlib.sha256(payloads[name]).hexdigest()
                and members[name].get("size_bytes") == len(payloads[name])
                for name in expected_payload_names
            ),
            "prior Stage 11 primary restart bundle authentication differs",
        )


def _require_closed_verification_restart_root(root: Path) -> None:
    if not root.exists() and not root.is_symlink():
        return
    _require(
        root.is_dir() and not root.is_symlink(),
        "Stage 11 restart verification root is unsafe",
    )
    for bundle in sorted(root.iterdir(), key=lambda value: value.name):
        payloads = _closed_bundle_payloads(
            bundle,
            members=VERIFICATION_MEMBERS,
            label="prior Stage 11 verification restart bundle",
        )
        try:
            report = json.loads(payloads["verification_report.json"])
            differences = json.loads(payloads["differences.json"])
            success = json.loads(payloads[SUCCESS_NAME])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage11PortfolioResumeError(
                "prior Stage 11 verification restart JSON is invalid"
            ) from exc
        _require(
            report.get("verification_id") == bundle.name
            and success.get("verification_id") == bundle.name
            and success.get("report_sha256")
            == hashlib.sha256(payloads["verification_report.json"]).hexdigest()
            and success.get("differences_sha256")
            == hashlib.sha256(payloads["differences.json"]).hexdigest()
            and success.get("closed_members") == list(VERIFICATION_MEMBERS)
            and success.get("status") == report.get("status")
            and differences.get("difference_count")
            == report.get("difference_count")
            and differences.get("differences") == report.get("differences"),
            "prior Stage 11 verification restart bundle authentication differs",
        )


def _require_closed_report_restart_root(root: Path) -> None:
    if not root.exists() and not root.is_symlink():
        return
    _require(
        root.is_dir() and not root.is_symlink(),
        "Stage 11 restart report root is unsafe",
    )
    for bundle in sorted(root.iterdir(), key=lambda value: value.name):
        payloads = _closed_bundle_payloads(
            bundle,
            members=REPORT_MEMBERS,
            label="prior Stage 11 report restart bundle",
        )
        try:
            summary = json.loads(payloads[REPORT_SUMMARY_NAME])
            success = json.loads(payloads[REPORT_SUCCESS_NAME])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage11PortfolioResumeError(
                "prior Stage 11 report restart JSON is invalid"
            ) from exc
        _require(
            summary.get("report_id") == bundle.name
            and success.get("report_id") == bundle.name
            and success.get("summary_sha256")
            == hashlib.sha256(payloads[REPORT_SUMMARY_NAME]).hexdigest()
            and success.get("markdown_sha256")
            == hashlib.sha256(payloads[REPORT_MARKDOWN_NAME]).hexdigest()
            and success.get("closed_members") == list(REPORT_MEMBERS),
            "prior Stage 11 report restart bundle authentication differs",
        )


def _require_restartable_execution_outputs(execution_inputs: object) -> None:
    """Permit only absent/empty or exact closed idempotent restart state."""

    execution_root = execution_inputs.output_roots["execution_results"]
    pointer = (
        PROJECT_ROOT
        / "artifacts/phase_05_validation"
        / DEFAULT_LATEST_VERIFIED_POINTER_NAME
    )
    _require(
        not pointer.exists() and not pointer.is_symlink(),
        "Stage 11 is already verified; an automatic restart is prohibited",
    )
    _require_closed_primary_restart_root(execution_root / "primary")
    _require_closed_verification_restart_root(execution_root / "verification")
    _require_closed_report_restart_root(
        PROJECT_ROOT
        / "artifacts/phase_05_validation"
        / DEFAULT_REPORT_DIRECTORY
    )


def _require_named_attestation_checks(
    checks: tuple[tuple[str, bool], ...],
) -> None:
    """Fail with one stable non-performance check name."""

    names = tuple(name for name, _passed in checks)
    _require(
        bool(names)
        and len(names) == len(set(names))
        and all(
            isinstance(name, str)
            and bool(name)
            and name.replace("_", "").isalnum()
            for name in names
        ),
        "Stage 11 resume attestation check registry differs",
    )
    _require(
        all(isinstance(passed, (bool, np.bool_)) for _name, passed in checks),
        "Stage 11 resume attestation check result is not boolean",
    )
    for name, passed in checks:
        if not bool(passed):
            raise Stage11PortfolioResumeError(
                "Stage 11 portfolio-resume attestation check failed: "
                f"{name}"
            )


def _publish_resume_attestation(
    *,
    authority: object,
    execution_inputs: object,
    amendment: Mapping[str, Any],
    amendment_sha256: str,
    runtime_versions: Mapping[str, str | None],
    walk_forward: object,
    backend: AuthenticatedStage11ForecastReplayBackend,
    checkpoint_before: Mapping[str, object],
    checkpoint_after: Mapping[str, object],
    forecast_before: Mapping[str, object],
    forecast_after: Mapping[str, object],
    outcome_publication: object,
    portfolio: object,
    missing_audit: MissingReturnProjectionAudit | Mapping[str, object],
    recheck_audit: PortfolioRecheckReuseAudit | Mapping[str, object],
    access_ledger: Mapping[str, object],
    clean_candidate: AuthenticatedStage11CleanPortfolioCandidate,
) -> tuple[str, str]:
    live = require_live_stage11_validation_unblind_authority(authority)
    broker = dict(access_ledger)
    missing = (
        missing_audit.to_dict()
        if isinstance(missing_audit, MissingReturnProjectionAudit)
        else dict(missing_audit)
    )
    recheck = (
        recheck_audit.to_dict()
        if isinstance(recheck_audit, PortfolioRecheckReuseAudit)
        else dict(recheck_audit)
    )
    prior = amendment["prior_failed_execution"]
    projection = amendment["missing_return_projection"]
    continuity = tuple(portfolio.mark_continuity_evidence)
    no_mark_keys = {
        (
            value.strategy_id,
            value.holding_month,
            value.trading_date,
            int(permno),
        )
        for value in continuity
        for permno in value.no_mark_permnos
    }
    resolved_keys = {
        (
            value.strategy_id,
            value.holding_month,
            value.trading_date,
            int(permno),
        )
        for value in continuity
        for permno in value.resolved_pending_permnos
    }
    no_mark_key_payload = [
        {
            "strategy_id": strategy_id,
            "holding_month": holding_month,
            "trading_date": trading_date.isoformat(),
            "permno": permno,
        }
        for strategy_id, holding_month, trading_date, permno in sorted(
            no_mark_keys
        )
    ]
    no_mark_key_sha256 = hashlib.sha256(
        recovery._canonical_json(no_mark_key_payload)
    ).hexdigest()
    march_no_mark_strategies = {
        strategy_id
        for strategy_id, holding_month, trading_date, permno in no_mark_keys
        if holding_month == "2015-03"
        and trading_date == dt.date(2015, 3, 19)
        and permno == 90_423
    }
    march_resolved_strategies = {
        strategy_id
        for strategy_id, holding_month, trading_date, permno in resolved_keys
        if holding_month == "2015-03"
        and trading_date == dt.date(2015, 3, 20)
        and permno == 90_423
    }
    continuity_payload = {
        "session_evidence_rows": len(continuity),
        "no_mark_session_rows": sum(
            bool(value.no_mark_permnos) for value in continuity
        ),
        "no_mark_identity_rows": len(no_mark_keys),
        "no_mark_identity_key_sha256": no_mark_key_sha256,
        "resolved_pending_identity_rows": len(resolved_keys),
        "observed_numeric_zero_identity_rows": sum(
            len(value.observed_zero_return_permnos) for value in continuity
        ),
        "held_90423_no_mark_then_exact_bridge_authenticated": bool(
            march_no_mark_strategies.intersection(march_resolved_strategies)
        ),
        "pending_rebalance_trade_rows": 0,
        "pending_corporate_action_rows": 0,
        "disconnected_or_overlapping_span_applications": 0,
        "final_mark_continuity_ledger_sha256_by_strategy": dict(
            sorted(portfolio.final_mark_continuity_ledger_sha256.items())
        ),
        "final_pending_permnos_by_strategy": {
            strategy_id: list(values)
            for strategy_id, values in sorted(
                portfolio.final_pending_permnos.items()
            )
        },
        "final_pending_nonzero_count": portfolio.final_pending_nonzero_count,
    }
    retired_evidence = tuple(portfolio.retired_identity_evidence)
    retired_payload = [
        {
            "entry_date": value.entry_date.isoformat(),
            "holding_month": value.holding_month,
            "permno": value.permno,
            "prior_long_entry_allowed": value.prior_long_entry_allowed,
            "prior_short_entry_allowed": value.prior_short_entry_allowed,
            "retirement_date": value.retirement_date.isoformat(),
            "signal_date": value.signal_date.isoformat(),
        }
        for value in sorted(
            retired_evidence,
            key=lambda item: (
                item.entry_date,
                item.holding_month,
                item.permno,
            ),
        )
    ]
    retired_sha256 = _line_terminated_canonical_json_sha256(
        retired_payload
    )
    retired_execution_gate = {
        "policy": (
            "preserve_signal_scores_and_universe; block_new_long_and_short_"
            "entries_after_authenticated_terminal_identity_retirement"
        ),
        "evidence_rows": len(retired_payload),
        "evidence_sha256": retired_sha256,
        "unique_permnos": len(
            {int(value["permno"]) for value in retired_payload}
        ),
        "future_terminal_events_used": sum(
            value.retirement_date > value.entry_date
            for value in retired_evidence
        ),
        "unresolved_terminal_events_used": 0,
        "permno_89558_march_entry_block_authenticated": any(
            value.holding_month == "2015-03"
            and value.signal_date == dt.date(2015, 2, 27)
            and value.entry_date == dt.date(2015, 3, 2)
            and value.permno == 89_558
            and value.retirement_date == dt.date(2015, 3, 2)
            and value.prior_long_entry_allowed
            and not value.prior_short_entry_allowed
            for value in retired_evidence
        ),
    }
    pending_dust_evidence = tuple(
        portfolio.pending_dust_retirement_evidence
    )
    pending_dust_event_keys = tuple(
        (
            value.strategy_id,
            value.holding_month,
            value.boundary_date,
            value.permno,
        )
        for value in pending_dust_evidence
    )
    pending_dust_event_key_payload = [
        {
            "strategy_id": strategy_id,
            "holding_month": holding_month,
            "boundary_date": boundary_date.isoformat(),
            "permno": permno,
        }
        for strategy_id, holding_month, boundary_date, permno in (
            pending_dust_event_keys
        )
    ]
    pending_dust_payload = {
        "policy_id": PENDING_MATERIALITY_POLICY_ID,
        "strict_threshold_dollars_hex": (
            PENDING_MATERIALITY_THRESHOLD_DOLLARS.hex()
        ),
        "evidence_rows": len(pending_dust_evidence),
        "unique_permnos": len(
            {value.permno for value in pending_dust_evidence}
        ),
        "event_key_sha256": hashlib.sha256(
            recovery._canonical_json(pending_dust_event_key_payload)
        ).hexdigest(),
        "evidence_sha256": (
            portfolio.pending_dust_retirement_evidence_sha256
        ),
        "execution_price_reads": sum(
            value.execution_price_read for value in pending_dust_evidence
        ),
        "return_imputations": sum(
            value.return_imputed for value in pending_dust_evidence
        ),
        "trades_emitted": sum(
            value.trade_emitted for value in pending_dust_evidence
        ),
        "transaction_cost_imputations": sum(
            value.transaction_cost_imputed for value in pending_dust_evidence
        ),
        "corporate_actions_applied": sum(
            value.corporate_action_applied for value in pending_dust_evidence
        ),
        "terminal_retirements": sum(
            value.identity_terminally_retired
            for value in pending_dust_evidence
        ),
        "nav_changes": sum(
            value.opening_nav_dollars != value.closing_nav_dollars
            for value in pending_dust_evidence
        ),
    }
    payload = {
        "schema_version": 1,
        "attestation_id": "phase05-stage11-portfolio-phase-resume-v1",
        "status": "AUTHENTICATED_PORTFOLIO_PHASE_RESUME_READY_FOR_PUBLICATION",
        "authority": {
            "fresh_preflight_build_id": live.build_id,
            "fresh_preflight_verification_id": live.verification_id,
            "approval_contract_file_sha256": live.approval_contract_file_sha256,
            "approval_contract_payload_sha256": (
                live.approval_contract_payload_sha256
            ),
            "execution_input_manifest_sha256": live.execution_input_manifest_sha256,
        },
        "amendment": {
            "path": RESUME_AMENDMENT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": amendment_sha256,
            "prior_failure_receipt_path": prior["failure_receipt_path"],
            "prior_failure_receipt_sha256": prior["failure_receipt_sha256"],
            "prior_failure_seal_path": prior["failure_seal_path"],
            "prior_failure_seal_sha256": prior["failure_seal_sha256"],
            "prior_preflight_primary_pointer_sha256": prior[
                "preflight_primary_pointer_sha256"
            ],
            "prior_preflight_verified_pointer_sha256": prior[
                "preflight_verified_pointer_sha256"
            ],
            "preceding_held_march_failure_receipt_path": prior[
                "preceding_held_march_failure_receipt_path"
            ],
            "preceding_held_march_failure_receipt_sha256": prior[
                "preceding_held_march_failure_receipt_sha256"
            ],
            "preceding_march_failure_receipt_path": prior[
                "preceding_march_failure_receipt_path"
            ],
            "preceding_march_failure_receipt_sha256": prior[
                "preceding_march_failure_receipt_sha256"
            ],
            "originating_february_failure_receipt_path": prior[
                "originating_february_failure_receipt_path"
            ],
            "originating_february_failure_receipt_sha256": prior[
                "originating_february_failure_receipt_sha256"
            ],
        },
        "runtime_versions": dict(sorted(runtime_versions.items())),
        "clean_portfolio_candidate": {
            "candidate_sha256": clean_candidate.candidate_sha256,
            "manifest_sha256": clean_candidate.manifest_sha256,
            "bundle_relative_to_execution_results": (
                clean_candidate.bundle_path.relative_to(
                    execution_inputs.output_roots["execution_results"]
                ).as_posix()
            ),
            "independently_reopened": True,
            "reused_without_portfolio_recomputation": clean_candidate.reused,
            "historical_input_authority_relabelled": False,
            "portfolio_input_access_authority": {
                name: broker[name]
                for name in sorted(_ACCESS_LEDGER_AUTHORITY_FIELDS)
            },
        },
        "forecast_authority": {
            "prior_execution_completed_full_forecast_recomputation": True,
            "prior_exact_recomputed_cache_matches": 1_947,
            "prior_recomputed_forecast_rows": 4_013_460,
            "repeat_model_fitting_performed": False,
            "authenticated_checkpoint_references": (
                backend.replayed_fit_reference_count
            ),
            "authenticated_forecast_receipts_replayed": (
                backend.replayed_forecast_receipt_count
            ),
            "authenticated_forecast_rows_replayed": backend.replayed_forecast_rows,
            "monthly_score_count": walk_forward.monthly_score_count,
            "checkpoint_tree_before": dict(checkpoint_before),
            "checkpoint_tree_after": dict(checkpoint_after),
            "forecast_tree_before": dict(forecast_before),
            "forecast_tree_after": dict(forecast_after),
        },
        "missing_return_projection": missing,
        "mark_continuity": continuity_payload,
        "pending_dust_retirement": pending_dust_payload,
        "retired_identity_execution_gate": retired_execution_gate,
        "portfolio_recheck_reuse": recheck,
        "outcome_authority": {
            "build_id": live.build_id,
            "verification_id": live.verification_id,
            "materialized_validation_label_rows": 121_620,
            "table_receipts_match_frozen_contract": (
                outcome_publication.receipt.get("tables")
                == recovery._load_recovery_contract()["outcome_tables"]
            ),
            "validation_label_content_authentication_count": (
                backend.validation_label_content_authentication_count
            ),
        },
        "portfolio": {
            "portfolio_weight_rows": len(portfolio.portfolio_weights),
            "portfolio_trade_rows": len(portfolio.portfolio_trades),
            "daily_accounting_rows": len(portfolio.daily_accounting),
            "monthly_metric_rows": len(portfolio.monthly_metrics),
            "calibration_adapter_restored": (
                portfolio_backend_module.build_authenticated_development_calibration_observations
                is _CANONICAL_PORTFOLIO_DEVELOPMENT
            ),
            "selection_recheck_adapter_restored": (
                portfolio_backend_module.derive_locked_validation_selection
                is _CANONICAL_PORTFOLIO_SELECTION
            ),
            "calibration_schedule_recheck_adapter_restored": (
                portfolio_backend_module.build_strategy_calibration_schedules
                is _CANONICAL_PORTFOLIO_CALIBRATION
            ),
            "numerical_adapters_restored": (
                execution_orchestrator_module.solve_hybrid
                is recovery._CANONICAL_HYBRID_SOLVER
                and allocation_rank_module.solve_projection
                is recovery._CANONICAL_RANK_PROJECTION_SOLVER
            ),
            "missing_projection_adapter_restored": (
                portfolio_backend_module._daily_inputs is _CANONICAL_DAILY_INPUTS
            ),
            "corporate_action_projection_adapter_restored": (
                portfolio_backend_module.map_delisting_rows_to_corporate_actions
                is _CANONICAL_ACTION_MAPPER
            ),
        },
        "scope": {
            "first_holding_month": "2015-01",
            "last_holding_month": "2019-11",
            "holding_month_count": 59,
            "maximum_observation_or_exit_date_read": (
                broker["maximum_observation_or_exit_date_read"]
            ),
            "final_2020_plus_rows_read": broker["final_period_rows_read"],
            "stage12_or_later_rows_read": broker["stage12_or_later_rows_read"],
        },
    }
    expected_pending_dust_keys = {
        (
            strategy_id,
            expected["holding_month"],
            expected["boundary_date"],
            permno,
        )
        for permno, expected in _EXPECTED_PENDING_DUST_BY_PERMNO.items()
        for strategy_id in expected["strategy_ids"]
    }
    pending_dust_details_match = all(
        value.permno in _EXPECTED_PENDING_DUST_BY_PERMNO
        and value.holding_month
        == _EXPECTED_PENDING_DUST_BY_PERMNO[value.permno]["holding_month"]
        and value.boundary_date
        == _EXPECTED_PENDING_DUST_BY_PERMNO[value.permno]["boundary_date"]
        and value.last_authoritative_mark_date
        == _EXPECTED_PENDING_DUST_BY_PERMNO[value.permno][
            "last_authoritative_mark_date"
        ]
        and value.pending_no_mark_count
        == _EXPECTED_PENDING_DUST_BY_PERMNO[value.permno][
            "pending_no_mark_count"
        ]
        and value.pending_evidence_sha256
        == _EXPECTED_PENDING_DUST_BY_PERMNO[value.permno][
            "pending_evidence_sha256"
        ]
        for value in pending_dust_evidence
    )
    _require_named_attestation_checks(
        (
            (
                "forecast_checkpoint_reference_count",
                payload["forecast_authority"][
                    "authenticated_checkpoint_references"
                ]
                == EXPECTED_TOTAL_FITS,
            ),
            (
                "forecast_receipt_replay_count",
                payload["forecast_authority"][
                    "authenticated_forecast_receipts_replayed"
                ]
                == 1_947,
            ),
            (
                "forecast_row_replay_count",
                payload["forecast_authority"][
                    "authenticated_forecast_rows_replayed"
                ]
                == 4_013_460,
            ),
            (
                "monthly_score_count",
                payload["forecast_authority"]["monthly_score_count"] == 59,
            ),
            ("checkpoint_tree_unchanged", checkpoint_before == checkpoint_after),
            ("forecast_tree_unchanged", forecast_before == forecast_after),
            (
                "authoritative_null_to_zero_count",
                missing["authoritative_nulls_converted_to_zero"] == 0,
            ),
            (
                "exact_no_mark_source_row_count",
                missing["unique_exact_no_mark_source_rows"]
                == projection["expected_exact_no_mark_source_count"],
            ),
            (
                "exact_no_mark_distinct_record_count",
                missing["unique_exact_no_mark_source_record_sha256"]
                == projection["expected_exact_no_mark_source_count"],
            ),
            (
                "exact_no_mark_source_key_digest",
                missing["exact_no_mark_source_key_sha256"]
                == projection["expected_exact_no_mark_source_key_sha256"],
            ),
            (
                "mark_continuity_session_row_count",
                continuity_payload["session_evidence_rows"]
                == payload["portfolio"]["daily_accounting_rows"],
            ),
            (
                "mark_continuity_nonempty_no_mark_domain",
                continuity_payload["no_mark_identity_rows"] > 0,
            ),
            (
                "held_90423_exact_mark_bridge",
                continuity_payload[
                    "held_90423_no_mark_then_exact_bridge_authenticated"
                ]
                is True,
            ),
            (
                "pending_rebalance_trade_count",
                continuity_payload["pending_rebalance_trade_rows"] == 0,
            ),
            (
                "pending_corporate_action_count",
                continuity_payload["pending_corporate_action_rows"] == 0,
            ),
            (
                "disconnected_return_span_application_count",
                continuity_payload[
                    "disconnected_or_overlapping_span_applications"
                ]
                == 0,
            ),
            (
                "final_pending_nonzero_count",
                continuity_payload["final_pending_nonzero_count"] == 0,
            ),
            (
                "pending_dust_policy_id",
                pending_dust_payload["policy_id"]
                == PENDING_MATERIALITY_POLICY_ID,
            ),
            (
                "pending_dust_threshold",
                pending_dust_payload["strict_threshold_dollars_hex"]
                == PENDING_MATERIALITY_THRESHOLD_DOLLARS.hex(),
            ),
            (
                "pending_dust_evidence_row_count",
                pending_dust_payload["evidence_rows"]
                == _EXPECTED_PENDING_DUST_EVIDENCE_ROWS,
            ),
            (
                "pending_dust_unique_permno_count",
                pending_dust_payload["unique_permnos"]
                == len(_EXPECTED_PENDING_DUST_UNIQUE_PERMNOS),
            ),
            (
                "pending_dust_event_key_set",
                set(pending_dust_event_keys) == expected_pending_dust_keys,
            ),
            (
                "pending_dust_event_key_digest",
                pending_dust_payload["event_key_sha256"]
                == _EXPECTED_PENDING_DUST_EVENT_KEY_SHA256,
            ),
            ("pending_dust_event_details", pending_dust_details_match),
            (
                "pending_dust_evidence_digest",
                pending_dust_payload["evidence_sha256"]
                == _EXPECTED_PENDING_DUST_EVIDENCE_SHA256,
            ),
            (
                "pending_dust_execution_price_reads",
                pending_dust_payload["execution_price_reads"] == 0,
            ),
            (
                "pending_dust_return_imputations",
                pending_dust_payload["return_imputations"] == 0,
            ),
            (
                "pending_dust_trades_emitted",
                pending_dust_payload["trades_emitted"] == 0,
            ),
            (
                "pending_dust_transaction_cost_imputations",
                pending_dust_payload["transaction_cost_imputations"] == 0,
            ),
            (
                "pending_dust_corporate_actions_applied",
                pending_dust_payload["corporate_actions_applied"] == 0,
            ),
            (
                "pending_dust_terminal_retirements",
                pending_dust_payload["terminal_retirements"] == 0,
            ),
            (
                "pending_dust_nav_changes",
                pending_dust_payload["nav_changes"] == 0,
            ),
            (
                "retired_execution_evidence_row_count",
                retired_execution_gate["evidence_rows"]
                == _EXPECTED_RETIRED_EXECUTION_GATE_ROWS,
            ),
            (
                "retired_execution_unique_permno_count",
                retired_execution_gate["unique_permnos"]
                == _EXPECTED_RETIRED_EXECUTION_GATE_ROWS,
            ),
            (
                "retired_execution_evidence_digest",
                retired_execution_gate["evidence_sha256"]
                == _EXPECTED_RETIRED_EXECUTION_GATE_SHA256,
            ),
            (
                "retired_execution_future_event_count",
                retired_execution_gate["future_terminal_events_used"] == 0,
            ),
            (
                "retired_execution_unresolved_event_count",
                retired_execution_gate["unresolved_terminal_events_used"] == 0,
            ),
            (
                "retired_89558_march_entry_gate",
                retired_execution_gate[
                    "permno_89558_march_entry_block_authenticated"
                ]
                is True,
            ),
            (
                "final_pending_identity_sets_empty",
                all(
                    not values
                    for values in continuity_payload[
                        "final_pending_permnos_by_strategy"
                    ].values()
                ),
            ),
            (
                "outcome_table_receipts",
                payload["outcome_authority"][
                    "table_receipts_match_frozen_contract"
                ]
                is True,
            ),
            (
                "validation_label_content_authentication_count",
                payload["outcome_authority"][
                    "validation_label_content_authentication_count"
                ]
                >= 118,
            ),
            (
                "monthly_metric_rows_nonempty",
                payload["portfolio"]["monthly_metric_rows"] > 0,
            ),
            (
                "calibration_adapter_restored",
                payload["portfolio"]["calibration_adapter_restored"] is True,
            ),
            (
                "selection_recheck_adapter_restored",
                payload["portfolio"]["selection_recheck_adapter_restored"]
                is True,
            ),
            (
                "calibration_schedule_adapter_restored",
                payload["portfolio"][
                    "calibration_schedule_recheck_adapter_restored"
                ]
                is True,
            ),
            (
                "selection_reuse_call_count",
                payload["portfolio_recheck_reuse"][
                    "selection_reuse_calls"
                ]
                == 1,
            ),
            (
                "development_oof_reuse_call_count",
                payload["portfolio_recheck_reuse"][
                    "development_oof_reuse_calls"
                ]
                == 1,
            ),
            (
                "calibration_schedule_reuse_call_count",
                payload["portfolio_recheck_reuse"][
                    "calibration_schedule_reuse_calls"
                ]
                == 1,
            ),
            (
                "forecast_values_unchanged",
                payload["portfolio_recheck_reuse"]["forecast_values_changed"]
                is False,
            ),
            (
                "selection_values_unchanged",
                payload["portfolio_recheck_reuse"]["selection_values_changed"]
                is False,
            ),
            (
                "calibration_values_unchanged",
                payload["portfolio_recheck_reuse"][
                    "calibration_values_changed"
                ]
                is False,
            ),
            (
                "numerical_adapters_restored",
                payload["portfolio"]["numerical_adapters_restored"] is True,
            ),
            (
                "missing_projection_adapter_restored",
                payload["portfolio"]["missing_projection_adapter_restored"]
                is True,
            ),
            (
                "corporate_action_projection_adapter_restored",
                payload["portfolio"][
                    "corporate_action_projection_adapter_restored"
                ]
                is True,
            ),
            (
                "corporate_action_projection_call_count",
                missing["action_projection_calls"] == 59,
            ),
            (
                "unresolved_terminal_action_count",
                missing[
                    "unique_unresolved_terminal_actions_retained_as_audit_only"
                ]
                == 20,
            ),
            (
                "unresolved_terminal_return_twin_count",
                missing["unique_unresolved_terminal_return_twins"] == 20,
            ),
            (
                "resolved_corporate_action_count",
                missing["resolved_action_rows_mapped_canonically"] == 2_275,
            ),
            (
                "unresolved_terminal_source_row_count",
                missing["unresolved_terminal_source_row_count"] == 20,
            ),
            (
                "unresolved_terminal_key_digest",
                missing["unresolved_terminal_key_sha256"]
                == (
                    "3888ef6b7e50a3eed374aeb1d53af00df8172d0cf1b62"
                    "bff86efcdec92473bec"
                ),
            ),
            (
                "unresolved_terminal_settlement_count",
                missing["unresolved_terminal_actions_mapped_or_settled"] == 0,
            ),
            (
                "maximum_observation_or_exit_date",
                payload["scope"]["maximum_observation_or_exit_date_read"]
                == "2019-12-02",
            ),
            (
                "final_2020_plus_row_count",
                payload["scope"]["final_2020_plus_rows_read"] == 0,
            ),
            (
                "stage12_or_later_row_count",
                payload["scope"]["stage12_or_later_rows_read"] == 0,
            ),
            (
                "clean_candidate_independent_reopen",
                payload["clean_portfolio_candidate"][
                    "independently_reopened"
                ]
                is True,
            ),
        )
    )
    encoded = recovery._canonical_json(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    root = (
        PROJECT_ROOT
        / "artifacts/phase_05_validation/stage_11_recovery_attestations"
    )
    root.mkdir(parents=True, exist_ok=True)
    recovery._require_safe_directory(root, label="resume attestation root")
    destination = root / f"portfolio-resume-attestation-{digest}.json"
    if not destination.exists():
        temporary = root / f".{destination.name}.tmp-{os.getpid()}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
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
    reopened = recovery._stable_file_bytes(
        destination,
        label="resume attestation independent reopen",
        expected_mode=0o644,
    )
    _require(
        reopened == encoded
        and hashlib.sha256(reopened).hexdigest() == digest
        and json.loads(reopened) == payload,
        "Stage 11 resume attestation independent reopen differs",
    )
    return destination.relative_to(PROJECT_ROOT).as_posix(), digest


def _prepare_stage11_portfolio_inputs(
    authority: object,
    *,
    progress: Callable[[str], None],
) -> _PreparedStage11PortfolioInputs:
    live = require_live_stage11_validation_unblind_authority(authority)
    contract = recovery._load_recovery_contract()
    numerical_amendment, numerical_sha = recovery._load_numerical_amendment()
    decimal_amendment, decimal_sha = (
        recovery._load_decimal_serialization_amendment()
    )
    resume_amendment, resume_amendment_sha256 = _load_resume_amendment()
    prior_outcome_root = resume_amendment["prior_failed_execution"][
        "outcome_root"
    ]
    _require(
        live.build_id != prior_outcome_root["authority_build_id"]
        and live.verification_id
        != prior_outcome_root["authority_verification_id"],
        "Stage 11 resume authority must be a fresh verified preflight",
    )
    progress("Stage 11 resume: authenticating frozen inputs and prior failure")
    execution_inputs = recovery.authenticate_stage11_execution_inputs(authority)
    recovery._validate_authority(authority, execution_inputs, contract)
    _authenticate_diagnostic_transition_archive(
        execution_inputs,
        resume_amendment,
    )
    _require_restartable_execution_outputs(execution_inputs)
    roots = recovery._contract_roots(execution_inputs, contract)
    checkpoint_before = recovery._require_tree_inventory(
        roots["selector_checkpoints"],
        contract["trees"]["selector_checkpoints"],
        label="resume selector checkpoint cache",
    )
    forecast_before = recovery._require_tree_inventory(
        roots["raw_forecasts"],
        contract["trees"]["raw_forecasts"],
        label="resume raw forecast cache",
    )
    _archive_prior_failure_outcomes(execution_inputs, resume_amendment)
    outcome_publication = recovery._prepare_outcome_publication(
        authority,
        execution_inputs,
        contract,
        numerical_amendment,
        decimal_amendment,
        progress=progress,
    )
    calendar = AuthenticatedValidationCalendar(
        outcome_publication.calendar_rows,
        EXPECTED_REBALANCE_CALENDAR_SHA256,
    )
    plan = build_stage11_execution_plan(calendar)
    base_code_hashes = stage11_execution_implementation_code_hashes()
    runtime_versions = validate_runtime_versions()
    progress(
        "Stage 11 resume: replaying 1,947 authenticated forecast payloads "
        "without refitting"
    )
    identity_compatibility = decimal_amendment[
        "checkpoint_identity_compatibility"
    ]
    with recovery._cached_checkpoint_model_code_identity_scope(
        identity_compatibility["cached_model_code_sha256_by_family"]
    ):
        backend = AuthenticatedStage11ForecastReplayBackend(
            authority,
            execution_inputs,
            outcome_publication,
            plan,
            recovery_contract=contract,
            checkpoint_identity_compatibility=identity_compatibility,
            progress=progress,
        )
        walk_forward = run_empirical_locked_walk_forward(
            authority,
            execution_inputs.broker,
            plan,
            backend,
        )
    _require(
        selector_backend_module._model_code_sha256 is recovery._model_code_sha256,
        "selector model-code binding was not restored after replay",
    )
    _require(
        backend.replayed_fit_reference_count == 660
        and backend.replayed_forecast_receipt_count == 1_947
        and backend.replayed_forecast_rows == 4_013_460
        and backend.replayed_month_count == 59
        and walk_forward.ledger.complete,
        "authenticated forecast replay is incomplete",
    )
    checkpoint_after = recovery._require_tree_inventory(
        roots["selector_checkpoints"],
        contract["trees"]["selector_checkpoints"],
        label="resume selector checkpoint cache after replay",
    )
    forecast_after = recovery._require_tree_inventory(
        roots["raw_forecasts"],
        contract["trees"]["raw_forecasts"],
        label="resume raw forecast cache after replay",
    )
    selection = derive_empirical_locked_validation_selection(
        authority,
        execution_inputs.broker,
        walk_forward.ledger,
    )
    development_oof = recovery._build_split_filtered_development_calibration(
        execution_inputs,
        selection,
    )
    _require(
        set(development_oof)
        == {PRIMARY_STRATEGY_ID, ELASTIC_NET_CHALLENGER_ID},
        "resumed development calibration strategy set differs",
    )
    calibration = build_strategy_calibration_schedules(
        selection,
        walk_forward.ledger,
        development_oof=development_oof,
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
    return _PreparedStage11PortfolioInputs(
        live=live,
        contract=contract,
        numerical_amendment=numerical_amendment,
        numerical_amendment_sha256=numerical_sha,
        decimal_amendment=decimal_amendment,
        decimal_amendment_sha256=decimal_sha,
        resume_amendment=resume_amendment,
        resume_amendment_sha256=resume_amendment_sha256,
        execution_inputs=execution_inputs,
        roots=roots,
        checkpoint_before=checkpoint_before,
        checkpoint_after=checkpoint_after,
        forecast_before=forecast_before,
        forecast_after=forecast_after,
        outcome_publication=outcome_publication,
        calendar=calendar,
        plan=plan,
        base_code_hashes=base_code_hashes,
        runtime_versions=runtime_versions,
        backend=backend,
        walk_forward=walk_forward,
        selection=selection,
        development_oof=development_oof,
        calibration=calibration,
        selector_stage=selector_stage,
    )


def run_stage11_authenticated_diagnostics(
    authority: object,
    *,
    progress: Callable[[str], None] | None = None,
) -> Stage11DiagnosticExecution:
    """Run only performance-blind diagnostics and resume at the next month."""

    report = progress if progress is not None else (lambda _message: None)
    prepared = _prepare_stage11_portfolio_inputs(
        authority,
        progress=report,
    )
    compatibility = _stage11_diagnostic_compatibility(prepared)
    session = Stage11DiagnosticSession(
        root=_stage11_diagnostic_root(prepared),
        compatibility=compatibility,
        producer_implementation_sha256=(
            _stage11_diagnostic_producer_sha256(prepared)
        ),
    )
    report(
        "Stage 11 diagnostics: resuming from the latest authenticated "
        "post-month checkpoint"
    )
    with (
        _portfolio_recheck_reuse_scope(
            execution_inputs=prepared.execution_inputs,
            ledger=prepared.walk_forward.ledger,
            selection=prepared.selection,
            development_oof=prepared.development_oof,
            calibration_schedules=prepared.calibration,
        ),
        recovery._portfolio_allocation_numerical_override(),
        stage11_pending_materiality_scope(
            diagnostic_session=session,
        ) as pending_materiality_audit,
        _diagnostic_missing_return_projection_scope(),
    ):
        portfolio_status = (
            portfolio_backend_module.run_empirical_portfolio_diagnostics(
                execution_inputs=prepared.execution_inputs,
                outcome_publication=prepared.outcome_publication,
                calendar_authority=prepared.calendar,
                ledger=prepared.walk_forward.ledger,
                selection=prepared.selection,
                calibration_schedules=prepared.calibration,
                diagnostic_session=(
                    pending_materiality_audit.diagnostic_session
                ),
                progress=report,
            )
        )
    prepared.execution_inputs.gate()
    if portfolio_status.status != "diagnostic_path_complete":
        return Stage11DiagnosticExecution(
            portfolio_status=portfolio_status,
            completion=None,
            regression_suite_sha256=None,
        )
    regression_mapping = _diagnostic_regression_mapping(
        prepared=prepared,
        compatibility=compatibility,
    )
    regression_suite_sha256 = _run_and_record_diagnostic_regression_suite(
        prepared=prepared,
    )
    completion = session.seal_completion(
        regression_case_sha256_by_failure_id=regression_mapping,
        regression_suite_sha256=regression_suite_sha256,
        regression_suite_passed=True,
    )
    _publish_diagnostic_completion_attestation(
        compatibility=compatibility,
        completion=completion,
        regression_evidence_sha256=regression_suite_sha256,
    )
    prepared.execution_inputs.gate()
    return Stage11DiagnosticExecution(
        portfolio_status=portfolio_status,
        completion=completion,
        regression_suite_sha256=regression_suite_sha256,
    )


def run_stage11_authenticated_portfolio_resume(
    authority: object,
    *,
    progress: Callable[[str], None] | None = None,
) -> Stage11EmpiricalExecution:
    """Resume Stage 11 without repeating the already completed model work."""

    report = progress if progress is not None else (lambda _message: None)
    prepared = _prepare_stage11_portfolio_inputs(
        authority,
        progress=report,
    )
    resume_amendment = prepared.resume_amendment
    resume_amendment_sha256 = prepared.resume_amendment_sha256
    execution_inputs = prepared.execution_inputs
    checkpoint_before = prepared.checkpoint_before
    checkpoint_after = prepared.checkpoint_after
    forecast_before = prepared.forecast_before
    forecast_after = prepared.forecast_after
    outcome_publication = prepared.outcome_publication
    plan = prepared.plan
    base_code_hashes = prepared.base_code_hashes
    runtime_versions = prepared.runtime_versions
    backend = prepared.backend
    walk_forward = prepared.walk_forward
    selection = prepared.selection
    calibration = prepared.calibration
    selector_stage = prepared.selector_stage
    diagnostic_compatibility = _stage11_diagnostic_compatibility(prepared)
    diagnostic_completion = load_authenticated_diagnostic_completion_seal(
        root=_stage11_diagnostic_root(prepared),
        compatibility=diagnostic_compatibility,
    )
    current_diagnostic_producer_sha256 = (
        _stage11_diagnostic_producer_sha256(prepared)
    )
    current_regression_mapping = _diagnostic_regression_mapping(
        prepared=prepared,
        compatibility=diagnostic_compatibility,
    )
    regression_evidence_path = (
        _stage11_diagnostic_root(prepared)
        / "regression_evidence"
        / (
            "regression-suite-"
            f"{diagnostic_completion.regression_suite_sha256}.json"
        )
    )
    regression_evidence = recovery._stable_file_bytes(
        regression_evidence_path,
        label="Stage 11 diagnostic regression evidence before clean run",
        expected_mode=0o600,
    )
    _require_diagnostic_completion_for_clean_run(
        completion=diagnostic_completion,
        current_producer_implementation_sha256=(
            current_diagnostic_producer_sha256
        ),
        current_regression_mapping=current_regression_mapping,
        regression_evidence=regression_evidence,
    )
    (
        diagnostic_completion_attestation_path,
        diagnostic_completion_attestation_sha,
    ) = _publish_diagnostic_completion_attestation(
        compatibility=diagnostic_compatibility,
        completion=diagnostic_completion,
        regression_evidence_sha256=hashlib.sha256(
            regression_evidence
        ).hexdigest(),
    )
    (
        clean_candidate,
        clean_candidate_access_mode,
        attested_access_ledger,
    ) = _load_or_run_clean_portfolio_candidate(
        prepared=prepared,
        diagnostic_compatibility=diagnostic_compatibility,
        diagnostic_completion=diagnostic_completion,
        report=report,
    )
    current_access_ledger = execution_inputs.broker.ledger().to_dict()
    portfolio = clean_candidate.portfolio
    missing_audit = clean_candidate.missing_return_projection
    recheck_audit = clean_candidate.portfolio_recheck_reuse
    attestation_path, attestation_sha256 = _publish_resume_attestation(
        authority=authority,
        execution_inputs=execution_inputs,
        amendment=resume_amendment,
        amendment_sha256=resume_amendment_sha256,
        runtime_versions=runtime_versions,
        walk_forward=walk_forward,
        backend=backend,
        checkpoint_before=checkpoint_before,
        checkpoint_after=checkpoint_after,
        forecast_before=forecast_before,
        forecast_after=forecast_after,
        outcome_publication=outcome_publication,
        portfolio=portfolio,
        missing_audit=missing_audit,
        recheck_audit=recheck_audit,
        access_ledger=attested_access_ledger,
        clean_candidate=clean_candidate,
    )
    publication_candidate = load_authenticated_stage11_clean_candidate(
        root=_stage11_clean_candidate_root(prepared),
        compatibility=clean_candidate.compatibility,
    )
    _require(
        publication_candidate is not None
        and publication_candidate.candidate_sha256
        == clean_candidate.candidate_sha256
        and publication_candidate.manifest_sha256
        == clean_candidate.manifest_sha256
        and dict(publication_candidate.table_logical_sha256)
        == dict(clean_candidate.table_logical_sha256),
        "clean portfolio candidate changed before publication",
    )
    portfolio = publication_candidate.portfolio
    failure_path = resume_amendment["prior_failed_execution"][
        "failure_receipt_path"
    ]
    failure_sha = resume_amendment["prior_failed_execution"][
        "failure_receipt_sha256"
    ]
    failure_seal_path = resume_amendment["prior_failed_execution"][
        "failure_seal_path"
    ]
    failure_seal_sha = resume_amendment["prior_failed_execution"][
        "failure_seal_sha256"
    ]
    preceding_march_failure_path = resume_amendment[
        "prior_failed_execution"
    ]["preceding_march_failure_receipt_path"]
    preceding_march_failure_sha = resume_amendment[
        "prior_failed_execution"
    ]["preceding_march_failure_receipt_sha256"]
    preceding_held_march_failure_path = resume_amendment[
        "prior_failed_execution"
    ]["preceding_held_march_failure_receipt_path"]
    preceding_held_march_failure_sha = resume_amendment[
        "prior_failed_execution"
    ]["preceding_held_march_failure_receipt_sha256"]
    originating_failure_path = resume_amendment["prior_failed_execution"][
        "originating_february_failure_receipt_path"
    ]
    originating_failure_sha = resume_amendment["prior_failed_execution"][
        "originating_february_failure_receipt_sha256"
    ]
    clean_candidate_manifest_authority = (
        "private_clean_portfolio_candidate/"
        f"{clean_candidate.bundle_path.name}/manifest.json"
    )
    implementation_hashes = {
        **base_code_hashes,
        attestation_path: attestation_sha256,
        diagnostic_completion_attestation_path: (
            diagnostic_completion_attestation_sha
        ),
        failure_path: failure_sha,
        failure_seal_path: failure_seal_sha,
        preceding_held_march_failure_path: preceding_held_march_failure_sha,
        preceding_march_failure_path: preceding_march_failure_sha,
        originating_failure_path: originating_failure_sha,
        clean_candidate_manifest_authority: (
            clean_candidate.manifest_sha256
        ),
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
        access_ledger=attested_access_ledger,
    )
    _require(
        stage11_execution_implementation_code_hashes() == base_code_hashes,
        "Stage 11 implementation changed during portfolio resume",
    )
    output_root = execution_inputs.output_roots["execution_results"]
    if clean_candidate_access_mode == "fresh":
        publication = publish_stage11_execution(
            authority,
            execution_inputs.broker,
            artifacts,
            output_root=output_root / "primary",
            implementation_code_hashes=implementation_hashes,
        )
    else:
        require_live_stage11_validation_unblind_authority(authority)
        bound_access_ledger = _candidate_reuse_access_provenance(
            candidate_post_portfolio=attested_access_ledger,
            current_resume=current_access_ledger,
            portfolio=portfolio,
        )
        publication = publish_stage11_execution_core(
            replace(artifacts, access_ledger=bound_access_ledger),
            output_root=output_root / "primary",
            implementation_code_hashes=implementation_hashes,
            synthetic_fixture=False,
        )
        require_live_stage11_validation_unblind_authority(authority)
    primary_manifest = json.loads(
        recovery._stable_file_bytes(
            publication.bundle_path / MANIFEST_NAME,
            label="resumed Stage 11 primary manifest",
        )
    )
    _require(
        primary_manifest.get("implementation_code_hashes", {}).get(
            attestation_path
        )
        == attestation_sha256
        and primary_manifest.get("implementation_code_hashes", {}).get(
            diagnostic_completion_attestation_path
        )
        == diagnostic_completion_attestation_sha
        and primary_manifest.get("implementation_code_hashes", {}).get(
            failure_path
        )
        == failure_sha
        and primary_manifest.get("implementation_code_hashes", {}).get(
            failure_seal_path
        )
        == failure_seal_sha
        and primary_manifest.get("implementation_code_hashes", {}).get(
            preceding_held_march_failure_path
        )
        == preceding_held_march_failure_sha
        and primary_manifest.get("implementation_code_hashes", {}).get(
            preceding_march_failure_path
        )
        == preceding_march_failure_sha
        and primary_manifest.get("implementation_code_hashes", {}).get(
            originating_failure_path
        )
        == originating_failure_sha
        and primary_manifest.get("implementation_code_hashes", {}).get(
            clean_candidate_manifest_authority
        )
        == clean_candidate.manifest_sha256
        and all(
            primary_manifest.get("logical_table_sha256", {}).get(name)
            == digest
            for name, digest in clean_candidate.table_logical_sha256.items()
        ),
        "resumed primary manifest omitted its recovery authority",
    )
    verification = verify_stage11_execution(
        authority,
        publication,
        verification_root=output_root / "verification",
    )
    _require(
        verification.difference_count == 0,
        "resumed Stage 11 publication failed independent verification",
    )
    verified_report = publish_verified_stage11_execution_report(
        authority,
        publication,
        verification,
    )
    report_summary = json.loads(
        recovery._stable_file_bytes(
            verified_report.bundle_path / REPORT_SUMMARY_NAME,
            label="resumed verified Stage 11 report summary",
        )
    )
    _require(
        report_summary.get("authority", {}).get("primary_manifest_sha256")
        == publication.manifest_sha256,
        "resumed verified report does not bind the primary manifest",
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
    "AuthenticatedStage11ForecastReplayBackend",
    "Stage11DiagnosticExecution",
    "Stage11PortfolioResumeError",
    "run_stage11_authenticated_diagnostics",
    "run_stage11_authenticated_portfolio_resume",
]
