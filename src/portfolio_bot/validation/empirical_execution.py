"""Single authority-bound orchestration entrypoint for empirical Stage 11.

The module composes already reviewed primitives; it does not accept data paths,
receipts, model grids, or date ranges from a caller.  Keeping this composition
in one place makes the eventual unblind command auditable end to end.
"""

from __future__ import annotations

import hashlib
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from portfolio_bot.validation.calibration import CalibrationObservation
from portfolio_bot.validation.development_calibration import (
    build_authenticated_development_calibration_observations,
)
from portfolio_bot.validation.empirical_selector_backend import (
    EmpiricalStage11SelectorBackend,
)
from portfolio_bot.validation.empirical_portfolio_backend import (
    EmpiricalPortfolioRun,
    run_empirical_portfolio_accounting,
)
from portfolio_bot.validation.execution_input_manifest import (
    AuthenticatedStage11ExecutionInputs,
    authenticate_stage11_execution_inputs,
)
from portfolio_bot.validation.execution_orchestrator import (
    LockedValidationSelection,
    Stage11ExecutionPlan,
    StrategyCalibrationSchedule,
    WalkForwardRunReceipt,
    build_stage11_execution_plan,
    build_strategy_calibration_schedules,
    derive_empirical_locked_validation_selection,
    run_empirical_locked_walk_forward,
)
from portfolio_bot.validation.outcome_materialization import (
    Stage11OutcomePublication,
    publish_from_authenticated_execution_inputs,
)
from portfolio_bot.validation.execution_publication import (
    Stage11ExecutionArtifactSet,
    Stage11ExecutionPublication,
    Stage11ExecutionVerification,
    publish_stage11_execution,
    verify_stage11_execution,
)
from portfolio_bot.validation.execution_reporting import (
    Stage11VerifiedExecutionReportPublication,
    publish_verified_stage11_execution_report,
)
from portfolio_bot.validation.validation_runner import (
    require_live_stage11_validation_unblind_authority,
)
from portfolio_bot.validation.walk_forward import (
    EXPECTED_REBALANCE_CALENDAR_SHA256,
    AuthenticatedValidationCalendar,
)


class Stage11EmpiricalExecutionError(RuntimeError):
    """Raised when the sealed empirical composition differs from its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11EmpiricalExecutionError(message)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_EXPLICIT_EXECUTION_IMPLEMENTATION_PATHS = (
    "pyproject.toml",
    "configs/phase_04_models/allocation_build_rules_v1.json",
    "configs/phase_04_models/allocation_policy_v1.toml",
    "configs/phase_05_validation/stage_11_corporate_action_mapping_v1.json",
    "configs/phase_05_validation/stage_11_preflight_approval_v1.json",
    "configs/phase_05_validation/stage_11_preflight_policy_v1.toml",
    "configs/phase_05_validation/stage_11_preflight_rules_v1.json",
    "configs/phase_05_validation/stage_11_source_materialization_approval_v1.schema.json",
    "configs/phase_05_validation/stage_11_source_materialization_manifest_v1.schema.json",
    "configs/phase_05_validation/stage_11_source_materialization_plan_v1.json",
    "scripts/materialize_stage_11_validation_sources.py",
    "scripts/run_stage_11_validation.py",
    "scripts/verify_stage_11_validation_sources.py",
    "outputs/methodology_v1_freeze.json",
    "outputs/methodology_v1_source.md",
    "src/portfolio_bot/allocation/candidates.py",
    "src/portfolio_bot/allocation/constraints.py",
    "src/portfolio_bot/allocation/costs.py",
    "src/portfolio_bot/allocation/decision_focused.py",
    "src/portfolio_bot/allocation/domain.py",
    "src/portfolio_bot/allocation/rank.py",
    "src/portfolio_bot/allocation/risk.py",
    "src/portfolio_bot/allocation/solver.py",
    "src/portfolio_bot/data/phase_02_auxiliary_core.py",
    "src/portfolio_bot/data/phase_02_crsp_return_reconciliation.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_discovery.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_transforms.py",
    "src/portfolio_bot/data/secure_publication.py",
    "src/portfolio_bot/selectors/elastic_net.py",
    "src/portfolio_bot/selectors/contract.py",
    "src/portfolio_bot/selectors/inputs.py",
    "src/portfolio_bot/selectors/metrics.py",
    "src/portfolio_bot/selectors/model_specs.py",
    "src/portfolio_bot/selectors/native_process.py",
    "src/portfolio_bot/selectors/native_worker.py",
    "src/portfolio_bot/selectors/neural_model.py",
    "src/portfolio_bot/selectors/preprocessing.py",
    "src/portfolio_bot/selectors/weights.py",
    "src/portfolio_bot/selectors/xgboost_model.py",
    "src/portfolio_bot/features/label_math.py",
    "src/portfolio_bot/features/label_store_contract.py",
    "src/portfolio_bot/features/label_store_materialization.py",
    "src/portfolio_bot/risk_cost/contract.py",
    "src/portfolio_bot/risk_cost/cost_math.py",
    "src/portfolio_bot/risk_cost/risk_math.py",
    "src/portfolio_bot/validation/access.py",
    "src/portfolio_bot/validation/accounting.py",
    "src/portfolio_bot/validation/authorization.py",
    "src/portfolio_bot/validation/calibration.py",
    "src/portfolio_bot/validation/calendar.py",
    "src/portfolio_bot/validation/candidate_adapter.py",
    "src/portfolio_bot/validation/contract.py",
    "src/portfolio_bot/validation/corporate_action_adapter.py",
    "src/portfolio_bot/validation/corporate_actions.py",
    "src/portfolio_bot/validation/daily_engine.py",
    "src/portfolio_bot/validation/development_calibration.py",
    "src/portfolio_bot/validation/empirical_execution.py",
    "src/portfolio_bot/validation/empirical_portfolio_backend.py",
    "src/portfolio_bot/validation/empirical_selector_backend.py",
    "src/portfolio_bot/validation/execution_access.py",
    "src/portfolio_bot/validation/execution_input_manifest.py",
    "src/portfolio_bot/validation/execution_orchestrator.py",
    "src/portfolio_bot/validation/execution_publication.py",
    "src/portfolio_bot/validation/execution_risk_inputs.py",
    "src/portfolio_bot/validation/locked_selection.py",
    "src/portfolio_bot/validation/outcome_materialization.py",
    "src/portfolio_bot/validation/predictor_adapter.py",
    "src/portfolio_bot/validation/preflight.py",
    "src/portfolio_bot/validation/problem_builder.py",
    "src/portfolio_bot/validation/publication.py",
    "src/portfolio_bot/validation/selector_native_runtime.py",
    "src/portfolio_bot/validation/selector_runtime.py",
    "src/portfolio_bot/validation/session_authority.py",
    "src/portfolio_bot/validation/validation_feature_input_adapter.py",
    "src/portfolio_bot/validation/validation_market_extension.py",
    "src/portfolio_bot/validation/validation_return_adapter.py",
    "src/portfolio_bot/validation/validation_source_extraction.py",
    "src/portfolio_bot/validation/validation_source_materialization.py",
    "src/portfolio_bot/validation/validation_source_publication.py",
    "src/portfolio_bot/validation/validation_source_verification.py",
    "src/portfolio_bot/validation/validation_build_code_identity.py",
    "src/portfolio_bot/validation/validation_data_contract.py",
    "src/portfolio_bot/validation/validation_data_independent_verification.py",
    "src/portfolio_bot/validation/validation_runner.py",
    "src/portfolio_bot/validation/walk_forward.py",
)
_ALL_PORTFOLIO_BOT_PYTHON_PATHS = tuple(
    sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "src/portfolio_bot").rglob("*.py")
    )
)
_ALL_REVIEWED_STAGE_SCRIPT_PATHS = tuple(
    sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "scripts").glob("*.py")
    )
)
_ALL_REVIEWED_STAGE_CONFIG_PATHS = tuple(
    sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for root in (
            PROJECT_ROOT / "configs/phase_04_models",
            PROJECT_ROOT / "configs/phase_05_validation",
        )
        for path in root.iterdir()
        if path.is_file() and path.suffix in {".json", ".toml"}
    )
)
# Bind the conservative complete local Python surface.  This makes recursive
# import omissions impossible: adding or changing any package module changes
# the reviewed execution identity even before its import graph is revisited.
STAGE11_EXECUTION_IMPLEMENTATION_PATHS = tuple(
    dict.fromkeys(
        (
            *_EXPLICIT_EXECUTION_IMPLEMENTATION_PATHS,
            *_ALL_PORTFOLIO_BOT_PYTHON_PATHS,
            *_ALL_REVIEWED_STAGE_SCRIPT_PATHS,
            *_ALL_REVIEWED_STAGE_CONFIG_PATHS,
        )
    )
)


def stage11_execution_implementation_code_hashes() -> Mapping[str, str]:
    """Hash the exact reviewed implementation surface used by the run."""

    current_python_paths = tuple(
        sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (PROJECT_ROOT / "src/portfolio_bot").rglob("*.py")
        )
    )
    current_script_paths = tuple(
        sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (PROJECT_ROOT / "scripts").glob("*.py")
        )
    )
    current_config_paths = tuple(
        sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for root in (
                PROJECT_ROOT / "configs/phase_04_models",
                PROJECT_ROOT / "configs/phase_05_validation",
            )
            for path in root.iterdir()
            if path.is_file() and path.suffix in {".json", ".toml"}
        )
    )
    _require(
        current_python_paths == _ALL_PORTFOLIO_BOT_PYTHON_PATHS
        and current_script_paths == _ALL_REVIEWED_STAGE_SCRIPT_PATHS
        and current_config_paths == _ALL_REVIEWED_STAGE_CONFIG_PATHS,
        "Stage 11 reviewed code/config closure changed after process start",
    )
    _require(
        len(STAGE11_EXECUTION_IMPLEMENTATION_PATHS)
        == len(set(STAGE11_EXECUTION_IMPLEMENTATION_PATHS)),
        "Stage 11 execution code-binding paths are duplicated",
    )
    result: dict[str, str] = {}
    for relative in STAGE11_EXECUTION_IMPLEMENTATION_PATHS:
        relative_path = Path(relative)
        _require(
            not relative_path.is_absolute()
            and ".." not in relative_path.parts
            and relative_path.as_posix() == relative,
            f"Stage 11 implementation code path is unsafe: {relative}",
        )
        path = PROJECT_ROOT.joinpath(*relative_path.parts)
        metadata = path.stat(follow_symlinks=False)
        _require(
            path.resolve(strict=False) == path
            and stat.S_ISREG(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and metadata.st_nlink == 1,
            f"Stage 11 implementation code path is unsafe: {relative}",
        )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 << 20):
                digest.update(block)
        after = path.stat(follow_symlinks=False)
        _require(
            (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            f"Stage 11 implementation code changed while hashed: {relative}",
        )
        result[relative] = digest.hexdigest()
    return result


@dataclass(frozen=True, slots=True)
class Stage11EmpiricalSelectorStage:
    """Completed selector/selection/calibration portion of Stage 11."""

    execution_inputs: AuthenticatedStage11ExecutionInputs
    outcome_publication: Stage11OutcomePublication
    calendar_authority: AuthenticatedValidationCalendar
    plan: Stage11ExecutionPlan
    walk_forward: WalkForwardRunReceipt
    selection: LockedValidationSelection
    development_oof: Mapping[str, tuple[CalibrationObservation, ...]]
    calibration_schedules: Mapping[str, StrategyCalibrationSchedule]


def run_stage11_empirical_selector_stage(
    authority: object,
    *,
    progress: Callable[[str], None] | None = None,
) -> Stage11EmpiricalSelectorStage:
    """Run the exact 59-month selector chronology under one live authority."""

    report = progress if progress is not None else (lambda _message: None)
    require_live_stage11_validation_unblind_authority(authority)
    report("Stage 11: authenticating the frozen empirical input bundle")
    execution_inputs = authenticate_stage11_execution_inputs(authority)
    execution_inputs.gate()

    report("Stage 11: materializing authenticated outcome/accounting tables")
    outcome_publication = publish_from_authenticated_execution_inputs(
        authority, execution_inputs
    )
    execution_inputs.gate()
    calendar = AuthenticatedValidationCalendar(
        outcome_publication.calendar_rows,
        EXPECTED_REBALANCE_CALENDAR_SHA256,
    )
    plan = build_stage11_execution_plan(calendar)

    report("Stage 11: starting the locked 20-refit, 59-month walk-forward")
    backend = EmpiricalStage11SelectorBackend(
        authority,
        execution_inputs,
        outcome_publication,
        plan,
        progress=report,
    )
    walk_forward = run_empirical_locked_walk_forward(
        authority,
        execution_inputs.broker,
        plan,
        backend,
    )
    _require(
        walk_forward.ledger.complete,
        "Stage 11 selector chronology ended with an incomplete outcome ledger",
    )

    report("Stage 11: applying the frozen validation IC selection")
    selection = derive_empirical_locked_validation_selection(
        authority,
        execution_inputs.broker,
        walk_forward.ledger,
    )
    development_oof = build_authenticated_development_calibration_observations(
        execution_inputs, selection
    )
    calibration = build_strategy_calibration_schedules(
        selection,
        walk_forward.ledger,
        development_oof=development_oof,
    )
    execution_inputs.gate()
    return Stage11EmpiricalSelectorStage(
        execution_inputs=execution_inputs,
        outcome_publication=outcome_publication,
        calendar_authority=calendar,
        plan=plan,
        walk_forward=walk_forward,
        selection=selection,
        development_oof=development_oof,
        calibration_schedules=calibration,
    )


@dataclass(frozen=True, slots=True)
class Stage11EmpiricalExecution:
    """Completed and independently reopened Stage 11 empirical execution."""

    selector_stage: Stage11EmpiricalSelectorStage
    portfolio: EmpiricalPortfolioRun
    publication: Stage11ExecutionPublication
    verification: Stage11ExecutionVerification
    verified_report: Stage11VerifiedExecutionReportPublication


def run_stage11_empirical_execution(
    authority: object,
    *,
    progress: Callable[[str], None] | None = None,
) -> Stage11EmpiricalExecution:
    """Run, publish, and independently reopen the exact Stage 11 bundle."""

    report = progress if progress is not None else (lambda _message: None)
    require_live_stage11_validation_unblind_authority(authority)
    code_hashes = stage11_execution_implementation_code_hashes()
    selector_stage = run_stage11_empirical_selector_stage(
        authority,
        progress=report,
    )
    selector_stage.execution_inputs.gate()

    report("Stage 11: running independent portfolio states and net accounting")
    portfolio = run_empirical_portfolio_accounting(
        execution_inputs=selector_stage.execution_inputs,
        outcome_publication=selector_stage.outcome_publication,
        calendar_authority=selector_stage.calendar_authority,
        ledger=selector_stage.walk_forward.ledger,
        selection=selector_stage.selection,
        calibration_schedules=selector_stage.calibration_schedules,
        progress=report,
    )
    selector_stage.execution_inputs.gate()
    artifacts = Stage11ExecutionArtifactSet(
        plan=selector_stage.plan,
        ledger=selector_stage.walk_forward.ledger,
        selection=selector_stage.selection,
        calibration_schedules=selector_stage.calibration_schedules,
        execution_sessions=portfolio.execution_sessions,
        execution_session_receipts=portfolio.execution_session_receipts,
        portfolio_weights=portfolio.portfolio_weights,
        portfolio_trades=portfolio.portfolio_trades,
        daily_accounting=portfolio.daily_accounting,
        monthly_metrics=portfolio.monthly_metrics,
        access_ledger=selector_stage.execution_inputs.broker.ledger().to_dict(),
    )
    _require(
        stage11_execution_implementation_code_hashes() == code_hashes,
        "Stage 11 implementation code changed during empirical execution",
    )
    output_root = selector_stage.execution_inputs.output_roots["execution_results"]
    report("Stage 11: publishing the immutable execution bundle")
    publication = publish_stage11_execution(
        authority,
        selector_stage.execution_inputs.broker,
        artifacts,
        output_root=output_root / "primary",
        implementation_code_hashes=code_hashes,
    )
    report("Stage 11: independently reopening every published member")
    verification = verify_stage11_execution(
        authority,
        publication,
        verification_root=output_root / "verification",
    )
    _require(
        verification.difference_count == 0,
        "Stage 11 execution publication failed independent verification",
    )
    _require(
        stage11_execution_implementation_code_hashes() == code_hashes,
        "Stage 11 implementation code changed during publication verification",
    )
    report("Stage 11: publishing the canonical verified results report")
    verified_report = publish_verified_stage11_execution_report(
        authority,
        publication,
        verification,
    )
    _require(
        stage11_execution_implementation_code_hashes() == code_hashes,
        "Stage 11 implementation code changed during results publication",
    )
    selector_stage.execution_inputs.gate()
    return Stage11EmpiricalExecution(
        selector_stage=selector_stage,
        portfolio=portfolio,
        publication=publication,
        verification=verification,
        verified_report=verified_report,
    )


__all__ = [
    "STAGE11_EXECUTION_IMPLEMENTATION_PATHS",
    "Stage11EmpiricalExecution",
    "Stage11EmpiricalExecutionError",
    "Stage11EmpiricalSelectorStage",
    "run_stage11_empirical_execution",
    "run_stage11_empirical_selector_stage",
    "stage11_execution_implementation_code_hashes",
]
