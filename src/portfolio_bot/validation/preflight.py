"""Deterministic, no-outcome integration preflight for roadmap Stage 11.

The runner authenticates the frozen authority graph, reads only the bounded
rebalance calendar and small development metadata through the access broker,
and exercises the portfolio path with fictitious securities.  It never opens
validation labels or returns, forms an empirical portfolio, or computes a
performance statistic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa

from portfolio_bot.allocation.evidence import build_synthetic_allocator_evidence
from portfolio_bot.features.feature_store_contract import (
    expanded_features,
    raw_feature_schema,
)
from portfolio_bot.validation.access import (
    PayloadRole,
    PreflightAccessBroker,
    RegisteredPayload,
    Stage11AccessError,
)
from portfolio_bot.validation.accounting import PortfolioState
from portfolio_bot.validation.authorization import (
    ApprovalScope,
    AuthorizationError,
    issue_approval_receipt,
    require_validation_outcome_unblind_authorization,
)
from portfolio_bot.validation.calendar import (
    FIRST_VALIDATION_ENTRY_DATE,
    FIRST_VALIDATION_SIGNAL_DATE,
    LAST_VALIDATION_EXIT_DATE,
    LAST_VALIDATION_SIGNAL_DATE,
    ValidationCalendarError,
    ValidationCalendarRow,
    validate_validation_calendar,
    validation_holding_month_descriptor,
)
from portfolio_bot.validation.calibration import (
    CalibrationObservation,
    fit_nonnegative_huber_calibration,
)
from portfolio_bot.validation.contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
    REQUIRED_PREFLIGHT_CHECK_NAMES,
    Stage11PreflightContractReceipt,
    authenticate_stage11_preflight_contract,
    canonical_json_bytes,
)
from portfolio_bot.validation.corporate_actions import (
    CASH_SETTLEMENT,
    CorporateAction,
)
from portfolio_bot.validation.daily_engine import (
    DatedMarketCloseInput,
    run_dated_holding_period,
)
from portfolio_bot.validation.problem_builder import (
    AuthenticatedMonthlyIdentity,
    InheritedHoldings,
    MarketSnapshot,
    MonthlyRiskInputs,
    ScoredSnapshot,
    build_monthly_problem,
)
from portfolio_bot.validation.session_authority import (
    EXPECTED_TRADING_CALENDAR_PATH,
    EXPECTED_TRADING_CALENDAR_SHA256,
    EXPECTED_TRADING_CALENDAR_SIZE_BYTES,
    authenticate_holding_period_session_authority,
)
from portfolio_bot.validation.publication import (
    PreflightCheck,
    PreflightEvidenceDigest,
    PreflightRunSummary,
    PrimaryPreflightPublication,
    VerifiedPreflightPublication,
    publish_independent_verification,
    publish_primary_preflight,
)
from portfolio_bot.validation.walk_forward import (
    AuthenticatedValidationCalendar,
    ForecastCommit,
    OutcomeCommit,
    WalkForwardError,
    WalkForwardLedger,
)


RUNNER_ID = "stage11-no-outcome-preflight-v1"
INDEPENDENT_RUNNER_ID = "stage11-no-outcome-preflight-independent-v1"
DATABASE_ROOT = Path("/Users/agraemirolmez/Desktop/Thesis")
CALENDAR_PATH = (
    DATABASE_ROOT
    / "interim/phase_02_canonical_panel/phase02-crsp-b03263a970e52fbd"
    / "calendar/calendar-ae2e4ba4a280d66e/rebalance_calendar.parquet"
)
CALENDAR_SHA256 = (
    "cc03002bb9c349c07f14d699bf3667fd4ccbcd012f20579b1ce1c22dd19f2591"
)
CALENDAR_SIZE_BYTES = 6_648

STAGE6_POINTER_PATH = (
    PROJECT_ROOT / "artifacts/phase_03_features/LATEST_FEATURE_STORE_VERIFIED.json"
)
STAGE6_POINTER_SHA256 = (
    "29421df4f8d5becd5f68af566dacad1b6683c57f98083c03cfa624649b591308"
)
STAGE6_POINTER_SIZE_BYTES = 715
STAGE6_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts/phase_03_features/feature_store_builds"
    / "phase03-features-fda21f0e51b8276c10246f9f/manifest.json"
)
STAGE6_MANIFEST_SHA256 = (
    "96ab135c0628054699fa3f27a76abd1cb30f3e3e46279a84052a29c895b51375"
)
STAGE6_MANIFEST_SIZE_BYTES = 16_517
STAGE6_REGISTRY_PATH = (
    PROJECT_ROOT / "configs/phase_03_features/feature_registry_v1.json"
)
STAGE6_REGISTRY_SHA256 = (
    "7bac81cd8083c9a5c44ab2482223a342ed6ce2ce265d73cd8cc1bd843e33805a"
)
STAGE6_REGISTRY_SIZE_BYTES = 34_546
STAGE6_RAW_SCHEMA_SHA256 = (
    "e67436adbe6e2adb2f445a799f07002a272a5cdf780662aca639f0eb3c0969c3"
)


class Stage11PreflightRunError(RuntimeError):
    """Raised when any required preflight observation does not pass."""


@dataclass(frozen=True, slots=True)
class Stage11PreflightExecution:
    summary: PreflightRunSummary
    checks: tuple[PreflightCheck, ...]
    evidence: tuple[PreflightEvidenceDigest, ...]
    access_broker: PreflightAccessBroker
    contract_receipt: Stage11PreflightContractReceipt


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11PreflightRunError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise Stage11PreflightRunError(f"{label} must be a JSON object")
    return value


def _physical_parquet_schema(schema: pa.Schema) -> pa.Schema:
    fields: list[pa.Field] = []
    for field in schema:
        field_type = field.type
        if pa.types.is_list(field_type):
            field_type = pa.list_(
                pa.field(
                    "element",
                    field_type.value_type,
                    nullable=field_type.value_field.nullable,
                )
            )
        fields.append(
            pa.field(
                field.name,
                field_type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        )
    return pa.schema(fields, metadata=schema.metadata)


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _records() -> dict[str, RegisteredPayload]:
    return {
        "validation_calendar": RegisteredPayload(
            role=PayloadRole.VALIDATION_CALENDAR_METADATA,
            path=CALENDAR_PATH,
            sha256=CALENDAR_SHA256,
            size_bytes=CALENDAR_SIZE_BYTES,
            allowed_columns=(
                "signal_date",
                "holding_month",
                "entry_date",
                "exit_date",
            ),
            minimum_signal_date=FIRST_VALIDATION_SIGNAL_DATE,
            maximum_signal_date=LAST_VALIDATION_SIGNAL_DATE,
        ),
        "validation_trading_sessions": RegisteredPayload(
            role=PayloadRole.VALIDATION_TRADING_SESSION_METADATA,
            path=EXPECTED_TRADING_CALENDAR_PATH,
            sha256=EXPECTED_TRADING_CALENDAR_SHA256,
            size_bytes=EXPECTED_TRADING_CALENDAR_SIZE_BYTES,
            allowed_columns=("calendar_date",),
            minimum_signal_date=FIRST_VALIDATION_ENTRY_DATE,
            maximum_signal_date=LAST_VALIDATION_EXIT_DATE,
        ),
        "stage6_verified_pointer": RegisteredPayload(
            role=PayloadRole.DEVELOPMENT_PARITY_FIXTURE,
            path=STAGE6_POINTER_PATH,
            sha256=STAGE6_POINTER_SHA256,
            size_bytes=STAGE6_POINTER_SIZE_BYTES,
        ),
        "stage6_primary_manifest": RegisteredPayload(
            role=PayloadRole.DEVELOPMENT_PARITY_FIXTURE,
            path=STAGE6_MANIFEST_PATH,
            sha256=STAGE6_MANIFEST_SHA256,
            size_bytes=STAGE6_MANIFEST_SIZE_BYTES,
        ),
        "stage6_feature_registry": RegisteredPayload(
            role=PayloadRole.DEVELOPMENT_PARITY_FIXTURE,
            path=STAGE6_REGISTRY_PATH,
            sha256=STAGE6_REGISTRY_SHA256,
            size_bytes=STAGE6_REGISTRY_SIZE_BYTES,
        ),
    }


def _read_validation_calendar(
    broker: PreflightAccessBroker,
) -> tuple[ValidationCalendarRow, ...]:
    request = broker.authorize_validation_table_request(
        "validation_calendar",
        role=PayloadRole.VALIDATION_CALENDAR_METADATA,
        columns=("signal_date", "holding_month", "entry_date", "exit_date"),
        minimum_signal_date=FIRST_VALIDATION_SIGNAL_DATE,
        maximum_signal_date=LAST_VALIDATION_SIGNAL_DATE,
    )
    table = broker.read_guarded_parquet(request)
    rows = tuple(
        ValidationCalendarRow.from_mapping(row) for row in table.to_pylist()
    )
    return validate_validation_calendar(rows)


def _development_schema_reconciliation(
    broker: PreflightAccessBroker,
) -> dict[str, object]:
    pointer = _json(
        broker.read_registered_bytes(
            "stage6_verified_pointer",
            role=PayloadRole.DEVELOPMENT_PARITY_FIXTURE,
        ),
        label="Stage 6 verified pointer",
    )
    manifest = _json(
        broker.read_registered_bytes(
            "stage6_primary_manifest",
            role=PayloadRole.DEVELOPMENT_PARITY_FIXTURE,
        ),
        label="Stage 6 primary manifest",
    )
    registry = _json(
        broker.read_registered_bytes(
            "stage6_feature_registry",
            role=PayloadRole.DEVELOPMENT_PARITY_FIXTURE,
        ),
        label="Stage 6 feature registry",
    )
    if (
        pointer.get("status") != "VERIFIED_PASSED"
        or pointer.get("build_id")
        != "phase03-features-fda21f0e51b8276c10246f9f"
        or pointer.get("verification_id") != "verify-350915782d3ef12e"
        or pointer.get("primary_manifest_sha256") != STAGE6_MANIFEST_SHA256
    ):
        raise Stage11PreflightRunError("Stage 6 verified authority differs")
    raw_receipt = (
        manifest.get("outputs", {})
        .get("interim", {})
        .get("files", {})
        .get("feature_store_raw.parquet", {})
    )
    summary = manifest.get("summary")
    features = expanded_features(registry)
    schema = _physical_parquet_schema(raw_feature_schema(features))
    observed_schema_sha = _schema_sha256(schema)
    if (
        not isinstance(summary, Mapping)
        or summary.get("raw_rows") != 1_263_048
        or summary.get("economic_features") != 163
        or not isinstance(raw_receipt, Mapping)
        or raw_receipt.get("schema_sha256") != STAGE6_RAW_SCHEMA_SHA256
        or observed_schema_sha != STAGE6_RAW_SCHEMA_SHA256
        or len(schema) != 170
    ):
        raise Stage11PreflightRunError(
            "development feature schema reconciliation differs"
        )
    return {
        "stage6_build_id": str(pointer["build_id"]),
        "stage6_verification_id": str(pointer["verification_id"]),
        "raw_rows": int(summary["raw_rows"]),
        "economic_features": int(summary["economic_features"]),
        "ordered_columns": len(schema),
        "schema_sha256": observed_schema_sha,
        "registry_sha256": STAGE6_REGISTRY_SHA256,
        "reconciliation_kind": "exact_schema_and_transform_authority",
    }


def _chronology_and_calibration_evidence(
    calendar: tuple[ValidationCalendarRow, ...],
) -> dict[str, object]:
    calendar_authority = AuthenticatedValidationCalendar(
        rows=calendar,
        source_payload_sha256=CALENDAR_SHA256,
    )
    november = ForecastCommit(
        forecast_id="synthetic-november-2014",
        signal_date=date(2014, 10, 31),
        entry_date=date(2014, 11, 3),
        exit_date=date(2014, 12, 1),
        payload_sha256="a" * 64,
    )
    december = ForecastCommit(
        forecast_id="synthetic-december-2014",
        signal_date=date(2014, 11, 28),
        entry_date=date(2014, 12, 1),
        exit_date=date(2015, 1, 2),
        payload_sha256="b" * 64,
    )
    november_result = OutcomeCommit(
        forecast_id=november.forecast_id,
        exit_date=november.exit_date,
        availability_date=november.exit_date,
        outcome_value=0.01,
    )
    premature_rejected = False
    try:
        WalkForwardLedger().commit_outcome(november_result)
    except WalkForwardError:
        premature_rejected = True
    if not premature_rejected:
        raise Stage11PreflightRunError("premature result commitment was accepted")
    ledger = (
        WalkForwardLedger()
        .commit_forecast(november)
        .commit_forecast(december)
        .commit_outcome(november_result)
    )
    selected = ledger.training_labels_for_refit(
        date(2014, 12, 31),
        calendar_authority=calendar_authority,
    )
    if (
        tuple(item.forecast_id for item in selected.included)
        != (november.forecast_id,)
        or tuple(item.forecast_id for item in selected.purged)
        != (december.forecast_id,)
    ):
        raise Stage11PreflightRunError("walk-forward purge membership differs")

    observations: list[CalibrationObservation] = []
    for month, signal, entry, exit_date, offset in (
        ("2014-10", date(2014, 9, 30), date(2014, 10, 1), date(2014, 11, 3), 0),
        ("2014-11", date(2014, 10, 31), date(2014, 11, 3), date(2014, 12, 1), 10),
    ):
        for side, score in enumerate((-0.5, 0.5), start=1):
            observations.append(
                CalibrationObservation(
                    forecast_id=f"synthetic-calibration-{month}-{side}",
                    holding_month=month,
                    permno=8_000_000 + offset + side,
                    score=score,
                    residual_return=0.05 * score,
                    signal_date=signal,
                    entry_date=entry,
                    exit_date=exit_date,
                    forecast_payload_sha256="c" * 64,
                    forecast_committed_at=datetime.combine(
                        signal, time(20), tzinfo=timezone.utc
                    ),
                    outcome_available_at=datetime.combine(
                        exit_date, time(20), tzinfo=timezone.utc
                    ),
                    source="development_oof",
                )
            )
    calibration = fit_nonnegative_huber_calibration(
        observations,
        fit_signal_date=date(2014, 12, 31),
    )
    if not np.isclose(calibration.kappa, 0.05, rtol=1e-5, atol=1e-10):
        raise Stage11PreflightRunError("synthetic calibration slope differs")
    return {
        "premature_commit_rejected": premature_rejected,
        "included_forecast_ids": [item.forecast_id for item in selected.included],
        "purged_forecast_ids": [item.forecast_id for item in selected.purged],
        "unavailable_forecast_ids": list(selected.unavailable_forecast_ids),
        "calibration_observations": calibration.observations,
        "calibration_months": calibration.distinct_months,
        "calibration_kappa_decimal": format(calibration.kappa, ".12g"),
    }


def _problem_builder_evidence() -> dict[str, object]:
    identity = AuthenticatedMonthlyIdentity(
        signal_date=date(2014, 12, 31),
        source_bundle_sha256="d" * 64,
    )
    permno = np.arange(1, 9, dtype=np.int64)
    score = np.linspace(-0.4375, 0.4375, 8, dtype=np.float64)
    scored = ScoredSnapshot(
        identity=identity,
        permno=permno,
        score=score,
        calibration_kappa_monthly=0.04,
        long_eligible=np.ones(8, dtype=np.bool_),
        short_eligible=np.ones(8, dtype=np.bool_),
    )
    market_permno = np.append(permno, 99)
    market = MarketSnapshot(
        identity=identity,
        permno=market_permno,
        permco=9_000 + market_permno,
        execution_price=np.full(9, 20.0),
        full_spread=np.full(9, 0.002),
        adv20_usd=np.full(9, 20_000_000.0),
        impact_volatility_daily=np.full(9, 0.02),
        factor_betas=np.zeros((9, 6)),
        specific_variance_monthly=np.full(9, 0.04),
        sector=np.full(9, "SYNTHETIC_SECTOR"),
        country=np.full(9, "USA"),
        new_long_trade_allowed=np.ones(9, dtype=np.bool_),
        new_short_trade_allowed=np.ones(9, dtype=np.bool_),
    )
    inherited = InheritedHoldings(
        identity=identity,
        permno=np.asarray([99]),
        permco=np.asarray([9_099]),
        previous_weight=np.asarray([0.012]),
        last_factor_betas=np.asarray([[0.8, 0.1, 0.0, 0.0, 0.0, -0.1]]),
        last_specific_variance_monthly=np.asarray([0.03]),
        risk_age_sessions=np.asarray([10]),
        sector=np.asarray(["SYNTHETIC_SECTOR"]),
        country=np.asarray(["USA"]),
        current_long_eligible=np.asarray([False]),
        current_short_eligible=np.asarray([False]),
        required_exit=np.asarray([False]),
    )
    covariance = np.diag(
        np.asarray([0.0025, 0.0004, 0.0003, 0.0002, 0.0002, 0.0005])
    )
    risk = MonthlyRiskInputs(
        identity=identity,
        factor_covariance_monthly=covariance,
        specific_variance_p95_monthly=0.09,
        required_exit_impact_volatility_p95_daily=0.07,
    )
    result = build_monthly_problem(
        scored,
        market,
        inherited,
        risk,
        nav_dollars=100_000_000.0,
        minimum_candidates=1,
        tail_fraction=0.25,
    )
    holding_index = int(np.flatnonzero(result.problem.permno == 99)[0])
    finite_alpha = np.isfinite(result.problem.score)
    if (
        len(result.universe.permno) != 9
        or result.problem.n_assets != 5
        or not result.problem.required_exit[holding_index]
        or not result.problem.reduction_only[holding_index]
        or not np.allclose(
            result.problem.expected_return[finite_alpha],
            0.04 * result.problem.score[finite_alpha],
            atol=1e-15,
            rtol=0.0,
        )
    ):
        raise Stage11PreflightRunError("monthly problem adapter evidence differs")
    return {
        "union_rows": int(len(result.universe.permno)),
        "optimization_rows": int(result.problem.n_assets),
        "candidate_status": result.selection.status,
        "required_exit_rows": int(np.count_nonzero(result.problem.required_exit)),
        "reduction_only_rows": int(np.count_nonzero(result.problem.reduction_only)),
        "alpha_is_calibrated_score": True,
        "identity_sha256": identity.source_bundle_sha256,
    }


def _daily_engine_evidence(
    calendar_row: ValidationCalendarRow,
    *,
    calendar_authority: AuthenticatedValidationCalendar,
    broker: PreflightAccessBroker,
) -> dict[str, object]:
    request = broker.authorize_validation_table_request(
        "validation_trading_sessions",
        role=PayloadRole.VALIDATION_TRADING_SESSION_METADATA,
        columns=("calendar_date",),
        minimum_signal_date=calendar_row.entry_date,
        maximum_signal_date=calendar_row.exit_date,
    )
    session_authority = authenticate_holding_period_session_authority(
        calendar_authority=calendar_authority,
        calendar_row=calendar_row,
        access_broker=broker,
        source_request=request,
    )
    first = session_authority.return_dates[0]
    action = CorporateAction(
        action_id="synthetic-cash-action",
        economic_event_date=first,
        assigned_return_date=first,
        public_availability_date=first,
        action_sequence=1,
        source_record_sha256="e" * 64,
        source_permno=101,
        settlement_kind=CASH_SETTLEMENT,
    )
    action_result = run_dated_holding_period(
        PortfolioState.all_cash(1_000.0),
        session_authority=session_authority,
        target_permno=np.asarray([101]),
        target_weights=np.asarray([0.10]),
        execution_price=np.asarray([10.0]),
        full_spread=np.asarray([0.0]),
        adv_dollars=np.asarray([1e20]),
        impact_volatility_daily=np.asarray([0.0]),
        daily_inputs=tuple(
            DatedMarketCloseInput(
                trading_date=trading_date,
                permno=(
                    np.asarray([101])
                    if index == 0
                    else np.empty(0, dtype=np.int64)
                ),
                total_return=(
                    np.asarray([0.20]) if index == 0 else np.empty(0)
                ),
                borrow_rate_annual=(
                    np.asarray([0.0]) if index == 0 else np.empty(0)
                ),
                daily_risk_free_rate=0.01 if index == 1 else 0.0,
                source_payload_sha256=("f" if index == 0 else "1") * 64,
            )
            for index, trading_date in enumerate(session_authority.return_dates)
        ),
        corporate_actions=(action,),
        financing_rate_annual=0.0,
    )
    if (
        action_result.state.permno.size != 0
        or not action_result.session_receipts[0].corporate_action_ids
        or action_result.session_receipts[0].post_return_state_sha256
        == action_result.session_receipts[0].closing_state_sha256
    ):
        raise Stage11PreflightRunError("corporate-action chronology evidence differs")

    accounting_result = run_dated_holding_period(
        PortfolioState.all_cash(1_000.0),
        session_authority=session_authority,
        target_permno=np.asarray([101, 202]),
        target_weights=np.asarray([0.20, -0.10]),
        execution_price=np.asarray([10.0, 10.0]),
        full_spread=np.asarray([0.0, 0.0]),
        adv_dollars=np.asarray([1e20, 1e20]),
        impact_volatility_daily=np.asarray([0.0, 0.0]),
        daily_inputs=tuple(
            DatedMarketCloseInput(
                trading_date=trading_date,
                permno=np.asarray([101, 202]),
                total_return=(
                    np.asarray([0.10, -0.05])
                    if index == 0
                    else np.asarray([0.0, 0.0])
                ),
                borrow_rate_annual=(
                    np.asarray([0.0, 0.252])
                    if index == 0
                    else np.asarray([0.0, 0.0])
                ),
                daily_risk_free_rate=0.001 if index == 0 else 0.0,
                source_payload_sha256=("2" if index == 0 else "3") * 64,
            )
            for index, trading_date in enumerate(session_authority.return_dates)
        ),
        financing_rate_annual=0.0,
    )
    # Independent arithmetic: 1 bp commission on 30% one-way turnover; the
    # short is charged 25.2% divided by 252 on opening notional.
    expected_transaction = 1_000.0 * 0.30 * 0.0001
    expected_borrow = 100.0 * 0.252 / 252.0
    expected_cash = (1_000.0 - 200.0 - (-100.0) - expected_transaction) * 1.001
    expected_cash -= expected_borrow
    expected_nav = 200.0 * 1.10 + (-100.0) * 0.95 + expected_cash
    if (
        not np.isclose(
            accounting_result.transaction_cost_dollars,
            expected_transaction,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            accounting_result.borrow_cost_dollars,
            expected_borrow,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            accounting_result.state.nav_dollars,
            expected_nav,
            rtol=0.0,
            atol=1e-10,
        )
        or accounting_result.session_authority_sha256
        != session_authority.authority_sha256
        or tuple(
            receipt.trading_date for receipt in accounting_result.session_receipts
        )
        != session_authority.return_dates
    ):
        raise Stage11PreflightRunError("independent daily accounting parity differs")
    return {
        "action_sessions": len(action_result.session_receipts),
        "action_receipts": len(action_result.corporate_action_receipts),
        "source_retired": action_result.state.permno.size == 0,
        "accounting_sessions": len(accounting_result.session_receipts),
        "transaction_cost_cents": int(round(expected_transaction * 100.0)),
        "borrow_cost_microusd": int(round(expected_borrow * 1_000_000.0)),
        "terminal_position_rows": int(accounting_result.state.permno.size),
        "session_authority_sha256": session_authority.authority_sha256,
        "entry_close_excluded": session_authority.return_dates[0]
        > calendar_row.entry_date,
        "terminal_exit_included": session_authority.return_dates[-1]
        == calendar_row.exit_date,
        "session_sequence_exact": True,
        "artificial_terminal_liquidation": False,
        "independent_parity": True,
    }


def _negative_boundary_evidence(
    contract: Stage11PreflightContractReceipt,
) -> dict[str, object]:
    prohibited_role_rejected = False
    reader_calls = 0

    def reader(_path: Path) -> bytes:
        nonlocal reader_calls
        reader_calls += 1
        return b""

    broker = PreflightAccessBroker(byte_reader=reader)
    try:
        broker.read_registered_bytes(
            "sealed",
            role=PayloadRole.VALIDATION_OUTCOME,
        )
    except Stage11AccessError:
        prohibited_role_rejected = True

    hash_substitution_rejected = False
    substituted = PreflightAccessBroker(
        {
            "altered": RegisteredPayload(
                role=PayloadRole.REPOSITORY_METADATA,
                path=STAGE6_POINTER_PATH,
                sha256="0" * 64,
                size_bytes=STAGE6_POINTER_SIZE_BYTES,
            )
        }
    )
    try:
        substituted.read_registered_bytes(
            "altered", role=PayloadRole.REPOSITORY_METADATA
        )
    except Stage11AccessError:
        hash_substitution_rejected = True

    preflight_scope_rejected = False
    preflight_receipt = issue_approval_receipt(
        receipt_id="synthetic-preflight-scope-check",
        scope=ApprovalScope.IMPLEMENTATION_NO_OUTCOME_PREFLIGHT,
        approved=True,
        approved_by="synthetic-preflight",
        approved_at_utc="2026-07-22T00:00:00Z",
        source_message_id="synthetic-preflight-message",
        source_message="Synthetic no-outcome authorization check.",
        stage11_policy_sha256=contract.policy_sha256,
    )
    try:
        require_validation_outcome_unblind_authorization(
            preflight_receipt,
            expected_stage11_policy_sha256=contract.policy_sha256,
        )
    except AuthorizationError:
        preflight_scope_rejected = True
    if not (
        prohibited_role_rejected
        and reader_calls == 0
        and hash_substitution_rejected
        and preflight_scope_rejected
    ):
        raise Stage11PreflightRunError("negative authorization boundary differs")
    return {
        "prohibited_role_rejected_before_read": prohibited_role_rejected,
        "prohibited_reader_calls": reader_calls,
        "hash_substitution_rejected": hash_substitution_rejected,
        "preflight_scope_rejected_for_unblind": preflight_scope_rejected,
    }


def _expect_raises(
    exception: type[BaseException],
    operation: Callable[[], object],
) -> bool:
    try:
        operation()
    except exception:
        return True
    return False


def _check(
    name: str,
    *,
    observed: bool | int | str | None,
    expected: bool | int | str | None,
    detail: str,
    evidence_value: object,
    observation_count: int,
) -> tuple[PreflightCheck, PreflightEvidenceDigest]:
    if name not in REQUIRED_PREFLIGHT_CHECK_NAMES:
        raise Stage11PreflightRunError(f"unknown required check: {name}")
    if observed != expected:
        raise Stage11PreflightRunError(f"required preflight check failed: {name}")
    return (
        PreflightCheck(
            check_name=name,
            status="PASS",
            observed_value=observed,
            expected_value=expected,
            detail=detail,
        ),
        PreflightEvidenceDigest(
            check_name=name,
            evidence_sha256=_digest(evidence_value),
            observation_count=observation_count,
        ),
    )


def run_stage11_no_outcome_preflight(
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> Stage11PreflightExecution:
    """Execute all sixteen gates and return only checks and evidence digests."""

    broker = PreflightAccessBroker(_records())
    contract = authenticate_stage11_preflight_contract(policy_path)
    calendar = _read_validation_calendar(broker)
    development = _development_schema_reconciliation(broker)
    chronology = _chronology_and_calibration_evidence(calendar)
    problem = _problem_builder_evidence()
    calendar_authority = AuthenticatedValidationCalendar(
        rows=calendar,
        source_payload_sha256=CALENDAR_SHA256,
    )
    daily = _daily_engine_evidence(
        calendar[0],
        calendar_authority=calendar_authority,
        broker=broker,
    )
    allocation = build_synthetic_allocator_evidence()
    negative = _negative_boundary_evidence(contract)

    calendar_rejections = {
        value: _expect_raises(
            ValidationCalendarError,
            lambda value=value: validation_holding_month_descriptor(value),
        )
        for value in ("2019-12", "2020-01")
    }
    if not all(calendar_rejections.values()):
        raise Stage11PreflightRunError("embargo or final calendar month was accepted")
    ledger = broker.ledger()
    forbidden_count = sum(
        (
            ledger.validation_predictor_forward_return_or_label_columns_read,
            ledger.validation_outcome_rows_read,
            ledger.final_period_rows_read,
        )
    )
    real_output_count = sum(
        (
            ledger.real_validation_weights_created,
            ledger.real_validation_trades_created,
            ledger.real_validation_pnl_or_performance_rows_created,
        )
    )
    allocator_scope = allocation["scope"]
    allocator_gates = allocation["gates"]
    if not isinstance(allocator_scope, Mapping) or not isinstance(
        allocator_gates, Mapping
    ):
        raise Stage11PreflightRunError("synthetic allocator evidence is malformed")

    items = [
        _check(
            "methodology_and_stage10_authority_authenticated",
            observed=contract.status,
            expected="AUTHENTICATED_NO_OUTCOME_PREFLIGHT_ONLY",
            detail="Frozen methodology and verified Stage 10 authority authenticated.",
            evidence_value=contract.to_dict(),
            observation_count=10,
        ),
        _check(
            "preflight_approval_authenticated_and_narrow",
            observed=contract.approval_scope,
            expected="stage11_implementation_no_outcome_preflight",
            detail="Approval is limited to implementation and no-outcome preflight.",
            evidence_value={
                "approval_id": contract.approval_id,
                "approval_scope": contract.approval_scope,
                "approval_payload_sha256": contract.approval_payload_sha256,
            },
            observation_count=1,
        ),
        _check(
            "validation_and_final_outcome_access_zero",
            observed=forbidden_count,
            expected=0,
            detail="Broker records zero validation-result, forward-label, and final-period rows.",
            evidence_value=ledger.__dict__ if hasattr(ledger, "__dict__") else {
                "validation_outcome_rows_read": ledger.validation_outcome_rows_read,
                "final_period_rows_read": ledger.final_period_rows_read,
                "forward_label_columns_read": ledger.validation_predictor_forward_return_or_label_columns_read,
            },
            observation_count=3,
        ),
        _check(
            "real_weights_trades_and_pnl_zero",
            observed=real_output_count,
            expected=0,
            detail="No empirical allocation, trade, or result row was created.",
            evidence_value={
                "real_weights": ledger.real_validation_weights_created,
                "real_trades": ledger.real_validation_trades_created,
                "real_result_rows": ledger.real_validation_pnl_or_performance_rows_created,
            },
            observation_count=3,
        ),
        _check(
            "exact_59_month_calendar_contract",
            observed=len(calendar),
            expected=59,
            detail="Authenticated calendar has the exact January 2015 through November 2019 domain.",
            evidence_value={
                "rows": len(calendar),
                "first": calendar[0].holding_month,
                "last": calendar[-1].holding_month,
                "calendar_sha256": CALENDAR_SHA256,
            },
            observation_count=len(calendar),
        ),
        _check(
            "december_2019_and_2020_plus_rejected",
            observed=sum(calendar_rejections.values()),
            expected=2,
            detail="Embargoed December 2019 and the first final-period month fail closed.",
            evidence_value=calendar_rejections,
            observation_count=2,
        ),
        _check(
            "forecast_precedes_outcome",
            observed=chronology["premature_commit_rejected"],
            expected=True,
            detail="A result cannot be committed before its hash-bound forecast.",
            evidence_value=chronology,
            observation_count=2,
        ),
        _check(
            "only_completed_labels_enter_refit_or_calibration",
            observed=len(chronology["included_forecast_ids"]),
            expected=1,
            detail="The completed November label is included and the overlapping December interval is purged.",
            evidence_value=chronology,
            observation_count=int(chronology["calibration_observations"]),
        ),
        _check(
            "development_slice_builder_parity_or_explicit_reconciliation",
            observed=development["ordered_columns"],
            expected=170,
            detail="Corrected Stage 6 transform authority reconciles to the exact frozen ordered schema.",
            evidence_value=development,
            observation_count=3,
        ),
        _check(
            "dynamic_candidate_and_inherited_holding_path_synthetic_pass",
            observed=problem["union_rows"],
            expected=9,
            detail="Fictitious scored names and inherited holdings form one dynamic monthly problem.",
            evidence_value=problem,
            observation_count=int(problem["union_rows"]),
        ),
        _check(
            "required_exit_and_corporate_action_path_synthetic_pass",
            observed=bool(problem["required_exit_rows"] and daily["source_retired"]),
            expected=True,
            detail="Screen failure forces exit and a dated post-return action retires its source identity.",
            evidence_value={"problem": problem, "daily": daily},
            observation_count=2,
        ),
        _check(
            "daily_fixed_share_accounting_independent_parity_pass",
            observed=daily["independent_parity"],
            expected=True,
            detail="Independent signed-value arithmetic matches the fixed-share daily engine.",
            evidence_value=daily,
            observation_count=int(daily["accounting_sessions"]),
        ),
        _check(
            "transaction_borrow_and_financing_debit_parity_pass",
            observed=allocator_gates[
                "transaction_borrow_financing_backend_parity_pass"
            ],
            expected=True,
            detail="NumPy, convex, and differentiable cost backends agree on fictitious inputs.",
            evidence_value=allocation["costs"],
            observation_count=3,
        ),
        _check(
            "ordinary_projection_and_emergency_rules_pass",
            observed=allocator_gates[
                "ordinary_all_cash_required_exit_and_emergency_flow_pass"
            ],
            expected=True,
            detail="Ordinary, all-cash, required-exit, and emergency solver paths pass synthetically.",
            evidence_value=allocation["solver_flow"],
            observation_count=9,
        ),
        _check(
            "decision_focused_empirical_boundary_fail_closed",
            observed=allocator_scope[
                "decision_focused_empirical_production_training_ready"
            ],
            expected=False,
            detail="Empirical decision-focused training remains explicitly inactive.",
            evidence_value={
                "ready": allocator_scope[
                    "decision_focused_empirical_production_training_ready"
                ],
                "performed": allocator_scope[
                    "decision_focused_real_data_training_performed"
                ],
            },
            observation_count=2,
        ),
        _check(
            "negative_authorization_and_hash_substitution_tests_pass",
            observed=all(bool(value) for key, value in negative.items() if key != "prohibited_reader_calls")
            and negative["prohibited_reader_calls"] == 0,
            expected=True,
            detail="Wrong scope, prohibited role, and altered hash all fail closed.",
            evidence_value=negative,
            observation_count=4,
        ),
    ]
    checks = tuple(item[0] for item in items)
    evidence = tuple(item[1] for item in items)
    if {item.check_name for item in checks} != set(REQUIRED_PREFLIGHT_CHECK_NAMES):
        raise Stage11PreflightRunError("preflight check membership differs")
    return Stage11PreflightExecution(
        summary=PreflightRunSummary(
            runner_id=RUNNER_ID,
            implementation_mode="synthetic_and_development_parity_no_outcome",
            required_check_count=len(REQUIRED_PREFLIGHT_CHECK_NAMES),
        ),
        checks=checks,
        evidence=evidence,
        access_broker=broker,
        contract_receipt=contract,
    )


def build_and_publish_stage11_preflight(
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> PrimaryPreflightPublication:
    execution = run_stage11_no_outcome_preflight(policy_path=policy_path)
    return publish_primary_preflight(
        summary=execution.summary,
        checks=execution.checks,
        evidence=execution.evidence,
        access_broker=execution.access_broker,
        policy_path=policy_path,
    )


def independently_verify_stage11_preflight(
    *,
    policy_path: Path = DEFAULT_POLICY,
    primary_pointer_path: Path | None = None,
) -> VerifiedPreflightPublication:
    execution = run_stage11_no_outcome_preflight(policy_path=policy_path)
    selected = primary_pointer_path or (
        PROJECT_ROOT
        / "artifacts/phase_05_validation/LATEST_STAGE_11_PREFLIGHT_PRIMARY.json"
    )
    independent_summary = PreflightRunSummary(
        runner_id=INDEPENDENT_RUNNER_ID,
        implementation_mode=execution.summary.implementation_mode,
        required_check_count=execution.summary.required_check_count,
    )
    return publish_independent_verification(
        primary_pointer_path=selected,
        summary=independent_summary,
        checks=execution.checks,
        evidence=execution.evidence,
        access_broker=execution.access_broker,
        policy_path=policy_path,
    )


__all__ = [
    "INDEPENDENT_RUNNER_ID",
    "RUNNER_ID",
    "Stage11PreflightExecution",
    "Stage11PreflightRunError",
    "build_and_publish_stage11_preflight",
    "independently_verify_stage11_preflight",
    "run_stage11_no_outcome_preflight",
]
