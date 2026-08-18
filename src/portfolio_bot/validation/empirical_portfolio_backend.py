"""Authority-bound portfolio and fixed-share accounting for Stage 11."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.risk_cost.contract import FACTOR_NAMES, PRESENT
from portfolio_bot.risk_cost.risk_math import type7_quantile
from portfolio_bot.validation.accounting import PortfolioState
from portfolio_bot.validation.calendar import VALIDATION_HOLDING_MONTHS
from portfolio_bot.validation.corporate_action_adapter import (
    map_delisting_rows_to_corporate_actions,
)
from portfolio_bot.validation.corporate_actions import CorporateActionLedger
from portfolio_bot.validation.corporate_actions import portfolio_state_sha256
from portfolio_bot.validation.daily_engine import (
    DatedMarketCloseInput,
    MarkContinuityLedger,
)
from portfolio_bot.validation.development_calibration import (
    build_authenticated_development_calibration_observations,
)
from portfolio_bot.validation.execution_input_manifest import (
    AuthenticatedStage11ExecutionInputs,
)
from portfolio_bot.validation.execution_orchestrator import (
    DECISION_FOCUSED_CHALLENGER_ID,
    ELASTIC_NET_CHALLENGER_ID,
    PRIMARY_STRATEGY_ID,
    RANK_CHALLENGER_ID,
    LockedValidationSelection,
    Stage11PseudoLiveLedger,
    StrategyAccountingInput,
    StrategyCalibrationSchedule,
    allocate_locked_monthly_strategies,
    build_strategy_calibration_schedules,
    diagnose_locked_monthly_accounting,
    diagnose_locked_monthly_strategies,
    derive_locked_validation_selection,
    run_locked_monthly_accounting,
)
from portfolio_bot.validation.outcome_materialization import (
    CONTRACT_ID as OUTCOME_CONTRACT_ID,
    STATUS_COMPLETE as OUTCOME_STATUS_COMPLETE,
    Stage11OutcomePublication,
)
from portfolio_bot.validation.problem_builder import (
    AuthenticatedMonthlyIdentity,
    InheritedHoldings,
    MarketSnapshot,
    MonthlyProblemBuild,
    MonthlyRiskInputs,
    ScoredSnapshot,
    build_monthly_problem,
)
from portfolio_bot.validation.session_authority import (
    authenticate_materialized_holding_period_session_authority,
)
from portfolio_bot.validation.stage11_diagnostics import (
    ELASTIC_NET_CHALLENGER_ID as DIAGNOSTIC_ELASTIC_NET_CHALLENGER_ID,
    PRIMARY_STRATEGY_ID as DIAGNOSTIC_PRIMARY_STRATEGY_ID,
    RANK_CHALLENGER_ID as DIAGNOSTIC_RANK_CHALLENGER_ID,
    STAGE11_DIAGNOSTIC_STRATEGY_IDS,
    DiagnosticFailureStatus,
    Stage11AggregateFailureReport,
    Stage11DiagnosticSession,
    StrategyDiagnosticResult,
    encode_stage11_strategy_problem,
)
from portfolio_bot.validation.walk_forward import AuthenticatedValidationCalendar


INITIAL_NAV_DOLLARS = 100_000_000.0
PENDING_DUST_RETIREMENT_POLICY_ID = (
    "stage11-pending-subcent-dust-retirement-v1"
)
PENDING_DUST_RETIREMENT_THRESHOLD_DOLLARS = 0.01
_RISK_NAMES = (
    "validation_risk_cost_audit",
    "validation_risk_cost_inputs",
    "validation_factor_covariance",
    "validation_risk_cost_month_summary",
)
_ACCOUNTING_BASE_NAMES = (
    "entry_closes",
    "daily_total_returns",
    "daily_factors",
    "daily_sessions",
    "delisting_inputs",
)
_OUTCOME_TABLE_NAMES = (
    "validation_labels.parquet",
    "validation_label_audit.parquet",
    "validation_label_leg_lineage.parquet",
    "validation_label_month_summary.parquet",
    *(
        name
        for base in _ACCOUNTING_BASE_NAMES
        for name in (f"{base}.parquet", f"outcome_reserve_{base}.parquet")
    ),
)


class EmpiricalPortfolioBackendError(RuntimeError):
    """Raised when portfolio inputs, state, or accounting are not exact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EmpiricalPortfolioBackendError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


@dataclass(frozen=True, slots=True)
class _BoundOutcomeTable:
    path: Path
    sha256: str
    identity: tuple[int, ...]


class AuthenticatedOutcomeAccountingStore:
    """Receipt-bound reader for the ten derived accounting shards."""

    def __init__(
        self,
        execution_inputs: AuthenticatedStage11ExecutionInputs,
        publication: Stage11OutcomePublication,
    ) -> None:
        _require(
            type(execution_inputs) is AuthenticatedStage11ExecutionInputs,
            "accounting store requires authenticated execution inputs",
        )
        execution_inputs.gate()
        _require(
            type(publication) is Stage11OutcomePublication
            and publication.output_root
            == execution_inputs.output_roots["outcome_tables"],
            "accounting store publication differs from execution inputs",
        )
        final_receipt_path = (
            publication.output_root / "outcome_materialization_receipt.json"
        )
        final_before = final_receipt_path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(final_before.st_mode)
            and not stat.S_ISLNK(final_before.st_mode)
            and final_before.st_nlink == 1,
            "outcome materialization receipt is unsafe",
        )
        try:
            final_receipt = json.loads(final_receipt_path.read_text("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmpiricalPortfolioBackendError(
                "outcome materialization receipt is invalid"
            ) from exc
        final_after = final_receipt_path.stat(follow_symlinks=False)
        _require(
            _stable_identity(final_before) == _stable_identity(final_after)
            and isinstance(final_receipt, Mapping)
            and dict(final_receipt) == dict(publication.receipt)
            and final_receipt.get("status") == OUTCOME_STATUS_COMPLETE
            and final_receipt.get("contract_id") == OUTCOME_CONTRACT_ID
            and final_receipt.get("build_id") == publication.build_id,
            "outcome publication is not the exact receipted materialization",
        )
        receipts = publication.receipt.get("tables")
        _require(
            isinstance(receipts, Mapping)
            and set(receipts) == set(_OUTCOME_TABLE_NAMES),
            "outcome materialization table receipt set differs",
        )
        bound: dict[str, _BoundOutcomeTable] = {}
        for base in _ACCOUNTING_BASE_NAMES:
            for name in (f"{base}.parquet", f"outcome_reserve_{base}.parquet"):
                receipt = receipts.get(name)
                _require(
                    isinstance(receipt, Mapping) and receipt.get("path") == name,
                    f"accounting table receipt differs: {name}",
                )
                path = publication.output_root / name
                metadata = path.stat(follow_symlinks=False)
                _require(
                    stat.S_ISREG(metadata.st_mode)
                    and not stat.S_ISLNK(metadata.st_mode)
                    and metadata.st_nlink == 1
                    and metadata.st_size == receipt.get("size_bytes")
                    and _sha256_file(path) == receipt.get("sha256"),
                    f"accounting table is unsafe or changed: {name}",
                )
                parquet = pq.ParquetFile(path)
                schema_sha = hashlib.sha256(
                    parquet.schema_arrow.serialize().to_pybytes()
                ).hexdigest()
                _require(
                    parquet.metadata.num_rows == receipt.get("rows")
                    and schema_sha == receipt.get("schema_sha256"),
                    f"accounting table footer differs: {name}",
                )
                bound[name] = _BoundOutcomeTable(
                    path=path,
                    sha256=str(receipt["sha256"]),
                    identity=_stable_identity(metadata),
                )
        self._execution_inputs = execution_inputs
        self._publication = publication
        self._bound = MappingProxyType(bound)
        execution_inputs.gate()

    def _read(
        self,
        name: str,
        *,
        signal_date: dt.date | None,
        columns: tuple[str, ...] | None = None,
    ) -> pa.Table:
        self._execution_inputs.gate()
        record = self._bound[name]
        _require(
            _stable_identity(record.path.stat(follow_symlinks=False))
            == record.identity,
            f"accounting table changed during execution: {name}",
        )
        table = pq.read_table(
            record.path,
            columns=None if columns is None else list(columns),
            filters=(
                None if signal_date is None else [("signal_date", "=", signal_date)]
            ),
            use_threads=False,
        )
        _require(
            _stable_identity(record.path.stat(follow_symlinks=False))
            == record.identity,
            f"accounting table changed while read: {name}",
        )
        self._execution_inputs.gate()
        return table.combine_chunks()

    def month(self, base: str, signal_date: dt.date) -> pa.Table:
        _require(base in _ACCOUNTING_BASE_NAMES, "unknown accounting table role")
        parts = (
            self._read(f"{base}.parquet", signal_date=signal_date),
            self._read(f"outcome_reserve_{base}.parquet", signal_date=signal_date),
        )
        return pa.concat_tables(parts).combine_chunks()

    def all_session_dates(
        self,
        calendar_authority: AuthenticatedValidationCalendar,
    ) -> tuple[dt.date, ...]:
        _require(
            type(calendar_authority) is AuthenticatedValidationCalendar,
            "risk-age clock requires authenticated calendar authority",
        )
        parts = (
            self._read(
                "daily_sessions.parquet",
                signal_date=None,
                columns=("calendar_date",),
            ),
            self._read(
                "outcome_reserve_daily_sessions.parquet",
                signal_date=None,
                columns=("calendar_date",),
            ),
        )
        values = tuple(
            sorted(
                {
                    *(
                        value
                        for table in parts
                        for value in table["calendar_date"].to_pylist()
                    ),
                    *(row.entry_date for row in calendar_authority.rows),
                }
            )
        )
        _require(
            bool(values)
            and all(type(value) is dt.date for value in values)
            and all(left < right for left, right in zip(values, values[1:])),
            "accounting session date domain differs",
        )
        return values

    def paired_source_sha256(self, base: str) -> tuple[str, str]:
        """Return the already authenticated main/reserve file identities."""

        _require(base in _ACCOUNTING_BASE_NAMES, "unknown accounting table role")
        self._execution_inputs.gate()
        return (
            self._bound[f"{base}.parquet"].sha256,
            self._bound[f"outcome_reserve_{base}.parquet"].sha256,
        )


@dataclass(frozen=True, slots=True)
class _RiskMemory:
    signal_date: dt.date
    permco: int
    factor_betas: np.ndarray
    specific_variance_monthly: float
    sector: str
    country: str


@dataclass(slots=True)
class _StrategyState:
    portfolio: PortfolioState
    action_ledger: CorporateActionLedger
    mark_continuity_ledger: MarkContinuityLedger


_DIAGNOSTIC_STATE_SCHEMA_VERSION = 1
_DIAGNOSTIC_STATE_SCHEMA_DESCRIPTOR = {
    "schema_version": _DIAGNOSTIC_STATE_SCHEMA_VERSION,
    "strategy_fields": (
        "portfolio_permno",
        "portfolio_value_dollars",
        "cash_dollars",
        "action_ids",
        "action_last_date",
        "action_last_sequence",
        "mark_permno",
        "mark_last_date",
        "mark_pending_count",
        "mark_evidence_sha256",
    ),
    "shared_fields": (
        "risk_permno",
        "risk_signal_date",
        "risk_permco",
        "risk_factor_betas",
        "risk_specific_variance_monthly",
        "risk_sector",
        "risk_country",
        "retired_permno",
        "retired_date",
    ),
}


def diagnostic_state_schema_sha256() -> str:
    return hashlib.sha256(
        _canonical_json(_DIAGNOSTIC_STATE_SCHEMA_DESCRIPTOR)
    ).hexdigest()


def _serialize_portfolio_diagnostic_state(
    *,
    states: Mapping[str, _StrategyState],
    risk_memory: Mapping[int, _RiskMemory],
    retired_on: Mapping[int, dt.date],
) -> Mapping[str, np.ndarray]:
    """Serialize only continuation state into pickle-free numeric/text arrays."""

    _require(bool(states), "diagnostic continuation state has no strategies")
    strategy_ids = tuple(sorted(states))
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(
            [_DIAGNOSTIC_STATE_SCHEMA_VERSION], dtype=np.int64
        ),
        "strategy_ids": np.asarray(strategy_ids, dtype=np.str_),
    }
    for strategy_id in strategy_ids:
        state = states[strategy_id]
        prefix = f"strategy.{strategy_id}."
        payload[prefix + "portfolio_permno"] = np.asarray(
            state.portfolio.permno, dtype=np.int64
        )
        payload[prefix + "portfolio_value_dollars"] = np.asarray(
            state.portfolio.position_value_dollars, dtype=np.float64
        )
        payload[prefix + "cash_dollars"] = np.asarray(
            [state.portfolio.cash_dollars], dtype=np.float64
        )
        payload[prefix + "action_ids"] = np.asarray(
            state.action_ledger.applied_action_ids, dtype=np.str_
        )
        payload[prefix + "action_last_date"] = np.asarray(
            [
                ""
                if state.action_ledger.last_assigned_return_date is None
                else state.action_ledger.last_assigned_return_date.isoformat()
            ],
            dtype=np.str_,
        )
        payload[prefix + "action_last_sequence"] = np.asarray(
            [state.action_ledger.last_action_sequence], dtype=np.int64
        )
        mark = state.mark_continuity_ledger
        payload[prefix + "mark_permno"] = np.asarray(
            mark.permno, dtype=np.int64
        )
        payload[prefix + "mark_last_date"] = np.asarray(
            [value.isoformat() for value in mark.last_authoritative_mark_date],
            dtype=np.str_,
        )
        payload[prefix + "mark_pending_count"] = np.asarray(
            mark.pending_no_mark_count, dtype=np.int64
        )
        payload[prefix + "mark_evidence_sha256"] = np.asarray(
            mark.pending_evidence_sha256, dtype=np.str_
        )

    risk_permnos = np.asarray(sorted(risk_memory), dtype=np.int64)
    payload["risk_permno"] = risk_permnos
    payload["risk_signal_date"] = np.asarray(
        [risk_memory[int(value)].signal_date.isoformat() for value in risk_permnos],
        dtype=np.str_,
    )
    payload["risk_permco"] = np.asarray(
        [risk_memory[int(value)].permco for value in risk_permnos],
        dtype=np.int64,
    )
    payload["risk_factor_betas"] = (
        np.vstack(
            [
                np.asarray(
                    risk_memory[int(value)].factor_betas,
                    dtype=np.float64,
                )
                for value in risk_permnos
            ]
        )
        if len(risk_permnos)
        else np.empty((0, 6), dtype=np.float64)
    )
    payload["risk_specific_variance_monthly"] = np.asarray(
        [
            risk_memory[int(value)].specific_variance_monthly
            for value in risk_permnos
        ],
        dtype=np.float64,
    )
    payload["risk_sector"] = np.asarray(
        [risk_memory[int(value)].sector for value in risk_permnos],
        dtype=np.str_,
    )
    payload["risk_country"] = np.asarray(
        [risk_memory[int(value)].country for value in risk_permnos],
        dtype=np.str_,
    )
    retired_permnos = np.asarray(sorted(retired_on), dtype=np.int64)
    payload["retired_permno"] = retired_permnos
    payload["retired_date"] = np.asarray(
        [retired_on[int(value)].isoformat() for value in retired_permnos],
        dtype=np.str_,
    )
    return MappingProxyType(payload)


def _restore_portfolio_diagnostic_state(
    payload: Mapping[str, np.ndarray],
) -> tuple[
    dict[str, _StrategyState],
    dict[int, _RiskMemory],
    dict[int, dt.date],
]:
    """Reconstruct and validate a private continuation-state payload."""

    try:
        schema = np.asarray(payload["schema_version"], dtype=np.int64)
        strategy_values = np.asarray(payload["strategy_ids"], dtype=np.str_)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise EmpiricalPortfolioBackendError(
            "diagnostic continuation state header is invalid"
        ) from exc
    _require(
        schema.shape == (1,)
        and int(schema[0]) == _DIAGNOSTIC_STATE_SCHEMA_VERSION
        and strategy_values.ndim == 1
        and len(strategy_values) > 0,
        "diagnostic continuation state header differs",
    )
    strategy_ids = tuple(str(value) for value in strategy_values)
    _require(
        strategy_ids == tuple(sorted(set(strategy_ids)))
        and all(strategy_ids),
        "diagnostic continuation strategy identities differ",
    )
    common_keys = {
        "schema_version",
        "strategy_ids",
        "risk_permno",
        "risk_signal_date",
        "risk_permco",
        "risk_factor_betas",
        "risk_specific_variance_monthly",
        "risk_sector",
        "risk_country",
        "retired_permno",
        "retired_date",
    }
    suffixes = {
        "portfolio_permno",
        "portfolio_value_dollars",
        "cash_dollars",
        "action_ids",
        "action_last_date",
        "action_last_sequence",
        "mark_permno",
        "mark_last_date",
        "mark_pending_count",
        "mark_evidence_sha256",
    }
    expected_keys = common_keys | {
        f"strategy.{strategy_id}.{suffix}"
        for strategy_id in strategy_ids
        for suffix in suffixes
    }
    _require(
        set(payload) == expected_keys,
        "diagnostic continuation state fields differ",
    )

    states: dict[str, _StrategyState] = {}
    for strategy_id in strategy_ids:
        prefix = f"strategy.{strategy_id}."
        action_date_values = np.asarray(
            payload[prefix + "action_last_date"], dtype=np.str_
        )
        action_sequence = np.asarray(
            payload[prefix + "action_last_sequence"], dtype=np.int64
        )
        cash = np.asarray(payload[prefix + "cash_dollars"], dtype=np.float64)
        _require(
            action_date_values.shape == (1,)
            and action_sequence.shape == (1,)
            and cash.shape == (1,),
            "diagnostic strategy scalar state differs",
        )
        action_date_text = str(action_date_values[0])
        action_date = (
            None
            if not action_date_text
            else dt.date.fromisoformat(action_date_text)
        )
        mark_dates = tuple(
            dt.date.fromisoformat(str(value))
            for value in np.asarray(
                payload[prefix + "mark_last_date"], dtype=np.str_
            )
        )
        states[strategy_id] = _StrategyState(
            portfolio=PortfolioState(
                permno=np.asarray(
                    payload[prefix + "portfolio_permno"], dtype=np.int64
                ),
                position_value_dollars=np.asarray(
                    payload[prefix + "portfolio_value_dollars"],
                    dtype=np.float64,
                ),
                cash_dollars=float(cash[0]),
            ),
            action_ledger=CorporateActionLedger(
                applied_action_ids=tuple(
                    str(value)
                    for value in np.asarray(
                        payload[prefix + "action_ids"], dtype=np.str_
                    )
                ),
                last_assigned_return_date=action_date,
                last_action_sequence=int(action_sequence[0]),
            ),
            mark_continuity_ledger=MarkContinuityLedger(
                permno=np.asarray(
                    payload[prefix + "mark_permno"], dtype=np.int64
                ),
                last_authoritative_mark_date=mark_dates,
                pending_no_mark_count=np.asarray(
                    payload[prefix + "mark_pending_count"], dtype=np.int64
                ),
                pending_evidence_sha256=tuple(
                    str(value)
                    for value in np.asarray(
                        payload[prefix + "mark_evidence_sha256"],
                        dtype=np.str_,
                    )
                ),
            ),
        )

    risk_permno = np.asarray(payload["risk_permno"], dtype=np.int64)
    risk_dates = np.asarray(payload["risk_signal_date"], dtype=np.str_)
    risk_permco = np.asarray(payload["risk_permco"], dtype=np.int64)
    risk_betas = np.asarray(payload["risk_factor_betas"], dtype=np.float64)
    risk_specific = np.asarray(
        payload["risk_specific_variance_monthly"], dtype=np.float64
    )
    risk_sector = np.asarray(payload["risk_sector"], dtype=np.str_)
    risk_country = np.asarray(payload["risk_country"], dtype=np.str_)
    risk_rows = len(risk_permno)
    _require(
        risk_permno.ndim == 1
        and (
            risk_rows == 0
            or (
                np.all(risk_permno > 0)
                and np.all(risk_permno[:-1] < risk_permno[1:])
            )
        )
        and risk_dates.shape == (risk_rows,)
        and risk_permco.shape == (risk_rows,)
        and risk_betas.shape == (risk_rows, 6)
        and risk_specific.shape == (risk_rows,)
        and risk_sector.shape == (risk_rows,)
        and risk_country.shape == (risk_rows,)
        and np.isfinite(risk_betas).all()
        and np.isfinite(risk_specific).all(),
        "diagnostic risk-memory state differs",
    )
    risk_memory: dict[int, _RiskMemory] = {}
    for index, raw_permno in enumerate(risk_permno):
        betas = np.array(risk_betas[index], dtype=np.float64, copy=True)
        betas.setflags(write=False)
        risk_memory[int(raw_permno)] = _RiskMemory(
            signal_date=dt.date.fromisoformat(str(risk_dates[index])),
            permco=int(risk_permco[index]),
            factor_betas=betas,
            specific_variance_monthly=float(risk_specific[index]),
            sector=str(risk_sector[index]),
            country=str(risk_country[index]),
        )

    retired_permno = np.asarray(payload["retired_permno"], dtype=np.int64)
    retired_dates = np.asarray(payload["retired_date"], dtype=np.str_)
    _require(
        retired_permno.ndim == 1
        and retired_dates.shape == retired_permno.shape
        and (
            len(retired_permno) == 0
            or (
                np.all(retired_permno > 0)
                and np.all(retired_permno[:-1] < retired_permno[1:])
            )
        ),
        "diagnostic retired-identity state differs",
    )
    retired_on = {
        int(raw_permno): dt.date.fromisoformat(str(raw_date))
        for raw_permno, raw_date in zip(
            retired_permno, retired_dates, strict=True
        )
    }
    return states, risk_memory, retired_on


@dataclass(frozen=True, slots=True)
class EmpiricalMarkContinuityEvidence:
    """One strategy/session receipt kept outside the published table contract."""

    strategy_id: str
    holding_month: str
    trading_date: dt.date
    opening_ledger_sha256: str
    closing_ledger_sha256: str
    no_mark_permnos: tuple[int, ...]
    resolved_pending_permnos: tuple[int, ...]
    observed_zero_return_permnos: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EmpiricalRetiredIdentityEvidence:
    """Execution-only block for a PERMNO retired by an earlier close."""

    holding_month: str
    signal_date: dt.date
    entry_date: dt.date
    permno: int
    retirement_date: dt.date
    prior_long_entry_allowed: bool
    prior_short_entry_allowed: bool

    def __post_init__(self) -> None:
        _require(
            isinstance(self.holding_month, str)
            and bool(self.holding_month)
            and type(self.signal_date) is dt.date
            and type(self.entry_date) is dt.date
            and type(self.retirement_date) is dt.date
            and type(self.permno) is int
            and self.permno > 0,
            "retired-identity execution evidence is malformed",
        )
        _require(
            self.signal_date < self.entry_date
            and self.retirement_date <= self.entry_date,
            "retired-identity execution evidence uses a future event",
        )
        _require(
            type(self.prior_long_entry_allowed) is bool
            and type(self.prior_short_entry_allowed) is bool
            and (
                self.prior_long_entry_allowed
                or self.prior_short_entry_allowed
            ),
            "retired-identity evidence did not change an execution gate",
        )


@dataclass(frozen=True, slots=True)
class EmpiricalPendingDustRetirementEvidence:
    """Exact nontrade evidence for one pending sub-cent boundary retirement."""

    policy_id: str
    strategy_id: str
    holding_month: str
    boundary_date: dt.date
    permno: int
    position_value_dollars: float
    threshold_dollars: float
    last_authoritative_mark_date: dt.date
    pending_no_mark_count: int
    pending_evidence_sha256: str
    opening_portfolio_state_sha256: str
    closing_portfolio_state_sha256: str
    opening_mark_ledger_sha256: str
    closing_mark_ledger_sha256: str
    opening_nav_dollars: float
    closing_nav_dollars: float
    opening_cash_dollars: float
    closing_cash_dollars: float
    cash_adjustment_dollars: float
    cash_rounding_residual_dollars: float
    execution_price_read: bool = False
    return_imputed: bool = False
    trade_emitted: bool = False
    transaction_cost_imputed: bool = False
    corporate_action_applied: bool = False
    identity_terminally_retired: bool = False

    def __post_init__(self) -> None:
        digests = (
            self.pending_evidence_sha256,
            self.opening_portfolio_state_sha256,
            self.closing_portfolio_state_sha256,
            self.opening_mark_ledger_sha256,
            self.closing_mark_ledger_sha256,
        )
        floats = (
            self.position_value_dollars,
            self.threshold_dollars,
            self.opening_nav_dollars,
            self.closing_nav_dollars,
            self.opening_cash_dollars,
            self.closing_cash_dollars,
            self.cash_adjustment_dollars,
            self.cash_rounding_residual_dollars,
        )
        _require(
            self.policy_id == PENDING_DUST_RETIREMENT_POLICY_ID
            and isinstance(self.strategy_id, str)
            and bool(self.strategy_id)
            and isinstance(self.holding_month, str)
            and bool(self.holding_month)
            and type(self.boundary_date) is dt.date
            and type(self.last_authoritative_mark_date) is dt.date
            and self.last_authoritative_mark_date < self.boundary_date
            and type(self.permno) is int
            and self.permno > 0
            and type(self.pending_no_mark_count) is int
            and self.pending_no_mark_count > 0,
            "pending-dust retirement evidence identity differs",
        )
        _require(
            all(type(value) is float and math.isfinite(value) for value in floats)
            and self.threshold_dollars
            == PENDING_DUST_RETIREMENT_THRESHOLD_DOLLARS
            and 0.0
            < abs(self.position_value_dollars)
            < self.threshold_dollars
            and self.opening_nav_dollars > 0.0
            and self.opening_nav_dollars == self.closing_nav_dollars
            and self.cash_adjustment_dollars
            == self.closing_cash_dollars - self.opening_cash_dollars
            and self.cash_rounding_residual_dollars
            == self.cash_adjustment_dollars - self.position_value_dollars
            and abs(self.cash_rounding_residual_dollars)
            < self.threshold_dollars,
            "pending-dust retirement evidence accounting differs",
        )
        _require(
            all(
                isinstance(value, str)
                and len(value) == 64
                and value == value.lower()
                and all(character in "0123456789abcdef" for character in value)
                for value in digests
            )
            and self.opening_portfolio_state_sha256
            != self.closing_portfolio_state_sha256
            and self.opening_mark_ledger_sha256
            != self.closing_mark_ledger_sha256,
            "pending-dust retirement evidence hashes differ",
        )
        _require(
            all(
                type(value) is bool and value is False
                for value in (
                    self.execution_price_read,
                    self.return_imputed,
                    self.trade_emitted,
                    self.transaction_cost_imputed,
                    self.corporate_action_applied,
                    self.identity_terminally_retired,
                )
            ),
            "pending-dust retirement fabricated an economic event",
        )

    @property
    def evidence_sha256(self) -> str:
        """Return a canonical digest with every float encoded exactly."""

        payload = {
            "policy_id": self.policy_id,
            "strategy_id": self.strategy_id,
            "holding_month": self.holding_month,
            "boundary_date": self.boundary_date.isoformat(),
            "permno": self.permno,
            "position_value_dollars_hex": self.position_value_dollars.hex(),
            "threshold_dollars_hex": self.threshold_dollars.hex(),
            "last_authoritative_mark_date": (
                self.last_authoritative_mark_date.isoformat()
            ),
            "pending_no_mark_count": self.pending_no_mark_count,
            "pending_evidence_sha256": self.pending_evidence_sha256,
            "opening_portfolio_state_sha256": (
                self.opening_portfolio_state_sha256
            ),
            "closing_portfolio_state_sha256": (
                self.closing_portfolio_state_sha256
            ),
            "opening_mark_ledger_sha256": self.opening_mark_ledger_sha256,
            "closing_mark_ledger_sha256": self.closing_mark_ledger_sha256,
            "opening_nav_dollars_hex": self.opening_nav_dollars.hex(),
            "closing_nav_dollars_hex": self.closing_nav_dollars.hex(),
            "opening_cash_dollars_hex": self.opening_cash_dollars.hex(),
            "closing_cash_dollars_hex": self.closing_cash_dollars.hex(),
            "cash_adjustment_dollars_hex": self.cash_adjustment_dollars.hex(),
            "cash_rounding_residual_dollars_hex": (
                self.cash_rounding_residual_dollars.hex()
            ),
            "execution_price_read": self.execution_price_read,
            "return_imputed": self.return_imputed,
            "trade_emitted": self.trade_emitted,
            "transaction_cost_imputed": self.transaction_cost_imputed,
            "corporate_action_applied": self.corporate_action_applied,
            "identity_terminally_retired": self.identity_terminally_retired,
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _portfolio_with_exact_nav(
    *,
    permno: np.ndarray,
    position_value_dollars: np.ndarray,
    target_nav_dollars: float,
) -> PortfolioState:
    """Solve the cash residual without accepting even a one-ULP NAV change."""

    position_sum = float(
        np.sum(position_value_dollars, dtype=np.float64)
    )
    cash = float(target_nav_dollars - position_sum)
    for _attempt in range(64):
        state = PortfolioState(
            permno=permno,
            position_value_dollars=position_value_dollars,
            cash_dollars=cash,
        )
        observed_nav = state.nav_dollars
        if observed_nav == target_nav_dollars:
            return state
        direction = math.inf if observed_nav < target_nav_dollars else -math.inf
        next_cash = float(np.nextafter(np.float64(cash), direction))
        if next_cash == cash:
            break
        cash = next_cash
    raise EmpiricalPortfolioBackendError(
        "pending-dust retirement cannot preserve NAV exactly"
    )


def _retire_pending_subcent_dust_at_month_boundary(
    *,
    strategy_id: str,
    holding_month: str,
    boundary_date: dt.date,
    state: _StrategyState,
) -> tuple[_StrategyState, tuple[EmpiricalPendingDustRetirementEvidence, ...]]:
    """Move only unresolved sub-cent marks to cash at a post-month boundary."""

    _require(
        isinstance(strategy_id, str)
        and bool(strategy_id)
        and isinstance(holding_month, str)
        and bool(holding_month)
        and type(boundary_date) is dt.date
        and type(state) is _StrategyState,
        "pending-dust boundary normalization input differs",
    )
    portfolio_index = {
        int(permno): index
        for index, permno in enumerate(state.portfolio.permno)
    }
    pending_permnos = state.mark_continuity_ledger.pending_permnos
    _require(
        all(permno in portfolio_index for permno in pending_permnos),
        "pending mark identity escaped its portfolio state",
    )
    candidates = tuple(
        permno
        for permno in pending_permnos
        if 0.0
        < abs(
            float(
                state.portfolio.position_value_dollars[
                    portfolio_index[permno]
                ]
            )
        )
        < PENDING_DUST_RETIREMENT_THRESHOLD_DOLLARS
    )
    if not candidates:
        return state, ()

    current = state
    evidence: list[EmpiricalPendingDustRetirementEvidence] = []
    for permno in candidates:
        opening_portfolio = current.portfolio
        opening_marks = current.mark_continuity_ledger
        current_portfolio_index = {
            int(value): index
            for index, value in enumerate(opening_portfolio.permno)
        }
        current_mark_index = {
            int(value): index for index, value in enumerate(opening_marks.permno)
        }
        portfolio_position = current_portfolio_index.get(permno)
        mark_position = current_mark_index.get(permno)
        _require(
            portfolio_position is not None and mark_position is not None,
            "pending-dust retirement state changed during normalization",
        )
        position_value = float(
            opening_portfolio.position_value_dollars[portfolio_position]
        )
        pending_count = int(
            opening_marks.pending_no_mark_count[mark_position]
        )
        _require(
            pending_count > 0
            and 0.0
            < abs(position_value)
            < PENDING_DUST_RETIREMENT_THRESHOLD_DOLLARS,
            "pending-dust retirement candidate escaped its strict boundary",
        )
        opening_nav = opening_portfolio.nav_dollars
        opening_state_sha = portfolio_state_sha256(opening_portfolio)
        opening_mark_sha = opening_marks.ledger_sha256

        closing_values = np.array(
            opening_portfolio.position_value_dollars,
            dtype=np.float64,
            copy=True,
        )
        closing_values[portfolio_position] = 0.0
        closing_portfolio = _portfolio_with_exact_nav(
            permno=opening_portfolio.permno,
            position_value_dollars=closing_values,
            target_nav_dollars=opening_nav,
        )
        keep_marks = np.ones(len(opening_marks.permno), dtype=bool)
        keep_marks[mark_position] = False
        closing_marks = MarkContinuityLedger(
            permno=opening_marks.permno[keep_marks],
            last_authoritative_mark_date=tuple(
                value
                for index, value in enumerate(
                    opening_marks.last_authoritative_mark_date
                )
                if bool(keep_marks[index])
            ),
            pending_no_mark_count=(
                opening_marks.pending_no_mark_count[keep_marks]
            ),
            pending_evidence_sha256=tuple(
                value
                for index, value in enumerate(
                    opening_marks.pending_evidence_sha256
                )
                if bool(keep_marks[index])
            ),
        )
        closing_state = _StrategyState(
            portfolio=closing_portfolio,
            action_ledger=current.action_ledger,
            mark_continuity_ledger=closing_marks,
        )
        cash_adjustment = float(
            closing_portfolio.cash_dollars
            - opening_portfolio.cash_dollars
        )
        cash_rounding_residual = float(cash_adjustment - position_value)
        evidence.append(
            EmpiricalPendingDustRetirementEvidence(
                policy_id=PENDING_DUST_RETIREMENT_POLICY_ID,
                strategy_id=strategy_id,
                holding_month=holding_month,
                boundary_date=boundary_date,
                permno=permno,
                position_value_dollars=position_value,
                threshold_dollars=(
                    PENDING_DUST_RETIREMENT_THRESHOLD_DOLLARS
                ),
                last_authoritative_mark_date=(
                    opening_marks.last_authoritative_mark_date[mark_position]
                ),
                pending_no_mark_count=pending_count,
                pending_evidence_sha256=(
                    opening_marks.pending_evidence_sha256[mark_position]
                ),
                opening_portfolio_state_sha256=opening_state_sha,
                closing_portfolio_state_sha256=portfolio_state_sha256(
                    closing_portfolio
                ),
                opening_mark_ledger_sha256=opening_mark_sha,
                closing_mark_ledger_sha256=closing_marks.ledger_sha256,
                opening_nav_dollars=opening_nav,
                closing_nav_dollars=closing_portfolio.nav_dollars,
                opening_cash_dollars=opening_portfolio.cash_dollars,
                closing_cash_dollars=closing_portfolio.cash_dollars,
                cash_adjustment_dollars=cash_adjustment,
                cash_rounding_residual_dollars=cash_rounding_residual,
            )
        )
        current = closing_state

    _require(
        current.action_ledger is state.action_ledger
        and current.portfolio.nav_dollars == state.portfolio.nav_dollars
        and all(
            permno not in current.mark_continuity_ledger.pending_permnos
            for permno in candidates
        ),
        "pending-dust retirement boundary invariants differ",
    )
    return current, tuple(evidence)


@dataclass(frozen=True, slots=True)
class EmpiricalPortfolioRun:
    execution_sessions: pd.DataFrame
    execution_session_receipts: Mapping[str, Mapping[str, str]]
    portfolio_weights: pd.DataFrame
    portfolio_trades: pd.DataFrame
    daily_accounting: pd.DataFrame
    monthly_metrics: pd.DataFrame
    mark_continuity_evidence: tuple[EmpiricalMarkContinuityEvidence, ...] = ()
    retired_identity_evidence: tuple[EmpiricalRetiredIdentityEvidence, ...] = ()
    pending_dust_retirement_evidence: tuple[
        EmpiricalPendingDustRetirementEvidence, ...
    ] = ()
    final_mark_continuity_ledger_sha256: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    final_pending_permnos: Mapping[str, tuple[int, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    final_pending_nonzero_count: int = 0

    def __post_init__(self) -> None:
        observed_pending_count = sum(
            len(values) for values in self.final_pending_permnos.values()
        )
        _require(
            type(self.final_pending_nonzero_count) is int
            and self.final_pending_nonzero_count == observed_pending_count
            and self.final_pending_nonzero_count == 0,
            "empirical portfolio final handoff retains a nonzero pending identity",
        )
        _require(
            set(self.final_mark_continuity_ledger_sha256)
            == set(self.final_pending_permnos),
            "empirical portfolio final mark-continuity evidence differs",
        )
        retired_keys = tuple(
            (value.holding_month, value.permno)
            for value in self.retired_identity_evidence
        )
        _require(
            all(
                isinstance(value, EmpiricalRetiredIdentityEvidence)
                for value in self.retired_identity_evidence
            )
            and len(retired_keys) == len(set(retired_keys)),
            "empirical retired-identity evidence is duplicated or malformed",
        )
        pending_dust_keys = tuple(
            (
                value.boundary_date,
                value.strategy_id,
                value.holding_month,
                value.permno,
            )
            for value in self.pending_dust_retirement_evidence
        )
        _require(
            all(
                isinstance(value, EmpiricalPendingDustRetirementEvidence)
                for value in self.pending_dust_retirement_evidence
            )
            and len(pending_dust_keys) == len(set(pending_dust_keys))
            and pending_dust_keys == tuple(sorted(pending_dust_keys)),
            "empirical pending-dust evidence is duplicated, unordered, or malformed",
        )

    @property
    def pending_dust_retirement_evidence_sha256(self) -> str:
        """Bind the exact ordered pending-dust evidence without table changes."""

        digest = hashlib.sha256(
            b"stage11-pending-subcent-dust-retirement-evidence-v1\0"
        )
        for value in self.pending_dust_retirement_evidence:
            digest.update(value.evidence_sha256.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Stage11PortfolioDiagnosticStatus:
    """Performance-blind progress from the resumable portfolio diagnostic path."""

    status: str
    completed_month_count: int
    next_holding_month: str | None
    latest_checkpoint_sha256: str | None
    failure_report_sha256: str | None
    failure_count: int
    performance_fields_emitted: int = 0

    def __post_init__(self) -> None:
        _require(
            self.status
            in {
                "diagnostic_path_complete",
                "diagnostic_month_failed",
            },
            "diagnostic portfolio status differs",
        )
        _require(
            type(self.completed_month_count) is int
            and 0 <= self.completed_month_count <= len(VALIDATION_HOLDING_MONTHS),
            "diagnostic completed-month count differs",
        )
        expected_next = (
            None
            if self.completed_month_count == len(VALIDATION_HOLDING_MONTHS)
            else VALIDATION_HOLDING_MONTHS[self.completed_month_count]
        )
        _require(
            self.next_holding_month == expected_next,
            "diagnostic next holding month differs",
        )
        if self.latest_checkpoint_sha256 is not None:
            _require(
                len(self.latest_checkpoint_sha256) == 64,
                "diagnostic checkpoint identity differs",
            )
        if self.status == "diagnostic_month_failed":
            _require(
                isinstance(self.failure_report_sha256, str)
                and len(self.failure_report_sha256) == 64
                and 1 <= self.failure_count <= 3,
                "diagnostic failure status differs",
            )
        else:
            _require(
                self.completed_month_count == len(VALIDATION_HOLDING_MONTHS)
                and self.failure_report_sha256 is None
                and self.failure_count == 0,
                "diagnostic completion status differs",
            )
        _require(
            self.performance_fields_emitted == 0,
            "diagnostic status exposed performance fields",
        )


@dataclass(frozen=True, slots=True)
class _DiagnosticExceptionEvidence:
    strategy_id: str
    failure_code: str
    exception_class: str
    private_detail: bytes = field(repr=False)


def _apply_retired_identity_execution_gate(
    market: MarketSnapshot,
    *,
    retired_on: Mapping[int, dt.date],
    holding_month: str,
    signal_date: dt.date,
    entry_date: dt.date,
) -> tuple[MarketSnapshot, tuple[EmpiricalRetiredIdentityEvidence, ...]]:
    """Block new trades only after a terminal action retired the PERMNO."""

    _require(
        isinstance(market, MarketSnapshot)
        and isinstance(retired_on, Mapping)
        and isinstance(holding_month, str)
        and bool(holding_month)
        and type(signal_date) is dt.date
        and type(entry_date) is dt.date
        and signal_date < entry_date,
        "retired-identity execution gate inputs differ",
    )
    normalized: dict[int, dt.date] = {}
    for raw_permno, retirement_date in retired_on.items():
        _require(
            type(raw_permno) is int
            and raw_permno > 0
            and type(retirement_date) is dt.date
            and retirement_date <= entry_date,
            "retired-identity execution gate received a future or invalid event",
        )
        normalized[raw_permno] = retirement_date

    long_allowed = market.new_long_trade_allowed.copy()
    short_allowed = market.new_short_trade_allowed.copy()
    evidence: list[EmpiricalRetiredIdentityEvidence] = []
    for position, raw_permno in enumerate(market.permno):
        permno = int(raw_permno)
        retirement_date = normalized.get(permno)
        if retirement_date is None:
            continue
        prior_long = bool(long_allowed[position])
        prior_short = bool(short_allowed[position])
        long_allowed[position] = False
        short_allowed[position] = False
        if prior_long or prior_short:
            evidence.append(
                EmpiricalRetiredIdentityEvidence(
                    holding_month=holding_month,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    permno=permno,
                    retirement_date=retirement_date,
                    prior_long_entry_allowed=prior_long,
                    prior_short_entry_allowed=prior_short,
                )
            )
    return (
        replace(
            market,
            new_long_trade_allowed=long_allowed,
            new_short_trade_allowed=short_allowed,
        ),
        tuple(evidence),
    )


def _read_risk_cost_tables(
    execution_inputs: AuthenticatedStage11ExecutionInputs,
) -> Mapping[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    verification = execution_inputs.predictor.validation_data_verification
    for name in _RISK_NAMES:
        record = execution_inputs.registrations[name]
        request = execution_inputs.broker.authorize_table_request(
            name,
            role=record.role,
            columns=record.allowed_columns,
            minimum_date=record.minimum_date,
            maximum_date=record.maximum_date,
        )
        table = execution_inputs.broker.read_guarded_parquet(request)
        receipt = verification.payload_receipt(name)
        _require(
            table.num_rows == receipt["rows"]
            and tuple(table.column_names) == record.allowed_columns,
            f"brokered risk/cost table differs: {name}",
        )
        frame = table.to_pandas()
        frame["signal_date"] = pd.to_datetime(
            frame["signal_date"], errors="raise"
        ).dt.date
        tables[name] = frame
    audit = tables["validation_risk_cost_audit"]
    processed = tables["validation_risk_cost_inputs"]
    expected_rows = (
        audit.loc[audit["long_eligible"].astype(bool)]
        .sort_values(["signal_date", "permno"], kind="mergesort")
        .reset_index(drop=True)
    )
    observed_rows = processed.sort_values(
        ["signal_date", "permno"], kind="mergesort"
    ).reset_index(drop=True)
    _require(
        expected_rows.equals(observed_rows),
        "processed risk/cost rows differ from the audited eligible domain",
    )
    execution_inputs.gate()
    return MappingProxyType(tables)


def _monthly_identity(
    execution_inputs: AuthenticatedStage11ExecutionInputs,
    publication: Stage11OutcomePublication,
    selection: LockedValidationSelection,
    signal_date: dt.date,
) -> AuthenticatedMonthlyIdentity:
    risk_receipts = (
        execution_inputs.predictor.validation_data_verification.payload_receipts
    )
    outcome_receipts = publication.receipt["tables"]
    payload = {
        "execution_input_manifest_sha256": execution_inputs.manifest_sha256,
        "selection_sha256": selection.selection_sha256,
        "signal_date": signal_date.isoformat(),
        "risk_cost_sha256": {
            name: risk_receipts[name]["sha256"] for name in _RISK_NAMES
        },
        "accounting_sha256": {
            name: outcome_receipts[name]["sha256"]
            for base in _ACCOUNTING_BASE_NAMES
            for name in (f"{base}.parquet", f"outcome_reserve_{base}.parquet")
        },
    }
    return AuthenticatedMonthlyIdentity(
        signal_date=signal_date,
        source_bundle_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
    )


def _monthly_market_and_risk(
    *,
    row: object,
    identity: AuthenticatedMonthlyIdentity,
    risk_tables: Mapping[str, pd.DataFrame],
    store: AuthenticatedOutcomeAccountingStore,
) -> tuple[pd.DataFrame, MarketSnapshot, MonthlyRiskInputs]:
    signal_date = row.signal_date
    audit = risk_tables["validation_risk_cost_audit"]
    month = audit.loc[audit["signal_date"].eq(signal_date)].copy()
    _require(
        not month.empty
        and not month["permno"].duplicated().any()
        and month["permno"].notna().all(),
        "monthly risk audit is empty or duplicated",
    )
    month["permno"] = pd.to_numeric(month["permno"], errors="raise").astype(np.int64)
    month["permco"] = pd.to_numeric(month["permco"], errors="raise").astype(np.int64)
    month = month.sort_values("permno", kind="mergesort").reset_index(drop=True)
    entry = store.month("entry_closes", signal_date).to_pandas()
    _require(
        not entry["permno"].duplicated().any(),
        "monthly entry-close domain is duplicated",
    )
    entry["permno"] = pd.to_numeric(entry["permno"], errors="raise").astype(np.int64)
    month = month.merge(
        entry[["permno", "close_usd_per_share", "entry_close_status"]],
        on="permno",
        how="left",
        validate="one_to_one",
    )
    _require(
        month["entry_close_status"].notna().all(),
        "a current predictor row lacks its authenticated entry-close status",
    )
    numeric = lambda name: pd.to_numeric(month[name], errors="coerce").to_numpy(float)
    betas = np.column_stack([numeric(f"beta_risk_{name}") for name in FACTOR_NAMES])
    market = MarketSnapshot(
        identity=identity,
        permno=month["permno"].to_numpy(np.int64),
        permco=month["permco"].to_numpy(np.int64),
        execution_price=numeric("close_usd_per_share"),
        full_spread=numeric("full_spread_20d"),
        adv20_usd=numeric("adv20_usd"),
        impact_volatility_daily=numeric("impact_volatility_daily"),
        factor_betas=betas,
        specific_variance_monthly=numeric("specific_variance_monthly"),
        sector=month["ff12_sector_category"]
        .fillna("UNKNOWN_SECTOR")
        .astype(str)
        .to_numpy(),
        country=month["domicile_country_category"]
        .fillna("UNKNOWN_COUNTRY")
        .astype(str)
        .to_numpy(),
        new_long_trade_allowed=month["new_long_trade_allowed"].astype(bool).to_numpy(),
        new_short_trade_allowed=month["new_short_trade_allowed"]
        .astype(bool)
        .to_numpy(),
    )

    covariance_rows = (
        risk_tables["validation_factor_covariance"]
        .loc[lambda value: value["signal_date"].eq(signal_date)]
        .copy()
    )
    _require(len(covariance_rows) == 36, "monthly factor covariance is not 6 by 6")
    covariance_rows = covariance_rows.sort_values(
        ["factor_row_ordinal", "factor_col_ordinal"], kind="mergesort"
    )
    row_ord = pd.to_numeric(
        covariance_rows["factor_row_ordinal"], errors="raise"
    ).to_numpy(int)
    col_ord = pd.to_numeric(
        covariance_rows["factor_col_ordinal"], errors="raise"
    ).to_numpy(int)
    _require(
        np.array_equal(row_ord, np.repeat(np.arange(6), 6))
        and np.array_equal(col_ord, np.tile(np.arange(6), 6))
        and tuple(covariance_rows["factor_row"].iloc[::6].astype(str)) == FACTOR_NAMES
        and tuple(covariance_rows["factor_col"].iloc[:6].astype(str)) == FACTOR_NAMES,
        "monthly factor covariance ordinal/name identity differs",
    )
    covariance = (
        pd.to_numeric(covariance_rows["covariance_monthly"], errors="coerce")
        .to_numpy(float)
        .reshape(6, 6)
    )
    _require(
        np.isfinite(covariance).all()
        and np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12),
        "monthly factor covariance is missing or asymmetric",
    )
    eligible_risk = risk_tables["validation_risk_cost_inputs"].loc[
        lambda value: value["signal_date"].eq(signal_date)
        & value["risk_status"].eq(PRESENT)
    ]
    present_specific = pd.to_numeric(
        eligible_risk["specific_variance_monthly"],
        errors="coerce",
    ).to_numpy(float)
    present_specific = present_specific[
        np.isfinite(present_specific) & (present_specific > 0.0)
    ]
    p95 = type7_quantile(present_specific, 0.95)
    _require(p95 is not None and p95 > 0.0, "monthly specific-variance p95 is absent")
    summary = risk_tables["validation_risk_cost_month_summary"].loc[
        lambda value: value["signal_date"].eq(signal_date)
    ]
    _require(len(summary) == 1, "monthly risk/cost summary is absent or duplicated")
    exit_p95 = float(summary.iloc[0]["exit_volatility_p95_daily"])
    risk = MonthlyRiskInputs(
        identity=identity,
        factor_covariance_monthly=covariance,
        specific_variance_p95_monthly=float(p95),
        required_exit_impact_volatility_p95_daily=exit_p95,
    )
    return month, market, risk


def _monthly_scored(
    *,
    selected_scores: pd.DataFrame,
    audit_month: pd.DataFrame,
    identity: AuthenticatedMonthlyIdentity,
    signal_date: dt.date,
    score_column: str,
    kappa: float,
) -> ScoredSnapshot:
    scores = selected_scores.loc[
        selected_scores["signal_date"].eq(signal_date),
        ["permno", score_column],
    ].copy()
    _require(
        not scores.empty and not scores["permno"].duplicated().any(),
        f"monthly selected score is empty or duplicated: {score_column}",
    )
    eligible = audit_month.loc[
        audit_month["long_eligible"].astype(bool),
        ["permno", "short_eligible"],
    ]
    _require(
        len(scores) == len(eligible)
        and set(pd.to_numeric(scores["permno"], errors="raise").astype(np.int64))
        == set(pd.to_numeric(eligible["permno"], errors="raise").astype(np.int64)),
        f"selected score domain differs from long eligibility: {score_column}",
    )
    scores = scores.merge(eligible, on="permno", how="inner", validate="one_to_one")
    _require(
        len(scores) == len(eligible),
        f"selected score does not cover the exact long-eligible domain: {score_column}",
    )
    scores = scores.sort_values("permno", kind="mergesort")
    return ScoredSnapshot(
        identity=identity,
        permno=scores["permno"].to_numpy(np.int64),
        score=pd.to_numeric(scores[score_column], errors="coerce").to_numpy(float),
        calibration_kappa_monthly=kappa,
        long_eligible=np.ones(len(scores), dtype=bool),
        short_eligible=scores["short_eligible"].astype(bool).to_numpy(),
    )


def _update_risk_memory(
    memory: dict[int, _RiskMemory],
    audit_month: pd.DataFrame,
    signal_date: dt.date,
) -> None:
    for row in audit_month.loc[audit_month["risk_status"].eq(PRESENT)].itertuples(
        index=False
    ):
        betas = np.asarray(
            [getattr(row, f"beta_risk_{name}") for name in FACTOR_NAMES],
            dtype=np.float64,
        )
        specific = float(row.specific_variance_monthly)
        _require(
            np.isfinite(betas).all() and math.isfinite(specific) and specific > 0.0,
            "PRESENT monthly risk row contains invalid numerics",
        )
        betas.setflags(write=False)
        memory[int(row.permno)] = _RiskMemory(
            signal_date=signal_date,
            permco=int(row.permco),
            factor_betas=betas,
            specific_variance_monthly=specific,
            sector=(
                "UNKNOWN_SECTOR"
                if pd.isna(row.ff12_sector_category)
                else str(row.ff12_sector_category)
            ),
            country=(
                "UNKNOWN_COUNTRY"
                if pd.isna(row.domicile_country_category)
                else str(row.domicile_country_category)
            ),
        )


def _inherited_holdings(
    *,
    state: PortfolioState,
    identity: AuthenticatedMonthlyIdentity,
    audit_month: pd.DataFrame,
    memory: Mapping[int, _RiskMemory],
    session_dates: tuple[dt.date, ...],
) -> InheritedHoldings:
    nonzero = np.flatnonzero(state.position_value_dollars != 0.0)
    if nonzero.size == 0:
        return InheritedHoldings.empty(identity)
    by_permno = {int(row.permno): row for row in audit_month.itertuples(index=False)}
    records: list[dict[str, object]] = []
    for position in nonzero:
        permno = int(state.permno[position])
        _require(
            permno in memory, f"inherited holding lacks prior risk authority: {permno}"
        )
        remembered = memory[permno]
        current = by_permno.get(permno)
        current_long = bool(current.long_eligible) if current is not None else False
        current_short = bool(current.short_eligible) if current is not None else False
        weight = float(state.weights[position])
        required_exit = (weight > 0.0 and not current_long) or (
            weight < 0.0 and not current_short
        )
        age = sum(
            remembered.signal_date < value <= identity.signal_date
            for value in session_dates
        )
        records.append(
            {
                "permno": permno,
                "permco": (
                    int(current.permco) if current is not None else remembered.permco
                ),
                "previous_weight": weight,
                "last_factor_betas": remembered.factor_betas,
                "last_specific_variance_monthly": remembered.specific_variance_monthly,
                "risk_age_sessions": age,
                "sector": (
                    remembered.sector
                    if current is None or pd.isna(current.ff12_sector_category)
                    else str(current.ff12_sector_category)
                ),
                "country": (
                    remembered.country
                    if current is None or pd.isna(current.domicile_country_category)
                    else str(current.domicile_country_category)
                ),
                "current_long_eligible": current_long,
                "current_short_eligible": current_short,
                "required_exit": required_exit,
            }
        )
    records.sort(key=lambda value: int(value["permno"]))
    return InheritedHoldings(
        identity=identity,
        permno=np.asarray([value["permno"] for value in records], dtype=np.int64),
        permco=np.asarray([value["permco"] for value in records], dtype=np.int64),
        previous_weight=np.asarray(
            [value["previous_weight"] for value in records], dtype=np.float64
        ),
        last_factor_betas=np.vstack([value["last_factor_betas"] for value in records]),
        last_specific_variance_monthly=np.asarray(
            [value["last_specific_variance_monthly"] for value in records],
            dtype=np.float64,
        ),
        risk_age_sessions=np.asarray(
            [value["risk_age_sessions"] for value in records], dtype=np.int64
        ),
        sector=np.asarray([value["sector"] for value in records], dtype=str),
        country=np.asarray([value["country"] for value in records], dtype=str),
        current_long_eligible=np.asarray(
            [value["current_long_eligible"] for value in records], dtype=bool
        ),
        current_short_eligible=np.asarray(
            [value["current_short_eligible"] for value in records], dtype=bool
        ),
        required_exit=np.asarray(
            [value["required_exit"] for value in records], dtype=bool
        ),
    )


def _execution_prices(
    problem: MonthlyProblemBuild, market: MarketSnapshot
) -> np.ndarray:
    index = {int(value): position for position, value in enumerate(market.permno)}
    return np.asarray(
        [
            market.execution_price[index[int(value)]] if int(value) in index else np.nan
            for value in problem.problem.permno
        ],
        dtype=np.float64,
    )


def _one_way_turnover(target: np.ndarray, previous: np.ndarray) -> float:
    target_value = np.asarray(target, dtype=np.float64)
    previous_value = np.asarray(previous, dtype=np.float64)
    _require(
        target_value.shape == previous_value.shape
        and target_value.ndim == 1
        and np.isfinite(target_value).all()
        and np.isfinite(previous_value).all(),
        "turnover vectors differ or are nonfinite",
    )
    return 0.5 * float(np.sum(np.abs(target_value - previous_value)))


def _binding_trade_limit(binding_permnos: set[int], permno: int) -> bool:
    _require(
        type(permno) is int
        and permno > 0
        and all(type(value) is int and value > 0 for value in binding_permnos),
        "binding trade-limit identities are malformed",
    )
    return permno in binding_permnos


def _daily_inputs(
    *,
    problem: MonthlyProblemBuild,
    target_weights: np.ndarray,
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    return_dates: tuple[dt.date, ...],
    source_sha256: str,
) -> tuple[DatedMarketCloseInput, ...]:
    economic = {
        int(value)
        for value, weight in zip(problem.problem.permno, target_weights, strict=True)
        if float(weight) != 0.0
    }
    borrow = {
        int(value): float(rate)
        for value, rate in zip(
            problem.problem.permno,
            problem.problem.borrow_rate_annual,
            strict=True,
        )
    }
    _require(
        economic.issubset(borrow), "economic holding escaped monthly borrow authority"
    )
    factor_rows = factors.copy()
    factor_rows["observation_date"] = pd.to_datetime(
        factor_rows["observation_date"], errors="raise"
    ).dt.date
    _require(
        not factor_rows["observation_date"].duplicated().any(),
        "daily factor path is duplicated",
    )
    rf = {
        row.observation_date: float(row.rf)
        for row in factor_rows.itertuples(index=False)
    }
    daily = returns.copy()
    daily["calendar_date"] = pd.to_datetime(
        daily["calendar_date"], errors="raise"
    ).dt.date
    _require(
        "return_span_start_date" in daily.columns,
        "daily security return span authority is absent",
    )
    daily["return_span_start_date"] = pd.to_datetime(
        daily["return_span_start_date"], errors="raise"
    ).dt.date
    daily["permno"] = pd.to_numeric(daily["permno"], errors="raise").astype(np.int64)
    daily = daily.loc[daily["permno"].isin(economic)]
    _require(
        not daily.duplicated(["calendar_date", "permno"]).any(),
        "daily security return path is duplicated",
    )
    result: list[DatedMarketCloseInput] = []
    for date_value in return_dates:
        _require(
            date_value in rf and math.isfinite(rf[date_value]), "daily RF is absent"
        )
        day = daily.loc[daily["calendar_date"].eq(date_value)].sort_values(
            "permno", kind="mergesort"
        )
        values = pd.to_numeric(
            day["authoritative_total_return"], errors="coerce"
        ).to_numpy(float)
        _require(
            np.isfinite(values).all() and np.all(values >= -1.0),
            "daily authoritative return contains a missing or invalid value",
        )
        permnos = day["permno"].to_numpy(np.int64)
        span_starts = tuple(day["return_span_start_date"])
        _require(
            all(
                type(value) is dt.date and value < date_value
                for value in span_starts
            ),
            "daily observed return-span start is missing or invalid",
        )
        payload_sha = hashlib.sha256(
            _canonical_json(
                {
                    "source_sha256": source_sha256,
                    "calendar_date": date_value.isoformat(),
                    "permnos": [int(value) for value in permnos],
                    "returns": [float(value).hex() for value in values],
                    "return_span_start_dates": [
                        value.isoformat() for value in span_starts
                    ],
                }
            )
        ).hexdigest()
        result.append(
            DatedMarketCloseInput(
                trading_date=date_value,
                permno=permnos,
                total_return=values,
                borrow_rate_annual=np.asarray(
                    [borrow[int(value)] for value in permnos], dtype=np.float64
                ),
                daily_risk_free_rate=rf[date_value],
                source_payload_sha256=payload_sha,
                observed_return_span_start_date=span_starts,
            )
        )
    return tuple(result)


def _encode_monthly_problem_for_diagnostics(
    *,
    holding_month: str,
    strategy_id: str,
    problem: MonthlyProblemBuild,
    opening_state: PortfolioState,
) -> bytes:
    data = problem.problem
    universe = problem.universe
    selection = problem.selection
    arrays = {
        "allocation.permno": data.permno,
        "allocation.permco": data.permco,
        "allocation.side": data.side,
        "allocation.previous_weight": data.previous_weight,
        "allocation.reduction_only": data.reduction_only,
        "allocation.required_exit": data.required_exit,
        "allocation.expected_return": data.expected_return,
        "allocation.score": data.score,
        "allocation.factor_betas": data.factor_betas,
        "allocation.factor_covariance_monthly": (
            data.factor_covariance_monthly
        ),
        "allocation.specific_variance_monthly": (
            data.specific_variance_monthly
        ),
        "allocation.forecast_volatility": data.forecast_volatility,
        "allocation.full_spread": data.full_spread,
        "allocation.adv20_usd": data.adv20_usd,
        "allocation.impact_volatility_daily": (
            data.impact_volatility_daily
        ),
        "allocation.borrow_liquidity_quintile": (
            data.borrow_liquidity_quintile
        ),
        "allocation.trade_capacity_weight": data.trade_capacity_weight,
        "allocation.sector": data.sector,
        "allocation.country": data.country,
        "allocation.mandatory_corporate_action": (
            data.mandatory_corporate_action
        ),
        "allocation.risk_root": data.risk_root,
        "allocation.borrow_rate_annual": data.borrow_rate_annual,
        "universe.permno": universe.permno,
        "universe.permco": universe.permco,
        "universe.score": universe.score,
        "universe.long_eligible": universe.long_eligible,
        "universe.short_eligible": universe.short_eligible,
        "universe.previous_weight": universe.previous_weight,
        "universe.adv20_usd": universe.adv20_usd,
        "universe.sector": universe.sector,
        "universe.country": universe.country,
        "universe.required_exit": universe.required_exit,
        "universe.mandatory_corporate_action": (
            universe.mandatory_corporate_action
        ),
        "selection.long_candidate": selection.long_candidate,
        "selection.short_candidate": selection.short_candidate,
        "selection.selected": selection.selected,
        "selection.side": selection.side,
        "selection.reduction_only": selection.reduction_only,
        "selection.borrow_liquidity_quintile": (
            selection.borrow_liquidity_quintile
        ),
        "audit.long_entry_allowed": problem.long_entry_allowed,
        "audit.short_entry_allowed": problem.short_entry_allowed,
        "audit.blocked_new_trade_permno": problem.blocked_new_trade_permno,
        "audit.frozen_holding_permno": problem.frozen_holding_permno,
        "audit.carried_risk_permno": problem.carried_risk_permno,
        "audit.stale_p95_risk_permno": problem.stale_p95_risk_permno,
        "audit.required_exit_volatility_substituted_permno": (
            problem.required_exit_volatility_substituted_permno
        ),
    }
    metadata = {
        "schema_version": 1,
        "status": "active",
        "holding_month": holding_month,
        "strategy_id": strategy_id,
        "signal_date": problem.identity.signal_date.isoformat(),
        "source_bundle_sha256": problem.identity.source_bundle_sha256,
        "opening_portfolio_state_sha256": portfolio_state_sha256(
            opening_state
        ),
        "borrow_scenario": data.borrow_scenario,
        "nav_dollars": data.nav_dollars,
        "market_variance_monthly": data.market_variance_monthly,
        "selection_status": selection.status,
        "long_entry_boundary": selection.long_entry_boundary,
        "short_entry_boundary": selection.short_entry_boundary,
        "tail_size": selection.tail_size,
        "minimum_candidates": selection.minimum_candidates,
        "available_long_candidates": selection.available_long_candidates,
        "available_short_candidates": selection.available_short_candidates,
        "overlap_count": selection.overlap_count,
    }
    return encode_stage11_strategy_problem(
        problem_arrays=arrays,
        problem_metadata=metadata,
    )


def _encode_inactive_problem_receipt(
    *,
    holding_month: str,
    strategy_id: str,
    source_problem: MonthlyProblemBuild,
) -> bytes:
    return encode_stage11_strategy_problem(
        problem_arrays={
            "allocation.permno": np.empty(0, dtype=np.int64),
        },
        problem_metadata={
            "schema_version": 1,
            "status": "inactive",
            "inactive_reason": "both_nonlinear_families_failed",
            "holding_month": holding_month,
            "strategy_id": strategy_id,
            "signal_date": source_problem.identity.signal_date.isoformat(),
            "source_bundle_sha256": (
                source_problem.identity.source_bundle_sha256
            ),
        },
    )


def _diagnostic_problem_payloads(
    *,
    holding_month: str,
    problems: Mapping[str, MonthlyProblemBuild],
    states: Mapping[str, _StrategyState],
    qualifies: bool,
) -> Mapping[str, bytes]:
    payloads: dict[str, bytes] = {}
    for strategy_id in STAGE11_DIAGNOSTIC_STRATEGY_IDS:
        if strategy_id in problems:
            payloads[strategy_id] = _encode_monthly_problem_for_diagnostics(
                holding_month=holding_month,
                strategy_id=strategy_id,
                problem=problems[strategy_id],
                opening_state=states[strategy_id].portfolio,
            )
        else:
            _require(
                not qualifies
                and strategy_id
                in {
                    DIAGNOSTIC_PRIMARY_STRATEGY_ID,
                    DIAGNOSTIC_RANK_CHALLENGER_ID,
                },
                "active diagnostic strategy problem is absent",
            )
            payloads[strategy_id] = _encode_inactive_problem_receipt(
                holding_month=holding_month,
                strategy_id=strategy_id,
                source_problem=problems[
                    DIAGNOSTIC_ELASTIC_NET_CHALLENGER_ID
                ],
            )
    return MappingProxyType(payloads)


def _diagnostic_failure_report(
    *,
    holding_month: str,
    problem_bundle_sha256: str,
    failures: tuple[object, ...],
    active_strategy_ids: tuple[str, ...],
) -> Stage11AggregateFailureReport:
    failure_by_strategy = {
        str(getattr(failure, "strategy_id")): failure for failure in failures
    }
    results: list[StrategyDiagnosticResult] = []
    for strategy_id in STAGE11_DIAGNOSTIC_STRATEGY_IDS:
        if strategy_id not in active_strategy_ids:
            results.append(
                StrategyDiagnosticResult.inactive(
                    strategy_id,
                    inactive_reason="nonlinear_families_not_qualified",
                )
            )
            continue
        failure = failure_by_strategy.get(strategy_id)
        if failure is None:
            results.append(StrategyDiagnosticResult.passed(strategy_id))
            continue
        if hasattr(failure, "failure_code"):
            failure_code = str(getattr(failure, "failure_code"))
            error_type = str(getattr(failure, "exception_class"))
            private_detail = bytes(getattr(failure, "private_detail"))
        else:
            error_type = str(getattr(failure, "error_type"))
            architecture = str(getattr(failure, "allocator_architecture"))
            failure_code = f"allocation_solver.{error_type}".casefold()
            private_detail = _canonical_json(
                {
                    "strategy_id": strategy_id,
                    "allocator_architecture": architecture,
                    "error_type": error_type,
                    "error_message": str(getattr(failure, "error_message")),
                }
            )
        results.append(
            StrategyDiagnosticResult.failed(
                strategy_id,
                failure_code=failure_code,
                exception_class=error_type,
                private_detail=private_detail,
            )
        )
    return Stage11AggregateFailureReport(
        holding_month=holding_month,
        problem_bundle_sha256=problem_bundle_sha256,
        strategy_results=tuple(results),
    )


def _diagnostic_exception_evidence(
    *,
    strategy_ids: tuple[str, ...],
    phase: str,
    error: Exception,
) -> tuple[_DiagnosticExceptionEvidence, ...]:
    error_class = type(error).__name__
    code_token = "".join(
        character.casefold()
        if character.isascii() and (character.isalnum() or character == "_")
        else "_"
        for character in error_class
    ).strip("_")
    if not code_token:
        code_token = "exception"
    return tuple(
        _DiagnosticExceptionEvidence(
            strategy_id=strategy_id,
            failure_code=f"{phase}.{code_token}",
            exception_class=error_class,
            private_detail=_canonical_json(
                {
                    "strategy_id": strategy_id,
                    "phase": phase,
                    "exception_class": error_class,
                    "error_message": str(error),
                }
            ),
        )
        for strategy_id in sorted(strategy_ids)
    )


def _record_diagnostic_failure_status(
    *,
    diagnostic_session: Stage11DiagnosticSession,
    failure_report: Stage11AggregateFailureReport,
    month_index: int,
    holding_month: str,
) -> Stage11PortfolioDiagnosticStatus:
    failure_status: DiagnosticFailureStatus = (
        diagnostic_session.record_failures(report=failure_report)
    )
    opening_checkpoint = (
        diagnostic_session.load_latest_checkpoint().completed_checkpoint
    )
    return Stage11PortfolioDiagnosticStatus(
        status="diagnostic_month_failed",
        completed_month_count=month_index,
        next_holding_month=holding_month,
        latest_checkpoint_sha256=(
            None
            if opening_checkpoint is None
            else opening_checkpoint.checkpoint_sha256
        ),
        failure_report_sha256=failure_status.failure_report_sha256,
        failure_count=failure_status.failure_count,
    )


def _record_diagnostic_exception_status(
    *,
    diagnostic_session: Stage11DiagnosticSession,
    problem_bundle_sha256: str,
    active_strategy_ids: tuple[str, ...],
    month_index: int,
    holding_month: str,
    phase: str,
    error: Exception,
) -> Stage11PortfolioDiagnosticStatus:
    report = _diagnostic_failure_report(
        holding_month=holding_month,
        problem_bundle_sha256=problem_bundle_sha256,
        failures=_diagnostic_exception_evidence(
            strategy_ids=active_strategy_ids,
            phase=phase,
            error=error,
        ),
        active_strategy_ids=active_strategy_ids,
    )
    return _record_diagnostic_failure_status(
        diagnostic_session=diagnostic_session,
        failure_report=report,
        month_index=month_index,
        holding_month=holding_month,
    )


@dataclass(frozen=True, slots=True)
class _MaterializedStrategyMonth:
    weights_rows: tuple[dict[str, object], ...]
    trade_rows: tuple[dict[str, object], ...]
    daily_rows: tuple[dict[str, object], ...]
    monthly_row: dict[str, object]
    mark_continuity_evidence: tuple[EmpiricalMarkContinuityEvidence, ...]
    closing_state: _StrategyState


def _materialize_accounted_strategy_month(
    *,
    strategy_id: str,
    calendar_row: Any,
    allocation: Any,
    problem: MonthlyProblemBuild,
    accounting_input: StrategyAccountingInput,
    accounting: Any,
    daily_inputs: tuple[DatedMarketCloseInput, ...],
) -> _MaterializedStrategyMonth:
    result = allocation.result
    _require(result is not None, "accounted strategy allocation is absent")
    data = problem.problem
    pre_nav = accounting_input.opening_state.nav_dollars
    binding = set(result.binding_trade_limits)
    local_weights: list[dict[str, object]] = []
    local_trades: list[dict[str, object]] = []
    for position, permno in enumerate(data.permno):
        target = float(result.weights[position])
        previous = float(data.previous_weight[position])
        local_weights.append(
            {
                "strategy_id": strategy_id,
                "holding_month": calendar_row.holding_month,
                "signal_date": calendar_row.signal_date,
                "entry_date": calendar_row.entry_date,
                "exit_date": calendar_row.exit_date,
                "permno": int(permno),
                "previous_weight": previous,
                "weight": target,
                "score": float(data.score[position]),
                "expected_return": float(data.expected_return[position]),
                "side": int(data.side[position]),
                "required_exit": bool(data.required_exit[position]),
                "solver_status": result.solver_status,
                "solver_mode": result.mode,
            }
        )
        local_trades.append(
            {
                "strategy_id": strategy_id,
                "holding_month": calendar_row.holding_month,
                "signal_date": calendar_row.signal_date,
                "entry_date": calendar_row.entry_date,
                "exit_date": calendar_row.exit_date,
                "permno": int(permno),
                "previous_weight": previous,
                "target_weight": target,
                "trade_weight": target - previous,
                "trade_dollars": (target - previous) * pre_nav,
                "execution_price": float(
                    accounting_input.execution_price[position]
                ),
                "full_spread": float(data.full_spread[position]),
                "adv20_usd": float(data.adv20_usd[position]),
                "impact_volatility_daily": float(
                    data.impact_volatility_daily[position]
                ),
                "trade_capacity_weight": float(
                    data.trade_capacity_weight[position]
                ),
                "binding_trade_limit": _binding_trade_limit(
                    binding,
                    int(permno),
                ),
            }
        )

    local_daily: list[dict[str, object]] = []
    local_mark_evidence: list[EmpiricalMarkContinuityEvidence] = []
    previous_nav = accounting.pre_trade_nav_dollars
    for index, (receipt, day, closing_nav) in enumerate(
        zip(
            accounting.session_receipts,
            daily_inputs,
            accounting.daily_nav_dollars,
            strict=True,
        )
    ):
        local_mark_evidence.append(
            EmpiricalMarkContinuityEvidence(
                strategy_id=strategy_id,
                holding_month=calendar_row.holding_month,
                trading_date=receipt.trading_date,
                opening_ledger_sha256=receipt.opening_mark_continuity_sha256,
                closing_ledger_sha256=receipt.closing_mark_continuity_sha256,
                no_mark_permnos=receipt.no_mark_permnos,
                resolved_pending_permnos=receipt.resolved_pending_permnos,
                observed_zero_return_permnos=(
                    receipt.observed_zero_return_permnos
                ),
            )
        )
        daily_return = float(closing_nav / previous_nav - 1.0)
        local_daily.append(
            {
                "strategy_id": strategy_id,
                "holding_month": calendar_row.holding_month,
                "signal_date": calendar_row.signal_date,
                "entry_date": calendar_row.entry_date,
                "exit_date": calendar_row.exit_date,
                "date": receipt.trading_date,
                "opening_nav_dollars": previous_nav,
                "nav_dollars": float(closing_nav),
                "net_total_return": daily_return,
                "risk_free_return": day.daily_risk_free_rate,
                "net_excess_return": daily_return - day.daily_risk_free_rate,
                "transaction_cost_dollars": (
                    accounting.transaction_cost_dollars if index == 0 else 0.0
                ),
                "borrow_cost_dollars": receipt.borrow_cost_dollars,
                "financing_cost_dollars": receipt.financing_cost_dollars,
                "opening_state_sha256": receipt.opening_state_sha256,
                "closing_state_sha256": receipt.closing_state_sha256,
                "corporate_action_ids": ",".join(receipt.corporate_action_ids),
            }
        )
        previous_nav = float(closing_nav)

    closing = accounting.state
    monthly_row = {
        "strategy_id": strategy_id,
        "holding_month": calendar_row.holding_month,
        "signal_date": calendar_row.signal_date,
        "entry_date": calendar_row.entry_date,
        "exit_date": calendar_row.exit_date,
        "pre_trade_nav_dollars": accounting.pre_trade_nav_dollars,
        "closing_nav_dollars": closing.nav_dollars,
        "net_total_return": accounting.net_total_return,
        "risk_free_total_return": accounting.risk_free_total_return,
        "net_excess_return": accounting.net_excess_return,
        "transaction_cost_dollars": accounting.transaction_cost_dollars,
        "borrow_cost_dollars": accounting.borrow_cost_dollars,
        "financing_cost_dollars": accounting.financing_cost_dollars,
        "turnover": _one_way_turnover(
            result.weights,
            data.previous_weight,
        ),
        "gross_exposure": (
            closing.gross_exposure_dollars / closing.nav_dollars
        ),
        "net_exposure": float(
            np.sum(closing.position_value_dollars) / closing.nav_dollars
        ),
        "cash_weight": closing.cash_weight,
        "position_count": int(
            np.count_nonzero(closing.position_value_dollars)
        ),
        "solver_status": result.solver_status,
        "solver_mode": result.mode,
        "objective_value": result.objective_value,
    }
    return _MaterializedStrategyMonth(
        weights_rows=tuple(local_weights),
        trade_rows=tuple(local_trades),
        daily_rows=tuple(local_daily),
        monthly_row=monthly_row,
        mark_continuity_evidence=tuple(local_mark_evidence),
        closing_state=_StrategyState(
            portfolio=closing,
            action_ledger=accounting.action_ledger,
            mark_continuity_ledger=accounting.mark_continuity_ledger,
        ),
    )


def _run_empirical_portfolio_engine(
    *,
    execution_inputs: AuthenticatedStage11ExecutionInputs,
    outcome_publication: Stage11OutcomePublication,
    calendar_authority: AuthenticatedValidationCalendar,
    ledger: Stage11PseudoLiveLedger,
    selection: LockedValidationSelection,
    calibration_schedules: Mapping[str, StrategyCalibrationSchedule],
    progress: Callable[[str], None] | None = None,
    diagnostic_session: Stage11DiagnosticSession | None = None,
) -> EmpiricalPortfolioRun | Stage11PortfolioDiagnosticStatus:
    """Run three independent monthly states with exact net accounting."""

    report = progress if progress is not None else (lambda _message: None)
    execution_inputs.gate()
    _require(
        type(calendar_authority) is AuthenticatedValidationCalendar
        and type(ledger) is Stage11PseudoLiveLedger
        and ledger.complete
        and type(selection) is LockedValidationSelection,
        "portfolio execution authority, ledger, or selection differs",
    )
    _require(
        ledger.plan.calendar_rows == calendar_authority.rows,
        "portfolio ledger and calendar authority differ",
    )
    _require(
        tuple(outcome_publication.calendar_rows) == calendar_authority.rows,
        "outcome publication and calendar authority differ",
    )
    _require(
        set(calibration_schedules) == {PRIMARY_STRATEGY_ID, ELASTIC_NET_CHALLENGER_ID},
        "portfolio calibration schedule set differs",
    )
    # Recompute both validation selection and every calibration from their
    # authenticated ledgers.  The public dataclasses are deliberately not
    # treated as authority merely because a caller can construct them.
    canonical_selection = derive_locked_validation_selection(ledger)
    _require(
        canonical_selection.selection_sha256 == selection.selection_sha256,
        "portfolio selection differs from the completed walk-forward ledger",
    )
    canonical_development = build_authenticated_development_calibration_observations(
        execution_inputs,
        canonical_selection,
    )
    canonical_calibration = build_strategy_calibration_schedules(
        canonical_selection,
        ledger,
        development_oof=canonical_development,
    )
    _require(
        {name: value.schedule_sha256 for name, value in calibration_schedules.items()}
        == {
            name: value.schedule_sha256 for name, value in canonical_calibration.items()
        },
        "portfolio calibration differs from the authenticated OOF/outcome chronology",
    )
    selection = canonical_selection
    calibration_schedules = canonical_calibration
    risk_tables = _read_risk_cost_tables(execution_inputs)
    store = AuthenticatedOutcomeAccountingStore(execution_inputs, outcome_publication)
    session_dates = store.all_session_dates(calendar_authority)
    qualifies = selection.ensemble_selection.active_v1_ml_strategy_qualifies
    active_ids = [ELASTIC_NET_CHALLENGER_ID]
    if qualifies:
        active_ids.extend((PRIMARY_STRATEGY_ID, RANK_CHALLENGER_ID))
    states = {
        name: _StrategyState(
            portfolio=PortfolioState.all_cash(INITIAL_NAV_DOLLARS),
            action_ledger=CorporateActionLedger(),
            mark_continuity_ledger=MarkContinuityLedger.empty(),
        )
        for name in active_ids
    }
    risk_memory: dict[int, _RiskMemory] = {}
    retired_on: dict[int, dt.date] = {}
    start_month_index = 0
    if diagnostic_session is not None:
        resume = diagnostic_session.load_latest_checkpoint()
        start_month_index = resume.next_month_index
        if resume.completed_checkpoint is not None:
            _require(
                set(resume.state_metadata)
                == {
                    "completed_month",
                    "next_month_index",
                    "active_strategy_ids",
                    "qualifies",
                }
                and resume.state_metadata["completed_month"]
                == resume.completed_checkpoint.holding_month
                and resume.state_metadata["next_month_index"]
                == start_month_index
                and tuple(resume.state_metadata["active_strategy_ids"])
                == tuple(sorted(active_ids))
                and resume.state_metadata["qualifies"] is qualifies,
                "diagnostic checkpoint metadata differs from the locked run",
            )
            states, risk_memory, retired_on = (
                _restore_portfolio_diagnostic_state(resume.state_arrays)
            )
            _require(
                set(states) == set(active_ids),
                "diagnostic checkpoint strategy state differs",
            )
        if start_month_index == len(VALIDATION_HOLDING_MONTHS):
            checkpoint = resume.completed_checkpoint
            assert checkpoint is not None
            return Stage11PortfolioDiagnosticStatus(
                status="diagnostic_path_complete",
                completed_month_count=start_month_index,
                next_holding_month=None,
                latest_checkpoint_sha256=checkpoint.checkpoint_sha256,
                failure_report_sha256=None,
                failure_count=0,
            )
    session_rows: list[dict[str, object]] = []
    session_receipts: dict[str, Mapping[str, str]] = {}
    weights_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    mark_continuity_evidence: list[EmpiricalMarkContinuityEvidence] = []
    retired_identity_evidence: list[EmpiricalRetiredIdentityEvidence] = []
    pending_dust_retirement_evidence: list[
        EmpiricalPendingDustRetirementEvidence
    ] = []
    selected_scores = selection.selected_scores.copy()
    selected_scores["signal_date"] = pd.to_datetime(
        selected_scores["signal_date"], errors="raise"
    ).dt.date

    for month_index, row in enumerate(calendar_authority.rows):
        if month_index < start_month_index:
            continue
        report(f"Stage 11 portfolio month {month_index + 1}/59: {row.holding_month}")
        identity = _monthly_identity(
            execution_inputs, outcome_publication, selection, row.signal_date
        )
        audit_month, market, risk = _monthly_market_and_risk(
            row=row,
            identity=identity,
            risk_tables=risk_tables,
            store=store,
        )
        market, month_retired_evidence = _apply_retired_identity_execution_gate(
            market,
            retired_on=retired_on,
            holding_month=row.holding_month,
            signal_date=row.signal_date,
            entry_date=row.entry_date,
        )
        retired_identity_evidence.extend(month_retired_evidence)
        _update_risk_memory(risk_memory, audit_month, row.signal_date)
        enet_scored = _monthly_scored(
            selected_scores=selected_scores,
            audit_month=audit_month,
            identity=identity,
            signal_date=row.signal_date,
            score_column="elastic_net_score",
            kappa=calibration_schedules[ELASTIC_NET_CHALLENGER_ID].kappa_for_signal(
                row.signal_date
            ),
        )
        enet_inherited = _inherited_holdings(
            state=states[ELASTIC_NET_CHALLENGER_ID].portfolio,
            identity=identity,
            audit_month=audit_month,
            memory=risk_memory,
            session_dates=session_dates,
        )
        problems: dict[str, MonthlyProblemBuild] = {
            ELASTIC_NET_CHALLENGER_ID: build_monthly_problem(
                enet_scored,
                market,
                enet_inherited,
                risk,
                nav_dollars=states[ELASTIC_NET_CHALLENGER_ID].portfolio.nav_dollars,
            )
        }
        if qualifies:
            primary_scored = _monthly_scored(
                selected_scores=selected_scores,
                audit_month=audit_month,
                identity=identity,
                signal_date=row.signal_date,
                score_column="ensemble_score",
                kappa=calibration_schedules[PRIMARY_STRATEGY_ID].kappa_for_signal(
                    row.signal_date
                ),
            )
            for strategy_id in (PRIMARY_STRATEGY_ID, RANK_CHALLENGER_ID):
                inherited = _inherited_holdings(
                    state=states[strategy_id].portfolio,
                    identity=identity,
                    audit_month=audit_month,
                    memory=risk_memory,
                    session_dates=session_dates,
                )
                problems[strategy_id] = build_monthly_problem(
                    primary_scored,
                    market,
                    inherited,
                    risk,
                    nav_dollars=states[strategy_id].portfolio.nav_dollars,
                )
        if diagnostic_session is None:
            allocations = allocate_locked_monthly_strategies(
                primary_problem=problems.get(PRIMARY_STRATEGY_ID),
                rank_problem=problems.get(RANK_CHALLENGER_ID),
                elastic_net_problem=problems[ELASTIC_NET_CHALLENGER_ID],
                active_v1_ml_strategy_qualifies=qualifies,
            )
            problem_bundle = None
        else:
            problem_bundle = diagnostic_session.save_problem_bundle(
                holding_month=row.holding_month,
                strategy_problem_payloads=_diagnostic_problem_payloads(
                    holding_month=row.holding_month,
                    problems=problems,
                    states=states,
                    qualifies=qualifies,
                ),
            )
            diagnosed = diagnose_locked_monthly_strategies(
                primary_problem=problems.get(PRIMARY_STRATEGY_ID),
                rank_problem=problems.get(RANK_CHALLENGER_ID),
                elastic_net_problem=problems[ELASTIC_NET_CHALLENGER_ID],
                active_v1_ml_strategy_qualifies=qualifies,
            )
            if diagnosed.failures:
                failure_report = _diagnostic_failure_report(
                    holding_month=row.holding_month,
                    problem_bundle_sha256=problem_bundle.bundle_sha256,
                    failures=diagnosed.failures,
                    active_strategy_ids=tuple(active_ids),
                )
                return _record_diagnostic_failure_status(
                    diagnostic_session=diagnostic_session,
                    failure_report=failure_report,
                    month_index=month_index,
                    holding_month=row.holding_month,
                )
            assert diagnosed.allocations is not None
            allocations = diagnosed.allocations
        try:
            _require(
                not allocations[DECISION_FOCUSED_CHALLENGER_ID].active,
                "uncertified decision-focused strategy became active",
            )
            session_authority = (
                authenticate_materialized_holding_period_session_authority(
                    calendar_authority=calendar_authority,
                    calendar_row=row,
                    outcome_publication=outcome_publication,
                )
            )
        except Exception as error:
            if diagnostic_session is None:
                raise
            assert problem_bundle is not None
            return _record_diagnostic_exception_status(
                diagnostic_session=diagnostic_session,
                problem_bundle_sha256=problem_bundle.bundle_sha256,
                active_strategy_ids=tuple(active_ids),
                month_index=month_index,
                holding_month=row.holding_month,
                phase="session_authority",
                error=error,
            )
        session_rows.extend(
            {
                "holding_month": row.holding_month,
                "signal_date": row.signal_date,
                "entry_date": row.entry_date,
                "exit_date": row.exit_date,
                "date": trading_date,
            }
            for trading_date in session_authority.return_dates
        )
        _require(
            row.holding_month not in session_receipts,
            "holding-period session authority was duplicated",
        )
        session_receipts[row.holding_month] = {
            "source_payload_sha256": session_authority.source_payload_sha256,
            "authority_sha256": session_authority.authority_sha256,
        }
        try:
            returns_table = store.month("daily_total_returns", row.signal_date)
            factors_table = store.month("daily_factors", row.signal_date)
            action_table = store.month("delisting_inputs", row.signal_date)
            actions = map_delisting_rows_to_corporate_actions(
                action_table,
                authority=execution_inputs.corporate_action_authority,
            )
            return_main_sha, return_reserve_sha = store.paired_source_sha256(
                "daily_total_returns"
            )
            factor_main_sha, factor_reserve_sha = store.paired_source_sha256(
                "daily_factors"
            )
            return_source_sha = hashlib.sha256(
                _canonical_json(
                    {
                        "main": return_main_sha,
                        "reserve": return_reserve_sha,
                        "factor_main": factor_main_sha,
                        "factor_reserve": factor_reserve_sha,
                        "holding_month": row.holding_month,
                    }
                )
            ).hexdigest()
        except Exception as error:
            if diagnostic_session is None:
                raise
            assert problem_bundle is not None
            return _record_diagnostic_exception_status(
                diagnostic_session=diagnostic_session,
                problem_bundle_sha256=problem_bundle.bundle_sha256,
                active_strategy_ids=tuple(active_ids),
                month_index=month_index,
                holding_month=row.holding_month,
                phase="monthly_source_preparation",
                error=error,
            )
        accounting_inputs: dict[str, StrategyAccountingInput] = {}
        daily_by_strategy: dict[str, tuple[DatedMarketCloseInput, ...]] = {}
        preparation_failures: list[_DiagnosticExceptionEvidence] = []
        for strategy_id in active_ids:
            try:
                allocation = allocations[strategy_id]
                problem = problems[strategy_id]
                _require(
                    allocation.active and allocation.result is not None,
                    "active allocation is absent",
                )
                accounting_inputs[strategy_id] = StrategyAccountingInput(
                    strategy_id=strategy_id,
                    problem=problem,
                    opening_state=states[strategy_id].portfolio,
                    execution_price=_execution_prices(problem, market),
                    action_ledger=states[strategy_id].action_ledger,
                    mark_continuity_ledger=(
                        states[strategy_id].mark_continuity_ledger
                    ),
                )
                daily_by_strategy[strategy_id] = _daily_inputs(
                    problem=problem,
                    target_weights=allocation.result.weights,
                    returns=returns_table.to_pandas(),
                    factors=factors_table.to_pandas(),
                    return_dates=session_authority.return_dates,
                    source_sha256=return_source_sha,
                )
            except Exception as error:
                if diagnostic_session is None:
                    raise
                preparation_failures.extend(
                    _diagnostic_exception_evidence(
                        strategy_ids=(strategy_id,),
                        phase="strategy_accounting_input",
                        error=error,
                    )
                )
        if preparation_failures:
            assert diagnostic_session is not None
            assert problem_bundle is not None
            preparation_failure_report = _diagnostic_failure_report(
                holding_month=row.holding_month,
                problem_bundle_sha256=problem_bundle.bundle_sha256,
                failures=tuple(preparation_failures),
                active_strategy_ids=tuple(active_ids),
            )
            return _record_diagnostic_failure_status(
                diagnostic_session=diagnostic_session,
                failure_report=preparation_failure_report,
                month_index=month_index,
                holding_month=row.holding_month,
            )
        if diagnostic_session is None:
            accounted = run_locked_monthly_accounting(
                allocations=allocations,
                inputs=accounting_inputs,
                session_authority=session_authority,
                daily_inputs_by_strategy=daily_by_strategy,
                corporate_actions=actions,
            )
        else:
            assert problem_bundle is not None
            diagnosed_accounting = diagnose_locked_monthly_accounting(
                allocations=allocations,
                inputs=accounting_inputs,
                session_authority=session_authority,
                daily_inputs_by_strategy=daily_by_strategy,
                corporate_actions=actions,
            )
            if diagnosed_accounting.failures:
                accounting_failure_report = _diagnostic_failure_report(
                    holding_month=row.holding_month,
                    problem_bundle_sha256=problem_bundle.bundle_sha256,
                    failures=diagnosed_accounting.failures,
                    active_strategy_ids=tuple(active_ids),
                )
                return _record_diagnostic_failure_status(
                    diagnostic_session=diagnostic_session,
                    failure_report=accounting_failure_report,
                    month_index=month_index,
                    holding_month=row.holding_month,
                )
            assert diagnosed_accounting.accounting is not None
            accounted = diagnosed_accounting.accounting
        materialized_by_strategy: dict[str, _MaterializedStrategyMonth] = {}
        materialization_failures: list[_DiagnosticExceptionEvidence] = []
        for strategy_id in active_ids:
            try:
                materialized_by_strategy[strategy_id] = (
                    _materialize_accounted_strategy_month(
                        strategy_id=strategy_id,
                        calendar_row=row,
                        allocation=allocations[strategy_id],
                        problem=problems[strategy_id],
                        accounting_input=accounting_inputs[strategy_id],
                        accounting=accounted[strategy_id],
                        daily_inputs=daily_by_strategy[strategy_id],
                    )
                )
            except Exception as error:
                if diagnostic_session is None:
                    raise
                materialization_failures.extend(
                    _diagnostic_exception_evidence(
                        strategy_ids=(strategy_id,),
                        phase="post_accounting_materialization",
                        error=error,
                    )
                )
        if materialization_failures:
            assert diagnostic_session is not None
            assert problem_bundle is not None
            materialization_failure_report = _diagnostic_failure_report(
                holding_month=row.holding_month,
                problem_bundle_sha256=problem_bundle.bundle_sha256,
                failures=tuple(materialization_failures),
                active_strategy_ids=tuple(active_ids),
            )
            return _record_diagnostic_failure_status(
                diagnostic_session=diagnostic_session,
                failure_report=materialization_failure_report,
                month_index=month_index,
                holding_month=row.holding_month,
            )
        for strategy_id in active_ids:
            materialized = materialized_by_strategy[strategy_id]
            weights_rows.extend(materialized.weights_rows)
            trade_rows.extend(materialized.trade_rows)
            daily_rows.extend(materialized.daily_rows)
            monthly_rows.append(materialized.monthly_row)
            mark_continuity_evidence.extend(
                materialized.mark_continuity_evidence
            )
            states[strategy_id] = materialized.closing_state
        try:
            for action in actions:
                prior_retirement = retired_on.get(action.source_permno)
                _require(
                    prior_retirement is None
                    or prior_retirement == action.assigned_return_date,
                    "terminal source PERMNO was retired on conflicting dates",
                )
                retired_on[action.source_permno] = action.assigned_return_date
            execution_inputs.gate()
        except Exception as error:
            if diagnostic_session is None:
                raise
            assert problem_bundle is not None
            return _record_diagnostic_exception_status(
                diagnostic_session=diagnostic_session,
                problem_bundle_sha256=problem_bundle.bundle_sha256,
                active_strategy_ids=tuple(active_ids),
                month_index=month_index,
                holding_month=row.holding_month,
                phase="post_accounting_authority",
                error=error,
            )
        try:
            for strategy_id in active_ids:
                (
                    states[strategy_id],
                    month_dust_evidence,
                ) = _retire_pending_subcent_dust_at_month_boundary(
                    strategy_id=strategy_id,
                    holding_month=row.holding_month,
                    boundary_date=row.exit_date,
                    state=states[strategy_id],
                )
                pending_dust_retirement_evidence.extend(
                    month_dust_evidence
                )
        except Exception as error:
            if diagnostic_session is None:
                raise
            assert problem_bundle is not None
            return _record_diagnostic_exception_status(
                diagnostic_session=diagnostic_session,
                problem_bundle_sha256=problem_bundle.bundle_sha256,
                active_strategy_ids=tuple(active_ids),
                month_index=month_index,
                holding_month=row.holding_month,
                phase="post_month_pending_dust_retirement",
                error=error,
            )
        if diagnostic_session is not None:
            assert problem_bundle is not None
            if month_index + 1 == len(VALIDATION_HOLDING_MONTHS):
                try:
                    _require(
                        all(
                            not state.mark_continuity_ledger.pending_permnos
                            for state in states.values()
                        ),
                        "diagnostic completion retains a pending identity",
                    )
                except Exception as error:
                    return _record_diagnostic_exception_status(
                        diagnostic_session=diagnostic_session,
                        problem_bundle_sha256=problem_bundle.bundle_sha256,
                        active_strategy_ids=tuple(active_ids),
                        month_index=month_index,
                        holding_month=row.holding_month,
                        phase="final_mark_continuity",
                        error=error,
                    )
            checkpoint = diagnostic_session.save_checkpoint(
                completed_month=row.holding_month,
                next_month_index=month_index + 1,
                state_arrays=_serialize_portfolio_diagnostic_state(
                    states=states,
                    risk_memory=risk_memory,
                    retired_on=retired_on,
                ),
                state_metadata={
                    "completed_month": row.holding_month,
                    "next_month_index": month_index + 1,
                    "active_strategy_ids": list(sorted(active_ids)),
                    "qualifies": qualifies,
                },
                problem_bundle_sha256=problem_bundle.bundle_sha256,
                post_month_attestation={
                    "all_three_allocation_problems_saved_before_solve": True,
                    "all_active_strategies_solved_or_inactive_receipted": True,
                    "accounting_complete": True,
                    "corporate_actions_applied": True,
                    "mark_continuity_verified": True,
                    "authority_gate_passed": True,
                    "independent_checkpoint_reopen_passed": True,
                    "stage12_rows_read_is_zero": True,
                },
            )
            report(
                "Stage 11 diagnostic checkpoint committed: "
                f"{row.holding_month} ({month_index + 1}/59)"
            )

    if diagnostic_session is not None:
        completed = diagnostic_session.load_latest_checkpoint()
        _require(
            completed.next_month_index == len(VALIDATION_HOLDING_MONTHS)
            and completed.completed_checkpoint is not None,
            "diagnostic checkpoint chain did not reach all 59 months",
        )
        return Stage11PortfolioDiagnosticStatus(
            status="diagnostic_path_complete",
            completed_month_count=completed.next_month_index,
            next_holding_month=None,
            latest_checkpoint_sha256=(
                completed.completed_checkpoint.checkpoint_sha256
            ),
            failure_report_sha256=None,
            failure_count=0,
        )

    final_mark_continuity_ledger_sha256 = MappingProxyType(
        {
            strategy_id: states[strategy_id].mark_continuity_ledger.ledger_sha256
            for strategy_id in sorted(states)
        }
    )
    final_pending_permnos = MappingProxyType(
        {
            strategy_id: states[strategy_id].mark_continuity_ledger.pending_permnos
            for strategy_id in sorted(states)
        }
    )
    final_pending_nonzero_count = sum(
        len(values) for values in final_pending_permnos.values()
    )
    _require(
        final_pending_nonzero_count == 0,
        "empirical portfolio final publication handoff retains a nonzero "
        "pending identity",
    )
    ordered_mark_continuity_evidence = tuple(
        sorted(
            mark_continuity_evidence,
            key=lambda value: (
                value.strategy_id,
                value.trading_date,
                value.holding_month,
            ),
        )
    )
    ordered_retired_identity_evidence = tuple(
        sorted(
            retired_identity_evidence,
            key=lambda value: (
                value.entry_date,
                value.holding_month,
                value.permno,
            ),
        )
    )
    ordered_pending_dust_retirement_evidence = tuple(
        sorted(
            pending_dust_retirement_evidence,
            key=lambda value: (
                value.boundary_date,
                value.strategy_id,
                value.holding_month,
                value.permno,
            ),
        )
    )
    frames = {
        "execution_sessions": pd.DataFrame.from_records(session_rows),
        "portfolio_weights": pd.DataFrame.from_records(weights_rows),
        "portfolio_trades": pd.DataFrame.from_records(trade_rows),
        "daily_accounting": pd.DataFrame.from_records(daily_rows),
        "monthly_metrics": pd.DataFrame.from_records(monthly_rows),
    }
    _require(
        all(not value.empty for value in frames.values()),
        "empirical portfolio execution produced an empty output table",
    )
    _require(
        tuple(session_receipts) == VALIDATION_HOLDING_MONTHS,
        "empirical session receipt coverage differs",
    )
    _require(
        len(ordered_mark_continuity_evidence) == len(daily_rows),
        "empirical mark-continuity evidence coverage differs",
    )
    execution_inputs.gate()
    return EmpiricalPortfolioRun(
        execution_session_receipts=MappingProxyType(session_receipts),
        mark_continuity_evidence=ordered_mark_continuity_evidence,
        retired_identity_evidence=ordered_retired_identity_evidence,
        pending_dust_retirement_evidence=(
            ordered_pending_dust_retirement_evidence
        ),
        final_mark_continuity_ledger_sha256=(
            final_mark_continuity_ledger_sha256
        ),
        final_pending_permnos=final_pending_permnos,
        final_pending_nonzero_count=final_pending_nonzero_count,
        **frames,
    )


def run_empirical_portfolio_accounting(
    *,
    execution_inputs: AuthenticatedStage11ExecutionInputs,
    outcome_publication: Stage11OutcomePublication,
    calendar_authority: AuthenticatedValidationCalendar,
    ledger: Stage11PseudoLiveLedger,
    selection: LockedValidationSelection,
    calibration_schedules: Mapping[str, StrategyCalibrationSchedule],
    progress: Callable[[str], None] | None = None,
) -> EmpiricalPortfolioRun:
    """Run the clean final path from all cash without diagnostic state access."""

    result = _run_empirical_portfolio_engine(
        execution_inputs=execution_inputs,
        outcome_publication=outcome_publication,
        calendar_authority=calendar_authority,
        ledger=ledger,
        selection=selection,
        calibration_schedules=calibration_schedules,
        progress=progress,
        diagnostic_session=None,
    )
    _require(
        isinstance(result, EmpiricalPortfolioRun),
        "clean portfolio path returned diagnostic state",
    )
    return result


def run_empirical_portfolio_diagnostics(
    *,
    execution_inputs: AuthenticatedStage11ExecutionInputs,
    outcome_publication: Stage11OutcomePublication,
    calendar_authority: AuthenticatedValidationCalendar,
    ledger: Stage11PseudoLiveLedger,
    selection: LockedValidationSelection,
    calibration_schedules: Mapping[str, StrategyCalibrationSchedule],
    diagnostic_session: Stage11DiagnosticSession,
    progress: Callable[[str], None] | None = None,
) -> Stage11PortfolioDiagnosticStatus:
    """Run or resume the private, performance-blind 59-month diagnostic path."""

    _require(
        type(diagnostic_session) is Stage11DiagnosticSession,
        "diagnostic portfolio path requires its authenticated session",
    )
    result = _run_empirical_portfolio_engine(
        execution_inputs=execution_inputs,
        outcome_publication=outcome_publication,
        calendar_authority=calendar_authority,
        ledger=ledger,
        selection=selection,
        calibration_schedules=calibration_schedules,
        progress=progress,
        diagnostic_session=diagnostic_session,
    )
    _require(
        isinstance(result, Stage11PortfolioDiagnosticStatus),
        "diagnostic portfolio path returned final performance state",
    )
    return result


__all__ = [
    "AuthenticatedOutcomeAccountingStore",
    "EmpiricalMarkContinuityEvidence",
    "EmpiricalPendingDustRetirementEvidence",
    "EmpiricalPortfolioBackendError",
    "EmpiricalPortfolioRun",
    "EmpiricalRetiredIdentityEvidence",
    "INITIAL_NAV_DOLLARS",
    "PENDING_DUST_RETIREMENT_POLICY_ID",
    "PENDING_DUST_RETIREMENT_THRESHOLD_DOLLARS",
    "Stage11PortfolioDiagnosticStatus",
    "diagnostic_state_schema_sha256",
    "run_empirical_portfolio_accounting",
    "run_empirical_portfolio_diagnostics",
]
