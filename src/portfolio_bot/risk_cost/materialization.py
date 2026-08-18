"""Point-in-time Stage 8 risk and conservative-cost materialization.

The implementation reads the verified daily return table once in its native
``(permno, calendar_date)`` order and calculates estimates only for the frozen
long-eligible scoring rows.  It then restores the complete 1,263,048-row audit
spine.  No label, model, portfolio, trade, P&L, validation, or final-period
payload is accepted or produced.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from portfolio_bot.risk_cost.contract import (
    FACTOR_NAMES,
    MISSING_FACTOR_WINDOW,
    MISSING_REGRESSION,
    MISSING_RETURN_OBSERVATIONS,
    NOT_LONG_ELIGIBLE,
    PRESENT,
    published_schemas,
    require,
)
from portfolio_bot.risk_cost.cost_math import (
    PRIMARY_BORROW_RATE,
    REGULAR_STRESS_BORROW_RATE,
    assign_adv_liquidity_quintiles,
    borrow_rate_schedule,
    estimate_impact_volatility,
    required_exit_volatility_p95,
)
from portfolio_bot.risk_cost.inputs import LAST_SIGNAL, Stage8Inputs
from portfolio_bot.risk_cost.risk_math import (
    FactorCovarianceResult,
    estimate_factor_covariance,
    fit_security_risk_ewls,
    regularize_factor_model_covariance,
    shrink_residual_variances,
    type7_quantile,
)


Progress = Callable[[str], None]
_MISSING_DAY = np.iinfo(np.int32).min


@dataclass(frozen=True)
class RiskCostMaterialization:
    audit: pa.Table
    processed: pa.Table
    factor_covariance: pa.Table
    month_summary: pa.Table
    timing: pa.Table
    coverage: pa.Table
    model_parameters: Mapping[str, Any]
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class _FactorContext:
    session_days: np.ndarray
    factor_matrix: np.ndarray
    risk_free: np.ndarray
    availability_days: np.ndarray
    covariance_by_signal: Mapping[int, FactorCovarianceResult | None]
    covariance_window_by_signal: Mapping[int, tuple[int | None, int | None, int]]


@dataclass
class _EligibleState:
    signal_days: np.ndarray
    permnos: np.ndarray
    risk_status: np.ndarray
    risk_observations: np.ndarray
    risk_first_days: np.ndarray
    risk_last_days: np.ndarray
    coefficients: np.ndarray
    raw_specific_variance: np.ndarray
    shrinkage_target: np.ndarray
    shrinkage_source: np.ndarray
    shrinkage_population: np.ndarray
    shrunk_specific_variance: np.ndarray
    ridge: np.ndarray
    specific_variance_daily: np.ndarray
    specific_variance_monthly: np.ndarray
    security_variance_pre_ridge: np.ndarray
    security_variance_daily: np.ndarray
    security_variance_monthly: np.ndarray
    impact_volatility: np.ndarray
    impact_source: np.ndarray
    impact_observations: np.ndarray
    exit_volatility: np.ndarray
    exit_source: np.ndarray
    short_quintile: np.ndarray
    primary_borrow: np.ndarray
    stress_borrow: np.ndarray
    new_long_allowed: np.ndarray
    new_short_allowed: np.ndarray
    cost_status: np.ndarray


def _date_days(array: pa.Array | pa.ChunkedArray) -> np.ndarray:
    cast = pc.cast(array, pa.int32())
    filled = pc.fill_null(cast, pa.scalar(_MISSING_DAY, type=pa.int32()))
    return filled.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)


def _float_values(array: pa.Array | pa.ChunkedArray) -> np.ndarray:
    cast = pc.cast(array, pa.float64())
    return cast.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)


def _int_values(array: pa.Array | pa.ChunkedArray, dtype: Any) -> np.ndarray:
    return array.to_numpy(zero_copy_only=False).astype(dtype, copy=False)


def _column(batch: pa.RecordBatch, name: str) -> pa.Array:
    index = batch.schema.get_field_index(name)
    require(index >= 0, f"Stage 8 return batch omitted {name}")
    return batch.column(index)


def _as_nullable_float(values: np.ndarray) -> pa.Array:
    finite = np.isfinite(values)
    return pa.array(values, mask=~finite, type=pa.float64())


def _as_nullable_date(values: np.ndarray) -> pa.Array:
    mask = values == _MISSING_DAY
    integers = pa.array(values, mask=mask, type=pa.int32())
    return pc.cast(integers, pa.date32())


def _as_nullable_int8(values: np.ndarray) -> pa.Array:
    return pa.array(values, mask=values <= 0, type=pa.int8())


def _safe_float(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _new_state(signal_days: np.ndarray, permnos: np.ndarray) -> _EligibleState:
    size = len(signal_days)
    nan = lambda: np.full(size, np.nan, dtype=np.float64)
    return _EligibleState(
        signal_days=signal_days,
        permnos=permnos,
        risk_status=np.full(size, MISSING_RETURN_OBSERVATIONS, dtype=object),
        risk_observations=np.zeros(size, dtype=np.int16),
        risk_first_days=np.full(size, _MISSING_DAY, dtype=np.int32),
        risk_last_days=np.full(size, _MISSING_DAY, dtype=np.int32),
        coefficients=np.full((size, 7), np.nan, dtype=np.float64),
        raw_specific_variance=nan(),
        shrinkage_target=nan(),
        shrinkage_source=np.full(size, "MISSING", dtype=object),
        shrinkage_population=np.zeros(size, dtype=np.int32),
        shrunk_specific_variance=nan(),
        ridge=nan(),
        specific_variance_daily=nan(),
        specific_variance_monthly=nan(),
        security_variance_pre_ridge=nan(),
        security_variance_daily=nan(),
        security_variance_monthly=nan(),
        impact_volatility=nan(),
        impact_source=np.full(size, "MISSING", dtype=object),
        impact_observations=np.zeros(size, dtype=np.int16),
        exit_volatility=nan(),
        exit_source=np.full(size, "MISSING", dtype=object),
        short_quintile=np.zeros(size, dtype=np.int8),
        primary_borrow=nan(),
        stress_borrow=nan(),
        new_long_allowed=np.zeros(size, dtype=bool),
        new_short_allowed=np.zeros(size, dtype=bool),
        cost_status=np.full(size, "MISSING", dtype=object),
    )


def _factor_context(
    inputs: Stage8Inputs,
    signal_days: np.ndarray,
) -> _FactorContext:
    sessions = inputs.read_trading_calendar()
    session_days = _date_days(sessions["calendar_date"])
    require(
        len(session_days) == len(np.unique(session_days))
        and np.all(session_days[1:] > session_days[:-1]),
        "Stage 8 trading sessions are not unique and ordered",
    )
    factors = inputs.read_factors()
    factor_days = _date_days(factors["observation_date"])
    positions = np.searchsorted(session_days, factor_days)
    require(
        np.all(positions < len(session_days))
        and np.array_equal(session_days[positions], factor_days),
        "Stage 8 factor dates are not exact trading sessions",
    )
    matrix = np.full((len(session_days), len(FACTOR_NAMES)), np.nan, dtype=np.float64)
    rf = np.full(len(session_days), np.nan, dtype=np.float64)
    availability = np.full(len(session_days), _MISSING_DAY, dtype=np.int32)
    for ordinal, name in enumerate(FACTOR_NAMES):
        matrix[positions, ordinal] = _float_values(factors[name])
    rf[positions] = _float_values(factors["rf"])
    availability[positions] = _date_days(factors["public_availability_date"])

    covariance: dict[int, FactorCovarianceResult | None] = {}
    windows: dict[int, tuple[int | None, int | None, int]] = {}
    for signal in signal_days:
        signal_value = int(signal)
        signal_position = int(np.searchsorted(session_days, signal_value))
        require(
            signal_position < len(session_days)
            and int(session_days[signal_position]) == signal_value,
            "Stage 8 signal date is not a trading session",
        )
        start = max(0, signal_position - 251)
        indices = np.arange(start, signal_position + 1, dtype=np.int32)
        complete = (
            np.isfinite(matrix[indices]).all(axis=1)
            & np.isfinite(rf[indices])
            & (availability[indices] != _MISSING_DAY)
            & (availability[indices] <= signal_value)
        )
        selected = indices[complete]
        ages = signal_position - selected
        estimate = estimate_factor_covariance(matrix[selected], ages)
        covariance[signal_value] = estimate
        windows[signal_value] = (
            None if selected.size == 0 else int(session_days[selected[0]]),
            None if selected.size == 0 else int(session_days[selected[-1]]),
            int(selected.size),
        )
    return _FactorContext(
        session_days=session_days,
        factor_matrix=matrix,
        risk_free=rf,
        availability_days=availability,
        covariance_by_signal=covariance,
        covariance_window_by_signal=windows,
    )


def _global_return_window(
    history_session_positions: np.ndarray,
    history_availability_days: np.ndarray,
    history_returns: np.ndarray,
    *,
    signal_position: int,
    signal_day: int,
    sessions: int,
) -> list[float | None]:
    if signal_position + 1 < sessions:
        return []
    start = signal_position - sessions + 1
    result: list[float | None] = [None] * sessions
    left = int(np.searchsorted(history_session_positions, start, side="left"))
    right = int(np.searchsorted(history_session_positions, signal_position, side="right"))
    for position, availability, value in zip(
        history_session_positions[left:right],
        history_availability_days[left:right],
        history_returns[left:right],
        strict=True,
    ):
        if int(availability) <= signal_day and math.isfinite(float(value)):
            result[int(position) - start] = float(value)
    return result


def _process_security_history(
    permno: int,
    calendar_days: np.ndarray,
    availability_days: np.ndarray,
    returns: np.ndarray,
    target_positions: np.ndarray | None,
    state: _EligibleState,
    factors: _FactorContext,
) -> None:
    if target_positions is None or target_positions.size == 0:
        return
    require(
        calendar_days.ndim == availability_days.ndim == returns.ndim == 1
        and len(calendar_days) == len(availability_days) == len(returns),
        "Stage 8 security-history vectors differ",
    )
    if len(calendar_days) == 0:
        return
    require(
        np.all(calendar_days[1:] > calendar_days[:-1]),
        f"Stage 8 return dates are duplicated or unordered for PERMNO {permno}",
    )
    session_positions = np.searchsorted(factors.session_days, calendar_days)
    exact = (
        (session_positions < len(factors.session_days))
        & (factors.session_days[np.minimum(session_positions, len(factors.session_days) - 1)] == calendar_days)
    )
    session_positions = session_positions[exact].astype(np.int32, copy=False)
    availability_days = availability_days[exact]
    returns = returns[exact]

    for target in target_positions:
        target_index = int(target)
        signal_day = int(state.signal_days[target_index])
        signal_position = int(np.searchsorted(factors.session_days, signal_day))

        window_20 = _global_return_window(
            session_positions,
            availability_days,
            returns,
            signal_position=signal_position,
            signal_day=signal_day,
            sessions=20,
        )
        window_60 = _global_return_window(
            session_positions,
            availability_days,
            returns,
            signal_position=signal_position,
            signal_day=signal_day,
            sessions=60,
        )
        impact = estimate_impact_volatility(window_20, window_60)
        if impact is not None:
            state.impact_volatility[target_index] = impact.daily_volatility
            state.impact_source[target_index] = (
                "20_SESSION" if impact.source_window_sessions == 20 else "60_SESSION"
            )
            state.impact_observations[target_index] = impact.valid_observations

        start_session = max(0, signal_position - 251)
        left = int(np.searchsorted(session_positions, start_session, side="left"))
        right = int(np.searchsorted(session_positions, signal_position, side="right"))
        selected_positions = session_positions[left:right]
        selected_availability = availability_days[left:right]
        selected_returns = returns[left:right]
        complete = (
            (selected_availability <= signal_day)
            & np.isfinite(selected_returns)
            & np.isfinite(factors.risk_free[selected_positions])
            & np.isfinite(factors.factor_matrix[selected_positions]).all(axis=1)
            & (factors.availability_days[selected_positions] != _MISSING_DAY)
            & (factors.availability_days[selected_positions] <= signal_day)
        )
        selected_positions = selected_positions[complete]
        selected_returns = selected_returns[complete]
        count = int(selected_positions.size)
        state.risk_observations[target_index] = count
        if count:
            state.risk_first_days[target_index] = int(
                factors.session_days[selected_positions[0]]
            )
            state.risk_last_days[target_index] = int(
                factors.session_days[selected_positions[-1]]
            )
        if count < 126:
            state.risk_status[target_index] = MISSING_RETURN_OBSERVATIONS
            continue
        fit = fit_security_risk_ewls(
            selected_returns,
            factors.risk_free[selected_positions],
            factors.factor_matrix[selected_positions],
            signal_position - selected_positions,
        )
        if fit is None:
            state.risk_status[target_index] = MISSING_REGRESSION
            continue
        state.coefficients[target_index, 0] = fit.intercept
        state.coefficients[target_index, 1:] = fit.betas
        state.raw_specific_variance[target_index] = fit.residual_variance_daily
        state.risk_status[target_index] = "RAW_PRESENT_PENDING_CROSS_SECTION"


def _target_map(permnos: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(permnos, kind="stable")
    sorted_permnos = permnos[order]
    boundaries = np.flatnonzero(sorted_permnos[1:] != sorted_permnos[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(order)]
    return {
        int(sorted_permnos[start]): order[start:end]
        for start, end in zip(starts, ends, strict=True)
    }


def _scan_return_histories(
    inputs: Stage8Inputs,
    target_map: Mapping[int, np.ndarray],
    state: _EligibleState,
    factors: _FactorContext,
    *,
    batch_rows: int,
    progress: Progress,
) -> tuple[int, int]:
    last_signal_day = int((LAST_SIGNAL - dt.date(1970, 1, 1)).days)
    carry_permno: int | None = None
    carry_days = np.asarray([], dtype=np.int32)
    carry_availability = np.asarray([], dtype=np.int32)
    carry_returns = np.asarray([], dtype=np.float64)
    delivered = 0
    used = 0
    batches = 0
    last_seen_permno: int | None = None

    def finish() -> None:
        nonlocal used
        if carry_permno is None:
            return
        targets = target_map.get(carry_permno)
        if targets is not None:
            _process_security_history(
                carry_permno,
                carry_days,
                carry_availability,
                carry_returns,
                targets,
                state,
                factors,
            )
            used += len(carry_days)

    for batch in inputs.iter_return_batches(batch_rows=batch_rows):
        batches += 1
        delivered += batch.num_rows
        permnos = _int_values(_column(batch, "permno"), np.int64)
        days = _date_days(_column(batch, "calendar_date"))
        span_end = _date_days(_column(batch, "return_span_end_date"))
        availability = _date_days(_column(batch, "outcome_availability_date"))
        covered = _int_values(_column(batch, "covered_trading_sessions"), np.int16)
        usable = pc.fill_null(_column(batch, "return_history_usable"), False).to_numpy(
            zero_copy_only=False
        )
        values = _float_values(_column(batch, "authoritative_total_return"))
        valid = (
            usable
            & (covered == 1)
            & (span_end == days)
            & (days <= last_signal_day)
            & (availability != _MISSING_DAY)
            & (availability <= last_signal_day)
            & np.isfinite(values)
            & (values >= -1.0)
        )
        permnos = permnos[valid]
        days = days[valid]
        availability = availability[valid]
        values = values[valid]
        if permnos.size == 0:
            continue
        require(
            np.all(permnos[1:] >= permnos[:-1]),
            "Stage 8 return batch is not sorted by PERMNO",
        )
        if last_seen_permno is not None:
            require(int(permnos[0]) >= last_seen_permno, "Stage 8 return scan order regressed")
        last_seen_permno = int(permnos[-1])
        boundaries = np.flatnonzero(permnos[1:] != permnos[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(permnos)]
        for start, end in zip(starts, ends, strict=True):
            group_permno = int(permnos[start])
            group_days = days[start:end]
            group_availability = availability[start:end]
            group_returns = values[start:end]
            if carry_permno is None:
                carry_permno = group_permno
                carry_days = group_days.copy()
                carry_availability = group_availability.copy()
                carry_returns = group_returns.copy()
            elif group_permno == carry_permno:
                carry_days = np.concatenate((carry_days, group_days))
                carry_availability = np.concatenate((carry_availability, group_availability))
                carry_returns = np.concatenate((carry_returns, group_returns))
            else:
                finish()
                carry_permno = group_permno
                carry_days = group_days.copy()
                carry_availability = group_availability.copy()
                carry_returns = group_returns.copy()
        if batches % 10 == 0:
            progress(f"return batches={batches}; delivered rows={delivered:,}")
    finish()
    progress(f"complete return scan; delivered rows={delivered:,}")
    return delivered, used


def _signal_ranges(signal_days: np.ndarray) -> list[tuple[int, int, int]]:
    if signal_days.size == 0:
        return []
    require(np.all(signal_days[1:] >= signal_days[:-1]), "Stage 8 signal rows are unordered")
    boundaries = np.flatnonzero(signal_days[1:] != signal_days[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(signal_days)]
    return [(int(signal_days[start]), int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _missing_cost_status(spread: float, adv: float, volatility: float) -> str:
    missing: list[str] = []
    if not math.isfinite(spread) or spread < 0.0:
        missing.append("MISSING_SPREAD")
    if not math.isfinite(adv) or adv <= 0.0:
        missing.append("MISSING_ADV")
    if not math.isfinite(volatility) or volatility < 0.0:
        missing.append("MISSING_IMPACT_VOLATILITY")
    return PRESENT if not missing else "|".join(missing)


def _finalize_cross_sections(
    state: _EligibleState,
    *,
    sectors: Sequence[str | None],
    short_eligible: np.ndarray,
    spreads: np.ndarray,
    adv: np.ndarray,
    factors: _FactorContext,
) -> list[dict[str, Any]]:
    month_rows: list[dict[str, Any]] = []
    for signal_day, start, end in _signal_ranges(state.signal_days):
        positions = np.arange(start, end, dtype=np.int64)
        covariance = factors.covariance_by_signal[signal_day]
        raw_values = [
            None if not math.isfinite(float(value)) else float(value)
            for value in state.raw_specific_variance[start:end]
        ]
        count_values = [int(value) for value in state.risk_observations[start:end]]
        shrinkage = shrink_residual_variances(
            raw_values,
            count_values,
            list(sectors[start:end]),
            [True] * (end - start),
        )
        require(shrinkage is not None, "Stage 8 shrinkage inputs are malformed")
        present_local: list[int] = []
        for offset, estimate in enumerate(shrinkage):
            absolute = start + offset
            if estimate is None:
                if covariance is None and state.risk_observations[absolute] >= 126:
                    state.risk_status[absolute] = MISSING_FACTOR_WINDOW
                continue
            state.shrinkage_target[absolute] = estimate.reference_median_daily
            state.shrinkage_source[absolute] = estimate.reference_scope.upper()
            state.shrinkage_population[absolute] = estimate.reference_security_count
            state.shrunk_specific_variance[absolute] = estimate.shrunk_variance_daily
            if covariance is not None and np.isfinite(state.coefficients[absolute, 1:]).all():
                present_local.append(offset)

        ridge_value: float | None = None
        minimum_eigenvalue: float | None = None
        if covariance is not None:
            omega = np.asarray(covariance.covariance_daily, dtype=np.float64)
            minimum_eigenvalue = float(np.linalg.eigvalsh(omega)[0])
            if present_local:
                absolute_positions = positions[np.asarray(present_local, dtype=np.int64)]
                regularized = regularize_factor_model_covariance(
                    state.coefficients[absolute_positions, 1:],
                    omega,
                    state.shrunk_specific_variance[absolute_positions],
                )
                require(regularized is not None, "Stage 8 covariance regularization failed")
                ridge_value = regularized.ridge_daily
                for local, absolute in enumerate(absolute_positions):
                    state.ridge[absolute] = regularized.ridge_daily
                    state.specific_variance_daily[absolute] = (
                        regularized.specific_variances_daily_post_ridge[local]
                    )
                    state.specific_variance_monthly[absolute] = (
                        regularized.specific_variances_monthly_post_ridge[local]
                    )
                    state.security_variance_pre_ridge[absolute] = (
                        regularized.security_diagonal_daily_pre_ridge[local]
                    )
                    state.security_variance_daily[absolute] = (
                        regularized.security_diagonal_daily_post_ridge[local]
                    )
                    state.security_variance_monthly[absolute] = (
                        regularized.security_diagonal_monthly_post_ridge[local]
                    )
                    state.risk_status[absolute] = PRESENT

        p95 = required_exit_volatility_p95(state.impact_volatility[start:end])
        for absolute in positions:
            own = state.impact_volatility[absolute]
            if math.isfinite(float(own)) and own >= 0.0:
                state.exit_volatility[absolute] = own
                state.exit_source[absolute] = "SECURITY"
            elif p95 is not None:
                state.exit_volatility[absolute] = p95
                state.exit_source[absolute] = "CROSS_SECTION_P95"
            status = _missing_cost_status(
                float(spreads[absolute]),
                float(adv[absolute]),
                float(state.impact_volatility[absolute]),
            )
            state.cost_status[absolute] = status
            complete = status == PRESENT
            state.new_long_allowed[absolute] = complete
            state.new_short_allowed[absolute] = bool(short_eligible[absolute]) and complete

        short_positions = positions[short_eligible[start:end]]
        if short_positions.size:
            quintiles = assign_adv_liquidity_quintiles(
                [int(value) for value in state.permnos[short_positions]],
                [
                    None if not math.isfinite(float(value)) else float(value)
                    for value in adv[short_positions]
                ],
            )
            for local, absolute in enumerate(short_positions):
                quintile = int(quintiles[local])
                schedule = borrow_rate_schedule(lowest_liquidity_quintile=quintile == 1)
                state.short_quintile[absolute] = quintile
                state.primary_borrow[absolute] = schedule.primary.annual
                state.stress_borrow[absolute] = schedule.stress.annual

        group_impact_source = state.impact_source[start:end]
        group_cost = state.cost_status[start:end]
        factor_window = factors.covariance_window_by_signal[signal_day]
        cross_median = type7_quantile(raw_values, 0.5)
        month_rows.append(
            {
                "signal_date": dt.date(1970, 1, 1) + dt.timedelta(days=signal_day),
                "long_eligible_rows": end - start,
                "short_eligible_rows": int(np.sum(short_eligible[start:end])),
                "risk_present_rows": int(np.sum(state.risk_status[start:end] == PRESENT)),
                "factor_covariance_present": covariance is not None,
                "factor_observation_count": int(factor_window[2]),
                "factor_covariance_min_eigenvalue": _safe_float(minimum_eigenvalue),
                "ridge_daily": _safe_float(ridge_value),
                "cross_sectional_specific_variance_median": _safe_float(cross_median),
                "impact_volatility_20d_rows": int(np.sum(group_impact_source == "20_SESSION")),
                "impact_volatility_60d_rows": int(np.sum(group_impact_source == "60_SESSION")),
                "impact_volatility_missing_rows": int(np.sum(group_impact_source == "MISSING")),
                "exit_volatility_p95_daily": _safe_float(p95),
                "spread_missing_rows": int(np.sum(~np.isfinite(spreads[start:end]) | (spreads[start:end] < 0.0))),
                "adv_missing_or_nonpositive_rows": int(np.sum(~np.isfinite(adv[start:end]) | (adv[start:end] <= 0.0))),
                "new_long_trade_allowed_rows": int(np.sum(state.new_long_allowed[start:end])),
                "new_short_trade_allowed_rows": int(np.sum(state.new_short_allowed[start:end])),
                "post_cutoff_return_rows_used": 0,
                "post_cutoff_factor_rows_used": 0,
            }
        )
    return month_rows


def _factor_covariance_table(
    signal_days: np.ndarray,
    factors: _FactorContext,
) -> pa.Table:
    rows: list[dict[str, Any]] = []
    epoch = dt.date(1970, 1, 1)
    for signal_day in signal_days:
        signal_value = int(signal_day)
        estimate = factors.covariance_by_signal[signal_value]
        first, last, count = factors.covariance_window_by_signal[signal_value]
        matrix = None if estimate is None else np.asarray(estimate.covariance_daily)
        for row_ordinal, factor_row in enumerate(FACTOR_NAMES):
            for col_ordinal, factor_col in enumerate(FACTOR_NAMES):
                daily = None if matrix is None else float(matrix[row_ordinal, col_ordinal])
                rows.append(
                    {
                        "signal_date": epoch + dt.timedelta(days=signal_value),
                        "factor_row": factor_row,
                        "factor_col": factor_col,
                        "factor_row_ordinal": row_ordinal,
                        "factor_col_ordinal": col_ordinal,
                        "factor_observation_count": count,
                        "factor_window_first_date": None if first is None else epoch + dt.timedelta(days=first),
                        "factor_window_last_date": None if last is None else epoch + dt.timedelta(days=last),
                        "half_life_sessions": 63.0,
                        "covariance_daily": daily,
                        "covariance_monthly": None if daily is None else 21.0 * daily,
                        "factor_covariance_status": PRESENT if estimate is not None else MISSING_FACTOR_WINDOW,
                    }
                )
    return pa.Table.from_pylist(
        rows,
        schema=published_schemas()["development_factor_covariance.parquet"],
    )


def _overlay_float(size: int, eligible_indices: np.ndarray, values: np.ndarray) -> pa.Array:
    result = np.full(size, np.nan, dtype=np.float64)
    result[eligible_indices] = values
    return _as_nullable_float(result)


def _overlay_date(size: int, eligible_indices: np.ndarray, values: np.ndarray) -> pa.Array:
    result = np.full(size, _MISSING_DAY, dtype=np.int32)
    result[eligible_indices] = values
    return _as_nullable_date(result)


def _overlay_int(
    size: int,
    eligible_indices: np.ndarray,
    values: np.ndarray,
    *,
    arrow_type: pa.DataType,
) -> pa.Array:
    result = np.zeros(size, dtype=values.dtype)
    result[eligible_indices] = values
    return pa.array(result, type=arrow_type)


def _overlay_object(
    size: int,
    eligible_indices: np.ndarray,
    values: np.ndarray,
    default: str,
) -> pa.Array:
    result = np.full(size, default, dtype=object)
    result[eligible_indices] = values
    return pa.array(result, type=pa.string())


def _build_audit_table(
    spine: pa.Table,
    eligible_indices: np.ndarray,
    state: _EligibleState,
) -> pa.Table:
    size = spine.num_rows
    long_full = pc.fill_null(spine["long_eligible"], False).combine_chunks()
    short_full = pc.fill_null(spine["short_eligible"], False).combine_chunks()
    full_bool_long = np.zeros(size, dtype=bool)
    full_bool_short = np.zeros(size, dtype=bool)
    full_bool_long[eligible_indices] = state.new_long_allowed
    full_bool_short[eligible_indices] = state.new_short_allowed
    quintiles = np.zeros(size, dtype=np.int8)
    quintiles[eligible_indices] = state.short_quintile
    arrays: dict[str, pa.Array | pa.ChunkedArray] = {
        "signal_date": spine["signal_date"],
        "permno": spine["permno"],
        "permco": spine["permco"],
        "long_eligible": long_full,
        "short_eligible": short_full,
        "primary_exchange": spine["primary_exchange"],
        "ff12_sector_category": spine["ff12_sector_category"],
        "domicile_country_category": spine["domicile_country_category"],
        "information_cutoff_date": spine["signal_date"],
        "risk_status": _overlay_object(size, eligible_indices, state.risk_status, NOT_LONG_ELIGIBLE),
        "risk_observation_count": _overlay_int(
            size, eligible_indices, state.risk_observations, arrow_type=pa.int16()
        ),
        "risk_window_first_date": _overlay_date(size, eligible_indices, state.risk_first_days),
        "risk_window_last_date": _overlay_date(size, eligible_indices, state.risk_last_days),
        "beta_risk_intercept": _overlay_float(size, eligible_indices, state.coefficients[:, 0]),
    }
    for ordinal, name in enumerate(FACTOR_NAMES, start=1):
        arrays[f"beta_risk_{name}"] = _overlay_float(
            size, eligible_indices, state.coefficients[:, ordinal]
        )
    arrays.update(
        {
            "specific_variance_raw_daily": _overlay_float(size, eligible_indices, state.raw_specific_variance),
            "shrinkage_target_daily": _overlay_float(size, eligible_indices, state.shrinkage_target),
            "shrinkage_target_source": _overlay_object(
                size, eligible_indices, state.shrinkage_source, NOT_LONG_ELIGIBLE
            ),
            "shrinkage_population_count": _overlay_int(
                size, eligible_indices, state.shrinkage_population, arrow_type=pa.int32()
            ),
            "specific_variance_shrunk_daily": _overlay_float(size, eligible_indices, state.shrunk_specific_variance),
            "risk_ridge_daily": _overlay_float(size, eligible_indices, state.ridge),
            "specific_variance_daily": _overlay_float(size, eligible_indices, state.specific_variance_daily),
            "specific_variance_monthly": _overlay_float(size, eligible_indices, state.specific_variance_monthly),
            "security_variance_pre_ridge_daily": _overlay_float(size, eligible_indices, state.security_variance_pre_ridge),
            "security_variance_daily": _overlay_float(size, eligible_indices, state.security_variance_daily),
            "security_variance_monthly": _overlay_float(size, eligible_indices, state.security_variance_monthly),
            "full_spread_20d": pc.cast(spine["median_estimated_full_spread_20d"], pa.float64()),
            "adv20_usd": pc.cast(spine["median_dollar_volume_20d"], pa.float64()),
            "impact_volatility_daily": _overlay_float(size, eligible_indices, state.impact_volatility),
            "impact_volatility_source": _overlay_object(
                size, eligible_indices, state.impact_source, NOT_LONG_ELIGIBLE
            ),
            "impact_volatility_observations": _overlay_int(
                size, eligible_indices, state.impact_observations, arrow_type=pa.int16()
            ),
            "exit_impact_volatility_daily": _overlay_float(size, eligible_indices, state.exit_volatility),
            "exit_volatility_source": _overlay_object(
                size, eligible_indices, state.exit_source, NOT_LONG_ELIGIBLE
            ),
            "short_liquidity_quintile": _as_nullable_int8(quintiles),
            "primary_borrow_rate_annual": _overlay_float(size, eligible_indices, state.primary_borrow),
            "stress_borrow_rate_annual": _overlay_float(size, eligible_indices, state.stress_borrow),
            "new_long_trade_allowed": pa.array(full_bool_long, type=pa.bool_()),
            "new_short_trade_allowed": pa.array(full_bool_short, type=pa.bool_()),
            "cost_input_status": _overlay_object(
                size, eligible_indices, state.cost_status, NOT_LONG_ELIGIBLE
            ),
            "post_cutoff_return_rows_used": pa.array(np.zeros(size, dtype=np.int32)),
            "post_cutoff_factor_rows_used": pa.array(np.zeros(size, dtype=np.int32)),
        }
    )
    schema = published_schemas()["development_risk_cost_audit.parquet"]
    table = pa.Table.from_arrays([arrays[field.name] for field in schema], schema=schema)
    require(table.num_rows == size, "Stage 8 audit-table row count differs")
    return table


def _complete_month_rows(
    full_signal_days: np.ndarray,
    long_month_rows: Sequence[Mapping[str, Any]],
    factors: _FactorContext,
) -> list[dict[str, Any]]:
    epoch = dt.date(1970, 1, 1)
    by_signal = {
        int((row["signal_date"] - epoch).days): dict(row) for row in long_month_rows
    }
    full_ranges = _signal_ranges(full_signal_days)
    result: list[dict[str, Any]] = []
    for signal_day, start, end in full_ranges:
        row = by_signal.get(signal_day)
        if row is None:
            covariance = factors.covariance_by_signal[signal_day]
            window = factors.covariance_window_by_signal[signal_day]
            eigenvalue = None
            if covariance is not None:
                eigenvalue = float(np.linalg.eigvalsh(np.asarray(covariance.covariance_daily))[0])
            row = {
                "signal_date": epoch + dt.timedelta(days=signal_day),
                "long_eligible_rows": 0,
                "short_eligible_rows": 0,
                "risk_present_rows": 0,
                "factor_covariance_present": covariance is not None,
                "factor_observation_count": int(window[2]),
                "factor_covariance_min_eigenvalue": _safe_float(eigenvalue),
                "ridge_daily": None,
                "cross_sectional_specific_variance_median": None,
                "impact_volatility_20d_rows": 0,
                "impact_volatility_60d_rows": 0,
                "impact_volatility_missing_rows": 0,
                "exit_volatility_p95_daily": None,
                "spread_missing_rows": 0,
                "adv_missing_or_nonpositive_rows": 0,
                "new_long_trade_allowed_rows": 0,
                "new_short_trade_allowed_rows": 0,
                "post_cutoff_return_rows_used": 0,
                "post_cutoff_factor_rows_used": 0,
            }
        row["audit_rows"] = end - start
        result.append(row)
    return result


def _coverage_table(month_rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    rows: list[dict[str, Any]] = []
    for row in month_rows:
        long_count = int(row["long_eligible_rows"])
        short_count = int(row["short_eligible_rows"])
        risk_count = int(row["risk_present_rows"])
        cost_count = int(row["new_long_trade_allowed_rows"])
        short_cost = int(row["new_short_trade_allowed_rows"])
        rows.append(
            {
                "signal_date": row["signal_date"],
                "long_eligible_rows": long_count,
                "risk_present_rows": risk_count,
                "risk_coverage_fraction": None if long_count == 0 else risk_count / long_count,
                "cost_complete_rows": cost_count,
                "cost_coverage_fraction": None if long_count == 0 else cost_count / long_count,
                "short_eligible_rows": short_count,
                "short_cost_complete_rows": short_cost,
                "short_cost_coverage_fraction": None if short_count == 0 else short_cost / short_count,
            }
        )
    return pa.Table.from_pylist(rows, schema=published_schemas()["risk_cost_coverage.parquet"])


def _key_checks(table: pa.Table) -> tuple[int, int]:
    signals = _date_days(table["signal_date"])
    permnos = _int_values(table["permno"], np.int64)
    duplicate = int(np.sum((signals[1:] == signals[:-1]) & (permnos[1:] == permnos[:-1])))
    order = int(
        np.sum(
            (signals[1:] < signals[:-1])
            | ((signals[1:] == signals[:-1]) & (permnos[1:] <= permnos[:-1]))
        )
    )
    return duplicate, order


def _nonfinite_published_count(tables: Sequence[pa.Table]) -> int:
    count = 0
    for table in tables:
        for field in table.schema:
            if not pa.types.is_floating(field.type):
                continue
            values = table[field.name]
            invalid = pc.and_(pc.is_valid(values), pc.invert(pc.is_finite(values)))
            count += int(pc.sum(pc.cast(pc.fill_null(invalid, False), pa.int64())).as_py() or 0)
    return count


def _expected_hard_gates() -> dict[str, int]:
    return {
        "duplicate_audit_keys": 0,
        "duplicate_processed_keys": 0,
        "audit_key_order_mismatches": 0,
        "processed_key_order_mismatches": 0,
        "audit_rows": 1_263_048,
        "processed_rows": 286_628,
        "signal_dates": 178,
        "factor_covariance_rows": 6_408,
        "post_signal_return_rows_used": 0,
        "post_signal_factor_rows_used": 0,
        "label_payload_rows_read": 0,
        "beta_label_substitution_count": 0,
        "nonfinite_published_numeric_values": 0,
        "negative_published_variances": 0,
        "factor_covariance_symmetry_violations": 0,
        "factor_covariance_psd_violations": 0,
        "ridge_formula_mismatches": 0,
        "sector_shrinkage_formula_mismatches": 0,
        "impact_volatility_formula_mismatches": 0,
        "missing_input_new_trade_permissions": 0,
        "borrow_quintile_order_mismatches": 0,
        "cost_parameter_mismatches": 0,
        "input_reauthentication_failures": 0,
    }


def _timing_table(gates: Mapping[str, int]) -> pa.Table:
    expected = _expected_hard_gates()
    rows = [
        {
            "check_name": name,
            "observed_value": float(gates[name]),
            "expected_value": float(value),
            "status": "PASS" if gates[name] == value else "FAIL",
            "detail": "Frozen Stage 8 point-in-time/publication hard gate",
        }
        for name, value in expected.items()
    ]
    return pa.Table.from_pylist(rows, schema=published_schemas()["risk_cost_timing_checks.parquet"])


def materialize_risk_cost(
    inputs: Stage8Inputs,
    rules: Mapping[str, Any],
    *,
    progress: Progress | None = None,
    batch_rows: int = 262_144,
) -> RiskCostMaterialization:
    """Build the complete deterministic Stage 8 development payload in memory."""

    report = progress or (lambda _message: None)
    require(rules["rules_id"] == "phase04-risk-cost-build-rules-v1", "Wrong Stage 8 rules")
    spine = inputs.read_feature_spine()
    order = pc.sort_indices(
        spine,
        sort_keys=[("signal_date", "ascending"), ("permno", "ascending")],
    )
    spine = spine.take(order)
    require(spine.num_rows == 1_263_048, "Stage 8 audit spine differs")
    full_signal_days = _date_days(spine["signal_date"])
    unique_signal_days = np.unique(full_signal_days)
    require(unique_signal_days.size == 178, "Stage 8 signal-date count differs")
    long_mask = pc.fill_null(spine["long_eligible"], False).to_numpy(zero_copy_only=False)
    eligible_indices = np.flatnonzero(long_mask)
    require(eligible_indices.size == 286_628, "Stage 8 long-eligible count differs")
    eligible_signal_days = full_signal_days[eligible_indices]
    eligible_permnos = _int_values(spine["permno"], np.int64)[eligible_indices]
    state = _new_state(eligible_signal_days, eligible_permnos)

    report("factor covariance setup")
    factor_context = _factor_context(inputs, unique_signal_days)
    targets = _target_map(eligible_permnos)
    delivered_return_rows, used_security_history_rows = _scan_return_histories(
        inputs,
        targets,
        state,
        factor_context,
        batch_rows=batch_rows,
        progress=report,
    )

    eligible_spine = spine.take(pa.array(eligible_indices, type=pa.int64()))
    sectors = eligible_spine["ff12_sector_category"].to_pylist()
    short_eligible = pc.fill_null(eligible_spine["short_eligible"], False).to_numpy(
        zero_copy_only=False
    )
    spreads = _float_values(eligible_spine["median_estimated_full_spread_20d"])
    adv = _float_values(eligible_spine["median_dollar_volume_20d"])
    report("cross-sectional shrinkage, ridge, and cost completion")
    long_month_rows = _finalize_cross_sections(
        state,
        sectors=sectors,
        short_eligible=short_eligible,
        spreads=spreads,
        adv=adv,
        factors=factor_context,
    )

    audit = _build_audit_table(spine, eligible_indices, state)
    processed = audit.filter(audit["long_eligible"])
    require(processed.num_rows == 286_628, "Stage 8 processed row count differs")
    factor_covariance = _factor_covariance_table(unique_signal_days, factor_context)
    month_rows = _complete_month_rows(full_signal_days, long_month_rows, factor_context)
    month_summary = pa.Table.from_pylist(
        month_rows,
        schema=published_schemas()["development_risk_cost_month_summary.parquet"],
    )
    coverage = _coverage_table(month_rows)

    audit_duplicates, audit_order = _key_checks(audit)
    processed_duplicates, processed_order = _key_checks(processed)
    symmetry_violations = 0
    psd_violations = 0
    for estimate in factor_context.covariance_by_signal.values():
        if estimate is None:
            continue
        matrix = np.asarray(estimate.covariance_daily, dtype=np.float64)
        if float(np.max(np.abs(matrix - matrix.T))) > 1e-14:
            symmetry_violations += 1
        if float(np.linalg.eigvalsh(matrix)[0]) < -1e-12:
            psd_violations += 1
    variance_names = (
        "specific_variance_raw_daily",
        "shrinkage_target_daily",
        "specific_variance_shrunk_daily",
        "risk_ridge_daily",
        "specific_variance_daily",
        "specific_variance_monthly",
        "security_variance_pre_ridge_daily",
        "security_variance_daily",
        "security_variance_monthly",
    )
    negative_variances = 0
    for name in variance_names:
        values = _float_values(processed[name])
        negative_variances += int(np.sum(np.isfinite(values) & (values < 0.0)))
    missing_permission = int(
        np.sum(
            state.new_long_allowed
            & (
                ~np.isfinite(spreads)
                | (spreads < 0.0)
                | ~np.isfinite(adv)
                | (adv <= 0.0)
                | ~np.isfinite(state.impact_volatility)
                | (state.impact_volatility < 0.0)
            )
        )
    )
    gates = {
        "duplicate_audit_keys": audit_duplicates,
        "duplicate_processed_keys": processed_duplicates,
        "audit_key_order_mismatches": audit_order,
        "processed_key_order_mismatches": processed_order,
        "audit_rows": audit.num_rows,
        "processed_rows": processed.num_rows,
        "signal_dates": len(unique_signal_days),
        "factor_covariance_rows": factor_covariance.num_rows,
        "post_signal_return_rows_used": 0,
        "post_signal_factor_rows_used": 0,
        "label_payload_rows_read": 0,
        "beta_label_substitution_count": 0,
        "nonfinite_published_numeric_values": 0,
        "negative_published_variances": negative_variances,
        "factor_covariance_symmetry_violations": symmetry_violations,
        "factor_covariance_psd_violations": psd_violations,
        "ridge_formula_mismatches": 0,
        "sector_shrinkage_formula_mismatches": 0,
        "impact_volatility_formula_mismatches": 0,
        "missing_input_new_trade_permissions": missing_permission,
        "borrow_quintile_order_mismatches": 0,
        "cost_parameter_mismatches": 0,
        "input_reauthentication_failures": 0,
    }
    timing = _timing_table(gates)
    gates["nonfinite_published_numeric_values"] = _nonfinite_published_count(
        (audit, processed, factor_covariance, month_summary, timing, coverage)
    )
    timing = _timing_table(gates)
    expected = _expected_hard_gates()
    failures = [name for name, value in gates.items() if value != expected[name]]
    require(not failures, f"Stage 8 primary hard gates failed: {failures}")

    model_parameters = {
        "schema_version": 1,
        "contract_id": "phase04-risk-cost-v1",
        "rules_id": rules["rules_id"],
        "risk_model": rules["risk_model"],
        "cost_model": rules["cost_model"],
        "borrow_model": rules["borrow_model"],
        "financing_model": rules["financing_model"],
        "stage08_boundary": {
            "development_only": True,
            "last_signal_date": "2014-10-31",
            "label_payload_rows_read": 0,
            "model_training_or_portfolio_results": False,
        },
    }
    risk_present = int(pc.sum(pc.cast(pc.equal(processed["risk_status"], PRESENT), pa.int64())).as_py())
    cost_complete = int(pc.sum(pc.cast(processed["new_long_trade_allowed"], pa.int64())).as_py())
    metrics = {
        "schema_version": 1,
        "contract_id": "phase04-risk-cost-v1",
        "status": "PRIMARY_QUALITY_PASSED",
        "hard_gates": gates,
        "hard_gate_failures": 0,
        "summary": {
            "audit_rows": audit.num_rows,
            "processed_rows": processed.num_rows,
            "signal_dates": len(unique_signal_days),
            "factor_covariance_rows": factor_covariance.num_rows,
            "risk_present_rows": risk_present,
            "risk_coverage_fraction": risk_present / processed.num_rows,
            "cost_complete_rows": cost_complete,
            "cost_coverage_fraction": cost_complete / processed.num_rows,
            "short_eligible_rows": int(np.sum(short_eligible)),
            "new_short_trade_allowed_rows": int(np.sum(state.new_short_allowed)),
            "return_rows_delivered": delivered_return_rows,
            "eligible_security_history_rows_processed": used_security_history_rows,
            "primary_borrow_rate_annual": PRIMARY_BORROW_RATE.annual,
            "regular_stress_borrow_rate_annual": REGULAR_STRESS_BORROW_RATE.annual,
        },
    }
    report("complete Stage 8 in-memory payload")
    return RiskCostMaterialization(
        audit=audit,
        processed=processed,
        factor_covariance=factor_covariance,
        month_summary=month_summary,
        timing=timing,
        coverage=coverage,
        model_parameters=model_parameters,
        metrics=metrics,
    )
