"""Causal rank-to-payoff mappings and cost hurdles for V6.1 Step 03."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_bot.research_v6_1.portfolio import CalibrationResult, rank_strength


def _available_history(
    history: pd.DataFrame,
    *,
    score_column: str,
    outcome_column: str,
    month_column: str,
    availability_column: str,
    signal_date: pd.Timestamp,
    minimum_history_months: int,
) -> tuple[pd.DataFrame, int]:
    columns = {
        score_column,
        outcome_column,
        month_column,
        availability_column,
    }
    missing = columns - set(history.columns)
    if missing:
        raise ValueError(f"calibration history omits columns: {sorted(missing)}")
    frame = history.loc[
        pd.to_datetime(history[availability_column], errors="coerce").le(signal_date),
        list(columns),
    ].copy()
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    frame[outcome_column] = pd.to_numeric(frame[outcome_column], errors="coerce")
    frame = frame.loc[
        np.isfinite(frame[score_column].to_numpy(float))
        & np.isfinite(frame[outcome_column].to_numpy(float))
    ]
    months = int(frame[month_column].astype(str).nunique())
    if months < int(minimum_history_months):
        raise ValueError(
            f"calibration has {months} completed months, requires {minimum_history_months}"
        )
    return frame, months


def _equal_month_weights(frame: pd.DataFrame, month_column: str) -> np.ndarray:
    counts = frame.groupby(month_column, observed=True)[month_column].transform("size")
    return 1.0 / pd.to_numeric(counts, errors="raise").to_numpy(float)


def causal_shrunk_linear_calibration(
    history: pd.DataFrame,
    current_scores: np.ndarray,
    *,
    score_column: str,
    outcome_column: str,
    month_column: str,
    availability_column: str,
    signal_date: pd.Timestamp,
    minimum_history_months: int,
    shrinkage_prior_months: float,
) -> CalibrationResult:
    """Fit a causal nonnegative rank-payoff slope with equal month weights."""

    frame, months = _available_history(
        history,
        score_column=score_column,
        outcome_column=outcome_column,
        month_column=month_column,
        availability_column=availability_column,
        signal_date=signal_date,
        minimum_history_months=minimum_history_months,
    )
    x = frame[score_column].to_numpy(float) - 0.5
    y = frame[outcome_column].to_numpy(float)
    weight = _equal_month_weights(frame, month_column)
    weight /= weight.sum()
    x_mean = float(weight @ x)
    y_mean = float(weight @ y)
    centered_x = x - x_mean
    numerator = float(weight @ (centered_x * (y - y_mean)))
    denominator = float(weight @ np.square(centered_x))
    prior = max(float(shrinkage_prior_months), 0.0) / max(float(months), 1.0)
    slope = max(numerator / max(denominator * (1.0 + prior), 1e-12), 0.0)
    current = np.asarray(current_scores, dtype=np.float64)
    values = y_mean + slope * (current - x_mean - 0.5)
    return CalibrationResult(
        values=values,
        history_months=months,
        bin_centers=np.asarray([0.0, 1.0]),
        fitted_bin_values=np.asarray([y_mean - 0.5 * slope, y_mean + 0.5 * slope]),
    )


def causal_direct_rank_payoff(
    history: pd.DataFrame,
    current_scores: np.ndarray,
    *,
    score_column: str,
    outcome_column: str,
    month_column: str,
    availability_column: str,
    signal_date: pd.Timestamp,
    minimum_history_months: int,
    shrinkage_prior_months: float,
    threshold: float,
) -> CalibrationResult:
    """Scale tail-rank strength with one causal side-level payoff coefficient."""

    frame, months = _available_history(
        history,
        score_column=score_column,
        outcome_column=outcome_column,
        month_column=month_column,
        availability_column=availability_column,
        signal_date=signal_date,
        minimum_history_months=minimum_history_months,
    )
    strength = rank_strength(
        frame[score_column].to_numpy(float), "linear_tail", threshold=threshold
    )
    selected = strength > 0.0
    if not selected.any():
        raise ValueError("direct-rank calibration has no tail observations")
    tail = frame.loc[selected].copy()
    x = strength[selected]
    y = tail[outcome_column].to_numpy(float)
    weight = _equal_month_weights(tail, month_column)
    weight /= weight.sum()
    prior = max(float(shrinkage_prior_months), 0.0) / max(float(months), 1.0)
    scale = max(
        float(weight @ (x * y)) / max(float(weight @ np.square(x)) * (1.0 + prior), 1e-12),
        0.0,
    )
    values = scale * rank_strength(
        np.asarray(current_scores, dtype=np.float64),
        "linear_tail",
        threshold=threshold,
    )
    return CalibrationResult(
        values=values,
        history_months=months,
        bin_centers=np.asarray([1.0]),
        fitted_bin_values=np.asarray([scale]),
    )


def stressed_round_trip_hurdle(
    *,
    full_spread: np.ndarray,
    adv20_usd: np.ndarray,
    impact_volatility_daily: np.ndarray,
    stress_borrow_rate_annual: np.ndarray,
    nav_dollars: float,
    representative_position_weight: float,
    spread_multiplier: float,
    impact_coefficient: float,
    commission_each_way: float,
    include_borrow: bool,
) -> np.ndarray:
    """Return a conservative per-dollar entry/exit hurdle for one holding month."""

    spread = np.asarray(full_spread, dtype=np.float64)
    adv = np.asarray(adv20_usd, dtype=np.float64)
    volatility = np.asarray(impact_volatility_daily, dtype=np.float64)
    borrow = np.asarray(stress_borrow_rate_annual, dtype=np.float64)
    participation = representative_position_weight * float(nav_dollars) / adv
    per_way = (
        float(spread_multiplier) * spread / 2.0
        + float(commission_each_way)
        + float(impact_coefficient) * volatility * np.sqrt(participation)
    )
    result = 2.0 * per_way
    if include_borrow:
        result = result + borrow / 12.0
    return result


__all__ = [
    "causal_direct_rank_payoff",
    "causal_shrunk_linear_calibration",
    "stressed_round_trip_hurdle",
]
