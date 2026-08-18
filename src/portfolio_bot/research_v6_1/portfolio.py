"""Frozen V6.1 rank-to-portfolio primitives.

The selector is immutable here.  This module only calibrates its cross-sectional
ranks, forms approved candidate sets, constructs deterministic rank baselines,
and solves the cost-aware portfolio problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Mapping

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from portfolio_bot.allocation.costs import cvxpy_total_cost, numpy_total_cost


FACTOR_NAMES = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
NORMAL = NormalDist()


@dataclass(frozen=True)
class CalibrationResult:
    values: np.ndarray
    history_months: int
    bin_centers: np.ndarray
    fitted_bin_values: np.ndarray


@dataclass(frozen=True)
class PortfolioSolution:
    weights: np.ndarray
    status: str
    objective_value: float
    forecast_volatility: float
    long_gross: float
    short_gross: float
    gross: float
    net: float
    turnover: float
    factor_exposures: np.ndarray
    transaction_cost: float
    borrow_cost: float
    financing_cost: float
    total_cost: float


def percentile_rank(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return deterministic within-vector percentile ranks in (0, 1)."""

    series = pd.Series(np.asarray(values, dtype=np.float64))
    valid = np.isfinite(series.to_numpy(float))
    result = np.full(len(series), np.nan, dtype=np.float64)
    if valid.any():
        ranks = series.loc[valid].rank(method="average").to_numpy(float)
        result[valid] = (ranks - 0.5) / int(valid.sum())
    return result


def rank_strength(percentile: np.ndarray, method: str, *, threshold: float = 0.80) -> np.ndarray:
    """Map selected-tail percentiles to nonnegative deterministic conviction."""

    value = np.asarray(percentile, dtype=np.float64)
    linear = np.maximum((value - threshold) / (1.0 - threshold), 0.0)
    if method == "linear_tail":
        return linear
    if method == "squared_tail":
        return np.square(linear)
    if method == "gaussian_tail":
        clipped = np.clip(value, 0.005, 0.995)
        z = np.asarray([NORMAL.inv_cdf(float(item)) for item in clipped])
        cutoff = NORMAL.inv_cdf(threshold)
        return np.maximum(z - cutoff, 0.0)
    if method == "equal_weight":
        return (value >= threshold).astype(np.float64)
    raise ValueError(f"unknown rank-weighting method: {method}")


def causal_monotonic_calibration(
    history: pd.DataFrame,
    current_scores: np.ndarray,
    *,
    score_column: str,
    outcome_column: str,
    month_column: str,
    availability_column: str,
    signal_date: pd.Timestamp,
    bin_edges: Iterable[float],
    minimum_history_months: int,
    shrinkage_prior_months: float,
) -> CalibrationResult:
    """Map ranks to expected payoffs using only outcomes available at signal time.

    Every month receives equal weight inside each fixed score bin.  Bin means
    are shrunk toward zero and then projected onto a monotonic curve.  Fine
    prespecified tail bins preserve extreme-payoff information without using
    the current or future portfolio months.
    """

    required = {score_column, outcome_column, month_column, availability_column}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"calibration history omits columns: {sorted(missing)}")
    frame = history.loc[:, list(required)].copy()
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    frame[outcome_column] = pd.to_numeric(frame[outcome_column], errors="coerce")
    frame[availability_column] = pd.to_datetime(frame[availability_column], errors="coerce")
    frame = frame.loc[
        frame[availability_column].le(pd.Timestamp(signal_date))
        & np.isfinite(frame[score_column].to_numpy(float))
        & np.isfinite(frame[outcome_column].to_numpy(float))
    ].copy()
    months = sorted(frame[month_column].astype(str).unique().tolist())
    if len(months) < int(minimum_history_months):
        raise ValueError(
            f"calibration has {len(months)} completed months, requires {minimum_history_months}"
        )

    edges = np.asarray(tuple(float(value) for value in bin_edges), dtype=np.float64)
    if len(edges) < 3 or edges[0] != 0.0 or edges[-1] != 1.0 or np.any(np.diff(edges) <= 0):
        raise ValueError("calibration bin_edges must increase strictly from zero to one")
    frame["_bin"] = pd.cut(
        np.clip(frame[score_column].to_numpy(float), 0.0, 1.0),
        bins=edges,
        labels=False,
        include_lowest=True,
        right=True,
    )
    monthly = (
        frame.groupby([month_column, "_bin"], observed=True, sort=True)[outcome_column]
        .mean()
        .rename("month_bin_mean")
        .reset_index()
    )
    grouped = monthly.groupby("_bin", observed=True, sort=True)["month_bin_mean"]
    bin_mean = grouped.mean()
    bin_months = grouped.count()
    centers = 0.5 * (edges[:-1] + edges[1:])
    raw = np.full(len(centers), np.nan, dtype=np.float64)
    weights = np.zeros(len(centers), dtype=np.float64)
    for index in range(len(centers)):
        if index in bin_mean.index:
            count = float(bin_months.loc[index])
            shrinkage = count / (count + float(shrinkage_prior_months))
            raw[index] = float(bin_mean.loc[index]) * shrinkage
            weights[index] = count
    valid = np.isfinite(raw) & (weights > 0.0)
    if valid.sum() < 2:
        raise ValueError("calibration has fewer than two populated score bins")
    filled = np.interp(centers, centers[valid], raw[valid])
    filled_weights = np.where(weights > 0.0, weights, 1e-6)
    fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
        centers, filled, sample_weight=filled_weights
    )
    current = np.asarray(current_scores, dtype=np.float64)
    values = np.interp(np.clip(current, 0.0, 1.0), centers, fitted)
    values[~np.isfinite(current)] = np.nan
    return CalibrationResult(
        values=values,
        history_months=len(months),
        bin_centers=centers,
        fitted_bin_values=np.asarray(fitted, dtype=np.float64),
    )


def factor_covariance_matrix(frame: pd.DataFrame) -> np.ndarray:
    required = {"factor_row", "factor_col", "covariance_monthly"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"factor covariance omits columns: {sorted(missing)}")
    lookup = {
        (str(row.factor_row), str(row.factor_col)): float(row.covariance_monthly)
        for row in frame.itertuples(index=False)
    }
    matrix = np.empty((len(FACTOR_NAMES), len(FACTOR_NAMES)), dtype=np.float64)
    for i, left in enumerate(FACTOR_NAMES):
        for j, right in enumerate(FACTOR_NAMES):
            value = lookup.get((left, right), lookup.get((right, left), np.nan))
            matrix[i, j] = value
    if not np.isfinite(matrix).all():
        raise ValueError("factor covariance matrix is incomplete or nonfinite")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T


def portfolio_risk(
    weights: np.ndarray,
    factor_betas: np.ndarray,
    factor_covariance_root: np.ndarray,
    specific_variance_monthly: np.ndarray,
) -> tuple[float, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    exposures = np.asarray(factor_betas, dtype=np.float64).T @ weights
    factor_component = factor_covariance_root @ exposures
    specific = np.sqrt(np.maximum(specific_variance_monthly, 0.0)) * weights
    monthly_variance = float(factor_component @ factor_component + specific @ specific)
    return float(np.sqrt(max(12.0 * monthly_variance, 0.0))), exposures


def _waterfill_side(
    strength: np.ndarray,
    target_gross: float,
    capacity: np.ndarray,
) -> np.ndarray:
    strength = np.maximum(np.asarray(strength, dtype=np.float64), 0.0)
    capacity = np.maximum(np.asarray(capacity, dtype=np.float64), 0.0)
    result = np.zeros(len(strength), dtype=np.float64)
    active = (strength > 0.0) & (capacity > 0.0)
    remaining = min(float(target_gross), float(capacity[active].sum()))
    while remaining > 1e-12 and active.any():
        allocation = remaining * strength / float(strength[active].sum())
        addition = np.minimum(allocation, capacity - result)
        addition[~active] = 0.0
        achieved = float(addition.sum())
        if achieved <= 1e-14:
            break
        result += addition
        remaining -= achieved
        active = active & (result < capacity - 1e-12)
        strength = np.where(active, strength, 0.0)
    return result


def deterministic_rank_weights(
    *,
    long_mask: np.ndarray,
    short_mask: np.ndarray,
    long_rank: np.ndarray,
    short_rank: np.ndarray,
    method: str,
    factor_betas: np.ndarray,
    factor_covariance_root: np.ndarray,
    specific_variance_monthly: np.ndarray,
    adv20_usd: np.ndarray,
    nav_dollars: float,
    forecast_volatility_target: float,
    maximum_gross: float,
    maximum_position_weight: float,
    maximum_one_way_adv_participation: float,
) -> np.ndarray:
    """Build a dollar-neutral, volatility-scaled rank baseline."""

    long_strength = np.where(long_mask, rank_strength(long_rank, method), 0.0)
    short_strength = np.where(short_mask, rank_strength(short_rank, method), 0.0)
    valid_adv = np.isfinite(adv20_usd) & (adv20_usd > 0.0)
    capacity = np.where(
        valid_adv,
        np.minimum(
            maximum_position_weight,
            maximum_one_way_adv_participation * adv20_usd / float(nav_dollars),
        ),
        0.0,
    )
    raw_long = _waterfill_side(long_strength, 0.5, capacity)
    raw_short = _waterfill_side(short_strength, 0.5, capacity)
    raw = raw_long - raw_short
    volatility, _ = portfolio_risk(
        raw, factor_betas, factor_covariance_root, specific_variance_monthly
    )
    scale = min(
        float(maximum_gross) / max(float(np.abs(raw).sum()), 1e-12),
        float(forecast_volatility_target) / max(volatility, 1e-12),
    )
    target_long = min(0.5 * scale, float(capacity[long_mask].sum()))
    target_short = min(0.5 * scale, float(capacity[short_mask].sum()))
    return _waterfill_side(long_strength, target_long, capacity) - _waterfill_side(
        short_strength, target_short, capacity
    )


def apply_stateful_trade_limits(
    target_weights: np.ndarray,
    previous_weights: np.ndarray,
    *,
    adv20_usd: np.ndarray,
    nav_dollars: float,
    maximum_one_way_adv_participation: float,
    maximum_one_way_turnover: float,
) -> np.ndarray:
    """Move toward a deterministic target without violating capacity/turnover."""

    target = np.asarray(target_weights, dtype=np.float64)
    previous = np.asarray(previous_weights, dtype=np.float64)
    valid_adv = np.isfinite(adv20_usd) & (adv20_usd > 0.0)
    trade_cap = np.where(
        valid_adv,
        maximum_one_way_adv_participation * adv20_usd / float(nav_dollars),
        0.0,
    )
    delta = np.clip(target - previous, -trade_cap, trade_cap)
    turnover = 0.5 * float(np.abs(delta).sum())
    if turnover > maximum_one_way_turnover:
        delta *= float(maximum_one_way_turnover) / turnover
    return previous + delta


def solve_expected_net_return_portfolio(
    *,
    expected_stock_excess_return: np.ndarray,
    long_candidate: np.ndarray,
    short_candidate: np.ndarray,
    reduction_only: np.ndarray,
    previous_weights: np.ndarray,
    factor_betas: np.ndarray,
    factor_covariance_root: np.ndarray,
    specific_variance_monthly: np.ndarray,
    full_spread: np.ndarray,
    adv20_usd: np.ndarray,
    impact_volatility_daily: np.ndarray,
    borrow_rate_annual: np.ndarray,
    sic2_industry: np.ndarray,
    nav_dollars: float,
    forecast_volatility_target: float,
    maximum_gross: float,
    minimum_long_gross: float,
    minimum_short_gross: float,
    minimum_net: float,
    maximum_net: float,
    maximum_position_weight: float,
    maximum_one_way_turnover: float,
    maximum_one_way_adv_participation: float,
    market_beta_absolute_limit: float,
    other_factor_absolute_limit: float,
    sic2_industry_net_absolute_limit: float,
    sic2_industry_gross_limit: float,
    spread_multiplier: float,
    impact_coefficient: float,
    financing_rate_annual: float,
    hedge_short_candidate: np.ndarray | None = None,
    maximum_hedge_short_gross: float | None = None,
    risk_aversion: float = 0.0,
    concentration_aversion: float = 0.0,
    objective_cost_multiplier: float = 1.0,
    solver: str = "CLARABEL",
) -> PortfolioSolution:
    """Unwind stale holdings, then maximize net return under portfolio limits."""

    mu = np.asarray(expected_stock_excess_return, dtype=np.float64)
    previous = np.asarray(previous_weights, dtype=np.float64)
    long_candidate = np.asarray(long_candidate, dtype=bool)
    short_candidate = np.asarray(short_candidate, dtype=bool)
    reduction_only = np.asarray(reduction_only, dtype=bool)
    n = len(mu)
    if hedge_short_candidate is None:
        hedge_short_candidate = np.zeros(n, dtype=bool)
    hedge_short_candidate = np.asarray(hedge_short_candidate, dtype=bool)
    arrays = [previous, long_candidate, short_candidate, reduction_only, full_spread,
              adv20_usd, impact_volatility_daily, borrow_rate_annual,
              specific_variance_monthly, sic2_industry, hedge_short_candidate]
    if any(len(value) != n for value in arrays):
        raise ValueError("portfolio optimizer input lengths differ")
    if np.any(long_candidate & short_candidate):
        raise ValueError("a row cannot be a long and short candidate simultaneously")
    if np.any(hedge_short_candidate & ~short_candidate):
        raise ValueError("hedge shorts must be a subset of short candidates")
    if maximum_hedge_short_gross is not None and maximum_hedge_short_gross < 0.0:
        raise ValueError("maximum hedge-short gross must be nonnegative")
    if risk_aversion < 0.0 or concentration_aversion < 0.0:
        raise ValueError("optimizer regularization coefficients must be nonnegative")
    if objective_cost_multiplier < 1.0:
        raise ValueError("objective cost multiplier must be at least one")
    if not np.isfinite(mu).all() or not np.isfinite(previous).all():
        raise ValueError("expected returns and previous weights must be finite")

    variable = cp.Variable(n)
    long_index = np.flatnonzero(long_candidate)
    short_index = np.flatnonzero(short_candidate)
    constraints: list[cp.Constraint] = []
    unused = ~(long_candidate | short_candidate | reduction_only)
    if unused.any():
        constraints.append(variable[np.flatnonzero(unused)] == 0.0)
    if long_index.size:
        constraints.extend([variable[long_index] >= 0.0, variable[long_index] <= maximum_position_weight])
    if short_index.size:
        constraints.extend([variable[short_index] <= 0.0, variable[short_index] >= -maximum_position_weight])
    for index in np.flatnonzero(reduction_only):
        if previous[index] > 0.0:
            constraints.extend([variable[index] >= 0.0, variable[index] <= previous[index]])
        elif previous[index] < 0.0:
            constraints.extend([variable[index] <= 0.0, variable[index] >= previous[index]])
        else:
            constraints.append(variable[index] == 0.0)

    delta = variable - previous
    valid_adv = np.isfinite(adv20_usd) & (adv20_usd > 0.0)
    trade_cap = np.where(
        valid_adv,
        maximum_one_way_adv_participation * adv20_usd / float(nav_dollars),
        0.0,
    )
    constraints.extend(
        [
            cp.abs(delta) <= trade_cap,
            0.5 * cp.norm1(delta) <= maximum_one_way_turnover,
        ]
    )
    hedge_short_index = np.flatnonzero(hedge_short_candidate)
    if hedge_short_index.size and maximum_hedge_short_gross is not None:
        constraints.append(
            -cp.sum(variable[hedge_short_index]) <= float(maximum_hedge_short_gross)
        )
    # Every active row has a frozen side, so side gross is linear.  Expressing
    # a minimum through ``sum(pos(w))`` would be non-convex even though the
    # economically equivalent side-specific sum is affine here.
    long_side = long_candidate | (reduction_only & (previous > 0.0))
    short_side = short_candidate | (reduction_only & (previous < 0.0))
    long_side_index = np.flatnonzero(long_side)
    short_side_index = np.flatnonzero(short_side)
    long_gross = cp.sum(variable[long_side_index]) if long_side_index.size else cp.Constant(0.0)
    short_gross = -cp.sum(variable[short_side_index]) if short_side_index.size else cp.Constant(0.0)
    gross = long_gross + short_gross
    net = cp.sum(variable)
    constraints.extend(
        [
            gross <= maximum_gross,
            long_gross >= minimum_long_gross,
            short_gross >= minimum_short_gross,
            net >= minimum_net,
            net <= maximum_net,
        ]
    )

    betas = np.asarray(factor_betas, dtype=np.float64)
    exposures = betas.T @ variable
    constraints.extend(
        [
            cp.abs(exposures[0]) <= market_beta_absolute_limit,
            cp.abs(exposures[1:]) <= other_factor_absolute_limit,
        ]
    )
    factor_component = factor_covariance_root @ exposures
    specific_component = cp.multiply(
        np.sqrt(np.maximum(specific_variance_monthly, 0.0)), variable
    )
    annual_variance = 12.0 * (
        cp.sum_squares(factor_component) + cp.sum_squares(specific_component)
    )
    constraints.append(annual_variance <= float(forecast_volatility_target) ** 2)

    industries = np.asarray(sic2_industry, dtype=str)
    for industry in sorted(set(industries.tolist())):
        index = np.flatnonzero(industries == industry)
        if index.size:
            constraints.extend(
                [
                    cp.abs(cp.sum(variable[index])) <= sic2_industry_net_absolute_limit,
                    cp.sum(cp.abs(variable[index])) <= sic2_industry_gross_limit,
                ]
            )

    cost = cvxpy_total_cost(
        variable,
        previous,
        full_spread,
        adv20_usd,
        impact_volatility_daily,
        borrow_rate_annual,
        nav_dollars,
        impact_coefficient=impact_coefficient,
        spread_multiplier=spread_multiplier,
        financing_rate_annual=financing_rate_annual,
    )
    constraints.extend(cost.constraints)
    settings: Mapping[str, float | int | bool] = {
        "max_iter": 1000,
        "tol_gap_abs": 1e-8,
        "tol_gap_rel": 1e-8,
        "tol_feas": 1e-8,
        "equilibrate_enable": True,
    }
    stale_long_index = np.flatnonzero(reduction_only & (previous > 0.0))
    stale_short_index = np.flatnonzero(reduction_only & (previous < 0.0))
    if stale_long_index.size or stale_short_index.size:
        stale_gross = cp.Constant(0.0)
        if stale_long_index.size:
            stale_gross += cp.sum(variable[stale_long_index])
        if stale_short_index.size:
            stale_gross -= cp.sum(variable[stale_short_index])
        repair_problem = cp.Problem(cp.Minimize(stale_gross), constraints)
        repair_problem.solve(solver=solver, **settings)
        if (
            repair_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
            or variable.value is None
        ):
            raise RuntimeError(
                f"stale-position repair failed with status {repair_problem.status}"
            )
        repaired = np.asarray(variable.value, dtype=np.float64).reshape(-1)
        minimum_stale_gross = float(
            repaired[stale_long_index].sum()
            - repaired[stale_short_index].sum()
        )
        # A $100 weight tolerance at the stated $100m NAV is economically
        # immaterial and avoids making the second solve infeasible on solver dust.
        constraints.append(stale_gross <= max(minimum_stale_gross, 0.0) + 1e-6)

    # Preserve the frozen optimizer's exact expression graph when every new
    # coefficient is at its default.  Adding algebraically zero CVXPY terms can
    # change conic canonicalization and select a different corner of a nearly
    # degenerate feasible set, which then compounds through stateful turnover.
    if (
        float(objective_cost_multiplier) == 1.0
        and float(risk_aversion) == 0.0
        and float(concentration_aversion) == 0.0
    ):
        utility = mu @ variable - cost.total
    else:
        utility = mu @ variable - float(objective_cost_multiplier) * cost.total
        if float(risk_aversion) != 0.0:
            utility -= 0.5 * float(risk_aversion) * annual_variance
        if float(concentration_aversion) != 0.0:
            utility -= 0.5 * float(concentration_aversion) * cp.sum_squares(variable)
    objective = cp.Maximize(10_000.0 * utility)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, **settings)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or variable.value is None:
        raise RuntimeError(f"portfolio optimizer failed with status {problem.status}")
    weights = np.asarray(variable.value, dtype=np.float64).reshape(-1)
    weights[np.abs(weights) < 1e-10] = 0.0
    risk, numeric_exposures = portfolio_risk(
        weights, betas, factor_covariance_root, specific_variance_monthly
    )
    breakdown = numpy_total_cost(
        weights,
        previous,
        full_spread,
        adv20_usd,
        impact_volatility_daily,
        borrow_rate_annual,
        nav_dollars,
        impact_coefficient=impact_coefficient,
        spread_multiplier=spread_multiplier,
        financing_rate_annual=financing_rate_annual,
    )
    long_value = float(np.maximum(weights, 0.0).sum())
    short_value = float(np.maximum(-weights, 0.0).sum())
    return PortfolioSolution(
        weights=weights,
        status=str(problem.status),
        objective_value=float(mu @ weights - breakdown.total),
        forecast_volatility=risk,
        long_gross=long_value,
        short_gross=short_value,
        gross=long_value + short_value,
        net=float(weights.sum()),
        turnover=0.5 * float(np.abs(weights - previous).sum()),
        factor_exposures=numeric_exposures,
        transaction_cost=breakdown.transaction,
        borrow_cost=breakdown.borrow,
        financing_cost=breakdown.financing,
        total_cost=breakdown.total,
    )


__all__ = [
    "FACTOR_NAMES",
    "CalibrationResult",
    "PortfolioSolution",
    "apply_stateful_trade_limits",
    "causal_monotonic_calibration",
    "deterministic_rank_weights",
    "factor_covariance_matrix",
    "percentile_rank",
    "portfolio_risk",
    "rank_strength",
    "solve_expected_net_return_portfolio",
]
